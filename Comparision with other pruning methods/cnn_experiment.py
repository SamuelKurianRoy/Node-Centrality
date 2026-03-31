"""
cnn_experiment.py — Spectral Pruning on CNN / CIFAR-10  (v2)
=============================================================
FIXES vs v1:
  - compute_aei_scores and compute_aei_scores_timed both defined;
    HybridConvPruner now calls compute_aei_scores (no more NameError)
  - Results printed IMMEDIATELY after every single run — crashes can't
    erase data already collected
  - Each run wrapped in try/except so one failure skips and continues
  - Baseline FLOPs + latency printed right after training

Output order per run:
  Baseline → [20% spectral] → [20% hybrid] → ... → [40% random] → SUMMARY

Usage (Colab):
    !python cnn_experiment.py | tee /content/drive/MyDrive/SpectralPruning_Results/cnn_cifar10.txt
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import copy
import time
import traceback
from scipy.linalg import eigh


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Model
# ─────────────────────────────────────────────────────────────────────────────

class SimpleCNN(nn.Module):
    """
    2-conv + 1-FC MLP head.
    CIFAR-10 input: 3×32×32
      after conv1 + pool  →  num_filters1 × 16 × 16
      after conv2 + pool  →  num_filters2 × 8  × 8
      fc1                 →  num_fc
      fc2                 →  10
    """
    def __init__(self, num_filters1=32, num_filters2=64, num_fc=256, num_classes=10):
        super().__init__()
        self.num_filters1 = num_filters1
        self.num_filters2 = num_filters2
        self.num_fc       = num_fc

        self.conv1 = nn.Conv2d(3,             num_filters1, 3, padding=1)
        self.conv2 = nn.Conv2d(num_filters1,  num_filters2, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(num_filters2 * 8 * 8, num_fc)
        self.fc2   = nn.Linear(num_fc, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # B×f1×16×16
        x = self.pool(F.relu(self.conv2(x)))   # B×f2×8×8
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Dataset
# ─────────────────────────────────────────────────────────────────────────────

def load_cifar10(batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    train_set = torchvision.datasets.CIFAR10('./data', train=True,
                                             download=True,
                                             transform=transform_train)
    test_set  = torchvision.datasets.CIFAR10('./data', train=False,
                                             download=True,
                                             transform=transform_test)
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Train / Evaluate
# ─────────────────────────────────────────────────────────────────────────────

def train_model(model, train_loader, device, epochs=20, lr=1e-3):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            print(f"    epoch {epoch+1:>2}/{epochs} — loss: "
                  f"{total_loss/len(train_loader):.4f}", flush=True)
    return model


def finetune_model(model, train_loader, device, epochs=5, lr=1e-4):
    return train_model(model, train_loader, device, epochs=epochs, lr=lr)


def evaluate(model, test_loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            correct += (model(inputs).argmax(1) == targets).sum().item()
            total   += targets.size(0)
    return 100.0 * correct / total


# ─────────────────────────────────────────────────────────────────────────────
# 4.  FLOPs and latency helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_flops(f1, f2, fc):
    """
    MACs × 2 = FLOPs for each layer of SimpleCNN on 32×32 CIFAR-10.
      conv1: 3 input channels, f1 output, 3×3 kernel, 32×32 output (before pool)
      conv2: f1 input, f2 output, 3×3, 16×16 output (before pool)
      fc1  : (f2 × 8 × 8) → fc
      fc2  : fc → 10
    """
    flops  = 2 * 3  * f1 * 9 * 32 * 32   # conv1
    flops += 2 * f1 * f2 * 9 * 16 * 16   # conv2
    flops += 2 * (f2 * 64) * fc           # fc1  (8×8=64 spatial cells)
    flops += 2 * fc * 10                  # fc2
    return flops


def measure_latency(model, device, n_runs=500):
    """
    Single-image inference latency in ms, GPU-synchronised.
    50-run warmup before timing.
    """
    model.eval()
    dummy = torch.randn(1, 3, 32, 32).to(device)
    with torch.no_grad():
        for _ in range(50):
            model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_runs * 1000   # ms


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Activation collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_filter_activations(model, data_loader, device,
                                layer_name='conv2', n_batches=5):
    """
    Spatially-averaged activations for every filter → [N_samples, N_filters].
    Spatial global-average pooling preserves co-activation signal without
    requiring a fixed spatial layout — analogous to scalar MLP activations.
    """
    model.eval()
    acts = []

    def hook(module, inp, out):
        acts.append(out.mean(dim=[2, 3]).detach().cpu().numpy())

    handle = getattr(model, layer_name).register_forward_hook(hook)
    with torch.no_grad():
        for i, (inputs, _) in enumerate(data_loader):
            if i >= n_batches:
                break
            model(inputs.to(device))
    handle.remove()
    return np.concatenate(acts, axis=0)   # [N, C]


# ─────────────────────────────────────────────────────────────────────────────
# 6.  AEI score computation — two variants
#     compute_aei_scores       : returns R only   (used by HybridConvPruner)
#     compute_aei_scores_timed : returns (R, timing_dict) (used by Spectral)
# ─────────────────────────────────────────────────────────────────────────────

def _build_graph_and_fiedler(activations, threshold=0.3):
    """Shared core: Pearson adj → normalised Laplacian → Fiedler vector."""
    n    = activations.shape[1]
    corr = np.corrcoef(activations.T)
    np.fill_diagonal(corr, 0.0)
    adj  = (np.abs(corr) > threshold).astype(float) * np.abs(corr)

    deg        = adj.sum(axis=1)
    deg_safe   = np.where(deg > 0, deg, 1.0)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(deg_safe))
    L_norm     = np.eye(n) - D_inv_sqrt @ adj @ D_inv_sqrt

    _, vecs = eigh(L_norm)
    v2      = vecs[:, 1]
    return adj, v2


def _aei_r_scores(adj, v2):
    """R_i = Σ_{j∈N(i)} |v2_i − v2_j|."""
    n = len(v2)
    R = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if adj[i, j] > 0:
                R[i] += abs(v2[i] - v2[j])
    return R


def compute_aei_scores(activations, threshold=0.3):
    """Returns R scores only.  Used by HybridConvPruner."""
    adj, v2 = _build_graph_and_fiedler(activations, threshold)
    return _aei_r_scores(adj, v2)


def compute_aei_scores_timed(activations, threshold=0.3):
    """
    Returns (R, timing_dict) with per-step ms timings.
    timing_dict keys:
      graph_construction_ms, eigendecomposition_ms, score_computation_ms
    Note: activation_collection_ms is timed separately in SpectralConvPruner.
    """
    n = activations.shape[1]

    # Graph construction (correlation + threshold)
    t0   = time.perf_counter()
    corr = np.corrcoef(activations.T)
    np.fill_diagonal(corr, 0.0)
    adj  = (np.abs(corr) > threshold).astype(float) * np.abs(corr)
    graph_ms = (time.perf_counter() - t0) * 1000

    # Eigendecomposition → Fiedler vector
    t0 = time.perf_counter()
    deg        = adj.sum(axis=1)
    deg_safe   = np.where(deg > 0, deg, 1.0)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(deg_safe))
    L_norm     = np.eye(n) - D_inv_sqrt @ adj @ D_inv_sqrt
    _, vecs    = eigh(L_norm)
    v2         = vecs[:, 1]
    eig_ms     = (time.perf_counter() - t0) * 1000

    # R scores
    t0 = time.perf_counter()
    R  = _aei_r_scores(adj, v2)
    score_ms = (time.perf_counter() - t0) * 1000

    timing = {
        'graph_construction_ms':  graph_ms,
        'eigendecomposition_ms':  eig_ms,
        'score_computation_ms':   score_ms,
    }
    return R, timing


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Pruned model construction
# ─────────────────────────────────────────────────────────────────────────────

def create_pruned_cnn(original, keep_indices, layer_name='conv2'):
    """
    Physically remove filters and return a smaller SimpleCNN.
    Supports pruning conv1 or conv2.
    """
    m    = original
    keep = sorted(keep_indices)
    nk   = len(keep)

    if layer_name == 'conv2':
        new = SimpleCNN(num_filters1=m.num_filters1,
                        num_filters2=nk,
                        num_fc=m.num_fc)
        new.conv1.weight.data = m.conv1.weight.data.clone()
        new.conv1.bias.data   = m.conv1.bias.data.clone()
        new.conv2.weight.data = m.conv2.weight.data[keep].clone()
        new.conv2.bias.data   = m.conv2.bias.data[keep].clone()
        spatial = 8 * 8
        cols = []
        for k in keep:
            cols.extend(range(k * spatial, (k + 1) * spatial))
        new.fc1.weight.data = m.fc1.weight.data[:, cols].clone()
        new.fc1.bias.data   = m.fc1.bias.data.clone()
        new.fc2.weight.data = m.fc2.weight.data.clone()
        new.fc2.bias.data   = m.fc2.bias.data.clone()

    elif layer_name == 'conv1':
        new = SimpleCNN(num_filters1=nk,
                        num_filters2=m.num_filters2,
                        num_fc=m.num_fc)
        new.conv1.weight.data = m.conv1.weight.data[keep].clone()
        new.conv1.bias.data   = m.conv1.bias.data[keep].clone()
        new.conv2.weight.data = m.conv2.weight.data[:, keep, :, :].clone()
        new.conv2.bias.data   = m.conv2.bias.data.clone()
        new.fc1.weight.data   = m.fc1.weight.data.clone()
        new.fc1.bias.data     = m.fc1.bias.data.clone()
        new.fc2.weight.data   = m.fc2.weight.data.clone()
        new.fc2.bias.data     = m.fc2.bias.data.clone()

    else:
        raise ValueError(f"Unknown layer_name: {layer_name}")

    return new


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Pruning classes
# ─────────────────────────────────────────────────────────────────────────────

class BaseConvPruner:
    def __init__(self, model, device, train_loader):
        self.model        = model
        self.device       = device
        self.train_loader = train_loader
        self.last_aei_timing = None   # populated by spectral/hybrid if applicable

    def get_scores(self, layer_name):
        raise NotImplementedError

    def prune(self, layer_name, prune_ratio):
        scores    = self.get_scores(layer_name)
        n_filters = len(scores)
        n_prune   = int(n_filters * prune_ratio)
        keep      = sorted(np.argsort(scores)[n_prune:].tolist())
        return create_pruned_cnn(self.model, keep, layer_name)


# ── AEI Spectral ──────────────────────────────────────────────────────────────
class SpectralConvPruner(BaseConvPruner):
    """
    AEI for conv filters.
    Activations → Pearson co-activation graph → Fiedler vector → R scores.
    Low R_i (peripheral) → prune.
    Stores full per-step timing in self.last_aei_timing.
    """
    def get_scores(self, layer_name):
        t0   = time.perf_counter()
        acts = collect_filter_activations(self.model, self.train_loader,
                                          self.device, layer_name)
        act_ms = (time.perf_counter() - t0) * 1000

        R, timing = compute_aei_scores_timed(acts)
        timing['activation_collection_ms'] = act_ms
        self.last_aei_timing = timing
        return R


# ── L1 Norm ───────────────────────────────────────────────────────────────────
class L1ConvPruner(BaseConvPruner):
    def get_scores(self, layer_name):
        w = getattr(self.model, layer_name).weight.data
        return w.abs().sum(dim=[1, 2, 3]).cpu().numpy()


# ── L2 Norm ───────────────────────────────────────────────────────────────────
class L2ConvPruner(BaseConvPruner):
    def get_scores(self, layer_name):
        w = getattr(self.model, layer_name).weight.data
        return w.pow(2).sum(dim=[1, 2, 3]).sqrt().cpu().numpy()


# ── SNIP ──────────────────────────────────────────────────────────────────────
class SNIPConvPruner(BaseConvPruner):
    def get_scores(self, layer_name):
        mc = copy.deepcopy(self.model).to(self.device)
        mc.train()
        inputs, targets = next(iter(self.train_loader))
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        nn.CrossEntropyLoss()(mc(inputs), targets).backward()
        layer  = getattr(mc, layer_name)
        scores = (layer.weight.grad.abs() * layer.weight.data.abs()
                  ).sum(dim=[1, 2, 3])
        return scores.detach().cpu().numpy()


# ── GraSP ─────────────────────────────────────────────────────────────────────
class GraSPConvPruner(BaseConvPruner):
    def get_scores(self, layer_name):
        mc = copy.deepcopy(self.model).to(self.device)
        mc.train()
        inputs, targets = next(iter(self.train_loader))
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        params = [p for p in mc.parameters() if p.requires_grad]
        loss   = nn.CrossEntropyLoss()(mc(inputs), targets)
        grads  = torch.autograd.grad(loss, params,
                                     create_graph=True, retain_graph=True)
        gnorm  = sum((g * g).sum() for g in grads)
        Hg     = torch.autograd.grad(gnorm, params)
        target_layer = getattr(mc, layer_name)
        for p, hg in zip(params, Hg):
            if p is target_layer.weight:
                scores = -(hg * p.data).sum(dim=[1, 2, 3])
                return scores.detach().cpu().numpy()
        raise ValueError(f"Layer {layer_name} not found in parameters.")


# ── Hybrid (AEI × L2) — FIXED: calls compute_aei_scores, not _timed ──────────
class HybridConvPruner(BaseConvPruner):
    """
    score_i = minmax(R_i) × minmax(L2_i)
    Parameter-free: prunes filters peripheral in BOTH graph structure AND weight
    magnitude.  Directly extends the MLP hybrid result from FashionMNIST.
    Uses compute_aei_scores (untimed) — no NameError.
    """
    def get_scores(self, layer_name):
        # AEI — uses compute_aei_scores (always defined)
        acts  = collect_filter_activations(self.model, self.train_loader,
                                           self.device, layer_name)
        R     = compute_aei_scores(acts)                      # ← FIX
        R_n   = (R - R.min()) / (R.max() - R.min() + 1e-8)

        # L2 magnitude
        w     = getattr(self.model, layer_name).weight.data
        L2    = w.pow(2).sum(dim=[1, 2, 3]).sqrt().cpu().numpy()
        L2_n  = (L2 - L2.min()) / (L2.max() - L2.min() + 1e-8)

        return R_n * L2_n


# ── Random ────────────────────────────────────────────────────────────────────
class RandomConvPruner(BaseConvPruner):
    def __init__(self, model, device, train_loader, seed=42):
        super().__init__(model, device, train_loader)
        self.seed = seed

    def get_scores(self, layer_name):
        rng = np.random.default_rng(self.seed)
        n   = getattr(self.model, layer_name).weight.shape[0]
        return rng.random(n)


PRUNER_MAP = {
    'spectral': SpectralConvPruner,
    'hybrid':   HybridConvPruner,
    'l1_norm':  L1ConvPruner,
    'l2_norm':  L2ConvPruner,
    'snip':     SNIPConvPruner,
    'grasp':    GraSPConvPruner,
    'random':   RandomConvPruner,
}


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Experiment runner — prints ALL metrics immediately after each run
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(base_model, method, layer_name, prune_ratio,
                   train_loader, test_loader, device,
                   base_acc, base_flops, base_latency):
    """
    Prune → post-prune eval → fine-tune → eval → measure latency/FLOPs.
    Prints a full result block immediately — safe against mid-run crashes.
    Returns a dict of results (also stored in caller's results dict).
    """
    n_orig = getattr(base_model, layer_name).weight.shape[0]
    n_keep = int(n_orig * (1 - prune_ratio))
    label  = f"[{method:10s}] {int(prune_ratio*100)}%  ({n_orig}→{n_keep} filters)"

    model_copy = copy.deepcopy(base_model)
    pruner     = PRUNER_MAP[method](model_copy, device, train_loader)

    # ── Score + prune (timed) ────────────────────────────────────────────────
    t0         = time.perf_counter()
    pruned     = pruner.prune(layer_name, prune_ratio)
    score_ms   = (time.perf_counter() - t0) * 1000

    # ── Post-prune accuracy ───────────────────────────────────────────────────
    post_acc   = evaluate(pruned, test_loader, device)

    # ── Fine-tune ─────────────────────────────────────────────────────────────
    print(f"\n  {label}  — fine-tuning …", flush=True)
    pruned     = finetune_model(pruned, train_loader, device, epochs=5, lr=1e-4)
    fine_acc   = evaluate(pruned, test_loader, device)

    # ── FLOPs ─────────────────────────────────────────────────────────────────
    pruned_flops     = compute_flops(pruned.num_filters1, pruned.num_filters2, pruned.num_fc)
    flops_pct        = 100.0 * (1 - pruned_flops / base_flops)

    # ── Inference latency ─────────────────────────────────────────────────────
    pruned_latency   = measure_latency(pruned, device)

    # ── Print full result block immediately ───────────────────────────────────
    sep = "  " + "─" * 60
    print(sep)
    print(f"  RESULT  {label}")
    print(f"    post-prune  accuracy : {post_acc:.2f}%")
    print(f"    fine-tuned  accuracy : {fine_acc:.2f}%"
          f"   (Δ {fine_acc - base_acc:+.2f}pp vs baseline)")
    print(f"    FLOPs                : {base_flops/1e6:.2f}M → {pruned_flops/1e6:.2f}M"
          f"  ({flops_pct:.1f}% reduction)")
    print(f"    inference latency    : {base_latency:.3f}ms → {pruned_latency:.3f}ms")
    print(f"    score+select time    : {score_ms:.1f}ms  [one-time offline cost]")

    if pruner.last_aei_timing:
        t = pruner.last_aei_timing
        print(f"    AEI step breakdown   : "
              f"act_collect={t.get('activation_collection_ms', 0):.1f}ms  "
              f"graph={t['graph_construction_ms']:.1f}ms  "
              f"eig={t['eigendecomposition_ms']:.1f}ms  "
              f"score={t['score_computation_ms']:.1f}ms")

    print(sep, flush=True)

    return {
        'fine_acc':       fine_acc,
        'post_acc':       post_acc,
        'pruned_flops':   pruned_flops,
        'pruned_latency': pruned_latency,
        'score_ms':       score_ms,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 10. Summary table printer
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results, methods, prune_ratios, base_acc, base_flops, base_latency):
    W = 13
    ratios_str = [f"{int(r*100)}%" for r in prune_ratios]

    def header_row(title):
        print(f"\n{'='*65}")
        print(title)
        print(f"{'='*65}")
        print(f"{'Method':<{W}}" + "".join(f"  {s:>8}" for s in ratios_str))
        print("-" * (W + 12 * len(prune_ratios)))

    def baseline_row(label, base_val, fmt):
        print(f"{'  Baseline':<{W}}" + "".join(f"  {fmt.format(base_val):>8}"
                                                 for _ in prune_ratios))

    # ── Table 1: Accuracy ────────────────────────────────────────────────────
    header_row("TABLE 1 — Fine-tuned accuracy (%)")
    for m in methods:
        vals = []
        for r in prune_ratios:
            res = results[m].get(r)
            vals.append(f"{res['fine_acc']:.2f}%" if res else "  FAIL")
        print(f"{m:<{W}}" + "".join(f"  {v:>8}" for v in vals))
    baseline_row("", base_acc, "{:.2f}%")

    # ── Table 2: Accuracy drop ───────────────────────────────────────────────
    header_row("TABLE 2 — Accuracy drop from baseline (pp, lower=better)")
    for m in methods:
        vals = []
        for r in prune_ratios:
            res = results[m].get(r)
            vals.append(f"{base_acc - res['fine_acc']:+.2f}pp" if res else "  FAIL")
        print(f"{m:<{W}}" + "".join(f"  {v:>8}" for v in vals))

    # ── Table 3: FLOPs ───────────────────────────────────────────────────────
    header_row(f"TABLE 3 — FLOPs after pruning (M)  [baseline={base_flops/1e6:.2f}M]")
    for m in methods:
        vals = []
        for r in prune_ratios:
            res = results[m].get(r)
            if res:
                pct = 100 * (1 - res['pruned_flops'] / base_flops)
                vals.append(f"{res['pruned_flops']/1e6:.2f}M")
            else:
                vals.append("  FAIL")
        print(f"{m:<{W}}" + "".join(f"  {v:>8}" for v in vals))
    # FLOPs reduction row (architecture-determined, same per method at same ratio)
    print(f"\n  FLOPs reduction vs baseline (architecture-determined, same across methods):")
    pct_row = f"  {'':>{W}}"
    for r in prune_ratios:
        # use first successful result at this ratio
        for m in methods:
            res = results[m].get(r)
            if res:
                pct = 100 * (1 - res['pruned_flops'] / base_flops)
                pct_row += f"  {pct:.1f}%  "
                break
    print(pct_row)

    # ── Table 4: Inference latency ───────────────────────────────────────────
    header_row(f"TABLE 4 — Inference latency (ms, single image)  [baseline={base_latency:.3f}ms]")
    for m in methods:
        vals = []
        for r in prune_ratios:
            res = results[m].get(r)
            vals.append(f"{res['pruned_latency']:.3f}ms" if res else "  FAIL")
        print(f"{m:<{W}}" + "".join(f"  {v:>8}" for v in vals))

    # ── Table 5: Pruning overhead ────────────────────────────────────────────
    header_row("TABLE 5 — Score+select overhead (ms, one-time offline cost, at 30%)")
    target_ratio = 0.3
    for m in methods:
        res = results[m].get(target_ratio)
        val = f"{res['score_ms']:.1f}ms" if res else "FAIL"
        print(f"  {m:<{W-2}}{val}")

    print(f"\n{'='*65}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# 11. Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("=" * 65)

    train_loader, test_loader = load_cifar10(batch_size=128)

    # ── Train base model ──────────────────────────────────────────────────────
    print("\nTraining baseline SimpleCNN on CIFAR-10 (20 epochs)…")
    base_model = SimpleCNN(num_filters1=32, num_filters2=64, num_fc=256)
    base_model = train_model(base_model, train_loader, device,
                             epochs=20, lr=1e-3)

    base_acc     = evaluate(base_model, test_loader, device)
    base_flops   = compute_flops(32, 64, 256)
    base_latency = measure_latency(base_model, device)
    n_params     = sum(p.numel() for p in base_model.parameters())

    # ── Print baseline metrics immediately ───────────────────────────────────
    print(f"\n{'='*65}")
    print("BASELINE — SimpleCNN / CIFAR-10")
    print(f"{'='*65}")
    print(f"  Architecture : Conv1(3→32) → Conv2(32→64) → FC1(4096→256) → FC2(256→10)")
    print(f"  Parameters   : {n_params:,}")
    print(f"  Test accuracy: {base_acc:.2f}%")
    print(f"  FLOPs        : {base_flops/1e6:.2f}M")
    print(f"  Latency      : {base_latency:.3f}ms  (single image, {500} runs, GPU-synced)")
    print(f"{'='*65}", flush=True)

    # ── Pruning experiments ───────────────────────────────────────────────────
    methods      = ['spectral', 'hybrid', 'l1_norm', 'l2_norm',
                    'snip', 'grasp', 'random']
    layer_name   = 'conv2'
    prune_ratios = [0.2, 0.3, 0.4]

    results = {m: {} for m in methods}

    for ratio in prune_ratios:
        n_keep = int(64 * (1 - ratio))
        print(f"\n{'='*65}")
        print(f"PRUNING {int(ratio*100)}%  —  conv2: 64 → {n_keep} filters  +  5-epoch fine-tune")
        print(f"{'='*65}", flush=True)

        for method in methods:
            try:
                res = run_experiment(
                    base_model, method, layer_name, ratio,
                    train_loader, test_loader, device,
                    base_acc, base_flops, base_latency,
                )
                results[method][ratio] = res
            except Exception as e:
                print(f"\n  [{method}] {int(ratio*100)}%  *** FAILED: {e} ***")
                traceback.print_exc()
                results[method][ratio] = None
                print("  Continuing with next method…", flush=True)

    # ── Summary tables ────────────────────────────────────────────────────────
    print_summary(results, methods, prune_ratios, base_acc, base_flops, base_latency)


if __name__ == '__main__':
    main()