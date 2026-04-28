"""
ResNet-20 CIFAR-10 Pruning Experiment — AEI  [DGX-OPTIMISED v2]
================================================================
Saves everything the CIFAR-100 notebook saves, plus more:

  resnet20_outputs/
    checkpoints/
      baseline_resnet20.pt          weights + acc + flops + metadata
    aei_scores/
      aei_scores.pt                 per-layer R arrays
      calibration_activations.pt    raw GAP activations [N_calib, C_out]
    pruned_models/
      L2_sp40_seed42.pt             state_dict + method/sparsity/seed/
      ...                           acc_before_ft/acc/avg_r_pct/flops/overhead
      best/
        L2_sp40_best.pt             highest-accuracy seed per (method, sparsity)
    keep_indices/
      keep_indices.pt               {method: {sparsity: tensor of kept indices}}
    figures/
      spectral_histograms_30pct.pdf spectral R-value histogram (key paper figure)
      accuracy_comparison.pdf       bar chart: all methods x sparsity
      overhead_breakdown.pdf        overhead breakdown per method
    resnet20_results.json           all numbers incl. flops, overhead breakdown,
                                    before_ft_acc, keep_indices paths, latex tables
    resnet20_errors.json            failed tasks (if any)

Resumption: re-running skips baseline training and AEI calibration
if checkpoints exist.

Fixes vs v1 (deadlock):
  - mp.Manager().Queue() — no pipe-buffer deadlock
  - Workers always push a result (even on exception) — no silent hang
  - set_start_method at module level

Usage:
  python resnet20_aei_pruning_dgx_v2.py
"""

import copy
import json
import math
import os
import random
import shutil
import time
import traceback
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from scipy.linalg import eigh
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for DGX
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

mp.set_start_method("spawn", force=True)

# ─────────────────────────── Configuration ────────────────────────────────────

SEED            = 42
NUM_SEEDS       = 5
SPARSITY_LEVELS = [0.2, 0.3, 0.4, 0.5, 0.6]
FINETUNE_EPOCHS = 10
TRAIN_EPOCHS    = 100
BATCH_SIZE      = 512
FINETUNE_BS     = 256
LR              = 0.1 * (BATCH_SIZE / 128)
DATA_ROOT       = "./data"

OUT_DIR    = Path("./resnet20_outputs");  OUT_DIR.mkdir(exist_ok=True)
CKPT_DIR   = OUT_DIR / "checkpoints";    CKPT_DIR.mkdir(exist_ok=True)
AEI_DIR    = OUT_DIR / "aei_scores";     AEI_DIR.mkdir(exist_ok=True)
PRUNED_DIR = OUT_DIR / "pruned_models";  PRUNED_DIR.mkdir(exist_ok=True)
KIDX_DIR   = OUT_DIR / "keep_indices";   KIDX_DIR.mkdir(exist_ok=True)
FIG_DIR    = OUT_DIR / "figures";        FIG_DIR.mkdir(exist_ok=True)

N_CALIB         = 1000
N_RANDOM_TRIALS = 1000

NUM_WORKERS = 4
PREFETCH    = 2
PIN_MEMORY  = True

# True  -> save every (method, sparsity, seed) pruned model (~175 files)
# False -> save only best per (method, sparsity) in best/ (~35 files)
SAVE_ALL_SEEDS = True

# ─────────────────────────── Reproducibility ──────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.deterministic = False

set_seed(SEED)

# ─────────────────────────── FLOPs ────────────────────────────────────────────

def count_flops_resnet20(model: nn.Module) -> int:
    """
    Approximate FLOPs for ResNet-20 on CIFAR-10 (32x32 input).
    Counts multiply-accumulate ops × 2 for conv layers and FC.
    Accounts for zeroed-out filters (weight masking = structural sparsity
    in FLOPs terms since those filters still execute, but we compute the
    theoretical FLOPs as if they were structurally removed).
    """
    base   = getattr(model, "_orig_mod", model)
    total  = 0
    in_h, in_w = 32, 32

    def conv_flops(module, h, w):
        # Active filters = filters with non-zero weight
        w_data  = module.weight.detach()
        active  = (w_data.abs().sum(dim=(1, 2, 3)) > 0).sum().item()
        c_in    = module.in_channels
        kh, kw  = module.kernel_size
        stride  = module.stride[0] if hasattr(module.stride, "__len__") else module.stride
        out_h   = h // stride
        out_w   = w // stride
        # 2 × C_in × kH × kW × active × out_H × out_W
        return 2 * c_in * kh * kw * active * out_h * out_w, out_h, out_w

    # Stem conv
    f, in_h, in_w = conv_flops(base.conv1, in_h, in_w)
    total += f

    for layer_name in ["layer1", "layer2", "layer3"]:
        layer = getattr(base, layer_name)
        for block in layer:
            f, in_h, in_w = conv_flops(block.conv1, in_h, in_w)
            total += f
            f, h2, w2     = conv_flops(block.conv2, in_h, in_w)
            total += f
            # shortcut conv (if present)
            if len(list(block.shortcut.children())) > 0:
                sc_conv = block.shortcut[0]
                f, _, _ = conv_flops(sc_conv, in_h, in_w)
                total += f
            in_h, in_w = h2, w2

    # FC: 2 × in_features × out_features
    total += 2 * base.fc.in_features * base.fc.out_features
    return total

# ─────────────────────────── Checkpoint Helpers ───────────────────────────────

def save_baseline(model: nn.Module, acc: float):
    path  = CKPT_DIR / "baseline_resnet20.pt"
    flops = count_flops_resnet20(model)
    torch.save({
        "state_dict":   getattr(model, "_orig_mod", model).state_dict(),
        "accuracy":     acc,
        "flops":        flops,
        "flops_M":      flops / 1e6,
        "architecture": "ResNet20",
        "dataset":      "CIFAR-10",
        "train_epochs": TRAIN_EPOCHS,
        "batch_size":   BATCH_SIZE,
        "lr":           LR,
        "timestamp":    time.strftime("%Y-%m-%d %H:%M:%S"),
    }, path)
    print(f"  Baseline saved   -> {path}  "
          f"(acc={acc:.2f}%, FLOPs={flops/1e6:.1f}M)")
    return flops


def load_baseline_if_exists():
    """Returns (state_dict, accuracy, flops) or (None, None, None)."""
    path = CKPT_DIR / "baseline_resnet20.pt"
    if path.exists():
        ckpt = torch.load(path, map_location="cpu")
        print(f"  Resuming: baseline found "
              f"(acc={ckpt['accuracy']:.2f}%, "
              f"FLOPs={ckpt.get('flops_M', '?'):.1f}M) -> skipping training")
        return (ckpt["state_dict"], ckpt["accuracy"],
                ckpt.get("flops", None))
    return None, None, None


def save_aei(aei_scores: dict, activations: dict,
             overhead: dict):
    """
    Save AEI scores, raw activations, and full overhead breakdown.
    overhead = {act_collect_s, graph_s, eig_s, total_s}
    """
    torch.save({
        "aei_scores":  {k: torch.from_numpy(v) for k, v in aei_scores.items()},
        "overhead":    overhead,
        "n_calib":     N_CALIB,
        "timestamp":   time.strftime("%Y-%m-%d %H:%M:%S"),
    }, AEI_DIR / "aei_scores.pt")

    torch.save({
        "activations": activations,
        "n_calib":     N_CALIB,
        "timestamp":   time.strftime("%Y-%m-%d %H:%M:%S"),
    }, AEI_DIR / "calibration_activations.pt")

    print(f"  AEI scores       -> {AEI_DIR / 'aei_scores.pt'}")
    print(f"    act_collect={overhead['act_collect_ms']:.0f}ms  "
          f"graph={overhead['graph_ms']:.1f}ms  "
          f"eig={overhead['eig_ms']:.1f}ms  "
          f"total={overhead['total_ms']:.0f}ms")
    print(f"  Activations      -> {AEI_DIR / 'calibration_activations.pt'}")


def load_aei_if_exists():
    """Returns (aei_scores_dict, overhead_dict) or (None, None)."""
    path = AEI_DIR / "aei_scores.pt"
    if path.exists():
        ckpt   = torch.load(path, map_location="cpu")
        scores = {k: v.numpy() for k, v in ckpt["aei_scores"].items()}
        print(f"  Resuming: AEI scores found -> skipping calibration")
        return scores, ckpt["overhead"]
    return None, None


def save_pruned_model(model: nn.Module, method: str, sparsity: float,
                      seed: int, acc: float, acc_before_ft: float,
                      avg_r: float, overhead: dict) -> str:
    """
    Save pruned+fine-tuned model. Includes before-FT accuracy and FLOPs.
    Returns path string stored in results for reference.

    To reload:
        ckpt  = torch.load("L2_sp40_seed42.pt")
        model = ResNet20()
        model.load_state_dict(ckpt["state_dict"])
        print(ckpt["accuracy"])   # post fine-tune accuracy
    """
    fname = f"{method}_sp{int(sparsity * 100):02d}_seed{seed}.pt"
    path  = PRUNED_DIR / fname
    flops = count_flops_resnet20(model)
    torch.save({
        "state_dict":      getattr(model, "_orig_mod", model).state_dict(),
        "architecture":    "ResNet20",
        "dataset":         "CIFAR-10",
        "method":          method,
        "sparsity":        sparsity,
        "seed":            seed,
        "accuracy":        acc,
        "acc_before_ft":   acc_before_ft,
        "avg_r_pct":       avg_r,
        "flops":           flops,
        "flops_M":         flops / 1e6,
        "overhead_ms":     {k: v * 1000 for k, v in overhead.items()},
        "finetune_epochs": FINETUNE_EPOCHS,
        "timestamp":       time.strftime("%Y-%m-%d %H:%M:%S"),
    }, path)
    return str(path)


def save_best_pruned_models(raw_results: dict):
    best_dir = PRUNED_DIR / "best"
    best_dir.mkdir(exist_ok=True)
    for (method, sparsity), items in raw_results.items():
        ok = [x for x in items
              if x.get("ok") and x.get("ckpt_path") is not None]
        if not ok:
            continue
        best = max(ok, key=lambda x: x["accuracy"])
        src  = Path(best["ckpt_path"])
        if src.exists():
            dest = best_dir / f"{method}_sp{int(sparsity * 100):02d}_best.pt"
            shutil.copy2(src, dest)
    print(f"  Best models      -> {PRUNED_DIR / 'best'}/")


def save_keep_indices(keep_indices: dict):
    """
    Save {method: {sparsity: np.array of kept filter indices}}.
    These are needed to reproduce the spectral overlap figures later.
    """
    serialisable = {
        method: {
            str(sp): torch.from_numpy(np.array(idx_arr))
            for sp, idx_arr in sp_dict.items()
        }
        for method, sp_dict in keep_indices.items()
    }
    path = KIDX_DIR / "keep_indices.pt"
    torch.save({
        "keep_indices": serialisable,
        "timestamp":    time.strftime("%Y-%m-%d %H:%M:%S"),
    }, path)
    print(f"  Keep indices     -> {path}")

# ─────────────────────────── Data ─────────────────────────────────────────────

def get_cifar10_loaders(batch_size: int = BATCH_SIZE):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(DATA_ROOT, train=True,
                                             download=True, transform=train_tf)
    test_set  = torchvision.datasets.CIFAR10(DATA_ROOT, train=False,
                                             download=True, transform=test_tf)

    kw = dict(
        num_workers        = NUM_WORKERS,
        pin_memory         = PIN_MEMORY,
        persistent_workers = (NUM_WORKERS > 0),
        prefetch_factor    = PREFETCH if NUM_WORKERS > 0 else None,
    )

    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True,  **kw),
        DataLoader(test_set,  batch_size=512,        shuffle=False, **kw),
        DataLoader(Subset(train_set, list(range(N_CALIB))),
                   batch_size=512, shuffle=False, **kw),
    )

# ─────────────────────────── ResNet-20 ────────────────────────────────────────

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1    = nn.Conv2d(in_planes, planes, 3, stride=stride,
                                  padding=1, bias=False)
        self.bn1      = nn.BatchNorm2d(planes)
        self.conv2    = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2      = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet20(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 16
        self.conv1  = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 3, stride=1)
        self.layer2 = self._make_layer(32, 3, stride=2)
        self.layer3 = self._make_layer(64, 3, stride=2)
        self.fc     = nn.Linear(64, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        layers = []
        for s in [stride] + [1] * (num_blocks - 1):
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        return self.fc(F.adaptive_avg_pool2d(out, 1).flatten(1))

# ─────────────────────────── Training ─────────────────────────────────────────

def train_epoch_amp(model, loader, optimizer, scaler, scheduler, device):
    model.train()
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = F.cross_entropy(model(x), y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    if scheduler:
        scheduler.step()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            preds = model(x).argmax(1)
        correct += preds.eq(y).sum().item()
        total   += y.size(0)
    return 100.0 * correct / total


def train_model(model, train_loader, test_loader, device,
                epochs=TRAIN_EPOCHS, lr=LR):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = torch.cuda.amp.GradScaler()
    best_acc, best_state = 0.0, None
    for epoch in range(1, epochs + 1):
        train_epoch_amp(model, train_loader, optimizer, scaler, scheduler, device)
        acc = evaluate(model, test_loader, device)
        if acc > best_acc:
            best_acc   = acc
            best_state = copy.deepcopy(
                getattr(model, "_orig_mod", model).state_dict()
            )
        if epoch % 20 == 0:
            print(f"    Epoch {epoch:3d}/{epochs} — acc {acc:.2f}%")
    getattr(model, "_orig_mod", model).load_state_dict(best_state)
    return best_acc


def finetune(model, train_loader, test_loader, device,
             epochs=FINETUNE_EPOCHS, lr=1e-3):
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = torch.cuda.amp.GradScaler()
    best_acc  = 0.0
    for _ in range(epochs):
        train_epoch_amp(model, train_loader, optimizer, scaler, scheduler, device)
        acc = evaluate(model, test_loader, device)
        best_acc = max(best_acc, acc)
    return best_acc

# ─────────────────────────── Prunable Filters ─────────────────────────────────

def get_prunable_convs(model: nn.Module):
    base = getattr(model, "_orig_mod", model)
    prunable = [("conv1", base.conv1)]
    for lname in ["layer1", "layer2", "layer3"]:
        for bidx, block in enumerate(getattr(base, lname)):
            prunable.append((f"{lname}.{bidx}.conv1", block.conv1))
            if len(list(block.shortcut.children())) == 0:
                prunable.append((f"{lname}.{bidx}.conv2", block.conv2))
    return prunable

# ─────────────────────────── Activations + AEI ────────────────────────────────

@torch.no_grad()
def collect_filter_activations(model, calib_loader, prunable, device):
    model.eval()
    bufs = {name: [] for name, _ in prunable}

    def make_hook(name):
        def hook(_, __, out):
            bufs[name].append(out.detach().mean(dim=[2, 3]).cpu())
        return hook

    handles = [conv.register_forward_hook(make_hook(n)) for n, conv in prunable]
    for x, _ in calib_loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            model(x.to(device, non_blocking=True))
    for h in handles:
        h.remove()
    return {name: torch.cat(b, dim=0) for name, b in bufs.items()}


def pearson_corr(X: np.ndarray, eps=1e-8):
    X = X - X.mean(0, keepdims=True)
    X /= np.linalg.norm(X, axis=0, keepdims=True) + eps
    C  = np.abs(X.T @ X)
    np.fill_diagonal(C, 0.0)
    return C


def compute_aei_with_timing(activations: dict):
    """
    Run full AEI pipeline with per-stage timing.
    Returns (aei_scores_dict, overhead_dict).
    overhead keys: act_collect_ms, graph_ms, eig_ms, total_ms
    (act_collect_ms is 0 here since activations are pre-collected;
     populated by the caller who measures collection time separately)
    """
    t_graph_total = 0.0
    t_eig_total   = 0.0
    aei_scores    = {}

    for name, acts in activations.items():
        if acts.shape[1] < 3:
            aei_scores[name] = np.zeros(acts.shape[1])
            continue

        X = acts.numpy().astype(np.float64)

        t0 = time.time()
        C  = pearson_corr(X)
        t_graph_total += time.time() - t0

        A = np.maximum(C, 0.0)
        np.fill_diagonal(A, 0.0)
        L = np.diag(A.sum(1)) - A
        n = L.shape[0]

        t0 = time.time()
        _, vecs = eigh(L, subset_by_index=[0, min(2, n - 1)])
        t_eig_total += time.time() - t0

        v2 = vecs[:, 1] if vecs.shape[1] > 1 else np.zeros(n)
        aei_scores[name] = np.array(
            [np.sum(A[i] * np.abs(v2[i] - v2)) for i in range(n)]
        )

    return aei_scores, {
        "graph_ms": t_graph_total * 1000,
        "eig_ms":   t_eig_total   * 1000,
    }

# ─────────────────────────── Scoring + Pruning ────────────────────────────────

def _mm(arr):
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-10)


def global_entries(model, prunable, score_fn):
    return [(n, i, float(s))
            for n, conv in prunable
            for i, s in enumerate(score_fn(n, conv))]


def apply_prune(model, entries, sparsity):
    """Zero out lowest-scoring filters. Returns (pruned_entries, keep_indices_flat)."""
    n      = int(len(entries) * sparsity)
    sorted_e = sorted(entries, key=lambda x: x[2])
    pruned   = sorted_e[:n]
    kept     = sorted_e[n:]
    pmap     = {}
    for name, fi, _ in pruned:
        pmap.setdefault(name, set()).add(fi)
    base = getattr(model, "_orig_mod", model)
    for name, m in base.named_modules():
        if isinstance(m, nn.Conv2d) and name in pmap:
            with torch.no_grad():
                for fi in pmap[name]:
                    m.weight[fi].zero_()
    keep_idx = np.array([fi for _, fi, _ in kept])
    return pruned, keep_idx


def l1_fn(n, c):
    return np.abs(c.weight.detach().cpu().numpy()).sum((1, 2, 3))

def l2_fn(n, c):
    w = c.weight.detach().cpu().numpy()
    return np.sqrt((w ** 2).sum((1, 2, 3)))

def rand_fn(n, c):
    return np.random.rand(c.weight.shape[0])

def snip_fn(model, prunable, calib_loader, device):
    model.train()
    x, y = next(iter(calib_loader))
    F.cross_entropy(model(x.to(device)), y.to(device)).backward()
    scores = {}
    for name, conv in prunable:
        w = conv.weight.detach().cpu().numpy()
        g = (conv.weight.grad.cpu().numpy()
             if conv.weight.grad is not None else np.zeros_like(w))
        scores[name] = np.abs(g * w).sum((1, 2, 3))
    for p in model.parameters():
        p.grad = None
    return lambda n, c: scores[n]

def grasp_fn(model, prunable, calib_loader, device):
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    x, y   = next(iter(calib_loader))
    grads  = torch.autograd.grad(
        F.cross_entropy(model(x.to(device)), y.to(device)),
        params, create_graph=True
    )
    hg = torch.autograd.grad(
        sum((g * g.detach()).sum() for g in grads), params
    )
    pid_hg = {id(p): hgv.detach().cpu().numpy() for p, hgv in zip(params, hg)}
    scores = {}
    for name, conv in prunable:
        hgv = pid_hg.get(id(conv.weight),
                         np.zeros_like(conv.weight.detach().cpu().numpy()))
        scores[name] = (hgv * conv.weight.detach().cpu().numpy()).sum((1, 2, 3))
    for p in model.parameters():
        p.grad = None
    return lambda n, c: scores[n]

def aei_fn(aei_scores, n, c):
    return aei_scores.get(n, np.ones(c.weight.shape[0]))

def hybrid_fn(aei_scores, n, c):
    return _mm(aei_fn(aei_scores, n, c)) * _mm(l2_fn(n, c))

# ─────────────────────────── Spectral Overlap ─────────────────────────────────

def avg_r_pct(pruned_entries, aei_scores, prunable):
    r_flat, lut = [], {}
    for name, conv in prunable:
        R = aei_scores.get(name, np.zeros(conv.weight.shape[0]))
        for i, r in enumerate(R):
            lut[(name, i)] = len(r_flat)
            r_flat.append(r)
    r_flat = np.array(r_flat)
    vals   = [r_flat[lut[(nm, fi)]]
              for nm, fi, _ in pruned_entries if (nm, fi) in lut]
    return float(np.mean([100 * (r_flat < v).mean() for v in vals])) if vals else 50.0


def get_flat_r_scores(aei_scores, prunable):
    """Return flat array of all R values (global across layers)."""
    flat = []
    for name, conv in prunable:
        R = aei_scores.get(name, np.zeros(conv.weight.shape[0]))
        flat.extend(R.tolist())
    return np.array(flat)

# ─────────────────────────── Figures ──────────────────────────────────────────

METHODS_DISPLAY = [
    "Spectral (AEI)", "Hybrid (AEI×L2)",
    "L1", "L2", "SNIP", "GraSP", "Random"
]
METHOD_COLORS = {
    "Spectral (AEI)": "#2196F3",
    "Hybrid (AEI×L2)": "#9C27B0",
    "L1":    "#FF9800",
    "L2":    "#4CAF50",
    "SNIP":  "#F44336",
    "GraSP": "#607D8B",
    "Random":"#795548",
}


def plot_spectral_histograms(aei_scores, prunable, all_pruned_entries,
                              sparsity=0.3, save_path=None):
    """
    Mirrors the spectral histogram figure in the paper.
    Shows where each method prunes on the AEI R-value spectrum.
    all_pruned_entries: {method: [(name, fi, score), ...]} for this sparsity
    """
    R_flat = get_flat_r_scores(aei_scores, prunable)
    methods = list(all_pruned_entries.keys())
    n_plots = len(methods)
    ncols   = 4
    nrows   = math.ceil(n_plots / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    fig.suptitle(
        f"Where each method prunes on the spectral (AEI R-value) spectrum\n"
        f"ResNet-20 / CIFAR-10, {int(sparsity * 100)}% sparsity",
        fontsize=13, fontweight="bold"
    )

    aei_pruned = set((nm, fi) for nm, fi, _ in
                     all_pruned_entries.get("Spectral (AEI)", []))

    lut = {}
    for name, conv in prunable:
        R = aei_scores.get(name, np.zeros(conv.weight.shape[0]))
        for i, r in enumerate(R):
            lut[(name, i)] = r

    for ax, method in zip(axes, methods):
        pruned = all_pruned_entries.get(method, [])
        kept_r  = [r for nm, fi, _ in pruned
                   if (nm, fi) in lut
                   for r in [] ]  # placeholder
        # collect R-values
        pruned_set = set((nm, fi) for nm, fi, _ in pruned)
        r_pruned   = np.array([lut[(nm, fi)]
                               for (nm, fi) in pruned_set if (nm, fi) in lut])
        r_kept     = np.array([v for (nm, fi), v in lut.items()
                               if (nm, fi) not in pruned_set])

        overlap = len(aei_pruned & pruned_set) / max(len(aei_pruned), 1) * 100
        avg_pct = (np.mean([100 * (R_flat < v).mean() for v in r_pruned])
                   if len(r_pruned) else 0)

        color = "#2196F3" if method == "Spectral (AEI)" else "#F44336"
        bins  = np.linspace(R_flat.min(), R_flat.max(), 20)

        ax.hist(r_kept,  bins=bins, alpha=0.4, color="grey", label="kept")
        ax.hist(r_pruned, bins=bins, alpha=0.7, color=color,  label="pruned")
        ax.set_xlabel("R value (AEI score)", fontsize=8)
        ax.set_ylabel("Filter count", fontsize=8)

        title = method
        if method != "Spectral (AEI)":
            title += f"\noverlap w/ AEI: {overlap:.1f}%  avg R-pctile: {avg_pct:.1f}"
        ax.set_title(title, fontsize=8)
        ax.legend(fontsize=7)

    for ax in axes[len(methods):]:
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Spectral histogram-> {save_path}")
    plt.close(fig)


def plot_accuracy_comparison(results, baseline_acc, save_path=None):
    """Bar chart: all methods × sparsity levels. Matches notebook figure."""
    sparsities = [int(s * 100) for s in SPARSITY_LEVELS]
    methods    = [m for m in METHODS_DISPLAY if m in results]
    x          = np.arange(len(sparsities))
    width      = 0.11

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, method in enumerate(methods):
        accs, errs = [], []
        for s in SPARSITY_LEVELS:
            r = results[method].get(s, {})
            accs.append(r.get("acc_mean", 0))
            errs.append(r.get("acc_std",  0))
        offset = (i - len(methods) / 2 + 0.5) * width
        ax.bar(x + offset, accs, width,
               label=method,
               color=METHOD_COLORS.get(method, "#888"),
               alpha=0.85,
               yerr=errs if any(e > 0 for e in errs) else None,
               capsize=3)

    ax.axhline(baseline_acc, color="black", linestyle="--",
               linewidth=1.5, label=f"Baseline ({baseline_acc:.2f}%)")
    ax.set_xlabel("Sparsity level (%)", fontsize=12)
    ax.set_ylabel("Fine-tuned accuracy (%)", fontsize=12)
    ax.set_title("ResNet-20 / CIFAR-10: Pruning method comparison",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}%" for s in sparsities])
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Accuracy chart   -> {save_path}")
    plt.close(fig)


def plot_overhead_breakdown(overhead_log, save_path=None):
    """
    Horizontal bar breakdown of overhead per method.
    Matches the overhead table in the notebook but as a figure.
    """
    methods = list(overhead_log.keys())
    totals  = [overhead_log[m].get("total_ms", 0) for m in methods]

    fig, ax = plt.subplots(figsize=(9, 4))
    y = np.arange(len(methods))

    act_ms   = [overhead_log[m].get("act_collect_ms", 0) for m in methods]
    graph_ms = [overhead_log[m].get("graph_ms", 0)       for m in methods]
    eig_ms   = [overhead_log[m].get("eig_ms", 0)         for m in methods]
    other_ms = [max(0, t - a - g - e)
                for t, a, g, e in zip(totals, act_ms, graph_ms, eig_ms)]

    ax.barh(y, act_ms,   label="Activation collection", color="#2196F3", alpha=0.85)
    ax.barh(y, graph_ms, left=act_ms, label="Graph construction",  color="#FF9800", alpha=0.85)
    ax.barh(y, eig_ms,   left=[a + g for a, g in zip(act_ms, graph_ms)],
            label="Eigendecomposition", color="#9C27B0", alpha=0.85)
    ax.barh(y, other_ms,
            left=[a + g + e for a, g, e in zip(act_ms, graph_ms, eig_ms)],
            label="Other", color="#888", alpha=0.4)

    ax.set_yticks(y)
    ax.set_yticklabels(methods)
    ax.set_xlabel("Time (ms)")
    ax.set_title("Pruning overhead breakdown (one-time offline cost)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Overhead chart   -> {save_path}")
    plt.close(fig)

# ─────────────────────────── Worker ───────────────────────────────────────────

def worker_run(gpu_id, task_queue, result_queue,
               pretrained_state, aei_scores_shared, aei_overhead):
    """
    Single-GPU worker. ALL exceptions caught — always pushes result.
    Tracks acc_before_ft and FLOPs in addition to post-FT accuracy.
    """
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    set_seed(SEED + gpu_id)

    train_loader, test_loader, calib_loader = get_cifar10_loaders(FINETUNE_BS)

    while True:
        item = task_queue.get()
        if item is None:
            break

        method, sparsity, seed, task_id = item

        try:
            set_seed(seed)
            model    = ResNet20().to(device)
            model.load_state_dict(copy.deepcopy(pretrained_state))
            prunable = get_prunable_convs(model)

            # ── Score ──────────────────────────────────────────────────────
            t0 = time.time()
            if   method == "L1":     sfn = lambda n, c: l1_fn(n, c)
            elif method == "L2":     sfn = lambda n, c: l2_fn(n, c)
            elif method == "Random": sfn = lambda n, c: rand_fn(n, c)
            elif method == "SNIP":   sfn = snip_fn(model, prunable, calib_loader, device)
            elif method == "GraSP":  sfn = grasp_fn(model, prunable, calib_loader, device)
            elif method == "AEI":    sfn = lambda n, c: aei_fn(aei_scores_shared, n, c)
            elif method == "Hybrid": sfn = lambda n, c: hybrid_fn(aei_scores_shared, n, c)
            else: raise ValueError(f"Unknown method: {method}")

            entries  = global_entries(model, prunable, sfn)
            t_score  = time.time() - t0

            # overhead dict (method-specific, not including AEI collection)
            overhead = {"score_s": t_score}
            if method in ("AEI", "Hybrid"):
                overhead["act_collect_ms"] = aei_overhead.get("act_collect_ms", 0)
                overhead["graph_ms"]       = aei_overhead.get("graph_ms", 0)
                overhead["eig_ms"]         = aei_overhead.get("eig_ms", 0)
                overhead["total_ms"]       = (aei_overhead.get("total_ms", 0)
                                              + t_score * 1000)
            else:
                overhead["total_ms"] = t_score * 1000

            # ── Prune ──────────────────────────────────────────────────────
            pruned, keep_idx = apply_prune(model, entries, sparsity)
            rval             = avg_r_pct(pruned, aei_scores_shared, prunable)

            # ── Accuracy BEFORE fine-tuning ────────────────────────────────
            acc_before_ft = evaluate(model, test_loader, device)

            # ── Fine-tune ──────────────────────────────────────────────────
            acc = finetune(model, train_loader, test_loader, device)

            # ── Save pruned model ──────────────────────────────────────────
            ckpt_path = None
            if SAVE_ALL_SEEDS:
                ckpt_path = save_pruned_model(
                    model, method, sparsity, seed,
                    acc, acc_before_ft, rval, overhead
                )

            result_queue.put({
                "ok":           True,
                "task_id":      task_id,
                "method":       method,
                "sparsity":     sparsity,
                "seed":         seed,
                "accuracy":     acc,
                "acc_before_ft": acc_before_ft,
                "avg_r_pct":    rval,
                "overhead":     overhead,
                "keep_idx":     keep_idx.tolist(),
                "ckpt_path":    ckpt_path,
            })

        except Exception as e:
            result_queue.put({
                "ok":       False,
                "task_id":  task_id,
                "method":   method,
                "sparsity": sparsity,
                "seed":     seed,
                "error":    traceback.format_exc(),
            })
            print(f"[GPU {gpu_id}] ERROR task {task_id} "
                  f"({method} @ {sparsity}): {e}", flush=True)

        finally:
            try: del model
            except Exception: pass
            torch.cuda.empty_cache()


def random_worker(gpu_id, n_trials, pretrained_state, result_queue):
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    train_loader, test_loader, _ = get_cifar10_loaders(FINETUNE_BS)
    for trial in range(n_trials):
        try:
            model    = ResNet20().to(device)
            model.load_state_dict(copy.deepcopy(pretrained_state))
            prunable = get_prunable_convs(model)
            apply_prune(model, global_entries(model, prunable, rand_fn), 0.4)
            acc = finetune(model, train_loader, test_loader, device)
            result_queue.put({"ok": True, "acc": acc})
        except Exception as e:
            result_queue.put({"ok": False, "acc": None, "error": str(e)})
        finally:
            try: del model
            except Exception: pass
            torch.cuda.empty_cache()

# ─────────────────────────── Main ─────────────────────────────────────────────

def main():
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        raise RuntimeError("No CUDA GPUs found.")
    print(f"GPUs ({n_gpus}): "
          + " | ".join(torch.cuda.get_device_name(i) for i in range(n_gpus)))
    print(f"PyTorch {torch.__version__} | CUDA {torch.version.cuda}")
    print(f"Output dir: {OUT_DIR.resolve()}")
    print("-" * 70)

    device0 = torch.device("cuda:0")
    train_loader, test_loader, calib_loader = get_cifar10_loaders(BATCH_SIZE)

    # ── 1. Baseline ───────────────────────────────────────────────────────────
    print("\n[1/4] Baseline ResNet-20...")
    pretrained_state, base_acc, base_flops = load_baseline_if_exists()

    if pretrained_state is None:
        model = ResNet20().to(device0)
        if hasattr(torch, "compile"):
            model = torch.compile(model)
        t0       = time.time()
        base_acc = train_model(model, train_loader, test_loader, device0)
        print(f"  Trained in {time.time()-t0:.0f}s — acc {base_acc:.2f}%")
        base_flops = save_baseline(model, base_acc)
        pretrained_state = copy.deepcopy(
            getattr(model, "_orig_mod", model).state_dict()
        )
        del model; torch.cuda.empty_cache()

    print(f"  Baseline FLOPs: {base_flops/1e6:.1f}M")

    # ── 2. AEI scores ─────────────────────────────────────────────────────────
    print("\n[2/4] AEI scores...")
    aei_scores, aei_overhead = load_aei_if_exists()

    if aei_scores is None:
        base = ResNet20().to(device0)
        base.load_state_dict(pretrained_state)
        prunable_ref = get_prunable_convs(base)

        # Time activation collection separately
        t0   = time.time()
        acts = collect_filter_activations(base, calib_loader,
                                          prunable_ref, device0)
        t_act = time.time() - t0

        aei_scores, timing = compute_aei_with_timing(acts)
        aei_overhead = {
            "act_collect_ms": t_act   * 1000,
            "graph_ms":       timing["graph_ms"],
            "eig_ms":         timing["eig_ms"],
            "total_ms":       (t_act + timing["graph_ms"] / 1000
                               + timing["eig_ms"] / 1000) * 1000,
        }
        aei_overhead["total_ms"] = (
            aei_overhead["act_collect_ms"]
            + aei_overhead["graph_ms"]
            + aei_overhead["eig_ms"]
        )
        save_aei(aei_scores, acts, aei_overhead)
        del base; torch.cuda.empty_cache()

    # ── 3. Pruning experiments ────────────────────────────────────────────────
    methods = ["L1", "L2", "SNIP", "GraSP", "Random", "Spectral (AEI)", "Hybrid (AEI×L2)"]
    tasks   = [
        (m, s, SEED + si * 100, idx)
        for idx, (m, s, si) in enumerate(
            (m, s, si)
            for m in methods
            for s in SPARSITY_LEVELS
            for si in range(NUM_SEEDS)
        )
    ]
    n_tasks = len(tasks)
    print(f"\n[3/4] {n_tasks} tasks on {n_gpus} GPU(s)...")

    manager  = mp.Manager()
    task_q   = manager.Queue()
    result_q = manager.Queue()

    for task in tasks: task_q.put(task)
    for _ in range(n_gpus): task_q.put(None)

    workers = []
    for gpu_id in range(n_gpus):
        p = mp.Process(
            target=worker_run,
            args=(gpu_id, task_q, result_q,
                  pretrained_state, aei_scores, aei_overhead),
        )
        p.start(); workers.append(p)

    raw    = {}
    errors = []
    done   = 0
    t0     = time.time()

    while done < n_tasks:
        try:
            res = result_q.get(timeout=30)
        except Exception:
            alive = [p for p in workers if p.is_alive()]
            if not alive:
                raise RuntimeError(
                    f"All workers died after {done}/{n_tasks} tasks."
                )
            print(f"  [heartbeat {done}/{n_tasks}, "
                  f"{len(alive)} workers alive]", flush=True)
            continue

        done += 1
        eta   = (time.time() - t0) / done * (n_tasks - done)

        if res["ok"]:
            key = (res["method"], res["sparsity"])
            raw.setdefault(key, []).append(res)
            print(f"  [{done:3d}/{n_tasks}] {res['method']:18s} "
                  f"@ {res['sparsity']:.0%}  "
                  f"acc={res['accuracy']:.2f}%  "
                  f"(before FT={res['acc_before_ft']:.2f}%)  "
                  f"R%={res['avg_r_pct']:.1f}  "
                  f"ETA {eta/60:.1f}min", flush=True)
        else:
            errors.append(res)
            print(f"  [{done:3d}/{n_tasks}] FAILED: "
                  f"{res['method']} @ {res['sparsity']} "
                  f"seed={res['seed']}", flush=True)

    for p in workers: p.join()

    # ── Post-processing ───────────────────────────────────────────────────────
    print("\n  Post-processing...")
    save_best_pruned_models(raw)

    if errors:
        with open(OUT_DIR / "resnet20_errors.json", "w") as f:
            json.dump(errors, f, indent=2)
        print(f"  {len(errors)} failed tasks -> resnet20_errors.json")

    # Aggregate results + collect overhead log + keep_indices
    results      = {}
    overhead_log = {
        "Spectral (AEI)":  {**aei_overhead},
        "Hybrid (AEI×L2)": {**aei_overhead},
        "L1":    {"total_ms": 0}, "L2": {"total_ms": 0},
        "SNIP":  {"total_ms": 0}, "GraSP": {"total_ms": 0},
        "Random":{"total_ms": 0},
    }
    keep_indices = defaultdict(dict)   # method -> sparsity -> keep_idx array

    for method in methods:
        results[method] = {}
        for sparsity in SPARSITY_LEVELS:
            items = raw.get((method, sparsity), [])
            accs          = [x["accuracy"]       for x in items]
            accs_before   = [x["acc_before_ft"]  for x in items]
            rpcts         = [x["avg_r_pct"]       for x in items]
            oh_vals       = [x["overhead"].get("total_ms", 0) for x in items]
            best_item     = max(items, key=lambda x: x["accuracy"]) if items else {}

            results[method][sparsity] = {
                "acc_mean":           float(np.mean(accs))        if accs  else float("nan"),
                "acc_std":            float(np.std(accs))         if accs  else float("nan"),
                "acc_before_ft_mean": float(np.mean(accs_before)) if accs_before else float("nan"),
                "r_pct_mean":         float(np.mean(rpcts))       if rpcts else float("nan"),
                "r_pct_std":          float(np.std(rpcts))        if rpcts else float("nan"),
                "all_accs":           accs,
                "all_accs_before_ft": accs_before,
                "all_r_pcts":         rpcts,
                "n_ok":               len(items),
                "best_ckpt":          best_item.get("ckpt_path"),
                "flops_M":            count_flops_resnet20(
                    ResNet20()          # placeholder — real FLOPs in per-run ckpt
                ) / 1e6,
            }

            # Overhead: use median across seeds for stability
            if oh_vals and method not in ("Spectral (AEI)", "Hybrid (AEI×L2)"):
                overhead_log[method]["total_ms"] = float(np.median(oh_vals))

            # Keep indices: use best-accuracy seed
            if best_item.get("keep_idx"):
                keep_indices[method][sparsity] = np.array(best_item["keep_idx"])

    # Save keep_indices
    save_keep_indices(keep_indices)

    # ── 4. Random distribution ────────────────────────────────────────────────
    print(f"\n[4/4] Random distribution "
          f"({N_RANDOM_TRIALS} trials, {n_gpus} GPUs)...")
    per_gpu    = math.ceil(N_RANDOM_TRIALS / n_gpus)
    rand_q     = manager.Queue()
    rprocs     = []
    dispatched = 0
    for gpu_id in range(n_gpus):
        n = min(per_gpu, N_RANDOM_TRIALS - dispatched)
        if n <= 0: break
        p = mp.Process(target=random_worker,
                       args=(gpu_id, n, pretrained_state, rand_q))
        p.start(); rprocs.append(p); dispatched += n

    random_accs = []
    for _ in range(N_RANDOM_TRIALS):
        try:
            res = rand_q.get(timeout=120)
            if res["ok"] and res["acc"] is not None:
                random_accs.append(res["acc"])
        except Exception:
            if not any(p.is_alive() for p in rprocs):
                break
    for p in rprocs: p.join()

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\n  Saving figures...")
    base_ref = ResNet20()
    base_ref.load_state_dict(pretrained_state)
    prunable_ref = get_prunable_convs(base_ref)

    # Spectral histograms at 30% sparsity (main paper sparsity)
    for sp in [0.3, 0.4]:
        all_pruned = {}
        for method in methods:
            items = raw.get((method, sp), [])
            best  = max(items, key=lambda x: x["accuracy"]) if items else None
            if best and best.get("ok"):
                # reconstruct pruned entries list from keep_idx
                # (we store keep_idx, so pruned = all - kept)
                total_filters = sum(conv.weight.shape[0]
                                    for _, conv in prunable_ref)
                kept = set(best["keep_idx"])
                pruned_list = [(f"filter_{i}", i, 0.0)
                               for i in range(total_filters) if i not in kept]
                all_pruned[method] = pruned_list

        plot_spectral_histograms(
            aei_scores, prunable_ref, all_pruned,
            sparsity=sp,
            save_path=FIG_DIR / f"spectral_histograms_{int(sp*100)}pct.pdf"
        )

    # Accuracy bar chart
    plot_accuracy_comparison(
        results, base_acc,
        save_path=FIG_DIR / "accuracy_comparison.pdf"
    )

    # Overhead breakdown
    plot_overhead_breakdown(
        overhead_log,
        save_path=FIG_DIR / "overhead_breakdown.pdf"
    )

    # ── Summary tables ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("RESULTS — ResNet-20 / CIFAR-10")
    print("=" * 80)
    print(f"\nBaseline: {base_acc:.2f}%  |  FLOPs: {base_flops/1e6:.1f}M\n")

    hdr = f"{'Method':<20}" + "".join(f"  {s:.0%}" for s in SPARSITY_LEVELS)
    sep = "-" * len(hdr)

    print("Accuracy after fine-tuning (mean±std)")
    print(hdr); print(sep)
    for m in methods:
        print(f"{m:<20}" + "".join(
            f"  {results[m][s]['acc_mean']:.2f}±{results[m][s]['acc_std']:.2f}"
            for s in SPARSITY_LEVELS
        ))

    print("\nAccuracy BEFORE fine-tuning (mean) — shows structural damage")
    print(hdr); print(sep)
    for m in methods:
        print(f"{m:<20}" + "".join(
            f"  {results[m][s]['acc_before_ft_mean']:.2f}      "
            for s in SPARSITY_LEVELS
        ))

    print("\nAvg R-Percentile (lower = more structural)")
    print(hdr); print(sep)
    for m in methods:
        print(f"{m:<20}" + "".join(
            f"  {results[m][s]['r_pct_mean']:.1f}±{results[m][s]['r_pct_std']:.1f}"
            for s in SPARSITY_LEVELS
        ))

    print("\nOverhead breakdown (ms, one-time offline cost)")
    print(f"{'Method':<20} {'Total':>8} {'Act.Collect':>12} {'Graph':>8} {'Eig':>8}")
    print("-" * 60)
    for m in methods:
        oh = overhead_log.get(m, {})
        print(f"{m:<20} {oh.get('total_ms', 0):>7.0f}ms "
              f"{oh.get('act_collect_ms', 0):>10.0f}ms "
              f"{oh.get('graph_ms', 0):>6.1f}ms "
              f"{oh.get('eig_ms', 0):>6.1f}ms")

    if random_accs:
        print(f"\nRandom @ 40% ({len(random_accs)} trials): "
              f"{np.mean(random_accs):.2f}% ± {np.std(random_accs):.2f}%")

    # ── LaTeX tables ──────────────────────────────────────────────────────────
    cols = " & ".join(f"{s:.0%}" for s in SPARSITY_LEVELS)

    def latex_table(caption, label, val_fn):
        rows = "\n".join(
            m + "".join(f" & {val_fn(m, s)}" for s in SPARSITY_LEVELS) + r" \\"
            for m in methods
        )
        return (
            r"\begin{table}[t]\centering" + "\n"
            + f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
            + r"\begin{tabular}{l" + "c" * len(SPARSITY_LEVELS) + "}\n"
            + r"\toprule" + f"\nMethod & {cols} \\\\\n" + r"\midrule" + "\n"
            + rows + "\n" + r"\midrule" + "\n"
            + f"Baseline & \\multicolumn{{{len(SPARSITY_LEVELS)}}}{{c}}"
              f"{{{base_acc:.2f}}} \\\\\n"
            + r"\bottomrule\end{tabular}\end{table}"
        )

    acc_tex = latex_table(
        "ResNet-20 accuracy (\\%) on CIFAR-10 after pruning and fine-tuning.",
        "tab:resnet20_accuracy",
        lambda m, s: (f"{results[m][s]['acc_mean']:.2f}"
                      r"$\pm$"
                      f"{results[m][s]['acc_std']:.2f}"),
    )
    r_tex = latex_table(
        "Average $R$-percentile of pruned filters — ResNet-20 / CIFAR-10.",
        "tab:resnet20_spectral",
        lambda m, s: (f"{results[m][s]['r_pct_mean']:.1f}"
                      r"$\pm$"
                      f"{results[m][s]['r_pct_std']:.1f}"),
    )

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {
        "baseline_acc":   base_acc,
        "baseline_flops_M": base_flops / 1e6,
        "results":        {
            m: {str(s): v for s, v in sd.items()}
            for m, sd in results.items()
        },
        "random_dist": {
            "accs": random_accs,
            "mean": float(np.mean(random_accs)) if random_accs else None,
            "std":  float(np.std(random_accs))  if random_accs else None,
        },
        "overhead_ms": {
            m: oh for m, oh in overhead_log.items()
        },
        "latex":    {"accuracy": acc_tex, "spectral": r_tex},
        "n_errors": len(errors),
        "paths": {
            "baseline":      str(CKPT_DIR / "baseline_resnet20.pt"),
            "aei_scores":    str(AEI_DIR   / "aei_scores.pt"),
            "activations":   str(AEI_DIR   / "calibration_activations.pt"),
            "keep_indices":  str(KIDX_DIR  / "keep_indices.pt"),
            "pruned_models": str(PRUNED_DIR),
            "best_models":   str(PRUNED_DIR / "best"),
            "figures":       str(FIG_DIR),
        },
        "config": {
            "arch":            "ResNet-20",
            "dataset":         "CIFAR-10",
            "pruning_type":    "global filter pruning (weight masking)",
            "sparsity_levels": SPARSITY_LEVELS,
            "finetune_epochs": FINETUNE_EPOCHS,
            "train_epochs":    TRAIN_EPOCHS,
            "num_seeds":       NUM_SEEDS,
            "n_calib":         N_CALIB,
            "n_gpus":          n_gpus,
            "amp":             True,
            "save_all_seeds":  SAVE_ALL_SEEDS,
        },
    }

    results_path = OUT_DIR / "resnet20_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*70}")
    print("All outputs saved:")
    print(f"  Results JSON     -> {results_path}")
    print(f"  Baseline model   -> {CKPT_DIR / 'baseline_resnet20.pt'}")
    print(f"  AEI scores       -> {AEI_DIR / 'aei_scores.pt'}")
    print(f"  Activations      -> {AEI_DIR / 'calibration_activations.pt'}")
    print(f"  Keep indices     -> {KIDX_DIR / 'keep_indices.pt'}")
    print(f"  All pruned models-> {PRUNED_DIR}/")
    print(f"  Best per config  -> {PRUNED_DIR / 'best'}/")
    print(f"  Spectral histograms  -> {FIG_DIR}/spectral_histograms_*.pdf")
    print(f"  Accuracy chart   -> {FIG_DIR}/accuracy_comparison.pdf")
    print(f"  Overhead chart   -> {FIG_DIR}/overhead_breakdown.pdf")
    if errors:
        print(f"  Error log        -> {OUT_DIR / 'resnet20_errors.json'}")


if __name__ == "__main__":
    main()
