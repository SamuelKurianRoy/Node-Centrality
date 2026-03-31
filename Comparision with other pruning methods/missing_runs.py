"""
missing_runs.py
===============
Runs ONLY the 5 missing combinations from statistical_rigor.py:
  seed=1024 | 40% sparsity | l1_norm, l2_norm, snip, grasp, hybrid

Prints output in the same format as statistical_rigor.py so results
can be directly copy-pasted into the existing table.

Usage (Colab):
    !python missing_runs.py | tee /content/drive/MyDrive/SpectralPruning_Results/missing_runs.txt
"""

import sys, os, copy, time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# ── Try to import HybridPruner from repo; define it inline if absent ──────────
try:
    from pruning_logic import (
        SpectralPruner, L1NormPruner, L2NormPruner,
        SNIPPruner, GraSPPruner, load_dataset, train_model,
    )
    # HybridPruner may not be in the repo yet — define below if needed
    try:
        from pruning_logic import HybridPruner
        _hybrid_from_repo = True
    except ImportError:
        _hybrid_from_repo = False
    _pruning_logic_available = True
    print("Imported pruning_logic from repo.", flush=True)
except ImportError:
    _pruning_logic_available = False
    print("pruning_logic not found — using self-contained implementations.", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Self-contained fallback (only used if pruning_logic.py is not on path)
# Mirrors the exact logic used in the original statistical_rigor.py
# ─────────────────────────────────────────────────────────────────────────────
if not _pruning_logic_available:
    from scipy.linalg import eigh

    # ── Model ────────────────────────────────────────────────────────────────
    class SimpleMLP(nn.Module):
        def __init__(self, input_dim=784, hidden=256, num_classes=10):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden)
            self.fc2 = nn.Linear(hidden, num_classes)
        def forward(self, x):
            x = x.view(x.size(0), -1)
            return self.fc2(torch.relu(self.fc1(x)))

    # ── Dataset ──────────────────────────────────────────────────────────────
    def load_dataset(name, batch_size=128):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        DS = {'FashionMNIST': torchvision.datasets.FashionMNIST,
              'MNIST':        torchvision.datasets.MNIST,
              'KMNIST':       torchvision.datasets.KMNIST}[name]
        train = DataLoader(DS('./data', train=True,  download=True, transform=transform),
                           batch_size=batch_size, shuffle=True,  num_workers=2)
        test  = DataLoader(DS('./data', train=False, download=True, transform=transform),
                           batch_size=batch_size, shuffle=False, num_workers=2)
        return train, test

    # ── Train / eval ─────────────────────────────────────────────────────────
    def train_model(model, loader, device, epochs=10, lr=1e-3):
        model.to(device).train()
        opt  = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(epochs):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss_fn(model(x), y).backward()
                opt.step()
        return model

    def evaluate(model, loader, device):
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total   += y.size(0)
        return 100. * correct / total

    def _create_pruned_model(original, keep_idx, device):
        keep = sorted(keep_idx)
        nk   = len(keep)
        new_model = SimpleMLP(784, nk, 10).to(device)
        new_model.fc1.weight.data = original.fc1.weight.data[keep].clone()
        new_model.fc1.bias.data   = original.fc1.bias.data[keep].clone()
        new_model.fc2.weight.data = original.fc2.weight.data[:, keep].clone()
        new_model.fc2.bias.data   = original.fc2.bias.data.clone()
        return new_model

    # ── Pruners ──────────────────────────────────────────────────────────────
    class L1NormPruner:
        def __init__(self, model, device): self.model, self.device = model, device
        def prune_layer(self, layer, ratio, **kw):
            w = self.model.fc1.weight.data
            scores = w.abs().sum(dim=1).cpu().numpy()
            n_prune = int(len(scores) * ratio)
            keep = np.argsort(scores)[n_prune:].tolist()
            n_keep = len(scores) - n_prune
            print(f"[Node/L1Norm]    Layer 'fc1': {len(scores)} -> {n_keep} neurons "
                  f"({ratio*100:.1f}% pruned by L1 norm)", flush=True)
            return _create_pruned_model(self.model, keep, self.device)

    class L2NormPruner:
        def __init__(self, model, device): self.model, self.device = model, device
        def prune_layer(self, layer, ratio, **kw):
            w = self.model.fc1.weight.data
            scores = w.pow(2).sum(dim=1).sqrt().cpu().numpy()
            n_prune = int(len(scores) * ratio)
            keep = np.argsort(scores)[n_prune:].tolist()
            n_keep = len(scores) - n_prune
            print(f"[Node/L2Norm]    Layer 'fc1': {len(scores)} -> {n_keep} neurons "
                  f"({ratio*100:.1f}% pruned by L2 norm)", flush=True)
            return _create_pruned_model(self.model, keep, self.device)

    class SNIPPruner:
        def __init__(self, model, device, train_loader):
            self.model, self.device, self.train_loader = model, device, train_loader
        def prune_layer(self, layer, ratio, **kw):
            mc = copy.deepcopy(self.model).to(self.device)
            mc.train()
            x, y = next(iter(self.train_loader))
            x, y = x.to(self.device), y.to(self.device)
            nn.CrossEntropyLoss()(mc(x), y).backward()
            scores = (mc.fc1.weight.grad.abs() * mc.fc1.weight.data.abs()
                      ).sum(dim=1).detach().cpu().numpy()
            n_prune = int(len(scores) * ratio)
            keep = np.argsort(scores)[n_prune:].tolist()
            n_keep = len(scores) - n_prune
            print(f"[Node/SNIP]      Layer 'fc1': {len(scores)} -> {n_keep} neurons "
                  f"({ratio*100:.1f}% pruned by SNIP sensitivity)", flush=True)
            return _create_pruned_model(self.model, keep, self.device)

    class GraSPPruner:
        def __init__(self, model, device, train_loader):
            self.model, self.device, self.train_loader = model, device, train_loader
        def prune_layer(self, layer, ratio, **kw):
            mc = copy.deepcopy(self.model).to(self.device)
            mc.train()
            x, y = next(iter(self.train_loader))
            x, y = x.to(self.device), y.to(self.device)
            params = [p for p in mc.parameters() if p.requires_grad]
            loss   = nn.CrossEntropyLoss()(mc(x), y)
            grads  = torch.autograd.grad(loss, params,
                                         create_graph=True, retain_graph=True)
            gnorm  = sum((g * g).sum() for g in grads)
            Hg     = torch.autograd.grad(gnorm, params)
            for p, hg in zip(params, Hg):
                if p is mc.fc1.weight:
                    scores = -(hg * p.data).sum(dim=1).detach().cpu().numpy()
                    break
            n_prune = int(len(scores) * ratio)
            keep = np.argsort(scores)[n_prune:].tolist()
            n_keep = len(scores) - n_prune
            print(f"[Node/GraSP]     Layer 'fc1': {len(scores)} -> {n_keep} neurons "
                  f"({ratio*100:.1f}% pruned by GraSP)", flush=True)
            return _create_pruned_model(self.model, keep, self.device)

    class SpectralPruner:  # needed only to keep the import block clean
        pass

    _hybrid_from_repo = False


# ── HybridPruner (inline — used whether or not repo version exists) ───────────
if not _hybrid_from_repo:
    from scipy.linalg import eigh as _eigh

    def _collect_activations(model, loader, device, n_batches=10):
        model.eval()
        acts = []
        def hook(m, i, o): acts.append(o.detach().cpu().numpy())
        h = model.fc1.register_forward_hook(hook)
        with torch.no_grad():
            for idx, (x, _) in enumerate(loader):
                if idx >= n_batches: break
                model(x.to(device))
        h.remove()
        return np.concatenate(acts, axis=0)

    def _aei_scores(acts, threshold=0.3):
        n    = acts.shape[1]
        corr = np.corrcoef(acts.T)
        np.fill_diagonal(corr, 0.)
        adj  = (np.abs(corr) > threshold).astype(float) * np.abs(corr)
        deg  = adj.sum(1)
        ds   = np.where(deg > 0, deg, 1.)
        L    = np.eye(n) - np.diag(1./np.sqrt(ds)) @ adj @ np.diag(1./np.sqrt(ds))
        _, vecs = _eigh(L)
        v2 = vecs[:, 1]
        R  = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if adj[i, j] > 0:
                    R[i] += abs(v2[i] - v2[j])
        return R

    def _create_pruned_model_hybrid(original, keep_idx, device):
        """Works for both standalone and repo model."""
        keep = sorted(keep_idx)
        nk   = len(keep)
        try:
            # repo path — use existing helper
            from pruning_logic import _create_pruned_model as _cpm
            return _cpm(original, 'fc1', keep)
        except Exception:
            # fallback to local SimpleMLP
            new_model = SimpleMLP(784, nk, 10).to(device)
            new_model.fc1.weight.data = original.fc1.weight.data[keep].clone()
            new_model.fc1.bias.data   = original.fc1.bias.data[keep].clone()
            new_model.fc2.weight.data = original.fc2.weight.data[:, keep].clone()
            new_model.fc2.bias.data   = original.fc2.bias.data.clone()
            return new_model

    class HybridPruner:
        def __init__(self, model, device, train_loader):
            self.model, self.device, self.train_loader = model, device, train_loader

        def prune_layer(self, layer, ratio, **kw):
            acts  = _collect_activations(self.model, self.train_loader, self.device)
            R     = _aei_scores(acts)
            R_n   = (R - R.min()) / (R.max() - R.min() + 1e-8)
            w     = self.model.fc1.weight.data
            L2    = w.pow(2).sum(dim=1).sqrt().cpu().numpy()
            L2_n  = (L2 - L2.min()) / (L2.max() - L2.min() + 1e-8)
            scores = R_n * L2_n
            n_prune = int(len(scores) * ratio)
            keep = np.argsort(scores)[n_prune:].tolist()
            n_keep = len(scores) - n_prune
            print(f"[Node/Hybrid] Layer 'fc1': {len(scores)} -> {n_keep} neurons "
                  f"({ratio*100:.1f}% pruned by AEI x L2 hybrid score)", flush=True)
            return _create_pruned_model_hybrid(self.model, keep, self.device)


# ─────────────────────────────────────────────────────────────────────────────
# Shared train/finetune/eval — works with both repo and standalone models
# ─────────────────────────────────────────────────────────────────────────────

def _train(model, loader, device, epochs, lr):
    model.to(device).train()
    opt = optim.Adam(model.parameters(), lr=lr)
    ce  = nn.CrossEntropyLoss()
    for ep in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            ce(model(x), y).backward()
            opt.step()
    return model

def _eval(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total   += y.size(0)
    return 100. * correct / total


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    # ── The 5 missing runs ────────────────────────────────────────────────────
    SEED        = 1024
    RATIO       = 0.40
    DATASET     = 'FashionMNIST'
    MISSING_METHODS = ['l1_norm', 'l2_norm', 'snip', 'grasp', 'hybrid']

    print(f"\nRunning MISSING RUNS ONLY")
    print(f"  seed={SEED}  |  ratio={int(RATIO*100)}%  |  dataset={DATASET}")
    print(f"  methods: {MISSING_METHODS}")
    print("=" * 65, flush=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    train_dataset, test_dataset, input_size, num_classes = load_dataset(DATASET)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=1000, shuffle=False,
                              num_workers=2, pin_memory=True)
    # ── Train base model with this seed ───────────────────────────────────────
    print(f"\nTraining base model (seed={SEED}, 10 epochs)…", flush=True)
    set_seed(SEED)

    if _pruning_logic_available:
        # Use repo model class
        from pruning_logic import SimpleMLP as RepoMLP
        base_model = RepoMLP().to(device)
    else:
        base_model = SimpleMLP().to(device)

    base_acc = train_model(base_model, train_loader, test_loader, device, epochs=10, lr=1e-3)
    print(f"Base model accuracy: {base_acc:.2f}%", flush=True)

    # ── Run each missing method ────────────────────────────────────────────────
    results = {}
    total   = len(MISSING_METHODS)

    # global run index — continuing from run 86 in the original output
    global_start = 87  # runs 87-91 (1-indexed, completing the 90-run series)

    for i, method in enumerate(MISSING_METHODS):
        run_idx = global_start + i
        print(f"\n[{run_idx}/90] seed={SEED} | {int(RATIO*100)}% | {method}", flush=True)

        model_copy = copy.deepcopy(base_model).to(device)

        try:
            # ── Instantiate pruner ────────────────────────────────────────────
            if method == 'l1_norm':
                pruner = L1NormPruner(model_copy, device)
                pruned = pruner.prune_layer('fc1', RATIO)

            elif method == 'l2_norm':
                pruner = L2NormPruner(model_copy, device)
                pruned = pruner.prune_layer('fc1', RATIO)

            elif method == 'snip':
                pruner = SNIPPruner(model_copy, device)
                pruned = pruner.prune_layer('fc1', RATIO, train_loader)

            elif method == 'grasp':
                pruner = GraSPPruner(model_copy, device)
                pruned = pruner.prune_layer('fc1', RATIO, train_loader)

            elif method == 'hybrid':
                pruner = HybridPruner(model_copy, device)
                pruned = pruner.prune_layer('fc1', RATIO, train_loader)

            # ── Fine-tune ─────────────────────────────────────────────────────
            set_seed(SEED + 9999)
            acc = train_model(pruned, train_loader, test_loader, device, epochs=5, lr=1e-4)
            results[method] = acc
            print(f"         -> {acc:.2f}%", flush=True)

        except Exception as e:
            import traceback
            print(f"  *** FAILED: {e} ***", flush=True)
            traceback.print_exc()
            results[method] = None
            print("  Continuing…", flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("MISSING RUNS COMPLETE — seed=1024, 40% sparsity, FashionMNIST")
    print("=" * 65)
    print(f"{'Method':<12}  {'Accuracy':>10}")
    print("-" * 28)
    for m in MISSING_METHODS:
        val = results.get(m)
        print(f"{m:<12}  {f'{val:.2f}%' if val else 'FAILED':>10}")
    print("=" * 65)

    # ── Full updated table with previously collected results ──────────────────
    print("\n── UPDATED COMPLETE TABLE (seed=1024, all ratios) ──────────────")
    prev = {
        '20%': {'spectral':88.13,'l1_norm':88.41,'l2_norm':88.33,
                'snip':88.29,'grasp':88.02,'hybrid':88.31},
        '30%': {'spectral':88.04,'l1_norm':88.27,'l2_norm':88.25,
                'snip':88.07,'grasp':87.91,'hybrid':88.08},
        '40%': {'spectral':87.53},  # already collected
    }
    for m in MISSING_METHODS:
        if results.get(m) is not None:
            prev['40%'][m] = results[m]

    methods_all = ['spectral','l1_norm','l2_norm','snip','grasp','hybrid']
    print(f"{'Method':<12}  {'20%':>8}  {'30%':>8}  {'40%':>8}")
    print("-" * 44)
    for m in methods_all:
        r20 = prev['20%'].get(m, '—')
        r30 = prev['30%'].get(m, '—')
        r40 = prev['40%'].get(m, '—')
        fmt = lambda v: f"{v:.2f}%" if isinstance(v, float) else str(v)
        print(f"{m:<12}  {fmt(r20):>8}  {fmt(r30):>8}  {fmt(r40):>8}")
    print("=" * 65, flush=True)


if __name__ == '__main__':
    main()
