"""
ResNet-20 CIFAR-10 Pruning Experiment — AEI  [DGX-OPTIMISED]
=============================================================
Optimisations over the baseline version:
  - Automatic Mixed Precision (AMP) throughout training & fine-tuning
  - torch.compile for ResNet-20 (PyTorch 2.x)
  - Multi-GPU experiment parallelism: each GPU runs an independent
    (method, sparsity, seed) triplet simultaneously via torch.multiprocessing
  - cudnn.benchmark = True (safe because input shape is fixed)
  - DataLoader tuned for DGX CPU topology (128 cores, NVMe datasets)
  - Persistent workers + prefetch_factor to hide I/O latency
  - AEI computed once on GPU 0, broadcast to all workers
  - Random-trial distribution parallelised across all GPUs

Expected walltime on DGX A100 (8×80 GB):
  Training:            ~3 min   (single GPU, AMP + compile)
  175 pruning runs:    ~25 min  (8-GPU parallel, AMP fine-tune)
  1000 random trials:  ~15 min  (8-GPU parallel)
  Total:               ~45 min  vs ~3 hrs on single GPU

Usage:
  # Single-node, all 8 GPUs:
  python resnet20_aei_pruning_dgx.py

  # Restrict to GPUs 0-3:
  CUDA_VISIBLE_DEVICES=0,1,2,3 python resnet20_aei_pruning_dgx.py
"""

import copy
import json
import math
import os
import random
import time
from pathlib import Path

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

# ─────────────────────────── Configuration ────────────────────────────────────

SEED            = 42
NUM_SEEDS       = 5
SPARSITY_LEVELS = [0.2, 0.3, 0.4, 0.5, 0.6]
FINETUNE_EPOCHS = 10
TRAIN_EPOCHS    = 100
BATCH_SIZE      = 512       # larger batch benefits A100 tensor cores
FINETUNE_BS     = 256       # per-GPU batch during fine-tune
LR              = 0.1 * (BATCH_SIZE / 128)  # linear LR scaling
DATA_ROOT       = "./data"
OUT_DIR         = Path("./resnet20_outputs")
OUT_DIR.mkdir(exist_ok=True)

N_CALIB         = 1000
N_RANDOM_TRIALS = 1000

# DGX DataLoader settings
NUM_WORKERS     = min(16, os.cpu_count() // max(1,
                       torch.cuda.device_count()))  # per-GPU workers
PREFETCH        = 4          # prefetch_factor for DataLoader
PIN_MEMORY      = True

# ─────────────────────────── Reproducibility ──────────────────────────────────

def set_seed(seed: int, deterministic: bool = False):
    """
    deterministic=False: allows cudnn.benchmark (faster, non-deterministic).
    deterministic=True:  fully reproducible but slower.
    Set deterministic=True only if bit-exact reproducibility is required.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark     = not deterministic
    torch.backends.cudnn.deterministic = deterministic

set_seed(SEED, deterministic=False)   # benchmark=True on DGX

# ─────────────────────────── Data ─────────────────────────────────────────────

def get_cifar10_loaders(batch_size: int = BATCH_SIZE,
                         gpu_id: int = 0):
    """
    CIFAR-10 loaders tuned for DGX.
    persistent_workers=True avoids fork overhead between epochs.
    prefetch_factor pre-loads batches into pinned memory.
    """
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

    loader_kw = dict(
        num_workers        = NUM_WORKERS,
        pin_memory         = PIN_MEMORY,
        persistent_workers = True,
        prefetch_factor    = PREFETCH,
    )

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, **loader_kw)
    test_loader  = DataLoader(test_set,  batch_size=512,
                              shuffle=False, **loader_kw)

    calib_idx    = list(range(N_CALIB))
    calib_loader = DataLoader(Subset(train_set, calib_idx),
                              batch_size=512, shuffle=False, **loader_kw)
    return train_loader, test_loader, calib_loader

# ─────────────────────────── ResNet-20 ────────────────────────────────────────

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet20(nn.Module):
    def __init__(self, num_classes: int = 10):
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
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


def build_model(device, compile_model: bool = True):
    """
    Instantiate ResNet-20.
    torch.compile (PyTorch 2.x) fuses ops and generates optimised CUDA kernels.
    Use compile_model=False for worker processes to avoid re-compilation overhead.
    """
    model = ResNet20().to(device)
    if compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)
    return model

# ─────────────────────────── Training & Evaluation ────────────────────────────

def train_epoch_amp(model, loader, optimizer, scaler, scheduler=None,
                    device=None):
    """Single training epoch with AMP (float16 forward, float32 param update)."""
    model.train()
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)   # faster than zero_grad()
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


def train_model(model, train_loader, test_loader,
                epochs=TRAIN_EPOCHS, lr=LR, device=None):
    """Train from scratch — cosine schedule + AMP + gradient clipping."""
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = torch.cuda.amp.GradScaler()

    best_acc, best_state = 0.0, None
    for epoch in range(1, epochs + 1):
        train_epoch_amp(model, train_loader, optimizer, scaler, scheduler, device)
        acc = evaluate(model, test_loader, device)
        if acc > best_acc:
            best_acc   = acc
            best_state = copy.deepcopy(model.state_dict())
        if epoch % 20 == 0:
            print(f"    Epoch {epoch:3d}/{epochs} — acc {acc:.2f}%")

    model.load_state_dict(best_state)
    return best_acc


def finetune(model, train_loader, test_loader,
             epochs=FINETUNE_EPOCHS, lr=1e-3, device=None):
    """Short fine-tune after pruning — Adam + AMP."""
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = torch.cuda.amp.GradScaler()

    best_acc = 0.0
    for _ in range(epochs):
        train_epoch_amp(model, train_loader, optimizer, scaler, scheduler, device)
        acc = evaluate(model, test_loader, device)
        best_acc = max(best_acc, acc)
    return best_acc

# ─────────────────────────── Prunable Filter Registry ─────────────────────────

def get_prunable_convs(model: nn.Module):
    """
    Returns (name, module) pairs safe to filter-prune.
    conv2 in blocks with non-trivial shortcuts is excluded (skip connection
    alignment constraint).
    Works whether or not the model is torch.compiled (accesses ._orig_mod).
    """
    # torch.compile wraps the model; unwrap if needed
    base = getattr(model, "_orig_mod", model)
    prunable = [("conv1", base.conv1)]

    for layer_name in ["layer1", "layer2", "layer3"]:
        layer = getattr(base, layer_name)
        for block_idx, block in enumerate(layer):
            prunable.append(
                (f"{layer_name}.{block_idx}.conv1", block.conv1)
            )
            has_shortcut = len(list(block.shortcut.children())) > 0
            if not has_shortcut:
                prunable.append(
                    (f"{layer_name}.{block_idx}.conv2", block.conv2)
                )
    return prunable

# ─────────────────────────── Activation Collection ────────────────────────────

@torch.no_grad()
def collect_filter_activations(model, calib_loader, prunable_convs, device):
    """
    Collect per-filter activations via spatial GAP.
    Returns dict {name: Tensor[N, C_out]} on CPU.
    """
    model.eval()
    buffers = {name: [] for name, _ in prunable_convs}

    def make_hook(name):
        def hook(_, __, out):
            # out: [B, C, H, W]  →  GAP  →  [B, C]
            buffers[name].append(out.detach().mean(dim=[2, 3]).cpu())
        return hook

    handles = [conv.register_forward_hook(make_hook(name))
               for name, conv in prunable_convs]

    for x, _ in calib_loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            model(x.to(device, non_blocking=True))

    for h in handles:
        h.remove()

    return {name: torch.cat(bufs, dim=0) for name, bufs in buffers.items()}

# ─────────────────────────── AEI Computation ──────────────────────────────────

def pearson_correlation_matrix(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    X = X - X.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(X, axis=0, keepdims=True) + eps
    X /= norms
    C = np.abs(X.T @ X)
    np.fill_diagonal(C, 0.0)
    return C


def compute_aei(corr_matrix: np.ndarray) -> np.ndarray:
    """Fiedler-vector based AEI scores. Low R → structurally peripheral."""
    A = np.maximum(corr_matrix, 0.0)
    np.fill_diagonal(A, 0.0)
    D = np.diag(A.sum(axis=1))
    L = D - A
    n = L.shape[0]
    if n < 3:
        return np.zeros(n)
    k = min(2, n - 1)
    _, eigvecs = eigh(L, subset_by_index=[0, k])
    v2 = eigvecs[:, 1] if k >= 1 else np.zeros(n)
    R  = np.array([np.sum(A[i] * np.abs(v2[i] - v2)) for i in range(n)])
    return R


def compute_all_aei_scores(activations: dict) -> dict:
    return {
        name: compute_aei(
            pearson_correlation_matrix(acts.numpy().astype(np.float64))
        )
        for name, acts in activations.items()
        if acts.shape[1] >= 3
    }

# ─────────────────────────── Pruning Methods ──────────────────────────────────

def _minmax(arr):
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-10)


def global_filter_scores(model, prunable_convs, score_fn):
    entries = []
    for name, conv in prunable_convs:
        for i, s in enumerate(score_fn(name, conv)):
            entries.append((name, i, float(s)))
    return entries


def apply_global_filter_prune(model, entries, sparsity: float):
    n_prune   = int(len(entries) * sparsity)
    to_prune  = sorted(entries, key=lambda x: x[2])[:n_prune]
    prune_map = {}
    for name, fi, _ in to_prune:
        prune_map.setdefault(name, set()).add(fi)

    base = getattr(model, "_orig_mod", model)
    for name, module in base.named_modules():
        if isinstance(module, nn.Conv2d) and name in prune_map:
            with torch.no_grad():
                for fi in prune_map[name]:
                    module.weight[fi].zero_()


# ── Score functions ────────────────────────────────────────────────────────────

def l1_score(name, conv):
    return np.abs(conv.weight.detach().cpu().numpy()).sum(axis=(1, 2, 3))

def l2_score(name, conv):
    w = conv.weight.detach().cpu().numpy()
    return np.sqrt((w ** 2).sum(axis=(1, 2, 3)))

def random_score(name, conv):
    return np.random.rand(conv.weight.shape[0])

def snip_score_fn(model, prunable, calib_loader, device):
    model.train()
    x, y = next(iter(calib_loader))
    x, y = x.to(device), y.to(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = F.cross_entropy(model(x), y)
    loss.backward()
    scores = {}
    for name, conv in prunable:
        w = conv.weight.detach().cpu().numpy()
        g = (conv.weight.grad.cpu().numpy()
             if conv.weight.grad is not None else np.zeros_like(w))
        scores[name] = np.abs(g * w).sum(axis=(1, 2, 3))
    for p in model.parameters():
        p.grad = None
    return lambda n, c: scores[n]

def grasp_score_fn(model, prunable, calib_loader, device):
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    x, y   = next(iter(calib_loader))
    x, y   = x.to(device), y.to(device)
    loss   = F.cross_entropy(model(x), y)
    grads  = torch.autograd.grad(loss, params, create_graph=True)
    gnorm  = sum((g * g.detach()).sum() for g in grads)
    hg     = torch.autograd.grad(gnorm, params)
    pid_hg = {id(p): hgv.detach().cpu().numpy() for p, hgv in zip(params, hg)}
    scores = {}
    for name, conv in prunable:
        hgv = pid_hg.get(id(conv.weight),
                         np.zeros_like(conv.weight.detach().cpu().numpy()))
        w   = conv.weight.detach().cpu().numpy()
        scores[name] = (hgv * w).sum(axis=(1, 2, 3))  # low → keep
    for p in model.parameters():
        p.grad = None
    return lambda n, c: scores[n]

def aei_score(aei_scores, name, conv):
    return aei_scores.get(name, np.ones(conv.weight.shape[0]))

def hybrid_score(aei_scores, name, conv):
    R   = _minmax(aei_scores.get(name, np.ones(conv.weight.shape[0])))
    l2  = _minmax(l2_score(name, conv))
    return R * l2

# ─────────────────────────── Spectral Overlap ─────────────────────────────────

def avg_r_percentile_of_pruned(pruned_entries, aei_scores, prunable_convs):
    r_flat, r_lut = [], {}
    for name, conv in prunable_convs:
        R = aei_scores.get(name, np.zeros(conv.weight.shape[0]))
        for i, r in enumerate(R):
            r_lut[(name, i)] = len(r_flat)
            r_flat.append(r)
    r_flat = np.array(r_flat)
    r_vals = [r_flat[r_lut[(n, fi)]]
              for n, fi, _ in pruned_entries if (n, fi) in r_lut]
    if not r_vals:
        return 50.0
    return float(np.mean([100 * (r_flat < rv).mean() for rv in r_vals]))

# ─────────────────────────── Worker Function (per GPU) ────────────────────────

def worker_run(gpu_id: int,
               task_queue: mp.Queue,
               result_queue: mp.Queue,
               pretrained_state: dict,
               aei_scores_shared: dict,
               aei_time: float):
    """
    Runs on a single GPU. Pulls (method, sparsity, seed, task_id) from
    task_queue, executes the pruning experiment, pushes results to result_queue.

    This function is the target of mp.spawn / mp.Process.
    """
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    set_seed(SEED + gpu_id, deterministic=False)

    # Build per-worker data loaders (each worker needs its own loader)
    train_loader, test_loader, calib_loader = get_cifar10_loaders(
        batch_size=FINETUNE_BS, gpu_id=gpu_id
    )

    while True:
        item = task_queue.get()
        if item is None:              # sentinel — exit
            break

        method, sparsity, seed, task_id = item
        set_seed(seed, deterministic=False)

        # Load model on this GPU (no torch.compile in worker to avoid
        # redundant compilation across 8 processes)
        model = ResNet20().to(device)
        model.load_state_dict(copy.deepcopy(pretrained_state))
        prunable = get_prunable_convs(model)

        # Score
        t_start = time.time()
        if method == "L1":
            sfn = lambda n, c: l1_score(n, c)
        elif method == "L2":
            sfn = lambda n, c: l2_score(n, c)
        elif method == "Random":
            sfn = lambda n, c: random_score(n, c)
        elif method == "SNIP":
            sfn = snip_score_fn(model, prunable, calib_loader, device)
        elif method == "GraSP":
            sfn = grasp_score_fn(model, prunable, calib_loader, device)
        elif method == "AEI":
            sfn = lambda n, c: aei_score(aei_scores_shared, n, c)
        elif method == "Hybrid":
            sfn = lambda n, c: hybrid_score(aei_scores_shared, n, c)
        else:
            raise ValueError(method)

        entries  = global_filter_scores(model, prunable, sfn)
        overhead = time.time() - t_start
        if method in ("AEI", "Hybrid"):
            overhead += aei_time

        # Prune
        n_prune        = int(len(entries) * sparsity)
        pruned_entries = sorted(entries, key=lambda x: x[2])[:n_prune]
        apply_global_filter_prune(model, entries, sparsity)

        # Spectral overlap
        avg_r = avg_r_percentile_of_pruned(pruned_entries,
                                            aei_scores_shared, prunable)

        # Fine-tune
        acc = finetune(model, train_loader, test_loader,
                       epochs=FINETUNE_EPOCHS, device=device)

        result_queue.put({
            "task_id":   task_id,
            "method":    method,
            "sparsity":  sparsity,
            "seed":      seed,
            "accuracy":  acc,
            "avg_r_pct": avg_r,
            "overhead_s": overhead,
        })
        del model
        torch.cuda.empty_cache()


def random_worker(gpu_id, n_trials, pretrained_state, result_queue):
    """Random-pruning distribution worker at 40% sparsity."""
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    train_loader, test_loader, _ = get_cifar10_loaders(
        batch_size=FINETUNE_BS, gpu_id=gpu_id
    )
    for _ in range(n_trials):
        model = ResNet20().to(device)
        model.load_state_dict(copy.deepcopy(pretrained_state))
        prunable = get_prunable_convs(model)
        entries  = global_filter_scores(model, prunable, random_score)
        apply_global_filter_prune(model, entries, 0.4)
        acc = finetune(model, train_loader, test_loader,
                       epochs=FINETUNE_EPOCHS, device=device)
        result_queue.put(acc)
        del model
        torch.cuda.empty_cache()

# ─────────────────────────── Main ─────────────────────────────────────────────

def main():
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        raise RuntimeError("No CUDA GPUs found. This script requires at least 1 GPU.")
    print(f"Found {n_gpus} GPU(s): "
          + ", ".join(torch.cuda.get_device_name(i) for i in range(n_gpus)))
    print(f"PyTorch {torch.__version__} | "
          f"CUDA {torch.version.cuda} | "
          f"compile={'available' if hasattr(torch,'compile') else 'N/A'}")
    print("-" * 70)

    device0 = torch.device("cuda:0")

    # ── 1. Train baseline on GPU 0 ────────────────────────────────────────────
    print("\n[1/4] Training baseline ResNet-20...")
    train_loader, test_loader, calib_loader = get_cifar10_loaders(
        batch_size=BATCH_SIZE, gpu_id=0
    )
    model = build_model(device0, compile_model=True)
    t0    = time.time()
    base_acc = train_model(model, train_loader, test_loader,
                           epochs=TRAIN_EPOCHS, lr=LR, device=device0)
    print(f"    Baseline acc: {base_acc:.2f}%  "
          f"[{time.time()-t0:.0f}s]")
    pretrained_state = copy.deepcopy(
        getattr(model, "_orig_mod", model).state_dict()
    )

    # ── 2. Compute AEI once on GPU 0 ─────────────────────────────────────────
    print("\n[2/4] Computing AEI scores (one-time calibration pass)...")
    # Use the non-compiled base model for hook registration
    base_model = ResNet20().to(device0)
    base_model.load_state_dict(pretrained_state)
    prunable_ref = get_prunable_convs(base_model)

    t0 = time.time()
    acts = collect_filter_activations(base_model, calib_loader,
                                      prunable_ref, device0)
    aei_scores = compute_all_aei_scores(acts)
    aei_time   = time.time() - t0
    print(f"    AEI computation: {aei_time*1000:.0f} ms")
    del base_model

    # ── 3. Build task list & dispatch across GPUs ─────────────────────────────
    methods = ["L1", "L2", "SNIP", "GraSP", "Random", "AEI", "Hybrid"]
    tasks   = []
    for method in methods:
        for sparsity in SPARSITY_LEVELS:
            for seed_offset in range(NUM_SEEDS):
                tasks.append((method, sparsity,
                              SEED + seed_offset * 100, len(tasks)))
    n_tasks = len(tasks)
    print(f"\n[3/4] Dispatching {n_tasks} tasks across {n_gpus} GPU(s)...")

    # Use multiprocessing spawn context for CUDA safety
    mp.set_start_method("spawn", force=True)
    task_q   = mp.Queue()
    result_q = mp.Queue()

    for task in tasks:
        task_q.put(task)
    for _ in range(n_gpus):
        task_q.put(None)   # one sentinel per worker

    processes = []
    for gpu_id in range(n_gpus):
        p = mp.Process(
            target=worker_run,
            args=(gpu_id, task_q, result_q,
                  pretrained_state, aei_scores, aei_time),
        )
        p.start()
        processes.append(p)

    # Collect results
    raw_results = {}
    completed   = 0
    t_start     = time.time()
    while completed < n_tasks:
        res = result_q.get()
        completed += 1
        key = (res["method"], res["sparsity"])
        raw_results.setdefault(key, []).append(res)
        elapsed = time.time() - t_start
        eta     = elapsed / completed * (n_tasks - completed)
        print(f"  [{completed:3d}/{n_tasks}] "
              f"{res['method']:6s} @ {res['sparsity']:.0%}  "
              f"acc={res['accuracy']:.2f}%  R%={res['avg_r_pct']:.1f}  "
              f"ETA {eta/60:.1f} min")

    for p in processes:
        p.join()

    # Aggregate
    results = {}
    for method in methods:
        results[method] = {}
        for sparsity in SPARSITY_LEVELS:
            items = raw_results.get((method, sparsity), [])
            accs  = [x["accuracy"]  for x in items]
            rpcts = [x["avg_r_pct"] for x in items]
            results[method][sparsity] = {
                "acc_mean":   float(np.mean(accs)),
                "acc_std":    float(np.std(accs)),
                "r_pct_mean": float(np.mean(rpcts)),
                "r_pct_std":  float(np.std(rpcts)),
                "all_accs":   accs,
                "all_r_pcts": rpcts,
            }

    # ── 4. Random distribution (8-GPU parallel) ───────────────────────────────
    print(f"\n[4/4] Running {N_RANDOM_TRIALS} random trials "
          f"(split across {n_gpus} GPUs)...")
    per_gpu  = math.ceil(N_RANDOM_TRIALS / n_gpus)
    rand_q   = mp.Queue()
    rnd_procs = []
    for gpu_id in range(n_gpus):
        n = min(per_gpu, N_RANDOM_TRIALS - gpu_id * per_gpu)
        if n <= 0:
            break
        p = mp.Process(
            target=random_worker,
            args=(gpu_id, n, pretrained_state, rand_q),
        )
        p.start()
        rnd_procs.append(p)

    random_accs = [rand_q.get() for _ in range(N_RANDOM_TRIALS)]
    for p in rnd_procs:
        p.join()

    # ── Print Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("RESULTS — ResNet-20 / CIFAR-10")
    print("=" * 80)
    print(f"Baseline (no pruning): {base_acc:.2f}%\n")

    header = f"{'Method':<8}" + "".join(f"  {s:.0%}" for s in SPARSITY_LEVELS)
    sep    = "-" * len(header)

    print("Accuracy (mean ± std %)")
    print(header); print(sep)
    for m in methods:
        row = f"{m:<8}"
        for s in SPARSITY_LEVELS:
            r = results[m][s]
            row += f"  {r['acc_mean']:.2f}±{r['acc_std']:.2f}"
        print(row)

    print("\nAvg R-Percentile (↓ = more structural = better for AEI/Hybrid)")
    print(header); print(sep)
    for m in methods:
        row = f"{m:<8}"
        for s in SPARSITY_LEVELS:
            r = results[m][s]
            row += f"  {r['r_pct_mean']:.1f}±{r['r_pct_std']:.1f}"
        print(row)

    print(f"\nRandom distribution @ 40% ({N_RANDOM_TRIALS} trials): "
          f"mean={np.mean(random_accs):.2f}% ± {np.std(random_accs):.2f}%")

    # ── LaTeX Tables ──────────────────────────────────────────────────────────
    cols = " & ".join(f"{s:.0%}" for s in SPARSITY_LEVELS)

    def latex_table(title, val_fn, label):
        lines = [
            f"% {title}",
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{" + title + r"}",
            r"\label{" + label + r"}",
            r"\begin{tabular}{l" + "c" * len(SPARSITY_LEVELS) + r"}",
            r"\toprule",
            f"Method & {cols} \\\\",
            r"\midrule",
        ]
        for m in methods:
            row = m
            for s in SPARSITY_LEVELS:
                row += " & " + val_fn(m, s)
            lines.append(row + r" \\")
        lines += [
            r"\midrule",
            f"Baseline & \\multicolumn{{{len(SPARSITY_LEVELS)}}}{{c}}{{{base_acc:.2f}}} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
        return "\n".join(lines)

    acc_table = latex_table(
        "ResNet-20 accuracy (\\%) on CIFAR-10 after pruning and fine-tuning.",
        lambda m, s: (f"{results[m][s]['acc_mean']:.2f}"
                      r"$\pm$"
                      f"{results[m][s]['acc_std']:.2f}"),
        "tab:resnet20_accuracy",
    )
    r_table = latex_table(
        "Average $R$-percentile of pruned filters on ResNet-20/CIFAR-10.",
        lambda m, s: (f"{results[m][s]['r_pct_mean']:.1f}"
                      r"$\pm$"
                      f"{results[m][s]['r_pct_std']:.1f}"),
        "tab:resnet20_spectral",
    )

    print("\n── LaTeX Accuracy Table ────────────────────────────────────────────")
    print(acc_table)
    print("\n── LaTeX Spectral Overlap Table ────────────────────────────────────")
    print(r_table)

    # ── Save JSON ──────────────────────────────────────────────────────────────
    output = {
        "baseline_acc": base_acc,
        "results":      results,
        "random_dist_40pct": {
            "accs": random_accs,
            "mean": float(np.mean(random_accs)),
            "std":  float(np.std(random_accs)),
        },
        "latex": {
            "accuracy_table": acc_table,
            "spectral_table": r_table,
        },
        "config": {
            "architecture":    "ResNet-20",
            "dataset":         "CIFAR-10",
            "sparsity_type":   "global filter pruning (weight masking)",
            "finetune_epochs": FINETUNE_EPOCHS,
            "train_epochs":    TRAIN_EPOCHS,
            "num_seeds":       NUM_SEEDS,
            "n_calib":         N_CALIB,
            "batch_size":      BATCH_SIZE,
            "n_gpus":          n_gpus,
            "amp":             True,
            "compile":         hasattr(torch, "compile"),
        },
    }

    out_path = OUT_DIR / "resnet20_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
