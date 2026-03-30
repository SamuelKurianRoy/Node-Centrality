"""
statistical_rigor.py
====================
Runs spectral, l1_norm, l2_norm, snip, grasp, hybrid across N seeds.
Outputs mean ± std table and t-tests (each method vs L2) per sparsity level.

Run:
  !python statistical_rigor.py | tee /content/drive/MyDrive/SpectralPruning_Results/statistical_rigor.txt
"""

import torch
import torch.nn as nn
import numpy as np
import random
from scipy import stats
from torch.utils.data import DataLoader
from pruning_logic import (
    SimpleMLP, load_dataset, train_model,
    SpectralPruner, L1NormPruner, L2NormPruner,
    SNIPPruner, GraSPPruner, HybridPruner
)

# =============================================================================
# CONFIG
# =============================================================================
DATASET      = 'FashionMNIST'
PRUNE_RATIOS = [0.2, 0.3, 0.4]
METHODS      = ['spectral', 'l1_norm', 'l2_norm', 'snip', 'grasp', 'hybrid']
N_SEEDS      = 5
SEEDS        = [42, 123, 456, 789, 1024]
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# SEED CONTROL
# =============================================================================
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# =============================================================================
# SINGLE EXPERIMENT
# =============================================================================
def run_one(method, prune_ratio, seed, train_loader, test_loader, input_size):
    set_seed(seed)

    # Train baseline
    model = SimpleMLP(input_size=input_size, hidden_size=256, num_classes=10)
    train_model(model, train_loader, test_loader, DEVICE, epochs=10, lr=1e-3)

    # Prune
    if method == 'spectral':
        pruner = SpectralPruner(model, DEVICE)
        pruned = pruner.prune_layer('fc1', prune_ratio, train_loader)
    elif method == 'l1_norm':
        pruner = L1NormPruner(model, DEVICE)
        pruned = pruner.prune_layer('fc1', prune_ratio)
    elif method == 'l2_norm':
        pruner = L2NormPruner(model, DEVICE)
        pruned = pruner.prune_layer('fc1', prune_ratio)
    elif method == 'snip':
        pruner = SNIPPruner(model, DEVICE)
        pruned = pruner.prune_layer('fc1', prune_ratio, train_loader)
    elif method == 'grasp':
        pruner = GraSPPruner(model, DEVICE)
        pruned = pruner.prune_layer('fc1', prune_ratio, train_loader)
    elif method == 'hybrid':
        pruner = HybridPruner(model, DEVICE)
        pruned = pruner.prune_layer('fc1', prune_ratio, train_loader)

    # Fine-tune
    set_seed(seed + 9999)   # different seed for fine-tuning shuffle
    acc = train_model(pruned, train_loader, test_loader, DEVICE,
                      epochs=5, lr=1e-4)
    return acc

# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print(f"\nDevice: {DEVICE}")
    print(f"Dataset: {DATASET} | Seeds: {SEEDS} | Methods: {METHODS}")
    print(f"Prune ratios: {[f'{r*100:.0f}%' for r in PRUNE_RATIOS]}\n")

    train_dataset, test_dataset, input_size, num_classes = load_dataset(DATASET)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=1000, shuffle=False,
                              num_workers=2, pin_memory=True)

    # results[method][prune_ratio] = [acc_seed1, acc_seed2, ...]
    results = {m: {r: [] for r in PRUNE_RATIOS} for m in METHODS}

    total = len(METHODS) * len(PRUNE_RATIOS) * N_SEEDS
    done  = 0

    for seed in SEEDS:
        for prune_ratio in PRUNE_RATIOS:
            for method in METHODS:
                done += 1
                print(f"[{done}/{total}] seed={seed} | {prune_ratio*100:.0f}% | {method}")
                acc = run_one(method, prune_ratio, seed,
                              train_loader, test_loader, input_size)
                results[method][prune_ratio].append(acc)
                print(f"         -> {acc:.2f}%")

    # =========================================================================
    # RESULTS TABLE — mean ± std
    # =========================================================================
    print(f"\n{'='*80}")
    print(f" STATISTICAL RESULTS — {DATASET}, {N_SEEDS} seeds")
    print(f"{'='*80}")
    print(f"{'Method':<12} | {'20% (mean±std)':>18} | {'30% (mean±std)':>18} | {'40% (mean±std)':>18}")
    print(f"{'-'*75}")

    stats_table = {}
    for method in METHODS:
        row = []
        stats_table[method] = {}
        for prune_ratio in PRUNE_RATIOS:
            vals = results[method][prune_ratio]
            mean = np.mean(vals)
            std  = np.std(vals)
            stats_table[method][prune_ratio] = vals
            row.append(f"{mean:.2f}±{std:.2f}%")
        print(f"{method:<12} | {row[0]:>18} | {row[1]:>18} | {row[2]:>18}")

    # =========================================================================
    # T-TESTS vs L2 NORM
    # =========================================================================
    print(f"\n{'='*80}")
    print(f" T-TESTS vs L2 NORM (two-sided, paired)")
    print(f"{'='*80}")
    print(f"{'Method':<12} | {'20% p-value':>14} {'sig':>4} | {'30% p-value':>14} {'sig':>4} | {'40% p-value':>14} {'sig':>4}")
    print(f"{'-'*75}")

    def sig_stars(p):
        if p < 0.001: return '***'
        if p < 0.01:  return '** '
        if p < 0.05:  return '*  '
        return 'ns '

    l2_vals = stats_table['l2_norm']

    for method in METHODS:
        if method == 'l2_norm':
            continue
        row = []
        for prune_ratio in PRUNE_RATIOS:
            t_stat, p_val = stats.ttest_rel(
                stats_table[method][prune_ratio],
                l2_vals[prune_ratio]
            )
            row.append(f"{p_val:.4f} {sig_stars(p_val)}")
        print(f"{method:<12} | {row[0]:>18} | {row[1]:>18} | {row[2]:>18}")

    print(f"\n  Significance: *** p<0.001  ** p<0.01  * p<0.05  ns = not significant")

    # =========================================================================
    # RAW VALUES (for your records)
    # =========================================================================
    print(f"\n{'='*80}")
    print(f" RAW ACCURACY VALUES PER SEED")
    print(f"{'='*80}")
    for method in METHODS:
        for prune_ratio in PRUNE_RATIOS:
            vals = results[method][prune_ratio]
            print(f"  {method:<12} {prune_ratio*100:.0f}%: {[f'{v:.2f}' for v in vals]}")

    print("\nDone.")