"""
spectral_analysis.py
=====================
Analyses WHERE other pruning methods (L1, L2, SNIP, GraSP, Random) prune
relative to the AEI spectral ranking.

Answers the question:
  "Are L2/SNIP/etc. accidentally picking spectrally peripheral neurons?
   Or are they pruning structurally important ones with small weights?"

Run in Colab:
  !python spectral_analysis.py | tee /content/drive/MyDrive/SpectralPruning_Results/spectral_analysis.txt

No changes to main.py or pruning_logic.py needed.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.linalg import eigh
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for Colab/server
import matplotlib.pyplot as plt
import os

# =============================================================================
# CONFIG
# =============================================================================
DATASET     = 'FashionMNIST'
PRUNE_RATIO = 0.3        # analyse at 30% — most informative sparsity level
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR    = '/content/drive/MyDrive/SpectralPruning_Results'  # change if not Colab

# =============================================================================
# MODEL (same as pruning_logic.py)
# =============================================================================
class SimpleMLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super().__init__()
        self.flatten   = nn.Flatten()
        self.fc1       = nn.Linear(input_size, hidden_size)
        self.relu      = nn.ReLU()
        self.dropout   = nn.Dropout(0.5)
        self.fc2       = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# =============================================================================
# DATASET
# =============================================================================
def load_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    train_ds = datasets.FashionMNIST('./data', train=True,  download=True, transform=transform)
    test_ds  = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=1000, shuffle=False, num_workers=2)
    return train_loader, test_loader

# =============================================================================
# TRAIN
# =============================================================================
def train(model, train_loader, test_loader, epochs=10, lr=1e-3):
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            nn.CrossEntropyLoss()(model(data), target).backward()
            optimizer.step()
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            _, pred = torch.max(model(data), 1)
            total   += target.size(0)
            correct += (pred == target).sum().item()
    return 100 * correct / total

# =============================================================================
# SPECTRAL: compute R values for all 256 neurons
# =============================================================================
def compute_all_R_values(model, train_loader, num_batches=20):
    """
    Returns R_values (np.array, shape [256]) — AEI score per neuron.
    Higher R = more central. Lower R = more peripheral = pruned by AEI.
    """
    activation_storage = {}

    def hook(module, inp, out):
        activation_storage['fc1'] = out.detach()

    handle = model.fc1.register_forward_hook(hook)
    model.eval()
    acts = []
    with torch.no_grad():
        for i, (data, _) in enumerate(train_loader):
            if i >= num_batches:
                break
            _ = model(data.to(DEVICE))
            if 'fc1' in activation_storage:
                acts.append(activation_storage['fc1'])
    handle.remove()

    acts_np = torch.cat(acts, dim=0).cpu().numpy()  # (N_samples, 256)

    # Pearson correlation -> adjacency
    corr    = np.nan_to_num(np.corrcoef(acts_np.T), nan=0.0)
    adj     = np.abs(corr)

    # Normalised Laplacian -> Fiedler vector
    deg          = np.maximum(adj.sum(axis=1), 1e-8)
    D_inv_sqrt   = np.diag(1.0 / np.sqrt(deg))
    L            = np.eye(len(deg)) - D_inv_sqrt @ adj @ D_inv_sqrt
    _, vecs      = eigh(L)
    v2           = vecs[:, 1]   # Fiedler vector

    # R_i = sum_j adj_ij * |v2_i - v2_j|
    diff    = np.abs(v2[:, None] - v2[None, :])
    R       = np.sum(adj * diff, axis=1)

    return R, v2, adj

# =============================================================================
# GET PRUNE INDICES FOR EACH METHOD
# =============================================================================
def get_pruned_indices(model, method, prune_ratio, train_loader):
    """
    Returns the SET of neuron indices each method would prune.
    Does NOT modify the model.
    """
    n        = model.fc1.out_features   # 256
    num_prune = int(n * prune_ratio)

    if method == 'spectral':
        R, _, _ = compute_all_R_values(model, train_loader)
        # Prune lowest R (most peripheral)
        pruned = set(np.argsort(R)[:num_prune].tolist())

    elif method == 'l1_norm':
        w      = model.fc1.weight.data.cpu().numpy()
        scores = np.sum(np.abs(w), axis=1)
        pruned = set(np.argsort(scores)[:num_prune].tolist())

    elif method == 'l2_norm':
        w      = model.fc1.weight.data.cpu().numpy()
        scores = np.sqrt(np.sum(w ** 2, axis=1))
        pruned = set(np.argsort(scores)[:num_prune].tolist())

    elif method == 'snip':
        criterion = nn.CrossEntropyLoss()
        model.train()
        model.zero_grad()
        for i, (data, target) in enumerate(train_loader):
            if i >= 1:
                break
            criterion(model(data.to(DEVICE)), target.to(DEVICE)).backward()
        grad    = model.fc1.weight.grad.cpu().numpy()
        weights = model.fc1.weight.data.cpu().numpy()
        scores  = np.sum(np.abs(grad * weights), axis=1)
        model.zero_grad()
        pruned  = set(np.argsort(scores)[:num_prune].tolist())

    elif method == 'grasp':
        temperature = 200.0
        criterion   = nn.CrossEntropyLoss()

        # Pass 1
        model.train()
        model.zero_grad()
        for i, (data, target) in enumerate(train_loader):
            if i >= 1:
                break
            (criterion(model(data.to(DEVICE)) / temperature, target.to(DEVICE))).backward()
        grad_dict = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

        # Pass 2
        model.zero_grad()
        for i, (data, target) in enumerate(train_loader):
            if i >= 1:
                break
            output = model(data.to(DEVICE)) / temperature
            loss   = criterion(output, target.to(DEVICE))
            grads  = torch.autograd.grad(loss, model.parameters(), create_graph=True)
            g_dot  = sum((grad_dict[nm] * g).sum()
                         for (nm, _), g in zip(model.named_parameters(), grads)
                         if nm in grad_dict)
            g_dot.backward()

        Hg     = model.fc1.weight.grad.cpu().numpy()
        w      = model.fc1.weight.data.cpu().numpy()
        scores = np.sum(-Hg * w, axis=1)
        model.zero_grad()
        pruned = set(np.argsort(scores)[:num_prune].tolist())

    elif method == 'random':
        rng    = np.random.RandomState(42)
        pruned = set(rng.choice(n, num_prune, replace=False).tolist())

    elif method == 'hybrid':
        R_h, _, _ = compute_all_R_values(model, train_loader)
        w = model.fc1.weight.data.cpu().numpy()
        l2 = np.sqrt(np.sum(w ** 2, axis=1))
        def minmax(v):
            return (v - v.min()) / (v.max() - v.min() + 1e-8)
        score = minmax(R_h) * minmax(l2)
        pruned = set(np.argsort(score)[:num_prune].tolist())


    else:
        raise ValueError(f"Unknown method: {method}")

    return pruned

# =============================================================================
# ANALYSIS
# =============================================================================
def analyse(model, train_loader, prune_ratio):
    n         = model.fc1.out_features
    num_prune = int(n * prune_ratio)

    print(f"\nComputing AEI R values for all {n} neurons...")
    R, v2, adj = compute_all_R_values(model, train_loader)

    # AEI prunes the lowest R neurons
    aei_pruned = set(np.argsort(R)[:num_prune].tolist())

    # R value percentile for each neuron (0 = most peripheral, 100 = most central)
    R_percentile = (np.argsort(np.argsort(R)) / (n - 1)) * 100

    methods = ['l1_norm', 'l2_norm', 'snip', 'grasp', 'random', 'hybrid']

    print(f"\n{'='*70}")
    print(f"  SPECTRAL ANALYSIS — {DATASET}, {prune_ratio*100:.0f}% sparsity")
    print(f"  AEI prunes {num_prune} neurons (lowest R = most peripheral)")
    print(f"{'='*70}")
    print(f"\n{'Method':<12} | {'Overlap w/ AEI':>16} | {'Avg R-pctile of pruned':>23} | {'Interpretation'}")
    print(f"{'-'*90}")

    results = {}
    for method in methods:
        print(f"  Computing {method}...")
        pruned = get_pruned_indices(model, method, prune_ratio, train_loader)

        overlap      = len(aei_pruned & pruned)
        overlap_pct  = overlap / num_prune * 100
        avg_R_pctile = np.mean([R_percentile[i] for i in pruned])

        # Interpretation
        if avg_R_pctile < 35:
            interp = "Pruning peripheral neurons (like AEI)"
        elif avg_R_pctile < 50:
            interp = "Slight bias toward peripheral"
        elif avg_R_pctile < 65:
            interp = "Pruning mid-range neurons"
        else:
            interp = "Pruning central neurons (opposite to AEI)"

        results[method] = {
            'pruned':       pruned,
            'overlap':      overlap,
            'overlap_pct':  overlap_pct,
            'avg_R_pctile': avg_R_pctile,
        }

        print(f"  {method:<12} | {overlap:>5} / {num_prune} ({overlap_pct:>5.1f}%) | "
              f"avg R-pctile = {avg_R_pctile:>6.1f}       | {interp}")

    print(f"\n  (Random expected overlap: {prune_ratio*100:.1f}% = {num_prune*prune_ratio:.1f} neurons)")
    print(f"{'='*70}\n")

    return R, R_percentile, aei_pruned, results

# =============================================================================
# PLOT
# =============================================================================
def plot_spectral_distribution(R, R_percentile, aei_pruned, results, prune_ratio, save_dir):
    """
    One plot per method: histogram of R values for ALL neurons,
    with pruned neurons highlighted in red.
    """
    methods = ['l1_norm', 'l2_norm', 'snip', 'grasp', 'random', 'hybrid']
    n       = len(R)

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle(f'Where each method prunes on the spectral (AEI R value) spectrum\n'
                 f'{DATASET}, {prune_ratio*100:.0f}% sparsity', fontsize=13)

    all_axes = axes.flatten()

    # First subplot: AEI itself
    ax = all_axes[0]
    ax.hist(R, bins=30, color='lightgrey', edgecolor='white', label='kept')
    aei_R = [R[i] for i in aei_pruned]
    ax.hist(aei_R, bins=30, color='steelblue', alpha=0.8, label='pruned')
    ax.set_title('Spectral (AEI) — proposed method', fontweight='bold')
    ax.set_xlabel('R value (AEI score)')
    ax.set_ylabel('Neuron count')
    ax.legend(fontsize=8)

    for idx, method in enumerate(methods):
        ax     = all_axes[idx + 1]
        pruned = results[method]['pruned']
        olap   = results[method]['overlap_pct']
        avg_rp = results[method]['avg_R_pctile']

        pruned_R = [R[i] for i in pruned]
        kept_R   = [R[i] for i in range(n) if i not in pruned]

        ax.hist(kept_R,   bins=30, color='lightgrey', edgecolor='white', label='kept')
        ax.hist(pruned_R, bins=30, color='crimson',   alpha=0.7, label='pruned')
        ax.set_title(f'{method}\noverlap w/ AEI: {olap:.1f}%  |  avg R-pctile: {avg_rp:.1f}',
                     fontsize=9)
        ax.set_xlabel('R value (AEI score)')
        ax.legend(fontsize=8)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f'spectral_analysis_{DATASET}_{int(prune_ratio*100)}pct.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {out_path}")
    plt.close()

# =============================================================================
# HYBRID SCORE ANALYSIS
# =============================================================================
def hybrid_score_analysis(model, train_loader, prune_ratio):
    """
    Tests the hybrid score idea:
        score_i = R_i * L2_norm_i
    Prune lowest combined score.
    Shows what it selects vs AEI alone vs L2 alone.
    """
    n         = model.fc1.out_features
    num_prune = int(n * prune_ratio)

    R, _, _  = compute_all_R_values(model, train_loader)
    w        = model.fc1.weight.data.cpu().numpy()
    l2_scores = np.sqrt(np.sum(w ** 2, axis=1))

    # Normalise both to [0, 1] before combining
    R_norm  = (R - R.min())  / (R.max()  - R.min()  + 1e-8)
    L2_norm = (l2_scores - l2_scores.min()) / (l2_scores.max() - l2_scores.min() + 1e-8)

    hybrid  = R_norm * L2_norm

    aei_pruned    = set(np.argsort(R)[:num_prune].tolist())
    l2_pruned     = set(np.argsort(l2_scores)[:num_prune].tolist())
    hybrid_pruned = set(np.argsort(hybrid)[:num_prune].tolist())

    aei_l2_overlap     = len(aei_pruned    & l2_pruned)     / num_prune * 100
    aei_hybrid_overlap = len(aei_pruned    & hybrid_pruned) / num_prune * 100
    l2_hybrid_overlap  = len(l2_pruned     & hybrid_pruned) / num_prune * 100

    print(f"\n{'='*60}")
    print(f"  HYBRID SCORE ANALYSIS (R_i × L2_norm_i)")
    print(f"  Prune neurons that are BOTH peripheral AND small-weight")
    print(f"{'='*60}")
    print(f"  AEI vs L2 overlap:     {aei_l2_overlap:.1f}%")
    print(f"  AEI vs Hybrid overlap: {aei_hybrid_overlap:.1f}%")
    print(f"  L2  vs Hybrid overlap: {l2_hybrid_overlap:.1f}%")
    print(f"\n  Interpretation:")
    if aei_l2_overlap < 40:
        print(f"  -> AEI and L2 are selecting VERY different neurons ({aei_l2_overlap:.1f}% overlap)")
        print(f"     Hybrid is a genuinely new selection — strong case for combining them.")
    elif aei_l2_overlap < 65:
        print(f"  -> AEI and L2 have moderate overlap ({aei_l2_overlap:.1f}%)")
        print(f"     Hybrid adds value by requiring BOTH criteria simultaneously.")
    else:
        print(f"  -> AEI and L2 largely agree ({aei_l2_overlap:.1f}% overlap)")
        print(f"     They may be capturing similar structure on this dataset.")
    print(f"{'='*60}\n")

# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print(f"\nDevice: {DEVICE}")
    print(f"Dataset: {DATASET}, Prune ratio: {PRUNE_RATIO*100:.0f}%\n")

    train_loader, test_loader = load_mnist()

    # Train a fresh model
    print("Training model...")
    model = SimpleMLP()
    acc   = train(model, train_loader, test_loader, epochs=10)
    print(f"Baseline accuracy: {acc:.2f}%\n")

    # Main spectral analysis
    R, R_percentile, aei_pruned, results = analyse(model, train_loader, PRUNE_RATIO)

    # Plots
    print("Generating plots...")
    plot_spectral_distribution(R, R_percentile, aei_pruned, results, PRUNE_RATIO, SAVE_DIR)

    # Hybrid score analysis
    hybrid_score_analysis(model, train_loader, PRUNE_RATIO)

    print("\nDone.")
