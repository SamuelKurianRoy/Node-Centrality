"""
threshold_density_ablation.py

Ablation: impact of the correlation threshold tau (graph density) on AEI.
Answers Reviewer 1's request for a discussion of the impact of graph
density / threshold selection, with real numbers rather than just prose.

Reuses building blocks from pruning_logic.py (SimpleMLP, load_dataset,
train_model, _create_pruned_model) - drop this file into the same folder
as pruning_logic.py ("Comparision with other pruning methods/") and run.

For each tau in a grid, reports:
  - density          : fraction of nonzero off-diagonal edges
  - lambda2          : algebraic connectivity (0 => disconnected graph)
  - n_components     : number of connected components (degeneracy check)
  - spearman_vs_dense: rank correlation of R(tau) against the tau=0 baseline
  - finetuned_acc     : downstream accuracy after pruning 30% of neurons by R(tau)

Usage:
    python threshold_density_ablation.py
"""
import numpy as np
import torch
from scipy.linalg import eigh
from scipy.stats import spearmanr
from scipy.sparse.csgraph import connected_components
from torch.utils.data import DataLoader

from pruning_logic import SimpleMLP, train_model, load_dataset, _create_pruned_model


def collect_activations(model, loader, device, num_batches=20):
    acts = []
    handle = model.fc1.register_forward_hook(lambda m, i, o: acts.append(o.detach()))
    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= num_batches:
                break
            model(x.to(device))
    handle.remove()
    return torch.cat(acts, dim=0).cpu().numpy()


def build_graph(acts, tau):
    """Pearson correlation -> absolute-value weighted adjacency, optionally
    thresholded at tau (tau=0 reproduces the dense MLP-experiment recipe;
    tau=0.3 reproduces the CNN-filter-experiment recipe)."""
    corr = np.nan_to_num(np.corrcoef(acts.T), nan=0.0)
    np.fill_diagonal(corr, 0.0)
    adj = np.abs(corr)
    if tau > 0:
        adj = np.where(adj > tau, adj, 0.0)
    return adj


def fiedler_and_R(adj):
    n = adj.shape[0]
    deg = adj.sum(axis=1)
    deg_safe = np.where(deg > 0, deg, 1e-8)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(deg_safe))
    L = np.eye(n) - D_inv_sqrt @ adj @ D_inv_sqrt
    eigvals, eigvecs = eigh(L)
    v2 = eigvecs[:, 1]
    diff = np.abs(v2[:, None] - v2[None, :])
    R = np.sum(adj * diff, axis=1)
    n_components, _ = connected_components(adj > 0, directed=False)
    return R, eigvals[1], n_components


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Swap 'FashionMNIST' for 'MNIST' / 'KMNIST' if you want the ablation
    # reported on a different dataset.
    train_ds, test_ds, input_size, num_classes = load_dataset('FashionMNIST')
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=1000, shuffle=False, num_workers=2)

    model = SimpleMLP(input_size=input_size, hidden_size=256, num_classes=num_classes).to(device)
    train_model(model, train_loader, test_loader, device, epochs=10)

    acts = collect_activations(model, train_loader, device, num_batches=20)

    taus = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    R_baseline = None
    results = []
    for tau in taus:
        adj = build_graph(acts, tau)
        R, lambda2, n_components = fiedler_and_R(adj)
        n = adj.shape[0]
        density = (adj > 0).sum() / (n * n - n)

        if R_baseline is None:
            R_baseline = R
            rho = 1.0
        else:
            rho, _ = spearmanr(R, R_baseline)

        num_keep = int(n * 0.7)  # keep 70% => 30% sparsity
        keep_indices = np.sort(np.argsort(R)[-num_keep:])
        pruned = _create_pruned_model(model, 'fc1', keep_indices, device)
        acc = train_model(pruned, train_loader, test_loader, device, epochs=5, lr=1e-4)

        row = dict(tau=tau, density=density, lambda2=lambda2,
                   n_components=n_components, spearman_vs_dense=rho,
                   finetuned_acc=acc)
        results.append(row)
        print(row)

    print("\nMarkdown table for the manuscript (tab:threshold_ablation):\n")
    print("| $\\tau$ | density | $\\lambda_2$ | components | $\\rho$ vs.\\ dense | acc.\\ (30\\%) |")
    print("|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['tau']:.2f} | {r['density']:.3f} | {r['lambda2']:.4f} | "
              f"{r['n_components']} | {r['spearman_vs_dense']:.3f} | {r['finetuned_acc']:.2f}\\% |")


if __name__ == '__main__':
    main()
