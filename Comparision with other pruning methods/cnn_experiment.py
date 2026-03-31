"""
cnn_experiment.py — Spectral Pruning on CNN / CIFAR-10
=======================================================
Extends AEI spectral pruning to convolutional filter-level structured pruning.
Compares: Spectral (AEI), Hybrid (AEI×L2), L1, L2, SNIP, GraSP, Random
Target layer: conv2 filters (64 filters, structurally analogous to FC1 in MLP)
Dataset: CIFAR-10

Usage (Colab):
    !python cnn_experiment.py | tee /content/drive/MyDrive/SpectralPruning_Results/cnn_cifar10.txt
"""

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
    All three size fields are stored so pruned copies can be constructed.
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
        x = self.pool(F.relu(self.conv1(x)))   # B×32×16×16
        x = self.pool(F.relu(self.conv2(x)))   # B×64×8×8
        x = x.view(x.size(0), -1)             # B×4096
        x = F.relu(self.fc1(x))               # B×256
        return self.fc2(x)                    # B×10


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
                  f"{total_loss/len(train_loader):.4f}")
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
# 4.  Activation collection (AEI needs this)
# ─────────────────────────────────────────────────────────────────────────────

def collect_filter_activations(model, data_loader, device,
                                layer_name='conv2', n_batches=5):
    """
    Collect spatially-averaged activations for every filter in `layer_name`.
    Returns ndarray of shape [N_samples, N_filters].
    Spatial average pooling preserves the co-activation signal while
    discarding spatial position — analogous to scalar neuron activations in MLP.
    """
    model.eval()
    acts = []

    def hook(module, inp, out):
        # out: [B, C, H, W]  →  global avg pool  →  [B, C]
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
# 5.  AEI score computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_aei_scores(activations, threshold=0.3):
    """
    Build Pearson co-activation graph, compute normalised graph Laplacian,
    extract Fiedler vector v2, return R_i = Σ_{j∈N(i)} |v2_i − v2_j|.
    Low R_i  →  structurally peripheral filter  →  prune.
    """
    n = activations.shape[1]

    # Pearson correlation → weighted adjacency (threshold weak edges)
    corr = np.corrcoef(activations.T)          # [C, C]
    np.fill_diagonal(corr, 0.0)
    adj  = (np.abs(corr) > threshold).astype(float) * np.abs(corr)

    # Normalised Laplacian  L = I − D^{-½} A D^{-½}
    deg       = adj.sum(axis=1)
    deg_safe  = np.where(deg > 0, deg, 1.0)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(deg_safe))
    L_norm    = np.eye(n) - D_inv_sqrt @ adj @ D_inv_sqrt

    # Fiedler vector (second smallest eigenvector)
    _, vecs = eigh(L_norm)
    v2 = vecs[:, 1]

    # AEI score per filter
    R = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if adj[i, j] > 0:
                R[i] += abs(v2[i] - v2[j])
    return R


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Pruned model construction
# ─────────────────────────────────────────────────────────────────────────────

def create_pruned_cnn(original, keep_indices, layer_name='conv2'):
    """
    Physically remove filters and return a smaller SimpleCNN with correct
    weight shapes.  Supports pruning conv1 or conv2.
    """
    m    = original
    keep = sorted(keep_indices)
    nk   = len(keep)

    if layer_name == 'conv2':
        new = SimpleCNN(num_filters1=m.num_filters1,
                        num_filters2=nk,
                        num_fc=m.num_fc)
        # conv1 — unchanged
        new.conv1.weight.data = m.conv1.weight.data.clone()
        new.conv1.bias.data   = m.conv1.bias.data.clone()
        # conv2 — keep selected output filters
        new.conv2.weight.data = m.conv2.weight.data[keep].clone()
        new.conv2.bias.data   = m.conv2.bias.data[keep].clone()
        # fc1 — select columns corresponding to kept filters
        # After view, filter k occupies columns [k*64 : (k+1)*64]
        spatial = 8 * 8
        cols = []
        for k in keep:
            cols.extend(range(k * spatial, (k + 1) * spatial))
        new.fc1.weight.data = m.fc1.weight.data[:, cols].clone()
        new.fc1.bias.data   = m.fc1.bias.data.clone()
        # fc2 — unchanged
        new.fc2.weight.data = m.fc2.weight.data.clone()
        new.fc2.bias.data   = m.fc2.bias.data.clone()

    elif layer_name == 'conv1':
        new = SimpleCNN(num_filters1=nk,
                        num_filters2=m.num_filters2,
                        num_fc=m.num_fc)
        # conv1 — keep selected output filters
        new.conv1.weight.data = m.conv1.weight.data[keep].clone()
        new.conv1.bias.data   = m.conv1.bias.data[keep].clone()
        # conv2 — reduce input channels
        new.conv2.weight.data = m.conv2.weight.data[:, keep, :, :].clone()
        new.conv2.bias.data   = m.conv2.bias.data.clone()
        # fc1, fc2 — unchanged
        new.fc1.weight.data = m.fc1.weight.data.clone()
        new.fc1.bias.data   = m.fc1.bias.data.clone()
        new.fc2.weight.data = m.fc2.weight.data.clone()
        new.fc2.bias.data   = m.fc2.bias.data.clone()

    else:
        raise ValueError(f"Unknown layer_name: {layer_name}")

    return new


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Pruning classes
# ─────────────────────────────────────────────────────────────────────────────

class BaseConvPruner:
    """Shared prune() logic — subclasses only implement get_scores()."""

    def __init__(self, model, device, train_loader):
        self.model        = model
        self.device       = device
        self.train_loader = train_loader

    def get_scores(self, layer_name):
        raise NotImplementedError

    def prune(self, layer_name, prune_ratio):
        scores    = self.get_scores(layer_name)
        n_filters = len(scores)
        n_prune   = int(n_filters * prune_ratio)
        # Prune lowest-scoring filters (all methods: low = less important)
        keep = sorted(np.argsort(scores)[n_prune:].tolist())
        return create_pruned_cnn(self.model, keep, layer_name)


# ── AEI Spectral ──────────────────────────────────────────────────────────────
class SpectralConvPruner(BaseConvPruner):
    """
    AEI for conv filters.
    Activations → Pearson co-activation graph → Fiedler vector → R scores.
    Low R_i  (peripheral)  →  prune.
    """
    def get_scores(self, layer_name):
        acts = collect_filter_activations(self.model, self.train_loader,
                                          self.device, layer_name)
        return compute_aei_scores(acts)


# ── L1 Norm ───────────────────────────────────────────────────────────────────
class L1ConvPruner(BaseConvPruner):
    """Sum of |weights| across all spatial dimensions of each filter."""
    def get_scores(self, layer_name):
        w = getattr(self.model, layer_name).weight.data
        return w.abs().sum(dim=[1, 2, 3]).cpu().numpy()


# ── L2 Norm ───────────────────────────────────────────────────────────────────
class L2ConvPruner(BaseConvPruner):
    """Frobenius norm of each filter."""
    def get_scores(self, layer_name):
        w = getattr(self.model, layer_name).weight.data
        return w.pow(2).sum(dim=[1, 2, 3]).sqrt().cpu().numpy()


# ── SNIP ──────────────────────────────────────────────────────────────────────
class SNIPConvPruner(BaseConvPruner):
    """
    SNIP: |grad_ij × w_ij| summed per filter from one forward-backward pass.
    """
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
    """
    GraSP: -(H·g)_i × w_i  summed per filter.
    Uses two backward passes to approximate the Hessian-gradient product.
    """
    def get_scores(self, layer_name):
        mc = copy.deepcopy(self.model).to(self.device)
        mc.train()
        inputs, targets = next(iter(self.train_loader))
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        params = [p for p in mc.parameters() if p.requires_grad]

        # First backward — retain graph for second pass
        loss  = nn.CrossEntropyLoss()(mc(inputs), targets)
        grads = torch.autograd.grad(loss, params,
                                    create_graph=True, retain_graph=True)

        # Second backward — Hessian-gradient product via ∇(‖g‖²)
        gnorm = sum((g * g).sum() for g in grads)
        Hg    = torch.autograd.grad(gnorm, params)

        target_layer = getattr(mc, layer_name)
        for p, hg in zip(params, Hg):
            if p is target_layer.weight:
                scores = -(hg * p.data).sum(dim=[1, 2, 3])
                return scores.detach().cpu().numpy()

        raise ValueError(f"Layer {layer_name} not found in parameters.")


# ── Hybrid (AEI × L2) ─────────────────────────────────────────────────────────
class HybridConvPruner(BaseConvPruner):
    """
    Parameter-free combination: score_i = minmax(R_i) × minmax(L2_i).
    Prunes filters that are simultaneously structurally peripheral AND
    have small weight magnitude.  Directly extends the MLP hybrid result.
    """
    def get_scores(self, layer_name):
        # AEI
        acts  = collect_filter_activations(self.model, self.train_loader,
                                           self.device, layer_name)
        R     = compute_aei_scores(acts)
        R_n   = (R - R.min()) / (R.max() - R.min() + 1e-8)

        # L2
        w     = getattr(self.model, layer_name).weight.data
        L2    = w.pow(2).sum(dim=[1, 2, 3]).sqrt().cpu().numpy()
        L2_n  = (L2 - L2.min()) / (L2.max() - L2.min() + 1e-8)

        return R_n * L2_n


# ── Random ────────────────────────────────────────────────────────────────────
class RandomConvPruner(BaseConvPruner):
    """Uniformly random filter removal — lower bound baseline."""
    def __init__(self, model, device, train_loader, seed=42):
        super().__init__(model, device, train_loader)
        self.seed = seed

    def get_scores(self, layer_name):
        rng = np.random.default_rng(self.seed)
        n   = getattr(self.model, layer_name).weight.shape[0]
        return rng.random(n)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Experiment runner
# ─────────────────────────────────────────────────────────────────────────────

PRUNER_MAP = {
    'spectral': SpectralConvPruner,
    'hybrid':   HybridConvPruner,
    'l1_norm':  L1ConvPruner,
    'l2_norm':  L2ConvPruner,
    'snip':     SNIPConvPruner,
    'grasp':    GraSPConvPruner,
    'random':   RandomConvPruner,
}


def run_experiment(base_model, method, layer_name, prune_ratio,
                   train_loader, test_loader, device):
    model_copy = copy.deepcopy(base_model)
    pruner     = PRUNER_MAP[method](model_copy, device, train_loader)

    # Prune
    t0          = time.time()
    pruned      = pruner.prune(layer_name, prune_ratio)
    prune_time  = time.time() - t0

    post_acc    = evaluate(pruned, test_loader, device)

    # Fine-tune
    pruned      = finetune_model(pruned, train_loader, device,
                                 epochs=5, lr=1e-4)
    fine_acc    = evaluate(pruned, test_loader, device)

    n_orig = getattr(base_model, layer_name).weight.shape[0]
    n_keep = int(n_orig * (1 - prune_ratio))
    print(f"  [{method:10s}] {int(prune_ratio*100)}% pruned | "
          f"post-prune={post_acc:.2f}%  fine-tuned={fine_acc:.2f}%  "
          f"filters {n_orig}→{n_keep}  (prune+score: {prune_time:.1f}s)")
    return fine_acc


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("=" * 65)

    train_loader, test_loader = load_cifar10(batch_size=128)

    # ── Train base model ──────────────────────────────────────────────────────
    print("\nTraining baseline SimpleCNN on CIFAR-10 (20 epochs)...")
    base_model = SimpleCNN(num_filters1=32, num_filters2=64, num_fc=256)
    base_model = train_model(base_model, train_loader, device,
                             epochs=20, lr=1e-3)
    base_acc = evaluate(base_model, test_loader, device)
    print(f"\nBaseline accuracy: {base_acc:.2f}%")
    print(f"Architecture: Conv1(3→32) → Conv2(32→64) → FC1(4096→256) → FC2(256→10)")
    print(f"Total params: "
          f"{sum(p.numel() for p in base_model.parameters()):,}")

    # ── Pruning experiments ───────────────────────────────────────────────────
    methods      = ['spectral', 'hybrid', 'l1_norm', 'l2_norm',
                    'snip', 'grasp', 'random']
    layer_name   = 'conv2'
    prune_ratios = [0.2, 0.3, 0.4]

    results = {m: {} for m in methods}

    for ratio in prune_ratios:
        print(f"\n{'='*65}")
        print(f"Pruning {int(ratio*100)}% of {layer_name} filters  "
              f"({64} → {int(64*(1-ratio))} filters)  + 5-epoch fine-tune")
        print(f"{'='*65}")
        for method in methods:
            results[method][ratio] = run_experiment(
                base_model, method, layer_name, ratio,
                train_loader, test_loader, device)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("RESULTS SUMMARY — SimpleCNN / CIFAR-10 / conv2 filter pruning")
    print("Fine-tuned test accuracy (%)")
    print("=" * 65)
    header = f"{'Method':<12}" + "".join(
        f"  {int(r*100):>3}%" for r in prune_ratios)
    print(header)
    print("-" * 35)
    for method in methods:
        row = f"{method:<12}" + "".join(
            f"  {results[method][r]:>5.2f}%" for r in prune_ratios)
        print(row)
    print("-" * 35)
    print(f"{'Baseline':<12}  {base_acc:.2f}%  {base_acc:.2f}%  {base_acc:.2f}%")

    # ── Drop vs baseline ──────────────────────────────────────────────────────
    print("\nAccuracy drop from baseline (lower = better)")
    print("-" * 35)
    for method in methods:
        row = f"{method:<12}" + "".join(
            f"  {base_acc - results[method][r]:>5.2f}pp" for r in prune_ratios)
        print(row)


if __name__ == '__main__':
    main()