import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from scipy.linalg import eigh
import copy

# =============================================================================
# MODEL DEFINITION
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
# SHARED HELPERS
# =============================================================================

def _rebuild_model(original_model, new_fc1, new_fc2, device):
    class PrunedMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten  = nn.Flatten()
            self.fc1      = new_fc1
            self.relu     = nn.ReLU()
            self.dropout  = nn.Dropout(0.5)
            self.fc2      = new_fc2

        def forward(self, x):
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    return PrunedMLP().to(device)


def _create_pruned_model(model, layer_name, keep_indices, device):
    """Physically remove neurons — creates a smaller model."""
    if layer_name != 'fc1':
        raise NotImplementedError("Only 'fc1' pruning is supported.")

    original_model = model.module if isinstance(model, nn.DataParallel) else model
    layer_to_prune = original_model.fc1
    next_layer     = original_model.fc2

    with torch.no_grad():
        new_fc1 = nn.Linear(
            layer_to_prune.in_features,
            len(keep_indices),
            bias=(layer_to_prune.bias is not None)
        ).to(device)
        new_fc1.weight.copy_(layer_to_prune.weight[keep_indices, :])
        if layer_to_prune.bias is not None:
            new_fc1.bias.copy_(layer_to_prune.bias[keep_indices])

        new_fc2 = nn.Linear(
            len(keep_indices),
            next_layer.out_features,
            bias=(next_layer.bias is not None)
        ).to(device)
        new_fc2.weight.copy_(next_layer.weight[:, keep_indices])
        if next_layer.bias is not None:
            new_fc2.bias.copy_(next_layer.bias)

    return _rebuild_model(original_model, new_fc1, new_fc2, device)


def apply_masks(model):
    """Re-zero pruned weights after each optimiser step (edge pruning only)."""
    for name, module in model.named_modules():
        if hasattr(module, 'weight_mask') and module.weight_mask is not None:
            with torch.no_grad():
                module.weight.data *= module.weight_mask


# =============================================================================
# NODE PRUNING 1 — Spectral (AEI / R value)
# =============================================================================

class SpectralPruner:
    """
    NODE pruning using the Adjacency Edge Index (AEI).
    R_i = sum_{(i,j) in E_i} |v2_i - v2_j|
    Low R_i  =>  structurally peripheral  =>  prune.
    """

    def __init__(self, model, device='cpu'):
        self.model              = model
        self.device             = device
        self.activation_storage = {}
        self.hooks              = []

    def _register_hooks(self, layer_names):
        def make_hook(name):
            def hook(module, inp, out):
                self.activation_storage[name] = out.detach()
            return hook
        for name, module in self.model.named_modules():
            if name in layer_names:
                self.hooks.append(module.register_forward_hook(make_hook(name)))

    def _remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def _collect_activations(self, dataloader, layer_name, num_batches=20):
        self.activation_storage = {}
        self._register_hooks([layer_name])
        self.model.eval()
        acts = []
        with torch.no_grad():
            for i, (data, _) in enumerate(dataloader):
                if i >= num_batches:
                    break
                _ = self.model(data.to(self.device))
                if layer_name in self.activation_storage:
                    acts.append(self.activation_storage[layer_name])
        self._remove_hooks()
        if not acts:
            raise ValueError(f"No activations collected for {layer_name}")
        return torch.cat(acts, dim=0)

    def _build_fiedler(self, activations):
        acts_np      = activations.view(activations.size(0), -1).cpu().numpy()
        corr         = np.nan_to_num(np.corrcoef(acts_np.T), nan=0.0)
        adj          = np.abs(corr)
        deg          = np.maximum(adj.sum(axis=1), 1e-8)
        D_inv_sqrt   = np.diag(1.0 / np.sqrt(deg))
        L            = np.eye(len(deg)) - D_inv_sqrt @ adj @ D_inv_sqrt
        _, vecs      = eigh(L)
        return vecs[:, 1], adj   # Fiedler vector + adjacency

    def _compute_R(self, activations):
        v2, adj = self._build_fiedler(activations)
        # Vectorised: R_i = sum_j adj_ij * |v2_i - v2_j|
        diff = np.abs(v2[:, None] - v2[None, :])
        R    = np.sum(adj * diff, axis=1)
        return R

    def prune_layer(self, layer_name, prune_ratio, dataloader, num_batches=20):
        acts         = self._collect_activations(dataloader, layer_name, num_batches)
        R            = self._compute_R(acts)
        n            = len(R)
        num_keep     = int(n * (1 - prune_ratio))
        keep_indices = np.sort(np.argsort(R)[-num_keep:])
        print(f"[Node/Spectral]  Layer '{layer_name}': {n} -> {num_keep} neurons "
              f"({prune_ratio*100:.1f}% pruned by AEI R value)")
        return _create_pruned_model(self.model, layer_name, keep_indices, self.device)


# =============================================================================
# NODE PRUNING 2 — Random (baseline)
# =============================================================================

class RandomPruner:
    """NODE pruning baseline: removes neurons chosen uniformly at random."""

    def __init__(self, model, device='cpu'):
        self.model  = model
        self.device = device

    def prune_layer(self, layer_name, prune_ratio, trial_seed=None):
        m        = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        layer    = dict(m.named_modules())[layer_name]
        n        = layer.out_features
        num_keep = int(n * (1 - prune_ratio))
        rng      = np.random.RandomState(trial_seed)
        keep_indices = np.sort(rng.choice(n, num_keep, replace=False))
        print(f"[Node/Random]    Layer '{layer_name}': {n} -> {num_keep} neurons "
              f"({prune_ratio*100:.1f}% pruned randomly)")
        return _create_pruned_model(self.model, layer_name, keep_indices, self.device)


# =============================================================================
# NODE PRUNING 3 — L1 Norm
# =============================================================================

class L1NormPruner:
    """
    NODE pruning using L1 norm of each neuron's weight vector.
    score_i = sum_j |w_ij|
    Low score  =>  neuron contributes little  =>  prune.
    This is the most common structured pruning baseline in the literature.
    """

    def __init__(self, model, device='cpu'):
        self.model  = model
        self.device = device

    def prune_layer(self, layer_name, prune_ratio):
        m            = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        layer        = dict(m.named_modules())[layer_name]
        weights      = layer.weight.data.cpu().numpy()   # (out_features, in_features)
        scores       = np.sum(np.abs(weights), axis=1)   # L1 norm per neuron
        n            = len(scores)
        num_keep     = int(n * (1 - prune_ratio))
        keep_indices = np.sort(np.argsort(scores)[-num_keep:])
        print(f"[Node/L1Norm]    Layer '{layer_name}': {n} -> {num_keep} neurons "
              f"({prune_ratio*100:.1f}% pruned by L1 norm)")
        return _create_pruned_model(self.model, layer_name, keep_indices, self.device)


# =============================================================================
# NODE PRUNING 4 — L2 Norm
# =============================================================================

class L2NormPruner:
    """
    NODE pruning using L2 norm of each neuron's weight vector.
    score_i = sqrt(sum_j w_ij^2)
    Low score  =>  neuron has small overall magnitude  =>  prune.
    """

    def __init__(self, model, device='cpu'):
        self.model  = model
        self.device = device

    def prune_layer(self, layer_name, prune_ratio):
        m            = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        layer        = dict(m.named_modules())[layer_name]
        weights      = layer.weight.data.cpu().numpy()
        scores       = np.sqrt(np.sum(weights ** 2, axis=1))   # L2 norm per neuron
        n            = len(scores)
        num_keep     = int(n * (1 - prune_ratio))
        keep_indices = np.sort(np.argsort(scores)[-num_keep:])
        print(f"[Node/L2Norm]    Layer '{layer_name}': {n} -> {num_keep} neurons "
              f"({prune_ratio*100:.1f}% pruned by L2 norm)")
        return _create_pruned_model(self.model, layer_name, keep_indices, self.device)




# =============================================================================
# NODE PRUNING 3b — Hybrid (AEI × L2 Norm)
# =============================================================================

class HybridPruner:
    """
    NODE pruning using a combined spectral + magnitude score.

    score_i = R_i_normalised * L2_norm_i_normalised

    R_i  = AEI spectral score (low => structurally peripheral)
    L2_i = L2 norm of neuron's weight row (low => small magnitude)

    Prunes neurons that are BOTH spectrally peripheral AND have small weights.
    Spectral analysis showed AEI vs L2 overlap is only ~30%, so this combination
    captures genuinely different information from either method alone.
    """

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.activation_storage = {}
        self.hooks = []

    def _register_hooks(self, layer_names):
        def make_hook(name):
            def hook(module, inp, out):
                self.activation_storage[name] = out.detach()
            return hook
        for name, module in self.model.named_modules():
            if name in layer_names:
                self.hooks.append(module.register_forward_hook(make_hook(name)))

    def _remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def _collect_activations(self, dataloader, layer_name, num_batches=20):
        self.activation_storage = {}
        self._register_hooks([layer_name])
        self.model.eval()
        acts = []
        with torch.no_grad():
            for i, (data, _) in enumerate(dataloader):
                if i >= num_batches:
                    break
                _ = self.model(data.to(self.device))
                if layer_name in self.activation_storage:
                    acts.append(self.activation_storage[layer_name])
        self._remove_hooks()
        if not acts:
            raise ValueError(f"No activations collected for {layer_name}")
        return torch.cat(acts, dim=0)

    def _compute_R(self, activations):
        acts_np = activations.view(activations.size(0), -1).cpu().numpy()
        corr = np.nan_to_num(np.corrcoef(acts_np.T), nan=0.0)
        adj = np.abs(corr)
        deg = np.maximum(adj.sum(axis=1), 1e-8)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
        L = np.eye(len(deg)) - D_inv_sqrt @ adj @ D_inv_sqrt
        _, vecs = eigh(L)
        v2 = vecs[:, 1]
        diff = np.abs(v2[:, None] - v2[None, :])
        R = np.sum(adj * diff, axis=1)
        return R

    def prune_layer(self, layer_name, prune_ratio, dataloader, num_batches=20):
        m = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        layer = dict(m.named_modules())[layer_name]

        # --- Spectral (R) score ---
        acts = self._collect_activations(dataloader, layer_name, num_batches)
        R = self._compute_R(acts)

        # --- L2 norm score ---
        weights = layer.weight.data.cpu().numpy()          # (out_features, in_features)
        L2 = np.sqrt(np.sum(weights ** 2, axis=1))        # (out_features,)

        # --- Normalise both to [0, 1] and multiply ---
        def minmax(v):
            vmin, vmax = v.min(), v.max()
            if vmax - vmin < 1e-12:
                return np.ones_like(v)
            return (v - vmin) / (vmax - vmin)

        score = minmax(R) * minmax(L2)   # low => peripheral AND weak => prune

        n = len(score)
        num_keep = int(n * (1 - prune_ratio))
        keep_indices = np.sort(np.argsort(score)[-num_keep:])   # keep highest scores

        print(f"[Node/Hybrid] Layer '{layer_name}': {n} -> {num_keep} neurons "
              f"({prune_ratio*100:.1f}% pruned by AEI x L2 hybrid score)")

        return _create_pruned_model(self.model, layer_name, keep_indices, self.device)

# =============================================================================
# NODE PRUNING 5 — SNIP (Single-shot Network Pruning based on Connection Sensitivity)
# Lee et al., 2019 — https://arxiv.org/abs/1810.02340
# Adapted here to NODE level: aggregate connection sensitivities per neuron.
# =============================================================================

class SNIPPruner:
    """
    NODE pruning using SNIP sensitivity scores.

    Original SNIP computes per-weight sensitivity:
        s_ij = |g_ij * w_ij|   where g_ij = dL/dw_ij

    We adapt this to node-level structured pruning by aggregating
    per-weight scores across each neuron's incoming weights:
        score_i = sum_j |g_ij * w_ij|

    Neurons with low total sensitivity are pruned.
    Only requires a single forward+backward pass — very cheap.
    """

    def __init__(self, model, device='cpu'):
        self.model  = model
        self.device = device

    def prune_layer(self, layer_name, prune_ratio, dataloader, num_batches=1):
        m         = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        layer     = dict(m.named_modules())[layer_name]
        criterion = nn.CrossEntropyLoss()

        # Single forward+backward pass to get gradients
        self.model.to(self.device)
        self.model.train()
        self.model.zero_grad()

        batches_used = 0
        for data, target in dataloader:
            if batches_used >= num_batches:
                break
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss   = criterion(output, target)
            loss.backward()
            batches_used += 1

        # SNIP score per neuron: sum of |grad * weight| across input connections
        with torch.no_grad():
            if layer.weight.grad is None:
                raise RuntimeError("No gradient found — check layer name.")
            grad     = layer.weight.grad.cpu().numpy()   # (out, in)
            weights  = layer.weight.data.cpu().numpy()   # (out, in)
            scores   = np.sum(np.abs(grad * weights), axis=1)  # per neuron

        self.model.zero_grad()

        n            = len(scores)
        num_keep     = int(n * (1 - prune_ratio))
        keep_indices = np.sort(np.argsort(scores)[-num_keep:])
        print(f"[Node/SNIP]      Layer '{layer_name}': {n} -> {num_keep} neurons "
              f"({prune_ratio*100:.1f}% pruned by SNIP sensitivity)")
        return _create_pruned_model(self.model, layer_name, keep_indices, self.device)


# =============================================================================
# NODE PRUNING 6 — GraSP (Gradient Signal Preservation)
# Wang et al., 2020 — https://arxiv.org/abs/2002.07376
# Adapted here to NODE level.
# =============================================================================

class GraSPPruner:
    """
    NODE pruning using GraSP scores.

    GraSP identifies weights whose removal least reduces the gradient
    signal (i.e. preserves the gradient flow). The per-weight score is:
        score_ij = -(H * g)_ij * w_ij
    where H*g is the Hessian-vector product approximated via two
    backward passes (no explicit Hessian needed).

    We adapt to node level by summing per-weight scores per neuron:
        score_i = sum_j [-(H*g)_ij * w_ij]

    High score => removing this neuron damages gradient flow => keep.
    Low score  => safe to prune.

    Requires two backward passes.
    """

    def __init__(self, model, device='cpu'):
        self.model  = model
        self.device = device

    def prune_layer(self, layer_name, prune_ratio, dataloader, num_batches=1,
                    temperature=200.0):
        m         = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        layer     = dict(m.named_modules())[layer_name]
        criterion = nn.CrossEntropyLoss()

        self.model.to(self.device)

        # --- Pass 1: compute gradient g ---
        self.model.train()
        self.model.zero_grad()
        batches_used = 0
        for data, target in dataloader:
            if batches_used >= num_batches:
                break
            data, target = data.to(self.device), target.to(self.device)
            # GraSP uses temperature-scaled softmax to sharpen gradients
            output = self.model(data) / temperature
            loss   = criterion(output, target)
            loss.backward()
            batches_used += 1

        # Save gradient vector g (flattened over all parameters)
        grad_dict = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_dict[name] = param.grad.clone()

        # --- Pass 2: compute H*g via grad of (g^T * grad(L)) ---
        self.model.zero_grad()
        batches_used = 0
        for data, target in dataloader:
            if batches_used >= num_batches:
                break
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data) / temperature
            loss   = criterion(output, target)
            # Compute grad of loss
            grads  = torch.autograd.grad(loss, self.model.parameters(),
                                         create_graph=True)
            # Dot product g^T * grad(L) — this is a scalar
            g_dot_grad = sum(
                (grad_dict[n] * g).sum()
                for (n, _), g in zip(self.model.named_parameters(), grads)
                if n in grad_dict
            )
            g_dot_grad.backward()
            batches_used += 1

        # H*g is now in param.grad for each parameter
        # GraSP score per weight: -(H*g)_ij * w_ij
        with torch.no_grad():
            if layer.weight.grad is None:
                raise RuntimeError("No Hessian-vector product found.")
            Hg      = layer.weight.grad.cpu().numpy()
            weights = layer.weight.data.cpu().numpy()
            # Negative because GraSP prunes weights that LEAST preserve gradient signal
            scores  = np.sum(-Hg * weights, axis=1)   # per neuron

        self.model.zero_grad()

        n            = len(scores)
        num_keep     = int(n * (1 - prune_ratio))
        keep_indices = np.sort(np.argsort(scores)[-num_keep:])
        print(f"[Node/GraSP]     Layer '{layer_name}': {n} -> {num_keep} neurons "
              f"({prune_ratio*100:.1f}% pruned by GraSP)")
        return _create_pruned_model(self.model, layer_name, keep_indices, self.device)


# =============================================================================
# EDGE PRUNING 1 — Spectral Edge (R² value)
# =============================================================================

class SpectralEdgePruner:
    """
    EDGE (unstructured) pruning using R² = (v2_i - v2_j)² per weight.
    Low R² => same spectral community => prune.
    """

    def __init__(self, model, device='cpu'):
        self.model              = model
        self.device             = device
        self.activation_storage = {}
        self.hooks              = []

    def _register_hooks(self, layer_names):
        def make_hook(name):
            def hook(module, inp, out):
                self.activation_storage[name] = out.detach()
            return hook
        for name, module in self.model.named_modules():
            if name in layer_names:
                self.hooks.append(module.register_forward_hook(make_hook(name)))

    def _remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def _collect_activations(self, dataloader, layer_name, num_batches=20):
        self.activation_storage = {}
        self._register_hooks([layer_name])
        self.model.eval()
        acts = []
        with torch.no_grad():
            for i, (data, _) in enumerate(dataloader):
                if i >= num_batches:
                    break
                _ = self.model(data.to(self.device))
                if layer_name in self.activation_storage:
                    acts.append(self.activation_storage[layer_name])
        self._remove_hooks()
        if not acts:
            raise ValueError(f"No activations for {layer_name}")
        return torch.cat(acts, dim=0)

    def _build_fiedler(self, activations):
        acts_np    = activations.view(activations.size(0), -1).cpu().numpy()
        corr       = np.nan_to_num(np.corrcoef(acts_np.T), nan=0.0)
        adj        = np.abs(corr)
        deg        = np.maximum(adj.sum(axis=1), 1e-8)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
        L          = np.eye(len(deg)) - D_inv_sqrt @ adj @ D_inv_sqrt
        _, vecs    = eigh(L)
        return vecs[:, 1]

    def prune_layer(self, layer_name, prune_ratio, dataloader, num_batches=20):
        m     = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        layer = dict(m.named_modules())[layer_name]

        print(f"[Edge/Spectral]  Collecting activations for '{layer_name}'...")
        acts = self._collect_activations(dataloader, layer_name, num_batches)

        print(f"[Edge/Spectral]  Computing Fiedler vector...")
        v2  = self._build_fiedler(acts)
        n   = len(v2)

        weight = layer.weight.data
        out_f, in_f = weight.shape

        row_idx = np.minimum(np.arange(out_f), n - 1)
        col_idx = np.minimum(np.arange(in_f),  n - 1)
        R_sq    = (v2[row_idx][:, None] - v2[col_idx][None, :]) ** 2

        total       = R_sq.size
        num_prune   = int(total * prune_ratio)
        threshold   = np.sort(R_sq.flatten())[num_prune]
        mask_np     = (R_sq >= threshold).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_np).to(self.device)

        with torch.no_grad():
            layer.weight.data = layer.weight.data * mask_tensor
        if hasattr(layer, 'weight_mask'):
            del layer.weight_mask
        layer.register_buffer('weight_mask', mask_tensor)

        pruned   = int(np.sum(mask_np == 0))
        sparsity = pruned / total * 100
        print(f"[Edge/Spectral]  Layer '{layer_name}': {total} weights -> "
              f"{pruned} pruned ({sparsity:.1f}% sparse) by R² value")
        return self.model


# =============================================================================
# EDGE PRUNING 2 — Random Edge (baseline)
# =============================================================================

class RandomEdgePruner:
    """EDGE pruning baseline: randomly zeros weights."""

    def __init__(self, model, device='cpu'):
        self.model  = model
        self.device = device

    def prune_layer(self, layer_name, prune_ratio, trial_seed=None):
        m     = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        layer = dict(m.named_modules())[layer_name]
        w     = layer.weight.data
        total = w.numel()
        n_prune = int(total * prune_ratio)
        rng   = np.random.RandomState(trial_seed)
        idx   = rng.choice(total, n_prune, replace=False)
        mask  = torch.ones(total, device=self.device)
        mask[idx] = 0.0
        mask  = mask.view(w.shape)
        with torch.no_grad():
            layer.weight.data = layer.weight.data * mask
        if hasattr(layer, 'weight_mask'):
            del layer.weight_mask
        layer.register_buffer('weight_mask', mask)
        print(f"[Edge/Random]    Layer '{layer_name}': {total} weights -> "
              f"{n_prune} pruned ({prune_ratio*100:.1f}% sparse) randomly")
        return self.model


# =============================================================================
# TRAINING
# =============================================================================

def train_model(model, train_loader, test_loader, device, epochs=10,
                respect_masks=False, lr=0.001):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss   = criterion(output, target)
            loss.backward()
            optimizer.step()
            if respect_masks:
                apply_masks(model)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            _, predicted = torch.max(model(data), 1)
            total   += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total


# =============================================================================
# DATASET LOADER
# =============================================================================

def load_dataset(dataset_name):
    if dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])
        train_ds   = datasets.CIFAR10('./data', train=True,  download=False, transform=transform)
        test_ds    = datasets.CIFAR10('./data', train=False, download=False, transform=transform)
        input_size = 32 * 32 * 3
    else:
        cfg = {
            'MNIST':        (datasets.MNIST,        (0.1307,), (0.3081,)),
            'FashionMNIST': (datasets.FashionMNIST, (0.5,),    (0.5,)),
            'KMNIST':       (datasets.KMNIST,       (0.5,),    (0.5,)),
        }
        if dataset_name not in cfg:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        ds_class, mean, std = cfg[dataset_name]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        train_ds   = ds_class('./data', train=True,  download=False, transform=transform)
        test_ds    = ds_class('./data', train=False, download=False, transform=transform)
        input_size = 28 * 28

    return train_ds, test_ds, input_size, 10


# =============================================================================
# RUN EXPERIMENT
# Supported pruning_method values:
#   Node-level (structured):
#     'spectral'          -> SpectralPruner  (AEI R value)
#     'random_node'       -> RandomPruner
#     'l1_norm'           -> L1NormPruner
#     'l2_norm'           -> L2NormPruner
#     'snip'              -> SNIPPruner
#     'grasp'             -> GraSPPruner
#   Edge-level (unstructured):
#     'spectral_edge'           -> SpectralEdgePruner (R² value)
#     'random_edge_trial{N}'    -> RandomEdgePruner   (seed=N)
# =============================================================================

def run_experiment(params):
    dataset_name, prune_ratio, pruning_method, device_id = params

    device = torch.device(
        f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'
    )

    exp_id = f"{dataset_name}/{prune_ratio*100:.0f}%/{pruning_method}"
    print(f"\nStarting experiment: {exp_id} on {str(device).upper()}")

    try:
        train_dataset, test_dataset, input_size, num_classes = load_dataset(dataset_name)

        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True,
                                  num_workers=2, pin_memory=True)
        test_loader  = DataLoader(test_dataset,  batch_size=1000, shuffle=False,
                                  num_workers=2, pin_memory=True)

        # --- Baseline training ---
        print(f"--- ({exp_id}) Training baseline model ---")
        original_model = SimpleMLP(input_size=input_size, hidden_size=256,
                                   num_classes=num_classes)
        original_accuracy = train_model(original_model, train_loader, test_loader,
                                        device, epochs=10)
        original_params   = sum(p.numel() for p in original_model.parameters())
        print(f"--- ({exp_id}) Baseline accuracy: {original_accuracy:.2f}% "
              f"({original_params:,} params) ---")

        # --- Pruning ---
        print(f"--- ({exp_id}) Pruning ---")
        is_edge_prune = False

        if pruning_method == 'spectral':
            pruner       = SpectralPruner(original_model, device)
            pruned_model = pruner.prune_layer('fc1', prune_ratio, train_loader)

        elif pruning_method.startswith('random_node'):
            trial_seed = (int(pruning_method.replace('random_node_trial', ''))
                          if 'trial' in pruning_method else None)
            pruner = RandomPruner(original_model, device)
            pruned_model = pruner.prune_layer('fc1', prune_ratio, trial_seed=trial_seed)

        elif pruning_method == 'l1_norm':
            pruner       = L1NormPruner(original_model, device)
            pruned_model = pruner.prune_layer('fc1', prune_ratio)

        elif pruning_method == 'l2_norm':
            pruner       = L2NormPruner(original_model, device)
            pruned_model = pruner.prune_layer('fc1', prune_ratio)

        elif pruning_method == 'snip':
            pruner       = SNIPPruner(original_model, device)
            pruned_model = pruner.prune_layer('fc1', prune_ratio, train_loader)

        elif pruning_method == 'grasp':
            pruner       = GraSPPruner(original_model, device)
            pruned_model = pruner.prune_layer('fc1', prune_ratio, train_loader)

        elif pruning_method == 'hybrid':
            pruner = HybridPruner(original_model, device)
            pruned_model = pruner.prune_layer('fc1', prune_ratio, train_loader)

        elif pruning_method == 'spectral_edge':
            pruner        = SpectralEdgePruner(original_model, device)
            pruned_model  = pruner.prune_layer('fc1', prune_ratio, train_loader)
            is_edge_prune = True

        elif pruning_method.startswith('random_edge'):
            trial_seed    = (int(pruning_method.replace('random_edge_trial', ''))
                             if 'trial' in pruning_method else None)
            pruner        = RandomEdgePruner(original_model, device)
            pruned_model  = pruner.prune_layer('fc1', prune_ratio, trial_seed=trial_seed)
            is_edge_prune = True

        else:
            raise ValueError(f"Unknown pruning method: {pruning_method}")

        # --- Fine-tuning ---
        print(f"--- ({exp_id}) Fine-tuning pruned model ---")
        finetuned_accuracy = train_model(
            pruned_model, train_loader, test_loader, device,
            epochs=5, respect_masks=is_edge_prune, lr=1e-4
        )
        pruned_params = sum(p.numel() for p in pruned_model.parameters())
        print(f"--- ({exp_id}) Fine-tuned accuracy: {finetuned_accuracy:.2f}% "
              f"({pruned_params:,} params) ---")

        # Compression label
        if is_edge_prune:
            compression = f"{prune_ratio*100:.1f}% sparse"
        else:
            compression = f"{(1 - pruned_params / original_params) * 100:.1f}% smaller"

        # Normalise random_edge_trialN -> random_edge for the results table
        if pruning_method.startswith('random_edge'):
            display_method = 'random_edge'
        elif pruning_method.startswith('random_node'):
            display_method = 'random_node'
        else:
            display_method = pruning_method

        return {
            'dataset':      dataset_name,
            'prune_ratio':  f"{prune_ratio*100:.0f}%",
            'method':       display_method,
            'original_acc': f"{original_accuracy:.2f}%",
            'finetuned_acc':f"{finetuned_accuracy:.2f}%",
            'acc_drop':     f"{original_accuracy - finetuned_accuracy:.2f}%",
            'compression':  compression,
        }

    except Exception as e:
        print(f"!!! ERROR in {exp_id}: {e} !!!")
        import traceback; traceback.print_exc()
        return None
