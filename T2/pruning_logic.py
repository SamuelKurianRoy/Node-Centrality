import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from scipy.linalg import eigh
import copy

# =============================================================================
# Existing Node Pruning (R value = sum of |v2_i - v2_j| over neighbours)
# Keep this exactly as-is from pruning_logic.py
# =============================================================================

class SpectralPruner:
    """
    NODE pruning using the Adjacency Edge Index (AEI) R value.

    For each neuron i, the saliency score is:
        R_i = sum_{(i,j) in E_i} |v2_i - v2_j|

    Neurons with the lowest R_i scores are removed entirely
    (structured pruning — whole rows/columns deleted from weight matrices).
    """

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.activation_storage = {}
        self.hooks = []

    def register_hooks(self, layer_names):
        def get_activation(name):
            def hook(module, input, output):
                self.activation_storage[name] = output.detach()
            return hook
        for name, module in self.model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(get_activation(name))
                self.hooks.append(hook)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def collect_activations(self, dataloader, layer_name, num_batches=20):
        self.activation_storage = {}
        self.register_hooks([layer_name])
        self.model.eval()
        activations_list = []
        with torch.no_grad():
            for i, (data, _) in enumerate(dataloader):
                if i >= num_batches:
                    break
                data = data.to(self.device)
                _ = self.model(data)
                if layer_name in self.activation_storage:
                    activations_list.append(self.activation_storage[layer_name])
        self.remove_hooks()
        if not activations_list:
            raise ValueError(f"No activations collected for layer {layer_name}")
        return torch.cat(activations_list, dim=0)

    def _build_graph_and_fiedler(self, activations):
        """
        Build neuron co-activation graph via Pearson correlation,
        compute graph Laplacian, and return the Fiedler vector v2.
        """
        acts_np = activations.view(activations.size(0), -1).cpu().numpy()
        corr_matrix = np.nan_to_num(np.corrcoef(acts_np.T), nan=0.0)
        adjacency = np.abs(corr_matrix)

        degree_vector = np.maximum(np.sum(adjacency, axis=1), 1e-8)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degree_vector))
        laplacian = np.eye(len(degree_vector)) - D_inv_sqrt @ adjacency @ D_inv_sqrt

        eigenvalues, eigenvectors = eigh(laplacian)
        fiedler_vector = eigenvectors[:, 1]  # second smallest eigenvector = v2
        return fiedler_vector, adjacency

    def compute_node_aei(self, activations):
        """
        Compute the AEI R value for each neuron i:
            R_i = sum_{(i,j) in E_i} |v2_i - v2_j|

        Higher R_i = more dynamically central = keep.
        Lower R_i = structurally peripheral = prune.
        """
        fiedler_vector, adjacency = self._build_graph_and_fiedler(activations)
        n = len(fiedler_vector)
        R = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j and adjacency[i, j] > 0:
                    R[i] += np.abs(fiedler_vector[i] - fiedler_vector[j])
        return R

    def prune_layer(self, layer_name, prune_ratio, dataloader, num_batches=20):
        """Remove the lowest-R neurons from the specified layer."""
        activations = self.collect_activations(dataloader, layer_name, num_batches)
        R_scores = self.compute_node_aei(activations)

        num_neurons = len(R_scores)
        num_to_keep = int(num_neurons * (1 - prune_ratio))
        keep_indices = np.sort(np.argsort(R_scores)[-num_to_keep:])

        print(f"[Node/Spectral] Layer '{layer_name}': {num_neurons} -> {num_to_keep} neurons "
              f"({prune_ratio*100:.1f}% pruned by R value)")

        return _create_pruned_model(self.model, layer_name, keep_indices, self.device)


# =============================================================================
# NEW: Spectral Edge Pruner (R² value = (v2_i - v2_j)^2 per edge/weight)
# =============================================================================

class SpectralEdgePruner:
    """
    EDGE (unstructured) pruning using the squared AEI R² value.

    For each individual weight w_{i,j} connecting neuron i to neuron j,
    the edge importance score is:
        R²_{ij} = (v2_i - v2_j)^2

    where v2 is the Fiedler vector of the neuron co-activation graph.

    Edges with LOW R²_{ij} scores straddle neurons that sit in the same
    spectral community — they contribute little to global synchronizability
    and can be zeroed out safely.

    This is UNSTRUCTURED pruning: individual weights are zeroed, but the
    overall layer shape is preserved (no neuron is fully removed).
    The weight mask is registered as a buffer so fine-tuning respects it.
    """

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.activation_storage = {}
        self.hooks = []

    def register_hooks(self, layer_names):
        def get_activation(name):
            def hook(module, input, output):
                self.activation_storage[name] = output.detach()
            return hook
        for name, module in self.model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(get_activation(name))
                self.hooks.append(hook)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def collect_activations(self, dataloader, layer_name, num_batches=20):
        self.activation_storage = {}
        self.register_hooks([layer_name])
        self.model.eval()
        activations_list = []
        with torch.no_grad():
            for i, (data, _) in enumerate(dataloader):
                if i >= num_batches:
                    break
                data = data.to(self.device)
                _ = self.model(data)
                if layer_name in self.activation_storage:
                    activations_list.append(self.activation_storage[layer_name])
        self.remove_hooks()
        if not activations_list:
            raise ValueError(f"No activations collected for layer {layer_name}")
        return torch.cat(activations_list, dim=0)

    def _build_fiedler_vector(self, activations):
        """
        Build neuron co-activation graph via Pearson correlation and
        return the Fiedler vector v2 of the normalised graph Laplacian.
        """
        acts_np = activations.view(activations.size(0), -1).cpu().numpy()
        corr_matrix = np.nan_to_num(np.corrcoef(acts_np.T), nan=0.0)
        adjacency = np.abs(corr_matrix)

        degree_vector = np.maximum(np.sum(adjacency, axis=1), 1e-8)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degree_vector))
        laplacian = np.eye(len(degree_vector)) - D_inv_sqrt @ adjacency @ D_inv_sqrt

        eigenvalues, eigenvectors = eigh(laplacian)
        fiedler_vector = eigenvectors[:, 1]  # v2
        return fiedler_vector

    def compute_edge_r_squared(self, fiedler_vector, weight_shape):
        """
        Compute the R² importance score for every weight w_{i,j}:
            R²_{ij} = (v2_i - v2_j)^2

        where i indexes the output neuron (row) and j indexes the
        input neuron (column) of the weight matrix.

        If the weight matrix is larger than the Fiedler vector (e.g.
        input dim > hidden dim), indices beyond the vector length fall
        back to absolute magnitude of the Fiedler component for that
        axis.

        Returns a numpy array of shape == weight_shape.
        """
        out_features, in_features = weight_shape
        n = len(fiedler_vector)

        # Vectorised computation — much faster than nested loops
        # Clamp indices to valid range
        row_idx = np.minimum(np.arange(out_features), n - 1)
        col_idx = np.minimum(np.arange(in_features), n - 1)

        v_rows = fiedler_vector[row_idx]   # shape (out_features,)
        v_cols = fiedler_vector[col_idx]   # shape (in_features,)

        # Broadcast: R²_{ij} = (v2_i - v2_j)^2
        R_squared = (v_rows[:, None] - v_cols[None, :]) ** 2  # (out, in)
        return R_squared

    def prune_layer(self, layer_name, prune_ratio, dataloader, num_batches=20):
        """
        Zero out the fraction `prune_ratio` of weights with the lowest
        R² scores in the specified layer. The surviving weights and a
        binary mask are stored back on the layer.

        Steps:
            1. Collect neuron activations via forward hooks.
            2. Build co-activation graph → compute Fiedler vector v2.
            3. Compute R²_{ij} = (v2_i − v2_j)² for every weight.
            4. Rank all edges by R² and zero the bottom `prune_ratio`
               fraction (low R² = same spectral community = redundant).
            5. Register the binary mask as a buffer so that fine-tuning
               only updates surviving weights.
        """
        model_inner = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        layer = dict(model_inner.named_modules())[layer_name]

        # Step 1 – collect activations
        print(f"[Edge/Spectral] Collecting activations for '{layer_name}'...")
        activations = self.collect_activations(dataloader, layer_name, num_batches)

        # Step 2 – Fiedler vector
        print(f"[Edge/Spectral] Computing Fiedler vector...")
        fiedler_vector = self._build_fiedler_vector(activations)

        # Step 3 – R² scores for every weight
        weight = layer.weight.data  # (out_features, in_features)
        R_sq = self.compute_edge_r_squared(fiedler_vector, weight.shape)
        # R_sq is a numpy array; shape matches weight

        # Step 4 – determine threshold and build mask
        total_weights = R_sq.size
        num_to_prune  = int(total_weights * prune_ratio)

        flat_R_sq = R_sq.flatten()
        # Sort ascending; the bottom num_to_prune entries are pruned
        threshold = np.sort(flat_R_sq)[num_to_prune]

        # Mask: 1 = keep (R² >= threshold), 0 = prune (R² < threshold)
        mask_np = (R_sq >= threshold).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_np).to(self.device)

        # Step 5 – apply mask and register as buffer
        with torch.no_grad():
            layer.weight.data = layer.weight.data * mask_tensor

        # Store mask so it can be re-applied during fine-tuning if needed
        if hasattr(layer, 'weight_mask'):
            del layer.weight_mask
        layer.register_buffer('weight_mask', mask_tensor)

        actual_pruned = int(np.sum(mask_np == 0))
        sparsity = actual_pruned / total_weights * 100
        print(f"[Edge/Spectral] Layer '{layer_name}': {total_weights} weights -> "
              f"{actual_pruned} pruned ({sparsity:.1f}% sparse) by R² value")

        return self.model


# =============================================================================
# NEW: Random Edge Pruner  (baseline for edge pruning comparison)
# =============================================================================

class RandomEdgePruner:
    """
    EDGE (unstructured) pruning baseline: randomly zero out `prune_ratio`
    fraction of weights in the specified layer.

    Mirrors the interface of SpectralEdgePruner so it can be dropped
    straight into run_experiment as the 'random_edge' method.
    """

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device

    def prune_layer(self, layer_name, prune_ratio, trial_seed=None):
        model_inner = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        layer = dict(model_inner.named_modules())[layer_name]

        weight = layer.weight.data
        total_weights = weight.numel()
        num_to_prune  = int(total_weights * prune_ratio)

        if trial_seed is not None:
            rng = np.random.RandomState(trial_seed)
        else:
            rng = np.random.RandomState()

        prune_indices = rng.choice(total_weights, num_to_prune, replace=False)
        mask = torch.ones(total_weights, device=self.device)
        mask[prune_indices] = 0.0
        mask = mask.view(weight.shape)

        with torch.no_grad():
            layer.weight.data = layer.weight.data * mask

        if hasattr(layer, 'weight_mask'):
            del layer.weight_mask
        layer.register_buffer('weight_mask', mask)

        print(f"[Edge/Random]   Layer '{layer_name}': {total_weights} weights -> "
              f"{num_to_prune} pruned ({prune_ratio*100:.1f}% sparse) randomly")

        return self.model


# =============================================================================
# Helper: re-apply mask after each fine-tuning step
# (prevents gradient updates from reviving pruned weights)
# =============================================================================

def apply_masks(model):
    """
    Re-zero any weights that were pruned (mask == 0).
    Call this after each optimiser step during fine-tuning to keep
    the network sparse and prevent pruned edges from coming back.

    Usage inside a fine-tuning loop:
        optimizer.step()
        apply_masks(model)
    """
    for name, module in model.named_modules():
        if hasattr(module, 'weight_mask') and module.weight_mask is not None:
            with torch.no_grad():
                module.weight.data *= module.weight_mask


# =============================================================================
# Existing helpers — keep exactly as in pruning_logic.py
# =============================================================================

class SimpleMLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super().__init__()
        self.flatten  = nn.Flatten()
        self.fc1      = nn.Linear(input_size, hidden_size)
        self.relu     = nn.ReLU()
        self.dropout  = nn.Dropout(0.5)
        self.fc2      = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def _rebuild_model(original_model, new_fc1, new_fc2, device):
    class PrunedMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.fc1     = new_fc1
            self.relu    = nn.ReLU()
            self.dropout = nn.Dropout(0.5)
            self.fc2     = new_fc2
        def forward(self, x):
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    return PrunedMLP().to(device)


def _create_pruned_model(model, layer_name, keep_indices, device):
    """Creates a new, physically smaller model with specified neurons removed."""
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


def train_model(model, train_loader, test_loader, device, epochs=10,
                respect_masks=False, lr=0.001):
    """
    Train model and return final accuracy.

    Set respect_masks=True during fine-tuning of edge-pruned models
    so that zeroed weights cannot be revived by gradient updates.
    """
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
                apply_masks(model)   # re-zero pruned weights after every step

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total   += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total


def load_dataset(dataset_name):
    if dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_ds = datasets.CIFAR10('./data', train=True,  download=False, transform=transform)
        test_ds  = datasets.CIFAR10('./data', train=False, download=False, transform=transform)
        input_size = 32 * 32 * 3
    else:
        cfg = {
            'MNIST':        (datasets.MNIST,        (0.1307,), (0.3081,)),
            'FashionMNIST': (datasets.FashionMNIST, (0.5,),    (0.5,)),
            'KMNIST':       (datasets.KMNIST,        (0.5,),    (0.5,)),
        }
        if dataset_name not in cfg:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        dataset_class, mean, std = cfg[dataset_name]
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])
        train_ds   = dataset_class('./data', train=True,  download=False, transform=transform)
        test_ds    = dataset_class('./data', train=False, download=False, transform=transform)
        input_size = 28 * 28
    return train_ds, test_ds, input_size, 10


# =============================================================================
# Updated run_experiment — now handles 4 methods:
#   'spectral'      -> SpectralPruner  (node pruning, R value)
#   'random'        -> RandomPruner    (node pruning, random)
#   'spectral_edge' -> SpectralEdgePruner (edge pruning, R² value)
#   'random_edge'   -> RandomEdgePruner   (edge pruning, random baseline)
# =============================================================================

def run_experiment(params):
    dataset_name, prune_ratio, pruning_method, device_id = params
    device = torch.device(
        f'cuda:{device_id}' if torch.cuda.is_available() and torch.cuda.device_count() > 0
        else 'cpu'
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
        original_model   = SimpleMLP(input_size=input_size, hidden_size=256,
                                     num_classes=num_classes)
        original_accuracy = train_model(original_model, train_loader, test_loader,
                                        device, epochs=10)
        original_params  = sum(p.numel() for p in original_model.parameters())
        print(f"--- ({exp_id}) Baseline accuracy: {original_accuracy:.2f}% "
              f"({original_params:,} params) ---")

        # --- Pruning ---
        print(f"--- ({exp_id}) Pruning ---")

        if pruning_method == 'spectral':
            # NODE pruning with R value (AEI)
            pruner       = SpectralPruner(original_model, device)
            pruned_model = pruner.prune_layer('fc1', prune_ratio, train_loader)
            is_edge_prune = False

        elif pruning_method == 'random':
            # NODE pruning randomly
            from pruning_logic import RandomPruner
            pruner       = RandomPruner(original_model, device)
            pruned_model = pruner.prune_layer('fc1', prune_ratio)
            is_edge_prune = False

        elif pruning_method == 'spectral_edge':
            # EDGE pruning with R² value
            pruner       = SpectralEdgePruner(original_model, device)
            pruned_model = pruner.prune_layer('fc1', prune_ratio, train_loader)
            is_edge_prune = True

        elif pruning_method.startswith('random_edge'):
            # Extract trial number as seed so each trial is reproducible but different
            trial_seed   = int(pruning_method.replace('random_edge_trial', '')) if 'trial' in pruning_method else None
            pruner       = RandomEdgePruner(original_model, device)
            pruned_model = pruner.prune_layer('fc1', prune_ratio, trial_seed=trial_seed)
            is_edge_prune = True
        else:
            raise ValueError(f"Unknown pruning method: {pruning_method}")

        # --- Fine-tuning ---
        # For edge pruning, pass respect_masks=True so pruned weights
        # cannot recover during fine-tuning gradient updates.
        print(f"--- ({exp_id}) Fine-tuning pruned model ---")
        finetuned_accuracy = train_model(
            pruned_model, train_loader, test_loader, device,
            epochs=5, respect_masks=is_edge_prune, lr=1e-4
        )

        pruned_params = sum(p.numel() for p in pruned_model.parameters())
        print(f"--- ({exp_id}) Fine-tuned accuracy: {finetuned_accuracy:.2f}% "
              f"({pruned_params:,} params) ---")

        return {
            'dataset':      dataset_name,
            'prune_ratio':  f"{prune_ratio*100:.0f}%",
            'method': 'random_edge' if pruning_method.startswith('random_edge') else pruning_method,
            'original_acc': f"{original_accuracy:.2f}%",
            'finetuned_acc':f"{finetuned_accuracy:.2f}%",
            'acc_drop':     f"{original_accuracy - finetuned_accuracy:.2f}%",
            'compression':  f"{(1 - pruned_params / original_params) * 100:.1f}%"
        }

    except Exception as e:
        print(f"!!! ERROR during experiment {exp_id}: {e} !!!")
        import traceback
        traceback.print_exc()
        return None