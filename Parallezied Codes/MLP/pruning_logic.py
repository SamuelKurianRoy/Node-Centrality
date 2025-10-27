import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from scipy.linalg import eigh
import copy

# =============================================================================
# Pruning Method Implementations
# =============================================================================

class SpectralPruner:
    # ... (rest of the SpectralPruner class code is identical to before)
    """
    Network pruning using spectral graph theory to identify structurally
    important neurons via the Fiedler vector.
    """

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.activation_storage = {}
        self.hooks = []

    def register_hooks(self, layer_names):
        """Register forward hooks to capture activations for specified layers."""
        def get_activation(name):
            def hook(module, input, output):
                self.activation_storage[name] = output.detach()
            return hook

        for name, module in self.model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(get_activation(name))
                self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def compute_fiedler_importance(self, activations):
        """Compute importance scores for neurons using the Fiedler vector."""
        activations = activations.view(activations.size(0), -1).cpu().numpy()
        corr_matrix = np.corrcoef(activations.T)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        adjacency = np.abs(corr_matrix)
        degree_vector = np.sum(adjacency, axis=1)
        degree_vector = np.maximum(degree_vector, 1e-8)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degree_vector))
        laplacian = np.eye(len(degree_vector)) - D_inv_sqrt @ adjacency @ D_inv_sqrt
        eigenvalues, eigenvectors = eigh(laplacian)
        fiedler_vector = eigenvectors[:, 1]
        importance_scores = np.abs(fiedler_vector)
        return importance_scores

    def collect_activations(self, dataloader, layer_name, num_batches=10):
        """Collect activations from a layer by running inference on data."""
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
        activations = torch.cat(activations_list, dim=0)
        return activations

    def prune_layer(self, layer_name, prune_ratio, dataloader, num_batches=20):
        """Prune a specific layer by removing the least important neurons."""
        activations = self.collect_activations(dataloader, layer_name, num_batches)
        importance_scores = self.compute_fiedler_importance(activations)
        num_neurons = len(importance_scores)
        num_to_keep = int(num_neurons * (1 - prune_ratio))
        important_indices = np.argsort(importance_scores)[-num_to_keep:]
        keep_indices = np.sort(important_indices)
        print(f"Spectrally pruning layer '{layer_name}': {num_neurons} -> {num_to_keep} neurons "
              f"({prune_ratio*100:.1f}% pruned)")
        return _create_pruned_model(self.model, layer_name, keep_indices, self.device)

class RandomPruner:
    # ... (rest of the RandomPruner class code is identical to before)
    """
    Network pruning by randomly selecting neurons to remove.
    """
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device

    def prune_layer(self, layer_name, prune_ratio):
        """Prune a specific layer by randomly removing neurons."""
        model_to_prune = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        layer_to_prune = dict(model_to_prune.named_modules())[layer_name]
        
        num_neurons = layer_to_prune.out_features
        num_to_keep = int(num_neurons * (1 - prune_ratio))
        all_indices = np.arange(num_neurons)
        keep_indices = np.sort(np.random.choice(all_indices, num_to_keep, replace=False))
        print(f"Randomly pruning layer '{layer_name}': {num_neurons} -> {num_to_keep} neurons "
              f"({prune_ratio*100:.1f}% pruned)")
        return _create_pruned_model(self.model, layer_name, keep_indices, self.device)

# =============================================================================
# Model Definition and Helper Functions
# =============================================================================
class SimpleMLP(nn.Module):
    # ... (rest of the SimpleMLP class code is identical to before)
    """Simple MLP for classification."""
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def _rebuild_model(original_model, new_fc1, new_fc2, device):
    # ... (rest of the _rebuild_model function code is identical to before)
    """Helper to reconstruct the MLP with new layers."""
    class PrunedMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.fc1 = new_fc1
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.5)
            self.fc2 = new_fc2

        def forward(self, x):
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    return PrunedMLP().to(device)

def _create_pruned_model(model, layer_name, keep_indices, device):
    # ... (rest of the _create_pruned_model function code is identical to before)
    """Creates a new model with specified neurons removed from a layer."""
    if layer_name != 'fc1':
        raise NotImplementedError("This script only supports pruning the 'fc1' layer.")

    original_model = model.module if isinstance(model, nn.DataParallel) else model

    layer_to_prune = original_model.fc1
    next_layer = original_model.fc2
    
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

def train_model(model, train_loader, test_loader, device, epochs=10):
    # ... (rest of the train_model function code is identical to before)
    """Train the model and return final accuracy."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total

def load_dataset(dataset_name):
    # ... (rest of the load_dataset function code is identical to before)
    """Load specified dataset with appropriate transforms."""
    if dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_ds = datasets.CIFAR10('./data', train=True, download=False, transform=transform)
        test_ds = datasets.CIFAR10('./data', train=False, download=False, transform=transform)
        input_size = 32 * 32 * 3
    else:
        if dataset_name == 'MNIST':
            normalize = transforms.Normalize((0.1307,), (0.3081,))
            dataset_class = datasets.MNIST
        elif dataset_name == 'FashionMNIST':
            normalize = transforms.Normalize((0.5,), (0.5,))
            dataset_class = datasets.FashionMNIST
        elif dataset_name == 'KMNIST':
            normalize = transforms.Normalize((0.5,), (0.5,))
            dataset_class = datasets.KMNIST
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        train_ds = dataset_class('./data', train=True, download=False, transform=transform)
        test_ds = dataset_class('./data', train=False, download=False, transform=transform)
        input_size = 28 * 28

    return train_ds, test_ds, input_size, 10

# =============================================================================
# Main Experiment Runner
# =============================================================================
def run_experiment(params):
    # ... (rest of the run_experiment function code is identical to before)
    """Main execution: Train, prune, fine-tune, and compare."""
    dataset_name, prune_ratio, pruning_method, device_id = params
    
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu')
    
    exp_id = f"{dataset_name}/{prune_ratio*100:.0f}%/{pruning_method.capitalize()}"
    print(f"Starting experiment: {exp_id} on {str(device).upper()}")

    try:
        train_dataset, test_dataset, input_size, num_classes = load_dataset(dataset_name)
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2, pin_memory=True)

        print(f"--- ({exp_id}) Training Original Model ---")
        original_model = SimpleMLP(input_size=input_size, hidden_size=256, num_classes=num_classes)
        original_accuracy = train_model(original_model, train_loader, test_loader, device, epochs=10)
        print(f"--- ({exp_id}) Original Model Accuracy: {original_accuracy:.2f}% ---")

        print(f"--- ({exp_id}) Pruning Model ---")
        if pruning_method == 'spectral':
            pruner = SpectralPruner(original_model, device)
            pruned_model = pruner.prune_layer('fc1', prune_ratio, train_loader)
        elif pruning_method == 'random':
            pruner = RandomPruner(original_model, device)
            pruned_model = pruner.prune_layer('fc1', prune_ratio)
        else:
            raise ValueError(f"Unknown pruning method: {pruning_method}")

        print(f"--- ({exp_id}) Fine-tuning Pruned Model ---")
        finetuned_accuracy = train_model(pruned_model, train_loader, test_loader, device, epochs=5)
        print(f"--- ({exp_id}) Pruned Model Accuracy (after fine-tuning): {finetuned_accuracy:.2f}% ---")

        original_params = sum(p.numel() for p in original_model.parameters())
        pruned_params = sum(p.numel() for p in pruned_model.parameters())
        
        return {
            'dataset': dataset_name,
            'prune_ratio': f"{prune_ratio*100:.0f}%",
            'method': pruning_method.capitalize(),
            'original_acc': f"{original_accuracy:.2f}%",
            'finetuned_acc': f"{finetuned_accuracy:.2f}%",
            'acc_drop': f"{original_accuracy - finetuned_accuracy:.2f}%",
            'compression': f"{(1 - pruned_params / original_params) * 100:.1f}%"
        }
    except Exception as e:
        print(f"!!! ERROR during experiment {exp_id}: {e} !!!")
        import traceback
        traceback.print_exc()
        return None