#!/usr/bin/env python3
"""
run_edge_pruning_dgx.py

DGX-optimized edge pruning pipeline with multi-GPU parallelization:
- Spectral edge pruning using Fiedler vector (from jazz network approach)
- Random edge pruning baseline with parallel trials
- Efficient CPU/GPU resource utilization
- Comprehensive metrics: sparsity, FLOPs, inference speed, compression
- Progress saving to CSV and Excel
"""

import os
import sys
import time
import tempfile
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from scipy.linalg import eigh
import pandas as pd
import psutil
import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

# =============================================================================
# CONFIGURATION
# =============================================================================
DATASET_NAME = "MNIST"            # MNIST, FashionMNIST, KMNIST, CIFAR10
PRUNE_RATIOS = [0.3, 0.5, 0.7]   # Edge pruning ratios to test
NUM_RANDOM_TRIALS = 500           # Random edge pruning trials per ratio
TOP_K = 10                        # Top candidates to finetune
BASELINE_EPOCHS = 10
FINETUNE_EPOCHS = 5
HIDDEN_SIZE = 256
BATCH_SIZE_TRAIN = 256
BATCH_SIZE_TEST = 1000
DATA_LOADER_WORKERS = 8
ACTIVATION_BATCHES = 20           # Batches for activation collection
MIN_FREE_MEM_MB = 15000
MAX_UTIL_PCT = 40
LOG_EVERY = 50
TEMP_ROOT = "/scratch" if os.path.exists("/scratch") else None

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def chunk_ranges(total, n_chunks):
    """Partition total items into n_chunks ranges."""
    if n_chunks <= 0:
        return [range(total)]
    base = total // n_chunks
    rem = total % n_chunks
    chunks = []
    start = 0
    for i in range(n_chunks):
        size = base + (1 if i < rem else 0)
        end = start + size
        chunks.append(range(start, end))
        start = end
    return chunks

def get_available_gpus(min_free_mem_mb=MIN_FREE_MEM_MB, max_util_pct=MAX_UTIL_PCT):
    """Select available GPUs based on memory and utilization."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.total,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            universal_newlines=True,
        )
        available = []
        for line in out.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            idx, mem_total, mem_used, util = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            free = mem_total - mem_used
            if free >= min_free_mem_mb and util <= max_util_pct:
                available.append(idx)
        if not available:
            return list(range(torch.cuda.device_count()))
        return available
    except Exception:
        return list(range(torch.cuda.device_count()))

def count_nonzero_params(model):
    """Count non-zero parameters in model."""
    return sum(torch.count_nonzero(p).item() for p in model.parameters())

def measure_flops_and_params(model, input_shape, device):
    """Measure FLOPs and total parameters."""
    try:
        from thop import profile
        model.eval()
        dummy = torch.randn((1, *input_shape), device=device)
        with torch.no_grad():
            flops, params = profile(model, inputs=(dummy,), verbose=False)
        return flops, params
    except Exception as e:
        return None, None

def measure_inference_throughput(model, input_shape, device, batch_size=128, num_batches=10):
    """Measure inference throughput (images/sec)."""
    model.eval()
    dummy = torch.randn((batch_size, *input_shape), device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(num_batches):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.time() - t0
    return (batch_size * num_batches) / elapsed if elapsed > 0 else None

# =============================================================================
# MODEL DEFINITION
# =============================================================================
class SimpleMLP(nn.Module):
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

# =============================================================================
# DATASET LOADER
# =============================================================================
def load_dataset(name):
    """Load dataset and return train/test sets with input size."""
    if name == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
        test = datasets.CIFAR10("./data", train=False, download=True, transform=transform)
        input_size = 32 * 32 * 3
    else:
        dataset_class = {
            "MNIST": datasets.MNIST,
            "FashionMNIST": datasets.FashionMNIST,
            "KMNIST": datasets.KMNIST
        }[name]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train = dataset_class("./data", train=True, download=True, transform=transform)
        test = dataset_class("./data", train=False, download=True, transform=transform)
        input_size = 28 * 28
    return train, test, input_size, 10

# =============================================================================
# SPECTRAL EDGE PRUNING (FIEDLER METHOD)
# =============================================================================
class SpectralEdgePruner:
    """Spectral edge pruning using Fiedler vector from graph Laplacian."""
    
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
        """Collect activations from specified layer."""
        self.activation_storage = {}
        self.register_hooks([layer_name])
        self.model.eval()
        acts = []
        with torch.no_grad():
            for i, (data, _) in enumerate(dataloader):
                if i >= num_batches:
                    break
                data = data.to(self.device)
                _ = self.model(data)
                if layer_name in self.activation_storage:
                    acts.append(self.activation_storage[layer_name])
        self.remove_hooks()
        return torch.cat(acts, dim=0) if acts else None

    def compute_edge_importance_fiedler(self, layer_weight, activations):
        """
        Compute edge importance using Fiedler vector method.
        
        Similar to jazz network's adjacent edge index:
        1. Build neuron correlation graph from activations
        2. Compute Fiedler vector from graph Laplacian
        3. Edge importance = |fiedler[i] - fiedler[j]| × |weight[i,j]|
        """
        out_features, in_features = layer_weight.shape
        
        # Compute neuron correlation matrix
        activations_np = activations.view(activations.size(0), -1).cpu().numpy()
        corr_matrix = np.nan_to_num(np.corrcoef(activations_np.T))
        adjacency = np.abs(corr_matrix)
        
        # Compute normalized graph Laplacian
        degree_vector = np.sum(adjacency, axis=1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degree_vector, 1e-8)))
        laplacian = np.eye(len(degree_vector)) - D_inv_sqrt @ adjacency @ D_inv_sqrt
        
        # Get Fiedler vector (second smallest eigenvector)
        eigenvalues, eigenvectors = eigh(laplacian)
        fiedler_vector = eigenvectors[:, 1]
        
        # Compute edge importance for each weight
        edge_importance = np.zeros((out_features, in_features))
        
        for i in range(out_features):
            for j in range(in_features):
                if i < len(fiedler_vector) and j < len(fiedler_vector):
                    # Jazz network style: difference in Fiedler values
                    fiedler_diff = abs(fiedler_vector[i] - fiedler_vector[j])
                    weight_magnitude = abs(layer_weight[i, j].item())
                    edge_importance[i, j] = fiedler_diff * weight_magnitude
                else:
                    edge_importance[i, j] = abs(layer_weight[i, j].item())
        
        return edge_importance

    def prune_layer_edges(self, layer_name, prune_ratio, dataloader, num_batches=20):
        """Prune edges in specified layer using spectral method."""
        model_to_prune = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        layer = dict(model_to_prune.named_modules())[layer_name]
        
        print(f"Collecting activations for spectral edge importance...")
        activations = self.collect_activations(dataloader, layer_name, num_batches)
        
        if activations is None:
            print("Failed to collect activations, aborting pruning.")
            return self.model
        
        weight = layer.weight.data.cpu()
        
        print(f"Computing Fiedler-based edge importance...")
        edge_importance = self.compute_edge_importance_fiedler(weight, activations)
        
        # Determine pruning threshold
        total_edges = edge_importance.size
        num_to_prune = int(total_edges * prune_ratio)
        flat_importance = edge_importance.flatten()
        threshold = np.sort(flat_importance)[num_to_prune]
        
        # Create mask: keep high importance edges
        mask = (edge_importance >= threshold).astype(np.float32)
        mask_tensor = torch.from_numpy(mask).to(self.device)
        
        # Apply mask to weights
        layer.weight.data = layer.weight.data * mask_tensor
        layer.register_buffer('weight_mask', mask_tensor)
        
        actual_pruned = np.sum(mask == 0)
        print(f"Spectral edge pruning {layer_name}: {total_edges} edges → "
              f"{total_edges - actual_pruned} kept, {actual_pruned} pruned ({100*prune_ratio:.1f}%)")
        
        return self.model

# =============================================================================
# RANDOM EDGE PRUNING
# =============================================================================
class RandomEdgePruner:
    """Random edge pruning baseline."""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device

    def prune_layer_edges(self, layer_name, prune_ratio, trial_idx=None):
        """Randomly prune edges in specified layer."""
        model_to_prune = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        layer = dict(model_to_prune.named_modules())[layer_name]
        
        weight_shape = layer.weight.data.shape
        total_edges = layer.weight.data.numel()
        num_to_prune = int(total_edges * prune_ratio)
        
        # Reproducible random seed per trial
        if trial_idx is not None:
            np.random.seed(trial_idx)
            torch.manual_seed(trial_idx)
        
        # Create random mask
        mask = torch.ones_like(layer.weight.data)
        flat_mask = mask.view(-1)
        prune_indices = torch.randperm(total_edges)[:num_to_prune]
        flat_mask[prune_indices] = 0
        mask = flat_mask.view(weight_shape)
        
        # Apply mask
        layer.weight.data = layer.weight.data * mask
        layer.register_buffer('weight_mask', mask)
        
        return self.model

# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================
def train_model(model, train_loader, test_loader, device, epochs=10):
    """Train model and return metrics."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = torch.cuda.amp.GradScaler()

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    train_time = time.time() - start_time

    # Evaluate accuracy
    model.eval()
    correct, total = 0, 0
    with torch.no_grad(), torch.cuda.amp.autocast():
        for data, target in test_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    acc = 100 * correct / total

    return acc, train_time

def eval_accuracy(model, test_loader, device):
    """Quick accuracy evaluation."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = torch.max(output.data, 1)
            total += target.size(0)
            correct += (pred == target).sum().item()
    return 100.0 * correct / total if total > 0 else 0.0

# =============================================================================
# PARALLEL WORKER: RANDOM EDGE PRUNING EVALUATION
# =============================================================================
def worker_eval_random_edges(chunk_range, baseline_state_cpu, prune_ratio, input_size, 
                             num_classes, dataset_name, device_index, local_tempdir, 
                             top_k_local, log_every=LOG_EVERY):
    """
    Worker function: evaluate random edge pruning candidates.
    Runs on single GPU, saves top-k local candidates.
    """
    torch.set_num_threads(2)
    
    if device_index is not None and torch.cuda.is_available():
        torch.cuda.set_device(device_index)
        device = torch.device(f"cuda:{device_index}")
    else:
        device = torch.device("cpu")
    
    # Load dataset
    train_ds, test_ds, _, _ = load_dataset(dataset_name)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE_TEST, shuffle=False,
                            num_workers=DATA_LOADER_WORKERS, pin_memory=(device.type=="cuda"))
    
    local_top = []  # (acc, path, iter)
    
    num_trials = len(chunk_range)
    for idx in chunk_range:
        # Create model and load baseline
        model = SimpleMLP(input_size=input_size, hidden_size=HIDDEN_SIZE, 
                         num_classes=num_classes).to(device)
        model.load_state_dict({k: v.to(device) for k, v in baseline_state_cpu.items()}, 
                             strict=False)
        
        # Apply random edge pruning
        pruner = RandomEdgePruner(model, device)
        pruned_model = pruner.prune_layer_edges("fc1", prune_ratio, trial_idx=idx)
        
        # Evaluate
        acc = eval_accuracy(pruned_model, test_loader, device)
        
        # Save if in top-k
        if len(local_top) < top_k_local or acc > local_top[0][0]:
            fname = os.path.join(local_tempdir, 
                               f"cand_dev{device_index}_iter{idx}_pr{int(prune_ratio*100)}.pt")
            state_dict = {k: v.cpu() for k, v in pruned_model.state_dict().items()}
            torch.save({"state_dict": state_dict, "acc": acc}, fname)
            
            local_top.append((acc, fname, idx))
            local_top.sort(key=lambda x: x[0])
            if len(local_top) > top_k_local:
                rem = local_top.pop(0)
                try:
                    os.remove(rem[1])
                except:
                    pass
        
        if (idx + 1) % log_every == 0:
            print(f"[GPU {device_index}] Completed {idx + 1}/{num_trials} trials")
    
    return [{"acc": float(a), "path": p, "iter": int(i), "device_index": device_index} 
            for (a, p, i) in local_top]

# =============================================================================
# PARALLEL WORKER: FINETUNE CANDIDATE
# =============================================================================
def worker_finetune_candidate(candidate_meta, baseline_state_cpu, prune_ratio, 
                              input_size, num_classes, dataset_name, device_index, 
                              finetune_epochs):
    """Worker function: finetune a pruned candidate and return metrics."""
    torch.set_num_threads(2)
    
    if device_index is not None and torch.cuda.is_available():
        torch.cuda.set_device(device_index)
        device = torch.device(f"cuda:{device_index}")
    else:
        device = torch.device("cpu")
    
    # Load dataset
    train_ds, test_ds, _, _ = load_dataset(dataset_name)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE_TRAIN, shuffle=True,
                             num_workers=DATA_LOADER_WORKERS, pin_memory=(device.type=="cuda"))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE_TEST, shuffle=False,
                            num_workers=DATA_LOADER_WORKERS, pin_memory=(device.type=="cuda"))
    
    # Load candidate checkpoint
    ckpt = torch.load(candidate_meta["path"], map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    
    # Create model and load state
    model = SimpleMLP(input_size=input_size, hidden_size=HIDDEN_SIZE, 
                     num_classes=num_classes).to(device)
    model.load_state_dict({k: v.to(device) for k, v in state.items()}, strict=False)
    
    # Compute pre-finetune metrics
    flops_before, params_before = measure_flops_and_params(model, (input_size,), device)
    nonzero_before = count_nonzero_params(model)
    sparsity_before = 1 - (nonzero_before / sum(p.numel() for p in model.parameters()))
    
    # Finetune
    acc_ft, train_time = train_model(model, train_loader, test_loader, device, 
                                    epochs=finetune_epochs)
    
    # Post-finetune metrics
    flops_after, params_after = measure_flops_and_params(model, (input_size,), device)
    nonzero_after = count_nonzero_params(model)
    sparsity_after = 1 - (nonzero_after / sum(p.numel() for p in model.parameters()))
    compression_rate = (params_before / params_after) if (params_before and params_after) else None
    inf_throughput = measure_inference_throughput(model, (input_size,), device)
    
    return {
        "Iteration": int(candidate_meta.get("iter", -1)) + 1,
        "Method": "Random_Edge",
        "Prune_Ratio": float(prune_ratio),
        "Baseline_Candidate_Acc": float(candidate_meta.get("acc", 0.0)),
        "Accuracy_After_Finetuning": acc_ft,
        "Candidate_Path": candidate_meta["path"],
        "Device": device_index,
        "Training_Time_sec": train_time,
        "Sparsity": sparsity_after,
        "Compression_Rate": compression_rate,
        "FLOPs": flops_after,
        "Params": params_after,
        "Nonzero_Params": nonzero_after,
        "Inference_Speed_img_per_s": inf_throughput,
    }

# =============================================================================
# SPECTRAL EDGE PRUNING (MAIN PROCESS)
# =============================================================================
def spectral_edge_prune_and_metrics(prune_ratio, baseline_model, baseline_acc, 
                                   train_loader, test_loader, device, input_size, 
                                   num_classes, dataset_name, finetune_epochs):
    """Run spectral edge pruning and return metrics."""
    model_copy = SimpleMLP(input_size=input_size, hidden_size=HIDDEN_SIZE, 
                          num_classes=num_classes).to(device)
    model_copy.load_state_dict(baseline_model.state_dict(), strict=False)
    
    # Apply spectral pruning
    pruner = SpectralEdgePruner(model_copy, device)
    pruned_model = pruner.prune_layer_edges("fc1", prune_ratio, train_loader, 
                                           num_batches=ACTIVATION_BATCHES)
    
    # Evaluate before finetune
    acc_pruned = eval_accuracy(pruned_model, test_loader, device)
    
    # Finetune
    acc_ft, train_time = train_model(pruned_model, train_loader, test_loader, 
                                    device, epochs=finetune_epochs)
    
    # Compute metrics
    flops_after, params_after = measure_flops_and_params(pruned_model, (input_size,), device)
    nonzero_after = count_nonzero_params(pruned_model)
    sparsity_after = 1 - (nonzero_after / sum(p.numel() for p in pruned_model.parameters()))
    inf_throughput = measure_inference_throughput(pruned_model, (input_size,), device)
    
    return {
        "Iteration": 1,
        "Method": "Spectral_Edge",
        "Prune_Ratio": float(prune_ratio),
        "Baseline_Accuracy": float(baseline_acc),
        "Accuracy_After_Pruning": float(acc_pruned),
        "Accuracy_After_Finetuning": float(acc_ft),
        "Training_Time_sec": train_time,
        "Sparsity": sparsity_after,
        "FLOPs": flops_after,
        "Params": params_after,
        "Nonzero_Params": nonzero_after,
        "Inference_Speed_img_per_s": inf_throughput,
    }

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    print("="*70)
    print("DGX EDGE PRUNING PIPELINE - Spectral vs Random")
    print("="*70)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"CPUs: {psutil.cpu_count(logical=True)}, RAM: {psutil.virtual_memory().total/1e9:.2f} GB")
    
    # Detect GPUs
    devices = get_available_gpus()
    if not devices:
        print("WARNING: No available GPUs detected. Running on CPU (slow).")
        devices = []
    else:
        print(f"Selected GPUs: {devices}")
    
    num_workers = max(1, len(devices))
    print(f"Using {num_workers} worker processes")
    
    # Create temp directory
    tmp_root = tempfile.mkdtemp(prefix="edge_prune_tmp_") if TEMP_ROOT is None \
              else tempfile.mkdtemp(prefix="edge_prune_tmp_", dir=TEMP_ROOT)
    print(f"Temp directory: {tmp_root}")
    
    # Load dataset and train baseline
    print("\n" + "="*70)
    print("LOADING DATASET AND TRAINING BASELINE")
    print("="*70)
    
    train_ds, test_ds, input_size, num_classes = load_dataset(DATASET_NAME)
    train_loader_main = DataLoader(train_ds, batch_size=BATCH_SIZE_TRAIN, shuffle=True,
                                  num_workers=DATA_LOADER_WORKERS, pin_memory=bool(devices))
    test_loader_main = DataLoader(test_ds, batch_size=BATCH_SIZE_TEST, shuffle=False,
                                 num_workers=DATA_LOADER_WORKERS, pin_memory=bool(devices))
    
    # Baseline device
    baseline_dev = torch.device(f"cuda:{devices[0]}") if devices else torch.device("cpu")
    if devices:
        torch.cuda.set_device(devices[0])
    
    print(f"Training baseline model on {baseline_dev} for {BASELINE_EPOCHS} epochs...")
    baseline_model = SimpleMLP(input_size=input_size, hidden_size=HIDDEN_SIZE, 
                              num_classes=num_classes).to(baseline_dev)
    baseline_acc, baseline_train_time = train_model(baseline_model, train_loader_main, 
                                                    test_loader_main, baseline_dev, 
                                                    epochs=BASELINE_EPOCHS)
    
    flops_base, params_base = measure_flops_and_params(baseline_model, (input_size,), baseline_dev)
    nonzero_base = count_nonzero_params(baseline_model)
    sparsity_base = 1 - (nonzero_base / sum(p.numel() for p in baseline_model.parameters()))
    
    baseline_metrics = {
        "Dataset": DATASET_NAME,
        "Params": params_base,
        "Nonzero_Params": nonzero_base,
        "Sparsity": sparsity_base,
        "FLOPs": flops_base,
        "Test_Accuracy": baseline_acc,
        "Training_Time_sec": baseline_train_time,
    }
    print(f"Baseline metrics: {baseline_metrics}")
    
    # Save baseline state to CPU
    baseline_state_cpu = {k: v.cpu() for k, v in baseline_model.state_dict().items()}
    
    # Results storage
    aggregated_random_eval = []
    aggregated_random_finetune = []
    aggregated_spectral = []
    
    # Process each prune ratio
    for prune_ratio in PRUNE_RATIOS:
        print("\n" + "="*70)
        print(f"PROCESSING PRUNE RATIO: {prune_ratio:.2f}")
        print("="*70)
        
        # Partition random trials across workers
        chunks = chunk_ranges(NUM_RANDOM_TRIALS, num_workers)
        print(f"Total trials: {NUM_RANDOM_TRIALS}, Workers: {num_workers}")
        print(f"Chunk sizes: {[len(c) for c in chunks]}")
        
        # Create worker temp directories
        worker_tempdirs = []
        for i in range(num_workers):
            d = os.path.join(tmp_root, f"worker_{i}_pr{int(prune_ratio*100)}")
            os.makedirs(d, exist_ok=True)
            worker_tempdirs.append(d)
        
        # PHASE 1: FAST RANDOM EVALUATION (PARALLEL)
        print("\n--- Phase 1: Fast Random Edge Pruning Evaluation ---")
        fast_start = time.time()
        all_local_candidates = []
        
        if num_workers == 1 and not devices:
            # CPU fallback
            local = worker_eval_random_edges(chunks[0], baseline_state_cpu, prune_ratio, 
                                           input_size, num_classes, DATASET_NAME, None, 
                                           worker_tempdirs[0], TOP_K)
            all_local_candidates.extend(local)
        else:
            with ProcessPoolExecutor(max_workers=num_workers) as exe:
                futures = []
                for i, chunk in enumerate(chunks):
                    dev = devices[i % len(devices)]
                    futures.append(exe.submit(worker_eval_random_edges, chunk, 
                                            baseline_state_cpu, prune_ratio, input_size, 
                                            num_classes, DATASET_NAME, dev, 
                                            worker_tempdirs[i], TOP_K))
                for fut in as_completed(futures):
                    try:
                        res = fut.result()
                        all_local_candidates.extend(res)
                    except Exception as e:
                        print(f"Worker error during eval: {e}")
        
        fast_elapsed = time.time() - fast_start
        print(f"Fast eval completed in {fast_elapsed/60.0:.2f} minutes")
        print(f"Collected {len(all_local_candidates)} local candidates")
        
        # Select global top-K
        all_local_candidates.sort(key=lambda x: x["acc"], reverse=True)
        top_candidates = all_local_candidates[:TOP_K]
        print(f"Selected global top {len(top_candidates)} candidates")
        if top_candidates:
            print(f"Best candidate accuracy: {top_candidates[0]['acc']:.2f}%")
        
        # Save random eval results
        for c in top_candidates:
            aggregated_random_eval.append({
                "Iteration": c["iter"] + 1,
                "Method": "Random_Edge",
                "Prune_Ratio": prune_ratio,
                "Baseline_Accuracy": baseline_acc,
                "Accuracy_After_Pruning": c["acc"],
                "Candidate_Path": c["path"],
                "Device": c["device_index"],
            })
        
        # PHASE 2: FINETUNE TOP CANDIDATES (PARALLEL)
        print("\n--- Phase 2: Finetuning Top Candidates ---")
        finetune_start = time.time()
        finetune_results = []
        
        if top_candidates:
            if not devices:
                # CPU fallback
                for cand in top_candidates:
                    res = worker_finetune_candidate(cand, baseline_state_cpu, prune_ratio,
                                                   input_size, num_classes, DATASET_NAME,
                                                   None, FINETUNE_EPOCHS)
                    finetune_results.append(res)
            else:
                with ProcessPoolExecutor(max_workers=min(len(top_candidates), len(devices))) as exe:
                    futures = []
                    for i, cand in enumerate(top_candidates):
                        dev = devices[i % len(devices)]
                        futures.append(exe.submit(worker_finetune_candidate, cand, 
                                                baseline_state_cpu, prune_ratio, input_size,
                                                num_classes, DATASET_NAME, dev, FINETUNE_EPOCHS))
                    for fut in as_completed(futures):
                        try:
                            res = fut.result()
                            finetune_results.append(res)
                        except Exception as e:
                            print(f"Finetune worker error: {e}")
        
        finetune_elapsed = time.time() - finetune_start
        print(f"Finetuning completed in {finetune_elapsed/60.0:.2f} minutes")
        print(f"Collected {len(finetune_results)} finetune records")
        
        aggregated_random_finetune.extend(finetune_results)
        
        # PHASE 3: SPECTRAL EDGE PRUNING
        print("\n--- Phase 3: Spectral Edge Pruning (Fiedler Method) ---")
        try:
            spec_res = spectral_edge_prune_and_metrics(prune_ratio, baseline_model, 
                                                      baseline_acc, train_loader_main, 
                                                      test_loader_main, baseline_dev, 
                                                      input_size, num_classes, DATASET_NAME,
                                                      FINETUNE_EPOCHS)
            aggregated_spectral.append(spec_res)
            print(f"Spectral edge pruning accuracy: {spec_res['Accuracy_After_Finetuning']:.2f}%")
        except Exception as e:
            print(f"Spectral pruning failed: {e}")
        
        # Save progress for this prune ratio
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv_eval = f"{DATASET_NAME}_random_edge_eval_pr{int(prune_ratio*100)}_{timestamp}.csv"
        out_csv_ft = f"{DATASET_NAME}_random_edge_finetune_pr{int(prune_ratio*100)}_{timestamp}.csv"
        out_csv_spec = f"{DATASET_NAME}_spectral_edge_pr{int(prune_ratio*100)}_{timestamp}.csv"
        
        df_eval = pd.DataFrame(aggregated_random_eval)
        df_ft = pd.DataFrame(aggregated_random_finetune)
        df_spec = pd.DataFrame(aggregated_spectral)
        
        df_eval.to_csv(out_csv_eval, index=False)
        df_ft.to_csv(out_csv_ft, index=False)
        df_spec.to_csv(out_csv_spec, index=False)
        print(f"Saved CSVs: {out_csv_eval}, {out_csv_ft}, {out_csv_spec}")
        
        # Save Excel workbook
        excel_path = f"edge_pruning_results_{DATASET_NAME}_pr{int(prune_ratio*100)}_{timestamp}.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            pd.DataFrame([baseline_metrics]).to_excel(writer, sheet_name="baseline", index=False)
            df_eval.to_excel(writer, sheet_name="random_eval", index=False)
            df_ft.to_excel(writer, sheet_name="random_finetune", index=False)
            if not df_spec.empty:
                df_spec.to_excel(writer, sheet_name="spectral_edge", index=False)
        print(f"Saved Excel: {excel_path}")
        
        # Cleanup worker temp dirs
        for d in worker_tempdirs:
            try:
                shutil.rmtree(d)
            except:
                pass
    
    # Final aggregated saves
    print("\n" + "="*70)
    print("SAVING FINAL RESULTS")
    print("="*70)
    
    final_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pd.DataFrame(aggregated_random_eval).to_csv(
        f"{DATASET_NAME}_random_edge_eval_all_{final_ts}.csv", index=False)
    pd.DataFrame(aggregated_random_finetune).to_csv(
        f"{DATASET_NAME}_random_edge_finetune_all_{final_ts}.csv", index=False)
    pd.DataFrame(aggregated_spectral).to_csv(
        f"{DATASET_NAME}_spectral_edge_all_{final_ts}.csv", index=False)
    
    # Final comprehensive Excel
    final_excel = f"edge_pruning_complete_{DATASET_NAME}_{final_ts}.xlsx"
    with pd.ExcelWriter(final_excel, engine="openpyxl") as writer:
        pd.DataFrame([baseline_metrics]).to_excel(writer, sheet_name="baseline", index=False)
        pd.DataFrame(aggregated_random_eval).to_excel(writer, sheet_name="random_eval_all", index=False)
        pd.DataFrame(aggregated_random_finetune).to_excel(writer, sheet_name="random_finetune_all", index=False)
        pd.DataFrame(aggregated_spectral).to_excel(writer, sheet_name="spectral_edge_all", index=False)
    
    print(f"All results saved to: {final_excel}")
    
    # Cleanup temp directory
    try:
        shutil.rmtree(tmp_root)
        print(f"Cleaned up temp directory: {tmp_root}")
    except Exception as e:
        print(f"Could not remove temp directory: {e}")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print(f"End time: {datetime.now().isoformat()}")
    print("="*70)

if __name__ == "__main__":
    main()