import torch
import torch.multiprocessing
from torchvision import datasets
from itertools import product
from concurrent.futures import ProcessPoolExecutor

# Import the main worker function from your logic file
from pruning_logic import run_experiment

def main():
    """Main function to orchestrate the pruning experiments."""
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"=========================================================")
    print(f"       Running Network Pruning on device: {device_name.upper()}        ")
    print(f"=========================================================\n")
    datasets_to_test = ['MNIST', 'FashionMNIST', 'CIFAR10', 'KMNIST']
    prune_ratios = [0.2, 0.3, 0.4]
    pruning_methods = ['spectral', 'random']
    print("--- Pre-downloading all datasets... ---")
    for name in datasets_to_test:
        try:
            if name == 'MNIST': datasets.MNIST('./data', train=True, download=True)
            elif name == 'FashionMNIST': datasets.FashionMNIST('./data', train=True, download=True)
            elif name == 'CIFAR10': datasets.CIFAR10('./data', train=True, download=True)
            elif name == 'KMNIST': datasets.KMNIST('./data', train=True, download=True)
            print(f"Dataset '{name}' is ready.")
        except Exception as e:
            print(f"Could not download {name}. Error: {e}")
    print("--- All datasets are downloaded and ready. ---\n")
    num_gpus = torch.cuda.device_count()
    experiment_params = list(product(datasets_to_test, prune_ratios, pruning_methods))
    experiment_params_with_device = [
        (*params, i % num_gpus if num_gpus > 0 else 0) 
        for i, params in enumerate(experiment_params)
    ]
    results_log = []
    max_workers = num_gpus if num_gpus > 0 else 4 
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(run_experiment, experiment_params_with_device)
        for result in results:
            if result:
                results_log.append(result)
    results_log.sort(key=lambda r: (r['dataset'], r['method'], r['prune_ratio']))
    print("\n" + "="*95)
    print(" " * 30 + "OVERALL PRUNING COMPARISON RESULTS")
    print("="*95)
    print(f"{'Dataset':<15} | {'Prune Ratio':<12} | {'Method':<10} | {'Original Acc':<15} | {'Finetuned Acc':<15} | {'Acc Drop':<12} | {'Compression':<12}")
    print("-"*95)
    for res in results_log:
        print(f"{res['dataset']:<15} | {res['prune_ratio']:<12} | {res['method']:<10} | "
              f"{res['original_acc']:<15} | {res['finetuned_acc']:<15} | "
              f"{res['acc_drop']:<12} | {res['compression']:<12}")
    print("-"*95)
if __name__ == "__main__":
    # This check is crucial for multiprocessing to work correctly.
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()