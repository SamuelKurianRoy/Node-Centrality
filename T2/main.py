import torch
import torch.multiprocessing
from torchvision import datasets
from concurrent.futures import ProcessPoolExecutor
from pruning_logic import run_experiment

def main():
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"=========================================================")
    print(f"       Running Network Pruning on device: {device_name.upper()}")
    print(f"=========================================================\n")

    datasets_to_test = ['MNIST']
    prune_ratios = [0.2, 0.3, 0.4]
    num_random_trials = 10

    print("--- Pre-downloading datasets... ---")
    for name in datasets_to_test:
        try:
            if name == 'MNIST':
                datasets.MNIST('./data', train=True, download=True)
                datasets.MNIST('./data', train=False, download=True)
            print(f"Dataset '{name}' is ready.")
        except Exception as e:
            print(f"Could not download {name}. Error: {e}")
    print("--- All datasets downloaded. ---\n")

    num_gpus = torch.cuda.device_count()
    device_id = 0  # single GPU or CPU

    # Build experiment list manually so we can control random trials
    experiment_params = []
    for prune_ratio in prune_ratios:
        # 1 spectral edge run per prune ratio
        experiment_params.append(('MNIST', prune_ratio, 'spectral_edge', device_id))
        # 10 random edge runs per prune ratio, each with a different seed
        for trial in range(num_random_trials):
            experiment_params.append(('MNIST', prune_ratio, f'random_edge_trial{trial}', device_id))

    results_log = []

    # Use 1 worker if no GPU, otherwise use available GPUs
    max_workers = max(1, num_gpus)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(run_experiment, experiment_params)
        for result in results:
            if result:
                results_log.append(result)

    results_log.sort(key=lambda r: (r['dataset'], r['prune_ratio'], r['method']))

    print("\n" + "="*110)
    print(" " * 35 + "OVERALL PRUNING COMPARISON RESULTS")
    print("="*110)
    print(f"{'Dataset':<15} | {'Prune Ratio':<12} | {'Method':<20} | {'Original Acc':<15} | {'Finetuned Acc':<15} | {'Acc Drop':<12} | {'Compression':<12}")
    print("-"*110)
    for res in results_log:
        print(f"{res['dataset']:<15} | {res['prune_ratio']:<12} | {res['method']:<20} | "
              f"{res['original_acc']:<15} | {res['finetuned_acc']:<15} | "
              f"{res['acc_drop']:<12} | {res['compression']:<12}")
    print("-"*110)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()