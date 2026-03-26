import torch
import torch.multiprocessing
from torchvision import datasets
from pruning_logic import run_experiment

def main():
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"=========================================================")
    print(f"       Running Network Pruning on device: {device_name.upper()}")
    print(f"=========================================================\n")

    datasets_to_test = ['MNIST']
    prune_ratios     = [0.2, 0.3, 0.4]
    num_random_trials = 5   # trials for random_node and random_edge baselines

    # --- Pre-download datasets ---
    print("--- Pre-downloading datasets... ---")
    for name in datasets_to_test:
        try:
            if name == 'MNIST':
                datasets.MNIST('./data', train=True,  download=True)
                datasets.MNIST('./data', train=False, download=True)
            print(f"  Dataset '{name}' is ready.")
        except Exception as e:
            print(f"  Could not download {name}: {e}")
    print("--- All datasets ready. ---\n")

    device_id = 0  # single GPU

    # -----------------------------------------------------------------------
    # Build experiment list
    # Node-level methods (single deterministic run each — no randomness):
    #   spectral, l1_norm, l2_norm, snip, grasp
    # Node-level random baseline (multiple trials for stable mean/std):
    #   random_node_trial{N}
    # Edge-level methods:
    #   spectral_edge, random_edge_trial{N}
    # -----------------------------------------------------------------------
    experiment_params = []

    # Deterministic node methods — 1 run per ratio
    node_methods = ['spectral', 'l1_norm', 'l2_norm', 'snip', 'grasp']

    for dataset in datasets_to_test:
        for prune_ratio in prune_ratios:
            # Deterministic node pruning methods
            for method in node_methods:
                experiment_params.append((dataset, prune_ratio, method, device_id))

            # Random node pruning — multiple trials
            for trial in range(num_random_trials):
                experiment_params.append(
                    (dataset, prune_ratio, f'random_node_trial{trial}', device_id)
                )

            # Spectral edge pruning
            experiment_params.append((dataset, prune_ratio, 'spectral_edge', device_id))

            # Random edge pruning — multiple trials
            for trial in range(num_random_trials):
                experiment_params.append(
                    (dataset, prune_ratio, f'random_edge_trial{trial}', device_id)
                )

    # -----------------------------------------------------------------------
    # Run sequentially (safe for single GPU / Colab / Jetson)
    # Switch to ProcessPoolExecutor on multi-GPU DGX
    # -----------------------------------------------------------------------
    results_log = []
    for params in experiment_params:
        result = run_experiment(params)
        if result:
            results_log.append(result)

    # -----------------------------------------------------------------------
    # Aggregate random trials -> mean ± std
    # -----------------------------------------------------------------------
    from collections import defaultdict
    import numpy as np

    # Separate out trial results
    trial_groups = defaultdict(list)  # key: (dataset, ratio, base_method)
    final_results = []

    for res in results_log:
        method = res['method']
        key    = (res['dataset'], res['prune_ratio'])

        if method in ('random_node', 'random_edge'):
            trial_groups[key + (method,)].append(float(res['finetuned_acc'].replace('%', '')))
        else:
            final_results.append(res)

    # Add aggregated random results
    for (dataset, ratio, base_method), accs in trial_groups.items():
        mean_acc = np.mean(accs)
        std_acc  = np.std(accs)
        # Find a representative result for other fields
        rep = next(r for r in results_log
                   if r['dataset'] == dataset
                   and r['prune_ratio'] == ratio
                   and r['method'] == base_method)
        final_results.append({
            'dataset':      dataset,
            'prune_ratio':  ratio,
            'method':       f"{base_method} (mean±std)",
            'original_acc': rep['original_acc'],
            'finetuned_acc':f"{mean_acc:.2f}% ± {std_acc:.2f}%",
            'acc_drop':     f"{float(rep['original_acc'].replace('%','')) - mean_acc:.2f}%",
            'compression':  rep['compression'],
        })

    # Sort: dataset -> ratio -> method
    method_order = {
        'spectral': 0, 'l1_norm': 1, 'l2_norm': 2,
        'snip': 3, 'grasp': 4, 'random_node (mean±std)': 5,
        'spectral_edge': 6, 'random_edge (mean±std)': 7,
    }
    final_results.sort(key=lambda r: (
        r['dataset'], r['prune_ratio'],
        method_order.get(r['method'], 99)
    ))

    # -----------------------------------------------------------------------
    # Print results table
    # -----------------------------------------------------------------------
    W = 125
    print("\n" + "=" * W)
    print(" " * 40 + "OVERALL PRUNING COMPARISON RESULTS")
    print("=" * W)
    print(f"{'Dataset':<12} | {'Ratio':<6} | {'Method':<28} | "
          f"{'Baseline Acc':<14} | {'Finetuned Acc':<22} | "
          f"{'Acc Drop':<10} | {'Compression':<18}")
    print("-" * W)

    prev_ratio = None
    for res in final_results:
        # Separator between sparsity groups
        if res['prune_ratio'] != prev_ratio:
            if prev_ratio is not None:
                print("-" * W)
            prev_ratio = res['prune_ratio']

        print(f"{res['dataset']:<12} | {res['prune_ratio']:<6} | {res['method']:<28} | "
              f"{res['original_acc']:<14} | {res['finetuned_acc']:<22} | "
              f"{res['acc_drop']:<10} | {res['compression']:<18}")

    print("=" * W)
    print("\nMethod key:")
    print("  spectral              = Spectral node pruning (AEI R value)  [PROPOSED]")
    print("  l1_norm               = L1 norm node pruning                 [BASELINE]")
    print("  l2_norm               = L2 norm node pruning                 [BASELINE]")
    print("  snip                  = SNIP (connection sensitivity)        [BASELINE]")
    print("  grasp                 = GraSP (gradient signal preservation) [BASELINE]")
    print("  random_node (mean±std)= Random node pruning                  [BASELINE]")
    print("  spectral_edge         = Spectral edge pruning (R² value)     [EXTENSION]")
    print("  random_edge (mean±std)= Random edge pruning                  [BASELINE]")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
