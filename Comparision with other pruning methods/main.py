import torch
import torch.multiprocessing
from torchvision import datasets
from pruning_logic import run_experiment

def main():
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"=========================================================")
    print(f"       Running Network Pruning on device: {device_name.upper()}")
    print(f"=========================================================\n")

    datasets_to_test = ['FashionMNIST']
    prune_ratios     = [0.2, 0.3, 0.4]

    # --- Pre-download datasets ---
    print("--- Pre-downloading datasets... ---")
    for name in datasets_to_test:
        try:
            if name == 'FashionMNIST':
                datasets.FashionMNIST('./data', train=True,  download=True)
                datasets.FashionMNIST('./data', train=False, download=True)
            print(f"  Dataset '{name}' is ready.")
        except Exception as e:
            print(f"  Could not download {name}: {e}")
    print("--- All datasets ready. ---\n")

    device_id = 0  # single GPU

    # -----------------------------------------------------------------------
    # Deterministic methods only — no random trials
    # Node-level: spectral, l1_norm, l2_norm, snip, grasp
    # Edge-level: spectral_edge
    # -----------------------------------------------------------------------
    node_methods = ['spectral', 'l1_norm', 'l2_norm', 'snip', 'grasp']

    experiment_params = []
    for dataset in datasets_to_test:
        for prune_ratio in prune_ratios:
            for method in node_methods:
                experiment_params.append((dataset, prune_ratio, method, device_id))
            experiment_params.append((dataset, prune_ratio, 'spectral_edge', device_id))

    # -----------------------------------------------------------------------
    # Run sequentially
    # -----------------------------------------------------------------------
    results_log = []
    for params in experiment_params:
        result = run_experiment(params)
        if result:
            results_log.append(result)

    # -----------------------------------------------------------------------
    # Sort: dataset -> ratio -> method
    # -----------------------------------------------------------------------
    method_order = {
        'spectral':      0,
        'l1_norm':       1,
        'l2_norm':       2,
        'snip':          3,
        'grasp':         4,
        'spectral_edge': 5,
    }
    results_log.sort(key=lambda r: (
        r['dataset'], r['prune_ratio'],
        method_order.get(r['method'], 99)
    ))

    # -----------------------------------------------------------------------
    # Print results table
    # -----------------------------------------------------------------------
    W = 115
    print("\n" + "=" * W)
    print(" " * 35 + "OVERALL PRUNING COMPARISON RESULTS")
    print("=" * W)
    print(f"{'Dataset':<14} | {'Ratio':<6} | {'Method':<16} | "
          f"{'Baseline Acc':<14} | {'Finetuned Acc':<14} | "
          f"{'Acc Drop':<10} | {'Compression':<18}")
    print("-" * W)

    prev_ratio = None
    for res in results_log:
        if res['prune_ratio'] != prev_ratio:
            if prev_ratio is not None:
                print("-" * W)
            prev_ratio = res['prune_ratio']

        print(f"{res['dataset']:<14} | {res['prune_ratio']:<6} | {res['method']:<16} | "
              f"{res['original_acc']:<14} | {res['finetuned_acc']:<14} | "
              f"{res['acc_drop']:<10} | {res['compression']:<18}")

    print("=" * W)
    print("\nMethod key:")
    print("  spectral      = Spectral node pruning (AEI R value)  [PROPOSED]")
    print("  l1_norm       = L1 norm node pruning                 [BASELINE]")
    print("  l2_norm       = L2 norm node pruning                 [BASELINE]")
    print("  snip          = SNIP (connection sensitivity)        [BASELINE]")
    print("  grasp         = GraSP (gradient signal preservation) [BASELINE]")
    print("  spectral_edge = Spectral edge pruning (R² value)     [EXTENSION]")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()