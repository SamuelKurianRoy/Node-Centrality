import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better-looking plots
sns.set_style("whitegrid")

# Load the data
df = pd.read_csv('MNIST_final_output.csv')

# Filter for Spectral 0.2 and Random 0.2
spectral_02 = df[(df['Method'] == 'Spectral') & (df['Prune_Ratio'] == 0.2)]
random_02 = df[(df['Method'] == 'Random') & (df['Prune_Ratio'] == 0.2)]

# Create a larger, more detailed figure
fig = plt.figure(figsize=(20, 10))

# 1. Main visualization: Bar chart for Spectral 0.2 with line overlay
ax1 = plt.subplot(1, 2, 1)

if len(spectral_02) > 0:
    spectral_sorted = spectral_02.sort_values('Iteration')
    experiment_nums = np.arange(1, len(spectral_sorted) + 1)
    
    # Bar chart
    bars = ax1.bar(experiment_nums, spectral_sorted['Accuracy_After_Finetuning'].values, 
                   alpha=0.6, color='darkred', edgecolor='darkred', linewidth=2, width=0.8,
                   label='Spectral 0.2 Accuracy')
    
    # Add value on top of bars
    for i, (bar, val) in enumerate(zip(bars, spectral_sorted['Accuracy_After_Finetuning'].values)):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.15, f'{val:.2f}%', 
                ha='center', va='bottom', fontsize=14, fontweight='bold', color='darkred')
    
    # Add line overlay
    ax1.plot(experiment_nums, spectral_sorted['Accuracy_After_Finetuning'].values, 
            color='darkred', linestyle='-', linewidth=3, marker='o', markersize=12, 
            markerfacecolor='lightcoral', markeredgecolor='darkred', markeredgewidth=2,
            label='Spectral 0.2 Trend')
    
    # Add Random 0.2 mean as horizontal line with band
    if len(random_02) > 0:
        random_mean = random_02['Accuracy_After_Finetuning'].mean()
        random_std = random_02['Accuracy_After_Finetuning'].std()
        
        ax1.axhline(y=random_mean, color='blue', linestyle='--', linewidth=3, 
                   label=f'Random 0.2 Mean: {random_mean:.2f}%')
        ax1.fill_between(experiment_nums, 
                        random_mean - random_std, 
                        random_mean + random_std,
                        alpha=0.25, color='blue', label=f'Random 0.2 Range (±{random_std:.2f}%)')
    
    ax1.set_xlabel('Experiment Number', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy After Finetuning (%)', fontsize=14, fontweight='bold')
    ax1.set_title('Spectral Method (Prune Ratio 0.2) vs Random Method\nAccuracy Comparison', 
                 fontsize=16, fontweight='bold', pad=20)
    ax1.legend(fontsize=12, loc='best', framealpha=0.95)
    ax1.grid(True, alpha=0.4, linestyle='--')
    ax1.set_ylim([97.3, 98.5])
    ax1.set_xticks(experiment_nums)
    
    # Add background color
    ax1.set_facecolor('#f8f9fa')

# 2. Statistical comparison
ax2 = plt.subplot(1, 2, 2)
ax2.axis('off')

# Create detailed statistics text
stats_text = ""
stats_text += "SPECTRAL 0.2 - DETAILED ANALYSIS\n"
stats_text += "=" * 50 + "\n\n"

if len(spectral_02) > 0:
    spec_acc = spectral_02['Accuracy_After_Finetuning'].values[0]
    spec_time = spectral_02['Training_Time_sec'].values[0]
    spec_baseline = spectral_02['Baseline_Accuracy'].values[0]
    spec_improvement = spec_acc - spec_baseline
    spec_params = spectral_02['Params'].values[0]
    spec_compression = spectral_02['Compression_Rate'].values[0]
    
    stats_text += "SPECTRAL 0.2 RESULTS:\n"
    stats_text += "-" * 50 + "\n"
    stats_text += f"  Iteration:              {spectral_02['Iteration'].values[0]:.0f}\n"
    stats_text += f"  Accuracy:               {spec_acc:.2f}%\n"
    stats_text += f"  Baseline Accuracy:      {spec_baseline:.2f}%\n"
    stats_text += f"  Improvement:            {spec_improvement:+.2f}%\n"
    stats_text += f"  Training Time:          {spec_time:.2f} seconds\n"
    stats_text += f"  Compression Rate:       {spec_compression:.4f}x\n"
    stats_text += f"  Parameters:             {spec_params:.0f}\n"

stats_text += "\n"
stats_text += "RANDOM 0.2 STATISTICS (1000 experiments):\n"
stats_text += "-" * 50 + "\n"

if len(random_02) > 0:
    rand_mean = random_02['Accuracy_After_Finetuning'].mean()
    rand_std = random_02['Accuracy_After_Finetuning'].std()
    rand_min = random_02['Accuracy_After_Finetuning'].min()
    rand_max = random_02['Accuracy_After_Finetuning'].max()
    rand_time_mean = random_02['Training_Time_sec'].mean()
    rand_time_std = random_02['Training_Time_sec'].std()
    
    stats_text += f"  Mean Accuracy:          {rand_mean:.2f}%\n"
    stats_text += f"  Std Deviation:          {rand_std:.2f}%\n"
    stats_text += f"  Min Accuracy:           {rand_min:.2f}%\n"
    stats_text += f"  Max Accuracy:           {rand_max:.2f}%\n"
    stats_text += f"  Mean Training Time:     {rand_time_mean:.2f} seconds\n"
    stats_text += f"  Std Dev Time:           {rand_time_std:.2f} seconds\n"

stats_text += "\n"
stats_text += "SPECTRAL vs RANDOM COMPARISON:\n"
stats_text += "-" * 50 + "\n"

if len(spectral_02) > 0 and len(random_02) > 0:
    acc_diff = spec_acc - rand_mean
    time_diff = spec_time - rand_time_mean
    time_savings_pct = (time_diff / rand_time_mean) * 100
    
    stats_text += f"  Accuracy Difference:    {acc_diff:+.2f}%\n"
    stats_text += f"  {'✓ SPECTRAL IS BETTER' if acc_diff > 0 else '✗ RANDOM IS BETTER'}\n\n"
    stats_text += f"  Training Time Diff:     {time_diff:+.2f}s\n"
    stats_text += f"  Time Savings:           {time_savings_pct:.1f}%\n"
    stats_text += f"  {'✓ SPECTRAL IS FASTER' if time_diff < 0 else '✗ RANDOM IS FASTER'}\n"

# Display statistics
ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, fontsize=11,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, pad=1))

plt.tight_layout()
plt.savefig('MNIST_Spectral_0.2_Detailed_visualization.png', dpi=300, bbox_inches='tight', 
           facecolor='white', edgecolor='none')
print("\nDetailed Spectral 0.2 visualization saved as 'MNIST_Spectral_0.2_Detailed_visualization.png'")
plt.show()

# Print summary to console
print("\n" + "="*70)
print("SPECTRAL 0.2 vs RANDOM 0.2 - COMPREHENSIVE SUMMARY")
print("="*70)
print(f"\n{'Metric':<30} {'Spectral 0.2':<20} {'Random 0.2':<20}")
print("-"*70)

if len(spectral_02) > 0:
    print(f"{'Accuracy':<30} {spectral_02['Accuracy_After_Finetuning'].values[0]:>18.2f}% {rand_mean:>18.2f}%")
    print(f"{'Training Time (seconds)':<30} {spectral_02['Training_Time_sec'].values[0]:>18.2f}s {rand_time_mean:>18.2f}s")
    print(f"{'Compression Rate':<30} {spectral_02['Compression_Rate'].values[0]:>18.4f}x {random_02['Compression_Rate'].mean():>18.4f}x")
    
    acc_diff = spectral_02['Accuracy_After_Finetuning'].values[0] - rand_mean
    time_diff = spectral_02['Training_Time_sec'].values[0] - rand_time_mean
    
    print("\n" + "="*70)
    print("KEY FINDINGS:")
    print("="*70)
    print(f"\n✓ Accuracy: Spectral is {acc_diff:+.2f}% {'BETTER' if acc_diff > 0 else 'WORSE'} than Random mean")
    print(f"✓ Speed: Spectral is {abs(time_diff):.2f}s {'FASTER' if time_diff < 0 else 'SLOWER'} than Random mean")
    print(f"✓ Efficiency: {abs(time_diff)/rand_time_mean*100:.1f}% {'faster' if time_diff < 0 else 'slower'} training")

print("\n" + "="*70)
