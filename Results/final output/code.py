import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

# Load the data
df = pd.read_csv('MNIST_final_output.csv')

# Filter for Spectral method with Prune_Ratio = 0.2
spectral_02 = df[(df['Method'] == 'Spectral') & (df['Prune_Ratio'] == 0.2)]
random_02 = df[(df['Method'] == 'Random') & (df['Prune_Ratio'] == 0.2)]

print(f"Spectral 0.2 data points: {len(spectral_02)}")
print(f"Random 0.2 data points: {len(random_02)}")

# Create visualization for Spectral 0.2
fig = plt.figure(figsize=(18, 10))

# 1. Spectral 0.2: Bar chart of accuracy by experiment
ax1 = plt.subplot(2, 2, 1)
if len(spectral_02) > 0:
    spectral_sorted = spectral_02.sort_values('Iteration')
    experiment_nums = range(1, len(spectral_sorted) + 1)
    ax1.bar(experiment_nums, spectral_sorted['Accuracy_After_Finetuning'].values, 
            alpha=0.7, color='steelblue', edgecolor='black', label='Spectral 0.2')
    ax1.set_xlabel('Experiment Number')
    ax1.set_ylabel('Accuracy After Finetuning (%)')
    ax1.set_title('Spectral Method (Prune Ratio 0.2) - Accuracy by Experiment')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
else:
    ax1.text(0.5, 0.5, 'No Spectral 0.2 data', ha='center', va='center')

# 2. Spectral 0.2: Line plot with Random 0.2 comparison
ax2 = plt.subplot(2, 2, 2)
if len(spectral_02) > 0:
    spectral_sorted = spectral_02.sort_values('Iteration')
    experiment_nums = range(1, len(spectral_sorted) + 1)
    ax2.plot(experiment_nums, spectral_sorted['Accuracy_After_Finetuning'].values, 
            marker='o', linestyle='-', linewidth=2, markersize=8, 
            color='darkred', label='Spectral 0.2')
    
    # Add Random 0.2 line for comparison
    if len(random_02) > 0:
        random_sorted = random_02.sort_values('Iteration')
        # Plot mean line for Random
        random_mean = random_sorted['Accuracy_After_Finetuning'].mean()
        ax2.axhline(y=random_mean, color='blue', linestyle='--', linewidth=2, 
                   label=f'Random 0.2 Mean ({random_mean:.2f}%)')
        ax2.fill_between(experiment_nums, 
                         random_mean - random_sorted['Accuracy_After_Finetuning'].std(),
                         random_mean + random_sorted['Accuracy_After_Finetuning'].std(),
                         alpha=0.2, color='blue', label='Random 0.2 ±Std')
    
    ax2.set_xlabel('Experiment Number')
    ax2.set_ylabel('Accuracy After Finetuning (%)')
    ax2.set_title('Spectral vs Random (Prune Ratio 0.2) - Line Plot')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
else:
    ax2.text(0.5, 0.5, 'No Spectral 0.2 data', ha='center', va='center')

# 3. Spectral 0.2: Comparison box plot
ax3 = plt.subplot(2, 2, 3)
comparison_data = []
comparison_labels = []
if len(spectral_02) > 0:
    comparison_data.append(spectral_02['Accuracy_After_Finetuning'].values)
    comparison_labels.append('Spectral 0.2')
if len(random_02) > 0:
    comparison_data.append(random_02['Accuracy_After_Finetuning'].values)
    comparison_labels.append('Random 0.2')

if len(comparison_data) > 0:
    bp = ax3.boxplot(comparison_data, labels=comparison_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['darkred', 'lightblue']):
        patch.set_facecolor(color)
    ax3.set_ylabel('Accuracy After Finetuning (%)')
    ax3.set_title('Accuracy Distribution: Spectral vs Random (Prune Ratio 0.2)')
    ax3.grid(True, alpha=0.3)

# 4. Statistics table
ax4 = plt.subplot(2, 2, 4)
ax4.axis('off')

# Create statistics text
stats_text = "SPECTRAL 0.2 STATISTICS\n" + "="*40 + "\n"
if len(spectral_02) > 0:
    stats_text += f"Count: {len(spectral_02)}\n"
    stats_text += f"Mean Accuracy: {spectral_02['Accuracy_After_Finetuning'].mean():.2f}%\n"
    stats_text += f"Std Accuracy: {spectral_02['Accuracy_After_Finetuning'].std():.2f}%\n"
    stats_text += f"Min Accuracy: {spectral_02['Accuracy_After_Finetuning'].min():.2f}%\n"
    stats_text += f"Max Accuracy: {spectral_02['Accuracy_After_Finetuning'].max():.2f}%\n"
    stats_text += f"Training Time: {spectral_02['Training_Time_sec'].values[0]:.2f}s\n"
    stats_text += f"Compression Rate: {spectral_02['Compression_Rate'].values[0]:.4f}\n"
else:
    stats_text += "No Spectral 0.2 data found\n"

stats_text += "\nRANDOM 0.2 STATISTICS\n" + "="*40 + "\n"
if len(random_02) > 0:
    stats_text += f"Count: {len(random_02)}\n"
    stats_text += f"Mean Accuracy: {random_02['Accuracy_After_Finetuning'].mean():.2f}%\n"
    stats_text += f"Std Accuracy: {random_02['Accuracy_After_Finetuning'].std():.2f}%\n"
    stats_text += f"Min Accuracy: {random_02['Accuracy_After_Finetuning'].min():.2f}%\n"
    stats_text += f"Max Accuracy: {random_02['Accuracy_After_Finetuning'].max():.2f}%\n"
    stats_text += f"Mean Training Time: {random_02['Training_Time_sec'].mean():.2f}s\n"
    stats_text += f"Mean Compression Rate: {random_02['Compression_Rate'].mean():.4f}\n"
else:
    stats_text += "No Random 0.2 data found\n"

ax4.text(0.1, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('MNIST_Spectral_0.2_visualization.png', dpi=300, bbox_inches='tight')
print("Spectral 0.2 visualization saved as 'MNIST_Spectral_0.2_visualization.png'")
plt.show()

# Print detailed statistics
print("\n" + "="*60)
print("SPECTRAL 0.2 vs RANDOM 0.2 ANALYSIS")
print("="*60)

if len(spectral_02) > 0:
    print("\nSPECTRAL 0.2 DETAILS:")
    print(f"Iteration: {spectral_02['Iteration'].values[0]}")
    print(f"Accuracy After Finetuning: {spectral_02['Accuracy_After_Finetuning'].values[0]:.2f}%")
    print(f"Baseline Accuracy: {spectral_02['Baseline_Accuracy'].values[0]:.2f}%")
    print(f"Training Time: {spectral_02['Training_Time_sec'].values[0]:.2f} seconds")
    print(f"Compression Rate: {spectral_02['Compression_Rate'].values[0]:.4f}")
    print(f"Parameters: {spectral_02['Params'].values[0]:.0f}")
    print(f"Baseline FLOPs: {spectral_02['Baseline_FLOPs'].values[0]:.0f}")
else:
    print("\nNo Spectral 0.2 data found")

if len(random_02) > 0:
    print("\nRANDOM 0.2 SUMMARY:")
    print(f"Sample Size: {len(random_02)}")
    print(f"Mean Accuracy: {random_02['Accuracy_After_Finetuning'].mean():.2f}%")
    print(f"Std Accuracy: {random_02['Accuracy_After_Finetuning'].std():.2f}%")
    print(f"Min Accuracy: {random_02['Accuracy_After_Finetuning'].min():.2f}%")
    print(f"Max Accuracy: {random_02['Accuracy_After_Finetuning'].max():.2f}%")
    print(f"Mean Training Time: {random_02['Training_Time_sec'].mean():.2f} seconds")
    print(f"Mean Compression Rate: {random_02['Compression_Rate'].mean():.4f}")
else:
    print("\nNo Random 0.2 data found")

print("\n" + "="*60)
