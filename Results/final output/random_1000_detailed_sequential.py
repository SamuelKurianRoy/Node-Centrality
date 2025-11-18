import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")

# Load the data
df = pd.read_csv('MNIST_final_output.csv')

# Get random experiments for all ratios
random_02 = df[(df['Method'] == 'Random') & (df['Prune_Ratio'] == 0.2)]
random_03 = df[(df['Method'] == 'Random') & (df['Prune_Ratio'] == 0.3)]
random_04 = df[(df['Method'] == 'Random') & (df['Prune_Ratio'] == 0.4)]

# Get spectral values for reference
spectral_02 = df[(df['Method'] == 'Spectral') & (df['Prune_Ratio'] == 0.2)]['Accuracy_After_Finetuning'].values[0]
spectral_03 = df[(df['Method'] == 'Spectral') & (df['Prune_Ratio'] == 0.3)]['Accuracy_After_Finetuning'].values[0]
spectral_04 = df[(df['Method'] == 'Spectral') & (df['Prune_Ratio'] == 0.4)]['Accuracy_After_Finetuning'].values[0]

print("=" * 80)
print("RANDOM METHOD - 1000 EXPERIMENTS VISUALIZATION (Sequential Mode)")
print("=" * 80)
print(f"\nRandom 0.2: {len(random_02)} experiments")
print(f"Random 0.3: {len(random_03)} experiments")
print(f"Random 0.4: {len(random_04)} experiments")
print(f"\nSpectral Reference Values:")
print(f"  Spectral 0.2: {spectral_02:.4f}%")
print(f"  Spectral 0.3: {spectral_03:.4f}%")
print(f"  Spectral 0.4: {spectral_04:.4f}%")
print("\nEach graph will appear one by one. Close the window to see the next visualization.\n")

# Color mapping
colors = {0.2: '#1f77b4', 0.3: '#ff7f0e', 0.4: '#2ca02c'}

# ===== VISUALIZATION 1: Bar Chart - Prune Ratio 0.2 =====
print("Graph 1/9: Bar Chart - Random Prune Ratio 0.2 (all 1000 experiments)")
fig, ax = plt.subplots(figsize=(14, 6))

data_sorted = random_02.sort_values('Iteration').reset_index(drop=True)
experiment_nums = np.arange(1, len(data_sorted) + 1)

bars = ax.bar(experiment_nums, data_sorted['Accuracy_After_Finetuning'].values,
              alpha=0.7, color=colors[0.2], edgecolor=colors[0.2], linewidth=0.5)

mean_acc = data_sorted['Accuracy_After_Finetuning'].mean()
min_acc = data_sorted['Accuracy_After_Finetuning'].min()
max_acc = data_sorted['Accuracy_After_Finetuning'].max()
std_acc = data_sorted['Accuracy_After_Finetuning'].std()

ax.axhline(y=mean_acc, color='red', linestyle='--', linewidth=3,
          label=f'Random Mean: {mean_acc:.4f}%')
ax.axhline(y=min_acc, color='gray', linestyle=':', linewidth=2, alpha=0.7,
          label=f'Random Min: {min_acc:.4f}%')
ax.axhline(y=max_acc, color='darkgreen', linestyle=':', linewidth=2, alpha=0.7,
          label=f'Random Max: {max_acc:.4f}%')
ax.axhline(y=spectral_02, color='purple', linestyle='-', linewidth=3.5,
          label=f'Spectral: {spectral_02:.4f}%')
ax.fill_between(experiment_nums, mean_acc - std_acc, mean_acc + std_acc,
                alpha=0.2, color='red', label=f'±1 Std Dev: {std_acc:.4f}%')

ax.set_xlabel('Experiment Number', fontsize=13, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('Random Pruning - Prune Ratio 0.2\nBar Chart of All 1000 Experiments (with Spectral Reference)',
            fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_ylim([97.2, 98.4])

plt.tight_layout()
plt.savefig('Random_0.2_Bar_Chart.png', dpi=300, bbox_inches='tight', facecolor='white')
print("  ✓ Saved as 'Random_0.2_Bar_Chart.png'")
plt.show()

# ===== VISUALIZATION 2: Bar Chart - Prune Ratio 0.3 =====
print("\nGraph 2/9: Bar Chart - Random Prune Ratio 0.3 (all 1000 experiments)")
fig, ax = plt.subplots(figsize=(14, 6))

data_sorted = random_03.sort_values('Iteration').reset_index(drop=True)
experiment_nums = np.arange(1, len(data_sorted) + 1)

bars = ax.bar(experiment_nums, data_sorted['Accuracy_After_Finetuning'].values,
              alpha=0.7, color=colors[0.3], edgecolor=colors[0.3], linewidth=0.5)

mean_acc = data_sorted['Accuracy_After_Finetuning'].mean()
min_acc = data_sorted['Accuracy_After_Finetuning'].min()
max_acc = data_sorted['Accuracy_After_Finetuning'].max()
std_acc = data_sorted['Accuracy_After_Finetuning'].std()

ax.axhline(y=mean_acc, color='red', linestyle='--', linewidth=3,
          label=f'Random Mean: {mean_acc:.4f}%')
ax.axhline(y=min_acc, color='gray', linestyle=':', linewidth=2, alpha=0.7,
          label=f'Random Min: {min_acc:.4f}%')
ax.axhline(y=max_acc, color='darkgreen', linestyle=':', linewidth=2, alpha=0.7,
          label=f'Random Max: {max_acc:.4f}%')
ax.axhline(y=spectral_03, color='purple', linestyle='-', linewidth=3.5,
          label=f'Spectral: {spectral_03:.4f}%')
ax.fill_between(experiment_nums, mean_acc - std_acc, mean_acc + std_acc,
                alpha=0.2, color='red', label=f'±1 Std Dev: {std_acc:.4f}%')

ax.set_xlabel('Experiment Number', fontsize=13, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('Random Pruning - Prune Ratio 0.3\nBar Chart of All 1000 Experiments (with Spectral Reference)',
            fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_ylim([97.2, 98.4])

plt.tight_layout()
plt.savefig('Random_0.3_Bar_Chart.png', dpi=300, bbox_inches='tight', facecolor='white')
print("  ✓ Saved as 'Random_0.3_Bar_Chart.png'")
plt.show()

# ===== VISUALIZATION 3: Bar Chart - Prune Ratio 0.4 =====
print("\nGraph 3/9: Bar Chart - Random Prune Ratio 0.4 (all 1000 experiments)")
fig, ax = plt.subplots(figsize=(14, 6))

data_sorted = random_04.sort_values('Iteration').reset_index(drop=True)
experiment_nums = np.arange(1, len(data_sorted) + 1)

bars = ax.bar(experiment_nums, data_sorted['Accuracy_After_Finetuning'].values,
              alpha=0.7, color=colors[0.4], edgecolor=colors[0.4], linewidth=0.5)

mean_acc = data_sorted['Accuracy_After_Finetuning'].mean()
min_acc = data_sorted['Accuracy_After_Finetuning'].min()
max_acc = data_sorted['Accuracy_After_Finetuning'].max()
std_acc = data_sorted['Accuracy_After_Finetuning'].std()

ax.axhline(y=mean_acc, color='red', linestyle='--', linewidth=3,
          label=f'Random Mean: {mean_acc:.4f}%')
ax.axhline(y=min_acc, color='gray', linestyle=':', linewidth=2, alpha=0.7,
          label=f'Random Min: {min_acc:.4f}%')
ax.axhline(y=max_acc, color='darkgreen', linestyle=':', linewidth=2, alpha=0.7,
          label=f'Random Max: {max_acc:.4f}%')
ax.axhline(y=spectral_04, color='purple', linestyle='-', linewidth=3.5,
          label=f'Spectral: {spectral_04:.4f}%')
ax.fill_between(experiment_nums, mean_acc - std_acc, mean_acc + std_acc,
                alpha=0.2, color='red', label=f'±1 Std Dev: {std_acc:.4f}%')

ax.set_xlabel('Experiment Number', fontsize=13, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('Random Pruning - Prune Ratio 0.4\nBar Chart of All 1000 Experiments (with Spectral Reference)',
            fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_ylim([97.2, 98.4])

plt.tight_layout()
plt.savefig('Random_0.4_Bar_Chart.png', dpi=300, bbox_inches='tight', facecolor='white')
print("  ✓ Saved as 'Random_0.4_Bar_Chart.png'")
plt.show()

# ===== VISUALIZATION 4: Histogram - Prune Ratio 0.2 =====
print("\nGraph 4/9: Histogram - Random Prune Ratio 0.2 (Accuracy Distribution)")
fig, ax = plt.subplots(figsize=(14, 6))

accuracies = random_02['Accuracy_After_Finetuning'].values
n, bins, patches = ax.hist(accuracies, bins=50, alpha=0.7, color=colors[0.2],
                            edgecolor='black', linewidth=0.8)

mean_acc = accuracies.mean()
std_acc = accuracies.std()

ax.axvline(mean_acc, color='red', linestyle='--', linewidth=3,
          label=f'Mean: {mean_acc:.4f}%')
ax.axvline(mean_acc - std_acc, color='orange', linestyle=':', linewidth=2.5,
          label=f'±1 Std: ±{std_acc:.4f}%')
ax.axvline(mean_acc + std_acc, color='orange', linestyle=':', linewidth=2.5)

# Add KDE curve
kde = stats.gaussian_kde(accuracies)
x_range = np.linspace(accuracies.min(), accuracies.max(), 200)
ax2 = ax.twinx()
ax2.plot(x_range, kde(x_range), 'k-', linewidth=2.5, label='KDE (Density Curve)')
ax2.set_ylabel('Density', fontsize=13, fontweight='bold')
ax2.legend(fontsize=11, loc='upper right')

ax.set_xlabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_ylabel('Frequency (Number of Experiments)', fontsize=13, fontweight='bold')
ax.set_title('Random Pruning - Prune Ratio 0.2\nAccuracy Distribution Histogram (1000 Experiments)',
            fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('Random_0.2_Histogram.png', dpi=300, bbox_inches='tight', facecolor='white')
print("  ✓ Saved as 'Random_0.2_Histogram.png'")
plt.show()

# ===== VISUALIZATION 5: Histogram - Prune Ratio 0.3 =====
print("\nGraph 5/9: Histogram - Random Prune Ratio 0.3 (Accuracy Distribution)")
fig, ax = plt.subplots(figsize=(14, 6))

accuracies = random_03['Accuracy_After_Finetuning'].values
n, bins, patches = ax.hist(accuracies, bins=50, alpha=0.7, color=colors[0.3],
                            edgecolor='black', linewidth=0.8)

mean_acc = accuracies.mean()
std_acc = accuracies.std()

ax.axvline(mean_acc, color='red', linestyle='--', linewidth=3,
          label=f'Mean: {mean_acc:.4f}%')
ax.axvline(mean_acc - std_acc, color='orange', linestyle=':', linewidth=2.5,
          label=f'±1 Std: ±{std_acc:.4f}%')
ax.axvline(mean_acc + std_acc, color='orange', linestyle=':', linewidth=2.5)

# Add KDE curve
kde = stats.gaussian_kde(accuracies)
x_range = np.linspace(accuracies.min(), accuracies.max(), 200)
ax2 = ax.twinx()
ax2.plot(x_range, kde(x_range), 'k-', linewidth=2.5, label='KDE (Density Curve)')
ax2.set_ylabel('Density', fontsize=13, fontweight='bold')
ax2.legend(fontsize=11, loc='upper right')

ax.set_xlabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_ylabel('Frequency (Number of Experiments)', fontsize=13, fontweight='bold')
ax.set_title('Random Pruning - Prune Ratio 0.3\nAccuracy Distribution Histogram (1000 Experiments)',
            fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('Random_0.3_Histogram.png', dpi=300, bbox_inches='tight', facecolor='white')
print("  ✓ Saved as 'Random_0.3_Histogram.png'")
plt.show()

# ===== VISUALIZATION 6: Histogram - Prune Ratio 0.4 =====
print("\nGraph 6/9: Histogram - Random Prune Ratio 0.4 (Accuracy Distribution)")
fig, ax = plt.subplots(figsize=(14, 6))

accuracies = random_04['Accuracy_After_Finetuning'].values
n, bins, patches = ax.hist(accuracies, bins=50, alpha=0.7, color=colors[0.4],
                            edgecolor='black', linewidth=0.8)

mean_acc = accuracies.mean()
std_acc = accuracies.std()

ax.axvline(mean_acc, color='red', linestyle='--', linewidth=3,
          label=f'Mean: {mean_acc:.4f}%')
ax.axvline(mean_acc - std_acc, color='orange', linestyle=':', linewidth=2.5,
          label=f'±1 Std: ±{std_acc:.4f}%')
ax.axvline(mean_acc + std_acc, color='orange', linestyle=':', linewidth=2.5)

# Add KDE curve
kde = stats.gaussian_kde(accuracies)
x_range = np.linspace(accuracies.min(), accuracies.max(), 200)
ax2 = ax.twinx()
ax2.plot(x_range, kde(x_range), 'k-', linewidth=2.5, label='KDE (Density Curve)')
ax2.set_ylabel('Density', fontsize=13, fontweight='bold')
ax2.legend(fontsize=11, loc='upper right')

ax.set_xlabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_ylabel('Frequency (Number of Experiments)', fontsize=13, fontweight='bold')
ax.set_title('Random Pruning - Prune Ratio 0.4\nAccuracy Distribution Histogram (1000 Experiments)',
            fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('Random_0.4_Histogram.png', dpi=300, bbox_inches='tight', facecolor='white')
print("  ✓ Saved as 'Random_0.4_Histogram.png'")
plt.show()

# ===== VISUALIZATION 7: Box Plot - Prune Ratio 0.2 =====
print("\nGraph 7/9: Box Plot - Random Prune Ratio 0.2")
fig, ax = plt.subplots(figsize=(10, 7))

data_02 = random_02['Accuracy_After_Finetuning'].values
bp = ax.boxplot([data_02], labels=['Prune Ratio 0.2\n(1000 experiments)'], 
                patch_artist=True, widths=0.6, showmeans=True, meanline=True)

for patch in bp['boxes']:
    patch.set_facecolor(colors[0.2])
    patch.set_alpha(0.7)

for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bp[element], color='black', linewidth=2)

# Add Spectral reference line
ax.axhline(y=spectral_02, color='purple', linestyle='-', linewidth=3.5, alpha=0.9,
          label=f'Spectral: {spectral_02:.4f}%')

# Add statistics text
stats_text = f'Mean: {data_02.mean():.4f}%\nMedian: {np.median(data_02):.4f}%\nStd: {data_02.std():.4f}%\nMin: {data_02.min():.4f}%\nMax: {data_02.max():.4f}%'
ax.text(1.35, data_02.min() + 0.15, stats_text, fontsize=10, 
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('Box Plot - Random Pruning Prune Ratio 0.2 with Spectral Reference\n(1000 Random Experiments)',
            fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=12, loc='upper right')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([97.2, 98.4])

plt.tight_layout()
plt.savefig('Random_0.2_BoxPlot.png', dpi=300, bbox_inches='tight', facecolor='white')
print("  ✓ Saved as 'Random_0.2_BoxPlot.png'")
plt.show()

# ===== VISUALIZATION 8: Box Plot - Prune Ratio 0.3 =====
print("\nGraph 8/9: Box Plot - Random Prune Ratio 0.3")
fig, ax = plt.subplots(figsize=(10, 7))

data_03 = random_03['Accuracy_After_Finetuning'].values
bp = ax.boxplot([data_03], labels=['Prune Ratio 0.3\n(1000 experiments)'], 
                patch_artist=True, widths=0.6, showmeans=True, meanline=True)

for patch in bp['boxes']:
    patch.set_facecolor(colors[0.3])
    patch.set_alpha(0.7)

for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bp[element], color='black', linewidth=2)

# Add Spectral reference line
ax.axhline(y=spectral_03, color='darkviolet', linestyle='-', linewidth=3.5, alpha=0.9,
          label=f'Spectral: {spectral_03:.4f}%')

# Add statistics text
stats_text = f'Mean: {data_03.mean():.4f}%\nMedian: {np.median(data_03):.4f}%\nStd: {data_03.std():.4f}%\nMin: {data_03.min():.4f}%\nMax: {data_03.max():.4f}%'
ax.text(1.35, data_03.min() + 0.15, stats_text, fontsize=10, 
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('Box Plot - Random Pruning Prune Ratio 0.3 with Spectral Reference\n(1000 Random Experiments)',
            fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=12, loc='upper right')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([97.2, 98.4])

plt.tight_layout()
plt.savefig('Random_0.3_BoxPlot.png', dpi=300, bbox_inches='tight', facecolor='white')
print("  ✓ Saved as 'Random_0.3_BoxPlot.png'")
plt.show()

# ===== VISUALIZATION 9: Box Plot - Prune Ratio 0.4 =====
print("\nGraph 9/9: Box Plot - Random Prune Ratio 0.4")
fig, ax = plt.subplots(figsize=(10, 7))

data_04 = random_04['Accuracy_After_Finetuning'].values
bp = ax.boxplot([data_04], labels=['Prune Ratio 0.4\n(1000 experiments)'], 
                patch_artist=True, widths=0.6, showmeans=True, meanline=True)

for patch in bp['boxes']:
    patch.set_facecolor(colors[0.4])
    patch.set_alpha(0.7)

for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bp[element], color='black', linewidth=2)

# Add Spectral reference line
ax.axhline(y=spectral_04, color='indigo', linestyle='-', linewidth=3.5, alpha=0.9,
          label=f'Spectral: {spectral_04:.4f}%')

# Add statistics text
stats_text = f'Mean: {data_04.mean():.4f}%\nMedian: {np.median(data_04):.4f}%\nStd: {data_04.std():.4f}%\nMin: {data_04.min():.4f}%\nMax: {data_04.max():.4f}%'
ax.text(1.35, data_04.min() + 0.15, stats_text, fontsize=10, 
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('Box Plot - Random Pruning Prune Ratio 0.4 with Spectral Reference\n(1000 Random Experiments)',
            fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=12, loc='upper right')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([97.2, 98.4])

plt.tight_layout()
plt.savefig('Random_0.4_BoxPlot.png', dpi=300, bbox_inches='tight', facecolor='white')
print("  ✓ Saved as 'Random_0.4_BoxPlot.png'")
plt.show()

# ===== VISUALIZATION 10: Training Time Histogram =====
print("\nGraph 10/11: Training Time Distribution (All 3000 Experiments)")
fig, ax = plt.subplots(figsize=(14, 6))

all_training_times = pd.concat([random_02, random_03, random_04])['Training_Time_sec']
ax.hist(all_training_times, bins=40, alpha=0.7, color='purple', edgecolor='black', linewidth=0.8)

mean_time = all_training_times.mean()
std_time = all_training_times.std()

ax.axvline(mean_time, color='red', linestyle='--', linewidth=3,
          label=f'Mean: {mean_time:.2f}s')
ax.axvline(mean_time - std_time, color='orange', linestyle=':', linewidth=2.5,
          label=f'±1 Std: ±{std_time:.2f}s')
ax.axvline(mean_time + std_time, color='orange', linestyle=':', linewidth=2.5)

# Add KDE
kde_time = stats.gaussian_kde(all_training_times)
x_range = np.linspace(all_training_times.min(), all_training_times.max(), 200)
ax2 = ax.twinx()
ax2.plot(x_range, kde_time(x_range), 'k-', linewidth=2.5, label='KDE (Density Curve)')
ax2.set_ylabel('Density', fontsize=13, fontweight='bold')
ax2.legend(fontsize=11, loc='upper right')

ax.set_xlabel('Training Time (seconds)', fontsize=13, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=13, fontweight='bold')
ax.set_title('Training Time Distribution - Random Pruning\n(All 3000 Experiments Combined)',
            fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('Random_Training_Time_Distribution.png', dpi=300, bbox_inches='tight', facecolor='white')
print("  ✓ Saved as 'Random_Training_Time_Distribution.png'")
plt.show()

# ===== VISUALIZATION 11: Cumulative Distribution Function =====
print("\nGraph 11/11: Cumulative Distribution Function (CDF)")
fig, ax = plt.subplots(figsize=(14, 7))

for ratio, color, label in [(0.2, colors[0.2], 'Prune Ratio 0.2'),
                             (0.3, colors[0.3], 'Prune Ratio 0.3'),
                             (0.4, colors[0.4], 'Prune Ratio 0.4')]:
    if ratio == 0.2:
        data = random_02['Accuracy_After_Finetuning'].values
    elif ratio == 0.3:
        data = random_03['Accuracy_After_Finetuning'].values
    else:
        data = random_04['Accuracy_After_Finetuning'].values
    
    sorted_data = np.sort(data)
    cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax.plot(sorted_data, cumulative, linewidth=3, color=color, label=label, marker='o', 
           markersize=2, alpha=0.8)

ax.set_xlabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_ylabel('Cumulative Probability', fontsize=13, fontweight='bold')
ax.set_title('Cumulative Distribution Function (CDF) - Random Pruning\n(All 3 Prune Ratios, 1000 Experiments Each)',
            fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=12, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig('Random_CDF_Comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
print("  ✓ Saved as 'Random_CDF_Comparison.png'")
plt.show()

# ===== PRINT COMPREHENSIVE STATISTICS =====
print("\n" + "="*80)
print("RANDOM METHOD - 1000 EXPERIMENTS DETAILED ANALYSIS")
print("="*80)

for ratio, label in [(0.2, "PRUNE RATIO 0.2"), (0.3, "PRUNE RATIO 0.3"), (0.4, "PRUNE RATIO 0.4")]:
    if ratio == 0.2:
        data = random_02
    elif ratio == 0.3:
        data = random_03
    else:
        data = random_04
    
    acc = data['Accuracy_After_Finetuning']
    time = data['Training_Time_sec']
    comp = data['Compression_Rate']
    
    print(f"\n{label}")
    print("-" * 80)
    print(f"Number of Experiments: {len(data)}")
    
    print(f"\nACCURACY STATISTICS:")
    print(f"  Mean:                 {acc.mean():.4f}%")
    print(f"  Median:               {acc.median():.4f}%")
    print(f"  Std Deviation:        {acc.std():.4f}%")
    print(f"  Min:                  {acc.min():.4f}%")
    print(f"  Max:                  {acc.max():.4f}%")
    print(f"  Range (Max - Min):    {acc.max() - acc.min():.4f}%")
    print(f"  Q1 (25th percentile): {acc.quantile(0.25):.4f}%")
    print(f"  Q3 (75th percentile): {acc.quantile(0.75):.4f}%")
    print(f"  IQR (Q3 - Q1):        {acc.quantile(0.75) - acc.quantile(0.25):.4f}%")
    
    print(f"\nTRAINING TIME STATISTICS:")
    print(f"  Mean:                 {time.mean():.2f} seconds")
    print(f"  Median:               {time.median():.2f} seconds")
    print(f"  Std Deviation:        {time.std():.2f} seconds")
    print(f"  Min:                  {time.min():.2f} seconds")
    print(f"  Max:                  {time.max():.2f} seconds")
    
    print(f"\nCOMPRESSION RATE:")
    print(f"  Mean:                 {comp.mean():.4f}x")
    print(f"  Min:                  {comp.min():.4f}x")
    print(f"  Max:                  {comp.max():.4f}x")

print("\n" + "="*80)
print("✓ ALL VISUALIZATIONS COMPLETED AND SAVED!")
print("="*80)
print("\nGenerated Files:")
print("  1. Random_0.2_Bar_Chart.png")
print("  2. Random_0.3_Bar_Chart.png")
print("  3. Random_0.4_Bar_Chart.png")
print("  4. Random_0.2_Histogram.png")
print("  5. Random_0.3_Histogram.png")
print("  6. Random_0.4_Histogram.png")
print("  7. Random_0.2_BoxPlot.png")
print("  8. Random_0.3_BoxPlot.png")
print("  9. Random_0.4_BoxPlot.png")
print("  10. Random_Training_Time_Distribution.png")
print("  11. Random_CDF_Comparison.png")
print("\n" + "="*80)
