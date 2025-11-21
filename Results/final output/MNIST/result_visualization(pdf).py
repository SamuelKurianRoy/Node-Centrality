import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# Helper: compute sensible axis limits from a pandas Series
def make_limits_from_series(series, pad_factor=0.05, min_span=0.5, lower_bound=None, upper_bound=None, clamp_min=0.0, clamp_max=100.0):
    """Return (lower, upper) bounds for plotting based on series values.
    - pad_factor: fraction of span to use as padding
    - min_span: minimum absolute padding
    - lower_bound / upper_bound: explicit clamps (None to ignore)
    - clamp_min / clamp_max: overall allowed bounds (e.g., 0-100 for percent)
    """
    if hasattr(series, 'min'):
        try:
            vmin = float(series.min())
            vmax = float(series.max())
        except Exception:
            vmin = float(np.min(series))
            vmax = float(np.max(series))
    else:
        vmin = float(np.min(series))
        vmax = float(np.max(series))

    span = max(vmax - vmin, 0.0)
    pad = max(span * pad_factor, min_span)
    lower = vmin - pad
    upper = vmax + pad

    if lower_bound is not None:
        lower = max(lower, lower_bound)
    if upper_bound is not None:
        upper = min(upper, upper_bound)

    lower = max(lower, clamp_min)
    upper = min(upper, clamp_max)

    # In case of degenerate span (all values equal), expand around the value
    if upper - lower < 1e-6:
        lower = max(vmin - min_span, clamp_min)
        upper = min(vmax + min_span, clamp_max)

    return lower, upper

print(f"Random 0.2: {len(random_02)} experiments")
print(f"Random 0.3: {len(random_03)} experiments")
print(f"Random 0.4: {len(random_04)} experiments")
print(f"\nSpectral Reference Values:")
print(f"  Spectral 0.2: {spectral_02:.4f}%")
print(f"  Spectral 0.3: {spectral_03:.4f}%")
print(f"  Spectral 0.4: {spectral_04:.4f}%")

# Create comprehensive visualization
fig = plt.figure(figsize=(22, 14))

# Color mapping
colors = {0.2: '#1f77b4', 0.3: '#ff7f0e', 0.4: '#2ca02c'}
color_names = {0.2: 'Blue', 0.3: 'Orange', 0.4: 'Green'}

# ===== ROW 1: BAR CHARTS (All 1000 experiments) =====
for idx, ratio in enumerate([0.2, 0.3, 0.4]):
    ax = plt.subplot(3, 3, idx + 1)
    
    if ratio == 0.2:
        data = random_02
        spectral_val = spectral_02
    elif ratio == 0.3:
        data = random_03
        spectral_val = spectral_03
    else:
        data = random_04
        spectral_val = spectral_04
    
    data_sorted = data.sort_values('Iteration').reset_index(drop=True)
    experiment_nums = np.arange(1, len(data_sorted) + 1)
    
    # Create bar chart
    bars = ax.bar(experiment_nums, data_sorted['Accuracy_After_Finetuning'].values,
                  alpha=0.7, color=colors[ratio], edgecolor=colors[ratio], linewidth=0.5)
    
    # Add mean line
    mean_acc = data_sorted['Accuracy_After_Finetuning'].mean()
    std_acc = data_sorted['Accuracy_After_Finetuning'].std()
    ax.axhline(y=mean_acc, color='red', linestyle='--', linewidth=2.5,
               label=f'Random Mean: {mean_acc:.4f}%')
    
    # Add min/max lines
    min_acc = data_sorted['Accuracy_After_Finetuning'].min()
    max_acc = data_sorted['Accuracy_After_Finetuning'].max()
    ax.axhline(y=min_acc, color='gray', linestyle=':', linewidth=1.5, alpha=0.7,
               label=f'Random Min: {min_acc:.4f}%')
    ax.axhline(y=max_acc, color='darkgreen', linestyle=':', linewidth=1.5, alpha=0.7,
               label=f'Random Max: {max_acc:.4f}%')
    
    # Add spectral reference line
    ax.axhline(y=spectral_val, color='purple', linestyle='-', linewidth=3,
               label=f'Spectral: {spectral_val:.4f}%')
    
    # Add std dev shading
    ax.fill_between(experiment_nums, mean_acc - std_acc, mean_acc + std_acc,
                    alpha=0.2, color='red', label=f'±1 Std Dev: {std_acc:.4f}%')
    
    ax.set_xlabel('Experiment Number', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    ymin, ymax = make_limits_from_series(data_sorted['Accuracy_After_Finetuning'])
    ax.set_ylim([ymin, ymax])
    
    # Print statistics
    print(f"\nPrune Ratio {ratio} - Bar Chart Statistics:")
    print(f"  Mean: {mean_acc:.2f}%")
    print(f"  Std:  {data_sorted['Accuracy_After_Finetuning'].std():.2f}%")
    print(f"  Min:  {min_acc:.2f}%")
    print(f"  Max:  {max_acc:.2f}%")

# ===== ROW 2: HISTOGRAMS =====
for idx, ratio in enumerate([0.2, 0.3, 0.4]):
    ax = plt.subplot(3, 3, idx + 4)
    
    if ratio == 0.2:
        data = random_02
    elif ratio == 0.3:
        data = random_03
    else:
        data = random_04
    
    accuracies = data['Accuracy_After_Finetuning'].values
    
    # Create histogram
    n, bins, patches = ax.hist(accuracies, bins=40, alpha=0.7, color=colors[ratio],
                               edgecolor='black', linewidth=0.5)
    
    # Add mean and std lines
    mean_acc = accuracies.mean()
    std_acc = accuracies.std()
    
    ax.axvline(mean_acc, color='red', linestyle='--', linewidth=2.5,
               label=f'Mean: {mean_acc:.4f}%')
    ax.axvline(mean_acc - std_acc, color='orange', linestyle=':', linewidth=2,
               label=f'±1 Std: ±{std_acc:.4f}%')
    ax.axvline(mean_acc + std_acc, color='orange', linestyle=':', linewidth=2)
    
    # Add KDE curve
    from scipy import stats
    kde = stats.gaussian_kde(accuracies)
    x_range = np.linspace(accuracies.min(), accuracies.max(), 200)
    ax2 = ax.twinx()
    ax2.plot(x_range, kde(x_range), 'k-', linewidth=2, label='KDE (Density Curve)')
    ax2.set_ylabel('Density', fontsize=10, fontweight='bold')
    ax2.legend(fontsize=8, loc='upper right')
    
    ax.set_xlabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency (Number of Experiments)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    # Use data-driven x-limits for the histogram
    xmin, xmax = make_limits_from_series(accuracies)
    ax.set_xlim([xmin, xmax])

# ===== ROW 3: ADDITIONAL STATISTICS =====

# Box plot comparison
ax = plt.subplot(3, 3, 7)
data_list = [random_02['Accuracy_After_Finetuning'].values,
             random_03['Accuracy_After_Finetuning'].values,
             random_04['Accuracy_After_Finetuning'].values]
labels = ['Ratio 0.2', 'Ratio 0.3', 'Ratio 0.4']

bp = ax.boxplot(data_list, labels=labels, patch_artist=True, widths=0.6, showmeans=True, meanline=True)
for patch, color in zip(bp['boxes'], ['#1f77b4', '#ff7f0e', '#2ca02c']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bp[element], color='black', linewidth=2)

# Add spectral reference lines
spectral_values = [spectral_02, spectral_03, spectral_04]
for i, (spectral_val, color) in enumerate(zip(spectral_values, ['purple', 'darkviolet', 'indigo'])):
    ax.plot([i + 0.7, i + 1.3], [spectral_val, spectral_val], color=color, linestyle='-', linewidth=2.5, alpha=0.9)

ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
# Use data-driven y-limits
all_data = np.concatenate(data_list)
ymin, ymax = make_limits_from_series(all_data)
ax.set_ylim([ymin, ymax])

# Cumulative distribution
ax = plt.subplot(3, 3, 9)
for ratio, color, label in [
    (0.2, '#1f77b4', 'Prune Ratio 0.2'),
    (0.3, '#ff7f0e', 'Prune Ratio 0.3'),
    (0.4, '#2ca02c', 'Prune Ratio 0.4')
]:
    if ratio == 0.2:
        data = random_02['Accuracy_After_Finetuning'].values
    elif ratio == 0.3:
        data = random_03['Accuracy_After_Finetuning'].values
    else:
        data = random_04['Accuracy_After_Finetuning'].values
    
    sorted_data = np.sort(data)
    cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax.plot(sorted_data, cumulative, linewidth=2.5, color=color, marker='o',
            markersize=2, alpha=0.8, label=label)

ax.set_xlabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax.set_ylabel('Cumulative Probability', fontsize=11, fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.0])

plt.tight_layout()

# ✅ Save as PDF instead of PNG
plt.savefig(
    'Random_1000_Experiments_Visualization.pdf',
    bbox_inches='tight',
    facecolor='white',
    edgecolor='none'
)

print("\n✓ Visualization saved as 'Random_1000_Experiments_Visualization.pdf'")
plt.show()

# Print comprehensive statistics
print("\n" + "=" * 80)
print("RANDOM METHOD - 1000 EXPERIMENTS ANALYSIS")
print("=" * 80)

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
    
    print(f"\nCOMPRESSION RATE STATISTICS:")
    print(f"  Mean:                 {comp.mean():.4f}x")
    print(f"  Std Deviation:        {comp.std():.4f}x")

print("\n" + "=" * 80)
