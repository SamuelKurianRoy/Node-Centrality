import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 12)

# Load the data
df = pd.read_csv('MNIST_final_output.csv')

# Filter for all Spectral methods
spectral_data = df[df['Method'] == 'Spectral']

# Filter for Random at different ratios for comparison
random_02 = df[(df['Method'] == 'Random') & (df['Prune_Ratio'] == 0.2)]
random_03 = df[(df['Method'] == 'Random') & (df['Prune_Ratio'] == 0.3)]
random_04 = df[(df['Method'] == 'Random') & (df['Prune_Ratio'] == 0.4)]

print(f"Spectral total data points: {len(spectral_data)}")
print(f"Spectral 0.2: {len(spectral_data[spectral_data['Prune_Ratio'] == 0.2])}")
print(f"Spectral 0.3: {len(spectral_data[spectral_data['Prune_Ratio'] == 0.3])}")
print(f"Spectral 0.4: {len(spectral_data[spectral_data['Prune_Ratio'] == 0.4])}")

# Create visualization
fig = plt.figure(figsize=(20, 12))

# Color mapping for prune ratios
colors = {0.2: 'darkred', 0.3: 'darkblue', 0.4: 'darkgreen'}
color_random = {0.2: 'lightcoral', 0.3: 'lightblue', 0.4: 'lightgreen'}

# 1. All Spectral methods - Bar charts
for idx, ratio in enumerate([0.2, 0.3, 0.4]):
    ax = plt.subplot(3, 3, idx + 1)
    spectral_ratio = spectral_data[spectral_data['Prune_Ratio'] == ratio]
    
    if len(spectral_ratio) > 0:
        spectral_sorted = spectral_ratio.sort_values('Iteration')
        experiment_nums = range(1, len(spectral_sorted) + 1)
        ax.bar(experiment_nums, spectral_sorted['Accuracy_After_Finetuning'].values, 
               alpha=0.8, color=colors[ratio], edgecolor='black', linewidth=1.5)
        ax.set_xlabel('Experiment Number', fontsize=10)
        ax.set_ylabel('Accuracy After Finetuning (%)', fontsize=10)
        ax.set_title(f'Spectral Method (Prune Ratio {ratio}) - Bar Chart', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add value on top of bar
        for i, v in enumerate(spectral_sorted['Accuracy_After_Finetuning'].values):
            ax.text(i+1, v+0.01, f'{v:.2f}%', ha='center', va='bottom', fontsize=9)
    else:
        ax.text(0.5, 0.5, f'No Spectral {ratio} data', ha='center', va='center')

# 2. All Spectral methods - Line plots with Random comparison
for idx, ratio in enumerate([0.2, 0.3, 0.4]):
    ax = plt.subplot(3, 3, idx + 4)
    spectral_ratio = spectral_data[spectral_data['Prune_Ratio'] == ratio]
    
    if len(spectral_ratio) > 0:
        spectral_sorted = spectral_ratio.sort_values('Iteration')
        experiment_nums = range(1, len(spectral_sorted) + 1)
        ax.plot(experiment_nums, spectral_sorted['Accuracy_After_Finetuning'].values, 
                marker='o', linestyle='-', linewidth=3, markersize=10, 
                color=colors[ratio], label=f'Spectral {ratio}')
        
        # Add Random comparison line
        if ratio == 0.2 and len(random_02) > 0:
            random_mean = random_02['Accuracy_After_Finetuning'].mean()
            random_std = random_02['Accuracy_After_Finetuning'].std()
            ax.axhline(y=random_mean, color=color_random[ratio], linestyle='--', linewidth=2.5, 
                      label=f'Random {ratio} Mean')
            ax.fill_between([0, len(experiment_nums)+1], 
                           random_mean - random_std, random_mean + random_std,
                           alpha=0.2, color=color_random[ratio])
        elif ratio == 0.3 and len(random_03) > 0:
            random_mean = random_03['Accuracy_After_Finetuning'].mean()
            random_std = random_03['Accuracy_After_Finetuning'].std()
            ax.axhline(y=random_mean, color=color_random[ratio], linestyle='--', linewidth=2.5, 
                      label=f'Random {ratio} Mean')
            ax.fill_between([0, len(experiment_nums)+1], 
                           random_mean - random_std, random_mean + random_std,
                           alpha=0.2, color=color_random[ratio])
        elif ratio == 0.4 and len(random_04) > 0:
            random_mean = random_04['Accuracy_After_Finetuning'].mean()
            random_std = random_04['Accuracy_After_Finetuning'].std()
            ax.axhline(y=random_mean, color=color_random[ratio], linestyle='--', linewidth=2.5, 
                      label=f'Random {ratio} Mean')
            ax.fill_between([0, len(experiment_nums)+1], 
                           random_mean - random_std, random_mean + random_std,
                           alpha=0.2, color=color_random[ratio])
        
        ax.set_xlabel('Experiment Number', fontsize=10)
        ax.set_ylabel('Accuracy After Finetuning (%)', fontsize=10)
        ax.set_title(f'Spectral {ratio} vs Random {ratio} - Line Plot', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([97.3, 98.4])
    else:
        ax.text(0.5, 0.5, f'No Spectral {ratio} data', ha='center', va='center')

# 3. Summary comparison table
ax = plt.subplot(3, 3, 7)
ax.axis('off')

table_data = []
table_data.append(['Metric', 'Spectral 0.2', 'Random 0.2', 'Spectral 0.3', 'Random 0.3', 'Spectral 0.4', 'Random 0.4'])

# Accuracy row
spec_acc_02 = spectral_data[spectral_data['Prune_Ratio'] == 0.2]['Accuracy_After_Finetuning'].values[0] if len(spectral_data[spectral_data['Prune_Ratio'] == 0.2]) > 0 else 0
spec_acc_03 = spectral_data[spectral_data['Prune_Ratio'] == 0.3]['Accuracy_After_Finetuning'].values[0] if len(spectral_data[spectral_data['Prune_Ratio'] == 0.3]) > 0 else 0
spec_acc_04 = spectral_data[spectral_data['Prune_Ratio'] == 0.4]['Accuracy_After_Finetuning'].values[0] if len(spectral_data[spectral_data['Prune_Ratio'] == 0.4]) > 0 else 0

rand_acc_02 = f"{random_02['Accuracy_After_Finetuning'].mean():.2f}±{random_02['Accuracy_After_Finetuning'].std():.2f}" if len(random_02) > 0 else "N/A"
rand_acc_03 = f"{random_03['Accuracy_After_Finetuning'].mean():.2f}±{random_03['Accuracy_After_Finetuning'].std():.2f}" if len(random_03) > 0 else "N/A"
rand_acc_04 = f"{random_04['Accuracy_After_Finetuning'].mean():.2f}±{random_04['Accuracy_After_Finetuning'].std():.2f}" if len(random_04) > 0 else "N/A"

table_data.append(['Accuracy', f'{spec_acc_02:.2f}%', rand_acc_02, f'{spec_acc_03:.2f}%', rand_acc_03, f'{spec_acc_04:.2f}%', rand_acc_04])

# Training time row
spec_time_02 = spectral_data[spectral_data['Prune_Ratio'] == 0.2]['Training_Time_sec'].values[0] if len(spectral_data[spectral_data['Prune_Ratio'] == 0.2]) > 0 else 0
spec_time_03 = spectral_data[spectral_data['Prune_Ratio'] == 0.3]['Training_Time_sec'].values[0] if len(spectral_data[spectral_data['Prune_Ratio'] == 0.3]) > 0 else 0
spec_time_04 = spectral_data[spectral_data['Prune_Ratio'] == 0.4]['Training_Time_sec'].values[0] if len(spectral_data[spectral_data['Prune_Ratio'] == 0.4]) > 0 else 0

rand_time_02 = f"{random_02['Training_Time_sec'].mean():.2f}s" if len(random_02) > 0 else "N/A"
rand_time_03 = f"{random_03['Training_Time_sec'].mean():.2f}s" if len(random_03) > 0 else "N/A"
rand_time_04 = f"{random_04['Training_Time_sec'].mean():.2f}s" if len(random_04) > 0 else "N/A"

table_data.append(['Time (s)', f'{spec_time_02:.2f}s', rand_time_02, f'{spec_time_03:.2f}s', rand_time_03, f'{spec_time_04:.2f}s', rand_time_04])

# Create table
table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.15, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.5)

# Color header row
for i in range(len(table_data[0])):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')

ax.set_title('Comparison Summary', fontsize=12, fontweight='bold', pad=20)

# 4. Comparison box plot
ax = plt.subplot(3, 3, 8)
comparison_data = []
comparison_labels = []

for ratio in [0.2, 0.3, 0.4]:
    spectral_ratio = spectral_data[spectral_data['Prune_Ratio'] == ratio]
    if len(spectral_ratio) > 0:
        comparison_data.append(spectral_ratio['Accuracy_After_Finetuning'].values)
        comparison_labels.append(f'Spectral\n{ratio}')

if len(comparison_data) > 0:
    bp = ax.boxplot(comparison_data, tick_labels=comparison_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], [colors[0.2], colors[0.3], colors[0.4]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_ylabel('Accuracy After Finetuning (%)', fontsize=10)
    ax.set_title('Spectral Methods - Accuracy Distribution', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

# 5. Performance metrics comparison
ax = plt.subplot(3, 3, 9)
ratios = ['0.2', '0.3', '0.4']
spectral_accs = []
spectral_times = []

for ratio in [0.2, 0.3, 0.4]:
    spec_data = spectral_data[spectral_data['Prune_Ratio'] == ratio]
    if len(spec_data) > 0:
        spectral_accs.append(spec_data['Accuracy_After_Finetuning'].values[0])
        spectral_times.append(spec_data['Training_Time_sec'].values[0])
    else:
        spectral_accs.append(0)
        spectral_times.append(0)

x = np.arange(len(ratios))
width = 0.35

# Normalize for dual axis
accs_normalized = [acc / 100 * 40 for acc in spectral_accs]  # Scale to 0-40 range

bars1 = ax.bar(x - width/2, accs_normalized, width, label='Accuracy (×100)', 
              color='steelblue', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x + width/2, spectral_times, width, label='Training Time (s)', 
              color='coral', alpha=0.8, edgecolor='black')

ax.set_xlabel('Prune Ratio', fontsize=10)
ax.set_ylabel('Value', fontsize=10)
ax.set_title('Spectral Methods - Accuracy vs Training Time', fontsize=11, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(ratios)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}',
               ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('MNIST_Spectral_All_Ratios_visualization.png', dpi=300, bbox_inches='tight')
print("\nSpectral all ratios visualization saved as 'MNIST_Spectral_All_Ratios_visualization.png'")
plt.show()

# Print comprehensive statistics
print("\n" + "="*70)
print("SPECTRAL METHOD ANALYSIS - ALL PRUNE RATIOS")
print("="*70)

for ratio in [0.2, 0.3, 0.4]:
    spec_data = spectral_data[spectral_data['Prune_Ratio'] == ratio]
    rand_data = df[(df['Method'] == 'Random') & (df['Prune_Ratio'] == ratio)]
    
    print(f"\nPRUNE RATIO {ratio}")
    print("-" * 70)
    
    if len(spec_data) > 0:
        print(f"SPECTRAL {ratio}:")
        print(f"  Iteration: {spec_data['Iteration'].values[0]}")
        print(f"  Accuracy: {spec_data['Accuracy_After_Finetuning'].values[0]:.2f}%")
        print(f"  Baseline Accuracy: {spec_data['Baseline_Accuracy'].values[0]:.2f}%")
        print(f"  Training Time: {spec_data['Training_Time_sec'].values[0]:.2f} seconds")
        print(f"  Compression Rate: {spec_data['Compression_Rate'].values[0]:.4f}")
        print(f"  Parameters: {spec_data['Params'].values[0]:.0f}")
    else:
        print(f"SPECTRAL {ratio}: No data available")
    
    if len(rand_data) > 0:
        print(f"\nRANDOM {ratio} (n={len(rand_data)}):")
        print(f"  Mean Accuracy: {rand_data['Accuracy_After_Finetuning'].mean():.2f}% ± {rand_data['Accuracy_After_Finetuning'].std():.2f}%")
        print(f"  Accuracy Range: [{rand_data['Accuracy_After_Finetuning'].min():.2f}%, {rand_data['Accuracy_After_Finetuning'].max():.2f}%]")
        print(f"  Mean Training Time: {rand_data['Training_Time_sec'].mean():.2f} ± {rand_data['Training_Time_sec'].std():.2f} seconds")
        print(f"  Mean Compression Rate: {rand_data['Compression_Rate'].mean():.4f}")
        
        if len(spec_data) > 0:
            acc_diff = spec_data['Accuracy_After_Finetuning'].values[0] - rand_data['Accuracy_After_Finetuning'].mean()
            time_diff = spec_data['Training_Time_sec'].values[0] - rand_data['Training_Time_sec'].mean()
            print(f"\nSPECTRAL vs RANDOM COMPARISON:")
            print(f"  Accuracy Difference: {acc_diff:+.2f}% {'(Spectral better)' if acc_diff > 0 else '(Random better)'}")
            print(f"  Training Time Difference: {time_diff:+.2f}s {'(Random faster)' if time_diff > 0 else '(Spectral faster)'}")
    else:
        print(f"RANDOM {ratio}: No data available")

print("\n" + "="*70)
