import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the CSV files
file_path = os.path.join(os.getcwd(), 'MNIST_random_20pct.csv')
spectral_path = os.path.join(os.getcwd(), 'MNIST_spectral_20pct.csv')

df = pd.read_csv(file_path)
df_spectral = pd.read_csv(spectral_path)

# Extract relevant columns
y = df['Accuracy_After_Finetuning']
baseline_accuracy = df['Baseline_Accuracy'].iloc[0]
spectral_accuracy = df_spectral['Accuracy_After_Finetuning'].iloc[0]  # ✅ pick a single value

# Create numeric x-axis labels (1, 2, 3, ...)
x = range(1, len(y) + 1)

# Plot bar chart
plt.figure(figsize=(10,6))
plt.bar(x, y, color='skyblue', label='Random Model Accuracy')

# Add baseline line
plt.axhline(y=baseline_accuracy, color='red', linestyle='--', linewidth=2,
            label=f'Baseline Accuracy ({baseline_accuracy:.2f})')

# Add spectral line
plt.axhline(y=spectral_accuracy, color='blue', linestyle='--', linewidth=2,
            label=f'Spectral Accuracy ({spectral_accuracy:.2f})')

# Labels and title
plt.xlabel('Experiment Number')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Random Pruning')
plt.ylim(97, 99)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
