import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = 'MNIST_random_30pct.csv'
df = pd.read_csv(file_path)

# Extract relevant columns
y = df['Accuracy_After_Finetuning']
baseline_accuracy = df['Baseline_Accuracy'].iloc[0]

# Create numeric x-axis labels (1, 2, 3, ...)
x = range(1, len(y) + 1)

# Plot bar chart
plt.figure(figsize=(10,6))
plt.bar(x, y, color='skyblue', label='Random Model Accuracy')

# Add baseline line
plt.axhline(y=baseline_accuracy, color='red', linestyle='--', linewidth=2,
            label=f'Baseline Accuracy ({baseline_accuracy:.2f})')

# Add labels and title
plt.xlabel('Experiment Number')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Random Pruning')
plt.ylim(97, 99)  # <<<<<< Changed y-axis range
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
