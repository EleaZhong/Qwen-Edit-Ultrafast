
import matplotlib.pyplot as plt
import pandas as pd

from qwenimage.debug import clear_cuda_memory, print_gpu_memory
from qwenimage.experiment import ExperimentConfig
from qwenimage.experiments.experiments_qwen import ExperimentRegistry, PipeInputs

experiment_names = ExperimentRegistry.keys()
print(experiment_names)

pipe_inputs = PipeInputs()

# Collect results from all experiments
all_results = []

for name in experiment_names:
    print(f"Running {name}")
    experiment = ExperimentRegistry.get(name)(
        config=ExperimentConfig(
            name=name,
            iterations=10,
        ), 
        pipe_inputs=pipe_inputs,
    )
    experiment.load()
    experiment.optimize()
    experiment.run()
    base_df, base_raw_data = experiment.report()
    
    # Add experiment name to the dataframe
    base_df['experiment'] = name
    all_results.append(base_df)

    experiment.cleanup()
    del experiment
    
    clear_cuda_memory()
    
    print_gpu_memory(clear_mem=None)

# Combine all results
combined_df = pd.concat(all_results, ignore_index=True)

# Define desired names to plot
desired_names = ["loop", "QwenBaseExperiment.run_once"]

# Filter for desired names
plot_data = combined_df[combined_df['name'].isin(desired_names)].copy()

print(plot_data)

# Sort by mean in descending order (rightmost = lowest mean)
plot_data = plot_data.sort_values('mean', ascending=False)

# Create bar plot
fig, ax = plt.subplots(figsize=(12, 6))

# Create x positions for bars
x_pos = range(len(plot_data))

# Plot bars with error bars
bars = ax.bar(x_pos, plot_data['mean'], yerr=plot_data['std'], 
               capsize=5, alpha=0.7, edgecolor='black')

# Customize plot
ax.set_xlabel('Method', fontsize=12, fontweight='bold')
ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
ax.set_title('Performance Comparison: Mean Execution Time with Standard Deviation', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([f"{row['experiment']}\n{row['name']}" 
                     for _, row in plot_data.iterrows()], 
                    rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)

# Add value labels on top of bars
for i, (idx, row) in enumerate(plot_data.iterrows()):
    ax.text(i, row['mean'] + row['std'], f"{row['mean']:.3f}s", 
            ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('reports/performance_comparison.png', dpi=300, bbox_inches='tight')
print("\nPerformance comparison plot saved to: reports/performance_comparison.png")
plt.show()


