import json
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the JSON files
with open('computation_times.json') as f:
    gpu_times = json.load(f)

with open('computation_times_cpu.json') as f:
    cpu_times = json.load(f)

# Extract relevant data for plotting
grid_sizes = list(gpu_times.keys())
shot_counts = [100, 200, 500, 1000]
ion_counts = [2, 3]

# Function to generate and save heatmaps for each ion count, including speed-up factor
def create_heatmaps(ion_count):
    # Prepare data for the specified ion count
    grid_size_shot_data = {
        "Grid Size": [],
        "Shot Count": [],
        "CPU Time": [],
        "GPU Time": [],
        "Speed-Up Factor": []
    }
    
    for grid in grid_sizes:
        for shot in shot_counts:
            cpu_time = cpu_times[grid][str(ion_count)][str(shot)]
            gpu_time = gpu_times[grid][str(ion_count)][str(shot)]
            speed_up = cpu_time / gpu_time if gpu_time > 0 else float('inf')  # Handle division by zero

            grid_size_shot_data["Grid Size"].append(int(grid))
            grid_size_shot_data["Shot Count"].append(shot)
            grid_size_shot_data["CPU Time"].append(cpu_time)
            grid_size_shot_data["GPU Time"].append(gpu_time)
            grid_size_shot_data["Speed-Up Factor"].append(speed_up)

    # Create a DataFrame for heatmap data
    df_times = pd.DataFrame(grid_size_shot_data)

    # Pivot tables for CPU, GPU, and Speed-Up Factor data
    cpu_heatmap_data = df_times.pivot_table(values="CPU Time", index="Grid Size", columns="Shot Count", aggfunc='mean')
    gpu_heatmap_data = df_times.pivot_table(values="GPU Time", index="Grid Size", columns="Shot Count", aggfunc='mean')
    speedup_heatmap_data = df_times.pivot_table(values="Speed-Up Factor", index="Grid Size", columns="Shot Count", aggfunc='mean')

    # Plot heatmaps and save to files
    plt.figure(figsize=(18, 6))

    # CPU Time Heatmap
    plt.subplot(1, 3, 1)
    sns.heatmap(cpu_heatmap_data, annot=True, cmap="Blues", fmt=".1f")
    plt.title(f"CPU Computation Time (s) for Ion Count {ion_count}")
    plt.xlabel("Shot Count")
    plt.ylabel("Grid Size")

    # GPU Time Heatmap
    plt.subplot(1, 3, 2)
    sns.heatmap(gpu_heatmap_data, annot=True, cmap="Oranges", fmt=".1f")
    plt.title(f"GPU Computation Time (s) for Ion Count {ion_count}")
    plt.xlabel("Shot Count")
    plt.ylabel("Grid Size")

    # Speed-Up Factor Heatmap
    plt.subplot(1, 3, 3)
    sns.heatmap(speedup_heatmap_data, annot=True, cmap="Greens", fmt=".1f", cbar_kws={'label': 'Speed-Up Factor'})
    plt.title(f"Speed-Up Factor (CPU/GPU) for Ion Count {ion_count}")
    plt.xlabel("Shot Count")
    plt.ylabel("Grid Size")

    plt.tight_layout()
    plt.savefig(f'computation_time_heatmaps_ion_{ion_count}.png')
    plt.close()

# Generate heatmaps for each ion count and save to files
for ion in ion_counts:
    create_heatmaps(ion)

print("Heatmaps, including speed-up factors, saved for each ion count as separate files.")