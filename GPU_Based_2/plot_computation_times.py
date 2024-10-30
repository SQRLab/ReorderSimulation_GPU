import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def read_computation_times(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def dynamic_polynomial(x, *params):
    return sum(param * x**i for i, param in enumerate(params))

def get_available_parameters(computation_times):
    """Extract all available ion counts and shot sizes from the data."""
    ion_counts = sorted(set(int(count) for size in computation_times.values() for count in size.keys()))
    shot_sizes = sorted(set(int(shots) for size in computation_times.values() 
                        for count in size.values() for shots in count.keys()))
    return ion_counts, shot_sizes

def plot_computation_times(computation_times, selected_ion_counts, selected_shot_sizes, polynomial_degree=2):
    """
    Plot computation times for selected ion counts and shot sizes.
    
    Args:
        computation_times (dict): The computation times data
        selected_ion_counts (list): List of ion counts to plot
        selected_shot_sizes (list): List of shot sizes to plot for each ion count
        polynomial_degree (int): Degree of polynomial fit
    """
    grid_sizes = sorted(set(int(size) for size in computation_times.keys()))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(selected_shot_sizes)))
    
    for ion_count in selected_ion_counts:
        plt.figure(figsize=(12, 8))
        plt.title(f"Computation Time vs Cell Size (Ion Count: {ion_count})")
        plt.xlabel("Cell Size")
        plt.ylabel("Computation Time (seconds)")
        
        for shot_size, color in zip(selected_shot_sizes, colors):
            x_data = []
            y_data = []
            for grid_size in grid_sizes:
                time = computation_times.get(str(grid_size), {}).get(str(ion_count), {}).get(str(shot_size), None)
                if time is not None:
                    x_data.append(grid_size)
                    y_data.append(time)
                    
            if x_data and y_data:
                plt.scatter(x_data, y_data, label=f"Shots: {shot_size}", color=color)
                # Fit curve
                popt, _ = curve_fit(lambda x, *params: dynamic_polynomial(x, *params), 
                                  x_data, y_data, p0=[1]*(polynomial_degree+1))
                x_fit = np.linspace(min(x_data), max(x_data), 100)
                y_fit = dynamic_polynomial(x_fit, *popt)
                plt.plot(x_fit, y_fit, '--', color=color)
        
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.tight_layout()
        plt.savefig(f"computation_time_plot_ion_count_{ion_count}.png")
        plt.close()

def main():
    # Read the data
    computation_times = read_computation_times("computation_times.json")
    
    # Get available parameters
    available_ion_counts, available_shot_sizes = get_available_parameters(computation_times)
    
    # Display available options
    print("\nAvailable ion counts:", available_ion_counts)
    print("Available shot sizes:", available_shot_sizes)
    
    # Get user input for ion counts
    print("\nEnter ion counts to plot (space-separated numbers):")
    print("Example: 2 4 6")
    selected_ion_counts = [int(x) for x in input().split()]
    
    # Validate ion counts
    selected_ion_counts = [count for count in selected_ion_counts if count in available_ion_counts]
    if not selected_ion_counts:
        print("No valid ion counts selected. Exiting.")
        return
    
    # Get user input for shot sizes
    print("\nEnter shot sizes to plot (space-separated numbers):")
    print("Example: 100 500 1000")
    selected_shot_sizes = [int(x) for x in input().split()]
    
    # Validate shot sizes
    selected_shot_sizes = [size for size in selected_shot_sizes if size in available_shot_sizes]
    if not selected_shot_sizes:
        print("No valid shot sizes selected. Exiting.")
        return
    
    # Create plots
    plot_computation_times(computation_times, selected_ion_counts, selected_shot_sizes)
    print(f"\nPlots have been generated and saved as PNG files for ion counts: {selected_ion_counts}")

if __name__ == "__main__":
    main()