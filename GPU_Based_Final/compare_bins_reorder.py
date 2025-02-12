import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def load_file(filename):
    """Load a simulation file and return the reorder column"""
    data = np.loadtxt(filename, skiprows=1)
    return data[:, -1]  # Last column contains reorder info

def calculate_reorder_prob(reorder_column):
    """Calculate reorder probability from a column of reorder values"""
    return np.sum(reorder_column != 0) / len(reorder_column)

def analyze_binned_data(reorder_column, bin_size=10000):
    """Break data into bins and calculate mean and std of reorder probability"""
    n_complete_bins = len(reorder_column) // bin_size
    binned_probs = []
    
    for i in range(n_complete_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size
        bin_data = reorder_column[start_idx:end_idx]
        binned_probs.append(calculate_reorder_prob(bin_data))
    
    return np.mean(binned_probs), np.std(binned_probs)

def create_plots(base_folder, grid_sizes, n_ions_list, shot_sizes):
    """Create combined plot with both analyses"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use same color for both lines but different styles
    color = '#1f77b4'  # Blue
    
    for idx, n_ions in enumerate(n_ions_list):
        for shot_size in shot_sizes:
            probs = []  # For direct probability
            means = []  # For binned analysis
            errors = []  # For binned analysis
            x_values = []
            
            for grid_size in grid_sizes:
                filename = os.path.join(base_folder, 
                                      f"{n_ions}ionSimulation_{grid_size}_{shot_size}shots.txt")
                if os.path.exists(filename):
                    reorder_col = load_file(filename)
                    prob = calculate_reorder_prob(reorder_col)
                    mean, std = analyze_binned_data(reorder_col)
                    
                    probs.append(prob)
                    means.append(mean)
                    errors.append(std)
                    x_values.append(grid_size)
            
            if probs:
                # Plot direct probability with very thick black dashed line
                ax.plot(x_values, probs, '--', color='black', linewidth=3,
                       label=f'{n_ions} ions (Direct)', dashes=(5, 5))
                
                # Plot binned analysis with error bars in blue
                ax.errorbar(x_values, means, yerr=errors, fmt='o-', color=color,
                          label=f'{n_ions} ions (Binned)', 
                          capsize=5, capthick=1, elinewidth=1, markersize=8)
    
    ax.set_xlabel('Number of cells', fontsize=12)
    ax.set_ylabel('Reorder Probability', fontsize=12)
    ax.set_title(f'Reorder Probability Analysis {shot_size}\n(Direct vs Binned 10k shots)', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Set y-axis limits with padding
    ymin = min(ax.get_ylim()[0], 0.46)
    ymax = max(ax.get_ylim()[1], 0.48)
    ax.set_ylim(ymin, ymax)
    
    # Customize x-axis ticks
    ax.set_xticks(grid_sizes)
    ax.set_xticklabels(grid_sizes)
    
    # Make legend more prominent
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc='best', fontsize=10, framealpha=1)
    
    plt.tight_layout()
    plt.savefig('reorder_probability_analysis_timestep.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    base_folder = "simulation_results_10k_timestep"
    grid_sizes = [11, 101, 1001]
    n_ions_list = [2]
    shot_sizes = [100000]
    
    create_plots(base_folder, grid_sizes, n_ions_list, shot_sizes)

if __name__ == "__main__":
    main()