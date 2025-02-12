import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_computation_times(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def get_available_parameters(data):
    nr_sizes = sorted([int(nr) for nr in data.keys()])
    nz_sizes = sorted(set(int(nz) for nr in data.values() for nz in nr.keys()))
    ion_counts = sorted(set(int(ion) for nr in data.values() 
                          for nz in nr.values() for ion in nz.keys()))
    shot_sizes = sorted(set(int(shot) for nr in data.values() 
                           for nz in nr.values() for ion in nz.values() 
                           for shot in ion.keys()))
    return nr_sizes, nz_sizes, ion_counts, shot_sizes

def get_user_selection(available_options, parameter_name):
    print(f"\nAvailable {parameter_name}s: {available_options}")
    print(f"Enter {parameter_name}s to plot (space-separated numbers) or press Enter for all:")
    user_input = input().strip()
    
    if not user_input:
        return available_options
    
    try:
        selected = [int(x) for x in user_input.split()]
        valid_selections = [x for x in selected if x in available_options]
        if not valid_selections:
            print(f"No valid {parameter_name}s selected. Using all available options.")
            return available_options
        return valid_selections
    except ValueError:
        print(f"Invalid input. Using all available {parameter_name}s.")
        return available_options

def calculate_average_rf_dc_time(data, nr, nz, ion_count):
    times = []
    for shot_size in data[str(nr)][str(nz)][str(ion_count)].keys():
        times.append(data[str(nr)][str(nz)][str(ion_count)][shot_size]["rf_dc_time"])
    return np.mean(times)

def plot_computation_times(data, selected_nr, selected_nz, selected_ions, selected_shots):
    plot_dir = "computation_time_plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    for nr in selected_nr:
        fig, axes = plt.subplots(1, 3, figsize=(30, 8))
        
        # Colors and markers for consistency
        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_ions) * len(selected_shots)))
        markers = ['o', 's', '^', 'D', 'v']
        
        # Plot 1: RF/DC Time (averaged)
        ax = axes[0]
        color_idx = 0
        for ion in selected_ions:
            rf_dc_times = []
            nz_values = []
            for nz in selected_nz:
                try:
                    avg_time = calculate_average_rf_dc_time(data, nr, nz, ion)
                    rf_dc_times.append(avg_time)
                    nz_values.append(nz)
                except KeyError:
                    continue
            
            if nz_values:
                ax.plot(nz_values, rf_dc_times, 
                       marker='o', label=f'Ion Count={ion}',
                       color=colors[color_idx], linewidth=2, markersize=8)
                color_idx += len(selected_shots)
        
        ax.set_xlabel('Number of Axial Grid Cells (Nz)')
        ax.set_ylabel('Average RF/DC Time (seconds)')
        ax.set_title(f'Average RF/DC Time vs Nz (Nr={nr})')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 2: Simulation Time
        ax = axes[1]
        color_idx = 0
        for ion in selected_ions:
            for shot in selected_shots:
                sim_times = []
                nz_values = []
                for nz in selected_nz:
                    try:
                        time = data[str(nr)][str(nz)][str(ion)][str(shot)]["simulation_time"]
                        sim_times.append(time)
                        nz_values.append(nz)
                    except KeyError:
                        continue
                
                if nz_values:
                    ax.plot(nz_values, sim_times, 
                           marker='o', label=f'Ion={ion}, Shots={shot}',
                           color=colors[color_idx], linewidth=2, markersize=8)
                    color_idx += 1
        
        ax.set_xlabel('Number of Axial Grid Cells (Nz)')
        ax.set_ylabel('Simulation Time (seconds)')
        ax.set_title(f'Simulation Time vs Nz (Nr={nr})')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 3: Total Time
        ax = axes[2]
        color_idx = 0
        for ion in selected_ions:
            for shot in selected_shots:
                total_times = []
                nz_values = []
                for nz in selected_nz:
                    try:
                        time = data[str(nr)][str(nz)][str(ion)][str(shot)]["total_time"]
                        total_times.append(time)
                        nz_values.append(nz)
                    except KeyError:
                        continue
                
                if nz_values:
                    ax.plot(nz_values, total_times, 
                           marker='o', label=f'Ion={ion}, Shots={shot}',
                           color=colors[color_idx], linewidth=2, markersize=8)
                    color_idx += 1
        
        ax.set_xlabel('Number of Axial Grid Cells (Nz)')
        ax.set_ylabel('Total Time (seconds)')
        ax.set_title(f'Total Time vs Nz (Nr={nr})')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout(w_pad=2)
        plot_path = os.path.join(plot_dir, f'computation_times_ion_{ion}_Nr_{nr}.png')
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()

def main():
    data = load_computation_times("computation_times_detailed.json")
    
    # Get available parameters and user selections
    nr_sizes, nz_sizes, ion_counts, shot_sizes = get_available_parameters(data)
    
    selected_nr = get_user_selection(nr_sizes, "Nr size")
    selected_nz = get_user_selection(nz_sizes, "Nz size")
    selected_ions = get_user_selection(ion_counts, "ion count")
    selected_shots = get_user_selection(shot_sizes, "shot size")
    
    print("\nGenerating plots with:")
    print(f"Nr sizes: {selected_nr}")
    print(f"Nz sizes: {selected_nz}")
    print(f"Ion counts: {selected_ions}")
    print(f"Shot sizes: {selected_shots}")
    
    plot_computation_times(data, selected_nr, selected_nz, selected_ions, selected_shots)
    print("\nPlots have been generated in the computation_time_plots directory.")

if __name__ == "__main__":
    main()