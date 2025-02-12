import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def read_simulation_file(file_path):
    columns = ['axial_trapping_frequency', 'velocity', 'ion_collided_with', 'angleXY', 'angleZ', 'collision_offset', 'reorder']
    df = pd.read_csv(file_path, sep='\t', names=columns, skiprows=1)
    return df

def extract_info_from_filename(filename):
    try:
        # Format: "{Ni}ionSimulation_{Nr} x {Nz}_{shots}shots.txt"
        parts = filename.split('_')
        ion_size = int(parts[0].replace('ionSimulation', ''))
        grid_parts = parts[1].split('x')
        nr_size = int(grid_parts[0].strip())
        nz_size = int(grid_parts[1].strip())
        shot_size = int(parts[2].split('shots')[0])
        return ion_size, nr_size, nz_size, shot_size
    except Exception as e:
        print(f"Error parsing filename {filename}: {e}")
        return None, None, None, None

def calculate_probabilities(df):
    total_rows = len(df)
    if total_rows == 0:
        return 0, 0, 0
    
    reorder_count = (df['reorder'] == 1).sum()
    ejection_count = (df['reorder'] == 2).sum()
    combined_count = reorder_count + ejection_count
    
    reorder_prob = reorder_count / total_rows
    ejection_prob = ejection_count / total_rows
    combined_prob = combined_count / total_rows
    
    return reorder_prob, ejection_prob, combined_prob

def get_available_parameters(directory):
    all_files = glob.glob(os.path.join(directory, '*ionSimulation_*x*_*shots.txt'))
    ion_sizes = set()
    nr_sizes = set()
    nz_sizes = set()
    shot_sizes = set()
    
    for file in all_files:
        ion_size, nr_size, nz_size, shot_size = extract_info_from_filename(os.path.basename(file))
        if ion_size is not None:
            ion_sizes.add(ion_size)
            nr_sizes.add(nr_size)
            nz_sizes.add(nz_size)
            shot_sizes.add(shot_size)
    
    return (sorted(list(ion_sizes)), sorted(list(nr_sizes)), 
            sorted(list(nz_sizes)), sorted(list(shot_sizes)))

def get_user_selection(available_options, parameter_name):
    print(f"\nAvailable {parameter_name}s: {available_options}")
    print(f"Enter {parameter_name}s to plot (space-separated numbers) or press Enter for all:")
    user_input = input().strip()
    
    if not user_input:  # If user just pressed Enter
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

def analyze_simulations(directory, selected_ion_sizes, selected_nr_sizes, 
                       selected_nz_sizes, selected_shot_sizes):
    data = []
    all_files = glob.glob(os.path.join(directory, '*ionSimulation_*x*_*shots.txt'))
    
    for file in all_files:
        info = extract_info_from_filename(os.path.basename(file))
        if info[0] is None:
            continue
        
        ion_size, nr_size, nz_size, shot_size = info
        if (ion_size in selected_ion_sizes and 
            nr_size in selected_nr_sizes and
            nz_size in selected_nz_sizes and
            shot_size in selected_shot_sizes):
            df = read_simulation_file(file)
            reorder_prob, ejection_prob, combined_prob = calculate_probabilities(df)
            data.append((ion_size, nr_size, nz_size, shot_size, 
                        reorder_prob, ejection_prob, combined_prob))
    
    return pd.DataFrame(data, columns=['ion_size', 'nr_size', 'nz_size', 
                                     'shot_size', 'reorder_prob', 
                                     'ejection_prob', 'combined_prob'])

def get_plot_preference():
    while True:
        print("\nHow would you like to plot different Nr values?")
        print("1: All Nr values in the same plot")
        print("2: Separate plots for each Nr value")
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice in ['1', '2']:
            return choice == '1'
        print("Invalid choice. Please enter 1 or 2.")

def plot_results(df, combine_nr=True):
    plot_dir = "reorder_probability_plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    event_types = [
        ('reorder_prob', 'Reorder'),
        ('ejection_prob', 'Ejection'),
        ('combined_prob', 'Combined (Reorder + Ejection)')
    ]
    
    # Plot for each ion size
    for ion_size in df['ion_size'].unique():
        if combine_nr:
            # Create three subplots for each probability type
            fig, axes = plt.subplots(1, 3, figsize=(30, 8))
            data_subset = df[df['ion_size'] == ion_size]
            
            if not data_subset.empty:
                markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*']
                colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'orange', 
                         'purple', 'brown', 'gray', 'olive', 'pink', 'teal']
                
                for idx, (prob_col, event_name) in enumerate(event_types):
                    ax = axes[idx]
                    color_index = 0
                    
                    for i, nr_size in enumerate(sorted(data_subset['nr_size'].unique())):
                        nr_data = data_subset[data_subset['nr_size'] == nr_size]
                        
                        for shot_size in sorted(nr_data['shot_size'].unique()):
                            shot_data = nr_data[nr_data['shot_size'] == shot_size]
                            shot_data = shot_data.sort_values('nz_size')
                            
                            label = f'Nr={nr_size}, Shots={shot_size}'
                            
                            ax.plot(shot_data['nz_size'], shot_data[prob_col], 
                                    marker=markers[i % len(markers)],
                                    color=colors[color_index % len(colors)],
                                    label=label,
                                    markersize=8,
                                    linewidth=2)
                            
                            color_index += 1
                    
                    ax.set_xlabel('Number of Axial Grid Cells (Nz)')
                    ax.set_ylabel(f'Probability of {event_name}')
                    ax.set_title(f'{event_name} Probability vs Nz for {ion_size}-ion Simulation')
                    ax.set_xticks(sorted(data_subset['nz_size'].unique()))
                    ax.grid(True, linestyle='--', alpha=0.7)
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                
                plt.tight_layout(w_pad=2)
                plot_path = os.path.join(plot_dir, f'probability_plots_{ion_size}ion_combined.png')
                plt.savefig(plot_path, bbox_inches='tight', dpi=300)
                plt.close()
                
        else:
            # Separate plots for each Nr value
            for nr_size in sorted(df[df['ion_size'] == ion_size]['nr_size'].unique()):
                fig, axes = plt.subplots(1, 3, figsize=(30, 8))
                data_subset = df[(df['ion_size'] == ion_size) & (df['nr_size'] == nr_size)]
                
                if not data_subset.empty:
                    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'orange', 
                             'purple', 'brown', 'gray', 'olive', 'pink', 'teal']
                    
                    for idx, (prob_col, event_name) in enumerate(event_types):
                        ax = axes[idx]
                        
                        for i, shot_size in enumerate(sorted(data_subset['shot_size'].unique())):
                            shot_data = data_subset[data_subset['shot_size'] == shot_size]
                            shot_data = shot_data.sort_values('nz_size')
                            
                            label = f'Shots={shot_size}'
                            
                            ax.plot(shot_data['nz_size'], shot_data[prob_col], 
                                    marker='o',
                                    color=colors[i % len(colors)],
                                    label=label,
                                    markersize=8,
                                    linewidth=2)
                        
                        ax.set_xlabel('Number of Axial Grid Cells (Nz)')
                        ax.set_ylabel(f'Probability of {event_name}')
                        ax.set_title(f'{event_name} Probability vs Nz\n{ion_size}-ion Simulation, Nr: {nr_size}')
                        ax.set_xticks(sorted(data_subset['nz_size'].unique()))
                        ax.grid(True, linestyle='--', alpha=0.7)
                        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    
                    plt.tight_layout(w_pad=2)
                    plot_path = os.path.join(plot_dir, f'probability_plots_{ion_size}ion_Nr_{nr_size}.png')
                    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
                    plt.close()
                    
def main():
    script_dir = "simulation_results_10k"
    
    # Get available parameters
    ion_sizes, nr_sizes, nz_sizes, shot_sizes = get_available_parameters(script_dir)
    
    # Get user selections
    selected_nr_sizes = get_user_selection(nr_sizes, "Nr size")
    selected_nz_sizes = get_user_selection(nz_sizes, "Nz size")
    selected_shot_sizes = get_user_selection(shot_sizes, "shot size")
    selected_ion_sizes = get_user_selection(ion_sizes, "ion size")
    
    # Get plotting preference
    combine_nr = get_plot_preference()
    
    print("\nAnalyzing simulations with:")
    print(f"Nr sizes: {selected_nr_sizes}")
    print(f"Nz sizes: {selected_nz_sizes}")
    print(f"Shot sizes: {selected_shot_sizes}")
    print(f"Ion sizes: {selected_ion_sizes}")
    
    results = analyze_simulations(script_dir, selected_ion_sizes, selected_nr_sizes,
                                selected_nz_sizes, selected_shot_sizes)
    
    if results.empty:
        print("No data found for the selected parameters. Please check your selections.")
    else:
        plot_results(results, combine_nr)
        print("Analysis complete. Check the generated plots in the reorder_probability_plots directory.")

if __name__ == "__main__":
    main()