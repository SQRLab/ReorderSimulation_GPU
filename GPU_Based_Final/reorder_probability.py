import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def read_simulation_file(file_path):
    columns = ['axial_trapping_frequency', 'velocity', 'ion_collided_with', 'angle', 'collision_offset', 'reorder']
    df = pd.read_csv(file_path, sep='\t', names=columns, skiprows=1)
    return df

def extract_info_from_filename(filename):
    try:
        parts = filename.split('_')
        ion_size = int(parts[0].replace('ionSimulation', ''))
        cell_size = int(parts[1])
        shot_size = int(parts[2].split('.')[0].replace('shots', ''))
        return ion_size, cell_size, shot_size
    except Exception as e:
        return None, None, None

def calculate_probability(df):
    total_rows = len(df)
    non_zero_count = (df['reorder'] != 0).sum()
    prob = non_zero_count / (total_rows - 1) if total_rows > 1 else 0
    return prob

def get_available_parameters(directory):
    all_files = glob.glob(os.path.join(directory, '*ionSimulation_*_*shots.txt'))
    ion_sizes = set()
    cell_sizes = set()
    shot_sizes = set()
    
    for file in all_files:
        ion_size, cell_size, shot_size = extract_info_from_filename(os.path.basename(file))
        if ion_size is not None:
            ion_sizes.add(ion_size)
            cell_sizes.add(cell_size)
            shot_sizes.add(shot_size)
    
    return sorted(list(ion_sizes)), sorted(list(cell_sizes)), sorted(list(shot_sizes))

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

def analyze_simulations(directory, selected_ion_sizes, selected_cell_sizes, selected_shot_sizes):
    data = []
    all_files = glob.glob(os.path.join(directory, '*ionSimulation_*_*shots.txt'))
    
    for file in all_files:
        info = extract_info_from_filename(os.path.basename(file))
        if info[0] is None:
            continue
        
        ion_size, cell_size, shot_size = info
        if (ion_size in selected_ion_sizes and 
            cell_size in selected_cell_sizes and 
            shot_size in selected_shot_sizes):
            df = read_simulation_file(file)
            probability = calculate_probability(df)
            data.append((ion_size, cell_size, shot_size, probability))
    
    return pd.DataFrame(data, columns=['ion_size', 'cell_size', 'shot_size', 'probability'])

def plot_results(df):
    # Create a directory for the plots if it doesn't exist
    plot_dir = r"C:\Users\caleb\OneDrive\Desktop\Raahul's Workspace\ReorderSimulation_GPU\GPU_Based_Final\reorder_probability_plots"
    os.makedirs(plot_dir, exist_ok=True)

    # Plot for each ion size
    for ion_size in df['ion_size'].unique():
        plt.figure(figsize=(10, 6))
        ion_data = df[df['ion_size'] == ion_size]
        
        if not ion_data.empty:
            for shot_size in ion_data['shot_size'].unique():
                subset = ion_data[ion_data['shot_size'] == shot_size]
                plt.plot(subset['cell_size'], subset['probability'], 
                        marker='o', label=f'{shot_size} shots')
            
            plt.xlabel('Cell Size')
            plt.ylabel('Probability of Ejection or Reorder')
            plt.title(f'Probability vs Cell Size for {ion_size}-ion GPU Simulation')
            plt.legend()
            plt.grid(True)
            
            # Save the plot
            plot_path = os.path.join(plot_dir, f'reorder_prob_plot_{ion_size}ion.png')
            plt.savefig(plot_path)
            plt.close()

def main():
    # Set the specific directory path
    script_dir = r"C:\Users\caleb\OneDrive\Desktop\Raahul's Workspace\ReorderSimulation_GPU\GPU_Based_Final\simulation_results_10k"
    
    # Get available parameters
    ion_sizes, cell_sizes, shot_sizes = get_available_parameters(script_dir)
    
    # Get user selections
    selected_cell_sizes = get_user_selection(cell_sizes, "cell size")
    selected_shot_sizes = get_user_selection(shot_sizes, "shot size")
    selected_ion_sizes = get_user_selection(ion_sizes, "ion size")
    
    print("\nAnalyzing simulations with:")
    print(f"Cell sizes: {selected_cell_sizes}")
    print(f"Shot sizes: {selected_shot_sizes}")
    print(f"Ion sizes: {selected_ion_sizes}")
    
    results = analyze_simulations(script_dir, selected_ion_sizes, selected_cell_sizes, selected_shot_sizes)
    
    if results.empty:
        print("No data found for the selected parameters. Please check your selections.")
    else:
        plot_results(results)
        print("Analysis complete. Check the generated plots in the reorder_probability_plots directory.")

if __name__ == "__main__":
    main()