import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def read_simulation_file(file_path):
    columns = ['axial_trapping_frequency', 'velocity', 'ion_collided_with', 'angle', 'collision_offset', 'reorder']
    df = pd.read_csv(file_path, sep='\t', names=columns, skiprows=1)
    #print(f"Read file: {file_path}")
    #print(f"Shape: {df.shape}")
    #print(df.head())
    return df

def extract_info_from_filename(filename):
    try:
        parts = filename.split('_')
        ion_size = int(parts[0].replace('ionSimulation', ''))
        cell_size = int(parts[1])
        shot_size = int(parts[2].split('.')[0].replace('shots', ''))
        #print(f"Extracted info: ion_size={ion_size}, cell_size={cell_size}, shot_size={shot_size}")
        return ion_size, cell_size, shot_size
    except Exception as e:
        #print(f"Error extracting info from filename '{filename}': {str(e)}")
        return None, None, None

def calculate_probability(df):
    total_rows = len(df)
    non_zero_count = (df['reorder'] != 0).sum()
    prob = non_zero_count / (total_rows - 1) if total_rows > 1 else 0
    #print(f"Probability calculation: non_zero_count={non_zero_count}, total_rows={total_rows}, prob={prob}")
    return prob

def analyze_simulations(directory, chosen_ion_size):
    data = []
    all_files = glob.glob(os.path.join(directory, '*ionSimulation_*_*shots.txt'))
    #print(f"Found {len(all_files)} files matching the pattern")
    
    for file in all_files:
        #print(f"\nProcessing file: {file}")
        info = extract_info_from_filename(os.path.basename(file))
        if info[0] is None:
            continue
        ion_size, cell_size, shot_size = info
        if ion_size == chosen_ion_size:
            df = read_simulation_file(file)
            probability = calculate_probability(df)
            data.append((cell_size, shot_size, probability))
        #else:
         #   print(f"Skipping file (ion size mismatch): {file}")
    
    result_df = pd.DataFrame(data, columns=['cell_size', 'shot_size', 'probability'])
    #print("\nFinal DataFrame:")
    #print(result_df)
    return result_df

def plot_results(df, chosen_ion_size):
    plt.figure(figsize=(10, 6))
    if not df.empty:
        for shot_size in df['shot_size'].unique():
            subset = df[df['shot_size'] == shot_size]
            plt.plot(subset['cell_size'], subset['probability'], marker='o', label=f'{shot_size} shots')
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'No data to plot', ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.xlabel('Cell Size')
    plt.ylabel('Probability of Ejection or Reorder')
    plt.title(f'Probability vs Cell Size for {chosen_ion_size}-ion GPU Simulation')
    plt.grid(True)
    plot_filename = f'ion_simulation_gpu_plot_{chosen_ion_size}ion.png'
    plt.savefig(plot_filename)
    plt.close()

# Main execution
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    #print(f"Script directory: {script_dir}")
    
    directory = script_dir
    chosen_ion_size = 10
    
    #print(f"Analyzing simulations for {chosen_ion_size}-ion in directory: {directory}")
    #print(f"Current working directory: {os.getcwd()}")
    #print(f"Directory contents: {os.listdir(directory)}")
    
    results = analyze_simulations(directory, chosen_ion_size)
    plot_results(results, chosen_ion_size)

    if results.empty:
        print("No data found for the chosen ion size. Please check your files and chosen_ion_size value.")
    else:
        print("Analysis complete. Check the generated plot.")