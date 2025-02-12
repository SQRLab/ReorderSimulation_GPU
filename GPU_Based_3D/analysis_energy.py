import numpy as np
import matplotlib.pyplot as plt
import h5py
import glob
import re
from tqdm import tqdm
import datetime

def find_trajectory_files(shot_number=None):
    """Find all trajectory files in current directory"""
    all_files = glob.glob("trajectory_shot*dt*.h5")
    file_info = []
    for file in all_files:
        match = re.match(r'trajectory_shot(\d+)_dt(\d+(?:\.\d+)?(?:e-?\d+)?)', file)
        if match:
            shot_num = int(match.group(1))
            dt = float(match.group(2))
            file_info.append({
                'filename': file,
                'shot': shot_num,
                'dt': dt
            })
    
    if shot_number is not None:
        file_info = [f for f in file_info if f['shot'] == shot_number]
    return sorted(file_info, key=lambda x: x['dt'])

def get_user_selection():
    """Get user input for file selection"""
    all_files = find_trajectory_files()
    if not all_files:
        print("No trajectory files found!")
        return None, None
    
    shots = sorted(list(set(f['shot'] for f in all_files)))
    print("\nAvailable shot numbers:", shots)
    
    while True:
        try:
            shot = int(input("Enter shot number to analyze: "))
            if shot in shots:
                break
            print(f"Invalid shot number. Available shots are: {shots}")
        except ValueError:
            print("Please enter a valid number")
    
    shot_files = [f for f in all_files if f['shot'] == shot]
    print("\nAvailable dt values for shot", shot)
    for i, f in enumerate(shot_files):
        print(f"{i+1}: dt = {f['dt']:.2e}")
    
    selected_files = []
    while True:
        try:
            selection = input("\nSelect file numbers (comma-separated) or 'all': ").strip()
            if selection.lower() == 'all':
                selected_files = shot_files
                break
            else:
                indices = [int(idx.strip()) - 1 for idx in selection.split(',')]
                if all(0 <= idx < len(shot_files) for idx in indices):
                    selected_files = [shot_files[idx] for idx in indices]
                    break
                print("Invalid selection")
        except ValueError:
            print("Please enter valid numbers or 'all'")
    
    return shot, selected_files

def load_energy_data(filename):
    """Load energy data from trajectory file"""
    with h5py.File(filename, 'r') as f:
        timesteps = f['trajectory/timesteps'][:]
        num_valid = int(timesteps[-1]) if len(timesteps) > 0 else 0
        
        if num_valid == 0:
            print(f"Warning: No valid timesteps in {filename}")
            return None
        
        try:
            # Load positions
            ion1_pos = f['trajectory/ion1_positions'][:num_valid]
            ion2_pos = f['trajectory/ion2_positions'][:num_valid]
            coll_pos = f['trajectory/collision_positions'][:num_valid]
            
            # Load energies
            energies = f['energies']
            data = {
                'timesteps': timesteps[:num_valid],
                'ion1_positions': ion1_pos,
                'ion2_positions': ion2_pos,
                'collision_positions': coll_pos,
                'ion_KE': energies['ion_kinetic'][:num_valid],
                'ion_PE': energies['ion_potential'][:num_valid],
                'ion_total': energies['ion_total'][:num_valid],
                'collision_KE': energies['collision_kinetic'][:num_valid],
                'collision_PE': energies['collision_potential'][:num_valid],
                'collision_total': energies['collision_total'][:num_valid],
                'dt': f.attrs['dtSmall'],
                'result': f.attrs['reorder_result'],
                'termination': f.attrs['termination_reason'],
                'physical_time': f.attrs['actual_physical_time']
            }
            
            # Calculate some additional metrics
            data['ion_collision_separation'] = np.sqrt(
                np.sum((ion1_pos - coll_pos)**2, axis=1)
            )
            data['ion_separation'] = np.sqrt(
                np.sum((ion1_pos - ion2_pos)**2, axis=1)
            )
            
            return data
            
        except Exception as e:
            print(f"Error loading data from {filename}: {str(e)}")
            return None

def plot_energy_analysis(shot_num, file_data_list):
    """Create comprehensive energy analysis plots"""
    print("\nCreating energy analysis plots...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 20))
    fig.suptitle(f'Energy Analysis - Shot {shot_num}', fontsize=14, y=0.98)
    
    # Create subplot grid
    gs = plt.GridSpec(4, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])  # Ion energies
    ax2 = fig.add_subplot(gs[0, 1])  # Collision particle energies
    ax3 = fig.add_subplot(gs[1, 0])  # Total system energy
    ax4 = fig.add_subplot(gs[1, 1])  # Energy conservation error
    ax5 = fig.add_subplot(gs[2, :])  # Energy exchange
    ax6 = fig.add_subplot(gs[3, :])  # Ion-collision separation
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(file_data_list)))
    
    for i, (file_info, data) in enumerate(file_data_list):
        dt = data['dt']
        time = data['timesteps']
        label = f'dt={dt:.2e}'
        
        # Calculate initial energies for normalization
        E0_ion = data['ion_total'][0]
        E0_coll = data['collision_total'][0]
        E0_system = E0_ion + E0_coll
        
        if abs(E0_system) < 1e-50:
            print(f"Warning: Very small initial energy for dt={dt}")
            E0_system = 1.0
        
        # Convert time to nanoseconds
        time_ns = time * 1e9
        
        # Plot ion energies (log scale)
        ax1.semilogy(time_ns, np.abs(data['ion_KE'])/abs(E0_system), '-', 
                    color=colors[i], label=f'KE {label}')
        ax1.semilogy(time_ns, np.abs(data['ion_PE'])/abs(E0_system), '--', 
                    color=colors[i], label=f'PE {label}')
        ax1.semilogy(time_ns, np.abs(data['ion_total'])/abs(E0_system), ':', 
                    color=colors[i], label=f'Total {label}')
        
        # Plot collision particle energies (log scale)
        ax2.semilogy(time_ns, np.abs(data['collision_KE'])/abs(E0_system), '-', 
                    color=colors[i], label=f'KE {label}')
        ax2.semilogy(time_ns, np.abs(data['collision_PE'])/abs(E0_system), '--', 
                    color=colors[i], label=f'PE {label}')
        ax2.semilogy(time_ns, np.abs(data['collision_total'])/abs(E0_system), ':', 
                    color=colors[i], label=f'Total {label}')
        
        # Total system energy (log scale)
        total_system = data['ion_total'] + data['collision_total']
        ax3.semilogy(time_ns, np.abs(total_system)/abs(E0_system), 
                    color=colors[i], label=label)
        
        # Energy conservation error
        relative_error = (total_system - E0_system)/E0_system * 100
        ax4.plot(time_ns, relative_error, color=colors[i], label=label)
        
        # Energy exchange
        energy_exchange = (data['ion_total'] - E0_ion)/E0_system * 100
        ax5.plot(time_ns, energy_exchange, color=colors[i], label=label)
        
        # Ion-collision separation
        ax6.semilogy(time_ns, data['ion_collision_separation'], '-', 
                    color=colors[i], label=f'Ion-Collision {label}')
        ax6.semilogy(time_ns, data['ion_separation'], '--', 
                    color=colors[i], label=f'Ion-Ion {label}')
        
        # Print statistics
        print(f"\nStatistics for dt={dt:.2e}:")
        print(f"Simulation time: {time[-1]*1e9:.2f} ns")
        print(f"Mean energy error: {np.mean(relative_error):.2e}%")
        print(f"Max absolute energy error: {np.max(np.abs(relative_error)):.2e}%")
        print(f"Max energy exchange: {np.max(np.abs(energy_exchange)):.2e}%")
        print(f"Min ion-collision separation: {np.min(data['ion_collision_separation']):.2e} m")
        print(f"Mean ion separation: {np.mean(data['ion_separation']):.2e} m")
    
    # Format axes
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=8, loc='best')
        ax.set_xlabel('Time (ns)', fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=8)
    
    ax1.set_ylabel('Ion Energy / |E₀| (log)', fontsize=10)
    ax2.set_ylabel('Collision Energy / |E₀| (log)', fontsize=10)
    ax3.set_ylabel('Total Energy / |E₀| (log)', fontsize=10)
    ax4.set_ylabel('Energy Error (%)', fontsize=10)
    ax5.set_ylabel('Energy Exchange (%)', fontsize=10)
    ax6.set_ylabel('Separation (m) (log)', fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    return fig

def analyze_energy_conservation(file_data_list):
    """Print energy conservation analysis"""
    print("\nEnergy Conservation Analysis:")
    print("-" * 120)
    headers = ['dt', 'Time (ns)', 'Mean Error %', 'Max Error %', 'RMS Error %', 
              'Max Exchange %', 'Min Sep. (m)', 'Term. Reason']
    print(f"{headers[0]:<12} {headers[1]:<15} {headers[2]:<15} {headers[3]:<15} "
          f"{headers[4]:<15} {headers[5]:<15} {headers[6]:<15} {headers[7]:<15}")
    print("-" * 120)
    
    for file_info, data in file_data_list:
        dt = data['dt']
        time = data['timesteps'][-1] * 1e9
        
        # Energy calculations
        E0_ion = data['ion_total'][0]
        E0_coll = data['collision_total'][0]
        E0_system = E0_ion + E0_coll
        total_system = data['ion_total'] + data['collision_total']
        
        # Error metrics
        if abs(E0_system) > 1e-50:
            relative_error = (total_system - E0_system)/E0_system * 100
            energy_exchange = (data['ion_total'] - E0_ion)/E0_system * 100
        else:
            relative_error = (total_system - E0_system) * 100
            energy_exchange = (data['ion_total'] - E0_ion) * 100
        
        mean_error = np.mean(relative_error)
        max_error = np.max(np.abs(relative_error))
        rms_error = np.sqrt(np.mean(relative_error**2))
        max_exchange = np.max(np.abs(energy_exchange))
        min_separation = np.min(data['ion_collision_separation'])
        
        print(f"{dt:<12.2e} {time:<15.2f} {mean_error:<15.2e} {max_error:<15.2e} "
              f"{rms_error:<15.2e} {max_exchange:<15.2e} {min_separation:<15.2e} "
              f"{data['termination']:<15d}")

def main():
    """Main analysis function"""
    shot_num, selected_files = get_user_selection()
    if not selected_files:
        return
    
    print("\nLoading and analyzing data...")
    file_data_list = []
    
    for file_info in selected_files:
        try:
            data = load_energy_data(file_info['filename'])
            if data is not None:
                file_data_list.append((file_info, data))
        except Exception as e:
            print(f"Error loading {file_info['filename']}: {str(e)}")
    
    if not file_data_list:
        print("No valid data files loaded!")
        return
    
    # Create and save plots
    try:
        fig = plot_energy_analysis(shot_num, file_data_list)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        plot_filename = f'energy_analysis_shot{shot_num}_{timestamp}.png'
        fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved as: {plot_filename}")
    except Exception as e:
        print(f"Error creating plots: {str(e)}")
        print(f"Exception details: {str(e)}")
    
    # Print analysis
    try:
        analyze_energy_conservation(file_data_list)
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        print(f"Exception details: {str(e)}")
    
    plt.show()

if __name__ == "__main__":
    main()