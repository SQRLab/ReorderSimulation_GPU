import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
import re

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
    return file_info

def get_user_selection():
    """Get user input for file selection and plotting options"""
    all_files = find_trajectory_files()
    if not all_files:
        print("No trajectory files found!")
        return None, None, False
    
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
    
    while True:
        try:
            idx1 = int(input("\nSelect first file number: ")) - 1
            if 0 <= idx1 < len(shot_files):
                break
            print("Invalid selection")
        except ValueError:
            print("Please enter a valid number")
    
    # Ask if user wants to plot a second trajectory
    while True:
        plot_second = input("\nDo you want to plot a second trajectory? (y/n): ").lower()
        if plot_second in ['y', 'n']:
            break
        print("Please enter 'y' or 'n'")
    
    idx2 = None
    if plot_second == 'y':
        while True:
            try:
                idx2 = int(input("Select second file number: ")) - 1
                if 0 <= idx2 < len(shot_files) and idx2 != idx1:
                    break
                print("Invalid selection or same as first file")
            except ValueError:
                print("Please enter a valid number")
    
    while True:
        show_collision = input("\nShow collision particle? (y/n): ").lower()
        if show_collision in ['y', 'n']:
            break
        print("Please enter 'y' or 'n'")
    
    return shot_files[idx1]['filename'], idx2 and shot_files[idx2]['filename'], show_collision == 'y'

def load_and_downsample(file, target_points=10000):
    """Load and downsample trajectory data from a file"""
    with h5py.File(file, 'r') as f:
        t = f['trajectory/timesteps'][:]
        length = len(t)
        stride = max(1, length // target_points)
        print(f"File {file}: Length = {length}, stride = {stride}")
        
        t = t[::stride]
        ion1 = f['trajectory/ion1_positions'][::stride]
        ion2 = f['trajectory/ion2_positions'][::stride]
        coll = f['trajectory/collision_positions'][::stride]
        dt = f.attrs['dtSmall']
        
        return t, ion1, ion2, coll, dt

def plot_trajectories(file1, file2=None, show_collision=True, target_points=10000):
    """Create plots comparing trajectories"""
    print("Loading and processing data...")
    
    # Load data for first file
    t1, ion1_1, ion2_1, coll1, dt1 = load_and_downsample(file1, target_points)
    
    # Load data for second file if provided
    if file2:
        t2, ion1_2, ion2_2, coll2, dt2 = load_and_downsample(file2, target_points)
    
    # Load result values
    with h5py.File(file1, 'r') as f1:
        result1 = interpret_result(f1.attrs['reorder_result'])
        shot_num = f1.attrs['shot_index']
        
    if file2:
        with h5py.File(file2, 'r') as f2:
            result2 = interpret_result(f2.attrs['reorder_result'])
            title_text = f'Shot {shot_num}\ndt={dt1}: {result1}\ndt={dt2}: {result2}'
    else:
        title_text = f'Shot {shot_num}\ndt={dt1}: {result1}'
    
    print("Creating plots...")
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(title_text, fontsize=12, y=0.98)
    
    # Determine y-axis limits for each coordinate and ion
    y_limits = []
    for i in range(3):  # for x, y, z coordinates
        ion1_min = np.min(ion1_1[:, i])
        ion1_max = np.max(ion1_1[:, i])
        ion2_min = np.min(ion2_1[:, i])
        ion2_max = np.max(ion2_1[:, i])
        
        if file2:
            ion1_min = min(ion1_min, np.min(ion1_2[:, i]))
            ion1_max = max(ion1_max, np.max(ion1_2[:, i]))
            ion2_min = min(ion2_min, np.min(ion2_2[:, i]))
            ion2_max = max(ion2_max, np.max(ion2_2[:, i]))
        
        # Set individual limits for each ion
        padding1 = (ion1_max - ion1_min) * 0.1
        padding2 = (ion2_max - ion2_min) * 0.1
        y_limits.append({
            'ion1': (ion1_min - padding1, ion1_max + padding1),
            'ion2': (ion2_min - padding2, ion2_max + padding2)
        })
    
    # Create plots
    coords = ['x', 'y', 'z']
    particles = [('Ion1', ion1_1, ion2_1), ('Ion2', ion2_1, ion2_1)]
    if show_collision:
        particles.append(('Collision', coll1, coll1))
    
    for idx, (particle_name, data1, data2) in enumerate(particles):
        for i, coord in enumerate(coords):
            plot_num = idx * 3 + i + 1
            ax = fig.add_subplot(len(particles), 3, plot_num)
            
            # Plot first trajectory
            ax.plot(t1, data1[:, i], 'b-', label=f'dt={dt1}', linewidth=1)
            
            # Plot second trajectory if provided
            if file2:
                data2 = [ion1_2, ion2_2, coll2][idx] if show_collision else [ion1_2, ion2_2][idx]
                ax.plot(t2, data2[:, i], 'r--', label=f'dt={dt2}', linewidth=1)
            
            # Set y-axis limits based on which ion we're plotting
            if particle_name == 'Ion1':
                ax.set_ylim(y_limits[i]['ion1'])
            elif particle_name == 'Ion2':
                ax.set_ylim(y_limits[i]['ion2'])
            
            # Format axes
            ax.set_xlabel('Time (s)', fontsize=10)
            ax.set_ylabel(f'{coord} position (m)', fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='best', fontsize=8)
            ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
            
            # Adjust tick parameters
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.tick_params(axis='both', which='minor', labelsize=6)
            
            # Clean subplot titles
            ax.set_title(f'{particle_name} {coord}-coordinate', fontsize=10, pad=10)
            
            # Fix scientific notation alignment
            ax.yaxis.get_offset_text().set_fontsize(8)
            ax.xaxis.get_offset_text().set_fontsize(8)
    
    # Adjust spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.4, wspace=0.3)
    
    print("Plot ready!")
    return fig

def interpret_result(reorder_result):
    """Convert numerical result to descriptive string"""
    if reorder_result == 0:
        return "No event"
    elif reorder_result == 1:
        return "Reorder"
    elif reorder_result == 2:
        return "Ejection"
    else:
        return f"Unknown ({reorder_result})"

if __name__ == "__main__":
    file1, file2, show_collision = get_user_selection()
    if file1:
        fig = plot_trajectories(file1, file2, show_collision)
        plt.show()