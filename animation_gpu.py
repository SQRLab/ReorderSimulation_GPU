import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import h5py

def main():
    # Specify the HDF5 file to read
    trajectory_filename = 'simulation_results_test/trajectory_grid20001_ions4_shot242.h5'  # Adjust as needed

    # Open the HDF5 file
    with h5py.File(trajectory_filename, 'r') as f:
        # Read the timesteps
        timesteps = f['timesteps'][:]
        total_timesteps = len(timesteps)
        
        # Read the ion data
        ions_data = []
        ion_names = [name for name in f.keys() if name.startswith('ion_')]
        ion_names.sort()  # Ensure the ions are in order

        for ion_name in ion_names:
            ion_group = f[ion_name]
            r_pos = ion_group['r_position'][:]
            z_pos = ion_group['z_position'][:]
            ion_data = {
                'r_position': r_pos,
                'z_position': z_pos,
            }
            ions_data.append(ion_data)
        
        # Read the collisional particle data
        coll_group = f['collisional_particle']
        coll_r_pos = coll_group['r_position'][:]
        coll_z_pos = coll_group['z_position'][:]
        coll_data = {
            'r_position': coll_r_pos,
            'z_position': coll_z_pos,
        }

    # Print position ranges for verification
    print("Ion positions over time:")
    for i, ion_data in enumerate(ions_data):
        r_positions = ion_data['r_position']
        z_positions = ion_data['z_position']
        print(f"Ion {i}:")
        print(f"  Radial position range: {r_positions.min()} to {r_positions.max()}")
        print(f"  Axial position range: {z_positions.min()} to {z_positions.max()}")

    coll_r_positions = coll_data['r_position']
    coll_z_positions = coll_data['z_position']
    print("Collisional Particle:")
    print(f"  Radial position range: {coll_r_positions.min()} to {coll_r_positions.max()}")
    print(f"  Axial position range: {coll_z_positions.min()} to {coll_z_positions.max()}")

    # Verify the first few frames
    num_initial_frames = 5
    for frame in range(min(num_initial_frames, total_timesteps)):
        r = coll_data['r_position'][frame]
        z = coll_data['z_position'][frame]
        print(f"Frame {frame}: Collisional Particle Position: r={r}, z={z}")

    # Now we have the data, let's create the animation

    # Set up the figure and axis
    fig, ax = plt.subplots()
    ax.set_xlabel('Radial Position (m)')
    ax.set_ylabel('Axial Position (m)')
    ax.set_title('Ion Collision Simulation')

    # Define the scale factor
    scale_factor = 5e6  # Adjust as needed

    # Set fixed axis limits based on scaled ion positions
    ax.set_xlim(-50, 50)  # Adjust based on expected movement
    ax.set_ylim(-50, 50)  # Adjust based on expected movement

    # Initialize scatter plots for ions and collisional particle
    ion_scatters = []
    for i in range(len(ions_data)):
        scat = ax.scatter([], [], label=f'Ion {i}')
        ion_scatters.append(scat)
    coll_scat = ax.scatter([], [], c='red', label='Collisional Particle', marker='X', s=20)  # Use a different marker
    ax.legend()

    # Initialize a text annotation for ejection
    ejection_text = ax.text(0.5, 0.95, '', transform=ax.transAxes, ha='center', va='top',
                            fontsize=12, color='red')

    def init():
        # Initialize scatter plots with empty data
        empty_array = np.empty((0, 2))
        for scat in ion_scatters:
            scat.set_offsets(empty_array)
        coll_scat.set_offsets(empty_array)
        coll_scat.set_visible(True)  # Ensure it's visible initially
        ejection_text.set_text('')
        return ion_scatters + [coll_scat, ejection_text]

    def update(frame):
        # For each ion, set the positions
        for i, ion_data in enumerate(ions_data):
            r_pos = ion_data['r_position'][frame] * scale_factor
            z_pos = ion_data['z_position'][frame] * scale_factor
            ion_scatters[i].set_offsets([[r_pos, z_pos]])
        
        # Set the collisional particle position
        coll_r_pos = coll_data['r_position'][frame]
        coll_z_pos = coll_data['z_position'][frame]
        
        # Calculate the distance from origin
        distance = np.sqrt((coll_r_pos)**2 + (coll_z_pos)**2)
        ejection_threshold = 1.0  # 1 meter threshold for ejection

        if distance < ejection_threshold:
            # Scale the positions
            scaled_r = coll_r_pos * scale_factor
            scaled_z = coll_z_pos * scale_factor
            # Only set offsets if within axis limits
            if -50 <= scaled_r <= 50 and -50 <= scaled_z <= 50:
                coll_scat.set_offsets([[scaled_r, scaled_z]])
                coll_scat.set_visible(True)
                ejection_text.set_text('')
            else:
                # Hide the collisional particle if it's out of view
                coll_scat.set_offsets(np.empty((0, 2)))
                coll_scat.set_visible(False)
                ejection_text.set_text('Collisional Particle not in frame')
        else:
            # Hide the collisional particle if ejected
            coll_scat.set_offsets(np.empty((0, 2)))
            coll_scat.set_visible(False)
            ejection_text.set_text('Collisional Particle Ejected')
        
        return ion_scatters + [coll_scat, ejection_text]

    # Limit the number of frames for the animation if needed
    total_frames = total_timesteps
    #print(total_frames)
    max_frames = 10000  # Adjust as needed
    step = max(1, total_frames // max_frames)
    frames = range(0, total_frames, step)

    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=20)
    plt.show()

if __name__ == '__main__':
    main()
