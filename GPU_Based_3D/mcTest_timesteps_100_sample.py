import numpy as np
from numba import cuda
import time
from cuda_3d_final import *
import os
from tqdm import tqdm

amu = 1.67e-27  # Atomic mass unit in kg
eps0 = 8.854e-12
qe = 1.6e-19  # Elementary charge in Coulombs

# Physical parameters
m = 40.0 * amu
q = 1.0 * qe
wr = 2 * np.pi * 5e6  # Radial angular frequency (SI units)
wz = 2 * np.pi * 0.5e6  # Axial angular frequency (SI units)
aH2 = 8e-31  # Polarizability of H2 in SI units
mH2 = 2.0 * amu  # Mass of H2 in kg

# Timestep parameters

#########################
###### Changing Nt ######
#########################

# Set A1: done
# dtSmall = 1e-12
# dtCollision = 1e-16 
# dtLarge = 1e-10
# Nt = 70000

# Set A2: done
# dtSmall = 1e-12
# dtCollision = 1e-16 
# dtLarge = 1e-10
# Nt = 70000000

# Set A3: done
# dtSmall = 1e-12
# dtCollision = 1e-16 
# dtLarge = 1e-10
# Nt = 7000000

#####################
###### dtSmall ######
#####################

# Set B1:
# dtSmall = 1e-10
# dtCollision = 1e-16 
# dtLarge = 1e-10
# Nt = 700000000

# Set B2:
# dtSmall = 1e-11
# dtCollision = 1e-16 
# dtLarge = 1e-10
# Nt = 700000000000

# Set B3:
# dtSmall = 1e-12
# dtCollision = 1e-16 
# dtLarge = 1e-10
# Nt = 700000000

# Set B4:
# dtSmall = 1e-13
# dtCollision = 1e-16 
# dtLarge = 1e-10
# Nt = 700000000

# Set B5:
# dtSmall = 1e-14
# dtCollision = 1e-16 
# dtLarge = 1e-10
# Nt = 700000000000

# Set B6:
dtSmall = 1e-15
dtCollision = 1e-16 
dtLarge = 1e-10
Nt = 7000000000000

#########################
###### dtCollision ######
#########################

# Set C1:
# dtSmall = 1e-12
# dtCollision = 1e-13 
# dtLarge = 1e-10
# Nt = 7000000000000

# Set C2:
# dtSmall = 1e-12
# dtCollision = 1e-14
# dtLarge = 1e-10
# Nt = 7000000000000

# Set C3:
# dtSmall = 1e-12
# dtCollision = 1e-15
# dtLarge = 1e-10
# Nt = 7000000000000

# Set C4:
# dtSmall = 1e-12
# dtCollision = 1e-17
# dtLarge = 1e-10
# Nt = 7000000000000

# Set C5:
# dtSmall = 1e-12
# dtCollision = 1e-18
# dtLarge = 1e-10
# Nt = 7000000000000

# Set C6:
# dtSmall = 1e-12
# dtCollision = 1e-19
# dtLarge = 1e-10
# Nt = 7000000000000

#####################
###### dtLarge ######
#####################

# Set D1:
# dtSmall = 1e-12
# dtCollision = 1e-16
# dtLarge = 1e-7
# Nt = 7000000

# Set D2:
# dtSmall = 1e-12
# dtCollision = 1e-16
# dtLarge = 1e-8
# Nt = 7000000

# Set D3:
# dtSmall = 1e-12
# dtCollision = 1e-16
# dtLarge = 1e-9
# Nt = 7000000

# Set D4:
# dtSmall = 1e-12
# dtCollision = 1e-16
# dtLarge = 1e-11
# Nt = 7000000

# Set D5:
# dtSmall = 1e-12
# dtCollision = 1e-16
# dtLarge = 1e-12
# Nt = 7000000

# Set D6:
# dtSmall = 1e-12
# dtCollision = 1e-16
# dtLarge = 1e-13
# Nt = 7000000

################################
###### dtCollision and Nt ######
################################


# Set E1:
# dtSmall = 1e-12
# dtCollision = 1e-13 
# dtLarge = 1e-10
# Nt = 700

# Set E2:
# dtSmall = 1e-12
# dtCollision = 1e-14
# dtLarge = 1e-10
# Nt = 7000

# Set E3:
# dtSmall = 1e-12
# dtCollision = 1e-15
# dtLarge = 1e-10
# Nt = 70000

# Set E4:
# dtSmall = 1e-12
# dtCollision = 1e-16 
# dtLarge = 1e-10
# Nt = 700000000

# Set E5:
# dtSmall = 1e-12
# dtCollision = 1e-17
# dtLarge = 1e-10
# Nt = 7000000

# Set E6:
# dtSmall = 1e-12
# dtCollision = 1e-18
# dtLarge = 1e-10
# Nt = 70000000

# Set E7:
# dtSmall = 1e-12
# dtCollision = 1e-19
# dtLarge = 1e-10
# Nt = 700000000

Ni = current_ions = 2
Nc = 1

# Grid parameters
Nz_values = [1001]  # AXIAL
Nr_values = [1001]  # RADIAL
ion_counts = [2]
shot_sizes = [100000]

total_physical_time = 1e-6

vf = makeVf(Ni, 1.0*q, m, wz)

#Nt = 700000
T = 300
collisionalMass = 2
vMin = 50
vMax = 5000
numBins = 1000

def sample_from_existing_parameters(param_file="simulation_parameters.npz", sample_size=100, indices_file="sampled_indices.npy"):
    """
    Load existing parameters and select random sample or use existing indices
    """
    # Load all parameters
    print("\nLoading existing parameters...")
    data = np.load(param_file)
    
    # Get total number of shots
    total_shots = len(data['xc_all'])
    
    # Check if indices file exists
    if os.path.exists(indices_file):
        print("Loading existing indices...")
        indices = np.load(indices_file)
    else:
        print("Generating new random indices...")
        # Set seed for reproducibility
        np.random.seed(42)
        indices = np.random.choice(total_shots, sample_size, replace=False)
        # Save indices for future use
        np.save(indices_file, indices)
        print(f"Saved indices to {indices_file}")
    
    # Sample the parameters
    xc_all = data['xc_all'][indices]
    yc_all = data['yc_all'][indices]
    zc_all = data['zc_all'][indices]
    vxc_all = data['vxc_all'][indices]
    vyc_all = data['vyc_all'][indices]
    vzc_all = data['vzc_all'][indices]
    vf_all = data['vf_all'][indices]
    Nt_all = data['Nt_all'][indices]
    
    return (xc_all, yc_all, zc_all, vxc_all, vyc_all, vzc_all, vf_all, Nt_all, indices)

def run_sampled_simulation(Nr, Nz, sample_size=100):
    """
    Run simulation with sampled parameters and save results with fundamental parameters
    """
    Nrmid = (Nr-1)/2
    Nzmid = (Nz-1)/2
    
    Dr = 3.00015e-5
    Dz = 9.00045e-5
    dr = Dr/float(Nr)
    dz = Dz/float(Nz)

    # Sample from existing parameters
    (xc_all, yc_all, zc_all, vxc_all, vyc_all, vzc_all, vf_all, Nt_all, 
     indices) = sample_from_existing_parameters("simulation_parameters.npz", sample_size)

    # Calculate original parameters from positions/velocities
    velocities = []
    angle_XYs = []
    angle_Zs = []
    offsets = []
    ions_collided = []

    for i in range(sample_size):
        # Calculate velocity
        velocity = np.sqrt(vxc_all[i]**2 + vyc_all[i]**2 + vzc_all[i]**2)
        
        # Calculate angles
        angle_XY = np.arctan2(vyc_all[i], vxc_all[i])
        angle_Z = np.arctan2(vzc_all[i], np.sqrt(vxc_all[i]**2 + vyc_all[i]**2))
        
        # Find closest ion (minimum z-distance)
        z_distances = [abs(vf_all[i,j,2] - zc_all[i]) for j in range(Ni)]
        ion_collided = np.argmin(z_distances)
        
        # Calculate offset from that ion's position
        offset = zc_all[i] - vf_all[i,ion_collided,2]

        velocities.append(velocity)
        angle_XYs.append(angle_XY)
        angle_Zs.append(angle_Z)
        offsets.append(offset)
        ions_collided.append(ion_collided)

    print("Generating fields...")
    # Generate fields
    RFx, RFy = makeRF0(m, q, wr, Nr, Nz, Nrmid, Nzmid, dr)
    DC = makeDC(m, q, wz, Nz, Nr, Nzmid, dz)

    print("Transferring data to GPU...")
    # Transfer data to GPU
    vf_device = cuda.to_device(vf_all)
    xc_device = cuda.to_device(xc_all)
    yc_device = cuda.to_device(yc_all)
    zc_device = cuda.to_device(zc_all)
    vxc_device = cuda.to_device(vxc_all)
    vyc_device = cuda.to_device(vyc_all)
    vzc_device = cuda.to_device(vzc_all)
    Nt_device = cuda.to_device(Nt_all)
    reorder_device = cuda.device_array(sample_size, dtype=np.int32)

    RFx_device = cuda.to_device(RFx)
    RFy_device = cuda.to_device(RFy)
    DC_device = cuda.to_device(DC)

    # Run simulation
    threads_per_block = 256
    blocks = (sample_size + threads_per_block - 1) // threads_per_block

    progress = cuda.mapped_array(1, dtype=np.int64)
    progress[0] = 0
    
    print("\nRunning simulation...")
    # Launch kernel with progress bar
    with tqdm(total=sample_size, desc="Simulation Progress") as pbar:
        mcCollision_kernel[blocks, threads_per_block](
            vf_device, xc_device, yc_device, zc_device, 
            vxc_device, vyc_device, vzc_device,
            q, mH2, aH2, Nt_device, dtSmall, dtCollision, 
            RFx_device, RFy_device, DC_device, 
            Nr, Nz, dr, dz, dtLarge, reorder_device, True, progress, total_physical_time
        )
        
        # Monitor progress
        last_count = 0
        while last_count < sample_size:
            current_count = progress[0]
            if current_count > last_count:
                pbar.update(current_count - last_count)
                last_count = current_count
            time.sleep(0.01)

    # Get results
    reorder = reorder_device.copy_to_host()

    print("\nSaving results...")
    # Create results directory using the timestep values in the name
    results_dir = f'sampled_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Save results with parameters
    results_file = f"{results_dir}/collision_results_dtSmall_{dtSmall}.txt"
    with open(results_file, 'w') as f:
        f.write("axial_trap_freq(MHz)\tvelocity(m/s)\tion_collided\tangleXY(rads)\tangleZ(rads)\tcollision_offset(m)\treorder\n")
        for i in range(sample_size):
            f.write(f"{wz/(2*np.pi*1e6):.6f}\t{velocities[i]:.6f}\t{ions_collided[i]}\t{angle_XYs[i]:.6f}\t{angle_Zs[i]:.6f}\t{offsets[i]:.6e}\t{reorder[i]}\n")

    print(f"Results saved to {results_file}")
    return reorder

def main_sampling():
    """
    Main function for sampling simulation
    """
    print("Starting sampled simulation...")
    reorder = run_sampled_simulation(1001, 1001, 100)  # Using default grid size
    
    # Print summary statistics
    reorders = np.sum(reorder == 1)
    ejections = np.sum(reorder == 2)
    print(f"\nSimulation complete!")
    print(f"Total reorders: {reorders}")
    print(f"Total ejections: {ejections}")
    print(f"Total events: {reorders + ejections}")

if __name__ == "__main__":
    main_sampling()