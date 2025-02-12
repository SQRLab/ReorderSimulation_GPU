import numpy as np
from numba import cuda
import time
from cuda_3d_final import *
import random
import json
import os
import subprocess
import threading
import queue
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
Nt = 700000000000

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

#######################################################
###### Functions for single set of random params ######
#######################################################

def save_simulation_parameters(shots, vf, save_file="simulation_parameters.npz"):
    """
    Generate and save simulation parameters
    """
    # Calculate velocity distribution
    boltzDist = Boltz(collisionalMass, T, vMin, vMax, numBins)
    v = np.linspace(vMin, vMax, numBins)
    angles = np.linspace(-np.pi/2, np.pi/2, 100)
    offsets = np.linspace(-2e-9, 2e-9, 200)
    max_hypotenuse = 1.5e-7

    # Initialize arrays
    xc_all = np.zeros(shots, dtype=np.float64)
    yc_all = np.zeros(shots, dtype=np.float64)
    zc_all = np.zeros(shots, dtype=np.float64)
    vxc_all = np.zeros(shots, dtype=np.float64)
    vyc_all = np.zeros(shots, dtype=np.float64)
    vzc_all = np.zeros(shots, dtype=np.float64)
    vf_all = np.zeros((shots, Ni, 9), dtype=np.float64)
    Nt_all = np.zeros(shots, dtype=np.int32)

    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Generate parameters
    for i in range(shots):
        velocity = random.choices(v, weights=boltzDist)[0]
        angle_choiceXY = random.choice(angles)
        angle_choiceZ = random.choice(angles)
        offset_choice = random.choice(offsets)
        ion_collided = random.randint(0, Ni-1)
        
        x = -np.cos(angle_choiceZ)*max_hypotenuse 
        y = np.sin(angle_choiceXY)*x
        x = np.cos(angle_choiceXY)*x
        vx = np.abs(velocity*np.cos(angle_choiceZ))
        vy = np.sin(angle_choiceXY)*vx
        vx = np.cos(angle_choiceXY)*vx
        z = vf[ion_collided,2] + np.sin(angle_choiceZ)*max_hypotenuse + offset_choice
        vz = -1*velocity*np.sin(angle_choiceZ)

        vf_all[i, :, :] = vf
        xc_all[i] = x
        yc_all[i] = y
        zc_all[i] = z
        vxc_all[i] = vx
        vyc_all[i] = vy
        vzc_all[i] = vz
        Nt_all[i] = Nt

    # Save parameters to file
    np.savez(save_file,
             xc_all=xc_all, yc_all=yc_all, zc_all=zc_all,
             vxc_all=vxc_all, vyc_all=vyc_all, vzc_all=vzc_all,
             vf_all=vf_all, Nt_all=Nt_all)

def load_simulation_parameters(param_file="simulation_parameters.npz"):
    """
    Load saved simulation parameters
    """
    data = np.load(param_file)
    return (data['xc_all'], data['yc_all'], data['zc_all'],
            data['vxc_all'], data['vyc_all'], data['vzc_all'],
            data['vf_all'], data['Nt_all'])

def run_simulation(Nr, Nz, shots, total_physical_time, param_file="simulation_parameters.npz"):
    """
    Run simulation with fixed parameters
    """
    Nrmid = (Nr-1)/2
    Nzmid = (Nz-1)/2
    
    Dr = 3.00015e-5
    Dz = 9.00045e-5
    dr = Dr/float(Nr)
    dz = Dz/float(Nz)

    BAR_FORMAT = '{desc:<20}: {percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    
    with tqdm(total=100, desc="Setup Progress", 
              position=0, leave=True,
              bar_format=BAR_FORMAT) as setup_pbar:

        # Generate fields
        RFx, RFy = makeRF0(m, q, wr, Nr, Nz, Nrmid, Nzmid, dr)
        setup_pbar.update(25)
        
        DC = makeDC(m, q, wz, Nz, Nr, Nzmid, dz)
        setup_pbar.update(25)

        # Load or generate parameters
        if os.path.exists(param_file):
            xc_all, yc_all, zc_all, vxc_all, vyc_all, vzc_all, vf_all, Nt_all = load_simulation_parameters(param_file)
            print("\nLoaded existing simulation parameters")
        else:
            save_simulation_parameters(shots, vf, param_file)
            xc_all, yc_all, zc_all, vxc_all, vyc_all, vzc_all, vf_all, Nt_all = load_simulation_parameters(param_file)
            print("\nGenerated and saved new simulation parameters")
        setup_pbar.update(25)

        # Transfer data to GPU
        vf_device = cuda.to_device(vf_all)
        xc_device = cuda.to_device(xc_all)
        yc_device = cuda.to_device(yc_all)
        zc_device = cuda.to_device(zc_all)
        vxc_device = cuda.to_device(vxc_all)
        vyc_device = cuda.to_device(vyc_all)
        vzc_device = cuda.to_device(vzc_all)
        Nt_device = cuda.to_device(Nt_all)
        reorder_device = cuda.device_array(shots, dtype=np.int32)

        RFx_device = cuda.to_device(RFx)
        RFy_device = cuda.to_device(RFy)
        DC_device = cuda.to_device(DC)
        setup_pbar.update(25)
        setup_pbar.close()

    # Run simulation
    threads_per_block = 256
    blocks = (shots + threads_per_block - 1) // threads_per_block

    progress = cuda.mapped_array(1, dtype=np.int64)
    progress[0] = 0
    
    # Simulation progress bar with matching format
    with tqdm(total=shots, desc="Simulation Progress", 
              position=0, leave=True,
              bar_format=BAR_FORMAT) as sim_pbar:
              
        # Launch kernel
        mcCollision_kernel[blocks, threads_per_block](
            vf_device, xc_device, yc_device, zc_device, 
            vxc_device, vyc_device, vzc_device,
            q, mH2, aH2, Nt_device, dtSmall, dtCollision, 
            RFx_device, RFy_device, DC_device, 
            Nr, Nz, dr, dz, dtLarge, reorder_device, True, progress, total_physical_time
        )
        
        # Monitor progress
        last_count = 0
        while last_count < shots:
            current_count = progress[0]
            if current_count > last_count:
                sim_pbar.update(current_count - last_count)
                last_count = current_count
            time.sleep(0.01)

    # Get results
    reorder = reorder_device.copy_to_host()
    reorders = np.sum(reorder == 1)
    ejections = np.sum(reorder == 2)

    return reorders, ejections


##################################################
###### Function for random params every run ######
##################################################

# def run_simulation(Nr, Nz, shots, total_physical_time):
#     """
#     Run simulation with current timestep values and return statistics
#     """
#     Nrmid = (Nr-1)/2
#     Nzmid = (Nz-1)/2

#     Dr = 3.00015e-5
#     Dz = 9.00045e-5
#     dr = Dr/float(Nr)
#     dz = Dz/float(Nz)

#     BAR_FORMAT = '{desc:<20}: {percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'

#     # Setup progress bar

#     setup_pbar = tqdm(total=100, desc="Setup Progress", 
#                      position=0, leave=True,
#                      bar_format=BAR_FORMAT)

#     # Generate fields

#     RFx, RFy = makeRF0(m, q, wr, Nr, Nz, Nrmid, Nzmid, dr)
#     setup_pbar.update(20)

#     DC = makeDC(m, q, wz, Nz, Nr, Nzmid, dz)
#     setup_pbar.update(20)

#     # Initialize arrays

#     vf_all = np.zeros((shots, Ni, 9), dtype=np.float64)
#     xc_all = np.zeros(shots, dtype=np.float64)
#     yc_all = np.zeros(shots, dtype=np.float64)
#     zc_all = np.zeros(shots, dtype=np.float64)
#     vxc_all = np.zeros(shots, dtype=np.float64)
#     vyc_all = np.zeros(shots, dtype=np.float64)
#     vzc_all = np.zeros(shots, dtype=np.float64)
#     Nt_all = np.zeros(shots, dtype=np.int32)
#     setup_pbar.update(20)

#     # Calculate velocity distribution

#     boltzDist = Boltz(collisionalMass, T, vMin, vMax, numBins)
#     v = np.linspace(vMin, vMax, numBins)
#     angles = np.linspace(-np.pi/2, np.pi/2, 100)
#     offsets = np.linspace(-2e-9, 2e-9, 200)
#     max_hypotenuse = 1.5e-7
#     setup_pbar.update(20)

#     # Initialize collision parameters for each shot

#     for i in range(shots):

#         velocity = random.choices(v, weights=boltzDist)[0]
#         angle_choiceXY = random.choice(angles)
#         angle_choiceZ = random.choice(angles)
#         offset_choice = random.choice(offsets)
#         ion_collided = random.randint(0, Ni-1)

#         x = -np.cos(angle_choiceZ)*max_hypotenuse 
#         y = np.sin(angle_choiceXY)*x
#         x = np.cos(angle_choiceXY)*x
#         vx = np.abs(velocity*np.cos(angle_choiceZ))
#         vy = np.sin(angle_choiceXY)*vx
#         vx = np.cos(angle_choiceXY)*vx
#         z = vf[ion_collided,2] + np.sin(angle_choiceZ)*max_hypotenuse + offset_choice
#         vz = -1*velocity*np.sin(angle_choiceZ)

#         vf_all[i, :, :] = vf
#         xc_all[i] = x
#         yc_all[i] = y
#         zc_all[i] = z
#         vxc_all[i] = vx
#         vyc_all[i] = vy
#         vzc_all[i] = vz
#         Nt_all[i] = Nt

#     setup_pbar.update(10)

#     # Transfer data to GPU
#     vf_device = cuda.to_device(vf_all)
#     xc_device = cuda.to_device(xc_all)
#     yc_device = cuda.to_device(yc_all)
#     zc_device = cuda.to_device(zc_all)
#     vxc_device = cuda.to_device(vxc_all)
#     vyc_device = cuda.to_device(vyc_all)
#     vzc_device = cuda.to_device(vzc_all)
#     Nt_device = cuda.to_device(Nt_all)
#     reorder_device = cuda.device_array(shots, dtype=np.int32)

#     RFx_device = cuda.to_device(RFx)
#     RFy_device = cuda.to_device(RFy)
#     DC_device = cuda.to_device(DC)

#     setup_pbar.update(10)
#     setup_pbar.close()

#     # Run simulation
#     threads_per_block = 256
#     blocks = (shots + threads_per_block - 1) // threads_per_block

#     progress = cuda.mapped_array(1, dtype=np.int64)
#     progress[0] = 0
    
#     # Simulation progress bar with matching format
#     with tqdm(total=shots, desc="Simulation Progress", 
#               position=0, leave=True,
#               bar_format=BAR_FORMAT) as sim_pbar:
              
#         # Launch kernel
#         mcCollision_kernel[blocks, threads_per_block](
#             vf_device, xc_device, yc_device, zc_device, 
#             vxc_device, vyc_device, vzc_device,
#             q, mH2, aH2, Nt_device, dtSmall, dtCollision, 
#             RFx_device, RFy_device, DC_device, 
#             Nr, Nz, dr, dz, dtLarge, reorder_device, True, progress, total_physical_time
#         )
        
#         # Monitor progress
#         last_count = 0
#         while last_count < shots:
#             current_count = progress[0]
#             if current_count > last_count:
#                 sim_pbar.update(current_count - last_count)
#                 last_count = current_count
#             time.sleep(0.01)

#     # Get results
#     reorder = reorder_device.copy_to_host()
#     reorders = np.sum(reorder == 1)
#     ejections = np.sum(reorder == 2)

#     return reorders, ejections

def main():
    # Create results directory if it doesn't exist
    os.makedirs("timestep_results_final", exist_ok=True)
    
    # Create results file with current timestep values in filename
    results_file = f"timestep_results_final/results_dtCollision_{dtCollision}.json"
    
    results = {
        "timestep_values": {
            "dtSmall": dtSmall,
            "dtCollision": dtCollision,
           #"dtLarge": dtLarge,
            "Nt": Nt
        },
        "grid_results": {}
    }

    print(f"\nRunning simulations with timesteps:")
    print(f"dtSmall: {dtSmall}")
    print(f"dtCollision: {dtCollision}")
    print(f"dtLarge: {dtLarge}")
    print(f"Nt: {Nt}\n")

    # Run simulations for each grid size
    for Nr in Nr_values:
        for Nz in Nz_values:
            print(f"Processing grid size Nr={Nr} x Nz={Nz}")
            
            start_time = time.perf_counter()
            reorders, ejections = run_simulation(Nr, Nz, shot_sizes[0], total_physical_time)
            end_time = time.perf_counter()
            
            # Store results
            grid_key = f"Nr{Nr}_Nz{Nz}"
            results["grid_results"][grid_key] = {
                "reorders": int(reorders),
                "ejections": int(ejections),
                "total_events": int(reorders + ejections),
                "computation_time": end_time - start_time
            }
            
                # Print results without progress bar formatting
            print(f"\nResults for Nr{Nr}_Nz{Nz}:")
            print(f"  Reorders: {reorders}")
            print(f"  Ejections: {ejections}")
            print(f"  Total events: {reorders + ejections}")
            print(f"  Computation time: {end_time - start_time:.2f} seconds\n")

    # Save results to JSON file
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    main()