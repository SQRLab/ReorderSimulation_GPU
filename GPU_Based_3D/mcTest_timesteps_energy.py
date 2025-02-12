import numpy as np
from numba import cuda
import time
from cuda_3d_final_energy import *
import h5py
from tqdm import tqdm
import os

# Physical constants
amu = 1.67e-27  # Atomic mass unit in kg
eps0 = 8.854e-12
qe = 1.6e-19  # Elementary charge in Coulombs

# Physical parameters
m = 40.0 * amu
q = 1.0 * qe
wr = 2 * np.pi * 5e6  # Radial angular frequency (SI units)
wz = 2 * np.pi * 0.5e6  # Axial angular frequency (SI units)
omega_rf = 2 * np.pi * 5e6  # RF drive frequency (SI units)
aH2 = 8e-31  # Polarizability of H2 in SI units
mH2 = 2.0 * amu  # Mass of H2 in kg

# Simulation parameters
dtSmall = 1e-12
dtCollision = 1e-16 
dtLarge = 1e-10
Nt = 700000000000000
total_physical_time = 1e-6

# Grid parameters
Nr = Nz = 1001
Nrmid = (Nr-1)/2
Nzmid = (Nz-1)/2
Dr = 3.00015e-5
Dz = 9.00045e-5
dr = Dr/float(Nr)
dz = Dz/float(Nz)

def get_termination_reason_str(reason_code):
    """Convert termination reason code to human-readable string"""
    reasons = {
        0: "Normal completion (reached total_physical_time)",
        1: "Storage limit reached",
        2: "Ion ejection",
        3: "Ion reordering",
        4: "Reached max timesteps",
        5: "Numerical instability detected"
    }
    return reasons.get(reason_code, "Unknown termination reason")

def load_or_generate_fields(m, q, wr, wz, Nr, Nz, Nrmid, Nzmid, dr, dz):
    """Load RF and DC fields from files if they exist, otherwise generate and save them"""
    rf_file = f"fields_{Nr}x{Nz}_rf.npz"
    dc_file = f"fields_{Nr}x{Nz}_dc.npy"
    
    if os.path.exists(rf_file) and os.path.exists(dc_file):
        print(f"Loading existing fields...")
        rf_data = np.load(rf_file)
        RFx = rf_data['RFx']
        RFy = rf_data['RFy']
        DC = np.load(dc_file)
    else:
        print(f"Generating new fields...")
        RFx, RFy = makeRF0(m, q, wr, Nr, Nz, Nrmid, Nzmid, dr)
        DC = makeDC(m, q, wz, Nz, Nr, Nzmid, dz)
        
        print(f"Saving fields...")
        np.savez(rf_file, RFx=RFx, RFy=RFy, omega_rf=omega_rf)
        np.save(dc_file, DC)
    
    return RFx, RFy, DC

def calculate_storage_params(total_physical_time, dt_small, dt_collision, max_memory_gb=12):
    """Calculate storage parameters based on available memory"""
    max_steps = int(total_physical_time / dt_collision)
    
    # Calculate memory per step for:
    # - positions (3 particles × 3 coordinates × float64)
    # - energies (7 values × float64) - now including relative error
    # - timestamp (float64)
    # - system state flags and additional metadata
    bytes_per_step = (3 * 3 * 8) + (7 * 8) + 8 + 32
    available_bytes = max_memory_gb * 1024**3 * 0.8  # Use 80% of available memory
    
    max_storable_steps = int(available_bytes / bytes_per_step)
    base_interval = max(1, int(max_steps / max_storable_steps))
    
    return {
        'total_steps': min(max_steps, max_storable_steps),
        'base_interval': base_interval,
        'collision_interval': max(1, base_interval // 10)
    }

def run_energy_simulation(selected_indices):
    """Run simulation for selected shots with energy tracking"""
    print("\nLoading parameters...")
    sampled_indices = np.load('sampled_indices.npy')
    param_data = np.load('simulation_parameters.npz')
    original_indices = sampled_indices[selected_indices]
    
    print("\nPreparing fields...")
    RFx, RFy, DC = load_or_generate_fields(m, q, wr, wz, Nr, Nz, Nrmid, Nzmid, dr, dz)
    
    # Calculate storage parameters
    storage_params = calculate_storage_params(total_physical_time, dtSmall, dtCollision)
    print("\nStorage parameters:")
    print(f"Total steps: {storage_params['total_steps']}")
    print(f"Base interval: {storage_params['base_interval']}")
    print(f"Collision interval: {storage_params['collision_interval']}")
    
    # Arrays for results
    all_reorders = np.zeros(len(selected_indices), dtype=np.int32)
    all_termination_reasons = np.zeros(len(selected_indices), dtype=np.int32)
    
    for idx, (selected_idx, original_idx) in enumerate(zip(selected_indices, original_indices)):
        print(f"\nProcessing shot {selected_idx} (original index: {original_idx})")
        
        try:
            # Reset CUDA context
            cuda.close()
            cuda.select_device(0)
            
            # Transfer fields to GPU
            RFx_device = cuda.to_device(RFx)
            RFy_device = cuda.to_device(RFy)
            DC_device = cuda.to_device(DC)
            
            # Extract parameters for this shot
            vf_all = param_data['vf_all'][original_idx:original_idx+1]
            xc_all = param_data['xc_all'][original_idx:original_idx+1]
            yc_all = param_data['yc_all'][original_idx:original_idx+1]
            zc_all = param_data['zc_all'][original_idx:original_idx+1]
            vxc_all = param_data['vxc_all'][original_idx:original_idx+1]
            vyc_all = param_data['vyc_all'][original_idx:original_idx+1]
            vzc_all = param_data['vzc_all'][original_idx:original_idx+1]
            
            # Transfer shot data to GPU
            vf_device = cuda.to_device(vf_all)
            xc_device = cuda.to_device(xc_all)
            yc_device = cuda.to_device(yc_all)
            zc_device = cuda.to_device(zc_all)
            vxc_device = cuda.to_device(vxc_all)
            vyc_device = cuda.to_device(vyc_all)
            vzc_device = cuda.to_device(vzc_all)
            
            # Create storage arrays
            positions_over_time = cuda.device_array(
                (1, storage_params['total_steps'], 3, 3),
                dtype=np.float64
            )
            energies_over_time = cuda.device_array(
                (1, storage_params['total_steps'], 7),  # Added slot for relative error
                dtype=np.float64
            )
            timesteps = cuda.device_array(
                (1, storage_params['total_steps']),
                dtype=np.float64
            )
            
            # Storage parameters array for GPU
            storage_params_array = cuda.to_device(np.array([
                storage_params['base_interval'],
                storage_params['collision_interval'],
                storage_params['total_steps']
            ], dtype=np.int32))
            
            # Other necessary arrays
            Nt_device = cuda.to_device(np.array([Nt], dtype=np.int64))
            reorder_device = cuda.device_array(1, dtype=np.int32)
            termination_reason_device = cuda.device_array(1, dtype=np.int32)
            progress = cuda.mapped_array(1, dtype=np.int64)
            progress[0] = 0
            
            print("Running simulation...")
            stream = cuda.stream()
            with stream.auto_synchronize():
                mcCollision_kernel[1, 1](
                    vf_device, xc_device, yc_device, zc_device,
                    vxc_device, vyc_device, vzc_device,
                    q, mH2, aH2, Nt_device, dtSmall, dtCollision,
                    RFx_device, RFy_device, DC_device,
                    Nr, Nz, dr, dz, dtLarge, reorder_device,
                    True, progress, total_physical_time,
                    positions_over_time, timesteps, energies_over_time,
                    storage_params_array, termination_reason_device
                )
            
            # Monitor progress
            with tqdm(total=1, desc="Simulation progress") as pbar:
                last_progress = 0
                while progress[0] < 1:
                    if progress[0] > last_progress:
                        pbar.update(progress[0] - last_progress)
                        last_progress = progress[0]
                    time.sleep(0.01)
            
            # Get results
            cuda.synchronize()
            reorder = reorder_device.copy_to_host()[0]
            termination_reason = termination_reason_device.copy_to_host()[0]
            positions = positions_over_time.copy_to_host()
            energies = energies_over_time.copy_to_host()
            times = timesteps.copy_to_host()
            
            # Store results
            all_reorders[idx] = reorder
            all_termination_reasons[idx] = termination_reason
            
            # Save trajectory and energy data
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            h5_filename = f'trajectory_shot{selected_idx}_dt{dtSmall}_{timestamp}.h5'
            
            num_steps = int(times[0, -1])
            physical_time = times[0, num_steps-1] if num_steps > 0 else 0
            
            print(f"\nSaving data to {h5_filename}")
            with h5py.File(h5_filename, 'w') as f:
                f.attrs.update({
                    'shot_index': selected_idx,
                    'original_index': original_idx,
                    'dtSmall': dtSmall,
                    'dtCollision': dtCollision,
                    'dtLarge': dtLarge,
                    'target_physical_time': total_physical_time,
                    'actual_physical_time': physical_time,
                    'reorder_result': reorder,
                    'termination_reason': termination_reason,
                    'termination_reason_str': get_termination_reason_str(termination_reason),
                    'rf_frequency': omega_rf,
                    'rf_frequency_MHz': omega_rf/(2*np.pi*1e6),
                    'mass_amu': m/amu,
                    'charge_e': q/qe
                })
                
                # Save positions
                traj_group = f.create_group('trajectory')
                traj_group.create_dataset('timesteps', data=times[0,:num_steps])
                traj_group.create_dataset('ion1_positions', data=positions[0,:num_steps,0,:])
                traj_group.create_dataset('ion2_positions', data=positions[0,:num_steps,1,:])
                traj_group.create_dataset('collision_positions', data=positions[0,:num_steps,2,:])
                
                # Save energies with improved organization
                energy_group = f.create_group('energies')
                
                # Ion energies
                ion_group = energy_group.create_group('ions')
                ion_group.create_dataset('kinetic', data=energies[0,:num_steps,0])
                ion_group.create_dataset('potential', data=energies[0,:num_steps,1])
                ion_group.create_dataset('total', data=energies[0,:num_steps,2])
                
                # Collision particle energies
                collision_group = energy_group.create_group('collision')
                collision_group.create_dataset('kinetic', data=energies[0,:num_steps,3])
                collision_group.create_dataset('potential', data=energies[0,:num_steps,4])
                collision_group.create_dataset('total', data=energies[0,:num_steps,5])
                
                # Energy conservation metrics
                metrics_group = energy_group.create_group('metrics')
                metrics_group.create_dataset('relative_error', data=energies[0,:num_steps,6])
                
                # Calculate and store energy conservation statistics
                initial_total = energies[0,0,2] + energies[0,0,5]  # Initial total energy
                final_total = energies[0,num_steps-1,2] + energies[0,num_steps-1,5]  # Final total energy
                max_error = np.max(energies[0,:num_steps,6]) if num_steps > 0 else 0
                
                metrics_group.attrs.update({
                    'initial_total_energy': initial_total,
                    'final_total_energy': final_total,
                    'absolute_energy_change': final_total - initial_total,
                    'relative_energy_change': (final_total - initial_total) / initial_total if initial_total != 0 else 0,
                    'maximum_relative_error': max_error
                })
            
            print(f"Termination reason: {get_termination_reason_str(termination_reason)}")
            
        except cuda.cudadrv.driver.CudaAPIError as e:
            print(f"\nCUDA Error: {str(e)}")
            print("Attempting cleanup and continuing...")
        
        finally:
            # Cleanup GPU resources
            try:
                del RFx_device, RFy_device, DC_device
                del vf_device, xc_device, yc_device, zc_device
                del vxc_device, vyc_device, vzc_device
                del positions_over_time, timesteps, energies_over_time
                cuda.synchronize()
                cuda.close()
            except:
                pass
    
    return all_reorders, all_termination_reasons

def main():
    """Main function for energy tracking simulation"""
    while True:
        try:
            input_str = input("Enter shot indices to simulate (comma-separated, e.g., '0,5,10'): ")
            selected_indices = [int(x.strip()) for x in input_str.split(',')]
            if all(0 <= idx < 100 for idx in selected_indices):
                break
            else:
                print("Error: Indices must be between 0 and 99")
        except ValueError:
            print("Error: Please enter valid comma-separated numbers")

    print(f"\nSimulating trajectories for shots: {selected_indices}")
    
    start_time = time.perf_counter()
    reorders, termination_reasons = run_energy_simulation(selected_indices)
    end_time = time.perf_counter()
    
    # Print summary statistics
    print("\nSimulation Results:")
    print("-" * 50)
    
    for idx, (reorder, reason) in enumerate(zip(reorders, termination_reasons)):
        shot_idx = selected_indices[idx]
        print(f"\nShot {shot_idx}:")
        print(f"Termination reason: {get_termination_reason_str(reason)}")
        
        # If there's a reorder/ejection event, print its type
        if reorder > 0:
            print(f"Reorder type: {'Ion reordering' if reorder == 1 else 'Ion ejection'}")
        
        # Open the corresponding HDF5 file to get energy statistics
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        h5_filename = f'trajectory_shot{shot_idx}_dt{dtSmall}_{timestamp}.h5'
        try:
            with h5py.File(h5_filename, 'r') as f:
                metrics = f['energies/metrics']
                print("\nEnergy Conservation:")
                print(f"Initial total energy: {metrics.attrs['initial_total_energy']:.2e} J")
                print(f"Final total energy: {metrics.attrs['final_total_energy']:.2e} J")
                print(f"Relative energy change: {metrics.attrs['relative_energy_change']:.2e}")
                print(f"Maximum relative error: {metrics.attrs['maximum_relative_error']:.2e}")
        except (OSError, KeyError) as e:
            print(f"\nCould not read energy metrics from {h5_filename}")
    
    print("\nOverall Statistics:")
    print(f"Total reorders: {np.sum(reorders == 1)}")
    print(f"Total ejections: {np.sum(reorders == 2)}")
    print(f"Total events: {np.sum(reorders > 0)}")
    print(f"Computation time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()