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

def force_gpu_cleanup():
    """
    Forces complete GPU cleanup, ensuring all resources are released
    """
    try:
        # Synchronize and clean current context
        cuda.synchronize()
        
        # Get current context and device
        ctx = cuda.current_context()
        
        # Force deallocation of all arrays in this context
        ctx.reset()
        
        # Close context
        cuda.close()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Small delay to ensure cleanup completes
        time.sleep(0.1)
        
        # Reinitialize CUDA
        cuda.select_device(0)
        
    except Exception as e:
        print(f"Warning during cleanup: {str(e)}")

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
        np.savez(rf_file, RFx=RFx, RFy=RFy)
        np.save(dc_file, DC)
    
    return RFx, RFy, DC

def calculate_adaptive_storage(total_physical_time, dt_small, dt_collision, max_memory_gb=12):
    """Calculate adaptive storage parameters based on available memory and timesteps"""
    max_steps = int(total_physical_time / dt_collision)
    
    bytes_per_step = (3 * 3 * 4) + 8  # 3 particles * 3 coords * float32 + float64 timestamp
    available_bytes = max_memory_gb * 1024**3 * 0.8
    max_storable_steps = int(available_bytes / bytes_per_step)
    
    if max_steps <= max_storable_steps:
        return {
            'base_storage_interval': 1,
            'collision_storage_interval': 1,
            'total_storage_steps': max_steps
        }
    else:
        estimated_collision_fraction = 0.1
        regular_steps = max_steps * (1 - estimated_collision_fraction)
        collision_steps = max_steps * estimated_collision_fraction
        
        base_interval = max(1, int(regular_steps / (max_storable_steps * 0.9)))
        collision_interval = max(1, int(base_interval / 10))
        
        total_storage_steps = (
            int(regular_steps / base_interval) +
            int(collision_steps / collision_interval)
        )
        
        return {
            'base_storage_interval': base_interval,
            'collision_storage_interval': collision_interval,
            'total_storage_steps': min(total_storage_steps, max_storable_steps)
        }

def run_trajectory_simulation(selected_indices):
    """Run simulation for selected shots with trajectory tracking"""
    print("\nLoading parameters...")
    sampled_indices = np.load('sampled_indices.npy')
    param_data = np.load('simulation_parameters.npz')
    original_indices = sampled_indices[selected_indices]
    
    print("\nPreparing fields...")
    RFx, RFy, DC = load_or_generate_fields(m, q, wr, wz, Nr, Nz, Nrmid, Nzmid, dr, dz)
    
    # Calculate storage parameters
    storage_params = calculate_adaptive_storage(total_physical_time, dtSmall, dtCollision)
    print("\nStorage parameters:")
    print(f"Base interval: {storage_params['base_storage_interval']}")
    print(f"Collision interval: {storage_params['collision_storage_interval']}")
    print(f"Total storage steps: {storage_params['total_storage_steps']}")
    
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
                (1, storage_params['total_storage_steps'], 3, 3),
                dtype=np.float32
            )
            timesteps = cuda.device_array(
                (1, storage_params['total_storage_steps']),
                dtype=np.float64
            )
            
            # Storage parameters array
            storage_params_array = cuda.to_device(np.array([
                storage_params['base_storage_interval'],
                storage_params['collision_storage_interval'],
                storage_params['total_storage_steps']
            ], dtype=np.int32))
            
            # Other necessary arrays
            Nt_device = cuda.to_device(np.array([Nt], dtype=np.int64))
            reorder_device = cuda.device_array(1, dtype=np.int32)
            termination_reason_device = cuda.device_array(1, dtype=np.int32)
            progress = cuda.mapped_array(1, dtype=np.int64)
            progress[0] = 0
            
            print("Running simulation...")
            # Launch kernel
            stream = cuda.stream()
            with stream.auto_synchronize():
                mcCollision_kernel[1, 1](
                    vf_device, xc_device, yc_device, zc_device,
                    vxc_device, vyc_device, vzc_device,
                    q, mH2, aH2, Nt_device, dtSmall, dtCollision,
                    RFx_device, RFy_device, DC_device,
                    Nr, Nz, dr, dz, dtLarge, reorder_device,
                    True, progress, total_physical_time,
                    positions_over_time, timesteps,
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
            times = timesteps.copy_to_host()
            
            # Store results
            all_reorders[idx] = reorder
            all_termination_reasons[idx] = termination_reason
            
            # Save trajectory data
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            h5_filename = f'trajectory_shot{selected_idx}_dt{dtSmall}_{timestamp}.h5'
            
            num_steps = int(times[0, -1])
            physical_time = times[0, num_steps-1] if num_steps > 0 else 0
            
            print(f"\nSaving trajectory data to {h5_filename}")
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
                    'termination_reason_str': get_termination_reason_str(termination_reason)
                })
                
                traj_group = f.create_group('trajectory')
                traj_group.create_dataset('timesteps', data=times[0,:num_steps])
                traj_group.create_dataset('ion1_positions', data=positions[0,:num_steps,0,:])
                traj_group.create_dataset('ion2_positions', data=positions[0,:num_steps,1,:])
                traj_group.create_dataset('collision_positions', data=positions[0,:num_steps,2,:])
            
            print(f"Termination reason: {get_termination_reason_str(termination_reason)}")
            
        except cuda.cudadrv.driver.CudaAPIError as e:
            print(f"\nCUDA Error: {str(e)}")
            print("Attempting cleanup and continuing...")
        
        finally:
            # Cleanup GPU resources
            force_gpu_cleanup()

    return all_reorders, all_termination_reasons

def main():
    """Main function for trajectory simulation"""
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
    reorders, termination_reasons = run_trajectory_simulation(selected_indices)
    end_time = time.perf_counter()
    
    # Print summary statistics
    print("\nSimulation Results:")
    print("-" * 50)
    
    for idx, (reorder, reason) in enumerate(zip(reorders, termination_reasons)):
        shot_idx = selected_indices[idx]
        print(f"\nShot {shot_idx}:")
        print(f"Termination reason: {get_termination_reason_str(reason)}")
        if reorder > 0:
            print(f"Reorder type: {'Ion reordering' if reorder == 1 else 'Ion ejection'}")
    
    print("\nOverall Statistics:")
    print(f"Total reorders: {np.sum(reorders == 1)}")
    print(f"Total ejections: {np.sum(reorders == 2)}")
    print(f"Total events: {np.sum(reorders > 0)}")
    print(f"Computation time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()