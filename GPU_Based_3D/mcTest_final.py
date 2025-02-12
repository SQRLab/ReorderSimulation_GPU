import numpy as np
from numba import cuda
from tqdm import tqdm
import time
import random
import json
import os
import subprocess
import threading
import queue

# Reading from
from cuda_3d_final import *


################################################################################
############### Final test function to simulate the collisions #################
################################################################################

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

dtSmall = 1e-12 ;dtCollision = 1e-16; dtLarge = 1e-10 # length of a time step in s

#######################
###### IMPORTANT ######
#######################

# Current number of ions in the chain
Ni = 3

# Nc represents the number of collision particles (currently fixed at 1)
Nc = 1

Nz_values = [5001]  # AXIAL
Nr_values = [1001]  #RADIAL
ion_counts = [3]  # Array of ion counts to test
shot_sizes = [10000, 50000, 100000]  # Array of shot sizes to test

vf = makeVf(Ni,1.0*q,m,wz)

Nt = 700000
T = 300
collisionalMass = 2
vMin = 50
vMax = 5000
numBins = 1000
boltzDist = Boltz(collisionalMass,T,vMin,vMax,numBins)
v = np.linspace(vMin,vMax,numBins)
angles = np.linspace(-np.pi/2,np.pi/2,100)
offsets = np.linspace(-2e-9,2e-9,200)
max_hypotenuse = 1.5e-7

def wait_for_gpu_cooldown(target_util=5, target_power=6, check_interval=5, timeout=300):
    """
    Wait for GPU to cool down to a stable idle state.
    
    Args:
        target_util (float): Target GPU utilization percentage to reach
        target_power (float): Target power consumption in watts (if None, will use initial power + 10W as threshold)
        check_interval (float): How often to check GPU status in seconds
        timeout (int): Maximum time to wait in seconds
    
    Returns:
        bool: True if GPU reached target state, False if timed out
    """
    print("\nWaiting for GPU to cool down...")
    
    # Get initial readings
    initial_stats = get_gpu_stats()
    if not initial_stats:
        print("Could not get GPU stats, skipping cooldown")
        return False
        
    if target_power is None:
        # Set target power to initial power + 10W as buffer
        target_power = initial_stats['power_draw'] + 10
    
    start_time = time.time()
    stable_readings = 0
    required_stable_readings = 3  # Number of consecutive readings that must be below threshold
    
    while (time.time() - start_time) < timeout:
        stats = get_gpu_stats()
        if not stats:
            continue
            
        current_util = stats['gpu_utilization']
        current_power = stats['power_draw']
        
        print(f"\rGPU Utilization: {current_util:.1f}%, Power: {current_power:.1f}W", end="")
        
        if current_util <= target_util and current_power <= target_power:
            stable_readings += 1
            if stable_readings >= required_stable_readings:
                print(f"\nGPU has cooled down. Utilization: {current_util:.1f}%, Power: {current_power:.1f}W")
                return True
        else:
            stable_readings = 0
            
        time.sleep(check_interval)
    
    print("\nGPU cooldown timed out!")
    return False

def get_gpu_stats():

    '''
    Retrieves current GPU statistics using nvidia-smi
    
    Returns:
        dict: Contains memory usage (MB), GPU utilization (%),
              and power consumption (W)
    '''

    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,utilization.gpu,power.draw', 
             '--format=csv,nounits,noheader'
        ], encoding='utf-8')
        memory_used, gpu_util, power_draw = map(float, result.strip().split(','))
        return {
            'memory_used': memory_used,  # MB
            'gpu_utilization': gpu_util,  # %
            'power_drawn': power_draw      # Watts
        }
    except:
        return None

class GPUMonitor:

    '''
    Class for continuous monitoring of GPU metrics during simulation
    
    Features:
    - Runs in separate thread to avoid impacting simulation
    - Collects utilization, memory, and power statistics
    - Provides queue-based data collection
    '''

    def __init__(self, interval=0.1):
        self.interval = interval
        self.stats_queue = queue.Queue()
        self.stop_flag = threading.Event()
        
    def monitor(self):
        while not self.stop_flag.is_set():
            stats = get_gpu_stats()
            if stats:
                self.stats_queue.put((time.time(), stats))
            time.sleep(self.interval)
    
    def start(self):
        self.stop_flag.clear()
        self.monitor_thread = threading.Thread(target=self.monitor)
        self.monitor_thread.start()
    
    def stop(self):
        self.stop_flag.set()
        self.monitor_thread.join()
        
    def get_stats(self):
        stats_list = []
        while not self.stats_queue.empty():
            stats_list.append(self.stats_queue.get())
        return stats_list

def main():

    '''
    Main simulation execution function
    
    Key steps:
    1. Configuration and File Setup
        - Loads/creates tracking files for computation times and GPU stats
        - Initializes GPU monitoring
        
    2. Grid Setup
        - Creates spatial grids for RF and DC fields
        - Transfers field data to GPU
        
    3. Monte Carlo Setup
        - Initializes arrays for particle positions and velocities
        - Generates random initial conditions for collisions
        - Prepares data buffers for results
        
    4. Simulation Execution
        - Launches CUDA kernels for parallel trajectory computation
        - Monitors progress with progress bars
        - Tracks GPU performance metrics
        
    5. Results Processing
        - Saves trajectory data to files
        - Records computation times and GPU statistics
        - Implements GPU cooldown periods between runs
        
    Parameters:
        None (uses global configuration variables)
    
    Output:
        - Creates detailed log files of simulation results
        - Generates performance metrics and GPU statistics
        - Saves timing information for different simulation configurations
    
    Notes:
        - Uses buffered writing for efficient I/O
        - Implements comprehensive error handling and GPU monitoring
        - Supports multiple grid sizes and ion configurations
        - Includes progress tracking and user feedback
    '''

    computation_times_file = "computation_times_detailed.json"
    gpu_stats_file = "gpu_stats_detailed.json"

    # Load existing files if they exist
    if os.path.exists(computation_times_file):
        with open(computation_times_file, 'r') as f:
            computation_times = json.load(f)
    else:
        computation_times = {}
        
    if os.path.exists(gpu_stats_file):
        with open(gpu_stats_file, 'r') as f:
            gpu_stats = json.load(f)
    else:
        gpu_stats = {}
    
    gpu_monitor = GPUMonitor()
    
    for Nz_value in Nz_values:
        for Nr_value in Nr_values:
            
            Nz = Nz_value ; Nr = Nr_value
            Nrmid = (Nr-1)/2 ; Nzmid = (Nz-1)/2
            
            Dr = 3.00015e-5 ; Dz = 9.00045e-5
            dr = Dr/float(Nr) ; dz = Dz/float(Nz)

            # Start GPU monitoring and timing for RF and DC calculations
            gpu_monitor.start()
            rf_dc_start_time = time.perf_counter()

            RFx,RFy = makeRF0(m,q,wr,Nr,Nz,Nrmid,Nzmid,dr)
            DC = makeDC(m,q,wz,Nz,Nr,Nzmid,dz)
            
            rf_dc_end_time = time.perf_counter()
            rf_dc_time = rf_dc_end_time - rf_dc_start_time
            
            # Get GPU stats for RF/DC calculation
            rf_dc_gpu_stats = gpu_monitor.get_stats()
            gpu_monitor.stop()
            
            for Ni in ion_counts:
                for shots in shot_sizes:
                    print(f"\nStarting simulation with grid size Nr: {Nr_value} x Nz: {Nz_value}, ion count {Ni}, and shots {shots}")
                    
                    # Start GPU monitoring and timing for simulation
                    gpu_monitor.start()
                    sim_start_time = time.perf_counter()

                    BAR_FORMAT = '{desc:<20}: {percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
                    
                    # Setup progress bar
                    with tqdm(total=100, desc="Setup Progress", 
                            position=0, leave=True,
                            bar_format=BAR_FORMAT) as setup_pbar:

                        buffer_size = 10
                        data_buffer = []
                        output_file = f"simulation_results_10k/{Ni}ionSimulation_{Nr_value} x {Nz_value}_{shots}shots.txt"
                        os.makedirs("simulation_results_10k", exist_ok=True)

                        with open(output_file, "w") as f:
                            f.write("axial trapping frequency (MHz) \t velocity(m/s) \t ion collided with \t angleXY(rads) \t angleZ(rads) \t collision offset(m) \t reorder? (1 is reorder 2 is ejection) \n")
                        setup_pbar.update(10)
                        
                        vf_all = np.zeros((shots, Ni, 9), dtype=np.float64)
                        xc_all = np.zeros(shots, dtype=np.float64)
                        yc_all = np.zeros(shots, dtype=np.float64)
                        zc_all = np.zeros(shots, dtype=np.float64)
                        vxc_all = np.zeros(shots, dtype=np.float64)
                        vyc_all = np.zeros(shots, dtype=np.float64)
                        vzc_all = np.zeros(shots, dtype=np.float64)
                        Nt_all = np.zeros(shots, dtype=np.int32)
                        setup_pbar.update(10)

                        collision = True

                        actual_velocities = np.zeros(shots)
                        actual_anglesXY = np.zeros(shots)
                        actual_anglesZ = np.zeros(shots)
                        actual_offsets = np.zeros(shots)
                        actual_ions = np.zeros(shots, dtype=int)
                        setup_pbar.update(10)
                        
                        for i in range(shots):
                            velocity = random.choices(v,weights=boltzDist)[0]
                            angle_choiceXY = random.choice(angles)
                            angle_choiceZ = random.choice(angles)
                            offset_choice = random.choice(offsets)
                            ion_collided = random.randint(0,Ni-1)

                            actual_velocities[i] = velocity
                            actual_anglesXY[i] = angle_choiceXY
                            actual_anglesZ[i] = angle_choiceZ
                            actual_offsets[i] = offset_choice
                            actual_ions[i] = ion_collided
                            
                            x = -np.cos(angle_choiceZ)*max_hypotenuse 
                            y = np.sin(angle_choiceXY)*x
                            x = np.cos(angle_choiceXY)*x
                            vx = np.abs(velocity*np.cos(angle_choiceZ))
                            vy = np.sin(angle_choiceXY)*vx
                            vx = np.cos(angle_choiceXY)*vx
                            z = vf[ion_collided,2] + np.sin(angle_choiceZ)*max_hypotenuse + offset_choice
                            vz=-1*velocity*np.sin(angle_choiceZ)

                            vf_all[i, :, :] = vf
                            xc_all[i] = x
                            yc_all[i] = y
                            zc_all[i] = z
                            vxc_all[i] = vx
                            vyc_all[i] = vy
                            vzc_all[i] = vz
                            Nt_all[i] = Nt
                            
                            if i % (shots // 40) == 0:  # Update progress every 2.5%
                                setup_pbar.update(1)
                        setup_pbar.update(30)

                        vf_device = cuda.to_device(vf_all)
                        xc_device = cuda.to_device(xc_all)
                        yc_device = cuda.to_device(yc_all)
                        zc_device = cuda.to_device(zc_all)
                        vxc_device = cuda.to_device(vxc_all)
                        vyc_device = cuda.to_device(vyc_all)
                        vzc_device = cuda.to_device(vzc_all)
                        Nt_device = cuda.to_device(Nt_all)
                        reorder_device = cuda.device_array(shots, dtype=np.int32)
                        setup_pbar.update(20)

                        RFx_device = cuda.to_device(RFx)
                        RFy_device = cuda.to_device(RFy)
                        DC_device = cuda.to_device(DC)
                        setup_pbar.update(20)

                    # Create mapped array for progress tracking
                    progress = cuda.mapped_array(1, dtype=np.int64)
                    progress[0] = 0
                    
                    # Simulation progress bar
                    with tqdm(total=shots, desc="Simulation Progress", 
                            position=0, leave=True,
                            bar_format=BAR_FORMAT) as sim_pbar:
                        
                        threads_per_block = 256
                        blocks = (shots + threads_per_block - 1) // threads_per_block

                        mcCollision_kernel[blocks, threads_per_block](
                            vf_device, xc_device, yc_device, zc_device, 
                            vxc_device, vyc_device, vzc_device,
                            q, mH2, aH2, Nt_device, dtSmall, dtCollision, 
                            RFx_device, RFy_device, DC_device, Nr, Nz, dr, dz,
                            dtLarge, reorder_device, collision, progress
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
                    
                    with open(output_file, "a") as f:
                        for i in range(shots):
                            output = f"{wz}\t{actual_velocities[i]}\t{actual_ions[i]+1}\t{actual_anglesXY[i]}\t{actual_anglesZ[i]}\t{actual_offsets[i]}\t{reorder[i]}\n"
                            data_buffer.append(output)

                            if len(data_buffer) == buffer_size or i == shots - 1:
                                f.writelines(data_buffer)
                                f.flush()
                                data_buffer = []
                    
                    sim_end_time = time.perf_counter()
                    sim_time = sim_end_time - sim_start_time
                    total_time = sim_time + rf_dc_time

                    # Get GPU stats for simulation
                    sim_gpu_stats = gpu_monitor.get_stats()
                    gpu_monitor.stop()

                    # Process GPU statistics
                    if rf_dc_gpu_stats:
                        rf_dc_max_memory = max(stat[1]['memory_used'] for stat in rf_dc_gpu_stats)
                        rf_dc_max_util = max(stat[1]['gpu_utilization'] for stat in rf_dc_gpu_stats)
                        rf_dc_max_power = max(stat[1]['power_draw'] for stat in rf_dc_gpu_stats)
                        
                        rf_dc_avg_memory = sum(stat[1]['memory_used'] for stat in rf_dc_gpu_stats) / len(rf_dc_gpu_stats)
                        rf_dc_avg_util = sum(stat[1]['gpu_utilization'] for stat in rf_dc_gpu_stats) / len(rf_dc_gpu_stats)
                        rf_dc_avg_power = sum(stat[1]['power_draw'] for stat in rf_dc_gpu_stats) / len(rf_dc_gpu_stats)
                    else:
                        rf_dc_max_memory = rf_dc_max_util = rf_dc_max_power = None
                        rf_dc_avg_memory = rf_dc_avg_util = rf_dc_avg_power = None

                    if sim_gpu_stats:
                        sim_max_memory = max(stat[1]['memory_used'] for stat in sim_gpu_stats)
                        sim_max_util = max(stat[1]['gpu_utilization'] for stat in sim_gpu_stats)
                        sim_max_power = max(stat[1]['power_draw'] for stat in sim_gpu_stats)
                        
                        sim_avg_memory = sum(stat[1]['memory_used'] for stat in sim_gpu_stats) / len(sim_gpu_stats)
                        sim_avg_util = sum(stat[1]['gpu_utilization'] for stat in sim_gpu_stats) / len(sim_gpu_stats)
                        sim_avg_power = sum(stat[1]['power_draw'] for stat in sim_gpu_stats) / len(sim_gpu_stats)
                    else:
                        sim_max_memory = sim_max_util = sim_max_power = None
                        sim_avg_memory = sim_avg_util = sim_avg_power = None

                    print(f"RF/DC calculation time: {rf_dc_time:.2f} seconds")
                    print(f"Simulation time: {sim_time:.2f} seconds")
                    print(f"Total time: {total_time:.2f} seconds")
                    
                    print("\nGPU Statistics:")
                    print("RF/DC Phase:")
                    print(f"  Max Memory: {rf_dc_max_memory:.1f}MB, Avg Memory: {rf_dc_avg_memory:.1f}MB")
                    print(f"  Max Utilization: {rf_dc_max_util:.1f}%, Avg Utilization: {rf_dc_avg_util:.1f}%")
                    print(f"  Max Power: {rf_dc_max_power:.1f}W, Avg Power: {rf_dc_avg_power:.1f}W")
                    print("Simulation Phase:")
                    print(f"  Max Memory: {sim_max_memory:.1f}MB, Avg Memory: {sim_avg_memory:.1f}MB")
                    print(f"  Max Utilization: {sim_max_util:.1f}%, Avg Utilization: {sim_avg_util:.1f}%")
                    print(f"  Max Power: {sim_max_power:.1f}W, Avg Power: {sim_avg_power:.1f}W")

                    # Store GPU statistics
                    if str(Nr_value) not in gpu_stats:
                        gpu_stats[str(Nr_value)] = {}
                    if str(Nz_value) not in gpu_stats[str(Nr_value)]:
                        gpu_stats[str(Nr_value)][str(Nz_value)] = {}
                    if str(Ni) not in gpu_stats[str(Nr_value)][str(Nz_value)]:
                        gpu_stats[str(Nr_value)][str(Nz_value)][str(Ni)] = {}
                    
                    gpu_stats[str(Nr_value)][str(Nz_value)][str(Ni)][str(shots)] = {
                        "rf_dc_phase": {
                            "max_memory_used_mb": rf_dc_max_memory,
                            "avg_memory_used_mb": rf_dc_avg_memory,
                            "max_gpu_utilization_percent": rf_dc_max_util,
                            "avg_gpu_utilization_percent": rf_dc_avg_util,
                            "max_power_draw_watts": rf_dc_max_power,
                            "avg_power_draw_watts": rf_dc_avg_power
                        },
                        "simulation_phase": {
                            "max_memory_used_mb": sim_max_memory,
                            "avg_memory_used_mb": sim_avg_memory,
                            "max_gpu_utilization_percent": sim_max_util,
                            "avg_gpu_utilization_percent": sim_avg_util,
                            "max_power_draw_watts": sim_max_power,
                            "avg_power_draw_watts": sim_avg_power
                        }
                    }

                    with open(gpu_stats_file, 'w') as f:
                        json.dump(gpu_stats, f, indent=2)

                    # Store timing information in nested dictionary
                    if str(Nr_value) not in computation_times:
                        computation_times[str(Nr_value)] = {}
                    if str(Nz_value) not in computation_times[str(Nr_value)]:
                        computation_times[str(Nr_value)][str(Nz_value)] = {}
                    if str(Ni) not in computation_times[str(Nr_value)][str(Nz_value)]:
                        computation_times[str(Nr_value)][str(Nz_value)][str(Ni)] = {}
                    
                    computation_times[str(Nr_value)][str(Nz_value)][str(Ni)][str(shots)] = {
                        "rf_dc_time": rf_dc_time,
                        "simulation_time": sim_time,
                        "total_time": total_time
                    }

                    with open(computation_times_file, 'w') as f:
                        json.dump(computation_times, f, indent=2)

                    print("\n" + "="*50)
                    wait_for_gpu_cooldown()
                    print("="*50 + "\n")

if __name__ == "__main__":
    main()