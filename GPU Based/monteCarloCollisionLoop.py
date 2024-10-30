import cupy as cp
import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import math
import time
import os
import json
from scipy import special
from Collision2DClean import *

@cuda.jit
def boltz_kernel(m, T, v, result):
    i = cuda.grid(1)
    if i < v.shape[0]:
        amu = 1.66e-27
        k = 1.386e-23
        a = math.sqrt(k * T / (m * amu))
        vhere = v[i]
        vlast = v[i-1] if i > 0 else 0
        result[i] = (math.erf(vhere/(a*math.sqrt(2))) - math.sqrt(2/math.pi)*(vhere/a)*math.exp(-vhere**2/(2*a**2))) - \
                    (math.erf(vlast/(a*math.sqrt(2))) - math.sqrt(2/math.pi)*(vlast/a)*math.exp(-vlast**2/(2*a**2)))

def Boltz(m, T, vmin=0, vmax=5000, bins=100):
    v = cp.linspace(vmin, vmax, bins)
    result = cp.zeros(bins)
    threads_per_block = 256
    blocks = (bins + threads_per_block - 1) // threads_per_block
    boltz_kernel[blocks, threads_per_block](m, T, v, result)
    return result / cp.sum(result)

@cuda.jit(device=True)
def mcCollision(vf, rc, zc, vrc, vzc, qc, mc, ac, Nt, dtSmall, RF, DC, Nr, Nz, dr, dz, dtLarge, dtCollision, eii=0.01, eid=0.01):
    # Move data to GPU
    d_vf = cp.asarray(vf)
    d_vc = cp.asarray([[rc, zc, vrc, vzc, qc, mc, ac]])
    d_RF = cp.asarray(RF)
    d_DC = cp.asarray(DC)
   
    Ni = vf.shape[0]
    Nc = 1

    # Allocate memory on GPU
    d_rs = cp.zeros((Ni, Nt), dtype=np.float32)
    d_zs = cp.zeros((Ni, Nt), dtype=np.float32)
    d_vrs = cp.zeros((Ni, Nt), dtype=np.float32)
    d_vzs = cp.zeros((Ni, Nt), dtype=np.float32)
    d_rcolls = cp.zeros((Nc, Nt), dtype=np.float32)
    d_zcolls = cp.zeros((Nc, Nt), dtype=np.float32)
    d_vrcolls = cp.zeros((Nc, Nt), dtype=np.float32)
    d_vzcolls = cp.zeros((Nc, Nt), dtype=np.float32)
    d_reorder = cp.zeros(1, dtype=np.int32)
   
    # Launch kernel
    threads_per_block = 256
    blocks = (Nt + threads_per_block - 1) // threads_per_block
    mc_collision_kernel[blocks, threads_per_block](
        d_vf, d_vc, d_RF, d_DC, d_rs, d_zs, d_vrs, d_vzs, d_rcolls, d_zcolls, d_vrcolls, d_vzcolls,
        Nt, Nr, Nz, dr, dz, dtSmall, dtLarge, dtCollision, eii, eid, d_reorder
    )
   
    # Copy results back to host
    vf = cp.asnumpy(d_vf)
    rs = cp.asnumpy(d_rs)
    zs = cp.asnumpy(d_zs)
    reorder = cp.asnumpy(d_reorder)[0]
   
    return reorder

def create_formatted_table_file(computation_times, output_file):
    grid_sizes = sorted(set(int(size) for size in computation_times.keys()))
    ion_counts = sorted(set(int(count) for size in computation_times.values() for count in size.keys()))
    shot_sizes = sorted(set(int(shots) for size in computation_times.values() for count in size.values() for shots in count.keys()))

    with open(output_file, 'w') as f:
        # Write header
        f.write("Computational Times in seconds\n")
        f.write(" ".join([f"{size//1000}k".rjust(8) for size in grid_sizes]) + "\n")

        for ion_count in ion_counts:
            for shot_size in shot_sizes:
                row = [f"{ion_count}".rjust(2)]
                for grid_size in grid_sizes:
                    time = computation_times.get(str(grid_size), {}).get(str(ion_count), {}).get(str(shot_size), None)
                    if time is not None:
                        row.append(f"{time:.0f}".rjust(8))
                    else:
                        row.append(" ".rjust(8))
                row.append(str(shot_size).rjust(8))
                f.write(" ".join(row) + "\n")
            f.write("\n")  # Add a blank line between different ion counts

    print(f"Formatted table has been written to {output_file}")

@cuda.jit
def simulate_collisions_kernel(rng_states, vf, v, boltzDist, angles, offsets, max_hypotenuse, 
                               Ni, RF, DC, Nr, Nz, dr, dz, dtLarge, dtCollision, results):
    i = cuda.grid(1)
    if i < results.shape[0]:
        # Use xoroshiro128p RNG for random number generation
        thread_id = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
        
        # Generate random numbers
        rand1 = xoroshiro128p_uniform_float32(rng_states, thread_id)
        rand2 = xoroshiro128p_uniform_float32(rng_states, thread_id)
        rand3 = xoroshiro128p_uniform_float32(rng_states, thread_id)
        rand4 = xoroshiro128p_uniform_float32(rng_states, thread_id)
        
        # Use the random numbers for various selections
        velocity_index = int(rand1 * len(v))
        velocity = v[velocity_index]
        
        angle_index = int(rand2 * len(angles))
        angle_choice = angles[angle_index]
        
        offset_index = int(rand3 * len(offsets))
        offset_choice = offsets[offset_index]
        
        ion_collided = int(rand4 * Ni)

        if velocity < 200:
            Nt = 700000
        elif velocity < 1500:
            Nt = 400000
        else:
            Nt = 250000

        r = -math.cos(angle_choice) * max_hypotenuse
        z = vf[ion_collided, 1] + math.sin(angle_choice) * max_hypotenuse + offset_choice
        vz = -1 * velocity * math.sin(angle_choice)
        vr = abs(velocity * math.cos(angle_choice))

        reorder = mcCollision(vf, r, z, vr, vz, 1.6e-19, 2.0 * 1.67e-27, 8e-31, Nt, RF, DC, Nr, Nz, dr, dz, dtLarge, dtCollision)

        results[i, 0] = velocity
        results[i, 1] = ion_collided + 1
        results[i, 2] = angle_choice
        results[i, 3] = offset_choice
        results[i, 4] = reorder

def run_simulation(Ni, grid_size, shots, wz, RF, DC, Nr, Nz, dr, dz, dtLarge, dtCollision):
    # Move constants and parameters to GPU
    d_v = cp.linspace(50, 7000, 1000)
    d_boltzDist = Boltz(2, 300, 50, 7000, 1000)
    d_angles = cp.linspace(-cp.pi/2, cp.pi/2, 100)
    d_offsets = cp.linspace(-2e-9, 2e-9, 200)
    max_hypotenuse = 1.5e-5

    d_vf = cp.array(makeVf(Ni, 1.6e-19, 40 * 1.67e-27, 1, wz, 0.0e-8, 0.0e-7, 0.00e0, -0.0e0))
    d_RF = cp.array(RF)
    d_DC = cp.array(DC)

    d_results = cp.zeros((shots, 5))

    threads_per_block = 256
    blocks = (shots + threads_per_block - 1) // threads_per_block

    # Create RNG states
    rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed=1234)

    simulate_collisions_kernel[blocks, threads_per_block](
        rng_states, d_vf, d_v, d_boltzDist, d_angles, d_offsets, max_hypotenuse, 
        Ni, d_RF, d_DC, Nr, Nz, dr, dz, dtLarge, dtCollision, d_results
    )

    return cp.asnumpy(d_results)

# Main script
amu = 1.67e-27
eps0 = 8.854e-12
qe = 1.6e-19

m = 40. * amu
q = 1. * qe
wr = 2 * np.pi * 3e6
wz = 2 * np.pi * 1e6

Dr = 30001.5e-9
Dz = 90001.5e-9

dtSmall = 1e-12
dtCollision = 1e-16
dtLarge = 1e-10

grid_sizes = [25001]
ion_counts = [2]
shot_sizes = [200]

computation_times_file = "computation_times_cuda.json"
formatted_table_file = "computation_times_table_cuda.txt"

if os.path.exists(computation_times_file):
    with open(computation_times_file, 'r') as f:
        computation_times = json.load(f)
else:
    computation_times = {}

os.makedirs("simulation_results_cuda", exist_ok=True)

for grid_size in grid_sizes:
    Nr = Nz = grid_size
    Nrmid = Nzmid = (Nr-1)/2
    dr = Dr/float(Nr)
    dz = Dz/float(Nz)

    RF = makeRF0_gpu(m, q, wr, Nr, Nz, Nrmid, dr)
    DC = makeDC(m, q, wz, Nz, Nr, Nzmid, dz)

    for Ni in ion_counts:
        for shots in shot_sizes:
            print(f"Starting simulation with grid size {grid_size}, ion count {Ni}, and shots {shots}")

            start_time = time.perf_counter()
            file_name = f"simulation_results_cuda/{Ni}ionSimulation_{grid_size}_{shots}shots.txt"

            results = run_simulation(Ni, grid_size, shots, wz, RF, DC, Nr, Nz, dr, dz, dtLarge, dtCollision)

            with open(file_name, "w") as f:
                f.write("axial trapping frequency (MHz) \t velocity(m/s) \t ion collided with \t angle(rads) \t collision offset(m) \t reorder? (1 is reorder 2 is ejection) \n")
                for result in results:
                    f.write(f"{wz}\t{result[0]}\t{int(result[1])}\t{result[2]}\t{result[3]}\t{int(result[4])}\n")

            finish_time = time.perf_counter()
            timeTaken = finish_time - start_time
            
            if str(grid_size) not in computation_times:
                computation_times[str(grid_size)] = {}
            if str(Ni) not in computation_times[str(grid_size)]:
                computation_times[str(grid_size)][str(Ni)] = {}
            computation_times[str(grid_size)][str(Ni)][str(shots)] = timeTaken
            
            with open(computation_times_file, 'w') as f:
                json.dump(computation_times, f, indent=2)
            
            print(f"Completed simulation for grid size {grid_size}, ion count {Ni}, and shots {shots}. It took {timeTaken} seconds!")

create_formatted_table_file(computation_times, formatted_table_file)
print("All simulations completed successfully!")