from cuda_functions import *
from scipy import special
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import time
import os
import json

def Boltz(m, T, vmin=0, vmax=5000, bins=100):
    amu = 1.66*10**-27
    m = m*amu
    k = 1.386e-23 # boltzmann constant
    boltz = np.zeros(bins) # initialize vector
    dv = (vmax - vmin)/bins # define bin spacing in speed
    a = (k*T/m)**(1/2) # normalization constant for distribution function

    for i in range(bins):
        vhere = vmin + i*dv # define speed of bin
        vlast = vhere-dv
        boltz[i] = (special.erf(vhere/(a*np.sqrt(2))) - np.sqrt(2/np.pi)*(vhere/a)*np.exp(-vhere**2/(2*a**2)) ) - (special.erf(vlast/(a*np.sqrt(2))) - np.sqrt(2/np.pi)*(vlast/a)*np.exp(-vlast**2/(2*a**2)) )

    return boltz/np.sum(boltz)

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

# Define useful constants
amu = 1.67e-27
eps0 = 8.854e-12
qe = 1.6e-19 # SI units 

# Define physical params
m = 40. *amu
q = 1. *qe
wr = 2*np.pi*3e6 # SI units

# Define sim params
Dr = 30001.5e-9
Dz = 90001.5e-9 # physical width in m of the sim

# wz can be varied if desired
wz = 2*np.pi*1e6

aH2 = 8e-31 # dipole moment of H2 in SI units
mH2 = 2.0*amu # mass of H2 in kg

dtSmall = 1e-12
dtCollision = 1e-16
dtLarge = 1e-10 # length of a time step in s

sigmaV = 100e-6 # fall-off of potential outside trapping region
dv = 20.0 # bin size for particle speed in determining if collision occurs
vmax = 5000 # maximum particle speed we allow
l = 1
vbumpr = 0.00e0
vbumpz = -0.0e0 # starting velocity in r and z of the lth ion in the chain
offsetz = 0.0e-7
offsetr = 0.0e-8 # starting distance from eq. in r and z of the lth ion in the chain

# Simulation parameters
T = 300
collisionalMass = 2
vMin = 50
vMax = 7000
numBins = 1000
boltzDist = Boltz(collisionalMass, T, vMin, vMax, numBins)
v = np.linspace(vMin, vMax, numBins)
angles = np.linspace(-np.pi/2, np.pi/2, 100)
offsets = np.linspace(-2e-9, 2e-9, 200)
max_hypotenuse = 1.5e-5

buffer_size = 10

grid_sizes = [10001]  # Array of grid sizes to test
ion_counts = [2, 3]  # Array of ion counts to test
shot_sizes = [1000]  # Array of shot sizes to test

# File to store computation times
computation_times_file = "computation_times.json"
formatted_table_file = "computation_times_table.txt"

# Load existing computation times if file exists
if os.path.exists(computation_times_file):
    with open(computation_times_file, 'r') as f:
        computation_times = json.load(f)
else:
    computation_times = {}

os.makedirs("simulation_results", exist_ok=True)

@cuda.jit
def determine_Nt(v, Nt):
    i = cuda.grid(1)
    if i < v.shape[0]:
        if v[i] < 200:
            Nt[i] = 700000
        elif v[i] < 1500:
            Nt[i] = 400000
        else:
            Nt[i] = 250000

@cuda.jit
def run_simulations(vf, RF, DC, Nr, Nz, dr, dz, dtSmall, dtLarge, dtCollision, wz, v, angles, offsets, max_hypotenuse, Nt, results):
    i = cuda.grid(1)
    if i < results.shape[0]:
        velocity = v[i]
        angle_choice = angles[i]
        offset_choice = offsets[i]
        ion_collided = i % vf.shape[0]

        r = -math.cos(angle_choice) * max_hypotenuse
        z = vf[ion_collided, 1] + math.sin(angle_choice) * max_hypotenuse + offset_choice
        vz = -1 * velocity * math.sin(angle_choice)
        vr = abs(velocity * math.cos(angle_choice))

        reorder = mcCollision(vf, r, z, vr, vz, q, mH2, aH2, Nt[i], dtSmall, RF, DC, Nr, Nz, dr, dz, dtLarge, dtCollision)

        results[i, 0] = wz
        results[i, 1] = velocity
        results[i, 2] = ion_collided + 1
        results[i, 3] = angle_choice
        results[i, 4] = offset_choice
        results[i, 5] = reorder

for grid_size in grid_sizes:
    Nr = Nz = grid_size
    Nrmid = Nzmid = (Nr-1)/2
    dr = Dr/float(Nr)
    dz = Dz/float(Nz)

    # Recalculate RF and DC fields for each grid size
    RF = makeRF0(m, q, wr, Nr, Nz, Nrmid, dr)
    DC = makeDC(m, q, wz, Nz, Nr, Nzmid, dz)

    # Move RF and DC to GPU
    RF_gpu = cuda.to_device(RF)
    DC_gpu = cuda.to_device(DC)

    for Ni in ion_counts:
        for shots in shot_sizes:
            print(f"Starting simulation with grid size {grid_size}, ion count {Ni}, and shots {shots}")

            start_time = time.perf_counter()
            file_name = f"simulation_results/{Ni}ionSimulation_{grid_size}_{shots}shots.txt"

            # Prepare GPU arrays
            v_gpu = cuda.to_device(np.random.choice(v, size=shots, p=boltzDist))
            angles_gpu = cuda.to_device(np.random.choice(angles, size=shots))
            offsets_gpu = cuda.to_device(np.random.choice(offsets, size=shots))
            results_gpu = cuda.device_array((shots, 6), dtype=np.float64)

            # Prepare initial ion positions
            vf = makeVf(Ni, 1.0*q, m, l, wz, offsetr, offsetz, vbumpr, vbumpz)
            vf_gpu = cuda.to_device(vf)

            # Determine Nt based on velocity ranges
            Nt_gpu = cuda.device_array(shots, dtype=np.int32)
            threads_per_block = 256
            blocks = (shots + threads_per_block - 1) // threads_per_block
            determine_Nt[blocks, threads_per_block](v_gpu, Nt_gpu)

            # Launch simulation kernel
            run_simulations[blocks, threads_per_block](vf_gpu, RF_gpu, DC_gpu, Nr, Nz, dr, dz, dtSmall, dtLarge, dtCollision, wz, v_gpu, angles_gpu, offsets_gpu, max_hypotenuse, Nt_gpu, results_gpu)

            # Copy results back to host
            results = results_gpu.copy_to_host()

            # Write results to file
            with open(file_name, "w") as f:
                f.write("axial trapping frequency (MHz) \t velocity(m/s) \t ion collided with \t angle(rads) \t collision offset(m) \t reorder? (1 is reorder 2 is ejection) \n")
                for result in results:
                    f.write("\t".join(map(str, result)) + "\n")

            finish_time = time.perf_counter()
            timeTaken = finish_time - start_time
            
            # Update computation times
            if str(grid_size) not in computation_times:
                computation_times[str(grid_size)] = {}
            if str(Ni) not in computation_times[str(grid_size)]:
                computation_times[str(grid_size)][str(Ni)] = {}
            computation_times[str(grid_size)][str(Ni)][str(shots)] = timeTaken
            
            # Save updated computation times
            with open(computation_times_file, 'w') as f:
                json.dump(computation_times, f, indent=2)
            
            print(f"Completed simulation for grid size {grid_size}, ion count {Ni}, and shots {shots}. It took {timeTaken} seconds!")

create_formatted_table_file(computation_times, formatted_table_file)
print("All simulations completed successfully!")