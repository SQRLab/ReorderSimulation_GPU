import numpy as np
from scipy.optimize import fsolve
import scipy.constants as con
import math
from numba import cuda, float64, int32
import random
import time
import os
import json
from scipy import special
import matplotlib.pyplot as plt

# ============================
# Constants and Physical Params
# ============================

# Constants
amu = 1.67e-27        # Atomic mass unit in kg
eps0 = 8.854e-12      # Vacuum permittivity in F/m
qe = 1.6e-19          # Elementary charge in Coulombs

# Physical parameters
m = 40.0 * amu        # Mass of ion in kg
q = 1.0 * qe          # Charge of ion in C
wr = 2 * np.pi * 3e6   # Radial angular frequency (rad/s)
wz = 2 * np.pi * 1e6   # Axial angular frequency (rad/s)
aH2 = 8e-31           # Polarizability of H2 in SI units (C m² / V)
mH2 = 2.0 * amu       # Mass of H2 molecule in kg

# Simulation parameters
Dr = 30001.5e-9       # Radial grid size in meters
Dz = 90001.5e-9       # Axial grid size in meters
dtSmall = 1e-11       # Small timestep in seconds
dtCollision = 1e-16   # Collision timestep in seconds
dtLarge = 1e-10       # Large timestep in seconds

# Boltzmann distribution parameters
sigmaV = 100e-6
dv = 20.0
vmax = 5000
l = 1
vbumpr = 0.00e0
vbumpz = -0.0e0
offsetz = 0.0e-7
offsetr = 0.0e-8

# ============================
# Helper Functions
# ============================

def ion_position_potential(x):
    N = len(x)
    return [
        x[m] - sum([1 / (x[m] - x[n]) ** 2 for n in range(m)]) +
        sum([1 / (x[m] - x[n]) ** 2 for n in range(m + 1, N)])
        for m in range(N)
    ]

def calcPositions(N):
    estimated_extreme = 0.481 * N ** 0.765  # Empirical estimate
    return fsolve(ion_position_potential, np.linspace(-estimated_extreme, estimated_extreme, N))

def lengthScale(ν, M=None, Z=None):
    if M is None:
        M = con.atomic_mass * 39.9626  # Default mass if not provided
    if Z is None:
        Z = 1  # Default charge if not provided
    return ((Z ** 2 * con.elementary_charge ** 2) / (4 * np.pi * con.epsilon_0 * M * ν ** 2)) ** (1 / 3)

def makeRF0(m, q, w, Nr, Nz, Nrmid, dr):
    C = -m * (w ** 2) / q
    RF = np.ones((Nr, Nz), dtype=np.float64)
    for jCell in range(Nr):
        RF[jCell, :] = -RF[jCell, :] * C * (Nrmid - jCell) * dr
    return RF

def makeDC(m, q, w, Nz, Nr, Nzmid, dz):
    C = -m * (w ** 2) / q
    DC = np.ones((Nr, Nz), dtype=np.float64)
    for kCell in range(Nz):
        DC[:, kCell] = -DC[:, kCell] * C * (Nzmid - kCell) * dz
    return DC

def makeVf(Ni, q, m, l, wr, offsetr, offsetz, vbumpr, vbumpz):
    # Initialize ion parameters
    vf = np.zeros((Ni, 7), dtype=np.float64)
    pos = calcPositions(Ni)
    lscale = lengthScale(wr)
    scaledPos = pos * lscale
    for i in range(Ni):
        vf[i, :] = [0.0e-6, -scaledPos[i], 0.0, 0.0, q, m, 0.0]
        # Assign higher initial z-velocities to each ion to facilitate reordering
        vf[i, 3] = 1e-1 * (i + 1)  # Example: 0.1, 0.2, 0.3 m/s for Ni=3
    if l < Ni:  # Prevent index out of bounds
        vf[l, 0] += offsetr
        vf[l, 1] += offsetz
        vf[l, 2] += vbumpr
        vf[l, 3] += vbumpz
    return vf

def Boltz(m, T, vmin=0, vmax=5000, bins=100):
    amu = 1.67e-27  # Atomic mass unit in kg
    m = m * amu
    k = 1.386e-23  # Boltzmann constant in J/K
    boltz = np.zeros(bins, dtype=np.float64)
    dv = (vmax - vmin) / bins
    a = (k * T / m) ** 0.5

    for i in range(bins):
        vhere = vmin + i * dv
        vlast = vhere - dv
        boltz[i] = (
            (special.erf(vhere / (a * math.sqrt(2))) -
             math.sqrt(2 / math.pi) * (vhere / a) * math.exp(-vhere ** 2 / (2 * a ** 2))) -
            (special.erf(vlast / (a * math.sqrt(2))) -
             math.sqrt(2 / math.pi) * (vlast / a) * math.exp(-vlast ** 2 / (2 * a ** 2)))
        )
    return boltz / np.sum(boltz)

def create_formatted_table_file(computation_times, output_file):
    grid_sizes = sorted(set(int(size) for size in computation_times.keys()))
    ion_counts = sorted(set(int(count) for size in computation_times.values() for count in size.keys()))
    shot_sizes = sorted(set(int(shots) for size in computation_times.values() for count in size.values() for shots in count.keys()))

    with open(output_file, 'w') as f:
        f.write("Computational Times in seconds\n")
        f.write(" ".join([f"{size // 1000}k".rjust(8) for size in grid_sizes]) + "\n")

        for ion_count in ion_counts:
            for shot_size in shot_sizes:
                row = [f"{ion_count}".rjust(2)]
                for grid_size in grid_sizes:
                    time_taken = computation_times.get(str(grid_size), {}).get(str(ion_count), {}).get(str(shot_size), None)
                    if time_taken is not None:
                        row.append(f"{time_taken:.0f}".rjust(8))
                    else:
                        row.append(" ".rjust(8))
                row.append(str(shot_size).rjust(8))
                f.write(" ".join(row) + "\n")
            f.write("\n")

    print(f"Formatted table has been written to {output_file}")

# ============================
# CUDA Kernel Definition
# ============================

@cuda.jit
def mcCollision_kernel(
    vf_all, rc_all, zc_all, vrc_all, vzc_all,
    qc, mc, ac, Nt_all, dtSmall,
    RF, DC, Nr, Nz, dr, dz,
    dtLarge, dtCollision, nullFields, reorder_all
):
    idx = cuda.grid(1)
    if idx >= vf_all.shape[0]:
        return

    # Initialize reorder counter
    reorder = 0

    # Extract initial parameters
    r = rc_all[idx]
    z = zc_all[idx]
    vr = vrc_all[idx]
    vz = vzc_all[idx]
    Nt = Nt_all[idx]

    # Assuming vf_all has shape (shots, Ni, 7)
    Ni = vf_all.shape[1]

    # Simulation loop
    for step in range(Nt):
        rid2 = 1e6
        rii2 = 1e6
        vid2 = 1e6
        vii2 = 1e6

        # Calculate minimum distances and velocities
        for i in range(Ni):
            if vf_all[idx, i, 5] >= 1e6:
                continue  # Skip ejected ions
            for j in range(i + 1, Ni):
                if vf_all[idx, j, 5] >= 1e6:
                    continue
                r_diff = vf_all[idx, i, 0] - vf_all[idx, j, 0]
                z_diff = vf_all[idx, i, 1] - vf_all[idx, j, 1]
                vr_diff = vf_all[idx, i, 2] - vf_all[idx, j, 2]
                vz_diff = vf_all[idx, i, 3] - vf_all[idx, j, 3]
                dist2 = r_diff * r_diff + z_diff * z_diff
                v2 = vr_diff * vr_diff + vz_diff * vz_diff
                if dist2 < rii2:
                    vii2 = v2
                    rii2 = dist2
            # Check collision with collisional particles (Nc=1)
            # Assuming Nc=1; modify if Nc >1
            # Calculate distance and velocity relative to collision particle
            # Here, collision particle is defined by (r, z, vr, vz)
            r_diff = vf_all[idx, i, 0] - r
            z_diff = vf_all[idx, i, 1] - z
            vr_diff = vf_all[idx, i, 2] - vr
            vz_diff = vf_all[idx, i, 3] - vz
            dist2 = r_diff * r_diff + z_diff * z_diff
            v2 = vr_diff * vr_diff + vz_diff * vz_diff
            if dist2 < rid2:
                vid2 = v2
                rid2 = dist2

        # Determine collision mode
        a = ac
        e = 0.3
        if rid2 > 0:
            collision = (a * rii2) / (rid2 ** 2.5) > e
        else:
            collision = False

        # Adjust timestep
        if collision and vid2 > 0:
            dtNow = math.sqrt(rid2) * 0.01 / (5 * math.sqrt(vid2))
        else:
            dtNow = dtSmall
        if dtNow < dtCollision:
            dtNow = dtCollision

        # Solve fields
        # Initialize field arrays
        Erf = 0.0
        Ezf = 0.0

        for i in range(Ni):
            if vf_all[idx, i, 5] >= 1e6:
                continue  # Skip ejected ions

            # Convert positions to grid indices
            jCell_f = vf_all[idx, i, 0] / dr + (Nr - 1) / 2
            kCell_f = vf_all[idx, i, 1] / dz + (Nz - 1) / 2
            jCell = int(jCell_f + 0.5)
            kCell = int(kCell_f + 0.5)

            # Bounds checking
            if jCell < 1 or jCell >= Nr - 1 or kCell < 1 or kCell >= Nz - 1:
                continue  # Skip this ion if it's out of bounds

            # Calculate electric fields from DC and RF
            Erf_i = RF[jCell, kCell]
            Ezf_i = DC[jCell, kCell]

            # Add contributions from other ions
            for j in range(Ni):
                if j == i or vf_all[idx, j, 5] >= 1e6:
                    continue  # Skip self and ejected ions
                rdist = vf_all[idx, j, 0] - vf_all[idx, i, 0]
                zdist = vf_all[idx, j, 1] - vf_all[idx, i, 1]
                sqDist = rdist * rdist + zdist * zdist
                if sqDist < 1e-24:
                    continue  # Avoid division by zero
                dist = math.sqrt(sqDist)
                projR = rdist / dist
                projZ = zdist / dist
                C1 = 4 * math.pi * eps0
                Erf_i += -projR * vf_all[idx, j, 4] / (C1 * sqDist)
                Ezf_i += -projZ * vf_all[idx, j, 4] / (C1 * sqDist)

            # Accumulate total fields
            Erf += Erf_i
            Ezf += Ezf_i

        # Update velocities based on fields
        for i in range(Ni):
            if vf_all[idx, i, 5] >= 1e6:
                continue  # Skip ejected ions
            Fr = vf_all[idx, i, 4] * Erf
            Fz = vf_all[idx, i, 4] * Ezf
            vf_all[idx, i, 2] += Fr * dtNow / vf_all[idx, i, 5]
            vf_all[idx, i, 3] += Fz * dtNow / vf_all[idx, i, 5]

        # Update positions
        for i in range(Ni):
            if vf_all[idx, i, 5] >= 1e6:
                continue  # Skip ejected ions
            vf_all[idx, i, 0] += vf_all[idx, i, 2] * dtNow
            vf_all[idx, i, 1] += vf_all[idx, i, 3] * dtNow

            # Check bounds (example logic, adjust as needed)
            if vf_all[idx, i, 0] > Nr * dr or vf_all[idx, i, 0] < 0 or \
               vf_all[idx, i, 1] > Nz * dz or vf_all[idx, i, 1] < 0:
                vf_all[idx, i, 5] = 1e6  # Mark as ejected
                reorder += 2  # Increment reorder for ejection

        # Check for reordering
        for i in range(1, Ni):
            if vf_all[idx, i, 5] >= 1e6 or vf_all[idx, i - 1, 5] >= 1e6:
                continue  # Skip ejected ions
            if vf_all[idx, i, 1] > vf_all[idx, i - 1, 1]:
                reorder += 1
                break  # Exit loop early if reordering detected

        # Early termination if reorder detected
        if reorder > 0:
            break

    # Assign the reorder count
    reorder_all[idx] = reorder

# ============================
# Main Simulation Function
# ============================

def main():
    # Seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Simulation parameters
    T = 300  # Temperature in K
    collisionalMass = 2  # Mass of collisional particle (H2) in amu
    vMin = 50  # Minimum velocity in m/s
    vMax = 7000  # Maximum velocity in m/s
    numBins = 1000  # Number of bins for Boltzmann distribution
    boltzDist = Boltz(collisionalMass, T, vMin, vMax, numBins)
    v = np.linspace(vMin, vMax, numBins)
    angles = np.linspace(-np.pi / 2, np.pi / 2, 100)
    offsets = np.linspace(-2e-9, 2e-9, 200)
    max_hypotenuse = 1.5e-5  # Maximum displacement

    buffer_size = 10  # Buffer size for writing to file

    # Simulation configurations
    grid_sizes = [10001]  # Grid sizes (Nr = Nz = grid_size)
    ion_counts = [3]  # Number of ions (Ni)
    shot_sizes = [1000]  # Number of shots (parallel simulations)

    computation_times_file = "computation_times.json"
    formatted_table_file = "computation_times_table.txt"

    # Load existing computation times if available
    if os.path.exists(computation_times_file):
        with open(computation_times_file, 'r') as f:
            computation_times = json.load(f)
    else:
        computation_times = {}

    # Create directory for simulation results
    os.makedirs("simulation_results", exist_ok=True)

    for grid_size in grid_sizes:
        Nr = Nz = grid_size
        Nrmid = Nzmid = (Nr - 1) // 2
        dr = Dr / float(Nr)
        dz = Dz / float(Nz)

        # Generate RF and DC fields
        RF = makeRF0(m, q, wr, Nr, Nz, Nrmid, dr)
        DC = makeDC(m, q, wz, Nz, Nr, Nzmid, dz)

        # Transfer fields to device
        RF_device = cuda.to_device(RF)
        DC_device = cuda.to_device(DC)
        nullFields_device = cuda.to_device(np.zeros((Nr, Nz), dtype=np.float64))

        for Ni in ion_counts:
            for shots in shot_sizes:
                print(f"Starting simulation with grid size {grid_size}, ion count {Ni}, and shots {shots}")

                start_time = time.perf_counter()
                file_name = f"simulation_results/{Ni}ionSimulation_{grid_size}_{shots}shots.txt"
                data_buffer = []

                # Initialize all simulations
                vf_all = []
                rc_all = []
                zc_all = []
                vrc_all = []
                vzc_all = []
                Nt_all = []
                ion_collided_all = []  # To store ion collided indices

                for i in range(shots):
                    vf = makeVf(Ni, q, m, l, wz, offsetr, offsetz, vbumpr, vbumpz)
                    velocity = random.choices(v, weights=boltzDist)[0]
                    angle_choice = random.choice(angles)
                    offset_choice = random.choice(offsets)
                    ion_collided = random.randint(0, Ni - 1)

                    if velocity < 200:
                        Nt = 7_000_000  # Increased from 700,000
                    elif velocity < 1500:
                        Nt = 4_000_000  # Increased from 400,000
                    else:
                        Nt = 2_500_000  # Increased from 2,500,000

                    r = math.cos(angle_choice) * -max_hypotenuse
                    z = vf[ion_collided, 1] + math.sin(angle_choice) * max_hypotenuse + offset_choice
                    vz = -1 * velocity * math.sin(angle_choice)
                    vr = math.fabs(velocity * math.cos(angle_choice))

                    vf_all.append(vf)
                    rc_all.append(r)
                    zc_all.append(z)
                    vrc_all.append(vr)
                    vzc_all.append(vz)
                    Nt_all.append(Nt)
                    ion_collided_all.append(ion_collided)  # Store the collided ion index

                # Convert to numpy arrays with appropriate shapes
                vf_all = np.array(vf_all, dtype=np.float64)  # Shape: (shots, Ni, 7)
                rc_all = np.array(rc_all, dtype=np.float64)  # Shape: (shots,)
                zc_all = np.array(zc_all, dtype=np.float64)  # Shape: (shots,)
                vrc_all = np.array(vrc_all, dtype=np.float64)  # Shape: (shots,)
                vzc_all = np.array(vzc_all, dtype=np.float64)  # Shape: (shots,)
                Nt_all = np.array(Nt_all, dtype=np.int32)      # Shape: (shots,)
                ion_collided_all = np.array(ion_collided_all, dtype=np.int32)  # Shape: (shots,)

                # Transfer data to device
                vf_all_device = cuda.to_device(vf_all)
                rc_all_device = cuda.to_device(rc_all)
                zc_all_device = cuda.to_device(zc_all)
                vrc_all_device = cuda.to_device(vrc_all)
                vzc_all_device = cuda.to_device(vzc_all)
                Nt_all_device = cuda.to_device(Nt_all)
                reorder_all_device = cuda.device_array(shots, dtype=np.int32)

                # Adjust kernel launch parameters for better GPU utilization
                threads_per_block = 256
                blocks_per_grid = (shots + threads_per_block - 1) // threads_per_block  # Ensures full coverage

                # Launch the kernel
                mcCollision_kernel[blocks_per_grid, threads_per_block](
                    vf_all_device,
                    rc_all_device,
                    zc_all_device,
                    vrc_all_device,
                    vzc_all_device,
                    q,
                    mH2,
                    aH2,
                    Nt_all_device,
                    dtSmall,
                    RF_device,
                    DC_device,
                    Nr,
                    Nz,
                    dr,
                    dz,
                    dtLarge,
                    dtCollision,
                    nullFields_device,  # Pass nullFields_device
                    reorder_all_device
                )

                # Retrieve results
                reorder_all = reorder_all_device.copy_to_host()
                vf_all_updated = vf_all_device.copy_to_host()

                # Open file to write results
                with open(file_name, "w") as f:
                    f.write("axial_trapping_frequency(MHz)\tvelocity(m/s)\tion_collided_with\tangle(rads)\tcollision_offset(m)\treorder?(1=reorder, 2=ejection)\n")
                    for i in range(shots):
                        # Calculate the values
                        velocity = np.linalg.norm([vrc_all[i], vzc_all[i]])
                        angle_choice = math.atan2(-vzc_all[i], -vrc_all[i])
                        ion_collided = ion_collided_all[i]  # Use the actual collided ion index
                        offset_choice = zc_all[i] - vf_all_updated[i, ion_collided, 1]
                        reorder_val = reorder_all[i]

                        # Optional: Debugging prints
                        print(f"Shot {i+1}:")
                        print(f"  Velocity (m/s): {velocity}")
                        print(f"  Ion Collided With: {ion_collided + 1}")
                        print(f"  Angle (radians): {angle_choice}")
                        print(f"  Collision Offset (m): {offset_choice}")
                        print(f"  Reorder Value: {reorder_val}\n")

                        # Write to file
                        output = f"{wz / (2 * np.pi * 1e6)}\t{velocity:.2f}\t{ion_collided + 1}\t{angle_choice:.4f}\t{offset_choice:.4e}\t{int(reorder_val)}\n"
                        data_buffer.append(output)

                        if len(data_buffer) == buffer_size or i == shots - 1:
                            f.writelines(data_buffer)
                            f.flush()
                            data_buffer = []

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

                print(f"Completed simulation for grid size {grid_size}, ion count {Ni}, and shots {shots}. It took {timeTaken:.2f} seconds!")

    # ============================
    # Execution Entry Point
    # ============================

    if __name__ == "__main__":
        main()