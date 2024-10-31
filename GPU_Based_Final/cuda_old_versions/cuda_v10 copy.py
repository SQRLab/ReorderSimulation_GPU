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

# Constants
amu = 1.67e-27  # Atomic mass unit in kg
eps0 = 8.854e-12
qe = 1.6e-19  # Elementary charge in Coulombs

# Physical parameters
m = 40.0 * amu
q = 1.0 * qe
wr = 2 * np.pi * 3e6  # Radial angular frequency (SI units)
wz = 2 * np.pi * 1e6  # Axial angular frequency (SI units)
aH2 = 8e-31  # Polarizability of H2 in SI units
mH2 = 2.0 * amu  # Mass of H2 in kg

# Simulation parameters
Dr = 30001.5e-9
Dz = 90001.5e-9
dtSmall = 1e-12
dtCollision = 1e-16
dtLarge = 1e-10

sigmaV = 100e-6
dv = 20.0
vmax = 5000
l = 1
vbumpr = 0.00e0
vbumpz = -0.0e0
offsetz = 0.0e-7
offsetr = 0.0e-8

def ion_position_potential(x):
    N = len(x)
    return [
        x[m] - sum([1 / (x[m] - x[n]) ** 2 for n in range(m)]) +
        sum([1 / (x[m] - x[n]) ** 2 for n in range(m + 1, N)])
        for m in range(N)
    ]

def calcPositions(N):
    estimated_extreme = 0.481 * N ** 0.765  # Hardcoded, should work for at least up to 50 ions
    return fsolve(ion_position_potential, np.linspace(-estimated_extreme, estimated_extreme, N))

def lengthScale(ν, M=None, Z=None):
    if M is None:
        M = con.atomic_mass * 39.9626
    if Z is None:
        Z = 1
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
    vf = np.zeros((Ni, 7), dtype=np.float64)
    pos = calcPositions(Ni)
    lscale = lengthScale(wr)
    scaledPos = pos * lscale
    for i in range(Ni):
        vf[i, :] = [0.0e-6, -scaledPos[i], 0.0, 0.0, q, m, 0.0]
    if l < Ni:  # Prevent index out of bounds
        vf[l, 0] += offsetr
        vf[l, 1] += offsetz
        vf[l, 2] += vbumpr
        vf[l, 3] += vbumpz
    print(vf)
    return vf

def Boltz(m, T, vmin=0, vmax=5000, bins=100):
    amu = 1.67e-27  # Consistent with defined amu
    m = m * amu
    k = 1.386e-23  # Boltzmann constant
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
                    '''if time_taken is not None:
                        row.append(f"{time_taken:.0f}".rjust(8))
                    else:
                        row.append(" ".rjust(8))'''
                row.append(str(shot_size).rjust(8))
                f.write(" ".join(row) + "\n")
            f.write("\n")

    print(f"Formatted table has been written to {output_file}")

# CUDA device functions
'''@cuda.jit(device=True)
def ptovPos_cuda(pos, Nmid, dcell):
    return int(pos / dcell + Nmid + 0.5)'''

@cuda.jit(device=True)
def ptovPos_cuda(pos, Nmid, dcell):
    return pos / dcell + Nmid

@cuda.jit(device=True)
def minDists_cuda(vf, vc):
    rid2 = 1e6
    rii2 = 1e6
    vid2 = 1e6
    vii2 = 1e6
    Ni = vf.shape[0]
    Nc = vc.shape[0]
    for i in range(Ni):
        for j in range(i + 1, Ni):
            r = vf[i, 0] - vf[j, 0]
            z = vf[i, 1] - vf[j, 1]
            vr = vf[i, 2] - vf[j, 2]
            vz = vf[i, 3] - vf[j, 3]
            dist2 = r * r + z * z
            v2 = vr * vr + vz * vz
            if dist2 < rii2:
                vii2 = v2
                rii2 = dist2
        for j in range(Nc):
            r = vf[i, 0] - vc[j, 0]
            z = vf[i, 1] - vc[j, 1]
            vr = vf[i, 2] - vc[j, 2]
            vz = vf[i, 3] - vc[j, 3]
            dist2 = r * r + z * z
            v2 = vr * vr + vz * vz
            if dist2 < rid2:
                vid2 = v2
                rid2 = dist2
    return math.sqrt(rid2), math.sqrt(rii2), math.sqrt(vid2), math.sqrt(vii2)

@cuda.jit(device=True)
def collisionMode_cuda(rii, rid, a, e=0.3):
    numerator = a * rii * rii
    denominator = rid ** 5
    return numerator / denominator > e

'''@cuda.jit(device=True)
def updatePoss_cuda(vf, dr, dz, dt, Nr, Nz, Nrmid, Nzmid):
    for i in range(vf.shape[0]):
        vf[i, 0] += vf[i, 2] * dt
        vf[i, 1] += vf[i, 3] * dt
        rCell = ptovPos_cuda(vf[i, 0], Nrmid, dr)
        zCell = ptovPos_cuda(vf[i, 1], Nzmid, dz)

        # Use softer boundary conditions
        if rCell >= Nr - 1 or rCell <= 0 or zCell >= Nz - 1 or zCell <= 0:
            # Reflect the particle back into the simulation area
            if rCell >= Nr - 1:
                vf[i, 0] = 2 * (Nr - 1) * dr - vf[i, 0]
                vf[i, 2] *= -0.9  # Reduce velocity slightly
            elif rCell <= 0:
                vf[i, 0] = -vf[i, 0]
                vf[i, 2] *= -0.9
            if zCell >= Nz - 1:
                vf[i, 1] = 2 * (Nz - 1) * dz - vf[i, 1]
                vf[i, 3] *= -0.9
            elif zCell <= 0:
                vf[i, 1] = -vf[i, 1]
                vf[i, 3] *= -0.9

            # Count boundary hits, but don't eject immediately
            vf[i, 6] += 1

            # Apply a velocity damping factor
            damping_factor = 0.99
            vf[i, 2] *= damping_factor
            vf[i, 3] *= damping_factor

            # Only eject if boundary hits exceed a higher threshold
            if vf[i, 6] > 50:
                vf[i, 5] = 1e6  # Set mass to large value to trigger ejection'''

@cuda.jit(device=True)
def updatePoss_cuda(vf, dr, dz, dt, Nr, Nz, Nrmid, Nzmid):
    for i in range(vf.shape[0]):
        vf[i, 0] += vf[i, 2] * dt
        vf[i, 1] += vf[i, 3] * dt
        rCell = ptovPos_cuda(vf[i, 0], Nrmid, dr)
        zCell = ptovPos_cuda(vf[i, 1], Nzmid, dz)
        if rCell > Nr - 2 or rCell < 1 or zCell > Nz - 2 or zCell < 1:
            for j in range(7):
                vf[i, j] = 0.0
            vf[i, 0] = 2.0
            vf[i, 1] = 2.0
            vf[i, 2] = 0.0
            vf[i, 3] = 0.0
            vf[i, 5] = 1e6
            vf[i, 6] += 1  # Increment boundary hit count


@cuda.jit(device=True)
def updateVels_cuda(vf, Erf, Ezf, dt):
    for i in range(vf.shape[0]):
        Fr = vf[i, 4] * Erf[i]
        Fz = vf[i, 4] * Ezf[i]
        vf[i, 2] += Fr * dt / (vf[i, 5])
        vf[i, 3] += Fz * dt / (vf[i, 5])

@cuda.jit(device=True)
def solveFields_cuda(vf, ErDC, EzDC, ErAC, EzAC, Nrmid, Nzmid, Ni, dr, dz, Erf2, Ezf2):
    eps0 = 8.854e-12
    C1 = 4 * math.pi * eps0  # SI units
    Nr = ErDC.shape[0]
    Nz = ErDC.shape[1]
    
    for i in range(Ni):
        Erf2[i] = 0.0
        Ezf2[i] = 0.0

    for i in range(Ni):
        jCell = int(ptovPos_cuda(vf[i, 0], Nrmid, dr) + 0.5)
        kCell = int(ptovPos_cuda(vf[i, 1], Nzmid, dz) + 0.5)
        
        # Bounds checking
        if jCell > Nr - 2 or jCell < 1 or kCell > Nz - 2 or kCell < 1:
            continue  # Skip this ion if out of bounds
        
        # Add background trap fields
        Erf2[i] += ErDC[jCell, kCell] + ErAC[jCell, kCell]
        Ezf2[i] += EzDC[jCell, kCell] + EzAC[jCell, kCell]
        
        # Add contributions from other ions
        for j in range(Ni):
            if j != i:
                rdist = vf[j, 0] - vf[i, 0]
                zdist = vf[j, 1] - vf[i, 1]
                sqDist = rdist**2 + zdist**2
                dist = math.sqrt(sqDist)
                projR = rdist / dist
                projZ = zdist / dist
                
                Erf2[i] += -projR * vf[j, 4] / (C1 * sqDist)
                Ezf2[i] += -projZ * vf[j, 4] / (C1 * sqDist)

@cuda.jit(device=True)
def collisionParticlesFields_cuda(vf, vc, Ni, ErDC, EzDC, ErAC, EzAC, dr, dz, dtNow, Nrmid, Nzmid, Nr, Nz, Erfi, Ezfi):
    eps0 = 8.854e-12
    C1 = 4 * math.pi * eps0
    Nc = vc.shape[0]
    # We can assume Nc = 1

    small = 1e-20  # Small value to prevent division by zero

    # For each collisional particle
    for i in range(Nc):
        Erfc = cuda.local.array(2, dtype=float64)  # [0]: field, [1]: gradient
        Ezfc = cuda.local.array(2, dtype=float64)

        Erfc[0] = 0.0  # Field
        Erfc[1] = 0.0  # Gradient
        Ezfc[0] = 0.0
        Ezfc[1] = 0.0

        # Compute cell indices
        jCell = int(ptovPos_cuda(vc[i, 0], Nrmid, dr) + 0.5)
        kCell = int(ptovPos_cuda(vc[i, 1], Nzmid, dz) + 0.5)

        if 1 <= jCell < Nr - 1 and 1 <= kCell < Nz - 1:
            # Background fields
            Erfc[0] += ErDC[jCell, kCell] + ErAC[jCell, kCell]
            Ezfc[0] += EzDC[jCell, kCell] + EzAC[jCell, kCell]

            # Field gradients
            Erfc[1] += ((ErDC[jCell + 1, kCell] + ErAC[jCell + 1, kCell]) - (ErDC[jCell - 1, kCell] + ErAC[jCell - 1, kCell])) / (2 * dr)
            Ezfc[1] += ((EzDC[jCell, kCell + 1] + EzAC[jCell, kCell + 1]) - (EzDC[jCell, kCell - 1] + EzAC[jCell, kCell - 1])) / (2 * dz)

            for j in range(Ni):
                rdist = vf[j, 0] - vc[i, 0]
                zdist = vf[j, 1] - vc[i, 1]
                sqDist_j = rdist * rdist + zdist * zdist + small
                dist_j = math.sqrt(sqDist_j)
                projR_j = rdist / dist_j
                projZ_j = zdist / dist_j

                Erfc[0] += -projR_j * vf[j, 4] / (C1 * sqDist_j)
                Ezfc[0] += -projZ_j * vf[j, 4] / (C1 * sqDist_j)

                Erfc[1] += 2 * projR_j * vf[j, 4] / (C1 * sqDist_j * dist_j)
                Ezfc[1] += 2 * projZ_j * vf[j, 4] / (C1 * sqDist_j * dist_j)

            # Induced dipole calculations
            if vc[i, 6] != 0.0:
                pR = -2 * math.pi * eps0 * vc[i, 6] * Erfc[0]
                pZ = -2 * math.pi * eps0 * vc[i, 6] * Ezfc[0]

                Fr = math.fabs(pR) * Erfc[1]
                Fz = math.fabs(pZ) * Ezfc[1]

                vc[i, 2] += Fr * dtNow / vc[i, 5]
                vc[i, 3] += Fz * dtNow / vc[i, 5]

                # Apply fields from collision particles to ions
                for j in range(Ni):
                    rdist = vf[j, 0] - vc[i, 0]
                    zdist = vf[j, 1] - vc[i, 1]
                    dist_j = math.sqrt(rdist * rdist + zdist * zdist + small)

                    Rhatr_j = rdist / dist_j
                    Rhatz_j = zdist / dist_j

                    Erfi[j] += -math.fabs(pR) * (2 * Rhatr_j) / (C1 * dist_j ** 3)
                    Ezfi[j] += -math.fabs(pZ) * (2 * Rhatz_j) / (C1 * dist_j ** 3)

MAX_IONS = 3

@cuda.jit(device=True)
def mcCollision_cuda(vf, rc, zc, vrc, vzc, qc, mc, ac, Nt, dtSmall, RF, DC, Nr, Nz, dr, dz, dtLarge, dtCollision, nullFields, reorder, sum_masses):
    # Initialize local variables
    reorder[0] = 0
    sum_masses[0] = 0.0
    Nrmid = (Nr - 1) / 2.0
    Nzmid = (Nz - 1) / 2.0
    Ni = vf.shape[0]
    Nc = 1
    vc = cuda.local.array((1, 7), dtype=float64)
    vc[0, 0] = rc
    vc[0, 1] = zc
    vc[0, 2] = vrc
    vc[0, 3] = vzc
    vc[0, 4] = qc
    vc[0, 5] = mc
    vc[0, 6] = ac

    dtNow = dtSmall
    Erfi = cuda.local.array((MAX_IONS,), dtype=float64)
    Ezfi = cuda.local.array((MAX_IONS,), dtype=float64)

    # Initialize history arrays are handled in the kernel
    for i in range(Nt):
        rid, rii, vid, vii = minDists_cuda(vf, vc)
        collision = collisionMode_cuda(rii, rid, vc[0, 6], 0.1)
        if collision:
            dtNow = rid * 0.01 / (5 * vid)
        else:
            dtNow = dtSmall
        if dtNow < dtCollision:
            dtNow = dtCollision

        # Initialize electric fields
        for j in range(Ni):
            Erfi[j] = 0.0
            Ezfi[j] = 0.0

        solveFields_cuda(vf, nullFields, DC, RF, nullFields, Nrmid, Nzmid, Ni, dr, dz, Erfi, Ezfi)

        if vc[0, 5] < 1e6:
            collisionParticlesFields_cuda(vf, vc, Ni, nullFields, nullFields, DC, RF, dr, dz, dtNow, Nrmid, Nzmid, Nr, Nz, Erfi, Ezfi)
            updatePoss_cuda(vc, dr, dz, dtNow, Nr, Nz, Nrmid, Nzmid)
        else:
            dtNow = dtLarge

        updateVels_cuda(vf, Erfi, Ezfi, dtNow)
        updatePoss_cuda(vf, dr, dz, dtNow, Nr, Nz, Nrmid, Nzmid)

        # Log Erfi and Ezfi after updatePoss_cuda
        '''for j in range(Ni):
            Erfi_history[i * Ni + j] = Erfi[j]
            Ezfi_history[i * Ni + j] = Ezfi[j]
            # Store positions and velocities
            pos_history[i * Ni * 2 + j * 2] = vf[j, 0]      # r position
            pos_history[i * Ni * 2 + j * 2 + 1] = vf[j, 1]  # z position
            vel_history[i * Ni * 2 + j * 2] = vf[j, 2]      # vr velocity
            vel_history[i * Ni * 2 + j * 2 + 1] = vf[j, 3]  # vz velocity'''

        # Check for ion ejection
        sum_masses[0] = 0.0
        for j in range(Ni):
            sum_masses[0] += vf[j, 5]
        if sum_masses[0] > 1e5:
            reorder[0] += 2
            break

        # Check for reordering
        for j in range(1, Ni):
            if vf[j, 1] > vf[j - 1, 1]:
                reorder[0] += 1
                Nt = min(Nt, i + 1000)
                break

        if reorder[0] != 0:
            break

@cuda.jit
def mcCollision_kernel(vf_all, rc_all, zc_all, vrc_all, vzc_all, qc, mc, ac, Nt_all, dtSmall, RF, DC, Nr, Nz, dr, dz, dtLarge, dtCollision, nullFields, reorder_all, sum_masses_all):
    idx = cuda.grid(1)
    if idx < vf_all.shape[0]:
        # Load vf for this simulation
        vf = cuda.local.array((MAX_IONS, 7), dtype=float64)
        for i in range(MAX_IONS):
            for j in range(7):
                vf[i, j] = vf_all[idx, i, j]
        
        rc = rc_all[idx]
        zc = zc_all[idx]
        vrc = vrc_all[idx]
        vzc = vzc_all[idx]
        Nt = Nt_all[idx]
        
        # Initialize output variables
        reorder = cuda.local.array((1,), dtype=int32)
        sum_masses = cuda.local.array((1,), dtype=float64)
        reorder[0] = 0
        sum_masses[0] = 0.0

        # Initialize history arrays (fixed size)
        '''pos_history = cuda.local.array((60,), dtype=float64)  # Nt=10, MAX_IONS=3
        vel_history = cuda.local.array((60,), dtype=float64)
        Erfi_history = cuda.local.array((30,), dtype=float64)  # Nt=10, MAX_IONS=3
        Ezfi_history = cuda.local.array((30,), dtype=float64)'''
        
        # Call the device function
        mcCollision_cuda(vf, rc, zc, vrc, vzc, qc, mc, ac, Nt, dtSmall, RF, DC, Nr, Nz, dr, dz, dtLarge, dtCollision,
                        nullFields, reorder, sum_masses)
        
        # Store the results back to global memory
        reorder_all[idx] = reorder[0]
        sum_masses_all[idx] = sum_masses[0]
        
        for i in range(MAX_IONS):
            for j in range(7):
                vf_all[idx, i, j] = vf[i, j]
        
        # Save Erfi and Ezfi histories
        '''for i in range(30):  # Erfi_history and Ezfi_history are size 30
            Erfi_history_all[idx, i] = Erfi_history[i]
            Ezfi_history_all[idx, i] = Ezfi_history[i]
        
        # Save positions and velocities histories
        for i in range(60):  # pos_history and vel_history are size 60
            pos_history_all[idx, i // 2, i % 2] = pos_history[i]
            vel_history_all[idx, i // 2, i % 2] = vel_history[i]'''

max_hypotenuse = 1.5e-5

# Main simulation code
def main():
    grid_size = 10001
    Ni = 3
    shots = 100

    Nr = Nz = grid_size
    Nrmid = Nzmid = (Nr - 1) // 2
    dr = Dr / float(Nr)
    dz = Dz / float(Nz)

    RF = makeRF0(m, q, wr, Nr, Nz, Nrmid, dr)
    DC = makeDC(m, q, wz, Nz, Nr, Nzmid, dz)

    print(f"Starting simulation with grid size {grid_size}, ion count {Ni}, and shots {shots}")

    start_time = time.perf_counter()

    # Allocate arrays for all shots
    vf_all = np.zeros((shots, Ni, 7), dtype=np.float64)
    rc_all = np.zeros(shots, dtype=np.float64)
    zc_all = np.zeros(shots, dtype=np.float64)
    vrc_all = np.zeros(shots, dtype=np.float64)
    vzc_all = np.zeros(shots, dtype=np.float64)
    Nt_all = np.zeros(shots, dtype=np.int32)

    T = 300
    collisionalMass = 2
    vMin = 50
    vMax = 7000
    numBins = 1000
    boltzDist = Boltz(collisionalMass, T, vMin, vMax, numBins)
    v = np.linspace(vMin, vMax, numBins)
    angles = np.linspace(-np.pi/2, np.pi/2, 100)
    offsets = np.linspace(-2e-9, 2e-9, 200)
    Nt = 10000000  # Number of timesteps

    for i in range(shots):
        # Generate initial ion positions and velocities
        vf = makeVf(Ni, q, m, l, wz, offsetr, offsetz, vbumpr, vbumpz)

        # Random initial conditions for colliding particle
        velocity = random.choices(v, weights=boltzDist)[0]
        angle_choice = random.choice(angles)
        offset_choice = random.choice(offsets)
        ion_collided = random.randint(0, Ni - 1)

        r = math.cos(angle_choice) * -max_hypotenuse
        z = vf[ion_collided, 1] + math.sin(angle_choice) * max_hypotenuse + offset_choice
        vz = -1 * velocity * math.sin(angle_choice)
        vr = math.fabs(velocity * math.cos(angle_choice))

        # Store initial conditions for this shot
        vf_all[i, :, :] = vf
        rc_all[i] = r
        zc_all[i] = z
        vrc_all[i] = vr
        vzc_all[i] = vz
        Nt_all[i] = Nt

    # Transfer data to device
    vf_device = cuda.to_device(vf_all)
    rc_device = cuda.to_device(rc_all)
    zc_device = cuda.to_device(zc_all)
    vrc_device = cuda.to_device(vrc_all)
    vzc_device = cuda.to_device(vzc_all)
    Nt_device = cuda.to_device(Nt_all)
    reorder_device = cuda.device_array(shots, dtype=np.int32)
    sum_masses_device = cuda.device_array(shots, dtype=np.float64)

    RF_device = cuda.to_device(RF)
    DC_device = cuda.to_device(DC)
    nullFields_device = cuda.to_device(np.zeros((Nr, Nz), dtype=np.float64))

    threads_per_block = 1
    blocks = shots  # Number of simulations

    mcCollision_kernel[blocks, threads_per_block](
        vf_device, rc_device, zc_device, vrc_device, vzc_device,
        q, mH2, aH2, Nt_device, dtSmall, RF_device, DC_device, Nr, Nz, dr, dz,
        dtLarge, dtCollision, nullFields_device, reorder_device, sum_masses_device
    )

    # Copy the results back to the host
    reorder = reorder_device.copy_to_host()
    sum_masses = sum_masses_device.copy_to_host()

    print(f"Final conditions:")
    for i in range(shots):
        print(f"Simulation {i+1}:")
        print(f"  Reorder = {reorder[i]}")
        print(f"  Sum of Masses = {sum_masses[i]:.2e}")

    finish_time = time.perf_counter()
    timeTaken = finish_time - start_time

    print(f"Simulation completed in {timeTaken:.2f} seconds")

if __name__ == "__main__":
    main()