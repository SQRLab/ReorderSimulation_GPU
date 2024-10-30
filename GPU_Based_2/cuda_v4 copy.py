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
                    if time_taken is not None:
                        row.append(f"{time_taken:.0f}".rjust(8))
                    else:
                        row.append(" ".rjust(8))
                row.append(str(shot_size).rjust(8))
                f.write(" ".join(row) + "\n")
            f.write("\n")

    print(f"Formatted table has been written to {output_file}")

# CUDA device functions
@cuda.jit(device=True)
def ptovPos_cuda(pos, Nmid, dcell):
    idx = (pos / dcell) + Nmid
    return idx

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

@cuda.jit(device=True)
def updatePoss_cuda(vf, dr, dz, dt, Nr, Nz, Nrmid, Nzmid):
    for i in range(vf.shape[0]):
        vf[i, 0] += vf[i, 2] * dt
        vf[i, 1] += vf[i, 3] * dt
        rCell = ptovPos_cuda(vf[i, 0], Nrmid, dr)
        zCell = ptovPos_cuda(vf[i, 1], Nzmid, dz)
        if rCell > Nr - 2 or rCell < 1:
            vf[i, 0] = 0.0
            vf[i, 1] = 0.0
            vf[i, 2] = 0.0
            vf[i, 3] = 0.0
            vf[i, 5] = 0.0

            vf[i, 0] = 2.0
            vf[i, 1] = 2.0
            vf[i, 2] = 0.0
            vf[i, 3] = 0.0
            vf[i, 5] = 1e6

        elif zCell > Nz - 2 or zCell < 1:
            vf[i, 0] = 0.0
            vf[i, 1] = 0.0
            vf[i, 2] = 0.0
            vf[i, 3] = 0.0
            vf[i, 5] = 0.0
            
            vf[i, 0] = 2.0
            vf[i, 1] = 2.0
            vf[i, 2] = 0.0
            vf[i, 3] = 0.0
            vf[i, 5] = 1e6

@cuda.jit(device=True)
def updateVels_cuda(vf, Erf, Ezf, dt):
    for i in range(vf.shape[0]):
        Fr = vf[i, 4] * Erf[i]
        Fz = vf[i, 4] * Ezf[i]
        vf[i, 2] += Fr * dt / (vf[i, 5])
        vf[i, 3] += Fz * dt / (vf[i, 5])

@cuda.jit(device=True)
def solveFields_cuda(vf, ErDC, EzDC, ErAC, EzAC, Nrmid, Nzmid, Ni, dr, dz, Erf2, Ezf2, Nr, Nz):
    eps0 = 8.854e-12
    C1 = 4 * math.pi * eps0
    for i in range(Ni):
        jCell = int(ptovPos_cuda(vf[i, 0], Nrmid, dr) + 0.5)
        kCell = int(ptovPos_cuda(vf[i, 1], Nzmid, dz) + 0.5)

        # Boundary checks
        if jCell < 0:
            jCell = 0
        elif jCell >= Nr:
            jCell = Nr - 1

        if kCell < 0:
            kCell = 0
        elif kCell >= Nz:
            kCell = Nz - 1

        Erf2[i] = ErDC[jCell, kCell] + ErAC[jCell, kCell]
        Ezf2[i] = EzDC[jCell, kCell] + EzAC[jCell, kCell]

        for j in range(Ni):
            if j != i:
                rdist = vf[j, 0] - vf[i, 0]
                zdist = vf[j, 1] - vf[i, 1]
                sqDist = rdist * rdist + zdist * zdist
                dist = math.sqrt(sqDist)
                projR = rdist / dist
                projZ = zdist / dist
                Erf2[i] += -projR * vf[j, 4] / (C1 * sqDist)
                Ezf2[i] += -projZ * vf[j, 4] / (C1 * sqDist)

@cuda.jit(device=True)
def collisionParticlesFields_cuda(vf, vc, Ni, ErDC, EzDC, ErAC, EzAC, dr, dz, dtNow, Nrmid, Nzmid, Nr, Nz, Erfi, Ezfi):
    eps0 = 8.854e-12
    Nc = vc.shape[0]
    C1 = 4 * math.pi * eps0
    for i in range(Ni):
        Erfi[i] = 0.0
        Ezfi[i] = 0.0
    for i in range(Nc):
        jCell = int(ptovPos_cuda(vc[i, 0], Nrmid, dr) + 0.5)
        kCell = int(ptovPos_cuda(vc[i, 1], Nzmid, dz) + 0.5)

        # Boundary checks
        if jCell < 1:
            jCell = 1
        elif jCell > Nr - 2:
            jCell = Nr - 2

        if kCell < 1:
            kCell = 1
        elif kCell > Nz - 2:
            kCell = Nz - 2

        Erfc0 = ErDC[jCell, kCell] + ErAC[jCell, kCell]
        Ezfc0 = EzDC[jCell, kCell] + EzAC[jCell, kCell]
        Erfc1 = ((ErDC[jCell + 1, kCell] + ErAC[jCell + 1, kCell]) - (ErDC[jCell - 1, kCell] + ErAC[jCell - 1, kCell])) / dr
        Ezfc1 = ((EzDC[jCell, kCell + 1] + EzAC[jCell, kCell + 1]) - (EzDC[jCell, kCell - 1] + EzAC[jCell, kCell - 1])) / dz
        pR = -2 * math.pi * eps0 * vc[i, 6] * Erfc0
        pZ = -2 * math.pi * eps0 * vc[i, 6] * Ezfc0
        Fr = math.fabs(pR) * Erfc1
        Fz = math.fabs(pZ) * Ezfc1
        vc[i, 2] += Fr * dtNow / vc[i, 5]
        vc[i, 3] += Fz * dtNow / vc[i, 5]
        for j in range(Ni):
            rdist = vf[j, 0] - vc[i, 0]
            zdist = vf[j, 1] - vc[i, 1]
            sqDist = rdist * rdist + zdist * zdist
            dist = math.sqrt(sqDist)
            Rhatr = rdist / dist
            Rhatz = zdist / dist
            Erfi[j] += -math.fabs(pR) * (2 * Rhatr) / (C1 * dist ** 3)
            Ezfi[j] += -math.fabs(pZ) * (2 * Rhatz) / (C1 * dist ** 3)

    # Update positions of the collision particles
    updatePoss_cuda(vc, dr, dz, dtNow, Nr, Nz, Nrmid, Nzmid)

'''@cuda.jit
def mcCollision_kernel(vf_all, rc_all, zc_all, vrc_all, vzc_all, qc, mc, ac, Nt_all, dtSmall, RF, DC, Nr, Nz, dr, dz, dtLarge, dtCollision, nullFields, reorder_all, collision_counts, ejection_counts, iteration_counts):
    idx = cuda.grid(1)
    if idx < vf_all.shape[0]:
        vf = vf_all[idx]
        rc = rc_all[idx]
        zc = zc_all[idx]
        vrc = vrc_all[idx]
        vzc = vzc_all[idx]
        Nt = Nt_all[idx]
        reorder, collisions, ejections, iterations = mcCollision_cuda(vf, rc, zc, vrc, vzc, qc, mc, ac, Nt, dtSmall, RF, DC, Nr, Nz, dr, dz, dtLarge, dtCollision, nullFields)
        reorder_all[idx] = reorder
        collision_counts[idx] = collisions
        ejection_counts[idx] = ejections
        iteration_counts[idx] = iterations'''

@cuda.jit
def mcCollision_kernel(vf_all, rc_all, zc_all, vrc_all, vzc_all, qc, mc, ac, Nt_all, dtSmall, RF, DC, Nr, Nz, dr, dz, dtLarge, dtCollision, reorder_all):
    idx = cuda.grid(1)
    if idx < vf_all.shape[0]:
        vf = vf_all[idx]
        rc = rc_all[idx]
        zc = zc_all[idx]
        vrc = vrc_all[idx]
        vzc = vzc_all[idx]
        Nt = Nt_all[idx]
        reorder = mcCollision_cuda(vf, rc, zc, vrc, vzc, qc, mc, ac, Nt, dtSmall, RF, DC, Nr, Nz, dr, dz, dtLarge, dtCollision)
        reorder_all[idx] = reorder

@cuda.jit(device=True)
def mcCollision_cuda(vf, rc, zc, vrc, vzc, qc, mc, ac, Nt, dtSmall, RF, DC, Nr, Nz, dr, dz, dtLarge, dtCollision):
    reorder = 0
    Nrmid = (Nr - 1) / 2
    Nzmid = (Nz - 1) / 2
    Ni = vf.shape[0]
    eid = 0.01
    
    # Initialize vc
    vc = cuda.local.array((1, 7), dtype=float64)
    vc[0] = cuda.local.array((7,), dtype=float64)
    vc[0][0] = rc
    vc[0][1] = zc
    vc[0][2] = vrc
    vc[0][3] = vzc
    vc[0][4] = qc
    vc[0][5] = mc
    vc[0][6] = ac

    dtNow = dtSmall
    nullFields = cuda.local.array((Nr, Nz), dtype=float64)
    for i in range(Nr):
        for j in range(Nz):
            nullFields[i, j] = 0.0

    for i in range(Nt):
        rid, rii, vid, vii = minDists_cuda(vf, vc)
        collision = collisionMode_cuda(rii, rid, vc[0][6], 0.1)
        
        if collision:
            dtNow = rid * eid / (5 * vid)
        else:
            dtNow = dtSmall
        
        if dtNow < dtCollision:
            dtNow = dtCollision

        Erfi, Ezfi = solveFields_cuda(vf, nullFields, DC, RF, nullFields, Nrmid, Nzmid, Ni, dr, dz)

        if vc[0][5] < 1e6:
            vc, Erfic, Ezfic, _, _ = collisionParticlesFields_cuda(vf, vc, Ni, nullFields, DC, RF, nullFields, dr, dz, dtNow, Nrmid, Nzmid)
            vc = updatePoss_cuda(vc, dr, dz, dtNow, Nr, Nz, Nrmid, Nzmid)
            for j in range(Ni):
                Erfi[j] += Erfic[j]
                Ezfi[j] += Ezfic[j]
        else:
            dtNow = dtLarge

        vf = updateVels_cuda(vf, Erfi, Ezfi, dtNow)
        vf = updatePoss_cuda(vf, dr, dz, dtNow, Nr, Nz, Nrmid, Nzmid)

        if cuda.threadfence():  # Ensure global visibility of shared memory updates
            if cuda.atomic.add(cuda.shared.array(1, dtype=int32), 0, 0) == 1:
                break

        sum_masses = 0.0
        for j in range(Ni):
            sum_masses += vf[j][5]
        
        if sum_masses > 1e5:
            reorder = 2
            break

        for j in range(1, Ni):
            if vf[j][1] > vf[j-1][1]:
                reorder = 1
                Nt = i + 1000
                break

    return reorder

'''@cuda.jit(device=True)
def mcCollision_cuda(vf, rc, zc, vrc, vzc, qc, mc, ac, Nt, dtSmall, RF, DC, Nr, Nz, dr, dz, dtLarge, dtCollision, nullFields):
    reorder = 0
    collision_count = 0
    ejection_count = 0
    
    Nrmid = (Nr - 1) / 2
    Nzmid = (Nz - 1) / 2
    Ni = vf.shape[0]
    Nc = 1
    # Initialize local array
    vc = cuda.local.array((1, 7), dtype=float64)
    vc[0, 0] = rc
    vc[0, 1] = zc
    vc[0, 2] = vrc
    vc[0, 3] = vzc
    vc[0, 4] = qc
    vc[0, 5] = mc
    vc[0, 6] = ac

    dtNow = dtSmall
    Erfi = cuda.local.array(100, dtype=float64)
    Ezfi = cuda.local.array(100, dtype=float64)

    max_iterations = Nt + 1000  # Set a fixed large number of iterations
    for i in range(max_iterations):
        rid, rii, vid, vii = minDists_cuda(vf, vc)
        collision = collisionMode_cuda(rii, rid, vc[0, 6], 0.1)
        if collision:
            dtNow = rid * 0.01 / (5 * vid)
            collision_count += 1
        else:
            dtNow = dtSmall
        if dtNow < dtCollision:
            dtNow = dtCollision
        solveFields_cuda(vf, nullFields, nullFields, DC, RF, Nrmid, Nzmid, Ni, dr, dz, Erfi, Ezfi, Nr, Nz)

        if vc[0, 5] < 1e6:
            Erfic = cuda.local.array(100, dtype=float64)
            Ezfic = cuda.local.array(100, dtype=float64)
            collisionParticlesFields_cuda(vf, vc, Ni, nullFields, nullFields, DC, RF, dr, dz, dtNow, Nrmid, Nzmid, Nr, Nz, Erfic, Ezfic)
            for j in range(Ni):
                Erfi[j] += Erfic[j]
                Ezfi[j] += Ezfic[j]
            updatePoss_cuda(vc, dr, dz, dtNow, Nr, Nz, Nrmid, Nzmid)
        else:
            dtNow = dtLarge

        updateVels_cuda(vf, Erfi, Ezfi, dtNow)
        updatePoss_cuda(vf, dr, dz, dtNow, Nr, Nz, Nrmid, Nzmid)

        sum_masses = 0.0
        for j in range(Ni):
            sum_masses += vf[j, 5]
        if sum_masses > 1e5:
            reorder += 2
            ejection_count += 1
            break  # Break out of the loop if an ion is ejected

        for j in range(1, Ni):
            if vf[j, 1] > vf[j - 1, 1]:
                reorder += 1

    return reorder, collision_count, ejection_count, i  # Return the iteration count as well'''

@cuda.jit
def debug_kernel(vf, reorder_flag, ejection_flag):
    shot_idx = cuda.grid(1)
    if shot_idx < vf.shape[0]:
        sum_masses = 0.0
        for i in range(vf.shape[1]):  # Iterate over ions
            sum_masses += vf[shot_idx, i, 5]
        
        if sum_masses > 1e5:
            ejection_flag[shot_idx] = 1
        
        for i in range(1, vf.shape[1]):  # Check for reordering
            if vf[shot_idx, i, 1] > vf[shot_idx, i-1, 1]:
                reorder_flag[shot_idx] = 1
                break


# Main simulation code
def main():
    T = 300
    collisionalMass = 2
    vMin = 50
    vMax = 7000
    numBins = 1000
    boltzDist = Boltz(collisionalMass, T, vMin, vMax, numBins)
    v = np.linspace(vMin, vMax, numBins)
    angles = np.linspace(-np.pi / 2, np.pi / 2, 100)
    offsets = np.linspace(-2e-9, 2e-9, 200)
    max_hypotenuse = 1.5e-5

    buffer_size = 10

    grid_sizes = [10001]
    ion_counts = [2, 3]
    shot_sizes = [10]

    computation_times_file = "computation_times.json"
    formatted_table_file = "computation_times_table.txt"

    if os.path.exists(computation_times_file):
        with open(computation_times_file, 'r') as f:
            computation_times = json.load(f)
    else:
        computation_times = {}

    os.makedirs("simulation_results_copy", exist_ok=True)

    # In your main function, replace the existing simulation loop with this:

    for grid_size in grid_sizes:
        Nr = Nz = grid_size
        Nrmid = Nzmid = (Nr-1)/2
        dr = Dr/float(Nr)
        dz = Dz/float(Nz)

        RF = makeRF0(m, q, wr, Nr, Nz, Nrmid, dr)
        DC = makeDC(m, q, wz, Nz, Nr, Nzmid, dz)

        RF_device = cuda.to_device(RF)
        DC_device = cuda.to_device(DC)

        for Ni in ion_counts:
            for shots in shot_sizes:
                print(f"Starting simulation with grid size {grid_size}, ion count {Ni}, and shots {shots}")

                start_time = time.perf_counter()
                file_name = f"simulation_results/{Ni}ionSimulation_{grid_size}_{shots}shots.txt"

                vf_all = []
                rc_all = []
                zc_all = []
                vrc_all = []
                vzc_all = []
                Nt_all = []

                for i in range(shots):
                    vf = makeVf(Ni, q, m, l, wz, offsetr, offsetz, vbumpr, vbumpz)
                    velocity = random.choices(v, weights=boltzDist)[0]
                    angle_choice = random.choice(angles)
                    offset_choice = random.choice(offsets)

                    if velocity < 200:
                        Nt = 700000
                    elif velocity < 1500:
                        Nt = 400000
                    else:
                        Nt = 250000

                    r = -np.cos(angle_choice)*max_hypotenuse
                    z = vf[random.randint(0, Ni-1), 1] + np.sin(angle_choice)*max_hypotenuse + offset_choice
                    vz = -1*velocity*np.sin(angle_choice)
                    vr = np.abs(velocity*np.cos(angle_choice))

                    vf_all.append(vf)
                    rc_all.append(r)
                    zc_all.append(z)
                    vrc_all.append(vr)
                    vzc_all.append(vz)
                    Nt_all.append(Nt)

                vf_all = np.array(vf_all, dtype=np.float64)
                rc_all = np.array(rc_all, dtype=np.float64)
                zc_all = np.array(zc_all, dtype=np.float64)
                vrc_all = np.array(vrc_all, dtype=np.float64)
                vzc_all = np.array(vzc_all, dtype=np.float64)
                Nt_all = np.array(Nt_all, dtype=np.int32)

                vf_all_device = cuda.to_device(vf_all)
                rc_all_device = cuda.to_device(rc_all)
                zc_all_device = cuda.to_device(zc_all)
                vrc_all_device = cuda.to_device(vrc_all)
                vzc_all_device = cuda.to_device(vzc_all)
                Nt_all_device = cuda.to_device(Nt_all)
                reorder_all_device = cuda.device_array(shots, dtype=np.int32)

                threads_per_block = 128
                blocks_per_grid = (shots + (threads_per_block - 1)) // threads_per_block

                mcCollision_kernel[blocks_per_grid, threads_per_block](
                vf_all_device, rc_all_device, zc_all_device, vrc_all_device, vzc_all_device,
                q, mH2, aH2, Nt_all_device, dtSmall, RF_device, DC_device, Nr, Nz, dr, dz,
                dtLarge, dtCollision, reorder_all_device
                )

                reorder_all = reorder_all_device.copy_to_host()

                with open(file_name, "w") as f:
                    f.write("axial trapping frequency (MHz) \t velocity(m/s) \t ion collided with \t angle(rads) \t collision offset(m) \t reorder? (1 is reorder 2 is ejection) \n")
                    for i in range(shots):
                        output = f"{wz/(2*np.pi*1e6):.6f}\t{np.sqrt(vrc_all[i]**2 + vzc_all[i]**2):.6f}\t{random.randint(1, Ni)}\t{np.arctan2(-vzc_all[i], -vrc_all[i]):.6f}\t{zc_all[i] - vf_all[i, random.randint(0, Ni-1), 1]:.6e}\t{reorder_all[i]}\n"
                        f.write(output)

                finish_time = time.perf_counter()
                timeTaken = finish_time - start_time
                
            ''' # Update computation times
                if str(grid_size) not in computation_times:
                    computation_times[str(grid_size)] = {}
                if str(Ni) not in computation_times[str(grid_size)]:
                    computation_times[str(grid_size)][str(Ni)] = {}
                computation_times[str(grid_size)][str(Ni)][str(shots)] = timeTaken
                
                # Save updated computation times
                with open(computation_times_file, 'w') as f:
                    json.dump(computation_times, f, indent=2)'''
                
            print(f"Completed simulation for grid size {grid_size}, ion count {Ni}, and shots {shots}. It took {timeTaken:.2f} seconds!")

    create_formatted_table_file(computation_times, formatted_table_file)
    print("All simulations completed successfully!")


if __name__ == "__main__":
    main()
