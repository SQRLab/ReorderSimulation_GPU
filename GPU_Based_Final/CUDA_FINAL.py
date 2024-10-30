import numpy as np
from scipy.optimize import fsolve
import scipy.constants as con
import math
from numba import cuda, float64, int32
import random
import time
from scipy import special
import json
import os

def ion_position_potential(x):
    """
    Calculates the potential energy of ions in a linear configuration.
    Takes an array of ion positions and returns the forces on each ion due to
    Coulomb interactions with other ions.
    
    Args:
        x (array): Array of ion positions
    Returns:
        array: Forces on each ion from Coulomb interactions
    """

    N = len(x)
    return [
        x[m] - sum([1 / (x[m] - x[n]) ** 2 for n in range(m)]) +
        sum([1 / (x[m] - x[n]) ** 2 for n in range(m + 1, N)])
        for m in range(N)
    ]

def calcPositions(N):
    """
    Calculates equilibrium positions for N ions in a linear trap.
    Uses an empirical scaling law to estimate the extreme positions and
    solves for the minimum energy configuration.
    
    Args:
        N (int): Number of ions
    Returns:
        array: Equilibrium positions of N ions
    """

    estimated_extreme = 0.481 * N ** 0.765  # Hardcoded, should work for at least up to 50 ions
    return fsolve(ion_position_potential, np.linspace(-estimated_extreme, estimated_extreme, N))

def lengthScale(ν, M=None, Z=None):
    """
    Calculates the characteristic length scale for the ion trap system.
    Uses trap frequency, ion mass and charge to determine spatial scaling.
    
    Args:
        ν (float): Trap frequency
        M (float, optional): Ion mass
        Z (float, optional): Ion charge
    Returns:
        float: Characteristic length scale
    """

    if M is None:
        M = con.atomic_mass * 39.9626
    if Z is None:
        Z = 1
    return ((Z ** 2 * con.elementary_charge ** 2) / (4 * np.pi * con.epsilon_0 * M * ν ** 2)) ** (1 / 3)

def makeRF0(m, q, w, Nr, Nz, Nrmid, dr):
    """
    Generates the RF potential field matrix.
    Creates a 2D array representing the RF trapping potential.
    
    Args:
        m (float): Ion mass
        q (float): Ion charge
        w (float): RF frequency
        Nr, Nz (int): Grid dimensions
        Nrmid (int): Midpoint of radial dimension
        dr (float): Radial grid spacing
    Returns:
        array: 2D array of RF potential values
    """

    C = -m * (w ** 2) / q
    RF = np.ones((Nr, Nz), dtype=np.float64)
    for jCell in range(Nr):
        RF[jCell, :] = -RF[jCell, :] * C * (Nrmid - jCell) * dr
    return RF

def makeDC(m, q, w, Nz, Nr, Nzmid, dz):
    """
    Generates the DC potential field matrix.
    Creates a 2D array representing the DC trapping potential.
    
    Args:
        m (float): Ion mass
        q (float): Ion charge
        w (float): DC frequency
        Nz, Nr (int): Grid dimensions
        Nzmid (int): Midpoint of axial dimension
        dz (float): Axial grid spacing
    Returns:
        array: 2D array of DC potential values
    """

    C = -m * (w ** 2) / q
    DC = np.ones((Nr, Nz), dtype=np.float64)
    for kCell in range(Nz):
        DC[:, kCell] = -DC[:, kCell] * C * (Nzmid - kCell) * dz
    return DC

def makeVf(Ni, q, m, l, wr, offsetr, offsetz, vbumpr, vbumpz):
    """
    Creates initial velocity and position vectors for ions.
    Initializes ion positions based on equilibrium calculations and
    adds specified offsets and velocity perturbations.
    
    Args:
        Ni (int): Number of ions
        q (float): Ion charge
        m (float): Ion mass
        l (int): Ion index for perturbation
        wr (float): Radial frequency
        offsetr, offsetz (float): Position offsets
        vbumpr, vbumpz (float): Velocity perturbations
    Returns:
        array: Array of ion positions and velocities
    """

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
    """
    Calculates the Boltzmann velocity distribution.
    Generates a discretized velocity distribution for a given mass and temperature.
    
    Args:
        m (float): Particle mass
        T (float): Temperature
        vmin, vmax (float): Velocity range
        bins (int): Number of velocity bins
    Returns:
        array: Normalized velocity distribution
    """

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

# CUDA device functions
@cuda.jit(device=True)
def ptovPos_cuda(pos, Nmid, dcell):
    """
    Converts physical position to grid cell index.
    CUDA device function for grid position calculations.
    
    Args:
        pos (float): Physical position
        Nmid (float): Grid midpoint
        dcell (float): Grid spacing
    Returns:
        float: Grid cell index
    """

    return pos / dcell + Nmid

@cuda.jit(device=True)
def minDists_cuda(vf, vc):
    """
    Calculates minimum distances between ions and collision particle.
    CUDA device function that finds closest approaches and relative velocities.
    
    Args:
        vf (array): Ion positions and velocities
        vc (array): Collision particle data
    Returns:
        tuple: Minimum distances and velocities between ions and particle
    """

    rid2 = 1e6
    rii2 = 1e6
    vid2 = 1e6
    vii2 = 1e6
    Ni = MAX_IONS
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
    """
    Determines if system is in collision mode.
    Evaluates collision conditions based on distances and parameters.
    
    Args:
        rii (float): Ion-ion distance
        rid (float): Ion-dipole distance
        a (float): Polarizability
        e (float): Threshold parameter
    Returns:
        bool: True if in collision mode
    """

    numerator = a * rii * rii
    denominator = rid ** 5
    return numerator / denominator > e

@cuda.jit(device=True)
def updatePoss_cuda(vf, dr, dz, dt, Nr, Nz, Nrmid, Nzmid):
    """
    Updates positions of ions based on velocities.
    CUDA device function for position evolution and boundary checking.
    
    Args:
        vf (array): Ion positions and velocities
        dr, dz (float): Grid spacings
        dt (float): Time step
        Nr, Nz (int): Grid dimensions
        Nrmid, Nzmid (float): Grid midpoints
    """

    for i in range(MAX_IONS):

        vf[i, 0] += vf[i, 2] * dt
        vf[i, 1] += vf[i, 3] * dt
        rCell = ptovPos_cuda(vf[i, 0], Nrmid, dr)
        zCell = ptovPos_cuda(vf[i, 1], Nzmid, dz)
        if rCell > Nr - 2 or rCell < 1 or zCell > Nz - 2 or zCell < 1:
            vf[i, 0] = 2.0
            vf[i, 1] = 2.0
            vf[i, 2] = 0.0
            vf[i, 3] = 0.0
            vf[i, 5] = 1e6
            vf[i, 6] += 1  # Increment boundary hit count

@cuda.jit(device=True)
def updateVels_cuda(vf, Erf, Ezf, dt):
    """
    Updates velocities of ions based on forces.
    CUDA device function for velocity evolution from field forces.
    
    Args:
        vf (array): Ion positions and velocities
        Erf, Ezf (array): Electric field components
        dt (float): Time step
    """

    Ni = MAX_IONS
    for i in range(Ni):
        # Compute force on ion i
        Fr = vf[i, 4] * Erf[i]  # Fr = q * E_r
        Fz = vf[i, 4] * Ezf[i]  # Fz = q * E_z

        # Update velocities based on forces
        vf[i, 2] += Fr * dt / vf[i, 5]  # v_r += (Fr / m) * dt
        vf[i, 3] += Fz * dt / vf[i, 5]  # v_z += (Fz / m) * dt


@cuda.jit(device=True)
def solveFields_cuda(vf, ErDC, EzDC, ErAC, EzAC, Nrmid, Nzmid, Ni, dr, dz, Erf2, Ezf2):
    """
    Calculates electric fields acting on ions.
    CUDA device function that combines trap fields and ion-ion interactions.
    
    Args:
        vf (array): Ion positions and velocities
        ErDC, EzDC, ErAC, EzAC (array): Trap field components
        Nrmid, Nzmid (float): Grid midpoints
        Ni (int): Number of ions
        dr, dz (float): Grid spacings
        Erf2, Ezf2 (array): Output field arrays
    """

    eps0 = 8.854e-12
    C1 = 4 * math.pi * eps0  # SI units
    Nr = ErDC.shape[0]
    Nz = ErDC.shape[1]

    for i in range(Ni):
        Erf2[i] = 0.0
        Ezf2[i] = 0.0

    for i in range(Ni):
        jCell = int(round(ptovPos_cuda(vf[i, 0], Nrmid, dr)))
        kCell = int(round(ptovPos_cuda(vf[i, 1], Nzmid, dz)))

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
                dist = math.sqrt(sqDist) + 1e-20  # Avoid division by zero
                projR = rdist / dist
                projZ = zdist / dist

                Erf2[i] += -projR * vf[j, 4] / (C1 * sqDist)
                Ezf2[i] += -projZ * vf[j, 4] / (C1 * sqDist)

@cuda.jit(device=True)
def collisionParticlesFields_cuda(vf, vc, Ni, ErDC, EzDC, ErAC, EzAC, dr, dz, dtNow, Nrmid, Nzmid, Nr, Nz, Erfi, Ezfi, erfc_debug, ezfc_debug):
    """
    Calculates fields from collision particles.
    CUDA device function for collision dynamics and induced dipole interactions.
    
    Args:
        vf (array): Ion positions and velocities
        vc (array): Collision particle data
        [Other parameters]: Various field and grid parameters
        erfc_debug, ezfc_debug (array): Debug arrays for field components
    """

    eps0 = 8.854e-12
    C1 = 4 * math.pi * eps0
    Nc = vc.shape[0]

    MAX_NC = 1  # Set to maximum expected number of collisional particles
    MAX_NI = MAX_IONS  # Set to maximum expected number of ions
    Nc = vc.shape[0]  # Actual number used in loops
    
    # Local arrays use maximum sizes
    Erfc = cuda.local.array((MAX_NC, 2), dtype=float64)
    Ezfc = cuda.local.array((MAX_NC, 2), dtype=float64)
    sqDist = cuda.local.array((MAX_NC, MAX_NI), dtype=float64)
    dist_j = cuda.local.array((MAX_NC, MAX_NI), dtype=float64)
    projR_j = cuda.local.array((MAX_NC, MAX_NI), dtype=float64)
    projZ_j = cuda.local.array((MAX_NC, MAX_NI), dtype=float64)
    sqDist_j_1p5 = cuda.local.array((MAX_NC, MAX_NI), dtype=float64)
    pR = cuda.local.array(MAX_NC, dtype=float64)
    pZ = cuda.local.array(MAX_NC, dtype=float64)

    for i in range(Nc):

        jCell = int(ptovPos_cuda(vc[i, 0], Nrmid, dr))
        kCell = int(ptovPos_cuda(vc[i, 1], Nzmid, dz))

        Erfc[i, 0] += ErDC[jCell, kCell] + ErAC[jCell, kCell]
        Ezfc[i, 0] += EzDC[jCell, kCell] + EzAC[jCell, kCell]

        Erfc[i,1] += ((ErDC[jCell+1,kCell] + ErAC[jCell+1,kCell])-(ErDC[jCell-1,kCell] + ErAC[jCell-1,kCell]))/dr
        Ezfc[i,1] += ((EzDC[jCell,kCell+1] + EzAC[jCell,kCell+1])-(EzDC[jCell,kCell-1] + EzAC[jCell,kCell-1]))/dz

        for j in range(Ni):
            rdist = vf[j, 0] - vc[i, 0]
            zdist = vf[j, 1] - vc[i, 1]
            sqDist[i, j] = rdist * rdist + zdist * zdist
            dist_j[i, j] = math.sqrt(rdist * rdist + zdist * zdist)
            projR_j[i, j] = rdist / dist_j[i, j]
            projZ_j[i, j] = zdist / dist_j[i, j]

            Erfc[i, 0] += -projR_j[i, j] * vf[j, 4] / (C1 * sqDist[i, j])
            Ezfc[i, 0] += -projZ_j[i, j] * vf[j, 4] / (C1 * sqDist[i, j])

            sqDist_j_1p5[i, j] = sqDist[i, j] * dist_j[i, j]

            Erfc[i, 1] += 2 * projR_j[i, j] * vf[j, 4] / (C1 * sqDist_j_1p5[i, j])
            Ezfc[i, 1] += 2 * projZ_j[i, j] * vf[j, 4] / (C1 * sqDist_j_1p5[i, j])

        erfc_debug[0] = Erfc[i, 0]
        ezfc_debug[1] = Ezfc[i, 1]

    for k in range(Nc):
        if vc[k,6] != 0.0:
            pR[k] = -2 * math.pi * eps0 * vc[k, 6] * Erfc[k, 0]
            pZ[k] = -2 * math.pi * eps0 * vc[k, 6] * Ezfc[k, 0]

            Fr = math.fabs(pR[k]) * Erfc[k, 1]
            Fz = math.fabs(pZ[k]) * Ezfc[k, 1]

            vc[k, 2] += Fr * dtNow / vc[k, 5]
            vc[k, 3] += Fz * dtNow / vc[k, 5]
    
    dist1 = 0

    for l in range(Ni):
        for m in range(Nc):        
            if vc[m,6]!=0.0:

                Rhatr_j = projR_j[m, l]
                Rhatz_j = projZ_j[m, l]
                dist1 = dist_j[m, l]

                Erfi[l] += -math.fabs(pR[m]) * (2 * Rhatr_j) / (C1 * dist1 ** 3)
                Ezfi[l] += -math.fabs(pZ[m]) * (2 * Rhatz_j) / (C1 * dist1 ** 3)

@cuda.jit(device=True)
def mcCollision_cuda(vf, rc, zc, vrc, vzc, qc, mc, ac, Nt, dtSmall, RF, DC, Nr, Nz, dr, dz, dtLarge, dtCollision, nullFields, reorder):
    """
    Main Monte Carlo collision simulation function.
    CUDA device function that handles entire collision process simulation.
    
    Args:
        vf (array): Ion positions and velocities
        rc, zc (float): Collision particle position
        vrc, vzc (float): Collision particle velocity
        [Other parameters]: Various simulation parameters
        reorder (array): Output flag for ion reordering
    """

    # Initialize local variables
    reorder[0] = 0
    Nrmid = (Nr - 1) / 2.0
    Nzmid = (Nz - 1) / 2.0
    Ni = MAX_IONS
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
    erfc_local = cuda.local.array(2, dtype=float64)
    ezfc_local = cuda.local.array(2, dtype=float64)

    for i in range(Nt):
        rid, rii, vid, vii = minDists_cuda(vf, vc)
        collision = collisionMode_cuda(rii, rid, vc[0, 6], 0.1)
        
        if collision:
            dtNow = rid * 0.01 / (5 * vid)
        else:
            dtNow = dtSmall
        
        if dtNow < dtCollision:
            dtNow = dtCollision

        solveFields_cuda(vf, nullFields, DC, RF, nullFields, Nrmid, Nzmid, Ni, dr, dz, Erfi, Ezfi)

        if vc[0, 5] < 1e6:
            collisionParticlesFields_cuda(vf, vc, Ni, nullFields, DC, RF, nullFields, dr, dz, dtNow, Nrmid, Nzmid, Nr, Nz, Erfi, Ezfi, erfc_local, ezfc_local)
            updatePoss_cuda(vc, dr, dz, dtNow, Nr, Nz, Nrmid, Nzmid)

        else:
            dtNow = dtLarge
        
        updateVels_cuda(vf, Erfi, Ezfi, dtNow)
        updatePoss_cuda(vf, dr, dz, dtNow, Nr, Nz, Nrmid, Nzmid)

        # Check for ion ejection
        mass_sum = 0.0
        for j in range(Ni):
            mass_sum += vf[j, 5]
        if mass_sum > 1e5:
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
def mcCollision_kernel(vf_all, rc_all, zc_all, vrc_all, vzc_all, qc, mc, ac, Nt_all, dtSmall, RF, DC, Nr, Nz, dr, dz, dtLarge, dtCollision, nullFields, reorder_all):
    """
    CUDA kernel for parallel collision simulations.
    Manages multiple simultaneous collision simulations on GPU.
    
    Args:
        vf_all (array): Array of ion positions and velocities for all simulations
        [Other parameters]: Arrays of simulation parameters for all runs
        reorder_all (array): Output array for reordering flags
    """

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
        reorder[0] = 0

        # Call the device function
        mcCollision_cuda(vf, rc, zc, vrc, vzc, qc, mc, ac, Nt, dtSmall, RF, DC, Nr, Nz, dr, dz, dtLarge, dtCollision,
                        nullFields, reorder)

        # Store the results back to global memory
        reorder_all[idx] = reorder[0]

        for i in range(MAX_IONS):
            for j in range(7):
                vf_all[idx, i, j] = vf[i, j]

max_hypotenuse = 1.5e-5

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

MAX_IONS = 2

grid_sizes = [20001]  # Array of grid sizes to test
ion_counts = [2]  # Array of ion counts to test
shot_sizes = [10000, 50000, 100000]  # Array of shot sizes to test

def main():

    """
    Main execution function for the simulation.
    Handles simulation setup, GPU memory management, file I/O,
    and runs multiple simulations with different parameters.
    """
    
    computation_times_file = "computation_times.json"

    # Load existing computation times if file exists
    if os.path.exists(computation_times_file):
        with open(computation_times_file, 'r') as f:
            computation_times = json.load(f)
    else:
        computation_times = {}
    
    for grid_size in grid_sizes:
        
        Nr = Nz = grid_size
        Nrmid = Nzmid = (Nr - 1) // 2
        dr = Dr / float(Nr)
        dz = Dz / float(Nz)

        RF = makeRF0(m, q, wr, Nr, Nz, Nrmid, dr)
        DC = makeDC(m, q, wz, Nz, Nr, Nzmid, dz)

        for Ni in ion_counts:    
            for shots in shot_sizes:
                print(f"Starting simulation with grid size {grid_size}, ion count {Ni}, and shots {shots}")
                start_time = time.perf_counter()

                # Prepare output file with buffer
                buffer_size = 10
                data_buffer = []
                output_file = f"simulation_results_10k/{Ni}ionSimulation_{grid_size}_{shots}shots.txt"
                os.makedirs("simulation_results_10k", exist_ok=True)

                with open(output_file, "w") as f:
                    f.write("axial trapping frequency (MHz) \t velocity(m/s) \t ion collided with \t angle(rads) \t collision offset(m) \t reorder? (1 is reorder 2 is ejection) \n")

                # Initialize arrays for batch processing
                vf_all = np.zeros((shots, Ni, 7), dtype=np.float64)
                rc_all = np.zeros(shots, dtype=np.float64)
                zc_all = np.zeros(shots, dtype=np.float64)
                vrc_all = np.zeros(shots, dtype=np.float64)
                vzc_all = np.zeros(shots, dtype=np.float64)
                Nt_all = np.zeros(shots, dtype=np.int32)

                # Initialize arrays for simulation data
                T = 300
                collisionalMass = 2
                vMin = 50
                vMax = 7000
                numBins = 1000
                boltzDist = Boltz(collisionalMass, T, vMin, vMax, numBins)
                v = np.linspace(vMin, vMax, numBins)
                #angles = np.linspace(-np.pi/2, np.pi/2, 100)
                #offsets = np.linspace(-2e-9, 2e-9, 200)
                max_hypotenuse = 1.5e-5

                actual_velocities = np.zeros(shots)
                actual_angles = np.zeros(shots)
                actual_offsets = np.zeros(shots)
                actual_ions = np.zeros(shots, dtype=int)

                for i in range(shots):
                    # Generate initial conditions
                    vf = makeVf(Ni, q, m, l, wz, offsetr, offsetz, vbumpr, vbumpz)
                    velocity = np.random.choice(v, p=boltzDist/np.sum(boltzDist))
                    angle_choice = np.random.uniform(-np.pi/2, np.pi/2)
                    offset_choice = np.random.uniform(-2e-9, 2e-9)
                    ion_collided = np.random.randint(0, Ni)

                    # Store the actual values used
                    actual_velocities[i] = velocity
                    actual_angles[i] = angle_choice
                    actual_offsets[i] = offset_choice
                    actual_ions[i] = ion_collided

                    # Determine number of timesteps based on velocity
                    if velocity < 200:
                        Nt = 700000
                    elif velocity < 1500:
                        Nt = 400000
                    else:
                        Nt = 250000

                    # Calculate initial positions and velocities
                    r = -np.cos(angle_choice) * max_hypotenuse
                    z = vf[ion_collided, 1] + np.sin(angle_choice) * max_hypotenuse + offset_choice
                    vz = -1 * velocity * np.sin(angle_choice)
                    vr = np.abs(velocity * np.cos(angle_choice))

                    # Store initial conditions
                    vf_all[i, :, :] = vf
                    rc_all[i] = r
                    zc_all[i] = z
                    vrc_all[i] = vr
                    vzc_all[i] = vz
                    Nt_all[i] = Nt

                # Transfer data to GPU
                vf_device = cuda.to_device(vf_all)
                rc_device = cuda.to_device(rc_all)
                zc_device = cuda.to_device(zc_all)
                vrc_device = cuda.to_device(vrc_all)
                vzc_device = cuda.to_device(vzc_all)
                Nt_device = cuda.to_device(Nt_all)
                reorder_device = cuda.device_array(shots, dtype=np.int32)

                RF_device = cuda.to_device(RF)
                DC_device = cuda.to_device(DC)
                nullFields_device = cuda.to_device(np.zeros((Nr, Nz), dtype=np.float64))

                # Launch kernel
                threads_per_block = 1
                blocks = shots

                mcCollision_kernel[blocks, threads_per_block](
                    vf_device, rc_device, zc_device, vrc_device, vzc_device,
                    q, mH2, aH2, Nt_device, dtSmall, RF_device, DC_device, Nr, Nz, dr, dz,
                    dtLarge, dtCollision, nullFields_device, reorder_device
                )

                # Get results back from GPU
                reorder = reorder_device.copy_to_host()

                # Write results to file with buffer
                with open(output_file, "a") as f:
                    for i in range(shots):
                        output = f"{wz}\t{actual_velocities[i]}\t{actual_ions[i]+1}\t{actual_angles[i]}\t{actual_offsets[i]}\t{reorder[i]}\n"
                        data_buffer.append(output)

                        if len(data_buffer) == buffer_size or i == shots - 1:
                            f.writelines(data_buffer)
                            f.flush()
                            data_buffer = []

                finish_time = time.perf_counter()
                timeTaken = finish_time - start_time

                print(f"Simulation completed in {timeTaken:.2f} seconds")

                if str(grid_size) not in computation_times:
                    computation_times[str(grid_size)] = {}
                if str(Ni) not in computation_times[str(grid_size)]:
                    computation_times[str(grid_size)][str(Ni)] = {}
                computation_times[str(grid_size)][str(Ni)][str(shots)] = timeTaken

                with open(computation_times_file, 'w') as f:
                    json.dump(computation_times, f, indent=2)

if __name__ == "__main__":
    main()