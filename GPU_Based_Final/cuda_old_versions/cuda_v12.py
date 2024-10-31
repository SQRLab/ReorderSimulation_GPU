'''import numpy as np
from scipy.optimize import fsolve
import scipy.constants as con
import math
import random
import time
from scipy import special
from numba import cuda, float64, int32

# Constants
amu = 1.67e-27  # Atomic mass unit in kg
eps0 = 8.854e-12  # Vacuum permittivity in F/m
qe = 1.6e-19  # Elementary charge in Coulombs

# Physical parameters
m = 40.0 * amu  # Mass of ion in kg
q = 1.0 * qe  # Charge of ion in Coulombs
wr = 2 * np.pi * 3e6  # Radial angular frequency (rad/s)
wz = 2 * np.pi * 1e6  # Axial angular frequency (rad/s)
aH2 = 8e-31  # Polarizability of H2 in SI units
mH2 = 2.0 * amu  # Mass of H2 in kg

# Simulation parameters
Dr = 30001.5e-9  # Physical width in r (meters)
Dz = 90001.5e-9  # Physical width in z (meters)
dtSmall = 1e-12  # Small timestep (seconds)
dtCollision = 1e-16  # Collision timestep (seconds)
dtLarge = 1e-10  # Large timestep (seconds)

sigmaV = 100e-6  # Fall-off of potential outside trapping region
dv = 20.0  # Bin size for particle speed
vmax = 5000  # Maximum particle speed allowed (m/s)
l = 1  # Index of the ion to bump
vbumpr = 0.00e0  # Starting velocity in r for the bumped ion
vbumpz = -0.0e0  # Starting velocity in z for the bumped ion
offsetz = 0.0e-7  # Starting offset in z for the bumped ion
offsetr = 0.0e-8  # Starting offset in r for the bumped ion
max_hypotenuse = 1.5e-5

# Maximum number of ions (set to 3 as per simulation)
MAX_IONS = 3

# CUDA Device Functions

@cuda.jit(device=True)
def ptovPos_cuda(pos, Nmid, dcell):
    """Converts physical position to virtual grid position."""
    return pos / dcell + Nmid

@cuda.jit(device=True)
def minDists_cuda(vf, vc):
    """Calculates minimum distances and velocities."""
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
    """Determines if a collision is occurring."""
    numerator = a * rii * rii
    denominator = rid ** 5
    # Prevent division by zero or extremely small denominator
    if denominator < 1e-30:
        return False
    return numerator / denominator > e

@cuda.jit(device=True)
def updatePoss_cuda(vf, dr, dz, dt, Nr, Nz, Nrmid, Nzmid):
    """Updates positions of ions based on their velocities."""
    for i in range(vf.shape[0]):
        vf[i, 0] += vf[i, 2] * dt
        vf[i, 1] += vf[i, 3] * dt
        rCell = ptovPos_cuda(vf[i, 0], Nrmid, dr)
        zCell = ptovPos_cuda(vf[i, 1], Nzmid, dz)
        if rCell > Nr - 2 or rCell < 1 or zCell > Nz - 2 or zCell < 1:
            # Reset ion to a valid position within the grid (2 grid cells)
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
    """Updates velocities of ions based on electric fields."""
    for i in range(vf.shape[0]):
        Fr = vf[i, 4] * Erf[i]
        Fz = vf[i, 4] * Ezf[i]
        vf[i, 2] += Fr * dt / (vf[i, 5])
        vf[i, 3] += Fz * dt / (vf[i, 5])

@cuda.jit(device=True)
def solveFields_cuda(vf, ErDC, EzDC, ErAC, EzAC, Nrmid, Nzmid, Ni, dr, dz, Erf2, Ezf2):
    """Calculates electric fields at each ion."""
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
                sqDist = rdist * rdist + zdist * zdist + 1e-20  # Avoid division by zero
                dist = math.sqrt(sqDist)
                projR = rdist / dist
                projZ = zdist / dist

                Erf2[i] += -projR * vf[j, 4] / (C1 * sqDist)
                Ezf2[i] += -projZ * vf[j, 4] / (C1 * sqDist)

@cuda.jit(device=True)
def collisionParticlesFields_cuda(vf, vc, Ni, ErDC, EzDC, ErAC, EzAC, dr, dz, dtNow, Nrmid, Nzmid, Nr, Nz, Erf2, Ezf2):
    """Applies fields from collisional particles to ions and updates collisional particles."""
    eps0 = 8.854e-12
    C1 = 4 * math.pi * eps0
    Nc = vc.shape[0]
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
            Erfc[1] += ((ErDC[jCell + 1, kCell] + ErAC[jCell + 1, kCell]) -
                        (ErDC[jCell - 1, kCell] + ErAC[jCell - 1, kCell])) / (2 * dr)
            Ezfc[1] += ((EzDC[jCell, kCell + 1] + EzAC[jCell, kCell + 1]) -
                        (EzDC[jCell, kCell - 1] + EzAC[jCell, kCell - 1])) / (2 * dz)

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

                Erf2[j] += -math.fabs(pR) * (2 * Rhatr_j) / (C1 * dist_j ** 3)
                Ezf2[j] += -math.fabs(pZ) * (2 * Rhatz_j) / (C1 * dist_j ** 3)

@cuda.jit
def mcCollision_kernel_refined(
    vf_all, rc_all, zc_all, vrc_all, vzc_all, qc, mc, ac,
    Nt_all, dtSmall, RF, DC, Nr, Nz, dr, dz,
    dtLarge, dtCollision, nullFields,
    reorder_all, pos_history, vel_history, Erfi_history, Ezfi_history
):
    """CUDA Kernel to perform the collision simulation with per-timestep data storage."""
    idx = cuda.grid(1)
    if idx >= vf_all.shape[0]:
        return  # Out of bounds

    # Number of ions in this simulation
    Ni = vf_all.shape[1]
    # Number of collisional particles (fixed at 1)
    Nc = 1

    # Initialize local ion data
    vf = cuda.local.array((MAX_IONS, 7), dtype=float64)
    for i in range(Ni):
        for j in range(7):
            vf[i, j] = vf_all[idx, i, j]

    # Initialize collisional particle
    vc = cuda.local.array((Nc, 7), dtype=float64)
    vc[0, 0] = rc_all[idx]
    vc[0, 1] = zc_all[idx]
    vc[0, 2] = vrc_all[idx]
    vc[0, 3] = vzc_all[idx]
    vc[0, 4] = qc
    vc[0, 5] = mc
    vc[0, 6] = ac

    Nt = Nt_all[idx]

    # Initialize reorder flag
    reorder = 0

    # Initialize electric fields
    Erf = cuda.local.array(MAX_IONS, dtype=float64)
    Ezf = cuda.local.array(MAX_IONS, dtype=float64)

    # Initialize history indices
    # Assuming pos_history, vel_history, Erfi_history, Ezfi_history are preallocated with shape (shots, Nt, Ni, ...)
    # Access them as pos_history[idx, t, i, coord]

    for t in range(Nt):
        # Calculate minimum distances and velocities
        rid, rii, vid, vii = minDists_cuda(vf, vc)

        # Determine if a collision is occurring
        collision = collisionMode_cuda(rii, rid, vc[0, 6], 0.1)
        if collision:
            dtNow = rid * 0.01 / (5 * vid)
        else:
            dtNow = dtSmall
        if dtNow < dtCollision:
            dtNow = dtCollision

        # Solve electric fields from ions
        solveFields_cuda(vf, DC, DC, RF, RF, (Nr - 1) / 2.0, (Nz - 1) / 2.0, Ni, dr, dz, Erf, Ezf)

        # If collisional particle exists
        if mc < 1e6:
            # Solve fields involving collisional particles
            collisionParticlesFields_cuda(vf, vc, Ni, DC, DC, RF, RF, dr, dz, dtNow, (Nr - 1) / 2.0, (Nz - 1) / 2.0, Nr, Nz, Erf, Ezf)

            # Update collisional particle positions
            updatePoss_cuda(vc, dr, dz, dtNow, Nr, Nz, (Nr - 1) / 2.0, (Nz - 1) / 2.0)

        else:
            dtNow = dtLarge

        # Update ion velocities based on electric fields
        updateVels_cuda(vf, Erf, Ezf, dtNow)

        # Update ion positions based on velocities
        updatePoss_cuda(vf, dr, dz, dtNow, Nr, Nz, (Nr - 1) / 2.0, (Nz - 1) / 2.0)

        # Store per-timestep data
        for i in range(Ni):
            pos_history[idx, t, i, 0] = vf[i, 0]
            pos_history[idx, t, i, 1] = vf[i, 1]
            vel_history[idx, t, i, 0] = vf[i, 2]
            vel_history[idx, t, i, 1] = vf[i, 3]
            Erfi_history[idx, t, i] = Erf[i]
            Ezfi_history[idx, t, i] = Ezf[i]

        # Check for ion ejection
        ion_ejected = False
        for j in range(Ni):
            if vf[j, 5] > 1e5:
                ion_ejected = True
                break
        if ion_ejected:
            reorder += 2
            break

        # Check for reordering
        for j in range(1, Ni):
            if vf[j, 1] > vf[j - 1, 1]:
                reorder += 1
                Nt = min(Nt, t + 1000)  # Continue simulation for 1000 more timesteps
                break

    # Store the reorder flag
    reorder_all[idx] = reorder

    # Store the updated ion data back to global memory
    for i in range(Ni):
        for j in range(7):
            vf_all[idx, i, j] = vf[i, j]

@cuda.jit
def makeRF0(m, q, wr, Nr, Nz, Nrmid, dr, RF):
    """Initializes the RF electric field."""
    i, j = cuda.grid(2)
    if i < Nr and j < Nz:
        RF[i, j] = math.cos(wr * 0.0)  # Initialize RF field as needed

@cuda.jit
def makeDC(m, q, wz, Nz, Nr, Nzmid, dz, DC):
    """Initializes the DC electric field."""
    i, j = cuda.grid(2)
    if i < Nr and j < Nz:
        DC[i, j] = math.cos(wz * 0.0)  # Initialize DC field as needed

@cuda.jit
def makeVf(Ni, q, m, l, wz, offsetr, offsetz, vbumpr, vbumpz, vf_all):
    """Initializes the ion parameters."""
    idx = cuda.grid(1)
    if idx < vf_all.shape[0]:
        for i in range(Ni):
            vf_all[idx, i, 0] = 0.0 + (offsetr if i == l else 0.0)
            vf_all[idx, i, 1] = 0.0 + (offsetz if i == l else 0.0)
            vf_all[idx, i, 2] = vbumpr if i == l else 0.0
            vf_all[idx, i, 3] = vbumpz if i == l else 0.0
            vf_all[idx, i, 4] = q
            vf_all[idx, i, 5] = m
            vf_all[idx, i, 6] = 0.0  # Polarizability

def main():
    grid_size = 20001  # Grid size
    Ni = 3  # Number of ions
    shots = 1  # Number of simulation shots

    Nr = Nz = grid_size
    Nrmid = Nzmid = (Nr - 1) / 2.0
    dr = Dr / float(Nr)
    dz = Dz / float(Nz)

    # Generate RF and DC fields
    RF = np.ones((Nr, Nz), dtype=np.float64, order='C')
    DC = np.ones((Nr, Nz), dtype=np.float64, order='C')

    start_time = time.perf_counter()

    # Initialize ion data
    vf_all = np.zeros((shots, Ni, 7), dtype=np.float64)
    rc_all = np.zeros(shots, dtype=np.float64)
    zc_all = np.zeros(shots, dtype=np.float64)
    vrc_all = np.zeros(shots, dtype=np.float64)
    vzc_all = np.zeros(shots, dtype=np.float64)
    Nt_all = np.full(shots, 10, dtype=np.int32)  # Number of timesteps

    # Initialize per-timestep history arrays
    pos_history = np.zeros((shots, Nt_all.max(), Ni, 2), dtype=np.float64)
    vel_history = np.zeros((shots, Nt_all.max(), Ni, 2), dtype=np.float64)
    Erfi_history = np.zeros((shots, Nt_all.max(), Ni), dtype=np.float64)
    Ezfi_history = np.zeros((shots, Nt_all.max(), Ni), dtype=np.float64)

    # Initialize initial conditions for each shot
    for i in range(shots):
        # Initialize ions
        for j in range(Ni):
            vf_all[i, j, 0] = 0.0 + (offsetr if j == l else 0.0)  # r position
            vf_all[i, j, 1] = 0.0 + (offsetz if j == l else 0.0)  # z position
            vf_all[i, j, 2] = vbumpr if j == l else 0.0  # vr
            vf_all[i, j, 3] = vbumpz if j == l else 0.0  # vz
            vf_all[i, j, 4] = q  # charge
            vf_all[i, j, 5] = m  # mass
            vf_all[i, j, 6] = 0.0  # polarizability

        # Initialize collisional particle
        velocity = 3340.640640640641  # m/s
        angle_choice = -1.5073298085405573  # Angle in radians (~-86.1 degrees)
        offset_choice = -1.2361809045226131e-09  # 1.236 nm
        ion_collided = 1  # Middle ion

        r = math.cos(angle_choice) * -max_hypotenuse
        z = vf_all[i, ion_collided, 1] + math.sin(angle_choice) * max_hypotenuse + offset_choice
        vz = -1 * velocity * math.sin(angle_choice)
        vr = math.fabs(velocity * math.cos(angle_choice))

        print(f"Initial conditions for shot {i+1}:")
        print(f"r = {r}, z = {z}, vr = {vr}, vz = {vz}")
        print(f"Ion positions:\n{vf_all[i, :, :2]}")
        print(f"Ion velocities:\n{vf_all[i, :, 2:4]}")

        # Set collisional particle initial conditions
        rc_all[i] = r
        zc_all[i] = z
        vrc_all[i] = vr
        vzc_all[i] = vz

    # Transfer data to device
    vf_device = cuda.to_device(vf_all)
    rc_device = cuda.to_device(rc_all)
    zc_device = cuda.to_device(zc_all)
    vrc_device = cuda.to_device(vrc_all)
    vzc_device = cuda.to_device(vzc_all)
    Nt_device = cuda.to_device(Nt_all)
    reorder_device = cuda.device_array(shots, dtype=np.int32)

    # Transfer history arrays to device
    pos_history_device = cuda.to_device(pos_history)
    vel_history_device = cuda.to_device(vel_history)
    Erfi_history_device = cuda.to_device(Erfi_history)
    Ezfi_history_device = cuda.to_device(Ezfi_history)

    RF_device = cuda.to_device(RF)
    DC_device = cuda.to_device(DC)
    nullFields_device = cuda.to_device(np.zeros((Nr, Nz), dtype=np.float64))

    # Define CUDA grid configuration
    threads_per_block = 1
    blocks = shots  # One block per simulation shot

    # Launch the CUDA kernel
    mcCollision_kernel_refined[blocks, threads_per_block](
        vf_device, rc_device, zc_device, vrc_device, vzc_device,
        q, mH2, aH2, Nt_device, dtSmall, RF_device, DC_device,
        Nr, Nz, dr, dz, dtLarge, dtCollision, nullFields_device,
        reorder_device, pos_history_device, vel_history_device,
        Erfi_history_device, Ezfi_history_device
    )

    # Synchronize to ensure kernel completion
    cuda.synchronize()

    # Copy the results back to the host
    reorder = reorder_device.copy_to_host()
    vf_all = vf_device.copy_to_host()
    pos_history = pos_history_device.copy_to_host()
    vel_history = vel_history_device.copy_to_host()
    Erfi_history = Erfi_history_device.copy_to_host()
    Ezfi_history = Ezfi_history_device.copy_to_host()

    # Print per-timestep data
    for i in range(shots):
        print(f"\nSimulation {i+1}: Reorder = {reorder[i]}")
        print("Final conditions:")
        for j in range(Ni):
            print(f"Ion {j}: Position ({vf_all[i, j, 0]:.6e}, {vf_all[i, j, 1]:.6e})")
            print(f"Ion {j}: Velocity ({vf_all[i, j, 2]:.6e}, {vf_all[i, j, 3]:.6e})")
            print(f"Ion {j}: Boundary hit count {int(vf_all[i, j, 6])}")

        # Print per-timestep history
        print("\nPer-timestep data:")
        for t in range(Nt_all[i]):
            print(f"Timestep {t}")
            for j in range(Ni):
                pos_r = pos_history[i, t, j, 0]
                pos_z = pos_history[i, t, j, 1]
                vel_r = vel_history[i, t, j, 0]
                vel_z = vel_history[i, t, j, 1]
                Erfi_val = Erfi_history[i, t, j]
                Ezfi_val = Ezfi_history[i, t, j]
                print(f"Ion {j} Position ({pos_r:.6e}, {pos_z:.6e}) Velocity ({vel_r:.6e}, {vel_z:.6e}) Erfi {Erfi_val:.6e} Ezfi {Ezfi_val:.6e}")
    
    finish_time = time.perf_counter()
    timeTaken = finish_time - start_time

    print(f"\nSimulation completed in {timeTaken:.2f} seconds")

if __name__ == "__main__":
    main()'''

import numpy as np
from scipy.optimize import fsolve
import scipy.constants as con
import math
from numba import cuda, float64, int32
import random
import time
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
max_hypotenuse = 1.5e-5

eii = 0.01  # ion-ion force fractional change limit
eid = 0.01
MAX_IONS = 3
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

# CUDA device functions
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

@cuda.jit(device=True)
def updatePoss_cuda(vf, dr, dz, dt, Nr, Nz, Nrmid, Nzmid):
    for i in range(vf.shape[0]):
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
            vf[i, 6] += 1

@cuda.jit(device=True)
def updateVels_cuda(vf, Erf, Ezf, dt):
    for i in range(vf.shape[0]):
        Fr = vf[i, 4] * Erf[i]
        Fz = vf[i, 4] * Ezf[i]
        vf[i, 2] += Fr * dt / (vf[i, 5])
        vf[i, 3] += Fz * dt / (vf[i, 5])

@cuda.jit(device=True)
def solveFields_cuda(vf, ErDC, EzDC, ErAC, EzAC, Nrmid, Nzmid, Ni, dr, dz, Erf2, Ezf2):
    """Calculates electric fields at each ion from the trap and ion-ion interactions."""
    eps0 = 8.854e-12
    C1 = 4 * math.pi * eps0  # SI units constant
    Nr = ErDC.shape[0]
    Nz = ErDC.shape[1]

    for i in range(Ni):
        Erf2[i] = 0.0
        Ezf2[i] = 0.0

    for i in range(Ni):
        jCell = int(ptovPos_cuda(vf[i, 0], Nrmid, dr) + 0.5)  # Radial cell index
        kCell = int(ptovPos_cuda(vf[i, 1], Nzmid, dz) + 0.5)  # Axial cell index

        # Bounds checking to avoid out-of-bounds access
        if jCell >= 1 and jCell < Nr-1 and kCell >= 1 and kCell < Nz-1:
            # Add trap fields
            Erf2[i] += ErDC[jCell, kCell] + ErAC[jCell, kCell]
            Ezf2[i] += EzDC[jCell, kCell] + EzAC[jCell, kCell]

            # Add ion-ion interactions
            for j in range(Ni):
                if j != i:
                    rdist = vf[j, 0] - vf[i, 0]  # Radial distance
                    zdist = vf[j, 1] - vf[i, 1]  # Axial distance
                    sqDist = rdist ** 2 + zdist ** 2 + 1e-20  # Square distance (avoid div by 0)
                    dist = math.sqrt(sqDist)
                    projR = rdist / dist  # Projection onto radial direction
                    projZ = zdist / dist  # Projection onto axial direction

                    Erf2[i] += -projR * vf[j, 4] / (C1 * sqDist)
                    Ezf2[i] += -projZ * vf[j, 4] / (C1 * sqDist)

@cuda.jit(device=True)
def collisionParticlesFields_cuda(vf, vc, Ni, ErDC, EzDC, ErAC, EzAC, dr, dz, dtNow, Nrmid, Nzmid, Nr, Nz, Erfi, Ezfi):
    eps0 = 8.854e-12
    C1 = 4 * math.pi * eps0
    Nc = vc.shape[0]
    small = 1e-20  # Small value to prevent division by zero

    # Initialize Erfi and Ezfi to zero
    for i in range(Ni):
        Erfi[i] = 0.0
        Ezfi[i] = 0.0

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

@cuda.jit
def mcCollision_kernel(vf_all, rc_all, zc_all, vrc_all, vzc_all, qc, mc, ac, Nt_all, dtSmall, RF, DC, Nr, Nz, dr, dz, dtLarge, dtCollision, nullFields, reorder_all, pos_out, vel_out, field_out):
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

        # Initialize electric fields
        Erfi = cuda.local.array((MAX_IONS,), dtype=float64)
        Ezfi = cuda.local.array((MAX_IONS,), dtype=float64)

        # Initialize collisional particle
        Nc = 1
        vc = cuda.local.array((1, 7), dtype=float64)
        vc[0] = [rc, zc, vrc, vzc, qc, mc, ac]

        Nrmid = (Nr - 1) / 2
        Nzmid = (Nz - 1) / 2

        # Time-stepping loop
        for t in range(Nt):
            # Calculate minimum distances and velocities
            rid, rii, vid, vii = minDists_cuda(vf, vc)

            # Determine collision mode
            collision = collisionMode_cuda(rii, rid, vc[0, 6], 0.1)

            # Set time step
            if collision:
                dtNow = min(rid * eid / (5 * vid), dtCollision)
            else:
                dtNow = min(rii * eii / (5 * vii), dtSmall)

            dtNow = max(dtNow, dtCollision)
            dtNow = min(dtNow, dtLarge)

            # Solve fields
            solveFields_cuda(vf, nullFields, DC, RF, nullFields, Nrmid, Nzmid, MAX_IONS, dr, dz, Erfi, Ezfi)

            if vc[0, 5] < 1e6:  # if collisional particle exists
                # Update collisional particle
                collisionParticlesFields_cuda(vf, vc, MAX_IONS, nullFields, DC, RF, nullFields, dr, dz, dtNow, Nrmid, Nzmid, Nr, Nz, Erfi, Ezfi)
                updatePoss_cuda(vc, dr, dz, dtNow, Nr, Nz, Nrmid, Nzmid)

            # Update ion velocities and positions
            updateVels_cuda(vf, Erfi, Ezfi, dtNow)
            updatePoss_cuda(vf, dr, dz, dtNow, Nr, Nz, Nrmid, Nzmid)

            # Check for reordering
            for j in range(1, MAX_IONS):
                if vf[j, 1] > vf[j-1, 1]:
                    reorder[0] += 1

            # Save positions, velocities, and fields
            for i in range(MAX_IONS):
                pos_out[idx, t, i, 0] = vf[i, 0]
                pos_out[idx, t, i, 1] = vf[i, 1]
                vel_out[idx, t, i, 0] = vf[i, 2]
                vel_out[idx, t, i, 1] = vf[i, 3]
                field_out[idx, t, i, 0] = Erfi[i]
                field_out[idx, t, i, 1] = Ezfi[i]

            # Check for ejection
            if cuda.sum(vf[:, 5]) > 1e5:
                reorder[0] += 2
                break

            # Check for NaN values
            for i in range(MAX_IONS):
                if math.isnan(vf[i, 0]) or math.isnan(vf[i, 1]) or math.isnan(vf[i, 2]) or math.isnan(vf[i, 3]):
                    print("NaN detected in ion parameters at timestep", t)
                    reorder[0] += 4
                    break

        reorder_all[idx] = reorder[0]

def main():
    grid_size = 20001
    Ni = 3
    shots = 1
    Nt = 10  # Number of timesteps

    Nr = Nz = grid_size
    dr = Dr / float(Nr)
    dz = Dz / float(Nz)

    RF = makeRF0(m, q, wr, Nr, Nz, Nr // 2, dr)
    DC = makeDC(m, q, wz, Nz, Nr, Nz // 2, dz)

    print(f"Starting simulation with grid size {grid_size}, ion count {Ni}, and shots {shots}")

    start_time = time.perf_counter()

    # Allocate arrays for all shots
    vf_all = np.zeros((shots, Ni, 7), dtype=np.float64)
    rc_all = np.zeros(shots, dtype=np.float64)
    zc_all = np.zeros(shots, dtype=np.float64)
    vrc_all = np.zeros(shots, dtype=np.float64)
    vzc_all = np.zeros(shots, dtype=np.float64)
    Nt_all = np.full(shots, Nt, dtype=np.int32)
    reorder_all = np.zeros(shots, dtype=np.int32)

    # Prepare to store position, velocity, and field results
    pos_out = np.zeros((shots, Nt, Ni, 2), dtype=np.float64)
    vel_out = np.zeros((shots, Nt, Ni, 2), dtype=np.float64)
    field_out = np.zeros((shots, Nt, Ni, 2), dtype=np.float64)

    velocity = 3340.640640640641  # m/s
    angle_choice = -1.5073298085405573  # 45 degrees
    offset_choice = -1.2361809045226131e-09  # 1 nm
    ion_collided = 1

    for i in range(shots):
        # Generate initial ion positions and velocities
        vf = makeVf(Ni, q, m, l, wz, offsetr, offsetz, vbumpr, vbumpz)

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

    # Transfer data to device
    vf_device = cuda.to_device(vf_all)
    rc_device = cuda.to_device(rc_all)
    zc_device = cuda.to_device(zc_all)
    vrc_device = cuda.to_device(vrc_all)
    vzc_device = cuda.to_device(vzc_all)
    Nt_device = cuda.to_device(Nt_all)
    reorder_device = cuda.to_device(reorder_all)
    pos_device = cuda.device_array_like(pos_out)
    vel_device = cuda.device_array_like(vel_out)
    field_device = cuda.device_array_like(field_out)

    RF_device = cuda.to_device(RF)
    DC_device = cuda.to_device(DC)
    nullFields_device = cuda.to_device(np.zeros((Nr, Nz), dtype=np.float64))

    threads_per_block = 1
    blocks = shots  # Number of simulations

    mcCollision_kernel[blocks, threads_per_block](
        vf_device, rc_device, zc_device, vrc_device, vzc_device,
        q, mH2, aH2, Nt_device, dtSmall, RF_device, DC_device, Nr, Nz, dr, dz,
        dtLarge, dtCollision, nullFields_device, reorder_device, pos_device, vel_device, field_device
    )

    # Copy the results back to the host
    pos_host = pos_device.copy_to_host()
    vel_host = vel_device.copy_to_host()
    field_host = field_device.copy_to_host()
    reorder_host = reorder_device.copy_to_host()

    print("Initial conditions:")
    print(f"r = {r}, z = {z}, vr = {vr}, vz = {vz}")
    for i in range(Ni):
        print(vf_all[0][i])

    for t in range(10):
        print(f"Timestep {t}")
        for ion in range(Ni):
            r, z = pos_host[0, t, ion]
            vr, vz = vel_host[0, t, ion]
            Erfi, Ezfi = field_host[0, t, ion]
            print(f"Ion {ion} Position {r:.3e} {z:.3e} Velocity {vr:.3e} {vz:.3e} Erfi {Erfi:.3e} Ezfi {Ezfi:.3e}")

    print("Final conditions:")
    print(f"Reorder = {reorder_host[0]}")
    print("Ion positions:")
    for ion in range(Ni):
        r, z = pos_host[0, -1, ion]
        print(f"Ion {ion}: ({r:.3e}, {z:.3e})")

    print("Ion velocities:")
    for ion in range(Ni):
        vr, vz = vel_host[0, -1, ion]
        print(f"Ion {ion}: ({vr:.3e}, {vz:.3e})")

    finish_time = time.perf_counter()
    timeTaken = finish_time - start_time

    print(f"Simulation completed in {timeTaken:.2f} seconds")

if __name__ == "__main__":
    main()
