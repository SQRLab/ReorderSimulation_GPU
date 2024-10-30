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
            for j in range(7):
                vf[i, j] = 0.0
            vf[i, 0] = 2.0
            vf[i, 1] = 2.0
            vf[i, 2] = 0.0
            vf[i, 3] = 0.0
            vf[i, 5] = 1e6
            vf[i, 6] += 1  # Increment boundary hit count

'''@cuda.jit(device=True)
def updateVels_cuda(vf, Erf, Ezf, dt):
    for i in range(vf.shape[0]):
        Fr = vf[i, 4] * Erf[i]
        Fz = vf[i, 4] * Ezf[i]
        vf[i, 2] += Fr * dt / (vf[i, 5])
        vf[i, 3] += Fz * dt / (vf[i, 5])'''

@cuda.jit(device=True)
def updateVels_cuda(vf, Erf, Ezf, dt):
    Ni = vf.shape[0]
    for i in range(Ni):
        # Compute force on ion i
        Fr = vf[i, 4] * Erf[i]  # Fr = q * E_r
        Fz = vf[i, 4] * Ezf[i]  # Fz = q * E_z

        # Update velocities based on forces
        vf[i, 2] += Fr * dt / vf[i, 5]  # v_r += (Fr / m) * dt
        vf[i, 3] += Fz * dt / vf[i, 5]  # v_z += (Fz / m) * dt


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
                dist = math.sqrt(sqDist) + 1e-20  # Avoid division by zero
                projR = rdist / dist
                projZ = zdist / dist

                Erf2[i] += -projR * vf[j, 4] / (C1 * sqDist)
                Ezf2[i] += -projZ * vf[j, 4] / (C1 * sqDist)

@cuda.jit(device=True)
def collisionParticlesFields_cuda(vf, vc, Ni, ErDC, EzDC, ErAC, EzAC, dr, dz, dtNow, Nrmid, Nzmid, Nr, Nz, Erfi, Ezfi, erfc_debug, ezfc_debug):
    eps0 = 8.854e-12
    C1 = 4 * math.pi * eps0
    Nc = vc.shape[0]
    small = 1e-20  # Small value to prevent division by zero

    # Initialize Erfi and Ezfi
    for i in range(Ni):
        Erfi[i] = 0.0
        Ezfi[i] = 0.0

    i = 0

    # For each collisional particle
    #for i in range(Nc):
    Erfc = cuda.local.array(2, dtype=float64)  # [0]: field, [1]: gradient
    Ezfc = cuda.local.array(2, dtype=float64)

    Erfc[0] = 0.0  # Field
    Erfc[1] = 0.0  # Gradient
    Ezfc[0] = 0.0
    Ezfc[1] = 0.0

    # Compute cell indices
    jCell = int(ptovPos_cuda(vc[i, 0], Nrmid, dr) + 0.5)
    kCell = int(ptovPos_cuda(vc[i, 1], Nzmid, dz) + 0.5)

    if 1 <= jCell < Nr - 1 and 1 <= kCell < Nz - 1: #added to make sure that particle is in the boundaries
        # Background fields
        Erfc[0] += ErDC[jCell, kCell] + ErAC[jCell, kCell]
        Ezfc[0] += EzDC[jCell, kCell] + EzAC[jCell, kCell]

        # Field gradients
        Erfc[1] += ((ErDC[jCell + 1, kCell] + ErAC[jCell + 1, kCell]) - (ErDC[jCell - 1, kCell] + ErAC[jCell - 1, kCell])) / dr
        Ezfc[1] += ((EzDC[jCell, kCell + 1] + EzAC[jCell, kCell + 1]) - (EzDC[jCell, kCell - 1] + EzAC[jCell, kCell - 1])) / dz
        

        # Compute fields from ions to collisional particle
        for j in range(Ni):
            rdist = vf[j, 0] - vc[i, 0]
            zdist = vf[j, 1] - vc[i, 1]
            sqDist_j = rdist * rdist + zdist * zdist + small
            dist_j = math.sqrt(sqDist_j)
            projR_j = rdist / dist_j
            projZ_j = zdist / dist_j

            Erfc[0] += -projR_j * vf[j, 4] / (C1 * sqDist_j)
            Ezfc[0] += -projZ_j * vf[j, 4] / (C1 * sqDist_j)

            sqDist_j_1p5 = sqDist_j * dist_j  # To compute sqDist_j ** 1.5

            Erfc[1] += 2 * projR_j * vf[j, 4] / (C1 * sqDist_j_1p5)
            Ezfc[1] += 2 * projZ_j * vf[j, 4] / (C1 * sqDist_j_1p5)

        erfc_debug[0] = Erfc[0]
        erfc_debug[1] = Erfc[1]
        ezfc_debug[0] = Ezfc[0]
        ezfc_debug[1] = Ezfc[1]

        # Induced dipole calculations
        if vc[i,6] != 0.0:
            pR = -2 * math.pi * eps0 * vc[i, 6] * Erfc[0]
            pZ = -2 * math.pi * eps0 * vc[i, 6] * Ezfc[0]

            Fr = math.fabs(pR) * Erfc[1]
            Fz = math.fabs(pZ) * Ezfc[1]

            vc[i, 2] += Fr * dtNow / vc[i, 5]
            vc[i, 3] += Fz * dtNow / vc[i, 5]

            # Apply fields from collisional particle to ions
            for j in range(Ni):
                rdist = vf[j, 0] - vc[i, 0]
                zdist = vf[j, 1] - vc[i, 1]
                dist_j = math.sqrt(rdist * rdist + zdist * zdist + small)
                Rhatr_j = rdist / dist_j
                Rhatz_j = zdist / dist_j

                Erfi[j] += -math.fabs(pR) * (2 * Rhatr_j) / (C1 * dist_j ** 3)
                Ezfi[j] += -math.fabs(pZ) * (2 * Rhatz_j) / (C1 * dist_j ** 3)


'''@cuda.jit(device=True)
def collisionParticlesFields_cuda(vf, vc, Ni, ErDC, EzDC, ErAC, EzAC, dr, dz, dtNow, Nrmid, Nzmid, Nr, Nz, Erfi, Ezfi):
    eps0 = 8.854e-12
    C1 = 4.0 * 3.141592653589793 * eps0  # 4πε₀
    Nc = vc.shape[0]
    small = 1e-20  # Prevent division by zero

    # Initialize electric fields for ions
    for i in range(Ni):
        Erfi[i] = 0.0
        Ezfi[i] = 0.0

    # Loop over ions first
    for i in range(Ni):
        # Add background trap fields to Erfi and Ezfi
        rCell = ptovPos_cuda(vf[i, 0], Nrmid, dr)
        zCell = ptovPos_cuda(vf[i, 1], Nzmid, dz)

        # Boundary check
        if rCell >= 0 and rCell < Nr and zCell >= 0 and zCell < Nz:
            rCell_int = int(rCell)
            zCell_int = int(zCell)
            Erfi[i] += ErDC[rCell_int, zCell_int] + ErAC[rCell_int, zCell_int]
            Ezfi[i] += EzDC[rCell_int, zCell_int] + EzAC[rCell_int, zCell_int]
        else:
            # If out of bounds, skip adding trap fields
            pass

        # Loop over collisional particles
        for j in range(Nc):
            # Compute distance between ion i and collisional particle j
            rdist = vf[i, 0] - vc[j, 0]
            zdist = vf[i, 1] - vc[j, 1]
            sqDist = rdist * rdist + zdist * zdist + small
            dist = math.sqrt(sqDist)

            # Projection factors
            projR = rdist / dist
            projZ = zdist / dist

            # Induced dipole moments only if polarizability is non-zero
            if vc[j, 6] != 0.0:
                # Calculate dipole moments based on background electric fields at collisional particle's position
                rCell_cp = ptovPos_cuda(vc[j, 0], Nrmid, dr)
                zCell_cp = ptovPos_cuda(vc[j, 1], Nzmid, dz)

                if rCell_cp >= 0 and rCell_cp < Nr and zCell_cp >= 0 and zCell_cp < Nz:
                    rCell_cp_int = int(rCell_cp)
                    zCell_cp_int = int(zCell_cp)
                    Erfc0 = ErDC[rCell_cp_int, zCell_cp_int] + ErAC[rCell_cp_int, zCell_cp_int]
                    Ezfc0 = EzDC[rCell_cp_int, zCell_cp_int] + EzAC[rCell_cp_int, zCell_cp_int]
                else:
                    Erfc0 = 0.0
                    Ezfc0 = 0.0

                # Calculate induced dipole moments (pR and pZ)
                pR = -2.0 * 3.141592653589793 * eps0 * vc[j, 6] * Erfc0
                pZ = -2.0 * 3.141592653589793 * eps0 * vc[j, 6] * Ezfc0

                # Update electric fields on ion i due to collisional particle j's dipole
                Erfi[i] += -math.fabs(pR) * (2.0 * projR) / (C1 * dist ** 3)
                Ezfi[i] += -math.fabs(pZ) * (2.0 * projZ) / (C1 * dist ** 3)

                # Update velocities of collisional particle j based on force from ion i
                Fr = math.fabs(pR) * (2.0 * projR) / (C1 * dist ** 3)
                Fz = math.fabs(pZ) * (2.0 * projZ) / (C1 * dist ** 3)

                vc[j, 2] += Fr * dtNow / vc[j, 5]  # Update radial velocity
                vc[j, 3] += Fz * dtNow / vc[j, 5]  # Update axial velocity'''

MAX_IONS = 3

@cuda.jit(device=True)
def mcCollision_cuda(vf, rc, zc, vrc, vzc, qc, mc, ac, Nt, dtSmall, RF, DC, Nr, Nz, dr, dz, dtLarge, dtCollision, nullFields, reorder, ion_data, collision_data, erfc_debug, ezfc_debug, erfc_diff, ezfc_diff, dtNow_array):
    # Initialize local variables
    reorder[0] = 0
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

        dtNow_array[i] = dtNow

        # Initialize electric fields
        for j in range(Ni):
            Erfi[j] = 0.0
            Ezfi[j] = 0.0

        solveFields_cuda(vf, nullFields, DC, RF, nullFields, Nrmid, Nzmid, Ni, dr, dz, Erfi, Ezfi)

        for j in range(Ni):
            ion_data[i, j+1, 4] = Erfi[j]
            ion_data[i, j+1, 5] = Ezfi[j]

        if vc[0, 5] < 1e6:
            collisionParticlesFields_cuda(vf, vc, Ni, nullFields, DC, RF, nullFields, dr, dz, dtNow, Nrmid, Nzmid, Nr, Nz, Erfi, Ezfi, erfc_local, ezfc_local)
            erfc_debug[i, 0] = erfc_local[0]  # Field
            erfc_debug[i, 1] = erfc_local[1]  # Gradient
            ezfc_debug[i, 0] = ezfc_local[0]  # Field
            ezfc_debug[i, 1] = ezfc_local[1]  # Gradient
            updatePoss_cuda(vc, dr, dz, dtNow, Nr, Nz, Nrmid, Nzmid)
            collision_data[i, 0] = vc[0, 0]  # r position
            collision_data[i, 1] = vc[0, 1]  # z position
            collision_data[i, 2] = vc[0, 2]  # r velocity
            collision_data[i, 3] = vc[0, 3]  # z velocity
            collision_data[i, 4] = dtNow     # current timestep

            for j in range(Ni):
                ion_data[i, j+1, 6] = Erfi[j]
                ion_data[i, j+1, 7] = Ezfi[j]
        else:
            dtNow = dtLarge
            for j in range(Ni):
                ion_data[i, j+1, 6] = Erfi[j]
                ion_data[i, j+1, 7] = Ezfi[j]
        
        for j in range(Ni):
                erfc_diff[i, j] = ion_data[i, j+1, 6] - ion_data[i, j+1, 4]
                ezfc_diff[i, j] = ion_data[i, j+1, 7] - ion_data[i, j+1, 5]

        for j in range(Ni):
            ion_data[i, j+1, 0] = vf[j, 0]  # r position
            ion_data[i, j+1, 1] = vf[j, 1]  # z position
            ion_data[i, j+1, 2] = vf[j, 2]  # r velocity
            ion_data[i, j+1, 3] = vf[j, 3]  # z velocity

        updateVels_cuda(vf, Erfi, Ezfi, dtNow)
        updatePoss_cuda(vf, dr, dz, dtNow, Nr, Nz, Nrmid, Nzmid)

        ion_data[i, 0, 0] = dtNow
        ion_data[i, 0, 1] = 1.0 if collision else 0.0
        ion_data[i, 0, 2] = rid
        ion_data[i, 0, 3] = rii
        ion_data[i, 0, 4] = vid
        ion_data[i, 0, 5] = vii

        # for j in range(Ni):
        #     ion_data[i, j+1, 0] = vf[j, 0]  # r position
        #     ion_data[i, j+1, 1] = vf[j, 1]  # z position
        #     ion_data[i, j+1, 2] = vf[j, 2]  # r velocity
        #     ion_data[i, j+1, 3] = vf[j, 3]  # z velocity

        # Check for ion ejection
        ion_ejected = False
        for j in range(Ni):
            if vf[j, 5] > 1e5:
                ion_ejected = True
                break
        if ion_ejected:
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
def mcCollision_kernel(vf_all, rc_all, zc_all, vrc_all, vzc_all, qc, mc, ac, Nt_all, dtSmall, RF, DC, Nr, Nz, dr, dz, dtLarge, dtCollision, nullFields, reorder_all, ion_data_all, collision_data_all, erfc_debug_all, ezfc_debug_all, erfc_diff_all, ezfc_diff_all, dtNow_all):
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

        ion_data = ion_data_all[idx]

        # Call the device function
        mcCollision_cuda(vf, rc, zc, vrc, vzc, qc, mc, ac, Nt, dtSmall, RF, DC, Nr, Nz, dr, dz, dtLarge, dtCollision,
                        nullFields, reorder, ion_data, collision_data_all[idx], erfc_debug_all[idx], ezfc_debug_all[idx], erfc_diff_all[idx], ezfc_diff_all[idx], dtNow_all[idx])

        # Store the results back to global memory
        reorder_all[idx] = reorder[0]

        for i in range(MAX_IONS):
            for j in range(7):
                vf_all[idx, i, j] = vf[i, j]

max_hypotenuse = 1.5e-5

# Main simulation code
def main():
    grid_size = 25001
    Ni = 3
    shots = 1
    Nt = 700000

    Nr = Nz = grid_size
    Nrmid = Nzmid = (Nr - 1) // 2
    dr = Dr / float(Nr)
    dz = Dz / float(Nz)

    RF = makeRF0(m, q, wr, Nr, Nz, Nrmid, dr)
    DC = makeDC(m, q, wz, Nz, Nr, Nzmid, dz)
    print(DC[12103, 17996])
    print(RF[12103, 17996])

    # def print_timestep_info(ion_data, timestep):
    #     print(f"Timestep {timestep}:")
    #     print(f"dtNow: {ion_data[timestep, 0, 0]:.6e}")
    #     print(f"Collision occurring: {'Yes' if ion_data[timestep, 0, 1] > 0.5 else 'No'}")
    #     print(f"Minimum distances - rid: {ion_data[timestep, 0, 2]:.6e}, rii: {ion_data[timestep, 0, 3]:.6e}")
    #     print(f"Minimum velocities - vid: {ion_data[timestep, 0, 4]:.6e}, vii: {ion_data[timestep, 0, 5]:.6e}")
    #     for j in range(1, Ni+1):
    #         print(f"Ion {j-1}:")
    #         print(f"  Position: ({ion_data[timestep, j, 0]:.6e}, {ion_data[timestep, j, 1]:.6e})")
    #         print(f"  Velocity: ({ion_data[timestep, j, 2]:.6e}, {ion_data[timestep, j, 3]:.6e})")
    #         print(f"  Electric field: Erfi = {ion_data[timestep, j, 4]:.6e}, Ezfi = {ion_data[timestep, j, 5]:.6e}")
    #     print("\n")

    print(f"Starting simulation with grid size {grid_size}, ion count {Ni}, and shots {shots}")

    start_time = time.perf_counter()

    # Allocate arrays for all shots
    vf_all = np.zeros((shots, Ni, 7), dtype=np.float64)
    rc_all = np.zeros(shots, dtype=np.float64)
    zc_all = np.zeros(shots, dtype=np.float64)
    vrc_all = np.zeros(shots, dtype=np.float64)
    vzc_all = np.zeros(shots, dtype=np.float64)
    Nt_all = np.zeros(shots, dtype=np.int32)
    ion_data_all = np.zeros((shots, Nt, Ni+1, 8), dtype=np.float64)
    collision_data_all = np.zeros((shots, Nt, 5), dtype=np.float64)
    erfc_debug_all = np.zeros((shots, Nt, 2), dtype=np.float64)
    ezfc_debug_all = np.zeros((shots, Nt, 2), dtype=np.float64)
    erfc_diff_all = np.zeros((shots, Nt, Ni), dtype=np.float64)
    ezfc_diff_all = np.zeros((shots, Nt, Ni), dtype=np.float64)
    dtNow_all = np.zeros((shots, Nt), dtype=np.float64)

    T = 300
    collisionalMass = 2
    vMin = 50
    vMax = 7000
    numBins = 1000
    boltzDist = Boltz(collisionalMass, T, vMin, vMax, numBins)
    v = np.linspace(vMin, vMax, numBins)
    angles = np.linspace(-np.pi/2, np.pi/2, 100)
    offsets = np.linspace(-2e-9, 2e-9, 200)

    for i in range(shots):
        # Generate initial ion positions and velocities
        vf = makeVf(Ni, q, m, l, wz, offsetr, offsetz, vbumpr, vbumpz)
        print(vf)

        # Random initial conditions for colliding particle
        '''velocity = random.choices(v, weights=boltzDist)[0]
        angle_choice = random.choice(angles)
        offset_choice = random.choice(offsets)
        ion_collided = random.randint(0, Ni - 1)'''

        '''velocity = 3340.640640640641  # m/s
        angle_choice = -1.5073298085405573  # 45 degrees
        offset_choice = -1.2361809045226131e-09  # 1 nm
        ion_collided = 1'''

        velocity = 2387.5375375375374  # m/s
        angle_choice = 1.539063067667727  # 45 degrees
        offset_choice = 4.120603015075375e-10  # 1 nm
        ion_collided = 0

        r = math.cos(angle_choice) * -max_hypotenuse
        z = vf[ion_collided, 1] + math.sin(angle_choice) * max_hypotenuse + offset_choice
        vz = -1 * velocity * math.sin(angle_choice)
        vr = math.fabs(velocity * math.cos(angle_choice))

        print(f"Initial conditions:")
        print(f"r = {r}, z = {z}, vr = {vr}, vz = {vz}")
        '''print(f"Ion positions:")
        for j in range(Ni):
            print(f"Ion {j}: ({vf[j, 0]:.6e}, {vf[j, 1]:.6e})")
        print(f"Ion velocities:")
        for j in range(Ni):
            print(f"Ion {j}: ({vf[j, 2]:.6e}, {vf[j, 3]:.6e})")'''

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

    RF_device = cuda.to_device(RF)
    DC_device = cuda.to_device(DC)
    nullFields_device = cuda.to_device(np.zeros((Nr, Nz), dtype=np.float64))
    ion_data_device = cuda.to_device(ion_data_all)
    collision_data_device = cuda.to_device(collision_data_all)
    erfc_debug_device = cuda.to_device(erfc_debug_all)
    ezfc_debug_device = cuda.to_device(ezfc_debug_all)
    erfc_diff_device = cuda.to_device(erfc_diff_all)
    ezfc_diff_device = cuda.to_device(ezfc_diff_all)
    dtNow_device = cuda.to_device(dtNow_all)

    threads_per_block = 1
    blocks = shots  # Number of simulations

    mcCollision_kernel[blocks, threads_per_block](
        vf_device, rc_device, zc_device, vrc_device, vzc_device,
        q, mH2, aH2, Nt_device, dtSmall, RF_device, DC_device, Nr, Nz, dr, dz,
        dtLarge, dtCollision, nullFields_device, reorder_device, ion_data_device, collision_data_device, erfc_debug_device, ezfc_debug_device, erfc_diff_device, ezfc_diff_device, dtNow_device
    )

    # Copy the results back to the host
    reorder = reorder_device.copy_to_host()
    ion_data_all = ion_data_device.copy_to_host()
    collision_data_all = collision_data_device.copy_to_host()
    erfc_debug_all = erfc_debug_device.copy_to_host()
    ezfc_debug_all = ezfc_debug_device.copy_to_host()
    erfc_diff_all = erfc_diff_device.copy_to_host()
    ezfc_diff_all = ezfc_diff_device.copy_to_host()
    dtNow_all = dtNow_device.copy_to_host()

    for i in range(shots):
        print(f"Simulation {i+1}: Reorder = {reorder[i]}")
        '''for t in range(Nt):
            print(f"Timestep {t}")
            for j in range(Ni):
                print(f"Ion {j}: Position ({ion_data_all[i, t, j, 0]:.6e}, {ion_data_all[i, t, j, 1]:.6e}), "
                      f"Velocity ({ion_data_all[i, t, j, 2]:.6e}, {ion_data_all[i, t, j, 3]:.6e}), "
                      f"Erfi {ion_data_all[i, t, j, 4]:.6e}, Ezfi {ion_data_all[i, t, j, 5]:.6e}")'''
        
    with open('simulation_data_gpu_700k.txt', 'w') as f:
        for i in range(shots):
            for t in range(Nt):
                f.write(f"Timestep {t}\n")
                '''f.write(f"dtNow: {ion_data_all[i, t, 0, 0]:.6e}\n")
                f.write(f"Collision occurring: {'Yes' if ion_data_all[i, t, 0, 1] > 0.5 else 'No'}\n")
                f.write(f"Minimum distances - rid: {ion_data_all[i, t, 0, 2]:.6e}, rii: {ion_data_all[i, t, 0, 3]:.6e}\n")
                f.write(f"Minimum velocities - vid: {ion_data_all[i, t, 0, 4]:.6e}, vii: {ion_data_all[i, t, 0, 5]:.6e}\n")'''
                for j in range(1, Ni+1):
                    f.write(f"Ion {j-1} Position {ion_data_all[i, t, j, 0]:.6e} {ion_data_all[i, t, j, 1]:.6e} "
                            f"Velocity {ion_data_all[i, t, j, 2]:.6e} {ion_data_all[i, t, j, 3]:.6e} "
                            f"Erfi before: {ion_data_all[i, t, j, 4]:.6e}, Ezfi before: {ion_data_all[i, t, j, 5]:.6e}\n")
                            #f"Erfi after: {ion_data_all[i, t, j, 6]:.6e}, Ezfi after: {ion_data_all[i, t, j, 7]:.6e}\n")
                            #f"Erfc: {erfc_diff_all[i, t, j-1]:.6e}, Ezfc: {ezfc_diff_all[i, t, j-1]:.6e}\n")
                f.write("\n")    
    
    # with open('collision_data_gpu_700k.txt', 'w') as f:
    #     for i in range(shots):
    #         for t in range(Nt):
    #             if collision_data_all[i, t, 4] != 0:  # Check if this timestep has collision data
    #                 f.write(f"Timestep {t} "
    #                         f"Position: {collision_data_all[i, t, 0]}, {collision_data_all[i, t, 1]} "
    #                         f"Velocity: {collision_data_all[i, t, 2]}, {collision_data_all[i, t, 3]}\n\n")
    #         f.write("\n")

    # with open('erfc_ezfc_debug_gpu_700k.txt', 'w') as f:
    #     for i in range(shots):
    #         for t in range(Nt):
    #             if collision_data_all[i, t, 4] != 0:  # Check if this timestep has collision data
    #                 f.write(f"Timestep {t} "
    #                         f"Erfc field: {erfc_debug_all[i, t, 0]:.6e}, "
    #                         f"Erfc gradient: {erfc_debug_all[i, t, 1]:.6e}, "
    #                         f"Ezfc field: {ezfc_debug_all[i, t, 0]:.6e}, "
    #                         f"Ezfc gradient: {ezfc_debug_all[i, t, 1]:.6e}\n\n")
    #         f.write("\n")

    # with open('dtNow_values_700k.txt', 'w') as f:
    #     for i in range(shots):
    #         f.write(f"Simulation {i+1}:\n")
    #         for t in range(Nt):
    #             f.write(f"Timestep {t}: dtNow = {dtNow_all[i, t]:.6e}\n")
    #         f.write("\n")

    finish_time = time.perf_counter()
    timeTaken = finish_time - start_time

    '''for i in range(shots):
        print(f"Simulation {i+1}: Reorder = {reorder[i]}")
        print_timestep_info(ion_data_all[i], 8353)
        print_timestep_info(ion_data_all[i], 8354)'''

    print(f"Simulation completed in {timeTaken:.2f} seconds")

if __name__ == "__main__":
    main()