import numpy as np
from scipy.optimize import fsolve
import scipy.constants as con
import math
import numba
from numba import cuda, float64, int32
import time
from scipy import special
import json
import os
from math import pi as π

MAX_IONS = 3
current_ion = 2

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
    return [x[m] - sum([1/(x[m]-x[n])**2 for n in range(m)]) + sum([1/(x[m]-x[n])**2 for n in range(m+1,N)])
               for m in range(N)]

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

    estimated_extreme = 0.481*N**0.765 # Hardcoded, should work for at least up to 50 ions
    return fsolve(ion_position_potential, np.linspace(-estimated_extreme, estimated_extreme, N))

def lengthScale(ν, M=None, Z=None):
    '''Calculate the length scale for the trap
    
    Params
        ν : float
            trap frequency, in units of radians/sec
        M : float
            mass of ion, in units of kg
        Z : int
            degree of ionization (net charge on ion)
        
    Returns
        float
        length scale in units of meters
    '''
    if M==None: M = con.atomic_mass*39.9626
    if Z==None: Z = 1
    return ((Z**2*con.elementary_charge**2)/(4*π*con.epsilon_0*M*ν**2))**(1/3)

def makeRF0(m,q,w,Nr,Nz,Nrmid,Nzmid,dr): 
    # We take in the mass, frequency for that mass, cell numbers, midpoint, and physical width of a cell and output the RF electric fields (constant in z) as a function of radial cell
    C = m*(w**2)/q ; RFx = np.ones((Nr,Nr,Nz)); RFy = np.ones((Nr,Nr,Nz))
    for jCell in range(Nr):
        RFx[jCell,:,:] = RFx[jCell,:,:]*C*(Nrmid-jCell)*dr# electric field in pseudo-potential and harmonic potential approximation
    for iCell in range(Nr):
        RFy[:,iCell,:] = RFy[:,iCell,:]*C*(Nrmid-iCell)*dr
    return RFx, RFy

def makeDC(m,q,w,Nz,Nr,Nzmid,dz): 
    C = m*(w**2)/q ; DC = np.ones((Nr,Nr,Nz))
    for jCell in range(Nr):
        for iCell in range(Nr):
            for kCell in range(Nz):
                DC[jCell,iCell,kCell] = DC[jCell,iCell,kCell]*C*(Nzmid-kCell)*dz # electric field for DC in harmonic potential approximation 
    return DC

def makeVf(Ni,q,m,wr,l=0,offsetx=0,offsety=0,offsetz=0,vbumpx=0,vbumpy=0,vbumpz=0):
    '''
    [x-position,y-position,axial(z)-position,x-velocity,y-velocity,z-velocity,charge,mass,polarizability]
    '''
    vf = np.zeros((Ni,9))
    pos = calcPositions(Ni); lscale = lengthScale(wr); scaledPos = pos*lscale
    for i in range(Ni):
        vf[i,:] = [0.0e-6,0.0e-6,-scaledPos[i],0,0,0,q,m,0.0]
    vf[l,0] += offsetx ; vf[l,1] += offsety; vf[l,2] += offsetz
    vf[l,3] += vbumpx ; vf[l,4] += vbumpy; vf[l,5] += vbumpz
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
    return numba.float64(pos / dcell + Nmid)

@cuda.jit(device=True)
def vtopPos_cuda(pos, Nmid, dcell):
    return numba.float64((pos-Nmid)*dcell)

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
    Ni = current_ion
    Nc = vc.shape[0]
    for i in range(Ni):
        for j in range(i + 1, Ni):
            x = vf[i, 0] - vf[j, 0]
            y = vf[i, 1] - vf[j, 1]
            z = vf[i, 2] - vf[j, 2]
            vx = vf[i, 3] - vf[j, 3]
            vy = vf[i, 4] - vf[j, 4]
            vz = vf[i, 5] - vf[j, 5]
            dist2 = x * x + y * y + z * z
            v2 = vx * vx + vy * vy + vz * vz
            if dist2 < rii2:
                vii2 = v2
                rii2 = dist2
        for j in range(Nc):
            x = vf[i, 0] - vc[j, 0]
            y = vf[i, 1] - vc[j, 1]
            z = vf[i, 2] - vc[j, 2]
            vx = vf[i, 3] - vc[j, 3]
            vy = vf[i, 4] - vc[j, 4]
            vz = vf[i, 5] - vc[j, 5]
            dist2 = x * x + y * y + z * z
            v2 = vx * vx + vy * vy + vz * vz
            if dist2 < rid2:
                vid2 = v2
                rid2 = dist2

    return math.sqrt(rid2), math.sqrt(rii2), math.sqrt(vid2), math.sqrt(vii2)

@cuda.jit(device=True)
def collisionMode_cuda(rii, rid, a, e=0.1):
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
def confirmValid(vf,dr,dz,Nr,Nz,Nrmid,Nzmid):
    
    Ni = current_ion

    for i in range(Ni):
        xCell = ptovPos_cuda(vf[i,0],Nrmid,dr) ; yCell = ptovPos_cuda(vf[i,1],Nrmid,dr) ; zCell = ptovPos_cuda(vf[i,2],Nzmid,dz) 
        if (xCell > Nr - 2 or xCell < 1 or 
            yCell > Nr - 2 or yCell < 1 or 
            zCell > Nz - 2 or zCell < 1):
            # Mark ion for ejection properly
            # for j in range(9):
            #     vf[i, j] = 0.0
            vf[i, 0] = 2.0
            vf[i, 1] = 2.0
            vf[i, 2] = 2.0
            vf[i, 3] = 0.0
            vf[i, 4] = 0.0
            vf[i, 5] = 0.0
            vf[i, 7] = 1e6
    return vf

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

    Ni = current_ion
    
    for i in range(Ni):

        vf[i, 0] += vf[i, 3] * dt
        vf[i, 1] += vf[i, 4] * dt
        vf[i, 2] += vf[i, 5] * dt
        confirmValid(vf, dr, dz, Nr, Nz, Nrmid, Nzmid)

@cuda.jit(device=True)
def updateVels_cuda(vf, Exf, Eyf, Ezf, dt):
    """
    Updates velocities of ions based on forces.
    CUDA device function for velocity evolution from field forces.
    
    Args:
        vf (array): Ion positions and velocities
        Erf, Ezf (array): Electric field components
        dt (float): Time step
    """

    Ni = current_ion
    for i in range(Ni):
        # Compute force on ion i
        Fx = vf[i, 6] * Exf[i]  # Fr = q * E_r
        Fy = vf[i, 6] * Eyf[i]
        Fz = vf[i, 6] * Ezf[i]  # Fz = q * E_z

        vf[i, 3] += Fx*dt / (vf[i,7])
        vf[i, 4] += Fy*dt / (vf[i,7])
        vf[i, 5] += Fz*dt / (vf[i,7])

@cuda.jit(device=True)
def solveFields_cuda(vf, Fx, Fy, Fz, Nrmid, Nzmid, Ni, dr, dz, Exf2, Eyf2, Ezf2):
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
    C1 = 4*np.pi*eps0 #SI units

    Nr = Fx.shape[0]
    Nz = Fz.shape[0]

    # for i in range(Ni):
    #     Erf2[i] = 0.0
    #     Ezf2[i] = 0.0

    for i in range(Ni):
        jCell = int(round(ptovPos_cuda(vf[i,0],Nrmid,dr)))
        iCell = int(round(ptovPos_cuda(vf[i,1],Nrmid,dr)))
        kCell = int(round(ptovPos_cuda(vf[i,2],Nzmid,dz)))

        # Bounds checking
        if jCell>Nr-2 or jCell<1 or iCell>Nr-2 or iCell<1 or kCell>Nz-2 or kCell<1:
            continue  # Skip this ion if out of bounds

        # Add background trap fields
        Exf2[i] += Fx[jCell,iCell,kCell]
        Ezf2[i] += Fz[jCell,iCell,kCell]
        Eyf2[i] += Fy[jCell,iCell,kCell]

        # Add contributions from other ions
        for j in range(Ni):
            if j != i:
                xdist = (vf[j,0]-vf[i,0]) ; ydist = (vf[j,1]-vf[i,1]) ; zdist = (vf[j,2]-vf[i,2])
                sqDist = xdist**2 + ydist**2 + zdist**2
                dist = math.sqrt(sqDist) #+ 1e-20  # Avoid division by zero
                projX = xdist / dist
                projY = ydist / dist
                projZ = zdist / dist

                Exf2[i] += -projX * vf[j, 6] / (C1 * sqDist)
                Eyf2[i] += -projY * vf[j, 6] / (C1 * sqDist)
                Ezf2[i] += -projZ * vf[j, 6] / (C1 * sqDist)

@cuda.jit(device=True)
def collisionParticlesFields_cuda(vf, vc, Ni, RFx, RFy, DCz, dr, dz, dtNow, Nrmid, Nzmid, Exfi, Eyfi, Ezfi, erfc_debug, ezfc_debug):
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

    MAX_NC = 1
    MAX_NI = current_ion
    Nc = vc.shape[0]  # Actual number used in loops
    
    # Local arrays use maximum sizes
    Exfc = cuda.local.array((MAX_NC, 2), dtype=float64)
    Eyfc = cuda.local.array((MAX_NC, 2), dtype=float64)
    Ezfc = cuda.local.array((MAX_NC, 2), dtype=float64)

    sqDist = cuda.local.array((MAX_NC, MAX_NI), dtype=float64)
    dist_j = cuda.local.array((MAX_NC, MAX_NI), dtype=float64)

    projX_j = cuda.local.array((MAX_NC, MAX_NI), dtype=float64)
    projY_j = cuda.local.array((MAX_NC, MAX_NI), dtype=float64)
    projZ_j = cuda.local.array((MAX_NC, MAX_NI), dtype=float64)
    sqDist_j_1p5 = cuda.local.array((MAX_NC, MAX_NI), dtype=float64)

    pX = cuda.local.array(MAX_NC, dtype=float64)
    pY = cuda.local.array(MAX_NC, dtype=float64)
    pZ = cuda.local.array(MAX_NC, dtype=float64)

    for i in range(Nc):

        jCell = int(round(ptovPos_cuda(vf[i,0],Nrmid,dr)))
        iCell = int(round(ptovPos_cuda(vf[i,1],Nrmid,dr)))
        kCell = int(round(ptovPos_cuda(vf[i,2],Nzmid,dz)))

        Exfc[i,0] += (RFx[jCell,iCell,kCell])
        Eyfc[i,0] += (RFy[jCell,iCell,kCell])
        Ezfc[i,0] += (DCz[jCell,iCell,kCell])

        Exfc[i,1] += (RFx[jCell+1,iCell,kCell])-(RFx[jCell-1,iCell,kCell])/dr
        Eyfc[i,1] += (RFy[jCell,iCell+1,kCell])-(RFy[jCell,iCell-1,kCell])/dr
        Ezfc[i,1] += (DCz[jCell,iCell,kCell+1])-(DCz[jCell,iCell,kCell-1])/dz

        for j in range(Ni):
            xdist = (vf[j,0]-vc[i,0]) ; ydist = (vf[j,1]-vc[i,1]) ; zdist = (vf[j,2]-vc[i,2])

            sqDist[i, j] = xdist * xdist + ydist * ydist + zdist * zdist

            dist_j[i, j] = math.sqrt(sqDist[i, j])

            projX_j[i, j] = xdist / dist_j[i, j]
            projY_j[i, j] = ydist / dist_j[i, j]
            projZ_j[i, j] = zdist / dist_j[i, j]

            Exfc[i,0] += -projX_j[i,j]*vf[j,6]/(C1*sqDist[i,j])
            Eyfc[i,0] += -projY_j[i,j]*vf[j,6]/(C1*sqDist[i,j])
            Ezfc[i,0] += -projZ_j[i,j]*vf[j,6]/(C1*sqDist[i,j])

            sqDist_j_1p5[i, j] = sqDist[i, j] * dist_j[i, j]

            Exfc[i, 1] += 2 * projX_j[i, j] * vf[j, 6] / (C1 * sqDist_j_1p5[i, j])
            Eyfc[i, 1] += 2 * projY_j[i, j] * vf[j, 6] / (C1 * sqDist_j_1p5[i, j])
            Ezfc[i, 1] += 2 * projZ_j[i, j] * vf[j, 6] / (C1 * sqDist_j_1p5[i, j])

        # erfc_debug[0] = Erfc[i, 0]
        # ezfc_debug[1] = Ezfc[i, 1]

    for k in range(Nc):
        if vc[k,6] != 0.0:
            pX[k] = -2 * math.pi * eps0 * vc[k, 8] * Exfc[k, 0]
            pY[k] = -2 * math.pi * eps0 * vc[k, 8] * Exfc[k, 0]
            pZ[k] = -2 * math.pi * eps0 * vc[k, 8] * Ezfc[k, 0]

            Fx = math.fabs(pX[k]) * Exfc[k, 1]
            Fy = math.fabs(pY[k]) * Exfc[k, 1]
            Fz = math.fabs(pZ[k]) * Ezfc[k, 1]

            vc[k, 3] += Fx * dtNow / vc[k, 7]
            vc[k, 4] += Fy * dtNow / vc[k, 7]
            vc[k, 5] += Fz * dtNow / vc[k, 7]
    
    dist1 = 0

    for l in range(Ni):
        for m in range(Nc):        
            if vc[m,6]!=0.0:

                Rhatx_j = projX_j[m, l]
                Rhaty_j = projY_j[m, l]
                Rhatz_j = projZ_j[m, l]

                dist1 = dist_j[m, l]

                Exfi[l] += -math.fabs(pX[m]) * (2 * Rhatx_j) / (C1 * dist1 ** 3)
                Eyfi[l] += -math.fabs(pY[m]) * (2 * Rhaty_j) / (C1 * dist1 ** 3)
                Ezfi[l] += -math.fabs(pZ[m]) * (2 * Rhatz_j) / (C1 * dist1 ** 3)

@cuda.jit(device=True)
def mcCollision_cuda(vf, xc, yc, zc, vxc, vyc, vzc, qc, mc, ac, Nt, dtSmall, 
                    dtCollision, RFx, RFy, DC, Nr, Nz, dr, dz, dtLarge, 
                    reorder, ion_positions, ion_velocities, cell_positions, 
                    ejection_info, withCollision):
    """
    Main Monte Carlo collision simulation function.
    Handles variable number of ions up to MAX_IONS.
    """
    # Initialize with explicit typing
    Nr = int32(Nr)
    Nz = int32(Nz)
    Nrmid = float64((Nr - 1) / 2.0)
    Nzmid = float64((Nz - 1) / 2.0)
    Ni = int32(vf.shape[0])  # Get actual number of ions from array
    Nc = int32(1)
    
    sample_interval = 1 #// 100  # Store every N steps
    sample_index = 0

    # Local arrays allocated for maximum size
    vc = cuda.local.array((1, 9), dtype=float64)
    Exfi = cuda.local.array(MAX_IONS, dtype=float64)
    Eyfi = cuda.local.array(MAX_IONS, dtype=float64)
    Ezfi = cuda.local.array(MAX_IONS, dtype=float64)
    
    # Initialize collision particle data
    vc[0, 0] = float64(xc)
    vc[0, 1] = float64(yc)
    vc[0, 2] = float64(zc)
    vc[0, 3] = float64(vxc)
    vc[0, 4] = float64(vyc)
    vc[0, 5] = float64(vzc)
    vc[0, 6] = float64(qc)
    vc[0, 7] = float64(mc)
    vc[0, 8] = float64(ac)

    # Initialize local variables
    dtNow = float64(dtSmall)
    erfc_local = cuda.local.array(2, dtype=float64)
    ezfc_local = cuda.local.array(2, dtype=float64)

    # Clear arrays
    # for i in range(MAX_IONS):
    #     Exfi[i] = 0.0
    #     Eyfi[i] = 0.0
    #     Ezfi[i] = 0.0

    crossTest = 0

    confirmValid(vc, dr, dz, Nr, Nz, Nrmid, Nzmid)

    for i in range(Nt):
        rid, rii, vid, vii = minDists_cuda(vf, vc)
        
        # Avoid division by zero
        if vid < 1e-10:
            vid = 1e-10
            
        collision = collisionMode_cuda(rii, rid, vc[0,8], 0.3)
        
        if collision:
            dtNow = float64(rid * 0.01 / (5.0 * vid))
        else:
            dtNow = float64(dtSmall)
        
        if dtNow < dtCollision:
            dtNow = float64(dtCollision)

        # Clear field arrays before calculation
        for j in range(Ni):
            Exfi[j] = 0.0
            Eyfi[j] = 0.0
            Ezfi[j] = 0.0

        solveFields_cuda(vf, RFx, RFy, DC, Nrmid, Nzmid, Ni, dr, dz, Exfi, Eyfi, Ezfi)

        if withCollision:
            if vc[0, 7] < 1e6:
                collisionParticlesFields_cuda(vf, vc, Ni, RFx, RFy, DC, dr, dz, dtNow, 
                                            Nrmid, Nzmid, Exfi, Eyfi, Ezfi, 
                                            erfc_local, ezfc_local)
                updatePoss_cuda(vc, dr, dz, dtNow, Nr, Nz, Nrmid, Nzmid)
            else:
                dtNow = float64(dtLarge)

        updateVels_cuda(vf, Exfi, Eyfi, Ezfi, dtNow)
        updatePoss_cuda(vf, dr, dz, dtNow, Nr, Nz, Nrmid, Nzmid)

        if i % sample_interval == 0 and sample_index < 100:
            for ion in range(Ni):
                # Store positions
                ion_positions[ion,0,sample_index] = vf[ion,0]
                ion_positions[ion,1,sample_index] = vf[ion,1]
                ion_positions[ion,2,sample_index] = vf[ion,2]
                
                # Store velocities
                ion_velocities[ion,0,sample_index] = vf[ion,3]
                ion_velocities[ion,1,sample_index] = vf[ion,4]
                ion_velocities[ion,2,sample_index] = vf[ion,5]
                
                # Store cell coordinates
                xCell = ptovPos_cuda(vf[ion,0],Nrmid,dr)
                yCell = ptovPos_cuda(vf[ion,1],Nrmid,dr)
                zCell = ptovPos_cuda(vf[ion,2],Nzmid,dz)
                cell_positions[ion,0,sample_index] = xCell
                cell_positions[ion,1,sample_index] = yCell
                cell_positions[ion,2,sample_index] = zCell
            
            sample_index += 1
        
        # Check for ion ejection
        mass_sum = float64(0.0)
        for j in range(Ni):
            mass_sum += vf[j, 7]
        if mass_sum > 1e5:
            # Store ejection information
            ejection_info[0] = float64(i)  # timestep
            ejection_info[1] = float64(j)  # ion index
            for k in range(3):
                ejection_info[k+2] = vf[j,k]  # position
            reorder[0] = int32(2)
            break

        for j in range(1, Ni):
            if vf[j, 2] > vf[j-1, 2] and crossTest<1:
                reorder[0] = int32(1)
                crossTest = int32(crossTest + 1)
                Nt = min(Nt, i + 1000)
                break

        if reorder[0] != 0:
            break

@cuda.jit
def mcCollision_kernel(vf_all, xc_all, yc_all, zc_all, vxc_all, vyc_all, vzc_all, 
                      qc, mc, ac, Nt_all, dtSmall, dtCollision, RFx, RFy, DC, 
                      Nr, Nz, dr, dz, dtLarge, reorder_all, ion_positions_all, 
                      ion_velocities_all, cell_positions_all, ejection_info_all, withCollision):
    """
    CUDA kernel for parallel collision simulations.
    Handles variable number of ions up to MAX_IONS.
    """
    idx = cuda.grid(1)
    
    if idx < vf_all.shape[0]:
        # Local array allocation for maximum size
        vf = cuda.local.array((current_ion, 9), dtype=float64)
        reorder = cuda.local.array(1, dtype=int32)
        
        # Initialize arrays
        reorder[0] = 0
        
        # Copy data with bounds checking
        actual_ions = min(vf_all.shape[1], MAX_IONS)
        for i in range(actual_ions):
            for j in range(9):
                vf[i, j] = vf_all[idx, i, j]
        
        # Zero out any unused ion slots
        for i in range(actual_ions, current_ion):
            for j in range(9):
                vf[i, j] = 0.0
        
        # Get scalar values
        xc = float64(xc_all[idx])
        yc = float64(yc_all[idx])
        zc = float64(zc_all[idx])
        vxc = float64(vxc_all[idx])
        vyc = float64(vyc_all[idx])
        vzc = float64(vzc_all[idx])
        Nt = int32(Nt_all[idx])
        
        # Call device function
        mcCollision_cuda(
            vf, xc, yc, zc, vxc, vyc, vzc,
            float64(qc), float64(mc), float64(ac),
            Nt, float64(dtSmall), float64(dtCollision),
            RFx, RFy, DC,
            int32(Nr), int32(Nz),
            float64(dr), float64(dz),
            float64(dtLarge),
            reorder,
            ion_positions_all[idx],
            ion_velocities_all[idx],
            cell_positions_all[idx],
            ejection_info_all[idx],
            withCollision
        )
        
        # Copy results back
        reorder_all[idx] = int32(reorder[0])
        for i in range(actual_ions):
            for j in range(9):
                vf_all[idx, i, j] = float64(vf[i, j])