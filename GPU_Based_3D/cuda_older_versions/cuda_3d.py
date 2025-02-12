import numpy as np
from scipy.optimize import fsolve
import scipy.constants as con
from numba import cuda, float64, int32
import math
from scipy import special
import time
import gc
import traceback
from math import exp, pi as π

def ion_position_potential(x):
    """
    Potential energy of the ion string as a function of the positions of the ions
    """
    N = len(x)
    potential = np.zeros(N)
    for m in range(N):
        sum1 = sum([1 / (x[m] - x[n]) ** 2 for n in range(m)])
        sum2 = sum([1 / (x[m] - x[n]) ** 2 for n in range(m + 1, N)])
        potential[m] = x[m] - sum1 + sum2
    return potential

def calcPositions(N):
    """
    Calculate the equilibrium ion positions
    """
    estimated_extreme = 0.481 * N ** 0.765  # Empirical estimate
    initial_guess = np.linspace(-estimated_extreme, estimated_extreme, N)
    positions = fsolve(ion_position_potential, initial_guess)
    return positions

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

def makeRF0(m, q, w, Nr, Nz, Nrmid, Nzmid, dr):
    """
    Generate the RF field in the grid
    """
    RFx = np.zeros((Nr, Nr, Nz), dtype=np.float64)
    RFy = np.zeros((Nr, Nr, Nz), dtype=np.float64)
    C = m * (w ** 2) / q
    for jCell in range(Nr):
        for iCell in range(Nr):
            for kCell in range(Nz):
                RFx[jCell, iCell, kCell] = C * (Nrmid - jCell) * dr
                RFy[jCell, iCell, kCell] = C * (Nrmid - iCell) * dr
    return RFx, RFy

def makeDC(m, q, w, Nz, Nr, Nzmid, dz):
    """
    Generate the DC field in the grid
    """
    DC = np.zeros((Nr, Nr, Nz), dtype=np.float64)
    C = m * (w ** 2) / q
    for jCell in range(Nr):
        for iCell in range(Nr):
            for kCell in range(Nz):
                DC[jCell, iCell, kCell] = C * (Nzmid - kCell) * dz
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


#CUDA FUNCTIONS

@cuda.jit(device=True)
def ptovPos(pos, Nmid, dcell):
    """
    Converts from physical to virtual units in position
    """
    return pos / dcell + Nmid

@cuda.jit(device=True)
def vtopPos(pos, Nmid, dcell):
    """
    Converts from virtual to physical units in position
    """
    return (pos - Nmid) * dcell

@cuda.jit(device=True)
def minDists(vf, vc):
    """
    Calculate minimum distances and velocities between particles
    """
    rid2 = 1e6
    rii2 = 1e6
    vid2 = 1e6
    vii2 = 1e6
    
    Ni = MAX_NI
    Nc = MAX_NC

    for i in range(Ni):
        for j in range(i + 1, Ni):
            x = vf[i, 0] - vf[j, 0]
            y = vf[i, 1] - vf[j, 1]
            z = vf[i, 2] - vf[j, 2]
            vx = vf[i, 3] - vf[j, 3]
            vy = vf[i, 4] - vf[j, 4]
            vz = vf[i, 5] - vf[j, 5]
            dist2 = x ** 2 + y ** 2 + z ** 2
            v2 = vx ** 2 + vy ** 2 + vz ** 2
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
            dist2 = x ** 2 + y ** 2 + z ** 2
            v2 = vx ** 2 + vy ** 2 + vz ** 2
            if dist2 < rid2:
                vid2 = v2
                rid2 = dist2
    
    return math.sqrt(rid2), math.sqrt(rii2), math.sqrt(vid2), math.sqrt(vii2)            

@cuda.jit(device=True)
def collisionMode(rii, rid, a, e=0.1):
    """
    Determine if collision mode should be activated
    """
    return (a * rii * rii) / (rid ** 5) > e

@cuda.jit(device=True)
def confirmValid(vf, dr, dz, Nr, Nz, Nrmid, Nzmid):
    """Updated boundary condition handling"""
    Ni = MAX_NI
    
    for i in range(Ni):
        xCell = ptovPos(vf[i, 0], Nrmid, dr)
        yCell = ptovPos(vf[i, 1], Nrmid, dr)
        zCell = ptovPos(vf[i, 2], Nzmid, dz)
        
        if (xCell > Nr - 2 or xCell < 1 or 
            yCell > Nr - 2 or yCell < 1 or 
            zCell > Nz - 2 or zCell < 1):
            # Mark ion for ejection properly
            vf[i, 0] = 2.0
            vf[i, 1] = 2.0
            vf[i, 2] = 2.0
            vf[i, 3] = 0.0
            vf[i, 4] = 0.0
            vf[i, 5] = 0.0
            vf[i, 7] = 1e6

@cuda.jit(device=True)
def updatePoss(vf, dr, dz, dt, Nr, Nz, Nrmid, Nzmid):
    """
    Update particle positions based on velocities
    """
    Ni = MAX_NI

    for i in range(Ni):
        vf[i, 0] += vf[i, 3] * dt
        vf[i, 1] += vf[i, 4] * dt
        vf[i, 2] += vf[i, 5] * dt
    confirmValid(vf, dr, dz, Nr, Nz, Nrmid, Nzmid)

@cuda.jit(device=True)
def updateVels(vf, Exf, Eyf, Ezf, dt):
    """
    Update particle velocities based on forces
    """
    Ni = MAX_NI

    for i in range(Ni):
        Fx = vf[i, 6] * Exf[i]
        Fy = vf[i, 6] * Eyf[i]
        Fz = vf[i, 6] * Ezf[i]
        vf[i, 3] += Fx * dt / vf[i, 7]
        vf[i, 4] += Fy * dt / vf[i, 7]
        vf[i, 5] += Fz * dt / vf[i, 7]

@cuda.jit(device=True)
def solveFields(vf, Fx, Fy, Fz, Nrmid, Nzmid, Ni, dr, dz, Exf2, Eyf2, Ezf2):
    """
    Solve for the electric fields at each ion
    """
    eps0 = 8.854e-12
    C1 = 4 * math.pi * eps0

    for i in range(Ni):
        jCell = int(ptovPos(vf[i, 0], Nrmid, dr))
        iCell = int(ptovPos(vf[i, 1], Nrmid, dr))
        kCell = int(ptovPos(vf[i, 2], Nzmid, dz))

        Exf2[i] += Fx[jCell, iCell, kCell]
        Eyf2[i] += Fy[jCell, iCell, kCell]
        Ezf2[i] += Fz[jCell, iCell, kCell]

        for j in range(Ni):
            if j != i:
                xdist = vf[j, 0] - vf[i, 0]
                ydist = vf[j, 1] - vf[i, 1]
                zdist = vf[j, 2] - vf[i, 2]
                sqDist = xdist ** 2 + ydist ** 2 + zdist ** 2
                dist = math.sqrt(sqDist) + 1e-20
                projX = xdist / dist
                projY = ydist / dist
                projZ = zdist / dist
                Exf2[i] += -projX * vf[j, 6] / (C1 * sqDist)
                Eyf2[i] += -projY * vf[j, 6] / (C1 * sqDist)
                Ezf2[i] += -projZ * vf[j, 6] / (C1 * sqDist)

@cuda.jit(device=True)
def collisionParticlesFields(vf, vc, Ni, RFx, RFy, DCz, dr, dz, dt, Nrmid, Nzmid, Exfi, Eyfi, Ezfi):
    """
    Calculate fields from collision particles and update their velocities
    """
    eps0 = 8.854e-12
    Nc = MAX_NC
    C1 = 4 * math.pi * eps0

    Exfc = cuda.local.array((Nc, 2), dtype=float64)
    Eyfc = cuda.local.array((Nc, 2), dtype=float64)
    Ezfc = cuda.local.array((Nc, 2), dtype=float64)
    sqDist = cuda.local.array((Nc, Ni), dtype=float64)
    projX = cuda.local.array((Nc, Ni), dtype=float64)
    projY = cuda.local.array((Nc, Ni), dtype=float64)
    projZ = cuda.local.array((Nc, Ni), dtype=float64)

    for i in range(Nc):

        jCell = int(ptovPos(vc[i, 0], Nrmid, dr))
        iCell = int(ptovPos(vc[i, 1], Nrmid, dr))
        kCell = int(ptovPos(vc[i, 2], Nzmid, dz))

        Exfc[i, 0] += RFx[jCell, iCell, kCell]
        Eyfc[i, 0] += RFy[jCell, iCell, kCell]
        Ezfc[i, 0] += DCz[jCell, iCell, kCell]

        Exfc[i, 1] += (RFx[jCell+1, iCell, kCell]) - (RFx[jCell - 1, iCell, kCell])/dr
        Eyfc[i, 1] += (RFy[jCell, iCell+1, kCell] ) - (RFy[jCell, iCell-1, kCell] )/dr
        Ezfc[i, 1] += (DCz[jCell, iCell, kCell+1]) - (DCz[jCell, iCell, kCell-1])/dz

        for j in range(Ni):
            xdist = vf[j, 0] - vc[i, 0]
            ydist = vf[j, 1] - vc[i, 1]
            zdist = vf[j, 2] - vc[i, 2]

            sqDist[i, j] = xdist ** 2 + ydist ** 2 + zdist ** 2 + 1e-20
            dist = math.sqrt(sqDist[i, j])

            projX[i, j] = xdist / dist
            projY[i, j] = ydist / dist
            projZ[i, j] = zdist / dist

            Exfc[i, 0] += -projX[i, j] * vf[j, 6] / (C1 * sqDist[i, j])
            Eyfc[i, 0] += -projY[i, j] * vf[j, 6] / (C1 * sqDist[i, j])
            Ezfc[i, 0] += -projZ[i, j] * vf[j, 6] / (C1 * sqDist[i, j])

            Exfc[i, 1] += 2 * projX[i, j] * vf[j, 6] / (C1 * sqDist[i, j] ** 1.5)
            Eyfc[i, 1] += 2 * projY[i, j] * vf[j, 6] / (C1 * sqDist[i, j] ** 1.5)
            Ezfc[i, 1] += 2 * projZ[i, j] * vf[j, 6] / (C1 * sqDist[i, j] ** 1.5)

    pX = cuda.local.array(Nc, dtype=float64)
    pY = cuda.local.array(Nc, dtype=float64)
    pZ = cuda.local.array(Nc, dtype=float64)

    for i in range(Nc):
        if vc[i, 6] != 0.0:
            pX[i] = -2 * math.pi * eps0 * vc[i, 8] * Exfc[i, 0]
            pY[i] = -2 * math.pi * eps0 * vc[i, 8] * Eyfc[i, 0]
            pZ[i] = -2 * math.pi * eps0 * vc[i, 8] * Ezfc[i, 0]

            Fx = abs(pX[i]) * Exfc[i, 1]
            Fy = abs(pY[i]) * Eyfc[i, 1]
            Fz = abs(pZ[i]) * Ezfc[i, 1]

            vc[i, 3] += Fx * dt / vc[i, 7]
            vc[i, 4] += Fy * dt / vc[i, 7]
            vc[i, 5] += Fz * dt / vc[i, 7]

    for i in range(Ni):
        for j in range(Nc):
            if vc[j, 6] != 0.0:
                dist = math.sqrt(sqDist[j, i])
                Exfi[i] += -abs(pX[j]) * (2 * projX[j, i]) / (C1 * dist ** 3)
                Eyfi[i] += -abs(pY[j]) * (2 * projY[j, i]) / (C1 * dist ** 3)
                Ezfi[i] += -abs(pZ[j]) * (2 * projZ[j, i]) / (C1 * dist ** 3)

@cuda.jit(device=True)
def runFasterCollision(vf, xc, yc, zc, vxc, vyc, vzc, qc, mc, ac, Nt, dtSmall, dtCollision, RFx, RFy, DC, Nr, Nz, dr, dz, dtLarge, reorder, withCollision = True):
    """
    CUDA kernel to run the collision simulation
    """
    reorder[0] = 0
    Nrmid = (Nr - 1) / 2
    Nzmid = (Nz - 1) / 2
    Ni = MAX_NI
    Nc = MAX_NC
    crossTest = 0

    # Initialize collisional particle
    vc = cuda.local.array((1, 9), dtype=float64)
    vc[0, 0] = xc
    vc[0, 1] = yc
    vc[0, 2] = zc
    vc[0, 3] = vxc
    vc[0, 4] = vyc
    vc[0, 5] = vzc
    vc[0, 6] = qc
    vc[0, 7] = mc
    vc[0, 8] = ac

    confirmValid(vc, dr, dz, Nr, Nz, Nrmid, Nzmid)

    dtNow = dtSmall
    Exfi = cuda.local.array((Ni,), dtype=float64)
    Eyfi = cuda.local.array((Ni,), dtype=float64)
    Ezfi = cuda.local.array((Ni,), dtype=float64)

    for i in range(Nt):

        rid,rii,vid,vii = minDists(vf, vc)
        collision = collisionMode(rii, rid, vc[0, 8], 0.3)

        if collision:
            dtNow = rid * 0.01 / (5 * vid)
        else:
            dtNow = dtSmall
        if dtNow < dtCollision:
            dtNow = dtCollision

        solveFields(vf, RFx, RFy, DC, Nrmid, Nzmid, Ni, dr, dz, Exfi, Eyfi, Ezfi)

        if withCollision==True:
            if vc[0,7] < 1e6: #if the collisional particle exists
                dtNow = dtSmall
                
                collisionParticlesFields(vf, vc, Ni, RFx, RFy, DC, dr, dz, dtNow, Nrmid, Nzmid, Exfi, Eyfi, Ezfi)
                updatePoss(vc,dr,dz,dtNow,Nr,Nz,Nrmid,Nzmid) # updates collisional particle positions

            else:
                dtNow = dtLarge
        else:
            pass

        updateVels(vf, Exfi, Eyfi, Ezfi, dtNow)
        updatePoss(vf, dr, dz, dtNow, Nr, Nz, Nrmid, Nzmid)

        mass_sum = 0.0
        for j in range(Ni):
            mass_sum += vf[j, 7]
        if mass_sum > 1e5:
            reorder[0] += 2
            break

        for j in range(1, Ni):
            if vf[j, 2] > vf[j - 1, 2] and crossTest < 1:
                reorder[0] += 1
                crossTest += 1
                Nt = min(Nt, i + 1000)
                break

        if reorder[0] != 0:
            break

@cuda.jit
def runFasterCollision_kernel(vf_all, xc_all, yc_all, zc_all, vxc_all, vyc_all, vzc_all, qc, mc, ac, Nt_all, dtSmall, dtCollision, RFx, RFy, DC, Nr, Nz, dr, dz, dtLarge, reorder_all, withCollision):
    """
    CUDA kernel function to run the collision simulation for multiple instances in parallel.
    """
    idx = cuda.grid(1)
    if idx < vf_all.shape[0]:  # Bounds check
        # Initialize local arrays for this simulation instance
        vf = cuda.local.array((MAX_NI, 9), dtype=float64)  # Fixed size for MAX_NI = 2

        # Copy input data to local arrays
        for i in range(MAX_NI):
            for j in range(9):
                vf[i, j] = vf_all[idx, i, j]

        # Get collision parameters for this instance
        xc = xc_all[idx]
        yc = yc_all[idx]
        zc = zc_all[idx]
        vxc = vxc_all[idx]
        vyc = vyc_all[idx]
        vzc = vzc_all[idx]
        Nt = Nt_all[idx]

        reorder = cuda.local.array((1,), dtype=int32)
        reorder[0] = 0

        # Run simulation
        runFasterCollision(vf, xc, yc, zc, vxc, vyc, vzc, qc, mc, ac, Nt, 
                          dtSmall, dtCollision, RFx, RFy, DC, Nr, Nz, dr, dz, 
                          dtLarge, reorder, withCollision)

        # Copy results back to global memory
        reorder_all[idx] = reorder[0]

        for i in range(MAX_NI):
            for j in range(9):
                vf_all[idx, i, j] = vf[i, j]

def Boltz(m,T,vmin=0,vmax=5000,bins=100):
    amu = 1.66*10**-27
    m = m*amu
    k = 1.386e-23 # boltzmann constant
    boltz = np.zeros(bins) # initialize vector
    dv = (vmax - vmin)/bins # define bin spacing in speed
    a = (k*T/m)**(1/2) # normalization constant for distribution function

    
    for i in range(bins):
        vhere = vmin + i*dv # define speed of bin
        vlast = vhere-dv
        boltz[i] = (special.erf(vhere/(a*np.sqrt(2))) - np.sqrt(2/np.pi)*(vhere/a)*np.exp(-vhere**2/(2*a**2)) ) - (special.erf(vlast/(a*np.sqrt(2))) - np.sqrt(2/np.pi)*(vlast/a)*np.exp(-vlast**2/(2*a**2)) ) # here we use the cumulative distribution function and subtract the one-step down value from the this step value for the probability density in this slice
    
    return boltz/np.sum(boltz)

MAX_NI = 2
MAX_NC = 1

# Constants
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

# Simulation parameters
# Dr = 7.515e-07
# Dz = 9.044999999999999e-05
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

grid_sizes = [20001]  # Array of grid sizes to test
ion_counts = [2]  # Array of ion counts to test
shot_sizes = [1]  # Array of shot sizes to test

def main():
    """
    Main function with corrected pinned memory handling
    """
    # Initialize CUDA context
    cuda.select_device(0)
    
    try:
        # Basic parameters
        Nr = 501
        Nz = 201
        Nrmid = (Nr - 1) / 2
        Nzmid = (Nz - 1) / 2
        Dr = Nr*1.5e-9 ; Dz = Nz*4.5e-7
        dr = Dr / float(Nr)
        dz = Dz / float(Nz)

        print(f"Starting simulation with grid size {Nr} x {Nz}")
        start_time = time.perf_counter()

        # Generate fields on CPU
        RFx, RFy = makeRF0(m, q, wr, Nr, Nz, Nrmid, Nzmid, dr)
        DC = makeDC(m, q, wz, Nz, Nr, Nzmid, dz)

        # print(RFx)
        # print(RFy)
        # print(DC)

        # Initial conditions
        vf = makeVf(MAX_NI, 1.0*q, m, wz)
        print(vf)
        
        # Test case parameters
        # velocity = 669.3693693693693
        # angle_choiceXY = -1.5390630676677268
        # angle_choiceZ = -1.2534637355232003
        # offset_choice = -6.13065326633166e-10
        # ion_collided = 0

        velocity =  1194.5945945945946
        angle_choiceXY = -0.015866629563584755
        angle_choiceZ = -0.30146596170811146
        offset_choice = -9.045226130653257e-11
        ion_collided = 0

        # velocity = 629.7297297297297
        # angle_choiceXY = -1.2851969946503699
        # angle_choiceZ = 0.4918655164711292
        # offset_choice = -7.738693467336684e-10
        # ion_collided = 1

        
        max_hypotenuse = 1.5e-7
        v = velocity
        x = -np.cos(angle_choiceZ)*max_hypotenuse 
        y = np.sin(angle_choiceXY)*x; x = np.cos(angle_choiceXY)*x
        vx = np.abs(velocity*np.cos(angle_choiceZ))
        vy = np.sin(angle_choiceXY)*vx; vx = np.cos(angle_choiceXY)*vx
        z = vf[ion_collided,2] + np.sin(angle_choiceZ)*max_hypotenuse + offset_choice
        vz=-1*velocity*np.sin(angle_choiceZ)

        # Create pinned memory arrays
        vf_pinned = cuda.pinned_array((1, MAX_NI, 9), dtype=np.float64)
        xc_pinned = cuda.pinned_array(1, dtype=np.float64)
        yc_pinned = cuda.pinned_array(1, dtype=np.float64)
        zc_pinned = cuda.pinned_array(1, dtype=np.float64)
        vxc_pinned = cuda.pinned_array(1, dtype=np.float64)
        vyc_pinned = cuda.pinned_array(1, dtype=np.float64)
        vzc_pinned = cuda.pinned_array(1, dtype=np.float64)
        Nt_pinned = cuda.pinned_array(1, dtype=np.int32)
        reorder_pinned = cuda.pinned_array(1, dtype=np.int32)

        try:
            # Copy data to pinned memory
            vf_pinned[0] = vf
            xc_pinned[0] = x
            yc_pinned[0] = y
            zc_pinned[0] = z
            vxc_pinned[0] = vx
            vyc_pinned[0] = vy
            vzc_pinned[0] = vz
            Nt_pinned[0] = 250000
            reorder_pinned[0] = 0

            # Create stream for operations
            stream = cuda.stream()
            
            try:
                # Allocate device memory
                vf_device = cuda.to_device(vf_pinned, stream)
                xc_device = cuda.to_device(xc_pinned, stream)
                yc_device = cuda.to_device(yc_pinned, stream)
                zc_device = cuda.to_device(zc_pinned, stream)
                vxc_device = cuda.to_device(vxc_pinned, stream)
                vyc_device = cuda.to_device(vyc_pinned, stream)
                vzc_device = cuda.to_device(vzc_pinned, stream)
                Nt_device = cuda.to_device(Nt_pinned, stream)
                reorder_device = cuda.to_device(reorder_pinned, stream)

                RFx_device = cuda.to_device(RFx, stream)
                RFy_device = cuda.to_device(RFy, stream)
                DC_device = cuda.to_device(DC, stream)

                # Launch kernel
                runFasterCollision_kernel[1, 1, stream](
                    vf_device, xc_device, yc_device, zc_device,
                    vxc_device, vyc_device, vzc_device,
                    q, mH2, aH2, Nt_device, dtSmall, dtCollision,
                    RFx_device, RFy_device, DC_device,
                    Nr, Nz, dr, dz, dtLarge, reorder_device, True
                )

                # Synchronize and copy result
                stream.synchronize()
                reorder = reorder_device.copy_to_host(stream=stream)
                stream.synchronize()

                finish_time = time.perf_counter()
                print(f"Simulation completed in {finish_time - start_time:.2f} seconds")
                print(f"Reorder status: {reorder}")
                
                return reorder[0]

            except cuda.cudadrv.driver.CudaAPIError as e:
                print(f"CUDA execution error: {str(e)}")
                return None
            
            finally:
                # Explicit cleanup of device memory
                stream.synchronize()
                del vf_device
                del xc_device
                del yc_device
                del zc_device
                del vxc_device
                del vyc_device
                del vzc_device
                del Nt_device
                del reorder_device
                del RFx_device
                del RFy_device
                del DC_device
                cuda.current_context().deallocations.clear()

        finally:
            # Free pinned memory
            del vf_pinned
            del xc_pinned
            del yc_pinned
            del zc_pinned
            del vxc_pinned
            del vyc_pinned
            del vzc_pinned
            del Nt_pinned
            del reorder_pinned

    except Exception as e:
        print(f"Error in main: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        # Reset CUDA context
        cuda.close()

if __name__ == "__main__":
    main()