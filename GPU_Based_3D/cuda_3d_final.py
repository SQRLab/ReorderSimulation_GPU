import numpy as np
from scipy.optimize import fsolve
import scipy.constants as con
import math
import numba
from numba import cuda, float64, int32
from scipy import special
from math import pi as π

# Constants and Configuration

#######################
###### IMPORTANT ######
#######################

# Current number of ions in the chain
current_ion = 2

# Nc represents the number of collision particles (currently fixed at 1)
Nc = 1

def ion_position_potential(x):
    '''
    Calculates the potential energy of an ion string based on Coulomb interactions
    Takes into account repulsive forces between ions and the trapping potential

    Physics:
    Calculates the potential energy of an ion string based on Coulomb interactions.
    The total potential includes:
    1. Harmonic trapping potential (x[m] term) - Models the quadratic RF/DC confinement
    2. Coulomb repulsion between ions (1/r^2 terms) - Long-range electrostatic force
    
    The equilibrium positions occur when the trapping force balances the Coulomb repulsion,
    leading to a crystal-like structure characteristic of trapped ion chains.
    
    Note: The forces are normalized to natural units where:
    - Distances are scaled by characteristic length l_0 = (e^2/4πε0mω^2)^(1/3)
    - Energies are scaled by mω^2l0^2
    This makes the equations dimensionless and numerically stable.
    
    Parameters:
        x (list): Positions of ions in normalized units
        
    Returns:
        list: Forces on each ion from combined Coulomb and trap potentials
    '''

    N = len(x)
    return [x[m] - sum([1/(x[m]-x[n])**2 for n in range(m)]) + sum([1/(x[m]-x[n])**2 for n in range(m+1,N)])
               for m in range(N)]

def calcPositions(N):
    '''
    Determines equilibrium positions of ions in the trap
    Uses numerical solver to find positions where forces balance
    
    Parameters:
        N (int): Number of ions to position
        
    Notes:
        - Uses empirical scaling for initial guess: 0.481*N^0.765
        - Formula valid for up to ~50 ions
    '''

    estimated_extreme = 0.481*N**0.765 # Hardcoded, should work for at least up to 50 ions
    return fsolve(ion_position_potential, np.linspace(-estimated_extreme, estimated_extreme, N))

def lengthScale(ν, M=None, Z=None):
    '''
    Calculates characteristic length scale of the ion trap
    Based on trap frequency, ion mass and charge
    
    Parameters:
        ν (float): Trap frequency in rad/s
        M (float, optional): Ion mass in kg, defaults to Ca-40
        Z (int, optional): Ion charge state, defaults to 1
        
    Returns:
        float: Characteristic length in meters
    '''

    if M==None: M = con.atomic_mass*39.9626
    if Z==None: Z = 1
    return ((Z**2*con.elementary_charge**2)/(4*π*con.epsilon_0*M*ν**2))**(1/3)

def makeRF0(m,q,w,Nr,Nz,Nrmid,Nzmid,dr): 

    '''
    Generates RF (radio frequency) electric field arrays
    Models the time-averaged pseudopotential in radial directions
    
    Parameters:
        m: Ion mass
        q: Ion charge
        w: RF frequency
        Nr,Nz: Grid dimensions
        Nrmid,Nzmid: Grid midpoints
        dr: Grid spacing
        
    Returns:
        tuple: (RFx, RFy) arrays giving radial RF field components
    '''

    # We take in the mass, frequency for that mass, cell numbers, midpoint, and physical width of a cell and output the RF electric fields (constant in z) as a function of radial cell
    C = - m*(w**2)/q ; RFx = np.ones((Nr,Nr,Nz)); RFy = np.ones((Nr,Nr,Nz))
    for jCell in range(Nr):
        RFx[jCell,:,:] = - RFx[jCell,:,:]*C*(Nrmid-jCell)*dr# electric field in pseudo-potential and harmonic potential approximation
    for iCell in range(Nr):
        RFy[:,iCell,:] = - RFy[:,iCell,:]*C*(Nrmid-iCell)*dr
    return RFx, RFy

# dummy function to generate the DC fields assuming that it is a harmonic potential about (0,0) and focuses in z
    # We take in the mass, frequency for that mass, cell numbers, midpoint, and physical width of a cell and output the DC electric fields (constant in r) as a function of longitudinal cell
def makeDC(m,q,w,Nz,Nr,Nzmid,dz): 

    '''
    Generates DC (static) electric field array
    Models axial confinement along z-direction
    
    Parameters:
        Similar to makeRF0
        
    Returns:
        array: DC field array for axial confinement
    '''

    C = - m*(w**2)/q ; DC = np.ones((Nr,Nr,Nz))
    for jCell in range(Nr):
        for iCell in range(Nr):
            for kCell in range(Nz):
                DC[jCell,iCell,kCell] = - DC[jCell,iCell,kCell]*C*(Nzmid-kCell)*dz # electric field for DC in harmonic potential approximation 
    return DC

def makeVf(Ni,q,m,wr,l=0,offsetx=0,offsety=0,offsetz=0,vbumpx=0,vbumpy=0,vbumpz=0):
    '''
    Initializes ion positions and velocities
    Creates array storing all ion parameters
    
    Parameters:
        Ni: Number of ions
        q: Ion charge
        m: Ion mass
        wr: Radial trap frequency
        l,offset*,vbump*: Optional perturbations to positions/velocities
        
    Returns:
        array: [x,y,z positions, vx,vy,vz velocities, charge, mass, polarizability]
        for each ion
    '''

    vf = np.zeros((Ni,9))
    pos = calcPositions(Ni); lscale = lengthScale(wr); scaledPos = pos*lscale
    for i in range(Ni):
        vf[i,:] = [0.0e-6,0.0e-6,-scaledPos[i],0,0,0,q,m,0.0]
    vf[l,0] += offsetx ; vf[l,1] += offsety; vf[l,2] += offsetz
    vf[l,3] += vbumpx ; vf[l,4] += vbumpy; vf[l,5] += vbumpz
    return vf

def Boltz(m,T,vmin=0,vmax=5000,bins=100):

    '''
    Generates Maxwell-Boltzmann velocity distribution
    For thermal initialization of particles
    
    Parameters:
        m: Particle mass
        T: Temperature
        vmin,vmax: Velocity range
        bins: Number of velocity bins
        
    Returns:
        array: Normalized probability distribution
    '''

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


# CUDA device functions
@cuda.jit(device=True)
def ptovPos_cuda(pos, Nmid, dcell):

    '''
    Converts physical position to grid cell index
    
    Parameters:
        pos: Physical position
        Nmid: Grid midpoint
        dcell: Grid spacing
    '''

    return (pos / dcell + numba.float64(Nmid))

@cuda.jit(device=True)
def vtopPos_cuda(pos, Nmid, dcell):

    '''
    Converts grid cell index to physical position
    Inverse of ptovPos_cuda
    '''

    return numba.float64((pos-Nmid))*dcell

@cuda.jit(device=True)
def minDists_cuda(vf, vc):
    '''
    Calculates minimum distances between ions and collision particles
    Critical for determining when collision dynamics become important

    Physics:
    - Ion-ion distance (rii): Determines Coulomb crystal structure stability
    - Ion-neutral distance (rid): Critical for langevin collision dynamics
    - Relative velocities (vid, vii): Important for:
        1. Doppler cooling efficiency
        2. Collision cross-section calculations
        3. Energy exchange during collisions
        4. Determining if quantum or classical treatment is needed
    
    Parameters:
        vf: Ion positions/velocities
        vc: Collision particle data
        
    Returns:
        tuple: (rid,rii,vid,vii) - Minimum distances and relative velocities
        between ions and collision particles
    '''

    rid2 = 1e6
    rii2 = 1e6
    vid2 = 1e6
    vii2 = 1e6
    Ni = current_ion
    Nc = 1
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
def collisionMode_cuda(rii, rid, a, e=0.3):
    '''
    Determines if system is in collision regime
    Based on ion-ion and ion-neutral distances

    Physics:
    The numerator (a * rii^2) represents the induced dipole interaction strength:
    - 'a' is polarizability of neutral particle
    - rii^2 scales with ion crystal spacing
    
    The denominator (rid^5) comes from:
    - rid^-4 from induced dipole potential (V ∝ -α|E|^2/2)
    - Additional rid^-1 from electric field scaling
    
    When this ratio exceeds threshold 'e':
    - Ion-neutral interaction becomes significant
    - Classical trajectory calculations become necessary
    - Quantum effects may need consideration
    
    Parameters:
        rii: Ion-ion distance
        rid: Ion-neutral distance
        a: Polarizability
        e: Threshold parameter
    '''

    numerator = a * rii * rii
    denominator = rid ** 5
    return numerator / denominator > e

@cuda.jit(device=True)
def confirmValid(vf,dr,dz,Nr,Nz,Nrmid,Nzmid):

    '''
    Validates and enforces boundary conditions for ion positions
    
    This function checks if ions are within valid simulation boundaries and
    handles cases where ions attempt to leave the simulation volume.
    
    Parameters:
        vf (array): Ion positions/velocities array [x,y,z,vx,vy,vz,q,m,a]
        dr (float): Radial grid spacing
        dz (float): Axial grid spacing
        Nr (int): Number of radial grid points
        Nz (int): Number of axial grid points
        Nrmid (float): Radial grid midpoint
        Nzmid (float): Axial grid midpoint
    
    Behavior:
        - Checks x, y, z positions against grid boundaries
        - If ion position invalid (outside grid):
            * Resets position to (2.0, 2.0, 2.0)
            * Zeros all velocities
            * Sets mass to 1e6 (effectively removing ion from simulation)
        - Acts as a "soft wall" boundary condition
        
    Notes:
        - Critical for simulation stability
        - Prevents ions from entering undefined grid regions
        - Mass modification acts as an ion loss detector
    '''
    
    Ni = current_ion

    for i in range(Ni):
        xCell = ptovPos_cuda(vf[i,0],Nrmid,dr) ; yCell = ptovPos_cuda(vf[i,1],Nrmid,dr) ; zCell = ptovPos_cuda(vf[i,2],Nzmid,dz) 
        if (xCell>Nr-2 or xCell<1):
            # for j in range(9):
            #     vf[i, j] = 0.0
            vf[i, 0] = 2.0
            vf[i, 1] = 2.0
            vf[i, 2] = 2.0
            vf[i, 3] = 0.0
            vf[i, 4] = 0.0
            vf[i, 5] = 0.0
            vf[i, 7] = 1e6
        elif (yCell>Nr-2 or yCell<1):
            # for j in range(9):
            #     vf[i, j] = 0.0
            vf[i, 0] = 2.0
            vf[i, 1] = 2.0
            vf[i, 2] = 2.0
            vf[i, 3] = 0.0
            vf[i, 4] = 0.0
            vf[i, 5] = 0.0
            vf[i, 7] = 1e6
        elif (zCell>Nz-2 or zCell<1):
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
    '''
    Updates ion positions based on velocities
    Includes boundary checking and validation

    Parameters:
        vf: Ion data array
        dr,dz: Grid spacings
        dt: Time step
        Nr,Nz: Grid dimensions
        Nrmid,Nzmid: Grid midpoints
    '''

    Ni = current_ion
    
    for i in range(Ni):

        vf[i,0] += vf[i,3]*dt
        vf[i,1] += vf[i,4]*dt
        vf[i,2] += vf[i,5]*dt
        confirmValid(vf, dr, dz, Nr, Nz, Nrmid, Nzmid)

@cuda.jit(device=True)
def updateVels_cuda(vf, Exf, Eyf, Ezf, dt):
    '''
    Updates ion velocities based on forces
    Applies force/acceleration from electric fields
    
    Parameters:
        vf: Ion data array
        Exf,Eyf,Ezf: Electric field components
        dt: Time step
    '''

    Ni = current_ion
    for i in range(Ni):
        # Compute force on ion i
        Fx = vf[i,6]*Exf[i] 
        Fy = vf[i,6]*Eyf[i]
        Fz = vf[i,6]*Ezf[i]

        vf[i,3] += Fx*dt/(vf[i,7])
        vf[i,4] += Fy*dt/(vf[i,7])
        vf[i,5] += Fz*dt/(vf[i,7])

@cuda.jit(device=True)
def solveFields_cuda(vf, Fx, Fy, Fz, Nrmid, Nzmid, Ni, dr, dz, Exf2, Eyf2, Ezf2):
    '''
    Calculates total electric fields acting on ions
    Combines trap fields and ion-ion Coulomb interactions
    
    Parameters:
        vf: Ion data
        Fx,Fy,Fz: Background field arrays
        Other parameters: Grid and spacing information
        Exf2,Eyf2,Ezf2: Output field arrays
    '''
    eps0 = 8.854e-12
    C1 = 4 * math.pi * eps0

    Nr = Fx.shape[0]
    Nz = Fz.shape[0]

    # Initialize fields
    for i in range(Ni):
        Exf2[i] = 0.0
        Eyf2[i] = 0.0
        Ezf2[i] = 0.0

    for i in range(Ni):
        jCell = int(round(ptovPos_cuda(vf[i,0],Nrmid,dr)))
        iCell = int(round(ptovPos_cuda(vf[i,1],Nrmid,dr)))
        kCell = int(round(ptovPos_cuda(vf[i,2],Nzmid,dz)))

        if jCell < 0 or jCell >= Nr or iCell < 0 or iCell >= Nr or kCell < 0 or kCell >= Nz:
            continue

        # Add background trap fields
        Exf2[i] += Fx[jCell,iCell,kCell]
        Eyf2[i] += Fy[jCell,iCell,kCell]
        Ezf2[i] += Fz[jCell,iCell,kCell]

        # Add contributions from other ions
        for j in range(Ni):
            if j != i:
                xdist = (vf[j,0]-vf[i,0])
                ydist = (vf[j,1]-vf[i,1])
                zdist = (vf[j,2]-vf[i,2])
                sqDist = xdist*xdist + ydist*ydist + zdist*zdist
                dist = math.sqrt(sqDist) + 1e-20  # Avoid division by zero
                projX = xdist / dist
                projY = ydist / dist
                projZ = zdist / dist

                Exf2[i] += -projX*vf[j,6]/(C1*sqDist)
                Eyf2[i] += -projY*vf[j,6]/(C1*sqDist)
                Ezf2[i] += -projZ*vf[j,6]/(C1*sqDist)

@cuda.jit(device=True)
def collisionParticlesFields_cuda(vf, vc, Ni, RFx, RFy, DCz, Nr, Nz, dr, dz, dt, Nrmid, Nzmid, Exfi, Eyfi, Ezfi):
    '''
    Handles collision particle dynamics and induced dipole interactions
    Complex function managing neutral-ion interactions

    Physics:
    1. Induced dipole calculation:
       - Neutral particle polarization by ion's electric field
       - p = α*E where α is polarizability
       - Field includes both trap fields and ion fields
    
    2. Force calculations:
       - Ion-induced dipole force: F ∝ -∇(p·E)
       - Leads to attractive r^-4 potential
       - Non-conservative forces due to field gradients
    
    3. Energy exchange:
       - Elastic collisions conserve total energy
       - Inelastic processes possible at close range
    
    4. Critical distances:
       - Langevin capture radius: when ion-induced dipole potential ≈ kT
       - Quantum effects important at very close range
       - Classical trajectories valid at larger distances
    '''

    eps0 = 8.854e-12
    C1 = 4 * math.pi * eps0

    Nc = 1
    
    # Local arrays use maximum sizes
    Exfc = cuda.local.array((Nc, 2), dtype=float64)
    Eyfc = cuda.local.array((Nc, 2), dtype=float64)
    Ezfc = cuda.local.array((Nc, 2), dtype=float64)

    sqDist = cuda.local.array((Nc, current_ion), dtype=float64)
    dist_j = cuda.local.array((Nc, current_ion), dtype=float64)

    projX_j = cuda.local.array((Nc, current_ion), dtype=float64)
    projY_j = cuda.local.array((Nc, current_ion), dtype=float64)
    projZ_j = cuda.local.array((Nc, current_ion), dtype=float64)
    sqDist_j_1p5 = cuda.local.array((Nc, current_ion), dtype=float64)

    for i in range(Nc):

        jCell = int(round(ptovPos_cuda(vc[i,0],Nrmid,dr)))
        iCell = int(round(ptovPos_cuda(vc[i,1],Nrmid,dr)))
        kCell = int(round(ptovPos_cuda(vc[i,2],Nzmid,dz)))

        if jCell < 1 or jCell >= Nr-1 or iCell < 1 or iCell >= Nr-1 or kCell < 1 or kCell >= Nz-1:
            continue

        Exfc[i,0] += (RFx[jCell,iCell,kCell])
        Eyfc[i,0] += (RFy[jCell,iCell,kCell])
        Ezfc[i,0] += (DCz[jCell,iCell,kCell])

        Exfc[i,1] += (RFx[jCell+1,iCell,kCell])-(RFx[jCell-1,iCell,kCell])/dr
        Eyfc[i,1] += (RFy[jCell,iCell+1,kCell] )-(RFy[jCell,iCell-1,kCell] )/dr
        Ezfc[i,1] += (DCz[jCell,iCell,kCell+1]-DCz[jCell,iCell,kCell-1])/dz

        for j in range(Ni):
            xdist = (vf[j,0]-vc[i,0]) ; ydist = (vf[j,1]-vc[i,1]) ; zdist = (vf[j,2]-vc[i,2])

            sqDist[i, j] = xdist * xdist + ydist * ydist + zdist * zdist

            dist_j[i, j] = math.sqrt(xdist * xdist + ydist * ydist + zdist * zdist)

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

    pX = cuda.local.array(Nc, dtype=float64); pY = cuda.local.array(Nc, dtype=float64); pZ = cuda.local.array(Nc, dtype=float64)
    
    for k in range(Nc):
        if vc[k,6] != 0.0:
            pX[k] = -2 * math.pi * eps0 * vc[k, 8] * Exfc[k, 0]
            pY[k] = -2 * math.pi * eps0 * vc[k, 8] * Eyfc[k, 0]
            pZ[k] = -2 * math.pi * eps0 * vc[k, 8] * Ezfc[k, 0]

            Fx = math.fabs(pX[k]) * Exfc[k, 1]
            Fy = math.fabs(pY[k]) * Eyfc[k, 1]
            Fz = math.fabs(pZ[k]) * Ezfc[k, 1]

            vc[k, 3] += Fx * dt / (vc[k, 7])
            vc[k, 4] += Fy * dt / (vc[k, 7])
            vc[k, 5] += Fz * dt / (vc[k, 7])
    
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

# @cuda.jit(device=True)
# def mcCollision_cuda(vf, xc, yc, zc, vxc, vyc, vzc, qc, mc, ac, Nt, dtSmall, 
#                     dtCollision, RFx, RFy, DC, Nr, Nz, dr, dz, dtLarge, 
#                     reorder, withCollision, progress, total_physical_time):
#     '''
#     Core device function implementing the Monte Carlo collision simulation
    
#     This function evolves the ion-neutral collision dynamics with adaptive time stepping
#     and monitors various physical conditions during the simulation.
    
#     Parameters:
#         vf (array): Ion positions/velocities array
#         xc, yc, zc (float): Initial collision particle position
#         vxc, vyc, vzc (float): Initial collision particle velocities
#         qc (float): Collision particle charge
#         mc (float): Collision particle mass
#         ac (float): Collision particle polarizability
#         Nt (int): Maximum number of time steps
#         dtSmall (float): Minimum time step size
#         dtCollision (float): Collision time step
#         RFx, RFy (array): RF field arrays
#         DC (array): DC field array
#         Nr, Nz (int): Grid dimensions
#         dr, dz (float): Grid spacings
#         dtLarge (float): Maximum time step size
#         reorder (array): Flag array for ion reordering events
#         withCollision (bool): Enable/disable collision dynamics
#         progress (array): Simulation progress counter
#         total_physical_time (float): Maximum physical time to simulate
    
#     Key Features:
#         - Adaptive time stepping based on:
#             * Ion-ion distances
#             * Ion-neutral distances
#             * Collision events
#         - Tracks multiple types of events:
#             * Ion ejection (mass_sum > 1e5)
#             * Ion reordering (position crossings)
#             * Collision dynamics
#         - Time evolution includes:
#             * Field solving
#             * Position/velocity updates
#             * Collision handling
#             * Boundary checking
    
#     Return behavior:
#         - Updates reorder flag:
#             * 0: Normal completion
#             * 1: Ion reordering detected
#             * 2: Ion ejection detected
#         - Updates progress counter atomically
        
#     Notes:
#         - Uses local arrays for efficiency
#         - Implements collision detection threshold (0.000000001)
#         - Terminates on physical time limit or ion loss
#         - Critical for accurate collision dynamics
#     '''

#     reorder[0] = 0
#     Nr = int32(Nr)
#     Nz = int32(Nz)
#     Nrmid = float64((Nr - 1) / 2.0)
#     Nzmid = float64((Nz - 1) / 2.0)
#     Ni = current_ion

#     # Local arrays
#     vc = cuda.local.array((Nc, 9), dtype=float64)
#     Exfi = cuda.local.array(Ni, dtype=float64)
#     Eyfi = cuda.local.array(Ni, dtype=float64)
#     Ezfi = cuda.local.array(Ni, dtype=float64)
    
#     # Initialize collision particle
#     vc[0, 0] = float64(xc)
#     vc[0, 1] = float64(yc)
#     vc[0, 2] = float64(zc)
#     vc[0, 3] = float64(vxc)
#     vc[0, 4] = float64(vyc)
#     vc[0, 5] = float64(vzc)
#     vc[0, 6] = float64(qc)
#     vc[0, 7] = float64(mc)
#     vc[0, 8] = float64(ac)

#     dtNow = float64(dtSmall)
#     crossTest = 0
#     accumulated_time = float64(0.0)

#     # Main simulation loop
#     for i in range(Nt):
#         # Check if we've reached total simulation time
#         if accumulated_time >= total_physical_time:
#             break

#         rid, rii, vid, vii = minDists_cuda(vf, vc)
#         collision = collisionMode_cuda(rii, rid, vc[0,8], 0.000000001)
        
#         if collision:
#             dtNow = rid * 0.01 / (5.0 * vid)
#         else:
#             dtNow = dtSmall
        
#         if dtNow < dtCollision:
#             dtNow = dtCollision

#         solveFields_cuda(vf, RFx, RFy, DC, Nrmid, Nzmid, Ni, dr, dz, Exfi, Eyfi, Ezfi)
        
#         if withCollision:
#             if vc[0, 7] < 1e6:
#                 collisionParticlesFields_cuda(vf, vc, Ni, RFx, RFy, DC, Nr, Nz, dr, dz, dtNow, Nrmid, Nzmid, Exfi, Eyfi, Ezfi)
#                 updatePoss_cuda(vc, dr, dz, dtNow, Nr, Nz, Nrmid, Nzmid)
#                 dtNow = dtSmall
        
#         updateVels_cuda(vf, Exfi, Eyfi, Ezfi, dtNow)
#         updatePoss_cuda(vf, dr, dz, dtNow, Nr, Nz, Nrmid, Nzmid)
        
#         # Update accumulated time
#         accumulated_time += dtNow
        
#         # Check for ion ejection
#         mass_sum = 0.0
#         for a in range(Ni):
#             mass_sum += vf[a, 7]
        
#         if mass_sum > 1e5:
#             reorder[0] += 2
#             break

#         for b in range(1, Ni):
#             if vf[b, 2] > vf[b-1, 2] and crossTest < 1:
#                 reorder[0] += 1
#                 crossTest += 1
#                 break

#         if reorder[0] != 0:
#             break

#     # Mark this thread as complete
#     cuda.atomic.add(progress, 0, 1)

# @cuda.jit
# def mcCollision_kernel(vf_all, xc_all, yc_all, zc_all, vxc_all, vyc_all, vzc_all, 
#                       qc, mc, ac, Nt_all, dtSmall, dtCollision, RFx, RFy, DC, 
#                       Nr, Nz, dr, dz, dtLarge, reorder_all, withCollision, progress,
#                       total_physical_time):
#     '''
#     Main CUDA kernel for parallel simulation execution
#     Manages multiple ion chains simultaneously
    
#     Features:
#     - Parallel tracking of multiple trajectories
#     - Time evolution with adaptive stepping
#     - Collision detection and handling
#     - Progress monitoring
#     - Physical time limit enforcement
    
#     Parameters:
#         vf_all: All ion trajectory data
#         Various collision parameters
#         Field arrays
#         Grid parameters
#         Timing controls
        
#     Notes:
#         - Uses shared memory for efficiency
#         - Implements adaptive time stepping
#         - Tracks simulation progress
#         - Enforces total physical time limits
#     '''
#     thread_id = cuda.grid(1)
    
#     if thread_id < vf_all.shape[0]:
#         # Local array for current thread's ions
#         vf = cuda.local.array((current_ion, 9), dtype=float64)
#         reorder = cuda.local.array(1, dtype=int32)
        
#         # Copy input data to local array
#         for i in range(current_ion):
#             for j in range(9):
#                 vf[i, j] = vf_all[thread_id, i, j]

#         # Get scalar values for this thread
#         xc = float64(xc_all[thread_id])
#         yc = float64(yc_all[thread_id])
#         zc = float64(zc_all[thread_id])
#         vxc = float64(vxc_all[thread_id])
#         vyc = float64(vyc_all[thread_id])
#         vzc = float64(vzc_all[thread_id])
#         Nt = int32(Nt_all[thread_id])

#         # Call device function with total_physical_time
#         mcCollision_cuda(
#             vf, xc, yc, zc, vxc, vyc, vzc,
#             qc, mc, ac, Nt, dtSmall, dtCollision,
#             RFx, RFy, DC, Nr, Nz, dr, dz, dtLarge,
#             reorder, withCollision, progress, total_physical_time
#         )
        
#         # Copy results back to global memory
#         reorder_all[thread_id] = reorder[0]
#         for i in range(current_ion):
#             for j in range(9):
#                 vf_all[thread_id, i, j] = vf[i, j]


############################
###### Animation part ######
############################


@cuda.jit(device=True)
def mcCollision_cuda(vf, xc, yc, zc, vxc, vyc, vzc, qc, mc, ac, Nt, dtSmall, 
                   dtCollision, RFx, RFy, DC, Nr, Nz, dr, dz, dtLarge, 
                   reorder, withCollision, progress, total_physical_time,
                   positions_over_time, timesteps):
   '''
   CUDA device function for Monte Carlo collision simulation with adaptive storage
   '''
   reorder[0] = 0
   Nr = int32(Nr)
   Nz = int32(Nz)
   Nrmid = float64((Nr - 1) / 2.0)
   Nzmid = float64((Nz - 1) / 2.0)
   Ni = current_ion

   # Local arrays
   vc = cuda.local.array((Nc, 9), dtype=float64)
   Exfi = cuda.local.array(Ni, dtype=float64)
   Eyfi = cuda.local.array(Ni, dtype=float64)
   Ezfi = cuda.local.array(Ni, dtype=float64)
   
   # Initialize collision particle
   vc[0, 0] = float64(xc)
   vc[0, 1] = float64(yc)
   vc[0, 2] = float64(zc)
   vc[0, 3] = float64(vxc)
   vc[0, 4] = float64(vyc)
   vc[0, 5] = float64(vzc)
   vc[0, 6] = float64(qc)
   vc[0, 7] = float64(mc)
   vc[0, 8] = float64(ac)

   accumulated_time = float64(0.0)
   storage_step = 0
   crossTest = 0
   compression_counter = 0
   collision_mode = False

   # Store initial positions
   timesteps[storage_step] = accumulated_time
   positions_over_time[storage_step, 0, 0] = vf[0, 0]  # ion 1 x
   positions_over_time[storage_step, 0, 1] = vf[0, 1]  # ion 1 y
   positions_over_time[storage_step, 0, 2] = vf[0, 2]  # ion 1 z
   
   positions_over_time[storage_step, 1, 0] = vf[1, 0]  # ion 2 x
   positions_over_time[storage_step, 1, 1] = vf[1, 1]  # ion 2 y
   positions_over_time[storage_step, 1, 2] = vf[1, 2]  # ion 2 z
   
   positions_over_time[storage_step, 2, 0] = vc[0, 0]  # collision x
   positions_over_time[storage_step, 2, 1] = vc[0, 1]  # collision y
   positions_over_time[storage_step, 2, 2] = vc[0, 2]  # collision z
   
   storage_step += 1

   # Main simulation loop
   while accumulated_time < total_physical_time:
       # Check storage space
       if storage_step >= positions_over_time.shape[0]-1:
           break

       rid, rii, vid, vii = minDists_cuda(vf, vc)
       collision = collisionMode_cuda(rii, rid, vc[0,8], 0.000000001)
       
       # Adaptive timestep
       if collision:
           dtNow = rid * 0.01 / (5.0 * vid)
           collision_mode = True
       else:
           dtNow = dtSmall
           collision_mode = False
       
       if dtNow < dtCollision:
           dtNow = dtCollision

       # Run the physics
       solveFields_cuda(vf, RFx, RFy, DC, Nrmid, Nzmid, Ni, dr, dz, Exfi, Eyfi, Ezfi)
       
       if withCollision and vc[0, 7] < 1e6:
           collisionParticlesFields_cuda(vf, vc, Ni, RFx, RFy, DC, Nr, Nz, dr, dz, dtNow, Nrmid, Nzmid, Exfi, Eyfi, Ezfi)
           updatePoss_cuda(vc, dr, dz, dtNow, Nr, Nz, Nrmid, Nzmid)
           dtNow = dtSmall
       
       updateVels_cuda(vf, Exfi, Eyfi, Ezfi, dtNow)
       updatePoss_cuda(vf, dr, dz, dtNow, Nr, Nz, Nrmid, Nzmid)
       
       accumulated_time += dtNow

       # Adaptive storage strategy
       store_this_step = False
       if not collision_mode:  # Regular timestep - store every step
           store_this_step = True
           compression_counter = 0
       else:  # Collision mode - store more frequently but compressed
           compression_counter += 1
           if compression_counter >= 10:  # Store every 10th step during collisions
               store_this_step = True
               compression_counter = 0

       if store_this_step:
           timesteps[storage_step] = accumulated_time
           positions_over_time[storage_step, 0, 0] = vf[0, 0]  # ion 1 x
           positions_over_time[storage_step, 0, 1] = vf[0, 1]  # ion 1 y
           positions_over_time[storage_step, 0, 2] = vf[0, 2]  # ion 1 z
           
           positions_over_time[storage_step, 1, 0] = vf[1, 0]  # ion 2 x
           positions_over_time[storage_step, 1, 1] = vf[1, 1]  # ion 2 y
           positions_over_time[storage_step, 1, 2] = vf[1, 2]  # ion 2 z
           
           positions_over_time[storage_step, 2, 0] = vc[0, 0]  # collision x
           positions_over_time[storage_step, 2, 1] = vc[0, 1]  # collision y
           positions_over_time[storage_step, 2, 2] = vc[0, 2]  # collision z
           
           storage_step += 1

       # Check for early termination conditions
       mass_sum = 0.0
       for a in range(Ni):
           mass_sum += vf[a, 7]
       
       if mass_sum > 1e5:  # Ion ejection
           reorder[0] += 2
           break

       for b in range(1, Ni):  # Ion reordering
           if vf[b, 2] > vf[b-1, 2] and crossTest < 1:
               reorder[0] += 1
               crossTest += 1
               break

       if reorder[0] != 0:
           break

   # Store final step count
   timesteps[-1] = float64(storage_step)
   
   # Mark completion
   cuda.atomic.add(progress, 0, 1)

@cuda.jit
def mcCollision_kernel(vf_all, xc_all, yc_all, zc_all, vxc_all, vyc_all, vzc_all, 
                     qc, mc, ac, Nt_all, dtSmall, dtCollision, RFx, RFy, DC, 
                     Nr, Nz, dr, dz, dtLarge, reorder_all, withCollision, progress,
                     total_physical_time, positions_over_time, timesteps):
   '''
   CUDA kernel for Monte Carlo collision simulation with trajectory tracking
   '''
   thread_id = cuda.grid(1)
   
   if thread_id < vf_all.shape[0]:
       # Local array for current thread's ions
       vf = cuda.local.array((current_ion, 9), dtype=float64)
       reorder = cuda.local.array(1, dtype=int32)
       
       # Copy input data to local array
       for i in range(current_ion):
           for j in range(9):
               vf[i, j] = vf_all[thread_id, i, j]

       # Get scalar values for this thread
       xc = float64(xc_all[thread_id])
       yc = float64(yc_all[thread_id])
       zc = float64(zc_all[thread_id])
       vxc = float64(vxc_all[thread_id])
       vyc = float64(vyc_all[thread_id])
       vzc = float64(vzc_all[thread_id])
       Nt = int32(Nt_all[thread_id])

       # Get this thread's portion of trajectory arrays
       thread_positions = positions_over_time[thread_id]
       thread_timesteps = timesteps[thread_id]

       # Call device function 
       mcCollision_cuda(
           vf, xc, yc, zc, vxc, vyc, vzc,
           qc, mc, ac, Nt, dtSmall, dtCollision,
           RFx, RFy, DC, Nr, Nz, dr, dz, dtLarge,
           reorder, withCollision, progress, total_physical_time,
           thread_positions, thread_timesteps
       )
       
       # Copy results back to global memory
       reorder_all[thread_id] = reorder[0]
       for i in range(current_ion):
           for j in range(9):
               vf_all[thread_id, i, j] = vf[i, j]