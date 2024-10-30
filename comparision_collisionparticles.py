import numpy as np
import matplotlib.pyplot as plt
import math
from numba import cuda, float64, int32

# Constants
amu = 1.67e-27
eps0 = 8.854e-12
qe = 1.6e-19

# Physical parameters
m = 40.0 * amu
q = 1.0 * qe
wr = 2 * np.pi * 3e6
wz = 2 * np.pi * 1e6
aH2 = 8e-31
mH2 = 2.0 * amu

# Simulation parameters
Dr = 30001.5e-9
Dz = 90001.5e-9
dtSmall = 1e-12
dtCollision = 1e-16
dtLarge = 1e-10

# Grid parameters
grid_size = 25001
Nr = Nz = grid_size
Nrmid = Nzmid = (Nr - 1) // 2
dr = Dr / float(Nr)
dz = Dz / float(Nz)

# Ion parameters
Ni = 3
vf = np.array([
    [-1.009804e-16, 4.792552e-06, -4.874243e-04, 1.158576e+01, q, m, 0.0],
    [-1.142330e-26, -8.002224e-20, 5.675089e-16, 2.931912e-11, q, m, 0.0],
    [-3.434519e-27, -4.792552e-06, 7.117355e-17, 1.074302e-02, q, m, 0.0]
])

# Collisional particle parameters
# rc = -4.759190024710139e-07
# zc = 1.978544579898285e-05
# vrc = 75.75163221513105
# vzc = -2386.3355135788324
# vc = np.array([[rc, zc, vrc, vzc, q, mH2, aH2]])

rc = -1.4260283408095726e-11
zc = 4.7930278057010626e-06
vrc = 75.85126804082506
vzc = -2495.6199663578495
vc = np.array([[rc, zc, vrc, vzc, q, mH2, aH2]])

# Time step
dt = dtSmall

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

def ptovPos(pos,Nmid,dcell):
    return (pos/dcell + float(Nmid))

RF = makeRF0(m, q, wr, Nr, Nz, Nrmid, dr)
DC = makeDC(m, q, wz, Nz, Nr, Nzmid, dz)
nullFields = np.zeros((Nr, Nz), dtype=np.float64)

def collisionParticlesFields_cpu(vf,vc,Ni,ErDC,EzDC,ErAC,EzAC,dr,dz,dt,Nrmid,Nzmid): # this applies the fields of all existing collisional particles and changes the velocity of those collisional particles
    """
    vf is the ion parameters, the first index is the ion, the second index is r,z,vr,vz,q,m,polarizability
    vc is the collisional particle parameters with the same scheme as vf
    Ni is the number of ions
    Erfi is the radial electric fields on ions
    Ezfi is the axial electric fields on ions
    Erfc is the radial electric fields on collisional particles at both pseudo-particle points
    Ezfc is the axial electric fields on collisional particles at both pseudo-particle points
    dr is the physical size of a cell in r, dz is the physical size of a cell in z
    ErDC,EzDC,ErAC,EzAC are the electric fields from the background
    note that we treat dipoles as if they instantly align with the dominant electric field
    """
    eps0 = 8.854e-12
    Nc = len(vc[:,0])
    # we begin by instantiating the electric field lists (fields are in physical units)
    Erfi = np.zeros(Ni); Ezfi = np.zeros(Ni)
    Erfc = np.zeros((Nc,2)); Ezfc = np.zeros((Nc,2)) # [i,1] is middle field, 2 is high index - low index divided by the size of a cell (ie, the local slope of the E-field)
    Erfc_history = np.zeros((Nc,2)); Ezfc_history = np.zeros((Nc,2))
    sqDist = np.zeros((Nc,Ni)); projR = np.zeros((Nc,Ni)); projZ = np.zeros((Nc,Ni))
    C1 = 4*np.pi*eps0 # commonly used set of constants put together
    # we solve the electric fields on the collisional particles
    for i in range(Nc): # for each collisional particle that exists        
        # In order to allow for electric field gradients of the background field, here we need to implement a linear E-field gradient between neighboring cells
        jCell = ptovPos(vc[i,0],Nrmid,dr) ; kCell = ptovPos(vc[i,1],Nzmid,dz)
        jCell = int(round(jCell)) ; kCell = int(round(kCell)) # local cell index in r and z
        # we initialize the interpolated field for each 
        Erfc[i,0] += (ErDC[jCell,kCell] + ErAC[jCell,kCell]) ; Ezfc[i,0] += (EzDC[jCell,kCell] + EzAC[jCell,kCell])
        Erfc[i,1] += ((ErDC[jCell+1,kCell] + ErAC[jCell+1,kCell])-(ErDC[jCell-1,kCell] + ErAC[jCell-1,kCell]))/dr ; Ezfc[i,1] += ((EzDC[jCell,kCell+1] + EzAC[jCell,kCell+1])-(EzDC[jCell,kCell-1] + EzAC[jCell,kCell-1]))/dz       
        Erfc_history[i, 0] = jCell; Ezfc_history[i, 0] = Ezfc[i, 0]
        Erfc_history[i, 1] = kCell; Ezfc_history[i, 1] = Ezfc[i, 1] 
        for j in range(Ni): # solve the electric field exerted by each ion
            rdist = (vf[j,0]-vc[i,0]) ; zdist = (vf[j,1]-vc[i,1])
            sqDist[i,j] = (rdist)**2 + (zdist)**2 #distance from particle to cell
            projR[i,j] = rdist/sqDist[i,j]**(1/2) ; projZ[i,j] = zdist/sqDist[i,j]**(1/2) #cos theta to project E field to z basis and sin to r basis
            Erfc[i,0] += -projR[i,j]*vf[j,4]/(C1*sqDist[i,j]) ; Ezfc[i,0] += -projZ[i,j]*vf[j,4]/(C1*sqDist[i,j]) # add fields in r and z   
            # I just need to add the gradient field from these now and the colliding particle should rebound
            Erfc[i,1] += 2*projR[i,j]*vf[j,4]/(C1*sqDist[i,j]**(3/2)) ; Ezfc[i,1] += 2*projZ[i,j]*vf[j,4]/(C1*sqDist[i,j]**(3/2)) # add fields in r and z                 
    pR = np.zeros(Nc); pZ = np.zeros(Nc); pTot = np.zeros(Nc)
    for i in range(Nc):    # a dipole is induced in the direction of the electric field vector with the positive pseudoparticle in the positive field direction
        if vc[i,6]!=0.0: # if there is a dipole moment that can be obtained
            pR[i] = -2*np.pi*eps0*vc[i,6]*Erfc[i,0] # dipole in r in SI units note this factor of 2 pi epsilon0 which corrects the units of m^-3 on alpha and Volts/m on E to give a dipole moment in Coulomb*meters
            pZ[i] = -2*np.pi*eps0*vc[i,6]*Ezfc[i,0] # dipole in z in SI units ###FIX THIS###
            pTot[i] = (pR[i]**2+pZ[i]**2)**(1/2) # total dipole length in physical units
            # we can now induce the force on the dipole
            Fr = abs(pR[i])*Erfc[i,1] ; Fz = abs(pZ[i])*Ezfc[i,1]
            #then we would need to convert back to virtual units once we apply the forces
            vc[i,2] += Fr*dt/(vc[i,5]) ; vc[i,3] += Fz*dt/(vc[i,5]) # update velocity with F*t/m                 
    # we then solve for the fields the collisional particles exert on the ions from the dipole (and quadrapole potentially) as well as the charge if the particle has one
    for i in range(Ni): # for each ion in the trap
        for j in range(Nc): # apply the field from each collisional particle
            # the dipole field is (3*(dipole moment dotted with vector from particle to ion)(in vector from particle to ion) - (dipole moment))/(3*pi*eps0*distance^3)
            # at close proximity, it should be treated as monopoles with charge qe separated by (dipole moment)/qe distance along the dipole moment vector
            # for now we treat the electric field it exerts as a pure dipole
            if vc[j,6]!=0.0: # if there is a potential dipole moment
                Rhatr = projR[j,i] ; Rhatz = projZ[j,i]
                dist = sqDist[j,i]**(1/2)
                Erfi[i] += -abs(pR[j])*(2*Rhatr)/(C1*dist**3) ; Ezfi[i] += -abs(pZ[j])*(2*Rhatz)/(C1*dist**3) # add dipole fields
    return vc,Erfi,Ezfi,Erfc,Ezfc # the ion electric fields are just from the collisional particles, the collisional electric fields are from all sources
# note that I haven't applied electric fields from collisional particles onto each other

@cuda.jit(device=True)
def collisionParticlesFields_gpu(vf, vc, Ni, ErDC, EzDC, ErAC, EzAC, dr, dz, dtNow, Nrmid, Nzmid, Nr, Nz, Erfi, Ezfi, Erfc_out, Ezfc_out):
    eps0 = 8.854e-12
    C1 = 4 * math.pi * eps0
    Nc = vc.shape[0]
    small = 1e-20  # Small value to prevent division by zero

    # Initialize Erfi and Ezfi
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

            Erfc_out[i, 0] = Erfc[0]
            Erfc_out[i, 1] = Erfc[1]
            Ezfc_out[i, 0] = Ezfc[0]
            Ezfc_out[i, 1] = Ezfc[1]

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

@cuda.jit(device=True)
def ptovPos_cuda(pos, Nmid, dcell):
    return pos / dcell + Nmid

# Run CPU version
vc_cpu, Erfi_cpu, Ezfi_cpu, Erfc_cpu, Ezfc_cpu = collisionParticlesFields_cpu(vf, vc, Ni, nullFields, DC, RF, nullFields, dr, dz, dt, Nrmid, Nzmid)

# Run GPU version
Erfi_gpu = np.zeros(Ni, dtype=np.float64)
Ezfi_gpu = np.zeros(Ni, dtype=np.float64)
Erfc_gpu = np.zeros((vc.shape[0], 2), dtype=np.float64)
Ezfc_gpu = np.zeros((vc.shape[0], 2), dtype=np.float64)
vc_gpu = vc.copy()

@cuda.jit
def gpu_wrapper(vf, vc, Ni, ErDC, EzDC, ErAC, EzAC, dr, dz, dt, Nrmid, Nzmid, Nr, Nz, Erfi, Ezfi, Erfc, Ezfc):
    collisionParticlesFields_gpu(vf, vc, Ni, ErDC, EzDC, ErAC, EzAC, dr, dz, dt, Nrmid, Nzmid, Nr, Nz, Erfi, Ezfi, Erfc, Ezfc)

gpu_wrapper[1, 1](vf, vc_gpu, Ni, nullFields, DC, RF, nullFields, dr, dz, dt, Nrmid, Nzmid, Nr, Nz, Erfi_gpu, Ezfi_gpu, Erfc_gpu, Ezfc_gpu)

# Compare results
print("CPU results:")
print("vc:", vc_cpu)
print("Erfi:", Erfi_cpu)
print("Ezfi:", Ezfi_cpu)
print("Erfc:", Erfc_cpu)
print("Ezfc:", Ezfc_cpu)

print("\nGPU results:")
print("vc:", vc_gpu)
print("Erfi:", Erfi_gpu)
print("Ezfi:", Ezfi_gpu)
print("Erfc:", Erfc_gpu)
print("Ezfc:", Ezfc_gpu)