import numpy as np
from numba import cuda, float64, int32
import math

# Constants and parameters
amu = 1.67e-27
eps0 = 8.854e-12
qe = 1.6e-19

# Physical parameters
m = 40.0 * amu
q = 1.0 * qe
wr = 2 * np.pi * 3e6
wz = 2 * np.pi * 1e6

# Grid parameters
Dr = 30001.5e-9
Dz = 90001.5e-9
grid_size = 25001
Nr = Nz = grid_size
Nrmid = Nzmid = (Nr - 1) // 2
dr = Dr / float(Nr)
dz = Dz / float(Nz)

# Test data
vf = np.array([
    [0.0, 4.79258560e-06, 0.0, 0.0, q, m, 0.0],
    [0.0, -8.03509216e-20, 0.0, 0.0, q, m, 0.0],
    [0.0, -4.79258560e-06, 0.0, 0.0, q, m, 0.0]
], dtype=np.float64)

# Helper functions
def ptovPos(pos, Nmid, dcell):
    return pos/dcell + float(Nmid)

@cuda.jit(device=True)
def ptovPos_cuda(pos, Nmid, dcell):
    return pos/dcell + Nmid

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

# Create fields
RF = makeRF0(m, q, wr, Nr, Nz, Nrmid, dr)
DC = makeDC(m, q, wz, Nz, Nr, Nzmid, dz)
nullFields = np.zeros((Nr, Nz), dtype=np.float64)

# CPU version
def solveFields_cpu(vf,ErDC,EzDC,ErAC,EzAC,Nrmid,Nzmid,Ni,dr,dz):
    """ this solves for the electric fields at  each ion from each ion (and the trap)
    vf is the vector of ion parameters, the first index is the ion, the second index is r,z,vr,vz,q,m,polarizability
    ErDC is the array of electric fields from the DC electrodes in r
    EzDC is the array of electric fields from the DC electrodes in z
    ErAC is the array of electric fields from the AC electrodes in r
    EzAC is the array of electric fields from the AC electrodes in z
    Nrmid, Nzmid are the midpoints of the grid
    Ni is the number of ions
    dr and dz are the cell sizes of the grid
    Erf2 and Ezf2 are the electric fields at each ion. These names could probably be improved
    """
    eps0 = 8.854e-12 ; C1 = 4*np.pi*eps0 #SI units 
    Erf2 = np.zeros(Ni) ; Ezf2 = np.zeros(Ni)
    for i in range(len(vf[:,0])): # note that this no longer takes the electric field from all particles, so chillax 
        jCell = ptovPos(vf[i,0],Nrmid,dr) ; kCell = ptovPos(vf[i,1],Nzmid,dz) # 
        jCell = int(round(jCell)) ; kCell = int(round(kCell)) # local cell index in r and z
        Erf2[i] += ErDC[jCell,kCell] + ErAC[jCell,kCell] ; Ezf2[i] += EzDC[jCell,kCell] + EzAC[jCell,kCell] # add trap fields
        for j in range(len(vf[:,0])):
            if j!=i: #here we solve for the fields from each other ion on this ion
                rdist = (vf[j,0]-vf[i,0]) ; zdist = (vf[j,1]-vf[i,1]) # get each distance
                sqDist = (rdist)**2 + (zdist)**2 #square distance from particle to cell
                projR = rdist/sqDist**(1/2) ; projZ = zdist/sqDist**(1/2) # cos theta to project E field to z basis, sin to r basis               
                Erf2[i] += -projR*vf[j,4]/(C1*sqDist) ; Ezf2[i] += -projZ*vf[j,4]/(C1*sqDist) # add fields in r and z 
    return Erf2,Ezf2


# GPU version
@cuda.jit(device=True)
def solveFields_gpu(vf, ErDC, EzDC, ErAC, EzAC, Nrmid, Nzmid, Ni, dr, dz, Erf2, Ezf2):
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

# Wrapper kernel for GPU function
@cuda.jit
def gpu_wrapper(vf, ErDC, EzDC, ErAC, EzAC, Nrmid, Nzmid, Ni, dr, dz, Erf2, Ezf2):
    solveFields_gpu(vf, ErDC, EzDC, ErAC, EzAC, Nrmid, Nzmid, Ni, dr, dz, Erf2, Ezf2)

# Run CPU version
Erf2_cpu, Ezf2_cpu = solveFields_cpu(vf, nullFields, DC, RF, nullFields, Nrmid, Nzmid, len(vf), dr, dz)

# Run GPU version
Erf2_gpu = np.zeros(len(vf), dtype=np.float64)
Ezf2_gpu = np.zeros(len(vf), dtype=np.float64)
gpu_wrapper[1, 1](vf, nullFields, DC, RF, nullFields, Nrmid, Nzmid, len(vf), dr, dz, Erf2_gpu, Ezf2_gpu)

# Print and compare results
print("CPU Results:")
print("Radial Electric Fields (Erf2):")
for i, val in enumerate(Erf2_cpu):
    print(f"Ion {i}: {val:.6e}")
print("\nAxial Electric Fields (Ezf2):")
for i, val in enumerate(Ezf2_cpu):
    print(f"Ion {i}: {val:.6e}")

print("\nGPU Results:")
print("Radial Electric Fields (Erf2):")
for i, val in enumerate(Erf2_gpu):
    print(f"Ion {i}: {val:.6e}")
print("\nAxial Electric Fields (Ezf2):")
for i, val in enumerate(Ezf2_gpu):
    print(f"Ion {i}: {val:.6e}")

# print("\nAbsolute Differences (CPU - GPU):")
# print("Radial Electric Fields (Erf2):")
# for i in range(len(vf)):
#     diff = abs(Erf2_cpu[i] - Erf2_gpu[i])
#     print(f"Ion {i}: {diff:.6e}")
# print("\nAxial Electric Fields (Ezf2):")
# for i in range(len(vf)):
#     diff = abs(Ezf2_cpu[i] - Ezf2_gpu[i])
#     print(f"Ion {i}: {diff:.6e}")