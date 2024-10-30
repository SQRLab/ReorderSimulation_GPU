import numpy as np
import cupy as cp
from numba import cuda, float64
import math
import random
import time
import json
import os
from scipy import special
import numba
import multiprocessing
from multiprocessing import Pool

pi = cp.pi
fsolve = cp.linalg.solve

# Constants
con = cp.array([1.602176634e-19, 9.1093837015e-31, 1.67262192369e-27, 6.02214076e23, 8.8541878128e-12])
atomic_mass = 1.66053906660e-27
elementary_charge = 1.602176634e-19
epsilon_0 = 8.8541878128e-12

@cuda.jit(device=True)
def ptovPos_device(pos, Nmid, dcell):
    return int(pos / float(dcell) + float(Nmid))

@cuda.jit
def ion_position_potential_kernel(x, result, N):
    i = cuda.grid(1)
    if i < N:
        sum_left = 0.0
        sum_right = 0.0
        for j in range(i):
            diff = x[i] - x[j]
            if abs(diff) > 1e-10:
                sum_left += 1.0 / (diff**2)
        for j in range(i+1, N):
            diff = x[i] - x[j]
            if abs(diff) > 1e-10:
                sum_right += 1.0 / (diff**2)
        result[i] = x[i] - sum_left + sum_right

def ion_position_potential(x):
    N = len(x)
    d_x = cuda.to_device(cp.asarray(x))
    d_result = cuda.device_array(N, dtype=cp.float64)
    threads_per_block = 256
    blocks = (N + threads_per_block - 1) // threads_per_block
    ion_position_potential_kernel[blocks, threads_per_block](d_x, d_result, N)
    return d_result.copy_to_host()

def calcPositions(N, max_iterations=1000, tolerance=1e-12):
    estimated_extreme = 0.481 * N**0.765
    x = np.linspace(-estimated_extreme, estimated_extreme, N)
    
    #print(f"Initial x: {x}")
    
    for iteration in range(max_iterations):
        fx = ion_position_potential(x)
        fx_norm = np.linalg.norm(fx)
        #print(f"Iteration {iteration}, fx norm: {fx_norm}")
        
        if fx_norm < tolerance:
         #   print(f"Converged after {iteration} iterations")
            break
        
        # Use numerical approximation for Jacobian
        J = np.zeros((N, N))
        h = 1e-8
        for i in range(N):
            x_plus_h = x.copy()
            x_plus_h[i] += h
            fx_plus_h = ion_position_potential(x_plus_h)
            J[:, i] = (fx_plus_h - fx) / h
        
        cond = np.linalg.cond(J)
       # print(f"Jacobian condition number: {cond}")
        
        if cond > 1e15:
        #    print("Jacobian is ill-conditioned. Using regularization.")
            J += np.eye(N) * 1e-6
        
        try:
            dx = np.linalg.solve(J, -fx)
        except np.linalg.LinAlgError:
         #   print("LinAlgError occurred. Using pseudoinverse.")
            dx = -np.linalg.pinv(J) @ fx
        
        # Simple backtracking line search
        alpha = 1.0
        while np.linalg.norm(ion_position_potential(x + alpha * dx)) > fx_norm:
            alpha *= 0.5
            if alpha < 1e-4:
                break
        
        x += alpha * dx
        
        #print(f"Updated x: {x}")
        
        if np.any(np.isnan(x)):
         #   print("NaN detected in x. Stopping iterations.")
            break
    
    return x

@cuda.jit(device=True)
def lengthScale(nu, M, Z):
    if M is None:
        M = atomic_mass * 39.9626
    if Z is None:
        Z = 1
    return ((Z**2 * elementary_charge**2) / (4 * pi * epsilon_0 * M * nu**2))**(1/3)

def lengthScale_host(nu, M=None, Z=None):
    if M is None:
        M = atomic_mass * 39.9626
    if Z is None:
        Z = 1
    return ((Z**2 * elementary_charge**2) / (4 * pi * epsilon_0 * M * nu**2))**(1/3)

@cuda.jit
def makeRF0_kernel(m, q, w, Nr, Nz, Nrmid, dr, RF):
    i, j = cuda.grid(2)
    if i < Nr and j < Nz:
        C = -m * (w**2) / q
        RF[i, j] = -C * (Nrmid - i) * dr

def makeRF0(m, q, w, Nr, Nz, Nrmid, dr):
    RF = cp.ones((Nr, Nz), dtype=cp.float64)
    threads_per_block = (16, 16)
    blocks = ((Nr + threads_per_block[0] - 1) // threads_per_block[0],
              (Nz + threads_per_block[1] - 1) // threads_per_block[1])
    makeRF0_kernel[blocks, threads_per_block](m, q, w, Nr, Nz, Nrmid, dr, RF)
    return RF

@cuda.jit
def makeDC_kernel(m, q, w, Nr, Nz, Nzmid, dz, DC):
    i, j = cuda.grid(2)
    if i < Nr and j < Nz:
        C = -m * (w**2) / q
        DC[i, j] = -C * (Nzmid - j) * dz

def makeDC(m, q, w, Nr, Nz, Nzmid, dz):
    DC = cp.ones((Nr, Nz), dtype=cp.float64)
    threads_per_block = (16, 16)
    blocks = ((Nr + threads_per_block[0] - 1) // threads_per_block[0],
              (Nz + threads_per_block[1] - 1) // threads_per_block[1])
    makeDC_kernel[blocks, threads_per_block](m, q, w, Nr, Nz, Nzmid, dz, DC)
    return DC

def makeVf(Ni, q, m, l, wr, offsetr, offsetz, vbumpr, vbumpz):
    # Initialize vf array on GPU
    vf = cp.zeros((Ni, 7), dtype=cp.float64)
    
    # Calculate positions using the CUDA-enabled calcPositions function
    pos = calcPositions(Ni)  # This returns a NumPy array
    
    # Calculate length scale
    lscale = lengthScale_host(wr)  # Using the host version as it's called only once
    
    # Scale positions and convert to CuPy array
    scaledPos = cp.asarray(pos * lscale)
    
    # Set up initial values
    vf[:, 4] = q  # Set charge
    vf[:, 5] = m  # Set mass
    vf[:, 1] = -scaledPos  # Set z positions
    
    # Apply offsets and bumps to the specific ion (index l)
    vf[l, 0] += offsetr
    vf[l, 1] += offsetz
    vf[l, 2] += vbumpr
    vf[l, 3] += vbumpz
    
    return vf

@cuda.jit
def minDists_kernel(vf, vc, result):
    tid = cuda.grid(1)
    
    # Shared memory for partial results
    s_rid2 = cuda.shared.array(shape=(256,), dtype=float64)
    s_rii2 = cuda.shared.array(shape=(256,), dtype=float64)
    s_vid2 = cuda.shared.array(shape=(256,), dtype=float64)
    s_vii2 = cuda.shared.array(shape=(256,), dtype=float64)
    
    # Initialize shared memory
    s_rid2[tid] = 1e6
    s_rii2[tid] = 1e6
    s_vid2[tid] = 1e6
    s_vii2[tid] = 1e6
    
    Ni = vf.shape[0]
    Nc = vc.shape[0]
    
    # Each thread processes a subset of ion pairs
    for i in range(cuda.blockIdx.x, Ni, cuda.gridDim.x):
        for j in range(i+1, Ni):
            r = vf[i,0] - vf[j,0]
            z = vf[i,1] - vf[j,1]
            vr = vf[i,2] - vf[j,2]
            vz = vf[i,3] - vf[j,3]
            dist2 = r**2 + z**2
            v2 = vr**2 + vz**2
            if dist2 < s_rii2[tid]:
                s_vii2[tid] = v2
                s_rii2[tid] = dist2
        
        # Check each ion-dipole pair
        for j in range(Nc):
            r = vf[i,0] - vc[j,0]
            z = vf[i,1] - vc[j,1]
            vr = vf[i,2] - vc[j,2]
            vz = vf[i,3] - vc[j,3]
            dist2 = r**2 + z**2
            v2 = vr**2 + vz**2
            if dist2 < s_rid2[tid]:
                s_vid2[tid] = v2
                s_rid2[tid] = dist2
    
    # Synchronize threads
    cuda.syncthreads()
    
    # Reduce results
    s = 256 // 2
    while s > 0:
        if tid < s:
            if s_rid2[tid + s] < s_rid2[tid]:
                s_rid2[tid] = s_rid2[tid + s]
                s_vid2[tid] = s_vid2[tid + s]
            if s_rii2[tid + s] < s_rii2[tid]:
                s_rii2[tid] = s_rii2[tid + s]
                s_vii2[tid] = s_vii2[tid + s]
        cuda.syncthreads()
        s //= 2
    
    # Write final results
    if tid == 0:
        result[0] = math.sqrt(s_rid2[0])  # Use built-in math.sqrt
        result[1] = math.sqrt(s_rii2[0])
        result[2] = math.sqrt(s_vid2[0])
        result[3] = math.sqrt(s_vii2[0])

def minDists(vf, vc):
    # Ensure inputs are on GPU
    vf_gpu = cp.asarray(vf)
    vc_gpu = cp.asarray(vc)
    
    # Allocate result array on GPU
    result_gpu = cp.zeros(4, dtype=cp.float64)
    
    # Launch kernel
    threads_per_block = 256
    blocks = min(32, (vf.shape[0] + threads_per_block - 1) // threads_per_block)
    minDists_kernel[blocks, threads_per_block](vf_gpu, vc_gpu, result_gpu)
    
    # Transfer result back to CPU
    result = result_gpu.get()
    
    return result[0], result[1], result[2], result[3]

@cuda.jit(device=True)
def collisionMode_device(rii, rid, a, e=0.3):
    # Convert inputs to float to ensure scalar operations
    rii_f = float(rii)
    rid_f = float(rid)
    a_f = float(a)
    e_f = float(e)
    return (a_f * rii_f**2) / (rid_f**5) > e_f

@cuda.jit(device=True)
def collisionMode_device(rii, rid, a, e):
    return (a * rii**2) / (rid**5) > e

@cuda.jit
def collisionMode_kernel(rii, rid, a, e, result):
    i = cuda.grid(1)
    if i < result.size:
        result[i] = collisionMode_device(rii[i], rid[i], a, e)

def collisionMode(rii, rid, a, e=0.3):
    # Ensure inputs are arrays
    rii = np.atleast_1d(rii).astype(np.float64)
    rid = np.atleast_1d(rid).astype(np.float64)
    
    # Convert to CuPy arrays
    rii_gpu = cp.asarray(rii)
    rid_gpu = cp.asarray(rid)
    result_gpu = cp.zeros(rii_gpu.shape, dtype=cp.bool_)
    
    # Ensure a and e are scalar float64
    a = np.float64(a)
    e = np.float64(e)
    
    # Set up the grid
    threads_per_block = 256
    blocks = (rii_gpu.size + threads_per_block - 1) // threads_per_block
    
    # Launch the kernel
    collisionMode_kernel[(blocks,), (threads_per_block,)](rii_gpu, rid_gpu, a, e, result_gpu)
    
    # Return the result as a NumPy array
    return cp.asnumpy(result_gpu)


@cuda.jit
def updatePoss_kernel(vf, dr, dz, dt, Nr, Nz, Nrmid, Nzmid):
    i = cuda.grid(1)
    if i < vf.shape[0]:
        vf[i, 0] += vf[i, 2] * dt
        vf[i, 1] += vf[i, 3] * dt
        
        rCell = ptovPos_device(vf[i, 0], Nrmid, dr)
        zCell = ptovPos_device(vf[i, 1], Nzmid, dz)
        
        if rCell > Nr - 2 or rCell < 1 or zCell > Nz - 2 or zCell < 1:
            vf[i, :] = 0.0
            vf[i, 0] = 2.0
            vf[i, 1] = 2.0
            vf[i, 2] = 0.0
            vf[i, 3] = 0.0
            vf[i, 5] = 1e6

def updatePoss(vf, dr, dz, dt, Nr, Nz, Nrmid, Nzmid):
    vf_gpu = cp.asarray(vf)
    
    threads_per_block = 256
    blocks = (vf_gpu.shape[0] + threads_per_block - 1) // threads_per_block
    
    # Convert dr and dz to scalars
    dr_scalar = float(dr.item()) if isinstance(dr, (np.ndarray, cp.ndarray)) else float(dr)
    dz_scalar = float(dz.item()) if isinstance(dz, (np.ndarray, cp.ndarray)) else float(dz)
    
    updatePoss_kernel[blocks, threads_per_block](vf_gpu, dr_scalar, dz_scalar, dt, Nr, Nz, Nrmid, Nzmid)
    
    return vf_gpu.get()

@cuda.jit
def updateVels_kernel(vf, Erf, Ezf, dt):
    i = cuda.grid(1)
    if i < vf.shape[0]:
        Fr = vf[i, 4] * Erf[i]
        Fz = vf[i, 4] * Ezf[i]
        vf[i, 2] += Fr * dt / vf[i, 5]
        vf[i, 3] += Fz * dt / vf[i, 5]

def updateVels(vf, Erf, Ezf, dt):
    vf_gpu = cp.asarray(vf)
    Erf_gpu = cp.asarray(Erf)
    Ezf_gpu = cp.asarray(Ezf)
    
    threads_per_block = 256
    blocks = (vf_gpu.shape[0] + threads_per_block - 1) // threads_per_block
    
    updateVels_kernel[blocks, threads_per_block](vf_gpu, Erf_gpu, Ezf_gpu, dt)
    
    return vf_gpu.get()

def ptovPos(pos, Nmid, dcell):
    pos_gpu = cp.asarray(pos)
    return (pos_gpu / float(dcell) + float(Nmid)).get()


@cuda.jit
def solveFields_kernel(vf, ErDC, EzDC, ErAC, EzAC, Nrmid, Nzmid, dr, dz, Erf2, Ezf2):
    i = cuda.grid(1)
    Ni = vf.shape[0]
    
    if i < Ni:
        eps0 = 8.854e-12
        C1 = 4 * math.pi * eps0
        
        r = vf[i, 0]
        z = vf[i, 1]
        
        jCell = ptovPos_device(r, Nrmid, dr)
        kCell = ptovPos_device(z, Nzmid, dz)
        
        # Ensure jCell and kCell are within bounds
        jCell = max(0, min(jCell, ErDC.shape[0] - 1))
        kCell = max(0, min(kCell, ErDC.shape[1] - 1))
        
        # Add trap fields
        Erf2[i] = ErDC[jCell, kCell] + ErAC[jCell, kCell]
        Ezf2[i] = EzDC[jCell, kCell] + EzAC[jCell, kCell]
        
        # Calculate fields from other ions
        for j in range(Ni):
            if j != i:
                rdist = vf[j, 0] - r
                zdist = vf[j, 1] - z
                sqDist = rdist**2 + zdist**2
                if sqDist > 1e-20:  # Avoid division by zero
                    dist = math.sqrt(sqDist)
                    projR = rdist / dist
                    projZ = zdist / dist
                    
                    E_field = vf[j, 4] / (C1 * sqDist)
                    Erf2[i] += -projR * E_field
                    Ezf2[i] += -projZ * E_field

def solveFields(vf, ErDC, EzDC, ErAC, EzAC, Nrmid, Nzmid, Ni, dr, dz):
    # Move data to GPU
    vf_gpu = cp.asarray(vf, dtype=cp.float64)
    ErDC_gpu = cp.asarray(ErDC, dtype=cp.float64)
    EzDC_gpu = cp.asarray(EzDC, dtype=cp.float64)
    ErAC_gpu = cp.asarray(ErAC, dtype=cp.float64)
    EzAC_gpu = cp.asarray(EzAC, dtype=cp.float64)
    
    # Ensure scalar values are floats
    Nrmid = float(Nrmid)
    Nzmid = float(Nzmid)
    dr = float(dr.item()) if isinstance(dr, (np.ndarray, cp.ndarray)) else float(dr)
    dz = float(dz.item()) if isinstance(dz, (np.ndarray, cp.ndarray)) else float(dz)
    
    # Initialize output arrays on GPU
    Erf2_gpu = cp.zeros(Ni, dtype=cp.float64)
    Ezf2_gpu = cp.zeros(Ni, dtype=cp.float64)
    
    # Launch kernel
    threads_per_block = 256
    blocks = (Ni + threads_per_block - 1) // threads_per_block
    
    solveFields_kernel[blocks, threads_per_block](
        vf_gpu, ErDC_gpu, EzDC_gpu, ErAC_gpu, EzAC_gpu,
        Nrmid, Nzmid, dr, dz, Erf2_gpu, Ezf2_gpu
    )
    
    # Transfer results back to CPU
    Erf2 = cp.asnumpy(Erf2_gpu)
    Ezf2 = cp.asnumpy(Ezf2_gpu)
    
    return Erf2, Ezf2

@cuda.jit(device=True)
def my_sqrt(x):
    return x ** 0.5

@cuda.jit(device=True)
def my_abs(x):
    return x if x >= 0 else -x

@cuda.jit
def collisionParticlesFields_kernel(vf, vc, ErDC, EzDC, ErAC, EzAC, dr, dz, dt, Nrmid, Nzmid, Erfi, Ezfi, Erfc, Ezfc):
    i = cuda.grid(1)
    Ni = vf.shape[0]
    Nc = vc.shape[0]
    C1 = 4 * math.pi * eps0
    
    if i < Nc:  # Process collisional particles
        jCell = ptovPos_device(vc[i, 0], Nrmid, dr)
        kCell = ptovPos_device(vc[i, 1], Nzmid, dz)
        
        # Initialize interpolated fields
        Erfc[i, 0] = ErDC[jCell, kCell] + ErAC[jCell, kCell]
        Ezfc[i, 0] = EzDC[jCell, kCell] + EzAC[jCell, kCell]
        
        if 0 < jCell < ErDC.shape[0] - 1 and 0 < kCell < ErDC.shape[1] - 1:
            Erfc[i, 1] = ((ErDC[jCell+1, kCell] + ErAC[jCell+1, kCell]) - (ErDC[jCell-1, kCell] + ErAC[jCell-1, kCell])) / (2 * dr)
            Ezfc[i, 1] = ((EzDC[jCell, kCell+1] + EzAC[jCell, kCell+1]) - (EzDC[jCell, kCell-1] + EzAC[jCell, kCell-1])) / (2 * dz)
        
        for j in range(Ni):
            rdist = vf[j, 0] - vc[i, 0]
            zdist = vf[j, 1] - vc[i, 1]
            sqDist = rdist**2 + zdist**2
            dist = my_sqrt(sqDist)
            projR = rdist / dist
            projZ = zdist / dist
            
            E_field = vf[j, 4] / (C1 * sqDist)
            Erfc[i, 0] -= projR * E_field
            Ezfc[i, 0] -= projZ * E_field
            Erfc[i, 1] += 2 * projR * E_field / dist
            Ezfc[i, 1] += 2 * projZ * E_field / dist
        
        # Calculate dipole moment and update velocity
        if vc[i, 6] != 0.0:
            pR = -2 * 3.14159265358979323846 * eps0 * vc[i, 6] * Erfc[i, 0]
            pZ = -2 * 3.14159265358979323846 * eps0 * vc[i, 6] * Ezfc[i, 0]
            pTot = my_sqrt(pR**2 + pZ**2)
            
            Fr = my_abs(pR) * Erfc[i, 1]
            Fz = my_abs(pZ) * Ezfc[i, 1]
            
            vc[i, 2] += Fr * dt / vc[i, 5]
            vc[i, 3] += Fz * dt / vc[i, 5]
    
    cuda.syncthreads()
    
    if i < Ni:  # Process ions
        for j in range(Nc):
            if vc[j, 6] != 0.0:
                rdist = vf[i, 0] - vc[j, 0]
                zdist = vf[i, 1] - vc[j, 1]
                sqDist = rdist**2 + zdist**2
                dist = my_sqrt(sqDist)
                Rhatr = rdist / dist
                Rhatz = zdist / dist
                
                pR = -2 * 3.14159265358979323846 * eps0 * vc[j, 6] * Erfc[j, 0]
                pZ = -2 * 3.14159265358979323846 * eps0 * vc[j, 6] * Ezfc[j, 0]
                
                Erfi[i] -= my_abs(pR) * (2 * Rhatr) / (C1 * dist**3)
                Ezfi[i] -= my_abs(pZ) * (2 * Rhatz) / (C1 * dist**3)


def collisionParticlesFields(vf, vc, Ni, ErDC, EzDC, ErAC, EzAC, dr, dz, dt, Nrmid, Nzmid):
    # Move data to GPU
    vf_gpu = cp.asarray(vf)
    vc_gpu = cp.asarray(vc)
    ErDC_gpu = cp.asarray(ErDC)
    EzDC_gpu = cp.asarray(EzDC)
    ErAC_gpu = cp.asarray(ErAC)
    EzAC_gpu = cp.asarray(EzAC)
    
    # Ensure dr and dz are 1D arrays
    '''dr_gpu = cp.asarray([dr] if np.isscalar(dr) else dr)
    dz_gpu = cp.asarray([dz] if np.isscalar(dz) else dz)'''
    
    Nc = vc.shape[0]
    
    # Initialize output arrays on GPU
    Erfi_gpu = cp.zeros(Ni, dtype=cp.float64)
    Ezfi_gpu = cp.zeros(Ni, dtype=cp.float64)
    Erfc_gpu = cp.zeros((Nc, 2), dtype=cp.float64)
    Ezfc_gpu = cp.zeros((Nc, 2), dtype=cp.float64)
    
    # Launch kernel
    threads_per_block = 256
    blocks = (max(Ni, Nc) + threads_per_block - 1) // threads_per_block

    dr_scalar = float(dr) if isinstance(dr, (float, int)) else float(dr.item())
    dz_scalar = float(dz) if isinstance(dz, (float, int)) else float(dz.item())
    
    collisionParticlesFields_kernel[blocks, threads_per_block](
        vf_gpu, vc_gpu, ErDC_gpu, EzDC_gpu, ErAC_gpu, EzAC_gpu,
        dr_scalar, dz_scalar, dt, Nrmid, Nzmid, Erfi_gpu, Ezfi_gpu, Erfc_gpu, Ezfc_gpu
    )
    
    # Transfer results back to CPU
    vc = cp.asnumpy(vc_gpu)
    Erfi = cp.asnumpy(Erfi_gpu)
    Ezfi = cp.asnumpy(Ezfi_gpu)
    Erfc = cp.asnumpy(Erfc_gpu)
    Ezfc = cp.asnumpy(Ezfc_gpu)
    
    return vc, Erfi, Ezfi, Erfc, Ezfc

MAX_NR = 10001
MAX_NZ = 10001

'''@cuda.jit
def run_single_shot_kernel(vf, RF, DC, Nr, Nz, dr, dz, Nrmid, Nzmid, dtSmall, dtLarge, dtCollision, 
                           velocities, angles, offsets, ion_collided, max_hypotenuse, q, mH2, aH2, 
                           Nt, reorder_results):
    i = cuda.grid(1)
    if i < velocities.shape[0]:
        # Process for shot i
        velocity = velocities[i]
        angle_choice = angles[i]
        offset_choice = offsets[i]
        ion_collided_i = ion_collided[i]
        Nt_i = Nt[i]

        r = -math.cos(angle_choice) * max_hypotenuse
        z = vf[ion_collided_i, 1] + math.sin(angle_choice) * max_hypotenuse + offset_choice
        vz = -1 * velocity * math.sin(angle_choice)
        vr = abs(velocity * math.cos(angle_choice))

        vc = cuda.local.array((1, 7), dtype=numba.float64)
        vc[0, 0] = r
        vc[0, 1] = z
        vc[0, 2] = vr
        vc[0, 3] = vz
        vc[0, 4] = q
        vc[0, 5] = mH2
        vc[0, 6] = aH2

        reorder = 0
        for t in range(Nt_i):
            rid, rii, vid, vii = minDists_device(vf, vc, vf.shape[0], 1)
            collision = collisionMode(rii, rid, vc[0, 6], 0.1)
            
            if collision:
                dtNow = rid * 0.01 / (5 * vid)
            else:
                dtNow = dtSmall
            
            dtNow = max(dtNow, dtCollision)
            
            Erf, Ezf = solveFields_device(vf, RF, DC, RF, RF, Nrmid, Nzmid, vf.shape[0], dr, dz)
            
            if vc[0, 5] < 1e6:
                vc, Erfic, Ezfic, Erfc, Ezfc = collisionParticlesFields_device(vf, vc, vf.shape[0], RF, DC, RF, RF, dr, dz, dtNow, Nrmid, Nzmid)
                vc = updatePoss_device(vc, dr, dz, dtNow, Nr, Nz, Nrmid, Nzmid)
                for j in range(vf.shape[0]):
                    Erf[j] += Erfic[j]
                    Ezf[j] += Ezfic[j]
            else:
                dtNow = dtLarge
            
            vf = updateVels_device(vf, Erf, Ezf, dtNow)
            vf = updatePoss_device(vf, dr, dz, dtNow, Nr, Nz, Nrmid, Nzmid)
            
            if math.isnan(vf[0, 0]):
                reorder = 1
                break
            
            if vf[0, 5] > 1e5:
                reorder = 2
                break
            
            for j in range(1, vf.shape[0]):
                if vf[j, 1] > vf[j-1, 1]:
                    reorder = 1
                    break
            
            if reorder != 0:
                break

        reorder_results[i] = reorder'''


@cuda.jit
def update_arrays_kernel(vf, vc, rs, zs, rcolls, zcolls, vrs, vzs, vrcolls, vzcolls, i):
    idx = cuda.grid(1)
    if idx < vf.shape[0]:
        rs[idx, i] = vf[idx, 0]
        zs[idx, i] = vf[idx, 1]
        vrs[idx, i] = vf[idx, 2]
        vzs[idx, i] = vf[idx, 3]
    
    if idx < vc.shape[0]:
        rcolls[idx, i] = vc[idx, 0]
        zcolls[idx, i] = vc[idx, 1]
        vrcolls[idx, i] = vc[idx, 2]
        vzcolls[idx, i] = vc[idx, 3]

@cuda.jit(device=True)
def mcCollision(vf, rc, zc, vrc, vzc, qc, mc, ac, Nt, dtSmall, RF, DC, Nr, Nz, dr, dz, dtLarge, dtCollision, eii=0.01, eid=0.01):
    # Convert all inputs to CuPy arrays
    vf_gpu = to_cupy(vf)
    rc, zc, vrc, vzc, qc, mc, ac = map(to_cupy, [rc, zc, vrc, vzc, qc, mc, ac])
    RF, DC, dr, dz, dtSmall, dtLarge, dtCollision = map(to_cupy, [RF, DC, dr, dz, dtSmall, dtLarge, dtCollision])

    # Start by defining some constants
    Nrmid = int((Nr-1)/2)
    Nzmid = int((Nz-1)/2)
    Ni = vf_gpu.shape[0]
    Nc = 1

    # Initialize arrays on GPU
    vc_gpu = cp.zeros((Nc, 7), dtype=cp.float64)
    vc_gpu[0] = cp.array([rc.item(), zc.item(), vrc.item(), vzc.item(), qc.item(), mc.item(), ac.item()])

    rs_gpu = cp.zeros((Ni, Nt), dtype=cp.float64)
    zs_gpu = cp.zeros((Ni, Nt), dtype=cp.float64)
    vrs_gpu = cp.zeros((Ni, Nt), dtype=cp.float64)
    vzs_gpu = cp.zeros((Ni, Nt), dtype=cp.float64)
    rcolls_gpu = cp.zeros((Nc, Nt), dtype=cp.float64)
    zcolls_gpu = cp.zeros((Nc, Nt), dtype=cp.float64)
    vrcolls_gpu = cp.zeros((Nc, Nt), dtype=cp.float64)
    vzcolls_gpu = cp.zeros((Nc, Nt), dtype=cp.float64)

    nullFields_gpu = cp.zeros((Nr, Nz), dtype=cp.float64)

    reorder = 0
    dtNow = dtSmall.item()

    # CUDA kernel configuration
    threads_per_block = 256
    blocks = (max(Ni, Nc) + threads_per_block - 1) // threads_per_block

    dr_scalar = float(dr.item()) if isinstance(dr, (np.ndarray, cp.ndarray)) else float(dr)
    dz_scalar = float(dz.item()) if isinstance(dz, (np.ndarray, cp.ndarray)) else float(dz)

    for i in range(Nt):
        rid, rii, vid, vii = minDists(vf_gpu, vc_gpu)
        collision = collisionMode(rii, rid, vc_gpu[0, 6], 0.1)
        
        if collision:
            dtNow = (rid * eid / (5 * vid)).item()
        else:
            dtNow = dtSmall.item()
        
        if dtNow < dtCollision.item():
            dtNow = dtCollision.item()

        '''print("vf shape:", vf_gpu.shape, "dtype:", vf_gpu.dtype)
        print("ErDC shape:", nullFields_gpu.shape, "dtype:", nullFields_gpu.dtype)
        print("EzDC shape:", DC.shape, "dtype:", DC.dtype)
        print("ErAC shape:", RF.shape, "dtype:", RF.dtype)
        print("EzAC shape:", nullFields_gpu.shape, "dtype:", nullFields_gpu.dtype)
        print("Nrmid:", Nrmid, "Nzmid:", Nzmid)
        print("dr:", dr, "dz:", dz)'''

        Erfi, Ezfi = solveFields(vf_gpu, nullFields_gpu, DC, RF, nullFields_gpu, Nrmid, Nzmid, Ni, dr, dz)

        if vc_gpu[0, 5] < 1e6:
            vc_gpu, Erfic, Ezfic, Erfc, Ezfc = collisionParticlesFields(vf_gpu, vc_gpu, Ni, nullFields_gpu, DC, RF, nullFields_gpu, dr_scalar, dz_scalar, dtNow, Nrmid, Nzmid)
            vc_gpu = updatePoss(vc_gpu, dr_scalar, dz_scalar, dtNow, Nr, Nz, Nrmid, Nzmid)
            Erfi += Erfic
            Ezfi += Ezfic
        else:
            dtNow = dtLarge.item()

        vf_gpu = updateVels(vf_gpu, Erfi, Ezfi, dtNow)
        vf_gpu = updatePoss(vf_gpu, dr_scalar, dz_scalar, dtNow, Nr, Nz, Nrmid, Nzmid)

        # Update arrays using CUDA kernel
        update_arrays_kernel[blocks, threads_per_block](vf_gpu, vc_gpu, rs_gpu, zs_gpu, rcolls_gpu, zcolls_gpu, vrs_gpu, vzs_gpu, vrcolls_gpu, vzcolls_gpu, i)

        vf_gpu = cp.asarray(vf_gpu)
        vc_gpu = cp.asarray(vc_gpu)
        if cp.any(cp.isnan(vf_gpu)):
            print("NaN detected in ion parameters!")
            vf_gpu.fill(0.0)
            vf_gpu[:, 5] = 1e1
            vc_gpu.fill(0.0)
            vc_gpu[:, 5] = 1e1
            break

        if cp.sum(vf_gpu[:, 5]) > 1e5:
            reorder += 2
            break

        for j in range(1, Ni):
            if zs_gpu[j, i] > zs_gpu[j-1, i]:
                reorder += 1
                Nt = i + 1000
                break

    return reorder

'''@cuda.jit
def parallel_shots_kernel(vf_list, r_list, z_list, vr_list, vz_list, q, m, a, Nt_list, dtSmall, RF, DC, Nr, Nz, dr, dz, dtLarge, dtCollision, reorder_list):
    shot_idx = cuda.grid(1)
    if shot_idx < vf_list.shape[0]:
        vf = vf_list[shot_idx]
        r = r_list[shot_idx]
        z = z_list[shot_idx]
        vr = vr_list[shot_idx]
        vz = vz_list[shot_idx]
        Nt = Nt_list[shot_idx]
        
        reorder = mcCollision(vf, r, z, vr, vz, q, m, a, Nt, dtSmall, RF, DC, Nr, Nz, dr, dz, dtLarge, dtCollision)
        reorder_list[shot_idx] = reorder

def parallel_simulation(grid_size, Ni, shots, q, m, l, wz, offsetr, offsetz, vbumpr, vbumpz, v, boltzDist, angles, offsets, max_hypotenuse, dtSmall, RF, DC, Nr, Nz, dr, dz, dtLarge, dtCollision):
    print(f"Starting simulation with grid size {grid_size}, ion count {Ni}, and shots {shots}")
    
    start_time = time.perf_counter()
    file_name = f"simulation_results/{Ni}ionSimulation_{grid_size}_{shots}shots.txt"

    # Prepare input data for all shots
    vf_list = cp.array([makeVf(Ni, 1.0*q, m, l, wz, offsetr, offsetz, vbumpr, vbumpz) for _ in range(shots)])
    velocities = cp.array([random.choices(v, weights=boltzDist)[0] for _ in range(shots)])
    angle_choices = cp.array([random.choice(angles) for _ in range(shots)])
    offset_choices = cp.array([random.choice(offsets) for _ in range(shots)])
    ion_collided_list = cp.array([random.randint(0, Ni-1) for _ in range(shots)])

    r_list = -cp.cos(angle_choices) * max_hypotenuse
    z_list = vf_list[cp.arange(shots), ion_collided_list, 1] + cp.sin(angle_choices) * max_hypotenuse + offset_choices
    vz_list = -velocities * cp.sin(angle_choices)
    vr_list = cp.abs(velocities * cp.cos(angle_choices))

    Nt_list = cp.where(velocities < 200, 700000, cp.where(velocities < 1500, 400000, 250000))

    # Prepare output array
    reorder_list = cp.zeros(shots, dtype=cp.int32)

    # Launch kernel
    threads_per_block = 256
    blocks = (shots + threads_per_block - 1) // threads_per_block
    parallel_shots_kernel[blocks, threads_per_block](vf_list, r_list, z_list, vr_list, vz_list, q, m, aH2, Nt_list, dtSmall, RF, DC, Nr, Nz, dr, dz, dtLarge, dtCollision, reorder_list)

    # Collect results
    results = cp.stack([cp.full(shots, wz), velocities, ion_collided_list + 1, angle_choices, offset_choices, reorder_list], axis=1)

    # Write results to file
    with open(file_name, "w") as f:
        f.write("axial trapping frequency (MHz) \t velocity(m/s) \t ion collided with \t angle(rads) \t collision offset(m) \t reorder? (1 is reorder 2 is ejection) \n")
        cp.savetxt(f, results.get(), delimiter='\t')

    finish_time = time.perf_counter()
    timeTaken = finish_time - start_time
    
    return timeTaken'''

def Boltz(m, T, vmin=0, vmax=5000, bins=100):
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

def create_formatted_table_file(computation_times, output_file):
    grid_sizes = sorted(set(int(size) for size in computation_times.keys()))
    ion_counts = sorted(set(int(count) for size in computation_times.values() for count in size.keys()))
    shot_sizes = sorted(set(int(shots) for size in computation_times.values() for count in size.values() for shots in count.keys()))

    with open(output_file, 'w') as f:
        # Write header
        f.write("Computational Times in seconds\n")
        f.write(" ".join([f"{size//1000}k".rjust(8) for size in grid_sizes]) + "\n")

        for ion_count in ion_counts:
            for shot_size in shot_sizes:
                row = [f"{ion_count}".rjust(2)]
                for grid_size in grid_sizes:
                    time = computation_times.get(str(grid_size), {}).get(str(ion_count), {}).get(str(shot_size), None)
                    if time is not None:
                        row.append(f"{time:.0f}".rjust(8))
                    else:
                        row.append(" ".rjust(8))
                row.append(str(shot_size).rjust(8))
                f.write(" ".join(row) + "\n")
            f.write("\n")  # Add a blank line between different ion counts

    print(f"Formatted table has been written to {output_file}")

# Define useful constants
amu = 1.67e-27
eps0 = 8.854e-12
qe = 1.6e-19 # SI units 

# Define physical params
m = 40. *amu
q = 1. *qe
wr = 2*np.pi*3e6 # SI units

# Define sim params
Dr = 30001.5e-9
Dz = 90001.5e-9 # physical width in m of the sim

# wz can be varied if desired
wz = 2*np.pi*1e6

aH2 = 8e-31 # dipole moment of H2 in SI units
mH2 = 2.0*amu # mass of H2 in kg

dtSmall = 1e-12
dtCollision = 1e-16
dtLarge = 1e-10 # length of a time step in s

sigmaV = 100e-6 # fall-off of potential outside trapping region
dv = 20.0 # bin size for particle speed in determining if collision occurs
vmax = 5000 # maximum particle speed we allow
l = 1
vbumpr = 0.00e0
vbumpz = -0.0e0 # starting velocity in r and z of the lth ion in the chain
offsetz = 0.0e-7
offsetr = 0.0e-8 # starting distance from eq. in r and z of the lth ion in the chain

# Simulation parameters
T = 300
collisionalMass = 2
vMin = 50
vMax = 7000
numBins = 1000
boltzDist = Boltz(collisionalMass, T, vMin, vMax, numBins)
v = np.linspace(vMin, vMax, numBins)
angles = np.linspace(-np.pi/2, np.pi/2, 100)
offsets = np.linspace(-2e-9, 2e-9, 200)
max_hypotenuse = 1.5e-5

buffer_size = 10

grid_sizes = [10001]  # Array of grid sizes to test
ion_counts = [2, 3]  # Array of ion counts to test
shot_sizes = [1000]  # Array of shot sizes to test

# File to store computation times
computation_times_file = "computation_times.json"
formatted_table_file = "computation_times_table.txt"

# Load existing computation times if file exists
if os.path.exists(computation_times_file):
    with open(computation_times_file, 'r') as f:
        computation_times = json.load(f)
else:
    computation_times = {}

os.makedirs("simulation_results", exist_ok=True)

def to_cupy(x):
    if isinstance(x, (int, float)):
        return cp.array([x])
    elif isinstance(x, np.ndarray):
        return cp.asarray(x)
    elif isinstance(x, cp.ndarray):
        return x
    else:
        raise TypeError(f"Unsupported type: {type(x)}")

'''for i in range(Nt):
    vf = cp.asarray(makeVf(Ni, 1.0*q, m, l, wz, offsetr, offsetz, vbumpr, vbumpz))
    z = vf[ion_collided,1] + cp.sin(angle_choice)*max_hypotenuse + offset_choice
    
    Nc = 1
    # Create vc as a CuPy array, ensuring all elements are converted to CuPy arrays
    vc = cp.zeros((Nc, 7), dtype=cp.float64)
    vc[0,0] = to_cupy(r).item()
    vc[0,1] = to_cupy(z).item()
    vc[0,2] = to_cupy(vr).item()
    vc[0,3] = to_cupy(vz).item()
    vc[0,4] = to_cupy(q).item()
    vc[0,5] = to_cupy(mH2).item()
    vc[0,6] = to_cupy(aH2).item()
    
    rid, rii, vid, vii = minDists(vf, vc)
    collision = collisionMode(rii, rid, vc[0, 6], 0.1)
    vc_new, Erfic, Ezfic, Erfc, Ezfc = collisionParticlesFields(vf, vc, Ni, nullFields, DC, RF, nullFields, dr, dz, dtCollision, Nrmid, Nzmid)
    print("Input vc:")
    print(vc)
    print("\nOutput vc:")
    print(vc_new)
    print("\nErfic:", Erfic)
    print("Ezfic:", Ezfic)
    print("Erfc:", Erfc)
    print("Ezfc:", Ezfc)
    print("\nErfic shape:", Erfic.shape)
    print("Ezfic shape:", Ezfic.shape)
    print("Erfc shape:", Erfc.shape)
    print("Ezfc shape:", Ezfc.shape)
    print("-" * 50)'''


'''if __name__ == "__main__":
    for grid_size in grid_sizes:
        Nr = Nz = grid_size
        Nrmid = Nzmid = (Nr-1)/2
        dr = Dr/float(Nr)
        dz = Dz/float(Nz)

        # Recalculate RF and DC fields for each grid size
        RF = makeRF0(m, q, wr, Nr, Nz, Nrmid, dr)
        DC = makeDC(m, q, wz, Nz, Nr, Nzmid, dz)

        for Ni in ion_counts:
            for shots in shot_sizes:
                timeTaken = parallel_simulation(grid_size, Ni, shots, q, m, l, wz, offsetr, offsetz, vbumpr, vbumpz, v, boltzDist, angles, offsets, max_hypotenuse, dtSmall, RF, DC, Nr, Nz, dr, dz, dtLarge, dtCollision)
                
                # Update computation times
                if str(grid_size) not in computation_times:
                    computation_times[str(grid_size)] = {}
                if str(Ni) not in computation_times[str(grid_size)]:
                    computation_times[str(grid_size)][str(Ni)] = {}
                computation_times[str(grid_size)][str(Ni)][str(shots)] = timeTaken
                
                # Save updated computation times
                with open(computation_times_file, 'w') as f:
                    json.dump(computation_times, f, indent=2)
                
                print(f"Completed simulation for grid size {grid_size}, ion count {Ni}, and shots {shots}. It took {timeTaken} seconds!")

    create_formatted_table_file(computation_times, formatted_table_file)
    print("All simulations completed successfully!")'''

def run_single_shot(args):
    vf, r, z, vr, vz, q, m, a, Nt, dtSmall, RF, DC, Nr, Nz, dr, dz, dtLarge, dtCollision = args
    return mcCollision(vf, r, z, vr, vz, q, m, a, Nt, dtSmall, RF, DC, Nr, Nz, dr, dz, dtLarge, dtCollision)

def parallel_simulation(grid_size, Ni, shots, q, m, l, wz, offsetr, offsetz, vbumpr, vbumpz, v, boltzDist, angles, offsets, max_hypotenuse, dtSmall, RF, DC, Nr, Nz, dr, dz, dtLarge, dtCollision):
    print(f"Starting simulation with grid size {grid_size}, ion count {Ni}, and shots {shots}")
    
    start_time = time.perf_counter()
    file_name = f"simulation_results/{Ni}ionSimulation_{grid_size}_{shots}shots.txt"

    # Prepare input data for all shots
    vf_list = [makeVf(Ni, 1.0*q, m, l, wz, offsetr, offsetz, vbumpr, vbumpz) for _ in range(shots)]
    velocities = [random.choices(v, weights=boltzDist)[0] for _ in range(shots)]
    angle_choices = [random.choice(angles) for _ in range(shots)]
    offset_choices = [random.choice(offsets) for _ in range(shots)]
    ion_collided_list = [random.randint(0, Ni-1) for _ in range(shots)]

    r_list = [-np.cos(angle) * max_hypotenuse for angle in angle_choices]
    z_list = [vf[ion, 1] + np.sin(angle) * max_hypotenuse + offset 
              for vf, ion, angle, offset in zip(vf_list, ion_collided_list, angle_choices, offset_choices)]
    vz_list = [-velocity * np.sin(angle) for velocity, angle in zip(velocities, angle_choices)]
    vr_list = [abs(velocity * np.cos(angle)) for velocity, angle in zip(velocities, angle_choices)]

    Nt_list = [700000 if v < 200 else 400000 if v < 1500 else 250000 for v in velocities]

    # Prepare arguments for multiprocessing
    args_list = [(vf, r, z, vr, vz, q, m, aH2, Nt, dtSmall, RF, DC, Nr, Nz, dr, dz, dtLarge, dtCollision)
                 for vf, r, z, vr, vz, Nt in zip(vf_list, r_list, z_list, vr_list, vz_list, Nt_list)]

    # Use multiprocessing to run mcCollision in parallel
    with Pool() as pool:
        results = pool.map(run_single_shot, args_list)

    # Collect results
    results_array = np.column_stack([
        np.full(shots, wz),
        velocities,
        np.array(ion_collided_list) + 1,
        angle_choices,
        offset_choices,
        results
    ])

    # Write results to file
    with open(file_name, "w") as f:
        f.write("axial trapping frequency (MHz) \t velocity(m/s) \t ion collided with \t angle(rads) \t collision offset(m) \t reorder? (1 is reorder 2 is ejection) \n")
        np.savetxt(f, results_array, delimiter='\t')

    finish_time = time.perf_counter()
    timeTaken = finish_time - start_time
    
    return timeTaken

if __name__ == "__main__":
    for grid_size in grid_sizes:
        Nr = Nz = grid_size
        Nrmid = Nzmid = (Nr-1)/2
        dr = Dr/float(Nr)
        dz = Dz/float(Nz)

        # Recalculate RF and DC fields for each grid size
        RF = makeRF0(m, q, wr, Nr, Nz, Nrmid, dr)
        DC = makeDC(m, q, wz, Nz, Nr, Nzmid, dz)

        for Ni in ion_counts:
            for shots in shot_sizes:
                timeTaken = parallel_simulation(grid_size, Ni, shots, q, m, l, wz, offsetr, offsetz, vbumpr, vbumpz, v, boltzDist, angles, offsets, max_hypotenuse, dtSmall, RF, DC, Nr, Nz, dr, dz, dtLarge, dtCollision)
                
                # Update computation times
                if str(grid_size) not in computation_times:
                    computation_times[str(grid_size)] = {}
                if str(Ni) not in computation_times[str(grid_size)]:
                    computation_times[str(grid_size)][str(Ni)] = {}
                computation_times[str(grid_size)][str(Ni)][str(shots)] = timeTaken
                
                # Save updated computation times
                with open(computation_times_file, 'w') as f:
                    json.dump(computation_times, f, indent=2)
                
                print(f"Completed simulation for grid size {grid_size}, ion count {Ni}, and shots {shots}. It took {timeTaken} seconds!")

    create_formatted_table_file(computation_times, formatted_table_file)
    print("All simulations completed successfully!")