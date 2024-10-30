import numpy as np
import cupy as cp
from numba import cuda
import math

@cuda.jit
def ion_position_potential_kernel(x, result):
    i = cuda.grid(1)
    if i < x.size:
        sum1 = 0.0
        sum2 = 0.0
        for n in range(i):
            diff = x[i] - x[n]
            if abs(diff) > 1e-10:
                sum1 += 1 / (diff * diff)
        for n in range(i + 1, x.size):
            diff = x[i] - x[n]
            if abs(diff) > 1e-10:
                sum2 += 1 / (diff * diff)
        result[i] = x[i] - sum1 + sum2

def ion_position_potential(x):
    x_gpu = cp.asarray(x)
    result = cp.zeros_like(x_gpu)
    threads_per_block = 256
    blocks = (x_gpu.size + threads_per_block - 1) // threads_per_block
    ion_position_potential_kernel[blocks, threads_per_block](x_gpu, result)
    return cp.asnumpy(result)

def calcPositions(N):
    estimated_extreme = 0.481 * N**0.765
    x = np.linspace(-estimated_extreme, estimated_extreme, N)
    for _ in range(1000):
        x = x - 0.1 * ion_position_potential(x)
    return x

def lengthScale(nu, M=None, Z=None):
    if M is None:
        M = 39.9626 * 1.66053906660e-27
    if Z is None:
        Z = 1
    e = 1.602176634e-19
    eps0 = 8.8541878128e-12
    return ((Z**2 * e**2) / (4 * math.pi * eps0 * M * nu**2))**(1/3)

def ptovPos(pos, Nmid, dcell):
    return (pos / dcell + float(Nmid))

@cuda.jit
def make_rf0_kernel(m, q, w, Nr, Nz, Nrmid, dr, RF):
    i, j = cuda.grid(2)
    if i < Nr and j < Nz:
        C = -m * (w**2) / q
        RF[i, j] = -RF[i, j] * C * (Nrmid - i) * dr

def makeRF0(m, q, w, Nr, Nz, Nrmid, dr):
    RF = cp.ones((Nr, Nz), dtype=cp.float64)
    threads_per_block = (16, 16)
    blocks = ((Nr + threads_per_block[0] - 1) // threads_per_block[0],
              (Nz + threads_per_block[1] - 1) // threads_per_block[1])
    make_rf0_kernel[blocks, threads_per_block](m, q, w, Nr, Nz, Nrmid, dr, RF)
    return cp.asnumpy(RF)

@cuda.jit
def min_dists_kernel(vf, vc, result):
    rid2 = 1e6
    rii2 = 1e6
    vid2 = 1e6
    vii2 = 1e6
    Ni = vf.shape[0]
    Nc = vc.shape[0]
    
    for i in range(Ni):
        for j in range(i+1, Ni):
            r = vf[i, 0] - vf[j, 0]
            z = vf[i, 1] - vf[j, 1]
            vr = vf[i, 2] - vf[j, 2]
            vz = vf[i, 3] - vf[j, 3]
            dist2 = r**2 + z**2
            v2 = vr**2 + vz**2
            if dist2 < rii2:
                vii2 = v2
                rii2 = dist2
        for j in range(Nc):
            r = vf[i, 0] - vc[j, 0]
            z = vf[i, 1] - vc[j, 1]
            vr = vf[i, 2] - vc[j, 2]
            vz = vf[i, 3] - vc[j, 3]
            dist2 = r**2 + z**2
            v2 = vr**2 + vz**2
            if dist2 < rid2:
                vid2 = v2
                rid2 = dist2
    
    result[0] = math.sqrt(rid2)
    result[1] = math.sqrt(rii2)
    result[2] = math.sqrt(vid2)
    result[3] = math.sqrt(vii2)

def minDists(vf, vc):
    vf_gpu = cp.asarray(vf)
    vc_gpu = cp.asarray(vc)
    result = cp.zeros(4, dtype=cp.float64)
    min_dists_kernel[1, 1](vf_gpu, vc_gpu, result)
    return cp.asnumpy(result)

def collisionMode(rii, rid, a, e=0.3):
    if rid == 0:
        return True  # Assume collision if rid is zero
    return (a * rii**2) / (rid**5) > e

@cuda.jit(device=True)
def ptov_pos_kernel(pos, Nmid, dcell):
    return (pos / dcell + Nmid)

@cuda.jit
def update_positions_kernel(vf, dr, dz, dt, Nr, Nz, Nrmid, Nzmid):
    i = cuda.grid(1)
    if i < vf.shape[0]:
        vf[i, 0] += vf[i, 2] * dt
        vf[i, 1] += vf[i, 3] * dt
        rCell = ptov_pos_kernel(vf[i, 0], Nrmid, dr)
        zCell = ptov_pos_kernel(vf[i, 1], Nzmid, dz)
        if rCell > Nr - 2 or rCell < 1 or zCell > Nz - 2 or zCell < 1:
            vf[i, 0] = 2.0
            vf[i, 1] = 2.0
            vf[i, 2] = 0.0
            vf[i, 3] = 0.0
            vf[i, 4] = 0.0
            vf[i, 5] = 1e6
            vf[i, 6] = 0.0

def updatePoss(vf, dr, dz, dt, Nr, Nz, Nrmid, Nzmid):
    vf_gpu = cp.asarray(vf)
    threads_per_block = 256
    blocks = (vf_gpu.shape[0] + threads_per_block - 1) // threads_per_block
    update_positions_kernel[blocks, threads_per_block](vf_gpu, dr, dz, dt, Nr, Nz, Nrmid, Nzmid)
    return cp.asnumpy(vf_gpu)

@cuda.jit
def update_velocities_kernel(vf, Erf, Ezf, dt):
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
    update_velocities_kernel[blocks, threads_per_block](vf_gpu, Erf_gpu, Ezf_gpu, dt)
    return cp.asnumpy(vf_gpu)

@cuda.jit
def solve_fields_kernel(vf, ErDC, EzDC, ErAC, EzAC, Nrmid, Nzmid, dr, dz, Erf, Ezf, field_factor):
    i = cuda.grid(1)
    if i < vf.shape[0]:
        eps0 = 8.854e-12
        C1 = 4 * math.pi * eps0
        
        jCell = int(round((vf[i, 0] / dr + Nrmid)))
        kCell = int(round((vf[i, 1] / dz + Nzmid)))
        
        jCell = max(0, min(jCell, ErDC.shape[0] - 1))
        kCell = max(0, min(kCell, ErDC.shape[1] - 1))
        
        Erf[i] = (ErDC[jCell, kCell] + ErAC[jCell, kCell]) * field_factor
        Ezf[i] = (EzDC[jCell, kCell] + EzAC[jCell, kCell]) * field_factor
        
        for j in range(vf.shape[0]):
            if j != i:
                rdist = vf[j, 0] - vf[i, 0]
                zdist = vf[j, 1] - vf[i, 1]
                sqDist = rdist**2 + zdist**2
                if sqDist > 1e-20:
                    invDist = 1.0 / math.sqrt(sqDist)
                    Erf[i] += -rdist * invDist**3 * vf[j, 4] / C1
                    Ezf[i] += -zdist * invDist**3 * vf[j, 4] / C1

def solveFields(vf, ErDC, EzDC, ErAC, EzAC, Nrmid, Nzmid, Ni, dr, dz, field_factor):
    vf_gpu = cp.asarray(vf)
    ErDC_gpu = cp.asarray(ErDC)
    EzDC_gpu = cp.asarray(EzDC)
    ErAC_gpu = cp.asarray(ErAC)
    EzAC_gpu = cp.asarray(EzAC)
    Erf = cp.zeros(Ni)
    Ezf = cp.zeros(Ni)
    
    threads_per_block = 256
    blocks = (Ni + threads_per_block - 1) // threads_per_block
    
    solve_fields_kernel[blocks, threads_per_block](vf_gpu, ErDC_gpu, EzDC_gpu, ErAC_gpu, EzAC_gpu, Nrmid, Nzmid, dr, dz, Erf, Ezf, field_factor)
    
    return cp.asnumpy(Erf), cp.asnumpy(Ezf)

@cuda.jit
def calculate_fields_kernel(vf, vc, ErDC, EzDC, ErAC, EzAC, Nrmid, Nzmid, dr, dz, Erfc, Ezfc):
    i = cuda.grid(1)
    if i < vc.shape[0]:
        eps0 = 8.854e-12
        C1 = 4 * math.pi * eps0
        
        jCell = int(round((vc[i, 0] / dr + Nrmid)))
        kCell = int(round((vc[i, 1] / dz + Nzmid)))
        
        if 0 <= jCell < ErDC.shape[0] and 0 <= kCell < ErDC.shape[1]:
            Erfc[i, 0] = ErDC[jCell, kCell] + ErAC[jCell, kCell]
            Ezfc[i, 0] = EzDC[jCell, kCell] + EzAC[jCell, kCell]
            
            if 0 < jCell < ErDC.shape[0] - 1 and 0 < kCell < ErDC.shape[1] - 1:
                Erfc[i, 1] = ((ErDC[jCell+1, kCell] + ErAC[jCell+1, kCell]) - 
                              (ErDC[jCell-1, kCell] + ErAC[jCell-1, kCell])) / (2*dr)
                Ezfc[i, 1] = ((EzDC[jCell, kCell+1] + EzAC[jCell, kCell+1]) - 
                              (EzDC[jCell, kCell-1] + EzAC[jCell, kCell-1])) / (2*dz)
        
        for j in range(vf.shape[0]):
            rdist = vf[j, 0] - vc[i, 0]
            zdist = vf[j, 1] - vc[i, 1]
            sqDist = rdist**2 + zdist**2
            if sqDist > 1e-20:
                invDist = 1.0 / math.sqrt(sqDist)
                projR = rdist * invDist
                projZ = zdist * invDist
                Erfc[i, 0] += -projR * vf[j, 4] / (C1 * sqDist)
                Ezfc[i, 0] += -projZ * vf[j, 4] / (C1 * sqDist)
                Erfc[i, 1] += 2 * projR * vf[j, 4] / (C1 * sqDist * math.sqrt(sqDist))
                Ezfc[i, 1] += 2 * projZ * vf[j, 4] / (C1 * sqDist * math.sqrt(sqDist))

@cuda.jit
def update_collision_particles_kernel(vc, Erfc, Ezfc, dt):
    i = cuda.grid(1)
    if i < vc.shape[0]:
        if vc[i, 6] != 0.0:
            eps0 = 8.854e-12
            pR = -2 * math.pi * eps0 * vc[i, 6] * Erfc[i, 0]
            pZ = -2 * math.pi * eps0 * vc[i, 6] * Ezfc[i, 0]
            Fr = abs(pR) * Erfc[i, 1]
            Fz = abs(pZ) * Ezfc[i, 1]
            vc[i, 2] += Fr * dt / vc[i, 5]
            vc[i, 3] += Fz * dt / vc[i, 5]

@cuda.jit
def calculate_ion_fields_kernel(vf, vc, Erfc, Ezfc, Erfi, Ezfi):
    i = cuda.grid(1)
    if i < vf.shape[0]:
        eps0 = 8.854e-12
        C1 = 4 * math.pi * eps0
        for j in range(vc.shape[0]):
            if vc[j, 6] != 0.0:
                rdist = vc[j, 0] - vf[i, 0]
                zdist = vc[j, 1] - vf[i, 1]
                sqDist = rdist**2 + zdist**2
                if sqDist > 1e-20:
                    invDist = 1.0 / math.sqrt(sqDist)
                    Rhatr = rdist * invDist
                    Rhatz = zdist * invDist
                    pR = -2 * math.pi * eps0 * vc[j, 6] * Erfc[j, 0]
                    pZ = -2 * math.pi * eps0 * vc[j, 6] * Ezfc[j, 0]
                    Erfi[i] += -abs(pR) * (2*Rhatr) / (C1 * sqDist * invDist)
                    Ezfi[i] += -abs(pZ) * (2*Rhatz) / (C1 * sqDist * invDist)

def collisionParticlesFields(vf, vc, ErDC, EzDC, ErAC, EzAC, dr, dz, dt, Nrmid, Nzmid):
    Ni = vf.shape[0]
    Nc = vc.shape[0]
    
    # Move data to GPU
    vf_gpu = cp.asarray(vf)
    vc_gpu = cp.asarray(vc)
    ErDC_gpu = cp.asarray(ErDC)
    EzDC_gpu = cp.asarray(EzDC)
    ErAC_gpu = cp.asarray(ErAC)
    EzAC_gpu = cp.asarray(EzAC)
    
    Erfc = cp.zeros((Nc, 2), dtype=cp.float64)
    Ezfc = cp.zeros((Nc, 2), dtype=cp.float64)
    Erfi = cp.zeros(Ni, dtype=cp.float64)
    Ezfi = cp.zeros(Ni, dtype=cp.float64)
    
    # Calculate fields on collision particles
    threads_per_block = 256
    blocks = (Nc + threads_per_block - 1) // threads_per_block
    calculate_fields_kernel[blocks, threads_per_block](vf_gpu, vc_gpu, ErDC_gpu, EzDC_gpu, ErAC_gpu, EzAC_gpu, Nrmid, Nzmid, dr, dz, Erfc, Ezfc)
    
    # Update collision particles
    update_collision_particles_kernel[blocks, threads_per_block](vc_gpu, Erfc, Ezfc, dt)
    
    # Calculate fields on ions from collision particles
    blocks = (Ni + threads_per_block - 1) // threads_per_block
    calculate_ion_fields_kernel[blocks, threads_per_block](vf_gpu, vc_gpu, Erfc, Ezfc, Erfi, Ezfi)
    
    # Move results back to CPU
    vc = cp.asnumpy(vc_gpu)
    Erfi = cp.asnumpy(Erfi)
    Ezfi = cp.asnumpy(Ezfi)
    Erfc = cp.asnumpy(Erfc)
    Ezfc = cp.asnumpy(Ezfc)
    
    return vc, Erfi, Ezfi, Erfc, Ezfc, None, None

def mcCollision(vf, rc, zc, vrc, vzc, qc, mc, ac, Nt, dtSmall, RF, DC, Nr, Nz, dr, dz, dtLarge, dtCollision, eii=0.01, eid=0.01):
    Nrmid = (Nr - 1) / 2
    Nzmid = (Nz - 1) / 2
    Ni = vf.shape[0]
    Nc = 1
    vc = np.zeros((Nc, 7))
    vc[0, :] = [rc, zc, vrc, vzc, qc, mc, ac]
    
    nullFields = np.zeros((Nr, Nz))
    
    print("Initial vf:")
    print(vf)
    
    for i in range(Nt):
        field_factor = min(1.0, i / 10000) * 0.1  # Reach full strength after 10000 iterations, but only 10% of original strength
        
        rid, rii, vid, vii = minDists(vf, vc)
        collision = collisionMode(rii, rid, vc[0, 6], 0.1)
        
        if collision:
            dtNow = rid * eid / (5 * vid) if vid > 1e-20 else dtSmall
        else:
            dtNow = dtSmall
        
        dtNow = max(dtNow, dtCollision)
        
        Erfi, Ezfi = solveFields(vf, nullFields, DC, RF, nullFields, Nrmid, Nzmid, Ni, dr, dz, field_factor)
        
        if i % 1000 == 0:
            print(f"Iteration {i}:")
            print(f"Electric fields: Erfi = {Erfi}, Ezfi = {Ezfi}")
            print(f"Field factor: {field_factor}")
            print(f"Ion positions: {vf[:, :2]}")
            print(f"Ion velocities: {vf[:, 2:4]}")
            
            forces = vf[:, 4][:, np.newaxis] * np.column_stack((Erfi, Ezfi))
            accelerations = forces / vf[:, 5][:, np.newaxis]
            print(f"Forces: {forces}")
            print(f"Accelerations: {accelerations}")
        
        if vc[0, 5] < 1e6:
            vc, Erfic, Ezfic, Erfc, Ezfc, _, _ = collisionParticlesFields(vf, vc, nullFields, DC, RF, nullFields, dr, dz, dtNow, Nrmid, Nzmid)
            vc = updatePoss(vc, dr, dz, dtNow, Nr, Nz, Nrmid, Nzmid)
            Erfi += Erfic
            Ezfi += Ezfic
        else:
            dtNow = dtLarge
        
        vf = updateVels(vf, Erfi, Ezfi, dtNow)
        vf = updatePoss(vf, dr, dz, dtNow, Nr, Nz, Nrmid, Nzmid)
        
        if np.any(np.isnan(vf)):
            print(f"NaN detected at iteration {i}")
            print("vf at NaN detection:")
            print(vf)
            break
        
        if i % 1000 == 0:
            print(f"Iteration {i}: Max ion position: {np.max(np.abs(vf[:, :2]))}, Max velocity: {np.max(np.abs(vf[:, 2:4]))}")
    
    return vf  # Return the final state of the ions

# Main execution block
if __name__ == "__main__":
    N = 3
    nu = 1e6
    m = 6.6335209e-26
    q = 1.60217663e-19
    Nr, Nz = 100, 100
    dr, dz = 1e-6, 1e-6
    dtSmall = 1e-10  # Time step
    dtLarge = 1e-9
    dtCollision = 1e-11
    Nt = 13000

    try:
        positions = calcPositions(N)
        vf = np.zeros((N, 7))
        vf[:, 0] = positions
        vf[:, 4] = q
        vf[:, 5] = m

        w = 2 * np.pi * nu
        RF = makeRF0(m, q, w, Nr, Nz, (Nr-1)/2, dr)
        RF *= 0.1
        DC = np.zeros((Nr, Nz))
        DC[:, :] = np.linspace(-1e-3, 1e-3, Nr)[:, np.newaxis]  # Add a small DC field in z-direction

        rc, zc, vrc, vzc = 0, 0, 0, 0
        qc, mc, ac = 0, 1e-26, 1e-40

        print("Initial ion positions:")
        print(vf[:, :2])

        final_state = mcCollision(vf, rc, zc, vrc, vzc, qc, mc, ac, Nt, dtSmall, RF, DC, Nr, Nz, dr, dz, dtLarge, dtCollision)
    
        print("Simulation completed.")
        print("Final ion positions:")
        print(final_state[:, :2])
        print("Final ion velocities:")
        print(final_state[:, 2:4])
        print(f"Simulation result: {0 if np.all(np.abs(final_state[:, :2]) <= 2.0) else 1}")

    except Exception as e:
        print(f"An error occurred during the simulation: {e}")
        import traceback
        traceback.print_exc()