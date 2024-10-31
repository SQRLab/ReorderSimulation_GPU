import cupy as cp
import numpy as np
import numba
from numba import cuda, float32, int32
import math
from scipy.optimize import fsolve
from math import exp, pi as π
import scipy.constants as con
import os
import time
from scipy import special

# Check CUDA availability
if not cp.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please check your installation.")

# Initialize CUDA device
cuda_device = cp.cuda.Device(0)  # Use the first CUDA device
cuda_device.use()

THREADS_PER_BLOCK = 256

# Helper functions
@cuda.jit(device=True)
def ptovPos(pos, Nmid, dcell):
    return (pos / dcell + float(Nmid))

@cuda.jit(device=True)
def minDists(vf, vc):
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
    return math.sqrt(rid2), math.sqrt(rii2), math.sqrt(vid2), math.sqrt(vii2)

@cuda.jit(device=True)
def collisionMode(rii, rid, a, e=0.3):
    return (a * rii**2) / (rid**5) > e

@cuda.jit(device=True)
def solveFields(vf, RF, DC, Nrmid, Nzmid, dr, dz, Erf, Ezf):
    Ni = vf.shape[0]
    for i in range(Ni):
        jCell = int(round(ptovPos(vf[i, 0], Nrmid, dr)))
        kCell = int(round(ptovPos(vf[i, 1], Nzmid, dz)))
        Erf[i] = DC[jCell, kCell] + RF[jCell, kCell]
        Ezf[i] = DC[jCell, kCell]
        for j in range(Ni):
            if j != i:
                rdist = vf[j, 0] - vf[i, 0]
                zdist = vf[j, 1] - vf[i, 1]
                sqDist = rdist**2 + zdist**2
                projR = rdist / math.sqrt(sqDist)
                projZ = zdist / math.sqrt(sqDist)
                E = vf[j, 4] / (4 * math.pi * 8.854e-12 * sqDist)
                Erf[i] -= projR * E
                Ezf[i] -= projZ * E

@cuda.jit(device=True)
def updateVels(vf, Erf, Ezf, dt):
    for i in range(vf.shape[0]):
        Fr = vf[i, 4] * Erf[i]
        Fz = vf[i, 4] * Ezf[i]
        vf[i, 2] += Fr * dt / vf[i, 5]
        vf[i, 3] += Fz * dt / vf[i, 5]

@cuda.jit(device=True)
def updatePoss(vf, dr, dz, dt, Nr, Nz, Nrmid, Nzmid):
    for i in range(vf.shape[0]):
        vf[i, 0] += vf[i, 2] * dt
        vf[i, 1] += vf[i, 3] * dt
        rCell = ptovPos(vf[i, 0], Nrmid, dr)
        zCell = ptovPos(vf[i, 1], Nzmid, dz)
        if rCell > Nr - 2 or rCell < 1 or zCell > Nz - 2 or zCell < 1:
            vf[i, :] = 0.0
            vf[i, 0] = 2.0
            vf[i, 1] = 2.0
            vf[i, 5] = 1e6

@cuda.jit
def makeRF0_kernel(m, q, w, Nr, Nz, Nrmid, dr, RF):
    j, k = cuda.grid(2)
    if j < Nr and k < Nz:
        C = -m * (w**2) / q
        RF[j, k] = -C * (Nrmid - j) * dr

@cuda.jit
def makeDC_kernel(m, q, w, Nz, Nr, Nzmid, dz, DC):
    j, k = cuda.grid(2)
    if j < Nr and k < Nz:
        C = -m * (w**2) / q
        DC[j, k] = -C * (Nzmid - k) * dz

def makeRF0_cuda(m, q, w, Nr, Nz, Nrmid, dr):
    RF = cp.zeros((Nr, Nz), dtype=cp.float32)
    threads_per_block = (16, 16)
    blocks_per_grid = ((Nr + threads_per_block[0] - 1) // threads_per_block[0],
                       (Nz + threads_per_block[1] - 1) // threads_per_block[1])
    makeRF0_kernel[blocks_per_grid, threads_per_block](m, q, w, Nr, Nz, Nrmid, dr, RF)
    return RF

def makeDC_cuda(m, q, w, Nz, Nr, Nzmid, dz):
    DC = cp.zeros((Nr, Nz), dtype=cp.float32)
    threads_per_block = (16, 16)
    blocks_per_grid = ((Nr + threads_per_block[0] - 1) // threads_per_block[0],
                       (Nz + threads_per_block[1] - 1) // threads_per_block[1])
    makeDC_kernel[blocks_per_grid, threads_per_block](m, q, w, Nz, Nr, Nzmid, dz, DC)
    return DC

@cuda.jit
def makeVf_kernel(Ni, q, m, l, wr, offsetr, offsetz, vbumpr, vbumpz, pos, lscale, vf):
    i = cuda.grid(1)
    if i < Ni:
        vf[i, 0] = 0.0e-6
        vf[i, 1] = -pos[i] * lscale
        vf[i, 2] = 0.0
        vf[i, 3] = 0.0
        vf[i, 4] = q
        vf[i, 5] = m
        vf[i, 6] = 0.0
    if i == l:
        vf[i, 0] += offsetr
        vf[i, 1] += offsetz
        vf[i, 2] += vbumpr
        vf[i, 3] += vbumpz

def makeVf(Ni, q, m, l, wr, offsetr, offsetz, vbumpr, vbumpz):
    pos = calcPositions(Ni)
    lscale = lengthScale(wr)
    pos_gpu = cp.asarray(pos)
    vf_gpu = cp.zeros((Ni, 7), dtype=cp.float32)
    threads_per_block = 256
    blocks_per_grid = (Ni + (threads_per_block - 1)) // threads_per_block
    makeVf_kernel[blocks_per_grid, threads_per_block](
        Ni, q, m, l, wr, offsetr, offsetz, vbumpr, vbumpz, pos_gpu, lscale, vf_gpu
    )
    return vf_gpu

@numba.njit
def ion_position_potential(x):
    N = len(x)
    return [x[m] - sum([1/(x[m]-x[n])**2 for n in range(m)]) + sum([1/(x[m]-x[n])**2 for n in range(m+1,N)])
               for m in range(N)]

def calcPositions(N):
    estimated_extreme = 0.481*N**0.765
    return fsolve(ion_position_potential, np.linspace(-estimated_extreme, estimated_extreme, N))

def lengthScale(ν, M=None, Z=None):
    if M==None: M = con.atomic_mass*39.9626
    if Z==None: Z = 1
    return ((Z**2*con.elementary_charge**2)/(4*π*con.epsilon_0*M*ν**2))**(1/3)

@cuda.jit(device=True)
def mcCollision_cuda(vf, rc, zc, vrc, vzc, qc, mc, ac, Nt, dtSmall, RF, DC, Nr, Nz, dr, dz, dtLarge, dtCollision, Nrmid, Nzmid):
    Ni = vf.shape[0]
    vc = cuda.local.array((1, 7), dtype=float32)
    vc[0] = [rc, zc, vrc, vzc, qc, mc, ac]
    
    Erfi = cuda.local.array(Ni, dtype=float32)
    Ezfi = cuda.local.array(Ni, dtype=float32)
    
    reorder = 0
    
    for i in range(Nt):
        rid, rii, vid, vii = minDists(vf, vc)
        collision = collisionMode(rii, rid, vc[0, 6], 0.1)
        
        if collision:
            dtNow = max(rid * 0.01 / (5 * vid), dtCollision)
        else:
            dtNow = dtSmall
        
        solveFields(vf, RF, DC, Nrmid, Nzmid, dr, dz, Erfi, Ezfi)
        
        if vc[0, 5] < 1e6:
            # Simplified collisionParticlesFields logic
            updatePoss(vc, dr, dz, dtNow, Nr, Nz, Nrmid, Nzmid)
        else:
            dtNow = dtLarge
        
        updateVels(vf, Erfi, Ezfi, dtNow)
        updatePoss(vf, dr, dz, dtNow, Nr, Nz, Nrmid, Nzmid)
        
        if cuda.any(vf[:, 5] > 1e5):
            reorder = 2
            break
        
        for j in range(1, Ni):
            if vf[j, 1] > vf[j-1, 1]:
                reorder = 1
                break
        
        if reorder != 0:
            break
    
    return reorder

@cuda.jit
def parallel_shots_kernel(vf, RF, DC, Nr, Nz, dr, dz, dtSmall, dtLarge, dtCollision, Nt, results, v, boltzDist, angles, offsets, max_hypotenuse, Nrmid, Nzmid, rng_states):
    shot_index = cuda.grid(1)
    if shot_index < results.shape[0]:
        # Use cuRAND for random number generation
        thread_id = cuda.grid(1)
        
        # Generate random velocity
        rand = cuda.random.xoroshiro128p_uniform_float32(rng_states, thread_id)
        velocity_index = int(rand * len(v))
        velocity = v[velocity_index]
        
        # Generate random angle
        rand = cuda.random.xoroshiro128p_uniform_float32(rng_states, thread_id)
        angle_choice = rand * (angles[-1] - angles[0]) + angles[0]
        
        # Generate random offset
        rand = cuda.random.xoroshiro128p_uniform_float32(rng_states, thread_id)
        offset_choice = rand * (offsets[-1] - offsets[0]) + offsets[0]
        
        # Generate random ion collision
        rand = cuda.random.xoroshiro128p_uniform_float32(rng_states, thread_id)
        ion_collided = int(rand * vf.shape[0])
        
        r = -math.cos(angle_choice) * max_hypotenuse
        z = vf[ion_collided, 1] + math.sin(angle_choice) * max_hypotenuse + offset_choice
        vz = -1 * velocity * math.sin(angle_choice)
        vr = abs(velocity * math.cos(angle_choice))
        
        if velocity < 200:
            Nt_actual = 700000
        elif velocity < 1500:
            Nt_actual = 400000
        else:
            Nt_actual = 250000
        
        vf_local = vf.copy()
        
        reorder = mcCollision_cuda(vf_local, r, z, vr, vz, vf[0, 4], 2.0 * 1.67e-27, 8e-31, 
                                   Nt_actual, dtSmall, RF, DC, Nr, Nz, dr, dz, dtLarge, dtCollision, Nrmid, Nzmid)
        
        results[shot_index, 0] = velocity
        results[shot_index, 1] = ion_collided + 1
        results[shot_index, 2] = angle_choice
        results[shot_index, 3] = offset_choice
        results[shot_index, 4] = reorder

def run_parallel_shots_cuda(shots, vf, RF, DC, Nr, Nz, dr, dz, dtSmall, dtLarge, dtCollision, Nt, v, boltzDist, angles, offsets, max_hypotenuse):
    vf_gpu = cp.asarray(vf)
    RF_gpu = cp.asarray(RF)
    DC_gpu = cp.asarray(DC)
    v_gpu = cp.asarray(v)
    boltzDist_gpu = cp.asarray(boltzDist)
    angles_gpu = cp.asarray(angles)
    offsets_gpu = cp.asarray(offsets)
    
    results_gpu = cp.zeros((shots, 5), dtype=cp.float32)
    
    threads_per_block = 256
    blocks_per_grid = (shots + (threads_per_block - 1)) // threads_per_block
    
    # Create a random number generator states
    rng_states = cuda.random.create_xoroshiro128p_states(threads_per_block * blocks_per_grid, seed=1)
    
    Nrmid = (Nr - 1) / 2
    Nzmid = (Nz - 1) / 2
    parallel_shots_kernel[blocks_per_grid, threads_per_block](
        vf_gpu, RF_gpu, DC_gpu, Nr, Nz, dr, dz, dtSmall, dtLarge, dtCollision, Nt, 
        results_gpu, v_gpu, boltzDist_gpu, angles_gpu, offsets_gpu, max_hypotenuse, Nrmid, Nzmid, rng_states
    )
    
    results = results_gpu.get()
    
    return results

def run_cuda_simulation(shots, Ni, Nr, Nz, dr, dz, dtSmall, dtLarge, dtCollision, m, q, wr, wz, offsetr, offsetz, vbumpr, vbumpz, v, boltzDist, angles, offsets, max_hypotenuse):
    Nrmid = (Nr - 1) / 2
    Nzmid = (Nz - 1) / 2
    RF = makeRF0_cuda(m, q, wr, Nr, Nz, Nrmid, dr)
    DC = makeDC_cuda(m, q, wz, Nz, Nr, Nzmid, dz)
    vf = makeVf(Ni, q, m, 1, wr, offsetr, offsetz, vbumpr, vbumpz)
    
    results = run_parallel_shots_cuda(shots, vf, RF, DC, Nr, Nz, dr, dz, dtSmall, dtLarge, dtCollision, 700000, v, boltzDist, angles, offsets, max_hypotenuse)
    
    with open(f"simulation_results/{Ni}ionSimulation_{Nr}_{shots}shots.txt", "w") as f:
        f.write("axial trapping frequency (MHz) \t velocity(m/s) \t ion collided with \t angle(rads) \t collision offset(m) \t reorder? (1 is reorder 2 is ejection) \n")
        for result in results:
            f.write(f"{wz/(2*np.pi*1e6)}\t{result[0]}\t{int(result[1])}\t{result[2]}\t{result[3]}\t{int(result[4])}\n")
    
    print(f"Completed simulation for ion count {Ni}, grid size {Nr}, and shots {shots}.")

def Boltz(m, T, vmin=0, vmax=5000, bins=100):
    m = m * amu
    k = 1.386e-23
    boltz = np.zeros(bins)
    dv = (vmax - vmin) / bins
    a = (k * T / m) ** (1/2)

    for i in range(bins):
        vhere = vmin + i * dv
        vlast = vhere - dv
        boltz[i] = (special.erf(vhere / (a * np.sqrt(2))) - np.sqrt(2/np.pi) * (vhere/a) * np.exp(-vhere**2 / (2*a**2))) - \
                   (special.erf(vlast / (a * np.sqrt(2))) - np.sqrt(2/np.pi) * (vlast/a) * np.exp(-vlast**2 / (2*a**2)))

    return boltz / np.sum(boltz)

# Define constants
amu = 1.67e-27
eps0 = 8.854e-12
qe = 1.6e-19

# Define physical params
m = 40. * amu
q = 1. * qe
wr = 2 * np.pi * 3e6
wz = 2 * np.pi * 1e6

# Define sim params
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

# Simulation parameters
T = 300
collisionalMass = 2
vMin = 50
vMax = 7000
numBins = 1000

# Generate Boltzmann distribution
boltzDist = Boltz(collisionalMass, T, vMin, vMax, numBins)
v = np.linspace(vMin, vMax, numBins)
angles = np.linspace(-np.pi/2, np.pi/2, 100)
offsets = np.linspace(-2e-9, 2e-9, 200)
max_hypotenuse = 1.5e-5

# Simulation parameters
grid_sizes = [10001]
ion_counts = [3]
shot_sizes = [1000]

# Ensure the results directory exists
os.makedirs("simulation_results", exist_ok=True)

# Main simulation loop
for grid_size in grid_sizes:
    Nr = Nz = grid_size
    dr = Dr / float(Nr)
    dz = Dz / float(Nz)

    for Ni in ion_counts:
        for shots in shot_sizes:
            print(f"Starting simulation with grid size {grid_size}, ion count {Ni}, and shots {shots}")
            
            start_time = time.perf_counter()
            
            # Run the CUDA simulation
            run_cuda_simulation(shots, Ni, Nr, Nz, dr, dz, dtSmall, dtLarge, dtCollision, 
                                m, q, wr, wz, offsetr, offsetz, vbumpr, vbumpz, 
                                v, boltzDist, angles, offsets, max_hypotenuse)
            
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            
            print(f"Simulation completed in {elapsed_time:.2f} seconds")

print("All simulations completed successfully!")