import numpy as np
from numba import cuda, float64, int32
import math

# Constants and parameters (same as in original code)
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

# Test data
vf = np.array([
    [0.0, 4.79258560e-06, 0.0, 0.0, q, m, 0.0],
    [0.0, -8.03509216e-20, 0.0, 0.0, q, m, 0.0],
    [0.0, -4.79258560e-06, 0.0, 0.0, q, m, 0.0]
], dtype=np.float64)

vc = np.array([
    [-4.759190024710139e-07, 1.978544579898285e-05, 75.75163221513105, -2386.3355135788324, q, mH2, aH2]
], dtype=np.float64)

# CPU version
def minDists_cpu(vf,vc):
    rid2 = 1e6 ; rii2 = 1e6 ; vid2 = 1e6 ; vii2 = 1e6
    Ni = len(vf[:,0]) ; Nc = len(vc[:,0])
    for i in range(Ni):
        for j in range(i+1,Ni): # check each pair of ions for distance and speed
            r = vf[i,0]-vf[j,0] ; z=vf[i,1]-vf[j,1] ; vr = vf[i,2]-vf[j,2] ; vz = vf[i,3]-vf[j,3]
            dist2 = r**2 + z**2
            v2 = vr**2 + vz**2
            if dist2<rii2:
                vii2 = v2 ; rii2 = dist2                
        for j in range(Nc): # check each ion-dipole pair for distance and speed
            r = vf[i,0]-vc[j,0] ; z=vf[i,1]-vc[j,1] ; vr = vf[i,2]-vc[j,2] ; vz = vf[i,3]-vc[j,3]
            dist2 = r**2 + z**2
            v2 = vr**2 + vz**2
            if dist2<rid2:
                vid2 = v2 ; rid2 = dist2                
    return np.sqrt(rid2),np.sqrt(rii2),np.sqrt(vid2),np.sqrt(vii2)

# GPU version
@cuda.jit(device=True)
def minDists_gpu(vf, vc):
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

# Wrapper kernel for GPU function
@cuda.jit
def gpu_wrapper(vf, vc, results):
    rid, rii, vid, vii = minDists_gpu(vf, vc)
    results[0] = rid
    results[1] = rii
    results[2] = vid
    results[3] = vii

# Run CPU version
cpu_results = minDists_cpu(vf, vc)

# Run GPU version
gpu_results_array = np.zeros(4, dtype=np.float64)
gpu_wrapper[1, 1](vf, vc, gpu_results_array)

# Print and compare results
print("CPU Results:")
print(f"rid (ion-dipole distance): {cpu_results[0]:.6e}")
print(f"rii (ion-ion distance): {cpu_results[1]:.6e}")
print(f"vid (ion-dipole velocity): {cpu_results[2]:.6e}")
print(f"vii (ion-ion velocity): {cpu_results[3]:.6e}")

print("\nGPU Results:")
print(f"rid (ion-dipole distance): {gpu_results_array[0]:.6e}")
print(f"rii (ion-ion distance): {gpu_results_array[1]:.6e}")
print(f"vid (ion-dipole velocity): {gpu_results_array[2]:.6e}")
print(f"vii (ion-ion velocity): {gpu_results_array[3]:.6e}")

# # Calculate and print differences
# print("\nAbsolute Differences (CPU - GPU):")
# print(f"rid difference: {abs(cpu_results[0] - gpu_results_array[0]):.6e}")
# print(f"rii difference: {abs(cpu_results[1] - gpu_results_array[1]):.6e}")
# print(f"vid difference: {abs(cpu_results[2] - gpu_results_array[2]):.6e}")
# print(f"vii difference: {abs(cpu_results[3] - gpu_results_array[3]):.6e}")