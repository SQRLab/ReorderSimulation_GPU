'''import numpy as np
import cupy as cp
from numba import cuda
import math
import ctypes

print("CUDA available:", cp.cuda.is_available())
print("CUDA version:", cp.cuda.runtime.runtimeGetVersion())
print("Number of CUDA devices:", cp.cuda.runtime.getDeviceCount())


# Try a simple CUDA operation
try:
    a = cp.array([1, 2, 3])
    b = cp.array([4, 5, 6])
    c = a + b
    print("Simple CUDA operation successful:", c)
except Exception as e:
    print("Error in simple CUDA operation:", e)

try:
    cp.random.seed(42)  # Use any integer seed
    print("CuPy random state initialized successfully")
except Exception as e:
    print(f"Error initializing CuPy random state: {e}")

# Memory info
mem_info = cp.cuda.runtime.memGetInfo()
print(f"Free memory: {mem_info[0] / 1024**3:.2f} GB")
print(f"Total memory: {mem_info[1] / 1024**3:.2f} GB")


# Load the CUDA C library
cuda_lib = ctypes.CDLL('./simple_collision.dll')

# Define function prototypes
cuda_lib.length_scale_kernel.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
cuda_lib.ptov_pos_kernel.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_double, ctypes.POINTER(ctypes.c_double)]
cuda_lib.ion_position_potential_kernel.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int]
cuda_lib.calc_positions_kernel.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_double, ctypes.c_int]
cuda_lib.min_dists_kernel.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]

collision_kernel_proto = ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(ctypes.c_double),  # vf
    ctypes.POINTER(ctypes.c_double),  # vc
    ctypes.c_int,  # Ni
    ctypes.c_int,  # Nc
    ctypes.POINTER(ctypes.c_double),  # ErDC
    ctypes.POINTER(ctypes.c_double),  # EzDC
    ctypes.POINTER(ctypes.c_double),  # ErAC
    ctypes.POINTER(ctypes.c_double),  # EzAC
    ctypes.c_double,  # dr
    ctypes.c_double,  # dz
    ctypes.c_double,  # dt
    ctypes.c_int,  # Nrmid
    ctypes.c_int,  # Nzmid
    ctypes.POINTER(ctypes.c_double),  # Erfi
    ctypes.POINTER(ctypes.c_double),  # Ezfi
    ctypes.POINTER(ctypes.c_double),  # Erfc
    ctypes.POINTER(ctypes.c_double),  # Ezfc
    ctypes.POINTER(ctypes.c_double),  # min_dists
    ctypes.POINTER(ctypes.c_int)      # collision_mode
)

collision_kernel = collision_kernel_proto(("collision_kernel", cuda_lib))

def lengthScale(nu, M=None, Z=None):
    if M is None:
        M = 39.9626 * 1.66053906660e-27  # atomic mass * mass of Ar-40
    if Z is None:
        Z = 1
    result = ctypes.c_double()
    print(f"Debug: Calling length_scale_kernel with nu={nu}, M={M}, Z={Z}")
    cuda_lib.length_scale_kernel(ctypes.c_double(nu), ctypes.c_double(M), ctypes.c_int(Z), ctypes.byref(result))
    print(f"Debug: length_scale_kernel returned {result.value}")
    return result.value

def ptovPos(pos, Nmid, dcell):
    result = cp.zeros_like(pos)
    pos_ptr = pos.data.ptr
    result_ptr = result.data.ptr
    cuda_lib.ptov_pos_kernel(ctypes.cast(pos_ptr, ctypes.POINTER(ctypes.c_double)),
                             ctypes.c_int(Nmid), ctypes.c_double(dcell),
                             ctypes.cast(result_ptr, ctypes.POINTER(ctypes.c_double)))
    return result

def ion_position_potential(x):
    result = cp.zeros_like(x)
    cuda_lib.ion_position_potential_kernel(
        ctypes.cast(x.data.ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(result.data.ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(x.size)
    )
    return result

def calcPositions(N):
    estimated_extreme = 0.481 * N**0.765
    x = cp.linspace(-estimated_extreme, estimated_extreme, N)
    cuda_lib.calc_positions_kernel(
        ctypes.cast(x.data.ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(N),
        ctypes.c_double(estimated_extreme),
        ctypes.c_int(1000)  # Number of iterations
    )
    return x

def minDists(vf, vc):
    Ni = vf.shape[0]
    Nc = vc.shape[0]
    result = cp.zeros(4, dtype=cp.float64)
    cuda_lib.min_dists_kernel(
        ctypes.cast(vf.data.ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(vc.data.ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(Ni),
        ctypes.c_int(Nc),
        ctypes.cast(result.data.ptr, ctypes.POINTER(ctypes.c_double))
    )
    return result[0], result[1], result[2], result[3]

@cuda.jit
def collision_mode_kernel(rii, rid, a, e, result):
    i = cuda.grid(1)
    if i == 0:
        result[0] = (a * rii * rii) / (rid * rid * rid * rid * rid) > e

def collisionMode(rii, rid, a, e=0.3):
    result = cp.zeros(1, dtype=cp.bool_)
    collision_mode_kernel[1, 1](rii, rid, a, e, result)
    return result[0].item()  # Convert to Python bool

@cuda.jit
def make_rf0_kernel(m, q, w, Nr, Nz, Nrmid, dr, RF):
    i, j = cuda.grid(2)
    if i < Nr and j < Nz:
        C = -m * (w**2) / q
        RF[i, j] = -RF[i, j] * C * (Nrmid - i) * dr

def makeRF0(m, q, w, Nr, Nz, Nrmid, dr):
    RF = cp.ones((Nr, Nz), dtype=cp.float64)
    threads_per_block = (32, 32)
    blocks = ((Nr + threads_per_block[0] - 1) // threads_per_block[0], 
              (Nz + threads_per_block[1] - 1) // threads_per_block[1])
    make_rf0_kernel[blocks, threads_per_block](m, q, w, Nr, Nz, Nrmid, dr, RF)
    return RF

def collisionParticlesFields(vf, vc, ErDC, EzDC, ErAC, EzAC, dr, dz, dt, Nrmid, Nzmid):
    Ni, Nc = vf.shape[0], vc.shape[0]
    print(f"Debug: Entering collisionParticlesFields, Ni: {Ni}, Nc: {Nc}")
    
    Erfi = cp.zeros(Ni, dtype=cp.float64)
    Ezfi = cp.zeros(Ni, dtype=cp.float64)
    Erfc = cp.zeros((Nc, 2), dtype=cp.float64)
    Ezfc = cp.zeros((Nc, 2), dtype=cp.float64)
    min_dists = cp.zeros(4, dtype=cp.float64)
    collision_mode = cp.zeros(1, dtype=cp.int32)
    
    print("Debug: About to call CUDA kernel")
    try:
        collision_kernel(
            ctypes.cast(vf.data.ptr, ctypes.POINTER(ctypes.c_double)),
            ctypes.cast(vc.data.ptr, ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(Ni),
            ctypes.c_int(Nc),
            ctypes.cast(ErDC.data.ptr, ctypes.POINTER(ctypes.c_double)),
            ctypes.cast(EzDC.data.ptr, ctypes.POINTER(ctypes.c_double)),
            ctypes.cast(ErAC.data.ptr, ctypes.POINTER(ctypes.c_double)),
            ctypes.cast(EzAC.data.ptr, ctypes.POINTER(ctypes.c_double)),
            ctypes.c_double(dr),
            ctypes.c_double(dz),
            ctypes.c_double(dt),
            ctypes.c_int(Nrmid),
            ctypes.c_int(Nzmid),
            ctypes.cast(Erfi.data.ptr, ctypes.POINTER(ctypes.c_double)),
            ctypes.cast(Ezfi.data.ptr, ctypes.POINTER(ctypes.c_double)),
            ctypes.cast(Erfc.data.ptr, ctypes.POINTER(ctypes.c_double)),
            ctypes.cast(Ezfc.data.ptr, ctypes.POINTER(ctypes.c_double)),
            ctypes.cast(min_dists.data.ptr, ctypes.POINTER(ctypes.c_double)),
            ctypes.cast(collision_mode.data.ptr, ctypes.POINTER(ctypes.c_int))
        )
    except Exception as e:
        print(f"Debug: Exception during CUDA kernel call: {e}")
        raise
    print("Debug: CUDA kernel call completed")
    
    return vc, Erfi, Ezfi, Erfc, Ezfc, min_dists, collision_mode[0]

def setup_simulation(N, nu, m, q, Nr, Nz):
    Nrmid = (Nr - 1) / 2
    Nzmid = (Nz - 1) / 2
    
    # Calculate initial positions
    positions = calcPositions(N)
    
    # Create vf array
    vf = cp.zeros((N, 7), dtype=cp.float64)
    vf[:, 0] = positions
    vf[:, 4] = q
    vf[:, 5] = m
    
    # Create RF field
    w = 2 * math.pi * nu
    RF = makeRF0(m, q, w, Nr, Nz, Nrmid, 1e-6)  # Assuming dr = 1e-6 for now
    
    return vf, RF

def test_part1_and_2():
    try:
        print("Testing lengthScale:")
        result = lengthScale(1e6)
        print(result)
    except Exception as e:
        print(f"Error in lengthScale: {e}")

    try:
        print("\nTesting ptovPos:")
        pos = cp.array([1e-6, 2e-6, 3e-6])
        result = ptovPos(pos, 50, 1e-6)
        print(result)
    except Exception as e:
        print(f"Error in ptovPos: {e}")

    try:
        print("\nTesting makeRF0:")
        m, q, nu = 6.6335209e-26, 1.60217663e-19, 1e6
        Nr, Nz = 100, 100
        RF = makeRF0(m, q, 2*math.pi*nu, Nr, Nz, (Nr-1)/2, 1e-6)
        print(RF[:5, :5])
    except Exception as e:
        print(f"Error in makeRF0: {e}")

    try:
        print("\nTesting setup_simulation:")
        N = 10
        vf, RF = setup_simulation(N, nu, m, q, Nr, Nz)
        print("vf shape:", vf.shape)
        print("RF shape:", RF.shape)
    except Exception as e:
        print(f"Error in setup_simulation: {e}")

    print("\nTesting collisionParticlesFields:")
    Ni, Nc = 10, 5
    vf = cp.random.rand(Ni, 7)
    vc = cp.random.rand(Nc, 7)
    ErDC = cp.random.rand(Nr, Nz)
    EzDC = cp.random.rand(Nr, Nz)
    ErAC = cp.random.rand(Nr, Nz)
    EzAC = cp.random.rand(Nr, Nz)
    dr, dz, dt = 1e-6, 1e-6, 1e-9
    Nrmid, Nzmid = Nr // 2, Nz // 2
    
    try:
        vc, Erfi, Ezfi, Erfc, Ezfc, min_dists, collision_mode = collisionParticlesFields(vf, vc, ErDC, EzDC, ErAC, EzAC, dr, dz, dt, Nrmid, Nzmid)
        print("Output shapes:")
        print(f"vc: {vc.shape}")
        print(f"Erfi: {Erfi.shape}")
        print(f"Ezfi: {Ezfi.shape}")
        print(f"Erfc: {Erfc.shape}")
        print(f"Ezfc: {Ezfc.shape}")
        print("Output sums:")
        print(f"vc sum: {vc.sum()}")
        print(f"Erfi sum: {Erfi.sum()}")
        print(f"Ezfi sum: {Ezfi.sum()}")
        print(f"Erfc sum: {Erfc.sum()}")
        print(f"Ezfc sum: {Ezfc.sum()}")
        print(f"Minimum distances: {min_dists}")
        print(f"Collision mode: {collision_mode}")
    except Exception as e:
        print(f"Error in collisionParticlesFields: {e}")

if __name__ == "__main__":
    test_part1_and_2()'''

import numpy as np
import cupy as cp
from numba import cuda
import ctypes
import math

# Load the CUDA C library
cuda_lib = ctypes.CDLL('./simple_collision.dll')

# Define function prototypes
cuda_lib.length_scale_kernel.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
cuda_lib.ptov_pos_kernel.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_double, ctypes.POINTER(ctypes.c_double)]
cuda_lib.calc_positions_kernel.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_double, ctypes.c_int]
cuda_lib.min_dists_kernel.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
cuda_lib.make_rf0_kernel.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.POINTER(ctypes.c_double)]


def calcPositions(N):
    estimated_extreme = 0.481 * N**0.765
    x = cp.linspace(-estimated_extreme, estimated_extreme, N)
    z = cp.linspace(-estimated_extreme/2, estimated_extreme/2, N)
    return cp.column_stack((x, z))

def lengthScale(nu, M=None, Z=None):
    if M is None:
        M = 39.9626 * 1.66053906660e-27  # atomic mass * mass of Ar-40
    if Z is None:
        Z = 1
    result = ctypes.c_double()
    cuda_lib.length_scale_kernel(ctypes.c_double(nu), ctypes.c_double(M), ctypes.c_int(Z), ctypes.byref(result))
    return result.value

def ptovPos(pos, Nmid, dcell):
    result = cp.zeros_like(pos)
    pos_ptr = pos.data.ptr
    result_ptr = result.data.ptr
    cuda_lib.ptov_pos_kernel(ctypes.cast(pos_ptr, ctypes.POINTER(ctypes.c_double)),
                             ctypes.c_int(Nmid), ctypes.c_double(dcell),
                             ctypes.cast(result_ptr, ctypes.POINTER(ctypes.c_double)))
    return result

def makeRF0(m, q, w, Nr, Nz, Nrmid, dr):
    RF = cp.ones((Nr, Nz), dtype=cp.float64)
    RF_ptr = RF.data.ptr
    cuda_lib.make_rf0_kernel(
        ctypes.c_double(m), ctypes.c_double(q), ctypes.c_double(w),
        ctypes.c_int(Nr), ctypes.c_int(Nz), ctypes.c_int(Nrmid),
        ctypes.c_double(dr), ctypes.cast(RF_ptr, ctypes.POINTER(ctypes.c_double))
    )
    return RF

def minDists(vf, vc):
    Ni = vf.shape[0]
    Nc = vc.shape[0]
    result = cp.zeros(4, dtype=cp.float64)
    cuda_lib.min_dists_kernel(
        ctypes.cast(vf.data.ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(vc.data.ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(Ni),
        ctypes.c_int(Nc),
        ctypes.cast(result.data.ptr, ctypes.POINTER(ctypes.c_double))
    )
    return result[0], result[1], result[2], result[3]

collision_kernel_proto = ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(ctypes.c_double),  # vf
    ctypes.POINTER(ctypes.c_double),  # vc
    ctypes.c_int,  # Ni
    ctypes.c_int,  # Nc
    ctypes.POINTER(ctypes.c_double),  # ErDC
    ctypes.POINTER(ctypes.c_double),  # EzDC
    ctypes.POINTER(ctypes.c_double),  # ErAC
    ctypes.POINTER(ctypes.c_double),  # EzAC
    ctypes.c_double,  # dr
    ctypes.c_double,  # dz
    ctypes.c_double,  # dt
    ctypes.c_int,  # Nrmid
    ctypes.c_int,  # Nzmid
    ctypes.POINTER(ctypes.c_double),  # Erfi
    ctypes.POINTER(ctypes.c_double),  # Ezfi
    ctypes.POINTER(ctypes.c_double),  # Erfc
    ctypes.POINTER(ctypes.c_double),  # Ezfc
    ctypes.POINTER(ctypes.c_double),  # min_dists
    ctypes.POINTER(ctypes.c_int)      # collision_mode
)

collision_kernel = collision_kernel_proto(("collision_kernel", cuda_lib))

@cuda.jit
def collision_mode_kernel(rii, rid, a, e, result):
    i = cuda.grid(1)
    if i == 0:
        result[0] = (a * rii * rii) / (rid * rid * rid * rid * rid) > e

def collisionMode(rii, rid, a, e=0.3):
    result = cp.zeros(1, dtype=cp.bool_)
    rii_scalar = float(rii)  # Convert to scalar
    rid_scalar = float(rid)  # Convert to scalar
    a_scalar = float(a)      # Convert to scalar
    collision_mode_kernel[1, 1](rii_scalar, rid_scalar, a_scalar, e, result)
    return result[0].item()  # Convert to Python bool

def collisionParticlesFields(vf, vc, ErDC, EzDC, ErAC, EzAC, dr, dz, dt, Nrmid, Nzmid):
    Ni, Nc = vf.shape[0], vc.shape[0]
    
    Erfi = cp.zeros(Ni, dtype=cp.float64)
    Ezfi = cp.zeros(Ni, dtype=cp.float64)
    Erfc = cp.zeros((Nc, 2), dtype=cp.float64)
    Ezfc = cp.zeros((Nc, 2), dtype=cp.float64)
    min_dists = cp.zeros(4, dtype=cp.float64)
    collision_mode = cp.zeros(1, dtype=cp.int32)
    
    collision_kernel(
        ctypes.cast(vf.data.ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(vc.data.ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(Ni),
        ctypes.c_int(Nc),
        ctypes.cast(ErDC.data.ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(EzDC.data.ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(ErAC.data.ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(EzAC.data.ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.c_double(dr),
        ctypes.c_double(dz),
        ctypes.c_double(dt),
        ctypes.c_int(Nrmid),
        ctypes.c_int(Nzmid),
        ctypes.cast(Erfi.data.ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(Ezfi.data.ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(Erfc.data.ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(Ezfc.data.ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(min_dists.data.ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(collision_mode.data.ptr, ctypes.POINTER(ctypes.c_int))
    )
    
    return vc, Erfi, Ezfi, Erfc, Ezfc, min_dists, collision_mode[0]

def setup_simulation(N, nu, m, q, Nr, Nz):
    Nrmid = (Nr - 1) // 2
    Nzmid = (Nz - 1) // 2
    
    # Calculate initial positions
    positions = calcPositions(N)
    
    # Create vf array
    vf = cp.zeros((N, 7), dtype=cp.float64)
    vf[:, 0] = positions[:, 0]  # r positions
    vf[:, 1] = positions[:, 1]  # z positions
    vf[:, 4] = q
    vf[:, 5] = m
    
    # Create RF field
    w = 2 * np.pi * nu
    RF = makeRF0(m, q, w, Nr, Nz, Nrmid, 1e-6)  # Assuming dr = 1e-6 for now
    
    return vf, RF

@cuda.jit
def update_positions_kernel(vf, dr, dz, dt, Nr, Nz, Nrmid, Nzmid):
    i = cuda.grid(1)
    if i < vf.shape[0]:
        vf[i, 0] += vf[i, 2] * dt
        vf[i, 1] += vf[i, 3] * dt
        rCell = (vf[i, 0] / dr + Nrmid)
        zCell = (vf[i, 1] / dz + Nzmid)
        if rCell > Nr - 2 or rCell < 1 or zCell > Nz - 2 or zCell < 1:
            vf[i, :] = 0.0
            vf[i, 0] = 2.0
            vf[i, 1] = 2.0
            vf[i, 5] = 1e6

def updatePoss(vf, dr, dz, dt, Nr, Nz, Nrmid, Nzmid):
    threads_per_block = 256
    blocks = (vf.shape[0] + threads_per_block - 1) // threads_per_block
    update_positions_kernel[blocks, threads_per_block](vf, dr, dz, dt, Nr, Nz, Nrmid, Nzmid)
    return vf

@cuda.jit
def update_velocities_kernel(vf, Erf, Ezf, dt):
    i = cuda.grid(1)
    if i < vf.shape[0]:
        Fr = vf[i, 4] * Erf[i]
        Fz = vf[i, 4] * Ezf[i]
        vf[i, 2] += Fr * dt / vf[i, 5]
        vf[i, 3] += Fz * dt / vf[i, 5]

def updateVels(vf, Erf, Ezf, dt):
    threads_per_block = 256
    blocks = (vf.shape[0] + threads_per_block - 1) // threads_per_block
    update_velocities_kernel[blocks, threads_per_block](vf, Erf, Ezf, dt)
    return vf

@cuda.jit
def solve_fields_kernel(vf, ErDC, EzDC, ErAC, EzAC, Nrmid, Nzmid, dr, dz, Erf, Ezf):
    i = cuda.grid(1)
    if i < vf.shape[0]:
        eps0 = 8.854e-12
        C1 = 4 * math.pi * eps0
        
        jCell = int(round((vf[i, 0] / dr + Nrmid)))
        kCell = int(round((vf[i, 1] / dz + Nzmid)))
        
        # Ensure we're not accessing out of bounds
        jCell = max(0, min(jCell, ErDC.shape[0] - 1))
        kCell = max(0, min(kCell, ErDC.shape[1] - 1))
        
        Erf[i] = ErDC[jCell, kCell] + ErAC[jCell, kCell]
        Ezf[i] = EzDC[jCell, kCell] + EzAC[jCell, kCell]
        
        for j in range(vf.shape[0]):
            if j != i:
                rdist = vf[j, 0] - vf[i, 0]
                zdist = vf[j, 1] - vf[i, 1]
                sqDist = rdist**2 + zdist**2
                if sqDist > 1e-20:  # Avoid division by zero
                    invDist = 1.0 / math.sqrt(sqDist)
                    Erf[i] += -rdist * invDist**3 * vf[j, 4] / C1
                    Ezf[i] += -zdist * invDist**3 * vf[j, 4] / C1


def solveFields(vf, ErDC, EzDC, ErAC, EzAC, Nrmid, Nzmid, Ni, dr, dz):
    Erf = cp.zeros(Ni, dtype=cp.float64)
    Ezf = cp.zeros(Ni, dtype=cp.float64)
    
    threads_per_block = 256
    blocks = (Ni + threads_per_block - 1) // threads_per_block
    
    solve_fields_kernel[blocks, threads_per_block](vf, ErDC, EzDC, ErAC, EzAC, Nrmid, Nzmid, dr, dz, Erf, Ezf)
    
    cp.cuda.Stream.null.synchronize()
    
    return Erf, Ezf

def mcCollision(vf, rc, zc, vrc, vzc, qc, mc, ac, Nt, dtSmall, RF, DC, Nr, Nz, dr, dz, dtLarge, dtCollision, eii=0.01, eid=0.01):
    Nrmid = (Nr - 1) // 2
    Nzmid = (Nz - 1) // 2
    Ni = vf.shape[0]
    Nc = 1
    vc = cp.zeros((Nc, 7), dtype=cp.float64)
    vc[0, :] = cp.array([rc, zc, vrc, vzc, qc, mc, ac], dtype=cp.float64)
    
    nullFields = cp.zeros((Nr, Nz), dtype=cp.float64)
    
    for i in range(Nt):
        rid, rii, vid, vii = minDists(vf, vc)
        collision = collisionMode(rii, rid, vc[0, 6].item(), 0.1)
        
        if collision:
            dtNow = float(rid * eid / (5 * vid)) if vid > 1e-20 else dtSmall
        else:
            dtNow = dtSmall
        
        dtNow = max(dtNow, dtCollision)
        
        Erfi, Ezfi = solveFields(vf, nullFields, DC, RF, nullFields, Nrmid, Nzmid, Ni, dr, dz)
        
        if vc[0, 5] < 1e6:
            vc, Erfic, Ezfic, Erfc, Ezfc, _, _ = collisionParticlesFields(vf, vc, nullFields, DC, RF, nullFields, dr, dz, dtNow, Nrmid, Nzmid)
            vc = updatePoss(vc, dr, dz, dtNow, Nr, Nz, Nrmid, Nzmid)
            Erfi += Erfic
            Ezfi += Ezfic
        else:
            dtNow = dtLarge
        
        vf = updateVels(vf, Erfi, Ezfi, dtNow)
        vf = updatePoss(vf, dr, dz, dtNow, Nr, Nz, Nrmid, Nzmid)
        
        if cp.any(cp.isnan(vf)):
            print(f"NaN detected at iteration {i}")
            print("vf at NaN detection:")
            print(vf)
            print(f"Erfi: {Erfi}")
            print(f"Ezfi: {Ezfi}")
            break
        
        if cp.sum(vf[:, 5]) > 1e5:
            print(f"Ion ejected at iteration {i}")
            print("vf at ejection:")
            print(vf)
            print(f"Electric fields at ejection: Erfi = {Erfi}, Ezfi = {Ezfi}")
            break
        
        for j in range(1, Ni):
            if vf[j, 1] > vf[j-1, 1]:
                print(f"Ion reordering detected at iteration {i}")
                print("vf at reordering:")
                print(vf)
                return 1
        
        if i % 100 == 0:
            print(f"Iteration {i}: Max ion position: {cp.max(cp.abs(vf[:, :2]))}")
    
    return 0

# Test the implemented functions
if __name__ == "__main__":
    # Test calcPositions
    N = 10
    positions = calcPositions(N)
    print("Calculated positions:", positions)

    # Test lengthScale
    nu = 1e6
    ls = lengthScale(nu)
    print("Length scale:", ls)

    # Test ptovPos
    pos = cp.array([1e-6, 2e-6, 3e-6])
    Nmid = 50
    dcell = 1e-6
    virtual_pos = ptovPos(pos, Nmid, dcell)
    print("Virtual positions:", virtual_pos)

    # Test makeRF0
    m = 6.6335209e-26
    q = 1.60217663e-19
    w = 2 * np.pi * 1e6
    Nr, Nz = 100, 100
    Nrmid = (Nr - 1) // 2
    dr = 1e-6
    RF = makeRF0(m, q, w, Nr, Nz, Nrmid, dr)
    print("RF field shape:", RF.shape)
    print("RF field sample:", RF[:5, :5])

    # Test minDists
    vf = cp.random.rand(10, 7)
    vc = cp.random.rand(5, 7)
    rid, rii, vid, vii = minDists(vf, vc)
    print("Minimum distances:", rid, rii, vid, vii)

    rii = 1e-6
    rid = 1e-5
    a = 1e-10
    e = 0.3
    collision = collisionMode(rii, rid, a, e)
    print("Collision mode:", collision)

    # Test collisionParticlesFields
    Ni, Nc = 10, 5
    Nr, Nz = 100, 100
    vf = cp.random.rand(Ni, 7)
    vc = cp.random.rand(Nc, 7)
    ErDC = cp.random.rand(Nr, Nz)
    EzDC = cp.random.rand(Nr, Nz)
    ErAC = cp.random.rand(Nr, Nz)
    EzAC = cp.random.rand(Nr, Nz)
    dr, dz, dt = 1e-6, 1e-6, 1e-9
    Nrmid, Nzmid = Nr // 2, Nz // 2
    
    vc, Erfi, Ezfi, Erfc, Ezfc, min_dists, collision_mode = collisionParticlesFields(
        vf, vc, ErDC, EzDC, ErAC, EzAC, dr, dz, dt, Nrmid, Nzmid
    )
    print("Collision particles fields:")
    print("vc shape:", vc.shape)
    print("Erfi shape:", Erfi.shape)
    print("Ezfi shape:", Ezfi.shape)
    print("Erfc shape:", Erfc.shape)
    print("Ezfc shape:", Ezfc.shape)
    print("Minimum distances:", min_dists)
    print("Collision mode:", collision_mode)

    N = 3  # Set to 3 ions
    nu = 1e6
    m = 6.6335209e-26
    q = 1.60217663e-19
    Nr, Nz = 100, 100
    dr, dz = 1e-6, 1e-6
    dtSmall = 1e-12  # Further reduced
    dtLarge = 1e-11  # Further reduced
    dtCollision = 1e-13  # Further reduced
    Nt = 100000  # Increased

    try:
        vf, RF = setup_simulation(N, nu, m, q, Nr, Nz)
        print("setup_simulation successful")
        print("vf shape:", vf.shape)
        print("RF shape:", RF.shape)
        print("Initial vf:")
        print(vf)
        
        DC = cp.zeros((Nr, Nz), dtype=cp.float64)  # Dummy DC field

        rc, zc, vrc, vzc = 0, 0, 0, 0
        qc, mc, ac = 0, 1e-26, 1e-40

        result = mcCollision(vf, rc, zc, vrc, vzc, qc, mc, ac, Nt, dtSmall, RF, DC, Nr, Nz, dr, dz, dtLarge, dtCollision)
        print("mcCollision result:", result)
    except Exception as e:
        print(f"Error in simulation setup or execution: {e}")
        import traceback
        traceback.print_exc()