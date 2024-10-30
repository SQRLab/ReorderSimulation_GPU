import ctypes
import cupy as cp

cuda_lib = ctypes.CDLL('./simple_collision.dll')

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
    
    return vc, Erfi, Ezfi, Erfc, Ezfc, min_dists, collision_mode

def test_collision():
    print("Testing collisionParticlesFields:")
    Ni, Nc = 10, 5
    Nr, Nz = 100, 100
    Nrmid, Nzmid = Nr // 2, Nz // 2
    dr, dz, dt = 1e-6, 1e-6, 1e-9
    
    vf = cp.random.rand(Ni, 7)
    vc = cp.random.rand(Nc, 7)
    ErDC = cp.random.rand(Nr, Nz)
    EzDC = cp.random.rand(Nr, Nz)
    ErAC = cp.random.rand(Nr, Nz)
    EzAC = cp.random.rand(Nr, Nz)
    
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
        print(f"Collision mode: {collision_mode[0]}")
    except Exception as e:
        print(f"Error in collisionParticlesFields: {e}")

if __name__ == "__main__":
    test_collision()