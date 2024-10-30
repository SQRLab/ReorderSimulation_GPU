#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>

#define PI 3.14159265358979323846
#define EPSILON0 8.8541878128e-12

__device__ double round_gpu(double x) {
    return floor(x + 0.5);
}

__global__ void length_scale_kernel_impl(double nu, double M, int Z, double* result) {
    const double e = 1.602176634e-19;
    const double eps0 = 8.8541878128e-12;
    const double pi = 3.141592653589793;
    
    *result = pow((Z * Z * e * e) / (4 * pi * eps0 * M * nu * nu), 1.0 / 3.0);
    printf("Debug: length_scale_kernel_impl calculated %e\n", *result);
}

__global__ void ptov_pos_kernel_impl(double* pos, int Nmid, double dcell, double* result) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < gridDim.x * blockDim.x) {
        result[i] = (pos[i] / dcell + Nmid);
    }
}

__global__ void ion_position_potential_kernel_impl(double* x, double* result, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double sum1 = 0, sum2 = 0;
        for (int n = 0; n < i; n++) {
            double diff = x[i] - x[n];
            if (fabs(diff) > 1e-10) {  // Avoid division by very small numbers
                sum1 += 1 / (diff * diff);
            }
        }
        for (int n = i + 1; n < N; n++) {
            double diff = x[i] - x[n];
            if (fabs(diff) > 1e-10) {  // Avoid division by very small numbers
                sum2 += 1 / (diff * diff);
            }
        }
        result[i] = x[i] - sum1 + sum2;
    }
}

__global__ void calc_positions_kernel_impl(double* x, int N, double estimated_extreme, int iterations) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        x[i] = -estimated_extreme + (2 * estimated_extreme * i) / (N - 1);
        for (int iter = 0; iter < iterations; iter++) {
            double sum1 = 0, sum2 = 0;
            for (int n = 0; n < i; n++)
                sum1 += 1 / ((x[i] - x[n]) * (x[i] - x[n]));
            for (int n = i + 1; n < N; n++)
                sum2 += 1 / ((x[i] - x[n]) * (x[i] - x[n]));
            x[i] = x[i] - 0.1 * (x[i] - sum1 + sum2);  // Simple gradient descent
        }
    }
}

__global__ void min_dists_kernel_impl(double* vf, double* vc, int Ni, int Nc, double* result) {
    __shared__ double s_rid2[256], s_rii2[256], s_vid2[256], s_vii2[256];
    int tid = threadIdx.x;
    s_rid2[tid] = s_rii2[tid] = s_vid2[tid] = s_vii2[tid] = 1e6;
    __syncthreads();

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < Ni; i += blockDim.x * gridDim.x) {
        for (int j = i + 1; j < Ni; j++) {
            double r = vf[i*7] - vf[j*7], z = vf[i*7+1] - vf[j*7+1];
            double vr = vf[i*7+2] - vf[j*7+2], vz = vf[i*7+3] - vf[j*7+3];
            double dist2 = r*r + z*z;
            double v2 = vr*vr + vz*vz;
            if (dist2 < s_rii2[tid]) {
                s_vii2[tid] = v2;
                s_rii2[tid] = dist2;
            }
        }
        for (int j = 0; j < Nc; j++) {
            double r = vf[i*7] - vc[j*7], z = vf[i*7+1] - vc[j*7+1];
            double vr = vf[i*7+2] - vc[j*7+2], vz = vf[i*7+3] - vc[j*7+3];
            double dist2 = r*r + z*z;
            double v2 = vr*vr + vz*vz;
            if (dist2 < s_rid2[tid]) {
                s_vid2[tid] = v2;
                s_rid2[tid] = dist2;
            }
        }
    }
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            s_rid2[tid] = min(s_rid2[tid], s_rid2[tid + s]);
            s_rii2[tid] = min(s_rii2[tid], s_rii2[tid + s]);
            s_vid2[tid] = min(s_vid2[tid], s_vid2[tid + s]);
            s_vii2[tid] = min(s_vii2[tid], s_vii2[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[0] = sqrt(s_rid2[0]);
        result[1] = sqrt(s_rii2[0]);
        result[2] = sqrt(s_vid2[0]);
        result[3] = sqrt(s_vii2[0]);
    }
}

__device__ int collision_mode(double rii, double rid, double a, double e) {
    return (a * rii * rii) / (rid * rid * rid * rid * rid) > e;
}

__device__ void min_dists_device(double* vf, double* vc, int Ni, int Nc, double* result) {
    double rid2 = 1e6, rii2 = 1e6, vid2 = 1e6, vii2 = 1e6;
    
    for (int i = 0; i < Ni; i++) {
        for (int j = i + 1; j < Ni; j++) {
            double r = vf[i*7] - vf[j*7], z = vf[i*7+1] - vf[j*7+1];
            double vr = vf[i*7+2] - vf[j*7+2], vz = vf[i*7+3] - vf[j*7+3];
            double dist2 = r*r + z*z;
            double v2 = vr*vr + vz*vz;
            if (dist2 < rii2) {
                vii2 = v2;
                rii2 = dist2;
            }
        }
        for (int j = 0; j < Nc; j++) {
            double r = vf[i*7] - vc[j*7], z = vf[i*7+1] - vc[j*7+1];
            double vr = vf[i*7+2] - vc[j*7+2], vz = vf[i*7+3] - vc[j*7+3];
            double dist2 = r*r + z*z;
            double v2 = vr*vr + vz*vz;
            if (dist2 < rid2) {
                vid2 = v2;
                rid2 = dist2;
            }
        }
    }
    
    result[0] = sqrt(rid2);
    result[1] = sqrt(rii2);
    result[2] = sqrt(vid2);
    result[3] = sqrt(vii2);
}

__global__ void collision_kernel_impl(double* vf, double* vc, int Ni, int Nc, 
                                      double* ErDC, double* EzDC, double* ErAC, double* EzAC,
                                      double dr, double dz, double dt, int Nrmid, int Nzmid,
                                      double* Erfi, double* Ezfi, double* Erfc, double* Ezfc,
                                      double* min_dists_result, int* collision_mode_result) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ double s_min_dists[4];
    if (threadIdx.x == 0) {
        min_dists_device(vf, vc, Ni, Nc, s_min_dists);
    }
    __syncthreads();
    
    if (i == 0) {
        for (int j = 0; j < 4; j++) {
            min_dists_result[j] = s_min_dists[j];
        }
        collision_mode_result[0] = collision_mode(s_min_dists[1], s_min_dists[0], vc[6], 0.3);
    }
    
    if (i < Nc) {
        // Process collision particles
        int jCell = (int)round_gpu((vc[i*7] / dr + Nrmid));
        int kCell = (int)round_gpu((vc[i*7+1] / dz + Nzmid));
        
        if (jCell >= 0 && jCell < Nrmid*2 && kCell >= 0 && kCell < Nzmid*2) {
            Erfc[i*2] = ErDC[jCell*Nzmid*2 + kCell] + ErAC[jCell*Nzmid*2 + kCell];
            Ezfc[i*2] = EzDC[jCell*Nzmid*2 + kCell] + EzAC[jCell*Nzmid*2 + kCell];
            
            if (jCell > 0 && jCell < Nrmid*2-1 && kCell > 0 && kCell < Nzmid*2-1) {
                Erfc[i*2+1] = ((ErDC[(jCell+1)*Nzmid*2 + kCell] + ErAC[(jCell+1)*Nzmid*2 + kCell]) - 
                               (ErDC[(jCell-1)*Nzmid*2 + kCell] + ErAC[(jCell-1)*Nzmid*2 + kCell])) / (2*dr);
                Ezfc[i*2+1] = ((EzDC[jCell*Nzmid*2 + kCell+1] + EzAC[jCell*Nzmid*2 + kCell+1]) - 
                               (EzDC[jCell*Nzmid*2 + kCell-1] + EzAC[jCell*Nzmid*2 + kCell-1])) / (2*dz);
            }
        }

        // Process ion interactions
        for (int j = 0; j < Ni; j++) {
            double rdist = vf[j*7] - vc[i*7];
            double zdist = vf[j*7+1] - vc[i*7+1];
            double sqDist = rdist*rdist + zdist*zdist;
            if (sqDist > 1e-20) {  // Avoid division by zero
                double invDist = 1.0 / sqrt(sqDist);
                double projR = rdist * invDist;
                double projZ = zdist * invDist;
                Erfc[i*2] += -projR * vf[j*7+4] / (4 * PI * EPSILON0 * sqDist);
                Ezfc[i*2] += -projZ * vf[j*7+4] / (4 * PI * EPSILON0 * sqDist);
                Erfc[i*2+1] += 2 * projR * vf[j*7+4] / (4 * PI * EPSILON0 * sqDist * sqrt(sqDist));
                Ezfc[i*2+1] += 2 * projZ * vf[j*7+4] / (4 * PI * EPSILON0 * sqDist * sqrt(sqDist));
            }
        }

        // Induce dipole moment and apply force
        if (vc[i*7+6] != 0.0) {
            double pR = -2 * PI * EPSILON0 * vc[i*7+6] * Erfc[i*2];
            double pZ = -2 * PI * EPSILON0 * vc[i*7+6] * Ezfc[i*2];
            double Fr = fabs(pR) * Erfc[i*2+1];
            double Fz = fabs(pZ) * Ezfc[i*2+1];
            vc[i*7+2] += Fr * dt / vc[i*7+5];
            vc[i*7+3] += Fz * dt / vc[i*7+5];
        }
    }

    if (i < Ni) {
        // Process ions
        Erfi[i] = 0;
        Ezfi[i] = 0;
        for (int j = 0; j < Nc; j++) {
            if (vc[j*7+6] != 0.0) {
                double rdist = vc[j*7] - vf[i*7];
                double zdist = vc[j*7+1] - vf[i*7+1];
                double sqDist = rdist*rdist + zdist*zdist;
                if (sqDist > 1e-20) {  // Avoid division by zero
                    double invDist = 1.0 / sqrt(sqDist);
                    double Rhatr = rdist * invDist;
                    double Rhatz = zdist * invDist;
                    double pR = -2 * PI * EPSILON0 * vc[j*7+6] * Erfc[j*2];
                    double pZ = -2 * PI * EPSILON0 * vc[j*7+6] * Ezfc[j*2];
                    Erfi[i] += -fabs(pR) * (2*Rhatr) / (4 * PI * EPSILON0 * sqDist * invDist);
                    Ezfi[i] += -fabs(pZ) * (2*Rhatz) / (4 * PI * EPSILON0 * sqDist * invDist);
                }
            }
        }
    }
}

extern "C" {

__declspec(dllexport) void length_scale_kernel(double nu, double M, int Z, double* result) {
    double* d_result;
    cudaMalloc(&d_result, sizeof(double));
    length_scale_kernel_impl<<<1, 1>>>(nu, M, Z, d_result);
    cudaMemcpy(result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in length_scale_kernel: %s\n", cudaGetErrorString(err));
    }
}

__declspec(dllexport) void ptov_pos_kernel(double* pos, int Nmid, double dcell, double* result) {
    int size = 3;  // Assuming a fixed size of 3 for simplicity
    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    double *d_pos, *d_result;
    cudaMalloc(&d_pos, size * sizeof(double));
    cudaMalloc(&d_result, size * sizeof(double));
    
    cudaMemcpy(d_pos, pos, size * sizeof(double), cudaMemcpyHostToDevice);
    
    ptov_pos_kernel_impl<<<blocks, threadsPerBlock>>>(d_pos, Nmid, dcell, d_result);
    
    cudaMemcpy(result, d_result, size * sizeof(double), cudaMemcpyDeviceToHost);
    
    cudaFree(d_pos);
    cudaFree(d_result);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in ptov_pos_kernel: %s\n", cudaGetErrorString(err));
    }
}

__declspec(dllexport) void ion_position_potential_kernel(double* x, double* result, int N) {
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    ion_position_potential_kernel_impl<<<blocks, threadsPerBlock>>>(x, result, N);
    cudaDeviceSynchronize();
}

__declspec(dllexport) void calc_positions_kernel(double* x, int N, double estimated_extreme, int iterations) {
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    calc_positions_kernel_impl<<<blocks, threadsPerBlock>>>(x, N, estimated_extreme, iterations);
    cudaDeviceSynchronize();
}

__declspec(dllexport) void min_dists_kernel(double* vf, double* vc, int Ni, int Nc, double* result) {
    int threadsPerBlock = 256;
    int blocks = (Ni + threadsPerBlock - 1) / threadsPerBlock;
    min_dists_kernel_impl<<<blocks, threadsPerBlock>>>(vf, vc, Ni, Nc, result);
    cudaDeviceSynchronize();
}

__declspec(dllexport) void collision_kernel(double* vf, double* vc, int Ni, int Nc, 
                                            double* ErDC, double* EzDC, double* ErAC, double* EzAC,
                                            double dr, double dz, double dt, int Nrmid, int Nzmid,
                                            double* Erfi, double* Ezfi, double* Erfc, double* Ezfc,
                                            double* min_dists_result, int* collision_mode_result) {
    int threadsPerBlock = 256;
    int blocks = (max(Ni, Nc) + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("Debug: Launching collision kernel with %d blocks, %d threads per block\n", blocks, threadsPerBlock);
    printf("Debug: Ni = %d, Nc = %d, Nrmid = %d, Nzmid = %d\n", Ni, Nc, Nrmid, Nzmid);
    
    collision_kernel_impl<<<blocks, threadsPerBlock>>>(vf, vc, Ni, Nc, ErDC, EzDC, ErAC, EzAC,
                                                       dr, dz, dt, Nrmid, Nzmid, Erfi, Ezfi, Erfc, Ezfc,
                                                       min_dists_result, collision_mode_result);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in collision_kernel: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}

// Add a make_rf0_kernel function (previously implemented in Python)
__global__ void make_rf0_kernel_impl(double m, double q, double w, int Nr, int Nz, int Nrmid, double dr, double* RF) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < Nr && j < Nz) {
        double C = -m * (w * w) / q;
        RF[i * Nz + j] = -RF[i * Nz + j] * C * (Nrmid - i) * dr;
    }
}

__declspec(dllexport) void make_rf0_kernel(double m, double q, double w, int Nr, int Nz, int Nrmid, double dr, double* RF) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((Nr + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (Nz + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    make_rf0_kernel_impl<<<numBlocks, threadsPerBlock>>>(m, q, w, Nr, Nz, Nrmid, dr, RF);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in make_rf0_kernel: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}

}