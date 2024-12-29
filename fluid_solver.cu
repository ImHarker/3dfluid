#include <cuda_runtime.h>
#include <omp.h>

#include <cmath>
#include <iostream>

#include "fluid_solver.h"

#define IX(i, j, k) ((i) + (M + 2) * (j) + (M + 2) * (N + 2) * (k))
#define SWAP(x0, x)      \
    {                    \
        float *tmp = x0; \
        x0 = x;          \
        x = tmp;         \
    }
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define LINEARSOLVERTIMES 20

// Add sources (density or velocity)
__global__ void add_source_kernel(int M, int N, int O, float *x, float *s,
                                  float dt) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int size = (M + 2) * (N + 2) * (O + 2);
    if (tid >= size) return;
    x[tid] += dt * s[tid];
}

void add_source(int M, int N, int O, float *x, float *s, float dt) {
    int size = (M + 2) * (N + 2) * (O + 2);
    int nThreads = 1024;
    int nBlocks = (size + nThreads - 1) / nThreads;

    add_source_kernel<<<nBlocks, nThreads>>>(M, N, O, x, s, dt);
}

// Kernel Z-faces (top and bottom)
__global__ void set_bnd_Z(int M, int N, int O, int b, float *x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i > M || j > N) return;

    x[IX(i, j, 0)] = (b == 3) ? -x[IX(i, j, 1)] : x[IX(i, j, 1)];
    x[IX(i, j, O + 1)] = (b == 3) ? -x[IX(i, j, O)] : x[IX(i, j, O)];
}

// Kernel X-Faces (left and right)
__global__ void set_bnd_X(int M, int N, int O, int b, float *x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i > N || j > O) return;
    x[IX(0, i, j)] = (b == 1) ? -x[IX(1, i, j)] : x[IX(1, i, j)];
    x[IX(M + 1, i, j)] = (b == 1) ? -x[IX(M, i, j)] : x[IX(M, i, j)];
}

// Kernel for Y-Faces (front and back)
__global__ void set_bnd_Y(int M, int N, int O, int b, float *x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i > M || j > O) return;
    x[IX(i, 0, j)] = (b == 2) ? -x[IX(i, 1, j)] : x[IX(i, 1, j)];
    x[IX(i, N + 1, j)] = (b == 2) ? -x[IX(i, N, j)] : x[IX(i, N, j)];
}

// Kernel for corners
__global__ void set_bnd_corners(int M, int N, int O, float *x) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    x[IX(0, 0, 0)] = 0.33f * (x[IX(1, 0, 0)] + x[IX(0, 1, 0)] + x[IX(0, 0, 1)]);

    x[IX(M + 1, 0, 0)] =
        0.33f * (x[IX(M, 0, 0)] + x[IX(M + 1, 1, 0)] + x[IX(M + 1, 0, 1)]);

    x[IX(0, N + 1, 0)] =
        0.33f * (x[IX(1, N + 1, 0)] + x[IX(0, N, 0)] + x[IX(0, N + 1, 1)]);

    x[IX(M + 1, N + 1, 0)] = 0.33f * (x[IX(M, N + 1, 0)] + x[IX(M + 1, N, 0)] +
                                      x[IX(M + 1, N + 1, 1)]);
}

void set_bnd(int M, int N, int O, int b, float *x) {
    dim3 blockDim(32, 32);

    // Z-face
    dim3 gridDimZ((M + blockDim.x - 1) / blockDim.x,
                  (N + blockDim.y - 1) / blockDim.y);
    set_bnd_Z<<<gridDimZ, blockDim>>>(M, N, O, b, x);

    // X-face
    dim3 gridDimX((N + blockDim.x - 1) / blockDim.x,
                  (O + blockDim.y - 1) / blockDim.y);
    set_bnd_X<<<gridDimX, blockDim>>>(M, N, O, b, x);

    // Y-Face
    dim3 gridDimY((M + blockDim.x - 1) / blockDim.x,
                  (O + blockDim.y - 1) / blockDim.y);
    set_bnd_Y<<<gridDimY, blockDim>>>(M, N, O, b, x);

    // Corners
    set_bnd_corners<<<1, 1>>>(M, N, O, x);
}

// Kernel for computing the red or black points
__global__ void red_black_kernel(int M, int N, int O, float *x, const float *x0,
                                 float a, float c, bool isRed,
                                 float *maxChange) {
    __shared__ float shared_max[16 * 16 * 4];
    float local_max = 0;
    int tid = threadIdx.x + threadIdx.y * blockDim.x +
              threadIdx.z * blockDim.x * blockDim.y;
    // Compute 3D thread ID
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    // Ensure within bounds and check for red/black points
    if (i <= 0 || j <= 0 || k <= 0 || i >= M || j >= N || k >= O ||
        (i + j + k) % 2 != isRed)
        return;

    int index = IX(i, j, k);
    float old_x = x[index];
    x[index] = (x0[index] + a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                                 x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                                 x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) /
               c;

    // Calculate the change
    float change = fabsf(x[index] - old_x);
    local_max = fmaxf(local_max, change);

    shared_max[tid] = local_max;
    __syncthreads();

    // Reduction
    for (int stride = (blockDim.x * blockDim.y * blockDim.z) >> 1; stride > 0;
         stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }

    // Write maxChange to global memory
    if (tid == 0) {
        atomicMax((int *)maxChange, __float_as_int(shared_max[0]));
    }
}

void lin_solve(int M, int N, int O, int b, float *x, float *x0, float a,
               float c) {
    // Allocate device memory
    float *d_maxChange;
    cudaMalloc((void **)&d_maxChange, sizeof(float));

    // Define block and grid sizes
    dim3 blockDim(16, 16, 4);
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y,
                 (O + blockDim.z - 1) / blockDim.z);

    float tol = 1e-7f, maxChange;
    int l = 0;

    do {
        maxChange = 0.0f;
        cudaMemset(d_maxChange, 0, sizeof(float));
        // Launch kernel for red points
        red_black_kernel<<<gridDim, blockDim>>>(M, N, O, x, x0, a, c, true,
                                                d_maxChange);
        // Launch kernel for black points
        red_black_kernel<<<gridDim, blockDim>>>(M, N, O, x, x0, a, c, false,
                                                d_maxChange);

        set_bnd(M, N, O, b, x);

        // Copy maxChange back to host
        cudaMemcpy(&maxChange, d_maxChange, sizeof(float),
                   cudaMemcpyDeviceToHost);

    } while (maxChange > tol && ++l < 20);

    cudaFree(d_maxChange);
}

// Diffusion step (uses implicit method)
void diffuse(int M, int N, int O, int b, float *x, float *x0, float diff,
             float dt) {
    int max = MAX(MAX(M, N), O);
    float a = dt * diff * max * max;
    lin_solve(M, N, O, b, x, x0, a, 1 + 6 * a);
}

__device__ float clamp(float value, float _min, float _max) {
    return max(min(value, _max), _min);
}

// Advection step (uses velocity field to move quantities)
__global__ void advect_kernel(int M, int N, int O, int b, float *d, float *d0,
                              float *u, float *v, float *w, float dt) {
    float dtX = dt * M, dtY = dt * N, dtZ = dt * O;
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i > M || j > N || k > O) return;
    float x = i - dtX * u[IX(i, j, k)];
    float y = j - dtY * v[IX(i, j, k)];
    float z = k - dtZ * w[IX(i, j, k)];

    x = clamp(x, 0.5f, M + 0.5f);
    y = clamp(y, 0.5f, N + 0.5f);
    z = clamp(z, 0.5f, O + 0.5f);

    int i0 = static_cast<int>(x), i1 = i0 + 1;
    int j0 = static_cast<int>(y), j1 = j0 + 1;
    int k0 = static_cast<int>(z), k1 = k0 + 1;

    float s1 = x - i0, s0 = 1 - s1;
    float t1 = y - j0, t0 = 1 - t1;
    float u1 = z - k0, u0 = 1 - u1;

    d[IX(i, j, k)] =
        s0 * (t0 * (u0 * d0[IX(i0, j0, k0)] + u1 * d0[IX(i0, j0, k1)]) +
              t1 * (u0 * d0[IX(i0, j1, k0)] + u1 * d0[IX(i0, j1, k1)])) +
        s1 * (t0 * (u0 * d0[IX(i1, j0, k0)] + u1 * d0[IX(i1, j0, k1)]) +
              t1 * (u0 * d0[IX(i1, j1, k0)] + u1 * d0[IX(i1, j1, k1)]));
}

void advect(int M, int N, int O, int b, float *d, float *d0, float *u, float *v,
            float *w, float dt) {
    dim3 blockDim(16, 16, 4);
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y,
                 (O + blockDim.z - 1) / blockDim.z);
    advect_kernel<<<gridDim, blockDim>>>(M, N, O, b, d, d0, u, v, w, dt);

    set_bnd(M, N, O, b, d);
}

// Projection step to ensure incompressibility (make the velocity field
// divergence-free)
__global__ void divergence_kernel(int M, int N, int O, const float *u,
                                  const float *v, const float *w, float *div,
                                  float *p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i > M || j > N || k > O) return;
    div[IX(i, j, k)] = -0.5f *
                        ((u[IX(i + 1, j, k)] - u[IX(i - 1, j, k)]) +
                         (v[IX(i, j + 1, k)] - v[IX(i, j - 1, k)]) +
                         (w[IX(i, j, k + 1)] - w[IX(i, j, k - 1)])) /
                        MAX(M, MAX(N, O));

    p[IX(i, j, k)] = 0.0f;
}

__global__ void velocity_kernel(int M, int N, int O, float *u, float *v,
                                float *w, const float *p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i > M || j > N || k > O) return;
    u[IX(i, j, k)] += -0.5f * (p[IX(i + 1, j, k)] - p[IX(i - 1, j, k)]);
    v[IX(i, j, k)] += -0.5f * (p[IX(i, j + 1, k)] - p[IX(i, j - 1, k)]);
    w[IX(i, j, k)] += -0.5f * (p[IX(i, j, k + 1)] - p[IX(i, j, k - 1)]);
}

void project(int M, int N, int O, float *u, float *v, float *w, float *p,
             float *div) {
    dim3 blockDim(16, 16, 4);
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y,
                 (O + blockDim.z) / blockDim.z);

    divergence_kernel<<<gridDim, blockDim>>>(M, N, O, u, v, w, div, p);

    set_bnd(M, N, O, 0, div);
    set_bnd(M, N, O, 0, p);
    lin_solve(M, N, O, 0, p, div, 1, 6);

    velocity_kernel<<<gridDim, blockDim>>>(M, N, O, u, v, w, p);

    set_bnd(M, N, O, 1, u);
    set_bnd(M, N, O, 2, v);
    set_bnd(M, N, O, 3, w);
}

// Step function for density
void dens_step(int M, int N, int O, float *x, float *x0, float *u, float *v,
               float *w, float diff, float dt) {         
    add_source(M, N, O, x, x0, dt);
    SWAP(x0, x);
    diffuse(M, N, O, 0, x, x0, diff, dt);
    SWAP(x0, x);
    advect(M, N, O, 0, x, x0, u, v, w, dt);
}

// Step function for velocity
void vel_step(int M, int N, int O, float *u, float *v, float *w, float *u0,
              float *v0, float *w0, float visc, float dt) {
    add_source(M, N, O, u, u0, dt);
    add_source(M, N, O, v, v0, dt);
    add_source(M, N, O, w, w0, dt);
    SWAP(u0, u);
    diffuse(M, N, O, 1, u, u0, visc, dt);
    SWAP(v0, v);
    diffuse(M, N, O, 2, v, v0, visc, dt);
    SWAP(w0, w);
    diffuse(M, N, O, 3, w, w0, visc, dt);
    project(M, N, O, u, v, w, u0, v0);
    SWAP(u0, u);
    SWAP(v0, v);
    SWAP(w0, w);
    advect(M, N, O, 1, u, u0, u0, v0, w0, dt);
    advect(M, N, O, 2, v, v0, u0, v0, w0, dt);
    advect(M, N, O, 3, w, w0, u0, v0, w0, dt);
    project(M, N, O, u, v, w, u0, v0);
}