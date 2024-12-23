#include <cuda_runtime.h>
#include <omp.h>
#include <iostream>

#include <cmath>

#include "fluid_solver.h"

#define IX(i, j, k) ((i) + (M + 2) * (j) + (M + 2) * (N + 2) * (k))
#define SWAP(x0, x)      \
    {                    \
        float* tmp = x0; \
        x0 = x;          \
        x = tmp;         \
    }
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define LINEARSOLVERTIMES 20

// Add sources (density or velocity)
void add_source(int M, int N, int O, float* x, float* s, float dt) {
    int size = (M + 2) * (N + 2) * (O + 2);
    for (int i = 0; i < size; i++) {
        x[i] += dt * s[i];
    }
}

// Set boundary conditions
void set_bnd(int M, int N, int O, int b, float* x) {
    int i, j;

// Set boundary on faces
#pragma omp parallel private(i, j)
    {
#pragma omp for collapse(2)
        for (j = 1; j <= N; j++) {
            for (i = 1; i <= M; i++) {
                x[IX(i, j, 0)] = b == 3 ? -x[IX(i, j, 1)] : x[IX(i, j, 1)];
                x[IX(i, j, O + 1)] = b == 3 ? -x[IX(i, j, O)] : x[IX(i, j, O)];
            }
        }
#pragma omp for collapse(2)
        for (j = 1; j <= O; j++) {
            for (i = 1; i <= N; i++) {
                x[IX(0, i, j)] = b == 1 ? -x[IX(1, i, j)] : x[IX(1, i, j)];
                x[IX(M + 1, i, j)] = b == 1 ? -x[IX(M, i, j)] : x[IX(M, i, j)];
            }
        }
#pragma omp for collapse(2)
        for (j = 1; j <= O; j++) {
            for (i = 1; i <= M; i++) {
                x[IX(i, 0, j)] = b == 2 ? -x[IX(i, 1, j)] : x[IX(i, 1, j)];
                x[IX(i, N + 1, j)] = b == 2 ? -x[IX(i, N, j)] : x[IX(i, N, j)];
            }
        }
    }

    // Set corners
    x[IX(0, 0, 0)] = 0.33f * (x[IX(1, 0, 0)] + x[IX(0, 1, 0)] + x[IX(0, 0, 1)]);
    x[IX(M + 1, 0, 0)] =
        0.33f * (x[IX(M, 0, 0)] + x[IX(M + 1, 1, 0)] + x[IX(M + 1, 0, 1)]);
    x[IX(0, N + 1, 0)] =
        0.33f * (x[IX(1, N + 1, 0)] + x[IX(0, N, 0)] + x[IX(0, N + 1, 1)]);
    x[IX(M + 1, N + 1, 0)] = 0.33f * (x[IX(M, N + 1, 0)] + x[IX(M + 1, N, 0)] +
                                      x[IX(M + 1, N + 1, 1)]);
}

// Linear solve for implicit methods (diffusion)

#if 0
// Original Code provided in the repo
void lin_solve(int M, int N, int O, int b, float* x, float* x0, float a, float c) {
    // ORIGINAL
    for (int l = 0; l < LINEARSOLVERTIMES; l++) {
        for (int i = 1; i <= M; i++) {
            for (int j = 1; j <= N; j++) {
                for (int k = 1; k <= O; k++) {
                    x[IX(i, j, k)] = (x0[IX(i, j, k)] +
                                      a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                                           x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                                           x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) /
                                     c;
                }
            }
        }
        set_bnd(M, N, O, b, x);
    }
}

#elif 0
// Buffer version that doesnt work. This was a test to evaluate performance if there wasnt data dependency between iterations.
void lin_solve(int M, int N, int O, int b, float* x, float* x0, float a, float c) {
    // BUFFER: POC doesnt work
    float* temp = (float*)malloc((M + 2) * (N + 2) * (O + 2) * sizeof(float));  // temp buffer

    for (int l = 0; l < LINEARSOLVERTIMES; l++) {
        for (int k = 1; k <= O; k++) {
            for (int j = 1; j <= N; j++) {
                for (int i = 1; i <= M; i++) {
                    temp[IX(i, j, k)] = (x0[IX(i, j, k)] +
                                         a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                                              x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                                              x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) /
                                        c;
                }
            }
        }

        for (int k = 1; k <= O; k++) {
            for (int j = 1; j <= N; j++) {
                for (int i = 1; i <= M; i++) {
                    x[IX(i, j, k)] = temp[IX(i, j, k)];
                }
            }
        }

        set_bnd(M, N, O, b, x);
    }

    free(temp);  // Free temp buffer
}

#elif 0
// Final Code
void lin_solve(int M, int N, int O, int b, float* x, float* x0, float a, float c) {
    constexpr int BLOCK_SIZE = 4;  // 4 = melhor tempo | 2 = menos cache misses
    const float cRecip = 1.0f / c;
    const int jOffset = (M + 2);
    const int kOffset = jOffset * (N + 2);
    for (int l = 0; l < LINEARSOLVERTIMES; ++l) {
        for (int k = 1; k <= O; k += BLOCK_SIZE) {
            for (int j = 1; j <= N; j += BLOCK_SIZE) {
                for (int i = 1; i <= M; i += BLOCK_SIZE) {
                    for (int kBlock = k; kBlock < k + BLOCK_SIZE && kBlock <= O; kBlock++) {
                        for (int jBlock = j; jBlock < j + BLOCK_SIZE && jBlock <= N; jBlock++) {
                            for (int iBlock = i; iBlock < i + BLOCK_SIZE && iBlock <= M; iBlock++) {
                                int index = IX(iBlock, jBlock, kBlock);
                                x[index] = (x0[index] +
                                            a * (x[index - 1] + x[index + 1] +
                                                 x[index - jOffset] + x[index + jOffset] +
                                                 x[index - kOffset] + x[index + kOffset])) *
                                           cRecip;
                            }
                        }
                    }
                }
            }
        }
        set_bnd(M, N, O, b, x);
    }
}

#elif 0

// red-black solver with convergence check
void lin_solve(int M, int N, int O, int b, float* x, float* x0, float a, float c) {
    constexpr int BLOCK_SIZE = 4;
    const int jOffset = (M + 2);
    const int kOffset = jOffset * (N + 2);
    float tol = 1e-7, max_c, old_x, change;
    int l = 0;

    do {
        max_c = 0.0f;

        // Red points
#pragma omp parallel private(old_x, change)
        {
            float local_max_c = 0.0f;  // Local max_c for each thread
#pragma omp for collapse(3)  // Collapse loops to optimize scheduling
            for (int k = 1; k <= O; k += BLOCK_SIZE) {
                for (int j = 1; j <= N; j += BLOCK_SIZE) {
                    for (int i = 1; i <= M; i += BLOCK_SIZE) {
                        for (int kBlock = k; kBlock < k + BLOCK_SIZE && kBlock <= O; kBlock++) {
                            for (int jBlock = j; jBlock < j + BLOCK_SIZE && jBlock <= N; jBlock++) {
                                for (int iBlock = i; iBlock < i + BLOCK_SIZE && iBlock <= M; iBlock++) {
                                    if ((iBlock + jBlock + kBlock) % 2 == 0) {  // Red points
                                        int index = IX(iBlock, jBlock, kBlock);
                                        old_x = x[index];
                                        x[index] = (x0[index] +
                                                    a * (x[index - 1] + x[index + 1] +
                                                         x[index - jOffset] + x[index + jOffset] +
                                                         x[index - kOffset] + x[index + kOffset])) /
                                                   c;
                                        change = fabs(x[index] - old_x);
                                        if (change > local_max_c) local_max_c = change;
                                    }
                                }
                            }
                        }
                    }
                }
            }
#pragma omp critical
            {
                if (local_max_c > max_c) {
                    max_c = local_max_c;  // Combine local results in a critical section
                }
            }

            // Black points
            local_max_c = 0.0f;  // Local max_c for each thread
#pragma omp for collapse(3)
            for (int k = 1; k <= O; k += BLOCK_SIZE) {
                for (int j = 1; j <= N; j += BLOCK_SIZE) {
                    for (int i = 1; i <= M; i += BLOCK_SIZE) {
                        for (int kBlock = k; kBlock < k + BLOCK_SIZE && kBlock <= O; kBlock++) {
                            for (int jBlock = j; jBlock < j + BLOCK_SIZE && jBlock <= N; jBlock++) {
                                for (int iBlock = i; iBlock < i + BLOCK_SIZE && iBlock <= M; iBlock++) {
                                    if ((iBlock + jBlock + kBlock) % 2 != 0) {  // Black points
                                        int index = IX(iBlock, jBlock, kBlock);
                                        old_x = x[index];
                                        x[index] = (x0[index] +
                                                    a * (x[index - 1] + x[index + 1] +
                                                         x[index - jOffset] + x[index + jOffset] +
                                                         x[index - kOffset] + x[index + kOffset])) /
                                                   c;
                                        change = fabs(x[index] - old_x);
                                        if (change > local_max_c) local_max_c = change;
                                    }
                                }
                            }
                        }
                    }
                }
            }
#pragma omp critical
            {
                if (local_max_c > max_c) {
                    max_c = local_max_c;  // Combine local results in a critical section
                }
            }
        }
        set_bnd(M, N, O, b, x);
    } while (max_c > tol && ++l < 20);
}

#else

//Kernel Z-faces (top and bottom)
__global__ void set_bnd_Z(int M, int N, int O, int b, float *x) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

  if (i <= M && j <= N) {
    x[IX(i, j, 0)] = (b == 3) ? -x[IX(i, j, 1)] : x[IX(i, j, 1)];
    x[IX(i, j, O + 1)] = (b == 3) ? -x[IX(i, j, O)] : x[IX(i, j, O)];
  }
}

//Kernel X-Faces (left and right)
__global__ void set_bnd_X(int M, int N, int O, int b, float *x) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

  if (i <= N && j <= O) {
    x[IX(0, i, j)] = (b == 1) ? -x[IX(1, i, j)] : x[IX(1, i, j)];
    x[IX(M + 1, i, j)] = (b == 1) ? -x[IX(M, i, j)] : x[IX(M, i, j)];
  }
}

//Kernel for Y-Faces (front and back)
__global__ void set_bnd_Y(int M, int N, int O, int b, float *x) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

  if (i <= M && j <= O) {
    x[IX(i, 0, j)] = (b == 2) ? -x[IX(i, 1, j)]: x[IX(i, 1, j)];
    x[IX(i, N + 1, j)] = (b == 2) ? -x[IX(i, N, j)] :  x[IX(i, N, j)];
  }
}

// Kernel for corners
__global__ void set_bnd_corners(int M, int N, int O, float *x) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    x[IX(0, 0, 0)] = 0.33f * (x[IX(1, 0, 0)] + x[IX(0, 1, 0)] + x[IX(0, 0, 1)]);

    x[IX(M + 1, 0, 0)] =
        0.33f * (x[IX(M, 0, 0)] + x[IX(M + 1, 1, 0)] + x[IX(M + 1, 0, 1)]);

    x[IX(0, N + 1, 0)] =
        0.33f * (x[IX(1, N + 1, 0)] + x[IX(0, N, 0)] + x[IX(0, N + 1, 1)]);

    x[IX(M + 1, N + 1, 0)] = 0.33f * (x[IX(M, N + 1, 0)] + x[IX(M + 1, N, 0)] +
                                      x[IX(M + 1, N + 1, 1)]);
  }
}


void set_bnd_cuda(int M, int N, int O, int b, float *x) {
  dim3 blockDim(16, 16);

  //Z-face
  dim3 gridDimZ((M + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y);
  set_bnd_Z<<<gridDimZ, blockDim>>>(M, N, O, b, x);

  //X-face
  dim3 gridDimX((N + blockDim.x - 1) / blockDim.x,
                 (O + blockDim.y - 1) / blockDim.y);
  set_bnd_X<<<gridDimX, blockDim>>>(M, N, O, b, x);

  //Y-Face
  dim3 gridDimY((M + blockDim.x - 1) / blockDim.x,
                 (O + blockDim.y - 1) / blockDim.y);
  set_bnd_Y<<<gridDimY, blockDim>>>(M, N, O, b, x);

  //Corners
  set_bnd_corners<<<1, 1>>>(M, N, O, x);
}

// Kernel for computing the red or black points
__global__ void red_black_kernel(int M, int N, int O, float* x, const float* x0, float a, float c, bool isRed, float* maxChange) {
    // Compute 3D thread ID
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    // Ensure within bounds and check for red/black points
    if (i <= 0 || j <= 0 || k <= 0 || i >= M || j >= N || k >= O || (i + j + k) % 2 != isRed)
        return;

    int idx = IX(i, j, k);
    float old_x = x[idx];
    x[idx] = (x0[idx] +
              a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                   x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                   x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) /
             c;

    // Calculate the change and atomically update maxChange
    float change = fabsf(x[idx] - old_x);
    atomicMax((int *)maxChange, __float_as_int(change));
}

void lin_solve(int M, int N, int O, int b, float* x, float* x0, float a, float c) {
    // Allocate device memory
    float *d_x, *d_x0, *d_maxChange;
    cudaMalloc((void**)&d_x, sizeof(float) * (M + 2) * (N + 2) * (O + 2));
    cudaMalloc((void**)&d_x0, sizeof(float) * (M + 2) * (N + 2) * (O + 2));
    cudaMalloc((void**)&d_maxChange, sizeof(float));

    // Copy data to device
    cudaMemcpy(d_x, x, sizeof(float) * (M + 2) * (N + 2) * (O + 2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x0, x0, sizeof(float) * (M + 2) * (N + 2) * (O + 2), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockDim(8, 8, 8);
    dim3 gridDim((M + 7) / 8, (N + 7) / 8, (O + 7) / 8);

    float tol = 1e-7f, maxChange;
    int l = 0;

    do {
        maxChange = 0.0f;
        cudaMemcpy(d_maxChange, &maxChange, sizeof(float), cudaMemcpyHostToDevice);

        // Launch kernel for red points
        red_black_kernel<<<gridDim, blockDim>>>(M, N, O, d_x, d_x0, a, c, true, d_maxChange);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cout << "CUDA error after kernel launch (red): " << cudaGetErrorString(err) << std::endl;
        }

        // Launch kernel for black points
        red_black_kernel<<<gridDim, blockDim>>>(M, N, O, d_x, d_x0, a, c, false, d_maxChange);
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cout << "CUDA error after kernel launch (black): " << cudaGetErrorString(err) << std::endl;
        }

        set_bnd_cuda(M, N, O, b, d_x);

        // Copy maxChange back to host
        cudaMemcpy(&maxChange, d_maxChange, sizeof(float), cudaMemcpyDeviceToHost);

    } while (maxChange > tol && ++l < 20);

    // Copy results back to host
    cudaMemcpy(x, d_x, sizeof(float) * (M + 2) * (N + 2) * (O + 2), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_x0);
    cudaFree(d_maxChange);
}

#endif

// Diffusion step (uses implicit method)
void diffuse(int M, int N, int O, int b, float* x, float* x0, float diff,
             float dt) {
    int max = MAX(MAX(M, N), O);
    float a = dt * diff * max * max;
    lin_solve(M, N, O, b, x, x0, a, 1 + 6 * a);
}

// Advection step (uses velocity field to move quantities)
void advect(int M, int N, int O, int b, float* d, float* d0, float* u, float* v,
            float* w, float dt) {
    float dtX = dt * M, dtY = dt * N, dtZ = dt * O;
#pragma omp parallel for collapse(3)
    for (int k = 1; k <= O; k++) {
        for (int j = 1; j <= N; j++) {
            for (int i = 1; i <= M; i++) {
                float x = i - dtX * u[IX(i, j, k)];
                float y = j - dtY * v[IX(i, j, k)];
                float z = k - dtZ * w[IX(i, j, k)];

                // Clamp to grid boundaries
                if (x < 0.5f)
                    x = 0.5f;
                if (x > M + 0.5f)
                    x = M + 0.5f;
                if (y < 0.5f)
                    y = 0.5f;
                if (y > N + 0.5f)
                    y = N + 0.5f;
                if (z < 0.5f)
                    z = 0.5f;
                if (z > O + 0.5f)
                    z = O + 0.5f;

                int i0 = (int)x, i1 = i0 + 1;
                int j0 = (int)y, j1 = j0 + 1;
                int k0 = (int)z, k1 = k0 + 1;

                float s1 = x - i0, s0 = 1 - s1;
                float t1 = y - j0, t0 = 1 - t1;
                float u1 = z - k0, u0 = 1 - u1;

                d[IX(i, j, k)] =
                    s0 * (t0 * (u0 * d0[IX(i0, j0, k0)] + u1 * d0[IX(i0, j0, k1)]) +
                          t1 * (u0 * d0[IX(i0, j1, k0)] + u1 * d0[IX(i0, j1, k1)])) +
                    s1 * (t0 * (u0 * d0[IX(i1, j0, k0)] + u1 * d0[IX(i1, j0, k1)]) +
                          t1 * (u0 * d0[IX(i1, j1, k0)] + u1 * d0[IX(i1, j1, k1)]));
            }
        }
    }
    set_bnd(M, N, O, b, d);
}

// Projection step to ensure incompressibility (make the velocity field
// divergence-free)
void project(int M, int N, int O, float* u, float* v, float* w, float* p,
             float* div) {
#pragma omp parallel for collapse(3)
    for (int k = 1; k <= O; k++) {
        for (int j = 1; j <= N; j++) {
            for (int i = 1; i <= M; i++) {
                div[IX(i, j, k)] =
                    -0.5f *
                    (u[IX(i + 1, j, k)] - u[IX(i - 1, j, k)] + v[IX(i, j + 1, k)] -
                     v[IX(i, j - 1, k)] + w[IX(i, j, k + 1)] - w[IX(i, j, k - 1)]) /
                    MAX(M, MAX(N, O));
                p[IX(i, j, k)] = 0;
            }
        }
    }
    set_bnd(M, N, O, 0, div);
    set_bnd(M, N, O, 0, p);
    lin_solve(M, N, O, 0, p, div, 1, 6);
#pragma omp parallel for collapse(3)
    for (int k = 1; k <= O; k++) {
        for (int j = 1; j <= N; j++) {
            for (int i = 1; i <= M; i++) {
                u[IX(i, j, k)] -= 0.5f * (p[IX(i + 1, j, k)] - p[IX(i - 1, j, k)]);
                v[IX(i, j, k)] -= 0.5f * (p[IX(i, j + 1, k)] - p[IX(i, j - 1, k)]);
                w[IX(i, j, k)] -= 0.5f * (p[IX(i, j, k + 1)] - p[IX(i, j, k - 1)]);
            }
        }
    }
    set_bnd(M, N, O, 1, u);

    set_bnd(M, N, O, 2, v);

    set_bnd(M, N, O, 3, w);
}

// Step function for density
void dens_step(int M, int N, int O, float* x, float* x0, float* u, float* v,
               float* w, float diff, float dt) {
    add_source(M, N, O, x, x0, dt);
    SWAP(x0, x);
    diffuse(M, N, O, 0, x, x0, diff, dt);
    SWAP(x0, x);
    advect(M, N, O, 0, x, x0, u, v, w, dt);
}

// Step function for velocity
void vel_step(int M, int N, int O, float* u, float* v, float* w, float* u0,
              float* v0, float* w0, float visc, float dt) {
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