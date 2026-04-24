#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <stdio.h>
#include <stdlib.h>

#define N_WARMUP 5
#define N_ITER 20

#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])

// Histogram
// grid(N/256), block(256)
// a: Nx1, y: count histogram, a >= 1
__global__ void histogram_i32_kernel(int *a, int *y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    atomicAdd(&(y[a[idx]]), 1);
}

// Histogram + Vec4
// grid(N/256), block(256/4)
// a: Nx1, y: count histogram, a >= 1
__global__ void histogram_i32x4_kernel(int *a, int *y, int N) {
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    int4 reg_a = INT4(a[idx]);
    atomicAdd(&(y[reg_a.x]), 1);
    atomicAdd(&(y[reg_a.y]), 1);
    atomicAdd(&(y[reg_a.z]), 1);
    atomicAdd(&(y[reg_a.w]), 1);
  }
}

// Compile-time kernel selection: pass e.g. -DHISTOGRAM_I32 to nvcc to profile only that variant.
// If none are defined, all variants are enabled.
#if !defined(HISTOGRAM_I32) && !defined(HISTOGRAM_I32X4)
#define HISTOGRAM_I32
#define HISTOGRAM_I32X4
#endif

int main() {
  const int N    = 1 << 20;
  // Use a fixed number of bins; values are initialized via pattern below.
  // We use 256 bins and fill input with values in [0, 255].
  const int BINS = 256;

  int *d_input, *d_hist;
  cudaMalloc(&d_input, N * sizeof(int));
  cudaMalloc(&d_hist,  BINS * sizeof(int));

  // Fill input with values in [0, 255] so atomicAdd targets are valid
  int *h_input = (int *)malloc(N * sizeof(int));
  for (int i = 0; i < N; i++) h_input[i] = i % BINS;
  cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);
  free(h_input);

#ifdef HISTOGRAM_I32
  // --- histogram_i32_kernel: block(256), grid(N/256) ---
  {
    const int block = 256;
    const int grid  = (N + block - 1) / block;
    cudaMemset(d_hist, 0, BINS * sizeof(int));
    for (int i = 0; i < N_WARMUP; i++)
      histogram_i32_kernel<<<grid, block>>>(d_input, d_hist, N);
    cudaDeviceSynchronize();
    nvtxRangePush("histogram_i32_kernel");
    for (int i = 0; i < N_ITER; i++) {
      cudaMemset(d_hist, 0, BINS * sizeof(int));
      histogram_i32_kernel<<<grid, block>>>(d_input, d_hist, N);
    }
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

#ifdef HISTOGRAM_I32X4
  // --- histogram_i32x4_kernel: block(256/4), grid(N/256) ---
  {
    const int block = 256 / 4;
    const int grid  = (N + 256 - 1) / 256;
    cudaMemset(d_hist, 0, BINS * sizeof(int));
    for (int i = 0; i < N_WARMUP; i++)
      histogram_i32x4_kernel<<<grid, block>>>(d_input, d_hist, N);
    cudaDeviceSynchronize();
    nvtxRangePush("histogram_i32x4_kernel");
    for (int i = 0; i < N_ITER; i++) {
      cudaMemset(d_hist, 0, BINS * sizeof(int));
      histogram_i32x4_kernel<<<grid, block>>>(d_input, d_hist, N);
    }
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

  cudaFree(d_input);
  cudaFree(d_hist);
  printf("Done.\n");
  return 0;
}
