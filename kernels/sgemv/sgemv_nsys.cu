#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <stdio.h>

#define N_WARMUP 5
#define N_ITER 20

#define WARP_SIZE 32
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

// FP32 warp reduce helper
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

// SGEMV: Warp SGEMV K32
// grid(M/4), block(32,4) blockDim.x=32=K, blockDim.y=4
// a: MxK, x: Kx1, y: Mx1, compute: y = a * x
__global__ void sgemv_k32_f32_kernel(float *a, float *x, float *y, int M,
                                     int K) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int lane = tx % WARP_SIZE;
  int m = bx * blockDim.y + ty;
  if (m < M) {
    float sum = 0.0f;
    int NUM_WARPS = (K + WARP_SIZE - 1) / WARP_SIZE;
#pragma unroll
    for (int w = 0; w < NUM_WARPS; ++w) {
      int k = w * WARP_SIZE + lane;
      sum += a[m * K + k] * x[k];
    }
    sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
    if (lane == 0)
      y[m] = sum;
  }
}

// SGEMV: Warp SGEMV K128 + Vec4
// grid(M/4), block(32,4) blockDim.x=32, blockDim.y=4
// a: MxK, x: Kx1, y: Mx1, compute: y = a * x
__global__ void sgemv_k128_f32x4_kernel(float *a, float *x, float *y, int M,
                                        int K) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int lane = tx % WARP_SIZE;
  int m = blockDim.y * bx + ty;

  if (m < M) {
    float sum = 0.0f;
    int NUM_WARPS = (((K + WARP_SIZE - 1) / WARP_SIZE) + 4 - 1) / 4;
#pragma unroll
    for (int w = 0; w < NUM_WARPS; ++w) {
      int k = (w * WARP_SIZE + lane) * 4;
      float4 reg_x = FLOAT4(x[k]);
      float4 reg_a = FLOAT4(a[m * K + k]);
      sum += (reg_a.x * reg_x.x + reg_a.y * reg_x.y + reg_a.z * reg_x.z +
              reg_a.w * reg_x.w);
    }
    sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
    if (lane == 0)
      y[m] = sum;
  }
}

// SGEMV: Warp SGEMV K16
// NUM_THREADS=128, NUM_WARPS=4, ROW_PER_WARP=2
// grid(M/NUM_ROWS), block(32, NUM_WARPS)
// a: MxK, x: Kx1, y: Mx1, compute: y = a * x
template <const int ROW_PER_WARP = 2>
__global__ void sgemv_k16_f32_kernel(float *A, float *x, float *y, int M,
                                     int K) {
  constexpr int K_WARP_SIZE = (WARP_SIZE + ROW_PER_WARP - 1) / ROW_PER_WARP;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int lane = tx % WARP_SIZE;
  int k = lane % K_WARP_SIZE;
  int m = (blockDim.y * bx + ty) * ROW_PER_WARP + lane / K_WARP_SIZE;
  if (m < M) {
    float sum = A[m * K + k] * x[k];
    sum = warp_reduce_sum_f32<K_WARP_SIZE>(sum);
    if (k == 0)
      y[m] = sum;
  }
}

#define RUN_KERNEL(label, launch_expr)                      \
  do {                                                      \
    for (int i = 0; i < N_WARMUP; i++) { launch_expr; }    \
    cudaDeviceSynchronize();                                \
    nvtxRangePush(label);                                   \
    for (int i = 0; i < N_ITER; i++) { launch_expr; }      \
    cudaDeviceSynchronize();                                \
    nvtxRangePop();                                         \
  } while (0)

// ---------------------------------------------------------------------------
// Default: if no macro is defined, enable all kernels
// ---------------------------------------------------------------------------
#if !defined(SGEMV_K32) && !defined(SGEMV_K128_F32X4) && !defined(SGEMV_K16)
#define SGEMV_K32
#define SGEMV_K128_F32X4
#define SGEMV_K16
#endif

int main() {
  const int M = 2048;
  const int K = 2048;

  // --- main float buffers (k32 and k128 kernels) ---
#if defined(SGEMV_K32) || defined(SGEMV_K128_F32X4)
  float *d_a, *d_x, *d_y;
  cudaMalloc(&d_a, (size_t)M * K * sizeof(float));
  cudaMalloc(&d_x, K * sizeof(float));
  cudaMalloc(&d_y, M * sizeof(float));
  cudaMemset(d_a, 0, (size_t)M * K * sizeof(float));
  cudaMemset(d_x, 0, K * sizeof(float));
  cudaMemset(d_y, 0, M * sizeof(float));
#endif

  // sgemv_k32_f32_kernel: block(32,4), grid(M/4)
#ifdef SGEMV_K32
  {
    dim3 block(32, 4);
    dim3 grid((M + 4 - 1) / 4);
    RUN_KERNEL("sgemv_k32_f32_kernel",
      sgemv_k32_f32_kernel<<<grid, block>>>(d_a, d_x, d_y, M, K));
  }
#endif

  // sgemv_k128_f32x4_kernel: block(32,4), grid(M/4)
#ifdef SGEMV_K128_F32X4
  {
    dim3 block(32, 4);
    dim3 grid((M + 4 - 1) / 4);
    RUN_KERNEL("sgemv_k128_f32x4_kernel",
      sgemv_k128_f32x4_kernel<<<grid, block>>>(d_a, d_x, d_y, M, K));
  }
#endif

  // sgemv_k16_f32_kernel: K must be 16; use K=16 slice for this kernel.
  // Allocate a separate smaller matrix to satisfy the K=16 requirement.
#ifdef SGEMV_K16
  {
    const int K16 = 16;
#if !defined(SGEMV_K32) && !defined(SGEMV_K128_F32X4)
    float *d_y;
    cudaMalloc(&d_y, M * sizeof(float));
    cudaMemset(d_y, 0, M * sizeof(float));
#endif
    float *d_a16, *d_x16;
    cudaMalloc(&d_a16, (size_t)M * K16 * sizeof(float));
    cudaMalloc(&d_x16, K16 * sizeof(float));
    cudaMemset(d_a16, 0, (size_t)M * K16 * sizeof(float));
    cudaMemset(d_x16, 0, K16 * sizeof(float));

    constexpr int ROW_PER_WARP = 2;
    constexpr int NUM_WARPS    = 4; // 128 / 32
    constexpr int NUM_ROWS     = NUM_WARPS * ROW_PER_WARP; // 8
    dim3 block(32, NUM_WARPS);
    dim3 grid((M + NUM_ROWS - 1) / NUM_ROWS);
    RUN_KERNEL("sgemv_k16_f32_kernel",
      sgemv_k16_f32_kernel<ROW_PER_WARP><<<grid, block>>>(d_a16, d_x16, d_y, M, K16));

    cudaFree(d_a16);
    cudaFree(d_x16);
#if !defined(SGEMV_K32) && !defined(SGEMV_K128_F32X4)
    cudaFree(d_y);
#endif
  }
#endif

#if defined(SGEMV_K32) || defined(SGEMV_K128_F32X4)
  cudaFree(d_a);
  cudaFree(d_x);
  cudaFree(d_y);
#endif

  printf("Done.\n");
  return 0;
}
