#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <stdio.h>

#define N_WARMUP 5
#define N_ITER 20

#define WARP_SIZE 32
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])

// FP16 warp reduce helper
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_f16(half val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

// HGEMV: Warp HGEMV K32
// grid(M/4), block(32,4)
// a: MxK, x: Kx1, y: Mx1, compute: y = a * x
__global__ void hgemv_k32_f16_kernel(half *a, half *x, half *y, int M, int K) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int lane = tx % WARP_SIZE;
  int m = bx * blockDim.y + ty;
  if (m < M) {
    half sum = 0.0f;
    int NUM_WARPS = (K + WARP_SIZE - 1) / WARP_SIZE;
#pragma unroll
    for (int w = 0; w < NUM_WARPS; ++w) {
      int k = w * WARP_SIZE + lane;
      sum += a[m * K + k] * x[k];
    }
    sum = warp_reduce_sum_f16<WARP_SIZE>(sum);
    if (lane == 0)
      y[m] = sum;
  }
}

// HGEMV: Warp HGEMV K128 + half2x2
// grid(M/4), block(32,4)
// a: MxK, x: Kx1, y: Mx1, compute: y = a * x
__global__ void hgemv_k128_f16x4_kernel(half *a, half *x, half *y, int M,
                                        int K) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int lane = tx % WARP_SIZE;
  int m = blockDim.y * bx + ty;

  if (m < M) {
    half sum = 0.0f;
    int NUM_WARPS = (((K + WARP_SIZE - 1) / WARP_SIZE) + 4 - 1) / 4;
#pragma unroll
    for (int w = 0; w < NUM_WARPS; ++w) {
      int k = (w * WARP_SIZE + lane) * 4;
      half2 reg_x_0 = HALF2(x[k + 0]);
      half2 reg_x_1 = HALF2(x[k + 2]);
      half2 reg_a_0 = HALF2(a[m * K + k + 0]);
      half2 reg_a_1 = HALF2(a[m * K + k + 2]);
      sum += (reg_x_0.x * reg_a_0.x + reg_x_0.y * reg_a_0.y +
              reg_x_1.x * reg_a_1.x + reg_x_1.y * reg_a_1.y);
    }
    sum = warp_reduce_sum_f16<WARP_SIZE>(sum);
    if (lane == 0)
      y[m] = sum;
  }
}

// HGEMV: Warp HGEMV K16
// NUM_THREADS=128, NUM_WARPS=4, ROW_PER_WARP=2
// grid(M/NUM_ROWS), block(32, NUM_WARPS)
// a: MxK, x: Kx1, y: Mx1, compute: y = a * x
template <const int ROW_PER_WARP = 2>
__global__ void hgemv_k16_f16_kernel(half *A, half *x, half *y, int M, int K) {
  constexpr int K_WARP_SIZE = (WARP_SIZE + ROW_PER_WARP - 1) / ROW_PER_WARP;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int lane = tx % WARP_SIZE;
  int k = lane % K_WARP_SIZE;
  int m = (blockDim.y * bx + ty) * ROW_PER_WARP + lane / K_WARP_SIZE;
  if (m < M) {
    half sum = A[m * K + k] * x[k];
    sum = warp_reduce_sum_f16<K_WARP_SIZE>(sum);
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
#if !defined(HGEMV_K32) && !defined(HGEMV_K128_F16X4) && !defined(HGEMV_K16)
#define HGEMV_K32
#define HGEMV_K128_F16X4
#define HGEMV_K16
#endif

int main() {
  const int M = 2048;
  const int K = 2048;

  // --- main half buffers (k32 and k128 kernels) ---
#if defined(HGEMV_K32) || defined(HGEMV_K128_F16X4)
  half *d_a, *d_x, *d_y;
  cudaMalloc(&d_a, (size_t)M * K * sizeof(half));
  cudaMalloc(&d_x, K * sizeof(half));
  cudaMalloc(&d_y, M * sizeof(half));
  cudaMemset(d_a, 0, (size_t)M * K * sizeof(half));
  cudaMemset(d_x, 0, K * sizeof(half));
  cudaMemset(d_y, 0, M * sizeof(half));
#endif

  // hgemv_k32_f16_kernel: block(32,4), grid(M/4)
#ifdef HGEMV_K32
  {
    dim3 block(32, 4);
    dim3 grid((M + 4 - 1) / 4);
    RUN_KERNEL("hgemv_k32_f16_kernel",
      hgemv_k32_f16_kernel<<<grid, block>>>(d_a, d_x, d_y, M, K));
  }
#endif

  // hgemv_k128_f16x4_kernel: block(32,4), grid(M/4)
#ifdef HGEMV_K128_F16X4
  {
    dim3 block(32, 4);
    dim3 grid((M + 4 - 1) / 4);
    RUN_KERNEL("hgemv_k128_f16x4_kernel",
      hgemv_k128_f16x4_kernel<<<grid, block>>>(d_a, d_x, d_y, M, K));
  }
#endif

  // hgemv_k16_f16_kernel: K must be 16; allocate a separate K=16 matrix.
#ifdef HGEMV_K16
  {
    const int K16 = 16;
#if !defined(HGEMV_K32) && !defined(HGEMV_K128_F16X4)
    half *d_y;
    cudaMalloc(&d_y, M * sizeof(half));
    cudaMemset(d_y, 0, M * sizeof(half));
#endif
    half *d_a16, *d_x16;
    cudaMalloc(&d_a16, (size_t)M * K16 * sizeof(half));
    cudaMalloc(&d_x16, K16 * sizeof(half));
    cudaMemset(d_a16, 0, (size_t)M * K16 * sizeof(half));
    cudaMemset(d_x16, 0, K16 * sizeof(half));

    constexpr int ROW_PER_WARP = 2;
    constexpr int NUM_WARPS    = 4; // 128 / 32
    constexpr int NUM_ROWS     = NUM_WARPS * ROW_PER_WARP; // 8
    dim3 block(32, NUM_WARPS);
    dim3 grid((M + NUM_ROWS - 1) / NUM_ROWS);
    RUN_KERNEL("hgemv_k16_f16_kernel",
      hgemv_k16_f16_kernel<ROW_PER_WARP><<<grid, block>>>(d_a16, d_x16, d_y, M, K16));

    cudaFree(d_a16);
    cudaFree(d_x16);
#if !defined(HGEMV_K32) && !defined(HGEMV_K128_F16X4)
    cudaFree(d_y);
#endif
  }
#endif

#if defined(HGEMV_K32) || defined(HGEMV_K128_F16X4)
  cudaFree(d_a);
  cudaFree(d_x);
  cudaFree(d_y);
#endif

  printf("Done.\n");
  return 0;
}
