#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <stdio.h>

#define N_WARMUP 5
#define N_ITER 20

#define WARP_SIZE 32
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// FP32 warp reduce sum
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

// FP16 -> FP32 warp reduce sum
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f16_f32(half val) {
  float val_f32 = __half2float(val);
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val_f32 += __shfl_xor_sync(0xffffffff, val_f32, mask);
  }
  return val_f32;
}

// Dot Product f32
// grid(N/256), block(256)
template <const int NUM_THREADS = 256>
__global__ void dot_prod_f32_f32_kernel(float *a, float *b, float *y, int N) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * NUM_THREADS + tid;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];

  float prod = (idx < N) ? a[idx] * b[idx] : 0.0f;
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  prod = warp_reduce_sum_f32<WARP_SIZE>(prod);
  if (lane == 0)
    reduce_smem[warp] = prod;
  __syncthreads();
  prod = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0)
    prod = warp_reduce_sum_f32<NUM_WARPS>(prod);
  if (tid == 0)
    atomicAdd(y, prod);
}

// Dot Product f32x4
// grid(N/256), block(256/4)
template <const int NUM_THREADS = 256 / 4>
__global__ void dot_prod_f32x4_f32_kernel(float *a, float *b, float *y, int N) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * NUM_THREADS + tid) * 4;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];

  float4 reg_a = FLOAT4(a[idx]);
  float4 reg_b = FLOAT4(b[idx]);
  float prod = (idx < N) ? (reg_a.x * reg_b.x + reg_a.y * reg_b.y +
                            reg_a.z * reg_b.z + reg_a.w * reg_b.w)
                         : 0.0f;
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  prod = warp_reduce_sum_f32<WARP_SIZE>(prod);
  if (lane == 0)
    reduce_smem[warp] = prod;
  __syncthreads();
  prod = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0)
    prod = warp_reduce_sum_f32<NUM_WARPS>(prod);
  if (tid == 0)
    atomicAdd(y, prod);
}

// Dot Product f16 -> f32
// grid(N/256), block(256)
template <const int NUM_THREADS = 256>
__global__ void dot_prod_f16_f32_kernel(half *a, half *b, float *y, int N) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * NUM_THREADS + tid;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];

  half prod_f16 = (idx < N) ? __hmul(a[idx], b[idx]) : __float2half(0.0f);
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  float prod = warp_reduce_sum_f16_f32<WARP_SIZE>(prod_f16);
  if (lane == 0)
    reduce_smem[warp] = prod;
  __syncthreads();
  prod = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0)
    prod = warp_reduce_sum_f32<NUM_WARPS>(prod);
  if (tid == 0)
    atomicAdd(y, prod);
}

// Dot Product f16x2 -> f32
// grid(N/256), block(256/2)
template <const int NUM_THREADS = 256 / 2>
__global__ void dot_prod_f16x2_f32_kernel(half *a, half *b, float *y, int N) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * NUM_THREADS + tid) * 2;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];

  half2 reg_a = HALF2(a[idx]);
  half2 reg_b = HALF2(b[idx]);
  half prod_f16 =
      (idx < N) ? __hadd(__hmul(reg_a.x, reg_b.x), __hmul(reg_a.y, reg_b.y))
                : __float2half(0.0f);
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  float prod = warp_reduce_sum_f16_f32<WARP_SIZE>(prod_f16);
  if (lane == 0)
    reduce_smem[warp] = prod;
  __syncthreads();
  prod = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0)
    prod = warp_reduce_sum_f32<NUM_WARPS>(prod);
  if (tid == 0)
    atomicAdd(y, prod);
}

// Dot Product f16x8 pack -> f32
// grid(N/256), block(256/8)
template <const int NUM_THREADS = 256 / 8>
__global__ void dot_prod_f16x8_pack_f32_kernel(half *a, half *b, float *y, int N) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * NUM_THREADS + tid) * 8;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];

  half pack_a[8], pack_b[8];
  LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);
  LDST128BITS(pack_b[0]) = LDST128BITS(b[idx]);
  const half z = __float2half(0.0f);

  half prod_f16 = z;
#pragma unroll
  for (int i = 0; i < 8; i += 2) {
    half2 v = __hmul2(HALF2(pack_a[i]), HALF2(pack_b[i]));
    prod_f16 += (((idx + i) < N) ? (v.x + v.y) : z);
  }

  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  float prod = warp_reduce_sum_f16_f32<WARP_SIZE>(prod_f16);
  if (lane == 0)
    reduce_smem[warp] = prod;
  __syncthreads();
  prod = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0)
    prod = warp_reduce_sum_f32<NUM_WARPS>(prod);
  if (tid == 0)
    atomicAdd(y, prod);
}

// Compile-time kernel selection: pass e.g. -DDOT_PROD_F32 to nvcc to profile only that variant.
// If none are defined, all variants are enabled.
#if !defined(DOT_PROD_F32) && !defined(DOT_PROD_F32X4) && !defined(DOT_PROD_F16) && \
    !defined(DOT_PROD_F16X2) && !defined(DOT_PROD_F16X8_PACK)
#define DOT_PROD_F32
#define DOT_PROD_F32X4
#define DOT_PROD_F16
#define DOT_PROD_F16X2
#define DOT_PROD_F16X8_PACK
#endif

int main() {
  const int N = 1 << 20;

  // Output buffer: all variants produce a float scalar — always allocated.
  float *d_out;
  cudaMalloc(&d_out, sizeof(float));

  // F32 input buffers (used by DOT_PROD_F32 and DOT_PROD_F32X4)
#if defined(DOT_PROD_F32) || defined(DOT_PROD_F32X4)
  float *d_a, *d_b;
  cudaMalloc(&d_a, N * sizeof(float));
  cudaMalloc(&d_b, N * sizeof(float));
  cudaMemset(d_a, 0, N * sizeof(float));
  cudaMemset(d_b, 0, N * sizeof(float));
#endif

  // F16 input buffers (used by DOT_PROD_F16, DOT_PROD_F16X2, DOT_PROD_F16X8_PACK)
#if defined(DOT_PROD_F16) || defined(DOT_PROD_F16X2) || defined(DOT_PROD_F16X8_PACK)
  half *d_a_f16, *d_b_f16;
  cudaMalloc(&d_a_f16, N * sizeof(half));
  cudaMalloc(&d_b_f16, N * sizeof(half));
  cudaMemset(d_a_f16, 0, N * sizeof(half));
  cudaMemset(d_b_f16, 0, N * sizeof(half));
#endif

#ifdef DOT_PROD_F32
  // --- dot_prod_f32_f32_kernel ---
  {
    const int block = 256;
    const int grid = (N + block - 1) / block;
    cudaMemset(d_out, 0, sizeof(float));
    for (int i = 0; i < N_WARMUP; i++)
      dot_prod_f32_f32_kernel<256><<<grid, block>>>(d_a, d_b, d_out, N);
    cudaDeviceSynchronize();
    nvtxRangePush("dot_prod_f32_f32_kernel");
    for (int i = 0; i < N_ITER; i++) {
      cudaMemset(d_out, 0, sizeof(float));
      dot_prod_f32_f32_kernel<256><<<grid, block>>>(d_a, d_b, d_out, N);
    }
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

#ifdef DOT_PROD_F32X4
  // --- dot_prod_f32x4_f32_kernel ---
  {
    const int block = 256 / 4;
    const int grid = (N / 4 + block - 1) / block;
    cudaMemset(d_out, 0, sizeof(float));
    for (int i = 0; i < N_WARMUP; i++)
      dot_prod_f32x4_f32_kernel<256 / 4><<<grid, block>>>(d_a, d_b, d_out, N);
    cudaDeviceSynchronize();
    nvtxRangePush("dot_prod_f32x4_f32_kernel");
    for (int i = 0; i < N_ITER; i++) {
      cudaMemset(d_out, 0, sizeof(float));
      dot_prod_f32x4_f32_kernel<256 / 4><<<grid, block>>>(d_a, d_b, d_out, N);
    }
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

#ifdef DOT_PROD_F16
  // --- dot_prod_f16_f32_kernel ---
  {
    const int block = 256;
    const int grid = (N + block - 1) / block;
    cudaMemset(d_out, 0, sizeof(float));
    for (int i = 0; i < N_WARMUP; i++)
      dot_prod_f16_f32_kernel<256><<<grid, block>>>(d_a_f16, d_b_f16, d_out, N);
    cudaDeviceSynchronize();
    nvtxRangePush("dot_prod_f16_f32_kernel");
    for (int i = 0; i < N_ITER; i++) {
      cudaMemset(d_out, 0, sizeof(float));
      dot_prod_f16_f32_kernel<256><<<grid, block>>>(d_a_f16, d_b_f16, d_out, N);
    }
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

#ifdef DOT_PROD_F16X2
  // --- dot_prod_f16x2_f32_kernel ---
  {
    const int block = 256 / 2;
    const int grid = (N / 2 + block - 1) / block;
    cudaMemset(d_out, 0, sizeof(float));
    for (int i = 0; i < N_WARMUP; i++)
      dot_prod_f16x2_f32_kernel<256 / 2><<<grid, block>>>(d_a_f16, d_b_f16, d_out, N);
    cudaDeviceSynchronize();
    nvtxRangePush("dot_prod_f16x2_f32_kernel");
    for (int i = 0; i < N_ITER; i++) {
      cudaMemset(d_out, 0, sizeof(float));
      dot_prod_f16x2_f32_kernel<256 / 2><<<grid, block>>>(d_a_f16, d_b_f16, d_out, N);
    }
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

#ifdef DOT_PROD_F16X8_PACK
  // --- dot_prod_f16x8_pack_f32_kernel ---
  {
    const int block = 256 / 8;
    const int grid = (N / 8 + block - 1) / block;
    cudaMemset(d_out, 0, sizeof(float));
    for (int i = 0; i < N_WARMUP; i++)
      dot_prod_f16x8_pack_f32_kernel<256 / 8><<<grid, block>>>(d_a_f16, d_b_f16, d_out, N);
    cudaDeviceSynchronize();
    nvtxRangePush("dot_prod_f16x8_pack_f32_kernel");
    for (int i = 0; i < N_ITER; i++) {
      cudaMemset(d_out, 0, sizeof(float));
      dot_prod_f16x8_pack_f32_kernel<256 / 8><<<grid, block>>>(d_a_f16, d_b_f16, d_out, N);
    }
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

#if defined(DOT_PROD_F32) || defined(DOT_PROD_F32X4)
  cudaFree(d_a);
  cudaFree(d_b);
#endif
  cudaFree(d_out);
#if defined(DOT_PROD_F16) || defined(DOT_PROD_F16X2) || defined(DOT_PROD_F16X8_PACK)
  cudaFree(d_a_f16);
  cudaFree(d_b_f16);
#endif
  printf("Done.\n");
  return 0;
}
