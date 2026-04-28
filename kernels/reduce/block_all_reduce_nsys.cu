#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <stdint.h>
#include <stdio.h>

#define N_WARMUP 5
#define N_ITER 20

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// ---------------------------------------------------------------------------
// FP32 device helpers
// ---------------------------------------------------------------------------
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

// ---------------------------------------------------------------------------
// FP16 device helpers
// ---------------------------------------------------------------------------
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_f16_f16(half val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val = __hadd(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f16_f32(half val) {
  float val_f32 = __half2float(val);
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val_f32 += __shfl_xor_sync(0xffffffff, val_f32, mask);
  }
  return val_f32;
}

// ---------------------------------------------------------------------------
// BF16 device helpers
// ---------------------------------------------------------------------------
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ __nv_bfloat16
warp_reduce_sum_bf16_bf16(__nv_bfloat16 val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val = __hadd(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_bf16_f32(__nv_bfloat16 val) {
  float val_f32 = __bfloat162float(val);
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val_f32 += __shfl_xor_sync(0xffffffff, val_f32, mask);
  }
  return val_f32;
}

// ---------------------------------------------------------------------------
// FP8 device helpers
// ---------------------------------------------------------------------------
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half
warp_reduce_sum_fp8_e4m3_f16(__nv_fp8_storage_t val) {
  half val_f16 = __nv_cvt_fp8_to_halfraw(val, __NV_E4M3);
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val_f16 = __hadd(val_f16, __shfl_xor_sync(0xffffffff, val_f16, mask));
  }
  return val_f16;
}

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half
warp_reduce_sum_fp8_e5m2_f16(__nv_fp8_storage_t val) {
  half val_f16 = __nv_cvt_fp8_to_halfraw(val, __NV_E5M2);
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val_f16 = __hadd(val_f16, __shfl_xor_sync(0xffffffff, val_f16, mask));
  }
  return val_f16;
}

// ---------------------------------------------------------------------------
// INT8 device helpers
// ---------------------------------------------------------------------------
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ int32_t warp_reduce_sum_i8_i32(int8_t val) {
  int32_t val_i32 = static_cast<int32_t>(val);
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val_i32 += __shfl_xor_sync(0xffffffff, val_i32, mask);
  }
  return val_i32;
}

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ int32_t warp_reduce_sum_i32_i32(int32_t val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// block_all_reduce_sum_f32_f32_kernel
template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_f32_f32_kernel(float *a, float *y, int N) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * NUM_THREADS + tid;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];
  float sum = (idx < N) ? a[idx] : 0.0f;
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
  if (lane == 0)
    reduce_smem[warp] = sum;
  __syncthreads();
  sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0)
    sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
  if (tid == 0)
    atomicAdd(y, sum);
}

// block_all_reduce_sum_f32x4_f32_kernel
template <const int NUM_THREADS = 256 / 4>
__global__ void block_all_reduce_sum_f32x4_f32_kernel(float *a, float *y,
                                                      int N) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * NUM_THREADS + tid) * 4;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];
  float4 reg_a = FLOAT4(a[idx]);
  float sum = (idx < N) ? (reg_a.x + reg_a.y + reg_a.z + reg_a.w) : 0.0f;
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
  if (lane == 0)
    reduce_smem[warp] = sum;
  __syncthreads();
  sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0)
    sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
  if (tid == 0)
    atomicAdd(y, sum);
}

// block_all_reduce_sum_f16_f16_kernel
template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_f16_f16_kernel(half *a, float *y, int N) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * NUM_THREADS + tid;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];
  half sum_f16 = (idx < N) ? a[idx] : __float2half(0.0f);
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  sum_f16 = warp_reduce_sum_f16_f16<WARP_SIZE>(sum_f16);
  if (lane == 0)
    reduce_smem[warp] = __half2float(sum_f16);
  __syncthreads();
  float sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0)
    sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
  if (tid == 0)
    atomicAdd(y, sum);
}

// block_all_reduce_sum_f16_f32_kernel
template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_f16_f32_kernel(half *a, float *y, int N) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * NUM_THREADS + tid;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];
  half sum_f16 = (idx < N) ? a[idx] : __float2half(0.0f);
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  float sum_f32 = warp_reduce_sum_f16_f32<WARP_SIZE>(sum_f16);
  if (lane == 0)
    reduce_smem[warp] = sum_f32;
  __syncthreads();
  float sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0)
    sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
  if (tid == 0)
    atomicAdd(y, sum);
}

// block_all_reduce_sum_f16x2_f32_kernel
template <const int NUM_THREADS = 256 / 2>
__global__ void block_all_reduce_sum_f16x2_f32_kernel(half *a, float *y,
                                                      int N) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * NUM_THREADS + tid) * 2;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];
  half2 reg_a = HALF2(a[idx]);
  half sum_f16 = (idx < N) ? __hadd(reg_a.x, reg_a.y) : __float2half(0.0f);
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  float sum_f32 = warp_reduce_sum_f16_f32<WARP_SIZE>(sum_f16);
  if (lane == 0)
    reduce_smem[warp] = sum_f32;
  __syncthreads();
  float sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0)
    sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
  if (tid == 0)
    atomicAdd(y, sum);
}

// block_all_reduce_sum_f16x2_f16_kernel
template <const int NUM_THREADS = 256 / 2>
__global__ void block_all_reduce_sum_f16x2_f16_kernel(half *a, float *y,
                                                      int N) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * NUM_THREADS + tid) * 2;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];
  half2 reg_a = HALF2(a[idx]);
  half sum_f16 = (idx < N) ? __hadd(reg_a.x, reg_a.y) : __float2half(0.0f);
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  sum_f16 = warp_reduce_sum_f16_f16<WARP_SIZE>(sum_f16);
  if (lane == 0)
    reduce_smem[warp] = __half2float(sum_f16);
  __syncthreads();
  float sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0)
    sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
  if (tid == 0)
    atomicAdd(y, sum);
}

// block_all_reduce_sum_f16x8_pack_f16_kernel
template <const int NUM_THREADS = 256 / 8>
__global__ void block_all_reduce_sum_f16x8_pack_f16_kernel(half *a, float *y,
                                                           int N) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * NUM_THREADS + tid) * 8;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];
  half pack_a[8];
  LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);
  const half z = __float2half(0.0f);
  half sum_f16 = z;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    sum_f16 += (((idx + i) < N) ? pack_a[i] : z);
  }
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  sum_f16 = warp_reduce_sum_f16_f16<WARP_SIZE>(sum_f16);
  if (lane == 0)
    reduce_smem[warp] = __half2float(sum_f16);
  __syncthreads();
  float sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0)
    sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
  if (tid == 0)
    atomicAdd(y, sum);
}

// block_all_reduce_sum_f16x8_pack_f32_kernel
template <const int NUM_THREADS = 256 / 8>
__global__ void block_all_reduce_sum_f16x8_pack_f32_kernel(half *a, float *y,
                                                           int N) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * NUM_THREADS + tid) * 8;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];
  half pack_a[8];
  LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);
  float sum_f32 = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    sum_f32 += (((idx + i) < N) ? __half2float(pack_a[i]) : 0.0f);
  }
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  sum_f32 = warp_reduce_sum_f32<WARP_SIZE>(sum_f32);
  if (lane == 0)
    reduce_smem[warp] = sum_f32;
  __syncthreads();
  float sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0)
    sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
  if (tid == 0)
    atomicAdd(y, sum);
}

// block_all_reduce_sum_bf16_bf16_kernel
template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_bf16_bf16_kernel(__nv_bfloat16 *a,
                                                      float *y, int N) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * NUM_THREADS + tid;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ __nv_bfloat16 reduce_smem[NUM_WARPS];
  __nv_bfloat16 sum_bf16 = (idx < N) ? a[idx] : __float2bfloat16(0.0f);
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  sum_bf16 = warp_reduce_sum_bf16_bf16<WARP_SIZE>(sum_bf16);
  if (lane == 0)
    reduce_smem[warp] = sum_bf16;
  __syncthreads();
  __nv_bfloat16 sum =
      (lane < NUM_WARPS) ? reduce_smem[lane] : __float2bfloat16(0.0f);
  if (warp == 0)
    sum = warp_reduce_sum_bf16_bf16<NUM_WARPS>(sum);
  if (tid == 0)
    atomicAdd(y, __bfloat162float(sum));
}

// block_all_reduce_sum_bf16_f32_kernel
template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_bf16_f32_kernel(__nv_bfloat16 *a, float *y,
                                                     int N) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * NUM_THREADS + tid;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];
  __nv_bfloat16 sum_bf16 = (idx < N) ? a[idx] : __float2bfloat16(0.0f);
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  float sum_f32 = warp_reduce_sum_bf16_f32<WARP_SIZE>(sum_bf16);
  if (lane == 0)
    reduce_smem[warp] = sum_f32;
  __syncthreads();
  float sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0)
    sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
  if (tid == 0)
    atomicAdd(y, sum);
}

// block_all_reduce_sum_bf16x2_bf16_kernel
template <const int NUM_THREADS = 256 / 2>
__global__ void block_all_reduce_sum_bf16x2_bf16_kernel(__nv_bfloat16 *a,
                                                        float *y, int N) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * NUM_THREADS + tid) * 2;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ __nv_bfloat16 reduce_smem[NUM_WARPS];
  __nv_bfloat162 reg_a = BFLOAT2(a[idx]);
  __nv_bfloat16 sum_bf16 =
      (idx < N) ? __hadd(reg_a.x, reg_a.y) : __float2bfloat16(0.0f);
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  sum_bf16 = warp_reduce_sum_bf16_bf16<WARP_SIZE>(sum_bf16);
  if (lane == 0)
    reduce_smem[warp] = sum_bf16;
  __syncthreads();
  __nv_bfloat16 sum =
      (lane < NUM_WARPS) ? reduce_smem[lane] : __float2bfloat16(0.0f);
  if (warp == 0)
    sum = warp_reduce_sum_bf16_bf16<NUM_WARPS>(sum);
  if (tid == 0)
    atomicAdd(y, __bfloat162float(sum));
}

// block_all_reduce_sum_bf16x2_f32_kernel
template <const int NUM_THREADS = 256 / 2>
__global__ void block_all_reduce_sum_bf16x2_f32_kernel(__nv_bfloat16 *a,
                                                       float *y, int N) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * NUM_THREADS + tid) * 2;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];
  __nv_bfloat162 reg_a = BFLOAT2(a[idx]);
  __nv_bfloat16 sum_bf16 =
      (idx < N) ? __hadd(reg_a.x, reg_a.y) : __float2bfloat16(0.0f);
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  float sum_f32 = warp_reduce_sum_bf16_f32<WARP_SIZE>(sum_bf16);
  if (lane == 0)
    reduce_smem[warp] = sum_f32;
  __syncthreads();
  float sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0)
    sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
  if (tid == 0)
    atomicAdd(y, sum);
}

// block_all_reduce_sum_bf16x8_pack_bf16_kernel
template <const int NUM_THREADS = 256 / 8>
__global__ void block_all_reduce_sum_bf16x8_pack_bf16_kernel(__nv_bfloat16 *a,
                                                             float *y, int N) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * NUM_THREADS + tid) * 8;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ __nv_bfloat16 reduce_smem[NUM_WARPS];
  __nv_bfloat16 pack_a[8];
  LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);
  const __nv_bfloat16 z = __float2bfloat16(0.0f);
  __nv_bfloat16 sum_bf16 = z;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    sum_bf16 += (((idx + i) < N) ? pack_a[i] : z);
  }
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  sum_bf16 = warp_reduce_sum_bf16_bf16<WARP_SIZE>(sum_bf16);
  if (lane == 0)
    reduce_smem[warp] = sum_bf16;
  __syncthreads();
  __nv_bfloat16 sum = (lane < NUM_WARPS) ? reduce_smem[lane] : z;
  if (warp == 0)
    sum = warp_reduce_sum_bf16_bf16<NUM_WARPS>(sum);
  if (tid == 0)
    atomicAdd(y, __bfloat162float(sum));
}

// block_all_reduce_sum_bf16x8_pack_f32_kernel
template <const int NUM_THREADS = 256 / 8>
__global__ void block_all_reduce_sum_bf16x8_pack_f32_kernel(__nv_bfloat16 *a,
                                                            float *y, int N) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * NUM_THREADS + tid) * 8;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];
  __nv_bfloat16 pack_a[8];
  LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);
  const __nv_bfloat16 z = __float2bfloat16(0.0f);
  __nv_bfloat16 sum_bf16 = z;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    sum_bf16 += (((idx + i) < N) ? pack_a[i] : z);
  }
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  float sum_f32 = warp_reduce_sum_bf16_f32<WARP_SIZE>(sum_bf16);
  if (lane == 0)
    reduce_smem[warp] = sum_f32;
  __syncthreads();
  float sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0)
    sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
  if (tid == 0)
    atomicAdd(y, sum);
}

// block_all_reduce_sum_fp8_e4m3_f16_kernel
template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_fp8_e4m3_f16_kernel(__nv_fp8_storage_t *a,
                                                         float *y, int N) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * NUM_THREADS + tid;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ half reduce_smem[NUM_WARPS];
  __nv_fp8_storage_t sum_f8 =
      (idx < N) ? a[idx]
                : __nv_cvt_float_to_fp8(0.0f, __NV_SATFINITE, __NV_E4M3);
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  half sum_f16 = warp_reduce_sum_fp8_e4m3_f16<WARP_SIZE>(sum_f8);
  if (lane == 0)
    reduce_smem[warp] = sum_f16;
  __syncthreads();
  half sum = (lane < NUM_WARPS) ? reduce_smem[lane] : __float2half(0.0f);
  if (warp == 0)
    sum = warp_reduce_sum_f16_f16<NUM_WARPS>(sum);
  if (tid == 0)
    atomicAdd(y, __half2float(sum));
}

// block_all_reduce_sum_fp8_e5m2_f16_kernel
template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_fp8_e5m2_f16_kernel(__nv_fp8_storage_t *a,
                                                         float *y, int N) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * NUM_THREADS + tid;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ half reduce_smem[NUM_WARPS];
  __nv_fp8_storage_t sum_f8 =
      (idx < N) ? a[idx]
                : __nv_cvt_float_to_fp8(0.0f, __NV_SATFINITE, __NV_E5M2);
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  half sum_f16 = warp_reduce_sum_fp8_e5m2_f16<WARP_SIZE>(sum_f8);
  if (lane == 0)
    reduce_smem[warp] = sum_f16;
  __syncthreads();
  half sum = (lane < NUM_WARPS) ? reduce_smem[lane] : __float2half(0.0f);
  if (warp == 0)
    sum = warp_reduce_sum_f16_f16<NUM_WARPS>(sum);
  if (tid == 0)
    atomicAdd(y, __half2float(sum));
}

// block_all_reduce_sum_i8_i32_kernel
template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_i8_i32_kernel(int8_t *a, int32_t *y,
                                                   int N) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * NUM_THREADS + tid;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ int32_t reduce_smem[NUM_WARPS];
  int8_t sum_i8 = (idx < N) ? a[idx] : 0;
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  int32_t sum_i32 = warp_reduce_sum_i8_i32<WARP_SIZE>(sum_i8);
  if (lane == 0)
    reduce_smem[warp] = sum_i32;
  __syncthreads();
  int32_t sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0;
  if (warp == 0)
    sum = warp_reduce_sum_i32_i32<NUM_WARPS>(sum);
  if (tid == 0)
    atomicAdd(y, sum);
}

// block_all_reduce_sum_i8x16_pack_i32_kernel
template <const int NUM_THREADS = 256 / 16>
__global__ void block_all_reduce_sum_i8x16_pack_i32_kernel(int8_t *a,
                                                           int32_t *y, int N) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * NUM_THREADS + tid) * 16;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ int32_t reduce_smem[NUM_WARPS];
  int8_t pack_a[16];
  LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);
  int32_t sum_i32 = 0;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    sum_i32 += (static_cast<int32_t>(pack_a[i]));
  }
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  sum_i32 = warp_reduce_sum_i32_i32<WARP_SIZE>(sum_i32);
  if (lane == 0)
    reduce_smem[warp] = sum_i32;
  __syncthreads();
  int32_t sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0;
  if (warp == 0)
    sum = warp_reduce_sum_i32_i32<NUM_WARPS>(sum);
  if (tid == 0)
    atomicAdd(y, sum);
}

// ---------------------------------------------------------------------------
// Helper macro for warmup + timed NVTX range
// ---------------------------------------------------------------------------
#define RUN_KERNEL(label, ...)                              \
  do {                                                      \
    for (int i = 0; i < N_WARMUP; i++) { __VA_ARGS__; }    \
    cudaDeviceSynchronize();                                \
    nvtxRangePush(label);                                   \
    for (int i = 0; i < N_ITER; i++) { __VA_ARGS__; }      \
    cudaDeviceSynchronize();                                \
    nvtxRangePop();                                         \
  } while (0)

// ---------------------------------------------------------------------------
// Default: if no macro is defined, enable all kernels
// ---------------------------------------------------------------------------
#if !defined(REDUCE_F32_F32) && !defined(REDUCE_F32X4_F32) &&     \
    !defined(REDUCE_F16_F16) && !defined(REDUCE_F16_F32) &&        \
    !defined(REDUCE_F16X2_F32) && !defined(REDUCE_F16X2_F16) &&    \
    !defined(REDUCE_F16X8_PACK_F16) && !defined(REDUCE_F16X8_PACK_F32) && \
    !defined(REDUCE_BF16_BF16) && !defined(REDUCE_BF16_F32) &&     \
    !defined(REDUCE_BF16X2_BF16) && !defined(REDUCE_BF16X2_F32) && \
    !defined(REDUCE_BF16X8_PACK_BF16) && !defined(REDUCE_BF16X8_PACK_F32) && \
    !defined(REDUCE_FP8_E4M3_F16) && !defined(REDUCE_FP8_E5M2_F16) && \
    !defined(REDUCE_I8_I32) && !defined(REDUCE_I8X16_PACK_I32)
#define REDUCE_F32_F32
#define REDUCE_F32X4_F32
#define REDUCE_F16_F16
#define REDUCE_F16_F32
#define REDUCE_F16X2_F32
#define REDUCE_F16X2_F16
#define REDUCE_F16X8_PACK_F16
#define REDUCE_F16X8_PACK_F32
#define REDUCE_BF16_BF16
#define REDUCE_BF16_F32
#define REDUCE_BF16X2_BF16
#define REDUCE_BF16X2_F32
#define REDUCE_BF16X8_PACK_BF16
#define REDUCE_BF16X8_PACK_F32
#define REDUCE_FP8_E4M3_F16
#define REDUCE_FP8_E5M2_F16
#define REDUCE_I8_I32
#define REDUCE_I8X16_PACK_I32
#endif

int main() {
  const int N = 1 << 20;
  const int block = 256;

  // --- shared float output buffer (used by all non-i8 kernels) ---
#if defined(REDUCE_F32_F32) || defined(REDUCE_F32X4_F32) ||           \
    defined(REDUCE_F16_F16) || defined(REDUCE_F16_F32) ||              \
    defined(REDUCE_F16X2_F32) || defined(REDUCE_F16X2_F16) ||          \
    defined(REDUCE_F16X8_PACK_F16) || defined(REDUCE_F16X8_PACK_F32) || \
    defined(REDUCE_BF16_BF16) || defined(REDUCE_BF16_F32) ||           \
    defined(REDUCE_BF16X2_BF16) || defined(REDUCE_BF16X2_F32) ||       \
    defined(REDUCE_BF16X8_PACK_BF16) || defined(REDUCE_BF16X8_PACK_F32) || \
    defined(REDUCE_FP8_E4M3_F16) || defined(REDUCE_FP8_E5M2_F16)
  float *d_f32_out;
  cudaMalloc(&d_f32_out, sizeof(float));
#endif

  // --- float input buffers (f32 kernels) ---
#if defined(REDUCE_F32_F32) || defined(REDUCE_F32X4_F32)
  float *d_f32_in;
  cudaMalloc(&d_f32_in, N * sizeof(float));
  cudaMemset(d_f32_in,  0, N * sizeof(float));
#endif

  // --- half buffers (f16 kernels) ---
#if defined(REDUCE_F16_F16) || defined(REDUCE_F16_F32) ||          \
    defined(REDUCE_F16X2_F32) || defined(REDUCE_F16X2_F16) ||      \
    defined(REDUCE_F16X8_PACK_F16) || defined(REDUCE_F16X8_PACK_F32)
  half *d_f16_in;
  cudaMalloc(&d_f16_in, N * sizeof(half));
  cudaMemset(d_f16_in,  0, N * sizeof(half));
#endif

  // --- bfloat16 buffers (bf16 kernels) ---
#if defined(REDUCE_BF16_BF16) || defined(REDUCE_BF16_F32) ||          \
    defined(REDUCE_BF16X2_BF16) || defined(REDUCE_BF16X2_F32) ||      \
    defined(REDUCE_BF16X8_PACK_BF16) || defined(REDUCE_BF16X8_PACK_F32)
  __nv_bfloat16 *d_bf16_in;
  cudaMalloc(&d_bf16_in, N * sizeof(__nv_bfloat16));
  cudaMemset(d_bf16_in,  0, N * sizeof(__nv_bfloat16));
#endif

  // --- fp8 e4m3 buffer ---
#if defined(REDUCE_FP8_E4M3_F16)
  __nv_fp8_storage_t *d_fp8_e4m3_in;
  cudaMalloc(&d_fp8_e4m3_in, N * sizeof(__nv_fp8_storage_t));
  cudaMemset(d_fp8_e4m3_in,  0, N * sizeof(__nv_fp8_storage_t));
#endif

  // --- fp8 e5m2 buffer ---
#if defined(REDUCE_FP8_E5M2_F16)
  __nv_fp8_storage_t *d_fp8_e5m2_in;
  cudaMalloc(&d_fp8_e5m2_in, N * sizeof(__nv_fp8_storage_t));
  cudaMemset(d_fp8_e5m2_in,  0, N * sizeof(__nv_fp8_storage_t));
#endif

  // --- int8 / int32 buffers ---
#if defined(REDUCE_I8_I32) || defined(REDUCE_I8X16_PACK_I32)
  int8_t  *d_i8_in;
  int32_t *d_i32_out;
  cudaMalloc(&d_i8_in,   N * sizeof(int8_t));
  cudaMalloc(&d_i32_out, sizeof(int32_t));
  cudaMemset(d_i8_in,    0, N * sizeof(int8_t));
#endif

  // Shared memory sizes (one element per warp).
  const size_t smem_f32  = (block / WARP_SIZE) * sizeof(float);
  const size_t smem_bf16 = (block / WARP_SIZE) * sizeof(__nv_bfloat16);
  const size_t smem_half = (block / WARP_SIZE) * sizeof(half);
  const size_t smem_i32  = (block / WARP_SIZE) * sizeof(int32_t);

  // --- block_all_reduce_sum_f32_f32_kernel ---
  // block=256, grid=1
#ifdef REDUCE_F32_F32
  {
    const int g = 1;
    cudaMemset(d_f32_out, 0, sizeof(float));
    RUN_KERNEL("block_all_reduce_sum_f32_f32_kernel",
      block_all_reduce_sum_f32_f32_kernel<256><<<g, block, smem_f32>>>(d_f32_in, d_f32_out, N));
  }
#endif

  // --- block_all_reduce_sum_f32x4_f32_kernel ---
  // NUM_THREADS=64, block=64, grid=1
#ifdef REDUCE_F32X4_F32
  {
    const int nt = 64;
    const int g  = 1;
    const size_t sm = (nt / WARP_SIZE) * sizeof(float);
    cudaMemset(d_f32_out, 0, sizeof(float));
    RUN_KERNEL("block_all_reduce_sum_f32x4_f32_kernel",
      block_all_reduce_sum_f32x4_f32_kernel<64><<<g, nt, sm>>>(d_f32_in, d_f32_out, N));
  }
#endif

  // --- block_all_reduce_sum_f16_f16_kernel ---
#ifdef REDUCE_F16_F16
  {
    const int g = 1;
    cudaMemset(d_f32_out, 0, sizeof(float));
    RUN_KERNEL("block_all_reduce_sum_f16_f16_kernel",
      block_all_reduce_sum_f16_f16_kernel<256><<<g, block, smem_f32>>>(d_f16_in, d_f32_out, N));
  }
#endif

  // --- block_all_reduce_sum_f16_f32_kernel ---
#ifdef REDUCE_F16_F32
  {
    const int g = 1;
    cudaMemset(d_f32_out, 0, sizeof(float));
    RUN_KERNEL("block_all_reduce_sum_f16_f32_kernel",
      block_all_reduce_sum_f16_f32_kernel<256><<<g, block, smem_f32>>>(d_f16_in, d_f32_out, N));
  }
#endif

  // --- block_all_reduce_sum_f16x2_f32_kernel ---
  // NUM_THREADS=128, block=128, grid=1
#ifdef REDUCE_F16X2_F32
  {
    const int nt = 128;
    const int g  = 1;
    const size_t sm = (nt / WARP_SIZE) * sizeof(float);
    cudaMemset(d_f32_out, 0, sizeof(float));
    RUN_KERNEL("block_all_reduce_sum_f16x2_f32_kernel",
      block_all_reduce_sum_f16x2_f32_kernel<128><<<g, nt, sm>>>(d_f16_in, d_f32_out, N));
  }
#endif

  // --- block_all_reduce_sum_f16x2_f16_kernel ---
#ifdef REDUCE_F16X2_F16
  {
    const int nt = 128;
    const int g  = 1;
    const size_t sm = (nt / WARP_SIZE) * sizeof(float);
    cudaMemset(d_f32_out, 0, sizeof(float));
    RUN_KERNEL("block_all_reduce_sum_f16x2_f16_kernel",
      block_all_reduce_sum_f16x2_f16_kernel<128><<<g, nt, sm>>>(d_f16_in, d_f32_out, N));
  }
#endif

  // --- block_all_reduce_sum_f16x8_pack_f16_kernel ---
  // NUM_THREADS=32, block=32, grid=1
#ifdef REDUCE_F16X8_PACK_F16
  {
    const int nt = 32;
    const int g  = 1;
    const size_t sm = (nt / WARP_SIZE) * sizeof(float);
    cudaMemset(d_f32_out, 0, sizeof(float));
    RUN_KERNEL("block_all_reduce_sum_f16x8_pack_f16_kernel",
      block_all_reduce_sum_f16x8_pack_f16_kernel<32><<<g, nt, sm>>>(d_f16_in, d_f32_out, N));
  }
#endif

  // --- block_all_reduce_sum_f16x8_pack_f32_kernel ---
#ifdef REDUCE_F16X8_PACK_F32
  {
    const int nt = 32;
    const int g  = 1;
    const size_t sm = (nt / WARP_SIZE) * sizeof(float);
    cudaMemset(d_f32_out, 0, sizeof(float));
    RUN_KERNEL("block_all_reduce_sum_f16x8_pack_f32_kernel",
      block_all_reduce_sum_f16x8_pack_f32_kernel<32><<<g, nt, sm>>>(d_f16_in, d_f32_out, N));
  }
#endif

  // --- block_all_reduce_sum_bf16_bf16_kernel ---
#ifdef REDUCE_BF16_BF16
  {
    const int g = 1;
    cudaMemset(d_f32_out, 0, sizeof(float));
    RUN_KERNEL("block_all_reduce_sum_bf16_bf16_kernel",
      block_all_reduce_sum_bf16_bf16_kernel<256><<<g, block, smem_bf16>>>(d_bf16_in, d_f32_out, N));
  }
#endif

  // --- block_all_reduce_sum_bf16_f32_kernel ---
#ifdef REDUCE_BF16_F32
  {
    const int g = 1;
    cudaMemset(d_f32_out, 0, sizeof(float));
    RUN_KERNEL("block_all_reduce_sum_bf16_f32_kernel",
      block_all_reduce_sum_bf16_f32_kernel<256><<<g, block, smem_f32>>>(d_bf16_in, d_f32_out, N));
  }
#endif

  // --- block_all_reduce_sum_bf16x2_bf16_kernel ---
  // NUM_THREADS=128
#ifdef REDUCE_BF16X2_BF16
  {
    const int nt = 128;
    const int g  = 1;
    const size_t sm = (nt / WARP_SIZE) * sizeof(__nv_bfloat16);
    cudaMemset(d_f32_out, 0, sizeof(float));
    RUN_KERNEL("block_all_reduce_sum_bf16x2_bf16_kernel",
      block_all_reduce_sum_bf16x2_bf16_kernel<128><<<g, nt, sm>>>(d_bf16_in, d_f32_out, N));
  }
#endif

  // --- block_all_reduce_sum_bf16x2_f32_kernel ---
#ifdef REDUCE_BF16X2_F32
  {
    const int nt = 128;
    const int g  = 1;
    const size_t sm = (nt / WARP_SIZE) * sizeof(float);
    cudaMemset(d_f32_out, 0, sizeof(float));
    RUN_KERNEL("block_all_reduce_sum_bf16x2_f32_kernel",
      block_all_reduce_sum_bf16x2_f32_kernel<128><<<g, nt, sm>>>(d_bf16_in, d_f32_out, N));
  }
#endif

  // --- block_all_reduce_sum_bf16x8_pack_bf16_kernel ---
  // NUM_THREADS=32
#ifdef REDUCE_BF16X8_PACK_BF16
  {
    const int nt = 32;
    const int g  = 1;
    const size_t sm = (nt / WARP_SIZE) * sizeof(__nv_bfloat16);
    cudaMemset(d_f32_out, 0, sizeof(float));
    RUN_KERNEL("block_all_reduce_sum_bf16x8_pack_bf16_kernel",
      block_all_reduce_sum_bf16x8_pack_bf16_kernel<32><<<g, nt, sm>>>(d_bf16_in, d_f32_out, N));
  }
#endif

  // --- block_all_reduce_sum_bf16x8_pack_f32_kernel ---
#ifdef REDUCE_BF16X8_PACK_F32
  {
    const int nt = 32;
    const int g  = 1;
    const size_t sm = (nt / WARP_SIZE) * sizeof(float);
    cudaMemset(d_f32_out, 0, sizeof(float));
    RUN_KERNEL("block_all_reduce_sum_bf16x8_pack_f32_kernel",
      block_all_reduce_sum_bf16x8_pack_f32_kernel<32><<<g, nt, sm>>>(d_bf16_in, d_f32_out, N));
  }
#endif

  // --- block_all_reduce_sum_fp8_e4m3_f16_kernel ---
#ifdef REDUCE_FP8_E4M3_F16
  {
    const int g = 1;
    cudaMemset(d_f32_out, 0, sizeof(float));
    RUN_KERNEL("block_all_reduce_sum_fp8_e4m3_f16_kernel",
      block_all_reduce_sum_fp8_e4m3_f16_kernel<256><<<g, block, smem_half>>>(d_fp8_e4m3_in, d_f32_out, N));
  }
#endif

  // --- block_all_reduce_sum_fp8_e5m2_f16_kernel ---
#ifdef REDUCE_FP8_E5M2_F16
  {
    const int g = 1;
    cudaMemset(d_f32_out, 0, sizeof(float));
    RUN_KERNEL("block_all_reduce_sum_fp8_e5m2_f16_kernel",
      block_all_reduce_sum_fp8_e5m2_f16_kernel<256><<<g, block, smem_half>>>(d_fp8_e5m2_in, d_f32_out, N));
  }
#endif

  // --- block_all_reduce_sum_i8_i32_kernel ---
#ifdef REDUCE_I8_I32
  {
    const int g = 1;
    cudaMemset(d_i32_out, 0, sizeof(int32_t));
    RUN_KERNEL("block_all_reduce_sum_i8_i32_kernel",
      block_all_reduce_sum_i8_i32_kernel<256><<<g, block, smem_i32>>>(d_i8_in, d_i32_out, N));
  }
#endif

  // --- block_all_reduce_sum_i8x16_pack_i32_kernel ---
  // NUM_THREADS=16, block=16, grid=1
#ifdef REDUCE_I8X16_PACK_I32
  {
    const int nt = 16;
    const int g  = 1;
    const size_t sm = ((nt + WARP_SIZE - 1) / WARP_SIZE) * sizeof(int32_t);
    cudaMemset(d_i32_out, 0, sizeof(int32_t));
    RUN_KERNEL("block_all_reduce_sum_i8x16_pack_i32_kernel",
      block_all_reduce_sum_i8x16_pack_i32_kernel<16><<<g, nt, sm>>>(d_i8_in, d_i32_out, N));
  }
#endif

#if defined(REDUCE_F32_F32) || defined(REDUCE_F32X4_F32) ||           \
    defined(REDUCE_F16_F16) || defined(REDUCE_F16_F32) ||              \
    defined(REDUCE_F16X2_F32) || defined(REDUCE_F16X2_F16) ||          \
    defined(REDUCE_F16X8_PACK_F16) || defined(REDUCE_F16X8_PACK_F32) || \
    defined(REDUCE_BF16_BF16) || defined(REDUCE_BF16_F32) ||           \
    defined(REDUCE_BF16X2_BF16) || defined(REDUCE_BF16X2_F32) ||       \
    defined(REDUCE_BF16X8_PACK_BF16) || defined(REDUCE_BF16X8_PACK_F32) || \
    defined(REDUCE_FP8_E4M3_F16) || defined(REDUCE_FP8_E5M2_F16)
  cudaFree(d_f32_out);
#endif
#if defined(REDUCE_F32_F32) || defined(REDUCE_F32X4_F32)
  cudaFree(d_f32_in);
#endif
#if defined(REDUCE_F16_F16) || defined(REDUCE_F16_F32) ||          \
    defined(REDUCE_F16X2_F32) || defined(REDUCE_F16X2_F16) ||      \
    defined(REDUCE_F16X8_PACK_F16) || defined(REDUCE_F16X8_PACK_F32)
  cudaFree(d_f16_in);
#endif
#if defined(REDUCE_BF16_BF16) || defined(REDUCE_BF16_F32) ||          \
    defined(REDUCE_BF16X2_BF16) || defined(REDUCE_BF16X2_F32) ||      \
    defined(REDUCE_BF16X8_PACK_BF16) || defined(REDUCE_BF16X8_PACK_F32)
  cudaFree(d_bf16_in);
#endif
#if defined(REDUCE_FP8_E4M3_F16)
  cudaFree(d_fp8_e4m3_in);
#endif
#if defined(REDUCE_FP8_E5M2_F16)
  cudaFree(d_fp8_e5m2_in);
#endif
#if defined(REDUCE_I8_I32) || defined(REDUCE_I8X16_PACK_I32)
  cudaFree(d_i8_in);
  cudaFree(d_i32_out);
#endif

  printf("Done.\n");
  return 0;
}
