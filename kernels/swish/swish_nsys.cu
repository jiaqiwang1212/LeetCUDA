#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <stdio.h>
#include <stdlib.h>

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

#define N_WARMUP 5
#define N_ITER 20

// FP32
__device__ __forceinline__ float swish(float x) {
  return x / (1.0f + expf(-x));
}

__global__ void swish_f32_kernel(float *x, float *y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    y[idx] = swish(x[idx]);
}

__global__ void swish_f32x4_kernel(float *x, float *y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx < N) {
    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_y;
    reg_y.x = swish(reg_x.x);
    reg_y.y = swish(reg_x.y);
    reg_y.z = swish(reg_x.z);
    reg_y.w = swish(reg_x.w);
    FLOAT4(y[idx]) = reg_y;
  }
}

// FP16
__device__ __forceinline__ half swish_half(half x) {
  return __hmul(x, __hdiv(__float2half(1.0f),
                          __hadd(__float2half(1.0f), hexp(__hneg(x)))));
}

__global__ void swish_f16_kernel(half *x, half *y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    y[idx] = swish_half(x[idx]);
}

__global__ void swish_f16x2_kernel(half *x, half *y, int N) {
  int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    half2 reg_x = HALF2(x[idx]);
    half2 reg_y;
    reg_y.x = swish_half(reg_x.x);
    reg_y.y = swish_half(reg_x.y);
    HALF2(y[idx]) = reg_y;
  }
}

__global__ void swish_f16x8_kernel(half *x, half *y, int N) {
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  half2 reg_x_0 = HALF2(x[idx + 0]);
  half2 reg_x_1 = HALF2(x[idx + 2]);
  half2 reg_x_2 = HALF2(x[idx + 4]);
  half2 reg_x_3 = HALF2(x[idx + 6]);
  half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;
  reg_y_0.x = swish_half(reg_x_0.x);
  reg_y_0.y = swish_half(reg_x_0.y);
  reg_y_1.x = swish_half(reg_x_1.x);
  reg_y_1.y = swish_half(reg_x_1.y);
  reg_y_2.x = swish_half(reg_x_2.x);
  reg_y_2.y = swish_half(reg_x_2.y);
  reg_y_3.x = swish_half(reg_x_3.x);
  reg_y_3.y = swish_half(reg_x_3.y);
  if ((idx + 0) < N) {
    HALF2(y[idx + 0]) = reg_y_0;
  }
  if ((idx + 2) < N) {
    HALF2(y[idx + 2]) = reg_y_1;
  }
  if ((idx + 4) < N) {
    HALF2(y[idx + 4]) = reg_y_2;
  }
  if ((idx + 6) < N) {
    HALF2(y[idx + 6]) = reg_y_3;
  }
}

__global__ void swish_f16x8_pack_kernel(half *x, half *y, int N) {
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  half pack_x[8], pack_y[8];
  LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);

#pragma unroll
  for (int i = 0; i < 8; i++) {
    pack_y[i] = swish_half(pack_x[i]);
  }
  if ((idx + 7) < N) {
    LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
  }
}

// Compile-time kernel selection: pass e.g. -DSWISH_F32 to nvcc to profile only that variant.
// If none are defined, all variants are enabled.
#if !defined(SWISH_F32) && !defined(SWISH_F32X4) && !defined(SWISH_F16) && \
    !defined(SWISH_F16X2) && !defined(SWISH_F16X8) && !defined(SWISH_F16X8_PACK)
#define SWISH_F32
#define SWISH_F32X4
#define SWISH_F16
#define SWISH_F16X2
#define SWISH_F16X8
#define SWISH_F16X8_PACK
#endif

int main() {
  const int N = 1 << 20;

#if defined(SWISH_F32) || defined(SWISH_F32X4)
  // FP32 buffers
  float *d_x_f32, *d_y_f32;
  cudaMalloc(&d_x_f32, N * sizeof(float));
  cudaMalloc(&d_y_f32, N * sizeof(float));

  // Initialize f32 input
  float *h_x_f32 = (float *)malloc(N * sizeof(float));
  for (int i = 0; i < N; i++) h_x_f32[i] = (float)(i % 256) - 128.0f;
  cudaMemcpy(d_x_f32, h_x_f32, N * sizeof(float), cudaMemcpyHostToDevice);
  free(h_x_f32);
#endif

#if defined(SWISH_F16) || defined(SWISH_F16X2) || defined(SWISH_F16X8) || defined(SWISH_F16X8_PACK)
  // FP16 buffers
  half *d_x_f16, *d_y_f16;
  cudaMalloc(&d_x_f16, N * sizeof(half));
  cudaMalloc(&d_y_f16, N * sizeof(half));

  // Initialize f16 input
  half *h_x_f16 = (half *)malloc(N * sizeof(half));
  for (int i = 0; i < N; i++)
    h_x_f16[i] = __float2half((float)(i % 256) - 128.0f);
  cudaMemcpy(d_x_f16, h_x_f16, N * sizeof(half), cudaMemcpyHostToDevice);
  free(h_x_f16);
#endif

#ifdef SWISH_F32
  // ------------------------------------------------------------------
  // swish_f32_kernel: block=256, grid=(N+255)/256
  {
    const int block = 256;
    const int grid = (N + 255) / 256;
    for (int i = 0; i < N_WARMUP; i++)
      swish_f32_kernel<<<grid, block>>>(d_x_f32, d_y_f32, N);
    cudaDeviceSynchronize();
    nvtxRangePush("swish_f32_kernel");
    for (int i = 0; i < N_ITER; i++)
      swish_f32_kernel<<<grid, block>>>(d_x_f32, d_y_f32, N);
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

#ifdef SWISH_F32X4
  // ------------------------------------------------------------------
  // swish_f32x4_kernel: block=64, grid=(N+255)/256
  {
    const int block = 64;
    const int grid = (N + 255) / 256;
    for (int i = 0; i < N_WARMUP; i++)
      swish_f32x4_kernel<<<grid, block>>>(d_x_f32, d_y_f32, N);
    cudaDeviceSynchronize();
    nvtxRangePush("swish_f32x4_kernel");
    for (int i = 0; i < N_ITER; i++)
      swish_f32x4_kernel<<<grid, block>>>(d_x_f32, d_y_f32, N);
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

#ifdef SWISH_F16
  // ------------------------------------------------------------------
  // swish_f16_kernel: block=256, grid=(N+255)/256
  {
    const int block = 256;
    const int grid = (N + 255) / 256;
    for (int i = 0; i < N_WARMUP; i++)
      swish_f16_kernel<<<grid, block>>>(d_x_f16, d_y_f16, N);
    cudaDeviceSynchronize();
    nvtxRangePush("swish_f16_kernel");
    for (int i = 0; i < N_ITER; i++)
      swish_f16_kernel<<<grid, block>>>(d_x_f16, d_y_f16, N);
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

#ifdef SWISH_F16X2
  // ------------------------------------------------------------------
  // swish_f16x2_kernel: block=128, grid=(N+255)/256
  {
    const int block = 128;
    const int grid = (N + 255) / 256;
    for (int i = 0; i < N_WARMUP; i++)
      swish_f16x2_kernel<<<grid, block>>>(d_x_f16, d_y_f16, N);
    cudaDeviceSynchronize();
    nvtxRangePush("swish_f16x2_kernel");
    for (int i = 0; i < N_ITER; i++)
      swish_f16x2_kernel<<<grid, block>>>(d_x_f16, d_y_f16, N);
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

#ifdef SWISH_F16X8
  // ------------------------------------------------------------------
  // swish_f16x8_kernel: block=32, grid=(N+255)/256
  {
    const int block = 32;
    const int grid = (N + 255) / 256;
    for (int i = 0; i < N_WARMUP; i++)
      swish_f16x8_kernel<<<grid, block>>>(d_x_f16, d_y_f16, N);
    cudaDeviceSynchronize();
    nvtxRangePush("swish_f16x8_kernel");
    for (int i = 0; i < N_ITER; i++)
      swish_f16x8_kernel<<<grid, block>>>(d_x_f16, d_y_f16, N);
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

#ifdef SWISH_F16X8_PACK
  // ------------------------------------------------------------------
  // swish_f16x8_pack_kernel: block=32, grid=(N+255)/256
  {
    const int block = 32;
    const int grid = (N + 255) / 256;
    for (int i = 0; i < N_WARMUP; i++)
      swish_f16x8_pack_kernel<<<grid, block>>>(d_x_f16, d_y_f16, N);
    cudaDeviceSynchronize();
    nvtxRangePush("swish_f16x8_pack_kernel");
    for (int i = 0; i < N_ITER; i++)
      swish_f16x8_pack_kernel<<<grid, block>>>(d_x_f16, d_y_f16, N);
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

#if defined(SWISH_F32) || defined(SWISH_F32X4)
  cudaFree(d_x_f32);
  cudaFree(d_y_f32);
#endif
#if defined(SWISH_F16) || defined(SWISH_F16X2) || defined(SWISH_F16X8) || defined(SWISH_F16X8_PACK)
  cudaFree(d_x_f16);
  cudaFree(d_y_f16);
#endif
  printf("Done.\n");
  return 0;
}
