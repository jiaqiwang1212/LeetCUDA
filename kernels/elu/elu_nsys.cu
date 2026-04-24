#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <stdio.h>

#define N_WARMUP 5
#define N_ITER 20

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

#define ALPHA 1.0f

// Device helpers
__device__ __forceinline__ float elu(float x) {
  return x > 0.f ? x : ALPHA * (expf(x) - 1.f);
}

__device__ __forceinline__ half elu_half(half x) {
  return __hgt(x, __float2half(0.f))
             ? x
             : __hmul(__float2half(ALPHA), __hsub(hexp(x), __float2half(1.f)));
}

// FP32 scalar
__global__ void elu_f32_kernel(float *x, float *y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    y[idx] = elu(x[idx]);
}

// FP32 x4
__global__ void elu_f32x4_kernel(float *x, float *y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx < N) {
    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_y;
    reg_y.x = elu(reg_x.x);
    reg_y.y = elu(reg_x.y);
    reg_y.z = elu(reg_x.z);
    reg_y.w = elu(reg_x.w);
    FLOAT4(y[idx]) = reg_y;
  }
}

// FP16 scalar
__global__ void elu_f16_kernel(half *x, half *y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    y[idx] = elu_half(x[idx]);
}

// FP16 x2
__global__ void elu_f16x2_kernel(half *x, half *y, int N) {
  int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    half2 reg_x = HALF2(x[idx]);
    half2 reg_y;
    reg_y.x = elu_half(reg_x.x);
    reg_y.y = elu_half(reg_x.y);
    HALF2(y[idx]) = reg_y;
  }
}

// FP16 x8 unpack
__global__ void elu_f16x8_kernel(half *x, half *y, int N) {
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  half2 reg_x_0 = HALF2(x[idx + 0]);
  half2 reg_x_1 = HALF2(x[idx + 2]);
  half2 reg_x_2 = HALF2(x[idx + 4]);
  half2 reg_x_3 = HALF2(x[idx + 6]);
  half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;
  reg_y_0.x = elu_half(reg_x_0.x);
  reg_y_0.y = elu_half(reg_x_0.y);
  reg_y_1.x = elu_half(reg_x_1.x);
  reg_y_1.y = elu_half(reg_x_1.y);
  reg_y_2.x = elu_half(reg_x_2.x);
  reg_y_2.y = elu_half(reg_x_2.y);
  reg_y_3.x = elu_half(reg_x_3.x);
  reg_y_3.y = elu_half(reg_x_3.y);
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

// FP16 x8 pack
__global__ void elu_f16x8_pack_kernel(half *x, half *y, int N) {
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  half pack_x[8], pack_y[8];
  LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);

#pragma unroll
  for (int i = 0; i < 8; i++) {
    pack_y[i] = elu_half(pack_x[i]);
  }
  if ((idx + 7) < N) {
    LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
  }
}

// Compile-time kernel selection: pass e.g. -DELU_F32 to nvcc to profile only that variant.
// If none are defined, all variants are enabled.
#if !defined(ELU_F32) && !defined(ELU_F32X4) && !defined(ELU_F16) && \
    !defined(ELU_F16X2) && !defined(ELU_F16X8) && !defined(ELU_F16X8_PACK)
#define ELU_F32
#define ELU_F32X4
#define ELU_F16
#define ELU_F16X2
#define ELU_F16X8
#define ELU_F16X8_PACK
#endif

int main() {
  const int N = 1 << 20;

#if defined(ELU_F32) || defined(ELU_F32X4)
  float *d_x_f32, *d_y_f32;
  cudaMalloc(&d_x_f32, N * sizeof(float));
  cudaMalloc(&d_y_f32, N * sizeof(float));
  cudaMemset(d_x_f32, 0, N * sizeof(float));
#endif

#if defined(ELU_F16) || defined(ELU_F16X2) || defined(ELU_F16X8) || defined(ELU_F16X8_PACK)
  half *d_x_f16, *d_y_f16;
  cudaMalloc(&d_x_f16, N * sizeof(half));
  cudaMalloc(&d_y_f16, N * sizeof(half));
  cudaMemset(d_x_f16, 0, N * sizeof(half));
#endif

#ifdef ELU_F32
  // --- elu_f32_kernel ---
  {
    int block = 256, grid = (N + 255) / 256;
    for (int i = 0; i < N_WARMUP; i++)
      elu_f32_kernel<<<grid, block>>>(d_x_f32, d_y_f32, N);
    cudaDeviceSynchronize();
    nvtxRangePush("elu_f32_kernel");
    for (int i = 0; i < N_ITER; i++)
      elu_f32_kernel<<<grid, block>>>(d_x_f32, d_y_f32, N);
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

#ifdef ELU_F32X4
  // --- elu_f32x4_kernel ---
  {
    int block = 64, grid = (N + 255) / 256;
    for (int i = 0; i < N_WARMUP; i++)
      elu_f32x4_kernel<<<grid, block>>>(d_x_f32, d_y_f32, N);
    cudaDeviceSynchronize();
    nvtxRangePush("elu_f32x4_kernel");
    for (int i = 0; i < N_ITER; i++)
      elu_f32x4_kernel<<<grid, block>>>(d_x_f32, d_y_f32, N);
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

#ifdef ELU_F16
  // --- elu_f16_kernel ---
  {
    int block = 256, grid = (N + 255) / 256;
    for (int i = 0; i < N_WARMUP; i++)
      elu_f16_kernel<<<grid, block>>>(d_x_f16, d_y_f16, N);
    cudaDeviceSynchronize();
    nvtxRangePush("elu_f16_kernel");
    for (int i = 0; i < N_ITER; i++)
      elu_f16_kernel<<<grid, block>>>(d_x_f16, d_y_f16, N);
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

#ifdef ELU_F16X2
  // --- elu_f16x2_kernel ---
  {
    int block = 128, grid = (N + 255) / 256;
    for (int i = 0; i < N_WARMUP; i++)
      elu_f16x2_kernel<<<grid, block>>>(d_x_f16, d_y_f16, N);
    cudaDeviceSynchronize();
    nvtxRangePush("elu_f16x2_kernel");
    for (int i = 0; i < N_ITER; i++)
      elu_f16x2_kernel<<<grid, block>>>(d_x_f16, d_y_f16, N);
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

#ifdef ELU_F16X8
  // --- elu_f16x8_kernel ---
  {
    int block = 32, grid = (N + 255) / 256;
    for (int i = 0; i < N_WARMUP; i++)
      elu_f16x8_kernel<<<grid, block>>>(d_x_f16, d_y_f16, N);
    cudaDeviceSynchronize();
    nvtxRangePush("elu_f16x8_kernel");
    for (int i = 0; i < N_ITER; i++)
      elu_f16x8_kernel<<<grid, block>>>(d_x_f16, d_y_f16, N);
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

#ifdef ELU_F16X8_PACK
  // --- elu_f16x8_pack_kernel ---
  {
    int block = 32, grid = (N + 255) / 256;
    for (int i = 0; i < N_WARMUP; i++)
      elu_f16x8_pack_kernel<<<grid, block>>>(d_x_f16, d_y_f16, N);
    cudaDeviceSynchronize();
    nvtxRangePush("elu_f16x8_pack_kernel");
    for (int i = 0; i < N_ITER; i++)
      elu_f16x8_pack_kernel<<<grid, block>>>(d_x_f16, d_y_f16, N);
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

#if defined(ELU_F32) || defined(ELU_F32X4)
  cudaFree(d_x_f32);
  cudaFree(d_y_f32);
#endif
#if defined(ELU_F16) || defined(ELU_F16X2) || defined(ELU_F16X8) || defined(ELU_F16X8_PACK)
  cudaFree(d_x_f16);
  cudaFree(d_y_f16);
#endif
  printf("Done.\n");
  return 0;
}
