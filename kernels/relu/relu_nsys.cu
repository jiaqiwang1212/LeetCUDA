#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <stdio.h>

#define N_WARMUP 5
#define N_ITER 20

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// FP32 scalar
__global__ void relu_f32_kernel(float *x, float *y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    y[idx] = fmaxf(0.0f, x[idx]);
}

// FP32 x4
__global__ void relu_f32x4_kernel(float *x, float *y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx < N) {
    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_y;
    reg_y.x = fmaxf(0.0f, reg_x.x);
    reg_y.y = fmaxf(0.0f, reg_x.y);
    reg_y.z = fmaxf(0.0f, reg_x.z);
    reg_y.w = fmaxf(0.0f, reg_x.w);
    FLOAT4(y[idx]) = reg_y;
  }
}

// FP16 scalar
__global__ void relu_f16_kernel(half *x, half *y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    y[idx] = __hmax(__float2half(0.0f), x[idx]);
}

// FP16 x2
__global__ void relu_f16x2_kernel(half *x, half *y, int N) {
  int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    half2 reg_x = HALF2(x[idx]);
    half2 reg_y = HALF2(y[idx]);
    reg_y.x = __hmax(__float2half(0.0f), reg_x.x);
    reg_y.y = __hmax(__float2half(0.0f), reg_x.y);
    HALF2(y[idx]) = reg_y;
  }
}

// FP16 x8 unpack
__global__ void relu_f16x8_kernel(half *x, half *y, int N) {
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  half2 reg_x_0 = HALF2(x[idx + 0]);
  half2 reg_x_1 = HALF2(x[idx + 2]);
  half2 reg_x_2 = HALF2(x[idx + 4]);
  half2 reg_x_3 = HALF2(x[idx + 6]);
  half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;
  reg_y_0.x = __hmax(__float2half(0.0f), reg_x_0.x);
  reg_y_0.y = __hmax(__float2half(0.0f), reg_x_0.y);
  reg_y_1.x = __hmax(__float2half(0.0f), reg_x_1.x);
  reg_y_1.y = __hmax(__float2half(0.0f), reg_x_1.y);
  reg_y_2.x = __hmax(__float2half(0.0f), reg_x_2.x);
  reg_y_2.y = __hmax(__float2half(0.0f), reg_x_2.y);
  reg_y_3.x = __hmax(__float2half(0.0f), reg_x_3.x);
  reg_y_3.y = __hmax(__float2half(0.0f), reg_x_3.y);
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
__global__ void relu_f16x8_pack_kernel(half *x, half *y, int N) {
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  const half2 z2 = {__float2half(0.0f), __float2half(0.0f)};
  half pack_x[8], pack_y[8];
  LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);

#pragma unroll
  for (int i = 0; i < 8; i += 2) {
    HALF2(pack_y[i]) = __hmax2(HALF2(pack_x[i]), z2);
  }
  if ((idx + 7) < N) {
    LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
  }
}

// Compile-time kernel selection: pass e.g. -DRELU_F32 to nvcc to profile only that variant.
// If none are defined, all variants are enabled.
#if !defined(RELU_F32) && !defined(RELU_F32X4) && !defined(RELU_F16) && \
    !defined(RELU_F16X2) && !defined(RELU_F16X8) && !defined(RELU_F16X8_PACK)
#define RELU_F32
#define RELU_F32X4
#define RELU_F16
#define RELU_F16X2
#define RELU_F16X8
#define RELU_F16X8_PACK
#endif

int main() {
  const int N = 1 << 20;

#if defined(RELU_F32) || defined(RELU_F32X4)
  float *d_x_f32, *d_y_f32;
  cudaMalloc(&d_x_f32, N * sizeof(float));
  cudaMalloc(&d_y_f32, N * sizeof(float));
  cudaMemset(d_x_f32, 0, N * sizeof(float));
#endif

#if defined(RELU_F16) || defined(RELU_F16X2) || defined(RELU_F16X8) || defined(RELU_F16X8_PACK)
  half *d_x_f16, *d_y_f16;
  cudaMalloc(&d_x_f16, N * sizeof(half));
  cudaMalloc(&d_y_f16, N * sizeof(half));
  cudaMemset(d_x_f16, 0, N * sizeof(half));
#endif

#ifdef RELU_F32
  // --- relu_f32_kernel ---
  {
    int block = 256, grid = (N + 255) / 256;
    for (int i = 0; i < N_WARMUP; i++)
      relu_f32_kernel<<<grid, block>>>(d_x_f32, d_y_f32, N);
    cudaDeviceSynchronize();
    nvtxRangePush("relu_f32_kernel");
    for (int i = 0; i < N_ITER; i++)
      relu_f32_kernel<<<grid, block>>>(d_x_f32, d_y_f32, N);
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

#ifdef RELU_F32X4
  // --- relu_f32x4_kernel ---
  {
    int block = 64, grid = (N + 255) / 256;
    for (int i = 0; i < N_WARMUP; i++)
      relu_f32x4_kernel<<<grid, block>>>(d_x_f32, d_y_f32, N);
    cudaDeviceSynchronize();
    nvtxRangePush("relu_f32x4_kernel");
    for (int i = 0; i < N_ITER; i++)
      relu_f32x4_kernel<<<grid, block>>>(d_x_f32, d_y_f32, N);
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

#ifdef RELU_F16
  // --- relu_f16_kernel ---
  {
    int block = 256, grid = (N + 255) / 256;
    for (int i = 0; i < N_WARMUP; i++)
      relu_f16_kernel<<<grid, block>>>(d_x_f16, d_y_f16, N);
    cudaDeviceSynchronize();
    nvtxRangePush("relu_f16_kernel");
    for (int i = 0; i < N_ITER; i++)
      relu_f16_kernel<<<grid, block>>>(d_x_f16, d_y_f16, N);
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

#ifdef RELU_F16X2
  // --- relu_f16x2_kernel ---
  {
    int block = 128, grid = (N + 255) / 256;
    for (int i = 0; i < N_WARMUP; i++)
      relu_f16x2_kernel<<<grid, block>>>(d_x_f16, d_y_f16, N);
    cudaDeviceSynchronize();
    nvtxRangePush("relu_f16x2_kernel");
    for (int i = 0; i < N_ITER; i++)
      relu_f16x2_kernel<<<grid, block>>>(d_x_f16, d_y_f16, N);
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

#ifdef RELU_F16X8
  // --- relu_f16x8_kernel ---
  {
    int block = 32, grid = (N + 255) / 256;
    for (int i = 0; i < N_WARMUP; i++)
      relu_f16x8_kernel<<<grid, block>>>(d_x_f16, d_y_f16, N);
    cudaDeviceSynchronize();
    nvtxRangePush("relu_f16x8_kernel");
    for (int i = 0; i < N_ITER; i++)
      relu_f16x8_kernel<<<grid, block>>>(d_x_f16, d_y_f16, N);
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

#ifdef RELU_F16X8_PACK
  // --- relu_f16x8_pack_kernel ---
  {
    int block = 32, grid = (N + 255) / 256;
    for (int i = 0; i < N_WARMUP; i++)
      relu_f16x8_pack_kernel<<<grid, block>>>(d_x_f16, d_y_f16, N);
    cudaDeviceSynchronize();
    nvtxRangePush("relu_f16x8_pack_kernel");
    for (int i = 0; i < N_ITER; i++)
      relu_f16x8_pack_kernel<<<grid, block>>>(d_x_f16, d_y_f16, N);
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

#if defined(RELU_F32) || defined(RELU_F32X4)
  cudaFree(d_x_f32);
  cudaFree(d_y_f32);
#endif
#if defined(RELU_F16) || defined(RELU_F16X2) || defined(RELU_F16X8) || defined(RELU_F16X8_PACK)
  cudaFree(d_x_f16);
  cudaFree(d_y_f16);
#endif
  printf("Done.\n");
  return 0;
}
