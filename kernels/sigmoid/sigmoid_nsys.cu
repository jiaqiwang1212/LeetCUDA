#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <stdio.h>

#define N_WARMUP 5
#define N_ITER 20

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
#define MAX_EXP_F32 88.3762626647949f
#define MIN_EXP_F32 -88.3762626647949f
#define MAX_EXP_F16 __float2half(11.089866488461016f)
#define MIN_EXP_F16 __float2half(-9.704060527839234f)

// FP32 scalar
__global__ void sigmoid_f32_kernel(float *x, float *y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    float v = x[idx];
    v = fminf(fmaxf(v, MIN_EXP_F32), MAX_EXP_F32);
    y[idx] = 1.0f / (1.0f + expf(-v));
  }
}

// FP32 x4
__global__ void sigmoid_f32x4_kernel(float *x, float *y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  float4 reg_x = FLOAT4(x[idx]);
  float4 reg_y;

  reg_x.x = fminf(fmaxf(reg_x.x, MIN_EXP_F32), MAX_EXP_F32);
  reg_x.y = fminf(fmaxf(reg_x.y, MIN_EXP_F32), MAX_EXP_F32);
  reg_x.z = fminf(fmaxf(reg_x.z, MIN_EXP_F32), MAX_EXP_F32);
  reg_x.w = fminf(fmaxf(reg_x.w, MIN_EXP_F32), MAX_EXP_F32);

  reg_y.x = 1.0f / (1.0f + expf(-reg_x.x));
  reg_y.y = 1.0f / (1.0f + expf(-reg_x.y));
  reg_y.z = 1.0f / (1.0f + expf(-reg_x.z));
  reg_y.w = 1.0f / (1.0f + expf(-reg_x.w));

  if ((idx + 0) < N) {
    FLOAT4(y[idx]) = reg_y;
  }
}

// FP16 scalar
__global__ void sigmoid_f16_kernel(half *x, half *y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const half f = __float2half(1.0f);
  if (idx < N) {
    half v = x[idx];
    v = __hmin(__hmax(v, MIN_EXP_F16), MAX_EXP_F16);
    y[idx] = f / (f + hexp(-v));
  }
}

// FP16 x2
__global__ void sigmoid_f16x2_kernel(half *x, half *y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  const half f = __float2half(1.0f);
  half2 reg_x = HALF2(x[idx]);
  half2 reg_y;
  reg_x.x = __hmin(__hmax(reg_x.x, MIN_EXP_F16), MAX_EXP_F16);
  reg_x.y = __hmin(__hmax(reg_x.y, MIN_EXP_F16), MAX_EXP_F16);

  reg_y.x = f / (f + hexp(-reg_x.x));
  reg_y.y = f / (f + hexp(-reg_x.y));

  if ((idx + 0) < N) {
    HALF2(y[idx]) = reg_y;
  }
}

// FP16 x8 unpack
__global__ void sigmoid_f16x8_kernel(half *x, half *y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  const half f = __float2half(1.0f);

  half2 reg_x_0 = HALF2(x[idx + 0]);
  half2 reg_x_1 = HALF2(x[idx + 2]);
  half2 reg_x_2 = HALF2(x[idx + 4]);
  half2 reg_x_3 = HALF2(x[idx + 6]);

  reg_x_0.x = __hmin(__hmax(reg_x_0.x, MIN_EXP_F16), MAX_EXP_F16);
  reg_x_0.y = __hmin(__hmax(reg_x_0.y, MIN_EXP_F16), MAX_EXP_F16);
  reg_x_1.x = __hmin(__hmax(reg_x_1.x, MIN_EXP_F16), MAX_EXP_F16);
  reg_x_1.y = __hmin(__hmax(reg_x_1.y, MIN_EXP_F16), MAX_EXP_F16);
  reg_x_2.x = __hmin(__hmax(reg_x_2.x, MIN_EXP_F16), MAX_EXP_F16);
  reg_x_2.y = __hmin(__hmax(reg_x_2.y, MIN_EXP_F16), MAX_EXP_F16);
  reg_x_3.x = __hmin(__hmax(reg_x_3.x, MIN_EXP_F16), MAX_EXP_F16);
  reg_x_3.y = __hmin(__hmax(reg_x_3.y, MIN_EXP_F16), MAX_EXP_F16);

  half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;

  reg_y_0.x = f / (f + hexp(-reg_x_0.x));
  reg_y_0.y = f / (f + hexp(-reg_x_0.y));
  reg_y_1.x = f / (f + hexp(-reg_x_1.x));
  reg_y_1.y = f / (f + hexp(-reg_x_1.y));
  reg_y_2.x = f / (f + hexp(-reg_x_2.x));
  reg_y_2.y = f / (f + hexp(-reg_x_2.y));
  reg_y_3.x = f / (f + hexp(-reg_x_3.x));
  reg_y_3.y = f / (f + hexp(-reg_x_3.y));

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
__global__ void sigmoid_f16x8_pack_kernel(half *x, half *y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  const half f = __float2half(1.0f);
  half pack_x[8], pack_y[8];
  LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    half v = __hmin(__hmax(pack_x[i], MIN_EXP_F16), MAX_EXP_F16);
    pack_y[i] = f / (f + hexp(-v));
  }
  if ((idx + 7) < N) {
    LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
  }
}

// Compile-time kernel selection: pass e.g. -DSIGMOID_F32 to nvcc to profile only that variant.
// If none are defined, all variants are enabled.
#if !defined(SIGMOID_F32) && !defined(SIGMOID_F32X4) && !defined(SIGMOID_F16) && \
    !defined(SIGMOID_F16X2) && !defined(SIGMOID_F16X8) && !defined(SIGMOID_F16X8_PACK)
#define SIGMOID_F32
#define SIGMOID_F32X4
#define SIGMOID_F16
#define SIGMOID_F16X2
#define SIGMOID_F16X8
#define SIGMOID_F16X8_PACK
#endif

int main() {
  const int N = 1 << 20;

#if defined(SIGMOID_F32) || defined(SIGMOID_F32X4)
  float *d_x_f32, *d_y_f32;
  cudaMalloc(&d_x_f32, N * sizeof(float));
  cudaMalloc(&d_y_f32, N * sizeof(float));
  cudaMemset(d_x_f32, 0, N * sizeof(float));
#endif

#if defined(SIGMOID_F16) || defined(SIGMOID_F16X2) || defined(SIGMOID_F16X8) || defined(SIGMOID_F16X8_PACK)
  half *d_x_f16, *d_y_f16;
  cudaMalloc(&d_x_f16, N * sizeof(half));
  cudaMalloc(&d_y_f16, N * sizeof(half));
  cudaMemset(d_x_f16, 0, N * sizeof(half));
#endif

#ifdef SIGMOID_F32
  // --- sigmoid_f32_kernel ---
  {
    int block = 256, grid = (N + 255) / 256;
    for (int i = 0; i < N_WARMUP; i++)
      sigmoid_f32_kernel<<<grid, block>>>(d_x_f32, d_y_f32, N);
    cudaDeviceSynchronize();
    nvtxRangePush("sigmoid_f32_kernel");
    for (int i = 0; i < N_ITER; i++)
      sigmoid_f32_kernel<<<grid, block>>>(d_x_f32, d_y_f32, N);
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

#ifdef SIGMOID_F32X4
  // --- sigmoid_f32x4_kernel ---
  {
    int block = 64, grid = (N + 255) / 256;
    for (int i = 0; i < N_WARMUP; i++)
      sigmoid_f32x4_kernel<<<grid, block>>>(d_x_f32, d_y_f32, N);
    cudaDeviceSynchronize();
    nvtxRangePush("sigmoid_f32x4_kernel");
    for (int i = 0; i < N_ITER; i++)
      sigmoid_f32x4_kernel<<<grid, block>>>(d_x_f32, d_y_f32, N);
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

#ifdef SIGMOID_F16
  // --- sigmoid_f16_kernel ---
  {
    int block = 256, grid = (N + 255) / 256;
    for (int i = 0; i < N_WARMUP; i++)
      sigmoid_f16_kernel<<<grid, block>>>(d_x_f16, d_y_f16, N);
    cudaDeviceSynchronize();
    nvtxRangePush("sigmoid_f16_kernel");
    for (int i = 0; i < N_ITER; i++)
      sigmoid_f16_kernel<<<grid, block>>>(d_x_f16, d_y_f16, N);
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

#ifdef SIGMOID_F16X2
  // --- sigmoid_f16x2_kernel ---
  {
    int block = 128, grid = (N + 255) / 256;
    for (int i = 0; i < N_WARMUP; i++)
      sigmoid_f16x2_kernel<<<grid, block>>>(d_x_f16, d_y_f16, N);
    cudaDeviceSynchronize();
    nvtxRangePush("sigmoid_f16x2_kernel");
    for (int i = 0; i < N_ITER; i++)
      sigmoid_f16x2_kernel<<<grid, block>>>(d_x_f16, d_y_f16, N);
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

#ifdef SIGMOID_F16X8
  // --- sigmoid_f16x8_kernel ---
  {
    int block = 32, grid = (N + 255) / 256;
    for (int i = 0; i < N_WARMUP; i++)
      sigmoid_f16x8_kernel<<<grid, block>>>(d_x_f16, d_y_f16, N);
    cudaDeviceSynchronize();
    nvtxRangePush("sigmoid_f16x8_kernel");
    for (int i = 0; i < N_ITER; i++)
      sigmoid_f16x8_kernel<<<grid, block>>>(d_x_f16, d_y_f16, N);
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

#ifdef SIGMOID_F16X8_PACK
  // --- sigmoid_f16x8_pack_kernel ---
  {
    int block = 32, grid = (N + 255) / 256;
    for (int i = 0; i < N_WARMUP; i++)
      sigmoid_f16x8_pack_kernel<<<grid, block>>>(d_x_f16, d_y_f16, N);
    cudaDeviceSynchronize();
    nvtxRangePush("sigmoid_f16x8_pack_kernel");
    for (int i = 0; i < N_ITER; i++)
      sigmoid_f16x8_pack_kernel<<<grid, block>>>(d_x_f16, d_y_f16, N);
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

#if defined(SIGMOID_F32) || defined(SIGMOID_F32X4)
  cudaFree(d_x_f32);
  cudaFree(d_y_f32);
#endif
#if defined(SIGMOID_F16) || defined(SIGMOID_F16X2) || defined(SIGMOID_F16X8) || defined(SIGMOID_F16X8_PACK)
  cudaFree(d_x_f16);
  cudaFree(d_y_f16);
#endif
  printf("Done.\n");
  return 0;
}
