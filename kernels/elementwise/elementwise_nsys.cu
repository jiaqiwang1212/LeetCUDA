#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <stdio.h>
#include <stdlib.h>

// Reinterpret a scalar address as a wider vector pointer for a single
// 128-bit (float4) or 64-bit (half2) aligned load/store instruction.
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// Warmup runs bring the GPU to steady-state clocks and warm the L2 cache
// before the timed NVTX region begins.
#define N_WARMUP 5
#define N_ITER 20

// FP32
__global__ void elementwise_add_f32_kernel(float *a, float *b, float *c, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    c[idx] = a[idx] + b[idx];
}

__global__ void elementwise_add_f32x4_kernel(float *a, float *b, float *c, int N) {
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    float4 reg_a = FLOAT4(a[idx]);
    float4 reg_b = FLOAT4(b[idx]);
    float4 reg_c;
    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    reg_c.z = reg_a.z + reg_b.z;
    reg_c.w = reg_a.w + reg_b.w;
    FLOAT4(c[idx]) = reg_c;
  }
}

// FP16
__global__ void elementwise_add_f16_kernel(half *a, half *b, half *c, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    c[idx] = __hadd(a[idx], b[idx]);
}

__global__ void elementwise_add_f16x2_kernel(half *a, half *b, half *c, int N) {
  int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    half2 reg_a = HALF2(a[idx]);
    half2 reg_b = HALF2(b[idx]);
    half2 reg_c;
    reg_c.x = __hadd(reg_a.x, reg_b.x);
    reg_c.y = __hadd(reg_a.y, reg_b.y);
    HALF2(c[idx]) = reg_c;
  }
}

__global__ void elementwise_add_f16x8_kernel(half *a, half *b, half *c, int N) {
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  // Issue all 4 load pairs before the bounds checks so the compiler can
  // overlap memory requests (memory-level parallelism); writes are guarded below.
  half2 reg_a_0 = HALF2(a[idx + 0]);
  half2 reg_a_1 = HALF2(a[idx + 2]);
  half2 reg_a_2 = HALF2(a[idx + 4]);
  half2 reg_a_3 = HALF2(a[idx + 6]);
  half2 reg_b_0 = HALF2(b[idx + 0]);
  half2 reg_b_1 = HALF2(b[idx + 2]);
  half2 reg_b_2 = HALF2(b[idx + 4]);
  half2 reg_b_3 = HALF2(b[idx + 6]);
  half2 reg_c_0, reg_c_1, reg_c_2, reg_c_3;
  reg_c_0.x = __hadd(reg_a_0.x, reg_b_0.x);
  reg_c_0.y = __hadd(reg_a_0.y, reg_b_0.y);
  reg_c_1.x = __hadd(reg_a_1.x, reg_b_1.x);
  reg_c_1.y = __hadd(reg_a_1.y, reg_b_1.y);
  reg_c_2.x = __hadd(reg_a_2.x, reg_b_2.x);
  reg_c_2.y = __hadd(reg_a_2.y, reg_b_2.y);
  reg_c_3.x = __hadd(reg_a_3.x, reg_b_3.x);
  reg_c_3.y = __hadd(reg_a_3.y, reg_b_3.y);
  if ((idx + 0) < N) {
    HALF2(c[idx + 0]) = reg_c_0;
  }
  if ((idx + 2) < N) {
    HALF2(c[idx + 2]) = reg_c_1;
  }
  if ((idx + 4) < N) {
    HALF2(c[idx + 4]) = reg_c_2;
  }
  if ((idx + 6) < N) {
    HALF2(c[idx + 6]) = reg_c_3;
  }
}

__global__ void elementwise_add_f16x8_pack_kernel(half *a, half *b, half *c, int N) {
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  half pack_a[8], pack_b[8], pack_c[8];
  // Single 128-bit load (LDG.E.128) fetches all 8 halfs in one transaction,
  // vs. the 4 separate 32-bit loads in elementwise_add_f16x8_kernel.
  LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);
  LDST128BITS(pack_b[0]) = LDST128BITS(b[idx]);

#pragma unroll
  for (int i = 0; i < 8; i += 2) {
    // __hadd2 fuses two fp16 additions into one SIMD instruction.
    HALF2(pack_c[i]) = __hadd2(HALF2(pack_a[i]), HALF2(pack_b[i]));
  }
  if ((idx + 7) < N) {
    LDST128BITS(c[idx]) = LDST128BITS(pack_c[0]);
  } else {
    // Tail: scalar fallback avoids writing past the end of the output buffer.
    for (int i = 0; idx + i < N; i++) {
      c[idx + i] = __hadd(a[idx + i], b[idx + i]);
    }
  }
}

// Compile-time kernel selection: pass e.g. -DELEMENTWISE_F32 to nvcc to profile only that variant.
// If none are defined, all variants are enabled.
#if !defined(ELEMENTWISE_F32) && !defined(ELEMENTWISE_F32X4) && !defined(ELEMENTWISE_F16) && \
    !defined(ELEMENTWISE_F16X2) && !defined(ELEMENTWISE_F16X8) && !defined(ELEMENTWISE_F16X8_PACK)
#define ELEMENTWISE_F32
#define ELEMENTWISE_F32X4
#define ELEMENTWISE_F16
#define ELEMENTWISE_F16X2
#define ELEMENTWISE_F16X8
#define ELEMENTWISE_F16X8_PACK
#endif

int main() {
  const int N = 1 << 20;

#if defined(ELEMENTWISE_F32) || defined(ELEMENTWISE_F32X4)
  // FP32 buffers
  float *d_a_f32, *d_b_f32, *d_c_f32;
  cudaMalloc(&d_a_f32, N * sizeof(float));
  cudaMalloc(&d_b_f32, N * sizeof(float));
  cudaMalloc(&d_c_f32, N * sizeof(float));

  // Initialize f32 inputs
  float *h_a_f32 = (float *)malloc(N * sizeof(float));
  float *h_b_f32 = (float *)malloc(N * sizeof(float));
  for (int i = 0; i < N; i++) {
    h_a_f32[i] = (float)(i % 256) - 128.0f;
    h_b_f32[i] = (float)(i % 128);
  }
  cudaMemcpy(d_a_f32, h_a_f32, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b_f32, h_b_f32, N * sizeof(float), cudaMemcpyHostToDevice);
  free(h_a_f32);
  free(h_b_f32);
#endif

#if defined(ELEMENTWISE_F16) || defined(ELEMENTWISE_F16X2) || defined(ELEMENTWISE_F16X8) || defined(ELEMENTWISE_F16X8_PACK)
  // FP16 buffers
  half *d_a_f16, *d_b_f16, *d_c_f16;
  cudaMalloc(&d_a_f16, N * sizeof(half));
  cudaMalloc(&d_b_f16, N * sizeof(half));
  cudaMalloc(&d_c_f16, N * sizeof(half));

  // Initialize f16 inputs
  half *h_a_f16 = (half *)malloc(N * sizeof(half));
  half *h_b_f16 = (half *)malloc(N * sizeof(half));
  for (int i = 0; i < N; i++) {
    h_a_f16[i] = __float2half((float)(i % 256) - 128.0f);
    h_b_f16[i] = __float2half((float)(i % 128));
  }
  cudaMemcpy(d_a_f16, h_a_f16, N * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b_f16, h_b_f16, N * sizeof(half), cudaMemcpyHostToDevice);
  free(h_a_f16);
  free(h_b_f16);
#endif

#ifdef ELEMENTWISE_F32
  // ------------------------------------------------------------------
  // elementwise_add_f32_kernel: block=256, grid=(N+255)/256
  {
    const int block = 256;
    const int grid = (N + 255) / 256;
    for (int i = 0; i < N_WARMUP; i++)
      elementwise_add_f32_kernel<<<grid, block>>>(d_a_f32, d_b_f32, d_c_f32, N);
    cudaDeviceSynchronize();
    nvtxRangePush("elementwise_add_f32_kernel");
    for (int i = 0; i < N_ITER; i++)
      elementwise_add_f32_kernel<<<grid, block>>>(d_a_f32, d_b_f32, d_c_f32, N);
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

#ifdef ELEMENTWISE_F32X4
  // ------------------------------------------------------------------
  // elementwise_add_f32x4_kernel: block=64, grid=(N+255)/256
  // block is 1/4 of the scalar variant because each thread covers 4 elements;
  // the grid stays the same so total element coverage is identical.
  {
    const int block = 64;
    const int grid = (N + 255) / 256;
    for (int i = 0; i < N_WARMUP; i++)
      elementwise_add_f32x4_kernel<<<grid, block>>>(d_a_f32, d_b_f32, d_c_f32, N);
    cudaDeviceSynchronize();
    nvtxRangePush("elementwise_add_f32x4_kernel");
    for (int i = 0; i < N_ITER; i++)
      elementwise_add_f32x4_kernel<<<grid, block>>>(d_a_f32, d_b_f32, d_c_f32, N);
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

#ifdef ELEMENTWISE_F16
  // ------------------------------------------------------------------
  // elementwise_add_f16_kernel: block=256, grid=(N+255)/256
  {
    const int block = 256;
    const int grid = (N + 255) / 256;
    for (int i = 0; i < N_WARMUP; i++)
      elementwise_add_f16_kernel<<<grid, block>>>(d_a_f16, d_b_f16, d_c_f16, N);
    cudaDeviceSynchronize();
    nvtxRangePush("elementwise_add_f16_kernel");
    for (int i = 0; i < N_ITER; i++)
      elementwise_add_f16_kernel<<<grid, block>>>(d_a_f16, d_b_f16, d_c_f16, N);
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

#ifdef ELEMENTWISE_F16X2
  // ------------------------------------------------------------------
  // elementwise_add_f16x2_kernel: block=128, grid=(N+255)/256
  {
    const int block = 128;
    const int grid = (N + 255) / 256;
    for (int i = 0; i < N_WARMUP; i++)
      elementwise_add_f16x2_kernel<<<grid, block>>>(d_a_f16, d_b_f16, d_c_f16, N);
    cudaDeviceSynchronize();
    nvtxRangePush("elementwise_add_f16x2_kernel");
    for (int i = 0; i < N_ITER; i++)
      elementwise_add_f16x2_kernel<<<grid, block>>>(d_a_f16, d_b_f16, d_c_f16, N);
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

#ifdef ELEMENTWISE_F16X8
  // ------------------------------------------------------------------
  // elementwise_add_f16x8_kernel: block=32, grid=(N+255)/256
  {
    const int block = 32;
    const int grid = (N + 255) / 256;
    for (int i = 0; i < N_WARMUP; i++)
      elementwise_add_f16x8_kernel<<<grid, block>>>(d_a_f16, d_b_f16, d_c_f16, N);
    cudaDeviceSynchronize();
    nvtxRangePush("elementwise_add_f16x8_kernel");
    for (int i = 0; i < N_ITER; i++)
      elementwise_add_f16x8_kernel<<<grid, block>>>(d_a_f16, d_b_f16, d_c_f16, N);
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

#ifdef ELEMENTWISE_F16X8_PACK
  // ------------------------------------------------------------------
  // elementwise_add_f16x8_pack_kernel: block=32, grid=(N+255)/256
  {
    const int block = 32;
    const int grid = (N + 255) / 256;
    for (int i = 0; i < N_WARMUP; i++)
      elementwise_add_f16x8_pack_kernel<<<grid, block>>>(d_a_f16, d_b_f16, d_c_f16, N);
    cudaDeviceSynchronize();
    nvtxRangePush("elementwise_add_f16x8_pack_kernel");
    for (int i = 0; i < N_ITER; i++)
      elementwise_add_f16x8_pack_kernel<<<grid, block>>>(d_a_f16, d_b_f16, d_c_f16, N);
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

#if defined(ELEMENTWISE_F32) || defined(ELEMENTWISE_F32X4)
  cudaFree(d_a_f32);
  cudaFree(d_b_f32);
  cudaFree(d_c_f32);
#endif
#if defined(ELEMENTWISE_F16) || defined(ELEMENTWISE_F16X2) || defined(ELEMENTWISE_F16X8) || defined(ELEMENTWISE_F16X8_PACK)
  cudaFree(d_a_f16);
  cudaFree(d_b_f16);
  cudaFree(d_c_f16);
#endif
  printf("Done.\n");
  return 0;
}
