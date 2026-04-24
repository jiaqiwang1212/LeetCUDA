#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <stdio.h>
#include <stdlib.h>

#define N_WARMUP 5
#define N_ITER 20

#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

__global__ void embedding_f32_kernel(const int *idx, float *weight,
                                     float *output, int n, int emb_size) {
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int offset = idx[bx] * emb_size;
  output[bx * emb_size + tx] = weight[offset + tx];
}

__global__ void embedding_f32x4_kernel(const int *idx, float *weight,
                                       float *output, int n, int emb_size) {
  int tx = threadIdx.x * 4;
  int bx = blockIdx.x;
  int offset = idx[bx] * emb_size;
  output[bx * emb_size + tx]     = weight[offset + tx];
  output[bx * emb_size + tx + 1] = weight[offset + tx + 1];
  output[bx * emb_size + tx + 2] = weight[offset + tx + 2];
  output[bx * emb_size + tx + 3] = weight[offset + tx + 3];
}

__global__ void embedding_f32x4_pack_kernel(const int *idx, float *weight,
                                            float *output, int n,
                                            int emb_size) {
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int offset = idx[bx] * emb_size;
  LDST128BITS(output[bx * emb_size + 4 * tx]) =
      LDST128BITS(weight[offset + 4 * tx]);
}

__global__ void embedding_f16_kernel(const int *idx, half *weight, half *output,
                                     int n, int emb_size) {
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int offset = idx[bx] * emb_size;
  output[bx * emb_size + tx] = weight[offset + tx];
}

__global__ void embedding_f16x8_kernel(const int *idx, half *weight,
                                       half *output, int n, int emb_size) {
  int tx = threadIdx.x * 8;
  int bx = blockIdx.x;
  int offset = idx[bx] * emb_size;
  output[bx * emb_size + tx]     = weight[offset + tx];
  output[bx * emb_size + tx + 1] = weight[offset + tx + 1];
  output[bx * emb_size + tx + 2] = weight[offset + tx + 2];
  output[bx * emb_size + tx + 3] = weight[offset + tx + 3];
  output[bx * emb_size + tx + 4] = weight[offset + tx + 4];
  output[bx * emb_size + tx + 5] = weight[offset + tx + 5];
  output[bx * emb_size + tx + 6] = weight[offset + tx + 6];
  output[bx * emb_size + tx + 7] = weight[offset + tx + 7];
}

__global__ void embedding_f16x8_pack_kernel(const int *idx, half *weight,
                                            half *output, int n, int emb_size) {
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int offset = idx[bx] * emb_size;
  LDST128BITS(output[bx * emb_size + 8 * tx]) =
      LDST128BITS(weight[offset + 8 * tx]);
}

// Compile-time kernel selection: pass e.g. -DEMBEDDING_F32 to nvcc to profile only that variant.
// If none are defined, all variants are enabled.
#if !defined(EMBEDDING_F32) && !defined(EMBEDDING_F32X4) && !defined(EMBEDDING_F32X4_PACK) && \
    !defined(EMBEDDING_F16) && !defined(EMBEDDING_F16X8) && !defined(EMBEDDING_F16X8_PACK)
#define EMBEDDING_F32
#define EMBEDDING_F32X4
#define EMBEDDING_F32X4_PACK
#define EMBEDDING_F16
#define EMBEDDING_F16X8
#define EMBEDDING_F16X8_PACK
#endif

int main() {
  const int S          = 1024;   // sequence length (number of indices)
  const int vocab_size = 4096;   // vocabulary size
  const int embed_dim  = 256;    // embedding dimension

  // Allocate index buffer (shared across all kernels)
  int *d_idx;
  cudaMalloc(&d_idx, S * sizeof(int));
  int *h_idx = (int *)malloc(S * sizeof(int));
  for (int i = 0; i < S; i++) h_idx[i] = i % vocab_size;
  cudaMemcpy(d_idx, h_idx, S * sizeof(int), cudaMemcpyHostToDevice);
  free(h_idx);

  // F32 weight/output buffers (used by EMBEDDING_F32, EMBEDDING_F32X4, EMBEDDING_F32X4_PACK)
#if defined(EMBEDDING_F32) || defined(EMBEDDING_F32X4) || defined(EMBEDDING_F32X4_PACK)
  float *d_weight_f32, *d_out_f32;
  cudaMalloc(&d_weight_f32, vocab_size * embed_dim * sizeof(float));
  cudaMalloc(&d_out_f32,    S          * embed_dim * sizeof(float));
  cudaMemset(d_weight_f32, 0, vocab_size * embed_dim * sizeof(float));
#endif

  // F16 weight/output buffers (used by EMBEDDING_F16, EMBEDDING_F16X8, EMBEDDING_F16X8_PACK)
#if defined(EMBEDDING_F16) || defined(EMBEDDING_F16X8) || defined(EMBEDDING_F16X8_PACK)
  half *d_weight_f16, *d_out_f16;
  cudaMalloc(&d_weight_f16, vocab_size * embed_dim * sizeof(half));
  cudaMalloc(&d_out_f16,    S          * embed_dim * sizeof(half));
  cudaMemset(d_weight_f16, 0, vocab_size * embed_dim * sizeof(half));
#endif

#ifdef EMBEDDING_F32
  // --- embedding_f32_kernel: block(embed_dim/1), grid(S) ---
  {
    dim3 block(embed_dim);
    dim3 grid(S);
    for (int i = 0; i < N_WARMUP; i++)
      embedding_f32_kernel<<<grid, block>>>(d_idx, d_weight_f32, d_out_f32, S, embed_dim);
    cudaDeviceSynchronize();
    nvtxRangePush("embedding_f32_kernel");
    for (int i = 0; i < N_ITER; i++)
      embedding_f32_kernel<<<grid, block>>>(d_idx, d_weight_f32, d_out_f32, S, embed_dim);
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

#ifdef EMBEDDING_F32X4
  // --- embedding_f32x4_kernel: block(embed_dim/4), grid(S) ---
  {
    dim3 block(embed_dim / 4);
    dim3 grid(S);
    for (int i = 0; i < N_WARMUP; i++)
      embedding_f32x4_kernel<<<grid, block>>>(d_idx, d_weight_f32, d_out_f32, S, embed_dim);
    cudaDeviceSynchronize();
    nvtxRangePush("embedding_f32x4_kernel");
    for (int i = 0; i < N_ITER; i++)
      embedding_f32x4_kernel<<<grid, block>>>(d_idx, d_weight_f32, d_out_f32, S, embed_dim);
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

#ifdef EMBEDDING_F32X4_PACK
  // --- embedding_f32x4_pack_kernel: block(embed_dim/4), grid(S) ---
  {
    dim3 block(embed_dim / 4);
    dim3 grid(S);
    for (int i = 0; i < N_WARMUP; i++)
      embedding_f32x4_pack_kernel<<<grid, block>>>(d_idx, d_weight_f32, d_out_f32, S, embed_dim);
    cudaDeviceSynchronize();
    nvtxRangePush("embedding_f32x4_pack_kernel");
    for (int i = 0; i < N_ITER; i++)
      embedding_f32x4_pack_kernel<<<grid, block>>>(d_idx, d_weight_f32, d_out_f32, S, embed_dim);
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

#ifdef EMBEDDING_F16
  // --- embedding_f16_kernel: block(embed_dim/1), grid(S) ---
  {
    dim3 block(embed_dim);
    dim3 grid(S);
    for (int i = 0; i < N_WARMUP; i++)
      embedding_f16_kernel<<<grid, block>>>(d_idx, d_weight_f16, d_out_f16, S, embed_dim);
    cudaDeviceSynchronize();
    nvtxRangePush("embedding_f16_kernel");
    for (int i = 0; i < N_ITER; i++)
      embedding_f16_kernel<<<grid, block>>>(d_idx, d_weight_f16, d_out_f16, S, embed_dim);
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

#ifdef EMBEDDING_F16X8
  // --- embedding_f16x8_kernel: block(embed_dim/8), grid(S) ---
  {
    dim3 block(embed_dim / 8);
    dim3 grid(S);
    for (int i = 0; i < N_WARMUP; i++)
      embedding_f16x8_kernel<<<grid, block>>>(d_idx, d_weight_f16, d_out_f16, S, embed_dim);
    cudaDeviceSynchronize();
    nvtxRangePush("embedding_f16x8_kernel");
    for (int i = 0; i < N_ITER; i++)
      embedding_f16x8_kernel<<<grid, block>>>(d_idx, d_weight_f16, d_out_f16, S, embed_dim);
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

#ifdef EMBEDDING_F16X8_PACK
  // --- embedding_f16x8_pack_kernel: block(embed_dim/8), grid(S) ---
  {
    dim3 block(embed_dim / 8);
    dim3 grid(S);
    for (int i = 0; i < N_WARMUP; i++)
      embedding_f16x8_pack_kernel<<<grid, block>>>(d_idx, d_weight_f16, d_out_f16, S, embed_dim);
    cudaDeviceSynchronize();
    nvtxRangePush("embedding_f16x8_pack_kernel");
    for (int i = 0; i < N_ITER; i++)
      embedding_f16x8_pack_kernel<<<grid, block>>>(d_idx, d_weight_f16, d_out_f16, S, embed_dim);
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

  cudaFree(d_idx);
#if defined(EMBEDDING_F32) || defined(EMBEDDING_F32X4) || defined(EMBEDDING_F32X4_PACK)
  cudaFree(d_weight_f32);
  cudaFree(d_out_f32);
#endif
#if defined(EMBEDDING_F16) || defined(EMBEDDING_F16X8) || defined(EMBEDDING_F16X8_PACK)
  cudaFree(d_weight_f16);
  cudaFree(d_out_f16);
#endif
  printf("Done.\n");
  return 0;
}
