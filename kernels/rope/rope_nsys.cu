#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <stdio.h>

#define N_WARMUP 5
#define N_ITER 20

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define BLOCK_SIZE 256
#define theta 10000.0f

__global__ void rope_f32_kernel(float *x, float *out, int seq_len, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float x1 = x[idx * 2];
  float x2 = x[idx * 2 + 1];
  int token_pos = idx / N;
  int token_idx = idx % N;
  float exp_v = 1.0f / powf(theta, 2 * token_idx / (N * 2.0f));
  float sin_v = sinf(token_pos * exp_v);
  float cos_v = cosf(token_pos * exp_v);
  float out1 = x1 * cos_v - x2 * sin_v;
  float out2 = x1 * sin_v + x2 * cos_v;
  out[idx * 2] = out1;
  out[idx * 2 + 1] = out2;
}

// another index method of rope.
__global__ void rope_f32_v2_kernel(float *x, float *out, int seq_len, int N) {
  int token_pos = blockIdx.x;
  int tid = threadIdx.x;
  float x1 = x[token_pos * N * 2 + tid * 2];
  float x2 = x[token_pos * N * 2 + tid * 2 + 1];
  float exp_v = 1.0f / powf(theta, 2 * tid / (N * 2.0f));
  float sin_v = sinf(token_pos * exp_v);
  float cos_v = cosf(token_pos * exp_v);
  float out1 = x1 * cos_v - x2 * sin_v;
  float out2 = x1 * sin_v + x2 * cos_v;
  out[token_pos * N * 2 + tid * 2] = out1;
  out[token_pos * N * 2 + tid * 2 + 1] = out2;
}

__global__ void rope_f32x4_pack_kernel(float *x, float *out, int seq_len,
                                       int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float4 x_v = FLOAT4(x[idx * 4]);
  int token_pos = idx / N;
  int token_idx = idx % N;
  float exp_f_v = 1.0f / powf(theta, 2 * token_idx * 2 / (N * 4.0f));
  float exp_s_v = 1.0f / powf(theta, 2 * (token_idx * 2 + 1) / (N * 4.0f));
  float sin_f_v = sinf(token_pos * exp_f_v);
  float cos_f_v = cosf(token_pos * exp_f_v);
  float sin_s_v = sinf(token_pos * exp_s_v);
  float cos_s_v = cosf(token_pos * exp_s_v);
  float4 out_v;
  out_v.x = x_v.x * cos_f_v - x_v.y * sin_f_v;
  out_v.y = x_v.x * sin_f_v + x_v.y * cos_f_v;
  out_v.z = x_v.z * cos_s_v - x_v.w * sin_s_v;
  out_v.w = x_v.z * sin_s_v + x_v.w * cos_s_v;
  FLOAT4(out[idx * 4]) = out_v;
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
#if !defined(ROPE_F32) && !defined(ROPE_F32_V2) && !defined(ROPE_F32X4_PACK)
#define ROPE_F32
#define ROPE_F32_V2
#define ROPE_F32X4_PACK
#endif

int main() {
  // Tensor shape: (head_num, seq_len, head_dim), all float
  const int head_num = 8;
  const int seq_len  = 512;
  const int head_dim = 128;
  // Total floats = head_num * seq_len * head_dim
  const int N_elem = head_num * seq_len * head_dim;

#if defined(ROPE_F32) || defined(ROPE_F32_V2) || defined(ROPE_F32X4_PACK)
  float *d_x, *d_out;
  cudaMalloc(&d_x,   N_elem * sizeof(float));
  cudaMalloc(&d_out, N_elem * sizeof(float));
  cudaMemset(d_x,   0, N_elem * sizeof(float));
  cudaMemset(d_out, 0, N_elem * sizeof(float));
#endif

  // rope_f32_kernel: each thread handles one (x1, x2) pair.
  // N = head_dim / 2; total threads = head_num * seq_len * N.
#ifdef ROPE_F32
  {
    const int N             = head_dim / 2; // 64
    const int total_threads = head_num * seq_len * N;
    dim3 grid((total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);
    RUN_KERNEL("rope_f32_kernel",
      rope_f32_kernel<<<grid, block>>>(d_x, d_out, seq_len, N));
  }
#endif

  // rope_f32_v2_kernel: grid(seq_len), block(N) where N = head_dim / 2.
  // Processes one head slice (seq_len tokens, N pairs each).
#ifdef ROPE_F32_V2
  {
    const int N = head_dim / 2; // 64
    dim3 grid(seq_len);
    dim3 block(N);
    RUN_KERNEL("rope_f32_v2_kernel",
      rope_f32_v2_kernel<<<grid, block>>>(d_x, d_out, seq_len, N));
  }
#endif

  // rope_f32x4_pack_kernel: each thread handles 4 floats (2 pairs).
  // N = head_dim / 4; total threads = head_num * seq_len * N.
#ifdef ROPE_F32X4_PACK
  {
    const int N             = head_dim / 4; // 32
    const int total_threads = head_num * seq_len * N;
    dim3 grid((total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);
    RUN_KERNEL("rope_f32x4_pack_kernel",
      rope_f32x4_pack_kernel<<<grid, block>>>(d_x, d_out, seq_len, N));
  }
#endif

#if defined(ROPE_F32) || defined(ROPE_F32_V2) || defined(ROPE_F32X4_PACK)
  cudaFree(d_x);
  cudaFree(d_out);
#endif

  printf("Done.\n");
  return 0;
}
