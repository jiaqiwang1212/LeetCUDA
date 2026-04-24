#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <nvtx3/nvToolsExt.h>
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
// FP32 warp/block reduce helpers
// ---------------------------------------------------------------------------
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

template <const int NUM_THREADS = 256>
__device__ float block_reduce_sum_f32(float val) {
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  static __shared__ float shared[NUM_WARPS];
  val = warp_reduce_sum_f32<WARP_SIZE>(val);
  if (lane == 0)
    shared[warp] = val;
  __syncthreads();
  val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
  val = warp_reduce_sum_f32<NUM_WARPS>(val);
  return val;
}

// ---------------------------------------------------------------------------
// FP16 warp/block reduce helpers
// ---------------------------------------------------------------------------
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_f16_f16(half val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
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

template <const int NUM_THREADS = 256>
__device__ half block_reduce_sum_f16_f16(half val) {
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  static __shared__ half shared[NUM_WARPS];
  val = warp_reduce_sum_f16_f16<WARP_SIZE>(val);
  if (lane == 0)
    shared[warp] = val;
  __syncthreads();
  val = (lane < NUM_WARPS) ? shared[lane] : __float2half(0.0f);
  val = warp_reduce_sum_f16_f16<NUM_WARPS>(val);
  return val;
}

template <const int NUM_THREADS = 256>
__device__ float block_reduce_sum_f16_f32(half val) {
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  static __shared__ float shared[NUM_WARPS];
  float val_f32 = warp_reduce_sum_f16_f32<WARP_SIZE>(val);
  if (lane == 0)
    shared[warp] = val_f32;
  __syncthreads();
  val_f32 = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
  val_f32 = warp_reduce_sum_f32<NUM_WARPS>(val_f32);
  return val_f32;
}

// ---------------------------------------------------------------------------
// Kernels (verbatim from layer_norm.cu)
// ---------------------------------------------------------------------------

// Layer Norm: x: NxK(K=256<1024), y': NxK, y'=x-mean(x)/std(x) each row
// mean(x) = sum(x)/K, 1/std(x) = rsqrtf( sum( (x-mean(x))^2 )/K ) each row
// grid(N*K/K), block(K<1024) N=batch_size*seq_len, K=hidden_size
// y=y'*g + b (g: scale, b: bias)
template <const int NUM_THREADS = 256>
__global__ void layer_norm_f32_kernel(float *x, float *y, float g, float b,
                                      int N, int K) {
  int tid = threadIdx.x; // 0..K-1
  int bid = blockIdx.x;  // 0..N-1
  int idx = bid * blockDim.x + threadIdx.x;
  const float epsilon = 1e-5f;

  __shared__ float s_mean;                     // shared within block
  __shared__ float s_variance;                 // shared within block
  float value = (idx < N * K) ? x[idx] : 0.0f; // load once only
  float sum = block_reduce_sum_f32<NUM_THREADS>(value);
  if (tid == 0)
    s_mean = sum / (float)K;
  __syncthreads();
  float variance = (value - s_mean) * (value - s_mean);
  variance = block_reduce_sum_f32<NUM_THREADS>(variance);
  if (tid == 0)
    s_variance = rsqrtf(variance / (float)K + epsilon);
  __syncthreads();
  if (idx < N * K)
    y[idx] = ((value - s_mean) * s_variance) * g + b;
}

// Layer Norm Vec4: x: NxK(K=256<1024), y': NxK, y'=x-mean(x)/std(x) each row
// mean(x) = sum(x)/K, 1/std(x) = rsqrtf( sum( (x-mean(x))^2 )/K ) each row
// grid(N*K/K), block(K/4<1024) N=batch_size*seq_len, K=hidden_size
// y=y'*g + b (g: scale, b: bias)
template <const int NUM_THREADS = 256 / 4>
__global__ void layer_norm_f32x4_kernel(float *x, float *y, float g, float b,
                                        int N, int K) {
  int tid = threadIdx.x; // 0..K-1
  int bid = blockIdx.x;  // 0..N-1
  int idx = (bid * blockDim.x + threadIdx.x) * 4;
  const float epsilon = 1e-5f;

  __shared__ float s_mean;     // shared within block
  __shared__ float s_variance; // shared within block
  float4 reg_x = FLOAT4(x[idx]);
  float value = (idx < N * K) ? (reg_x.x + reg_x.y + reg_x.z + reg_x.w) : 0.0f;
  float sum = block_reduce_sum_f32<NUM_THREADS>(value);
  if (tid == 0)
    s_mean = sum / (float)K;
  __syncthreads();
  float4 reg_x_hat;
  reg_x_hat.x = reg_x.x - s_mean;
  reg_x_hat.y = reg_x.y - s_mean;
  reg_x_hat.z = reg_x.z - s_mean;
  reg_x_hat.w = reg_x.w - s_mean;
  float variance = reg_x_hat.x * reg_x_hat.x + reg_x_hat.y * reg_x_hat.y +
                   reg_x_hat.z * reg_x_hat.z + reg_x_hat.w * reg_x_hat.w;
  variance = block_reduce_sum_f32<NUM_THREADS>(variance);
  if (tid == 0)
    s_variance = rsqrtf(variance / (float)K + epsilon);
  __syncthreads();
  float4 reg_y;
  reg_y.x = reg_x_hat.x * s_variance * g + b;
  reg_y.y = reg_x_hat.y * s_variance * g + b;
  reg_y.z = reg_x_hat.z * s_variance * g + b;
  reg_y.w = reg_x_hat.w * s_variance * g + b;
  if (idx < N * K)
    FLOAT4(y[idx]) = reg_y;
}

template <const int NUM_THREADS = 256>
__global__ void layer_norm_f16_f16_kernel(half *x, half *y, float g, float b,
                                          int N, int K) {
  int tid = threadIdx.x; // 0..K-1
  int bid = blockIdx.x;  // 0..N-1
  int idx = bid * blockDim.x + threadIdx.x;
  const half epsilon = __float2half(1e-5f);
  const half g_ = __float2half(g);
  const half b_ = __float2half(b);
  const half K_ = __int2half_rn(K);

  __shared__ half s_mean;     // shared within block
  __shared__ half s_variance; // shared within block
  half value = (idx < N * K) ? x[idx] : __float2half(0.0f);
  half sum = block_reduce_sum_f16_f16<NUM_THREADS>(value);
  if (tid == 0)
    s_mean = sum / K_;
  __syncthreads();
  half variance = (value - s_mean) * (value - s_mean);
  variance = block_reduce_sum_f16_f16<NUM_THREADS>(variance);
  if (tid == 0)
    s_variance = hrsqrt(variance / K_ + epsilon);
  __syncthreads();
  if (idx < N * K) {
    y[idx] = __hfma((value - s_mean) * s_variance, g_, b_);
  }
}

template <const int NUM_THREADS = 256>
__global__ void layer_norm_f16x2_f16_kernel(half *x, half *y, float g, float b,
                                            int N, int K) {
  int tid = threadIdx.x; // 0..K-1
  int bid = blockIdx.x;  // 0..N-1
  int idx = (bid * blockDim.x + threadIdx.x) * 2;
  const half epsilon = __float2half(1e-5f);
  const half g_ = __float2half(g);
  const half b_ = __float2half(b);
  const half K_ = __int2half_rn(K);

  __shared__ half s_mean;     // shared within block
  __shared__ half s_variance; // shared within block
  half2 reg_x = HALF2(x[idx]);
  half value = (idx < N * K) ? (reg_x.x + reg_x.y) : __float2half(0.0f);
  half sum = block_reduce_sum_f16_f16<NUM_THREADS>(value);
  if (tid == 0)
    s_mean = sum / K_;
  __syncthreads();
  half2 reg_x_hat;
  reg_x_hat.x = reg_x.x - s_mean;
  reg_x_hat.y = reg_x.y - s_mean;
  half variance = reg_x_hat.x * reg_x_hat.x + reg_x_hat.y * reg_x_hat.y;
  variance = block_reduce_sum_f16_f16<NUM_THREADS>(variance);
  if (tid == 0)
    s_variance = hrsqrt(variance / K_ + epsilon);
  __syncthreads();
  if (idx < N * K) {
    half2 reg_y;
    reg_y.x = __hfma(reg_x_hat.x * s_variance, g_, b_);
    reg_y.y = __hfma(reg_x_hat.y * s_variance, g_, b_);
    HALF2(y[idx]) = reg_y;
  }
}

#define HALF2_SUM(reg, i)                                                      \
  (((idx + (i)) < N * K) ? ((reg).x + (reg).y) : __float2half(0.0f))

#define HALF2_SUB(reg_y, reg_x)                                                \
  (reg_y).x = (reg_x).x - s_mean;                                              \
  (reg_y).y = (reg_x).y - s_mean;

#define HALF2_VARIANCE(reg, i)                                                 \
  (((idx + (i)) < N * K) ? ((reg).x * (reg).x + (reg).y * (reg).y)             \
                         : __float2half(0.0f))

#define HALF2_LAYER_NORM(reg_y, reg_x, g_, b_)                                 \
  (reg_y).x = __hfma((reg_x).x * s_variance, g_, b_);                          \
  (reg_y).y = __hfma((reg_x).y * s_variance, g_, b_);

template <const int NUM_THREADS = 256>
__global__ void layer_norm_f16x8_f16_kernel(half *x, half *y, float g, float b,
                                            int N, int K) {
  int tid = threadIdx.x; // 0..K-1
  int bid = blockIdx.x;  // 0..N-1
  int idx = (bid * blockDim.x + threadIdx.x) * 8;
  const half epsilon = __float2half(1e-5f);
  const half g_ = __float2half(g);
  const half b_ = __float2half(b);
  const half K_ = __int2half_rn(K);

  __shared__ half s_mean;     // shared within block
  __shared__ half s_variance; // shared within block
  half2 reg_x_0 = HALF2(x[idx + 0]);
  half2 reg_x_1 = HALF2(x[idx + 2]);
  half2 reg_x_2 = HALF2(x[idx + 4]);
  half2 reg_x_3 = HALF2(x[idx + 6]);

  half value = HALF2_SUM(reg_x_0, 0);
  value += HALF2_SUM(reg_x_1, 2);
  value += HALF2_SUM(reg_x_2, 4);
  value += HALF2_SUM(reg_x_3, 6);

  half sum = block_reduce_sum_f16_f16<NUM_THREADS>(value);
  if (tid == 0)
    s_mean = sum / K_;
  __syncthreads();
  half2 reg_x_hat_0, reg_x_hat_1, reg_x_hat_2, reg_x_hat_3;
  HALF2_SUB(reg_x_hat_0, reg_x_0);
  HALF2_SUB(reg_x_hat_1, reg_x_1);
  HALF2_SUB(reg_x_hat_2, reg_x_2);
  HALF2_SUB(reg_x_hat_3, reg_x_3);

  half variance = HALF2_VARIANCE(reg_x_hat_0, 0);
  variance += HALF2_VARIANCE(reg_x_hat_1, 2);
  variance += HALF2_VARIANCE(reg_x_hat_2, 4);
  variance += HALF2_VARIANCE(reg_x_hat_3, 6);

  variance = block_reduce_sum_f16_f16<NUM_THREADS>(variance);
  if (tid == 0)
    s_variance = hrsqrt(variance / K_ + epsilon);
  __syncthreads();
  half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;
  HALF2_LAYER_NORM(reg_y_0, reg_x_hat_0, g_, b_);
  HALF2_LAYER_NORM(reg_y_1, reg_x_hat_1, g_, b_);
  HALF2_LAYER_NORM(reg_y_2, reg_x_hat_2, g_, b_);
  HALF2_LAYER_NORM(reg_y_3, reg_x_hat_3, g_, b_);

  if ((idx + 0) < N * K) {
    HALF2(y[idx + 0]) = reg_y_0;
  }
  if ((idx + 2) < N * K) {
    HALF2(y[idx + 2]) = reg_y_1;
  }
  if ((idx + 4) < N * K) {
    HALF2(y[idx + 4]) = reg_y_2;
  }
  if ((idx + 6) < N * K) {
    HALF2(y[idx + 6]) = reg_y_3;
  }
}

template <const int NUM_THREADS = 256>
__global__ void layer_norm_f16_f32_kernel(half *x, half *y, float g, float b,
                                          int N, int K) {
  int tid = threadIdx.x; // 0..K-1
  int bid = blockIdx.x;  // 0..N-1
  int idx = bid * blockDim.x + threadIdx.x;
  const float epsilon = 1e-5f;

  __shared__ float s_mean;     // shared within block
  __shared__ float s_variance; // shared within block
  float value = (idx < N * K) ? __half2float(x[idx]) : 0.0f;
  float sum = block_reduce_sum_f32<NUM_THREADS>(value);
  if (tid == 0)
    s_mean = sum / (float)K;
  __syncthreads();
  float variance = (value - s_mean) * (value - s_mean);
  variance = block_reduce_sum_f32<NUM_THREADS>(variance);
  if (tid == 0)
    s_variance = rsqrtf(variance / (float)K + epsilon);
  __syncthreads();
  if (idx < N * K) {
    y[idx] = __float2half(__fmaf_rn(((value - s_mean) * s_variance), g, b));
  }
}

template <const int NUM_THREADS = 256>
__global__ void layer_norm_f16x8_pack_f16_kernel(half *x, half *y, float g,
                                                 float b, int N, int K) {
  int tid = threadIdx.x; // 0..K-1
  int bid = blockIdx.x;  // 0..N-1
  int idx = (bid * blockDim.x + threadIdx.x) * 8;
  const half epsilon = __float2half(1e-5f);
  const half g_ = __float2half(g);
  const half b_ = __float2half(b);
  const half K_ = __int2half_rn(K);
  const half z_ = __float2half(0.0f);

  __shared__ half s_mean;     // shared within block
  __shared__ half s_variance; // shared within block
  half pack_x[8], pack_y[8];
  LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);

  half value = z_;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    value += ((idx + i) < N * K ? pack_x[i] : z_);
  }
  half sum = block_reduce_sum_f16_f16<NUM_THREADS>(value);
  if (tid == 0)
    s_mean = sum / K_;
  __syncthreads();

  half variance = z_;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    half v_hat = pack_x[i] - s_mean;
    variance += ((idx + i) < N * K ? v_hat * v_hat : z_);
  }
  variance = block_reduce_sum_f16_f16<NUM_THREADS>(variance);
  if (tid == 0)
    s_variance = hrsqrt(variance / K_ + epsilon);
  __syncthreads();

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    pack_y[i] = __hfma((pack_x[i] - s_mean) * s_variance, g_, b_);
  }
  if ((idx + 7) < N * K) {
    LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
  }
}

template <const int NUM_THREADS = 256>
__global__ void layer_norm_f16x8_pack_f32_kernel(half *x, half *y, float g,
                                                 float b, int N, int K) {
  int tid = threadIdx.x; // 0..K-1
  int bid = blockIdx.x;  // 0..N-1
  int idx = (bid * blockDim.x + threadIdx.x) * 8;
  const float epsilon = 1e-5f;

  __shared__ float s_mean;     // shared within block
  __shared__ float s_variance; // shared within block
  half pack_x[8], pack_y[8];
  LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);

  float value = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    value += ((idx + i) < N * K ? __half2float(pack_x[i]) : 0.0f);
  }
  float sum = block_reduce_sum_f32<NUM_THREADS>(value);
  if (tid == 0)
    s_mean = sum / (float)K;
  __syncthreads();

  float variance = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float v_hat = __half2float(pack_x[i]) - s_mean;
    variance += ((idx + i) < N * K ? v_hat * v_hat : 0.0f);
  }
  variance = block_reduce_sum_f32<NUM_THREADS>(variance);
  if (tid == 0)
    s_variance = rsqrtf(variance / (float)K + epsilon);
  __syncthreads();

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    pack_y[i] = __float2half(
        __fmaf_rn(((__half2float(pack_x[i]) - s_mean) * s_variance), g, b));
  }
  if ((idx + 7) < N * K) {
    LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
  }
}

// Compile-time kernel selection: pass e.g. -DLAYERNORM_F32 to nvcc to profile only that variant.
// If none are defined, all variants are enabled.
#if !defined(LAYERNORM_F32) && !defined(LAYERNORM_F32X4) && \
    !defined(LAYERNORM_F16_F16) && !defined(LAYERNORM_F16X2_F16) && \
    !defined(LAYERNORM_F16X8_F16) && !defined(LAYERNORM_F16_F32) && \
    !defined(LAYERNORM_F16X8_PACK_F16) && !defined(LAYERNORM_F16X8_PACK_F32)
#define LAYERNORM_F32
#define LAYERNORM_F32X4
#define LAYERNORM_F16_F16
#define LAYERNORM_F16X2_F16
#define LAYERNORM_F16X8_F16
#define LAYERNORM_F16_F32
#define LAYERNORM_F16X8_PACK_F16
#define LAYERNORM_F16X8_PACK_F32
#endif

// ---------------------------------------------------------------------------
// main: allocate buffers, warmup, NVTX-profiled iterations for each variant
// S=512 rows, K=1024 cols
// ---------------------------------------------------------------------------
int main() {
  const int S = 512;  // number of rows (batch*seq)
  const int K = 1024; // hidden size (cols per row)
  const float g_val = 1.0f;
  const float b_val = 0.0f;

  // ---- f32 buffers (used by F32, F32X4) ----
#if defined(LAYERNORM_F32) || defined(LAYERNORM_F32X4)
  float *d_x_f32, *d_y_f32;
  cudaMalloc(&d_x_f32, (size_t)S * K * sizeof(float));
  cudaMalloc(&d_y_f32, (size_t)S * K * sizeof(float));
  cudaMemset(d_x_f32, 0, (size_t)S * K * sizeof(float));
#endif

  // ---- f16 input buffer (used by all half-input variants) ----
#if defined(LAYERNORM_F16_F16) || defined(LAYERNORM_F16X2_F16) || \
    defined(LAYERNORM_F16X8_F16) || defined(LAYERNORM_F16_F32) || \
    defined(LAYERNORM_F16X8_PACK_F16) || defined(LAYERNORM_F16X8_PACK_F32)
  half *d_x_f16;
  cudaMalloc(&d_x_f16, (size_t)S * K * sizeof(half));
  cudaMemset(d_x_f16, 0, (size_t)S * K * sizeof(half));
#endif

  // ---- f16 output buffer (half in, half out: F16_F16, F16X2_F16, F16X8_F16, F16X8_PACK_F16) ----
#if defined(LAYERNORM_F16_F16) || defined(LAYERNORM_F16X2_F16) || \
    defined(LAYERNORM_F16X8_F16) || defined(LAYERNORM_F16X8_PACK_F16)
  half *d_y_f16;
  cudaMalloc(&d_y_f16, (size_t)S * K * sizeof(half));
#endif

  // ---- f16 output buffer for f32-accumulate variants (F16_F32, F16X8_PACK_F32) ----
  // layer_norm_f16_f32_kernel and layer_norm_f16x8_pack_f32_kernel write __float2half into half* y
#if defined(LAYERNORM_F16_F32) || defined(LAYERNORM_F16X8_PACK_F32)
  half *d_y_f16out;
  cudaMalloc(&d_y_f16out, (size_t)S * K * sizeof(half));
#endif

  // ------------------------------------------------------------------
  // layer_norm_f32_kernel  block=K=1024, grid=S
  // ------------------------------------------------------------------
#ifdef LAYERNORM_F32
  {
    dim3 grid(S), block(K);
    for (int i = 0; i < N_WARMUP; i++)
      layer_norm_f32_kernel<1024><<<grid, block>>>(d_x_f32, d_y_f32, g_val, b_val, S, K);
    cudaDeviceSynchronize();
    nvtxRangePush("layer_norm_f32_kernel");
    for (int i = 0; i < N_ITER; i++)
      layer_norm_f32_kernel<1024><<<grid, block>>>(d_x_f32, d_y_f32, g_val, b_val, S, K);
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

  // ------------------------------------------------------------------
  // layer_norm_f32x4_kernel  block=K/4=256, grid=S
  // ------------------------------------------------------------------
#ifdef LAYERNORM_F32X4
  {
    dim3 grid(S), block(K / 4);
    for (int i = 0; i < N_WARMUP; i++)
      layer_norm_f32x4_kernel<256><<<grid, block>>>(d_x_f32, d_y_f32, g_val, b_val, S, K);
    cudaDeviceSynchronize();
    nvtxRangePush("layer_norm_f32x4_kernel");
    for (int i = 0; i < N_ITER; i++)
      layer_norm_f32x4_kernel<256><<<grid, block>>>(d_x_f32, d_y_f32, g_val, b_val, S, K);
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

  // ------------------------------------------------------------------
  // layer_norm_f16_f16_kernel  block=K=1024, grid=S
  // ------------------------------------------------------------------
#ifdef LAYERNORM_F16_F16
  {
    dim3 grid(S), block(K);
    for (int i = 0; i < N_WARMUP; i++)
      layer_norm_f16_f16_kernel<1024><<<grid, block>>>(d_x_f16, d_y_f16, g_val, b_val, S, K);
    cudaDeviceSynchronize();
    nvtxRangePush("layer_norm_f16_f16_kernel");
    for (int i = 0; i < N_ITER; i++)
      layer_norm_f16_f16_kernel<1024><<<grid, block>>>(d_x_f16, d_y_f16, g_val, b_val, S, K);
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

  // ------------------------------------------------------------------
  // layer_norm_f16x2_f16_kernel  block=K/2=512, grid=S
  // ------------------------------------------------------------------
#ifdef LAYERNORM_F16X2_F16
  {
    dim3 grid(S), block(K / 2);
    for (int i = 0; i < N_WARMUP; i++)
      layer_norm_f16x2_f16_kernel<512><<<grid, block>>>(d_x_f16, d_y_f16, g_val, b_val, S, K);
    cudaDeviceSynchronize();
    nvtxRangePush("layer_norm_f16x2_f16_kernel");
    for (int i = 0; i < N_ITER; i++)
      layer_norm_f16x2_f16_kernel<512><<<grid, block>>>(d_x_f16, d_y_f16, g_val, b_val, S, K);
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

  // ------------------------------------------------------------------
  // layer_norm_f16x8_f16_kernel  block=K/8=128, grid=S
  // ------------------------------------------------------------------
#ifdef LAYERNORM_F16X8_F16
  {
    dim3 grid(S), block(K / 8);
    for (int i = 0; i < N_WARMUP; i++)
      layer_norm_f16x8_f16_kernel<128><<<grid, block>>>(d_x_f16, d_y_f16, g_val, b_val, S, K);
    cudaDeviceSynchronize();
    nvtxRangePush("layer_norm_f16x8_f16_kernel");
    for (int i = 0; i < N_ITER; i++)
      layer_norm_f16x8_f16_kernel<128><<<grid, block>>>(d_x_f16, d_y_f16, g_val, b_val, S, K);
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

  // ------------------------------------------------------------------
  // layer_norm_f16_f32_kernel  block=K=1024, grid=S
  // output is half* (stores __float2half result)
  // ------------------------------------------------------------------
#ifdef LAYERNORM_F16_F32
  {
    dim3 grid(S), block(K);
    for (int i = 0; i < N_WARMUP; i++)
      layer_norm_f16_f32_kernel<1024><<<grid, block>>>(d_x_f16, d_y_f16out, g_val, b_val, S, K);
    cudaDeviceSynchronize();
    nvtxRangePush("layer_norm_f16_f32_kernel");
    for (int i = 0; i < N_ITER; i++)
      layer_norm_f16_f32_kernel<1024><<<grid, block>>>(d_x_f16, d_y_f16out, g_val, b_val, S, K);
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

  // ------------------------------------------------------------------
  // layer_norm_f16x8_pack_f16_kernel  block=K/8=128, grid=S
  // ------------------------------------------------------------------
#ifdef LAYERNORM_F16X8_PACK_F16
  {
    dim3 grid(S), block(K / 8);
    for (int i = 0; i < N_WARMUP; i++)
      layer_norm_f16x8_pack_f16_kernel<128><<<grid, block>>>(d_x_f16, d_y_f16, g_val, b_val, S, K);
    cudaDeviceSynchronize();
    nvtxRangePush("layer_norm_f16x8_pack_f16_kernel");
    for (int i = 0; i < N_ITER; i++)
      layer_norm_f16x8_pack_f16_kernel<128><<<grid, block>>>(d_x_f16, d_y_f16, g_val, b_val, S, K);
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

  // ------------------------------------------------------------------
  // layer_norm_f16x8_pack_f32_kernel  block=K/8=128, grid=S
  // output is half* (stores __float2half result)
  // ------------------------------------------------------------------
#ifdef LAYERNORM_F16X8_PACK_F32
  {
    dim3 grid(S), block(K / 8);
    for (int i = 0; i < N_WARMUP; i++)
      layer_norm_f16x8_pack_f32_kernel<128><<<grid, block>>>(d_x_f16, d_y_f16out, g_val, b_val, S, K);
    cudaDeviceSynchronize();
    nvtxRangePush("layer_norm_f16x8_pack_f32_kernel");
    for (int i = 0; i < N_ITER; i++)
      layer_norm_f16x8_pack_f32_kernel<128><<<grid, block>>>(d_x_f16, d_y_f16out, g_val, b_val, S, K);
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
#endif

#if defined(LAYERNORM_F32) || defined(LAYERNORM_F32X4)
  cudaFree(d_x_f32);
  cudaFree(d_y_f32);
#endif
#if defined(LAYERNORM_F16_F16) || defined(LAYERNORM_F16X2_F16) || \
    defined(LAYERNORM_F16X8_F16) || defined(LAYERNORM_F16_F32) || \
    defined(LAYERNORM_F16X8_PACK_F16) || defined(LAYERNORM_F16X8_PACK_F32)
  cudaFree(d_x_f16);
#endif
#if defined(LAYERNORM_F16_F16) || defined(LAYERNORM_F16X2_F16) || \
    defined(LAYERNORM_F16X8_F16) || defined(LAYERNORM_F16X8_PACK_F16)
  cudaFree(d_y_f16);
#endif
#if defined(LAYERNORM_F16_F32) || defined(LAYERNORM_F16X8_PACK_F32)
  cudaFree(d_y_f16out);
#endif

  printf("Done.\n");
  return 0;
}
