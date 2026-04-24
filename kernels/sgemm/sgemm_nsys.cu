// Standalone NSYS profiling harness for sgemm kernels.
// Covers all __global__ kernels defined in sgemm.cu (CUDA-core variants) plus
// the two WMMA TF32 stage kernels from sgemm_wmma_tf32_stage.cu.
// Kernels from sgemm_async.cu are not included here (they are a separate TU).

#include <cuda_runtime.h>
#include <mma.h>
#include <nvtx3/nvToolsExt.h>
#include <stdio.h>

#define N_WARMUP 5
#define N_ITER   20

using namespace nvcuda;

// Macros copied verbatim from the source files
#define WARP_SIZE 32
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

#define DEVICE_INLINE __device__ inline
#define HOST_DEVICE_INLINE __device__ __host__ inline

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL()     asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n)  asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
#define CP_ASYNC_CA(dst, src, bytes)                                           \
  asm volatile(                                                                \
      "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),     \
      "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes)                                           \
  asm volatile(                                                                \
      "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),     \
      "l"(src), "n"(bytes))

HOST_DEVICE_INLINE
int div_ceil(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

// ---------------------------------------------------------------------------
// CUDA-core kernels – copied verbatim from sgemm.cu
// ---------------------------------------------------------------------------

// SGEMM naive
__global__ void sgemm_naive_f32_kernel(float *a, float *b, float *c, int M,
                                       int N, int K) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;

  if (m < M && n < N) {
    float psum = 0.0;
#pragma unroll
    for (int k = 0; k < K; k++) {
      psum += a[m * K + k] * b[k * N + n];
    }
    c[m * N + n] = psum;
  }
}

// SGEMM: Block Tile + K Tile, with smem
template <const int BM = 32, const int BN = 32, const int BK = 32>
__global__ void sgemm_sliced_k_f32_kernel(float *a, float *b, float *c, int M,
                                          int N, int K) {
  __shared__ float s_a[BM][BK], s_b[BK][BN];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = threadIdx.y * blockDim.x + tx;

  int load_smem_a_m = tid / 32;
  int load_smem_a_k = tid % 32;
  int load_smem_b_k = tid / 32;
  int load_smem_b_n = tid % 32;
  int load_gmem_a_m = by * BM + load_smem_a_m;
  int load_gmem_b_n = bx * BN + load_smem_b_n;

  float sum = 0.f;
  for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
    int load_gmem_a_k = bk * BK + load_smem_a_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    s_a[load_smem_a_m][load_smem_a_k] = a[load_gmem_a_addr];
    int load_gmem_b_k = bk * BK + load_smem_b_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
    s_b[load_smem_b_k][load_smem_b_n] = b[load_gmem_b_addr];
    __syncthreads();
#pragma unroll
    for (int k = 0; k < BK; ++k) {
      int comp_smem_a_m = load_smem_a_m;
      int comp_smem_b_n = load_smem_b_n;
      sum += s_a[comp_smem_a_m][k] * s_b[k][comp_smem_b_n];
    }
    __syncthreads();
  }
  int store_gmem_c_m = load_gmem_a_m;
  int store_gmem_c_n = load_gmem_b_n;
  int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
  c[store_gmem_c_addr] = sum;
}

// SGEMM: Block Tile + Thread Tile + K Tile + Vec4
template <const int BM = 128, const int BN = 128, const int BK = 8,
          const int TM = 8, const int TN = 8>
__global__ void sgemm_t_8x8_sliced_k_f32x4_kernel(float *a, float *b, float *c,
                                                  int M, int N, int K) {
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = threadIdx.y * blockDim.x + tx;
  __shared__ float s_a[BM][BK], s_b[BK][BN];

  int load_smem_a_m = tid / 2;
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 4;
  int load_smem_b_k = tid / 32;
  int load_smem_b_n = (tid % 32) * 4;
  int load_gmem_a_m = by * BM + load_smem_a_m;
  int load_gmem_b_n = bx * BN + load_smem_b_n;

  float r_c[TM][TN] = {0.0};
  for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
    int load_gmem_a_k = bk * BK + load_smem_a_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    FLOAT4(s_a[load_smem_a_m][load_smem_a_k]) = FLOAT4(a[load_gmem_a_addr]);
    int load_gmem_b_k = bk * BK + load_smem_b_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
    FLOAT4(s_b[load_smem_b_k][load_smem_b_n]) = FLOAT4(b[load_gmem_b_addr]);
    __syncthreads();
#pragma unroll
    for (int k = 0; k < BK; k++) {
#pragma unroll
      for (int m = 0; m < TM; m++) {
#pragma unroll
        for (int n = 0; n < TN; n++) {
          int comp_smem_a_m = ty * TM + m;
          int comp_smem_b_n = tx * TN + n;
          r_c[m][n] += s_a[comp_smem_a_m][k] * s_b[k][comp_smem_b_n];
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int m = 0; m < TM; ++m) {
    int store_gmem_c_m = by * BM + ty * TM + m;
#pragma unroll
    for (int n = 0; n < TN; n += 4) {
      int store_gmem_c_n = bx * BN + tx * TN + n;
      int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
      FLOAT4(c[store_gmem_c_addr]) = FLOAT4(r_c[m][n]);
    }
  }
}

// SGEMM: Block Tile + Thread Tile + K Tile + Vec4 + BCF
template <const int BM = 128, const int BN = 128, const int BK = 8,
          const int TM = 8, const int TN = 8, const int OFFSET = 0>
__global__ void
sgemm_t_8x8_sliced_k_f32x4_bcf_kernel(float *a, float *b, float *c, const int M,
                                      const int N, const int K) {
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;

  __shared__ float s_a[BK][BM + OFFSET];
  __shared__ float s_b[BK][BN + OFFSET];

  float r_load_a[TM / 2];
  float r_load_b[TN / 2];
  float r_comp_a[TM];
  float r_comp_b[TN];
  float r_c[TM][TN] = {0.0};

  int load_a_smem_m = tid / 2;
  int load_a_smem_k = (tid & 1) << 2;
  int load_b_smem_k = tid / 32;
  int load_b_smem_n = (tid & 31) << 2;

  int load_a_gmem_m = by * BM + load_a_smem_m;
  int load_b_gmem_n = bx * BN + load_b_smem_n;

  if (load_a_gmem_m >= M || load_b_gmem_n >= N)
    return;

  for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
    int load_a_gmem_k = bk * BK + load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = bk * BK + load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
    FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
    FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

    s_a[load_a_smem_k][load_a_smem_m]     = r_load_a[0];
    s_a[load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
    s_a[load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
    s_a[load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
    FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);

    __syncthreads();

#pragma unroll
    for (int tk = 0; tk < BK; tk++) {
      FLOAT4(r_comp_a[0]) = FLOAT4(s_a[tk][ty * TM / 2]);
      FLOAT4(r_comp_a[4]) = FLOAT4(s_a[tk][ty * TM / 2 + BM / 2]);
      FLOAT4(r_comp_b[0]) = FLOAT4(s_b[tk][tx * TN / 2]);
      FLOAT4(r_comp_b[4]) = FLOAT4(s_b[tk][tx * TN / 2 + BN / 2]);
#pragma unroll
      for (int tm = 0; tm < TM; tm++) {
#pragma unroll
        for (int tn = 0; tn < TN; tn++) {
          r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM / 2; i++) {
    int store_c_gmem_m = by * BM + ty * TM / 2 + i;
    int store_c_gmem_n = bx * BN + tx * TN / 2;
    int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
    FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
    FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i][4]);
  }
#pragma unroll
  for (int i = 0; i < TM / 2; i++) {
    int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
    int store_c_gmem_n = bx * BN + tx * TN / 2;
    int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
    FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i + TM / 2][0]);
    FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
  }
}

// SGEMM: BCF + Double-buffer
template <const int BM = 128, const int BN = 128, const int BK = 8,
          const int TM = 8, const int TN = 8, const int OFFSET = 0>
__global__ void sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_kernel(
    float *a, float *b, float *c, const int M, const int N, const int K) {
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;

  __shared__ float s_a[2][BK][BM + OFFSET];
  __shared__ float s_b[2][BK][BN + OFFSET];

  float r_load_a[TM / 2];
  float r_load_b[TN / 2];
  float r_comp_a[TM];
  float r_comp_b[TN];
  float r_c[TM][TN] = {0.0};

  int load_a_smem_m = tid / 2;
  int load_a_smem_k = (tid & 1) << 2;
  int load_b_smem_k = tid / 32;
  int load_b_smem_n = (tid & 31) << 2;

  int load_a_gmem_m = by * BM + load_a_smem_m;
  int load_b_gmem_n = bx * BN + load_b_smem_n;

  // bk = 0 loading, buffer 0
  {
    int load_a_gmem_k = load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
    FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
    FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

    s_a[0][load_a_smem_k + 0][load_a_smem_m] = r_load_a[0];
    s_a[0][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
    s_a[0][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
    s_a[0][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
    FLOAT4(s_b[0][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);
  }
  __syncthreads();

  for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {
    int smem_sel = (bk - 1) & 1;
    int smem_sel_next = bk & 1;

    int load_a_gmem_k = bk * BK + load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = bk * BK + load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
    FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
    FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

#pragma unroll
    for (int tk = 0; tk < BK; tk++) {
      FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2]);
      FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2 + BM / 2]);
      FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2]);
      FLOAT4(r_comp_b[4]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2 + BN / 2]);

#pragma unroll
      for (int tm = 0; tm < TM; tm++) {
#pragma unroll
        for (int tn = 0; tn < TN; tn++) {
          r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
        }
      }
    }

    s_a[smem_sel_next][load_a_smem_k + 0][load_a_smem_m] = r_load_a[0];
    s_a[smem_sel_next][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
    s_a[smem_sel_next][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
    s_a[smem_sel_next][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
    FLOAT4(s_b[smem_sel_next][load_b_smem_k][load_b_smem_n]) =
        FLOAT4(r_load_b[0]);

    __syncthreads();
  }

#pragma unroll
  for (int tk = 0; tk < BK; tk++) {
    FLOAT4(r_comp_a[0]) = FLOAT4(s_a[1][tk][ty * TM / 2]);
    FLOAT4(r_comp_a[4]) = FLOAT4(s_a[1][tk][ty * TM / 2 + BM / 2]);
    FLOAT4(r_comp_b[0]) = FLOAT4(s_b[1][tk][tx * TN / 2]);
    FLOAT4(r_comp_b[4]) = FLOAT4(s_b[1][tk][tx * TN / 2 + BN / 2]);

#pragma unroll
    for (int tm = 0; tm < TM; tm++) {
#pragma unroll
      for (int tn = 0; tn < TN; tn++) {
        r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
      }
    }
  }

#pragma unroll
  for (int i = 0; i < TM / 2; i++) {
    int store_c_gmem_m = by * BM + ty * TM / 2 + i;
    int store_c_gmem_n = bx * BN + tx * TN / 2;
    int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
    FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
    FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i][4]);
  }
#pragma unroll
  for (int i = 0; i < TM / 2; i++) {
    int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
    int store_c_gmem_n = bx * BN + tx * TN / 2;
    int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
    FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i + TM / 2][0]);
    FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
  }
}

// ---------------------------------------------------------------------------
// WMMA TF32 stage kernels – copied verbatim from sgemm_wmma_tf32_stage.cu
// ---------------------------------------------------------------------------

// Helper used by wmma kernels
__global__ void f32x4_tf32x4_kernel(float *x, float *y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx < N) {
    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_y;
    reg_y.x = wmma::__float_to_tf32(reg_x.x);
    reg_y.y = wmma::__float_to_tf32(reg_x.y);
    reg_y.z = wmma::__float_to_tf32(reg_x.z);
    reg_y.w = wmma::__float_to_tf32(reg_x.w);
    FLOAT4(y[idx]) = reg_y;
  }
}

template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 8,
          const int WMMA_TILE_M = 4, const int WMMA_TILE_N = 2,
          const int WARP_TILE_M = 2, const int WARP_TILE_N = 4,
          const int A_PAD = 0, const int B_PAD = 0, const int K_STAGE = 2,
          const bool BLOCK_SWIZZLE = false>
__global__ void
sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_kernel(float *A, float *B, float *C,
                                                 int M, int N, int K) {
  const int bx = ((int)BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, WMMA_K);
  constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M;
  constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N;
  constexpr int BK = WMMA_K;
  __shared__ float s_a[K_STAGE][BM][BK + A_PAD], s_b[K_STAGE][BK][BN + B_PAD];

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int warp_m = warp_id / 2;
  const int warp_n = warp_id % 2;

  int load_smem_a_m = tid / 2;
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 4;
  int load_smem_b_k = tid / 32;
  int load_smem_b_n = (tid % 32) * 4;
  int load_gmem_a_m = by * BM + load_smem_a_m;
  int load_gmem_b_n = bx * BN + load_smem_b_n;

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
      C_frag[WARP_TILE_M][WARP_TILE_N];

#pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      wmma::fill_fragment(C_frag[i][j], 0.0);
    }
  }

#pragma unroll
  for (int k = 0; k < (K_STAGE - 1); ++k) {
    int load_gmem_a_k = k * WMMA_K + load_smem_a_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * WMMA_K + load_smem_b_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

    uint32_t load_smem_a_ptr =
        __cvta_generic_to_shared(&s_a[k][load_smem_a_m][load_smem_a_k]);
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

    uint32_t load_smem_b_ptr =
        __cvta_generic_to_shared(&s_b[k][load_smem_b_k][load_smem_b_n]);
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);

    CP_ASYNC_COMMIT_GROUP();
  }

  CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
  __syncthreads();

#pragma unroll
  for (int k = (K_STAGE - 1); k < NUM_K_TILES; k++) {
    int smem_sel = (k + 1) % K_STAGE;
    int smem_sel_next = k % K_STAGE;

    int load_gmem_a_k = k * WMMA_K + load_smem_a_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * WMMA_K + load_smem_b_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

    uint32_t load_smem_a_ptr = __cvta_generic_to_shared(
        &s_a[smem_sel_next][load_smem_a_m][load_smem_a_k]);
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

    uint32_t load_smem_b_ptr = __cvta_generic_to_shared(
        &s_b[smem_sel_next][load_smem_b_k][load_smem_b_n]);
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
    CP_ASYNC_COMMIT_GROUP();

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                   wmma::precision::tf32, wmma::row_major>
        A_frag[WARP_TILE_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                   wmma::precision::tf32, wmma::row_major>
        B_frag[WARP_TILE_N];

#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      wmma::load_matrix_sync(A_frag[i], &s_a[smem_sel][warp_smem_a_m][0],
                             BK + A_PAD);
    }

#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
      wmma::load_matrix_sync(B_frag[j], &s_b[smem_sel][0][warp_smem_b_n],
                             BN + B_PAD);
    }

#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
      }
    }

    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
    __syncthreads();
  }

  if ((K_STAGE - 2) > 0) {
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads();
  }

  {
#pragma unroll
    for (int k = 0; k < (K_STAGE - 1); k++) {
      const int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);
      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                     wmma::precision::tf32, wmma::row_major>
          A_frag[WARP_TILE_M];
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                     wmma::precision::tf32, wmma::row_major>
          B_frag[WARP_TILE_N];

#pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
        wmma::load_matrix_sync(A_frag[i], &s_a[stage_sel][warp_smem_a_m][0],
                               BK + A_PAD);
      }
#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
        wmma::load_matrix_sync(B_frag[j], &s_b[stage_sel][0][warp_smem_b_n],
                               BN + B_PAD);
      }
#pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
        }
      }
    }
  }

#pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      const int store_gmem_a_m =
          by * BM + warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      const int store_gmem_a_n =
          bx * BN + warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
      wmma::store_matrix_sync(C + store_gmem_a_m * N + store_gmem_a_n,
                              C_frag[i][j], N, wmma::mem_row_major);
    }
  }
}

// ---------------------------------------------------------------------------
// Compile-time kernel selection via -D macros.
// Usage: nvcc -DSGEMM_NAIVE_F32 ...  (selects only that kernel)
// If none are defined, all kernels are enabled by default.
// ---------------------------------------------------------------------------
#if !defined(SGEMM_NAIVE_F32)              && \
    !defined(SGEMM_SLICED_K_F32)           && \
    !defined(SGEMM_T_8X8_F32X4)            && \
    !defined(SGEMM_T_8X8_F32X4_BCF)        && \
    !defined(SGEMM_T_8X8_F32X4_BCF_OFFSET) && \
    !defined(SGEMM_T_8X8_F32X4_BCF_DBUF)   && \
    !defined(SGEMM_T_8X8_F32X4_BCF_DBUF_OFFSET) && \
    !defined(SGEMM_WMMA_TF32_STAGE2)       && \
    !defined(SGEMM_WMMA_TF32_STAGE3)
#define SGEMM_NAIVE_F32
#define SGEMM_SLICED_K_F32
#define SGEMM_T_8X8_F32X4
#define SGEMM_T_8X8_F32X4_BCF
#define SGEMM_T_8X8_F32X4_BCF_OFFSET
#define SGEMM_T_8X8_F32X4_BCF_DBUF
#define SGEMM_T_8X8_F32X4_BCF_DBUF_OFFSET
#define SGEMM_WMMA_TF32_STAGE2
#define SGEMM_WMMA_TF32_STAGE3
#endif

// ---------------------------------------------------------------------------
// Profiling harness
// ---------------------------------------------------------------------------
#define RUN_KERNEL(label, launch)                      \
  do {                                                 \
    for (int _i = 0; _i < N_WARMUP; _i++) { launch; } \
    cudaDeviceSynchronize();                           \
    nvtxRangePush(label);                              \
    for (int _i = 0; _i < N_ITER; _i++) { launch; }   \
    cudaDeviceSynchronize();                           \
    nvtxRangePop();                                    \
  } while (0)

int main() {
  const int M = 1024, N = 1024, K = 1024;

  // All kernel variants use float* buffers – allocate unconditionally.
  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, (size_t)M * K * sizeof(float));
  cudaMalloc(&d_B, (size_t)K * N * sizeof(float));
  cudaMalloc(&d_C, (size_t)M * N * sizeof(float));
  cudaMemset(d_A, 0, (size_t)M * K * sizeof(float));
  cudaMemset(d_B, 0, (size_t)K * N * sizeof(float));
  cudaMemset(d_C, 0, (size_t)M * N * sizeof(float));

  // ------------------------------------------------------------------
  // sgemm_naive_f32
  // block(BN=32, BM=32), grid((N+BN-1)/BN, (M+BM-1)/BM)
  // ------------------------------------------------------------------
#ifdef SGEMM_NAIVE_F32
  {
    constexpr int BM = 32, BN = 32;
    dim3 block(BN, BM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    RUN_KERNEL("sgemm_naive_f32",
               sgemm_naive_f32_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K));
  }
#endif

  // ------------------------------------------------------------------
  // sgemm_sliced_k_f32  BM=BN=BK=32
  // ------------------------------------------------------------------
#ifdef SGEMM_SLICED_K_F32
  {
    constexpr int BM = 32, BN = 32, BK = 32;
    dim3 block(BN, BM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    RUN_KERNEL("sgemm_sliced_k_f32",
               (sgemm_sliced_k_f32_kernel<BM, BN, BK>
                <<<grid, block>>>(d_A, d_B, d_C, M, N, K)));
  }
#endif

  // ------------------------------------------------------------------
  // sgemm_t_8x8_sliced_k_f32x4  BM=BN=128 BK=8 TM=TN=8
  // block(BN/TN, BM/TM) = (16,16), grid((N+BN-1)/BN, (M+BM-1)/BM)
  // ------------------------------------------------------------------
#ifdef SGEMM_T_8X8_F32X4
  {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    RUN_KERNEL("sgemm_t_8x8_sliced_k_f32x4",
               (sgemm_t_8x8_sliced_k_f32x4_kernel<BM, BN, BK, TM, TN>
                <<<grid, block>>>(d_A, d_B, d_C, M, N, K)));
  }
#endif

  // ------------------------------------------------------------------
  // sgemm_t_8x8_sliced_k_f32x4_bcf  OFFSET=0
  // ------------------------------------------------------------------
#ifdef SGEMM_T_8X8_F32X4_BCF
  {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8, OFFSET = 0;
    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    RUN_KERNEL("sgemm_t_8x8_sliced_k_f32x4_bcf",
               (sgemm_t_8x8_sliced_k_f32x4_bcf_kernel<BM, BN, BK, TM, TN, OFFSET>
                <<<grid, block>>>(d_A, d_B, d_C, M, N, K)));
  }
#endif

  // ------------------------------------------------------------------
  // sgemm_t_8x8_sliced_k_f32x4_bcf_offset  OFFSET=4
  // ------------------------------------------------------------------
#ifdef SGEMM_T_8X8_F32X4_BCF_OFFSET
  {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8, OFFSET = 4;
    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    RUN_KERNEL("sgemm_t_8x8_sliced_k_f32x4_bcf_offset",
               (sgemm_t_8x8_sliced_k_f32x4_bcf_kernel<BM, BN, BK, TM, TN, OFFSET>
                <<<grid, block>>>(d_A, d_B, d_C, M, N, K)));
  }
#endif

  // ------------------------------------------------------------------
  // sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf  OFFSET=0
  // ------------------------------------------------------------------
#ifdef SGEMM_T_8X8_F32X4_BCF_DBUF
  {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8, OFFSET = 0;
    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    RUN_KERNEL("sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf",
               (sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_kernel<BM, BN, BK, TM, TN, OFFSET>
                <<<grid, block>>>(d_A, d_B, d_C, M, N, K)));
  }
#endif

  // ------------------------------------------------------------------
  // sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_offset  OFFSET=4
  // ------------------------------------------------------------------
#ifdef SGEMM_T_8X8_F32X4_BCF_DBUF_OFFSET
  {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8, OFFSET = 4;
    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    RUN_KERNEL("sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_offset",
               (sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_kernel<BM, BN, BK, TM, TN, OFFSET>
                <<<grid, block>>>(d_A, d_B, d_C, M, N, K)));
  }
#endif

  // ------------------------------------------------------------------
  // WMMA TF32 stage2 (no swizzle)
  // 256 threads per block (8 warps), BM=BN=128, K_STAGE=2
  // block=dim3(WARP_SIZE, 8) = dim3(32,8), grid=(N/BN, M/BM)
  // ------------------------------------------------------------------
#ifdef SGEMM_WMMA_TF32_STAGE2
  {
    constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 8;
    constexpr int WMMA_TILE_M = 4, WMMA_TILE_N = 2;
    constexpr int WARP_TILE_M = 2, WARP_TILE_N = 4;
    constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M; // 128
    constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N; // 128
    constexpr int NUM_THREADS = 256; // 8 warps
    dim3 block(WARP_SIZE, NUM_THREADS / WARP_SIZE); // (32, 8)
    dim3 grid(div_ceil(N, BN), div_ceil(M, BM));
    RUN_KERNEL("sgemm_wmma_m16n16k8_mma4x2_warp2x4_stage2",
               (sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_kernel<
                   WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N,
                   WARP_TILE_M, WARP_TILE_N, 0, 0, 2, false>
                <<<grid, block>>>(d_A, d_B, d_C, M, N, K)));
  }
#endif

  // ------------------------------------------------------------------
  // WMMA TF32 stage3 (no swizzle)
  // ------------------------------------------------------------------
#ifdef SGEMM_WMMA_TF32_STAGE3
  {
    constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 8;
    constexpr int WMMA_TILE_M = 4, WMMA_TILE_N = 2;
    constexpr int WARP_TILE_M = 2, WARP_TILE_N = 4;
    constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M;
    constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N;
    constexpr int NUM_THREADS = 256;
    dim3 block(WARP_SIZE, NUM_THREADS / WARP_SIZE);
    dim3 grid(div_ceil(N, BN), div_ceil(M, BM));
    RUN_KERNEL("sgemm_wmma_m16n16k8_mma4x2_warp2x4_stage3",
               (sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_kernel<
                   WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N,
                   WARP_TILE_M, WARP_TILE_N, 0, 0, 3, false>
                <<<grid, block>>>(d_A, d_B, d_C, M, N, K)));
  }
#endif

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  printf("Done.\n");
  return 0;
}
