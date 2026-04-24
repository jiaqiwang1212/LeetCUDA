#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <nvtx3/nvToolsExt.h>
#include <stdio.h>
#include <stdlib.h>

using namespace nvcuda;

#define N_WARMUP 5
#define N_ITER 20

// MMA-based HGEMM: each warp computes a 16x16 output tile using wmma
__global__ void hgemm_mma_m16n8k16_stages_kernel(half* A, half* B, half* C, int M, int N, int K) {
    // Each warp handles a 16x16 C tile
    int warp_row = (blockIdx.y * blockDim.y + threadIdx.y);
    int warp_col = (blockIdx.x * blockDim.x + threadIdx.x) / 32;

    if (warp_row * 16 >= M || warp_col * 16 >= N) return;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;
    wmma::fill_fragment(c_frag, __float2half(0.0f));

    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(a_frag, A + warp_row * 16 * K + k, K);
        wmma::load_matrix_sync(b_frag, B + k * N + warp_col * 16, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    wmma::store_matrix_sync(C + warp_row * 16 * N + warp_col * 16, c_frag, N, wmma::mem_row_major);
}

int main() {
    const int M = 1024, N = 1024, K = 1024;
    size_t sA = M * K * sizeof(half);
    size_t sB = K * N * sizeof(half);
    size_t sC = M * N * sizeof(half);

    half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sA);
    cudaMalloc(&d_B, sB);
    cudaMalloc(&d_C, sC);
    cudaMemset(d_A, 0, sA);
    cudaMemset(d_B, 0, sB);
    cudaMemset(d_C, 0, sC);

    // grid: (N/16, M/16), block: (32, 1) per warp
    dim3 block(32, 1);
    dim3 grid(N / 16, M / 16);

    for (int i = 0; i < N_WARMUP; i++)
        hgemm_mma_m16n8k16_stages_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    nvtxRangePush("hgemm_mma_m16n8k16_stages_kernel");
    for (int i = 0; i < N_ITER; i++)
        hgemm_mma_m16n8k16_stages_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    nvtxRangePop();

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    printf("Done.\n");
    return 0;
}
