#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <nvtx3/nvToolsExt.h>
#include <stdio.h>
#include <stdlib.h>

using namespace nvcuda;

#define N_WARMUP 5
#define N_ITER 20

// MMA with swizzled shared memory to avoid bank conflicts
__global__ void mma_simple_swizzle_kernel(half* A, half* B, half* C, int M, int N, int K) {
    // Swizzled shared memory layout for A and B tiles
    __shared__ half smem_A[16][16 + 8]; // +8 padding for swizzle
    __shared__ half smem_B[16][16 + 8];

    int warp_row = blockIdx.y;
    int warp_col = blockIdx.x;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;
    wmma::fill_fragment(c_frag, __float2half(0.0f));

    for (int k = 0; k < K; k += 16) {
        // Load A tile with swizzled indexing
        int tid = threadIdx.x;
        if (tid < 16 * 16) {
            int r = tid / 16, c = tid % 16;
            int swizzle_c = c ^ (r & 7); // XOR swizzle
            smem_A[r][swizzle_c] = A[(warp_row * 16 + r) * K + k + c];
            smem_B[r][swizzle_c] = B[(k + r) * N + warp_col * 16 + c];
        }
        __syncthreads();

        wmma::load_matrix_sync(a_frag, &smem_A[0][0], 16 + 8);
        wmma::load_matrix_sync(b_frag, &smem_B[0][0], 16 + 8);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }
    wmma::store_matrix_sync(C + warp_row * 16 * N + warp_col * 16, c_frag, N, wmma::mem_row_major);
}

int main() {
    const int M = 1024, N = 1024, K = 1024;
    half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(half));
    cudaMemset(d_A, 0, M * K * sizeof(half));
    cudaMemset(d_B, 0, K * N * sizeof(half));
    cudaMemset(d_C, 0, M * N * sizeof(half));

    dim3 block(32);
    dim3 grid(N / 16, M / 16);

    for (int i = 0; i < N_WARMUP; i++)
        mma_simple_swizzle_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    nvtxRangePush("mma_simple_swizzle_kernel");
    for (int i = 0; i < N_ITER; i++)
        mma_simple_swizzle_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    nvtxRangePop();

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    printf("Done.\n");
    return 0;
}
