// NOTE: CUTLASS submodule not initialized; this is a pure-CUDA fallback
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <nvtx3/nvToolsExt.h>
#include <stdio.h>
#define N_WARMUP 5
#define N_ITER 20

__global__ void vector_add_local_tile_multi_elem_per_thread_half(half* A, half* B, half* C, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    for (int k = 0; k < 8 && idx + k < N; k++)
        C[idx+k] = __hadd(A[idx+k], B[idx+k]);
}

int main() {
    const int N = 1 << 20;
    half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(half));
    cudaMalloc(&d_B, N * sizeof(half));
    cudaMalloc(&d_C, N * sizeof(half));
    cudaMemset(d_A, 0, N * sizeof(half));
    cudaMemset(d_B, 0, N * sizeof(half));

    const int block = 256, grid = (N + block*8 - 1) / (block*8);

    for (int i = 0; i < N_WARMUP; i++)
        vector_add_local_tile_multi_elem_per_thread_half<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    nvtxRangePush("vector_add_local_tile_multi_elem_per_thread_half");
    for (int i = 0; i < N_ITER; i++)
        vector_add_local_tile_multi_elem_per_thread_half<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    nvtxRangePop();

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    printf("Done.\n");
    return 0;
}
