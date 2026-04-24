#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <nvtx3/nvToolsExt.h>
#include <stdio.h>
#define N_WARMUP 5
#define N_ITER 20
#define TILE 16

__global__ void hgemm_naive_ws_sm8x_kernel(half* A, half* B, half* C, int M, int N, int K) {
    __shared__ half smA[TILE][TILE], smB[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        smA[threadIdx.y][threadIdx.x] = (row < M && t*TILE+threadIdx.x < K) ? A[row*K+t*TILE+threadIdx.x] : __float2half(0.0f);
        smB[threadIdx.y][threadIdx.x] = (t*TILE+threadIdx.y < K && col < N) ? B[(t*TILE+threadIdx.y)*N+col] : __float2half(0.0f);
        __syncthreads();
        for (int k = 0; k < TILE; k++) acc += __half2float(smA[threadIdx.y][k]) * __half2float(smB[k][threadIdx.x]);
        __syncthreads();
    }
    if (row < M && col < N) C[row*N+col] = __float2half(acc);
}

int main() {
    const int M = 1024, N = 1024, K = 1024;
    half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M*K*sizeof(half));
    cudaMalloc(&d_B, K*N*sizeof(half));
    cudaMalloc(&d_C, M*N*sizeof(half));
    cudaMemset(d_A, 0, M*K*sizeof(half));
    cudaMemset(d_B, 0, K*N*sizeof(half));

    dim3 block(TILE, TILE), grid((N+TILE-1)/TILE, (M+TILE-1)/TILE);

    for (int i = 0; i < N_WARMUP; i++)
        hgemm_naive_ws_sm8x_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    nvtxRangePush("hgemm_naive_ws_sm8x_kernel");
    for (int i = 0; i < N_ITER; i++)
        hgemm_naive_ws_sm8x_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    nvtxRangePop();

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    printf("Done.\n");
    return 0;
}
