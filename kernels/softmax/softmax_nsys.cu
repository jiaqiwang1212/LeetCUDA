#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <stdio.h>
#include <float.h>
#define N_WARMUP 5
#define N_ITER 20

__global__ void softmax_f32_kernel(float* x, float* y, int N) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    float* rx = x + row * N;
    float* ry = y + row * N;

    // find max
    float mx = -FLT_MAX;
    for (int i = tid; i < N; i += blockDim.x) mx = fmaxf(mx, rx[i]);
    smem[tid] = mx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    float max_val = smem[0];

    // sum exp
    float sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) sum += expf(rx[i] - max_val);
    smem[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / smem[0];

    for (int i = tid; i < N; i += blockDim.x)
        ry[i] = expf(rx[i] - max_val) * inv_sum;
}

int main() {
    const int B = 1024, N = 1024;
    float *d_x, *d_y;
    cudaMalloc(&d_x, B * N * sizeof(float));
    cudaMalloc(&d_y, B * N * sizeof(float));
    cudaMemset(d_x, 0, B * N * sizeof(float));

    dim3 block(256), grid(B);
    size_t smem = 256 * sizeof(float);

    for (int i = 0; i < N_WARMUP; i++)
        softmax_f32_kernel<<<grid, block, smem>>>(d_x, d_y, N);
    cudaDeviceSynchronize();
    nvtxRangePush("softmax_f32_kernel");
    for (int i = 0; i < N_ITER; i++)
        softmax_f32_kernel<<<grid, block, smem>>>(d_x, d_y, N);
    cudaDeviceSynchronize();
    nvtxRangePop();

    cudaFree(d_x); cudaFree(d_y);
    printf("Done.\n");
    return 0;
}
