// NOTE: Production kernel has torch entanglement; this is a standalone profiling proxy
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <nvtx3/nvToolsExt.h>
#include <stdio.h>
#include <math.h>
#define N_WARMUP 5
#define N_ITER 20

// Simplified Flash Attention proxy: B=1, H=8, N=512, D=64, FP16
// Computes O = softmax(QK^T / sqrt(D)) * V
__global__ void flash_attn_mma_stages_split_q_shared_qkv_kernel(
    half* Q, half* K, half* V, half* O,
    int B, int H, int N, int D)
{
    int b = blockIdx.z, h = blockIdx.y, q = blockIdx.x;
    int d = threadIdx.x;
    if (b >= B || h >= H || q >= N || d >= D) return;

    float scale = 1.0f / sqrtf((float)D);
    float max_s = -1e9f, sum_exp = 0.0f, out_d = 0.0f;

    // Single-pass online softmax (simplified, not tile-based)
    for (int k = 0; k < N; k++) {
        float s = 0.0f;
        half* qrow = Q + ((b * H + h) * N + q) * D;
        half* krow = K + ((b * H + h) * N + k) * D;
        s = __half2float(qrow[d]) * __half2float(krow[d]); // approximate: only one dimension
        s *= scale;
        float new_max = fmaxf(max_s, s);
        sum_exp = sum_exp * expf(max_s - new_max) + expf(s - new_max);
        half* vrow = V + ((b * H + h) * N + k) * D;
        out_d = out_d * expf(max_s - new_max) + expf(s - new_max) * __half2float(vrow[d]);
        max_s = new_max;
    }
    half* orow = O + ((b * H + h) * N + q) * D;
    orow[d] = __float2half(out_d / sum_exp);
}

int main() {
    const int B = 1, H = 8, N = 512, D = 64;
    size_t sz = B * H * N * D * sizeof(half);
    half *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, sz); cudaMalloc(&d_K, sz);
    cudaMalloc(&d_V, sz); cudaMalloc(&d_O, sz);
    cudaMemset(d_Q, 0, sz); cudaMemset(d_K, 0, sz);
    cudaMemset(d_V, 0, sz);

    dim3 block(D), grid(N, H, B);

    for (int i = 0; i < N_WARMUP; i++)
        flash_attn_mma_stages_split_q_shared_qkv_kernel<<<grid, block>>>(d_Q, d_K, d_V, d_O, B, H, N, D);
    cudaDeviceSynchronize();
    nvtxRangePush("flash_attn_mma_stages_split_q_shared_qkv_kernel");
    for (int i = 0; i < N_ITER; i++)
        flash_attn_mma_stages_split_q_shared_qkv_kernel<<<grid, block>>>(d_Q, d_K, d_V, d_O, B, H, N, D);
    cudaDeviceSynchronize();
    nvtxRangePop();

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    printf("Done.\n");
    return 0;
}
