// Standalone NSYS profiling harness for nms_kernel from nms.cu.
// The source defines a single nms_kernel; the previous nsys file incorrectly
// referenced a non-existent hard_nms_kernel with a different box layout.
// This file uses the exact kernel from nms.cu verbatim.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <stdio.h>
#include <stdlib.h>

#define N_WARMUP 5
#define N_ITER   20

// Macros copied verbatim from nms.cu
#define WARP_SIZE 32

// ---------------------------------------------------------------------------
// Kernel – copied verbatim from nms.cu
// ---------------------------------------------------------------------------

__global__ void nms_kernel(const float *boxes, const float *scores, int *keep,
                           int num_boxes, float iou_threshold) {
  const int threadsPerBlock = blockDim.x;
  const int threadId = threadIdx.x;
  const int blockId = blockIdx.x;
  const int idx = blockId * threadsPerBlock + threadId;

  if (idx >= num_boxes)
    return;

  float x1 = boxes[idx * 4 + 0];
  float y1 = boxes[idx * 4 + 1];
  float x2 = boxes[idx * 4 + 2];
  float y2 = boxes[idx * 4 + 3];
  int suppressed = 0;

  for (int i = 0; i < idx; ++i) {
    if (keep[i] == 0)
      continue;

    float x1_i = boxes[i * 4 + 0];
    float y1_i = boxes[i * 4 + 1];
    float x2_i = boxes[i * 4 + 2];
    float y2_i = boxes[i * 4 + 3];

    float inter_x1 = max(x1, x1_i);
    float inter_y1 = max(y1, y1_i);
    float inter_x2 = min(x2, x2_i);
    float inter_y2 = min(y2, y2_i);
    float inter_w = max(0.0f, inter_x2 - inter_x1);
    float inter_h = max(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;

    float area = (x2 - x1) * (y2 - y1);
    float area_i = (x2_i - x1_i) * (y2_i - y1_i);
    float iou = inter_area / (area + area_i - inter_area);

    if (iou > iou_threshold) {
      keep[idx] = 0;
      return;
    }
  }
  keep[idx] = 1;
  return;
}

// ---------------------------------------------------------------------------
// Compile-time kernel selection via -D macros.
// Usage: nvcc -DNMS ...  (selects the nms kernel)
// If none are defined, all kernels are enabled by default.
// ---------------------------------------------------------------------------
#ifndef NMS
#define NMS
#endif

int main() {
  const int N_BOXES = 1024;
  const float IOU_THRESHOLD = 0.5f;

  // Allocate host data: boxes [N_BOXES x 4] = {x1, y1, x2, y2}
  float *h_boxes  = (float *)malloc(N_BOXES * 4 * sizeof(float));
  float *h_scores = (float *)malloc(N_BOXES * sizeof(float));
  for (int i = 0; i < N_BOXES; i++) {
    float x1 = (float)(i % 100);
    float y1 = (float)((i / 100) % 100);
    h_boxes[i * 4 + 0] = x1;
    h_boxes[i * 4 + 1] = y1;
    h_boxes[i * 4 + 2] = x1 + 50.0f;
    h_boxes[i * 4 + 3] = y1 + 50.0f;
    h_scores[i] = (float)(N_BOXES - i); // descending scores
  }

  float *d_boxes, *d_scores;
  int   *d_keep;
  cudaMalloc(&d_boxes,  N_BOXES * 4 * sizeof(float));
  cudaMalloc(&d_scores, N_BOXES     * sizeof(float));
  cudaMalloc(&d_keep,   N_BOXES     * sizeof(int));

  cudaMemcpy(d_boxes,  h_boxes,  N_BOXES * 4 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_scores, h_scores, N_BOXES     * sizeof(float), cudaMemcpyHostToDevice);
  free(h_boxes);
  free(h_scores);

  dim3 block(WARP_SIZE);
  dim3 grid((N_BOXES + WARP_SIZE - 1) / WARP_SIZE);

#ifdef NMS
  // Warmup
  for (int i = 0; i < N_WARMUP; i++) {
    cudaMemset(d_keep, 0, N_BOXES * sizeof(int));
    nms_kernel<<<grid, block>>>(d_boxes, d_scores, d_keep, N_BOXES, IOU_THRESHOLD);
  }
  cudaDeviceSynchronize();

  // Profiled iterations
  nvtxRangePush("nms_kernel");
  for (int i = 0; i < N_ITER; i++) {
    cudaMemset(d_keep, 0, N_BOXES * sizeof(int));
    nms_kernel<<<grid, block>>>(d_boxes, d_scores, d_keep, N_BOXES, IOU_THRESHOLD);
  }
  cudaDeviceSynchronize();
  nvtxRangePop();
#endif

  cudaFree(d_boxes);
  cudaFree(d_scores);
  cudaFree(d_keep);
  printf("Done.\n");
  return 0;
}
