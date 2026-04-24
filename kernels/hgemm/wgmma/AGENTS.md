<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# hgemm/wgmma

## Purpose
WGMMA (Warpgroup Matrix Multiply Accumulate) HGEMM targeting NVIDIA Hopper (sm_90) architecture. WGMMA operates at the warpgroup (4 warps = 128 threads) level and is significantly more powerful than MMA, enabling higher utilization of Hopper's 4th-gen Tensor Cores.

## Key Files

| File | Description |
|------|-------------|
| `hgemm_wgmma_fp16acc_stages_tn.cu` | WGMMA HGEMM with FP16 accumulator, multi-stage pipelining, TN layout |
| `hgemm_wgmma_fp32acc_stages_tn.cu` | Same but with FP32 accumulator for higher numerical precision |

## For AI Agents

### Working In This Directory
- Requires Hopper GPU (sm_90 / H100, H200). Will not compile for earlier architectures.
- `_tn` suffix: A matrix row-major, B matrix column-major (avoids transpose cost).
- FP16 accumulator is faster; FP32 accumulator is more numerically stable — choose based on use case.
- WGMMA uses `wgmma.mma_async` PTX — see the CUDA Programming Guide for Hopper architecture.

<!-- MANUAL: -->
