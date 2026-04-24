<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# hgemm/wmma

## Purpose
Tensor Core HGEMM using the WMMA (Warp Matrix Multiply Accumulate) C++ API — a higher-level abstraction over raw MMA PTX. Good intermediate step between naive GEMM and raw MMA PTX kernels.

## Key Files

| File | Description |
|------|-------------|
| `hgemm_wmma.cu` | Single-stage WMMA HGEMM with warp-level 16×16×16 tile operations |
| `hgemm_wmma_stage.cu` | Multi-stage WMMA HGEMM with `cp.async` double buffering |

## For AI Agents

### Common Patterns
- Uses `nvcuda::wmma::fragment` and `wmma::mma_sync` — no raw PTX needed.
- WMMA tiles: m=16, n=16, k=16 for FP16.
- Compare with `mma/basic/` which uses equivalent raw PTX instructions for full control.

<!-- MANUAL: -->
