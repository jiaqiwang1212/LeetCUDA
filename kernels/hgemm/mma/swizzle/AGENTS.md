<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# hgemm/mma/swizzle

## Purpose
Production-quality MMA HGEMM variants with XOR swizzle applied to shared memory layouts. These achieve peak performance by eliminating bank conflicts on shared memory loads.

## Key Files

| File | Description |
|------|-------------|
| `hgemm_mma_stage_swizzle.cu` | Multi-stage MMA HGEMM with swizzle for NN layout |
| `hgemm_mma_stage_tn_swizzle.cu` | Multi-stage MMA HGEMM with swizzle for TN layout |
| `hgemm_mma_stage_tn_swizzle_x2.cu` | TN+swizzle with 2× warp tile unrolling along N dimension |
| `hgemm_mma_stage_tn_swizzle_x4.cu` | TN+swizzle with 4× warp tile unrolling — highest throughput variant |

## For AI Agents

### Working In This Directory
- `_x2` and `_x4` variants increase instruction-level parallelism by processing more output tiles per warp, hiding MMA latency.
- XOR swizzle formula: `smem_col ^= (smem_row >> LOG2_SWIZZLE_PERIOD)` — see `kernels/swizzle/README.md` for derivation.
- These are the reference implementations that `kernels/hgemm/bench/` benchmarks against cuBLAS.
- The `_tn_swizzle_x4` variant typically achieves the highest TFLOPS.

<!-- MANUAL: -->
