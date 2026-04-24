<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# hgemm/naive

## Purpose
Baseline FP16 GEMM implementations without Tensor Core acceleration. Starting point for understanding shared memory tiling before progressing to WMMA/MMA variants.

## Key Files

| File | Description |
|------|-------------|
| `hgemm.cu` | Shared memory tiled HGEMM using standard thread-block blocking |
| `hgemm_async.cu` | Same tiling strategy with `cp.async` for overlapping global→shared memory loads |

## For AI Agents

### Common Patterns
- `hgemm.cu`: classic `__syncthreads()`-separated load/compute loop.
- `hgemm_async.cu`: uses `cuda::memcpy_async` + `cuda::pipeline` for double buffering without explicit syncs in the load phase.

<!-- MANUAL: -->
