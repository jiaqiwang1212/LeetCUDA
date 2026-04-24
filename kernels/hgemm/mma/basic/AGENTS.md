<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# hgemm/mma/basic

## Purpose
Core MMA PTX HGEMM variants without swizzle — the foundational Tensor Core programming examples. Study these to understand the raw MMA programming model before adding swizzle optimization.

## Key Files

| File | Description |
|------|-------------|
| `hgemm_mma.cu` | Single-stage MMA HGEMM: basic `mma.sync.aligned.m16n8k16` usage with shared memory tiling |
| `hgemm_mma_stage.cu` | Multi-stage (double-buffered) MMA HGEMM with `cp.async` for NN layout |
| `hgemm_mma_stage_tn.cu` | Multi-stage MMA HGEMM for TN layout (A row-major, B col-major) — avoids transposing B |

## For AI Agents

### Working In This Directory
- Start with `hgemm_mma.cu` to understand the MMA register file layout and warp-level tile mapping.
- `_stage` adds `cp.async` double buffering — compare the two to understand pipelining overhead vs benefit.
- `_tn` variant changes the B matrix access pattern, reducing global memory transpose cost.
- These are the "reference" implementations — compare with `../swizzle/` counterparts to see bank conflict elimination.

### Common Patterns
- Each warp handles a 16×16 output tile using four `m16n8k16` MMA operations.
- Register file layout: A fragment = 8 FP16 values, B fragment = 4 FP16 values, C fragment = 4 FP32 values per `mma.sync`.
- Shared memory layout (without swizzle) causes bank conflicts on the 32-wide load — fixed in `../swizzle/`.

<!-- MANUAL: -->
