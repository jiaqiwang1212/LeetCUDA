<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# hgemm/cutlass

## Purpose
CuTe-based HGEMM implementation using CUTLASS 3.x layout abstractions. Demonstrates how CuTe tiles and atoms simplify the expression of complex tiled MMA patterns compared to raw PTX.

## Key Files

| File | Description |
|------|-------------|
| `hgemm_mma_stage_tn_cute.cu` | Multi-stage pipelined HGEMM using CuTe tiled_mma, copy atoms, and layout composition |

## For AI Agents

### Working In This Directory
- Requires `third-party/cutlass` submodule initialized.
- Compare with `../mma/swizzle/hgemm_mma_stage_tn_swizzle.cu` to see raw PTX vs CuTe expressing the same algorithm.

<!-- MANUAL: -->
