<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# flash-attn/cutlass

## Purpose
CuTe-based Flash Attention implementation using CUTLASS layout abstractions. Provides a higher-level alternative to the raw MMA PTX implementations in `mma/`.

## Key Files

| File | Description |
|------|-------------|
| `flash_attn_cute.cu` | Flash attention kernel using CuTe tiled MMA, copy atoms, and layout composition |

## For AI Agents

### Working In This Directory
- Requires `third-party/cutlass` submodule initialized.
- Compare with `../mma/basic/flash_attn_mma_split_q.cu` to see raw PTX vs CuTe expressing the same algorithm.

<!-- MANUAL: -->
