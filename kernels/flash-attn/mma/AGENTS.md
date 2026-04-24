<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# flash-attn/mma

## Purpose
MMA PTX Flash Attention kernels organized by memory-sharing strategy. Each subdirectory groups variants of the same strategy (with and without swizzle, with and without FP32 accumulator, etc.).

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `basic/` | Core flash attention MMA variants without swizzle (see `basic/AGENTS.md`) |
| `swizzle/` | Same variants with XOR swizzle for bank-conflict-free shared memory (see `swizzle/AGENTS.md`) |
| `others/` | Experimental variants: register-reuse (`_rr`) and output-to-global (`Os2g`) optimizations |

## For AI Agents

### Working In This Directory
- Study order: `basic/flash_attn_mma_split_kv.cu` → `split_q` → `share_kv` → `share_qkv` → `tiling_qk` → `tiling_qkv`.
- Then compare any `basic/` variant with its matching `swizzle/` counterpart to learn bank conflict elimination.

<!-- MANUAL: -->
