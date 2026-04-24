<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# swizzle

## Purpose
Dedicated shared memory swizzle demonstration directory. Shows how XOR-based address remapping eliminates bank conflicts in GEMM and matrix transpose kernels, and how to inspect the resulting memory layout. The concepts here are applied in `hgemm/mma/swizzle/` and `flash-attn/mma/swizzle/`.

## Key Files

| File | Description |
|------|-------------|
| `hgemm_mma_swizzle.cu` | HGEMM with MMA using swizzled shared memory layout |
| `mat_trans_swizzle.cu` | Matrix transpose with XOR-swizzled shared memory |
| `mma_simple_swizzle.cu` | Minimal MMA swizzle example for learning purposes |
| `print_swizzle_layout.py` | Python script that prints the swizzle address mapping for visual inspection |
| `makefile` | Build rules for all swizzle examples |
| `README.md` | Explanation of bank conflicts, XOR swizzle math, and layout diagrams |

## For AI Agents

### Working In This Directory
- Start with `README.md` to understand bank conflicts before reading kernel code.
- Run `uv run python print_swizzle_layout.py` to visualize how XOR swizzle rearranges addresses.
- The swizzle formula is: `smem_addr_col ^= (smem_addr_row >> log2(SWIZZLE_PERIOD))`.
- This directory is self-contained — does not depend on `third-party/cutlass`.

<!-- MANUAL: -->
