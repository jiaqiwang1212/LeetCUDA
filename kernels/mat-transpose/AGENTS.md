<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# mat-transpose

## Purpose
CUDA kernels for matrix transpose, demonstrating why naive transpose causes shared memory bank conflicts and how tiled transpose with padding (or swizzling) resolves them. Includes a CuTe variant.

## Key Files

| File | Description |
|------|-------------|
| `mat_transpose.cu` | Tiled transpose kernel with shared memory padding to avoid bank conflicts |
| `mat_transpose_cute.cu` | CuTe-based transpose using CUTLASS layout abstractions |
| `mat_transpose.py` | PyTorch correctness test vs `torch.t` |
| `README.md` | Bank conflict analysis and tiling explanation |

## For AI Agents

### Working In This Directory
- Test with `uv run python mat_transpose.py`.
- The `_cute.cu` variant requires the `third-party/cutlass` submodule.
- This is a foundational example for understanding bank conflicts before studying `kernels/swizzle/`.

<!-- MANUAL: -->
