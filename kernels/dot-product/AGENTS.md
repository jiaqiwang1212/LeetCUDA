<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# dot-product

## Purpose
CUDA kernel for computing the dot product of two vectors, demonstrating vectorized global memory loads (FLOAT4) and warp-level reduction via shuffle instructions.

## Key Files

| File | Description |
|------|-------------|
| `dot_product.cu` | CUDA kernel implementation with vectorized loads and warp shuffle reduction |
| `dot_product.py` | PyTorch correctness test and benchmark against `torch.dot` |
| `README.md` | Explanation of the algorithm and memory access pattern |

## For AI Agents

### Working In This Directory
- Modify both `.cu` and `.py` together; the Python file compiles the CUDA kernel inline via `torch.utils.cpp_extension`.
- Test with `uv run python dot_product.py`.

### Common Patterns
- Uses `FLOAT4` macro for 128-bit vectorized loads.
- Warp reduction via `__shfl_down_sync` before shared memory accumulation.

<!-- MANUAL: -->
