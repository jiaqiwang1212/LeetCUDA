<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# reduce

## Purpose
Block-level all-reduce kernels demonstrating the classic parallel reduction tree pattern: reduces all elements in a block to a single scalar via shared memory and warp shuffle instructions.

## Key Files

| File | Description |
|------|-------------|
| `block_all_reduce.cu` | CUDA kernel: warp-shuffle → shared memory → block reduction pipeline |
| `block_all_reduce.py` | PyTorch test vs `torch.sum` |
| `README.md` | Reduction tree explanation and warp divergence avoidance |

## For AI Agents

### Working In This Directory
- Test with `uv run python block_all_reduce.py`.
- This pattern appears as a subroutine in `layer-norm/`, `dot-product/`, `softmax/`, and `rms-norm/`.

<!-- MANUAL: -->
