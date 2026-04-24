<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# histogram

## Purpose
CUDA kernel for computing a histogram of integer values. Demonstrates atomic operations (`atomicAdd`) for concurrent bin updates and strategies for reducing atomic contention via shared memory privatization.

## Key Files

| File | Description |
|------|-------------|
| `histogram.cu` | CUDA kernel with shared memory privatization and global atomic merge |
| `histogram.py` | PyTorch test vs `torch.histc` |
| `README.md` | Explanation of atomic contention and privatization strategy |

## For AI Agents

### Working In This Directory
- Test with `uv run python histogram.py`.

<!-- MANUAL: -->
