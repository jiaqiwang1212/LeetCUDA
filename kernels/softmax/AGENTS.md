<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# softmax

## Purpose
CUDA softmax kernel implementing the numerically stable "online softmax" (safe softmax) algorithm: `softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))`. Core operation in attention mechanisms.

## Key Files

| File | Description |
|------|-------------|
| `softmax.cu` | Online safe softmax kernel with shared memory max/sum reduction |
| `softmax.py` | PyTorch test vs `torch.nn.functional.softmax` |
| `README.md` | Numerical stability explanation and online algorithm derivation |

## For AI Agents

### Working In This Directory
- Test with `uv run python softmax.py`.
- The "online" variant computes max and sum in a single pass using a running maximum trick — prerequisite concept for understanding Flash Attention.
- See also `kernels/flash-attn/` where per-block online softmax is the core innovation.

<!-- MANUAL: -->
