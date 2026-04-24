<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# sigmoid

## Purpose
CUDA kernel for the Sigmoid activation: `f(x) = 1 / (1 + exp(-x))`. Demonstrates use of CUDA math intrinsics (`__expf`) for fast approximation.

## Key Files

| File | Description |
|------|-------------|
| `sigmoid.cu` | CUDA kernel using `__expf` intrinsic for speed |
| `sigmoid.py` | PyTorch test vs `torch.sigmoid` |
| `README.md` | Precision vs speed tradeoffs for exp intrinsics |

## For AI Agents

### Working In This Directory
- Test with `uv run python sigmoid.py`.

<!-- MANUAL: -->
