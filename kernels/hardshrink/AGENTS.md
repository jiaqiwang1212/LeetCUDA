<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# hardshrink

## Purpose
CUDA kernel for the Hard Shrink activation: `f(x) = x if |x| > lambda, else 0`. Simple thresholding operation used in sparse representations.

## Key Files

| File | Description |
|------|-------------|
| `hardshrink.cu` | CUDA kernel implementation |
| `hardshrink.py` | PyTorch test vs `torch.nn.Hardshrink` |
| `README.md` | Algorithm description |

## For AI Agents

### Working In This Directory
- Test with `uv run python hardshrink.py`.

<!-- MANUAL: -->
