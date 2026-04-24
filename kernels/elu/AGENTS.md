<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# elu

## Purpose
CUDA kernel implementing the ELU (Exponential Linear Unit) activation function: `f(x) = x if x > 0, else alpha*(exp(x)-1)`.

## Key Files

| File | Description |
|------|-------------|
| `elu.cu` | CUDA kernel implementation |
| `elu.py` | PyTorch correctness test vs `torch.nn.functional.elu` |
| `README.md` | Algorithm description |

## For AI Agents

### Working In This Directory
- Test with `uv run python elu.py`.
- The alpha parameter is configurable; default is typically 1.0.

<!-- MANUAL: -->
