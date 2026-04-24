<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# gelu

## Purpose
CUDA kernel implementing the GELU (Gaussian Error Linear Unit) activation: `f(x) = 0.5*x*(1 + erf(x/sqrt(2)))`. Widely used in transformer models (BERT, GPT).

## Key Files

| File | Description |
|------|-------------|
| `gelu.cu` | CUDA kernel — uses fast approximation or exact erf |
| `gelu.py` | PyTorch test vs `torch.nn.functional.gelu` |
| `README.md` | Algorithm and approximation notes |

## For AI Agents

### Working In This Directory
- Test with `uv run python gelu.py`.
- Some implementations use the tanh approximation: `0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))`.

<!-- MANUAL: -->
