<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# swish

## Purpose
CUDA kernel for the Swish/SiLU activation: `f(x) = x * sigmoid(x)`. Used in LLaMA's SwiGLU FFN layers and EfficientNet.

## Key Files

| File | Description |
|------|-------------|
| `swish.cu` | CUDA kernel combining element-wise multiply and sigmoid |
| `swish.py` | PyTorch test vs `torch.nn.functional.silu` |
| `README.md` | Algorithm notes |

## For AI Agents

### Working In This Directory
- Test with `uv run python swish.py`.

<!-- MANUAL: -->
