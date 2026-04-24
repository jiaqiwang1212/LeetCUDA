<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# hardswish

## Purpose
CUDA kernel for the Hard Swish activation: `f(x) = x * ReLU6(x+3)/6`. A computationally efficient approximation of the Swish activation used in MobileNetV3.

## Key Files

| File | Description |
|------|-------------|
| `hardswish.cu` | CUDA kernel implementation |
| `hardswish.py` | PyTorch test vs `torch.nn.Hardswish` |
| `README.md` | Algorithm description |

## For AI Agents

### Working In This Directory
- Test with `uv run python hardswish.py`.

<!-- MANUAL: -->
