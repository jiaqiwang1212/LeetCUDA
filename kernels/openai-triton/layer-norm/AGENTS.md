<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# openai-triton/layer-norm

## Purpose
Triton layer normalization with both forward and backward pass implementations. The backward pass demonstrates how Triton handles gradient computation with lock-based atomic updates for the weight gradient accumulation.

## Key Files

| File | Description |
|------|-------------|
| `triton_layer_norm.py` | Forward + backward Triton layer norm with benchmark |
| `layer-norm-forward.csv` / `.png` | Forward pass benchmark data and chart |
| `layer-norm-backward.csv` / `.png` | Backward pass benchmark data and chart |
| `bwd-math.png` | Mathematical derivation diagram for the backward pass |
| `bwd.png` | Visualization of backward pass computation graph |
| `results.html` | Full benchmark report |
| `README.md` | Algorithm explanation including backward pass math |

## For AI Agents

### Working In This Directory
- Test with `uv run python triton_layer_norm.py`.
- The backward pass is non-trivial — `bwd-math.png` explains the gradient equations before reading the code.

<!-- MANUAL: -->
