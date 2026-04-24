<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# rms-norm

## Purpose
CUDA kernel implementing RMS Normalization: `f(x) = x / sqrt(mean(x^2) + eps) * gamma`. Used in LLaMA, Mistral, and other modern LLMs instead of Layer Norm (no mean subtraction, faster).

## Key Files

| File | Description |
|------|-------------|
| `rms_norm.cu` | CUDA kernel using warp-level squared-sum reduction |
| `rms_norm.py` | PyTorch test vs manual RMS norm reference |
| `README.md` | Algorithm description and comparison with Layer Norm |

## For AI Agents

### Working In This Directory
- Test with `uv run python rms_norm.py`.
- Single-pass reduction: compute sum of squares, take sqrt, then scale — no separate mean pass needed.

<!-- MANUAL: -->
