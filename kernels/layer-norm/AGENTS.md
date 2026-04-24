<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# layer-norm

## Purpose
CUDA kernel implementing Layer Normalization: normalizes inputs across the feature dimension with learnable scale (gamma) and bias (beta). A core building block in transformer models.

## Key Files

| File | Description |
|------|-------------|
| `layer_norm.cu` | CUDA kernel using warp-level mean/variance reduction |
| `layer_norm.py` | PyTorch test vs `torch.nn.LayerNorm` |
| `README.md` | Algorithm description and numerical stability notes |

## For AI Agents

### Working In This Directory
- Test with `uv run python layer_norm.py`.
- Uses two-pass reduction: first pass computes mean, second pass computes variance (or one-pass with Welford's algorithm).
- Warp shuffle (`__shfl_down_sync`) is the primary reduction primitive.

<!-- MANUAL: -->
