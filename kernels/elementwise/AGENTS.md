<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# elementwise

## Purpose
Simple element-wise vector operation kernel — the canonical template for adding new kernels to LeetCUDA. Per CONTRIBUTE.md, new contributors should follow this directory's structure exactly.

## Key Files

| File | Description |
|------|-------------|
| `elementwise.cu` | CUDA kernel: element-wise addition with FLOAT4 vectorized loads |
| `elementwise.py` | PyTorch test and benchmark |
| `README.md` | Usage instructions |

## For AI Agents

### Working In This Directory
- This is the reference template. When instructed to add a new kernel elsewhere, replicate this directory's layout.
- Test with `uv run python elementwise.py`.

<!-- MANUAL: -->
