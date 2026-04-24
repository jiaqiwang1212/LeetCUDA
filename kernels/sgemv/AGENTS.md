<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# sgemv

## Purpose
Single-precision (FP32) General Matrix-Vector multiplication (GEMV) kernel. GEMV is memory-bandwidth bound, making it a useful contrast to compute-bound GEMM kernels.

## Key Files

| File | Description |
|------|-------------|
| `sgemv.cu` | CUDA GEMV kernel with vectorized loads per row |
| `sgemv.py` | PyTorch test vs `torch.mv` |
| `README.md` | Memory access pattern and bandwidth analysis |

## For AI Agents

### Working In This Directory
- Test with `uv run python sgemv.py`.

<!-- MANUAL: -->
