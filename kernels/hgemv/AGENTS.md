<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# hgemv

## Purpose
Half-precision (FP16) General Matrix-Vector multiplication (GEMV) kernels. Includes a standard MMA-based implementation and a CuTe variant demonstrating the CUTLASS layout abstraction library.

## Key Files

| File | Description |
|------|-------------|
| `hgemv.cu` | Standard CUDA HGEMV kernel using warp-level reduction |
| `hgemv_cute.cu` | CuTe-based HGEMV using CUTLASS layout abstractions |
| `hgemv.py` | PyTorch correctness test and benchmark vs `torch.mv` with FP16 |
| `README.md` | Algorithm and performance notes |

## For AI Agents

### Working In This Directory
- Test with `uv run python hgemv.py`.
- The `_cute.cu` variant requires the `third-party/cutlass` submodule to be initialized.

<!-- MANUAL: -->
