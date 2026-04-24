<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# sgemm

## Purpose
Single-precision (FP32) GEMM kernel collection progressing from naive tiled implementations to WMMA TF32 (Tensor Core) variants, with a cuBLAS baseline for performance comparison.

## Key Files

| File | Description |
|------|-------------|
| `sgemm.cu` | Shared memory tiled SGEMM — core blocked algorithm |
| `sgemm_async.cu` | SGEMM with `cp.async` pipeline for overlapping loads and compute |
| `sgemm_cublas.cu` | cuBLAS SGEMM baseline for benchmarking |
| `sgemm_wmma_tf32_stage.cu` | Tensor Core SGEMM using WMMA API with TF32 precision and multi-stage pipelining |
| `sgemm.py` | PyTorch test and benchmark across all variants |
| `README.md` | Progressive explanation from naive to Tensor Core |

## For AI Agents

### Working In This Directory
- Test with `uv run python sgemm.py`.
- Good learning progression: `sgemm.cu` → `sgemm_async.cu` → `sgemm_wmma_tf32_stage.cu`.
- TF32 on Ampere+ GPUs provides ~8x theoretical speedup over FP32 with minimal accuracy loss.

<!-- MANUAL: -->
