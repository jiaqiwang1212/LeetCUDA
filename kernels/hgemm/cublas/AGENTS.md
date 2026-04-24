<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# hgemm/cublas

## Purpose
cuBLAS HGEMM baseline wrapper used for performance comparison across all custom HGEMM variants.

## Key Files

| File | Description |
|------|-------------|
| `hgemm_cublas.cu` | Wraps `cublasHgemm` / `cublasGemmEx` with timing harness |

## For AI Agents

### Working In This Directory
- This file is the performance target — custom kernels in sibling directories aim to match this.
- Uses `CUBLAS_GEMM_DEFAULT_TENSOR_OP` algorithm which selects the best Tensor Core path automatically.

<!-- MANUAL: -->
