<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# kernels

## Purpose
The heart of LeetCUDA — contains 200+ CUDA kernel implementations organized by operation type. Each subdirectory is a self-contained kernel with a `.cu` CUDA implementation and a `.py` PyTorch test/benchmark. More complex kernels (hgemm, flash-attn) include pybind C++ wrappers, build tooling, and multiple implementation variants.

## Key Files

| File | Description |
|------|-------------|
| `notes-v1.cu` | Annotated reference SGEMM implementation demonstrating block tiling, shared memory, and FLOAT4 vectorized loads — a good starting point for understanding the code style |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `dot-product/` | Dot product kernel with vectorized loads (see `dot-product/AGENTS.md`) |
| `elementwise/` | Element-wise operations — reference template for adding new kernels |
| `elu/` | ELU activation (Exponential Linear Unit) |
| `embedding/` | Embedding lookup kernel |
| `gelu/` | GELU activation (Gaussian Error Linear Unit) |
| `hardshrink/` | Hard Shrink activation |
| `hardswish/` | Hard Swish activation |
| `hgemv/` | Half-precision GEMV (matrix-vector multiply), includes CuTe variant |
| `histogram/` | Histogram computation kernel |
| `layer-norm/` | Layer normalization kernel |
| `mat-transpose/` | Matrix transpose, includes CuTe variant |
| `nms/` | Non-Maximum Suppression (object detection post-processing) |
| `reduce/` | Block-level all-reduce kernels |
| `relu/` | ReLU activation |
| `rms-norm/` | RMS normalization (used in LLaMA-style models) |
| `rope/` | Rotary Position Embedding |
| `sgemm/` | Single-precision GEMM variants including WMMA TF32 and cuBLAS baseline |
| `sgemv/` | Single-precision GEMV |
| `sigmoid/` | Sigmoid activation |
| `softmax/` | Softmax kernel |
| `swish/` | Swish/SiLU activation |
| `hgemm/` | Half-precision GEMM — multiple optimization strategies achieving 98-100% cuBLAS (see `hgemm/AGENTS.md`) |
| `flash-attn/` | Flash Attention using pure MMA PTX with many tiling/sharing strategies (see `flash-attn/AGENTS.md`) |
| `cutlass/` | CUTLASS and CuTe integration examples (see `cutlass/AGENTS.md`) |
| `openai-triton/` | OpenAI Triton kernel implementations (see `openai-triton/AGENTS.md`) |
| `swizzle/` | Shared memory swizzle demonstrations for bank conflict avoidance |
| `nvidia-nsight/` | Nsight profiling examples and bank conflict analysis |
| `transformer/` | Reserved for transformer-level kernels (currently empty) |
| `ws-hgemm/` | Warp-specialized HGEMM for SM8x (Ampere) |

## For AI Agents

### Working In This Directory
- The standard kernel layout is: `<op>.cu` (CUDA kernel) + `<op>.py` (PyTorch test). Always maintain both files.
- Use `kernels/elementwise/` as the canonical template when adding a new kernel.
- Simple kernels (activation functions, reductions) live directly in their directory.
- Complex kernels (hgemm, flash-attn) use subdirectories: `mma/basic/`, `mma/swizzle/`, `pybind/`, `utils/`, `tools/`, `bench/`.
- Filenames encode precision: `F32F16F16F32` = accumulator F32, A-matrix F16, B-matrix F16, output F32.
- `_cute.cu` suffix indicates a CuTe (CUTLASS layout abstraction) variant.

### Testing Requirements
- Run `uv run python kernels/<op>/<op>.py` to test any kernel's correctness and benchmark it.
- CUDA compilation is handled inline via `torch.utils.cpp_extension` inside each `.py` file.
- For hgemm and flash-attn: use `tools/install.sh` to build the pybind extension, then `<op>.py` for benchmarks.

### Common Patterns
- `FLOAT4(value)` and `INT4(value)` macros for 128-bit vectorized global memory loads.
- `__shared__` arrays tiled as `[BM][BK]` for A-tile and `[BK][BN]` for B-tile in GEMM kernels.
- Warp-level MMA uses `mma.sync.aligned.m16n8k16` PTX instruction with F16 inputs.
- Swizzle variants use XOR-based address remapping to eliminate shared memory bank conflicts.

## Dependencies

### Internal
- `third-party/cutlass` submodule — required for `cutlass/` and `hgemm/cutlass/` kernels

### External
- CUDA Toolkit (nvcc ≥ 11.0 for MMA; ≥ 12.0 for WGMMA)
- PyTorch ≥ 2.0 with CUDA support
- OpenAI Triton (for `openai-triton/` kernels)

<!-- MANUAL: -->
