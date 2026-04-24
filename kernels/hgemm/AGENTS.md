<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# hgemm

## Purpose
High-performance Half-precision GEMM (HGEMM) library achieving 98–100% of cuBLAS TFLOPS on L20, RTX 4090, and RTX 3080 Laptop. Implements a full optimization progression from naive FP16 GEMM through WMMA, MMA PTX, and WGMMA (Hopper), with swizzle variants and a CuTe/CUTLASS implementation. This is a standalone Python-installable library with pybind11 bindings.

## Key Files

| File | Description |
|------|-------------|
| `hgemm.py` | Main benchmark script comparing all HGEMM variants vs cuBLAS |
| `makefile` | nvcc build rules for all `.cu` variants |
| `setup.py` | `torch.utils.cpp_extension` build for pybind11 installable package |
| `README.md` | Benchmark results, optimization roadmap, and API documentation |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `naive/` | Baseline FP16 GEMM: shared memory tiling and async copy variant (see `naive/AGENTS.md`) |
| `wmma/` | Tensor Core GEMM via WMMA API (warp-level abstraction) (see `wmma/AGENTS.md`) |
| `mma/` | MMA PTX GEMM — basic and swizzle variants (see `mma/AGENTS.md`) |
| `wgmma/` | WGMMA GEMM targeting Hopper (sm_90), FP16 and FP32 accumulators (see `wgmma/AGENTS.md`) |
| `cublas/` | cuBLAS baseline wrapper for performance comparison (see `cublas/AGENTS.md`) |
| `cutlass/` | CuTe-based HGEMM using CUTLASS layout abstractions (see `cutlass/AGENTS.md`) |
| `pybind/` | C++ pybind11 entry point (`hgemm.cc`) exposing kernels to Python |
| `utils/` | Shared C++ header (`utils.h`) with timing, correctness checks, and helper macros |
| `tools/` | Build/install scripts and swizzle layout printer |
| `bench/` | Benchmark result images (RTX 4090, L20, RTX 3080) and profiling script |

## For AI Agents

### Working In This Directory
- **Install as a Python package**: `cd kernels/hgemm && bash tools/install.sh`, then use `import hgemm` in Python.
- **Run benchmarks**: `uv run python hgemm.py` (after installation).
- **Build individual kernels**: `make` from this directory.
- All kernel variants share `utils/utils.h` — modify it carefully, as it affects every variant.
- The `pybind/hgemm.cc` file registers all exported CUDA functions; add new variants here when creating new kernels.

### Testing Requirements
- `uv run python hgemm.py` benchmarks correctness and TFLOPS for all variants.
- `bench/prof.py` runs Nsight Systems profiling on selected kernels.

### Common Patterns
- Naming convention: `hgemm_mma_stage_tn_swizzle_x4.cu` = MMA + multi-stage pipelining + TN layout + swizzle + x4 unrolling.
- `_tn` suffix: A in row-major, B in column-major (transposed-N) — avoids explicit matrix transposition.
- `_swizzle` suffix: XOR-based shared memory layout to eliminate bank conflicts.
- `_stage` suffix: multi-stage (`cp.async`) double/triple buffering for latency hiding.

## Dependencies

### Internal
- `third-party/cutlass` — required for `cutlass/` subdirectory only

### External
- CUDA ≥ 11.0 (WMMA/MMA), ≥ 12.0 (WGMMA/sm_90)
- PyTorch ≥ 2.0 with CUDA

<!-- MANUAL: -->
