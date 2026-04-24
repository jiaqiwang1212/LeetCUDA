<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# hgemm/pybind

## Purpose
C++ pybind11 entry point that exposes all HGEMM CUDA kernels as a Python-importable module.

## Key Files

| File | Description |
|------|-------------|
| `hgemm.cc` | pybind11 module definition — registers each kernel variant as a Python-callable function |

## For AI Agents

### Working In This Directory
- When adding a new HGEMM kernel variant, register it here so it appears in `hgemm.py` benchmarks.
- Built by `tools/install.sh` using `torch.utils.cpp_extension.CUDAExtension`.

<!-- MANUAL: -->
