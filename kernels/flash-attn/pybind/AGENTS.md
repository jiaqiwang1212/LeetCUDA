<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# flash-attn/pybind

## Purpose
C++ pybind11 entry point exposing all flash attention CUDA kernels to Python.

## Key Files

| File | Description |
|------|-------------|
| `flash_attn.cc` | pybind11 module — registers each flash attention variant as a callable Python function |

## For AI Agents

### Working In This Directory
- When adding a new flash attention variant, register the launcher function here.
- Built by `tools/install.sh` using `torch.utils.cpp_extension.CUDAExtension`.

<!-- MANUAL: -->
