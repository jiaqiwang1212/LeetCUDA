<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# tensorrt

## Purpose
TensorRT integration examples covering FMHA (Fused Multi-Head Attention) export and custom plugin development. Shows how to integrate custom CUDA kernels into a TensorRT inference engine.

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `fmha/` | FMHA pattern matching and ONNX export for TensorRT (see `fmha/AGENTS.md`) |
| `plugin/` | Custom TensorRT plugin template and documentation (see `plugin/AGENTS.md`) |

## For AI Agents

### Working In This Directory
- Requires TensorRT installation; see `README.md` for version requirements.
- FMHA export: `uv run python others/tensorrt/fmha/export_fmha.py`.
- Plugin development requires the TensorRT SDK headers and linking against `libnvinfer`.

<!-- MANUAL: -->
