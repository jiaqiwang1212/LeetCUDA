<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# tensorrt/fmha

## Purpose
TensorRT FMHA (Fused Multi-Head Attention) integration examples — exporting FMHA to ONNX and using TensorRT's pattern matcher to recognize and replace attention subgraphs with optimized FMHA kernels.

## Key Files

| File | Description |
|------|-------------|
| `export_fmha.py` | Exports a PyTorch attention model to ONNX with FMHA-compatible graph structure |
| `fmha_pattern_match_ops.py` | Demonstrates TensorRT graph pattern matching for FMHA subgraph fusion |
| `README.md` | FMHA fusion setup and TensorRT version requirements |

## For AI Agents

### Working In This Directory
- Requires TensorRT ≥ 8.6 for FMHA pattern matching.
- Run `uv run python export_fmha.py` to generate the ONNX model, then use TensorRT Builder to compile.

<!-- MANUAL: -->
