<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# openai-triton/vector-add

## Purpose
Triton hello-world: element-wise vector addition. The simplest possible Triton kernel — good starting point for learning `tl.load`, `tl.store`, and `tl.program_id`.

## Key Files

| File | Description |
|------|-------------|
| `triton_vector_add.py` | `@triton.jit` kernel + benchmark vs PyTorch |
| `vector-add-performance.png` | Performance comparison chart |
| `results.html` | Detailed benchmark result report |
| `README.md` | Code walkthrough |

## For AI Agents

### Working In This Directory
- Test with `uv run python triton_vector_add.py`.

<!-- MANUAL: -->
