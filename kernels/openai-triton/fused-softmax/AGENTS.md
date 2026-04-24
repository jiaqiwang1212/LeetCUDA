<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# openai-triton/fused-softmax

## Purpose
Triton fused softmax kernel that fuses the row-max, subtract, exp, and sum steps into a single kernel pass, eliminating intermediate global memory writes.

## Key Files

| File | Description |
|------|-------------|
| `triton_fused_softmax.py` | Fused Triton softmax with performance benchmark vs PyTorch naive and fused |
| `softmax_kernel.ptx` | Compiled PTX output — useful for inspecting generated code |
| `softmax-performance.png` | Performance chart vs PyTorch |
| `README.md` | Fusion strategy explanation |

## For AI Agents

### Working In This Directory
- Test with `uv run python triton_fused_softmax.py`.
- The `.ptx` file is a build artifact — regenerate with `triton` if the kernel changes.

<!-- MANUAL: -->
