<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# openai-triton

## Purpose
OpenAI Triton kernel implementations covering common GPU operations. Triton provides a Python-native GPU kernel language that compiles to PTX, enabling high-level kernel authoring without raw CUDA. Each subdirectory contains a Triton kernel with performance benchmarks against PyTorch and sometimes CUDA reference implementations.

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `vector-add/` | Hello-world Triton kernel: element-wise vector addition with performance chart |
| `fused-softmax/` | Triton fused softmax — fuses row max, subtraction, exp, and sum into one kernel pass |
| `layer-norm/` | Triton layer normalization with forward and backward pass implementations |
| `merge-attn-states/` | Merges partial attention states (used in chunked/paged attention); includes both Triton and CUDA reference |
| `matrix-multiplication/` | Reserved for Triton GEMM (currently empty) |
| `fused-attention/` | Reserved for Triton fused attention (currently empty) |

## For AI Agents

### Working In This Directory
- Install Triton: `uv pip install triton` (or it comes with recent PyTorch builds).
- Run any kernel with `uv run python triton_<op>.py`.
- Triton kernels use `@triton.jit` decorator and `tl` (triton.language) primitives — no CUDA C required.
- `merge-attn-states/` is the most complex example: compare `triton_merge_attn_states.py` with `cuda_merge_attn_states.cu` to see Triton vs CUDA implementation parity.

### Common Patterns
- `tl.load` / `tl.store` with mask for boundary-safe memory access.
- `tl.program_id` for block index (analogous to `blockIdx` in CUDA).
- `tl.constexpr` for compile-time constants (block size, etc.).

## Dependencies

### External
- OpenAI Triton ≥ 2.0 (or PyTorch ≥ 2.0 which bundles Triton)
- PyTorch ≥ 2.0

<!-- MANUAL: -->
