<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# slides

## Purpose
Reference documentation and presentations in PDF format covering CUDA programming, CUTLASS/CuTe optimization techniques, and vLLM internals. These are read-only learning resources — not source code.

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `cuda-slides/` | NVIDIA CUDA programming guides, architecture overviews, profiling docs, and CUTLASS research papers (see `cuda-slides/AGENTS.md`) |
| `vllm-slides/` | vLLM presentations, architecture diagrams, and blog materials on prefix caching and paged attention (see `vllm-slides/AGENTS.md`) |

## For AI Agents

### Working In This Directory
- This directory contains only PDF and image files — no code to compile or test.
- When a kernel in `kernels/` references a technique (e.g., swizzling, WGMMA, flash attention), the relevant explanatory paper is likely here.
- Do not modify PDFs; add new reference material only if explicitly requested.

<!-- MANUAL: -->
