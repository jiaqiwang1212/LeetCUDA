<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# vllm-slides

## Purpose
Presentations and blog materials covering vLLM internals — paged attention, prefix caching, and Triton kernel integration. Useful context for understanding the `kernels/openai-triton/merge-attn-states/` and `kernels/flash-attn/` kernels in a production LLM inference context.

## Key Files

| File | Description |
|------|-------------|
| `README.md` | Index of slides and blog posts |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `blogs/` | Drawio diagrams and PNGs illustrating vLLM prefix caching and prefill Triton kernel tiling |

## For AI Agents

### Working In This Directory
- Read-only reference material. Do not modify PDFs or diagrams.
- `blogs/vllm-automatic-prefix-caching.drawio.png` is useful when understanding chunked prefill and KV cache reuse.

<!-- MANUAL: -->
