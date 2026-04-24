<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# embedding

## Purpose
CUDA kernel for embedding table lookup — maps integer token indices to dense float vectors. Demonstrates irregular (gather-style) memory access patterns common in NLP workloads.

## Key Files

| File | Description |
|------|-------------|
| `embedding.cu` | CUDA kernel: index-based gather from embedding weight matrix |
| `embedding.py` | PyTorch test vs `torch.nn.Embedding` |
| `README.md` | Explanation of the lookup pattern |

## For AI Agents

### Working In This Directory
- Test with `uv run python embedding.py`.

<!-- MANUAL: -->
