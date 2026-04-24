<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# openai-triton/merge-attn-states

## Purpose
Merges partial attention output states from chunked/paged attention computation. Essential for systems like vLLM that split attention across multiple KV cache chunks and need to combine partial `(output, softmax_lse)` states. Provides both Triton and CUDA reference implementations for comparison.

## Key Files

| File | Description |
|------|-------------|
| `triton_merge_attn_states.py` | Triton kernel for merging `(O, lse)` partial attention states |
| `cuda_merge_attn_states.cu` | CUDA reference implementation of the same operation |
| `cuda_merge_attn_states.py` | Python wrapper for the CUDA version |
| `test_merge_attn_states.py` | Correctness test comparing Triton vs CUDA vs PyTorch reference |
| `README.md` | Mathematical derivation of the state merge formula |

## For AI Agents

### Working In This Directory
- Test both implementations: `uv run python test_merge_attn_states.py`.
- The merge formula: given partial states `(O_i, lse_i)`, compute `lse = log(sum(exp(lse_i)))` then weighted-sum outputs.
- This pattern is used in vLLM's chunked prefill; see `slides/vllm-slides/` for context.

<!-- MANUAL: -->
