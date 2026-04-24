<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# rope

## Purpose
CUDA kernel implementing Rotary Position Embedding (RoPE): encodes token positions by rotating query/key vectors in pairs using sin/cos tables. Used in LLaMA, PaLM, and most modern transformer LLMs.

## Key Files

| File | Description |
|------|-------------|
| `rope.cu` | CUDA kernel applying complex-number rotation to interleaved Q/K pairs |
| `rope.py` | PyTorch test vs reference RoPE implementation |
| `README.md` | Mathematical derivation and application in attention |

## For AI Agents

### Working In This Directory
- Test with `uv run python rope.py`.
- Each pair of adjacent elements `(x_{2i}, x_{2i+1})` is treated as a complex number and rotated by angle `theta_i * position`.

<!-- MANUAL: -->
