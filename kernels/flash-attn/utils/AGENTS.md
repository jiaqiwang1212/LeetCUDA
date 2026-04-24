<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# flash-attn/utils

## Purpose
Shared header providing MMA helper macros, online softmax state management, and timing utilities used by all flash attention kernel variants.

## Key Files

| File | Description |
|------|-------------|
| `utils.h` | Online softmax `(max, sum)` state struct, MMA register helpers, CUDA event timing, correctness check |

## For AI Agents

### Working In This Directory
- `utils.h` is included by every `.cu` in `mma/basic/`, `mma/swizzle/`, and `mma/others/`.
- The `OnlineSoftmax` or equivalent state struct here is the key data structure for Flash Attention — understand it before reading kernel code.
- A fix for guarding shared memory reads is tracked in commit `bdc28e9` ("guard online softmax shared memory read").

<!-- MANUAL: -->
