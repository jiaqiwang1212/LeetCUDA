<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# hgemm/utils

## Purpose
Shared C++ header providing timing utilities, correctness checking, and helper macros used by all HGEMM kernel variants.

## Key Files

| File | Description |
|------|-------------|
| `utils.h` | `cuda_check`, `cpu_rand_init`, `check_correctness`, GPU timing via CUDA events, FLOAT4/INT4 macros |

## For AI Agents

### Working In This Directory
- This header is included by every `.cu` file in sibling directories — changes here affect all variants.
- Modify carefully; a broken `utils.h` will cause all HGEMM builds to fail.

<!-- MANUAL: -->
