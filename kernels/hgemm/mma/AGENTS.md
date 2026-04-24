<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# hgemm/mma

## Purpose
Raw MMA PTX HGEMM kernels — highest control, highest performance. Uses `mma.sync.aligned` PTX instruction directly for m16n8k16 warp-level matrix multiply. Organized into basic (no swizzle) and swizzle (bank-conflict-free) subdirectories.

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `basic/` | MMA HGEMM without swizzle — demonstrates the core MMA programming model (see `basic/AGENTS.md`) |
| `swizzle/` | MMA HGEMM with XOR swizzle — production-quality, bank-conflict-free variants (see `swizzle/AGENTS.md`) |
| `others/` | Empty — reserved for future experimental variants |

## For AI Agents

### Common Patterns
- `basic/` → understand how MMA PTX works; `swizzle/` → apply bank-conflict elimination for peak performance.
- All variants use the `m16n8k16` MMA shape with FP16 inputs and FP32 or FP16 accumulator.

<!-- MANUAL: -->
