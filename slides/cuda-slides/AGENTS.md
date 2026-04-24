<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# cuda-slides

## Purpose
Reference PDF library covering CUDA programming, GPU architecture, optimization techniques, and CUTLASS/CuTe internals. Contains official NVIDIA documentation alongside research papers. Read-only — use as reference when implementing or understanding kernels.

## Key Files

| File | Description |
|------|-------------|
| `CUDA_C_Programming_Guide_125.pdf` | NVIDIA's official CUDA C Programming Guide (v12.5) — authoritative reference for all CUDA APIs |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `CUTLASS/` | CUTLASS optimization papers and CuTe layout documentation (see `CUTLASS/AGENTS.md`) |

## For AI Agents

### Working In This Directory
- Do not modify these files.
- When implementing MMA or WGMMA kernels, the PTX ISA guide and architecture-specific optimization guides here are the authoritative references.
- CUTLASS papers in `CUTLASS/` explain the mathematical abstractions behind CuTe tiles and atoms.

<!-- MANUAL: -->
