<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# cutlass/cute

## Purpose
Minimal CuTe examples demonstrating tiled MMA with texture memory and basic vector operations using CUTLASS layout abstractions.

## Key Files

| File | Description |
|------|-------------|
| `mma_tile_tex.cc` | CuTe tiled MMA using texture memory for matrix A — shows how to combine texture cache with CuTe atoms |
| `vector_add.cu` | Basic vector addition using CuTe `make_tensor` and `copy` — a CuTe hello-world |

## For AI Agents

### Working In This Directory
- Requires `third-party/cutlass` submodule initialized.
- `vector_add.cu` is the simplest entry point to CuTe; read it before attempting `mma_tile_tex.cc`.
- Compile: `nvcc -I../../../third-party/cutlass/include -std=c++17 vector_add.cu -o vector_add`.

<!-- MANUAL: -->
