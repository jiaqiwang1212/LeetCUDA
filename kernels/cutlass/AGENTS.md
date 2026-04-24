<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# cutlass

## Purpose
Standalone CUTLASS and CuTe integration examples — separate from the `cutlass/` subdirectories inside `hgemm/` and `flash-attn/`. Demonstrates direct use of CUTLASS 3.x APIs and CuTe tiled MMA abstractions.

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `cute/` | CuTe examples: tiled MMA with texture memory (`mma_tile_tex.cc`) and basic vector add (`vector_add.cu`) |
| `cutlass-3.x/` | Reserved for CUTLASS 3.x API examples (currently empty — submodule must be initialized) |

## For AI Agents

### Working In This Directory
- Requires `third-party/cutlass` submodule: `git submodule update --init --recursive --force`.
- `cute/mma_tile_tex.cc` uses texture memory with CuTe tiled copy — compile with nvcc and link against CUTLASS headers.
- `cutlass-3.x/` will be populated as examples are added; check `third-party/cutlass/examples/` for upstream reference.

<!-- MANUAL: -->
