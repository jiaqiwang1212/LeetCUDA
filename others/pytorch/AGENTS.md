<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# pytorch

## Purpose
PyTorch integration examples covering distributed communication primitives and a reference PyTorch 2.x slide. The distributed tests demonstrate NCCL-based collective operations that underpin large-scale model training.

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `distributed/` | Individual test scripts for each PyTorch distributed collective (see `distributed/AGENTS.md`) |
| `slides/` | `pytorch_2.pdf` — PyTorch 2.x internals presentation (read-only reference) |
| `custom_ops/` | Reserved for custom PyTorch operator examples (currently empty) |

## For AI Agents

### Working In This Directory
- Distributed tests require multiple GPUs or a multi-process setup.
- Run with `uv run torchrun --nproc_per_node=<N> others/pytorch/distributed/<test>.py`.

<!-- MANUAL: -->
