<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# others

## Purpose
Integration examples showing how CUDA kernels connect to higher-level frameworks — specifically PyTorch distributed communication primitives and TensorRT custom plugin/FMHA patterns. These are practical usage examples rather than from-scratch kernel implementations.

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `pytorch/` | PyTorch distributed communication tests and a PyTorch 2.x slide (see `pytorch/AGENTS.md`) |
| `tensorrt/` | TensorRT FMHA export and plugin integration examples (see `tensorrt/AGENTS.md`) |

## For AI Agents

### Working In This Directory
- These are framework-level examples, not raw CUDA kernels — they depend on PyTorch and TensorRT being installed.
- Run PyTorch distributed tests with `uv run python others/pytorch/distributed/<test>.py`.
- TensorRT examples require a TensorRT installation; see `others/tensorrt/README.md` for setup.

### Common Patterns
- Distributed tests use `torch.distributed` with NCCL backend and span multiple processes via `torchrun` or `torch.multiprocessing.spawn`.

## Dependencies

### External
- PyTorch ≥ 2.0 with NCCL support (for distributed tests)
- TensorRT (for tensorrt/ examples)

<!-- MANUAL: -->
