<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# relu

## Purpose
CUDA kernel for the ReLU activation function: `f(x) = max(0, x)`. One of the simplest kernels — useful as a sanity check that the CUDA build pipeline and PyTorch extension loading work correctly.

## Key Files

| File | Description |
|------|-------------|
| `relu.cu` | CUDA kernel implementation |
| `relu.py` | PyTorch test vs `torch.nn.functional.relu` |
| `README.md` | Usage notes |

## For AI Agents

### Working In This Directory
- Test with `uv run python relu.py`.
- Also present in `kernels/nvidia-nsight/relu.cu` with Nsight profiling annotations.

<!-- MANUAL: -->
