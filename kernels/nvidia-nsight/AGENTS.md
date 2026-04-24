<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# nvidia-nsight

## Purpose
Profiling and analysis examples specifically designed to be used with NVIDIA Nsight Systems/Compute. Demonstrates how to identify performance bottlenecks — particularly shared memory bank conflicts — using profiling tools alongside annotated CUDA kernels.

## Key Files

| File | Description |
|------|-------------|
| `elementwise.cu` | Annotated elementwise kernel with Nsight NVTX range markers |
| `relu.cu` | ReLU kernel instrumented for Nsight profiling |
| `bank_conflicts.md` | Written guide: how to detect and interpret bank conflicts in Nsight Compute |
| `README.md` | Instructions for launching Nsight Systems/Compute with these kernels |

## For AI Agents

### Working In This Directory
- These kernels are meant to be profiled, not just run for correctness.
- Launch with: `nsys profile --trace=cuda uv run python <script>` or `ncu --set full ./kernel`.
- `bank_conflicts.md` is a standalone reference document — read it before studying the swizzle kernels.

<!-- MANUAL: -->
