<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# LeetCUDA

## Purpose
A comprehensive CUDA learning repository featuring 200+ GPU kernel implementations with PyTorch bindings, high-performance GEMM (HGEMM achieving 98–100% of cuBLAS), Flash Attention variants using pure MMA PTX, OpenAI Triton examples, and extensive technical literature. Targeted at beginners and intermediate GPU programmers who want to learn CUDA by reading and running real kernels.

## Key Files

| File | Description |
|------|-------------|
| `README.md` | Project overview, benchmark results, kernel index, and blog links |
| `CONTRIBUTE.md` | Developer guide: pre-commit setup, how to add new kernels |
| `LICENSE` | GPLv3 license |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `kernels/` | All CUDA kernel implementations — activations, GEMM, attention, normalization, and more (see `kernels/AGENTS.md`) |
| `others/` | PyTorch and TensorRT integration examples (see `others/AGENTS.md`) |
| `slides/` | Reference PDFs — CUDA programming guides, CUTLASS papers, vLLM presentations (see `slides/AGENTS.md`) |
| `ffpa-attn/` | Git submodule: FFPA (Faster Flash Prefill Attention) with O(1) SRAM complexity |
| `HGEMM/` | Git submodule: standalone high-performance HGEMM library |
| `third-party/` | Git submodule: CUTLASS library (see `third-party/AGENTS.md`) |
| `docs/` | Reserved for future documentation (currently empty) |

## For AI Agents

### Working In This Repository
- Each kernel in `kernels/` follows a consistent pattern: a `.cu` file (CUDA implementation) and a `.py` file (PyTorch test/benchmark). Always keep both in sync when modifying a kernel.
- When adding a new kernel, use `kernels/elementwise/` as the reference template (per CONTRIBUTE.md).
- Run `pre-commit` before committing: `pip3 install pre-commit && pre-commit install && pre-commit run --all-files`.
- `ffpa-attn/`, `HGEMM/`, and `third-party/cutlass` are git submodules — do not modify their contents directly; update via `git submodule update`.
- The `.dev/update_submodules.sh` script automates submodule synchronization.

### Testing Requirements
- Each kernel's `.py` file serves as both a correctness test and a benchmark against PyTorch/cuBLAS baselines.
- Run Python tests with `uv run python kernels/<op>/<op>.py` (uses `uv` per project conventions).
- CUDA kernels are compiled inline via PyTorch's `torch.utils.cpp_extension` in the `.py` files.

### Common Patterns
- Kernels are self-contained: `.cu` + `.py` in one directory.
- Complex kernels (hgemm, flash-attn) add `pybind/`, `utils/`, `tools/`, `bench/` subdirectories and a `makefile`/`setup.py`.
- MMA-based kernels are organized into `basic/`, `swizzle/`, and `others/` sub-variants.
- Precision suffixes in filenames: `F32F16F16F32` means accumulator=F32, A=F16, B=F16, output=F32.

## Dependencies

### External
- CUDA Toolkit (nvcc) — kernel compilation
- PyTorch — Python test harness and correctness baselines
- CUTLASS / CuTe — used in advanced MMA kernels (`third-party/cutlass` submodule)
- OpenAI Triton — alternative GPU kernel language (`kernels/openai-triton/`)

<!-- MANUAL: -->
