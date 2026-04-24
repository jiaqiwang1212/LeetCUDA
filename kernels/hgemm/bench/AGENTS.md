<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# hgemm/bench

## Purpose
Benchmark results and profiling script for HGEMM variants across different NVIDIA GPUs.

## Key Files

| File | Description |
|------|-------------|
| `NVIDIA_GeForce_RTX_4090.png` | Benchmark chart: HGEMM variants vs cuBLAS on RTX 4090 |
| `NVIDIA_L20.png` | Benchmark chart: HGEMM variants vs cuBLAS on NVIDIA L20 |
| `NVIDIA_GeForce_RTX_3080_Laptop_GPU_WSL2.png` | Benchmark chart on RTX 3080 Laptop (WSL2) |
| `prof.py` | Nsight Systems profiling launcher for selected HGEMM variants |

## For AI Agents

### Working In This Directory
- PNG files are read-only result snapshots — they show what performance levels to expect.
- Run `uv run python bench/prof.py` from the `hgemm/` parent to launch Nsight profiling.

<!-- MANUAL: -->
