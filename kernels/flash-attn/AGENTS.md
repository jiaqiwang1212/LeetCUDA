<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# flash-attn

## Purpose
Flash Attention implementations using pure MMA PTX (no CUTLASS abstractions). Covers the full algorithmic progression from FA-1 (Split-KV) to FA-2 (Split-Q) with shared KV, shared QKV, and QK/QKV tiling strategies. Each strategy has basic, swizzle, and experimental variants. Also includes a CuTe reference implementation.

## Key Files

| File | Description |
|------|-------------|
| `flash_attn_mma.py` | Main Python benchmark: compares all MMA flash-attn variants vs PyTorch SDPA |
| `makefile` | nvcc build rules |
| `setup.py` | pybind11 package build |
| `README.md` | Algorithm explanation, benchmark results, and variant taxonomy |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `mma/basic/` | Core flash attention variants without swizzle (see `mma/basic/AGENTS.md`) |
| `mma/swizzle/` | Same variants with XOR swizzle for bank-conflict-free shared memory (see `mma/swizzle/AGENTS.md`) |
| `mma/others/` | Experimental and alternative variants (`_rr` = register reuse, `Os2g` = output to global) |
| `cutlass/` | CuTe-based flash attention using CUTLASS abstractions |
| `pybind/` | C++ pybind11 entry point (`flash_attn.cc`) |
| `utils/` | Shared header (`utils.h`) with MMA helper macros and online softmax utilities |
| `tools/` | Build/install scripts and swizzle layout printer |
| `bench/` | Reserved for benchmark result images (currently empty) |

## For AI Agents

### Working In This Directory
- **Install**: `bash tools/install.sh`, then `import flash_attn_mma` in Python.
- **Benchmark**: `uv run python flash_attn_mma.py`.
- When studying the code, start with `mma/basic/flash_attn_mma_split_kv.cu` (simplest FA-1) then progress to `flash_attn_mma_split_q.cu` (FA-2).

### Variant Taxonomy (filename encoding)
- `split_kv` — FA-1 style: outer loop over Q rows, inner over KV blocks
- `split_q` — FA-2 style: outer loop over KV, inner over Q blocks (higher parallelism)
- `share_kv` / `share_qkv` — K and/or V are kept in shared memory across Q iterations
- `tiling_qk` / `tiling_qkv` — additional tiling of Q×K and/or V in shared memory
- `F32F16F16F32` — accumulator F32, A=F16, B=F16, output F32 (vs default all-F16)
- `swizzle_q/qk/qkv` — which of Q, K, V use XOR swizzle layout
- `_rr` — register reuse optimization variant

### Common Patterns
- Online softmax: running `(max, sum)` state updated per KV block; see `utils/utils.h` for the `OnlineSoftmax` helper.
- All MMA kernels use `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` PTX.
- Guard shared memory reads with boundary checks (see bug fix in commit bdc28e9).

## Dependencies

### Internal
- `third-party/cutlass` — required for `cutlass/` subdirectory

### External
- CUDA ≥ 11.0 (MMA PTX); sm_80+ recommended for `cp.async`
- PyTorch ≥ 2.0 (for SDPA baseline comparison)

<!-- MANUAL: -->
