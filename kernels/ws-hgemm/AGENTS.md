<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# ws-hgemm

## Purpose
Warp-specialized (WS) HGEMM for NVIDIA SM8x (Ampere) architecture. Separates warps into producer and consumer roles using asynchronous `cp.async` memory copies for overlapping data loading and MMA computation.

## Key Files

| File | Description |
|------|-------------|
| `naive_ws_hgemm_sm8x.cu` | Warp-specialized HGEMM targeting Ampere (sm_80) with `cp.async` producer warps |
| `ws_hgemm.py` | PyTorch benchmark vs cuBLAS baseline |
| `README.md` | Warp specialization concept and SM8x async copy pipeline |

## For AI Agents

### Working In This Directory
- Test with `uv run python ws_hgemm.py`.
- Requires Ampere (sm_80) or newer GPU; will not compile for older architectures.
- Conceptual predecessor to the WGMMA-based `hgemm/wgmma/` kernels for Hopper (sm_90).

<!-- MANUAL: -->
