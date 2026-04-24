<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# flash-attn/mma/basic

## Purpose
Core Flash Attention MMA PTX implementations without swizzle. Covers the full algorithmic progression from FA-1 (Split-KV) to FA-2 (Split-Q) with three shared-memory strategies (none, share-KV, share-QKV) and two tiling strategies (QK-tiling, QKV-tiling). Some variants have an `F32F16F16F32` counterpart using FP32 accumulators.

## Key Files

| File | Description |
|------|-------------|
| `flash_attn_mma_split_kv.cu` | FA-1: outer loop over Q, inner over KV blocks (baseline) |
| `flash_attn_mma_split_q.cu` | FA-2: outer loop over KV, inner over Q (higher parallelism, lower memory) |
| `flash_attn_mma_share_kv.cu` | Split-Q + K and V kept in shared memory across Q iterations |
| `flash_attn_mma_share_kv_F32F16F16F32.cu` | Same with FP32 softmax accumulator |
| `flash_attn_mma_share_qkv.cu` | Split-Q + Q, K, V all in shared memory |
| `flash_attn_mma_share_qkv_F32F16F16F32.cu` | Same with FP32 accumulator |
| `flash_attn_mma_share_qkv_smooth_qkv.cu` | Share-QKV with smooth (non-swizzled) layout for comparison |
| `flash_attn_mma_tiling_qk.cu` | QK tiling: further tile Q×K computation within each block |
| `flash_attn_mma_tiling_qk_F32F16F16F32.cu` | QK tiling with FP32 accumulator |
| `flash_attn_mma_tiling_qkv.cu` | Full QKV tiling: tile Q, K, and V simultaneously |
| `flash_attn_mma_tiling_qkv_F32F16F16F32.cu` | QKV tiling with FP32 accumulator |

## For AI Agents

### Recommended Study Order
1. `split_kv` — simplest FA-1, understand online softmax loop
2. `split_q` — FA-2, understand how parallelizing over Q reduces HBM reads
3. `share_kv` — understand shared memory KV reuse
4. `share_qkv` — all three matrices in SRAM
5. `tiling_qk` / `tiling_qkv` — innermost computation tiling

### Common Patterns
- `F32F16F16F32` variants use FP32 for the running `(max, sum)` softmax state and output accumulator — more numerically stable at slight throughput cost.
- All kernels use `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` or `.f16.f16.f16.f16` PTX.
- Online softmax state is updated per KV block in the outer loop.

<!-- MANUAL: -->
