<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# flash-attn/mma/swizzle

## Purpose
Bank-conflict-free Flash Attention variants matching the algorithms in `../basic/` but with XOR swizzle applied to Q, K, and/or V shared memory tiles. The swizzle suffix (`_swizzle_q`, `_swizzle_qk`, `_swizzle_qkv`) indicates which matrices use the swizzled layout.

## Key Files

| File | Description |
|------|-------------|
| `flash_attn_mma_share_kv_swizzle_q.cu` | Share-KV with swizzle on Q tile only |
| `flash_attn_mma_share_kv_swizzle_qk.cu` | Share-KV with swizzle on Q and K tiles |
| `flash_attn_mma_share_kv_swizzle_qkv.cu` | Share-KV with swizzle on all three tiles |
| `flash_attn_mma_share_qkv_swizzle_q.cu` | Share-QKV with swizzle on Q only |
| `flash_attn_mma_share_qkv_swizzle_qk.cu` | Share-QKV with swizzle on Q and K |
| `flash_attn_mma_share_qkv_swizzle_qkv.cu` | Share-QKV with swizzle on all tiles |
| `flash_attn_mma_tiling_qk_swizzle_q.cu` | QK-tiling + swizzle Q |
| `flash_attn_mma_tiling_qk_swizzle_qk.cu` | QK-tiling + swizzle Q and K |
| `flash_attn_mma_tiling_qk_swizzle_qkv.cu` | QK-tiling + swizzle all |
| `flash_attn_mma_tiling_qkv_swizzle_q.cu` | QKV-tiling + swizzle Q |
| `flash_attn_mma_tiling_qkv_swizzle_q_F32F16F16F32.cu` | QKV-tiling + swizzle Q + FP32 accumulator |
| `flash_attn_mma_tiling_qkv_swizzle_qk.cu` | QKV-tiling + swizzle Q and K |
| `flash_attn_mma_tiling_qkv_swizzle_qk_F32F16F16F32.cu` | QKV-tiling + swizzle Q and K + FP32 accumulator |
| `flash_attn_mma_tiling_qkv_swizzle_qkv.cu` | QKV-tiling + swizzle all — maximum optimization |
| `flash_attn_mma_tiling_qkv_swizzle_qkv_F32F16F16F32.cu` | QKV-tiling + swizzle all + FP32 accumulator |

## For AI Agents

### Working In This Directory
- Pick the variant matching your target `basic/` algorithm, then compare the swizzle variant to see the layout change.
- `_swizzle_qkv` variants (all three matrices swizzled) typically give the highest throughput.
- `F32F16F16F32` variants trade ~5% throughput for improved numerical precision in the softmax and output.
- Use `tools/print_swizzle_layout.py` to visualize the specific swizzle pattern for a given head dimension.

<!-- MANUAL: -->
