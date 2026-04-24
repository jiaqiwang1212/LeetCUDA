<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# flash-attn/mma/others

## Purpose
Experimental and alternative Flash Attention variants that don't fit cleanly into the basic/swizzle taxonomy — register reuse optimizations and output-to-global variants.

## Key Files

| File | Description |
|------|-------------|
| `flash_attn_mma_share_kv_F32F16F16F32_rr.cu` | Share-KV + FP32 accumulator + register reuse (`_rr`) |
| `flash_attn_mma_share_qkv_F32F16F16F32_rr.cu` | Share-QKV + FP32 accumulator + register reuse |
| `flash_attn_mma_share_qkv_Os2g.cu` | Share-QKV with output written directly to global memory (no shared output buffer) |
| `flash_attn_mma_tiling_qk_F32F16F16F32_rr.cu` | QK-tiling + FP32 accumulator + register reuse |
| `flash_attn_mma_tiling_qk_rr.cu` | QK-tiling + register reuse (FP16 accumulator) |

## For AI Agents

### Working In This Directory
- `_rr` (register reuse) variants hold more data in registers across MMA iterations, reducing shared memory traffic at the cost of higher register pressure.
- `Os2g` writes the output tensor directly to global memory instead of staging through shared memory — useful for large output tiles that don't fit in SRAM.
- These are experimental — compare against matching `basic/` variants to measure the impact.

<!-- MANUAL: -->
