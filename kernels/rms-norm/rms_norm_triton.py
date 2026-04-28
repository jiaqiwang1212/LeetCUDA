"""
Triton implementations of RMS Norm kernels from rms_norm.cu.

Each Triton program handles one row of K elements (grid=(N,)).
BLOCK_SIZE = next_power_of_2(K) so a single program covers the full row in one
pass — matching the CUDA pattern of one block per row.

Variants and their CUDA analogues:
  f32             - FP32 in/out, FP32 accumulator (rms_norm_f32_kernel)
  f32x4           - FP32 in/out, FP32 acc (float4 vectorization transparent in Triton)
  f16_f16         - FP16 in/out, FP16 accumulator for sum-of-squares
  f16_f32         - FP16 in/out, FP32 accumulator (higher precision, no overflow)
  f16x2_f16       - FP16 in/out, FP16 acc (half2 vectorization transparent in Triton)
  f16x8_f16       - FP16 in/out, FP16 acc
  f16x8_f32       - FP16 in/out, FP32 acc
  f16x8_pack_f16  - FP16 in/out, FP16 acc (128-bit LDST transparent in Triton)
  f16x8_pack_f32  - FP16 in/out, FP32 acc
"""

import time
from typing import Optional

import torch
import triton
import triton.language as tl

DEVICE = torch.device("cuda:0")


# ---------------------------------------------------------------------------
# Triton kernels — one program per row
# ---------------------------------------------------------------------------


@triton.jit
def rms_norm_f32_kernel(x_ptr, y_ptr, g, N, K, BLOCK_SIZE: tl.constexpr):
    """FP32 in/out, FP32 accumulator."""
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < K
    x = tl.load(x_ptr + row * K + offsets, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / K
    inv_rms = 1.0 / tl.sqrt(var + 1e-5)
    tl.store(y_ptr + row * K + offsets, (x * inv_rms * g).to(tl.float32), mask=mask)


@triton.jit
def rms_norm_f16_f16_kernel(x_ptr, y_ptr, g, N, K, BLOCK_SIZE: tl.constexpr):
    """FP16 in/out, FP16 accumulator for sum-of-squares.

    Mirrors CUDA kernels that keep the variance reduction in fp16.
    Susceptible to overflow on large input values (see benchmark overflow section).
    Final rsqrt and output scaling are done in fp32 for numerical stability.
    """
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < K
    x = tl.load(x_ptr + row * K + offsets, mask=mask, other=0.0).to(tl.float16)
    # sum-of-squares in f16; promote to f32 only for rsqrt
    var = tl.sum(x * x, axis=0).to(tl.float32) / K
    inv_rms = 1.0 / tl.sqrt(var + 1e-5)
    tl.store(y_ptr + row * K + offsets, (x.to(tl.float32) * inv_rms * g).to(tl.float16), mask=mask)


@triton.jit
def rms_norm_f16_f32_kernel(x_ptr, y_ptr, g, N, K, BLOCK_SIZE: tl.constexpr):
    """FP16 in/out, FP32 accumulator for sum-of-squares.

    Mirrors CUDA kernels that upcast to fp32 before the variance reduction.
    Numerically stable under large inputs; equivalent to rms_norm_f16_f32_kernel in CUDA.
    """
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < K
    x = tl.load(x_ptr + row * K + offsets, mask=mask, other=0.0).to(tl.float16)
    xf = x.to(tl.float32)
    var = tl.sum(xf * xf, axis=0) / K  # accumulate in f32 — no overflow
    inv_rms = 1.0 / tl.sqrt(var + 1e-5)
    tl.store(y_ptr + row * K + offsets, (xf * inv_rms * g).to(tl.float16), mask=mask)


# ---------------------------------------------------------------------------
# Python wrappers — each takes (x, g) and returns y, matching the out=None
# path in run_benchmark (same as naive_rms_norm / torch baseline).
# ---------------------------------------------------------------------------


def rms_norm_f32(x: torch.Tensor, g: float) -> torch.Tensor:
    N, K = x.shape
    y = torch.empty_like(x)
    rms_norm_f32_kernel[(N,)](x, y, g, N, K, BLOCK_SIZE=triton.next_power_of_2(K))
    return y


def rms_norm_f32x4(x: torch.Tensor, g: float) -> torch.Tensor:
    """float4 vectorization is transparent in Triton; delegates to rms_norm_f32_kernel."""
    N, K = x.shape
    y = torch.empty_like(x)
    rms_norm_f32_kernel[(N,)](x, y, g, N, K, BLOCK_SIZE=triton.next_power_of_2(K))
    return y


def rms_norm_f16_f16(x: torch.Tensor, g: float) -> torch.Tensor:
    N, K = x.shape
    y = torch.empty_like(x)
    rms_norm_f16_f16_kernel[(N,)](x, y, g, N, K, BLOCK_SIZE=triton.next_power_of_2(K))
    return y


def rms_norm_f16_f32(x: torch.Tensor, g: float) -> torch.Tensor:
    N, K = x.shape
    y = torch.empty_like(x)
    rms_norm_f16_f32_kernel[(N,)](x, y, g, N, K, BLOCK_SIZE=triton.next_power_of_2(K))
    return y


def rms_norm_f16x2_f16(x: torch.Tensor, g: float) -> torch.Tensor:
    """half2 vectorization is transparent in Triton; delegates to rms_norm_f16_f16_kernel."""
    N, K = x.shape
    y = torch.empty_like(x)
    rms_norm_f16_f16_kernel[(N,)](x, y, g, N, K, BLOCK_SIZE=triton.next_power_of_2(K))
    return y


def rms_norm_f16x8_f16(x: torch.Tensor, g: float) -> torch.Tensor:
    N, K = x.shape
    y = torch.empty_like(x)
    rms_norm_f16_f16_kernel[(N,)](x, y, g, N, K, BLOCK_SIZE=triton.next_power_of_2(K))
    return y


def rms_norm_f16x8_f32(x: torch.Tensor, g: float) -> torch.Tensor:
    N, K = x.shape
    y = torch.empty_like(x)
    rms_norm_f16_f32_kernel[(N,)](x, y, g, N, K, BLOCK_SIZE=triton.next_power_of_2(K))
    return y


def rms_norm_f16x8_pack_f16(x: torch.Tensor, g: float) -> torch.Tensor:
    """128-bit LDST pack is transparent in Triton; delegates to rms_norm_f16_f16_kernel."""
    N, K = x.shape
    y = torch.empty_like(x)
    rms_norm_f16_f16_kernel[(N,)](x, y, g, N, K, BLOCK_SIZE=triton.next_power_of_2(K))
    return y


def rms_norm_f16x8_pack_f32(x: torch.Tensor, g: float) -> torch.Tensor:
    """128-bit LDST pack is transparent in Triton; delegates to rms_norm_f16_f32_kernel."""
    N, K = x.shape
    y = torch.empty_like(x)
    rms_norm_f16_f32_kernel[(N,)](x, y, g, N, K, BLOCK_SIZE=triton.next_power_of_2(K))
    return y


# ---------------------------------------------------------------------------
# Benchmark harness — mirrors rms_norm.py structure exactly
# ---------------------------------------------------------------------------


def naive_rms_norm(x: torch.Tensor, g: float) -> torch.Tensor:
    s_rms = torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-5)
    return (x * s_rms) * g


def run_benchmark(
    perf_func: callable,
    x: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 1000,
    show_all: bool = False,
):
    g = 1.0
    if out is not None:
        out.fill_(0)
    if out is not None:
        for _ in range(warmup):
            perf_func(x, out, g)
    else:
        for _ in range(warmup):
            _ = perf_func(x, g)
    torch.cuda.synchronize()
    start = time.time()
    if out is not None:
        for _ in range(iters):
            perf_func(x, out, g)
    else:
        for _ in range(iters):
            out = perf_func(x, g)
    torch.cuda.synchronize()
    end = time.time()
    mean_time = (end - start) * 1000 / iters
    out_info = f"out_{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()[:3]
    out_val = [round(v, 8) for v in out_val]
    out_val = [f"{v:<12}" for v in out_val]
    print(f"{out_info:>22}: {out_val}, time:{mean_time:.8f}ms")
    if show_all:
        print(out)
    return out, mean_time


print("-" * 90)
N, K = 4096, 512
print(" " * 40 + f"N={N}, K={K}")
x = torch.randn((N, K), device=DEVICE, dtype=torch.float32).contiguous()
run_benchmark(rms_norm_f32,    x, "f32_triton")
run_benchmark(rms_norm_f32x4,  x, "f32x4_triton")
run_benchmark(naive_rms_norm,  x, "f32_th")

print("-" * 90)
x_f16 = x.half()
run_benchmark(rms_norm_f16_f16,       x_f16, "f16f16_triton")
run_benchmark(rms_norm_f16_f32,       x_f16, "f16f32_triton")
run_benchmark(rms_norm_f16x2_f16,     x_f16, "f16x2f16_triton")
run_benchmark(rms_norm_f16x8_f16,     x_f16, "f16x8f16_triton")
run_benchmark(rms_norm_f16x8_f32,     x_f16, "f16x8f32_triton")
run_benchmark(rms_norm_f16x8_pack_f16, x_f16, "f16x8packf16_triton")
run_benchmark(rms_norm_f16x8_pack_f32, x_f16, "f16x8packf32_triton")
run_benchmark(naive_rms_norm,          x_f16, "f16_th")

print("-" * 90)
print(" " * 40 + f"f16 overflow without f32 acc")
print("-" * 90)
x_f16_ov = x.half() * 100  # triggers overflow in f16 acc variants
run_benchmark(rms_norm_f16_f16,       x_f16_ov, "f16f16_triton")
run_benchmark(rms_norm_f16_f32,       x_f16_ov, "f16f32_triton")
run_benchmark(rms_norm_f16x8_f16,     x_f16_ov, "f16x8f16_triton")
run_benchmark(rms_norm_f16x8_f32,     x_f16_ov, "f16x8f32_triton")
run_benchmark(rms_norm_f16x8_pack_f16, x_f16_ov, "f16x8packf16_triton")
run_benchmark(rms_norm_f16x8_pack_f32, x_f16_ov, "f16x8packf32_triton")
run_benchmark(naive_rms_norm,          x_f16_ov, "f16_th")

print("-" * 90)
N, K = 4096, 1024
print(" " * 40 + f"N={N}, K={K}")
x = torch.randn((N, K), device=DEVICE, dtype=torch.float32).contiguous()
run_benchmark(rms_norm_f32,   x, "f32_triton")
run_benchmark(rms_norm_f32x4, x, "f32x4_triton")
run_benchmark(naive_rms_norm, x, "f32_th")

print("-" * 90)
x_f16 = x.half()
run_benchmark(rms_norm_f16_f16,       x_f16, "f16f16_triton")
run_benchmark(rms_norm_f16_f32,       x_f16, "f16f32_triton")
run_benchmark(rms_norm_f16x2_f16,     x_f16, "f16x2f16_triton")
run_benchmark(rms_norm_f16x8_f16,     x_f16, "f16x8f16_triton")
run_benchmark(rms_norm_f16x8_f32,     x_f16, "f16x8f32_triton")
run_benchmark(rms_norm_f16x8_pack_f16, x_f16, "f16x8packf16_triton")
run_benchmark(rms_norm_f16x8_pack_f32, x_f16, "f16x8packf32_triton")
run_benchmark(naive_rms_norm,          x_f16, "f16_th")

print("-" * 90)
N, K = 4096, 2048
print(" " * 40 + f"N={N}, K={K}")
x = torch.randn((N, K), device=DEVICE, dtype=torch.float32).contiguous()
run_benchmark(rms_norm_f32x4, x, "f32x4_triton")
run_benchmark(naive_rms_norm, x, "f32_th")

print("-" * 90)
x_f16 = x.half()
run_benchmark(rms_norm_f16x2_f16,     x_f16, "f16x2f16_triton")
run_benchmark(rms_norm_f16x8_f16,     x_f16, "f16x8f16_triton")
run_benchmark(rms_norm_f16x8_f32,     x_f16, "f16x8f32_triton")
run_benchmark(rms_norm_f16x8_pack_f16, x_f16, "f16x8packf16_triton")
run_benchmark(rms_norm_f16x8_pack_f32, x_f16, "f16x8packf32_triton")
run_benchmark(naive_rms_norm,          x_f16, "f16_th")

print("-" * 90)
N, K = 4096, 4096
print(" " * 40 + f"N={N}, K={K}")
x = torch.randn((N, K), device=DEVICE, dtype=torch.float32).contiguous()
run_benchmark(rms_norm_f32x4, x, "f32x4_triton")
run_benchmark(naive_rms_norm, x, "f32_th")

print("-" * 90)
x_f16 = x.half()
run_benchmark(rms_norm_f16x8_f16,     x_f16, "f16x8f16_triton")
run_benchmark(rms_norm_f16x8_f32,     x_f16, "f16x8f32_triton")
run_benchmark(rms_norm_f16x8_pack_f16, x_f16, "f16x8packf16_triton")
run_benchmark(rms_norm_f16x8_pack_f32, x_f16, "f16x8packf32_triton")
run_benchmark(naive_rms_norm,          x_f16, "f16_th")

print("-" * 90)
N, K = 4096, 8192
print(" " * 40 + f"N={N}, K={K}")
x_f16 = torch.randn((N, K), device=DEVICE, dtype=torch.float16).contiguous()
run_benchmark(rms_norm_f16x8_f16,     x_f16, "f16x8f16_triton")
run_benchmark(rms_norm_f16x8_f32,     x_f16, "f16x8f32_triton")
run_benchmark(rms_norm_f16x8_pack_f16, x_f16, "f16x8packf16_triton")
run_benchmark(rms_norm_f16x8_pack_f32, x_f16, "f16x8packf32_triton")
run_benchmark(naive_rms_norm,          x_f16, "f16_th")

print("-" * 90)
N, K = 8192, 8192
print(" " * 40 + f"N={N}, K={K}")
x_f16 = torch.randn((N, K), device=DEVICE, dtype=torch.float16).contiguous()
run_benchmark(rms_norm_f16x8_f16,     x_f16, "f16x8f16_triton")
run_benchmark(rms_norm_f16x8_f32,     x_f16, "f16x8f32_triton")
run_benchmark(rms_norm_f16x8_pack_f16, x_f16, "f16x8packf16_triton")
run_benchmark(rms_norm_f16x8_pack_f32, x_f16, "f16x8packf32_triton")
run_benchmark(naive_rms_norm,          x_f16, "f16_th")
print("-" * 90)
