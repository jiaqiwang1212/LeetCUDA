"""
Triton implementations of dot product kernels from dot_product.cu.

Each Triton program loads a BLOCK_SIZE-element tile from both input vectors,
computes a partial dot product with tl.sum, then atomically accumulates into
a pre-zeroed scalar output tensor (stored as a 1-element float32 tensor).
This mirrors the CUDA pattern: thread-local multiply-accumulate → warp
shuffle reduction → atomicAdd to global output.

Variants and their CUDA analogues:
  f32_f32          - FP32 inputs, FP32 accumulator, BLOCK_SIZE=1024
  f32x4_f32        - FP32 inputs, BLOCK_SIZE=1024 (mirrors float4 load)
  f16_f32          - FP16 inputs, FP32 accumulator, BLOCK_SIZE=1024
  f16x2_f32        - FP16 inputs, BLOCK_SIZE=1024 (mirrors half2)
  f16x8_pack_f32   - FP16 inputs, BLOCK_SIZE=1024 (mirrors LDST128BITS)
"""

import time
from typing import Optional

import torch
import triton
import triton.language as tl

DEVICE = torch.device("cuda:0")

# ---------------------------------------------------------------------------
# FP32 kernels
# ---------------------------------------------------------------------------


@triton.jit
def dot_prod_f32_kernel(a_ptr, b_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    partial = tl.sum(a * b, axis=0)
    tl.atomic_add(out_ptr, partial)


# Mirrors float4 CUDA variant: same algorithm, larger conceptual tile width.
@triton.jit
def dot_prod_f32x4_kernel(a_ptr, b_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    partial = tl.sum(a * b, axis=0)
    tl.atomic_add(out_ptr, partial)


# ---------------------------------------------------------------------------
# FP16 kernels
# ---------------------------------------------------------------------------


# Mirrors f16_f32_kernel: load fp16, upcast to fp32 before multiply-accumulate.
@triton.jit
def dot_prod_f16_f32_kernel(a_ptr, b_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    partial = tl.sum(a * b, axis=0)
    tl.atomic_add(out_ptr, partial)


# Mirrors f16x2_f32_kernel: half2-width tile, fp32 accumulator.
@triton.jit
def dot_prod_f16x2_f32_kernel(a_ptr, b_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    partial = tl.sum(a * b, axis=0)
    tl.atomic_add(out_ptr, partial)


# Mirrors f16x8_pack_f32_kernel: LDST128BITS-width tile, fp32 accumulator.
@triton.jit
def dot_prod_f16x8_pack_f32_kernel(a_ptr, b_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    partial = tl.sum(a * b, axis=0)
    tl.atomic_add(out_ptr, partial)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------


def dot_prod_f32_f32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    N = a.numel()
    out = torch.zeros(1, device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(N, 1024),)
    dot_prod_f32_kernel[grid](a, b, out, N, BLOCK_SIZE=1024)
    return out


def dot_prod_f32x4_f32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    N = a.numel()
    out = torch.zeros(1, device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(N, 1024),)
    dot_prod_f32x4_kernel[grid](a, b, out, N, BLOCK_SIZE=1024)
    return out


def dot_prod_f16_f32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    N = a.numel()
    out = torch.zeros(1, device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(N, 1024),)
    dot_prod_f16_f32_kernel[grid](a, b, out, N, BLOCK_SIZE=1024)
    return out


def dot_prod_f16x2_f32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    N = a.numel()
    out = torch.zeros(1, device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(N, 1024),)
    dot_prod_f16x2_f32_kernel[grid](a, b, out, N, BLOCK_SIZE=1024)
    return out


def dot_prod_f16x8_pack_f32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    N = a.numel()
    out = torch.zeros(1, device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(N, 1024),)
    dot_prod_f16x8_pack_f32_kernel[grid](a, b, out, N, BLOCK_SIZE=1024)
    return out


# ---------------------------------------------------------------------------
# Correctness checks
# ---------------------------------------------------------------------------


def check_correctness():
    torch.manual_seed(42)
    N = 1024 * 1024

    a_f32 = torch.randn(N, device=DEVICE, dtype=torch.float32)
    b_f32 = torch.randn(N, device=DEVICE, dtype=torch.float32)
    ref_f32 = torch.dot(a_f32, b_f32)

    out = dot_prod_f32_f32(a_f32, b_f32)
    assert torch.allclose(out, ref_f32.unsqueeze(0), atol=1e-3), "f32_f32 mismatch"

    out = dot_prod_f32x4_f32(a_f32, b_f32)
    assert torch.allclose(out, ref_f32.unsqueeze(0), atol=1e-3), "f32x4_f32 mismatch"

    a_f16 = a_f32.half()
    b_f16 = b_f32.half()
    ref_f16 = torch.dot(a_f16.float(), b_f16.float())

    out = dot_prod_f16_f32(a_f16, b_f16)
    assert torch.allclose(out, ref_f16.unsqueeze(0), atol=1e-3), "f16_f32 mismatch"

    out = dot_prod_f16x2_f32(a_f16, b_f16)
    assert torch.allclose(out, ref_f16.unsqueeze(0), atol=1e-3), "f16x2_f32 mismatch"

    out = dot_prod_f16x8_pack_f32(a_f16, b_f16)
    assert torch.allclose(out, ref_f16.unsqueeze(0), atol=1e-3), "f16x8_pack_f32 mismatch"

    print("All correctness checks passed.")


# ---------------------------------------------------------------------------
# Manual timing benchmark (matches dot_product.py format)
# ---------------------------------------------------------------------------


def run_benchmark(
    perf_func: callable,
    a: torch.Tensor,
    b: torch.Tensor,
    tag: str,
    warmup: int = 10,
    iters: int = 1000,
):
    for _ in range(warmup):
        out = perf_func(a, b)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        out = perf_func(a, b)
    torch.cuda.synchronize()
    end = time.time()
    mean_time = (end - start) * 1000 / iters
    out_info = f"out_{tag}"
    out_val = out.item()
    print(f"{out_info:>25}: {out_val:<15.8f}, time:{mean_time:.8f}ms")
    return out, mean_time


# ---------------------------------------------------------------------------
# triton.testing.perf_report benchmark (GB/s)
# ---------------------------------------------------------------------------


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[2**i for i in range(14, 26)],
        x_log=True,
        line_arg="kernel",
        line_vals=[
            "f32_f32",
            "f32x4_f32",
            "f16_f32",
            "f16x2_f32",
            "f16x8_pack_f32",
            "f32_torch",
            "f16_torch",
        ],
        line_names=[
            "f32_f32",
            "f32x4_f32",
            "f16_f32",
            "f16x2_f32",
            "f16x8_pack_f32",
            "f32 (torch)",
            "f16 (torch)",
        ],
        styles=[
            ("blue", "-"),
            ("blue", "--"),
            ("red", "-"),
            ("red", "--"),
            ("red", ":"),
            ("green", "-"),
            ("green", "--"),
        ],
        ylabel="GB/s",
        plot_name="dot-product-performance",
        args={},
    )
)
def bench_dot_product(N, kernel):
    quantiles = [0.5, 0.2, 0.8]

    if kernel in ("f32_f32", "f32x4_f32", "f32_torch"):
        a = torch.randn(N, device=DEVICE, dtype=torch.float32)
        b = torch.randn(N, device=DEVICE, dtype=torch.float32)
        elem_bytes = 4
    else:
        a = torch.randn(N, device=DEVICE, dtype=torch.float16)
        b = torch.randn(N, device=DEVICE, dtype=torch.float16)
        elem_bytes = 2

    if kernel == "f32_f32":
        fn = lambda: dot_prod_f32_f32(a, b)
    elif kernel == "f32x4_f32":
        fn = lambda: dot_prod_f32x4_f32(a, b)
    elif kernel == "f16_f32":
        fn = lambda: dot_prod_f16_f32(a, b)
    elif kernel == "f16x2_f32":
        fn = lambda: dot_prod_f16x2_f32(a, b)
    elif kernel == "f16x8_pack_f32":
        fn = lambda: dot_prod_f16x8_pack_f32(a, b)
    elif kernel == "f32_torch":
        fn = lambda: torch.dot(a, b)
    elif kernel == "f16_torch":
        fn = lambda: torch.dot(a, b)

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    # 2 reads of N elements each
    gbps = lambda ms: 2 * N * elem_bytes * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    check_correctness()

    Ss = [1024, 2048, 4096]
    Ks = [1024, 2048, 4096]
    SKs = [(S, K) for S in Ss for K in Ks]

    for S, K in SKs:
        print("-" * 80)
        print(" " * 25 + f"S={S}, K={K}")
        a = torch.randn(S * K, device=DEVICE, dtype=torch.float32)
        b = torch.randn(S * K, device=DEVICE, dtype=torch.float32)
        run_benchmark(dot_prod_f32_f32, a, b, "f32f32_triton")
        run_benchmark(dot_prod_f32x4_f32, a, b, "f32x4f32_triton")
        run_benchmark(torch.dot, a, b, "f32f32_torch")

        print("-" * 80)
        a_f16 = a.half()
        b_f16 = b.half()
        run_benchmark(dot_prod_f16_f32, a_f16, b_f16, "f16f32_triton")
        run_benchmark(dot_prod_f16x2_f32, a_f16, b_f16, "f16x2f32_triton")
        run_benchmark(dot_prod_f16x8_pack_f32, a_f16, b_f16, "f16x8packf32_triton")
        run_benchmark(torch.dot, a_f16, b_f16, "f16f16_torch")
        print("-" * 80)

    print("\nRunning triton.testing.perf_report benchmark (GB/s by N)...")
    bench_dot_product.run(print_data=True, show_plots=False, save_path="./")
