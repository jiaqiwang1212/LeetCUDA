"""
Triton implementations of SGEMV kernels from sgemv.cu.

Operation: y[i] = sum_k A[i, k] * x[k]  (matrix-vector product)
  A has shape (M, K), x has shape (K,), y has shape (M,).

Each Triton program is responsible for one output row: it iterates over
K-element tiles of A's row and x, accumulating a float32 dot product, then
stores the scalar result.  This mirrors the CUDA pattern: one thread block
per output row, warp-level reduction within the row.

Variants and their CUDA analogues:
  k32_f32     - BLOCK_K=32,  FP32  (sgemv_k32_f32_kernel)
  k128_f32x4  - BLOCK_K=128, FP32  (mirrors float4 load, sgemv_k128_f32x4_kernel)
  k16_f32     - BLOCK_K=16,  FP32  (sgemv_k16_f32_kernel, thin-K variant)
"""

import time
from functools import partial
from typing import Optional

import torch
import triton
import triton.language as tl

DEVICE = torch.device("cuda:0")

# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


@triton.jit
def sgemv_f32_kernel(a_ptr, x_ptr, out_ptr, M, K, BLOCK_K: tl.constexpr):
    """One program per output row; loops over K in BLOCK_K-sized tiles."""
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    acc = tl.zeros([1], dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offsets
        mask = k_offs < K
        a = tl.load(a_ptr + row * K + k_offs, mask=mask, other=0.0)
        x_val = tl.load(x_ptr + k_offs, mask=mask, other=0.0)
        acc += tl.sum(a * x_val, axis=0)
    tl.store(out_ptr + row, acc)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------


def sgemv_k32_f32(a: torch.Tensor, x: torch.Tensor, out: torch.Tensor):
    M, K = a.shape
    x_flat = x.view(K)
    out_flat = out.view(M)
    sgemv_f32_kernel[(M,)](a, x_flat, out_flat, M, K, BLOCK_K=32)


def sgemv_k128_f32x4(a: torch.Tensor, x: torch.Tensor, out: torch.Tensor):
    M, K = a.shape
    x_flat = x.view(K)
    out_flat = out.view(M)
    sgemv_f32_kernel[(M,)](a, x_flat, out_flat, M, K, BLOCK_K=128)


def sgemv_k16_f32(a: torch.Tensor, x: torch.Tensor, out: torch.Tensor):
    M, K = a.shape
    x_flat = x.view(K)
    out_flat = out.view(M)
    sgemv_f32_kernel[(M,)](a, x_flat, out_flat, M, K, BLOCK_K=16)


# ---------------------------------------------------------------------------
# Correctness checks
# ---------------------------------------------------------------------------


def check_correctness():
    torch.manual_seed(42)

    # K=128 case
    M, N, K = 1024, 1, 128
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float32).contiguous()
    x = torch.randn((K, N), device=DEVICE, dtype=torch.float32).contiguous()
    ref = torch.matmul(a, x)

    out = torch.zeros((M, N), device=DEVICE, dtype=torch.float32)
    sgemv_k32_f32(a, x, out)
    assert torch.allclose(out, ref, atol=1e-3), "k32_f32 mismatch (K=128)"

    out.zero_()
    sgemv_k128_f32x4(a, x, out)
    assert torch.allclose(out, ref, atol=1e-3), "k128_f32x4 mismatch (K=128)"

    # K=16 case
    M, N, K = 1024, 1, 16
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float32).contiguous()
    x = torch.randn((K, N), device=DEVICE, dtype=torch.float32).contiguous()
    ref = torch.matmul(a, x)

    out = torch.zeros((M, N), device=DEVICE, dtype=torch.float32)
    sgemv_k16_f32(a, x, out)
    assert torch.allclose(out, ref, atol=1e-3), "k16_f32 mismatch (K=16)"

    print("All correctness checks passed.")


# ---------------------------------------------------------------------------
# Manual timing benchmark (matches sgemv.py format)
# ---------------------------------------------------------------------------


def run_benchmark(
    perf_func: callable,
    a: torch.Tensor,
    b: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 200,
    show_all: bool = False,
):
    if out is not None:
        out.fill_(0)
    if out is not None:
        for _ in range(warmup):
            perf_func(a, b, out)
    else:
        for _ in range(warmup):
            _ = perf_func(a, b)
    torch.cuda.synchronize()
    start = time.time()
    if out is not None:
        for _ in range(iters):
            perf_func(a, b, out)
    else:
        for _ in range(iters):
            out = perf_func(a, b)
    torch.cuda.synchronize()
    end = time.time()
    mean_time = (end - start) * 1000 / iters
    out_info = f"out_{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()[:3]
    out_val = [round(v, 8) for v in out_val]
    print(f"{out_info:>20}: {out_val}, time:{mean_time:.8f}ms")
    if show_all:
        print(out)
    return out.clone(), mean_time


# ---------------------------------------------------------------------------
# triton.testing.perf_report benchmark (FLOP/s)
# ---------------------------------------------------------------------------


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M"],
        x_vals=[2**i for i in range(8, 13)],
        x_log=True,
        line_arg="kernel",
        line_vals=[
            "k32_f32",
            "k128_f32x4",
            "f32_torch",
        ],
        line_names=[
            "k32_f32",
            "k128_f32x4",
            "f32 (torch)",
        ],
        styles=[
            ("blue", "-"),
            ("blue", "--"),
            ("green", "-"),
        ],
        ylabel="GFLOP/s",
        plot_name="sgemv-performance-K128",
        args={"K": 128},
    )
)
def bench_sgemv_k128(M, K, kernel):
    quantiles = [0.5, 0.2, 0.8]
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float32).contiguous()
    x = torch.randn((K, 1), device=DEVICE, dtype=torch.float32).contiguous()
    out = torch.zeros((M, 1), device=DEVICE, dtype=torch.float32)

    if kernel == "k32_f32":
        fn = lambda: sgemv_k32_f32(a, x, out)
    elif kernel == "k128_f32x4":
        fn = lambda: sgemv_k128_f32x4(a, x, out)
    elif kernel == "f32_torch":
        fn = lambda: torch.matmul(a, x)

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    # 2*M*K FLOPs (M dot products of length K, each with K muls + K-1 adds)
    gflops = lambda ms: 2 * M * K * 1e-9 / (ms * 1e-3)
    return gflops(ms), gflops(max_ms), gflops(min_ms)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M"],
        x_vals=[2**i for i in range(8, 13)],
        x_log=True,
        line_arg="kernel",
        line_vals=[
            "k16_f32",
            "f32_torch",
        ],
        line_names=[
            "k16_f32",
            "f32 (torch)",
        ],
        styles=[
            ("blue", "-"),
            ("green", "-"),
        ],
        ylabel="GFLOP/s",
        plot_name="sgemv-performance-K16",
        args={"K": 16},
    )
)
def bench_sgemv_k16(M, K, kernel):
    quantiles = [0.5, 0.2, 0.8]
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float32).contiguous()
    x = torch.randn((K, 1), device=DEVICE, dtype=torch.float32).contiguous()
    out = torch.zeros((M, 1), device=DEVICE, dtype=torch.float32)

    if kernel == "k16_f32":
        fn = lambda: sgemv_k16_f32(a, x, out)
    elif kernel == "f32_torch":
        fn = lambda: torch.matmul(a, x)

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    gflops = lambda ms: 2 * M * K * 1e-9 / (ms * 1e-3)
    return gflops(ms), gflops(max_ms), gflops(min_ms)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    check_correctness()

    print("-" * 80)
    M, N, K = 1024, 1, 128
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float32).contiguous()
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float32).contiguous()
    c = torch.zeros((M, N), device=DEVICE, dtype=torch.float32).contiguous()
    run_benchmark(sgemv_k32_f32, a, b, "k32f32_triton", c)
    run_benchmark(sgemv_k128_f32x4, a, b, "k128f32x4_triton", c)
    run_benchmark(partial(torch.matmul, out=c), a, b, "f32_torch")
    print("-" * 80)

    M, N, K = 1024, 1, 16
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float32).contiguous()
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float32).contiguous()
    c = torch.zeros((M, N), device=DEVICE, dtype=torch.float32).contiguous()
    run_benchmark(sgemv_k16_f32, a, b, "k16f32_triton", c)
    run_benchmark(partial(torch.matmul, out=c), a, b, "f32_torch")
    print("-" * 80)

    print("\nRunning triton.testing.perf_report benchmarks (GFLOP/s by M)...")
    bench_sgemv_k128.run(print_data=True, show_plots=False, save_path="./")
    bench_sgemv_k16.run(print_data=True, show_plots=False, save_path="./")
