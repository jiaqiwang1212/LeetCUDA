"""
Triton implementations of HGEMV kernels from hgemv.cu.

Operation: y[i] = sum_k A[i, k] * x[k]  (matrix-vector product in FP16)
  A has shape (M, K) in float16, x has shape (K,) in float16,
  y has shape (M,) in float16.

Each Triton program handles one output row: it iterates over K-element tiles,
upcasting fp16 inputs to fp32 for accumulation, then downcasts the result back
to fp16 before storing.  This mirrors the CUDA half2-based pattern: fp16
loads → fp32 accumulation → fp16 store.

Variants and their CUDA analogues:
  k32_f16     - BLOCK_K=32,  FP16 in/out  (hgemv_k32_f16_kernel)
  k128_f16x4  - BLOCK_K=128, FP16 in/out  (mirrors half4 load, hgemv_k128_f16x4_kernel)
  k16_f16     - BLOCK_K=16,  FP16 in/out  (hgemv_k16_f16_kernel, thin-K variant)
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
def hgemv_f16_kernel(a_ptr, x_ptr, out_ptr, M, K, BLOCK_K: tl.constexpr):
    """One program per output row; fp16 loads, fp32 accumulator, fp16 store."""
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    acc = tl.zeros([1], dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offsets
        mask = k_offs < K
        a = tl.load(a_ptr + row * K + k_offs, mask=mask, other=0.0).to(tl.float32)
        x_val = tl.load(x_ptr + k_offs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(a * x_val, axis=0)
    tl.store(out_ptr + row, acc.to(tl.float16))


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------


def hgemv_k32_f16(a: torch.Tensor, x: torch.Tensor, out: torch.Tensor):
    M, K = a.shape
    x_flat = x.view(K)
    out_flat = out.view(M)
    hgemv_f16_kernel[(M,)](a, x_flat, out_flat, M, K, BLOCK_K=32)


def hgemv_k128_f16x4(a: torch.Tensor, x: torch.Tensor, out: torch.Tensor):
    M, K = a.shape
    x_flat = x.view(K)
    out_flat = out.view(M)
    hgemv_f16_kernel[(M,)](a, x_flat, out_flat, M, K, BLOCK_K=128)


def hgemv_k16_f16(a: torch.Tensor, x: torch.Tensor, out: torch.Tensor):
    M, K = a.shape
    x_flat = x.view(K)
    out_flat = out.view(M)
    hgemv_f16_kernel[(M,)](a, x_flat, out_flat, M, K, BLOCK_K=16)


# ---------------------------------------------------------------------------
# Correctness checks
# ---------------------------------------------------------------------------


def check_correctness():
    torch.manual_seed(42)

    # K=128 case
    M, N, K = 1024, 1, 128
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16).contiguous()
    x = torch.randn((K, N), device=DEVICE, dtype=torch.float16).contiguous()
    ref = torch.matmul(a, x)

    out = torch.zeros((M, N), device=DEVICE, dtype=torch.float16)
    hgemv_k32_f16(a, x, out)
    assert torch.allclose(out, ref, atol=1e-2), "k32_f16 mismatch (K=128)"

    out.zero_()
    hgemv_k128_f16x4(a, x, out)
    assert torch.allclose(out, ref, atol=1e-2), "k128_f16x4 mismatch (K=128)"

    # K=16 case
    M, N, K = 1024, 1, 16
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16).contiguous()
    x = torch.randn((K, N), device=DEVICE, dtype=torch.float16).contiguous()
    ref = torch.matmul(a, x)

    out = torch.zeros((M, N), device=DEVICE, dtype=torch.float16)
    hgemv_k16_f16(a, x, out)
    assert torch.allclose(out, ref, atol=1e-2), "k16_f16 mismatch (K=16)"

    print("All correctness checks passed.")


# ---------------------------------------------------------------------------
# Manual timing benchmark (matches hgemv.py format)
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
            "k32_f16",
            "k128_f16x4",
            "f16_torch",
        ],
        line_names=[
            "k32_f16",
            "k128_f16x4",
            "f16 (torch)",
        ],
        styles=[
            ("red", "-"),
            ("red", "--"),
            ("green", "-"),
        ],
        ylabel="GFLOP/s",
        plot_name="hgemv-performance-K128",
        args={"K": 128},
    )
)
def bench_hgemv_k128(M, K, kernel):
    quantiles = [0.5, 0.2, 0.8]
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16).contiguous()
    x = torch.randn((K, 1), device=DEVICE, dtype=torch.float16).contiguous()
    out = torch.zeros((M, 1), device=DEVICE, dtype=torch.float16)

    if kernel == "k32_f16":
        fn = lambda: hgemv_k32_f16(a, x, out)
    elif kernel == "k128_f16x4":
        fn = lambda: hgemv_k128_f16x4(a, x, out)
    elif kernel == "f16_torch":
        fn = lambda: torch.matmul(a, x)

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    gflops = lambda ms: 2 * M * K * 1e-9 / (ms * 1e-3)
    return gflops(ms), gflops(max_ms), gflops(min_ms)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M"],
        x_vals=[2**i for i in range(8, 13)],
        x_log=True,
        line_arg="kernel",
        line_vals=[
            "k16_f16",
            "f16_torch",
        ],
        line_names=[
            "k16_f16",
            "f16 (torch)",
        ],
        styles=[
            ("red", "-"),
            ("green", "-"),
        ],
        ylabel="GFLOP/s",
        plot_name="hgemv-performance-K16",
        args={"K": 16},
    )
)
def bench_hgemv_k16(M, K, kernel):
    quantiles = [0.5, 0.2, 0.8]
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16).contiguous()
    x = torch.randn((K, 1), device=DEVICE, dtype=torch.float16).contiguous()
    out = torch.zeros((M, 1), device=DEVICE, dtype=torch.float16)

    if kernel == "k16_f16":
        fn = lambda: hgemv_k16_f16(a, x, out)
    elif kernel == "f16_torch":
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
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16).contiguous()
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16).contiguous()
    c = torch.zeros((M, N), device=DEVICE, dtype=torch.float16).contiguous()
    run_benchmark(hgemv_k32_f16, a, b, "k32f16_triton", c)
    run_benchmark(hgemv_k128_f16x4, a, b, "k128f16x4_triton", c)
    run_benchmark(partial(torch.matmul, out=c), a, b, "f16_torch")
    print("-" * 80)

    M, N, K = 1024, 1, 16
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16).contiguous()
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16).contiguous()
    c = torch.zeros((M, N), device=DEVICE, dtype=torch.float16).contiguous()
    run_benchmark(hgemv_k16_f16, a, b, "k16f16_triton", c)
    run_benchmark(partial(torch.matmul, out=c), a, b, "f16_torch")
    print("-" * 80)

    print("\nRunning triton.testing.perf_report benchmarks (GFLOP/s by M)...")
    bench_hgemv_k128.run(print_data=True, show_plots=False, save_path="./")
    bench_hgemv_k16.run(print_data=True, show_plots=False, save_path="./")
