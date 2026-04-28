"""
Triton implementations of histogram kernels from histogram.cu.

Variants:
  histogram_i32   - int32 input, BLOCK_SIZE=1024, atomic_add per element
  histogram_i32x4 - same kernel (x4 vectorization transparent in Triton)
"""

import time
from typing import Optional

import torch
import triton
import triton.language as tl

DEVICE = torch.device("cuda:0")


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


@triton.jit
def histogram_kernel(
    a_ptr,
    hist_ptr,
    N,
    NUM_BINS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    vals = tl.load(a_ptr + offsets, mask=mask, other=NUM_BINS)
    valid = (vals >= 0) & (vals < NUM_BINS) & mask
    tl.atomic_add(hist_ptr + vals, 1, mask=valid)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------


def histogram_i32(a: torch.Tensor, num_bins: int = 10) -> torch.Tensor:
    N = a.numel()
    hist = torch.zeros(num_bins, dtype=torch.int32, device=a.device)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    histogram_kernel[grid](a, hist, N, NUM_BINS=num_bins, BLOCK_SIZE=BLOCK_SIZE)
    return hist


def histogram_i32x4(a: torch.Tensor, num_bins: int = 10) -> torch.Tensor:
    # Same kernel; x4 vectorization is transparent in Triton
    return histogram_i32(a, num_bins)


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------


def check_correctness():
    torch.manual_seed(42)
    a = torch.tensor(list(range(10)) * 1000, dtype=torch.int32, device=DEVICE)
    ref = torch.bincount(a, minlength=10).to(torch.int32)

    h_i32 = histogram_i32(a)
    assert torch.equal(h_i32, ref), f"histogram_i32 mismatch: {h_i32} vs {ref}"

    h_i32x4 = histogram_i32x4(a)
    assert torch.equal(h_i32x4, ref), f"histogram_i32x4 mismatch: {h_i32x4} vs {ref}"

    print("All correctness checks passed.")


# ---------------------------------------------------------------------------
# Manual timing benchmark
# ---------------------------------------------------------------------------


def run_benchmark(
    perf_func: callable,
    a: torch.Tensor,
    tag: str,
    warmup: int = 10,
    iters: int = 1000,
):
    for _ in range(warmup):
        perf_func(a)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        perf_func(a)
    torch.cuda.synchronize()
    end = time.time()
    mean_time = (end - start) * 1000 / iters
    result = perf_func(a)
    out_val = result.flatten().detach().cpu().numpy().tolist()[:2]
    out_val = [round(v, 8) for v in out_val]
    print(f"{'out_' + tag:>22}: {out_val}, time:{mean_time:.8f}ms")
    return result, mean_time


# ---------------------------------------------------------------------------
# triton.testing.perf_report benchmark (bandwidth in GB/s)
# ---------------------------------------------------------------------------


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[2**i for i in range(14, 26)],
        x_log=True,
        line_arg="kernel",
        line_vals=["i32", "i32x4", "i32_torch"],
        line_names=["i32", "i32x4", "i32 (torch)"],
        styles=[
            ("blue", "-"),
            ("blue", "--"),
            ("green", "-"),
        ],
        ylabel="GB/s",
        plot_name="histogram-performance",
        args={},
    )
)
def bench_histogram(N, kernel):
    quantiles = [0.5, 0.2, 0.8]
    a = torch.randint(0, 10, (N,), dtype=torch.int32, device=DEVICE)

    if kernel == "i32":
        fn = lambda: histogram_i32(a)
    elif kernel == "i32x4":
        fn = lambda: histogram_i32x4(a)
    elif kernel == "i32_torch":
        fn = lambda: torch.bincount(a, minlength=10)

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    # 1 read of N int32 elements (4 bytes each)
    gbps = lambda ms: N * 4 * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    check_correctness()

    # Replicate histogram.py test output exactly
    a = torch.tensor(list(range(10)) * 1000, dtype=torch.int32).cuda()
    h_i32 = histogram_i32(a)
    print("-" * 80)
    for i in range(h_i32.shape[0]):
        print(f"h_i32   {i}: {h_i32[i]}")
    print("-" * 80)
    h_i32x4 = histogram_i32x4(a)
    for i in range(h_i32x4.shape[0]):
        print(f"h_i32x4 {i}: {h_i32x4[i]}")
    print("-" * 80)

    # Manual timing benchmark
    Ns = [2**i for i in range(14, 26)]
    for N in Ns:
        print("-" * 85)
        print(" " * 40 + f"N={N}")
        a_bench = torch.randint(0, 10, (N,), dtype=torch.int32, device=DEVICE)
        run_benchmark(histogram_i32, a_bench, "i32_triton")
        run_benchmark(histogram_i32x4, a_bench, "i32x4_triton")
        run_benchmark(lambda x: torch.bincount(x, minlength=10), a_bench, "i32_torch")
        print("-" * 85)

    print("\nRunning triton.testing.perf_report benchmark (GB/s by N)...")
    bench_histogram.run(print_data=True, show_plots=False, save_path="./")
