"""
Triton implementation of SGEMM (C = A @ B, float32).

Uses @triton.autotune for tile size selection and a swizzled program-ID
mapping for better L2 cache reuse, matching the spirit of the sliced-k /
double-buffered CUDA variants in sgemm.cu.

Aliases exposed:
  sgemm_t_8x8_sliced_k_f32x4
  sgemm_t_8x8_sliced_k_f32x4_bcf
  sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf
"""

import time
from typing import Optional

import torch
import triton
import triton.language as tl

DEVICE = torch.device("cuda:0")

MAX_TFLOPS = -1


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def sgemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Swizzled program ID for better L2 cache reuse
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak
    b_ptrs = b_ptr + offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a_mask = (offsets_m[:, None] < M) & (offsets_k[None, :] < K - k * BLOCK_K)
        b_mask = (offsets_k[:, None] < K - k * BLOCK_K) & (offsets_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn
    c_mask = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float32), mask=c_mask)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------


def sgemm_triton(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Inner dimensions must match: {K} vs {K2}"
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    sgemm_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


# Aliases matching sgemm.py CUDA variant naming style
def sgemm_t_8x8_sliced_k_f32x4(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return sgemm_triton(a, b, c)


def sgemm_t_8x8_sliced_k_f32x4_bcf(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return sgemm_triton(a, b, c)


def sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return sgemm_triton(a, b, c)


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------


def check_correctness():
    torch.manual_seed(42)
    M, N, K = 512, 512, 256
    a = torch.randn((M, K), dtype=torch.float32, device=DEVICE)
    b = torch.randn((K, N), dtype=torch.float32, device=DEVICE)
    c = torch.zeros((M, N), dtype=torch.float32, device=DEVICE)
    ref = torch.mm(a, b)
    sgemm_triton(a, b, c)
    assert torch.allclose(c, ref, atol=1e-3), f"sgemm_triton mismatch, max_diff={( c - ref).abs().max()}"
    print("Correctness check passed.")


# ---------------------------------------------------------------------------
# Manual timing benchmark (matches sgemm.py format)
# ---------------------------------------------------------------------------


def run_benchmark(
    perf_func: callable,
    a: torch.Tensor,
    b: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 2,
    iters: int = 20,
):
    global MAX_TFLOPS

    M = a.size(0)
    K = a.size(1)
    N = b.size(1)

    if out is not None:
        out.fill_(0)
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

    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.flatten()[:2].detach().cpu().numpy().tolist()
    out_val = [round(v, 8) for v in out_val]
    out_val = [f"{v:<12}"[:10] for v in out_val]
    TFLOPS = (2 * M * N * K) * 1e-9 / mean_time
    mean_time_str = str(f"{mean_time:<12}")[:8]

    if TFLOPS > MAX_TFLOPS:
        if MAX_TFLOPS > 0:
            improve = ((TFLOPS - MAX_TFLOPS) / MAX_TFLOPS) * 100
            improve = round(improve, 2)
        else:
            improve = 0
        MAX_TFLOPS = TFLOPS
        print(
            f"{out_info:>35}: {out_val}, time:{mean_time_str}ms, "
            f"TFLOPS: {TFLOPS:<6.2f}(+{improve:.2f}%)"
        )
    else:
        print(
            f"{out_info:>35}: {out_val}, time:{mean_time_str}ms, "
            f"TFLOPS: {TFLOPS:<6.2f}"
        )
    return out, mean_time_str


# ---------------------------------------------------------------------------
# triton.testing.perf_report benchmark (TFLOPS)
# ---------------------------------------------------------------------------


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=[(M, N, K) for M in [2048, 4096] for N in [2048, 4096] for K in [1024, 2048]],
        line_arg="kernel",
        line_vals=["triton", "torch"],
        line_names=["Triton SGEMM", "torch.mm"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="TFLOPS",
        plot_name="sgemm-performance",
        args={},
    )
)
def bench_sgemm(M, N, K, kernel):
    quantiles = [0.5, 0.2, 0.8]
    a = torch.randn((M, K), dtype=torch.float32, device=DEVICE)
    b = torch.randn((K, N), dtype=torch.float32, device=DEVICE)
    c = torch.zeros((M, N), dtype=torch.float32, device=DEVICE)

    if kernel == "triton":
        fn = lambda: sgemm_triton(a, b, c)
    elif kernel == "torch":
        fn = lambda: torch.mm(a, b)

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    tflops = lambda ms: (2 * M * N * K) * 1e-9 / (ms * 1e-3)
    return tflops(ms), tflops(max_ms), tflops(min_ms)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    check_correctness()

    Ms = [2048, 4096, 8192]
    Ns = [2048, 4096, 8192]
    Ks = [1024, 2048, 4096]

    MAX_M, MAX_N, MAX_K = max(Ms), max(Ns), max(Ks)
    A = torch.randn((MAX_M, MAX_K), dtype=torch.float32, device=DEVICE)
    B = torch.randn((MAX_K, MAX_N), dtype=torch.float32, device=DEVICE)
    C = torch.zeros((MAX_M, MAX_N), dtype=torch.float32, device=DEVICE)
    torch.cuda.synchronize()

    MNKs = [(M, N, K) for M in Ms for N in Ns for K in Ks]
    for M, N, K in MNKs:
        MAX_TFLOPS = -1
        print("-" * 130)
        print(" " * 55 + f"M={M}, N={N}, K={K}")
        a = A[:M, :K].contiguous()
        b = B[:K, :N].contiguous()
        c = C[:M, :N].contiguous()
        torch.cuda.synchronize()

        run_benchmark(sgemm_t_8x8_sliced_k_f32x4, a, b, "f32x4(t8x8sk)", c)
        run_benchmark(sgemm_t_8x8_sliced_k_f32x4_bcf, a, b, "f32x4(t8x8bcf)", c)
        run_benchmark(sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf, a, b, "f32x4(t8x8dbuf)", c)
        run_benchmark(lambda a, b: torch.mm(a, b), a, b, "f32_th")
        print("-" * 130)

    print("\nRunning triton.testing.perf_report benchmark (TFLOPS by M,N,K)...")
    bench_sgemm.run(print_data=True, show_plots=False, save_path="./")
