"""
Triton implementations of all RoPE kernels from rope.cu.

Operation: Rotary Position Encoding -- pair adjacent elements and apply rotation.

For row i (position), for each pair index j in [0, N//2):
  freq  = 1.0 / (10000.0 ** (2*j / N))
  angle = i * freq
  out[j]       = x[j]       * cos(angle) - x[j + N//2] * sin(angle)
  out[j+N//2]  = x[j]       * sin(angle) + x[j + N//2] * cos(angle)

Variants and their CUDA analogues:
  rope_f32         - FP32, one program per row (matches rope_f32_kernel)
  rope_f32x4_pack  - FP32, same kernel (vectorization transparent in Triton)
"""

import time
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

DEVICE = torch.device("cuda:0")

# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------


@triton.jit
def rope_f32_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    HALF_N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # one program per row (position)
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)  # [0, 1, ..., BLOCK - 1]
    mask = offsets < HALF_N

    # load both halves of the row
    x_first = tl.load(x_ptr + row * N + offsets, mask=mask)
    x_second = tl.load(x_ptr + row * N + HALF_N + offsets, mask=mask)

    # compute per-dimension frequencies and rotation angle for this position
    j = offsets.to(tl.float32)
    freq = 1.0 / tl.math.pow(10000.0, 2.0 * j / N)
    angle = row.to(tl.float32) * freq
    cos_val = tl.math.cos(angle)
    sin_val = tl.math.sin(angle)

    out_first = x_first * cos_val - x_second * sin_val
    out_second = x_first * sin_val + x_second * cos_val

    tl.store(out_ptr + row * N + offsets, out_first, mask=mask)
    tl.store(out_ptr + row * N + HALF_N + offsets, out_second, mask=mask)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------


def rope_f32(x: torch.Tensor, out: torch.Tensor):
    M, N = x.shape
    HALF_N = N // 2
    BLOCK = triton.next_power_of_2(HALF_N)
    grid = (M,)
    rope_f32_kernel[grid](x, out, M, N, HALF_N=HALF_N, BLOCK=BLOCK)


def rope_f32x4_pack(x: torch.Tensor, out: torch.Tensor):
    # Vectorization is transparent in Triton; same kernel as rope_f32.
    M, N = x.shape
    HALF_N = N // 2
    BLOCK = triton.next_power_of_2(HALF_N)
    grid = (M,)
    rope_f32_kernel[grid](x, out, M, N, HALF_N=HALF_N, BLOCK=BLOCK)


# ---------------------------------------------------------------------------
# Naive reference (from rope.py) used for correctness verification
# ---------------------------------------------------------------------------


def naive_rope(
    x: torch.Tensor,
    theta: float = 10000.0,
) -> torch.Tensor:
    dim = x.shape[-1]
    seq_len = x.shape[-2]
    x_ = x.float().reshape(*x.shape[:-1], -1, 2)
    x_ = torch.view_as_complex(x_)
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
    )
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs).float().cuda()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    xq_out = torch.view_as_real(x_ * freqs_cis).flatten(1)
    return xq_out.type_as(x)


# ---------------------------------------------------------------------------
# Correctness checks
# ---------------------------------------------------------------------------


def check_correctness():
    torch.manual_seed(42)
    M, N = 512, 128

    x = torch.randn((M, N), device=DEVICE, dtype=torch.float32)
    ref = naive_rope(x)

    out = torch.zeros_like(x)
    rope_f32(x, out)
    assert torch.allclose(out, ref, atol=1e-3), "rope_f32 mismatch"

    out.zero_()
    rope_f32x4_pack(x, out)
    assert torch.allclose(out, ref, atol=1e-3), "rope_f32x4_pack mismatch"

    print("All correctness checks passed.")


# ---------------------------------------------------------------------------
# Manual timing benchmark (matches rope.py format)
# ---------------------------------------------------------------------------


def run_benchmark(
    perf_func: callable,
    a: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 2,
    iters: int = 20,
    show_all: bool = False,
):
    if out is not None:
        out.fill_(0)
    if out is not None:
        for _ in range(warmup):
            perf_func(a, out)
    else:
        for _ in range(warmup):
            _ = perf_func(a)

    torch.cuda.synchronize()
    start = time.time()
    if out is not None:
        for _ in range(iters):
            perf_func(a, out)
    else:
        for _ in range(iters):
            out = perf_func(a)
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()[:3]
    out_val = [round(v, 8) for v in out_val]
    out_val = [f"{v:<12}" for v in out_val]
    print(f"{out_info:>20}: {out_val}, time:{mean_time:.6f}ms")
    if show_all:
        print(out)
    return out.clone(), mean_time


# ---------------------------------------------------------------------------
# triton.testing.perf_report benchmark (bandwidth in GB/s)
# ---------------------------------------------------------------------------


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N"],
        x_vals=[(4096, 512), (4096, 1024), (8192, 512), (8192, 1024)],
        x_log=False,
        line_arg="kernel",
        line_vals=["f32", "f32x4_pack"],
        line_names=["f32", "f32x4_pack"],
        styles=[("blue", "-"), ("blue", "--")],
        ylabel="GB/s",
        plot_name="rope-performance",
        args={},
    )
)
def bench_rope(M, N, kernel):
    quantiles = [0.5, 0.2, 0.8]
    x = torch.randn((M, N), device=DEVICE, dtype=torch.float32)
    out = torch.zeros_like(x)

    fn_map = {
        "f32": lambda: rope_f32(x, out),
        "f32x4_pack": lambda: rope_f32x4_pack(x, out),
    }
    fn = fn_map[kernel]

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    # 1 read + 1 write, float32 = 4 bytes
    gbps = lambda ms: 2 * M * N * 4 * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    check_correctness()

    M_vals = [4096, 8192]
    N_vals = [512, 1024]
    MN = [[m, n] for m in M_vals for n in N_vals]

    for M, N in MN:
        print("-" * 100)
        print(" " * 40 + f"M={M}, N={N}")
        print("-" * 100)
        x = torch.randn((M, N), device=DEVICE, dtype=torch.float32).contiguous()
        out = torch.zeros_like(x).contiguous()
        run_benchmark(rope_f32, x, "f32", out)
        run_benchmark(rope_f32x4_pack, x, "f32x4_pack", out)
        run_benchmark(naive_rope, x, "f32_th")
        print("-" * 100)

    bench_rope.run(show_plots=False, print_data=True)
