"""
Triton implementations of all GELU kernels from gelu.cu.

Variants and their CUDA analogues:
  f32         - FP32, BLOCK_SIZE=256   (1 float/thread, matches gelu_f32_kernel)
  f32x4       - FP32, BLOCK_SIZE=1024  (4 floats/thread via float4 in CUDA)
  f16         - FP16, BLOCK_SIZE=256   (1 half/thread, matches gelu_f16_kernel)
  f16x2       - FP16, BLOCK_SIZE=512   (half2, 2 halves/thread)
  f16x8       - FP16, BLOCK_SIZE=2048  (4 separate half2 sub-loads, 8 halves/thread)
  f16x8_pack  - FP16, BLOCK_SIZE=2048  (single 128-bit packed load, 8 halves/thread)

Uses the tanh approximation: GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
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
def gelu_f32_kernel(
    x_ptr,
    y_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    SQRT_2_OVER_PI: tl.constexpr = 0.7978845608028654
    COEF: tl.constexpr = 0.044715
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    inner = SQRT_2_OVER_PI * (x + COEF * x * x * x)
    y = 0.5 * x * (1.0 + tl.math.tanh(inner))
    tl.store(y_ptr + offsets, y, mask=mask)


# Mirrors the float4 CUDA variant: each Triton program processes 4× as many
# elements as the scalar f32 kernel, coalescing 4 consecutive floats per lane.
@triton.jit
def gelu_f32x4_kernel(
    x_ptr,
    y_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    SQRT_2_OVER_PI: tl.constexpr = 0.7978845608028654
    COEF: tl.constexpr = 0.044715
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    inner = SQRT_2_OVER_PI * (x + COEF * x * x * x)
    y = 0.5 * x * (1.0 + tl.math.tanh(inner))
    tl.store(y_ptr + offsets, y, mask=mask)


# ---------------------------------------------------------------------------
# FP16 kernels
# ---------------------------------------------------------------------------


@triton.jit
def gelu_f16_kernel(
    x_ptr,
    y_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    xf = x.to(tl.float32)
    inner = 0.7978845608028654 * (xf + 0.044715 * xf * xf * xf)
    y = (0.5 * xf * (1.0 + tl.math.tanh(inner))).to(tl.float16)
    tl.store(y_ptr + offsets, y, mask=mask)


# Mirrors the half2 CUDA variant: each program processes 2 fp16 values.
@triton.jit
def gelu_f16x2_kernel(
    x_ptr,
    y_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    xf = x.to(tl.float32)
    inner = 0.7978845608028654 * (xf + 0.044715 * xf * xf * xf)
    y = (0.5 * xf * (1.0 + tl.math.tanh(inner))).to(tl.float16)
    tl.store(y_ptr + offsets, y, mask=mask)


# Mirrors gelu_f16x8_kernel: 4 separate half2 sub-loads per program, each
# covering BLOCK_SIZE//4 elements.  The static_range loop is unrolled at
# compile time, generating 4 independent load/gelu/store sequences just like
# the 4 explicit half2 loads in the CUDA source.
@triton.jit
def gelu_f16x8_kernel(
    x_ptr,
    y_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    CHUNK: tl.constexpr = BLOCK_SIZE // 4
    pid = tl.program_id(0)
    base = pid * BLOCK_SIZE
    for i in tl.static_range(4):
        offsets = base + i * CHUNK + tl.arange(0, CHUNK)
        mask = offsets < N
        x = tl.load(x_ptr + offsets, mask=mask)
        xf = x.to(tl.float32)
        inner = 0.7978845608028654 * (xf + 0.044715 * xf * xf * xf)
        y = (0.5 * xf * (1.0 + tl.math.tanh(inner))).to(tl.float16)
        tl.store(y_ptr + offsets, y, mask=mask)


# Mirrors gelu_f16x8_pack_kernel: a single contiguous 128-bit load
# (LDST128BITS in CUDA) for the full 8-element tile per program.
@triton.jit
def gelu_f16x8_pack_kernel(
    x_ptr,
    y_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    xf = x.to(tl.float32)
    inner = 0.7978845608028654 * (xf + 0.044715 * xf * xf * xf)
    y = (0.5 * xf * (1.0 + tl.math.tanh(inner))).to(tl.float16)
    tl.store(y_ptr + offsets, y, mask=mask)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------


def gelu_f32(x: torch.Tensor, y: torch.Tensor):
    N = x.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    gelu_f32_kernel[grid](x, y, N, BLOCK_SIZE=BLOCK_SIZE)


def gelu_f32x4(x: torch.Tensor, y: torch.Tensor):
    N = x.numel()
    BLOCK_SIZE = 1024  # 4× the scalar variant
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    gelu_f32x4_kernel[grid](x, y, N, BLOCK_SIZE=BLOCK_SIZE)


def gelu_f16(x: torch.Tensor, y: torch.Tensor):
    N = x.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    gelu_f16_kernel[grid](x, y, N, BLOCK_SIZE=BLOCK_SIZE)


def gelu_f16x2(x: torch.Tensor, y: torch.Tensor):
    N = x.numel()
    BLOCK_SIZE = 512  # 2× the scalar variant
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    gelu_f16x2_kernel[grid](x, y, N, BLOCK_SIZE=BLOCK_SIZE)


def gelu_f16x8(x: torch.Tensor, y: torch.Tensor):
    N = x.numel()
    BLOCK_SIZE = 2048  # 4 sub-loads of 512 elements each
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    gelu_f16x8_kernel[grid](x, y, N, BLOCK_SIZE=BLOCK_SIZE)


def gelu_f16x8_pack(x: torch.Tensor, y: torch.Tensor):
    N = x.numel()
    BLOCK_SIZE = 2048  # single 128-bit packed load per lane
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    gelu_f16x8_pack_kernel[grid](x, y, N, BLOCK_SIZE=BLOCK_SIZE)


# ---------------------------------------------------------------------------
# Correctness checks
# ---------------------------------------------------------------------------


def check_correctness():
    torch.manual_seed(42)
    S, K = 1024, 1024
    N = S * K

    gelu_torch = torch.nn.GELU("tanh")

    x_f32 = torch.randn(N, device=DEVICE, dtype=torch.float32)
    y_f32 = torch.zeros(N, device=DEVICE, dtype=torch.float32)
    ref_f32 = gelu_torch(x_f32)

    gelu_f32(x_f32, y_f32)
    assert torch.allclose(y_f32, ref_f32, atol=1e-5), "f32 mismatch"

    y_f32.zero_()
    gelu_f32x4(x_f32, y_f32)
    assert torch.allclose(y_f32, ref_f32, atol=1e-5), "f32x4 mismatch"

    x_f16 = x_f32.half()
    y_f16 = torch.zeros(N, device=DEVICE, dtype=torch.float16)
    ref_f16 = gelu_torch(x_f16)

    gelu_f16(x_f16, y_f16)
    assert torch.allclose(y_f16, ref_f16, atol=1e-2), "f16 mismatch"

    y_f16.zero_()
    gelu_f16x2(x_f16, y_f16)
    assert torch.allclose(y_f16, ref_f16, atol=1e-2), "f16x2 mismatch"

    y_f16.zero_()
    gelu_f16x8(x_f16, y_f16)
    assert torch.allclose(y_f16, ref_f16, atol=1e-2), "f16x8 mismatch"

    y_f16.zero_()
    gelu_f16x8_pack(x_f16, y_f16)
    assert torch.allclose(y_f16, ref_f16, atol=1e-2), "f16x8_pack mismatch"

    print("All correctness checks passed.")


# ---------------------------------------------------------------------------
# Manual timing benchmark (matches relu.py format)
# ---------------------------------------------------------------------------


def run_benchmark(
    perf_func: callable,
    x: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 1000,
):
    if out is not None:
        out.fill_(0)
    if out is not None:
        for _ in range(warmup):
            perf_func(x, out)
    else:
        for _ in range(warmup):
            perf_func(x)
    torch.cuda.synchronize()
    start = time.time()
    if out is not None:
        for _ in range(iters):
            perf_func(x, out)
    else:
        for _ in range(iters):
            perf_func(x)
    torch.cuda.synchronize()
    end = time.time()
    mean_time = (end - start) * 1000 / iters
    result = out if out is not None else perf_func(x)
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
        line_vals=[
            "f32",
            "f32x4",
            "f32_torch",
            "f16",
            "f16x2",
            "f16x8",
            "f16x8_pack",
            "f16_torch",
        ],
        line_names=[
            "f32",
            "f32x4",
            "f32 (torch)",
            "f16",
            "f16x2",
            "f16x8",
            "f16x8_pack",
            "f16 (torch)",
        ],
        styles=[
            ("blue", "-"),
            ("blue", "--"),
            ("blue", ":"),
            ("red", "-"),
            ("red", "--"),
            ("red", "-."),
            ("red", ":"),
            ("green", "-"),
        ],
        ylabel="GB/s",
        plot_name="gelu-performance",
        args={},
    )
)
def bench_gelu(N, kernel):
    quantiles = [0.5, 0.2, 0.8]
    gelu_torch = torch.nn.GELU("tanh")

    if kernel in ("f32", "f32x4", "f32_torch"):
        x = torch.randn(N, device=DEVICE, dtype=torch.float32)
        y = torch.empty(N, device=DEVICE, dtype=torch.float32)
        elem_bytes = 4
    else:
        x = torch.randn(N, device=DEVICE, dtype=torch.float16)
        y = torch.empty(N, device=DEVICE, dtype=torch.float16)
        elem_bytes = 2

    if kernel == "f32":
        fn = lambda: gelu_f32(x, y)
    elif kernel == "f32x4":
        fn = lambda: gelu_f32x4(x, y)
    elif kernel == "f32_torch":
        fn = lambda: gelu_torch(x)
    elif kernel == "f16":
        fn = lambda: gelu_f16(x, y)
    elif kernel == "f16x2":
        fn = lambda: gelu_f16x2(x, y)
    elif kernel == "f16x8":
        fn = lambda: gelu_f16x8(x, y)
    elif kernel == "f16x8_pack":
        fn = lambda: gelu_f16x8_pack(x, y)
    elif kernel == "f16_torch":
        fn = lambda: gelu_torch(x)

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    # 1 read + 1 write = 2 tensors of N elements
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

    gelu_torch = torch.nn.GELU("tanh")

    for S, K in SKs:
        print("-" * 85)
        print(" " * 40 + f"S={S}, K={K}")

        x = torch.randn((S, K), device=DEVICE, dtype=torch.float32).contiguous()
        y = torch.zeros((S, K), device=DEVICE, dtype=torch.float32).contiguous()

        run_benchmark(gelu_f32, x, "f32_triton", y)
        run_benchmark(gelu_f32x4, x, "f32x4_triton", y)
        run_benchmark(gelu_torch, x, "f32_torch")

        print("-" * 85)

        x_f16 = x.half().contiguous()
        y_f16 = y.half().contiguous()

        run_benchmark(gelu_f16, x_f16, "f16_triton", y_f16)
        run_benchmark(gelu_f16x2, x_f16, "f16x2_triton", y_f16)
        run_benchmark(gelu_f16x8, x_f16, "f16x8_triton", y_f16)
        run_benchmark(gelu_f16x8_pack, x_f16, "f16x8pack_triton", y_f16)
        run_benchmark(gelu_torch, x_f16, "f16_torch")
        print("-" * 85)

    print("\nRunning triton.testing.perf_report benchmark (GB/s by N)...")
    bench_gelu.run(print_data=True, show_plots=False, save_path="./")
