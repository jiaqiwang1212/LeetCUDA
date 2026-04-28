"""
Triton implementations of all ReLU kernels from relu.cu.

Variants and their CUDA analogues:
  f32         - FP32, BLOCK_SIZE=256   (1 float/thread, matches relu_f32_kernel)
  f32x4       - FP32, BLOCK_SIZE=1024  (4 floats/thread via float4 in CUDA)
  f16         - FP16, BLOCK_SIZE=256   (1 half/thread, matches relu_f16_kernel)
  f16x2       - FP16, BLOCK_SIZE=512   (half2, 2 halves/thread)
  f16x8       - FP16, BLOCK_SIZE=2048  (4 separate half2 sub-loads, 8 halves/thread)
  f16x8_pack  - FP16, BLOCK_SIZE=2048  (single 128-bit packed load, 8 halves/thread)
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
def relu_f32_kernel(
    x_ptr,
    y_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, tl.maximum(x, 0.0), mask=mask)


# Mirrors the float4 CUDA variant: each Triton program processes 4× as many
# elements as the scalar f32 kernel, coalescing 4 consecutive floats per lane.
@triton.jit
def relu_f32x4_kernel(
    x_ptr,
    y_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, tl.maximum(x, 0.0), mask=mask)


# ---------------------------------------------------------------------------
# FP16 kernels
# ---------------------------------------------------------------------------


@triton.jit
def relu_f16_kernel(
    x_ptr,
    y_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, tl.maximum(x, 0.0), mask=mask)


# Mirrors the half2 CUDA variant: each program processes 2 fp16 values.
@triton.jit
def relu_f16x2_kernel(
    x_ptr,
    y_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, tl.maximum(x, 0.0), mask=mask)


# Mirrors relu_f16x8_kernel: 4 separate half2 sub-loads per program, each
# covering BLOCK_SIZE//4 elements.  The static_range loop is unrolled at
# compile time, generating 4 independent load/relu/store sequences just like
# the 4 explicit half2 loads in the CUDA source.
@triton.jit
def relu_f16x8_kernel(
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
        tl.store(y_ptr + offsets, tl.maximum(x, 0.0), mask=mask)


# Mirrors relu_f16x8_pack_kernel: a single contiguous 128-bit load
# (LDST128BITS in CUDA) for the full 8-element tile per program.
@triton.jit
def relu_f16x8_pack_kernel(
    x_ptr,
    y_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, tl.maximum(x, 0.0), mask=mask)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------


def relu_f32(x: torch.Tensor, y: torch.Tensor):
    N = x.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    relu_f32_kernel[grid](x, y, N, BLOCK_SIZE=BLOCK_SIZE)


def relu_f32x4(x: torch.Tensor, y: torch.Tensor):
    N = x.numel()
    BLOCK_SIZE = 1024  # 4× the scalar variant
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    relu_f32x4_kernel[grid](x, y, N, BLOCK_SIZE=BLOCK_SIZE)


def relu_f16(x: torch.Tensor, y: torch.Tensor):
    N = x.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    relu_f16_kernel[grid](x, y, N, BLOCK_SIZE=BLOCK_SIZE)


def relu_f16x2(x: torch.Tensor, y: torch.Tensor):
    N = x.numel()
    BLOCK_SIZE = 512  # 2× the scalar variant
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    relu_f16x2_kernel[grid](x, y, N, BLOCK_SIZE=BLOCK_SIZE)


def relu_f16x8(x: torch.Tensor, y: torch.Tensor):
    N = x.numel()
    BLOCK_SIZE = 2048  # 4 sub-loads of 512 elements each
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    relu_f16x8_kernel[grid](x, y, N, BLOCK_SIZE=BLOCK_SIZE)


def relu_f16x8_pack(x: torch.Tensor, y: torch.Tensor):
    N = x.numel()
    BLOCK_SIZE = 2048  # single 128-bit packed load per lane
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    relu_f16x8_pack_kernel[grid](x, y, N, BLOCK_SIZE=BLOCK_SIZE)


# ---------------------------------------------------------------------------
# Correctness checks
# ---------------------------------------------------------------------------


def check_correctness():
    torch.manual_seed(42)
    S, K = 1024, 1024
    N = S * K

    x_f32 = torch.randn(N, device=DEVICE, dtype=torch.float32)
    y_f32 = torch.zeros(N, device=DEVICE, dtype=torch.float32)
    ref_f32 = torch.relu(x_f32)

    relu_f32(x_f32, y_f32)
    assert torch.allclose(y_f32, ref_f32), "f32 mismatch"

    y_f32.zero_()
    relu_f32x4(x_f32, y_f32)
    assert torch.allclose(y_f32, ref_f32), "f32x4 mismatch"

    x_f16 = x_f32.half()
    y_f16 = torch.zeros(N, device=DEVICE, dtype=torch.float16)
    ref_f16 = torch.relu(x_f16)

    relu_f16(x_f16, y_f16)
    assert torch.allclose(y_f16, ref_f16), "f16 mismatch"

    y_f16.zero_()
    relu_f16x2(x_f16, y_f16)
    assert torch.allclose(y_f16, ref_f16), "f16x2 mismatch"

    y_f16.zero_()
    relu_f16x8(x_f16, y_f16)
    assert torch.allclose(y_f16, ref_f16), "f16x8 mismatch"

    y_f16.zero_()
    relu_f16x8_pack(x_f16, y_f16)
    assert torch.allclose(y_f16, ref_f16), "f16x8_pack mismatch"

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
        plot_name="relu-performance",
        args={},
    )
)
def bench_relu(N, kernel):
    quantiles = [0.5, 0.2, 0.8]

    if kernel in ("f32", "f32x4", "f32_torch"):
        x = torch.randn(N, device=DEVICE, dtype=torch.float32)
        y = torch.empty(N, device=DEVICE, dtype=torch.float32)
        elem_bytes = 4
    else:
        x = torch.randn(N, device=DEVICE, dtype=torch.float16)
        y = torch.empty(N, device=DEVICE, dtype=torch.float16)
        elem_bytes = 2

    if kernel == "f32":
        fn = lambda: relu_f32(x, y)
    elif kernel == "f32x4":
        fn = lambda: relu_f32x4(x, y)
    elif kernel == "f32_torch":
        fn = lambda: torch.relu(x)
    elif kernel == "f16":
        fn = lambda: relu_f16(x, y)
    elif kernel == "f16x2":
        fn = lambda: relu_f16x2(x, y)
    elif kernel == "f16x8":
        fn = lambda: relu_f16x8(x, y)
    elif kernel == "f16x8_pack":
        fn = lambda: relu_f16x8_pack(x, y)
    elif kernel == "f16_torch":
        fn = lambda: torch.relu(x)

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

    for S, K in SKs:
        print("-" * 85)
        print(" " * 40 + f"S={S}, K={K}")

        x = torch.randn((S, K), device=DEVICE, dtype=torch.float32).contiguous()
        y = torch.zeros((S, K), device=DEVICE, dtype=torch.float32).contiguous()

        run_benchmark(relu_f32, x, "f32_triton", y)
        run_benchmark(relu_f32x4, x, "f32x4_triton", y)
        run_benchmark(torch.relu, x, "f32_torch")

        print("-" * 85)

        x_f16 = x.half().contiguous()
        y_f16 = y.half().contiguous()

        run_benchmark(relu_f16, x_f16, "f16_triton", y_f16)
        run_benchmark(relu_f16x2, x_f16, "f16x2_triton", y_f16)
        run_benchmark(relu_f16x8, x_f16, "f16x8_triton", y_f16)
        run_benchmark(relu_f16x8_pack, x_f16, "f16x8pack_triton", y_f16)
        run_benchmark(torch.relu, x_f16, "f16_torch")
        print("-" * 85)

    print("\nRunning triton.testing.perf_report benchmark (GB/s by N)...")
    bench_relu.run(print_data=True, show_plots=False, save_path="./")
