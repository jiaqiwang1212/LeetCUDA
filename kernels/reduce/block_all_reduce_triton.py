"""
Triton implementations of block all-reduce sum kernels from block_all_reduce.cu.

Each Triton program loads a BLOCK_SIZE-element tile, reduces it locally with
tl.sum, then contributes to the global scalar result via tl.atomic_add.
This mirrors the CUDA pattern: warp-level shfl reduction → shared-memory
inter-warp reduction → atomicAdd to global output.

Variants and their CUDA analogues:
  f32              - FP32, BLOCK_SIZE=256  (1 float/thread)
  f32x4            - FP32, BLOCK_SIZE=1024 (mirrors float4 CUDA load)
  f16_f16          - FP16 input, FP16 block accumulator → FP32 output
  f16_f32          - FP16 input, FP32 block accumulator → FP32 output
  f16x2_f16        - FP16 input, BLOCK_SIZE=512, FP16 acc  (mirrors half2)
  f16x2_f32        - FP16 input, BLOCK_SIZE=512, FP32 acc
  f16x8_pack_f16   - FP16 input, BLOCK_SIZE=2048 (mirrors LDST128BITS), FP16 acc
  f16x8_pack_f32   - FP16 input, BLOCK_SIZE=2048, FP32 acc
  bf16_bf16        - BF16 input, BF16 block accumulator → FP32 output
  bf16_f32         - BF16 input, FP32 block accumulator → FP32 output
  bf16x2_bf16      - BF16 input, BLOCK_SIZE=512, BF16 acc
  bf16x2_f32       - BF16 input, BLOCK_SIZE=512, FP32 acc
  bf16x8_pack_bf16 - BF16 input, BLOCK_SIZE=2048, BF16 acc
  bf16x8_pack_f32  - BF16 input, BLOCK_SIZE=2048, FP32 acc
"""

import time

import torch
import triton
import triton.language as tl

DEVICE = torch.device("cuda:0")

# ---------------------------------------------------------------------------
# FP32 kernels
# ---------------------------------------------------------------------------


@triton.jit
def block_all_reduce_sum_f32_kernel(a_ptr, y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    block_sum = tl.sum(a, axis=0)
    tl.atomic_add(y_ptr, block_sum)


# Mirrors float4 CUDA variant: 4x larger tile covers same work with fewer programs.
@triton.jit
def block_all_reduce_sum_f32x4_kernel(a_ptr, y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    block_sum = tl.sum(a, axis=0)
    tl.atomic_add(y_ptr, block_sum)


# ---------------------------------------------------------------------------
# FP16 kernels
# ---------------------------------------------------------------------------

# Mirrors f16_f16_kernel: accumulate in fp16 (lower precision, faster on older hw).
@triton.jit
def block_all_reduce_sum_f16_f16_kernel(a_ptr, y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float16)
    block_sum = tl.sum(a, axis=0)
    tl.atomic_add(y_ptr, block_sum.to(tl.float32))


# Mirrors f16_f32_kernel: convert to fp32 before accumulation for higher precision.
@triton.jit
def block_all_reduce_sum_f16_f32_kernel(a_ptr, y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    block_sum = tl.sum(a, axis=0)
    tl.atomic_add(y_ptr, block_sum)


# Mirrors f16x2_f16_kernel: 2x larger tile mirrors half2 (2 halves/thread).
@triton.jit
def block_all_reduce_sum_f16x2_f16_kernel(a_ptr, y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float16)
    block_sum = tl.sum(a, axis=0)
    tl.atomic_add(y_ptr, block_sum.to(tl.float32))


@triton.jit
def block_all_reduce_sum_f16x2_f32_kernel(a_ptr, y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    block_sum = tl.sum(a, axis=0)
    tl.atomic_add(y_ptr, block_sum)


# Mirrors f16x8_pack kernels: 8x tile corresponds to LDST128BITS (128-bit load).
@triton.jit
def block_all_reduce_sum_f16x8_pack_f16_kernel(
    a_ptr, y_ptr, N, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float16)
    block_sum = tl.sum(a, axis=0)
    tl.atomic_add(y_ptr, block_sum.to(tl.float32))


@triton.jit
def block_all_reduce_sum_f16x8_pack_f32_kernel(
    a_ptr, y_ptr, N, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    block_sum = tl.sum(a, axis=0)
    tl.atomic_add(y_ptr, block_sum)


# ---------------------------------------------------------------------------
# BF16 kernels
# ---------------------------------------------------------------------------


@triton.jit
def block_all_reduce_sum_bf16_bf16_kernel(a_ptr, y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.bfloat16)
    block_sum = tl.sum(a, axis=0)
    tl.atomic_add(y_ptr, block_sum.to(tl.float32))


@triton.jit
def block_all_reduce_sum_bf16_f32_kernel(a_ptr, y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    block_sum = tl.sum(a, axis=0)
    tl.atomic_add(y_ptr, block_sum)


@triton.jit
def block_all_reduce_sum_bf16x2_bf16_kernel(
    a_ptr, y_ptr, N, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.bfloat16)
    block_sum = tl.sum(a, axis=0)
    tl.atomic_add(y_ptr, block_sum.to(tl.float32))


@triton.jit
def block_all_reduce_sum_bf16x2_f32_kernel(a_ptr, y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    block_sum = tl.sum(a, axis=0)
    tl.atomic_add(y_ptr, block_sum)


@triton.jit
def block_all_reduce_sum_bf16x8_pack_bf16_kernel(
    a_ptr, y_ptr, N, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.bfloat16)
    block_sum = tl.sum(a, axis=0)
    tl.atomic_add(y_ptr, block_sum.to(tl.float32))


@triton.jit
def block_all_reduce_sum_bf16x8_pack_f32_kernel(
    a_ptr, y_ptr, N, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    block_sum = tl.sum(a, axis=0)
    tl.atomic_add(y_ptr, block_sum)


# ---------------------------------------------------------------------------
# Python wrappers — each allocates a fresh zeros({1}) output and returns it,
# matching the TORCH_BINDING_REDUCE macro pattern in block_all_reduce.cu.
# ---------------------------------------------------------------------------


def _zeros_out() -> torch.Tensor:
    return torch.zeros(1, device=DEVICE, dtype=torch.float32)


def block_all_reduce_sum_f32(a: torch.Tensor) -> torch.Tensor:
    N = a.numel()
    BLOCK_SIZE = 256
    y = _zeros_out()
    block_all_reduce_sum_f32_kernel[(triton.cdiv(N, BLOCK_SIZE),)](
        a, y, N, BLOCK_SIZE=BLOCK_SIZE
    )
    return y


def block_all_reduce_sum_f32x4(a: torch.Tensor) -> torch.Tensor:
    N = a.numel()
    BLOCK_SIZE = 1024
    y = _zeros_out()
    block_all_reduce_sum_f32x4_kernel[(triton.cdiv(N, BLOCK_SIZE),)](
        a, y, N, BLOCK_SIZE=BLOCK_SIZE
    )
    return y


def block_all_reduce_sum_f16_f16(a: torch.Tensor) -> torch.Tensor:
    N = a.numel()
    BLOCK_SIZE = 256
    y = _zeros_out()
    block_all_reduce_sum_f16_f16_kernel[(triton.cdiv(N, BLOCK_SIZE),)](
        a, y, N, BLOCK_SIZE=BLOCK_SIZE
    )
    return y


def block_all_reduce_sum_f16_f32(a: torch.Tensor) -> torch.Tensor:
    N = a.numel()
    BLOCK_SIZE = 256
    y = _zeros_out()
    block_all_reduce_sum_f16_f32_kernel[(triton.cdiv(N, BLOCK_SIZE),)](
        a, y, N, BLOCK_SIZE=BLOCK_SIZE
    )
    return y


def block_all_reduce_sum_f16x2_f16(a: torch.Tensor) -> torch.Tensor:
    N = a.numel()
    BLOCK_SIZE = 512
    y = _zeros_out()
    block_all_reduce_sum_f16x2_f16_kernel[(triton.cdiv(N, BLOCK_SIZE),)](
        a, y, N, BLOCK_SIZE=BLOCK_SIZE
    )
    return y


def block_all_reduce_sum_f16x2_f32(a: torch.Tensor) -> torch.Tensor:
    N = a.numel()
    BLOCK_SIZE = 512
    y = _zeros_out()
    block_all_reduce_sum_f16x2_f32_kernel[(triton.cdiv(N, BLOCK_SIZE),)](
        a, y, N, BLOCK_SIZE=BLOCK_SIZE
    )
    return y


def block_all_reduce_sum_f16x8_pack_f16(a: torch.Tensor) -> torch.Tensor:
    N = a.numel()
    BLOCK_SIZE = 2048
    y = _zeros_out()
    block_all_reduce_sum_f16x8_pack_f16_kernel[(triton.cdiv(N, BLOCK_SIZE),)](
        a, y, N, BLOCK_SIZE=BLOCK_SIZE
    )
    return y


def block_all_reduce_sum_f16x8_pack_f32(a: torch.Tensor) -> torch.Tensor:
    N = a.numel()
    BLOCK_SIZE = 2048
    y = _zeros_out()
    block_all_reduce_sum_f16x8_pack_f32_kernel[(triton.cdiv(N, BLOCK_SIZE),)](
        a, y, N, BLOCK_SIZE=BLOCK_SIZE
    )
    return y


def block_all_reduce_sum_bf16_bf16(a: torch.Tensor) -> torch.Tensor:
    N = a.numel()
    BLOCK_SIZE = 256
    y = _zeros_out()
    block_all_reduce_sum_bf16_bf16_kernel[(triton.cdiv(N, BLOCK_SIZE),)](
        a, y, N, BLOCK_SIZE=BLOCK_SIZE
    )
    return y


def block_all_reduce_sum_bf16_f32(a: torch.Tensor) -> torch.Tensor:
    N = a.numel()
    BLOCK_SIZE = 256
    y = _zeros_out()
    block_all_reduce_sum_bf16_f32_kernel[(triton.cdiv(N, BLOCK_SIZE),)](
        a, y, N, BLOCK_SIZE=BLOCK_SIZE
    )
    return y


def block_all_reduce_sum_bf16x2_bf16(a: torch.Tensor) -> torch.Tensor:
    N = a.numel()
    BLOCK_SIZE = 512
    y = _zeros_out()
    block_all_reduce_sum_bf16x2_bf16_kernel[(triton.cdiv(N, BLOCK_SIZE),)](
        a, y, N, BLOCK_SIZE=BLOCK_SIZE
    )
    return y


def block_all_reduce_sum_bf16x2_f32(a: torch.Tensor) -> torch.Tensor:
    N = a.numel()
    BLOCK_SIZE = 512
    y = _zeros_out()
    block_all_reduce_sum_bf16x2_f32_kernel[(triton.cdiv(N, BLOCK_SIZE),)](
        a, y, N, BLOCK_SIZE=BLOCK_SIZE
    )
    return y


def block_all_reduce_sum_bf16x8_pack_bf16(a: torch.Tensor) -> torch.Tensor:
    N = a.numel()
    BLOCK_SIZE = 2048
    y = _zeros_out()
    block_all_reduce_sum_bf16x8_pack_bf16_kernel[(triton.cdiv(N, BLOCK_SIZE),)](
        a, y, N, BLOCK_SIZE=BLOCK_SIZE
    )
    return y


def block_all_reduce_sum_bf16x8_pack_f32(a: torch.Tensor) -> torch.Tensor:
    N = a.numel()
    BLOCK_SIZE = 2048
    y = _zeros_out()
    block_all_reduce_sum_bf16x8_pack_f32_kernel[(triton.cdiv(N, BLOCK_SIZE),)](
        a, y, N, BLOCK_SIZE=BLOCK_SIZE
    )
    return y


# ---------------------------------------------------------------------------
# Correctness checks
# ---------------------------------------------------------------------------


def _close_enough(got: float, ref: float, rtol: float = 0.05) -> bool:
    denom = abs(ref) + 1e-6
    return abs(got - ref) / denom < rtol


def check_correctness():
    torch.manual_seed(42)
    N = 1024 * 1024

    # FP32
    a_f32 = torch.randn(N, device=DEVICE, dtype=torch.float32)
    ref_f32 = torch.sum(a_f32).item()

    assert _close_enough(block_all_reduce_sum_f32(a_f32).item(), ref_f32), "f32 mismatch"
    assert _close_enough(block_all_reduce_sum_f32x4(a_f32).item(), ref_f32), "f32x4 mismatch"

    # FP16
    a_f16 = a_f32.half()
    ref_f16 = torch.sum(a_f32).item()  # reference in f32

    assert _close_enough(block_all_reduce_sum_f16_f16(a_f16).item(), ref_f16, rtol=0.1), "f16_f16 mismatch"
    assert _close_enough(block_all_reduce_sum_f16_f32(a_f16).item(), ref_f16, rtol=0.1), "f16_f32 mismatch"
    assert _close_enough(block_all_reduce_sum_f16x2_f16(a_f16).item(), ref_f16, rtol=0.1), "f16x2_f16 mismatch"
    assert _close_enough(block_all_reduce_sum_f16x2_f32(a_f16).item(), ref_f16, rtol=0.1), "f16x2_f32 mismatch"
    assert _close_enough(block_all_reduce_sum_f16x8_pack_f16(a_f16).item(), ref_f16, rtol=0.1), "f16x8_pack_f16 mismatch"
    assert _close_enough(block_all_reduce_sum_f16x8_pack_f32(a_f16).item(), ref_f16, rtol=0.1), "f16x8_pack_f32 mismatch"

    # BF16
    a_bf16 = a_f32.bfloat16()
    ref_bf16 = torch.sum(a_f32).item()

    assert _close_enough(block_all_reduce_sum_bf16_bf16(a_bf16).item(), ref_bf16, rtol=0.1), "bf16_bf16 mismatch"
    assert _close_enough(block_all_reduce_sum_bf16_f32(a_bf16).item(), ref_bf16, rtol=0.1), "bf16_f32 mismatch"
    assert _close_enough(block_all_reduce_sum_bf16x2_bf16(a_bf16).item(), ref_bf16, rtol=0.1), "bf16x2_bf16 mismatch"
    assert _close_enough(block_all_reduce_sum_bf16x2_f32(a_bf16).item(), ref_bf16, rtol=0.1), "bf16x2_f32 mismatch"
    assert _close_enough(block_all_reduce_sum_bf16x8_pack_bf16(a_bf16).item(), ref_bf16, rtol=0.1), "bf16x8_pack_bf16 mismatch"
    assert _close_enough(block_all_reduce_sum_bf16x8_pack_f32(a_bf16).item(), ref_bf16, rtol=0.1), "bf16x8_pack_f32 mismatch"

    print("All correctness checks passed.")


# ---------------------------------------------------------------------------
# Manual timing benchmark (matches block_all_reduce.py format)
# ---------------------------------------------------------------------------


def run_benchmark(
    perf_func: callable,
    values: torch.Tensor,
    tag: str,
    warmup: int = 10,
    iters: int = 1000,
):
    for _ in range(warmup):
        out = perf_func(values)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        out = perf_func(values)
    torch.cuda.synchronize()
    end = time.time()
    mean_time = (end - start) * 1000 / iters
    out_val = out.item()
    print(f"{'out_' + tag:>30}: {out_val:<15.6f}, time:{mean_time:.8f}ms")
    return out, mean_time


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
            "f16_f16",
            "f16_f32",
            "f16x8_pack_f32",
            "f16_torch",
        ],
        line_names=[
            "f32",
            "f32x4",
            "f32 (torch)",
            "f16_f16",
            "f16_f32",
            "f16x8_pack_f32",
            "f16 (torch)",
        ],
        styles=[
            ("blue", "-"),
            ("blue", "--"),
            ("blue", ":"),
            ("red", "-"),
            ("red", "--"),
            ("red", "-."),
            ("green", "-"),
        ],
        ylabel="GB/s",
        plot_name="block-all-reduce-sum-performance",
        args={},
    )
)
def bench_reduce(N, kernel):
    quantiles = [0.5, 0.2, 0.8]

    if kernel in ("f32", "f32x4", "f32_torch"):
        a = torch.randn(N, device=DEVICE, dtype=torch.float32)
        elem_bytes = 4
    else:
        a = torch.randn(N, device=DEVICE, dtype=torch.float16)
        elem_bytes = 2

    if kernel == "f32":
        fn = lambda: block_all_reduce_sum_f32(a)
    elif kernel == "f32x4":
        fn = lambda: block_all_reduce_sum_f32x4(a)
    elif kernel == "f32_torch":
        fn = lambda: torch.sum(a)
    elif kernel == "f16_f16":
        fn = lambda: block_all_reduce_sum_f16_f16(a)
    elif kernel == "f16_f32":
        fn = lambda: block_all_reduce_sum_f16_f32(a)
    elif kernel == "f16x8_pack_f32":
        fn = lambda: block_all_reduce_sum_f16x8_pack_f32(a)
    elif kernel == "f16_torch":
        fn = lambda: torch.sum(a)

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    # Only reads: 1 tensor of N elements
    gbps = lambda ms: N * elem_bytes * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    check_correctness()

    Ss = [1024, 2048, 4096]
    Ks = [1024, 2048, 4096]

    for S, K in [(S, K) for S in Ss for K in Ks]:
        print("-" * 85)
        print(" " * 40 + f"S={S}, K={K}")

        values_f32 = torch.randn((S, K), device=DEVICE, dtype=torch.float32).contiguous()
        run_benchmark(block_all_reduce_sum_f32, values_f32, "f32_triton")
        run_benchmark(block_all_reduce_sum_f32x4, values_f32, "f32x4_triton")
        run_benchmark(torch.sum, values_f32, "f32_torch")

        print("-" * 85)

        values_f16 = values_f32.half().contiguous()
        run_benchmark(block_all_reduce_sum_f16_f16, values_f16, "f16_f16_triton")
        run_benchmark(block_all_reduce_sum_f16_f32, values_f16, "f16_f32_triton")
        run_benchmark(block_all_reduce_sum_f16x2_f16, values_f16, "f16x2_f16_triton")
        run_benchmark(block_all_reduce_sum_f16x2_f32, values_f16, "f16x2_f32_triton")
        run_benchmark(block_all_reduce_sum_f16x8_pack_f16, values_f16, "f16x8pack_f16_triton")
        run_benchmark(block_all_reduce_sum_f16x8_pack_f32, values_f16, "f16x8pack_f32_triton")
        run_benchmark(torch.sum, values_f16, "f16_torch")

        print("-" * 85)

        values_bf16 = values_f32.bfloat16().contiguous()
        run_benchmark(block_all_reduce_sum_bf16_bf16, values_bf16, "bf16_bf16_triton")
        run_benchmark(block_all_reduce_sum_bf16_f32, values_bf16, "bf16_f32_triton")
        run_benchmark(block_all_reduce_sum_bf16x2_bf16, values_bf16, "bf16x2_bf16_triton")
        run_benchmark(block_all_reduce_sum_bf16x2_f32, values_bf16, "bf16x2_f32_triton")
        run_benchmark(block_all_reduce_sum_bf16x8_pack_bf16, values_bf16, "bf16x8pack_bf16_triton")
        run_benchmark(block_all_reduce_sum_bf16x8_pack_f32, values_bf16, "bf16x8pack_f32_triton")
        run_benchmark(torch.sum, values_bf16, "bf16_torch")

        print("-" * 85)

    print("\nRunning triton.testing.perf_report benchmark (GB/s by N)...")
    bench_reduce.run(print_data=True, show_plots=False, save_path="./")
