"""
Triton implementations of Layer Norm kernels from layer_norm.cu.

Each Triton program handles one row of K elements (grid=(N,)).
BLOCK_SIZE = next_power_of_2(K) so a single program covers the full row in one
pass — matching the CUDA pattern of one block per row.

Variants and their CUDA analogues:
  f32             - FP32 in/out, FP32 accumulator (layer_norm_f32_kernel)
  f32x4           - FP32 in/out, FP32 acc (float4 vectorization transparent in Triton)
  f16_f16         - FP16 in/out, FP16 accumulator for variance (overflow-prone)
  f16_f32         - FP16 in/out, FP32 accumulator (higher precision, no overflow)
  f16x2_f16       - FP16 in/out, FP16 acc (half2 vectorization transparent in Triton)
  f16x8_f16       - FP16 in/out, FP16 acc
  f16x8_pack_f16  - FP16 in/out, FP16 acc (128-bit LDST transparent in Triton)
  f16x8_pack_f32  - FP16 in/out, FP32 acc
"""

import time
from functools import partial
from typing import Optional

import torch
import triton
import triton.language as tl

torch.set_grad_enabled(False)

DEVICE = torch.device("cuda:0")


# ---------------------------------------------------------------------------
# Triton kernels — one program per row
# ---------------------------------------------------------------------------


@triton.jit
def layer_norm_f32_kernel(x_ptr, y_ptr, g, b, N, K, BLOCK_SIZE: tl.constexpr):
    """FP32 in/out, FP32 accumulator."""
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < K
    x = tl.load(x_ptr + row * K + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / K
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / K
    inv_std = 1.0 / tl.sqrt(var + 1e-5)
    y = x_centered * inv_std * g + b
    tl.store(y_ptr + row * K + offsets, y.to(tl.float32), mask=mask)


@triton.jit
def layer_norm_f16_f16_kernel(x_ptr, y_ptr, g, b, N, K, BLOCK_SIZE: tl.constexpr):
    """FP16 in/out, FP16 accumulator for variance.

    Mirrors CUDA kernels that keep the variance reduction in fp16.
    Susceptible to overflow on large input values (see benchmark overflow section).
    """
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < K
    x = tl.load(x_ptr + row * K + offsets, mask=mask, other=0.0).to(tl.float16)
    mean = tl.sum(x, axis=0).to(tl.float32) / K
    # Use f16 for the x*x sum to demonstrate overflow
    x16 = x - mean.to(tl.float16)
    var16 = tl.sum(x16 * x16, axis=0).to(tl.float32) / K
    inv_std = 1.0 / tl.sqrt(var16 + 1e-5)
    y = x16 * inv_std.to(tl.float16) * g.to(tl.float16) + b.to(tl.float16)
    tl.store(y_ptr + row * K + offsets, y.to(tl.float16), mask=mask)


@triton.jit
def layer_norm_f16_f32_kernel(x_ptr, y_ptr, g, b, N, K, BLOCK_SIZE: tl.constexpr):
    """FP16 in/out, FP32 accumulator for variance.

    Mirrors CUDA kernels that upcast to fp32 before the variance reduction.
    Numerically stable under large inputs.
    """
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < K
    x = tl.load(x_ptr + row * K + offsets, mask=mask, other=0.0).to(tl.float16)
    xf = x.to(tl.float32)
    mean = tl.sum(xf, axis=0) / K
    x_centered = xf - mean
    var = tl.sum(x_centered * x_centered, axis=0) / K  # accumulate in f32 — no overflow
    inv_std = 1.0 / tl.sqrt(var + 1e-5)
    y = x_centered * inv_std * g + b
    tl.store(y_ptr + row * K + offsets, y.to(tl.float16), mask=mask)


# ---------------------------------------------------------------------------
# Python wrappers — each takes (x, out, g, b) and writes into out,
# matching the signature used by run_benchmark in layer_norm.py.
# g and b are scalar floats.
# ---------------------------------------------------------------------------


def layer_norm_f32(x: torch.Tensor, out: torch.Tensor, g: float, b: float):
    N, K = x.shape
    layer_norm_f32_kernel[(N,)](x, out, g, b, N, K, BLOCK_SIZE=triton.next_power_of_2(K))


def layer_norm_f32x4(x: torch.Tensor, out: torch.Tensor, g: float, b: float):
    """float4 vectorization is transparent in Triton; delegates to layer_norm_f32_kernel."""
    N, K = x.shape
    layer_norm_f32_kernel[(N,)](x, out, g, b, N, K, BLOCK_SIZE=triton.next_power_of_2(K))


def layer_norm_f16_f16(x: torch.Tensor, out: torch.Tensor, g: float, b: float):
    N, K = x.shape
    layer_norm_f16_f16_kernel[(N,)](x, out, g, b, N, K, BLOCK_SIZE=triton.next_power_of_2(K))


def layer_norm_f16_f32(x: torch.Tensor, out: torch.Tensor, g: float, b: float):
    N, K = x.shape
    layer_norm_f16_f32_kernel[(N,)](x, out, g, b, N, K, BLOCK_SIZE=triton.next_power_of_2(K))


def layer_norm_f16x2_f16(x: torch.Tensor, out: torch.Tensor, g: float, b: float):
    """half2 vectorization is transparent in Triton; delegates to layer_norm_f16_f16_kernel."""
    N, K = x.shape
    layer_norm_f16_f16_kernel[(N,)](x, out, g, b, N, K, BLOCK_SIZE=triton.next_power_of_2(K))


def layer_norm_f16x8_f16(x: torch.Tensor, out: torch.Tensor, g: float, b: float):
    N, K = x.shape
    layer_norm_f16_f16_kernel[(N,)](x, out, g, b, N, K, BLOCK_SIZE=triton.next_power_of_2(K))


def layer_norm_f16x8_pack_f16(x: torch.Tensor, out: torch.Tensor, g: float, b: float):
    """128-bit LDST pack is transparent in Triton; delegates to layer_norm_f16_f16_kernel."""
    N, K = x.shape
    layer_norm_f16_f16_kernel[(N,)](x, out, g, b, N, K, BLOCK_SIZE=triton.next_power_of_2(K))


def layer_norm_f16x8_pack_f32(x: torch.Tensor, out: torch.Tensor, g: float, b: float):
    """128-bit LDST pack is transparent in Triton; delegates to layer_norm_f16_f32_kernel."""
    N, K = x.shape
    layer_norm_f16_f32_kernel[(N,)](x, out, g, b, N, K, BLOCK_SIZE=triton.next_power_of_2(K))


# ---------------------------------------------------------------------------
# PyTorch reference implementation
# ---------------------------------------------------------------------------


def naive_layer_norm(x: torch.Tensor, g: float, b: float) -> torch.Tensor:
    mean = torch.mean(x, dim=1, keepdim=True)
    std_inv = 1 / torch.std(x, dim=1, keepdim=True)
    return ((x - mean) * std_inv) * g + b


# ---------------------------------------------------------------------------
# Benchmark harness — mirrors layer_norm.py structure exactly
# ---------------------------------------------------------------------------


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
    b = 0.0
    if out is not None:
        out.fill_(0)
    if out is not None:
        for i in range(warmup):
            perf_func(x, out, g, b)
    else:
        for i in range(warmup):
            _ = perf_func(x, g, b)
    torch.cuda.synchronize()
    start = time.time()
    if out is not None:
        for i in range(iters):
            perf_func(x, out, g, b)
    else:
        for i in range(iters):
            out = perf_func(x, g, b)
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()[:3]
    out_val = [round(v, 8) for v in out_val]
    out_val = [f"{v:<12}" for v in out_val]
    print(f"{out_info:>30}: {out_val}, time:{mean_time:.8f}ms")
    if show_all:
        print(out)
    return out, mean_time


# ---------------------------------------------------------------------------
# triton.testing.perf_report benchmark
# ---------------------------------------------------------------------------

configs = [
    triton.testing.Benchmark(
        x_names=["K"],
        x_vals=[512, 1024, 2048, 4096, 8192],
        line_arg="provider",
        line_vals=["f32_triton", "f32x4_triton", "f16f16_triton", "f16f32_triton",
                   "f16x8packf16_triton", "f16x8packf32_triton", "torch_f32", "torch_f16"],
        line_names=["f32 (Triton)", "f32x4 (Triton)", "f16f16 (Triton)", "f16f32 (Triton)",
                    "f16x8packf16 (Triton)", "f16x8packf32 (Triton)", "Torch f32", "Torch f16"],
        styles=[("blue", "-"), ("blue", "--"), ("red", "-"), ("red", "--"),
                ("green", "-"), ("green", "--"), ("black", "-"), ("black", "--")],
        ylabel="GB/s",
        plot_name="layer-norm-performance",
        args={"N": 4096},
    )
]


@triton.testing.perf_report(configs)
def bench_layer_norm(N, K, provider):
    g = 1.0
    b = 0.0
    if "f16" in provider:
        x = torch.randn((N, K), device=DEVICE, dtype=torch.float16).contiguous()
        out = torch.empty_like(x)
    else:
        x = torch.randn((N, K), device=DEVICE, dtype=torch.float32).contiguous()
        out = torch.empty_like(x)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "f32_triton":
        fn = lambda: layer_norm_f32(x, out, g, b)
    elif provider == "f32x4_triton":
        fn = lambda: layer_norm_f32x4(x, out, g, b)
    elif provider == "f16f16_triton":
        fn = lambda: layer_norm_f16_f16(x, out, g, b)
    elif provider == "f16f32_triton":
        fn = lambda: layer_norm_f16_f32(x, out, g, b)
    elif provider == "f16x8packf16_triton":
        fn = lambda: layer_norm_f16x8_pack_f16(x, out, g, b)
    elif provider == "f16x8packf32_triton":
        fn = lambda: layer_norm_f16x8_pack_f32(x, out, g, b)
    elif provider == "torch_f32":
        fn = lambda: naive_layer_norm(x, g, b)
    elif provider == "torch_f16":
        fn = lambda: naive_layer_norm(x, g, b)

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    # 2 passes over the data (read + write), each element is x.element_size() bytes
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


# ---------------------------------------------------------------------------
# Main benchmark runs — mirrors layer_norm.py structure exactly
# ---------------------------------------------------------------------------

print("-" * 85)
N, K = 4096, 512
print(" " * 40 + f"N={N}, K={K}")
print("-" * 85)
x = torch.randn((N, K), device=DEVICE, dtype=torch.float32).contiguous()
out = torch.zeros_like(x).contiguous()
run_benchmark(layer_norm_f32,    x, "f32_triton",    out)
run_benchmark(layer_norm_f32x4,  x, "f32x4_triton",  out)
run_benchmark(naive_layer_norm,  x, "f32_th")

print("-" * 85)
x_f16 = x.half()
out_f16 = out.half()
run_benchmark(layer_norm_f16_f16,       x_f16, "f16f16_triton",       out_f16)
run_benchmark(layer_norm_f16_f32,       x_f16, "f16f32_triton",       out_f16)
run_benchmark(layer_norm_f16x2_f16,     x_f16, "f16x2f16_triton",     out_f16)
run_benchmark(layer_norm_f16x8_f16,     x_f16, "f16x8f16_triton",     out_f16)
run_benchmark(layer_norm_f16x8_pack_f16, x_f16, "f16x8packf16_triton", out_f16)
run_benchmark(layer_norm_f16x8_pack_f32, x_f16, "f16x8packf32_triton", out_f16)
run_benchmark(naive_layer_norm,          x_f16, "f16_th")
print("-" * 85)

print(" " * 40 + f"f16 overflow without f32")
print("-" * 85)
x_f16 = x.half() * 100  # this will cause overflow for kernels without `f32`
run_benchmark(layer_norm_f16_f16,       x_f16, "f16f16_triton",       out_f16)
run_benchmark(layer_norm_f16_f32,       x_f16, "f16f32_triton",       out_f16)
run_benchmark(layer_norm_f16x2_f16,     x_f16, "f16x2f16_triton",     out_f16)
run_benchmark(layer_norm_f16x8_f16,     x_f16, "f16x8f16_triton",     out_f16)
run_benchmark(layer_norm_f16x8_pack_f16, x_f16, "f16x8packf16_triton", out_f16)
run_benchmark(layer_norm_f16x8_pack_f32, x_f16, "f16x8packf32_triton", out_f16)
run_benchmark(naive_layer_norm,          x_f16, "f16_th")
print("-" * 85)

print("-" * 85)
N, K = 4096, 1024
print(" " * 40 + f"N={N}, K={K}")
print("-" * 85)
x = torch.randn((N, K), device=DEVICE, dtype=torch.float32).contiguous()
out = torch.zeros_like(x).contiguous()
run_benchmark(layer_norm_f32,    x, "f32_triton",    out)
run_benchmark(layer_norm_f32x4,  x, "f32x4_triton",  out)
run_benchmark(naive_layer_norm,  x, "f32_th")

print("-" * 85)
x_f16 = x.half()
out_f16 = out.half()
run_benchmark(layer_norm_f16_f16,       x_f16, "f16f16_triton",       out_f16)
run_benchmark(layer_norm_f16_f32,       x_f16, "f16f32_triton",       out_f16)
run_benchmark(layer_norm_f16x2_f16,     x_f16, "f16x2f16_triton",     out_f16)
run_benchmark(layer_norm_f16x8_f16,     x_f16, "f16x8f16_triton",     out_f16)
run_benchmark(layer_norm_f16x8_pack_f16, x_f16, "f16x8packf16_triton", out_f16)
run_benchmark(layer_norm_f16x8_pack_f32, x_f16, "f16x8packf32_triton", out_f16)
run_benchmark(naive_layer_norm,          x_f16, "f16_th")
print("-" * 85)

print("-" * 85)
N, K = 4096, 2048
print(" " * 40 + f"N={N}, K={K}")
print("-" * 85)
x = torch.randn((N, K), device=DEVICE, dtype=torch.float32).contiguous()
out = torch.zeros_like(x).contiguous()
run_benchmark(layer_norm_f32x4,  x, "f32x4_triton",  out)
run_benchmark(naive_layer_norm,  x, "f32_th")

print("-" * 85)
x_f16 = x.half()
out_f16 = out.half()
run_benchmark(layer_norm_f16x2_f16,     x_f16, "f16x2f16_triton",     out_f16)
run_benchmark(layer_norm_f16x8_f16,     x_f16, "f16x8f16_triton",     out_f16)
run_benchmark(layer_norm_f16x8_pack_f16, x_f16, "f16x8packf16_triton", out_f16)
run_benchmark(layer_norm_f16x8_pack_f32, x_f16, "f16x8packf32_triton", out_f16)
run_benchmark(naive_layer_norm,          x_f16, "f16_th")
print("-" * 85)

print("-" * 85)
N, K = 4096, 4096
print(" " * 40 + f"N={N}, K={K}")
print("-" * 85)
x = torch.randn((N, K), device=DEVICE, dtype=torch.float32).contiguous()
out = torch.zeros_like(x).contiguous()
run_benchmark(layer_norm_f32x4,  x, "f32x4_triton",  out)
run_benchmark(naive_layer_norm,  x, "f32_th")

print("-" * 85)
x_f16 = x.half()
out_f16 = out.half()
run_benchmark(layer_norm_f16x8_f16,     x_f16, "f16x8f16_triton",     out_f16)
run_benchmark(layer_norm_f16x8_pack_f16, x_f16, "f16x8packf16_triton", out_f16)
run_benchmark(layer_norm_f16x8_pack_f32, x_f16, "f16x8packf32_triton", out_f16)
run_benchmark(naive_layer_norm,          x_f16, "f16_th")
print("-" * 85)

print("-" * 85)
N, K = 4096, 8192
print(" " * 40 + f"N={N}, K={K}")
print("-" * 85)
x_f16 = torch.randn((N, K), device=DEVICE, dtype=torch.float16).contiguous()
out_f16 = torch.zeros_like(x_f16).contiguous()
run_benchmark(layer_norm_f16x8_f16,     x_f16, "f16x8f16_triton",     out_f16)
run_benchmark(layer_norm_f16x8_pack_f16, x_f16, "f16x8packf16_triton", out_f16)
run_benchmark(layer_norm_f16x8_pack_f32, x_f16, "f16x8packf32_triton", out_f16)
run_benchmark(naive_layer_norm,          x_f16, "f16_th")
print("-" * 85)

print("-" * 85)
N, K = 8192, 8192
print(" " * 40 + f"N={N}, K={K}")
print("-" * 85)
x_f16 = torch.randn((N, K), device=DEVICE, dtype=torch.float16).contiguous()
out_f16 = torch.zeros_like(x_f16).contiguous()
run_benchmark(layer_norm_f16x8_f16,     x_f16, "f16x8f16_triton",     out_f16)
run_benchmark(layer_norm_f16x8_pack_f16, x_f16, "f16x8packf16_triton", out_f16)
run_benchmark(layer_norm_f16x8_pack_f32, x_f16, "f16x8packf32_triton", out_f16)
run_benchmark(naive_layer_norm,          x_f16, "f16_th")
print("-" * 85)

# Run the triton.testing.perf_report benchmark (saves plots)
bench_layer_norm.run(show_plots=False, print_data=True)
