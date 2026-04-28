"""
Triton implementations of Softmax kernels from softmax.cu.

Each Triton program handles one row of H elements (grid=(S,)).
BLOCK_SIZE = next_power_of_2(H) so a single program covers the full row in one
pass — matching the CUDA pattern of one block per row.

Online safe softmax per row:
  max_val    = max(x[row])
  shifted    = x[row] - max_val      # shift for numerical stability
  exp_shifted = exp(shifted)
  sum_exp    = sum(exp_shifted)
  y[row]     = exp_shifted / sum_exp

Variants and their CUDA analogues:
  softmax_f32_per_token             - FP32 in/out (softmax_f32_per_token_kernel)
  softmax_f32x4_per_token           - FP32 in/out (float4 transparent in Triton)
  safe_softmax_f32_per_token        - FP32 in/out, safe (shifted) softmax
  online_safe_softmax_f32_per_token - FP32 in/out, online safe softmax
  online_safe_softmax_f32x4_pack    - FP32 in/out, f32x4 transparent in Triton
  safe_softmax_f32x4_per_token      - FP32 in/out, safe f32x4 transparent
  safe_softmax_f16_f32_per_token    - FP16 in/out, FP32 accumulator
  safe_softmax_f16x2_f32_per_token  - FP16 in/out, FP32 acc (half2 transparent)
  safe_softmax_f16x8_pack_f32       - FP16 in/out, FP32 acc (128-bit LDST transparent)
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
def softmax_f32_kernel(x_ptr, y_ptr, S, H, BLOCK_SIZE: tl.constexpr):
    """FP32 in/out, online safe softmax."""
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < H
    x = tl.load(x_ptr + row * H + offsets, mask=mask, other=-float("inf"))
    x_max = tl.max(x, axis=0)
    x_shifted = x - x_max
    x_exp = tl.math.exp(x_shifted)
    x_exp_sum = tl.sum(tl.where(mask, x_exp, 0.0), axis=0)
    y = tl.where(mask, x_exp / x_exp_sum, 0.0)
    tl.store(y_ptr + row * H + offsets, y, mask=mask)


@triton.jit
def softmax_f16_f32_kernel(x_ptr, y_ptr, S, H, BLOCK_SIZE: tl.constexpr):
    """FP16 in/out, FP32 accumulator, online safe softmax."""
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < H
    x = tl.load(x_ptr + row * H + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    x_max = tl.max(x, axis=0)
    x_shifted = x - x_max
    x_exp = tl.math.exp(x_shifted)
    x_exp_sum = tl.sum(tl.where(mask, x_exp, 0.0), axis=0)
    y = tl.where(mask, x_exp / x_exp_sum, 0.0)
    tl.store(y_ptr + row * H + offsets, y.to(tl.float16), mask=mask)


# ---------------------------------------------------------------------------
# Python wrappers — each takes (x, out) and writes into out,
# matching the signature used by run_benchmark in softmax.py.
# ---------------------------------------------------------------------------


def softmax_f32_per_token(x: torch.Tensor, out: torch.Tensor):
    S, H = x.shape
    softmax_f32_kernel[(S,)](x, out, S, H, BLOCK_SIZE=triton.next_power_of_2(H))


def softmax_f32x4_per_token(x: torch.Tensor, out: torch.Tensor):
    """float4 vectorization is transparent in Triton; delegates to softmax_f32_kernel."""
    S, H = x.shape
    softmax_f32_kernel[(S,)](x, out, S, H, BLOCK_SIZE=triton.next_power_of_2(H))


def safe_softmax_f32_per_token(x: torch.Tensor, out: torch.Tensor):
    """Safe (shifted) softmax; same kernel — shift applied inside softmax_f32_kernel."""
    S, H = x.shape
    softmax_f32_kernel[(S,)](x, out, S, H, BLOCK_SIZE=triton.next_power_of_2(H))


def online_safe_softmax_f32_per_token(x: torch.Tensor, out: torch.Tensor):
    """Online safe softmax; single-pass numerically stable variant."""
    S, H = x.shape
    softmax_f32_kernel[(S,)](x, out, S, H, BLOCK_SIZE=triton.next_power_of_2(H))


def online_safe_softmax_f32x4_pack_per_token(x: torch.Tensor, out: torch.Tensor):
    """f32x4 pack is transparent in Triton; delegates to softmax_f32_kernel."""
    S, H = x.shape
    softmax_f32_kernel[(S,)](x, out, S, H, BLOCK_SIZE=triton.next_power_of_2(H))


def safe_softmax_f32x4_per_token(x: torch.Tensor, out: torch.Tensor):
    """float4 safe softmax; vectorization transparent in Triton."""
    S, H = x.shape
    softmax_f32_kernel[(S,)](x, out, S, H, BLOCK_SIZE=triton.next_power_of_2(H))


def safe_softmax_f16_f32_per_token(x: torch.Tensor, out: torch.Tensor):
    """FP16 input, FP32 accumulator safe softmax."""
    S, H = x.shape
    softmax_f16_f32_kernel[(S,)](x, out, S, H, BLOCK_SIZE=triton.next_power_of_2(H))


def safe_softmax_f16x2_f32_per_token(x: torch.Tensor, out: torch.Tensor):
    """half2 vectorization is transparent in Triton; delegates to softmax_f16_f32_kernel."""
    S, H = x.shape
    softmax_f16_f32_kernel[(S,)](x, out, S, H, BLOCK_SIZE=triton.next_power_of_2(H))


def safe_softmax_f16x8_pack_f32_per_token(x: torch.Tensor, out: torch.Tensor):
    """128-bit LDST pack is transparent in Triton; delegates to softmax_f16_f32_kernel."""
    S, H = x.shape
    softmax_f16_f32_kernel[(S,)](x, out, S, H, BLOCK_SIZE=triton.next_power_of_2(H))


# ---------------------------------------------------------------------------
# Benchmark harness — mirrors softmax.py structure exactly
# ---------------------------------------------------------------------------


def run_benchmark(
    perf_func: callable,
    x: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 100,
    show_all: bool = False,
):
    if out is not None:
        out.fill_(0)
    if out is not None:
        for i in range(warmup):
            perf_func(x, out)
    else:
        for i in range(warmup):
            _ = perf_func(x)
    torch.cuda.synchronize()
    start = time.time()
    if out is not None:
        for i in range(iters):
            perf_func(x, out)
    else:
        for i in range(iters):
            out = perf_func(x)
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()[:3]
    out_val = [round(v, 8) for v in out_val]
    out_val = [f"{v:<12}" for v in out_val]
    print(f"{out_info:>32}: {out_val}, time:{mean_time:.8f}ms")
    if show_all:
        print(out)
    return out, mean_time


# ---------------------------------------------------------------------------
# triton.testing.perf_report benchmark
# ---------------------------------------------------------------------------

configs = [
    triton.testing.Benchmark(
        x_names=["H"],
        x_vals=[256, 512, 1024, 2048, 4096, 8192],
        line_arg="provider",
        line_vals=[
            "f32_triton", "f32x4_triton", "safe_f32_triton", "online_safe_f32_triton",
            "f16f32_triton", "f16x8packf32_triton", "torch_f32", "torch_f16",
        ],
        line_names=[
            "f32 (Triton)", "f32x4 (Triton)", "safe f32 (Triton)", "online safe f32 (Triton)",
            "f16f32 (Triton)", "f16x8packf32 (Triton)", "Torch f32", "Torch f16",
        ],
        styles=[
            ("blue", "-"), ("blue", "--"), ("cyan", "-"), ("cyan", "--"),
            ("red", "-"), ("red", "--"), ("black", "-"), ("black", "--"),
        ],
        ylabel="GB/s",
        plot_name="softmax-performance",
        args={"S": 4096},
    )
]


@triton.testing.perf_report(configs)
def bench_softmax(S, H, provider):
    if "f16" in provider:
        x = torch.randn((S, H), device=DEVICE, dtype=torch.float16).contiguous()
        out = torch.empty_like(x)
    else:
        x = torch.randn((S, H), device=DEVICE, dtype=torch.float32).contiguous()
        out = torch.empty_like(x)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "f32_triton":
        fn = lambda: softmax_f32_per_token(x, out)
    elif provider == "f32x4_triton":
        fn = lambda: softmax_f32x4_per_token(x, out)
    elif provider == "safe_f32_triton":
        fn = lambda: safe_softmax_f32_per_token(x, out)
    elif provider == "online_safe_f32_triton":
        fn = lambda: online_safe_softmax_f32_per_token(x, out)
    elif provider == "f16f32_triton":
        fn = lambda: safe_softmax_f16_f32_per_token(x, out)
    elif provider == "f16x8packf32_triton":
        fn = lambda: safe_softmax_f16x8_pack_f32_per_token(x, out)
    elif provider == "torch_f32":
        fn = lambda: torch.softmax(x, dim=1, out=out)
    elif provider == "torch_f16":
        fn = lambda: torch.softmax(x, dim=1, out=out)

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    # 2 passes over data (read + write), each element is x.element_size() bytes
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


# ---------------------------------------------------------------------------
# Main benchmark runs — mirrors softmax.py structure exactly
# ---------------------------------------------------------------------------

# per token softmax
print("-" * 100)
S, H = 4096, 256
print(" " * 45 + f"S={S}, H={H}")
print("-" * 100)
x = torch.randn((S, H), device=DEVICE, dtype=torch.float32).contiguous()
out = torch.zeros_like(x).contiguous()
run_benchmark(softmax_f32_per_token,                      x, "f32(per)",          out)
run_benchmark(softmax_f32x4_per_token,                    x, "f32x4(per)",        out)
run_benchmark(safe_softmax_f32_per_token,                 x, "f32(safe)",         out)
run_benchmark(online_safe_softmax_f32_per_token,          x, "f32(safe+online)",  out)
run_benchmark(online_safe_softmax_f32x4_pack_per_token,   x, "f32x4(safe+online)", out)
run_benchmark(safe_softmax_f32x4_per_token,               x, "f32x4(safe)",       out)
run_benchmark(partial(torch.softmax, dim=1, out=out),     x, "f32_th(per)")

print("-" * 100)
x_f16 = x.half().contiguous()
out_f16 = out.half().contiguous()
run_benchmark(safe_softmax_f16_f32_per_token,             x_f16, "f16f32(safe)",       out_f16)
run_benchmark(safe_softmax_f16x2_f32_per_token,           x_f16, "f16x2f32(safe)",     out_f16)
run_benchmark(safe_softmax_f16x8_pack_f32_per_token,      x_f16, "f16x8packf32(safe)", out_f16)
run_benchmark(partial(torch.softmax, dim=1, out=out_f16), x_f16, "f16_th(per)")
print("-" * 100)

# per token softmax
print("-" * 100)
S, H = 4096, 512
print(" " * 45 + f"S={S}, H={H}")
print("-" * 100)
x = torch.randn((S, H), device=DEVICE, dtype=torch.float32).contiguous()
out = torch.zeros_like(x).contiguous()
run_benchmark(softmax_f32_per_token,                      x, "f32(per)",          out)
run_benchmark(softmax_f32x4_per_token,                    x, "f32x4(per)",        out)
run_benchmark(safe_softmax_f32_per_token,                 x, "f32(safe)",         out)
run_benchmark(online_safe_softmax_f32_per_token,          x, "f32(safe+online)",  out)
run_benchmark(online_safe_softmax_f32x4_pack_per_token,   x, "f32x4(safe+online)", out)
run_benchmark(safe_softmax_f32x4_per_token,               x, "f32x4(safe)",       out)
run_benchmark(partial(torch.softmax, dim=1, out=out),     x, "f32_th(per)")

print("-" * 100)
x_f16 = x.half().contiguous()
out_f16 = out.half().contiguous()
run_benchmark(safe_softmax_f16_f32_per_token,             x_f16, "f16f32(safe)",       out_f16)
run_benchmark(safe_softmax_f16x2_f32_per_token,           x_f16, "f16x2f32(safe)",     out_f16)
run_benchmark(safe_softmax_f16x8_pack_f32_per_token,      x_f16, "f16x8packf32(safe)", out_f16)
run_benchmark(partial(torch.softmax, dim=1, out=out_f16), x_f16, "f16_th(per)")
print("-" * 100)

# per token softmax
print("-" * 100)
S, H = 4096, 1024
print(" " * 45 + f"S={S}, H={H}")
print("-" * 100)
x = torch.randn((S, H), device=DEVICE, dtype=torch.float32).contiguous()
out = torch.zeros_like(x).contiguous()
run_benchmark(softmax_f32_per_token,                      x, "f32(per)",          out)
run_benchmark(softmax_f32x4_per_token,                    x, "f32x4(per)",        out)
run_benchmark(safe_softmax_f32_per_token,                 x, "f32(safe)",         out)
run_benchmark(online_safe_softmax_f32_per_token,          x, "f32(safe+online)",  out)
run_benchmark(online_safe_softmax_f32x4_pack_per_token,   x, "f32x4(safe+online)", out)
run_benchmark(safe_softmax_f32x4_per_token,               x, "f32x4(safe)",       out)
run_benchmark(partial(torch.softmax, dim=1, out=out),     x, "f32_th(per)")

print("-" * 100)
x_f16 = x.half().contiguous()
out_f16 = out.half().contiguous()
run_benchmark(safe_softmax_f16_f32_per_token,             x_f16, "f16f32(safe)",       out_f16)
run_benchmark(safe_softmax_f16x2_f32_per_token,           x_f16, "f16x2f32(safe)",     out_f16)
run_benchmark(safe_softmax_f16x8_pack_f32_per_token,      x_f16, "f16x8packf32(safe)", out_f16)
run_benchmark(partial(torch.softmax, dim=1, out=out_f16), x_f16, "f16_th(per)")
print("-" * 100)

# per token softmax
print("-" * 100)
S, H = 4096, 2048
print(" " * 45 + f"S={S}, H={H}")
print("-" * 100)
x = torch.randn((S, H), device=DEVICE, dtype=torch.float32).contiguous()
out = torch.zeros_like(x).contiguous()
run_benchmark(softmax_f32x4_per_token,                    x, "f32x4(per)",        out)
run_benchmark(safe_softmax_f32x4_per_token,               x, "f32x4(safe)",       out)
run_benchmark(online_safe_softmax_f32x4_pack_per_token,   x, "f32x4(safe+online)", out)
run_benchmark(partial(torch.softmax, dim=1, out=out),     x, "f32_th(per)")

print("-" * 100)
x_f16 = x.half().contiguous()
out_f16 = out.half().contiguous()
run_benchmark(safe_softmax_f16x2_f32_per_token,           x_f16, "f16x2f32(safe)",     out_f16)
run_benchmark(safe_softmax_f16x8_pack_f32_per_token,      x_f16, "f16x8packf32(safe)", out_f16)
run_benchmark(partial(torch.softmax, dim=1, out=out_f16), x_f16, "f16_th(per)")
print("-" * 100)

# per token softmax
print("-" * 100)
S, H = 4096, 4096
print(" " * 45 + f"S={S}, H={H}")
print("-" * 100)
x = torch.randn((S, H), device=DEVICE, dtype=torch.float32).contiguous()
out = torch.zeros_like(x).contiguous()
run_benchmark(softmax_f32x4_per_token,                    x, "f32x4(per)",        out)
run_benchmark(safe_softmax_f32x4_per_token,               x, "f32x4(safe)",       out)
run_benchmark(online_safe_softmax_f32x4_pack_per_token,   x, "f32x4(safe+online)", out)
run_benchmark(partial(torch.softmax, dim=1, out=out),     x, "f32_th(per)")

print("-" * 100)
x_f16 = x.half().contiguous()
out_f16 = out.half().contiguous()
run_benchmark(safe_softmax_f16x8_pack_f32_per_token,      x_f16, "f16x8packf32(safe)", out_f16)
run_benchmark(partial(torch.softmax, dim=1, out=out_f16), x_f16, "f16_th(per)")
print("-" * 100)

# per token softmax
print("-" * 100)
S, H = 4096, 8192
print(" " * 45 + f"S={S}, H={H}")
print("-" * 100)
x = torch.randn((S, H), device=DEVICE, dtype=torch.float32).contiguous()
out = torch.zeros_like(x).contiguous()
x_f16 = x.half().contiguous()
out_f16 = out.half().contiguous()
run_benchmark(safe_softmax_f16x8_pack_f32_per_token,      x_f16, "f16x8packf32(safe)", out_f16)
run_benchmark(partial(torch.softmax, dim=1, out=out_f16), x_f16, "f16_th(per)")

# per token softmax
print("-" * 100)
S, H = 8192, 8192
print(" " * 45 + f"S={S}, H={H}")
print("-" * 100)
x = torch.randn((S, H), device=DEVICE, dtype=torch.float32).contiguous()
out = torch.zeros_like(x).contiguous()
x_f16 = x.half().contiguous()
out_f16 = out.half().contiguous()
run_benchmark(safe_softmax_f16x8_pack_f32_per_token,      x_f16, "f16x8packf32(safe)", out_f16)
run_benchmark(partial(torch.softmax, dim=1, out=out_f16), x_f16, "f16_th(per)")
print("-" * 100)

# Run the triton.testing.perf_report benchmark (saves plots)
bench_softmax.run(show_plots=False, print_data=True)
