"""
Triton implementations of all embedding kernels from embedding.cu.

Operation: out[i] = weight[indices[i]]  -- table lookup, one row per token.

Variants and their CUDA analogues:
  embedding_f32         - FP32 weight, one program per token row
  embedding_f32x4       - FP32 weight, same kernel (vectorization transparent in Triton)
  embedding_f32x4_pack  - FP32 weight, same kernel (packed load variant)
  embedding_f16         - FP16 weight, one program per token row
  embedding_f16x8       - FP16 weight, same kernel (vectorization transparent in Triton)
  embedding_f16x8_pack  - FP16 weight, same kernel (packed load variant)
"""

import time
from typing import Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

DEVICE = torch.device("cuda:0")

# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------


@triton.jit
def embedding_f32_kernel(
    indices_ptr,
    weight_ptr,
    out_ptr,
    N,
    K,
    BLOCK_K: tl.constexpr,
):
    # one program per token (row)
    i = tl.program_id(0)
    idx = tl.load(indices_ptr + i)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < K
    row = tl.load(weight_ptr + idx * K + offsets, mask=mask)
    tl.store(out_ptr + i * K + offsets, row, mask=mask)


@triton.jit
def embedding_f16_kernel(
    indices_ptr,
    weight_ptr,
    out_ptr,
    N,
    K,
    BLOCK_K: tl.constexpr,
):
    # one program per token (row)
    i = tl.program_id(0)
    idx = tl.load(indices_ptr + i)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < K
    row = tl.load(weight_ptr + idx * K + offsets, mask=mask)
    tl.store(out_ptr + i * K + offsets, row, mask=mask)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------


def embedding_f32(indices: torch.Tensor, weight: torch.Tensor, out: torch.Tensor):
    N = indices.shape[0]
    K = weight.shape[1]
    BLOCK_K = triton.next_power_of_2(K)
    grid = (N,)
    embedding_f32_kernel[grid](indices, weight, out, N, K, BLOCK_K=BLOCK_K)


def embedding_f32x4(indices: torch.Tensor, weight: torch.Tensor, out: torch.Tensor):
    N = indices.shape[0]
    K = weight.shape[1]
    BLOCK_K = triton.next_power_of_2(K)
    grid = (N,)
    embedding_f32_kernel[grid](indices, weight, out, N, K, BLOCK_K=BLOCK_K)


def embedding_f32x4_pack(indices: torch.Tensor, weight: torch.Tensor, out: torch.Tensor):
    N = indices.shape[0]
    K = weight.shape[1]
    BLOCK_K = triton.next_power_of_2(K)
    grid = (N,)
    embedding_f32_kernel[grid](indices, weight, out, N, K, BLOCK_K=BLOCK_K)


def embedding_f16(indices: torch.Tensor, weight: torch.Tensor, out: torch.Tensor):
    N = indices.shape[0]
    K = weight.shape[1]
    BLOCK_K = triton.next_power_of_2(K)
    grid = (N,)
    embedding_f16_kernel[grid](indices, weight, out, N, K, BLOCK_K=BLOCK_K)


def embedding_f16x8(indices: torch.Tensor, weight: torch.Tensor, out: torch.Tensor):
    N = indices.shape[0]
    K = weight.shape[1]
    BLOCK_K = triton.next_power_of_2(K)
    grid = (N,)
    embedding_f16_kernel[grid](indices, weight, out, N, K, BLOCK_K=BLOCK_K)


def embedding_f16x8_pack(indices: torch.Tensor, weight: torch.Tensor, out: torch.Tensor):
    N = indices.shape[0]
    K = weight.shape[1]
    BLOCK_K = triton.next_power_of_2(K)
    grid = (N,)
    embedding_f16_kernel[grid](indices, weight, out, N, K, BLOCK_K=BLOCK_K)


# ---------------------------------------------------------------------------
# Correctness checks
# ---------------------------------------------------------------------------


def check_correctness():
    torch.manual_seed(42)
    M, N, K = 1024, 256, 128

    indices = torch.randint(0, M, size=(N,), device=DEVICE, dtype=torch.int32)
    weight_f32 = torch.randn((M, K), device=DEVICE, dtype=torch.float32)
    ref_f32 = F.embedding(indices.long(), weight_f32)

    out = torch.zeros((N, K), device=DEVICE, dtype=torch.float32)
    embedding_f32(indices, weight_f32, out)
    assert torch.allclose(out, ref_f32), "embedding_f32 mismatch"

    out.zero_()
    embedding_f32x4(indices, weight_f32, out)
    assert torch.allclose(out, ref_f32), "embedding_f32x4 mismatch"

    out.zero_()
    embedding_f32x4_pack(indices, weight_f32, out)
    assert torch.allclose(out, ref_f32), "embedding_f32x4_pack mismatch"

    weight_f16 = weight_f32.half()
    ref_f16 = F.embedding(indices.long(), weight_f16)
    out_f16 = torch.zeros((N, K), device=DEVICE, dtype=torch.float16)

    embedding_f16(indices, weight_f16, out_f16)
    assert torch.allclose(out_f16, ref_f16), "embedding_f16 mismatch"

    out_f16.zero_()
    embedding_f16x8(indices, weight_f16, out_f16)
    assert torch.allclose(out_f16, ref_f16), "embedding_f16x8 mismatch"

    out_f16.zero_()
    embedding_f16x8_pack(indices, weight_f16, out_f16)
    assert torch.allclose(out_f16, ref_f16), "embedding_f16x8_pack mismatch"

    print("All correctness checks passed.")


# ---------------------------------------------------------------------------
# Manual timing benchmark (matches embedding.py format)
# ---------------------------------------------------------------------------


def run_benchmark(
    perf_func: callable,
    a: torch.Tensor,
    b: torch.Tensor,
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
    out_val = out.flatten().detach().cpu().numpy().tolist()[:3]
    out_val = [round(v, 8) for v in out_val]
    out_val = [f"{v:<12}" for v in out_val]
    print(f"{out_info:>23}: {out_val}, time:{mean_time:.6f}ms")
    if show_all:
        print(out)
    return out.clone(), mean_time


# ---------------------------------------------------------------------------
# triton.testing.perf_report benchmark (bandwidth in GB/s)
# ---------------------------------------------------------------------------


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N", "K"],
        x_vals=[(2048, 512), (2048, 1024), (4096, 512), (4096, 1024)],
        x_log=False,
        line_arg="kernel",
        line_vals=[
            "f32",
            "f32x4",
            "f32x4_pack",
            "f16",
            "f16x8",
            "f16x8_pack",
        ],
        line_names=[
            "f32",
            "f32x4",
            "f32x4_pack",
            "f16",
            "f16x8",
            "f16x8_pack",
        ],
        styles=[
            ("blue", "-"),
            ("blue", "--"),
            ("blue", ":"),
            ("red", "-"),
            ("red", "--"),
            ("red", ":"),
        ],
        ylabel="GB/s",
        plot_name="embedding-performance",
        args={"M": 4096},
    )
)
def bench_embedding(M, N, K, kernel):
    quantiles = [0.5, 0.2, 0.8]
    indices = torch.randint(0, M, size=(N,), device=DEVICE, dtype=torch.int32)

    if kernel in ("f32", "f32x4", "f32x4_pack"):
        weight = torch.randn((M, K), device=DEVICE, dtype=torch.float32)
        out = torch.zeros((N, K), device=DEVICE, dtype=torch.float32)
        elem_bytes = 4
    else:
        weight = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
        out = torch.zeros((N, K), device=DEVICE, dtype=torch.float16)
        elem_bytes = 2

    fn_map = {
        "f32": lambda: embedding_f32(indices, weight, out),
        "f32x4": lambda: embedding_f32x4(indices, weight, out),
        "f32x4_pack": lambda: embedding_f32x4_pack(indices, weight, out),
        "f16": lambda: embedding_f16(indices, weight, out),
        "f16x8": lambda: embedding_f16x8(indices, weight, out),
        "f16x8_pack": lambda: embedding_f16x8_pack(indices, weight, out),
    }
    fn = fn_map[kernel]

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    # 1 read (N rows of K elements) + 1 write (N rows of K elements)
    gbps = lambda ms: 2 * N * K * elem_bytes * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    check_correctness()

    Ms = [1024, 4096]
    Ns = [2048, 4096]
    Ks = [512, 1024]
    MNKs = [(M, N, K) for M in Ms for N in Ns for K in Ks]

    for M, N, K in MNKs:
        print("-" * 110)
        print(" " * 45 + f"MaxV={M}, SeqLen={N}, EmbSize={K}")
        i = torch.randint(0, M, size=(N,), device=DEVICE, dtype=torch.int32)
        weight = torch.randn((M, K), device=DEVICE, dtype=torch.float32).contiguous()
        o = torch.zeros((N, K), device=DEVICE, dtype=torch.float32).contiguous()

        run_benchmark(embedding_f32, i, weight, "f32", o)
        run_benchmark(embedding_f32x4, i, weight, "f32x4", o)
        run_benchmark(embedding_f32x4_pack, i, weight, "f32x4_pack", o)

        print("-" * 110)
        weight_f16 = torch.randn((M, K), device=DEVICE, dtype=torch.float16).contiguous()
        o_f16 = torch.zeros((N, K), device=DEVICE, dtype=torch.float16).contiguous()
        run_benchmark(embedding_f16, i, weight_f16, "f16", o_f16)
        run_benchmark(embedding_f16x8, i, weight_f16, "f16x8", o_f16)
        run_benchmark(embedding_f16x8_pack, i, weight_f16, "f16x8_pack", o_f16)
        print("-" * 110)

    bench_embedding.run(show_plots=False, print_data=True)
