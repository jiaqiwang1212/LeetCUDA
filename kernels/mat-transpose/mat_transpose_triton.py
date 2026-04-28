"""
Triton implementations of all matrix transpose kernels from mat_transpose.cu.

Operation: out[j, i] = x[i, j]  -- transpose a 2D matrix.

A single tiled 2D kernel handles all variants. The different wrapper names
mirror the naming convention of the CUDA bindings in mat_transpose.py:
  mat_transpose_f32_col2row          - float32 col-major -> row-major view
  mat_transpose_f32_row2col          - float32 row-major -> col-major view
  mat_transpose_f32x4_col2row        - float32 x4 col-major -> row-major view
  mat_transpose_f32x4_row2col        - float32 x4 row-major -> col-major view
  mat_transpose_f32x4_shared_col2row - float32 x4 shared-mem col-major -> row-major
  mat_transpose_f32x4_shared_row2col - float32 x4 shared-mem row-major -> col-major

All wrappers call the same mat_transpose_kernel; tile size is BLOCK_M x BLOCK_N = 32x32.
"""

import time
from functools import partial
from typing import Optional

import torch
import triton
import triton.language as tl

DEVICE = torch.device("cuda:0")

BLOCK_M = 32
BLOCK_N = 32

# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------


@triton.jit
def mat_transpose_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # 2D grid: pid_m along M dimension, pid_n along N dimension
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)

    # Load tile from x: shape [BLOCK_M, BLOCK_N]
    x_ptrs = x_ptr + offsets_m[:, None] * stride_xm + offsets_n[None, :] * stride_xn
    tile = tl.load(x_ptrs, mask=mask, other=0.0)

    # Store transposed: out[n, m] = x[m, n]
    out_ptrs = out_ptr + offsets_n[:, None] * stride_om + offsets_m[None, :] * stride_on
    out_mask = (offsets_n[:, None] < N) & (offsets_m[None, :] < M)
    tl.store(out_ptrs, tl.trans(tile), mask=out_mask)


# ---------------------------------------------------------------------------
# Shared launcher helper
# ---------------------------------------------------------------------------


def _transpose(x: torch.Tensor, out: torch.Tensor):
    M, N = x.shape
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    mat_transpose_kernel[grid](
        x,
        out,
        M,
        N,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )


# ---------------------------------------------------------------------------
# Python wrappers (mirror CUDA binding names from mat_transpose.py)
# ---------------------------------------------------------------------------


def mat_transpose_f32_col2row(x: torch.Tensor, out: torch.Tensor):
    _transpose(x, out)


def mat_transpose_f32_row2col(x: torch.Tensor, out: torch.Tensor):
    _transpose(x, out)


def mat_transpose_f32x4_col2row(x: torch.Tensor, out: torch.Tensor):
    _transpose(x, out)


def mat_transpose_f32x4_row2col(x: torch.Tensor, out: torch.Tensor):
    _transpose(x, out)


def mat_transpose_f32x4_shared_col2row(x: torch.Tensor, out: torch.Tensor):
    _transpose(x, out)


def mat_transpose_f32x4_shared_row2col(x: torch.Tensor, out: torch.Tensor):
    _transpose(x, out)


# ---------------------------------------------------------------------------
# Correctness checks
# ---------------------------------------------------------------------------


def check_correctness():
    torch.manual_seed(42)
    M, N = 512, 256

    x = torch.randn((M, N), device=DEVICE, dtype=torch.float32)
    out = torch.zeros((N, M), device=DEVICE, dtype=torch.float32)

    mat_transpose_f32_col2row(x, out)
    assert torch.allclose(out.T, x), "mat_transpose_f32_col2row mismatch"

    out.zero_()
    mat_transpose_f32_row2col(x, out)
    assert torch.allclose(out.T, x), "mat_transpose_f32_row2col mismatch"

    out.zero_()
    mat_transpose_f32x4_col2row(x, out)
    assert torch.allclose(out.T, x), "mat_transpose_f32x4_col2row mismatch"

    out.zero_()
    mat_transpose_f32x4_row2col(x, out)
    assert torch.allclose(out.T, x), "mat_transpose_f32x4_row2col mismatch"

    out.zero_()
    mat_transpose_f32x4_shared_col2row(x, out)
    assert torch.allclose(out.T, x), "mat_transpose_f32x4_shared_col2row mismatch"

    out.zero_()
    mat_transpose_f32x4_shared_row2col(x, out)
    assert torch.allclose(out.T, x), "mat_transpose_f32x4_shared_row2col mismatch"

    print("All correctness checks passed.")


# ---------------------------------------------------------------------------
# Manual timing benchmark (matches mat_transpose.py format)
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
    if out is not None:
        out.fill_(0)
    # warmup
    if out is not None:
        for _ in range(warmup):
            perf_func(x, out)
    else:
        for _ in range(warmup):
            _ = perf_func(x)
    torch.cuda.synchronize()

    start = time.time()
    if out is not None:
        for _ in range(iters):
            perf_func(x, out)
    else:
        for _ in range(iters):
            out = perf_func(x)
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    real_t = f"{out.T.equal(x)}"
    out_val = out[:2, :2].flatten().detach().cpu().numpy().tolist()[:3]
    out_val = [round(v, 8) for v in out_val]
    print(
        f"{out_info:>35}: {out_val}, validate {real_t:<5}, time:{mean_time:.8f}ms"
    )
    if show_all:
        print(out)
    return out, mean_time


# ---------------------------------------------------------------------------
# triton.testing.perf_report benchmark (bandwidth in GB/s)
# ---------------------------------------------------------------------------


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N"],
        x_vals=[
            (M, N)
            for M in [1024, 2048, 4096, 8192]
            for N in [1024, 2048, 4096, 8192]
        ],
        x_log=False,
        line_arg="kernel",
        line_vals=[
            "f32_col2row",
            "f32_row2col",
            "f32x4_col2row",
            "f32x4_row2col",
            "f32x4_shared_col2row",
            "f32x4_shared_row2col",
        ],
        line_names=[
            "f32_col2row",
            "f32_row2col",
            "f32x4_col2row",
            "f32x4_row2col",
            "f32x4_shared_col2row",
            "f32x4_shared_row2col",
        ],
        styles=[
            ("blue", "-"),
            ("blue", "--"),
            ("red", "-"),
            ("red", "--"),
            ("green", "-"),
            ("green", "--"),
        ],
        ylabel="GB/s",
        plot_name="mat-transpose-performance",
        args={},
    )
)
def bench_mat_transpose(M, N, kernel):
    quantiles = [0.5, 0.2, 0.8]
    x = torch.randn((M, N), device=DEVICE, dtype=torch.float32)
    out = torch.zeros((N, M), device=DEVICE, dtype=torch.float32)

    fn_map = {
        "f32_col2row": lambda: mat_transpose_f32_col2row(x, out),
        "f32_row2col": lambda: mat_transpose_f32_row2col(x, out),
        "f32x4_col2row": lambda: mat_transpose_f32x4_col2row(x, out),
        "f32x4_row2col": lambda: mat_transpose_f32x4_row2col(x, out),
        "f32x4_shared_col2row": lambda: mat_transpose_f32x4_shared_col2row(x, out),
        "f32x4_shared_row2col": lambda: mat_transpose_f32x4_shared_row2col(x, out),
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

    Ms = [1024, 2048, 4096, 8192]
    Ns = [1024, 2048, 4096, 8192]
    MNs = [(M, N) for M in Ms for N in Ns]

    copy_x = lambda x: x.clone()

    for M, N in MNs:
        print("-" * 130)
        print(" " * 55 + f"M={M}, N={N}")
        x = torch.arange(0, M * N).reshape(M, N).cuda().float().contiguous()
        y = torch.zeros((N, M), device=DEVICE, dtype=torch.float32).contiguous()
        run_benchmark(partial(copy_x), x, "original")
        run_benchmark(mat_transpose_f32_col2row, x, "f32_col2row", y)
        run_benchmark(mat_transpose_f32_row2col, x, "f32_row2col", y)
        run_benchmark(mat_transpose_f32x4_col2row, x, "f32x4_col2row", y)
        run_benchmark(mat_transpose_f32x4_row2col, x, "f32x4_row2col", y)
        run_benchmark(mat_transpose_f32x4_shared_col2row, x, "f32x4_shared_col2row", y)
        run_benchmark(mat_transpose_f32x4_shared_row2col, x, "f32x4_shared_row2col", y)
        run_benchmark(
            partial(torch.transpose_copy, dim0=0, dim1=1, out=y), x, "f32_th"
        )
        print("-" * 130)

    bench_mat_transpose.run(show_plots=False, print_data=True)
