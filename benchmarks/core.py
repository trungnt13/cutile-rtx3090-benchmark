#!/usr/bin/env python3

"""Shared benchmark primitives for the cuTile public benchmark repo."""

from __future__ import annotations

import argparse
import time
from collections import namedtuple
from pathlib import Path
from textwrap import dedent

import cupy as cp
import cuda.tile as ct
import numpy as np
import torch
import triton
import triton.language as tl
from cuda.tile._compile import compile_tile, default_tile_context
from cuda.tile._compiler_options import CompilerOptions


TimingResult = namedtuple("TimingResult", ["mean", "std"])

# RTX 3090 peak theoretical throughput by dtype (TFLOP/s or TOP/s).
PEAK_TFLOPS = {"float16": 142.0, "bfloat16": 142.0, "float32": 35.6, "int8": 284.0}

# Rectangular GEMM shapes representative of MLP and attention-like workloads.
RECTANGULAR_SHAPES = [
    (4096, 4096, 128),
    (4096, 4096, 256),
    (4096, 4096, 512),
    (512, 64, 512),
    (1024, 64, 1024),
    (2048, 64, 2048),
]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FULL_ARTIFACTS_DIR = ARTIFACTS_DIR / "full"
FP16_ARTIFACTS_DIR = ARTIFACTS_DIR / "fp16_focus"
PTX_ITERATION_ARTIFACTS_DIR = ARTIFACTS_DIR / "ptx_iterations"
NSYS_ARTIFACTS_DIR = ARTIFACTS_DIR / "nsys"
SYSTEM_ARTIFACTS_DIR = ARTIFACTS_DIR / "system"
INVESTIGATIONS_DIR = PROJECT_ROOT / "investigations"
INT8_IR_DIR = INVESTIGATIONS_DIR / "int8_ir"


# This PTX/CUDA source is intentionally modest. It gives cuTile and Triton a readable
# low-level baseline, but it is not meant to rival library-grade tensor-core kernels.
PTX_INLINE_KERNEL_SOURCE = dedent(
    r"""
    #include <cuda_fp16.h>
    #include <cuda_bf16.h>

    __device__ __forceinline__ float fma_ptx(float a, float b, float c) {
        asm volatile ("fma.rn.f32 %0, %1, %2, %3;" : "=f"(c) : "f"(a), "f"(b), "f"(c));
        return c;
    }

    __device__ __forceinline__ float to_float(float x) { return x; }
    __device__ __forceinline__ float to_float(__half x) { return __half2float(x); }
    __device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }
    __device__ __forceinline__ int to_int(signed char x) { return static_cast<int>(x); }

    template <typename T>
    __device__ __forceinline__ void matmul_tiled_fp_impl(
        const T* __restrict__ A,
        const T* __restrict__ B,
        float* __restrict__ C,
        int M,
        int N,
        int K
    ) {
        constexpr int TM = 16;
        constexpr int TN = 16;
        constexpr int TK = 16;

        __shared__ float As[TM][TK];
        __shared__ float Bs[TK][TN];

        int row = blockIdx.y * TM + threadIdx.y;
        int col = blockIdx.x * TN + threadIdx.x;
        float acc = 0.0f;

        for (int k0 = 0; k0 < K; k0 += TK) {
            int a_col = k0 + threadIdx.x;
            int b_row = k0 + threadIdx.y;

            As[threadIdx.y][threadIdx.x] =
                (row < M && a_col < K) ? to_float(A[row * K + a_col]) : 0.0f;
            Bs[threadIdx.y][threadIdx.x] =
                (b_row < K && col < N) ? to_float(B[b_row * N + col]) : 0.0f;

            __syncthreads();

            #pragma unroll
            for (int kk = 0; kk < TK; ++kk) {
                acc = fma_ptx(As[threadIdx.y][kk], Bs[kk][threadIdx.x], acc);
            }

            __syncthreads();
        }

        if (row < M && col < N) {
            C[row * N + col] = acc;
        }
    }

    __device__ __forceinline__ void matmul_tiled_i8_impl(
        const signed char* __restrict__ A,
        const signed char* __restrict__ B,
        int* __restrict__ C,
        int M,
        int N,
        int K
    ) {
        constexpr int TM = 16;
        constexpr int TN = 16;
        constexpr int TK = 16;

        __shared__ int As[TM][TK];
        __shared__ int Bs[TK][TN];

        int row = blockIdx.y * TM + threadIdx.y;
        int col = blockIdx.x * TN + threadIdx.x;
        int acc = 0;

        for (int k0 = 0; k0 < K; k0 += TK) {
            int a_col = k0 + threadIdx.x;
            int b_row = k0 + threadIdx.y;

            As[threadIdx.y][threadIdx.x] =
                (row < M && a_col < K) ? to_int(A[row * K + a_col]) : 0;
            Bs[threadIdx.y][threadIdx.x] =
                (b_row < K && col < N) ? to_int(B[b_row * N + col]) : 0;

            __syncthreads();

            #pragma unroll
            for (int kk = 0; kk < TK; ++kk) {
                acc += As[threadIdx.y][kk] * Bs[kk][threadIdx.x];
            }

            __syncthreads();
        }

        if (row < M && col < N) {
            C[row * N + col] = acc;
        }
    }

    extern "C" __global__
    void matmul_tiled_f32(
        const float* __restrict__ A,
        const float* __restrict__ B,
        float* __restrict__ C,
        int M,
        int N,
        int K
    ) {
        matmul_tiled_fp_impl<float>(A, B, C, M, N, K);
    }

    extern "C" __global__
    void matmul_tiled_f16(
        const __half* __restrict__ A,
        const __half* __restrict__ B,
        float* __restrict__ C,
        int M,
        int N,
        int K
    ) {
        matmul_tiled_fp_impl<__half>(A, B, C, M, N, K);
    }

    extern "C" __global__
    void matmul_tiled_bf16(
        const __nv_bfloat16* __restrict__ A,
        const __nv_bfloat16* __restrict__ B,
        float* __restrict__ C,
        int M,
        int N,
        int K
    ) {
        matmul_tiled_fp_impl<__nv_bfloat16>(A, B, C, M, N, K);
    }

    extern "C" __global__
    void matmul_tiled_i8(
        const signed char* __restrict__ A,
        const signed char* __restrict__ B,
        int* __restrict__ C,
        int M,
        int N,
        int K
    ) {
        matmul_tiled_i8_impl(A, B, C, M, N, K);
    }
    """
)


# Dtype metadata is central to both correctness and presentation. In particular, int8
# uses an int32 output because exact GEMM semantics accumulate into int32 even when the
# current cuTile path later exposes wrapped behavior in the investigative IR dump.
DTYPE_INFO = {
    "float32": {
        "torch": torch.float32,
        "out": torch.float32,
        "ptx_kernel": "matmul_tiled_f32",
        "label": "float32",
        "tensor_core_friendly": False,
        "supported": True,
    },
    "float16": {
        "torch": torch.float16,
        "out": torch.float32,
        "ptx_kernel": "matmul_tiled_f16",
        "label": "float16",
        "tensor_core_friendly": True,
        "supported": True,
    },
    "bfloat16": {
        "torch": torch.bfloat16,
        "out": torch.float32,
        "ptx_kernel": "matmul_tiled_bf16",
        "label": "bfloat16",
        "tensor_core_friendly": True,
        "supported": True,
    },
    "int8": {
        "torch": torch.int8,
        "out": torch.int32,
        "ptx_kernel": "matmul_tiled_i8",
        "label": "int8",
        "tensor_core_friendly": True,
        "supported": True,
    },
    "fp8": {
        "label": "fp8",
        "supported": False,
        "reason": (
            "FP8 Tensor Core matmul is not a meaningful path on RTX 3090 / cc 8.6. "
            "Triton's official FP8 matmul tutorial targets cc >= 9.0."
        ),
    },
    "int4": {
        "label": "int4",
        "supported": False,
        "reason": (
            "INT4 matmul is not wired as a simple first-class path in the current "
            "cuTile/Triton/PTX benchmark on this RTX 3090."
        ),
    },
}


@triton.jit
def triton_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m,
    n,
    k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    out_dtype: tl.constexpr,
):
    """Triton reference kernel used as the hand-written kernel baseline."""

    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(n, block_n)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_k = tl.arange(0, block_k)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((block_m, block_n), dtype=out_dtype)
    for _ in range(0, tl.cdiv(k, block_k)):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < m) & (offs_k[None, :] < k), other=0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k) & (offs_n[None, :] < n), other=0)
        acc = tl.dot(a, b, acc, out_dtype=out_dtype)
        a_ptrs += block_k * stride_ak
        b_ptrs += block_k * stride_bk
        offs_k += block_k

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


def check_dtype_supported(dtype_name: str) -> None:
    """Raise a user-facing error when the requested dtype is intentionally unsupported."""

    info = DTYPE_INFO[dtype_name]
    if info.get("supported", False):
        return
    raise RuntimeError(info["reason"])


def make_inputs(dtype_name: str, m: int, n: int, k: int, seed: int = 0):
    """Create deterministic benchmark inputs for the requested dtype and shape."""

    info = DTYPE_INFO[dtype_name]
    dtype = info["torch"]
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    if dtype_name == "int8":
        a = torch.randint(-8, 8, (m, k), device="cuda", dtype=torch.int8)
        b = torch.randint(-8, 8, (k, n), device="cuda", dtype=torch.int8)
    else:
        a = torch.randn((m, k), device="cuda", dtype=torch.float32).to(dtype)
        b = torch.randn((k, n), device="cuda", dtype=torch.float32).to(dtype)
    return a, b


def compile_ptx(dtype_name: str) -> cp.RawKernel:
    """Compile the fixed PTX-inline baseline for the requested dtype."""

    module = cp.RawModule(code=PTX_INLINE_KERNEL_SOURCE, options=("-std=c++14",))
    return module.get_function(DTYPE_INFO[dtype_name]["ptx_kernel"])


def make_cutile_kernel(dtype_name: str, num_ctas: int | None, occupancy: int | None):
    """Build a cuTile kernel specialized for the chosen dtype and launch policy.

    The int8 path is kept separate because its accumulation semantics are the main
    correctness caveat in this repo. The float paths always accumulate into float32.
    """

    kernel_kwargs = {}
    if num_ctas is not None:
        kernel_kwargs["num_ctas"] = num_ctas
    if occupancy is not None:
        kernel_kwargs["occupancy"] = occupancy

    if dtype_name == "int8":

        @ct.kernel(**kernel_kwargs)
        def kernel(
            a,
            b,
            c,
            tile_m: ct.Constant[int],
            tile_n: ct.Constant[int],
            tile_k: ct.Constant[int],
            k_tiles: ct.Constant[int],
        ):
            bid_m = ct.bid(0)
            bid_n = ct.bid(1)
            acc = ct.zeros((tile_m, tile_n), dtype=ct.int32)
            for kk in range(k_tiles):
                a_tile = ct.load(a, index=(bid_m, kk), shape=(tile_m, tile_k))
                b_tile = ct.load(b, index=(kk, bid_n), shape=(tile_k, tile_n))
                acc = acc + ct.matmul(a_tile, b_tile)
            ct.store(c, index=(bid_m, bid_n), tile=acc)

    else:

        @ct.kernel(**kernel_kwargs)
        def kernel(
            a,
            b,
            c,
            tile_m: ct.Constant[int],
            tile_n: ct.Constant[int],
            tile_k: ct.Constant[int],
            k_tiles: ct.Constant[int],
        ):
            bid_m = ct.bid(0)
            bid_n = ct.bid(1)
            acc = ct.zeros((tile_m, tile_n), dtype=ct.float32)
            for kk in range(k_tiles):
                a_tile = ct.load(a, index=(bid_m, kk), shape=(tile_m, tile_k))
                b_tile = ct.load(b, index=(kk, bid_n), shape=(tile_k, tile_n))
                acc = acc + ct.matmul(a_tile, b_tile)
            ct.store(c, index=(bid_m, bid_n), tile=acc)

    return kernel


def cutile_compiler_options(num_ctas: int | None, occupancy: int | None) -> CompilerOptions:
    """Create compiler options for explicit cuTile occupancy experiments."""

    return CompilerOptions(num_ctas=num_ctas, occupancy=occupancy)


def compile_cutile_kernel(kernel, args, num_ctas: int | None, occupancy: int | None):
    """Compile a cuTile kernel without including the launch in the measured timing."""

    return compile_tile(
        kernel._pyfunc,
        args,
        cutile_compiler_options(num_ctas, occupancy),
        default_tile_context,
    )


def pct_of_peak(dtype_name: str, achieved_tflops: float) -> float:
    """Return achieved throughput as a percentage of the RTX 3090 theoretical peak."""

    return (achieved_tflops / PEAK_TFLOPS.get(dtype_name, 35.6)) * 100.0


def benchmark_ms_cupy(fn, warmup: int, iters: int) -> TimingResult:
    """Measure steady-state GPU time with CuPy events.

    Warmup happens before recording so JIT, cache fill, and lazy init costs are kept
    out of the steady-state number. Returns (mean, std) over per-iteration measurements.
    """

    for _ in range(warmup):
        fn()
    cp.cuda.get_current_stream().synchronize()

    times = []
    for _ in range(iters):
        start = cp.cuda.Event()
        stop = cp.cuda.Event()
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, stop))
    return TimingResult(float(np.mean(times)), float(np.std(times)))


def benchmark_ms_torch(fn, warmup: int, iters: int) -> TimingResult:
    """Measure steady-state GPU time with Torch events.

    Triton integrates cleanly with Torch's event/timing stack, so this path mirrors the
    same warmup policy but uses Torch-owned events instead of CuPy-owned ones.
    Returns (mean, std) over per-iteration measurements.
    """

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        times.append(start.elapsed_time(stop))
    return TimingResult(float(np.mean(times)), float(np.std(times)))


def sync_all() -> None:
    """Synchronize both CuPy and Torch views of the current CUDA stream."""

    cp.cuda.get_current_stream().synchronize()
    torch.cuda.synchronize()


def measure_host_ms(fn) -> float:
    """Measure host-visible wall time for compile or first-launch phases."""

    sync_all()
    start = time.perf_counter()
    fn()
    sync_all()
    return (time.perf_counter() - start) * 1.0e3


def tflops(m: int, n: int, k: int, ms: float) -> float:
    """Convert GEMM latency into TFLOP/s."""

    return (2.0 * m * n * k) / (ms * 1.0e-3) / 1.0e12


def tops(m: int, n: int, k: int, ms: float) -> float:
    """Convert integer GEMM latency into TOP/s."""

    return (2.0 * m * n * k) / (ms * 1.0e-3) / 1.0e12


def metric(dtype_name: str, m: int, n: int, k: int, ms: float) -> str:
    """Format the appropriate throughput unit for the requested dtype."""

    if dtype_name == "int8":
        return f"{tops(m, n, k, ms):.3f} TOP/s"
    return f"{tflops(m, n, k, ms):.3f} TFLOP/s"


def run_cutile(kernel, a, b, c, m: int, n: int, k: int, tile_m: int, tile_n: int, tile_k: int) -> None:
    """Launch a cuTile kernel using tile-sized grid decomposition."""

    grid = (ct.cdiv(m, tile_m), ct.cdiv(n, tile_n), 1)
    k_tiles = ct.cdiv(k, tile_k)
    ct.launch(cp.cuda.get_current_stream(), grid, kernel, (a, b, c, tile_m, tile_n, tile_k, k_tiles))


def run_ptx(kernel, a, b, c, m: int, n: int, k: int) -> None:
    """Launch the fixed 16x16 PTX baseline."""

    block = (16, 16, 1)
    grid = ((n + block[0] - 1) // block[0], (m + block[1] - 1) // block[1], 1)
    kernel(grid, block, (a.data_ptr(), b.data_ptr(), c.data_ptr(), np.int32(m), np.int32(n), np.int32(k)))


def run_triton(a, b, c, m: int, n: int, k: int, tile_m: int, tile_n: int, tile_k: int, out_dtype) -> None:
    """Launch the Triton reference kernel for a single tile configuration."""

    grid = (triton.cdiv(m, tile_m) * triton.cdiv(n, tile_n),)
    triton_matmul_kernel[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        block_m=tile_m,
        block_n=tile_n,
        block_k=tile_k,
        out_dtype=out_dtype,
    )


def run_torch(a, b, c, dtype_name: str) -> None:
    """Run the Torch baseline using the backend that matches the dtype semantics."""

    if dtype_name == "int8":
        c.copy_(torch._int_mm(a, b))
    else:
        torch.mm(a, b, out=c)


def run_cublas(a, b, c, dtype_name: str) -> None:
    """Run a direct cuBLAS GEMM via CuPy, using cupy.matmul on DLPack-transferred arrays.

    This avoids Torch's dispatcher overhead while still going through cuBLAS internally
    (CuPy dispatches to cuBLAS for supported types). Pre-conversion happens outside the
    timing loop so DLPack overhead is excluded from measurement.
    """

    if dtype_name == "int8":
        # cuBLAS int8 GEMM is not wired here; fall back to Torch path.
        run_torch(a, b, c, dtype_name)
        return
    # Convert torch tensors to cupy arrays via DLPack (zero-copy on same device).
    a_cp = cp.from_dlpack(a)
    b_cp = cp.from_dlpack(b)
    c_cp = cp.from_dlpack(c)
    # CuPy matmul dispatches to cuBLAS for float32/float16/bfloat16.
    # Use out= to write directly into the pre-allocated output.
    if c.dtype == torch.float32 and a.dtype != torch.float32:
        # Half inputs with float32 output: cast in cupy then matmul.
        result = cp.matmul(a_cp.astype(cp.float32), b_cp.astype(cp.float32))
        c_cp[:] = result
    else:
        cp.matmul(a_cp, b_cp, out=c_cp)


def make_reference(a, b, dtype_name: str) -> torch.Tensor:
    """Build the correctness reference tensor for a benchmark case.

    TF32 is disabled for float references so comparison is against the full-precision
    accumulator path rather than Ampere's faster but numerically different default.
    """

    if dtype_name == "int8":
        return torch._int_mm(a, b)

    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return torch.mm(a.float(), b.float())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def make_chunk_wrapped_i8_reference(a, b, tile_k: int) -> torch.Tensor:
    """Model the wrapped-per-tile int8 behavior observed in the cuTile IR investigation."""

    total = torch.zeros((a.shape[0], b.shape[1]), device="cuda", dtype=torch.int32)
    for k0 in range(0, a.shape[1], tile_k):
        partial = torch._int_mm(a[:, k0 : k0 + tile_k].contiguous(), b[k0 : k0 + tile_k, :].contiguous())
        total += partial.to(torch.int8).to(torch.int32)
    return total


def triton_out_dtype(dtype_name: str):
    """Return the Triton accumulator dtype for the requested benchmark dtype."""

    return tl.int32 if dtype_name == "int8" else tl.float32


def benchmark_tile_config(
    dtype_name: str,
    a,
    b,
    reference,
    wrapped_reference,
    ptx_kernel,
    tile_config: tuple[int, int, int],
    warmup: int,
    iters: int,
    num_ctas: int | None,
    occupancy: int | None,
):
    """Benchmark one shared tile configuration across cuTile, PTX-inline, and Triton.

    Errors are computed after a priming launch and before the timed steady-state loop so
    correctness checks are not polluted by lazily initialized outputs or first-launch cost.
    PTX remains fixed at 16x16 because the inline baseline is intentionally not a generic
    tile-sweep kernel; that tradeoff is documented in the public benchmark narrative.
    """

    tile_m, tile_n, tile_k = tile_config
    out_dtype = DTYPE_INFO[dtype_name]["out"]
    c_cutile = torch.empty((a.shape[0], b.shape[1]), device="cuda", dtype=out_dtype)
    c_triton = torch.empty_like(c_cutile)
    c_ptx = torch.empty_like(c_cutile)

    cutile_kernel = make_cutile_kernel(dtype_name, num_ctas, occupancy)
    run_cutile(cutile_kernel, a, b, c_cutile, a.shape[0], b.shape[1], a.shape[1], tile_m, tile_n, tile_k)
    run_ptx(ptx_kernel, a, b, c_ptx, a.shape[0], b.shape[1], a.shape[1])
    run_triton(a, b, c_triton, a.shape[0], b.shape[1], a.shape[1], tile_m, tile_n, tile_k, triton_out_dtype(dtype_name))
    sync_all()

    cutile_max_err = float((c_cutile - reference).abs().max().item())
    ptx_max_err = float((c_ptx - reference).abs().max().item())
    triton_max_err = float((c_triton - reference).abs().max().item())
    cutile_wrapped_err = float((c_cutile - wrapped_reference).abs().max().item()) if wrapped_reference is not None else None
    cutile_chunk_wrapped_err = (
        float((c_cutile - make_chunk_wrapped_i8_reference(a, b, tile_k)).abs().max().item())
        if dtype_name == "int8"
        else None
    )

    cutile_timing = benchmark_ms_cupy(
        lambda: run_cutile(cutile_kernel, a, b, c_cutile, a.shape[0], b.shape[1], a.shape[1], tile_m, tile_n, tile_k),
        warmup,
        iters,
    )
    ptx_timing = benchmark_ms_cupy(
        lambda: run_ptx(ptx_kernel, a, b, c_ptx, a.shape[0], b.shape[1], a.shape[1]),
        warmup,
        iters,
    )
    triton_timing = benchmark_ms_torch(
        lambda: run_triton(a, b, c_triton, a.shape[0], b.shape[1], a.shape[1], tile_m, tile_n, tile_k, triton_out_dtype(dtype_name)),
        warmup,
        iters,
    )

    return {
        "tile": tile_config,
        "cutile_ms": cutile_timing.mean,
        "cutile_std": cutile_timing.std,
        "ptx_ms": ptx_timing.mean,
        "ptx_std": ptx_timing.std,
        "triton_ms": triton_timing.mean,
        "triton_std": triton_timing.std,
        "cutile_err": cutile_max_err,
        "cutile_wrapped_err": cutile_wrapped_err,
        "cutile_chunk_wrapped_err": cutile_chunk_wrapped_err,
        "ptx_err": ptx_max_err,
        "triton_err": triton_max_err,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the public benchmark sweep entrypoint."""

    parser = argparse.ArgumentParser(
        description="Benchmark cuTile, PTX-inline, Triton, and Torch matmul on CUDA."
    )
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--sizes", type=str, default="")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--dtype", choices=tuple(DTYPE_INFO), default="float16")
    parser.add_argument("--tile-m", type=int, default=32)
    parser.add_argument("--tile-n", type=int, default=32)
    parser.add_argument("--tile-k", type=int, default=32)
    parser.add_argument(
        "--tile-sweep",
        type=str,
        default="16,16,16;32,32,16;32,32,32;64,64,32",
    )
    parser.add_argument("--num-ctas", type=int, default=None)
    parser.add_argument("--occupancy", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def parse_tile_configs(args: argparse.Namespace) -> list[tuple[int, int, int]]:
    """Parse semicolon-delimited tile specs into a tile sweep."""

    configs = []
    if not args.tile_sweep:
        return [(args.tile_m, args.tile_n, args.tile_k)]

    for spec in args.tile_sweep.split(";"):
        parts = [int(x.strip()) for x in spec.split(",") if x.strip()]
        if len(parts) != 3:
            raise ValueError(f"Invalid tile spec '{spec}'. Use m,n,k;...")
        configs.append(tuple(parts))
    return configs


def parse_sizes(args: argparse.Namespace) -> list[tuple[int, int, int]]:
    """Parse the sweep shape list, defaulting to the public report sizes."""

    if not args.sizes:
        return [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024)]

    sizes = []
    for spec in args.sizes.split(";"):
        parts = [int(x.strip()) for x in spec.split(",") if x.strip()]
        if len(parts) == 1:
            sizes.append((parts[0], parts[0], parts[0]))
        elif len(parts) == 3:
            sizes.append(tuple(parts))
        else:
            raise ValueError(f"Invalid size spec '{spec}'. Use s or m,n,k;...")
    return sizes


def main() -> None:
    """Run the benchmark sweep and print a human-readable summary to stdout."""

    args = parse_args()
    check_dtype_supported(args.dtype)

    if any(t <= 0 for t in (args.tile_m, args.tile_n, args.tile_k)):
        raise ValueError("Tile sizes must be positive.")

    info = DTYPE_INFO[args.dtype]
    dtype = info["torch"]
    tile_configs = parse_tile_configs(args)
    sizes = parse_sizes(args)
    device_props = torch.cuda.get_device_properties(0)
    device_name = torch.cuda.get_device_name(0)
    tensor_core_note = (
        "Tensor Core friendly on Ampere" if info["tensor_core_friendly"] else "scalar/baseline dtype"
    )

    print(f"device: {device_name} (cc {device_props.major}.{device_props.minor})")
    print(f"dtype: {info['label']} ({tensor_core_note})")
    if args.dtype == "int8":
        print(
            "note: cuTile i8 matmul does not currently preserve exact int32 GEMM semantics; "
            "see investigations/int8_ir for the narrowed-then-widened accumulator path"
        )
    print(f"cuTile config: num_ctas={args.num_ctas} occupancy={args.occupancy}")
    print(f"sizes: {', '.join(f'{m}x{n}x{k}' for m, n, k in sizes)}")
    print(f"tiles: {', '.join(f'({tm},{tn},{tk})' for tm, tn, tk in tile_configs)}")

    ptx_kernel = compile_ptx(args.dtype)

    for m, n, k in sizes:
        a, b = make_inputs(args.dtype, m, n, k, seed=args.seed)
        reference = make_reference(a, b, args.dtype)
        wrapped_reference = reference.to(torch.int8).to(torch.int32) if args.dtype == "int8" else None
        torch_baseline = torch.empty(
            (m, n),
            device="cuda",
            dtype=(dtype if args.dtype != "int8" else torch.int32),
        )
        run_torch(a, b, torch_baseline, args.dtype)
        torch.cuda.synchronize()
        torch_timing = benchmark_ms_torch(
            lambda: run_torch(a, b, torch_baseline, args.dtype),
            args.warmup,
            args.iters,
        )
        torch_ms = torch_timing.mean

        print(f"shape: M={m} N={n} K={k}")
        print(f"Torch:  {torch_ms:.3f} ms (±{torch_timing.std:.3f})  {metric(args.dtype, m, n, k, torch_ms)}  baseline")

        for tile_config in tile_configs:
            result = benchmark_tile_config(
                args.dtype,
                a,
                b,
                reference,
                wrapped_reference,
                ptx_kernel,
                tile_config,
                args.warmup,
                args.iters,
                args.num_ctas,
                args.occupancy,
            )
            tile_m, tile_n, tile_k = result["tile"]
            print(f"tile:   cuTile/Triton=({tile_m}, {tile_n}, {tile_k}) PTX-inline=(16, 16, 16)")
            if args.dtype == "int8":
                print(
                    f"cuTile: {result['cutile_ms']:.3f} ms  {metric(args.dtype, m, n, k, result['cutile_ms'])}  "
                    f"max_err_exact={result['cutile_err']:.3e}  "
                    f"max_err_wrapped_i8={result['cutile_wrapped_err']:.3e}  "
                    f"max_err_tile_wrapped_i8={result['cutile_chunk_wrapped_err']:.3e}"
                )
            else:
                print(
                    f"cuTile: {result['cutile_ms']:.3f} ms  "
                    f"{metric(args.dtype, m, n, k, result['cutile_ms'])}  max_err={result['cutile_err']:.3e}"
                )
            print(
                f"PTX:    {result['ptx_ms']:.3f} ms  "
                f"{metric(args.dtype, m, n, k, result['ptx_ms'])}  max_err={result['ptx_err']:.3e}"
            )
            print(
                f"Triton: {result['triton_ms']:.3f} ms  "
                f"{metric(args.dtype, m, n, k, result['triton_ms'])}  max_err={result['triton_err']:.3e}"
            )
