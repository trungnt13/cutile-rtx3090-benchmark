#!/usr/bin/env python3

"""Study successive PTX kernel iterations against Torch and Triton baselines."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

import cupy as cp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from benchmarks import core


OUTDIR = core.PTX_ITERATION_ARTIFACTS_DIR
SIZES = [128, 256, 512, 1024]
WARMUP = 5
ITERS = 50
SEED = 0


@dataclass
class Row:
    """One measurement row for the PTX iteration study."""

    size: int
    backend: str
    variant: str
    tile: str
    compile_ms: float | None
    first_launch_ms: float | None
    steady_latency_ms: float
    tflops: float


PTX_BASELINE_16 = dedent(
    r"""
    #include <cuda_fp16.h>
    __device__ __forceinline__ float to_float(__half x) { return __half2float(x); }
    __device__ __forceinline__ float fma_ptx(float a, float b, float c) {
        asm volatile ("fma.rn.f32 %0, %1, %2, %3;" : "=f"(c) : "f"(a), "f"(b), "f"(c));
        return c;
    }
    extern "C" __global__
    void matmul(const __half* __restrict__ A, const __half* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
        constexpr int TM = 16, TN = 16, TK = 16;
        __shared__ float As[TM][TK];
        __shared__ float Bs[TK][TN];
        int row = blockIdx.y * TM + threadIdx.y;
        int col = blockIdx.x * TN + threadIdx.x;
        float acc = 0.0f;
        for (int k0 = 0; k0 < K; k0 += TK) {
            int a_col = k0 + threadIdx.x;
            int b_row = k0 + threadIdx.y;
            As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? to_float(A[row * K + a_col]) : 0.0f;
            Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? to_float(B[b_row * N + col]) : 0.0f;
            __syncthreads();
            #pragma unroll
            for (int kk = 0; kk < TK; ++kk) acc = fma_ptx(As[threadIdx.y][kk], Bs[kk][threadIdx.x], acc);
            __syncthreads();
        }
        if (row < M && col < N) C[row * N + col] = acc;
    }
    """
)


PTX_SCALAR_32 = dedent(
    r"""
    #include <cuda_fp16.h>
    __device__ __forceinline__ float to_float(__half x) { return __half2float(x); }
    __device__ __forceinline__ float fma_ptx(float a, float b, float c) {
        asm volatile ("fma.rn.f32 %0, %1, %2, %3;" : "=f"(c) : "f"(a), "f"(b), "f"(c));
        return c;
    }
    extern "C" __global__
    void matmul(const __half* __restrict__ A, const __half* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
        constexpr int TM = 32, TN = 32, TK = 32;
        __shared__ float As[TM][TK];
        __shared__ float Bs[TK][TN];
        int row = blockIdx.y * TM + threadIdx.y;
        int col = blockIdx.x * TN + threadIdx.x;
        float acc = 0.0f;
        for (int k0 = 0; k0 < K; k0 += TK) {
            int a_col = k0 + threadIdx.x;
            int b_row = k0 + threadIdx.y;
            As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? to_float(A[row * K + a_col]) : 0.0f;
            Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? to_float(B[b_row * N + col]) : 0.0f;
            __syncthreads();
            #pragma unroll
            for (int kk = 0; kk < TK; ++kk) acc = fma_ptx(As[threadIdx.y][kk], Bs[kk][threadIdx.x], acc);
            __syncthreads();
        }
        if (row < M && col < N) C[row * N + col] = acc;
    }
    """
)


PTX_WMMA_32 = dedent(
    r"""
    #include <mma.h>
    #include <cuda_fp16.h>
    using namespace nvcuda;
    extern "C" __global__
    void matmul(const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
        int warp_id = threadIdx.x / 32;
        if (warp_id >= 4) return;
        int warp_row = warp_id / 2;
        int warp_col = warp_id % 2;
        int row = blockIdx.y * 32 + warp_row * 16;
        int col = blockIdx.x * 32 + warp_col * 16;

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        for (int k0 = 0; k0 < K; k0 += 16) {
            const half* a_ptr = A + row * K + k0;
            const half* b_ptr = B + k0 * N + col;
            wmma::load_matrix_sync(a_frag, a_ptr, K);
            wmma::load_matrix_sync(b_frag, b_ptr, N);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        if (row < M && col < N) {
            wmma::store_matrix_sync(C + row * N + col, c_frag, N, wmma::mem_row_major);
        }
    }
    """
)


VARIANTS = {
    "iter1_scalar16": {"code": PTX_BASELINE_16, "block": (16, 16, 1), "grid_div": (16, 16), "tile": "16x16x16"},
    "iter2_scalar32": {"code": PTX_SCALAR_32, "block": (32, 32, 1), "grid_div": (32, 32), "tile": "32x32x32"},
    "iter3_wmma32": {"code": PTX_WMMA_32, "block": (128, 1, 1), "grid_div": (32, 32), "tile": "32x32x16-tc"},
}


def compile_variant(code: str) -> cp.RawKernel:
    """Compile one PTX study variant."""

    mod = cp.RawModule(code=code, options=("-std=c++14",))
    return mod.get_function("matmul")


def run_variant(kernel: cp.RawKernel, variant: str, a, b, c, m: int, n: int, k: int) -> None:
    """Launch the chosen PTX study variant with its fixed block/grid policy."""

    cfg = VARIANTS[variant]
    gx = (n + cfg["grid_div"][0] - 1) // cfg["grid_div"][0]
    gy = (m + cfg["grid_div"][1] - 1) // cfg["grid_div"][1]
    kernel((gx, gy, 1), cfg["block"], (a.data_ptr(), b.data_ptr(), c.data_ptr(), np.int32(m), np.int32(n), np.int32(k)))


def metric_tflops(size: int, ms: float) -> float:
    """Convert square GEMM latency to TFLOP/s for the PTX study tables and plots."""

    return (2.0 * size * size * size) / (ms * 1e-3) / 1e12


def best_triton(a, b, size: int):
    """Pick Triton's best steady-state tile so PTX iterations compare to a fair tuned baseline."""

    candidates = [(16, 16, 16), (32, 32, 32), (64, 64, 16), (64, 64, 32)]
    out = core.torch.empty((size, size), device="cuda", dtype=core.torch.float32)
    best = None
    for tile in candidates:
        ms = core.benchmark_ms_cupy(
            lambda: core.run_triton(a, b, out, size, size, size, tile[0], tile[1], tile[2], core.triton_out_dtype("float16")),
            WARMUP,
            ITERS,
        )
        if best is None or ms < best[0]:
            best = (ms, tile)
    return best


def collect() -> list[Row]:
    """Collect benchmark data for Torch, best Triton, and each PTX iteration."""

    rows: list[Row] = []

    for size in SIZES:
        a, b = core.make_inputs("float16", size, size, size, seed=SEED)
        out_torch = core.torch.empty((size, size), device="cuda", dtype=core.torch.float16)
        torch_first_ms = core.measure_host_ms(lambda: core.run_torch(a, b, out_torch, "float16"))
        torch_steady = core.benchmark_ms_torch(lambda: core.run_torch(a, b, out_torch, "float16"), WARMUP, ITERS)
        rows.append(Row(size, "Torch", "baseline", "", 0.0, torch_first_ms, torch_steady, metric_tflops(size, torch_steady)))

        triton_out = core.torch.empty((size, size), device="cuda", dtype=core.torch.float32)
        triton_ms, triton_tile = best_triton(a, b, size)
        triton_first_ms = core.measure_host_ms(
            lambda: core.run_triton(
                a,
                b,
                triton_out,
                size,
                size,
                size,
                triton_tile[0],
                triton_tile[1],
                triton_tile[2],
                core.triton_out_dtype("float16"),
            )
        )
        rows.append(
            Row(
                size,
                "Triton",
                "best",
                f"{triton_tile[0]}x{triton_tile[1]}x{triton_tile[2]}",
                None,
                triton_first_ms,
                triton_ms,
                metric_tflops(size, triton_ms),
            )
        )

        for variant in VARIANTS:
            c = core.torch.empty((size, size), device="cuda", dtype=core.torch.float32)
            holder = {}
            compile_ms = core.measure_host_ms(lambda: holder.setdefault("kernel", compile_variant(VARIANTS[variant]["code"])))
            kernel = holder["kernel"]
            first_ms = core.measure_host_ms(lambda: run_variant(kernel, variant, a, b, c, size, size, size))
            steady_ms = core.benchmark_ms_cupy(lambda: run_variant(kernel, variant, a, b, c, size, size, size), WARMUP, ITERS)
            rows.append(Row(size, "PTX", variant, VARIANTS[variant]["tile"], compile_ms, first_ms, steady_ms, metric_tflops(size, steady_ms)))

    return rows


def write_outputs(rows: list[Row]) -> None:
    """Write raw PTX iteration data, plots, and a short analysis markdown file."""

    OUTDIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTDIR / "ptx_iteration_raw.csv"
    json_path = OUTDIR / "ptx_iteration_raw.json"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(Row.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump([row.__dict__ for row in rows], f, indent=2)

    colors = {
        "Torch": "#444444",
        "Triton": "#11aa88",
        "iter1_scalar16": "#3366cc",
        "iter2_scalar32": "#6699cc",
        "iter3_wmma32": "#ff6600",
    }
    labels = {
        "Torch": "Torch",
        "Triton": "Triton-best",
        "iter1_scalar16": "PTX iter1 scalar16",
        "iter2_scalar32": "PTX iter2 scalar32",
        "iter3_wmma32": "PTX iter3 wmma32",
    }

    def plot(metric: str, ylabel: str, path: Path) -> None:
        """Plot a grouped bar comparison for one metric across all PTX iterations."""

        fig, ax = plt.subplots(figsize=(12, 6))
        entities = ["Torch", "Triton", "iter1_scalar16", "iter2_scalar32", "iter3_wmma32"]
        width = 0.15
        x = list(range(len(SIZES)))
        for idx, ent in enumerate(entities):
            vals = []
            for size in SIZES:
                row = next(r for r in rows if r.size == size and (r.backend == ent or r.variant == ent))
                vals.append(getattr(row, metric))
            offsets = [xi + (idx - 2) * width for xi in x]
            ax.bar(offsets, vals, width=width, label=labels[ent], color=colors[ent])
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in SIZES])
        ax.set_xlabel("Matrix size (M=N=K)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, ncol=2)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)

    plot("tflops", "TFLOP/s", OUTDIR / "ptx_iteration_throughput.png")
    plot("steady_latency_ms", "Steady latency (ms)", OUTDIR / "ptx_iteration_latency.png")
    plot("first_launch_ms", "First launch latency (ms)", OUTDIR / "ptx_iteration_first_launch.png")

    md = [
        "# PTX Iteration Study",
        "",
        "## Iterations",
        "",
        "1. `iter1_scalar16`: current scalar FMA kernel with 16x16x16 tiling.",
        "2. `iter2_scalar32`: larger scalar tile to improve reuse and reduce grid overhead.",
        "3. `iter3_wmma32`: tensor-core WMMA kernel with 4 warps/block over a 32x32 output tile.",
        "",
        "## Bottleneck analysis",
        "",
        "- Iteration 1 bottleneck: scalar FP32 FMA path after fp16->float conversion; no Tensor Core usage.",
        "- Iteration 2 bottleneck: better locality, but still scalar math and large thread blocks; still no Tensor Core path.",
        "- Iteration 3 bottleneck: enters Tensor Core path, but still lacks library-grade pipelining, multi-stage scheduling, and deeper tiling compared with cuBLAS/Triton.",
        "",
        "## Artifacts",
        "",
        f"- Raw CSV: `{OUTDIR / 'ptx_iteration_raw.csv'}`",
        f"- Raw JSON: `{OUTDIR / 'ptx_iteration_raw.json'}`",
        f"- Throughput figure: `{OUTDIR / 'ptx_iteration_throughput.png'}`",
        f"- Steady-state latency figure: `{OUTDIR / 'ptx_iteration_latency.png'}`",
        f"- First-launch latency figure: `{OUTDIR / 'ptx_iteration_first_launch.png'}`",
    ]
    (OUTDIR / "ptx_iteration_analysis.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    """Run the PTX iteration study and write the artifact bundle."""

    rows = collect()
    write_outputs(rows)
    print(json.dumps({"outdir": str(OUTDIR), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
