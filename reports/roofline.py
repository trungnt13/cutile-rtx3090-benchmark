#!/usr/bin/env python3

"""Generate a roofline plot from benchmark raw data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from benchmarks.core import PEAK_TFLOPS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FULL_ARTIFACTS = PROJECT_ROOT / "artifacts" / "full"
OUTDIR = FULL_ARTIFACTS


# RTX 3090 specs
MEM_BW_GBS = 936.0  # GB/s
DTYPE_SIZEOF_IN = {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1}
DTYPE_SIZEOF_OUT = {"float32": 4, "float16": 4, "bfloat16": 4, "int8": 4}


def arithmetic_intensity(m: int, n: int, k: int, dtype_name: str) -> float:
    """Compute arithmetic intensity: FLOPs / bytes transferred."""
    flops = 2.0 * m * n * k
    sizeof_in = DTYPE_SIZEOF_IN[dtype_name]
    sizeof_out = DTYPE_SIZEOF_OUT[dtype_name]
    bytes_transferred = (m * k + k * n) * sizeof_in + m * n * sizeof_out
    return flops / bytes_transferred


def roofline_ceiling(ai: float, peak_tflops: float, mem_bw_gbs: float) -> float:
    """Return the roofline-limited throughput at a given arithmetic intensity."""
    mem_bound = ai * mem_bw_gbs / 1000.0  # Convert GB/s to TFLOP/s
    return min(peak_tflops, mem_bound)


def main() -> None:
    raw_json = FULL_ARTIFACTS / "benchmark_raw.json"
    if not raw_json.exists():
        print(f"Missing {raw_json}. Run reports.full_report first.")
        return

    data = json.loads(raw_json.read_text(encoding="utf-8"))
    OUTDIR.mkdir(parents=True, exist_ok=True)

    dtypes = ["float32", "float16", "bfloat16", "int8"]
    dtype_colors = {"float32": "#3366cc", "float16": "#ff6600", "bfloat16": "#11aa88", "int8": "#cc3333"}
    backend_markers = {"cuTile": "o", "PTX-inline": "s", "Triton": "^", "Torch": "D", "cuBLAS": "P"}

    fig, ax = plt.subplots(figsize=(12, 8))

    # Draw roofline ceilings
    ai_range = np.logspace(-1, 4, 500)
    for dtype_name in dtypes:
        peak = PEAK_TFLOPS[dtype_name]
        ceilings = [roofline_ceiling(ai, peak, MEM_BW_GBS) for ai in ai_range]
        ax.plot(ai_range, ceilings, "--", color=dtype_colors[dtype_name], alpha=0.5, label=f"{dtype_name} roofline")

    # Scatter benchmark points
    for row in data:
        dtype_name = row["dtype"]
        m, n, k = row["m"], row["n"], row["k"]
        ai = arithmetic_intensity(m, n, k, dtype_name)
        perf = row["perf"]
        backend = row["backend"]
        # Normalize backend name (strip tile size from cuTile variants)
        base_backend = backend.split()[0] if backend.startswith("cuTile") else backend
        marker = backend_markers.get(base_backend, "x")
        ax.scatter(ai, perf, c=dtype_colors[dtype_name], marker=marker, s=30, alpha=0.6, zorder=5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)")
    ax.set_ylabel("Throughput (TFLOP/s or TOP/s)")
    ax.set_title("RTX 3090 Roofline: cuTile Benchmark Points")
    ax.grid(True, alpha=0.3, which="both")

    # Custom legend for backends
    import matplotlib.lines as mlines
    backend_handles = [mlines.Line2D([], [], color="gray", marker=m, linestyle="", markersize=8, label=b)
                       for b, m in backend_markers.items()]
    dtype_handles = [mlines.Line2D([], [], color=dtype_colors[d], linestyle="--", label=f"{d} roofline")
                     for d in dtypes]
    ax.legend(handles=dtype_handles + backend_handles, loc="lower right", fontsize=9, ncol=2)

    fig.tight_layout()
    out_path = OUTDIR / "roofline.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(json.dumps({"roofline_plot": str(out_path)}))


if __name__ == "__main__":
    main()
