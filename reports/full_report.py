#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from benchmarks import core as bench


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IR_ARTIFACT_DIR = PROJECT_ROOT / "investigations" / "int8_ir"
OUTDIR = PROJECT_ROOT / "artifacts" / "full"
PTX_VALIDATION_JSON = OUTDIR / "ptx_latency_validation_summary.json"
NSYS_DIR = PROJECT_ROOT / "artifacts" / "nsys"

SIZES = [128, 256, 512, 1024]
TILES = [
    (16, 16, 16),
    (32, 32, 16),
    (32, 32, 32),
    (64, 64, 16),
    (64, 64, 32),
    (64, 64, 64),
    (128, 128, 128),
]
DTYPES = ["float32", "float16", "bfloat16", "int8"]
BACKENDS = ["cuTile", "PTX-inline", "Triton", "Torch"]
COMPARISON_BACKENDS = [
    "Torch",
    "PTX-inline",
    "Triton",
    "cuTile 32x32x32",
    "cuTile 64x64x64",
    "cuTile 128x128x128",
]
WARMUP = 3
ITERS = 20
SEED = 0


@dataclass
class Row:
    """One measured point in the full benchmark sweep."""

    dtype: str
    backend: str
    m: int
    n: int
    k: int
    tile_m: int | None
    tile_n: int | None
    tile_k: int | None
    compile_ms: float | None
    first_launch_ms: float | None
    latency_ms: float
    perf: float
    perf_unit: str
    max_err_exact: float | None
    max_err_wrapped_i8: float | None
    max_err_tile_wrapped_i8: float | None


def make_inputs(dtype_name: str, m: int, n: int, k: int):
    """Create deterministic inputs so plots remain reproducible across reruns."""

    return bench.make_inputs(dtype_name, m, n, k, seed=SEED)


def metric_value(dtype_name: str, m: int, n: int, k: int, ms: float) -> tuple[float, str]:
    """Return the human-facing throughput metric for a dtype."""

    if dtype_name == "int8":
        return bench.tops(m, n, k, ms), "TOP/s"
    return bench.tflops(m, n, k, ms), "TFLOP/s"


def collect_cutile_only_row(
    dtype_name: str,
    a,
    b,
    reference,
    wrapped_reference,
    tile_m: int,
    tile_n: int,
    tile_k: int,
) -> Row:
    """Measure the cuTile-only 128^3 tile that has no matching Triton/PTX comparison row."""

    cutile_kernel = bench.make_cutile_kernel(dtype_name, None, None)
    out_dtype = bench.DTYPE_INFO[dtype_name]["out"]
    c_cutile = torch.empty((a.shape[0], b.shape[1]), device="cuda", dtype=out_dtype)
    compile_args = (a, b, c_cutile, tile_m, tile_n, tile_k, bench.ct.cdiv(a.shape[1], tile_k))
    compile_ms = bench.measure_host_ms(lambda: bench.compile_cutile_kernel(cutile_kernel, compile_args, None, None))
    first_launch_ms = bench.measure_host_ms(
        lambda: bench.run_cutile(cutile_kernel, a, b, c_cutile, a.shape[0], b.shape[1], a.shape[1], tile_m, tile_n, tile_k)
    )
    bench.run_cutile(cutile_kernel, a, b, c_cutile, a.shape[0], b.shape[1], a.shape[1], tile_m, tile_n, tile_k)
    torch.cuda.synchronize()

    latency_ms = bench.benchmark_ms_cupy(
        lambda: bench.run_cutile(cutile_kernel, a, b, c_cutile, a.shape[0], b.shape[1], a.shape[1], tile_m, tile_n, tile_k),
        WARMUP,
        ITERS,
    )
    perf, unit = metric_value(dtype_name, a.shape[0], b.shape[1], a.shape[1], latency_ms)
    wrapped_err = float((c_cutile - wrapped_reference).abs().max().item()) if wrapped_reference is not None else None
    tile_wrapped_err = (
        float((c_cutile - bench.make_chunk_wrapped_i8_reference(a, b, tile_k)).abs().max().item())
        if dtype_name == "int8"
        else None
    )
    return Row(
        dtype=dtype_name,
        backend="cuTile",
        m=a.shape[0],
        n=b.shape[1],
        k=a.shape[1],
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        compile_ms=compile_ms,
        first_launch_ms=first_launch_ms,
        latency_ms=latency_ms,
        perf=perf,
        perf_unit=unit,
        max_err_exact=float((c_cutile - reference).abs().max().item()),
        max_err_wrapped_i8=wrapped_err,
        max_err_tile_wrapped_i8=tile_wrapped_err,
    )


def collect_rows() -> list[Row]:
    """Run the full dtype/size/tile sweep and collect comparable backend rows."""

    rows: list[Row] = []
    ptx_kernels = {dtype: bench.compile_ptx(dtype) for dtype in DTYPES}

    for dtype_name in DTYPES:
        info = bench.DTYPE_INFO[dtype_name]
        tile_list = TILES if dtype_name != "int8" else [tile for tile in TILES if tile[2] >= 32]
        for size in SIZES:
            m = n = k = size
            a, b = make_inputs(dtype_name, m, n, k)
            reference = bench.make_reference(a, b, dtype_name)
            wrapped_reference = reference.to(torch.int8).to(torch.int32) if dtype_name == "int8" else None

            torch_baseline = torch.empty(
                (m, n),
                device="cuda",
                dtype=(info["torch"] if dtype_name != "int8" else torch.int32),
            )
            bench.run_torch(a, b, torch_baseline, dtype_name)
            torch.cuda.synchronize()
            torch_first_launch_ms = bench.measure_host_ms(lambda: bench.run_torch(a, b, torch_baseline, dtype_name))
            torch_ms = bench.benchmark_ms_torch(
                lambda: bench.run_torch(a, b, torch_baseline, dtype_name),
                WARMUP,
                ITERS,
            )
            torch_perf, torch_unit = metric_value(dtype_name, m, n, k, torch_ms)
            rows.append(
                Row(
                    dtype=dtype_name,
                    backend="Torch",
                    m=m,
                    n=n,
                    k=k,
                    tile_m=None,
                    tile_n=None,
                    tile_k=None,
                    compile_ms=0.0,
                    first_launch_ms=torch_first_launch_ms,
                    latency_ms=torch_ms,
                    perf=torch_perf,
                    perf_unit=torch_unit,
                    max_err_exact=0.0,
                    max_err_wrapped_i8=0.0 if dtype_name == "int8" else None,
                    max_err_tile_wrapped_i8=0.0 if dtype_name == "int8" else None,
                )
            )

            for tile_m, tile_n, tile_k in tile_list:
                if (tile_m, tile_n, tile_k) == (128, 128, 128):
                    rows.append(
                        collect_cutile_only_row(dtype_name, a, b, reference, wrapped_reference, tile_m, tile_n, tile_k)
                    )
                    continue

                result = bench.benchmark_tile_config(
                    dtype_name,
                    a,
                    b,
                    reference,
                    wrapped_reference,
                    ptx_kernels[dtype_name],
                    (tile_m, tile_n, tile_k),
                    WARMUP,
                    ITERS,
                    None,
                    None,
                )

                for backend_name, latency_ms, max_err in [
                    ("cuTile", result["cutile_ms"], result["cutile_err"]),
                    ("PTX-inline", result["ptx_ms"], result["ptx_err"]),
                    ("Triton", result["triton_ms"], result["triton_err"]),
                ]:
                    compile_ms = None
                    first_launch_ms = None
                    out_dtype = bench.DTYPE_INFO[dtype_name]["out"]
                    if backend_name == "cuTile":
                        cutile_kernel = bench.make_cutile_kernel(dtype_name, None, None)
                        c_tmp = torch.empty((m, n), device="cuda", dtype=out_dtype)
                        compile_args = (a, b, c_tmp, tile_m, tile_n, tile_k, bench.ct.cdiv(k, tile_k))
                        compile_ms = bench.measure_host_ms(
                            lambda: bench.compile_cutile_kernel(cutile_kernel, compile_args, None, None)
                        )
                        first_launch_ms = bench.measure_host_ms(
                            lambda: bench.run_cutile(cutile_kernel, a, b, c_tmp, m, n, k, tile_m, tile_n, tile_k)
                        )
                    elif backend_name == "PTX-inline":
                        c_tmp = torch.empty((m, n), device="cuda", dtype=out_dtype)
                        compiled: dict[str, object] = {}

                        def compile_only() -> None:
                            compiled["kernel"] = bench.compile_ptx(dtype_name)

                        compile_ms = bench.measure_host_ms(compile_only)
                        first_launch_ms = bench.measure_host_ms(
                            lambda: bench.run_ptx(compiled["kernel"], a, b, c_tmp, m, n, k)
                        )
                    elif backend_name == "Triton":
                        c_tmp = torch.empty((m, n), device="cuda", dtype=out_dtype)
                        first_launch_ms = bench.measure_host_ms(
                            lambda: bench.run_triton(
                                a,
                                b,
                                c_tmp,
                                m,
                                n,
                                k,
                                tile_m,
                                tile_n,
                                tile_k,
                                bench.triton_out_dtype(dtype_name),
                            )
                        )

                    perf, unit = metric_value(dtype_name, m, n, k, latency_ms)
                    rows.append(
                        Row(
                            dtype=dtype_name,
                            backend=backend_name,
                            m=m,
                            n=n,
                            k=k,
                            tile_m=tile_m,
                            tile_n=tile_n,
                            tile_k=tile_k,
                            compile_ms=compile_ms,
                            first_launch_ms=first_launch_ms,
                            latency_ms=latency_ms,
                            perf=perf,
                            perf_unit=unit,
                            max_err_exact=max_err,
                            max_err_wrapped_i8=result.get("cutile_wrapped_err")
                            if backend_name == "cuTile" and dtype_name == "int8"
                            else None,
                            max_err_tile_wrapped_i8=result.get("cutile_chunk_wrapped_err")
                            if backend_name == "cuTile" and dtype_name == "int8"
                            else None,
                        )
                    )

    return rows


def write_csv_json(rows: list[Row]) -> tuple[Path, Path]:
    """Write the raw full-sweep measurements in machine-readable form."""

    OUTDIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTDIR / "benchmark_raw.csv"
    json_path = OUTDIR / "benchmark_raw.json"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(Row.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump([row.__dict__ for row in rows], file, indent=2)

    return csv_path, json_path


def write_coldstart_exports(rows: list[Row]) -> tuple[Path, Path]:
    """Export the subset of rows that contain compile or first-launch timing."""

    csv_path = OUTDIR / "benchmark_coldstart.csv"
    json_path = OUTDIR / "benchmark_coldstart.json"
    cold_rows = [row for row in rows if row.first_launch_ms is not None or row.compile_ms is not None]

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(Row.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in cold_rows:
            writer.writerow(row.__dict__)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump([row.__dict__ for row in cold_rows], file, indent=2)

    return csv_path, json_path


def select_best_rows(rows: list[Row]) -> list[Row]:
    """Keep the best-throughput row for each dtype/backend/size group."""

    best: dict[tuple[str, str, int], Row] = {}
    for row in rows:
        key = (row.dtype, row.backend, row.m)
        current = best.get(key)
        if current is None or row.perf > current.perf:
            best[key] = row
    return list(best.values())


def select_comparison_rows(rows: list[Row]) -> list[Row]:
    """Build the curated rows used in the public comparison figures."""

    selected: list[Row] = []
    for dtype_name in DTYPES:
        for size in SIZES:
            torch_row = next((r for r in rows if r.dtype == dtype_name and r.backend == "Torch" and r.m == size), None)
            if torch_row is not None:
                selected.append(torch_row)

            # PTX is tile-invariant in this benchmark, so the median run is more representative than taking
            # the best repeated measurement and accidentally overstating a fixed kernel.
            ptx_rows = [r for r in rows if r.dtype == dtype_name and r.backend == "PTX-inline" and r.m == size]
            if ptx_rows:
                ptx_rows = sorted(ptx_rows, key=lambda row: row.latency_ms)
                median = ptx_rows[len(ptx_rows) // 2]
                perf, unit = metric_value(dtype_name, size, size, size, median.latency_ms)
                selected.append(
                    Row(
                        dtype=dtype_name,
                        backend="PTX-inline",
                        m=size,
                        n=size,
                        k=size,
                        tile_m=median.tile_m,
                        tile_n=median.tile_n,
                        tile_k=median.tile_k,
                        compile_ms=median.compile_ms,
                        first_launch_ms=median.first_launch_ms,
                        latency_ms=median.latency_ms,
                        perf=perf,
                        perf_unit=unit,
                        max_err_exact=median.max_err_exact,
                        max_err_wrapped_i8=median.max_err_wrapped_i8,
                        max_err_tile_wrapped_i8=median.max_err_tile_wrapped_i8,
                    )
                )

            triton_rows = [r for r in rows if r.dtype == dtype_name and r.backend == "Triton" and r.m == size]
            if triton_rows:
                selected.append(max(triton_rows, key=lambda row: row.perf))

            # The public comparison intentionally carries a fixed set of cuTile tiles so size-to-size plots
            # show the trade-off between explicit tuning choices rather than changing the candidate set.
            for tile in ((32, 32, 32), (64, 64, 64), (128, 128, 128)):
                row = next(
                    (
                        candidate
                        for candidate in rows
                        if candidate.dtype == dtype_name
                        and candidate.backend == "cuTile"
                        and candidate.m == size
                        and (candidate.tile_m, candidate.tile_n, candidate.tile_k) == tile
                    ),
                    None,
                )
                if row is not None:
                    selected.append(
                        Row(
                            dtype=row.dtype,
                            backend=f"cuTile {tile[0]}x{tile[1]}x{tile[2]}",
                            m=row.m,
                            n=row.n,
                            k=row.k,
                            tile_m=row.tile_m,
                            tile_n=row.tile_n,
                            tile_k=row.tile_k,
                            compile_ms=row.compile_ms,
                            first_launch_ms=row.first_launch_ms,
                            latency_ms=row.latency_ms,
                            perf=row.perf,
                            perf_unit=row.perf_unit,
                            max_err_exact=row.max_err_exact,
                            max_err_wrapped_i8=row.max_err_wrapped_i8,
                            max_err_tile_wrapped_i8=row.max_err_tile_wrapped_i8,
                        )
                    )
    return selected


def write_best_csv(rows: list[Row]) -> Path:
    """Persist the best-of-sweep summary used by the public report."""

    path = OUTDIR / "benchmark_best.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(Row.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item.dtype, item.backend, item.m)):
            writer.writerow(row.__dict__)
    return path


def select_first_launch_rows(rows: list[Row]) -> list[Row]:
    """Filter the comparison rows down to entries that belong in the cold-start chart."""

    comparison = select_comparison_rows(rows)
    return [
        row
        for row in comparison
        if row.backend.startswith("cuTile ") or row.backend in ("Torch", "PTX-inline", "Triton")
    ]


def plot_metric_bar(
    rows: list[Row],
    metric_name: str,
    ylabel_name: str,
    out_path: Path,
    backends: list[str],
    logy: bool = False,
) -> None:
    """Plot a 2x2 grid of dtype-specific comparison bars."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    backend_style = {
        "Torch": {"color": "#444444", "hatch": ""},
        "PTX-inline": {"color": "#3366cc", "hatch": ""},
        "Triton": {"color": "#11aa88", "hatch": ""},
        "cuTile 32x32x32": {"color": "#ff6600", "hatch": ""},
        "cuTile 64x64x64": {"color": "#ff6600", "hatch": "//"},
        "cuTile 128x128x128": {"color": "#ff6600", "hatch": "xx"},
    }
    sizes = sorted({row.m for row in rows})
    width = 0.11

    for axis, dtype_name in zip(axes.flat, DTYPES, strict=True):
        subset = [row for row in rows if row.dtype == dtype_name]
        x = list(range(len(sizes)))
        for index, backend_name in enumerate(backends):
            values = []
            for size in sizes:
                row = next((candidate for candidate in subset if candidate.backend == backend_name and candidate.m == size), None)
                values.append(getattr(row, metric_name) if row is not None and getattr(row, metric_name) is not None else 0.0)
            offsets = [xpos + (index - (len(backends) - 1) / 2) * width for xpos in x]
            style = backend_style[backend_name]
            axis.bar(
                offsets,
                values,
                width=width,
                label=backend_name,
                color=style["color"],
                hatch=style["hatch"],
                edgecolor="#222222" if backend_name.startswith("cuTile") else style["color"],
                linewidth=0.8 if backend_name.startswith("cuTile") else 0.0,
            )
        axis.set_title(dtype_name)
        axis.set_xlabel("Matrix size (M=N=K)")
        axis.set_xticks(x)
        axis.set_xticklabels([str(size) for size in sizes])
        if dtype_name == "int8" and metric_name == "perf":
            axis.set_ylabel("TOP/s")
        elif metric_name == "perf":
            axis.set_ylabel("TFLOP/s")
        else:
            axis.set_ylabel(ylabel_name)
        if logy:
            axis.set_yscale("log")
        axis.grid(True, alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_cutile_tile_bars(
    rows: list[Row],
    metric_name: str,
    ylabel_name: str,
    out_path: Path,
    logy: bool = False,
) -> None:
    """Plot every explicit cuTile tile candidate for each dtype and size."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    cutile_rows = [row for row in rows if row.backend == "cuTile"]
    sizes = sorted({row.m for row in cutile_rows})
    tile_labels = [f"{tm}x{tn}x{tk}" for tm, tn, tk in TILES]
    colors = ["#ffcc00", "#ffad1f", "#ff6600", "#e06a00", "#cc5500", "#994400", "#663300"]
    width = 0.12

    for axis, dtype_name in zip(axes.flat, DTYPES, strict=True):
        subset = [row for row in cutile_rows if row.dtype == dtype_name]
        x = list(range(len(sizes)))
        for index, tile in enumerate(TILES):
            values = []
            for size in sizes:
                row = next(
                    (candidate for candidate in subset if candidate.m == size and (candidate.tile_m, candidate.tile_n, candidate.tile_k) == tile),
                    None,
                )
                values.append(getattr(row, metric_name) if row is not None else 0.0)
            offsets = [xpos + (index - (len(TILES) - 1) / 2) * width for xpos in x]
            axis.bar(offsets, values, width=width, label=tile_labels[index], color=colors[index])
        axis.set_title(f"{dtype_name} / cuTile")
        axis.set_xlabel("Matrix size (M=N=K)")
        axis.set_xticks(x)
        axis.set_xticklabels([str(size) for size in sizes])
        if dtype_name == "int8" and metric_name == "perf":
            axis.set_ylabel("TOP/s")
        elif metric_name == "perf":
            axis.set_ylabel("TFLOP/s")
        else:
            axis.set_ylabel(ylabel_name)
        if logy:
            axis.set_yscale("log")
        axis.grid(True, alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_tile_note(best_rows: list[Row], dtype_name: str) -> list[str]:
    """Summarize the best tile per size for a dtype/backend pair."""

    lines = []
    for backend_name in ("cuTile", "Triton"):
        rows = sorted(
            (row for row in best_rows if row.dtype == dtype_name and row.backend == backend_name),
            key=lambda row: row.m,
        )
        if not rows:
            continue
        tile_summary = ", ".join(f"{row.m}:{row.tile_m}x{row.tile_n}x{row.tile_k}" for row in rows)
        lines.append(f"- {backend_name} best tiles by size: {tile_summary}")
    return lines


def write_ir_summary(best_rows: list[Row]) -> Path:
    """Write the narrative markdown summary that ties results to the int8 IR investigation."""

    summary_json = json.loads((IR_ARTIFACT_DIR / "summary.json").read_text(encoding="utf-8"))
    ir_text = (IR_ARTIFACT_DIR / "mm_i8.cutileir.txt").read_text(encoding="utf-8").splitlines()
    ir_excerpt = "\n".join(ir_text[36:42])
    ptx_validation = json.loads(PTX_VALIDATION_JSON.read_text(encoding="utf-8")) if PTX_VALIDATION_JSON.exists() else None

    latency_notes = []
    for dtype_name in ("float16", "bfloat16", "float32", "int8"):
        rows = [row for row in best_rows if row.dtype == dtype_name]
        if not rows:
            continue
        fastest = min(rows, key=lambda row: row.latency_ms)
        latency_notes.append(
            f"- Fastest observed {dtype_name} latency: {fastest.latency_ms:.3f} ms "
            f"on {fastest.backend} at size {fastest.m}."
        )

    lines = [
        "# cuTile Benchmark and IR Summary",
        "",
        "## Thesis-oriented findings",
        "",
        "1. cuTile can be a throughput and latency competitive path on Ampere for FP16/BF16 when tile shapes are tuned.",
        "2. cuTile still has correctness and semantics issues, most clearly on the int8 path.",
        "3. TileIR/cuTile is promising, but the current stack needs careful validation before claiming production correctness.",
        "4. This artifact set does not yet prove a memory-footprint advantage; that needs a separate workspace/allocator instrumentation pass.",
        "",
        "## Latency emphasis",
        "",
        "The benchmark data includes average kernel latency in milliseconds for every backend, dtype, size, and tile.",
        "For real-time AI/ML, the low-size regime (128/256) is the most important latency view in this artifact set; throughput alone is not sufficient.",
        "",
        *latency_notes,
        "",
        "Cold-start timing is also reported separately via `compile_ms` and `first_launch_ms`.",
        "For PTX, steady-state latency excludes module compile time; compile and first-launch costs are exported separately.",
        "",
        "## PTX timing validation",
        "",
    ]
    if ptx_validation is not None:
        lines.extend(
            [
                f"- PTX validation case: `{ptx_validation['dtype']}` at shape `{ptx_validation['shape'][0]}x{ptx_validation['shape'][1]}x{ptx_validation['shape'][2]}`",
                f"- Compile time: `{ptx_validation['compile_ms']:.3f} ms`",
                f"- First launch time: `{ptx_validation['first_launch_ms']:.3f} ms`",
                f"- Steady-state latency: `{ptx_validation['steady_latency_ms']:.3f} ms`",
                "- Nsight Systems trace was captured with NVTX ranges `ptx_compile`, `ptx_first_launch`, and `ptx_steady_state` to verify the phase separation.",
                "",
            ]
        )
    else:
        lines.extend(["- PTX latency validation JSON is missing.", ""])

    lines.extend(
        [
            "## Int8 IR finding",
            "",
            "The exported int8 repro under `investigations/int8_ir/` shows that cuTile does not preserve exact int32 GEMM semantics for `i8 @ i8`.",
            f"- `max_err_exact`: {summary_json['max_err_exact']}",
            f"- `max_err_tile_wrapped_i8`: {summary_json['max_err_tile_wrapped_i8']}",
            "",
            "That means the cuTile result matches a per-tile wrapped-int8 accumulation model rather than an exact int32 accumulation model.",
            "",
            "Critical IR excerpt:",
            "",
            "```text",
            ir_excerpt,
            "```",
            "",
            "Interpretation:",
            "",
            "- `tile_mma(...)` first produces an `int32` tile.",
            "- cuTile then inserts `tile_astype` to narrow that tile to `int8`.",
            "- It then widens the wrapped `int8` tile back to `int32` before accumulating.",
            "",
            "So the int8 bug is visible directly in the cuTile IR, not just in the benchmark output.",
            "",
            "## Best tile notes",
            "",
        ]
    )
    for dtype_name in DTYPES:
        lines.extend([f"### {dtype_name}", *make_tile_note(best_rows, dtype_name), ""])

    lines.extend(
        [
            "## Artifact files",
            "",
            f"- Raw benchmark CSV: `{OUTDIR / 'benchmark_raw.csv'}`",
            f"- Raw benchmark JSON: `{OUTDIR / 'benchmark_raw.json'}`",
            f"- Cold-start benchmark CSV: `{OUTDIR / 'benchmark_coldstart.csv'}`",
            f"- Cold-start benchmark JSON: `{OUTDIR / 'benchmark_coldstart.json'}`",
            f"- Best-of-sweep CSV: `{OUTDIR / 'benchmark_best.csv'}`",
            f"- Comparison throughput barplot: `{OUTDIR / 'comparison_throughput.png'}`",
            f"- Comparison latency barplot: `{OUTDIR / 'comparison_latency.png'}`",
            f"- Comparison first-launch latency barplot: `{OUTDIR / 'comparison_first_launch_latency.png'}`",
            f"- cuTile tile throughput barplot: `{OUTDIR / 'cutile_tile_sweep_throughput.png'}`",
            f"- cuTile tile latency barplot: `{OUTDIR / 'cutile_tile_sweep_latency.png'}`",
            f"- PTX timing validation JSON: `{PTX_VALIDATION_JSON}`",
            f"- Nsight Systems trace dir: `{NSYS_DIR}`",
            f"- Int8 IR summary source: `{IR_ARTIFACT_DIR / 'summary.json'}`",
            f"- Int8 cuTile IR text: `{IR_ARTIFACT_DIR / 'mm_i8.cutileir.txt'}`",
        ]
    )

    path = OUTDIR / "report_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    """Generate raw exports, summary CSVs, figures, and report markdown for the full sweep."""

    rows = collect_rows()
    best_rows = select_best_rows(rows)
    comparison_rows = select_comparison_rows(rows)
    first_launch_rows = select_first_launch_rows(rows)
    raw_csv, raw_json = write_csv_json(rows)
    coldstart_csv, coldstart_json = write_coldstart_exports(rows)
    best_csv = write_best_csv(best_rows)
    throughput_figure = OUTDIR / "comparison_throughput.png"
    latency_figure = OUTDIR / "comparison_latency.png"
    first_launch_figure = OUTDIR / "comparison_first_launch_latency.png"
    cutile_throughput_figure = OUTDIR / "cutile_tile_sweep_throughput.png"
    cutile_latency_figure = OUTDIR / "cutile_tile_sweep_latency.png"
    report_summary = write_ir_summary(best_rows)

    plot_metric_bar(comparison_rows, "perf", "Performance", throughput_figure, COMPARISON_BACKENDS)
    plot_metric_bar(comparison_rows, "latency_ms", "Steady latency (ms)", latency_figure, COMPARISON_BACKENDS, logy=True)
    plot_metric_bar(
        first_launch_rows,
        "first_launch_ms",
        "First launch latency (ms)",
        first_launch_figure,
        COMPARISON_BACKENDS,
        logy=True,
    )
    plot_cutile_tile_bars(rows, "perf", "Performance", cutile_throughput_figure)
    plot_cutile_tile_bars(rows, "latency_ms", "Steady latency (ms)", cutile_latency_figure, logy=True)

    print(
        json.dumps(
            {
                "outdir": str(OUTDIR),
                "raw_csv": str(raw_csv),
                "raw_json": str(raw_json),
                "coldstart_csv": str(coldstart_csv),
                "coldstart_json": str(coldstart_json),
                "best_csv": str(best_csv),
                "throughput_figure": str(throughput_figure),
                "latency_figure": str(latency_figure),
                "first_launch_latency_figure": str(first_launch_figure),
                "cutile_tile_throughput_figure": str(cutile_throughput_figure),
                "cutile_tile_latency_figure": str(cutile_latency_figure),
                "report_summary": str(report_summary),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
