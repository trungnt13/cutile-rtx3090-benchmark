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


plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTDIR = PROJECT_ROOT / "artifacts" / "fp16_focus"

SIZES = [128, 256, 512, 1024, 2048, 4096, 8192]
CUTILE_TILES = [
    (32, 32, 32),
    (64, 64, 32),
    (64, 64, 64),
    (128, 64, 32),
    (64, 128, 32),
    (128, 64, 64),
    (64, 128, 64),
    (128, 128, 32),
    (128, 128, 64),
    (128, 128, 128),
]
TRITON_TILES = [(16, 16, 16), (32, 32, 32), (64, 64, 16), (64, 64, 32)]
OCCUPANCIES = [None, 1, 2, 4, 8]
WARMUP = 3
ITERS = 15
SEED = 0


@dataclass
class Row:
    """One FP16-focused tuning experiment."""

    backend: str
    size: int
    tile_m: int | None
    tile_n: int | None
    tile_k: int | None
    occupancy: int | None
    compile_ms: float | None
    first_launch_ms: float | None
    steady_latency_ms: float
    tflops: float
    latency_std_ms: float
    pct_peak: float
    max_err: float | None


def make_inputs(size: int):
    """Create deterministic FP16 inputs for a square GEMM benchmark."""

    return bench.make_inputs("float16", size, size, size, seed=SEED)


def metric_tflops(size: int, ms: float) -> float:
    """Convert runtime into TFLOP/s for a square GEMM."""

    return (2.0 * size * size * size) / (ms * 1.0e-3) / 1.0e12


def collect_torch(a, b, size: int) -> Row:
    """Measure the Torch baseline for one matrix size."""

    c = torch.empty((size, size), device="cuda", dtype=torch.float16)
    first_launch_ms = bench.measure_host_ms(lambda: bench.run_torch(a, b, c, "float16"))
    timing = bench.benchmark_ms_torch(lambda: bench.run_torch(a, b, c, "float16"), WARMUP, ITERS)
    steady = timing.mean
    std = timing.std
    tflops_val = metric_tflops(size, steady)
    return Row("Torch", size, None, None, None, None, 0.0, first_launch_ms, steady, tflops_val, std, bench.pct_of_peak("float16", tflops_val), 0.0)


def collect_triton(a, b, reference, size: int) -> list[Row]:
    """Measure every Triton tile candidate in the focused FP16 sweep."""

    rows: list[Row] = []
    for tile_m, tile_n, tile_k in TRITON_TILES:
        c = torch.empty((size, size), device="cuda", dtype=torch.float32)
        first_launch_ms = bench.measure_host_ms(
            lambda: bench.run_triton(a, b, c, size, size, size, tile_m, tile_n, tile_k, bench.triton_out_dtype("float16"))
        )
        timing = bench.benchmark_ms_torch(
            lambda: bench.run_triton(a, b, c, size, size, size, tile_m, tile_n, tile_k, bench.triton_out_dtype("float16")),
            WARMUP,
            ITERS,
        )
        steady = timing.mean
        std = timing.std
        err = float((c - reference).abs().max().item())
        tflops_val = metric_tflops(size, steady)
        rows.append(
            Row("Triton", size, tile_m, tile_n, tile_k, None, None, first_launch_ms, steady, tflops_val, std, bench.pct_of_peak("float16", tflops_val), err)
        )
    return rows


def collect_cutile(a, b, reference, size: int) -> list[Row]:
    """Sweep cuTile tile and occupancy choices for one matrix size."""

    rows: list[Row] = []
    for tile_m, tile_n, tile_k in CUTILE_TILES:
        for occupancy in OCCUPANCIES:
            try:
                kernel = bench.make_cutile_kernel("float16", None, occupancy)
                c = torch.empty((size, size), device="cuda", dtype=torch.float32)
                compile_args = (a, b, c, tile_m, tile_n, tile_k, bench.ct.cdiv(size, tile_k))
                compile_ms = bench.measure_host_ms(lambda: bench.compile_cutile_kernel(kernel, compile_args, None, occupancy))
                first_launch_ms = bench.measure_host_ms(
                    lambda: bench.run_cutile(kernel, a, b, c, size, size, size, tile_m, tile_n, tile_k)
                )
                timing = bench.benchmark_ms_cupy(
                    lambda: bench.run_cutile(kernel, a, b, c, size, size, size, tile_m, tile_n, tile_k),
                    WARMUP,
                    ITERS,
                )
                steady = timing.mean
                std = timing.std
                err = float((c - reference).abs().max().item())
                tflops_val = metric_tflops(size, steady)
                rows.append(
                    Row(
                        "cuTile",
                        size,
                        tile_m,
                        tile_n,
                        tile_k,
                        occupancy,
                        compile_ms,
                        first_launch_ms,
                        steady,
                        tflops_val,
                        std,
                        bench.pct_of_peak("float16", tflops_val),
                        err,
                    )
                )
            except Exception:
                continue
    return rows


def write_raw(rows: list[Row]) -> None:
    """Write the raw FP16-focused sweep outputs."""

    OUTDIR.mkdir(parents=True, exist_ok=True)
    with (OUTDIR / "fp16_raw.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(Row.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)
    with (OUTDIR / "fp16_raw.json").open("w", encoding="utf-8") as file:
        json.dump([row.__dict__ for row in rows], file, indent=2)


def plot_grouped(rows: list[Row], metric: str, ylabel: str, out_path: Path) -> None:
    """Plot the high-level Torch/Triton/cuTile comparison bars for FP16."""

    fig, ax = plt.subplots(figsize=(12, 6))
    entities = ["Torch", "Triton-best", "cuTile-best", "cuTile 32³", "cuTile 64³", "cuTile 128³"]
    style = {
        "Torch": {"color": "#444444", "hatch": ""},
        "Triton-best": {"color": "#11aa88", "hatch": ""},
        "cuTile-best": {"color": "#ff6600", "hatch": ""},
        "cuTile 32³": {"color": "#ff6600", "hatch": ""},
        "cuTile 64³": {"color": "#ff6600", "hatch": "//"},
        "cuTile 128³": {"color": "#ff6600", "hatch": "xx"},
    }
    width = 0.11
    x = list(range(len(SIZES)))
    ymax = 0.0
    for index, entity in enumerate(entities):
        vals = []
        entity_rows = []
        for size in SIZES:
            row = next((candidate for candidate in rows if candidate.backend == entity and candidate.size == size), None)
            entity_rows.append(row)
            vals.append(getattr(row, metric) if row is not None else 0.0)
        offsets = [xpos + (index - (len(entities) - 1) / 2) * width for xpos in x]
        style_entry = style[entity]
        ax.bar(
            offsets,
            vals,
            width=width,
            label=entity,
            color=style_entry["color"],
            hatch=style_entry["hatch"],
            edgecolor="#222222" if entity.startswith("cuTile") else style_entry["color"],
            linewidth=0.8 if entity.startswith("cuTile") else 0.0,
        )
        ymax = max(ymax, max(vals, default=0.0))
        if entity == "cuTile-best":
            for xpos, row, val in zip(offsets, entity_rows, vals, strict=True):
                if row is None or val <= 0.0:
                    continue
                occ = "auto" if row.occupancy is None else str(row.occupancy)
                ax.annotate(
                    f"{row.tile_m}x{row.tile_n}x{row.tile_k}\nocc={occ}",
                    xy=(xpos, val),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                    color="#7a2f00",
                )
    ax.set_xticks(x)
    ax.set_xticklabels([str(size) for size in SIZES])
    ax.set_xlabel("Matrix size (M=N=K)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if ymax > 0.0:
        ax.set_ylim(0, ymax * 1.28)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_cutile_tiles(rows: list[Row], metric: str, ylabel: str, out_path: Path) -> None:
    """Plot a size sweep for the three named cuTile tile families."""

    fig, ax = plt.subplots(figsize=(14, 6))
    focus_tiles = [(32, 32, 32), (64, 64, 64), (128, 128, 128)]
    style = {
        (32, 32, 32): {"color": "#ff6600", "hatch": ""},
        (64, 64, 64): {"color": "#ff6600", "hatch": "//"},
        (128, 128, 128): {"color": "#ff6600", "hatch": "xx"},
    }
    width = 0.2
    x = list(range(len(SIZES)))
    for index, tile in enumerate(focus_tiles):
        vals = []
        for size in SIZES:
            row = next(
                (
                    candidate
                    for candidate in rows
                    if candidate.backend == "cuTile"
                    and candidate.size == size
                    and (candidate.tile_m, candidate.tile_n, candidate.tile_k) == tile
                ),
                None,
            )
            vals.append(getattr(row, metric) if row is not None else 0.0)
        offsets = [xpos + (index - 1) * width for xpos in x]
        style_entry = style[tile]
        ax.bar(
            offsets,
            vals,
            width=width,
            label=f"{tile[0]}³",
            color=style_entry["color"],
            hatch=style_entry["hatch"],
            edgecolor="#222222",
            linewidth=0.8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([str(size) for size in SIZES])
    ax.set_xlabel("Matrix size (M=N=K)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def is_pareto_optimal(points: list[tuple[float, float]]) -> list[bool]:
    """Return a mask for points that are not dominated on latency/throughput."""

    result = []
    for i, (lat_i, thr_i) in enumerate(points):
        dominated = False
        for j, (lat_j, thr_j) in enumerate(points):
            if i == j:
                continue
            if lat_j <= lat_i and thr_j >= thr_i and (lat_j < lat_i or thr_j > thr_i):
                dominated = True
                break
        result.append(not dominated)
    return result


def plot_fp16_pareto_tradeoff(rows: list[Row], out_path: Path) -> None:
    """Plot throughput-vs-latency scatter plots for the curated FP16 comparison set."""

    ncols = min(4, len(SIZES))
    nrows = (len(SIZES) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5))
    entities = ["Torch", "Triton-best", "cuTile-best", "cuTile 32³", "cuTile 64³", "cuTile 128³"]
    style = {
        "Torch": {"color": "#444444", "marker": "s"},
        "Triton-best": {"color": "#11aa88", "marker": "^"},
        "cuTile-best": {"color": "#ff6600", "marker": "o"},
        "cuTile 32³": {"color": "#ff6600", "marker": "P"},
        "cuTile 64³": {"color": "#ff6600", "marker": "X"},
        "cuTile 128³": {"color": "#ff6600", "marker": "D"},
    }
    legend_handles = [
        plt.Line2D([], [], color=style[entity]["color"], marker=style[entity]["marker"], linestyle="", markersize=8, label=entity)
        for entity in entities
    ]
    all_axes = axes.flat if hasattr(axes, 'flat') else [axes]
    for idx, axis in enumerate(all_axes):
        if idx >= len(SIZES):
            axis.set_visible(False)
            continue
        size = SIZES[idx]
        subset = [row for row in rows if row.size == size]
        points = []
        for entity in entities:
            row = next((candidate for candidate in subset if candidate.backend == entity), None)
            if row is None:
                continue
            style_entry = style[entity]
            neg_latency = -row.steady_latency_ms
            axis.scatter(neg_latency, row.tflops, s=80, color=style_entry["color"], marker=style_entry["marker"], label=entity)
            if row is not None and row.latency_std_ms > 0:
                axis.errorbar(neg_latency, row.tflops, xerr=row.latency_std_ms, fmt='none', ecolor=style_entry["color"], alpha=0.5, capsize=2)
            if entity == "cuTile-best":
                occ = "auto" if row.occupancy is None else str(row.occupancy)
                axis.annotate(
                    f"{row.tile_m}x{row.tile_n}x{row.tile_k}\nocc={occ}",
                    (neg_latency, row.tflops),
                    xytext=(6, 4),
                    textcoords="offset points",
                    fontsize=10,
                    color="#7a2f00",
                    ha="left",
                    va="bottom",
                )
            points.append((neg_latency, row.tflops))
        if points:
            # Draw the non-dominated frontier so the plot highlights the true latency/throughput envelope.
            mask = is_pareto_optimal(points)
            frontier = sorted([point for point, keep in zip(points, mask, strict=True) if keep], key=lambda point: point[0])
            if len(frontier) >= 2:
                axis.plot(
                    [point[0] for point in frontier],
                    [point[1] for point in frontier],
                    linestyle="--",
                    color="#888888",
                    linewidth=1.2,
                )
        axis.set_title(f"FP16 Pareto @ {size}")
        axis.set_xlabel("- Steady latency (ms)")
        axis.set_ylabel("Throughput (TFLOP/s)")
        xmin, xmax = axis.get_xlim()
        ymin, ymax = axis.get_ylim()
        axis.axvspan(xmax - 0.22 * (xmax - xmin), xmax, color="#d8f3dc", alpha=0.18, zorder=0)
        axis.axhspan(ymin + 0.78 * (ymax - ymin), ymax, color="#d8f3dc", alpha=0.18, zorder=0)
        axis.annotate(
            "better latency",
            xy=(0.92, 0.96),
            xytext=(0.64, 0.96),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="#2d6a4f", lw=1.5),
            ha="center",
            va="center",
            fontsize=11,
            color="#2d6a4f",
        )
        axis.annotate(
            "better throughput",
            xy=(0.96, 0.88),
            xytext=(0.96, 0.56),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="#2d6a4f", lw=1.5),
            ha="center",
            va="center",
            rotation=90,
            fontsize=11,
            color="#2d6a4f",
        )
        axis.grid(True, alpha=0.3)
        axis.legend(
            handles=legend_handles,
            loc="upper left",
            frameon=True,
            framealpha=0.95,
            facecolor="white",
            edgecolor="#888888",
            fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_cutile_fp16_pareto_tiles(rows: list[Row], out_path: Path) -> None:
    """Plot Pareto frontiers restricted to the named cuTile tile families."""

    ncols = min(4, len(SIZES))
    nrows = (len(SIZES) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5))
    focus_tiles = [(32, 32, 32), (64, 64, 64), (128, 128, 128)]
    style = {
        (32, 32, 32): {"color": "#ff6600", "marker": "P"},
        (64, 64, 64): {"color": "#ff6600", "marker": "X"},
        (128, 128, 128): {"color": "#ff6600", "marker": "D"},
    }
    legend_handles = [
        plt.Line2D([], [], color=style[tile]["color"], marker=style[tile]["marker"], linestyle="", markersize=8, label=f"{tile[0]}³")
        for tile in focus_tiles
    ]
    all_axes = axes.flat if hasattr(axes, 'flat') else [axes]
    for idx, axis in enumerate(all_axes):
        if idx >= len(SIZES):
            axis.set_visible(False)
            continue
        size = SIZES[idx]
        subset = [row for row in rows if row.backend == "cuTile" and row.size == size]
        points = []
        for tile in focus_tiles:
            for row in [candidate for candidate in subset if (candidate.tile_m, candidate.tile_n, candidate.tile_k) == tile]:
                style_entry = style[tile]
                occ = "auto" if row.occupancy is None else str(row.occupancy)
                neg_latency = -row.steady_latency_ms
                axis.scatter(neg_latency, row.tflops, s=70, color=style_entry["color"], marker=style_entry["marker"], alpha=0.9)
                axis.annotate(
                    f"{tile[0]}³\nocc={occ}",
                    (neg_latency, row.tflops),
                    xytext=(5, 4),
                    textcoords="offset points",
                    fontsize=9,
                    color="#7a2f00",
                    ha="left",
                    va="bottom",
                )
                points.append((neg_latency, row.tflops))
        if points:
            mask = is_pareto_optimal(points)
            frontier = sorted([point for point, keep in zip(points, mask, strict=True) if keep], key=lambda point: point[0])
            if len(frontier) >= 2:
                axis.plot(
                    [point[0] for point in frontier],
                    [point[1] for point in frontier],
                    linestyle="--",
                    color="#888888",
                    linewidth=1.2,
                )
        axis.set_title(f"cuTile FP16 tiles @ {size}")
        axis.set_xlabel("- Steady latency (ms)")
        axis.set_ylabel("Throughput (TFLOP/s)")
        xmin, xmax = axis.get_xlim()
        ymin, ymax = axis.get_ylim()
        axis.axvspan(xmax - 0.22 * (xmax - xmin), xmax, color="#d8f3dc", alpha=0.18, zorder=0)
        axis.axhspan(ymin + 0.78 * (ymax - ymin), ymax, color="#d8f3dc", alpha=0.18, zorder=0)
        axis.annotate(
            "better latency",
            xy=(0.92, 0.96),
            xytext=(0.64, 0.96),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="#2d6a4f", lw=1.5),
            ha="center",
            va="center",
            fontsize=11,
            color="#2d6a4f",
        )
        axis.annotate(
            "better throughput",
            xy=(0.96, 0.88),
            xytext=(0.96, 0.56),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="#2d6a4f", lw=1.5),
            ha="center",
            va="center",
            rotation=90,
            fontsize=11,
            color="#2d6a4f",
        )
        axis.grid(True, alpha=0.3)
        axis.legend(
            handles=legend_handles,
            loc="upper left",
            frameon=True,
            framealpha=0.95,
            facecolor="white",
            edgecolor="#888888",
            fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_comparison(rows: list[Row]) -> list[Row]:
    """Build the reduced comparison set used in the FP16 summary plots."""

    comparison: list[Row] = []
    for size in SIZES:
        torch_row = next(row for row in rows if row.backend == "Torch" and row.size == size)
        comparison.append(
            Row(
                "Torch",
                size,
                None,
                None,
                None,
                None,
                torch_row.compile_ms,
                torch_row.first_launch_ms,
                torch_row.steady_latency_ms,
                torch_row.tflops,
                torch_row.latency_std_ms,
                torch_row.pct_peak,
                torch_row.max_err,
            )
        )

        triton_rows = [row for row in rows if row.backend == "Triton" and row.size == size]
        triton_best = max(triton_rows, key=lambda row: row.tflops)
        comparison.append(
            Row(
                "Triton-best",
                size,
                triton_best.tile_m,
                triton_best.tile_n,
                triton_best.tile_k,
                None,
                triton_best.compile_ms,
                triton_best.first_launch_ms,
                triton_best.steady_latency_ms,
                triton_best.tflops,
                triton_best.latency_std_ms,
                triton_best.pct_peak,
                triton_best.max_err,
            )
        )

        cutile_rows = [row for row in rows if row.backend == "cuTile" and row.size == size]
        cutile_best = max(cutile_rows, key=lambda row: row.tflops)
        comparison.append(
            Row(
                "cuTile-best",
                size,
                cutile_best.tile_m,
                cutile_best.tile_n,
                cutile_best.tile_k,
                cutile_best.occupancy,
                cutile_best.compile_ms,
                cutile_best.first_launch_ms,
                cutile_best.steady_latency_ms,
                cutile_best.tflops,
                cutile_best.latency_std_ms,
                cutile_best.pct_peak,
                cutile_best.max_err,
            )
        )

        # Keep named tile families in the public comparison even when they are not the global winner so
        # the figures show the tuning trade-off instead of hiding those shapes behind a single best row.
        for tile, label in [((32, 32, 32), "cuTile 32³"), ((64, 64, 64), "cuTile 64³"), ((128, 128, 128), "cuTile 128³")]:
            match = next((row for row in cutile_rows if (row.tile_m, row.tile_n, row.tile_k) == tile), None)
            if match is not None:
                comparison.append(
                    Row(
                        label,
                        size,
                        match.tile_m,
                        match.tile_n,
                        match.tile_k,
                        match.occupancy,
                        match.compile_ms,
                        match.first_launch_ms,
                        match.steady_latency_ms,
                        match.tflops,
                        match.latency_std_ms,
                        match.pct_peak,
                        match.max_err,
                    )
                )
    return comparison


def write_summary(rows: list[Row], comparison: list[Row]) -> None:
    """Write the concise FP16 optimization markdown summary."""

    del comparison
    best_by_size = {
        size: max((row for row in rows if row.backend == "cuTile" and row.size == size), key=lambda row: row.tflops)
        for size in SIZES
    }
    markdown = [
        "# FP16 cuTile Optimization Summary",
        "",
        "## Headline",
        "",
        "This report removes PTX from the main figures and focuses on whether tuned cuTile FP16 can beat Triton and Torch.",
        "",
        "## Best cuTile configs by size",
        "",
        *[
            f"- {size}: {row.tile_m}x{row.tile_n}x{row.tile_k}, occupancy={row.occupancy}, {row.tflops:.2f} TFLOP/s ({row.pct_peak:.1f}% of peak), {row.steady_latency_ms:.3f} ms"
            for size, row in best_by_size.items()
        ],
        "",
        "## Artifact files",
        "",
        f"- Raw CSV: `{OUTDIR / 'fp16_raw.csv'}`",
        f"- Raw JSON: `{OUTDIR / 'fp16_raw.json'}`",
        f"- Comparison throughput: `{OUTDIR / 'comparison_fp16_throughput.png'}`",
        f"- Comparison latency: `{OUTDIR / 'comparison_fp16_latency.png'}`",
        f"- Comparison first-launch latency: `{OUTDIR / 'comparison_fp16_first_launch_latency.png'}`",
        f"- cuTile tile throughput: `{OUTDIR / 'cutile_fp16_tile_sweep_throughput.png'}`",
        f"- cuTile tile latency: `{OUTDIR / 'cutile_fp16_tile_sweep_latency.png'}`",
        f"- FP16 Pareto tradeoff: `{OUTDIR / 'fp16_pareto_tradeoff.png'}`",
        f"- cuTile FP16 tile Pareto: `{OUTDIR / 'cutile_fp16_pareto_tiles.png'}`",
    ]
    (OUTDIR / "cutile_fp16_optimization_summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


def main() -> None:
    """Generate the FP16-focused tuning artifacts and markdown summary."""

    rows: list[Row] = []
    for size in SIZES:
        a, b = make_inputs(size)
        reference = bench.make_reference(a, b, "float16")
        rows.append(collect_torch(a, b, size))
        rows.extend(collect_triton(a, b, reference, size))
        rows.extend(collect_cutile(a, b, reference, size))

    write_raw(rows)
    comparison = build_comparison(rows)
    plot_grouped(comparison, "tflops", "TFLOP/s", OUTDIR / "comparison_fp16_throughput.png")
    plot_grouped(comparison, "steady_latency_ms", "Steady latency (ms)", OUTDIR / "comparison_fp16_latency.png")
    plot_grouped(comparison, "first_launch_ms", "First launch latency (ms)", OUTDIR / "comparison_fp16_first_launch_latency.png")
    plot_cutile_tiles(rows, "tflops", "TFLOP/s", OUTDIR / "cutile_fp16_tile_sweep_throughput.png")
    plot_cutile_tiles(rows, "steady_latency_ms", "Steady latency (ms)", OUTDIR / "cutile_fp16_tile_sweep_latency.png")
    plot_fp16_pareto_tradeoff(comparison, OUTDIR / "fp16_pareto_tradeoff.png")
    plot_cutile_fp16_pareto_tiles(rows, OUTDIR / "cutile_fp16_pareto_tiles.png")
    write_summary(rows, comparison)
    print(json.dumps({"outdir": str(OUTDIR), "rows": len(rows), "comparison_rows": len(comparison)}, indent=2))


if __name__ == "__main__":
    main()
