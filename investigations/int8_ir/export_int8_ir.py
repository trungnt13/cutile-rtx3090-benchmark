#!/usr/bin/env python3

"""Reproduce and export the cuTile int8 IR used in the benchmark caveat section."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cupy as cp
import cuda.tile as ct
import torch
from cuda.tile._compile import compile_tile, default_tile_context
from cuda.tile._compiler_options import CompilerOptions


def parse_args() -> argparse.Namespace:
    """Parse arguments for the int8 IR export repro."""

    parser = argparse.ArgumentParser(description="Repro and export IRs for cuTile int8 matmul.")
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--tile-m", type=int, default=32)
    parser.add_argument("--tile-n", type=int, default=32)
    parser.add_argument("--tile-k", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    return parser.parse_args()


def make_chunk_wrapped_i8_reference(a: torch.Tensor, b: torch.Tensor, tile_k: int) -> torch.Tensor:
    """Model the wrapped accumulation behavior visible in the exported cuTile IR."""

    total = torch.zeros((a.shape[0], b.shape[1]), device="cuda", dtype=torch.int32)
    for k0 in range(0, a.shape[1], tile_k):
        partial = torch._int_mm(a[:, k0:k0 + tile_k].contiguous(), b[k0:k0 + tile_k, :].contiguous())
        total += partial.to(torch.int8).to(torch.int32)
    return total


def main() -> None:
    """Export bytecode, TileIR, and summary files for the int8 correctness investigation."""

    args = parse_args()
    outdir = args.outdir.expanduser().resolve()
    bytecode_dir = outdir / "bytecode"
    mlir_dir = outdir / "mlir"
    bytecode_dir.mkdir(parents=True, exist_ok=True)
    mlir_dir.mkdir(parents=True, exist_ok=True)

    os.environ["CUDA_TILE_DUMP_BYTECODE"] = str(bytecode_dir)
    os.environ["CUDA_TILE_DUMP_TILEIR"] = str(mlir_dir)

    @ct.kernel
    def mm_i8(a, b, c, tile_m: ct.Constant[int], tile_n: ct.Constant[int], tile_k: ct.Constant[int], k_tiles: ct.Constant[int]):
        bid_m = ct.bid(0)
        bid_n = ct.bid(1)
        acc = ct.zeros((tile_m, tile_n), dtype=ct.int32)
        for kk in range(k_tiles):
            a_tile = ct.load(a, index=(bid_m, kk), shape=(tile_m, tile_k))
            b_tile = ct.load(b, index=(kk, bid_n), shape=(tile_k, tile_n))
            acc = acc + ct.matmul(a_tile, b_tile)
        ct.store(c, index=(bid_m, bid_n), tile=acc)

    torch.manual_seed(args.seed)
    a = torch.randint(-8, 8, (args.m, args.k), device="cuda", dtype=torch.int8)
    b = torch.randint(-8, 8, (args.k, args.n), device="cuda", dtype=torch.int8)
    c = torch.empty((args.m, args.n), device="cuda", dtype=torch.int32)

    grid = (ct.cdiv(args.m, args.tile_m), ct.cdiv(args.n, args.tile_n), 1)
    k_tiles = ct.cdiv(args.k, args.tile_k)
    stream = cp.cuda.get_current_stream()
    ct.launch(stream, grid, mm_i8, (a, b, c, args.tile_m, args.tile_n, args.tile_k, k_tiles))
    stream.synchronize()

    # Export the final cuTile IR text directly from the compiler pipeline.
    tile_lib = compile_tile(
        mm_i8._pyfunc,
        (a, b, c, args.tile_m, args.tile_n, args.tile_k, k_tiles),
        CompilerOptions(),
        default_tile_context,
    )
    (outdir / "mm_i8.cutileir.txt").write_text(
        tile_lib.final_ir.to_string(include_loc=False) + "\n",
        encoding="utf-8",
    )

    ref_exact = torch._int_mm(a, b)
    ref_tile_wrapped = make_chunk_wrapped_i8_reference(a, b, args.tile_k)

    summary = {
        "shape": [args.m, args.n, args.k],
        "tile": [args.tile_m, args.tile_n, args.tile_k],
        "grid": list(grid),
        "k_tiles": int(k_tiles),
        "max_err_exact": int((c - ref_exact).abs().max().item()),
        "max_err_tile_wrapped_i8": int((c - ref_tile_wrapped).abs().max().item()),
        "sample_c_row0": c[0, :8].cpu().tolist(),
        "sample_ref_exact_row0": ref_exact[0, :8].cpu().tolist(),
        "sample_ref_tile_wrapped_row0": ref_tile_wrapped[0, :8].cpu().tolist(),
        "bytecode_dir": str(bytecode_dir),
        "mlir_dir": str(mlir_dir),
        "bytecode_files": sorted(p.name for p in bytecode_dir.iterdir()),
        "mlir_files": sorted(p.name for p in mlir_dir.iterdir()),
        "cutile_ir_file": str(outdir / "mm_i8.cutileir.txt"),
    }

    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"summary_path: {summary_path}")


if __name__ == "__main__":
    main()
