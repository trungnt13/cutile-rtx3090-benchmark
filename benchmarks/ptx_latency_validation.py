#!/usr/bin/env python3

"""Validate PTX compile, first-launch, and steady-state timing as separate phases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from benchmarks import core


def parse_args() -> argparse.Namespace:
    """Parse arguments for the PTX latency phase-separation validation."""

    parser = argparse.ArgumentParser(description="Validate PTX latency separation: compile, first launch, steady-state.")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16", "int8"), default="float16")
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=core.FULL_ARTIFACTS_DIR / "ptx_latency_validation_summary.json")
    return parser.parse_args()


def main() -> None:
    """Record phase-separated PTX timings and annotate them with NVTX ranges.

    NVTX markers are intentionally aligned to compile, first-launch, and steady-state so
    Nsight Systems traces can verify the benchmark policy rather than only the final number.
    """

    args = parse_args()
    info = core.DTYPE_INFO[args.dtype]

    a, b = core.make_inputs(args.dtype, args.m, args.n, args.k, seed=args.seed)
    c = core.torch.empty((args.m, args.n), device="cuda", dtype=info["out"])

    compiled: dict[str, object] = {}

    core.torch.cuda.nvtx.range_push("ptx_compile")
    compile_ms = core.measure_host_ms(lambda: compiled.setdefault("kernel", core.compile_ptx(args.dtype)))
    core.torch.cuda.nvtx.range_pop()

    kernel = compiled["kernel"]

    core.torch.cuda.nvtx.range_push("ptx_first_launch")
    first_launch_ms = core.measure_host_ms(lambda: core.run_ptx(kernel, a, b, c, args.m, args.n, args.k))
    core.torch.cuda.nvtx.range_pop()

    core.torch.cuda.nvtx.range_push("ptx_steady_state")
    steady_timing = core.benchmark_ms_cupy(
        lambda: core.run_ptx(kernel, a, b, c, args.m, args.n, args.k),
        args.warmup,
        args.iters,
    )
    core.torch.cuda.nvtx.range_pop()

    summary = {
        "dtype": args.dtype,
        "shape": [args.m, args.n, args.k],
        "compile_ms": compile_ms,
        "first_launch_ms": first_launch_ms,
        "steady_latency_ms": steady_timing.mean,
        "steady_latency_std_ms": steady_timing.std,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
