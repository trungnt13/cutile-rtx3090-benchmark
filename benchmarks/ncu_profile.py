#!/usr/bin/env python3

"""Collect NVIDIA Nsight Compute (ncu) profiling metrics for selected best configs.

If ncu requires elevated permissions, this script generates shell commands for manual execution.
Otherwise, it runs ncu directly and collects results.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from benchmarks.core import FULL_ARTIFACTS_DIR, PROJECT_ROOT


NCU_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "ncu"

# Best configs per dtype from the benchmark sweep (can be updated from benchmark_best.csv)
PROFILE_CONFIGS = [
    {"dtype": "float16", "m": 1024, "n": 1024, "k": 1024, "tile": "128,64,64"},
    {"dtype": "float16", "m": 512, "n": 512, "k": 512, "tile": "64,64,64"},
    {"dtype": "bfloat16", "m": 1024, "n": 1024, "k": 1024, "tile": "64,64,16"},
    {"dtype": "float32", "m": 1024, "n": 1024, "k": 1024, "tile": "64,64,16"},
]

# ncu metrics of interest
NCU_METRICS = [
    "sm__warps_active.avg.pct_of_peak_sustained_active",  # achieved occupancy
    "dram__bytes.sum.per_second",  # memory bandwidth
    "launch__registers_per_thread",  # registers per thread
    "sm__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "sm__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
]


def find_ncu() -> str | None:
    """Locate ncu binary."""
    # Check common paths
    for path in ["/usr/local/cuda-13.2/bin/ncu", "/usr/local/cuda/bin/ncu"]:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return shutil.which("ncu")


def build_ncu_command(ncu_path: str, config: dict, output_prefix: str) -> list[str]:
    """Build the ncu command line for a given config."""
    dtype = config["dtype"]
    m, n, k = config["m"], config["n"], config["k"]
    tile = config["tile"]
    metrics_str = ",".join(NCU_METRICS)

    return [
        ncu_path,
        "--metrics", metrics_str,
        "--csv",
        "--target-processes", "all",
        "--set", "full",
        "-o", output_prefix,
        "python", "-m", "benchmarks.matmul_sweep",
        "--dtype", dtype,
        "--sizes", f"{m},{n},{k}",
        "--tile-sweep", tile,
        "--iters", "3",
        "--warmup", "1",
    ]


def main() -> None:
    NCU_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    ncu_path = find_ncu()

    if ncu_path is None:
        print("ncu not found. Generating shell commands for manual execution.")
        commands = []
        for i, config in enumerate(PROFILE_CONFIGS):
            output_prefix = str(NCU_ARTIFACTS_DIR / f"profile_{config['dtype']}_{config['m']}")
            cmd = build_ncu_command("/usr/local/cuda-13.2/bin/ncu", config, output_prefix)
            commands.append({"config": config, "command": " ".join(cmd)})
            # Also generate sudo version
            commands.append({"config": config, "command": "sudo " + " ".join(cmd), "note": "elevated permissions"})

        script_path = NCU_ARTIFACTS_DIR / "ncu_commands.json"
        script_path.write_text(json.dumps(commands, indent=2), encoding="utf-8")

        shell_path = NCU_ARTIFACTS_DIR / "run_ncu.sh"
        with shell_path.open("w") as f:
            f.write("#!/bin/bash\n# Generated ncu profiling commands\nset -e\n\n")
            for entry in commands:
                if "note" not in entry:
                    cfg = entry["config"]
                    f.write(f"echo 'Profiling {cfg['dtype']} {cfg['m']}...'\n")
                    f.write(f"sudo {entry['command']}\n\n")
        shell_path.chmod(0o755)
        print(f"Commands written to {script_path}")
        print(f"Shell script written to {shell_path}")
        return

    # Try to run ncu directly
    results = []
    for config in PROFILE_CONFIGS:
        output_prefix = str(NCU_ARTIFACTS_DIR / f"profile_{config['dtype']}_{config['m']}")
        cmd = build_ncu_command(ncu_path, config, output_prefix)
        print(f"Running: {' '.join(cmd[:6])}...")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(PROJECT_ROOT))
            if result.returncode != 0:
                print(f"ncu returned {result.returncode}. May need sudo. stderr: {result.stderr[:200]}")
                # Generate manual commands instead
                results.append({"config": config, "status": "needs_sudo", "command": "sudo " + " ".join(cmd)})
            else:
                results.append({"config": config, "status": "success", "output": output_prefix})
        except subprocess.TimeoutExpired:
            results.append({"config": config, "status": "timeout"})
        except Exception as e:
            results.append({"config": config, "status": "error", "error": str(e)})

    summary_path = NCU_ARTIFACTS_DIR / "ncu_profile_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
