#!/usr/bin/env python3

"""Capture benchmark machine and runtime details as reproducibility artifacts."""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path


if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from benchmarks import core


def run_text(*cmd: str) -> str | None:
    """Run a command and return stripped stdout, or None on failure."""

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def package_version(name: str) -> str | None:
    """Return the installed version for a Python package if present."""

    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def total_memory_bytes() -> int | None:
    """Return system memory in bytes when available from POSIX sysconf."""

    if not hasattr(os, "sysconf"):
        return None
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        return None
    if isinstance(page_size, int) and isinstance(phys_pages, int):
        return page_size * phys_pages
    return None


def nvidia_smi_table() -> str | None:
    """Return the first lines of `nvidia-smi` for human-readable provenance."""

    output = run_text("nvidia-smi")
    if output is None:
        return None
    return "\n".join(output.splitlines()[:12])


def nvidia_driver_and_cuda() -> tuple[str | None, str | None]:
    """Extract driver and reported CUDA version from `nvidia-smi` output."""

    output = nvidia_smi_table()
    if output is None:
        return None, None
    driver = None
    cuda = None
    for line in output.splitlines():
        if "Driver Version:" in line and "CUDA Version:" in line:
            parts = line.split("Driver Version:", 1)[1].strip()
            driver, cuda_part = parts.split("CUDA Version:", 1)
            driver = driver.strip().strip("|").strip()
            cuda = cuda_part.strip().strip("|").strip()
            break
    return driver, cuda


def collect_system_info() -> dict[str, object]:
    """Collect system, GPU, Python, and package metadata for benchmark provenance."""

    driver_version, nvidia_smi_cuda = nvidia_driver_and_cuda()
    gpu_name = core.torch.cuda.get_device_name(0) if core.torch.cuda.is_available() else None
    gpu_props = core.torch.cuda.get_device_properties(0) if core.torch.cuda.is_available() else None

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "project_root": str(core.PROJECT_ROOT),
        "git_commit": run_text("git", "-C", str(core.PROJECT_ROOT), "rev-parse", "HEAD"),
        "os": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "kernel": run_text("uname", "-a"),
        },
        "cpu": {
            "processor": platform.processor() or None,
            "machine": platform.machine(),
            "python_cpu_count": os.cpu_count(),
            "lscpu": run_text("lscpu"),
        },
        "memory": {
            "total_bytes": total_memory_bytes(),
        },
        "gpu": {
            "name": gpu_name,
            "total_memory_bytes": getattr(gpu_props, "total_memory", None),
            "compute_capability": (
                f"{gpu_props.major}.{gpu_props.minor}" if gpu_props is not None else None
            ),
            "multi_processor_count": getattr(gpu_props, "multi_processor_count", None),
            "nvidia_smi_overview": nvidia_smi_table(),
            "driver_version": driver_version,
            "nvidia_smi_reported_cuda_version": nvidia_smi_cuda,
            "torch_cuda_version": core.torch.version.cuda,
        },
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "packages": {
            "torch": package_version("torch"),
            "triton": package_version("triton"),
            "cupy-cuda13x": package_version("cupy-cuda13x"),
            "cuda-tile": package_version("cuda-tile"),
            "cuda-bindings": package_version("cuda-bindings"),
            "numpy": package_version("numpy"),
            "matplotlib": package_version("matplotlib"),
        },
        "toolchain": {
            "nvcc_version": run_text("nvcc", "--version"),
        },
    }


def write_markdown(payload: dict[str, object], out_path: Path) -> None:
    """Write a short Markdown companion for the JSON system spec artifact."""

    gpu = payload["gpu"]
    python_info = payload["python"]
    packages = payload["packages"]
    git_commit = payload["git_commit"] or "unavailable (workspace is not a git repo)"
    md = [
        "# Benchmark System Specification",
        "",
        f"- Timestamp (UTC): `{payload['timestamp_utc']}`",
        f"- Hostname: `{payload['hostname']}`",
        f"- Project root: `{payload['project_root']}`",
        f"- Git commit: `{git_commit}`",
        f"- OS: `{payload['os']['platform']}`",
        f"- CPU: `{payload['cpu']['machine']}` / logical CPUs `{payload['cpu']['python_cpu_count']}`",
        f"- GPU: `{gpu['name']}`",
        f"- GPU memory: `{gpu['total_memory_bytes']}` bytes",
        f"- Compute capability: `{gpu['compute_capability']}`",
        f"- NVIDIA driver: `{gpu['driver_version']}`",
        f"- NVIDIA reported CUDA version: `{gpu['nvidia_smi_reported_cuda_version']}`",
        f"- Torch CUDA version: `{gpu['torch_cuda_version']}`",
        f"- Python: `{python_info['version']}` at `{python_info['executable']}`",
        "",
        "## Key packages",
        "",
        *[f"- `{name}`: `{version}`" for name, version in packages.items()],
        "",
        "## nvcc",
        "",
        "```text",
        payload["toolchain"]["nvcc_version"] or "nvcc not found on PATH",
        "```",
        "",
        "## lscpu",
        "",
        "```text",
        payload["cpu"]["lscpu"] or "lscpu unavailable",
        "```",
    ]
    out_path.write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    """Collect and write machine/runtime provenance artifacts."""

    core.SYSTEM_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = collect_system_info()
    json_path = core.SYSTEM_ARTIFACTS_DIR / "benchmark_system.json"
    md_path = core.SYSTEM_ARTIFACTS_DIR / "benchmark_system.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(payload, md_path)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
