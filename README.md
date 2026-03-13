# cuTile Benchmark

Public benchmark artifact comparing cuTile, a simple Parallel Thread Execution (PTX)-inline baseline, Triton, and Torch matmul implementations on NVIDIA Ampere.

This repo is benchmark-first, not library-first. Its goal is to answer a narrow set of questions clearly:

- Can tuned cuTile compete on latency and throughput for practical half-precision floating-point (FP16) and brain floating point (BF16) general matrix multiplications (GEMMs) on RTX 3090?
- How much of the gap to Torch and Triton comes from kernel design choices versus compile and launch overhead?
- Which parts of the current cuTile stack are promising, and which parts still fail correctness or product-readiness standards?

The current published run was collected on an NVIDIA GeForce RTX 3090 graphics processing unit (GPU) with an AMD Ryzen 7 5800X central processing unit (CPU). The artifact bundle includes raw comma-separated values (CSV) and JavaScript Object Notation (JSON) exports, summary plots, an int8 intermediate representation (IR) investigation, and a machine-readable system specification.

## Headline Findings

- cuTile is competitive on FP16 and BF16 when tile shapes are tuned for the target size.
- The PTX-inline baseline is useful as a readable low-level reference, but it is not a library-grade competitor to Triton or Torch.
- The current int8 cuTile path is not semantically equivalent to exact int32 GEMM accumulation; the repo treats that as a correctness caveat, not a footnote.
- Cold-start and steady-state costs are reported separately so compile latency does not get hidden inside throughput claims.

## Repository Layout

```text
.
├── README.md
├── RESULTS.md
├── requirements.txt
├── benchmarks/
├── reports/
├── docs/
├── artifacts/
├── investigations/
└── examples/
```

- `benchmarks/`: source-of-truth benchmark runners and shared runtime helpers.
- `reports/`: report generation scripts that read benchmark outputs and emit plots/markdown.
- `artifacts/`: generated benchmark outputs for the published run.
- `investigations/int8_ir/`: supporting evidence for the int8 correctness caveat.
- `docs/`: methodology, implementation notes, and trade-off analysis.

## Environment Setup

This benchmark was run on Linux with an NVIDIA GPU stack and a Python virtual environment. Reuse the existing `.venv` when working on the benchmark host, or create a fresh environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The pinned environment used for the published run is summarized in [artifacts/system/benchmark_system.md](artifacts/system/benchmark_system.md).

## Core Commands

Capture system provenance:

```bash
python -m benchmarks.system_info
```

Run the main sweep:

```bash
python -m benchmarks.matmul_sweep --dtype float16 --sizes "128;256;512;1024" --tile-sweep "16,16,16;32,32,16;32,32,32;64,64,16;64,64,32;64,64,64;128,128,128"
```

Regenerate the full benchmark report:

```bash
python -m reports.full_report
```

Regenerate the FP16-focused report:

```bash
python -m reports.fp16_focus
```

Validate PTX phase separation:

```bash
python -m benchmarks.ptx_latency_validation
```

Run the PTX iteration study:

```bash
python -m benchmarks.ptx_iteration_study
```

## Benchmark Policy

- Steady-state latency is measured with Compute Unified Device Architecture (CUDA) events after warmup iterations.
- Compile time and first-launch time are measured separately with host wall time.
- Floating-point correctness references disable TensorFloat-32 (TF32) so the comparison is against a stricter accumulator path.
- PTX-inline uses a fixed 16x16 tiled baseline in the main comparison. That is deliberate: the PTX path is meant to be understandable, not maximally optimized.
- Int8 results are reported with both exact-int32 and wrapped-per-tile comparison metrics because the current cuTile path does not preserve exact int32 accumulation semantics.

## Code Structure

The shared benchmark logic lives in `benchmarks/core.py`. The split is intentional:

```python
def benchmark_ms_cupy(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    cp.cuda.get_current_stream().synchronize()
    start = cp.cuda.Event()
    stop = cp.cuda.Event()
    start.record()
```

That helper exists so steady-state timing excludes just-in-time (JIT) compilation and lazy initialization costs. Compile and first-launch timings are reported by separate host-side helpers rather than being mixed into the GPU event number.

The main CLI stays thin:

```python
result = core.benchmark_tile_config(
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
```

That keeps policy decisions readable while centralizing the backend-specific runtime logic in one place.

## Read Next

- [RESULTS.md](RESULTS.md)
- [docs/methodology.md](docs/methodology.md)
- [docs/implementation.md](docs/implementation.md)
- [docs/tradeoffs.md](docs/tradeoffs.md)
- [investigations/int8_ir/README.md](investigations/int8_ir/README.md)
