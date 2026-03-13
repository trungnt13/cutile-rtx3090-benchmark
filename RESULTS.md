# Benchmark Results

## Summary

This benchmark evaluates cuTile against three references:

- `Torch`: production baseline.
- `Triton`: hand-written kernel baseline with tile tuning.
- `PTX-inline`: readable low-level baseline.

The benchmark answers four questions:

1. Whether cuTile can be latency- and throughput-competitive on Ampere for tuned FP16 and BF16 GEMMs.
2. Whether the compile and first-launch costs are acceptable relative to steady-state performance.
3. Whether a readable PTX baseline explains enough of the low-level behavior to guide optimization work.
4. Whether the current int8 path is correct enough to support performance claims.

## Published System

The current published run was collected on:

- GPU: NVIDIA GeForce RTX 3090, 24 GB, compute capability 8.6
- CPU: AMD Ryzen 7 5800X 8-Core Processor
- Driver: `580.126.20`
- NVIDIA reported CUDA version: `13.0`
- Python: `3.13.12` in the benchmark venv

The machine-readable source of truth is:

- [artifacts/system/benchmark_system.json](artifacts/system/benchmark_system.json)
- [artifacts/system/benchmark_system.md](artifacts/system/benchmark_system.md)

## Benchmark Configuration

- Sizes: `128`, `256`, `512`, `1024` with `M=N=K`
- Dtypes: `float32`, `float16`, `bfloat16`, `int8`
- Full-sweep cuTile/Triton tile families:
  - `16x16x16`
  - `32x32x16`
  - `32x32x32`
  - `64x64x16`
  - `64x64x32`
  - `64x64x64`
  - `128x128x128`
- Warmup / iterations in the full report: `3 / 20`
- PTX phase validation warmup / iterations: `5 / 50`

## Main Findings

### FP16 and BF16

cuTile can be competitive on Ampere when the tile is tuned for the problem size.

- Fastest observed `float16` latency: `0.005 ms` on cuTile at size `128`
- Fastest observed `bfloat16` latency: `0.005 ms` on cuTile at size `128`
- Best FP16-focused tuned cuTile configurations:
  - `128`: `32x32x32`, `0.004 ms`, `0.96 TFLOP/s`
  - `256`: `32x32x32`, `0.005 ms`, `7.45 TFLOP/s`
  - `512`: `64x64x64`, `0.009 ms`, `29.57 TFLOP/s`
  - `1024`: `128x64x64`, `0.035 ms`, `61.80 TFLOP/s`

Artifacts:

- [artifacts/full/comparison_throughput.png](artifacts/full/comparison_throughput.png)
- [artifacts/full/comparison_latency.png](artifacts/full/comparison_latency.png)
- [artifacts/fp16_focus/comparison_fp16_throughput.png](artifacts/fp16_focus/comparison_fp16_throughput.png)
- [artifacts/fp16_focus/fp16_pareto_tradeoff.png](artifacts/fp16_focus/fp16_pareto_tradeoff.png)

### Float32

The PTX-inline baseline is often the fastest of the hand-written non-library paths at small sizes, but it should be read as a scalar baseline, not as evidence that inline PTX is a superior general strategy.

- Fastest observed `float32` latency: `0.007 ms` on PTX-inline at size `128`

Artifacts:

- [artifacts/full/comparison_latency.png](artifacts/full/comparison_latency.png)
- [artifacts/ptx_iterations/ptx_iteration_analysis.md](artifacts/ptx_iterations/ptx_iteration_analysis.md)

### PTX Compile vs Launch vs Steady-State

The benchmark keeps compile cost separate from runtime cost. For the published PTX validation case (`float16`, `1024x1024x1024`):

- Compile time: `19.876 ms`
- First launch time: `0.823 ms`
- Steady-state latency: `0.754 ms`

Artifacts:

- [artifacts/full/ptx_latency_validation_summary.json](artifacts/full/ptx_latency_validation_summary.json)
- [artifacts/nsys/ptx_latency_validation_stats_cuda_gpu_kern_sum.txt](artifacts/nsys/ptx_latency_validation_stats_cuda_gpu_kern_sum.txt)

### Int8 Caveat

The current cuTile int8 path does not preserve exact int32 GEMM semantics.

- `max_err_exact`: `768`
- `max_err_tile_wrapped_i8`: `0`

Interpretation: the current path matches a wrapped-per-tile accumulation model rather than exact int32 accumulation. That is a correctness problem, so int8 performance numbers in this repo are diagnostic only.

Artifacts:

- [investigations/int8_ir/summary.json](investigations/int8_ir/summary.json)
- [investigations/int8_ir/mm_i8.cutileir.txt](investigations/int8_ir/mm_i8.cutileir.txt)
- [investigations/int8_ir/README.md](investigations/int8_ir/README.md)

## Trade-Off Analysis

- cuTile strength: explicit tile control and competitive tuned FP16/BF16 results.
- cuTile weakness: current correctness caveat on int8 and noticeable compile/first-launch cost.
- Triton strength: strong tuned baseline without dropping to handwritten CUDA/PTX source.
- PTX-inline strength: clear low-level reference and useful iteration study.
- PTX-inline weakness: fixed baseline design, not optimized enough to stand in for production kernels.
- Torch strength: production-grade baseline and semantics reference.

## Commands Used

Capture system spec:

```bash
python -m benchmarks.system_info
```

Generate the full report bundle:

```bash
python -m reports.full_report
```

Generate the FP16-focused bundle:

```bash
python -m reports.fp16_focus
```

Validate PTX timing phases:

```bash
python -m benchmarks.ptx_latency_validation
```

Generate the PTX iteration study:

```bash
python -m benchmarks.ptx_iteration_study
```

## Raw Data

- [artifacts/full/benchmark_raw.csv](artifacts/full/benchmark_raw.csv)
- [artifacts/full/benchmark_raw.json](artifacts/full/benchmark_raw.json)
- [artifacts/full/benchmark_coldstart.csv](artifacts/full/benchmark_coldstart.csv)
- [artifacts/full/benchmark_best.csv](artifacts/full/benchmark_best.csv)
- [artifacts/fp16_focus/fp16_raw.csv](artifacts/fp16_focus/fp16_raw.csv)
- [artifacts/ptx_iterations/ptx_iteration_raw.csv](artifacts/ptx_iterations/ptx_iteration_raw.csv)
