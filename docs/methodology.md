# Methodology

## Scope

This repo benchmarks GEMMs on a single RTX 3090 setup across five backends:

- **cuTile** — compiler-generated tile kernels via `cuda.tile`
- **PTX-inline** — fixed 16x16 scalar baseline (CuPy `RawModule`)
- **Triton** — hand-written reference kernel with configurable tiles
- **Torch** — `torch.mm` / `torch._int_mm` (dispatches to cuBLAS internally)
- **cuBLAS** — direct cuBLAS calls via CuPy's cublas wrapper (zero-copy DLPack)

The intent is comparative engineering analysis, not a claim that all backends receive identical optimization effort.

## Shapes

- **Square GEMMs**: M=N=K in {128, 256, 512, 1024, 2048, 4096, 8192}
- **Rectangular GEMMs**: MLP-like (4096x4096x{128,256,512}) and attention-like (512x64x512, 1024x64x1024, 2048x64x2048)
- **Dtypes**: float32, float16, bfloat16, int8

## Timing Policy

The benchmark separates three phases:

1. **Compile time** — host wall time for JIT compilation (cuTile, PTX, Triton) or 0 (Torch, cuBLAS)
2. **First launch time** — host wall time for the first kernel invocation after compilation
3. **Steady-state latency** — per-iteration GPU time via CUDA events after warmup

Steady-state timing returns both mean and standard deviation across iterations. Mixing all three phases into one number would hide whether a backend is slow because of runtime work or one-time overhead.

The steady-state helpers in `benchmarks/core.py` warm up before recording CUDA events. Compile and first-launch timings use host wall time with explicit synchronization on both Torch and CuPy stream views.

## Statistical Reporting

All steady-state measurements report (mean, stddev) over per-iteration CUDA event pairs. Error bars appear on comparison plots. The % of peak throughput is computed against RTX 3090 theoretical peaks:

| Dtype | Peak |
|-------|------|
| float16 | 142.0 TFLOP/s |
| bfloat16 | 142.0 TFLOP/s |
| float32 | 35.6 TFLOP/s |
| int8 | 284.0 TOP/s |

## Correctness Policy

### Floating-Point

Floating-point references are computed with TF32 disabled:

```python
torch.backends.cuda.matmul.allow_tf32 = False
return torch.mm(a.float(), b.float())
```

That policy matters because Ampere defaults can make a fast path look more accurate than it really is relative to a strict reference.

### Int8

The int8 path is checked against two references:

- Exact `torch._int_mm` int32 accumulation
- A wrapped-per-tile reference that models the behavior seen in the exported cuTile IR

Reports carry both `max_err_exact` and `max_err_tile_wrapped_i8`.

## Tile Policy

The public full benchmark compares a fixed set of named cuTile tile families. That keeps the charts readable and makes size-to-size trade-offs interpretable.

The PTX-inline baseline intentionally keeps a fixed 16x16 tile. A separate PTX iteration study explores alternative PTX designs without conflating them with the main charts.

## Cold-Start Consistency

- **cuTile / Triton**: `compile_ms` = JIT compilation of the first call; `first_launch_ms` = second call (post-compilation)
- **Torch / cuBLAS**: `compile_ms = 0.0` (no JIT); `first_launch_ms` = first kernel invocation

## Published Artifact Flow

1. `python -m benchmarks.system_info`
2. `python -m reports.full_report`
3. `python -m reports.fp16_focus`
4. `python -m benchmarks.ptx_latency_validation`
5. `python -m benchmarks.ptx_iteration_study`
6. `python -m reports.roofline` (reads artifacts, no GPU needed)
7. `python -m benchmarks.cost_model` (reads artifacts, no GPU needed)
8. `python -m benchmarks.ncu_profile` (requires ncu + elevated permissions)

Outputs are written under `artifacts/`. Supporting int8 evidence lives under `investigations/int8_ir/`.
