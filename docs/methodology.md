# Methodology

## Scope

This repo benchmarks square general matrix multiplications (GEMMs) on a single RTX 3090-focused setup across four paths:

- cuTile
- Parallel Thread Execution (PTX)-inline baseline
- Triton
- Torch

The intent is comparative engineering analysis, not a claim that all backends receive identical optimization effort.

## Timing Policy

The benchmark separates three phases:

1. Compile time
2. First launch time
3. Steady-state latency

That split is deliberate. Mixing all three into one number would hide whether a backend is slow because of runtime work or because of one-time compilation and initialization overhead.

The steady-state helpers in `benchmarks/core.py` warm up before recording Compute Unified Device Architecture (CUDA) events. Compile and first-launch timings use host wall time with explicit synchronization on both Torch and CuPy stream views.

## Correctness Policy

### Floating-Point

Floating-point references are computed with TensorFloat-32 (TF32) disabled:

```python
prev = torch.backends.cuda.matmul.allow_tf32
torch.backends.cuda.matmul.allow_tf32 = False
try:
    return torch.mm(a.float(), b.float())
finally:
    torch.backends.cuda.matmul.allow_tf32 = prev
```

That policy matters because Ampere defaults can make a fast path look more accurate than it really is relative to a strict reference.

### Int8

The int8 path is checked against two references:

- Exact `torch._int_mm` int32 accumulation
- A wrapped-per-tile reference that models the behavior seen in the exported cuTile intermediate representation (IR)

This is why the reports carry both `max_err_exact` and `max_err_tile_wrapped_i8`.

## Tile Policy

The public full benchmark compares a fixed set of named cuTile tile families. That keeps the charts readable and makes size-to-size trade-offs interpretable.

The PTX-inline baseline intentionally keeps a fixed 16x16 tile in the main benchmark. This makes it an understandable baseline rather than a moving-target hand-tuned competitor. A separate PTX iteration study explores alternative PTX designs without conflating them with the main benchmark charts.

## Published Artifact Flow

1. `python -m benchmarks.system_info`
2. `python -m reports.full_report`
3. `python -m reports.fp16_focus`
4. `python -m benchmarks.ptx_latency_validation`
5. `python -m benchmarks.ptx_iteration_study`

The outputs are written under `artifacts/`, while supporting int8 evidence lives under `investigations/int8_ir/`.
