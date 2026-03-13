# Implementation Details

## Benchmark Architecture

The repo is split into:

- `benchmarks/core.py`: shared runtime helpers, kernel builders, timing helpers, and correctness references
- `benchmarks/*.py`: benchmark entrypoints
- `reports/*.py`: aggregation, plots, and markdown summaries

This separation keeps benchmark execution logic independent from reporting logic. Raw measurements are collected first, then reports transform them into public artifacts.

## Backend Execution Paths

### cuTile

The cuTile kernel is generated in Python and launched with explicit tile metadata:

```python
def run_cutile(kernel, a, b, c, m, n, k, tile_m, tile_n, tile_k):
    grid = (ct.cdiv(m, tile_m), ct.cdiv(n, tile_n), 1)
    k_tiles = ct.cdiv(k, tile_k)
    ct.launch(cp.cuda.get_current_stream(), grid, kernel, (a, b, c, tile_m, tile_n, tile_k, k_tiles))
```

The grid policy is tied directly to tile dimensions so the benchmark can study how tile shape changes latency and throughput.

### PTX-inline

The PTX path is intentionally simple. In the full benchmark it is a fixed 16x16 scalar tiled kernel that serves as a readable baseline. A separate PTX iteration study expands that into:

- scalar 16x16
- scalar 32x32
- WMMA 32x32 tensor-core path

That split lets the repo keep the main report readable while still documenting how PTX design choices affect results.

### Triton

Triton serves as the hand-written kernel baseline with a small tile search. The benchmark uses it both in the full sweep and in a tighter FP16-focused tuning study.

## Reasoning Behind the Shared Helpers

### Timing helpers

`benchmark_ms_cupy`, `benchmark_ms_torch`, and `measure_host_ms` exist because timing policy is load-bearing in this repo. They separate:

- GPU event time for steady-state work
- host-visible time for compilation
- host-visible time for first launch

### Reference helpers

`make_reference` and `make_chunk_wrapped_i8_reference` exist because correctness is not a single-number question for the current int8 path. The benchmark must distinguish:

- exact GEMM semantics
- the wrapped behavior actually produced by the current cuTile IR

### Shared comparison helper

`benchmark_tile_config` launches all backends once, computes errors, and only then runs the timed loops. That order prevents lazy initialization from contaminating correctness or steady-state numbers.

## Report Generators

`reports/full_report.py` produces:

- full raw data exports
- best-row summaries
- full comparison charts
- cold-start summaries

`reports/fp16_focus.py` produces:

- a denser FP16-only tile/occupancy sweep
- Pareto charts for latency vs throughput
- a tuned cuTile summary oriented around size-specific best configs
