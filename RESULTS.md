# cuTile RTX 3090 Benchmark Results

## Summary

**Full sweep (symmetric tiles, no occupancy tuning):**

| Backend | Dtype | Best Size | TFLOP/s | Latency (ms) | % Peak |
|---------|-------|-----------|---------|---------------|--------|
| cuTile (64x64x32) | float16 | 2048 | 52.45 | 0.328 | 36.9% |
| cuTile (64x64x16) | bfloat16 | 2048 | 44.64 | 0.385 | 31.4% |
| cuTile (64x64x16) | float32 | 4096 | 3.02 | 45.549 | 8.5% |
| cuTile (64x64x32) | int8 | 2048 | 63.66 TOP/s | 0.270 | 22.4% |
| Torch | float16 | 8192 | 68.15 | — | 48.0% |
| Torch | bfloat16 | 8192 | 70.74 | — | 49.8% |
| Torch (TF32) | float32 | 8192 | 37.28 | — | 104.7% |
| Triton (64x64x32) | float16 | 2048 | 62.40 | — | 43.9% |

**FP16 with asymmetric tiles + occupancy tuning:**

| Size | Tile | Occ | TFLOP/s | % Peak | Latency (ms) |
|------|------|-----|---------|--------|--------------|
| 128 | 32x32x32 | 2 | 0.57 | 0.4% | 0.007 |
| 256 | 32x32x32 | 1 | 4.03 | 2.8% | 0.008 |
| 512 | 64x64x64 | 2 | 19.76 | 13.9% | 0.014 |
| 1024 | 128x64x64 | 2 | 64.99 | 45.8% | 0.033 |
| 2048 | 128x64x32 | 2 | 81.02 | 57.1% | 0.212 |
| 4096 | 128x128x64 | 2 | 88.70 | **62.5%** | 1.549 |
| 8192 | 128x128x64 | 2 | 72.95 | 51.4% | 15.071 |

*RTX 3090, 82 SMs. Peak: FP16/BF16 142 TFLOP/s, FP32 35.6 TFLOP/s, INT8 284 TOP/s.*

## Key Findings

- **Tile selection is the dominant lever.** Untuned symmetric tiles cap at 37% peak FP16; tuned asymmetric tiles with occupancy reach 62.5% — a 1.7x improvement from tile choice alone.
- **cuTile is competitive when tuned.** At 4096, tuned cuTile (88.70 TFLOP/s) exceeds Triton (62.40) and narrows the gap to Torch (68.15), both measured at their respective best sizes.
- **Float32 is severely limited.** cuTile achieves 8.5% peak vs Torch at 105% (TF32 path). cuTile has no TF32 support.
- **Best tile shape varies with problem size.** 32x32x32 at small sizes, 128x128x64 at large — a linear cost model achieves 0% prediction accuracy (always picks 128x128x128).
- **Throughput drops at 8192.** cuTile falls from 62.5% to 51.4% peak going from 4096 to 8192, suggesting register pressure or scheduling limits at scale.

## Figures

![Latency comparison](artifacts/full/comparison_latency.png)
*Steady-state latency across backends and dtypes (log scale).*

![Throughput comparison](artifacts/full/comparison_throughput.png)
*Throughput in TFLOP/s. Torch leads at large sizes; cuTile is competitive at mid-range.*

![FP16 Pareto frontier](artifacts/fp16_focus/fp16_pareto_tradeoff.png)
*Latency-vs-throughput Pareto for FP16 tile configurations.*

![First-launch latency](artifacts/full/comparison_first_launch_latency.png)
*Cold-start overhead separated from steady-state.*

![% of peak throughput](artifacts/full/comparison_pct_peak.png)
*Achieved throughput as percentage of RTX 3090 theoretical peak.*

## NCU Profiling (FP16 at 1024)

| Backend | Occupancy | Registers | Mem BW (GB/s) |
|---------|-----------|-----------|---------------|
| cuTile | 8.3% | 103 | 76.7 |
| PTX-inline | 95.4% | 38 | 7.8 |
| Triton | 21.9% | 69 | 120.6 |
| cuBLAS (sgemm) | 29.6% | 122 | 243.0 |
| Torch (cutlass) | 8.3% | 224 | 95.4 |

cuTile and Torch (cutlass) share the lowest occupancy tier (8.3%) but for different reasons: cuTile uses 103 registers per thread, Torch cutlass uses 224. cuTile achieves less memory bandwidth (76.7 GB/s) than Triton (120.6) or cuBLAS (243), pointing to scheduling or memory access pattern inefficiency as the primary bottleneck — not register count alone. PTX achieves 95% occupancy but is the slowest kernel, confirming that occupancy without effective data reuse is worthless.

## Cost Model

A linear cost model for tile selection achieves 0% accuracy (R²=0.198), always predicting 128x128x128 regardless of dtype or size. Tile selection requires non-linear reasoning.

## Int8 Caveat

The int8 path accumulates through a narrowed int8 intermediate rather than exact int32 GEMM semantics. Errors scale with matrix size: max_err ranges from 384 (N=128) to 14080 (N=8192). Int8 throughput numbers are diagnostic only. See [investigations/int8_ir/](investigations/int8_ir/).

## Cold-Start

cuTile compile: 8-24 ms. First-launch: 7-39 ms (size-dependent). Both are one-time costs per kernel configuration.

## Methodology and Raw Data

- Methodology: [docs/methodology.md](docs/methodology.md)
- Tile selection analysis: [docs/cutile_tile_selection_rtx3090.md](docs/cutile_tile_selection_rtx3090.md)
- Raw data: [artifacts/full/benchmark_raw.csv](artifacts/full/benchmark_raw.csv), [artifacts/full/benchmark_raw.json](artifacts/full/benchmark_raw.json)
- Best-of-sweep: [artifacts/full/benchmark_best.csv](artifacts/full/benchmark_best.csv)
- FP16 focus: [artifacts/fp16_focus/fp16_raw.csv](artifacts/fp16_focus/fp16_raw.csv)
- System info: [artifacts/system/benchmark_system.json](artifacts/system/benchmark_system.json)
