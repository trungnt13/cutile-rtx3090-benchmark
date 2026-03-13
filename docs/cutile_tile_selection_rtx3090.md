# Tile Selection on RTX 3090: Why the Best Tile Changes with Size

cuTile has no single best tile. The winner depends on workload scale, tile symmetry, and occupancy tuning.

## Summary Table (FP16, square GEMM M=N=K)

| Size | Tuned Tile | TFLOP/s | % Peak | Untuned Tile | TFLOP/s | % Peak | Speedup |
|------|-----------|---------|--------|-------------|---------|--------|---------|
| 128 | 32x32x32 | 0.57 | 0.4% | 32x32x32 | 0.57 | 0.4% | 1.00x |
| 256 | 32x32x32 | 4.03 | 2.8% | 32x32x32 | 4.07 | 2.9% | 0.99x |
| 512 | 64x64x64 | 19.76 | 13.9% | 64x64x32 | 18.40 | 13.0% | 1.07x |
| 1024 | 128x64x64 | 64.99 | 45.8% | 64x64x16 | 30.13 | 21.2% | 2.16x |
| 2048 | 128x64x32 | 81.02 | 57.1% | 64x64x32 | 52.45 | 36.9% | 1.54x |
| 4096 | 128x128x64 | 88.70 | 62.5% | 64x64x64 | 52.11 | 36.7% | 1.70x |
| 8192 | 128x128x64 | 72.95 | 51.4% | 64x64x64 | 44.60 | 31.4% | 1.64x |

"Tuned" = asymmetric tiles + occupancy tuning. "Untuned" = symmetric tiles only, no occupancy hints.

## Three Forces

Every tile choice balances three competing pressures:

1. **CTA count** `(M/tile_m) * (N/tile_n)` -- must cover 82 SMs; too few CTAs starve the GPU
2. **K-loop depth** `K/tile_k` -- fewer iterations means less loop overhead and better data reuse
3. **Per-CTA footprint** -- larger tiles consume more registers/shared memory, reducing scheduling flexibility

## Size-by-Size Analysis

**128 (32x32x32, 16 CTAs, 4 K-tiles):** Launch overhead dominates. Only 0.4% peak regardless of tile choice. All configs are tightly clustered; tuning cannot help at this scale.

**256 (32x32x32, 64 CTAs, 8 K-tiles):** Grid coverage matters. 64 CTAs finally distribute across 82 SMs. Larger tiles collapse CTA count too aggressively. Tuned and untuned agree.

**512 (64x64x64, 64 CTAs, 8 K-tiles):** Reuse starts to dominate. 64 CTAs still cover the GPU while halving K-iterations vs 32x32x32. Tuned picks tile_k=64 over untuned's 32 -- deeper K-loop amortization at this scale.

**1024 (128x64x64, 128 CTAs, 16 K-tiles):** The tuning gap explodes. Asymmetric 128x64x64 doubles output work per CTA vs the untuned 64x64x16 winner. 2.16x speedup -- the largest gap in the sweep.

**2048 (128x64x32, 512 CTAs, 64 K-tiles):** Asymmetric tile with small tile_k=32. The kernel needs 512 CTAs for grid saturation at this scale; tuned config finds the right footprint-to-parallelism tradeoff. 1.54x over untuned.

**4096 (128x128x64, 1024 CTAs, 64 K-tiles):** Peak throughput at 88.70 TFLOP/s (62.5% peak). Enough CTAs (1024) that the largest tile wins outright. Untuned symmetric 64x64x64 leaves 25.8 percentage points on the table.

**8192 (128x128x64, 4096 CTAs, 128 K-tiles):** Same tile as 4096 but throughput drops to 72.95 TFLOP/s. Memory system saturates at 4096 CTAs; diminishing returns from additional parallelism.

## Tuning Gap: Symmetric vs Asymmetric + Occupancy

The gap between tuned and untuned is negligible at small sizes but dramatic at 1024+:

- At 1024: 64.99 vs 30.13 TFLOP/s (2.16x) -- asymmetric tiles unlock configurations the symmetric sweep never explores
- At 4096: 88.70 vs 52.11 TFLOP/s (1.70x) -- the largest absolute gap at 36.59 TFLOP/s

The untuned sweep is stuck at 64x64x64 for large sizes because symmetric tiles cannot simultaneously maximize per-CTA work and maintain sufficient grid coverage. Asymmetric tiles (128x64x32, 128x128x64) decouple M/N tile dimensions to solve this.

## Occupancy Hints

| Size | Best Occ | CTAs | Observation |
|------|----------|------|-------------|
| 128 | 2 | 16 | Tiny kernel -- occupancy irrelevant |
| 256 | 1 | 64 | Lightweight CTA, hint is secondary |
| 512 | 2 | 64 | Balanced residency vs per-CTA resources |
| 1024 | 2 | 128 | Same balance at larger scale |
| 2048 | 2 | 512 | Grid-saturated, occ=2 sufficient |
| 4096 | 2 | 1024 | Occ=2 dominates across large sizes |
| 8192 | 2 | 4096 | Consistent with 4096 |

Occupancy=2 wins almost universally. NCU profiling at FP16 1024 reveals: cuTile achieves only 8.3% occupancy (103 registers) with 76.7 GB/s bandwidth, while Triton at 21.9% occupancy (69 registers) reaches 120.6 GB/s and cuBLAS at 29.6% occupancy hits 243 GB/s. Meanwhile PTX at 95% occupancy (38 registers) manages only 7.8 GB/s. The lesson: occupancy alone is not the bottleneck — memory access efficiency and register-enabled data reuse determine actual throughput.

## Other Dtypes

- **BF16**: Similar progression to FP16; larger tiles win as size grows
- **Float32**: Prefers smaller tile_k (16) due to doubled byte footprint; best tiles shift to 16x16x16 through 64x64x16
- **Int8**: Diagnostically interesting but not valid for tuning recommendations (semantics issue)

## Cost Model

A linear cost model achieves **0% tile prediction accuracy**. Tile selection is fundamentally non-linear -- the interaction between CTA count, K-loop depth, and per-CTA footprint cannot be captured by a simple product model. See `benchmarks/cost_model.py` and `artifacts/cost_model/`.

## Practical Heuristic

1. **Tiny (128--256):** Use `32x32x32`. Launch overhead dominates; tuning irrelevant
2. **Medium (512):** Move to `64x64x64`. Reuse starts paying off
3. **Large (1024--2048):** Use asymmetric tiles (`128x64x64`, `128x64x32`). The tuning gap is 1.5--2.2x
4. **Very large (4096+):** Use `128x128x64` with occupancy=2. Peak throughput regime
5. Never assume the largest symmetric tile is best -- it leaves up to 25 percentage points on the table
