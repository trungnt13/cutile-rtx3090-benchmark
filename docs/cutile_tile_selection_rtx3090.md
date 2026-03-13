# Tile Selection on RTX 3090: Why the Best Tile Changes with Size

cuTile does not have a single best tile. The winner depends on workload scale, dtype, and machine balance.

## Summary Table

| Size | Best FP16 Tile | Occ | Latency (ms) | TFLOP/s | CTAs | K-tiles |
|------|---------------|-----|-------------|---------|------|---------|
| 128 | `64x64x64` | 8 | 0.005 | 0.93 | 4 | 2 |
| 256 | `32x32x32` | 1 | 0.005 | 7.34 | 64 | 8 |
| 512 | `64x64x64` | 2 | 0.009 | 29.57 | 64 | 8 |
| 1024 | `128x64x64` | 2 | 0.035 | 61.93 | 128 | 16 |

## Three Forces

The progression is driven by the interaction of:

1. **CTA count**: `(M/tile_m) * (N/tile_n)` — must be enough to cover 82 SMs
2. **K-loop iterations**: `K/tile_k` — fewer iterations means less loop overhead and better reuse
3. **Per-CTA footprint**: larger tiles increase register/shared-memory pressure, reducing scheduling flexibility

## Size 128: Latency Microkernel

The problem is too small for CTA abundance to matter. `64x64x64` wins by cutting K-tiles from 4 to 2 and reducing CTAs from 16 to 4. The top configurations are tightly clustered (0.93 vs 0.92 TFLOP/s) — this is a launch-overhead-dominated regime.

## Size 256: Grid Coverage Matters

`32x32x32` produces 64 CTAs — finally enough to distribute across 82 SMs. `64x64x64` drops to 16 CTAs, which is too few. The larger tile's reuse advantage doesn't compensate for poor grid coverage at this size.

## Size 512: Reuse Starts to Dominate

`64x64x64` returns as the winner: 64 CTAs cover the GPU, and halving K-loop iterations (8 vs 16) pays off. This is the cleanest crossover — the workload is now large enough to amortize per-CTA cost of bigger tiles.

## Size 1024: Asymmetric Tiles Win

`128x64x64` beats `64x64x64` by doubling output work per CTA while maintaining 128 CTAs. The biggest symmetric tiles (`128x128x64`, `128x128x128`) overshoot — only 64 CTAs and excessive per-CTA footprint.

![FP16 tile sweep throughput](../artifacts/fp16_focus/cutile_fp16_tile_sweep_throughput.png)
*Tile preference is size-dependent: small problems want low overhead, large problems reward reuse.*

![FP16 Pareto](../artifacts/fp16_focus/cutile_fp16_pareto_tiles.png)
*The best tiles move onto the useful latency-throughput frontier, not just maximize one metric.*

## Occupancy Hints

| Size | Best Occ | Rationale |
|------|----------|-----------|
| 128 | 8 | Tiny kernel — high residency shaves microseconds |
| 256 | 1 | Lightweight CTA — occupancy hint is secondary |
| 512 | 2 | Balance between per-CTA resources and residency |
| 1024 | 2 | Same balance at larger scale |

Higher occupancy is not monotonically better. Once CTAs are compute-heavy, the system prefers per-CTA efficiency over maximum residency.

## Other Dtypes

- **BF16**: Similar to FP16 — larger tiles win as size grows
- **Float32**: Prefers smaller `tile_k` (16) due to doubled byte footprint; best tiles are 16x16x16 (128) through 64x64x16 (1024)
- **Int8**: Diagnostically interesting but not valid for tuning recommendations (semantics issue)

## Cost Model

A linear cost model `latency = CTA_count * K_iters * (alpha * tile_volume + beta)` is fitted to benchmark data in `benchmarks/cost_model.py`. See `artifacts/cost_model/` for prediction accuracy against measured winners.

## Practical Heuristic

- Tiny GEMMs (128): favor low-overhead tiles; top configs are tightly clustered
- Small GEMMs (256): prefer `32x32x32`; problem too small for heavyweight CTAs
- Medium GEMMs (512): move to `64x64x64`; reuse starts to dominate
- Large GEMMs (1024+): prefer asymmetric tiles like `128x64x64`; maximize per-CTA work without starving the grid
- Avoid assuming the largest symmetric tile is best
