# FP16 cuTile Optimization Summary

## Headline

This report removes PTX from the main figures and focuses on whether tuned cuTile FP16 can beat Triton and Torch.

## Best cuTile configs by size

- 128: 32x32x32, occupancy=2, 0.57 TFLOP/s (0.4% of peak), 0.007 ms
- 256: 32x32x32, occupancy=1, 4.03 TFLOP/s (2.8% of peak), 0.008 ms
- 512: 64x64x64, occupancy=2, 19.76 TFLOP/s (13.9% of peak), 0.014 ms
- 1024: 128x64x64, occupancy=2, 64.99 TFLOP/s (45.8% of peak), 0.033 ms
- 2048: 128x64x32, occupancy=2, 81.02 TFLOP/s (57.1% of peak), 0.212 ms
- 4096: 128x128x64, occupancy=2, 88.70 TFLOP/s (62.5% of peak), 1.550 ms
- 8192: 128x128x64, occupancy=2, 72.95 TFLOP/s (51.4% of peak), 15.071 ms

## Artifact files

- Raw CSV: `/home/trungnt13/codes/cutile/artifacts/fp16_focus/fp16_raw.csv`
- Raw JSON: `/home/trungnt13/codes/cutile/artifacts/fp16_focus/fp16_raw.json`
- Comparison throughput: `/home/trungnt13/codes/cutile/artifacts/fp16_focus/comparison_fp16_throughput.png`
- Comparison latency: `/home/trungnt13/codes/cutile/artifacts/fp16_focus/comparison_fp16_latency.png`
- Comparison first-launch latency: `/home/trungnt13/codes/cutile/artifacts/fp16_focus/comparison_fp16_first_launch_latency.png`
- cuTile tile throughput: `/home/trungnt13/codes/cutile/artifacts/fp16_focus/cutile_fp16_tile_sweep_throughput.png`
- cuTile tile latency: `/home/trungnt13/codes/cutile/artifacts/fp16_focus/cutile_fp16_tile_sweep_latency.png`
- FP16 Pareto tradeoff: `/home/trungnt13/codes/cutile/artifacts/fp16_focus/fp16_pareto_tradeoff.png`
- cuTile FP16 tile Pareto: `/home/trungnt13/codes/cutile/artifacts/fp16_focus/cutile_fp16_pareto_tiles.png`
