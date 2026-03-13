# Half-Precision Floating-Point (FP16) cuTile Optimization Summary

## Headline

This report removes Parallel Thread Execution (PTX) from the main figures and focuses on whether tuned cuTile FP16 can beat Triton and Torch.

## Best cuTile configs by size

- 128: 64x64x64, occupancy=8, 0.93 TFLOP/s, 0.005 ms
- 256: 32x32x32, occupancy=1, 7.34 TFLOP/s, 0.005 ms
- 512: 64x64x64, occupancy=2, 29.57 TFLOP/s, 0.009 ms
- 1024: 128x64x64, occupancy=2, 61.93 TFLOP/s, 0.035 ms

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
