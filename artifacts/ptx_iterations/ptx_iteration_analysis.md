# Parallel Thread Execution (PTX) Iteration Study

## Iterations

1. `iter1_scalar16`: current scalar FMA kernel with 16x16x16 tiling.
2. `iter2_scalar32`: larger scalar tile to improve reuse and reduce grid overhead.
3. `iter3_wmma32`: tensor-core Warp Matrix Multiply Accumulate (WMMA) kernel with 4 warps per block over a 32x32 output tile.

## Bottleneck analysis

- Iteration 1 bottleneck: scalar single-precision floating-point (FP32) fused multiply-add (FMA) path after fp16->float conversion; no Tensor Core usage.
- Iteration 2 bottleneck: better locality, but still scalar math and large thread blocks; still no Tensor Core path.
- Iteration 3 bottleneck: enters Tensor Core path, but still lacks library-grade pipelining, multi-stage scheduling, and deeper tiling compared with cuBLAS/Triton.

## Artifacts

- Raw CSV: `/home/trungnt13/codes/cutile/artifacts/ptx_iterations/ptx_iteration_raw.csv`
- Raw JSON: `/home/trungnt13/codes/cutile/artifacts/ptx_iterations/ptx_iteration_raw.json`
- Throughput figure: `/home/trungnt13/codes/cutile/artifacts/ptx_iterations/ptx_iteration_throughput.png`
- Steady-state latency figure: `/home/trungnt13/codes/cutile/artifacts/ptx_iterations/ptx_iteration_latency.png`
- First-launch latency figure: `/home/trungnt13/codes/cutile/artifacts/ptx_iterations/ptx_iteration_first_launch.png`
