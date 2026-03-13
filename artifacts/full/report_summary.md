# cuTile Benchmark and IR Summary

## Thesis-oriented findings

1. cuTile can be a throughput and latency competitive path on Ampere for FP16/BF16 when tile shapes are tuned.
2. cuTile still has correctness and semantics issues, most clearly on the int8 path.
3. TileIR/cuTile is promising, but the current stack needs careful validation before claiming production correctness.
4. This artifact set does not yet prove a memory-footprint advantage; that needs a separate workspace/allocator instrumentation pass.

## Latency emphasis

The benchmark data includes average kernel latency in milliseconds for every backend, dtype, size, and tile.
For real-time AI/ML, the low-size regime (128/256) is the most important latency view in this artifact set; throughput alone is not sufficient.

- Fastest observed float16 latency: 0.005 ms on cuTile at size 128.
- Fastest observed bfloat16 latency: 0.005 ms on cuTile at size 128.
- Fastest observed float32 latency: 0.006 ms on PTX-inline at size 128.
- Fastest observed int8 latency: 0.005 ms on cuTile at size 128.

Cold-start timing is also reported separately via `compile_ms` and `first_launch_ms`.
For PTX, steady-state latency excludes module compile time; compile and first-launch costs are exported separately.

## PTX timing validation

- PTX validation case: `float16` at shape `128x128x128`
- Compile time: `24.476 ms`
- First launch time: `0.076 ms`
- Steady-state latency: `0.013 ms`
- Nsight Systems trace was captured with NVTX ranges `ptx_compile`, `ptx_first_launch`, and `ptx_steady_state` to verify the phase separation.

## Int8 IR finding

The exported int8 repro under `investigations/int8_ir/` shows that cuTile does not preserve exact int32 GEMM semantics for `i8 @ i8`.
- `max_err_exact`: 768
- `max_err_tile_wrapped_i8`: 0

That means the cuTile result matches a per-tile wrapped-int8 accumulation model rather than an exact int32 accumulation model.

Critical IR excerpt:

```text
    $110: Tile[int8,(32,32)], $token.4: Token = tile_load_token_ordered(array=a{$4, $6, $8, $10, a_4}, index=($34, $82), token=$token, order=(0, 1), padding_mode=PaddingMode.UNDETERMINED, latency=None, allow_tma=None)
    $138: Tile[int8,(32,32)], $token.6: Token = tile_load_token_ordered(array=b{$11, $13, $15, $17, b_4}, index=($82, $47), token=$token, order=(0, 1), padding_mode=PaddingMode.UNDETERMINED, latency=None, allow_tma=None)
    $155: Tile[int32,(32,32)] = tile_mma(x=$110, y=$138, acc=$154)
    $156: Tile[int8,(32,32)] = tile_astype(x=$155)
    $158: Tile[int32,(32,32)] = tile_astype(x=$156)
    $159: Tile[int32,(32,32)] = raw_binary_arith(lhs=acc.0, rhs=$158, fn="add", rounding_mode=None, flush_to_zero=False)
```

Interpretation:

- `tile_mma(...)` first produces an `int32` tile.
- cuTile then inserts `tile_astype` to narrow that tile to `int8`.
- It then widens the wrapped `int8` tile back to `int32` before accumulating.

So the int8 bug is visible directly in the cuTile IR, not just in the benchmark output.

## Best tile notes

### float32
- cuTile best tiles by size: 128:16x16x16, 256:32x32x16, 512:64x64x16, 1024:64x64x16
- Triton best tiles by size: 128:64x64x64, 256:64x64x32, 512:64x64x32, 1024:64x64x32

### float16
- cuTile best tiles by size: 128:32x32x32, 256:32x32x16, 512:64x64x64, 1024:64x64x16
- Triton best tiles by size: 128:64x64x64, 256:64x64x32, 512:64x64x64, 1024:64x64x16

### bfloat16
- cuTile best tiles by size: 128:32x32x32, 256:32x32x32, 512:64x64x32, 1024:64x64x16
- Triton best tiles by size: 128:64x64x32, 256:16x16x16, 512:64x64x32, 1024:64x64x32

### int8
- cuTile best tiles by size: 128:64x64x64, 256:32x32x32, 512:64x64x64, 1024:64x64x64
- Triton best tiles by size: 128:64x64x32, 256:32x32x32, 512:32x32x32, 1024:64x64x64

## Artifact files

- Raw benchmark CSV: `/home/trungnt13/codes/cutile/artifacts/full/benchmark_raw.csv`
- Raw benchmark JSON: `/home/trungnt13/codes/cutile/artifacts/full/benchmark_raw.json`
- Cold-start benchmark CSV: `/home/trungnt13/codes/cutile/artifacts/full/benchmark_coldstart.csv`
- Cold-start benchmark JSON: `/home/trungnt13/codes/cutile/artifacts/full/benchmark_coldstart.json`
- Best-of-sweep CSV: `/home/trungnt13/codes/cutile/artifacts/full/benchmark_best.csv`
- Comparison throughput barplot: `/home/trungnt13/codes/cutile/artifacts/full/comparison_throughput.png`
- Comparison latency barplot: `/home/trungnt13/codes/cutile/artifacts/full/comparison_latency.png`
- Comparison first-launch latency barplot: `/home/trungnt13/codes/cutile/artifacts/full/comparison_first_launch_latency.png`
- cuTile tile throughput barplot: `/home/trungnt13/codes/cutile/artifacts/full/cutile_tile_sweep_throughput.png`
- cuTile tile latency barplot: `/home/trungnt13/codes/cutile/artifacts/full/cutile_tile_sweep_latency.png`
- PTX timing validation JSON: `/home/trungnt13/codes/cutile/artifacts/full/ptx_latency_validation_summary.json`
- Nsight Systems trace dir: `/home/trungnt13/codes/cutile/artifacts/nsys`
- Int8 IR summary source: `/home/trungnt13/codes/cutile/investigations/int8_ir/summary.json`
- Int8 cuTile IR text: `/home/trungnt13/codes/cutile/investigations/int8_ir/mm_i8.cutileir.txt`
