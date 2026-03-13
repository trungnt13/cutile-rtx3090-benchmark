# Int8 Intermediate Representation (IR) Investigation

This directory contains the evidence behind the repo's main int8 caveat.

## Why This Exists

The benchmark results showed that cuTile's int8 output did not match exact int32 general matrix multiplication (GEMM) accumulation. Rather than hide that behind a single error number, the repo keeps the supporting evidence here:

- `summary.json`: measured discrepancy summary
- `mm_i8.cutileir.txt`: exported cuTile intermediate representation (IR)

## Key Finding

The current path matches a wrapped-per-tile accumulation model instead of exact int32 accumulation. In the published summary:

- `max_err_exact = 768`
- `max_err_tile_wrapped_i8 = 0`

That is why the main benchmark report treats int8 as a correctness caveat, not as a clean performance win.

## Reproducing the Export

```bash
python investigations/int8_ir/export_int8_ir.py --outdir investigations/int8_ir
```

## Why It Is Separate From `artifacts/`

`artifacts/` contains benchmark outputs. This directory contains supporting investigative evidence for one specific semantic issue. Keeping them separate makes the public report cleaner while still preserving the proof trail.
