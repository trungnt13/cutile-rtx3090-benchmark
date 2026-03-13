# Trade-Off Analysis

## What cuTile Currently Proves

- It can be competitive on half-precision floating-point (FP16) and brain floating point (BF16) when tuned.
- It exposes tile decisions clearly enough to support engineering analysis.
- It is promising as a research and compiler exploration tool on Ampere-class hardware.

## What cuTile Does Not Yet Prove

- Production-ready correctness for int8 general matrix multiplication (GEMM).
- End-to-end superiority over mature library stacks.
- Memory-footprint advantages; this repo does not instrument allocator behavior in a way that supports that claim.

## Why the Parallel Thread Execution (PTX) Baseline Is Intentionally Limited

The PTX-inline kernel in the main report is not trying to be the best possible handwritten CUDA implementation. That would create a moving target and turn the benchmark into an optimization contest instead of a readable comparison.

The current design keeps PTX useful for two purposes:

- low-level reasoning about tiled scalar execution
- a stepwise iteration study that shows where tensor-core adoption begins to matter

## Why Triton Matters Here

Triton is the strongest “manual kernel” comparison in this repo because it gives explicit kernel control without dropping to raw Compute Unified Device Architecture (CUDA) or PTX source. If cuTile is only competitive against the PTX baseline but not against Triton, that would be weak evidence. The repo therefore keeps Triton in the main figures.

## Why Int8 Is Treated Conservatively

The int8 intermediate representation (IR) investigation shows a semantic mismatch, not just a numerical tolerance issue. The current cuTile path narrows and re-widens partial tiles in a way that matches wrapped accumulation rather than exact int32 accumulation.

That means:

- int8 performance numbers can still be useful diagnostically
- they should not be presented as production-correct GEMM results
- the correctness caveat belongs in the headline narrative, not only in an appendix

## Public Interpretation

The right public reading of this benchmark is:

- tuned cuTile is promising on FP16/BF16
- PTX-inline explains useful low-level behavior but is not a production baseline
- Triton remains a strong comparison point
- int8 requires correctness work before performance claims should carry product weight
