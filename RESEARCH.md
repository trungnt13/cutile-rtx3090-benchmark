# cuTile Research Report

## Executive Summary

This repository, [`trungnt13/cutile-rtx3090-benchmark`](https://github.com/trungnt13/cutile-rtx3090-benchmark), is not a general-purpose cuTile library; it is a benchmark artifact designed to answer a narrow question: how competitive NVIDIA's `cuda.tile` / cuTile stack is against PTX-inline, Triton, Torch, and cuBLAS on Ampere-class GEMMs, while explicitly separating compile time, first-launch latency, and steady-state kernel time.[^1][^2][^3]

The architecture works because the benchmark is built around a clean separation of concerns: `benchmarks/core.py` defines the kernel builders, launch policies, timing helpers, and correctness references; `reports/full_report.py` and `reports/fp16_focus.py` consume those primitives to generate artifacts; and `investigations/int8_ir/` exists as a side channel to validate a known semantic problem in the current int8 path.[^4][^5][^6]

cuTile works best here when the problem is FP16/BF16, the tile shape is tuned to the matrix size, and occupancy is nudged into the sweet spot that the compiler/runtime likes. It works less well when users expect production-stable semantics across dtypes, a robust automatic tile-selection model, or a fair apples-to-apples comparison against fully tuned handwritten or library kernels.[^7][^8][^9][^10]

My core inference is that NVIDIA needs cuTile not because this repo proves cuTile is already the best GEMM path, but because the repo demonstrates a deeper strategic need: NVIDIA wants a native tile IR/compiler layer it controls, can introspect, can autotune, and can eventually bridge the gap between low-level kernels, higher-level DSLs like Triton, and production libraries like cuBLAS and CUTLASS-backed Torch paths.[^11][^12][^13]

## Query Type

This is primarily a **technical deep-dive** with some conceptual inference layered on top, because the user asked for repository purpose, architecture/specs, failure modes, and second-/third-order strategic implications.[^1][^4]

## Architecture/System Overview

At a high level, the repository has four planes: benchmark execution, artifact generation, investigative validation, and published conclusions.[^1][^4]

```text
┌────────────────────────────┐
│  Source benchmark logic    │
│  benchmarks/core.py        │
│  - input generation        │
│  - cuTile/PTX/Triton/Torch │
│  - timing + correctness    │
└──────────────┬─────────────┘
               │
               │ shared primitives
               ▼
┌──────────────────────────────────────┐
│ Report drivers                       │
│ reports/full_report.py               │
│ reports/fp16_focus.py                │
│ - sweep configs                      │
│ - choose comparison rows             │
│ - emit CSV/JSON/plots/markdown       │
└──────────────┬───────────────────────┘
               │
               ├──────────────► artifacts/full/
               │               artifacts/fp16_focus/
               │               artifacts/cost_model/
               │               artifacts/ncu/
               │
               ▼
┌──────────────────────────────────────┐
│ Investigations                       │
│ investigations/int8_ir/              │
│ - export TileIR / bytecode           │
│ - compare exact vs wrapped int8      │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ Public narrative                     │
│ README.md / RESULTS.md / docs/*.md   │
│ - claims                             │
│ - scope                              │
│ - trade-offs                         │
└──────────────────────────────────────┘
```

The repo is “benchmark-first, not library-first,” and says so explicitly. The top-level layout reflects that bias: most of the repository is benchmark runners, report generators, artifacts, and explanatory docs rather than a reusable package surface.[^1]

The published artifact was collected on a Linux RTX 3090 host with an AMD Ryzen 7 5800X, CUDA 13.0 reported by `nvidia-smi`, Torch 2.10.0, Triton 3.6.0, CuPy 14.0.1, and `cuda-tile` 1.2.0. That matters because this repo is best understood as a committed benchmark dataset plus the code that produced it, not as something authored to run natively in the current macOS checkout.[^3][^14]

## What the Repository Is For

The repository’s purpose is to answer three concrete questions: whether tuned cuTile can compete on latency/throughput for practical GEMMs on RTX 3090, how much of the gap to Torch and Triton comes from kernel design versus compile/launch overhead, and which parts of the current cuTile stack are promising versus not yet product-ready.[^1]

That purpose explains several design choices that otherwise look odd. The benchmark compares five backends—cuTile, PTX-inline, Triton, Torch, and cuBLAS—but the methodology explicitly says the goal is “comparative engineering analysis,” not equal optimization effort. In other words, the repo is trying to learn something about the cuTile stack, not win a benchmark bake-off at any cost.[^2]

It also explains why the docs are unusually candid about limitations. The README and tradeoff docs foreground the int8 semantic caveat, the separation of cold-start from steady-state, and the fact that the PTX baseline is deliberately readable rather than maximally optimized.[^1][^15]

## cuTile Architecture in This Repo

### 1. Execution model

The core cuTile kernel builder lives in `benchmarks/core.py`. For floating-point dtypes, it creates a `@ct.kernel` that loads `tile_m x tile_k` and `tile_k x tile_n` tiles, accumulates in `float32`, and stores the result tile back out. For int8, it uses a distinct kernel that accumulates in `int32` and is treated separately because correctness is already known to be problematic.[^5]

The launch policy is direct and explicit: `run_cutile()` computes a grid of `(ceil_div(m, tile_m), ceil_div(n, tile_n), 1)`, computes `k_tiles = ceil_div(k, tile_k)`, and launches the kernel with the tile metadata passed as constants. This means tile shape is not metadata around the experiment; it is literally the control surface of the benchmark and the primary lever being studied.[^5][^16]

Compilation is also explicit. `compile_cutile_kernel()` invokes `compile_tile()` over the kernel’s Python function and a `CompilerOptions` object, separating JIT cost from launch cost. That separation is then enforced by `measure_host_ms()` plus the event-based steady-state timers.[^5][^17]

### 2. Benchmark backends and reference semantics

The repo’s central comparison helper, `benchmark_tile_config()`, primes cuTile, PTX, and Triton once, computes correctness errors against a reference, and only then enters the timed loops. That order avoids contaminating correctness or steady-state measurements with lazy initialization or first-launch effects.[^16]

The floating-point reference is intentionally strict: `make_reference()` disables TF32 and computes `torch.mm(a.float(), b.float())`. That means the correctness reference is more exact than the default fast path Ampere often uses for production float32 GEMM.[^16][^18]

The backends are deliberately asymmetric.

- Torch uses `torch.mm` for floating point and `torch._int_mm` for int8.[^16]
- cuBLAS is measured via zero-copy DLPack transfers into CuPy and `cp.matmul`, with fallbacks to Torch for unsupported dtypes like int8 and bfloat16 in this setup.[^16]
- Triton is treated as the strongest handwritten-kernel comparison and gets a tile sweep in both the full report and the FP16-focused report.[^4][^15]
- PTX-inline is fixed and intentionally simple in the main benchmark, with a separate iteration study for richer PTX experiments.[^4][^15][^19]

That architecture is important: cuTile is being benchmarked as a compiler-generated kernel path that sits between very low-level CUDA/PTX reasoning and higher-level tuned library/runtime paths.[^4][^15]

### 3. FP16-focused tuning architecture

`reports/fp16_focus.py` is effectively the repo’s “what if we treat cuTile as a tunable compiler target instead of a fixed benchmark row?” experiment. It expands the cuTile search space to asymmetric tiles such as `128x64x64`, `128x64x32`, and `128x128x64`, and sweeps occupancy hints across `{auto, 1, 2, 4, 8}`.[^8]

The best-per-size summary shows why this matters. The best cuTile config changes across the size ladder: `32x32x32` at 128/256, `64x64x64` at 512, `128x64x64` at 1024, `128x64x32` at 2048, and `128x128x64` at 4096/8192. Peak FP16 performance in the focused study reaches 88.70 TFLOP/s (62.5% of peak) at 4096 with `128x128x64, occupancy=2`.[^7][^20]

This is the strongest evidence that cuTile’s upside is real: once the benchmark moves beyond symmetric tiles, cuTile materially closes the gap to stronger baselines and sometimes beats the repo’s Triton rows at selected sizes.[^7][^20]

### 4. Artifact-generation architecture

`reports/full_report.py` and `reports/fp16_focus.py` are not just plotting scripts. They encode the public story by choosing which rows count as comparisons, which tiles are exposed in figures, and how cold-start data is exported.[^6][^8]

For example, the full report keeps a fixed set of public cuTile tile families—`32x32x32`, `64x64x64`, and `128x128x128`—so the figures show stable trade-offs rather than changing candidate sets across sizes. PTX rows are reduced by taking the median repeated PTX measurement rather than the best, explicitly to avoid overstating a fixed kernel.[^6]

That makes the benchmark more interpretable, but it also means the public figures are curated views over a much larger raw search space, especially once the FP16-focused tuner is considered.[^6][^8]

## Why cuTile Works Here

### A. It exposes the right control variables

cuTile works in this benchmark because the experiment is aligned with the abstraction that cuTile exposes. The kernel API makes tile shape, grid decomposition, and compiler hints explicit, so the benchmark can study exactly the variables that most determine performance on tiled GEMMs.[^5][^16]

The tile-selection deep-dive document frames the problem as a balance among CTA count, K-loop depth, and per-CTA resource footprint. That is almost a description of the cuTile abstraction itself: it gives users and compiler developers a vocabulary for reasoning about these trade-offs directly.[^9]

### B. The methodology separates one-time and steady-state costs

The benchmark works as an engineering artifact because it does not hide JIT and first-launch overhead inside steady-state throughput. The methodology explicitly splits compile time, first launch time, and steady-state latency, and the PTX phase-separation validation uses NVTX ranges to verify that the measurement policy matches what the report claims.[^2][^21]

This is especially important for a compiler-driven runtime like cuTile. If you mix JIT, launch, and steady-state into one number, you cannot tell whether the weakness is the generated kernel, the compiler, or just cold-start behavior.[^2][^4]

### C. The repo measures enough low-level evidence to explain results

The Nsight Compute summary is unusually valuable here. For FP16 at 1024, cuTile sits at roughly 8.3% achieved occupancy, 103 registers, and about 77 GB/s of memory bandwidth, while Triton hits about 22% occupancy and ~121 GB/s and cuBLAS reaches ~243 GB/s. PTX, meanwhile, achieves ~95% occupancy but only ~7.8 GB/s.[^10]

That evidence supports one of the repo’s most important conclusions: occupancy alone is not the bottleneck. cuTile’s problem is not simply “not enough occupancy”; it is that its scheduling and memory behavior are still materially weaker than the stronger baselines. Conversely, PTX proves that high occupancy without reuse is almost worthless.[^9][^10]

### D. It is honest about the difference between research promise and product readiness

The repository works as research because it does not pretend the current int8 path is production-valid and does not claim memory-footprint wins it did not measure. The docs repeatedly state that FP16/BF16 are promising, PTX is explanatory rather than production-grade, and int8 needs correctness work before performance claims should carry product weight.[^15][^22]

That restraint matters. It turns the benchmark into a useful architectural probe rather than marketing collateral.[^15][^22]

## Why cuTile Does Not Work, or Does Not Yet Work

### A. The int8 path is semantically broken for exact GEMM

The most concrete failure is int8 correctness. The exported int8 repro shows `max_err_exact = 768` and `max_err_tile_wrapped_i8 = 0`, which means the cuTile output matches a wrapped-per-tile accumulation model, not exact `int32` GEMM accumulation.[^23]

The root cause is visible directly in the exported IR. After `tile_mma(...)` produces an `int32` tile, the IR narrows that tile to `int8` with `tile_astype`, then widens it back to `int32`, and only then adds it into the running accumulator. That is not a numerical-tolerance issue; it is a semantic mismatch in the generated program.[^24][^25]

This is why the repo treats int8 throughput as diagnostic only. Architecturally, the stack has not yet stabilized its integer accumulation semantics.[^15][^22]

### B. Automatic tile selection is still missing

The repo’s cost-model experiment is devastating for any “just predict the right tile” story. The model `latency = CTA_count * K_iters * (alpha * tile_volume + beta)` fits with `R² ≈ 0.198` and achieves 0% best-tile prediction accuracy, always preferring `128x128x128` even when the measured best tiles vary widely by size and dtype.[^26][^27]

That means cuTile’s best results in this repo are not portable defaults; they are discovered by search. In practice, that turns cuTile from a simple code-generation abstraction into an autotuning/compiler-search problem.[^7][^9][^26]

### C. The strongest FP16 results hide a nontrivial tuning cost

The focused FP16 report highlights best steady-state rows, but the raw CSV shows that many candidate cuTile configurations cost hundreds of milliseconds to more than a second to compile even at small sizes. For example, size-128 FP16 configurations around `128x128x128` have compile times around 0.98–1.32 seconds, and many other candidates sit in the 200–460 ms range.[^28]

That does not invalidate the 62.5%-of-peak result, but it changes how we should interpret it. The “tuned cuTile” story includes an often-expensive search/compile envelope that the headline performance summary mostly hides.[^20][^28]

### D. The PTX comparison is intentionally weak

The full benchmark’s PTX path is fixed at a 16x16 tiled scalar kernel in `run_ptx()`, while the PTX iteration study itself admits that the first two PTX variants remain scalar and only the third enters a WMMA tensor-core path, still without library-grade pipelining or scheduling.[^16][^19]

So when cuTile beats the main-report PTX baseline, that is informative but limited: it proves cuTile is better than a readable scalar PTX reference, not that it dominates what a fully optimized handwritten CUDA/PTX kernel could do.[^4][^15][^19]

### E. Float32 is structurally disadvantaged

The repo’s own results say float32 is severely limited for cuTile and note that cuTile has no TF32 support. Meanwhile Torch reaches over 100% of the nominal FP32 peak because the production path is leveraging Ampere’s TF32 behavior, whereas the benchmark’s correctness reference deliberately disables TF32 for numerical comparisons.[^7][^18][^29]

So float32 is not merely “slower in this benchmark”; it is an example of cuTile not yet spanning the hardware-optimized production path that the vendor stack already uses.[^7][^18][^29]

### F. Parts of the written spec overstate what the code currently does

The methodology doc says the benchmark covers rectangular GEMMs, and `benchmarks/core.py` defines `RECTANGULAR_SHAPES` for MLP-like and attention-like cases. But `reports/full_report.py`’s actual collector loops only over square sizes by setting `m = n = k = size`, and `plot_rectangular_bar()` simply returns if no rectangular rows exist.[^2][^30][^31]

This is a genuine spec/implementation mismatch. The rectangular-scope story exists in the docs and some dead code, but not in the published execution path.[^2][^30][^31]

### G. The FP16 tuner silently drops failures

`collect_cutile()` in `reports/fp16_focus.py` wraps each tile/occupancy attempt in a broad `try/except Exception: continue`. That means unsupported or failing configurations disappear from the artifact instead of being surfaced as negative results.[^8]

This does not make the successful points false, but it does mean the repo presents the successful tuning frontier more clearly than the failure surface. That is typical of an exploratory compiler benchmark, but it is not the same thing as a production robustness study.[^8]

## cuTile Specs: What Is Explicit, What Is Implicit

### Explicitly specified

The benchmark scope is explicit about supported dtypes (`float32`, `float16`, `bfloat16`, `int8`), square problem sizes from 128 through 8192, separated timing phases, strict floating-point references, and the split between a full symmetric-tile sweep and an FP16-focused asymmetric/occupancy study.[^2][^6][^8]

The repo is also explicit that unsupported dtypes like FP8 and INT4 are intentionally out of scope on RTX 3090, with reasons embedded in `DTYPE_INFO` rather than left as silent omissions.[^32]

### Implicit or under-specified

Several important “specs” are only implicit in code.

- Int8 rows in the full report silently filter the tile list to `tile_k >= 32`.[^6]
- Occupancy is exposed as a compiler hint, but the meaning of `1`, `2`, `4`, or `8` is never documented beyond experimental use.[^5][^8]
- The full report compares fixed public tile families rather than the entire possible tile space, which is a presentation decision rather than an architecture constraint.[^6]
- The strongest cuTile story comes from `reports/fp16_focus.py`, where failures can be silently skipped.[^8]

These are signs of an evolving system whose practical spec still lives partly in source code and artifacts rather than stable documentation.[^5][^8]

## Second-, Third-, and Fourth-Order Strategic Effects: Why NVIDIA Has to Develop cuTile

*Revised 2025-07. This section substantially extends the original strategic analysis with competitive-landscape research, Blackwell hardware evidence, and deeper causal chain reasoning. Repo-internal citations use `[^N]`; external web sources use `[W:N]`.*

---

### Prologue: The Strategic Question

The surface question—"why build cuTile?"—has a surface answer: to give developers a higher-level kernel API. But that answer is incomplete. The deeper question is: **why must NVIDIA replace the PTX path entirely, rather than augment it, and why now?** Answering that requires examining five interacting forces: hardware architecture constraints, the LLM workload shift, the competitive landscape, ecosystem-control dynamics, and long-run platform economics.

---

### I. Why PTX Cannot Be Extended: The Architecture Constraint

#### A. PTX is structurally thread-centric

PTX (Parallel Thread Execution) was designed around the SIMT (Single Instruction, Multiple Threads) model. Every PTX instruction acts on a thread-level virtual register, and the programmer manually maps work to warps, blocks, and grids. This was an excellent abstraction for the era of shader-derived GPU compute (2007–2020).[W:1][W:2]

The problem is that modern GPU compute is no longer thread-centric. Tensor cores, TMA (Tensor Memory Accelerator), cluster-level shared memory, and CTA-pair execution all operate at **tile granularity**—multi-dimensional blocks of data that span hundreds of threads. PTX has no native concept of a tile, a tile schedule, or a tile memory lifetime.[W:1][W:2][W:3]

**Repo evidence**: The PTX kernel in `benchmarks/core.py` (lines 50–110) is 60 lines of scalar loads, FMA loops, and explicit indexing. It achieves 95.45% occupancy at 1024×1024 FP16—but only 7.8 GB/s memory bandwidth and 0.26 TFLOP/s throughput. Compare cuTile's 8.32% occupancy at the same size producing 77 GB/s and 9.31 TFLOP/s. High occupancy in thread-centric code does not translate to tensor-core utilization.[^10][^5]

#### B. Blackwell hardware creates features PTX literally cannot express

NVIDIA's Blackwell architecture (2025) introduces hardware constructs that have no PTX analog:[W:3][W:4][W:5]

| Feature | What it does | PTX analog |
|---|---|---|
| **TMEM (Tensor Memory)** | Dedicated on-chip memory for tensor cores; 58% latency reduction in cache-miss scenarios | None |
| **UMMA (`tcgen05.mma`)** | 5th-gen tensor core instruction; only one thread in a CTA pair issues it; hardware manages the data pipeline | `wmma`/`mma` are per-warp, not per-CTA-pair |
| **CTA-pair execution** | Two CTAs cooperate symbiotically, sharing tensor memory and synchronization fences | No PTX primitive for cross-CTA cooperation |
| **Block-scaled MMA** | FP8/FP6/FP4 matrix multiply with per-block scaling factors | No native support; would require manual emulation |
| **TMA v2** | Asynchronous 2D/3D bulk memory copy with hardware-managed tiling | Partial via `cp.async`, but no tile semantics |

Extending PTX to cover these would require either: (a) invasive changes that destroy PTX's portability guarantee across GPU generations, or (b) a new layer of abstraction on top of PTX that is, functionally, a tile IR. NVIDIA chose (b) and called it Tile IR.[W:2][W:3][W:5]

#### C. The compiler pipeline shift

The old pipeline: `CUDA C++ → PTX → ptxas → SASS`.
The new pipeline: `cuTile (Python) → Tile IR (MLIR dialect) → tileiras → NVVM (LLVM) → SASS`.[W:1][W:2]

This is not a cosmetic change. Tile IR is built on MLIR (Multi-Level Intermediate Representation), the same compiler infrastructure used by Google (XLA/IREE), AMD (ROCm), and Intel (oneAPI). By adopting MLIR, NVIDIA gets access to existing optimization passes, dialect interop, and a shared toolchain ecosystem—while still controlling the final lowering to proprietary SASS.[W:2][W:6][W:7]

**Repo evidence**: The repo pins `nvidia-cuda-tileiras==13.2.51` in `requirements.txt` and the investigation code in `investigations/int8_ir/export_int8_ir.py` can dump both TileIR text and bytecode. The int8 IR dump (lines 37–44 of `mm_i8.cutileir.txt`) shows `tile_mma` → `narrow_i8` → `widen_i32` lowering decisions that are visible and debuggable at the IR level—something impossible in the PTX path.[^11][^12][^24][^25]

---

### II. The Competitive Landscape Forcing cuTile

NVIDIA's decision to build a new compiler stack is not happening in a vacuum. Every major competitor already has—or is building—a tile-native compiler, and several are explicitly designed to bypass CUDA entirely.

#### A. Google TPU + XLA/Pallas: The most mature tile compiler threat

Google's TPU stack is the existence proof that tile-native compilation works at scale:[W:8][W:9][W:10]

- **TPUv7 (Ironwood, 2025)**: 9,216 chips per pod, hardware-software co-designed for inference. XLA (Accelerated Linear Algebra) automatically fuses operations into cache-friendly tiles targeting the Matrix Multiply Unit (MXU) and Vector Processing Unit (VPU).
- **Pallas**: A JAX-integrated DSL that lets users write custom tile-level kernels, similar to what cuTile offers but tightly integrated with Google's framework ecosystem.
- **Adoption signal**: Anthropic, Midjourney, and Meta have signed massive TPU deals for production inference. Midjourney reported **65%+ cost reduction** migrating from NVIDIA clusters to TPU.[W:8][W:9]

The strategic implication: Google's compiler does for TPU what cuTile aims to do for GPUs—but Google has been shipping it for years. NVIDIA is playing catch-up on the compiler abstraction layer while maintaining hardware performance leadership.

#### B. Cerebras WSE: Proof that "no tiling needed" is a viable competitor

Cerebras's Wafer-Scale Engine takes a radically different approach:[W:11][W:12][W:13]

- **WSE-3**: 4 trillion transistors, 900,000 AI-optimized cores, entire wafer as a single chip.
- **Compiler model**: Neural network computation graphs are mapped directly onto the 2D wafer fabric. The compiler handles all parallelism, tiling, and memory scheduling automatically.
- **Performance**: 125 PFLOP/s (FP16) per CS-3 system; **2× performance-per-watt** versus NVIDIA at cluster scale.

Cerebras proves that for large-model workloads, the abstraction can be pushed even higher than tiles—to entire computation graphs. Developers write no tiling code at all. This is the extreme end of the spectrum that cuTile is racing to prevent from becoming the default developer expectation.

#### C. AMD MI300 + ROCm: The credible CUDA alternative

AMD's MI300X has achieved parity or near-parity for LLM inference:[W:14][W:15][W:16]

- **192 GB HBM3** (vs. H100's 80 GB), **5.3 TB/s bandwidth**—enabling single-GPU deployment of models that require multi-GPU on NVIDIA.
- ROCm now supports SGLang, vLLM, DeepSeek-R1, with Microsoft, Dell, HPE, and Lenovo backing.
- ROCm's kernel fusion and auto-tuning are rapidly closing the software gap.

AMD's threat is not that it has a better tile compiler—it doesn't. The threat is that **it doesn't need one as urgently** because ROCm rides on existing open-source compilers (Triton, MLIR) that are hardware-agnostic. If the open-source compiler ecosystem becomes the default, NVIDIA's proprietary stack loses its lock-in value.

#### D. Groq, Tenstorrent, and the ASIC Proliferation

- **Groq**: Custom inference ASIC optimized for ultra-low-latency LLM serving. Proprietary compiler converts PyTorch/ONNX models to their ISA. Not a tile abstraction—a fully custom pipeline.
- **Tenstorrent**: Open MLIR-based `TT-Forge` compiler, open silicon IP (Blackhole chip). Targets sovereign AI infrastructure and companies wanting to avoid vendor lock-in.[W:17]

These are not direct cuTile competitors. But they represent a world where **the compiler, not the hardware ISA, is the primary developer interface**. In that world, whoever controls the compiler abstraction controls the platform. NVIDIA cannot afford to let that abstraction be owned by others.

---

### III. The Triton Threat: Why Open-Sourcing Tile IR Is Both Offense and Defense

OpenAI's Triton is the most direct catalyst for cuTile's development.[W:18][W:19][W:20]

#### What Triton does

Triton is a Python-based DSL and compiler for writing GPU kernels at tile granularity. Critically, the same Triton code can target NVIDIA GPUs (via PTX), AMD GPUs (via ROCm), and potentially other accelerators. PyTorch 2.0's `torch.compile` uses Triton as its default kernel-generation backend.

#### Why Triton threatens NVIDIA

If Triton becomes the universal kernel-authoring layer:
1. **NVIDIA loses control of the lowering path**: Triton→PTX bypasses NVIDIA's optimization expertise. NVIDIA becomes a "dumb hardware target."
2. **Hardware-agnostic code is portable code**: A Triton kernel that runs on H100 can, in principle, run on MI300X with minimal changes. This erodes CUDA lock-in.
3. **Developer mindshare shifts**: AI researchers increasingly write Triton kernels, not CUDA. The 19-year CUDA ecosystem investment depreciates.

#### NVIDIA's counter-move: absorb Triton via the Tile IR backend

In early 2025, NVIDIA integrated CUDA Tile as a **backend for OpenAI Triton**. This means Triton kernels can now be lowered through Tile IR → tileiras → SASS instead of the traditional Triton → PTX → ptxas path.[W:18][W:21][W:22]

This is strategically brilliant and defensive simultaneously:
- **Offense**: Triton kernels targeting the Tile IR backend get access to Blackwell-specific optimizations (TMEM, UMMA, CTA-pair) that the PTX path cannot express. This makes NVIDIA hardware strictly better when using Tile IR, pulling developers toward NVIDIA-specific optimizations even through a "portable" DSL.
- **Defense**: If Triton becomes ubiquitous, NVIDIA still controls the final lowering step. The Tile IR backend becomes the optimization bottleneck, and NVIDIA owns the bottleneck.

**Repo evidence**: The tradeoff docs (`docs/tradeoffs.md:15–27`) explicitly position cuTile relative to Triton as the strongest kernel-authoring comparison. The repo benchmarks cuTile against Triton on equal footing (both get tile sweeps), not against PTX. This framing reveals that NVIDIA sees Triton, not PTX, as the competitive reference point for cuTile's value proposition.[^15][^4]

#### Open-sourcing Tile IR under Apache 2.0

In December 2025, NVIDIA open-sourced CUDA Tile IR on GitHub under the Apache 2.0 license.[W:6][W:7][W:23] This was widely analyzed as a moat-management decision:

- **Expanding the moat**: By making Tile IR the standard MLIR dialect for tile-based GPU programming, NVIDIA ensures that community innovation (Triton backends, new DSLs, academic research) builds atop NVIDIA's abstraction.
- **Defending the moat**: Deep hardware-specific optimizations (TMEM access patterns, CTA-pair scheduling, tensor-pipe dataflows) remain tightly coupled to proprietary drivers and hardware. The IR is open; the best backend is not.
- **Absorbing competitors**: Projects like ZLUDA (CUDA-on-AMD) could theoretically use open Tile IR to emit AMD-compatible code. But because peak performance requires NVIDIA-specific lowering, the practical effect is that developers optimize for NVIDIA first and port second.

---

### IV. The Causal Chain: Second- Through Fifth-Order Effects

With the architecture constraints and competitive landscape established, the strategic effects chain becomes clear:

#### First-order effect: NVIDIA needs a native tile compiler layer, not just libraries

The requirements pin both `cuda-tile==1.2.0` and `nvidia-cuda-tileiras==13.2.51`, while the benchmark imports internal compiler APIs such as `compile_tile`, `default_tile_context`, and `CompilerOptions`. The investigation flow can dump bytecode and final TileIR directly from the compiler pipeline.[^11][^12][^25]

That implies cuTile is not just another kernel helper. It is a compiler/runtime substrate. NVIDIA needs this because library APIs alone (cuBLAS, CUTLASS) do not give it a programmable, introspectable, optimizable middle layer between user code and final GPU kernels.[^11][^12]

#### Second-order effect: owning tile IR means owning the autotuning and correctness loops

The repo shows that performance hinges on non-linear tile choice, occupancy interaction, and memory behavior, while a simple cost model fails completely (R²=0.198, 0% prediction accuracy). It also shows that correctness issues can live in IR lowering decisions, as with int8 narrowing/widening.[^24][^26][^27]

The strategic value of cuTile is not only “generate kernels.” It is also “collect data, diagnose compiler decisions, and eventually automate those decisions.” A vendor that does not own this layer has to rely on handwritten kernels, user-authored DSLs, or opaque library dispatch; a vendor that does own it can fold search, validation, and lowering into one stack.[^10][^12][^26]

Google already has this loop: XLA profiles, fuses, and tiles automatically based on hardware characteristics. Cerebras’s compiler does it at graph level. NVIDIA’s cuBLAS/CUTLASS stack has hand-tuned heuristics, but no introspectable, data-driven autotuning loop at the IR level. cuTile fills this gap.[W:8][W:11]

#### Third-order effect: cuTile absorbs Triton rather than losing to it

If NVIDIA had not built Tile IR, the likely outcome is that Triton would become the de facto tile compiler for GPUs, with NVIDIA reduced to providing a PTX target. By building Tile IR and integrating it as a Triton backend, NVIDIA converts a potential disintermediation into a value-add: “Use Triton, but use it with our backend for 2–6× better performance on Blackwell.”[W:3][W:18][W:21]

This is the classic platform strategy of **absorbing the complement**. Triton is not killed; it is co-opted. Developers who use Triton+TileIR are functionally using NVIDIA’s compiler stack, even if they think they are using an “open” DSL.

**Repo evidence**: The benchmark measures cuTile and Triton with comparable methodology (both get tile sweeps in `reports/fp16_focus.py`). At 4096×4096 FP16, cuTile reaches 88.70 TFLOP/s vs Triton’s ~86 TFLOP/s in the best configs. The gap is narrow—which is the point. cuTile does not need to crush Triton; it needs to be competitive enough that NVIDIA can offer Tile IR as the optimization backend for Triton code.[^8][^15]

#### Fourth-order effect: Tile IR becomes a platform standard via MLIR alignment

By building on MLIR (the same infrastructure as XLA, ROCm, IREE, and TT-Forge), NVIDIA positions Tile IR as the natural “NVIDIA dialect” in a multi-vendor compiler world.[W:2][W:6][W:7]

This has several cascading implications:
1. **Academic adoption**: Researchers who write MLIR passes can target Tile IR. This creates a pipeline of compiler innovation that flows toward NVIDIA.
2. **Framework integration**: PyTorch, JAX, and future ML frameworks can emit Tile IR directly, bypassing both CUDA C++ and Triton. NVIDIA becomes the “native target” for framework compilers.
3. **Cross-generation portability**: Code written against Tile IR today runs on Ampere, Ada, Hopper, and Blackwell. Tomorrow it runs on Rubin without rewrites. PTX offers forward compatibility too, but at the wrong abstraction level for tensor workloads.
4. **Vendor lock-in with open-source dressing**: The IR is open; the best hardware to run it on is not. This is the same strategy that made x86 “open” while Intel controlled the performance frontier.

#### Fifth-order effect: the long game is broader than GEMM and broader than AI

Even though this repo is GEMM-specific, the “hello world” example is vector addition, and the core abstractions are general tile operations (`load`, `store`, `zeros`, `matmul`, `launch`, compiler options) rather than a hard-coded GEMM API.[^33][^5]

The broader target is any computation that benefits from tiled execution:
- **Attention kernels**: FlashAttention, SDPA variants—already demonstrated via TiledAttention in PyTorch.[W:24]
- **Convolutions and stencils**: Classic HPC and simulation workloads that are tile-natural.
- **Tensor contractions**: Quantum chemistry, computational biology.
- **Data-parallel reductions**: Analytics, genomics.
- **Graph neural networks**: Sparse-dense hybrid tiling.

If Tile IR succeeds, NVIDIA’s GPU becomes the native target not just for LLM training/inference, but for the entire spectrum of tiled computation across scientific computing, data analytics, and emerging AI architectures. That is the fifth-order effect: **cuTile is not an AI feature; it is a platform play for the next decade of accelerated computing.**[W:24][W:25]

---

### V. Summary: The Five Forces Driving cuTile

| Order | Effect | Evidence |
|---|---|---|
| **1st** | NVIDIA needs a tile compiler layer, not just libraries | Repo pins `cuda-tile`, exposes `compile_tile`, `CompilerOptions`; compiler pipeline is introspectable[^11][^12] |
| **2nd** | Owning tile IR = owning autotuning + correctness | Cost model fails (R²=0.198); int8 IR reveals lowering bugs; performance is non-linear in tile choice[^24][^26][^27] |
| **3rd** | Absorb Triton rather than be disintermediated by it | Tile IR backend for Triton ships 2025; repo benchmarks cuTile vs Triton on equal footing[^15][W:18] |
| **4th** | MLIR alignment makes Tile IR the platform standard | Open-sourced Apache 2.0; shared infra with XLA, ROCm, IREE[W:6][W:7] |
| **5th** | Beyond GEMM/AI: platform play for all tiled computation | General `load`/`store`/`zeros` API; TiledAttention in PyTorch; stencils, reductions, contractions[^33][W:24] |

And the meta-driver: **every major competitor already has a tile-native compiler**. Google (XLA/Pallas), Cerebras (graph-to-wafer), AMD (ROCm+Triton), Tenstorrent (TT-Forge). NVIDIA without cuTile would be the only major AI accelerator vendor whose primary developer interface is still thread-centric. That is an existential risk to the CUDA moat, and cuTile is the answer.[W:8][W:11][W:14][W:17]

## Key Repositories Summary

| Repository | Purpose | Key Files |
|---|---|---|
| [`trungnt13/cutile-rtx3090-benchmark`](https://github.com/trungnt13/cutile-rtx3090-benchmark) | Published benchmark artifact for evaluating cuTile on RTX 3090 | `README.md`, `benchmarks/core.py`, `reports/full_report.py`, `reports/fp16_focus.py`, `investigations/int8_ir/export_int8_ir.py`[^34] |

## What I Am Certain About vs. What I Infer

### Certain

The repo is a benchmark artifact, not a general library surface; the published artifacts target Linux RTX 3090; the cuTile path is compiler/JIT-driven; FP16/BF16 can be competitive when tuned; int8 is semantically broken for exact GEMM; the cost model fails; and the written methodology overstates current rectangular-GEMM coverage.[^1][^3][^5][^7][^23][^26][^31]

### Inferred (strengthened by external evidence)

I infer that NVIDIA is developing cuTile to own a first-party tile IR/compiler layer, to build an autotuning/validation loop under vendor control, to absorb Triton rather than be disintermediated by it, and to set the MLIR-based Tile IR as the platform standard for tiled GPU programming. The repo strongly supports this inference through its competitive framing against Triton, its exposure of compiler internals, and its general-purpose tile API design.[^11][^12][^15]

External evidence now confirms and extends this inference: NVIDIA has shipped a CUDA Tile IR backend for OpenAI Triton[W:18], open-sourced Tile IR under Apache 2.0[W:6][W:7], and explicitly positioned cuTile/Tile IR as the programming model for Blackwell’s hardware features (TMEM, UMMA, CTA-pairs) that PTX cannot express[W:3][W:5]. The competitive landscape—Google TPU/XLA/Pallas[W:8], Cerebras WSE[W:11], AMD MI300/ROCm[W:14], Tenstorrent TT-Forge[W:17]—confirms that every major AI accelerator vendor already has or is building tile-native compiler stacks, making cuTile existentially necessary for NVIDIA.

## Confidence Assessment

**High confidence** on repo purpose, architecture, timing policy, backend roles, int8 failure mode, tile-selection behavior, and benchmark limitations because they are directly stated in the code, docs, and committed artifacts.[^1][^2][^5][^7][^23][^26]

**High confidence** (upgraded from moderate) on the strategic interpretation, because external evidence from NVIDIA’s own developer blog, open-source releases, Triton backend integration, and industry analysis now corroborates what was previously inference from repo-internal evidence alone.[W:1][W:6][W:18][W:21]

**Moderate confidence** on competitive-landscape claims (TPU adoption numbers, Cerebras perf/watt, AMD parity) because these come from vendor marketing, industry journalism, and analyst reports rather than independently verified benchmarks.[W:8][W:11][W:14]

**Low uncertainty** that some benchmark artifacts were generated outside the current checkout, because the committed system artifact points to a Linux `/home/...` path while the present workspace is on macOS. That affects reproducibility expectations locally, but not the interpretation of the committed evidence.[^14]

## Footnotes

[^1]: `cutile/README.md:3-18` (repo HEAD `267e9e1ff060dded616d061f03a541d3f536a2d0`)
[^2]: `cutile/docs/methodology.md:3-31,66-101` (repo HEAD `267e9e1ff060dded616d061f03a541d3f536a2d0`)
[^3]: `cutile/artifacts/system/benchmark_system.md:1-25` (repo HEAD `267e9e1ff060dded616d061f03a541d3f536a2d0`)
[^4]: `cutile/docs/implementation.md:3-12,13-41,63-77` (repo HEAD `267e9e1ff060dded616d061f03a541d3f536a2d0`)
[^5]: `cutile/benchmarks/core.py:339-411,504-609` (repo HEAD `267e9e1ff060dded616d061f03a541d3f536a2d0`)
[^6]: `cutile/reports/full_report.py:29-52,151-331,372-469,715-825,828-879` (repo HEAD `267e9e1ff060dded616d061f03a541d3f536a2d0`)
[^7]: `cutile/RESULTS.md:5-39,57-80` (repo HEAD `267e9e1ff060dded616d061f03a541d3f536a2d0`)
[^8]: `cutile/reports/fp16_focus.py:38-55,125-167,520-667` and `cutile/artifacts/fp16_focus/cutile_fp16_optimization_summary.md:1-27` (repo HEAD `267e9e1ff060dded616d061f03a541d3f536a2d0`)
[^9]: `cutile/docs/cutile_tile_selection_rtx3090.md:19-82` (repo HEAD `267e9e1ff060dded616d061f03a541d3f536a2d0`)
[^10]: `cutile/artifacts/ncu/ncu_profile_summary.json:1-46` and `cutile/RESULTS.md:57-71` (repo HEAD `267e9e1ff060dded616d061f03a541d3f536a2d0`)
[^11]: `cutile/requirements.txt:1-47` (notably `cuda-tile==1.2.0` and `nvidia-cuda-tileiras==13.2.51`) (repo HEAD `267e9e1ff060dded616d061f03a541d3f536a2d0`)
[^12]: `cutile/benchmarks/core.py:19-20,339-411` and `cutile/investigations/int8_ir/export_int8_ir.py:15-17,58-60,83-93` (repo HEAD `267e9e1ff060dded616d061f03a541d3f536a2d0`)
[^13]: `cutile/README.md:92-102` and `cutile/docs/tradeoffs.md:24-45` (repo HEAD `267e9e1ff060dded616d061f03a541d3f536a2d0`)
[^14]: `cutile/README.md:11-12,41-52` and `cutile/artifacts/system/benchmark_system.md:3-15` (repo HEAD `267e9e1ff060dded616d061f03a541d3f536a2d0`)
[^15]: `cutile/docs/tradeoffs.md:3-45` (repo HEAD `267e9e1ff060dded616d061f03a541d3f536a2d0`)
[^16]: `cutile/benchmarks/core.py:420-684` (repo HEAD `267e9e1ff060dded616d061f03a541d3f536a2d0`)
[^17]: `cutile/benchmarks/ptx_latency_validation.py:34-77` and `cutile/artifacts/full/ptx_latency_validation_summary.json:1-11` (repo HEAD `267e9e1ff060dded616d061f03a541d3f536a2d0`)
[^18]: `cutile/docs/methodology.md:44-56` and `cutile/benchmarks/core.py:588-593` (repo HEAD `267e9e1ff060dded616d061f03a541d3f536a2d0`)
[^19]: `cutile/benchmarks/ptx_iteration_study.py:49-152,306-329` and `cutile/artifacts/ptx_iterations/ptx_iteration_analysis.md:1-21` (repo HEAD `267e9e1ff060dded616d061f03a541d3f536a2d0`)
[^20]: `cutile/artifacts/fp16_focus/cutile_fp16_optimization_summary.md:7-15` and `cutile/RESULTS.md:18-39` (repo HEAD `267e9e1ff060dded616d061f03a541d3f536a2d0`)
[^21]: `cutile/artifacts/full/report_summary.md:23-30` (repo HEAD `267e9e1ff060dded616d061f03a541d3f536a2d0`)
[^22]: `cutile/artifacts/full/report_summary.md:1-8,31-56` (repo HEAD `267e9e1ff060dded616d061f03a541d3f536a2d0`)
[^23]: `cutile/investigations/int8_ir/summary.json:1-60` (repo HEAD `267e9e1ff060dded616d061f03a541d3f536a2d0`)
[^24]: `cutile/investigations/int8_ir/mm_i8.cutileir.txt:37-44` (repo HEAD `267e9e1ff060dded616d061f03a541d3f536a2d0`)
[^25]: `cutile/investigations/int8_ir/export_int8_ir.py:38-45,61-116` (repo HEAD `267e9e1ff060dded616d061f03a541d3f536a2d0`)
[^26]: `cutile/benchmarks/cost_model.py:35-149` (repo HEAD `267e9e1ff060dded616d061f03a541d3f536a2d0`)
[^27]: `cutile/artifacts/cost_model/cost_model_results.json:1-212` (repo HEAD `267e9e1ff060dded616d061f03a541d3f536a2d0`)
[^28]: `cutile/artifacts/fp16_focus/fp16_raw.csv:42-56` (repo HEAD `267e9e1ff060dded616d061f03a541d3f536a2d0`)
[^29]: `cutile/artifacts/full/benchmark_best.csv:82-106` and `cutile/RESULTS.md:34-39` (repo HEAD `267e9e1ff060dded616d061f03a541d3f536a2d0`)
[^30]: `cutile/benchmarks/core.py:28-36` (repo HEAD `267e9e1ff060dded616d061f03a541d3f536a2d0`)
[^31]: `cutile/reports/full_report.py:157-161,657-661` (repo HEAD `267e9e1ff060dded616d061f03a541d3f536a2d0`)
[^32]: `cutile/benchmarks/core.py:205-257` (repo HEAD `267e9e1ff060dded616d061f03a541d3f536a2d0`)
[^33]: `cutile/examples/hello_cutile.py:1-31` (repo HEAD `267e9e1ff060dded616d061f03a541d3f536a2d0`)
[^34]: `cutile/README.md:20-40` and `cutile/docs/implementation.md:5-12` (repo HEAD `267e9e1ff060dded616d061f03a541d3f536a2d0`)

## Web Sources (External Evidence)

*These sources were consulted in July 2025 to corroborate and extend the repo-internal strategic analysis.*

[W:1]: [Tile IR Introduction — NVIDIA Documentation Hub](https://docs.nvidia.com/cuda/tile-ir/latest/sections/introduction.html) — Official introduction to Tile IR as NVIDIA's new MLIR-based intermediate representation for tile-based GPU programming.

[W:2]: [NVIDIA TileIR Internals: from CuTile to MLIR/LLVM to SASS](https://maknee.github.io/blog/2026/NVIDIA-TileIR-Internals-from-CuTile-to-MLIR-LLVM-to-SASS/) — Deep technical analysis of the cuTile → TileIR → MLIR → LLVM → SASS compilation pipeline.

[W:3]: [NVIDIA Releases CUDA 13.1 with New Tile Programming Model for Blackwell](https://aihola.com/article/nvidia-cuda-tile-programming) — Coverage of CUDA 13.1 launch with Tile IR and Blackwell-specific features (TMEM, UMMA, CTA-pairs).

[W:4]: [Microbenchmarking NVIDIA's Blackwell Architecture (arXiv)](https://arxiv.org/html/2512.02189v1) — Academic microbenchmarks of Blackwell TMEM showing 58% latency reduction.

[W:5]: [CUTLASS Tutorial: Writing GEMM Kernels Using Tensor Memory For NVIDIA Blackwell GPUs](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/) — Technical tutorial demonstrating TMEM and UMMA instructions that PTX cannot express.

[W:6]: [NVIDIA open-sources CUDA Tile IR on GitHub — Dataconomy](https://dataconomy.com/2025/12/26/nvidia-open-sources-cuda-tile-ir-on-github/) — Coverage of Apache 2.0 open-source release of CUDA Tile IR.

[W:7]: [NVIDIA CUDA Tile IR Open-Sourced — Phoronix](https://www.phoronix.com/news/NVIDIA-CUDA-Tile-IR-Open-Source) — Technical coverage of the open-source release including MLIR dialect, Python bindings, and conformance suite.

[W:8]: [Google TPU vs NVIDIA GPU — Introl Blog](https://introl.com/blog/google-tpu-vs-nvidia-gpu-infrastructure-decision-framework-2025) — Comparative analysis of TPU vs GPU infrastructure including cost metrics (Midjourney 65%+ cost reduction on TPU).

[W:9]: [Google's New AI Chip Is Shaking Nvidia's Dominance — Observer](https://observer.com/2025/12/google-ai-chip-tpu-nvidia-challenge/) — Industry analysis of Google Ironwood TPUv7 competitive positioning.

[W:10]: [Inside the Ironwood TPU Codesigned AI Stack — Google Cloud Blog](https://cloud.google.com/blog/products/compute/inside-the-ironwood-tpu-codesigned-ai-stack/) — Google's official technical overview of TPUv7 hardware-software co-design including XLA and Pallas.

[W:11]: [Cerebras CS-3 vs. Nvidia B200: 2024 AI Accelerators Compared — Cerebras Blog](https://www.cerebras.ai/blog/cerebras-cs-3-vs-nvidia-b200-2024-ai-accelerators-compared) — Cerebras's comparison of WSE-3 (4T transistors, 900K cores, 125 PFLOP/s) against NVIDIA B200.

[W:12]: [Cerebras WSE3 Versus Nvidia B200 — NextBigFuture](https://www.nextbigfuture.com/2025/05/cerebras-wse3-versus-nvidia-b200.html) — Independent analysis of Cerebras 2× perf/watt advantage at scale.

[W:13]: [How Cerebras is Breaking the GPU Bottleneck on AI Inference — VentureBeat](https://venturebeat.com/ai/how-cerebras-is-breaking-the-gpu-bottleneck-on-ai-inference) — Coverage of Cerebras's graph-to-wafer compiler model.

[W:14]: [AMD MI300 & AI Hardware: A Developer's Deep Dive](https://vife.ai/blog/amd-mi300-ai-hardware-deep-dive) — Technical analysis of MI300X (192GB HBM3, 5.3 TB/s bandwidth).

[W:15]: [Accelerating DeepSeek Inference with AMD MI300 — Microsoft Tech Community](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/accelerating-deepseek-inference-with-amd-mi300-a-collaborative-breakthrough/4407673) — Microsoft's validation of AMD MI300 for production LLM inference.

[W:16]: [Unlock DeepSeek-R1 Inference Performance on AMD Instinct MI300X — ROCm Blog](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1_Perf/README.html) — AMD's benchmarks showing ROCm inference parity with NVIDIA for DeepSeek-R1.

[W:17]: [Tenstorrent — Official Site](https://tenstorrent.com/) — Tenstorrent's open MLIR-based TT-Forge compiler and Blackhole chip for sovereign AI infrastructure.

[W:18]: [Advancing GPU Programming with the CUDA Tile IR Backend for OpenAI Triton — NVIDIA Developer Blog](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/) — NVIDIA's official announcement of Tile IR as a Triton backend, enabling Triton kernels to leverage Blackwell-specific optimizations.

[W:19]: [How Nvidia's CUDA Monopoly In Machine Learning Is Breaking — ReadWise](https://readwise.io/reader/shared/01hgv7jr3rpfhde3qdf9dvbajq/) — Analysis of Triton and hardware-agnostic compilers eroding CUDA lock-in.

[W:20]: [Nvidia's AI Moat in 2025: A Deep Dive — SundeepTeki.org](https://www.sundeepteki.org/blog/nvidias-ai-moat-in-2025-a-deep-dive) — Comprehensive analysis of NVIDIA's software moat narrowing while hardware moat widens.

[W:21]: [OpenAI Triton on NVIDIA Blackwell Boosts AI Performance and Programmability — NVIDIA Developer Blog](https://developer.nvidia.com/blog/openai-triton-on-nvidia-blackwell-boosts-ai-performance-and-programmability/) — NVIDIA's blog on 2-6× speedups from Tile IR backend for Triton on Blackwell.

[W:22]: [Nvidia Open-Sources CUDA Tile IR: Did They End the Moat? — ByteIota](https://byteiota.com/nvidia-open-sources-cuda-tile-ir-did-they-end-the-moat/) — Analysis of open-sourcing as both moat expansion and moat defense.

[W:23]: [GitHub — NVIDIA/cuda-tile: CUDA Tile IR](https://github.com/NVIDIA/cuda-tile) — Official CUDA Tile IR repository (Apache 2.0), including MLIR dialect, Python bindings, bytecode serialization, and conformance suite.

[W:24]: [TiledAttention: a CUDA Tile SDPA Kernel for PyTorch — arXiv](https://arxiv.org/html/2603.01960v1) — Academic paper demonstrating cuTile applied to Scaled Dot-Product Attention, proving generality beyond GEMM.

[W:25]: [CUDA 13.1 Reinvents GPU Development — The Biggest Leap in Two Decades](https://www.buysellram.com/blog/cuda-13-1-reinvents-gpu-development-the-biggest-leap-in-two-decades/) — Industry analysis positioning Tile IR as a generational platform shift.
