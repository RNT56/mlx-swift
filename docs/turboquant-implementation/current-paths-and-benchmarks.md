# Current TurboQuant Core Paths And Benchmarks

This document is the runnable inventory for the implemented `mlx-swift` core
TurboQuant surfaces. It focuses on primitive/operator coverage and the JSON
fields that downstream LM and Pines evidence consume.

## Core Attention Paths

| Path | CLI value | Backend report | Current role |
| --- | --- | --- | --- |
| Raw SDPA baseline | `baseline` | `rawSDPA` | FP16 reference route. |
| Native MLX compressed | `native-mlx` | `nativeMLX` | Native segmented API/capability surface. Requires backend availability evidence. |
| Affine int4 native | `affine-int4-native` | `nativeMLX` | Router label only; the benchmark reports it as not separately callable until a public primitive dispatcher exists. |
| Affine K8/V4 native | `affine-k8v4-native` | `nativeMLX` | Main K8/V4 speed candidate; used by real-model parity as `affineK8V4`. |
| Affine K8/Vx native | `affine-k8vx-native` | `nativeMLX` | Lower-V K8/V3 and K8/V2 candidate path. |
| Residual affine K8/Vx native | `affine-k8vx-residual` | `nativeMLX` | Lower-V residual candidate label; shares the native segmented primitive in the current benchmark surface. |
| Sparse-V native | `native-mlx` with Sparse-V options | `nativeMLXCompressed` / `nativeMLX` | Rejected promotion family kept for explicit diagnostics: threshold, top-k, cumulative-mass, hybrid, block-threshold, pageTopK, and candidateSparse. |
| Online fused Polar/QJL | `online-fused` | `swiftMetalKernel` | Fused compressed-domain TurboQuant route. |
| Tiled online fused | `tiled-online-fused` | `swiftMetalKernel` | Long-context tiled fused route. |
| Sparse-value two-stage compressed | `sparse-value-two-stage` | `swiftMetalKernel` | Explicit Sparse-V path label. Current core JSON reports it as requested/skipped/not-callable and emits native Sparse-V evidence through native rows. |
| Two-stage compressed | `two-stage` | `swiftMetalKernel` | Debug/fallback comparison path. |
| MLX packed fallback | `mlx-packed-fallback` | `decodedReference` | Compatibility fallback, not a promotion path. |
| Unavailable sentinel | `unavailable` | `unavailable` | JSON/report sentinel only. |

## Implemented Core Optimizations

- K8/V4 and K8/Vx mixed affine SDPA with native capability reporting.
- Sparse-V diagnostics for threshold, top-k, cumulative-mass, and hybrid
  cumulative-plus-top-k value skipping. Threshold and cumulative modes use
  normalized softmax weights; pure TopK can select in score order because it is
  monotonic with softmax weight order. Dense compressed AV remains the
  reference/fallback.
- Experimental `blockThreshold` Sparse-V (`--sparse-v blockThreshold`) is
  decode-only and split-path only. It computes normalized softmax mass per
  512-token block and skips whole V blocks below `--sparse-v-threshold`.
- Experimental `pageTopK` (`--sparse-v pageTopK`) is decode-only and supports
  GQA when query heads are a positive multiple of KV heads. It selects
  512-token KV pages before exact QK/AV and computes attention over retained
  pages only. This is page-sparse attention, not exact Sparse-V, and remains an
  explicit research path. The sampled page scorer reports native `kernelKind`
  13. The cached key-page-summary two-dispatch scorer reports native
  `kernelKind` 14. The fused cached-summary pageTopK decode path reports native
  `kernelKind` 15 for `topK <= 8`; disabling
  `TURBOQUANT_SPARSE_V_PAGE_FUSED=0` or using a larger page TopK returns to
  kernel 14, while missing/unsafe summaries fall back to sampled scoring rather
  than dense fallback. `TURBOQUANT_SPARSE_V_PAGE_RECENT_TOKENS=<n>` is an
  off-by-default research knob that keeps an exact recent-token floor in
  addition to selected older pages; recency variants report kernel kinds 16
  (sampled), 17 (cached two-dispatch), or 18 (fused cached-summary).
- Experimental `candidateSparse` (`--sparse-v candidateSparse`) is decode-only.
  It keeps an exact recent window, scores older 512-token pages from cached
  key candidate sketches, computes exact QK for selected pages, then retains an
  older-token TopK before AV. The prototype native path reports `kernelKind` 19.
  Candidate sketches are valid for pinned-prefix logical layouts and invalid for
  ring-offset layouts; missing/unsafe sketches route to dense compressed
  fallback with requested-but-inactive diagnostics.
- Decode-only single-block Sparse-V TopK uses the native fused kernel instead of
  the split score/select/partials/reduce pipeline. Equal-score TopK ties retain
  lower token indices until the requested TopK is reached.
- Decode-only split Sparse-V TopK now has a compact kernel-kind-12 path. For
  shorter split contexts it performs score-domain TopK selection and AV
  accumulation in one fused dispatch after block/global score stats. At 16+
  active blocks it first extracts per-block local TopK candidates, then fuses
  global candidate selection with AV accumulation. Threshold, cumulative-mass,
  and hybrid Sparse-V modes remain on the older normalized-weight split path.
  A fused GQA block-stats-plus-candidate experiment was measured after this
  route and is not currently dispatched because it regressed 8192/32768 latency.
- Qwen GQA4/HD256 benchmark shape support: `--head-dim 256 --query-heads 16
  --kv-heads 4 --query-length 1`.
- Long-context quantized decode block scheduling: decode-shaped 32K+ K8/Vx
  attention uses smaller split blocks to avoid overlong Metal command buffers.
- Block-token recommendation fields for 32K/64K/128K/256K decode.
- p50/p95 latency and decode-throughput reporting.
- FP16 raw-SDPA reference timing, per-path speed ratios against that reference,
  KV byte estimates, actual bits/value, memory bytes saved, memory reduction
  ratio/percent, and hidden-copy audit fields.
- `pathMeasurements` rows for every `TurboQuantAttentionPath` enum case. Rows
  are marked `measured`, `reference`, `skipped`, `failed`, `notCallable`, or
  `unavailable` so downstream evidence can distinguish true primitive timings
  from labels/fallbacks that are not independently dispatchable for the current
  shape.
- Sparse-V report fields: skipped/considered tokens, retained mass, max error,
  cosine, fallback reason, and optional per-layer/per-head diagnostics.

## Core Operator JSON

Build:

```bash
swift build --product TurboQuantBenchmark -c release
```

Run a single Qwen-shaped cell:

```bash
.build/release/TurboQuantBenchmark \
  --json --include-timestamp \
  --iterations 12 --warmup 3 \
  --cooldown-ms 0 --path-cooldown-ms 0 \
  --context 32768 \
  --preset turbo4v2 \
  --path affine-k8v4-native \
  --head-dim 256 --query-heads 16 --kv-heads 4 --query-length 1
```

Run the current broad core matrix through the LM-side artifact script:

```bash
cd /Users/mt/Programming/Schtack/mlx-forks/mlx-swift-lm
TQ_SKIP_REAL_MODEL=1 scripts/run-turboquant-current-benchmarks.sh
```

Run a native Sparse-V microbench row:

```bash
cd /Users/mt/Programming/Schtack/mlx-forks/mlx-swift
swift run -c release TurboQuantBenchmark \
  --iterations 5 --warmup 2 \
  --context 512 --query-length 1 \
  --sparse-v topK --sparse-v-top-k 128
```

Run the experimental page-sparse probe:

```bash
cd /Users/mt/Programming/Schtack/mlx-forks/mlx-swift
swift run TurboQuantBenchmark \
  --iterations 5 --warmup 1 \
  --context 65536 --query-heads 4 --kv-heads 1 \
  --sparse-v pageTopK --sparse-v-top-k 1
```

Add `--sparse-v-page-summary` to route explicit `pageTopK` through cached
key-page summaries. With valid summaries and retained page count `<= 8`, the
current fast path is fused `kernelKind` 15. Set
`TURBOQUANT_SPARSE_V_PAGE_FUSED=0` to A/B against cached two-dispatch
`kernelKind` 14.

The legacy JSON report includes `attention.native_sparse.<mode>` rows with dense
native reference latency, selection parameters, skipped/total value tokens, skip
ratio, active blocks, block tokens, and native kernel kind.

## Report Fields To Preserve

Every benchmark row that reaches Pines evidence must retain:

- route, runtime mode, backend, selected path, fallback reason;
- full `pathMeasurements` coverage for every attention path enum case, including
  status/reason for skipped, unavailable, fallback, and non-callable labels;
- context tokens, head dimension, query length, preset, value bits, group size;
- p50/p95 attention latency and decode tokens/sec;
- FP16 raw-SDPA p50/p95 reference latency and compressed-vs-raw p50/p95 speed
  ratio where measured;
- compressed bytes, FP16 raw-SDPA KV bytes, memory bytes saved, memory reduction
  ratio/percent, actual bits/value;
- cooldown controls used for the run (`cooldownMS` and `pathCooldownMS`);
- hidden-copy audit status;
- Sparse-V diagnostics when enabled;
- hybrid selector fields when the LM surface emits them.

Current Sparse-V diagnostics include mode, skipped tokens, considered tokens,
retained mass, dense cosine, max error, fallback reason, and optional
per-head/per-layer records. These fields are reportable now, but product
promotion remains blocked until `mlx-swift-lm` real-model evidence and Pines
real-device evidence pass.

Current synthetic native Sparse-V TopK evidence from
`mlx-swift/artifacts/turboquant-sparsev-final-routing-20260602T0340Z`:

| Context | Mode | Kernel kind | Active blocks | Skip ratio | Sparse/dense native speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| 512 | `topK(128)` | 5 | 1 | 75.00% | 2.95x |
| 2048 | `topK(128)` | 12 | 4 | 93.75% | 2.77x |
| 4096 | `topK(128)` | 12 | 8 | 96.88% | 0.71x |
| 8192 | `topK(128)` | 12 | 16 | 98.44% | 0.55x |
| 32768 | `topK(128)` | 12 | 64 | 99.61% | 0.76x |

These rows are microbench evidence only. They show the compact TopK work is a
real short-context improvement, but they also show Sparse-V is still not a
long-context promotion candidate on this machine. The 4096+ rows are slower than
dense K8/V4, so auto policy must continue routing to dense K8/V4 unless a
model/device/profile-specific real-model artifact proves the promotion gate.

Current block-threshold follow-up evidence from
`artifacts/turboquant-sparsev-e2e-20260602T0200Z/block-threshold`:

| Context | Threshold | Kernel kind | Active blocks | Skip ratio | Sparse/dense native speedup | Quality |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2048 | `0.20` | 7 | 4 | 0.00% | 1.44x | pass-like synthetic cosine `0.9999999` |
| 2048 | `0.30` | 7 | 4 | 100.00% | 2.37x | fail, cosine `0.0` |
| 8192 | `0.05` | 7 | 16 | 0.00% | 0.73x | pass-like synthetic cosine `1.0000001` |
| 8192 | `0.065` | 7 | 16 | 100.00% | 0.84x | fail, cosine `0.0` |
| 32768 | `0.015` | 7 | 64 | 0.00% | 0.60x | pass-like synthetic cosine `1.0000001` |

Block-threshold is therefore an implemented diagnostic/research mode, not a
promotion candidate. In this synthetic setup it collapses to no-skip or
all-skip near the uniform block-mass boundary, and no-skip long-context rows are
slower than dense.

Current pageTopK follow-up evidence from
`artifacts/turboquant-sparsev-e2e-20260602T0200Z/optimization-status.md`:

| Context | Retained pages | Kernel kind | Active pages | Skip ratio | Sparse/dense native speedup | Quality |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 8192 | 4 | 13 | 16 | 75.00% | 0.38x | synthetic cosine `0.9954657` |
| 32768 | 8 | 13 | 64 | 87.50% | 0.40x | synthetic cosine `0.9987608` |
| 32768 | 1 | 13 | 64 | 98.44% | 1.03x | synthetic cosine `0.9982831` |
| 65536 | 1 | 13 | 128 | 99.22% | 2.46x | synthetic cosine `0.9991667` |

Current GQA pageTopK follow-up evidence from
`mlx-swift/artifacts/turboquant-sparsev-page-topk-gqa-20260602T1256Z`:

| Context | Q heads | KV heads | Retained pages | Kernel kind | Active pages | Skip ratio | Sparse/dense native speedup | Quality |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32768 | 4 | 1 | 1 | 13 | 64 | 98.44% | 1.25x | synthetic cosine `0.9994767`, max abs `0.00144341` |
| 65536 | 4 | 1 | 1 | 13 | 128 | 99.22% | 2.29x | synthetic cosine `0.9994995`, max abs `0.00146994` |

PageTopK is not promoted. It shows that page-sparse attention can beat dense at
very long context with extremely aggressive page retention, and the GQA route
now makes real decoder-model testing possible. Real-model quality and p50 decode
throughput must still pass before any auto routing.

Fused cached-summary pageTopK evidence from
`/Users/mt/Programming/Schtack/mlx-forks/artifacts/sparse-v-page-fused-20260602`:

| Context | Retained pages | Sampled 13 ms | Cached 14 ms | Fused 15 ms | Fused/sample | Fused/cached | Skip ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32768 | 1 | 0.872 | 0.894 | 0.782 | 1.11x | 1.14x | 98.44% |
| 32768 | 2 | 1.379 | 1.303 | 1.255 | 1.10x | 1.04x | 96.88% |
| 32768 | 4 | 2.294 | 2.312 | 2.252 | 1.02x | 1.03x | 93.75% |
| 32768 | 8 | 4.293 | 4.438 | 4.330 | 0.99x | 1.02x | 87.50% |
| 65536 | 1 | 1.006 | 0.862 | 0.847 | 1.19x | 1.02x | 99.22% |
| 65536 | 2 | 1.401 | 1.420 | 1.323 | 1.06x | 1.07x | 98.44% |
| 65536 | 4 | 2.452 | 2.357 | 2.329 | 1.05x | 1.01x | 96.88% |
| 65536 | 8 | 4.673 | 4.488 | 4.539 | 1.03x | 0.99x | 93.75% |
| 131072 | 1 | 1.057 | 1.073 | 0.883 | 1.20x | 1.21x | 99.61% |
| 131072 | 2 | 1.653 | 1.526 | 1.444 | 1.14x | 1.06x | 99.22% |
| 131072 | 4 | 2.688 | 2.676 | 2.645 | 1.02x | 1.01x | 98.44% |
| 131072 | 8 | 5.215 | 5.291 | 5.087 | 1.03x | 1.04x | 96.88% |

Fused `kernelKind` 15 removes the separate page-score dispatch and improves the
operator path in most measured rows, but pageTopK remains explicit research
until real-model quality and throughput gates pass.

Current CandidateSparse real-model evidence from
`/Users/mt/Programming/Schtack/mlx-forks/artifacts/candidate-sparse-20260603`:

| Context | Config | Kernel kind | Active layers | Skip ratio | Candidate tok/s | Dense K8/V4 tok/s | Ratio |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | `r256 p1 older64` | 19 | 6/6 | 38.46% | 4.13 | 42.96 | 0.096x |
| 2048 | `r512 p2 older128` | 19 | 6/6 | 68.81% | 4.10 | 46.47 | 0.088x |
| 8192 | `r512 p2 older128` | 19 | 6/6 | 92.19% | 2.06 | 42.26 | 0.049x |

This is an active diagnostics/prototype result, not a promotion result. The
current kernel still repeats candidate scoring and compressed K/V decode per
query head, and even very high skip ratios do not overcome that overhead.

The cooperative GQA fused follow-up, `kernelKind` 20, is also rejected as a
promotion path. On Qwen3.5-2B-4bit at context 2048, dense `affineK8V4` measured
16.32 tok/s, prototype CandidateSparse `kernelKind` 19 measured 2.60 tok/s, and
cooperative GQA fused `kernelKind` 20 measured 2.34 tok/s. Because kernel 20 is
both slower and more approximate, it is opt-in only with
`TURBOQUANT_CANDIDATE_SPARSE_FUSED=1`; default explicit CandidateSparse routes
to `kernelKind` 19.

CandidateSparse remains explicit proof/debug only unless a different architecture
can beat dense K8/V4 by the real-model promotion gate. Current runtime policy
must not auto-select Sparse-V/CandidateSparse.

Real-model Qwen3.5-2B-4bit fused pageTopK1 evidence:

- Kernel routing proof:
  `/Users/mt/Programming/Schtack/mlx-forks/artifacts/turboquant-acceptance-sparse-v-qwen35-2b-page-fused-diagnostic2-20260602T202338Z/pageTopK1.json`
  reports all 6 Sparse-V layers using `nativeKernelKind=15`, with
  `keyPageSummaryAvailable=true` and summary shape `1x2x2x4`.
- Bounded throughput artifact:
  `/Users/mt/Programming/Schtack/mlx-forks/artifacts/turboquant-acceptance-sparse-v-qwen35-2b-page-fused-final-20260602T202406Z/pageTopK1.out`
  completed throughput through 8192 before manual termination during the 2048
  quality row:

| Context | Dense K8/V4 tok/s | Fused pageTopK1 tok/s | Ratio vs K8/V4 | Skip ratio |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 72.76 | 6.53 | 0.090x | 0.78% |
| 2048 | 81.52 | 5.74 | 0.070x | 75.05% |
| 8192 | 76.27 | 6.04 | 0.079x | 93.75% |

- Completed quality artifact:
  `/Users/mt/Programming/Schtack/mlx-forks/artifacts/turboquant-acceptance-sparse-v-qwen35-2b-page-fused-quality-20260602T202641Z/pageTopK1-quality.json`
  fails at context 512: top-1 `1.0`, KL `0.069922`, p95 abs error `3.34375`,
  cosine `0.96738`.

Conclusion: fused cached-summary pageTopK is implemented and routed in the real
model, but it is rejected for promotion. It is materially slower than dense
K8/V4 and fails the p95 quality gate even at 512. Dense K8/V4 remains the
default production route; pageTopK remains explicit research/diagnostic only.
The optional recent-token floor is intended to test whether conservative
retention can recover quality; it is not enabled by default and does not change
promotion status without passing real-model throughput and quality gates.

## Promotion Rules

Core operator success is necessary but insufficient for product claims. A path is
not `Verified` or `Certified` until `mlx-swift-lm` real-model inference evidence
and Pines real-device evidence pass for the same model/profile/device/mode tuple.
Synthetic operator rows remain smoke and regression diagnostics.
