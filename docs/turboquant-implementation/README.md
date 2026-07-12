# TurboQuant Core Implementation Packet

This folder contains the `mlx-swift` side of the TurboQuant implementation train. `mlx-swift` is the primitive layer: it owns storage layouts, Metal kernels, capability probes, storage estimates, validators, attention decisions, benchmark primitives, hidden-copy audit, kernel warmup, the Layout V6 default write layout, and V4/V5 compatibility.

Cross-repo release-train docs live in:

```text
/Users/mt/Programming/Schtack/pines/docs/turboquant-implementation
```

This repo-local packet defines what core workers must implement and expose.

Current status:

- The active Pines compatibility pair is non-green. Core operator coverage is
  necessary but not sufficient for product promotion.
- Dense K8/V4 is the current compressed real-model reference in the latest Mac
  baseline. K8/V3 and K8/V2 remain guarded measured candidates.
- Native Sparse-V threshold, top-k, cumulative-mass, hybrid, pageTopK, and
  CandidateSparse diagnostics are exposed by the core report surfaces, but the
  current sparse family is rejected for promotion. Dense compressed AV remains
  the reference/fallback, and sparse modes are explicit proof/debug routes only.
- Real-device model/device/mode verification is still owned by the Pines
  evidence pipeline and is not implied by this core branch alone.

The executable launch order is owned by the Pines packet:

```text
/Users/mt/Programming/Schtack/pines/docs/turboquant-implementation/14-worker-launch-schedule.md
```

The PR and merge train is owned by:

```text
/Users/mt/Programming/Schtack/pines/docs/turboquant-implementation/15-pr-merge-plan.md
```

For this repo, the launch order is:

1. Wave 0: W1 core contracts.
2. Wave 1: W2 validation/router.
3. Wave 3: W3 benchmark JSON and hidden-copy audit.
4. Wave 5: W13 Layout V5 and kernels.
5. Wave 7: W29+ platform contracts for adaptive precision, open KV descriptors, and feature gates.

W1 can run immediately in parallel with LM W4 and Pines W7/W24. W13 must not product-activate before benchmark, quality, and memory-calibration gates exist.

Worker PRs for the completed train targeted:

```text
tq/wave7-core-platform
```

Final merge to the repo default branch should preserve the cross-repo
compatibility pair and Pines production pin gate recorded in the Pines packet.

## Core responsibilities

`mlx-swift` owns:

- TurboQuant presets;
- compressed storage layout;
- Metal encode/decode kernels;
- compressed attention kernels;
- path-specific capability probes;
- symbolic and actual storage estimates;
- canonical validators;
- attention-path decision routing;
- benchmark JSON output;
- hidden-copy audit;
- kernel warmup;
- Layout V6 default layout and V4/V5 compatibility;
- fused head-dimension specializations.
- Wave 7 platform policy contracts, disabled by default.

## Required reading

1. [Core Worker Cards](worker-cards.md)
2. [Core Contracts](core-contracts.md)
3. [Validation and Attention Router](validation-router.md)
4. [Benchmark JSON and Hidden-Copy Audit](benchmark-json-hidden-copy.md)
5. [Current Paths and Benchmarks](current-paths-and-benchmarks.md)
6. [Layout V5 and Kernels](layout-v5-kernels.md)

## Non-negotiables

1. Bad compressed layout fails before Metal dispatch.
2. Every rejected path has a reason.
3. Capabilities are path-specific; broad pass/fail gates are not sufficient.
4. bfloat output is gated independently.
5. Linear matmul remains disabled unless separately certified.
6. Long-KV hot paths must not accidentally create full-cache row-contiguous copies.
7. Benchmark JSON must include storage estimates and path decisions.
8. Layout V6 is the production/default/current write layout; V4 and V5 remain supported for compatibility, and V5/fp16-scale-only paths stay explicitly gated.
9. Wave 7 platform features must fail closed when unsupported or enabled without evidence.
