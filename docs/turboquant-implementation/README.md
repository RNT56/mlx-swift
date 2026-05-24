# TurboQuant Core Implementation Packet

This folder contains the `mlx-swift` side of the TurboQuant implementation train. `mlx-swift` is the primitive layer: it owns storage layouts, Metal kernels, capability probes, storage estimates, validators, attention decisions, benchmark primitives, hidden-copy audit, kernel warmup, and future Layout V5 optimization.

Cross-repo release-train docs live in:

```text
/Users/mt/Programming/Schtack/pines/docs/turboquant-implementation
```

This repo-local packet defines what core workers must implement and expose.

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
- Layout V5;
- fused head-dimension specializations.

## Required reading

1. [Core Contracts](core-contracts.md)
2. [Validation and Attention Router](validation-router.md)
3. [Benchmark JSON and Hidden-Copy Audit](benchmark-json-hidden-copy.md)
4. [Layout V5 and Kernels](layout-v5-kernels.md)
5. [Core Worker Cards](worker-cards.md)

## Non-negotiables

1. Bad compressed layout fails before Metal dispatch.
2. Every rejected path has a reason.
3. Capabilities are path-specific; broad pass/fail gates are not sufficient.
4. bfloat output is gated independently.
5. Linear matmul remains disabled unless separately certified.
6. Long-KV hot paths must not accidentally create full-cache row-contiguous copies.
7. Benchmark JSON must include storage estimates and path decisions.
8. Layout V5 must stay feature-gated and V4-compatible.
