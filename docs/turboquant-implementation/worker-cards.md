# Core Worker Cards

This file preserves the `mlx-swift` worker scope from the cross-repo plan.

Use the Pines [Worker Launch Schedule](/Users/mt/Programming/Schtack/pines/docs/turboquant-implementation/14-worker-launch-schedule.md) for execution order. For `mlx-swift`, the executable order is:

| Wave | Worker | Can run when |
| --- | --- | --- |
| Wave 0 | W1 core public contracts | immediately |
| Wave 1 | W2 validation/router | W1 contracts compile |
| Wave 3 | W3 benchmark JSON | W1/W2 stable enough for reports |
| Wave 5 | W13 Layout V5/kernels | benchmark JSON, quality gates, memory calibration exist |

W1 and W2 are correctness/control-plane enablers. W13 is optimization and must remain gated until measurement proves improvement.

## Wave 0 - W1 core public contracts

Branch: `tq/core-contracts`

Owned files:

- `Source/MLX/TurboQuantContracts.swift`
- `Source/MLX/TurboQuantStorageEstimate.swift`
- `Tests/MLXTests/TurboQuantContractsTests.swift`

Tasks:

- add `TurboQuantKernelCapabilities`;
- add `TurboQuantStorageEstimate`;
- add `TurboQuantAttentionDecision`;
- add `RejectedTurboQuantPath`;
- expose `currentCapabilities()`;
- expose symbolic storage estimator;
- expose actual storage estimator;
- add Codable tests;
- add no-Metal defaults.

Acceptance:

- no Metal required;
- capabilities are path-specific;
- bfloat failure does not disable fp16/fp32;
- fused failure does not disable two-stage;
- storage estimate can be computed before allocation.

## Wave 1 - W2 validation and router

Branch: `tq/core-validation-router`

Owned files:

- `Source/MLX/TurboQuantValidation.swift`
- `Source/MLX/TurboQuantAttentionRouter.swift`
- `Tests/MLXTests/TurboQuantValidationTests.swift`
- `Tests/MLXTests/TurboQuantAttentionRouterTests.swift`

Tasks:

- validate `TurboQuantAttentionCode`;
- validate layout version;
- validate rank/dtype/shape;
- validate capacity/logical length;
- validate pinned prefix/ring offset;
- validate group and word axes;
- select attention path;
- emit rejected-path reasons;
- gate dtype/mask/head/query.

Acceptance:

- bad layout fails before Metal;
- every rejected path has a reason;
- unsupported shape never dispatches kernel.

## Wave 3 - W3 core benchmark JSON

Branch: `tq/core-benchmark-json`

Owned files:

- `Source/MLX/TurboQuantBenchmarkReport.swift`;
- benchmark executable JSON output;
- hidden-copy audit docs.

Tasks:

- add JSON schema;
- add `--json` flag;
- capture commit;
- capture capabilities;
- capture storage estimate;
- capture path decision;
- capture latency;
- capture actual bits/value;
- add hidden-copy audit.

Acceptance:

- benchmark emits stable JSON;
- Pines can import schema;
- hidden-copy risk is recorded.

## Wave 5 - W13 Layout V5 and kernels

Branch: `tq/layout-v5-kernels`

Tasks:

- add Layout V5 flag;
- add deterministic high mask;
- add popcount offsets;
- add fp16 scale option;
- keep V4 compatibility;
- specialize fused kernels: 64, 80, 96, 128, 192, 256;
- keep QK float;
- keep softmax float;
- gate bfloat separately;
- avoid hidden full-cache copy;
- benchmark before/after.

Acceptance:

- V5 improves speed or bytes;
- QualityGate remains green;
- unsupported dimensions fall back safely.

## Core backlog

| ID | Task |
| --- | --- |
| CORE-001 | Kernel capabilities |
| CORE-002 | Storage estimate |
| CORE-003 | Attention decision |
| CORE-004 | No-Metal defaults |
| CORE-005 | Contract tests |
| CORE-006 | Path router |
| CORE-007 | Rejected reasons |
| CORE-008 | dtype/mask/head gates |
| CORE-009 | Hidden-copy audit |
| CORE-010 | `ensure_row_contiguous` review |
| CORE-011 | Kernel warmup |
| CORE-012 | Cold/warm metrics |
| CORE-013 | Layout V5 |
| CORE-014 | Deterministic high mask |
| CORE-015 | Popcount offsets |
| CORE-016 | fp16 scales |
| CORE-017 | V4 compatibility |
| CORE-018 | Fused dim 64 |
| CORE-019 | Fused dim 80 |
| CORE-020 | Fused dim 96 |
| CORE-021 | Fused dim 128 |
| CORE-022 | Fused dim 192 |
| CORE-023 | Fused dim 256 |
| CORE-024 | bfloat gate |
| CORE-025 | Value decode tiling |
| CORE-026 | No full K/V hot path |
| CORE-027 | Benchmark JSON |
| CORE-028 | Quality A/B report |
| CORE-029 | TurboQuant linear remains gated |
| CORE-030 | Open-format prep |
