# MLX Forks Workspace Guide

This file is the root operating guide for agents working in
`/Users/mt/Programming/Schtack/mlx-forks`. It covers the multi-repo structure,
current TurboQuant goals, development rules, validation expectations, and where
to continue work.

If a nested repo has its own `AGENTS.md`, read it after this file. Nested
instructions override this root guide for that repo.

## Workspace Purpose

This workspace contains the RNT56 MLX fork stack used to develop, validate, and
ship TurboQuant and related MLX runtime changes for downstream apps, especially
Pines. The main engineering objective is long-context KV-cache compression that
is correct, observable, and promotable only when real-model gates pass.

Current high-level goals:

- Preserve exact prefill semantics and safe runtime behavior.
- Make compressed KV-cache paths fail closed when native kernels, quality gates,
  memory gates, or fallback policy do not pass.
- Advance affine K8/V4 and related affine K/V paths as the practical
  upstream-comparable baseline.
- Keep full PolarWHT K/V and hybrid K8+PolarWHT-V as experimental diagnostics
  until they pass native-path, quality, speed, memory, and same-machine upstream
  gates.
- Build evidence that separates kernel-only wins from real-model throughput and
  resident-memory wins.

## Repository Layout

The workspace root is not the main source repo. It contains several independent
git repositories plus artifacts and worktrees:

```text
/Users/mt/Programming/Schtack/mlx-forks
  mlx/                         C++/Metal MLX core fork
  mlx-c/                       C API bridge fork for MLX runtime APIs
  mlx-swift/                   Swift MLX bindings, Swift-side native APIs,
                               Metal packaging, and low-level TurboQuant calls
  mlx-swift-lm/                Model integration, KV cache policy/lifecycle,
                               profiles, benchmarks, quality gates, CLI tools
  mlx-swift-quantized-sdpa-worktree/
                               historical/parallel Swift worktree
  artifacts/                   benchmark outputs and local evidence
  .pr-worktrees/               generated PR worktrees
  .codex-worktrees/            generated Codex worktrees
  .upstream-pr-worktrees/      upstream comparison worktrees
```

Current observed local branches on 2026-06-07:

```text
mlx            codex/mlx-core-distributed-autodiff-backends
mlx-c          codex/mlx-c-quantized-sdpa-parity
mlx-swift      tq/layout-v5-default-device-tests
mlx-swift-lm   tq/lm-layout-v5-default-device-tests
```

Treat branch names as local state, not as a source of truth. Always inspect
`git status --short --branch` inside the repo you are about to edit.

## Dependency Shape

Conceptually, downstream model execution depends on the stack like this:

```text
Pines / downstream app
  -> mlx-swift-lm
      -> mlx-swift
          -> MLX native runtime / Metal kernels
          -> mlx-c when C API surface is involved
          -> mlx core code or submodule pins when native runtime behavior changes
```

Practical ownership:

- `mlx-swift-lm` owns model/cache integration, policies, exact prefill, quality
  gates, TurboQuant profiles, snapshots, benchmark CLIs, and promotion reports.
- `mlx-swift` owns Swift-visible MLX APIs, native capability probes, kernel
  wrappers, Metal library loading, and low-level TurboQuant entry points.
- `mlx-c` owns C ABI exposure when Swift or downstream code needs a native API
  that crosses the C boundary.
- `mlx` owns lower-level C++/Metal runtime behavior and core kernels.

When a feature crosses repos, update from the bottom up:

1. Implement or expose native behavior in `mlx` / `mlx-c` if required.
2. Wire the Swift API and capability probes in `mlx-swift`.
3. Route model/cache policy, diagnostics, and benchmarks in `mlx-swift-lm`.
4. Update downstream app pins only after the fork stack is validated.

## Current TurboQuant State

Start with the current handoff before changing code:

```text
mlx-swift-lm/docs/turboquant-implementation/continuation-handoff-2026-06-07.md
```

The important current facts are:

- The practical upstream-comparable route is affine K8/V4. Upstream's visible
  pinned K8+V4 row at `arozanov/turboquant-mlx` commit `6e928d7` is the mixed
  affine route, not the PolarWHT-V scaffold.
- Do not claim `0.98x` FP16. The visible upstream table shows about `0.72x`
  FP16 for K8+V4 on Qwen2.5-7B.
- Affine K8/V4 now respects `quantizedKVStart`; below-threshold runs remain raw
  and should report `1.00x` resident KV compression.
- Dynamic conversion now triggers at `offset >= quantizedKVStart`, then
  materializes converted state, synchronizes, and clears cache pressure.
- Promotion blocks compressed candidates when throughput or quality did not
  select the expected native compressed path.
- Full `polarWHTV3` and `hybridK8PolarWHTV3/V4` are diagnostic. Hybrid
  PolarWHT promotion is disabled by default unless
  `TURBOQUANT_ENABLE_POLARWHT_HYBRID_PROMOTION=1` is deliberately set for an
  experiment.

The first continuation task is to rerun the combined 16K affine K8/V4 quality
plus 32-token throughput gate documented in the handoff. The previous combined
attempt wrote a zero-byte JSONL, so no promotion claim can use it.

## Evidence And Artifact Rules

Evidence must be specific enough for another agent to reproduce it:

- repo and commit for every involved repo;
- model path and model family;
- context length, prompt length, generated tokens, warmup, repeats, cooldown;
- selected backend/path and native kernel kinds;
- quality result and quality selected path;
- resident KV bytes or compression ratio;
- peak and steady active memory;
- raw fallback, decoded fallback, unsupported shape, and sparse inactive state;
- artifact path.

Do not promote a TurboQuant path from:

- synthetic operator benchmarks alone;
- throughput without same-report quality;
- quality without selected native compressed path diagnostics;
- runs with raw fallback allocation or decoded fallback;
- sparse runs where sparse was requested but inactive;
- PolarWHT hybrid runs that only beat a diagnostic path but not affine K8/V4;
- old artifacts generated before the current diagnostics schema.

Keep generated artifacts under `artifacts/` or the repo-specific artifact folder
already used by the tool. Summaries should name exact artifact paths.

## Documentation Map

Core workspace/root docs:

- `TURBOQUANT_MLX_OPTIMIZATION_ANALYSIS_2026-06-04.md`
- `TURBOQUANT_KERNEL_OVERHAUL_PROGRESS.md`
- `TURBOQUANT_OVERHAUL_PLAN_2026-05-30.md`
- `TURBOQUANT_PERF_AUDIT_2026-05-29.md`

Current LM implementation docs:

- `mlx-swift-lm/docs/turboquant-implementation/README.md`
- `mlx-swift-lm/docs/turboquant-implementation/continuation-handoff-2026-06-07.md`
- `mlx-swift-lm/docs/turboquant-implementation/current-paths-and-benchmarks.md`
- `mlx-swift-lm/docs/turboquant-implementation/external-port-optimization-map.md`
- `mlx-swift-lm/docs/turboquant-implementation/cache-lifecycle.md`
- `mlx-swift-lm/docs/turboquant-implementation/model-profile-v2-quality.md`

Per-repo agent notes:

- `mlx-swift-lm/AGENTS.md`
- `mlx-swift/AGENTS.md`
- `mlx-swift-quantized-sdpa-worktree/AGENTS.md`

Downstream Pines docs live outside this workspace:

```text
/Users/mt/Programming/Schtack/pines/docs/turboquant-implementation
```

## Development Practices

Read first:

- Use `rg` and `rg --files` for code search.
- Inspect the local implementation before assuming upstream behavior.
- Read nearby tests and existing style before adding abstractions.
- For online/current facts about external repos, browse or inspect the local
  clone at the pinned commit. Do not rely on memory for moving upstream targets.

Editing:

- Keep changes scoped to the repo and feature being worked on.
- Use `apply_patch` for manual file edits.
- Do not rewrite unrelated files or reformat broad areas.
- Do not revert user or prior-agent changes unless explicitly asked.
- Do not use destructive git commands such as `git reset --hard` or
  `git checkout --` for cleanup.
- Prefer ASCII unless the file already uses non-ASCII for a clear reason.

Git:

- The workspace root is not a single git repo. Run git commands inside the
  target repo.
- Use `git status --short --branch` before edits and before final reports.
- Dirty worktrees are expected. Separate your changes from unrelated changes in
  the summary.
- Do not update dependency pins casually. Pin changes require cross-repo
  validation and downstream coordination.

Code quality:

- Add tests proportional to risk and blast radius.
- Keep production paths fail-closed and observable.
- Do not add silent fallbacks that return guessed tensors, zeros, or decoded
  fallbacks without diagnostics.
- Keep exact prefill exact unless a future certified profile explicitly allows
  approximate prefill.
- For GQA/MQA paths, avoid materializing repeated K/V tensors when a kernel can
  map query heads to KV heads directly.
- For quantized attention, avoid full-cache dequantization in hot paths unless
  the path is explicitly diagnostic.

## Validation Commands

Common `mlx-swift-lm` checks:

```bash
cd /Users/mt/Programming/Schtack/mlx-forks/mlx-swift-lm
swift build --target MLXLMCommon
swift build -c release --product TurboQuantInferenceParity
swift test --filter TurboQuantProfileTests
swift test --filter TurboQuantInferenceParitySparseVTests
swift test --filter KVCacheTests
python3 scripts/turboquant-polarwht-acceptance-gate.py --self-test
```

Common `mlx-swift` checks:

```bash
cd /Users/mt/Programming/Schtack/mlx-forks/mlx-swift
git submodule status --recursive
swift build --target MLX
swift build --target MLXNN
```

Focused native K8/V4 microbenchmark:

```bash
cd /Users/mt/Programming/Schtack/mlx-forks/mlx-swift-lm
swift build -c release --product TurboQuantNativeVxBenchmark
.build/release/TurboQuantNativeVxBenchmark \
  --contexts 16384,32768 \
  --value-bits 4 \
  --iterations 5 \
  --warmup 1 \
  --cooldown-ms 25 \
  --query-heads 16 \
  --kv-heads 8 \
  --head-dim 128 \
  --output artifacts/turboquant-hybrid-smoke-20260607/native-vx-qwen06b-k8v4.json
```

Current required real-model rerun is documented in:

```text
mlx-swift-lm/docs/turboquant-implementation/continuation-handoff-2026-06-07.md
```

## Benchmark Best Practices

- Use release builds for throughput.
- Run quality and throughput in the same diagnostics report for promotion
  evidence.
- Use repeated randomized throughput order for claims.
- Include FP16 when it fits; otherwise state the best FP16-fitting context and
  use dense affine K8/V4 as the compressed reference.
- Keep sparse diagnostics explicit. Sparse auto defaults should remain off until
  real-model speed and quality prove a win.
- Record cooldowns and memory source.
- Treat microbenchmarks as kernel regression tests, not product claims.

## TurboQuant Promotion Checklist

A path is not promotable unless all of these are true:

- real-model throughput completed for the target model/context;
- quality gate passed in the same report or an explicitly paired report from the
  same code/model/context;
- quality selected the expected native compressed path;
- throughput selected the expected native compressed path;
- no raw fallback or decoded fallback was allocated;
- resident KV compression is greater than `1.0x` and reported from the effective
  cache policy;
- peak and steady active memory are reported;
- sparse was not requested-but-inactive;
- fallback reasons and unsupported shapes are empty or explicitly non-blocking;
- same-machine upstream comparison exists for upstream-parity claims;
- downstream device evidence exists for product/device claims.

## Current Nonclaims

Do not state or imply these as achieved:

- `0.98x` FP16 TurboQuant speed.
- Upstream parity for PolarWHT.
- Product promotion of full PolarWHT K/V or hybrid K8+PolarWHT-V.
- Sparse-V promotion.
- Real memory savings from estimated KV bytes alone.
- iOS product readiness without physical-device evidence.

## How To Hand Off Work

Every substantial change should leave a short trail:

- changed files and why;
- tests/benchmarks run and exact command lines;
- artifact paths;
- blockers or nonclaims;
- next command another agent should run;
- any dirty unrelated files observed but not touched.

Update the relevant repo docs when the active state changes. For TurboQuant, the
primary handoff doc is:

```text
mlx-swift-lm/docs/turboquant-implementation/continuation-handoff-2026-06-07.md
```

