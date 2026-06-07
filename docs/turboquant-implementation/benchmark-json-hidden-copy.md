# Benchmark JSON and Hidden-Copy Audit

W3 owns stable core benchmark output and the hidden-copy audit. This work feeds Pines evidence import and prevents long-context cache duplication from hiding inside Metal launch preparation.

Launch wave: Wave 3. Start after W1/W2 are stable enough to report capabilities, storage estimates, and path decisions.

## Worker

| Worker | Branch | Phase | Priority |
| --- | --- | --- | --- |
| W3 | `tq/core-benchmark-json` | MVP 1.5 | P1 |

## Owned files

Add:

- `Source/MLX/TurboQuantBenchmarkReport.swift`
- benchmark executable JSON output changes;
- `docs/turboquant-implementation/benchmark-json-hidden-copy.md` updates;
- tests for JSON encoding.

## Core benchmark report

```swift
public struct TurboQuantCoreBenchmarkReport: Codable, Sendable {
    public var schemaVersion: Int
    public var mlxSwiftCommit: String?
    public var capabilities: TurboQuantKernelCapabilities
    public var storageEstimate: TurboQuantStorageEstimate
    public var pathDecision: TurboQuantAttentionDecision?
    public var pathMeasurements: [TurboQuantCoreBenchmarkPathMeasurement]
    public var metrics: TurboQuantCoreBenchmarkMetrics
    public var hiddenCopyAudit: TurboQuantHiddenCopyAudit
}

public struct TurboQuantCoreBenchmarkMetrics: Codable, Sendable {
    public var contextTokens: Int
    public var headDimension: Int
    public var queryLength: Int
    public var preset: String
    public var valueBits: Int?
    public var groupSize: Int
    public var encodeMS: Double?
    public var decodeMS: Double?
    public var qkMS: Double?
    public var avMS: Double?
    public var fusedMS: Double?
    public var firstTokenLatencyMS: Double?
    public var prefillTokensPerSecond: Double?
    public var decodeTokensPerSecondP50: Double?
    public var decodeTokensPerSecondP95: Double?
    public var rawSDPAReferenceDType: String?
    public var rawSDPAAttentionLatencyMSP50: Double?
    public var rawSDPAAttentionLatencyMSP95: Double?
    public var rawSDPADecodeTokensPerSecondP50: Double?
    public var rawSDPADecodeTokensPerSecondP95: Double?
    public var speedRatioToRawSDPAP50: Double?
    public var speedRatioToRawSDPAP95: Double?
    public var totalBytes: Int
    public var compressedKVBytes: Int
    public var rawSDPAKVBytes: Int?
    public var memoryBytesSavedVsRawSDPA: Int?
    public var memoryReductionPercent: Double?
    public var peakMemoryBytes: Int?
    public var actualBitsPerValue: Double
    public var fallbackUsed: Bool
    public var fallbackReason: String?
    public var memoryWarningsSeen: Int
    public var jetsamObserved: Bool
    public var cooldownMS: Int?
    public var pathCooldownMS: Int?
}
```

Each core JSON report now includes `pathMeasurements`, one row for every
`TurboQuantAttentionPath` enum case. Row `status` values are:

| Status | Meaning |
| --- | --- |
| `reference` | FP16 raw SDPA baseline row. |
| `measured` | Primitive path was timed for this shape/capability set. |
| `skipped` | Path is meaningful but was not requested or not applicable to this value-bit/shape cell. |
| `failed` | Primitive path was attempted and threw. |
| `notCallable` | Path label exists, but the current public benchmark target cannot force a distinct primitive dispatch for it. |
| `unavailable` | Capability, route, or sentinel is unavailable. |

## CLI requirements

Add flags where applicable:

```text
--json
--head-dim
--context
--query-length
--preset
--value-bits
--group-size
--path
--warmup
--cooldown-ms
--path-cooldown-ms
```

Current path values accepted by `TurboQuantBenchmark --path`:

| Value | Meaning |
| --- | --- |
| `auto` or omitted | Use router-selected path. |
| `native-mlx` / `native-mlx-compressed` | Native segmented MLX compressed attention API. |
| `affine-int4-native` / `native-affine-int4` | Affine int4 native label. Reported as not separately callable until a public primitive dispatcher exists. |
| `affine-k8v4-native` / `native-affine-k8v4` | Native mixed affine K8/V4 path. |
| `affine-k8vx-native` / `native-affine-k8vx` | Native mixed affine K8/Vx path for lower-value-bit experiments. |
| `affine-k8vx-residual` / `native-affine-k8vx-residual` | Residual native mixed affine K8/Vx label. |
| `online-fused` | Swift Metal online fused Polar/QJL compressed path. |
| `tiled-online-fused` | Swift Metal tiled online fused long-context path. |
| `sparse-value-two-stage` / `sparse-value-two-stage-compressed` | Explicit sparse-value two-stage label. Current evidence is surfaced through native Sparse-V rows when `--sparse-v` is set. |
| `two-stage` / `two-stage-compressed` | QK and AV split compressed path. |
| `mlx-packed-fallback` | MLX packed fallback comparison path. |
| `baseline` | Raw SDPA baseline. |
| `unavailable` | Sentinel route for blocked paths. |

`--cooldown-ms` sleeps between warmup/measured samples without adding the sleep
to measured duration. `--path-cooldown-ms` sleeps between path families after a
timing block. Both default to `0` and are recorded in `metrics`.

Sparse-V options are exposed by the native compressed/affine paths for
threshold, top-k, cumulative-mass, and hybrid cumulative-plus-top-k selection.
Reports must preserve the Sparse-V mode, skipped/considered tokens, retained
mass, dense-reference cosine/max error, fallback reason, and per-head/per-layer
diagnostics when present. Dense compressed AV remains the reference/fallback for
promotion evidence.

Qwen GQA4/HD256 decode cells should always pass:

```bash
--head-dim 256 --query-heads 16 --kv-heads 4 --query-length 1
```

The LM-side runner
`/Users/mt/Programming/Schtack/mlx-forks/mlx-swift-lm/scripts/run-turboquant-current-benchmarks.sh`
loops this executable across the current path/preset/context matrix and stores
the stdout/stderr for every cell.

## Hidden-copy audit

Long-context K/V paths must not accidentally duplicate full cache arrays through row-contiguous conversion. Audit every TurboQuant Metal invocation.

For each kernel, record:

- kernel name;
- large inputs;
- whether input may be non-contiguous;
- whether row-contiguous copy can occur;
- mitigation;
- benchmark status.

Audit table template:

| Kernel | Large input | Copy risk | Mitigation | Status |
| --- | --- | --- | --- | --- |
| encode flat | source K/V chunk | low | benchmark input is chunk-bounded; no long-cache array is prepared | audited-bounded |
| decode flat | compressed code | medium | canonical storage validation rejects non-row-contiguous code arrays before dispatch | guarded |
| compressed QK | compressed K cache | high | canonical compressed storage validation runs before dispatch; no decoded K cache is materialized | guarded |
| compressed AV | compressed V cache | high | canonical compressed storage validation runs before dispatch; attention weights must already be row-contiguous | guarded |
| online fused | compressed K/V cache | high | fused dispatch consumes canonical compressed K/V arrays directly and does not decode a full cache | guarded |
| tiled fused | compressed K/V cache | high | tiled path shares fused dispatch guards; non-canonical compressed storage is rejected before launch | guarded |

Current W3 audit status: `pass`.

This status means the current source paths and validation gates have been audited for hidden full-cache K/V copies. It is not a production verification claim. Query tensors may require bounded row-contiguous preparation; compressed K/V cache tensors must remain canonical or be rejected before Metal launch.

## Audit acceptance

- every TurboQuant kernel is listed;
- every long-KV input has an explicit copy-risk status;
- benchmark can fail if a known hidden full-cache copy path is enabled;
- query tensors may copy if small and bounded;
- compressed K/V cache tensors must be canonical or explicitly stride-indexed.

## Benchmark JSON acceptance

- stable JSON schema;
- report includes capabilities;
- report includes storage estimates;
- report includes path decision;
- report includes per-path coverage rows for every attention path enum case;
- report includes actual bits/value;
- report includes FP16 raw-SDPA reference timing, speed ratios, bytes saved, and
  cooldown settings when measurement is available;
- Pines can parse the report into BenchmarkReport.v1.

## Tests

Required:

- JSON encode/decode;
- schema version present;
- missing required fields fail import;
- storage estimate included;
- selected/rejected paths included;
- hidden-copy audit doc remains present and updated for new kernels.
