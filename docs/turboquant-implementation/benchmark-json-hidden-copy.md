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
    public var totalBytes: Int
    public var compressedKVBytes: Int
    public var peakMemoryBytes: Int?
    public var actualBitsPerValue: Double
    public var fallbackUsed: Bool
    public var fallbackReason: String?
    public var memoryWarningsSeen: Int
    public var jetsamObserved: Bool
}
```

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
```

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
- report includes actual bits/value;
- Pines can parse the report into BenchmarkReport.v1.

## Tests

Required:

- JSON encode/decode;
- schema version present;
- missing required fields fail import;
- storage estimate included;
- selected/rejected paths included;
- hidden-copy audit doc remains present and updated for new kernels.
