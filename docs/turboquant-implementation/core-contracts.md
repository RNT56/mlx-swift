# Core Contracts

W1 owns the public TurboQuant contracts consumed by `mlx-swift-lm` and Pines. These contracts must be additive, stable, Codable where exported, Sendable where crossing concurrency boundaries, and usable without Metal for simulator/no-GPU tests.

Launch wave: Wave 0. This can start immediately and should run in parallel with LM W4 and Pines W7/W24.

## Worker

| Worker | Branch | Phase | Priority |
| --- | --- | --- | --- |
| W1 | `tq/core-contracts` | MVP 0 | P0 |

## Owned files

Add:

- `Source/MLX/TurboQuantContracts.swift`
- `Source/MLX/TurboQuantStorageEstimate.swift`
- `Tests/MLXTests/TurboQuantContractsTests.swift`

Minimal edits allowed:

- `Source/MLX/TurboQuant.swift` only to expose existing internal data required by the contracts.

Do not touch:

- Metal kernel source strings;
- Layout V5 code;
- benchmark executable logic;
- unrelated MLXNN quantization code.

## TurboQuantKernelCapabilities

```swift
public struct TurboQuantKernelCapabilities: Hashable, Codable, Sendable {
    public var flatEncodeDecode: Bool
    public var attentionEncode: Bool
    public var attentionDecode: Bool
    public var qk: Bool
    public var av: Bool
    public var onlineFused: Bool
    public var tiledFused: Bool
    public var bfloatOutput: Bool
    public var linearMatmul: Bool
    public var supportedHeadDimensions: [Int]
    public var selectedKernelProfile: TurboQuantKernelProfile
    public var failureReasons: [String]
}
```

Rules:

- `linearMatmul` is false unless separately certified.
- `bfloatOutput` failure does not disable fp16/fp32 output.
- fused failure does not disable two-stage QK/AV.
- simulator/no-Metal default values must be explicit and safe.
- capability probes must not throw for ordinary unsupported hardware.

Required API:

```swift
public enum TurboQuantKernelAvailability {
    public static func currentCapabilities() -> TurboQuantKernelCapabilities
}
```

## TurboQuantStorageEstimate

```swift
public struct TurboQuantStorageEstimate: Hashable, Codable, Sendable {
    public var role: TurboQuantTensorRole
    public var logicalValues: Int
    public var packedBytes: Int
    public var bitsetBytes: Int
    public var scaleBytes: Int
    public var totalBytes: Int
    public var actualBitsPerValue: Double
}
```

Responsibilities:

- symbolic estimate before cache arrays exist;
- actual estimate from allocated code arrays;
- separate key and value estimates;
- include packed bytes, bitset bytes, scale bytes, and total bytes;
- emit actual bits/value.

Required APIs:

```swift
public func estimateTurboQuantStorage(
    role: TurboQuantTensorRole,
    logicalValues: Int,
    preset: TurboQuantPreset,
    valueBits: Int?,
    groupSize: Int,
    dtype: DType
) -> TurboQuantStorageEstimate

public func estimateTurboQuantStorage(
    code: TurboQuantAttentionCode
) -> TurboQuantStorageEstimate
```

Acceptance:

- symbolic estimate can run without allocating cache;
- actual estimate matches array `nbytes` within known metadata tolerance;
- Pines can use symbolic estimate during admission.

## TurboQuantAttentionDecision

```swift
public struct TurboQuantAttentionDecision: Hashable, Codable, Sendable {
    public var selectedPath: TurboQuantAttentionPath
    public var rejectedPaths: [RejectedTurboQuantPath]
    public var headDimension: Int
    public var queryLength: Int
    public var logicalLength: Int
    public var dtype: String
    public var maskKind: String
    public var kernelProfile: TurboQuantKernelProfile
    public var fallbackReason: String?
}

public struct RejectedTurboQuantPath: Hashable, Codable, Sendable {
    public var path: TurboQuantAttentionPath
    public var reason: String
}
```

Rules:

- every fallback has a reason;
- every rejected path has a reason;
- unsupported mask/head/dtype are explicit rejections;
- decision does not dispatch kernels by itself;
- decision can be serialized into benchmarks and support exports.

## No-Metal defaults

Simulator/no-Metal capabilities:

- encode/decode false unless CPU reference path is intentionally exposed;
- compressed attention false;
- supported dimensions empty;
- selected profile portable/noMetal;
- failure reasons explain no Metal device.

## Tests

Required:

- JSON/Codable roundtrip for every DTO;
- no-Metal defaults are safe;
- bfloat unavailable does not disable fp16/fp32;
- fused unavailable does not disable QK/AV;
- storage estimate is nonnegative and totals correctly;
- actual estimate matches known synthetic arrays;
- decision encodes rejected reasons.

## Acceptance gate

W1 is complete when:

- public DTOs exist;
- no Metal is required for tests;
- symbolic storage estimate works before allocation;
- capabilities are path-specific;
- downstream repos can import contracts without editing kernels.
