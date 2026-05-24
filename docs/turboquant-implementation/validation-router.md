# Validation and Attention Router

W2 owns canonical validation and path selection. The goal is to fail invalid compressed state before Metal dispatch and to make path selection explainable.

## Worker

| Worker | Branch | Phase | Priority |
| --- | --- | --- | --- |
| W2 | `tq/core-validation-router` | MVP 1 | P1 |

## Owned files

Add:

- `Source/MLX/TurboQuantValidation.swift`
- `Source/MLX/TurboQuantAttentionRouter.swift`
- `Tests/MLXTests/TurboQuantValidationTests.swift`
- `Tests/MLXTests/TurboQuantAttentionRouterTests.swift`

Do not touch:

- kernel source strings;
- Layout V5 optimization;
- benchmark executable except to consume decisions after W3.

## Canonical validator

Required API:

```swift
public func validateTurboQuantAttentionCode(
    _ code: TurboQuantAttentionCode,
    expectedRole: TurboQuantTensorRole?,
    requireWritableCapacity: Bool = false
) throws
```

Validation checks:

- layout version;
- tensor role;
- dtype;
- rank;
- capacity axis;
- logical length;
- ring offset;
- pinned prefix;
- head dimension;
- groups per vector;
- packed words per group;
- bitset words per group;
- scales per group;
- key codec arrays;
- value codec arrays;
- writable capacity when required.

Failure messages must name actual and expected values.

## Attention request

Define a request type if one does not already exist:

```swift
public struct TurboQuantAttentionRequest: Hashable, Sendable {
    public var headDimension: Int
    public var queryLength: Int
    public var logicalLength: Int
    public var dtype: String
    public var maskKind: String
    public var preferredPaths: [TurboQuantAttentionPath]
    public var allowPackedFallback: Bool
    public var allowBaselineFallback: Bool
}
```

## Path selector

Required API:

```swift
public func selectTurboQuantAttentionPath(
    request: TurboQuantAttentionRequest,
    capabilities: TurboQuantKernelCapabilities
) -> TurboQuantAttentionDecision
```

Selection rules:

- Q=1 prefers online fused when supported;
- short multi-token query prefers tiled fused when supported;
- unsupported fused shape can choose two-stage;
- QK or AV unavailable rejects two-stage;
- packed fallback selected only when allowed;
- baseline fallback selected only when allowed;
- unsupported mask/dtype/head dimension records rejected paths.

## Rejection reasons

Required reason categories:

- `unsupportedHeadDimension`;
- `unsupportedQueryLength`;
- `unsupportedMaskKind`;
- `unsupportedDType`;
- `capabilityUnavailable`;
- `bfloatOutputUnavailable`;
- `packedFallbackNotAllowed`;
- `baselineFallbackNotAllowed`;
- `layoutInvalid`;
- `kernelProfileUnsupported`.

## Dispatch safety

Before compressed attention dispatch:

1. validate key code;
2. validate value code;
3. build attention request;
4. select path;
5. dispatch only if selected path is supported;
6. attach decision to diagnostics.

## Tests

Required:

- invalid layout fails before Metal;
- bad logical length names expected/actual;
- unsupported head dim rejects fused/two-stage;
- unsupported mask rejects compressed path;
- fused unavailable selects two-stage when QK/AV are true;
- QK unavailable selects packed fallback if allowed;
- no fallback allowed returns unavailable decision;
- every rejected path has nonempty reason.

## Acceptance gate

W2 is complete when unsupported shapes never dispatch a TurboQuant kernel and downstream callers can inspect selected/rejected path reasons.
