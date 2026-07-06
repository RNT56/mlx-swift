// Copyright © 2026 RNT56.

import Cmlx
import Foundation

#if canImport(Darwin)
    import Darwin
#endif

#if canImport(Metal)
    import Metal
#endif

/// TurboQuant preset requested by higher-level runtime code.
///
/// This additive Swift API gives callers one stable surface for the fast packed
/// MLX compatibility path, a deterministic TurboQuantProd/QJL reference codec,
/// and the TurboQuantProd key plus bitpacked-value Metal backend.
public enum TurboQuantPreset: String, Codable, Sendable, CaseIterable {
    case turbo2_5
    case turbo3_5
    case turbo4
    case turbo4v2
    case turbo8

    public var displayName: String {
        switch self {
        case .turbo2_5:
            "TurboQuant 2.5-bit"
        case .turbo3_5:
            "TurboQuant 3.5-bit"
        case .turbo4:
            "TurboQuant 4-bit"
        case .turbo4v2:
            "TurboQuant 4-bit V2"
        case .turbo8:
            "TurboQuant 8-bit"
        }
    }

    /// Current native MLX packed-lane width used by the compatibility path.
    ///
    /// MLX's public packed quantized matmul kernels accept integer lane widths.
    /// The mixed-bit Metal path uses ``baseMagnitudeBits`` and
    /// ``highMagnitudeBits`` directly; this value exists for MLX packed fallback
    /// interoperability.
    public var effectiveBits: Int {
        switch self {
        case .turbo2_5:
            2
        case .turbo3_5, .turbo4, .turbo4v2:
            4
        case .turbo8:
            8
        }
    }

    public var baseMagnitudeBits: Int {
        switch self {
        case .turbo2_5:
            2
        case .turbo3_5:
            3
        case .turbo4, .turbo4v2:
            4
        case .turbo8:
            8
        }
    }

    public var highMagnitudeBits: Int {
        switch self {
        case .turbo2_5:
            3
        case .turbo3_5:
            4
        case .turbo4, .turbo4v2:
            4
        case .turbo8:
            8
        }
    }

    public var targetMagnitudeBits: Float {
        switch self {
        case .turbo2_5:
            2.5
        case .turbo3_5:
            3.5
        case .turbo4, .turbo4v2:
            4
        case .turbo8:
            8
        }
    }

    public var defaultValueBits: Int {
        switch self {
        case .turbo2_5:
            2
        case .turbo3_5, .turbo4, .turbo4v2:
            4
        case .turbo8:
            8
        }
    }
}

/// Minimum context length at which the cooperative coalesced decode engages. Below this the
/// strided one-thread-per-token decode is kept: measurements show coop is neutral-to-slightly-
/// negative at short context (its extra passes/shuffles aren't amortized when little memory
/// traffic is moved) and a clear win at long context (≥32K), where it is +20–60% across schemes
/// on M2 Pro. So a single device-independent context gate keeps the win and avoids the regression.
let turboQuantCooperativeDecodeMinContext = 32_768

/// TQCOOP master switch. **Default OFF (opt-in)**; set `TQ_COOP=1` to enable the cooperative
/// quad-per-key (coalesced) QK decode at long context (see min-context) on geometry/codec-
/// compatible paths (uniform turbo8/turbo4v2 and split turbo3_5); everything else is byte-
/// identical. Read once.
///
/// Ships opt-in because on a shipped iOS app the environment is fixed at launch — an env var is
/// not a production runtime control, so default-on would ship an A-series-unvalidated GPU path
/// with no field kill switch. Enable for production via a proper runtime toggle after A-series
/// validation; `TQ_COOP=1` is the dev/benchmark switch in the meantime.
let turboQuantCooperativeDecodeEnabled: Bool =
    ProcessInfo.processInfo.environment["TQ_COOP"] == "1"

/// T2.2 H16 tgmem-diet master switch. **Default OFF (opt-in)**; set `TQ_H16=1` to select
/// the half-staged (partial/tile_scores in `half`, tile_has_weight as a bitset) GQA
/// block-partials kernel variant instead of the fp32-staged default. Composes with
/// `TQ_COOP`: kernel SELECTION (strided-H16 vs coop-H16) uses `coopActive && repeats==4`,
/// but the LANES_PER_TOKEN template value fed to the selected kernel is NOT simply
/// `coopActive` — the strided-H16 kernel's coop branch (`lanes_per_token == 4u`) still
/// hardcodes `r<4u` loops over `query_cache` rows that are only initialized up to
/// `repeat_count`, so LANES_PER_TOKEN must fall back to 1 whenever `useH16 && repeats<4`
/// even though `coopActive` (the widened repeats 2-4 guard) is true. See the dispatch-site
/// comments near the two `("LANES_PER_TOKEN", ...)` template entries. Cosine-gated, not
/// byte-identical to the fp32 kernel; read once.
let turboQuantH16DietEnabled: Bool =
    ProcessInfo.processInfo.environment["TQ_H16"] == "1"

public let turboQuantKeyPageSummaryPageSize = 512

/// Gate for the cooperative quad-per-key coalesced QK decode (TQCOOP). Active only when
/// env-enabled AND the geometry/codec match what the kernel's coop branch implements:
/// the GQA quad path (4 repeats), uniform magnitude bits (turbo8 — base == high, so the
/// split/variable-bit branches are unneeded), head_dim divisible by 4, and each lane's
/// head_dim/4 chunk lying within a single group. The coop variant uses a distinct kernel
/// name so its compiled MLX variant never shares a cache slot with the strided kernel.
func turboQuantCooperativeQuadDecodeActive(
    enabled: Bool,
    queryHeadRepeats: Int,
    preset: TurboQuantPreset,
    layoutVersion: Int,
    headDim: Int,
    groupSize: Int,
    logicalLength: Int
) -> Bool {
    guard turboQuantCooperativeDecodeEnabled, enabled,
          queryHeadRepeats >= 2 && queryHeadRepeats <= 4 else { return false }
    // Coop is excluded from layout v7: its offsets are hardcoded to the v6 token-major
    // packed/bitset/scale layout and would silently misread v7's tile-swizzled K planes.
    // Gated to exactly v6 (not v7, and no floor for legacy v4/v5 either -- see below).
    guard layoutVersion == TurboQuantAttentionLayout.splitMagnitudeVersion else { return false }
    // Coop only pays off once memory traffic dominates the decode; keep the strided path at
    // short context where it is neutral/negative (see turboQuantCooperativeDecodeMinContext).
    guard logicalLength >= turboQuantCooperativeDecodeMinContext else { return false }
    let base = Swift.max(1, preset.baseMagnitudeBits - 1)
    let high = Swift.max(base, preset.highMagnitudeBits - 1)
    // uniform = turbo8/turbo4v2 (base==high); split = turbo3_5 (high==base+1 at layout v6).
    // The kernel's coop branch implements exactly these two; the (dead-at-v6) variable-bit
    // branch-2 is excluded so coop never runs a decode shape it doesn't handle.
    let uniform = base == high
    let split = high == base + 1 && layoutVersion == TurboQuantAttentionLayout.splitMagnitudeVersion
    guard uniform || split else { return false }
    guard headDim == 128 || headDim == 256 else { return false }
    let chunk = headDim / 4
    return chunk >= 1 && chunk <= groupSize && groupSize % chunk == 0
}

public enum TurboQuantUserMode: String, Codable, Sendable {
    case fastest
    case balanced
    case maxContext
    case batterySaver
}

public enum TurboQuantFallbackPolicy: Sendable, Codable {
    case exactRequired
    case packedAllowed
    case compressedDecodeAllowed
    case fatalOnFailure
}

public enum TurboQuantCacheLifecycle: Sendable, Codable {
    case empty
    case rawPrefillChunkOpen
    case compressingChunk(start: Int, count: Int)
    case compressedCommitted(logicalLength: Int, capacity: Int)
    case decodeCompressed
    case degradedPackedFallback(reason: String)
    case degradedDecodedFallback(reason: String)
    case failed(reason: String)
}

public enum TurboQuantTensorRole: String, Codable, Sendable, CaseIterable {
    case key
    case value
    case vector
}

public enum TurboQuantBackend: String, Codable, Sendable, CaseIterable {
    /// MLX's native packed quantization and quantized matrix-multiply kernels.
    ///
    /// This is the production backend Pine uses today on iOS.
    case mlxPacked

    /// Deterministic CPU reference implementation for the TurboQuantProd key
    /// path, affine value path, and QJL residual sign estimator.
    case polarQJLReference

    /// Deterministic reference slot for the upstream PolarWHT codec contract.
    ///
    /// This is intentionally separate from ``polarQJLReference`` because the
    /// upstream-compatible value path stores Lloyd-Max centroid indices and
    /// vector norms so WHT can be pulled out of value attention. The current
    /// QJL path keeps affine value decode semantics.
    case polarWHTReference

    /// Mixed-bit key and bitpacked-value PolarQuant/QJL Metal kernels.
    case metalPolarQJL

    /// Upstream-compatible PolarWHT Metal kernels.
    ///
    /// Availability is fail-closed until the fused encode/decode, pre-rotated
    /// QK, and WHT-pulled AV kernels are all proven by the native probe.
    case metalPolarWHT
}

public enum TurboQuantScaleStorage: String, Codable, Sendable, Hashable, CaseIterable {
    case float32
    case float16

    public var dtype: DType {
        switch self {
        case .float32:
            .float32
        case .float16:
            .float16
        }
    }
}

public enum TurboQuantReferenceFormat: String, Codable, Sendable, Hashable, CaseIterable {
    case magnitudeResidualSign
    case turboQuantProd
    case affineValue
    /// N4 payload diet: per-group RMS norm + packed indices into a data-free Gaussian
    /// Lloyd-Max codebook (the optimal scalar quantizer for the post-rotation Gaussian
    /// distribution). Reaches the magnitude codec's quality at fewer payload bits/value.
    case gaussianLloydMax
}

public enum TurboQuantKernelProfile: String, Codable, Sendable, CaseIterable {
    case portableA16A17
    case wideA18A19
    case sustainedA19Pro
    case macAppleSilicon
    case mlxPackedFallback

    public var displayName: String {
        switch self {
        case .portableA16A17:
            "Portable A16/A17"
        case .wideA18A19:
            "Wide A18/A19"
        case .sustainedA19Pro:
            "Sustained A19 Pro"
        case .macAppleSilicon:
            "Mac Apple Silicon"
        case .mlxPackedFallback:
            "MLX packed fallback"
        }
    }

    var fusedDecodeThreadgroupWidth: Int {
        switch self {
        case .portableA16A17:
            128
        case .wideA18A19, .sustainedA19Pro, .macAppleSilicon:
            256
        case .mlxPackedFallback:
            128
        }
    }

    var blockParallelFusedTokenBlockSize: Int {
        switch self {
        case .macAppleSilicon:
            512
        case .portableA16A17, .wideA18A19, .sustainedA19Pro:
            256
        case .mlxPackedFallback:
            128
        }
    }

    public static func selected(
        architectureName: String,
        hardwareModelIdentifier: String? = nil,
        supportedGPUFamilies: [String: Bool],
        recommendedWorkingSetBytes: Int? = nil
    ) -> TurboQuantKernelProfile {
        selectTurboQuantKernelProfile(
            architectureName: architectureName,
            hardwareModelIdentifier: hardwareModelIdentifier,
            supportedGPUFamilies: supportedGPUFamilies,
            recommendedWorkingSetBytes: recommendedWorkingSetBytes
        )
    }
}

private func turboQuantOnlineFusedThreadgroupWidth(minimum: Int) -> Int {
    let target = max(1, min(256, minimum))
    var width = 1
    while width < target {
        width <<= 1
    }
    return width
}

private func turboQuantBlockParallelFusedThreadgroupWidth(minimum: Int) -> Int {
    let target = max(1, min(512, minimum))
    var width = 1
    while width < target {
        width <<= 1
    }
    return width
}

public func turboQuantRecommendedBlockParallelTokenBlockSize(
    logicalLength: Int,
    headDimension: Int,
    queryLength: Int,
    kernelProfile: TurboQuantKernelProfile
) -> Int? {
    guard queryLength == 1, logicalLength >= 4_096 else { return nil }
    let profileDefault = kernelProfile.blockParallelFusedTokenBlockSize
    var minimum = max(1, headDimension, profileDefault)

    if logicalLength > profileDefault * profileDefault {
        minimum = max(minimum, Int(ceil(sqrt(Double(logicalLength)))))
    }

    let blockWidth = turboQuantBlockParallelFusedThreadgroupWidth(minimum: minimum)
    let activeBlockCount = (logicalLength + blockWidth - 1) / blockWidth
    guard activeBlockCount > 1, activeBlockCount <= blockWidth else { return nil }
    return blockWidth
}

private func turboQuantResolvedBlockParallelTokenBlockSize(
    logicalLength: Int,
    headDimension: Int,
    queryLength: Int,
    kernelProfile: TurboQuantKernelProfile,
    requestedBlockParallelTokenBlockSize: Int?
) -> Int? {
    let environmentBlockSize =
        ProcessInfo.processInfo.environment["TURBOQUANT_HYBRID_POLARWHT_BLOCK_TOKENS"]
        .flatMap(Int.init)
    if let requestedBlockParallelTokenBlockSize = requestedBlockParallelTokenBlockSize
        ?? environmentBlockSize
    {
        return turboQuantBlockParallelFusedThreadgroupWidth(
            minimum: max(1, headDimension, requestedBlockParallelTokenBlockSize)
        )
    }
    return turboQuantRecommendedBlockParallelTokenBlockSize(
        logicalLength: logicalLength,
        headDimension: headDimension,
        queryLength: queryLength,
        kernelProfile: kernelProfile
    )
}

public enum TurboQuantRuntimeSelfTestStatus: String, Codable, Sendable, CaseIterable {
    case notRun
    case passed
    case failed
}

public struct TurboQuantRuntimeProbeResult: Equatable, Codable, Sendable {
    public static let throughputOptimizedOnlineFusedHeadDimensions = [64, 80, 96, 128, 192, 256]
    public static func defaultOnlineFusedHeadDimensions(
        for profile: TurboQuantKernelProfile
    ) -> [Int] {
        switch profile {
        case .portableA16A17:
            return [64, 80, 96, 128]
        case .wideA18A19, .sustainedA19Pro, .macAppleSilicon:
            return throughputOptimizedOnlineFusedHeadDimensions
        case .mlxPackedFallback:
            return []
        }
    }

    public var status: TurboQuantRuntimeSelfTestStatus
    public var metalRuntimeAvailable: Bool
    public var flatCodecPassed: Bool
    public var encodeDecodePassed: Bool
    public var qkPassed: Bool
    public var avPassed: Bool
    public var tiledFusedPassed: Bool
    public var bfloatOutputPassed: Bool
    public var polarWHTCodecPassed: Bool
    public var polarWHTAttentionPassed: Bool
    public var hybridK8PolarWHTValueAttentionPassed: Bool
    public var selectedKernelProfile: TurboQuantKernelProfile
    public var failureReason: String?
    public var polarWHTFailureReason: String?
    public var encodeDecodeLatencySeconds: Double?
    public var twoStageLatencySeconds: Double?
    public var tiledFusedLatencySeconds: Double?
    public var onlineFusedHeadDimensions: [Int]

    private enum CodingKeys: String, CodingKey {
        case status
        case metalRuntimeAvailable
        case flatCodecPassed
        case encodeDecodePassed
        case qkPassed
        case avPassed
        case tiledFusedPassed
        case bfloatOutputPassed
        case polarWHTCodecPassed
        case polarWHTAttentionPassed
        case hybridK8PolarWHTValueAttentionPassed
        case selectedKernelProfile
        case failureReason
        case polarWHTFailureReason
        case encodeDecodeLatencySeconds
        case twoStageLatencySeconds
        case tiledFusedLatencySeconds
        case onlineFusedHeadDimensions
    }

    public init(
        status: TurboQuantRuntimeSelfTestStatus = .notRun,
        metalRuntimeAvailable: Bool = false,
        flatCodecPassed: Bool = false,
        encodeDecodePassed: Bool = false,
        qkPassed: Bool = false,
        avPassed: Bool = false,
        tiledFusedPassed: Bool = false,
        bfloatOutputPassed: Bool = false,
        polarWHTCodecPassed: Bool = false,
        polarWHTAttentionPassed: Bool = false,
        hybridK8PolarWHTValueAttentionPassed: Bool = false,
        selectedKernelProfile: TurboQuantKernelProfile = .mlxPackedFallback,
        failureReason: String? = nil,
        polarWHTFailureReason: String? = nil,
        encodeDecodeLatencySeconds: Double? = nil,
        twoStageLatencySeconds: Double? = nil,
        tiledFusedLatencySeconds: Double? = nil,
        onlineFusedHeadDimensions: [Int] = TurboQuantRuntimeProbeResult
            .throughputOptimizedOnlineFusedHeadDimensions
    ) {
        self.status = status
        self.metalRuntimeAvailable = metalRuntimeAvailable
        self.flatCodecPassed = flatCodecPassed
        self.encodeDecodePassed = encodeDecodePassed
        self.qkPassed = qkPassed
        self.avPassed = avPassed
        self.tiledFusedPassed = tiledFusedPassed
        self.bfloatOutputPassed = bfloatOutputPassed
        self.polarWHTCodecPassed = polarWHTCodecPassed
        self.polarWHTAttentionPassed = polarWHTAttentionPassed
        self.hybridK8PolarWHTValueAttentionPassed = hybridK8PolarWHTValueAttentionPassed
        self.selectedKernelProfile = selectedKernelProfile
        self.failureReason = failureReason
        self.polarWHTFailureReason = polarWHTFailureReason
        self.encodeDecodeLatencySeconds = encodeDecodeLatencySeconds
        self.twoStageLatencySeconds = twoStageLatencySeconds
        self.tiledFusedLatencySeconds = tiledFusedLatencySeconds
        self.onlineFusedHeadDimensions = onlineFusedHeadDimensions
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        status = try container.decode(TurboQuantRuntimeSelfTestStatus.self, forKey: .status)
        metalRuntimeAvailable = try container.decode(Bool.self, forKey: .metalRuntimeAvailable)
        flatCodecPassed =
            try container.decodeIfPresent(Bool.self, forKey: .flatCodecPassed) ?? false
        encodeDecodePassed = try container.decode(Bool.self, forKey: .encodeDecodePassed)
        qkPassed = try container.decode(Bool.self, forKey: .qkPassed)
        avPassed = try container.decode(Bool.self, forKey: .avPassed)
        tiledFusedPassed = try container.decode(Bool.self, forKey: .tiledFusedPassed)
        bfloatOutputPassed =
            try container.decodeIfPresent(Bool.self, forKey: .bfloatOutputPassed) ?? false
        polarWHTCodecPassed =
            try container.decodeIfPresent(Bool.self, forKey: .polarWHTCodecPassed) ?? false
        polarWHTAttentionPassed =
            try container.decodeIfPresent(Bool.self, forKey: .polarWHTAttentionPassed) ?? false
        hybridK8PolarWHTValueAttentionPassed =
            try container.decodeIfPresent(
                Bool.self,
                forKey: .hybridK8PolarWHTValueAttentionPassed
            ) ?? false
        selectedKernelProfile =
            try container.decode(TurboQuantKernelProfile.self, forKey: .selectedKernelProfile)
        failureReason = try container.decodeIfPresent(String.self, forKey: .failureReason)
        polarWHTFailureReason =
            try container.decodeIfPresent(String.self, forKey: .polarWHTFailureReason)
        encodeDecodeLatencySeconds =
            try container.decodeIfPresent(Double.self, forKey: .encodeDecodeLatencySeconds)
        twoStageLatencySeconds =
            try container.decodeIfPresent(Double.self, forKey: .twoStageLatencySeconds)
        tiledFusedLatencySeconds =
            try container.decodeIfPresent(Double.self, forKey: .tiledFusedLatencySeconds)
        onlineFusedHeadDimensions =
            try container.decodeIfPresent([Int].self, forKey: .onlineFusedHeadDimensions)
            ?? Self.throughputOptimizedOnlineFusedHeadDimensions
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(status, forKey: .status)
        try container.encode(metalRuntimeAvailable, forKey: .metalRuntimeAvailable)
        try container.encode(flatCodecPassed, forKey: .flatCodecPassed)
        try container.encode(encodeDecodePassed, forKey: .encodeDecodePassed)
        try container.encode(qkPassed, forKey: .qkPassed)
        try container.encode(avPassed, forKey: .avPassed)
        try container.encode(tiledFusedPassed, forKey: .tiledFusedPassed)
        try container.encode(bfloatOutputPassed, forKey: .bfloatOutputPassed)
        try container.encode(polarWHTCodecPassed, forKey: .polarWHTCodecPassed)
        try container.encode(polarWHTAttentionPassed, forKey: .polarWHTAttentionPassed)
        try container.encode(
            hybridK8PolarWHTValueAttentionPassed,
            forKey: .hybridK8PolarWHTValueAttentionPassed
        )
        try container.encode(selectedKernelProfile, forKey: .selectedKernelProfile)
        try container.encodeIfPresent(failureReason, forKey: .failureReason)
        try container.encodeIfPresent(polarWHTFailureReason, forKey: .polarWHTFailureReason)
        try container.encodeIfPresent(
            encodeDecodeLatencySeconds, forKey: .encodeDecodeLatencySeconds)
        try container.encodeIfPresent(twoStageLatencySeconds, forKey: .twoStageLatencySeconds)
        try container.encodeIfPresent(
            tiledFusedLatencySeconds, forKey: .tiledFusedLatencySeconds)
        try container.encode(onlineFusedHeadDimensions, forKey: .onlineFusedHeadDimensions)
    }

    public var passed: Bool {
        status == .passed
            && metalRuntimeAvailable
            && flatCodecPassed
            && encodeDecodePassed
            && qkPassed
            && avPassed
    }

    public var kernelCapabilities: TurboQuantKernelCapabilities {
        let attentionCodecPassed = metalRuntimeAvailable && encodeDecodePassed
        let qkAvailable = attentionCodecPassed && qkPassed
        let avAvailable = attentionCodecPassed && avPassed
        return TurboQuantKernelCapabilities(
            nativeQuantizeAppendKV: metalRuntimeAvailable
                ? turboQuantNativeQuantizeAppendKVAvailable() : false,
            flatEncodeDecode: metalRuntimeAvailable && flatCodecPassed,
            linearMatmul: turboQuantExperimentalLinearMetalEnabled()
                && metalRuntimeAvailable && flatCodecPassed,
            attentionEncode: attentionCodecPassed,
            attentionDecode: attentionCodecPassed,
            attentionQK: qkAvailable,
            attentionAV: avAvailable,
            attentionFusedDecode: qkAvailable && avAvailable && tiledFusedPassed,
            attentionTiledFusedDecode: qkAvailable && avAvailable && tiledFusedPassed,
            polarWHTCodec: metalRuntimeAvailable && polarWHTCodecPassed,
            polarWHTAttention: metalRuntimeAvailable && polarWHTAttentionPassed,
            hybridK8PolarWHTValueAttention: metalRuntimeAvailable
                && hybridK8PolarWHTValueAttentionPassed,
            bfloatOutput: attentionCodecPassed && bfloatOutputPassed,
            supportedHeadDimensions: (qkAvailable && avAvailable && tiledFusedPassed) ? onlineFusedHeadDimensions : [],
            selectedKernelProfile: selectedKernelProfile,
            failureReasons: [failureReason, polarWHTFailureReason].compactMap { $0 }
        )
    }
}

public struct TurboQuantDeviceCapabilities: Equatable, Codable, Sendable {
    public var metalAvailable: Bool
    public var architectureName: String
    public var hardwareModelIdentifier: String?
    public var supportedGPUFamilies: [String: Bool]
    public var maxBufferBytes: Int
    public var recommendedWorkingSetBytes: Int?
    public var physicalMemoryBytes: Int?
    public var maxThreadgroupWidth: Int?
    public var runtimeProbe: TurboQuantRuntimeProbeResult

    public init(
        metalAvailable: Bool,
        architectureName: String,
        hardwareModelIdentifier: String? = nil,
        supportedGPUFamilies: [String: Bool] = [:],
        maxBufferBytes: Int = 0,
        recommendedWorkingSetBytes: Int? = nil,
        physicalMemoryBytes: Int? = nil,
        maxThreadgroupWidth: Int? = nil,
        runtimeProbe: TurboQuantRuntimeProbeResult = TurboQuantRuntimeProbeResult()
    ) {
        self.metalAvailable = metalAvailable
        self.architectureName = architectureName
        self.hardwareModelIdentifier = hardwareModelIdentifier
        self.supportedGPUFamilies = supportedGPUFamilies
        self.maxBufferBytes = maxBufferBytes
        self.recommendedWorkingSetBytes = recommendedWorkingSetBytes
        self.physicalMemoryBytes = physicalMemoryBytes
        self.maxThreadgroupWidth = maxThreadgroupWidth
        self.runtimeProbe = runtimeProbe
    }

    public var selectedKernelProfile: TurboQuantKernelProfile {
        runtimeProbe.selectedKernelProfile
    }

    public static var current: TurboQuantDeviceCapabilities {
        var capabilities = detectedTurboQuantDeviceCapabilities()
        capabilities.runtimeProbe = TurboQuantRuntimeProbe.shared.result()
        return capabilities
    }
}

// P1-1 fused quantize-append native probe. Runs the op once on a tiny tensor
// and checks the six outputs bit-exactly against the stock quantized() +
// slice-update reference. Fails closed (false) on any throw or mismatch, and
// caches the verdict for the process lifetime (mirrors the availability cache).
private final class TurboQuantQuantizeAppendKVProbeCache: @unchecked Sendable {
    static let shared = TurboQuantQuantizeAppendKVProbeCache()
    private let lock = NSLock()
    private var cachedResult: Bool?
    private init() {}
    func result(_ compute: () -> Bool) -> Bool {
        if turboQuantHostCachesDisabled { return compute() }
        lock.lock()
        if let cachedResult { lock.unlock(); return cachedResult }
        lock.unlock()
        let result = compute()
        lock.lock(); cachedResult = result; lock.unlock()
        return result
    }
    func resetForTesting() { lock.lock(); cachedResult = nil; lock.unlock() }
}

/// Fail-closed probe for the native fused quantize-append (P1-1) kernel.
///
/// Quantizes a tiny K (gs64/b8) and V (gs32/b4) row via ``MLXFast/quantizeAppendKV``
/// into zeroed capacity-4 planes at two offsets, and compares each of the six
/// planes bit-exactly (codes and scales/biases) against the stock
/// ``quantized(_:groupSize:bits:mode:globalScale:stream:)`` + slice-update
/// reference. Returns `false` on any error or mismatch.
public func turboQuantNativeQuantizeAppendKVAvailable() -> Bool {
    TurboQuantQuantizeAppendKVProbeCache.shared.result {
        turboQuantProbeQuantizeAppendKV()
    }
}

func turboQuantResetQuantizeAppendKVProbeForTesting() {
    TurboQuantQuantizeAppendKVProbeCache.shared.resetForTesting()
}

private func turboQuantProbeQuantizeAppendKV() -> Bool {
    guard metalRuntimeAvailable() else { return false }
    return turboQuantProbeQuantizeAppendKVImpl(
        headDim: 64,
        keyGroupSize: 64,
        keyBits: 8,
        valueGroupSize: 32,
        valueBits: 4,
        capacity: 4)
}

private func turboQuantProbeQuantizeAppendKVImpl(
    headDim: Int,
    keyGroupSize: Int,
    keyBits: Int,
    valueGroupSize: Int,
    valueBits: Int,
    capacity: Int
) -> Bool {
    let keyWords = headDim * keyBits / 32
    let keyGroups = headDim / keyGroupSize
    let valueWords = headDim * valueBits / 32
    let valueGroups = headDim / valueGroupSize

    func deterministicRow(_ phase: Double) -> [Float] {
        (0 ..< headDim).map { index in
            let position = Double(index)
            return Float(0.37 * sin(position * 0.053 + phase) + 0.11 * cos(position * 0.017))
        }
    }
    // [B=1, H=1, steps=1, head_dim]
    let kNew = MLXArray(deterministicRow(0.13), [1, 1, 1, headDim]).asType(.float16)
    let vNew = MLXArray(deterministicRow(0.71), [1, 1, 1, headDim]).asType(.float16)

    func zeroPlane(_ lastDim: Int, dtype: DType) -> MLXArray {
        MLXArray.zeros([1, 1, capacity, lastDim], dtype: dtype)
    }

    // Reference via stock quantized() written into zeroed planes.
    let (kCodesRef0, kScalesRef0, kBiasesRef0) = quantized(
        kNew, groupSize: keyGroupSize, bits: keyBits, mode: .affine)
    let (vCodesRef0, vScalesRef0, vBiasesRef0) = quantized(
        vNew, groupSize: valueGroupSize, bits: valueBits, mode: .affine)
    guard let kBiasesRef0, let vBiasesRef0 else { return false }

    let refKCodes = zeroPlane(keyWords, dtype: .uint32)
    let refKScales = zeroPlane(keyGroups, dtype: .float16)
    let refKBiases = zeroPlane(keyGroups, dtype: .float16)
    let refVCodes = zeroPlane(valueWords, dtype: .uint32)
    let refVScales = zeroPlane(valueGroups, dtype: .float16)
    let refVBiases = zeroPlane(valueGroups, dtype: .float16)

    var opKCodes = zeroPlane(keyWords, dtype: .uint32)
    var opKScales = zeroPlane(keyGroups, dtype: .float16)
    var opKBiases = zeroPlane(keyGroups, dtype: .float16)
    var opVCodes = zeroPlane(valueWords, dtype: .uint32)
    var opVScales = zeroPlane(valueGroups, dtype: .float16)
    var opVBiases = zeroPlane(valueGroups, dtype: .float16)

    // Exercise two offsets to catch destination-row arithmetic bugs.
    for offset in [0, 3] {
        let range = offset ..< (offset + 1)
        refKCodes[0..., 0..., range, 0...] = kCodesRef0
        refKScales[0..., 0..., range, 0...] = kScalesRef0
        refKBiases[0..., 0..., range, 0...] = kBiasesRef0
        refVCodes[0..., 0..., range, 0...] = vCodesRef0
        refVScales[0..., 0..., range, 0...] = vScalesRef0
        refVBiases[0..., 0..., range, 0...] = vBiasesRef0

        let out:
            (
                kCodes: MLXArray, kScales: MLXArray, kBiases: MLXArray,
                vCodes: MLXArray, vScales: MLXArray, vBiases: MLXArray
            )
        do {
            out = try MLXFast.quantizeAppendKV(
                keysNew: kNew,
                valuesNew: vNew,
                kCodes: opKCodes,
                kScales: opKScales,
                kBiases: opKBiases,
                vCodes: opVCodes,
                vScales: opVScales,
                vBiases: opVBiases,
                seqOffset: offset,
                steps: 1,
                keyGroupSize: keyGroupSize,
                keyBits: keyBits,
                valueGroupSize: valueGroupSize,
                valueBits: valueBits)
        } catch {
            // Fail closed: any native error means the capability is unavailable.
            return false
        }
        opKCodes = out.kCodes
        opKScales = out.kScales
        opKBiases = out.kBiases
        opVCodes = out.vCodes
        opVScales = out.vScales
        opVBiases = out.vBiases
    }

    eval(
        refKCodes, refKScales, refKBiases, refVCodes, refVScales, refVBiases,
        opKCodes, opKScales, opKBiases, opVCodes, opVScales, opVBiases)

    func bitwiseEqual(_ lhs: MLXArray, _ rhs: MLXArray) -> Bool {
        lhs.shape == rhs.shape && all(lhs .== rhs).item(Bool.self)
    }
    return bitwiseEqual(opKCodes, refKCodes)
        && bitwiseEqual(opKScales, refKScales)
        && bitwiseEqual(opKBiases, refKBiases)
        && bitwiseEqual(opVCodes, refVCodes)
        && bitwiseEqual(opVScales, refVScales)
        && bitwiseEqual(opVBiases, refVBiases)
}

// TQ_T11: process-lifetime cache for TurboQuantKernelAvailability.current. All inputs are
// cached or process-stable except turboQuantNativeMLXAttentionEnabled() (env read), so caching
// freezes the env at first access -- matching this codebase's start-of-process env-gate style
// (e.g. TQ_COOP). TURBOQUANT_DISABLE_HOST_CACHES=1 restores per-call rebuilds for diagnostics.
private final class TurboQuantKernelAvailabilityCache: @unchecked Sendable {
    static let shared = TurboQuantKernelAvailabilityCache()
    private let lock = NSLock()
    private var cachedResult: TurboQuantKernelAvailability?
    private init() {}
    func result(_ compute: () -> TurboQuantKernelAvailability) -> TurboQuantKernelAvailability {
        if turboQuantHostCachesDisabled { return compute() }
        lock.lock()
        if let cachedResult { lock.unlock(); return cachedResult }
        lock.unlock()
        let result = compute()
        lock.lock(); cachedResult = result; lock.unlock()
        return result
    }
    func resetForTesting() { lock.lock(); cachedResult = nil; lock.unlock() }
}

public struct TurboQuantKernelAvailability: Equatable, Codable, Sendable {
    public var supportsMLXPacked: Bool
    public var supportsPolarQJLReference: Bool
    public var supportsPolarWHTReference: Bool
    public var supportsMetalPolarQJLCodec: Bool
    public var supportsMetalPolarQJLAttention: Bool
    public var supportsMetalPolarQJL: Bool
    public var supportsMetalPolarWHTCodec: Bool
    public var supportsMetalPolarWHTAttention: Bool
    public var supportsMetalPolarWHT: Bool
    public var nativeCompressedAttention: Bool?
    public var nativeSparseVSupport: Bool?
    public var nativeDiagnosticsSupport: Bool?
    public var nativeBackendVersion: Int?
    public var nativeSegmentedAttentionBackend: TurboQuantNativeSegmentedAttentionBackend?
    public var nativePolarWHTSegmentedAttentionBackend: TurboQuantNativeSegmentedAttentionBackend?
    public var nativeFallbackReason: String?
    public var selectedKernelProfile: TurboQuantKernelProfile
    public var selfTestStatus: TurboQuantRuntimeSelfTestStatus
    public var selfTestFailureReason: String?
    public var onlineFusedHeadDimensions: [Int]

    public var kernelCapabilities: TurboQuantKernelCapabilities {
        let probeCapabilities = TurboQuantRuntimeProbe.shared.result().kernelCapabilities
        return TurboQuantKernelCapabilities(
            nativeCompressedAttention: nativeCompressedAttention,
            nativeSparseVSupport: nativeSparseVSupport,
            nativeDiagnosticsSupport: nativeDiagnosticsSupport,
            nativeBackendVersion: nativeBackendVersion,
            nativeSegmentedAttentionBackend: nativeSegmentedAttentionBackend,
            nativePolarWHTSegmentedAttentionBackend: nativePolarWHTSegmentedAttentionBackend,
            nativeFallbackReason: nativeFallbackReason,
            flatEncodeDecode: supportsMetalPolarQJLCodec && probeCapabilities.flatEncodeDecode,
            linearMatmul: supportsMetalPolarQJLCodec
                && probeCapabilities.linearMatmul
                && turboQuantExperimentalLinearMetalEnabled(),
            attentionEncode: supportsMetalPolarQJLAttention && probeCapabilities.attentionEncode,
            attentionDecode: supportsMetalPolarQJLAttention && probeCapabilities.attentionDecode,
            attentionQK: supportsMetalPolarQJLAttention && probeCapabilities.attentionQK,
            attentionAV: supportsMetalPolarQJLAttention && probeCapabilities.attentionAV,
            attentionFusedDecode: supportsMetalPolarQJLAttention
                && probeCapabilities.attentionFusedDecode,
            attentionTiledFusedDecode: supportsMetalPolarQJLAttention
                && probeCapabilities.attentionTiledFusedDecode,
            polarWHTCodec: supportsMetalPolarWHTCodec,
            polarWHTAttention: supportsMetalPolarWHTAttention,
            hybridK8PolarWHTValueAttention:
                supportsMetalPolarWHTAttention && supportsMetalPolarWHTCodec
                    && probeCapabilities.hybridK8PolarWHTValueAttention,
            bfloatOutput: supportsMetalPolarQJLAttention && probeCapabilities.bfloatOutput,
            supportedHeadDimensions: onlineFusedHeadDimensions,
            selectedKernelProfile: selectedKernelProfile,
            failureReasons: selfTestFailureReason.map { [$0] } ?? probeCapabilities.failureReasons
        )
    }

    public var attentionCapabilities: TurboQuantAttentionCapabilities {
        var capabilities = kernelCapabilities.attentionCapabilities
        capabilities.supportedOnlineFusedHeadDimensions = onlineFusedHeadDimensions
        capabilities.supportedDeviceFamilies =
            detectedTurboQuantDeviceCapabilities()
            .supportedGPUFamilies
            .filter { $0.value }
            .map(\.key)
            .sorted()
        return capabilities
    }

    public init(
        supportsMLXPacked: Bool = true,
        supportsPolarQJLReference: Bool = true,
        supportsPolarWHTReference: Bool = false,
        supportsMetalPolarQJLCodec: Bool = false,
        supportsMetalPolarQJLAttention: Bool = false,
        supportsMetalPolarQJL: Bool = false,
        supportsMetalPolarWHTCodec: Bool = false,
        supportsMetalPolarWHTAttention: Bool = false,
        supportsMetalPolarWHT: Bool = false,
        nativeCompressedAttention: Bool? = nil,
        nativeSparseVSupport: Bool? = nil,
        nativeDiagnosticsSupport: Bool? = nil,
        nativeBackendVersion: Int? = nil,
        nativeSegmentedAttentionBackend: TurboQuantNativeSegmentedAttentionBackend? = nil,
        nativePolarWHTSegmentedAttentionBackend: TurboQuantNativeSegmentedAttentionBackend? = nil,
        nativeFallbackReason: String? = nil,
        selectedKernelProfile: TurboQuantKernelProfile = .mlxPackedFallback,
        selfTestStatus: TurboQuantRuntimeSelfTestStatus = .notRun,
        selfTestFailureReason: String? = nil,
        onlineFusedHeadDimensions: [Int] = TurboQuantRuntimeProbeResult
            .throughputOptimizedOnlineFusedHeadDimensions
    ) {
        self.supportsMLXPacked = supportsMLXPacked
        self.supportsPolarQJLReference = supportsPolarQJLReference
        self.supportsPolarWHTReference = supportsPolarWHTReference
        self.supportsMetalPolarQJLCodec = supportsMetalPolarQJLCodec
        self.supportsMetalPolarQJLAttention = supportsMetalPolarQJLAttention
        self.supportsMetalPolarQJL = supportsMetalPolarQJL
        self.supportsMetalPolarWHTCodec = supportsMetalPolarWHTCodec
        self.supportsMetalPolarWHTAttention = supportsMetalPolarWHTAttention
        self.supportsMetalPolarWHT = supportsMetalPolarWHT
        self.nativeCompressedAttention = nativeCompressedAttention
        self.nativeSparseVSupport = nativeSparseVSupport
        self.nativeDiagnosticsSupport = nativeDiagnosticsSupport
        self.nativeBackendVersion = nativeBackendVersion
        self.nativeSegmentedAttentionBackend = nativeSegmentedAttentionBackend
        self.nativePolarWHTSegmentedAttentionBackend = nativePolarWHTSegmentedAttentionBackend
        self.nativeFallbackReason = nativeFallbackReason
        self.selectedKernelProfile = selectedKernelProfile
        self.selfTestStatus = selfTestStatus
        self.selfTestFailureReason = selfTestFailureReason
        self.onlineFusedHeadDimensions = onlineFusedHeadDimensions
    }

    public static var current: TurboQuantKernelAvailability {
        TurboQuantKernelAvailabilityCache.shared.result { rebuildCurrent() }
    }

    private static func rebuildCurrent() -> TurboQuantKernelAvailability {
        let tqProbeStart = TurboQuantHostProbe.enabled ? DispatchTime.now().uptimeNanoseconds : 0
        defer {
            if TurboQuantHostProbe.enabled {
                TurboQuantHostProbe.shared.recordAvailabilityRebuild(
                    nanos: DispatchTime.now().uptimeNanoseconds - tqProbeStart)
            }
        }
        let metalAvailable = metalRuntimeAvailable()
        let probe = TurboQuantRuntimeProbe.shared.result()
        let probeCapabilities = probe.kernelCapabilities
        let codecAvailable = metalAvailable && probeCapabilities.flatEncodeDecode
        let attentionAvailable =
            metalAvailable && probeCapabilities.attentionQK && probeCapabilities.attentionAV
        let nativeEnabled = turboQuantNativeMLXAttentionEnabled()
        let nativeProbe =
            nativeEnabled && metalAvailable && attentionAvailable
            ? TurboQuantNativeAttentionSelfTest.result
            : TurboQuantNativeAttentionSelfTestResult(
                nativeCompressedAttention: false,
                nativeSparseVSupport: false,
                nativeDiagnosticsSupport: false,
                nativeBackendVersion: nil,
                nativeSegmentedAttentionBackend: .unavailable,
                nativeFallbackReason: nativeEnabled
                    ? "native MLX compressed attention prerequisites have not passed"
                    : "native MLX compressed attention is disabled by rollout gate"
            )
        let polarWHTNativeBackend =
            nativeEnabled && metalAvailable && probeCapabilities.polarWHTAttention
            ? TurboQuantNativeSegmentedAttentionBackend.experimentalJIT
            : TurboQuantNativeSegmentedAttentionBackend.unavailable
        let polarWHTCodecAvailable = metalAvailable && probeCapabilities.polarWHTCodec
        let polarWHTAttentionAvailable =
            metalAvailable && probeCapabilities.polarWHTAttention
        return TurboQuantKernelAvailability(
            supportsPolarWHTReference: true,
            supportsMetalPolarQJLCodec: codecAvailable,
            supportsMetalPolarQJLAttention: attentionAvailable,
            supportsMetalPolarQJL: codecAvailable || attentionAvailable,
            supportsMetalPolarWHTCodec: polarWHTCodecAvailable,
            supportsMetalPolarWHTAttention: polarWHTAttentionAvailable,
            supportsMetalPolarWHT: polarWHTCodecAvailable && polarWHTAttentionAvailable,
            nativeCompressedAttention: nativeProbe.nativeCompressedAttention,
            nativeSparseVSupport: nativeProbe.nativeSparseVSupport,
            nativeDiagnosticsSupport: nativeProbe.nativeDiagnosticsSupport,
            nativeBackendVersion: nativeProbe.nativeBackendVersion,
            nativeSegmentedAttentionBackend: nativeProbe.nativeSegmentedAttentionBackend,
            nativePolarWHTSegmentedAttentionBackend: polarWHTNativeBackend,
            nativeFallbackReason: nativeProbe.nativeFallbackReason,
            selectedKernelProfile: probe.selectedKernelProfile,
            selfTestStatus: probe.status,
            selfTestFailureReason: probe.failureReason,
            onlineFusedHeadDimensions: probe.onlineFusedHeadDimensions
        )
    }

    public static func currentCapabilities() -> TurboQuantKernelCapabilities {
        current.kernelCapabilities
    }

    public func supports(_ backend: TurboQuantBackend) -> Bool {
        switch backend {
        case .mlxPacked:
            supportsMLXPacked
        case .polarQJLReference:
            supportsPolarQJLReference
        case .polarWHTReference:
            supportsPolarWHTReference
        case .metalPolarQJL:
            supportsMetalPolarQJL
        case .metalPolarWHT:
            supportsMetalPolarWHT
        }
    }

    public func runtimeBackend(for requestedBackend: TurboQuantBackend) -> TurboQuantBackend {
        if supports(requestedBackend) {
            requestedBackend
        } else {
            .mlxPacked
        }
    }

    public func fallbackReason(for requestedBackend: TurboQuantBackend) -> String? {
        guard !supports(requestedBackend) else { return nil }

        switch requestedBackend {
        case .mlxPacked:
            return nil
        case .polarQJLReference:
            return
                "PolarQuant/QJL reference backend unavailable; using MLX packed TurboQuant lanes."
        case .polarWHTReference:
            return
                "PolarWHT reference backend is not implemented yet; using MLX packed TurboQuant lanes."
        case .metalPolarQJL:
            if let selfTestFailureReason {
                return
                    "TurboQuant Metal self-test failed: \(selfTestFailureReason); using MLX packed TurboQuant lanes."
            }
            return
                "TurboQuant Metal kernels unavailable; using MLX packed TurboQuant lanes."
        case .metalPolarWHT:
            if let selfTestFailureReason {
                return
                    "PolarWHT Metal self-test failed: \(selfTestFailureReason); using MLX packed TurboQuant lanes."
            }
            return
                "PolarWHT Metal kernels unavailable; using MLX packed TurboQuant lanes."
        }
    }
}

public enum TurboQuantError: Error, Equatable, CustomStringConvertible {
    case invalidGroupSize(Int)
    case invalidMetalConfiguration(String)
    case invalidQualityInput(String)
    case invalidReferenceCode(String)
    case unsupportedBackend(TurboQuantBackend, String)

    public var description: String {
        switch self {
        case .invalidGroupSize(let groupSize):
            "TurboQuant group size must be positive, got \(groupSize)."
        case .invalidMetalConfiguration(let message):
            "Invalid TurboQuant Metal configuration: \(message)"
        case .invalidQualityInput(let message):
            "Invalid TurboQuant quality input: \(message)"
        case .invalidReferenceCode(let message):
            "Invalid TurboQuant reference code: \(message)"
        case .unsupportedBackend(let backend, let message):
            "Unsupported TurboQuant backend \(backend.rawValue): \(message)"
        }
    }
}

public struct TurboQuantConfiguration: Hashable, Codable, Sendable {
    public var preset: TurboQuantPreset
    public var role: TurboQuantTensorRole
    public var groupSize: Int
    public var mode: QuantizationMode
    public var backend: TurboQuantBackend
    public var seed: UInt64
    public var qjlResidualScale: Float
    public var valueBits: Int?
    public var attentionLayoutVersion: Int
    public var allowExperimentalLayoutV5: Bool
    public var allowExperimentalLayoutV7: Bool
    public var attentionScaleStorage: TurboQuantScaleStorage
    public var deterministicHighPrecisionMask: Bool

    private enum CodingKeys: String, CodingKey {
        case preset
        case role
        case groupSize
        case mode
        case backend
        case seed
        case qjlResidualScale
        case valueBits
        case attentionLayoutVersion
        case allowExperimentalLayoutV5
        case allowExperimentalLayoutV7
        case attentionScaleStorage
        case deterministicHighPrecisionMask
    }

    public init(
        preset: TurboQuantPreset = .turbo3_5,
        role: TurboQuantTensorRole = .vector,
        groupSize: Int = 64,
        mode: QuantizationMode = .affine,
        backend: TurboQuantBackend = .mlxPacked,
        seed: UInt64 = 0x9E37_79B9_7F4A_7C15,
        qjlResidualScale: Float = 0.5,
        valueBits: Int? = nil,
        attentionLayoutVersion: Int = TurboQuantAttentionLayout.productionDefaultVersion,
        allowExperimentalLayoutV5: Bool = false,
        allowExperimentalLayoutV7: Bool = false,
        attentionScaleStorage: TurboQuantScaleStorage = .float32,
        deterministicHighPrecisionMask: Bool = true
    ) {
        self.preset = preset
        self.role = role
        self.groupSize = groupSize
        self.mode = mode
        self.backend = backend
        self.seed = seed
        self.qjlResidualScale = qjlResidualScale
        self.valueBits = valueBits
        self.attentionLayoutVersion = attentionLayoutVersion
        self.allowExperimentalLayoutV5 = allowExperimentalLayoutV5
        self.allowExperimentalLayoutV7 = allowExperimentalLayoutV7
        self.attentionScaleStorage = attentionScaleStorage
        self.deterministicHighPrecisionMask = deterministicHighPrecisionMask
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        preset = try container.decodeIfPresent(TurboQuantPreset.self, forKey: .preset) ?? .turbo3_5
        role = try container.decodeIfPresent(TurboQuantTensorRole.self, forKey: .role) ?? .vector
        groupSize = try container.decodeIfPresent(Int.self, forKey: .groupSize) ?? 64
        mode = try container.decodeIfPresent(QuantizationMode.self, forKey: .mode) ?? .affine
        backend = try container.decodeIfPresent(TurboQuantBackend.self, forKey: .backend) ?? .mlxPacked
        seed = try container.decodeIfPresent(UInt64.self, forKey: .seed) ?? 0x9E37_79B9_7F4A_7C15
        qjlResidualScale = try container.decodeIfPresent(Float.self, forKey: .qjlResidualScale) ?? 0.5
        valueBits = try container.decodeIfPresent(Int.self, forKey: .valueBits)
        attentionLayoutVersion = try container.decodeIfPresent(
            Int.self,
            forKey: .attentionLayoutVersion
        ) ?? TurboQuantAttentionLayout.productionDefaultVersion
        allowExperimentalLayoutV5 = try container.decodeIfPresent(
            Bool.self,
            forKey: .allowExperimentalLayoutV5
        ) ?? false
        allowExperimentalLayoutV7 = try container.decodeIfPresent(
            Bool.self,
            forKey: .allowExperimentalLayoutV7
        ) ?? false
        attentionScaleStorage = try container.decodeIfPresent(
            TurboQuantScaleStorage.self,
            forKey: .attentionScaleStorage
        ) ?? .float32
        deterministicHighPrecisionMask = try container.decodeIfPresent(
            Bool.self,
            forKey: .deterministicHighPrecisionMask
        ) ?? true
    }

    public var effectiveBits: Int { preset.effectiveBits }

    public var resolvedValueBits: Int {
        valueBits ?? preset.defaultValueBits
    }

    public var runtimeBackend: TurboQuantBackend {
        TurboQuantKernelAvailability.current.runtimeBackend(for: backend)
    }

    public var runtimeFallbackReason: String? {
        TurboQuantKernelAvailability.current.fallbackReason(for: backend)
    }

    public static func deterministicSeed(
        modelID: String,
        revision: String,
        cacheLayoutVersion: Int
    ) -> UInt64 {
        var hash: UInt64 = 0xCBF2_9CE4_8422_2325
        for byte in "\(modelID)#\(revision)#\(cacheLayoutVersion)".utf8 {
            hash ^= UInt64(byte)
            hash &*= 0x0000_0100_0000_01B3
        }
        return hash == 0 ? 0x9E37_79B9_7F4A_7C15 : hash
    }
}

public typealias TurboQuantPackedTensor = (
    weight: MLXArray,
    scales: MLXArray,
    biases: MLXArray?
)

public struct TurboQuantReferenceCode: Hashable, Codable, Sendable {
    public var shape: [Int]
    public var preset: TurboQuantPreset
    public var role: TurboQuantTensorRole
    public var format: TurboQuantReferenceFormat
    public var groupSize: Int
    public var seed: UInt64
    public var residualScale: Float
    public var baseMagnitudeBits: Int
    public var highMagnitudeBits: Int
    public var valueCount: Int
    public var baseScales: [Float]
    public var highScales: [Float]
    public var residualScales: [Float]
    public var signs: Data
    public var highPrecisionMask: Data
    public var residualSigns: Data
    public var packedMagnitudes: Data

    private enum CodingKeys: String, CodingKey {
        case shape
        case preset
        case role
        case format
        case groupSize
        case seed
        case residualScale
        case baseMagnitudeBits
        case highMagnitudeBits
        case valueCount
        case baseScales
        case highScales
        case residualScales
        case signs
        case highPrecisionMask
        case residualSigns
        case packedMagnitudes
    }

    public init(
        shape: [Int],
        preset: TurboQuantPreset,
        role: TurboQuantTensorRole,
        format: TurboQuantReferenceFormat = .magnitudeResidualSign,
        groupSize: Int,
        seed: UInt64,
        residualScale: Float,
        baseMagnitudeBits: Int,
        highMagnitudeBits: Int,
        valueCount: Int,
        baseScales: [Float],
        highScales: [Float],
        residualScales: [Float]? = nil,
        signs: Data,
        highPrecisionMask: Data,
        residualSigns: Data,
        packedMagnitudes: Data
    ) {
        self.shape = shape
        self.preset = preset
        self.role = role
        self.format = format
        self.groupSize = groupSize
        self.seed = seed
        self.residualScale = residualScale
        self.baseMagnitudeBits = baseMagnitudeBits
        self.highMagnitudeBits = highMagnitudeBits
        self.valueCount = valueCount
        self.baseScales = baseScales
        self.highScales = highScales
        self.residualScales = residualScales ?? []
        self.signs = signs
        self.highPrecisionMask = highPrecisionMask
        self.residualSigns = residualSigns
        self.packedMagnitudes = packedMagnitudes
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        shape = try container.decode([Int].self, forKey: .shape)
        preset = try container.decode(TurboQuantPreset.self, forKey: .preset)
        role = try container.decode(TurboQuantTensorRole.self, forKey: .role)
        format =
            try container.decodeIfPresent(TurboQuantReferenceFormat.self, forKey: .format)
            ?? .magnitudeResidualSign
        groupSize = try container.decode(Int.self, forKey: .groupSize)
        seed = try container.decode(UInt64.self, forKey: .seed)
        residualScale = try container.decodeIfPresent(Float.self, forKey: .residualScale) ?? 0.5
        baseMagnitudeBits = try container.decode(Int.self, forKey: .baseMagnitudeBits)
        highMagnitudeBits = try container.decode(Int.self, forKey: .highMagnitudeBits)
        valueCount = try container.decode(Int.self, forKey: .valueCount)
        baseScales = try container.decode([Float].self, forKey: .baseScales)
        highScales = try container.decode([Float].self, forKey: .highScales)
        residualScales = try container.decodeIfPresent([Float].self, forKey: .residualScales) ?? []
        signs = try container.decode(Data.self, forKey: .signs)
        highPrecisionMask = try container.decode(Data.self, forKey: .highPrecisionMask)
        residualSigns = try container.decode(Data.self, forKey: .residualSigns)
        packedMagnitudes = try container.decode(Data.self, forKey: .packedMagnitudes)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(shape, forKey: .shape)
        try container.encode(preset, forKey: .preset)
        try container.encode(role, forKey: .role)
        try container.encode(format, forKey: .format)
        try container.encode(groupSize, forKey: .groupSize)
        try container.encode(seed, forKey: .seed)
        try container.encode(residualScale, forKey: .residualScale)
        try container.encode(baseMagnitudeBits, forKey: .baseMagnitudeBits)
        try container.encode(highMagnitudeBits, forKey: .highMagnitudeBits)
        try container.encode(valueCount, forKey: .valueCount)
        try container.encode(baseScales, forKey: .baseScales)
        try container.encode(highScales, forKey: .highScales)
        try container.encode(residualScales, forKey: .residualScales)
        try container.encode(signs, forKey: .signs)
        try container.encode(highPrecisionMask, forKey: .highPrecisionMask)
        try container.encode(residualSigns, forKey: .residualSigns)
        try container.encode(packedMagnitudes, forKey: .packedMagnitudes)
    }

    public var storageByteCount: Int {
        switch format {
        case .affineValue:
            packedMagnitudes.count
                + (baseScales.count + highScales.count) * MemoryLayout<Float>.stride
        case .turboQuantProd:
            packedMagnitudes.count
                + signs.count
                + (baseScales.count + highScales.count) * MemoryLayout<Float>.stride
        case .magnitudeResidualSign:
            packedMagnitudes.count
                + signs.count
                + highPrecisionMask.count
                + residualSigns.count
                + (baseScales.count + highScales.count + residualScales.count)
                * MemoryLayout<Float>.stride
        case .gaussianLloydMax:
            // packed centroid indices + one norm per group (baseScales)
            packedMagnitudes.count + baseScales.count * MemoryLayout<Float>.stride
        }
    }

    public var approximateBitsPerValue: Double {
        guard valueCount > 0 else { return 0 }
        return Double(storageByteCount * 8) / Double(valueCount)
    }
}

public struct TurboQuantPolarWHTReferenceCode: Hashable, Codable, Sendable {
    public var shape: [Int]
    public var bits: Int
    public var headDimension: Int
    public var seed: UInt64
    public var valueCount: Int
    public var vectorCount: Int
    public var packedWordsPerVector: Int
    public var centroids: [Float]
    public var boundaries: [Float]
    public var signs: [Float]
    public var norms: [Float]
    public var packedIndices: [UInt32]

    public init(
        shape: [Int],
        bits: Int,
        headDimension: Int,
        seed: UInt64,
        valueCount: Int,
        vectorCount: Int,
        packedWordsPerVector: Int,
        centroids: [Float],
        boundaries: [Float],
        signs: [Float],
        norms: [Float],
        packedIndices: [UInt32]
    ) {
        self.shape = shape
        self.bits = bits
        self.headDimension = headDimension
        self.seed = seed
        self.valueCount = valueCount
        self.vectorCount = vectorCount
        self.packedWordsPerVector = packedWordsPerVector
        self.centroids = centroids
        self.boundaries = boundaries
        self.signs = signs
        self.norms = norms
        self.packedIndices = packedIndices
    }

    public var storageByteCount: Int {
        residentPayloadByteCount + signs.count * MemoryLayout<Float>.stride
    }

    public var residentPayloadByteCount: Int {
        packedIndices.count * MemoryLayout<UInt32>.stride
            + norms.count * MemoryLayout<Float>.stride
    }

    public var approximateBitsPerValue: Double {
        guard valueCount > 0 else { return 0 }
        return Double(residentPayloadByteCount * 8) / Double(valueCount)
    }
}

public struct TurboQuantMetalCode {
    public var shape: [Int]
    public var preset: TurboQuantPreset
    public var role: TurboQuantTensorRole
    public var groupSize: Int
    public var seed: UInt64
    public var valueBits: Int
    public var valueCount: Int
    public var groupCount: Int
    public var magnitudeWordsPerGroup: Int
    public var bitsetWordsPerGroup: Int
    public var scalesPerGroup: Int
    public var packedMagnitudes: MLXArray
    public var signs: MLXArray
    public var highPrecisionMask: MLXArray
    public var residualSigns: MLXArray
    public var scales: MLXArray

    public var storageByteCount: Int {
        if role == .value {
            return packedMagnitudes.nbytes + scales.nbytes
        }
        return packedMagnitudes.nbytes
            + signs.nbytes
            + highPrecisionMask.nbytes
            + residualSigns.nbytes
            + scales.nbytes
    }

    public var approximateBitsPerValue: Double {
        guard valueCount > 0 else { return 0 }
        return Double(storageByteCount * 8) / Double(valueCount)
    }
}

public enum TurboQuantAttentionPath: String, Codable, Sendable, CaseIterable {
    case nativeMLXCompressed
    case onlineFused
    case tiledOnlineFused
    case sparseValueTwoStageCompressed
    case twoStageCompressed
    case polarWHTReferenceHybrid
    case metalPolarWHTHybrid
    case metalHybridK8PolarWHTValue
    case affineInt4Native
    case affineK8V4Native
    case affineK8VxNative
    case affineK8VxResidual
    case mlxPackedFallback
    case baseline
    case unavailable
}

public struct TurboQuantSparseValueDiagnostics: Equatable, Codable, Sendable {
    public var enabled: Bool
    public var threshold: Float?
    public var skipped: Int
    public var considered: Int
    public var retainedMass: Double?

    public init(
        enabled: Bool,
        threshold: Float? = nil,
        skipped: Int = 0,
        considered: Int = 0,
        retainedMass: Double? = nil
    ) {
        self.enabled = enabled
        self.threshold = threshold
        self.skipped = max(0, skipped)
        self.considered = max(0, considered)
        self.retainedMass = retainedMass.map { max(0, min(1, $0)) }
    }

    public var skipRatio: Double {
        guard considered > 0 else { return 0 }
        return Double(skipped) / Double(considered)
    }
}

public struct TurboQuantScaledDotProductAttentionResult {
    public var output: MLXArray
    public var sparseValueDiagnostics: TurboQuantSparseValueDiagnostics?

    public init(
        output: MLXArray,
        sparseValueDiagnostics: TurboQuantSparseValueDiagnostics? = nil
    ) {
        self.output = output
        self.sparseValueDiagnostics = sparseValueDiagnostics
    }
}

public enum TurboQuantSparseValueNativeSelectionMode: Int32, Codable, Sendable, CaseIterable {
    case off = 0
    case threshold = 1
    case topK = 2
    case cumulativeMass = 3
    case hybridCumulativeMassTopK = 4
    case blockThreshold = 5
    case pageTopK = 6
    case candidateSparse = 7
}

public enum TurboQuantNativeSegmentedAttentionCodec: Int32, Codable, Sendable, CaseIterable {
    case polarQJL = 0
    case polarWHT = 1
    case hybridK8PolarWHTValue = 2

    public var requestedBackend: TurboQuantBackend {
        switch self {
        case .polarQJL:
            .metalPolarQJL
        case .polarWHT, .hybridK8PolarWHTValue:
            .metalPolarWHT
        }
    }
}

public struct TurboQuantNativeAttentionOptions: Equatable, Sendable {
    public static let backendVersion = 3

    public var scale: Float
    public var causal: Bool
    public var splitKBlockCount: Int
    public var sparseVThreshold: Float
    public var sparseVSelectionMode: TurboQuantSparseValueNativeSelectionMode
    public var sparseVTopK: Int
    public var sparseVCumulativeMass: Float
    public var sparseVMaxTopK: Int
    public var sparseVRecentTokens: Int
    public var sparseVCandidatePages: Int
    public var diagnostics: Bool
    public var backendVersion: Int

    public init(
        scale: Float,
        causal: Bool = false,
        splitKBlockCount: Int = 0,
        sparseVThreshold: Float = 0,
        sparseVSelectionMode: TurboQuantSparseValueNativeSelectionMode = .threshold,
        sparseVTopK: Int = 0,
        sparseVCumulativeMass: Float = 0,
        sparseVMaxTopK: Int = 0,
        sparseVRecentTokens: Int = 0,
        sparseVCandidatePages: Int = 0,
        diagnostics: Bool = false,
        backendVersion: Int = Self.backendVersion
    ) {
        self.scale = scale
        self.causal = causal
        self.splitKBlockCount = max(0, splitKBlockCount)
        self.sparseVThreshold = max(0, sparseVThreshold)
        self.sparseVSelectionMode = sparseVSelectionMode
        self.sparseVTopK = max(0, sparseVTopK)
        self.sparseVCumulativeMass = min(1, max(0, sparseVCumulativeMass))
        self.sparseVMaxTopK = max(0, sparseVMaxTopK)
        self.sparseVRecentTokens = max(0, sparseVRecentTokens)
        self.sparseVCandidatePages = max(0, sparseVCandidatePages)
        self.diagnostics = diagnostics
        self.backendVersion = backendVersion
    }
}

public enum TurboQuantNativeSegmentedAttentionBackend: Int32, Codable, Sendable, CaseIterable {
    case unavailable = 0
    case experimentalJIT = 1
    case nativeFused = 2
}

public struct TurboQuantNativeAttentionDiagnostics: Equatable, Sendable {
    public var backendVersion: Int
    public var kernelKind: Int
    public var activeBlocks: Int
    public var blockTokens: Int
    public var sparseSkippedTokens: Int
    public var sparseTotalTokens: Int
    public var fallbackCode: Int
    public var flags: Int
    public var recentTokens: Int
    public var selectedOlderTokens: Int
    public var selectedPages: Int
    public var candidatePagesConsidered: Int
    public var candidateTokensConsidered: Int
    public var retainedTokens: Int

    public init(values: [Int32]) {
        let padded = values + Array(repeating: 0, count: max(0, 16 - values.count))
        backendVersion = Int(padded[0])
        kernelKind = Int(padded[1])
        activeBlocks = Int(padded[2])
        blockTokens = Int(padded[3])
        sparseSkippedTokens = Int(padded[4])
        sparseTotalTokens = Int(padded[5])
        fallbackCode = Int(padded[6])
        flags = Int(padded[7])
        recentTokens = Int(padded[8])
        selectedOlderTokens = Int(padded[9])
        selectedPages = Int(padded[10])
        candidatePagesConsidered = Int(padded[11])
        candidateTokensConsidered = Int(padded[12])
        retainedTokens = Int(padded[13])
    }

    public var sparseSkipRatio: Double {
        guard sparseTotalTokens > 0 else { return 0 }
        return Double(sparseSkippedTokens) / Double(sparseTotalTokens)
    }
}

public struct TurboQuantNativeScaledDotProductAttentionResult {
    public var output: MLXArray
    public var diagnostics: TurboQuantNativeAttentionDiagnostics?

    public init(output: MLXArray, diagnostics: TurboQuantNativeAttentionDiagnostics? = nil) {
        self.output = output
        self.diagnostics = diagnostics
    }
}

public typealias TurboQuantNativeSegmentedAttentionResult =
    TurboQuantNativeScaledDotProductAttentionResult

public struct RejectedPath: Hashable, Codable, Sendable {
    public var path: TurboQuantAttentionPath
    public var reason: String

    public init(path: TurboQuantAttentionPath, reason: String) {
        self.path = path
        self.reason = reason
    }
}

public struct TurboQuantFallbackResult: Equatable, Codable, Sendable {
    public var requestedPath: TurboQuantAttentionPath
    public var selectedPath: TurboQuantAttentionPath
    public var reason: String

    public init(
        requestedPath: TurboQuantAttentionPath,
        selectedPath: TurboQuantAttentionPath,
        reason: String
    ) {
        self.requestedPath = requestedPath
        self.selectedPath = selectedPath
        self.reason = reason
    }
}

public struct TurboQuantLayerCacheFootprint: Equatable, Codable, Sendable {
    public var layerIndex: Int
    public var keyBytes: Int
    public var valueBytes: Int
    public var rawShadowBytes: Int
    public var packedFallbackBytes: Int
    public var decodedTransientBytes: Int

    public init(
        layerIndex: Int,
        keyBytes: Int,
        valueBytes: Int,
        rawShadowBytes: Int = 0,
        packedFallbackBytes: Int = 0,
        decodedTransientBytes: Int = 0
    ) {
        self.layerIndex = layerIndex
        self.keyBytes = keyBytes
        self.valueBytes = valueBytes
        self.rawShadowBytes = rawShadowBytes
        self.packedFallbackBytes = packedFallbackBytes
        self.decodedTransientBytes = decodedTransientBytes
    }

    public var totalBytes: Int {
        keyBytes + valueBytes + rawShadowBytes + packedFallbackBytes + decodedTransientBytes
    }
}

public struct TurboQuantRuntimeMemoryZones: Equatable, Codable, Sendable {
    public var modelResidentBytes: Int
    public var compressedKVBytes: Int
    public var fallbackReserveBytes: Int
    public var metalScratchBytes: Int
    public var promptAndTokenizerBytes: Int
    public var uiReserveBytes: Int
    public var safetyReserveBytes: Int

    public init(
        modelResidentBytes: Int = 0,
        compressedKVBytes: Int = 0,
        fallbackReserveBytes: Int = 0,
        metalScratchBytes: Int = 0,
        promptAndTokenizerBytes: Int = 0,
        uiReserveBytes: Int = 0,
        safetyReserveBytes: Int = 0
    ) {
        self.modelResidentBytes = modelResidentBytes
        self.compressedKVBytes = compressedKVBytes
        self.fallbackReserveBytes = fallbackReserveBytes
        self.metalScratchBytes = metalScratchBytes
        self.promptAndTokenizerBytes = promptAndTokenizerBytes
        self.uiReserveBytes = uiReserveBytes
        self.safetyReserveBytes = safetyReserveBytes
    }

    public var totalRuntimeBytes: Int {
        modelResidentBytes + compressedKVBytes + fallbackReserveBytes + metalScratchBytes
            + promptAndTokenizerBytes + uiReserveBytes + safetyReserveBytes
    }
}

public struct TurboQuantMemoryPlan: Equatable, Codable, Sendable {
    public var requestedContextLength: Int
    public var admittedContextLength: Int
    public var runtimeBudgetBytes: Int
    public var zones: TurboQuantRuntimeMemoryZones
    public var downgradeReason: String?

    public init(
        requestedContextLength: Int,
        admittedContextLength: Int,
        runtimeBudgetBytes: Int,
        zones: TurboQuantRuntimeMemoryZones,
        downgradeReason: String? = nil
    ) {
        self.requestedContextLength = requestedContextLength
        self.admittedContextLength = admittedContextLength
        self.runtimeBudgetBytes = runtimeBudgetBytes
        self.zones = zones
        self.downgradeReason = downgradeReason
    }
}

public struct TurboQuantAdmission: Equatable, Codable, Sendable {
    public var admitted: Bool
    public var mode: TurboQuantUserMode
    public var memoryPlan: TurboQuantMemoryPlan
    public var userMessage: String
    public var machineReason: String?

    public init(
        admitted: Bool,
        mode: TurboQuantUserMode,
        memoryPlan: TurboQuantMemoryPlan,
        userMessage: String,
        machineReason: String? = nil
    ) {
        self.admitted = admitted
        self.mode = mode
        self.memoryPlan = memoryPlan
        self.userMessage = userMessage
        self.machineReason = machineReason
    }
}

public struct TurboQuantDiagnosticEvent: Equatable, Codable, Sendable {
    public var name: String
    public var message: String
    public var fields: [String: String]

    public init(name: String, message: String, fields: [String: String] = [:]) {
        self.name = name
        self.message = message
        self.fields = fields
    }
}

public enum TurboQuantAttentionMaskKind: String, Codable, Sendable {
    case none
    case causal
    case materializedArray
    case unsupportedMaterializedArrays
}

public enum TurboQuantDTypeKind: String, Codable, Sendable, CaseIterable {
    case float16
    case bfloat16
    case float32

    public init?(_ dtype: DType) {
        switch dtype {
        case .float16:
            self = .float16
        case .bfloat16:
            self = .bfloat16
        case .float32:
            self = .float32
        default:
            return nil
        }
    }
}

public struct TurboQuantAttentionFallbackState: Equatable, Codable, Sendable {
    public var packedFallbackAvailable: Bool
    public var decodedFallbackAvailable: Bool
    public var baselineAvailable: Bool

    public init(
        packedFallbackAvailable: Bool = false,
        decodedFallbackAvailable: Bool = false,
        baselineAvailable: Bool = false
    ) {
        self.packedFallbackAvailable = packedFallbackAvailable
        self.decodedFallbackAvailable = decodedFallbackAvailable
        self.baselineAvailable = baselineAvailable
    }

    public static let none = TurboQuantAttentionFallbackState()
}

public struct TurboQuantAttentionCapabilities: Equatable, Codable, Sendable {
    public var nativeCompressedAttention: Bool?
    public var nativeSparseVSupport: Bool?
    public var nativeDiagnosticsSupport: Bool?
    public var nativeBackendVersion: Int?
    public var nativeSegmentedAttentionBackend: TurboQuantNativeSegmentedAttentionBackend?
    public var nativePolarWHTSegmentedAttentionBackend: TurboQuantNativeSegmentedAttentionBackend?
    public var nativeFallbackReason: String?
    public var encode: Bool
    public var decode: Bool
    public var qk: Bool
    public var av: Bool
    public var onlineFused: Bool
    public var tiledOnlineFused: Bool
    public var polarWHTCodec: Bool
    public var polarWHTAttention: Bool
    public var hybridK8PolarWHTValueAttention: Bool
    public var bfloatOutput: Bool
    public var supportedOnlineFusedHeadDimensions: [Int]
    public var maxOnlineFusedQueryLength: Int
    public var maxTiledOnlineFusedQueryLength: Int
    public var materializedMaskTwoStage: Bool
    public var supportedDTypes: [TurboQuantDTypeKind]
    public var supportedMasks: [TurboQuantAttentionMaskKind]
    public var supportedDeviceFamilies: [String]

    public init(
        nativeCompressedAttention: Bool? = nil,
        nativeSparseVSupport: Bool? = nil,
        nativeDiagnosticsSupport: Bool? = nil,
        nativeBackendVersion: Int? = nil,
        nativeSegmentedAttentionBackend: TurboQuantNativeSegmentedAttentionBackend? = nil,
        nativePolarWHTSegmentedAttentionBackend: TurboQuantNativeSegmentedAttentionBackend? = nil,
        nativeFallbackReason: String? = nil,
        encode: Bool = false,
        decode: Bool = false,
        qk: Bool = false,
        av: Bool = false,
        onlineFused: Bool = false,
        tiledOnlineFused: Bool? = nil,
        polarWHTCodec: Bool = false,
        polarWHTAttention: Bool = false,
        hybridK8PolarWHTValueAttention: Bool = false,
        bfloatOutput: Bool = false,
        supportedOnlineFusedHeadDimensions: [Int] =
            TurboQuantRuntimeProbeResult.throughputOptimizedOnlineFusedHeadDimensions,
        maxOnlineFusedQueryLength: Int = 1,
        maxTiledOnlineFusedQueryLength: Int = 8,
        materializedMaskTwoStage: Bool = true,
        supportedDTypes: [TurboQuantDTypeKind] = [.float16, .bfloat16, .float32],
        supportedMasks: [TurboQuantAttentionMaskKind] = [.none, .causal, .materializedArray],
        supportedDeviceFamilies: [String] = []
    ) {
        self.nativeCompressedAttention = nativeCompressedAttention
        self.nativeSparseVSupport = nativeSparseVSupport
        self.nativeDiagnosticsSupport = nativeDiagnosticsSupport
        self.nativeBackendVersion = nativeBackendVersion
        self.nativeSegmentedAttentionBackend = nativeSegmentedAttentionBackend
        self.nativePolarWHTSegmentedAttentionBackend = nativePolarWHTSegmentedAttentionBackend
        self.nativeFallbackReason = nativeFallbackReason
        self.encode = encode
        self.decode = decode
        self.qk = qk
        self.av = av
        self.onlineFused = onlineFused
        self.tiledOnlineFused = tiledOnlineFused ?? onlineFused
        self.polarWHTCodec = polarWHTCodec
        self.polarWHTAttention = polarWHTAttention
        self.hybridK8PolarWHTValueAttention = hybridK8PolarWHTValueAttention
        self.bfloatOutput = bfloatOutput
        self.supportedOnlineFusedHeadDimensions = supportedOnlineFusedHeadDimensions
        self.maxOnlineFusedQueryLength = maxOnlineFusedQueryLength
        self.maxTiledOnlineFusedQueryLength = maxTiledOnlineFusedQueryLength
        self.materializedMaskTwoStage = materializedMaskTwoStage
        self.supportedDTypes = supportedDTypes
        self.supportedMasks = supportedMasks
        self.supportedDeviceFamilies = supportedDeviceFamilies
    }

    public var twoStageCompressed: Bool {
        qk && av
    }

    private enum CodingKeys: String, CodingKey {
        case nativeCompressedAttention
        case nativeSparseVSupport
        case nativeDiagnosticsSupport
        case nativeBackendVersion
        case nativeSegmentedAttentionBackend
        case nativePolarWHTSegmentedAttentionBackend
        case nativeFallbackReason
        case encode
        case decode
        case qk
        case av
        case onlineFused
        case tiledOnlineFused
        case polarWHTCodec
        case polarWHTAttention
        case hybridK8PolarWHTValueAttention
        case bfloatOutput
        case supportedOnlineFusedHeadDimensions
        case maxOnlineFusedQueryLength
        case maxTiledOnlineFusedQueryLength
        case materializedMaskTwoStage
        case supportedDTypes
        case supportedMasks
        case supportedDeviceFamilies
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let onlineFused =
            try container.decodeIfPresent(Bool.self, forKey: .onlineFused) ?? false
        self.init(
            nativeCompressedAttention: try container.decodeIfPresent(
                Bool.self, forKey: .nativeCompressedAttention),
            nativeSparseVSupport: try container.decodeIfPresent(
                Bool.self, forKey: .nativeSparseVSupport),
            nativeDiagnosticsSupport: try container.decodeIfPresent(
                Bool.self, forKey: .nativeDiagnosticsSupport),
            nativeBackendVersion: try container.decodeIfPresent(
                Int.self, forKey: .nativeBackendVersion),
            nativeSegmentedAttentionBackend: try container.decodeIfPresent(
                TurboQuantNativeSegmentedAttentionBackend.self,
                forKey: .nativeSegmentedAttentionBackend
            ),
            nativePolarWHTSegmentedAttentionBackend: try container.decodeIfPresent(
                TurboQuantNativeSegmentedAttentionBackend.self,
                forKey: .nativePolarWHTSegmentedAttentionBackend
            ),
            nativeFallbackReason: try container.decodeIfPresent(
                String.self, forKey: .nativeFallbackReason),
            encode: try container.decodeIfPresent(Bool.self, forKey: .encode) ?? false,
            decode: try container.decodeIfPresent(Bool.self, forKey: .decode) ?? false,
            qk: try container.decodeIfPresent(Bool.self, forKey: .qk) ?? false,
            av: try container.decodeIfPresent(Bool.self, forKey: .av) ?? false,
            onlineFused: onlineFused,
            tiledOnlineFused: try container.decodeIfPresent(
                Bool.self,
                forKey: .tiledOnlineFused
            ) ?? onlineFused,
            polarWHTCodec: try container.decodeIfPresent(Bool.self, forKey: .polarWHTCodec) ?? false,
            polarWHTAttention: try container.decodeIfPresent(Bool.self, forKey: .polarWHTAttention) ?? false,
            hybridK8PolarWHTValueAttention: try container.decodeIfPresent(
                Bool.self,
                forKey: .hybridK8PolarWHTValueAttention
            ) ?? false,
            bfloatOutput: try container.decodeIfPresent(Bool.self, forKey: .bfloatOutput) ?? false,
            supportedOnlineFusedHeadDimensions: try container.decodeIfPresent(
                [Int].self,
                forKey: .supportedOnlineFusedHeadDimensions
            ) ?? TurboQuantRuntimeProbeResult.throughputOptimizedOnlineFusedHeadDimensions,
            maxOnlineFusedQueryLength: try container.decodeIfPresent(
                Int.self,
                forKey: .maxOnlineFusedQueryLength
            ) ?? 1,
            maxTiledOnlineFusedQueryLength: try container.decodeIfPresent(
                Int.self,
                forKey: .maxTiledOnlineFusedQueryLength
            ) ?? 8,
            materializedMaskTwoStage: try container.decodeIfPresent(
                Bool.self,
                forKey: .materializedMaskTwoStage
            ) ?? true,
            supportedDTypes: try container.decodeIfPresent(
                [TurboQuantDTypeKind].self,
                forKey: .supportedDTypes
            ) ?? [.float16, .bfloat16, .float32],
            supportedMasks: try container.decodeIfPresent(
                [TurboQuantAttentionMaskKind].self,
                forKey: .supportedMasks
            ) ?? [.none, .causal, .materializedArray],
            supportedDeviceFamilies: try container.decodeIfPresent(
                [String].self,
                forKey: .supportedDeviceFamilies
            ) ?? []
        )
    }
}

public struct TurboQuantAttentionRequest: Equatable, Codable, Sendable {
    public var queryShape: [Int]
    public var keyLayout: TurboQuantAttentionLayout
    public var valueLayout: TurboQuantAttentionLayout
    public var queryDType: DType
    public var outputDType: DType
    public var maskKind: TurboQuantAttentionMaskKind
    public var hasSinks: Bool
    public var preferOnlineFused: Bool
    public var memoryBudgetBytes: Int?
    public var fallbackState: TurboQuantAttentionFallbackState
    public var deviceFamily: String?
    public var sparseVThreshold: Float?

    public init(
        queryShape: [Int],
        keyLayout: TurboQuantAttentionLayout,
        valueLayout: TurboQuantAttentionLayout,
        queryDType: DType,
        outputDType: DType,
        maskKind: TurboQuantAttentionMaskKind = .none,
        hasSinks: Bool = false,
        preferOnlineFused: Bool = true,
        memoryBudgetBytes: Int? = nil,
        fallbackState: TurboQuantAttentionFallbackState = .none,
        deviceFamily: String? = nil,
        sparseVThreshold: Float? = nil
    ) {
        self.queryShape = queryShape
        self.keyLayout = keyLayout
        self.valueLayout = valueLayout
        self.queryDType = queryDType
        self.outputDType = outputDType
        self.maskKind = maskKind
        self.hasSinks = hasSinks
        self.preferOnlineFused = preferOnlineFused
        self.memoryBudgetBytes = memoryBudgetBytes
        self.fallbackState = fallbackState
        self.deviceFamily = deviceFamily
        self.sparseVThreshold = sparseVThreshold
    }
}

public struct TurboQuantAttentionLayout: Hashable, Codable, Sendable {
    public static let legacyVersion = 4
    public static let splitMagnitudeVersion = 6
    public static let currentVersion = splitMagnitudeVersion
    public static let nextVersion = currentVersion
    public static let productionDefaultVersion = currentVersion
    public static let supportedVersions = [legacyVersion, 5, splitMagnitudeVersion]
    // Layout v7 (tile-transposed K planes): deliberately NOT added to supportedVersions.
    // supportedVersions feeds the shared validators guarding ~12 unported entry points;
    // adding 7 there would be fail-open. v7 is admitted only at the explicitly ported
    // entries enumerated in SPEC 2 section 5.
    public static let tileTransposedVersion = 7

    public var layoutVersion: Int
    public var batchSize: Int
    public var kvHeadCount: Int
    public var capacity: Int
    public var logicalLength: Int
    public var ringOffset: Int
    public var pinnedPrefixLength: Int
    public var headDimension: Int
    public var groupsPerVector: Int
    public var magnitudeWordsPerGroup: Int
    public var bitsetWordsPerGroup: Int

    public init(
        layoutVersion: Int = TurboQuantAttentionLayout.productionDefaultVersion,
        batchSize: Int,
        kvHeadCount: Int,
        capacity: Int,
        logicalLength: Int,
        ringOffset: Int = 0,
        pinnedPrefixLength: Int = 0,
        headDimension: Int,
        groupsPerVector: Int,
        magnitudeWordsPerGroup: Int,
        bitsetWordsPerGroup: Int
    ) {
        self.layoutVersion = layoutVersion
        self.batchSize = batchSize
        self.kvHeadCount = kvHeadCount
        self.capacity = capacity
        self.logicalLength = logicalLength
        self.ringOffset = ringOffset
        self.pinnedPrefixLength = pinnedPrefixLength
        self.headDimension = headDimension
        self.groupsPerVector = groupsPerVector
        self.magnitudeWordsPerGroup = magnitudeWordsPerGroup
        self.bitsetWordsPerGroup = bitsetWordsPerGroup
    }

    public var logicalShape: [Int] {
        [batchSize, kvHeadCount, logicalLength, headDimension]
    }

    public var storageShape: [Int] {
        [batchSize, kvHeadCount, capacity, headDimension]
    }

    public var isLayoutV5: Bool {
        layoutVersion >= 5
    }
}

public struct TurboQuantAttentionCode {
    public var layout: TurboQuantAttentionLayout
    public var preset: TurboQuantPreset
    public var role: TurboQuantTensorRole
    public var groupSize: Int
    public var seed: UInt64
    public var valueBits: Int
    public var scalesPerGroup: Int
    public var packedMagnitudes: MLXArray
    public var signs: MLXArray
    public var highPrecisionMask: MLXArray
    public var residualSigns: MLXArray
    public var scales: MLXArray

    public init(
        layout: TurboQuantAttentionLayout,
        preset: TurboQuantPreset,
        role: TurboQuantTensorRole,
        groupSize: Int,
        seed: UInt64,
        valueBits: Int? = nil,
        scalesPerGroup: Int? = nil,
        packedMagnitudes: MLXArray,
        signs: MLXArray,
        highPrecisionMask: MLXArray,
        residualSigns: MLXArray,
        scales: MLXArray
    ) {
        self.layout = layout
        self.preset = preset
        self.role = role
        self.groupSize = groupSize
        self.seed = seed
        self.valueBits = valueBits ?? preset.defaultValueBits
        self.scalesPerGroup = scalesPerGroup ?? 2
        self.packedMagnitudes = packedMagnitudes
        self.signs = signs
        self.highPrecisionMask = highPrecisionMask
        self.residualSigns = residualSigns
        self.scales = scales
    }

    public var storageByteCount: Int {
        if role == .value {
            return packedMagnitudes.nbytes + scales.nbytes
        }
        return packedMagnitudes.nbytes
            + signs.nbytes
            + highPrecisionMask.nbytes
            + residualSigns.nbytes
            + scales.nbytes
    }

    public var approximateBitsPerValue: Double {
        let values =
            layout.batchSize * layout.kvHeadCount
            * Swift.max(layout.logicalLength, 1) * layout.headDimension
        return Double(storageByteCount * 8) / Double(values)
    }
}

public struct TurboQuantPolarWHTAttentionValueCode {
    public var layout: TurboQuantAttentionLayout
    public var bits: Int
    public var seed: UInt64
    public var packedWordsPerVector: Int
    public var packedIndices: MLXArray
    public var norms: MLXArray

    public init(
        layout: TurboQuantAttentionLayout,
        bits: Int,
        seed: UInt64,
        packedWordsPerVector: Int,
        packedIndices: MLXArray,
        norms: MLXArray
    ) {
        self.layout = layout
        self.bits = bits
        self.seed = seed
        self.packedWordsPerVector = packedWordsPerVector
        self.packedIndices = packedIndices
        self.norms = norms
    }

    public var logicalValueCount: Int {
        layout.batchSize * layout.kvHeadCount * layout.logicalLength * layout.headDimension
    }

    public var vectorCount: Int {
        layout.batchSize * layout.kvHeadCount * layout.logicalLength
    }

    public var capacityVectorCount: Int {
        layout.batchSize * layout.kvHeadCount * layout.capacity
    }

    public var packedIndexShape: [Int] {
        [layout.batchSize, layout.kvHeadCount, layout.capacity, packedWordsPerVector]
    }

    public var normShape: [Int] {
        [layout.batchSize, layout.kvHeadCount, layout.capacity]
    }

    public var residentPayloadByteCount: Int {
        packedIndices.nbytes + norms.nbytes
    }

    public var storageByteCount: Int {
        residentPayloadByteCount
    }

    public var approximateBitsPerValue: Double {
        guard logicalValueCount > 0 else { return 0 }
        return Double(residentPayloadByteCount * 8) / Double(logicalValueCount)
    }
}

public struct TurboQuantHybridAffineK8PolarWHTValueEncodeResult {
    public var key: TurboQuantPackedTensor
    public var value: TurboQuantPolarWHTAttentionValueCode

    public init(
        key: TurboQuantPackedTensor,
        value: TurboQuantPolarWHTAttentionValueCode
    ) {
        self.key = key
        self.value = value
    }
}

public func turboQuantAttentionDecision(
    request: TurboQuantAttentionRequest,
    capabilities: TurboQuantAttentionCapabilities =
        TurboQuantKernelAvailability.current.attentionCapabilities
) throws -> TurboQuantAttentionDecision {
    try validateAttentionDecisionRequest(request)

    var rejected: [RejectedPath] = []
    let requiresBFloatOutput = request.outputDType == .bfloat16
    let queryDTypeKind = TurboQuantDTypeKind(request.queryDType)
    let outputDTypeKind = TurboQuantDTypeKind(request.outputDType)

    func reject(_ path: TurboQuantAttentionPath, _ reason: String) {
        rejected.append(RejectedPath(path: path, reason: reason))
    }

    func decision(
        _ path: TurboQuantAttentionPath,
        scratchBytes: Int = 0,
        fallbackReason: String? = nil
    ) -> TurboQuantAttentionDecision {
        TurboQuantAttentionDecision(
            selectedPath: path,
            outputDType: request.outputDType,
            estimatedScratchBytes: scratchBytes,
            rejectedPaths: rejected,
            headDimension: request.queryShape[3],
            queryLength: request.queryShape[2],
            logicalLength: request.keyLayout.logicalLength,
            dtype: "\(request.queryDType)->\(request.outputDType)",
            maskKind: request.maskKind.rawValue,
            kernelProfile: TurboQuantRuntimeProbe.shared.selectedKernelProfileWithoutRunningProbe(),
            fallbackReason: fallbackReason
        )
    }

    func supportsRequestDTypes(_ path: TurboQuantAttentionPath) -> Bool {
        guard let queryDTypeKind, let outputDTypeKind else {
            reject(path, "compressed attention supports only float16, bfloat16, or float32 tensors")
            return false
        }
        guard capabilities.supportedDTypes.contains(queryDTypeKind),
            capabilities.supportedDTypes.contains(outputDTypeKind)
        else {
            reject(
                path,
                "dtype query=\(request.queryDType) output=\(request.outputDType) is not certified for compressed attention"
            )
            return false
        }
        return true
    }

    func supportsRequestMask(_ path: TurboQuantAttentionPath) -> Bool {
        guard capabilities.supportedMasks.contains(request.maskKind) else {
            reject(
                path, "mask \(request.maskKind.rawValue) is not certified for compressed attention")
            return false
        }
        return true
    }

    func supportsRequestDevice(_ path: TurboQuantAttentionPath) -> Bool {
        guard let deviceFamily = request.deviceFamily,
            !capabilities.supportedDeviceFamilies.isEmpty
        else {
            return true
        }
        guard capabilities.supportedDeviceFamilies.contains(deviceFamily) else {
            reject(path, "device family \(deviceFamily) is not certified for compressed attention")
            return false
        }
        return true
    }

    let nativeDTypesSupported = supportsRequestDTypes(.nativeMLXCompressed)
    let nativeMaskSupported = supportsRequestMask(.nativeMLXCompressed)
    let nativeDeviceSupported = supportsRequestDevice(.nativeMLXCompressed)
    if capabilities.nativeCompressedAttention != true {
        reject(
            .nativeMLXCompressed,
            capabilities.nativeFallbackReason
                ?? "native MLX compressed attention capability is unavailable"
        )
    } else if !nativeDTypesSupported || !nativeMaskSupported || !nativeDeviceSupported {
        // Rejection was recorded by the capability helper.
    } else if requiresBFloatOutput {
        reject(.nativeMLXCompressed, "bfloat16 native compressed attention output is gated off")
    } else if request.hasSinks {
        reject(.nativeMLXCompressed, "native MLX compressed attention does not support sinks")
    } else if let sparseVThreshold = request.sparseVThreshold,
        sparseVThreshold > 0,
        capabilities.nativeSparseVSupport != true
    {
        reject(
            .nativeMLXCompressed,
            "native MLX compressed attention Sparse V has not passed capability probing"
        )
    } else if request.maskKind == .materializedArray
        || request.maskKind == .unsupportedMaterializedArrays
    {
        reject(.nativeMLXCompressed, "native MLX compressed attention supports only none/causal masks")
    } else if request.queryShape[2] > 8 {
        reject(.nativeMLXCompressed, "query length \(request.queryShape[2]) exceeds native limit 8")
    } else if ![64, 128, 256].contains(request.queryShape[3]) {
        reject(.nativeMLXCompressed, "head dimension \(request.queryShape[3]) is not native-certified")
    } else if request.keyLayout.layoutVersion < 4 || request.keyLayout.layoutVersion > 6 {
        reject(
            .nativeMLXCompressed,
            "layout version \(request.keyLayout.layoutVersion) is not supported natively"
        )
    } else if request.keyLayout.layoutVersion != request.valueLayout.layoutVersion
        || request.keyLayout.batchSize != request.valueLayout.batchSize
        || request.keyLayout.kvHeadCount != request.valueLayout.kvHeadCount
        || request.keyLayout.capacity != request.valueLayout.capacity
        || request.keyLayout.logicalLength != request.valueLayout.logicalLength
        || request.keyLayout.ringOffset != request.valueLayout.ringOffset
        || request.keyLayout.pinnedPrefixLength != request.valueLayout.pinnedPrefixLength
        || request.keyLayout.headDimension != request.valueLayout.headDimension
        || request.keyLayout.groupsPerVector != request.valueLayout.groupsPerVector
        || request.keyLayout.bitsetWordsPerGroup != request.valueLayout.bitsetWordsPerGroup
    {
        reject(.nativeMLXCompressed, "native MLX compressed attention requires aligned K/V layouts")
    } else if request.queryShape[3] != request.keyLayout.headDimension {
        reject(.nativeMLXCompressed, "query and key head dimensions differ")
    } else {
        return decision(.nativeMLXCompressed)
    }

    if request.preferOnlineFused {
        let onlineDTypesSupported = supportsRequestDTypes(.onlineFused)
        let onlineMaskSupported = supportsRequestMask(.onlineFused)
        let onlineDeviceSupported = supportsRequestDevice(.onlineFused)
        if !capabilities.onlineFused {
            reject(.onlineFused, "online fused compressed attention capability is unavailable")
        } else if !onlineDTypesSupported
            || !onlineMaskSupported
            || !onlineDeviceSupported
        {
            // Rejection was recorded by the capability helper.
        } else if requiresBFloatOutput && !capabilities.bfloatOutput {
            reject(.onlineFused, "bfloat16 compressed attention output is unavailable")
        } else if request.hasSinks {
            reject(.onlineFused, "online fused compressed attention does not support sinks")
        } else if request.keyLayout.headDimension != request.valueLayout.headDimension
            || request.keyLayout.groupsPerVector != request.valueLayout.groupsPerVector
        {
            reject(
                .onlineFused, "online fused compressed attention requires matching K/V dimensions")
        } else if !capabilities.supportedOnlineFusedHeadDimensions.contains(request.queryShape[3]) {
            reject(
                .onlineFused,
                "head dimension \(request.queryShape[3]) is not certified for online fused attention"
            )
        } else if request.queryShape[2] > capabilities.maxOnlineFusedQueryLength {
            reject(
                .onlineFused,
                "query length \(request.queryShape[2]) exceeds online fused limit \(capabilities.maxOnlineFusedQueryLength)"
            )
        } else if request.queryShape[3] != request.keyLayout.headDimension {
            reject(.onlineFused, "query and key head dimensions differ")
        } else if request.maskKind == .materializedArray
            || request.maskKind == .unsupportedMaterializedArrays
        {
            reject(
                .onlineFused, "online fused compressed attention supports only none/causal masks")
        } else {
            return decision(.onlineFused)
        }

        if request.queryShape[2] > capabilities.maxOnlineFusedQueryLength {
            let tiledDTypesSupported = supportsRequestDTypes(.tiledOnlineFused)
            let tiledMaskSupported = supportsRequestMask(.tiledOnlineFused)
            let tiledDeviceSupported = supportsRequestDevice(.tiledOnlineFused)
            if !capabilities.tiledOnlineFused {
                reject(
                    .tiledOnlineFused,
                    "tiled online fused compressed attention capability is unavailable")
            } else if !tiledDTypesSupported
                || !tiledMaskSupported
                || !tiledDeviceSupported
            {
                // Rejection was recorded by the capability helper.
            } else if requiresBFloatOutput && !capabilities.bfloatOutput {
                reject(.tiledOnlineFused, "bfloat16 compressed attention output is unavailable")
            } else if request.hasSinks {
                reject(
                    .tiledOnlineFused,
                    "tiled online fused compressed attention does not support sinks")
            } else if request.keyLayout.headDimension != request.valueLayout.headDimension
                || request.keyLayout.groupsPerVector != request.valueLayout.groupsPerVector
            {
                reject(
                    .tiledOnlineFused,
                    "tiled online fused compressed attention requires matching K/V dimensions"
                )
            } else if !capabilities.supportedOnlineFusedHeadDimensions.contains(
                request.queryShape[3])
            {
                reject(
                    .tiledOnlineFused,
                    "head dimension \(request.queryShape[3]) is not certified for tiled online fused attention"
                )
            } else if request.queryShape[2] > capabilities.maxTiledOnlineFusedQueryLength {
                reject(
                    .tiledOnlineFused,
                    "query length \(request.queryShape[2]) exceeds tiled online fused limit \(capabilities.maxTiledOnlineFusedQueryLength)"
                )
            } else if request.queryShape[3] != request.keyLayout.headDimension {
                reject(.tiledOnlineFused, "query and key head dimensions differ")
            } else if request.maskKind == .materializedArray
                || request.maskKind == .unsupportedMaterializedArrays
            {
                reject(
                    .tiledOnlineFused,
                    "tiled online fused compressed attention supports only none/causal masks"
                )
            } else {
                return decision(.tiledOnlineFused)
            }
        }
    } else {
        reject(.onlineFused, "caller disabled online fused compressed attention")
        reject(.tiledOnlineFused, "caller disabled online fused compressed attention")
    }

    let scoreScratchBytes = turboQuantTwoStageAttentionScratchBytes(
        queryShape: request.queryShape,
        keyLength: request.keyLayout.logicalLength
    )
    let twoStageDTypesSupported = supportsRequestDTypes(.twoStageCompressed)
    let twoStageMaskSupported = supportsRequestMask(.twoStageCompressed)
    let twoStageDeviceSupported = supportsRequestDevice(.twoStageCompressed)
    if !capabilities.twoStageCompressed {
        reject(.twoStageCompressed, "two-stage compressed QK/AV capability is unavailable")
    } else if !twoStageDTypesSupported
        || !twoStageMaskSupported
        || !twoStageDeviceSupported
    {
        // Rejection was recorded by the capability helper.
    } else if requiresBFloatOutput && !capabilities.bfloatOutput {
        reject(.twoStageCompressed, "bfloat16 compressed attention output is unavailable")
    } else if request.maskKind == .unsupportedMaterializedArrays {
        reject(.twoStageCompressed, "multiple materialized masks are not supported")
    } else if request.maskKind == .materializedArray && !capabilities.materializedMaskTwoStage {
        reject(.twoStageCompressed, "materialized masks are disabled for two-stage attention")
    } else if let memoryBudgetBytes = request.memoryBudgetBytes,
        scoreScratchBytes > memoryBudgetBytes
    {
        reject(
            .twoStageCompressed,
            "estimated score scratch \(scoreScratchBytes) bytes exceeds budget \(memoryBudgetBytes)"
        )
    } else {
        return TurboQuantAttentionDecision(
            selectedPath: .twoStageCompressed,
            outputDType: request.outputDType,
            estimatedScratchBytes: scoreScratchBytes,
            rejectedPaths: rejected,
            headDimension: request.queryShape[3],
            queryLength: request.queryShape[2],
            logicalLength: request.keyLayout.logicalLength,
            dtype: "\(request.queryDType)->\(request.outputDType)",
            maskKind: request.maskKind.rawValue,
            kernelProfile: TurboQuantRuntimeProbe.shared.selectedKernelProfileWithoutRunningProbe()
        )
    }

    if request.fallbackState.packedFallbackAvailable {
        return decision(
            .mlxPackedFallback,
            fallbackReason: rejected.map { "\($0.path.rawValue): \($0.reason)" }.joined(separator: "; ")
        )
    }
    if request.fallbackState.decodedFallbackAvailable || request.fallbackState.baselineAvailable {
        return decision(
            .baseline,
            fallbackReason: rejected.map { "\($0.path.rawValue): \($0.reason)" }.joined(separator: "; ")
        )
    }

    let reasons = rejected.map { "\($0.path.rawValue): \($0.reason)" }.joined(separator: "; ")
    throw TurboQuantError.unsupportedBackend(
        .metalPolarQJL,
        "No semantically correct compressed attention path is available"
            + (reasons.isEmpty ? "." : ": \(reasons).")
    )
}

public struct TurboQuantQualityThresholds: Hashable, Codable, Sendable {
    public var maxRelativeMSE: Float
    public var minCosineSimilarity: Float
    public var maxInnerProductRelativeError: Float

    public init(
        maxRelativeMSE: Float = 0.02,
        minCosineSimilarity: Float = 0.99,
        maxInnerProductRelativeError: Float = 0.08
    ) {
        self.maxRelativeMSE = maxRelativeMSE
        self.minCosineSimilarity = minCosineSimilarity
        self.maxInnerProductRelativeError = maxInnerProductRelativeError
    }
}

public struct TurboQuantQualityReport: Hashable, Codable, Sendable {
    public var mse: Float
    public var relativeMSE: Float
    public var maxAbsoluteError: Float
    public var cosineSimilarity: Float
    public var innerProductRelativeError: Float
    public var thresholds: TurboQuantQualityThresholds

    public var passes: Bool {
        relativeMSE <= thresholds.maxRelativeMSE
            && cosineSimilarity >= thresholds.minCosineSimilarity
            && innerProductRelativeError <= thresholds.maxInnerProductRelativeError
    }
}

public func turboQuantized(
    _ array: MLXArray,
    configuration: TurboQuantConfiguration = TurboQuantConfiguration(),
    stream: StreamOrDevice = .default
) -> TurboQuantPackedTensor {
    let packed = quantized(
        array,
        groupSize: configuration.groupSize,
        bits: configuration.effectiveBits,
        mode: configuration.mode,
        stream: stream
    )
    return (packed.wq, packed.scales, packed.biases)
}

public func turboDequantized(
    _ packed: TurboQuantPackedTensor,
    configuration: TurboQuantConfiguration = TurboQuantConfiguration(),
    dtype: DType? = nil,
    stream: StreamOrDevice = .default
) -> MLXArray {
    dequantized(
        packed.weight,
        scales: packed.scales,
        biases: packed.biases,
        groupSize: configuration.groupSize,
        bits: configuration.effectiveBits,
        mode: configuration.mode,
        dtype: dtype,
        stream: stream
    )
}

public func turboQuantizedMM(
    _ x: MLXArray,
    _ packed: TurboQuantPackedTensor,
    transpose: Bool = true,
    configuration: TurboQuantConfiguration = TurboQuantConfiguration(),
    stream: StreamOrDevice = .default
) -> MLXArray {
    quantizedMM(
        x,
        packed.weight,
        scales: packed.scales,
        biases: packed.biases,
        transpose: transpose,
        groupSize: configuration.groupSize,
        bits: configuration.effectiveBits,
        mode: configuration.mode,
        stream: stream
    )
}

public func turboQuantizedMM(
    _ x: MLXArray,
    _ code: TurboQuantMetalCode,
    transpose: Bool = true,
    outputDType: DType? = nil,
    stream: StreamOrDevice = .gpu
) throws -> MLXArray {
    try turboQuantMetalMM(
        x,
        code,
        transpose: transpose,
        outputDType: outputDType,
        stream: stream
    )
}

public func turboQuantReferenceEncode(
    _ array: MLXArray,
    configuration: TurboQuantConfiguration = TurboQuantConfiguration(
        backend: .polarQJLReference
    )
) throws -> TurboQuantReferenceCode {
    guard configuration.groupSize > 0 else {
        throw TurboQuantError.invalidGroupSize(configuration.groupSize)
    }

    let values = array.asArray(Float.self)
    return try encodeTurboQuantReference(
        values: values, shape: array.shape, configuration: configuration)
}

public func turboQuantReferenceDecode(
    _ code: TurboQuantReferenceCode
) throws -> MLXArray {
    let values = try decodeTurboQuantReference(code)
    return MLXArray(values, code.shape)
}

public func turboQuantReferenceQuality(
    _ array: MLXArray,
    configuration: TurboQuantConfiguration = TurboQuantConfiguration(
        backend: .polarQJLReference
    ),
    thresholds: TurboQuantQualityThresholds = TurboQuantQualityThresholds()
) throws -> TurboQuantQualityReport {
    let original = array.asArray(Float.self)
    let code = try turboQuantReferenceEncode(array, configuration: configuration)
    let decoded = try turboQuantReferenceDecode(code).asArray(Float.self)
    return try turboQuantQuality(
        original: original,
        decoded: decoded,
        seed: configuration.seed,
        thresholds: thresholds
    )
}

public func turboQuantReferenceInnerProduct(
    query: MLXArray,
    code: TurboQuantReferenceCode
) throws -> Float {
    let queryValues = query.asArray(Float.self)
    guard queryValues.count == code.valueCount else {
        throw TurboQuantError.invalidQualityInput(
            "query contains \(queryValues.count) values but code contains \(code.valueCount)"
        )
    }
    if code.format == .turboQuantProd {
        return try turboQuantProductInnerProduct(query: queryValues, code: code)
    }
    let decoded = try decodeTurboQuantReference(code)
    return zip(queryValues, decoded).reduce(Float(0)) { partial, pair in
        partial + pair.0 * pair.1
    }
}

public func turboQuantPolarWHTCentroids(bits: Int) throws -> [Float] {
    switch bits {
    case 1:
        return [-0.7979, 0.7979]
    case 2:
        return [-1.5104, -0.4528, 0.4528, 1.5104]
    case 3:
        return [-2.1520, -1.3440, -0.7560, -0.2451, 0.2451, 0.7560, 1.3440, 2.1520]
    case 4:
        return [
            -2.7326, -2.0690, -1.6180, -1.2562,
            -0.9423, -0.6568, -0.3881, -0.1284,
            0.1284, 0.3881, 0.6568, 0.9423,
            1.2562, 1.6180, 2.0690, 2.7326,
        ]
    default:
        throw TurboQuantError.invalidReferenceCode(
            "PolarWHT reference supports Lloyd-Max bit widths 1...4, got \(bits)"
        )
    }
}

public func turboQuantPolarWHTBoundaries(bits: Int) throws -> [Float] {
    let centroids = try turboQuantPolarWHTCentroids(bits: bits)
    guard centroids.count > 1 else { return [] }
    return (0 ..< centroids.count - 1).map {
        (centroids[$0] + centroids[$0 + 1]) * 0.5
    }
}

public func turboQuantPolarWHTSigns(dimension: Int, seed: UInt64) throws -> [Float] {
    guard dimension > 0 else {
        throw TurboQuantError.invalidReferenceCode("PolarWHT dimension must be positive")
    }
    return (0 ..< dimension).map { randomSign(index: $0, seed: seed) ? -1 : 1 }
}

public func turboQuantPolarWHT(_ values: [Float]) throws -> [Float] {
    guard isPowerOfTwo(values.count) else {
        throw TurboQuantError.invalidReferenceCode(
            "PolarWHT requires power-of-two dimension, got \(values.count)"
        )
    }
    var transformed = values
    fastHadamardTransform(&transformed)
    let scale = 1 / sqrt(Float(values.count))
    for index in transformed.indices {
        transformed[index] *= scale
    }
    return transformed
}

public func turboQuantPolarWHTPackedWordCount(dimension: Int, bits: Int) throws -> Int {
    guard dimension >= 0 else {
        throw TurboQuantError.invalidReferenceCode("PolarWHT packed dimension must be nonnegative")
    }
    let valuesPerWord = try turboQuantPolarWHTValuesPerWord(bits: bits)
    return (dimension + valuesPerWord - 1) / valuesPerWord
}

public func turboQuantPolarWHTPackIndices(_ indices: [UInt8], bits: Int) throws -> [UInt32] {
    let valuesPerWord = try turboQuantPolarWHTValuesPerWord(bits: bits)
    let maxIndex = UInt8((1 << bits) - 1)
    var packed = [UInt32](repeating: 0, count: (indices.count + valuesPerWord - 1) / valuesPerWord)
    for (index, value) in indices.enumerated() {
        guard value <= maxIndex else {
            throw TurboQuantError.invalidReferenceCode(
                "PolarWHT codebook index \(value) exceeds \(maxIndex) for \(bits)-bit packing"
            )
        }
        let wordIndex = index / valuesPerWord
        let offset = (index % valuesPerWord) * bits
        packed[wordIndex] |= UInt32(value) << UInt32(offset)
    }
    return packed
}

public func turboQuantPolarWHTUnpackIndices(
    _ packed: [UInt32],
    bits: Int,
    count: Int
) throws -> [UInt8] {
    guard count >= 0 else {
        throw TurboQuantError.invalidReferenceCode("PolarWHT unpack count must be nonnegative")
    }
    let valuesPerWord = try turboQuantPolarWHTValuesPerWord(bits: bits)
    let requiredWords = (count + valuesPerWord - 1) / valuesPerWord
    guard packed.count >= requiredWords else {
        throw TurboQuantError.invalidReferenceCode("PolarWHT packed index storage is truncated")
    }
    let mask = UInt32((1 << bits) - 1)
    var indices = [UInt8]()
    indices.reserveCapacity(count)
    for wordIndex in 0 ..< requiredWords {
        let word = packed[wordIndex]
        for localIndex in 0 ..< valuesPerWord where indices.count < count {
            let offset = localIndex * bits
            indices.append(UInt8((word >> UInt32(offset)) & mask))
        }
    }
    return indices
}

public func turboQuantPolarWHTReferenceEncode(
    _ array: MLXArray,
    bits: Int = 3,
    seed: UInt64 = 0x9E37_79B9_7F4A_7C15,
    headDimension requestedHeadDimension: Int? = nil
) throws -> TurboQuantPolarWHTReferenceCode {
    let values = array.asArray(Float.self)
    return try turboQuantPolarWHTReferenceEncode(
        values: values,
        shape: array.shape,
        bits: bits,
        seed: seed,
        headDimension: requestedHeadDimension
    )
}

public func turboQuantPolarWHTReferenceDecode(
    _ code: TurboQuantPolarWHTReferenceCode
) throws -> MLXArray {
    MLXArray(try turboQuantPolarWHTReferenceDecodeValues(code), code.shape)
}

public func turboQuantPolarWHTReferenceScores(
    query: [Float],
    code: TurboQuantPolarWHTReferenceCode,
    scale: Float = 1
) throws -> [Float] {
    try validateTurboQuantPolarWHTCode(code)
    guard query.count == code.headDimension else {
        throw TurboQuantError.invalidReferenceCode(
            "PolarWHT query has dimension \(query.count), expected \(code.headDimension)"
        )
    }
    let queryRotated = try turboQuantPolarWHT(zip(query, code.signs).map { $0 * $1 })
    let centroidScale = 1 / sqrt(Float(code.headDimension))
    let indices = try turboQuantPolarWHTReferenceUnpackedIndices(code)
    var scores = [Float](repeating: 0, count: code.vectorCount)
    for vectorIndex in 0 ..< code.vectorCount {
        var dot = Float(0)
        let base = vectorIndex * code.headDimension
        for dimensionIndex in 0 ..< code.headDimension {
            let centroid = code.centroids[Int(indices[base + dimensionIndex])] * centroidScale
            dot += queryRotated[dimensionIndex] * centroid
        }
        scores[vectorIndex] = dot * code.norms[vectorIndex] * scale
    }
    return scores
}

public func turboQuantPolarWHTReferenceAccumulate(
    weights: [Float],
    code: TurboQuantPolarWHTReferenceCode
) throws -> [Float] {
    try validateTurboQuantPolarWHTCode(code)
    guard weights.count == code.vectorCount else {
        throw TurboQuantError.invalidReferenceCode(
            "PolarWHT weights count \(weights.count), expected \(code.vectorCount)"
        )
    }
    let indices = try turboQuantPolarWHTReferenceUnpackedIndices(code)
    let centroidScale = 1 / sqrt(Float(code.headDimension))
    var accumulated = [Float](repeating: 0, count: code.headDimension)
    for vectorIndex in 0 ..< code.vectorCount {
        let weight = weights[vectorIndex] * code.norms[vectorIndex]
        let base = vectorIndex * code.headDimension
        for dimensionIndex in 0 ..< code.headDimension {
            accumulated[dimensionIndex] +=
                weight * code.centroids[Int(indices[base + dimensionIndex])] * centroidScale
        }
    }
    let inverseRotated = try turboQuantPolarWHT(accumulated)
    return zip(inverseRotated, code.signs).map { $0 * $1 }
}

public func turboQuantEmptyPolarWHTAttentionValueCode(
    layout: TurboQuantAttentionLayout,
    bits: Int = 3,
    seed: UInt64 = 0x9E37_79B9_7F4A_7C15,
    normStorage: DType = .float32
) throws -> TurboQuantPolarWHTAttentionValueCode {
    let packedWordsPerVector = try turboQuantPolarWHTPackedWordCount(
        dimension: layout.headDimension,
        bits: bits
    )
    let normalizedLayout = TurboQuantAttentionLayout(
        layoutVersion: layout.layoutVersion,
        batchSize: layout.batchSize,
        kvHeadCount: layout.kvHeadCount,
        capacity: layout.capacity,
        logicalLength: layout.logicalLength,
        ringOffset: layout.ringOffset,
        pinnedPrefixLength: layout.pinnedPrefixLength,
        headDimension: layout.headDimension,
        groupsPerVector: 1,
        magnitudeWordsPerGroup: packedWordsPerVector,
        bitsetWordsPerGroup: 0
    )
    let code = TurboQuantPolarWHTAttentionValueCode(
        layout: normalizedLayout,
        bits: bits,
        seed: seed,
        packedWordsPerVector: packedWordsPerVector,
        packedIndices: MLXArray.zeros(
            [
                normalizedLayout.batchSize,
                normalizedLayout.kvHeadCount,
                normalizedLayout.capacity,
                packedWordsPerVector,
            ],
            dtype: .uint32
        ),
        norms: MLXArray.zeros(
            [
                normalizedLayout.batchSize,
                normalizedLayout.kvHeadCount,
                normalizedLayout.capacity,
            ],
            dtype: normStorage
        )
    )
    try validatePolarWHTAttentionValueCode(code)
    return code
}

public func turboQuantPolarWHTReferenceEncodeAttentionValues(
    _ array: MLXArray,
    bits: Int = 3,
    seed: UInt64 = 0x9E37_79B9_7F4A_7C15,
    capacity requestedCapacity: Int? = nil,
    logicalLength requestedLogicalLength: Int? = nil,
    ringOffset: Int = 0,
    pinnedPrefixLength: Int = 0,
    normStorage: DType = .float32
) throws -> TurboQuantPolarWHTAttentionValueCode {
    try validatePolarWHTAttentionValueArray(array)
    let batchSize = array.dim(0)
    let kvHeadCount = array.dim(1)
    let inputLength = array.dim(2)
    let headDimension = array.dim(3)
    let logicalLength = requestedLogicalLength ?? inputLength
    let capacity = requestedCapacity ?? logicalLength
    guard logicalLength == inputLength else {
        throw TurboQuantError.invalidMetalConfiguration(
            "PolarWHT reference attention encode requires input length \(inputLength) to match logical length \(logicalLength)"
        )
    }
    guard capacity >= logicalLength else {
        throw TurboQuantError.invalidMetalConfiguration(
            "PolarWHT reference attention capacity \(capacity) is smaller than logical length \(logicalLength)"
        )
    }
    guard pinnedPrefixLength >= 0, pinnedPrefixLength <= logicalLength else {
        throw TurboQuantError.invalidMetalConfiguration(
            "PolarWHT reference attention pinned prefix \(pinnedPrefixLength) is outside logical length \(logicalLength)"
        )
    }
    let ringCapacity = capacity - pinnedPrefixLength
    if ringCapacity == 0 {
        guard ringOffset == 0 else {
            throw TurboQuantError.invalidMetalConfiguration(
                "PolarWHT reference attention ring offset must be zero without ring capacity"
            )
        }
    } else {
        guard ringOffset >= 0, ringOffset < ringCapacity else {
            throw TurboQuantError.invalidMetalConfiguration(
                "PolarWHT reference attention ring offset \(ringOffset) is outside ring capacity \(ringCapacity)"
            )
        }
    }

    let packedWordsPerVector = try turboQuantPolarWHTPackedWordCount(
        dimension: headDimension,
        bits: bits
    )
    let reference = try turboQuantPolarWHTReferenceEncode(
        array,
        bits: bits,
        seed: seed,
        headDimension: headDimension
    )
    let vectorCount = batchSize * kvHeadCount * logicalLength
    guard reference.vectorCount == vectorCount else {
        throw TurboQuantError.invalidReferenceCode(
            "PolarWHT reference vector count \(reference.vectorCount), expected \(vectorCount)"
        )
    }

    var packed = [UInt32](
        repeating: 0,
        count: batchSize * kvHeadCount * capacity * packedWordsPerVector
    )
    var norms = [Float](repeating: 0, count: batchSize * kvHeadCount * capacity)
    for batch in 0 ..< batchSize {
        for head in 0 ..< kvHeadCount {
            for token in 0 ..< logicalLength {
                let sourceVector = (batch * kvHeadCount + head) * logicalLength + token
                let physicalToken = turboQuantPolarWHTPhysicalToken(
                    logicalToken: token,
                    capacity: capacity,
                    ringOffset: ringOffset,
                    pinnedPrefixLength: pinnedPrefixLength
                )
                let destinationVector = (batch * kvHeadCount + head) * capacity + physicalToken
                norms[destinationVector] = reference.norms[sourceVector]
                let sourceWord = sourceVector * packedWordsPerVector
                let destinationWord = destinationVector * packedWordsPerVector
                for word in 0 ..< packedWordsPerVector {
                    packed[destinationWord + word] = reference.packedIndices[sourceWord + word]
                }
            }
        }
    }

    let layout = TurboQuantAttentionLayout(
        batchSize: batchSize,
        kvHeadCount: kvHeadCount,
        capacity: capacity,
        logicalLength: logicalLength,
        ringOffset: ringOffset,
        pinnedPrefixLength: pinnedPrefixLength,
        headDimension: headDimension,
        groupsPerVector: 1,
        magnitudeWordsPerGroup: packedWordsPerVector,
        bitsetWordsPerGroup: 0
    )
    let normArray = MLXArray(norms, [batchSize, kvHeadCount, capacity])
    let code = TurboQuantPolarWHTAttentionValueCode(
        layout: layout,
        bits: bits,
        seed: seed,
        packedWordsPerVector: packedWordsPerVector,
        packedIndices: MLXArray(
            packed,
            [batchSize, kvHeadCount, capacity, packedWordsPerVector]
        ),
        norms: normStorage == .float32 ? normArray : normArray.asType(normStorage)
    )
    try validatePolarWHTAttentionValueCode(code)
    return code
}

public func turboQuantMetalPolarWHTEncodeAttentionValues(
    _ array: MLXArray,
    bits: Int = 3,
    seed: UInt64 = 0x9E37_79B9_7F4A_7C15,
    capacity requestedCapacity: Int? = nil,
    logicalLength requestedLogicalLength: Int? = nil,
    ringOffset: Int = 0,
    pinnedPrefixLength: Int = 0,
    normStorage: DType = .float32,
    stream: StreamOrDevice = .gpu
) throws -> TurboQuantPolarWHTAttentionValueCode {
    try validatePolarWHTAttentionValueArray(array)
    guard metalRuntimeAvailable() else {
        throw TurboQuantError.unsupportedBackend(
            .metalPolarWHT,
            "Metal runtime is unavailable for PolarWHT attention encode."
        )
    }
    guard normStorage == .float32 || normStorage == .float16 else {
        throw TurboQuantError.invalidMetalConfiguration(
            "PolarWHT norm storage must be float32 or float16"
        )
    }
    let batchSize = array.dim(0)
    let kvHeadCount = array.dim(1)
    let inputLength = array.dim(2)
    let headDimension = array.dim(3)
    let logicalLength = requestedLogicalLength ?? inputLength
    let capacity = requestedCapacity ?? logicalLength
    guard logicalLength == inputLength else {
        throw TurboQuantError.invalidMetalConfiguration(
            "PolarWHT Metal attention encode requires input length \(inputLength) to match logical length \(logicalLength)"
        )
    }
    guard capacity >= logicalLength else {
        throw TurboQuantError.invalidMetalConfiguration(
            "PolarWHT Metal attention capacity \(capacity) is smaller than logical length \(logicalLength)"
        )
    }
    guard headDimension <= 256 else {
        throw TurboQuantError.invalidMetalConfiguration(
            "PolarWHT Metal attention encode supports head dimensions up to 256"
        )
    }
    guard pinnedPrefixLength >= 0, pinnedPrefixLength <= logicalLength else {
        throw TurboQuantError.invalidMetalConfiguration(
            "PolarWHT Metal attention pinned prefix \(pinnedPrefixLength) is outside logical length \(logicalLength)"
        )
    }
    let ringCapacity = capacity - pinnedPrefixLength
    if ringCapacity == 0 {
        guard ringOffset == 0 else {
            throw TurboQuantError.invalidMetalConfiguration(
                "PolarWHT Metal attention ring offset must be zero without ring capacity"
            )
        }
    } else {
        guard ringOffset >= 0, ringOffset < ringCapacity else {
            throw TurboQuantError.invalidMetalConfiguration(
                "PolarWHT Metal attention ring offset \(ringOffset) is outside ring capacity \(ringCapacity)"
            )
        }
    }

    let packedWordsPerVector = try turboQuantPolarWHTPackedWordCount(
        dimension: headDimension,
        bits: bits
    )
    let layout = TurboQuantAttentionLayout(
        batchSize: batchSize,
        kvHeadCount: kvHeadCount,
        capacity: capacity,
        logicalLength: logicalLength,
        ringOffset: ringOffset,
        pinnedPrefixLength: pinnedPrefixLength,
        headDimension: headDimension,
        groupsPerVector: 1,
        magnitudeWordsPerGroup: packedWordsPerVector,
        bitsetWordsPerGroup: 0
    )
    let vectorCount = batchSize * kvHeadCount * inputLength
    let template: [(String, any KernelTemplateArg)] = [
        ("BATCH_SIZE", batchSize),
        ("KV_HEADS", kvHeadCount),
        ("INPUT_LENGTH", inputLength),
        ("CAPACITY", capacity),
        ("HEAD_DIM", headDimension),
        ("PACKED_WORDS_PER_VECTOR", packedWordsPerVector),
        ("POLAR_WHT_BITS", bits),
        ("RING_OFFSET", ringOffset),
        ("PINNED_PREFIX_LENGTH", pinnedPrefixLength),
    ] + metalTemplateSeedWords(prefix: "SEED", value: seed)

    let encodeKernel =
        inputLength == 1
        ? TurboQuantMetalKernels.polarWHTEncodeAttention
        : TurboQuantMetalKernels.polarWHTEncodeAttentionBulk
    let outputs = encodeKernel(
        [array],
        template: template,
        grid: (vectorCount * headDimension, 1, 1),
        threadGroup: (headDimension, 1, 1),
        outputShapes: [
            [batchSize, kvHeadCount, capacity, packedWordsPerVector],
            [batchSize, kvHeadCount, capacity],
        ],
        outputDTypes: [.uint32, normStorage],
        initValue: 0,
        stream: stream
    )
    let code = TurboQuantPolarWHTAttentionValueCode(
        layout: layout,
        bits: bits,
        seed: seed,
        packedWordsPerVector: packedWordsPerVector,
        packedIndices: outputs[0],
        norms: outputs[1]
    )
    try validatePolarWHTAttentionValueCode(code)
    return code
}

public func turboQuantMetalHybridAffineK8PolarWHTValueEncode(
    keys: MLXArray,
    values: MLXArray,
    keyGroupSize: Int = 64,
    valueBits: Int = 4,
    valueSeed: UInt64 = 0x9E37_79B9_7F4A_7C15,
    capacity requestedCapacity: Int? = nil,
    logicalLength requestedLogicalLength: Int? = nil,
    ringOffset: Int = 0,
    pinnedPrefixLength: Int = 0,
    normStorage: DType = .float32,
    stream: StreamOrDevice = .gpu
) throws -> TurboQuantHybridAffineK8PolarWHTValueEncodeResult {
    try validatePolarWHTAttentionValueArray(keys)
    try validatePolarWHTAttentionValueArray(values)
    guard metalRuntimeAvailable() else {
        throw TurboQuantError.unsupportedBackend(
            .metalPolarWHT,
            "Metal runtime is unavailable for hybrid affine K8 + PolarWHT-V encode."
        )
    }
    guard normStorage == .float32 || normStorage == .float16 else {
        throw TurboQuantError.invalidMetalConfiguration(
            "PolarWHT norm storage must be float32 or float16"
        )
    }
    guard keys.shape == values.shape else {
        throw TurboQuantError.invalidMetalConfiguration(
            "hybrid affine K8 + PolarWHT-V encode requires matching key/value shapes"
        )
    }
    let batchSize = keys.dim(0)
    let kvHeadCount = keys.dim(1)
    let inputLength = keys.dim(2)
    let headDimension = keys.dim(3)
    let logicalLength = requestedLogicalLength ?? inputLength
    let capacity = requestedCapacity ?? logicalLength
    guard logicalLength == inputLength else {
        throw TurboQuantError.invalidMetalConfiguration(
            "hybrid affine K8 + PolarWHT-V encode requires input length \(inputLength) to match logical length \(logicalLength)"
        )
    }
    guard capacity >= logicalLength else {
        throw TurboQuantError.invalidMetalConfiguration(
            "hybrid affine K8 + PolarWHT-V capacity \(capacity) is smaller than logical length \(logicalLength)"
        )
    }
    guard keyGroupSize == 32 || keyGroupSize == 64 || keyGroupSize == 128,
        headDimension % keyGroupSize == 0
    else {
        throw TurboQuantError.invalidMetalConfiguration(
            "hybrid affine K8 + PolarWHT-V encode requires key group size 32, 64, or 128 dividing head dimension \(headDimension)"
        )
    }
    guard headDimension <= 256 else {
        throw TurboQuantError.invalidMetalConfiguration(
            "hybrid affine K8 + PolarWHT-V encode supports head dimensions up to 256"
        )
    }
    guard pinnedPrefixLength >= 0, pinnedPrefixLength <= logicalLength else {
        throw TurboQuantError.invalidMetalConfiguration(
            "hybrid affine K8 + PolarWHT-V pinned prefix \(pinnedPrefixLength) is outside logical length \(logicalLength)"
        )
    }
    let ringCapacity = capacity - pinnedPrefixLength
    if ringCapacity == 0 {
        guard ringOffset == 0 else {
            throw TurboQuantError.invalidMetalConfiguration(
                "hybrid affine K8 + PolarWHT-V ring offset must be zero without ring capacity"
            )
        }
    } else {
        guard ringOffset >= 0, ringOffset < ringCapacity else {
            throw TurboQuantError.invalidMetalConfiguration(
                "hybrid affine K8 + PolarWHT-V ring offset \(ringOffset) is outside ring capacity \(ringCapacity)"
            )
        }
    }

    let packedWordsPerVector = try turboQuantPolarWHTPackedWordCount(
        dimension: headDimension,
        bits: valueBits
    )
    let keyGroupsPerVector = headDimension / keyGroupSize
    let keyPackedWordsPerVector = headDimension / 4
    let layout = TurboQuantAttentionLayout(
        batchSize: batchSize,
        kvHeadCount: kvHeadCount,
        capacity: capacity,
        logicalLength: logicalLength,
        ringOffset: ringOffset,
        pinnedPrefixLength: pinnedPrefixLength,
        headDimension: headDimension,
        groupsPerVector: 1,
        magnitudeWordsPerGroup: packedWordsPerVector,
        bitsetWordsPerGroup: 0
    )
    let vectorCount = batchSize * kvHeadCount * inputLength
    let template: [(String, any KernelTemplateArg)] = [
        ("BATCH_SIZE", batchSize),
        ("KV_HEADS", kvHeadCount),
        ("INPUT_LENGTH", inputLength),
        ("CAPACITY", capacity),
        ("HEAD_DIM", headDimension),
        ("KEY_GROUP_SIZE", keyGroupSize),
        ("KEY_GROUPS_PER_VECTOR", keyGroupsPerVector),
        ("KEY_PACKED_WORDS_PER_VECTOR", keyPackedWordsPerVector),
        ("PACKED_WORDS_PER_VECTOR", packedWordsPerVector),
        ("POLAR_WHT_BITS", valueBits),
        ("KEY_SCALE_DTYPE", keys.dtype),
        ("RING_OFFSET", ringOffset),
        ("PINNED_PREFIX_LENGTH", pinnedPrefixLength),
    ] + metalTemplateSeedWords(prefix: "VALUE_SEED", value: valueSeed)

    let encodeKernel =
        inputLength == 1
        ? TurboQuantMetalKernels.hybridAffineK8PolarWHTValueEncode
        : TurboQuantMetalKernels.hybridAffineK8PolarWHTValueEncodeBulk
    let outputs = encodeKernel(
        [keys, values],
        template: template,
        grid: (vectorCount * headDimension, 1, 1),
        threadGroup: (headDimension, 1, 1),
        outputShapes: [
            [batchSize, kvHeadCount, capacity, keyPackedWordsPerVector],
            [batchSize, kvHeadCount, capacity, keyGroupsPerVector],
            [batchSize, kvHeadCount, capacity, keyGroupsPerVector],
            [batchSize, kvHeadCount, capacity, packedWordsPerVector],
            [batchSize, kvHeadCount, capacity],
        ],
        outputDTypes: [.uint32, keys.dtype, keys.dtype, .uint32, normStorage],
        initValue: 0,
        stream: stream
    )
    let valueCode = TurboQuantPolarWHTAttentionValueCode(
        layout: layout,
        bits: valueBits,
        seed: valueSeed,
        packedWordsPerVector: packedWordsPerVector,
        packedIndices: outputs[3],
        norms: outputs[4]
    )
    try validatePolarWHTAttentionValueCode(valueCode)
    return TurboQuantHybridAffineK8PolarWHTValueEncodeResult(
        key: (weight: outputs[0], scales: outputs[1], biases: outputs[2]),
        value: valueCode
    )
}

public func turboQuantMetalPolarWHTDecodeAttentionValues(
    _ code: TurboQuantPolarWHTAttentionValueCode,
    outputDType: DType = .float32,
    stream: StreamOrDevice = .gpu
) throws -> MLXArray {
    try validatePolarWHTAttentionValueCode(code)
    guard outputDType.isFloatingPoint else {
        throw TurboQuantError.invalidMetalConfiguration(
            "PolarWHT decode output dtype must be floating point"
        )
    }
    guard metalRuntimeAvailable() else {
        throw TurboQuantError.unsupportedBackend(
            .metalPolarWHT,
            "Metal runtime is unavailable for PolarWHT attention decode."
        )
    }
    guard code.layout.headDimension <= 256 else {
        throw TurboQuantError.invalidMetalConfiguration(
            "PolarWHT Metal attention decode supports head dimensions up to 256"
        )
    }

    let outputShape = code.layout.logicalShape
    let vectorCount =
        code.layout.batchSize * code.layout.kvHeadCount * code.layout.logicalLength
    let template: [(String, any KernelTemplateArg)] = [
        ("BATCH_SIZE", code.layout.batchSize),
        ("KV_HEADS", code.layout.kvHeadCount),
        ("CAPACITY", code.layout.capacity),
        ("HEAD_DIM", code.layout.headDimension),
        ("PACKED_WORDS_PER_VECTOR", code.packedWordsPerVector),
        ("POLAR_WHT_BITS", code.bits),
        ("OUTPUT_DTYPE", outputDType),
    ] + metalTemplateSeedWords(prefix: "SEED", value: code.seed)

    return TurboQuantMetalKernels.polarWHTDecodeAttention(
        [
            code.packedIndices,
            code.norms,
            Int32(code.layout.logicalLength),
            Int32(code.layout.ringOffset),
            Int32(code.layout.pinnedPrefixLength),
        ],
        template: template,
        grid: (vectorCount * code.layout.headDimension, 1, 1),
        threadGroup: (code.layout.headDimension, 1, 1),
        outputShapes: [outputShape],
        outputDTypes: [outputDType],
        stream: stream
    )[0]
}

public func turboQuantPolarWHTReferenceCode(
    attentionValueCode code: TurboQuantPolarWHTAttentionValueCode
) throws -> TurboQuantPolarWHTReferenceCode {
    try validatePolarWHTAttentionValueCode(code)
    let packedStorage = code.packedIndices.asArray(UInt32.self)
    let normStorage = code.norms.asType(.float32).asArray(Float.self)
    var packed = [UInt32]()
    var norms = [Float]()
    packed.reserveCapacity(code.vectorCount * code.packedWordsPerVector)
    norms.reserveCapacity(code.vectorCount)
    for batch in 0 ..< code.layout.batchSize {
        for head in 0 ..< code.layout.kvHeadCount {
            for token in 0 ..< code.layout.logicalLength {
                let physicalToken = turboQuantPolarWHTPhysicalToken(
                    logicalToken: token,
                    capacity: code.layout.capacity,
                    ringOffset: code.layout.ringOffset,
                    pinnedPrefixLength: code.layout.pinnedPrefixLength
                )
                let storageVector =
                    (batch * code.layout.kvHeadCount + head) * code.layout.capacity + physicalToken
                norms.append(normStorage[storageVector])
                let storageWord = storageVector * code.packedWordsPerVector
                for word in 0 ..< code.packedWordsPerVector {
                    packed.append(packedStorage[storageWord + word])
                }
            }
        }
    }
    return TurboQuantPolarWHTReferenceCode(
        shape: code.layout.logicalShape,
        bits: code.bits,
        headDimension: code.layout.headDimension,
        seed: code.seed,
        valueCount: code.logicalValueCount,
        vectorCount: code.vectorCount,
        packedWordsPerVector: code.packedWordsPerVector,
        centroids: try turboQuantPolarWHTCentroids(bits: code.bits),
        boundaries: try turboQuantPolarWHTBoundaries(bits: code.bits),
        signs: try turboQuantPolarWHTSigns(dimension: code.layout.headDimension, seed: code.seed),
        norms: norms,
        packedIndices: packed
    )
}

private func turboQuantPolarWHTPhysicalToken(
    logicalToken: Int,
    capacity: Int,
    ringOffset: Int,
    pinnedPrefixLength: Int
) -> Int {
    let pinned = pinnedPrefixLength
    if logicalToken < pinned {
        return logicalToken
    }
    let ringCapacity = capacity - pinned
    if ringCapacity == 0 {
        return min(logicalToken, max(0, capacity - 1))
    }
    let ringLogical = logicalToken - pinned
    return pinned + ((ringOffset + ringLogical) % ringCapacity)
}

public func turboQuantPolarWHTReferenceDecodeAttentionValues(
    _ code: TurboQuantPolarWHTAttentionValueCode
) throws -> MLXArray {
    try turboQuantPolarWHTReferenceDecode(
        try turboQuantPolarWHTReferenceCode(attentionValueCode: code)
    )
}

public func turboQuantPolarWHTReferenceScores(
    query: [Float],
    code: TurboQuantPolarWHTAttentionValueCode,
    scale: Float = 1
) throws -> [Float] {
    try turboQuantPolarWHTReferenceScores(
        query: query,
        code: try turboQuantPolarWHTReferenceCode(attentionValueCode: code),
        scale: scale
    )
}

public func turboQuantPolarWHTReferenceAccumulate(
    weights: [Float],
    code: TurboQuantPolarWHTAttentionValueCode
) throws -> [Float] {
    try turboQuantPolarWHTReferenceAccumulate(
        weights: weights,
        code: try turboQuantPolarWHTReferenceCode(attentionValueCode: code)
    )
}

public func turboQuantPolarWHTReferenceAccumulateAttentionValue(
    weights: [Float],
    code: TurboQuantPolarWHTAttentionValueCode,
    batchIndex: Int,
    kvHeadIndex: Int
) throws -> [Float] {
    try validatePolarWHTAttentionValueCode(code)
    guard batchIndex >= 0, batchIndex < code.layout.batchSize else {
        throw TurboQuantError.invalidReferenceCode(
            "PolarWHT attention batch index \(batchIndex) is outside 0..<\(code.layout.batchSize)"
        )
    }
    guard kvHeadIndex >= 0, kvHeadIndex < code.layout.kvHeadCount else {
        throw TurboQuantError.invalidReferenceCode(
            "PolarWHT attention head index \(kvHeadIndex) is outside 0..<\(code.layout.kvHeadCount)"
        )
    }
    guard weights.count == code.layout.logicalLength else {
        throw TurboQuantError.invalidReferenceCode(
            "PolarWHT attention weights count \(weights.count), expected \(code.layout.logicalLength)"
        )
    }

    let headDimension = code.layout.headDimension
    let centroidScale = 1 / sqrt(Float(headDimension))
    let centroids = try turboQuantPolarWHTCentroids(bits: code.bits)
    let signs = try turboQuantPolarWHTSigns(dimension: headDimension, seed: code.seed)
    let packedStorage = code.packedIndices.asArray(UInt32.self)
    let normStorage = code.norms.asType(.float32).asArray(Float.self)
    var accumulated = [Float](repeating: 0, count: headDimension)

    for logicalToken in 0 ..< code.layout.logicalLength {
        let physicalToken = turboQuantPolarWHTPhysicalToken(
            logicalToken: logicalToken,
            capacity: code.layout.capacity,
            ringOffset: code.layout.ringOffset,
            pinnedPrefixLength: code.layout.pinnedPrefixLength
        )
        let storageVector =
            (batchIndex * code.layout.kvHeadCount + kvHeadIndex) * code.layout.capacity
            + physicalToken
        let wordStart = storageVector * code.packedWordsPerVector
        let indices = try turboQuantPolarWHTUnpackIndices(
            Array(packedStorage[wordStart ..< wordStart + code.packedWordsPerVector]),
            bits: code.bits,
            count: headDimension
        )
        let weight = weights[logicalToken] * normStorage[storageVector]
        for dimensionIndex in 0 ..< headDimension {
            accumulated[dimensionIndex] +=
                weight * centroids[Int(indices[dimensionIndex])] * centroidScale
        }
    }

    let inverseRotated = try turboQuantPolarWHT(accumulated)
    return zip(inverseRotated, signs).map { $0 * $1 }
}

private func validatePolarWHTAttentionValueArray(_ array: MLXArray) throws {
    guard array.shape.count == 4 else {
        throw TurboQuantError.invalidMetalConfiguration(
            "PolarWHT attention values must have shape [B, H, T, D]"
        )
    }
    guard array.shape.reduce(1, *) > 0 else {
        throw TurboQuantError.invalidMetalConfiguration(
            "empty PolarWHT attention value tensors are not supported"
        )
    }
    guard array.dtype.isFloatingPoint else {
        throw TurboQuantError.invalidMetalConfiguration(
            "PolarWHT attention values must use floating point dtype"
        )
    }
    guard isPowerOfTwo(array.dim(3)) else {
        throw TurboQuantError.invalidMetalConfiguration(
            "PolarWHT attention head dimension must be a power of two"
        )
    }
    guard array.contiguousToDimension() == 0 else {
        throw TurboQuantError.invalidMetalConfiguration(
            "PolarWHT attention values must be canonical row-contiguous storage"
        )
    }
}

public func turboQuantMetalEncode(
    _ array: MLXArray,
    configuration: TurboQuantConfiguration = TurboQuantConfiguration(backend: .metalPolarQJL),
    stream: StreamOrDevice = .gpu
) throws -> TurboQuantMetalCode {
    try validateMetalConfiguration(array: array, configuration: configuration)
    try requireTurboQuantMetalCodec()

    let valueCount = array.size
    let groupSize = configuration.groupSize
    let groupCount = (valueCount + groupSize - 1) / groupSize
    let magnitudeWordsPerGroup = metalMagnitudeWordsPerGroup(
        groupSize: groupSize,
        preset: configuration.preset,
        role: configuration.role,
        valueBits: configuration.resolvedValueBits
    )
    let bitsetWordsPerGroup = (groupSize + 31) / 32
    let scalesPerGroup = metalScalesPerGroup(role: configuration.role)
    let threadGroupSize = Swift.max(1, Swift.min(groupCount, 64))
    let bitsetShape = [groupCount * bitsetWordsPerGroup]
    let unusedBitsetShape = turboQuantCompactUnusedBitsetShape
    let signsShape = configuration.role == .value ? unusedBitsetShape : bitsetShape
    let highMaskShape = configuration.role == .value ? unusedBitsetShape : bitsetShape

    let outputs = TurboQuantMetalKernels.encode(
        [array],
        template: metalTemplate(
            configuration: configuration,
            valueCount: valueCount,
            groupCount: groupCount,
            magnitudeWordsPerGroup: magnitudeWordsPerGroup,
            bitsetWordsPerGroup: bitsetWordsPerGroup
        ),
        grid: (groupCount, 1, 1),
        threadGroup: (threadGroupSize, 1, 1),
        outputShapes: [
            [groupCount * magnitudeWordsPerGroup],
            signsShape,
            highMaskShape,
            unusedBitsetShape,
            [groupCount, scalesPerGroup],
        ],
        outputDTypes: [.uint32, .uint32, .uint32, .uint32, .float32],
        initValue: 0,
        stream: stream
    )

    return TurboQuantMetalCode(
        shape: array.shape,
        preset: configuration.preset,
        role: configuration.role,
        groupSize: groupSize,
        seed: configuration.seed,
        valueBits: configuration.resolvedValueBits,
        valueCount: valueCount,
        groupCount: groupCount,
        magnitudeWordsPerGroup: magnitudeWordsPerGroup,
        bitsetWordsPerGroup: bitsetWordsPerGroup,
        scalesPerGroup: scalesPerGroup,
        packedMagnitudes: outputs[0],
        signs: outputs[1],
        highPrecisionMask: outputs[2],
        residualSigns: outputs[3],
        scales: outputs[4]
    )
}

public func turboQuantMetalDecode(
    _ code: TurboQuantMetalCode,
    dtype: DType = .float32,
    stream: StreamOrDevice = .gpu
) throws -> MLXArray {
    try validateMetalCodeStorage(code)
    try requireTurboQuantMetalCodec()
    guard dtype.isFloatingPoint else {
        throw TurboQuantError.invalidMetalConfiguration(
            "decode output dtype must be floating point")
    }

    let threadGroupSize = Swift.max(1, Swift.min(code.valueCount, 256))
    let configuration = TurboQuantConfiguration(
        preset: code.preset,
        role: code.role,
        groupSize: code.groupSize,
        backend: .metalPolarQJL,
        seed: code.seed,
        valueBits: code.valueBits
    )
    let outputs = TurboQuantMetalKernels.decode(
        [
            code.packedMagnitudes,
            code.signs,
            code.highPrecisionMask,
            code.residualSigns,
            code.scales,
        ],
        template: metalTemplate(
            configuration: configuration,
            valueCount: code.valueCount,
            groupCount: code.groupCount,
            magnitudeWordsPerGroup: code.magnitudeWordsPerGroup,
            bitsetWordsPerGroup: code.bitsetWordsPerGroup,
            outputDType: dtype
        ),
        grid: (code.valueCount, 1, 1),
        threadGroup: (threadGroupSize, 1, 1),
        outputShapes: [code.shape],
        outputDTypes: [dtype],
        stream: stream
    )

    return outputs[0]
}

public func turboQuantMetalMM(
    _ x: MLXArray,
    _ code: TurboQuantMetalCode,
    transpose: Bool = true,
    outputDType: DType? = nil,
    stream: StreamOrDevice = .gpu
) throws -> MLXArray {
    try validateMetalCodeStorage(code)
    try requireTurboQuantMetalCodec()
    guard x.ndim == 2 else {
        throw TurboQuantError.invalidMetalConfiguration(
            "mixed-bit matmul input must have shape [M, K]"
        )
    }
    guard code.shape.count == 2 else {
        throw TurboQuantError.invalidMetalConfiguration(
            "mixed-bit matmul weight code must have shape [N, K] or [K, N]"
        )
    }
    guard x.dtype.isFloatingPoint else {
        throw TurboQuantError.invalidMetalConfiguration(
            "mixed-bit matmul input must be floating point")
    }
    guard (outputDType ?? x.dtype).isFloatingPoint else {
        throw TurboQuantError.invalidMetalConfiguration(
            "mixed-bit matmul output dtype must be floating point")
    }

    let xRows = x.dim(0)
    let xColumns = x.dim(1)
    let weightRows = code.shape[0]
    let weightColumns = code.shape[1]
    let outputColumns: Int
    if transpose {
        guard xColumns == weightColumns else {
            throw TurboQuantError.invalidMetalConfiguration(
                "transpose matmul expects x columns \(xColumns) to match encoded weight columns \(weightColumns)"
            )
        }
        outputColumns = weightRows
    } else {
        guard xColumns == weightRows else {
            throw TurboQuantError.invalidMetalConfiguration(
                "matmul expects x columns \(xColumns) to match encoded weight rows \(weightRows)"
            )
        }
        outputColumns = weightColumns
    }

    let outputShape = [xRows, outputColumns]
    let elementCount = outputShape.reduce(1, *)
    let configuration = TurboQuantConfiguration(
        preset: code.preset,
        role: code.role,
        groupSize: code.groupSize,
        backend: .metalPolarQJL,
        seed: code.seed,
        valueBits: code.valueBits
    )
    return TurboQuantMetalKernels.matmul(
        [
            x,
            code.packedMagnitudes,
            code.signs,
            code.highPrecisionMask,
            code.residualSigns,
            code.scales,
        ],
        template: metalTemplate(
            configuration: configuration,
            valueCount: code.valueCount,
            groupCount: code.groupCount,
            magnitudeWordsPerGroup: code.magnitudeWordsPerGroup,
            bitsetWordsPerGroup: code.bitsetWordsPerGroup,
            outputDType: outputDType ?? x.dtype
        ) + [
            ("X_ROWS", xRows),
            ("X_COLUMNS", xColumns),
            ("WEIGHT_ROWS", weightRows),
            ("WEIGHT_COLUMNS", weightColumns),
            ("TRANSPOSE_WEIGHT", transpose),
        ],
        grid: (elementCount, 1, 1),
        threadGroup: (Swift.max(1, Swift.min(elementCount, 256)), 1, 1),
        outputShapes: [outputShape],
        outputDTypes: [outputDType ?? x.dtype],
        stream: stream
    )[0]
}

public func turboQuantEmptyAttentionCode(
    layout: TurboQuantAttentionLayout,
    preset: TurboQuantPreset = .turbo3_5,
    role: TurboQuantTensorRole,
    groupSize: Int = 64,
    seed: UInt64 = 0x9E37_79B9_7F4A_7C15,
    valueBits: Int? = nil,
    attentionScaleStorage: TurboQuantScaleStorage = .float32,
    allowExperimentalLayoutV5: Bool = false
) throws -> TurboQuantAttentionCode {
    try validateAttentionLayout(layout, role: role, groupSize: groupSize)
    try validateRequestedAttentionLayoutVersion(
        layout.layoutVersion,
        allowExperimentalLayoutV5: allowExperimentalLayoutV5
    )
    try validateAttentionScaleStorage(
        attentionScaleStorage,
        layoutVersion: layout.layoutVersion,
        allowExperimentalLayoutV5: allowExperimentalLayoutV5
    )
    let resolvedValueBits = valueBits ?? preset.defaultValueBits
    let bitsetShape = [
        layout.batchSize, layout.kvHeadCount, layout.capacity,
        layout.groupsPerVector, layout.bitsetWordsPerGroup,
    ]
    let unusedBitsetShape = turboQuantCompactUnusedBitsetShape
    let signsShape = role == .value ? unusedBitsetShape : bitsetShape
    let highMaskShape =
        turboQuantStoresHighPrecisionMask(
            preset: preset,
            role: role,
            layoutVersion: layout.layoutVersion
        )
        ? bitsetShape
        : unusedBitsetShape
    let residualSignsShape = unusedBitsetShape
    let scalesPerGroup = metalScalesPerGroup(role: role)
    return TurboQuantAttentionCode(
        layout: layout,
        preset: preset,
        role: role,
        groupSize: groupSize,
        seed: seed,
        valueBits: resolvedValueBits,
        scalesPerGroup: scalesPerGroup,
        packedMagnitudes: MLXArray.zeros(
            [
                layout.batchSize, layout.kvHeadCount, layout.capacity,
                layout.groupsPerVector, layout.magnitudeWordsPerGroup,
            ],
            dtype: .uint32
        ),
        signs: MLXArray.zeros(signsShape, dtype: .uint32),
        highPrecisionMask: MLXArray.zeros(highMaskShape, dtype: .uint32),
        residualSigns: MLXArray.zeros(residualSignsShape, dtype: .uint32),
        scales: MLXArray.zeros(
            [
                layout.batchSize, layout.kvHeadCount, layout.capacity,
                layout.groupsPerVector, scalesPerGroup,
            ],
            dtype: attentionScaleStorage.dtype
        )
    )
}

public func turboQuantAttentionLayout(
    for array: MLXArray,
    preset: TurboQuantPreset = .turbo3_5,
    role: TurboQuantTensorRole = .key,
    groupSize: Int = 64,
    valueBits: Int? = nil,
    capacity: Int? = nil,
    logicalLength: Int? = nil,
    ringOffset: Int = 0,
    pinnedPrefixLength: Int = 0,
    layoutVersion: Int = TurboQuantAttentionLayout.productionDefaultVersion,
    allowExperimentalLayoutV5: Bool = false,
    allowExperimentalLayoutV7: Bool = false
) throws -> TurboQuantAttentionLayout {
    try validateAttentionShape(array.shape, dtype: array.dtype, groupSize: groupSize)
    return try turboQuantAttentionLayout(
        shape: array.shape,
        dtype: array.dtype,
        preset: preset,
        role: role,
        groupSize: groupSize,
        valueBits: valueBits,
        capacity: capacity,
        logicalLength: logicalLength,
        ringOffset: ringOffset,
        pinnedPrefixLength: pinnedPrefixLength,
        layoutVersion: layoutVersion,
        allowExperimentalLayoutV5: allowExperimentalLayoutV5,
        allowExperimentalLayoutV7: allowExperimentalLayoutV7
    )
}

public func turboQuantAttentionLayout(
    shape: [Int],
    dtype: DType = .float32,
    preset: TurboQuantPreset = .turbo3_5,
    role: TurboQuantTensorRole = .key,
    groupSize: Int = 64,
    valueBits: Int? = nil,
    capacity: Int? = nil,
    logicalLength: Int? = nil,
    ringOffset: Int = 0,
    pinnedPrefixLength: Int = 0,
    layoutVersion: Int = TurboQuantAttentionLayout.productionDefaultVersion,
    allowExperimentalLayoutV5: Bool = false,
    allowExperimentalLayoutV7: Bool = false
) throws -> TurboQuantAttentionLayout {
    try validateRequestedAttentionLayoutVersion(
        layoutVersion,
        allowExperimentalLayoutV5: allowExperimentalLayoutV5,
        allowExperimentalLayoutV7: allowExperimentalLayoutV7
    )
    try validateAttentionShape(shape, dtype: dtype, groupSize: groupSize)
    let headDimension = shape[3]
    let groupsPerVector = (headDimension + groupSize - 1) / groupSize
    var resolvedCapacity = capacity ?? shape[2]
    if layoutVersion == TurboQuantAttentionLayout.tileTransposedVersion, capacity == nil {
        resolvedCapacity = (resolvedCapacity + 31) / 32 * 32
    }
    let resolvedLogicalLength = logicalLength ?? shape[2]
    let layout = TurboQuantAttentionLayout(
        layoutVersion: layoutVersion,
        batchSize: shape[0],
        kvHeadCount: shape[1],
        capacity: resolvedCapacity,
        logicalLength: resolvedLogicalLength,
        ringOffset: ringOffset,
        pinnedPrefixLength: pinnedPrefixLength,
        headDimension: headDimension,
        groupsPerVector: groupsPerVector,
        magnitudeWordsPerGroup: metalMagnitudeWordsPerGroup(
            groupSize: groupSize,
            preset: preset,
            role: role,
            valueBits: valueBits ?? preset.defaultValueBits,
            layoutVersion: layoutVersion
        ),
        bitsetWordsPerGroup: (groupSize + 31) / 32
    )
    try validateAttentionLayout(
        layout,
        role: role,
        groupSize: groupSize,
        allowTileTransposedV7: layoutVersion == TurboQuantAttentionLayout.tileTransposedVersion
    )
    return layout
}

public func turboQuantMetalEncodeAttention(
    _ array: MLXArray,
    configuration: TurboQuantConfiguration = TurboQuantConfiguration(
        role: .key,
        backend: .metalPolarQJL
    ),
    capacity: Int? = nil,
    logicalLength: Int? = nil,
    ringOffset: Int = 0,
    pinnedPrefixLength: Int = 0,
    stream: StreamOrDevice = .gpu
) throws -> TurboQuantAttentionCode {
    try validateAttentionShape(array.shape, dtype: array.dtype, groupSize: configuration.groupSize)
    try validateAttentionConfiguration(configuration)
    if configuration.role == .value {
        try validateTurboQuantValueBits(configuration.resolvedValueBits)
    }
    try requireTurboQuantMetalAttention()

    let layout = try turboQuantAttentionLayout(
        for: array,
        preset: configuration.preset,
        role: configuration.role,
        groupSize: configuration.groupSize,
        valueBits: configuration.resolvedValueBits,
        capacity: capacity,
        logicalLength: logicalLength,
        ringOffset: ringOffset,
        pinnedPrefixLength: pinnedPrefixLength,
        layoutVersion: configuration.attentionLayoutVersion,
        allowExperimentalLayoutV5: configuration.allowExperimentalLayoutV5,
        allowExperimentalLayoutV7: configuration.allowExperimentalLayoutV7
    )
    guard layout.logicalLength <= layout.capacity else {
        throw TurboQuantError.invalidMetalConfiguration(
            "logical length cannot exceed compressed attention capacity"
        )
    }
    guard array.dim(2) <= layout.capacity else {
        throw TurboQuantError.invalidMetalConfiguration(
            "attention input length \(array.dim(2)) exceeds compressed attention capacity \(layout.capacity)"
        )
    }

    let rowGroupCount =
        layout.batchSize * layout.kvHeadCount
        * array.dim(2) * layout.groupsPerVector
    let bitsetShape = [
        layout.batchSize, layout.kvHeadCount, layout.capacity,
        layout.groupsPerVector, layout.bitsetWordsPerGroup,
    ]
    let unusedBitsetShape = turboQuantCompactUnusedBitsetShape
    let signsShape = configuration.role == .value ? unusedBitsetShape : bitsetShape
    let highMaskShape =
        turboQuantStoresHighPrecisionMask(
            preset: configuration.preset,
            role: configuration.role,
            layoutVersion: layout.layoutVersion
        )
        ? bitsetShape
        : unusedBitsetShape
    let residualSignsShape = unusedBitsetShape
    let scalesPerGroup = metalScalesPerGroup(role: configuration.role)
    let encodeKernel =
        layout.layoutVersion == TurboQuantAttentionLayout.tileTransposedVersion
        ? TurboQuantMetalKernels.encodeAttentionV7
        : TurboQuantMetalKernels.encodeAttention
    let outputs = encodeKernel(
        [array],
        template: attentionTemplate(
            configuration: configuration,
            layout: layout,
            inputLength: array.dim(2),
            outputLength: array.dim(2),
            queryHeadCount: 0,
            queryLength: 0,
            outputDType: .float32,
            causal: false
        ),
        grid: (rowGroupCount, 1, 1),
        threadGroup: (Swift.max(1, Swift.min(rowGroupCount, 256)), 1, 1),
        outputShapes: [
            [
                layout.batchSize, layout.kvHeadCount, layout.capacity,
                layout.groupsPerVector, layout.magnitudeWordsPerGroup,
            ],
            signsShape,
            highMaskShape,
            residualSignsShape,
            [
                layout.batchSize, layout.kvHeadCount, layout.capacity,
                layout.groupsPerVector, scalesPerGroup,
            ],
        ],
        outputDTypes: [
            .uint32, .uint32, .uint32, .uint32, configuration.attentionScaleStorage.dtype,
        ],
        initValue: 0,
        stream: stream
    )

    return TurboQuantAttentionCode(
        layout: layout,
        preset: configuration.preset,
        role: configuration.role,
        groupSize: configuration.groupSize,
        seed: configuration.seed,
        valueBits: configuration.resolvedValueBits,
        scalesPerGroup: scalesPerGroup,
        packedMagnitudes: outputs[0],
        signs: outputs[1],
        highPrecisionMask: outputs[2],
        residualSigns: outputs[3],
        scales: outputs[4]
    )
}

public func turboQuantMetalDecodeAttention(
    _ code: TurboQuantAttentionCode,
    outputDType: DType = .float32,
    stream: StreamOrDevice = .gpu
) throws -> MLXArray {
    try validateAttentionLayout(code.layout, role: code.role, groupSize: code.groupSize)
    try validateAttentionCodeStorage(code)
    try requireTurboQuantMetalAttentionOutputDType(outputDType)
    try requireTurboQuantMetalAttention()

    let outputShape = code.layout.logicalShape
    let elementCount = outputShape.reduce(1, *)
    return TurboQuantMetalKernels.decodeAttention(
        [
            code.packedMagnitudes,
            code.signs,
            code.highPrecisionMask,
            code.residualSigns,
            code.scales,
            Int32(code.layout.logicalLength),
            Int32(code.layout.ringOffset),
            Int32(code.layout.pinnedPrefixLength),
        ],
        template: runtimeLayoutAttentionTemplate(
            configuration: TurboQuantConfiguration(
                preset: code.preset,
                role: code.role,
                groupSize: code.groupSize,
                backend: .metalPolarQJL,
                seed: code.seed,
                valueBits: code.valueBits,
                attentionLayoutVersion: code.layout.layoutVersion,
                allowExperimentalLayoutV5: code.layout.isLayoutV5,
                attentionScaleStorage: turboQuantAttentionScaleStorage(for: code)
            ),
            layout: code.layout,
            inputLength: code.layout.logicalLength,
            outputLength: code.layout.logicalLength,
            queryHeadCount: 0,
            queryLength: 0,
            outputDType: outputDType,
            causal: false
        ),
        grid: (elementCount, 1, 1),
        threadGroup: (Swift.max(1, Swift.min(elementCount, 256)), 1, 1),
        outputShapes: [outputShape],
        outputDTypes: [outputDType],
        stream: stream
    )[0]
}

public func turboQuantKeyPageSummaries(
    keyCode: TurboQuantAttentionCode,
    pageSize: Int = turboQuantKeyPageSummaryPageSize,
    stream: StreamOrDevice = .gpu
) throws -> MLXArray {
    try validateAttentionLayout(keyCode.layout, role: keyCode.role, groupSize: keyCode.groupSize)
    try validateAttentionCodeStorage(keyCode)
    try validateTurboQuantAttentionCode(keyCode, expectedRole: .key)
    try requireTurboQuantMetalAttention()
    guard pageSize > 0 else {
        throw TurboQuantError.invalidMetalConfiguration("page summary size must be positive")
    }
    guard pageSize == turboQuantKeyPageSummaryPageSize else {
        throw TurboQuantError.invalidMetalConfiguration(
            "cached key page summaries currently require \(turboQuantKeyPageSummaryPageSize)-token pages")
    }
    guard keyCode.scalesPerGroup >= 2 else {
        throw TurboQuantError.invalidMetalConfiguration(
            "key page summaries require key scale and residual scale slots")
    }

    let pageCapacity = (keyCode.layout.capacity + pageSize - 1) / pageSize
    let outputShape = [
        keyCode.layout.batchSize,
        keyCode.layout.kvHeadCount,
        pageCapacity,
        keyCode.layout.groupsPerVector,
    ]
    let summaryCount = outputShape.reduce(1, *)
    return TurboQuantMetalKernels.keyPageSummary(
        [
            keyCode.scales,
            Int32(keyCode.layout.logicalLength),
            Int32(keyCode.layout.ringOffset),
            Int32(keyCode.layout.pinnedPrefixLength),
        ],
        template: runtimeLayoutAttentionTemplate(
            configuration: TurboQuantConfiguration(
                preset: keyCode.preset,
                role: keyCode.role,
                groupSize: keyCode.groupSize,
                backend: .metalPolarQJL,
                seed: keyCode.seed,
                valueBits: keyCode.valueBits,
                attentionLayoutVersion: keyCode.layout.layoutVersion,
                allowExperimentalLayoutV5: keyCode.layout.isLayoutV5,
                attentionScaleStorage: turboQuantAttentionScaleStorage(for: keyCode)
            ),
            layout: keyCode.layout,
            inputLength: keyCode.layout.logicalLength,
            outputLength: keyCode.layout.logicalLength,
            queryHeadCount: 0,
            queryLength: 0,
            outputDType: .float32,
            causal: false
        ) + [
            ("PAGE_SIZE", pageSize),
            ("PAGE_CAPACITY", pageCapacity),
        ],
        grid: (summaryCount * pageSize, 1, 1),
        threadGroup: (pageSize, 1, 1),
        outputShapes: [outputShape],
        outputDTypes: [.float32],
        stream: stream
    )[0]
}

public func turboQuantMetalQK(
    queries: MLXArray,
    keyCode: TurboQuantAttentionCode,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    stream: StreamOrDevice = .gpu
) throws -> MLXArray {
    try validateAttentionQuery(queries, code: keyCode)
    try validateAttentionMask(
        mask,
        scoreShape: [
            queries.dim(0), queries.dim(1), queries.dim(2), keyCode.layout.logicalLength,
        ]
    )
    try validateAttentionCodeStorage(
        keyCode,
        allowTileTransposedV7: keyCode.layout.layoutVersion
            == TurboQuantAttentionLayout.tileTransposedVersion
    )
    try requireTurboQuantMetalAttention()
    guard keyCode.role == .key else {
        throw TurboQuantError.invalidMetalConfiguration("QK requires a key code")
    }

    let outputShape = [
        queries.dim(0), queries.dim(1), queries.dim(2), keyCode.layout.logicalLength,
    ]
    let elementCount = outputShape.reduce(1, *)
    var scores = TurboQuantMetalKernels.qk(
        [
            queries,
            keyCode.packedMagnitudes,
            keyCode.signs,
            keyCode.highPrecisionMask,
            keyCode.residualSigns,
            keyCode.scales,
            Int32(keyCode.layout.logicalLength),
            Int32(keyCode.layout.ringOffset),
            Int32(keyCode.layout.pinnedPrefixLength),
            scale,
        ],
        template: runtimeLayoutAttentionTemplate(
            configuration: TurboQuantConfiguration(
                preset: keyCode.preset,
                role: keyCode.role,
                groupSize: keyCode.groupSize,
                backend: .metalPolarQJL,
                seed: keyCode.seed,
                valueBits: keyCode.valueBits,
                attentionLayoutVersion: keyCode.layout.layoutVersion,
                allowExperimentalLayoutV5: keyCode.layout.isLayoutV5,
                attentionScaleStorage: turboQuantAttentionScaleStorage(for: keyCode)
            ),
            layout: keyCode.layout,
            inputLength: keyCode.layout.logicalLength,
            outputLength: keyCode.layout.logicalLength,
            queryHeadCount: queries.dim(1),
            queryLength: queries.dim(2),
            outputDType: .float32,
            causal: false
        ),
        grid: (elementCount, 1, 1),
        threadGroup: (Swift.max(1, Swift.min(elementCount, 256)), 1, 1),
        outputShapes: [outputShape],
        outputDTypes: [.float32],
        stream: stream
    )[0]

    try applyAttentionMask(&scores, mask: mask, stream: stream)
    return scores
}

public func turboQuantMetalPolarWHTQK(
    queries: MLXArray,
    keyCode: TurboQuantPolarWHTAttentionValueCode,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    stream: StreamOrDevice = .gpu
) throws -> MLXArray {
    try validatePolarWHTAttentionValueCode(keyCode)
    guard metalRuntimeAvailable() else {
        throw TurboQuantError.unsupportedBackend(
            .metalPolarWHT,
            "Metal runtime is unavailable for PolarWHT QK scoring."
        )
    }
    guard queries.ndim == 4 else {
        throw TurboQuantError.invalidMetalConfiguration("queries must be [B, Hq, L, D]")
    }
    guard queries.dim(0) == keyCode.layout.batchSize,
        queries.dim(3) == keyCode.layout.headDimension
    else {
        throw TurboQuantError.invalidMetalConfiguration(
            "queries do not match the PolarWHT key layout"
        )
    }
    guard queries.dim(1) % keyCode.layout.kvHeadCount == 0 else {
        throw TurboQuantError.invalidMetalConfiguration(
            "query heads must be a multiple of KV heads"
        )
    }
    guard keyCode.layout.headDimension <= 256 else {
        throw TurboQuantError.invalidMetalConfiguration(
            "PolarWHT QK supports head dimensions up to 256"
        )
    }
    guard keyCode.packedIndices.contiguousToDimension() == 0,
        keyCode.norms.contiguousToDimension() == 0
    else {
        throw TurboQuantError.invalidMetalConfiguration(
            "PolarWHT key sidecar storage must be canonical row-contiguous storage"
        )
    }

    let outputShape = [
        queries.dim(0), queries.dim(1), queries.dim(2), keyCode.layout.logicalLength,
    ]
    try validateAttentionMask(mask, scoreShape: outputShape)
    let scoreCount = outputShape.reduce(1, *)
    let template: [(String, any KernelTemplateArg)] = [
        ("BATCH_SIZE", keyCode.layout.batchSize),
        ("KV_HEADS", keyCode.layout.kvHeadCount),
        ("QUERY_HEADS", queries.dim(1)),
        ("QUERY_LENGTH", queries.dim(2)),
        ("CAPACITY", keyCode.layout.capacity),
        ("HEAD_DIM", keyCode.layout.headDimension),
        ("PACKED_WORDS_PER_VECTOR", keyCode.packedWordsPerVector),
        ("POLAR_WHT_BITS", keyCode.bits),
        ("OUTPUT_DTYPE", DType.float32),
    ] + metalTemplateSeedWords(prefix: "SEED", value: keyCode.seed)

    var scores = TurboQuantMetalKernels.polarWHTQK(
        [
            queries,
            keyCode.packedIndices,
            keyCode.norms,
            Int32(keyCode.layout.logicalLength),
            Int32(keyCode.layout.ringOffset),
            Int32(keyCode.layout.pinnedPrefixLength),
            scale,
        ],
        template: template,
        grid: (scoreCount * keyCode.layout.headDimension, 1, 1),
        threadGroup: (keyCode.layout.headDimension, 1, 1),
        outputShapes: [outputShape],
        outputDTypes: [.float32],
        stream: stream
    )[0]

    try applyAttentionMask(&scores, mask: mask, stream: stream)
    return scores
}

public func turboQuantMetalAV(
    attentionWeights: MLXArray,
    valueCode: TurboQuantAttentionCode,
    outputDType: DType = .float32,
    stream: StreamOrDevice = .gpu
) throws -> MLXArray {
    try validateAttentionCodeStorage(
        valueCode,
        allowTileTransposedV7: valueCode.layout.layoutVersion
            == TurboQuantAttentionLayout.tileTransposedVersion
    )
    try requireTurboQuantMetalAttentionOutputDType(outputDType)
    try requireTurboQuantMetalAttention()
    guard valueCode.role == .value else {
        throw TurboQuantError.invalidMetalConfiguration("AV requires a value code")
    }
    guard attentionWeights.ndim == 4 else {
        throw TurboQuantError.invalidMetalConfiguration("attention weights must be [B, Hq, L, T]")
    }
    guard attentionWeights.contiguousToDimension() == 0 else {
        throw TurboQuantError.invalidMetalConfiguration(
            "attention weights must be canonical row-contiguous storage"
        )
    }
    guard attentionWeights.dim(0) == valueCode.layout.batchSize,
        attentionWeights.dim(3) == valueCode.layout.logicalLength
    else {
        throw TurboQuantError.invalidMetalConfiguration(
            "attention weights do not match the compressed value layout"
        )
    }
    guard attentionWeights.dim(1) % valueCode.layout.kvHeadCount == 0 else {
        throw TurboQuantError.invalidMetalConfiguration(
            "query heads must be a multiple of KV heads"
        )
    }

    let outputShape = [
        attentionWeights.dim(0), attentionWeights.dim(1), attentionWeights.dim(2),
        valueCode.layout.headDimension,
    ]
    let elementCount = outputShape.reduce(1, *)
    return TurboQuantMetalKernels.av(
        [
            attentionWeights,
            valueCode.packedMagnitudes,
            valueCode.signs,
            valueCode.highPrecisionMask,
            valueCode.residualSigns,
            valueCode.scales,
            Int32(valueCode.layout.logicalLength),
            Int32(valueCode.layout.ringOffset),
            Int32(valueCode.layout.pinnedPrefixLength),
        ],
        template: runtimeLayoutAttentionTemplate(
            configuration: TurboQuantConfiguration(
                preset: valueCode.preset,
                role: valueCode.role,
                groupSize: valueCode.groupSize,
                backend: .metalPolarQJL,
                seed: valueCode.seed,
                valueBits: valueCode.valueBits,
                attentionLayoutVersion: valueCode.layout.layoutVersion,
                allowExperimentalLayoutV5: valueCode.layout.isLayoutV5,
                attentionScaleStorage: turboQuantAttentionScaleStorage(for: valueCode)
            ),
            layout: valueCode.layout,
            inputLength: valueCode.layout.logicalLength,
            outputLength: valueCode.layout.logicalLength,
            queryHeadCount: attentionWeights.dim(1),
            queryLength: attentionWeights.dim(2),
            outputDType: outputDType,
            causal: false
        ),
        grid: (elementCount, 1, 1),
        threadGroup: (Swift.max(1, Swift.min(elementCount, 256)), 1, 1),
        outputShapes: [outputShape],
        outputDTypes: [outputDType],
        stream: stream
    )[0]
}

public func turboQuantMetalPolarWHTAV(
    attentionWeights: MLXArray,
    valueCode: TurboQuantPolarWHTAttentionValueCode,
    outputDType: DType = .float32,
    stream: StreamOrDevice = .gpu
) throws -> MLXArray {
    try validatePolarWHTAttentionValueCode(valueCode)
    try requireTurboQuantMetalAttentionOutputDType(outputDType)
    guard metalRuntimeAvailable() else {
        throw TurboQuantError.unsupportedBackend(
            .metalPolarWHT,
            "Metal runtime is unavailable for PolarWHT value accumulation."
        )
    }
    guard attentionWeights.ndim == 4 else {
        throw TurboQuantError.invalidMetalConfiguration("attention weights must be [B, Hq, L, T]")
    }
    guard attentionWeights.contiguousToDimension() == 0 else {
        throw TurboQuantError.invalidMetalConfiguration(
            "attention weights must be canonical row-contiguous storage"
        )
    }
    guard valueCode.packedIndices.contiguousToDimension() == 0,
        valueCode.norms.contiguousToDimension() == 0
    else {
        throw TurboQuantError.invalidMetalConfiguration(
            "PolarWHT value sidecar storage must be canonical row-contiguous storage"
        )
    }
    guard attentionWeights.dim(0) == valueCode.layout.batchSize,
        attentionWeights.dim(3) == valueCode.layout.logicalLength
    else {
        throw TurboQuantError.invalidMetalConfiguration(
            "attention weights do not match the PolarWHT value layout"
        )
    }
    guard attentionWeights.dim(1) % valueCode.layout.kvHeadCount == 0 else {
        throw TurboQuantError.invalidMetalConfiguration(
            "query heads must be a multiple of KV heads"
        )
    }

    let outputShape = [
        attentionWeights.dim(0), attentionWeights.dim(1), attentionWeights.dim(2),
        valueCode.layout.headDimension,
    ]
    let rowCount = outputShape[0] * outputShape[1] * outputShape[2]
    let template: [(String, any KernelTemplateArg)] = [
        ("BATCH_SIZE", valueCode.layout.batchSize),
        ("KV_HEADS", valueCode.layout.kvHeadCount),
        ("QUERY_HEADS", attentionWeights.dim(1)),
        ("QUERY_LENGTH", attentionWeights.dim(2)),
        ("CAPACITY", valueCode.layout.capacity),
        ("HEAD_DIM", valueCode.layout.headDimension),
        ("PACKED_WORDS_PER_VECTOR", valueCode.packedWordsPerVector),
        ("POLAR_WHT_BITS", valueCode.bits),
        ("OUTPUT_DTYPE", outputDType),
    ] + metalTemplateSeedWords(prefix: "SEED", value: valueCode.seed)

    return TurboQuantMetalKernels.polarWHTAV(
        [
            attentionWeights,
            valueCode.packedIndices,
            valueCode.norms,
            Int32(valueCode.layout.logicalLength),
            Int32(valueCode.layout.ringOffset),
            Int32(valueCode.layout.pinnedPrefixLength),
        ],
        template: template,
        grid: (rowCount * valueCode.layout.headDimension, 1, 1),
        threadGroup: (valueCode.layout.headDimension, 1, 1),
        outputShapes: [outputShape],
        outputDTypes: [outputDType],
        stream: stream
    )[0]
}

private func validatePolarWHTAttentionPair(
    keyCode: TurboQuantPolarWHTAttentionValueCode,
    valueCode: TurboQuantPolarWHTAttentionValueCode
) throws {
    try validatePolarWHTAttentionValueCode(keyCode)
    try validatePolarWHTAttentionValueCode(valueCode)
    guard keyCode.layout.batchSize == valueCode.layout.batchSize,
        keyCode.layout.kvHeadCount == valueCode.layout.kvHeadCount,
        keyCode.layout.capacity == valueCode.layout.capacity,
        keyCode.layout.logicalLength == valueCode.layout.logicalLength,
        keyCode.layout.ringOffset == valueCode.layout.ringOffset,
        keyCode.layout.pinnedPrefixLength == valueCode.layout.pinnedPrefixLength,
        keyCode.layout.headDimension == valueCode.layout.headDimension
    else {
        throw TurboQuantError.invalidMetalConfiguration(
            "PolarWHT K/V attention layouts must match"
        )
    }
}

private func turboQuantPolarWHTThresholdedWeights(
    _ weights: MLXArray,
    threshold: Float?,
    stream: StreamOrDevice
) -> MLXArray {
    guard let threshold, threshold > 0 else { return weights }
    return MLX.where(
        weights .>= threshold,
        weights,
        MLXArray.zeros(like: weights, stream: stream),
        stream: stream
    )
}

public func turboQuantMetalPolarWHTScaledDotProductAttention(
    queries: MLXArray,
    keyCode: TurboQuantPolarWHTAttentionValueCode,
    valueCode: TurboQuantPolarWHTAttentionValueCode,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    sinks: MLXArray? = nil,
    sparseVThreshold: Float? = nil,
    outputDType: DType? = nil,
    stream: StreamOrDevice = .gpu
) throws -> MLXArray {
    try validatePolarWHTAttentionPair(keyCode: keyCode, valueCode: valueCode)
    try validateAttentionSinks(sinks, queryHeadCount: queries.dim(1))
    let scores = try turboQuantMetalPolarWHTQK(
        queries: queries,
        keyCode: keyCode,
        scale: scale,
        mask: mask,
        stream: stream
    )
    var logits = scores.asType(.float32)
    logits = try prependAttentionSinks(
        logits,
        sinks: sinks,
        queryHeadCount: queries.dim(1),
        stream: stream
    )
    var weights = softmax(logits, axis: -1, stream: stream)
    if sinks != nil {
        weights = weights[.ellipsis, 1...].contiguous(stream: stream)
    }
    weights = turboQuantPolarWHTThresholdedWeights(
        weights,
        threshold: sparseVThreshold,
        stream: stream
    )
    return try turboQuantMetalPolarWHTAV(
        attentionWeights: weights,
        valueCode: valueCode,
        outputDType: outputDType ?? queries.dtype,
        stream: stream
    )
}

public func turboQuantMetalPolarWHTScaledDotProductAttentionWithDiagnostics(
    queries: MLXArray,
    keyCode: TurboQuantPolarWHTAttentionValueCode,
    valueCode: TurboQuantPolarWHTAttentionValueCode,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    sinks: MLXArray? = nil,
    sparseVThreshold: Float? = nil,
    outputDType: DType? = nil,
    stream: StreamOrDevice = .gpu
) throws -> TurboQuantScaledDotProductAttentionResult {
    let output = try turboQuantMetalPolarWHTScaledDotProductAttention(
        queries: queries,
        keyCode: keyCode,
        valueCode: valueCode,
        scale: scale,
        mask: mask,
        sinks: sinks,
        sparseVThreshold: sparseVThreshold,
        outputDType: outputDType,
        stream: stream
    )
    guard let threshold = sparseVThreshold, threshold > 0 else {
        return TurboQuantScaledDotProductAttentionResult(
            output: output,
            sparseValueDiagnostics: TurboQuantSparseValueDiagnostics(enabled: false)
        )
    }
    let scores = try turboQuantMetalPolarWHTQK(
        queries: queries,
        keyCode: keyCode,
        scale: scale,
        mask: mask,
        stream: stream
    )
    let weights = softmax(scores.asType(.float32), axis: -1, stream: stream)
    let skipped = (weights .< threshold).asType(.int32).sum().item(Int.self)
    let retainedMassTotal = MLX.where(
        weights .>= threshold,
        weights,
        MLXArray.zeros(like: weights, stream: stream),
        stream: stream
    ).sum().item(Float.self)
    let rowCount = max(1, queries.dim(0) * queries.dim(1) * queries.dim(2))
    return TurboQuantScaledDotProductAttentionResult(
        output: output,
        sparseValueDiagnostics: TurboQuantSparseValueDiagnostics(
            enabled: true,
            threshold: threshold,
            skipped: skipped,
            considered: weights.size,
            retainedMass: Double(retainedMassTotal) / Double(rowCount)
        )
    )
}

public func turboQuantMetalHybridPolarWHTValueScaledDotProductAttention(
    queries: MLXArray,
    keyCode: TurboQuantAttentionCode,
    valueCode: TurboQuantPolarWHTAttentionValueCode,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    sinks: MLXArray? = nil,
    sparseVThreshold: Float? = nil,
    outputDType: DType? = nil,
    stream: StreamOrDevice = .gpu
) throws -> MLXArray {
    try validateTurboQuantAttentionCode(keyCode, expectedRole: .key)
    try validatePolarWHTAttentionValueCode(valueCode)
    guard keyCode.layout.batchSize == valueCode.layout.batchSize,
        keyCode.layout.kvHeadCount == valueCode.layout.kvHeadCount,
        keyCode.layout.logicalLength == valueCode.layout.logicalLength,
        keyCode.layout.headDimension == valueCode.layout.headDimension
    else {
        throw TurboQuantError.invalidMetalConfiguration(
            "hybrid PolarWHT value attention requires aligned key/value layouts"
        )
    }
    if let fused = try turboQuantMetalHybridPolarWHTValueOnlineFusedAttentionIfSupported(
        queries: queries,
        keyCode: keyCode,
        valueCode: valueCode,
        scale: scale,
        mask: mask,
        sinks: sinks,
        sparseVThreshold: sparseVThreshold,
        outputDType: outputDType ?? queries.dtype,
        stream: stream
    ) {
        return fused
    }
    let scores = try turboQuantMetalQK(
        queries: queries,
        keyCode: keyCode,
        scale: scale,
        mask: mask,
        stream: stream
    )
    var logits = scores.asType(.float32)
    logits = try prependAttentionSinks(
        logits,
        sinks: sinks,
        queryHeadCount: queries.dim(1),
        stream: stream
    )
    var weights = softmax(logits, axis: -1, stream: stream)
    if sinks != nil {
        weights = weights[.ellipsis, 1...].contiguous(stream: stream)
    }
    weights = turboQuantPolarWHTThresholdedWeights(
        weights,
        threshold: sparseVThreshold,
        stream: stream
    )
    return try turboQuantMetalPolarWHTAV(
        attentionWeights: weights,
        valueCode: valueCode,
        outputDType: outputDType ?? queries.dtype,
        stream: stream
    )
}

private func turboQuantMetalHybridPolarWHTValueOnlineFusedAttentionIfSupported(
    queries: MLXArray,
    keyCode: TurboQuantAttentionCode,
    valueCode: TurboQuantPolarWHTAttentionValueCode,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    sinks: MLXArray?,
    sparseVThreshold: Float?,
    outputDType: DType,
    stream: StreamOrDevice
) throws -> MLXArray? {
    guard sinks == nil else { return nil }
    guard sparseVThreshold == nil || sparseVThreshold == 0 else { return nil }
    guard queries.ndim == 4, queries.dim(2) <= 8 else { return nil }
    guard outputDType == .float32 || outputDType == .float16 || outputDType == .bfloat16 else {
        return nil
    }
    switch mask {
    case .none, .causal:
        break
    case .array, .arrays:
        return nil
    }
    let headDimension = valueCode.layout.headDimension
    guard headDimension > 0,
        headDimension <= 256,
        (headDimension & (headDimension - 1)) == 0,
        headDimension == queries.dim(3)
    else {
        return nil
    }
    guard keyCode.layout.capacity == valueCode.layout.capacity,
        keyCode.layout.ringOffset == valueCode.layout.ringOffset,
        keyCode.layout.pinnedPrefixLength == valueCode.layout.pinnedPrefixLength
    else {
        return nil
    }
    guard keyCode.packedMagnitudes.contiguousToDimension() == 0,
        keyCode.signs.contiguousToDimension() == 0,
        keyCode.highPrecisionMask.contiguousToDimension() == 0,
        keyCode.residualSigns.contiguousToDimension() == 0,
        keyCode.scales.contiguousToDimension() == 0,
        valueCode.packedIndices.contiguousToDimension() == 0,
        valueCode.norms.contiguousToDimension() == 0
    else {
        return nil
    }

    let outputShape = [queries.dim(0), queries.dim(1), queries.dim(2), queries.dim(3)]
    let rowCount = queries.dim(0) * queries.dim(1) * queries.dim(2)
    let kernelProfile = TurboQuantRuntimeProbe.shared.selectedKernelProfileWithoutRunningProbe()
    if let blockParallel = try turboQuantMetalHybridPolarWHTValueBlockParallelFusedAttentionIfSupported(
        queries: queries,
        keyCode: keyCode,
        valueCode: valueCode,
        scale: scale,
        outputDType: outputDType,
        kernelProfile: kernelProfile,
        stream: stream
    ) {
        return blockParallel
    }
    let threadgroupWidth = turboQuantOnlineFusedThreadgroupWidth(
        minimum: max(headDimension, kernelProfile.fusedDecodeThreadgroupWidth)
    )
    let causal: Bool
    switch mask {
    case .causal:
        causal = true
    default:
        causal = false
    }
    let template =
        runtimeLayoutAttentionTemplate(
            configuration: TurboQuantConfiguration(
                preset: keyCode.preset,
                role: .key,
                groupSize: keyCode.groupSize,
                backend: .metalPolarQJL,
                seed: keyCode.seed,
                valueBits: keyCode.valueBits,
                attentionLayoutVersion: keyCode.layout.layoutVersion,
                allowExperimentalLayoutV5: keyCode.layout.isLayoutV5,
                attentionScaleStorage: turboQuantAttentionScaleStorage(for: keyCode)
            ),
            layout: keyCode.layout,
            inputLength: keyCode.layout.logicalLength,
            outputLength: keyCode.layout.logicalLength,
            queryHeadCount: queries.dim(1),
            queryLength: queries.dim(2),
            outputDType: outputDType,
            causal: causal
        ) + [
            ("THREADS_PER_ROW", threadgroupWidth),
            ("PACKED_WORDS_PER_VECTOR", valueCode.packedWordsPerVector),
            ("POLAR_WHT_BITS", valueCode.bits),
        ] + metalTemplateSeedWords(prefix: "VALUE_SEED", value: valueCode.seed)

    return TurboQuantMetalKernels.hybridPolarWHTValueFusedAttention(
        [
            queries,
            keyCode.packedMagnitudes,
            keyCode.signs,
            keyCode.highPrecisionMask,
            keyCode.residualSigns,
            keyCode.scales,
            valueCode.packedIndices,
            valueCode.norms,
            Int32(keyCode.layout.logicalLength),
            Int32(keyCode.layout.ringOffset),
            Int32(keyCode.layout.pinnedPrefixLength),
            scale,
        ],
        template: template,
        grid: (rowCount * threadgroupWidth, 1, 1),
        threadGroup: (threadgroupWidth, 1, 1),
        outputShapes: [outputShape],
        outputDTypes: [outputDType],
        stream: stream
    )[0]
}

private func turboQuantMetalHybridPolarWHTValueBlockParallelFusedAttentionIfSupported(
    queries: MLXArray,
    keyCode: TurboQuantAttentionCode,
    valueCode: TurboQuantPolarWHTAttentionValueCode,
    scale: Float,
    outputDType: DType,
    kernelProfile: TurboQuantKernelProfile,
    stream: StreamOrDevice
) throws -> MLXArray? {
    guard queries.dim(2) == 1 else { return nil }
    guard
        let blockWidth = turboQuantResolvedBlockParallelTokenBlockSize(
            logicalLength: keyCode.layout.logicalLength,
            headDimension: queries.dim(3),
            queryLength: queries.dim(2),
            kernelProfile: kernelProfile,
            requestedBlockParallelTokenBlockSize: nil
        )
    else {
        return nil
    }
    let activeBlockCount = (keyCode.layout.logicalLength + blockWidth - 1) / blockWidth
    guard activeBlockCount > 1, activeBlockCount <= blockWidth else { return nil }

    let rowCount = queries.dim(0) * queries.dim(1) * queries.dim(2)
    let queryHeadRepeats = queries.dim(1) / keyCode.layout.kvHeadCount
    let useGroupedQueryKernel =
        kernelProfile == .macAppleSilicon
        && queries.dim(1) % keyCode.layout.kvHeadCount == 0
        && queryHeadRepeats == 4
    let template =
        runtimeLayoutAttentionTemplate(
            configuration: TurboQuantConfiguration(
                preset: keyCode.preset,
                role: .key,
                groupSize: keyCode.groupSize,
                backend: .metalPolarQJL,
                seed: keyCode.seed,
                valueBits: keyCode.valueBits,
                attentionLayoutVersion: keyCode.layout.layoutVersion,
                allowExperimentalLayoutV5: keyCode.layout.isLayoutV5,
                attentionScaleStorage: turboQuantAttentionScaleStorage(for: keyCode)
            ),
            layout: keyCode.layout,
            inputLength: keyCode.layout.logicalLength,
            outputLength: keyCode.layout.logicalLength,
            queryHeadCount: queries.dim(1),
            queryLength: queries.dim(2),
            outputDType: outputDType,
            causal: true
        ) + [
            ("THREADS_PER_BLOCK", blockWidth),
            ("BLOCK_TOKENS", blockWidth),
            ("BLOCK_COUNT", activeBlockCount),
            ("GQA_REPEATS", useGroupedQueryKernel ? queryHeadRepeats : 1),
            ("PACKED_WORDS_PER_VECTOR", valueCode.packedWordsPerVector),
            ("POLAR_WHT_BITS", valueCode.bits),
        ]

    let partialRows =
        useGroupedQueryKernel
        ? queries.dim(0) * keyCode.layout.kvHeadCount * queries.dim(2)
        : rowCount
    let partialKernel =
        useGroupedQueryKernel
        ? TurboQuantMetalKernels.hybridPolarWHTValueGQAFusedBlockPartials
        : TurboQuantMetalKernels.hybridPolarWHTValueFusedBlockPartials
    let partials = partialKernel(
        [
            queries,
            keyCode.packedMagnitudes,
            keyCode.signs,
            keyCode.highPrecisionMask,
            keyCode.residualSigns,
            keyCode.scales,
            valueCode.packedIndices,
            valueCode.norms,
            Int32(keyCode.layout.logicalLength),
            Int32(keyCode.layout.ringOffset),
            Int32(keyCode.layout.pinnedPrefixLength),
            scale,
        ],
        template: template,
        grid: (partialRows * activeBlockCount * blockWidth, 1, 1),
        threadGroup: (blockWidth, 1, 1),
        outputShapes: [
            [rowCount, activeBlockCount, 2],
            [rowCount, activeBlockCount, queries.dim(3)],
        ],
        outputDTypes: [.float32, .float32],
        stream: stream
    )

    let reduceWidth = turboQuantBlockParallelFusedThreadgroupWidth(
        minimum: max(activeBlockCount, queries.dim(3))
    )
    return TurboQuantMetalKernels.hybridPolarWHTValueFusedBlockReduce(
        partials,
        template: [
            ("ROW_COUNT", rowCount),
            ("HEAD_DIM", queries.dim(3)),
            ("BLOCK_COUNT", activeBlockCount),
            ("THREADS_PER_BLOCK", reduceWidth),
            ("OUTPUT_DTYPE", outputDType),
        ] + metalTemplateSeedWords(prefix: "VALUE_SEED", value: valueCode.seed),
        grid: (rowCount * reduceWidth, 1, 1),
        threadGroup: (reduceWidth, 1, 1),
        outputShapes: [[queries.dim(0), queries.dim(1), queries.dim(2), queries.dim(3)]],
        outputDTypes: [outputDType],
        stream: stream
    )[0]
}

public func turboQuantMetalHybridPolarWHTValueScaledDotProductAttentionWithDiagnostics(
    queries: MLXArray,
    keyCode: TurboQuantAttentionCode,
    valueCode: TurboQuantPolarWHTAttentionValueCode,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    sinks: MLXArray? = nil,
    sparseVThreshold: Float? = nil,
    outputDType: DType? = nil,
    stream: StreamOrDevice = .gpu
) throws -> TurboQuantScaledDotProductAttentionResult {
    let output = try turboQuantMetalHybridPolarWHTValueScaledDotProductAttention(
        queries: queries,
        keyCode: keyCode,
        valueCode: valueCode,
        scale: scale,
        mask: mask,
        sinks: sinks,
        sparseVThreshold: sparseVThreshold,
        outputDType: outputDType,
        stream: stream
    )
    guard let threshold = sparseVThreshold, threshold > 0 else {
        return TurboQuantScaledDotProductAttentionResult(
            output: output,
            sparseValueDiagnostics: TurboQuantSparseValueDiagnostics(enabled: false)
        )
    }
    let scores = try turboQuantMetalQK(
        queries: queries,
        keyCode: keyCode,
        scale: scale,
        mask: mask,
        stream: stream
    )
    let weights = softmax(scores.asType(.float32), axis: -1, stream: stream)
    let skipped = (weights .< threshold).asType(.int32).sum().item(Int.self)
    let retainedMassTotal = MLX.where(
        weights .>= threshold,
        weights,
        MLXArray.zeros(like: weights, stream: stream),
        stream: stream
    ).sum().item(Float.self)
    let rowCount = max(1, queries.dim(0) * queries.dim(1) * queries.dim(2))
    return TurboQuantScaledDotProductAttentionResult(
        output: output,
        sparseValueDiagnostics: TurboQuantSparseValueDiagnostics(
            enabled: true,
            threshold: threshold,
            skipped: skipped,
            considered: weights.size,
            retainedMass: Double(retainedMassTotal) / Double(rowCount)
        )
    )
}

public func turboQuantMetalHybridAffineK8PolarWHTValueScaledDotProductAttentionIfSupported(
    queries: MLXArray,
    keyWeight: MLXArray,
    keyScales: MLXArray,
    keyBiases: MLXArray,
    keyGroupSize: Int,
    valueCode: TurboQuantPolarWHTAttentionValueCode,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    sinks: MLXArray? = nil,
    sparseVThreshold: Float? = nil,
    outputDType: DType? = nil,
    stream: StreamOrDevice = .gpu
) throws -> MLXArray? {
    try validatePolarWHTAttentionValueCode(valueCode)
    guard metalRuntimeAvailable() else {
        throw TurboQuantError.unsupportedBackend(
            .metalPolarWHT,
            "Metal runtime is unavailable for hybrid affine K8 + PolarWHT value attention."
        )
    }
    guard sinks == nil else { return nil }
    guard sparseVThreshold == nil || sparseVThreshold == 0 else { return nil }
    guard queries.ndim == 4, queries.dim(2) == 1 else { return nil }
    switch mask {
    case .none, .causal:
        break
    case .array, .arrays:
        return nil
    }

    let batchSize = valueCode.layout.batchSize
    let kvHeadCount = valueCode.layout.kvHeadCount
    let logicalLength = valueCode.layout.logicalLength
    let headDimension = valueCode.layout.headDimension
    guard keyWeight.dtype == .uint32,
        keyScales.dtype.isFloatingPoint,
        keyBiases.dtype == keyScales.dtype,
        keyWeight.ndim == 4,
        keyScales.ndim == 4,
        keyBiases.ndim == 4,
        keyWeight.dim(0) == batchSize,
        keyWeight.dim(1) == kvHeadCount,
        keyWeight.dim(2) >= logicalLength,
        keyWeight.dim(3) * 4 == headDimension,
        keyScales.dim(0) == batchSize,
        keyScales.dim(1) == kvHeadCount,
        keyScales.dim(2) == keyWeight.dim(2),
        keyBiases.shape == keyScales.shape,
        keyGroupSize > 0,
        headDimension % keyGroupSize == 0,
        keyScales.dim(3) == headDimension / keyGroupSize,
        queries.dim(0) == batchSize,
        queries.dim(3) == headDimension,
        queries.dim(1) % kvHeadCount == 0
    else {
        return nil
    }
    guard headDimension > 0,
        headDimension <= 256,
        (headDimension & (headDimension - 1)) == 0
    else {
        return nil
    }
    guard keyWeight.contiguousToDimension() == 0,
        keyScales.contiguousToDimension() == 0,
        keyBiases.contiguousToDimension() == 0,
        valueCode.packedIndices.contiguousToDimension() == 0,
        valueCode.norms.contiguousToDimension() == 0
    else {
        return nil
    }

    let resolvedOutputDType = outputDType ?? queries.dtype
    guard resolvedOutputDType == .float32 || resolvedOutputDType == .float16
        || resolvedOutputDType == .bfloat16
    else {
        return nil
    }

    let kernelProfile = TurboQuantRuntimeProbe.shared.selectedKernelProfileWithoutRunningProbe()
    let blockParallelDisabledValue =
        ProcessInfo.processInfo.environment["TURBOQUANT_DISABLE_HYBRID_POLARWHT_BLOCK"]?
        .lowercased()
    let blockParallelDisabled =
        blockParallelDisabledValue.map { ["1", "true", "yes", "on"].contains($0) } ?? false
    if !blockParallelDisabled,
        let blockParallel =
        try turboQuantMetalHybridAffineK8PolarWHTValueBlockParallelFusedAttentionIfSupported(
            queries: queries,
            keyWeight: keyWeight,
            keyScales: keyScales,
            keyBiases: keyBiases,
            keyGroupSize: keyGroupSize,
            valueCode: valueCode,
            scale: scale,
            outputDType: resolvedOutputDType,
            kernelProfile: kernelProfile,
            stream: stream
        )
    {
        return blockParallel
    }

    let rowCount = queries.dim(0) * queries.dim(1) * queries.dim(2)
    let threadgroupWidth = turboQuantOnlineFusedThreadgroupWidth(
        minimum: max(headDimension, kernelProfile.fusedDecodeThreadgroupWidth)
    )
    let causal: Bool
    switch mask {
    case .causal:
        causal = true
    default:
        causal = false
    }
    let outputShape = [queries.dim(0), queries.dim(1), queries.dim(2), queries.dim(3)]
    let template: [(String, any KernelTemplateArg)] = [
        ("BATCH_SIZE", batchSize),
        ("KV_HEADS", kvHeadCount),
        ("QUERY_HEADS", queries.dim(1)),
        ("QUERY_LENGTH", queries.dim(2)),
        ("CAPACITY", valueCode.layout.capacity),
        ("KEY_CAPACITY", keyWeight.dim(2)),
        ("HEAD_DIM", headDimension),
        ("KEY_GROUP_SIZE", keyGroupSize),
        ("KEY_GROUPS_PER_VECTOR", headDimension / keyGroupSize),
        ("KEY_PACKED_WORDS_PER_VECTOR", headDimension / 4),
        ("PACKED_WORDS_PER_VECTOR", valueCode.packedWordsPerVector),
        ("POLAR_WHT_BITS", valueCode.bits),
        ("THREADS_PER_ROW", threadgroupWidth),
        ("OUTPUT_DTYPE", resolvedOutputDType),
        ("DO_CAUSAL", causal),
    ] + metalTemplateSeedWords(prefix: "VALUE_SEED", value: valueCode.seed)

    return TurboQuantMetalKernels.hybridAffineK8PolarWHTValueFusedAttention(
        [
            queries,
            keyWeight,
            keyScales,
            keyBiases,
            valueCode.packedIndices,
            valueCode.norms,
            Int32(logicalLength),
            Int32(valueCode.layout.ringOffset),
            Int32(valueCode.layout.pinnedPrefixLength),
            scale,
        ],
        template: template,
        grid: (rowCount * threadgroupWidth, 1, 1),
        threadGroup: (threadgroupWidth, 1, 1),
        outputShapes: [outputShape],
        outputDTypes: [resolvedOutputDType],
        stream: stream
    )[0]
}

private func turboQuantMetalHybridAffineK8PolarWHTValueBlockPartials(
    queries: MLXArray,
    keyWeight: MLXArray,
    keyScales: MLXArray,
    keyBiases: MLXArray,
    keyGroupSize: Int,
    valueCode: TurboQuantPolarWHTAttentionValueCode,
    scale: Float,
    outputDType: DType,
    kernelProfile: TurboQuantKernelProfile,
    stream: StreamOrDevice
) throws -> (stats: MLXArray, values: MLXArray, blockCount: Int)? {
    guard queries.dim(2) == 1 else { return nil }
    let logicalLength = valueCode.layout.logicalLength
    guard logicalLength > 0 else { return nil }
    let blockWidth =
        turboQuantResolvedBlockParallelTokenBlockSize(
            logicalLength: logicalLength,
            headDimension: queries.dim(3),
            queryLength: queries.dim(2),
            kernelProfile: kernelProfile,
            requestedBlockParallelTokenBlockSize: nil
        )
        ?? turboQuantBlockParallelFusedThreadgroupWidth(
            minimum: max(queries.dim(3), min(logicalLength, 512))
        )
    let activeBlockCount = max(1, (logicalLength + blockWidth - 1) / blockWidth)
    guard activeBlockCount <= blockWidth else { return nil }

    let rowCount = queries.dim(0) * queries.dim(1) * queries.dim(2)
    let queryHeadRepeats = queries.dim(1) / valueCode.layout.kvHeadCount
    let disableGroupedQueryKernel =
        ProcessInfo.processInfo.environment[
            "TURBOQUANT_DISABLE_HYBRID_POLARWHT_GQA_PARTIALS"
        ] == "1"
    let useGroupedQueryKernel =
        !disableGroupedQueryKernel
        &&
        kernelProfile == .macAppleSilicon
        && queries.dim(1) % valueCode.layout.kvHeadCount == 0
        && queryHeadRepeats == 2
    let template: [(String, any KernelTemplateArg)] = [
        ("BATCH_SIZE", valueCode.layout.batchSize),
        ("KV_HEADS", valueCode.layout.kvHeadCount),
        ("QUERY_HEADS", queries.dim(1)),
        ("QUERY_LENGTH", queries.dim(2)),
        ("CAPACITY", valueCode.layout.capacity),
        ("KEY_CAPACITY", keyWeight.dim(2)),
        ("HEAD_DIM", valueCode.layout.headDimension),
        ("KEY_GROUP_SIZE", keyGroupSize),
        ("KEY_GROUPS_PER_VECTOR", valueCode.layout.headDimension / keyGroupSize),
        ("KEY_PACKED_WORDS_PER_VECTOR", valueCode.layout.headDimension / 4),
        ("PACKED_WORDS_PER_VECTOR", valueCode.packedWordsPerVector),
        ("POLAR_WHT_BITS", valueCode.bits),
        ("THREADS_PER_BLOCK", blockWidth),
        ("BLOCK_TOKENS", blockWidth),
        ("BLOCK_COUNT", activeBlockCount),
        ("GQA_REPEATS", useGroupedQueryKernel ? queryHeadRepeats : 1),
        ("OUTPUT_DTYPE", outputDType),
        ("DO_CAUSAL", true),
    ]

    let partialRows =
        useGroupedQueryKernel
        ? queries.dim(0) * valueCode.layout.kvHeadCount * queries.dim(2)
        : rowCount
    let partialKernel =
        useGroupedQueryKernel
        ? TurboQuantMetalKernels.hybridAffineK8PolarWHTValueGQAFusedBlockPartials
        : TurboQuantMetalKernels.hybridAffineK8PolarWHTValueFusedBlockPartials
    let partials = partialKernel(
        [
            queries,
            keyWeight,
            keyScales,
            keyBiases,
            valueCode.packedIndices,
            valueCode.norms,
            Int32(logicalLength),
            Int32(valueCode.layout.ringOffset),
            Int32(valueCode.layout.pinnedPrefixLength),
            scale,
        ],
        template: template,
        grid: (partialRows * activeBlockCount * blockWidth, 1, 1),
        threadGroup: (blockWidth, 1, 1),
        outputShapes: [
            [rowCount, activeBlockCount, 2],
            [rowCount, activeBlockCount, queries.dim(3)],
        ],
        outputDTypes: [.float32, .float32],
        stream: stream
    )
    return (partials[0], partials[1], activeBlockCount)
}

public func turboQuantMetalSegmentedHybridAffineK8PolarWHTValueScaledDotProductAttentionIfSupported(
    queries: MLXArray,
    baseKeyWeight: MLXArray,
    baseKeyScales: MLXArray,
    baseKeyBiases: MLXArray,
    tailKeyWeight: MLXArray,
    tailKeyScales: MLXArray,
    tailKeyBiases: MLXArray,
    keyGroupSize: Int,
    baseValueCode: TurboQuantPolarWHTAttentionValueCode,
    tailValueCode: TurboQuantPolarWHTAttentionValueCode,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    sinks: MLXArray? = nil,
    sparseVThreshold: Float? = nil,
    outputDType: DType? = nil,
    stream: StreamOrDevice = .gpu
) throws -> MLXArray? {
    try validatePolarWHTAttentionValueCode(baseValueCode)
    try validatePolarWHTAttentionValueCode(tailValueCode)
    guard metalRuntimeAvailable() else {
        throw TurboQuantError.unsupportedBackend(
            .metalPolarWHT,
            "Metal runtime is unavailable for segmented hybrid affine K8 + PolarWHT value attention."
        )
    }
    guard sinks == nil else { return nil }
    guard sparseVThreshold == nil || sparseVThreshold == 0 else { return nil }
    guard queries.ndim == 4, queries.dim(2) == 1 else { return nil }
    switch mask {
    case .none, .causal:
        break
    case .array, .arrays:
        return nil
    }
    guard baseValueCode.layout.logicalLength > 0,
        tailValueCode.layout.logicalLength > 0,
        baseValueCode.seed == tailValueCode.seed,
        baseValueCode.bits == tailValueCode.bits,
        baseValueCode.layout.batchSize == tailValueCode.layout.batchSize,
        baseValueCode.layout.kvHeadCount == tailValueCode.layout.kvHeadCount,
        baseValueCode.layout.headDimension == tailValueCode.layout.headDimension
    else {
        return nil
    }

    let batchSize = baseValueCode.layout.batchSize
    let kvHeadCount = baseValueCode.layout.kvHeadCount
    let headDimension = baseValueCode.layout.headDimension
    guard keyGroupSize > 0,
        headDimension % keyGroupSize == 0,
        headDimension > 0,
        headDimension <= 256,
        (headDimension & (headDimension - 1)) == 0,
        queries.dim(0) == batchSize,
        queries.dim(3) == headDimension,
        queries.dim(1) % kvHeadCount == 0
    else {
        return nil
    }

    func validateKeyStorage(
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray,
        valueCode: TurboQuantPolarWHTAttentionValueCode
    ) -> Bool {
        weight.dtype == .uint32
            && scales.dtype.isFloatingPoint
            && biases.dtype == scales.dtype
            && weight.ndim == 4
            && scales.ndim == 4
            && biases.ndim == 4
            && weight.dim(0) == batchSize
            && weight.dim(1) == kvHeadCount
            && weight.dim(2) >= valueCode.layout.logicalLength
            && weight.dim(3) * 4 == headDimension
            && scales.dim(0) == batchSize
            && scales.dim(1) == kvHeadCount
            && scales.dim(2) == weight.dim(2)
            && biases.shape == scales.shape
            && scales.dim(3) == headDimension / keyGroupSize
            && weight.contiguousToDimension() == 0
            && scales.contiguousToDimension() == 0
            && biases.contiguousToDimension() == 0
            && valueCode.packedIndices.contiguousToDimension() == 0
            && valueCode.norms.contiguousToDimension() == 0
    }

    guard validateKeyStorage(
        weight: baseKeyWeight,
        scales: baseKeyScales,
        biases: baseKeyBiases,
        valueCode: baseValueCode
    ),
        validateKeyStorage(
            weight: tailKeyWeight,
            scales: tailKeyScales,
            biases: tailKeyBiases,
            valueCode: tailValueCode
        )
    else {
        return nil
    }

    let resolvedOutputDType = outputDType ?? queries.dtype
    guard resolvedOutputDType == .float32 || resolvedOutputDType == .float16
        || resolvedOutputDType == .bfloat16
    else {
        return nil
    }

    let kernelProfile = TurboQuantRuntimeProbe.shared.selectedKernelProfileWithoutRunningProbe()

    if ProcessInfo.processInfo.environment[
        "TURBOQUANT_SEGMENTED_HYBRID_USE_CONCAT_REFERENCE"
    ] == "1" {
        let baseLength = baseValueCode.layout.logicalLength
        let tailLength = tailValueCode.layout.logicalLength
        guard baseValueCode.layout.ringOffset == 0,
            tailValueCode.layout.ringOffset == 0,
            baseKeyWeight.dim(2) >= baseLength,
            tailKeyWeight.dim(2) >= tailLength
        else {
            return nil
        }
        let baseTokenRange = 0 ..< baseLength
        let tailTokenRange = 0 ..< tailLength
        let combinedKeyWeight = concatenated(
            [
                baseKeyWeight[.ellipsis, baseTokenRange, 0...],
                tailKeyWeight[.ellipsis, tailTokenRange, 0...],
            ],
            axis: 2,
            stream: stream
        ).contiguous(stream: stream)
        let combinedKeyScales = concatenated(
            [
                baseKeyScales[.ellipsis, baseTokenRange, 0...],
                tailKeyScales[.ellipsis, tailTokenRange, 0...],
            ],
            axis: 2,
            stream: stream
        ).contiguous(stream: stream)
        let combinedKeyBiases = concatenated(
            [
                baseKeyBiases[.ellipsis, baseTokenRange, 0...],
                tailKeyBiases[.ellipsis, tailTokenRange, 0...],
            ],
            axis: 2,
            stream: stream
        ).contiguous(stream: stream)
        var combinedLayout = baseValueCode.layout
        combinedLayout.capacity = baseLength + tailLength
        combinedLayout.logicalLength = baseLength + tailLength
        combinedLayout.ringOffset = 0
        combinedLayout.pinnedPrefixLength = min(
            baseValueCode.layout.pinnedPrefixLength,
            combinedLayout.logicalLength
        )
        let combinedValueCode = TurboQuantPolarWHTAttentionValueCode(
            layout: combinedLayout,
            bits: baseValueCode.bits,
            seed: baseValueCode.seed,
            packedWordsPerVector: baseValueCode.packedWordsPerVector,
            packedIndices: concatenated(
                [
                    baseValueCode.packedIndices[.ellipsis, baseTokenRange, 0...],
                    tailValueCode.packedIndices[.ellipsis, tailTokenRange, 0...],
                ],
                axis: 2,
                stream: stream
            ).contiguous(stream: stream),
            norms: concatenated(
                [
                    baseValueCode.norms[.ellipsis, baseTokenRange],
                    tailValueCode.norms[.ellipsis, tailTokenRange],
                ],
                axis: 2,
                stream: stream
            ).contiguous(stream: stream)
        )
        return try turboQuantMetalHybridAffineK8PolarWHTValueScaledDotProductAttentionIfSupported(
            queries: queries,
            keyWeight: combinedKeyWeight,
            keyScales: combinedKeyScales,
            keyBiases: combinedKeyBiases,
            keyGroupSize: keyGroupSize,
            valueCode: combinedValueCode,
            scale: scale,
            mask: mask,
            sinks: sinks,
            sparseVThreshold: sparseVThreshold,
            outputDType: resolvedOutputDType,
            stream: stream
        )
    }

    if ProcessInfo.processInfo.environment[
        "TURBOQUANT_ENABLE_SEGMENTED_HYBRID_ONLINE"
    ] == "1" {
        let rowCount = queries.dim(0) * queries.dim(1) * queries.dim(2)
        let threadgroupWidth = turboQuantOnlineFusedThreadgroupWidth(
            minimum: max(headDimension, kernelProfile.fusedDecodeThreadgroupWidth)
        )
        let causal: Bool
        switch mask {
        case .causal:
            causal = true
        default:
            causal = false
        }
        return TurboQuantMetalKernels.segmentedHybridAffineK8PolarWHTValueFusedAttention(
            [
                queries,
                baseKeyWeight,
                baseKeyScales,
                baseKeyBiases,
                baseValueCode.packedIndices,
                baseValueCode.norms,
                tailKeyWeight,
                tailKeyScales,
                tailKeyBiases,
                tailValueCode.packedIndices,
                tailValueCode.norms,
                Int32(baseValueCode.layout.logicalLength),
                Int32(baseValueCode.layout.ringOffset),
                Int32(baseValueCode.layout.pinnedPrefixLength),
                Int32(tailValueCode.layout.logicalLength),
                Int32(tailValueCode.layout.ringOffset),
                Int32(tailValueCode.layout.pinnedPrefixLength),
                scale,
            ],
            template: [
                ("BATCH_SIZE", batchSize),
                ("KV_HEADS", kvHeadCount),
                ("QUERY_HEADS", queries.dim(1)),
                ("QUERY_LENGTH", queries.dim(2)),
                ("BASE_CAPACITY", baseValueCode.layout.capacity),
                ("TAIL_CAPACITY", tailValueCode.layout.capacity),
                ("BASE_KEY_CAPACITY", baseKeyWeight.dim(2)),
                ("TAIL_KEY_CAPACITY", tailKeyWeight.dim(2)),
                ("HEAD_DIM", headDimension),
                ("KEY_GROUP_SIZE", keyGroupSize),
                ("KEY_GROUPS_PER_VECTOR", headDimension / keyGroupSize),
                ("KEY_PACKED_WORDS_PER_VECTOR", headDimension / 4),
                ("PACKED_WORDS_PER_VECTOR", baseValueCode.packedWordsPerVector),
                ("POLAR_WHT_BITS", baseValueCode.bits),
                ("THREADS_PER_ROW", threadgroupWidth),
                ("OUTPUT_DTYPE", resolvedOutputDType),
                ("DO_CAUSAL", causal),
            ] + metalTemplateSeedWords(prefix: "VALUE_SEED", value: baseValueCode.seed),
            grid: (rowCount * threadgroupWidth, 1, 1),
            threadGroup: (threadgroupWidth, 1, 1),
            outputShapes: [[queries.dim(0), queries.dim(1), queries.dim(2), queries.dim(3)]],
            outputDTypes: [resolvedOutputDType],
            stream: stream
        )[0]
    }

    guard let basePartials = try turboQuantMetalHybridAffineK8PolarWHTValueBlockPartials(
        queries: queries,
        keyWeight: baseKeyWeight,
        keyScales: baseKeyScales,
        keyBiases: baseKeyBiases,
        keyGroupSize: keyGroupSize,
        valueCode: baseValueCode,
        scale: scale,
        outputDType: resolvedOutputDType,
        kernelProfile: kernelProfile,
        stream: stream
    ),
        let tailPartials = try turboQuantMetalHybridAffineK8PolarWHTValueBlockPartials(
            queries: queries,
            keyWeight: tailKeyWeight,
            keyScales: tailKeyScales,
            keyBiases: tailKeyBiases,
            keyGroupSize: keyGroupSize,
            valueCode: tailValueCode,
            scale: scale,
            outputDType: resolvedOutputDType,
            kernelProfile: kernelProfile,
            stream: stream
        )
    else {
        return nil
    }

    let totalBlocks = basePartials.blockCount + tailPartials.blockCount
    let rowCount = queries.dim(0) * queries.dim(1) * queries.dim(2)
    let stats = MLXArray.zeros([rowCount, totalBlocks, 2], dtype: .float32)
    let values = MLXArray.zeros([rowCount, totalBlocks, queries.dim(3)], dtype: .float32)
    stats[0..., 0 ..< basePartials.blockCount, 0...] = basePartials.stats
    stats[0..., basePartials.blockCount ..< totalBlocks, 0...] = tailPartials.stats
    values[0..., 0 ..< basePartials.blockCount, 0...] = basePartials.values
    values[0..., basePartials.blockCount ..< totalBlocks, 0...] = tailPartials.values
    let reduceWidth = turboQuantBlockParallelFusedThreadgroupWidth(
        minimum: max(totalBlocks, queries.dim(3))
    )
    return TurboQuantMetalKernels.hybridPolarWHTValueFusedBlockReduce(
        [stats, values],
        template: [
            ("ROW_COUNT", rowCount),
            ("HEAD_DIM", queries.dim(3)),
            ("BLOCK_COUNT", totalBlocks),
            ("THREADS_PER_BLOCK", reduceWidth),
            ("OUTPUT_DTYPE", resolvedOutputDType),
        ] + metalTemplateSeedWords(prefix: "VALUE_SEED", value: baseValueCode.seed),
        grid: (rowCount * reduceWidth, 1, 1),
        threadGroup: (reduceWidth, 1, 1),
        outputShapes: [[queries.dim(0), queries.dim(1), queries.dim(2), queries.dim(3)]],
        outputDTypes: [resolvedOutputDType],
        stream: stream
    )[0]
}

public func turboQuantMetalHybridAffineK8DecodedValueScaledDotProductAttentionIfSupported(
    queries: MLXArray,
    keyWeight: MLXArray,
    keyScales: MLXArray,
    keyBiases: MLXArray,
    keyGroupSize: Int,
    decodedValues: MLXArray,
    decodedValueLogicalLength: Int? = nil,
    decodedValueRingOffset: Int = 0,
    decodedValuePinnedPrefixLength: Int = 0,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    sinks: MLXArray? = nil,
    sparseVThreshold: Float? = nil,
    outputDType: DType? = nil,
    stream: StreamOrDevice = .gpu
) throws -> MLXArray? {
    guard metalRuntimeAvailable() else {
        throw TurboQuantError.unsupportedBackend(
            .metalPolarWHT,
            "Metal runtime is unavailable for hybrid affine K8 + decoded value attention."
        )
    }
    guard sinks == nil else { return nil }
    guard sparseVThreshold == nil || sparseVThreshold == 0 else { return nil }
    guard queries.ndim == 4, queries.dim(2) == 1 else { return nil }
    switch mask {
    case .none, .causal:
        break
    case .array, .arrays:
        return nil
    }

    guard decodedValues.ndim == 4 else { return nil }
    let batchSize = decodedValues.dim(0)
    let kvHeadCount = decodedValues.dim(1)
    let valueCapacity = decodedValues.dim(2)
    let logicalLength = decodedValueLogicalLength ?? valueCapacity
    let headDimension = decodedValues.dim(3)
    guard keyWeight.dtype == .uint32,
        keyScales.dtype.isFloatingPoint,
        keyBiases.dtype == keyScales.dtype,
        decodedValues.dtype.isFloatingPoint,
        keyWeight.ndim == 4,
        keyScales.ndim == 4,
        keyBiases.ndim == 4,
        keyWeight.dim(0) == batchSize,
        keyWeight.dim(1) == kvHeadCount,
        keyWeight.dim(2) >= logicalLength,
        keyWeight.dim(3) * 4 == headDimension,
        keyScales.dim(0) == batchSize,
        keyScales.dim(1) == kvHeadCount,
        keyScales.dim(2) == keyWeight.dim(2),
        keyBiases.shape == keyScales.shape,
        keyGroupSize > 0,
        headDimension % keyGroupSize == 0,
        keyScales.dim(3) == headDimension / keyGroupSize,
        queries.dim(0) == batchSize,
        queries.dim(3) == headDimension,
        queries.dim(1) % kvHeadCount == 0
    else {
        return nil
    }
    guard logicalLength > 0,
        logicalLength <= valueCapacity,
        decodedValuePinnedPrefixLength >= 0,
        decodedValuePinnedPrefixLength <= logicalLength,
        decodedValuePinnedPrefixLength <= valueCapacity,
        decodedValueRingOffset >= 0,
        headDimension > 0,
        headDimension <= 256,
        headDimension % 4 == 0
    else {
        return nil
    }
    let valueRingCapacity = valueCapacity - decodedValuePinnedPrefixLength
    guard valueRingCapacity > 0
        ? decodedValueRingOffset < valueRingCapacity
        : decodedValueRingOffset == 0
    else {
        return nil
    }
    guard keyWeight.contiguousToDimension() == 0,
        keyScales.contiguousToDimension() == 0,
        keyBiases.contiguousToDimension() == 0,
        decodedValues.contiguousToDimension() == 0
    else {
        return nil
    }

    let resolvedOutputDType = outputDType ?? queries.dtype
    guard resolvedOutputDType == .float32 || resolvedOutputDType == .float16
        || resolvedOutputDType == .bfloat16
    else {
        return nil
    }

    let kernelProfile = TurboQuantRuntimeProbe.shared.selectedKernelProfileWithoutRunningProbe()
    if let blockParallel =
        try turboQuantMetalHybridAffineK8DecodedValueBlockParallelFusedAttentionIfSupported(
            queries: queries,
            keyWeight: keyWeight,
            keyScales: keyScales,
            keyBiases: keyBiases,
            keyGroupSize: keyGroupSize,
            decodedValues: decodedValues,
            decodedValueLogicalLength: logicalLength,
            decodedValueRingOffset: decodedValueRingOffset,
            decodedValuePinnedPrefixLength: decodedValuePinnedPrefixLength,
            scale: scale,
            outputDType: resolvedOutputDType,
            kernelProfile: kernelProfile,
            stream: stream
        )
    {
        return blockParallel
    }

    let rowCount = queries.dim(0) * queries.dim(1) * queries.dim(2)
    let threadgroupWidth = turboQuantOnlineFusedThreadgroupWidth(
        minimum: max(headDimension, kernelProfile.fusedDecodeThreadgroupWidth)
    )
    let causal: Bool
    switch mask {
    case .causal:
        causal = true
    default:
        causal = false
    }
    let outputShape = [queries.dim(0), queries.dim(1), queries.dim(2), queries.dim(3)]
    let template: [(String, any KernelTemplateArg)] = [
        ("BATCH_SIZE", batchSize),
        ("KV_HEADS", kvHeadCount),
        ("QUERY_HEADS", queries.dim(1)),
        ("QUERY_LENGTH", queries.dim(2)),
        ("HEAD_DIM", headDimension),
        ("KEY_GROUP_SIZE", keyGroupSize),
        ("KEY_GROUPS_PER_VECTOR", headDimension / keyGroupSize),
        ("KEY_PACKED_WORDS_PER_VECTOR", headDimension / 4),
        ("KEY_CAPACITY", keyWeight.dim(2)),
        ("CAPACITY", valueCapacity),
        ("THREADS_PER_ROW", threadgroupWidth),
        ("OUTPUT_DTYPE", resolvedOutputDType),
        ("DO_CAUSAL", causal),
    ]

    return TurboQuantMetalKernels.hybridAffineK8DecodedValueFusedAttention(
        [
            queries,
            keyWeight,
            keyScales,
            keyBiases,
            decodedValues,
            Int32(logicalLength),
            Int32(decodedValueRingOffset),
            Int32(decodedValuePinnedPrefixLength),
            scale,
        ],
        template: template,
        grid: (rowCount * threadgroupWidth, 1, 1),
        threadGroup: (threadgroupWidth, 1, 1),
        outputShapes: [outputShape],
        outputDTypes: [resolvedOutputDType],
        stream: stream
    )[0]
}

private func turboQuantMetalHybridAffineK8DecodedValueBlockParallelFusedAttentionIfSupported(
    queries: MLXArray,
    keyWeight: MLXArray,
    keyScales: MLXArray,
    keyBiases: MLXArray,
    keyGroupSize: Int,
    decodedValues: MLXArray,
    decodedValueLogicalLength: Int,
    decodedValueRingOffset: Int,
    decodedValuePinnedPrefixLength: Int,
    scale: Float,
    outputDType: DType,
    kernelProfile: TurboQuantKernelProfile,
    stream: StreamOrDevice
) throws -> MLXArray? {
    guard queries.dim(2) == 1 else { return nil }
    guard
        let blockWidth = turboQuantResolvedBlockParallelTokenBlockSize(
            logicalLength: decodedValueLogicalLength,
            headDimension: queries.dim(3),
            queryLength: queries.dim(2),
            kernelProfile: kernelProfile,
            requestedBlockParallelTokenBlockSize: nil
        )
    else {
        return nil
    }
    let activeBlockCount = (decodedValueLogicalLength + blockWidth - 1) / blockWidth
    guard activeBlockCount > 1, activeBlockCount <= blockWidth else { return nil }

    let rowCount = queries.dim(0) * queries.dim(1) * queries.dim(2)
    let partials = TurboQuantMetalKernels.hybridAffineK8DecodedValueFusedBlockPartials(
        [
            queries,
            keyWeight,
            keyScales,
            keyBiases,
            decodedValues,
            Int32(decodedValueLogicalLength),
            Int32(decodedValueRingOffset),
            Int32(decodedValuePinnedPrefixLength),
            scale,
        ],
        template: [
            ("BATCH_SIZE", decodedValues.dim(0)),
            ("KV_HEADS", decodedValues.dim(1)),
            ("QUERY_HEADS", queries.dim(1)),
            ("QUERY_LENGTH", queries.dim(2)),
            ("HEAD_DIM", decodedValues.dim(3)),
            ("KEY_GROUP_SIZE", keyGroupSize),
            ("KEY_GROUPS_PER_VECTOR", decodedValues.dim(3) / keyGroupSize),
            ("KEY_PACKED_WORDS_PER_VECTOR", decodedValues.dim(3) / 4),
            ("KEY_CAPACITY", keyWeight.dim(2)),
            ("CAPACITY", decodedValues.dim(2)),
            ("THREADS_PER_BLOCK", blockWidth),
            ("BLOCK_TOKENS", blockWidth),
            ("BLOCK_COUNT", activeBlockCount),
            ("DO_CAUSAL", true),
        ],
        grid: (rowCount * activeBlockCount * blockWidth, 1, 1),
        threadGroup: (blockWidth, 1, 1),
        outputShapes: [
            [rowCount, activeBlockCount, 2],
            [rowCount, activeBlockCount, queries.dim(3)],
        ],
        outputDTypes: [.float32, .float32],
        stream: stream
    )

    let reduceWidth = turboQuantBlockParallelFusedThreadgroupWidth(
        minimum: max(activeBlockCount, queries.dim(3))
    )
    return TurboQuantMetalKernels.hybridDecodedValueFusedBlockReduce(
        partials,
        template: [
            ("ROW_COUNT", rowCount),
            ("BLOCK_COUNT", activeBlockCount),
            ("HEAD_DIM", queries.dim(3)),
            ("THREADS_PER_BLOCK", reduceWidth),
            ("OUTPUT_DTYPE", outputDType),
        ],
        grid: (rowCount * reduceWidth, 1, 1),
        threadGroup: (reduceWidth, 1, 1),
        outputShapes: [[queries.dim(0), queries.dim(1), queries.dim(2), queries.dim(3)]],
        outputDTypes: [outputDType],
        stream: stream
    )[0]
}

private func turboQuantMetalHybridAffineK8PolarWHTValueBlockParallelFusedAttentionIfSupported(
    queries: MLXArray,
    keyWeight: MLXArray,
    keyScales: MLXArray,
    keyBiases: MLXArray,
    keyGroupSize: Int,
    valueCode: TurboQuantPolarWHTAttentionValueCode,
    scale: Float,
    outputDType: DType,
    kernelProfile: TurboQuantKernelProfile,
    stream: StreamOrDevice
) throws -> MLXArray? {
    guard queries.dim(2) == 1 else { return nil }
    guard
        let blockWidth = turboQuantResolvedBlockParallelTokenBlockSize(
            logicalLength: valueCode.layout.logicalLength,
            headDimension: queries.dim(3),
            queryLength: queries.dim(2),
            kernelProfile: kernelProfile,
            requestedBlockParallelTokenBlockSize: nil
        )
    else {
        return nil
    }
    let activeBlockCount = (valueCode.layout.logicalLength + blockWidth - 1) / blockWidth
    guard activeBlockCount > 1, activeBlockCount <= blockWidth else { return nil }

    let rowCount = queries.dim(0) * queries.dim(1) * queries.dim(2)
    let queryHeadRepeats = queries.dim(1) / valueCode.layout.kvHeadCount
    let useGroupedQueryKernel =
        kernelProfile == .macAppleSilicon
        && queries.dim(1) % valueCode.layout.kvHeadCount == 0
        && queryHeadRepeats == 2
    let template: [(String, any KernelTemplateArg)] = [
        ("BATCH_SIZE", valueCode.layout.batchSize),
        ("KV_HEADS", valueCode.layout.kvHeadCount),
        ("QUERY_HEADS", queries.dim(1)),
        ("QUERY_LENGTH", queries.dim(2)),
        ("CAPACITY", valueCode.layout.capacity),
        ("KEY_CAPACITY", keyWeight.dim(2)),
        ("HEAD_DIM", valueCode.layout.headDimension),
        ("KEY_GROUP_SIZE", keyGroupSize),
        ("KEY_GROUPS_PER_VECTOR", valueCode.layout.headDimension / keyGroupSize),
        ("KEY_PACKED_WORDS_PER_VECTOR", valueCode.layout.headDimension / 4),
        ("PACKED_WORDS_PER_VECTOR", valueCode.packedWordsPerVector),
        ("POLAR_WHT_BITS", valueCode.bits),
        ("THREADS_PER_BLOCK", blockWidth),
        ("BLOCK_TOKENS", blockWidth),
        ("BLOCK_COUNT", activeBlockCount),
        ("GQA_REPEATS", useGroupedQueryKernel ? queryHeadRepeats : 1),
        ("OUTPUT_DTYPE", outputDType),
        ("DO_CAUSAL", true),
    ]

    let partialRows =
        useGroupedQueryKernel
        ? queries.dim(0) * valueCode.layout.kvHeadCount * queries.dim(2)
        : rowCount
    let partialKernel =
        useGroupedQueryKernel
        ? TurboQuantMetalKernels.hybridAffineK8PolarWHTValueGQAFusedBlockPartials
        : TurboQuantMetalKernels.hybridAffineK8PolarWHTValueFusedBlockPartials
    let partials = partialKernel(
        [
            queries,
            keyWeight,
            keyScales,
            keyBiases,
            valueCode.packedIndices,
            valueCode.norms,
            Int32(valueCode.layout.logicalLength),
            Int32(valueCode.layout.ringOffset),
            Int32(valueCode.layout.pinnedPrefixLength),
            scale,
        ],
        template: template,
        grid: (partialRows * activeBlockCount * blockWidth, 1, 1),
        threadGroup: (blockWidth, 1, 1),
        outputShapes: [
            [rowCount, activeBlockCount, 2],
            [rowCount, activeBlockCount, queries.dim(3)],
        ],
        outputDTypes: [.float32, .float32],
        stream: stream
    )

    let reduceWidth = turboQuantBlockParallelFusedThreadgroupWidth(
        minimum: max(activeBlockCount, queries.dim(3))
    )
    return TurboQuantMetalKernels.hybridPolarWHTValueFusedBlockReduce(
        partials,
        template: [
            ("ROW_COUNT", rowCount),
            ("HEAD_DIM", queries.dim(3)),
            ("BLOCK_COUNT", activeBlockCount),
            ("THREADS_PER_BLOCK", reduceWidth),
            ("OUTPUT_DTYPE", outputDType),
        ] + metalTemplateSeedWords(prefix: "VALUE_SEED", value: valueCode.seed),
        grid: (rowCount * reduceWidth, 1, 1),
        threadGroup: (reduceWidth, 1, 1),
        outputShapes: [[queries.dim(0), queries.dim(1), queries.dim(2), queries.dim(3)]],
        outputDTypes: [outputDType],
        stream: stream
    )[0]
}

private func turboQuantResolvedSparseValueThreshold(
    requestedThreshold: Float?,
    queries: MLXArray,
    keyCode: TurboQuantAttentionCode,
    valueCode: TurboQuantAttentionCode,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    sinks: MLXArray?
) -> Float? {
    guard let threshold = requestedThreshold, threshold > 0 else { return nil }
    guard queries.dim(2) == 1, sinks == nil else { return nil }
    guard keyCode.layout.headDimension == valueCode.layout.headDimension,
        keyCode.layout.logicalLength == valueCode.layout.logicalLength
    else {
        return nil
    }
    switch mask {
    case .none, .causal:
        return threshold
    case .array, .arrays:
        return nil
    }
}

public func turboQuantNativeMLXAttentionEnabled() -> Bool {
    let environment = ProcessInfo.processInfo.environment
    for name in ["MLX_TURBOQUANT_NATIVE_ATTENTION", "TURBOQUANT_NATIVE_MLX_ATTENTION"] {
        guard let value = environment[name]?.lowercased() else { continue }
        if ["1", "true", "yes", "on"].contains(value) {
            return true
        }
        if ["0", "false", "no", "off", "disabled"].contains(value) {
            return false
        }
    }
    return true
}

private struct TurboQuantNativeAttentionSelfTestResult: Sendable {
    var nativeCompressedAttention: Bool
    var nativeSparseVSupport: Bool
    var nativeDiagnosticsSupport: Bool
    var nativeBackendVersion: Int?
    var nativeSegmentedAttentionBackend: TurboQuantNativeSegmentedAttentionBackend
    var nativeFallbackReason: String?
}

private func turboQuantNativeCCodec(
    _ codec: TurboQuantNativeSegmentedAttentionCodec
) -> mlx_fast_turbo_quant_segmented_attention_codec {
    switch codec {
    case .polarQJL:
        MLX_FAST_TURBO_QUANT_SEGMENTED_ATTENTION_CODEC_POLAR_QJL
    case .polarWHT:
        MLX_FAST_TURBO_QUANT_SEGMENTED_ATTENTION_CODEC_POLAR_WHT
    case .hybridK8PolarWHTValue:
        MLX_FAST_TURBO_QUANT_SEGMENTED_ATTENTION_CODEC_HYBRID_K8_POLAR_WHT_VALUE
    }
}

public func turboQuantNativeSegmentedAttentionBackend(
    codec: TurboQuantNativeSegmentedAttentionCodec = .polarQJL,
    allowExperimentalJIT: Bool = turboQuantNativeMLXAttentionEnabled(),
    stream: StreamOrDevice = .gpu
) -> TurboQuantNativeSegmentedAttentionBackend {
    if codec == .polarWHT || codec == .hybridK8PolarWHTValue {
        guard allowExperimentalJIT else { return .unavailable }
        let capabilities = TurboQuantRuntimeProbe.shared.result().kernelCapabilities
        switch codec {
        case .polarWHT:
            return capabilities.polarWHTAttention ? .experimentalJIT : .unavailable
        case .hybridK8PolarWHTValue:
            return capabilities.hybridK8PolarWHTValueAttention ? .experimentalJIT : .unavailable
        case .polarQJL:
            break
        }
    }
    var backend = MLX_FAST_TURBO_QUANT_SEGMENTED_ATTENTION_UNAVAILABLE
    let status = mlx_fast_turbo_quant_segmented_attention_get_backend_for_codec(
        &backend,
        turboQuantNativeCCodec(codec),
        allowExperimentalJIT,
        stream.ctx
    )
    guard status == MLX_STATUS_SUCCESS else {
        return .unavailable
    }
    return TurboQuantNativeSegmentedAttentionBackend(rawValue: Int32(backend.rawValue))
        ?? .unavailable
}

public func turboQuantNativeSegmentedAttentionIsAvailable(
    codec: TurboQuantNativeSegmentedAttentionCodec = .polarQJL,
    allowExperimentalJIT: Bool = turboQuantNativeMLXAttentionEnabled(),
    stream: StreamOrDevice = .gpu
) -> Bool {
    if codec == .polarWHT || codec == .hybridK8PolarWHTValue {
        return turboQuantNativeSegmentedAttentionBackend(
            codec: codec,
            allowExperimentalJIT: allowExperimentalJIT,
            stream: stream
        ) != .unavailable
    }
    var available = false
    let status = mlx_fast_turbo_quant_segmented_attention_is_available_for_codec(
        &available,
        turboQuantNativeCCodec(codec),
        allowExperimentalJIT,
        stream.ctx
    )
    guard status == MLX_STATUS_SUCCESS else {
        return false
    }
    return available
}

private final class TurboQuantNativeAttentionSelfTest: @unchecked Sendable {
    static var result: TurboQuantNativeAttentionSelfTestResult {
        shared.result()
    }

    private static let shared = TurboQuantNativeAttentionSelfTest()

    private let lock = NSLock()
    private var cachedResult: TurboQuantNativeAttentionSelfTestResult?

    private init() {}

    private func result() -> TurboQuantNativeAttentionSelfTestResult {
        lock.lock()
        if let cachedResult {
            lock.unlock()
            return cachedResult
        }
        lock.unlock()

        let result = run()

        lock.lock()
        cachedResult = result
        lock.unlock()
        return result
    }

    private func run() -> TurboQuantNativeAttentionSelfTestResult {
        func failed(_ reason: String) -> TurboQuantNativeAttentionSelfTestResult {
            TurboQuantNativeAttentionSelfTestResult(
                nativeCompressedAttention: false,
                nativeSparseVSupport: false,
                nativeDiagnosticsSupport: false,
                nativeBackendVersion: nil,
                nativeSegmentedAttentionBackend: .unavailable,
                nativeFallbackReason: reason
            )
        }

        do {
            let backend = turboQuantNativeSegmentedAttentionBackend(allowExperimentalJIT: true)
            guard backend != .unavailable else {
                return failed("native MLX compressed attention backend probe is unavailable")
            }
            let tokenCount = 16
            let headDimension = 64
            let queryHeadCount = 4
            let keys = MLXArray(
                makeProbeValues(
                    count: tokenCount * headDimension,
                    sinScale: 0.031,
                    sinWeight: 0.2,
                    cosScale: 0.017,
                    cosWeight: 0.1
                ),
                [1, 1, tokenCount, headDimension]
            )
            let values = MLXArray(
                makeProbeValues(
                    count: tokenCount * headDimension,
                    sinScale: 0.041,
                    sinWeight: -0.07,
                    cosScale: 0.023,
                    cosWeight: 0.3
                ),
                [1, 1, tokenCount, headDimension]
            )
            let queries = MLXArray(
                makeProbeValues(
                    count: queryHeadCount * headDimension,
                    sinScale: 0.071,
                    sinWeight: 0.15,
                    cosScale: 0,
                    cosWeight: 0
                ),
                [1, queryHeadCount, 1, headDimension]
            )
            let keyCode = try turboQuantMetalEncodeAttention(
                keys,
                configuration: TurboQuantConfiguration(
                    preset: .turbo3_5,
                    role: .key,
                    groupSize: 64,
                    backend: .metalPolarQJL,
                    seed: 0xA77E_0000_0000_0101
                )
            )
            let valueCode = try turboQuantMetalEncodeAttention(
                values,
                configuration: TurboQuantConfiguration(
                    preset: .turbo3_5,
                    role: .value,
                    groupSize: 64,
                    backend: .metalPolarQJL,
                    seed: 0xA77E_0000_0000_0102,
                    valueBits: 4
                )
            )
            let scale = 1 / sqrt(Float(headDimension))
            let exact = try turboQuantNativeScaledDotProductAttentionWithDiagnostics(
                queries: queries,
                keyCode: keyCode,
                valueCode: valueCode,
                options: TurboQuantNativeAttentionOptions(
                    scale: scale,
                    causal: true,
                    diagnostics: true
                )
            )
            eval(exact.output)
            guard exact.output.shape == [1, queryHeadCount, 1, headDimension],
                isFinite(exact.output).all().item(Bool.self),
                let diagnostics = exact.diagnostics,
                diagnostics.fallbackCode == 0
            else {
                return failed(
                    "native MLX compressed attention self-test returned invalid output")
            }

            let sparseAvailable: Bool
            do {
                let sparse = try turboQuantNativeScaledDotProductAttentionWithDiagnostics(
                    queries: queries,
                    keyCode: keyCode,
                    valueCode: valueCode,
                    options: TurboQuantNativeAttentionOptions(
                        scale: scale,
                        causal: true,
                        sparseVThreshold: 1e-6,
                        diagnostics: true
                    )
                )
                eval(sparse.output)
                sparseAvailable = sparse.output.shape == exact.output.shape
                    && isFinite(sparse.output).all().item(Bool.self)
                    && (sparse.diagnostics?.fallbackCode ?? 1) == 0
            } catch {
                sparseAvailable = false
            }

            return TurboQuantNativeAttentionSelfTestResult(
                nativeCompressedAttention: true,
                nativeSparseVSupport: sparseAvailable,
                nativeDiagnosticsSupport: true,
                nativeBackendVersion: diagnostics.backendVersion,
                nativeSegmentedAttentionBackend: backend,
                nativeFallbackReason: nil
            )
        } catch {
            return failed("native MLX compressed attention self-test failed: \(error)")
        }
    }

    private func makeProbeValues(
        count: Int,
        sinScale: Double,
        sinWeight: Double,
        cosScale: Double,
        cosWeight: Double
    ) -> [Float] {
        var values: [Float] = []
        values.reserveCapacity(count)
        for index in 0 ..< count {
            let position = Double(index)
            let sinPart = Foundation.sin(position * sinScale) * sinWeight
            let cosPart = cosScale == 0 ? 0 : Foundation.cos(position * cosScale) * cosWeight
            values.append(Float(sinPart + cosPart))
        }
        return values
    }
}

private struct TurboQuantPolarWHTMetalSelfTestResult: Sendable {
    var codecPassed: Bool
    var attentionPassed: Bool
    var hybridK8PolarWHTValueAttentionPassed: Bool
    var failureReason: String?

    static func failed(_ reason: String) -> TurboQuantPolarWHTMetalSelfTestResult {
        TurboQuantPolarWHTMetalSelfTestResult(
            codecPassed: false,
            attentionPassed: false,
            hybridK8PolarWHTValueAttentionPassed: false,
            failureReason: reason
        )
    }
}

private func turboQuantRunPolarWHTMetalSelfTest()
    -> TurboQuantPolarWHTMetalSelfTestResult
{
    guard metalRuntimeAvailable() else {
        return .failed("Metal runtime is unavailable for PolarWHT self-test")
    }
    do {
        let tokenCount = 4
        let capacity = 6
        let headDimension = 64
        let kvHeadCount = 2
        let queryHeadCount = 4
        let queryLength = 2
        let keyValues: [Float] =
            (0 ..< (kvHeadCount * tokenCount * headDimension)).map { index in
                let position = Double(index)
                return Float(0.29 * sin(position * 0.031) + 0.11 * cos(position * 0.017))
            }
        let valueValues: [Float] =
            (0 ..< (kvHeadCount * tokenCount * headDimension)).map { index in
                let position = Double(index)
                return Float(0.23 * cos(position * 0.043) - 0.19 * sin(position * 0.029))
            }
        let queryValues: [Float] =
            (0 ..< (queryHeadCount * queryLength * headDimension)).map { index in
                let position = Double(index)
                return Float(0.17 * sin(position * 0.071) + 0.07 * cos(position * 0.019))
            }
        let keys = MLXArray(keyValues, [1, kvHeadCount, tokenCount, headDimension])
        let values = MLXArray(valueValues, [1, kvHeadCount, tokenCount, headDimension])
        let queries = MLXArray(queryValues, [1, queryHeadCount, queryLength, headDimension])
            .contiguous(stream: .gpu)
        let keyCode = try turboQuantMetalPolarWHTEncodeAttentionValues(
            keys,
            bits: 3,
            seed: 0xBADC_0FFE_0000_0101,
            capacity: capacity,
            logicalLength: tokenCount,
            ringOffset: 1,
            pinnedPrefixLength: 1
        )
        let valueCode = try turboQuantMetalPolarWHTEncodeAttentionValues(
            values,
            bits: 3,
            seed: 0xBADC_0FFE_0000_0102,
            capacity: capacity,
            logicalLength: tokenCount,
            ringOffset: 1,
            pinnedPrefixLength: 1
        )
        let decodedValues = try turboQuantMetalPolarWHTDecodeAttentionValues(
            valueCode,
            outputDType: .float32
        )
        let referenceDecodedValues = try turboQuantPolarWHTReferenceDecodeAttentionValues(
            valueCode
        )
        eval(decodedValues, referenceDecodedValues)
        let decodedRelativeMSE = turboQuantRelativeMSE(
            referenceDecodedValues.asArray(Float.self),
            decodedValues.asArray(Float.self)
        )
        let codecPassed =
            keyCode.packedIndices.shape == [1, kvHeadCount, capacity, keyCode.packedWordsPerVector]
            && valueCode.packedIndices.shape == [
                1, kvHeadCount, capacity, valueCode.packedWordsPerVector,
            ]
            && decodedValues.shape == values.shape
            && decodedValues.asArray(Float.self).allSatisfy(\.isFinite)
            && decodedRelativeMSE < 1e-6

        let scale = 1 / sqrt(Float(headDimension))
        let scores = try turboQuantMetalPolarWHTQK(
            queries: queries,
            keyCode: keyCode,
            scale: scale,
            mask: .causal
        )
        let weights = softmax(scores.asType(.float32), axis: -1)
        let av = try turboQuantMetalPolarWHTAV(
            attentionWeights: weights,
            valueCode: valueCode,
            outputDType: .float32
        )
        let fused = try turboQuantMetalPolarWHTScaledDotProductAttention(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            scale: scale,
            mask: .causal,
            outputDType: .float32
        )
        eval(scores, av, fused)
        let avValues = av.asArray(Float.self)
        let fusedValues = fused.asArray(Float.self)
        let fusedDelta = zip(avValues, fusedValues).reduce(Float(0)) { current, pair in
            Swift.max(current, Swift.abs(pair.0 - pair.1))
        }
        let attentionPassed =
            scores.shape == [1, queryHeadCount, queryLength, tokenCount]
            && av.shape == [1, queryHeadCount, queryLength, headDimension]
            && fused.shape == av.shape
            && scores.asArray(Float.self).allSatisfy(\.isFinite)
            && avValues.allSatisfy(\.isFinite)
            && fusedValues.allSatisfy(\.isFinite)
            && fusedDelta < 1e-4

        let qjlKeyCode = try turboQuantMetalEncodeAttention(
            keys,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .key,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 0xBADC_0FFE_0000_0201
            ),
            capacity: capacity,
            logicalLength: tokenCount,
            ringOffset: 1,
            pinnedPrefixLength: 1
        )
        let hybrid = try turboQuantMetalHybridPolarWHTValueScaledDotProductAttention(
            queries: queries,
            keyCode: qjlKeyCode,
            valueCode: valueCode,
            scale: scale,
            mask: .causal,
            outputDType: .float32
        )
        eval(hybrid)
        let hybridPassed =
            hybrid.shape == av.shape
            && hybrid.asArray(Float.self).allSatisfy(\.isFinite)

        let passed = codecPassed && attentionPassed
        return TurboQuantPolarWHTMetalSelfTestResult(
            codecPassed: codecPassed,
            attentionPassed: attentionPassed,
            hybridK8PolarWHTValueAttentionPassed: hybridPassed,
            failureReason: passed
                ? nil
                : "PolarWHT Metal self-test failed: codec=\(codecPassed), attention=\(attentionPassed), hybridValue=\(hybridPassed), decodeRelativeMSE=\(decodedRelativeMSE), fusedDelta=\(fusedDelta)."
        )
    } catch {
        return .failed("PolarWHT Metal self-test failed: \(error)")
    }
}

private func turboQuantNativePresetCode(_ preset: TurboQuantPreset) -> Int32 {
    switch preset {
    case .turbo2_5:
        25
    case .turbo3_5:
        35
    case .turbo4:
        40
    case .turbo4v2:
        42
    case .turbo8:
        80
    }
}

private func turboQuantNativeLayoutDescriptor(
    _ layout: TurboQuantAttentionLayout
) -> mlx_fast_turbo_quant_attention_layout_descriptor {
    mlx_fast_turbo_quant_attention_layout_descriptor(
        layout_version: Int32(layout.layoutVersion),
        batch_size: Int32(layout.batchSize),
        kv_head_count: Int32(layout.kvHeadCount),
        capacity: Int32(layout.capacity),
        logical_length: Int32(layout.logicalLength),
        ring_offset: Int32(layout.ringOffset),
        pinned_prefix_length: Int32(layout.pinnedPrefixLength),
        head_dimension: Int32(layout.headDimension),
        groups_per_vector: Int32(layout.groupsPerVector),
        magnitude_words_per_group: Int32(layout.magnitudeWordsPerGroup),
        bitset_words_per_group: Int32(layout.bitsetWordsPerGroup)
    )
}

private func turboQuantNativePrecisionDescriptor(
    keyCode: TurboQuantAttentionCode,
    valueCode: TurboQuantAttentionCode
) -> mlx_fast_turbo_quant_precision_policy_descriptor {
    let keyBaseBits = max(1, keyCode.preset.baseMagnitudeBits - 1)
    let keyHighBits = max(keyBaseBits, keyCode.preset.highMagnitudeBits - 1)
    let highFraction = mixedPrecisionHighFraction(preset: keyCode.preset)
    return mlx_fast_turbo_quant_precision_policy_descriptor(
        preset: turboQuantNativePresetCode(keyCode.preset),
        group_size: Int32(keyCode.groupSize),
        key_base_bits: Int32(keyBaseBits),
        key_high_bits: Int32(keyHighBits),
        high_precision_numerator: Int32(highFraction.numerator),
        high_precision_denominator: Int32(highFraction.denominator),
        value_bits: Int32(valueCode.valueBits),
        key_scales_per_group: Int32(keyCode.scalesPerGroup),
        value_scales_per_group: Int32(valueCode.scalesPerGroup),
        value_magnitude_words_per_group: Int32(valueCode.layout.magnitudeWordsPerGroup),
        key_seed: keyCode.seed,
        value_seed: valueCode.seed
    )
}

private func turboQuantNativeCOptions(
    _ options: TurboQuantNativeAttentionOptions
) -> mlx_fast_turbo_quant_attention_options {
    mlx_fast_turbo_quant_attention_options(
        scale: options.scale,
        causal: options.causal,
        split_k_blocks: Int32(options.splitKBlockCount),
        sparse_v_threshold: options.sparseVThreshold,
        sparse_v_selection_mode: options.sparseVSelectionMode.rawValue,
        sparse_v_top_k: Int32(options.sparseVTopK),
        sparse_v_cumulative_mass: options.sparseVCumulativeMass,
        sparse_v_max_top_k: Int32(options.sparseVMaxTopK),
        sparse_v_recent_tokens: Int32(options.sparseVRecentTokens),
        sparse_v_candidate_pages: Int32(options.sparseVCandidatePages),
        diagnostics: options.diagnostics,
        backend_version: Int32(options.backendVersion)
    )
}

public func turboQuantNativeScaledDotProductAttention(
    queries: MLXArray,
    keyCode: TurboQuantAttentionCode,
    valueCode: TurboQuantAttentionCode,
    options: TurboQuantNativeAttentionOptions,
    keyPageSummary: MLXArray? = nil,
    keyCandidateSketch: MLXArray? = nil,
    stream: StreamOrDevice = .gpu
) throws -> MLXArray {
    let result = try turboQuantNativeSegmentedAttentionWithDiagnostics(
        queries: queries,
        keyCode: keyCode,
        valueCode: valueCode,
        options: options,
        keyPageSummary: keyPageSummary,
        keyCandidateSketch: keyCandidateSketch,
        stream: stream
    )
    return result.output
}

public func turboQuantNativeScaledDotProductAttentionWithDiagnostics(
    queries: MLXArray,
    keyCode: TurboQuantAttentionCode,
    valueCode: TurboQuantAttentionCode,
    options: TurboQuantNativeAttentionOptions,
    keyPageSummary: MLXArray? = nil,
    keyCandidateSketch: MLXArray? = nil,
    stream: StreamOrDevice = .gpu
) throws -> TurboQuantNativeScaledDotProductAttentionResult {
    try turboQuantNativeSegmentedAttentionWithDiagnostics(
        queries: queries,
        keyCode: keyCode,
        valueCode: valueCode,
        options: options,
        keyPageSummary: keyPageSummary,
        keyCandidateSketch: keyCandidateSketch,
        stream: stream
    )
}

public func turboQuantNativeSegmentedAttention(
    queries: MLXArray,
    keyCode: TurboQuantAttentionCode,
    valueCode: TurboQuantAttentionCode,
    options: TurboQuantNativeAttentionOptions,
    keyPageSummary: MLXArray? = nil,
    keyCandidateSketch: MLXArray? = nil,
    stream: StreamOrDevice = .gpu
) throws -> MLXArray {
    let result = try turboQuantNativeSegmentedAttentionWithDiagnostics(
        queries: queries,
        keyCode: keyCode,
        valueCode: valueCode,
        options: options,
        keyPageSummary: keyPageSummary,
        keyCandidateSketch: keyCandidateSketch,
        stream: stream
    )
    return result.output
}

public func turboQuantNativeSegmentedAttentionWithDiagnostics(
    queries: MLXArray,
    keyCode: TurboQuantAttentionCode,
    valueCode: TurboQuantAttentionCode,
    options: TurboQuantNativeAttentionOptions,
    keyPageSummary: MLXArray? = nil,
    keyCandidateSketch: MLXArray? = nil,
    stream: StreamOrDevice = .gpu
) throws -> TurboQuantNativeSegmentedAttentionResult {
    try validateAttentionPair(keyCode: keyCode, valueCode: valueCode, allowTileTransposedV7: true)
    try validateAttentionQuery(queries, code: keyCode)
    try validateTurboQuantAttentionCode(keyCode, expectedRole: .key, allowTileTransposedV7: true)
    try validateTurboQuantAttentionCode(valueCode, expectedRole: .value, allowTileTransposedV7: true)
    try validateAttentionCodeStorage(keyCode, allowTileTransposedV7: true)
    try validateAttentionCodeStorage(valueCode, allowTileTransposedV7: true)
    guard keyCode.layout.layoutVersion == valueCode.layout.layoutVersion,
        keyCode.layout.batchSize == valueCode.layout.batchSize,
        keyCode.layout.kvHeadCount == valueCode.layout.kvHeadCount,
        keyCode.layout.capacity == valueCode.layout.capacity,
        keyCode.layout.logicalLength == valueCode.layout.logicalLength,
        keyCode.layout.ringOffset == valueCode.layout.ringOffset,
        keyCode.layout.pinnedPrefixLength == valueCode.layout.pinnedPrefixLength,
        keyCode.layout.headDimension == valueCode.layout.headDimension,
        keyCode.layout.groupsPerVector == valueCode.layout.groupsPerVector,
        keyCode.layout.bitsetWordsPerGroup == valueCode.layout.bitsetWordsPerGroup
    else {
        throw TurboQuantError.invalidMetalConfiguration(
            "native MLX compressed attention requires aligned K/V layouts")
    }

    let layout = turboQuantNativeLayoutDescriptor(keyCode.layout)
    let precision = turboQuantNativePrecisionDescriptor(keyCode: keyCode, valueCode: valueCode)
    var cOptions = turboQuantNativeCOptions(options)
    cOptions.diagnostics = options.diagnostics

    return try withError { error in
        if options.diagnostics {
            var vector = mlx_vector_array_new()
            defer { mlx_vector_array_free(vector) }
            let status: mlx_status
            if let keyCandidateSketch {
                status =
                    mlx_fast_turbo_quant_segmented_attention_with_candidate_sketches_and_diagnostics(
                        &vector,
                        queries.ctx,
                        keyCode.packedMagnitudes.ctx,
                        keyCode.signs.ctx,
                        keyCode.highPrecisionMask.ctx,
                        keyCode.residualSigns.ctx,
                        keyCode.scales.ctx,
                        valueCode.packedMagnitudes.ctx,
                        valueCode.signs.ctx,
                        valueCode.highPrecisionMask.ctx,
                        valueCode.residualSigns.ctx,
                        valueCode.scales.ctx,
                        keyCandidateSketch.ctx,
                        layout,
                        precision,
                        cOptions,
                        stream.ctx
                    )
            } else if let keyPageSummary {
                status =
                    mlx_fast_turbo_quant_segmented_attention_with_page_summaries_and_diagnostics(
                        &vector,
                        queries.ctx,
                        keyCode.packedMagnitudes.ctx,
                        keyCode.signs.ctx,
                        keyCode.highPrecisionMask.ctx,
                        keyCode.residualSigns.ctx,
                        keyCode.scales.ctx,
                        valueCode.packedMagnitudes.ctx,
                        valueCode.signs.ctx,
                        valueCode.highPrecisionMask.ctx,
                        valueCode.residualSigns.ctx,
                        valueCode.scales.ctx,
                        keyPageSummary.ctx,
                        layout,
                        precision,
                        cOptions,
                        stream.ctx
                    )
            } else {
                status = mlx_fast_turbo_quant_segmented_attention_with_diagnostics(
                    &vector,
                    queries.ctx,
                    keyCode.packedMagnitudes.ctx,
                    keyCode.signs.ctx,
                    keyCode.highPrecisionMask.ctx,
                    keyCode.residualSigns.ctx,
                    keyCode.scales.ctx,
                    valueCode.packedMagnitudes.ctx,
                    valueCode.signs.ctx,
                    valueCode.highPrecisionMask.ctx,
                    valueCode.residualSigns.ctx,
                    valueCode.scales.ctx,
                    layout,
                    precision,
                    cOptions,
                    stream.ctx
                )
            }
            if status != MLX_STATUS_SUCCESS {
                try error.check()
                throw TurboQuantError.unsupportedBackend(
                    .metalPolarQJL,
                    "native MLX compressed attention failed")
            }
            var output = mlx_array_new()
            var diagnostics = mlx_array_new()
            mlx_vector_array_get(&output, vector, 0)
            mlx_vector_array_get(&diagnostics, vector, 1)
            let diagnosticsArray = MLXArray(diagnostics)
            eval(diagnosticsArray)
            return TurboQuantNativeScaledDotProductAttentionResult(
                output: MLXArray(output),
                diagnostics: TurboQuantNativeAttentionDiagnostics(
                    values: diagnosticsArray.asArray(Int32.self)
                )
            )
        }

        var output = mlx_array_new()
        let status: mlx_status
        if let keyCandidateSketch {
            status = mlx_fast_turbo_quant_segmented_attention_with_candidate_sketches(
                &output,
                queries.ctx,
                keyCode.packedMagnitudes.ctx,
                keyCode.signs.ctx,
                keyCode.highPrecisionMask.ctx,
                keyCode.residualSigns.ctx,
                keyCode.scales.ctx,
                valueCode.packedMagnitudes.ctx,
                valueCode.signs.ctx,
                valueCode.highPrecisionMask.ctx,
                valueCode.residualSigns.ctx,
                valueCode.scales.ctx,
                keyCandidateSketch.ctx,
                layout,
                precision,
                cOptions,
                stream.ctx
            )
        } else if let keyPageSummary {
            status = mlx_fast_turbo_quant_segmented_attention_with_page_summaries(
                &output,
                queries.ctx,
                keyCode.packedMagnitudes.ctx,
                keyCode.signs.ctx,
                keyCode.highPrecisionMask.ctx,
                keyCode.residualSigns.ctx,
                keyCode.scales.ctx,
                valueCode.packedMagnitudes.ctx,
                valueCode.signs.ctx,
                valueCode.highPrecisionMask.ctx,
                valueCode.residualSigns.ctx,
                valueCode.scales.ctx,
                keyPageSummary.ctx,
                layout,
                precision,
                cOptions,
                stream.ctx
            )
        } else {
            status = mlx_fast_turbo_quant_segmented_attention(
                &output,
                queries.ctx,
                keyCode.packedMagnitudes.ctx,
                keyCode.signs.ctx,
                keyCode.highPrecisionMask.ctx,
                keyCode.residualSigns.ctx,
                keyCode.scales.ctx,
                valueCode.packedMagnitudes.ctx,
                valueCode.signs.ctx,
                valueCode.highPrecisionMask.ctx,
                valueCode.residualSigns.ctx,
                valueCode.scales.ctx,
                layout,
                precision,
                cOptions,
                stream.ctx
            )
        }
        if status != MLX_STATUS_SUCCESS {
            try error.check()
            throw TurboQuantError.unsupportedBackend(
                .metalPolarQJL,
                "native MLX compressed attention failed")
        }
        return TurboQuantNativeScaledDotProductAttentionResult(output: MLXArray(output))
    }
}

public func turboQuantMetalScaledDotProductAttention(
    queries: MLXArray,
    keyCode: TurboQuantAttentionCode,
    valueCode: TurboQuantAttentionCode,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    sinks: MLXArray? = nil,
    preferOnlineFused: Bool = true,
    memoryBudgetBytes: Int? = nil,
    fallbackState: TurboQuantAttentionFallbackState = .none,
    kernelProfile: TurboQuantKernelProfile? = nil,
    blockParallelTokenBlockSize: Int? = nil,
    sparseVThreshold: Float? = nil,
    stream: StreamOrDevice = .gpu
) throws -> MLXArray {
    TurboQuantHostProbe.shared.recordAttentionCall()
    try validateAttentionPair(keyCode: keyCode, valueCode: valueCode, allowTileTransposedV7: true)
    try validateAttentionQuery(queries, code: keyCode)
    try validateTurboQuantAttentionCode(keyCode, expectedRole: .key, allowTileTransposedV7: true)
    try validateTurboQuantAttentionCode(
        valueCode, expectedRole: .value, allowTileTransposedV7: true)
    try validateAttentionMask(
        mask,
        scoreShape: [
            queries.dim(0), queries.dim(1), queries.dim(2), keyCode.layout.logicalLength,
        ]
    )
    try validateAttentionSinks(sinks, queryHeadCount: queries.dim(1))
    try requireTurboQuantMetalAttention()

    let attentionCapabilities =
        TurboQuantRuntimeProbe.shared.isRunningSelfTest()
        ? TurboQuantAttentionCapabilities(
            encode: true,
            decode: true,
            qk: true,
            av: true,
            onlineFused: true,
            bfloatOutput: true
        )
        : TurboQuantKernelAvailability.current.attentionCapabilities
    let resolvedSparseVThreshold = turboQuantResolvedSparseValueThreshold(
        requestedThreshold: sparseVThreshold,
        queries: queries,
        keyCode: keyCode,
        valueCode: valueCode,
        mask: mask,
        sinks: sinks
    )
    let decision = try turboQuantAttentionDecision(
        request: TurboQuantAttentionRequest(
            queryShape: queries.shape,
            keyLayout: keyCode.layout,
            valueLayout: valueCode.layout,
            queryDType: queries.dtype,
            outputDType: queries.dtype,
            maskKind: turboQuantAttentionMaskKind(mask),
            hasSinks: sinks != nil,
            preferOnlineFused: preferOnlineFused && resolvedSparseVThreshold == nil,
            memoryBudgetBytes: memoryBudgetBytes,
            fallbackState: fallbackState,
            sparseVThreshold: resolvedSparseVThreshold
        ),
        capabilities: attentionCapabilities
    )

    if decision.selectedPath == .nativeMLXCompressed {
        return try turboQuantNativeScaledDotProductAttention(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            options: TurboQuantNativeAttentionOptions(
                scale: scale,
                causal: mask.isCausal,
                sparseVThreshold: resolvedSparseVThreshold ?? 0,
                diagnostics: false,
                backendVersion: attentionCapabilities.nativeBackendVersion
                    ?? TurboQuantNativeAttentionOptions.backendVersion
            ),
            stream: stream
        )
    }

    if decision.selectedPath == .onlineFused || decision.selectedPath == .tiledOnlineFused {
        return try turboQuantMetalOnlineFusedAttention(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            scale: scale,
            mask: mask,
            kernelProfile: kernelProfile
                ?? TurboQuantRuntimeProbe.shared.selectedKernelProfileWithoutRunningProbe(),
            blockParallelTokenBlockSize: blockParallelTokenBlockSize,
            outputDType: decision.outputDType,
            stateAlreadyValidated: true,
            stream: stream
        )
    }

    guard decision.selectedPath == .twoStageCompressed else {
        throw TurboQuantError.unsupportedBackend(
            .metalPolarQJL,
            "Selected TurboQuant attention path \(decision.selectedPath.rawValue) requires a higher-level fallback."
        )
    }

    let scores = try turboQuantMetalQK(
        queries: queries,
        keyCode: keyCode,
        scale: scale,
        mask: mask,
        stream: stream
    )
    var logits = scores.asType(.float32)
    logits = try prependAttentionSinks(
        logits,
        sinks: sinks,
        queryHeadCount: queries.dim(1),
        stream: stream
    )
    var weights = softmax(logits, axis: -1, stream: stream)
    if sinks != nil {
        weights = weights[.ellipsis, 1...].contiguous(stream: stream)
    }
    if let threshold = resolvedSparseVThreshold {
        weights = MLX.where(
            weights .>= threshold,
            weights,
            MLXArray.zeros(like: weights, stream: stream),
            stream: stream
        )
    }
    return try turboQuantMetalAV(
        attentionWeights: weights,
        valueCode: valueCode,
        outputDType: queries.dtype,
        stream: stream
    )
}

public func turboQuantMetalScaledDotProductAttentionWithDiagnostics(
    queries: MLXArray,
    keyCode: TurboQuantAttentionCode,
    valueCode: TurboQuantAttentionCode,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    sinks: MLXArray? = nil,
    preferOnlineFused: Bool = true,
    memoryBudgetBytes: Int? = nil,
    fallbackState: TurboQuantAttentionFallbackState = .none,
    kernelProfile: TurboQuantKernelProfile? = nil,
    blockParallelTokenBlockSize: Int? = nil,
    sparseVThreshold: Float? = nil,
    stream: StreamOrDevice = .gpu
) throws -> TurboQuantScaledDotProductAttentionResult {
    let output = try turboQuantMetalScaledDotProductAttention(
        queries: queries,
        keyCode: keyCode,
        valueCode: valueCode,
        scale: scale,
        mask: mask,
        sinks: sinks,
        preferOnlineFused: preferOnlineFused,
        memoryBudgetBytes: memoryBudgetBytes,
        fallbackState: fallbackState,
        kernelProfile: kernelProfile,
        blockParallelTokenBlockSize: blockParallelTokenBlockSize,
        sparseVThreshold: sparseVThreshold,
        stream: stream
    )
    guard let threshold = turboQuantResolvedSparseValueThreshold(
        requestedThreshold: sparseVThreshold,
        queries: queries,
        keyCode: keyCode,
        valueCode: valueCode,
        mask: mask,
        sinks: sinks
    ) else {
        return TurboQuantScaledDotProductAttentionResult(
            output: output,
            sparseValueDiagnostics: TurboQuantSparseValueDiagnostics(enabled: false)
        )
    }
    let scores = try turboQuantMetalQK(
        queries: queries,
        keyCode: keyCode,
        scale: scale,
        mask: mask,
        stream: stream
    )
    let weights = softmax(scores.asType(.float32), axis: -1, stream: stream)
    let skipped = (weights .< threshold).asType(.int32).sum().item(Int.self)
    let retainedMassTotal = MLX.where(
        weights .>= threshold,
        weights,
        MLXArray.zeros(like: weights, stream: stream),
        stream: stream
    ).sum().item(Float.self)
    let rowCount = max(1, queries.dim(0) * queries.dim(1) * queries.dim(2))
    let diagnostics = TurboQuantSparseValueDiagnostics(
        enabled: true,
        threshold: threshold,
        skipped: skipped,
        considered: weights.size,
        retainedMass: Double(retainedMassTotal) / Double(rowCount)
    )
    return TurboQuantScaledDotProductAttentionResult(
        output: output,
        sparseValueDiagnostics: diagnostics
    )
}

private func turboQuantRepeatKVHeadsForQueries(
    _ array: MLXArray,
    queryHeadCount: Int,
    stream: StreamOrDevice
) throws -> MLXArray {
    let kvHeadCount = array.dim(1)
    guard kvHeadCount > 0, queryHeadCount % kvHeadCount == 0 else {
        throw TurboQuantError.invalidMetalConfiguration(
            "query heads must be a multiple of KV heads"
        )
    }
    let repeats = queryHeadCount / kvHeadCount
    guard repeats > 1 else { return array }
    return repeated(array, count: repeats, axis: 1, stream: stream)
}

private func turboQuantMetalSparseSegmentedScaledDotProductAttention(
    queries: MLXArray,
    rawKeys: MLXArray,
    rawValues: MLXArray,
    coldSegments: [(key: TurboQuantAttentionCode, value: TurboQuantAttentionCode)],
    scale: Float,
    threshold: Float,
    outputDType: DType,
    stream: StreamOrDevice
) throws -> MLXArray {
    var scoreParts: [MLXArray] = []
    var coldLengths: [Int] = []

    for (keyCode, valueCode) in coldSegments {
        try validateAttentionPair(keyCode: keyCode, valueCode: valueCode)
        try validateAttentionQuery(queries, code: keyCode)
        try validateAttentionCodeStorage(keyCode)
        try validateAttentionCodeStorage(valueCode)
        guard keyCode.layout.headDimension == queries.dim(3),
            valueCode.layout.headDimension == queries.dim(3)
        else {
            throw TurboQuantError.invalidMetalConfiguration(
                "compressed segmented attention head dimension must match query head dimension"
            )
        }
        let scores = try turboQuantMetalQK(
            queries: queries,
            keyCode: keyCode,
            scale: scale,
            mask: .none,
            stream: stream
        )
        scoreParts.append(scores.asType(.float32, stream: stream))
        coldLengths.append(keyCode.layout.logicalLength)
    }

    if rawKeys.dim(2) > 0 {
        let expandedRawKeys = try turboQuantRepeatKVHeadsForQueries(
            rawKeys,
            queryHeadCount: queries.dim(1),
            stream: stream
        )
        let rawScores = matmul(
            multiply(queries.asType(.float32, stream: stream), scale, stream: stream),
            expandedRawKeys.asType(.float32, stream: stream).transposed(0, 1, 3, 2, stream: stream),
            stream: stream
        )
        scoreParts.append(rawScores.asType(.float32, stream: stream))
    }

    guard !scoreParts.isEmpty else {
        throw TurboQuantError.invalidMetalConfiguration(
            "segmented TurboQuant attention requires a raw or compressed segment"
        )
    }

    let scores = scoreParts.count == 1 ? scoreParts[0] : concatenated(scoreParts, axis: -1, stream: stream)
    let weights = softmax(scores, axis: -1, stream: stream)
    var outputs: [MLXArray] = []
    var cursor = 0

    for (segmentIndex, segment) in coldSegments.enumerated() {
        let length = coldLengths[segmentIndex]
        guard length > 0 else { continue }
        let coldWeights = weights[.ellipsis, cursor ..< cursor + length]
            .contiguous(stream: stream)
        let sparseColdWeights = MLX.where(
            coldWeights .>= threshold,
            coldWeights,
            MLXArray.zeros(like: coldWeights, stream: stream),
            stream: stream
        )
        let coldOutput = try turboQuantMetalAV(
            attentionWeights: sparseColdWeights,
            valueCode: segment.value,
            outputDType: .float32,
            stream: stream
        )
        outputs.append(coldOutput.asType(.float32, stream: stream))
        cursor += length
    }

    if rawKeys.dim(2) > 0 {
        let expandedRawValues = try turboQuantRepeatKVHeadsForQueries(
            rawValues,
            queryHeadCount: queries.dim(1),
            stream: stream
        )
        let rawWeights = weights[.ellipsis, cursor...].contiguous(stream: stream)
        let rawOutput = matmul(rawWeights, expandedRawValues, stream: stream)
        outputs.append(rawOutput.asType(.float32, stream: stream))
    }

    guard var output = outputs.first else {
        throw TurboQuantError.invalidMetalConfiguration(
            "sparse segmented TurboQuant attention produced no output segments"
        )
    }
    for partial in outputs.dropFirst() {
        output = output + partial
    }
    return output.asType(outputDType, stream: stream)
}

public func turboQuantMetalSegmentedScaledDotProductAttention(
    queries: MLXArray,
    rawKeys: MLXArray,
    rawValues: MLXArray,
    coldSegments: [(key: TurboQuantAttentionCode, value: TurboQuantAttentionCode)],
    scale: Float,
    outputDType: DType = .float32,
    kernelProfile: TurboQuantKernelProfile? = nil,
    sparseVThreshold: Float? = nil,
    stream: StreamOrDevice = .gpu
) throws -> MLXArray {
    TurboQuantHostProbe.shared.recordAttentionCall()
    try requireTurboQuantMetalAttentionOutputDType(outputDType)
    try requireTurboQuantMetalAttention()
    try validateAttentionShape(queries.shape, dtype: queries.dtype, groupSize: 32)
    try validateAttentionShape(rawKeys.shape, dtype: rawKeys.dtype, groupSize: 32)
    try validateAttentionShape(rawValues.shape, dtype: rawValues.dtype, groupSize: 32)
    guard queries.dim(2) == 1 else {
        throw TurboQuantError.invalidMetalConfiguration(
            "segmented TurboQuant attention currently supports decode query length 1"
        )
    }
    guard rawKeys.dim(2) > 0 || !coldSegments.isEmpty else {
        throw TurboQuantError.invalidMetalConfiguration(
            "segmented TurboQuant attention requires a raw or compressed segment"
        )
    }
    guard queries.dim(0) == rawKeys.dim(0), rawKeys.shape == rawValues.shape else {
        throw TurboQuantError.invalidMetalConfiguration(
            "raw segmented attention keys and values must share batch, heads, tokens, and dimensions"
        )
    }
    guard queries.dim(3) == rawKeys.dim(3), queries.dim(3) == rawValues.dim(3) else {
        throw TurboQuantError.invalidMetalConfiguration(
            "raw segmented attention head dimension must match query head dimension"
        )
    }
    guard queries.dim(1) % rawKeys.dim(1) == 0 else {
        throw TurboQuantError.invalidMetalConfiguration(
            "query heads must be a multiple of raw KV heads"
        )
    }
    if let threshold = sparseVThreshold, threshold > 0, !coldSegments.isEmpty {
        return try turboQuantMetalSparseSegmentedScaledDotProductAttention(
            queries: queries,
            rawKeys: rawKeys,
            rawValues: rawValues,
            coldSegments: coldSegments,
            scale: scale,
            threshold: threshold,
            outputDType: outputDType,
            stream: stream
        )
    }

    let profile =
        kernelProfile
        ?? TurboQuantRuntimeProbe.shared.selectedKernelProfileWithoutRunningProbe()
    let rowCount = queries.dim(0) * queries.dim(1) * queries.dim(2)
    var partialStats: [MLXArray] = []
    var partialOut: [MLXArray] = []
    var totalBlocks = 0

    if rawKeys.dim(2) > 0 {
        let rawWidth = turboQuantBlockParallelFusedThreadgroupWidth(
            minimum: max(queries.dim(3), min(rawKeys.dim(2), 512))
        )
        let rawPartials = TurboQuantMetalKernels.segmentedRawAttentionStats(
            [queries, rawKeys, rawValues, scale, Int32(rawKeys.dim(2))],
            template: [
                ("BATCH_SIZE", queries.dim(0)),
                ("QUERY_HEADS", queries.dim(1)),
                ("KV_HEADS", rawKeys.dim(1)),
                ("QUERY_LENGTH", queries.dim(2)),
                ("HEAD_DIM", queries.dim(3)),
                ("THREADS_PER_BLOCK", rawWidth),
            ],
            grid: (rowCount * rawWidth, 1, 1),
            threadGroup: (rawWidth, 1, 1),
            outputShapes: [
                [rowCount, 1, 2],
                [rowCount, 1, queries.dim(3)],
            ],
            outputDTypes: [.float32, .float32],
            stream: stream
        )
        partialStats.append(rawPartials[0])
        partialOut.append(rawPartials[1])
        totalBlocks += 1
    }

    for (keyCode, valueCode) in coldSegments {
        try validateAttentionPair(keyCode: keyCode, valueCode: valueCode)
        try validateAttentionQuery(queries, code: keyCode)
        try validateAttentionCodeStorage(keyCode)
        try validateAttentionCodeStorage(valueCode)
        guard keyCode.layout.headDimension == queries.dim(3),
            valueCode.layout.headDimension == queries.dim(3)
        else {
            throw TurboQuantError.invalidMetalConfiguration(
                "compressed segmented attention head dimension must match query head dimension"
            )
        }

        let blockWidth =
            turboQuantResolvedBlockParallelTokenBlockSize(
                logicalLength: keyCode.layout.logicalLength,
                headDimension: queries.dim(3),
                queryLength: queries.dim(2),
                kernelProfile: profile,
                requestedBlockParallelTokenBlockSize: nil
            )
            ?? turboQuantBlockParallelFusedThreadgroupWidth(
                minimum: max(queries.dim(3), min(keyCode.layout.logicalLength, 512))
            )
        let activeBlockCount = max(1, (keyCode.layout.logicalLength + blockWidth - 1) / blockWidth)
        guard activeBlockCount <= blockWidth else {
            throw TurboQuantError.invalidMetalConfiguration(
                "segmented compressed attention active block count \(activeBlockCount) exceeds block width \(blockWidth)"
            )
        }
        let queryHeadRepeats = queries.dim(1) / keyCode.layout.kvHeadCount
        let useGroupedQueryKernel =
            profile == .macAppleSilicon
            && queries.dim(1) % keyCode.layout.kvHeadCount == 0
            && queryHeadRepeats > 1
            && queryHeadRepeats <= 4
        let coopActive = turboQuantCooperativeQuadDecodeActive(
            enabled: useGroupedQueryKernel,
            queryHeadRepeats: queryHeadRepeats,
            preset: keyCode.preset,
            layoutVersion: keyCode.layout.layoutVersion,
            headDim: queries.dim(3),
            groupSize: keyCode.groupSize,
            logicalLength: keyCode.layout.logicalLength)
        let partialRows =
            useGroupedQueryKernel
            ? queries.dim(0) * keyCode.layout.kvHeadCount * queries.dim(2)
            : rowCount
        let useH16 = turboQuantH16DietEnabled && useGroupedQueryKernel
        // COOPW is not ported to the H16 diet kernel (out of scope for T2.4 stage-2); the
        // H16-diet coop branch still hardcodes r<4u, so H16-coop dispatch stays gated to
        // exactly repeats==4 regardless of the widened turboQuantCooperativeQuadDecodeActive
        // guard above.
        let coopWActive = coopActive && queryHeadRepeats < 4 && !useH16
        let partialKernel =
            useH16
            ? ((coopActive && queryHeadRepeats == 4)
                ? TurboQuantMetalKernels.fusedAttentionGQABlockPartialsCoopH16_rf1
                : TurboQuantMetalKernels.fusedAttentionGQABlockPartialsH16_rf1)
            : (coopWActive
                ? TurboQuantMetalKernels.fusedAttentionGQABlockPartialsCoopW_rf1
                : (coopActive
                    ? TurboQuantMetalKernels.fusedAttentionGQABlockPartialsCoop_rf1
                    : (useGroupedQueryKernel
                        ? TurboQuantMetalKernels.fusedAttentionGQABlockPartials_rf1
                        : TurboQuantMetalKernels.fusedAttentionBlockPartials)))
        TurboQuantKernelDispatchTrace.shared.record(
            "segmented:" + (useH16
                ? ((coopActive && queryHeadRepeats == 4)
                    ? "turboquant_attention_fused_gqa_block_partials_coop_runtime_layout_rtu1_s2_h16_rf1"
                    : "turboquant_attention_fused_gqa_block_partials_runtime_layout_rtu1_s2_h16_rf1")
                : (coopWActive
                    ? "turboquant_attention_fused_gqa_block_partials_coopw_runtime_layout_rtu1_s2_rf1"
                    : (coopActive
                        ? "turboquant_attention_fused_gqa_block_partials_coop_runtime_layout_rtu1_s2_rf1"
                        : (useGroupedQueryKernel
                            ? "turboquant_attention_fused_gqa_block_partials_runtime_layout_rtu1_s2_rf1"
                            : "turboquant_attention_fused_block_partials_runtime_layout_rtu1_s2")))))
        let template =
            runtimeLayoutAttentionTemplate(
                configuration: TurboQuantConfiguration(
                    preset: keyCode.preset,
                    role: .key,
                    groupSize: keyCode.groupSize,
                    backend: .metalPolarQJL,
                    seed: keyCode.seed,
                    valueBits: valueCode.valueBits,
                    attentionLayoutVersion: keyCode.layout.layoutVersion,
                    allowExperimentalLayoutV5: keyCode.layout.isLayoutV5,
                    attentionScaleStorage: turboQuantAttentionScaleStorage(for: keyCode)
                ),
                layout: keyCode.layout,
                inputLength: keyCode.layout.logicalLength,
                outputLength: keyCode.layout.logicalLength,
                queryHeadCount: queries.dim(1),
                queryLength: queries.dim(2),
                outputDType: .float32,
                causal: false
            ) + [
                ("VALUE_MAG_WORDS_PER_GROUP", valueCode.layout.magnitudeWordsPerGroup),
                ("VALUE_SCALES_PER_GROUP", valueCode.scalesPerGroup),
                ("THREADS_PER_BLOCK", blockWidth),
                ("BLOCK_TOKENS", blockWidth),
                ("GQA_REPEATS", useGroupedQueryKernel ? queryHeadRepeats : 1),
                // H16-BUG-FIX: when useH16 is true and repeats < 4, the selected kernel is
                // the strided fusedAttentionGQABlockPartialsH16 (see partialKernel below),
                // whose coop branch (lanes_per_token == 4u) hardcodes r<4u loops over
                // query_cache rows that the shared prologue only initializes up to
                // repeat_count. Feeding LANES_PER_TOKEN=4 into that strided-H16 kernel at
                // repeats<4 would read uninitialized threadgroup memory (UB / shader-
                // validation trap). LANES_PER_TOKEN must therefore track whether the coop
                // branch is actually safe for the kernel that will be dispatched, not just
                // whether coop is "active" in the abstract.
                ("LANES_PER_TOKEN", (coopActive && !(useH16 && queryHeadRepeats < 4)) ? 4 : 1),
            ] + metalTemplateSeedWords(prefix: "VALUE_SEED", value: valueCode.seed)

        let compressedPartials = partialKernel(
            [
                queries,
                keyCode.packedMagnitudes,
                keyCode.signs,
                keyCode.highPrecisionMask,
                keyCode.residualSigns,
                keyCode.scales,
                valueCode.packedMagnitudes,
                valueCode.signs,
                valueCode.highPrecisionMask,
                valueCode.residualSigns,
                valueCode.scales,
                Int32(keyCode.layout.logicalLength),
                Int32(keyCode.layout.ringOffset),
                Int32(keyCode.layout.pinnedPrefixLength),
                scale,
                Int32(activeBlockCount),
            ],
            template: template,
            grid: (partialRows * activeBlockCount * blockWidth, 1, 1),
            threadGroup: (blockWidth, 1, 1),
            outputShapes: [
                [rowCount, activeBlockCount, 2],
                [rowCount, activeBlockCount, queries.dim(3)],
            ],
            outputDTypes: [.float32, .float32],
            stream: stream
        )
        partialStats.append(compressedPartials[0])
        partialOut.append(compressedPartials[1])
        totalBlocks += activeBlockCount
    }

    let stats = partialStats.count == 1 ? partialStats[0] : concatenated(partialStats, axis: 1)
    let values = partialOut.count == 1 ? partialOut[0] : concatenated(partialOut, axis: 1)
    let reduceWidth = turboQuantBlockParallelFusedThreadgroupWidth(
        minimum: max(totalBlocks, queries.dim(3))
    )
    return TurboQuantMetalKernels.fusedAttentionBlockReduce(
        [stats, values, Int32(totalBlocks)],
        template: [
            ("ROW_COUNT", rowCount),
            ("HEAD_DIM", queries.dim(3)),
            ("THREADS_PER_BLOCK", reduceWidth),
            ("OUTPUT_DTYPE", outputDType),
        ],
        grid: (rowCount * reduceWidth, 1, 1),
        threadGroup: (reduceWidth, 1, 1),
        outputShapes: [[queries.dim(0), queries.dim(1), queries.dim(2), queries.dim(3)]],
        outputDTypes: [outputDType],
        stream: stream
    )[0]
}

public func turboQuantMetalSupportsOnlineFusedAttention(
    queries: MLXArray,
    keyCode: TurboQuantAttentionCode,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
) -> Bool {
    turboQuantMetalSupportsOnlineFusedAttention(
        queryShape: queries.shape,
        keyCode: keyCode,
        mask: mask
    )
}

public func turboQuantMetalSupportsOnlineFusedAttention(
    queryShape: [Int],
    keyCode: TurboQuantAttentionCode,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
) -> Bool {
    turboQuantMetalSupportsOnlineFusedAttention(
        queryShape: queryShape,
        keyLayout: keyCode.layout,
        mask: mask
    )
}

public func turboQuantMetalSupportsOnlineFusedAttention(
    queryShape: [Int],
    keyLayout: TurboQuantAttentionLayout,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
) -> Bool {
    guard queryShape.count == 4 else { return false }
    guard queryShape[0] == keyLayout.batchSize, queryShape[2] <= 8 else { return false }
    guard
        TurboQuantRuntimeProbeResult.throughputOptimizedOnlineFusedHeadDimensions
            .contains(queryShape[3])
    else { return false }
    guard queryShape[3] == keyLayout.headDimension else { return false }
    switch mask {
    case .none, .causal:
        return true
    case .array, .arrays:
        return false
    }
}

public func turboQuantWarmAttentionKernelVariants(
    headDimensions: [Int] = TurboQuantRuntimeProbeResult
        .throughputOptimizedOnlineFusedHeadDimensions,
    longContextTokenCounts: [Int] = [],
    preset: TurboQuantPreset = .turbo4v2,
    groupSize: Int = 64,
    kernelProfile: TurboQuantKernelProfile? = nil,
    stream: StreamOrDevice = .gpu
) throws {
    if !TurboQuantRuntimeProbe.shared.isRunningSelfTest() {
        try requireTurboQuantMetalAttention()
    }

    let profile =
        kernelProfile
        ?? TurboQuantRuntimeProbe.shared.selectedKernelProfileWithoutRunningProbe()
    for headDimension in headDimensions {
        guard
            TurboQuantRuntimeProbeResult.throughputOptimizedOnlineFusedHeadDimensions
                .contains(headDimension)
        else {
            continue
        }

        let query = MLXArray.zeros([1, 1, 1, headDimension], dtype: .float32)
        let keys = MLXArray.zeros([1, 1, 1, headDimension], dtype: .float32)
        let values = MLXArray.zeros([1, 1, 1, headDimension], dtype: .float32)
        let keyCode = try turboQuantMetalEncodeAttention(
            keys,
            configuration: TurboQuantConfiguration(
                preset: preset,
                role: .key,
                groupSize: groupSize,
                backend: .metalPolarQJL,
                seed: UInt64(headDimension) ^ 0xA77E_0000_0000_0001
            ),
            stream: stream
        )
        let valueCode = try turboQuantMetalEncodeAttention(
            values,
            configuration: TurboQuantConfiguration(
                preset: preset,
                role: .value,
                groupSize: groupSize,
                backend: .metalPolarQJL,
                seed: UInt64(headDimension) ^ 0xA77E_0000_0000_0002
            ),
            stream: stream
        )
        let output = try turboQuantMetalOnlineFusedAttention(
            queries: query,
            keyCode: keyCode,
            valueCode: valueCode,
            scale: 1 / sqrt(Float(headDimension)),
            mask: .causal,
            kernelProfile: profile,
            outputDType: .float32,
            stream: stream
        )
        eval(output)

        for contextTokens in longContextTokenCounts where contextTokens > 1 {
            let queryHeadCount = profile == .macAppleSilicon ? 4 : 1
            let longQuery = MLXArray.zeros(
                [1, queryHeadCount, 1, headDimension],
                dtype: .float32
            )
            let longKeys = MLXArray.zeros(
                [1, 1, contextTokens, headDimension],
                dtype: .float32
            )
            let longValues = MLXArray.zeros(
                [1, 1, contextTokens, headDimension],
                dtype: .float32
            )
            let longKeyCode = try turboQuantMetalEncodeAttention(
                longKeys,
                configuration: TurboQuantConfiguration(
                    preset: preset,
                    role: .key,
                    groupSize: groupSize,
                    backend: .metalPolarQJL,
                    seed: UInt64(headDimension) ^ UInt64(contextTokens)
                        ^ 0xA77E_0000_0000_0101
                ),
                stream: stream
            )
            let longValueCode = try turboQuantMetalEncodeAttention(
                longValues,
                configuration: TurboQuantConfiguration(
                    preset: preset,
                    role: .value,
                    groupSize: groupSize,
                    backend: .metalPolarQJL,
                    seed: UInt64(headDimension) ^ UInt64(contextTokens)
                        ^ 0xA77E_0000_0000_0102
                ),
                stream: stream
            )
            let longOutput = try turboQuantMetalOnlineFusedAttention(
                queries: longQuery,
                keyCode: longKeyCode,
                valueCode: longValueCode,
                scale: 1 / sqrt(Float(headDimension)),
                mask: .causal,
                kernelProfile: profile,
                outputDType: .float32,
                stream: stream
            )
            eval(longOutput)
        }
    }
}

private func turboQuantMetalOnlineFusedAttention(
    queries: MLXArray,
    keyCode: TurboQuantAttentionCode,
    valueCode: TurboQuantAttentionCode,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    kernelProfile: TurboQuantKernelProfile,
    blockParallelTokenBlockSize: Int? = nil,
    outputDType: DType,
    stateAlreadyValidated: Bool = false,
    stream: StreamOrDevice
) throws -> MLXArray {
    if !stateAlreadyValidated {
        try validateAttentionPair(keyCode: keyCode, valueCode: valueCode, allowTileTransposedV7: true)
        try validateAttentionQuery(queries, code: keyCode)
        try validateAttentionCodeStorage(keyCode, allowTileTransposedV7: true)
        try validateAttentionCodeStorage(valueCode, allowTileTransposedV7: true)
    }
    let outputShape = [queries.dim(0), queries.dim(1), queries.dim(2), queries.dim(3)]
    let rowCount = queries.dim(0) * queries.dim(1) * queries.dim(2)
    let threadgroupWidth = turboQuantOnlineFusedThreadgroupWidth(
        minimum: max(queries.dim(3), kernelProfile.fusedDecodeThreadgroupWidth)
    )
    let causal: Bool
    switch mask {
    case .causal:
        causal = true
    case .none:
        causal = false
    case .array, .arrays:
        throw TurboQuantError.invalidMetalConfiguration(
            "online fused TurboQuant attention does not support materialized masks"
        )
    }

    let useBlockParallel = turboQuantShouldUseBlockParallelFusedAttention(
        queries: queries,
        keyCode: keyCode,
        valueCode: valueCode,
        kernelProfile: kernelProfile,
        blockParallelTokenBlockSize: blockParallelTokenBlockSize
    )
    if keyCode.layout.layoutVersion == TurboQuantAttentionLayout.tileTransposedVersion,
        !useBlockParallel
    {
        throw TurboQuantError.invalidMetalConfiguration(
            "layout v7 is only ported to the block-parallel GQA partials kernel; the single-pass fused kernel is not ported"
        )
    }
    if useBlockParallel {
        return try turboQuantMetalBlockParallelFusedAttention(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            scale: scale,
            kernelProfile: kernelProfile,
            blockParallelTokenBlockSize: blockParallelTokenBlockSize,
            outputDType: outputDType,
            causal: causal,
            stream: stream
        )
    }

    TurboQuantKernelDispatchTrace.shared.record("singlePass:turboquant_attention_fused_decode_runtime_layout_s2")
    return TurboQuantMetalKernels.fusedAttention(
        [
            queries,
            keyCode.packedMagnitudes,
            keyCode.signs,
            keyCode.highPrecisionMask,
            keyCode.residualSigns,
            keyCode.scales,
            valueCode.packedMagnitudes,
            valueCode.signs,
            valueCode.highPrecisionMask,
            valueCode.residualSigns,
            valueCode.scales,
            Int32(keyCode.layout.logicalLength),
            Int32(keyCode.layout.ringOffset),
            Int32(keyCode.layout.pinnedPrefixLength),
            scale,
        ],
        template: runtimeLayoutAttentionTemplate(
            configuration: TurboQuantConfiguration(
                preset: keyCode.preset,
                role: .key,
                groupSize: keyCode.groupSize,
                backend: .metalPolarQJL,
                seed: keyCode.seed,
                valueBits: valueCode.valueBits,
                attentionLayoutVersion: keyCode.layout.layoutVersion,
                allowExperimentalLayoutV5: keyCode.layout.isLayoutV5,
                attentionScaleStorage: turboQuantAttentionScaleStorage(for: keyCode)
            ),
            layout: keyCode.layout,
            inputLength: keyCode.layout.logicalLength,
            outputLength: keyCode.layout.logicalLength,
            queryHeadCount: queries.dim(1),
            queryLength: queries.dim(2),
            outputDType: outputDType,
            causal: causal
        ) + [
            ("VALUE_MAG_WORDS_PER_GROUP", valueCode.layout.magnitudeWordsPerGroup),
            ("VALUE_SCALES_PER_GROUP", valueCode.scalesPerGroup),
            ("THREADS_PER_ROW", threadgroupWidth),
        ] + metalTemplateSeedWords(prefix: "VALUE_SEED", value: valueCode.seed),
        grid: (rowCount * threadgroupWidth, 1, 1),
        threadGroup: (threadgroupWidth, 1, 1),
        outputShapes: [outputShape],
        outputDTypes: [outputDType],
        stream: stream
    )[0]
}

private func turboQuantShouldUseBlockParallelFusedAttention(
    queries: MLXArray,
    keyCode: TurboQuantAttentionCode,
    valueCode: TurboQuantAttentionCode,
    kernelProfile: TurboQuantKernelProfile,
    blockParallelTokenBlockSize: Int? = nil
) -> Bool {
    guard queries.dim(2) == 1 else { return false }
    guard queries.dim(3) == keyCode.layout.headDimension else { return false }
    guard keyCode.layout.headDimension == valueCode.layout.headDimension else { return false }
    guard
        let blockWidth = turboQuantResolvedBlockParallelTokenBlockSize(
            logicalLength: keyCode.layout.logicalLength,
            headDimension: queries.dim(3),
            queryLength: queries.dim(2),
            kernelProfile: kernelProfile,
            requestedBlockParallelTokenBlockSize: blockParallelTokenBlockSize
        )
    else { return false }
    let activeBlockCount = (keyCode.layout.logicalLength + blockWidth - 1) / blockWidth
    guard activeBlockCount > 1, activeBlockCount <= blockWidth else { return false }
    return true
}

private func turboQuantMetalBlockParallelFusedAttention(
    queries: MLXArray,
    keyCode: TurboQuantAttentionCode,
    valueCode: TurboQuantAttentionCode,
    scale: Float,
    kernelProfile: TurboQuantKernelProfile,
    blockParallelTokenBlockSize: Int? = nil,
    outputDType: DType,
    causal: Bool,
    stream: StreamOrDevice
) throws -> MLXArray {
    let outputShape = [queries.dim(0), queries.dim(1), queries.dim(2), queries.dim(3)]
    let rowCount = queries.dim(0) * queries.dim(1) * queries.dim(2)
    guard
        let blockWidth = turboQuantResolvedBlockParallelTokenBlockSize(
            logicalLength: keyCode.layout.logicalLength,
            headDimension: queries.dim(3),
            queryLength: queries.dim(2),
            kernelProfile: kernelProfile,
            requestedBlockParallelTokenBlockSize: blockParallelTokenBlockSize
        )
    else {
        throw TurboQuantError.invalidMetalConfiguration(
            "block-parallel fused attention is not recommended for logical length \(keyCode.layout.logicalLength), query length \(queries.dim(2)), head dimension \(queries.dim(3)), and profile \(kernelProfile.rawValue)"
        )
    }
    let activeBlockCount = (keyCode.layout.logicalLength + blockWidth - 1) / blockWidth
    guard activeBlockCount > 1, activeBlockCount <= blockWidth else {
        throw TurboQuantError.invalidMetalConfiguration(
            "block-parallel fused attention requires 2...\(blockWidth) active blocks, got \(activeBlockCount)"
        )
    }
    let queryHeadCount = queries.dim(1)
    let kvHeadCount = keyCode.layout.kvHeadCount
    let queryHeadRepeats = queryHeadCount / kvHeadCount
    let useGroupedQueryKernel =
        kernelProfile == .macAppleSilicon
        && queryHeadCount % kvHeadCount == 0
        && queryHeadRepeats > 1
        && queryHeadRepeats <= 4
    let coopActive = turboQuantCooperativeQuadDecodeActive(
        enabled: useGroupedQueryKernel,
        queryHeadRepeats: queryHeadRepeats,
        preset: keyCode.preset,
        layoutVersion: keyCode.layout.layoutVersion,
        headDim: queries.dim(3),
        groupSize: keyCode.groupSize,
        logicalLength: keyCode.layout.logicalLength)

    let isTileTransposedV7 =
        keyCode.layout.layoutVersion == TurboQuantAttentionLayout.tileTransposedVersion
    if isTileTransposedV7 {
        guard useGroupedQueryKernel else {
            throw TurboQuantError.invalidMetalConfiguration(
                "layout v7 supports only the grouped-query block-partials kernel"
            )
        }
        guard !coopActive else {
            throw TurboQuantError.invalidMetalConfiguration(
                "cooperative decode is not ported to layout v7"
            )
        }
    }

    let useH16 = turboQuantH16DietEnabled && useGroupedQueryKernel && !isTileTransposedV7

    let template =
        runtimeLayoutAttentionTemplate(
            configuration: TurboQuantConfiguration(
                preset: keyCode.preset,
                role: .key,
                groupSize: keyCode.groupSize,
                backend: .metalPolarQJL,
                seed: keyCode.seed,
                valueBits: valueCode.valueBits,
                attentionLayoutVersion: keyCode.layout.layoutVersion,
                allowExperimentalLayoutV5: keyCode.layout.isLayoutV5,
                attentionScaleStorage: turboQuantAttentionScaleStorage(for: keyCode)
            ),
            layout: keyCode.layout,
            inputLength: keyCode.layout.logicalLength,
            outputLength: keyCode.layout.logicalLength,
            queryHeadCount: queries.dim(1),
            queryLength: queries.dim(2),
            outputDType: outputDType,
            causal: causal
        ) + [
            ("VALUE_MAG_WORDS_PER_GROUP", valueCode.layout.magnitudeWordsPerGroup),
            ("VALUE_SCALES_PER_GROUP", valueCode.scalesPerGroup),
            ("THREADS_PER_BLOCK", blockWidth),
            ("BLOCK_TOKENS", blockWidth),
            ("GQA_REPEATS", useGroupedQueryKernel ? queryHeadRepeats : 1),
        ]
        // H16-BUG-FIX: see the segmented-site comment above the analogous LANES_PER_TOKEN
        // line. When useH16 is true and repeats < 4, the selected kernel is the strided
        // fusedAttentionGQABlockPartialsH16, whose coop branch (lanes_per_token == 4u)
        // hardcodes r<4u loops over query_cache rows only initialized up to repeat_count.
        // LANES_PER_TOKEN must not be 4 for that combination.
        + (isTileTransposedV7
            ? []
            : [("LANES_PER_TOKEN", (coopActive && !(useH16 && queryHeadRepeats < 4)) ? 4 : 1)])
        + metalTemplateSeedWords(prefix: "VALUE_SEED", value: valueCode.seed)

    let partialRows =
        useGroupedQueryKernel
        ? queries.dim(0) * kvHeadCount * queries.dim(2)
        : rowCount
    // COOPW is not ported to the H16 diet kernel (out of scope for T2.4 stage-2); H16-coop
    // dispatch stays gated to exactly repeats==4 regardless of the widened
    // turboQuantCooperativeQuadDecodeActive guard above.
    let coopWActive = coopActive && queryHeadRepeats < 4 && !useH16 && !isTileTransposedV7
    let partialKernel =
        isTileTransposedV7
        ? TurboQuantMetalKernels.fusedAttentionGQABlockPartialsV7
        : (useH16
            ? ((coopActive && queryHeadRepeats == 4)
                ? TurboQuantMetalKernels.fusedAttentionGQABlockPartialsCoopH16_rf1
                : TurboQuantMetalKernels.fusedAttentionGQABlockPartialsH16_rf1)
            : (coopWActive
                ? TurboQuantMetalKernels.fusedAttentionGQABlockPartialsCoopW_rf1
                : (coopActive
                    ? TurboQuantMetalKernels.fusedAttentionGQABlockPartialsCoop_rf1
                    : (useGroupedQueryKernel
                        ? TurboQuantMetalKernels.fusedAttentionGQABlockPartials_rf1
                        : TurboQuantMetalKernels.fusedAttentionBlockPartials))))
    TurboQuantKernelDispatchTrace.shared.record(
        "blockParallel:" + (isTileTransposedV7
            ? "turboquant_attention_fused_gqa_block_partials_runtime_layout_rtu1_s2_v7"
            : (useH16
                ? ((coopActive && queryHeadRepeats == 4)
                    ? "turboquant_attention_fused_gqa_block_partials_coop_runtime_layout_rtu1_s2_h16_rf1"
                    : "turboquant_attention_fused_gqa_block_partials_runtime_layout_rtu1_s2_h16_rf1")
                : (coopWActive
                    ? "turboquant_attention_fused_gqa_block_partials_coopw_runtime_layout_rtu1_s2_rf1"
                    : (coopActive
                        ? "turboquant_attention_fused_gqa_block_partials_coop_runtime_layout_rtu1_s2_rf1"
                        : (useGroupedQueryKernel
                            ? "turboquant_attention_fused_gqa_block_partials_runtime_layout_rtu1_s2_rf1"
                            : "turboquant_attention_fused_block_partials_runtime_layout_rtu1_s2"))))))
    let partialOutputDType = outputDType == .float32 ? DType.float32 : outputDType
    let partials = partialKernel(
        [
            queries,
            keyCode.packedMagnitudes,
            keyCode.signs,
            keyCode.highPrecisionMask,
            keyCode.residualSigns,
            keyCode.scales,
            valueCode.packedMagnitudes,
            valueCode.signs,
            valueCode.highPrecisionMask,
            valueCode.residualSigns,
            valueCode.scales,
            Int32(keyCode.layout.logicalLength),
            Int32(keyCode.layout.ringOffset),
            Int32(keyCode.layout.pinnedPrefixLength),
            scale,
            Int32(activeBlockCount),
        ],
        template: template,
        grid: (partialRows * activeBlockCount * blockWidth, 1, 1),
        threadGroup: (blockWidth, 1, 1),
        outputShapes: [
            [rowCount, activeBlockCount, 2],
            [rowCount, activeBlockCount, queries.dim(3)],
        ],
        outputDTypes: [.float32, partialOutputDType],
        stream: stream
    )

    let reduceWidth = turboQuantBlockParallelFusedThreadgroupWidth(
        minimum: max(activeBlockCount, queries.dim(3))
    )
    let reduceInputs: [any ScalarOrArray] =
        partials.map { $0 as any ScalarOrArray } + [Int32(activeBlockCount)]
    return TurboQuantMetalKernels.fusedAttentionBlockReduce(
        reduceInputs,
        template: [
            ("ROW_COUNT", rowCount),
            ("HEAD_DIM", queries.dim(3)),
            ("THREADS_PER_BLOCK", reduceWidth),
            ("OUTPUT_DTYPE", outputDType),
        ],
        grid: (rowCount * reduceWidth, 1, 1),
        threadGroup: (reduceWidth, 1, 1),
        outputShapes: [outputShape],
        outputDTypes: [outputDType],
        stream: stream
    )[0]
}

public func requireTurboQuantBackend(_ backend: TurboQuantBackend) throws {
    let availability = TurboQuantKernelAvailability.current
    guard availability.supports(backend) else {
        throw TurboQuantError.unsupportedBackend(
            backend,
            availability.fallbackReason(for: backend) ?? "Backend unavailable."
        )
    }
}

public func requireTurboQuantMetalAttention() throws {
    guard metalRuntimeAvailable() else {
        throw TurboQuantError.unsupportedBackend(
            .metalPolarQJL,
            "Metal runtime is unavailable for PolarQuant/QJL compressed attention."
        )
    }
    guard !TurboQuantRuntimeProbe.shared.isRunningSelfTest() else { return }
    let probe = TurboQuantRuntimeProbe.shared.result()
    let capabilities = probe.kernelCapabilities
    guard capabilities.attentionQK && capabilities.attentionAV else {
        throw TurboQuantError.unsupportedBackend(
            .metalPolarQJL,
            probe.failureReason
                ?? "PolarQuant/QJL compressed two-stage attention self-test has not passed."
        )
    }
}

private func requireTurboQuantMetalAttentionOutputDType(_ dtype: DType) throws {
    guard dtype.isFloatingPoint else {
        throw TurboQuantError.invalidMetalConfiguration(
            "compressed attention output dtype must be floating point")
    }
    guard
        dtype != .bfloat16 || TurboQuantRuntimeProbe.shared.isRunningSelfTest()
            || TurboQuantRuntimeProbe.shared.result().kernelCapabilities.bfloatOutput
    else {
        throw TurboQuantError.unsupportedBackend(
            .metalPolarQJL,
            "bfloat16 compressed attention output has not passed runtime certification."
        )
    }
}

public func requireTurboQuantMetalCodec() throws {
    guard metalRuntimeAvailable() else {
        throw TurboQuantError.unsupportedBackend(
            .metalPolarQJL,
            "Metal runtime is unavailable for the PolarQuant/QJL codec."
        )
    }
    guard !TurboQuantRuntimeProbe.shared.isRunningSelfTest() else { return }
    guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLCodec else {
        throw TurboQuantError.unsupportedBackend(
            .metalPolarQJL,
            TurboQuantRuntimeProbe.shared.result().failureReason
                ?? "PolarQuant/QJL codec self-test has not passed."
        )
    }
}

private func encodeTurboQuantReference(
    values: [Float],
    shape: [Int],
    configuration: TurboQuantConfiguration
) throws -> TurboQuantReferenceCode {
    let expectedCount = shape.reduce(1, *)
    guard expectedCount == values.count else {
        throw TurboQuantError.invalidReferenceCode(
            "shape \(shape) contains \(expectedCount) values but input has \(values.count)"
        )
    }

    if configuration.role == .value {
        return try encodeTurboQuantAffineValueReference(
            values: values,
            shape: shape,
            configuration: configuration
        )
    }

    if configuration.role == .key {
        return try encodeTurboQuantProductReference(
            values: values,
            shape: shape,
            configuration: configuration
        )
    }

    let groupSize = configuration.groupSize
    let baseBits = configuration.preset.baseMagnitudeBits
    let highBits = configuration.preset.highMagnitudeBits
    let groupCount = (values.count + groupSize - 1) / groupSize
    var baseScales = Array(repeating: Float(1), count: groupCount)
    var highScales = Array(repeating: Float(1), count: groupCount)
    var residualScales = Array(repeating: Float(0), count: groupCount)
    var signs = [UInt8](repeating: 0, count: packedBitByteCount(values.count))
    var highPrecisionMask = [UInt8](repeating: 0, count: packedBitByteCount(values.count))
    var residualSigns = [UInt8](repeating: 0, count: packedBitByteCount(values.count))
    var magnitudes = [UInt8]()
    var magnitudeBitOffset = 0

    for groupIndex in 0 ..< groupCount {
        let start = groupIndex * groupSize
        let end = Swift.min(start + groupSize, values.count)
        let count = end - start
        guard count > 0 else { continue }

        var transformed = Array(repeating: Float(0), count: count)
        var maxAbs = Float(0)
        for localIndex in 0 ..< count {
            let absoluteIndex = start + localIndex
            let value = preconditionedValue(
                values[absoluteIndex],
                index: absoluteIndex,
                seed: configuration.seed
            )
            transformed[localIndex] = value
            maxAbs = Swift.max(maxAbs, Swift.abs(value))
        }

        let baseMax = Float((1 << baseBits) - 1)
        let highMax = Float((1 << highBits) - 1)
        let safeMaxAbs = Swift.max(maxAbs, Float.leastNonzeroMagnitude)
        baseScales[groupIndex] = safeMaxAbs / baseMax
        highScales[groupIndex] = safeMaxAbs / highMax

        let highPrecisionCount = mixedPrecisionHighCount(
            valueCount: count,
            baseBits: baseBits,
            highBits: highBits,
            targetBits: configuration.preset.targetMagnitudeBits
        )
        var highPrecisionIndices = Set<Int>()
        if highPrecisionCount > 0 {
            let ranked = transformed.indices.sorted { lhs, rhs in
                let leftMagnitude = Swift.abs(transformed[lhs])
                let rightMagnitude = Swift.abs(transformed[rhs])
                if leftMagnitude == rightMagnitude {
                    return lhs < rhs
                }
                return leftMagnitude > rightMagnitude
            }
            highPrecisionIndices = Set(ranked.prefix(highPrecisionCount))
        }

        var residuals = Array(repeating: Float(0), count: count)
        var residualMagnitudeSum = Float(0)
        for localIndex in 0 ..< count {
            let value = transformed[localIndex]
            let highPrecision = highPrecisionIndices.contains(localIndex)
            let bits = highPrecision ? highBits : baseBits
            let scale = highPrecision ? highScales[groupIndex] : baseScales[groupIndex]
            let levelMax = Float((1 << bits) - 1)
            let magnitude = Swift.abs(value)
            let quantizedMagnitude = UInt8(
                Swift.max(0, Swift.min(Int((magnitude / scale).rounded()), Int(levelMax)))
            )
            let signedDecoded = (value.sign == .minus ? -1 : 1) * Float(quantizedMagnitude) * scale
            let residual = value - signedDecoded
            residuals[localIndex] = residual
            residualMagnitudeSum += Swift.abs(residual)
        }
        if configuration.role != .value {
            residualScales[groupIndex] = residualMagnitudeSum / Float(count)
        }

        for localIndex in 0 ..< count {
            let absoluteIndex = start + localIndex
            let value = transformed[localIndex]
            let highPrecision = highPrecisionIndices.contains(localIndex)
            let bits = highPrecision ? highBits : baseBits
            let scale = highPrecision ? highScales[groupIndex] : baseScales[groupIndex]
            let levelMax = Float((1 << bits) - 1)
            let magnitude = Swift.abs(value)
            let quantizedMagnitude = UInt8(
                Swift.max(0, Swift.min(Int((magnitude / scale).rounded()), Int(levelMax)))
            )
            setPackedBit(&signs, index: absoluteIndex, value: value.sign == .minus)
            setPackedBit(&highPrecisionMask, index: absoluteIndex, value: highPrecision)
            if configuration.role != .value {
                setPackedBit(
                    &residualSigns, index: absoluteIndex,
                    value: residuals[localIndex].sign == .minus)
            }
            appendPackedBits(
                UInt32(quantizedMagnitude),
                bitCount: bits,
                bytes: &magnitudes,
                bitOffset: &magnitudeBitOffset
            )
        }
    }

    if configuration.role == .value {
        residualSigns.removeAll(keepingCapacity: false)
    }

    return TurboQuantReferenceCode(
        shape: shape,
        preset: configuration.preset,
        role: configuration.role,
        format: .magnitudeResidualSign,
        groupSize: groupSize,
        seed: configuration.seed,
        residualScale: configuration.qjlResidualScale,
        baseMagnitudeBits: baseBits,
        highMagnitudeBits: highBits,
        valueCount: values.count,
        baseScales: baseScales,
        highScales: highScales,
        residualScales: residualScales,
        signs: Data(signs),
        highPrecisionMask: Data(highPrecisionMask),
        residualSigns: Data(residualSigns),
        packedMagnitudes: Data(magnitudes)
    )
}

// MARK: - N4 data-free Gaussian Lloyd-Max payload quantizer (reference codec)

/// Optimal scalar-quantizer reproduction points for a unit Gaussian at 2^bits levels,
/// computed by the Lloyd algorithm over the N(0,1) density on a fixed grid. Data-free
/// (no input dependence) — it targets the post-rotation Gaussian distribution QJL produces.
private func gaussianLloydMaxCentroids(bits: Int) -> [Float] {
    let levels = Swift.max(2, 1 << Swift.max(1, bits))
    let gridCount = 8192
    var xs = [Double](repeating: 0, count: gridCount)
    var w = [Double](repeating: 0, count: gridCount)
    for i in 0 ..< gridCount {
        let x = -6.0 + 12.0 * Double(i) / Double(gridCount - 1)
        xs[i] = x
        w[i] = exp(-0.5 * x * x)  // unnormalized Gaussian density
    }
    var c = (0 ..< levels).map { -3.0 + 6.0 * Double($0) / Double(levels - 1) }
    for _ in 0 ..< 80 {
        var sum = [Double](repeating: 0, count: levels)
        var cnt = [Double](repeating: 0, count: levels)
        for i in 0 ..< gridCount {
            var best = 0
            var bd = Double.infinity
            for k in 0 ..< levels {
                let d = abs(xs[i] - c[k])
                if d < bd { bd = d; best = k }
            }
            sum[best] += xs[i] * w[i]
            cnt[best] += w[i]
        }
        for k in 0 ..< levels where cnt[k] > 0 { c[k] = sum[k] / cnt[k] }
    }
    return c.map { Float($0) }
}

private func packGaussianIndex(_ value: Int, bits: Int, bit: Int, into buffer: inout [UInt8]) {
    for b in 0 ..< bits where (value >> b) & 1 == 1 {
        let pos = bit + b
        buffer[pos >> 3] |= UInt8(1 << (pos & 7))
    }
}

private func unpackGaussianIndex(bits: Int, bit: Int, from buffer: [UInt8]) -> Int {
    var value = 0
    for b in 0 ..< bits {
        let pos = bit + b
        if (buffer[pos >> 3] >> UInt8(pos & 7)) & 1 == 1 { value |= (1 << b) }
    }
    return value
}

/// Encode `array` with the data-free Gaussian Lloyd-Max payload quantizer (N4): a per-group
/// RMS norm (stored in `baseScales`) + packed centroid indices (`packedMagnitudes`). Decode
/// via the standard ``turboQuantReferenceDecode(_:)`` (it dispatches on the format).
public func turboQuantGaussianReferenceEncode(
    _ array: MLXArray,
    bits: Int,
    groupSize: Int = 64,
    role: TurboQuantTensorRole = .vector,
    preset: TurboQuantPreset = .turbo3_5,
    seed: UInt64 = 0x9E37_79B9_7F4A_7C15
) throws -> TurboQuantReferenceCode {
    guard bits >= 1, bits <= 8 else {
        throw TurboQuantError.invalidReferenceCode("gaussian quantizer bits must be 1...8")
    }
    guard groupSize > 0 else { throw TurboQuantError.invalidGroupSize(groupSize) }
    let values = array.asArray(Float.self)
    let n = values.count
    let groupCount = (n + groupSize - 1) / groupSize
    let centroids = gaussianLloydMaxCentroids(bits: bits)
    let levels = centroids.count

    var norms = [Float](repeating: 1, count: groupCount)
    var packed = [UInt8](repeating: 0, count: (n * bits + 7) / 8)
    for g in 0 ..< groupCount {
        let start = g * groupSize
        let end = Swift.min(start + groupSize, n)
        var sumSq: Double = 0
        for i in start ..< end { sumSq += Double(values[i]) * Double(values[i]) }
        let rms = (sumSq / Double(Swift.max(1, end - start))).squareRoot()
        let sigma = Float(rms > 0 ? rms : 1)
        norms[g] = sigma
        for i in start ..< end {
            let xn = values[i] / sigma
            var best = 0
            var bd = Float.infinity
            for k in 0 ..< levels {
                let d = abs(xn - centroids[k])
                if d < bd { bd = d; best = k }
            }
            packGaussianIndex(best, bits: bits, bit: i * bits, into: &packed)
        }
    }
    return TurboQuantReferenceCode(
        shape: array.shape, preset: preset, role: role, format: .gaussianLloydMax,
        groupSize: groupSize, seed: seed, residualScale: 0,
        baseMagnitudeBits: bits, highMagnitudeBits: bits, valueCount: n,
        baseScales: norms, highScales: [], residualScales: [],
        signs: Data(), highPrecisionMask: Data(), residualSigns: Data(),
        packedMagnitudes: Data(packed))
}

private func decodeTurboQuantGaussianLloydMaxReference(
    _ code: TurboQuantReferenceCode
) throws -> [Float] {
    guard code.groupSize > 0 else { throw TurboQuantError.invalidGroupSize(code.groupSize) }
    guard code.shape.reduce(1, *) == code.valueCount else {
        throw TurboQuantError.invalidReferenceCode(
            "shape \(code.shape) does not match value count \(code.valueCount)")
    }
    let bits = code.baseMagnitudeBits
    guard bits >= 1, bits <= 8 else {
        throw TurboQuantError.invalidReferenceCode("gaussian quantizer bits must be 1...8")
    }
    let n = code.valueCount
    let groupCount = (n + code.groupSize - 1) / code.groupSize
    guard code.baseScales.count == groupCount else {
        throw TurboQuantError.invalidReferenceCode("gaussian norm table count does not match groups")
    }
    let packed = [UInt8](code.packedMagnitudes)
    guard packed.count >= (n * bits + 7) / 8 else {
        throw TurboQuantError.invalidReferenceCode("gaussian packed index storage is truncated")
    }
    let centroids = gaussianLloydMaxCentroids(bits: bits)
    var values = [Float](repeating: 0, count: n)
    for i in 0 ..< n {
        let idx = unpackGaussianIndex(bits: bits, bit: i * bits, from: packed)
        let sigma = code.baseScales[i / code.groupSize]
        values[i] = centroids[Swift.min(idx, centroids.count - 1)] * sigma
    }
    return values
}

private func decodeTurboQuantReference(_ code: TurboQuantReferenceCode) throws -> [Float] {
    switch code.format {
    case .affineValue:
        return try decodeTurboQuantAffineValueReference(code)
    case .turboQuantProd:
        return try decodeTurboQuantProductReference(code)
    case .gaussianLloydMax:
        return try decodeTurboQuantGaussianLloydMaxReference(code)
    case .magnitudeResidualSign:
        break
    }

    guard code.groupSize > 0 else {
        throw TurboQuantError.invalidGroupSize(code.groupSize)
    }
    guard code.shape.reduce(1, *) == code.valueCount else {
        throw TurboQuantError.invalidReferenceCode(
            "shape \(code.shape) does not match value count \(code.valueCount)"
        )
    }

    let groupCount = (code.valueCount + code.groupSize - 1) / code.groupSize
    guard code.baseScales.count == groupCount, code.highScales.count == groupCount else {
        throw TurboQuantError.invalidReferenceCode("scale table count does not match groups")
    }
    guard code.residualScales.isEmpty || code.residualScales.count == groupCount else {
        throw TurboQuantError.invalidReferenceCode(
            "residual scale table count does not match groups")
    }
    guard code.signs.count >= packedBitByteCount(code.valueCount),
        code.highPrecisionMask.count >= packedBitByteCount(code.valueCount)
    else {
        throw TurboQuantError.invalidReferenceCode("bitset storage is truncated")
    }
    if code.role != .value && code.residualSigns.count < packedBitByteCount(code.valueCount) {
        throw TurboQuantError.invalidReferenceCode("residual sign storage is truncated")
    }

    var values = Array(repeating: Float(0), count: code.valueCount)
    var magnitudeBitOffset = 0

    for groupIndex in 0 ..< groupCount {
        let start = groupIndex * code.groupSize
        let end = Swift.min(start + code.groupSize, code.valueCount)
        for absoluteIndex in start ..< end {
            let highPrecision = getPackedBit(code.highPrecisionMask, index: absoluteIndex)
            let bits = highPrecision ? code.highMagnitudeBits : code.baseMagnitudeBits
            let scale = highPrecision ? code.highScales[groupIndex] : code.baseScales[groupIndex]
            let magnitude = Float(
                try readPackedBits(
                    code.packedMagnitudes,
                    bitOffset: &magnitudeBitOffset,
                    bitCount: bits
                )
            )
            let sign: Float = getPackedBit(code.signs, index: absoluteIndex) ? -1 : 1
            var reconstructed = sign * magnitude * scale

            if code.role != .value {
                let residualSign: Float =
                    getPackedBit(code.residualSigns, index: absoluteIndex) ? -1 : 1
                let residualScale =
                    code.residualScales.isEmpty
                    ? code.residualScale * scale
                    : code.residualScales[groupIndex]
                reconstructed += residualSign * residualScale
            }

            values[absoluteIndex] = unpreconditionedValue(
                reconstructed,
                index: absoluteIndex,
                seed: code.seed
            )
        }
    }

    return values
}

private func encodeTurboQuantAffineValueReference(
    values: [Float],
    shape: [Int],
    configuration: TurboQuantConfiguration
) throws -> TurboQuantReferenceCode {
    let groupSize = configuration.groupSize
    let valueBits = configuration.resolvedValueBits
    try validateTurboQuantValueBits(valueBits)

    let groupCount = (values.count + groupSize - 1) / groupSize
    var scales = Array(repeating: Float(0), count: groupCount)
    var zeros = Array(repeating: Float(0), count: groupCount)
    var packed = [UInt8]()
    var bitOffset = 0
    let levelMax = Float((1 << valueBits) - 1)

    for groupIndex in 0 ..< groupCount {
        let start = groupIndex * groupSize
        let end = Swift.min(start + groupSize, values.count)
        guard start < end else { continue }

        var minimum = Float.greatestFiniteMagnitude
        var maximum = -Float.greatestFiniteMagnitude
        for index in start ..< end {
            minimum = Swift.min(minimum, values[index])
            maximum = Swift.max(maximum, values[index])
        }

        let range = maximum - minimum
        let scale = range > Float.leastNonzeroMagnitude ? range / levelMax : 0
        scales[groupIndex] = scale
        zeros[groupIndex] = minimum

        for index in start ..< end {
            let quantized: UInt32
            if scale == 0 {
                quantized = 0
            } else {
                quantized = UInt32(
                    Swift.max(
                        0,
                        Swift.min(
                            Int(((values[index] - minimum) / scale).rounded()),
                            Int(levelMax)
                        )
                    )
                )
            }
            appendPackedBits(
                quantized,
                bitCount: valueBits,
                bytes: &packed,
                bitOffset: &bitOffset
            )
        }
    }

    return TurboQuantReferenceCode(
        shape: shape,
        preset: configuration.preset,
        role: configuration.role,
        format: .affineValue,
        groupSize: groupSize,
        seed: configuration.seed,
        residualScale: configuration.qjlResidualScale,
        baseMagnitudeBits: valueBits,
        highMagnitudeBits: valueBits,
        valueCount: values.count,
        baseScales: scales,
        highScales: zeros,
        residualScales: [],
        signs: Data(),
        highPrecisionMask: Data(),
        residualSigns: Data(),
        packedMagnitudes: Data(packed)
    )
}

private func decodeTurboQuantAffineValueReference(_ code: TurboQuantReferenceCode) throws
    -> [Float]
{
    guard code.groupSize > 0 else {
        throw TurboQuantError.invalidGroupSize(code.groupSize)
    }
    try validateTurboQuantValueBits(code.baseMagnitudeBits)
    let groupCount = (code.valueCount + code.groupSize - 1) / code.groupSize
    guard code.baseScales.count == groupCount, code.highScales.count == groupCount else {
        throw TurboQuantError.invalidReferenceCode("affine value scale table count mismatch")
    }

    var values = Array(repeating: Float(0), count: code.valueCount)
    var bitOffset = 0
    for groupIndex in 0 ..< groupCount {
        let start = groupIndex * code.groupSize
        let end = Swift.min(start + code.groupSize, code.valueCount)
        let scale = code.baseScales[groupIndex]
        let zero = code.highScales[groupIndex]
        for index in start ..< end {
            let quantized = try readPackedBits(
                code.packedMagnitudes,
                bitOffset: &bitOffset,
                bitCount: code.baseMagnitudeBits
            )
            values[index] = zero + Float(quantized) * scale
        }
    }
    return values
}

private func encodeTurboQuantProductReference(
    values: [Float],
    shape: [Int],
    configuration: TurboQuantConfiguration
) throws -> TurboQuantReferenceCode {
    let groupSize = configuration.groupSize
    let baseBits = Swift.max(1, configuration.preset.baseMagnitudeBits - 1)
    let highBits = Swift.max(baseBits, configuration.preset.highMagnitudeBits - 1)
    let targetBits = Swift.max(1, configuration.preset.targetMagnitudeBits - 1)
    let groupCount = (values.count + groupSize - 1) / groupSize
    var norms = Array(repeating: Float(0), count: groupCount)
    var residualNorms = Array(repeating: Float(0), count: groupCount)
    var qjlSigns = [UInt8](repeating: 0, count: packedBitByteCount(values.count))
    var highPrecisionMask = [UInt8](repeating: 0, count: packedBitByteCount(values.count))
    var packed = [UInt8]()
    var bitOffset = 0

    for groupIndex in 0 ..< groupCount {
        let start = groupIndex * groupSize
        let end = Swift.min(start + groupSize, values.count)
        let count = end - start
        guard count > 0 else { continue }

        var group = Array(values[start ..< end])
        let norm = sqrt(group.reduce(Float(0)) { $0 + $1 * $1 })
        norms[groupIndex] = norm
        if norm > Float.leastNonzeroMagnitude {
            for index in group.indices {
                group[index] /= norm
            }
        }

        let rotated = applyTurboQuantRotation(
            group,
            seed: configuration.seed,
            groupIndex: groupIndex,
            inverse: false
        )
        let highCount = mixedPrecisionHighCount(
            valueCount: count,
            baseBits: baseBits,
            highBits: highBits,
            targetBits: targetBits
        )
        let usesDerivedHighMask =
            configuration.role == .key
            && !configuration.deterministicHighPrecisionMask
            && highBits == baseBits + 1
        let highMask =
            usesDerivedHighMask
            ? splitHighPrecisionMask(valueCount: count, highCount: highCount)
            : productHighPrecisionMask(
                valueCount: count,
                highCount: highCount,
                seed: configuration.seed,
                groupIndex: groupIndex
            )
        var quantizedRotated = Array(repeating: Float(0), count: count)

        for localIndex in 0 ..< count {
            let bits = highMask[localIndex] ? highBits : baseBits
            setPackedBit(
                &highPrecisionMask,
                index: start + localIndex,
                value: highMask[localIndex]
            )
            let codebook = turboQuantLloydMaxCodebook(
                bits: bits,
                coordinateStdDev: 1 / sqrt(Float(count))
            )
            let codeIndex = nearestCodebookIndex(rotated[localIndex], codebook: codebook)
            quantizedRotated[localIndex] = codebook[codeIndex]
            appendPackedBits(
                UInt32(codeIndex),
                bitCount: bits,
                bytes: &packed,
                bitOffset: &bitOffset
            )
        }

        var residualSquared = Float(0)
        for localIndex in 0 ..< count {
            let residual = rotated[localIndex] - quantizedRotated[localIndex]
            residualSquared += residual * residual
            setPackedBit(
                &qjlSigns,
                index: start + localIndex,
                value: residual.sign == .minus
            )
        }
        residualNorms[groupIndex] = norm * sqrt(residualSquared)
    }

    return TurboQuantReferenceCode(
        shape: shape,
        preset: configuration.preset,
        role: configuration.role,
        format: .turboQuantProd,
        groupSize: groupSize,
        seed: configuration.seed,
        residualScale: configuration.qjlResidualScale,
        baseMagnitudeBits: baseBits,
        highMagnitudeBits: highBits,
        valueCount: values.count,
        baseScales: norms,
        highScales: residualNorms,
        residualScales: [],
        signs: Data(qjlSigns),
        highPrecisionMask: Data(highPrecisionMask),
        residualSigns: Data(),
        packedMagnitudes: Data(packed)
    )
}

private func decodeTurboQuantProductReference(_ code: TurboQuantReferenceCode) throws -> [Float] {
    guard code.groupSize > 0 else {
        throw TurboQuantError.invalidGroupSize(code.groupSize)
    }
    let groupCount = (code.valueCount + code.groupSize - 1) / code.groupSize
    guard code.baseScales.count == groupCount, code.highScales.count == groupCount else {
        throw TurboQuantError.invalidReferenceCode("TurboQuantProd norm table count mismatch")
    }

    var values = Array(repeating: Float(0), count: code.valueCount)
    var bitOffset = 0
    for groupIndex in 0 ..< groupCount {
        let start = groupIndex * code.groupSize
        let end = Swift.min(start + code.groupSize, code.valueCount)
        let count = end - start
        guard count > 0 else { continue }

        let highCount = mixedPrecisionHighCount(
            valueCount: count,
            baseBits: code.baseMagnitudeBits,
            highBits: code.highMagnitudeBits,
            targetBits: Swift.max(1, code.preset.targetMagnitudeBits - 1)
        )
        let highMask = productHighPrecisionMask(
            code: code,
            start: start,
            count: count,
            highCount: highCount,
            groupIndex: groupIndex
        )
        var rotated = Array(repeating: Float(0), count: count)
        for localIndex in 0 ..< count {
            let bits = highMask[localIndex] ? code.highMagnitudeBits : code.baseMagnitudeBits
            let codebook = turboQuantLloydMaxCodebook(
                bits: bits,
                coordinateStdDev: 1 / sqrt(Float(count))
            )
            let codeIndex = Int(
                try readPackedBits(
                    code.packedMagnitudes,
                    bitOffset: &bitOffset,
                    bitCount: bits
                )
            )
            guard codeIndex < codebook.count else {
                throw TurboQuantError.invalidReferenceCode("TurboQuantProd codebook index overflow")
            }
            rotated[localIndex] = codebook[codeIndex]
        }

        let unrotated = applyTurboQuantRotation(
            rotated,
            seed: code.seed,
            groupIndex: groupIndex,
            inverse: true
        )
        let norm = code.baseScales[groupIndex]
        for localIndex in 0 ..< count {
            values[start + localIndex] = unrotated[localIndex] * norm
        }
    }
    return values
}

private func turboQuantProductInnerProduct(query: [Float], code: TurboQuantReferenceCode) throws
    -> Float
{
    let groupCount = (code.valueCount + code.groupSize - 1) / code.groupSize
    guard code.baseScales.count == groupCount, code.highScales.count == groupCount else {
        throw TurboQuantError.invalidReferenceCode("TurboQuantProd norm table count mismatch")
    }
    guard code.signs.count >= packedBitByteCount(code.valueCount) else {
        throw TurboQuantError.invalidReferenceCode("TurboQuantProd QJL sign storage is truncated")
    }

    var total = Float(0)
    var bitOffset = 0
    for groupIndex in 0 ..< groupCount {
        let start = groupIndex * code.groupSize
        let end = Swift.min(start + code.groupSize, code.valueCount)
        let count = end - start
        guard count > 0 else { continue }

        let highCount = mixedPrecisionHighCount(
            valueCount: count,
            baseBits: code.baseMagnitudeBits,
            highBits: code.highMagnitudeBits,
            targetBits: Swift.max(1, code.preset.targetMagnitudeBits - 1)
        )
        let highMask = productHighPrecisionMask(
            code: code,
            start: start,
            count: count,
            highCount: highCount,
            groupIndex: groupIndex
        )
        var quantizedRotated = Array(repeating: Float(0), count: count)
        for localIndex in 0 ..< count {
            let bits = highMask[localIndex] ? code.highMagnitudeBits : code.baseMagnitudeBits
            let codebook = turboQuantLloydMaxCodebook(
                bits: bits,
                coordinateStdDev: 1 / sqrt(Float(count))
            )
            let codeIndex = Int(
                try readPackedBits(
                    code.packedMagnitudes,
                    bitOffset: &bitOffset,
                    bitCount: bits
                )
            )
            guard codeIndex < codebook.count else {
                throw TurboQuantError.invalidReferenceCode("TurboQuantProd codebook index overflow")
            }
            quantizedRotated[localIndex] = codebook[codeIndex]
        }

        let queryRotated = applyTurboQuantRotation(
            Array(query[start ..< end]),
            seed: code.seed,
            groupIndex: groupIndex,
            inverse: false
        )
        let norm = code.baseScales[groupIndex]
        for localIndex in 0 ..< count {
            total += norm * quantizedRotated[localIndex] * queryRotated[localIndex]
        }

        let residualNorm = code.highScales[groupIndex]
        if residualNorm > 0 {
            var signDot = Float(0)
            for localIndex in 0 ..< count {
                let sign: Float =
                    getPackedBit(code.signs, index: start + localIndex) ? -1 : 1
                signDot += sign * queryRotated[localIndex]
            }
            total += residualNorm * sqrt(Float.pi / (2 * Float(count))) * signDot
        }
    }
    return total
}

private func turboQuantPolarWHTValuesPerWord(bits: Int) throws -> Int {
    switch bits {
    case 1:
        return 32
    case 2:
        return 16
    case 3:
        return 10
    case 4:
        return 8
    default:
        throw TurboQuantError.invalidReferenceCode(
            "PolarWHT packing supports bit widths 1...4, got \(bits)"
        )
    }
}

private func turboQuantPolarWHTReferenceEncode(
    values: [Float],
    shape: [Int],
    bits: Int,
    seed: UInt64,
    headDimension requestedHeadDimension: Int?
) throws -> TurboQuantPolarWHTReferenceCode {
    let expectedCount = shape.reduce(1, *)
    guard expectedCount == values.count else {
        throw TurboQuantError.invalidReferenceCode(
            "shape \(shape) contains \(expectedCount) values but input has \(values.count)"
        )
    }
    let headDimension = requestedHeadDimension ?? shape.last ?? values.count
    guard headDimension > 0, isPowerOfTwo(headDimension) else {
        throw TurboQuantError.invalidReferenceCode(
            "PolarWHT head dimension must be a positive power of two, got \(headDimension)"
        )
    }
    guard values.count % headDimension == 0 else {
        throw TurboQuantError.invalidReferenceCode(
            "PolarWHT value count \(values.count) is not divisible by head dimension \(headDimension)"
        )
    }

    let centroids = try turboQuantPolarWHTCentroids(bits: bits)
    let boundaries = try turboQuantPolarWHTBoundaries(bits: bits)
    let signs = try turboQuantPolarWHTSigns(dimension: headDimension, seed: seed)
    let vectorCount = values.count / headDimension
    let packedWordsPerVector = try turboQuantPolarWHTPackedWordCount(
        dimension: headDimension,
        bits: bits
    )
    var norms = [Float](repeating: 0, count: vectorCount)
    var packedIndices = [UInt32]()
    packedIndices.reserveCapacity(vectorCount * packedWordsPerVector)
    let coordinateScale = sqrt(Float(headDimension))

    for vectorIndex in 0 ..< vectorCount {
        let base = vectorIndex * headDimension
        var normSquared = Float(0)
        for dimensionIndex in 0 ..< headDimension {
            let value = values[base + dimensionIndex]
            normSquared += value * value
        }
        let norm = sqrt(normSquared)
        norms[vectorIndex] = norm

        var unit = [Float](repeating: 0, count: headDimension)
        if norm > Float.leastNonzeroMagnitude {
            for dimensionIndex in 0 ..< headDimension {
                unit[dimensionIndex] = values[base + dimensionIndex] / norm
            }
        }
        let signedUnit = zip(unit, signs).map { $0 * $1 }
        let rotated = try turboQuantPolarWHT(signedUnit)
        var vectorIndices = [UInt8]()
        vectorIndices.reserveCapacity(headDimension)
        for value in rotated {
            let scaled = value * coordinateScale
            var index = UInt8(0)
            for boundary in boundaries where scaled > boundary {
                index &+= 1
            }
            vectorIndices.append(index)
        }
        packedIndices += try turboQuantPolarWHTPackIndices(vectorIndices, bits: bits)
    }

    let expectedPackedWords = vectorCount * packedWordsPerVector
    guard packedIndices.count == expectedPackedWords else {
        throw TurboQuantError.invalidReferenceCode(
            "PolarWHT packed word count \(packedIndices.count), expected \(expectedPackedWords)"
        )
    }

    return TurboQuantPolarWHTReferenceCode(
        shape: shape,
        bits: bits,
        headDimension: headDimension,
        seed: seed,
        valueCount: values.count,
        vectorCount: vectorCount,
        packedWordsPerVector: packedWordsPerVector,
        centroids: centroids,
        boundaries: boundaries,
        signs: signs,
        norms: norms,
        packedIndices: packedIndices
    )
}

private func turboQuantPolarWHTReferenceDecodeValues(
    _ code: TurboQuantPolarWHTReferenceCode
) throws -> [Float] {
    try validateTurboQuantPolarWHTCode(code)
    let indices = try turboQuantPolarWHTReferenceUnpackedIndices(code)
    let centroidScale = 1 / sqrt(Float(code.headDimension))
    var values = [Float](repeating: 0, count: code.valueCount)
    for vectorIndex in 0 ..< code.vectorCount {
        let base = vectorIndex * code.headDimension
        var rotated = [Float](repeating: 0, count: code.headDimension)
        for dimensionIndex in 0 ..< code.headDimension {
            rotated[dimensionIndex] =
                code.centroids[Int(indices[base + dimensionIndex])] * centroidScale
        }
        let inverseRotated = try turboQuantPolarWHT(rotated)
        let norm = code.norms[vectorIndex]
        for dimensionIndex in 0 ..< code.headDimension {
            values[base + dimensionIndex] =
                inverseRotated[dimensionIndex] * code.signs[dimensionIndex] * norm
        }
    }
    return values
}

private func turboQuantPolarWHTReferenceUnpackedIndices(
    _ code: TurboQuantPolarWHTReferenceCode
) throws -> [UInt8] {
    try validateTurboQuantPolarWHTCode(code)
    var indices = [UInt8]()
    indices.reserveCapacity(code.valueCount)
    for vectorIndex in 0 ..< code.vectorCount {
        let start = vectorIndex * code.packedWordsPerVector
        let end = start + code.packedWordsPerVector
        indices += try turboQuantPolarWHTUnpackIndices(
            Array(code.packedIndices[start ..< end]),
            bits: code.bits,
            count: code.headDimension
        )
    }
    return indices
}

private func validatePolarWHTAttentionValueCode(
    _ code: TurboQuantPolarWHTAttentionValueCode
) throws {
    guard code.layout.batchSize > 0,
        code.layout.kvHeadCount > 0,
        code.layout.capacity >= 0,
        code.layout.logicalLength >= 0,
        code.layout.logicalLength <= code.layout.capacity
    else {
        throw TurboQuantError.invalidMetalConfiguration(
            "PolarWHT attention value layout has invalid batch/head/capacity metadata"
        )
    }
    guard code.layout.ringOffset >= 0 else {
        throw TurboQuantError.invalidMetalConfiguration(
            "PolarWHT attention value ring offset cannot be negative"
        )
    }
    guard code.layout.pinnedPrefixLength >= 0,
        code.layout.pinnedPrefixLength <= code.layout.capacity,
        code.layout.pinnedPrefixLength <= code.layout.logicalLength
    else {
        throw TurboQuantError.invalidMetalConfiguration(
            "PolarWHT attention value pinned prefix is outside cache layout"
        )
    }
    let ringCapacity = code.layout.capacity - code.layout.pinnedPrefixLength
    if ringCapacity == 0 {
        guard code.layout.ringOffset == 0 else {
            throw TurboQuantError.invalidMetalConfiguration(
                "PolarWHT attention value ring offset must be zero without ring capacity"
            )
        }
    } else {
        guard code.layout.ringOffset < ringCapacity else {
            throw TurboQuantError.invalidMetalConfiguration(
                "PolarWHT attention value ring offset is outside rotating region"
            )
        }
    }
    guard code.layout.logicalLength == 0 || code.layout.capacity > 0 else {
        throw TurboQuantError.invalidMetalConfiguration(
            "PolarWHT attention value capacity must be positive for non-empty layouts"
        )
    }
    guard code.layout.headDimension > 0, isPowerOfTwo(code.layout.headDimension) else {
        throw TurboQuantError.invalidMetalConfiguration(
            "PolarWHT attention head dimension must be a positive power of two"
        )
    }
    guard code.layout.groupsPerVector == 1 else {
        throw TurboQuantError.invalidMetalConfiguration(
            "PolarWHT attention values store one packed WHT vector per token"
        )
    }
    let expectedWords = try turboQuantPolarWHTPackedWordCount(
        dimension: code.layout.headDimension,
        bits: code.bits
    )
    guard code.packedWordsPerVector == expectedWords,
        code.layout.magnitudeWordsPerGroup == expectedWords
    else {
        throw TurboQuantError.invalidMetalConfiguration(
            "PolarWHT packed words per vector \(code.packedWordsPerVector), expected \(expectedWords)"
        )
    }
    guard code.layout.bitsetWordsPerGroup == 0 else {
        throw TurboQuantError.invalidMetalConfiguration(
            "PolarWHT attention values do not store affine bitset planes"
        )
    }
    _ = try turboQuantPolarWHTCentroids(bits: code.bits)
    try validateStorageArray(
        code.packedIndices,
        name: "PolarWHT attention packed centroid indices",
        expectedShape: code.packedIndexShape,
        expectedDType: .uint32
    )
    try validateStorageArray(
        code.norms,
        name: "PolarWHT attention vector norms",
        expectedShape: code.normShape,
        expectedDTypes: [.float32, .float16]
    )
}

private func validateTurboQuantPolarWHTCode(_ code: TurboQuantPolarWHTReferenceCode) throws {
    guard code.shape.reduce(1, *) == code.valueCount else {
        throw TurboQuantError.invalidReferenceCode(
            "PolarWHT shape \(code.shape) does not match value count \(code.valueCount)"
        )
    }
    guard code.headDimension > 0, isPowerOfTwo(code.headDimension) else {
        throw TurboQuantError.invalidReferenceCode(
            "PolarWHT head dimension must be a positive power of two"
        )
    }
    guard code.valueCount == code.vectorCount * code.headDimension else {
        throw TurboQuantError.invalidReferenceCode("PolarWHT vector metadata is inconsistent")
    }
    let expectedWordsPerVector = try turboQuantPolarWHTPackedWordCount(
        dimension: code.headDimension,
        bits: code.bits
    )
    guard code.packedWordsPerVector == expectedWordsPerVector else {
        throw TurboQuantError.invalidReferenceCode(
            "PolarWHT packed words per vector \(code.packedWordsPerVector), expected \(expectedWordsPerVector)"
        )
    }
    guard code.packedIndices.count == code.vectorCount * code.packedWordsPerVector else {
        throw TurboQuantError.invalidReferenceCode("PolarWHT packed index storage count mismatch")
    }
    guard code.norms.count == code.vectorCount else {
        throw TurboQuantError.invalidReferenceCode("PolarWHT norm table count mismatch")
    }
    guard code.signs.count == code.headDimension else {
        throw TurboQuantError.invalidReferenceCode("PolarWHT sign table count mismatch")
    }
    let expectedCentroids = try turboQuantPolarWHTCentroids(bits: code.bits)
    guard code.centroids.count == expectedCentroids.count else {
        throw TurboQuantError.invalidReferenceCode("PolarWHT centroid table count mismatch")
    }
    guard code.boundaries.count == code.centroids.count - 1 else {
        throw TurboQuantError.invalidReferenceCode("PolarWHT boundary table count mismatch")
    }
}

private func turboQuantQuality(
    original: [Float],
    decoded: [Float],
    seed: UInt64,
    thresholds: TurboQuantQualityThresholds
) throws -> TurboQuantQualityReport {
    guard !original.isEmpty else {
        throw TurboQuantError.invalidQualityInput("quality input must not be empty")
    }
    guard original.count == decoded.count else {
        throw TurboQuantError.invalidQualityInput("original and decoded counts differ")
    }

    var squaredError = Float(0)
    var squaredSignal = Float(0)
    var maxAbsoluteError = Float(0)
    var dot = Float(0)
    var originalNormSquared = Float(0)
    var decodedNormSquared = Float(0)
    var probeOriginalDot = Float(0)
    var probeDecodedDot = Float(0)

    for index in original.indices {
        let lhs = original[index]
        let rhs = decoded[index]
        let delta = lhs - rhs
        squaredError += delta * delta
        squaredSignal += lhs * lhs
        maxAbsoluteError = Swift.max(maxAbsoluteError, Swift.abs(delta))
        dot += lhs * rhs
        originalNormSquared += lhs * lhs
        decodedNormSquared += rhs * rhs

        let probe = deterministicProbeValue(index: index, seed: seed)
        probeOriginalDot += probe * lhs
        probeDecodedDot += probe * rhs
    }

    let count = Float(original.count)
    let mse = squaredError / count
    let relativeMSE = squaredError / Swift.max(squaredSignal, Float.leastNonzeroMagnitude)
    let cosineDenominator = sqrt(originalNormSquared) * sqrt(decodedNormSquared)
    let cosineSimilarity = dot / Swift.max(cosineDenominator, Float.leastNonzeroMagnitude)
    let innerProductRelativeError =
        Swift.abs(probeOriginalDot - probeDecodedDot)
        / Swift.max(Swift.abs(probeOriginalDot), Float.leastNonzeroMagnitude)

    return TurboQuantQualityReport(
        mse: mse,
        relativeMSE: relativeMSE,
        maxAbsoluteError: maxAbsoluteError,
        cosineSimilarity: cosineSimilarity,
        innerProductRelativeError: innerProductRelativeError,
        thresholds: thresholds
    )
}

private func deterministicProbeValue(index: Int, seed: UInt64) -> Float {
    var state = seed ^ 0xD1B5_4A32_D192_ED03
    state &+= UInt64(index) &* 0x9E37_79B9_7F4A_7C15
    state ^= state >> 30
    state &*= 0xBF58_476D_1CE4_E5B9
    state ^= state >> 27
    state &*= 0x94D0_49BB_1331_11EB
    state ^= state >> 31
    let unit = Float(UInt32(truncatingIfNeeded: state)) / Float(UInt32.max)
    return unit * 2 - 1
}

private func mixedPrecisionHighCount(
    valueCount: Int,
    baseBits: Int,
    highBits: Int,
    targetBits: Float
) -> Int {
    guard highBits > baseBits else { return 0 }
    let fraction = (targetBits - Float(baseBits)) / Float(highBits - baseBits)
    let clampedFraction = Swift.max(0, Swift.min(1, fraction))
    return Int((Float(valueCount) * clampedFraction).rounded())
}

private func mixedPrecisionHighFraction(
    preset: TurboQuantPreset,
    denominator: Int = 1000
) -> (numerator: Int, denominator: Int) {
    let baseBits = Swift.max(1, preset.baseMagnitudeBits - 1)
    let highBits = Swift.max(baseBits, preset.highMagnitudeBits - 1)
    guard highBits > baseBits else { return (0, 1) }
    let targetBits = Swift.max(1, preset.targetMagnitudeBits - 1)
    let fraction = (targetBits - Float(baseBits)) / Float(highBits - baseBits)
    let clamped = Swift.max(0, Swift.min(1, fraction))
    return (Int((clamped * Float(denominator)).rounded()), denominator)
}

private func validateTurboQuantValueBits(_ bits: Int) throws {
    guard (2 ... 8).contains(bits) else {
        throw TurboQuantError.invalidReferenceCode(
            "TurboQuant value bits must be in 2...8, got \(bits)"
        )
    }
}

private func productHighPrecisionMask(
    code: TurboQuantReferenceCode,
    start: Int,
    count: Int,
    highCount: Int,
    groupIndex: Int
) -> [Bool] {
    if code.highPrecisionMask.count >= packedBitByteCount(code.valueCount) {
        return (0 ..< count).map { localIndex in
            getPackedBit(code.highPrecisionMask, index: start + localIndex)
        }
    }
    return productHighPrecisionMask(
        valueCount: count,
        highCount: highCount,
        seed: code.seed,
        groupIndex: groupIndex
    )
}

private func splitHighPrecisionMask(
    valueCount: Int,
    highCount: Int
) -> [Bool] {
    guard highCount > 0 else { return Array(repeating: false, count: valueCount) }
    guard highCount < valueCount else { return Array(repeating: true, count: valueCount) }
    return (0 ..< valueCount).map { $0 < highCount }
}

private func productHighPrecisionMask(
    valueCount: Int,
    highCount: Int,
    seed: UInt64,
    groupIndex: Int
) -> [Bool] {
    guard highCount > 0 else { return Array(repeating: false, count: valueCount) }
    guard highCount < valueCount else { return Array(repeating: true, count: valueCount) }

    let ranked = (0 ..< valueCount).sorted { lhs, rhs in
        let lhsRank = productChannelRank(seed: seed, groupIndex: groupIndex, localIndex: lhs)
        let rhsRank = productChannelRank(seed: seed, groupIndex: groupIndex, localIndex: rhs)
        if lhsRank == rhsRank {
            return lhs < rhs
        }
        return lhsRank < rhsRank
    }
    var mask = Array(repeating: false, count: valueCount)
    for index in ranked.prefix(highCount) {
        mask[index] = true
    }
    return mask
}

private func productChannelRank(seed: UInt64, groupIndex: Int, localIndex: Int) -> UInt64 {
    var state = seed
    state ^= UInt64(groupIndex) &* 0x9E37_79B9_7F4A_7C15
    state &+= UInt64(localIndex) &* 0xD1B5_4A32_D192_ED03
    state ^= state >> 30
    state &*= 0xBF58_476D_1CE4_E5B9
    state ^= state >> 27
    state &*= 0x94D0_49BB_1331_11EB
    state ^= state >> 31
    return state
}

private func turboQuantLloydMaxCodebook(bits: Int, coordinateStdDev: Float) -> [Float] {
    let levelCount = Swift.max(2, 1 << bits)
    let sigma = Swift.max(Double(coordinateStdDev), Double(Float.leastNonzeroMagnitude))
    var levels = (0 ..< levelCount).map { index -> Double in
        let centered = (Double(index) + 0.5) / Double(levelCount) * 2 - 1
        return centered * 2.5 * sigma
    }

    for _ in 0 ..< 16 {
        var boundaries = Array(repeating: -Double.infinity, count: levelCount + 1)
        boundaries[levelCount] = Double.infinity
        if levelCount > 1 {
            for index in 1 ..< levelCount {
                boundaries[index] = (levels[index - 1] + levels[index]) * 0.5
            }
        }

        for index in 0 ..< levelCount {
            let lower = boundaries[index] / sigma
            let upper = boundaries[index + 1] / sigma
            let probability = normalCDF(upper) - normalCDF(lower)
            if probability > 1e-12 {
                levels[index] = sigma * (normalPDF(lower) - normalPDF(upper)) / probability
            }
        }
    }

    return levels.map(Float.init)
}

private func nearestCodebookIndex(_ value: Float, codebook: [Float]) -> Int {
    var bestIndex = 0
    var bestDistance = Float.greatestFiniteMagnitude
    for (index, level) in codebook.enumerated() {
        let distance = Swift.abs(value - level)
        if distance < bestDistance {
            bestDistance = distance
            bestIndex = index
        }
    }
    return bestIndex
}

private func normalPDF(_ x: Double) -> Double {
    guard x.isFinite else { return 0 }
    return exp(-0.5 * x * x) / sqrt(2 * Double.pi)
}

private func normalCDF(_ x: Double) -> Double {
    if x == Double.infinity { return 1 }
    if x == -Double.infinity { return 0 }
    return 0.5 * (1 + erf(x / sqrt(2)))
}

private func applyTurboQuantRotation(
    _ values: [Float],
    seed: UInt64,
    groupIndex: Int,
    inverse: Bool
) -> [Float] {
    guard values.count > 1 else {
        return values.enumerated().map { localIndex, value in
            randomSign(index: groupIndex &* 4099 &+ localIndex, seed: seed) ? -value : value
        }
    }
    if isPowerOfTwo(values.count) {
        return applyRandomizedHadamardRotation(
            values,
            seed: seed,
            groupIndex: groupIndex,
            inverse: inverse
        )
    }
    return applyDeterministicGivensRotation(
        values,
        seed: seed,
        groupIndex: groupIndex,
        inverse: inverse
    )
}

private func isPowerOfTwo(_ value: Int) -> Bool {
    value > 0 && (value & (value - 1)) == 0
}

private func applyRandomizedHadamardRotation(
    _ values: [Float],
    seed: UInt64,
    groupIndex: Int,
    inverse: Bool
) -> [Float] {
    var result = values
    if inverse {
        fastHadamardTransform(&result)
        applyRotationSigns(&result, seed: seed, groupIndex: groupIndex)
    } else {
        applyRotationSigns(&result, seed: seed, groupIndex: groupIndex)
        fastHadamardTransform(&result)
    }
    let scale = 1 / sqrt(Float(values.count))
    for index in result.indices {
        result[index] *= scale
    }
    return result
}

private func fastHadamardTransform(_ values: inout [Float]) {
    var width = 1
    while width < values.count {
        var start = 0
        while start < values.count {
            for offset in 0 ..< width {
                let lhs = values[start + offset]
                let rhs = values[start + offset + width]
                values[start + offset] = lhs + rhs
                values[start + offset + width] = lhs - rhs
            }
            start += width * 2
        }
        width *= 2
    }
}

private func applyRotationSigns(_ values: inout [Float], seed: UInt64, groupIndex: Int) {
    for index in values.indices {
        if randomSign(index: groupIndex &* 4099 &+ index, seed: seed) {
            values[index] = -values[index]
        }
    }
}

private func applyDeterministicGivensRotation(
    _ values: [Float],
    seed: UInt64,
    groupIndex: Int,
    inverse: Bool
) -> [Float] {
    var result = values
    let passes = Array(0 ..< 4)
    let orderedPasses = inverse ? Array(passes.reversed()) : passes
    for pass in orderedPasses {
        let offset = pass % 2
        var index = offset
        while index + 1 < result.count {
            let angle =
                deterministicRotationAngle(
                    seed: seed,
                    groupIndex: groupIndex,
                    pass: pass,
                    pairIndex: index / 2
                ) * (inverse ? -1 : 1)
            let c = cos(angle)
            let s = sin(angle)
            let lhs = result[index]
            let rhs = result[index + 1]
            result[index] = c * lhs - s * rhs
            result[index + 1] = s * lhs + c * rhs
            index += 2
        }
    }
    return result
}

private func deterministicRotationAngle(
    seed: UInt64,
    groupIndex: Int,
    pass: Int,
    pairIndex: Int
) -> Float {
    let rank = productChannelRank(
        seed: seed ^ (UInt64(pass) &* 0xA24B_AED4_963E_E407),
        groupIndex: groupIndex,
        localIndex: pairIndex
    )
    let unit = Float(UInt32(truncatingIfNeeded: rank)) / Float(UInt32.max)
    return (unit - 0.5) * Float.pi
}

private func packedBitByteCount(_ bitCount: Int) -> Int {
    (bitCount + 7) / 8
}

private func setPackedBit(_ bytes: inout [UInt8], index: Int, value: Bool) {
    guard value else { return }
    let byteIndex = index / 8
    let bitIndex = index % 8
    bytes[byteIndex] |= UInt8(1 << bitIndex)
}

private func getPackedBit(_ data: Data, index: Int) -> Bool {
    let byteIndex = index / 8
    let bitIndex = index % 8
    guard byteIndex < data.count else { return false }
    return (data[byteIndex] & UInt8(1 << bitIndex)) != 0
}

private func appendPackedBits(
    _ value: UInt32,
    bitCount: Int,
    bytes: inout [UInt8],
    bitOffset: inout Int
) {
    for localBit in 0 ..< bitCount {
        if bitOffset / 8 == bytes.count {
            bytes.append(0)
        }
        let bitSet = (value & (1 << UInt32(localBit))) != 0
        if bitSet {
            bytes[bitOffset / 8] |= UInt8(1 << (bitOffset % 8))
        }
        bitOffset += 1
    }
}

private func readPackedBits(
    _ data: Data,
    bitOffset: inout Int,
    bitCount: Int
) throws -> UInt32 {
    var value: UInt32 = 0
    for localBit in 0 ..< bitCount {
        let byteIndex = bitOffset / 8
        guard byteIndex < data.count else {
            throw TurboQuantError.invalidReferenceCode("packed magnitude storage is truncated")
        }
        if (data[byteIndex] & UInt8(1 << (bitOffset % 8))) != 0 {
            value |= 1 << UInt32(localBit)
        }
        bitOffset += 1
    }
    return value
}

private func preconditionedValue(_ value: Float, index: Int, seed: UInt64) -> Float {
    randomSign(index: index, seed: seed) ? -value : value
}

private func unpreconditionedValue(_ value: Float, index: Int, seed: UInt64) -> Float {
    randomSign(index: index, seed: seed) ? -value : value
}

private func randomSign(index: Int, seed: UInt64) -> Bool {
    var state = seed &+ UInt64(index) &* 0x9E37_79B9_7F4A_7C15
    state ^= state >> 30
    state &*= 0xBF58_476D_1CE4_E5B9
    state ^= state >> 27
    state &*= 0x94D0_49BB_1331_11EB
    state ^= state >> 31
    return (state & 1) == 1
}

private func metalTemplateSeedWords(
    prefix: String,
    value: UInt64
) -> [(String, any KernelTemplateArg)] {
    [
        ("\(prefix)_3", Int((value >> 48) & 0xFFFF)),
        ("\(prefix)_2", Int((value >> 32) & 0xFFFF)),
        ("\(prefix)_1", Int((value >> 16) & 0xFFFF)),
        ("\(prefix)_0", Int(value & 0xFFFF)),
    ]
}

// TQ_T11: process-lifetime cache for the Metal-runtime probe. Device presence and
// bundled-metallib presence cannot change within a process; the fs scan was measured
// as a per-attention-call cost by TQ_HOST_PROBE. TURBOQUANT_DISABLE_HOST_CACHES=1
// restores the uncached behavior for diagnostics.
let turboQuantHostCachesDisabled: Bool =
    ProcessInfo.processInfo.environment["TURBOQUANT_DISABLE_HOST_CACHES"] == "1"

// TQ_T11: validateStorageArray's eval() only forced early materialization of metadata
// (shape/dtype/nbytes/contiguousToDimension) that is readable on an unevaluated MLXArray.
// Structural (no-eval) tier is now the default for every caller; TURBOQUANT_DEEP_VALIDATE=1
// restores the eval for diagnostics.
let turboQuantDeepValidationEnabled: Bool =
    ProcessInfo.processInfo.environment["TURBOQUANT_DEEP_VALIDATE"] == "1"

private final class TurboQuantMetalRuntimeAvailabilityCache: @unchecked Sendable {
    static let shared = TurboQuantMetalRuntimeAvailabilityCache()
    private let lock = NSLock()
    private var cachedResult: Bool?
    private init() {}
    func result(_ compute: () -> Bool) -> Bool {
        if turboQuantHostCachesDisabled { return compute() }
        lock.lock()
        if let cachedResult { lock.unlock(); return cachedResult }
        lock.unlock()
        let result = compute()
        lock.lock(); cachedResult = result; lock.unlock()
        return result
    }
    func resetForTesting() { lock.lock(); cachedResult = nil; lock.unlock() }
}

// TQ_HOST_PROBE_V1: env-gated host-overhead probe (TQ_HOST_PROBE=1).
// Zero behavior change when the variable is unset.
private final class TurboQuantHostProbe: @unchecked Sendable {
    static let shared = TurboQuantHostProbe()
    static let enabled: Bool = {
        let on = ProcessInfo.processInfo.environment["TQ_HOST_PROBE"] == "1"
        if on { atexit { TurboQuantHostProbe.shared.emit(reason: "atexit") } }
        return on
    }()
    private let lock = NSLock()
    private var attentionCalls = 0
    private var metalRuntimeAvailableCalls = 0
    private var metalRuntimeAvailableNanos: UInt64 = 0
    private var availabilityRebuilds = 0
    private var availabilityNanos: UInt64 = 0
    private var validateEvalCalls = 0
    private var validateEvalNanos: UInt64 = 0
    func recordAttentionCall() {
        guard Self.enabled else { return }
        lock.lock()
        attentionCalls += 1
        let emitNow = attentionCalls % 1000 == 0
        lock.unlock()
        if emitNow { emit(reason: "periodic") }
    }
    func recordMetalRuntimeAvailable(nanos: UInt64) {
        guard Self.enabled else { return }
        lock.lock(); metalRuntimeAvailableCalls += 1; metalRuntimeAvailableNanos += nanos; lock.unlock()
    }
    func recordAvailabilityRebuild(nanos: UInt64) {
        guard Self.enabled else { return }
        lock.lock(); availabilityRebuilds += 1; availabilityNanos += nanos; lock.unlock()
    }
    func recordValidateEval(nanos: UInt64) {
        guard Self.enabled else { return }
        lock.lock(); validateEvalCalls += 1; validateEvalNanos += nanos; lock.unlock()
    }
    func emit(reason: String) {
        guard Self.enabled else { return }
        lock.lock()
        let line = "TQ_HOST_PROBE_V1 swift reason=\(reason)"
            + " attention_calls=\(attentionCalls)"
            + " metal_runtime_available_calls=\(metalRuntimeAvailableCalls) metal_runtime_available_ns=\(metalRuntimeAvailableNanos)"
            + " availability_rebuilds=\(availabilityRebuilds) availability_ns=\(availabilityNanos)"
            + " validate_eval_calls=\(validateEvalCalls) validate_eval_ns=\(validateEvalNanos)\n"
        lock.unlock()
        FileHandle.standardError.write(Data(line.utf8))
    }
}

// TQ_KERNEL_TRACE_V1: env-gated dispatched-fused-kernel trace (TQ_KERNEL_TRACE=1).
// Zero behavior change when unset. Diagnostic observability only; not a kernel change.
private final class TurboQuantKernelDispatchTrace: @unchecked Sendable {
    static let shared = TurboQuantKernelDispatchTrace()
    static let enabled: Bool = {
        let on = ProcessInfo.processInfo.environment["TQ_KERNEL_TRACE"] == "1"
        if on { atexit { TurboQuantKernelDispatchTrace.shared.emit() } }
        return on
    }()
    private let lock = NSLock()
    private var counts: [String: Int] = [:]
    // Counting is ALWAYS-ON (unconditional) so engagement telemetry accumulates
    // without TQ_KERNEL_TRACE. Only the atexit/stderr emit path stays env-gated.
    func record(_ name: String) {
        lock.lock(); counts[name, default: 0] += 1; lock.unlock()
    }
    func snapshot() -> [String: Int] {
        lock.lock(); defer { lock.unlock() }
        return counts
    }
    func reset() {
        lock.lock(); counts.removeAll(); lock.unlock()
    }
    func emit() {
        guard Self.enabled else { return }
        lock.lock()
        let lines = counts.sorted { $0.key < $1.key }
            .map { "TQ_KERNEL_TRACE_V1 kernel=\($0.key) count=\($0.value)\n" }.joined()
        lock.unlock()
        FileHandle.standardError.write(Data(lines.utf8))
    }
}

/// Public façade over the always-on dispatched-kernel counters. Benchmarks call
/// `reset()` before a timed decode loop and `snapshot()` after to prove which
/// native TurboQuant kernel family actually dispatched (engagement verification).
public enum TurboQuantKernelDispatchTelemetry {
    public static func snapshot() -> [String: Int] { TurboQuantKernelDispatchTrace.shared.snapshot() }
    public static func reset() { TurboQuantKernelDispatchTrace.shared.reset() }
}

private func metalRuntimeAvailable() -> Bool {
    guard TurboQuantHostProbe.enabled else { return metalRuntimeAvailableUnprobed() }
    let start = DispatchTime.now().uptimeNanoseconds
    let result = metalRuntimeAvailableUnprobed()
    TurboQuantHostProbe.shared.recordMetalRuntimeAvailable(
        nanos: DispatchTime.now().uptimeNanoseconds - start)
    return result
}
private func metalRuntimeAvailableUnprobed() -> Bool {
    TurboQuantMetalRuntimeAvailabilityCache.shared.result { metalRuntimeAvailableComputed() }
}
private func metalRuntimeAvailableComputed() -> Bool {
    #if canImport(Metal)
        guard MTLCreateSystemDefaultDevice() != nil else { return false }
    #endif
    return metalLibraryResourceAvailable()
}

private func metalLibraryResourceAvailable() -> Bool {
    let fileManager = FileManager.default
    var candidates: [URL] = []

    if let executablePath = CommandLine.arguments.first, !executablePath.isEmpty {
        let executableDirectory = URL(fileURLWithPath: executablePath).deletingLastPathComponent()
        candidates.append(executableDirectory.appendingPathComponent("mlx.metallib"))
        candidates.append(executableDirectory.appendingPathComponent("default.metallib"))
        candidates.append(executableDirectory.appendingPathComponent("Resources/mlx.metallib"))
        candidates.append(executableDirectory.appendingPathComponent("Resources/default.metallib"))
        appendSwiftPMMetalBundleCandidates(from: executableDirectory, to: &candidates)
    }

    if let executableDirectory = Bundle.main.executableURL?.deletingLastPathComponent() {
        appendSwiftPMMetalBundleCandidates(from: executableDirectory, to: &candidates)
    }

    let currentDirectory = URL(fileURLWithPath: fileManager.currentDirectoryPath)
    candidates.append(currentDirectory.appendingPathComponent("mlx.metallib"))
    candidates.append(currentDirectory.appendingPathComponent("default.metallib"))

    for bundle in [Bundle.main] + Bundle.allBundles {
        if bundle.url(forResource: "default", withExtension: "metallib") != nil
            || bundle.url(forResource: "mlx", withExtension: "metallib") != nil
        {
            return true
        }
        appendSwiftPMMetalBundleCandidates(from: bundle.bundleURL, to: &candidates)
        if let resourceURL = bundle.resourceURL {
            candidates.append(resourceURL.appendingPathComponent("default.metallib"))
            candidates.append(resourceURL.appendingPathComponent("mlx.metallib"))
            candidates.append(
                resourceURL.appendingPathComponent("mlx-swift_Cmlx.bundle/default.metallib"))
            candidates.append(
                resourceURL.appendingPathComponent("mlx-swift_Cmlx.bundle/mlx.metallib"))
            appendSwiftPMMetalBundleCandidates(from: resourceURL, to: &candidates)
        }
    }

    return candidates.contains { fileManager.fileExists(atPath: $0.path) }
}

private func appendSwiftPMMetalBundleCandidates(from directory: URL, to candidates: inout [URL]) {
    var root = directory
    for _ in 0 ..< 5 {
        candidates.append(root.appendingPathComponent("mlx-swift_Cmlx.bundle/default.metallib"))
        candidates.append(root.appendingPathComponent("mlx-swift_Cmlx.bundle/mlx.metallib"))

        let parent = root.deletingLastPathComponent()
        guard parent.path != root.path else { break }
        root = parent
    }
}

private func detectedTurboQuantDeviceCapabilities() -> TurboQuantDeviceCapabilities {
    let metalAvailable = metalRuntimeAvailable()
    let physicalMemory = Int(ProcessInfo.processInfo.physicalMemory)
    let hardwareModelIdentifier = turboQuantHardwareModelIdentifier()

    #if canImport(Metal)
        if let device = MTLCreateSystemDefaultDevice() {
            let architecture: String
            if #available(macOS 14.0, iOS 17.0, tvOS 17.0, *) {
                architecture = device.architecture.name
            } else {
                architecture = device.name
            }

            let recommendedWorkingSet: Int?
            if device.recommendedMaxWorkingSetSize > UInt64(Int.max) {
                recommendedWorkingSet = Int.max
            } else if device.recommendedMaxWorkingSetSize > 0 {
                recommendedWorkingSet = Int(device.recommendedMaxWorkingSetSize)
            } else {
                recommendedWorkingSet = nil
            }

            return TurboQuantDeviceCapabilities(
                metalAvailable: metalAvailable,
                architectureName: architecture,
                hardwareModelIdentifier: hardwareModelIdentifier,
                supportedGPUFamilies: turboQuantSupportedGPUFamilies(device),
                maxBufferBytes: device.maxBufferLength,
                recommendedWorkingSetBytes: recommendedWorkingSet,
                physicalMemoryBytes: physicalMemory,
                maxThreadgroupWidth: device.maxThreadsPerThreadgroup.width
            )
        }
    #endif

    return TurboQuantDeviceCapabilities(
        metalAvailable: metalAvailable,
        architectureName: "Unknown",
        hardwareModelIdentifier: hardwareModelIdentifier,
        physicalMemoryBytes: physicalMemory
    )
}

private func turboQuantHardwareModelIdentifier() -> String? {
    #if canImport(Darwin)
        var systemInfo = utsname()
        guard uname(&systemInfo) == 0 else { return nil }
        let identifier = Mirror(reflecting: systemInfo.machine).children.reduce(into: "") {
            partial, element in
            guard let value = element.value as? Int8, value != 0 else { return }
            partial.append(String(UnicodeScalar(UInt8(value))))
        }
        return identifier.isEmpty ? nil : identifier
    #else
        return nil
    #endif
}

#if canImport(Metal)
    private func turboQuantSupportedGPUFamilies(_ device: MTLDevice) -> [String: Bool] {
        var families = [
            "apple7": device.supportsFamily(.apple7),
            "apple8": device.supportsFamily(.apple8),
            "apple9": device.supportsFamily(.apple9),
            "apple10": device.supportsFamily(.apple10),
            "mac2": device.supportsFamily(.mac2),
            "metal3": device.supportsFamily(.metal3),
        ]
        #if targetEnvironment(simulator)
            families["metal4"] = false
        #else
            if #available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *) {
                families["metal4"] = device.supportsFamily(.metal4)
            } else {
                families["metal4"] = false
            }
        #endif
        return families
    }
#endif

private func selectTurboQuantKernelProfile(
    architectureName: String,
    hardwareModelIdentifier: String?,
    supportedGPUFamilies: [String: Bool],
    recommendedWorkingSetBytes: Int?
) -> TurboQuantKernelProfile {
    let architecture = architectureName.lowercased()
    let hardwareModel = hardwareModelIdentifier?.lowercased() ?? ""
    let workingSet = recommendedWorkingSetBytes ?? 0

    if let iPhoneGeneration = turboQuantIPhoneGeneration(from: hardwareModel),
        iPhoneGeneration <= 16
    {
        return .portableA16A17
    }

    if supportedGPUFamilies["mac2"] == true
        || hardwareModel == "arm64"
        || architecture.contains("applegpu_g")
        || architecture.contains("mac")
    {
        return .macAppleSilicon
    }

    if workingSet >= 10_000_000_000
        || architecture.contains("a19pro")
        || architecture.contains("a19 pro")
    {
        return .sustainedA19Pro
    }

    if supportedGPUFamilies["apple10"] == true
        || supportedGPUFamilies["apple9"] == true
        || supportedGPUFamilies["apple8"] == true
        || workingSet >= 7_000_000_000
        || architecture.contains("a18")
        || architecture.contains("a19")
    {
        return .wideA18A19
    }

    return .portableA16A17
}

private func turboQuantIPhoneGeneration(from hardwareModel: String) -> Int? {
    guard hardwareModel.hasPrefix("iphone") else { return nil }
    let suffix = hardwareModel.dropFirst("iphone".count)
    let generation = suffix.prefix { $0.isNumber }
    return Int(generation)
}

private func turboQuantExperimentalLinearMetalEnabled() -> Bool {
    ProcessInfo.processInfo.environment["TURBOQUANT_ENABLE_EXPERIMENTAL_LINEAR_METAL"] == "1"
}

private func turboQuantRelativeMSE(_ expected: [Float], _ actual: [Float]) -> Float {
    guard expected.count == actual.count, !expected.isEmpty else {
        return .infinity
    }
    let energy = expected.reduce(Float(0)) { partial, value in
        partial + value * value
    }
    let mse = zip(expected, actual).reduce(Float(0)) { partial, pair in
        let delta = pair.0 - pair.1
        return partial + delta * delta
    }
    return mse / Swift.max(energy, Float.leastNonzeroMagnitude)
}

private func validateAttentionDecisionRequest(_ request: TurboQuantAttentionRequest) throws {
    guard request.queryShape.count == 4 else {
        throw TurboQuantError.invalidMetalConfiguration(
            "attention query shape must have rank 4"
        )
    }
    guard request.queryDType.isFloatingPoint, request.outputDType.isFloatingPoint else {
        throw TurboQuantError.invalidMetalConfiguration(
            "compressed attention query and output dtypes must be floating point"
        )
    }
    guard attentionLayoutsShareSequence(request.keyLayout, request.valueLayout) else {
        throw TurboQuantError.invalidMetalConfiguration(
            "key and value compressed sequence layouts differ"
        )
    }
    guard request.queryShape[0] == request.keyLayout.batchSize,
        request.queryShape[3] == request.keyLayout.headDimension,
        request.queryShape[1] % request.keyLayout.kvHeadCount == 0
    else {
        throw TurboQuantError.invalidMetalConfiguration(
            "query shape \(request.queryShape) is incompatible with compressed attention layout"
        )
    }
    if request.maskKind == .causal {
        guard request.queryShape[2] <= request.keyLayout.logicalLength else {
            throw TurboQuantError.invalidMetalConfiguration(
                "causal compressed attention requires query length \(request.queryShape[2]) <= key length \(request.keyLayout.logicalLength)"
            )
        }
    }
}

private func turboQuantTwoStageAttentionScratchBytes(
    queryShape: [Int],
    keyLength: Int
) -> Int {
    guard queryShape.count == 4 else { return Int.max }
    return queryShape[0] * queryShape[1] * queryShape[2] * keyLength * DType.float32.size
}

private func turboQuantAttentionMaskKind(
    _ mask: MLXFast.ScaledDotProductAttentionMaskMode
) -> TurboQuantAttentionMaskKind {
    switch mask {
    case .none:
        return .none
    case .causal:
        return .causal
    case .array:
        return .materializedArray
    case .arrays(let arrays):
        return arrays.count <= 1 ? .materializedArray : .unsupportedMaterializedArrays
    }
}

private let turboQuantCompactUnusedBitsetShape = [1]

public final class TurboQuantRuntimeProbe: @unchecked Sendable {
    public static let shared = TurboQuantRuntimeProbe()

    private let lock = NSLock()
    private var cachedResult: TurboQuantRuntimeProbeResult?
    private var runningSelfTest = false

    private init() {}

    public static var current: TurboQuantRuntimeProbeResult {
        shared.result()
    }

    public func result() -> TurboQuantRuntimeProbeResult {
        lock.lock()
        if let cachedResult {
            lock.unlock()
            return cachedResult
        }
        lock.unlock()

        let result = run(on: detectedTurboQuantDeviceCapabilities())

        lock.lock()
        cachedResult = result
        lock.unlock()
        return result
    }

    func selectedKernelProfileWithoutRunningProbe() -> TurboQuantKernelProfile {
        lock.lock()
        let cached = cachedResult?.selectedKernelProfile
        lock.unlock()
        if let cached { return cached }

        let capabilities = detectedTurboQuantDeviceCapabilities()
        guard capabilities.metalAvailable else { return .mlxPackedFallback }
        return selectTurboQuantKernelProfile(
            architectureName: capabilities.architectureName,
            hardwareModelIdentifier: capabilities.hardwareModelIdentifier,
            supportedGPUFamilies: capabilities.supportedGPUFamilies,
            recommendedWorkingSetBytes: capabilities.recommendedWorkingSetBytes
        )
    }

    func isRunningSelfTest() -> Bool {
        lock.lock()
        let running = runningSelfTest
        lock.unlock()
        return running
    }

    private func run(on capabilities: TurboQuantDeviceCapabilities) -> TurboQuantRuntimeProbeResult
    {
        guard capabilities.metalAvailable else {
            return TurboQuantRuntimeProbeResult(
                status: .failed,
                metalRuntimeAvailable: false,
                selectedKernelProfile: .mlxPackedFallback,
                failureReason: "Metal runtime or bundled metallib is unavailable."
            )
        }

        let selectedProfile = selectTurboQuantKernelProfile(
            architectureName: capabilities.architectureName,
            hardwareModelIdentifier: capabilities.hardwareModelIdentifier,
            supportedGPUFamilies: capabilities.supportedGPUFamilies,
            recommendedWorkingSetBytes: capabilities.recommendedWorkingSetBytes
        )
        let onlineFusedHeadDimensions = TurboQuantRuntimeProbeResult
            .defaultOnlineFusedHeadDimensions(for: selectedProfile)

        lock.lock()
        runningSelfTest = true
        lock.unlock()
        defer {
            lock.lock()
            runningSelfTest = false
            lock.unlock()
        }

        do {
            let flatKeyValues: [Float] = (0 ..< 128).map { index in
                let position = Double(index)
                return Float(0.42 * sin(position * 0.061) + 0.17 * cos(position * 0.017))
            }
            let flatValueValues: [Float] = (0 ..< 128).map { index in
                let position = Double(index)
                return Float(0.31 * cos(position * 0.049) - 0.12 * sin(position * 0.109))
            }
            let flatKeys = MLXArray(flatKeyValues, [2, 64])
            let flatValues = MLXArray(flatValueValues, [2, 64])
            let flatKeyCode = try turboQuantMetalEncode(
                flatKeys,
                configuration: TurboQuantConfiguration(
                    preset: .turbo3_5,
                    role: .key,
                    groupSize: 64,
                    backend: .metalPolarQJL,
                    seed: 0x5EED_F1A7_0000_0001
                )
            )
            let flatValueCode = try turboQuantMetalEncode(
                flatValues,
                configuration: TurboQuantConfiguration(
                    preset: .turbo3_5,
                    role: .value,
                    groupSize: 64,
                    backend: .metalPolarQJL,
                    seed: 0x5EED_F1A7_0000_0002,
                    valueBits: 4
                )
            )
            let decodedFlatKeys = try turboQuantMetalDecode(flatKeyCode, dtype: .float32)
            let decodedFlatValues = try turboQuantMetalDecode(flatValueCode, dtype: .float32)
            eval(decodedFlatKeys, decodedFlatValues)
            let decodedFlatKeyValues = decodedFlatKeys.asArray(Float.self)
            let decodedFlatValueValues = decodedFlatValues.asArray(Float.self)
            let flatCodecPassed =
                flatKeyCode.shape == flatKeys.shape
                && flatValueCode.shape == flatValues.shape
                && decodedFlatKeys.shape == flatKeys.shape
                && decodedFlatValues.shape == flatValues.shape
                && decodedFlatKeyValues.allSatisfy(\.isFinite)
                && decodedFlatValueValues.allSatisfy(\.isFinite)
                && turboQuantRelativeMSE(flatKeyValues, decodedFlatKeyValues) < 0.2
                && turboQuantRelativeMSE(flatValueValues, decodedFlatValueValues) < 0.02

            let selfTestHeadDimension = onlineFusedHeadDimensions.last ?? 128
            let queryValues: [Float] = (0 ..< (1 * 4 * 2 * selfTestHeadDimension)).map { index in
                let position = Double(index)
                return Float(sin(position * 0.07) + 0.25 * cos(position * 0.013))
            }
            let keyValues: [Float] = (0 ..< (1 * 2 * 5 * selfTestHeadDimension)).map { index in
                let position = Double(index)
                return Float(0.5 * cos(position * 0.05) + 0.1 * sin(position * 0.19))
            }
            let valueValues: [Float] = (0 ..< (1 * 2 * 5 * selfTestHeadDimension)).map { index in
                let position = Double(index)
                return Float(0.35 * sin(position * 0.09) - 0.15 * cos(position * 0.17))
            }
            let queries = MLXArray(queryValues, [1, 4, 2, selfTestHeadDimension])
            let keys = MLXArray(keyValues, [1, 2, 5, selfTestHeadDimension])
            let values = MLXArray(valueValues, [1, 2, 5, selfTestHeadDimension])
            let encodeStart = Date.timeIntervalSinceReferenceDate
            let keyCode = try turboQuantMetalEncodeAttention(
                keys,
                configuration: TurboQuantConfiguration(
                    preset: .turbo3_5,
                    role: .key,
                    groupSize: 64,
                    backend: .metalPolarQJL,
                    seed: 0x5EED_A11C_0000_0001
                )
            )
            let valueCode = try turboQuantMetalEncodeAttention(
                values,
                configuration: TurboQuantConfiguration(
                    preset: .turbo3_5,
                    role: .value,
                    groupSize: 64,
                    backend: .metalPolarQJL,
                    seed: 0x5EED_A11C_0000_0002
                )
            )
            let decodedKeys = try turboQuantMetalDecodeAttention(keyCode, outputDType: .float32)
            let decodedValues = try turboQuantMetalDecodeAttention(valueCode, outputDType: .float32)
            eval(decodedKeys, decodedValues)
            let encodeDecodeLatency = Date.timeIntervalSinceReferenceDate - encodeStart
            let encodeDecodePassed =
                decodedKeys.shape == keys.shape
                && decodedValues.shape == values.shape
                && decodedKeys.asArray(Float.self).allSatisfy(\.isFinite)
                && decodedValues.asArray(Float.self).allSatisfy(\.isFinite)

            let scale = 1 / sqrt(Float(selfTestHeadDimension))
            let reference = MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                mask: .causal
            )
            eval(reference)

            let qk = try turboQuantMetalQK(
                queries: queries,
                keyCode: keyCode,
                scale: scale,
                mask: .causal
            )
            eval(qk)
            let qkPassed =
                qk.shape == [1, 4, 2, 5]
                && qk.asArray(Float.self).allSatisfy(\.isFinite)

            let twoStageStart = Date.timeIntervalSinceReferenceDate
            let weights = softmax(qk.asType(DType.float32), axis: -1)
            let av = try turboQuantMetalAV(
                attentionWeights: weights,
                valueCode: valueCode,
                outputDType: .float32
            )
            eval(av)
            let twoStageLatency = Date.timeIntervalSinceReferenceDate - twoStageStart

            let fusedStart = Date.timeIntervalSinceReferenceDate
            let fused = try turboQuantMetalScaledDotProductAttention(
                queries: queries,
                keyCode: keyCode,
                valueCode: valueCode,
                scale: scale,
                mask: .causal,
                preferOnlineFused: true,
                kernelProfile: selectedProfile
            )
            eval(av, fused)
            let referenceValues = reference.asArray(Float.self)
            let fusedLatency = Date.timeIntervalSinceReferenceDate - fusedStart
            let avValues = av.asArray(Float.self)
            let fusedValues = fused.asArray(Float.self)
            let maxDelta = zip(avValues, fusedValues).reduce(Float(0)) { current, pair in
                Swift.max(current, Swift.abs(pair.0 - pair.1))
            }
            let referenceEnergy = referenceValues.reduce(Float(0)) { partial, value in
                partial + value * value
            }
            let avReferenceRelativeMSE =
                zip(avValues, referenceValues).reduce(Float(0)) {
                    current, pair in
                    let delta = pair.0 - pair.1
                    return current + delta * delta
                } / Swift.max(referenceEnergy, Float.leastNonzeroMagnitude)
            let fusedReferenceRelativeMSE =
                zip(fusedValues, referenceValues).reduce(Float(0)) {
                    current, pair in
                    let delta = pair.0 - pair.1
                    return current + delta * delta
                } / Swift.max(referenceEnergy, Float.leastNonzeroMagnitude)
            let avPassed =
                av.shape == [1, 4, 2, selfTestHeadDimension]
                && avValues.allSatisfy(\.isFinite)
                && avReferenceRelativeMSE < 0.12
            let fusedPassed =
                av.shape == fused.shape && maxDelta < 1e-3
                && fusedReferenceRelativeMSE < 0.12
                && fusedValues.allSatisfy(\.isFinite)
            let bfloatOutputPassed: Bool
            do {
                let bfloatDecode = try turboQuantMetalDecodeAttention(
                    valueCode,
                    outputDType: .bfloat16
                )
                let bfloatQueries = queries.asType(.bfloat16)
                let bfloatAV = try turboQuantMetalAV(
                    attentionWeights: weights,
                    valueCode: valueCode,
                    outputDType: bfloatQueries.dtype
                )
                let bfloatFused = try turboQuantMetalScaledDotProductAttention(
                    queries: bfloatQueries,
                    keyCode: keyCode,
                    valueCode: valueCode,
                    scale: scale,
                    mask: .causal,
                    preferOnlineFused: true,
                    kernelProfile: selectedProfile
                )
                eval(bfloatDecode, bfloatAV, bfloatFused)
                bfloatOutputPassed =
                    bfloatDecode.dtype == .bfloat16
                    && bfloatDecode.shape == values.shape
                    && bfloatAV.dtype == .bfloat16
                    && bfloatAV.shape == av.shape
                    && bfloatFused.dtype == .bfloat16
                    && bfloatFused.shape == fused.shape
                    && bfloatFused.asArray(Float.self).allSatisfy(\.isFinite)
            } catch {
                bfloatOutputPassed = false
            }
            try turboQuantWarmAttentionKernelVariants(
                headDimensions: onlineFusedHeadDimensions,
                kernelProfile: selectedProfile
            )
            let polarWHTSelfTest = turboQuantRunPolarWHTMetalSelfTest()
            let passed =
                flatCodecPassed && encodeDecodePassed && qkPassed && avPassed
            let failureReason =
                passed
                ? nil
                : "TurboQuant Metal self-test failed: flatCodec=\(flatCodecPassed), attentionCodec=\(encodeDecodePassed), qk=\(qkPassed), av=\(avPassed), fused=\(fusedPassed), bfloat=\(bfloatOutputPassed), avRelativeMSE=\(avReferenceRelativeMSE), fusedRelativeMSE=\(fusedReferenceRelativeMSE)."

            return TurboQuantRuntimeProbeResult(
                status: passed ? .passed : .failed,
                metalRuntimeAvailable: true,
                flatCodecPassed: flatCodecPassed,
                encodeDecodePassed: encodeDecodePassed,
                qkPassed: qkPassed,
                avPassed: avPassed,
                tiledFusedPassed: fusedPassed,
                bfloatOutputPassed: bfloatOutputPassed,
                polarWHTCodecPassed: polarWHTSelfTest.codecPassed,
                polarWHTAttentionPassed: polarWHTSelfTest.attentionPassed,
                hybridK8PolarWHTValueAttentionPassed:
                    polarWHTSelfTest.hybridK8PolarWHTValueAttentionPassed,
                selectedKernelProfile: passed ? selectedProfile : .mlxPackedFallback,
                failureReason: failureReason,
                polarWHTFailureReason: polarWHTSelfTest.failureReason,
                encodeDecodeLatencySeconds: encodeDecodeLatency,
                twoStageLatencySeconds: twoStageLatency,
                tiledFusedLatencySeconds: fusedLatency,
                onlineFusedHeadDimensions: fusedPassed ? onlineFusedHeadDimensions : []
            )
        } catch {
            return TurboQuantRuntimeProbeResult(
                status: .failed,
                metalRuntimeAvailable: true,
                selectedKernelProfile: .mlxPackedFallback,
                failureReason: String(describing: error)
            )
        }
    }
}

private func validateMetalConfiguration(
    array: MLXArray,
    configuration: TurboQuantConfiguration
) throws {
    guard array.size > 0 else {
        throw TurboQuantError.invalidMetalConfiguration("empty arrays are not supported")
    }
    guard array.dtype.isFloatingPoint else {
        throw TurboQuantError.invalidMetalConfiguration("input dtype must be floating point")
    }
    guard configuration.groupSize > 0 else {
        throw TurboQuantError.invalidGroupSize(configuration.groupSize)
    }
    guard configuration.groupSize <= 128, configuration.groupSize % 32 == 0 else {
        throw TurboQuantError.invalidMetalConfiguration(
            "group size must be 32, 64, 96, or 128 for the Metal codec"
        )
    }
    if configuration.role == .value {
        try validateTurboQuantValueBits(configuration.resolvedValueBits)
    }
    try requireTurboQuantMetalCodec()
}

private func validateMetalCodeStorage(_ code: TurboQuantMetalCode) throws {
    guard code.valueCount > 0, code.shape.reduce(1, *) == code.valueCount else {
        throw TurboQuantError.invalidMetalConfiguration(
            "flat code shape \(code.shape) does not match value count \(code.valueCount)"
        )
    }
    guard code.groupSize > 0 else {
        throw TurboQuantError.invalidGroupSize(code.groupSize)
    }
    guard code.groupSize <= 128, code.groupSize % 32 == 0 else {
        throw TurboQuantError.invalidMetalConfiguration(
            "group size must be 32, 64, 96, or 128 for the Metal codec"
        )
    }
    guard code.groupCount == (code.valueCount + code.groupSize - 1) / code.groupSize else {
        throw TurboQuantError.invalidMetalConfiguration("flat code group count is inconsistent")
    }
    let expectedMagnitudeWords = metalMagnitudeWordsPerGroup(
        groupSize: code.groupSize,
        preset: code.preset,
        role: code.role,
        valueBits: code.valueBits
    )
    guard code.magnitudeWordsPerGroup == expectedMagnitudeWords else {
        throw TurboQuantError.invalidMetalConfiguration(
            "flat code magnitude words per group \(code.magnitudeWordsPerGroup) does not match expected \(expectedMagnitudeWords)"
        )
    }
    let expectedBitsetWords = (code.groupSize + 31) / 32
    guard code.bitsetWordsPerGroup == expectedBitsetWords else {
        throw TurboQuantError.invalidMetalConfiguration(
            "flat code bitset words per group \(code.bitsetWordsPerGroup) does not match expected \(expectedBitsetWords)"
        )
    }
    let expectedScalesPerGroup = metalScalesPerGroup(role: code.role)
    guard code.scalesPerGroup == expectedScalesPerGroup else {
        throw TurboQuantError.invalidMetalConfiguration(
            "flat code scales per group \(code.scalesPerGroup) does not match expected \(expectedScalesPerGroup)"
        )
    }
    if code.role == .value {
        try validateTurboQuantValueBits(code.valueBits)
    }

    let packedShape = [code.groupCount * code.magnitudeWordsPerGroup]
    try validateStorageArray(
        code.packedMagnitudes,
        name: "flat packed magnitudes",
        expectedShape: packedShape,
        expectedDType: .uint32
    )
    let bitsetShape = [code.groupCount * code.bitsetWordsPerGroup]
    try validateStorageArray(
        code.signs,
        name: "flat signs",
        expectedShapes: code.role == .value ? [turboQuantCompactUnusedBitsetShape] : [bitsetShape],
        expectedDType: .uint32
    )
    try validateStorageArray(
        code.highPrecisionMask,
        name: "flat high precision mask",
        expectedShapes: code.role == .value ? [turboQuantCompactUnusedBitsetShape] : [bitsetShape],
        expectedDType: .uint32
    )
    try validateStorageArray(
        code.residualSigns,
        name: "flat residual signs",
        expectedShapes: [turboQuantCompactUnusedBitsetShape],
        expectedDType: .uint32
    )
    try validateStorageArray(
        code.scales,
        name: "flat scales",
        expectedShape: [code.groupCount, code.scalesPerGroup],
        expectedDType: .float32
    )
}

private func validateStorageArray(
    _ array: MLXArray,
    name: String,
    expectedShape: [Int],
    expectedDType: DType
) throws {
    try validateStorageArray(
        array,
        name: name,
        expectedShapes: [expectedShape],
        expectedDType: expectedDType
    )
}

private func validateStorageArray(
    _ array: MLXArray,
    name: String,
    expectedShapes: [[Int]],
    expectedDType: DType
) throws {
    try validateStorageArray(
        array,
        name: name,
        expectedShapes: expectedShapes,
        expectedDTypes: [expectedDType]
    )
}

private func validateStorageArray(
    _ array: MLXArray,
    name: String,
    expectedShape: [Int],
    expectedDTypes: [DType]
) throws {
    try validateStorageArray(
        array,
        name: name,
        expectedShapes: [expectedShape],
        expectedDTypes: expectedDTypes
    )
}

private func validateStorageArray(
    _ array: MLXArray,
    name: String,
    expectedShapes: [[Int]],
    expectedDTypes: [DType]
) throws {
    guard expectedShapes.contains(array.shape) else {
        throw TurboQuantError.invalidMetalConfiguration(
            "\(name) has shape \(array.shape), expected one of \(expectedShapes)"
        )
    }
    guard expectedDTypes.contains(array.dtype) else {
        throw TurboQuantError.invalidMetalConfiguration(
            "\(name) has dtype \(array.dtype), expected one of \(expectedDTypes)"
        )
    }
    let expectedByteCount = array.shape.reduce(1, *) * array.dtype.size
    guard array.nbytes == expectedByteCount else {
        throw TurboQuantError.invalidMetalConfiguration(
            "\(name) uses \(array.nbytes) storage bytes, expected \(expectedByteCount)"
        )
    }
    // TQ_T11: shape/dtype/nbytes/contiguousToDimension are metadata reads on an
    // unevaluated MLXArray (proven by the eval-free twin in TurboQuantValidation.swift);
    // the eval here only forced early materialization. Deep tier restores it for
    // diagnostics: TURBOQUANT_DEEP_VALIDATE=1.
    if turboQuantDeepValidationEnabled {
        if TurboQuantHostProbe.enabled {
            let start = DispatchTime.now().uptimeNanoseconds
            array.eval()
            TurboQuantHostProbe.shared.recordValidateEval(
                nanos: DispatchTime.now().uptimeNanoseconds - start)
        } else {
            array.eval()
        }
    }
    guard array.contiguousToDimension() == 0 else {
        throw TurboQuantError.invalidMetalConfiguration(
            "\(name) must be canonical row-contiguous storage"
        )
    }
}

private func metalMagnitudeWordsPerGroup(
    groupSize: Int,
    preset: TurboQuantPreset,
    role: TurboQuantTensorRole = .key,
    valueBits: Int? = nil,
    layoutVersion: Int? = nil
) -> Int {
    if role == .value {
        let bitCount = groupSize * (valueBits ?? preset.defaultValueBits)
        return (bitCount + 31) / 32
    }
    let baseBits = Swift.max(1, preset.baseMagnitudeBits - 1)
    let highBits = Swift.max(baseBits, preset.highMagnitudeBits - 1)
    if turboQuantUsesSplitMagnitudePlane(
        preset: preset,
        role: role,
        layoutVersion: layoutVersion,
        baseBits: baseBits,
        highBits: highBits
    ) {
        let highCount = mixedPrecisionHighCount(
            valueCount: groupSize,
            baseBits: baseBits,
            highBits: highBits,
            targetBits: Swift.max(1, preset.targetMagnitudeBits - 1)
        )
        return (groupSize * baseBits + highCount * (highBits - baseBits) + 31) / 32
    }
    let highCount = mixedPrecisionHighCount(
        valueCount: groupSize,
        baseBits: baseBits,
        highBits: highBits,
        targetBits: Swift.max(1, preset.targetMagnitudeBits - 1)
    )
    let bitCount =
        groupSize * baseBits
        + highCount * (highBits - baseBits)
    return (bitCount + 31) / 32
}

private func turboQuantUsesSplitMagnitudePlane(
    preset: TurboQuantPreset,
    role: TurboQuantTensorRole,
    layoutVersion: Int?,
    baseBits: Int? = nil,
    highBits: Int? = nil
) -> Bool {
    guard role == .key,
        (layoutVersion ?? TurboQuantAttentionLayout.legacyVersion)
            >= TurboQuantAttentionLayout.splitMagnitudeVersion
    else { return false }
    let resolvedBaseBits = baseBits ?? Swift.max(1, preset.baseMagnitudeBits - 1)
    let resolvedHighBits =
        highBits ?? Swift.max(resolvedBaseBits, preset.highMagnitudeBits - 1)
    return resolvedHighBits == resolvedBaseBits + 1
}

private func turboQuantStoresHighPrecisionMask(
    preset: TurboQuantPreset,
    role: TurboQuantTensorRole,
    layoutVersion: Int?
) -> Bool {
    guard role == .key else { return false }
    let baseBits = Swift.max(1, preset.baseMagnitudeBits - 1)
    let highBits = Swift.max(baseBits, preset.highMagnitudeBits - 1)
    if highBits <= baseBits {
        return false
    }
    return !turboQuantUsesSplitMagnitudePlane(
        preset: preset,
        role: role,
        layoutVersion: layoutVersion,
        baseBits: baseBits,
        highBits: highBits
    )
}

private func metalScalesPerGroup(role: TurboQuantTensorRole) -> Int {
    // K scale plane dieted to 2 (norm, residual_norm); the third slot was dead (written 0.0, never read).
    return 2
}

private func validateRequestedAttentionLayoutVersion(
    _ layoutVersion: Int,
    allowExperimentalLayoutV5: Bool,
    allowExperimentalLayoutV7: Bool = false
) throws {
    guard
        TurboQuantAttentionLayout.supportedVersions.contains(layoutVersion)
            || layoutVersion == TurboQuantAttentionLayout.tileTransposedVersion
    else {
        throw TurboQuantError.invalidMetalConfiguration(
            "unsupported compressed attention layout version \(layoutVersion)"
        )
    }
    if layoutVersion == 5 && !allowExperimentalLayoutV5 {
        throw TurboQuantError.invalidMetalConfiguration(
            "Layout V5 requires allowExperimentalLayoutV5"
        )
    }
    if layoutVersion == TurboQuantAttentionLayout.tileTransposedVersion && !allowExperimentalLayoutV7 {
        throw TurboQuantError.invalidMetalConfiguration(
            "Layout V7 requires allowExperimentalLayoutV7"
        )
    }
}

private func validateAttentionScaleStorage(
    _ scaleStorage: TurboQuantScaleStorage,
    layoutVersion: Int,
    allowExperimentalLayoutV5: Bool
) throws {
    switch scaleStorage {
    case .float32:
        return
    case .float16:
        guard layoutVersion >= 5, allowExperimentalLayoutV5
        else {
            throw TurboQuantError.invalidMetalConfiguration(
                "fp16 TurboQuant attention scales require explicitly enabled Layout V5 or newer"
            )
        }
    }
}

private func validateAttentionConfiguration(_ configuration: TurboQuantConfiguration) throws {
    try validateRequestedAttentionLayoutVersion(
        configuration.attentionLayoutVersion,
        allowExperimentalLayoutV5: configuration.allowExperimentalLayoutV5,
        allowExperimentalLayoutV7: configuration.allowExperimentalLayoutV7
    )
    try validateAttentionScaleStorage(
        configuration.attentionScaleStorage,
        layoutVersion: configuration.attentionLayoutVersion,
        allowExperimentalLayoutV5: configuration.allowExperimentalLayoutV5
    )
}

private func turboQuantAttentionScaleStorage(
    for code: TurboQuantAttentionCode
) -> TurboQuantScaleStorage {
    code.scales.dtype == .float16 ? .float16 : .float32
}

private func supportedAttentionScaleDTypes(
    for layoutVersion: Int
) -> [DType] {
    layoutVersion >= 5
        ? [.float32, .float16]
        : [.float32]
}

private func metalTemplate(
    configuration: TurboQuantConfiguration,
    valueCount: Int,
    groupCount: Int,
    magnitudeWordsPerGroup: Int,
    bitsetWordsPerGroup: Int,
    outputDType: DType = .float32
) -> [(String, any KernelTemplateArg)] {
    let highFraction = mixedPrecisionHighFraction(preset: configuration.preset)
    return [
        ("GROUP_SIZE", configuration.groupSize),
        ("VALUE_COUNT", valueCount),
        ("GROUP_COUNT", groupCount),
        ("BASE_BITS", configuration.preset.baseMagnitudeBits),
        ("HIGH_BITS", configuration.preset.highMagnitudeBits),
        ("KEY_BASE_BITS", Swift.max(1, configuration.preset.baseMagnitudeBits - 1)),
        (
            "KEY_HIGH_BITS",
            Swift.max(
                Swift.max(1, configuration.preset.baseMagnitudeBits - 1),
                configuration.preset.highMagnitudeBits - 1
            )
        ),
        ("HIGH_NUMERATOR", highFraction.numerator),
        ("HIGH_DENOMINATOR", highFraction.denominator),
        ("MAG_WORDS_PER_GROUP", magnitudeWordsPerGroup),
        ("BITSET_WORDS_PER_GROUP", bitsetWordsPerGroup),
        ("VALUE_BITS", configuration.resolvedValueBits),
        ("SCALES_PER_GROUP", metalScalesPerGroup(role: configuration.role)),
        ("ROLE", metalRoleValue(configuration.role)),
        ("OUTPUT_DTYPE", outputDType),
    ] + metalTemplateSeedWords(prefix: "SEED", value: configuration.seed)
}

private func metalRoleValue(_ role: TurboQuantTensorRole) -> Int {
    switch role {
    case .key:
        0
    case .value:
        1
    case .vector:
        2
    }
}

private func validateAttentionArray(_ array: MLXArray, groupSize: Int) throws {
    try validateAttentionShape(array.shape, dtype: array.dtype, groupSize: groupSize)
    guard array.contiguousToDimension() == 0 else {
        throw TurboQuantError.invalidMetalConfiguration(
            "TurboQuant attention tensors must be canonical row-contiguous storage"
        )
    }
}

private func validateAttentionShape(_ shape: [Int], dtype: DType, groupSize: Int) throws {
    guard shape.count == 4 else {
        throw TurboQuantError.invalidMetalConfiguration(
            "attention tensors must have shape [B, H, T, D]"
        )
    }
    guard shape.reduce(1, *) > 0 else {
        throw TurboQuantError.invalidMetalConfiguration("empty attention tensors are not supported")
    }
    guard dtype.isFloatingPoint else {
        throw TurboQuantError.invalidMetalConfiguration(
            "attention tensor dtype must be floating point")
    }
    guard groupSize > 0 else {
        throw TurboQuantError.invalidGroupSize(groupSize)
    }
    guard groupSize <= 128, groupSize % 32 == 0 else {
        throw TurboQuantError.invalidMetalConfiguration(
            "group size must be 32, 64, 96, or 128 for compressed attention"
        )
    }
    guard shape[3] <= 512 else {
        throw TurboQuantError.invalidMetalConfiguration(
            "head dimension \(shape[3]) is not supported by compressed attention"
        )
    }
}

private func validateAttentionLayout(
    _ layout: TurboQuantAttentionLayout,
    role: TurboQuantTensorRole,
    groupSize: Int,
    allowTileTransposedV7: Bool = false
) throws {
    guard role == .key || role == .value else {
        throw TurboQuantError.invalidMetalConfiguration(
            "compressed attention codes must be encoded as key or value"
        )
    }
    guard
        TurboQuantAttentionLayout.supportedVersions.contains(layout.layoutVersion)
            || (allowTileTransposedV7
                && layout.layoutVersion == TurboQuantAttentionLayout.tileTransposedVersion)
    else {
        throw TurboQuantError.invalidMetalConfiguration(
            "unsupported compressed attention layout version \(layout.layoutVersion)"
        )
    }
    guard layout.batchSize > 0, layout.kvHeadCount > 0, layout.capacity > 0,
        layout.logicalLength >= 0, layout.logicalLength <= layout.capacity,
        layout.headDimension > 0
    else {
        throw TurboQuantError.invalidMetalConfiguration("invalid compressed attention layout shape")
    }
    guard layout.ringOffset >= 0, layout.ringOffset < layout.capacity else {
        throw TurboQuantError.invalidMetalConfiguration("ring offset is outside cache capacity")
    }
    guard layout.pinnedPrefixLength >= 0, layout.pinnedPrefixLength <= layout.capacity else {
        throw TurboQuantError.invalidMetalConfiguration("pinned prefix is outside cache capacity")
    }
    guard layout.pinnedPrefixLength <= layout.logicalLength else {
        throw TurboQuantError.invalidMetalConfiguration(
            "pinned prefix cannot exceed logical length"
        )
    }
    let ringCapacity = layout.capacity - layout.pinnedPrefixLength
    if ringCapacity == 0 {
        guard layout.ringOffset == 0 else {
            throw TurboQuantError.invalidMetalConfiguration(
                "ring offset must be zero without ring capacity")
        }
    } else {
        guard layout.ringOffset < ringCapacity else {
            throw TurboQuantError.invalidMetalConfiguration(
                "ring offset is outside rotating region")
        }
    }
    guard layout.groupsPerVector == (layout.headDimension + groupSize - 1) / groupSize else {
        throw TurboQuantError.invalidMetalConfiguration("groups per vector does not match layout")
    }
    guard layout.magnitudeWordsPerGroup > 0, layout.bitsetWordsPerGroup > 0 else {
        throw TurboQuantError.invalidMetalConfiguration(
            "packed-word and bitset axes must be positive"
        )
    }
    if layout.layoutVersion == TurboQuantAttentionLayout.tileTransposedVersion {
        guard layout.capacity % 32 == 0 else {
            throw TurboQuantError.invalidMetalConfiguration(
                "layout v7 requires capacity to be a multiple of 32; got \(layout.capacity)"
            )
        }
    }
}

private func validateAttentionCodeStorage(
    _ code: TurboQuantAttentionCode,
    allowTileTransposedV7: Bool = false
) throws {
    try validateAttentionLayout(
        code.layout,
        role: code.role,
        groupSize: code.groupSize,
        allowTileTransposedV7: allowTileTransposedV7
    )
    if code.role == .value {
        try validateTurboQuantValueBits(code.valueBits)
    }
    let expectedMagnitudeWords = metalMagnitudeWordsPerGroup(
        groupSize: code.groupSize,
        preset: code.preset,
        role: code.role,
        valueBits: code.valueBits,
        layoutVersion: code.layout.layoutVersion
    )
    guard code.layout.magnitudeWordsPerGroup == expectedMagnitudeWords else {
        throw TurboQuantError.invalidMetalConfiguration(
            "compressed attention magnitude words per group \(code.layout.magnitudeWordsPerGroup) does not match expected \(expectedMagnitudeWords)"
        )
    }
    let expectedBitsetWords = (code.groupSize + 31) / 32
    guard code.layout.bitsetWordsPerGroup == expectedBitsetWords else {
        throw TurboQuantError.invalidMetalConfiguration(
            "compressed attention bitset words per group \(code.layout.bitsetWordsPerGroup) does not match expected \(expectedBitsetWords)"
        )
    }
    let expectedScalesPerGroup = metalScalesPerGroup(role: code.role)
    guard code.scalesPerGroup == expectedScalesPerGroup else {
        throw TurboQuantError.invalidMetalConfiguration(
            "compressed attention scales per group \(code.scalesPerGroup) does not match expected \(expectedScalesPerGroup)"
        )
    }

    let packedShape = [
        code.layout.batchSize, code.layout.kvHeadCount, code.layout.capacity,
        code.layout.groupsPerVector, code.layout.magnitudeWordsPerGroup,
    ]
    let bitsetShape = [
        code.layout.batchSize, code.layout.kvHeadCount, code.layout.capacity,
        code.layout.groupsPerVector, code.layout.bitsetWordsPerGroup,
    ]
    let scalesShape = [
        code.layout.batchSize, code.layout.kvHeadCount, code.layout.capacity,
        code.layout.groupsPerVector, code.scalesPerGroup,
    ]
    try validateStorageArray(
        code.packedMagnitudes,
        name: "compressed attention packed magnitudes",
        expectedShape: packedShape,
        expectedDType: .uint32
    )
    try validateStorageArray(
        code.signs,
        name: "compressed attention signs",
        expectedShapes: code.role == .value ? [turboQuantCompactUnusedBitsetShape] : [bitsetShape],
        expectedDType: .uint32
    )
    try validateStorageArray(
        code.highPrecisionMask,
        name: "compressed attention high precision mask",
        expectedShapes: turboQuantStoresHighPrecisionMask(
            preset: code.preset,
            role: code.role,
            layoutVersion: code.layout.layoutVersion
        ) ? [bitsetShape] : [turboQuantCompactUnusedBitsetShape, bitsetShape],
        expectedDType: .uint32
    )
    try validateStorageArray(
        code.residualSigns,
        name: "compressed attention residual signs",
        expectedShapes: [turboQuantCompactUnusedBitsetShape],
        expectedDType: .uint32
    )
    try validateStorageArray(
        code.scales,
        name: "compressed attention scales",
        expectedShape: scalesShape,
        expectedDTypes: supportedAttentionScaleDTypes(for: code.layout.layoutVersion)
    )
}

private func validateAttentionQuery(
    _ queries: MLXArray,
    code: TurboQuantAttentionCode
) throws {
    try validateAttentionShape(queries.shape, dtype: queries.dtype, groupSize: code.groupSize)
    guard queries.dim(0) == code.layout.batchSize else {
        throw TurboQuantError.invalidMetalConfiguration(
            "query batch size does not match compressed attention cache"
        )
    }
    guard queries.dim(3) == code.layout.headDimension else {
        throw TurboQuantError.invalidMetalConfiguration(
            "query head dimension does not match compressed attention cache"
        )
    }
    guard queries.dim(1) % code.layout.kvHeadCount == 0 else {
        throw TurboQuantError.invalidMetalConfiguration(
            "query heads must be a multiple of KV heads"
        )
    }
}

private func validateAttentionPair(
    keyCode: TurboQuantAttentionCode,
    valueCode: TurboQuantAttentionCode,
    allowTileTransposedV7: Bool = false
) throws {
    try validateAttentionLayout(
        keyCode.layout, role: keyCode.role, groupSize: keyCode.groupSize,
        allowTileTransposedV7: allowTileTransposedV7)
    try validateAttentionLayout(
        valueCode.layout, role: valueCode.role, groupSize: valueCode.groupSize,
        allowTileTransposedV7: allowTileTransposedV7)
    guard keyCode.role == .key, valueCode.role == .value else {
        throw TurboQuantError.invalidMetalConfiguration(
            "compressed attention requires key and value codes")
    }
    guard attentionLayoutsShareSequence(keyCode.layout, valueCode.layout) else {
        throw TurboQuantError.invalidMetalConfiguration(
            "key and value compressed sequence layouts differ"
        )
    }
    guard keyCode.preset == valueCode.preset, keyCode.groupSize == valueCode.groupSize else {
        throw TurboQuantError.invalidMetalConfiguration("key and value compressed presets differ")
    }
}

private func attentionLayoutsShareSequence(
    _ keyLayout: TurboQuantAttentionLayout,
    _ valueLayout: TurboQuantAttentionLayout
) -> Bool {
    keyLayout.layoutVersion == valueLayout.layoutVersion
        && keyLayout.batchSize == valueLayout.batchSize
        && keyLayout.kvHeadCount == valueLayout.kvHeadCount
        && keyLayout.capacity == valueLayout.capacity
        && keyLayout.logicalLength == valueLayout.logicalLength
        && keyLayout.ringOffset == valueLayout.ringOffset
        && keyLayout.pinnedPrefixLength == valueLayout.pinnedPrefixLength
}

private func validateAttentionSinks(_ sinks: MLXArray?, queryHeadCount: Int) throws {
    guard let sinks else { return }
    guard sinks.ndim == 1, sinks.dim(0) == queryHeadCount else {
        throw TurboQuantError.invalidMetalConfiguration(
            "attention sinks must have shape [query heads]"
        )
    }
    guard sinks.dtype.isFloatingPoint else {
        throw TurboQuantError.invalidMetalConfiguration("attention sinks must be floating point")
    }
}

private func validateAttentionMask(
    _ mask: MLXFast.ScaledDotProductAttentionMaskMode,
    scoreShape: [Int]
) throws {
    guard scoreShape.count == 4 else {
        throw TurboQuantError.invalidMetalConfiguration("attention score shape must be rank 4")
    }
    func validateMaskArray(_ maskArray: MLXArray) throws {
        guard maskArray.dtype == .bool || maskArray.dtype.isFloatingPoint else {
            throw TurboQuantError.invalidMetalConfiguration(
                "attention mask must be bool or floating point"
            )
        }
        guard maskArray.ndim <= scoreShape.count else {
            throw TurboQuantError.invalidMetalConfiguration(
                "attention mask rank \(maskArray.ndim) cannot broadcast to score rank \(scoreShape.count)"
            )
        }
        let paddedShape =
            Array(repeating: 1, count: scoreShape.count - maskArray.ndim) + maskArray.shape
        for (actual, expected) in zip(paddedShape, scoreShape) {
            guard actual == 1 || actual == expected else {
                throw TurboQuantError.invalidMetalConfiguration(
                    "attention mask shape \(maskArray.shape) cannot broadcast to score shape \(scoreShape)"
                )
            }
        }
    }

    switch mask {
    case .causal:
        guard scoreShape[2] <= scoreShape[3] else {
            throw TurboQuantError.invalidMetalConfiguration(
                "causal compressed attention requires query length \(scoreShape[2]) <= key length \(scoreShape[3])"
            )
        }
    case .array(let maskArray):
        try validateMaskArray(maskArray)
    case .arrays(let maskArrays):
        guard maskArrays.count <= 1 else {
            throw TurboQuantError.invalidMetalConfiguration(
                "TurboQuant compressed attention supports at most one materialized mask"
            )
        }
        if let maskArray = maskArrays.first {
            try validateMaskArray(maskArray)
        }
    case .none:
        break
    }
}

private func prependAttentionSinks(
    _ scores: MLXArray,
    sinks: MLXArray?,
    queryHeadCount: Int,
    stream: StreamOrDevice
) throws -> MLXArray {
    guard let sinks else { return scores }
    try validateAttentionSinks(sinks, queryHeadCount: queryHeadCount)
    let sinkScores = broadcast(
        expandedDimensions(sinks.asType(.float32), axes: [0, 2, 3], stream: stream),
        to: [scores.dim(0), scores.dim(1), scores.dim(2), 1],
        stream: stream
    )
    return concatenated([sinkScores, scores], axis: -1, stream: stream)
}

private func applyAttentionMask(
    _ scores: inout MLXArray,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    stream: StreamOrDevice
) throws {
    try validateAttentionMask(mask, scoreShape: scores.shape)
    switch mask {
    case .causal:
        let (qL, kL) = (scores.dim(-2), scores.dim(-1))
        let qIndices = MLXArray(0 ..< qL) + MLXArray(kL - qL)
        let kIndices = MLXArray(0 ..< kL)
        let causalMask = greaterEqual(
            expandedDimensions(qIndices, axis: -1),
            expandedDimensions(kIndices, axis: -2),
            stream: stream
        )
        scores = `where`(
            causalMask,
            scores,
            MLXArray(-Float.greatestFiniteMagnitude),
            stream: stream
        )

    case .array(let maskArray):
        if maskArray.dtype == .bool {
            scores = `where`(
                maskArray,
                scores,
                MLXArray(-Float.greatestFiniteMagnitude),
                stream: stream
            )
        } else {
            scores = scores + maskArray
        }

    case .arrays(let maskArrays):
        guard let maskArray = maskArrays.first else {
            break
        }
        if maskArray.dtype == .bool {
            scores = `where`(
                maskArray,
                scores,
                MLXArray(-Float.greatestFiniteMagnitude),
                stream: stream
            )
        } else {
            scores = scores + maskArray
        }

    case .none:
        break
    }
}

private func attentionTemplate(
    configuration: TurboQuantConfiguration,
    layout: TurboQuantAttentionLayout,
    inputLength: Int,
    outputLength: Int,
    queryHeadCount: Int,
    queryLength: Int,
    outputDType: DType,
    causal: Bool
) -> [(String, any KernelTemplateArg)] {
    let highFraction = mixedPrecisionHighFraction(preset: configuration.preset)
    return [
        ("BATCH_SIZE", layout.batchSize),
        ("KV_HEADS", layout.kvHeadCount),
        ("QUERY_HEADS", queryHeadCount),
        ("INPUT_LENGTH", inputLength),
        ("OUTPUT_LENGTH", outputLength),
        ("CAPACITY", layout.capacity),
        ("LOGICAL_LENGTH", layout.logicalLength),
        ("RING_OFFSET", layout.ringOffset),
        ("PINNED_PREFIX_LENGTH", layout.pinnedPrefixLength),
        ("QUERY_LENGTH", queryLength),
        ("HEAD_DIM", layout.headDimension),
        ("GROUP_SIZE", configuration.groupSize),
        ("GROUPS_PER_VECTOR", layout.groupsPerVector),
        ("BASE_BITS", configuration.preset.baseMagnitudeBits),
        ("HIGH_BITS", configuration.preset.highMagnitudeBits),
        ("HIGH_NUMERATOR", highFraction.numerator),
        ("HIGH_DENOMINATOR", highFraction.denominator),
        ("KEY_BASE_BITS", Swift.max(1, configuration.preset.baseMagnitudeBits - 1)),
        (
            "KEY_HIGH_BITS",
            Swift.max(
                Swift.max(1, configuration.preset.baseMagnitudeBits - 1),
                configuration.preset.highMagnitudeBits - 1
            )
        ),
        ("MAG_WORDS_PER_GROUP", layout.magnitudeWordsPerGroup),
        ("BITSET_WORDS_PER_GROUP", layout.bitsetWordsPerGroup),
        ("VALUE_BITS", configuration.resolvedValueBits),
        ("SCALES_PER_GROUP", metalScalesPerGroup(role: configuration.role)),
        ("LAYOUT_VERSION", layout.layoutVersion),
        ("DETERMINISTIC_HIGH_MASK", configuration.deterministicHighPrecisionMask),
        ("ROLE", metalRoleValue(configuration.role)),
        ("OUTPUT_DTYPE", outputDType),
        ("DO_CAUSAL", causal),
    ] + metalTemplateSeedWords(prefix: "SEED", value: configuration.seed)
}

private let runtimeLayoutTemplateKeys: Set<String> = [
    "INPUT_LENGTH",
    "OUTPUT_LENGTH",
    "LOGICAL_LENGTH",
    "RING_OFFSET",
    "PINNED_PREFIX_LENGTH",
]

private func runtimeLayoutAttentionTemplate(
    configuration: TurboQuantConfiguration,
    layout: TurboQuantAttentionLayout,
    inputLength: Int,
    outputLength: Int,
    queryHeadCount: Int,
    queryLength: Int,
    outputDType: DType,
    causal: Bool
) -> [(String, any KernelTemplateArg)] {
    attentionTemplate(
        configuration: configuration,
        layout: layout,
        inputLength: inputLength,
        outputLength: outputLength,
        queryHeadCount: queryHeadCount,
        queryLength: queryLength,
        outputDType: outputDType,
        causal: causal
    ).filter { !runtimeLayoutTemplateKeys.contains($0.0) }
}

private enum TurboQuantMetalKernels {
    static let encode = MLXFast.metalKernel(
        name: "turboquant_polar_qjl_encode_s2",
        inputNames: ["x"],
        outputNames: ["packed", "signs", "high_mask", "residual_signs", "scales"],
        source: encodeSource,
        header: vectorHeader
    )

    static let decode = MLXFast.metalKernel(
        name: "turboquant_polar_qjl_decode",
        inputNames: ["packed", "signs", "high_mask", "residual_signs", "scales"],
        outputNames: ["out"],
        source: decodeSource,
        header: vectorHeader
    )

    static let matmul = MLXFast.metalKernel(
        name: "turboquant_polar_qjl_matmul",
        inputNames: ["x", "packed", "signs", "high_mask", "residual_signs", "scales"],
        outputNames: ["out"],
        source: matmulSource,
        header: vectorHeader
    )

    static let encodeAttention = MLXFast.metalKernel(
        name: "turboquant_attention_encode_s2",
        inputNames: ["x"],
        outputNames: ["packed", "signs", "high_mask", "residual_signs", "scales"],
        source: encodeAttentionSource,
        header: attentionHeader,
        ensureRowContiguous: false
    )

    // Layout v7 (tile-transposed): swizzles K packed/signs/scales writes onto the
    // 32-token tile layout. V planes stay token-major (v6 format) even at v7; see
    // SPEC 2 section 4.
    static let encodeAttentionV7 = MLXFast.metalKernel(
        name: "turboquant_attention_encode_s2_v7",
        inputNames: ["x"],
        outputNames: ["packed", "signs", "high_mask", "residual_signs", "scales"],
        source: encodeAttentionV7Source,
        header: attentionHeader + attentionHeaderV7Extension,
        ensureRowContiguous: false
    )

    static let decodeAttention = MLXFast.metalKernel(
        name: "turboquant_attention_decode_runtime_layout_s2",
        inputNames: [
            "packed", "signs", "high_mask", "residual_signs", "scales",
            "runtime_logical_length",
            "runtime_ring_offset",
            "runtime_pinned_prefix_length",
        ],
        outputNames: ["out"],
        source: decodeAttentionSource,
        header: attentionHeader,
        ensureRowContiguous: false
    )

    static let keyPageSummary = MLXFast.metalKernel(
        name: "turboquant_attention_key_page_summary_runtime_layout_s2",
        inputNames: [
            "scales",
            "runtime_logical_length",
            "runtime_ring_offset",
            "runtime_pinned_prefix_length",
        ],
        outputNames: ["summary"],
        source: keyPageSummarySource,
        header: attentionHeader,
        ensureRowContiguous: false
    )

    static let qk = MLXFast.metalKernel(
        name: "turboquant_attention_qk_runtime_layout_s2",
        inputNames: [
            "q", "k_packed", "k_signs", "k_high_mask", "k_residual_signs", "k_scales",
            "runtime_logical_length",
            "runtime_ring_offset",
            "runtime_pinned_prefix_length",
            "runtime_attention_scale",
        ],
        outputNames: ["scores"],
        source: qkSource,
        header: attentionHeader,
        ensureRowContiguous: false
    )

    static let polarWHTEncodeAttention = MLXFast.metalKernel(
        name: "turboquant_polar_wht_attention_encode_s2",
        inputNames: ["x"],
        outputNames: ["packed_indices", "norms"],
        source: polarWHTEncodeAttentionSource,
        header: polarWHTAttentionHeader,
        ensureRowContiguous: false
    )

    static let polarWHTEncodeAttentionBulk = MLXFast.metalKernel(
        name: "turboquant_polar_wht_attention_encode_bulk_s2",
        inputNames: ["x"],
        outputNames: ["packed_indices", "norms"],
        source: polarWHTEncodeAttentionBulkSource,
        header: polarWHTAttentionHeader,
        ensureRowContiguous: false
    )

    static let hybridAffineK8PolarWHTValueEncode = MLXFast.metalKernel(
        name: "turboquant_hybrid_affine_k8_polar_wht_value_encode_s2",
        inputNames: ["keys", "values"],
        outputNames: [
            "key_packed", "key_scales", "key_biases",
            "value_packed_indices", "value_norms",
        ],
        source: hybridAffineK8PolarWHTValueEncodeSource,
        header: polarWHTAttentionHeader,
        ensureRowContiguous: false
    )

    static let hybridAffineK8PolarWHTValueEncodeBulk = MLXFast.metalKernel(
        name: "turboquant_hybrid_affine_k8_polar_wht_value_encode_bulk_s2",
        inputNames: ["keys", "values"],
        outputNames: [
            "key_packed", "key_scales", "key_biases",
            "value_packed_indices", "value_norms",
        ],
        source: hybridAffineK8PolarWHTValueEncodeBulkSource,
        header: polarWHTAttentionHeader,
        ensureRowContiguous: false
    )

    static let polarWHTDecodeAttention = MLXFast.metalKernel(
        name: "turboquant_polar_wht_attention_decode_runtime_layout_s2",
        inputNames: [
            "packed_indices", "norms",
            "runtime_logical_length",
            "runtime_ring_offset",
            "runtime_pinned_prefix_length",
        ],
        outputNames: ["out"],
        source: polarWHTDecodeAttentionSource,
        header: polarWHTAttentionHeader,
        ensureRowContiguous: false
    )

    static let polarWHTQK = MLXFast.metalKernel(
        name: "turboquant_polar_wht_attention_qk_runtime_layout_s2",
        inputNames: [
            "q", "k_packed_indices", "k_norms",
            "runtime_logical_length",
            "runtime_ring_offset",
            "runtime_pinned_prefix_length",
            "runtime_attention_scale",
        ],
        outputNames: ["scores"],
        source: polarWHTQKSource,
        header: polarWHTAttentionHeader,
        ensureRowContiguous: false
    )

    static let av = MLXFast.metalKernel(
        name: "turboquant_attention_av_runtime_layout_s2",
        inputNames: [
            "weights", "v_packed", "v_signs", "v_high_mask", "v_residual_signs", "v_scales",
            "runtime_logical_length",
            "runtime_ring_offset",
            "runtime_pinned_prefix_length",
        ],
        outputNames: ["out"],
        source: avSource,
        header: attentionHeader,
        ensureRowContiguous: false
    )

    static let polarWHTAV = MLXFast.metalKernel(
        name: "turboquant_polar_wht_attention_av_runtime_layout_s2",
        inputNames: [
            "weights", "v_packed_indices", "v_norms",
            "runtime_logical_length",
            "runtime_ring_offset",
            "runtime_pinned_prefix_length",
        ],
        outputNames: ["out"],
        source: polarWHTAVSource,
        header: polarWHTAttentionHeader,
        ensureRowContiguous: false
    )

    static let hybridPolarWHTValueFusedAttention = MLXFast.metalKernel(
        name: "turboquant_hybrid_polar_wht_value_fused_decode_runtime_layout_s2",
        inputNames: [
            "q",
            "k_packed", "k_signs", "k_high_mask", "k_residual_signs", "k_scales",
            "v_packed_indices", "v_norms",
            "runtime_logical_length",
            "runtime_ring_offset",
            "runtime_pinned_prefix_length",
            "runtime_attention_scale",
        ],
        outputNames: ["out"],
        source: hybridPolarWHTValueFusedAttentionSource,
        header: polarWHTAttentionHeader,
        ensureRowContiguous: false
    )

    static let hybridPolarWHTValueFusedBlockPartials = MLXFast.metalKernel(
        name: "turboquant_hybrid_polar_wht_value_fused_block_partials_runtime_layout_s2",
        inputNames: [
            "q",
            "k_packed", "k_signs", "k_high_mask", "k_residual_signs", "k_scales",
            "v_packed_indices", "v_norms",
            "runtime_logical_length",
            "runtime_ring_offset",
            "runtime_pinned_prefix_length",
            "runtime_attention_scale",
        ],
        outputNames: ["partial_stats", "partial_out"],
        source: hybridPolarWHTValueFusedBlockPartialsSource,
        header: polarWHTAttentionHeader,
        ensureRowContiguous: false
    )

    static let hybridPolarWHTValueGQAFusedBlockPartials = MLXFast.metalKernel(
        name: "turboquant_hybrid_polar_wht_value_fused_gqa_block_partials_runtime_layout_s2",
        inputNames: [
            "q",
            "k_packed", "k_signs", "k_high_mask", "k_residual_signs", "k_scales",
            "v_packed_indices", "v_norms",
            "runtime_logical_length",
            "runtime_ring_offset",
            "runtime_pinned_prefix_length",
            "runtime_attention_scale",
        ],
        outputNames: ["partial_stats", "partial_out"],
        source: hybridPolarWHTValueGQAFusedBlockPartialsSource,
        header: polarWHTAttentionHeader,
        ensureRowContiguous: false
    )

    static let hybridPolarWHTValueFusedBlockReduce = MLXFast.metalKernel(
        name: "turboquant_hybrid_polar_wht_value_fused_block_reduce_s2",
        inputNames: ["partial_stats", "partial_out"],
        outputNames: ["out"],
        source: hybridPolarWHTValueFusedBlockReduceSource,
        header: polarWHTAttentionHeader,
        ensureRowContiguous: false
    )

    static let hybridAffineK8PolarWHTValueFusedAttention = MLXFast.metalKernel(
        name: "turboquant_hybrid_affine_k8_polar_wht_value_fused_decode_runtime_layout_s2",
        inputNames: [
            "q",
            "k_packed", "k_scales", "k_biases",
            "v_packed_indices", "v_norms",
            "runtime_logical_length",
            "runtime_ring_offset",
            "runtime_pinned_prefix_length",
            "runtime_attention_scale",
        ],
        outputNames: ["out"],
        source: hybridAffineK8PolarWHTValueFusedAttentionSource,
        header: polarWHTAttentionHeader,
        ensureRowContiguous: false
    )

    static let segmentedHybridAffineK8PolarWHTValueFusedAttention = MLXFast.metalKernel(
        name: "turboquant_segmented_hybrid_affine_k8_polar_wht_value_fused_decode_s2",
        inputNames: [
            "q",
            "base_k_packed", "base_k_scales", "base_k_biases",
            "base_v_packed_indices", "base_v_norms",
            "tail_k_packed", "tail_k_scales", "tail_k_biases",
            "tail_v_packed_indices", "tail_v_norms",
            "runtime_base_logical_length",
            "runtime_base_ring_offset",
            "runtime_base_pinned_prefix_length",
            "runtime_tail_logical_length",
            "runtime_tail_ring_offset",
            "runtime_tail_pinned_prefix_length",
            "runtime_attention_scale",
        ],
        outputNames: ["out"],
        source: segmentedHybridAffineK8PolarWHTValueFusedAttentionSource,
        header: polarWHTAttentionHeader,
        ensureRowContiguous: false
    )

    static let hybridAffineK8DecodedValueFusedAttention = MLXFast.metalKernel(
        name: "turboquant_hybrid_affine_k8_decoded_value_fused_decode_s2",
        inputNames: [
            "q",
            "k_packed", "k_scales", "k_biases",
            "v_decoded",
            "runtime_logical_length",
            "runtime_ring_offset",
            "runtime_pinned_prefix_length",
            "runtime_attention_scale",
        ],
        outputNames: ["out"],
        source: hybridAffineK8DecodedValueFusedAttentionSource,
        header: polarWHTAttentionHeader,
        ensureRowContiguous: false
    )

    static let hybridAffineK8DecodedValueFusedBlockPartials = MLXFast.metalKernel(
        name: "turboquant_hybrid_affine_k8_decoded_value_fused_block_partials_s2",
        inputNames: [
            "q",
            "k_packed", "k_scales", "k_biases",
            "v_decoded",
            "runtime_logical_length",
            "runtime_ring_offset",
            "runtime_pinned_prefix_length",
            "runtime_attention_scale",
        ],
        outputNames: ["partial_stats", "partial_out"],
        source: hybridAffineK8DecodedValueFusedBlockPartialsSource,
        header: polarWHTAttentionHeader,
        ensureRowContiguous: false
    )

    static let hybridDecodedValueFusedBlockReduce = MLXFast.metalKernel(
        name: "turboquant_hybrid_decoded_value_fused_block_reduce_s2",
        inputNames: ["partial_stats", "partial_out"],
        outputNames: ["out"],
        source: hybridDecodedValueFusedBlockReduceSource,
        header: polarWHTAttentionHeader,
        ensureRowContiguous: false
    )

    static let hybridAffineK8PolarWHTValueFusedBlockPartials = MLXFast.metalKernel(
        name: "turboquant_hybrid_affine_k8_polar_wht_value_fused_block_partials_runtime_layout_s2",
        inputNames: [
            "q",
            "k_packed", "k_scales", "k_biases",
            "v_packed_indices", "v_norms",
            "runtime_logical_length",
            "runtime_ring_offset",
            "runtime_pinned_prefix_length",
            "runtime_attention_scale",
        ],
        outputNames: ["partial_stats", "partial_out"],
        source: hybridAffineK8PolarWHTValueFusedBlockPartialsSource,
        header: polarWHTAttentionHeader,
        ensureRowContiguous: false
    )

    static let hybridAffineK8PolarWHTValueGQAFusedBlockPartials = MLXFast.metalKernel(
        name: "turboquant_hybrid_affine_k8_polar_wht_value_fused_gqa_block_partials_runtime_layout_s2",
        inputNames: [
            "q",
            "k_packed", "k_scales", "k_biases",
            "v_packed_indices", "v_norms",
            "runtime_logical_length",
            "runtime_ring_offset",
            "runtime_pinned_prefix_length",
            "runtime_attention_scale",
        ],
        outputNames: ["partial_stats", "partial_out"],
        source: hybridAffineK8PolarWHTValueGQAFusedBlockPartialsSource,
        header: polarWHTAttentionHeader,
        ensureRowContiguous: false
    )

    static let fusedAttention = MLXFast.metalKernel(
        name: "turboquant_attention_fused_decode_runtime_layout_s2",
        inputNames: [
            "q",
            "k_packed", "k_signs", "k_high_mask", "k_residual_signs", "k_scales",
            "v_packed", "v_signs", "v_high_mask", "v_residual_signs", "v_scales",
            "runtime_logical_length",
            "runtime_ring_offset",
            "runtime_pinned_prefix_length",
            "runtime_attention_scale",
        ],
        outputNames: ["out"],
        source: fusedAttentionSource,
        header: attentionHeader,
        ensureRowContiguous: false
    )

    static let fusedAttentionBlockPartials = MLXFast.metalKernel(
        name: "turboquant_attention_fused_block_partials_runtime_layout_rtu1_s2",
        inputNames: [
            "q",
            "k_packed", "k_signs", "k_high_mask", "k_residual_signs", "k_scales",
            "v_packed", "v_signs", "v_high_mask", "v_residual_signs", "v_scales",
            "runtime_logical_length",
            "runtime_ring_offset",
            "runtime_pinned_prefix_length",
            "runtime_attention_scale",
            "runtime_block_count",
        ],
        outputNames: ["partial_stats", "partial_out"],
        source: fusedAttentionBlockPartialsSource,
        header: attentionHeader,
        ensureRowContiguous: false
    )

    static let fusedAttentionGQABlockPartials_rf1 = MLXFast.metalKernel(
        name: "turboquant_attention_fused_gqa_block_partials_runtime_layout_rtu1_s2_rf1",
        inputNames: [
            "q",
            "k_packed", "k_signs", "k_high_mask", "k_residual_signs", "k_scales",
            "v_packed", "v_signs", "v_high_mask", "v_residual_signs", "v_scales",
            "runtime_logical_length",
            "runtime_ring_offset",
            "runtime_pinned_prefix_length",
            "runtime_attention_scale",
            "runtime_block_count",
        ],
        outputNames: ["partial_stats", "partial_out"],
        source: fusedAttentionGQABlockPartialsSource_rf1,
        header: attentionHeader,
        ensureRowContiguous: false
    )

    // Layout v7 (tile-transposed): strided GQA block-partials kernel bound to the
    // v7-swizzled K planes. No coop variant exists for v7 (coop is excluded, see
    // turboQuantCooperativeQuadDecodeActive). See SPEC 2 section 3.
    static let fusedAttentionGQABlockPartialsV7 = MLXFast.metalKernel(
        name: "turboquant_attention_fused_gqa_block_partials_runtime_layout_rtu1_s2_v7",
        inputNames: [
            "q",
            "k_packed", "k_signs", "k_high_mask", "k_residual_signs", "k_scales",
            "v_packed", "v_signs", "v_high_mask", "v_residual_signs", "v_scales",
            "runtime_logical_length",
            "runtime_ring_offset",
            "runtime_pinned_prefix_length",
            "runtime_attention_scale",
            "runtime_block_count",
        ],
        outputNames: ["partial_stats", "partial_out"],
        source: fusedAttentionGQABlockPartialsV7Source,
        header: attentionHeader + attentionHeaderV7Extension,
        ensureRowContiguous: false
    )

    // TQCOOP: identical source to fusedAttentionGQABlockPartials, but a DISTINCT kernel
    // name so its compiled MLX variant (LANES_PER_TOKEN=4, cooperative coalesced decode)
    // never shares a compiled-variant cache slot with the strided LANES_PER_TOKEN=1
    // kernel. Selected only when turboQuantCooperativeQuadDecodeActive() is true.
    static let fusedAttentionGQABlockPartialsCoop_rf1 = MLXFast.metalKernel(
        name: "turboquant_attention_fused_gqa_block_partials_coop_runtime_layout_rtu1_s2_rf1",
        inputNames: [
            "q",
            "k_packed", "k_signs", "k_high_mask", "k_residual_signs", "k_scales",
            "v_packed", "v_signs", "v_high_mask", "v_residual_signs", "v_scales",
            "runtime_logical_length",
            "runtime_ring_offset",
            "runtime_pinned_prefix_length",
            "runtime_attention_scale",
            "runtime_block_count",
        ],
        outputNames: ["partial_stats", "partial_out"],
        source: fusedAttentionGQABlockPartialsSource_rf1,
        header: attentionHeader,
        ensureRowContiguous: false
    )

    // COOPW (T2.4 stage-2): repeats-2/3 coop. The clamp in fusedAttentionGQABlockPartialsSource_rf1
    // (r < repeat_count instead of the hardcoded r < 4u) makes that ONE source constant
    // correct for repeats 2, 3, and 4 -- so this registration reuses it unchanged. It still
    // needs a DISTINCT kernel name (same compiled-variant-cache reasoning as
    // fusedAttentionGQABlockPartialsCoop above) so repeats<4 dispatch never aliases with the
    // repeats==4 coop kernel's compiled variant. Selected only when
    // turboQuantCooperativeQuadDecodeActive() is true AND queryHeadRepeats < 4 (repeats==4
    // keeps using fusedAttentionGQABlockPartialsCoop above, byte-frozen).
    static let fusedAttentionGQABlockPartialsCoopW_rf1 = MLXFast.metalKernel(
        name: "turboquant_attention_fused_gqa_block_partials_coopw_runtime_layout_rtu1_s2_rf1",
        inputNames: [
            "q",
            "k_packed", "k_signs", "k_high_mask", "k_residual_signs", "k_scales",
            "v_packed", "v_signs", "v_high_mask", "v_residual_signs", "v_scales",
            "runtime_logical_length",
            "runtime_ring_offset",
            "runtime_pinned_prefix_length",
            "runtime_attention_scale",
            "runtime_block_count",
        ],
        outputNames: ["partial_stats", "partial_out"],
        source: fusedAttentionGQABlockPartialsSource_rf1,
        header: attentionHeader,
        ensureRowContiguous: false
    )

    // T2.2 H16 tgmem diet: same back-half algorithm as fusedAttentionGQABlockPartials /
    // fusedAttentionGQABlockPartialsCoop, but partial/tile_scores are staged as half and
    // tile_has_weight is a 32-lane bitset, halving static threadgroup memory so 2
    // threadgroups can be resident per core instead of 1 (see G5 probe / roadmap T2.2).
    // Default-off (TQ_H16 env gate); cosine-gated, not byte-identical to the fp32 kernel.
    // Shares LANES_PER_TOKEN with the strided/coop split (this constant is dispatched at
    // LANES_PER_TOKEN=1; the coop variant below reuses the identical source at LANES=4),
    // so it needs its own DISTINCT kernel name for the same compiled-variant-cache reason
    // as TQCOOP above.
    static let fusedAttentionGQABlockPartialsH16_rf1 = MLXFast.metalKernel(
        name: "turboquant_attention_fused_gqa_block_partials_runtime_layout_rtu1_s2_h16_rf1",
        inputNames: [
            "q",
            "k_packed", "k_signs", "k_high_mask", "k_residual_signs", "k_scales",
            "v_packed", "v_signs", "v_high_mask", "v_residual_signs", "v_scales",
            "runtime_logical_length",
            "runtime_ring_offset",
            "runtime_pinned_prefix_length",
            "runtime_attention_scale",
            "runtime_block_count",
        ],
        outputNames: ["partial_stats", "partial_out"],
        source: fusedAttentionGQABlockPartialsH16Source_rf1,
        header: attentionHeader,
        ensureRowContiguous: false
    )

    // T2.2 H16 diet + TQCOOP: identical source to fusedAttentionGQABlockPartialsH16
    // (LANES_PER_TOKEN=4 selects the cooperative quad-per-key branch at the template
    // level), but a DISTINCT kernel name so its compiled MLX variant never shares a
    // compiled-variant cache slot with the strided LANES_PER_TOKEN=1 H16 kernel above.
    static let fusedAttentionGQABlockPartialsCoopH16_rf1 = MLXFast.metalKernel(
        name: "turboquant_attention_fused_gqa_block_partials_coop_runtime_layout_rtu1_s2_h16_rf1",
        inputNames: [
            "q",
            "k_packed", "k_signs", "k_high_mask", "k_residual_signs", "k_scales",
            "v_packed", "v_signs", "v_high_mask", "v_residual_signs", "v_scales",
            "runtime_logical_length",
            "runtime_ring_offset",
            "runtime_pinned_prefix_length",
            "runtime_attention_scale",
            "runtime_block_count",
        ],
        outputNames: ["partial_stats", "partial_out"],
        source: fusedAttentionGQABlockPartialsH16Source_rf1,
        header: attentionHeader,
        ensureRowContiguous: false
    )

    static let fusedAttentionBlockReduce = MLXFast.metalKernel(
        name: "turboquant_attention_fused_block_reduce_rtu1_s2",
        inputNames: ["partial_stats", "partial_out", "runtime_block_count"],
        outputNames: ["out"],
        source: fusedAttentionBlockReduceSource,
        header: attentionHeader,
        ensureRowContiguous: false
    )

    static let segmentedRawAttentionStats = MLXFast.metalKernel(
        name: "turboquant_segmented_raw_attention_stats_rtu1_s2",
        inputNames: ["q", "k", "v", "runtime_attention_scale", "runtime_raw_length"],
        outputNames: ["partial_stats", "partial_out"],
        source: segmentedRawAttentionStatsSource,
        header: attentionHeader,
        ensureRowContiguous: false
    )

    private static let vectorHeader = """
        inline ulong tq_vector_mix_index(ulong seed, ulong index) {
            ulong mixed = seed + index * 0x9E3779B97F4A7C15ul;
            mixed ^= mixed >> 30;
            mixed *= 0xBF58476D1CE4E5B9ul;
            mixed ^= mixed >> 27;
            mixed *= 0x94D049BB133111EBul;
            mixed ^= mixed >> 31;
            return mixed;
        }

        inline bool tq_vector_random_sign(ulong seed, ulong index) {
            return (tq_vector_mix_index(seed, index) & 1ul) != 0ul;
        }

        inline ulong tq_make_seed(uint word3, uint word2, uint word1, uint word0) {
            return (ulong(word3) << 48)
                | (ulong(word2) << 32)
                | (ulong(word1) << 16)
                | ulong(word0);
        }

        inline ulong tq_product_channel_rank(ulong seed, uint group_index, uint local_index) {
            ulong state = seed;
            state ^= ulong(group_index) * 0x9E3779B97F4A7C15ul;
            state += ulong(local_index) * 0xD1B54A32D192ED03ul;
            state ^= state >> 30;
            state *= 0xBF58476D1CE4E5B9ul;
            state ^= state >> 27;
            state *= 0x94D049BB133111EBul;
            state ^= state >> 31;
            return state;
        }

        inline bool tq_product_high_precision(
            ulong seed,
            uint group_index,
            uint local,
            uint count,
            uint high_count
        ) {
            if (high_count == 0u) {
                return false;
            }
            if (high_count >= count) {
                return true;
            }
            ulong local_rank = tq_product_channel_rank(seed, group_index, local);
            uint rank = 0u;
            for (uint other = 0u; other < count; other++) {
                ulong other_rank = tq_product_channel_rank(seed, group_index, other);
                if (other_rank < local_rank || (other_rank == local_rank && other < local)) {
                    rank += 1u;
                }
            }
            return rank < high_count;
        }

        inline bool tq_split_high_precision(uint local, uint high_count) {
            return local < high_count;
        }

        inline uint tq_high_precision_count(uint count, uint numerator, uint denominator) {
            if (denominator == 0u) {
                return 0u;
            }
            return uint(round(float(count * numerator) / float(denominator)));
        }

        inline float tq_codebook_unit(uint bits, uint code) {
            if (bits <= 1u) {
                return code == 0u ? -0.797884561f : 0.797884561f;
            }
            if (bits == 2u) {
                switch (min(code, 3u)) {
                case 0u: return -1.510499245f;
                case 1u: return -0.452819573f;
                case 2u: return 0.452819573f;
                default: return 1.510499245f;
                }
            }
            if (bits == 3u) {
                switch (min(code, 7u)) {
                case 0u: return -2.175028018f;
                case 1u: return -1.367204388f;
                case 2u: return -0.773020220f;
                case 3u: return -0.251312159f;
                case 4u: return 0.251312159f;
                case 5u: return 0.773020220f;
                case 6u: return 1.367204388f;
                default: return 2.175028018f;
                }
            }
            if (bits == 5u) {
                uint clamped = min(code, 31u);
                uint magnitude_index = clamped < 16u ? clamped : 31u - clamped;
                float magnitude = 0.0f;
                switch (magnitude_index) {
                case 0u: magnitude = 3.167510584f; break;
                case 1u: magnitude = 2.601080629f; break;
                case 2u: magnitude = 2.248054067f; break;
                case 3u: magnitude = 1.990376987f; break;
                case 4u: magnitude = 1.784481424f; break;
                case 5u: magnitude = 1.607119170f; break;
                case 6u: magnitude = 1.444524024f; break;
                case 7u: magnitude = 1.288831640f; break;
                case 8u: magnitude = 1.135990256f; break;
                case 9u: magnitude = 0.984174410f; break;
                case 10u: magnitude = 0.832676140f; break;
                case 11u: magnitude = 0.681261776f; break;
                case 12u: magnitude = 0.529866428f; break;
                case 13u: magnitude = 0.378475081f; break;
                case 14u: magnitude = 0.227084777f; break;
                default: magnitude = 0.075694884f; break;
                }
                return clamped < 16u ? -magnitude : magnitude;
            }
            if (bits == 6u) {
                uint clamped = min(code, 63u);
                uint magnitude_index = clamped < 32u ? clamped : 63u - clamped;
                float magnitude = 0.0f;
                switch (magnitude_index) {
                case 0u: magnitude = 3.370567258f; break;
                case 1u: magnitude = 2.846634435f; break;
                case 2u: magnitude = 2.539498403f; break;
                case 3u: magnitude = 2.334801410f; break;
                case 4u: magnitude = 2.189068534f; break;
                case 5u: magnitude = 2.077692738f; break;
                case 6u: magnitude = 1.985038395f; break;
                case 7u: magnitude = 1.901543224f; break;
                case 8u: magnitude = 1.821977755f; break;
                case 9u: magnitude = 1.743867835f; break;
                case 10u: magnitude = 1.666217206f; break;
                case 11u: magnitude = 1.588688278f; break;
                case 12u: magnitude = 1.511185949f; break;
                case 13u: magnitude = 1.433688315f; break;
                case 14u: magnitude = 1.356191352f; break;
                case 15u: magnitude = 1.278694490f; break;
                case 16u: magnitude = 1.201197668f; break;
                case 17u: magnitude = 1.123700882f; break;
                case 18u: magnitude = 1.046204128f; break;
                case 19u: magnitude = 0.968707404f; break;
                case 20u: magnitude = 0.891210709f; break;
                case 21u: magnitude = 0.813714039f; break;
                case 22u: magnitude = 0.736217393f; break;
                case 23u: magnitude = 0.658720768f; break;
                case 24u: magnitude = 0.581224162f; break;
                case 25u: magnitude = 0.503727573f; break;
                case 26u: magnitude = 0.426230999f; break;
                case 27u: magnitude = 0.348734437f; break;
                case 28u: magnitude = 0.271237885f; break;
                case 29u: magnitude = 0.193741341f; break;
                case 30u: magnitude = 0.116244802f; break;
                default: magnitude = 0.038748267f; break;
                }
                return clamped < 32u ? -magnitude : magnitude;
            }
            if (bits >= 7u) {
                uint clamped = min(code, 127u);
                uint magnitude_index = clamped < 64u ? clamped : 127u - clamped;
                float magnitude = 0.0f;
                switch (magnitude_index) {
                case 0u: magnitude = 3.471692079f; break;
                case 1u: magnitude = 2.967922351f; break;
                case 2u: magnitude = 2.682760472f; break;
                case 3u: magnitude = 2.503778860f; break;
                case 4u: magnitude = 2.387735667f; break;
                case 5u: magnitude = 2.309487569f; break;
                case 6u: magnitude = 2.252475440f; break;
                case 7u: magnitude = 2.206174364f; break;
                case 8u: magnitude = 2.164605881f; break;
                case 9u: magnitude = 2.124839178f; break;
                case 10u: magnitude = 2.085655475f; break;
                case 11u: magnitude = 2.046629763f; break;
                case 12u: magnitude = 2.007639247f; break;
                case 13u: magnitude = 1.968655011f; break;
                case 14u: magnitude = 1.929671639f; break;
                case 15u: magnitude = 1.890688353f; break;
                case 16u: magnitude = 1.851705074f; break;
                case 17u: magnitude = 1.812721796f; break;
                case 18u: magnitude = 1.773738519f; break;
                case 19u: magnitude = 1.734755243f; break;
                case 20u: magnitude = 1.695771967f; break;
                case 21u: magnitude = 1.656788693f; break;
                case 22u: magnitude = 1.617805419f; break;
                case 23u: magnitude = 1.578822145f; break;
                case 24u: magnitude = 1.539838873f; break;
                case 25u: magnitude = 1.500855601f; break;
                case 26u: magnitude = 1.461872330f; break;
                case 27u: magnitude = 1.422889060f; break;
                case 28u: magnitude = 1.383905790f; break;
                case 29u: magnitude = 1.344922521f; break;
                case 30u: magnitude = 1.305939253f; break;
                case 31u: magnitude = 1.266955985f; break;
                case 32u: magnitude = 1.227972718f; break;
                case 33u: magnitude = 1.188989451f; break;
                case 34u: magnitude = 1.150006185f; break;
                case 35u: magnitude = 1.111022919f; break;
                case 36u: magnitude = 1.072039654f; break;
                case 37u: magnitude = 1.033056390f; break;
                case 38u: magnitude = 0.994073126f; break;
                case 39u: magnitude = 0.955089862f; break;
                case 40u: magnitude = 0.916106599f; break;
                case 41u: magnitude = 0.877123336f; break;
                case 42u: magnitude = 0.838140074f; break;
                case 43u: magnitude = 0.799156812f; break;
                case 44u: magnitude = 0.760173551f; break;
                case 45u: magnitude = 0.721190290f; break;
                case 46u: magnitude = 0.682207029f; break;
                case 47u: magnitude = 0.643223768f; break;
                case 48u: magnitude = 0.604240508f; break;
                case 49u: magnitude = 0.565257248f; break;
                case 50u: magnitude = 0.526273989f; break;
                case 51u: magnitude = 0.487290729f; break;
                case 52u: magnitude = 0.448307470f; break;
                case 53u: magnitude = 0.409324211f; break;
                case 54u: magnitude = 0.370340952f; break;
                case 55u: magnitude = 0.331357694f; break;
                case 56u: magnitude = 0.292374435f; break;
                case 57u: magnitude = 0.253391177f; break;
                case 58u: magnitude = 0.214407919f; break;
                case 59u: magnitude = 0.175424661f; break;
                case 60u: magnitude = 0.136441403f; break;
                case 61u: magnitude = 0.097458145f; break;
                case 62u: magnitude = 0.058474887f; break;
                default: magnitude = 0.019491629f; break;
                }
                return clamped < 64u ? -magnitude : magnitude;
            }
            switch (min(code, 15u)) {
            case 0u: return -2.778927695f;
            case 1u: return -2.124836923f;
            case 2u: return -1.680512470f;
            case 3u: return -1.321175453f;
            case 4u: return -1.003692455f;
            case 5u: return -0.707453186f;
            case 6u: return -0.421537889f;
            case 7u: return -0.140103661f;
            case 8u: return 0.140103661f;
            case 9u: return 0.421537889f;
            case 10u: return 0.707453186f;
            case 11u: return 1.003692455f;
            case 12u: return 1.321175453f;
            case 13u: return 1.680512470f;
            case 14u: return 2.124836923f;
            default: return 2.778927695f;
            }
        }

        inline float tq_codebook_level(uint bits, uint code, uint count) {
            return tq_codebook_unit(bits, code) * rsqrt(float(max(count, 1u)));
        }

        inline uint tq_nearest_codebook_index(float value, uint bits, uint count) {
            uint level_count = 1u << bits;
            uint low = 0u;
            uint high = level_count - 1u;
            while (low < high) {
                uint mid = (low + high) >> 1u;
                float boundary =
                    0.5f * (tq_codebook_level(bits, mid, count)
                        + tq_codebook_level(bits, mid + 1u, count));
                if (value <= boundary) {
                    high = mid;
                } else {
                    low = mid + 1u;
                }
            }
            return low;
        }

        inline void tq_fast_hadamard(thread float* values, uint count) {
            for (uint width = 1u; width < count; width <<= 1u) {
                for (uint start = 0u; start < count; start += width << 1u) {
                    for (uint offset = 0u; offset < width; offset++) {
                        float lhs = values[start + offset];
                        float rhs = values[start + offset + width];
                        values[start + offset] = lhs + rhs;
                        values[start + offset + width] = lhs - rhs;
                    }
                }
            }
        }

        inline void tq_apply_rotation_signs(
            thread float* values,
            uint count,
            ulong seed,
            uint group_index
        ) {
            for (uint local = 0u; local < count; local++) {
                ulong sign_index = ulong(group_index) * 4099ul + ulong(local);
                if (tq_vector_random_sign(seed, sign_index)) {
                    values[local] = -values[local];
                }
            }
        }

        inline void tq_apply_givens_pass(
            thread float* values,
            uint count,
            ulong seed,
            uint group_index,
            uint pass,
            float direction
        ) {
            uint offset = pass & 1u;
            for (uint index = offset; index + 1u < count; index += 2u) {
                ulong angle_rank = tq_product_channel_rank(
                    seed ^ (ulong(pass) * 0xA24BAED4963EE407ul),
                    group_index,
                    index >> 1u);
                float unit = float(uint(angle_rank)) / 4294967295.0f;
                float angle = (unit - 0.5f) * 3.14159265358979323846f * direction;
                float c = cos(angle);
                float s = sin(angle);
                float lhs = values[index];
                float rhs = values[index + 1u];
                values[index] = c * lhs - s * rhs;
                values[index + 1u] = s * lhs + c * rhs;
            }
        }

        inline void tq_apply_product_rotation(
            thread float* values,
            uint count,
            ulong seed,
            uint group_index,
            bool inverse
        ) {
            if (count <= 1u) {
                tq_apply_rotation_signs(values, count, seed, group_index);
                return;
            }
            if ((count & (count - 1u)) == 0u) {
                if (inverse) {
                    tq_fast_hadamard(values, count);
                    tq_apply_rotation_signs(values, count, seed, group_index);
                } else {
                    tq_apply_rotation_signs(values, count, seed, group_index);
                    tq_fast_hadamard(values, count);
                }
                float scale = rsqrt(float(count));
                for (uint local = 0u; local < count; local++) {
                    values[local] *= scale;
                }
                return;
            }
            if (inverse) {
                for (uint pass_index = 0u; pass_index < 4u; pass_index++) {
                    tq_apply_givens_pass(values, count, seed, group_index, 3u - pass_index, -1.0f);
                }
            } else {
                for (uint pass = 0u; pass < 4u; pass++) {
                    tq_apply_givens_pass(values, count, seed, group_index, pass, 1.0f);
                }
            }
        }

        template <typename UIntPtr>
        inline bool tq_flat_high_precision(
            UIntPtr high_mask,
            uint group_id,
            uint local,
            uint bitset_words_per_group
        ) {
            uint bitset_base = group_id * bitset_words_per_group;
            uint word_index = local >> 5;
            uint word_bit = local & 31u;
            return (high_mask[bitset_base + word_index] & (1u << word_bit)) != 0u;
        }

        template <typename UIntPtr>
        inline uint tq_flat_high_count_before(
            UIntPtr high_mask,
            uint group_id,
            uint local,
            uint bitset_words_per_group
        ) {
            uint bitset_base = group_id * bitset_words_per_group;
            uint full_words = local >> 5;
            uint count = 0u;
            for (uint word = 0u; word < full_words; word++) {
                count += popcount(high_mask[bitset_base + word]);
            }
            uint remainder = local & 31u;
            if (remainder > 0u && full_words < bitset_words_per_group) {
                uint mask = (1u << remainder) - 1u;
                count += popcount(high_mask[bitset_base + full_words] & mask);
            }
            return count;
        }

        template <typename PackedPtr, typename HighMaskPtr>
        inline uint tq_read_flat_code(
            PackedPtr packed,
            HighMaskPtr high_mask,
            uint group_id,
            uint local,
            uint mag_words_per_group,
            uint bitset_words_per_group,
            uint base_bits,
            uint high_bits
        ) {
            uint packed_base = group_id * mag_words_per_group;
            bool high_precision = tq_flat_high_precision(
                high_mask, group_id, local, bitset_words_per_group);
            uint bits = high_precision ? high_bits : base_bits;
            uint high_before = tq_flat_high_count_before(
                high_mask, group_id, local, bitset_words_per_group);
            uint bit_offset = local * base_bits + high_before * (high_bits - base_bits);

            uint quantized = 0u;
            for (uint bit = 0u; bit < bits; bit++) {
                uint global_bit = bit_offset + bit;
                uint packed_word = global_bit >> 5;
                uint packed_bit = global_bit & 31u;
                if ((packed[packed_base + packed_word] & (1u << packed_bit)) != 0u) {
                    quantized |= 1u << bit;
                }
            }
            return quantized;
        }

        template <
            typename PackedPtr,
            typename SignsPtr,
            typename HighMaskPtr,
            typename ResidualSignsPtr,
            typename ScalesPtr
        >
        inline float tq_decode_flat_value(
            PackedPtr packed,
            SignsPtr signs,
            HighMaskPtr high_mask,
            ResidualSignsPtr residual_signs,
            ScalesPtr scales,
            uint index,
            ulong seed,
            uint role,
            uint group_size,
            uint mag_words_per_group,
            uint bitset_words_per_group,
            uint base_bits,
            uint high_bits,
            uint key_base_bits,
            uint key_high_bits,
            uint value_bits,
            uint scales_per_group,
            uint value_count
        ) {
            uint group_id = index / group_size;
            uint local = index - group_id * group_size;
            uint packed_base = group_id * mag_words_per_group;
            if (role == 1u) {
                uint bit_offset = local * value_bits;
                uint quantized = 0u;
                for (uint bit = 0u; bit < value_bits; bit++) {
                    uint global_bit = bit_offset + bit;
                    uint packed_word = global_bit >> 5;
                    uint packed_bit = global_bit & 31u;
                    if ((packed[packed_base + packed_word] & (1u << packed_bit)) != 0u) {
                        quantized |= 1u << bit;
                    }
                }
                uint scale_base = group_id * scales_per_group;
                return scales[scale_base + 1u] + float(quantized) * scales[scale_base];
            }

            uint count = min(group_size, value_count - group_id * group_size);
            thread float rotated[128];
            for (uint decode_local = 0u; decode_local < count; decode_local++) {
                bool high_precision = tq_flat_high_precision(
                    high_mask, group_id, decode_local, bitset_words_per_group);
                uint bits = high_precision ? key_high_bits : key_base_bits;
                uint code = tq_read_flat_code(
                    packed, high_mask, group_id, decode_local,
                    mag_words_per_group, bitset_words_per_group,
                    key_base_bits, key_high_bits);
                rotated[decode_local] = tq_codebook_level(bits, code, count);
            }
            tq_apply_product_rotation(rotated, count, seed, group_id, true);
            return rotated[local] * scales[group_id * scales_per_group];
        }

        template <
            typename PackedPtr,
            typename SignsPtr,
            typename HighMaskPtr,
            typename ScalesPtr
        >
        inline float tq_flat_product_inner_product_group(
            PackedPtr packed,
            SignsPtr signs,
            HighMaskPtr high_mask,
            ScalesPtr scales,
            thread float* query_values,
            uint group_id,
            ulong seed,
            uint count,
            uint mag_words_per_group,
            uint bitset_words_per_group,
            uint key_base_bits,
            uint key_high_bits,
            uint scales_per_group
        ) {
            tq_apply_product_rotation(query_values, count, seed, group_id, false);
            float quantized_dot = 0.0f;
            float sign_dot = 0.0f;
            uint bitset_base = group_id * bitset_words_per_group;
            for (uint local = 0u; local < count; local++) {
                bool high_precision = tq_flat_high_precision(
                    high_mask, group_id, local, bitset_words_per_group);
                uint bits = high_precision ? key_high_bits : key_base_bits;
                uint code = tq_read_flat_code(
                    packed, high_mask, group_id, local,
                    mag_words_per_group, bitset_words_per_group,
                    key_base_bits, key_high_bits);
                quantized_dot += query_values[local] * tq_codebook_level(bits, code, count);

                uint word_index = local >> 5;
                uint word_bit = local & 31u;
                float qjl_sign =
                    (signs[bitset_base + word_index] & (1u << word_bit)) != 0u
                    ? -1.0f : 1.0f;
                sign_dot += qjl_sign * query_values[local];
            }

            float norm = scales[group_id * scales_per_group];
            float residual_norm = scales[group_id * scales_per_group + 1u];
            float residual =
                residual_norm * sqrt(3.14159265358979323846f / (2.0f * float(count)))
                * sign_dot;
            return norm * quantized_dot + residual;
        }
        """

    private static let encodeSource = """
        uint group_id = thread_position_in_grid.x;
        if (group_id >= GROUP_COUNT) {
            return;
        }

        uint start = group_id * GROUP_SIZE;
        uint count = min(uint(GROUP_SIZE), uint(VALUE_COUNT) - start);
        if (count == 0) {
            return;
        }

        thread float values[GROUP_SIZE];
        ulong seed = tq_make_seed(uint(SEED_3), uint(SEED_2), uint(SEED_1), uint(SEED_0));

        if (ROLE == 1) {
            float minimum = INFINITY;
            float maximum = -INFINITY;
            for (uint local = 0; local < count; local++) {
                float value = float(x[start + local]);
                minimum = min(minimum, value);
                maximum = max(maximum, value);
            }

            float value_max = float((1 << VALUE_BITS) - 1);
            float range = maximum - minimum;
            float value_scale = range > 1.17549435e-38f ? range / value_max : 0.0f;
            uint scale_base = group_id * uint(SCALES_PER_GROUP);
            scales[scale_base] = value_scale;
            scales[scale_base + 1] = minimum;

            uint packed_base = group_id * MAG_WORDS_PER_GROUP;
            for (uint word = 0; word < MAG_WORDS_PER_GROUP; word++) {
                packed[packed_base + word] = 0u;
            }

            for (uint local = 0; local < count; local++) {
                float value = float(x[start + local]);
                uint quantized = value_scale == 0.0f
                    ? 0u
                    : uint(clamp(round((value - minimum) / value_scale), 0.0f, value_max));
                uint bit_offset = local * uint(VALUE_BITS);
                for (uint bit = 0; bit < uint(VALUE_BITS); bit++) {
                    if ((quantized & (1u << bit)) != 0u) {
                        uint global_bit = bit_offset + bit;
                        uint packed_word = global_bit >> 5;
                        uint packed_bit = global_bit & 31u;
                        packed[packed_base + packed_word] |= 1u << packed_bit;
                    }
                }
            }
            return;
        }

        float norm_squared = 0.0f;
        for (uint local = 0; local < count; local++) {
            float value = float(x[start + local]);
            values[local] = value;
            norm_squared += value * value;
        }

        float norm = sqrt(norm_squared);
        float inv_norm = norm > 1.17549435e-38f ? 1.0f / norm : 0.0f;
        for (uint local = 0; local < count; local++) {
            values[local] *= inv_norm;
        }
        tq_apply_product_rotation(values, count, seed, group_id, false);

        uint scale_base = group_id * uint(SCALES_PER_GROUP);
        scales[scale_base] = norm;
        scales[scale_base + 1] = 0.0f;

        uint bitset_base = group_id * BITSET_WORDS_PER_GROUP;
        for (uint word = 0; word < BITSET_WORDS_PER_GROUP; word++) {
            signs[bitset_base + word] = 0u;
            high_mask[bitset_base + word] = 0u;
        }

        uint packed_base = group_id * MAG_WORDS_PER_GROUP;
        for (uint word = 0; word < MAG_WORDS_PER_GROUP; word++) {
            packed[packed_base + word] = 0u;
        }

        uint high_count = uint(round(float(count * uint(HIGH_NUMERATOR)) / float(uint(HIGH_DENOMINATOR))));
        float residual_squared = 0.0f;
        uint bit_offset = 0;
        for (uint local = 0; local < count; local++) {
            bool high_precision = tq_product_high_precision(seed, group_id, local, count, high_count);
            uint bits = high_precision ? uint(KEY_HIGH_BITS) : uint(KEY_BASE_BITS);
            uint quantized = tq_nearest_codebook_index(values[local], bits, count);
            float reconstructed = tq_codebook_level(bits, quantized, count);

            uint word_index = local >> 5;
            uint word_bit = local & 31u;
            uint mask_bit = 1u << word_bit;
            if (high_precision) {
                high_mask[bitset_base + word_index] |= mask_bit;
            }
            float residual = values[local] - reconstructed;
            residual_squared += residual * residual;
            if (residual < 0.0f) {
                signs[bitset_base + word_index] |= mask_bit;
            }

            for (uint bit = 0; bit < bits; bit++) {
                if ((quantized & (1u << bit)) != 0u) {
                    uint global_bit = bit_offset + bit;
                    uint packed_word = global_bit >> 5;
                    uint packed_bit = global_bit & 31u;
                    packed[packed_base + packed_word] |= 1u << packed_bit;
                }
            }
            bit_offset += bits;
        }
        scales[scale_base + 1] = norm * sqrt(residual_squared);
        """

    private static let decodeSource = """
        uint index = thread_position_in_grid.x;
        if (index >= VALUE_COUNT) {
            return;
        }

        ulong seed = tq_make_seed(uint(SEED_3), uint(SEED_2), uint(SEED_1), uint(SEED_0));
        uint group_id = index / uint(GROUP_SIZE);
        uint local = index - group_id * uint(GROUP_SIZE);
        uint packed_base = group_id * uint(MAG_WORDS_PER_GROUP);
        if (ROLE == 1) {
            uint bit_offset = local * uint(VALUE_BITS);
            uint quantized = 0u;
            for (uint bit = 0; bit < uint(VALUE_BITS); bit++) {
                uint global_bit = bit_offset + bit;
                uint packed_word = global_bit >> 5;
                uint packed_bit = global_bit & 31u;
                if ((packed[packed_base + packed_word] & (1u << packed_bit)) != 0u) {
                    quantized |= 1u << bit;
                }
            }
            uint scale_base = group_id * uint(SCALES_PER_GROUP);
            out[index] = static_cast<OUTPUT_DTYPE>(
                scales[scale_base + 1] + float(quantized) * scales[scale_base]);
            return;
        }

        uint count = min(uint(GROUP_SIZE), uint(VALUE_COUNT) - group_id * uint(GROUP_SIZE));
        thread float rotated[GROUP_SIZE];
        uint bitset_base = group_id * uint(BITSET_WORDS_PER_GROUP);
        for (uint decode_local = 0u; decode_local < count; decode_local++) {
            uint word_index = decode_local >> 5;
            uint word_bit = decode_local & 31u;
            bool high_precision = (high_mask[bitset_base + word_index] & (1u << word_bit)) != 0u;
            uint bits = high_precision ? uint(KEY_HIGH_BITS) : uint(KEY_BASE_BITS);
            uint code = tq_read_flat_code(
                packed, high_mask, group_id, decode_local,
                uint(MAG_WORDS_PER_GROUP), uint(BITSET_WORDS_PER_GROUP),
                uint(KEY_BASE_BITS), uint(KEY_HIGH_BITS));
            rotated[decode_local] = tq_codebook_level(bits, code, count);
        }
        tq_apply_product_rotation(rotated, count, seed, group_id, true);
        out[index] = static_cast<OUTPUT_DTYPE>(
            rotated[local] * scales[group_id * uint(SCALES_PER_GROUP)]);
        """

    private static let matmulSource = """
        uint index = thread_position_in_grid.x;
        uint total = uint(X_ROWS) * (TRANSPOSE_WEIGHT ? uint(WEIGHT_ROWS) : uint(WEIGHT_COLUMNS));
        if (index >= total) {
            return;
        }

        uint output_columns = TRANSPOSE_WEIGHT ? uint(WEIGHT_ROWS) : uint(WEIGHT_COLUMNS);
        uint row = index / output_columns;
        uint column = index - row * output_columns;
        uint reduction = uint(X_COLUMNS);
        ulong seed = tq_make_seed(uint(SEED_3), uint(SEED_2), uint(SEED_1), uint(SEED_0));
        float sum = 0.0f;

        if (uint(ROLE) != 1u && TRANSPOSE_WEIGHT
            && (uint(WEIGHT_COLUMNS) % uint(GROUP_SIZE)) == 0u
            && (reduction % uint(GROUP_SIZE)) == 0u) {
            for (uint group_start = 0u; group_start < reduction; group_start += uint(GROUP_SIZE)) {
                uint count = min(uint(GROUP_SIZE), reduction - group_start);
                thread float query_values[GROUP_SIZE];
                for (uint local = 0u; local < count; local++) {
                    query_values[local] = float(x[row * uint(X_COLUMNS) + group_start + local]);
                }
                uint weight_group =
                    (column * uint(WEIGHT_COLUMNS) + group_start) / uint(GROUP_SIZE);
                sum += tq_flat_product_inner_product_group(
                    packed, signs, high_mask, scales, query_values,
                    weight_group, seed, count,
                    uint(MAG_WORDS_PER_GROUP), uint(BITSET_WORDS_PER_GROUP),
                    uint(KEY_BASE_BITS), uint(KEY_HIGH_BITS), uint(SCALES_PER_GROUP));
            }
            out[index] = static_cast<OUTPUT_DTYPE>(sum);
            return;
        }

        for (uint k = 0u; k < reduction; k++) {
            uint x_index = row * uint(X_COLUMNS) + k;
            uint weight_index = TRANSPOSE_WEIGHT
                ? column * uint(WEIGHT_COLUMNS) + k
                : k * uint(WEIGHT_COLUMNS) + column;
            float weight = tq_decode_flat_value(
                packed, signs, high_mask, residual_signs, scales,
                weight_index, seed, uint(ROLE),
                uint(GROUP_SIZE), uint(MAG_WORDS_PER_GROUP), uint(BITSET_WORDS_PER_GROUP),
                uint(BASE_BITS), uint(HIGH_BITS), uint(KEY_BASE_BITS), uint(KEY_HIGH_BITS),
                uint(VALUE_BITS), uint(SCALES_PER_GROUP), uint(VALUE_COUNT));
            sum += float(x[x_index]) * weight;
        }
        out[index] = static_cast<OUTPUT_DTYPE>(sum);
        """

    private static let attentionHeader = """
        inline ulong tq_mix(ulong seed, uint index) {
            ulong mixed = seed + ulong(index) * 0x9E3779B97F4A7C15ul;
            mixed ^= mixed >> 30;
            mixed *= 0xBF58476D1CE4E5B9ul;
            mixed ^= mixed >> 27;
            mixed *= 0x94D049BB133111EBul;
            mixed ^= mixed >> 31;
            return mixed;
        }

        inline bool tq_random_sign(ulong seed, uint index) {
            return (tq_mix(seed, index) & 1ul) != 0ul;
        }

        inline ulong tq_mix_index(ulong seed, ulong index) {
            ulong mixed = seed + index * 0x9E3779B97F4A7C15ul;
            mixed ^= mixed >> 30;
            mixed *= 0xBF58476D1CE4E5B9ul;
            mixed ^= mixed >> 27;
            mixed *= 0x94D049BB133111EBul;
            mixed ^= mixed >> 31;
            return mixed;
        }

        inline bool tq_random_sign_index(ulong seed, ulong index) {
            return (tq_mix_index(seed, index) & 1ul) != 0ul;
        }

        inline ulong tq_make_seed(uint word3, uint word2, uint word1, uint word0) {
            return (ulong(word3) << 48)
                | (ulong(word2) << 32)
                | (ulong(word1) << 16)
                | ulong(word0);
        }

        inline ulong tq_product_channel_rank(ulong seed, uint group_index, uint local_index) {
            ulong state = seed;
            state ^= ulong(group_index) * 0x9E3779B97F4A7C15ul;
            state += ulong(local_index) * 0xD1B54A32D192ED03ul;
            state ^= state >> 30;
            state *= 0xBF58476D1CE4E5B9ul;
            state ^= state >> 27;
            state *= 0x94D049BB133111EBul;
            state ^= state >> 31;
            return state;
        }

        inline bool tq_product_high_precision(
            ulong seed,
            uint group_index,
            uint local,
            uint count,
            uint high_count
        ) {
            if (high_count == 0u) {
                return false;
            }
            if (high_count >= count) {
                return true;
            }
            ulong local_rank = tq_product_channel_rank(seed, group_index, local);
            uint rank = 0u;
            for (uint other = 0u; other < count; other++) {
                ulong other_rank = tq_product_channel_rank(seed, group_index, other);
                if (other_rank < local_rank || (other_rank == local_rank && other < local)) {
                    rank += 1u;
                }
            }
            return rank < high_count;
        }

        inline bool tq_split_high_precision(uint local, uint high_count) {
            return local < high_count;
        }

        inline uint tq_high_precision_count(uint count, uint numerator, uint denominator) {
            if (denominator == 0u) {
                return 0u;
            }
            return uint(round(float(count * numerator) / float(denominator)));
        }

        inline float tq_codebook_unit(uint bits, uint code) {
            if (bits <= 1u) {
                return code == 0u ? -0.797884561f : 0.797884561f;
            }
            if (bits == 2u) {
                switch (min(code, 3u)) {
                case 0u: return -1.510499245f;
                case 1u: return -0.452819573f;
                case 2u: return 0.452819573f;
                default: return 1.510499245f;
                }
            }
            if (bits == 3u) {
                switch (min(code, 7u)) {
                case 0u: return -2.175028018f;
                case 1u: return -1.367204388f;
                case 2u: return -0.773020220f;
                case 3u: return -0.251312159f;
                case 4u: return 0.251312159f;
                case 5u: return 0.773020220f;
                case 6u: return 1.367204388f;
                default: return 2.175028018f;
                }
            }
            if (bits == 5u) {
                uint clamped = min(code, 31u);
                uint magnitude_index = clamped < 16u ? clamped : 31u - clamped;
                float magnitude = 0.0f;
                switch (magnitude_index) {
                case 0u: magnitude = 3.167510584f; break;
                case 1u: magnitude = 2.601080629f; break;
                case 2u: magnitude = 2.248054067f; break;
                case 3u: magnitude = 1.990376987f; break;
                case 4u: magnitude = 1.784481424f; break;
                case 5u: magnitude = 1.607119170f; break;
                case 6u: magnitude = 1.444524024f; break;
                case 7u: magnitude = 1.288831640f; break;
                case 8u: magnitude = 1.135990256f; break;
                case 9u: magnitude = 0.984174410f; break;
                case 10u: magnitude = 0.832676140f; break;
                case 11u: magnitude = 0.681261776f; break;
                case 12u: magnitude = 0.529866428f; break;
                case 13u: magnitude = 0.378475081f; break;
                case 14u: magnitude = 0.227084777f; break;
                default: magnitude = 0.075694884f; break;
                }
                return clamped < 16u ? -magnitude : magnitude;
            }
            if (bits == 6u) {
                uint clamped = min(code, 63u);
                uint magnitude_index = clamped < 32u ? clamped : 63u - clamped;
                float magnitude = 0.0f;
                switch (magnitude_index) {
                case 0u: magnitude = 3.370567258f; break;
                case 1u: magnitude = 2.846634435f; break;
                case 2u: magnitude = 2.539498403f; break;
                case 3u: magnitude = 2.334801410f; break;
                case 4u: magnitude = 2.189068534f; break;
                case 5u: magnitude = 2.077692738f; break;
                case 6u: magnitude = 1.985038395f; break;
                case 7u: magnitude = 1.901543224f; break;
                case 8u: magnitude = 1.821977755f; break;
                case 9u: magnitude = 1.743867835f; break;
                case 10u: magnitude = 1.666217206f; break;
                case 11u: magnitude = 1.588688278f; break;
                case 12u: magnitude = 1.511185949f; break;
                case 13u: magnitude = 1.433688315f; break;
                case 14u: magnitude = 1.356191352f; break;
                case 15u: magnitude = 1.278694490f; break;
                case 16u: magnitude = 1.201197668f; break;
                case 17u: magnitude = 1.123700882f; break;
                case 18u: magnitude = 1.046204128f; break;
                case 19u: magnitude = 0.968707404f; break;
                case 20u: magnitude = 0.891210709f; break;
                case 21u: magnitude = 0.813714039f; break;
                case 22u: magnitude = 0.736217393f; break;
                case 23u: magnitude = 0.658720768f; break;
                case 24u: magnitude = 0.581224162f; break;
                case 25u: magnitude = 0.503727573f; break;
                case 26u: magnitude = 0.426230999f; break;
                case 27u: magnitude = 0.348734437f; break;
                case 28u: magnitude = 0.271237885f; break;
                case 29u: magnitude = 0.193741341f; break;
                case 30u: magnitude = 0.116244802f; break;
                default: magnitude = 0.038748267f; break;
                }
                return clamped < 32u ? -magnitude : magnitude;
            }
            if (bits >= 7u) {
                uint clamped = min(code, 127u);
                uint magnitude_index = clamped < 64u ? clamped : 127u - clamped;
                float magnitude = 0.0f;
                switch (magnitude_index) {
                case 0u: magnitude = 3.471692079f; break;
                case 1u: magnitude = 2.967922351f; break;
                case 2u: magnitude = 2.682760472f; break;
                case 3u: magnitude = 2.503778860f; break;
                case 4u: magnitude = 2.387735667f; break;
                case 5u: magnitude = 2.309487569f; break;
                case 6u: magnitude = 2.252475440f; break;
                case 7u: magnitude = 2.206174364f; break;
                case 8u: magnitude = 2.164605881f; break;
                case 9u: magnitude = 2.124839178f; break;
                case 10u: magnitude = 2.085655475f; break;
                case 11u: magnitude = 2.046629763f; break;
                case 12u: magnitude = 2.007639247f; break;
                case 13u: magnitude = 1.968655011f; break;
                case 14u: magnitude = 1.929671639f; break;
                case 15u: magnitude = 1.890688353f; break;
                case 16u: magnitude = 1.851705074f; break;
                case 17u: magnitude = 1.812721796f; break;
                case 18u: magnitude = 1.773738519f; break;
                case 19u: magnitude = 1.734755243f; break;
                case 20u: magnitude = 1.695771967f; break;
                case 21u: magnitude = 1.656788693f; break;
                case 22u: magnitude = 1.617805419f; break;
                case 23u: magnitude = 1.578822145f; break;
                case 24u: magnitude = 1.539838873f; break;
                case 25u: magnitude = 1.500855601f; break;
                case 26u: magnitude = 1.461872330f; break;
                case 27u: magnitude = 1.422889060f; break;
                case 28u: magnitude = 1.383905790f; break;
                case 29u: magnitude = 1.344922521f; break;
                case 30u: magnitude = 1.305939253f; break;
                case 31u: magnitude = 1.266955985f; break;
                case 32u: magnitude = 1.227972718f; break;
                case 33u: magnitude = 1.188989451f; break;
                case 34u: magnitude = 1.150006185f; break;
                case 35u: magnitude = 1.111022919f; break;
                case 36u: magnitude = 1.072039654f; break;
                case 37u: magnitude = 1.033056390f; break;
                case 38u: magnitude = 0.994073126f; break;
                case 39u: magnitude = 0.955089862f; break;
                case 40u: magnitude = 0.916106599f; break;
                case 41u: magnitude = 0.877123336f; break;
                case 42u: magnitude = 0.838140074f; break;
                case 43u: magnitude = 0.799156812f; break;
                case 44u: magnitude = 0.760173551f; break;
                case 45u: magnitude = 0.721190290f; break;
                case 46u: magnitude = 0.682207029f; break;
                case 47u: magnitude = 0.643223768f; break;
                case 48u: magnitude = 0.604240508f; break;
                case 49u: magnitude = 0.565257248f; break;
                case 50u: magnitude = 0.526273989f; break;
                case 51u: magnitude = 0.487290729f; break;
                case 52u: magnitude = 0.448307470f; break;
                case 53u: magnitude = 0.409324211f; break;
                case 54u: magnitude = 0.370340952f; break;
                case 55u: magnitude = 0.331357694f; break;
                case 56u: magnitude = 0.292374435f; break;
                case 57u: magnitude = 0.253391177f; break;
                case 58u: magnitude = 0.214407919f; break;
                case 59u: magnitude = 0.175424661f; break;
                case 60u: magnitude = 0.136441403f; break;
                case 61u: magnitude = 0.097458145f; break;
                case 62u: magnitude = 0.058474887f; break;
                default: magnitude = 0.019491629f; break;
                }
                return clamped < 64u ? -magnitude : magnitude;
            }
            switch (min(code, 15u)) {
            case 0u: return -2.778927695f;
            case 1u: return -2.124836923f;
            case 2u: return -1.680512470f;
            case 3u: return -1.321175453f;
            case 4u: return -1.003692455f;
            case 5u: return -0.707453186f;
            case 6u: return -0.421537889f;
            case 7u: return -0.140103661f;
            case 8u: return 0.140103661f;
            case 9u: return 0.421537889f;
            case 10u: return 0.707453186f;
            case 11u: return 1.003692455f;
            case 12u: return 1.321175453f;
            case 13u: return 1.680512470f;
            case 14u: return 2.124836923f;
            default: return 2.778927695f;
            }
        }

        inline float tq_codebook_level(uint bits, uint code, uint count) {
            return tq_codebook_unit(bits, code) * rsqrt(float(max(count, 1u)));
        }

        inline uint tq_nearest_codebook_index(float value, uint bits, uint count) {
            uint level_count = 1u << bits;
            uint low = 0u;
            uint high = level_count - 1u;
            while (low < high) {
                uint mid = (low + high) >> 1u;
                float boundary =
                    0.5f * (tq_codebook_level(bits, mid, count)
                        + tq_codebook_level(bits, mid + 1u, count));
                if (value <= boundary) {
                    high = mid;
                } else {
                    low = mid + 1u;
                }
            }
            return low;
        }

        inline void tq_fast_hadamard(thread float* values, uint count) {
            for (uint width = 1u; width < count; width <<= 1u) {
                for (uint start = 0u; start < count; start += width << 1u) {
                    for (uint offset = 0u; offset < width; offset++) {
                        float lhs = values[start + offset];
                        float rhs = values[start + offset + width];
                        values[start + offset] = lhs + rhs;
                        values[start + offset + width] = lhs - rhs;
                    }
                }
            }
        }

        inline void tq_apply_rotation_signs(
            thread float* values,
            uint count,
            ulong seed,
            uint group_index
        ) {
            for (uint local = 0u; local < count; local++) {
                ulong sign_index = ulong(group_index) * 4099ul + ulong(local);
                if (tq_random_sign_index(seed, sign_index)) {
                    values[local] = -values[local];
                }
            }
        }

        inline void tq_apply_givens_pass(
            thread float* values,
            uint count,
            ulong seed,
            uint group_index,
            uint pass,
            float direction
        ) {
            uint offset = pass & 1u;
            for (uint index = offset; index + 1u < count; index += 2u) {
                ulong angle_rank = tq_product_channel_rank(
                    seed ^ (ulong(pass) * 0xA24BAED4963EE407ul),
                    group_index,
                    index >> 1u);
                float unit = float(uint(angle_rank)) / 4294967295.0f;
                float angle = (unit - 0.5f) * 3.14159265358979323846f * direction;
                float c = cos(angle);
                float s = sin(angle);
                float lhs = values[index];
                float rhs = values[index + 1u];
                values[index] = c * lhs - s * rhs;
                values[index + 1u] = s * lhs + c * rhs;
            }
        }

        inline void tq_apply_product_rotation(
            thread float* values,
            uint count,
            ulong seed,
            uint group_index,
            bool inverse
        ) {
            if (count <= 1u) {
                tq_apply_rotation_signs(values, count, seed, group_index);
                return;
            }
            if ((count & (count - 1u)) == 0u) {
                if (inverse) {
                    tq_fast_hadamard(values, count);
                    tq_apply_rotation_signs(values, count, seed, group_index);
                } else {
                    tq_apply_rotation_signs(values, count, seed, group_index);
                    tq_fast_hadamard(values, count);
                }
                float scale = rsqrt(float(count));
                for (uint local = 0u; local < count; local++) {
                    values[local] *= scale;
                }
                return;
            }
            if (inverse) {
                for (uint pass_index = 0u; pass_index < 4u; pass_index++) {
                    tq_apply_givens_pass(values, count, seed, group_index, 3u - pass_index, -1.0f);
                }
            } else {
                for (uint pass = 0u; pass < 4u; pass++) {
                    tq_apply_givens_pass(values, count, seed, group_index, pass, 1.0f);
                }
            }
        }

        inline uint tq_bitset_offset(
            uint batch,
            uint head,
            uint token,
            uint group,
            uint word,
            uint kv_heads,
            uint capacity,
            uint groups_per_vector,
            uint bitset_words_per_group
        ) {
            return (((batch * kv_heads + head) * capacity + token)
                * groups_per_vector + group) * bitset_words_per_group + word;
        }

        inline uint tq_packed_offset(
            uint batch,
            uint head,
            uint token,
            uint group,
            uint word,
            uint kv_heads,
            uint capacity,
            uint groups_per_vector,
            uint mag_words_per_group
        ) {
            return (((batch * kv_heads + head) * capacity + token)
                * groups_per_vector + group) * mag_words_per_group + word;
        }

        template <typename PackedPtr>
        inline uint tq_read_packed_unsigned(
            PackedPtr packed,
            uint batch,
            uint head,
            uint token,
            uint group,
            uint bit_offset,
            uint bits,
            uint kv_heads,
            uint capacity,
            uint groups_per_vector,
            uint mag_words_per_group
        ) {
            uint packed_word = bit_offset >> 5;
            uint packed_bit = bit_offset & 31u;
            uint first = packed[tq_packed_offset(
                batch, head, token, group, packed_word,
                kv_heads, capacity, groups_per_vector, mag_words_per_group)] >> packed_bit;
            if (packed_bit + bits > 32u) {
                uint next = packed[tq_packed_offset(
                    batch, head, token, group, packed_word + 1u,
                    kv_heads, capacity, groups_per_vector, mag_words_per_group)];
                first |= next << (32u - packed_bit);
            }
            return first & ((1u << bits) - 1u);
        }

        template <typename PackedPtr>
        inline uint tq_read_aligned_affine_unsigned(
            PackedPtr packed,
            uint batch,
            uint head,
            uint token,
            uint group,
            uint local,
            uint bits,
            uint kv_heads,
            uint capacity,
            uint groups_per_vector,
            uint mag_words_per_group
        ) {
            uint bit_offset = local * bits;
            uint packed_word = bit_offset >> 5;
            uint packed_bit = bit_offset & 31u;
            uint word = packed[tq_packed_offset(
                batch, head, token, group, packed_word,
                kv_heads, capacity, groups_per_vector, mag_words_per_group)];
            return (word >> packed_bit) & ((1u << bits) - 1u);
        }

        template <typename PackedPtr>
        inline void tq_write_packed_unsigned(
            PackedPtr packed,
            uint quantized,
            uint batch,
            uint head,
            uint token,
            uint group,
            uint bit_offset,
            uint bits,
            uint kv_heads,
            uint capacity,
            uint groups_per_vector,
            uint mag_words_per_group
        ) {
            uint packed_word = bit_offset >> 5;
            uint packed_bit = bit_offset & 31u;
            uint mask = ((1u << bits) - 1u);
            uint value = quantized & mask;
            packed[tq_packed_offset(
                batch, head, token, group, packed_word,
                kv_heads, capacity, groups_per_vector, mag_words_per_group)] |=
                value << packed_bit;
            if (packed_bit + bits > 32u) {
                packed[tq_packed_offset(
                    batch, head, token, group, packed_word + 1u,
                    kv_heads, capacity, groups_per_vector, mag_words_per_group)] |=
                    value >> (32u - packed_bit);
            }
        }

        inline uint tq_scale_offset(
            uint batch,
            uint head,
            uint token,
            uint group,
            uint scale_index,
            uint kv_heads,
            uint capacity,
            uint groups_per_vector
        ) {
            return ((((batch * kv_heads + head) * capacity + token)
                * groups_per_vector + group) * 2u) + scale_index;
        }

        inline uint tq_physical_token(
            uint logical_token,
            uint capacity,
            uint ring_offset,
            uint pinned_prefix_length
        ) {
            uint pinned = pinned_prefix_length;
            if (logical_token < pinned) {
                return logical_token;
            }
            uint ring_capacity = capacity - pinned;
            if (ring_capacity == 0u) {
                return min(logical_token, capacity - 1u);
            }
            uint ring_logical = logical_token - pinned;
            return pinned + ((ring_offset + ring_logical) % ring_capacity);
        }

        template <typename HighMaskPtr>
        inline uint tq_attention_high_count_before(
            HighMaskPtr high_mask,
            uint batch,
            uint head,
            uint token,
            uint group,
            uint local,
            uint kv_heads,
            uint capacity,
            uint groups_per_vector,
            uint bitset_words_per_group
        ) {
            uint full_words = local >> 5;
            uint count = 0u;
            for (uint word = 0u; word < full_words; word++) {
                count += popcount(high_mask[tq_bitset_offset(
                    batch, head, token, group, word,
                    kv_heads, capacity, groups_per_vector, bitset_words_per_group)]);
            }
            uint remainder = local & 31u;
            if (remainder > 0u && full_words < bitset_words_per_group) {
                uint mask = (1u << remainder) - 1u;
                count += popcount(high_mask[tq_bitset_offset(
                    batch, head, token, group, full_words,
                    kv_heads, capacity, groups_per_vector, bitset_words_per_group)] & mask);
            }
            return count;
        }

        template <typename HighMaskPtr>
        inline uint tq_attention_magnitude_bit_offset(
            HighMaskPtr high_mask,
            uint batch,
            uint head,
            uint token,
            uint group,
            uint local,
            uint kv_heads,
            uint capacity,
            uint groups_per_vector,
            uint bitset_words_per_group,
            uint base_bits,
            uint high_bits,
            thread uint* bits_out
        ) {
            uint bits = base_bits;
            uint bit_offset = local * base_bits;
            if (high_bits > base_bits) {
                uint bitset_word = local >> 5;
                uint bitset_bit = local & 31u;
                bool high_precision =
                    (high_mask[tq_bitset_offset(
                        batch, head, token, group, bitset_word,
                        kv_heads, capacity, groups_per_vector, bitset_words_per_group)]
                        & (1u << bitset_bit)) != 0u;
                bits = high_precision ? high_bits : base_bits;

                uint high_before = tq_attention_high_count_before(
                    high_mask, batch, head, token, group, local,
                    kv_heads, capacity, groups_per_vector, bitset_words_per_group);
                bit_offset += high_before * (high_bits - base_bits);
            }
            *bits_out = bits;
            return bit_offset;
        }

        template <typename PackedPtr, typename HighMaskPtr>
        inline uint tq_read_magnitude(
            PackedPtr packed,
            HighMaskPtr high_mask,
            uint batch,
            uint head,
            uint token,
            uint group,
            uint local,
            uint kv_heads,
            uint capacity,
            uint groups_per_vector,
            uint mag_words_per_group,
            uint bitset_words_per_group,
            uint base_bits,
            uint high_bits
        ) {
            uint bits = base_bits;
            uint bit_offset = tq_attention_magnitude_bit_offset(
                high_mask, batch, head, token, group, local,
                kv_heads, capacity, groups_per_vector, bitset_words_per_group,
                base_bits, high_bits, &bits);
            return tq_read_packed_unsigned(
                packed, batch, head, token, group, bit_offset, bits,
                kv_heads, capacity, groups_per_vector, mag_words_per_group);
        }

        inline uint tq_storage_group_index(
            uint batch,
            uint head,
            uint token,
            uint group,
            uint kv_heads,
            uint capacity,
            uint groups_per_vector
        ) {
            // TurboQuant (arXiv:2504.19874) uses a SINGLE shared random rotation, not a
            // per-token one: the data-oblivious rotation only needs to be random, not unique
            // per vector, to hit the distortion bound. Keying the rotation / high-precision
            // seed by (batch, head, group) — and NOT by token/capacity — makes the rotation
            // identical for every key token in a head, so the query can be rotated ONCE per
            // group and reused across all keys instead of being re-rotated per key (the prior
            // behaviour, which cost O(N) query rotations per attention step for no quality
            // gain). Encode and decode both route through this function, so the codec stays
            // self-consistent. `token`/`capacity` are intentionally unused.
            (void)token;
            (void)capacity;
            return (batch * kv_heads + head) * groups_per_vector + group;
        }

        template <
            typename PackedPtr,
            typename SignsPtr,
            typename HighMaskPtr,
            typename ResidualSignsPtr,
            typename ScalesPtr
        >
        inline float tq_decode_attention_value(
            PackedPtr packed,
            SignsPtr signs,
            HighMaskPtr high_mask,
            ResidualSignsPtr residual_signs,
            ScalesPtr scales,
            uint batch,
            uint head,
            uint token,
            uint dimension,
            ulong seed,
            uint role,
            uint group_size,
            uint kv_heads,
            uint capacity,
            uint groups_per_vector,
            uint mag_words_per_group,
            uint bitset_words_per_group,
            uint base_bits,
            uint high_bits,
            uint value_bits,
            uint key_base_bits,
            uint key_high_bits,
            uint layout_version,
            uint head_dim,
            uint high_count,
            thread float* rotated
        ) {
            uint group = dimension / group_size;
            uint local = dimension - group * group_size;
            if (role == 1u) {
                uint quantized = value_bits == 4u || value_bits == 8u
                    ? tq_read_aligned_affine_unsigned(
                        packed, batch, head, token, group, local, value_bits,
                        kv_heads, capacity, groups_per_vector, mag_words_per_group)
                    : tq_read_packed_unsigned(
                        packed, batch, head, token, group, local * value_bits, value_bits,
                        kv_heads, capacity, groups_per_vector, mag_words_per_group);
                uint scale_base = ((((batch * kv_heads + head) * capacity + token)
                    * groups_per_vector + group) * 2u);
                return scales[scale_base + 1u] + float(quantized) * scales[scale_base];
            }

            uint group_start = group * group_size;
            uint count = min(group_size, head_dim - group_start);
            uint storage_group = tq_storage_group_index(
                batch, head, token, group, kv_heads, capacity, groups_per_vector);
            uint bit_offset = 0u;
            uint cached_high_word = 0xffffffffu;
            uint cached_high_bits = 0u;
            bool split_magnitude =
                layout_version >= 6u
                && key_high_bits == key_base_bits + 1u
                && key_high_bits > key_base_bits;
            float inv_sqrt_count = rsqrt(float(max(count, 1u)));
            for (uint decode_local = 0u; decode_local < count; decode_local++) {
                uint bitset_word = decode_local >> 5;
                uint bitset_bit = decode_local & 31u;
                uint bit_mask = 1u << bitset_bit;
                uint bits = key_base_bits;
                uint code = 0u;
                if (split_magnitude) {
                    bool high_precision = tq_split_high_precision(decode_local, high_count);
                    bits = high_precision ? key_high_bits : key_base_bits;
                    code = tq_read_packed_unsigned(
                        packed, batch, head, token, group, decode_local * key_base_bits,
                        key_base_bits, kv_heads, capacity, groups_per_vector,
                        mag_words_per_group);
                    if (high_precision) {
                        uint extra_code = tq_read_packed_unsigned(
                            packed, batch, head, token, group,
                            group_size * key_base_bits + decode_local,
                            key_high_bits - key_base_bits,
                            kv_heads, capacity, groups_per_vector, mag_words_per_group);
                        code |= extra_code << key_base_bits;
                    }
                } else if (key_high_bits > key_base_bits) {
                    if (bitset_word != cached_high_word) {
                        cached_high_word = bitset_word;
                        cached_high_bits = high_mask[tq_bitset_offset(
                            batch, head, token, group, bitset_word,
                            kv_heads, capacity, groups_per_vector, bitset_words_per_group)];
                    }
                    bool high_precision = (cached_high_bits & bit_mask) != 0u;
                    bits = high_precision ? key_high_bits : key_base_bits;
                    code = tq_read_packed_unsigned(
                        packed, batch, head, token, group, bit_offset, bits,
                        kv_heads, capacity, groups_per_vector, mag_words_per_group);
                    bit_offset += bits;
                } else {
                    code = tq_read_packed_unsigned(
                        packed, batch, head, token, group, bit_offset, bits,
                        kv_heads, capacity, groups_per_vector, mag_words_per_group);
                    bit_offset += bits;
                }
                rotated[decode_local] = tq_codebook_unit(bits, code) * inv_sqrt_count;
            }
            tq_apply_product_rotation(rotated, count, seed, storage_group, true);
            return rotated[local] * scales[tq_scale_offset(
                batch, head, token, group, 0u, kv_heads, capacity, groups_per_vector)];
        }

        template <
            typename PackedPtr,
            typename SignsPtr,
            typename HighMaskPtr,
            typename ResidualSignsPtr,
            typename ScalesPtr
        >
        inline float tq_product_attention_inner_product_group(
            PackedPtr packed,
            SignsPtr signs,
            HighMaskPtr high_mask,
            ResidualSignsPtr residual_signs,
            ScalesPtr scales,
            thread float* query_values,
            uint batch,
            uint head,
            uint token,
            uint group,
            ulong seed,
            uint group_size,
            uint kv_heads,
            uint capacity,
            uint groups_per_vector,
            uint mag_words_per_group,
            uint bitset_words_per_group,
            uint key_base_bits,
            uint key_high_bits,
            uint layout_version,
            uint head_dim,
            uint high_count
        ) {
            uint group_start = group * group_size;
            uint count = min(group_size, head_dim - group_start);
            uint storage_group = tq_storage_group_index(
                batch, head, token, group, kv_heads, capacity, groups_per_vector);
            tq_apply_product_rotation(query_values, count, seed, storage_group, false);

            float quantized_dot = 0.0f;
            float sign_dot = 0.0f;
            // TQPROF_OPT3 packed-word caches (base-stream + extra-bit slot), mirroring the pair/quad
            // estimators so the split-magnitude decode avoids per-element packed reloads.
            uint tqopt_packed_base = tq_packed_offset(
                batch, head, token, group, 0u,
                kv_heads, capacity, groups_per_vector, mag_words_per_group);
            uint tqopt_cached_idx = 0xffffffffu;
            uint tqopt_cached_val = 0u;
            uint tqopt_extra_idx = 0xffffffffu;
            uint tqopt_extra_val = 0u;
            uint cached_bitset_word = 0xffffffffu;
            uint cached_sign_bits = 0u;
            uint cached_high_word = 0xffffffffu;
            uint cached_high_bits = 0u;
            uint bit_offset = 0u;
            bool split_magnitude =
                layout_version >= 6u
                && key_high_bits == key_base_bits + 1u
                && key_high_bits > key_base_bits;
            float inv_sqrt_count = rsqrt(float(max(count, 1u)));
            for (uint local = 0u; local < count; local++) {
                uint bitset_word = local >> 5;
                uint bitset_bit = local & 31u;
                uint bit_mask = 1u << bitset_bit;
                if (bitset_word != cached_bitset_word) {
                    cached_bitset_word = bitset_word;
                    cached_sign_bits = signs[tq_bitset_offset(
                        batch, head, token, group, bitset_word,
                        kv_heads, capacity, groups_per_vector, bitset_words_per_group)];
                }
                uint bits = key_base_bits;
                uint code = 0u;
                if (split_magnitude) {
                    bool high_precision = tq_split_high_precision(local, high_count);
                    bits = high_precision ? key_high_bits : key_base_bits;
                    // TQPROF_OPT3 split-magnitude fast path: cache the base-bits stream (offset
                    // local*key_base_bits, uniform stride) and the high-precision extra-bit stream
                    // (offset group_size*key_base_bits+local, stride 1) in two slots, instead of the
                    // two per-element tq_read_packed_unsigned reloads. Bit-exact. This is the live
                    // turbo3_5 path (verified by negation litmus).
                    uint base_bo = local * key_base_bits;
                    uint base_pw = base_bo >> 5;
                    uint base_pbit = base_bo & 31u;
                    if (base_pw != tqopt_cached_idx) {
                        tqopt_cached_idx = base_pw;
                        tqopt_cached_val = packed[tqopt_packed_base + base_pw];
                    }
                    uint base_asm = tqopt_cached_val >> base_pbit;
                    if (base_pbit + key_base_bits > 32u) {
                        base_asm |= packed[tqopt_packed_base + base_pw + 1u] << (32u - base_pbit);
                    }
                    code = base_asm & ((1u << key_base_bits) - 1u);
                    if (high_precision) {
                        uint extra_bits = key_high_bits - key_base_bits;
                        uint extra_bo = group_size * key_base_bits + local;
                        uint extra_pw = extra_bo >> 5;
                        uint extra_pbit = extra_bo & 31u;
                        if (extra_pw != tqopt_extra_idx) {
                            tqopt_extra_idx = extra_pw;
                            tqopt_extra_val = packed[tqopt_packed_base + extra_pw];
                        }
                        uint extra_asm = tqopt_extra_val >> extra_pbit;
                        if (extra_pbit + extra_bits > 32u) {
                            extra_asm |= packed[tqopt_packed_base + extra_pw + 1u] << (32u - extra_pbit);
                        }
                        uint extra_code = extra_asm & ((1u << extra_bits) - 1u);
                        code |= extra_code << key_base_bits;
                    }
                } else if (key_high_bits > key_base_bits) {
                    if (bitset_word != cached_high_word) {
                        cached_high_word = bitset_word;
                        cached_high_bits = high_mask[tq_bitset_offset(
                            batch, head, token, group, bitset_word,
                            kv_heads, capacity, groups_per_vector, bitset_words_per_group)];
                    }
                    bool high_precision = (cached_high_bits & bit_mask) != 0u;
                    bits = high_precision ? key_high_bits : key_base_bits;
                    code = tq_read_packed_unsigned(
                        packed, batch, head, token, group, bit_offset, bits,
                        kv_heads, capacity, groups_per_vector, mag_words_per_group);
                    bit_offset += bits;
                } else {
                    code = tq_read_packed_unsigned(
                        packed, batch, head, token, group, bit_offset, bits,
                        kv_heads, capacity, groups_per_vector, mag_words_per_group);
                    bit_offset += bits;
                }
                quantized_dot += query_values[local] * tq_codebook_unit(bits, code) * inv_sqrt_count;
                float qjl_sign = (cached_sign_bits & bit_mask) != 0u ? -1.0f : 1.0f;
                sign_dot += qjl_sign * query_values[local];
            }

            float norm = scales[tq_scale_offset(
                batch, head, token, group, 0u, kv_heads, capacity, groups_per_vector)];
            float residual_norm = scales[tq_scale_offset(
                batch, head, token, group, 1u, kv_heads, capacity, groups_per_vector)];
            float residual = residual_norm * sqrt(3.14159265358979323846f / (2.0f * float(count))) * sign_dot;
            return norm * quantized_dot + residual;
        }

        template <
            typename PackedPtr,
            typename SignsPtr,
            typename HighMaskPtr,
            typename ResidualSignsPtr,
            typename ScalesPtr
        >
        inline void tq_product_attention_inner_product_group_pair(
            PackedPtr packed,
            SignsPtr signs,
            HighMaskPtr high_mask,
            ResidualSignsPtr residual_signs,
            ScalesPtr scales,
            thread float* query_values,
            thread float* scores,
            uint pair_repeats,
            uint batch,
            uint head,
            uint token,
            uint group,
            ulong seed,
            uint group_size,
            uint kv_heads,
            uint capacity,
            uint groups_per_vector,
            uint mag_words_per_group,
            uint bitset_words_per_group,
            uint key_base_bits,
            uint key_high_bits,
            uint layout_version,
            uint head_dim,
            uint high_count,
            bool query_prerotated
        ) {
            uint group_start = group * group_size;
            uint count = min(group_size, head_dim - group_start);
            uint storage_group = tq_storage_group_index(
                batch, head, token, group, kv_heads, capacity, groups_per_vector);
            uint repeats = min(pair_repeats, 2u);

            if (!query_prerotated) {
                for (uint repeat = 0u; repeat < repeats; repeat++) {
                    tq_apply_product_rotation(
                        query_values + repeat * group_size, count, seed, storage_group, false);
                }
            }

            float quantized_dot[2];
            float sign_dot[2];
            quantized_dot[0] = 0.0f;
            quantized_dot[1] = 0.0f;
            sign_dot[0] = 0.0f;
            sign_dot[1] = 0.0f;
            // TQPROF_OPT/OPT2/OPT3 packed-word caches (hoisted base + base-stream slot + extra-bit
            // slot), mirroring the quad estimator so all three decode branches avoid per-element
            // packed reloads.
            uint tqopt_packed_base = tq_packed_offset(
                batch, head, token, group, 0u,
                kv_heads, capacity, groups_per_vector, mag_words_per_group);
            uint tqopt_cached_idx = 0xffffffffu;
            uint tqopt_cached_val = 0u;
            uint tqopt_extra_idx = 0xffffffffu;
            uint tqopt_extra_val = 0u;
            uint cached_bitset_word = 0xffffffffu;
            uint cached_sign_bits = 0u;
            uint cached_high_word = 0xffffffffu;
            uint cached_high_bits = 0u;
            uint bit_offset = 0u;
            bool split_magnitude =
                layout_version >= 6u
                && key_high_bits == key_base_bits + 1u
                && key_high_bits > key_base_bits;
            float inv_sqrt_count = rsqrt(float(max(count, 1u)));

            for (uint local = 0u; local < count; local++) {
                uint bitset_word = local >> 5;
                uint bitset_bit = local & 31u;
                uint bit_mask = 1u << bitset_bit;
                if (bitset_word != cached_bitset_word) {
                    cached_bitset_word = bitset_word;
                    cached_sign_bits = signs[tq_bitset_offset(
                        batch, head, token, group, bitset_word,
                        kv_heads, capacity, groups_per_vector, bitset_words_per_group)];
                }
                uint bits = key_base_bits;
                uint code = 0u;
                if (split_magnitude) {
                    bool high_precision = tq_split_high_precision(local, high_count);
                    bits = high_precision ? key_high_bits : key_base_bits;
                    // TQPROF_OPT3 split-magnitude fast path: cache the base-bits stream (offset
                    // local*key_base_bits, uniform stride) and the high-precision extra-bit stream
                    // (offset group_size*key_base_bits+local, stride 1) in two slots, instead of the
                    // two per-element tq_read_packed_unsigned reloads. Bit-exact. This is the live
                    // turbo3_5 path (verified by negation litmus).
                    uint base_bo = local * key_base_bits;
                    uint base_pw = base_bo >> 5;
                    uint base_pbit = base_bo & 31u;
                    if (base_pw != tqopt_cached_idx) {
                        tqopt_cached_idx = base_pw;
                        tqopt_cached_val = packed[tqopt_packed_base + base_pw];
                    }
                    uint base_asm = tqopt_cached_val >> base_pbit;
                    if (base_pbit + key_base_bits > 32u) {
                        base_asm |= packed[tqopt_packed_base + base_pw + 1u] << (32u - base_pbit);
                    }
                    code = base_asm & ((1u << key_base_bits) - 1u);
                    if (high_precision) {
                        uint extra_bits = key_high_bits - key_base_bits;
                        uint extra_bo = group_size * key_base_bits + local;
                        uint extra_pw = extra_bo >> 5;
                        uint extra_pbit = extra_bo & 31u;
                        if (extra_pw != tqopt_extra_idx) {
                            tqopt_extra_idx = extra_pw;
                            tqopt_extra_val = packed[tqopt_packed_base + extra_pw];
                        }
                        uint extra_asm = tqopt_extra_val >> extra_pbit;
                        if (extra_pbit + extra_bits > 32u) {
                            extra_asm |= packed[tqopt_packed_base + extra_pw + 1u] << (32u - extra_pbit);
                        }
                        uint extra_code = extra_asm & ((1u << extra_bits) - 1u);
                        code |= extra_code << key_base_bits;
                    }
                } else if (key_high_bits > key_base_bits) {
                    if (bitset_word != cached_high_word) {
                        cached_high_word = bitset_word;
                        cached_high_bits = high_mask[tq_bitset_offset(
                            batch, head, token, group, bitset_word,
                            kv_heads, capacity, groups_per_vector, bitset_words_per_group)];
                    }
                    bool high_precision = (cached_high_bits & bit_mask) != 0u;
                    bits = high_precision ? key_high_bits : key_base_bits;
                    code = tq_read_packed_unsigned(
                        packed, batch, head, token, group, bit_offset, bits,
                        kv_heads, capacity, groups_per_vector, mag_words_per_group);
                    bit_offset += bits;
                } else {
                    code = tq_read_packed_unsigned(
                        packed, batch, head, token, group, bit_offset, bits,
                        kv_heads, capacity, groups_per_vector, mag_words_per_group);
                    bit_offset += bits;
                }
                float level = tq_codebook_unit(bits, code) * inv_sqrt_count;
                float qjl_sign = (cached_sign_bits & bit_mask) != 0u ? -1.0f : 1.0f;
                for (uint repeat = 0u; repeat < repeats; repeat++) {
                    float query_value = query_values[repeat * group_size + local];
                    quantized_dot[repeat] += query_value * level;
                    sign_dot[repeat] += qjl_sign * query_value;
                }
            }

            float norm = scales[tq_scale_offset(
                batch, head, token, group, 0u, kv_heads, capacity, groups_per_vector)];
            float residual_norm = scales[tq_scale_offset(
                batch, head, token, group, 1u, kv_heads, capacity, groups_per_vector)];
            float residual_scale = residual_norm * sqrt(3.14159265358979323846f / (2.0f * float(count)));
            for (uint repeat = 0u; repeat < repeats; repeat++) {
                scores[repeat] += norm * quantized_dot[repeat] + residual_scale * sign_dot[repeat];
            }
        }

        template <
            typename PackedPtr,
            typename SignsPtr,
            typename HighMaskPtr,
            typename ResidualSignsPtr,
            typename ScalesPtr
        >
        inline void tq_product_attention_inner_product_group_quad(
            PackedPtr packed,
            SignsPtr signs,
            HighMaskPtr high_mask,
            ResidualSignsPtr residual_signs,
            ScalesPtr scales,
            thread float* query_values,
            thread float* scores,
            uint batch,
            uint head,
            uint token,
            uint group,
            ulong seed,
            uint group_size,
            uint kv_heads,
            uint capacity,
            uint groups_per_vector,
            uint mag_words_per_group,
            uint bitset_words_per_group,
            uint key_base_bits,
            uint key_high_bits,
            uint layout_version,
            uint head_dim,
            uint high_count,
            bool query_prerotated
        ) {
            uint group_start = group * group_size;
            uint count = min(group_size, head_dim - group_start);
            uint storage_group = tq_storage_group_index(
                batch, head, token, group, kv_heads, capacity, groups_per_vector);

            if (!query_prerotated) {
                for (uint repeat = 0u; repeat < 4u; repeat++) {
                    tq_apply_product_rotation(
                        query_values + repeat * group_size, count, seed, storage_group, false);
                }
            }

            float quantized_dot[4];
            float sign_dot[4];
            for (uint repeat = 0u; repeat < 4u; repeat++) {
                quantized_dot[repeat] = 0.0f;
                sign_dot[repeat] = 0.0f;
            }
            // TQPROF_OPT hoist invariant packed base + cache packed word across uniform codes
            uint tqopt_packed_base = tq_packed_offset(
                batch, head, token, group, 0u,
                kv_heads, capacity, groups_per_vector, mag_words_per_group);
            uint tqopt_cached_idx = 0xffffffffu;
            uint tqopt_cached_val = 0u;
            // Second cache slot for the split-magnitude high-precision (extra-bit) stream, which
            // lives in a separate region of the packed buffer from the base-bits stream.
            uint tqopt_extra_idx = 0xffffffffu;
            uint tqopt_extra_val = 0u;
            uint cached_bitset_word = 0xffffffffu;
            uint cached_sign_bits = 0u;
            uint cached_high_word = 0xffffffffu;
            uint cached_high_bits = 0u;
            uint bit_offset = 0u;
            bool split_magnitude =
                layout_version >= 6u
                && key_high_bits == key_base_bits + 1u
                && key_high_bits > key_base_bits;
            float inv_sqrt_count = rsqrt(float(max(count, 1u)));

            for (uint local = 0u; local < count; local++) {
                uint bitset_word = local >> 5;
                uint bitset_bit = local & 31u;
                uint bit_mask = 1u << bitset_bit;
                if (bitset_word != cached_bitset_word) {
                    cached_bitset_word = bitset_word;
                    cached_sign_bits = signs[tq_bitset_offset(
                        batch, head, token, group, bitset_word,
                        kv_heads, capacity, groups_per_vector, bitset_words_per_group)];
                }
                uint bits = key_base_bits;
                uint code = 0u;
                if (split_magnitude) {
                    bool high_precision = tq_split_high_precision(local, high_count);
                    bits = high_precision ? key_high_bits : key_base_bits;
                    // TQPROF_OPT3 split-magnitude fast path: cache the base-bits stream (offset
                    // local*key_base_bits, uniform stride) and the high-precision extra-bit stream
                    // (offset group_size*key_base_bits+local, stride 1) in two slots, instead of the
                    // two per-element tq_read_packed_unsigned reloads. Bit-exact. This is the live
                    // turbo3_5 path (verified by negation litmus).
                    uint base_bo = local * key_base_bits;
                    uint base_pw = base_bo >> 5;
                    uint base_pbit = base_bo & 31u;
                    if (base_pw != tqopt_cached_idx) {
                        tqopt_cached_idx = base_pw;
                        tqopt_cached_val = packed[tqopt_packed_base + base_pw];
                    }
                    uint base_asm = tqopt_cached_val >> base_pbit;
                    if (base_pbit + key_base_bits > 32u) {
                        base_asm |= packed[tqopt_packed_base + base_pw + 1u] << (32u - base_pbit);
                    }
                    code = base_asm & ((1u << key_base_bits) - 1u);
                    if (high_precision) {
                        uint extra_bits = key_high_bits - key_base_bits;
                        uint extra_bo = group_size * key_base_bits + local;
                        uint extra_pw = extra_bo >> 5;
                        uint extra_pbit = extra_bo & 31u;
                        if (extra_pw != tqopt_extra_idx) {
                            tqopt_extra_idx = extra_pw;
                            tqopt_extra_val = packed[tqopt_packed_base + extra_pw];
                        }
                        uint extra_asm = tqopt_extra_val >> extra_pbit;
                        if (extra_pbit + extra_bits > 32u) {
                            extra_asm |= packed[tqopt_packed_base + extra_pw + 1u] << (32u - extra_pbit);
                        }
                        uint extra_code = extra_asm & ((1u << extra_bits) - 1u);
                        code |= extra_code << key_base_bits;
                    }
                } else if (key_high_bits > key_base_bits) {
                    if (bitset_word != cached_high_word) {
                        cached_high_word = bitset_word;
                        cached_high_bits = high_mask[tq_bitset_offset(
                            batch, head, token, group, bitset_word,
                            kv_heads, capacity, groups_per_vector, bitset_words_per_group)];
                    }
                    bool high_precision = (cached_high_bits & bit_mask) != 0u;
                    bits = high_precision ? key_high_bits : key_base_bits;
                    // TQPROF_OPT2 variable-bit fast path: reuse the same packed-word cache as the
                    // uniform branch. Valid because bit_offset advances monotonically over a single
                    // contiguous magnitude bitstream even though `bits` varies per element, so
                    // consecutive variable-width codes still share 32-bit words. Removes the
                    // per-element packed reload that tq_read_packed_unsigned did (turbo3_5 path).
                    uint tqopt_pw = bit_offset >> 5;
                    uint tqopt_pbit = bit_offset & 31u;
                    if (tqopt_pw != tqopt_cached_idx) {
                        tqopt_cached_idx = tqopt_pw;
                        tqopt_cached_val = packed[tqopt_packed_base + tqopt_pw];
                    }
                    uint tqopt_asm = tqopt_cached_val >> tqopt_pbit;
                    if (tqopt_pbit + bits > 32u) {
                        tqopt_asm |= packed[tqopt_packed_base + tqopt_pw + 1u] << (32u - tqopt_pbit);
                    }
                    code = tqopt_asm & ((1u << bits) - 1u);
                    bit_offset += bits;
                } else {
                    // TQPROF_OPT cached uniform-width packed read (1 load per 32/bits codes)
                    uint tqopt_pw = bit_offset >> 5;
                    uint tqopt_pbit = bit_offset & 31u;
                    if (tqopt_pw != tqopt_cached_idx) {
                        tqopt_cached_idx = tqopt_pw;
                        tqopt_cached_val = packed[tqopt_packed_base + tqopt_pw];
                    }
                    uint tqopt_asm = tqopt_cached_val >> tqopt_pbit;
                    if (tqopt_pbit + bits > 32u) {
                        tqopt_asm |= packed[tqopt_packed_base + tqopt_pw + 1u] << (32u - tqopt_pbit);
                    }
                    code = tqopt_asm & ((1u << bits) - 1u);
                    bit_offset += bits;
                }
                float level = tq_codebook_unit(bits, code) * inv_sqrt_count;
                float qjl_sign = (cached_sign_bits & bit_mask) != 0u ? -1.0f : 1.0f;
                for (uint repeat = 0u; repeat < 4u; repeat++) {
                    float query_value = query_values[repeat * group_size + local];
                    quantized_dot[repeat] += query_value * level;
                    sign_dot[repeat] += qjl_sign * query_value;
                }
            }

            float norm = scales[tq_scale_offset(
                batch, head, token, group, 0u, kv_heads, capacity, groups_per_vector)];
            float residual_norm = scales[tq_scale_offset(
                batch, head, token, group, 1u, kv_heads, capacity, groups_per_vector)];
            float residual_scale = residual_norm * sqrt(3.14159265358979323846f / (2.0f * float(count)));
            for (uint repeat = 0u; repeat < 4u; repeat++) {
                scores[repeat] += norm * quantized_dot[repeat] + residual_scale * sign_dot[repeat];
            }
        }
        """

    private static let attentionHeaderV7Extension = """
        inline uint tq_packed_offset_v7(
            uint batch,
            uint head,
            uint token,
            uint group,
            uint word,
            uint kv_heads,
            uint capacity,
            uint groups_per_vector,
            uint mag_words_per_group
        ) {
            uint words_per_token = groups_per_vector * mag_words_per_group;
            uint plane_base = (batch * kv_heads + head) * capacity * words_per_token;
            uint tile_base = plane_base + (token & ~31u) * words_per_token;
            return tile_base + ((group * mag_words_per_group + word) << 5) + (token & 31u);
        }

        inline uint tq_bitset_offset_v7(
            uint batch,
            uint head,
            uint token,
            uint group,
            uint word,
            uint kv_heads,
            uint capacity,
            uint groups_per_vector,
            uint bitset_words_per_group
        ) {
            uint words_per_token = groups_per_vector * bitset_words_per_group;
            uint plane_base = (batch * kv_heads + head) * capacity * words_per_token;
            uint tile_base = plane_base + (token & ~31u) * words_per_token;
            return tile_base + ((group * bitset_words_per_group + word) << 5) + (token & 31u);
        }

        inline uint tq_scale_offset_v7(
            uint batch,
            uint head,
            uint token,
            uint group,
            uint scale_index,
            uint kv_heads,
            uint capacity,
            uint groups_per_vector
        ) {
            uint scales_per_token = groups_per_vector * 2u;
            uint plane_base = (batch * kv_heads + head) * capacity * scales_per_token;
            uint tile_base = plane_base + (token & ~31u) * scales_per_token;
            return tile_base + ((group * 2u + scale_index) << 5) + (token & 31u);
        }

        template <typename PackedPtr>
        inline uint tq_read_packed_unsigned_v7(
            PackedPtr packed,
            uint batch,
            uint head,
            uint token,
            uint group,
            uint bit_offset,
            uint bits,
            uint kv_heads,
            uint capacity,
            uint groups_per_vector,
            uint mag_words_per_group
        ) {
            uint packed_word = bit_offset >> 5;
            uint packed_bit = bit_offset & 31u;
            uint first = packed[tq_packed_offset_v7(
                batch, head, token, group, packed_word,
                kv_heads, capacity, groups_per_vector, mag_words_per_group)] >> packed_bit;
            if (packed_bit + bits > 32u) {
                uint next = packed[tq_packed_offset_v7(
                    batch, head, token, group, packed_word + 1u,
                    kv_heads, capacity, groups_per_vector, mag_words_per_group)];
                first |= next << (32u - packed_bit);
            }
            return first & ((1u << bits) - 1u);
        }

        template <typename PackedPtr>
        inline void tq_write_packed_unsigned_v7(
            PackedPtr packed,
            uint quantized,
            uint batch,
            uint head,
            uint token,
            uint group,
            uint bit_offset,
            uint bits,
            uint kv_heads,
            uint capacity,
            uint groups_per_vector,
            uint mag_words_per_group
        ) {
            uint packed_word = bit_offset >> 5;
            uint packed_bit = bit_offset & 31u;
            uint mask = ((1u << bits) - 1u);
            uint value = quantized & mask;
            packed[tq_packed_offset_v7(
                batch, head, token, group, packed_word,
                kv_heads, capacity, groups_per_vector, mag_words_per_group)] |=
                value << packed_bit;
            if (packed_bit + bits > 32u) {
                packed[tq_packed_offset_v7(
                    batch, head, token, group, packed_word + 1u,
                    kv_heads, capacity, groups_per_vector, mag_words_per_group)] |=
                    value >> (32u - packed_bit);
            }
        }

        template <
            typename PackedPtr,
            typename SignsPtr,
            typename HighMaskPtr,
            typename ResidualSignsPtr,
            typename ScalesPtr
        >
        inline void tq_product_attention_inner_product_group_pair_v7(
            PackedPtr packed,
            SignsPtr signs,
            HighMaskPtr high_mask,
            ResidualSignsPtr residual_signs,
            ScalesPtr scales,
            thread float* query_values,
            thread float* scores,
            uint pair_repeats,
            uint batch,
            uint head,
            uint token,
            uint group,
            ulong seed,
            uint group_size,
            uint kv_heads,
            uint capacity,
            uint groups_per_vector,
            uint mag_words_per_group,
            uint bitset_words_per_group,
            uint key_base_bits,
            uint key_high_bits,
            uint layout_version,
            uint head_dim,
            uint high_count,
            bool query_prerotated
        ) {
            uint group_start = group * group_size;
            uint count = min(group_size, head_dim - group_start);
            uint storage_group = tq_storage_group_index(
                batch, head, token, group, kv_heads, capacity, groups_per_vector);
            uint repeats = min(pair_repeats, 2u);

            if (!query_prerotated) {
                for (uint repeat = 0u; repeat < repeats; repeat++) {
                    tq_apply_product_rotation(
                        query_values + repeat * group_size, count, seed, storage_group, false);
                }
            }

            float quantized_dot[2];
            float sign_dot[2];
            quantized_dot[0] = 0.0f;
            quantized_dot[1] = 0.0f;
            sign_dot[0] = 0.0f;
            sign_dot[1] = 0.0f;
            // TQPROF_OPT/OPT2/OPT3 packed-word caches (hoisted base + base-stream slot + extra-bit
            // slot), mirroring the quad estimator so all three decode branches avoid per-element
            // packed reloads.
            uint tqopt_packed_base = tq_packed_offset_v7(
                batch, head, token, group, 0u,
                kv_heads, capacity, groups_per_vector, mag_words_per_group);
            uint tqopt_cached_idx = 0xffffffffu;
            uint tqopt_cached_val = 0u;
            uint tqopt_extra_idx = 0xffffffffu;
            uint tqopt_extra_val = 0u;
            uint cached_bitset_word = 0xffffffffu;
            uint cached_sign_bits = 0u;
            uint cached_high_word = 0xffffffffu;
            uint cached_high_bits = 0u;
            uint bit_offset = 0u;
            bool split_magnitude =
                layout_version >= 6u
                && key_high_bits == key_base_bits + 1u
                && key_high_bits > key_base_bits;
            float inv_sqrt_count = rsqrt(float(max(count, 1u)));

            for (uint local = 0u; local < count; local++) {
                uint bitset_word = local >> 5;
                uint bitset_bit = local & 31u;
                uint bit_mask = 1u << bitset_bit;
                if (bitset_word != cached_bitset_word) {
                    cached_bitset_word = bitset_word;
                    cached_sign_bits = signs[tq_bitset_offset_v7(
                        batch, head, token, group, bitset_word,
                        kv_heads, capacity, groups_per_vector, bitset_words_per_group)];
                }
                uint bits = key_base_bits;
                uint code = 0u;
                if (split_magnitude) {
                    bool high_precision = tq_split_high_precision(local, high_count);
                    bits = high_precision ? key_high_bits : key_base_bits;
                    // TQPROF_OPT3 split-magnitude fast path: cache the base-bits stream (offset
                    // local*key_base_bits, uniform stride) and the high-precision extra-bit stream
                    // (offset group_size*key_base_bits+local, stride 1) in two slots, instead of the
                    // two per-element tq_read_packed_unsigned reloads. Bit-exact. This is the live
                    // turbo3_5 path (verified by negation litmus).
                    uint base_bo = local * key_base_bits;
                    uint base_pw = base_bo >> 5;
                    uint base_pbit = base_bo & 31u;
                    if (base_pw != tqopt_cached_idx) {
                        tqopt_cached_idx = base_pw;
                        tqopt_cached_val = packed[tqopt_packed_base + (base_pw << 5)];
                    }
                    uint base_asm = tqopt_cached_val >> base_pbit;
                    if (base_pbit + key_base_bits > 32u) {
                        base_asm |= packed[tqopt_packed_base + ((base_pw + 1u) << 5)] << (32u - base_pbit);
                    }
                    code = base_asm & ((1u << key_base_bits) - 1u);
                    if (high_precision) {
                        uint extra_bits = key_high_bits - key_base_bits;
                        uint extra_bo = group_size * key_base_bits + local;
                        uint extra_pw = extra_bo >> 5;
                        uint extra_pbit = extra_bo & 31u;
                        if (extra_pw != tqopt_extra_idx) {
                            tqopt_extra_idx = extra_pw;
                            tqopt_extra_val = packed[tqopt_packed_base + (extra_pw << 5)];
                        }
                        uint extra_asm = tqopt_extra_val >> extra_pbit;
                        if (extra_pbit + extra_bits > 32u) {
                            extra_asm |= packed[tqopt_packed_base + ((extra_pw + 1u) << 5)] << (32u - extra_pbit);
                        }
                        uint extra_code = extra_asm & ((1u << extra_bits) - 1u);
                        code |= extra_code << key_base_bits;
                    }
                } else if (key_high_bits > key_base_bits) {
                    if (bitset_word != cached_high_word) {
                        cached_high_word = bitset_word;
                        cached_high_bits = high_mask[tq_bitset_offset_v7(
                            batch, head, token, group, bitset_word,
                            kv_heads, capacity, groups_per_vector, bitset_words_per_group)];
                    }
                    bool high_precision = (cached_high_bits & bit_mask) != 0u;
                    bits = high_precision ? key_high_bits : key_base_bits;
                    code = tq_read_packed_unsigned_v7(
                        packed, batch, head, token, group, bit_offset, bits,
                        kv_heads, capacity, groups_per_vector, mag_words_per_group);
                    bit_offset += bits;
                } else {
                    code = tq_read_packed_unsigned_v7(
                        packed, batch, head, token, group, bit_offset, bits,
                        kv_heads, capacity, groups_per_vector, mag_words_per_group);
                    bit_offset += bits;
                }
                float level = tq_codebook_unit(bits, code) * inv_sqrt_count;
                float qjl_sign = (cached_sign_bits & bit_mask) != 0u ? -1.0f : 1.0f;
                for (uint repeat = 0u; repeat < repeats; repeat++) {
                    float query_value = query_values[repeat * group_size + local];
                    quantized_dot[repeat] += query_value * level;
                    sign_dot[repeat] += qjl_sign * query_value;
                }
            }

            float norm = scales[tq_scale_offset_v7(
                batch, head, token, group, 0u, kv_heads, capacity, groups_per_vector)];
            float residual_norm = scales[tq_scale_offset_v7(
                batch, head, token, group, 1u, kv_heads, capacity, groups_per_vector)];
            float residual_scale = residual_norm * sqrt(3.14159265358979323846f / (2.0f * float(count)));
            for (uint repeat = 0u; repeat < repeats; repeat++) {
                scores[repeat] += norm * quantized_dot[repeat] + residual_scale * sign_dot[repeat];
            }
        }

        template <
            typename PackedPtr,
            typename SignsPtr,
            typename HighMaskPtr,
            typename ResidualSignsPtr,
            typename ScalesPtr
        >
        inline void tq_product_attention_inner_product_group_quad_v7(
            PackedPtr packed,
            SignsPtr signs,
            HighMaskPtr high_mask,
            ResidualSignsPtr residual_signs,
            ScalesPtr scales,
            thread float* query_values,
            thread float* scores,
            uint batch,
            uint head,
            uint token,
            uint group,
            ulong seed,
            uint group_size,
            uint kv_heads,
            uint capacity,
            uint groups_per_vector,
            uint mag_words_per_group,
            uint bitset_words_per_group,
            uint key_base_bits,
            uint key_high_bits,
            uint layout_version,
            uint head_dim,
            uint high_count,
            bool query_prerotated
        ) {
            uint group_start = group * group_size;
            uint count = min(group_size, head_dim - group_start);
            uint storage_group = tq_storage_group_index(
                batch, head, token, group, kv_heads, capacity, groups_per_vector);

            if (!query_prerotated) {
                for (uint repeat = 0u; repeat < 4u; repeat++) {
                    tq_apply_product_rotation(
                        query_values + repeat * group_size, count, seed, storage_group, false);
                }
            }

            float quantized_dot[4];
            float sign_dot[4];
            for (uint repeat = 0u; repeat < 4u; repeat++) {
                quantized_dot[repeat] = 0.0f;
                sign_dot[repeat] = 0.0f;
            }
            // TQPROF_OPT hoist invariant packed base + cache packed word across uniform codes
            uint tqopt_packed_base = tq_packed_offset_v7(
                batch, head, token, group, 0u,
                kv_heads, capacity, groups_per_vector, mag_words_per_group);
            uint tqopt_cached_idx = 0xffffffffu;
            uint tqopt_cached_val = 0u;
            // Second cache slot for the split-magnitude high-precision (extra-bit) stream, which
            // lives in a separate region of the packed buffer from the base-bits stream.
            uint tqopt_extra_idx = 0xffffffffu;
            uint tqopt_extra_val = 0u;
            uint cached_bitset_word = 0xffffffffu;
            uint cached_sign_bits = 0u;
            uint cached_high_word = 0xffffffffu;
            uint cached_high_bits = 0u;
            uint bit_offset = 0u;
            bool split_magnitude =
                layout_version >= 6u
                && key_high_bits == key_base_bits + 1u
                && key_high_bits > key_base_bits;
            float inv_sqrt_count = rsqrt(float(max(count, 1u)));

            for (uint local = 0u; local < count; local++) {
                uint bitset_word = local >> 5;
                uint bitset_bit = local & 31u;
                uint bit_mask = 1u << bitset_bit;
                if (bitset_word != cached_bitset_word) {
                    cached_bitset_word = bitset_word;
                    cached_sign_bits = signs[tq_bitset_offset_v7(
                        batch, head, token, group, bitset_word,
                        kv_heads, capacity, groups_per_vector, bitset_words_per_group)];
                }
                uint bits = key_base_bits;
                uint code = 0u;
                if (split_magnitude) {
                    bool high_precision = tq_split_high_precision(local, high_count);
                    bits = high_precision ? key_high_bits : key_base_bits;
                    // TQPROF_OPT3 split-magnitude fast path: cache the base-bits stream (offset
                    // local*key_base_bits, uniform stride) and the high-precision extra-bit stream
                    // (offset group_size*key_base_bits+local, stride 1) in two slots, instead of the
                    // two per-element tq_read_packed_unsigned reloads. Bit-exact. This is the live
                    // turbo3_5 path (verified by negation litmus).
                    uint base_bo = local * key_base_bits;
                    uint base_pw = base_bo >> 5;
                    uint base_pbit = base_bo & 31u;
                    if (base_pw != tqopt_cached_idx) {
                        tqopt_cached_idx = base_pw;
                        tqopt_cached_val = packed[tqopt_packed_base + (base_pw << 5)];
                    }
                    uint base_asm = tqopt_cached_val >> base_pbit;
                    if (base_pbit + key_base_bits > 32u) {
                        base_asm |= packed[tqopt_packed_base + ((base_pw + 1u) << 5)] << (32u - base_pbit);
                    }
                    code = base_asm & ((1u << key_base_bits) - 1u);
                    if (high_precision) {
                        uint extra_bits = key_high_bits - key_base_bits;
                        uint extra_bo = group_size * key_base_bits + local;
                        uint extra_pw = extra_bo >> 5;
                        uint extra_pbit = extra_bo & 31u;
                        if (extra_pw != tqopt_extra_idx) {
                            tqopt_extra_idx = extra_pw;
                            tqopt_extra_val = packed[tqopt_packed_base + (extra_pw << 5)];
                        }
                        uint extra_asm = tqopt_extra_val >> extra_pbit;
                        if (extra_pbit + extra_bits > 32u) {
                            extra_asm |= packed[tqopt_packed_base + ((extra_pw + 1u) << 5)] << (32u - extra_pbit);
                        }
                        uint extra_code = extra_asm & ((1u << extra_bits) - 1u);
                        code |= extra_code << key_base_bits;
                    }
                } else if (key_high_bits > key_base_bits) {
                    if (bitset_word != cached_high_word) {
                        cached_high_word = bitset_word;
                        cached_high_bits = high_mask[tq_bitset_offset_v7(
                            batch, head, token, group, bitset_word,
                            kv_heads, capacity, groups_per_vector, bitset_words_per_group)];
                    }
                    bool high_precision = (cached_high_bits & bit_mask) != 0u;
                    bits = high_precision ? key_high_bits : key_base_bits;
                    // TQPROF_OPT2 variable-bit fast path: reuse the same packed-word cache as the
                    // uniform branch. Valid because bit_offset advances monotonically over a single
                    // contiguous magnitude bitstream even though `bits` varies per element, so
                    // consecutive variable-width codes still share 32-bit words. Removes the
                    // per-element packed reload that tq_read_packed_unsigned did (turbo3_5 path).
                    uint tqopt_pw = bit_offset >> 5;
                    uint tqopt_pbit = bit_offset & 31u;
                    if (tqopt_pw != tqopt_cached_idx) {
                        tqopt_cached_idx = tqopt_pw;
                        tqopt_cached_val = packed[tqopt_packed_base + (tqopt_pw << 5)];
                    }
                    uint tqopt_asm = tqopt_cached_val >> tqopt_pbit;
                    if (tqopt_pbit + bits > 32u) {
                        tqopt_asm |= packed[tqopt_packed_base + ((tqopt_pw + 1u) << 5)] << (32u - tqopt_pbit);
                    }
                    code = tqopt_asm & ((1u << bits) - 1u);
                    bit_offset += bits;
                } else {
                    // TQPROF_OPT cached uniform-width packed read (1 load per 32/bits codes)
                    uint tqopt_pw = bit_offset >> 5;
                    uint tqopt_pbit = bit_offset & 31u;
                    if (tqopt_pw != tqopt_cached_idx) {
                        tqopt_cached_idx = tqopt_pw;
                        tqopt_cached_val = packed[tqopt_packed_base + (tqopt_pw << 5)];
                    }
                    uint tqopt_asm = tqopt_cached_val >> tqopt_pbit;
                    if (tqopt_pbit + bits > 32u) {
                        tqopt_asm |= packed[tqopt_packed_base + ((tqopt_pw + 1u) << 5)] << (32u - tqopt_pbit);
                    }
                    code = tqopt_asm & ((1u << bits) - 1u);
                    bit_offset += bits;
                }
                float level = tq_codebook_unit(bits, code) * inv_sqrt_count;
                float qjl_sign = (cached_sign_bits & bit_mask) != 0u ? -1.0f : 1.0f;
                for (uint repeat = 0u; repeat < 4u; repeat++) {
                    float query_value = query_values[repeat * group_size + local];
                    quantized_dot[repeat] += query_value * level;
                    sign_dot[repeat] += qjl_sign * query_value;
                }
            }

            float norm = scales[tq_scale_offset_v7(
                batch, head, token, group, 0u, kv_heads, capacity, groups_per_vector)];
            float residual_norm = scales[tq_scale_offset_v7(
                batch, head, token, group, 1u, kv_heads, capacity, groups_per_vector)];
            float residual_scale = residual_norm * sqrt(3.14159265358979323846f / (2.0f * float(count)));
            for (uint repeat = 0u; repeat < 4u; repeat++) {
                scores[repeat] += norm * quantized_dot[repeat] + residual_scale * sign_dot[repeat];
            }
        }
        """

    private static let polarWHTAttentionHeader = attentionHeader + """

        inline float tq_polar_wht_centroid(uint bits, uint code) {
            if (bits <= 1u) {
                return code == 0u ? -0.7979f : 0.7979f;
            }
            if (bits == 2u) {
                switch (min(code, 3u)) {
                case 0u: return -1.5104f;
                case 1u: return -0.4528f;
                case 2u: return 0.4528f;
                default: return 1.5104f;
                }
            }
            if (bits == 3u) {
                switch (min(code, 7u)) {
                case 0u: return -2.1520f;
                case 1u: return -1.3440f;
                case 2u: return -0.7560f;
                case 3u: return -0.2451f;
                case 4u: return 0.2451f;
                case 5u: return 0.7560f;
                case 6u: return 1.3440f;
                default: return 2.1520f;
                }
            }
            switch (min(code, 15u)) {
            case 0u: return -2.7326f;
            case 1u: return -2.0690f;
            case 2u: return -1.6180f;
            case 3u: return -1.2562f;
            case 4u: return -0.9423f;
            case 5u: return -0.6568f;
            case 6u: return -0.3881f;
            case 7u: return -0.1284f;
            case 8u: return 0.1284f;
            case 9u: return 0.3881f;
            case 10u: return 0.6568f;
            case 11u: return 0.9423f;
            case 12u: return 1.2562f;
            case 13u: return 1.6180f;
            case 14u: return 2.0690f;
            default: return 2.7326f;
            }
        }

        template <typename PackedPtr>
        inline uint tq_polar_wht_read_index(
            PackedPtr packed,
            uint packed_base,
            uint dimension,
            uint bits
        ) {
            uint values_per_word = 32u / bits;
            uint word = packed[packed_base + dimension / values_per_word];
            uint offset = (dimension % values_per_word) * bits;
            return (word >> offset) & ((1u << bits) - 1u);
        }

        inline uint tq_polar_wht_quantize(float value, uint bits) {
            if (bits <= 1u) {
                return value > 0.0f ? 1u : 0u;
            }
            if (bits == 2u) {
                if (value <= -0.9816f) { return 0u; }
                if (value <= 0.0f) { return 1u; }
                if (value <= 0.9816f) { return 2u; }
                return 3u;
            }
            if (bits == 3u) {
                if (value <= -1.7480f) { return 0u; }
                if (value <= -1.0500f) { return 1u; }
                if (value <= -0.50055f) { return 2u; }
                if (value <= 0.0f) { return 3u; }
                if (value <= 0.50055f) { return 4u; }
                if (value <= 1.0500f) { return 5u; }
                if (value <= 1.7480f) { return 6u; }
                return 7u;
            }
            if (value <= -2.4008f) { return 0u; }
            if (value <= -1.8435f) { return 1u; }
            if (value <= -1.4371f) { return 2u; }
            if (value <= -1.09925f) { return 3u; }
            if (value <= -0.79955f) { return 4u; }
            if (value <= -0.52245f) { return 5u; }
            if (value <= -0.25825f) { return 6u; }
            if (value <= 0.0f) { return 7u; }
            if (value <= 0.25825f) { return 8u; }
            if (value <= 0.52245f) { return 9u; }
            if (value <= 0.79955f) { return 10u; }
            if (value <= 1.09925f) { return 11u; }
            if (value <= 1.4371f) { return 12u; }
            if (value <= 1.8435f) { return 13u; }
            if (value <= 2.4008f) { return 14u; }
            return 15u;
        }

        inline float tq_polar_wht_simdgroup_sum(
            float value,
            uint lane,
            uint head_dim,
            threadgroup float* partial
        ) {
            constexpr uint simd_width = 32u;
            float group_sum = simd_sum(value);
            if ((lane & (simd_width - 1u)) == 0u) {
                partial[lane >> 5u] = group_sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint group_count = (head_dim + simd_width - 1u) >> 5u;
            float total = lane < group_count ? partial[lane] : 0.0f;
            total = simd_sum(total);
            if (lane == 0u) {
                partial[0] = total;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            return partial[0];
        }

        inline void tq_polar_wht_simdgroup_wht(
            threadgroup float* values,
            uint lane,
            uint head_dim
        ) {
            bool active = lane < head_dim;
            for (uint width = 1u; width < head_dim && width < 32u; width <<= 1u) {
                float current = active ? values[lane] : 0.0f;
                float paired = simd_shuffle_xor(current, ushort(width));
                if (active) {
                    values[lane] = (lane & width) == 0u ? current + paired : paired - current;
                }
            }

            if (head_dim > 32u) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint width = 32u; width < head_dim; width <<= 1u) {
                    if (active && (lane & width) == 0u) {
                        uint paired = lane | width;
                        float lhs = values[lane];
                        float rhs = values[paired];
                        values[lane] = lhs + rhs;
                        values[paired] = lhs - rhs;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
        }
        """

    private static let encodeAttentionSource = """
        uint row_group_id = thread_position_in_grid.x;
        uint kv_heads = uint(KV_HEADS);
        uint capacity = uint(CAPACITY);
        uint groups_per_vector = uint(GROUPS_PER_VECTOR);
        uint mag_words_per_group = uint(MAG_WORDS_PER_GROUP);
        uint bitset_words_per_group = uint(BITSET_WORDS_PER_GROUP);
        uint total = uint(BATCH_SIZE) * kv_heads * uint(INPUT_LENGTH) * groups_per_vector;
        if (row_group_id >= total) {
            return;
        }

        uint group = row_group_id % groups_per_vector;
        uint token = (row_group_id / groups_per_vector) % uint(INPUT_LENGTH);
        uint head = (row_group_id / (groups_per_vector * uint(INPUT_LENGTH))) % kv_heads;
        uint batch = row_group_id / (groups_per_vector * uint(INPUT_LENGTH) * kv_heads);
        if (token >= capacity) {
            return;
        }

        uint group_start = group * uint(GROUP_SIZE);
        uint count = min(uint(GROUP_SIZE), uint(HEAD_DIM) - group_start);
        if (ROLE == 1) {
            float minimum = INFINITY;
            float maximum = -INFINITY;
            for (uint local = 0; local < count; local++) {
                uint dimension = group_start + local;
                long input_index =
                    long(batch) * x_strides[0]
                    + long(head) * x_strides[1]
                    + long(token) * x_strides[2]
                    + long(dimension) * x_strides[3];
                float value = float(x[input_index]);
                minimum = min(minimum, value);
                maximum = max(maximum, value);
            }

            float value_max = float((1 << VALUE_BITS) - 1);
            float range = maximum - minimum;
            float value_scale = range > 1.17549435e-38f ? range / value_max : 0.0f;
            uint scale_base = ((((batch * kv_heads + head) * capacity + token)
                * groups_per_vector + group) * 2u);
            scales[scale_base] = value_scale;
            scales[scale_base + 1u] = minimum;

            for (uint word = 0; word < mag_words_per_group; word++) {
                packed[tq_packed_offset(batch, head, token, group, word, kv_heads, capacity, groups_per_vector, mag_words_per_group)] = 0u;
            }
            for (uint local = 0; local < count; local++) {
                uint dimension = group_start + local;
                long input_index =
                    long(batch) * x_strides[0]
                    + long(head) * x_strides[1]
                    + long(token) * x_strides[2]
                    + long(dimension) * x_strides[3];
                float value = float(x[input_index]);
                uint quantized = value_scale == 0.0f
                    ? 0u
                    : uint(clamp(round((value - minimum) / value_scale), 0.0f, value_max));
                uint bit_offset = local * uint(VALUE_BITS);
                tq_write_packed_unsigned(
                    packed, quantized, batch, head, token, group, bit_offset, uint(VALUE_BITS),
                    kv_heads, capacity, groups_per_vector, mag_words_per_group);
            }
            return;
        }

        thread float values[GROUP_SIZE];
        ulong seed = tq_make_seed(uint(SEED_3), uint(SEED_2), uint(SEED_1), uint(SEED_0));
        uint storage_group = tq_storage_group_index(
            batch, head, token, group, kv_heads, capacity, groups_per_vector);
        float norm_squared = 0.0f;

        for (uint local = 0; local < count; local++) {
            uint dimension = group_start + local;
            long input_index =
                long(batch) * x_strides[0]
                + long(head) * x_strides[1]
                + long(token) * x_strides[2]
                + long(dimension) * x_strides[3];
            float value = float(x[input_index]);
            values[local] = value;
            norm_squared += value * value;
        }

        float norm = sqrt(norm_squared);
        float inv_norm = norm > 1.17549435e-38f ? 1.0f / norm : 0.0f;
        for (uint local = 0; local < count; local++) {
            values[local] *= inv_norm;
        }
        tq_apply_product_rotation(values, count, seed, storage_group, false);

        scales[tq_scale_offset(batch, head, token, group, 0u, kv_heads, capacity, groups_per_vector)] = norm;
        scales[tq_scale_offset(batch, head, token, group, 1u, kv_heads, capacity, groups_per_vector)] = 0.0f;

        bool split_magnitude =
            uint(LAYOUT_VERSION) >= 6u
            && uint(KEY_HIGH_BITS) == uint(KEY_BASE_BITS) + 1u
            && uint(KEY_HIGH_BITS) > uint(KEY_BASE_BITS);
        for (uint word = 0; word < bitset_words_per_group; word++) {
            signs[tq_bitset_offset(batch, head, token, group, word, kv_heads, capacity, groups_per_vector, bitset_words_per_group)] = 0u;
            if (!split_magnitude && uint(KEY_HIGH_BITS) > uint(KEY_BASE_BITS)) {
                high_mask[tq_bitset_offset(batch, head, token, group, word, kv_heads, capacity, groups_per_vector, bitset_words_per_group)] = 0u;
            }
        }
        for (uint word = 0; word < mag_words_per_group; word++) {
            packed[tq_packed_offset(batch, head, token, group, word, kv_heads, capacity, groups_per_vector, mag_words_per_group)] = 0u;
        }

        uint high_count = uint(round(float(count * uint(HIGH_NUMERATOR)) / float(uint(HIGH_DENOMINATOR))));
        float residual_squared = 0.0f;
        uint bit_offset = 0u;
        for (uint local = 0; local < count; local++) {
            bool high_precision = false;
            if (uint(KEY_HIGH_BITS) > uint(KEY_BASE_BITS) && high_count > 0u) {
                high_precision = split_magnitude
                    ? tq_split_high_precision(local, high_count)
                    : bool(DETERMINISTIC_HIGH_MASK)
                    ? tq_product_high_precision(seed, storage_group, local, count, high_count)
                    : local < high_count;
            }
            uint bits = high_precision ? uint(KEY_HIGH_BITS) : uint(KEY_BASE_BITS);
            uint quantized = tq_nearest_codebook_index(values[local], bits, count);
            float reconstructed = tq_codebook_level(bits, quantized, count);

            uint word = local >> 5;
            uint bit = local & 31u;
            uint mask = 1u << bit;
            if (high_precision && !split_magnitude) {
                high_mask[tq_bitset_offset(batch, head, token, group, word, kv_heads, capacity, groups_per_vector, bitset_words_per_group)] |= mask;
            }
            float residual = values[local] - reconstructed;
            residual_squared += residual * residual;
            if (residual < 0.0f) {
                signs[tq_bitset_offset(batch, head, token, group, word, kv_heads, capacity, groups_per_vector, bitset_words_per_group)] |= mask;
            }

            uint storage_bits = bits;
            uint storage_code = quantized;
            if (split_magnitude) {
                storage_bits = uint(KEY_BASE_BITS);
                storage_code = quantized & ((1u << uint(KEY_BASE_BITS)) - 1u);
                if (high_precision && ((quantized >> uint(KEY_BASE_BITS)) & 1u) != 0u) {
                    tq_write_packed_unsigned(
                        packed, 1u, batch, head, token, group,
                        uint(GROUP_SIZE) * uint(KEY_BASE_BITS) + local,
                        uint(KEY_HIGH_BITS) - uint(KEY_BASE_BITS),
                        kv_heads, capacity, groups_per_vector, mag_words_per_group);
                }
            }
            tq_write_packed_unsigned(
                packed, storage_code, batch, head, token, group, bit_offset, storage_bits,
                kv_heads, capacity, groups_per_vector, mag_words_per_group);
            bit_offset += storage_bits;
        }
        scales[tq_scale_offset(batch, head, token, group, 1u, kv_heads, capacity, groups_per_vector)] =
            norm * sqrt(residual_squared);
        """

    private static let encodeAttentionV7Source = """
        uint row_group_id = thread_position_in_grid.x;
        uint kv_heads = uint(KV_HEADS);
        uint capacity = uint(CAPACITY);
        uint groups_per_vector = uint(GROUPS_PER_VECTOR);
        uint mag_words_per_group = uint(MAG_WORDS_PER_GROUP);
        uint bitset_words_per_group = uint(BITSET_WORDS_PER_GROUP);
        uint total = uint(BATCH_SIZE) * kv_heads * uint(INPUT_LENGTH) * groups_per_vector;
        if (row_group_id >= total) {
            return;
        }

        uint group = row_group_id % groups_per_vector;
        uint token = (row_group_id / groups_per_vector) % uint(INPUT_LENGTH);
        uint head = (row_group_id / (groups_per_vector * uint(INPUT_LENGTH))) % kv_heads;
        uint batch = row_group_id / (groups_per_vector * uint(INPUT_LENGTH) * kv_heads);
        if (token >= capacity) {
            return;
        }

        uint group_start = group * uint(GROUP_SIZE);
        uint count = min(uint(GROUP_SIZE), uint(HEAD_DIM) - group_start);
        if (ROLE == 1) {
            float minimum = INFINITY;
            float maximum = -INFINITY;
            for (uint local = 0; local < count; local++) {
                uint dimension = group_start + local;
                long input_index =
                    long(batch) * x_strides[0]
                    + long(head) * x_strides[1]
                    + long(token) * x_strides[2]
                    + long(dimension) * x_strides[3];
                float value = float(x[input_index]);
                minimum = min(minimum, value);
                maximum = max(maximum, value);
            }

            float value_max = float((1 << VALUE_BITS) - 1);
            float range = maximum - minimum;
            float value_scale = range > 1.17549435e-38f ? range / value_max : 0.0f;
            uint scale_base = ((((batch * kv_heads + head) * capacity + token)
                * groups_per_vector + group) * 2u);
            scales[scale_base] = value_scale;
            scales[scale_base + 1u] = minimum;

            for (uint word = 0; word < mag_words_per_group; word++) {
                packed[tq_packed_offset(batch, head, token, group, word, kv_heads, capacity, groups_per_vector, mag_words_per_group)] = 0u;
            }
            for (uint local = 0; local < count; local++) {
                uint dimension = group_start + local;
                long input_index =
                    long(batch) * x_strides[0]
                    + long(head) * x_strides[1]
                    + long(token) * x_strides[2]
                    + long(dimension) * x_strides[3];
                float value = float(x[input_index]);
                uint quantized = value_scale == 0.0f
                    ? 0u
                    : uint(clamp(round((value - minimum) / value_scale), 0.0f, value_max));
                uint bit_offset = local * uint(VALUE_BITS);
                tq_write_packed_unsigned(
                    packed, quantized, batch, head, token, group, bit_offset, uint(VALUE_BITS),
                    kv_heads, capacity, groups_per_vector, mag_words_per_group);
            }
            return;
        }

        thread float values[GROUP_SIZE];
        ulong seed = tq_make_seed(uint(SEED_3), uint(SEED_2), uint(SEED_1), uint(SEED_0));
        uint storage_group = tq_storage_group_index(
            batch, head, token, group, kv_heads, capacity, groups_per_vector);
        float norm_squared = 0.0f;

        for (uint local = 0; local < count; local++) {
            uint dimension = group_start + local;
            long input_index =
                long(batch) * x_strides[0]
                + long(head) * x_strides[1]
                + long(token) * x_strides[2]
                + long(dimension) * x_strides[3];
            float value = float(x[input_index]);
            values[local] = value;
            norm_squared += value * value;
        }

        float norm = sqrt(norm_squared);
        float inv_norm = norm > 1.17549435e-38f ? 1.0f / norm : 0.0f;
        for (uint local = 0; local < count; local++) {
            values[local] *= inv_norm;
        }
        tq_apply_product_rotation(values, count, seed, storage_group, false);

        scales[tq_scale_offset_v7(batch, head, token, group, 0u, kv_heads, capacity, groups_per_vector)] = norm;
        scales[tq_scale_offset_v7(batch, head, token, group, 1u, kv_heads, capacity, groups_per_vector)] = 0.0f;

        bool split_magnitude =
            uint(LAYOUT_VERSION) >= 6u
            && uint(KEY_HIGH_BITS) == uint(KEY_BASE_BITS) + 1u
            && uint(KEY_HIGH_BITS) > uint(KEY_BASE_BITS);
        for (uint word = 0; word < bitset_words_per_group; word++) {
            signs[tq_bitset_offset_v7(batch, head, token, group, word, kv_heads, capacity, groups_per_vector, bitset_words_per_group)] = 0u;
            if (!split_magnitude && uint(KEY_HIGH_BITS) > uint(KEY_BASE_BITS)) {
                high_mask[tq_bitset_offset_v7(batch, head, token, group, word, kv_heads, capacity, groups_per_vector, bitset_words_per_group)] = 0u;
            }
        }
        for (uint word = 0; word < mag_words_per_group; word++) {
            packed[tq_packed_offset_v7(batch, head, token, group, word, kv_heads, capacity, groups_per_vector, mag_words_per_group)] = 0u;
        }

        uint high_count = uint(round(float(count * uint(HIGH_NUMERATOR)) / float(uint(HIGH_DENOMINATOR))));
        float residual_squared = 0.0f;
        uint bit_offset = 0u;
        for (uint local = 0; local < count; local++) {
            bool high_precision = false;
            if (uint(KEY_HIGH_BITS) > uint(KEY_BASE_BITS) && high_count > 0u) {
                high_precision = split_magnitude
                    ? tq_split_high_precision(local, high_count)
                    : bool(DETERMINISTIC_HIGH_MASK)
                    ? tq_product_high_precision(seed, storage_group, local, count, high_count)
                    : local < high_count;
            }
            uint bits = high_precision ? uint(KEY_HIGH_BITS) : uint(KEY_BASE_BITS);
            uint quantized = tq_nearest_codebook_index(values[local], bits, count);
            float reconstructed = tq_codebook_level(bits, quantized, count);

            uint word = local >> 5;
            uint bit = local & 31u;
            uint mask = 1u << bit;
            if (high_precision && !split_magnitude) {
                high_mask[tq_bitset_offset_v7(batch, head, token, group, word, kv_heads, capacity, groups_per_vector, bitset_words_per_group)] |= mask;
            }
            float residual = values[local] - reconstructed;
            residual_squared += residual * residual;
            if (residual < 0.0f) {
                signs[tq_bitset_offset_v7(batch, head, token, group, word, kv_heads, capacity, groups_per_vector, bitset_words_per_group)] |= mask;
            }

            uint storage_bits = bits;
            uint storage_code = quantized;
            if (split_magnitude) {
                storage_bits = uint(KEY_BASE_BITS);
                storage_code = quantized & ((1u << uint(KEY_BASE_BITS)) - 1u);
                if (high_precision && ((quantized >> uint(KEY_BASE_BITS)) & 1u) != 0u) {
                    tq_write_packed_unsigned_v7(
                        packed, 1u, batch, head, token, group,
                        uint(GROUP_SIZE) * uint(KEY_BASE_BITS) + local,
                        uint(KEY_HIGH_BITS) - uint(KEY_BASE_BITS),
                        kv_heads, capacity, groups_per_vector, mag_words_per_group);
                }
            }
            tq_write_packed_unsigned_v7(
                packed, storage_code, batch, head, token, group, bit_offset, storage_bits,
                kv_heads, capacity, groups_per_vector, mag_words_per_group);
            bit_offset += storage_bits;
        }
        scales[tq_scale_offset_v7(batch, head, token, group, 1u, kv_heads, capacity, groups_per_vector)] =
            norm * sqrt(residual_squared);
        """

    private static let qkSource = """
        uint index = thread_position_in_grid.x;
        uint logical_length = uint(runtime_logical_length);
        uint ring_offset = uint(runtime_ring_offset);
        uint pinned_prefix_length = uint(runtime_pinned_prefix_length);
        uint total = uint(BATCH_SIZE) * uint(QUERY_HEADS) * uint(QUERY_LENGTH) * logical_length;
        if (index >= total) {
            return;
        }

        float attention_scale = float(runtime_attention_scale);
        uint logical_token = index % logical_length;
        uint q_token = (index / logical_length) % uint(QUERY_LENGTH);
        uint q_head = (index / (logical_length * uint(QUERY_LENGTH))) % uint(QUERY_HEADS);
        uint batch = index / (logical_length * uint(QUERY_LENGTH) * uint(QUERY_HEADS));
        uint repeats = uint(QUERY_HEADS) / uint(KV_HEADS);
        uint kv_head = q_head / repeats;
        uint physical_token = tq_physical_token(
            logical_token, uint(CAPACITY), ring_offset, pinned_prefix_length);

        float sum = 0.0f;
        ulong seed = tq_make_seed(uint(SEED_3), uint(SEED_2), uint(SEED_1), uint(SEED_0));
        for (uint group = 0u; group < uint(GROUPS_PER_VECTOR); group++) {
            uint group_start = group * uint(GROUP_SIZE);
            uint count = min(uint(GROUP_SIZE), uint(HEAD_DIM) - group_start);
            thread float query_values[GROUP_SIZE];
            for (uint local = 0u; local < count; local++) {
                uint dimension = group_start + local;
                long q_index =
                    long(batch) * q_strides[0]
                    + long(q_head) * q_strides[1]
                    + long(q_token) * q_strides[2]
                    + long(dimension) * q_strides[3];
                query_values[local] = float(q[q_index]);
            }
                sum += tq_product_attention_inner_product_group(
                    k_packed, k_signs, k_high_mask, k_residual_signs, k_scales, query_values,
                    batch, kv_head, physical_token, group, seed,
                    uint(GROUP_SIZE), uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR),
                    uint(MAG_WORDS_PER_GROUP), uint(BITSET_WORDS_PER_GROUP),
                    uint(KEY_BASE_BITS), uint(KEY_HIGH_BITS), uint(LAYOUT_VERSION), uint(HEAD_DIM),
                    tq_high_precision_count(count, uint(HIGH_NUMERATOR), uint(HIGH_DENOMINATOR)));
        }
        scores[index] = sum * attention_scale;
        """

    private static let polarWHTEncodeAttentionSource = """
        constexpr uint head_dim = uint(HEAD_DIM);
        uint lane = thread_position_in_threadgroup.x;
        uint vector_id = threadgroup_position_in_grid.x;
        uint vector_count = uint(BATCH_SIZE) * uint(KV_HEADS) * uint(INPUT_LENGTH);
        if (vector_id >= vector_count || lane >= head_dim) {
            return;
        }

        threadgroup float rotated[HEAD_DIM];
        threadgroup float partial[HEAD_DIM];
        threadgroup uint codes[HEAD_DIM];

        uint token = vector_id % uint(INPUT_LENGTH);
        uint head = (vector_id / uint(INPUT_LENGTH)) % uint(KV_HEADS);
        uint batch = vector_id / (uint(INPUT_LENGTH) * uint(KV_HEADS));
        long x_index =
            long(batch) * x_strides[0]
            + long(head) * x_strides[1]
            + long(token) * x_strides[2]
            + long(lane) * x_strides[3];
        float value = float(x[x_index]);
        float norm = sqrt(tq_polar_wht_simdgroup_sum(value * value, lane, head_dim, partial));
        float inv_norm = norm > 1.17549435e-38f ? 1.0f / norm : 0.0f;
        ulong seed = tq_make_seed(uint(SEED_3), uint(SEED_2), uint(SEED_1), uint(SEED_0));
        float sign = tq_random_sign(seed, lane) ? -1.0f : 1.0f;
        rotated[lane] = value * sign * inv_norm;
        tq_polar_wht_simdgroup_wht(rotated, lane, head_dim);

        codes[lane] = tq_polar_wht_quantize(rotated[lane], uint(POLAR_WHT_BITS));
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint physical_token = tq_physical_token(
            token, uint(CAPACITY), uint(RING_OFFSET), uint(PINNED_PREFIX_LENGTH));
        uint vector_index =
            (batch * uint(KV_HEADS) + head) * uint(CAPACITY) + physical_token;
        if (lane == 0u) {
            norms[vector_index] = norm;
        }

        uint values_per_word = 32u / uint(POLAR_WHT_BITS);
        if (lane < uint(PACKED_WORDS_PER_VECTOR)) {
            uint word = 0u;
            uint first_dimension = lane * values_per_word;
            for (uint local = 0u; local < values_per_word; local++) {
                uint dimension = first_dimension + local;
                if (dimension < head_dim) {
                    word |= codes[dimension] << (local * uint(POLAR_WHT_BITS));
                }
            }
            packed_indices[vector_index * uint(PACKED_WORDS_PER_VECTOR) + lane] = word;
        }
        """

    private static let hybridAffineK8PolarWHTValueEncodeSource = """
        constexpr uint head_dim = uint(HEAD_DIM);
        uint lane = thread_position_in_threadgroup.x;
        uint vector_id = threadgroup_position_in_grid.x;
        uint vector_count = uint(BATCH_SIZE) * uint(KV_HEADS) * uint(INPUT_LENGTH);
        if (vector_id >= vector_count || lane >= head_dim) {
            return;
        }

        threadgroup float rotated[HEAD_DIM];
        threadgroup float partial[HEAD_DIM];
        threadgroup uint codes[HEAD_DIM];

        uint token = vector_id % uint(INPUT_LENGTH);
        uint head = (vector_id / uint(INPUT_LENGTH)) % uint(KV_HEADS);
        uint batch = vector_id / (uint(INPUT_LENGTH) * uint(KV_HEADS));
        uint physical_token = tq_physical_token(
            token, uint(CAPACITY), uint(RING_OFFSET), uint(PINNED_PREFIX_LENGTH));

        if (lane < uint(KEY_GROUPS_PER_VECTOR)) {
            uint group = lane;
            uint group_start = group * uint(KEY_GROUP_SIZE);
            float minimum = INFINITY;
            float maximum = 0.0f;
            thread float key_values[KEY_GROUP_SIZE];
            for (uint local = 0u; local < uint(KEY_GROUP_SIZE); local++) {
                uint dimension = group_start + local;
                long key_index =
                    long(batch) * keys_strides[0]
                    + long(head) * keys_strides[1]
                    + long(token) * keys_strides[2]
                    + long(dimension) * keys_strides[3];
                float key_value = float(keys[key_index]);
                key_values[local] = key_value;
                minimum = min(minimum, key_value);
                maximum = max(maximum, key_value);
            }

            constexpr float n_bins = 255.0f;
            constexpr float eps = 1e-7f;
            float key_scale = max((maximum - minimum) / n_bins, eps);
            bool use_min_edge = abs(minimum) > abs(maximum);
            key_scale = use_min_edge ? key_scale : -key_scale;
            float edge = use_min_edge ? minimum : maximum;
            float q0 = round(edge / key_scale);
            bool at_zero = q0 == 0.0f;
            key_scale = at_zero ? key_scale : edge / q0;
            float key_bias = at_zero ? 0.0f : edge;

            uint key_vector =
                (batch * uint(KV_HEADS) + head) * uint(CAPACITY) + physical_token;
            uint scale_base = key_vector * uint(KEY_GROUPS_PER_VECTOR) + group;
            key_scales[scale_base] = static_cast<KEY_SCALE_DTYPE>(key_scale);
            key_biases[scale_base] = static_cast<KEY_SCALE_DTYPE>(key_bias);

            uint packed_base = key_vector * uint(KEY_PACKED_WORDS_PER_VECTOR);
            uint group_words = uint(KEY_GROUP_SIZE) >> 2;
            for (uint word = 0u; word < group_words; word++) {
                uint packed = 0u;
                for (uint local = 0u; local < 4u; local++) {
                    uint group_local = word * 4u + local;
                    float normalized = round((key_values[group_local] - key_bias) / key_scale);
                    uint code = uint(clamp(normalized, 0.0f, n_bins));
                    packed |= code << (local << 3);
                }
                key_packed[packed_base + group * group_words + word] = packed;
            }
        }

        long value_index =
            long(batch) * values_strides[0]
            + long(head) * values_strides[1]
            + long(token) * values_strides[2]
            + long(lane) * values_strides[3];
        float value = float(values[value_index]);
        float norm = sqrt(tq_polar_wht_simdgroup_sum(value * value, lane, head_dim, partial));
        float inv_norm = norm > 1.17549435e-38f ? 1.0f / norm : 0.0f;
        ulong value_seed = tq_make_seed(
            uint(VALUE_SEED_3), uint(VALUE_SEED_2), uint(VALUE_SEED_1),
            uint(VALUE_SEED_0));
        float sign = tq_random_sign(value_seed, lane) ? -1.0f : 1.0f;
        rotated[lane] = value * sign * inv_norm;
        tq_polar_wht_simdgroup_wht(rotated, lane, head_dim);

        codes[lane] = tq_polar_wht_quantize(rotated[lane], uint(POLAR_WHT_BITS));
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint vector_index =
            (batch * uint(KV_HEADS) + head) * uint(CAPACITY) + physical_token;
        if (lane == 0u) {
            value_norms[vector_index] = norm;
        }

        uint values_per_word = 32u / uint(POLAR_WHT_BITS);
        if (lane < uint(PACKED_WORDS_PER_VECTOR)) {
            uint word = 0u;
            uint first_dimension = lane * values_per_word;
            for (uint local = 0u; local < values_per_word; local++) {
                uint dimension = first_dimension + local;
                if (dimension < head_dim) {
                    word |= codes[dimension] << (local * uint(POLAR_WHT_BITS));
                }
            }
            value_packed_indices[vector_index * uint(PACKED_WORDS_PER_VECTOR) + lane] = word;
        }
        """

    private static let polarWHTSIMDEncodeNormSnippet =
        "float norm = sqrt(tq_polar_wht_simdgroup_sum(value * value, lane, head_dim, partial));"

    private static let polarWHTThreadgroupEncodeNormSnippet = """
        partial[lane] = value * value;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = head_dim >> 1; stride > 0u; stride >>= 1) {
            if (lane < stride) {
                partial[lane] += partial[lane + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        float norm = sqrt(partial[0]);
        """

    private static let polarWHTSIMDEncodeWHTSnippet =
        "tq_polar_wht_simdgroup_wht(rotated, lane, head_dim);"

    private static let polarWHTThreadgroupEncodeWHTSnippet = """
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint width = 1u; width < head_dim; width <<= 1u) {
            if ((lane & width) == 0u) {
                uint paired = lane | width;
                float lhs = rotated[lane];
                float rhs = rotated[paired];
                rotated[lane] = lhs + rhs;
                rotated[paired] = lhs - rhs;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        """

    private static let polarWHTEncodeAttentionBulkSource =
        polarWHTEncodeAttentionSource
        .replacingOccurrences(
            of: polarWHTSIMDEncodeNormSnippet,
            with: polarWHTThreadgroupEncodeNormSnippet
        )
        .replacingOccurrences(
            of: polarWHTSIMDEncodeWHTSnippet,
            with: polarWHTThreadgroupEncodeWHTSnippet
        )

    private static let hybridAffineK8PolarWHTValueEncodeBulkSource =
        hybridAffineK8PolarWHTValueEncodeSource
        .replacingOccurrences(
            of: polarWHTSIMDEncodeNormSnippet,
            with: polarWHTThreadgroupEncodeNormSnippet
        )
        .replacingOccurrences(
            of: polarWHTSIMDEncodeWHTSnippet,
            with: polarWHTThreadgroupEncodeWHTSnippet
        )

    private static let polarWHTDecodeAttentionSource = """
        constexpr uint head_dim = uint(HEAD_DIM);
        uint lane = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;
        uint logical_length = uint(runtime_logical_length);
        uint ring_offset = uint(runtime_ring_offset);
        uint pinned_prefix_length = uint(runtime_pinned_prefix_length);
        uint row_count = uint(BATCH_SIZE) * uint(KV_HEADS) * logical_length;
        if (row >= row_count || lane >= head_dim) {
            return;
        }

        threadgroup float accumulated[HEAD_DIM];

        uint logical_token = row % logical_length;
        uint head = (row / logical_length) % uint(KV_HEADS);
        uint batch = row / (logical_length * uint(KV_HEADS));
        uint physical_token = tq_physical_token(
            logical_token, uint(CAPACITY), ring_offset, pinned_prefix_length);
        uint vector_index =
            (batch * uint(KV_HEADS) + head) * uint(CAPACITY) + physical_token;
        uint code = tq_polar_wht_read_index(
            packed_indices,
            vector_index * uint(PACKED_WORDS_PER_VECTOR),
            lane,
            uint(POLAR_WHT_BITS));
        accumulated[lane] = tq_polar_wht_centroid(uint(POLAR_WHT_BITS), code)
            * rsqrt(float(head_dim));
        tq_polar_wht_simdgroup_wht(accumulated, lane, head_dim);

        ulong seed = tq_make_seed(uint(SEED_3), uint(SEED_2), uint(SEED_1), uint(SEED_0));
        float sign = tq_random_sign(seed, lane) ? -1.0f : 1.0f;
        out[row * head_dim + lane] = static_cast<OUTPUT_DTYPE>(
            accumulated[lane] * rsqrt(float(head_dim))
                * sign * float(norms[vector_index]));
        """

    private static let polarWHTQKSource = """
        constexpr uint head_dim = uint(HEAD_DIM);
        uint lane = thread_position_in_threadgroup.x;
        uint score_id = threadgroup_position_in_grid.x;
        uint logical_length = uint(runtime_logical_length);
        uint ring_offset = uint(runtime_ring_offset);
        uint pinned_prefix_length = uint(runtime_pinned_prefix_length);
        uint score_count = uint(BATCH_SIZE) * uint(QUERY_HEADS) * uint(QUERY_LENGTH)
            * logical_length;
        if (score_id >= score_count || lane >= head_dim) {
            return;
        }

        threadgroup float query_rotated[HEAD_DIM];
        threadgroup float partial[HEAD_DIM];

        uint logical_token = score_id % logical_length;
        uint q_token = (score_id / logical_length) % uint(QUERY_LENGTH);
        uint q_head = (score_id / (logical_length * uint(QUERY_LENGTH))) % uint(QUERY_HEADS);
        uint batch = score_id / (logical_length * uint(QUERY_LENGTH) * uint(QUERY_HEADS));
        uint repeats = uint(QUERY_HEADS) / uint(KV_HEADS);
        uint kv_head = q_head / repeats;

        long q_index =
            long(batch) * q_strides[0]
            + long(q_head) * q_strides[1]
            + long(q_token) * q_strides[2]
            + long(lane) * q_strides[3];
        ulong seed = tq_make_seed(uint(SEED_3), uint(SEED_2), uint(SEED_1), uint(SEED_0));
        float sign = tq_random_sign(seed, lane) ? -1.0f : 1.0f;
        query_rotated[lane] = float(q[q_index]) * sign;
        tq_polar_wht_simdgroup_wht(query_rotated, lane, head_dim);

        uint physical_token = tq_physical_token(
            logical_token, uint(CAPACITY), ring_offset, pinned_prefix_length);
        uint vector_index =
            (batch * uint(KV_HEADS) + kv_head) * uint(CAPACITY) + physical_token;
        uint code = tq_polar_wht_read_index(
            k_packed_indices,
            vector_index * uint(PACKED_WORDS_PER_VECTOR),
            lane,
            uint(POLAR_WHT_BITS));
        float inv_sqrt_dim = rsqrt(float(head_dim));
        float rotated_query = query_rotated[lane] * inv_sqrt_dim;
        float key_level = tq_polar_wht_centroid(uint(POLAR_WHT_BITS), code) * inv_sqrt_dim;
        partial[lane] = rotated_query * key_level;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = head_dim >> 1; stride > 0u; stride >>= 1) {
            if (lane < stride) {
                partial[lane] += partial[lane + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0u) {
            scores[score_id] = partial[0] * float(k_norms[vector_index])
                * float(runtime_attention_scale);
        }
        """

    private static let keyPageSummarySource = """
        constexpr uint page_size = uint(PAGE_SIZE);
        uint lane = thread_position_in_threadgroup.x;
        uint summary_id = threadgroup_position_in_grid.x;
        uint total =
            uint(BATCH_SIZE) * uint(KV_HEADS) * uint(PAGE_CAPACITY) *
            uint(GROUPS_PER_VECTOR);
        if (summary_id >= total) {
            return;
        }

        threadgroup float partial[PAGE_SIZE];

        uint group = summary_id % uint(GROUPS_PER_VECTOR);
        uint page = (summary_id / uint(GROUPS_PER_VECTOR)) % uint(PAGE_CAPACITY);
        uint head = (summary_id / (uint(GROUPS_PER_VECTOR) * uint(PAGE_CAPACITY))) %
            uint(KV_HEADS);
        uint batch = summary_id /
            (uint(GROUPS_PER_VECTOR) * uint(PAGE_CAPACITY) * uint(KV_HEADS));
        uint logical_length = uint(runtime_logical_length);
        uint ring_offset = uint(runtime_ring_offset);
        uint pinned_prefix_length = uint(runtime_pinned_prefix_length);

        float summary_value = 0.0f;
        uint logical_token = page * page_size + lane;
        if (lane < page_size && logical_token < logical_length) {
            uint physical_token = tq_physical_token(
                logical_token, uint(CAPACITY), ring_offset, pinned_prefix_length);
            long base =
                long(batch) * scales_strides[0]
                + long(head) * scales_strides[1]
                + long(physical_token) * scales_strides[2]
                + long(group) * scales_strides[3];
            float key_norm = abs(float(scales[base]));
            float residual_norm = uint(SCALES_PER_GROUP) > 1u
                ? abs(float(scales[base + scales_strides[4]]))
                : 0.0f;
            summary_value = key_norm + residual_norm;
        }
        partial[lane] = summary_value;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = page_size >> 1; stride > 0u; stride >>= 1) {
            if (lane < stride) {
                partial[lane] = max(partial[lane], partial[lane + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0u) {
            summary[summary_id] = partial[0];
        }
        """

    private static let decodeAttentionSource = """
        uint index = thread_position_in_grid.x;
        uint logical_length = uint(runtime_logical_length);
        uint ring_offset = uint(runtime_ring_offset);
        uint pinned_prefix_length = uint(runtime_pinned_prefix_length);
        uint total = uint(BATCH_SIZE) * uint(KV_HEADS) * logical_length * uint(HEAD_DIM);
        if (index >= total) {
            return;
        }

        uint dimension = index % uint(HEAD_DIM);
        uint logical_token = (index / uint(HEAD_DIM)) % logical_length;
        uint head = (index / (uint(HEAD_DIM) * logical_length)) % uint(KV_HEADS);
        uint batch = index / (uint(HEAD_DIM) * logical_length * uint(KV_HEADS));
        uint physical_token = tq_physical_token(
            logical_token, uint(CAPACITY), ring_offset, pinned_prefix_length);
        uint group_start = (dimension / uint(GROUP_SIZE)) * uint(GROUP_SIZE);
        uint count = min(uint(GROUP_SIZE), uint(HEAD_DIM) - group_start);
        thread float decode_scratch[GROUP_SIZE];
        out[index] = static_cast<OUTPUT_DTYPE>(tq_decode_attention_value(
            packed, signs, high_mask, residual_signs, scales,
            batch, head, physical_token, dimension,
            tq_make_seed(uint(SEED_3), uint(SEED_2), uint(SEED_1), uint(SEED_0)), uint(ROLE),
            uint(GROUP_SIZE), uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR),
            uint(MAG_WORDS_PER_GROUP), uint(BITSET_WORDS_PER_GROUP), uint(BASE_BITS), uint(HIGH_BITS),
            uint(VALUE_BITS), uint(KEY_BASE_BITS), uint(KEY_HIGH_BITS), uint(LAYOUT_VERSION),
            uint(HEAD_DIM), tq_high_precision_count(count, uint(HIGH_NUMERATOR), uint(HIGH_DENOMINATOR)),
            decode_scratch));
        """

    private static let avSource = """
        uint index = thread_position_in_grid.x;
        uint total = uint(BATCH_SIZE) * uint(QUERY_HEADS) * uint(QUERY_LENGTH) * uint(HEAD_DIM);
        uint logical_length = uint(runtime_logical_length);
        uint ring_offset = uint(runtime_ring_offset);
        uint pinned_prefix_length = uint(runtime_pinned_prefix_length);
        if (index >= total) {
            return;
        }

        uint dimension = index % uint(HEAD_DIM);
        uint q_token = (index / uint(HEAD_DIM)) % uint(QUERY_LENGTH);
        uint q_head = (index / (uint(HEAD_DIM) * uint(QUERY_LENGTH))) % uint(QUERY_HEADS);
        uint batch = index / (uint(HEAD_DIM) * uint(QUERY_LENGTH) * uint(QUERY_HEADS));
        uint repeats = uint(QUERY_HEADS) / uint(KV_HEADS);
        uint kv_head = q_head / repeats;

        float sum = 0.0f;
        thread float decode_scratch[GROUP_SIZE];
        for (uint logical_token = 0; logical_token < logical_length; logical_token++) {
            uint weight_index =
                (((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH) + q_token)
                    * logical_length) + logical_token;
            float weight = float(weights[weight_index]);
            if (weight == 0.0f) {
                continue;
            }
            uint physical_token = tq_physical_token(
                logical_token, uint(CAPACITY), ring_offset, pinned_prefix_length);
            float value = tq_decode_attention_value(
                v_packed, v_signs, v_high_mask, v_residual_signs, v_scales,
                batch, kv_head, physical_token, dimension,
                tq_make_seed(uint(SEED_3), uint(SEED_2), uint(SEED_1), uint(SEED_0)), 1u,
                uint(GROUP_SIZE), uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR),
            uint(MAG_WORDS_PER_GROUP), uint(BITSET_WORDS_PER_GROUP), uint(BASE_BITS), uint(HIGH_BITS),
            uint(VALUE_BITS), uint(KEY_BASE_BITS), uint(KEY_HIGH_BITS), uint(LAYOUT_VERSION),
            uint(HEAD_DIM), 0u,
            decode_scratch);
            sum += weight * value;
        }
        out[index] = static_cast<OUTPUT_DTYPE>(sum);
        """

    private static let polarWHTAVSource = """
        constexpr uint head_dim = uint(HEAD_DIM);
        uint lane = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;
        uint row_count = uint(BATCH_SIZE) * uint(QUERY_HEADS) * uint(QUERY_LENGTH);
        uint logical_length = uint(runtime_logical_length);
        uint ring_offset = uint(runtime_ring_offset);
        uint pinned_prefix_length = uint(runtime_pinned_prefix_length);
        if (row >= row_count || lane >= head_dim) {
            return;
        }

        threadgroup float accumulated[HEAD_DIM];

        uint q_token = row % uint(QUERY_LENGTH);
        uint q_head = (row / uint(QUERY_LENGTH)) % uint(QUERY_HEADS);
        uint batch = row / (uint(QUERY_LENGTH) * uint(QUERY_HEADS));
        uint repeats = uint(QUERY_HEADS) / uint(KV_HEADS);
        uint kv_head = q_head / repeats;
        float centroid_scale = rsqrt(float(head_dim));

        float sum = 0.0f;
        for (uint logical_token = 0u; logical_token < logical_length; logical_token++) {
            uint weight_index =
                (((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH) + q_token)
                    * logical_length) + logical_token;
            float weight = float(weights[weight_index]);
            if (weight == 0.0f) {
                continue;
            }
            uint physical_token = tq_physical_token(
                logical_token, uint(CAPACITY), ring_offset, pinned_prefix_length);
            uint vector_index =
                (batch * uint(KV_HEADS) + kv_head) * uint(CAPACITY) + physical_token;
            uint code = tq_polar_wht_read_index(
                v_packed_indices,
                vector_index * uint(PACKED_WORDS_PER_VECTOR),
                lane,
                uint(POLAR_WHT_BITS));
            float centroid = tq_polar_wht_centroid(uint(POLAR_WHT_BITS), code);
            sum += weight * float(v_norms[vector_index]) * centroid * centroid_scale;
        }
        accumulated[lane] = sum;
        tq_polar_wht_simdgroup_wht(accumulated, lane, head_dim);

        ulong seed = tq_make_seed(uint(SEED_3), uint(SEED_2), uint(SEED_1), uint(SEED_0));
        float sign = tq_random_sign(seed, lane) ? -1.0f : 1.0f;
        out[row * head_dim + lane] = static_cast<OUTPUT_DTYPE>(
            accumulated[lane] * rsqrt(float(head_dim)) * sign);
        """

    private static let hybridPolarWHTValueFusedAttentionSource = """
        constexpr uint head_dim = uint(HEAD_DIM);
        constexpr uint threads_per_row = uint(THREADS_PER_ROW);
        uint lane = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;
        uint total_rows = uint(BATCH_SIZE) * uint(QUERY_HEADS) * uint(QUERY_LENGTH);
        if (row >= total_rows) {
            return;
        }

        threadgroup float partial[256];
        threadgroup float tile_scores[256];
        threadgroup uint tile_physical_tokens[256];
        threadgroup float query_cache[HEAD_DIM];
        threadgroup float output_accum[HEAD_DIM];

        uint logical_length = uint(runtime_logical_length);
        uint ring_offset = uint(runtime_ring_offset);
        uint pinned_prefix_length = uint(runtime_pinned_prefix_length);
        float attention_scale = float(runtime_attention_scale);
        uint q_token = row % uint(QUERY_LENGTH);
        uint q_head = (row / uint(QUERY_LENGTH)) % uint(QUERY_HEADS);
        uint batch = row / (uint(QUERY_LENGTH) * uint(QUERY_HEADS));
        uint repeats = uint(QUERY_HEADS) / uint(KV_HEADS);
        uint kv_head = q_head / repeats;
        uint causal_limit = logical_length - uint(QUERY_LENGTH) + q_token;
        ulong key_seed = tq_make_seed(uint(SEED_3), uint(SEED_2), uint(SEED_1), uint(SEED_0));

        float row_max = -INFINITY;
        float row_sum = 0.0f;
        if (lane < head_dim) {
            long q_index =
                long(batch) * q_strides[0]
                + long(q_head) * q_strides[1]
                + long(q_token) * q_strides[2]
                + long(lane) * q_strides[3];
            query_cache[lane] = float(q[q_index]);
            output_accum[lane] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint tile_start = 0u; tile_start < logical_length; tile_start += threads_per_row) {
            uint logical_token = tile_start + lane;
            bool active = lane < threads_per_row
                && logical_token < logical_length
                && (!DO_CAUSAL || logical_token <= causal_limit);
            float scaled_score = -INFINITY;
            uint physical_token = 0u;
            if (active) {
                physical_token = tq_physical_token(
                    logical_token, uint(CAPACITY), ring_offset, pinned_prefix_length);
                float score = 0.0f;
                for (uint group = 0u; group < uint(GROUPS_PER_VECTOR); group++) {
                    uint group_start = group * uint(GROUP_SIZE);
                    uint count = min(uint(GROUP_SIZE), head_dim - group_start);
                    thread float query_values[GROUP_SIZE];
                    for (uint local = 0u; local < count; local++) {
                        query_values[local] = query_cache[group_start + local];
                    }
                    score += tq_product_attention_inner_product_group(
                        k_packed, k_signs, k_high_mask, k_residual_signs, k_scales, query_values,
                        batch, kv_head, physical_token, group, key_seed,
                        uint(GROUP_SIZE), uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR),
                        uint(MAG_WORDS_PER_GROUP), uint(BITSET_WORDS_PER_GROUP),
                        uint(KEY_BASE_BITS), uint(KEY_HIGH_BITS), uint(LAYOUT_VERSION),
                        head_dim,
                        tq_high_precision_count(count, uint(HIGH_NUMERATOR), uint(HIGH_DENOMINATOR)));
                }
                scaled_score = score * attention_scale;
            }
            tile_scores[lane] = scaled_score;
            tile_physical_tokens[lane] = physical_token;
            partial[lane] = scaled_score;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = threads_per_row >> 1; stride > 0u; stride >>= 1) {
                if (lane < stride) {
                    partial[lane] = max(partial[lane], partial[lane + stride]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            float tile_max = partial[0];
            float new_row_max = max(row_max, tile_max);
            float old_scale = row_sum > 0.0f ? exp(row_max - new_row_max) : 0.0f;
            if (lane < head_dim) {
                output_accum[lane] *= old_scale;
            }

            float weight = active ? exp(tile_scores[lane] - new_row_max) : 0.0f;
            tile_scores[lane] = weight;
            partial[lane] = weight;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = threads_per_row >> 1; stride > 0u; stride >>= 1) {
                if (lane < stride) {
                    partial[lane] += partial[lane + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            float next_row_sum = row_sum * old_scale + partial[0];
            if (lane < head_dim) {
                float centroid_scale = rsqrt(float(head_dim));
                float dimension_accum = output_accum[lane];
                for (uint tile_lane = 0u; tile_lane < threads_per_row; tile_lane++) {
                    float tile_weight = tile_scores[tile_lane];
                    if (tile_weight > 0.0f) {
                        uint vector_index =
                            (batch * uint(KV_HEADS) + kv_head) * uint(CAPACITY)
                            + tile_physical_tokens[tile_lane];
                        uint code = tq_polar_wht_read_index(
                            v_packed_indices,
                            vector_index * uint(PACKED_WORDS_PER_VECTOR),
                            lane,
                            uint(POLAR_WHT_BITS));
                        float centroid = tq_polar_wht_centroid(uint(POLAR_WHT_BITS), code);
                        dimension_accum +=
                            tile_weight * float(v_norms[vector_index]) * centroid * centroid_scale;
                    }
                }
                output_accum[lane] = dimension_accum;
            }
            row_max = new_row_max;
            row_sum = next_row_sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane < head_dim) {
            output_accum[lane] *= 1.0f / max(row_sum, 1.17549435e-38f);
        }
        tq_polar_wht_simdgroup_wht(output_accum, lane, head_dim);

        if (lane < head_dim) {
            ulong value_seed = tq_make_seed(
                uint(VALUE_SEED_3), uint(VALUE_SEED_2),
                uint(VALUE_SEED_1), uint(VALUE_SEED_0));
            float sign = tq_random_sign(value_seed, lane) ? -1.0f : 1.0f;
            out[row * head_dim + lane] = static_cast<OUTPUT_DTYPE>(
                output_accum[lane] * rsqrt(float(head_dim)) * sign);
        }
        """

    private static let hybridPolarWHTValueFusedBlockPartialsSource = """
        constexpr uint threads_per_block = uint(THREADS_PER_BLOCK);
        constexpr uint head_dim = uint(HEAD_DIM);
        uint lane = thread_position_in_threadgroup.x;
        uint group_index = threadgroup_position_in_grid.x;
        uint block_index = group_index % uint(BLOCK_COUNT);
        uint row = group_index / uint(BLOCK_COUNT);
        uint total_rows = uint(BATCH_SIZE) * uint(QUERY_HEADS) * uint(QUERY_LENGTH);
        if (row >= total_rows) {
            return;
        }

        threadgroup float partial[512];
        threadgroup float tile_scores[512];
        threadgroup uint tile_physical_tokens[512];
        threadgroup float query_cache[HEAD_DIM];

        uint logical_length = uint(runtime_logical_length);
        uint ring_offset = uint(runtime_ring_offset);
        uint pinned_prefix_length = uint(runtime_pinned_prefix_length);
        float attention_scale = float(runtime_attention_scale);
        uint q_token = row % uint(QUERY_LENGTH);
        uint q_head = (row / uint(QUERY_LENGTH)) % uint(QUERY_HEADS);
        uint batch = row / (uint(QUERY_LENGTH) * uint(QUERY_HEADS));
        uint repeats = uint(QUERY_HEADS) / uint(KV_HEADS);
        uint kv_head = q_head / repeats;
        uint causal_limit = logical_length - uint(QUERY_LENGTH) + q_token;
        uint block_start = block_index * uint(BLOCK_TOKENS);
        ulong key_seed = tq_make_seed(uint(SEED_3), uint(SEED_2), uint(SEED_1), uint(SEED_0));

        if (DO_CAUSAL && block_start > causal_limit) {
            if (lane == 0u) {
                uint stat_index = ((row * uint(BLOCK_COUNT) + block_index) * 2u);
                partial_stats[stat_index] = -INFINITY;
                partial_stats[stat_index + 1u] = 0.0f;
            }
            if (lane < head_dim) {
                uint out_index = ((row * uint(BLOCK_COUNT) + block_index) * head_dim) + lane;
                partial_out[out_index] = 0.0f;
            }
            return;
        }

        if (lane < head_dim) {
            long q_index =
                long(batch) * q_strides[0]
                + long(q_head) * q_strides[1]
                + long(q_token) * q_strides[2]
                + long(lane) * q_strides[3];
            query_cache[lane] = float(q[q_index]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint logical_token = block_start + lane;
        bool active = lane < uint(BLOCK_TOKENS)
            && logical_token < logical_length
            && (!DO_CAUSAL || logical_token <= causal_limit);
        float scaled_score = -INFINITY;
        uint physical_token = 0u;
        if (active) {
            physical_token = tq_physical_token(
                logical_token, uint(CAPACITY), ring_offset, pinned_prefix_length);
            float score = 0.0f;
            for (uint group = 0u; group < uint(GROUPS_PER_VECTOR); group++) {
                uint group_start = group * uint(GROUP_SIZE);
                uint count = min(uint(GROUP_SIZE), head_dim - group_start);
                thread float query_values[GROUP_SIZE];
                for (uint local = 0u; local < count; local++) {
                    query_values[local] = query_cache[group_start + local];
                }
                score += tq_product_attention_inner_product_group(
                    k_packed, k_signs, k_high_mask, k_residual_signs, k_scales, query_values,
                    batch, kv_head, physical_token, group, key_seed,
                    uint(GROUP_SIZE), uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR),
                    uint(MAG_WORDS_PER_GROUP), uint(BITSET_WORDS_PER_GROUP),
                    uint(KEY_BASE_BITS), uint(KEY_HIGH_BITS), uint(LAYOUT_VERSION),
                    head_dim,
                    tq_high_precision_count(count, uint(HIGH_NUMERATOR), uint(HIGH_DENOMINATOR)));
            }
            scaled_score = score * attention_scale;
        }
        tile_scores[lane] = scaled_score;
        tile_physical_tokens[lane] = physical_token;
        partial[lane] = scaled_score;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
            if (lane < stride) {
                partial[lane] = max(partial[lane], partial[lane + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        float tile_max = partial[0];
        float tile_weight = active ? exp(tile_scores[lane] - tile_max) : 0.0f;
        tile_scores[lane] = tile_weight;
        partial[lane] = tile_weight;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
            if (lane < stride) {
                partial[lane] += partial[lane + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0u) {
            uint stat_index = ((row * uint(BLOCK_COUNT) + block_index) * 2u);
            partial_stats[stat_index] = tile_max;
            partial_stats[stat_index + 1u] = partial[0];
        }

        if (lane < head_dim) {
            float centroid_scale = rsqrt(float(head_dim));
            float dimension_accum = 0.0f;
            for (uint tile_lane = 0u; tile_lane < threads_per_block; tile_lane++) {
                float weight = tile_scores[tile_lane];
                if (weight > 0.0f) {
                    uint vector_index =
                        (batch * uint(KV_HEADS) + kv_head) * uint(CAPACITY)
                        + tile_physical_tokens[tile_lane];
                    uint code = tq_polar_wht_read_index(
                        v_packed_indices,
                        vector_index * uint(PACKED_WORDS_PER_VECTOR),
                        lane,
                        uint(POLAR_WHT_BITS));
                    float centroid = tq_polar_wht_centroid(uint(POLAR_WHT_BITS), code);
                    dimension_accum +=
                        weight * float(v_norms[vector_index]) * centroid * centroid_scale;
                }
            }
            uint out_index = ((row * uint(BLOCK_COUNT) + block_index) * head_dim) + lane;
            partial_out[out_index] = dimension_accum;
        }
        """

    private static let hybridPolarWHTValueGQAFusedBlockPartialsSource = """
        constexpr uint threads_per_block = uint(THREADS_PER_BLOCK);
        constexpr uint head_dim = uint(HEAD_DIM);
        constexpr uint gqa_repeats = uint(GQA_REPEATS);
        constexpr uint repeat_count = 4u;
        uint lane = thread_position_in_threadgroup.x;
        uint group_index = threadgroup_position_in_grid.x;
        uint block_index = group_index % uint(BLOCK_COUNT);
        uint gqa_row = group_index / uint(BLOCK_COUNT);
        uint total_gqa_rows = uint(BATCH_SIZE) * uint(KV_HEADS) * uint(QUERY_LENGTH);
        if (gqa_row >= total_gqa_rows) {
            return;
        }

        threadgroup float partial[4 * THREADS_PER_BLOCK];
        threadgroup float tile_scores[4 * THREADS_PER_BLOCK];
        threadgroup uint tile_physical_tokens[THREADS_PER_BLOCK];
        threadgroup float query_cache[4 * HEAD_DIM];

        uint logical_length = uint(runtime_logical_length);
        uint ring_offset = uint(runtime_ring_offset);
        uint pinned_prefix_length = uint(runtime_pinned_prefix_length);
        float attention_scale = float(runtime_attention_scale);
        uint q_token = gqa_row % uint(QUERY_LENGTH);
        uint kv_head = (gqa_row / uint(QUERY_LENGTH)) % uint(KV_HEADS);
        uint batch = gqa_row / (uint(QUERY_LENGTH) * uint(KV_HEADS));
        uint causal_limit = logical_length - uint(QUERY_LENGTH) + q_token;
        uint block_start = block_index * uint(BLOCK_TOKENS);
        ulong key_seed = tq_make_seed(uint(SEED_3), uint(SEED_2), uint(SEED_1), uint(SEED_0));

        if (DO_CAUSAL && block_start > causal_limit) {
            if (lane == 0u) {
                for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                    uint q_head = kv_head * gqa_repeats + repeat;
                    uint row = ((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH)) + q_token;
                    uint stat_index = ((row * uint(BLOCK_COUNT) + block_index) * 2u);
                    partial_stats[stat_index] = -INFINITY;
                    partial_stats[stat_index + 1u] = 0.0f;
                }
            }
            if (lane < head_dim) {
                for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                    uint q_head = kv_head * gqa_repeats + repeat;
                    uint row = ((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH)) + q_token;
                    uint out_index = ((row * uint(BLOCK_COUNT) + block_index) * head_dim) + lane;
                    partial_out[out_index] = 0.0f;
                }
            }
            return;
        }

        if (lane < head_dim) {
            for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                uint q_head = kv_head * gqa_repeats + repeat;
                long q_index =
                    long(batch) * q_strides[0]
                    + long(q_head) * q_strides[1]
                    + long(q_token) * q_strides[2]
                    + long(lane) * q_strides[3];
                query_cache[repeat * head_dim + lane] = float(q[q_index]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            uint rg_total = repeat_count * uint(GROUPS_PER_VECTOR);
            if (lane < rg_total) {
                uint repeat = lane / uint(GROUPS_PER_VECTOR);
                uint group = lane % uint(GROUPS_PER_VECTOR);
                uint group_start = group * uint(GROUP_SIZE);
                uint count = min(uint(GROUP_SIZE), head_dim - group_start);
                uint storage_group = tq_storage_group_index(
                    batch, kv_head, 0u, group, uint(KV_HEADS), uint(CAPACITY),
                    uint(GROUPS_PER_VECTOR));
                thread float rotated[GROUP_SIZE];
                for (uint local = 0u; local < count; local++) {
                    rotated[local] = query_cache[repeat * head_dim + group_start + local];
                }
                tq_apply_product_rotation(rotated, count, key_seed, storage_group, false);
                for (uint local = 0u; local < count; local++) {
                    query_cache[repeat * head_dim + group_start + local] = rotated[local];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint logical_token = block_start + lane;
        bool active = lane < uint(BLOCK_TOKENS)
            && logical_token < logical_length
            && (!DO_CAUSAL || logical_token <= causal_limit);
        uint physical_token = 0u;
        thread float scaled_scores[4];
        scaled_scores[0] = -INFINITY;
        scaled_scores[1] = -INFINITY;
        scaled_scores[2] = -INFINITY;
        scaled_scores[3] = -INFINITY;
        if (active) {
            physical_token = tq_physical_token(
                logical_token, uint(CAPACITY), ring_offset, pinned_prefix_length);
            scaled_scores[0] = 0.0f;
            scaled_scores[1] = 0.0f;
            scaled_scores[2] = 0.0f;
            scaled_scores[3] = 0.0f;
            for (uint group = 0u; group < uint(GROUPS_PER_VECTOR); group++) {
                uint group_start = group * uint(GROUP_SIZE);
                uint count = min(uint(GROUP_SIZE), head_dim - group_start);
                thread float query_values[4 * GROUP_SIZE];
                thread float quad_scores[4];
                quad_scores[0] = 0.0f;
                quad_scores[1] = 0.0f;
                quad_scores[2] = 0.0f;
                quad_scores[3] = 0.0f;
                for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                    for (uint local = 0u; local < count; local++) {
                        query_values[repeat * uint(GROUP_SIZE) + local] =
                            query_cache[repeat * head_dim + group_start + local];
                    }
                }
                tq_product_attention_inner_product_group_quad(
                    k_packed, k_signs, k_high_mask, k_residual_signs, k_scales, query_values,
                    quad_scores,
                    batch, kv_head, physical_token, group, key_seed,
                    uint(GROUP_SIZE), uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR),
                    uint(MAG_WORDS_PER_GROUP), uint(BITSET_WORDS_PER_GROUP),
                    uint(KEY_BASE_BITS), uint(KEY_HIGH_BITS), uint(LAYOUT_VERSION),
                    head_dim,
                    tq_high_precision_count(count, uint(HIGH_NUMERATOR), uint(HIGH_DENOMINATOR)),
                    true);
                for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                    scaled_scores[repeat] += quad_scores[repeat];
                }
            }
            for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                scaled_scores[repeat] *= attention_scale;
            }
        }
        tile_physical_tokens[lane] = physical_token;
        for (uint repeat = 0u; repeat < repeat_count; repeat++) {
            uint score_base = repeat * threads_per_block;
            tile_scores[score_base + lane] = scaled_scores[repeat];
            partial[score_base + lane] = scaled_scores[repeat];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
            if (lane < stride) {
                for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                    uint score_base = repeat * threads_per_block;
                    partial[score_base + lane] =
                        max(partial[score_base + lane], partial[score_base + lane + stride]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        thread float tile_maxes[4];
        tile_maxes[0] = partial[0 * threads_per_block];
        tile_maxes[1] = partial[1 * threads_per_block];
        tile_maxes[2] = partial[2 * threads_per_block];
        tile_maxes[3] = partial[3 * threads_per_block];
        for (uint repeat = 0u; repeat < repeat_count; repeat++) {
            uint score_base = repeat * threads_per_block;
            float tile_weight = active
                ? exp(tile_scores[score_base + lane] - tile_maxes[repeat])
                : 0.0f;
            tile_scores[score_base + lane] = tile_weight;
            partial[score_base + lane] = tile_weight;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
            if (lane < stride) {
                for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                    uint score_base = repeat * threads_per_block;
                    partial[score_base + lane] += partial[score_base + lane + stride];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0u) {
            for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                uint q_head = kv_head * gqa_repeats + repeat;
                uint row = ((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH)) + q_token;
                uint stat_index = ((row * uint(BLOCK_COUNT) + block_index) * 2u);
                uint score_base = repeat * threads_per_block;
                partial_stats[stat_index] = tile_maxes[repeat];
                partial_stats[stat_index + 1u] = partial[score_base];
            }
        }

        if (lane < head_dim) {
            float centroid_scale = rsqrt(float(head_dim));
            thread float dimension_accum[4];
            dimension_accum[0] = 0.0f;
            dimension_accum[1] = 0.0f;
            dimension_accum[2] = 0.0f;
            dimension_accum[3] = 0.0f;
            for (uint tile_lane = 0u; tile_lane < threads_per_block; tile_lane++) {
                uint vector_index =
                    (batch * uint(KV_HEADS) + kv_head) * uint(CAPACITY)
                    + tile_physical_tokens[tile_lane];
                uint code = tq_polar_wht_read_index(
                    v_packed_indices,
                    vector_index * uint(PACKED_WORDS_PER_VECTOR),
                    lane,
                    uint(POLAR_WHT_BITS));
                float centroid = tq_polar_wht_centroid(uint(POLAR_WHT_BITS), code);
                float value = float(v_norms[vector_index]) * centroid * centroid_scale;
                for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                    float weight = tile_scores[repeat * threads_per_block + tile_lane];
                    dimension_accum[repeat] += weight * value;
                }
            }
            for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                uint q_head = kv_head * gqa_repeats + repeat;
                uint row = ((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH)) + q_token;
                uint out_index = ((row * uint(BLOCK_COUNT) + block_index) * head_dim) + lane;
                partial_out[out_index] = dimension_accum[repeat];
            }
        }
        """

    private static let hybridPolarWHTValueFusedBlockReduceSource = """
        constexpr uint threads_per_block = uint(THREADS_PER_BLOCK);
        constexpr uint head_dim = uint(HEAD_DIM);
        uint lane = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;
        if (row >= uint(ROW_COUNT)) {
            return;
        }

        threadgroup float partial[512];
        threadgroup float tile_scales[512];
        threadgroup float accumulated[HEAD_DIM];

        if (lane < uint(BLOCK_COUNT)) {
            partial[lane] = partial_stats[(row * uint(BLOCK_COUNT) + lane) * 2u];
        } else {
            partial[lane] = -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
            if (lane < stride) {
                partial[lane] = max(partial[lane], partial[lane + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        float row_max = partial[0];
        if (lane < uint(BLOCK_COUNT)) {
            uint stat_index = (row * uint(BLOCK_COUNT) + lane) * 2u;
            float tile_sum = partial_stats[stat_index + 1u];
            float tile_scale = tile_sum > 0.0f ? exp(partial_stats[stat_index] - row_max) : 0.0f;
            tile_scales[lane] = tile_scale;
            partial[lane] = tile_scale * tile_sum;
        } else {
            tile_scales[lane] = 0.0f;
            partial[lane] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
            if (lane < stride) {
                partial[lane] += partial[lane + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        float row_sum = partial[0];
        if (lane < head_dim) {
            float accum = 0.0f;
            for (uint block = 0u; block < uint(BLOCK_COUNT); block++) {
                float tile_scale = tile_scales[block];
                if (tile_scale > 0.0f) {
                    uint partial_index = ((row * uint(BLOCK_COUNT) + block) * head_dim) + lane;
                    accum += tile_scale * partial_out[partial_index];
                }
            }
            accumulated[lane] = accum / max(row_sum, 1.17549435e-38f);
        }
        tq_polar_wht_simdgroup_wht(accumulated, lane, head_dim);

        if (lane < head_dim) {
            ulong value_seed = tq_make_seed(
                uint(VALUE_SEED_3), uint(VALUE_SEED_2),
                uint(VALUE_SEED_1), uint(VALUE_SEED_0));
            float sign = tq_random_sign(value_seed, lane) ? -1.0f : 1.0f;
            out[row * head_dim + lane] = static_cast<OUTPUT_DTYPE>(
                accumulated[lane] * rsqrt(float(head_dim)) * sign);
        }
        """

    private static let hybridAffineK8PolarWHTValueFusedAttentionSource = """
        constexpr uint head_dim = uint(HEAD_DIM);
        constexpr uint threads_per_row = uint(THREADS_PER_ROW);
        uint lane = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;
        uint total_rows = uint(BATCH_SIZE) * uint(QUERY_HEADS) * uint(QUERY_LENGTH);
        if (row >= total_rows) {
            return;
        }

        threadgroup float partial[256];
        threadgroup float tile_scores[256];
        threadgroup uint tile_physical_tokens[256];
        threadgroup float query_cache[HEAD_DIM];
        threadgroup float output_accum[HEAD_DIM];

        uint logical_length = uint(runtime_logical_length);
        uint ring_offset = uint(runtime_ring_offset);
        uint pinned_prefix_length = uint(runtime_pinned_prefix_length);
        float attention_scale = float(runtime_attention_scale);
        uint q_token = row % uint(QUERY_LENGTH);
        uint q_head = (row / uint(QUERY_LENGTH)) % uint(QUERY_HEADS);
        uint batch = row / (uint(QUERY_LENGTH) * uint(QUERY_HEADS));
        uint repeats = uint(QUERY_HEADS) / uint(KV_HEADS);
        uint kv_head = q_head / repeats;
        uint causal_limit = logical_length - uint(QUERY_LENGTH) + q_token;

        float row_max = -INFINITY;
        float row_sum = 0.0f;
        if (lane < head_dim) {
            long q_index =
                long(batch) * q_strides[0]
                + long(q_head) * q_strides[1]
                + long(q_token) * q_strides[2]
                + long(lane) * q_strides[3];
            query_cache[lane] = float(q[q_index]);
            output_accum[lane] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint tile_start = 0u; tile_start < logical_length; tile_start += threads_per_row) {
            uint logical_token = tile_start + lane;
            bool active = lane < threads_per_row
                && logical_token < logical_length
                && (!DO_CAUSAL || logical_token <= causal_limit);
            float scaled_score = -INFINITY;
            uint physical_token = 0u;
            if (active) {
                physical_token = tq_physical_token(
                    logical_token, uint(CAPACITY), ring_offset, pinned_prefix_length);
                uint key_vector =
                    (batch * uint(KV_HEADS) + kv_head) * uint(KEY_CAPACITY) + logical_token;
                uint key_packed_base = key_vector * uint(KEY_PACKED_WORDS_PER_VECTOR);
                uint key_scale_base = key_vector * uint(KEY_GROUPS_PER_VECTOR);
                float score = 0.0f;
                for (uint word = 0u; word < uint(KEY_PACKED_WORDS_PER_VECTOR); word++) {
                    uint packed = uint(k_packed[key_packed_base + word]);
                    uint dim = word << 2;
                    uint group = dim / uint(KEY_GROUP_SIZE);
                    float key_scale = float(k_scales[key_scale_base + group]);
                    float key_bias = float(k_biases[key_scale_base + group]);
                    score += query_cache[dim] * (key_scale * float(packed & 0xffu) + key_bias);
                    score += query_cache[dim + 1u]
                        * (key_scale * float((packed >> 8u) & 0xffu) + key_bias);
                    score += query_cache[dim + 2u]
                        * (key_scale * float((packed >> 16u) & 0xffu) + key_bias);
                    score += query_cache[dim + 3u]
                        * (key_scale * float((packed >> 24u) & 0xffu) + key_bias);
                }
                scaled_score = score * attention_scale;
            }
            tile_scores[lane] = scaled_score;
            tile_physical_tokens[lane] = physical_token;
            partial[lane] = scaled_score;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = threads_per_row >> 1; stride > 0u; stride >>= 1) {
                if (lane < stride) {
                    partial[lane] = max(partial[lane], partial[lane + stride]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            float tile_max = partial[0];
            float new_row_max = max(row_max, tile_max);
            float old_scale = row_sum > 0.0f ? exp(row_max - new_row_max) : 0.0f;
            if (lane < head_dim) {
                output_accum[lane] *= old_scale;
            }

            float weight = active ? exp(tile_scores[lane] - new_row_max) : 0.0f;
            tile_scores[lane] = weight;
            partial[lane] = weight;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = threads_per_row >> 1; stride > 0u; stride >>= 1) {
                if (lane < stride) {
                    partial[lane] += partial[lane + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            float next_row_sum = row_sum * old_scale + partial[0];
            if (lane < head_dim) {
                float centroid_scale = rsqrt(float(head_dim));
                float dimension_accum = output_accum[lane];
                for (uint tile_lane = 0u; tile_lane < threads_per_row; tile_lane++) {
                    float tile_weight = tile_scores[tile_lane];
                    if (tile_weight > 0.0f) {
                        uint vector_index =
                            (batch * uint(KV_HEADS) + kv_head) * uint(CAPACITY)
                            + tile_physical_tokens[tile_lane];
                        uint code = tq_polar_wht_read_index(
                            v_packed_indices,
                            vector_index * uint(PACKED_WORDS_PER_VECTOR),
                            lane,
                            uint(POLAR_WHT_BITS));
                        float centroid = tq_polar_wht_centroid(uint(POLAR_WHT_BITS), code);
                        dimension_accum +=
                            tile_weight * float(v_norms[vector_index]) * centroid * centroid_scale;
                    }
                }
                output_accum[lane] = dimension_accum;
            }
            row_max = new_row_max;
            row_sum = next_row_sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane < head_dim) {
            output_accum[lane] *= 1.0f / max(row_sum, 1.17549435e-38f);
        }
        tq_polar_wht_simdgroup_wht(output_accum, lane, head_dim);

        if (lane < head_dim) {
            ulong value_seed = tq_make_seed(
                uint(VALUE_SEED_3), uint(VALUE_SEED_2),
                uint(VALUE_SEED_1), uint(VALUE_SEED_0));
            float sign = tq_random_sign(value_seed, lane) ? -1.0f : 1.0f;
            out[row * head_dim + lane] = static_cast<OUTPUT_DTYPE>(
                output_accum[lane] * rsqrt(float(head_dim)) * sign);
        }
        """

    private static let segmentedHybridAffineK8PolarWHTValueFusedAttentionSource = """
        constexpr uint head_dim = uint(HEAD_DIM);
        constexpr uint threads_per_row = uint(THREADS_PER_ROW);
        uint lane = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;
        uint total_rows = uint(BATCH_SIZE) * uint(QUERY_HEADS) * uint(QUERY_LENGTH);
        if (row >= total_rows) {
            return;
        }

        threadgroup float partial[256];
        threadgroup float tile_scores[256];
        threadgroup uint tile_physical_tokens[256];
        threadgroup float query_cache[HEAD_DIM];
        threadgroup float output_accum[HEAD_DIM];

        float attention_scale = float(runtime_attention_scale);
        uint q_token = row % uint(QUERY_LENGTH);
        uint q_head = (row / uint(QUERY_LENGTH)) % uint(QUERY_HEADS);
        uint batch = row / (uint(QUERY_LENGTH) * uint(QUERY_HEADS));
        uint repeats = uint(QUERY_HEADS) / uint(KV_HEADS);
        uint kv_head = q_head / repeats;

        float row_max = -INFINITY;
        float row_sum = 0.0f;
        if (lane < head_dim) {
            long q_index =
                long(batch) * q_strides[0]
                + long(q_head) * q_strides[1]
                + long(q_token) * q_strides[2]
                + long(lane) * q_strides[3];
            query_cache[lane] = float(q[q_index]);
            output_accum[lane] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint segment = 0u; segment < 2u; segment++) {
            bool is_tail = segment == 1u;
            uint logical_length =
                is_tail ? uint(runtime_tail_logical_length)
                : uint(runtime_base_logical_length);
            uint capacity = is_tail ? uint(TAIL_CAPACITY) : uint(BASE_CAPACITY);
            uint key_capacity =
                is_tail ? uint(TAIL_KEY_CAPACITY) : uint(BASE_KEY_CAPACITY);
            uint ring_offset =
                is_tail ? uint(runtime_tail_ring_offset) : uint(runtime_base_ring_offset);
            uint pinned_prefix_length =
                is_tail
                ? uint(runtime_tail_pinned_prefix_length)
                : uint(runtime_base_pinned_prefix_length);
            uint causal_limit = logical_length - uint(QUERY_LENGTH) + q_token;

            for (uint tile_start = 0u; tile_start < logical_length; tile_start += threads_per_row) {
                uint logical_token = tile_start + lane;
                bool active = lane < threads_per_row
                    && logical_token < logical_length
                    && (!DO_CAUSAL || logical_token <= causal_limit);
                float scaled_score = -INFINITY;
                uint physical_token = 0u;
                if (active) {
                    physical_token = tq_physical_token(
                        logical_token, capacity, ring_offset, pinned_prefix_length);
                    uint key_vector =
                        (batch * uint(KV_HEADS) + kv_head) * key_capacity + logical_token;
                    uint key_packed_base = key_vector * uint(KEY_PACKED_WORDS_PER_VECTOR);
                    uint key_scale_base = key_vector * uint(KEY_GROUPS_PER_VECTOR);
                    float score = 0.0f;
                    for (uint word = 0u; word < uint(KEY_PACKED_WORDS_PER_VECTOR); word++) {
                        uint packed = is_tail
                            ? uint(tail_k_packed[key_packed_base + word])
                            : uint(base_k_packed[key_packed_base + word]);
                        uint dim = word << 2;
                        uint group = dim / uint(KEY_GROUP_SIZE);
                        float key_scale = is_tail
                            ? float(tail_k_scales[key_scale_base + group])
                            : float(base_k_scales[key_scale_base + group]);
                        float key_bias = is_tail
                            ? float(tail_k_biases[key_scale_base + group])
                            : float(base_k_biases[key_scale_base + group]);
                        score += query_cache[dim] * (key_scale * float(packed & 0xffu) + key_bias);
                        score += query_cache[dim + 1u]
                            * (key_scale * float((packed >> 8u) & 0xffu) + key_bias);
                        score += query_cache[dim + 2u]
                            * (key_scale * float((packed >> 16u) & 0xffu) + key_bias);
                        score += query_cache[dim + 3u]
                            * (key_scale * float((packed >> 24u) & 0xffu) + key_bias);
                    }
                    scaled_score = score * attention_scale;
                }
                tile_scores[lane] = scaled_score;
                tile_physical_tokens[lane] = physical_token;
                partial[lane] = scaled_score;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint stride = threads_per_row >> 1; stride > 0u; stride >>= 1) {
                    if (lane < stride) {
                        partial[lane] = max(partial[lane], partial[lane + stride]);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }

                float tile_max = partial[0];
                float new_row_max = max(row_max, tile_max);
                float old_scale = row_sum > 0.0f ? exp(row_max - new_row_max) : 0.0f;
                if (lane < head_dim) {
                    output_accum[lane] *= old_scale;
                }

                float weight = active ? exp(tile_scores[lane] - new_row_max) : 0.0f;
                tile_scores[lane] = weight;
                partial[lane] = weight;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint stride = threads_per_row >> 1; stride > 0u; stride >>= 1) {
                    if (lane < stride) {
                        partial[lane] += partial[lane + stride];
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }

                float next_row_sum = row_sum * old_scale + partial[0];
                if (lane < head_dim) {
                    float centroid_scale = rsqrt(float(head_dim));
                    float dimension_accum = output_accum[lane];
                    for (uint tile_lane = 0u; tile_lane < threads_per_row; tile_lane++) {
                        float tile_weight = tile_scores[tile_lane];
                        if (tile_weight > 0.0f) {
                            uint vector_index =
                                (batch * uint(KV_HEADS) + kv_head) * capacity
                                + tile_physical_tokens[tile_lane];
                            uint code;
                            float value_norm;
                            if (is_tail) {
                                code = tq_polar_wht_read_index(
                                    tail_v_packed_indices,
                                    vector_index * uint(PACKED_WORDS_PER_VECTOR),
                                    lane,
                                    uint(POLAR_WHT_BITS));
                                value_norm = float(tail_v_norms[vector_index]);
                            } else {
                                code = tq_polar_wht_read_index(
                                    base_v_packed_indices,
                                    vector_index * uint(PACKED_WORDS_PER_VECTOR),
                                    lane,
                                    uint(POLAR_WHT_BITS));
                                value_norm = float(base_v_norms[vector_index]);
                            }
                            float centroid = tq_polar_wht_centroid(uint(POLAR_WHT_BITS), code);
                            dimension_accum +=
                                tile_weight * value_norm * centroid * centroid_scale;
                        }
                    }
                    output_accum[lane] = dimension_accum;
                }
                row_max = new_row_max;
                row_sum = next_row_sum;
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        if (lane < head_dim) {
            output_accum[lane] *= 1.0f / max(row_sum, 1.17549435e-38f);
        }
        tq_polar_wht_simdgroup_wht(output_accum, lane, head_dim);

        if (lane < head_dim) {
            ulong value_seed = tq_make_seed(
                uint(VALUE_SEED_3), uint(VALUE_SEED_2),
                uint(VALUE_SEED_1), uint(VALUE_SEED_0));
            float sign = tq_random_sign(value_seed, lane) ? -1.0f : 1.0f;
            out[row * head_dim + lane] = static_cast<OUTPUT_DTYPE>(
                output_accum[lane] * rsqrt(float(head_dim)) * sign);
        }
        """

    private static let hybridAffineK8DecodedValueFusedAttentionSource = """
        constexpr uint head_dim = uint(HEAD_DIM);
        constexpr uint threads_per_row = uint(THREADS_PER_ROW);
        uint lane = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;
        uint total_rows = uint(BATCH_SIZE) * uint(QUERY_HEADS) * uint(QUERY_LENGTH);
        if (row >= total_rows) {
            return;
        }

        threadgroup float partial[256];
        threadgroup float tile_scores[256];
        threadgroup float query_cache[HEAD_DIM];
        threadgroup float output_accum[HEAD_DIM];

        uint logical_length = uint(runtime_logical_length);
        uint ring_offset = uint(runtime_ring_offset);
        uint pinned_prefix_length = uint(runtime_pinned_prefix_length);
        float attention_scale = float(runtime_attention_scale);
        uint q_token = row % uint(QUERY_LENGTH);
        uint q_head = (row / uint(QUERY_LENGTH)) % uint(QUERY_HEADS);
        uint batch = row / (uint(QUERY_LENGTH) * uint(QUERY_HEADS));
        uint repeats = uint(QUERY_HEADS) / uint(KV_HEADS);
        uint kv_head = q_head / repeats;
        uint causal_limit = logical_length - uint(QUERY_LENGTH) + q_token;

        float row_max = -INFINITY;
        float row_sum = 0.0f;
        if (lane < head_dim) {
            long q_index =
                long(batch) * q_strides[0]
                + long(q_head) * q_strides[1]
                + long(q_token) * q_strides[2]
                + long(lane) * q_strides[3];
            query_cache[lane] = float(q[q_index]);
            output_accum[lane] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint tile_start = 0u; tile_start < logical_length; tile_start += threads_per_row) {
            uint logical_token = tile_start + lane;
            bool active = lane < threads_per_row
                && logical_token < logical_length
                && (!DO_CAUSAL || logical_token <= causal_limit);
            float scaled_score = -INFINITY;
            if (active) {
                uint key_vector =
                    (batch * uint(KV_HEADS) + kv_head) * uint(KEY_CAPACITY) + logical_token;
                uint key_packed_base = key_vector * uint(KEY_PACKED_WORDS_PER_VECTOR);
                uint key_scale_base = key_vector * uint(KEY_GROUPS_PER_VECTOR);
                float score = 0.0f;
                for (uint word = 0u; word < uint(KEY_PACKED_WORDS_PER_VECTOR); word++) {
                    uint packed = uint(k_packed[key_packed_base + word]);
                    uint dim = word << 2;
                    uint group = dim / uint(KEY_GROUP_SIZE);
                    float key_scale = float(k_scales[key_scale_base + group]);
                    float key_bias = float(k_biases[key_scale_base + group]);
                    score += query_cache[dim] * (key_scale * float(packed & 0xffu) + key_bias);
                    score += query_cache[dim + 1u]
                        * (key_scale * float((packed >> 8u) & 0xffu) + key_bias);
                    score += query_cache[dim + 2u]
                        * (key_scale * float((packed >> 16u) & 0xffu) + key_bias);
                    score += query_cache[dim + 3u]
                        * (key_scale * float((packed >> 24u) & 0xffu) + key_bias);
                }
                scaled_score = score * attention_scale;
            }
            tile_scores[lane] = scaled_score;
            partial[lane] = scaled_score;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = threads_per_row >> 1; stride > 0u; stride >>= 1) {
                if (lane < stride) {
                    partial[lane] = max(partial[lane], partial[lane + stride]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            float tile_max = partial[0];
            float new_row_max = max(row_max, tile_max);
            float old_scale = row_sum > 0.0f ? exp(row_max - new_row_max) : 0.0f;
            if (lane < head_dim) {
                output_accum[lane] *= old_scale;
            }

            float weight = active ? exp(tile_scores[lane] - new_row_max) : 0.0f;
            tile_scores[lane] = weight;
            partial[lane] = weight;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = threads_per_row >> 1; stride > 0u; stride >>= 1) {
                if (lane < stride) {
                    partial[lane] += partial[lane + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            float next_row_sum = row_sum * old_scale + partial[0];
            if (lane < head_dim) {
                float dimension_accum = output_accum[lane];
                for (uint tile_lane = 0u; tile_lane < threads_per_row; tile_lane++) {
                    float tile_weight = tile_scores[tile_lane];
                    if (tile_weight > 0.0f) {
                        uint value_token = tile_start + tile_lane;
                        uint physical_token = tq_physical_token(
                            value_token, uint(CAPACITY), ring_offset, pinned_prefix_length);
                        long value_index =
                            long(batch) * v_decoded_strides[0]
                            + long(kv_head) * v_decoded_strides[1]
                            + long(physical_token) * v_decoded_strides[2]
                            + long(lane) * v_decoded_strides[3];
                        dimension_accum += tile_weight * float(v_decoded[value_index]);
                    }
                }
                output_accum[lane] = dimension_accum;
            }
            row_max = new_row_max;
            row_sum = next_row_sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane < head_dim) {
            out[row * head_dim + lane] = static_cast<OUTPUT_DTYPE>(
                output_accum[lane] / max(row_sum, 1.17549435e-38f));
        }
        """

    private static let hybridAffineK8DecodedValueFusedBlockPartialsSource = """
        constexpr uint threads_per_block = uint(THREADS_PER_BLOCK);
        constexpr uint head_dim = uint(HEAD_DIM);
        uint lane = thread_position_in_threadgroup.x;
        uint group_index = threadgroup_position_in_grid.x;
        uint block_index = group_index % uint(BLOCK_COUNT);
        uint row = group_index / uint(BLOCK_COUNT);
        uint total_rows = uint(BATCH_SIZE) * uint(QUERY_HEADS) * uint(QUERY_LENGTH);
        if (row >= total_rows) {
            return;
        }

        threadgroup float partial[512];
        threadgroup float tile_scores[512];
        threadgroup float query_cache[HEAD_DIM];

        uint logical_length = uint(runtime_logical_length);
        uint ring_offset = uint(runtime_ring_offset);
        uint pinned_prefix_length = uint(runtime_pinned_prefix_length);
        float attention_scale = float(runtime_attention_scale);
        uint q_token = row % uint(QUERY_LENGTH);
        uint q_head = (row / uint(QUERY_LENGTH)) % uint(QUERY_HEADS);
        uint batch = row / (uint(QUERY_LENGTH) * uint(QUERY_HEADS));
        uint repeats = uint(QUERY_HEADS) / uint(KV_HEADS);
        uint kv_head = q_head / repeats;
        uint causal_limit = logical_length - uint(QUERY_LENGTH) + q_token;
        uint block_start = block_index * uint(BLOCK_TOKENS);

        if (DO_CAUSAL && block_start > causal_limit) {
            if (lane == 0u) {
                uint stat_index = ((row * uint(BLOCK_COUNT) + block_index) * 2u);
                partial_stats[stat_index] = -INFINITY;
                partial_stats[stat_index + 1u] = 0.0f;
            }
            if (lane < head_dim) {
                uint out_index = ((row * uint(BLOCK_COUNT) + block_index) * head_dim) + lane;
                partial_out[out_index] = 0.0f;
            }
            return;
        }

        if (lane < head_dim) {
            long q_index =
                long(batch) * q_strides[0]
                + long(q_head) * q_strides[1]
                + long(q_token) * q_strides[2]
                + long(lane) * q_strides[3];
            query_cache[lane] = float(q[q_index]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint logical_token = block_start + lane;
        bool active = lane < uint(BLOCK_TOKENS)
            && logical_token < logical_length
            && (!DO_CAUSAL || logical_token <= causal_limit);
        float scaled_score = -INFINITY;
        if (active) {
            uint key_vector =
                (batch * uint(KV_HEADS) + kv_head) * uint(KEY_CAPACITY) + logical_token;
            uint key_packed_base = key_vector * uint(KEY_PACKED_WORDS_PER_VECTOR);
            uint key_scale_base = key_vector * uint(KEY_GROUPS_PER_VECTOR);
            float score = 0.0f;
            for (uint word = 0u; word < uint(KEY_PACKED_WORDS_PER_VECTOR); word++) {
                uint packed = uint(k_packed[key_packed_base + word]);
                uint dim = word << 2;
                uint group = dim / uint(KEY_GROUP_SIZE);
                float key_scale = float(k_scales[key_scale_base + group]);
                float key_bias = float(k_biases[key_scale_base + group]);
                score += query_cache[dim] * (key_scale * float(packed & 0xffu) + key_bias);
                score += query_cache[dim + 1u]
                    * (key_scale * float((packed >> 8u) & 0xffu) + key_bias);
                score += query_cache[dim + 2u]
                    * (key_scale * float((packed >> 16u) & 0xffu) + key_bias);
                score += query_cache[dim + 3u]
                    * (key_scale * float((packed >> 24u) & 0xffu) + key_bias);
            }
            scaled_score = score * attention_scale;
        }
        tile_scores[lane] = scaled_score;
        partial[lane] = scaled_score;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
            if (lane < stride) {
                partial[lane] = max(partial[lane], partial[lane + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        float tile_max = partial[0];
        float tile_weight = active ? exp(tile_scores[lane] - tile_max) : 0.0f;
        tile_scores[lane] = tile_weight;
        partial[lane] = tile_weight;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
            if (lane < stride) {
                partial[lane] += partial[lane + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0u) {
            uint stat_index = ((row * uint(BLOCK_COUNT) + block_index) * 2u);
            partial_stats[stat_index] = tile_max;
            partial_stats[stat_index + 1u] = partial[0];
        }

        if (lane < head_dim) {
            float dimension_accum = 0.0f;
            for (uint tile_lane = 0u; tile_lane < threads_per_block; tile_lane++) {
                float weight = tile_scores[tile_lane];
                if (weight > 0.0f) {
                    uint value_token = block_start + tile_lane;
                    uint physical_token = tq_physical_token(
                        value_token, uint(CAPACITY), ring_offset, pinned_prefix_length);
                    long value_index =
                        long(batch) * v_decoded_strides[0]
                        + long(kv_head) * v_decoded_strides[1]
                        + long(physical_token) * v_decoded_strides[2]
                        + long(lane) * v_decoded_strides[3];
                    dimension_accum += weight * float(v_decoded[value_index]);
                }
            }
            uint out_index = ((row * uint(BLOCK_COUNT) + block_index) * head_dim) + lane;
            partial_out[out_index] = dimension_accum;
        }
        """

    private static let hybridDecodedValueFusedBlockReduceSource = """
        constexpr uint threads_per_block = uint(THREADS_PER_BLOCK);
        constexpr uint head_dim = uint(HEAD_DIM);
        uint lane = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;
        if (row >= uint(ROW_COUNT)) {
            return;
        }

        threadgroup float partial[512];
        threadgroup float tile_scales[512];

        if (lane < uint(BLOCK_COUNT)) {
            partial[lane] = partial_stats[(row * uint(BLOCK_COUNT) + lane) * 2u];
        } else {
            partial[lane] = -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
            if (lane < stride) {
                partial[lane] = max(partial[lane], partial[lane + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        float row_max = partial[0];
        if (lane < uint(BLOCK_COUNT)) {
            uint stat_index = (row * uint(BLOCK_COUNT) + lane) * 2u;
            float tile_sum = partial_stats[stat_index + 1u];
            float tile_scale = tile_sum > 0.0f ? exp(partial_stats[stat_index] - row_max) : 0.0f;
            tile_scales[lane] = tile_scale;
            partial[lane] = tile_scale * tile_sum;
        } else {
            tile_scales[lane] = 0.0f;
            partial[lane] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
            if (lane < stride) {
                partial[lane] += partial[lane + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        float row_sum = partial[0];
        if (lane < head_dim) {
            float accum = 0.0f;
            for (uint block = 0u; block < uint(BLOCK_COUNT); block++) {
                float tile_scale = tile_scales[block];
                if (tile_scale > 0.0f) {
                    uint partial_index = ((row * uint(BLOCK_COUNT) + block) * head_dim) + lane;
                    accum += tile_scale * partial_out[partial_index];
                }
            }
            out[row * head_dim + lane] = static_cast<OUTPUT_DTYPE>(
                accum / max(row_sum, 1.17549435e-38f));
        }
        """

    private static let hybridAffineK8PolarWHTValueFusedBlockPartialsSource = """
        constexpr uint threads_per_block = uint(THREADS_PER_BLOCK);
        constexpr uint head_dim = uint(HEAD_DIM);
        uint lane = thread_position_in_threadgroup.x;
        uint group_index = threadgroup_position_in_grid.x;
        uint block_index = group_index % uint(BLOCK_COUNT);
        uint row = group_index / uint(BLOCK_COUNT);
        uint total_rows = uint(BATCH_SIZE) * uint(QUERY_HEADS) * uint(QUERY_LENGTH);
        if (row >= total_rows) {
            return;
        }

        threadgroup float partial[512];
        threadgroup float tile_scores[512];
        threadgroup uint tile_physical_tokens[512];
        threadgroup float query_cache[HEAD_DIM];

        uint logical_length = uint(runtime_logical_length);
        uint ring_offset = uint(runtime_ring_offset);
        uint pinned_prefix_length = uint(runtime_pinned_prefix_length);
        float attention_scale = float(runtime_attention_scale);
        uint q_token = row % uint(QUERY_LENGTH);
        uint q_head = (row / uint(QUERY_LENGTH)) % uint(QUERY_HEADS);
        uint batch = row / (uint(QUERY_LENGTH) * uint(QUERY_HEADS));
        uint repeats = uint(QUERY_HEADS) / uint(KV_HEADS);
        uint kv_head = q_head / repeats;
        uint causal_limit = logical_length - uint(QUERY_LENGTH) + q_token;
        uint block_start = block_index * uint(BLOCK_TOKENS);

        if (DO_CAUSAL && block_start > causal_limit) {
            if (lane == 0u) {
                uint stat_index = ((row * uint(BLOCK_COUNT) + block_index) * 2u);
                partial_stats[stat_index] = -INFINITY;
                partial_stats[stat_index + 1u] = 0.0f;
            }
            if (lane < head_dim) {
                uint out_index = ((row * uint(BLOCK_COUNT) + block_index) * head_dim) + lane;
                partial_out[out_index] = 0.0f;
            }
            return;
        }

        if (lane < head_dim) {
            long q_index =
                long(batch) * q_strides[0]
                + long(q_head) * q_strides[1]
                + long(q_token) * q_strides[2]
                + long(lane) * q_strides[3];
            query_cache[lane] = float(q[q_index]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint logical_token = block_start + lane;
        bool active = lane < uint(BLOCK_TOKENS)
            && logical_token < logical_length
            && (!DO_CAUSAL || logical_token <= causal_limit);
        float scaled_score = -INFINITY;
        uint physical_token = 0u;
        if (active) {
            physical_token = tq_physical_token(
                logical_token, uint(CAPACITY), ring_offset, pinned_prefix_length);
            uint key_vector =
                (batch * uint(KV_HEADS) + kv_head) * uint(KEY_CAPACITY) + logical_token;
            uint key_packed_base = key_vector * uint(KEY_PACKED_WORDS_PER_VECTOR);
            uint key_scale_base = key_vector * uint(KEY_GROUPS_PER_VECTOR);
            float score = 0.0f;
            for (uint word = 0u; word < uint(KEY_PACKED_WORDS_PER_VECTOR); word++) {
                uint packed = uint(k_packed[key_packed_base + word]);
                uint dim = word << 2;
                uint group = dim / uint(KEY_GROUP_SIZE);
                float key_scale = float(k_scales[key_scale_base + group]);
                float key_bias = float(k_biases[key_scale_base + group]);
                score += query_cache[dim] * (key_scale * float(packed & 0xffu) + key_bias);
                score += query_cache[dim + 1u]
                    * (key_scale * float((packed >> 8u) & 0xffu) + key_bias);
                score += query_cache[dim + 2u]
                    * (key_scale * float((packed >> 16u) & 0xffu) + key_bias);
                score += query_cache[dim + 3u]
                    * (key_scale * float((packed >> 24u) & 0xffu) + key_bias);
            }
            scaled_score = score * attention_scale;
        }
        tile_scores[lane] = scaled_score;
        tile_physical_tokens[lane] = physical_token;
        partial[lane] = scaled_score;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
            if (lane < stride) {
                partial[lane] = max(partial[lane], partial[lane + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        float tile_max = partial[0];
        float tile_weight = active ? exp(tile_scores[lane] - tile_max) : 0.0f;
        tile_scores[lane] = tile_weight;
        partial[lane] = tile_weight;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
            if (lane < stride) {
                partial[lane] += partial[lane + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0u) {
            uint stat_index = ((row * uint(BLOCK_COUNT) + block_index) * 2u);
            partial_stats[stat_index] = tile_max;
            partial_stats[stat_index + 1u] = partial[0];
        }

        if (lane < head_dim) {
            float centroid_scale = rsqrt(float(head_dim));
            float dimension_accum = 0.0f;
            for (uint tile_lane = 0u; tile_lane < threads_per_block; tile_lane++) {
                float weight = tile_scores[tile_lane];
                if (weight > 0.0f) {
                    uint vector_index =
                        (batch * uint(KV_HEADS) + kv_head) * uint(CAPACITY)
                        + tile_physical_tokens[tile_lane];
                    uint code = tq_polar_wht_read_index(
                        v_packed_indices,
                        vector_index * uint(PACKED_WORDS_PER_VECTOR),
                        lane,
                        uint(POLAR_WHT_BITS));
                    float centroid = tq_polar_wht_centroid(uint(POLAR_WHT_BITS), code);
                    dimension_accum +=
                        weight * float(v_norms[vector_index]) * centroid * centroid_scale;
                }
            }
            uint out_index = ((row * uint(BLOCK_COUNT) + block_index) * head_dim) + lane;
            partial_out[out_index] = dimension_accum;
        }
        """

    private static let hybridAffineK8PolarWHTValueGQAFusedBlockPartialsSource = """
        constexpr uint threads_per_block = uint(THREADS_PER_BLOCK);
        constexpr uint head_dim = uint(HEAD_DIM);
        constexpr uint gqa_repeats = uint(GQA_REPEATS);
        constexpr uint repeat_count = 2u;
        uint lane = thread_position_in_threadgroup.x;
        uint group_index = threadgroup_position_in_grid.x;
        uint block_index = group_index % uint(BLOCK_COUNT);
        uint gqa_row = group_index / uint(BLOCK_COUNT);
        uint total_gqa_rows = uint(BATCH_SIZE) * uint(KV_HEADS) * uint(QUERY_LENGTH);
        if (gqa_row >= total_gqa_rows) {
            return;
        }

        threadgroup float partial[2 * THREADS_PER_BLOCK];
        threadgroup float tile_scores[2 * THREADS_PER_BLOCK];
        threadgroup uint tile_physical_tokens[THREADS_PER_BLOCK];
        threadgroup float query_cache[2 * HEAD_DIM];

        uint logical_length = uint(runtime_logical_length);
        uint ring_offset = uint(runtime_ring_offset);
        uint pinned_prefix_length = uint(runtime_pinned_prefix_length);
        float attention_scale = float(runtime_attention_scale);
        uint q_token = gqa_row % uint(QUERY_LENGTH);
        uint kv_head = (gqa_row / uint(QUERY_LENGTH)) % uint(KV_HEADS);
        uint batch = gqa_row / (uint(QUERY_LENGTH) * uint(KV_HEADS));
        uint causal_limit = logical_length - uint(QUERY_LENGTH) + q_token;
        uint block_start = block_index * uint(BLOCK_TOKENS);

        if (DO_CAUSAL && block_start > causal_limit) {
            if (lane == 0u) {
                for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                    uint q_head = kv_head * gqa_repeats + repeat;
                    uint row = ((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH)) + q_token;
                    uint stat_index = ((row * uint(BLOCK_COUNT) + block_index) * 2u);
                    partial_stats[stat_index] = -INFINITY;
                    partial_stats[stat_index + 1u] = 0.0f;
                }
            }
            if (lane < head_dim) {
                for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                    uint q_head = kv_head * gqa_repeats + repeat;
                    uint row = ((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH)) + q_token;
                    uint out_index = ((row * uint(BLOCK_COUNT) + block_index) * head_dim) + lane;
                    partial_out[out_index] = 0.0f;
                }
            }
            return;
        }

        if (lane < head_dim) {
            for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                uint q_head = kv_head * gqa_repeats + repeat;
                long q_index =
                    long(batch) * q_strides[0]
                    + long(q_head) * q_strides[1]
                    + long(q_token) * q_strides[2]
                    + long(lane) * q_strides[3];
                query_cache[repeat * head_dim + lane] = float(q[q_index]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint logical_token = block_start + lane;
        bool active = lane < uint(BLOCK_TOKENS)
            && logical_token < logical_length
            && (!DO_CAUSAL || logical_token <= causal_limit);
        uint physical_token = 0u;
        thread float scaled_scores[2];
        scaled_scores[0] = -INFINITY;
        scaled_scores[1] = -INFINITY;
        if (active) {
            physical_token = tq_physical_token(
                logical_token, uint(CAPACITY), ring_offset, pinned_prefix_length);
            uint key_vector =
                (batch * uint(KV_HEADS) + kv_head) * uint(KEY_CAPACITY) + logical_token;
            uint key_packed_base = key_vector * uint(KEY_PACKED_WORDS_PER_VECTOR);
            uint key_scale_base = key_vector * uint(KEY_GROUPS_PER_VECTOR);
            scaled_scores[0] = 0.0f;
            scaled_scores[1] = 0.0f;
            for (uint word = 0u; word < uint(KEY_PACKED_WORDS_PER_VECTOR); word++) {
                uint packed = uint(k_packed[key_packed_base + word]);
                uint dim = word << 2;
                uint group = dim / uint(KEY_GROUP_SIZE);
                float key_scale = float(k_scales[key_scale_base + group]);
                float key_bias = float(k_biases[key_scale_base + group]);
                float key_value0 = key_scale * float(packed & 0xffu) + key_bias;
                float key_value1 = key_scale * float((packed >> 8u) & 0xffu) + key_bias;
                float key_value2 = key_scale * float((packed >> 16u) & 0xffu) + key_bias;
                float key_value3 = key_scale * float((packed >> 24u) & 0xffu) + key_bias;
                scaled_scores[0] += query_cache[0 * head_dim + dim] * key_value0;
                scaled_scores[0] += query_cache[0 * head_dim + dim + 1u] * key_value1;
                scaled_scores[0] += query_cache[0 * head_dim + dim + 2u] * key_value2;
                scaled_scores[0] += query_cache[0 * head_dim + dim + 3u] * key_value3;
                scaled_scores[1] += query_cache[1 * head_dim + dim] * key_value0;
                scaled_scores[1] += query_cache[1 * head_dim + dim + 1u] * key_value1;
                scaled_scores[1] += query_cache[1 * head_dim + dim + 2u] * key_value2;
                scaled_scores[1] += query_cache[1 * head_dim + dim + 3u] * key_value3;
            }
            scaled_scores[0] *= attention_scale;
            scaled_scores[1] *= attention_scale;
        }
        tile_physical_tokens[lane] = physical_token;
        for (uint repeat = 0u; repeat < repeat_count; repeat++) {
            uint score_base = repeat * threads_per_block;
            tile_scores[score_base + lane] = scaled_scores[repeat];
            partial[score_base + lane] = scaled_scores[repeat];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
            if (lane < stride) {
                for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                    uint score_base = repeat * threads_per_block;
                    partial[score_base + lane] =
                        max(partial[score_base + lane], partial[score_base + lane + stride]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        thread float tile_maxes[2];
        tile_maxes[0] = partial[0 * threads_per_block];
        tile_maxes[1] = partial[1 * threads_per_block];
        for (uint repeat = 0u; repeat < repeat_count; repeat++) {
            uint score_base = repeat * threads_per_block;
            float tile_weight = active
                ? exp(tile_scores[score_base + lane] - tile_maxes[repeat])
                : 0.0f;
            tile_scores[score_base + lane] = tile_weight;
            partial[score_base + lane] = tile_weight;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
            if (lane < stride) {
                for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                    uint score_base = repeat * threads_per_block;
                    partial[score_base + lane] += partial[score_base + lane + stride];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0u) {
            for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                uint q_head = kv_head * gqa_repeats + repeat;
                uint row = ((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH)) + q_token;
                uint stat_index = ((row * uint(BLOCK_COUNT) + block_index) * 2u);
                uint score_base = repeat * threads_per_block;
                partial_stats[stat_index] = tile_maxes[repeat];
                partial_stats[stat_index + 1u] = partial[score_base];
            }
        }

        if (lane < head_dim) {
            float centroid_scale = rsqrt(float(head_dim));
            thread float dimension_accum[2];
            dimension_accum[0] = 0.0f;
            dimension_accum[1] = 0.0f;
            for (uint tile_lane = 0u; tile_lane < threads_per_block; tile_lane++) {
                uint vector_index =
                    (batch * uint(KV_HEADS) + kv_head) * uint(CAPACITY)
                    + tile_physical_tokens[tile_lane];
                uint code = tq_polar_wht_read_index(
                    v_packed_indices,
                    vector_index * uint(PACKED_WORDS_PER_VECTOR),
                    lane,
                    uint(POLAR_WHT_BITS));
                float centroid = tq_polar_wht_centroid(uint(POLAR_WHT_BITS), code);
                float value = float(v_norms[vector_index]) * centroid * centroid_scale;
                for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                    float weight = tile_scores[repeat * threads_per_block + tile_lane];
                    dimension_accum[repeat] += weight * value;
                }
            }
            for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                uint q_head = kv_head * gqa_repeats + repeat;
                uint row = ((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH)) + q_token;
                uint out_index = ((row * uint(BLOCK_COUNT) + block_index) * head_dim) + lane;
                partial_out[out_index] = dimension_accum[repeat];
            }
        }
        """

    private static let fusedAttentionBlockPartialsSource = """
        constexpr uint threads_per_block = uint(THREADS_PER_BLOCK);
        uint block_count = uint(runtime_block_count);
        uint lane = thread_position_in_threadgroup.x;
        uint group_index = threadgroup_position_in_grid.x;
        uint block_index = group_index % block_count;
        uint row = group_index / block_count;
        uint total_rows = uint(BATCH_SIZE) * uint(QUERY_HEADS) * uint(QUERY_LENGTH);
        if (row >= total_rows) {
            return;
        }

        threadgroup float partial[THREADS_PER_BLOCK];
        threadgroup float tile_scores[THREADS_PER_BLOCK];
        threadgroup uint tile_physical_tokens[THREADS_PER_BLOCK];
        threadgroup float query_cache[HEAD_DIM];

        uint logical_length = uint(runtime_logical_length);
        uint ring_offset = uint(runtime_ring_offset);
        uint pinned_prefix_length = uint(runtime_pinned_prefix_length);
        float attention_scale = float(runtime_attention_scale);
        uint q_token = row % uint(QUERY_LENGTH);
        uint q_head = (row / uint(QUERY_LENGTH)) % uint(QUERY_HEADS);
        uint batch = row / (uint(QUERY_LENGTH) * uint(QUERY_HEADS));
        uint repeats = uint(QUERY_HEADS) / uint(KV_HEADS);
        uint kv_head = q_head / repeats;
        uint causal_limit = logical_length - uint(QUERY_LENGTH) + q_token;
        uint block_start = block_index * uint(BLOCK_TOKENS);
        if (DO_CAUSAL && block_start > causal_limit) {
            if (lane == 0u) {
                uint stat_index = ((row * block_count + block_index) * 2u);
                partial_stats[stat_index] = -INFINITY;
                partial_stats[stat_index + 1u] = 0.0f;
            }
            if (lane < uint(HEAD_DIM)) {
                uint out_index = ((row * block_count + block_index) * uint(HEAD_DIM)) + lane;
                partial_out[out_index] = static_cast<OUTPUT_DTYPE>(0.0f);
            }
            return;
        }
        ulong key_seed = tq_make_seed(uint(SEED_3), uint(SEED_2), uint(SEED_1), uint(SEED_0));
        ulong value_seed = tq_make_seed(
            uint(VALUE_SEED_3), uint(VALUE_SEED_2),
            uint(VALUE_SEED_1), uint(VALUE_SEED_0));

        if (lane < uint(HEAD_DIM)) {
            long q_index =
                long(batch) * q_strides[0]
                + long(q_head) * q_strides[1]
                + long(q_token) * q_strides[2]
                + long(lane) * q_strides[3];
            query_cache[lane] = float(q[q_index]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint logical_token = block_start + lane;
        bool active = lane < uint(BLOCK_TOKENS)
            && logical_token < logical_length
            && (!DO_CAUSAL || logical_token <= causal_limit);
        float scaled_score = -INFINITY;
        uint physical_token = 0u;
        if (active) {
            physical_token = tq_physical_token(
                logical_token, uint(CAPACITY), ring_offset, pinned_prefix_length);
            float score = 0.0f;
            for (uint group = 0u; group < uint(GROUPS_PER_VECTOR); group++) {
                uint group_start = group * uint(GROUP_SIZE);
                uint count = min(uint(GROUP_SIZE), uint(HEAD_DIM) - group_start);
                thread float query_values[GROUP_SIZE];
                for (uint local = 0u; local < count; local++) {
                    query_values[local] = query_cache[group_start + local];
                }
                score += tq_product_attention_inner_product_group(
                    k_packed, k_signs, k_high_mask, k_residual_signs, k_scales, query_values,
                    batch, kv_head, physical_token, group, key_seed,
                    uint(GROUP_SIZE), uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR),
                    uint(MAG_WORDS_PER_GROUP), uint(BITSET_WORDS_PER_GROUP),
                    uint(KEY_BASE_BITS), uint(KEY_HIGH_BITS), uint(LAYOUT_VERSION),
                    uint(HEAD_DIM),
                    tq_high_precision_count(count, uint(HIGH_NUMERATOR), uint(HIGH_DENOMINATOR)));
            }
            scaled_score = score * attention_scale;
        }
        tile_scores[lane] = scaled_score;
        tile_physical_tokens[lane] = physical_token;
        partial[lane] = scaled_score;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
            if (lane < stride) {
                partial[lane] = max(partial[lane], partial[lane + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        float tile_max = partial[0];
        float tile_weight = active ? exp(tile_scores[lane] - tile_max) : 0.0f;
        tile_scores[lane] = tile_weight;
        partial[lane] = tile_weight;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
            if (lane < stride) {
                partial[lane] += partial[lane + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0u) {
            uint stat_index = ((row * block_count + block_index) * 2u);
            partial_stats[stat_index] = tile_max;
            partial_stats[stat_index + 1u] = partial[0];
        }

        if (lane < uint(HEAD_DIM)) {
            thread float decode_scratch[GROUP_SIZE];
            float dimension_accum = 0.0f;
            for (uint tile_lane = 0u; tile_lane < threads_per_block; tile_lane++) {
                float weight = tile_scores[tile_lane];
                if (weight > 0.0f) {
                    float value = tq_decode_attention_value(
                        v_packed, v_signs, v_high_mask, v_residual_signs, v_scales,
                        batch, kv_head, tile_physical_tokens[tile_lane], lane,
                        value_seed, 1u,
                        uint(GROUP_SIZE), uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR),
                        uint(VALUE_MAG_WORDS_PER_GROUP), uint(BITSET_WORDS_PER_GROUP), uint(BASE_BITS), uint(HIGH_BITS),
                        uint(VALUE_BITS), uint(KEY_BASE_BITS), uint(KEY_HIGH_BITS),
                        uint(LAYOUT_VERSION), uint(HEAD_DIM), 0u,
                        decode_scratch);
                    dimension_accum += weight * value;
                }
            }
            uint out_index = ((row * block_count + block_index) * uint(HEAD_DIM)) + lane;
            partial_out[out_index] = static_cast<OUTPUT_DTYPE>(dimension_accum);
        }
        """

    private static let fusedAttentionGQABlockPartialsSource_rf1 = """
        constexpr uint threads_per_block = uint(THREADS_PER_BLOCK);
        uint block_count = uint(runtime_block_count);
        constexpr uint gqa_repeats = uint(GQA_REPEATS);
        uint lane = thread_position_in_threadgroup.x;
        uint group_index = threadgroup_position_in_grid.x;
        uint block_index = group_index % block_count;
        uint gqa_row = group_index / block_count;
        uint total_gqa_rows = uint(BATCH_SIZE) * uint(KV_HEADS) * uint(QUERY_LENGTH);
        if (gqa_row >= total_gqa_rows) {
            return;
        }

        // Sized to the actual threadgroup width (was a fixed 4*512 / 512) so threadgroup
        // memory tracks the real block size — at blocks < 512 this frees enough threadgroup
        // memory for multiple threadgroups to be resident per core, raising occupancy.
        threadgroup float partial[4 * THREADS_PER_BLOCK];
        threadgroup float tile_scores[4 * THREADS_PER_BLOCK];
        threadgroup uint tile_has_weight[THREADS_PER_BLOCK];
        threadgroup uint tile_physical_tokens[THREADS_PER_BLOCK];
        threadgroup float query_cache[4 * HEAD_DIM];

        uint logical_length = uint(runtime_logical_length);
        uint ring_offset = uint(runtime_ring_offset);
        uint pinned_prefix_length = uint(runtime_pinned_prefix_length);
        float attention_scale = float(runtime_attention_scale);
        uint q_token = gqa_row % uint(QUERY_LENGTH);
        uint kv_head = (gqa_row / uint(QUERY_LENGTH)) % uint(KV_HEADS);
        uint batch = gqa_row / (uint(QUERY_LENGTH) * uint(KV_HEADS));
        uint causal_limit = logical_length - uint(QUERY_LENGTH) + q_token;
        uint block_start = block_index * uint(BLOCK_TOKENS);
        constexpr uint repeat_count = uint(GQA_REPEATS) < 4u ? uint(GQA_REPEATS) : 4u;
        if (DO_CAUSAL && block_start > causal_limit) {
            if (lane == 0u) {
                for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                    uint q_head = kv_head * gqa_repeats + repeat;
                    uint row = ((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH)) + q_token;
                    uint stat_index = ((row * block_count + block_index) * 2u);
                    partial_stats[stat_index] = -INFINITY;
                    partial_stats[stat_index + 1u] = 0.0f;
                }
            }
            if (lane < uint(HEAD_DIM)) {
                for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                    uint q_head = kv_head * gqa_repeats + repeat;
                    uint row = ((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH)) + q_token;
                    uint out_index = ((row * block_count + block_index) * uint(HEAD_DIM)) + lane;
                    partial_out[out_index] = static_cast<OUTPUT_DTYPE>(0.0f);
                }
            }
            return;
        }
        ulong key_seed = tq_make_seed(uint(SEED_3), uint(SEED_2), uint(SEED_1), uint(SEED_0));
        ulong value_seed = tq_make_seed(
            uint(VALUE_SEED_3), uint(VALUE_SEED_2),
            uint(VALUE_SEED_1), uint(VALUE_SEED_0));

        if (lane < uint(HEAD_DIM)) {
            for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                uint q_head = kv_head * gqa_repeats + repeat;
                long q_index =
                    long(batch) * q_strides[0]
                    + long(q_head) * q_strides[1]
                    + long(q_token) * q_strides[2]
                    + long(lane) * q_strides[3];
                query_cache[repeat * uint(HEAD_DIM) + lane] = float(q[q_index]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Part B: rotate the query ONCE per (repeat, group) and reuse it across every key
        // token. Because the rotation seed is token-independent (see tq_storage_group_index),
        // the rotation is identical for all keys, so hoisting it here turns the prior O(N)
        // per-key query rotations into O(repeat_count * groups_per_vector) per attention step.
        {
            uint rg_total = repeat_count * uint(GROUPS_PER_VECTOR);
            if (lane < rg_total) {
                uint r = lane / uint(GROUPS_PER_VECTOR);
                uint g = lane % uint(GROUPS_PER_VECTOR);
                uint gs = g * uint(GROUP_SIZE);
                uint cnt = min(uint(GROUP_SIZE), uint(HEAD_DIM) - gs);
                uint rot_seed_index = tq_storage_group_index(
                    batch, kv_head, 0u, g, uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR));
                thread float tmp[GROUP_SIZE];
                for (uint i = 0u; i < cnt; i++) {
                    tmp[i] = query_cache[r * uint(HEAD_DIM) + gs + i];
                }
                tq_apply_product_rotation(tmp, cnt, key_seed, rot_seed_index, false);
                for (uint i = 0u; i < cnt; i++) {
                    query_cache[r * uint(HEAD_DIM) + gs + i] = tmp[i];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        constexpr uint lanes_per_token = uint(LANES_PER_TOKEN);
        uint physical_token = 0u;
        bool active = false;
        thread float scaled_scores[4];
        scaled_scores[0] = -INFINITY;
        scaled_scores[1] = -INFINITY;
        scaled_scores[2] = -INFINITY;
        scaled_scores[3] = -INFINITY;

        if (lanes_per_token == 4u) {
            // TQCOOP cooperative quad-per-key coalesced decode (turbo8 uniform path).
            // 4 lanes cooperate on one key, each decoding a contiguous HEAD_DIM/4 chunk
            // (so the quad's 4 chunks tile one cache line -> coalesced, ~4x less L1
            // pressure than the strided 1-thread-per-token mapping). Each quad walks 4
            // tokens in 4 passes; lane j keeps the score of pass j so the existing
            // per-lane back-half (tile_scores[lane], reductions, AV) is unchanged.
            uint lane_in_quad = lane & 3u;
            uint quad_id = lane >> 2u;
            uint num_quads = uint(THREADS_PER_BLOCK) >> 2u;
            uint dims_per_lane = uint(HEAD_DIM) / 4u;
            uint dim_start = lane_in_quad * dims_per_lane;
            uint g = dim_start / uint(GROUP_SIZE);
            uint local_start = dim_start - g * uint(GROUP_SIZE);
            uint count_g = min(uint(GROUP_SIZE), uint(HEAD_DIM) - g * uint(GROUP_SIZE));
            float inv_sqrt_count = rsqrt(float(max(count_g, 1u)));
            float residual_scale_factor =
                sqrt(3.14159265358979323846f / (2.0f * float(count_g)));
            constexpr uint coop_base_bits = uint(KEY_BASE_BITS);
            constexpr uint coop_high_bits = uint(KEY_HIGH_BITS);
            // Coop handles uniform (turbo8/turbo4v2: base==high) AND split-magnitude
            // (turbo3_5: high==base+1 at layout v6 — a base-bits stream + a 1-bit high stream,
            // both per-group contiguous, so each lane's chunk still coalesces). Branch-2
            // variable-bit is dead at v6 and gated out, so these two cases are exhaustive.
            constexpr bool coop_split =
                uint(LAYOUT_VERSION) >= 6u && coop_high_bits == coop_base_bits + 1u;
            uint coop_high_count = coop_split
                ? tq_high_precision_count(
                    count_g, uint(HIGH_NUMERATOR), uint(HIGH_DENOMINATOR))
                : 0u;
            uint my_token = 0u;
            for (uint j = 0u; j < 4u; j++) {
                uint logical_token = block_start + quad_id + j * num_quads;
                bool tok_active = logical_token < logical_length
                    && (!DO_CAUSAL || logical_token <= causal_limit);
                thread float ts[4];
                ts[0] = 0.0f; ts[1] = 0.0f; ts[2] = 0.0f; ts[3] = 0.0f;
                uint phys = 0u;
                if (tok_active) {
                    phys = tq_physical_token(
                        logical_token, uint(CAPACITY), ring_offset, pinned_prefix_length);
                    uint base = tq_packed_offset(
                        batch, kv_head, phys, g, 0u,
                        uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR),
                        uint(MAG_WORDS_PER_GROUP));
                    uint cached_idx = 0xffffffffu;
                    uint cached_val = 0u;
                    uint extra_idx = 0xffffffffu;
                    uint extra_val = 0u;
                    uint cached_bw = 0xffffffffu;
                    uint cached_sign = 0u;
                    thread float qd[4];
                    thread float sd[4];
                    qd[0] = 0.0f; qd[1] = 0.0f; qd[2] = 0.0f; qd[3] = 0.0f;
                    sd[0] = 0.0f; sd[1] = 0.0f; sd[2] = 0.0f; sd[3] = 0.0f;
                    for (uint i = 0u; i < dims_per_lane; i++) {
                        uint local = local_start + i;
                        uint dim = dim_start + i;
                        uint bits;
                        uint code;
                        if (coop_split) {
                            bool hp = local < coop_high_count;
                            bits = hp ? coop_high_bits : coop_base_bits;
                            uint base_bo = local * coop_base_bits;
                            uint base_pw = base_bo >> 5;
                            uint base_pb = base_bo & 31u;
                            if (base_pw != cached_idx) {
                                cached_idx = base_pw;
                                cached_val = k_packed[base + base_pw];
                            }
                            uint base_asm = cached_val >> base_pb;
                            if (base_pb + coop_base_bits > 32u) {
                                base_asm |= k_packed[base + base_pw + 1u] << (32u - base_pb);
                            }
                            code = base_asm & ((1u << coop_base_bits) - 1u);
                            if (hp) {
                                uint extra_bits = coop_high_bits - coop_base_bits;
                                uint extra_bo = uint(GROUP_SIZE) * coop_base_bits + local;
                                uint extra_pw = extra_bo >> 5;
                                uint extra_pb = extra_bo & 31u;
                                if (extra_pw != extra_idx) {
                                    extra_idx = extra_pw;
                                    extra_val = k_packed[base + extra_pw];
                                }
                                uint extra_asm = extra_val >> extra_pb;
                                if (extra_pb + extra_bits > 32u) {
                                    extra_asm |=
                                        k_packed[base + extra_pw + 1u] << (32u - extra_pb);
                                }
                                code |= (extra_asm & ((1u << extra_bits) - 1u)) << coop_base_bits;
                            }
                        } else {
                            bits = coop_base_bits;
                            uint bo = local * coop_base_bits;
                            uint pw = bo >> 5;
                            uint pb = bo & 31u;
                            if (pw != cached_idx) {
                                cached_idx = pw;
                                cached_val = k_packed[base + pw];
                            }
                            uint aw = cached_val >> pb;
                            if (pb + coop_base_bits > 32u) {
                                aw |= k_packed[base + pw + 1u] << (32u - pb);
                            }
                            code = aw & ((1u << coop_base_bits) - 1u);
                        }
                        uint bw = local >> 5;
                        if (bw != cached_bw) {
                            cached_bw = bw;
                            cached_sign = k_signs[tq_bitset_offset(
                                batch, kv_head, phys, g, bw,
                                uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR),
                                uint(BITSET_WORDS_PER_GROUP))];
                        }
                        float level = tq_codebook_unit(bits, code) * inv_sqrt_count;
                        float sgn = (cached_sign & (1u << (local & 31u))) != 0u ? -1.0f : 1.0f;
                        // COOPW invariant: rows [repeat_count, 4) of query_cache are
                        // NEVER initialized by the shared prologue (init loop and rotation
                        // are both bounded by repeat_count). Clamping every coop r-loop to
                        // repeat_count is load-bearing: the hardcoded r<4u form read those
                        // uninitialized threadgroup rows (UB, shader-validation trap risk).
                        // At repeat_count==4 the bound is unchanged -> byte-identical.
                        for (uint r = 0u; r < repeat_count; r++) {
                            float qv = query_cache[r * uint(HEAD_DIM) + dim];
                            qd[r] += qv * level;
                            sd[r] += sgn * qv;
                        }
                    }
                    float norm = k_scales[tq_scale_offset(
                        batch, kv_head, phys, g, 0u,
                        uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR))];
                    float residual_norm = k_scales[tq_scale_offset(
                        batch, kv_head, phys, g, 1u,
                        uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR))];
                    float residual_scale = residual_norm * residual_scale_factor;
                    for (uint r = 0u; r < repeat_count; r++) {
                        ts[r] = norm * qd[r] + residual_scale * sd[r];
                    }
                }
                for (uint r = 0u; r < repeat_count; r++) {
                    ts[r] += simd_shuffle_xor(ts[r], 1u);
                    ts[r] += simd_shuffle_xor(ts[r], 2u);
                }
                if (lane_in_quad == j) {
                    active = tok_active;
                    my_token = tok_active ? phys : 0u;
                    for (uint r = 0u; r < repeat_count; r++) {
                        scaled_scores[r] = tok_active ? ts[r] * attention_scale : -INFINITY;
                    }
                }
            }
            physical_token = my_token;
        } else {
        uint logical_token = block_start + lane;
        active = lane < uint(BLOCK_TOKENS)
            && logical_token < logical_length
            && (!DO_CAUSAL || logical_token <= causal_limit);

        if (active) {
            physical_token = tq_physical_token(
                logical_token, uint(CAPACITY), ring_offset, pinned_prefix_length);
            for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                scaled_scores[repeat] = 0.0f;
            }
            for (uint group = 0u; group < uint(GROUPS_PER_VECTOR); group++) {
                uint group_start = group * uint(GROUP_SIZE);
                uint count = min(uint(GROUP_SIZE), uint(HEAD_DIM) - group_start);
                if (repeat_count == 4u) {
                    thread float query_values[4 * GROUP_SIZE];
                    thread float quad_scores[4];
                    for (uint repeat = 0u; repeat < 4u; repeat++) {
                        quad_scores[repeat] = 0.0f;
                        for (uint local = 0u; local < count; local++) {
                            query_values[repeat * uint(GROUP_SIZE) + local] =
                                query_cache[repeat * uint(HEAD_DIM) + group_start + local];
                        }
                    }
                    tq_product_attention_inner_product_group_quad(
                        k_packed, k_signs, k_high_mask, k_residual_signs, k_scales, query_values,
                        quad_scores,
                        batch, kv_head, physical_token, group, key_seed,
                        uint(GROUP_SIZE), uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR),
                        uint(MAG_WORDS_PER_GROUP), uint(BITSET_WORDS_PER_GROUP),
                        uint(KEY_BASE_BITS), uint(KEY_HIGH_BITS), uint(LAYOUT_VERSION),
                        uint(HEAD_DIM),
                        tq_high_precision_count(count, uint(HIGH_NUMERATOR), uint(HIGH_DENOMINATOR)),
                        true);
                    for (uint repeat = 0u; repeat < 4u; repeat++) {
                        scaled_scores[repeat] += quad_scores[repeat];
                    }
                    continue;
                }
                for (uint pair_start = 0u; pair_start < repeat_count; pair_start += 2u) {
                    uint pair_repeats = min(2u, repeat_count - pair_start);
                    thread float query_values[2 * GROUP_SIZE];
                    thread float pair_scores[2];
                    pair_scores[0] = 0.0f;
                    pair_scores[1] = 0.0f;
                    for (uint pair = 0u; pair < pair_repeats; pair++) {
                        uint repeat = pair_start + pair;
                        for (uint local = 0u; local < count; local++) {
                            query_values[pair * uint(GROUP_SIZE) + local] =
                                query_cache[repeat * uint(HEAD_DIM) + group_start + local];
                        }
                    }
                    tq_product_attention_inner_product_group_pair(
                        k_packed, k_signs, k_high_mask, k_residual_signs, k_scales, query_values,
                        pair_scores,
                        pair_repeats, batch, kv_head, physical_token, group, key_seed,
                        uint(GROUP_SIZE), uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR),
                        uint(MAG_WORDS_PER_GROUP), uint(BITSET_WORDS_PER_GROUP),
                        uint(KEY_BASE_BITS), uint(KEY_HIGH_BITS), uint(LAYOUT_VERSION),
                        uint(HEAD_DIM),
                        tq_high_precision_count(count, uint(HIGH_NUMERATOR), uint(HIGH_DENOMINATOR)),
                        true);
                    for (uint pair = 0u; pair < pair_repeats; pair++) {
                        scaled_scores[pair_start + pair] += pair_scores[pair];
                    }
                }
            }
            for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                scaled_scores[repeat] *= attention_scale;
            }
        }
        }
        tile_physical_tokens[lane] = physical_token;

        for (uint repeat = 0u; repeat < repeat_count; repeat++) {
            uint score_base = repeat * threads_per_block;
            tile_scores[score_base + lane] = scaled_scores[repeat];
            partial[score_base + lane] = scaled_scores[repeat];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
            if (lane < stride) {
                for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                    uint score_base = repeat * threads_per_block;
                    partial[score_base + lane] =
                        max(partial[score_base + lane], partial[score_base + lane + stride]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        thread float tile_maxes[4];
        tile_maxes[0] = -INFINITY;
        tile_maxes[1] = -INFINITY;
        tile_maxes[2] = -INFINITY;
        tile_maxes[3] = -INFINITY;
        for (uint repeat = 0u; repeat < repeat_count; repeat++) {
            tile_maxes[repeat] = partial[repeat * threads_per_block];
        }
        // Every lane must finish reading the reduced maxes above BEFORE any lane
        // overwrites partial[] with exp-weights below. Without this barrier, a lagging
        // simdgroup can read lane 0's already-written weight as the "max" for a later
        // repeat (drift across the barrier-free repeat loop is widest at the last
        // repeat), which intermittently corrupts the whole output row of one GQA
        // repeat at BT>=512 / 131072 (256 active blocks). Root-caused 2026-07-06 as
        // the same class as the v7 quad race; see
        // artifacts/turboquant-w2-regrad-20260706 and
        // artifacts/turboquant-v7-20260703/quad-nondeterminism.md.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint has_weight = 0u;
        for (uint repeat = 0u; repeat < repeat_count; repeat++) {
            uint score_base = repeat * threads_per_block;
            float tile_weight = active
                ? exp(tile_scores[score_base + lane] - tile_maxes[repeat])
                : 0.0f;
            tile_scores[score_base + lane] = tile_weight;
            partial[score_base + lane] = tile_weight;
            if (tile_weight > 0.0f) {
                has_weight = 1u;
            }
        }
        tile_has_weight[lane] = has_weight;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
            if (lane < stride) {
                for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                    uint score_base = repeat * threads_per_block;
                    partial[score_base + lane] += partial[score_base + lane + stride];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0u) {
            for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                uint q_head = kv_head * gqa_repeats + repeat;
                uint row = ((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH)) + q_token;
                uint stat_index = ((row * block_count + block_index) * 2u);
                uint score_base = repeat * threads_per_block;
                partial_stats[stat_index] = tile_maxes[repeat];
                partial_stats[stat_index + 1u] = partial[score_base];
            }
        }

        if (lane < uint(HEAD_DIM)) {
            thread float decode_scratch[GROUP_SIZE];
            thread float dimension_accum[4];
            dimension_accum[0] = 0.0f;
            dimension_accum[1] = 0.0f;
            dimension_accum[2] = 0.0f;
            dimension_accum[3] = 0.0f;

            for (uint tile_lane = 0u; tile_lane < threads_per_block; tile_lane++) {
                if (tile_has_weight[tile_lane] != 0u) {
                    float value = tq_decode_attention_value(
                        v_packed, v_signs, v_high_mask, v_residual_signs, v_scales,
                        batch, kv_head, tile_physical_tokens[tile_lane], lane,
                        value_seed, 1u,
                        uint(GROUP_SIZE), uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR),
                        uint(VALUE_MAG_WORDS_PER_GROUP), uint(BITSET_WORDS_PER_GROUP), uint(BASE_BITS), uint(HIGH_BITS),
                        uint(VALUE_BITS), uint(KEY_BASE_BITS), uint(KEY_HIGH_BITS),
                        uint(LAYOUT_VERSION), uint(HEAD_DIM), 0u,
                        decode_scratch);
                    for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                        dimension_accum[repeat] +=
                            tile_scores[repeat * threads_per_block + tile_lane] * value;
                    }
                }
            }

            for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                uint q_head = kv_head * gqa_repeats + repeat;
                uint row = ((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH)) + q_token;
                uint out_index = ((row * block_count + block_index) * uint(HEAD_DIM)) + lane;
                partial_out[out_index] = static_cast<OUTPUT_DTYPE>(dimension_accum[repeat]);
            }
        }
        """

    private static let fusedAttentionGQABlockPartialsH16Source_rf1 = """
        constexpr uint threads_per_block = uint(THREADS_PER_BLOCK);
        uint block_count = uint(runtime_block_count);
        constexpr uint gqa_repeats = uint(GQA_REPEATS);
        uint lane = thread_position_in_threadgroup.x;
        uint group_index = threadgroup_position_in_grid.x;
        uint block_index = group_index % block_count;
        uint gqa_row = group_index / block_count;
        uint total_gqa_rows = uint(BATCH_SIZE) * uint(KV_HEADS) * uint(QUERY_LENGTH);
        if (gqa_row >= total_gqa_rows) {
            return;
        }

        // T2.2 H16 diet: partial/tile_scores staged as half (was float) and
        // tile_has_weight folded into a 32-lane bitset (was uint[THREADS_PER_BLOCK]).
        // This halves the two largest tgmem arrays and shrinks the mask array from
        // THREADS_PER_BLOCK*4B to (THREADS_PER_BLOCK/32)*4B, clearing the < 16384 B
        // static-tgmem boundary needed for 2 threadgroups/core (see G5 probe). All
        // reduction math still runs in fp32 registers; only the threadgroup-memory
        // storage dtype changes. query_cache and tile_physical_tokens are unchanged
        // (rotation precision / full 32-bit token indices at 131K).
        threadgroup half partial[4 * THREADS_PER_BLOCK];
        threadgroup half tile_scores[4 * THREADS_PER_BLOCK];
        threadgroup uint tile_has_weight_bits[(THREADS_PER_BLOCK + 31u) / 32u];
        threadgroup uint tile_physical_tokens[THREADS_PER_BLOCK];
        threadgroup float query_cache[4 * HEAD_DIM];

        uint logical_length = uint(runtime_logical_length);
        uint ring_offset = uint(runtime_ring_offset);
        uint pinned_prefix_length = uint(runtime_pinned_prefix_length);
        float attention_scale = float(runtime_attention_scale);
        uint q_token = gqa_row % uint(QUERY_LENGTH);
        uint kv_head = (gqa_row / uint(QUERY_LENGTH)) % uint(KV_HEADS);
        uint batch = gqa_row / (uint(QUERY_LENGTH) * uint(KV_HEADS));
        uint causal_limit = logical_length - uint(QUERY_LENGTH) + q_token;
        uint block_start = block_index * uint(BLOCK_TOKENS);
        constexpr uint repeat_count = uint(GQA_REPEATS) < 4u ? uint(GQA_REPEATS) : 4u;
        if (DO_CAUSAL && block_start > causal_limit) {
            if (lane == 0u) {
                for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                    uint q_head = kv_head * gqa_repeats + repeat;
                    uint row = ((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH)) + q_token;
                    uint stat_index = ((row * block_count + block_index) * 2u);
                    partial_stats[stat_index] = -INFINITY;
                    partial_stats[stat_index + 1u] = 0.0f;
                }
            }
            if (lane < uint(HEAD_DIM)) {
                for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                    uint q_head = kv_head * gqa_repeats + repeat;
                    uint row = ((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH)) + q_token;
                    uint out_index = ((row * block_count + block_index) * uint(HEAD_DIM)) + lane;
                    partial_out[out_index] = static_cast<OUTPUT_DTYPE>(0.0f);
                }
            }
            return;
        }
        ulong key_seed = tq_make_seed(uint(SEED_3), uint(SEED_2), uint(SEED_1), uint(SEED_0));
        ulong value_seed = tq_make_seed(
            uint(VALUE_SEED_3), uint(VALUE_SEED_2),
            uint(VALUE_SEED_1), uint(VALUE_SEED_0));

        if (lane < uint(HEAD_DIM)) {
            for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                uint q_head = kv_head * gqa_repeats + repeat;
                long q_index =
                    long(batch) * q_strides[0]
                    + long(q_head) * q_strides[1]
                    + long(q_token) * q_strides[2]
                    + long(lane) * q_strides[3];
                query_cache[repeat * uint(HEAD_DIM) + lane] = float(q[q_index]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Part B: rotate the query ONCE per (repeat, group) and reuse it across every key
        // token. Because the rotation seed is token-independent (see tq_storage_group_index),
        // the rotation is identical for all keys, so hoisting it here turns the prior O(N)
        // per-key query rotations into O(repeat_count * groups_per_vector) per attention step.
        {
            uint rg_total = repeat_count * uint(GROUPS_PER_VECTOR);
            if (lane < rg_total) {
                uint r = lane / uint(GROUPS_PER_VECTOR);
                uint g = lane % uint(GROUPS_PER_VECTOR);
                uint gs = g * uint(GROUP_SIZE);
                uint cnt = min(uint(GROUP_SIZE), uint(HEAD_DIM) - gs);
                uint rot_seed_index = tq_storage_group_index(
                    batch, kv_head, 0u, g, uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR));
                thread float tmp[GROUP_SIZE];
                for (uint i = 0u; i < cnt; i++) {
                    tmp[i] = query_cache[r * uint(HEAD_DIM) + gs + i];
                }
                tq_apply_product_rotation(tmp, cnt, key_seed, rot_seed_index, false);
                for (uint i = 0u; i < cnt; i++) {
                    query_cache[r * uint(HEAD_DIM) + gs + i] = tmp[i];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        constexpr uint lanes_per_token = uint(LANES_PER_TOKEN);
        uint physical_token = 0u;
        bool active = false;
        thread float scaled_scores[4];
        scaled_scores[0] = -INFINITY;
        scaled_scores[1] = -INFINITY;
        scaled_scores[2] = -INFINITY;
        scaled_scores[3] = -INFINITY;

        if (lanes_per_token == 4u) {
            // TQCOOP cooperative quad-per-key coalesced decode (turbo8 uniform path).
            // 4 lanes cooperate on one key, each decoding a contiguous HEAD_DIM/4 chunk
            // (so the quad's 4 chunks tile one cache line -> coalesced, ~4x less L1
            // pressure than the strided 1-thread-per-token mapping). Each quad walks 4
            // tokens in 4 passes; lane j keeps the score of pass j so the existing
            // per-lane back-half (tile_scores[lane], reductions, AV) is unchanged.
            uint lane_in_quad = lane & 3u;
            uint quad_id = lane >> 2u;
            uint num_quads = uint(THREADS_PER_BLOCK) >> 2u;
            uint dims_per_lane = uint(HEAD_DIM) / 4u;
            uint dim_start = lane_in_quad * dims_per_lane;
            uint g = dim_start / uint(GROUP_SIZE);
            uint local_start = dim_start - g * uint(GROUP_SIZE);
            uint count_g = min(uint(GROUP_SIZE), uint(HEAD_DIM) - g * uint(GROUP_SIZE));
            float inv_sqrt_count = rsqrt(float(max(count_g, 1u)));
            float residual_scale_factor =
                sqrt(3.14159265358979323846f / (2.0f * float(count_g)));
            constexpr uint coop_base_bits = uint(KEY_BASE_BITS);
            constexpr uint coop_high_bits = uint(KEY_HIGH_BITS);
            // Coop handles uniform (turbo8/turbo4v2: base==high) AND split-magnitude
            // (turbo3_5: high==base+1 at layout v6 — a base-bits stream + a 1-bit high stream,
            // both per-group contiguous, so each lane's chunk still coalesces). Branch-2
            // variable-bit is dead at v6 and gated out, so these two cases are exhaustive.
            constexpr bool coop_split =
                uint(LAYOUT_VERSION) >= 6u && coop_high_bits == coop_base_bits + 1u;
            uint coop_high_count = coop_split
                ? tq_high_precision_count(
                    count_g, uint(HIGH_NUMERATOR), uint(HIGH_DENOMINATOR))
                : 0u;
            uint my_token = 0u;
            for (uint j = 0u; j < 4u; j++) {
                uint logical_token = block_start + quad_id + j * num_quads;
                bool tok_active = logical_token < logical_length
                    && (!DO_CAUSAL || logical_token <= causal_limit);
                thread float ts[4];
                ts[0] = 0.0f; ts[1] = 0.0f; ts[2] = 0.0f; ts[3] = 0.0f;
                uint phys = 0u;
                if (tok_active) {
                    phys = tq_physical_token(
                        logical_token, uint(CAPACITY), ring_offset, pinned_prefix_length);
                    uint base = tq_packed_offset(
                        batch, kv_head, phys, g, 0u,
                        uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR),
                        uint(MAG_WORDS_PER_GROUP));
                    uint cached_idx = 0xffffffffu;
                    uint cached_val = 0u;
                    uint extra_idx = 0xffffffffu;
                    uint extra_val = 0u;
                    uint cached_bw = 0xffffffffu;
                    uint cached_sign = 0u;
                    thread float qd[4];
                    thread float sd[4];
                    qd[0] = 0.0f; qd[1] = 0.0f; qd[2] = 0.0f; qd[3] = 0.0f;
                    sd[0] = 0.0f; sd[1] = 0.0f; sd[2] = 0.0f; sd[3] = 0.0f;
                    for (uint i = 0u; i < dims_per_lane; i++) {
                        uint local = local_start + i;
                        uint dim = dim_start + i;
                        uint bits;
                        uint code;
                        if (coop_split) {
                            bool hp = local < coop_high_count;
                            bits = hp ? coop_high_bits : coop_base_bits;
                            uint base_bo = local * coop_base_bits;
                            uint base_pw = base_bo >> 5;
                            uint base_pb = base_bo & 31u;
                            if (base_pw != cached_idx) {
                                cached_idx = base_pw;
                                cached_val = k_packed[base + base_pw];
                            }
                            uint base_asm = cached_val >> base_pb;
                            if (base_pb + coop_base_bits > 32u) {
                                base_asm |= k_packed[base + base_pw + 1u] << (32u - base_pb);
                            }
                            code = base_asm & ((1u << coop_base_bits) - 1u);
                            if (hp) {
                                uint extra_bits = coop_high_bits - coop_base_bits;
                                uint extra_bo = uint(GROUP_SIZE) * coop_base_bits + local;
                                uint extra_pw = extra_bo >> 5;
                                uint extra_pb = extra_bo & 31u;
                                if (extra_pw != extra_idx) {
                                    extra_idx = extra_pw;
                                    extra_val = k_packed[base + extra_pw];
                                }
                                uint extra_asm = extra_val >> extra_pb;
                                if (extra_pb + extra_bits > 32u) {
                                    extra_asm |=
                                        k_packed[base + extra_pw + 1u] << (32u - extra_pb);
                                }
                                code |= (extra_asm & ((1u << extra_bits) - 1u)) << coop_base_bits;
                            }
                        } else {
                            bits = coop_base_bits;
                            uint bo = local * coop_base_bits;
                            uint pw = bo >> 5;
                            uint pb = bo & 31u;
                            if (pw != cached_idx) {
                                cached_idx = pw;
                                cached_val = k_packed[base + pw];
                            }
                            uint aw = cached_val >> pb;
                            if (pb + coop_base_bits > 32u) {
                                aw |= k_packed[base + pw + 1u] << (32u - pb);
                            }
                            code = aw & ((1u << coop_base_bits) - 1u);
                        }
                        uint bw = local >> 5;
                        if (bw != cached_bw) {
                            cached_bw = bw;
                            cached_sign = k_signs[tq_bitset_offset(
                                batch, kv_head, phys, g, bw,
                                uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR),
                                uint(BITSET_WORDS_PER_GROUP))];
                        }
                        float level = tq_codebook_unit(bits, code) * inv_sqrt_count;
                        float sgn = (cached_sign & (1u << (local & 31u))) != 0u ? -1.0f : 1.0f;
                        for (uint r = 0u; r < 4u; r++) {
                            float qv = query_cache[r * uint(HEAD_DIM) + dim];
                            qd[r] += qv * level;
                            sd[r] += sgn * qv;
                        }
                    }
                    float norm = k_scales[tq_scale_offset(
                        batch, kv_head, phys, g, 0u,
                        uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR))];
                    float residual_norm = k_scales[tq_scale_offset(
                        batch, kv_head, phys, g, 1u,
                        uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR))];
                    float residual_scale = residual_norm * residual_scale_factor;
                    for (uint r = 0u; r < 4u; r++) {
                        ts[r] = norm * qd[r] + residual_scale * sd[r];
                    }
                }
                for (uint r = 0u; r < 4u; r++) {
                    ts[r] += simd_shuffle_xor(ts[r], 1u);
                    ts[r] += simd_shuffle_xor(ts[r], 2u);
                }
                if (lane_in_quad == j) {
                    active = tok_active;
                    my_token = tok_active ? phys : 0u;
                    for (uint r = 0u; r < 4u; r++) {
                        scaled_scores[r] = tok_active ? ts[r] * attention_scale : -INFINITY;
                    }
                }
            }
            physical_token = my_token;
        } else {
        uint logical_token = block_start + lane;
        active = lane < uint(BLOCK_TOKENS)
            && logical_token < logical_length
            && (!DO_CAUSAL || logical_token <= causal_limit);

        if (active) {
            physical_token = tq_physical_token(
                logical_token, uint(CAPACITY), ring_offset, pinned_prefix_length);
            for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                scaled_scores[repeat] = 0.0f;
            }
            for (uint group = 0u; group < uint(GROUPS_PER_VECTOR); group++) {
                uint group_start = group * uint(GROUP_SIZE);
                uint count = min(uint(GROUP_SIZE), uint(HEAD_DIM) - group_start);
                if (repeat_count == 4u) {
                    thread float query_values[4 * GROUP_SIZE];
                    thread float quad_scores[4];
                    for (uint repeat = 0u; repeat < 4u; repeat++) {
                        quad_scores[repeat] = 0.0f;
                        for (uint local = 0u; local < count; local++) {
                            query_values[repeat * uint(GROUP_SIZE) + local] =
                                query_cache[repeat * uint(HEAD_DIM) + group_start + local];
                        }
                    }
                    tq_product_attention_inner_product_group_quad(
                        k_packed, k_signs, k_high_mask, k_residual_signs, k_scales, query_values,
                        quad_scores,
                        batch, kv_head, physical_token, group, key_seed,
                        uint(GROUP_SIZE), uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR),
                        uint(MAG_WORDS_PER_GROUP), uint(BITSET_WORDS_PER_GROUP),
                        uint(KEY_BASE_BITS), uint(KEY_HIGH_BITS), uint(LAYOUT_VERSION),
                        uint(HEAD_DIM),
                        tq_high_precision_count(count, uint(HIGH_NUMERATOR), uint(HIGH_DENOMINATOR)),
                        true);
                    for (uint repeat = 0u; repeat < 4u; repeat++) {
                        scaled_scores[repeat] += quad_scores[repeat];
                    }
                    continue;
                }
                for (uint pair_start = 0u; pair_start < repeat_count; pair_start += 2u) {
                    uint pair_repeats = min(2u, repeat_count - pair_start);
                    thread float query_values[2 * GROUP_SIZE];
                    thread float pair_scores[2];
                    pair_scores[0] = 0.0f;
                    pair_scores[1] = 0.0f;
                    for (uint pair = 0u; pair < pair_repeats; pair++) {
                        uint repeat = pair_start + pair;
                        for (uint local = 0u; local < count; local++) {
                            query_values[pair * uint(GROUP_SIZE) + local] =
                                query_cache[repeat * uint(HEAD_DIM) + group_start + local];
                        }
                    }
                    tq_product_attention_inner_product_group_pair(
                        k_packed, k_signs, k_high_mask, k_residual_signs, k_scales, query_values,
                        pair_scores,
                        pair_repeats, batch, kv_head, physical_token, group, key_seed,
                        uint(GROUP_SIZE), uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR),
                        uint(MAG_WORDS_PER_GROUP), uint(BITSET_WORDS_PER_GROUP),
                        uint(KEY_BASE_BITS), uint(KEY_HIGH_BITS), uint(LAYOUT_VERSION),
                        uint(HEAD_DIM),
                        tq_high_precision_count(count, uint(HIGH_NUMERATOR), uint(HIGH_DENOMINATOR)),
                        true);
                    for (uint pair = 0u; pair < pair_repeats; pair++) {
                        scaled_scores[pair_start + pair] += pair_scores[pair];
                    }
                }
            }
            for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                scaled_scores[repeat] *= attention_scale;
            }
        }
        }
        tile_physical_tokens[lane] = physical_token;

        for (uint repeat = 0u; repeat < repeat_count; repeat++) {
            uint score_base = repeat * threads_per_block;
            // H16 overflow guard: the raw pre-softmax logit is unbounded and is
            // stored as half before the max-subtraction. Clamp to a safe fp16
            // sub-max range; -INFINITY (masked/inactive lanes) maps to -65504,
            // which the exp(x - max) step drives to 0 identically to -inf.
            float raw = scaled_scores[repeat];
            float guarded = raw;
            if (!(raw <= 60000.0f)) {
                guarded = (raw == -INFINITY) ? -65504.0f : 60000.0f;
            } else if (raw < -65504.0f) {
                guarded = -65504.0f;
            }
            tile_scores[score_base + lane] = half(guarded);
            partial[score_base + lane] = half(guarded);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
            if (lane < stride) {
                for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                    uint score_base = repeat * threads_per_block;
                    float a = float(partial[score_base + lane]);
                    float b = float(partial[score_base + lane + stride]);
                    partial[score_base + lane] = half(max(a, b));
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        thread float tile_maxes[4];
        tile_maxes[0] = -INFINITY;
        tile_maxes[1] = -INFINITY;
        tile_maxes[2] = -INFINITY;
        tile_maxes[3] = -INFINITY;
        // Clear this lane's bitset word once (only the 32 lane-0-of-word threads).
        if ((lane & 31u) == 0u) { tile_has_weight_bits[lane >> 5] = 0u; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint repeat = 0u; repeat < repeat_count; repeat++) {
            tile_maxes[repeat] = float(partial[repeat * threads_per_block]);
        }
        // Read the reduced maxes above BEFORE any lane overwrites partial[] with
        // exp-weights below (same v6-family RAW race fixed in the fp32 GQA kernel;
        // see artifacts/turboquant-w2-regrad-20260706).
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint has_weight = 0u;
        for (uint repeat = 0u; repeat < repeat_count; repeat++) {
            uint score_base = repeat * threads_per_block;
            float tile_weight = active
                ? exp(float(tile_scores[score_base + lane]) - tile_maxes[repeat])
                : 0.0f;
            tile_scores[score_base + lane] = half(tile_weight);
            partial[score_base + lane] = half(tile_weight);
            if (tile_weight > 0.0f) {
                has_weight = 1u;
            }
        }
        if (has_weight != 0u) {
            atomic_fetch_or_explicit(
                (threadgroup atomic_uint*)&tile_has_weight_bits[lane >> 5],
                1u << (lane & 31u), memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
            if (lane < stride) {
                for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                    uint score_base = repeat * threads_per_block;
                    float s = float(partial[score_base + lane])
                            + float(partial[score_base + lane + stride]);
                    partial[score_base + lane] = half(s);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0u) {
            for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                uint q_head = kv_head * gqa_repeats + repeat;
                uint row = ((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH)) + q_token;
                uint stat_index = ((row * block_count + block_index) * 2u);
                uint score_base = repeat * threads_per_block;
                partial_stats[stat_index] = tile_maxes[repeat];
                partial_stats[stat_index + 1u] = float(partial[score_base]);
            }
        }

        if (lane < uint(HEAD_DIM)) {
            thread float decode_scratch[GROUP_SIZE];
            thread float dimension_accum[4];
            dimension_accum[0] = 0.0f;
            dimension_accum[1] = 0.0f;
            dimension_accum[2] = 0.0f;
            dimension_accum[3] = 0.0f;

            for (uint tile_lane = 0u; tile_lane < threads_per_block; tile_lane++) {
                if ((tile_has_weight_bits[tile_lane >> 5] & (1u << (tile_lane & 31u))) != 0u) {
                    float value = tq_decode_attention_value(
                        v_packed, v_signs, v_high_mask, v_residual_signs, v_scales,
                        batch, kv_head, tile_physical_tokens[tile_lane], lane,
                        value_seed, 1u,
                        uint(GROUP_SIZE), uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR),
                        uint(VALUE_MAG_WORDS_PER_GROUP), uint(BITSET_WORDS_PER_GROUP), uint(BASE_BITS), uint(HIGH_BITS),
                        uint(VALUE_BITS), uint(KEY_BASE_BITS), uint(KEY_HIGH_BITS),
                        uint(LAYOUT_VERSION), uint(HEAD_DIM), 0u,
                        decode_scratch);
                    for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                        dimension_accum[repeat] +=
                            float(tile_scores[repeat * threads_per_block + tile_lane]) * value;
                    }
                }
            }

            for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                uint q_head = kv_head * gqa_repeats + repeat;
                uint row = ((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH)) + q_token;
                uint out_index = ((row * block_count + block_index) * uint(HEAD_DIM)) + lane;
                partial_out[out_index] = static_cast<OUTPUT_DTYPE>(dimension_accum[repeat]);
            }
        }
        """

    private static let fusedAttentionGQABlockPartialsV7Source = """
        constexpr uint threads_per_block = uint(THREADS_PER_BLOCK);
        uint block_count = uint(runtime_block_count);
        constexpr uint gqa_repeats = uint(GQA_REPEATS);
        uint lane = thread_position_in_threadgroup.x;
        uint group_index = threadgroup_position_in_grid.x;
        uint block_index = group_index % block_count;
        uint gqa_row = group_index / block_count;
        uint total_gqa_rows = uint(BATCH_SIZE) * uint(KV_HEADS) * uint(QUERY_LENGTH);
        if (gqa_row >= total_gqa_rows) {
            return;
        }

        // Sized to the actual threadgroup width (was a fixed 4*512 / 512) so threadgroup
        // memory tracks the real block size — at blocks < 512 this frees enough threadgroup
        // memory for multiple threadgroups to be resident per core, raising occupancy.
        threadgroup float partial[4 * THREADS_PER_BLOCK];
        threadgroup float tile_scores[4 * THREADS_PER_BLOCK];
        threadgroup uint tile_has_weight[THREADS_PER_BLOCK];
        threadgroup uint tile_physical_tokens[THREADS_PER_BLOCK];
        threadgroup float query_cache[4 * HEAD_DIM];

        uint logical_length = uint(runtime_logical_length);
        uint ring_offset = uint(runtime_ring_offset);
        uint pinned_prefix_length = uint(runtime_pinned_prefix_length);
        float attention_scale = float(runtime_attention_scale);
        uint q_token = gqa_row % uint(QUERY_LENGTH);
        uint kv_head = (gqa_row / uint(QUERY_LENGTH)) % uint(KV_HEADS);
        uint batch = gqa_row / (uint(QUERY_LENGTH) * uint(KV_HEADS));
        uint causal_limit = logical_length - uint(QUERY_LENGTH) + q_token;
        uint block_start = block_index * uint(BLOCK_TOKENS);
        constexpr uint repeat_count = uint(GQA_REPEATS) < 4u ? uint(GQA_REPEATS) : 4u;
        if (DO_CAUSAL && block_start > causal_limit) {
            if (lane == 0u) {
                for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                    uint q_head = kv_head * gqa_repeats + repeat;
                    uint row = ((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH)) + q_token;
                    uint stat_index = ((row * block_count + block_index) * 2u);
                    partial_stats[stat_index] = -INFINITY;
                    partial_stats[stat_index + 1u] = 0.0f;
                }
            }
            if (lane < uint(HEAD_DIM)) {
                for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                    uint q_head = kv_head * gqa_repeats + repeat;
                    uint row = ((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH)) + q_token;
                    uint out_index = ((row * block_count + block_index) * uint(HEAD_DIM)) + lane;
                    partial_out[out_index] = static_cast<OUTPUT_DTYPE>(0.0f);
                }
            }
            return;
        }
        ulong key_seed = tq_make_seed(uint(SEED_3), uint(SEED_2), uint(SEED_1), uint(SEED_0));
        ulong value_seed = tq_make_seed(
            uint(VALUE_SEED_3), uint(VALUE_SEED_2),
            uint(VALUE_SEED_1), uint(VALUE_SEED_0));

        if (lane < uint(HEAD_DIM)) {
            for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                uint q_head = kv_head * gqa_repeats + repeat;
                long q_index =
                    long(batch) * q_strides[0]
                    + long(q_head) * q_strides[1]
                    + long(q_token) * q_strides[2]
                    + long(lane) * q_strides[3];
                query_cache[repeat * uint(HEAD_DIM) + lane] = float(q[q_index]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Part B: rotate the query ONCE per (repeat, group) and reuse it across every key
        // token. Because the rotation seed is token-independent (see tq_storage_group_index),
        // the rotation is identical for all keys, so hoisting it here turns the prior O(N)
        // per-key query rotations into O(repeat_count * groups_per_vector) per attention step.
        {
            uint rg_total = repeat_count * uint(GROUPS_PER_VECTOR);
            if (lane < rg_total) {
                uint r = lane / uint(GROUPS_PER_VECTOR);
                uint g = lane % uint(GROUPS_PER_VECTOR);
                uint gs = g * uint(GROUP_SIZE);
                uint cnt = min(uint(GROUP_SIZE), uint(HEAD_DIM) - gs);
                uint rot_seed_index = tq_storage_group_index(
                    batch, kv_head, 0u, g, uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR));
                thread float tmp[GROUP_SIZE];
                for (uint i = 0u; i < cnt; i++) {
                    tmp[i] = query_cache[r * uint(HEAD_DIM) + gs + i];
                }
                tq_apply_product_rotation(tmp, cnt, key_seed, rot_seed_index, false);
                for (uint i = 0u; i < cnt; i++) {
                    query_cache[r * uint(HEAD_DIM) + gs + i] = tmp[i];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint physical_token = 0u;
        bool active = false;
        thread float scaled_scores[4];
        scaled_scores[0] = -INFINITY;
        scaled_scores[1] = -INFINITY;
        scaled_scores[2] = -INFINITY;
        scaled_scores[3] = -INFINITY;

        uint logical_token = block_start + lane;
        active = lane < uint(BLOCK_TOKENS)
            && logical_token < logical_length
            && (!DO_CAUSAL || logical_token <= causal_limit);

        if (active) {
            physical_token = tq_physical_token(
                logical_token, uint(CAPACITY), ring_offset, pinned_prefix_length);
            for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                scaled_scores[repeat] = 0.0f;
            }
            for (uint group = 0u; group < uint(GROUPS_PER_VECTOR); group++) {
                uint group_start = group * uint(GROUP_SIZE);
                uint count = min(uint(GROUP_SIZE), uint(HEAD_DIM) - group_start);
                if (repeat_count == 4u) {
                    thread float query_values[4 * GROUP_SIZE];
                    thread float quad_scores[4];
                    for (uint repeat = 0u; repeat < 4u; repeat++) {
                        quad_scores[repeat] = 0.0f;
                        for (uint local = 0u; local < count; local++) {
                            query_values[repeat * uint(GROUP_SIZE) + local] =
                                query_cache[repeat * uint(HEAD_DIM) + group_start + local];
                        }
                    }
                    tq_product_attention_inner_product_group_quad_v7(
                        k_packed, k_signs, k_high_mask, k_residual_signs, k_scales, query_values,
                        quad_scores,
                        batch, kv_head, physical_token, group, key_seed,
                        uint(GROUP_SIZE), uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR),
                        uint(MAG_WORDS_PER_GROUP), uint(BITSET_WORDS_PER_GROUP),
                        uint(KEY_BASE_BITS), uint(KEY_HIGH_BITS), uint(LAYOUT_VERSION),
                        uint(HEAD_DIM),
                        tq_high_precision_count(count, uint(HIGH_NUMERATOR), uint(HIGH_DENOMINATOR)),
                        true);
                    for (uint repeat = 0u; repeat < 4u; repeat++) {
                        scaled_scores[repeat] += quad_scores[repeat];
                    }
                    continue;
                }
                for (uint pair_start = 0u; pair_start < repeat_count; pair_start += 2u) {
                    uint pair_repeats = min(2u, repeat_count - pair_start);
                    thread float query_values[2 * GROUP_SIZE];
                    thread float pair_scores[2];
                    pair_scores[0] = 0.0f;
                    pair_scores[1] = 0.0f;
                    for (uint pair = 0u; pair < pair_repeats; pair++) {
                        uint repeat = pair_start + pair;
                        for (uint local = 0u; local < count; local++) {
                            query_values[pair * uint(GROUP_SIZE) + local] =
                                query_cache[repeat * uint(HEAD_DIM) + group_start + local];
                        }
                    }
                    tq_product_attention_inner_product_group_pair_v7(
                        k_packed, k_signs, k_high_mask, k_residual_signs, k_scales, query_values,
                        pair_scores,
                        pair_repeats, batch, kv_head, physical_token, group, key_seed,
                        uint(GROUP_SIZE), uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR),
                        uint(MAG_WORDS_PER_GROUP), uint(BITSET_WORDS_PER_GROUP),
                        uint(KEY_BASE_BITS), uint(KEY_HIGH_BITS), uint(LAYOUT_VERSION),
                        uint(HEAD_DIM),
                        tq_high_precision_count(count, uint(HIGH_NUMERATOR), uint(HIGH_DENOMINATOR)),
                        true);
                    for (uint pair = 0u; pair < pair_repeats; pair++) {
                        scaled_scores[pair_start + pair] += pair_scores[pair];
                    }
                }
            }
            for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                scaled_scores[repeat] *= attention_scale;
            }
        }
        tile_physical_tokens[lane] = physical_token;

        for (uint repeat = 0u; repeat < repeat_count; repeat++) {
            uint score_base = repeat * threads_per_block;
            tile_scores[score_base + lane] = scaled_scores[repeat];
            partial[score_base + lane] = scaled_scores[repeat];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
            if (lane < stride) {
                for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                    uint score_base = repeat * threads_per_block;
                    partial[score_base + lane] =
                        max(partial[score_base + lane], partial[score_base + lane + stride]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        thread float tile_maxes[4];
        tile_maxes[0] = -INFINITY;
        tile_maxes[1] = -INFINITY;
        tile_maxes[2] = -INFINITY;
        tile_maxes[3] = -INFINITY;
        for (uint repeat = 0u; repeat < repeat_count; repeat++) {
            tile_maxes[repeat] = partial[repeat * threads_per_block];
        }
        // Every lane must finish reading the reduced maxes above BEFORE any lane
        // overwrites partial[] with exp-weights below. Without this barrier, a lagging
        // simdgroup can read lane 0's already-written weight as the "max" for a later
        // repeat (the drift across the barrier-free repeat loop is widest at the last
        // repeat), which intermittently corrupted the whole output row of the last GQA
        // repeat (q_head = kv_head*4+3) at a 5-25% per-dispatch rate. Root-caused
        // 2026-07-03; see artifacts/turboquant-v7-20260703/quad-nondeterminism.md.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint has_weight = 0u;
        for (uint repeat = 0u; repeat < repeat_count; repeat++) {
            uint score_base = repeat * threads_per_block;
            float tile_weight = active
                ? exp(tile_scores[score_base + lane] - tile_maxes[repeat])
                : 0.0f;
            tile_scores[score_base + lane] = tile_weight;
            partial[score_base + lane] = tile_weight;
            if (tile_weight > 0.0f) {
                has_weight = 1u;
            }
        }
        tile_has_weight[lane] = has_weight;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
            if (lane < stride) {
                for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                    uint score_base = repeat * threads_per_block;
                    partial[score_base + lane] += partial[score_base + lane + stride];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0u) {
            for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                uint q_head = kv_head * gqa_repeats + repeat;
                uint row = ((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH)) + q_token;
                uint stat_index = ((row * block_count + block_index) * 2u);
                uint score_base = repeat * threads_per_block;
                partial_stats[stat_index] = tile_maxes[repeat];
                partial_stats[stat_index + 1u] = partial[score_base];
            }
        }

        if (lane < uint(HEAD_DIM)) {
            thread float decode_scratch[GROUP_SIZE];
            thread float dimension_accum[4];
            dimension_accum[0] = 0.0f;
            dimension_accum[1] = 0.0f;
            dimension_accum[2] = 0.0f;
            dimension_accum[3] = 0.0f;

            for (uint tile_lane = 0u; tile_lane < threads_per_block; tile_lane++) {
                if (tile_has_weight[tile_lane] != 0u) {
                    float value = tq_decode_attention_value(
                        v_packed, v_signs, v_high_mask, v_residual_signs, v_scales,
                        batch, kv_head, tile_physical_tokens[tile_lane], lane,
                        value_seed, 1u,
                        uint(GROUP_SIZE), uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR),
                        uint(VALUE_MAG_WORDS_PER_GROUP), uint(BITSET_WORDS_PER_GROUP), uint(BASE_BITS), uint(HIGH_BITS),
                        uint(VALUE_BITS), uint(KEY_BASE_BITS), uint(KEY_HIGH_BITS),
                        uint(LAYOUT_VERSION), uint(HEAD_DIM), 0u,
                        decode_scratch);
                    for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                        dimension_accum[repeat] +=
                            tile_scores[repeat * threads_per_block + tile_lane] * value;
                    }
                }
            }

            for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                uint q_head = kv_head * gqa_repeats + repeat;
                uint row = ((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH)) + q_token;
                uint out_index = ((row * block_count + block_index) * uint(HEAD_DIM)) + lane;
                partial_out[out_index] = static_cast<OUTPUT_DTYPE>(dimension_accum[repeat]);
            }
        }
        """

    private static let fusedAttentionBlockReduceSource = """
        constexpr uint threads_per_block = uint(THREADS_PER_BLOCK);
        uint block_count = uint(runtime_block_count);
        uint lane = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;
        if (row >= uint(ROW_COUNT)) {
            return;
        }

        threadgroup float partial[512];
        threadgroup float tile_scales[512];

        if (lane < block_count) {
            partial[lane] = partial_stats[(row * block_count + lane) * 2u];
        } else {
            partial[lane] = -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
            if (lane < stride) {
                partial[lane] = max(partial[lane], partial[lane + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        float row_max = partial[0];
        if (lane < block_count) {
            uint stat_index = (row * block_count + lane) * 2u;
            float tile_sum = partial_stats[stat_index + 1u];
            float tile_scale = tile_sum > 0.0f ? exp(partial_stats[stat_index] - row_max) : 0.0f;
            tile_scales[lane] = tile_scale;
            partial[lane] = tile_scale * tile_sum;
        } else {
            tile_scales[lane] = 0.0f;
            partial[lane] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
            if (lane < stride) {
                partial[lane] += partial[lane + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        float row_sum = partial[0];
        if (lane < uint(HEAD_DIM)) {
            float accum = 0.0f;
            for (uint block = 0u; block < block_count; block++) {
                float tile_scale = tile_scales[block];
                if (tile_scale > 0.0f) {
                    uint partial_index = ((row * block_count + block) * uint(HEAD_DIM)) + lane;
                    accum += tile_scale * partial_out[partial_index];
                }
            }
            out[row * uint(HEAD_DIM) + lane] = static_cast<OUTPUT_DTYPE>(
                accum / max(row_sum, 1.17549435e-38f));
        }
        """

    private static let segmentedRawAttentionStatsSource = """
        constexpr uint threads_per_block = uint(THREADS_PER_BLOCK);
        uint raw_length = uint(runtime_raw_length);
        uint lane = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;
        uint row_count = uint(BATCH_SIZE) * uint(QUERY_HEADS) * uint(QUERY_LENGTH);
        if (row >= row_count) {
            return;
        }

        threadgroup float partial[512];
        threadgroup float tile_scores[512];
        threadgroup float query_cache[HEAD_DIM];
        threadgroup float output_accum[HEAD_DIM];

        uint q_token = row % uint(QUERY_LENGTH);
        uint q_head = (row / uint(QUERY_LENGTH)) % uint(QUERY_HEADS);
        uint batch = row / (uint(QUERY_LENGTH) * uint(QUERY_HEADS));
        uint repeats = uint(QUERY_HEADS) / uint(KV_HEADS);
        uint kv_head = q_head / repeats;
        float attention_scale = float(runtime_attention_scale);

        float row_max = -INFINITY;
        float row_sum = 0.0f;
        if (lane < uint(HEAD_DIM)) {
            long q_index =
                long(batch) * q_strides[0]
                + long(q_head) * q_strides[1]
                + long(q_token) * q_strides[2]
                + long(lane) * q_strides[3];
            query_cache[lane] = float(q[q_index]);
            output_accum[lane] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint tile_start = 0u; tile_start < raw_length; tile_start += threads_per_block) {
            uint token = tile_start + lane;
            bool active = token < raw_length;
            float scaled_score = -INFINITY;
            if (active) {
                float score = 0.0f;
                for (uint dim = 0u; dim < uint(HEAD_DIM); dim++) {
                    long k_index =
                        long(batch) * k_strides[0]
                        + long(kv_head) * k_strides[1]
                        + long(token) * k_strides[2]
                        + long(dim) * k_strides[3];
                    score += query_cache[dim] * float(k[k_index]);
                }
                scaled_score = score * attention_scale;
            }
            tile_scores[lane] = scaled_score;
            partial[lane] = scaled_score;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
                if (lane < stride) {
                    partial[lane] = max(partial[lane], partial[lane + stride]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            float tile_max = partial[0];
            float tile_weight = active ? exp(tile_scores[lane] - tile_max) : 0.0f;
            tile_scores[lane] = tile_weight;
            partial[lane] = tile_weight;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
                if (lane < stride) {
                    partial[lane] += partial[lane + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            float tile_sum = partial[0];
            float new_max = max(row_max, tile_max);
            float previous_scale = row_sum > 0.0f ? exp(row_max - new_max) : 0.0f;
            float tile_scale = tile_sum > 0.0f ? exp(tile_max - new_max) : 0.0f;

            if (lane < uint(HEAD_DIM)) {
                output_accum[lane] *= previous_scale;
                float dimension_accum = 0.0f;
                for (uint tile_lane = 0u; tile_lane < threads_per_block; tile_lane++) {
                    float weight = tile_scores[tile_lane];
                    if (weight > 0.0f) {
                        uint value_token = tile_start + tile_lane;
                        long v_index =
                            long(batch) * v_strides[0]
                            + long(kv_head) * v_strides[1]
                            + long(value_token) * v_strides[2]
                            + long(lane) * v_strides[3];
                        dimension_accum += weight * float(v[v_index]);
                    }
                }
                output_accum[lane] += tile_scale * dimension_accum;
            }
            row_sum = row_sum * previous_scale + tile_sum * tile_scale;
            row_max = new_max;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0u) {
            uint stat_index = row * 2u;
            partial_stats[stat_index] = row_max;
            partial_stats[stat_index + 1u] = row_sum;
        }
        if (lane < uint(HEAD_DIM)) {
            partial_out[row * uint(HEAD_DIM) + lane] = output_accum[lane];
        }
        """

    private static let fusedAttentionSource = """
        constexpr uint threads_per_row = uint(THREADS_PER_ROW);
        uint lane = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;
        uint total_rows = uint(BATCH_SIZE) * uint(QUERY_HEADS) * uint(QUERY_LENGTH);
        if (row >= total_rows) {
            return;
        }

        threadgroup float partial[256];
        threadgroup float tile_scores[256];
        threadgroup uint tile_physical_tokens[256];
        threadgroup float query_cache[HEAD_DIM];
        threadgroup float output_accum[HEAD_DIM];

        uint logical_length = uint(runtime_logical_length);
        uint ring_offset = uint(runtime_ring_offset);
        uint pinned_prefix_length = uint(runtime_pinned_prefix_length);
        float attention_scale = float(runtime_attention_scale);
        uint q_token = row % uint(QUERY_LENGTH);
        uint q_head = (row / uint(QUERY_LENGTH)) % uint(QUERY_HEADS);
        uint batch = row / (uint(QUERY_LENGTH) * uint(QUERY_HEADS));
        uint repeats = uint(QUERY_HEADS) / uint(KV_HEADS);
        uint kv_head = q_head / repeats;
        uint causal_limit = logical_length - uint(QUERY_LENGTH) + q_token;
        ulong key_seed = tq_make_seed(uint(SEED_3), uint(SEED_2), uint(SEED_1), uint(SEED_0));
        ulong value_seed = tq_make_seed(
            uint(VALUE_SEED_3), uint(VALUE_SEED_2),
            uint(VALUE_SEED_1), uint(VALUE_SEED_0));

        float row_max = -INFINITY;
        float row_sum = 0.0f;
        if (lane < uint(HEAD_DIM)) {
            long q_index =
                long(batch) * q_strides[0]
                + long(q_head) * q_strides[1]
                + long(q_token) * q_strides[2]
                + long(lane) * q_strides[3];
            query_cache[lane] = float(q[q_index]);
            output_accum[lane] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint tile_start = 0u; tile_start < logical_length; tile_start += threads_per_row) {
            uint logical_token = tile_start + lane;
            bool active = logical_token < logical_length
                && (!DO_CAUSAL || logical_token <= causal_limit);
            float scaled_score = -INFINITY;
            uint physical_token = 0u;
            if (active) {
                physical_token = tq_physical_token(
                    logical_token, uint(CAPACITY), ring_offset, pinned_prefix_length);
                float score = 0.0f;
                for (uint group = 0u; group < uint(GROUPS_PER_VECTOR); group++) {
                    uint group_start = group * uint(GROUP_SIZE);
                    uint count = min(uint(GROUP_SIZE), uint(HEAD_DIM) - group_start);
                    thread float query_values[GROUP_SIZE];
                    for (uint local = 0u; local < count; local++) {
                        query_values[local] = query_cache[group_start + local];
                    }
                    score += tq_product_attention_inner_product_group(
                        k_packed, k_signs, k_high_mask, k_residual_signs, k_scales, query_values,
                        batch, kv_head, physical_token, group, key_seed,
                        uint(GROUP_SIZE), uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR),
                        uint(MAG_WORDS_PER_GROUP), uint(BITSET_WORDS_PER_GROUP),
                        uint(KEY_BASE_BITS), uint(KEY_HIGH_BITS), uint(LAYOUT_VERSION),
                        uint(HEAD_DIM),
                        tq_high_precision_count(count, uint(HIGH_NUMERATOR), uint(HIGH_DENOMINATOR)));
                }
                scaled_score = score * attention_scale;
            }
            tile_scores[lane] = scaled_score;
            tile_physical_tokens[lane] = physical_token;
            partial[lane] = scaled_score;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = threads_per_row >> 1; stride > 0u; stride >>= 1) {
                if (lane < stride) {
                    partial[lane] = max(partial[lane], partial[lane + stride]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            float tile_max = partial[0];
            float new_row_max = max(row_max, tile_max);
            float old_scale = row_sum > 0.0f ? exp(row_max - new_row_max) : 0.0f;
            if (lane < uint(HEAD_DIM)) {
                output_accum[lane] *= old_scale;
            }

            float weight = active ? exp(tile_scores[lane] - new_row_max) : 0.0f;
            tile_scores[lane] = weight;
            partial[lane] = weight;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = threads_per_row >> 1; stride > 0u; stride >>= 1) {
                if (lane < stride) {
                    partial[lane] += partial[lane + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            float next_row_sum = row_sum * old_scale + partial[0];
            if (lane < uint(HEAD_DIM)) {
                thread float decode_scratch[GROUP_SIZE];
                float dimension_accum = output_accum[lane];
                for (uint tile_lane = 0u; tile_lane < threads_per_row; tile_lane++) {
                    float tile_weight = tile_scores[tile_lane];
                    if (tile_weight > 0.0f) {
                        float value = tq_decode_attention_value(
                            v_packed, v_signs, v_high_mask, v_residual_signs, v_scales,
                            batch, kv_head, tile_physical_tokens[tile_lane], lane,
                            value_seed, 1u,
                            uint(GROUP_SIZE), uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR),
                            uint(VALUE_MAG_WORDS_PER_GROUP), uint(BITSET_WORDS_PER_GROUP), uint(BASE_BITS), uint(HIGH_BITS),
                            uint(VALUE_BITS), uint(KEY_BASE_BITS), uint(KEY_HIGH_BITS),
                            uint(LAYOUT_VERSION), uint(HEAD_DIM), 0u,
                            decode_scratch);
                        dimension_accum += tile_weight * value;
                    }
                }
                output_accum[lane] = dimension_accum;
            }
            row_max = new_row_max;
            row_sum = next_row_sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane < uint(HEAD_DIM)) {
            float inv_sum = 1.0f / max(row_sum, 1.17549435e-38f);
            uint out_index =
                (((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH) + q_token)
                    * uint(HEAD_DIM)) + lane;
            out[out_index] = static_cast<OUTPUT_DTYPE>(output_accum[lane] * inv_sum);
        }
        """
}
