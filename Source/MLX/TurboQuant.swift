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

    /// Mixed-bit key and bitpacked-value PolarQuant/QJL Metal kernels.
    case metalPolarQJL
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
    public var selectedKernelProfile: TurboQuantKernelProfile
    public var failureReason: String?
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
        case selectedKernelProfile
        case failureReason
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
        selectedKernelProfile: TurboQuantKernelProfile = .mlxPackedFallback,
        failureReason: String? = nil,
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
        self.selectedKernelProfile = selectedKernelProfile
        self.failureReason = failureReason
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
        selectedKernelProfile =
            try container.decode(TurboQuantKernelProfile.self, forKey: .selectedKernelProfile)
        failureReason = try container.decodeIfPresent(String.self, forKey: .failureReason)
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
        try container.encode(selectedKernelProfile, forKey: .selectedKernelProfile)
        try container.encodeIfPresent(failureReason, forKey: .failureReason)
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
            flatEncodeDecode: metalRuntimeAvailable && flatCodecPassed,
            linearMatmul: turboQuantExperimentalLinearMetalEnabled()
                && metalRuntimeAvailable && flatCodecPassed,
            attentionEncode: attentionCodecPassed,
            attentionDecode: attentionCodecPassed,
            attentionQK: qkAvailable,
            attentionAV: avAvailable,
            attentionFusedDecode: qkAvailable && avAvailable && tiledFusedPassed,
            attentionTiledFusedDecode: qkAvailable && avAvailable && tiledFusedPassed,
            bfloatOutput: attentionCodecPassed && bfloatOutputPassed,
            supportedHeadDimensions: (qkAvailable && avAvailable && tiledFusedPassed) ? onlineFusedHeadDimensions : [],
            selectedKernelProfile: selectedKernelProfile,
            failureReasons: failureReason.map { [$0] } ?? []
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

public struct TurboQuantKernelAvailability: Equatable, Codable, Sendable {
    public var supportsMLXPacked: Bool
    public var supportsPolarQJLReference: Bool
    public var supportsMetalPolarQJLCodec: Bool
    public var supportsMetalPolarQJLAttention: Bool
    public var supportsMetalPolarQJL: Bool
    public var selectedKernelProfile: TurboQuantKernelProfile
    public var selfTestStatus: TurboQuantRuntimeSelfTestStatus
    public var selfTestFailureReason: String?
    public var onlineFusedHeadDimensions: [Int]

    public var kernelCapabilities: TurboQuantKernelCapabilities {
        let probeCapabilities = TurboQuantRuntimeProbe.shared.result().kernelCapabilities
        return TurboQuantKernelCapabilities(
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
        supportsMetalPolarQJLCodec: Bool = false,
        supportsMetalPolarQJLAttention: Bool = false,
        supportsMetalPolarQJL: Bool = false,
        selectedKernelProfile: TurboQuantKernelProfile = .mlxPackedFallback,
        selfTestStatus: TurboQuantRuntimeSelfTestStatus = .notRun,
        selfTestFailureReason: String? = nil,
        onlineFusedHeadDimensions: [Int] = TurboQuantRuntimeProbeResult
            .throughputOptimizedOnlineFusedHeadDimensions
    ) {
        self.supportsMLXPacked = supportsMLXPacked
        self.supportsPolarQJLReference = supportsPolarQJLReference
        self.supportsMetalPolarQJLCodec = supportsMetalPolarQJLCodec
        self.supportsMetalPolarQJLAttention = supportsMetalPolarQJLAttention
        self.supportsMetalPolarQJL = supportsMetalPolarQJL
        self.selectedKernelProfile = selectedKernelProfile
        self.selfTestStatus = selfTestStatus
        self.selfTestFailureReason = selfTestFailureReason
        self.onlineFusedHeadDimensions = onlineFusedHeadDimensions
    }

    public static var current: TurboQuantKernelAvailability {
        let metalAvailable = metalRuntimeAvailable()
        let probe = TurboQuantRuntimeProbe.shared.result()
        let probeCapabilities = probe.kernelCapabilities
        let codecAvailable = metalAvailable && probeCapabilities.flatEncodeDecode
        let attentionAvailable =
            metalAvailable && probeCapabilities.attentionQK && probeCapabilities.attentionAV
        return TurboQuantKernelAvailability(
            supportsMetalPolarQJLCodec: codecAvailable,
            supportsMetalPolarQJLAttention: attentionAvailable,
            supportsMetalPolarQJL: codecAvailable || attentionAvailable,
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
        case .metalPolarQJL:
            supportsMetalPolarQJL
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
        case .metalPolarQJL:
            if let selfTestFailureReason {
                return
                    "TurboQuant Metal self-test failed: \(selfTestFailureReason); using MLX packed TurboQuant lanes."
            }
            return
                "TurboQuant Metal kernels unavailable; using MLX packed TurboQuant lanes."
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
        attentionLayoutVersion: Int = TurboQuantAttentionLayout.currentVersion,
        allowExperimentalLayoutV5: Bool = false,
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
        ) ?? TurboQuantAttentionLayout.currentVersion
        allowExperimentalLayoutV5 = try container.decodeIfPresent(
            Bool.self,
            forKey: .allowExperimentalLayoutV5
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
        }
    }

    public var approximateBitsPerValue: Double {
        guard valueCount > 0 else { return 0 }
        return Double(storageByteCount * 8) / Double(valueCount)
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
    case onlineFused
    case tiledOnlineFused
    case twoStageCompressed
    case mlxPackedFallback
    case baseline
    case unavailable
}

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
    public var encode: Bool
    public var decode: Bool
    public var qk: Bool
    public var av: Bool
    public var onlineFused: Bool
    public var tiledOnlineFused: Bool
    public var bfloatOutput: Bool
    public var supportedOnlineFusedHeadDimensions: [Int]
    public var maxOnlineFusedQueryLength: Int
    public var maxTiledOnlineFusedQueryLength: Int
    public var materializedMaskTwoStage: Bool
    public var supportedDTypes: [TurboQuantDTypeKind]
    public var supportedMasks: [TurboQuantAttentionMaskKind]
    public var supportedDeviceFamilies: [String]

    public init(
        encode: Bool = false,
        decode: Bool = false,
        qk: Bool = false,
        av: Bool = false,
        onlineFused: Bool = false,
        tiledOnlineFused: Bool? = nil,
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
        self.encode = encode
        self.decode = decode
        self.qk = qk
        self.av = av
        self.onlineFused = onlineFused
        self.tiledOnlineFused = tiledOnlineFused ?? onlineFused
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
        deviceFamily: String? = nil
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
    }
}

public struct TurboQuantAttentionLayout: Hashable, Codable, Sendable {
    public static let legacyVersion = 4
    public static let splitMagnitudeVersion = 6
    public static let currentVersion = splitMagnitudeVersion
    public static let nextVersion = currentVersion
    public static let supportedVersions = [legacyVersion, 5, splitMagnitudeVersion]

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
        layoutVersion: Int = TurboQuantAttentionLayout.currentVersion,
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
        self.scalesPerGroup = scalesPerGroup ?? (role == .value ? 2 : 3)
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
    layoutVersion: Int = TurboQuantAttentionLayout.currentVersion,
    allowExperimentalLayoutV5: Bool = false
) throws -> TurboQuantAttentionLayout {
    try validateAttentionArray(array, groupSize: groupSize)
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
        allowExperimentalLayoutV5: allowExperimentalLayoutV5
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
    layoutVersion: Int = TurboQuantAttentionLayout.currentVersion,
    allowExperimentalLayoutV5: Bool = false
) throws -> TurboQuantAttentionLayout {
    try validateRequestedAttentionLayoutVersion(
        layoutVersion,
        allowExperimentalLayoutV5: allowExperimentalLayoutV5
    )
    try validateAttentionShape(shape, dtype: dtype, groupSize: groupSize)
    let headDimension = shape[3]
    let groupsPerVector = (headDimension + groupSize - 1) / groupSize
    let resolvedCapacity = capacity ?? shape[2]
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
    try validateAttentionLayout(layout, role: role, groupSize: groupSize)
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
    try validateAttentionArray(array, groupSize: configuration.groupSize)
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
        allowExperimentalLayoutV5: configuration.allowExperimentalLayoutV5
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
    let outputs = TurboQuantMetalKernels.encodeAttention(
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
    try validateAttentionCodeStorage(keyCode)
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

public func turboQuantMetalAV(
    attentionWeights: MLXArray,
    valueCode: TurboQuantAttentionCode,
    outputDType: DType = .float32,
    stream: StreamOrDevice = .gpu
) throws -> MLXArray {
    try validateAttentionCodeStorage(valueCode)
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
    stream: StreamOrDevice = .gpu
) throws -> MLXArray {
    try validateAttentionPair(keyCode: keyCode, valueCode: valueCode)
    try validateAttentionQuery(queries, code: keyCode)
    try validateTurboQuantAttentionCode(keyCode, expectedRole: .key)
    try validateTurboQuantAttentionCode(valueCode, expectedRole: .value)
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
    let decision = try turboQuantAttentionDecision(
        request: TurboQuantAttentionRequest(
            queryShape: queries.shape,
            keyLayout: keyCode.layout,
            valueLayout: valueCode.layout,
            queryDType: queries.dtype,
            outputDType: queries.dtype,
            maskKind: turboQuantAttentionMaskKind(mask),
            hasSinks: sinks != nil,
            preferOnlineFused: preferOnlineFused,
            memoryBudgetBytes: memoryBudgetBytes,
            fallbackState: fallbackState
        ),
        capabilities: attentionCapabilities
    )

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
    return try turboQuantMetalAV(
        attentionWeights: weights,
        valueCode: valueCode,
        outputDType: queries.dtype,
        stream: stream
    )
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
    stream: StreamOrDevice
) throws -> MLXArray {
    try validateAttentionPair(keyCode: keyCode, valueCode: valueCode)
    try validateAttentionQuery(queries, code: keyCode)
    try validateAttentionCodeStorage(keyCode)
    try validateAttentionCodeStorage(valueCode)
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

    if turboQuantShouldUseBlockParallelFusedAttention(
        queries: queries,
        keyCode: keyCode,
        valueCode: valueCode,
        kernelProfile: kernelProfile,
        blockParallelTokenBlockSize: blockParallelTokenBlockSize
    ) {
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
    let blockWidth = turboQuantBlockParallelFusedThreadgroupWidth(
        minimum: max(
            queries.dim(3),
            blockParallelTokenBlockSize ?? kernelProfile.blockParallelFusedTokenBlockSize
        )
    )
    let activeBlockCount = (keyCode.layout.logicalLength + blockWidth - 1) / blockWidth
    guard activeBlockCount > 1, activeBlockCount <= blockWidth else { return false }
    return keyCode.layout.logicalLength >= 4_096
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
    let blockWidth = turboQuantBlockParallelFusedThreadgroupWidth(
        minimum: max(
            queries.dim(3),
            blockParallelTokenBlockSize ?? kernelProfile.blockParallelFusedTokenBlockSize
        )
    )
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
            ("BLOCK_COUNT", activeBlockCount),
            ("GQA_REPEATS", useGroupedQueryKernel ? queryHeadRepeats : 1),
        ] + metalTemplateSeedWords(prefix: "VALUE_SEED", value: valueCode.seed)

    let partialRows =
        useGroupedQueryKernel
        ? queries.dim(0) * kvHeadCount * queries.dim(2)
        : rowCount
    let partialKernel =
        useGroupedQueryKernel
        ? TurboQuantMetalKernels.fusedAttentionGQABlockPartials
        : TurboQuantMetalKernels.fusedAttentionBlockPartials
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
    return TurboQuantMetalKernels.fusedAttentionBlockReduce(
        partials,
        template: [
            ("ROW_COUNT", rowCount),
            ("HEAD_DIM", queries.dim(3)),
            ("BLOCK_COUNT", activeBlockCount),
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

private func decodeTurboQuantReference(_ code: TurboQuantReferenceCode) throws -> [Float] {
    switch code.format {
    case .affineValue:
        return try decodeTurboQuantAffineValueReference(code)
    case .turboQuantProd:
        return try decodeTurboQuantProductReference(code)
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

private func metalRuntimeAvailable() -> Bool {
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
                selectedKernelProfile: passed ? selectedProfile : .mlxPackedFallback,
                failureReason: failureReason,
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
    array.eval()
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
    role == .value ? 2 : 3
}

private func validateRequestedAttentionLayoutVersion(
    _ layoutVersion: Int,
    allowExperimentalLayoutV5: Bool
) throws {
    _ = allowExperimentalLayoutV5
    if TurboQuantAttentionLayout.supportedVersions.contains(layoutVersion) {
        return
    }
    throw TurboQuantError.invalidMetalConfiguration(
        "unsupported compressed attention layout version \(layoutVersion)"
    )
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
        _ = allowExperimentalLayoutV5
        guard layoutVersion >= 5
        else {
            throw TurboQuantError.invalidMetalConfiguration(
                "fp16 TurboQuant attention scales require Layout V5 or newer"
            )
        }
    }
}

private func validateAttentionConfiguration(_ configuration: TurboQuantConfiguration) throws {
    try validateRequestedAttentionLayoutVersion(
        configuration.attentionLayoutVersion,
        allowExperimentalLayoutV5: configuration.allowExperimentalLayoutV5
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
    groupSize: Int
) throws {
    guard role == .key || role == .value else {
        throw TurboQuantError.invalidMetalConfiguration(
            "compressed attention codes must be encoded as key or value"
        )
    }
    guard TurboQuantAttentionLayout.supportedVersions.contains(layout.layoutVersion) else {
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
}

private func validateAttentionCodeStorage(_ code: TurboQuantAttentionCode) throws {
    try validateAttentionLayout(code.layout, role: code.role, groupSize: code.groupSize)
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
    try validateAttentionArray(queries, groupSize: code.groupSize)
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
    valueCode: TurboQuantAttentionCode
) throws {
    try validateAttentionLayout(keyCode.layout, role: keyCode.role, groupSize: keyCode.groupSize)
    try validateAttentionLayout(
        valueCode.layout, role: valueCode.role, groupSize: valueCode.groupSize)
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
        name: "turboquant_polar_qjl_encode",
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
        name: "turboquant_attention_encode",
        inputNames: ["x"],
        outputNames: ["packed", "signs", "high_mask", "residual_signs", "scales"],
        source: encodeAttentionSource,
        header: attentionHeader,
        ensureRowContiguous: false
    )

    static let decodeAttention = MLXFast.metalKernel(
        name: "turboquant_attention_decode_runtime_layout",
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

    static let qk = MLXFast.metalKernel(
        name: "turboquant_attention_qk_runtime_layout",
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

    static let av = MLXFast.metalKernel(
        name: "turboquant_attention_av_runtime_layout",
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

    static let fusedAttention = MLXFast.metalKernel(
        name: "turboquant_attention_fused_decode_runtime_layout",
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
        name: "turboquant_attention_fused_block_partials_runtime_layout",
        inputNames: [
            "q",
            "k_packed", "k_signs", "k_high_mask", "k_residual_signs", "k_scales",
            "v_packed", "v_signs", "v_high_mask", "v_residual_signs", "v_scales",
            "runtime_logical_length",
            "runtime_ring_offset",
            "runtime_pinned_prefix_length",
            "runtime_attention_scale",
        ],
        outputNames: ["partial_stats", "partial_out"],
        source: fusedAttentionBlockPartialsSource,
        header: attentionHeader,
        ensureRowContiguous: false
    )

    static let fusedAttentionGQABlockPartials = MLXFast.metalKernel(
        name: "turboquant_attention_fused_gqa_block_partials_runtime_layout",
        inputNames: [
            "q",
            "k_packed", "k_signs", "k_high_mask", "k_residual_signs", "k_scales",
            "v_packed", "v_signs", "v_high_mask", "v_residual_signs", "v_scales",
            "runtime_logical_length",
            "runtime_ring_offset",
            "runtime_pinned_prefix_length",
            "runtime_attention_scale",
        ],
        outputNames: ["partial_stats", "partial_out"],
        source: fusedAttentionGQABlockPartialsSource,
        header: attentionHeader,
        ensureRowContiguous: false
    )

    static let fusedAttentionBlockReduce = MLXFast.metalKernel(
        name: "turboquant_attention_fused_block_reduce",
        inputNames: ["partial_stats", "partial_out"],
        outputNames: ["out"],
        source: fusedAttentionBlockReduceSource,
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
        scales[scale_base + 2] = 0.0f;

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
                * groups_per_vector + group) * 3u) + scale_index;
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
            return ((batch * kv_heads + head) * capacity + token) * groups_per_vector + group;
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
                    code = tq_read_packed_unsigned(
                        packed, batch, head, token, group, local * key_base_bits,
                        key_base_bits, kv_heads, capacity, groups_per_vector,
                        mag_words_per_group);
                    if (high_precision) {
                        uint extra_code = tq_read_packed_unsigned(
                            packed, batch, head, token, group,
                            group_size * key_base_bits + local,
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
            uint high_count
        ) {
            uint group_start = group * group_size;
            uint count = min(group_size, head_dim - group_start);
            uint storage_group = tq_storage_group_index(
                batch, head, token, group, kv_heads, capacity, groups_per_vector);
            uint repeats = min(pair_repeats, 2u);

            for (uint repeat = 0u; repeat < repeats; repeat++) {
                tq_apply_product_rotation(
                    query_values + repeat * group_size, count, seed, storage_group, false);
            }

            float quantized_dot[2];
            float sign_dot[2];
            quantized_dot[0] = 0.0f;
            quantized_dot[1] = 0.0f;
            sign_dot[0] = 0.0f;
            sign_dot[1] = 0.0f;
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
                    code = tq_read_packed_unsigned(
                        packed, batch, head, token, group, local * key_base_bits,
                        key_base_bits, kv_heads, capacity, groups_per_vector,
                        mag_words_per_group);
                    if (high_precision) {
                        uint extra_code = tq_read_packed_unsigned(
                            packed, batch, head, token, group,
                            group_size * key_base_bits + local,
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
            uint high_count
        ) {
            uint group_start = group * group_size;
            uint count = min(group_size, head_dim - group_start);
            uint storage_group = tq_storage_group_index(
                batch, head, token, group, kv_heads, capacity, groups_per_vector);

            for (uint repeat = 0u; repeat < 4u; repeat++) {
                tq_apply_product_rotation(
                    query_values + repeat * group_size, count, seed, storage_group, false);
            }

            float quantized_dot[4];
            float sign_dot[4];
            for (uint repeat = 0u; repeat < 4u; repeat++) {
                quantized_dot[repeat] = 0.0f;
                sign_dot[repeat] = 0.0f;
            }
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
                    code = tq_read_packed_unsigned(
                        packed, batch, head, token, group, local * key_base_bits,
                        key_base_bits, kv_heads, capacity, groups_per_vector,
                        mag_words_per_group);
                    if (high_precision) {
                        uint extra_code = tq_read_packed_unsigned(
                            packed, batch, head, token, group,
                            group_size * key_base_bits + local,
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
                uint input_index =
                    (((batch * uint(KV_HEADS) + head) * uint(INPUT_LENGTH) + token)
                        * uint(HEAD_DIM)) + dimension;
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
                uint input_index =
                    (((batch * uint(KV_HEADS) + head) * uint(INPUT_LENGTH) + token)
                        * uint(HEAD_DIM)) + dimension;
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
            uint input_index =
                (((batch * uint(KV_HEADS) + head) * uint(INPUT_LENGTH) + token)
                    * uint(HEAD_DIM)) + dimension;
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
        scales[tq_scale_offset(batch, head, token, group, 2u, kv_heads, capacity, groups_per_vector)] = 0.0f;

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
                uint q_index =
                    (((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH) + q_token)
                        * uint(HEAD_DIM)) + dimension;
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
            uint physical_token = tq_physical_token(
                logical_token, uint(CAPACITY), ring_offset, pinned_prefix_length);
            uint weight_index =
                (((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH) + q_token)
                    * logical_length) + logical_token;
            float value = tq_decode_attention_value(
                v_packed, v_signs, v_high_mask, v_residual_signs, v_scales,
                batch, kv_head, physical_token, dimension,
                tq_make_seed(uint(SEED_3), uint(SEED_2), uint(SEED_1), uint(SEED_0)), 1u,
                uint(GROUP_SIZE), uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR),
            uint(MAG_WORDS_PER_GROUP), uint(BITSET_WORDS_PER_GROUP), uint(BASE_BITS), uint(HIGH_BITS),
            uint(VALUE_BITS), uint(KEY_BASE_BITS), uint(KEY_HIGH_BITS), uint(LAYOUT_VERSION),
            uint(HEAD_DIM), 0u,
            decode_scratch);
            sum += float(weights[weight_index]) * value;
        }
        out[index] = static_cast<OUTPUT_DTYPE>(sum);
        """

    private static let fusedAttentionBlockPartialsSource = """
        constexpr uint threads_per_block = uint(THREADS_PER_BLOCK);
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
        threadgroup float tile_weights[512];
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
        ulong value_seed = tq_make_seed(
            uint(VALUE_SEED_3), uint(VALUE_SEED_2),
            uint(VALUE_SEED_1), uint(VALUE_SEED_0));

        if (lane < uint(HEAD_DIM)) {
            uint q_index =
                (((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH) + q_token)
                    * uint(HEAD_DIM)) + lane;
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
        tile_weights[lane] = tile_weight;
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

        if (lane < uint(HEAD_DIM)) {
            thread float decode_scratch[GROUP_SIZE];
            float dimension_accum = 0.0f;
            for (uint tile_lane = 0u; tile_lane < threads_per_block; tile_lane++) {
                float weight = tile_weights[tile_lane];
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
            uint out_index = ((row * uint(BLOCK_COUNT) + block_index) * uint(HEAD_DIM)) + lane;
            partial_out[out_index] = dimension_accum;
        }
        """

    private static let fusedAttentionGQABlockPartialsSource = """
        constexpr uint threads_per_block = uint(THREADS_PER_BLOCK);
        constexpr uint gqa_repeats = uint(GQA_REPEATS);
        uint lane = thread_position_in_threadgroup.x;
        uint group_index = threadgroup_position_in_grid.x;
        uint block_index = group_index % uint(BLOCK_COUNT);
        uint gqa_row = group_index / uint(BLOCK_COUNT);
        uint total_gqa_rows = uint(BATCH_SIZE) * uint(KV_HEADS) * uint(QUERY_LENGTH);
        if (gqa_row >= total_gqa_rows) {
            return;
        }

        threadgroup float partial[4 * 512];
        threadgroup float tile_scores[4 * 512];
        threadgroup float tile_weights[4 * 512];
        threadgroup uint tile_has_weight[512];
        threadgroup uint tile_physical_tokens[512];
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
        ulong value_seed = tq_make_seed(
            uint(VALUE_SEED_3), uint(VALUE_SEED_2),
            uint(VALUE_SEED_1), uint(VALUE_SEED_0));
        uint repeat_count = min(gqa_repeats, 4u);

        if (lane < uint(HEAD_DIM)) {
            for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                uint q_head = kv_head * gqa_repeats + repeat;
                uint q_index =
                    (((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH) + q_token)
                        * uint(HEAD_DIM)) + lane;
                query_cache[repeat * uint(HEAD_DIM) + lane] = float(q[q_index]);
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
                        tq_high_precision_count(count, uint(HIGH_NUMERATOR), uint(HIGH_DENOMINATOR)));
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
                        tq_high_precision_count(count, uint(HIGH_NUMERATOR), uint(HIGH_DENOMINATOR)));
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
        uint has_weight = 0u;
        for (uint repeat = 0u; repeat < repeat_count; repeat++) {
            uint score_base = repeat * threads_per_block;
            tile_maxes[repeat] = partial[score_base];
            float tile_weight = active
                ? exp(tile_scores[score_base + lane] - tile_maxes[repeat])
                : 0.0f;
            tile_weights[score_base + lane] = tile_weight;
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
                uint stat_index = ((row * uint(BLOCK_COUNT) + block_index) * 2u);
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
                            tile_weights[repeat * threads_per_block + tile_lane] * value;
                    }
                }
            }

            for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                uint q_head = kv_head * gqa_repeats + repeat;
                uint row = ((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH)) + q_token;
                uint out_index = ((row * uint(BLOCK_COUNT) + block_index) * uint(HEAD_DIM)) + lane;
                partial_out[out_index] = dimension_accum[repeat];
            }
        }
        """

    private static let fusedAttentionBlockReduceSource = """
        constexpr uint threads_per_block = uint(THREADS_PER_BLOCK);
        uint lane = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;
        if (row >= uint(ROW_COUNT)) {
            return;
        }

        threadgroup float partial[512];

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
            partial[lane] = tile_sum > 0.0f ? exp(partial_stats[stat_index] - row_max) * tile_sum : 0.0f;
        } else {
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
            for (uint block = 0u; block < uint(BLOCK_COUNT); block++) {
                uint stat_index = (row * uint(BLOCK_COUNT) + block) * 2u;
                float tile_sum = partial_stats[stat_index + 1u];
                if (tile_sum > 0.0f) {
                    float tile_scale = exp(partial_stats[stat_index] - row_max);
                    uint partial_index = ((row * uint(BLOCK_COUNT) + block) * uint(HEAD_DIM)) + lane;
                    accum += tile_scale * partial_out[partial_index];
                }
            }
            out[row * uint(HEAD_DIM) + lane] = static_cast<OUTPUT_DTYPE>(
                accum / max(row_sum, 1.17549435e-38f));
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
        threadgroup float tile_weights[256];
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
            uint q_index =
                (((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH) + q_token)
                    * uint(HEAD_DIM)) + lane;
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
            tile_weights[lane] = weight;
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
                    float tile_weight = tile_weights[tile_lane];
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
