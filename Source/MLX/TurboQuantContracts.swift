// Copyright © 2026 RNT56.

import Foundation

public struct TurboQuantKernelCapabilities: Hashable, Codable, Sendable {
    public var flatEncodeDecode: Bool
    public var linearMatmul: Bool
    public var attentionEncode: Bool
    public var attentionDecode: Bool
    public var attentionQK: Bool
    public var attentionAV: Bool
    public var attentionFusedDecode: Bool
    public var attentionTiledFusedDecode: Bool
    public var bfloatOutput: Bool
    public var supportedHeadDimensions: [Int]
    public var selectedKernelProfile: TurboQuantKernelProfile
    public var failureReasons: [String]

    public init(
        flatEncodeDecode: Bool = false,
        linearMatmul: Bool = false,
        attentionEncode: Bool = false,
        attentionDecode: Bool = false,
        attentionQK: Bool = false,
        attentionAV: Bool = false,
        attentionFusedDecode: Bool = false,
        attentionTiledFusedDecode: Bool? = nil,
        bfloatOutput: Bool = false,
        supportedHeadDimensions: [Int] = TurboQuantRuntimeProbeResult
            .throughputOptimizedOnlineFusedHeadDimensions,
        selectedKernelProfile: TurboQuantKernelProfile = .mlxPackedFallback,
        failureReasons: [String] = []
    ) {
        self.flatEncodeDecode = flatEncodeDecode
        self.linearMatmul = linearMatmul
        self.attentionEncode = attentionEncode
        self.attentionDecode = attentionDecode
        self.attentionQK = attentionQK
        self.attentionAV = attentionAV
        self.attentionFusedDecode = attentionFusedDecode
        self.attentionTiledFusedDecode = attentionTiledFusedDecode ?? attentionFusedDecode
        self.bfloatOutput = bfloatOutput
        self.supportedHeadDimensions = supportedHeadDimensions
        self.selectedKernelProfile = selectedKernelProfile
        self.failureReasons = failureReasons
    }

    public var qk: Bool { attentionQK }
    public var av: Bool { attentionAV }
    public var onlineFused: Bool { attentionFusedDecode }
    public var tiledFused: Bool { attentionTiledFusedDecode }

    public var attentionCapabilities: TurboQuantAttentionCapabilities {
        TurboQuantAttentionCapabilities(
            encode: attentionEncode,
            decode: attentionDecode,
            qk: attentionQK,
            av: attentionAV,
            onlineFused: attentionFusedDecode,
            tiledOnlineFused: attentionTiledFusedDecode,
            bfloatOutput: bfloatOutput,
            supportedOnlineFusedHeadDimensions: supportedHeadDimensions
        )
    }

    private enum CodingKeys: String, CodingKey {
        case flatEncodeDecode
        case linearMatmul
        case attentionEncode
        case attentionDecode
        case attentionQK
        case attentionAV
        case attentionFusedDecode
        case attentionTiledFusedDecode
        case bfloatOutput
        case supportedHeadDimensions
        case selectedKernelProfile
        case failureReasons
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let attentionFusedDecode =
            try container.decodeIfPresent(Bool.self, forKey: .attentionFusedDecode) ?? false
        self.init(
            flatEncodeDecode: try container.decodeIfPresent(Bool.self, forKey: .flatEncodeDecode) ?? false,
            linearMatmul: try container.decodeIfPresent(Bool.self, forKey: .linearMatmul) ?? false,
            attentionEncode: try container.decodeIfPresent(Bool.self, forKey: .attentionEncode) ?? false,
            attentionDecode: try container.decodeIfPresent(Bool.self, forKey: .attentionDecode) ?? false,
            attentionQK: try container.decodeIfPresent(Bool.self, forKey: .attentionQK) ?? false,
            attentionAV: try container.decodeIfPresent(Bool.self, forKey: .attentionAV) ?? false,
            attentionFusedDecode: attentionFusedDecode,
            attentionTiledFusedDecode: try container.decodeIfPresent(
                Bool.self,
                forKey: .attentionTiledFusedDecode
            ) ?? attentionFusedDecode,
            bfloatOutput: try container.decodeIfPresent(Bool.self, forKey: .bfloatOutput) ?? false,
            supportedHeadDimensions: try container.decodeIfPresent(
                [Int].self,
                forKey: .supportedHeadDimensions
            ) ?? TurboQuantRuntimeProbeResult.throughputOptimizedOnlineFusedHeadDimensions,
            selectedKernelProfile: try container.decodeIfPresent(
                TurboQuantKernelProfile.self,
                forKey: .selectedKernelProfile
            ) ?? .mlxPackedFallback,
            failureReasons: try container.decodeIfPresent([String].self, forKey: .failureReasons) ?? []
        )
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(flatEncodeDecode, forKey: .flatEncodeDecode)
        try container.encode(linearMatmul, forKey: .linearMatmul)
        try container.encode(attentionEncode, forKey: .attentionEncode)
        try container.encode(attentionDecode, forKey: .attentionDecode)
        try container.encode(attentionQK, forKey: .attentionQK)
        try container.encode(attentionAV, forKey: .attentionAV)
        try container.encode(attentionFusedDecode, forKey: .attentionFusedDecode)
        try container.encode(attentionTiledFusedDecode, forKey: .attentionTiledFusedDecode)
        try container.encode(bfloatOutput, forKey: .bfloatOutput)
        try container.encode(supportedHeadDimensions, forKey: .supportedHeadDimensions)
        try container.encode(selectedKernelProfile, forKey: .selectedKernelProfile)
        try container.encode(failureReasons, forKey: .failureReasons)
    }
}

public typealias RejectedTurboQuantPath = RejectedPath

public struct TurboQuantAttentionDecision: Equatable, Codable, Sendable {
    public var selectedPath: TurboQuantAttentionPath
    public var outputDType: DType
    public var estimatedScratchBytes: Int
    public var rejectedPaths: [RejectedPath]
    public var headDimension: Int?
    public var queryLength: Int?
    public var logicalLength: Int?
    public var dtype: String?
    public var maskKind: String?
    public var kernelProfile: TurboQuantKernelProfile
    public var fallbackReason: String?

    public init(
        selectedPath: TurboQuantAttentionPath,
        outputDType: DType,
        estimatedScratchBytes: Int = 0,
        rejectedPaths: [RejectedPath] = [],
        headDimension: Int? = nil,
        queryLength: Int? = nil,
        logicalLength: Int? = nil,
        dtype: String? = nil,
        maskKind: String? = nil,
        kernelProfile: TurboQuantKernelProfile = .mlxPackedFallback,
        fallbackReason: String? = nil
    ) {
        self.selectedPath = selectedPath
        self.outputDType = outputDType
        self.estimatedScratchBytes = estimatedScratchBytes
        self.rejectedPaths = rejectedPaths
        self.headDimension = headDimension
        self.queryLength = queryLength
        self.logicalLength = logicalLength
        self.dtype = dtype
        self.maskKind = maskKind
        self.kernelProfile = kernelProfile
        self.fallbackReason = fallbackReason
    }

    private enum CodingKeys: String, CodingKey {
        case selectedPath
        case outputDType
        case estimatedScratchBytes
        case rejectedPaths
        case headDimension
        case queryLength
        case logicalLength
        case dtype
        case maskKind
        case kernelProfile
        case fallbackReason
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        selectedPath = try container.decode(TurboQuantAttentionPath.self, forKey: .selectedPath)
        estimatedScratchBytes = try container.decodeIfPresent(Int.self, forKey: .estimatedScratchBytes) ?? 0
        rejectedPaths = try container.decodeIfPresent([RejectedPath].self, forKey: .rejectedPaths) ?? []
        headDimension = try container.decodeIfPresent(Int.self, forKey: .headDimension)
        queryLength = try container.decodeIfPresent(Int.self, forKey: .queryLength)
        logicalLength = try container.decodeIfPresent(Int.self, forKey: .logicalLength)
        dtype = try container.decodeIfPresent(String.self, forKey: .dtype)
        maskKind = try container.decodeIfPresent(String.self, forKey: .maskKind)
        kernelProfile =
            try container.decodeIfPresent(TurboQuantKernelProfile.self, forKey: .kernelProfile)
            ?? .mlxPackedFallback
        fallbackReason = try container.decodeIfPresent(String.self, forKey: .fallbackReason)

        if let outputDTypeString = try? container.decode(String.self, forKey: .outputDType),
           let decoded = DType(turboQuantName: outputDTypeString) {
            outputDType = decoded
        } else {
            outputDType = try container.decode(DType.self, forKey: .outputDType)
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(selectedPath, forKey: .selectedPath)
        try container.encode(outputDType.turboQuantName, forKey: .outputDType)
        try container.encode(estimatedScratchBytes, forKey: .estimatedScratchBytes)
        try container.encode(rejectedPaths, forKey: .rejectedPaths)
        try container.encodeIfPresent(headDimension, forKey: .headDimension)
        try container.encodeIfPresent(queryLength, forKey: .queryLength)
        try container.encodeIfPresent(logicalLength, forKey: .logicalLength)
        try container.encodeIfPresent(dtype, forKey: .dtype)
        try container.encodeIfPresent(maskKind, forKey: .maskKind)
        try container.encode(kernelProfile, forKey: .kernelProfile)
        try container.encodeIfPresent(fallbackReason, forKey: .fallbackReason)
    }
}

private extension DType {
    var turboQuantName: String {
        switch self {
        case .bool: "bool"
        case .uint8: "uint8"
        case .uint16: "uint16"
        case .uint32: "uint32"
        case .uint64: "uint64"
        case .int8: "int8"
        case .int16: "int16"
        case .int32: "int32"
        case .int64: "int64"
        case .float16: "float16"
        case .float32: "float32"
        case .bfloat16: "bfloat16"
        case .complex64: "complex64"
        case .float64: "float64"
        }
    }

    init?(turboQuantName: String) {
        switch turboQuantName {
        case "bool": self = .bool
        case "uint8": self = .uint8
        case "uint16": self = .uint16
        case "uint32": self = .uint32
        case "uint64": self = .uint64
        case "int8": self = .int8
        case "int16": self = .int16
        case "int32": self = .int32
        case "int64": self = .int64
        case "float16": self = .float16
        case "float32": self = .float32
        case "bfloat16": self = .bfloat16
        case "complex64": self = .complex64
        case "float64": self = .float64
        default: return nil
        }
    }
}
