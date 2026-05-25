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
    public var bfloatOutput: Bool

    public init(
        flatEncodeDecode: Bool = false,
        linearMatmul: Bool = false,
        attentionEncode: Bool = false,
        attentionDecode: Bool = false,
        attentionQK: Bool = false,
        attentionAV: Bool = false,
        attentionFusedDecode: Bool = false,
        bfloatOutput: Bool = false
    ) {
        self.flatEncodeDecode = flatEncodeDecode
        self.linearMatmul = linearMatmul
        self.attentionEncode = attentionEncode
        self.attentionDecode = attentionDecode
        self.attentionQK = attentionQK
        self.attentionAV = attentionAV
        self.attentionFusedDecode = attentionFusedDecode
        self.bfloatOutput = bfloatOutput
    }

    public var qk: Bool { attentionQK }
    public var av: Bool { attentionAV }
    public var onlineFused: Bool { attentionFusedDecode }
    public var tiledFused: Bool { attentionFusedDecode }
    public var supportedHeadDimensions: [Int] {
        TurboQuantRuntimeProbeResult.throughputOptimizedOnlineFusedHeadDimensions
    }

    public var attentionCapabilities: TurboQuantAttentionCapabilities {
        TurboQuantAttentionCapabilities(
            encode: attentionEncode,
            decode: attentionDecode,
            qk: attentionQK,
            av: attentionAV,
            onlineFused: attentionFusedDecode,
            tiledOnlineFused: attentionFusedDecode,
            bfloatOutput: bfloatOutput,
            supportedOnlineFusedHeadDimensions: supportedHeadDimensions
        )
    }
}

public typealias RejectedTurboQuantPath = RejectedPath

public struct TurboQuantAttentionDecision: Equatable, Codable, Sendable {
    public var selectedPath: TurboQuantAttentionPath
    public var outputDType: DType
    public var estimatedScratchBytes: Int
    public var rejectedPaths: [RejectedPath]

    public init(
        selectedPath: TurboQuantAttentionPath,
        outputDType: DType,
        estimatedScratchBytes: Int = 0,
        rejectedPaths: [RejectedPath] = []
    ) {
        self.selectedPath = selectedPath
        self.outputDType = outputDType
        self.estimatedScratchBytes = estimatedScratchBytes
        self.rejectedPaths = rejectedPaths
    }
}
