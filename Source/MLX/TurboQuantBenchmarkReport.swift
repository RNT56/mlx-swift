// Copyright © 2026 RNT56.

import Foundation

public struct TurboQuantCoreBenchmarkReport: Codable, Sendable {
    public static let currentSchemaVersion = 1

    public var schemaVersion: Int
    public var mlxSwiftCommit: String?
    public var capabilities: TurboQuantKernelCapabilities
    public var storageEstimate: TurboQuantStorageEstimate
    public var pathDecision: TurboQuantAttentionDecision?
    public var metrics: TurboQuantCoreBenchmarkMetrics
    public var hiddenCopyAudit: TurboQuantHiddenCopyAudit

    public init(
        schemaVersion: Int = TurboQuantCoreBenchmarkReport.currentSchemaVersion,
        mlxSwiftCommit: String? = nil,
        capabilities: TurboQuantKernelCapabilities,
        storageEstimate: TurboQuantStorageEstimate,
        pathDecision: TurboQuantAttentionDecision?,
        metrics: TurboQuantCoreBenchmarkMetrics,
        hiddenCopyAudit: TurboQuantHiddenCopyAudit
    ) {
        self.schemaVersion = schemaVersion
        self.mlxSwiftCommit = mlxSwiftCommit
        self.capabilities = capabilities
        self.storageEstimate = storageEstimate
        self.pathDecision = pathDecision
        self.metrics = metrics
        self.hiddenCopyAudit = hiddenCopyAudit
    }
}

public struct TurboQuantCoreBenchmarkMetrics: Codable, Sendable {
    public var contextTokens: Int
    public var headDimension: Int
    public var queryLength: Int
    public var preset: String
    public var valueBits: Int?
    public var groupSize: Int
    public var layoutVersion: Int?
    public var scaleStorage: String?
    public var warmupIterations: Int?
    public var encodeMS: Double?
    public var decodeMS: Double?
    public var qkMS: Double?
    public var avMS: Double?
    public var fusedMS: Double?
    public var firstTokenLatencyMS: Double?
    public var attentionLatencyMSP50: Double?
    public var attentionLatencyMSP95: Double?
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

    public init(
        contextTokens: Int,
        headDimension: Int,
        queryLength: Int,
        preset: String,
        valueBits: Int?,
        groupSize: Int,
        layoutVersion: Int? = nil,
        scaleStorage: String? = nil,
        warmupIterations: Int? = nil,
        encodeMS: Double? = nil,
        decodeMS: Double? = nil,
        qkMS: Double? = nil,
        avMS: Double? = nil,
        fusedMS: Double? = nil,
        firstTokenLatencyMS: Double? = nil,
        attentionLatencyMSP50: Double? = nil,
        attentionLatencyMSP95: Double? = nil,
        prefillTokensPerSecond: Double? = nil,
        decodeTokensPerSecondP50: Double? = nil,
        decodeTokensPerSecondP95: Double? = nil,
        totalBytes: Int,
        compressedKVBytes: Int? = nil,
        peakMemoryBytes: Int? = nil,
        actualBitsPerValue: Double,
        fallbackUsed: Bool = false,
        fallbackReason: String? = nil,
        memoryWarningsSeen: Int = 0,
        jetsamObserved: Bool = false
    ) {
        self.contextTokens = contextTokens
        self.headDimension = headDimension
        self.queryLength = queryLength
        self.preset = preset
        self.valueBits = valueBits
        self.groupSize = groupSize
        self.layoutVersion = layoutVersion
        self.scaleStorage = scaleStorage
        self.warmupIterations = warmupIterations
        self.encodeMS = encodeMS
        self.decodeMS = decodeMS
        self.qkMS = qkMS
        self.avMS = avMS
        self.fusedMS = fusedMS
        self.firstTokenLatencyMS = firstTokenLatencyMS
        self.attentionLatencyMSP50 = attentionLatencyMSP50
        self.attentionLatencyMSP95 = attentionLatencyMSP95
        self.prefillTokensPerSecond = prefillTokensPerSecond
        self.decodeTokensPerSecondP50 = decodeTokensPerSecondP50
        self.decodeTokensPerSecondP95 = decodeTokensPerSecondP95
        self.totalBytes = Swift.max(0, totalBytes)
        self.compressedKVBytes = Swift.max(0, compressedKVBytes ?? totalBytes)
        self.peakMemoryBytes = peakMemoryBytes
        self.actualBitsPerValue = actualBitsPerValue
        self.fallbackUsed = fallbackUsed
        self.fallbackReason = fallbackReason
        self.memoryWarningsSeen = Swift.max(0, memoryWarningsSeen)
        self.jetsamObserved = jetsamObserved
    }
}

public enum TurboQuantHiddenCopyAuditStatus: String, Codable, Sendable {
    case pass
    case warning
    case fail
    case pending
    case skipped
}

public struct TurboQuantHiddenCopyAudit: Codable, Sendable {
    public var status: TurboQuantHiddenCopyAuditStatus
    public var entries: [TurboQuantHiddenCopyAuditEntry]
    public var notes: [String]

    public init(
        status: TurboQuantHiddenCopyAuditStatus,
        entries: [TurboQuantHiddenCopyAuditEntry],
        notes: [String] = []
    ) {
        self.status = status
        self.entries = entries
        self.notes = notes
    }

    public static let currentW3 = TurboQuantHiddenCopyAudit(
        status: .pass,
        entries: [
            TurboQuantHiddenCopyAuditEntry(
                kernelName: "encode flat",
                largeInput: "source K/V chunk",
                copyRisk: "low",
                mitigation: "benchmark input is chunk-bounded; no long-cache array is prepared",
                status: "audited-bounded"
            ),
            TurboQuantHiddenCopyAuditEntry(
                kernelName: "decode flat",
                largeInput: "compressed code",
                copyRisk: "medium",
                mitigation: "canonical storage validation rejects non-row-contiguous code arrays",
                status: "guarded"
            ),
            TurboQuantHiddenCopyAuditEntry(
                kernelName: "compressed QK",
                largeInput: "compressed K cache",
                copyRisk: "high",
                mitigation: "canonical compressed storage validation runs before dispatch; no decoded K cache is materialized",
                status: "guarded"
            ),
            TurboQuantHiddenCopyAuditEntry(
                kernelName: "compressed AV",
                largeInput: "compressed V cache",
                copyRisk: "high",
                mitigation: "canonical compressed storage validation runs before dispatch; attention weights must already be row-contiguous",
                status: "guarded"
            ),
            TurboQuantHiddenCopyAuditEntry(
                kernelName: "online fused",
                largeInput: "compressed K/V cache",
                copyRisk: "high",
                mitigation: "fused dispatch consumes canonical compressed K/V arrays directly and does not decode a full cache",
                status: "guarded"
            ),
            TurboQuantHiddenCopyAuditEntry(
                kernelName: "tiled fused",
                largeInput: "compressed K/V cache",
                copyRisk: "high",
                mitigation: "tiled path shares fused dispatch guards; non-canonical compressed storage is rejected before launch",
                status: "guarded"
            ),
        ],
        notes: [
            "Query tensors may require bounded row-contiguous preparation; compressed K/V cache tensors must not be copied into decoded full-cache arrays.",
            "This is a source audit and validation-gate status, not a production verification claim.",
        ]
    )

    public static let currentW5 = TurboQuantHiddenCopyAudit(
        status: .pass,
        entries: currentW3.entries + [
            TurboQuantHiddenCopyAuditEntry(
                kernelName: "layout V5 fp16 scales",
                largeInput: "compressed scale tables",
                copyRisk: "medium",
                mitigation: "V5 scale tables are emitted directly by the Metal encode kernel and validated as canonical float16/float32 storage before dispatch",
                status: "guarded"
            ),
            TurboQuantHiddenCopyAuditEntry(
                kernelName: "block-parallel fused",
                largeInput: "compressed K/V cache",
                copyRisk: "high",
                mitigation: "block partial kernels consume canonical compressed K/V arrays and emit bounded per-block partials, not decoded cache copies",
                status: "guarded"
            ),
        ],
        notes: currentW3.notes + [
            "Layout V5 is the default write layout; V4 remains supported for compatibility comparisons.",
        ]
    )
}

public struct TurboQuantHiddenCopyAuditEntry: Codable, Sendable {
    public var kernelName: String
    public var largeInput: String
    public var copyRisk: String
    public var mitigation: String
    public var status: String

    public init(
        kernelName: String,
        largeInput: String,
        copyRisk: String,
        mitigation: String,
        status: String
    ) {
        self.kernelName = kernelName
        self.largeInput = largeInput
        self.copyRisk = copyRisk
        self.mitigation = mitigation
        self.status = status
    }
}
