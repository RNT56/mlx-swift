// Copyright © 2026 RNT56.

import Foundation

public struct TurboQuantCoreBenchmarkReport: Codable, Sendable {
    public static let currentSchemaVersion = 1

    public var schemaVersion: Int
    public var mlxSwiftCommit: String?
    public var capabilities: TurboQuantKernelCapabilities
    public var storageEstimate: TurboQuantStorageEstimate
    public var pathDecision: TurboQuantAttentionDecision?
    public var pathMeasurements: [TurboQuantCoreBenchmarkPathMeasurement]
    public var metrics: TurboQuantCoreBenchmarkMetrics
    public var hiddenCopyAudit: TurboQuantHiddenCopyAudit
    // Provenance guardrails: this report comes from a synthetic sinusoid-input
    // kernel microbench with no checkpoint loaded, so it is not real-model and
    // not promotable. Emitted so downstream consumers cannot mistake it for a
    // real-model measurement.
    public var synthetic: Bool
    public var realModel: Bool

    public init(
        schemaVersion: Int = TurboQuantCoreBenchmarkReport.currentSchemaVersion,
        mlxSwiftCommit: String? = nil,
        capabilities: TurboQuantKernelCapabilities,
        storageEstimate: TurboQuantStorageEstimate,
        pathDecision: TurboQuantAttentionDecision?,
        pathMeasurements: [TurboQuantCoreBenchmarkPathMeasurement] = [],
        metrics: TurboQuantCoreBenchmarkMetrics,
        hiddenCopyAudit: TurboQuantHiddenCopyAudit,
        synthetic: Bool = true,
        realModel: Bool = false
    ) {
        self.schemaVersion = schemaVersion
        self.mlxSwiftCommit = mlxSwiftCommit
        self.capabilities = capabilities
        self.storageEstimate = storageEstimate
        self.pathDecision = pathDecision
        self.pathMeasurements = pathMeasurements
        self.metrics = metrics
        self.hiddenCopyAudit = hiddenCopyAudit
        self.synthetic = synthetic
        self.realModel = realModel
    }

    private enum CodingKeys: String, CodingKey {
        case schemaVersion
        case mlxSwiftCommit
        case capabilities
        case storageEstimate
        case pathDecision
        case pathMeasurements
        case metrics
        case hiddenCopyAudit
        case synthetic
        case realModel
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            schemaVersion: try container.decode(Int.self, forKey: .schemaVersion),
            mlxSwiftCommit: try container.decodeIfPresent(String.self, forKey: .mlxSwiftCommit),
            capabilities: try container.decode(TurboQuantKernelCapabilities.self, forKey: .capabilities),
            storageEstimate: try container.decode(TurboQuantStorageEstimate.self, forKey: .storageEstimate),
            pathDecision: try container.decodeIfPresent(
                TurboQuantAttentionDecision.self, forKey: .pathDecision),
            pathMeasurements: try container.decodeIfPresent(
                [TurboQuantCoreBenchmarkPathMeasurement].self, forKey: .pathMeasurements) ?? [],
            metrics: try container.decode(TurboQuantCoreBenchmarkMetrics.self, forKey: .metrics),
            hiddenCopyAudit: try container.decode(TurboQuantHiddenCopyAudit.self, forKey: .hiddenCopyAudit),
            synthetic: try container.decodeIfPresent(Bool.self, forKey: .synthetic) ?? true,
            realModel: try container.decodeIfPresent(Bool.self, forKey: .realModel) ?? false
        )
    }
}

public enum TurboQuantCoreBenchmarkPathStatus: String, Codable, Sendable {
    case measured
    case reference
    case skipped
    case failed
    case notCallable
    case unavailable
}

public struct TurboQuantCoreBenchmarkPathMeasurement: Codable, Sendable {
    public var path: TurboQuantAttentionPath
    public var route: String
    public var backend: String
    public var status: TurboQuantCoreBenchmarkPathStatus
    public var selected: Bool
    public var validForRequest: Bool
    public var referenceDType: String?
    public var reason: String?
    public var latencyMSAverage: Double?
    public var latencyMSP50: Double?
    public var latencyMSP95: Double?
    public var decodeTokensPerSecondP50: Double?
    public var decodeTokensPerSecondP95: Double?
    public var rawSDPALatencyMSP50: Double?
    public var rawSDPALatencyMSP95: Double?
    public var rawSDPADecodeTokensPerSecondP50: Double?
    public var rawSDPADecodeTokensPerSecondP95: Double?
    public var speedRatioToRawSDPAP50: Double?
    public var speedRatioToRawSDPAP95: Double?
    public var compressedKVBytes: Int?
    public var rawSDPAKVBytes: Int?
    public var memoryBytesSavedVsRawSDPA: Int?
    public var memoryReductionRatio: Double?
    public var memoryReductionPercent: Double?
    public var actualBitsPerValue: Double?
    public var maxAbsoluteErrorVsRawSDPA: Double?
    public var cosineSimilarityVsRawSDPA: Double?

    public init(
        path: TurboQuantAttentionPath,
        route: String,
        backend: String,
        status: TurboQuantCoreBenchmarkPathStatus,
        selected: Bool = false,
        validForRequest: Bool = false,
        referenceDType: String? = nil,
        reason: String? = nil,
        latencyMSAverage: Double? = nil,
        latencyMSP50: Double? = nil,
        latencyMSP95: Double? = nil,
        decodeTokensPerSecondP50: Double? = nil,
        decodeTokensPerSecondP95: Double? = nil,
        rawSDPALatencyMSP50: Double? = nil,
        rawSDPALatencyMSP95: Double? = nil,
        rawSDPADecodeTokensPerSecondP50: Double? = nil,
        rawSDPADecodeTokensPerSecondP95: Double? = nil,
        speedRatioToRawSDPAP50: Double? = nil,
        speedRatioToRawSDPAP95: Double? = nil,
        compressedKVBytes: Int? = nil,
        rawSDPAKVBytes: Int? = nil,
        memoryBytesSavedVsRawSDPA: Int? = nil,
        memoryReductionRatio: Double? = nil,
        memoryReductionPercent: Double? = nil,
        actualBitsPerValue: Double? = nil,
        maxAbsoluteErrorVsRawSDPA: Double? = nil,
        cosineSimilarityVsRawSDPA: Double? = nil
    ) {
        self.path = path
        self.route = route
        self.backend = backend
        self.status = status
        self.selected = selected
        self.validForRequest = validForRequest
        self.referenceDType = referenceDType
        self.reason = reason
        self.latencyMSAverage = latencyMSAverage.map { Swift.max(0, $0) }
        self.latencyMSP50 = latencyMSP50.map { Swift.max(0, $0) }
        self.latencyMSP95 = latencyMSP95.map { Swift.max(0, $0) }
        self.decodeTokensPerSecondP50 = decodeTokensPerSecondP50.map { Swift.max(0, $0) }
        self.decodeTokensPerSecondP95 = decodeTokensPerSecondP95.map { Swift.max(0, $0) }
        self.rawSDPALatencyMSP50 = rawSDPALatencyMSP50.map { Swift.max(0, $0) }
        self.rawSDPALatencyMSP95 = rawSDPALatencyMSP95.map { Swift.max(0, $0) }
        self.rawSDPADecodeTokensPerSecondP50 = rawSDPADecodeTokensPerSecondP50.map {
            Swift.max(0, $0)
        }
        self.rawSDPADecodeTokensPerSecondP95 = rawSDPADecodeTokensPerSecondP95.map {
            Swift.max(0, $0)
        }
        self.speedRatioToRawSDPAP50 = speedRatioToRawSDPAP50.map { Swift.max(0, $0) }
        self.speedRatioToRawSDPAP95 = speedRatioToRawSDPAP95.map { Swift.max(0, $0) }
        self.compressedKVBytes = compressedKVBytes.map { Swift.max(0, $0) }
        self.rawSDPAKVBytes = rawSDPAKVBytes.map { Swift.max(0, $0) }
        self.memoryBytesSavedVsRawSDPA = memoryBytesSavedVsRawSDPA.map { Swift.max(0, $0) }
        self.memoryReductionRatio = memoryReductionRatio.map { Swift.max(0, $0) }
        self.memoryReductionPercent = memoryReductionPercent.map { Swift.max(0, $0) }
        self.actualBitsPerValue = actualBitsPerValue.map { Swift.max(0, $0) }
        self.maxAbsoluteErrorVsRawSDPA = maxAbsoluteErrorVsRawSDPA.map { Swift.max(0, $0) }
        self.cosineSimilarityVsRawSDPA = cosineSimilarityVsRawSDPA.map {
            Swift.max(-1, Swift.min(1, $0))
        }
    }
}

public struct TurboQuantCoreBenchmarkMetrics: Codable, Sendable {
    public var route: String?
    public var runtimeMode: String?
    public var backend: String?
    public var kernelFlags: TurboQuantBenchmarkKernelFlags?
    public var sparseVEnabled: Bool
    public var sparseVSelectionMode: TurboQuantSparseVSelectionMode?
    public var sparseVThreshold: Float?
    public var sparseVSkippedValueTokens: Int?
    public var sparseVConsideredValueTokens: Int?
    public var sparseVRecentTokenCount: Int?
    public var sparseVOlderTokenCount: Int?
    public var sparseVPageCandidateCount: Int?
    public var sparseVSkipRatio: Double
    public var sparseVRetainedAttentionMass: Double?
    public var sparseVMaxOutputErrorVsDenseReference: Double?
    public var sparseVCosineVsDenseReference: Double?
    public var sparseVFallbackReason: String?
    public var sparseVDiagnostics: [TurboQuantSparseVDiagnostic]?
    public var lowerVAndSparseV: TurboQuantLowerVAndSparseVReport?
    public var boundaryProtectedLayerCount: Int
    public var boundaryProtectionReason: String?
    public var selectedBudgetedColdTokens: Int?
    public var anchorColdTokens: Int?
    public var anchorOverflowTokens: Int?
    public var maxColdBudgetTokens: Int?
    public var selectorInitialConfidence: Double?
    public var selectorFinalConfidence: Double?
    public var selectorEscalation: String?
    public var selectorReasonFlags: [String]?
    public var contextTokens: Int
    public var headDimension: Int
    public var queryLength: Int
    public var preset: String
    public var valueBits: Int?
    public var groupSize: Int
    public var layoutVersion: Int?
    public var scaleStorage: String?
    public var hotTokens: Int?
    public var coldBlockCount: Int?
    public var selectedColdTokens: Int?
    public var coldBudgetTokens: Int?
    public var selectorConfidence: Double?
    public var fullScanFallbackCount: Int?
    public var blockParallelTokenBlockSize: Int?
    public var recommendedBlockParallelTokenBlockSize: Int?
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
    public var plainAttentionLatencyMSP50: Double?
    public var plainAttentionLatencyMSP95: Double?
    public var plainDecodeTokensPerSecondP50: Double?
    public var plainDecodeTokensPerSecondP95: Double?
    public var speedRatioToPlainP50: Double?
    public var speedRatioToPlainP95: Double?
    public var rawSDPAReferenceDType: String?
    public var rawSDPAAttentionLatencyMSP50: Double?
    public var rawSDPAAttentionLatencyMSP95: Double?
    public var rawSDPADecodeTokensPerSecondP50: Double?
    public var rawSDPADecodeTokensPerSecondP95: Double?
    public var speedRatioToRawSDPAP50: Double?
    public var speedRatioToRawSDPAP95: Double?
    public var totalBytes: Int
    public var compressedKVBytes: Int
    public var plainKVBytes: Int?
    public var rawSDPAKVBytes: Int?
    public var memoryBytesSavedVsRawSDPA: Int?
    public var memoryReductionRatio: Double?
    public var memoryReductionPercent: Double?
    public var peakMemoryBytes: Int?
    public var actualBitsPerValue: Double
    public var fallbackUsed: Bool
    public var fallbackReason: String?
    public var memoryWarningsSeen: Int
    public var jetsamObserved: Bool
    public var cooldownMS: Int?
    public var pathCooldownMS: Int?

    public init(
        route: String? = nil,
        runtimeMode: String? = nil,
        backend: String? = nil,
        kernelFlags: TurboQuantBenchmarkKernelFlags? = nil,
        sparseVEnabled: Bool = false,
        sparseVSelectionMode: TurboQuantSparseVSelectionMode? = nil,
        sparseVThreshold: Float? = nil,
        sparseVSkippedValueTokens: Int? = nil,
        sparseVConsideredValueTokens: Int? = nil,
        sparseVRecentTokenCount: Int? = nil,
        sparseVOlderTokenCount: Int? = nil,
        sparseVPageCandidateCount: Int? = nil,
        sparseVSkipRatio: Double = 0,
        sparseVRetainedAttentionMass: Double? = nil,
        sparseVMaxOutputErrorVsDenseReference: Double? = nil,
        sparseVCosineVsDenseReference: Double? = nil,
        sparseVFallbackReason: String? = nil,
        sparseVDiagnostics: [TurboQuantSparseVDiagnostic]? = nil,
        lowerVAndSparseV: TurboQuantLowerVAndSparseVReport? = nil,
        boundaryProtectedLayerCount: Int = 0,
        boundaryProtectionReason: String? = nil,
        contextTokens: Int,
        headDimension: Int,
        queryLength: Int,
        preset: String,
        valueBits: Int?,
        groupSize: Int,
        layoutVersion: Int? = nil,
        scaleStorage: String? = nil,
        hotTokens: Int? = nil,
        coldBlockCount: Int? = nil,
        selectedColdTokens: Int? = nil,
        coldBudgetTokens: Int? = nil,
        selectorConfidence: Double? = nil,
        selectedBudgetedColdTokens: Int? = nil,
        anchorColdTokens: Int? = nil,
        anchorOverflowTokens: Int? = nil,
        maxColdBudgetTokens: Int? = nil,
        selectorInitialConfidence: Double? = nil,
        selectorFinalConfidence: Double? = nil,
        selectorEscalation: String? = nil,
        selectorReasonFlags: [String]? = nil,
        fullScanFallbackCount: Int? = nil,
        blockParallelTokenBlockSize: Int? = nil,
        recommendedBlockParallelTokenBlockSize: Int? = nil,
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
        plainAttentionLatencyMSP50: Double? = nil,
        plainAttentionLatencyMSP95: Double? = nil,
        plainDecodeTokensPerSecondP50: Double? = nil,
        plainDecodeTokensPerSecondP95: Double? = nil,
        speedRatioToPlainP50: Double? = nil,
        speedRatioToPlainP95: Double? = nil,
        rawSDPAReferenceDType: String? = nil,
        rawSDPAAttentionLatencyMSP50: Double? = nil,
        rawSDPAAttentionLatencyMSP95: Double? = nil,
        rawSDPADecodeTokensPerSecondP50: Double? = nil,
        rawSDPADecodeTokensPerSecondP95: Double? = nil,
        speedRatioToRawSDPAP50: Double? = nil,
        speedRatioToRawSDPAP95: Double? = nil,
        totalBytes: Int,
        compressedKVBytes: Int? = nil,
        plainKVBytes: Int? = nil,
        rawSDPAKVBytes: Int? = nil,
        memoryBytesSavedVsRawSDPA: Int? = nil,
        memoryReductionRatio: Double? = nil,
        memoryReductionPercent: Double? = nil,
        peakMemoryBytes: Int? = nil,
        actualBitsPerValue: Double,
        fallbackUsed: Bool = false,
        fallbackReason: String? = nil,
        memoryWarningsSeen: Int = 0,
        jetsamObserved: Bool = false,
        cooldownMS: Int? = nil,
        pathCooldownMS: Int? = nil
    ) {
        self.route = route
        self.runtimeMode = runtimeMode
        self.backend = backend
        self.kernelFlags = kernelFlags
        self.sparseVEnabled = sparseVEnabled
        self.sparseVSelectionMode = sparseVSelectionMode
        self.sparseVThreshold = sparseVThreshold
        self.sparseVSkippedValueTokens = sparseVSkippedValueTokens.map { Swift.max(0, $0) }
        self.sparseVConsideredValueTokens = sparseVConsideredValueTokens.map { Swift.max(0, $0) }
        self.sparseVRecentTokenCount = sparseVRecentTokenCount.map { Swift.max(0, $0) }
        self.sparseVOlderTokenCount = sparseVOlderTokenCount.map { Swift.max(0, $0) }
        self.sparseVPageCandidateCount = sparseVPageCandidateCount.map { Swift.max(0, $0) }
        self.sparseVSkipRatio = Swift.max(0, Swift.min(1, sparseVSkipRatio))
        self.sparseVRetainedAttentionMass = sparseVRetainedAttentionMass.map {
            Swift.max(0, Swift.min(1, $0))
        }
        self.sparseVMaxOutputErrorVsDenseReference = sparseVMaxOutputErrorVsDenseReference.map {
            Swift.max(0, $0)
        }
        self.sparseVCosineVsDenseReference = sparseVCosineVsDenseReference.map {
            Swift.max(-1, Swift.min(1, $0))
        }
        self.sparseVFallbackReason = sparseVFallbackReason
        self.sparseVDiagnostics = sparseVDiagnostics
        self.lowerVAndSparseV = lowerVAndSparseV
        self.boundaryProtectedLayerCount = Swift.max(0, boundaryProtectedLayerCount)
        self.boundaryProtectionReason = boundaryProtectionReason
        self.contextTokens = contextTokens
        self.headDimension = headDimension
        self.queryLength = queryLength
        self.preset = preset
        self.valueBits = valueBits
        self.groupSize = groupSize
        self.layoutVersion = layoutVersion
        self.scaleStorage = scaleStorage
        self.hotTokens = hotTokens
        self.coldBlockCount = coldBlockCount
        self.selectedColdTokens = selectedColdTokens
        self.coldBudgetTokens = coldBudgetTokens
        self.selectorConfidence = selectorConfidence
        self.selectedBudgetedColdTokens = selectedBudgetedColdTokens.map { Swift.max(0, $0) }
        self.anchorColdTokens = anchorColdTokens.map { Swift.max(0, $0) }
        self.anchorOverflowTokens = anchorOverflowTokens.map { Swift.max(0, $0) }
        self.maxColdBudgetTokens = maxColdBudgetTokens.map { Swift.max(0, $0) }
        self.selectorInitialConfidence = selectorInitialConfidence
        self.selectorFinalConfidence = selectorFinalConfidence
        self.selectorEscalation = selectorEscalation
        self.selectorReasonFlags = selectorReasonFlags
        self.fullScanFallbackCount = fullScanFallbackCount
        self.blockParallelTokenBlockSize = blockParallelTokenBlockSize
        self.recommendedBlockParallelTokenBlockSize = recommendedBlockParallelTokenBlockSize
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
        self.plainAttentionLatencyMSP50 = plainAttentionLatencyMSP50
        self.plainAttentionLatencyMSP95 = plainAttentionLatencyMSP95
        self.plainDecodeTokensPerSecondP50 = plainDecodeTokensPerSecondP50
        self.plainDecodeTokensPerSecondP95 = plainDecodeTokensPerSecondP95
        self.speedRatioToPlainP50 = speedRatioToPlainP50
        self.speedRatioToPlainP95 = speedRatioToPlainP95
        self.rawSDPAReferenceDType = rawSDPAReferenceDType
        self.rawSDPAAttentionLatencyMSP50 = rawSDPAAttentionLatencyMSP50
        self.rawSDPAAttentionLatencyMSP95 = rawSDPAAttentionLatencyMSP95
        self.rawSDPADecodeTokensPerSecondP50 = rawSDPADecodeTokensPerSecondP50
        self.rawSDPADecodeTokensPerSecondP95 = rawSDPADecodeTokensPerSecondP95
        self.speedRatioToRawSDPAP50 = speedRatioToRawSDPAP50
        self.speedRatioToRawSDPAP95 = speedRatioToRawSDPAP95
        self.totalBytes = Swift.max(0, totalBytes)
        self.compressedKVBytes = Swift.max(0, compressedKVBytes ?? totalBytes)
        self.plainKVBytes = plainKVBytes.map { Swift.max(0, $0) }
        self.rawSDPAKVBytes = rawSDPAKVBytes.map { Swift.max(0, $0) }
        self.memoryBytesSavedVsRawSDPA = memoryBytesSavedVsRawSDPA.map { Swift.max(0, $0) }
        self.memoryReductionRatio = memoryReductionRatio.map { Swift.max(0, $0) }
        self.memoryReductionPercent = memoryReductionPercent.map { Swift.max(0, $0) }
        self.peakMemoryBytes = peakMemoryBytes
        self.actualBitsPerValue = actualBitsPerValue
        self.fallbackUsed = fallbackUsed
        self.fallbackReason = fallbackReason
        self.memoryWarningsSeen = Swift.max(0, memoryWarningsSeen)
        self.jetsamObserved = jetsamObserved
        self.cooldownMS = cooldownMS.map { Swift.max(0, $0) }
        self.pathCooldownMS = pathCooldownMS.map { Swift.max(0, $0) }
    }

    private enum CodingKeys: String, CodingKey {
        case route
        case runtimeMode
        case backend
        case kernelFlags
        case sparseVEnabled
        case sparseVSelectionMode
        case sparseVThreshold
        case sparseVSkippedValueTokens
        case sparseVConsideredValueTokens
        case sparseVRecentTokenCount
        case sparseVOlderTokenCount
        case sparseVPageCandidateCount
        case sparseVSkipRatio
        case sparseVRetainedAttentionMass
        case sparseVMaxOutputErrorVsDenseReference
        case sparseVCosineVsDenseReference
        case sparseVFallbackReason
        case sparseVDiagnostics
        case lowerVAndSparseV
        case boundaryProtectedLayerCount
        case boundaryProtectionReason
        case selectedBudgetedColdTokens
        case anchorColdTokens
        case anchorOverflowTokens
        case maxColdBudgetTokens
        case selectorInitialConfidence
        case selectorFinalConfidence
        case selectorEscalation
        case selectorReasonFlags
        case contextTokens
        case headDimension
        case queryLength
        case preset
        case valueBits
        case groupSize
        case layoutVersion
        case scaleStorage
        case hotTokens
        case coldBlockCount
        case selectedColdTokens
        case coldBudgetTokens
        case selectorConfidence
        case fullScanFallbackCount
        case blockParallelTokenBlockSize
        case recommendedBlockParallelTokenBlockSize
        case warmupIterations
        case encodeMS
        case decodeMS
        case qkMS
        case avMS
        case fusedMS
        case firstTokenLatencyMS
        case attentionLatencyMSP50
        case attentionLatencyMSP95
        case prefillTokensPerSecond
        case decodeTokensPerSecondP50
        case decodeTokensPerSecondP95
        case plainAttentionLatencyMSP50
        case plainAttentionLatencyMSP95
        case plainDecodeTokensPerSecondP50
        case plainDecodeTokensPerSecondP95
        case speedRatioToPlainP50
        case speedRatioToPlainP95
        case rawSDPAReferenceDType
        case rawSDPAAttentionLatencyMSP50
        case rawSDPAAttentionLatencyMSP95
        case rawSDPADecodeTokensPerSecondP50
        case rawSDPADecodeTokensPerSecondP95
        case speedRatioToRawSDPAP50
        case speedRatioToRawSDPAP95
        case totalBytes
        case compressedKVBytes
        case plainKVBytes
        case rawSDPAKVBytes
        case memoryBytesSavedVsRawSDPA
        case memoryReductionRatio
        case memoryReductionPercent
        case peakMemoryBytes
        case actualBitsPerValue
        case fallbackUsed
        case fallbackReason
        case memoryWarningsSeen
        case jetsamObserved
        case cooldownMS
        case pathCooldownMS
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let totalBytes = try container.decode(Int.self, forKey: .totalBytes)
        self.init(
            route: try container.decodeIfPresent(String.self, forKey: .route),
            runtimeMode: try container.decodeIfPresent(String.self, forKey: .runtimeMode),
            backend: try container.decodeIfPresent(String.self, forKey: .backend),
            kernelFlags: try container.decodeIfPresent(
                TurboQuantBenchmarkKernelFlags.self, forKey: .kernelFlags),
            sparseVEnabled: try container.decodeIfPresent(Bool.self, forKey: .sparseVEnabled)
                ?? false,
            sparseVSelectionMode: try container.decodeIfPresent(
                TurboQuantSparseVSelectionMode.self, forKey: .sparseVSelectionMode),
            sparseVThreshold: try container.decodeIfPresent(Float.self, forKey: .sparseVThreshold),
            sparseVSkippedValueTokens: try container.decodeIfPresent(
                Int.self, forKey: .sparseVSkippedValueTokens),
            sparseVConsideredValueTokens: try container.decodeIfPresent(
                Int.self, forKey: .sparseVConsideredValueTokens),
            sparseVRecentTokenCount: try container.decodeIfPresent(
                Int.self, forKey: .sparseVRecentTokenCount),
            sparseVOlderTokenCount: try container.decodeIfPresent(
                Int.self, forKey: .sparseVOlderTokenCount),
            sparseVPageCandidateCount: try container.decodeIfPresent(
                Int.self, forKey: .sparseVPageCandidateCount),
            sparseVSkipRatio: try container.decodeIfPresent(Double.self, forKey: .sparseVSkipRatio)
                ?? 0,
            sparseVRetainedAttentionMass: try container.decodeIfPresent(
                Double.self, forKey: .sparseVRetainedAttentionMass),
            sparseVMaxOutputErrorVsDenseReference: try container.decodeIfPresent(
                Double.self, forKey: .sparseVMaxOutputErrorVsDenseReference),
            sparseVCosineVsDenseReference: try container.decodeIfPresent(
                Double.self, forKey: .sparseVCosineVsDenseReference),
            sparseVFallbackReason: try container.decodeIfPresent(
                String.self, forKey: .sparseVFallbackReason),
            sparseVDiagnostics: try container.decodeIfPresent(
                [TurboQuantSparseVDiagnostic].self, forKey: .sparseVDiagnostics),
            lowerVAndSparseV: try container.decodeIfPresent(
                TurboQuantLowerVAndSparseVReport.self, forKey: .lowerVAndSparseV),
            boundaryProtectedLayerCount: try container.decodeIfPresent(
                Int.self, forKey: .boundaryProtectedLayerCount) ?? 0,
            boundaryProtectionReason: try container.decodeIfPresent(
                String.self, forKey: .boundaryProtectionReason),
            contextTokens: try container.decode(Int.self, forKey: .contextTokens),
            headDimension: try container.decode(Int.self, forKey: .headDimension),
            queryLength: try container.decode(Int.self, forKey: .queryLength),
            preset: try container.decode(String.self, forKey: .preset),
            valueBits: try container.decodeIfPresent(Int.self, forKey: .valueBits),
            groupSize: try container.decode(Int.self, forKey: .groupSize),
            layoutVersion: try container.decodeIfPresent(Int.self, forKey: .layoutVersion),
            scaleStorage: try container.decodeIfPresent(String.self, forKey: .scaleStorage),
            hotTokens: try container.decodeIfPresent(Int.self, forKey: .hotTokens),
            coldBlockCount: try container.decodeIfPresent(Int.self, forKey: .coldBlockCount),
            selectedColdTokens: try container.decodeIfPresent(
                Int.self, forKey: .selectedColdTokens),
            coldBudgetTokens: try container.decodeIfPresent(Int.self, forKey: .coldBudgetTokens),
            selectorConfidence: try container.decodeIfPresent(
                Double.self, forKey: .selectorConfidence),
            selectedBudgetedColdTokens: try container.decodeIfPresent(
                Int.self, forKey: .selectedBudgetedColdTokens),
            anchorColdTokens: try container.decodeIfPresent(Int.self, forKey: .anchorColdTokens),
            anchorOverflowTokens: try container.decodeIfPresent(
                Int.self, forKey: .anchorOverflowTokens),
            maxColdBudgetTokens: try container.decodeIfPresent(
                Int.self, forKey: .maxColdBudgetTokens),
            selectorInitialConfidence: try container.decodeIfPresent(
                Double.self, forKey: .selectorInitialConfidence),
            selectorFinalConfidence: try container.decodeIfPresent(
                Double.self, forKey: .selectorFinalConfidence),
            selectorEscalation: try container.decodeIfPresent(
                String.self, forKey: .selectorEscalation),
            selectorReasonFlags: try container.decodeIfPresent(
                [String].self, forKey: .selectorReasonFlags),
            fullScanFallbackCount: try container.decodeIfPresent(
                Int.self, forKey: .fullScanFallbackCount),
            blockParallelTokenBlockSize: try container.decodeIfPresent(
                Int.self, forKey: .blockParallelTokenBlockSize),
            recommendedBlockParallelTokenBlockSize: try container.decodeIfPresent(
                Int.self, forKey: .recommendedBlockParallelTokenBlockSize),
            warmupIterations: try container.decodeIfPresent(Int.self, forKey: .warmupIterations),
            encodeMS: try container.decodeIfPresent(Double.self, forKey: .encodeMS),
            decodeMS: try container.decodeIfPresent(Double.self, forKey: .decodeMS),
            qkMS: try container.decodeIfPresent(Double.self, forKey: .qkMS),
            avMS: try container.decodeIfPresent(Double.self, forKey: .avMS),
            fusedMS: try container.decodeIfPresent(Double.self, forKey: .fusedMS),
            firstTokenLatencyMS: try container.decodeIfPresent(
                Double.self, forKey: .firstTokenLatencyMS),
            attentionLatencyMSP50: try container.decodeIfPresent(
                Double.self, forKey: .attentionLatencyMSP50),
            attentionLatencyMSP95: try container.decodeIfPresent(
                Double.self, forKey: .attentionLatencyMSP95),
            prefillTokensPerSecond: try container.decodeIfPresent(
                Double.self, forKey: .prefillTokensPerSecond),
            decodeTokensPerSecondP50: try container.decodeIfPresent(
                Double.self, forKey: .decodeTokensPerSecondP50),
            decodeTokensPerSecondP95: try container.decodeIfPresent(
                Double.self, forKey: .decodeTokensPerSecondP95),
            plainAttentionLatencyMSP50: try container.decodeIfPresent(
                Double.self, forKey: .plainAttentionLatencyMSP50),
            plainAttentionLatencyMSP95: try container.decodeIfPresent(
                Double.self, forKey: .plainAttentionLatencyMSP95),
            plainDecodeTokensPerSecondP50: try container.decodeIfPresent(
                Double.self, forKey: .plainDecodeTokensPerSecondP50),
            plainDecodeTokensPerSecondP95: try container.decodeIfPresent(
                Double.self, forKey: .plainDecodeTokensPerSecondP95),
            speedRatioToPlainP50: try container.decodeIfPresent(
                Double.self, forKey: .speedRatioToPlainP50),
            speedRatioToPlainP95: try container.decodeIfPresent(
                Double.self, forKey: .speedRatioToPlainP95),
            rawSDPAReferenceDType: try container.decodeIfPresent(
                String.self, forKey: .rawSDPAReferenceDType),
            rawSDPAAttentionLatencyMSP50: try container.decodeIfPresent(
                Double.self, forKey: .rawSDPAAttentionLatencyMSP50),
            rawSDPAAttentionLatencyMSP95: try container.decodeIfPresent(
                Double.self, forKey: .rawSDPAAttentionLatencyMSP95),
            rawSDPADecodeTokensPerSecondP50: try container.decodeIfPresent(
                Double.self, forKey: .rawSDPADecodeTokensPerSecondP50),
            rawSDPADecodeTokensPerSecondP95: try container.decodeIfPresent(
                Double.self, forKey: .rawSDPADecodeTokensPerSecondP95),
            speedRatioToRawSDPAP50: try container.decodeIfPresent(
                Double.self, forKey: .speedRatioToRawSDPAP50),
            speedRatioToRawSDPAP95: try container.decodeIfPresent(
                Double.self, forKey: .speedRatioToRawSDPAP95),
            totalBytes: totalBytes,
            compressedKVBytes: try container.decodeIfPresent(Int.self, forKey: .compressedKVBytes),
            plainKVBytes: try container.decodeIfPresent(Int.self, forKey: .plainKVBytes),
            rawSDPAKVBytes: try container.decodeIfPresent(Int.self, forKey: .rawSDPAKVBytes),
            memoryBytesSavedVsRawSDPA: try container.decodeIfPresent(
                Int.self, forKey: .memoryBytesSavedVsRawSDPA),
            memoryReductionRatio: try container.decodeIfPresent(
                Double.self, forKey: .memoryReductionRatio),
            memoryReductionPercent: try container.decodeIfPresent(
                Double.self, forKey: .memoryReductionPercent),
            peakMemoryBytes: try container.decodeIfPresent(Int.self, forKey: .peakMemoryBytes),
            actualBitsPerValue: try container.decode(
                Double.self, forKey: .actualBitsPerValue),
            fallbackUsed: try container.decodeIfPresent(Bool.self, forKey: .fallbackUsed) ?? false,
            fallbackReason: try container.decodeIfPresent(String.self, forKey: .fallbackReason),
            memoryWarningsSeen: try container.decodeIfPresent(
                Int.self, forKey: .memoryWarningsSeen) ?? 0,
            jetsamObserved: try container.decodeIfPresent(Bool.self, forKey: .jetsamObserved)
                ?? false,
            cooldownMS: try container.decodeIfPresent(Int.self, forKey: .cooldownMS),
            pathCooldownMS: try container.decodeIfPresent(Int.self, forKey: .pathCooldownMS)
        )
    }
}

public enum TurboQuantValueBitPolicy: String, Codable, Sendable {
    case denseV4
    case calibratedV3
    case calibratedV2
    case residualVx
}

public struct TurboQuantLowerVAndSparseVReport: Codable, Sendable {
    public var referenceConfig: String
    public var candidateConfig: String
    public var valueBits: Int?
    public var valueBitPolicy: TurboQuantValueBitPolicy?
    public var sparseVMode: TurboQuantSparseVSelectionMode?
    public var sparseVTopK: Int?
    public var sparseVCumulativeMass: Double?
    public var sparseVMaxTopK: Int?
    public var sparseVRecentTokenCount: Int?
    public var sparseVOlderTokenCount: Int?
    public var sparseVPageCandidateCount: Int?
    public var selectionLatencyMS: Double?
    public var qkMS: Double?
    public var softmaxMS: Double?
    public var maskOrCompactionMS: Double?
    public var avLatencyMS: Double?
    public var totalMS: Double?
    public var denseK8V4ReferenceMS: Double?
    public var skippedValueTokens: Int?
    public var consideredValueTokens: Int?
    public var retainedMass: Double?
    public var skipRatio: Double?
    public var fallbackCount: Int
    public var fallbackReason: String?
    public var actualMixedBitsPerValue: Double?
    public var layerIndex: Int?
    public var headIndex: Int?

    public init(
        referenceConfig: String,
        candidateConfig: String,
        valueBits: Int? = nil,
        valueBitPolicy: TurboQuantValueBitPolicy? = nil,
        sparseVMode: TurboQuantSparseVSelectionMode? = nil,
        sparseVTopK: Int? = nil,
        sparseVCumulativeMass: Double? = nil,
        sparseVMaxTopK: Int? = nil,
        sparseVRecentTokenCount: Int? = nil,
        sparseVOlderTokenCount: Int? = nil,
        sparseVPageCandidateCount: Int? = nil,
        selectionLatencyMS: Double? = nil,
        qkMS: Double? = nil,
        softmaxMS: Double? = nil,
        maskOrCompactionMS: Double? = nil,
        avLatencyMS: Double? = nil,
        totalMS: Double? = nil,
        denseK8V4ReferenceMS: Double? = nil,
        skippedValueTokens: Int? = nil,
        consideredValueTokens: Int? = nil,
        retainedMass: Double? = nil,
        skipRatio: Double? = nil,
        fallbackCount: Int = 0,
        fallbackReason: String? = nil,
        actualMixedBitsPerValue: Double? = nil,
        layerIndex: Int? = nil,
        headIndex: Int? = nil
    ) {
        self.referenceConfig = referenceConfig
        self.candidateConfig = candidateConfig
        self.valueBits = valueBits.map { Swift.max(0, $0) }
        self.valueBitPolicy = valueBitPolicy
        self.sparseVMode = sparseVMode
        self.sparseVTopK = sparseVTopK.map { Swift.max(0, $0) }
        self.sparseVCumulativeMass = sparseVCumulativeMass.map { Swift.max(0, Swift.min(1, $0)) }
        self.sparseVMaxTopK = sparseVMaxTopK.map { Swift.max(0, $0) }
        self.sparseVRecentTokenCount = sparseVRecentTokenCount.map { Swift.max(0, $0) }
        self.sparseVOlderTokenCount = sparseVOlderTokenCount.map { Swift.max(0, $0) }
        self.sparseVPageCandidateCount = sparseVPageCandidateCount.map { Swift.max(0, $0) }
        self.selectionLatencyMS = selectionLatencyMS.map { Swift.max(0, $0) }
        self.qkMS = qkMS.map { Swift.max(0, $0) }
        self.softmaxMS = softmaxMS.map { Swift.max(0, $0) }
        self.maskOrCompactionMS = maskOrCompactionMS.map { Swift.max(0, $0) }
        self.avLatencyMS = avLatencyMS.map { Swift.max(0, $0) }
        self.totalMS = totalMS.map { Swift.max(0, $0) }
        self.denseK8V4ReferenceMS = denseK8V4ReferenceMS.map { Swift.max(0, $0) }
        self.skippedValueTokens = skippedValueTokens.map { Swift.max(0, $0) }
        self.consideredValueTokens = consideredValueTokens.map { Swift.max(0, $0) }
        self.retainedMass = retainedMass.map { Swift.max(0, Swift.min(1, $0)) }
        self.skipRatio = skipRatio.map { Swift.max(0, Swift.min(1, $0)) }
        self.fallbackCount = Swift.max(0, fallbackCount)
        self.fallbackReason = fallbackReason
        self.actualMixedBitsPerValue = actualMixedBitsPerValue.map { Swift.max(0, $0) }
        self.layerIndex = layerIndex.map { Swift.max(0, $0) }
        self.headIndex = headIndex.map { Swift.max(0, $0) }
    }
}

public enum TurboQuantSparseVSelectionMode: Sendable, Equatable {
    case threshold
    case topK
    case cumulativeMass
    case hybridCumulativeMassTopK
    case blockThreshold
    case pageTopK
    case candidateSparse

    public var rawValue: String {
        switch self {
        case .threshold:
            return "threshold"
        case .topK:
            return "topK"
        case .cumulativeMass:
            return "cumulativeMass"
        case .hybridCumulativeMassTopK:
            return "hybridCumulativeMassTopK"
        case .blockThreshold:
            return "blockThreshold"
        case .pageTopK:
            return "pageTopK"
        case .candidateSparse:
            return "candidateSparse"
        }
    }

    public init?(rawValue: String) {
        switch rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "_", with: "-")
            .lowercased()
        {
        case "threshold":
            self = .threshold
        case "topk", "top-k":
            self = .topK
        case "cumulativemass", "cumulative-mass", "mass":
            self = .cumulativeMass
        case "hybrid", "hybrid-cumulative", "hybrid-cumulative-mass-top-k",
             "hybridcumulativemasstopk":
            self = .hybridCumulativeMassTopK
        case "blockthreshold", "block-threshold", "blockmass", "block-mass":
            self = .blockThreshold
        case "pagetopk", "page-top-k", "page":
            self = .pageTopK
        case "candidatesparse", "candidate-sparse":
            self = .candidateSparse
        default:
            return nil
        }
    }
}

extension TurboQuantSparseVSelectionMode: Codable {
    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let raw = try container.decode(String.self)
        guard let mode = Self(rawValue: raw) else {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Unknown Sparse-V selection mode '\(raw)'."
            )
        }
        self = mode
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(rawValue)
    }
}

public struct TurboQuantSparseVDiagnostic: Codable, Sendable {
    public var layer: Int?
    public var head: Int?
    public var selectionMode: TurboQuantSparseVSelectionMode?
    public var skippedValueTokens: Int
    public var consideredValueTokens: Int
    public var recentTokenCount: Int?
    public var olderTokenCount: Int?
    public var pageCandidateCount: Int?
    public var retainedAttentionMass: Double?
    public var maxOutputErrorVsDenseReference: Double?
    public var cosineVsDenseReference: Double?
    public var fallbackReason: String?

    public init(
        layer: Int? = nil,
        head: Int? = nil,
        selectionMode: TurboQuantSparseVSelectionMode? = nil,
        skippedValueTokens: Int = 0,
        consideredValueTokens: Int = 0,
        recentTokenCount: Int? = nil,
        olderTokenCount: Int? = nil,
        pageCandidateCount: Int? = nil,
        retainedAttentionMass: Double? = nil,
        maxOutputErrorVsDenseReference: Double? = nil,
        cosineVsDenseReference: Double? = nil,
        fallbackReason: String? = nil
    ) {
        self.layer = layer.map { Swift.max(0, $0) }
        self.head = head.map { Swift.max(0, $0) }
        self.selectionMode = selectionMode
        self.skippedValueTokens = Swift.max(0, skippedValueTokens)
        self.consideredValueTokens = Swift.max(0, consideredValueTokens)
        self.recentTokenCount = recentTokenCount.map { Swift.max(0, $0) }
        self.olderTokenCount = olderTokenCount.map { Swift.max(0, $0) }
        self.pageCandidateCount = pageCandidateCount.map { Swift.max(0, $0) }
        self.retainedAttentionMass = retainedAttentionMass.map { Swift.max(0, Swift.min(1, $0)) }
        self.maxOutputErrorVsDenseReference = maxOutputErrorVsDenseReference.map {
            Swift.max(0, $0)
        }
        self.cosineVsDenseReference = cosineVsDenseReference.map {
            Swift.max(-1, Swift.min(1, $0))
        }
        self.fallbackReason = fallbackReason
    }
}

public enum TurboQuantBenchmarkRoute: String, Codable, Sendable {
    case rawSDPA
    case adaptiveTurboQuant
    case hybridTurboQuant
    case compressedFused
    case decodedFallback
    case unavailable
}

public enum TurboQuantBenchmarkBackend: String, Codable, Sendable {
    case swiftMetalKernel
    case nativeMLX
    case decodedReference
    case rawSDPA
    case unavailable
}

public struct TurboQuantBenchmarkKernelFlags: Codable, Sendable {
    public var tqCoopEnabled: Bool
    public var blockTokenSize: Int?
    public var gqaSpecialization: String?
    public var outputDType: String

    public init(
        tqCoopEnabled: Bool,
        blockTokenSize: Int? = nil,
        gqaSpecialization: String? = nil,
        outputDType: String
    ) {
        self.tqCoopEnabled = tqCoopEnabled
        self.blockTokenSize = blockTokenSize
        self.gqaSpecialization = gqaSpecialization
        self.outputDType = outputDType
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
                mitigation:
                    "V5 scale tables are emitted directly by the Metal encode kernel and validated as canonical float16/float32 storage before dispatch",
                status: "guarded"
            ),
            TurboQuantHiddenCopyAuditEntry(
                kernelName: "block-parallel fused",
                largeInput: "compressed K/V cache",
                copyRisk: "high",
                mitigation:
                    "block partial kernels consume canonical compressed K/V arrays and emit bounded per-block partials, not decoded cache copies",
                status: "guarded"
            ),
            TurboQuantHiddenCopyAuditEntry(
                kernelName: "GQA block-parallel fused",
                largeInput: "compressed K/V cache shared by grouped query heads",
                copyRisk: "high",
                mitigation:
                    "Mac-gated grouped-query block partial kernels reuse compressed K/V reads across Qwen query-head pairs without materializing decoded cache copies",
                status: "guarded"
            ),
            TurboQuantHiddenCopyAuditEntry(
                kernelName: "segmented hybrid attention",
                largeInput: "selected compressed cold blocks plus raw hot tail",
                copyRisk: "high",
                mitigation:
                    "raw and compressed segment partials are merged by online softmax statistics while selected cold blocks remain compressed",
                status: "guarded"
            ),
        ],
        notes: currentW3.notes + [
            "Layout V6 is the default/current write layout; V4 and V5 remain supported for compatibility comparisons."
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
