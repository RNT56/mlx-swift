import Foundation
import MLX

private let defaultBenchmarkBatchSize = 1
private let defaultBenchmarkQueryHeadCount = 4
private let defaultBenchmarkKVHeadCount = 2

struct QualityMetrics: Codable {
    var relativeMSE: Float
    var maxAbsoluteError: Float
    var cosineSimilarity: Float
}

struct BenchmarkResult: Codable {
    var name: String
    var status: String
    var selectedPath: String?
    var dtype: String?
    var shape: [Int]
    var queryShape: [Int]?
    var keyShape: [Int]?
    var valueShape: [Int]?
    var preset: TurboQuantPreset?
    var valueBits: Int?
    var actualBitsPerValue: Double?
    var memoryBytes: Int?
    var latencySeconds: Double?
    var denseReferenceLatencySeconds: Double?
    var quality: QualityMetrics?
    var sparseVSelectionMode: String?
    var sparseVThreshold: Float?
    var sparseVTopK: Int?
    var sparseVCumulativeMass: Float?
    var sparseVMaxTopK: Int?
    var sparseVRecentTokenCount: Int?
    var sparseVOlderTokenCount: Int?
    var sparseVPageCandidateCount: Int?
    var sparseVPageSummary: Bool?
    var sparseVSkippedTokens: Int?
    var sparseVTotalTokens: Int?
    var sparseVSkipRatio: Double?
    var activeBlocks: Int?
    var blockTokens: Int?
    var kernelKind: Int?
    var error: String?
}

struct BenchmarkReport: Codable {
    var schemaVersion: Int
    var generatedAt: String?
    var iterations: Int
    var availability: TurboQuantKernelAvailability
    var capabilities: TurboQuantKernelCapabilities
    var attentionCapabilities: TurboQuantAttentionCapabilities
    var device: TurboQuantDeviceCapabilities
    var results: [BenchmarkResult]
}

private enum BenchmarkSparseVSelectionMode: Equatable {
    case off
    case threshold
    case topK
    case cumulativeMass
    case hybridCumulativeMassTopK
    case blockThreshold
    case pageTopK
    case candidateSparse

    var nativeMode: TurboQuantSparseValueNativeSelectionMode {
        switch self {
        case .off:
            return .off
        case .threshold:
            return .threshold
        case .topK:
            return .topK
        case .cumulativeMass:
            return .cumulativeMass
        case .hybridCumulativeMassTopK:
            return .hybridCumulativeMassTopK
        case .blockThreshold:
            return .blockThreshold
        case .pageTopK, .candidateSparse:
            return .pageTopK
        }
    }

    var reportMode: TurboQuantSparseVSelectionMode? {
        switch self {
        case .off:
            return nil
        case .threshold:
            return .threshold
        case .topK:
            return .topK
        case .cumulativeMass:
            return .cumulativeMass
        case .hybridCumulativeMassTopK:
            return .hybridCumulativeMassTopK
        case .blockThreshold:
            return .blockThreshold
        case .pageTopK:
            return .pageTopK
        case .candidateSparse:
            return .candidateSparse
        }
    }
}

private enum BenchmarkCLIError: Error, CustomStringConvertible {
    case invalidInteger(String)
    case invalidFloat(String)
    case invalidPreset(String)
    case invalidPath(String)
    case invalidScaleStorage(String)
    case invalidSparseVSelection(String)

    var description: String {
        switch self {
        case .invalidInteger(let flag):
            "Invalid integer value for \(flag)."
        case .invalidFloat(let flag):
            "Invalid floating-point value for \(flag)."
        case .invalidPreset(let value):
            "Invalid TurboQuant preset '\(value)'."
        case .invalidPath(let value):
            "Invalid TurboQuant path '\(value)'."
        case .invalidScaleStorage(let value):
            "Invalid TurboQuant scale storage '\(value)'."
        case .invalidSparseVSelection(let value):
            "Invalid Sparse-V selection mode '\(value)'."
        }
    }
}

private struct BenchmarkOptions {
    var emitCoreJSON: Bool
    var includeTimestamp: Bool
    var iterations: Int
    var warmup: Int
    var cooldownMilliseconds: Int
    var pathCooldownMilliseconds: Int
    var batchSize: Int
    var queryHeadCount: Int
    var kvHeadCount: Int
    var headDimension: Int
    var contextTokens: Int
    var queryLength: Int
    var preset: TurboQuantPreset
    var valueBits: Int?
    var groupSize: Int
    var layoutVersion: Int
    var enableLayoutV5: Bool
    var enableLayoutV7: Bool
    var scaleStorage: TurboQuantScaleStorage
    var blockParallelTokenBlockSize: Int?
    var requestedPath: TurboQuantAttentionPath?
    var sparseVSelectionMode: BenchmarkSparseVSelectionMode?
    var sparseVThreshold: Float?
    var sparseVTopK: Int?
    var sparseVCumulativeMass: Float?
    var sparseVMaxTopK: Int?
    var sparseVRecentTokens: Int?
    var sparseVCandidatePages: Int?
    var sparseVUsePageSummary: Bool

    var resolvedValueBits: Int {
        valueBits ?? preset.defaultValueBits
    }

    var sparseVEnabled: Bool {
        guard let sparseVSelectionMode else { return false }
        return sparseVSelectionMode != .off
    }

    var resolvedSparseVThreshold: Float {
        sparseVThreshold ?? 1e-5
    }

    var resolvedSparseVTopK: Int {
        sparseVTopK ?? 128
    }

    var resolvedSparseVCumulativeMass: Float {
        sparseVCumulativeMass ?? 0.995
    }

    var resolvedSparseVMaxTopK: Int {
        sparseVMaxTopK ?? sparseVTopK ?? 512
    }

    var resolvedSparseVRecentTokens: Int {
        sparseVRecentTokens ?? 256
    }

    var resolvedSparseVCandidatePages: Int {
        sparseVCandidatePages ?? 4
    }

    var sparseVRecentTokenCount: Int? {
        sparseVSelectionMode == .candidateSparse ? resolvedSparseVRecentTokens : nil
    }

    var sparseVOlderTokenCount: Int? {
        sparseVSelectionMode == .candidateSparse ? resolvedSparseVTopK : nil
    }

    var sparseVPageCandidateCount: Int? {
        switch sparseVSelectionMode {
        case .candidateSparse:
            return resolvedSparseVCandidatePages
        case .pageTopK:
            return resolvedSparseVTopK
        case .off, .threshold, .topK, .cumulativeMass, .hybridCumulativeMassTopK,
            .blockThreshold, nil:
            return nil
        }
    }

    static func parse(_ arguments: [String] = CommandLine.arguments) throws -> BenchmarkOptions {
        let presetName =
            stringValue("--preset", in: arguments) ?? TurboQuantPreset.turbo4v2.rawValue
        guard let preset = TurboQuantPreset(rawValue: presetName) else {
            throw BenchmarkCLIError.invalidPreset(presetName)
        }

        return BenchmarkOptions(
            emitCoreJSON: arguments.contains("--json"),
            includeTimestamp: arguments.contains("--include-timestamp"),
            iterations: try intValue("--iterations", in: arguments, default: 10, minimum: 1),
            warmup: try intValue("--warmup", in: arguments, default: 1, minimum: 0),
            cooldownMilliseconds: try intValue(
                "--cooldown-ms", in: arguments, default: 0, minimum: 0),
            pathCooldownMilliseconds: try intValue(
                "--path-cooldown-ms", in: arguments, default: 0, minimum: 0),
            batchSize: try intValue(
                "--batch-size", in: arguments, default: defaultBenchmarkBatchSize, minimum: 1),
            queryHeadCount: try intValue(
                "--query-heads", in: arguments, default: defaultBenchmarkQueryHeadCount,
                minimum: 1),
            kvHeadCount: try intValue(
                "--kv-heads", in: arguments, default: defaultBenchmarkKVHeadCount, minimum: 1),
            headDimension: try intValue("--head-dim", in: arguments, default: 128, minimum: 1),
            contextTokens: try intValue("--context", in: arguments, default: 256, minimum: 1),
            queryLength: try intValue("--query-length", in: arguments, default: 1, minimum: 1),
            preset: preset,
            valueBits: try optionalIntValue("--value-bits", in: arguments, minimum: 1),
            groupSize: try intValue("--group-size", in: arguments, default: 64, minimum: 1),
            layoutVersion: try intValue(
                "--layout-version",
                in: arguments,
                default: TurboQuantAttentionLayout.productionDefaultVersion,
                minimum: 1
            ),
            enableLayoutV5: arguments.contains("--enable-layout-v5"),
            enableLayoutV7: arguments.contains("--enable-layout-v7"),
            scaleStorage: try scaleStorage(in: arguments),
            blockParallelTokenBlockSize: try optionalIntValue(
                "--block-tokens", in: arguments, minimum: 1),
            requestedPath: try requestedPath(in: arguments),
            sparseVSelectionMode: try sparseVSelectionMode(in: arguments),
            sparseVThreshold: try optionalFloatValue("--sparse-v-threshold", in: arguments),
            sparseVTopK: try optionalIntValue("--sparse-v-top-k", in: arguments, minimum: 0),
            sparseVCumulativeMass: try optionalFloatValue(
                "--sparse-v-cumulative-mass", in: arguments)
                ?? optionalFloatValue("--sparse-v-mass", in: arguments),
            sparseVMaxTopK: try optionalIntValue("--sparse-v-max-top-k", in: arguments, minimum: 0)
                ?? optionalIntValue("--sparse-v-hybrid-top-k", in: arguments, minimum: 0),
            sparseVRecentTokens: try optionalIntValue(
                "--sparse-v-recent-tokens", in: arguments, minimum: 0),
            sparseVCandidatePages: try optionalIntValue(
                "--sparse-v-candidate-pages", in: arguments, minimum: 0),
            sparseVUsePageSummary: arguments.contains("--sparse-v-page-summary")
        )
    }

    private static func stringValue(_ name: String, in arguments: [String]) -> String? {
        guard let index = arguments.firstIndex(of: name), arguments.indices.contains(index + 1)
        else {
            return nil
        }
        return arguments[index + 1]
    }

    private static func intValue(
        _ name: String,
        in arguments: [String],
        default defaultValue: Int,
        minimum: Int
    ) throws -> Int {
        guard let rawValue = stringValue(name, in: arguments) else {
            return defaultValue
        }
        guard let value = Int(rawValue), value >= minimum else {
            throw BenchmarkCLIError.invalidInteger(name)
        }
        return value
    }

    private static func optionalIntValue(
        _ name: String,
        in arguments: [String],
        minimum: Int
    ) throws -> Int? {
        guard let rawValue = stringValue(name, in: arguments) else {
            return nil
        }
        guard let value = Int(rawValue), value >= minimum else {
            throw BenchmarkCLIError.invalidInteger(name)
        }
        return value
    }

    private static func optionalFloatValue(_ name: String, in arguments: [String]) throws -> Float? {
        guard let rawValue = stringValue(name, in: arguments) else {
            return nil
        }
        guard let value = Float(rawValue), value.isFinite else {
            throw BenchmarkCLIError.invalidFloat(name)
        }
        return value
    }

    private static func requestedPath(in arguments: [String]) throws -> TurboQuantAttentionPath? {
        guard let value = stringValue("--path", in: arguments), value != "auto" else {
            return nil
        }

        switch value {
        case TurboQuantAttentionPath.nativeMLXCompressed.rawValue, "native-mlx",
            "native-mlx-compressed":
            return .nativeMLXCompressed
        case TurboQuantAttentionPath.sparseValueTwoStageCompressed.rawValue,
            "sparse-value-two-stage", "sparse-value-two-stage-compressed", "sparse-two-stage":
            return .sparseValueTwoStageCompressed
        case TurboQuantAttentionPath.affineInt4Native.rawValue, "affine-int4-native",
            "native-affine-int4":
            return .affineInt4Native
        case TurboQuantAttentionPath.affineK8V4Native.rawValue, "affine-k8v4-native",
            "native-affine-k8v4":
            return .affineK8V4Native
        case TurboQuantAttentionPath.affineK8VxNative.rawValue, "affine-k8vx-native",
            "native-affine-k8vx":
            return .affineK8VxNative
        case TurboQuantAttentionPath.affineK8VxResidual.rawValue, "affine-k8vx-residual",
            "native-affine-k8vx-residual":
            return .affineK8VxResidual
        case TurboQuantAttentionPath.metalHybridK8PolarWHTValue.rawValue,
            "metal-hybrid-k8-polarwht-value", "hybrid-k8-polarwht-value":
            return .metalHybridK8PolarWHTValue
        case TurboQuantAttentionPath.onlineFused.rawValue, "online-fused":
            return .onlineFused
        case TurboQuantAttentionPath.tiledOnlineFused.rawValue, "tiled-online-fused":
            return .tiledOnlineFused
        case TurboQuantAttentionPath.twoStageCompressed.rawValue, "two-stage",
            "two-stage-compressed":
            return .twoStageCompressed
        case TurboQuantAttentionPath.mlxPackedFallback.rawValue, "mlx-packed-fallback":
            return .mlxPackedFallback
        case TurboQuantAttentionPath.baseline.rawValue:
            return .baseline
        case TurboQuantAttentionPath.unavailable.rawValue:
            return .unavailable
        default:
            throw BenchmarkCLIError.invalidPath(value)
        }
    }

    private static func scaleStorage(in arguments: [String]) throws -> TurboQuantScaleStorage {
        guard let value = stringValue("--scale-storage", in: arguments) else {
            return .float32
        }
        guard let storage = TurboQuantScaleStorage(rawValue: value) else {
            throw BenchmarkCLIError.invalidScaleStorage(value)
        }
        return storage
    }

    private static func sparseVSelectionMode(
        in arguments: [String]
    ) throws -> BenchmarkSparseVSelectionMode? {
        guard let value = stringValue("--sparse-v", in: arguments)
            ?? stringValue("--sparse-v-mode", in: arguments)
        else {
            return nil
        }

        switch value.lowercased().replacingOccurrences(of: "-", with: "")
            .replacingOccurrences(of: "_", with: "")
        {
        case "off", "none", "false", "0":
            return .off
        case "threshold":
            return .threshold
        case "topk":
            return .topK
        case "cumulative", "cumulativemass", "mass":
            return .cumulativeMass
        case "hybrid", "hybridcumulativemass", "hybridcumulativemasstopk":
            return .hybridCumulativeMassTopK
        case "blockthreshold", "blockmass":
            return .blockThreshold
        case "pagetopk", "page":
            return .pageTopK
        case "candidatesparse":
            return .candidateSparse
        default:
            throw BenchmarkCLIError.invalidSparseVSelection(value)
        }
    }
}

private func values(count: Int, scale: Double, phase: Double = 0) -> [Float] {
    (0 ..< count).map { index in
        let position = Double(index)
        return Float(0.31 * sin(position * scale + phase) + 0.17 * cos(position * 0.037))
    }
}

private func qualityMetrics(_ lhs: MLXArray, _ rhs: MLXArray) -> QualityMetrics {
    let left = lhs.asArray(Float.self)
    let right = rhs.asArray(Float.self)
    let error = zip(left, right).reduce(Float(0)) { partial, pair in
        let delta = pair.0 - pair.1
        return partial + delta * delta
    }
    let maxAbsoluteError = zip(left, right).reduce(Float(0)) { partial, pair in
        max(partial, abs(pair.0 - pair.1))
    }
    let signal = left.reduce(Float(0)) { $0 + $1 * $1 }
    let dot = zip(left, right).reduce(Float(0)) { $0 + $1.0 * $1.1 }
    let leftNorm = sqrt(left.reduce(Float(0)) { $0 + $1 * $1 })
    let rightNorm = sqrt(right.reduce(Float(0)) { $0 + $1 * $1 })
    return QualityMetrics(
        relativeMSE: error / max(signal, Float.leastNonzeroMagnitude),
        maxAbsoluteError: maxAbsoluteError,
        cosineSimilarity: dot / max(leftNorm * rightNorm, Float.leastNonzeroMagnitude)
    )
}

private func timed(
    iterations: Int,
    warmup: Int = 1,
    cooldownMilliseconds: Int = 0,
    _ body: () throws -> MLXArray
) throws -> (Double, MLXArray) {
    try timedValue(
        iterations: iterations,
        warmup: warmup,
        cooldownMilliseconds: cooldownMilliseconds,
        evaluate: { eval($0) },
        body
    )
}

private struct TimingSummary {
    var averageSeconds: Double
    var p50Seconds: Double
    var p95Seconds: Double
}

private struct TimedValueSummary<T> {
    var timing: TimingSummary
    var value: T
}

private func timedSampled(
    iterations: Int,
    warmup: Int = 1,
    cooldownMilliseconds: Int = 0,
    _ body: () throws -> MLXArray
) throws -> TimedValueSummary<MLXArray> {
    try timedValueSampled(
        iterations: iterations,
        warmup: warmup,
        cooldownMilliseconds: cooldownMilliseconds,
        evaluate: { eval($0) },
        body
    )
}

private func timedValue<T>(
    iterations: Int,
    warmup: Int,
    cooldownMilliseconds: Int = 0,
    evaluate: (T) -> Void,
    _ body: () throws -> T
) throws -> (Double, T) {
    let measuredIterations = max(1, iterations)
    var last: T?
    var elapsedTotal = 0.0

    for _ in 0 ..< max(0, warmup) {
        let value = try body()
        evaluate(value)
        last = value
        cooldown(milliseconds: cooldownMilliseconds)
    }

    for iteration in 0 ..< measuredIterations {
        let start = Date.timeIntervalSinceReferenceDate
        let value = try body()
        evaluate(value)
        elapsedTotal += Date.timeIntervalSinceReferenceDate - start
        last = value
        if iteration + 1 < measuredIterations {
            cooldown(milliseconds: cooldownMilliseconds)
        }
    }

    return (elapsedTotal / Double(measuredIterations), last!)
}

private func timedValueSampled<T>(
    iterations: Int,
    warmup: Int,
    cooldownMilliseconds: Int = 0,
    evaluate: (T) -> Void,
    _ body: () throws -> T
) throws -> TimedValueSummary<T> {
    let measuredIterations = max(1, iterations)
    var last: T?
    var samples: [Double] = []
    samples.reserveCapacity(measuredIterations)

    for _ in 0 ..< max(0, warmup) {
        let value = try body()
        evaluate(value)
        last = value
        cooldown(milliseconds: cooldownMilliseconds)
    }

    for iteration in 0 ..< measuredIterations {
        let start = Date.timeIntervalSinceReferenceDate
        let value = try body()
        evaluate(value)
        let elapsed = Date.timeIntervalSinceReferenceDate - start
        samples.append(elapsed)
        last = value
        if iteration + 1 < measuredIterations {
            cooldown(milliseconds: cooldownMilliseconds)
        }
    }

    let sorted = samples.sorted()
    let average = samples.reduce(0, +) / Double(measuredIterations)
    return TimedValueSummary(
        timing: TimingSummary(
            averageSeconds: average,
            p50Seconds: percentile(sortedSamples: sorted, percentile: 0.50),
            p95Seconds: percentile(sortedSamples: sorted, percentile: 0.95)
        ),
        value: last!
    )
}

private func cooldown(milliseconds: Int) {
    guard milliseconds > 0 else { return }
    Thread.sleep(forTimeInterval: Double(milliseconds) / 1000)
}

private func percentile(sortedSamples: [Double], percentile: Double) -> Double {
    guard let first = sortedSamples.first else { return 0 }
    guard sortedSamples.count > 1 else { return first }
    let clamped = min(max(percentile, 0), 1)
    let index = Int(ceil(clamped * Double(sortedSamples.count))) - 1
    return sortedSamples[min(max(index, 0), sortedSamples.count - 1)]
}

private func skipped(_ name: String, reason: String) -> BenchmarkResult {
    BenchmarkResult(name: name, status: "skipped", shape: [], error: reason)
}

private func writeJSON<T: Encodable>(_ value: T) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try encoder.encode(value)
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data("\n".utf8))
}

private func runCoreBenchmarkJSON(options: BenchmarkOptions) throws {
    try validateCoreBenchmarkOptions(options)
    let availability = TurboQuantKernelAvailability.current
    let capabilities = availability.kernelCapabilities
    let hiddenCopyAudit = TurboQuantHiddenCopyAudit.currentW5
    guard hiddenCopyAudit.status != .fail else {
        throw TurboQuantError.invalidMetalConfiguration(
            "TurboQuant hidden-copy audit failed; benchmark report is blocked."
        )
    }

    var storageEstimate = symbolicAggregateStorageEstimate(options: options)
    var pathDecision = corePathDecision(options: options, availability: availability)
    var benchmarkError: String?
    var encodeMS: Double?
    var decodeMS: Double?
    var qkMS: Double?
    var avMS: Double?
    var fusedMS: Double?
    var firstTokenLatencyMS: Double?
    var attentionLatencyMSP50: Double?
    var attentionLatencyMSP95: Double?
    var prefillTokensPerSecond: Double?
    var decodeTokensPerSecondP50: Double?
    var decodeTokensPerSecondP95: Double?
    var plainAttentionLatencyMSP50: Double?
    var plainAttentionLatencyMSP95: Double?
    var plainDecodeTokensPerSecondP50: Double?
    var plainDecodeTokensPerSecondP95: Double?
    var speedRatioToPlainP50: Double?
    var speedRatioToPlainP95: Double?
    var rawSDPAAttentionLatencyMSP50: Double?
    var rawSDPAAttentionLatencyMSP95: Double?
    var rawSDPADecodeTokensPerSecondP50: Double?
    var rawSDPADecodeTokensPerSecondP95: Double?
    var speedRatioToRawSDPAP50: Double?
    var speedRatioToRawSDPAP95: Double?
    var rawSDPAKVBytes: Int?
    var memoryBytesSavedVsRawSDPA: Int?
    var memoryReductionRatio: Double?
    var memoryReductionPercent: Double?
    var pathMeasurements: [TurboQuantCoreBenchmarkPathMeasurement] = []

    if availability.supportsMetalPolarQJLAttention {
        do {
            let measurement = try measureCoreAttention(
                options: options,
                decision: pathDecision,
                availability: availability
            )
            storageEstimate = measurement.storageEstimate
            encodeMS = milliseconds(measurement.encodeSeconds)
            decodeMS = milliseconds(measurement.decodeSeconds)
            qkMS = milliseconds(measurement.qkSeconds)
            avMS = milliseconds(measurement.avSeconds)
            fusedMS = milliseconds(measurement.fusedSeconds)
            rawSDPAKVBytes = measurement.rawSDPAKVBytes
            pathMeasurements = measurement.pathMeasurements

            if let encodeSeconds = measurement.encodeSeconds, encodeSeconds > 0 {
                prefillTokensPerSecond = Double(options.contextTokens) / encodeSeconds
            }

            if let attentionTiming = measurement.attentionTiming,
                attentionTiming.averageSeconds > 0
            {
                firstTokenLatencyMS = milliseconds(attentionTiming.averageSeconds)
                attentionLatencyMSP50 = milliseconds(attentionTiming.p50Seconds)
                attentionLatencyMSP95 = milliseconds(attentionTiming.p95Seconds)
                if attentionTiming.p50Seconds > 0 {
                    decodeTokensPerSecondP50 =
                        Double(options.queryLength) / attentionTiming.p50Seconds
                }
                if attentionTiming.p95Seconds > 0 {
                    decodeTokensPerSecondP95 =
                        Double(options.queryLength) / attentionTiming.p95Seconds
                }
            }

            if let rawTiming = measurement.rawSDPAAttentionTiming {
                rawSDPAAttentionLatencyMSP50 = milliseconds(rawTiming.p50Seconds)
                rawSDPAAttentionLatencyMSP95 = milliseconds(rawTiming.p95Seconds)
                plainAttentionLatencyMSP50 = rawSDPAAttentionLatencyMSP50
                plainAttentionLatencyMSP95 = rawSDPAAttentionLatencyMSP95
                if rawTiming.p50Seconds > 0 {
                    rawSDPADecodeTokensPerSecondP50 =
                        Double(options.queryLength) / rawTiming.p50Seconds
                    plainDecodeTokensPerSecondP50 = rawSDPADecodeTokensPerSecondP50
                }
                if rawTiming.p95Seconds > 0 {
                    rawSDPADecodeTokensPerSecondP95 =
                        Double(options.queryLength) / rawTiming.p95Seconds
                    plainDecodeTokensPerSecondP95 = rawSDPADecodeTokensPerSecondP95
                }
            }

            if let decodeTokensPerSecondP50,
                let rawSDPADecodeTokensPerSecondP50,
                rawSDPADecodeTokensPerSecondP50 > 0
            {
                speedRatioToRawSDPAP50 =
                    decodeTokensPerSecondP50 / rawSDPADecodeTokensPerSecondP50
                speedRatioToPlainP50 = speedRatioToRawSDPAP50
            }
            if let decodeTokensPerSecondP95,
                let rawSDPADecodeTokensPerSecondP95,
                rawSDPADecodeTokensPerSecondP95 > 0
            {
                speedRatioToRawSDPAP95 =
                    decodeTokensPerSecondP95 / rawSDPADecodeTokensPerSecondP95
                speedRatioToPlainP95 = speedRatioToRawSDPAP95
            }
            if let rawSDPAKVBytes, storageEstimate.totalBytes > 0 {
                memoryBytesSavedVsRawSDPA = max(0, rawSDPAKVBytes - storageEstimate.totalBytes)
                memoryReductionRatio =
                    Double(rawSDPAKVBytes) / Double(storageEstimate.totalBytes)
                if rawSDPAKVBytes > 0, let memoryBytesSavedVsRawSDPA {
                    memoryReductionPercent =
                        Double(memoryBytesSavedVsRawSDPA) / Double(rawSDPAKVBytes) * 100
                }
            }
        } catch {
            benchmarkError = String(describing: error)
            var rejected = pathDecision.rejectedPaths
            rejected.append(
                RejectedPath(path: pathDecision.selectedPath, reason: "benchmark failed: \(error)")
            )
            pathDecision = TurboQuantAttentionDecision(
                selectedPath: .unavailable,
                outputDType: pathDecision.outputDType,
                estimatedScratchBytes: pathDecision.estimatedScratchBytes,
                rejectedPaths: rejected,
                headDimension: pathDecision.headDimension,
                queryLength: pathDecision.queryLength,
                logicalLength: pathDecision.logicalLength,
                dtype: pathDecision.dtype,
                maskKind: pathDecision.maskKind,
                kernelProfile: pathDecision.kernelProfile,
                fallbackReason: "benchmark failed: \(error)"
            )
        }
    }

    let fallbackUsed =
        pathDecision.selectedPath == .baseline
        || pathDecision.selectedPath == .mlxPackedFallback
        || benchmarkError != nil
    let fallbackReason =
        benchmarkError
        ?? (fallbackUsed
            ? pathDecision.rejectedPaths.map { "\($0.path.rawValue): \($0.reason)" }
                .joined(separator: "; ")
            : nil)

    let metrics = TurboQuantCoreBenchmarkMetrics(
        route: benchmarkRoute(for: pathDecision.selectedPath).rawValue,
        runtimeMode: "capacityTurboQuant",
        backend: benchmarkBackend(for: pathDecision.selectedPath).rawValue,
        kernelFlags: TurboQuantBenchmarkKernelFlags(
            tqCoopEnabled: ProcessInfo.processInfo.environment["TQ_COOP"] == "1",
            blockTokenSize: options.blockParallelTokenBlockSize,
            gqaSpecialization: options.queryHeadCount % options.kvHeadCount == 0
                && options.queryHeadCount / options.kvHeadCount > 1
                ? "gqa\(options.queryHeadCount / options.kvHeadCount)" : nil,
            outputDType: String(describing: pathDecision.outputDType)
        ),
        sparseVEnabled: options.sparseVEnabled,
        sparseVSelectionMode: options.sparseVSelectionMode?.reportMode,
        sparseVThreshold: (options.sparseVSelectionMode == .threshold
            || options.sparseVSelectionMode == .blockThreshold)
            ? options.resolvedSparseVThreshold : nil,
        sparseVRecentTokenCount: options.sparseVRecentTokenCount,
        sparseVOlderTokenCount: options.sparseVOlderTokenCount,
        sparseVPageCandidateCount: options.sparseVPageCandidateCount,
        contextTokens: options.contextTokens,
        headDimension: options.headDimension,
        queryLength: options.queryLength,
        preset: options.preset.rawValue,
        valueBits: options.resolvedValueBits,
        groupSize: options.groupSize,
        layoutVersion: options.layoutVersion,
        scaleStorage: options.scaleStorage.rawValue,
        blockParallelTokenBlockSize: options.blockParallelTokenBlockSize,
        recommendedBlockParallelTokenBlockSize: turboQuantRecommendedBlockParallelTokenBlockSize(
            logicalLength: options.contextTokens,
            headDimension: options.headDimension,
            queryLength: options.queryLength,
            kernelProfile: pathDecision.kernelProfile
        ),
        warmupIterations: options.warmup,
        encodeMS: encodeMS,
        decodeMS: decodeMS,
        qkMS: qkMS,
        avMS: avMS,
        fusedMS: fusedMS,
        firstTokenLatencyMS: firstTokenLatencyMS,
        attentionLatencyMSP50: attentionLatencyMSP50,
        attentionLatencyMSP95: attentionLatencyMSP95,
        prefillTokensPerSecond: prefillTokensPerSecond,
        decodeTokensPerSecondP50: decodeTokensPerSecondP50,
        decodeTokensPerSecondP95: decodeTokensPerSecondP95,
        plainAttentionLatencyMSP50: plainAttentionLatencyMSP50,
        plainAttentionLatencyMSP95: plainAttentionLatencyMSP95,
        plainDecodeTokensPerSecondP50: plainDecodeTokensPerSecondP50,
        plainDecodeTokensPerSecondP95: plainDecodeTokensPerSecondP95,
        speedRatioToPlainP50: speedRatioToPlainP50,
        speedRatioToPlainP95: speedRatioToPlainP95,
        rawSDPAReferenceDType: rawSDPAKVBytes == nil ? nil : "float16",
        rawSDPAAttentionLatencyMSP50: rawSDPAAttentionLatencyMSP50,
        rawSDPAAttentionLatencyMSP95: rawSDPAAttentionLatencyMSP95,
        rawSDPADecodeTokensPerSecondP50: rawSDPADecodeTokensPerSecondP50,
        rawSDPADecodeTokensPerSecondP95: rawSDPADecodeTokensPerSecondP95,
        speedRatioToRawSDPAP50: speedRatioToRawSDPAP50,
        speedRatioToRawSDPAP95: speedRatioToRawSDPAP95,
        totalBytes: storageEstimate.totalBytes,
        compressedKVBytes: storageEstimate.totalBytes,
        plainKVBytes: rawSDPAKVBytes,
        rawSDPAKVBytes: rawSDPAKVBytes,
        memoryBytesSavedVsRawSDPA: memoryBytesSavedVsRawSDPA,
        memoryReductionRatio: memoryReductionRatio,
        memoryReductionPercent: memoryReductionPercent,
        peakMemoryBytes: nil,
        actualBitsPerValue: storageEstimate.actualBitsPerValue,
        fallbackUsed: fallbackUsed,
        fallbackReason: fallbackReason?.isEmpty == true ? nil : fallbackReason,
        memoryWarningsSeen: 0,
        jetsamObserved: false,
        cooldownMS: options.cooldownMilliseconds,
        pathCooldownMS: options.pathCooldownMilliseconds
    )

    let report = TurboQuantCoreBenchmarkReport(
        mlxSwiftCommit: currentGitCommit(),
        capabilities: capabilities,
        storageEstimate: storageEstimate,
        pathDecision: pathDecision,
        pathMeasurements: pathMeasurements,
        metrics: metrics,
        hiddenCopyAudit: hiddenCopyAudit
    )
    try writeJSON(report)
}

private func benchmarkRoute(for path: TurboQuantAttentionPath) -> TurboQuantBenchmarkRoute {
    switch path {
    case .baseline:
        return .rawSDPA
    case .nativeMLXCompressed, .affineK8V4Native, .affineK8VxNative,
        .affineK8VxResidual, .affineInt4Native:
        return .compressedFused
    case .onlineFused, .tiledOnlineFused, .sparseValueTwoStageCompressed,
        .metalPolarWHTHybrid, .metalHybridK8PolarWHTValue:
        return .compressedFused
    case .twoStageCompressed, .polarWHTReferenceHybrid, .mlxPackedFallback:
        return .decodedFallback
    case .unavailable:
        return .unavailable
    }
}

private func benchmarkBackend(for path: TurboQuantAttentionPath) -> TurboQuantBenchmarkBackend {
    switch path {
    case .baseline:
        return .rawSDPA
    case .nativeMLXCompressed, .affineK8V4Native, .affineK8VxNative,
        .affineK8VxResidual, .affineInt4Native:
        return .nativeMLX
    case .onlineFused, .tiledOnlineFused, .sparseValueTwoStageCompressed,
        .twoStageCompressed, .metalPolarWHTHybrid, .metalHybridK8PolarWHTValue:
        return .swiftMetalKernel
    case .polarWHTReferenceHybrid, .mlxPackedFallback:
        return .decodedReference
    case .unavailable:
        return .unavailable
    }
}

private struct CoreAttentionMeasurement {
    var storageEstimate: TurboQuantStorageEstimate
    var encodeSeconds: Double?
    var decodeSeconds: Double?
    var qkSeconds: Double?
    var avSeconds: Double?
    var fusedSeconds: Double?
    var attentionTiming: TimingSummary?
    var rawSDPAAttentionTiming: TimingSummary?
    var rawSDPAKVBytes: Int?
    var pathMeasurements: [TurboQuantCoreBenchmarkPathMeasurement]

    var twoStageAttentionSeconds: Double? {
        guard let qkSeconds, let avSeconds else { return nil }
        return qkSeconds + avSeconds
    }
}

private struct CoreAttentionInputs {
    var query: MLXArray
    var keys: MLXArray
    var values: MLXArray
    var rawSDPAQuery: MLXArray
    var rawSDPAKeys: MLXArray
    var rawSDPAValues: MLXArray
    var scale: Float
}

private func makeCoreAttentionInputs(options: BenchmarkOptions) -> CoreAttentionInputs {
    let query = MLXArray(
        values(
            count: options.batchSize * options.queryHeadCount * options.queryLength
                * options.headDimension,
            scale: 0.019
        ),
        [options.batchSize, options.queryHeadCount, options.queryLength, options.headDimension]
    )
    let keys = MLXArray(
        values(
            count: options.batchSize * options.kvHeadCount * options.contextTokens
                * options.headDimension,
            scale: 0.007,
            phase: 0.1
        ),
        [options.batchSize, options.kvHeadCount, options.contextTokens, options.headDimension]
    )
    let valuesArray = MLXArray(
        values(
            count: options.batchSize * options.kvHeadCount * options.contextTokens
                * options.headDimension,
            scale: 0.009,
            phase: 0.2
        ),
        [options.batchSize, options.kvHeadCount, options.contextTokens, options.headDimension]
    )
    let rawSDPAQuery = query.asType(.float16)
    let rawSDPAKeys = keys.asType(.float16)
    let rawSDPAValues = valuesArray.asType(.float16)
    eval(rawSDPAQuery, rawSDPAKeys, rawSDPAValues)
    return CoreAttentionInputs(
        query: query,
        keys: keys,
        values: valuesArray,
        rawSDPAQuery: rawSDPAQuery,
        rawSDPAKeys: rawSDPAKeys,
        rawSDPAValues: rawSDPAValues,
        scale: 1 / sqrt(Float(options.headDimension))
    )
}

private func rawSDPAAttention(inputs: CoreAttentionInputs) -> MLXArray {
    MLXFast.scaledDotProductAttention(
        queries: inputs.rawSDPAQuery,
        keys: inputs.rawSDPAKeys,
        values: inputs.rawSDPAValues,
        scale: inputs.scale,
        mask: .causal
    )
    .asType(.float32)
}

private func twoStageAttention(
    query: MLXArray,
    keyCode: TurboQuantAttentionCode,
    valueCode: TurboQuantAttentionCode,
    scale: Float
) throws -> MLXArray {
    let scores = try turboQuantMetalQK(
        queries: query,
        keyCode: keyCode,
        scale: scale,
        mask: .causal
    )
    let weights = softmax(scores.asType(.float32), axis: -1)
    eval(weights)
    return try turboQuantMetalAV(
        attentionWeights: weights,
        valueCode: valueCode,
        outputDType: .float32
    )
}

private func pathMeasurement(
    path: TurboQuantAttentionPath,
    status: TurboQuantCoreBenchmarkPathStatus,
    selected: Bool,
    validForRequest: Bool,
    queryLength: Int,
    reason: String? = nil,
    timing: TimingSummary? = nil,
    output: MLXArray? = nil,
    rawReference: TimedValueSummary<MLXArray>? = nil,
    compressedKVBytes: Int? = nil,
    rawSDPAKVBytes: Int? = nil,
    actualBitsPerValue: Double? = nil
) -> TurboQuantCoreBenchmarkPathMeasurement {
    let p50TokensPerSecond = tokensPerSecond(queryLength: queryLength, seconds: timing?.p50Seconds)
    let p95TokensPerSecond = tokensPerSecond(queryLength: queryLength, seconds: timing?.p95Seconds)
    let rawP50TokensPerSecond = tokensPerSecond(
        queryLength: queryLength,
        seconds: rawReference?.timing.p50Seconds
    )
    let rawP95TokensPerSecond = tokensPerSecond(
        queryLength: queryLength,
        seconds: rawReference?.timing.p95Seconds
    )
    let quality = output.flatMap { output in
        rawReference.map { qualityMetrics($0.value, output) }
    }
    let savedBytes = memoryBytesSaved(compressedBytes: compressedKVBytes, rawBytes: rawSDPAKVBytes)
    return TurboQuantCoreBenchmarkPathMeasurement(
        path: path,
        route: benchmarkRoute(for: path).rawValue,
        backend: benchmarkBackend(for: path).rawValue,
        status: status,
        selected: selected,
        validForRequest: validForRequest,
        referenceDType: rawReference == nil ? nil : "float16",
        reason: reason,
        latencyMSAverage: milliseconds(timing?.averageSeconds),
        latencyMSP50: milliseconds(timing?.p50Seconds),
        latencyMSP95: milliseconds(timing?.p95Seconds),
        decodeTokensPerSecondP50: p50TokensPerSecond,
        decodeTokensPerSecondP95: p95TokensPerSecond,
        rawSDPALatencyMSP50: milliseconds(rawReference?.timing.p50Seconds),
        rawSDPALatencyMSP95: milliseconds(rawReference?.timing.p95Seconds),
        rawSDPADecodeTokensPerSecondP50: rawP50TokensPerSecond,
        rawSDPADecodeTokensPerSecondP95: rawP95TokensPerSecond,
        speedRatioToRawSDPAP50: speedRatio(p50TokensPerSecond, rawP50TokensPerSecond),
        speedRatioToRawSDPAP95: speedRatio(p95TokensPerSecond, rawP95TokensPerSecond),
        compressedKVBytes: compressedKVBytes,
        rawSDPAKVBytes: rawSDPAKVBytes,
        memoryBytesSavedVsRawSDPA: savedBytes,
        memoryReductionRatio: memoryReductionRatio(
            compressedBytes: compressedKVBytes,
            rawBytes: rawSDPAKVBytes
        ),
        memoryReductionPercent: memoryReductionPercent(savedBytes: savedBytes, rawBytes: rawSDPAKVBytes),
        actualBitsPerValue: actualBitsPerValue,
        maxAbsoluteErrorVsRawSDPA: quality.map { Double($0.maxAbsoluteError) },
        cosineSimilarityVsRawSDPA: quality.map { Double($0.cosineSimilarity) }
    )
}

private func tokensPerSecond(queryLength: Int, seconds: Double?) -> Double? {
    guard let seconds, seconds > 0 else { return nil }
    return Double(queryLength) / seconds
}

private func speedRatio(_ candidate: Double?, _ reference: Double?) -> Double? {
    guard let candidate, let reference, reference > 0 else { return nil }
    return candidate / reference
}

private func memoryBytesSaved(compressedBytes: Int?, rawBytes: Int?) -> Int? {
    guard let compressedBytes, let rawBytes else { return nil }
    return max(0, rawBytes - compressedBytes)
}

private func memoryReductionRatio(compressedBytes: Int?, rawBytes: Int?) -> Double? {
    guard let compressedBytes, let rawBytes, compressedBytes > 0 else { return nil }
    return Double(rawBytes) / Double(compressedBytes)
}

private func memoryReductionPercent(savedBytes: Int?, rawBytes: Int?) -> Double? {
    guard let savedBytes, let rawBytes, rawBytes > 0 else { return nil }
    return Double(savedBytes) / Double(rawBytes) * 100
}

private func measureCoreAttention(
    options: BenchmarkOptions,
    decision: TurboQuantAttentionDecision,
    availability: TurboQuantKernelAvailability
) throws -> CoreAttentionMeasurement {
    let inputs = makeCoreAttentionInputs(options: options)
    let rawSDPAKVBytes = inputs.rawSDPAKeys.nbytes + inputs.rawSDPAValues.nbytes
    let rawReference = try timedSampled(
        iterations: options.iterations,
        warmup: options.warmup,
        cooldownMilliseconds: options.cooldownMilliseconds
    ) {
        rawSDPAAttention(inputs: inputs)
    }
    cooldown(milliseconds: options.pathCooldownMilliseconds)

    let (encodeSeconds, codes) = try timedValue(
        iterations: options.iterations,
        warmup: options.warmup,
        cooldownMilliseconds: options.cooldownMilliseconds,
        evaluate: evaluateAttentionCodes
    ) {
        let keyCode = try turboQuantMetalEncodeAttention(
            inputs.keys,
            configuration: TurboQuantConfiguration(
                preset: options.preset,
                role: .key,
                groupSize: options.groupSize,
                backend: .metalPolarQJL,
                seed: 0xBEEF_0000_0000_0101,
                attentionLayoutVersion: options.layoutVersion,
                allowExperimentalLayoutV5: options.enableLayoutV5,
                allowExperimentalLayoutV7: options.enableLayoutV7,
                attentionScaleStorage: options.scaleStorage
            )
        )
        let valueCode = try turboQuantMetalEncodeAttention(
            inputs.values,
            configuration: TurboQuantConfiguration(
                preset: options.preset,
                role: .value,
                groupSize: options.groupSize,
                backend: .metalPolarQJL,
                seed: 0xBEEF_0000_0000_0102,
                valueBits: options.resolvedValueBits,
                attentionLayoutVersion: options.layoutVersion,
                allowExperimentalLayoutV5: options.enableLayoutV5,
                allowExperimentalLayoutV7: options.enableLayoutV7,
                attentionScaleStorage: options.scaleStorage
            )
        )
        return (keyCode, valueCode)
    }

    let keyCode = codes.0
    let valueCode = codes.1
    let polarWHTValueCode = try turboQuantMetalPolarWHTEncodeAttentionValues(
        inputs.values,
        bits: options.resolvedValueBits,
        seed: 0xBEEF_0000_0000_0302
    )
    evaluatePolarWHTAttentionValueCode(polarWHTValueCode)
    let storageEstimate = actualAggregateStorageEstimate(keyCode: keyCode, valueCode: valueCode)
    let hybridStorageEstimate = actualHybridAggregateStorageEstimate(
        keyCode: keyCode,
        valueCode: polarWHTValueCode
    )
    let compressedKVBytes = storageEstimate.totalBytes
    let actualBitsPerValue = storageEstimate.actualBitsPerValue
    cooldown(milliseconds: options.pathCooldownMilliseconds)

    // `turboQuantMetalDecodeAttention` is a standalone dequantize-only diagnostic kernel
    // that is intentionally NOT layout-v7-aware (the tile-transposed v7 layout is only
    // consumed by the QK/AV/pair/quad/fused kernels below). Skip this diagnostic leg for
    // v7 codes rather than loosening the kernel's own admission check.
    let decodeSeconds: Double?
    if keyCode.layout.layoutVersion == TurboQuantAttentionLayout.tileTransposedVersion
        || valueCode.layout.layoutVersion == TurboQuantAttentionLayout.tileTransposedVersion
    {
        decodeSeconds = nil
    } else {
        let (measuredDecodeSeconds, _) = try timedValue(
            iterations: options.iterations,
            warmup: options.warmup,
            cooldownMilliseconds: options.cooldownMilliseconds,
            evaluate: { eval($0.0, $0.1) }
        ) {
            (
                try turboQuantMetalDecodeAttention(keyCode, outputDType: .float32),
                try turboQuantMetalDecodeAttention(valueCode, outputDType: .float32)
            )
        }
        decodeSeconds = measuredDecodeSeconds
    }
    cooldown(milliseconds: options.pathCooldownMilliseconds)

    let (qkSeconds, scores) = try timed(
        iterations: options.iterations,
        warmup: options.warmup,
        cooldownMilliseconds: options.cooldownMilliseconds
    ) {
        try turboQuantMetalQK(
            queries: inputs.query,
            keyCode: keyCode,
            scale: inputs.scale,
            mask: .causal
        )
    }
    let weights = softmax(scores.asType(.float32), axis: -1)
    eval(weights)

    let (avSeconds, _) = try timed(
        iterations: options.iterations,
        warmup: options.warmup,
        cooldownMilliseconds: options.cooldownMilliseconds
    ) {
        try turboQuantMetalAV(
            attentionWeights: weights,
            valueCode: valueCode,
            outputDType: .float32
        )
    }
    cooldown(milliseconds: options.pathCooldownMilliseconds)

    var pathMeasurements: [TurboQuantCoreBenchmarkPathMeasurement] = []
    var timingByPath: [TurboQuantAttentionPath: TimingSummary] = [:]

    func appendMeasurement(
        _ path: TurboQuantAttentionPath,
        status: TurboQuantCoreBenchmarkPathStatus,
        validForRequest: Bool,
        reason: String? = nil,
        timing: TimingSummary? = nil,
        output: MLXArray? = nil,
        compressedBytes: Int? = compressedKVBytes,
        bitsPerValue: Double? = actualBitsPerValue
    ) {
        if let timing, status == .measured || status == .reference {
            timingByPath[path] = timing
        }
        pathMeasurements.append(
            pathMeasurement(
                path: path,
                status: status,
                selected: decision.selectedPath == path,
                validForRequest: validForRequest,
                queryLength: options.queryLength,
                reason: reason,
                timing: timing,
                output: output,
                rawReference: rawReference,
                compressedKVBytes: compressedBytes,
                rawSDPAKVBytes: rawSDPAKVBytes,
                actualBitsPerValue: bitsPerValue
            )
        )
    }

    appendMeasurement(
        .baseline,
        status: .reference,
        validForRequest: true,
        reason: "FP16 raw SDPA reference",
        timing: rawReference.timing,
        output: rawReference.value,
        compressedBytes: rawSDPAKVBytes,
        bitsPerValue: 16
    )

    let nativeOptions =
        sparseVNativeAttentionOptions(benchmark: options, scale: inputs.scale, diagnostics: false)
        ?? TurboQuantNativeAttentionOptions(
            scale: inputs.scale,
            causal: true,
            backendVersion: availability.attentionCapabilities.nativeBackendVersion
                ?? TurboQuantNativeAttentionOptions.backendVersion
        )

    func nativePathSkipReason(_ path: TurboQuantAttentionPath) -> String? {
        switch path {
        case .affineK8V4Native where options.resolvedValueBits != 4:
            return "affine K8/V4 evidence requires --value-bits 4"
        case .affineK8VxNative where options.resolvedValueBits >= 4,
            .affineK8VxResidual where options.resolvedValueBits >= 4:
            return "affine K8/Vx evidence requires --value-bits below 4"
        default:
            return nil
        }
    }

    for path in [
        TurboQuantAttentionPath.nativeMLXCompressed,
        .affineK8V4Native,
        .affineK8VxNative,
        .affineK8VxResidual,
    ] {
        if availability.attentionCapabilities.nativeCompressedAttention != true {
            appendMeasurement(
                path,
                status: .unavailable,
                validForRequest: false,
                reason: availability.attentionCapabilities.nativeFallbackReason
                    ?? "native compressed attention capability is unavailable"
            )
            continue
        }
        if let reason = nativePathSkipReason(path) {
            appendMeasurement(path, status: .skipped, validForRequest: false, reason: reason)
            continue
        }
        do {
            let measured = try timedSampled(
                iterations: options.iterations,
                warmup: options.warmup,
                cooldownMilliseconds: options.cooldownMilliseconds
            ) {
                try turboQuantNativeScaledDotProductAttention(
                    queries: inputs.query,
                    keyCode: keyCode,
                    valueCode: valueCode,
                    options: nativeOptions
                )
            }
            appendMeasurement(
                path,
                status: .measured,
                validForRequest: true,
                timing: measured.timing,
                output: measured.value
            )
        } catch {
            appendMeasurement(
                path,
                status: .failed,
                validForRequest: true,
                reason: String(describing: error)
            )
        }
        cooldown(milliseconds: options.pathCooldownMilliseconds)
    }

    do {
        let routed = selectTurboQuantAttentionPath(
            request: coreAttentionRequest(options: options, preferOnlineFused: true),
            capabilities: availability.attentionCapabilities
        )
        for path in [TurboQuantAttentionPath.onlineFused, .tiledOnlineFused] {
            let capabilityAvailable =
                path == .onlineFused
                ? availability.attentionCapabilities.onlineFused
                : availability.attentionCapabilities.tiledOnlineFused
            guard capabilityAvailable else {
                appendMeasurement(
                    path,
                    status: .unavailable,
                    validForRequest: false,
                    reason: "\(path.rawValue) capability is unavailable"
                )
                continue
            }
            guard routed.selectedPath == path else {
                appendMeasurement(
                    path,
                    status: .notCallable,
                    validForRequest: false,
                    reason:
                        "public compressed attention wrapper routes this request to \(routed.selectedPath.rawValue)"
                )
                continue
            }
            let measured = try timedSampled(
                iterations: options.iterations,
                warmup: options.warmup,
                cooldownMilliseconds: options.cooldownMilliseconds
            ) {
                try turboQuantMetalScaledDotProductAttention(
                    queries: inputs.query,
                    keyCode: keyCode,
                    valueCode: valueCode,
                    scale: inputs.scale,
                    mask: .causal,
                    preferOnlineFused: true,
                    blockParallelTokenBlockSize: options.blockParallelTokenBlockSize
                )
            }
            appendMeasurement(
                path,
                status: .measured,
                validForRequest: true,
                timing: measured.timing,
                output: measured.value
            )
            cooldown(milliseconds: options.pathCooldownMilliseconds)
        }
    } catch {
        appendMeasurement(
            .onlineFused,
            status: .failed,
            validForRequest: true,
            reason: String(describing: error)
        )
        appendMeasurement(
            .tiledOnlineFused,
            status: .failed,
            validForRequest: true,
            reason: String(describing: error)
        )
    }

    if availability.attentionCapabilities.qk && availability.attentionCapabilities.av {
        do {
            let measured = try timedSampled(
                iterations: options.iterations,
                warmup: options.warmup,
                cooldownMilliseconds: options.cooldownMilliseconds
            ) {
                try twoStageAttention(
                    query: inputs.query,
                    keyCode: keyCode,
                    valueCode: valueCode,
                    scale: inputs.scale
                )
            }
            appendMeasurement(
                .twoStageCompressed,
                status: .measured,
                validForRequest: true,
                timing: measured.timing,
                output: measured.value
            )
        } catch {
            appendMeasurement(
                .twoStageCompressed,
                status: .failed,
                validForRequest: true,
                reason: String(describing: error)
            )
        }
    } else {
        appendMeasurement(
            .twoStageCompressed,
            status: .unavailable,
            validForRequest: false,
            reason: "compressed QK/AV capabilities are unavailable"
        )
    }
    cooldown(milliseconds: options.pathCooldownMilliseconds)

    if availability.attentionCapabilities.hybridK8PolarWHTValueAttention {
        do {
            let measured = try timedSampled(
                iterations: options.iterations,
                warmup: options.warmup,
                cooldownMilliseconds: options.cooldownMilliseconds
            ) {
                try turboQuantMetalHybridPolarWHTValueScaledDotProductAttention(
                    queries: inputs.query,
                    keyCode: keyCode,
                    valueCode: polarWHTValueCode,
                    scale: inputs.scale,
                    mask: .causal,
                    outputDType: .float32
                )
            }
            appendMeasurement(
                .metalHybridK8PolarWHTValue,
                status: .measured,
                validForRequest: true,
                timing: measured.timing,
                output: measured.value,
                compressedBytes: hybridStorageEstimate.totalBytes,
                bitsPerValue: hybridStorageEstimate.actualBitsPerValue
            )
        } catch {
            appendMeasurement(
                .metalHybridK8PolarWHTValue,
                status: .failed,
                validForRequest: true,
                reason: String(describing: error),
                compressedBytes: hybridStorageEstimate.totalBytes,
                bitsPerValue: hybridStorageEstimate.actualBitsPerValue
            )
        }
    } else {
        appendMeasurement(
            .metalHybridK8PolarWHTValue,
            status: .unavailable,
            validForRequest: false,
            reason: "hybrid K8 + PolarWHT-V attention capability is unavailable",
            compressedBytes: hybridStorageEstimate.totalBytes,
            bitsPerValue: hybridStorageEstimate.actualBitsPerValue
        )
    }
    cooldown(milliseconds: options.pathCooldownMilliseconds)

    appendMeasurement(
        .sparseValueTwoStageCompressed,
        status: options.sparseVEnabled ? .notCallable : .skipped,
        validForRequest: false,
        reason: options.sparseVEnabled
            ? "Sparse-V core primitive evidence is emitted through native compressed path rows"
            : "pass --sparse-v to request Sparse-V evidence"
    )
    appendMeasurement(
        .affineInt4Native,
        status: .notCallable,
        validForRequest: false,
        reason: "affine int4 native is a router label; no public primitive dispatcher is exposed"
    )
    appendMeasurement(
        .mlxPackedFallback,
        status: .skipped,
        validForRequest: false,
        reason: "MLX packed fallback is a compatibility fallback, not an optimization path"
    )
    appendMeasurement(
        .unavailable,
        status: .unavailable,
        validForRequest: false,
        reason: "sentinel path"
    )

    let selectedAttentionTiming = timingByPath[decision.selectedPath]
    let fusedSeconds: Double?
    if decision.selectedPath == .onlineFused || decision.selectedPath == .tiledOnlineFused {
        fusedSeconds = selectedAttentionTiming?.averageSeconds
    } else {
        fusedSeconds = nil
    }

    let selectedStorageEstimate =
        decision.selectedPath == .metalHybridK8PolarWHTValue
        ? hybridStorageEstimate
        : storageEstimate

    return CoreAttentionMeasurement(
        storageEstimate: selectedStorageEstimate,
        encodeSeconds: encodeSeconds,
        decodeSeconds: decodeSeconds,
        qkSeconds: qkSeconds,
        avSeconds: avSeconds,
        fusedSeconds: fusedSeconds,
        attentionTiming: selectedAttentionTiming,
        rawSDPAAttentionTiming: rawReference.timing,
        rawSDPAKVBytes: rawSDPAKVBytes,
        pathMeasurements: pathMeasurements
    )
}

private func sparseVModeLabel(_ mode: BenchmarkSparseVSelectionMode) -> String {
    switch mode {
    case .off:
        return "off"
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

private func sparseVNativeAttentionOptions(
    benchmark options: BenchmarkOptions,
    scale: Float,
    diagnostics: Bool
) -> TurboQuantNativeAttentionOptions? {
    guard let mode = options.sparseVSelectionMode, mode != .off else {
        return nil
    }

    switch mode {
    case .off:
        return nil
    case .threshold:
        return TurboQuantNativeAttentionOptions(
            scale: scale,
            causal: true,
            sparseVThreshold: options.resolvedSparseVThreshold,
            sparseVSelectionMode: .threshold,
            diagnostics: diagnostics
        )
    case .blockThreshold:
        return TurboQuantNativeAttentionOptions(
            scale: scale,
            causal: true,
            sparseVThreshold: options.resolvedSparseVThreshold,
            sparseVSelectionMode: .blockThreshold,
            diagnostics: diagnostics
        )
    case .topK:
        return TurboQuantNativeAttentionOptions(
            scale: scale,
            causal: true,
            sparseVSelectionMode: .topK,
            sparseVTopK: options.resolvedSparseVTopK,
            diagnostics: diagnostics
        )
    case .pageTopK:
        return TurboQuantNativeAttentionOptions(
            scale: scale,
            causal: true,
            sparseVSelectionMode: .pageTopK,
            sparseVTopK: options.resolvedSparseVTopK,
            diagnostics: diagnostics
        )
    case .candidateSparse:
        return TurboQuantNativeAttentionOptions(
            scale: scale,
            causal: true,
            sparseVSelectionMode: .pageTopK,
            sparseVTopK: options.resolvedSparseVTopK,
            sparseVRecentTokens: options.resolvedSparseVRecentTokens,
            sparseVCandidatePages: options.resolvedSparseVCandidatePages,
            diagnostics: diagnostics
        )
    case .cumulativeMass:
        return TurboQuantNativeAttentionOptions(
            scale: scale,
            causal: true,
            sparseVSelectionMode: .cumulativeMass,
            sparseVCumulativeMass: options.resolvedSparseVCumulativeMass,
            diagnostics: diagnostics
        )
    case .hybridCumulativeMassTopK:
        return TurboQuantNativeAttentionOptions(
            scale: scale,
            causal: true,
            sparseVSelectionMode: .hybridCumulativeMassTopK,
            sparseVCumulativeMass: options.resolvedSparseVCumulativeMass,
            sparseVMaxTopK: options.resolvedSparseVMaxTopK,
            diagnostics: diagnostics
        )
    }
}

private func runNativeSparseVBenchmark(options: BenchmarkOptions) throws -> BenchmarkResult {
    guard let mode = options.sparseVSelectionMode, mode != .off else {
        return skipped("attention.native_sparse", reason: "Sparse-V benchmark not requested")
    }
    guard options.queryLength == 1 else {
        return skipped(
            "attention.native_sparse.\(sparseVModeLabel(mode))",
            reason: "native Sparse-V benchmark is decode-only; pass --query-length 1")
    }
    guard options.queryHeadCount % options.kvHeadCount == 0 else {
        throw TurboQuantError.invalidMetalConfiguration(
            "query heads must be a multiple of KV heads")
    }

    let q = MLXArray(
        values(
            count: options.batchSize * options.queryHeadCount * options.queryLength
                * options.headDimension,
            scale: 0.019
        ),
        [options.batchSize, options.queryHeadCount, options.queryLength, options.headDimension]
    )
    let k = MLXArray(
        values(
            count: options.batchSize * options.kvHeadCount * options.contextTokens
                * options.headDimension,
            scale: 0.007,
            phase: 0.1
        ),
        [options.batchSize, options.kvHeadCount, options.contextTokens, options.headDimension]
    )
    let v = MLXArray(
        values(
            count: options.batchSize * options.kvHeadCount * options.contextTokens
                * options.headDimension,
            scale: 0.009,
            phase: 0.2
        ),
        [options.batchSize, options.kvHeadCount, options.contextTokens, options.headDimension]
    )
    let keyCode = try turboQuantMetalEncodeAttention(
        k,
        configuration: TurboQuantConfiguration(
            preset: options.preset,
            role: .key,
            groupSize: options.groupSize,
            backend: .metalPolarQJL,
            seed: 0xBEEF_0000_0000_0201,
            attentionLayoutVersion: options.layoutVersion,
            allowExperimentalLayoutV5: options.enableLayoutV5,
            allowExperimentalLayoutV7: options.enableLayoutV7,
            attentionScaleStorage: options.scaleStorage
        )
    )
    let valueCode = try turboQuantMetalEncodeAttention(
        v,
        configuration: TurboQuantConfiguration(
            preset: options.preset,
            role: .value,
            groupSize: options.groupSize,
            backend: .metalPolarQJL,
            seed: 0xBEEF_0000_0000_0202,
            valueBits: options.resolvedValueBits,
            attentionLayoutVersion: options.layoutVersion,
            allowExperimentalLayoutV5: options.enableLayoutV5,
            allowExperimentalLayoutV7: options.enableLayoutV7,
            attentionScaleStorage: options.scaleStorage
        )
    )
    let scale = 1 / sqrt(Float(options.headDimension))
    let denseOptions = TurboQuantNativeAttentionOptions(scale: scale, causal: true)
    let sparseOptions = sparseVNativeAttentionOptions(
        benchmark: options,
        scale: scale,
        diagnostics: false
    )!
    let sparseDiagnosticOptions = sparseVNativeAttentionOptions(
        benchmark: options,
        scale: scale,
        diagnostics: true
    )!
    let keyPageSummary =
        (mode == .pageTopK || mode == .candidateSparse) && options.sparseVUsePageSummary
        ? try turboQuantKeyPageSummaries(keyCode: keyCode)
        : nil

    let dense = try timedSampled(iterations: options.iterations, warmup: options.warmup) {
        try turboQuantNativeScaledDotProductAttention(
            queries: q,
            keyCode: keyCode,
            valueCode: valueCode,
            options: denseOptions
        )
    }
    let sparse = try timedSampled(iterations: options.iterations, warmup: options.warmup) {
        try turboQuantNativeScaledDotProductAttention(
            queries: q,
            keyCode: keyCode,
            valueCode: valueCode,
            options: sparseOptions,
            keyPageSummary: keyPageSummary
        )
    }
    let diagnostic = try turboQuantNativeScaledDotProductAttentionWithDiagnostics(
        queries: q,
        keyCode: keyCode,
        valueCode: valueCode,
        options: sparseDiagnosticOptions,
        keyPageSummary: keyPageSummary
    )
    eval(diagnostic.output)

    let codeMemoryBytes = keyCode.storageByteCount + valueCode.storageByteCount
    let codeValueCount =
        keyCode.layout.batchSize * keyCode.layout.kvHeadCount
        * max(keyCode.layout.logicalLength, 1) * keyCode.layout.headDimension
        + valueCode.layout.batchSize * valueCode.layout.kvHeadCount
        * max(valueCode.layout.logicalLength, 1) * valueCode.layout.headDimension
    let actualBitsPerValue = Double(codeMemoryBytes * 8) / Double(codeValueCount)
    let diagnostics = diagnostic.diagnostics

    return BenchmarkResult(
        name: "attention.native_sparse.\(sparseVModeLabel(mode))",
        status: "ok",
        selectedPath: TurboQuantAttentionPath.nativeMLXCompressed.rawValue,
        dtype: "\(sparse.value.dtype)",
        shape: sparse.value.shape,
        queryShape: q.shape,
        keyShape: k.shape,
        valueShape: v.shape,
        preset: keyCode.preset,
        valueBits: valueCode.valueBits,
        actualBitsPerValue: actualBitsPerValue,
        memoryBytes: codeMemoryBytes,
        latencySeconds: sparse.timing.averageSeconds,
        denseReferenceLatencySeconds: dense.timing.averageSeconds,
        quality: qualityMetrics(dense.value, sparse.value),
        sparseVSelectionMode: sparseVModeLabel(mode),
        sparseVThreshold: mode == .threshold || mode == .blockThreshold
            ? options.resolvedSparseVThreshold : nil,
        sparseVTopK: mode == .topK || mode == .pageTopK || mode == .candidateSparse
            ? options.resolvedSparseVTopK : nil,
        sparseVCumulativeMass: mode == .cumulativeMass || mode == .hybridCumulativeMassTopK
            ? options.resolvedSparseVCumulativeMass : nil,
        sparseVMaxTopK: mode == .hybridCumulativeMassTopK ? options.resolvedSparseVMaxTopK : nil,
        sparseVRecentTokenCount: options.sparseVRecentTokenCount,
        sparseVOlderTokenCount: options.sparseVOlderTokenCount,
        sparseVPageCandidateCount: options.sparseVPageCandidateCount,
        sparseVPageSummary: mode == .pageTopK || mode == .candidateSparse
            ? options.sparseVUsePageSummary : nil,
        sparseVSkippedTokens: diagnostics?.sparseSkippedTokens,
        sparseVTotalTokens: diagnostics?.sparseTotalTokens,
        sparseVSkipRatio: diagnostics?.sparseSkipRatio,
        activeBlocks: diagnostics?.activeBlocks,
        blockTokens: diagnostics?.blockTokens,
        kernelKind: diagnostics?.kernelKind
    )
}

private func runLegacyBenchmark(options: BenchmarkOptions) throws {
    let availability = TurboQuantKernelAvailability.current
    var results: [BenchmarkResult] = []

    if availability.supportsMetalPolarQJLCodec {
        do {
            let input = MLXArray(values(count: 4 * 64, scale: 0.011), [4, 64])
            let configuration = TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .key,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 0xBEEF_0000_0000_0001
            )
            let code = try turboQuantMetalEncode(input, configuration: configuration)
            let (latency, decoded) = try timed(
                iterations: options.iterations,
                warmup: options.warmup
            ) {
                try turboQuantMetalDecode(code, dtype: .float32)
            }
            results.append(
                BenchmarkResult(
                    name: "flat.decode",
                    status: "ok",
                    selectedPath: "decodeCompressed",
                    dtype: "\(decoded.dtype)",
                    shape: decoded.shape,
                    preset: configuration.preset,
                    valueBits: configuration.resolvedValueBits,
                    actualBitsPerValue: code.approximateBitsPerValue,
                    memoryBytes: code.storageByteCount,
                    latencySeconds: latency,
                    quality: qualityMetrics(input, decoded)
                ))
        } catch {
            results.append(
                BenchmarkResult(name: "flat.decode", status: "failed", shape: [], error: "\(error)")
            )
        }

        do {
            let x = MLXArray(values(count: 8 * 64, scale: 0.017), [8, 64])
            let w = MLXArray(values(count: 16 * 64, scale: 0.023, phase: 0.3), [16, 64])
            let configuration = TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .key,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 0xBEEF_0000_0000_0002
            )
            let code = try turboQuantMetalEncode(w, configuration: configuration)
            let (latency, output) = try timed(
                iterations: options.iterations,
                warmup: options.warmup
            ) {
                try turboQuantizedMM(x, code, transpose: true, outputDType: .float32)
            }
            results.append(
                BenchmarkResult(
                    name: "flat.matmul.product_estimator",
                    status: availability.kernelCapabilities.linearMatmul
                        ? "production" : "experimental",
                    selectedPath: "linearMatmul",
                    dtype: "\(output.dtype)",
                    shape: output.shape,
                    preset: configuration.preset,
                    valueBits: configuration.resolvedValueBits,
                    actualBitsPerValue: code.approximateBitsPerValue,
                    memoryBytes: code.storageByteCount,
                    latencySeconds: latency
                ))
        } catch {
            results.append(
                BenchmarkResult(
                    name: "flat.matmul.product_estimator", status: "failed", shape: [],
                    error: "\(error)"))
        }
    } else {
        results.append(skipped("flat.decode", reason: "Metal codec unavailable"))
        results.append(skipped("flat.matmul.product_estimator", reason: "Metal codec unavailable"))
    }

    if availability.supportsMetalPolarQJLAttention {
        do {
            let q = MLXArray(values(count: 1 * 4 * 2 * 128, scale: 0.019), [1, 4, 2, 128])
            let k = MLXArray(
                values(count: 1 * 2 * 256 * 128, scale: 0.007, phase: 0.1), [1, 2, 256, 128])
            let v = MLXArray(
                values(count: 1 * 2 * 256 * 128, scale: 0.009, phase: 0.2), [1, 2, 256, 128])
            let keyCode = try turboQuantMetalEncodeAttention(
                k,
                configuration: TurboQuantConfiguration(
                    preset: .turbo4v2,
                    role: .key,
                    groupSize: 64,
                    backend: .metalPolarQJL,
                    seed: 0xBEEF_0000_0000_0003
                ))
            let valueCode = try turboQuantMetalEncodeAttention(
                v,
                configuration: TurboQuantConfiguration(
                    preset: .turbo4v2,
                    role: .value,
                    groupSize: 64,
                    backend: .metalPolarQJL,
                    seed: 0xBEEF_0000_0000_0004,
                    valueBits: 4
                ))
            let scale = 1 / sqrt(Float(q.dim(-1)))
            let codeMemoryBytes = keyCode.storageByteCount + valueCode.storageByteCount
            let codeValueCount =
                keyCode.layout.batchSize * keyCode.layout.kvHeadCount
                * max(keyCode.layout.logicalLength, 1) * keyCode.layout.headDimension
                + valueCode.layout.batchSize * valueCode.layout.kvHeadCount
                * max(valueCode.layout.logicalLength, 1) * valueCode.layout.headDimension
            let actualBitsPerValue = Double(codeMemoryBytes * 8) / Double(codeValueCount)
            let (twoStageLatency, twoStage) = try timed(
                iterations: options.iterations,
                warmup: options.warmup
            ) {
                try turboQuantMetalScaledDotProductAttention(
                    queries: q,
                    keyCode: keyCode,
                    valueCode: valueCode,
                    scale: scale,
                    mask: .causal,
                    preferOnlineFused: false,
                    blockParallelTokenBlockSize: options.blockParallelTokenBlockSize
                )
            }
            let (fusedLatency, fused) = try timed(
                iterations: options.iterations,
                warmup: options.warmup
            ) {
                try turboQuantMetalScaledDotProductAttention(
                    queries: q,
                    keyCode: keyCode,
                    valueCode: valueCode,
                    scale: scale,
                    mask: .causal,
                    preferOnlineFused: true,
                    blockParallelTokenBlockSize: options.blockParallelTokenBlockSize
                )
            }
            results.append(
                BenchmarkResult(
                    name: "attention.two_stage",
                    status: "ok",
                    selectedPath: TurboQuantAttentionPath.twoStageCompressed.rawValue,
                    dtype: "\(twoStage.dtype)",
                    shape: twoStage.shape,
                    queryShape: q.shape,
                    keyShape: k.shape,
                    valueShape: v.shape,
                    preset: keyCode.preset,
                    valueBits: valueCode.valueBits,
                    actualBitsPerValue: actualBitsPerValue,
                    memoryBytes: codeMemoryBytes,
                    latencySeconds: twoStageLatency
                ))
            results.append(
                BenchmarkResult(
                    name: "attention.fused",
                    status: "ok",
                    selectedPath: TurboQuantAttentionPath.onlineFused.rawValue,
                    dtype: "\(fused.dtype)",
                    shape: fused.shape,
                    queryShape: q.shape,
                    keyShape: k.shape,
                    valueShape: v.shape,
                    preset: keyCode.preset,
                    valueBits: valueCode.valueBits,
                    actualBitsPerValue: actualBitsPerValue,
                    memoryBytes: codeMemoryBytes,
                    latencySeconds: fusedLatency,
                    quality: qualityMetrics(twoStage, fused)
                ))
        } catch {
            results.append(
                BenchmarkResult(name: "attention", status: "failed", shape: [], error: "\(error)")
            )
        }
    } else {
        results.append(skipped("attention", reason: "Metal attention unavailable or probe failed"))
    }

    if options.sparseVSelectionMode != nil {
        if availability.attentionCapabilities.nativeSparseVSupport == true {
            do {
                results.append(try runNativeSparseVBenchmark(options: options))
            } catch {
                results.append(
                    BenchmarkResult(
                        name: "attention.native_sparse",
                        status: "failed",
                        shape: [],
                        error: "\(error)"
                    ))
            }
        } else {
            results.append(
                skipped("attention.native_sparse", reason: "native Sparse-V support unavailable"))
        }
    }

    let report = BenchmarkReport(
        schemaVersion: 2,
        generatedAt: options.includeTimestamp ? ISO8601DateFormatter().string(from: Date()) : nil,
        iterations: options.iterations,
        availability: availability,
        capabilities: availability.kernelCapabilities,
        attentionCapabilities: availability.attentionCapabilities,
        device: TurboQuantDeviceCapabilities.current,
        results: results
    )
    try writeJSON(report)
}

private func corePathDecision(
    options: BenchmarkOptions,
    availability: TurboQuantKernelAvailability
) -> TurboQuantAttentionDecision {
    let request = coreAttentionRequest(
        options: options,
        preferOnlineFused: options.requestedPath != .twoStageCompressed
            && options.requestedPath != .sparseValueTwoStageCompressed
    )

    switch options.requestedPath {
    case .nativeMLXCompressed:
        return forcedFallbackDecision(
            selectedPath: .nativeMLXCompressed,
            outputDType: request.outputDType,
            reason: "caller requested native MLX compressed attention path"
        )
    case .onlineFused:
        return forcedFallbackDecision(
            selectedPath: .onlineFused,
            outputDType: request.outputDType,
            reason: "caller requested online fused compressed attention path"
        )
    case .tiledOnlineFused:
        return forcedFallbackDecision(
            selectedPath: .tiledOnlineFused,
            outputDType: request.outputDType,
            reason: "caller requested tiled online fused compressed attention path"
        )
    case .sparseValueTwoStageCompressed:
        return forcedFallbackDecision(
            selectedPath: .sparseValueTwoStageCompressed,
            outputDType: request.outputDType,
            reason: "caller requested sparse-value two-stage compressed attention path"
        )
    case .metalHybridK8PolarWHTValue:
        return forcedFallbackDecision(
            selectedPath: .metalHybridK8PolarWHTValue,
            outputDType: request.outputDType,
            reason: "caller requested hybrid K8 + PolarWHT-V attention path"
        )
    case .twoStageCompressed:
        return forcedFallbackDecision(
            selectedPath: .twoStageCompressed,
            outputDType: request.outputDType,
            reason: "caller requested two-stage compressed attention path"
        )
    case .unavailable:
        return forcedFallbackDecision(
            selectedPath: .unavailable,
            outputDType: request.outputDType,
            reason: "caller requested unavailable path"
        )
    case .baseline:
        return forcedFallbackDecision(
            selectedPath: .baseline,
            outputDType: request.outputDType,
            reason: "caller requested baseline path"
        )
    case .affineInt4Native:
        return forcedFallbackDecision(
            selectedPath: .affineInt4Native,
            outputDType: request.outputDType,
            reason: "caller requested native affine int4 path"
        )
    case .affineK8V4Native:
        return forcedFallbackDecision(
            selectedPath: .affineK8V4Native,
            outputDType: request.outputDType,
            reason: "caller requested native affine K8/V4 path"
        )
    case .affineK8VxNative:
        return forcedFallbackDecision(
            selectedPath: .affineK8VxNative,
            outputDType: request.outputDType,
            reason: "caller requested native affine K8/Vx path"
        )
    case .affineK8VxResidual:
        return forcedFallbackDecision(
            selectedPath: .affineK8VxResidual,
            outputDType: request.outputDType,
            reason: "caller requested residual affine K8/Vx path"
        )
    case .mlxPackedFallback:
        return forcedFallbackDecision(
            selectedPath: .mlxPackedFallback,
            outputDType: request.outputDType,
            reason: "caller requested MLX packed fallback path"
        )
    default:
        return selectTurboQuantAttentionPath(
            request: request,
            capabilities: availability.attentionCapabilities
        )
    }
}

private func coreAttentionRequest(
    options: BenchmarkOptions,
    preferOnlineFused: Bool
) -> TurboQuantAttentionRequest {
    TurboQuantAttentionRequest(
        queryShape: [
            options.batchSize, options.queryHeadCount, options.queryLength, options.headDimension,
        ],
        keyLayout: symbolicAttentionLayout(options: options, role: .key),
        valueLayout: symbolicAttentionLayout(options: options, role: .value),
        queryDType: .float32,
        outputDType: .float32,
        maskKind: .causal,
        preferOnlineFused: preferOnlineFused,
        fallbackState: TurboQuantAttentionFallbackState(
            packedFallbackAvailable: true,
            baselineAvailable: true
        )
    )
}

private func validateCoreBenchmarkOptions(_ options: BenchmarkOptions) throws {
    guard options.queryHeadCount % options.kvHeadCount == 0 else {
        throw TurboQuantError.invalidMetalConfiguration(
            "query heads must be a multiple of KV heads")
    }
    if options.scaleStorage == .float16 {
        guard options.layoutVersion >= 5, options.enableLayoutV5
        else {
            throw TurboQuantError.invalidMetalConfiguration(
                "float16 attention scale storage requires --enable-layout-v5 and Layout V5 or newer"
            )
        }
    }
}

private func forcedFallbackDecision(
    selectedPath: TurboQuantAttentionPath,
    outputDType: DType,
    reason: String
) -> TurboQuantAttentionDecision {
    let rejectedPaths = [
        TurboQuantAttentionPath.nativeMLXCompressed,
        .onlineFused,
        .tiledOnlineFused,
        .sparseValueTwoStageCompressed,
        .metalHybridK8PolarWHTValue,
        .twoStageCompressed,
        .affineInt4Native,
        .affineK8V4Native,
        .affineK8VxNative,
        .affineK8VxResidual,
        .mlxPackedFallback,
        .baseline,
    ]
    .filter { $0 != selectedPath }
    .map { RejectedPath(path: $0, reason: reason) }

    return TurboQuantAttentionDecision(
        selectedPath: selectedPath,
        outputDType: outputDType,
        rejectedPaths: rejectedPaths
    )
}

private func symbolicAttentionLayout(
    options: BenchmarkOptions,
    role: TurboQuantTensorRole
) -> TurboQuantAttentionLayout {
    let groupsPerVector = ceilDivide(options.headDimension, by: options.groupSize)
    let logicalValues =
        options.batchSize * options.kvHeadCount * options.contextTokens * options.headDimension
    let estimate = estimateTurboQuantStorage(
        role: role,
        logicalValues: logicalValues,
        preset: options.preset,
        valueBits: role == .value ? options.resolvedValueBits : nil,
        groupSize: options.groupSize,
        dtype: .float32,
        scaleStorage: options.scaleStorage
    )
    let groupCount =
        options.batchSize * options.kvHeadCount * options.contextTokens
        * groupsPerVector
    let magnitudeWordsPerGroup = max(
        1,
        estimate.packedBytes / max(1, groupCount * MemoryLayout<UInt32>.size)
    )

    return TurboQuantAttentionLayout(
        layoutVersion: options.layoutVersion,
        batchSize: options.batchSize,
        kvHeadCount: options.kvHeadCount,
        capacity: options.contextTokens,
        logicalLength: options.contextTokens,
        headDimension: options.headDimension,
        groupsPerVector: groupsPerVector,
        magnitudeWordsPerGroup: magnitudeWordsPerGroup,
        bitsetWordsPerGroup: max(1, ceilDivide(options.groupSize, by: 32))
    )
}

private func symbolicAggregateStorageEstimate(options: BenchmarkOptions)
    -> TurboQuantStorageEstimate
{
    let logicalValues =
        options.batchSize * options.kvHeadCount * options.contextTokens * options.headDimension
    let keyEstimate = estimateTurboQuantStorage(
        role: .key,
        logicalValues: logicalValues,
        preset: options.preset,
        groupSize: options.groupSize,
        dtype: .float32,
        scaleStorage: options.scaleStorage
    )
    let valueEstimate = estimateTurboQuantStorage(
        role: .value,
        logicalValues: logicalValues,
        preset: options.preset,
        valueBits: options.resolvedValueBits,
        groupSize: options.groupSize,
        dtype: .float32,
        scaleStorage: options.scaleStorage
    )
    return aggregateStorageEstimate(keyEstimate: keyEstimate, valueEstimate: valueEstimate)
}

private func actualAggregateStorageEstimate(
    keyCode: TurboQuantAttentionCode,
    valueCode: TurboQuantAttentionCode
) -> TurboQuantStorageEstimate {
    aggregateStorageEstimate(
        keyEstimate: estimateTurboQuantStorage(code: keyCode),
        valueEstimate: estimateTurboQuantStorage(code: valueCode)
    )
}

private func actualHybridAggregateStorageEstimate(
    keyCode: TurboQuantAttentionCode,
    valueCode: TurboQuantPolarWHTAttentionValueCode
) -> TurboQuantStorageEstimate {
    let keyEstimate = estimateTurboQuantStorage(code: keyCode)
    let valueEstimate = TurboQuantStorageEstimate(
        role: .value,
        logicalValues: valueCode.logicalValueCount,
        packedBytes: valueCode.storageByteCount,
        bitsetBytes: 0,
        scaleBytes: 0
    )
    return aggregateStorageEstimate(keyEstimate: keyEstimate, valueEstimate: valueEstimate)
}

private func aggregateStorageEstimate(
    keyEstimate: TurboQuantStorageEstimate,
    valueEstimate: TurboQuantStorageEstimate
) -> TurboQuantStorageEstimate {
    TurboQuantStorageEstimate(
        role: .vector,
        logicalValues: keyEstimate.logicalValues + valueEstimate.logicalValues,
        packedBytes: keyEstimate.packedBytes + valueEstimate.packedBytes,
        bitsetBytes: keyEstimate.bitsetBytes + valueEstimate.bitsetBytes,
        scaleBytes: keyEstimate.scaleBytes + valueEstimate.scaleBytes
    )
}

private func evaluateAttentionCodes(
    _ codes: (TurboQuantAttentionCode, TurboQuantAttentionCode)
) {
    evaluateAttentionCode(codes.0)
    evaluateAttentionCode(codes.1)
}

private func evaluateAttentionCode(_ code: TurboQuantAttentionCode) {
    eval(
        code.packedMagnitudes,
        code.signs,
        code.highPrecisionMask,
        code.residualSigns,
        code.scales
    )
}

private func evaluatePolarWHTAttentionValueCode(_ code: TurboQuantPolarWHTAttentionValueCode) {
    eval(code.packedIndices, code.norms)
}

private func milliseconds(_ seconds: Double?) -> Double? {
    seconds.map { $0 * 1000 }
}

private func ceilDivide(_ value: Int, by divisor: Int) -> Int {
    guard value > 0 else { return 0 }
    return (value + divisor - 1) / divisor
}

private func currentGitCommit() -> String? {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    process.arguments = ["git", "rev-parse", "HEAD"]

    let output = Pipe()
    process.standardOutput = output
    process.standardError = Pipe()

    do {
        try process.run()
        process.waitUntilExit()
    } catch {
        return nil
    }

    guard process.terminationStatus == 0 else {
        return nil
    }

    let data = output.fileHandleForReading.readDataToEndOfFile()
    let commit = String(data: data, encoding: .utf8)?
        .trimmingCharacters(in: .whitespacesAndNewlines)
    return commit?.isEmpty == false ? commit : nil
}

extension TurboQuantAttentionPath {
    fileprivate var usesNativeCompressedAttention: Bool {
        switch self {
        case .nativeMLXCompressed, .affineK8V4Native, .affineK8VxNative,
            .affineK8VxResidual:
            return true
        case .onlineFused, .tiledOnlineFused, .sparseValueTwoStageCompressed,
            .twoStageCompressed, .metalPolarWHTHybrid, .metalHybridK8PolarWHTValue,
            .polarWHTReferenceHybrid, .affineInt4Native, .mlxPackedFallback, .baseline,
            .unavailable:
            return false
        }
    }

    fileprivate var usesCompressedMetal: Bool {
        switch self {
        case .nativeMLXCompressed:
            return false
        case .onlineFused, .tiledOnlineFused, .sparseValueTwoStageCompressed,
            .twoStageCompressed, .metalPolarWHTHybrid, .metalHybridK8PolarWHTValue:
            return true
        case .affineInt4Native, .affineK8V4Native, .affineK8VxNative, .affineK8VxResidual,
            .polarWHTReferenceHybrid, .mlxPackedFallback, .baseline, .unavailable:
            return false
        }
    }
}

private let options = try BenchmarkOptions.parse()
if options.emitCoreJSON {
    try runCoreBenchmarkJSON(options: options)
} else {
    try runLegacyBenchmark(options: options)
}
