import Foundation
import MLX

private let benchmarkBatchSize = 1
private let benchmarkQueryHeadCount = 4
private let benchmarkKVHeadCount = 2

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
    var quality: QualityMetrics?
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

private enum BenchmarkCLIError: Error, CustomStringConvertible {
    case invalidInteger(String)
    case invalidPreset(String)
    case invalidPath(String)
    case invalidScaleStorage(String)

    var description: String {
        switch self {
        case .invalidInteger(let flag):
            "Invalid integer value for \(flag)."
        case .invalidPreset(let value):
            "Invalid TurboQuant preset '\(value)'."
        case .invalidPath(let value):
            "Invalid TurboQuant path '\(value)'."
        case .invalidScaleStorage(let value):
            "Invalid TurboQuant scale storage '\(value)'."
        }
    }
}

private struct BenchmarkOptions {
    var emitCoreJSON: Bool
    var includeTimestamp: Bool
    var iterations: Int
    var warmup: Int
    var headDimension: Int
    var contextTokens: Int
    var queryLength: Int
    var preset: TurboQuantPreset
    var valueBits: Int?
    var groupSize: Int
    var layoutVersion: Int
    var enableLayoutV5: Bool
    var scaleStorage: TurboQuantScaleStorage
    var requestedPath: TurboQuantAttentionPath?

    var resolvedValueBits: Int {
        valueBits ?? preset.defaultValueBits
    }

    static func parse(_ arguments: [String] = CommandLine.arguments) throws -> BenchmarkOptions {
        let presetName = stringValue("--preset", in: arguments) ?? TurboQuantPreset.turbo4v2.rawValue
        guard let preset = TurboQuantPreset(rawValue: presetName) else {
            throw BenchmarkCLIError.invalidPreset(presetName)
        }

        return BenchmarkOptions(
            emitCoreJSON: arguments.contains("--json"),
            includeTimestamp: arguments.contains("--include-timestamp"),
            iterations: try intValue("--iterations", in: arguments, default: 10, minimum: 1),
            warmup: try intValue("--warmup", in: arguments, default: 1, minimum: 0),
            headDimension: try intValue("--head-dim", in: arguments, default: 128, minimum: 1),
            contextTokens: try intValue("--context", in: arguments, default: 256, minimum: 1),
            queryLength: try intValue("--query-length", in: arguments, default: 1, minimum: 1),
            preset: preset,
            valueBits: try optionalIntValue("--value-bits", in: arguments, minimum: 1),
            groupSize: try intValue("--group-size", in: arguments, default: 64, minimum: 1),
            layoutVersion: try intValue(
                "--layout-version",
                in: arguments,
                default: TurboQuantAttentionLayout.currentVersion,
                minimum: 1
            ),
            enableLayoutV5: arguments.contains("--enable-layout-v5"),
            scaleStorage: try scaleStorage(in: arguments),
            requestedPath: try requestedPath(in: arguments)
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

    private static func requestedPath(in arguments: [String]) throws -> TurboQuantAttentionPath? {
        guard let value = stringValue("--path", in: arguments), value != "auto" else {
            return nil
        }

        switch value {
        case TurboQuantAttentionPath.onlineFused.rawValue, "online-fused":
            return .onlineFused
        case TurboQuantAttentionPath.tiledOnlineFused.rawValue, "tiled-online-fused":
            return .tiledOnlineFused
        case TurboQuantAttentionPath.twoStageCompressed.rawValue, "two-stage", "two-stage-compressed":
            return .twoStageCompressed
        case TurboQuantAttentionPath.mlxPackedFallback.rawValue, "mlx-packed-fallback":
            return .mlxPackedFallback
        case TurboQuantAttentionPath.baseline.rawValue:
            return .baseline
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
    _ body: () throws -> MLXArray
) throws -> (Double, MLXArray) {
    try timedValue(iterations: iterations, warmup: warmup, evaluate: { eval($0) }, body)
}

private func timedValue<T>(
    iterations: Int,
    warmup: Int,
    evaluate: (T) -> Void,
    _ body: () throws -> T
) throws -> (Double, T) {
    let measuredIterations = max(1, iterations)
    var last: T?

    for _ in 0 ..< max(0, warmup) {
        let value = try body()
        evaluate(value)
        last = value
    }

    let start = Date.timeIntervalSinceReferenceDate
    for _ in 0 ..< measuredIterations {
        let value = try body()
        evaluate(value)
        last = value
    }
    let elapsed = Date.timeIntervalSinceReferenceDate - start

    return (elapsed / Double(measuredIterations), last!)
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
    var prefillTokensPerSecond: Double?
    var decodeTokensPerSecondP50: Double?
    var decodeTokensPerSecondP95: Double?

    if availability.supportsMetalPolarQJLAttention && pathDecision.selectedPath.usesCompressedMetal {
        do {
            let measurement = try measureCoreAttention(options: options, decision: pathDecision)
            storageEstimate = measurement.storageEstimate
            encodeMS = milliseconds(measurement.encodeSeconds)
            decodeMS = milliseconds(measurement.decodeSeconds)
            qkMS = milliseconds(measurement.qkSeconds)
            avMS = milliseconds(measurement.avSeconds)
            fusedMS = milliseconds(measurement.fusedSeconds)

            if let encodeSeconds = measurement.encodeSeconds, encodeSeconds > 0 {
                prefillTokensPerSecond = Double(options.contextTokens) / encodeSeconds
            }

            let attentionSeconds =
                measurement.fusedSeconds ?? measurement.twoStageAttentionSeconds
            if let attentionSeconds, attentionSeconds > 0 {
                firstTokenLatencyMS = milliseconds(attentionSeconds)
                let decodeTokensPerSecond = Double(options.queryLength) / attentionSeconds
                decodeTokensPerSecondP50 = decodeTokensPerSecond
                decodeTokensPerSecondP95 = decodeTokensPerSecond
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
        contextTokens: options.contextTokens,
        headDimension: options.headDimension,
        queryLength: options.queryLength,
        preset: options.preset.rawValue,
        valueBits: options.resolvedValueBits,
        groupSize: options.groupSize,
        layoutVersion: options.layoutVersion,
        scaleStorage: options.scaleStorage.rawValue,
        warmupIterations: options.warmup,
        encodeMS: encodeMS,
        decodeMS: decodeMS,
        qkMS: qkMS,
        avMS: avMS,
        fusedMS: fusedMS,
        firstTokenLatencyMS: firstTokenLatencyMS,
        prefillTokensPerSecond: prefillTokensPerSecond,
        decodeTokensPerSecondP50: decodeTokensPerSecondP50,
        decodeTokensPerSecondP95: decodeTokensPerSecondP95,
        totalBytes: storageEstimate.totalBytes,
        compressedKVBytes: storageEstimate.totalBytes,
        peakMemoryBytes: nil,
        actualBitsPerValue: storageEstimate.actualBitsPerValue,
        fallbackUsed: fallbackUsed,
        fallbackReason: fallbackReason?.isEmpty == true ? nil : fallbackReason,
        memoryWarningsSeen: 0,
        jetsamObserved: false
    )

    let report = TurboQuantCoreBenchmarkReport(
        mlxSwiftCommit: currentGitCommit(),
        capabilities: capabilities,
        storageEstimate: storageEstimate,
        pathDecision: pathDecision,
        metrics: metrics,
        hiddenCopyAudit: hiddenCopyAudit
    )
    try writeJSON(report)
}

private struct CoreAttentionMeasurement {
    var storageEstimate: TurboQuantStorageEstimate
    var encodeSeconds: Double?
    var decodeSeconds: Double?
    var qkSeconds: Double?
    var avSeconds: Double?
    var fusedSeconds: Double?

    var twoStageAttentionSeconds: Double? {
        guard let qkSeconds, let avSeconds else { return nil }
        return qkSeconds + avSeconds
    }
}

private func measureCoreAttention(
    options: BenchmarkOptions,
    decision: TurboQuantAttentionDecision
) throws -> CoreAttentionMeasurement {
    let query = MLXArray(
        values(
            count: benchmarkBatchSize * benchmarkQueryHeadCount * options.queryLength
                * options.headDimension,
            scale: 0.019
        ),
        [benchmarkBatchSize, benchmarkQueryHeadCount, options.queryLength, options.headDimension]
    )
    let keys = MLXArray(
        values(
            count: benchmarkBatchSize * benchmarkKVHeadCount * options.contextTokens
                * options.headDimension,
            scale: 0.007,
            phase: 0.1
        ),
        [benchmarkBatchSize, benchmarkKVHeadCount, options.contextTokens, options.headDimension]
    )
    let valuesArray = MLXArray(
        values(
            count: benchmarkBatchSize * benchmarkKVHeadCount * options.contextTokens
                * options.headDimension,
            scale: 0.009,
            phase: 0.2
        ),
        [benchmarkBatchSize, benchmarkKVHeadCount, options.contextTokens, options.headDimension]
    )

    let (encodeSeconds, codes) = try timedValue(
        iterations: options.iterations,
        warmup: options.warmup,
        evaluate: evaluateAttentionCodes
    ) {
        let keyCode = try turboQuantMetalEncodeAttention(
            keys,
            configuration: TurboQuantConfiguration(
                preset: options.preset,
                role: .key,
                groupSize: options.groupSize,
                backend: .metalPolarQJL,
                seed: 0xBEEF_0000_0000_0101,
                attentionLayoutVersion: options.layoutVersion,
                allowExperimentalLayoutV5: options.enableLayoutV5,
                attentionScaleStorage: options.scaleStorage
            )
        )
        let valueCode = try turboQuantMetalEncodeAttention(
            valuesArray,
            configuration: TurboQuantConfiguration(
                preset: options.preset,
                role: .value,
                groupSize: options.groupSize,
                backend: .metalPolarQJL,
                seed: 0xBEEF_0000_0000_0102,
                valueBits: options.resolvedValueBits,
                attentionLayoutVersion: options.layoutVersion,
                allowExperimentalLayoutV5: options.enableLayoutV5,
                attentionScaleStorage: options.scaleStorage
            )
        )
        return (keyCode, valueCode)
    }

    let keyCode = codes.0
    let valueCode = codes.1
    let scale = 1 / sqrt(Float(options.headDimension))
    let (decodeSeconds, _) = try timedValue(
        iterations: options.iterations,
        warmup: options.warmup,
        evaluate: { eval($0.0, $0.1) }
    ) {
        (
            try turboQuantMetalDecodeAttention(keyCode, outputDType: .float32),
            try turboQuantMetalDecodeAttention(valueCode, outputDType: .float32)
        )
    }

    let (qkSeconds, scores) = try timed(iterations: options.iterations, warmup: options.warmup) {
        try turboQuantMetalQK(
            queries: query,
            keyCode: keyCode,
            scale: scale,
            mask: .causal
        )
    }
    let weights = softmax(scores.asType(.float32), axis: -1)
    eval(weights)

    let (avSeconds, _) = try timed(iterations: options.iterations, warmup: options.warmup) {
        try turboQuantMetalAV(
            attentionWeights: weights,
            valueCode: valueCode,
            outputDType: .float32
        )
    }

    let fusedSeconds: Double?
    if decision.selectedPath == .onlineFused || decision.selectedPath == .tiledOnlineFused {
        fusedSeconds = try timed(iterations: options.iterations, warmup: options.warmup) {
            try turboQuantMetalScaledDotProductAttention(
                queries: query,
                keyCode: keyCode,
                valueCode: valueCode,
                scale: scale,
                mask: .causal,
                preferOnlineFused: true
            )
        }.0
    } else {
        fusedSeconds = nil
    }

    return CoreAttentionMeasurement(
        storageEstimate: actualAggregateStorageEstimate(keyCode: keyCode, valueCode: valueCode),
        encodeSeconds: encodeSeconds,
        decodeSeconds: decodeSeconds,
        qkSeconds: qkSeconds,
        avSeconds: avSeconds,
        fusedSeconds: fusedSeconds
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
                    preferOnlineFused: false
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
                    preferOnlineFused: true
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
    let request = TurboQuantAttentionRequest(
        queryShape: [
            benchmarkBatchSize, benchmarkQueryHeadCount, options.queryLength, options.headDimension,
        ],
        keyLayout: symbolicAttentionLayout(options: options, role: .key),
        valueLayout: symbolicAttentionLayout(options: options, role: .value),
        queryDType: .float32,
        outputDType: .float32,
        maskKind: .causal,
        preferOnlineFused: options.requestedPath != .twoStageCompressed,
        fallbackState: TurboQuantAttentionFallbackState(
            packedFallbackAvailable: true,
            baselineAvailable: true
        )
    )

    switch options.requestedPath {
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

private func validateCoreBenchmarkOptions(_ options: BenchmarkOptions) throws {
    if options.scaleStorage == .float16 {
        guard options.layoutVersion == TurboQuantAttentionLayout.nextVersion,
              options.enableLayoutV5
        else {
            throw TurboQuantError.invalidMetalConfiguration(
                "float16 attention scale storage requires --layout-version \(TurboQuantAttentionLayout.nextVersion) and --enable-layout-v5"
            )
        }
    }
}

private func forcedFallbackDecision(
    selectedPath: TurboQuantAttentionPath,
    outputDType: DType,
    reason: String
) -> TurboQuantAttentionDecision {
    TurboQuantAttentionDecision(
        selectedPath: selectedPath,
        outputDType: outputDType,
        rejectedPaths: [
            RejectedPath(path: .onlineFused, reason: reason),
            RejectedPath(path: .tiledOnlineFused, reason: reason),
            RejectedPath(path: .twoStageCompressed, reason: reason),
        ]
    )
}

private func symbolicAttentionLayout(
    options: BenchmarkOptions,
    role: TurboQuantTensorRole
) -> TurboQuantAttentionLayout {
    let groupsPerVector = ceilDivide(options.headDimension, by: options.groupSize)
    let logicalValues =
        benchmarkBatchSize * benchmarkKVHeadCount * options.contextTokens * options.headDimension
    let estimate = estimateTurboQuantStorage(
        role: role,
        logicalValues: logicalValues,
        preset: options.preset,
        valueBits: role == .value ? options.resolvedValueBits : nil,
        groupSize: options.groupSize,
        dtype: .float32,
        scaleStorage: options.scaleStorage
    )
    let groupCount = benchmarkBatchSize * benchmarkKVHeadCount * options.contextTokens
        * groupsPerVector
    let magnitudeWordsPerGroup = max(
        1,
        estimate.packedBytes / max(1, groupCount * MemoryLayout<UInt32>.size)
    )

    return TurboQuantAttentionLayout(
        layoutVersion: options.layoutVersion,
        batchSize: benchmarkBatchSize,
        kvHeadCount: benchmarkKVHeadCount,
        capacity: options.contextTokens,
        logicalLength: options.contextTokens,
        headDimension: options.headDimension,
        groupsPerVector: groupsPerVector,
        magnitudeWordsPerGroup: magnitudeWordsPerGroup,
        bitsetWordsPerGroup: max(1, ceilDivide(options.groupSize, by: 32))
    )
}

private func symbolicAggregateStorageEstimate(options: BenchmarkOptions) -> TurboQuantStorageEstimate {
    let logicalValues =
        benchmarkBatchSize * benchmarkKVHeadCount * options.contextTokens * options.headDimension
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

private extension TurboQuantAttentionPath {
    var usesCompressedMetal: Bool {
    switch self {
        case .onlineFused, .tiledOnlineFused, .twoStageCompressed:
            return true
        case .mlxPackedFallback, .baseline, .unavailable:
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
