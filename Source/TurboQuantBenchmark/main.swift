import Foundation
import MLX

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

func argumentValue(_ name: String, default defaultValue: Int) -> Int {
    let arguments = CommandLine.arguments
    guard let index = arguments.firstIndex(of: name), arguments.indices.contains(index + 1),
        let value = Int(arguments[index + 1])
    else {
        return defaultValue
    }
    return value
}

func values(count: Int, scale: Double, phase: Double = 0) -> [Float] {
    (0 ..< count).map { index in
        let position = Double(index)
        return Float(0.31 * sin(position * scale + phase) + 0.17 * cos(position * 0.037))
    }
}

func relativeMSE(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
    qualityMetrics(lhs, rhs).relativeMSE
}

func qualityMetrics(_ lhs: MLXArray, _ rhs: MLXArray) -> QualityMetrics {
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

func timed(iterations: Int, _ body: () throws -> MLXArray) throws -> (Double, MLXArray) {
    let warmup = try body()
    eval(warmup)
    let start = Date.timeIntervalSinceReferenceDate
    var last = warmup
    for _ in 0 ..< iterations {
        last = try body()
        eval(last)
    }
    let elapsed = Date.timeIntervalSinceReferenceDate - start
    return (elapsed / Double(max(1, iterations)), last)
}

func skipped(_ name: String, reason: String) -> BenchmarkResult {
    BenchmarkResult(name: name, status: "skipped", shape: [], error: reason)
}

let iterations = argumentValue("--iterations", default: 10)
let includeTimestamp = CommandLine.arguments.contains("--include-timestamp")
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
        let (latency, decoded) = try timed(iterations: iterations) {
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
            BenchmarkResult(name: "flat.decode", status: "failed", shape: [], error: "\(error)"))
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
        let (latency, output) = try timed(iterations: iterations) {
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
        let (twoStageLatency, twoStage) = try timed(iterations: iterations) {
            try turboQuantMetalScaledDotProductAttention(
                queries: q,
                keyCode: keyCode,
                valueCode: valueCode,
                scale: scale,
                mask: .causal,
                preferOnlineFused: false
            )
        }
        let (fusedLatency, fused) = try timed(iterations: iterations) {
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
            BenchmarkResult(name: "attention", status: "failed", shape: [], error: "\(error)"))
    }
} else {
    results.append(skipped("attention", reason: "Metal attention unavailable or probe failed"))
}

let report = BenchmarkReport(
    schemaVersion: 2,
    generatedAt: includeTimestamp ? ISO8601DateFormatter().string(from: Date()) : nil,
    iterations: iterations,
    availability: availability,
    capabilities: availability.kernelCapabilities,
    attentionCapabilities: availability.attentionCapabilities,
    device: TurboQuantDeviceCapabilities.current,
    results: results
)
let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
let data = try encoder.encode(report)
FileHandle.standardOutput.write(data)
FileHandle.standardOutput.write(Data("\n".utf8))
