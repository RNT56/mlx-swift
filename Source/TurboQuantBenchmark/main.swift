import Foundation
import MLX

struct BenchmarkResult: Codable {
    var name: String
    var status: String
    var shape: [Int]
    var latencySeconds: Double?
    var relativeMSE: Float?
    var error: String?
}

struct BenchmarkReport: Codable {
    var generatedAt: String
    var iterations: Int
    var availability: TurboQuantKernelAvailability
    var capabilities: TurboQuantKernelCapabilities
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
    let left = lhs.asArray(Float.self)
    let right = rhs.asArray(Float.self)
    let error = zip(left, right).reduce(Float(0)) { partial, pair in
        let delta = pair.0 - pair.1
        return partial + delta * delta
    }
    let signal = left.reduce(Float(0)) { $0 + $1 * $1 }
    return error / max(signal, Float.leastNonzeroMagnitude)
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

func evalAttentionCode(_ code: TurboQuantAttentionCode) {
    eval(
        code.packedMagnitudes,
        code.signs,
        code.highPrecisionMask,
        code.residualSigns,
        code.scales
    )
}

func timedAttentionCode(
    iterations: Int,
    _ body: () throws -> TurboQuantAttentionCode
) throws -> (Double, TurboQuantAttentionCode) {
    let warmup = try body()
    evalAttentionCode(warmup)
    let start = Date.timeIntervalSinceReferenceDate
    var last = warmup
    for _ in 0 ..< iterations {
        last = try body()
        evalAttentionCode(last)
    }
    let elapsed = Date.timeIntervalSinceReferenceDate - start
    return (elapsed / Double(max(1, iterations)), last)
}

func skipped(_ name: String, reason: String) -> BenchmarkResult {
    BenchmarkResult(name: name, status: "skipped", shape: [], error: reason)
}

let iterations = argumentValue("--iterations", default: 10)
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
                shape: decoded.shape,
                latencySeconds: latency,
                relativeMSE: relativeMSE(input, decoded)
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
                status: availability.kernelCapabilities.linearMatmul ? "production" : "experimental",
                shape: output.shape,
                latencySeconds: latency
            ))
    } catch {
        results.append(
            BenchmarkResult(name: "flat.matmul.product_estimator", status: "failed", shape: [], error: "\(error)"))
    }
} else {
    results.append(skipped("flat.decode", reason: "Metal codec unavailable"))
    results.append(skipped("flat.matmul.product_estimator", reason: "Metal codec unavailable"))
}

if availability.supportsMetalPolarQJLAttention {
    for headDimension in TurboQuantRuntimeProbeResult.throughputOptimizedOnlineFusedHeadDimensions {
        do {
            let namePrefix = "attention.d\(headDimension)"
            let q = MLXArray(
                values(count: 1 * 4 * 2 * headDimension, scale: 0.019),
                [1, 4, 2, headDimension]
            )
            let k = MLXArray(
                values(count: 1 * 2 * 256 * headDimension, scale: 0.007, phase: 0.1),
                [1, 2, 256, headDimension]
            )
            let v = MLXArray(
                values(count: 1 * 2 * 256 * headDimension, scale: 0.009, phase: 0.2),
                [1, 2, 256, headDimension]
            )
            let keyConfiguration = TurboQuantConfiguration(
                preset: .turbo4v2,
                role: .key,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 0xBEEF_0000_0000_0003
            )
            let valueConfiguration = TurboQuantConfiguration(
                preset: .turbo4v2,
                role: .value,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 0xBEEF_0000_0000_0004,
                valueBits: 4
            )
            let (keyEncodeLatency, keyCode) = try timedAttentionCode(iterations: iterations) {
                try turboQuantMetalEncodeAttention(k, configuration: keyConfiguration)
            }
            let (valueEncodeLatency, valueCode) = try timedAttentionCode(iterations: iterations) {
                try turboQuantMetalEncodeAttention(v, configuration: valueConfiguration)
            }
            results.append(
                BenchmarkResult(
                    name: "\(namePrefix).encode.key",
                    status: "ok",
                    shape: k.shape,
                    latencySeconds: keyEncodeLatency
                ))
            results.append(
                BenchmarkResult(
                    name: "\(namePrefix).encode.value",
                    status: "ok",
                    shape: v.shape,
                    latencySeconds: valueEncodeLatency
                ))

            let (_, decodedValues) = try timed(iterations: 1) {
                try turboQuantMetalDecodeAttention(valueCode, outputDType: .float32)
            }
            let (decodeLatency, _) = try timed(iterations: iterations) {
                try turboQuantMetalDecodeAttention(valueCode, outputDType: .float32)
            }
            results.append(
                BenchmarkResult(
                    name: "\(namePrefix).decode.value",
                    status: "ok",
                    shape: decodedValues.shape,
                    latencySeconds: decodeLatency,
                    relativeMSE: relativeMSE(v, decodedValues)
                ))

            let scale = 1 / sqrt(Float(q.dim(-1)))
            let (qkLatency, scores) = try timed(iterations: iterations) {
                try turboQuantMetalQK(queries: q, keyCode: keyCode, scale: scale, mask: .causal)
            }
            let weights = softmax(scores.asType(.float32), axis: -1)
            let (avLatency, avOutput) = try timed(iterations: iterations) {
                try turboQuantMetalAV(
                    attentionWeights: weights,
                    valueCode: valueCode,
                    outputDType: .float32
                )
            }
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
                    name: "\(namePrefix).qk",
                    status: "ok",
                    shape: scores.shape,
                    latencySeconds: qkLatency
                ))
            results.append(
                BenchmarkResult(
                    name: "\(namePrefix).av",
                    status: "ok",
                    shape: avOutput.shape,
                    latencySeconds: avLatency
                ))
            results.append(
                BenchmarkResult(
                    name: "\(namePrefix).two_stage",
                    status: "ok",
                    shape: twoStage.shape,
                    latencySeconds: twoStageLatency
                ))
            results.append(
                BenchmarkResult(
                    name: "\(namePrefix).fused",
                    status: "ok",
                    shape: fused.shape,
                    latencySeconds: fusedLatency,
                    relativeMSE: relativeMSE(twoStage, fused)
                ))
        } catch {
            results.append(
                BenchmarkResult(
                    name: "attention.d\(headDimension)",
                    status: "failed",
                    shape: [headDimension],
                    error: "\(error)"
                ))
        }
    }
} else {
    results.append(skipped("attention", reason: "Metal attention unavailable or probe failed"))
}

let report = BenchmarkReport(
    generatedAt: ISO8601DateFormatter().string(from: Date()),
    iterations: iterations,
    availability: availability,
    capabilities: availability.kernelCapabilities,
    device: TurboQuantDeviceCapabilities.current,
    results: results
)
let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
let data = try encoder.encode(report)
FileHandle.standardOutput.write(data)
FileHandle.standardOutput.write(Data("\n".utf8))
