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
    do {
        let q = MLXArray(values(count: 1 * 4 * 2 * 128, scale: 0.019), [1, 4, 2, 128])
        let k = MLXArray(values(count: 1 * 2 * 256 * 128, scale: 0.007, phase: 0.1), [1, 2, 256, 128])
        let v = MLXArray(values(count: 1 * 2 * 256 * 128, scale: 0.009, phase: 0.2), [1, 2, 256, 128])
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
                shape: twoStage.shape,
                latencySeconds: twoStageLatency
            ))
        results.append(
            BenchmarkResult(
                name: "attention.fused",
                status: "ok",
                shape: fused.shape,
                latencySeconds: fusedLatency,
                relativeMSE: relativeMSE(twoStage, fused)
            ))
    } catch {
        results.append(
            BenchmarkResult(name: "attention", status: "failed", shape: [], error: "\(error)"))
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
