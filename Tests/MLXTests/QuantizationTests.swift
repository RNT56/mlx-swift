// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

class QuantizationTests: XCTestCase {
    private func requireMLXRuntime() throws {
        guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLCodec else {
            throw XCTSkip("MLX runtime metallib unavailable in this package context")
        }
    }

    private func relativeMSE(_ lhs: [Float], _ rhs: [Float]) -> Float {
        let squaredError = zip(lhs, rhs).reduce(Float(0)) { partial, pair in
            let delta = pair.0 - pair.1
            return partial + delta * delta
        }
        let signal = lhs.reduce(Float(0)) { $0 + $1 * $1 }
        return squaredError / max(signal, Float.leastNonzeroMagnitude)
    }

    private func pearsonCorrelation(_ lhs: [Float], _ rhs: [Float]) -> Float {
        let count = Float(lhs.count)
        let lhsMean = lhs.reduce(Float(0), +) / count
        let rhsMean = rhs.reduce(Float(0), +) / count
        var numerator = Float(0)
        var lhsVariance = Float(0)
        var rhsVariance = Float(0)
        for (left, right) in zip(lhs, rhs) {
            let lhsCentered = left - lhsMean
            let rhsCentered = right - rhsMean
            numerator += lhsCentered * rhsCentered
            lhsVariance += lhsCentered * lhsCentered
            rhsVariance += rhsCentered * rhsCentered
        }
        return numerator / max(sqrt(lhsVariance * rhsVariance), Float.leastNonzeroMagnitude)
    }

    func testQuantizedLinearShapeDesc() {
        let linear1 = Linear(512, 1024)
        let quantized1 = linear1.toQuantized(groupSize: 64, bits: 4)
        XCTAssertEqual(
            quantized1.describeExtra(0), "(inputDimensions=512, outputDimensions=1024, bias=true)")
        let linear2 = Linear(1024, 512, bias: false)
        let quantized2 = linear2.toQuantized(groupSize: 128, bits: 8)
        XCTAssertEqual(
            quantized2.describeExtra(0), "(inputDimensions=1024, outputDimensions=512, bias=false)")
        let linear3 = Linear(512, 1024)
        let quantized3 = linear3.toQuantized(groupSize: 32, bits: 4, mode: .mxfp4)
        XCTAssertEqual(
            quantized3.describeExtra(0), "(inputDimensions=512, outputDimensions=1024, bias=true)")
    }

    func testQuantizedEmbeddingShapeDesc() {
        let embedding1 = Embedding(embeddingCount: 512, dimensions: 1024)
        let quantized1 = embedding1.toQuantized(groupSize: 64, bits: 4)
        XCTAssertEqual(quantized1.describeExtra(0), "(embeddingCount=512, dimensions=1024)")
        let embedding2 = Embedding(embeddingCount: 1024, dimensions: 512)
        let quantized2 = embedding2.toQuantized(groupSize: 128, bits: 8)
        XCTAssertEqual(
            quantized2.describeExtra(0), "(embeddingCount=1024, dimensions=512)")
        let embedding3 = Embedding(embeddingCount: 512, dimensions: 1024)
        let quantized3 = embedding3.toQuantized(groupSize: 32, bits: 4, mode: .mxfp4)
        XCTAssertEqual(
            quantized3.describeExtra(0), "(embeddingCount=512, dimensions=1024)")
    }

    func testQuantizedLinearMxfp4DoesNotCreateAffineBiases() {
        let quantized = QuantizedLinear(64, 64, groupSize: 32, bits: 4, mode: .mxfp4)
        XCTAssertNil(quantized.biases)
    }

    func testTurboQuantPresetsExposeBalancedFourBitVariant() {
        XCTAssertEqual(TurboQuantPreset.turbo4.defaultValueBits, 4)
        XCTAssertEqual(TurboQuantPreset.turbo4.effectiveBits, 4)
        XCTAssertEqual(TurboQuantPreset.turbo4v2.defaultValueBits, 4)
        XCTAssertEqual(TurboQuantPreset.turbo4v2.targetMagnitudeBits, 4)
    }

    func testTurboQuantLinearMatchesDecodedFallbackShape() {
        let values = (0 ..< 192).map { index in
            let position = Double(index)
            let signal = 0.2 * sin(position * 0.11) + 0.1 * cos(position * 0.07)
            return Float(signal)
        }
        let weight = MLXArray(values, [3, 64])
        let bias = MLXArray([Float](repeating: 0.05, count: 3))
        let layer = TurboQuantLinear(
            weight: weight,
            bias: bias,
            preset: .turbo4v2,
            groupSize: 64,
            backend: .mlxPacked,
            seed: 0x1234
        )
        let x = MLXArray.ones([2, 4, 64], dtype: .float32)
        let output = layer(x)
        XCTAssertEqual(output.shape, [2, 4, 3])
        XCTAssertEqual(layer.shape.0, 3)
        XCTAssertEqual(layer.shape.1, 64)
        XCTAssertEqual(layer.requestedBackend, TurboQuantBackend.mlxPacked)

        let restored = TurboQuantLinear(
            packedWeight: layer.weight,
            bias: layer.bias,
            scales: layer.scales,
            biases: layer.biases,
            preset: layer.preset,
            groupSize: layer.groupSize,
            seed: layer.seed,
            valueBits: layer.valueBits
        )
        XCTAssertEqual(restored(x).shape, [2, 4, 3])
        XCTAssertEqual(restored.activeBackend, TurboQuantBackend.mlxPacked)
    }

    func testTurboQuantConvertedArraysCreatePackedLinearCheckpoint() throws {
        let weightValues = (0 ..< 128).map { Float($0) / 128 }
        let arrays = [
            "model.layers.0.self_attn.q_proj.weight": MLXArray(weightValues, [2, 64]),
            "model.embed_tokens.weight": MLXArray.ones([4, 64], dtype: .float32),
            "model.layers.0.self_attn.q_proj.bias": MLXArray.zeros([2], dtype: .float32),
        ]
        let options = TurboQuantCheckpointConversionOptions(
            preset: .turbo4v2,
            groupSize: 64,
            seed: 0xCAFE
        )

        let converted = try turboQuantConvertedArrays(
            arrays,
            metadata: ["format": "mlx"],
            options: options
        )

        XCTAssertEqual(
            converted.report.converted.map(\.name), ["model.layers.0.self_attn.q_proj.weight"])
        XCTAssertNotNil(converted.arrays["model.layers.0.self_attn.q_proj.scales"])
        XCTAssertNotNil(converted.arrays["model.layers.0.self_attn.q_proj.biases"])
        XCTAssertEqual(converted.arrays["model.embed_tokens.weight"]?.shape, [4, 64])
        XCTAssertEqual(converted.metadata["quant_method"], "turboquant")
        XCTAssertEqual(converted.metadata["linear_class"], "TurboQuantLinear")
        XCTAssertEqual(converted.metadata["turboquant_preset"], "turbo4v2")
        XCTAssertEqual(converted.metadata["turboquant_schema_version"], "2")
        XCTAssertEqual(
            converted.metadata["turboquant_attention_layout_version"],
            "\(TurboQuantAttentionLayout.currentVersion)"
        )
        XCTAssertEqual(converted.metadata["turboquant_linear_format"], "mlx_packed")

        let layer = TurboQuantLinear(
            packedWeight: try XCTUnwrap(converted.arrays["model.layers.0.self_attn.q_proj.weight"]),
            bias: arrays["model.layers.0.self_attn.q_proj.bias"],
            scales: try XCTUnwrap(converted.arrays["model.layers.0.self_attn.q_proj.scales"]),
            biases: converted.arrays["model.layers.0.self_attn.q_proj.biases"],
            preset: .turbo4v2,
            groupSize: 64,
            seed: 0xCAFE
        )
        XCTAssertEqual(layer(MLXArray.ones([3, 64], dtype: .float32)).shape, [3, 2])
    }

    func testTurboQuantConvertSafetensorsWritesMetadata() throws {
        let temporaryPath = FileManager.default.temporaryDirectory.appending(
            path: UUID().uuidString, directoryHint: .isDirectory)
        try FileManager.default.createDirectory(
            at: temporaryPath, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temporaryPath) }

        let input = temporaryPath.appending(path: "input.safetensors")
        let output = temporaryPath.appending(path: "output.safetensors")
        try save(
            arrays: [
                "linear.weight": MLXArray.ones([2, 64], dtype: .float32)
            ],
            metadata: ["format": "mlx"],
            url: input
        )

        let report = try turboQuantConvertSafetensors(
            from: input,
            to: output,
            options: TurboQuantCheckpointConversionOptions(groupSize: 64)
        )
        let (loadedArrays, loadedMetadata) = try loadArraysAndMetadata(url: output)

        XCTAssertEqual(report.convertedCount, 1)
        XCTAssertNotNil(loadedArrays["linear.scales"])
        XCTAssertEqual(loadedMetadata["quant_method"], "turboquant")
        XCTAssertEqual(loadedMetadata["turboquant_format"], "mlx_packed")
        XCTAssertEqual(loadedMetadata["turboquant_schema_version"], "2")
        XCTAssertEqual(loadedMetadata["turboquant_seed_policy"], "fixed")
    }

    func testTurboQuantConfigurationDecodesLegacyPayloadWithV4Defaults() throws {
        let legacyJSON = Data("""
        {
          "preset": "turbo4v2",
          "role": "key",
          "groupSize": 64,
          "mode": "affine",
          "backend": "mlxPacked",
          "seed": 7,
          "qjlResidualScale": 0.5,
          "valueBits": 4
        }
        """.utf8)

        let configuration = try JSONDecoder().decode(TurboQuantConfiguration.self, from: legacyJSON)

        XCTAssertEqual(configuration.preset, .turbo4v2)
        XCTAssertEqual(configuration.role, .key)
        XCTAssertEqual(configuration.attentionLayoutVersion, TurboQuantAttentionLayout.currentVersion)
        XCTAssertFalse(configuration.allowExperimentalLayoutV5)
        XCTAssertEqual(configuration.attentionScaleStorage, .float32)
        XCTAssertTrue(configuration.deterministicHighPrecisionMask)
    }

    func testTurboQuantPackedRoundTrip() throws {
        try requireMLXRuntime()

        let x = MLXArray.ones([1, 32], dtype: .float32, stream: .device(.cpu))
        let configuration = TurboQuantConfiguration(preset: .turbo3_5, groupSize: 32)
        let packed = turboQuantized(x, configuration: configuration, stream: .device(.cpu))
        let decoded = turboDequantized(packed, configuration: configuration, stream: .device(.cpu))

        XCTAssertEqual(decoded.shape, x.shape)
        XCTAssertTrue(allClose(decoded, x).item(Bool.self))
    }

    func testTurboQuantMatmulShape() throws {
        try requireMLXRuntime()

        let x = MLXArray.ones([2, 32], dtype: .float32, stream: .device(.cpu))
        let w = MLXArray.ones([4, 32], dtype: .float32, stream: .device(.cpu))
        let configuration = TurboQuantConfiguration(preset: .turbo2_5, groupSize: 32)
        let packed = turboQuantized(w, configuration: configuration, stream: .device(.cpu))
        let output = turboQuantizedMM(
            x, packed, configuration: configuration, stream: .device(.cpu))

        XCTAssertEqual(output.shape, [2, 4])
    }

    func testTurboQuantReferenceCodecIsDeterministic() throws {
        try requireMLXRuntime()

        let values = (0 ..< 128).map { index in
            Float(sin(Double(index) * 0.17) + cos(Double(index) * 0.03))
        }
        let x = MLXArray(values, [2, 64])
        let configuration = TurboQuantConfiguration(
            preset: .turbo3_5,
            role: .key,
            groupSize: 32,
            backend: .polarQJLReference,
            seed: 42
        )

        let first = try turboQuantReferenceEncode(x, configuration: configuration)
        let second = try turboQuantReferenceEncode(x, configuration: configuration)

        XCTAssertEqual(first, second)
        XCTAssertEqual(first.shape, [2, 64])
        XCTAssertEqual(first.format, TurboQuantReferenceFormat.turboQuantProd)
        XCTAssertGreaterThan(first.storageByteCount, 0)
        XCTAssertFalse(first.highScales.isEmpty)
    }

    func testTurboQuantReferenceCodecUsesFullWidthSeed() throws {
        try requireMLXRuntime()

        let values = (0 ..< 128).map { index in
            Float(sin(Double(index) * 0.11) + cos(Double(index) * 0.19))
        }
        let x = MLXArray(values, [2, 64])
        let lowSeedConfiguration = TurboQuantConfiguration(
            preset: .turbo3_5,
            role: .key,
            groupSize: 64,
            backend: .polarQJLReference,
            seed: 0x0000_0000_0123_4567
        )
        let highSeedConfiguration = TurboQuantConfiguration(
            preset: .turbo3_5,
            role: .key,
            groupSize: 64,
            backend: .polarQJLReference,
            seed: 0xDEAD_BEEF_0123_4567
        )

        let lowSeed = try turboQuantReferenceEncode(x, configuration: lowSeedConfiguration)
        let highSeed = try turboQuantReferenceEncode(x, configuration: highSeedConfiguration)

        XCTAssertNotEqual(lowSeed.signs, highSeed.signs)
    }

    func testTurboQuantReferenceCodecDistortionThreshold() throws {
        try requireMLXRuntime()

        let values = (0 ..< 256).map { index in
            let position = Double(index)
            let sineTerm = sin(position * 0.11) * 0.7
            let cosineTerm = cos(position * 0.07) * 0.3
            return Float(sineTerm + cosineTerm)
        }
        let x = MLXArray(values, [4, 64])
        let configuration = TurboQuantConfiguration(
            preset: .turbo3_5,
            role: .vector,
            groupSize: 64,
            backend: .polarQJLReference,
            seed: 17
        )

        let code = try turboQuantReferenceEncode(x, configuration: configuration)
        let decoded = try turboQuantReferenceDecode(code).asArray(Float.self)
        let mse =
            zip(values, decoded)
            .map { lhs, rhs in
                let delta = lhs - rhs
                return delta * delta
            }
            .reduce(Float(0), +) / Float(values.count)

        XCTAssertLessThan(mse, 0.01)
    }

    func testTurboQuantReferenceQualityGatePassesFixture() throws {
        try requireMLXRuntime()

        let values = (0 ..< 256).map { index in
            let position = Double(index)
            let sineTerm = sin(position * 0.09) * 0.5
            let cosineTerm = cos(position * 0.13) * 0.25
            return Float(sineTerm + cosineTerm)
        }
        let x = MLXArray(values, [4, 64])
        let configuration = TurboQuantConfiguration(
            preset: .turbo3_5,
            role: .key,
            groupSize: 64,
            backend: .polarQJLReference,
            seed: 99
        )

        let report = try turboQuantReferenceQuality(x, configuration: configuration)

        XCTAssertLessThan(report.relativeMSE, 0.085)
        XCTAssertGreaterThan(report.cosineSimilarity, 0.955)
    }

    func testTurboQuantReferenceValueBitsStorageAccounting() throws {
        try requireMLXRuntime()

        let values = (0 ..< 256).map { index in
            let position = Double(index)
            let sineTerm = 0.4 * sin(position * 0.07)
            let cosineTerm = 0.15 * cos(position * 0.17)
            return Float(sineTerm + cosineTerm)
        }
        let x = MLXArray(values, [4, 64])
        let twoBit = try turboQuantReferenceEncode(
            x,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .value,
                groupSize: 64,
                backend: .polarQJLReference,
                valueBits: 2
            )
        )
        let fourBit = try turboQuantReferenceEncode(
            x,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .value,
                groupSize: 64,
                backend: .polarQJLReference,
                valueBits: 4
            )
        )

        XCTAssertEqual(twoBit.format, TurboQuantReferenceFormat.affineValue)
        XCTAssertEqual(fourBit.format, TurboQuantReferenceFormat.affineValue)
        XCTAssertLessThan(twoBit.approximateBitsPerValue, 3.1)
        XCTAssertLessThan(fourBit.approximateBitsPerValue, 5.1)
        XCTAssertLessThan(twoBit.storageByteCount, fourBit.storageByteCount)
    }

    func testTurboQuantProductInnerProductBiasAndRetrieval() throws {
        try requireMLXRuntime()

        let queryValues = (0 ..< 64).map { index in
            let position = Double(index)
            let sineTerm = 0.35 * sin(position * 0.13)
            let cosineTerm = 0.2 * cos(position * 0.05)
            return Float(sineTerm + cosineTerm)
        }
        let needleValues = queryValues.map { $0 * 1.35 }
        let query = MLXArray(queryValues, [64])
        let keys = (0 ..< 16).map { keyIndex in
            (0 ..< 64).map { dim in
                if keyIndex == 7 { return needleValues[dim] }
                let position = Double(keyIndex * 64 + dim)
                return Float(0.25 * sin(position * 0.071) - 0.18 * cos(position * 0.113))
            }
        }

        var exactScores: [Float] = []
        var estimatedScores: [Float] = []
        for (keyIndex, keyValues) in keys.enumerated() {
            let exactScore = zip(queryValues, keyValues).reduce(Float(0)) { partial, pair in
                partial + pair.0 * pair.1
            }
            exactScores.append(exactScore)
            let code = try turboQuantReferenceEncode(
                MLXArray(keyValues, [64]),
                configuration: TurboQuantConfiguration(
                    preset: .turbo3_5,
                    role: .key,
                    groupSize: 64,
                    backend: .polarQJLReference,
                    seed: UInt64(0x600D_0000 + keyIndex)
                )
            )
            estimatedScores.append(try turboQuantReferenceInnerProduct(query: query, code: code))
        }

        XCTAssertEqual(estimatedScores.enumerated().max(by: { $0.element < $1.element })?.offset, 7)
        XCTAssertGreaterThan(pearsonCorrelation(exactScores, estimatedScores), 0.7)

        let target = MLXArray(keys[3], [64])
        let exact = exactScores[3]
        let estimates = try (0 ..< 32).map { seedOffset in
            let code = try turboQuantReferenceEncode(
                target,
                configuration: TurboQuantConfiguration(
                    preset: .turbo3_5,
                    role: .key,
                    groupSize: 64,
                    backend: .polarQJLReference,
                    seed: UInt64(0xB1A5_0000 + seedOffset)
                )
            )
            return try turboQuantReferenceInnerProduct(query: query, code: code)
        }
        let average = estimates.reduce(Float(0), +) / Float(estimates.count)
        XCTAssertLessThan(abs(average - exact) / max(abs(exact), Float.leastNonzeroMagnitude), 0.25)
    }

    func testTurboQuantBackendAvailabilityContract() throws {
        XCTAssertNoThrow(try requireTurboQuantBackend(.mlxPacked))
        XCTAssertNoThrow(try requireTurboQuantBackend(.polarQJLReference))

        let availability = TurboQuantKernelAvailability.current
        if availability.supportsMetalPolarQJL {
            XCTAssertNoThrow(try requireTurboQuantBackend(.metalPolarQJL))
            XCTAssertEqual(availability.runtimeBackend(for: .metalPolarQJL), .metalPolarQJL)
            XCTAssertNil(availability.fallbackReason(for: .metalPolarQJL))
        } else {
            XCTAssertThrowsError(try requireTurboQuantBackend(.metalPolarQJL))
            XCTAssertEqual(availability.runtimeBackend(for: .metalPolarQJL), .mlxPacked)
            XCTAssertNotNil(availability.fallbackReason(for: .metalPolarQJL))
        }
    }

    func testTurboQuantDeviceCapabilitiesAndProbeContract() throws {
        let capabilities = TurboQuantDeviceCapabilities.current
        let availability = TurboQuantKernelAvailability.current

        XCTAssertFalse(capabilities.architectureName.isEmpty)
        XCTAssertEqual(capabilities.runtimeProbe, TurboQuantRuntimeProbe.current)
        XCTAssertEqual(availability.selfTestStatus, capabilities.runtimeProbe.status)
        XCTAssertEqual(
            availability.selectedKernelProfile, capabilities.runtimeProbe.selectedKernelProfile)
        XCTAssertEqual(
            availability.supportsMetalPolarQJLCodec,
            capabilities.runtimeProbe.metalRuntimeAvailable
                && capabilities.runtimeProbe.flatCodecPassed
        )

        if availability.supportsMetalPolarQJLAttention {
            XCTAssertEqual(capabilities.runtimeProbe.status, .passed)
            XCTAssertTrue(capabilities.runtimeProbe.flatCodecPassed)
            XCTAssertNotEqual(capabilities.runtimeProbe.selectedKernelProfile, .mlxPackedFallback)
            XCTAssertNil(capabilities.runtimeProbe.failureReason)
        } else {
            XCTAssertNotEqual(capabilities.runtimeProbe.status, .notRun)
            XCTAssertEqual(availability.runtimeBackend(for: .metalPolarQJL), .mlxPacked)
        }
    }

    func testTurboQuantKernelCapabilitiesAreIndependentlyGated() {
        let availability = TurboQuantKernelAvailability.current
        let capabilities = availability.kernelCapabilities
        let probeCapabilities = TurboQuantRuntimeProbe.current.kernelCapabilities

        XCTAssertEqual(capabilities.flatEncodeDecode, availability.supportsMetalPolarQJLCodec)
        XCTAssertEqual(capabilities.attentionEncode, probeCapabilities.attentionEncode)
        XCTAssertEqual(capabilities.attentionDecode, probeCapabilities.attentionDecode)
        XCTAssertEqual(capabilities.attentionQK, probeCapabilities.attentionQK)
        XCTAssertEqual(capabilities.attentionAV, probeCapabilities.attentionAV)
        XCTAssertEqual(capabilities.attentionFusedDecode, probeCapabilities.attentionFusedDecode)
        XCTAssertEqual(capabilities.bfloatOutput, probeCapabilities.bfloatOutput)
        XCTAssertEqual(
            capabilities.linearMatmul,
            ProcessInfo.processInfo.environment["TURBOQUANT_ENABLE_EXPERIMENTAL_LINEAR_METAL"]
                == "1"
                && availability.supportsMetalPolarQJLCodec
        )
    }

    func testTurboQuantAttentionDecisionFallsBackFromFusedToTwoStage() throws {
        let keyLayout = try turboQuantAttentionLayout(
            shape: [1, 2, 16, 64],
            preset: .turbo3_5,
            role: .key,
            groupSize: 64
        )
        let valueLayout = try turboQuantAttentionLayout(
            shape: [1, 2, 16, 64],
            preset: .turbo3_5,
            role: .value,
            groupSize: 64
        )
        let request = TurboQuantAttentionRequest(
            queryShape: [1, 4, 1, 64],
            keyLayout: keyLayout,
            valueLayout: valueLayout,
            queryDType: .float16,
            outputDType: .float16,
            maskKind: .causal
        )
        let decision = try turboQuantAttentionDecision(
            request: request,
            capabilities: TurboQuantAttentionCapabilities(
                encode: true,
                decode: true,
                qk: true,
                av: true,
                onlineFused: false
            )
        )

        XCTAssertEqual(decision.selectedPath, .twoStageCompressed)
        XCTAssertTrue(decision.rejectedPaths.contains { $0.path == .onlineFused })
    }

    func testTurboQuantBFloatGateDoesNotDisableFloatOutputs() throws {
        let keyLayout = try turboQuantAttentionLayout(
            shape: [1, 2, 16, 64],
            preset: .turbo3_5,
            role: .key,
            groupSize: 64
        )
        let valueLayout = try turboQuantAttentionLayout(
            shape: [1, 2, 16, 64],
            preset: .turbo3_5,
            role: .value,
            groupSize: 64
        )
        let capabilities = TurboQuantAttentionCapabilities(
            encode: true,
            decode: true,
            qk: true,
            av: true,
            onlineFused: true,
            bfloatOutput: false
        )
        let fp32Request = TurboQuantAttentionRequest(
            queryShape: [1, 4, 1, 64],
            keyLayout: keyLayout,
            valueLayout: valueLayout,
            queryDType: .float32,
            outputDType: .float32,
            maskKind: .causal
        )
        let bf16Request = TurboQuantAttentionRequest(
            queryShape: [1, 4, 1, 64],
            keyLayout: keyLayout,
            valueLayout: valueLayout,
            queryDType: .bfloat16,
            outputDType: .bfloat16,
            maskKind: .causal
        )

        XCTAssertEqual(
            try turboQuantAttentionDecision(
                request: fp32Request,
                capabilities: capabilities
            ).selectedPath,
            .onlineFused
        )
        XCTAssertThrowsError(
            try turboQuantAttentionDecision(
                request: bf16Request,
                capabilities: capabilities
            )
        ) { error in
            XCTAssertTrue(String(describing: error).contains("bfloat16"))
        }
    }

    func testTurboQuantMaterializedMaskRoutesToTwoStage() throws {
        let keyLayout = try turboQuantAttentionLayout(
            shape: [1, 2, 16, 64],
            preset: .turbo3_5,
            role: .key,
            groupSize: 64
        )
        let valueLayout = try turboQuantAttentionLayout(
            shape: [1, 2, 16, 64],
            preset: .turbo3_5,
            role: .value,
            groupSize: 64
        )
        let decision = try turboQuantAttentionDecision(
            request: TurboQuantAttentionRequest(
                queryShape: [1, 4, 1, 64],
                keyLayout: keyLayout,
                valueLayout: valueLayout,
                queryDType: .float32,
                outputDType: .float32,
                maskKind: .materializedArray
            ),
            capabilities: TurboQuantAttentionCapabilities(
                encode: true,
                decode: true,
                qk: true,
                av: true,
                onlineFused: true
            )
        )

        XCTAssertEqual(decision.selectedPath, .twoStageCompressed)
        XCTAssertTrue(decision.rejectedPaths.contains { $0.path == .onlineFused })
    }

    func testTurboQuantAttentionDecisionUsesFallbackOrFailsTyped() throws {
        let keyLayout = try turboQuantAttentionLayout(
            shape: [1, 2, 16, 64],
            preset: .turbo3_5,
            role: .key,
            groupSize: 64
        )
        let valueLayout = try turboQuantAttentionLayout(
            shape: [1, 2, 16, 64],
            preset: .turbo3_5,
            role: .value,
            groupSize: 64
        )
        let request = TurboQuantAttentionRequest(
            queryShape: [1, 4, 2, 64],
            keyLayout: keyLayout,
            valueLayout: valueLayout,
            queryDType: .float32,
            outputDType: .float32,
            memoryBudgetBytes: 1
        )

        XCTAssertThrowsError(
            try turboQuantAttentionDecision(
                request: request,
                capabilities: TurboQuantAttentionCapabilities(
                    encode: true,
                    decode: true,
                    qk: true,
                    av: true,
                    onlineFused: false
                )
            )
        ) { error in
            XCTAssertTrue(String(describing: error).contains("No semantically correct"))
        }

        var fallbackRequest = request
        fallbackRequest.fallbackState = TurboQuantAttentionFallbackState(
            decodedFallbackAvailable: true
        )
        XCTAssertEqual(
            try turboQuantAttentionDecision(
                request: fallbackRequest,
                capabilities: TurboQuantAttentionCapabilities(
                    encode: true,
                    decode: true,
                    qk: true,
                    av: true,
                    onlineFused: false
                )
            ).selectedPath,
            .baseline
        )
    }

    func testTurboQuantAttentionDecisionSelectsTiledFusedForShortDecodeWindows() throws {
        let keyLayout = try turboQuantAttentionLayout(
            shape: [1, 2, 16, 64],
            preset: .turbo3_5,
            role: .key,
            groupSize: 64
        )
        let valueLayout = try turboQuantAttentionLayout(
            shape: [1, 2, 16, 64],
            preset: .turbo3_5,
            role: .value,
            groupSize: 64
        )

        let decision = try turboQuantAttentionDecision(
            request: TurboQuantAttentionRequest(
                queryShape: [1, 4, 4, 64],
                keyLayout: keyLayout,
                valueLayout: valueLayout,
                queryDType: .float16,
                outputDType: .float16,
                maskKind: .causal
            ),
            capabilities: TurboQuantAttentionCapabilities(
                encode: true,
                decode: true,
                qk: true,
                av: true,
                onlineFused: true,
                tiledOnlineFused: true,
                maxOnlineFusedQueryLength: 1,
                maxTiledOnlineFusedQueryLength: 8
            )
        )

        XCTAssertEqual(decision.selectedPath, .tiledOnlineFused)
        XCTAssertTrue(decision.rejectedPaths.contains { $0.path == .onlineFused })
    }

    func testTurboQuantAttentionDecisionHonorsDTypeMaskAndDeviceCapabilitySets() throws {
        let keyLayout = try turboQuantAttentionLayout(
            shape: [1, 2, 16, 64],
            preset: .turbo3_5,
            role: .key,
            groupSize: 64
        )
        let valueLayout = try turboQuantAttentionLayout(
            shape: [1, 2, 16, 64],
            preset: .turbo3_5,
            role: .value,
            groupSize: 64
        )

        XCTAssertThrowsError(
            try turboQuantAttentionDecision(
                request: TurboQuantAttentionRequest(
                    queryShape: [1, 4, 1, 64],
                    keyLayout: keyLayout,
                    valueLayout: valueLayout,
                    queryDType: .float16,
                    outputDType: .float16,
                    maskKind: .causal,
                    deviceFamily: "apple7"
                ),
                capabilities: TurboQuantAttentionCapabilities(
                    encode: true,
                    decode: true,
                    qk: false,
                    av: false,
                    onlineFused: true,
                    supportedDTypes: [.float32],
                    supportedMasks: [.none],
                    supportedDeviceFamilies: ["apple9"]
                )
            )
        ) { error in
            let description = String(describing: error)
            XCTAssertTrue(description.contains("dtype"))
            XCTAssertTrue(description.contains("mask"))
            XCTAssertTrue(description.contains("device family"))
        }
    }

    func testTurboQuantLinearKeepsMetalMatmulBehindProductionGate() {
        let weight = MLXArray.ones([2, 64], dtype: .float32)
        let layer = TurboQuantLinear(
            weight: weight,
            bias: nil,
            preset: .turbo4v2,
            groupSize: 64,
            backend: .metalPolarQJL
        )

        XCTAssertNil(layer.metalCode)
        if TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention {
            XCTAssertEqual(layer.activeBackend, .metalPolarQJL)
            XCTAssertTrue(layer.backendFallbackReason?.contains("linear matmul") ?? false)
        } else {
            XCTAssertEqual(layer.activeBackend, .mlxPacked)
            XCTAssertNotNil(layer.backendFallbackReason)
        }
    }

    func testTurboQuantMetalDecodeRejectsMalformedFlatStorageWhenAvailable() throws {
        guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLCodec else {
            throw XCTSkip("Metal runtime unavailable")
        }

        let x = MLXArray.ones([2, 64], dtype: .float32)
        var code = try turboQuantMetalEncode(
            x,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .key,
                groupSize: 64,
                backend: .metalPolarQJL
            )
        )
        code.signs = MLXArray.zeros([1], dtype: .uint32)

        XCTAssertThrowsError(try turboQuantMetalDecode(code)) { error in
            XCTAssertTrue(String(describing: error).contains("flat signs"))
        }
    }

    func testTurboQuantAttentionEncodeRejectsCapacityShorterThanInputWhenAvailable() throws {
        guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention else {
            throw XCTSkip("Metal compressed attention unavailable")
        }

        let values = MLXArray.ones([1, 1, 2, 64], dtype: .float32)
        XCTAssertThrowsError(
            try turboQuantMetalEncodeAttention(
                values,
                configuration: TurboQuantConfiguration(
                    preset: .turbo3_5,
                    role: .value,
                    groupSize: 64,
                    backend: .metalPolarQJL
                ),
                capacity: 1,
                logicalLength: 0
            )
        ) { error in
            XCTAssertTrue(
                String(describing: error).contains("exceeds compressed attention capacity"))
        }
    }

    func testTurboQuantAttentionDecodeRejectsMalformedStorageBeforeLaunch() throws {
        let layout = TurboQuantAttentionLayout(
            batchSize: 1,
            kvHeadCount: 1,
            capacity: 2,
            logicalLength: 2,
            headDimension: 64,
            groupsPerVector: 1,
            magnitudeWordsPerGroup: 5,
            bitsetWordsPerGroup: 2
        )
        let code = TurboQuantAttentionCode(
            layout: layout,
            preset: .turbo3_5,
            role: .key,
            groupSize: 64,
            seed: 0,
            packedMagnitudes: MLXArray.zeros([1, 1, 2, 1, 5], dtype: .uint32),
            signs: MLXArray.zeros([1, 1, 1, 1, 2], dtype: .uint32),
            highPrecisionMask: MLXArray.zeros([1, 1, 2, 1, 2], dtype: .uint32),
            residualSigns: MLXArray.zeros([1, 1, 2, 1, 2], dtype: .uint32),
            scales: MLXArray.zeros([1, 1, 2, 1, 3], dtype: .float32)
        )

        XCTAssertThrowsError(try turboQuantMetalDecodeAttention(code)) { error in
            XCTAssertTrue(String(describing: error).contains("compressed attention signs"))
        }
    }

    func testTurboQuantAttentionRejectsMalformedLayoutFieldsBeforeLaunch() throws {
        let invalidVersion = TurboQuantAttentionLayout(
            layoutVersion: 3,
            batchSize: 1,
            kvHeadCount: 1,
            capacity: 2,
            logicalLength: 2,
            headDimension: 64,
            groupsPerVector: 1,
            magnitudeWordsPerGroup: 5,
            bitsetWordsPerGroup: 2
        )
        let invalidPinnedPrefix = TurboQuantAttentionLayout(
            batchSize: 1,
            kvHeadCount: 1,
            capacity: 4,
            logicalLength: 2,
            pinnedPrefixLength: 3,
            headDimension: 64,
            groupsPerVector: 1,
            magnitudeWordsPerGroup: 5,
            bitsetWordsPerGroup: 2
        )
        let invalidGroupMath = TurboQuantAttentionLayout(
            batchSize: 1,
            kvHeadCount: 1,
            capacity: 2,
            logicalLength: 2,
            headDimension: 80,
            groupsPerVector: 1,
            magnitudeWordsPerGroup: 5,
            bitsetWordsPerGroup: 2
        )

        for (layout, expectedMessage) in [
            (invalidVersion, "layout version"),
            (invalidPinnedPrefix, "pinned prefix"),
            (invalidGroupMath, "groups per vector"),
        ] {
            let code = TurboQuantAttentionCode(
                layout: layout,
                preset: .turbo3_5,
                role: .key,
                groupSize: 64,
                seed: 0,
                packedMagnitudes: MLXArray.zeros([1, 1, 2, 1, 5], dtype: .uint32),
                signs: MLXArray.zeros([1, 1, 2, 1, 2], dtype: .uint32),
                highPrecisionMask: MLXArray.zeros([1, 1, 2, 1, 2], dtype: .uint32),
                residualSigns: MLXArray.zeros([1], dtype: .uint32),
                scales: MLXArray.zeros([1, 1, 2, 1, 3], dtype: .float32)
            )

            XCTAssertThrowsError(try turboQuantMetalDecodeAttention(code)) { error in
                XCTAssertTrue(String(describing: error).contains(expectedMessage))
            }
        }
    }

    func testTurboQuantAttentionRejectsNonCanonicalStorageBeforeLaunch() throws {
        let layout = TurboQuantAttentionLayout(
            batchSize: 1,
            kvHeadCount: 1,
            capacity: 2,
            logicalLength: 2,
            headDimension: 64,
            groupsPerVector: 1,
            magnitudeWordsPerGroup: 5,
            bitsetWordsPerGroup: 2
        )
        let backing = MLXArray.zeros([1, 1, 4, 1, 5], dtype: .uint32)
        let nonContiguousPacked = asStrided(
            backing,
            [1, 1, 2, 1, 5],
            strides: [20, 20, 10, 5, 1]
        )
        let code = TurboQuantAttentionCode(
            layout: layout,
            preset: .turbo3_5,
            role: .key,
            groupSize: 64,
            seed: 0,
            packedMagnitudes: nonContiguousPacked,
            signs: MLXArray.zeros([1, 1, 2, 1, 2], dtype: .uint32),
            highPrecisionMask: MLXArray.zeros([1, 1, 2, 1, 2], dtype: .uint32),
            residualSigns: MLXArray.zeros([1], dtype: .uint32),
            scales: MLXArray.zeros([1, 1, 2, 1, 3], dtype: .float32)
        )

        XCTAssertThrowsError(try turboQuantMetalDecodeAttention(code)) { error in
            XCTAssertTrue(String(describing: error).contains("row-contiguous"))
        }
    }

    func testTurboQuantAttentionRejectsRoleSpecificValueBitsetsBeforeLaunch() throws {
        let layout = TurboQuantAttentionLayout(
            batchSize: 1,
            kvHeadCount: 1,
            capacity: 2,
            logicalLength: 2,
            headDimension: 64,
            groupsPerVector: 1,
            magnitudeWordsPerGroup: 8,
            bitsetWordsPerGroup: 2
        )
        let code = TurboQuantAttentionCode(
            layout: layout,
            preset: .turbo3_5,
            role: .value,
            groupSize: 64,
            seed: 0,
            packedMagnitudes: MLXArray.zeros([1, 1, 2, 1, 8], dtype: .uint32),
            signs: MLXArray.zeros([1, 1, 2, 1, 2], dtype: .uint32),
            highPrecisionMask: MLXArray.zeros([1], dtype: .uint32),
            residualSigns: MLXArray.zeros([1], dtype: .uint32),
            scales: MLXArray.zeros([1, 1, 2, 1, 2], dtype: .float32)
        )

        XCTAssertThrowsError(try turboQuantMetalDecodeAttention(code)) { error in
            XCTAssertTrue(String(describing: error).contains("compressed attention signs"))
        }
    }

    func testTurboQuantAttentionRejectsMultipleMaterializedMasksBeforeLaunch() throws {
        let layout = TurboQuantAttentionLayout(
            batchSize: 1,
            kvHeadCount: 1,
            capacity: 2,
            logicalLength: 2,
            headDimension: 64,
            groupsPerVector: 1,
            magnitudeWordsPerGroup: 5,
            bitsetWordsPerGroup: 2
        )
        let code = TurboQuantAttentionCode(
            layout: layout,
            preset: .turbo3_5,
            role: .key,
            groupSize: 64,
            seed: 0,
            packedMagnitudes: MLXArray.zeros([1, 1, 2, 1, 5], dtype: .uint32),
            signs: MLXArray.zeros([1, 1, 2, 1, 2], dtype: .uint32),
            highPrecisionMask: MLXArray.zeros([1, 1, 2, 1, 2], dtype: .uint32),
            residualSigns: MLXArray.zeros([1, 1, 2, 1, 2], dtype: .uint32),
            scales: MLXArray.zeros([1, 1, 2, 1, 3], dtype: .float32)
        )
        let mask = MLXArray.zeros([1, 1, 1, 2], dtype: .float32)

        XCTAssertThrowsError(
            try turboQuantMetalQK(
                queries: MLXArray.ones([1, 1, 1, 64], dtype: .float32),
                keyCode: code,
                scale: 0.125,
                mask: .arrays([mask, mask])
            )
        ) { error in
            XCTAssertTrue(String(describing: error).contains("at most one materialized mask"))
        }
    }

    func testTurboQuantAttentionRejectsInvalidCausalQueryLengthBeforeLaunch() throws {
        let layout = TurboQuantAttentionLayout(
            batchSize: 1,
            kvHeadCount: 1,
            capacity: 2,
            logicalLength: 2,
            headDimension: 64,
            groupsPerVector: 1,
            magnitudeWordsPerGroup: 5,
            bitsetWordsPerGroup: 2
        )
        let code = TurboQuantAttentionCode(
            layout: layout,
            preset: .turbo3_5,
            role: .key,
            groupSize: 64,
            seed: 0,
            packedMagnitudes: MLXArray.zeros([1, 1, 2, 1, 5], dtype: .uint32),
            signs: MLXArray.zeros([1, 1, 2, 1, 2], dtype: .uint32),
            highPrecisionMask: MLXArray.zeros([1, 1, 2, 1, 2], dtype: .uint32),
            residualSigns: MLXArray.zeros([1, 1, 2, 1, 2], dtype: .uint32),
            scales: MLXArray.zeros([1, 1, 2, 1, 3], dtype: .float32)
        )

        XCTAssertThrowsError(
            try turboQuantMetalQK(
                queries: MLXArray.ones([1, 1, 3, 64], dtype: .float32),
                keyCode: code,
                scale: 0.125,
                mask: .causal
            )
        ) { error in
            XCTAssertTrue(String(describing: error).contains("query length 3 <= key length 2"))
        }
    }

    func testTurboQuantMetalCodecRoundTripWhenAvailable() throws {
        guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLCodec else {
            throw XCTSkip("Metal runtime unavailable")
        }

        let values = (0 ..< 128).map { index in
            Float(sin(Double(index) * 0.05))
        }
        let x = MLXArray(values, [2, 64])
        for seed in [UInt64(0xDEAD_BEEF_0000_0017), UInt64(0x0000_0000_DEAD_BEEF)] {
            let configuration = TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .key,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: seed
            )

            let code = try turboQuantMetalEncode(x, configuration: configuration)
            let decoded = try turboQuantMetalDecode(code).asArray(Float.self)
            XCTAssertEqual(code.shape, [2, 64])
            XCTAssertLessThan(relativeMSE(values, decoded), 0.1)
        }
    }

    func testTurboQuantMetalCodecUsesCompactUnusedBitsetsWhenAvailable() throws {
        guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLCodec else {
            throw XCTSkip("Metal runtime unavailable")
        }

        let values = (0 ..< 128).map { index in
            Float(0.2 * sin(Double(index) * 0.07))
        }
        let x = MLXArray(values, [2, 64])
        let keyCode = try turboQuantMetalEncode(
            x,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .key,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 0xC0DE
            )
        )
        let valueCode = try turboQuantMetalEncode(
            x,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .value,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 0xC0DE,
                valueBits: 4
            )
        )
        let uniformPrecisionKeyCode = try turboQuantMetalEncode(
            x,
            configuration: TurboQuantConfiguration(
                preset: .turbo4v2,
                role: .key,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 0xC0DE
            )
        )

        XCTAssertEqual(keyCode.signs.shape, [keyCode.groupCount * keyCode.bitsetWordsPerGroup])
        XCTAssertEqual(
            keyCode.highPrecisionMask.shape, [keyCode.groupCount * keyCode.bitsetWordsPerGroup])
        XCTAssertEqual(keyCode.residualSigns.shape, [1])
        XCTAssertEqual(valueCode.signs.shape, [1])
        XCTAssertEqual(valueCode.highPrecisionMask.shape, [1])
        XCTAssertEqual(valueCode.residualSigns.shape, [1])
        XCTAssertTrue(
            uniformPrecisionKeyCode.highPrecisionMask.asArray(UInt32.self).allSatisfy {
                $0 == 0
            })
        XCTAssertEqual(try turboQuantMetalDecode(keyCode).shape, x.shape)
        XCTAssertEqual(try turboQuantMetalDecode(valueCode).shape, x.shape)
    }

    func testTurboQuantMetalCodecUsesGPUStreamWhenDefaultDeviceIsCPU() throws {
        guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLCodec else {
            throw XCTSkip("Metal runtime unavailable")
        }

        let values = (0 ..< 128).map { index in
            Float(sin(Double(index) * 0.07))
        }
        let x = MLXArray(values, [2, 64])
        let configuration = TurboQuantConfiguration(
            preset: .turbo3_5,
            role: .key,
            groupSize: 64,
            backend: .metalPolarQJL,
            seed: 0xDEAD_BEEF_0000_0017
        )

        try Device.withDefaultDevice(.cpu) {
            XCTAssertTrue(StreamOrDevice.default.description.contains("cpu"))

            let code = try turboQuantMetalEncode(x, configuration: configuration)
            let decoded = try turboQuantMetalDecode(code).asArray(Float.self)

            XCTAssertEqual(code.shape, [2, 64])
            XCTAssertEqual(decoded.count, values.count)
        }
    }

    func testTurboQuantMetalMatmulMatchesDecodedReferenceWhenAvailable() throws {
        guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLCodec else {
            throw XCTSkip("Metal runtime unavailable")
        }

        let xValues = (0 ..< 192).map { index in
            let position = Double(index)
            return Float(0.4 * sin(position * 0.07) + 0.2 * cos(position * 0.17))
        }
        let wValues = (0 ..< 320).map { index in
            let position = Double(index)
            return Float(0.3 * cos(position * 0.05) - 0.15 * sin(position * 0.11))
        }
        let x = MLXArray(xValues, [3, 64])
        let w = MLXArray(wValues, [5, 64])
        let configuration = TurboQuantConfiguration(
            preset: .turbo3_5,
            role: .key,
            groupSize: 64,
            backend: .metalPolarQJL,
            seed: 0xC0FF_EE00_0000_0042
        )

        let code = try turboQuantMetalEncode(w, configuration: configuration)
        let referenceCode = try turboQuantReferenceEncode(w, configuration: configuration)
        var expectedValues: [Float] = []
        for row in 0 ..< 3 {
            for column in 0 ..< 5 {
                var query = [Float](repeating: 0, count: wValues.count)
                for k in 0 ..< 64 {
                    query[column * 64 + k] = xValues[row * 64 + k]
                }
                expectedValues.append(
                    try turboQuantReferenceInnerProduct(
                        query: MLXArray(query, [5, 64]),
                        code: referenceCode
                    )
                )
            }
        }
        let reference = MLXArray(expectedValues, [3, 5])
        let output = try turboQuantizedMM(x, code, transpose: true, outputDType: .float32)

        XCTAssertEqual(output.shape, [3, 5])
        XCTAssertTrue(allClose(output, reference, rtol: 1e-4, atol: 1e-4).item(Bool.self))
        XCTAssertEqual(code.magnitudeWordsPerGroup, 5)

        let decoded = try turboQuantMetalDecode(code, dtype: .float32)
        let columnMajorWeight = decoded.transposed()
        let columnCode = try turboQuantMetalEncode(columnMajorWeight, configuration: configuration)
        let columnReference = matmul(x, try turboQuantMetalDecode(columnCode, dtype: .float32))
        let columnOutput = try turboQuantizedMM(
            x, columnCode, transpose: false, outputDType: .float32)

        XCTAssertEqual(columnOutput.shape, [3, 5])
        XCTAssertTrue(
            allClose(columnOutput, columnReference, rtol: 1e-4, atol: 1e-4).item(Bool.self))
    }

    func testTurboQuantMetalMatmulSupportsBFloat16OutputWhenAvailable() throws {
        guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLCodec else {
            throw XCTSkip("Metal runtime unavailable")
        }

        let xValues = (0 ..< 128).map { index in
            let position = Double(index)
            return Float(0.2 * sin(position * 0.07) + 0.1 * cos(position * 0.13))
        }
        let wValues = (0 ..< 192).map { index in
            let position = Double(index)
            return Float(0.25 * cos(position * 0.05) - 0.08 * sin(position * 0.17))
        }
        let x = MLXArray(xValues, [2, 64]).asType(.bfloat16)
        let w = MLXArray(wValues, [3, 64])
        let configuration = TurboQuantConfiguration(
            preset: .turbo4v2,
            role: .vector,
            groupSize: 64,
            backend: .metalPolarQJL,
            seed: 0xBEEF_0000_0000_0042
        )

        let code = try turboQuantMetalEncode(w, configuration: configuration)
        let output = try turboQuantizedMM(x, code, transpose: true, outputDType: x.dtype)
        eval(output)

        XCTAssertEqual(output.dtype, DType.bfloat16)
        XCTAssertEqual(output.shape, [2, 3])
        XCTAssertTrue(output.asArray(Float.self).allSatisfy { $0.isFinite })
    }

    func testTurboQuantAttentionLayoutIsRowWise() throws {
        let layout = try turboQuantAttentionLayout(shape: [1, 2, 3, 80], groupSize: 64)

        XCTAssertEqual(layout.layoutVersion, TurboQuantAttentionLayout.currentVersion)
        XCTAssertEqual(layout.logicalShape, [1, 2, 3, 80])
        XCTAssertEqual(layout.pinnedPrefixLength, 0)
        XCTAssertEqual(layout.groupsPerVector, 2)
        XCTAssertEqual(layout.bitsetWordsPerGroup, 2)
    }

    func testTurboQuantAttentionLayoutV5RequiresExplicitOptIn() throws {
        XCTAssertThrowsError(
            try turboQuantAttentionLayout(
                shape: [1, 2, 3, 80],
                groupSize: 64,
                layoutVersion: TurboQuantAttentionLayout.nextVersion
            )
        ) { error in
            XCTAssertEqual(
                error as? TurboQuantError,
                .invalidMetalConfiguration(
                    "TurboQuant layout V5 is experimental and disabled by default"
                )
            )
        }

        let layout = try turboQuantAttentionLayout(
            shape: [1, 2, 3, 80],
            groupSize: 64,
            layoutVersion: TurboQuantAttentionLayout.nextVersion,
            allowExperimentalLayoutV5: true
        )

        XCTAssertEqual(layout.layoutVersion, TurboQuantAttentionLayout.nextVersion)
        XCTAssertTrue(layout.isLayoutV5)
        XCTAssertEqual(layout.logicalShape, [1, 2, 3, 80])

        XCTAssertThrowsError(
            try turboQuantEmptyAttentionCode(
                layout: layout,
                preset: .turbo4v2,
                role: .key,
                groupSize: 64
            )
        ) { error in
            XCTAssertEqual(
                error as? TurboQuantError,
                .invalidMetalConfiguration(
                    "TurboQuant layout V5 is experimental and disabled by default"
                )
            )
        }

        _ = try turboQuantEmptyAttentionCode(
            layout: layout,
            preset: .turbo4v2,
            role: .key,
            groupSize: 64,
            allowExperimentalLayoutV5: true
        )
    }

    func testTurboQuantAttentionLayoutV5Fp16ScalesReduceStorageEstimate() {
        let logicalValues = 1 * 2 * 256 * 128
        let v4 = estimateTurboQuantStorage(
            role: .key,
            logicalValues: logicalValues,
            preset: .turbo4v2,
            groupSize: 64,
            dtype: .float32
        )
        let v5 = estimateTurboQuantStorage(
            role: .key,
            logicalValues: logicalValues,
            preset: .turbo4v2,
            groupSize: 64,
            dtype: .float32,
            scaleStorage: .float16
        )

        XCTAssertLessThan(v5.scaleBytes, v4.scaleBytes)
        XCTAssertLessThan(v5.actualBitsPerValue, v4.actualBitsPerValue)
        XCTAssertEqual(v5.packedBytes, v4.packedBytes)
        XCTAssertEqual(v5.bitsetBytes, v4.bitsetBytes)
    }

    func testTurboQuantAttentionLayoutUsesPresetHighPrecisionFraction() throws {
        let turbo35 = try turboQuantAttentionLayout(
            shape: [1, 1, 1, 64],
            preset: .turbo3_5,
            role: .key,
            groupSize: 64
        )
        let turbo25 = try turboQuantAttentionLayout(
            shape: [1, 1, 1, 64],
            preset: .turbo2_5,
            role: .key,
            groupSize: 64
        )
        let turbo4v2 = try turboQuantAttentionLayout(
            shape: [1, 1, 1, 64],
            preset: .turbo4v2,
            role: .key,
            groupSize: 64
        )

        XCTAssertEqual(turbo25.magnitudeWordsPerGroup, 3)
        XCTAssertEqual(turbo35.magnitudeWordsPerGroup, 5)
        XCTAssertEqual(turbo4v2.magnitudeWordsPerGroup, 6)
    }

    func testTurboQuantAttentionCodeUsesCompactUnusedBitsetsWhenAvailable() throws {
        guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention else {
            throw XCTSkip("Metal compressed attention unavailable")
        }

        let keys = MLXArray.ones([1, 1, 2, 64], dtype: .float32)
        let values = keys + 0.25
        let keyCode = try turboQuantMetalEncodeAttention(
            keys,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .key,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 0xCAFE
            )
        )
        let valueCode = try turboQuantMetalEncodeAttention(
            values,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .value,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 0xCAFE,
                valueBits: 4
            )
        )

        XCTAssertEqual(keyCode.signs.shape, [1, 1, 2, 1, 2])
        XCTAssertEqual(keyCode.highPrecisionMask.shape, [1, 1, 2, 1, 2])
        XCTAssertEqual(keyCode.residualSigns.shape, [1])
        XCTAssertEqual(valueCode.signs.shape, [1])
        XCTAssertEqual(valueCode.highPrecisionMask.shape, [1])
        XCTAssertEqual(valueCode.residualSigns.shape, [1])
        XCTAssertEqual(try turboQuantMetalDecodeAttention(keyCode).shape, keys.shape)
        XCTAssertEqual(try turboQuantMetalDecodeAttention(valueCode).shape, values.shape)
    }

    func testTurboQuantCompressedAttentionUsesProductEstimatorWhenAvailable() throws {
        guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention else {
            throw XCTSkip("Metal compressed attention unavailable")
        }

        let qValues: [Float] = (0 ..< 512).map { index in
            let position = Double(index)
            return Float(sin(position * 0.03) + 0.2 * cos(position * 0.11))
        }
        let kValues: [Float] = (0 ..< 640).map { index in
            let position = Double(index)
            return Float(cos(position * 0.05) * 0.5 + sin(position * 0.17) * 0.1)
        }
        let vValues: [Float] = (0 ..< 640).map { index in
            let position = Double(index)
            return Float(sin(position * 0.07) * 0.25 - cos(position * 0.13) * 0.2)
        }
        let queries = MLXArray(qValues, [1, 4, 2, 64])
        let keys = MLXArray(kValues, [1, 2, 5, 64])
        let values = MLXArray(vValues, [1, 2, 5, 64])
        let keyCode = try turboQuantMetalEncodeAttention(
            keys,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .key,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 11
            )
        )
        let valueCode = try turboQuantMetalEncodeAttention(
            values,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .value,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 13
            )
        )
        let fullPrecisionReference = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: 1 / sqrt(Float(64)),
            mask: .causal
        )

        let twoStage = try turboQuantMetalScaledDotProductAttention(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            scale: 1 / sqrt(Float(64)),
            mask: .causal,
            preferOnlineFused: false
        )
        let fused = try turboQuantMetalScaledDotProductAttention(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            scale: 1 / sqrt(Float(64)),
            mask: .causal,
            preferOnlineFused: true
        )

        XCTAssertEqual(twoStage.shape, [1, 4, 2, 64])
        XCTAssertEqual(fused.shape, [1, 4, 2, 64])
        XCTAssertTrue(allClose(fused, twoStage, rtol: 1e-4, atol: 1e-4).item(Bool.self))
        XCTAssertLessThan(
            relativeMSE(
                fullPrecisionReference.asArray(Float.self),
                fused.asArray(Float.self)
            ),
            0.12
        )
        XCTAssertLessThan(
            relativeMSE(
                fullPrecisionReference.asArray(Float.self),
                twoStage.asArray(Float.self)
            ),
            0.12
        )
    }

    func testTurboQuantCompressedAttentionUsesOnlineFusedForLargeHeadsWhenAvailable() throws {
        guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention else {
            throw XCTSkip("Metal compressed attention unavailable")
        }

        for headDimension in [192, 256] {
            let qValues: [Float] = (0 ..< (1 * 4 * 2 * headDimension)).map { index in
                let position = Double(index)
                return Float(0.31 * sin(position * 0.017) + 0.19 * cos(position * 0.059))
            }
            let kValues: [Float] = (0 ..< (1 * 2 * 5 * headDimension)).map { index in
                let position = Double(index)
                return Float(0.24 * cos(position * 0.023) - 0.13 * sin(position * 0.083))
            }
            let vValues: [Float] = (0 ..< (1 * 2 * 5 * headDimension)).map { index in
                let position = Double(index)
                return Float(0.21 * sin(position * 0.037) + 0.15 * cos(position * 0.067))
            }
            let queries = MLXArray(qValues, [1, 4, 2, headDimension])
            let keys = MLXArray(kValues, [1, 2, 5, headDimension])
            let values = MLXArray(vValues, [1, 2, 5, headDimension])
            let keyCode = try turboQuantMetalEncodeAttention(
                keys,
                configuration: TurboQuantConfiguration(
                    preset: .turbo4v2,
                    role: .key,
                    groupSize: 64,
                    backend: .metalPolarQJL,
                    seed: 71
                )
            )
            let valueCode = try turboQuantMetalEncodeAttention(
                values,
                configuration: TurboQuantConfiguration(
                    preset: .turbo4v2,
                    role: .value,
                    groupSize: 64,
                    backend: .metalPolarQJL,
                    seed: 73,
                    valueBits: 4
                )
            )

            XCTAssertTrue(
                turboQuantMetalSupportsOnlineFusedAttention(
                    queries: queries,
                    keyCode: keyCode,
                    mask: .causal
                ),
                "Expected online fused support for \(headDimension)-dimensional heads"
            )
            let twoStage = try turboQuantMetalScaledDotProductAttention(
                queries: queries,
                keyCode: keyCode,
                valueCode: valueCode,
                scale: 1 / sqrt(Float(headDimension)),
                mask: .causal,
                preferOnlineFused: false
            )
            let fused = try turboQuantMetalScaledDotProductAttention(
                queries: queries,
                keyCode: keyCode,
                valueCode: valueCode,
                scale: 1 / sqrt(Float(headDimension)),
                mask: .causal,
                preferOnlineFused: true
            )

            XCTAssertEqual(twoStage.shape, [1, 4, 2, headDimension])
            XCTAssertEqual(fused.shape, [1, 4, 2, headDimension])
            XCTAssertTrue(
                allClose(fused, twoStage, rtol: 1e-4, atol: 1e-4).item(Bool.self),
                "Expected online fused output to match two-stage output for \(headDimension)-dimensional heads"
            )
        }
    }

    func testTurboQuantCompressedAttentionSupportsBatchedInputsWhenAvailable() throws {
        guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention else {
            throw XCTSkip("Metal compressed attention unavailable")
        }

        let qValues: [Float] = (0 ..< 1024).map { index in
            let position = Double(index)
            return Float(0.3 * sin(position * 0.021) + 0.17 * cos(position * 0.071))
        }
        let kValues: [Float] = (0 ..< 1280).map { index in
            let position = Double(index)
            return Float(0.25 * cos(position * 0.037) - 0.11 * sin(position * 0.097))
        }
        let vValues: [Float] = (0 ..< 1280).map { index in
            let position = Double(index)
            return Float(0.19 * sin(position * 0.043) + 0.13 * cos(position * 0.083))
        }
        let queries = MLXArray(qValues, [2, 4, 2, 64])
        let keys = MLXArray(kValues, [2, 2, 5, 64])
        let values = MLXArray(vValues, [2, 2, 5, 64])
        let keyCode = try turboQuantMetalEncodeAttention(
            keys,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .key,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 31
            )
        )
        let valueCode = try turboQuantMetalEncodeAttention(
            values,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .value,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 37
            )
        )
        let fullPrecisionReference = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: 1 / sqrt(Float(64)),
            mask: .causal
        )

        let twoStage = try turboQuantMetalScaledDotProductAttention(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            scale: 1 / sqrt(Float(64)),
            mask: .causal,
            preferOnlineFused: false
        )
        let fused = try turboQuantMetalScaledDotProductAttention(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            scale: 1 / sqrt(Float(64)),
            mask: .causal,
            preferOnlineFused: true
        )

        XCTAssertEqual(twoStage.shape, [2, 4, 2, 64])
        XCTAssertEqual(fused.shape, [2, 4, 2, 64])
        XCTAssertTrue(allClose(fused, twoStage, rtol: 1e-4, atol: 1e-4).item(Bool.self))
        XCTAssertLessThan(
            relativeMSE(
                fullPrecisionReference.asArray(Float.self),
                fused.asArray(Float.self)
            ),
            0.12
        )
    }

    func testTurboQuantCompressedAttentionSupportsSinksWhenAvailable() throws {
        guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention else {
            throw XCTSkip("Metal compressed attention unavailable")
        }

        let qValues: [Float] = (0 ..< 512).map { index in
            let position = Double(index)
            return Float(0.24 * sin(position * 0.031) + 0.12 * cos(position * 0.089))
        }
        let kValues: [Float] = (0 ..< 640).map { index in
            let position = Double(index)
            return Float(0.2 * cos(position * 0.047) - 0.08 * sin(position * 0.101))
        }
        let vValues: [Float] = (0 ..< 640).map { index in
            let position = Double(index)
            return Float(0.18 * sin(position * 0.053) + 0.09 * cos(position * 0.077))
        }
        let queries = MLXArray(qValues, [1, 4, 2, 64])
        let keys = MLXArray(kValues, [1, 2, 5, 64])
        let values = MLXArray(vValues, [1, 2, 5, 64])
        let sinks = MLXArray([0.3 as Float, -0.2, 0.1, -0.4])
        let keyCode = try turboQuantMetalEncodeAttention(
            keys,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .key,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 41
            )
        )
        let valueCode = try turboQuantMetalEncodeAttention(
            values,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .value,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 43
            )
        )
        let reference = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: 1 / sqrt(Float(64)),
            mask: .causal,
            sinks: sinks
        )

        let output = try turboQuantMetalScaledDotProductAttention(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            scale: 1 / sqrt(Float(64)),
            mask: .causal,
            sinks: sinks,
            preferOnlineFused: true
        )

        XCTAssertEqual(output.shape, [1, 4, 2, 64])
        XCTAssertLessThan(
            relativeMSE(
                reference.asArray(Float.self),
                output.asArray(Float.self)
            ),
            0.12
        )
    }

    func testTurboQuantCompressedAttentionSupportsSplitKeyValueDimensionsWhenAvailable() throws {
        guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention else {
            throw XCTSkip("Metal compressed attention unavailable")
        }

        let qValues: [Float] = (0 ..< 512).map { index in
            let position = Double(index)
            return Float(0.21 * sin(position * 0.029) + 0.16 * cos(position * 0.061))
        }
        let kValues: [Float] = (0 ..< 640).map { index in
            let position = Double(index)
            return Float(0.18 * cos(position * 0.041) - 0.12 * sin(position * 0.087))
        }
        let vValues: [Float] = (0 ..< 800).map { index in
            let position = Double(index)
            return Float(0.22 * sin(position * 0.049) + 0.10 * cos(position * 0.093))
        }
        let queries = MLXArray(qValues, [1, 4, 2, 64])
        let keys = MLXArray(kValues, [1, 2, 5, 64])
        let values = MLXArray(vValues, [1, 2, 5, 80])
        let keyCode = try turboQuantMetalEncodeAttention(
            keys,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .key,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 51
            )
        )
        let valueCode = try turboQuantMetalEncodeAttention(
            values,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .value,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 53
            )
        )

        let scores = try turboQuantMetalQK(
            queries: queries,
            keyCode: keyCode,
            scale: 1 / sqrt(Float(64)),
            mask: .causal
        )
        let twoStage = try turboQuantMetalAV(
            attentionWeights: softmax(scores.asType(.float32), axis: -1),
            valueCode: valueCode,
            outputDType: queries.dtype
        )
        let fusedPreferred = try turboQuantMetalScaledDotProductAttention(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            scale: 1 / sqrt(Float(64)),
            mask: .causal,
            preferOnlineFused: true
        )

        XCTAssertEqual(twoStage.shape, [1, 4, 2, 80])
        XCTAssertEqual(fusedPreferred.shape, [1, 4, 2, 80])
        XCTAssertTrue(allClose(fusedPreferred, twoStage, rtol: 1e-4, atol: 1e-4).item(Bool.self))
    }

    func testTurboQuantCompressedAttentionSupportsBFloat16OutputsWhenAvailable() throws {
        guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention else {
            throw XCTSkip("Metal compressed attention unavailable")
        }
        guard TurboQuantKernelAvailability.current.kernelCapabilities.bfloatOutput else {
            throw XCTSkip("TurboQuant bfloat16 compressed attention output unavailable")
        }

        let qValues: [Float] = (0 ..< 512).map { index in
            let position = Double(index)
            return Float(0.28 * sin(position * 0.023) + 0.11 * cos(position * 0.079))
        }
        let kValues: [Float] = (0 ..< 640).map { index in
            let position = Double(index)
            return Float(0.19 * cos(position * 0.043) - 0.07 * sin(position * 0.097))
        }
        let vValues: [Float] = (0 ..< 640).map { index in
            let position = Double(index)
            return Float(0.23 * sin(position * 0.037) + 0.12 * cos(position * 0.071))
        }
        let queries = MLXArray(qValues, [1, 4, 2, 64]).asType(.bfloat16)
        let keys = MLXArray(kValues, [1, 2, 5, 64]).asType(.bfloat16)
        let values = MLXArray(vValues, [1, 2, 5, 64]).asType(.bfloat16)
        let keyCode = try turboQuantMetalEncodeAttention(
            keys,
            configuration: TurboQuantConfiguration(
                preset: .turbo4v2,
                role: .key,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 61
            )
        )
        let valueCode = try turboQuantMetalEncodeAttention(
            values,
            configuration: TurboQuantConfiguration(
                preset: .turbo4v2,
                role: .value,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 67
            )
        )

        let decodedValues = try turboQuantMetalDecodeAttention(
            valueCode, outputDType: queries.dtype)
        let scores = try turboQuantMetalQK(
            queries: queries,
            keyCode: keyCode,
            scale: 1 / sqrt(Float(64)),
            mask: .causal
        )
        let twoStage = try turboQuantMetalAV(
            attentionWeights: softmax(scores.asType(.float32), axis: -1),
            valueCode: valueCode,
            outputDType: queries.dtype
        )
        let fusedPreferred = try turboQuantMetalScaledDotProductAttention(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            scale: 1 / sqrt(Float(64)),
            mask: .causal,
            preferOnlineFused: true
        )
        eval(decodedValues, twoStage, fusedPreferred)

        XCTAssertEqual(decodedValues.dtype, DType.bfloat16)
        XCTAssertEqual(decodedValues.shape, [1, 2, 5, 64])
        XCTAssertEqual(twoStage.dtype, DType.bfloat16)
        XCTAssertEqual(twoStage.shape, [1, 4, 2, 64])
        XCTAssertEqual(fusedPreferred.dtype, DType.bfloat16)
        XCTAssertEqual(fusedPreferred.shape, [1, 4, 2, 64])
        XCTAssertTrue(twoStage.asArray(Float.self).allSatisfy { $0.isFinite })
        XCTAssertTrue(fusedPreferred.asArray(Float.self).allSatisfy { $0.isFinite })
    }

    func testTurboQuantAttentionDecodeHonorsRotatingLayoutWhenAvailable() throws {
        guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention else {
            throw XCTSkip("Metal compressed attention unavailable")
        }

        let capacity = 6
        let headDimension = 64
        let physicalValues = (0 ..< capacity).flatMap { token in
            Array(repeating: Float(token + 1) * 0.25, count: headDimension)
        }
        let physical = MLXArray(physicalValues, [1, 1, capacity, headDimension])
        let code = try turboQuantMetalEncodeAttention(
            physical,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .value,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 29
            ),
            capacity: capacity,
            logicalLength: capacity,
            ringOffset: 2,
            pinnedPrefixLength: 2
        )

        let decoded = try turboQuantMetalDecodeAttention(code, outputDType: .float32)
        let expectedTokenOrder = [0, 1, 4, 5, 2, 3]
        let expectedValues = expectedTokenOrder.flatMap { token in
            Array(repeating: Float(token + 1) * 0.25, count: headDimension)
        }
        let expected = MLXArray(expectedValues, [1, 1, capacity, headDimension])

        XCTAssertTrue(allClose(decoded, expected, rtol: 1e-6, atol: 1e-6).item(Bool.self))
    }

    func testTurboQuantOnlineFusedSupportContract() throws {
        for headDimension in TurboQuantRuntimeProbeResult
            .throughputOptimizedOnlineFusedHeadDimensions
        {
            let keyLayout = try turboQuantAttentionLayout(
                shape: [1, 2, 8, headDimension],
                groupSize: 64
            )

            XCTAssertTrue(
                turboQuantMetalSupportsOnlineFusedAttention(
                    queryShape: [1, 4, 1, headDimension],
                    keyLayout: keyLayout,
                    mask: .none
                )
            )
        }
    }

    func testTurboQuantOnlineFusedSupportsLargeContextContract() throws {
        let keyLayout = try turboQuantAttentionLayout(shape: [1, 2, 513, 64], groupSize: 64)

        XCTAssertTrue(
            turboQuantMetalSupportsOnlineFusedAttention(
                queryShape: [1, 4, 1, 64],
                keyLayout: keyLayout,
                mask: .none
            )
        )
    }
}
