// Copyright © 2026 RNT56.

import Foundation
import MLX
import XCTest

final class TurboQuantNativeAttentionTests: XCTestCase {
    private func withEnvironment(_ name: String, value: String?, _ body: () throws -> Void)
        rethrows
    {
        let previous = getenv(name).map { String(cString: $0) }
        if let value {
            setenv(name, value, 1)
        } else {
            unsetenv(name)
        }
        defer {
            if let previous {
                setenv(name, previous, 1)
            } else {
                unsetenv(name)
            }
        }
        try body()
    }

    private func requireNativeBackend() throws -> TurboQuantNativeSegmentedAttentionBackend {
        guard turboQuantNativeMLXAttentionEnabled() else {
            throw XCTSkip("native MLX TurboQuant attention is explicitly disabled")
        }

        let backend = turboQuantNativeSegmentedAttentionBackend(allowExperimentalJIT: true)
        guard backend == .nativeFused || backend == .experimentalJIT else {
            throw XCTSkip(
                "native MLX TurboQuant attention is unavailable: \(backend)"
            )
        }
        return backend
    }

    func testSegmentedSwiftSurfaceRetainsScaledCompatibilityNames() {
        XCTAssertEqual(TurboQuantNativeSegmentedAttentionBackend.unavailable.rawValue, 0)
        XCTAssertEqual(TurboQuantNativeSegmentedAttentionBackend.experimentalJIT.rawValue, 1)
        XCTAssertEqual(TurboQuantNativeSegmentedAttentionBackend.nativeFused.rawValue, 2)
        XCTAssertEqual(TurboQuantNativeSegmentedAttentionCodec.polarQJL.rawValue, 0)
        XCTAssertEqual(TurboQuantNativeSegmentedAttentionCodec.polarWHT.rawValue, 1)
        XCTAssertEqual(TurboQuantNativeSegmentedAttentionCodec.hybridK8PolarWHTValue.rawValue, 2)
        XCTAssertEqual(TurboQuantNativeSegmentedAttentionCodec.polarQJL.requestedBackend, .metalPolarQJL)
        XCTAssertEqual(TurboQuantNativeSegmentedAttentionCodec.polarWHT.requestedBackend, .metalPolarWHT)

        let result = TurboQuantNativeSegmentedAttentionResult(
            output: MLXArray.zeros([1], dtype: .float32)
        )
        XCTAssertEqual(result.output.shape, [1])
    }

    func testPolarWHTNativeProbeReflectsCertifiedKernels() {
        let defaultBackend = turboQuantNativeSegmentedAttentionBackend(
            allowExperimentalJIT: false
        )
        let explicitPolarQJLBackend = turboQuantNativeSegmentedAttentionBackend(
            codec: .polarQJL,
            allowExperimentalJIT: false
        )
        let polarWHTBackend = turboQuantNativeSegmentedAttentionBackend(
            codec: .polarWHT,
            allowExperimentalJIT: true
        )
        let hybridBackend = turboQuantNativeSegmentedAttentionBackend(
            codec: .hybridK8PolarWHTValue,
            allowExperimentalJIT: true
        )

        XCTAssertEqual(explicitPolarQJLBackend, defaultBackend)
        let capabilities = TurboQuantKernelAvailability.current.attentionCapabilities
        if capabilities.polarWHTAttention {
            XCTAssertEqual(polarWHTBackend, .experimentalJIT)
            XCTAssertTrue(
                turboQuantNativeSegmentedAttentionIsAvailable(
                    codec: .polarWHT,
                    allowExperimentalJIT: true
                )
            )
            XCTAssertTrue(capabilities.polarWHTCodec)
            XCTAssertTrue(TurboQuantKernelAvailability.current.supports(.metalPolarWHT))
        } else {
            XCTAssertEqual(polarWHTBackend, .unavailable)
            XCTAssertFalse(
                turboQuantNativeSegmentedAttentionIsAvailable(
                    codec: .polarWHT,
                    allowExperimentalJIT: true
                )
            )
            XCTAssertFalse(TurboQuantKernelAvailability.current.supports(.metalPolarWHT))
        }
        if capabilities.hybridK8PolarWHTValueAttention {
            XCTAssertEqual(hybridBackend, .experimentalJIT)
        } else {
            XCTAssertEqual(hybridBackend, .unavailable)
        }
    }

    func testProductionNativeCapabilityReportsMetalBackend() {
        XCTAssertEqual(
            turboQuantNativeSegmentedAttentionBackend(allowExperimentalJIT: false),
            .nativeFused
        )
    }

    func testNativeCapabilityProbePassesWhenEnabled() throws {
        let backend = try requireNativeBackend()

        let capabilities = TurboQuantKernelAvailability.current.attentionCapabilities
        XCTAssertEqual(capabilities.nativeCompressedAttention, true)
        XCTAssertEqual(capabilities.nativeSparseVSupport, true)
        XCTAssertEqual(capabilities.nativeDiagnosticsSupport, true)
        XCTAssertEqual(capabilities.nativeBackendVersion, TurboQuantNativeAttentionOptions.backendVersion)
        XCTAssertEqual(capabilities.nativeSegmentedAttentionBackend, backend)
    }

    func testNativeFusedAttentionMatchesSwiftMetalWhenEnabled() throws {
        _ = try requireNativeBackend()

        let tokenCount = 32
        let headDimension = 64
        let queryHeadCount = 4
        let keyValues = makeWaveValues(
            count: tokenCount * headDimension,
            sinScale: 0.031,
            sinWeight: 0.2,
            cosScale: 0.017,
            cosWeight: 0.1
        )
        let valueValues = makeWaveValues(
            count: tokenCount * headDimension,
            sinScale: 0.041,
            sinWeight: -0.07,
            cosScale: 0.023,
            cosWeight: 0.3
        )
        let queryValues = makeWaveValues(
            count: queryHeadCount * headDimension,
            sinScale: 0.071,
            sinWeight: 0.15,
            cosScale: 0,
            cosWeight: 0
        )

        let keys = MLXArray(keyValues, [1, 1, tokenCount, headDimension])
        let values = MLXArray(valueValues, [1, 1, tokenCount, headDimension])
        let queries = MLXArray(queryValues, [1, queryHeadCount, 1, headDimension])

        let keyCode = try turboQuantMetalEncodeAttention(
            keys,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .key,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 0xA77E_0000_0000_0101
            )
        )
        let valueCode = try turboQuantMetalEncodeAttention(
            values,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .value,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 0xA77E_0000_0000_0102,
                valueBits: 4
            )
        )
        let scale = 1 / sqrt(Float(headDimension))

        let swiftMetal = try swiftMetalTwoStageReference(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            scale: scale,
            mask: .causal
        )
        let native = try turboQuantNativeScaledDotProductAttentionWithDiagnostics(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            options: TurboQuantNativeAttentionOptions(
                scale: scale,
                causal: true,
                diagnostics: true
            )
        )

        eval(swiftMetal, native.output)
        assertEqual(native.output, swiftMetal, rtol: 1e-4, atol: 1e-4)
        XCTAssertEqual(native.diagnostics?.kernelKind, 1)
        XCTAssertEqual(native.diagnostics?.fallbackCode, 0)
    }

    func testHybridK8PolarWHTValueFusedAttentionMatchesTwoStageReference() throws {
        guard TurboQuantKernelAvailability.current.attentionCapabilities.hybridK8PolarWHTValueAttention
        else {
            throw XCTSkip("hybrid K8 + PolarWHT-V attention is unavailable")
        }

        let tokenCount = 128
        let headDimension = 64
        let queryHeadCount = 4
        let keyValues = makeWaveValues(
            count: tokenCount * headDimension,
            sinScale: 0.017,
            sinWeight: 0.18,
            cosScale: 0.011,
            cosWeight: 0.07
        )
        let valueValues = makeWaveValues(
            count: tokenCount * headDimension,
            sinScale: 0.029,
            sinWeight: -0.05,
            cosScale: 0.013,
            cosWeight: 0.21
        )
        let queryValues = makeWaveValues(
            count: queryHeadCount * headDimension,
            sinScale: 0.041,
            sinWeight: 0.13,
            cosScale: 0.023,
            cosWeight: 0.03
        )

        let keys = MLXArray(keyValues, [1, 1, tokenCount, headDimension])
        let values = MLXArray(valueValues, [1, 1, tokenCount, headDimension])
        let queries = MLXArray(queryValues, [1, queryHeadCount, 1, headDimension])
        let keyCode = try turboQuantMetalEncodeAttention(
            keys,
            configuration: TurboQuantConfiguration(
                preset: .turbo8,
                role: .key,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 0xA77E_0000_0000_0701
            )
        )
        let valueCode = try turboQuantMetalPolarWHTEncodeAttentionValues(
            values,
            bits: 4,
            seed: 0xA77E_0000_0000_0702
        )
        let scale = 1 / sqrt(Float(headDimension))
        let scores = try turboQuantMetalQK(
            queries: queries,
            keyCode: keyCode,
            scale: scale,
            mask: .causal
        )
        let weights = softmax(scores.asType(.float32), axis: -1)
        let reference = try turboQuantMetalPolarWHTAV(
            attentionWeights: weights,
            valueCode: valueCode,
            outputDType: queries.dtype
        )
        let fused = try turboQuantMetalHybridPolarWHTValueScaledDotProductAttention(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            scale: scale,
            mask: .causal,
            outputDType: queries.dtype
        )

        eval(reference, fused)
        assertEqual(fused, reference, rtol: 1e-4, atol: 1e-4)
    }

    func testHybridK8PolarWHTValueBlockParallelFusedMatchesTwoStageReference() throws {
        guard TurboQuantKernelAvailability.current.attentionCapabilities.hybridK8PolarWHTValueAttention
        else {
            throw XCTSkip("hybrid K8 + PolarWHT-V attention is unavailable")
        }

        let tokenCount = 4096
        let headDimension = 64
        let queryHeadCount = 4
        let keys = MLXArray(
            makeWaveValues(
                count: tokenCount * headDimension,
                sinScale: 0.003,
                sinWeight: 0.18,
                cosScale: 0.005,
                cosWeight: 0.07
            ),
            [1, 1, tokenCount, headDimension]
        )
        let values = MLXArray(
            makeWaveValues(
                count: tokenCount * headDimension,
                sinScale: 0.007,
                sinWeight: -0.05,
                cosScale: 0.011,
                cosWeight: 0.21
            ),
            [1, 1, tokenCount, headDimension]
        )
        let queries = MLXArray(
            makeWaveValues(
                count: queryHeadCount * headDimension,
                sinScale: 0.041,
                sinWeight: 0.13,
                cosScale: 0.023,
                cosWeight: 0.03
            ),
            [1, queryHeadCount, 1, headDimension]
        )
        let keyCode = try turboQuantMetalEncodeAttention(
            keys,
            configuration: TurboQuantConfiguration(
                preset: .turbo8,
                role: .key,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 0xA77E_0000_0000_0711
            )
        )
        let valueCode = try turboQuantMetalPolarWHTEncodeAttentionValues(
            values,
            bits: 4,
            seed: 0xA77E_0000_0000_0712
        )
        let scale = 1 / sqrt(Float(headDimension))
        let scores = try turboQuantMetalQK(
            queries: queries,
            keyCode: keyCode,
            scale: scale,
            mask: .causal
        )
        let weights = softmax(scores.asType(.float32), axis: -1)
        let reference = try turboQuantMetalPolarWHTAV(
            attentionWeights: weights,
            valueCode: valueCode,
            outputDType: queries.dtype
        )
        let fused = try turboQuantMetalHybridPolarWHTValueScaledDotProductAttention(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            scale: scale,
            mask: .causal,
            outputDType: queries.dtype
        )

        eval(reference, fused)
        assertEqual(fused, reference, rtol: 1e-4, atol: 1e-4)
    }

    func testHybridAffineK8PolarWHTValueFusedAttentionMatchesSplitReference() throws {
        guard TurboQuantKernelAvailability.current.attentionCapabilities.hybridK8PolarWHTValueAttention
        else {
            throw XCTSkip("hybrid K8 + PolarWHT-V attention is unavailable")
        }

        let tokenCount = 128
        let headDimension = 64
        let keys = MLXArray(
            makeWaveValues(
                count: tokenCount * headDimension,
                sinScale: 0.019,
                sinWeight: 0.17,
                cosScale: 0.013,
                cosWeight: 0.08
            ),
            [1, 1, tokenCount, headDimension]
        )
        let values = MLXArray(
            makeWaveValues(
                count: tokenCount * headDimension,
                sinScale: 0.031,
                sinWeight: -0.06,
                cosScale: 0.017,
                cosWeight: 0.19
            ),
            [1, 1, tokenCount, headDimension]
        )
        let queries = MLXArray(
            makeWaveValues(
                count: headDimension,
                sinScale: 0.047,
                sinWeight: 0.11,
                cosScale: 0.029,
                cosWeight: 0.04
            ),
            [1, 1, 1, headDimension]
        )
        let (keyWeight, keyScales, keyBiases) = quantized(
            keys,
            groupSize: 64,
            bits: 8,
            mode: .affine
        )
        guard let keyBiases else {
            XCTFail("affine quantization should return key biases")
            return
        }
        let valueCode = try turboQuantMetalPolarWHTEncodeAttentionValues(
            values,
            bits: 4,
            seed: 0xA77E_0000_0000_0722
        )
        let scale = 1 / sqrt(Float(headDimension))
        let scores = quantizedMM(
            queries * scale,
            keyWeight,
            scales: keyScales,
            biases: keyBiases,
            transpose: true,
            groupSize: 64,
            bits: 8,
            mode: .affine
        )
        let weights = softmax(scores.asType(.float32), axis: -1)
        let reference = try turboQuantMetalPolarWHTAV(
            attentionWeights: weights.contiguous(stream: .gpu),
            valueCode: valueCode,
            outputDType: queries.dtype
        )
        let fused = try turboQuantMetalHybridAffineK8PolarWHTValueScaledDotProductAttentionIfSupported(
            queries: queries,
            keyWeight: keyWeight.contiguous(stream: .gpu),
            keyScales: keyScales.contiguous(stream: .gpu),
            keyBiases: keyBiases.contiguous(stream: .gpu),
            keyGroupSize: 64,
            valueCode: valueCode,
            scale: scale,
            mask: .causal,
            outputDType: queries.dtype
        )
        guard let fused else {
            XCTFail("affine K8 + PolarWHT-V fused attention rejected supported test shape")
            return
        }

        eval(reference, fused)
        assertEqual(fused, reference, rtol: 5e-4, atol: 5e-4)
    }

    func testHybridAffineK8DecodedValueFusedAttentionMatchesMLXReference() throws {
        guard TurboQuantKernelAvailability.current.attentionCapabilities.hybridK8PolarWHTValueAttention
        else {
            throw XCTSkip("hybrid K8 attention kernels are unavailable")
        }

        let tokenCount = 1024
        let headDimension = 64
        let kvHeadCount = 2
        let queryHeadCount = 4
        let keys = MLXArray(
            makeWaveValues(
                count: kvHeadCount * tokenCount * headDimension,
                sinScale: 0.021,
                sinWeight: 0.15,
                cosScale: 0.017,
                cosWeight: 0.06
            ),
            [1, kvHeadCount, tokenCount, headDimension]
        )
        let values = MLXArray(
            makeWaveValues(
                count: kvHeadCount * tokenCount * headDimension,
                sinScale: 0.027,
                sinWeight: -0.04,
                cosScale: 0.019,
                cosWeight: 0.18
            ),
            [1, kvHeadCount, tokenCount, headDimension]
        )
        let queries = MLXArray(
            makeWaveValues(
                count: queryHeadCount * headDimension,
                sinScale: 0.043,
                sinWeight: 0.12,
                cosScale: 0.031,
                cosWeight: 0.05
            ),
            [1, queryHeadCount, 1, headDimension]
        )
        let (keyWeight, keyScales, keyBiases) = quantized(
            keys,
            groupSize: 64,
            bits: 8,
            mode: .affine
        )
        guard let keyBiases else {
            XCTFail("affine quantization should return key biases")
            return
        }
        let valueCode = try turboQuantMetalPolarWHTEncodeAttentionValues(
            values,
            bits: 4,
            seed: 0xA77E_0000_0000_0732
        )
        let decodedValues = try turboQuantMetalPolarWHTDecodeAttentionValues(
            valueCode,
            outputDType: .float32
        )
        let scale = 1 / sqrt(Float(headDimension))
        let repeats = queryHeadCount / kvHeadCount
        let groupedQueries = (queries * scale).reshaped([
            1, kvHeadCount, repeats, 1, headDimension,
        ])
        let groupedKeys = (
            expandedDimensions(keyWeight, axis: -3),
            expandedDimensions(keyScales, axis: -3),
            expandedDimensions(keyBiases, axis: -3)
        )
        let scores = quantizedMM(
            groupedQueries,
            groupedKeys.0,
            scales: groupedKeys.1,
            biases: groupedKeys.2,
            transpose: true,
            groupSize: 64,
            bits: 8,
            mode: .affine
        )
        let weights = softmax(scores.asType(.float32), axis: -1)
        let reference = matmul(
            weights,
            expandedDimensions(decodedValues, axis: 2)
        )
        .reshaped([1, queryHeadCount, 1, headDimension])
        let fused = try turboQuantMetalHybridAffineK8DecodedValueScaledDotProductAttentionIfSupported(
            queries: queries,
            keyWeight: keyWeight.contiguous(stream: .gpu),
            keyScales: keyScales.contiguous(stream: .gpu),
            keyBiases: keyBiases.contiguous(stream: .gpu),
            keyGroupSize: 64,
            decodedValues: decodedValues.contiguous(stream: .gpu),
            scale: scale,
            mask: .causal,
            outputDType: .float32
        )
        guard let fused else {
            XCTFail("affine K8 + decoded-value fused attention rejected supported test shape")
            return
        }

        eval(reference, fused)
        assertEqual(fused, reference, rtol: 5e-4, atol: 5e-4)
    }

    func testHybridAffineK8PolarWHTValueEncodeMatchesSplitEncoders() throws {
        guard TurboQuantKernelAvailability.current.attentionCapabilities.polarWHTCodec
        else {
            throw XCTSkip("PolarWHT codec is unavailable")
        }

        let tokenCount = 17
        let headDimension = 128
        let kvHeadCount = 2
        let keys = MLXArray(
            makeWaveValues(
                count: kvHeadCount * tokenCount * headDimension,
                sinScale: 0.023,
                sinWeight: 0.16,
                cosScale: 0.019,
                cosWeight: 0.09
            ),
            [1, kvHeadCount, tokenCount, headDimension]
        )
        let values = MLXArray(
            makeWaveValues(
                count: kvHeadCount * tokenCount * headDimension,
                sinScale: 0.031,
                sinWeight: -0.07,
                cosScale: 0.017,
                cosWeight: 0.20
            ),
            [1, kvHeadCount, tokenCount, headDimension]
        )
        let seed: UInt64 = 0xA77E_0000_0000_0752
        let fused = try turboQuantMetalHybridAffineK8PolarWHTValueEncode(
            keys: keys,
            values: values,
            keyGroupSize: 64,
            valueBits: 4,
            valueSeed: seed
        )
        let splitKey = quantized(
            keys,
            groupSize: 64,
            bits: 8,
            mode: .affine
        )
        let splitValue = try turboQuantMetalPolarWHTEncodeAttentionValues(
            values,
            bits: 4,
            seed: seed
        )
        guard let splitBiases = splitKey.biases, let fusedBiases = fused.key.biases else {
            XCTFail("affine K8 encode should produce biases")
            return
        }

        eval(
            fused.key.weight,
            fused.key.scales,
            fusedBiases,
            fused.value.packedIndices,
            fused.value.norms,
            splitKey.wq,
            splitKey.scales,
            splitBiases,
            splitValue.packedIndices,
            splitValue.norms
        )
        XCTAssertEqual(fused.key.weight.asArray(UInt32.self), splitKey.wq.asArray(UInt32.self))
        XCTAssertEqual(
            fused.value.packedIndices.asArray(UInt32.self),
            splitValue.packedIndices.asArray(UInt32.self)
        )
        assertEqual(fused.key.scales, splitKey.scales, rtol: 1e-5, atol: 1e-5)
        assertEqual(fusedBiases, splitBiases, rtol: 1e-5, atol: 1e-5)
        assertEqual(fused.value.norms, splitValue.norms, rtol: 1e-5, atol: 1e-5)
    }

    func testHybridAffineK8PolarWHTValueBlockParallelFusedMatchesSplitReference() throws {
        guard TurboQuantKernelAvailability.current.attentionCapabilities.hybridK8PolarWHTValueAttention
        else {
            throw XCTSkip("hybrid K8 + PolarWHT-V attention is unavailable")
        }

        let tokenCount = 4096
        let headDimension = 64
        let keys = MLXArray(
            makeWaveValues(
                count: tokenCount * headDimension,
                sinScale: 0.004,
                sinWeight: 0.16,
                cosScale: 0.006,
                cosWeight: 0.09
            ),
            [1, 1, tokenCount, headDimension]
        )
        let values = MLXArray(
            makeWaveValues(
                count: tokenCount * headDimension,
                sinScale: 0.008,
                sinWeight: -0.05,
                cosScale: 0.010,
                cosWeight: 0.18
            ),
            [1, 1, tokenCount, headDimension]
        )
        let queries = MLXArray(
            makeWaveValues(
                count: headDimension,
                sinScale: 0.043,
                sinWeight: 0.12,
                cosScale: 0.027,
                cosWeight: 0.03
            ),
            [1, 1, 1, headDimension]
        )
        let (keyWeight, keyScales, keyBiases) = quantized(
            keys,
            groupSize: 64,
            bits: 8,
            mode: .affine
        )
        guard let keyBiases else {
            XCTFail("affine quantization should return key biases")
            return
        }
        let valueCode = try turboQuantMetalPolarWHTEncodeAttentionValues(
            values,
            bits: 4,
            seed: 0xA77E_0000_0000_0732
        )
        let scale = 1 / sqrt(Float(headDimension))
        let scores = quantizedMM(
            queries * scale,
            keyWeight,
            scales: keyScales,
            biases: keyBiases,
            transpose: true,
            groupSize: 64,
            bits: 8,
            mode: .affine
        )
        let weights = softmax(scores.asType(.float32), axis: -1)
        let reference = try turboQuantMetalPolarWHTAV(
            attentionWeights: weights.contiguous(stream: .gpu),
            valueCode: valueCode,
            outputDType: queries.dtype
        )
        let fused = try turboQuantMetalHybridAffineK8PolarWHTValueScaledDotProductAttentionIfSupported(
            queries: queries,
            keyWeight: keyWeight.contiguous(stream: .gpu),
            keyScales: keyScales.contiguous(stream: .gpu),
            keyBiases: keyBiases.contiguous(stream: .gpu),
            keyGroupSize: 64,
            valueCode: valueCode,
            scale: scale,
            mask: .causal,
            outputDType: queries.dtype
        )
        guard let fused else {
            XCTFail("affine K8 + PolarWHT-V block fused attention rejected supported test shape")
            return
        }

        eval(reference, fused)
        assertEqual(fused, reference, rtol: 5e-4, atol: 5e-4)
    }

    func testHybridAffineK8PolarWHTValueGQAPairBlockParallelFusedMatchesSplitReference() throws {
        guard TurboQuantKernelAvailability.current.attentionCapabilities.hybridK8PolarWHTValueAttention
        else {
            throw XCTSkip("hybrid K8 + PolarWHT-V attention is unavailable")
        }

        let tokenCount = 4096
        let headDimension = 64
        let kvHeadCount = 2
        let queryHeadCount = 4
        let keys = MLXArray(
            makeWaveValues(
                count: kvHeadCount * tokenCount * headDimension,
                sinScale: 0.005,
                sinWeight: 0.13,
                cosScale: 0.007,
                cosWeight: 0.10
            ),
            [1, kvHeadCount, tokenCount, headDimension]
        )
        let values = MLXArray(
            makeWaveValues(
                count: kvHeadCount * tokenCount * headDimension,
                sinScale: 0.009,
                sinWeight: -0.04,
                cosScale: 0.011,
                cosWeight: 0.17
            ),
            [1, kvHeadCount, tokenCount, headDimension]
        )
        let queries = MLXArray(
            makeWaveValues(
                count: queryHeadCount * headDimension,
                sinScale: 0.041,
                sinWeight: 0.10,
                cosScale: 0.023,
                cosWeight: 0.05
            ),
            [1, queryHeadCount, 1, headDimension]
        )
        let (keyWeight, keyScales, keyBiases) = quantized(
            keys,
            groupSize: 64,
            bits: 8,
            mode: .affine
        )
        guard let keyBiases else {
            XCTFail("affine quantization should return key biases")
            return
        }
        let valueCode = try turboQuantMetalPolarWHTEncodeAttentionValues(
            values,
            bits: 4,
            seed: 0xA77E_0000_0000_0742
        )
        let scale = 1 / sqrt(Float(headDimension))
        let dequantizedKeys = dequantized(
            keyWeight,
            scales: keyScales,
            biases: keyBiases,
            groupSize: 64,
            bits: 8,
            mode: .affine,
            dtype: queries.dtype
        )
        let repeatedKeys = repeated(
            dequantizedKeys,
            count: queryHeadCount / kvHeadCount,
            axis: 1
        )
        let scores = (queries * scale).matmul(repeatedKeys.transposed(0, 1, 3, 2))
        let weights = softmax(scores.asType(.float32), axis: -1)
        let reference = try turboQuantMetalPolarWHTAV(
            attentionWeights: weights.contiguous(stream: .gpu),
            valueCode: valueCode,
            outputDType: queries.dtype
        )
        let fused = try turboQuantMetalHybridAffineK8PolarWHTValueScaledDotProductAttentionIfSupported(
            queries: queries,
            keyWeight: keyWeight.contiguous(stream: .gpu),
            keyScales: keyScales.contiguous(stream: .gpu),
            keyBiases: keyBiases.contiguous(stream: .gpu),
            keyGroupSize: 64,
            valueCode: valueCode,
            scale: scale,
            mask: .causal,
            outputDType: queries.dtype
        )
        guard let fused else {
            XCTFail("affine K8 + PolarWHT-V GQA block fused attention rejected supported test shape")
            return
        }

        eval(reference, fused)
        assertEqual(fused, reference, rtol: 5e-4, atol: 5e-4)
    }

    func testSegmentedHybridAffineK8PolarWHTValueMatchesFullResidentGQAPair() throws {
        guard ProcessInfo.processInfo.environment[
            "TURBOQUANT_ENABLE_HYBRID_POLARWHT_TAIL_TESTS"
        ] == "1" else {
            throw XCTSkip("segmented hybrid K8 + PolarWHT-V tail parity is experimental")
        }
        guard TurboQuantKernelAvailability.current.attentionCapabilities.hybridK8PolarWHTValueAttention
        else {
            throw XCTSkip("hybrid K8 + PolarWHT-V attention is unavailable")
        }

        let baseTokenCount = 1024
        let tailTokenCount = 8
        let headDimension = 128
        let kvHeadCount = 2
        let queryHeadCount = 4
        let baseKeys = MLXArray(
            makeWaveValues(
                count: kvHeadCount * baseTokenCount * headDimension,
                sinScale: 0.005,
                sinWeight: 0.13,
                cosScale: 0.007,
                cosWeight: 0.10
            ),
            [1, kvHeadCount, baseTokenCount, headDimension]
        )
        let tailKeys = MLXArray(
            makeWaveValues(
                count: kvHeadCount * tailTokenCount * headDimension,
                sinScale: 0.031,
                sinWeight: -0.07,
                cosScale: 0.017,
                cosWeight: 0.11
            ),
            [1, kvHeadCount, tailTokenCount, headDimension]
        )
        let baseValues = MLXArray(
            makeWaveValues(
                count: kvHeadCount * baseTokenCount * headDimension,
                sinScale: 0.009,
                sinWeight: -0.04,
                cosScale: 0.011,
                cosWeight: 0.17
            ),
            [1, kvHeadCount, baseTokenCount, headDimension]
        )
        let tailValues = MLXArray(
            makeWaveValues(
                count: kvHeadCount * tailTokenCount * headDimension,
                sinScale: 0.037,
                sinWeight: 0.08,
                cosScale: 0.019,
                cosWeight: -0.09
            ),
            [1, kvHeadCount, tailTokenCount, headDimension]
        )
        let keys = concatenated([baseKeys, tailKeys], axis: 2)
        let values = concatenated([baseValues, tailValues], axis: 2)
        let queries = MLXArray(
            makeWaveValues(
                count: queryHeadCount * headDimension,
                sinScale: 0.041,
                sinWeight: 0.10,
                cosScale: 0.023,
                cosWeight: 0.05
            ),
            [1, queryHeadCount, 1, headDimension]
        )

        let (fullKeyWeight, fullKeyScales, fullKeyBiases) = quantized(
            keys,
            groupSize: 64,
            bits: 8,
            mode: .affine
        )
        let (baseKeyWeight, baseKeyScales, baseKeyBiases) = quantized(
            baseKeys,
            groupSize: 64,
            bits: 8,
            mode: .affine
        )
        let (tailKeyWeight, tailKeyScales, tailKeyBiases) = quantized(
            tailKeys,
            groupSize: 64,
            bits: 8,
            mode: .affine
        )
        guard let fullKeyBiases, let baseKeyBiases, let tailKeyBiases else {
            XCTFail("affine quantization should return key biases")
            return
        }

        let valueSeed: UInt64 = 0xA77E_0000_0000_0752
        let fullValueCode = try turboQuantMetalPolarWHTEncodeAttentionValues(
            values,
            bits: 4,
            seed: valueSeed
        )
        let baseValueCode = try turboQuantMetalPolarWHTEncodeAttentionValues(
            baseValues,
            bits: 4,
            seed: valueSeed
        )
        let tailValueCode = try turboQuantMetalPolarWHTEncodeAttentionValues(
            tailValues,
            bits: 4,
            seed: valueSeed
        )
        let scale = 1 / sqrt(Float(headDimension))
        let full = try turboQuantMetalHybridAffineK8PolarWHTValueScaledDotProductAttentionIfSupported(
            queries: queries,
            keyWeight: fullKeyWeight.contiguous(stream: .gpu),
            keyScales: fullKeyScales.contiguous(stream: .gpu),
            keyBiases: fullKeyBiases.contiguous(stream: .gpu),
            keyGroupSize: 64,
            valueCode: fullValueCode,
            scale: scale,
            mask: .causal,
            outputDType: queries.dtype
        )
        let segmented =
            try turboQuantMetalSegmentedHybridAffineK8PolarWHTValueScaledDotProductAttentionIfSupported(
                queries: queries,
                baseKeyWeight: baseKeyWeight.contiguous(stream: .gpu),
                baseKeyScales: baseKeyScales.contiguous(stream: .gpu),
                baseKeyBiases: baseKeyBiases.contiguous(stream: .gpu),
                tailKeyWeight: tailKeyWeight.contiguous(stream: .gpu),
                tailKeyScales: tailKeyScales.contiguous(stream: .gpu),
                tailKeyBiases: tailKeyBiases.contiguous(stream: .gpu),
                keyGroupSize: 64,
                baseValueCode: baseValueCode,
                tailValueCode: tailValueCode,
                scale: scale,
                mask: .causal,
                outputDType: queries.dtype
            )
        guard let full, let segmented else {
            XCTFail("hybrid K8 + PolarWHT-V attention rejected supported segmented test shape")
            return
        }

        eval(full, segmented)
        assertEqual(segmented, full, rtol: 5e-4, atol: 5e-4)
    }

    func testNativeBlockParallelGQAMatchesSwiftMetalWhenEnabled() throws {
        _ = try requireNativeBackend()

        let tokenCount = 1024
        let headDimension = 64
        let queryHeadCount = 4
        let keys = MLXArray(
            makeWaveValues(
                count: tokenCount * headDimension,
                sinScale: 0.013,
                sinWeight: 0.16,
                cosScale: 0.019,
                cosWeight: 0.09
            ),
            [1, 1, tokenCount, headDimension]
        )
        let values = MLXArray(
            makeWaveValues(
                count: tokenCount * headDimension,
                sinScale: 0.029,
                sinWeight: -0.05,
                cosScale: 0.011,
                cosWeight: 0.23
            ),
            [1, 1, tokenCount, headDimension]
        )
        let queries = MLXArray(
            makeWaveValues(
                count: queryHeadCount * headDimension,
                sinScale: 0.047,
                sinWeight: 0.12,
                cosScale: 0.031,
                cosWeight: 0.04
            ),
            [1, queryHeadCount, 1, headDimension]
        )

        let keyCode = try turboQuantMetalEncodeAttention(
            keys,
            configuration: TurboQuantConfiguration(
                preset: .turbo4v2,
                role: .key,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 0xB10C_0000_0000_0101
            )
        )
        let valueCode = try turboQuantMetalEncodeAttention(
            values,
            configuration: TurboQuantConfiguration(
                preset: .turbo4v2,
                role: .value,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 0xB10C_0000_0000_0102,
                valueBits: 4
            )
        )
        let scale = 1 / sqrt(Float(headDimension))

        let swiftMetal = try swiftMetalTwoStageReference(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            scale: scale,
            mask: .causal
        )
        let native = try turboQuantNativeScaledDotProductAttentionWithDiagnostics(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            options: TurboQuantNativeAttentionOptions(
                scale: scale,
                causal: true,
                splitKBlockCount: 2,
                diagnostics: true
            )
        )

        eval(swiftMetal, native.output)
        assertEqual(native.output, swiftMetal, rtol: 1e-4, atol: 1e-4)
        XCTAssertEqual(native.diagnostics?.kernelKind, 3)
        XCTAssertEqual(native.diagnostics?.activeBlocks, 2)
        XCTAssertEqual(native.diagnostics?.blockTokens, 512)
    }

    func testNativeSparseVMatchesSwiftMetalWhenEnabled() throws {
        _ = try requireNativeBackend()

        let tokenCount = 32
        let headDimension = 64
        let queryHeadCount = 4
        let keys = MLXArray(
            makeWaveValues(
                count: tokenCount * headDimension,
                sinScale: 0.019,
                sinWeight: 0.18,
                cosScale: 0.007,
                cosWeight: 0.11
            ),
            [1, 1, tokenCount, headDimension]
        )
        let values = MLXArray(
            makeWaveValues(
                count: tokenCount * headDimension,
                sinScale: 0.037,
                sinWeight: -0.06,
                cosScale: 0.013,
                cosWeight: 0.21
            ),
            [1, 1, tokenCount, headDimension]
        )
        let queries = MLXArray(
            makeWaveValues(
                count: queryHeadCount * headDimension,
                sinScale: 0.053,
                sinWeight: 0.13,
                cosScale: 0.029,
                cosWeight: 0.02
            ),
            [1, queryHeadCount, 1, headDimension]
        )

        let keyCode = try turboQuantMetalEncodeAttention(
            keys,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .key,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 0x5A51_0000_0000_0101
            )
        )
        let valueCode = try turboQuantMetalEncodeAttention(
            values,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .value,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 0x5A51_0000_0000_0102,
                valueBits: 4
            )
        )
        let scale = 1 / sqrt(Float(headDimension))
        let threshold: Float = 0.04

        let scores = try turboQuantMetalQK(
            queries: queries,
            keyCode: keyCode,
            scale: scale,
            mask: .causal
        )
        var weights = softmax(scores.asType(.float32), axis: -1)
        weights = MLX.where(
            weights .>= threshold,
            weights,
            MLXArray.zeros(like: weights),
            stream: .gpu
        )
        let swiftMetal = try turboQuantMetalAV(
            attentionWeights: weights,
            valueCode: valueCode,
            outputDType: queries.dtype
        )
        let native = try turboQuantNativeScaledDotProductAttentionWithDiagnostics(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            options: TurboQuantNativeAttentionOptions(
                scale: scale,
                causal: true,
                sparseVThreshold: threshold,
                diagnostics: true
            )
        )

        eval(swiftMetal, native.output)
        assertEqual(native.output, swiftMetal, rtol: 1e-4, atol: 1e-4)
        XCTAssertEqual(native.diagnostics?.kernelKind, 5)
        XCTAssertEqual(native.diagnostics?.sparseTotalTokens, tokenCount * queryHeadCount)
        XCTAssertGreaterThan(native.diagnostics?.sparseSkippedTokens ?? 0, 0)
    }

    func testNativeSparseVDiagnosticsAndNonDiagnosticsOutputsMatch() throws {
        _ = try requireNativeBackend()

        let tokenCount = 32
        let queryHeadCount = 4
        let (queries, keyCode, valueCode, scale) = try makeSparseVNativeAttentionCase(
            tokenCount: tokenCount,
            queryHeadCount: queryHeadCount,
            keySeed: 0x5A51_0000_0000_0201,
            valueSeed: 0x5A51_0000_0000_0202
        )
        let threshold: Float = 0.04

        let plain = try turboQuantNativeScaledDotProductAttention(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            options: TurboQuantNativeAttentionOptions(
                scale: scale,
                causal: true,
                sparseVThreshold: threshold,
                diagnostics: false
            )
        )
        let diagnostic = try turboQuantNativeScaledDotProductAttentionWithDiagnostics(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            options: TurboQuantNativeAttentionOptions(
                scale: scale,
                causal: true,
                sparseVThreshold: threshold,
                diagnostics: true
            )
        )

        eval(plain, diagnostic.output)
        assertEqual(diagnostic.output, plain, rtol: 1e-4, atol: 1e-4)
        XCTAssertNotNil(diagnostic.diagnostics)
        XCTAssertEqual(diagnostic.diagnostics?.kernelKind, 5)
        XCTAssertEqual(diagnostic.diagnostics?.sparseTotalTokens, tokenCount * queryHeadCount)
        XCTAssertGreaterThan(diagnostic.diagnostics?.sparseSkippedTokens ?? 0, 0)
    }

    func testNativeSparseVForcedSplitKMatchesSingleBlockAndReportsFlags() throws {
        _ = try requireNativeBackend()

        let tokenCount = 1024
        let queryHeadCount = 4
        let (queries, keyCode, valueCode, scale) = try makeSparseVNativeAttentionCase(
            tokenCount: tokenCount,
            queryHeadCount: queryHeadCount,
            keySeed: 0x5A51_0000_0000_0301,
            valueSeed: 0x5A51_0000_0000_0302
        )
        let threshold: Float = 0.04
        let singleBlock = try turboQuantNativeScaledDotProductAttentionWithDiagnostics(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            options: TurboQuantNativeAttentionOptions(
                scale: scale,
                causal: true,
                sparseVThreshold: threshold,
                diagnostics: true
            )
        )
        let splitK = try turboQuantNativeScaledDotProductAttentionWithDiagnostics(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            options: TurboQuantNativeAttentionOptions(
                scale: scale,
                causal: true,
                splitKBlockCount: 2,
                sparseVThreshold: threshold,
                diagnostics: true
            )
        )

        eval(singleBlock.output, splitK.output)
        assertEqual(splitK.output, singleBlock.output, rtol: 1e-4, atol: 1e-4)
        XCTAssertGreaterThan(splitK.diagnostics?.activeBlocks ?? 0, 1)
        XCTAssertNotEqual((splitK.diagnostics?.flags ?? 0) & 2, 0)
        XCTAssertNotEqual((splitK.diagnostics?.flags ?? 0) & 16, 0)
        XCTAssertEqual(splitK.diagnostics?.sparseTotalTokens, tokenCount * queryHeadCount)
        XCTAssertGreaterThan(splitK.diagnostics?.sparseSkippedTokens ?? 0, 0)
    }

    func testNativeSparseVSplitKUsesCooperativeGQAWhenEligible() throws {
        _ = try requireNativeBackend()

        let tokenCount = 32_768
        let kvHeadCount = 4
        let queryHeadCount = 16
        let headDimension = 256
        func waveArray(
            count: Int,
            shape: [Int],
            sinScale: Float,
            sinWeight: Float,
            cosScale: Float,
            cosWeight: Float
        ) -> MLXArray {
            let positions = MLXArray.arange(count, dtype: .float32)
            return (sin(positions * sinScale) * sinWeight
                + cos(positions * cosScale) * cosWeight).reshaped(shape)
        }

        let keys = waveArray(
            count: kvHeadCount * tokenCount * headDimension,
            shape: [1, kvHeadCount, tokenCount, headDimension],
            sinScale: 0.00073,
            sinWeight: 0.16,
            cosScale: 0.0019,
            cosWeight: 0.09
        )
        let values = waveArray(
            count: kvHeadCount * tokenCount * headDimension,
            shape: [1, kvHeadCount, tokenCount, headDimension],
            sinScale: 0.0011,
            sinWeight: -0.05,
            cosScale: 0.00091,
            cosWeight: 0.23
        )
        let queries = waveArray(
            count: queryHeadCount * headDimension,
            shape: [1, queryHeadCount, 1, headDimension],
            sinScale: 0.013,
            sinWeight: 0.12,
            cosScale: 0.031,
            cosWeight: 0.04
        )

        let keyCode = try turboQuantMetalEncodeAttention(
            keys,
            configuration: TurboQuantConfiguration(
                preset: .turbo4v2,
                role: .key,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 0x5A51_0000_0000_0401,
                attentionLayoutVersion: 6,
                allowExperimentalLayoutV5: true
            )
        )
        let valueCode = try turboQuantMetalEncodeAttention(
            values,
            configuration: TurboQuantConfiguration(
                preset: .turbo4v2,
                role: .value,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 0x5A51_0000_0000_0402,
                valueBits: 4,
                attentionLayoutVersion: 6,
                allowExperimentalLayoutV5: true
            )
        )
        let scale = 1 / sqrt(Float(headDimension))
        let threshold: Float = 5e-5

        let scores = try turboQuantMetalQK(
            queries: queries,
            keyCode: keyCode,
            scale: scale,
            mask: .causal
        )
        var weights = softmax(scores.asType(.float32), axis: -1)
        weights = MLX.where(
            weights .>= threshold,
            weights,
            MLXArray.zeros(like: weights),
            stream: .gpu
        )
        let swiftMetal = try turboQuantMetalAV(
            attentionWeights: weights,
            valueCode: valueCode,
            outputDType: queries.dtype
        )
        let native = try turboQuantNativeScaledDotProductAttentionWithDiagnostics(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            options: TurboQuantNativeAttentionOptions(
                scale: scale,
                causal: true,
                sparseVThreshold: threshold,
                diagnostics: true
            )
        )

        eval(swiftMetal, native.output)
        assertEqual(native.output, swiftMetal, rtol: 1e-3, atol: 1e-3)
        XCTAssertEqual(native.diagnostics?.kernelKind, 8)
        XCTAssertGreaterThan(native.diagnostics?.activeBlocks ?? 0, 1)
        XCTAssertEqual(native.diagnostics?.blockTokens, 512)
        XCTAssertNotEqual((native.diagnostics?.flags ?? 0) & 2, 0)
        XCTAssertNotEqual((native.diagnostics?.flags ?? 0) & 4, 0)
        XCTAssertNotEqual((native.diagnostics?.flags ?? 0) & 8, 0)
        XCTAssertNotEqual((native.diagnostics?.flags ?? 0) & 16, 0)
        XCTAssertEqual(native.diagnostics?.sparseTotalTokens, tokenCount * queryHeadCount)
        XCTAssertGreaterThan(native.diagnostics?.sparseSkippedTokens ?? 0, 0)
    }

    func testNativeSparseVTopKSplitKMatchesMaskedAVReference() throws {
        _ = try requireNativeBackend()

        let tokenCount = 1024
        let queryHeadCount = 4
        let topK = 64
        let (queries, keyCode, valueCode, scale) = try makeSparseVNativeAttentionCase(
            tokenCount: tokenCount,
            queryHeadCount: queryHeadCount,
            keySeed: 0x5A51_0000_0000_0501,
            valueSeed: 0x5A51_0000_0000_0502
        )

        let reference = try sparseReferenceOutput(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            scale: scale,
            selection: .topK(topK)
        )
        let native = try turboQuantNativeScaledDotProductAttentionWithDiagnostics(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            options: TurboQuantNativeAttentionOptions(
                scale: scale,
                causal: true,
                splitKBlockCount: 2,
                sparseVSelectionMode: .topK,
                sparseVTopK: topK,
                diagnostics: true
            )
        )

        eval(reference, native.output)
        assertEqual(native.output, reference, rtol: 1e-3, atol: 1e-3)
        XCTAssertEqual(native.diagnostics?.kernelKind, 12)
        XCTAssertEqual(native.diagnostics?.activeBlocks, 2)
        XCTAssertEqual(native.diagnostics?.sparseTotalTokens, tokenCount * queryHeadCount)
        XCTAssertEqual(native.diagnostics?.sparseSkippedTokens, (tokenCount - topK) * queryHeadCount)
    }

    func testNativeSparseVTopKCandidateCompactMatchesMaskedAVReference() throws {
        _ = try requireNativeBackend()

        let tokenCount = 8192
        let queryHeadCount = 4
        let topK = 128
        let (queries, keyCode, valueCode, scale) = try makeSparseVNativeAttentionCase(
            tokenCount: tokenCount,
            queryHeadCount: queryHeadCount,
            keySeed: 0x5A51_0000_0000_0521,
            valueSeed: 0x5A51_0000_0000_0522
        )

        let reference = try sparseReferenceOutput(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            scale: scale,
            selection: .topK(topK)
        )
        let native = try turboQuantNativeScaledDotProductAttentionWithDiagnostics(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            options: TurboQuantNativeAttentionOptions(
                scale: scale,
                causal: true,
                sparseVSelectionMode: .topK,
                sparseVTopK: topK,
                diagnostics: true
            )
        )

        eval(reference, native.output)
        assertEqual(native.output, reference, rtol: 1e-3, atol: 1e-3)
        XCTAssertEqual(native.diagnostics?.kernelKind, 12)
        XCTAssertEqual(native.diagnostics?.activeBlocks, 16)
        XCTAssertEqual(native.diagnostics?.sparseTotalTokens, tokenCount * queryHeadCount)
        XCTAssertEqual(native.diagnostics?.sparseSkippedTokens, (tokenCount - topK) * queryHeadCount)
    }

    func testNativeSparseVTopKSplitKCompactTieKeepsLowerTokens() throws {
        _ = try requireNativeBackend()

        let tokenCount = 1024
        let queryHeadCount = 4
        let headDimension = 64
        let topK = 64
        let keys = MLXArray.zeros([1, 1, tokenCount, headDimension])
        let values = MLXArray(
            makeWaveValues(
                count: tokenCount * headDimension,
                sinScale: 0.037,
                sinWeight: -0.06,
                cosScale: 0.013,
                cosWeight: 0.21
            ),
            [1, 1, tokenCount, headDimension]
        )
        let queries = MLXArray.zeros([1, queryHeadCount, 1, headDimension])
        let keyCode = try turboQuantMetalEncodeAttention(
            keys,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .key,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 0x5A51_0000_0000_0541
            )
        )
        let valueCode = try turboQuantMetalEncodeAttention(
            values,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .value,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 0x5A51_0000_0000_0542,
                valueBits: 4
            )
        )
        var selectedWeights = Array(repeating: Float(0), count: queryHeadCount * tokenCount)
        for row in 0 ..< queryHeadCount {
            for token in 0 ..< topK {
                selectedWeights[row * tokenCount + token] = 1 / Float(tokenCount)
            }
        }
        let referenceWeights = MLXArray(
            selectedWeights,
            [1, queryHeadCount, 1, tokenCount]
        )
        let reference = try turboQuantMetalAV(
            attentionWeights: referenceWeights,
            valueCode: valueCode,
            outputDType: queries.dtype
        )
        let native = try turboQuantNativeScaledDotProductAttentionWithDiagnostics(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            options: TurboQuantNativeAttentionOptions(
                scale: 1 / sqrt(Float(headDimension)),
                causal: true,
                sparseVSelectionMode: .topK,
                sparseVTopK: topK,
                diagnostics: true
            )
        )

        eval(reference, native.output)
        assertEqual(native.output, reference, rtol: 1e-3, atol: 1e-3)
        XCTAssertEqual(native.diagnostics?.kernelKind, 12)
        XCTAssertEqual(native.diagnostics?.activeBlocks, 2)
        XCTAssertEqual(native.diagnostics?.sparseSkippedTokens, (tokenCount - topK) * queryHeadCount)
    }

    func testNativeSparseVPageTopKPrototypeKeepsLowerPageOnTie() throws {
        _ = try requireNativeBackend()

        let tokenCount = 1024
        let headDimension = 64
        let pageTopK = 1
        let keys = MLXArray.zeros([1, 1, tokenCount, headDimension])
        let values = MLXArray(
            makeWaveValues(
                count: tokenCount * headDimension,
                sinScale: 0.037,
                sinWeight: -0.06,
                cosScale: 0.013,
                cosWeight: 0.21
            ),
            [1, 1, tokenCount, headDimension]
        )
        let queries = MLXArray.zeros([1, 1, 1, headDimension])
        let keyCode = try turboQuantMetalEncodeAttention(
            keys,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .key,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 0x5A51_0000_0000_0561
            )
        )
        let valueCode = try turboQuantMetalEncodeAttention(
            values,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .value,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 0x5A51_0000_0000_0562,
                valueBits: 4
            )
        )
        let reference = try pageTopKDecodedMeanReference(
            valueCode: valueCode,
            queryHeadCount: 1,
            retainedTokenCount: 512
        )
        let native = try turboQuantNativeScaledDotProductAttentionWithDiagnostics(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            options: TurboQuantNativeAttentionOptions(
                scale: 1 / sqrt(Float(headDimension)),
                causal: true,
                sparseVSelectionMode: .pageTopK,
                sparseVTopK: pageTopK,
                diagnostics: true
            )
        )

        eval(reference, native.output)
        assertEqual(native.output, reference, rtol: 1e-3, atol: 1e-3)
        XCTAssertEqual(native.diagnostics?.kernelKind, 13)
        XCTAssertEqual(native.diagnostics?.activeBlocks, 2)
        XCTAssertEqual(native.diagnostics?.blockTokens, 512)
        XCTAssertEqual(native.diagnostics?.sparseTotalTokens, tokenCount)
        XCTAssertEqual(native.diagnostics?.sparseSkippedTokens, 512)
    }

    func testNativeCandidateSparsePrototypeKeepsRecentAndLowerOlderTokensOnTie() throws {
        _ = try requireNativeBackend()

        let tokenCount = 1024
        let headDimension = 64
        let recentTokens = 512
        let olderTopK = 128
        let candidatePages = 1
        let keys = MLXArray.zeros([1, 1, tokenCount, headDimension])
        let values = MLXArray(
            makeWaveValues(
                count: tokenCount * headDimension,
                sinScale: 0.031,
                sinWeight: 0.11,
                cosScale: 0.017,
                cosWeight: -0.19
            ),
            [1, 1, tokenCount, headDimension]
        )
        let queries = MLXArray.zeros([1, 1, 1, headDimension])
        let keyCode = try turboQuantMetalEncodeAttention(
            keys,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .key,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 0x5A51_0000_0000_05B1
            )
        )
        let valueCode = try turboQuantMetalEncodeAttention(
            values,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .value,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 0x5A51_0000_0000_05B2,
                valueBits: 4
            )
        )
        let keyCandidateSketch = MLXArray.zeros([1, 1, 2, 64], dtype: .float32)
        let reference = try pageTopKDecodedMeanReference(
            valueCode: valueCode,
            queryHeadCount: 1,
            retainedRanges: [0 ..< olderTopK, (tokenCount - recentTokens) ..< tokenCount]
        )
        let native = try turboQuantNativeScaledDotProductAttentionWithDiagnostics(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            options: TurboQuantNativeAttentionOptions(
                scale: 1 / sqrt(Float(headDimension)),
                causal: true,
                sparseVSelectionMode: .candidateSparse,
                sparseVTopK: olderTopK,
                sparseVRecentTokens: recentTokens,
                sparseVCandidatePages: candidatePages,
                diagnostics: true
            ),
            keyCandidateSketch: keyCandidateSketch
        )

        eval(reference, native.output)
        assertEqual(native.output, reference, rtol: 1e-3, atol: 1e-3)
        XCTAssertEqual(native.diagnostics?.kernelKind, 19)
        XCTAssertEqual(native.diagnostics?.activeBlocks, 2)
        XCTAssertEqual(native.diagnostics?.blockTokens, 512)
        XCTAssertEqual(native.diagnostics?.fallbackCode, 0)
        XCTAssertEqual(native.diagnostics?.recentTokens, recentTokens)
        XCTAssertEqual(native.diagnostics?.selectedOlderTokens, olderTopK)
        XCTAssertEqual(native.diagnostics?.selectedPages, candidatePages)
        XCTAssertEqual(native.diagnostics?.candidatePagesConsidered, 1)
        XCTAssertEqual(native.diagnostics?.candidateTokensConsidered, 512)
        XCTAssertEqual(native.diagnostics?.retainedTokens, recentTokens + olderTopK)
        XCTAssertEqual(native.diagnostics?.sparseSkippedTokens, tokenCount - recentTokens - olderTopK)
    }

    func testNativeCandidateSparseRetainAllCandidatePageMatchesReference() throws {
        _ = try requireNativeBackend()

        let tokenCount = 1024
        let headDimension = 64
        let recentTokens = 256
        let olderTopK = 512
        let candidatePages = 1
        let keys = MLXArray.zeros([1, 1, tokenCount, headDimension])
        let values = MLXArray(
            makeWaveValues(
                count: tokenCount * headDimension,
                sinScale: 0.029,
                sinWeight: -0.08,
                cosScale: 0.019,
                cosWeight: 0.17
            ),
            [1, 1, tokenCount, headDimension]
        )
        let queries = MLXArray.zeros([1, 1, 1, headDimension])
        let keyCode = try turboQuantMetalEncodeAttention(
            keys,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .key,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 0x5A51_0000_0000_05C1
            )
        )
        let valueCode = try turboQuantMetalEncodeAttention(
            values,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .value,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 0x5A51_0000_0000_05C2,
                valueBits: 4
            )
        )
        let keyCandidateSketch = MLXArray.zeros([1, 1, 2, 64], dtype: .float32)
        let reference = try pageTopKDecodedMeanReference(
            valueCode: valueCode,
            queryHeadCount: 1,
            retainedRanges: [0 ..< 512, (tokenCount - recentTokens) ..< tokenCount]
        )
        let native = try turboQuantNativeScaledDotProductAttentionWithDiagnostics(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            options: TurboQuantNativeAttentionOptions(
                scale: 1 / sqrt(Float(headDimension)),
                causal: true,
                sparseVSelectionMode: .candidateSparse,
                sparseVTopK: olderTopK,
                sparseVRecentTokens: recentTokens,
                sparseVCandidatePages: candidatePages,
                diagnostics: true
            ),
            keyCandidateSketch: keyCandidateSketch
        )

        eval(reference, native.output)
        assertEqual(native.output, reference, rtol: 1e-3, atol: 1e-3)
        XCTAssertEqual(native.diagnostics?.kernelKind, 19)
        XCTAssertEqual(native.diagnostics?.fallbackCode, 0)
        XCTAssertEqual(native.diagnostics?.recentTokens, recentTokens)
        XCTAssertEqual(native.diagnostics?.selectedOlderTokens, olderTopK)
        XCTAssertEqual(native.diagnostics?.selectedPages, candidatePages)
        XCTAssertEqual(native.diagnostics?.candidateTokensConsidered, 512)
        XCTAssertEqual(native.diagnostics?.retainedTokens, recentTokens + olderTopK)
        XCTAssertEqual(native.diagnostics?.sparseSkippedTokens, tokenCount - recentTokens - olderTopK)
    }

    func testNativeCandidateSparseCooperativeGQAKernelRequiresOptIn() throws {
        _ = try requireNativeBackend()

        let tokenCount = 1024
        let headDimension = 64
        let queryHeadCount = 4
        let recentTokens = 512
        let olderTopK = 128
        let candidatePages = 1
        let keys = MLXArray.zeros([1, 1, tokenCount, headDimension])
        let values = MLXArray(
            makeWaveValues(
                count: tokenCount * headDimension,
                sinScale: 0.033,
                sinWeight: 0.09,
                cosScale: 0.021,
                cosWeight: -0.15
            ),
            [1, 1, tokenCount, headDimension]
        )
        let queries = MLXArray.zeros([1, queryHeadCount, 1, headDimension])
        let keyCode = try turboQuantMetalEncodeAttention(
            keys,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .key,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 0x5A51_0000_0000_05D1
            )
        )
        let valueCode = try turboQuantMetalEncodeAttention(
            values,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .value,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 0x5A51_0000_0000_05D2,
                valueBits: 4
            )
        )
        let keyCandidateSketch = MLXArray.zeros([1, 1, 2, 64], dtype: .float32)
        let reference = try pageTopKDecodedMeanReference(
            valueCode: valueCode,
            queryHeadCount: queryHeadCount,
            retainedRanges: [0 ..< olderTopK, (tokenCount - recentTokens) ..< tokenCount]
        )

        let defaultNative = try turboQuantNativeScaledDotProductAttentionWithDiagnostics(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            options: TurboQuantNativeAttentionOptions(
                scale: 1 / sqrt(Float(headDimension)),
                causal: true,
                sparseVSelectionMode: .candidateSparse,
                sparseVTopK: olderTopK,
                sparseVRecentTokens: recentTokens,
                sparseVCandidatePages: candidatePages,
                diagnostics: true
            ),
            keyCandidateSketch: keyCandidateSketch
        )

        eval(reference, defaultNative.output)
        assertEqual(defaultNative.output, reference, rtol: 1e-3, atol: 1e-3)
        XCTAssertEqual(defaultNative.diagnostics?.kernelKind, 19)

        try withEnvironment("TURBOQUANT_CANDIDATE_SPARSE_FUSED", value: "1") {
            let native = try turboQuantNativeScaledDotProductAttentionWithDiagnostics(
                queries: queries,
                keyCode: keyCode,
                valueCode: valueCode,
                options: TurboQuantNativeAttentionOptions(
                    scale: 1 / sqrt(Float(headDimension)),
                    causal: true,
                    sparseVSelectionMode: .candidateSparse,
                    sparseVTopK: olderTopK,
                    sparseVRecentTokens: recentTokens,
                    sparseVCandidatePages: candidatePages,
                    diagnostics: true
                ),
                keyCandidateSketch: keyCandidateSketch
            )

            eval(reference, native.output)
            assertEqual(native.output, reference, rtol: 1e-3, atol: 1e-3)
            XCTAssertEqual(native.diagnostics?.kernelKind, 20)
            XCTAssertEqual(native.diagnostics?.activeBlocks, 2)
            XCTAssertEqual(native.diagnostics?.blockTokens, 512)
            XCTAssertEqual(native.diagnostics?.fallbackCode, 0)
            XCTAssertEqual(native.diagnostics?.recentTokens, recentTokens * queryHeadCount)
            XCTAssertEqual(native.diagnostics?.selectedOlderTokens, olderTopK * queryHeadCount)
            XCTAssertEqual(native.diagnostics?.selectedPages, candidatePages * queryHeadCount)
            XCTAssertEqual(native.diagnostics?.candidatePagesConsidered, 1 * queryHeadCount)
            XCTAssertEqual(native.diagnostics?.candidateTokensConsidered, 512 * queryHeadCount)
            XCTAssertEqual(native.diagnostics?.retainedTokens, (recentTokens + olderTopK) * queryHeadCount)
            XCTAssertEqual(
                native.diagnostics?.sparseSkippedTokens,
                (tokenCount - recentTokens - olderTopK) * queryHeadCount
            )
        }

        try withEnvironment("TURBOQUANT_CANDIDATE_SPARSE_FUSED", value: "0") {
            let fallback = try turboQuantNativeScaledDotProductAttentionWithDiagnostics(
                queries: queries,
                keyCode: keyCode,
                valueCode: valueCode,
                options: TurboQuantNativeAttentionOptions(
                    scale: 1 / sqrt(Float(headDimension)),
                    causal: true,
                    sparseVSelectionMode: .candidateSparse,
                    sparseVTopK: olderTopK,
                    sparseVRecentTokens: recentTokens,
                    sparseVCandidatePages: candidatePages,
                    diagnostics: true
                ),
                keyCandidateSketch: keyCandidateSketch
            )
            eval(fallback.output)
            assertEqual(fallback.output, reference, rtol: 1e-3, atol: 1e-3)
            XCTAssertEqual(fallback.diagnostics?.kernelKind, 19)
        }
    }

    func testNativeSparseVPageTopKPrototypeSupportsGQA() throws {
        _ = try requireNativeBackend()

        let tokenCount = 1024
        let queryHeadCount = 4
        let headDimension = 64
        let pageTopK = 1
        let keys = MLXArray.zeros([1, 1, tokenCount, headDimension])
        let values = MLXArray(
            makeWaveValues(
                count: tokenCount * headDimension,
                sinScale: 0.037,
                sinWeight: -0.06,
                cosScale: 0.013,
                cosWeight: 0.21
            ),
            [1, 1, tokenCount, headDimension]
        )
        let queries = MLXArray.zeros([1, queryHeadCount, 1, headDimension])
        let keyCode = try turboQuantMetalEncodeAttention(
            keys,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .key,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 0x5A51_0000_0000_0571
            )
        )
        let valueCode = try turboQuantMetalEncodeAttention(
            values,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .value,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 0x5A51_0000_0000_0572,
                valueBits: 4
            )
        )
        let summary = try turboQuantKeyPageSummaries(keyCode: keyCode)
        let reference = try pageTopKDecodedMeanReference(
            valueCode: valueCode,
            queryHeadCount: queryHeadCount,
            retainedTokenCount: 512
        )
        let native = try turboQuantNativeScaledDotProductAttentionWithDiagnostics(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            options: TurboQuantNativeAttentionOptions(
                scale: 1 / sqrt(Float(headDimension)),
                causal: true,
                sparseVSelectionMode: .pageTopK,
                sparseVTopK: pageTopK,
                diagnostics: true
            ),
            keyPageSummary: summary
        )

        eval(reference, native.output)
        assertEqual(native.output, reference, rtol: 1e-3, atol: 1e-3)
        XCTAssertEqual(native.diagnostics?.kernelKind, 15)
        XCTAssertEqual(native.diagnostics?.activeBlocks, 2)
        XCTAssertEqual(native.diagnostics?.blockTokens, 512)
        XCTAssertEqual(native.diagnostics?.sparseTotalTokens, tokenCount * queryHeadCount)
        XCTAssertEqual(native.diagnostics?.sparseSkippedTokens, 512 * queryHeadCount)
    }

    func testKeyPageSummariesMatchScaleMaxReference() throws {
        _ = try requireNativeBackend()

        let tokenCount = 1025
        let headDimension = 64
        let keys = MLXArray(
            makeWaveValues(
                count: tokenCount * headDimension,
                sinScale: 0.023,
                sinWeight: 0.17,
                cosScale: 0.011,
                cosWeight: -0.08
            ),
            [1, 1, tokenCount, headDimension]
        )
        let keyCode = try turboQuantMetalEncodeAttention(
            keys,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .key,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 0x5A51_0000_0000_0581
            )
        )
        let summary = try turboQuantKeyPageSummaries(keyCode: keyCode)
        eval(summary)
        XCTAssertEqual(summary.shape, [1, 1, 3, 1])

        let scales = keyCode.scales.asArray(Float.self)
        let actual = summary.asArray(Float.self)
        let capacity = keyCode.layout.capacity
        let groups = keyCode.layout.groupsPerVector
        let scalesPerGroup = keyCode.scalesPerGroup
        for page in 0 ..< 3 {
            var expected: Float = 0
            let start = page * turboQuantKeyPageSummaryPageSize
            let end = min(tokenCount, start + turboQuantKeyPageSummaryPageSize)
            if start < end {
                for token in start ..< end {
                    let base = ((token * groups) * scalesPerGroup)
                    expected = max(expected, abs(scales[base]) + abs(scales[base + 1]))
                }
            }
            XCTAssertEqual(actual[page], expected, accuracy: 1e-6)
        }
        XCTAssertEqual(capacity, tokenCount)
    }

    func testNativeSparseVPageTopKCachedSummaryUsesFusedKernel() throws {
        _ = try requireNativeBackend()

        let tokenCount = 1024
        let headDimension = 64
        let pageTopK = 1
        let keys = MLXArray.zeros([1, 1, tokenCount, headDimension])
        let values = MLXArray(
            makeWaveValues(
                count: tokenCount * headDimension,
                sinScale: 0.037,
                sinWeight: -0.06,
                cosScale: 0.013,
                cosWeight: 0.21
            ),
            [1, 1, tokenCount, headDimension]
        )
        let queries = MLXArray.zeros([1, 1, 1, headDimension])
        let keyCode = try turboQuantMetalEncodeAttention(
            keys,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .key,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 0x5A51_0000_0000_0591
            )
        )
        let valueCode = try turboQuantMetalEncodeAttention(
            values,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .value,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 0x5A51_0000_0000_0592,
                valueBits: 4
            )
        )
        let summary = try turboQuantKeyPageSummaries(keyCode: keyCode)
        let reference = try pageTopKDecodedMeanReference(
            valueCode: valueCode,
            queryHeadCount: 1,
            retainedTokenCount: 512
        )
        let native = try turboQuantNativeScaledDotProductAttentionWithDiagnostics(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            options: TurboQuantNativeAttentionOptions(
                scale: 1 / sqrt(Float(headDimension)),
                causal: true,
                sparseVSelectionMode: .pageTopK,
                sparseVTopK: pageTopK,
                diagnostics: true
            ),
            keyPageSummary: summary
        )

        eval(reference, native.output)
        assertEqual(native.output, reference, rtol: 1e-3, atol: 1e-3)
        XCTAssertEqual(native.diagnostics?.kernelKind, 15)
        XCTAssertEqual(native.diagnostics?.activeBlocks, 2)
        XCTAssertEqual(native.diagnostics?.sparseTotalTokens, tokenCount)
        XCTAssertEqual(native.diagnostics?.sparseSkippedTokens, 512)
    }

    func testNativeSparseVPageTopKRecentFloorKeepsRecentWindow() throws {
        _ = try requireNativeBackend()

        try withEnvironment("TURBOQUANT_SPARSE_V_PAGE_RECENT_TOKENS", value: "512") {
            let tokenCount = 1536
            let headDimension = 64
            let pageTopK = 1
            let keys = MLXArray.zeros([1, 1, tokenCount, headDimension])
            let values = MLXArray(
                makeWaveValues(
                    count: tokenCount * headDimension,
                    sinScale: 0.037,
                    sinWeight: -0.06,
                    cosScale: 0.013,
                    cosWeight: 0.21
                ),
                [1, 1, tokenCount, headDimension]
            )
            let queries = MLXArray.zeros([1, 1, 1, headDimension])
            let keyCode = try turboQuantMetalEncodeAttention(
                keys,
                configuration: TurboQuantConfiguration(
                    preset: .turbo3_5,
                    role: .key,
                    groupSize: 64,
                    backend: .metalPolarQJL,
                    seed: 0x5A51_0000_0000_0591
                )
            )
            let valueCode = try turboQuantMetalEncodeAttention(
                values,
                configuration: TurboQuantConfiguration(
                    preset: .turbo3_5,
                    role: .value,
                    groupSize: 64,
                    backend: .metalPolarQJL,
                    seed: 0x5A51_0000_0000_0592,
                    valueBits: 4
                )
            )
            let summary = try turboQuantKeyPageSummaries(keyCode: keyCode)
            let reference = try pageTopKDecodedMeanReference(
                valueCode: valueCode,
                queryHeadCount: 1,
                retainedRanges: [0 ..< 512, 1024 ..< 1536]
            )
            let native = try turboQuantNativeScaledDotProductAttentionWithDiagnostics(
                queries: queries,
                keyCode: keyCode,
                valueCode: valueCode,
                options: TurboQuantNativeAttentionOptions(
                    scale: 1 / sqrt(Float(headDimension)),
                    causal: true,
                    sparseVSelectionMode: .pageTopK,
                    sparseVTopK: pageTopK,
                    diagnostics: true
                ),
                keyPageSummary: summary
            )

            eval(reference, native.output)
            assertEqual(native.output, reference, rtol: 1e-3, atol: 1e-3)
            XCTAssertEqual(native.diagnostics?.kernelKind, 18)
            XCTAssertEqual(native.diagnostics?.activeBlocks, 3)
            XCTAssertEqual(native.diagnostics?.sparseTotalTokens, tokenCount)
            XCTAssertEqual(native.diagnostics?.sparseSkippedTokens, 512)
        }
    }

    func testNativeSparseVPageTopKCachedSummaryFusedDisableFallsBackToKernel14() throws {
        _ = try requireNativeBackend()

        try withEnvironment("TURBOQUANT_SPARSE_V_PAGE_FUSED", value: "0") {
            let tokenCount = 1024
            let headDimension = 64
            let keys = MLXArray.zeros([1, 1, tokenCount, headDimension])
            let values = MLXArray(
                makeWaveValues(
                    count: tokenCount * headDimension,
                    sinScale: 0.037,
                    sinWeight: -0.06,
                    cosScale: 0.013,
                    cosWeight: 0.21
                ),
                [1, 1, tokenCount, headDimension]
            )
            let queries = MLXArray.zeros([1, 1, 1, headDimension])
            let keyCode = try turboQuantMetalEncodeAttention(
                keys,
                configuration: TurboQuantConfiguration(
                    preset: .turbo3_5,
                    role: .key,
                    groupSize: 64,
                    backend: .metalPolarQJL,
                    seed: 0x5A51_0000_0000_0593
                )
            )
            let valueCode = try turboQuantMetalEncodeAttention(
                values,
                configuration: TurboQuantConfiguration(
                    preset: .turbo3_5,
                    role: .value,
                    groupSize: 64,
                    backend: .metalPolarQJL,
                    seed: 0x5A51_0000_0000_0594,
                    valueBits: 4
                )
            )
            let summary = try turboQuantKeyPageSummaries(keyCode: keyCode)
            let native = try turboQuantNativeScaledDotProductAttentionWithDiagnostics(
                queries: queries,
                keyCode: keyCode,
                valueCode: valueCode,
                options: TurboQuantNativeAttentionOptions(
                    scale: 1 / sqrt(Float(headDimension)),
                    causal: true,
                    sparseVSelectionMode: .pageTopK,
                    sparseVTopK: 1,
                    diagnostics: true
                ),
                keyPageSummary: summary
            )

            eval(native.output)
            XCTAssertEqual(native.diagnostics?.kernelKind, 14)
            XCTAssertEqual(native.diagnostics?.activeBlocks, 2)
            XCTAssertEqual(native.diagnostics?.sparseSkippedTokens, 512)
        }
    }

    func testNativeSparseVPageTopKCachedSummaryTopKAboveFusedLimitUsesKernel14() throws {
        _ = try requireNativeBackend()

        let tokenCount = 1024
        let headDimension = 64
        let keys = MLXArray.zeros([1, 1, tokenCount, headDimension])
        let values = MLXArray(
            makeWaveValues(
                count: tokenCount * headDimension,
                sinScale: 0.037,
                sinWeight: -0.06,
                cosScale: 0.013,
                cosWeight: 0.21
            ),
            [1, 1, tokenCount, headDimension]
        )
        let queries = MLXArray.zeros([1, 1, 1, headDimension])
        let keyCode = try turboQuantMetalEncodeAttention(
            keys,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .key,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 0x5A51_0000_0000_0595
            )
        )
        let valueCode = try turboQuantMetalEncodeAttention(
            values,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .value,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 0x5A51_0000_0000_0596,
                valueBits: 4
            )
        )
        let summary = try turboQuantKeyPageSummaries(keyCode: keyCode)
        let reference = try pageTopKDecodedMeanReference(
            valueCode: valueCode,
            queryHeadCount: 1,
            retainedTokenCount: tokenCount
        )
        let native = try turboQuantNativeScaledDotProductAttentionWithDiagnostics(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            options: TurboQuantNativeAttentionOptions(
                scale: 1 / sqrt(Float(headDimension)),
                causal: true,
                sparseVSelectionMode: .pageTopK,
                sparseVTopK: 9,
                diagnostics: true
            ),
            keyPageSummary: summary
        )

        eval(reference, native.output)
        assertEqual(native.output, reference, rtol: 1e-3, atol: 1e-3)
        XCTAssertEqual(native.diagnostics?.kernelKind, 14)
        XCTAssertEqual(native.diagnostics?.activeBlocks, 2)
        XCTAssertEqual(native.diagnostics?.sparseSkippedTokens, 0)
    }

    func testNativeSparseVPageTopKInvalidSummaryFallsBackToSampledKernel() throws {
        _ = try requireNativeBackend()

        let tokenCount = 1024
        let headDimension = 64
        let keys = MLXArray.zeros([1, 1, tokenCount, headDimension])
        let values = MLXArray(
            makeWaveValues(
                count: tokenCount * headDimension,
                sinScale: 0.037,
                sinWeight: -0.06,
                cosScale: 0.013,
                cosWeight: 0.21
            ),
            [1, 1, tokenCount, headDimension]
        )
        let queries = MLXArray.zeros([1, 1, 1, headDimension])
        let keyCode = try turboQuantMetalEncodeAttention(
            keys,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .key,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 0x5A51_0000_0000_05A1
            )
        )
        let valueCode = try turboQuantMetalEncodeAttention(
            values,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .value,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 0x5A51_0000_0000_05A2,
                valueBits: 4
            )
        )
        let native = try turboQuantNativeScaledDotProductAttentionWithDiagnostics(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            options: TurboQuantNativeAttentionOptions(
                scale: 1 / sqrt(Float(headDimension)),
                causal: true,
                sparseVSelectionMode: .pageTopK,
                sparseVTopK: 1,
                diagnostics: true
            ),
            keyPageSummary: MLXArray.zeros([1], dtype: .float32)
        )

        eval(native.output)
        XCTAssertEqual(native.diagnostics?.kernelKind, 13)
        XCTAssertEqual(native.diagnostics?.activeBlocks, 2)
        XCTAssertEqual(native.diagnostics?.sparseSkippedTokens, 512)
    }

    func testNativeSparseVTopKSingleBlockUsesFusedKernel() throws {
        _ = try requireNativeBackend()

        let tokenCount = 512
        let queryHeadCount = 4
        let topK = 128
        let (queries, keyCode, valueCode, scale) = try makeSparseVNativeAttentionCase(
            tokenCount: tokenCount,
            queryHeadCount: queryHeadCount,
            keySeed: 0x5A51_0000_0000_0551,
            valueSeed: 0x5A51_0000_0000_0552
        )

        let reference = try sparseReferenceOutput(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            scale: scale,
            selection: .topK(topK)
        )
        let native = try turboQuantNativeScaledDotProductAttentionWithDiagnostics(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            options: TurboQuantNativeAttentionOptions(
                scale: scale,
                causal: true,
                sparseVSelectionMode: .topK,
                sparseVTopK: topK,
                diagnostics: true
            )
        )

        eval(reference, native.output)
        assertEqual(native.output, reference, rtol: 1e-3, atol: 1e-3)
        XCTAssertEqual(native.diagnostics?.kernelKind, 5)
        XCTAssertEqual(native.diagnostics?.activeBlocks, 1)
        XCTAssertEqual(native.diagnostics?.blockTokens, 512)
        XCTAssertEqual(native.diagnostics?.sparseTotalTokens, tokenCount * queryHeadCount)
        XCTAssertEqual(native.diagnostics?.sparseSkippedTokens, (tokenCount - topK) * queryHeadCount)
    }

    func testNativeSparseVTopKSingleBlockTieKeepsLowerTokens() throws {
        _ = try requireNativeBackend()

        let tokenCount = 512
        let queryHeadCount = 4
        let headDimension = 64
        let topK = 128
        let keys = MLXArray.zeros([1, 1, tokenCount, headDimension])
        let values = MLXArray(
            makeWaveValues(
                count: tokenCount * headDimension,
                sinScale: 0.037,
                sinWeight: -0.06,
                cosScale: 0.013,
                cosWeight: 0.21
            ),
            [1, 1, tokenCount, headDimension]
        )
        let queries = MLXArray.zeros([1, queryHeadCount, 1, headDimension])
        let keyCode = try turboQuantMetalEncodeAttention(
            keys,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .key,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 0x5A51_0000_0000_0553
            )
        )
        let valueCode = try turboQuantMetalEncodeAttention(
            values,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .value,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 0x5A51_0000_0000_0554,
                valueBits: 4
            )
        )
        var selectedWeights = Array(repeating: Float(0), count: queryHeadCount * tokenCount)
        for row in 0 ..< queryHeadCount {
            for token in 0 ..< topK {
                selectedWeights[row * tokenCount + token] = 1 / Float(tokenCount)
            }
        }
        let referenceWeights = MLXArray(
            selectedWeights,
            [1, queryHeadCount, 1, tokenCount]
        )
        let reference = try turboQuantMetalAV(
            attentionWeights: referenceWeights,
            valueCode: valueCode,
            outputDType: queries.dtype
        )
        let native = try turboQuantNativeScaledDotProductAttentionWithDiagnostics(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            options: TurboQuantNativeAttentionOptions(
                scale: 1 / sqrt(Float(headDimension)),
                causal: true,
                sparseVSelectionMode: .topK,
                sparseVTopK: topK,
                diagnostics: true
            )
        )

        eval(reference, native.output)
        assertEqual(native.output, reference, rtol: 1e-3, atol: 1e-3)
        XCTAssertEqual(native.diagnostics?.kernelKind, 5)
        XCTAssertEqual(native.diagnostics?.sparseSkippedTokens, (tokenCount - topK) * queryHeadCount)
    }

    func testNativeSparseVBlockThresholdSplitKMatchesBlockMassReference() throws {
        _ = try requireNativeBackend()

        let tokenCount = 1024
        let queryHeadCount = 1
        let blockSize = 512
        let (queries, keyCode, valueCode, scale) = try makeSparseVNativeAttentionCase(
            tokenCount: tokenCount,
            queryHeadCount: queryHeadCount,
            keySeed: 0x5A51_0000_0000_0561,
            valueSeed: 0x5A51_0000_0000_0562
        )

        let scores = try turboQuantMetalQK(
            queries: queries,
            keyCode: keyCode,
            scale: scale,
            mask: .causal
        )
        let weights = softmax(scores.asType(.float32), axis: -1)
        let threshold = sparseBlockMassThreshold(weights, blockSize: blockSize)
        let selected = sparseBlockThresholdSelectedWeights(
            weights,
            threshold: threshold,
            blockSize: blockSize
        )
        let reference = try turboQuantMetalAV(
            attentionWeights: selected.weights,
            valueCode: valueCode,
            outputDType: queries.dtype
        )
        let native = try turboQuantNativeScaledDotProductAttentionWithDiagnostics(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            options: TurboQuantNativeAttentionOptions(
                scale: scale,
                causal: true,
                splitKBlockCount: 2,
                sparseVThreshold: threshold,
                sparseVSelectionMode: .blockThreshold,
                diagnostics: true
            )
        )

        eval(reference, native.output)
        assertEqual(native.output, reference, rtol: 1e-3, atol: 1e-3)
        XCTAssertEqual(native.diagnostics?.kernelKind, 6)
        XCTAssertEqual(native.diagnostics?.activeBlocks, 2)
        XCTAssertEqual(native.diagnostics?.blockTokens, blockSize)
        XCTAssertEqual(native.diagnostics?.sparseTotalTokens, tokenCount * queryHeadCount)
        XCTAssertEqual(native.diagnostics?.sparseSkippedTokens, selected.skipped)
        XCTAssertGreaterThan(native.diagnostics?.sparseSkippedTokens ?? 0, 0)
    }

    func testNativeSparseVCumulativeSplitKMatchesMaskedAVReference() throws {
        _ = try requireNativeBackend()

        let tokenCount = 1024
        let queryHeadCount = 4
        let mass: Float = 0.80
        let (queries, keyCode, valueCode, scale) = try makeSparseVNativeAttentionCase(
            tokenCount: tokenCount,
            queryHeadCount: queryHeadCount,
            keySeed: 0x5A51_0000_0000_0601,
            valueSeed: 0x5A51_0000_0000_0602
        )

        let reference = try sparseReferenceOutput(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            scale: scale,
            selection: .cumulativeMass(mass)
        )
        let native = try turboQuantNativeScaledDotProductAttentionWithDiagnostics(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            options: TurboQuantNativeAttentionOptions(
                scale: scale,
                causal: true,
                splitKBlockCount: 2,
                sparseVSelectionMode: .cumulativeMass,
                sparseVCumulativeMass: mass,
                diagnostics: true
            )
        )

        eval(reference, native.output)
        assertEqual(native.output, reference, rtol: 1e-3, atol: 1e-3)
        XCTAssertEqual(native.diagnostics?.kernelKind, 10)
        XCTAssertEqual(native.diagnostics?.sparseTotalTokens, tokenCount * queryHeadCount)
        XCTAssertGreaterThan(native.diagnostics?.sparseSkippedTokens ?? 0, 0)
    }

    func testNativeSparseVHybridSplitKMatchesMaskedAVReference() throws {
        _ = try requireNativeBackend()

        let tokenCount = 1024
        let queryHeadCount = 4
        let mass: Float = 0.60
        let maxTopK = 128
        let (queries, keyCode, valueCode, scale) = try makeSparseVNativeAttentionCase(
            tokenCount: tokenCount,
            queryHeadCount: queryHeadCount,
            keySeed: 0x5A51_0000_0000_0701,
            valueSeed: 0x5A51_0000_0000_0702
        )

        let reference = try sparseReferenceOutput(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            scale: scale,
            selection: .hybrid(cumulativeMass: mass, maxTopK: maxTopK)
        )
        let native = try turboQuantNativeScaledDotProductAttentionWithDiagnostics(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            options: TurboQuantNativeAttentionOptions(
                scale: scale,
                causal: true,
                splitKBlockCount: 2,
                sparseVSelectionMode: .hybridCumulativeMassTopK,
                sparseVCumulativeMass: mass,
                sparseVMaxTopK: maxTopK,
                diagnostics: true
            )
        )

        eval(reference, native.output)
        assertEqual(native.output, reference, rtol: 1e-3, atol: 1e-3)
        XCTAssertEqual(native.diagnostics?.kernelKind, 10)
        XCTAssertEqual(native.diagnostics?.sparseTotalTokens, tokenCount * queryHeadCount)
        XCTAssertGreaterThan(native.diagnostics?.sparseSkippedTokens ?? 0, 0)
    }

    private func makeWaveValues(
        count: Int,
        sinScale: Double,
        sinWeight: Double,
        cosScale: Double,
        cosWeight: Double
    ) -> [Float] {
        var values: [Float] = []
        values.reserveCapacity(count)
        for index in 0 ..< count {
            let position = Double(index)
            let sinPart = Foundation.sin(position * sinScale) * sinWeight
            let cosPart = cosScale == 0 ? 0 : Foundation.cos(position * cosScale) * cosWeight
            values.append(Float(sinPart + cosPart))
        }
        return values
    }

    private enum SparseReferenceSelection {
        case topK(Int)
        case cumulativeMass(Float)
        case hybrid(cumulativeMass: Float, maxTopK: Int)
    }

    private func makeSparseVNativeAttentionCase(
        tokenCount: Int,
        queryHeadCount: Int,
        keySeed: UInt64,
        valueSeed: UInt64
    ) throws -> (
        queries: MLXArray,
        keyCode: TurboQuantAttentionCode,
        valueCode: TurboQuantAttentionCode,
        scale: Float
    ) {
        let headDimension = 64
        let keys = MLXArray(
            makeWaveValues(
                count: tokenCount * headDimension,
                sinScale: 0.019,
                sinWeight: 0.18,
                cosScale: 0.007,
                cosWeight: 0.11
            ),
            [1, 1, tokenCount, headDimension]
        )
        let values = MLXArray(
            makeWaveValues(
                count: tokenCount * headDimension,
                sinScale: 0.037,
                sinWeight: -0.06,
                cosScale: 0.013,
                cosWeight: 0.21
            ),
            [1, 1, tokenCount, headDimension]
        )
        let queries = MLXArray(
            makeWaveValues(
                count: queryHeadCount * headDimension,
                sinScale: 0.053,
                sinWeight: 0.13,
                cosScale: 0.029,
                cosWeight: 0.02
            ),
            [1, queryHeadCount, 1, headDimension]
        )

        let keyCode = try turboQuantMetalEncodeAttention(
            keys,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .key,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: keySeed
            )
        )
        let valueCode = try turboQuantMetalEncodeAttention(
            values,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .value,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: valueSeed,
                valueBits: 4
            )
        )

        return (queries, keyCode, valueCode, 1 / sqrt(Float(headDimension)))
    }

    private func pageTopKDecodedMeanReference(
        valueCode: TurboQuantAttentionCode,
        queryHeadCount: Int,
        retainedTokenCount: Int
    ) throws -> MLXArray {
        let decodedValues = try turboQuantMetalDecodeAttention(valueCode, outputDType: .float32)
        eval(decodedValues)
        let tokenCount = decodedValues.dim(2)
        let headDimension = decodedValues.dim(3)
        XCTAssertLessThanOrEqual(retainedTokenCount, tokenCount)
        let values = decodedValues.asArray(Float.self)
        var averaged = Array(repeating: Float(0), count: queryHeadCount * headDimension)
        for head in 0 ..< queryHeadCount {
            for dim in 0 ..< headDimension {
                var sum = Float(0)
                for token in 0 ..< retainedTokenCount {
                    sum += values[token * headDimension + dim]
                }
                averaged[head * headDimension + dim] = sum / Float(retainedTokenCount)
            }
        }
        return MLXArray(averaged, [1, queryHeadCount, 1, headDimension])
    }

    private func pageTopKDecodedMeanReference(
        valueCode: TurboQuantAttentionCode,
        queryHeadCount: Int,
        retainedRanges: [Range<Int>]
    ) throws -> MLXArray {
        let decodedValues = try turboQuantMetalDecodeAttention(valueCode, outputDType: .float32)
        eval(decodedValues)
        let tokenCount = decodedValues.dim(2)
        let headDimension = decodedValues.dim(3)
        let retainedTokenCount = retainedRanges.reduce(0) { $0 + $1.count }
        XCTAssertGreaterThan(retainedTokenCount, 0)
        for range in retainedRanges {
            XCTAssertGreaterThanOrEqual(range.lowerBound, 0)
            XCTAssertLessThanOrEqual(range.upperBound, tokenCount)
        }
        let values = decodedValues.asArray(Float.self)
        var averaged = Array(repeating: Float(0), count: queryHeadCount * headDimension)
        for head in 0 ..< queryHeadCount {
            for dim in 0 ..< headDimension {
                var sum = Float(0)
                for range in retainedRanges {
                    for token in range {
                        sum += values[token * headDimension + dim]
                    }
                }
                averaged[head * headDimension + dim] = sum / Float(retainedTokenCount)
            }
        }
        return MLXArray(averaged, [1, queryHeadCount, 1, headDimension])
    }

    private func swiftMetalTwoStageReference(
        queries: MLXArray,
        keyCode: TurboQuantAttentionCode,
        valueCode: TurboQuantAttentionCode,
        scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) throws -> MLXArray {
        let scores = try turboQuantMetalQK(
            queries: queries,
            keyCode: keyCode,
            scale: scale,
            mask: mask
        )
        let weights = softmax(scores.asType(.float32), axis: -1)
        return try turboQuantMetalAV(
            attentionWeights: weights,
            valueCode: valueCode,
            outputDType: queries.dtype
        )
    }

    private func sparseReferenceOutput(
        queries: MLXArray,
        keyCode: TurboQuantAttentionCode,
        valueCode: TurboQuantAttentionCode,
        scale: Float,
        selection: SparseReferenceSelection
    ) throws -> MLXArray {
        let scores = try turboQuantMetalQK(
            queries: queries,
            keyCode: keyCode,
            scale: scale,
            mask: .causal
        )
        let weights = softmax(scores.asType(.float32), axis: -1)
        let selected = sparseSelectedWeights(weights, selection: selection)
        return try turboQuantMetalAV(
            attentionWeights: selected,
            valueCode: valueCode,
            outputDType: queries.dtype
        )
    }

    private func sparseSelectedWeights(
        _ weights: MLXArray,
        selection: SparseReferenceSelection
    ) -> MLXArray {
        eval(weights)
        let values = weights.asArray(Float.self)
        let columns = weights.dim(-1)
        let rows = values.count / columns
        var selected = Array(repeating: Float(0), count: values.count)

        for row in 0 ..< rows {
            let start = row * columns
            let end = start + columns
            let rowWeights = Array(values[start ..< end])
            let cutoff = sparseCutoff(rowWeights, selection: selection)
            for column in 0 ..< columns where rowWeights[column] >= cutoff {
                selected[start + column] = rowWeights[column]
            }
        }
        return MLXArray(selected, weights.shape)
    }

    private func sparseBlockMassThreshold(_ weights: MLXArray, blockSize: Int) -> Float {
        eval(weights)
        let values = weights.asArray(Float.self)
        let columns = weights.dim(-1)
        let rows = values.count / columns
        var minMass = Float.greatestFiniteMagnitude
        var maxMass = Float.leastNonzeroMagnitude
        for row in 0 ..< rows {
            let rowStart = row * columns
            for blockStart in stride(from: 0, to: columns, by: blockSize) {
                let blockEnd = min(columns, blockStart + blockSize)
                let mass = values[(rowStart + blockStart) ..< (rowStart + blockEnd)]
                    .reduce(Float(0), +)
                minMass = min(minMass, mass)
                maxMass = max(maxMass, mass)
            }
        }
        guard minMass.isFinite, maxMass.isFinite else { return 1 }
        if maxMass - minMass > 1e-6 {
            return (minMass + maxMass) * 0.5
        }
        return min(maxMass + 1e-6, 1)
    }

    private func sparseBlockThresholdSelectedWeights(
        _ weights: MLXArray,
        threshold: Float,
        blockSize: Int
    ) -> (weights: MLXArray, skipped: Int) {
        eval(weights)
        let values = weights.asArray(Float.self)
        let columns = weights.dim(-1)
        let rows = values.count / columns
        var selected = Array(repeating: Float(0), count: values.count)
        var skipped = 0

        for row in 0 ..< rows {
            let rowStart = row * columns
            for blockStart in stride(from: 0, to: columns, by: blockSize) {
                let blockEnd = min(columns, blockStart + blockSize)
                let sourceRange = (rowStart + blockStart) ..< (rowStart + blockEnd)
                let mass = values[sourceRange].reduce(Float(0), +)
                if mass >= threshold {
                    for index in sourceRange {
                        selected[index] = values[index]
                    }
                } else {
                    skipped += blockEnd - blockStart
                }
            }
        }
        return (MLXArray(selected, weights.shape), skipped)
    }

    private func sparseCutoff(
        _ weights: [Float],
        selection: SparseReferenceSelection
    ) -> Float {
        switch selection {
        case .topK(let topK):
            let limit = min(max(0, topK), weights.count)
            guard limit > 0 else { return .infinity }
            return weights.sorted(by: >)[limit - 1]
        case .cumulativeMass(let mass):
            let target = min(1, max(0, mass))
            guard target > 0 else { return .infinity }
            guard target < 1 else { return 0 }
            var low: Float = 0
            var high = weights.max() ?? 0
            for _ in 0 ..< 24 {
                let mid = (low + high) * 0.5
                let retained = weights.reduce(Float(0)) { $0 + ($1 >= mid ? $1 : 0) }
                if retained >= target {
                    low = mid
                } else {
                    high = mid
                }
            }
            return low
        case .hybrid(let mass, let maxTopK):
            let target = min(1, max(0, mass))
            let limit = min(max(0, maxTopK), weights.count)
            guard target > 0, limit > 0 else { return .infinity }
            let sorted = weights.sorted(by: >)
            var retained: Float = 0
            var cutoff = Float.infinity
            for value in sorted.prefix(limit) {
                cutoff = value
                retained += value
                if retained >= target {
                    break
                }
            }
            return cutoff
        }
    }
}
