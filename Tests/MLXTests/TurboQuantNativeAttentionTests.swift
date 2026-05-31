// Copyright © 2026 RNT56.

import Foundation
import MLX
import XCTest

final class TurboQuantNativeAttentionTests: XCTestCase {

    func testSegmentedSwiftSurfaceRetainsScaledCompatibilityNames() {
        XCTAssertEqual(TurboQuantNativeSegmentedAttentionBackend.unavailable.rawValue, 0)
        XCTAssertEqual(TurboQuantNativeSegmentedAttentionBackend.experimentalJIT.rawValue, 1)
        XCTAssertEqual(TurboQuantNativeSegmentedAttentionBackend.nativeFused.rawValue, 2)

        let result = TurboQuantNativeSegmentedAttentionResult(
            output: MLXArray.zeros([1], dtype: .float32)
        )
        XCTAssertEqual(result.output.shape, [1])
    }

    func testNativeCapabilityProbePassesWhenEnabled() throws {
        guard ProcessInfo.processInfo.environment["MLX_TURBOQUANT_NATIVE_ATTENTION"] == "1" else {
            throw XCTSkip("native MLX TurboQuant attention gate is disabled")
        }

        let capabilities = TurboQuantKernelAvailability.current.attentionCapabilities
        XCTAssertEqual(capabilities.nativeCompressedAttention, true)
        XCTAssertEqual(capabilities.nativeSparseVSupport, true)
        XCTAssertEqual(capabilities.nativeDiagnosticsSupport, true)
        XCTAssertEqual(capabilities.nativeBackendVersion, TurboQuantNativeAttentionOptions.backendVersion)
    }

    func testNativeFusedAttentionMatchesSwiftMetalWhenEnabled() throws {
        guard ProcessInfo.processInfo.environment["MLX_TURBOQUANT_NATIVE_ATTENTION"] == "1" else {
            throw XCTSkip("native MLX TurboQuant attention gate is disabled")
        }

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

    func testNativeBlockParallelGQAMatchesSwiftMetalWhenEnabled() throws {
        guard ProcessInfo.processInfo.environment["MLX_TURBOQUANT_NATIVE_ATTENTION"] == "1" else {
            throw XCTSkip("native MLX TurboQuant attention gate is disabled")
        }

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
        guard ProcessInfo.processInfo.environment["MLX_TURBOQUANT_NATIVE_ATTENTION"] == "1" else {
            throw XCTSkip("native MLX TurboQuant attention gate is disabled")
        }

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
}
