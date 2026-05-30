// Copyright © 2026 RNT56.

import Foundation
import MLX
import XCTest

final class TurboQuantContractsTests: XCTestCase {
    func testStorageEstimateRoundTripsCodableAndHashable() throws {
        let estimate = TurboQuantStorageEstimate(
            role: .key,
            logicalValues: 128,
            packedBytes: 40,
            bitsetBytes: 48,
            scaleBytes: 24
        )

        let data = try JSONEncoder().encode(estimate)
        let decoded = try JSONDecoder().decode(TurboQuantStorageEstimate.self, from: data)

        XCTAssertEqual(decoded, estimate)
        XCTAssertEqual(Set([estimate, decoded]).count, 1)
        XCTAssertEqual(decoded.totalBytes, 112)
        XCTAssertEqual(decoded.actualBitsPerValue, 7)
    }

    func testSymbolicStorageEstimateIsNonnegativeAndTotalsBytes() {
        let keyEstimate = estimateTurboQuantStorage(
            role: .key,
            logicalValues: 128,
            preset: .turbo3_5,
            groupSize: 64,
            dtype: .float32
        )
        let valueEstimate = estimateTurboQuantStorage(
            role: .value,
            logicalValues: 128,
            preset: .turbo4v2,
            valueBits: 4,
            groupSize: 64,
            dtype: .float32
        )
        let fp16KeyEstimate = estimateTurboQuantStorage(
            role: .key,
            logicalValues: 128,
            preset: .turbo3_5,
            groupSize: 64,
            dtype: .float16
        )
        let emptyEstimate = estimateTurboQuantStorage(
            role: .key,
            logicalValues: -1,
            preset: .turbo3_5,
            groupSize: 0,
            dtype: .float32
        )

        XCTAssertEqual(keyEstimate.packedBytes, 40)
        XCTAssertEqual(keyEstimate.bitsetBytes, 16)
        XCTAssertEqual(keyEstimate.scaleBytes, 24)
        XCTAssertEqual(keyEstimate.totalBytes, 80)
        XCTAssertEqual(keyEstimate.actualBitsPerValue, 5.0)

        XCTAssertEqual(valueEstimate.packedBytes, 64)
        XCTAssertEqual(valueEstimate.bitsetBytes, 0)
        XCTAssertEqual(valueEstimate.scaleBytes, 16)
        XCTAssertEqual(valueEstimate.totalBytes, 80)
        XCTAssertEqual(valueEstimate.actualBitsPerValue, 5)
        XCTAssertEqual(fp16KeyEstimate.scaleBytes, keyEstimate.scaleBytes)
        XCTAssertEqual(fp16KeyEstimate.totalBytes, keyEstimate.totalBytes)

        for estimate in [keyEstimate, valueEstimate, fp16KeyEstimate, emptyEstimate] {
            XCTAssertGreaterThanOrEqual(estimate.logicalValues, 0)
            XCTAssertGreaterThanOrEqual(estimate.packedBytes, 0)
            XCTAssertGreaterThanOrEqual(estimate.bitsetBytes, 0)
            XCTAssertGreaterThanOrEqual(estimate.scaleBytes, 0)
            XCTAssertEqual(
                estimate.totalBytes,
                estimate.packedBytes + estimate.bitsetBytes + estimate.scaleBytes
            )
            XCTAssertGreaterThanOrEqual(estimate.actualBitsPerValue, 0)
        }
    }

    func testActualStorageEstimateMatchesAttentionCodeStorage() {
        let keyLayout = TurboQuantAttentionLayout(
            batchSize: 1,
            kvHeadCount: 1,
            capacity: 2,
            logicalLength: 2,
            headDimension: 64,
            groupsPerVector: 1,
            magnitudeWordsPerGroup: 5,
            bitsetWordsPerGroup: 2
        )
        let valueLayout = TurboQuantAttentionLayout(
            batchSize: 1,
            kvHeadCount: 1,
            capacity: 2,
            logicalLength: 2,
            headDimension: 64,
            groupsPerVector: 1,
            magnitudeWordsPerGroup: 8,
            bitsetWordsPerGroup: 2
        )
        let keyCode = TurboQuantAttentionCode(
            layout: keyLayout,
            preset: .turbo3_5,
            role: .key,
            groupSize: 64,
            seed: 0,
            packedMagnitudes: MLXArray.zeros([1, 1, 2, 1, 5], dtype: .uint32),
            signs: MLXArray.zeros([1, 1, 2, 1, 2], dtype: .uint32),
            highPrecisionMask: MLXArray.zeros([1], dtype: .uint32),
            residualSigns: MLXArray.zeros([1], dtype: .uint32),
            scales: MLXArray.zeros([1, 1, 2, 1, 3], dtype: .float32)
        )
        let valueCode = TurboQuantAttentionCode(
            layout: valueLayout,
            preset: .turbo4v2,
            role: .value,
            groupSize: 64,
            seed: 0,
            packedMagnitudes: MLXArray.zeros([1, 1, 2, 1, 8], dtype: .uint32),
            signs: MLXArray.zeros([1], dtype: .uint32),
            highPrecisionMask: MLXArray.zeros([1], dtype: .uint32),
            residualSigns: MLXArray.zeros([1], dtype: .uint32),
            scales: MLXArray.zeros([1, 1, 2, 1, 2], dtype: .float32)
        )

        let keyEstimate = estimateTurboQuantStorage(code: keyCode)
        let valueEstimate = estimateTurboQuantStorage(code: valueCode)

        XCTAssertEqual(keyEstimate.role, .key)
        XCTAssertEqual(keyEstimate.logicalValues, 128)
        XCTAssertEqual(keyEstimate.packedBytes, keyCode.packedMagnitudes.nbytes)
        XCTAssertEqual(
            keyEstimate.bitsetBytes,
            keyCode.signs.nbytes + keyCode.highPrecisionMask.nbytes + keyCode.residualSigns.nbytes
        )
        XCTAssertEqual(keyEstimate.scaleBytes, keyCode.scales.nbytes)
        XCTAssertEqual(keyEstimate.totalBytes, keyCode.storageByteCount)
        XCTAssertEqual(keyEstimate.actualBitsPerValue, keyCode.approximateBitsPerValue)

        XCTAssertEqual(valueEstimate.role, .value)
        XCTAssertEqual(valueEstimate.logicalValues, 128)
        XCTAssertEqual(valueEstimate.packedBytes, valueCode.packedMagnitudes.nbytes)
        XCTAssertEqual(valueEstimate.bitsetBytes, 0)
        XCTAssertEqual(valueEstimate.scaleBytes, valueCode.scales.nbytes)
        XCTAssertEqual(valueEstimate.totalBytes, valueCode.storageByteCount)
        XCTAssertEqual(valueEstimate.actualBitsPerValue, valueCode.approximateBitsPerValue)
    }

    func testContractDTORoundTripsWithoutMetal() throws {
        let capabilities = TurboQuantKernelCapabilities(
            flatEncodeDecode: false,
            linearMatmul: false,
            attentionEncode: true,
            attentionDecode: true,
            attentionQK: true,
            attentionAV: true,
            attentionFusedDecode: false,
            bfloatOutput: false
        )
        let decision = TurboQuantAttentionDecision(
            selectedPath: .twoStageCompressed,
            outputDType: .float16,
            estimatedScratchBytes: 4096,
            rejectedPaths: [
                RejectedPath(path: .onlineFused, reason: "fused path is not certified")
            ]
        )

        XCTAssertEqual(
            try JSONDecoder().decode(
                TurboQuantKernelCapabilities.self,
                from: JSONEncoder().encode(capabilities)
            ),
            capabilities
        )
        XCTAssertEqual(
            try JSONDecoder().decode(
                TurboQuantAttentionDecision.self,
                from: JSONEncoder().encode(decision)
            ),
            decision
        )
        XCTAssertEqual(Set(decision.rejectedPaths).count, 1)
        XCTAssertEqual(decision.rejectedPaths.first?.reason, "fused path is not certified")
        XCTAssertTrue(capabilities.qk)
        XCTAssertTrue(capabilities.av)
        XCTAssertFalse(capabilities.onlineFused)
        XCTAssertFalse(capabilities.tiledFused)
        XCTAssertEqual(
            capabilities.supportedHeadDimensions,
            TurboQuantRuntimeProbeResult.throughputOptimizedOnlineFusedHeadDimensions
        )
    }

    func testRejectedTurboQuantPathAliasMatchesRouterContract() {
        let rejected = RejectedTurboQuantPath(path: .twoStageCompressed, reason: "unsupported mask")

        XCTAssertEqual(rejected.path, .twoStageCompressed)
        XCTAssertEqual(rejected.reason, "unsupported mask")
    }

    func testKernelCapabilityDefaultsAreSafeAndPathSpecific() {
        let defaults = TurboQuantKernelCapabilities()
        let twoStageOnly = TurboQuantKernelCapabilities(
            attentionEncode: true,
            attentionDecode: true,
            attentionQK: true,
            attentionAV: true,
            attentionFusedDecode: false,
            bfloatOutput: false
        )

        XCTAssertFalse(defaults.flatEncodeDecode)
        XCTAssertFalse(defaults.linearMatmul)
        XCTAssertFalse(defaults.attentionEncode)
        XCTAssertFalse(defaults.attentionDecode)
        XCTAssertFalse(defaults.attentionQK)
        XCTAssertFalse(defaults.attentionAV)
        XCTAssertFalse(defaults.attentionFusedDecode)
        XCTAssertFalse(defaults.bfloatOutput)

        XCTAssertTrue(twoStageOnly.attentionQK)
        XCTAssertTrue(twoStageOnly.attentionAV)
        XCTAssertFalse(twoStageOnly.attentionFusedDecode)
        XCTAssertFalse(twoStageOnly.bfloatOutput)
        XCTAssertTrue(
            TurboQuantKernelAvailability.currentCapabilities().attentionCapabilities.supportedDTypes
                .contains(.float16))
    }
}
