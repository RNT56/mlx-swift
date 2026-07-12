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
        // K scale plane dieted to 2 scales/group (T1.4 stage 1): 2 groups * 2 * 4B = 16 (was 24).
        XCTAssertEqual(keyEstimate.scaleBytes, 16)
        XCTAssertEqual(keyEstimate.totalBytes, 72)
        XCTAssertEqual(keyEstimate.actualBitsPerValue, 4.5)

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
            // K scale plane dieted to 2 scales/group (T1.4 stage 1); was 3.
            scales: MLXArray.zeros([1, 1, 2, 1, 2], dtype: .float32)
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

    func testPolarWHTReferenceSignsWHTBoundariesAndPacking() throws {
        let signs = try turboQuantPolarWHTSigns(dimension: 8, seed: 0x1234)
        let signsAgain = try turboQuantPolarWHTSigns(dimension: 8, seed: 0x1234)
        let transformed = try turboQuantPolarWHT([1, 2, 3, 4, 5, 6, 7, 8].map(Float.init))
        let restored = try turboQuantPolarWHT(transformed)
        let centroids = try turboQuantPolarWHTCentroids(bits: 3)
        let boundaries = try turboQuantPolarWHTBoundaries(bits: 3)
        let indices: [UInt8] = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2]
        let packed = try turboQuantPolarWHTPackIndices(indices, bits: 3)
        let unpacked = try turboQuantPolarWHTUnpackIndices(packed, bits: 3, count: indices.count)

        XCTAssertEqual(signs, signsAgain)
        XCTAssertEqual(signs.count, 8)
        XCTAssertTrue(signs.allSatisfy { $0 == -1 || $0 == 1 })
        XCTAssertClose(restored, [1, 2, 3, 4, 5, 6, 7, 8].map(Float.init), tolerance: 1e-5)
        XCTAssertEqual(centroids.count, 8)
        XCTAssertEqual(boundaries.count, 7)
        XCTAssertEqual(centroids.first!, -2.1520, accuracy: 1e-4)
        XCTAssertEqual(centroids.last!, 2.1520, accuracy: 1e-4)
        XCTAssertEqual(boundaries[3], 0, accuracy: 1e-5)
        XCTAssertEqual(packed.count, 2)
        XCTAssertEqual(unpacked, indices)
    }

    func testPolarWHTReferenceEncodeDecodeScoresAndPullOut() throws {
        let values = (0 ..< 24).map { index -> Float in
            let x = Double(index)
            return Float(0.5 * sin(x * 0.37) + 0.25 * cos(x * 0.19))
        }
        let array = MLXArray(values, [3, 8])
        let code = try turboQuantPolarWHTReferenceEncode(array, bits: 3, seed: 0xCAFE)
        let decoded = try turboQuantPolarWHTReferenceDecode(code).asArray(Float.self)
        let encodedAgain = try JSONDecoder().decode(
            TurboQuantPolarWHTReferenceCode.self,
            from: JSONEncoder().encode(code)
        )
        let query = (0 ..< 8).map { index -> Float in
            Float(0.2 * cos(Double(index) * 0.41))
        }
        let scores = try turboQuantPolarWHTReferenceScores(query: query, code: code)
        let weights: [Float] = [0.2, -0.4, 0.7]
        let pulledOut = try turboQuantPolarWHTReferenceAccumulate(weights: weights, code: code)
        var denseWeighted = [Float](repeating: 0, count: 8)

        for vectorIndex in 0 ..< 3 {
            let base = vectorIndex * 8
            let denseScore = zip(query, decoded[base ..< base + 8]).reduce(Float(0)) {
                $0 + $1.0 * $1.1
            }
            XCTAssertEqual(scores[vectorIndex], denseScore, accuracy: 1e-5)
            for dimensionIndex in 0 ..< 8 {
                denseWeighted[dimensionIndex] += weights[vectorIndex] * decoded[base + dimensionIndex]
            }
        }

        XCTAssertEqual(code, encodedAgain)
        XCTAssertEqual(code.vectorCount, 3)
        XCTAssertEqual(code.headDimension, 8)
        XCTAssertEqual(code.packedWordsPerVector, 1)
        XCTAssertEqual(code.packedIndices.count, 3)
        XCTAssertEqual(code.norms.count, 3)
        XCTAssertEqual(code.residentPayloadByteCount, 24)
        XCTAssertEqual(code.approximateBitsPerValue, 8)
        XCTAssertTrue(decoded.allSatisfy(\.isFinite))
        XCTAssertClose(pulledOut, denseWeighted, tolerance: 1e-5)
    }

    func testPolarWHTAttentionValuePayloadRoundTripsReferenceContract() throws {
        let shape = [1, 2, 3, 8]
        let values = (0 ..< shape.reduce(1, *)).map { index -> Float in
            let x = Double(index)
            return Float(0.3 * sin(x * 0.23) - 0.2 * cos(x * 0.17))
        }
        let array = MLXArray(values, shape)
        let valueCode = try turboQuantPolarWHTReferenceEncodeAttentionValues(
            array,
            bits: 3,
            seed: 0xC0DE,
            capacity: 5
        )
        let reference = try turboQuantPolarWHTReferenceEncode(
            array,
            bits: 3,
            seed: 0xC0DE
        )
        let decoded = try turboQuantPolarWHTReferenceDecodeAttentionValues(valueCode)
            .asArray(Float.self)
        let referenceDecoded = try turboQuantPolarWHTReferenceDecode(reference)
            .asArray(Float.self)
        let query = (0 ..< 8).map { Float(0.1 * sin(Double($0) * 0.31)) }
        let weights = (0 ..< valueCode.vectorCount).map { Float($0 + 1) / 7 }

        XCTAssertEqual(valueCode.layout.logicalShape, shape)
        XCTAssertEqual(valueCode.layout.capacity, 5)
        XCTAssertEqual(valueCode.layout.groupsPerVector, 1)
        XCTAssertEqual(valueCode.layout.magnitudeWordsPerGroup, 1)
        XCTAssertEqual(valueCode.layout.bitsetWordsPerGroup, 0)
        XCTAssertEqual(valueCode.packedIndexShape, [1, 2, 5, 1])
        XCTAssertEqual(valueCode.normShape, [1, 2, 5])
        XCTAssertEqual(valueCode.packedIndices.dtype, .uint32)
        XCTAssertEqual(valueCode.norms.dtype, .float32)
        XCTAssertEqual(valueCode.logicalValueCount, 48)
        XCTAssertEqual(valueCode.vectorCount, 6)
        XCTAssertEqual(valueCode.capacityVectorCount, 10)

        let estimate = estimateTurboQuantStorage(code: valueCode)
        XCTAssertEqual(estimate.role, .value)
        XCTAssertEqual(estimate.logicalValues, 48)
        XCTAssertEqual(estimate.packedBytes, valueCode.packedIndices.nbytes)
        XCTAssertEqual(estimate.bitsetBytes, 0)
        XCTAssertEqual(estimate.scaleBytes, valueCode.norms.nbytes)
        XCTAssertEqual(estimate.totalBytes, valueCode.storageByteCount)
        XCTAssertEqual(valueCode.residentPayloadByteCount, 80)
        XCTAssertEqual(valueCode.storageByteCount, 80)

        XCTAssertEqual(
            try turboQuantPolarWHTReferenceCode(attentionValueCode: valueCode).packedIndices,
            reference.packedIndices
        )
        XCTAssertClose(decoded, referenceDecoded, tolerance: 1e-5)
        XCTAssertClose(
            try turboQuantPolarWHTReferenceScores(query: query, code: valueCode),
            try turboQuantPolarWHTReferenceScores(query: query, code: reference),
            tolerance: 1e-5
        )
        XCTAssertClose(
            try turboQuantPolarWHTReferenceAccumulate(weights: weights, code: valueCode),
            try turboQuantPolarWHTReferenceAccumulate(weights: weights, code: reference),
            tolerance: 1e-5
        )
        for batch in 0 ..< shape[0] {
            for head in 0 ..< shape[1] {
                let start = (batch * shape[1] + head) * shape[2]
                let headWeights = Array(weights[start ..< start + shape[2]])
                let direct = try turboQuantPolarWHTReferenceAccumulateAttentionValue(
                    weights: headWeights,
                    code: valueCode,
                    batchIndex: batch,
                    kvHeadIndex: head
                )
                let referenceFull = try turboQuantPolarWHTReferenceAccumulate(
                    weights: (0 ..< valueCode.vectorCount).map { index in
                        index >= start && index < start + shape[2] ? weights[index] : 0
                    },
                    code: reference
                )
                XCTAssertClose(direct, referenceFull, tolerance: 1e-5)
            }
        }
    }

    func testPolarWHTAttentionValuePayloadUsesRingPhysicalSlots() throws {
        let shape = [1, 1, 4, 8]
        let values = (0 ..< shape.reduce(1, *)).map { Float($0 + 1) / 13 }
        let array = MLXArray(values, shape)
        let valueCode = try turboQuantPolarWHTReferenceEncodeAttentionValues(
            array,
            bits: 3,
            seed: 0xDAD,
            capacity: 6,
            ringOffset: 2,
            pinnedPrefixLength: 2
        )
        let reference = try turboQuantPolarWHTReferenceEncode(
            array,
            bits: 3,
            seed: 0xDAD
        )
        let decoded = try turboQuantPolarWHTReferenceDecodeAttentionValues(valueCode)
            .asArray(Float.self)
        let referenceDecoded = try turboQuantPolarWHTReferenceDecode(reference)
            .asArray(Float.self)
        let norms = valueCode.norms.asArray(Float.self)
        let query = (0 ..< 8).map { Float(0.05 * cos(Double($0))) }
        let weights = (0 ..< valueCode.vectorCount).map { Float($0 + 2) / 9 }

        XCTAssertEqual(valueCode.layout.ringOffset, 2)
        XCTAssertEqual(valueCode.layout.pinnedPrefixLength, 2)
        XCTAssertGreaterThan(norms[0], 0)
        XCTAssertGreaterThan(norms[1], 0)
        XCTAssertEqual(norms[2], 0)
        XCTAssertEqual(norms[3], 0)
        XCTAssertGreaterThan(norms[4], 0)
        XCTAssertGreaterThan(norms[5], 0)
        XCTAssertEqual(
            try turboQuantPolarWHTReferenceCode(attentionValueCode: valueCode).packedIndices,
            reference.packedIndices
        )
        XCTAssertClose(decoded, referenceDecoded, tolerance: 1e-5)
        XCTAssertClose(
            try turboQuantPolarWHTReferenceScores(query: query, code: valueCode),
            try turboQuantPolarWHTReferenceScores(query: query, code: reference),
            tolerance: 1e-5
        )
        XCTAssertClose(
            try turboQuantPolarWHTReferenceAccumulate(weights: weights, code: valueCode),
            try turboQuantPolarWHTReferenceAccumulate(weights: weights, code: reference),
            tolerance: 1e-5
        )
        XCTAssertClose(
            try turboQuantPolarWHTReferenceAccumulateAttentionValue(
                weights: weights,
                code: valueCode,
                batchIndex: 0,
                kvHeadIndex: 0
            ),
            try turboQuantPolarWHTReferenceAccumulate(weights: weights, code: reference),
            tolerance: 1e-5
        )
    }

    func testMetalPolarWHTAVMatchesReferencePullOut() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("PolarWHT Metal AV requires a GPU device")
        }

        let shape = [1, 2, 4, 8]
        let queryHeadCount = 4
        let queryLength = 2
        let values = (0 ..< shape.reduce(1, *)).map { index -> Float in
            let x = Double(index)
            return Float(0.35 * sin(x * 0.17) - 0.21 * cos(x * 0.29))
        }
        let attentionWeightValues = (0 ..< shape[0] * queryHeadCount * queryLength * shape[2])
            .map { index -> Float in
                let x = Double(index)
                return Float(0.2 + 0.05 * sin(x * 0.41) + 0.03 * cos(x * 0.13))
            }
        var valueCode = try turboQuantPolarWHTReferenceEncodeAttentionValues(
            MLXArray(values, shape),
            bits: 3,
            seed: 0xBEE5,
            capacity: 6,
            ringOffset: 1,
            pinnedPrefixLength: 1
        )
        valueCode.packedIndices = valueCode.packedIndices.contiguous(stream: .gpu)
        valueCode.norms = valueCode.norms.contiguous(stream: .gpu)
        let attentionWeights = MLXArray(
            attentionWeightValues,
            [shape[0], queryHeadCount, queryLength, shape[2]]
        ).contiguous(stream: .gpu)

        var expected = [Float]()
        let repeats = queryHeadCount / shape[1]
        for batch in 0 ..< shape[0] {
            for queryHead in 0 ..< queryHeadCount {
                let kvHead = queryHead / repeats
                for queryToken in 0 ..< queryLength {
                    let weightStart =
                        (((batch * queryHeadCount + queryHead) * queryLength + queryToken)
                            * shape[2])
                    let weights = Array(
                        attentionWeightValues[weightStart ..< weightStart + shape[2]]
                    )
                    expected += try turboQuantPolarWHTReferenceAccumulateAttentionValue(
                        weights: weights,
                        code: valueCode,
                        batchIndex: batch,
                        kvHeadIndex: kvHead
                    )
                }
            }
        }

        do {
            let output = try turboQuantMetalPolarWHTAV(
                attentionWeights: attentionWeights,
                valueCode: valueCode,
                outputDType: .float32
            )
            XCTAssertClose(output.asArray(Float.self), expected, tolerance: 2e-4)
        } catch TurboQuantError.unsupportedBackend(_, let reason) {
            throw XCTSkip(reason)
        }
    }

    func testMetalPolarWHTEncodeDecodeMatchesReferencePayload() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("PolarWHT Metal codec requires a GPU device")
        }

        let shape = [1, 2, 4, 64]
        let values = (0 ..< shape.reduce(1, *)).map { index -> Float in
            let x = Double(index)
            return Float(0.27 * sin(x * 0.019) - 0.13 * cos(x * 0.041))
        }
        let input = MLXArray(values, shape)
        let code = try turboQuantMetalPolarWHTEncodeAttentionValues(
            input,
            bits: 3,
            seed: 0xE11C_0DE,
            capacity: 6,
            ringOffset: 1,
            pinnedPrefixLength: 1
        )
        let decoded = try turboQuantMetalPolarWHTDecodeAttentionValues(
            code,
            outputDType: .float32
        )
        let reference = try turboQuantPolarWHTReferenceDecodeAttentionValues(code)

        XCTAssertEqual(code.packedIndices.shape, [1, 2, 6, 7])
        XCTAssertEqual(code.norms.shape, [1, 2, 6])
        XCTAssertEqual(decoded.shape, shape)
        XCTAssertClose(
            decoded.asArray(Float.self),
            reference.asArray(Float.self),
            tolerance: 2e-4
        )
    }

    func testMetalPolarWHTScaledAttentionMatchesTwoStageComposition() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("PolarWHT Metal scaled attention requires a GPU device")
        }

        let tokenCount = 4
        let headDimension = 64
        let keyValues = (0 ..< 1 * 2 * tokenCount * headDimension).map { index -> Float in
            let x = Double(index)
            return Float(0.21 * sin(x * 0.013) + 0.09 * cos(x * 0.037))
        }
        let valueValues = (0 ..< 1 * 2 * tokenCount * headDimension).map { index -> Float in
            let x = Double(index)
            return Float(0.33 * cos(x * 0.017) - 0.15 * sin(x * 0.029))
        }
        let queryValues = (0 ..< 1 * 4 * 2 * headDimension).map { index -> Float in
            let x = Double(index)
            return Float(0.18 * sin(x * 0.023) - 0.07 * cos(x * 0.031))
        }
        let keys = MLXArray(keyValues, [1, 2, tokenCount, headDimension])
        let values = MLXArray(valueValues, [1, 2, tokenCount, headDimension])
        let queries = MLXArray(queryValues, [1, 4, 2, headDimension]).contiguous(stream: .gpu)
        let keyCode = try turboQuantMetalPolarWHTEncodeAttentionValues(
            keys,
            bits: 3,
            seed: 0xA11E_0101,
            capacity: 6,
            ringOffset: 1,
            pinnedPrefixLength: 1
        )
        let valueCode = try turboQuantMetalPolarWHTEncodeAttentionValues(
            values,
            bits: 3,
            seed: 0xA11E_0102,
            capacity: 6,
            ringOffset: 1,
            pinnedPrefixLength: 1
        )
        let scale = 1 / sqrt(Float(headDimension))
        let scores = try turboQuantMetalPolarWHTQK(
            queries: queries,
            keyCode: keyCode,
            scale: scale,
            mask: .causal
        )
        let expected = try turboQuantMetalPolarWHTAV(
            attentionWeights: softmax(scores.asType(.float32), axis: -1),
            valueCode: valueCode,
            outputDType: .float32
        )
        let actual = try turboQuantMetalPolarWHTScaledDotProductAttention(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            scale: scale,
            mask: .causal,
            outputDType: .float32
        )

        XCTAssertClose(
            actual.asArray(Float.self),
            expected.asArray(Float.self),
            tolerance: 1e-4
        )
    }

    func testMetalPolarWHTQKMatchesReferenceAcrossHeadDimensions() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("PolarWHT Metal QK requires a GPU device")
        }

        for headDimension in [64, 128, 256] {
            let keyShape = [1, 2, 4, headDimension]
            let queryHeadCount = 4
            let queryLength = 2
            let scale = Float(0.125)
            let keyValues = (0 ..< keyShape.reduce(1, *)).map { index -> Float in
                let x = Double(index)
                return Float(0.25 * sin(x * 0.013) + 0.17 * cos(x * 0.031))
            }
            let queryValues = (0 ..< keyShape[0] * queryHeadCount * queryLength * headDimension)
                .map { index -> Float in
                    let x = Double(index)
                    return Float(0.19 * sin(x * 0.021) - 0.11 * cos(x * 0.017))
                }
            var keyCode = try turboQuantPolarWHTReferenceEncodeAttentionValues(
                MLXArray(keyValues, keyShape),
                bits: 3,
                seed: 0xC001 + UInt64(headDimension),
                capacity: 6,
                ringOffset: 1,
                pinnedPrefixLength: 1
            )
            keyCode.packedIndices = keyCode.packedIndices.contiguous(stream: .gpu)
            keyCode.norms = keyCode.norms.contiguous(stream: .gpu)
            let queries = MLXArray(
                queryValues,
                [keyShape[0], queryHeadCount, queryLength, headDimension]
            ).contiguous(stream: .gpu)
            let referenceCode = try turboQuantPolarWHTReferenceCode(attentionValueCode: keyCode)
            let repeats = queryHeadCount / keyShape[1]
            var expected = [Float]()
            expected.reserveCapacity(keyShape[0] * queryHeadCount * queryLength * keyShape[2])

            for batch in 0 ..< keyShape[0] {
                for queryHead in 0 ..< queryHeadCount {
                    let kvHead = queryHead / repeats
                    for queryToken in 0 ..< queryLength {
                        let queryStart =
                            (((batch * queryHeadCount + queryHead) * queryLength + queryToken)
                                * headDimension)
                        let query = Array(queryValues[queryStart ..< queryStart + headDimension])
                        let scores = try turboQuantPolarWHTReferenceScores(
                            query: query,
                            code: referenceCode,
                            scale: scale
                        )
                        let scoreStart = (batch * keyShape[1] + kvHead) * keyShape[2]
                        expected += Array(scores[scoreStart ..< scoreStart + keyShape[2]])
                    }
                }
            }

            do {
                let output = try turboQuantMetalPolarWHTQK(
                    queries: queries,
                    keyCode: keyCode,
                    scale: scale
                )
                XCTAssertClose(
                    output.asArray(Float.self),
                    expected,
                    tolerance: 6e-4
                )
            } catch TurboQuantError.unsupportedBackend(_, let reason) {
                throw XCTSkip(reason)
            }
        }
    }

    func testEmptyPolarWHTAttentionValuePayloadUsesDeterministicResidentShape() throws {
        let layout = TurboQuantAttentionLayout(
            batchSize: 1,
            kvHeadCount: 2,
            capacity: 4,
            logicalLength: 0,
            headDimension: 8,
            groupsPerVector: 99,
            magnitudeWordsPerGroup: 99,
            bitsetWordsPerGroup: 99
        )
        let code = try turboQuantEmptyPolarWHTAttentionValueCode(
            layout: layout,
            bits: 4,
            normStorage: .float16
        )

        XCTAssertEqual(code.layout.groupsPerVector, 1)
        XCTAssertEqual(code.layout.magnitudeWordsPerGroup, 1)
        XCTAssertEqual(code.layout.bitsetWordsPerGroup, 0)
        XCTAssertEqual(code.packedIndexShape, [1, 2, 4, 1])
        XCTAssertEqual(code.normShape, [1, 2, 4])
        XCTAssertEqual(code.packedIndices.nbytes, 32)
        XCTAssertEqual(code.norms.nbytes, 16)
        XCTAssertEqual(code.storageByteCount, 48)
        XCTAssertEqual(code.approximateBitsPerValue, 0)
    }

    func testContractDTORoundTripsWithoutMetal() throws {
        let capabilities = TurboQuantKernelCapabilities(
            nativeCompressedAttention: true,
            nativeSparseVSupport: true,
            nativeDiagnosticsSupport: true,
            nativeBackendVersion: TurboQuantNativeAttentionOptions.backendVersion,
            nativeSegmentedAttentionBackend: .experimentalJIT,
            nativePolarWHTSegmentedAttentionBackend: .unavailable,
            flatEncodeDecode: false,
            linearMatmul: false,
            attentionEncode: true,
            attentionDecode: true,
            attentionQK: true,
            attentionAV: true,
            attentionFusedDecode: false,
            polarWHTCodec: false,
            polarWHTAttention: false,
            hybridK8PolarWHTValueAttention: false,
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
        XCTAssertEqual(capabilities.nativeCompressedAttention, true)
        XCTAssertEqual(capabilities.nativeBackendVersion, TurboQuantNativeAttentionOptions.backendVersion)
        XCTAssertEqual(capabilities.nativeSegmentedAttentionBackend, .experimentalJIT)
        XCTAssertEqual(
            capabilities.attentionCapabilities.nativeSegmentedAttentionBackend,
            .experimentalJIT
        )
        XCTAssertEqual(
            capabilities.attentionCapabilities.nativePolarWHTSegmentedAttentionBackend,
            .unavailable
        )
        XCTAssertFalse(capabilities.attentionCapabilities.polarWHTCodec)
        XCTAssertFalse(capabilities.attentionCapabilities.polarWHTAttention)
        XCTAssertFalse(capabilities.attentionCapabilities.hybridK8PolarWHTValueAttention)
        XCTAssertEqual(
            capabilities.supportedHeadDimensions,
            TurboQuantRuntimeProbeResult.throughputOptimizedOnlineFusedHeadDimensions
        )
    }

    func testOldKernelCapabilitySnapshotDecodesWithoutNativeFields() throws {
        let oldSnapshot = Data(
            """
            {
              "flatEncodeDecode": false,
              "linearMatmul": false,
              "attentionEncode": true,
              "attentionDecode": true,
              "attentionQK": true,
              "attentionAV": true,
              "attentionFusedDecode": false,
              "bfloatOutput": false,
              "supportedHeadDimensions": [64, 128, 256],
              "selectedKernelProfile": "mlxPackedFallback",
              "failureReasons": []
            }
            """.utf8)

        let decoded = try JSONDecoder().decode(TurboQuantKernelCapabilities.self, from: oldSnapshot)

        XCTAssertNil(decoded.nativeCompressedAttention)
        XCTAssertNil(decoded.nativeSparseVSupport)
        XCTAssertNil(decoded.nativeDiagnosticsSupport)
        XCTAssertNil(decoded.nativeBackendVersion)
        XCTAssertNil(decoded.nativeSegmentedAttentionBackend)
        XCTAssertNil(decoded.nativePolarWHTSegmentedAttentionBackend)
        XCTAssertNil(decoded.nativeFallbackReason)
        XCTAssertFalse(decoded.polarWHTCodec)
        XCTAssertFalse(decoded.polarWHTAttention)
        XCTAssertFalse(decoded.hybridK8PolarWHTValueAttention)
        XCTAssertTrue(decoded.attentionQK)
        XCTAssertFalse(decoded.attentionFusedDecode)
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
        XCTAssertFalse(defaults.polarWHTCodec)
        XCTAssertFalse(defaults.polarWHTAttention)
        XCTAssertFalse(defaults.hybridK8PolarWHTValueAttention)
        XCTAssertFalse(defaults.bfloatOutput)

        XCTAssertTrue(twoStageOnly.attentionQK)
        XCTAssertTrue(twoStageOnly.attentionAV)
        XCTAssertFalse(twoStageOnly.attentionFusedDecode)
        XCTAssertFalse(twoStageOnly.bfloatOutput)
        XCTAssertTrue(
            TurboQuantKernelAvailability.currentCapabilities().attentionCapabilities.supportedDTypes
                .contains(.float16))
    }

    private func XCTAssertClose(
        _ lhs: [Float],
        _ rhs: [Float],
        tolerance: Float,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(lhs.count, rhs.count, file: file, line: line)
        for (left, right) in zip(lhs, rhs) {
            XCTAssertLessThanOrEqual(abs(left - right), tolerance, file: file, line: line)
        }
    }
}
