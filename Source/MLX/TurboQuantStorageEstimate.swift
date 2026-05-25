// Copyright © 2026 RNT56.

import Foundation

public struct TurboQuantStorageEstimate: Hashable, Codable, Sendable {
    public var role: TurboQuantTensorRole
    public var logicalValues: Int
    public var packedBytes: Int
    public var bitsetBytes: Int
    public var scaleBytes: Int
    public var totalBytes: Int
    public var actualBitsPerValue: Double

    public init(
        role: TurboQuantTensorRole,
        logicalValues: Int,
        packedBytes: Int,
        bitsetBytes: Int,
        scaleBytes: Int,
        totalBytes: Int? = nil,
        actualBitsPerValue: Double? = nil
    ) {
        let clampedLogicalValues = Swift.max(0, logicalValues)
        let clampedPackedBytes = Swift.max(0, packedBytes)
        let clampedBitsetBytes = Swift.max(0, bitsetBytes)
        let clampedScaleBytes = Swift.max(0, scaleBytes)
        let computedTotalBytes = clampedPackedBytes + clampedBitsetBytes + clampedScaleBytes
        let clampedTotalBytes = Swift.max(0, totalBytes ?? computedTotalBytes)

        self.role = role
        self.logicalValues = clampedLogicalValues
        self.packedBytes = clampedPackedBytes
        self.bitsetBytes = clampedBitsetBytes
        self.scaleBytes = clampedScaleBytes
        self.totalBytes = clampedTotalBytes
        self.actualBitsPerValue =
            actualBitsPerValue
            ?? TurboQuantStorageEstimate.bitsPerValue(
                totalBytes: clampedTotalBytes,
                logicalValues: clampedLogicalValues
            )
    }

    private static func bitsPerValue(totalBytes: Int, logicalValues: Int) -> Double {
        guard logicalValues > 0 else { return 0 }
        return Double(totalBytes * 8) / Double(logicalValues)
    }
}

public func estimateTurboQuantStorage(
    role: TurboQuantTensorRole,
    logicalValues: Int,
    preset: TurboQuantPreset,
    valueBits: Int? = nil,
    groupSize: Int,
    dtype: DType,
    scaleStorage: TurboQuantScaleStorage = .float32
) -> TurboQuantStorageEstimate {
    let clampedLogicalValues = Swift.max(0, logicalValues)
    let clampedGroupSize = Swift.max(1, groupSize)
    let groupCount = ceilDivide(clampedLogicalValues, by: clampedGroupSize)
    let magnitudeWords = turboQuantEstimateMagnitudeWordsPerGroup(
        groupSize: clampedGroupSize,
        preset: preset,
        role: role,
        valueBits: valueBits
    )
    let bitsetWords = ceilDivide(clampedGroupSize, by: 32)
    let scalesPerGroup = role == .value ? 2 : 3
    let packedBytes = groupCount * magnitudeWords * MemoryLayout<UInt32>.size
    let bitsetBytes = role == .value ? 0 : groupCount * bitsetWords * 3 * MemoryLayout<UInt32>.size
    let scaleBytes = groupCount * scalesPerGroup * scaleStorage.dtype.size

    return TurboQuantStorageEstimate(
        role: role,
        logicalValues: clampedLogicalValues,
        packedBytes: packedBytes,
        bitsetBytes: bitsetBytes,
        scaleBytes: scaleBytes
    )
}

public func estimateTurboQuantStorage(
    code: TurboQuantAttentionCode
) -> TurboQuantStorageEstimate {
    let logicalValues =
        code.layout.batchSize * code.layout.kvHeadCount
        * Swift.max(0, code.layout.logicalLength) * code.layout.headDimension
    let packedBytes = code.packedMagnitudes.nbytes
    let bitsetBytes =
        code.role == .value
        ? 0
        : code.signs.nbytes + code.highPrecisionMask.nbytes + code.residualSigns.nbytes
    let scaleBytes = code.scales.nbytes

    return TurboQuantStorageEstimate(
        role: code.role,
        logicalValues: logicalValues,
        packedBytes: packedBytes,
        bitsetBytes: bitsetBytes,
        scaleBytes: scaleBytes,
        totalBytes: code.storageByteCount,
        actualBitsPerValue: code.approximateBitsPerValue
    )
}

private func ceilDivide(_ value: Int, by divisor: Int) -> Int {
    guard value > 0 else { return 0 }
    return (value + divisor - 1) / divisor
}

private func turboQuantEstimateMagnitudeWordsPerGroup(
    groupSize: Int,
    preset: TurboQuantPreset,
    role: TurboQuantTensorRole,
    valueBits: Int?
) -> Int {
    if role == .value {
        let bits = Swift.max(1, valueBits ?? preset.defaultValueBits)
        return ceilDivide(groupSize * bits, by: 32)
    }

    let baseBits = Swift.max(1, preset.baseMagnitudeBits - 1)
    let highBits = Swift.max(baseBits, preset.highMagnitudeBits - 1)
    let highCount = turboQuantEstimateHighCount(
        valueCount: groupSize,
        baseBits: baseBits,
        highBits: highBits,
        targetBits: Swift.max(1, preset.targetMagnitudeBits - 1)
    )
    let bitCount = groupSize * baseBits + highCount * (highBits - baseBits)
    return ceilDivide(bitCount, by: 32)
}

private func turboQuantEstimateHighCount(
    valueCount: Int,
    baseBits: Int,
    highBits: Int,
    targetBits: Float
) -> Int {
    guard valueCount > 0, highBits > baseBits else { return 0 }
    let highFraction = (targetBits - Float(baseBits)) / Float(highBits - baseBits)
    let boundedFraction = Swift.max(0, Swift.min(1, highFraction))
    return Int((Float(valueCount) * boundedFraction).rounded())
}
