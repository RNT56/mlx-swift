// Copyright © 2026 RNT56.

import Foundation

public func validateTurboQuantAttentionCode(
    _ code: TurboQuantAttentionCode,
    expectedRole: TurboQuantTensorRole?,
    requireWritableCapacity: Bool = false
) throws {
    if let expectedRole {
        guard code.role == expectedRole else {
            throw turboQuantAttentionValidationError(
                "tensor role actual \(code.role.rawValue), expected \(expectedRole.rawValue)"
            )
        }
    }

    try turboQuantValidateAttentionLayoutDescriptor(
        code.layout,
        role: code.role,
        groupSize: code.groupSize
    )

    if requireWritableCapacity {
        let writableSlots = code.layout.capacity - code.layout.logicalLength
        guard writableSlots > 0 else {
            throw turboQuantAttentionValidationError(
                "writable capacity actual \(writableSlots), expected at least 1 "
                    + "(capacity \(code.layout.capacity), logical length \(code.layout.logicalLength))"
            )
        }
    }

    if code.role == .value {
        try turboQuantValidateAttentionValueBits(code.valueBits)
    }

    let expectedMagnitudeWords = turboQuantAttentionMagnitudeWordsPerGroup(
        groupSize: code.groupSize,
        preset: code.preset,
        role: code.role,
        valueBits: code.valueBits
    )
    guard code.layout.magnitudeWordsPerGroup == expectedMagnitudeWords else {
        throw turboQuantAttentionValidationError(
            "magnitude words per group actual \(code.layout.magnitudeWordsPerGroup), "
                + "expected \(expectedMagnitudeWords)"
        )
    }

    let expectedBitsetWords = turboQuantAttentionCeilDivide(code.groupSize, by: 32)
    guard code.layout.bitsetWordsPerGroup == expectedBitsetWords else {
        throw turboQuantAttentionValidationError(
            "bitset words per group actual \(code.layout.bitsetWordsPerGroup), "
                + "expected \(expectedBitsetWords)"
        )
    }

    let expectedScalesPerGroup = turboQuantAttentionScalesPerGroup(role: code.role)
    guard code.scalesPerGroup == expectedScalesPerGroup else {
        throw turboQuantAttentionValidationError(
            "scales per group actual \(code.scalesPerGroup), expected \(expectedScalesPerGroup)"
        )
    }

    let packedShape = [
        code.layout.batchSize, code.layout.kvHeadCount, code.layout.capacity,
        code.layout.groupsPerVector, code.layout.magnitudeWordsPerGroup,
    ]
    let bitsetShape = [
        code.layout.batchSize, code.layout.kvHeadCount, code.layout.capacity,
        code.layout.groupsPerVector, code.layout.bitsetWordsPerGroup,
    ]
    let scalesShape = [
        code.layout.batchSize, code.layout.kvHeadCount, code.layout.capacity,
        code.layout.groupsPerVector, code.scalesPerGroup,
    ]
    let compactUnusedBitsetShape = [1]

    try turboQuantValidateAttentionStorageArray(
        code.packedMagnitudes,
        name: "compressed attention packed magnitudes",
        expectedShape: packedShape,
        expectedDType: .uint32
    )
    try turboQuantValidateAttentionStorageArray(
        code.signs,
        name: "compressed attention signs",
        expectedShapes: code.role == .value ? [compactUnusedBitsetShape] : [bitsetShape],
        expectedDType: .uint32
    )
    try turboQuantValidateAttentionStorageArray(
        code.highPrecisionMask,
        name: "compressed attention high precision mask",
        expectedShapes: code.role == .value ? [compactUnusedBitsetShape] : [bitsetShape],
        expectedDType: .uint32
    )
    try turboQuantValidateAttentionStorageArray(
        code.residualSigns,
        name: "compressed attention residual signs",
        expectedShapes: [compactUnusedBitsetShape],
        expectedDType: .uint32
    )
    try turboQuantValidateAttentionStorageArray(
        code.scales,
        name: "compressed attention scales",
        expectedShape: scalesShape,
        expectedDTypes: turboQuantAttentionSupportedScaleDTypes(
            layoutVersion: code.layout.layoutVersion
        )
    )
}

func turboQuantValidateAttentionLayoutBasics(
    _ layout: TurboQuantAttentionLayout,
    context: String
) throws {
    guard TurboQuantAttentionLayout.supportedVersions.contains(layout.layoutVersion) else {
        throw turboQuantAttentionValidationError(
            "\(context) layout version actual \(layout.layoutVersion), "
                + "expected one of \(TurboQuantAttentionLayout.supportedVersions)"
        )
    }
    guard layout.batchSize > 0 else {
        throw turboQuantAttentionValidationError(
            "\(context) batch size actual \(layout.batchSize), expected greater than 0"
        )
    }
    guard layout.kvHeadCount > 0 else {
        throw turboQuantAttentionValidationError(
            "\(context) KV head count actual \(layout.kvHeadCount), expected greater than 0"
        )
    }
    guard layout.capacity > 0 else {
        throw turboQuantAttentionValidationError(
            "\(context) capacity actual \(layout.capacity), expected greater than 0"
        )
    }
    guard (0 ... layout.capacity).contains(layout.logicalLength) else {
        throw turboQuantAttentionValidationError(
            "\(context) logical length actual \(layout.logicalLength), "
                + "expected 0...\(layout.capacity)"
        )
    }
    guard (0 ..< layout.capacity).contains(layout.ringOffset) else {
        throw turboQuantAttentionValidationError(
            "\(context) ring offset actual \(layout.ringOffset), "
                + "expected 0..<\(layout.capacity)"
        )
    }
    guard (0 ... layout.capacity).contains(layout.pinnedPrefixLength) else {
        throw turboQuantAttentionValidationError(
            "\(context) pinned prefix length actual \(layout.pinnedPrefixLength), "
                + "expected 0...\(layout.capacity)"
        )
    }
    guard layout.pinnedPrefixLength <= layout.logicalLength else {
        throw turboQuantAttentionValidationError(
            "\(context) pinned prefix length actual \(layout.pinnedPrefixLength), "
                + "expected <= logical length \(layout.logicalLength)"
        )
    }

    let rotatingCapacity = layout.capacity - layout.pinnedPrefixLength
    if rotatingCapacity == 0 {
        guard layout.ringOffset == 0 else {
            throw turboQuantAttentionValidationError(
                "\(context) ring offset actual \(layout.ringOffset), "
                    + "expected 0 when rotating capacity is 0"
            )
        }
    } else {
        guard layout.ringOffset < rotatingCapacity else {
            throw turboQuantAttentionValidationError(
                "\(context) ring offset actual \(layout.ringOffset), "
                    + "expected < rotating capacity \(rotatingCapacity)"
            )
        }
    }

    guard (1 ... 512).contains(layout.headDimension) else {
        throw turboQuantAttentionValidationError(
            "\(context) head dimension actual \(layout.headDimension), expected 1...512"
        )
    }
    guard layout.groupsPerVector > 0 else {
        throw turboQuantAttentionValidationError(
            "\(context) groups per vector actual \(layout.groupsPerVector), "
                + "expected greater than 0"
        )
    }
    guard layout.magnitudeWordsPerGroup > 0 else {
        throw turboQuantAttentionValidationError(
            "\(context) magnitude words per group actual \(layout.magnitudeWordsPerGroup), "
                + "expected greater than 0"
        )
    }
    guard layout.bitsetWordsPerGroup > 0 else {
        throw turboQuantAttentionValidationError(
            "\(context) bitset words per group actual \(layout.bitsetWordsPerGroup), "
                + "expected greater than 0"
        )
    }
}

func turboQuantValidateAttentionLayoutDescriptor(
    _ layout: TurboQuantAttentionLayout,
    role: TurboQuantTensorRole,
    groupSize: Int
) throws {
    guard role == .key || role == .value else {
        throw turboQuantAttentionValidationError(
            "tensor role actual \(role.rawValue), expected key or value"
        )
    }

    try turboQuantValidateAttentionLayoutBasics(layout, context: "compressed attention")

    guard groupSize > 0 else {
        throw turboQuantAttentionValidationError(
            "group size actual \(groupSize), expected 32, 64, 96, or 128"
        )
    }
    guard groupSize <= 128, groupSize % 32 == 0 else {
        throw turboQuantAttentionValidationError(
            "group size actual \(groupSize), expected 32, 64, 96, or 128"
        )
    }

    let expectedGroupsPerVector = turboQuantAttentionCeilDivide(layout.headDimension, by: groupSize)
    guard layout.groupsPerVector == expectedGroupsPerVector else {
        throw turboQuantAttentionValidationError(
            "groups per vector actual \(layout.groupsPerVector), "
                + "expected \(expectedGroupsPerVector) for head dimension "
                + "\(layout.headDimension) and group size \(groupSize)"
        )
    }
}

func turboQuantAttentionLayoutsShareSequence(
    _ keyLayout: TurboQuantAttentionLayout,
    _ valueLayout: TurboQuantAttentionLayout
) -> Bool {
    keyLayout.layoutVersion == valueLayout.layoutVersion
        && keyLayout.batchSize == valueLayout.batchSize
        && keyLayout.kvHeadCount == valueLayout.kvHeadCount
        && keyLayout.capacity == valueLayout.capacity
        && keyLayout.logicalLength == valueLayout.logicalLength
        && keyLayout.ringOffset == valueLayout.ringOffset
        && keyLayout.pinnedPrefixLength == valueLayout.pinnedPrefixLength
}

func turboQuantAttentionValidationError(_ message: String) -> TurboQuantError {
    .invalidMetalConfiguration(message)
}

private func turboQuantValidateAttentionValueBits(_ bits: Int) throws {
    guard (2 ... 8).contains(bits) else {
        throw turboQuantAttentionValidationError(
            "value bits actual \(bits), expected 2...8"
        )
    }
}

private func turboQuantValidateAttentionStorageArray(
    _ array: MLXArray,
    name: String,
    expectedShape: [Int],
    expectedDType: DType
) throws {
    try turboQuantValidateAttentionStorageArray(
        array,
        name: name,
        expectedShapes: [expectedShape],
        expectedDType: expectedDType
    )
}

private func turboQuantValidateAttentionStorageArray(
    _ array: MLXArray,
    name: String,
    expectedShapes: [[Int]],
    expectedDType: DType
) throws {
    try turboQuantValidateAttentionStorageArray(
        array,
        name: name,
        expectedShapes: expectedShapes,
        expectedDTypes: [expectedDType]
    )
}

private func turboQuantValidateAttentionStorageArray(
    _ array: MLXArray,
    name: String,
    expectedShape: [Int],
    expectedDTypes: [DType]
) throws {
    try turboQuantValidateAttentionStorageArray(
        array,
        name: name,
        expectedShapes: [expectedShape],
        expectedDTypes: expectedDTypes
    )
}

private func turboQuantValidateAttentionStorageArray(
    _ array: MLXArray,
    name: String,
    expectedShapes: [[Int]],
    expectedDTypes: [DType]
) throws {
    guard expectedShapes.contains(array.shape) else {
        throw turboQuantAttentionValidationError(
            "\(name) shape actual \(array.shape), expected one of \(expectedShapes)"
        )
    }
    guard expectedDTypes.contains(array.dtype) else {
        throw turboQuantAttentionValidationError(
            "\(name) dtype actual \(array.dtype), expected one of \(expectedDTypes)"
        )
    }
    let expectedByteCount = array.shape.reduce(1, *) * array.dtype.size
    guard array.nbytes == expectedByteCount else {
        throw turboQuantAttentionValidationError(
            "\(name) storage bytes actual \(array.nbytes), expected \(expectedByteCount)"
        )
    }
    guard array.contiguousToDimension() == 0 else {
        throw turboQuantAttentionValidationError(
            "\(name) layout actual non-contiguous, expected canonical row-contiguous storage"
        )
    }
}

private func turboQuantAttentionSupportedScaleDTypes(layoutVersion: Int) -> [DType] {
    layoutVersion == TurboQuantAttentionLayout.currentVersion
        ? [.float32, .float16]
        : [.float32]
}

private func turboQuantAttentionMagnitudeWordsPerGroup(
    groupSize: Int,
    preset: TurboQuantPreset,
    role: TurboQuantTensorRole,
    valueBits: Int
) -> Int {
    if role == .value {
        return turboQuantAttentionCeilDivide(groupSize * Swift.max(1, valueBits), by: 32)
    }

    let baseBits = Swift.max(1, preset.baseMagnitudeBits - 1)
    let highBits = Swift.max(baseBits, preset.highMagnitudeBits - 1)
    let highCount = turboQuantAttentionHighCount(
        valueCount: groupSize,
        baseBits: baseBits,
        highBits: highBits,
        targetBits: Swift.max(1, preset.targetMagnitudeBits - 1)
    )
    let bitCount = groupSize * baseBits + highCount * (highBits - baseBits)
    return turboQuantAttentionCeilDivide(bitCount, by: 32)
}

private func turboQuantAttentionHighCount(
    valueCount: Int,
    baseBits: Int,
    highBits: Int,
    targetBits: Float
) -> Int {
    guard valueCount > 0, highBits > baseBits else { return 0 }
    let fraction = (targetBits - Float(baseBits)) / Float(highBits - baseBits)
    let boundedFraction = Swift.max(0, Swift.min(1, fraction))
    return Int((Float(valueCount) * boundedFraction).rounded())
}

private func turboQuantAttentionScalesPerGroup(role: TurboQuantTensorRole) -> Int {
    role == .value ? 2 : 3
}

private func turboQuantAttentionCeilDivide(_ value: Int, by divisor: Int) -> Int {
    guard value > 0 else { return 0 }
    return (value + divisor - 1) / divisor
}
