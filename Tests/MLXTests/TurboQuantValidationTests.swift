// Copyright © 2026 RNT56.

import MLX
import XCTest

final class TurboQuantValidationTests: XCTestCase {
    func testValidKeyAndValueCodesPassValidation() throws {
        try validateTurboQuantAttentionCode(Self.makeCode(role: .key), expectedRole: .key)
        try validateTurboQuantAttentionCode(Self.makeCode(role: .value), expectedRole: .value)
    }

    func testInvalidLayoutFailsBeforeDispatchWithExpectedAndActualValues() {
        var code = Self.makeCode(role: .key)
        code.layout.logicalLength = code.layout.capacity + 1

        XCTAssertThrowsError(try validateTurboQuantAttentionCode(code, expectedRole: .key)) { error in
            let message = String(describing: error)
            XCTAssertTrue(message.contains("logical length actual 3"))
            XCTAssertTrue(message.contains("expected 0...2"))
        }
    }

    func testExpectedRoleMismatchNamesActualAndExpectedRoles() {
        let code = Self.makeCode(role: .key)

        XCTAssertThrowsError(try validateTurboQuantAttentionCode(code, expectedRole: .value)) { error in
            let message = String(describing: error)
            XCTAssertTrue(message.contains("actual key"))
            XCTAssertTrue(message.contains("expected value"))
        }
    }

    func testWritableCapacityRequirementRejectsFullCache() {
        var code = Self.makeCode(role: .value)
        code.layout.logicalLength = code.layout.capacity

        XCTAssertThrowsError(
            try validateTurboQuantAttentionCode(
                code,
                expectedRole: .value,
                requireWritableCapacity: true
            )
        ) { error in
            XCTAssertTrue(String(describing: error).contains("writable capacity actual 0"))
        }
    }

    func testKeyResidualSignsUseCompactUnusedStorage() {
        var code = Self.makeCode(role: .key)
        code.residualSigns = MLXArray.zeros([1, 1, 2, 1, 2], dtype: .uint32)

        XCTAssertThrowsError(try validateTurboQuantAttentionCode(code, expectedRole: .key)) { error in
            let message = String(describing: error)
            XCTAssertTrue(message.contains("compressed attention residual signs"))
            XCTAssertTrue(message.contains("expected one of [[1]]"))
        }
    }

    func testLayoutV5AcceptsFp16ScaleStorage() throws {
        var code = Self.makeCode(role: .key)
        code.layout.layoutVersion = TurboQuantAttentionLayout.currentVersion
        code.scales = MLXArray.zeros([1, 1, 2, 1, 3], dtype: .float16)

        try validateTurboQuantAttentionCode(code, expectedRole: .key)
    }

    func testLayoutV4RejectsFp16ScaleStorage() {
        var code = Self.makeCode(role: .key)
        code.layout.layoutVersion = TurboQuantAttentionLayout.legacyVersion
        code.scales = MLXArray.zeros([1, 1, 2, 1, 3], dtype: .float16)

        XCTAssertThrowsError(try validateTurboQuantAttentionCode(code, expectedRole: .key)) {
            error in
            let message = String(describing: error)
            XCTAssertTrue(message.contains("compressed attention scales"))
            XCTAssertTrue(message.contains("float32"))
        }
    }

    private static func makeCode(role: TurboQuantTensorRole) -> TurboQuantAttentionCode {
        let layout = TurboQuantAttentionLayout(
            batchSize: 1,
            kvHeadCount: 1,
            capacity: 2,
            logicalLength: 1,
            headDimension: 64,
            groupsPerVector: 1,
            magnitudeWordsPerGroup: role == .value ? 8 : 5,
            bitsetWordsPerGroup: 2
        )
        let bitset = role == .value
            ? MLXArray.zeros([1], dtype: .uint32)
            : MLXArray.zeros([1, 1, 2, 1, 2], dtype: .uint32)
        let compactUnusedBitset = MLXArray.zeros([1], dtype: .uint32)
        return TurboQuantAttentionCode(
            layout: layout,
            preset: role == .value ? .turbo4v2 : .turbo3_5,
            role: role,
            groupSize: 64,
            seed: 0,
            valueBits: 4,
            packedMagnitudes: MLXArray.zeros(
                [1, 1, 2, 1, role == .value ? 8 : 5],
                dtype: .uint32
            ),
            signs: bitset,
            highPrecisionMask: bitset,
            residualSigns: compactUnusedBitset,
            scales: MLXArray.zeros([1, 1, 2, 1, role == .value ? 2 : 3], dtype: .float32)
        )
    }
}
