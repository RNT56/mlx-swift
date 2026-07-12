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

    func testLayoutV6KeyResidualSignsUseSplitMagnitudePlane() {
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
        // K scale plane dieted to 2 scales/group (T1.4 stage 1); was 3.
        code.scales = MLXArray.zeros([1, 1, 2, 1, 2], dtype: .float16)

        try validateTurboQuantAttentionCode(code, expectedRole: .key)
    }

    func testDefaultAttentionLayoutUsesProductionVersion() throws {
        let layout = try turboQuantAttentionLayout(
            shape: [1, 1, 2, 64],
            dtype: .float16
        )

        XCTAssertEqual(layout.layoutVersion, TurboQuantAttentionLayout.productionDefaultVersion)
        XCTAssertEqual(layout.layoutVersion, TurboQuantAttentionLayout.currentVersion)
    }

    func testPolarWHTBackendsArePublicAndFailClosedByDefault() throws {
        let decodedBackend = try JSONDecoder().decode(
            TurboQuantBackend.self,
            from: Data(#""metalPolarWHT""#.utf8)
        )
        let defaultAvailability = TurboQuantKernelAvailability()
        let availablePolarWHT = TurboQuantKernelAvailability(
            supportsPolarWHTReference: true,
            supportsMetalPolarWHTCodec: true,
            supportsMetalPolarWHTAttention: true,
            supportsMetalPolarWHT: true
        )

        XCTAssertEqual(decodedBackend, .metalPolarWHT)
        XCTAssertTrue(availablePolarWHT.supports(.polarWHTReference))
        XCTAssertTrue(availablePolarWHT.supports(.metalPolarWHT))
        XCTAssertFalse(defaultAvailability.supports(.polarWHTReference))
        XCTAssertFalse(defaultAvailability.supports(.metalPolarWHT))
        XCTAssertEqual(defaultAvailability.runtimeBackend(for: .metalPolarWHT), .mlxPacked)
        XCTAssertTrue(
            defaultAvailability.fallbackReason(for: .metalPolarWHT)?
                .contains("PolarWHT Metal kernels unavailable") == true
        )
    }

    func testPolarWHTBackendRequiresCodecAndAttentionCapabilities() throws {
        let attentionOnly = TurboQuantKernelAvailability(
            supportsPolarWHTReference: true,
            supportsMetalPolarWHTCodec: false,
            supportsMetalPolarWHTAttention: true,
            supportsMetalPolarWHT: false
        )

        XCTAssertTrue(attentionOnly.supports(.polarWHTReference))
        XCTAssertTrue(attentionOnly.supportsMetalPolarWHTAttention)
        XCTAssertFalse(attentionOnly.supportsMetalPolarWHTCodec)
        XCTAssertFalse(attentionOnly.supports(.metalPolarWHT))
        XCTAssertEqual(attentionOnly.runtimeBackend(for: .metalPolarWHT), .mlxPacked)
    }

    func testLayoutV5RequiresExplicitOptInWhileV6IsDefault() throws {
        XCTAssertThrowsError(
            try turboQuantAttentionLayout(
                shape: [1, 1, 2, 64],
                dtype: .float16,
                layoutVersion: 5
            )
        ) { error in
            XCTAssertTrue(String(describing: error).contains("allowExperimentalLayoutV5"))
        }

        let v5Layout = try turboQuantAttentionLayout(
            shape: [1, 1, 2, 64],
            dtype: .float16,
            layoutVersion: 5,
            allowExperimentalLayoutV5: true
        )
        XCTAssertEqual(v5Layout.layoutVersion, 5)

        let currentLayout = try turboQuantAttentionLayout(
            shape: [1, 1, 2, 64],
            dtype: .float16,
            layoutVersion: TurboQuantAttentionLayout.currentVersion
        )
        XCTAssertEqual(currentLayout.layoutVersion, TurboQuantAttentionLayout.currentVersion)
    }

    func testLayoutV4RejectsFp16ScaleStorage() {
        var code = Self.makeCode(role: .key)
        code.layout.layoutVersion = TurboQuantAttentionLayout.legacyVersion
        code.layout.magnitudeWordsPerGroup = 5
        code.packedMagnitudes = MLXArray.zeros([1, 1, 2, 1, 5], dtype: .uint32)
        code.highPrecisionMask = MLXArray.zeros([1, 1, 2, 1, 2], dtype: .uint32)
        code.residualSigns = MLXArray.zeros([1], dtype: .uint32)
        // K scale plane dieted to 2 scales/group (T1.4 stage 1); was 3.
        code.scales = MLXArray.zeros([1, 1, 2, 1, 2], dtype: .float16)

        XCTAssertThrowsError(try validateTurboQuantAttentionCode(code, expectedRole: .key)) {
            error in
            let message = String(describing: error)
            XCTAssertTrue(message.contains("compressed attention scales"))
            XCTAssertTrue(message.contains("float32"))
        }
    }

    private static func makeCode(role: TurboQuantTensorRole) -> TurboQuantAttentionCode {
        let layout = TurboQuantAttentionLayout(
            layoutVersion: TurboQuantAttentionLayout.currentVersion,
            batchSize: 1,
            kvHeadCount: 1,
            capacity: 2,
            logicalLength: 1,
            headDimension: 64,
            groupsPerVector: 1,
            magnitudeWordsPerGroup: role == .value ? 8 : 5,
            bitsetWordsPerGroup: 2
        )
        let signs = role == .value
            ? MLXArray.zeros([1], dtype: .uint32)
            : MLXArray.zeros([1, 1, 2, 1, 2], dtype: .uint32)
        let compact = MLXArray.zeros([1], dtype: .uint32)
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
            signs: signs,
            highPrecisionMask: compact,
            residualSigns: compact,
            scales: MLXArray.zeros([1, 1, 2, 1, 2], dtype: .float32)
        )
    }
}
