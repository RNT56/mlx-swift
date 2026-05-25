// Copyright © 2026 RNT56.

import Foundation
import MLX
import XCTest

final class TurboQuantWave7PlatformPolicyTests: XCTestCase {
    func testWave7ContractsRoundTripWithDisabledDefaults() throws {
        let report = TurboQuantPlatformCapabilityReport.disabled
        let policy = TurboQuantAdaptivePrecisionPolicy.disabled
        let descriptor = TurboQuantOpenKVFormatDescriptor()

        XCTAssertNoThrow(try validateTurboQuantPlatformCapabilityReport(report))
        XCTAssertNoThrow(
            try validateTurboQuantAdaptivePrecisionPolicy(policy, capabilityReport: report)
        )

        let data = try JSONEncoder().encode(
            Wave7Payload(report: report, policy: policy, descriptor: descriptor)
        )
        let decoded = try JSONDecoder().decode(Wave7Payload.self, from: data)

        XCTAssertEqual(decoded.report, report)
        XCTAssertEqual(decoded.policy, policy)
        XCTAssertEqual(decoded.descriptor, descriptor)
        XCTAssertEqual(decoded.report.supportedAttentionLayoutVersions, [4, 5])
        XCTAssertFalse(decoded.policy.enabled)
        XCTAssertFalse(decoded.descriptor.readEnabled)
        XCTAssertFalse(decoded.descriptor.writeEnabled)
    }

    func testFeatureGateFailsClosedWhenEnabledWithoutSupportOrEvidence() {
        XCTAssertThrowsError(
            try validateTurboQuantPlatformFeatureGate(
                TurboQuantPlatformFeatureGate(
                    feature: .adaptivePrecision,
                    supported: false,
                    enabled: true,
                    evidenceLevel: .verified,
                    evidenceID: "evidence-1"
                )
            )
        ) { error in
            XCTAssertTrue(String(describing: error).contains("without platform support"))
        }

        XCTAssertThrowsError(
            try validateTurboQuantPlatformFeatureGate(
                TurboQuantPlatformFeatureGate(
                    feature: .openKVFormat,
                    supported: true,
                    enabled: true,
                    evidenceLevel: .declared,
                    evidenceID: "evidence-1"
                )
            )
        ) { error in
            XCTAssertTrue(
                String(describing: error).contains("without measured or verified evidence"))
        }
    }

    func testAdaptivePrecisionPolicyRequiresEnabledGateAndRegisteredEvidence() throws {
        let segment = TurboQuantPrecisionSegment(
            id: "recent-window",
            role: .keyMagnitude,
            tokenStart: 0,
            tokenEnd: 128,
            baseBits: 3,
            highBits: 4
        )
        let policy = TurboQuantAdaptivePrecisionPolicy(
            enabled: true,
            policyID: "ap-v1",
            segments: [segment],
            evidenceLevel: .measured,
            evidenceID: "ap-evidence"
        )

        XCTAssertThrowsError(
            try validateTurboQuantAdaptivePrecisionPolicy(
                policy,
                capabilityReport: TurboQuantPlatformCapabilityReport.disabled
            )
        ) { error in
            XCTAssertTrue(String(describing: error).contains("adaptivePrecision is disabled"))
        }

        var report = Self.enabledReport(feature: .adaptivePrecision, evidenceID: "ap-evidence")
        report.supportedAdaptivePrecisionPolicyVersions = [
            TurboQuantAdaptivePrecisionPolicy.currentSchemaVersion
        ]
        XCTAssertNoThrow(
            try validateTurboQuantAdaptivePrecisionPolicy(policy, capabilityReport: report))

        var unregisteredEvidenceReport = report
        unregisteredEvidenceReport.evidenceIDs = []
        XCTAssertThrowsError(
            try validateTurboQuantAdaptivePrecisionPolicy(
                policy,
                capabilityReport: unregisteredEvidenceReport
            )
        ) { error in
            XCTAssertTrue(String(describing: error).contains("without registered evidence"))
        }
    }

    func testPrecisionSegmentValidationRejectsInvalidRangesAndBits() {
        XCTAssertThrowsError(
            try validateTurboQuantPrecisionSegment(
                TurboQuantPrecisionSegment(
                    id: "bad-range",
                    role: .valueMagnitude,
                    tokenStart: 8,
                    tokenEnd: 8,
                    baseBits: 4,
                    highBits: 4
                )
            )
        )

        XCTAssertThrowsError(
            try validateTurboQuantPrecisionSegment(
                TurboQuantPrecisionSegment(
                    id: "bad-bits",
                    role: .keyMagnitude,
                    tokenStart: 0,
                    tokenEnd: 16,
                    baseBits: 5,
                    highBits: 4
                )
            )
        )
    }

    func testOpenKVDescriptorRequiresSupportedFormatAndEvidenceBeforeActivation() throws {
        let disabledDescriptor = TurboQuantOpenKVFormatDescriptor(formatVersion: 1)
        let disabledReport = TurboQuantPlatformCapabilityReport.disabled

        XCTAssertThrowsError(
            try validateTurboQuantOpenKVFormatDescriptor(
                disabledDescriptor,
                capabilityReport: disabledReport
            )
        ) { error in
            XCTAssertTrue(String(describing: error).contains("format version 1 is unsupported"))
        }

        var supportedButDisabledReport = disabledReport
        supportedButDisabledReport.supportedOpenKVFormatVersions = [1]
        XCTAssertNoThrow(
            try validateTurboQuantOpenKVFormatDescriptor(
                disabledDescriptor,
                capabilityReport: supportedButDisabledReport
            )
        )

        let activeDescriptor = TurboQuantOpenKVFormatDescriptor(
            formatVersion: 1,
            writeEnabled: true,
            evidenceLevel: .measured,
            evidenceID: "open-kv-evidence"
        )
        XCTAssertThrowsError(
            try validateTurboQuantOpenKVFormatDescriptor(
                activeDescriptor,
                capabilityReport: supportedButDisabledReport
            )
        ) { error in
            XCTAssertTrue(String(describing: error).contains("openKVFormat is disabled"))
        }

        var enabledReport = Self.enabledReport(
            feature: .openKVFormat, evidenceID: "open-kv-evidence")
        enabledReport.supportedOpenKVFormatVersions = [1]
        XCTAssertNoThrow(
            try validateTurboQuantOpenKVFormatDescriptor(
                activeDescriptor,
                capabilityReport: enabledReport
            )
        )
    }

    private struct Wave7Payload: Codable, Equatable {
        var report: TurboQuantPlatformCapabilityReport
        var policy: TurboQuantAdaptivePrecisionPolicy
        var descriptor: TurboQuantOpenKVFormatDescriptor
    }

    private static func enabledReport(
        feature: TurboQuantPlatformFeature,
        evidenceID: String
    ) -> TurboQuantPlatformCapabilityReport {
        TurboQuantPlatformCapabilityReport(
            featureGates: [
                TurboQuantPlatformFeatureGate(
                    feature: feature,
                    supported: true,
                    enabled: true,
                    evidenceLevel: .verified,
                    evidenceID: evidenceID
                )
            ],
            supportedAttentionLayoutVersions: TurboQuantAttentionLayout.supportedVersions,
            evidenceIDs: [evidenceID]
        )
    }
}
