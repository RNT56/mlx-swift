// Copyright © 2026 RNT56.

import Foundation

public enum TurboQuantEvidenceLevel: String, Codable, Sendable, Hashable, CaseIterable {
    case none
    case declared
    case measured
    case verified

    public var permitsActivation: Bool {
        self == .measured || self == .verified
    }
}

public enum TurboQuantPlatformFeature: String, Codable, Sendable, Hashable, CaseIterable {
    case adaptivePrecision
    case openKVFormat
    case platformCapabilityReport
    case layoutV5
}

public struct TurboQuantPlatformFeatureGate: Codable, Sendable, Hashable {
    public var feature: TurboQuantPlatformFeature
    public var supported: Bool
    public var enabled: Bool
    public var evidenceLevel: TurboQuantEvidenceLevel
    public var evidenceID: String?
    public var reason: String?

    public init(
        feature: TurboQuantPlatformFeature,
        supported: Bool = false,
        enabled: Bool = false,
        evidenceLevel: TurboQuantEvidenceLevel = .none,
        evidenceID: String? = nil,
        reason: String? = nil
    ) {
        self.feature = feature
        self.supported = supported
        self.enabled = enabled
        self.evidenceLevel = evidenceLevel
        self.evidenceID = evidenceID
        self.reason = reason
    }

    public static func disabled(
        _ feature: TurboQuantPlatformFeature,
        reason: String? = nil
    ) -> TurboQuantPlatformFeatureGate {
        TurboQuantPlatformFeatureGate(feature: feature, reason: reason)
    }
}

public enum TurboQuantPrecisionRole: String, Codable, Sendable, Hashable, CaseIterable {
    case keyMagnitude
    case valueMagnitude
    case scale
}

public struct TurboQuantPrecisionSegment: Codable, Sendable, Hashable {
    public var id: String
    public var role: TurboQuantPrecisionRole
    public var tokenStart: Int
    public var tokenEnd: Int
    public var baseBits: Int
    public var highBits: Int
    public var scaleStorage: TurboQuantScaleStorage
    public var evidenceID: String?

    public init(
        id: String,
        role: TurboQuantPrecisionRole,
        tokenStart: Int,
        tokenEnd: Int,
        baseBits: Int,
        highBits: Int,
        scaleStorage: TurboQuantScaleStorage = .float32,
        evidenceID: String? = nil
    ) {
        self.id = id
        self.role = role
        self.tokenStart = tokenStart
        self.tokenEnd = tokenEnd
        self.baseBits = baseBits
        self.highBits = highBits
        self.scaleStorage = scaleStorage
        self.evidenceID = evidenceID
    }
}

public struct TurboQuantAdaptivePrecisionPolicy: Codable, Sendable, Hashable {
    public static let currentSchemaVersion = 1

    public var schemaVersion: Int
    public var enabled: Bool
    public var policyID: String
    public var segments: [TurboQuantPrecisionSegment]
    public var evidenceLevel: TurboQuantEvidenceLevel
    public var evidenceID: String?

    public init(
        schemaVersion: Int = TurboQuantAdaptivePrecisionPolicy.currentSchemaVersion,
        enabled: Bool = false,
        policyID: String = "disabled",
        segments: [TurboQuantPrecisionSegment] = [],
        evidenceLevel: TurboQuantEvidenceLevel = .none,
        evidenceID: String? = nil
    ) {
        self.schemaVersion = schemaVersion
        self.enabled = enabled
        self.policyID = policyID
        self.segments = segments
        self.evidenceLevel = evidenceLevel
        self.evidenceID = evidenceID
    }

    public static let disabled = TurboQuantAdaptivePrecisionPolicy()
}

public struct TurboQuantOpenKVFormatDescriptor: Codable, Sendable, Hashable {
    public static let currentSchemaVersion = 1

    public var schemaVersion: Int
    public var formatID: String
    public var formatVersion: Int
    public var layoutVersion: Int
    public var preset: TurboQuantPreset
    public var groupSize: Int
    public var keyScaleStorage: TurboQuantScaleStorage
    public var valueScaleStorage: TurboQuantScaleStorage
    public var supportsRotatingCache: Bool
    public var supportsPinnedPrefix: Bool
    public var writeEnabled: Bool
    public var readEnabled: Bool
    public var evidenceLevel: TurboQuantEvidenceLevel
    public var evidenceID: String?

    public init(
        schemaVersion: Int = TurboQuantOpenKVFormatDescriptor.currentSchemaVersion,
        formatID: String = "turboquant-open-kv",
        formatVersion: Int = 1,
        layoutVersion: Int = TurboQuantAttentionLayout.currentVersion,
        preset: TurboQuantPreset = .turbo4v2,
        groupSize: Int = 64,
        keyScaleStorage: TurboQuantScaleStorage = .float32,
        valueScaleStorage: TurboQuantScaleStorage = .float32,
        supportsRotatingCache: Bool = true,
        supportsPinnedPrefix: Bool = true,
        writeEnabled: Bool = false,
        readEnabled: Bool = false,
        evidenceLevel: TurboQuantEvidenceLevel = .none,
        evidenceID: String? = nil
    ) {
        self.schemaVersion = schemaVersion
        self.formatID = formatID
        self.formatVersion = formatVersion
        self.layoutVersion = layoutVersion
        self.preset = preset
        self.groupSize = groupSize
        self.keyScaleStorage = keyScaleStorage
        self.valueScaleStorage = valueScaleStorage
        self.supportsRotatingCache = supportsRotatingCache
        self.supportsPinnedPrefix = supportsPinnedPrefix
        self.writeEnabled = writeEnabled
        self.readEnabled = readEnabled
        self.evidenceLevel = evidenceLevel
        self.evidenceID = evidenceID
    }
}

public struct TurboQuantPlatformCapabilityReport: Codable, Sendable, Hashable {
    public static let currentSchemaVersion = 1

    public var schemaVersion: Int
    public var kernelCapabilities: TurboQuantKernelCapabilities
    public var featureGates: [TurboQuantPlatformFeatureGate]
    public var supportedOpenKVFormatVersions: [Int]
    public var supportedAdaptivePrecisionPolicyVersions: [Int]
    public var supportedAttentionLayoutVersions: [Int]
    public var evidenceIDs: [String]
    public var notes: [String]

    public init(
        schemaVersion: Int = TurboQuantPlatformCapabilityReport.currentSchemaVersion,
        kernelCapabilities: TurboQuantKernelCapabilities = TurboQuantKernelCapabilities(),
        featureGates: [TurboQuantPlatformFeatureGate] = TurboQuantPlatformFeature.allCases.map {
            TurboQuantPlatformFeatureGate.disabled($0)
        },
        supportedOpenKVFormatVersions: [Int] = [],
        supportedAdaptivePrecisionPolicyVersions: [Int] = [],
        supportedAttentionLayoutVersions: [Int] = TurboQuantAttentionLayout.supportedVersions,
        evidenceIDs: [String] = [],
        notes: [String] = []
    ) {
        self.schemaVersion = schemaVersion
        self.kernelCapabilities = kernelCapabilities
        self.featureGates = featureGates
        self.supportedOpenKVFormatVersions = supportedOpenKVFormatVersions
        self.supportedAdaptivePrecisionPolicyVersions = supportedAdaptivePrecisionPolicyVersions
        self.supportedAttentionLayoutVersions = supportedAttentionLayoutVersions
        self.evidenceIDs = evidenceIDs
        self.notes = notes
    }

    public static let disabled = TurboQuantPlatformCapabilityReport()

    public func gate(for feature: TurboQuantPlatformFeature) -> TurboQuantPlatformFeatureGate? {
        featureGates.first { $0.feature == feature }
    }
}

public func validateTurboQuantPlatformCapabilityReport(
    _ report: TurboQuantPlatformCapabilityReport
) throws {
    guard report.schemaVersion == TurboQuantPlatformCapabilityReport.currentSchemaVersion else {
        throw turboQuantPlatformPolicyError(
            "platform capability report schema version \(report.schemaVersion) is unsupported"
        )
    }

    for gate in report.featureGates {
        try validateTurboQuantPlatformFeatureGate(gate)
    }
}

public func validateTurboQuantPlatformFeatureGate(
    _ gate: TurboQuantPlatformFeatureGate
) throws {
    guard !gate.enabled || gate.supported else {
        throw turboQuantPlatformPolicyError(
            "\(gate.feature.rawValue) is enabled without platform support"
        )
    }
    guard !gate.enabled || gate.evidenceLevel.permitsActivation else {
        throw turboQuantPlatformPolicyError(
            "\(gate.feature.rawValue) is enabled without measured or verified evidence"
        )
    }
    guard !gate.enabled || hasTurboQuantEvidenceID(gate.evidenceID) else {
        throw turboQuantPlatformPolicyError(
            "\(gate.feature.rawValue) is enabled without an evidence ID"
        )
    }
}

public func validateTurboQuantAdaptivePrecisionPolicy(
    _ policy: TurboQuantAdaptivePrecisionPolicy,
    capabilityReport report: TurboQuantPlatformCapabilityReport
) throws {
    try validateTurboQuantPlatformCapabilityReport(report)

    guard policy.schemaVersion == TurboQuantAdaptivePrecisionPolicy.currentSchemaVersion else {
        throw turboQuantPlatformPolicyError(
            "adaptive precision policy schema version \(policy.schemaVersion) is unsupported"
        )
    }
    guard policy.enabled else { return }

    try validateTurboQuantFeatureActive(
        .adaptivePrecision,
        evidenceLevel: policy.evidenceLevel,
        evidenceID: policy.evidenceID,
        capabilityReport: report
    )
    guard report.supportedAdaptivePrecisionPolicyVersions.contains(policy.schemaVersion) else {
        throw turboQuantPlatformPolicyError(
            "adaptive precision policy version \(policy.schemaVersion) is unsupported"
        )
    }
    guard !policy.segments.isEmpty else {
        throw turboQuantPlatformPolicyError(
            "adaptive precision policy is enabled without precision segments"
        )
    }

    for segment in policy.segments {
        try validateTurboQuantPrecisionSegment(segment)
    }
}

public func validateTurboQuantPrecisionSegment(
    _ segment: TurboQuantPrecisionSegment
) throws {
    guard !segment.id.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
        throw turboQuantPlatformPolicyError("precision segment ID is empty")
    }
    guard segment.tokenStart >= 0, segment.tokenEnd > segment.tokenStart else {
        throw turboQuantPlatformPolicyError(
            "precision segment \(segment.id) token range is invalid"
        )
    }
    guard (2 ... 8).contains(segment.baseBits), (2 ... 8).contains(segment.highBits),
        segment.highBits >= segment.baseBits
    else {
        throw turboQuantPlatformPolicyError(
            "precision segment \(segment.id) bits must be 2...8 with highBits >= baseBits"
        )
    }
}

public func validateTurboQuantOpenKVFormatDescriptor(
    _ descriptor: TurboQuantOpenKVFormatDescriptor,
    capabilityReport report: TurboQuantPlatformCapabilityReport
) throws {
    try validateTurboQuantPlatformCapabilityReport(report)

    guard descriptor.schemaVersion == TurboQuantOpenKVFormatDescriptor.currentSchemaVersion else {
        throw turboQuantPlatformPolicyError(
            "open KV descriptor schema version \(descriptor.schemaVersion) is unsupported"
        )
    }
    guard descriptor.formatVersion > 0,
        report.supportedOpenKVFormatVersions.contains(descriptor.formatVersion)
    else {
        throw turboQuantPlatformPolicyError(
            "open KV format version \(descriptor.formatVersion) is unsupported"
        )
    }
    guard report.supportedAttentionLayoutVersions.contains(descriptor.layoutVersion) else {
        throw turboQuantPlatformPolicyError(
            "open KV layout version \(descriptor.layoutVersion) is unsupported"
        )
    }
    guard descriptor.groupSize > 0, descriptor.groupSize <= 128,
        descriptor.groupSize % 32 == 0
    else {
        throw turboQuantPlatformPolicyError(
            "open KV group size \(descriptor.groupSize) is unsupported"
        )
    }

    guard descriptor.readEnabled || descriptor.writeEnabled else { return }
    try validateTurboQuantFeatureActive(
        .openKVFormat,
        evidenceLevel: descriptor.evidenceLevel,
        evidenceID: descriptor.evidenceID,
        capabilityReport: report
    )
}

private func validateTurboQuantFeatureActive(
    _ feature: TurboQuantPlatformFeature,
    evidenceLevel: TurboQuantEvidenceLevel,
    evidenceID: String?,
    capabilityReport report: TurboQuantPlatformCapabilityReport
) throws {
    guard let gate = report.gate(for: feature) else {
        throw turboQuantPlatformPolicyError("\(feature.rawValue) gate is missing")
    }
    guard gate.enabled else {
        throw turboQuantPlatformPolicyError("\(feature.rawValue) is disabled")
    }
    guard evidenceLevel.permitsActivation else {
        throw turboQuantPlatformPolicyError(
            "\(feature.rawValue) is active without measured or verified evidence"
        )
    }
    guard hasTurboQuantEvidenceID(evidenceID),
        evidenceID.map(report.evidenceIDs.contains) == true
    else {
        throw turboQuantPlatformPolicyError(
            "\(feature.rawValue) is active without registered evidence"
        )
    }
}

private func hasTurboQuantEvidenceID(_ evidenceID: String?) -> Bool {
    guard let evidenceID else { return false }
    return !evidenceID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
}

private func turboQuantPlatformPolicyError(_ message: String) -> TurboQuantError {
    .invalidMetalConfiguration("TurboQuant platform policy: \(message)")
}
