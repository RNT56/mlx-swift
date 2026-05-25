// Copyright © 2026 RNT56.

import Foundation
import MLX
import XCTest

final class TurboQuantBenchmarkReportTests: XCTestCase {
    func testCoreBenchmarkReportRoundTripsRequiredSchemaFields() throws {
        let report = Self.sampleReport()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]

        let data = try encoder.encode(report)
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )

        XCTAssertEqual(object["schemaVersion"] as? Int, 1)
        XCTAssertNotNil(object["mlxSwiftCommit"])
        XCTAssertNotNil(object["capabilities"])
        XCTAssertNotNil(object["storageEstimate"])
        XCTAssertNotNil(object["pathDecision"])
        XCTAssertNotNil(object["metrics"])
        XCTAssertNotNil(object["hiddenCopyAudit"])

        let decoded = try JSONDecoder().decode(TurboQuantCoreBenchmarkReport.self, from: data)
        XCTAssertEqual(decoded.schemaVersion, TurboQuantCoreBenchmarkReport.currentSchemaVersion)
        XCTAssertEqual(decoded.mlxSwiftCommit, "abcdef123456")
        XCTAssertEqual(decoded.storageEstimate.totalBytes, 112)
        XCTAssertEqual(decoded.pathDecision?.selectedPath, .twoStageCompressed)
        XCTAssertEqual(decoded.metrics.contextTokens, 256)
        XCTAssertEqual(decoded.metrics.compressedKVBytes, decoded.metrics.totalBytes)
        XCTAssertEqual(decoded.hiddenCopyAudit.status, .pass)
    }

    func testMissingRequiredStorageEstimateFailsDecode() throws {
        let json = """
            {
              "schemaVersion": 1,
              "mlxSwiftCommit": "abcdef123456",
              "capabilities": {
                "flatEncodeDecode": false,
                "linearMatmul": false,
                "attentionEncode": true,
                "attentionDecode": true,
                "attentionQK": true,
                "attentionAV": true,
                "attentionFusedDecode": false,
                "bfloatOutput": false
              },
              "pathDecision": {
                "selectedPath": "twoStageCompressed",
                "outputDType": "float32",
                "estimatedScratchBytes": 4096,
                "rejectedPaths": []
              },
              "metrics": {
                "contextTokens": 256,
                "headDimension": 128,
                "queryLength": 1,
                "preset": "turbo4v2",
                "valueBits": 4,
                "groupSize": 64,
                "totalBytes": 112,
                "compressedKVBytes": 112,
                "actualBitsPerValue": 3.5,
                "fallbackUsed": false,
                "memoryWarningsSeen": 0,
                "jetsamObserved": false
              },
              "hiddenCopyAudit": {
                "status": "pass",
                "entries": [],
                "notes": []
              }
            }
            """

        XCTAssertThrowsError(
            try JSONDecoder().decode(
                TurboQuantCoreBenchmarkReport.self,
                from: Data(json.utf8)
            )
        )
    }

    func testSelectedAndRejectedPathsAreEncoded() throws {
        let report = Self.sampleReport()
        let data = try JSONEncoder().encode(report)
        let decoded = try JSONDecoder().decode(TurboQuantCoreBenchmarkReport.self, from: data)

        XCTAssertEqual(decoded.pathDecision?.selectedPath, .twoStageCompressed)
        XCTAssertEqual(decoded.pathDecision?.rejectedPaths.count, 1)
        XCTAssertEqual(decoded.pathDecision?.rejectedPaths.first?.path, .onlineFused)
        XCTAssertFalse(decoded.pathDecision?.rejectedPaths.first?.reason.isEmpty ?? true)
    }

    func testCurrentHiddenCopyAuditListsAllBenchmarkKernels() {
        let audit = TurboQuantHiddenCopyAudit.currentW3
        let kernelNames = Set(audit.entries.map(\.kernelName))

        XCTAssertEqual(audit.status, .pass)
        XCTAssertTrue(kernelNames.contains("encode flat"))
        XCTAssertTrue(kernelNames.contains("decode flat"))
        XCTAssertTrue(kernelNames.contains("compressed QK"))
        XCTAssertTrue(kernelNames.contains("compressed AV"))
        XCTAssertTrue(kernelNames.contains("online fused"))
        XCTAssertTrue(kernelNames.contains("tiled fused"))
        XCTAssertTrue(audit.entries.allSatisfy { !$0.status.isEmpty })
    }

    private static func sampleReport() -> TurboQuantCoreBenchmarkReport {
        TurboQuantCoreBenchmarkReport(
            mlxSwiftCommit: "abcdef123456",
            capabilities: TurboQuantKernelCapabilities(
                flatEncodeDecode: false,
                linearMatmul: false,
                attentionEncode: true,
                attentionDecode: true,
                attentionQK: true,
                attentionAV: true,
                attentionFusedDecode: false,
                bfloatOutput: false
            ),
            storageEstimate: TurboQuantStorageEstimate(
                role: .key,
                logicalValues: 128,
                packedBytes: 40,
                bitsetBytes: 48,
                scaleBytes: 24
            ),
            pathDecision: TurboQuantAttentionDecision(
                selectedPath: .twoStageCompressed,
                outputDType: .float32,
                estimatedScratchBytes: 4096,
                rejectedPaths: [
                    RejectedPath(path: .onlineFused, reason: "not certified for this device")
                ]
            ),
            metrics: TurboQuantCoreBenchmarkMetrics(
                contextTokens: 256,
                headDimension: 128,
                queryLength: 1,
                preset: TurboQuantPreset.turbo4v2.rawValue,
                valueBits: 4,
                groupSize: 64,
                qkMS: 0.4,
                avMS: 0.5,
                totalBytes: 112,
                actualBitsPerValue: 3.5
            ),
            hiddenCopyAudit: TurboQuantHiddenCopyAudit.currentW3
        )
    }
}
