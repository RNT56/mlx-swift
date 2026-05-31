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
        XCTAssertEqual(decoded.metrics.route, TurboQuantBenchmarkRoute.compressedFused.rawValue)
        XCTAssertEqual(decoded.metrics.runtimeMode, "capacityTurboQuant")
        XCTAssertEqual(decoded.metrics.backend, TurboQuantBenchmarkBackend.swiftMetalKernel.rawValue)
        XCTAssertEqual(decoded.metrics.kernelFlags?.tqCoopEnabled, true)
        XCTAssertEqual(decoded.metrics.kernelFlags?.blockTokenSize, 512)
        XCTAssertEqual(decoded.metrics.kernelFlags?.gqaSpecialization, "gqa4")
        XCTAssertEqual(decoded.metrics.kernelFlags?.outputDType, "float32")
        XCTAssertEqual(decoded.metrics.contextTokens, 256)
        XCTAssertEqual(decoded.metrics.layoutVersion, TurboQuantAttentionLayout.currentVersion)
        XCTAssertEqual(decoded.metrics.scaleStorage, TurboQuantScaleStorage.float32.rawValue)
        XCTAssertEqual(decoded.metrics.hotTokens, 128)
        XCTAssertEqual(decoded.metrics.selectedColdTokens, 64)
        XCTAssertEqual(decoded.metrics.coldBudgetTokens, 128)
        XCTAssertEqual(decoded.metrics.selectorConfidence, 0.75)
        XCTAssertEqual(decoded.metrics.selectedBudgetedColdTokens, 32)
        XCTAssertEqual(decoded.metrics.anchorColdTokens, 32)
        XCTAssertEqual(decoded.metrics.anchorOverflowTokens, 0)
        XCTAssertEqual(decoded.metrics.maxColdBudgetTokens, 256)
        XCTAssertEqual(decoded.metrics.selectorInitialConfidence, 0.5)
        XCTAssertEqual(decoded.metrics.selectorFinalConfidence, 0.75)
        XCTAssertEqual(decoded.metrics.selectorEscalation, "maxBudget")
        XCTAssertEqual(
            decoded.metrics.selectorReasonFlags,
            ["anchor", "nearest", "max_budget_escalation"]
        )
        XCTAssertEqual(decoded.metrics.compressedKVBytes, decoded.metrics.totalBytes)
        XCTAssertEqual(decoded.metrics.plainDecodeTokensPerSecondP50, 200)
        XCTAssertEqual(decoded.metrics.plainDecodeTokensPerSecondP95, 180)
        XCTAssertEqual(decoded.metrics.speedRatioToPlainP50, 0.5)
        XCTAssertEqual(decoded.metrics.memoryReductionRatio, 4)
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
        XCTAssertFalse(kernelNames.contains("layout V5 fp16 scales"))
        XCTAssertTrue(audit.entries.allSatisfy { !$0.status.isEmpty })
    }

    func testCurrentW5HiddenCopyAuditAddsLayoutV5ScalePath() {
        let audit = TurboQuantHiddenCopyAudit.currentW5
        let kernelNames = Set(audit.entries.map(\.kernelName))

        XCTAssertEqual(audit.status, .pass)
        XCTAssertTrue(kernelNames.contains("layout V5 fp16 scales"))
        XCTAssertTrue(kernelNames.contains("block-parallel fused"))
        XCTAssertTrue(kernelNames.contains("GQA block-parallel fused"))
        XCTAssertTrue(kernelNames.contains("segmented hybrid attention"))
        XCTAssertTrue(audit.notes.contains { $0.contains("Layout V5") })
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
                route: TurboQuantBenchmarkRoute.compressedFused.rawValue,
                runtimeMode: "capacityTurboQuant",
                backend: TurboQuantBenchmarkBackend.swiftMetalKernel.rawValue,
                kernelFlags: TurboQuantBenchmarkKernelFlags(
                    tqCoopEnabled: true,
                    blockTokenSize: 512,
                    gqaSpecialization: "gqa4",
                    outputDType: "float32"
                ),
                contextTokens: 256,
                headDimension: 128,
                queryLength: 1,
                preset: TurboQuantPreset.turbo4v2.rawValue,
                valueBits: 4,
                groupSize: 64,
                layoutVersion: TurboQuantAttentionLayout.currentVersion,
                scaleStorage: TurboQuantScaleStorage.float32.rawValue,
                hotTokens: 128,
                coldBlockCount: 2,
                selectedColdTokens: 64,
                coldBudgetTokens: 128,
                selectorConfidence: 0.75,
                selectedBudgetedColdTokens: 32,
                anchorColdTokens: 32,
                anchorOverflowTokens: 0,
                maxColdBudgetTokens: 256,
                selectorInitialConfidence: 0.5,
                selectorFinalConfidence: 0.75,
                selectorEscalation: "maxBudget",
                selectorReasonFlags: ["anchor", "nearest", "max_budget_escalation"],
                fullScanFallbackCount: 0,
                warmupIterations: 1,
                qkMS: 0.4,
                avMS: 0.5,
                decodeTokensPerSecondP50: 100,
                decodeTokensPerSecondP95: 90,
                plainAttentionLatencyMSP50: 5,
                plainAttentionLatencyMSP95: 5.5,
                plainDecodeTokensPerSecondP50: 200,
                plainDecodeTokensPerSecondP95: 180,
                speedRatioToPlainP50: 0.5,
                speedRatioToPlainP95: 0.5,
                totalBytes: 112,
                plainKVBytes: 448,
                memoryReductionRatio: 4,
                actualBitsPerValue: 3.5
            ),
            hiddenCopyAudit: TurboQuantHiddenCopyAudit.currentW5
        )
    }
}
