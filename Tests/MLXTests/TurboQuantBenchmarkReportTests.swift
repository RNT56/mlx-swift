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
        XCTAssertNotNil(object["pathMeasurements"])
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
        XCTAssertEqual(decoded.metrics.sparseVEnabled, true)
        XCTAssertEqual(decoded.metrics.sparseVSelectionMode, .hybridCumulativeMassTopK)
        XCTAssertEqual(decoded.metrics.sparseVThreshold, 0.001)
        XCTAssertEqual(decoded.metrics.sparseVSkippedValueTokens, 96)
        XCTAssertEqual(decoded.metrics.sparseVConsideredValueTokens, 256)
        XCTAssertEqual(decoded.metrics.sparseVSkipRatio, 0.375)
        XCTAssertEqual(decoded.metrics.sparseVRetainedAttentionMass, 0.985)
        XCTAssertEqual(decoded.metrics.sparseVMaxOutputErrorVsDenseReference, 0.004)
        XCTAssertEqual(decoded.metrics.sparseVCosineVsDenseReference, 0.999)
        XCTAssertEqual(decoded.metrics.sparseVFallbackReason, "layer 1 head 3 fell back to dense")
        XCTAssertEqual(decoded.metrics.sparseVDiagnostics?.count, 2)
        XCTAssertEqual(decoded.metrics.sparseVDiagnostics?.first?.layer, 0)
        XCTAssertEqual(decoded.metrics.sparseVDiagnostics?.first?.head, 1)
        XCTAssertEqual(decoded.metrics.sparseVDiagnostics?.first?.selectionMode, .threshold)
        XCTAssertEqual(decoded.metrics.sparseVDiagnostics?.first?.skippedValueTokens, 64)
        XCTAssertEqual(decoded.metrics.sparseVDiagnostics?.first?.consideredValueTokens, 128)
        XCTAssertEqual(decoded.metrics.sparseVDiagnostics?.first?.retainedAttentionMass, 0.99)
        XCTAssertEqual(
            decoded.metrics.sparseVDiagnostics?.first?.maxOutputErrorVsDenseReference,
            0.003
        )
        XCTAssertEqual(decoded.metrics.sparseVDiagnostics?.first?.cosineVsDenseReference, 0.9995)
        XCTAssertEqual(
            decoded.metrics.sparseVDiagnostics?.last?.fallbackReason,
            "dense reference required"
        )
        XCTAssertEqual(decoded.metrics.lowerVAndSparseV?.referenceConfig, "dense K8/V4")
        XCTAssertEqual(decoded.metrics.lowerVAndSparseV?.candidateConfig, "K8/V4 Sparse-V hybrid")
        XCTAssertEqual(decoded.metrics.lowerVAndSparseV?.valueBitPolicy, .denseV4)
        XCTAssertEqual(decoded.metrics.lowerVAndSparseV?.sparseVMode, .hybridCumulativeMassTopK)
        XCTAssertEqual(decoded.metrics.lowerVAndSparseV?.selectionLatencyMS, 0.2)
        XCTAssertEqual(decoded.metrics.lowerVAndSparseV?.avLatencyMS, 0.5)
        XCTAssertEqual(decoded.metrics.lowerVAndSparseV?.denseK8V4ReferenceMS, 0.9)
        XCTAssertEqual(decoded.metrics.lowerVAndSparseV?.fallbackCount, 1)
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
        XCTAssertEqual(decoded.metrics.rawSDPAReferenceDType, "float16")
        XCTAssertEqual(decoded.metrics.rawSDPAAttentionLatencyMSP50, 5)
        XCTAssertEqual(decoded.metrics.rawSDPADecodeTokensPerSecondP50, 200)
        XCTAssertEqual(decoded.metrics.speedRatioToRawSDPAP50, 0.5)
        XCTAssertEqual(decoded.metrics.rawSDPAKVBytes, 448)
        XCTAssertEqual(decoded.metrics.memoryBytesSavedVsRawSDPA, 336)
        XCTAssertEqual(decoded.metrics.memoryReductionRatio, 4)
        XCTAssertEqual(decoded.metrics.memoryReductionPercent, 75)
        XCTAssertEqual(decoded.metrics.cooldownMS, 7)
        XCTAssertEqual(decoded.metrics.pathCooldownMS, 25)
        XCTAssertEqual(decoded.pathMeasurements.count, TurboQuantAttentionPath.allCases.count)
        XCTAssertEqual(decoded.pathMeasurements.first?.path, .baseline)
        XCTAssertEqual(decoded.pathMeasurements.first?.status, .reference)
        XCTAssertEqual(decoded.pathMeasurements.first?.referenceDType, "float16")
        XCTAssertEqual(decoded.pathMeasurements.first?.rawSDPAKVBytes, 448)
        XCTAssertEqual(decoded.pathMeasurements.first?.memoryBytesSavedVsRawSDPA, 0)
        XCTAssertTrue(decoded.pathMeasurements.contains { $0.path == .affineK8VxResidual })
        XCTAssertTrue(
            decoded.pathMeasurements.contains { $0.path == .sparseValueTwoStageCompressed })
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

    func testNativeAffinePathIsEncodedForBenchmarkEvidence() throws {
        var report = Self.sampleReport()
        report.pathDecision = TurboQuantAttentionDecision(
            selectedPath: .affineK8V4Native,
            outputDType: .float32,
            estimatedScratchBytes: 0,
            rejectedPaths: []
        )

        let data = try JSONEncoder().encode(report)
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        let decision = try XCTUnwrap(object["pathDecision"] as? [String: Any])
        XCTAssertEqual(decision["selectedPath"] as? String, "affineK8V4Native")

        let decoded = try JSONDecoder().decode(TurboQuantCoreBenchmarkReport.self, from: data)
        XCTAssertEqual(decoded.pathDecision?.selectedPath, .affineK8V4Native)
    }

    func testCoreBenchmarkReportEncodesSparseVSelectionModes() throws {
        let data = try JSONEncoder().encode(Self.sampleReport())
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        let metrics = try XCTUnwrap(object["metrics"] as? [String: Any])
        let diagnostics = try XCTUnwrap(metrics["sparseVDiagnostics"] as? [[String: Any]])

        XCTAssertEqual(metrics["sparseVSelectionMode"] as? String, "hybridCumulativeMassTopK")
        XCTAssertEqual(diagnostics.first?["selectionMode"] as? String, "threshold")
        XCTAssertEqual(diagnostics.last?["selectionMode"] as? String, "cumulativeMass")
        let lower = try XCTUnwrap(metrics["lowerVAndSparseV"] as? [String: Any])
        XCTAssertEqual(lower["sparseVMode"] as? String, "hybridCumulativeMassTopK")
    }

    func testLegacyCoreBenchmarkMetricsDecodeWithDefaultedSparseVFields() throws {
        let json = """
            {
              "contextTokens": 256,
              "headDimension": 128,
              "queryLength": 1,
              "preset": "turbo4v2",
              "valueBits": 4,
              "groupSize": 64,
              "totalBytes": 112,
              "actualBitsPerValue": 3.5
            }
            """

        let decoded = try JSONDecoder().decode(
            TurboQuantCoreBenchmarkMetrics.self,
            from: Data(json.utf8)
        )

        XCTAssertEqual(decoded.contextTokens, 256)
        XCTAssertEqual(decoded.sparseVEnabled, false)
        XCTAssertNil(decoded.sparseVSelectionMode)
        XCTAssertEqual(decoded.sparseVSkipRatio, 0)
        XCTAssertEqual(decoded.boundaryProtectedLayerCount, 0)
        XCTAssertEqual(decoded.compressedKVBytes, decoded.totalBytes)
        XCTAssertNil(decoded.rawSDPAReferenceDType)
        XCTAssertNil(decoded.memoryBytesSavedVsRawSDPA)
        XCTAssertNil(decoded.cooldownMS)
        XCTAssertEqual(decoded.fallbackUsed, false)
        XCTAssertEqual(decoded.memoryWarningsSeen, 0)
        XCTAssertEqual(decoded.jetsamObserved, false)
    }

    func testLegacyCoreBenchmarkReportDecodesWithEmptyPathMeasurements() throws {
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
              "storageEstimate": {
                "role": "key",
                "logicalValues": 128,
                "packedBytes": 40,
                "bitsetBytes": 48,
                "scaleBytes": 24,
                "totalBytes": 112,
                "actualBitsPerValue": 7
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

        let decoded = try JSONDecoder().decode(
            TurboQuantCoreBenchmarkReport.self,
            from: Data(json.utf8)
        )

        XCTAssertTrue(decoded.pathMeasurements.isEmpty)
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
        XCTAssertTrue(audit.notes.contains { $0.contains("Layout V6") })
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
            pathMeasurements: Self.samplePathMeasurements(),
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
                sparseVEnabled: true,
                sparseVSelectionMode: .hybridCumulativeMassTopK,
                sparseVThreshold: 0.001,
                sparseVSkippedValueTokens: 96,
                sparseVConsideredValueTokens: 256,
                sparseVSkipRatio: 0.375,
                sparseVRetainedAttentionMass: 0.985,
                sparseVMaxOutputErrorVsDenseReference: 0.004,
                sparseVCosineVsDenseReference: 0.999,
                sparseVFallbackReason: "layer 1 head 3 fell back to dense",
                sparseVDiagnostics: [
                    TurboQuantSparseVDiagnostic(
                        layer: 0,
                        head: 1,
                        selectionMode: .threshold,
                        skippedValueTokens: 64,
                        consideredValueTokens: 128,
                        retainedAttentionMass: 0.99,
                        maxOutputErrorVsDenseReference: 0.003,
                        cosineVsDenseReference: 0.9995
                    ),
                    TurboQuantSparseVDiagnostic(
                        layer: 1,
                        head: 3,
                        selectionMode: .cumulativeMass,
                        skippedValueTokens: 32,
                        consideredValueTokens: 128,
                        retainedAttentionMass: 0.98,
                        maxOutputErrorVsDenseReference: 0.004,
                        cosineVsDenseReference: 0.998,
                        fallbackReason: "dense reference required"
                    ),
                ],
                lowerVAndSparseV: TurboQuantLowerVAndSparseVReport(
                    referenceConfig: "dense K8/V4",
                    candidateConfig: "K8/V4 Sparse-V hybrid",
                    valueBits: 4,
                    valueBitPolicy: .denseV4,
                    sparseVMode: .hybridCumulativeMassTopK,
                    sparseVTopK: 256,
                    sparseVCumulativeMass: 0.995,
                    sparseVMaxTopK: 256,
                    selectionLatencyMS: 0.2,
                    qkMS: 0.4,
                    softmaxMS: 0.1,
                    maskOrCompactionMS: 0.05,
                    avLatencyMS: 0.5,
                    totalMS: 1.25,
                    denseK8V4ReferenceMS: 0.9,
                    skippedValueTokens: 96,
                    consideredValueTokens: 256,
                    retainedMass: 0.985,
                    skipRatio: 0.375,
                    fallbackCount: 1,
                    fallbackReason: "layer 1 head 3 fell back to dense",
                    actualMixedBitsPerValue: 4,
                    layerIndex: 1,
                    headIndex: 3
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
                rawSDPAReferenceDType: "float16",
                rawSDPAAttentionLatencyMSP50: 5,
                rawSDPAAttentionLatencyMSP95: 5.5,
                rawSDPADecodeTokensPerSecondP50: 200,
                rawSDPADecodeTokensPerSecondP95: 180,
                speedRatioToRawSDPAP50: 0.5,
                speedRatioToRawSDPAP95: 0.5,
                totalBytes: 112,
                plainKVBytes: 448,
                rawSDPAKVBytes: 448,
                memoryBytesSavedVsRawSDPA: 336,
                memoryReductionRatio: 4,
                memoryReductionPercent: 75,
                actualBitsPerValue: 3.5,
                cooldownMS: 7,
                pathCooldownMS: 25
            ),
            hiddenCopyAudit: TurboQuantHiddenCopyAudit.currentW5
        )
    }

    private static func samplePathMeasurements() -> [TurboQuantCoreBenchmarkPathMeasurement] {
        ([TurboQuantAttentionPath.baseline]
            + TurboQuantAttentionPath.allCases.filter { $0 != .baseline }
        ).map { path in
            if path == .baseline {
                return TurboQuantCoreBenchmarkPathMeasurement(
                    path: path,
                    route: TurboQuantBenchmarkRoute.rawSDPA.rawValue,
                    backend: TurboQuantBenchmarkBackend.rawSDPA.rawValue,
                    status: .reference,
                    selected: false,
                    validForRequest: true,
                    referenceDType: "float16",
                    reason: "FP16 raw SDPA reference",
                    latencyMSAverage: 5.2,
                    latencyMSP50: 5,
                    latencyMSP95: 5.5,
                    decodeTokensPerSecondP50: 200,
                    decodeTokensPerSecondP95: 180,
                    rawSDPALatencyMSP50: 5,
                    rawSDPALatencyMSP95: 5.5,
                    rawSDPADecodeTokensPerSecondP50: 200,
                    rawSDPADecodeTokensPerSecondP95: 180,
                    speedRatioToRawSDPAP50: 1,
                    speedRatioToRawSDPAP95: 1,
                    compressedKVBytes: 448,
                    rawSDPAKVBytes: 448,
                    memoryBytesSavedVsRawSDPA: 0,
                    memoryReductionRatio: 1,
                    memoryReductionPercent: 0,
                    actualBitsPerValue: 16,
                    maxAbsoluteErrorVsRawSDPA: 0,
                    cosineSimilarityVsRawSDPA: 1
                )
            }

            return TurboQuantCoreBenchmarkPathMeasurement(
                path: path,
                route: path == .twoStageCompressed
                    ? TurboQuantBenchmarkRoute.compressedFused.rawValue
                    : TurboQuantBenchmarkRoute.unavailable.rawValue,
                backend: path == .twoStageCompressed
                    ? TurboQuantBenchmarkBackend.swiftMetalKernel.rawValue
                    : TurboQuantBenchmarkBackend.unavailable.rawValue,
                status: path == .twoStageCompressed ? .measured : .skipped,
                selected: path == .twoStageCompressed,
                validForRequest: path == .twoStageCompressed,
                referenceDType: "float16",
                reason: path == .twoStageCompressed ? nil : "not selected in sample",
                latencyMSAverage: path == .twoStageCompressed ? 10 : nil,
                latencyMSP50: path == .twoStageCompressed ? 10 : nil,
                latencyMSP95: path == .twoStageCompressed ? 11 : nil,
                decodeTokensPerSecondP50: path == .twoStageCompressed ? 100 : nil,
                decodeTokensPerSecondP95: path == .twoStageCompressed ? 90 : nil,
                rawSDPALatencyMSP50: 5,
                rawSDPALatencyMSP95: 5.5,
                rawSDPADecodeTokensPerSecondP50: 200,
                rawSDPADecodeTokensPerSecondP95: 180,
                speedRatioToRawSDPAP50: path == .twoStageCompressed ? 0.5 : nil,
                speedRatioToRawSDPAP95: path == .twoStageCompressed ? 0.5 : nil,
                compressedKVBytes: 112,
                rawSDPAKVBytes: 448,
                memoryBytesSavedVsRawSDPA: 336,
                memoryReductionRatio: 4,
                memoryReductionPercent: 75,
                actualBitsPerValue: 3.5,
                maxAbsoluteErrorVsRawSDPA: path == .twoStageCompressed ? 0.004 : nil,
                cosineSimilarityVsRawSDPA: path == .twoStageCompressed ? 0.999 : nil
            )
        }
    }
}
