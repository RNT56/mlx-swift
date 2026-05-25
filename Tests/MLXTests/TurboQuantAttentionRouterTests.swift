// Copyright © 2026 RNT56.

import MLX
import XCTest

final class TurboQuantAttentionRouterTests: XCTestCase {
    func testFusedUnavailableSelectsTwoStageWhenQKAndAVAreAvailable() {
        let decision = selectTurboQuantAttentionPath(
            request: Self.request(),
            capabilities: TurboQuantKernelCapabilities(
                attentionEncode: true,
                attentionDecode: true,
                attentionQK: true,
                attentionAV: true,
                attentionFusedDecode: false,
                bfloatOutput: false
            )
        )

        XCTAssertEqual(decision.selectedPath, .twoStageCompressed)
        XCTAssertTrue(decision.rejectedPaths.contains { $0.path == .onlineFused })
        XCTAssertTrue(decision.rejectedPaths.allSatisfy { !$0.reason.isEmpty })
    }

    func testQKUnavailableSelectsPackedFallbackWhenAllowed() {
        let decision = selectTurboQuantAttentionPath(
            request: Self.request(
                fallbackState: TurboQuantAttentionFallbackState(packedFallbackAvailable: true)
            ),
            capabilities: TurboQuantKernelCapabilities()
        )

        XCTAssertEqual(decision.selectedPath, .mlxPackedFallback)
        XCTAssertTrue(decision.rejectedPaths.contains { $0.path == .twoStageCompressed })
        XCTAssertTrue(decision.rejectedPaths.allSatisfy { !$0.reason.isEmpty })
    }

    func testUnsupportedMaskRejectsCompressedPathsBeforeKernelDispatch() {
        let decision = selectTurboQuantAttentionPath(
            request: Self.request(maskKind: .unsupportedMaterializedArrays),
            capabilities: TurboQuantKernelCapabilities(
                attentionEncode: true,
                attentionDecode: true,
                attentionQK: true,
                attentionAV: true,
                attentionFusedDecode: true,
                bfloatOutput: true
            )
        )

        XCTAssertEqual(decision.selectedPath, .unavailable)
        XCTAssertTrue(decision.rejectedPaths.contains { $0.path == .twoStageCompressed })
        XCTAssertTrue(decision.rejectedPaths.allSatisfy { !$0.reason.isEmpty })
        XCTAssertTrue(decision.rejectedPaths.contains { $0.path == .baseline })
        XCTAssertFalse(decision.fallbackReason?.isEmpty ?? true)
    }

    private static func request(
        maskKind: TurboQuantAttentionMaskKind = .causal,
        fallbackState: TurboQuantAttentionFallbackState = .none
    ) -> TurboQuantAttentionRequest {
        let layout = TurboQuantAttentionLayout(
            batchSize: 1,
            kvHeadCount: 1,
            capacity: 16,
            logicalLength: 8,
            headDimension: 64,
            groupsPerVector: 1,
            magnitudeWordsPerGroup: 5,
            bitsetWordsPerGroup: 2
        )
        return TurboQuantAttentionRequest(
            queryShape: [1, 1, 1, 64],
            keyLayout: layout,
            valueLayout: layout,
            queryDType: .float16,
            outputDType: .float16,
            maskKind: maskKind,
            fallbackState: fallbackState
        )
    }
}
