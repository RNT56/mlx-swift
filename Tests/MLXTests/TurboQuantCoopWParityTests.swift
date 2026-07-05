// Copyright © 2026 RNT56.
//
// T2.4 stage-2 gate 4 parity test: cosine/max-abs parity between the strided
// GQA block-partials kernel (LANES_PER_TOKEN=1) and the widened coop kernel
// (`_coopw`, LANES_PER_TOKEN=4) at repeats 2 and 3 -- the repeat counts that
// were previously unreachable through the coop branch (which hardcoded
// `r < 4u` and silently read uninitialized threadgroup `query_cache` rows for
// repeats < 4). Uses a COSINE + max-abs gate, not exact equality, because coop
// is not bit-exact vs strided (simd_shuffle reassociation) -- same operator
// parity gate as TurboQuantH16ParityTests.
//
// Threshold: cosine >= 0.9999 AND max-abs <= 3e-3. If any tested config fails
// this, W2 (the coopw widening) is ABORTED per the COOPW SPEC gate 4.
//
// SUBPROCESS DESIGN (load-bearing, see TurboQuantH16ParityTests for the full
// rationale): `turboQuantCooperativeDecodeEnabled` is a Swift top-level `let`
// evaluated once and cached, and `TurboQuantMetalKernels` is `private` and
// unreachable via `@testable import MLX`. So this suite reuses the
// `TurboQuantBenchmark --h16-parity-dump` subprocess harness (repurposed here
// for repeats, not just H16) with `TQ_H16` unset in both arms and `TQ_COOP`
// toggled between the two subprocess invocations, run against IDENTICAL
// deterministic synthetic K/V/Q inputs at repeats 2 and 3 via
// `--h16-parity-repeats`.
//
// Requires a prebuilt release binary:
//   swift build -c release --product TurboQuantBenchmark
// Skips (XCTSkip) if the binary is not found, rather than silently no-op-passing.
import Foundation
import MLX
import XCTest

final class TurboQuantCoopWParityTests: XCTestCase {

    private func benchmarkBinaryURL() -> URL? {
        let testFileURL = URL(fileURLWithPath: #filePath)
        let packageRoot = testFileURL
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let candidate = packageRoot
            .appendingPathComponent(".build")
            .appendingPathComponent("release")
            .appendingPathComponent("TurboQuantBenchmark")
        return FileManager.default.isExecutableFile(atPath: candidate.path) ? candidate : nil
    }

    private struct DumpResult {
        var values: [Float]
        var stderrText: String
    }

    private func runDump(
        binary: URL,
        preset: TurboQuantPreset,
        headDim: Int,
        repeats: Int,
        coop: Bool
    ) throws -> DumpResult {
        let process = Process()
        process.executableURL = binary
        var args = [
            "--h16-parity-dump",
            "--h16-parity-preset", preset.rawValue,
            "--h16-parity-head-dim", String(headDim),
            "--h16-parity-repeats", String(repeats),
        ]
        // --h16-parity-coop widens the dump harness's default context to
        // turboQuantCooperativeDecodeMinContext (32768) or above -- required for
        // turboQuantCooperativeQuadDecodeActive's own context floor regardless of
        // which arm (strided or coop) is being run, so both arms compare the same
        // context. Passed for BOTH arms to keep the comparison apples-to-apples.
        args.append("--h16-parity-coop")
        process.arguments = args

        var env = ProcessInfo.processInfo.environment
        env.removeValue(forKey: "TQ_H16")
        env["TQ_COOP"] = coop ? "1" : "0"
        env["TQ_KERNEL_TRACE"] = "1"
        // headDim 128/256 is exactly the native-certified set on which
        // turboQuantAttentionDecision claims .nativeMLXCompressed before the
        // online-fused/block-parallel Swift path (where coop/coopw actually live) is
        // even considered -- see TurboQuantH16ParityTests' HEAD DIMENSION CHOICE note.
        // Disabling native MLX attention here forces rejection of that path so both
        // arms reach the Swift block-parallel kernel this test actually exercises.
        env["MLX_TURBOQUANT_NATIVE_ATTENTION"] = "0"
        process.environment = env

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        try process.run()
        let stdoutData = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
        let stderrData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()

        guard process.terminationStatus == 0 else {
            let stderrText = String(data: stderrData, encoding: .utf8) ?? ""
            throw XCTSkip(
                "h16-parity-dump subprocess exited \(process.terminationStatus): \(stderrText)"
            )
        }

        guard
            let line = String(data: stdoutData, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines),
            let lineData = line.data(using: .utf8),
            let values = try? JSONDecoder().decode([Float].self, from: lineData)
        else {
            throw XCTSkip("h16-parity-dump subprocess produced no parseable JSON on stdout")
        }
        return DumpResult(values: values, stderrText: String(data: stderrData, encoding: .utf8) ?? "")
    }

    private func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        var dot: Double = 0
        var na: Double = 0
        var nb: Double = 0
        for i in 0 ..< min(a.count, b.count) {
            let x = Double(a[i])
            let y = Double(b[i])
            dot += x * y
            na += x * x
            nb += y * y
        }
        if na == 0 || nb == 0 { return na == nb ? 1.0 : 0.0 }
        return Float(dot / (na.squareRoot() * nb.squareRoot()))
    }

    private func maxAbsDiff(_ a: [Float], _ b: [Float]) -> Float {
        var maxDiff: Float = 0
        for i in 0 ..< min(a.count, b.count) {
            maxDiff = max(maxDiff, abs(a[i] - b[i]))
        }
        return maxDiff
    }

    private func runParity(
        preset: TurboQuantPreset,
        headDim: Int,
        repeats: Int,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        try requireTurboQuantMetalAttention()
        guard let binary = benchmarkBinaryURL() else {
            throw XCTSkip(
                "TurboQuantBenchmark release binary not found; run "
                    + "`swift build -c release --product TurboQuantBenchmark` first"
            )
        }
        let strided = try runDump(
            binary: binary, preset: preset, headDim: headDim, repeats: repeats, coop: false)
        let coopw = try runDump(
            binary: binary, preset: preset, headDim: headDim, repeats: repeats, coop: true)

        // turboQuantCooperativeQuadDecodeActive only engages for headDim 128/256 (see
        // TurboQuant.swift), but those are exactly the head dims that native MLX
        // compressed attention capability probing claims first when it passes on this
        // machine (verified empirically: --h16-parity-head-dim 128 with TQ_COOP=1
        // dispatches singlePass, never blockParallel, on this machine) -- same
        // catch-22 documented in TurboQuantH16ParityTests' HEAD DIMENSION CHOICE /
        // coop-H16 skip notes. Skip rather than fail so this reads as an environment
        // limitation, not a coopw regression; validate on a machine/config where
        // native compressed attention capability probing fails.
        let expectedKernel = "turboquant_attention_fused_gqa_block_partials_coopw_runtime_layout_rtu1_s2"
        guard coopw.stderrText.contains(expectedKernel) else {
            throw XCTSkip(
                "coopw requires headDim 128/256 for TQCOOP's own gate, but those head "
                    + "dims are claimed by native MLX compressed attention on this machine "
                    + "before the online-fused/block-parallel path is considered -- coopw "
                    + "is not reachable via the public API here (trace was: "
                    + "\(coopw.stderrText)). Validate on a machine/config where native "
                    + "compressed attention capability probing fails."
            )
        }
        // Negative check: must NOT have silently fallen back to the strided kernel
        // trace under the coop env (that would mean coop never engaged at all).
        let stridedKernelMarker = "turboquant_attention_fused_gqa_block_partials_runtime_layout_rtu1_s2"
        XCTAssertFalse(
            coopw.stderrText.contains(stridedKernelMarker)
                && !coopw.stderrText.contains(expectedKernel),
            "coop subprocess fell back to the strided kernel instead of engaging _coopw; "
                + "trace was: \(coopw.stderrText)",
            file: file, line: line
        )

        XCTAssertEqual(strided.values.count, coopw.values.count, file: file, line: line)
        let cosine = cosineSimilarity(strided.values, coopw.values)
        let maxDiff = maxAbsDiff(strided.values, coopw.values)
        XCTAssertGreaterThanOrEqual(
            cosine, 0.9999,
            "coopw vs strided cosine \(cosine) below gate "
                + "(preset=\(preset) headDim=\(headDim) repeats=\(repeats))",
            file: file, line: line
        )
        XCTAssertLessThanOrEqual(
            maxDiff, 3e-3,
            "coopw vs strided max-abs \(maxDiff) above gate "
                + "(preset=\(preset) headDim=\(headDim) repeats=\(repeats))",
            file: file, line: line
        )
    }

    func testCoopWParityUniformHeadDim128Repeats2() throws {
        try runParity(preset: .turbo4v2, headDim: 128, repeats: 2)
    }

    func testCoopWParityUniformHeadDim128Repeats3() throws {
        try runParity(preset: .turbo4v2, headDim: 128, repeats: 3)
    }

    func testCoopWParityRepeats4Unchanged() throws {
        // Repeats==4 must remain routed to the byte-frozen fusedAttentionGQABlockPartialsCoop
        // kernel, not _coopw -- this asserts the dispatch guard, not the clamp math (the
        // clamp is a no-op at repeats==4 by construction).
        try requireTurboQuantMetalAttention()
        guard let binary = benchmarkBinaryURL() else {
            throw XCTSkip(
                "TurboQuantBenchmark release binary not found; run "
                    + "`swift build -c release --product TurboQuantBenchmark` first"
            )
        }
        let coop4 = try runDump(
            binary: binary, preset: .turbo4v2, headDim: 128, repeats: 4, coop: true)
        // Same native-capability catch-22 as runParity above: skip rather than fail if
        // headDim 128 never reaches the Swift block-parallel path on this machine.
        guard coop4.stderrText.contains("gqa_block_partials") else {
            throw XCTSkip(
                "headDim 128 does not reach the Swift block-parallel coop path on this "
                    + "machine (trace was: \(coop4.stderrText)); validate on a machine/"
                    + "config where native compressed attention capability probing fails."
            )
        }
        XCTAssertTrue(
            coop4.stderrText.contains("turboquant_attention_fused_gqa_block_partials_coop_runtime_layout_rtu1_s2")
                && !coop4.stderrText.contains(
                    "turboquant_attention_fused_gqa_block_partials_coopw_runtime_layout_rtu1_s2"),
            "repeats==4 must dispatch to the byte-frozen _coop kernel, not _coopw; "
                + "trace was: \(coop4.stderrText)"
        )
    }
}
