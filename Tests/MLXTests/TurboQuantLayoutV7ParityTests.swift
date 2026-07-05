// Copyright © 2026 RNT56.
//
// SPEC 2 section 7 parity gate for Layout v7 (tile-transposed K planes).
//
// v7 permutes bytes WITHIN each (batch, kv_head) plane of the K packed/signs/scales
// tensors relative to v6; the MLXArray shapes are unchanged. This suite proves the
// swizzle is a pure bijection (T1), that attention output matches between v6 and v7
// at ring_offset 0 (T2) and at a nonzero wrapping ring_offset with a pinned prefix
// (T3), and that the admission plumbing fails closed exactly where the spec requires
// (T4).
//
// T2/T3 numerical tolerance (and the RESOLVED 2026-07-03 quad non-determinism):
//
// T1 (element-for-element UInt32/Float equality on the packed/signs/scales planes,
// the spec's "strongest encoder check") passes with ZERO tolerance on every observed
// run, and an independent CPU reimplementation of the full v6 AND v7 decode path
// (written directly from the section-0 algebra, not from the Metal source text)
// reproduces both the GPU v6 AND the GPU v7 attention output to float16-rounding
// precision. The tolerance below covers pure float16/FMA rounding between the two
// GPU kernel variants only.
//
// HISTORY: this file previously quarantined (via XCTExpectFailure(strict: false))
// an intermittent v7 quad-path (GQA_REPEATS==4) non-determinism: at a 5-25%
// per-dispatch rate the ENTIRE output row of the last GQA repeat (q_head =
// kv_head*4+3) was corrupted (the earlier "one (head, dim) element" description was
// a max-diff reporting artifact). Root cause (2026-07-03): a threadgroup-memory
// read-after-write race in the GQA block-partials kernel's weight-conversion loop --
// each lane read the reduced per-repeat max from partial[repeat*TPB] while lane 0
// concurrently overwrote that same slot with its exp-weight, with no barrier between
// the read and the overwrite; inter-simdgroup drift accumulates across the
// barrier-free repeat loop, so the last repeat had the widest race window (hence
// "always q_head = kv_head*4+3"). Fixed by hoisting the tile_maxes reads ahead of a
// dedicated threadgroup_barrier in the v7 kernel (both source copies). Gate: 50
// consecutive strict runs of this suite, 0 failures, plus ~800 sentinel-hardened
// probe dispatches with 0 divergence (pre-fix: 25% under the same probe). Full
// incident record: artifacts/turboquant-v7-20260703/quad-nondeterminism.md. NOTE:
// the SAME latent race pattern exists in the v6-family block-partials kernels
// (empirically never observed firing there); see the incident record before
// touching their back-halves.
private let turboQuantV7NumericalTolerance: Float = 2e-3

import Foundation
import MLX
import XCTest

final class TurboQuantLayoutV7ParityTests: XCTestCase {

    // MARK: - Shared geometry (SPEC 2 section 7)

    private let batchSize = 1
    private let kvHeads = 4
    private let queryHeads = 16  // repeats 4 -- required for useGroupedQueryKernel
    private let headDim = 256
    private let groupSize = 64

    private func makeQueries(length: Int, seed: UInt64) -> MLXArray {
        MLXRandom.seed(seed)
        return (MLXRandom.normal([batchSize, queryHeads, length, headDim]) * 1.0)
            .asType(.float16)
    }

    private func makeKV(length: Int, seed: UInt64) -> MLXArray {
        MLXRandom.seed(seed)
        return (MLXRandom.normal([batchSize, kvHeads, length, headDim]) * 1.0)
            .asType(.float16)
    }

    private func assertClose(
        _ a: [Float16], _ b: [Float16], tolerance: Float = turboQuantV7NumericalTolerance,
        _ message: String, file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(a.count, b.count, message, file: file, line: line)
        var maxDiff: Float = 0
        var maxDiffIndex = -1
        for i in 0 ..< min(a.count, b.count) {
            let diff = abs(Float(a[i]) - Float(b[i]))
            if diff > maxDiff {
                maxDiff = diff
                maxDiffIndex = i
            }
        }
        XCTAssertLessThanOrEqual(
            maxDiff, tolerance,
            "\(message): max abs diff \(maxDiff) at index \(maxDiffIndex) "
                + "(a=\(a[maxDiffIndex]) b=\(b[maxDiffIndex])) exceeds tolerance \(tolerance)",
            file: file, line: line
        )
    }

    // MARK: - T0: independent reimplementation of the v6/v7 offset algebra
    //
    // Written directly from SPEC 2 section 0's algebra, NOT copied from the Metal
    // source text, so a swizzle bug in the Metal offset helpers cannot also be
    // present here.

    private func off6(
        batch: Int, head: Int, token: Int, group: Int, word: Int,
        kvHeads: Int, capacity: Int, groupsPerVector: Int, wordsPerGroup: Int
    ) -> Int {
        let wordsPerToken = groupsPerVector * wordsPerGroup
        let plane = (batch * kvHeads + head) * capacity * wordsPerToken
        let j = group * wordsPerGroup + word
        return plane + token * wordsPerToken + j
    }

    private func off7(
        batch: Int, head: Int, token: Int, group: Int, word: Int,
        kvHeads: Int, capacity: Int, groupsPerVector: Int, wordsPerGroup: Int
    ) -> Int {
        let wordsPerToken = groupsPerVector * wordsPerGroup
        let plane = (batch * kvHeads + head) * capacity * wordsPerToken
        let tileBase = plane + (token & ~31) * wordsPerToken
        let j = group * wordsPerGroup + word
        return tileBase + j * 32 + (token & 31)
    }

    // MARK: - T1: pure permutation (encoder only, both presets)

    private func runT1(preset: TurboQuantPreset) throws {
        try requireTurboQuantMetalAttention()

        let length = 1056  // 33 tiles of 32
        let keys = makeKV(length: length, seed: 0x7000_0000_0000_0001)

        let v6Code = try turboQuantMetalEncodeAttention(
            keys,
            configuration: TurboQuantConfiguration(
                preset: preset,
                role: .key,
                groupSize: groupSize,
                backend: .metalPolarQJL,
                seed: 0x7000_0000_0000_0002,
                attentionLayoutVersion: TurboQuantAttentionLayout.splitMagnitudeVersion
            )
        )
        let v7Code = try turboQuantMetalEncodeAttention(
            keys,
            configuration: TurboQuantConfiguration(
                preset: preset,
                role: .key,
                groupSize: groupSize,
                backend: .metalPolarQJL,
                seed: 0x7000_0000_0000_0002,
                attentionLayoutVersion: TurboQuantAttentionLayout.tileTransposedVersion,
                allowExperimentalLayoutV7: true
            )
        )

        XCTAssertEqual(v6Code.layout.capacity, v7Code.layout.capacity)
        XCTAssertEqual(v6Code.layout.groupsPerVector, v7Code.layout.groupsPerVector)
        XCTAssertEqual(v6Code.layout.magnitudeWordsPerGroup, v7Code.layout.magnitudeWordsPerGroup)
        XCTAssertEqual(v6Code.layout.bitsetWordsPerGroup, v7Code.layout.bitsetWordsPerGroup)
        // Capacity was defaulted, so v7 must have padded to a 32-multiple; the source
        // length (1056) is already 32-aligned, so both should match it exactly here.
        XCTAssertEqual(v6Code.layout.capacity, length)
        XCTAssertEqual(v7Code.layout.capacity, length)

        let capacity = v6Code.layout.capacity
        let groupsPerVector = v6Code.layout.groupsPerVector
        let magWordsPerGroup = v6Code.layout.magnitudeWordsPerGroup
        let bitsetWordsPerGroup = v6Code.layout.bitsetWordsPerGroup

        let v6Packed = v6Code.packedMagnitudes.asArray(UInt32.self)
        let v7Packed = v7Code.packedMagnitudes.asArray(UInt32.self)
        let v6Signs = v6Code.signs.asArray(UInt32.self)
        let v7Signs = v7Code.signs.asArray(UInt32.self)
        let v6Scales = v6Code.scales.asArray(Float.self)
        let v7Scales = v7Code.scales.asArray(Float.self)

        for b in 0 ..< batchSize {
            for h in 0 ..< kvHeads {
                for t in 0 ..< capacity {
                    for g in 0 ..< groupsPerVector {
                        for w in 0 ..< magWordsPerGroup {
                            let v6i = off6(
                                batch: b, head: h, token: t, group: g, word: w,
                                kvHeads: kvHeads, capacity: capacity,
                                groupsPerVector: groupsPerVector, wordsPerGroup: magWordsPerGroup)
                            let v7i = off7(
                                batch: b, head: h, token: t, group: g, word: w,
                                kvHeads: kvHeads, capacity: capacity,
                                groupsPerVector: groupsPerVector, wordsPerGroup: magWordsPerGroup)
                            XCTAssertEqual(
                                v7Packed[v7i], v6Packed[v6i],
                                "packed mismatch at b=\(b) h=\(h) t=\(t) g=\(g) w=\(w)")
                        }
                        for w in 0 ..< bitsetWordsPerGroup {
                            let v6i = off6(
                                batch: b, head: h, token: t, group: g, word: w,
                                kvHeads: kvHeads, capacity: capacity,
                                groupsPerVector: groupsPerVector, wordsPerGroup: bitsetWordsPerGroup)
                            let v7i = off7(
                                batch: b, head: h, token: t, group: g, word: w,
                                kvHeads: kvHeads, capacity: capacity,
                                groupsPerVector: groupsPerVector, wordsPerGroup: bitsetWordsPerGroup)
                            XCTAssertEqual(
                                v7Signs[v7i], v6Signs[v6i],
                                "signs mismatch at b=\(b) h=\(h) t=\(t) g=\(g) w=\(w)")
                        }
                        for s in 0 ..< 2 {
                            let v6i = off6(
                                batch: b, head: h, token: t, group: g, word: s,
                                kvHeads: kvHeads, capacity: capacity,
                                groupsPerVector: groupsPerVector, wordsPerGroup: 2)
                            let v7i = off7(
                                batch: b, head: h, token: t, group: g, word: s,
                                kvHeads: kvHeads, capacity: capacity,
                                groupsPerVector: groupsPerVector, wordsPerGroup: 2)
                            XCTAssertEqual(
                                v7Scales[v7i], v6Scales[v6i],
                                "scales mismatch at b=\(b) h=\(h) t=\(t) g=\(g) s=\(s)")
                        }
                    }
                }
            }
        }
    }

    func testT1PurePermutationUniformPreset() throws {
        try runT1(preset: .turbo4v2)
    }

    func testT1PurePermutationSplitPreset() throws {
        try runT1(preset: .turbo3_5)
    }

    // MARK: - T2: attention parity, ring_offset 0

    private func runT2(preset: TurboQuantPreset, mask: MLXFast.ScaledDotProductAttentionMaskMode)
        throws
    {
        try requireTurboQuantMetalAttention()

        let length = 1056
        let keys = makeKV(length: length, seed: 0x7000_0000_0000_0011)
        let values = makeKV(length: length, seed: 0x7000_0000_0000_0012)
        let queries = makeQueries(length: 1, seed: 0x7000_0000_0000_0013)
        let scale = Float(1.0 / Double(headDim).squareRoot())

        func encode(layoutVersion: Int, allowV7: Bool) throws -> (
            key: TurboQuantAttentionCode, value: TurboQuantAttentionCode
        ) {
            let key = try turboQuantMetalEncodeAttention(
                keys,
                configuration: TurboQuantConfiguration(
                    preset: preset, role: .key, groupSize: groupSize, backend: .metalPolarQJL,
                    seed: 0x7000_0000_0000_0014,
                    attentionLayoutVersion: layoutVersion,
                    allowExperimentalLayoutV7: allowV7
                )
            )
            let value = try turboQuantMetalEncodeAttention(
                values,
                configuration: TurboQuantConfiguration(
                    preset: preset, role: .value, groupSize: groupSize, backend: .metalPolarQJL,
                    seed: 0x7000_0000_0000_0015,
                    attentionLayoutVersion: layoutVersion,
                    allowExperimentalLayoutV7: allowV7
                )
            )
            return (key, value)
        }

        let v6 = try encode(
            layoutVersion: TurboQuantAttentionLayout.splitMagnitudeVersion, allowV7: false)
        let v7 = try encode(
            layoutVersion: TurboQuantAttentionLayout.tileTransposedVersion, allowV7: true)
        // Force full materialization of the encoded planes before dispatching attention,
        // so a lazy-graph fusion/ordering race under concurrent GPU load cannot read a
        // partially-written K/V plane.
        eval(
            v6.key.packedMagnitudes, v6.key.signs, v6.key.scales,
            v6.value.packedMagnitudes, v6.value.scales,
            v7.key.packedMagnitudes, v7.key.signs, v7.key.scales,
            v7.value.packedMagnitudes, v7.value.scales
        )

        let v6Out = try turboQuantMetalScaledDotProductAttention(
            queries: queries,
            keyCode: v6.key,
            valueCode: v6.value,
            scale: scale,
            mask: mask,
            preferOnlineFused: true,
            blockParallelTokenBlockSize: 256
        )
        eval(v6Out)
        let v7Out = try turboQuantMetalScaledDotProductAttention(
            queries: queries,
            keyCode: v7.key,
            valueCode: v7.value,
            scale: scale,
            mask: mask,
            preferOnlineFused: true,
            blockParallelTokenBlockSize: 256
        )
        eval(v7Out)

        let v6Values = v6Out.asArray(Float16.self)
        let v7Values = v7Out.asArray(Float16.self)
        // GQA repeats == 4 at this geometry (queryHeads 16 / kvHeads 4) always engages
        // the quad kernel. Strict since the 2026-07-03 barrier fix (see file header).
        assertClose(v6Values, v7Values, "v7 attention output diverged from v6 (mask=\(mask))")
    }

    func testT2AttentionParityUniformNoneMask() throws {
        try runT2(preset: .turbo4v2, mask: .none)
    }

    func testT2AttentionParityUniformCausalMask() throws {
        try runT2(preset: .turbo4v2, mask: .causal)
    }

    func testT2AttentionParitySplitNoneMask() throws {
        try runT2(preset: .turbo3_5, mask: .none)
    }

    func testT2AttentionParitySplitCausalMask() throws {
        try runT2(preset: .turbo3_5, mask: .causal)
    }

    // MARK: - T3: nonzero ring offset + wrap + pinned prefix

    private func runT3(preset: TurboQuantPreset, mask: MLXFast.ScaledDotProductAttentionMaskMode)
        throws
    {
        try requireTurboQuantMetalAttention()

        let capacity = 1056
        let ringOffset = 37
        let pinnedPrefixLength = 5
        let logicalLength = capacity
        let keys = makeKV(length: capacity, seed: 0x7000_0000_0000_0021)
        let values = makeKV(length: capacity, seed: 0x7000_0000_0000_0022)
        let queries = makeQueries(length: 1, seed: 0x7000_0000_0000_0023)
        let scale = Float(1.0 / Double(headDim).squareRoot())

        func encode(layoutVersion: Int, allowV7: Bool) throws -> (
            key: TurboQuantAttentionCode, value: TurboQuantAttentionCode
        ) {
            let key = try turboQuantMetalEncodeAttention(
                keys,
                configuration: TurboQuantConfiguration(
                    preset: preset, role: .key, groupSize: groupSize, backend: .metalPolarQJL,
                    seed: 0x7000_0000_0000_0024,
                    attentionLayoutVersion: layoutVersion,
                    allowExperimentalLayoutV7: allowV7
                ),
                capacity: capacity,
                logicalLength: logicalLength,
                ringOffset: ringOffset,
                pinnedPrefixLength: pinnedPrefixLength
            )
            let value = try turboQuantMetalEncodeAttention(
                values,
                configuration: TurboQuantConfiguration(
                    preset: preset, role: .value, groupSize: groupSize, backend: .metalPolarQJL,
                    seed: 0x7000_0000_0000_0025,
                    attentionLayoutVersion: layoutVersion,
                    allowExperimentalLayoutV7: allowV7
                ),
                capacity: capacity,
                logicalLength: logicalLength,
                ringOffset: ringOffset,
                pinnedPrefixLength: pinnedPrefixLength
            )
            return (key, value)
        }

        let v6 = try encode(
            layoutVersion: TurboQuantAttentionLayout.splitMagnitudeVersion, allowV7: false)
        let v7 = try encode(
            layoutVersion: TurboQuantAttentionLayout.tileTransposedVersion, allowV7: true)
        // Force full materialization of the encoded planes before dispatching attention,
        // so a lazy-graph fusion/ordering race under concurrent GPU load cannot read a
        // partially-written K/V plane.
        eval(
            v6.key.packedMagnitudes, v6.key.signs, v6.key.scales,
            v6.value.packedMagnitudes, v6.value.scales,
            v7.key.packedMagnitudes, v7.key.signs, v7.key.scales,
            v7.value.packedMagnitudes, v7.value.scales
        )

        let v6Out = try turboQuantMetalScaledDotProductAttention(
            queries: queries,
            keyCode: v6.key,
            valueCode: v6.value,
            scale: scale,
            mask: mask,
            preferOnlineFused: true,
            blockParallelTokenBlockSize: 256
        )
        eval(v6Out)
        let v7Out = try turboQuantMetalScaledDotProductAttention(
            queries: queries,
            keyCode: v7.key,
            valueCode: v7.value,
            scale: scale,
            mask: mask,
            preferOnlineFused: true,
            blockParallelTokenBlockSize: 256
        )
        eval(v7Out)

        let v6Values = v6Out.asArray(Float16.self)
        let v7Values = v7Out.asArray(Float16.self)
        // GQA repeats == 4 at this geometry always engages the quad kernel. Strict
        // since the 2026-07-03 barrier fix (see file header).
        assertClose(
            v6Values, v7Values,
            "v7 attention output diverged from v6 with wrapping ring offset (mask=\(mask))")
    }

    func testT3RingOffsetWrapAndPinnedPrefixNoneMask() throws {
        try runT3(preset: .turbo3_5, mask: .none)
    }

    func testT3RingOffsetWrapAndPinnedPrefixCausalMask() throws {
        try runT3(preset: .turbo3_5, mask: .causal)
    }

    // MARK: - T4: fail-closed

    func testT4ExplicitUnalignedCapacityThrows() throws {
        try requireTurboQuantMetalAttention()
        let keys = makeKV(length: 1000, seed: 0x7000_0000_0000_0031)
        XCTAssertThrowsError(
            try turboQuantMetalEncodeAttention(
                keys,
                configuration: TurboQuantConfiguration(
                    preset: .turbo4v2, role: .key, groupSize: groupSize, backend: .metalPolarQJL,
                    seed: 0x7000_0000_0000_0032,
                    attentionLayoutVersion: TurboQuantAttentionLayout.tileTransposedVersion,
                    allowExperimentalLayoutV7: true
                ),
                capacity: 1000
            )
        )
    }

    func testT4DecodeAttentionRejectsV7() throws {
        try requireTurboQuantMetalAttention()
        let length = 1056
        let keys = makeKV(length: length, seed: 0x7000_0000_0000_0033)
        let v7Code = try turboQuantMetalEncodeAttention(
            keys,
            configuration: TurboQuantConfiguration(
                preset: .turbo4v2, role: .key, groupSize: groupSize, backend: .metalPolarQJL,
                seed: 0x7000_0000_0000_0034,
                attentionLayoutVersion: TurboQuantAttentionLayout.tileTransposedVersion,
                allowExperimentalLayoutV7: true
            )
        )
        XCTAssertThrowsError(try turboQuantMetalDecodeAttention(v7Code))
    }

    // `turboQuantCooperativeQuadDecodeActive` is internal to the MLX module and this test
    // target imports MLX non-@testable, so the gate is exercised indirectly here: run the
    // block-parallel v7 path at a shape that WOULD select the coop kernel at v6 (GQA
    // repeats 4, uniform preset, headDim % 4 == 0). If the v7 Swift-side gate at
    // `turboQuantCooperativeQuadDecodeActive` (or the C++ `tq_cooperative_gqa_path_allowed`
    // gate) ever regressed to allow coop for v7, the coop kernel's v6-hardcoded offsets
    // would misread the v7-swizzled K planes and this would diverge sharply from the v6
    // reference (or throw the "cooperative decode is not ported to layout v7" guard).
    // Passing here proves neither happened.
    func testT4CoopGateNeverSelectsV7() throws {
        try requireTurboQuantMetalAttention()

        let length = 1056
        let keys = makeKV(length: length, seed: 0x7000_0000_0000_0041)
        let values = makeKV(length: length, seed: 0x7000_0000_0000_0042)
        let queries = makeQueries(length: 1, seed: 0x7000_0000_0000_0043)
        let scale = Float(1.0 / Double(headDim).squareRoot())

        func encode(layoutVersion: Int, allowV7: Bool) throws -> (
            key: TurboQuantAttentionCode, value: TurboQuantAttentionCode
        ) {
            let key = try turboQuantMetalEncodeAttention(
                keys,
                configuration: TurboQuantConfiguration(
                    preset: .turbo4v2, role: .key, groupSize: groupSize, backend: .metalPolarQJL,
                    seed: 0x7000_0000_0000_0044,
                    attentionLayoutVersion: layoutVersion,
                    allowExperimentalLayoutV7: allowV7
                )
            )
            let value = try turboQuantMetalEncodeAttention(
                values,
                configuration: TurboQuantConfiguration(
                    preset: .turbo4v2, role: .value, groupSize: groupSize,
                    backend: .metalPolarQJL,
                    seed: 0x7000_0000_0000_0045,
                    attentionLayoutVersion: layoutVersion,
                    allowExperimentalLayoutV7: allowV7
                )
            )
            return (key, value)
        }

        let v6 = try encode(
            layoutVersion: TurboQuantAttentionLayout.splitMagnitudeVersion, allowV7: false)
        let v7 = try encode(
            layoutVersion: TurboQuantAttentionLayout.tileTransposedVersion, allowV7: true)
        eval(
            v6.key.packedMagnitudes, v6.key.signs, v6.key.scales,
            v6.value.packedMagnitudes, v6.value.scales,
            v7.key.packedMagnitudes, v7.key.signs, v7.key.scales,
            v7.value.packedMagnitudes, v7.value.scales
        )

        let v6Out = try turboQuantMetalScaledDotProductAttention(
            queries: queries, keyCode: v6.key, valueCode: v6.value, scale: scale, mask: .none,
            preferOnlineFused: true, blockParallelTokenBlockSize: 256
        )
        eval(v6Out)
        let v7Out = try turboQuantMetalScaledDotProductAttention(
            queries: queries, keyCode: v7.key, valueCode: v7.value, scale: scale, mask: .none,
            preferOnlineFused: true, blockParallelTokenBlockSize: 256
        )
        eval(v7Out)
        // This test's job is catching a wrongly-selected coop kernel (v6-hardcoded
        // offsets misreading v7-swizzled planes corrupts the output far beyond
        // rounding). Since the 2026-07-03 quad barrier fix (see file header) the quad
        // path is deterministic, so this uses the standard tight tolerance -- it now
        // catches BOTH a coop-selection regression and any quad-path regression.
        assertClose(
            v6Out.asArray(Float16.self), v7Out.asArray(Float16.self),
            "v7 output diverged -- either the coop kernel (v6-only offsets) was "
                + "incorrectly selected for a v7 code, or the quad path regressed")
    }

    func testT4MissingAllowExperimentalFlagThrows() throws {
        try requireTurboQuantMetalAttention()
        let keys = makeKV(length: 1056, seed: 0x7000_0000_0000_0035)
        XCTAssertThrowsError(
            try turboQuantMetalEncodeAttention(
                keys,
                configuration: TurboQuantConfiguration(
                    preset: .turbo4v2, role: .key, groupSize: groupSize, backend: .metalPolarQJL,
                    seed: 0x7000_0000_0000_0036,
                    attentionLayoutVersion: TurboQuantAttentionLayout.tileTransposedVersion
                    // allowExperimentalLayoutV7 defaults to false.
                )
            )
        )
    }
}
