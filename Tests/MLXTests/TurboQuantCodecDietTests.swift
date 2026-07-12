// Copyright © 2026 Apple Inc.
//
// N4 PolarQJL metadata-diet measurement + validation (CPU reference codec — fast).
//
// Quantifies where the bits go in the reference PolarQJL code (the diet target the
// roadmap never measured): payload (packed magnitudes + sign/mask bitsets) vs metadata
// (the 3 fp32 scale arrays per group). Then validates diet step 1 — fp32→fp16 scales —
// is quality-safe and measures its byte saving, and computes the one-norm-per-vector
// metadata saving analytically. The substantive 4–7× lever is the payload quantizer
// (data-free Gaussian Lloyd-Max), reported here as the remaining step with a baseline.

import Foundation
import MLX
import XCTest

final class TurboQuantCodecDietTests: XCTestCase {

    private func cosine(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count)
        var dot: Double = 0, na: Double = 0, nb: Double = 0
        for i in 0 ..< a.count {
            dot += Double(a[i]) * Double(b[i])
            na += Double(a[i]) * Double(a[i])
            nb += Double(b[i]) * Double(b[i])
        }
        let denom = (na.squareRoot() * nb.squareRoot())
        return denom > 0 ? Float(dot / denom) : 1
    }

    private func bytes(_ code: TurboQuantReferenceCode) -> (scales: Int, payload: Int, total: Int) {
        let scales = (code.baseScales.count + code.highScales.count + code.residualScales.count) * 4
        let payload =
            code.packedMagnitudes.count + code.signs.count + code.highPrecisionMask.count
            + code.residualSigns.count
        return (scales, payload, scales + payload)
    }

    func testPolarQJLCodecDietBreakdownAndFp16ScaleSafety() throws {
        // Representative Gaussian source (QJL targets Gaussian-like K/V); deterministic seed.
        let n = 4096
        let groupSize = 64
        MLXRandom.seed(0xD1E7)
        let arr = (MLXRandom.normal([n]) * 1.0).asType(.float32)
        let original = arr.asArray(Float.self)

        let config = TurboQuantConfiguration(
            preset: .turbo3_5, role: .vector, groupSize: groupSize, backend: .polarQJLReference)
        let code = try turboQuantReferenceEncode(arr, configuration: config)

        // Baseline quality + byte breakdown.
        let baseDecoded = try turboQuantReferenceDecode(code).asArray(Float.self)
        let baseCos = cosine(original, baseDecoded)
        let b = bytes(code)
        let bitsPerValue = Double(b.total) * 8.0 / Double(code.valueCount)
        let scaleBitsPerValue = Double(b.scales) * 8.0 / Double(code.valueCount)
        let compressionVsFp16 = 16.0 / bitsPerValue

        // Diet step 1: fp32 -> fp16 scales. Round the 3 scale arrays to fp16 precision,
        // re-decode, and check the quality impact + byte saving.
        var fp16 = code
        fp16.baseScales = code.baseScales.map { Float(Float16($0)) }
        fp16.highScales = code.highScales.map { Float(Float16($0)) }
        fp16.residualScales = code.residualScales.map { Float(Float16($0)) }
        let fp16Decoded = try turboQuantReferenceDecode(fp16).asArray(Float.self)
        let fp16Cos = cosine(original, fp16Decoded)
        let fp16ScaleBytes = (code.baseScales.count + code.highScales.count + code.residualScales.count) * 2
        let fp16BitsPerValue = Double(b.payload + fp16ScaleBytes) * 8.0 / Double(code.valueCount)

        // Diet step 2 (analytic): one norm per vector instead of 3 scales per group.
        // Upper-bound saving on the scale metadata (payload unchanged).
        let oneNormScaleBytes = (code.valueCount + groupSize - 1) / groupSize * 2  // 1 fp16 norm/group floor
        let oneNormBitsPerValue = Double(b.payload + oneNormScaleBytes) * 8.0 / Double(code.valueCount)

        print(
            """

            === N4 PolarQJL codec diet (reference, role=.vector, turbo3_5, group=\(groupSize), n=\(n)) ===
            baseline: \(String(format: "%.3f", bitsPerValue)) bits/value  (\(String(format: "%.2f", compressionVsFp16))x vs fp16)  cosine \(String(format: "%.6f", baseCos))
              payload \(b.payload) B (\(String(format: "%.3f", Double(b.payload)*8/Double(n))) b/val), scales \(b.scales) B (\(String(format: "%.3f", scaleBitsPerValue)) b/val = metadata)
            step1 fp16 scales: \(String(format: "%.3f", fp16BitsPerValue)) bits/value  cosine \(String(format: "%.6f", fp16Cos))  (saves \(String(format: "%.3f", bitsPerValue - fp16BitsPerValue)) b/val, cosΔ \(String(format: "%.2e", baseCos - fp16Cos)))
            step2 one-norm/vec (analytic): \(String(format: "%.3f", oneNormBitsPerValue)) bits/value  (saves \(String(format: "%.3f", bitsPerValue - oneNormBitsPerValue)) b/val of metadata)
            remaining 4–7x lever: payload quantizer (data-free Gaussian Lloyd-Max) — payload is \(String(format: "%.0f", Double(b.payload)*100/Double(b.total)))%% of bytes
            """)

        // Gates: fp16-scale storage must be quality-safe (negligible cosine loss) and reduce bytes.
        XCTAssertGreaterThan(fp16Cos, baseCos - 1e-3, "fp16 scales degraded quality beyond 1e-3 cosine")
        XCTAssertLessThan(fp16BitsPerValue, bitsPerValue, "fp16 scales did not reduce bytes")
        XCTAssertLessThanOrEqual(oneNormBitsPerValue, bitsPerValue, "one-norm should not increase bytes")
        XCTAssertGreaterThan(baseCos, 0.9, "reference codec baseline cosine implausibly low")
    }

    /// Data-free Gaussian Lloyd-Max centroids for `levels` (= 2^bits) reproduction points,
    /// computed from the N(0,1) pdf on a fine grid (no data dependence). This is the optimal
    /// scalar quantizer for a Gaussian source — the post-rotation distribution QJL produces.
    private func gaussianLloydMax(levels: Int, iterations: Int = 80) -> [Float] {
        let g = 8192
        let xs = (0 ..< g).map { -6.0 + 12.0 * Double($0) / Double(g - 1) }
        let w = xs.map { exp(-0.5 * $0 * $0) }  // unnormalized Gaussian density (data-free)
        var c = (0 ..< levels).map { -3.0 + 6.0 * Double($0) / Double(max(1, levels - 1)) }
        for _ in 0 ..< iterations {
            var sum = [Double](repeating: 0, count: levels)
            var cnt = [Double](repeating: 0, count: levels)
            for i in 0 ..< g {
                var best = 0
                var bd = Double.infinity
                for k in 0 ..< levels {
                    let d = abs(xs[i] - c[k])
                    if d < bd { bd = d; best = k }
                }
                sum[best] += xs[i] * w[i]
                cnt[best] += w[i]
            }
            for k in 0 ..< levels where cnt[k] > 0 { c[k] = sum[k] / cnt[k] }
        }
        return c.map { Float($0) }
    }

    func testDataFreeGaussianLloydMaxIsThePayloadLever() throws {
        // Same Gaussian source as the diet baseline (8.0 bits/value, cosine 0.998762).
        let n = 4096
        MLXRandom.seed(0xD1E7)
        let original = ((MLXRandom.normal([n]) * 1.0).asType(.float32)).asArray(Float.self)

        // Per-vector norm (the single fp16 scale this quantizer needs — "one norm per vector").
        let sigma = (original.reduce(0) { $0 + Double($1) * Double($1) } / Double(n)).squareRoot()
        let sigmaF = Float(sigma)

        func quantizeCosineBits(bits: Int) -> (cos: Float, bitsPerValue: Double) {
            let levels = 1 << bits
            let centroids = gaussianLloydMax(levels: levels)
            var decoded = [Float](repeating: 0, count: n)
            for i in 0 ..< n {
                let xn = original[i] / sigmaF  // normalize to ~N(0,1)
                var best = 0
                var bd = Float.infinity
                for k in 0 ..< levels {
                    let d = abs(xn - centroids[k])
                    if d < bd { bd = d; best = k }
                }
                decoded[i] = centroids[best] * sigmaF  // dequantize
            }
            // payload = bits/value; metadata = one fp16 norm per 4096-vector (negligible).
            let bpv = Double(bits) + 16.0 / Double(n)
            return (cosine(original, decoded), bpv)
        }

        // The current reference codec: 8.0 bits/value total, cosine 0.998762 (measured above).
        let codecBitsPerValue = 8.0
        let codecCosine: Float = 0.998762
        print("\n=== N4 data-free Gaussian Lloyd-Max payload quantizer (vs current \(codecBitsPerValue) b/val codec, cosine \(codecCosine)) ===")
        var results: [(bits: Int, cos: Float, bpv: Double)] = []
        for bits in [3, 4, 5] {
            let r = quantizeCosineBits(bits: bits)
            results.append((bits, r.cos, r.bitsPerValue))
            print(String(
                format: "  %d-bit: %.3f bits/value  (%.2fx vs fp16)  cosine %.6f", bits, r.bitsPerValue,
                16.0 / r.bitsPerValue, r.cos))
        }
        let fiveBit = results.first { $0.bits == 5 }!
        let threeBit = results.first { $0.bits == 3 }!
        print(String(
            format:
                "  => EQUAL-QUALITY: 5-bit (%.2fx) matches the codec's cosine (%.6f ≈ %.6f) at %.0f%% fewer bits\n"
                + "     -> the payload is over-coded by ~%.1f bits/value; optimal quantizer = ~%.1fx vs the codec's 2.0x at equal quality.\n"
                + "     LOWER-QUALITY operating points reach paper range: 3-bit %.2fx (cosine %.4f).\n"
                + "     Conclusion: the 4–7x lever is the PAYLOAD quantizer (81%% of bytes), NOT the scale metadata.",
            16.0 / fiveBit.bpv, fiveBit.cos, codecCosine, (codecBitsPerValue - fiveBit.bpv) / codecBitsPerValue * 100,
            codecBitsPerValue - fiveBit.bpv, 16.0 / fiveBit.bpv, 16.0 / threeBit.bpv, threeBit.cos))

        // Gates: at ~equal quality the optimal data-free quantizer beats the codec's bits/value,
        // and an aggressive operating point reaches paper-range compression.
        XCTAssertGreaterThan(fiveBit.cos, 0.9985, "5-bit Gaussian Lloyd-Max should ~match the codec quality")
        XCTAssertLessThan(fiveBit.bpv, codecBitsPerValue, "optimal quantizer should beat the codec's bits/value at equal quality")
        XCTAssertGreaterThan(16.0 / threeBit.bpv, 4.0, "3-bit should reach paper-range compression")
        XCTAssertGreaterThan(threeBit.cos, 0.97, "3-bit quality should remain usable")
    }

    /// Validates the INTEGRATED production codec format (.gaussianLloydMax): encode via the new
    /// `turboQuantGaussianReferenceEncode` and decode via the standard `turboQuantReferenceDecode`
    /// dispatch. Confirms round-trip correctness + that the real packed-storage path matches the
    /// standalone quantizer quality, and that `storageByteCount` reflects the diet.
    func testIntegratedGaussianCodecRoundTrip() throws {
        let n = 4096
        MLXRandom.seed(0xD1E7)
        let arr = ((MLXRandom.normal([n]) * 1.0).asType(.float32))
        let original = arr.asArray(Float.self)

        print("\n=== N4 integrated codec (.gaussianLloydMax) round-trip ===")
        for bits in [3, 4, 5] {
            let code = try turboQuantGaussianReferenceEncode(arr, bits: bits, groupSize: 64)
            XCTAssertEqual(code.format, .gaussianLloydMax)
            let decoded = try turboQuantReferenceDecode(code).asArray(Float.self)  // standard dispatch
            XCTAssertEqual(decoded.count, n)
            let cos = cosine(original, decoded)
            let bpv = Double(code.storageByteCount) * 8.0 / Double(n)
            print(String(
                format: "  %d-bit: %.3f bits/value (%.2fx vs fp16, incl. norms)  cosine %.6f", bits,
                bpv, 16.0 / bpv, cos))
            // Real packed path must match the standalone quantizer quality (within noise) and
            // stay near the nominal payload bits (+ tiny per-group norm metadata).
            XCTAssertGreaterThan(cos, bits == 3 ? 0.97 : (bits == 4 ? 0.99 : 0.998))
            XCTAssertLessThan(bpv, Double(bits) + 1.0, "storage should be ~bits/value + small norm metadata")
            XCTAssertGreaterThan(16.0 / bpv, 2.0, "integrated codec must beat fp16 by >2x")
        }
        print("  => integrated production codec format works end-to-end (encode + standard decode dispatch).")
    }

    /// T1.4 stage 1: K scale plane diet, scalesPerGroup 3 -> 2 (the dead third slot, always
    /// written 0.0 and never read, is removed). Confirms both the Metal attention codec's
    /// 5-D scale plane (`[batch, kvHeads, capacity, groupsPerVector, scalesPerGroup]`) and the
    /// flat Metal codec's `[groupCount, scalesPerGroup]` layout now report scalesPerGroup == 2
    /// for the key role, matching the shrunk allocation. Pattern mirrors
    /// TurboQuantNativeAttentionTests.swift's `testKeyPageSummariesMatchScaleMaxReference`.
    func testKeyScalePlaneDietDropsThirdScaleSlot() throws {
        try requireTurboQuantMetalAttention()

        let tokenCount = 128
        let headDimension = 64
        MLXRandom.seed(0x5CA1_E000_0000_0002)
        let keys = (MLXRandom.normal([1, 1, tokenCount, headDimension]) * 1.0).asType(.float32)

        let keyCode = try turboQuantMetalEncodeAttention(
            keys,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .key,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 0x5CA1_E000_0000_0003
            )
        )
        XCTAssertEqual(keyCode.scalesPerGroup, 2)
        // Attention scale plane is 5-D; last dim is scalesPerGroup.
        XCTAssertEqual(keyCode.scales.ndim, 5)
        XCTAssertEqual(keyCode.scales.dim(4), 2)

        try requireTurboQuantMetalCodec()
        let flatValues = (MLXRandom.normal([4096]) * 1.0).asType(.float32)
        let flatKeyCode = try turboQuantMetalEncode(
            flatValues,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .key,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 0x5CA1_E000_0000_0004
            )
        )
        XCTAssertEqual(flatKeyCode.scalesPerGroup, 2)
        XCTAssertEqual(flatKeyCode.scales.dim(1), 2)
    }
}
