import MLX
import Testing

@testable import MLXTimeSeries

@Suite("CausalPatchScaler Tests")
struct ScalerTests {

    @Test("Scaler output shapes match input dimensions")
    func testOutputShapes() {
        let scaler = CausalPatchScaler(patchSize: 64)
        let B = 2, V = 3, T = 192  // 3 patches of 64
        let data = MLXArray.ones([B, V, T])
        let mask = MLXArray.ones([B, V, T])

        let (normalized, locs, scales, _, _) = scaler(data, mask: mask)

        #expect(normalized.shape == [B, V, T])
        #expect(locs.shape == [B, V, 3])  // 3 patches
        #expect(scales.shape == [B, V, 3])
    }

    @Test("Scaler enforces minimum scale")
    func testMinimumScale() {
        let scaler = CausalPatchScaler(patchSize: 64, minimumScale: 1e-5)
        let B = 1, V = 1, T = 64
        // Constant data → std = 0 → should be clamped to minimumScale
        let data = MLXArray.ones([B, V, T]) * 5.0
        let mask = MLXArray.ones([B, V, T])

        let (_, _, scales, _, _) = scaler(data, mask: mask)

        let scaleValue = scales[0, 0, 0].item(Float.self)
        #expect(scaleValue >= 1e-5)
    }

    @Test("Scaler is causal: changing future data doesn't affect past normalization")
    func testCausalProperty() {
        let scaler = CausalPatchScaler(patchSize: 64)
        let B = 1, V = 1, T = 128  // 2 patches

        // Dataset 1: [1, 1, ..., 1, 1, ..., 1]
        let data1 = MLXArray.ones([B, V, T])
        let mask = MLXArray.ones([B, V, T])
        let (norm1, locs1, scales1, _, _) = scaler(data1, mask: mask)

        // Dataset 2: same first patch, different second patch
        var values2 = [Float](repeating: 1.0, count: T)
        for i in 64 ..< 128 { values2[i] = 100.0 }
        let data2 = MLXArray(values2).reshaped(B, V, T)
        let (norm2, locs2, scales2, _, _) = scaler(data2, mask: mask)

        // The loc/scale for the FIRST patch should be identical
        let loc1_p0 = locs1[0, 0, 0].item(Float.self)
        let loc2_p0 = locs2[0, 0, 0].item(Float.self)
        #expect(abs(loc1_p0 - loc2_p0) < 1e-6)

        let scale1_p0 = scales1[0, 0, 0].item(Float.self)
        let scale2_p0 = scales2[0, 0, 0].item(Float.self)
        #expect(abs(scale1_p0 - scale2_p0) < 1e-6)
    }

    @Test("Scaler handles masked (padded) regions")
    func testMaskedRegions() {
        let scaler = CausalPatchScaler(patchSize: 64)
        let B = 1, V = 1, T = 128

        // Data with first 32 positions masked (padded)
        var values = [Float](repeating: 5.0, count: T)
        var maskValues = [Float](repeating: 1.0, count: T)
        for i in 0 ..< 32 {
            values[i] = 999.0  // Shouldn't affect stats
            maskValues[i] = 0.0  // Masked out
        }

        let data = MLXArray(values).reshaped(B, V, T)
        let mask = MLXArray(maskValues).reshaped(B, V, T)
        let (_, locs, _, _, _) = scaler(data, mask: mask)

        // The mean should be close to 5.0, not influenced by 999.0
        let loc = locs[0, 0, 0].item(Float.self)
        #expect(abs(loc - 5.0) < 0.1)
    }
}
