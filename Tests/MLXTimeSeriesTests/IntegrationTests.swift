import MLX
import MLXNN
import Testing

@testable import MLXTimeSeries

@Suite("Integration Tests")
struct IntegrationTests {

    @Test("Small model forward pass produces correct output shapes")
    func testSmallModelForwardPass() {
        // Create a small model for fast testing
        let config = TotoConfiguration(
            patchSize: 4,
            stride: 4,
            embedDim: 32,
            numLayers: 2,
            numHeads: 4,
            mlpHiddenDim: 64,
            spacewiseEveryNLayers: 2,  // Layer 1 is space-wise
            kComponents: 2
        )

        let model = TotoModel(config)
        let B = 1, V = 2, T = 8  // 2 patches of 4

        // Create synthetic input
        let series = MLXArray.zeros([B, V, T])
        let mask = MLXArray.ones([B, V, T])

        // Normalize
        let scaler = CausalPatchScaler(patchSize: config.patchSize)
        let (normalized, _, _,  _, _) = scaler(series, mask: mask)

        // Embed
        let embedded = model.backbone.embed(normalized)
        #expect(embedded.shape == [B, V, 2, config.embedDim])  // 2 patches

        // Forward through backbone
        let caches = makeTimeSeriesCaches(config)
        let output = model.backbone(
            patchEmbedded: embedded,
            caches: caches,
            idMask: MLXArray.zeros([B, V], type: Int32.self)
        )

        // Check distribution parameter shapes
        // [B, V, nPatches, patchSize, kComponents]
        #expect(output.params.df.shape == [B, V, 2, config.patchSize, config.kComponents])
        #expect(output.params.loc.shape == [B, V, 2, config.patchSize, config.kComponents])
        #expect(output.params.weights.shape == [B, V, 2, config.patchSize, config.kComponents])

        // Check hidden state shape
        #expect(output.hidden.shape == [B, V, 2, config.embedDim])
    }

    @Test("KV cache works across multiple forward passes")
    func testCachedForwardPass() {
        let config = TotoConfiguration(
            patchSize: 4,
            stride: 4,
            embedDim: 32,
            numLayers: 2,
            numHeads: 4,
            mlpHiddenDim: 64,
            spacewiseEveryNLayers: 2,
            kComponents: 2
        )

        let model = TotoModel(config)
        let B = 1, V = 1

        // First pass: 2 patches
        let embedded1 = MLXArray.zeros([B, V, 2, config.embedDim])
        let caches = makeTimeSeriesCaches(config)

        let output1 = model.backbone(
            patchEmbedded: embedded1,
            caches: caches,
            idMask: MLXArray.zeros([B, V], type: Int32.self)
        )
        eval(output1.hidden)

        // Verify cache was filled (time-wise layer should have offset=2)
        for (i, cache) in caches.enumerated() {
            if config.isSpacewise(layerIndex: i) {
                #expect(cache == nil)
            } else {
                #expect(cache?.offset == 2, "Layer \(i) cache should have offset 2")
            }
        }

        // Second pass: 1 more patch (autoregressive step)
        let embedded2 = MLXArray.zeros([B, V, 1, config.embedDim])
        let output2 = model.backbone(
            patchEmbedded: embedded2,
            caches: caches,
            idMask: MLXArray.zeros([B, V], type: Int32.self)
        )
        eval(output2.hidden)

        // Cache should now have offset=3
        for (i, cache) in caches.enumerated() {
            if !config.isSpacewise(layerIndex: i) {
                #expect(cache?.offset == 3, "Layer \(i) cache should have offset 3")
            }
        }
    }

    @Test("End-to-end forecast with small model")
    func testEndToEndForecast() {
        let config = TotoConfiguration(
            patchSize: 4,
            stride: 4,
            embedDim: 32,
            numLayers: 2,
            numHeads: 4,
            mlpHiddenDim: 64,
            spacewiseEveryNLayers: 2,
            kComponents: 2
        )

        let model = TotoModel(config)
        let forecaster = TotoForecaster(model: model, config: config)

        // Create a simple input: 8 time steps of data
        let input = TimeSeriesInput.univariate([Float](repeating: 1.0, count: 8))

        // Forecast 4 steps ahead
        let result = forecaster.forecast(input: input, predictionLength: 4)

        #expect(result.mean.shape == [1, 1, 4])
        #expect(result.predictionLength == 4)
    }

    @Test("RoPE XPOS produces consistent shapes")
    func testRoPEXPOS() {
        let rope = RoPEXPOS(dims: 64)

        let B = 2, nHeads = 12, L = 4
        let q = MLXArray.zeros([B, nHeads, L, 64])
        let k = MLXArray.zeros([B, nHeads, L, 64])

        let (rotQ, rotK) = rope(queries: q, keys: k, offset: 0)
        #expect(rotQ.shape == [B, nHeads, L, 64])
        #expect(rotK.shape == [B, nHeads, L, 64])

        // Test with offset (cached generation)
        let q2 = MLXArray.zeros([B, nHeads, 1, 64])
        let k2 = MLXArray.zeros([B, nHeads, 1, 64])
        let (rotQ2, rotK2) = rope(queries: q2, keys: k2, offset: 4)
        #expect(rotQ2.shape == [B, nHeads, 1, 64])
        #expect(rotK2.shape == [B, nHeads, 1, 64])
    }
}
