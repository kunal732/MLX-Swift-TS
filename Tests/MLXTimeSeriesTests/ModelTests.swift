import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXTimeSeries

@Suite("Model Component Tests")
struct ModelTests {

    let config = TotoConfiguration()

    @Test("Configuration default values match Toto-Open-Base-1.0")
    func testDefaultConfig() {
        #expect(config.patchSize == 64)
        #expect(config.stride == 64)
        #expect(config.embedDim == 768)
        #expect(config.numLayers == 12)
        #expect(config.numHeads == 12)
        #expect(config.mlpHiddenDim == 3072)
        #expect(config.kComponents == 24)
        #expect(config.headDim == 64)
    }

    @Test("Configuration decodes from JSON")
    func testConfigDecoding() throws {
        let json = """
            {
                "patch_size": 64,
                "stride": 64,
                "embed_dim": 768,
                "num_layers": 12,
                "num_heads": 12,
                "mlp_hidden_dim": 3072,
                "dropout": 0.1,
                "spacewise_every_n_layers": 12,
                "spacewise_first": false,
                "output_distribution_kwargs": {"k_components": 24},
                "use_memory_efficient_attention": true,
                "stabilize_with_global": true,
                "scale_factor_exponent": 10.0
            }
            """
        let data = json.data(using: .utf8)!
        let decoded = try JSONDecoder().decode(TotoConfiguration.self, from: data)

        #expect(decoded.patchSize == 64)
        #expect(decoded.kComponents == 24)
        #expect(decoded.spacewiseFirst == false)
    }

    @Test("Space-wise layer detection")
    func testSpacewiseDetection() {
        // Default: spacewise_every_n_layers=12, spacewise_first=false
        // Layers 0-10 are time-wise, layer 11 is space-wise
        for i in 0 ..< 11 {
            #expect(config.isSpacewise(layerIndex: i) == false, "Layer \(i) should be time-wise")
        }
        #expect(config.isSpacewise(layerIndex: 11) == true, "Layer 11 should be space-wise")
    }

    @Test("PatchEmbedding output shape")
    func testPatchEmbedding() {
        let embed = PatchEmbedding(
            patchSize: config.patchSize, stride: config.stride, embedDim: config.embedDim)
        let B = 2, V = 3, T = 192
        let x = MLXArray.zeros([B, V, T])
        let out = embed(x)
        #expect(out.shape == [B, V, 3, config.embedDim])  // 192/64 = 3 patches
    }

    @Test("SwiGLU MLP output shape")
    func testSwiGLUMLP() {
        let mlp = SwiGLUMLP(embedDim: config.embedDim, mlpHiddenDim: config.mlpHiddenDim)
        let x = MLXArray.zeros([2, 4, config.embedDim])
        let out = mlp(x)
        #expect(out.shape == [2, 4, config.embedDim])
    }

    @Test("TotoAttention output shape (time-wise)")
    func testTimewiseAttention() {
        let attn = TotoAttention(config, isSpacewise: false)
        let rope = RoPEXPOS(dims: config.headDim)
        let x = MLXArray.zeros([6, 3, config.embedDim])  // B*V=6, nPatches=3
        let out = attn(x, cache: nil, idMask: nil, rope: rope)
        #expect(out.shape == [6, 3, config.embedDim])
    }

    @Test("TotoAttention output shape (space-wise)")
    func testSpacewiseAttention() {
        let attn = TotoAttention(config, isSpacewise: true)
        let V = 4
        let x = MLXArray.zeros([9, V, config.embedDim])  // B*nPatches=9, V=4
        let idMask = MLXArray.zeros([9, V], type: Int32.self)
        let out = attn(x, cache: nil, idMask: idMask, rope: nil)
        #expect(out.shape == [9, V, config.embedDim])
    }

    @Test("TransformerBlock output shape")
    func testTransformerBlock() {
        let block = TotoTransformerBlock(config, isSpacewise: false)
        let rope = RoPEXPOS(dims: config.headDim)
        let B = 2, V = 3, L = 4
        let x = MLXArray.zeros([B, V, L, config.embedDim])
        let out = block(x, cache: nil, idMask: nil, rope: rope)
        #expect(out.shape == [B, V, L, config.embedDim])
    }

    @Test("MixtureOfStudentTs output shape")
    func testMixtureHead() {
        let head = MixtureOfStudentTs(embedDim: config.embedDim, kComponents: config.kComponents)
        let x = MLXArray.zeros([2, 3, 4, 64, config.embedDim])  // [B, V, nPatches, patchSize, D]
        let params = head(x)

        #expect(params.df.shape == [2, 3, 4, 64, config.kComponents])
        #expect(params.loc.shape == [2, 3, 4, 64, config.kComponents])
        #expect(params.scale.shape == [2, 3, 4, 64, config.kComponents])
        #expect(params.weights.shape == [2, 3, 4, 64, config.kComponents])
    }

    @Test("MixtureParams mean computation")
    func testMixtureMean() {
        let loc = MLXArray([Float(1.0), 2.0, 3.0]).reshaped(1, 1, 3)
        let weights = MLXArray([Float(0.5), 0.3, 0.2]).reshaped(1, 1, 3)
        let params = MixtureParams(
            df: MLXArray.ones([1, 1, 3]) * 5,
            loc: loc,
            scale: MLXArray.ones([1, 1, 3]),
            weights: weights
        )
        let mean = params.mean()
        // 0.5*1 + 0.3*2 + 0.2*3 = 0.5 + 0.6 + 0.6 = 1.7
        let value = mean[0, 0].item(Float.self)
        #expect(abs(value - 1.7) < 1e-5)
    }

    @Test("KV cache accumulates correctly")
    func testKVCache() {
        let cache = TimeSeriesKVCache()
        #expect(cache.offset == 0)

        let k1 = MLXArray.ones([1, 12, 3, 64])  // B=1, nHeads=12, L=3, headDim=64
        let v1 = MLXArray.ones([1, 12, 3, 64])
        let (keys1, _) = cache.update(keys: k1, values: v1)
        #expect(cache.offset == 3)
        #expect(keys1.shape == [1, 12, 3, 64])

        let k2 = MLXArray.ones([1, 12, 1, 64])
        let v2 = MLXArray.ones([1, 12, 1, 64])
        let (keys2, _) = cache.update(keys: k2, values: v2)
        #expect(cache.offset == 4)
        #expect(keys2.shape == [1, 12, 4, 64])
    }

    @Test("TimeSeriesInput univariate convenience")
    func testUnivariateInput() {
        let input = TimeSeriesInput.univariate([1.0, 2.0, 3.0, 4.0])
        #expect(input.series.shape == [1, 1, 4])
        #expect(input.paddingMask.shape == [1, 1, 4])
        #expect(input.idMask.shape == [1, 1])
    }

    @Test("TimeSeriesInput padding to patch size")
    func testInputPadding() {
        // 100 time steps, patch_size=64 → should pad to 128
        let input = TimeSeriesInput.univariate([Float](repeating: 1.0, count: 100))
        let padded = input.padded(toPatchSize: 64)
        #expect(padded.series.shape == [1, 1, 128])
        #expect(padded.paddingMask.shape == [1, 1, 128])

        // First 28 positions should be zero (padding mask)
        let maskPrefix = padded.paddingMask[0, 0, 0 ..< 28]
        let maskSum = maskPrefix.sum().item(Float.self)
        #expect(maskSum == 0.0)

        // Remaining 100 positions should be 1
        let maskSuffix = padded.paddingMask[0, 0, 28 ..< 128]
        let maskSuffixSum = maskSuffix.sum().item(Float.self)
        #expect(maskSuffixSum == 100.0)
    }

    @Test("Full TotoModel weight key paths")
    func testModelKeyPaths() {
        let smallConfig = TotoConfiguration(
            patchSize: 4, stride: 4, embedDim: 32, numLayers: 2, numHeads: 4,
            mlpHiddenDim: 64, kComponents: 2)
        let model = TotoModel(smallConfig)

        // Get all parameter keys
        let keys = flattenKeys(model.parameters())

        // Check expected key paths exist
        #expect(keys.contains("model.patch_embed.projection.weight"))
        #expect(keys.contains("model.patch_embed.projection.bias"))
        #expect(keys.contains("model.transformer.layers.0.norm1.weight"))
        #expect(keys.contains("model.transformer.layers.0.attention.wQKV.weight"))
        #expect(keys.contains("model.transformer.layers.0.mlp.gate_up.weight"))
        #expect(keys.contains("model.transformer.layers.0.mlp.down.weight"))
        #expect(keys.contains("model.unembed.weight"))
        #expect(keys.contains("model.output_distribution.df.weight"))
        #expect(keys.contains("model.output_distribution.loc_proj.weight"))
    }

    @Test("Sanitize remaps RMSNorm scale to weight")
    func testSanitize() {
        let model = TotoModel(config)

        let weights: [String: MLXArray] = [
            "model.transformer.layers.0.norm1.scale": MLXArray.ones([768]),
            "model.transformer.layers.0.norm2.scale": MLXArray.ones([768]),
            "model.transformer.layers.0.attention.wQKV.weight": MLXArray.ones([2304, 768]),
            "model.transformer.rotary_emb.freqs": MLXArray.ones([32]),
        ]

        let sanitized = model.sanitize(weights: weights)

        #expect(sanitized["model.transformer.layers.0.norm1.weight"] != nil)
        #expect(sanitized["model.transformer.layers.0.norm2.weight"] != nil)
        #expect(sanitized["model.transformer.layers.0.norm1.scale"] == nil)
        #expect(sanitized["model.transformer.rotary_emb.freqs"] == nil)
        #expect(sanitized["model.transformer.layers.0.attention.wQKV.weight"] != nil)
    }

}

/// Helper to get all parameter key paths from a module.
private func flattenKeys(_ params: ModuleParameters) -> Set<String> {
    Set(params.flattened().map { $0.0 })
}
