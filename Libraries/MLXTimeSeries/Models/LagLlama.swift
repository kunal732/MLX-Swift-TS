import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - Configuration

/// Configuration for the Lag-Llama time series model.
public struct LagLlamaConfiguration: Codable, Sendable {
    public var dModel: Int
    public var intermediateSize: Int
    public var numLayers: Int
    public var numAttentionHeads: Int
    public var contextLength: Int
    public var predictionLength: Int
    public var ropeTheta: Float
    public var lagsSequence: [Int]

    public var headDim: Int { dModel / numAttentionHeads }

    enum CodingKeys: String, CodingKey {
        case dModel = "d_model"
        case intermediateSize = "intermediate_size"
        case numLayers = "num_layers"
        case numAttentionHeads = "num_attention_heads"
        case contextLength = "context_length"
        case predictionLength = "prediction_length"
        case ropeTheta = "rope_theta"
        case lagsSequence = "lags_sequence"
    }

    public init(
        dModel: Int = 256,
        intermediateSize: Int = 1024,
        numLayers: Int = 2,
        numAttentionHeads: Int = 2,
        contextLength: Int = 32,
        predictionLength: Int = 24,
        ropeTheta: Float = 10000.0,
        lagsSequence: [Int] = [1, 2, 3, 4, 5, 6, 7, 24, 168, 720, 4320, 8760]
    ) {
        self.dModel = dModel
        self.intermediateSize = intermediateSize
        self.numLayers = numLayers
        self.numAttentionHeads = numAttentionHeads
        self.contextLength = contextLength
        self.predictionLength = predictionLength
        self.ropeTheta = ropeTheta
        self.lagsSequence = lagsSequence
    }
}

// MARK: - Lag Feature Extraction

/// Extracts lag features from a time series.
///
/// For each time step, gathers values at specific lag indices in the past.
/// This creates a feature vector of historical values at predetermined offsets.
public struct LagFeatureExtractor: Sendable {
    let lags: [Int]

    /// Number of lag features (input dimension to the model).
    public var numFeatures: Int { lags.count + 1 }  // lags + current value

    public init(lags: [Int]) {
        self.lags = lags
    }

    /// Extract lag features from a time series.
    ///
    /// - Parameters:
    ///   - values: Time series values, shape `[B, T]`.
    ///   - timeIndex: Which time steps to extract features for, shape `[L]`.
    /// - Returns: Feature matrix of shape `[B, L, numFeatures]`.
    public func callAsFunction(_ values: MLXArray, contextLength: Int) -> MLXArray {
        let B = values.dim(0)
        let T = values.dim(1)

        // We need enough history to look back by the maximum lag
        let maxLag = lags.max() ?? 0
        let startIdx = max(maxLag, 0)
        let L = min(T - startIdx, contextLength)

        var features = [MLXArray]()

        // Current values
        let currentSlice = values[.ellipsis, startIdx ..< (startIdx + L)]
        features.append(currentSlice.expandedDimensions(axis: -1))

        // Lag features
        for lag in lags {
            let lagStart = startIdx - lag
            let lagEnd = lagStart + L
            if lagStart >= 0 && lagEnd <= T {
                let lagSlice = values[.ellipsis, lagStart ..< lagEnd]
                features.append(lagSlice.expandedDimensions(axis: -1))
            } else {
                // Pad with zeros if lag goes before the start of the series
                features.append(MLXArray.zeros([B, L, 1]))
            }
        }

        return MLX.concatenated(features, axis: -1)  // [B, L, numFeatures]
    }
}

// MARK: - Lag-Llama Components

/// Lag-Llama attention (standard causal with RoPE, reuses shared RoPE).
class LagLlamaAttention: Module {
    @ModuleInfo(key: "q_proj") var wQ: Linear
    @ModuleInfo(key: "k_proj") var wK: Linear
    @ModuleInfo(key: "v_proj") var wV: Linear
    @ModuleInfo(key: "o_proj") var wO: Linear

    let numHeads: Int
    let headDim: Int
    let scale: Float

    init(_ config: LagLlamaConfiguration) {
        self.numHeads = config.numAttentionHeads
        self.headDim = config.headDim
        self.scale = pow(Float(config.headDim), -0.5)
        let totalDim = numHeads * headDim
        self._wQ.wrappedValue = Linear(config.dModel, totalDim, bias: false)
        self._wK.wrappedValue = Linear(config.dModel, totalDim, bias: false)
        self._wV.wrappedValue = Linear(config.dModel, totalDim, bias: false)
        self._wO.wrappedValue = Linear(totalDim, config.dModel, bias: false)
    }

    func callAsFunction(
        _ x: MLXArray,
        rope: RoPEXPOS,
        cache: TimeSeriesKVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        var q = wQ(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        var k = wK(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        var v = wV(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)

        // Apply RoPE (reuse shared implementation, XPOS scaling is fine for small models)
        let offset = cache?.offset ?? 0
        (q, k) = rope(queries: q, keys: k, offset: offset)

        if let cache {
            let (cachedK, cachedV) = cache.update(keys: k, values: v)
            k = cachedK
            v = cachedV
        }

        let mask: MLXFast.ScaledDotProductAttentionMaskMode = (L == 1) ? .none : .causal

        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask
        )

        let out = output.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return wO(out)
    }
}

/// Lag-Llama transformer block (Llama-style: RMSNorm + Attention + SwiGLU).
class LagLlamaBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: RMSNorm
    @ModuleInfo(key: "norm2") var norm2: RMSNorm
    @ModuleInfo(key: "attention") var attention: LagLlamaAttention
    @ModuleInfo(key: "mlp") var mlp: SwiGLUMLP

    init(_ config: LagLlamaConfiguration) {
        self._norm1.wrappedValue = RMSNorm(dimensions: config.dModel)
        self._norm2.wrappedValue = RMSNorm(dimensions: config.dModel)
        self._attention.wrappedValue = LagLlamaAttention(config)
        self._mlp.wrappedValue = SwiGLUMLP(
            embedDim: config.dModel, mlpHiddenDim: config.intermediateSize)
    }

    func callAsFunction(
        _ x: MLXArray, rope: RoPEXPOS, cache: TimeSeriesKVCache?
    ) -> MLXArray {
        let h = x + attention(norm1(x), rope: rope, cache: cache)
        return h + mlp(norm2(h))
    }
}

// MARK: - LagLlamaModel

/// Lag-Llama time series forecasting model.
///
/// Llama-style decoder that uses lag features as input.
/// Produces a single Student-T distribution per time step.
/// Very small (2.5M params), loads instantly.
public class LagLlamaModel: Module, TimeSeriesModel {

    @ModuleInfo(key: "input_proj") var inputProj: Linear
    @ModuleInfo(key: "layers") var layers: [LagLlamaBlock]
    @ModuleInfo(key: "final_norm") var finalNorm: RMSNorm
    @ModuleInfo(key: "distribution_head") var distributionHead: StudentTHead

    let config: LagLlamaConfiguration
    let rope: RoPEXPOS
    let lagExtractor: LagFeatureExtractor

    public init(_ config: LagLlamaConfiguration) {
        self.config = config
        self.rope = RoPEXPOS(dims: config.headDim, theta: config.ropeTheta)
        self.lagExtractor = LagFeatureExtractor(lags: config.lagsSequence)

        let inputDim = lagExtractor.numFeatures
        self._inputProj.wrappedValue = Linear(inputDim, config.dModel)
        self._layers.wrappedValue = (0 ..< config.numLayers).map { _ in LagLlamaBlock(config) }
        self._finalNorm.wrappedValue = RMSNorm(dimensions: config.dModel)
        self._distributionHead.wrappedValue = StudentTHead(inputDim: config.dModel)
    }

    public func forecast(
        input: TimeSeriesInput,
        predictionLength: Int,
        caches: [TimeSeriesKVCache?]
    ) -> TimeSeriesPrediction {
        let series = input.series  // [B, V, T]
        let B = series.dim(0)
        let V = series.dim(1)

        // Flatten to [B*V, T] for univariate processing
        let flat = series.reshaped(B * V, -1)

        // Compute mean/scale for normalization
        let mask = input.paddingMask.reshaped(B * V, -1)
        let nObs = MLX.maximum(mask.sum(axis: -1, keepDims: true), MLXArray(Float(1)))
        let meanVal = (flat * mask).sum(axis: -1, keepDims: true) / nObs
        let variance = ((flat - meanVal) * (flat - meanVal) * mask).sum(axis: -1, keepDims: true) / nObs
        let scaleVal = MLX.maximum(MLX.sqrt(variance), MLXArray(Float(1e-5)))

        let normalized = (flat - meanVal) / scaleVal

        // Extract lag features
        let lagFeatures = lagExtractor(normalized, contextLength: config.contextLength)
        // lagFeatures: [B*V, L, numFeatures]

        // Project to hidden dim
        var hidden = inputProj(lagFeatures)  // [B*V, L, dModel]

        // Transformer layers
        for (i, layer) in layers.enumerated() {
            let cache = caches.indices.contains(i) ? caches[i] : nil
            hidden = layer(hidden, rope: rope, cache: cache)
        }
        hidden = finalNorm(hidden)

        // Autoregressive prediction
        var allPredictions = [MLXArray]()
        var lastHidden = hidden[.ellipsis, (hidden.dim(1) - 1), 0...]  // [B*V, dModel]
        lastHidden = lastHidden.expandedDimensions(axis: 1)  // [B*V, 1, dModel]

        for _ in 0 ..< predictionLength {
            let params = distributionHead(lastHidden)  // StudentTParams with shape [B*V, 1]
            let pred = params.mean()  // [B*V, 1]
            allPredictions.append(pred)

            // Feed prediction back as a simple input (using zero lags as approximation)
            let predFeatures = MLXArray.zeros([B * V, 1, lagExtractor.numFeatures])
            // Set current value feature
            let currentVal = pred.expandedDimensions(axis: -1)
            // Use concatenation to set the first feature
            let restFeatures = predFeatures[.ellipsis, 1...]
            let nextFeatures = MLX.concatenated([currentVal, restFeatures], axis: -1)
            lastHidden = inputProj(nextFeatures)

            for (i, layer) in layers.enumerated() {
                let cache = caches.indices.contains(i) ? caches[i] : nil
                lastHidden = layer(lastHidden, rope: rope, cache: cache)
            }
            lastHidden = finalNorm(lastHidden)
        }

        // Stack predictions: [B*V, predictionLength]
        let predictions = MLX.concatenated(allPredictions, axis: -1)

        // Denormalize
        let denormalized = predictions * scaleVal + meanVal

        // Reshape: [B, V, predictionLength]
        let meanOut = denormalized.reshaped(B, V, predictionLength)

        return TimeSeriesPrediction(
            mean: meanOut,
            predictionLength: predictionLength
        )
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = [String: MLXArray]()
        for (key, value) in weights {
            // Skip RoPE buffers
            if key.contains("rotary_emb") { continue }

            var newKey = key
            // Strip "model." prefix from Lightning state_dict
            if newKey.hasPrefix("model.") {
                newKey = String(newKey.dropFirst("model.".count))
            }
            // Remap GluonTS naming
            newKey = newKey.replacingOccurrences(of: "backbone.", with: "")
            newKey = newKey.replacingOccurrences(of: "self_attn.", with: "attention.")
            newKey = newKey.replacingOccurrences(of: "mlp.fc1.", with: "mlp.gate_up.")
            newKey = newKey.replacingOccurrences(of: "mlp.fc2.", with: "mlp.down.")
            newKey = newKey.replacingOccurrences(of: "input_layernorm.", with: "norm1.")
            newKey = newKey.replacingOccurrences(of: "post_attention_layernorm.", with: "norm2.")
            newKey = newKey.replacingOccurrences(of: "param_proj.proj.", with: "distribution_head.proj.")

            result[newKey] = value
        }
        return result
    }

    public func newCaches() -> [TimeSeriesKVCache?] {
        (0 ..< config.numLayers).map { _ in TimeSeriesKVCache() }
    }
}
