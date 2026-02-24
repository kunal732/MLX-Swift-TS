import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - Configuration

/// Configuration for the TimesFM time series model.
public struct TimesFMConfiguration: Codable, Sendable {
    public var hiddenSize: Int
    public var numLayers: Int
    public var numHeads: Int
    public var intermediateSize: Int
    public var headDim: Int
    public var patchLength: Int
    public var quantileHorizonLength: Int
    public var numQuantiles: Int
    public var contextLength: Int
    public var predictionLength: Int
    public var usePositionalEncoding: Bool

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numLayers = "num_layers"
        case numHeads = "num_heads"
        case intermediateSize = "intermediate_size"
        case headDim = "head_dim"
        case patchLength = "patch_length"
        case quantileHorizonLength = "quantile_horizon_length"
        case numQuantiles = "num_quantiles"
        case contextLength = "context_length"
        case predictionLength = "prediction_length"
        case usePositionalEncoding = "use_positional_encoding"
    }

    public init(
        hiddenSize: Int = 1280,
        numLayers: Int = 20,
        numHeads: Int = 16,
        intermediateSize: Int = 1280,
        headDim: Int = 80,
        patchLength: Int = 32,
        quantileHorizonLength: Int = 1024,
        numQuantiles: Int = 9,
        contextLength: Int = 16384,
        predictionLength: Int = 128,
        usePositionalEncoding: Bool = false
    ) {
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.numHeads = numHeads
        self.intermediateSize = intermediateSize
        self.headDim = headDim
        self.patchLength = patchLength
        self.quantileHorizonLength = quantileHorizonLength
        self.numQuantiles = numQuantiles
        self.contextLength = contextLength
        self.predictionLength = predictionLength
        self.usePositionalEncoding = usePositionalEncoding
    }
}

// MARK: - TimesFM Components

/// Residual block: output_layer(silu(hidden_layer(x))) + residual_layer(x)
class TimesFMResidualBlock: Module {
    @ModuleInfo(key: "hidden_layer") var hiddenLayer: Linear
    @ModuleInfo(key: "output_layer") var outputLayer: Linear
    @ModuleInfo(key: "residual_layer") var residualLayer: Linear

    init(inputDim: Int, outputDim: Int, hiddenDim: Int? = nil, bias: Bool = true) {
        let hDim = hiddenDim ?? outputDim
        self._hiddenLayer.wrappedValue = Linear(inputDim, hDim, bias: bias)
        self._outputLayer.wrappedValue = Linear(hDim, outputDim, bias: bias)
        self._residualLayer.wrappedValue = Linear(inputDim, outputDim, bias: bias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        outputLayer(silu(hiddenLayer(x))) + residualLayer(x)
    }
}

/// Wrapper module for per_dim_scale nested parameter.
class TimesFMPerDimScale: Module {
    @ModuleInfo(key: "per_dim_scale") var perDimScale: MLXArray

    init(headDim: Int) {
        self._perDimScale.wrappedValue = MLXArray.ones([headDim])
    }
}

/// TimesFM attention with fused QKV, per-dimension scaling, and QK norms.
class TimesFMAttention: Module {
    @ModuleInfo(key: "qkv_proj") var qkvProj: Linear
    @ModuleInfo(key: "out") var wO: Linear
    @ModuleInfo(key: "per_dim_scale") var perDimScaleModule: TimesFMPerDimScale
    @ModuleInfo(key: "query_ln") var queryLN: RMSNorm
    @ModuleInfo(key: "key_ln") var keyLN: RMSNorm

    let numHeads: Int
    let headDim: Int

    init(_ config: TimesFMConfiguration) {
        self.numHeads = config.numHeads
        self.headDim = config.headDim
        let totalDim = numHeads * headDim
        self._qkvProj.wrappedValue = Linear(config.hiddenSize, 3 * totalDim, bias: false)
        self._wO.wrappedValue = Linear(totalDim, config.hiddenSize, bias: false)
        self._perDimScaleModule.wrappedValue = TimesFMPerDimScale(headDim: config.headDim)
        self._queryLN.wrappedValue = RMSNorm(dimensions: config.headDim)
        self._keyLN.wrappedValue = RMSNorm(dimensions: config.headDim)
    }

    func callAsFunction(
        _ x: MLXArray,
        cache: TimeSeriesKVCache? = nil
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        let totalDim = numHeads * headDim

        let qkv = qkvProj(x)
        var q = qkv[.ellipsis, 0..<totalDim]
            .reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        var k = qkv[.ellipsis, totalDim..<(2 * totalDim)]
            .reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = qkv[.ellipsis, (2 * totalDim)..<(3 * totalDim)]
            .reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)

        // QK norms
        q = queryLN(q)
        k = keyLN(k)

        // Per-dimension learned scaling
        let scale = softplus(perDimScaleModule.perDimScale).reshaped(1, 1, 1, headDim)
        q = q * scale / Float(headDim)

        var values = v
        if let cache {
            let (cachedK, cachedV) = cache.update(keys: k, values: v)
            k = cachedK
            values = cachedV
        }

        let mask: MLXFast.ScaledDotProductAttentionMaskMode = (L == 1) ? .none : .causal

        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: values, scale: 1.0, mask: mask
        )

        let out = output.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return wO(out)
    }
}

/// TimesFM transformer block with sandwich norms (pre + post per sub-layer).
class TimesFMTransformerBlock: Module {
    @ModuleInfo(key: "pre_attn_ln") var preAttnLN: RMSNorm
    @ModuleInfo(key: "post_attn_ln") var postAttnLN: RMSNorm
    @ModuleInfo(key: "pre_ff_ln") var preFFLN: RMSNorm
    @ModuleInfo(key: "post_ff_ln") var postFFLN: RMSNorm
    @ModuleInfo(key: "attn") var attn: TimesFMAttention
    @ModuleInfo(key: "ff0") var ff0: Linear
    @ModuleInfo(key: "ff1") var ff1: Linear

    init(_ config: TimesFMConfiguration) {
        self._preAttnLN.wrappedValue = RMSNorm(dimensions: config.hiddenSize)
        self._postAttnLN.wrappedValue = RMSNorm(dimensions: config.hiddenSize)
        self._preFFLN.wrappedValue = RMSNorm(dimensions: config.hiddenSize)
        self._postFFLN.wrappedValue = RMSNorm(dimensions: config.hiddenSize)
        self._attn.wrappedValue = TimesFMAttention(config)
        self._ff0.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: false)
        self._ff1.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray, cache: TimeSeriesKVCache?) -> MLXArray {
        let h = x + postAttnLN(attn(preAttnLN(x), cache: cache))
        return h + postFFLN(ff1(relu(ff0(preFFLN(h)))))
    }
}

// MARK: - TimesFMModel

/// TimesFM time series forecasting model (v2.0 / v2.5).
///
/// Decoder-only architecture with residual patch embedding,
/// per-dimension attention scaling, QK norms, and quantile output.
/// Produces all future time steps in one forward pass via the quantile head.
public class TimesFMModel: Module, TimeSeriesModel {

    @ModuleInfo(key: "tokenizer") var tokenizer: TimesFMResidualBlock
    @ModuleInfo(key: "stacked_xf") var layers: [TimesFMTransformerBlock]
    @ModuleInfo(key: "output_projection_point") var outputPoint: TimesFMResidualBlock
    @ModuleInfo(key: "output_projection_quantiles") var outputQuantiles: TimesFMResidualBlock

    let config: TimesFMConfiguration

    public init(_ config: TimesFMConfiguration) {
        self.config = config

        self._tokenizer.wrappedValue = TimesFMResidualBlock(
            inputDim: config.patchLength, outputDim: config.hiddenSize)
        self._layers.wrappedValue = (0..<config.numLayers).map { _ in
            TimesFMTransformerBlock(config)
        }
        // Point forecast head: hidden → hidden (no bias in weights)
        self._outputPoint.wrappedValue = TimesFMResidualBlock(
            inputDim: config.hiddenSize, outputDim: config.hiddenSize, bias: false)
        // Quantile head: hidden → quantileHorizonLength * (numQuantiles + 1) (no bias)
        // hidden_layer stays at hiddenSize, output/residual project to full quantile dim
        let quantileOutputDim = config.quantileHorizonLength * (config.numQuantiles + 1)
        self._outputQuantiles.wrappedValue = TimesFMResidualBlock(
            inputDim: config.hiddenSize, outputDim: quantileOutputDim,
            hiddenDim: config.hiddenSize, bias: false)
    }

    public func forecast(
        input: TimeSeriesInput,
        predictionLength: Int,
        caches: [TimeSeriesKVCache?]
    ) -> TimeSeriesPrediction {
        let patchSize = config.patchLength
        let series = input.series  // [B, V, T]
        let mask = input.paddingMask
        let scaler = CausalPatchScaler(patchSize: patchSize)

        let B = series.dim(0)
        let V = series.dim(1)
        let T = series.dim(2)

        // Pad to multiple of patchSize
        let padLen = (patchSize - T % patchSize) % patchSize
        let paddedSeries: MLXArray
        let paddedMask: MLXArray
        if padLen > 0 {
            paddedSeries = MLX.concatenated([MLXArray.zeros([B, V, padLen]), series], axis: -1)
            paddedMask = MLX.concatenated([MLXArray.zeros([B, V, padLen]), mask], axis: -1)
        } else {
            paddedSeries = series
            paddedMask = mask
        }

        // Normalize
        let (normalized, _, _, runLoc, runScale) = scaler(paddedSeries, mask: paddedMask)

        // Reshape to patches: [B, V, nPatches, patchSize]
        let paddedT = normalized.dim(2)
        let nPatches = paddedT / patchSize
        let patches = normalized.reshaped(B, V, nPatches, patchSize)

        // Flatten B*V for batch processing: [B*V, nPatches, patchSize]
        let flatPatches = patches.reshaped(B * V, nPatches, patchSize)

        // Patch embedding via residual block
        var hidden = tokenizer(flatPatches)  // [B*V, nPatches, hiddenSize]

        // Transformer layers
        for (i, layer) in layers.enumerated() {
            let cache = caches.indices.contains(i) ? caches[i] : nil
            hidden = layer(hidden, cache: cache)
        }

        // Take last hidden state
        let lastHidden = hidden[0..., (nPatches - 1), 0...]  // [B*V, hiddenSize]

        // Point forecast refinement
        let pointHidden = outputPoint(lastHidden.expandedDimensions(axis: 1))
            .squeezed(axis: 1)  // [B*V, hiddenSize]

        // Quantile forecast: produces all future steps at once
        let quantileOut = outputQuantiles(pointHidden.expandedDimensions(axis: 1))
            .squeezed(axis: 1)  // [B*V, qHL * (nQ+1)]

        let qHL = config.quantileHorizonLength
        let nQ = config.numQuantiles

        // Reshape: [B*V, nQ+1, qHL] — channel 0 is point forecast, 1-9 are quantiles
        let quantileReshaped = quantileOut.reshaped(B * V, nQ + 1, qHL)
        let meanPred = quantileReshaped[0..., 0, 0...]  // [B*V, qHL] — point forecast

        // Trim to requested prediction length
        let actualPredLen = min(predictionLength, qHL)
        let trimmed = meanPred[0..., 0..<actualPredLen]  // [B*V, actualPredLen]

        // Denormalize using running stats at the last timestep
        let lastT = runLoc.dim(2) - 1
        let denomLoc = runLoc[0..., 0..., lastT].reshaped(B * V, 1)
        let denomScale = runScale[0..., 0..., lastT].reshaped(B * V, 1)
        let denormalized = trimmed * denomScale + denomLoc

        // Reshape back: [B, V, predictionLength]
        let meanOut = denormalized.reshaped(B, V, actualPredLen)

        return TimeSeriesPrediction(
            mean: meanOut,
            predictionLength: actualPredLen
        )
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = [String: MLXArray]()
        for (key, value) in weights {
            // RMSNorm uses .weight in MLXNN but TimesFM saves as .scale
            let newKey = key.replacingOccurrences(of: ".scale", with: ".weight")
            result[newKey] = value
        }
        return result
    }

    public func newCaches() -> [TimeSeriesKVCache?] {
        (0..<config.numLayers).map { _ in TimeSeriesKVCache() }
    }
}
