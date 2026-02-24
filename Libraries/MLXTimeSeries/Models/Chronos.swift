import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - Configuration

/// Configuration for the Chronos (T5-based) time series model.
public struct ChronosConfiguration: Codable, Sendable {
    public var dModel: Int
    public var dFf: Int
    public var dKv: Int
    public var numHeads: Int
    public var numLayers: Int
    public var numDecoderLayers: Int
    public var vocabSize: Int
    public var relativeAttentionNumBuckets: Int
    public var relativeAttentionMaxDistance: Int
    public var isGatedAct: Bool
    public var denseActFn: String
    public var chronosConfig: ChronosTokenizerConfig

    public var headDim: Int { dKv }

    enum CodingKeys: String, CodingKey {
        case dModel = "d_model"
        case dFf = "d_ff"
        case dKv = "d_kv"
        case numHeads = "num_heads"
        case numLayers = "num_layers"
        case numDecoderLayers = "num_decoder_layers"
        case vocabSize = "vocab_size"
        case relativeAttentionNumBuckets = "relative_attention_num_buckets"
        case relativeAttentionMaxDistance = "relative_attention_max_distance"
        case isGatedAct = "is_gated_act"
        case denseActFn = "dense_act_fn"
        case chronosConfig = "chronos_config"
    }

    public init(
        dModel: Int = 512,
        dFf: Int = 1024,
        dKv: Int = 64,
        numHeads: Int = 6,
        numLayers: Int = 8,
        numDecoderLayers: Int = 8,
        vocabSize: Int = 4096,
        relativeAttentionNumBuckets: Int = 32,
        relativeAttentionMaxDistance: Int = 128,
        isGatedAct: Bool = false,
        denseActFn: String = "relu",
        chronosConfig: ChronosTokenizerConfig = ChronosTokenizerConfig()
    ) {
        self.dModel = dModel
        self.dFf = dFf
        self.dKv = dKv
        self.numHeads = numHeads
        self.numLayers = numLayers
        self.numDecoderLayers = numDecoderLayers
        self.vocabSize = vocabSize
        self.relativeAttentionNumBuckets = relativeAttentionNumBuckets
        self.relativeAttentionMaxDistance = relativeAttentionMaxDistance
        self.isGatedAct = isGatedAct
        self.denseActFn = denseActFn
        self.chronosConfig = chronosConfig
    }
}

/// Chronos tokenizer configuration.
public struct ChronosTokenizerConfig: Codable, Sendable {
    public var nTokens: Int
    public var nSpecialTokens: Int
    public var contextLength: Int
    public var predictionLength: Int
    public var numSamples: Int
    public var temperature: Float
    public var topK: Int
    public var topP: Float

    enum CodingKeys: String, CodingKey {
        case nTokens = "n_tokens"
        case nSpecialTokens = "n_special_tokens"
        case contextLength = "context_length"
        case predictionLength = "prediction_length"
        case numSamples = "num_samples"
        case temperature
        case topK = "top_k"
        case topP = "top_p"
    }

    public init(
        nTokens: Int = 4096,
        nSpecialTokens: Int = 2,
        contextLength: Int = 512,
        predictionLength: Int = 64,
        numSamples: Int = 20,
        temperature: Float = 1.0,
        topK: Int = 50,
        topP: Float = 1.0
    ) {
        self.nTokens = nTokens
        self.nSpecialTokens = nSpecialTokens
        self.contextLength = contextLength
        self.predictionLength = predictionLength
        self.numSamples = numSamples
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
    }
}

// MARK: - MeanScaleUniformBins tokenizer

/// Pure-math tokenizer that maps continuous values to bin indices.
///
/// Implements the Chronos `MeanScaleUniformBins` tokenization:
/// 1. Compute mean absolute scale of the input
/// 2. Normalize by the scale
/// 3. Map to uniformly-spaced bins via linear interpolation
public struct ChronosTokenizer: Sendable {
    let nTokens: Int
    let nSpecialTokens: Int
    let centers: MLXArray
    let edges: MLXArray

    public init(config: ChronosTokenizerConfig) {
        self.nTokens = config.nTokens
        self.nSpecialTokens = config.nSpecialTokens

        let nBins = nTokens - nSpecialTokens
        // Bin centers uniformly spaced in [-1, 1] (before scaling)
        // Actual range is roughly [-15, 15] for typical Chronos models
        let low: Float = -15.0
        let high: Float = 15.0
        let step = (high - low) / Float(nBins)
        var centerValues = [Float]()
        for i in 0 ..< nBins {
            centerValues.append(low + step * (Float(i) + 0.5))
        }
        self.centers = MLXArray(centerValues)

        // Bin edges: midpoints between centers, plus -inf and +inf
        var edgeValues = [Float]()
        edgeValues.append(-Float.infinity)
        for i in 0 ..< (nBins - 1) {
            edgeValues.append((centerValues[i] + centerValues[i + 1]) / 2)
        }
        edgeValues.append(Float.infinity)
        self.edges = MLXArray(edgeValues)
    }

    /// Tokenize a time series by normalizing and binning.
    ///
    /// - Parameters:
    ///   - values: Raw time series values, shape `[T]` or `[B, T]`.
    ///   - mask: Observation mask (1=observed, 0=padding).
    /// - Returns: Tuple of (token IDs as Int32, scale used for normalization).
    public func encode(_ values: MLXArray, mask: MLXArray) -> (tokens: MLXArray, scale: MLXArray) {
        // Compute mean absolute value for scaling
        let absMasked = MLX.abs(values) * mask
        let nObs = MLX.maximum(mask.sum(axis: -1, keepDims: true), MLXArray(Float(1)))
        let scale = absMasked.sum(axis: -1, keepDims: true) / nObs
        let safeScale = MLX.maximum(scale, MLXArray(Float(1e-9)))

        // Normalize
        let normalized = values / safeScale

        // Quantize to bin indices using searchsorted-like logic
        // For each value, find which bin it falls into
        let nBins = nTokens - nSpecialTokens
        let low: Float = -15.0
        let high: Float = 15.0
        let clipped = MLX.clip(normalized, min: MLXArray(low), max: MLXArray(high))

        // Linear map to bin index: (value - low) / (high - low) * nBins
        let binFloat = (clipped - low) / (high - low) * Float(nBins)
        let binIdx = MLX.clip(
            binFloat.asType(.int32),
            min: MLXArray(Int32(0)),
            max: MLXArray(Int32(nBins - 1))
        )

        // Offset by special tokens (PAD=0, EOS=1, then bin tokens start at 2)
        let tokens = binIdx + Int32(nSpecialTokens)

        return (tokens: tokens, scale: safeScale)
    }

    /// Decode token logits back to continuous values.
    ///
    /// - Parameters:
    ///   - logits: Token logits from the model, shape `[..., vocabSize]`.
    ///   - scale: The scale used during encoding.
    ///   - temperature: Sampling temperature.
    /// - Returns: Decoded continuous values.
    public func decodeMean(logits: MLXArray, scale: MLXArray) -> MLXArray {
        // Get probabilities for bin tokens only (skip special tokens)
        let binLogits = logits[.ellipsis, nSpecialTokens...]
        let probs = MLX.softmax(binLogits, axis: -1)

        // Expected value: weighted sum of bin centers
        let mean = (probs * centers).sum(axis: -1)

        // Rescale back to original space
        return mean * scale.squeezed(axis: -1)
    }
}

// MARK: - T5 Components

/// T5-style relative attention bias.
class T5RelativeAttentionBias: Module {
    @ModuleInfo(key: "relative_attention_bias") var bias: Embedding

    let numBuckets: Int
    let maxDistance: Int
    let numHeads: Int
    let isBidirectional: Bool

    init(numBuckets: Int, maxDistance: Int, numHeads: Int, isBidirectional: Bool) {
        self.numBuckets = numBuckets
        self.maxDistance = maxDistance
        self.numHeads = numHeads
        self.isBidirectional = isBidirectional
        self._bias.wrappedValue = Embedding(embeddingCount: numBuckets, dimensions: numHeads)
    }

    func callAsFunction(queryLength: Int, keyLength: Int) -> MLXArray {
        // Compute relative position bucket indices
        let contextPos = MLXArray(Array(0 ..< queryLength))
        let memoryPos = MLXArray(Array(0 ..< keyLength))
        let relativePosition = memoryPos.reshaped(1, -1) - contextPos.reshaped(-1, 1)

        let buckets = computeBuckets(relativePosition)
        let values = bias(buckets)  // [qLen, kLen, numHeads]
        // Transpose to [numHeads, qLen, kLen] -> [1, numHeads, qLen, kLen]
        return values.transposed(2, 0, 1).expandedDimensions(axis: 0)
    }

    private func computeBuckets(_ relativePosition: MLXArray) -> MLXArray {
        var rp = relativePosition
        var numBuckets = Int32(self.numBuckets)
        var ret = MLXArray.zeros(like: rp).asType(.int32)

        if isBidirectional {
            numBuckets /= 2
            let positiveOffset = MLX.where(rp .> 0, MLXArray(numBuckets), MLXArray(Int32(0)))
            ret = ret + positiveOffset.asType(.int32)
            rp = MLX.abs(rp)
        } else {
            rp = MLX.negative(MLX.minimum(rp, MLXArray(Int32(0))))
        }

        let maxExact = numBuckets / 2
        let isSmall = rp .< Int32(maxExact)

        let rpFloat = rp.asType(.float32)
        let valLarge = Float(maxExact) + (
            MLX.log(rpFloat / Float(maxExact))
            / log(Float(self.maxDistance) / Float(maxExact))
            * Float(numBuckets - maxExact)
        )
        let valLargeClamped = MLX.minimum(valLarge, MLXArray(Float(numBuckets - 1))).asType(.int32)

        ret = ret + MLX.where(isSmall, rp.asType(.int32), valLargeClamped)
        return ret
    }
}

/// T5-style multi-head attention.
class T5Attention: Module {
    @ModuleInfo(key: "q") var wQ: Linear
    @ModuleInfo(key: "k") var wK: Linear
    @ModuleInfo(key: "v") var wV: Linear
    @ModuleInfo(key: "o") var wO: Linear

    let numHeads: Int
    let headDim: Int
    let scale: Float

    init(dModel: Int, dKv: Int, numHeads: Int) {
        self.numHeads = numHeads
        self.headDim = dKv
        self.scale = pow(Float(dKv), -0.5)
        self._wQ.wrappedValue = Linear(dModel, numHeads * dKv, bias: false)
        self._wK.wrappedValue = Linear(dModel, numHeads * dKv, bias: false)
        self._wV.wrappedValue = Linear(dModel, numHeads * dKv, bias: false)
        self._wO.wrappedValue = Linear(numHeads * dKv, dModel, bias: false)
    }

    func callAsFunction(
        _ x: MLXArray,
        kvInput: MLXArray? = nil,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
        positionBias: MLXArray? = nil,
        cache: TimeSeriesKVCache? = nil
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        let source = kvInput ?? x

        let q = wQ(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        var k = wK(source).reshaped(B, source.dim(1), numHeads, headDim).transposed(0, 2, 1, 3)
        var v = wV(source).reshaped(B, source.dim(1), numHeads, headDim).transposed(0, 2, 1, 3)

        if let cache {
            let (cachedK, cachedV) = cache.update(keys: k, values: v)
            k = cachedK
            v = cachedV
        }

        // Combine position bias with mask if present
        var effectiveMask = mask
        if let positionBias {
            effectiveMask = .array(positionBias)
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: effectiveMask
        )

        let out = output.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return wO(out)
    }
}

/// T5-style feed-forward layer.
class T5FeedForward: Module {
    @ModuleInfo(key: "DenseReluDense") var dense: T5DenseReluDense

    init(dModel: Int, dFf: Int, isGated: Bool, actFn: String) {
        self._dense.wrappedValue = T5DenseReluDense(
            dModel: dModel, dFf: dFf, isGated: isGated, actFn: actFn)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        dense(x)
    }
}

/// T5 DenseReluDense (or gated variant).
class T5DenseReluDense: Module {
    @ModuleInfo(key: "wi") var wi: Linear
    @ModuleInfo(key: "wi_0") var wi0: Linear?
    @ModuleInfo(key: "wi_1") var wi1: Linear?
    @ModuleInfo(key: "wo") var wo: Linear

    let isGated: Bool
    let actFn: String

    init(dModel: Int, dFf: Int, isGated: Bool, actFn: String) {
        self.isGated = isGated
        self.actFn = actFn
        if isGated {
            self._wi0.wrappedValue = Linear(dModel, dFf, bias: false)
            self._wi1.wrappedValue = Linear(dModel, dFf, bias: false)
            self._wi.wrappedValue = Linear(dModel, dFf, bias: false)  // placeholder
        } else {
            self._wi.wrappedValue = Linear(dModel, dFf, bias: false)
            self._wi0.wrappedValue = nil
            self._wi1.wrappedValue = nil
        }
        self._wo.wrappedValue = Linear(dFf, dModel, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if isGated, let wi0, let wi1 {
            let gate = applyAct(wi0(x))
            let value = wi1(x)
            return wo(gate * value)
        } else {
            return wo(applyAct(wi(x)))
        }
    }

    private func applyAct(_ x: MLXArray) -> MLXArray {
        switch actFn {
        case "gelu", "gelu_new":
            return gelu(x)
        case "silu", "swish":
            return silu(x)
        default:
            return relu(x)
        }
    }
}

/// T5 layer norm (no bias, no mean subtraction).
class T5LayerNorm: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray

    let eps: Float

    init(dimensions: Int, eps: Float = 1e-6) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let variance = (x * x).mean(axis: -1, keepDims: true)
        let normalized = x * MLX.rsqrt(variance + eps)
        return weight * normalized
    }
}

/// T5 encoder block.
class T5EncoderBlock: Module {
    @ModuleInfo(key: "layer") var subLayers: [T5EncoderSubLayer]

    init(_ config: ChronosConfiguration) {
        self._subLayers.wrappedValue = [
            T5EncoderSubLayer(config, hasSelfAttn: true, hasCrossAttn: false, hasFF: false),
            T5EncoderSubLayer(config, hasSelfAttn: false, hasCrossAttn: false, hasFF: true),
        ]
    }

    func callAsFunction(_ x: MLXArray, positionBias: MLXArray?) -> MLXArray {
        var h = x
        // Self attention
        h = subLayers[0](h, positionBias: positionBias)
        // Feed forward
        h = subLayers[1](h, positionBias: nil)
        return h
    }
}

/// T5 encoder sub-layer (self-attention or feed-forward).
class T5EncoderSubLayer: Module {
    @ModuleInfo(key: "SelfAttention") var selfAttention: T5Attention?
    @ModuleInfo(key: "layer_norm") var layerNorm: T5LayerNorm
    @ModuleInfo(key: "DenseReluDense") var ff: T5DenseReluDense?

    init(_ config: ChronosConfiguration, hasSelfAttn: Bool, hasCrossAttn: Bool, hasFF: Bool) {
        self._layerNorm.wrappedValue = T5LayerNorm(dimensions: config.dModel)
        if hasSelfAttn {
            self._selfAttention.wrappedValue = T5Attention(
                dModel: config.dModel, dKv: config.dKv, numHeads: config.numHeads)
        } else {
            self._selfAttention.wrappedValue = nil
        }
        if hasFF {
            self._ff.wrappedValue = T5DenseReluDense(
                dModel: config.dModel, dFf: config.dFf,
                isGated: config.isGatedAct, actFn: config.denseActFn)
        } else {
            self._ff.wrappedValue = nil
        }
    }

    func callAsFunction(_ x: MLXArray, positionBias: MLXArray?) -> MLXArray {
        let normed = layerNorm(x)
        if let selfAttention {
            return x + selfAttention(normed, positionBias: positionBias)
        } else if let ff {
            return x + ff(normed)
        }
        return x
    }
}

/// T5 decoder block.
class T5DecoderBlock: Module {
    @ModuleInfo(key: "layer") var subLayers: [T5DecoderSubLayer]

    init(_ config: ChronosConfiguration) {
        self._subLayers.wrappedValue = [
            // Self attention
            T5DecoderSubLayer(config, kind: .selfAttention),
            // Cross attention
            T5DecoderSubLayer(config, kind: .crossAttention),
            // Feed forward
            T5DecoderSubLayer(config, kind: .feedForward),
        ]
    }

    func callAsFunction(
        _ x: MLXArray,
        encoderOutput: MLXArray,
        selfPositionBias: MLXArray?,
        cache: TimeSeriesKVCache?
    ) -> MLXArray {
        var h = x
        h = subLayers[0](h, kvInput: nil, positionBias: selfPositionBias, cache: cache)
        h = subLayers[1](h, kvInput: encoderOutput, positionBias: nil, cache: nil)
        h = subLayers[2](h, kvInput: nil, positionBias: nil, cache: nil)
        return h
    }
}

/// Kind of T5 decoder sub-layer.
enum T5DecoderSubLayerKind {
    case selfAttention
    case crossAttention
    case feedForward
}

/// T5 decoder sub-layer.
class T5DecoderSubLayer: Module {
    @ModuleInfo(key: "SelfAttention") var selfAttention: T5Attention?
    @ModuleInfo(key: "EncDecAttention") var crossAttention: T5Attention?
    @ModuleInfo(key: "layer_norm") var layerNorm: T5LayerNorm
    @ModuleInfo(key: "DenseReluDense") var ff: T5DenseReluDense?

    let kind: T5DecoderSubLayerKind

    init(_ config: ChronosConfiguration, kind: T5DecoderSubLayerKind) {
        self.kind = kind
        self._layerNorm.wrappedValue = T5LayerNorm(dimensions: config.dModel)

        switch kind {
        case .selfAttention:
            self._selfAttention.wrappedValue = T5Attention(
                dModel: config.dModel, dKv: config.dKv, numHeads: config.numHeads)
            self._crossAttention.wrappedValue = nil
            self._ff.wrappedValue = nil
        case .crossAttention:
            self._selfAttention.wrappedValue = nil
            self._crossAttention.wrappedValue = T5Attention(
                dModel: config.dModel, dKv: config.dKv, numHeads: config.numHeads)
            self._ff.wrappedValue = nil
        case .feedForward:
            self._selfAttention.wrappedValue = nil
            self._crossAttention.wrappedValue = nil
            self._ff.wrappedValue = T5DenseReluDense(
                dModel: config.dModel, dFf: config.dFf,
                isGated: config.isGatedAct, actFn: config.denseActFn)
        }
    }

    func callAsFunction(
        _ x: MLXArray,
        kvInput: MLXArray?,
        positionBias: MLXArray?,
        cache: TimeSeriesKVCache?
    ) -> MLXArray {
        let normed = layerNorm(x)
        switch kind {
        case .selfAttention:
            return x + selfAttention!(normed, positionBias: positionBias, cache: cache)
        case .crossAttention:
            return x + crossAttention!(normed, kvInput: kvInput, positionBias: positionBias)
        case .feedForward:
            return x + ff!(normed)
        }
    }
}

// MARK: - ChronosModel

/// Chronos time series forecasting model (T5 encoder-decoder).
///
/// Tokenizes continuous values into bins, runs T5 encoder-decoder,
/// then decodes token logits back to continuous values.
public class ChronosModel: Module, TimeSeriesModel {

    @ModuleInfo(key: "shared") var shared: Embedding
    @ModuleInfo(key: "encoder") var encoder: ChronosEncoder
    @ModuleInfo(key: "decoder") var decoder: ChronosDecoder
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    let config: ChronosConfiguration
    let tokenizer: ChronosTokenizer

    public init(_ config: ChronosConfiguration) {
        self.config = config
        self.tokenizer = ChronosTokenizer(config: config.chronosConfig)
        self._shared.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.dModel)
        self._encoder.wrappedValue = ChronosEncoder(config)
        self._decoder.wrappedValue = ChronosDecoder(config)
        self._lmHead.wrappedValue = Linear(config.dModel, config.vocabSize, bias: false)
    }

    public func forecast(
        input: TimeSeriesInput,
        predictionLength: Int,
        caches: [TimeSeriesKVCache?]
    ) -> TimeSeriesPrediction {
        let series = input.series.squeezed(axis: 1)  // [B, T] (univariate)
        let mask = input.paddingMask.squeezed(axis: 1)

        // Tokenize input
        let (tokens, scale) = tokenizer.encode(series, mask: mask)

        // Encode
        let tokenEmbeddings = shared(tokens)
        let encoderOutput = encoder(tokenEmbeddings)

        // Decode autoregressively
        let predLength = predictionLength
        // Start with EOS token (id=1) as decoder input
        var decoderInput = MLXArray([Int32(1)]).reshaped(1, 1)
        decoderInput = MLX.broadcast(decoderInput, to: [series.dim(0), 1])

        var allLogits = [MLXArray]()

        for _ in 0 ..< predLength {
            let decoderEmbed = shared(decoderInput)
            let decoderOutput = decoder(decoderEmbed, encoderOutput: encoderOutput, caches: caches)
            let logits = lmHead(decoderOutput)  // [B, 1, vocabSize]
            allLogits.append(logits[.ellipsis, (logits.dim(1) - 1), 0...])

            // Greedy: take argmax as next input
            let nextToken = MLX.argMax(logits[.ellipsis, (logits.dim(1) - 1), 0...], axis: -1)
                .asType(.int32)
            decoderInput = nextToken.expandedDimensions(axis: -1)
        }

        // Decode logits to continuous values
        let stackedLogits = MLX.stacked(allLogits, axis: 1)  // [B, predLength, vocabSize]
        let mean = tokenizer.decodeMean(logits: stackedLogits, scale: scale)
        // Add back variable dimension: [B, predLength] -> [B, 1, predLength]
        let meanOut = mean.expandedDimensions(axis: 1)

        return TimeSeriesPrediction(
            mean: meanOut,
            predictionLength: predictionLength
        )
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Chronos uses standard T5 naming — minimal remapping needed
        var result = [String: MLXArray]()
        for (key, value) in weights {
            // Skip tied embedding weights
            if key == "lm_head.weight" && weights["shared.weight"] != nil {
                continue
            }
            result[key] = value
        }
        return result
    }

    public func newCaches() -> [TimeSeriesKVCache?] {
        (0 ..< config.numDecoderLayers).map { _ in TimeSeriesKVCache() }
    }
}

/// Chronos T5 encoder.
class ChronosEncoder: Module {
    @ModuleInfo(key: "block") var blocks: [T5EncoderBlock]
    @ModuleInfo(key: "final_layer_norm") var finalNorm: T5LayerNorm

    let relBias: T5RelativeAttentionBias

    init(_ config: ChronosConfiguration) {
        self._blocks.wrappedValue = (0 ..< config.numLayers).map { _ in T5EncoderBlock(config) }
        self._finalNorm.wrappedValue = T5LayerNorm(dimensions: config.dModel)
        self.relBias = T5RelativeAttentionBias(
            numBuckets: config.relativeAttentionNumBuckets,
            maxDistance: config.relativeAttentionMaxDistance,
            numHeads: config.numHeads,
            isBidirectional: true
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let L = x.dim(1)
        let positionBias = relBias(queryLength: L, keyLength: L)

        var h = x
        for block in blocks {
            h = block(h, positionBias: positionBias)
        }
        return finalNorm(h)
    }
}

/// Chronos T5 decoder.
class ChronosDecoder: Module {
    @ModuleInfo(key: "block") var blocks: [T5DecoderBlock]
    @ModuleInfo(key: "final_layer_norm") var finalNorm: T5LayerNorm

    let relBias: T5RelativeAttentionBias

    init(_ config: ChronosConfiguration) {
        self._blocks.wrappedValue = (0 ..< config.numDecoderLayers).map { _ in
            T5DecoderBlock(config)
        }
        self._finalNorm.wrappedValue = T5LayerNorm(dimensions: config.dModel)
        self.relBias = T5RelativeAttentionBias(
            numBuckets: config.relativeAttentionNumBuckets,
            maxDistance: config.relativeAttentionMaxDistance,
            numHeads: config.numHeads,
            isBidirectional: false
        )
    }

    func callAsFunction(
        _ x: MLXArray,
        encoderOutput: MLXArray,
        caches: [TimeSeriesKVCache?]
    ) -> MLXArray {
        let L = x.dim(1)
        let selfPositionBias = relBias(queryLength: L, keyLength: L)

        var h = x
        for (i, block) in blocks.enumerated() {
            h = block(h, encoderOutput: encoderOutput, selfPositionBias: selfPositionBias,
                       cache: caches.indices.contains(i) ? caches[i] : nil)
        }
        return finalNorm(h)
    }
}
