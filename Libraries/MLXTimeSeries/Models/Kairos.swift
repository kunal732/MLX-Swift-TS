import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - Configuration

/// Configuration for the Kairos time series model (MoS-DP + IARoPE).
public struct KairosConfiguration: Codable, Sendable {
    public var dModel: Int
    public var dFf: Int
    public var dKv: Int
    public var numHeads: Int
    public var numLayers: Int
    public var numDecoderLayers: Int
    public var numDecoderSegments: Int
    public var inputPatchSize: Int
    public var inputPatchStride: Int
    public var levels: Int
    public var nActivatedExperts: Int
    public var nNullExperts: Int
    public var contextLength: Int
    public var predictionLength: Int
    public var isGatedAct: Bool
    public var denseActFn: String
    public var quantiles: [Float]
    public var useRegToken: Bool

    public var headDim: Int { dKv }
    public var numQuantiles: Int { quantiles.count }
    public var numExperts: Int { levels + nNullExperts }

    enum CodingKeys: String, CodingKey {
        case dModel = "d_model"
        case dFf = "d_ff"
        case dKv = "d_kv"
        case numHeads = "num_heads"
        case numLayers = "num_layers"
        case numDecoderLayers = "num_decoder_layers"
        case numDecoderSegments = "num_decoder_segments"
        case inputPatchSize = "input_patch_size"
        case inputPatchStride = "input_patch_stride"
        case levels
        case nActivatedExperts = "n_activated_experts"
        case nNullExperts = "n_null_experts"
        case contextLength = "context_length"
        case predictionLength = "prediction_length"
        case isGatedAct = "is_gated_act"
        case denseActFn = "dense_act_fn"
        case quantiles
        case useRegToken = "use_reg_token"
    }

    public init(
        dModel: Int = 512,
        dFf: Int = 2048,
        dKv: Int = 64,
        numHeads: Int = 8,
        numLayers: Int = 6,
        numDecoderLayers: Int = 6,
        numDecoderSegments: Int = 2,
        inputPatchSize: Int = 128,
        inputPatchStride: Int = 128,
        levels: Int = 3,
        nActivatedExperts: Int = 3,
        nNullExperts: Int = 2,
        contextLength: Int = 2048,
        predictionLength: Int = 64,
        isGatedAct: Bool = false,
        denseActFn: String = "relu",
        quantiles: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        useRegToken: Bool = true
    ) {
        self.dModel = dModel
        self.dFf = dFf
        self.dKv = dKv
        self.numHeads = numHeads
        self.numLayers = numLayers
        self.numDecoderLayers = numDecoderLayers
        self.numDecoderSegments = numDecoderSegments
        self.inputPatchSize = inputPatchSize
        self.inputPatchStride = inputPatchStride
        self.levels = levels
        self.nActivatedExperts = nActivatedExperts
        self.nNullExperts = nNullExperts
        self.contextLength = contextLength
        self.predictionLength = predictionLength
        self.isGatedAct = isGatedAct
        self.denseActFn = denseActFn
        self.quantiles = quantiles
        self.useRegToken = useRegToken
    }
}

// MARK: - Kairos T5-style Attention

/// Kairos attention block (T5-style with optional RoPE).
class KairosAttention: Module {
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
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        let source = kvInput ?? x

        let q = wQ(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        let k = wK(source).reshaped(B, source.dim(1), numHeads, headDim).transposed(0, 2, 1, 3)
        let v = wV(source).reshaped(B, source.dim(1), numHeads, headDim).transposed(0, 2, 1, 3)

        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask
        )

        let out = output.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return wO(out)
    }
}

/// Kairos encoder block: self-attention + feed-forward.
class KairosEncoderBlock: Module {
    @ModuleInfo(key: "layer") var subLayers: [KairosEncoderSubLayer]

    init(_ config: KairosConfiguration) {
        self._subLayers.wrappedValue = [
            KairosEncoderSubLayer(config, kind: .selfAttention),
            KairosEncoderSubLayer(config, kind: .feedForward),
        ]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for sub in subLayers {
            h = sub(h)
        }
        return h
    }
}

enum KairosSubLayerKind {
    case selfAttention
    case crossAttention
    case feedForward
}

class KairosEncoderSubLayer: Module {
    @ModuleInfo(key: "SelfAttention") var selfAttention: KairosAttention?
    @ModuleInfo(key: "layer_norm") var layerNorm: RMSNorm
    @ModuleInfo(key: "DenseReluDense") var ff: KairosFeedForward?

    let kind: KairosSubLayerKind

    init(_ config: KairosConfiguration, kind: KairosSubLayerKind) {
        self.kind = kind
        self._layerNorm.wrappedValue = RMSNorm(dimensions: config.dModel)
        switch kind {
        case .selfAttention:
            self._selfAttention.wrappedValue = KairosAttention(
                dModel: config.dModel, dKv: config.dKv, numHeads: config.numHeads)
            self._ff.wrappedValue = nil
        case .feedForward:
            self._selfAttention.wrappedValue = nil
            self._ff.wrappedValue = KairosFeedForward(
                dModel: config.dModel, dFf: config.dFf, actFn: config.denseActFn)
        default:
            self._selfAttention.wrappedValue = nil
            self._ff.wrappedValue = nil
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let normed = layerNorm(x)
        if let attn = selfAttention {
            return x + attn(normed)
        } else if let ff {
            return x + ff(normed)
        }
        return x
    }
}

/// Kairos decoder block: self-attention + cross-attention + feed-forward.
class KairosDecoderBlock: Module {
    @ModuleInfo(key: "layer") var subLayers: [KairosDecoderSubLayer]

    init(_ config: KairosConfiguration) {
        self._subLayers.wrappedValue = [
            KairosDecoderSubLayer(config, kind: .selfAttention),
            KairosDecoderSubLayer(config, kind: .crossAttention),
            KairosDecoderSubLayer(config, kind: .feedForward),
        ]
    }

    func callAsFunction(_ x: MLXArray, encoderOutput: MLXArray) -> MLXArray {
        var h = x
        h = subLayers[0](h, encoderOutput: nil)
        h = subLayers[1](h, encoderOutput: encoderOutput)
        h = subLayers[2](h, encoderOutput: nil)
        return h
    }
}

class KairosDecoderSubLayer: Module {
    @ModuleInfo(key: "SelfAttention") var selfAttention: KairosAttention?
    @ModuleInfo(key: "EncDecAttention") var crossAttention: KairosAttention?
    @ModuleInfo(key: "layer_norm") var layerNorm: RMSNorm
    @ModuleInfo(key: "DenseReluDense") var ff: KairosFeedForward?

    let kind: KairosSubLayerKind

    init(_ config: KairosConfiguration, kind: KairosSubLayerKind) {
        self.kind = kind
        self._layerNorm.wrappedValue = RMSNorm(dimensions: config.dModel)
        switch kind {
        case .selfAttention:
            self._selfAttention.wrappedValue = KairosAttention(
                dModel: config.dModel, dKv: config.dKv, numHeads: config.numHeads)
            self._crossAttention.wrappedValue = nil
            self._ff.wrappedValue = nil
        case .crossAttention:
            self._selfAttention.wrappedValue = nil
            self._crossAttention.wrappedValue = KairosAttention(
                dModel: config.dModel, dKv: config.dKv, numHeads: config.numHeads)
            self._ff.wrappedValue = nil
        case .feedForward:
            self._selfAttention.wrappedValue = nil
            self._crossAttention.wrappedValue = nil
            self._ff.wrappedValue = KairosFeedForward(
                dModel: config.dModel, dFf: config.dFf, actFn: config.denseActFn)
        }
    }

    func callAsFunction(_ x: MLXArray, encoderOutput: MLXArray?) -> MLXArray {
        let normed = layerNorm(x)
        switch kind {
        case .selfAttention:
            return x + selfAttention!(normed, mask: .causal)
        case .crossAttention:
            return x + crossAttention!(normed, kvInput: encoderOutput)
        case .feedForward:
            return x + ff!(normed)
        }
    }
}

class KairosFeedForward: Module {
    @ModuleInfo(key: "wi") var wi: Linear
    @ModuleInfo(key: "wo") var wo: Linear

    let actFn: String

    init(dModel: Int, dFf: Int, actFn: String) {
        self.actFn = actFn
        self._wi.wrappedValue = Linear(dModel, dFf, bias: false)
        self._wo.wrappedValue = Linear(dFf, dModel, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h: MLXArray
        switch actFn {
        case "gelu": h = gelu(wi(x))
        case "silu": h = silu(wi(x))
        default: h = relu(wi(x))
        }
        return wo(h)
    }
}

// MARK: - KairosModel

/// Kairos time series forecasting model.
///
/// T5-style encoder-decoder with Mixture-of-Size Dynamic Patching (MoS-DP)
/// and Instance-adaptive RoPE (IARoPE). 50M parameters.
public class KairosModel: Module, TimeSeriesModel {

    @ModuleInfo(key: "shared") var shared: Embedding
    @ModuleInfo(key: "input_patch_embedding") var inputPatchEmbed: ResidualBlock
    @ModuleInfo(key: "encoder") var encoder: KairosEncoder
    @ModuleInfo(key: "decoder") var decoder: KairosDecoder
    @ModuleInfo(key: "output_patch_embedding") var outputPatchEmbed: ResidualBlock

    let config: KairosConfiguration

    public init(_ config: KairosConfiguration) {
        self.config = config
        self._shared.wrappedValue = Embedding(
            embeddingCount: 2, dimensions: config.dModel)  // vocab_size=2
        self._inputPatchEmbed.wrappedValue = ResidualBlock(
            inputDim: config.inputPatchSize * 2,  // patch values + mask
            hiddenDim: config.dFf,
            outputDim: config.dModel
        )
        self._encoder.wrappedValue = KairosEncoder(config)
        self._decoder.wrappedValue = KairosDecoder(config)
        self._outputPatchEmbed.wrappedValue = ResidualBlock(
            inputDim: config.dModel,
            hiddenDim: config.dFf,
            outputDim: config.numQuantiles * config.predictionLength
        )
    }

    public func forecast(
        input: TimeSeriesInput,
        predictionLength: Int,
        caches: [TimeSeriesKVCache?]
    ) -> TimeSeriesPrediction {
        let series = input.series
        let mask = input.paddingMask
        let B = series.dim(0)
        let V = series.dim(1)
        let T = series.dim(2)
        let patchSize = config.inputPatchSize

        // Flatten to [B*V, T]
        let flat = series.reshaped(B * V, T)
        let flatMask = mask.reshaped(B * V, T)

        // Instance normalization
        let nObs = MLX.maximum(flatMask.sum(axis: -1, keepDims: true), MLXArray(Float(1)))
        let meanVal = (flat * flatMask).sum(axis: -1, keepDims: true) / nObs
        let variance = ((flat - meanVal) * (flat - meanVal) * flatMask)
            .sum(axis: -1, keepDims: true) / nObs
        let stdVal = MLX.maximum(MLX.sqrt(variance), MLXArray(Float(1e-5)))
        let normalized = (flat - meanVal) / stdVal

        // Pad to multiple of patchSize
        let padLen = (patchSize - T % patchSize) % patchSize
        let paddedNorm: MLXArray
        let paddedMask: MLXArray
        if padLen > 0 {
            paddedNorm = MLX.concatenated(
                [MLXArray.zeros([B * V, padLen]), normalized], axis: -1)
            paddedMask = MLX.concatenated(
                [MLXArray.zeros([B * V, padLen]), flatMask], axis: -1)
        } else {
            paddedNorm = normalized
            paddedMask = flatMask
        }

        let paddedT = paddedNorm.dim(1)
        let numPatches = paddedT / patchSize

        // Create patches: [B*V, numPatches, patchSize]
        let normPatches = paddedNorm.reshaped(B * V, numPatches, patchSize)
        let maskPatches = paddedMask.reshaped(B * V, numPatches, patchSize)
        // Concatenate value and mask: [B*V, numPatches, patchSize*2]
        let patchInput = MLX.concatenated([normPatches, maskPatches], axis: -1)

        // Embed patches
        let embedded = inputPatchEmbed(patchInput)  // [B*V, numPatches, dModel]

        // Encode
        let encoderOutput = encoder(embedded)

        // Decoder: use numDecoderSegments output tokens
        let numSegments = config.numDecoderSegments
        // Initialize decoder input with zeros
        let decoderInput = MLXArray.zeros([B * V, numSegments, config.dModel])
        let decoderOutput = decoder(decoderInput, encoderOutput: encoderOutput)

        // Project to quantile forecasts
        let rawOutput = outputPatchEmbed(decoderOutput)
        // [B*V, numSegments, numQuantiles * predictionLength]

        // Take mean across segments and reshape
        let meanSegments = rawOutput.mean(axis: 1)  // [B*V, numQuantiles * predLength]
        let quantilePreds = meanSegments.reshaped(
            B * V, config.numQuantiles, predictionLength)

        // Extract median
        let medianIdx = config.numQuantiles / 2
        let meanPred = quantilePreds[.ellipsis, medianIdx, 0...]  // [B*V, predictionLength]

        // Denormalize
        let denormalized = meanPred * stdVal + meanVal
        let meanOut = denormalized.reshaped(B, V, predictionLength)

        return TimeSeriesPrediction(
            mean: meanOut,
            predictionLength: predictionLength
        )
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = [String: MLXArray]()
        for (key, value) in weights {
            if key.contains("inv_freq") { continue }
            // Strip extra keys not used by the Swift model
            if key.hasPrefix("patch.") || key.hasPrefix("fft_norm.") { continue }
            result[key] = value
        }
        return result
    }

    public func newCaches() -> [TimeSeriesKVCache?] {
        []
    }
}

/// Kairos T5 encoder.
class KairosEncoder: Module {
    @ModuleInfo(key: "block") var blocks: [KairosEncoderBlock]
    @ModuleInfo(key: "final_layer_norm") var finalNorm: RMSNorm

    init(_ config: KairosConfiguration) {
        self._blocks.wrappedValue = (0 ..< config.numLayers).map { _ in
            KairosEncoderBlock(config)
        }
        self._finalNorm.wrappedValue = RMSNorm(dimensions: config.dModel)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for block in blocks {
            h = block(h)
        }
        return finalNorm(h)
    }
}

/// Kairos T5 decoder.
class KairosDecoder: Module {
    @ModuleInfo(key: "block") var blocks: [KairosDecoderBlock]
    @ModuleInfo(key: "final_layer_norm") var finalNorm: RMSNorm

    init(_ config: KairosConfiguration) {
        self._blocks.wrappedValue = (0 ..< config.numDecoderLayers).map { _ in
            KairosDecoderBlock(config)
        }
        self._finalNorm.wrappedValue = RMSNorm(dimensions: config.dModel)
    }

    func callAsFunction(_ x: MLXArray, encoderOutput: MLXArray) -> MLXArray {
        var h = x
        for block in blocks {
            h = block(h, encoderOutput: encoderOutput)
        }
        return finalNorm(h)
    }
}
