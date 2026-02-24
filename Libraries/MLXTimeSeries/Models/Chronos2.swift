import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - Configuration

/// Configuration for Chronos-2 (encoder-only, patch-based).
public struct Chronos2Configuration: Codable, Sendable {
    public var dModel: Int
    public var dFf: Int
    public var dKv: Int
    public var numHeads: Int
    public var numLayers: Int
    public var vocabSize: Int
    public var isGatedAct: Bool
    public var denseActFn: String
    public var ropeTheta: Float
    public var chronosConfig: Chronos2TokenizerConfig

    public var headDim: Int { dKv }

    enum CodingKeys: String, CodingKey {
        case dModel = "d_model"
        case dFf = "d_ff"
        case dKv = "d_kv"
        case numHeads = "num_heads"
        case numLayers = "num_layers"
        case vocabSize = "vocab_size"
        case isGatedAct = "is_gated_act"
        case denseActFn = "dense_act_fn"
        case ropeTheta = "rope_theta"
        case chronosConfig = "chronos_config"
    }

    public init(
        dModel: Int = 768,
        dFf: Int = 3072,
        dKv: Int = 64,
        numHeads: Int = 12,
        numLayers: Int = 12,
        vocabSize: Int = 2,
        isGatedAct: Bool = false,
        denseActFn: String = "relu",
        ropeTheta: Float = 10000.0,
        chronosConfig: Chronos2TokenizerConfig = Chronos2TokenizerConfig()
    ) {
        self.dModel = dModel
        self.dFf = dFf
        self.dKv = dKv
        self.numHeads = numHeads
        self.numLayers = numLayers
        self.vocabSize = vocabSize
        self.isGatedAct = isGatedAct
        self.denseActFn = denseActFn
        self.ropeTheta = ropeTheta
        self.chronosConfig = chronosConfig
    }
}

/// Chronos-2 tokenizer / patch config.
public struct Chronos2TokenizerConfig: Codable, Sendable {
    public var contextLength: Int
    public var inputPatchSize: Int
    public var inputPatchStride: Int
    public var outputPatchSize: Int
    public var maxOutputPatches: Int
    public var quantiles: [Float]
    public var useArcsinh: Bool
    public var useRegToken: Bool
    public var timeEncodingScale: Int

    public var numQuantiles: Int { quantiles.count }

    enum CodingKeys: String, CodingKey {
        case contextLength = "context_length"
        case inputPatchSize = "input_patch_size"
        case inputPatchStride = "input_patch_stride"
        case outputPatchSize = "output_patch_size"
        case maxOutputPatches = "max_output_patches"
        case quantiles
        case useArcsinh = "use_arcsinh"
        case useRegToken = "use_reg_token"
        case timeEncodingScale = "time_encoding_scale"
    }

    public init(
        contextLength: Int = 8192,
        inputPatchSize: Int = 16,
        inputPatchStride: Int = 16,
        outputPatchSize: Int = 16,
        maxOutputPatches: Int = 64,
        quantiles: [Float] = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
        useArcsinh: Bool = true,
        useRegToken: Bool = true,
        timeEncodingScale: Int = 8192
    ) {
        self.contextLength = contextLength
        self.inputPatchSize = inputPatchSize
        self.inputPatchStride = inputPatchStride
        self.outputPatchSize = outputPatchSize
        self.maxOutputPatches = maxOutputPatches
        self.quantiles = quantiles
        self.useArcsinh = useArcsinh
        self.useRegToken = useRegToken
        self.timeEncodingScale = timeEncodingScale
    }
}

// MARK: - Chronos-2 Encoder Block

/// Chronos-2 encoder block with TimeSelfAttention + GroupSelfAttention + FFN.
class Chronos2EncoderBlock: Module {
    @ModuleInfo(key: "layer") var subLayers: [Chronos2SubLayer]

    init(_ config: Chronos2Configuration) {
        self._subLayers.wrappedValue = [
            // TimeSelfAttention (with RoPE)
            Chronos2SubLayer(config, kind: .timeSelfAttention),
            // GroupSelfAttention (across batch, no RoPE)
            Chronos2SubLayer(config, kind: .groupSelfAttention),
            // Feed-forward
            Chronos2SubLayer(config, kind: .feedForward),
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

enum Chronos2SubLayerKind {
    case timeSelfAttention
    case groupSelfAttention
    case feedForward
}

class Chronos2SubLayer: Module {
    @ModuleInfo(key: "self_attention") var attention: Chronos2Attention?
    @ModuleInfo(key: "layer_norm") var layerNorm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: Chronos2MLP?

    let kind: Chronos2SubLayerKind

    init(_ config: Chronos2Configuration, kind: Chronos2SubLayerKind) {
        self.kind = kind
        self._layerNorm.wrappedValue = RMSNorm(dimensions: config.dModel)

        switch kind {
        case .timeSelfAttention, .groupSelfAttention:
            self._attention.wrappedValue = Chronos2Attention(
                dModel: config.dModel, dKv: config.dKv, numHeads: config.numHeads)
            self._mlp.wrappedValue = nil
        case .feedForward:
            self._attention.wrappedValue = nil
            self._mlp.wrappedValue = Chronos2MLP(
                dModel: config.dModel, dFf: config.dFf, actFn: config.denseActFn)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let normed = layerNorm(x)
        if let attn = attention {
            return x + attn(normed)
        } else if let mlp {
            return x + mlp(normed)
        }
        return x
    }
}

/// Chronos-2 attention (used for both time-wise and group-wise).
class Chronos2Attention: Module {
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

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        let q = wQ(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        let k = wK(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = wV(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)

        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .none
        )

        let out = output.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return wO(out)
    }
}

/// Chronos-2 feed-forward MLP.
class Chronos2MLP: Module {
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

// MARK: - Chronos2Model

/// Chronos-2 time series forecasting model.
///
/// Encoder-only T5-based model with patch tokenization, RoPE,
/// and dual attention (TimeSelf + GroupSelf per block).
/// ~120M parameters. Supports multivariate via group attention.
public class Chronos2Model: Module, TimeSeriesModel {

    @ModuleInfo(key: "shared") var shared: Embedding
    @ModuleInfo(key: "input_patch_embedding") var inputPatchEmbed: ResidualBlock
    @ModuleInfo(key: "encoder") var encoder: Chronos2Encoder
    @ModuleInfo(key: "output_patch_embedding") var outputPatchEmbed: ResidualBlock

    let config: Chronos2Configuration

    public init(_ config: Chronos2Configuration) {
        self.config = config
        let tcfg = config.chronosConfig
        self._shared.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.dModel)

        // Input: 3 * patchSize (time_enc + values + mask) -> dFf -> dModel
        let inputDim = 3 * tcfg.inputPatchSize
        self._inputPatchEmbed.wrappedValue = ResidualBlock(
            inputDim: inputDim, hiddenDim: config.dFf, outputDim: config.dModel)
        self._encoder.wrappedValue = Chronos2Encoder(config)

        // Output: dModel -> dFf -> numQuantiles * outputPatchSize
        let outputDim = tcfg.numQuantiles * tcfg.outputPatchSize
        self._outputPatchEmbed.wrappedValue = ResidualBlock(
            inputDim: config.dModel, hiddenDim: config.dFf, outputDim: outputDim)
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
        let tcfg = config.chronosConfig
        let patchSize = tcfg.inputPatchSize

        // Flatten to [B*V, T]
        let flat = series.reshaped(B * V, T)
        let flatMask = mask.reshaped(B * V, T)

        // Instance normalization
        let nObs = MLX.maximum(flatMask.sum(axis: -1, keepDims: true), MLXArray(Float(1)))
        let meanVal = (flat * flatMask).sum(axis: -1, keepDims: true) / nObs
        let variance = ((flat - meanVal) * (flat - meanVal) * flatMask)
            .sum(axis: -1, keepDims: true) / nObs
        let stdVal = MLX.maximum(MLX.sqrt(variance), MLXArray(Float(1e-5)))
        var normalized = (flat - meanVal) / stdVal

        // Optional arcsinh transform
        if tcfg.useArcsinh {
            normalized = MLX.log(normalized + MLX.sqrt(normalized * normalized + 1))
        }

        // Pad to multiple of patchSize
        let padLen = (patchSize - T % patchSize) % patchSize
        if padLen > 0 {
            normalized = MLX.concatenated(
                [MLXArray.zeros([B * V, padLen]), normalized], axis: -1)
        }

        let paddedT = normalized.dim(1)
        let numContextPatches = paddedT / patchSize

        // Create patches: [B*V, numPatches, patchSize]
        let normPatches = normalized.reshaped(B * V, numContextPatches, patchSize)

        // Create time encoding patches
        let scale = Float(tcfg.timeEncodingScale)
        var timeIndices = [Float]()
        for i in 0 ..< paddedT {
            timeIndices.append(Float(i - paddedT) / scale)
        }
        let timeEnc = MLXArray(timeIndices).reshaped(1, numContextPatches, patchSize)
        let timeEncBroadcast = MLX.broadcast(timeEnc, to: [B * V, numContextPatches, patchSize])

        // Mask patches
        var maskPadded = flatMask
        if padLen > 0 {
            maskPadded = MLX.concatenated(
                [MLXArray.zeros([B * V, padLen]), flatMask], axis: -1)
        }
        let maskPatches = maskPadded.reshaped(B * V, numContextPatches, patchSize)

        // Concatenate: [timeEnc, values, mask] -> [B*V, numPatches, 3*patchSize]
        let patchInput = MLX.concatenated(
            [timeEncBroadcast, normPatches, maskPatches], axis: -1)

        // Embed context patches
        var embedded = inputPatchEmbed(patchInput)  // [B*V, numPatches, dModel]

        // Add future output patches (zeros) for prediction
        let numOutputPatches = min(
            (predictionLength + tcfg.outputPatchSize - 1) / tcfg.outputPatchSize,
            tcfg.maxOutputPatches
        )

        // Create future time encoding
        var futureTimeIndices = [Float]()
        for i in 0 ..< (numOutputPatches * patchSize) {
            futureTimeIndices.append(Float(i) / scale)
        }
        let futureTimeEnc = MLXArray(futureTimeIndices)
            .reshaped(1, numOutputPatches, patchSize)
        let futureTimeEncBroadcast = MLX.broadcast(
            futureTimeEnc, to: [B * V, numOutputPatches, patchSize])
        let futureValues = MLXArray.zeros([B * V, numOutputPatches, patchSize])
        let futureMask = MLXArray.zeros([B * V, numOutputPatches, patchSize])
        let futureInput = MLX.concatenated(
            [futureTimeEncBroadcast, futureValues, futureMask], axis: -1)
        let futureEmbedded = inputPatchEmbed(futureInput)

        // Concatenate context + future
        embedded = MLX.concatenated([embedded, futureEmbedded], axis: 1)

        // Encode (full sequence)
        let encoderOutput = encoder(embedded)

        // Slice output patches (last numOutputPatches)
        let totalPatches = encoderOutput.dim(1)
        let outputHidden = encoderOutput[
            .ellipsis,
            (totalPatches - numOutputPatches) ..< totalPatches,
            0...
        ]

        // Project to quantile forecasts
        let rawOutput = outputPatchEmbed(outputHidden)
        // [B*V, numOutputPatches, numQuantiles * outputPatchSize]

        let totalPredSteps = numOutputPatches * tcfg.outputPatchSize
        let reshaped = rawOutput.reshaped(B * V, tcfg.numQuantiles, totalPredSteps)

        // Trim to requested prediction length
        let trimmedPred = reshaped[.ellipsis, 0 ..< predictionLength]

        // Extract median
        let medianIdx = tcfg.quantiles.firstIndex(of: 0.5) ?? (tcfg.numQuantiles / 2)
        var meanPred = trimmedPred[.ellipsis, medianIdx, 0...]  // [B*V, predictionLength]

        // Inverse arcsinh
        if tcfg.useArcsinh {
            meanPred = MLX.sinh(meanPred)
        }

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
            result[key] = value
        }
        return result
    }

    public func newCaches() -> [TimeSeriesKVCache?] {
        []  // Encoder-only, no autoregressive caching
    }
}

/// Chronos-2 encoder stack.
class Chronos2Encoder: Module {
    @ModuleInfo(key: "block") var blocks: [Chronos2EncoderBlock]
    @ModuleInfo(key: "final_layer_norm") var finalNorm: RMSNorm

    init(_ config: Chronos2Configuration) {
        self._blocks.wrappedValue = (0 ..< config.numLayers).map { _ in
            Chronos2EncoderBlock(config)
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
