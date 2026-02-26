import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - Configuration

/// Configuration for the TiRex time series model (sLSTM-based).
public struct TiRexConfiguration: Codable, Sendable {
    public var embeddingDim: Int
    public var numLayers: Int
    public var numHeads: Int
    public var ffnProjFactor: Float
    public var inputPatchSize: Int
    public var outputPatchSize: Int
    public var contextLength: Int
    public var quantiles: [Float]

    public var headDim: Int { embeddingDim / numHeads }
    public var numQuantiles: Int { quantiles.count }
    public var ffnDim: Int {
        // Round up to next multiple of 64
        let raw = Int(Float(embeddingDim) * ffnProjFactor)
        return ((raw + 63) / 64) * 64
    }

    enum CodingKeys: String, CodingKey {
        case embeddingDim = "embedding_dim"
        case numLayers = "num_layers"
        case numHeads = "num_heads"
        case ffnProjFactor = "ffn_proj_factor"
        case inputPatchSize = "input_patch_size"
        case outputPatchSize = "output_patch_size"
        case contextLength = "context_length"
        case quantiles
    }

    public init(
        embeddingDim: Int = 512,
        numLayers: Int = 8,
        numHeads: Int = 4,
        ffnProjFactor: Float = 2.6667,
        inputPatchSize: Int = 32,
        outputPatchSize: Int = 32,
        contextLength: Int = 2048,
        quantiles: [Float] = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]
    ) {
        self.embeddingDim = embeddingDim
        self.numLayers = numLayers
        self.numHeads = numHeads
        self.ffnProjFactor = ffnProjFactor
        self.inputPatchSize = inputPatchSize
        self.outputPatchSize = outputPatchSize
        self.contextLength = contextLength
        self.quantiles = quantiles
    }
}

// MARK: - sLSTM Cell

/// Scalar LSTM cell with exponential gating from the xLSTM paper.
///
/// Maintains 4 states: hidden (h), cell (c), normalizer (n), stabilizer (m).
/// Uses exponential gating for improved gradient flow.
class SLSTMCell: Module {
    @ModuleInfo(key: "_recurrent_kernel_") var recurrentKernel: MLXArray
    @ModuleInfo(key: "_bias_") var bias: MLXArray

    let numHeads: Int
    let headDim: Int

    init(numHeads: Int, headDim: Int) {
        self.numHeads = numHeads
        self.headDim = headDim
        // Recurrent kernel: [numHeads, headDim, 4 * headDim] for f, i, z, o gates
        self._recurrentKernel.wrappedValue = MLXArray.zeros([numHeads, headDim, 4 * headDim])
        // Bias: [numHeads * 4 * headDim]
        self._bias.wrappedValue = MLXArray.zeros([numHeads * 4 * headDim])
    }

    /// Run one step of the sLSTM cell.
    ///
    /// - Parameters:
    ///   - gateInputs: Pre-computed gate inputs from the projection layers,
    ///     shape `[B, numHeads, 4, headDim]` for (f, i, z, o).
    ///   - prevH: Previous hidden state `[B, numHeads, headDim]`.
    ///   - prevC: Previous cell state `[B, numHeads, headDim]`.
    ///   - prevN: Previous normalizer `[B, numHeads, headDim]`.
    ///   - prevM: Previous stabilizer `[B, numHeads, 1]`.
    /// - Returns: Tuple of (h, c, n, m) new states.
    func callAsFunction(
        gateInputs: MLXArray,
        prevH: MLXArray,
        prevC: MLXArray,
        prevN: MLXArray,
        prevM: MLXArray
    ) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        let B = prevH.dim(0)

        // Recurrent contribution: prevH @ recurrentKernel -> [B, numHeads, 4*headDim]
        let recurrent = MLX.matmul(
            prevH.expandedDimensions(axis: 2),
            recurrentKernel
        ).squeezed(axis: 2)  // [B, numHeads, 4*headDim]

        // Reshape to [B, numHeads, 4, headDim]
        let recurrentGates = recurrent.reshaped(B, numHeads, 4, headDim)

        // Add bias: [numHeads * 4 * headDim] -> [1, numHeads, 4, headDim]
        let biasReshaped = bias.reshaped(1, numHeads, 4, headDim)

        // Total gate pre-activations
        let gates = gateInputs + recurrentGates + biasReshaped

        let fRaw = gates[.ellipsis, 0, 0...]   // forget gate [B, numHeads, headDim]
        let iRaw = gates[.ellipsis, 1, 0...]   // input gate
        let zRaw = gates[.ellipsis, 2, 0...]   // cell input
        let oRaw = gates[.ellipsis, 3, 0...]   // output gate

        // Exponential gating with stabilization
        let logFPlusM = prevM + MLX.log(sigmoid(fRaw) + 1e-6)
        let mNew = MLX.maximum(iRaw, logFPlusM)

        // Stabilized gates
        let iGate = MLX.minimum(MLX.exp(iRaw - mNew), MLXArray(Float(1.0)))
        let fGate = MLX.minimum(MLX.exp(logFPlusM - mNew), MLXArray(Float(1.0)))
        let zGate = tanh(zRaw)
        let oGate = sigmoid(oRaw)

        // State updates
        let cNew = fGate * prevC + iGate * zGate
        let nNew = fGate * prevN + iGate
        let safeN = MLX.maximum(MLX.abs(nNew), MLXArray(Float(1.0)))
        let hNew = oGate * (cNew / safeN)

        return (hNew, cNew, nNew, mNew)
    }
}

/// Per-head linear projection matching TiRex's headwise gate structure.
///
/// Stores weight as `[numHeads, headDim, headDim]` (matching the checkpoint)
/// and applies each head's projection independently.
class PerHeadLinear: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray

    let numHeads: Int
    let headDim: Int

    init(numHeads: Int, headDim: Int) {
        self.numHeads = numHeads
        self.headDim = headDim
        self._weight.wrappedValue = MLXArray.zeros([numHeads, headDim, headDim])
    }

    /// - Parameter x: Input `[B, L, numHeads * headDim]`
    /// - Returns: Output `[B, L, numHeads * headDim]`
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        // [B, L, numHeads*headDim] -> [numHeads, B*L, headDim]
        let xHeads = x.reshaped(B * L, numHeads, headDim).transposed(1, 0, 2)
        // weight: [numHeads, headDim_out, headDim_in] -> apply as xHeads @ weight.T
        // [numHeads, B*L, headDim] @ [numHeads, headDim, headDim] -> [numHeads, B*L, headDim]
        let out = MLX.matmul(xHeads, weight.transposed(0, 2, 1))
        // [numHeads, B*L, headDim] -> [B*L, numHeads, headDim] -> [B, L, numHeads*headDim]
        return out.transposed(1, 0, 2).reshaped(B, L, numHeads * headDim)
    }
}

/// sLSTM layer with headwise gate projections.
class SLSTMLayer: Module {
    @ModuleInfo(key: "fgate") var fGate: PerHeadLinear
    @ModuleInfo(key: "igate") var iGate: PerHeadLinear
    @ModuleInfo(key: "zgate") var zGate: PerHeadLinear
    @ModuleInfo(key: "ogate") var oGate: PerHeadLinear
    @ModuleInfo(key: "slstm_cell") var cell: SLSTMCell
    @ModuleInfo(key: "group_norm") var groupNorm: RMSNorm

    let numHeads: Int
    let headDim: Int

    init(embeddingDim: Int, numHeads: Int) {
        self.numHeads = numHeads
        self.headDim = embeddingDim / numHeads
        self._fGate.wrappedValue = PerHeadLinear(numHeads: numHeads, headDim: headDim)
        self._iGate.wrappedValue = PerHeadLinear(numHeads: numHeads, headDim: headDim)
        self._zGate.wrappedValue = PerHeadLinear(numHeads: numHeads, headDim: headDim)
        self._oGate.wrappedValue = PerHeadLinear(numHeads: numHeads, headDim: headDim)
        self._cell.wrappedValue = SLSTMCell(numHeads: numHeads, headDim: headDim)
        self._groupNorm.wrappedValue = RMSNorm(dimensions: embeddingDim)
    }

    /// Forward pass through sLSTM layer (sequential over time).
    ///
    /// - Parameter x: Input of shape `[B, L, embeddingDim]`.
    /// - Returns: Output of shape `[B, L, embeddingDim]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        let embDim = numHeads * headDim

        // Compute per-head gate projections for all time steps at once
        let fAll = fGate(x).reshaped(B, L, numHeads, headDim)
        let iAll = iGate(x).reshaped(B, L, numHeads, headDim)
        let zAll = zGate(x).reshaped(B, L, numHeads, headDim)
        let oAll = oGate(x).reshaped(B, L, numHeads, headDim)

        // Initialize states
        var h = MLXArray.zeros([B, numHeads, headDim])
        var c = MLXArray.zeros([B, numHeads, headDim])
        var n = MLXArray.zeros([B, numHeads, headDim])
        var m = MLXArray.zeros([B, numHeads, 1])

        var outputs = [MLXArray]()

        // Sequential recurrence
        for t in 0 ..< L {
            let gateInputs = MLX.stacked(
                [fAll[.ellipsis, t, 0..., 0...],
                 iAll[.ellipsis, t, 0..., 0...],
                 zAll[.ellipsis, t, 0..., 0...],
                 oAll[.ellipsis, t, 0..., 0...]],
                axis: 2
            )  // [B, numHeads, 4, headDim]

            (h, c, n, m) = cell(
                gateInputs: gateInputs,
                prevH: h, prevC: c, prevN: n, prevM: m
            )
            outputs.append(h.reshaped(B, 1, embDim))
        }

        let result = MLX.concatenated(outputs, axis: 1)  // [B, L, embDim]
        return groupNorm(result)
    }
}

/// TiRex transformer block: RMSNorm + sLSTM + residual, then RMSNorm + SwiGLU FFN + residual.
class TiRexBlock: Module {
    @ModuleInfo(key: "norm_slstm") var normSLSTM: RMSNorm
    @ModuleInfo(key: "slstm_layer") var slstmLayer: SLSTMLayer
    @ModuleInfo(key: "norm_ffn") var normFFN: RMSNorm
    @ModuleInfo(key: "ffn") var ffn: TiRexFFN

    init(_ config: TiRexConfiguration) {
        self._normSLSTM.wrappedValue = RMSNorm(dimensions: config.embeddingDim)
        self._slstmLayer.wrappedValue = SLSTMLayer(
            embeddingDim: config.embeddingDim, numHeads: config.numHeads)
        self._normFFN.wrappedValue = RMSNorm(dimensions: config.embeddingDim)
        self._ffn.wrappedValue = TiRexFFN(
            embeddingDim: config.embeddingDim, ffnDim: config.ffnDim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = x + slstmLayer(normSLSTM(x))
        return h + ffn(normFFN(h))
    }
}

/// TiRex SwiGLU feed-forward network.
class TiRexFFN: Module {
    @ModuleInfo(key: "proj_up_gate") var projUpGate: Linear
    @ModuleInfo(key: "proj_up") var projUp: Linear
    @ModuleInfo(key: "proj_down") var projDown: Linear

    init(embeddingDim: Int, ffnDim: Int) {
        self._projUpGate.wrappedValue = Linear(embeddingDim, ffnDim, bias: false)
        self._projUp.wrappedValue = Linear(embeddingDim, ffnDim, bias: false)
        self._projDown.wrappedValue = Linear(ffnDim, embeddingDim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        projDown(silu(projUpGate(x)) * projUp(x))
    }
}

/// Residual block for patch embedding (shared by TiRex, Kairos, Chronos-2).
class ResidualBlock: Module {
    @ModuleInfo(key: "hidden_layer") var hiddenLayer: Linear
    @ModuleInfo(key: "output_layer") var outputLayer: Linear
    @ModuleInfo(key: "residual_layer") var residualLayer: Linear

    init(inputDim: Int, hiddenDim: Int, outputDim: Int) {
        self._hiddenLayer.wrappedValue = Linear(inputDim, hiddenDim)
        self._outputLayer.wrappedValue = Linear(hiddenDim, outputDim)
        self._residualLayer.wrappedValue = Linear(inputDim, outputDim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let hidden = relu(hiddenLayer(x))
        return outputLayer(hidden) + residualLayer(x)
    }
}

// MARK: - TiRexModel

/// TiRex time series forecasting model.
///
/// Stacked sLSTM blocks with patch-based input and quantile output.
/// 35M parameters, uses exponential gating for improved gradient flow.
public class TiRexModel: Module, TimeSeriesModel {

    @ModuleInfo(key: "input_patch_embedding") var inputEmbed: ResidualBlock
    @ModuleInfo(key: "blocks") var blocks: [TiRexBlock]
    @ModuleInfo(key: "out_norm") var outNorm: RMSNorm
    @ModuleInfo(key: "output_patch_embedding") var outputEmbed: ResidualBlock

    let config: TiRexConfiguration

    public init(_ config: TiRexConfiguration) {
        self.config = config
        // Input: patchSize * 2 (value + mask) -> ffnDim -> embeddingDim
        self._inputEmbed.wrappedValue = ResidualBlock(
            inputDim: config.inputPatchSize * 2,
            hiddenDim: config.ffnDim,
            outputDim: config.embeddingDim
        )
        self._blocks.wrappedValue = (0 ..< config.numLayers).map { _ in TiRexBlock(config) }
        self._outNorm.wrappedValue = RMSNorm(dimensions: config.embeddingDim)
        // Output: embeddingDim -> ffnDim -> numQuantiles * outputPatchSize
        self._outputEmbed.wrappedValue = ResidualBlock(
            inputDim: config.embeddingDim,
            hiddenDim: config.ffnDim,
            outputDim: config.numQuantiles * config.outputPatchSize
        )
    }

    public func forecast(
        input: TimeSeriesInput,
        predictionLength: Int,
        caches: [TimeSeriesKVCache?]
    ) -> TimeSeriesPrediction {
        let series = input.series  // [B, V, T]
        let mask = input.paddingMask
        let B = series.dim(0)
        let V = series.dim(1)
        let T = series.dim(2)
        let patchSize = config.inputPatchSize

        // Flatten to [B*V, T]
        let flat = series.reshaped(B * V, T)
        let flatMask = mask.reshaped(B * V, T)

        // Normalize
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

        // Create patches with mask: [B*V, numPatches, patchSize*2]
        let normPatches = paddedNorm.reshaped(B * V, numPatches, patchSize)
        let maskPatches = paddedMask.reshaped(B * V, numPatches, patchSize)
        let patchInput = MLX.concatenated([normPatches, maskPatches], axis: -1)

        // Embed patches
        var hidden = inputEmbed(patchInput)  // [B*V, numPatches, embeddingDim]

        // Process through sLSTM blocks
        for block in blocks {
            hidden = block(hidden)
        }
        hidden = outNorm(hidden)

        // Autoregressive prediction
        let numOutputPatches = (predictionLength + config.outputPatchSize - 1) / config.outputPatchSize
        var allPreds = [MLXArray]()

        for _ in 0 ..< numOutputPatches {
            // Use last hidden state
            let lastH = hidden[.ellipsis, (hidden.dim(1) - 1), 0...]  // [B*V, embeddingDim]
            let rawOut = outputEmbed(lastH.expandedDimensions(axis: 1))
            // [B*V, 1, numQuantiles * outputPatchSize]

            let reshaped = rawOut.reshaped(B * V, config.numQuantiles, config.outputPatchSize)
            let medianIdx = config.numQuantiles / 2
            let medianPred = reshaped[.ellipsis, medianIdx, 0...]  // [B*V, outputPatchSize]
            allPreds.append(medianPred)

            // Feed prediction back as next input patch (with mask=0 for NaN)
            let nextPatch = MLX.concatenated(
                [medianPred, MLXArray.zeros([B * V, config.outputPatchSize])],
                axis: -1
            ).expandedDimensions(axis: 1)  // [B*V, 1, patchSize*2]
            let nextEmbedded = inputEmbed(nextPatch)
            hidden = MLX.concatenated([hidden, nextEmbedded], axis: 1)
            for block in blocks {
                hidden = block(hidden)
            }
            hidden = outNorm(hidden)
        }

        // Concatenate and trim
        let fullPred = MLX.concatenated(allPreds, axis: -1)  // [B*V, numOutputPatches * patchSize]
        let trimmed = fullPred[.ellipsis, 0 ..< predictionLength]

        // Denormalize
        let denormalized = trimmed * stdVal + meanVal

        let meanOut = denormalized.reshaped(B, V, predictionLength)

        return TimeSeriesPrediction(
            mean: meanOut,
            predictionLength: predictionLength
        )
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = [String: MLXArray]()
        for (key, value) in weights {
            var newKey = key
            if newKey.hasPrefix("block_stack.") {
                newKey = String(newKey.dropFirst("block_stack.".count))
            }
            result[newKey] = value
        }
        return result
    }

    public func newCaches() -> [TimeSeriesKVCache?] {
        []  // sLSTM maintains internal recurrent state, not KV caches
    }
}
