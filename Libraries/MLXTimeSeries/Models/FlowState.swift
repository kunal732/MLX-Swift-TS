import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - Configuration

/// Configuration for the FlowState time series model (SSM encoder + Legendre decoder).
public struct FlowStateConfiguration: Codable, Sendable {
    public var embeddingFeatureDim: Int
    public var encoderStateDim: Int
    public var encoderNumLayers: Int
    public var encoderNumHippoBlocks: Int
    public var decoderDim: Int
    public var decoderPatchLen: Int
    public var decoderType: String
    public var contextLength: Int
    public var minContext: Int
    public var quantiles: [Float]
    public var withMissing: Bool

    /// Input channels: 1 (value) or 2 (value + missing mask).
    public var inputChannels: Int { withMissing ? 2 : 1 }

    /// Number of quantiles in the output.
    public var numQuantiles: Int { quantiles.count }

    enum CodingKeys: String, CodingKey {
        case embeddingFeatureDim = "embedding_feature_dim"
        case encoderStateDim = "encoder_state_dim"
        case encoderNumLayers = "encoder_num_layers"
        case encoderNumHippoBlocks = "encoder_num_hippo_blocks"
        case decoderDim = "decoder_dim"
        case decoderPatchLen = "decoder_patch_len"
        case decoderType = "decoder_type"
        case contextLength = "context_length"
        case minContext = "min_context"
        case quantiles
        case withMissing = "with_missing"
    }

    public init(
        embeddingFeatureDim: Int = 512,
        encoderStateDim: Int = 512,
        encoderNumLayers: Int = 6,
        encoderNumHippoBlocks: Int = 8,
        decoderDim: Int = 256,
        decoderPatchLen: Int = 24,
        decoderType: String = "legs",
        contextLength: Int = 2048,
        minContext: Int = 2048,
        quantiles: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        withMissing: Bool = true
    ) {
        self.embeddingFeatureDim = embeddingFeatureDim
        self.encoderStateDim = encoderStateDim
        self.encoderNumLayers = encoderNumLayers
        self.encoderNumHippoBlocks = encoderNumHippoBlocks
        self.decoderDim = decoderDim
        self.decoderPatchLen = decoderPatchLen
        self.decoderType = decoderType
        self.contextLength = contextLength
        self.minContext = minContext
        self.quantiles = quantiles
        self.withMissing = withMissing
    }
}

// MARK: - S5 Block (Simplified State Space Sequence)

/// S5 SSM block operating in the diagonalized complex eigenspace.
///
/// Uses FFT-based convolution for efficient parallel computation.
/// Parameters are stored as real/imaginary pairs since MLX doesn't have native complex support.
class S5Block: Module {
    @ModuleInfo(key: "log_Lambda_real") var logLambdaReal: MLXArray
    @ModuleInfo(key: "Lambda_imag") var lambdaImag: MLXArray
    @ModuleInfo(key: "B_tilde_r") var bTildeR: MLXArray
    @ModuleInfo(key: "B_tilde_i") var bTildeI: MLXArray
    @ModuleInfo(key: "C_tilde_r") var cTildeR: MLXArray
    @ModuleInfo(key: "C_tilde_i") var cTildeI: MLXArray
    @ModuleInfo(key: "D") var dMatrix: MLXArray
    @ModuleInfo(key: "log_Delta") var logDelta: MLXArray

    let stateDim: Int

    init(stateDim: Int) {
        self.stateDim = stateDim
        self._logLambdaReal.wrappedValue = MLXArray.zeros([stateDim])
        self._lambdaImag.wrappedValue = MLXArray.zeros([stateDim])
        self._bTildeR.wrappedValue = MLXArray.zeros([stateDim, stateDim])
        self._bTildeI.wrappedValue = MLXArray.zeros([stateDim, stateDim])
        self._cTildeR.wrappedValue = MLXArray.zeros([stateDim, stateDim])
        self._cTildeI.wrappedValue = MLXArray.zeros([stateDim, stateDim])
        self._dMatrix.wrappedValue = MLXArray.zeros([stateDim])
        self._logDelta.wrappedValue = MLXArray.zeros([stateDim])
    }

    /// Forward pass using full complex S5 recurrence.
    ///
    /// The S5 SSM operates in the diagonalized complex eigenspace.
    /// State, eigenvalues, B and C matrices all have real + imaginary parts.
    /// MLX has no native complex type so real/imaginary are tracked separately.
    ///
    /// State update: s_{t+1} = A_bar * s_t + B_tilde * u_t  (complex multiply)
    /// Output:       y_t = 2 * Re(C_tilde * s_t) + D * u_t
    ///
    /// - Parameter x: Input of shape `[L, B, stateDim]`.
    /// - Returns: Output of shape `[L, B, stateDim]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let L = x.dim(0)
        let B = x.dim(1)

        // Discretize complex eigenvalue: A_bar = exp((lambda_r + i*lambda_i) * delta)
        let lambdaR = MLX.negative(MLX.exp(logLambdaReal))  // [stateDim]
        let delta = MLX.exp(logDelta)                        // [stateDim]
        let rdelta = lambdaR * delta
        let idelta = lambdaImag * delta
        let expR = MLX.exp(rdelta)
        let aBarR = expR * MLX.cos(idelta)  // real part of A_bar [stateDim]
        let aBarI = expR * MLX.sin(idelta)  // imag part of A_bar [stateDim]

        // Project input through B (complex): Bu = (B_r + i*B_i) * u
        // x: [L, B, stateDim], B_tilde: [stateDim, stateDim]
        let buR = MLX.matmul(x, bTildeR.transposed())  // [L, B, stateDim]
        let buI = MLX.matmul(x, bTildeI.transposed())  // [L, B, stateDim]

        // Sequential complex recurrence
        var sR = MLXArray.zeros([B, stateDim])  // real part of state
        var sI = MLXArray.zeros([B, stateDim])  // imag part of state
        var outputs = [MLXArray]()

        for t in 0 ..< L {
            let inR = buR[t]  // [B, stateDim]
            let inI = buI[t]

            // Complex multiply: A_bar * state
            let newSR = aBarR * sR - aBarI * sI + inR
            let newSI = aBarR * sI + aBarI * sR + inI
            sR = newSR
            sI = newSI

            // Output: 2 * Re(C_tilde * state) + D * u
            // Re(C * s) = C_r * s_r - C_i * s_i
            let y = 2 * (MLX.matmul(sR, cTildeR.transposed())
                       - MLX.matmul(sI, cTildeI.transposed()))
                      + x[t] * dMatrix
            outputs.append(y.expandedDimensions(axis: 0))
        }

        return MLX.concatenated(outputs, axis: 0)  // [L, B, stateDim]
    }
}

/// S5 layer wrapping SSM block with gated MLP and layer norm.
class S5Layer: Module {
    @ModuleInfo(key: "ssm") var ssm: S5Block
    @ModuleInfo(key: "out") var outProj: Linear
    @ModuleInfo(key: "norm") var norm: LayerNorm

    init(stateDim: Int) {
        self._ssm.wrappedValue = S5Block(stateDim: stateDim)
        self._outProj.wrappedValue = Linear(stateDim, stateDim)
        self._norm.wrappedValue = LayerNorm(dimensions: stateDim)
    }

    /// Forward pass: SSM -> SELU -> sigmoid-gated MLP -> LayerNorm + skip.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let ssmOut = ssm(x)
        let activated = selu(ssmOut)
        let gated = activated * sigmoid(outProj(activated))
        return norm(gated + x)
    }
}

// MARK: - Legendre Basis Decoder

/// Functional basis decoder using Legendre polynomials.
///
/// Instead of producing discrete point predictions, outputs coefficients
/// for Legendre polynomial basis functions which are evaluated at arbitrary
/// time points to produce the forecast.
class LegendreBasisDecoder: Module {
    @ModuleInfo(key: "lin") var lin: Linear

    let decoderDim: Int
    let numQuantiles: Int
    let patchLen: Int
    let basisRange: (Float, Float)

    init(inputDim: Int, decoderDim: Int, numQuantiles: Int, patchLen: Int) {
        self.decoderDim = decoderDim
        self.numQuantiles = numQuantiles
        self.patchLen = patchLen
        self.basisRange = (-1.0, 0.95)
        // Output: numQuantiles * decoderDim coefficients
        self._lin.wrappedValue = Linear(inputDim, numQuantiles * decoderDim)
    }

    /// Decode encoder output to forecasts via Legendre polynomial reconstruction.
    ///
    /// - Parameters:
    ///   - x: Encoder hidden states, shape `[numPositions, B, inputDim]`.
    ///   - predictionLength: Number of forecast time steps.
    ///   - scaleFactor: Sampling rate scaling (affects basis evaluation range).
    /// - Returns: Forecast of shape `[B, numQuantiles, predictionLength]`.
    func callAsFunction(_ x: MLXArray, predictionLength: Int, scaleFactor: Float = 1.0) -> MLXArray {
        // Take the last position's hidden state
        let lastHidden = x[x.dim(0) - 1]  // [B, inputDim]

        // Project to Legendre coefficients
        let raw = lin(lastHidden)  // [B, numQuantiles * decoderDim]
        let B = raw.dim(0)
        let coefficients = raw.reshaped(B, numQuantiles, decoderDim)

        // Build Legendre basis kernel
        let kernel = buildLegendreKernel(
            predictionLength: predictionLength,
            scaleFactor: scaleFactor
        )  // [predictionLength, decoderDim]

        // Reconstruct: [B, numQuantiles, decoderDim] x [decoderDim, predictionLength]
        let forecast = MLX.matmul(coefficients, kernel.transposed())
        // [B, numQuantiles, predictionLength]

        return forecast
    }

    /// Build Legendre polynomial basis evaluated at forecast time points.
    private func buildLegendreKernel(predictionLength: Int, scaleFactor: Float) -> MLXArray {
        let (low, high) = basisRange
        let dt = scaleFactor * (high - low) / Float(patchLen)

        // Evaluation points
        var tValues = [Float]()
        for i in 1 ... predictionLength {
            let t = max(low, low + Float(i) * dt)
            tValues.append(t)
        }
        let t = MLXArray(tValues)  // [predictionLength]

        // Compute Legendre polynomials P_0(t) through P_{decoderDim-1}(t)
        var polynomials = [MLXArray]()
        let p0 = MLXArray.ones([predictionLength])
        polynomials.append(p0)

        if decoderDim > 1 {
            polynomials.append(t)
        }

        for n in 1 ..< (decoderDim - 1) {
            let nf = Float(n)
            let pn = polynomials[n]
            let pnm1 = polynomials[n - 1]
            let pnp1 = ((2 * nf + 1) * t * pn - nf * pnm1) / (nf + 1)
            polynomials.append(pnp1)
        }

        // Stack: [predictionLength, decoderDim]
        let kernel = MLX.stacked(polynomials, axis: -1) / 4.0
        return kernel
    }
}

// MARK: - FlowStateModel

/// FlowState time series forecasting model.
///
/// SSM (S5) encoder + Legendre polynomial functional basis decoder.
/// 9.07M parameters, processes raw time series without patching.
public class FlowStateModel: Module, TimeSeriesModel {

    @ModuleInfo(key: "embed") var embed: FlowStateEmbedding
    @ModuleInfo(key: "encoder") var encoder: FlowStateEncoder
    @ModuleInfo(key: "decoder") var decoder: LegendreBasisDecoder

    let config: FlowStateConfiguration

    public init(_ config: FlowStateConfiguration) {
        self.config = config
        self._embed.wrappedValue = FlowStateEmbedding(
            inputDim: config.inputChannels, outputDim: config.embeddingFeatureDim)
        self._encoder.wrappedValue = FlowStateEncoder(config)
        self._decoder.wrappedValue = LegendreBasisDecoder(
            inputDim: config.embeddingFeatureDim,
            decoderDim: config.decoderDim,
            numQuantiles: config.numQuantiles,
            patchLen: config.decoderPatchLen
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

        // Flatten to [B*V, T]
        let flat = series.reshaped(B * V, T)
        let flatMask = mask.reshaped(B * V, T)

        // Causal RevIN normalization
        let nObs = MLX.maximum(flatMask.sum(axis: -1, keepDims: true), MLXArray(Float(1)))
        let meanVal = (flat * flatMask).sum(axis: -1, keepDims: true) / nObs
        let variance = ((flat - meanVal) * (flat - meanVal) * flatMask)
            .sum(axis: -1, keepDims: true) / nObs
        let stdVal = MLX.maximum(MLX.sqrt(variance), MLXArray(Float(1e-5)))
        let normalized = (flat - meanVal) / stdVal

        // Prepare input: [T, B*V, channels]
        var inputFeatures: MLXArray
        if config.withMissing {
            let stacked = MLX.stacked(
                [normalized.transposed(), flatMask.transposed()], axis: -1)
            inputFeatures = stacked  // [T, B*V, 2]
        } else {
            inputFeatures = normalized.transposed().expandedDimensions(axis: -1)  // [T, B*V, 1]
        }

        // Embed
        let embedded = embed(inputFeatures)  // [T, B*V, embDim]

        // Encode
        let encoded = encoder(embedded)  // [T, B*V, embDim]

        // Decode
        let forecast = decoder(encoded, predictionLength: predictionLength)
        // [B*V, numQuantiles, predictionLength]

        // Extract median (quantile 0.5, index 4 for default 9 quantiles)
        let medianIdx = config.quantiles.count / 2
        let meanPred = forecast[.ellipsis, medianIdx, 0...]  // [B*V, predictionLength]

        // Denormalize
        let denormalized = meanPred * stdVal + meanVal

        // Reshape: [B, V, predictionLength]
        let meanOut = denormalized.reshaped(B, V, predictionLength)

        return TimeSeriesPrediction(
            mean: meanOut,
            predictionLength: predictionLength
        )
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights  // FlowState uses standard naming
    }

    public func newCaches() -> [TimeSeriesKVCache?] {
        []  // SSM doesn't use KV caches
    }
}

/// FlowState embedding layer.
class FlowStateEmbedding: Module {
    @ModuleInfo(key: "embed") var embed: Linear

    init(inputDim: Int, outputDim: Int) {
        self._embed.wrappedValue = Linear(inputDim, outputDim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        embed(x)
    }
}

/// FlowState S5 encoder stack.
class FlowStateEncoder: Module {
    @ModuleInfo(key: "layers") var layers: [S5Layer]

    init(_ config: FlowStateConfiguration) {
        self._layers.wrappedValue = (0 ..< config.encoderNumLayers).map { _ in
            S5Layer(stateDim: config.embeddingFeatureDim)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in layers {
            h = layer(h)
        }
        return h
    }
}
