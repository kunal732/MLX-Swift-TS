import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - Student-T Distribution Head (Lag-Llama)

/// Single Student-T distribution output head.
///
/// Produces parameters (df, loc, scale) for a Student-T distribution.
/// Used by Lag-Llama for probabilistic forecasting.
public class StudentTHead: Module {

    @ModuleInfo(key: "df_proj") var dfProj: Linear
    @ModuleInfo(key: "loc_proj") var locProj: Linear
    @ModuleInfo(key: "scale_proj") var scaleProj: Linear

    public init(inputDim: Int) {
        self._dfProj.wrappedValue = Linear(inputDim, 1)
        self._locProj.wrappedValue = Linear(inputDim, 1)
        self._scaleProj.wrappedValue = Linear(inputDim, 1)
    }

    /// Compute Student-T parameters from hidden states.
    ///
    /// - Parameter x: Hidden states of shape `[..., inputDim]`.
    /// - Returns: `StudentTParams` with df, loc, scale each of shape `[...]`.
    public func callAsFunction(_ x: MLXArray) -> StudentTParams {
        let df = softplus(dfProj(x).squeezed(axis: -1)) + 2.0
        let loc = locProj(x).squeezed(axis: -1)
        let scale = softplus(scaleProj(x).squeezed(axis: -1)) + 1e-6
        return StudentTParams(df: df, loc: loc, scale: scale)
    }
}

/// Parameters of a Student-T distribution.
public struct StudentTParams: Sendable {
    /// Degrees of freedom. Always > 2.
    public let df: MLXArray
    /// Location parameter (mean when df > 1).
    public let loc: MLXArray
    /// Scale parameter. Always > 0.
    public let scale: MLXArray

    /// The mean of the distribution (equals loc when df > 1).
    public func mean() -> MLXArray {
        loc
    }
}

// MARK: - Quantile Head (TimesFM)

/// Quantile regression output head.
///
/// Produces a point forecast (mean) plus quantile forecasts.
/// Used by TimesFM for probabilistic forecasting.
public class QuantileHead: Module {

    @ModuleInfo(key: "mean_proj") var meanProj: Linear
    @ModuleInfo(key: "quantile_proj") var quantileProj: Linear

    let numQuantiles: Int

    public init(inputDim: Int, outputDim: Int, numQuantiles: Int) {
        self.numQuantiles = numQuantiles
        self._meanProj.wrappedValue = Linear(inputDim, outputDim)
        self._quantileProj.wrappedValue = Linear(inputDim, outputDim * numQuantiles)
    }

    /// Compute mean and quantile forecasts from hidden states.
    ///
    /// - Parameter x: Hidden states of shape `[..., inputDim]`.
    /// - Returns: `QuantileOutput` with mean and quantile predictions.
    public func callAsFunction(_ x: MLXArray) -> QuantileOutput {
        let meanPred = meanProj(x)
        let quantileRaw = quantileProj(x)
        // Reshape: [..., outputDim * numQuantiles] -> [..., outputDim, numQuantiles]
        let shape = quantileRaw.shape
        let lastDim = shape[shape.count - 1]
        let outputDim = lastDim / numQuantiles
        var newShape = Array(shape.dropLast())
        newShape.append(outputDim)
        newShape.append(numQuantiles)
        let quantilePred = quantileRaw.reshaped(newShape)
        return QuantileOutput(mean: meanPred, quantiles: quantilePred)
    }
}

/// Output of the quantile head.
public struct QuantileOutput: Sendable {
    /// Point forecast (mean), shape `[..., outputDim]`.
    public let mean: MLXArray
    /// Quantile forecasts, shape `[..., outputDim, numQuantiles]`.
    public let quantiles: MLXArray
}
