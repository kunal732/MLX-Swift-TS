import Foundation
@preconcurrency import MLX
import MLXNN

/// Mixture of Student-T distribution output head.
///
/// Takes transformer hidden states and produces parameters for a mixture
/// of `k` Student-T distributions per time step. Used for probabilistic forecasting.
public class MixtureOfStudentTs: Module {

    @ModuleInfo(key: "df") var df: Linear
    @ModuleInfo(key: "loc_proj") var locProj: Linear
    @ModuleInfo(key: "scale_proj") var scaleProj: Linear
    @ModuleInfo(key: "mixture_weights") var mixtureWeights: Linear

    let kComponents: Int

    public init(embedDim: Int, kComponents: Int) {
        self.kComponents = kComponents
        self._df.wrappedValue = Linear(embedDim, kComponents)
        self._locProj.wrappedValue = Linear(embedDim, kComponents)
        self._scaleProj.wrappedValue = Linear(embedDim, kComponents)
        self._mixtureWeights.wrappedValue = Linear(embedDim, kComponents)
    }

    /// Compute distribution parameters from hidden states.
    ///
    /// - Parameter x: Hidden states of shape `[..., embedDim]`.
    /// - Returns: `MixtureParams` with all parameters of shape `[..., kComponents]`.
    public func callAsFunction(_ x: MLXArray) -> MixtureParams {
        // df: softplus + 2 to ensure df > 2 (finite variance)
        let dfValues = softplus(df(x)) + 2.0
        // loc: unconstrained
        let locValues = locProj(x)
        // scale: softplus to ensure positive
        let scaleValues = softplus(scaleProj(x))
        // weights: softmax to ensure they sum to 1
        let weightValues = MLX.softmax(mixtureWeights(x), axis: -1)

        return MixtureParams(
            df: dfValues,
            loc: locValues,
            scale: scaleValues,
            weights: weightValues
        )
    }
}

/// Parameters of a mixture of Student-T distributions.
public struct MixtureParams: Sendable {
    /// Degrees of freedom, shape `[..., k]`. Always > 2.
    public let df: MLXArray
    /// Location (mean) of each component, shape `[..., k]`.
    public let loc: MLXArray
    /// Scale of each component, shape `[..., k]`. Always > 0.
    public let scale: MLXArray
    /// Mixture weights, shape `[..., k]`. Sum to 1 along last axis.
    public let weights: MLXArray

    /// Compute the mean forecast as a weighted sum of component locations.
    ///
    /// For Student-T with df > 1, the mean equals the location parameter.
    /// - Returns: Weighted mean of shape `[...]` (last dimension summed out).
    public func mean() -> MLXArray {
        (weights * loc).sum(axis: -1)
    }
}
