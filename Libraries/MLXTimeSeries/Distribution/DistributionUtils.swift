import Foundation
import MLX

/// Numerically stable softplus: log(1 + exp(x)).
public func softplus(_ x: MLXArray) -> MLXArray {
    // For large x, softplus(x) ≈ x. For small x, use log(1 + exp(x)).
    MLX.log(1 + MLX.exp(MLX.minimum(x, MLXArray(Float(20))))) + MLX.maximum(
        x - 20, MLXArray(Float(0)))
}

/// Affine denormalization: transforms predictions from normalized space back to original scale.
///
/// - Parameters:
///   - values: Normalized values, shape `[B, V, T]` or `[B, V, nPatches, patchSize]`.
///   - loc: Per-patch mean from the scaler, shape `[B, V, nPatches]`.
///   - scale: Per-patch std from the scaler, shape `[B, V, nPatches]`.
///   - patchSize: Number of time steps per patch.
/// - Returns: Denormalized values in the original scale.
public func affineDenormalize(
    values: MLXArray, loc: MLXArray, scale: MLXArray, patchSize: Int
) -> MLXArray {
    // loc/scale: [B, V, nPatches] -> [B, V, nPatches, 1] for broadcasting
    let locExpanded = loc.expandedDimensions(axis: -1)
    let scaleExpanded = scale.expandedDimensions(axis: -1)
    // values: [B, V, nPatches, patchSize]
    return values * scaleExpanded + locExpanded
}
