import Foundation
@preconcurrency import MLX

/// Per-model preprocessing utilities.
///
/// Each model type may need different preprocessing:
/// - **Toto**: Patch-based causal normalization (handled by CausalPatchScaler)
/// - **Chronos**: Token-based mean-scale normalization (handled by ChronosTokenizer)
/// - **TimesFM**: Patch-based normalization with residual embedding
/// - **Lag-Llama**: Lag feature extraction (handled by LagFeatureExtractor)

/// Standard mean/scale normalization for time series.
///
/// Computes the mean and standard deviation of observed values and normalizes.
/// Simpler than CausalPatchScaler — no patch-based causality.
public struct StandardScaler: Sendable {
    let minimumScale: Float

    public init(minimumScale: Float = 1e-5) {
        self.minimumScale = minimumScale
    }

    /// Normalize a time series.
    ///
    /// - Parameters:
    ///   - data: Input time series, shape `[B, T]` or `[B, V, T]`.
    ///   - mask: Observation mask (1=observed, 0=padding), same shape as data.
    /// - Returns: Tuple of (normalized, mean, scale).
    public func callAsFunction(
        _ data: MLXArray, mask: MLXArray
    ) -> (normalized: MLXArray, mean: MLXArray, scale: MLXArray) {
        let maskedData = data * mask
        let nObs = MLX.maximum(mask.sum(axis: -1, keepDims: true), MLXArray(Float(1)))
        let mean = maskedData.sum(axis: -1, keepDims: true) / nObs
        let variance = ((maskedData - mean * mask) * (maskedData - mean * mask) * mask)
            .sum(axis: -1, keepDims: true) / nObs
        let scale = MLX.maximum(MLX.sqrt(variance), MLXArray(minimumScale))

        let normalized = (data - mean) / scale
        return (normalized: normalized, mean: mean, scale: scale)
    }
}

/// Denormalize predictions back to original scale.
///
/// - Parameters:
///   - predictions: Normalized predictions.
///   - mean: Mean used for normalization.
///   - scale: Scale used for normalization.
/// - Returns: Denormalized predictions.
public func denormalize(_ predictions: MLXArray, mean: MLXArray, scale: MLXArray) -> MLXArray {
    predictions * scale + mean
}
