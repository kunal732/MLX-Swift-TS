import Foundation
import MLX

/// Causal patch-wise mean/std normalization using Welford's online algorithm.
///
/// Each patch is normalized using statistics computed from ALL PREVIOUS patches only,
/// ensuring no information leakage from future data. The first patch uses its own statistics.
///
/// This is a pure computation module with no learnable parameters.
public struct CausalPatchScaler: Sendable {

    let patchSize: Int
    let minimumScale: Float

    public init(patchSize: Int, minimumScale: Float = 1e-5) {
        self.patchSize = patchSize
        self.minimumScale = minimumScale
    }

    /// Normalize a time series using causal patch-wise statistics.
    ///
    /// - Parameters:
    ///   - data: Input time series, shape `[B, V, T]`.
    ///   - mask: Observation mask, shape `[B, V, T]`. 1 for observed, 0 for padding.
    /// - Returns: Tuple of (normalized data, per-patch causal locations, per-patch causal scales,
    ///   per-timestep running mean, per-timestep running std).
    ///   - `normalized`: Shape `[B, V, T]`
    ///   - `locs`: Shape `[B, V, nPatches]` — causal mean used to normalize each patch
    ///   - `scales`: Shape `[B, V, nPatches]` — causal std used to normalize each patch
    ///   - `runLoc`: Shape `[B, V, T]` — running mean at each timestep
    ///   - `runScale`: Shape `[B, V, T]` — running std at each timestep
    public func callAsFunction(
        _ data: MLXArray, mask: MLXArray
    ) -> (normalized: MLXArray, locs: MLXArray, scales: MLXArray, runLoc: MLXArray, runScale: MLXArray) {
        let B = data.dim(0)
        let V = data.dim(1)
        let T = data.dim(2)
        let nPatches = T / patchSize

        // Apply mask: zero out unobserved values
        let maskedData = data * mask

        // Compute running statistics using cumulative sums (Welford's)
        let cumN = MLX.cumsum(mask, axis: -1)  // [B, V, T]
        let cumSum = MLX.cumsum(maskedData, axis: -1)  // [B, V, T]
        let cumSumSq = MLX.cumsum(maskedData * maskedData, axis: -1)  // [B, V, T]

        // Safe division: avoid division by zero
        let safeN = MLX.maximum(cumN, MLXArray(Float(1)))
        let runningMean = cumSum / safeN  // [B, V, T]
        let runningVar = cumSumSq / safeN - runningMean * runningMean
        let runningStd = MLX.sqrt(MLX.maximum(runningVar, MLXArray(Float(0))))

        // Sample statistics at patch boundaries.
        // For causal normalization, patch i uses stats from the END of patch i-1.
        // Patch 0 uses its own end-of-patch stats (no prior data available).

        // Reshape to patches: [B, V, nPatches, patchSize]
        let meanPatches = runningMean.reshaped(B, V, nPatches, patchSize)
        let stdPatches = runningStd.reshaped(B, V, nPatches, patchSize)

        // Take stats at the last timestep of each patch: [B, V, nPatches]
        let patchEndMeans = meanPatches[.ellipsis, patchSize - 1]  // [B, V, nPatches]
        let patchEndStds = stdPatches[.ellipsis, patchSize - 1]  // [B, V, nPatches]

        // Shift by one for causality: patch i uses stats from end of patch i-1
        // Patch 0 gets the stats from its own end (no future leakage since it's the first patch)
        var locs: MLXArray
        var scales: MLXArray

        if nPatches > 1 {
            // Shift: [stats_patch0, stats_patch0, stats_patch1, ..., stats_patch_{n-2}]
            let firstLoc = patchEndMeans[.ellipsis, 0 ..< 1]  // [B, V, 1]
            let shiftedLocs = patchEndMeans[.ellipsis, 0 ..< (nPatches - 1)]  // [B, V, nPatches-1]
            locs = MLX.concatenated([firstLoc, shiftedLocs], axis: -1)  // [B, V, nPatches]

            let firstScale = patchEndStds[.ellipsis, 0 ..< 1]
            let shiftedScales = patchEndStds[.ellipsis, 0 ..< (nPatches - 1)]
            scales = MLX.concatenated([firstScale, shiftedScales], axis: -1)
        } else {
            locs = patchEndMeans
            scales = patchEndStds
        }

        // Enforce minimum scale
        scales = MLX.maximum(scales, MLXArray(minimumScale))

        // Normalize: expand locs/scales to [B, V, nPatches, 1] for broadcasting
        let locsExpanded = locs.expandedDimensions(axis: -1)  // [B, V, nPatches, 1]
        let scalesExpanded = scales.expandedDimensions(axis: -1)

        let dataPatches = data.reshaped(B, V, nPatches, patchSize)
        let normalizedPatches = (dataPatches - locsExpanded) / scalesExpanded
        let normalized = normalizedPatches.reshaped(B, V, T)

        // Per-timestep running stats (for denormalization of forecasts)
        let runScale = MLX.maximum(runningStd, MLXArray(minimumScale))

        return (normalized: normalized, locs: locs, scales: scales, runLoc: runningMean, runScale: runScale)
    }
}
