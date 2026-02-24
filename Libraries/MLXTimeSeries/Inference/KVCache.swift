import Foundation
import MLX

/// Minimal KV cache for time-wise attention layers.
///
/// Concatenates new keys/values to the existing cache on each step.
/// Only used for time-wise layers; space-wise layers have no cache.
public class TimeSeriesKVCache {

    /// Cached keys, shape `[B', nHeads, cachedLen, headDim]`.
    public private(set) var keys: MLXArray?

    /// Cached values, shape `[B', nHeads, cachedLen, headDim]`.
    public private(set) var values: MLXArray?

    /// Number of cached time steps.
    public private(set) var offset: Int = 0

    public init() {}

    /// Append new keys and values to the cache and return the full cached sequence.
    ///
    /// - Parameters:
    ///   - keys: New keys, shape `[B', nHeads, newLen, headDim]`.
    ///   - values: New values, shape `[B', nHeads, newLen, headDim]`.
    /// - Returns: Tuple of (allKeys, allValues) including the newly appended data.
    public func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        if let existingKeys = self.keys, let existingValues = self.values {
            self.keys = MLX.concatenated([existingKeys, newKeys], axis: 2)
            self.values = MLX.concatenated([existingValues, newValues], axis: 2)
        } else {
            self.keys = newKeys
            self.values = newValues
        }
        offset += newKeys.dim(2)
        return (self.keys!, self.values!)
    }

    /// Reset the cache (e.g. for a new sequence).
    public func reset() {
        keys = nil
        values = nil
        offset = 0
    }
}

/// Create an array of KV caches for a Toto model.
///
/// Time-wise layers get a cache; space-wise layers get `nil`.
/// - Parameter config: Model configuration.
/// - Returns: Array of optional caches, one per layer.
public func makeTimeSeriesCaches(_ config: TotoConfiguration) -> [TimeSeriesKVCache?] {
    (0 ..< config.numLayers).map { i in
        config.isSpacewise(layerIndex: i) ? nil : TimeSeriesKVCache()
    }
}
