import Foundation
@preconcurrency import MLX

/// Input structure for the Toto forecasting model.
///
/// Bundles time series data with its associated masks and metadata.
public struct TimeSeriesInput: Sendable {

    /// Time series values, shape `[B, V, T]`.
    /// B = batch size, V = number of variables, T = number of time steps.
    public let series: MLXArray

    /// Padding mask, shape `[B, V, T]`. 1 for observed values, 0 for padding.
    public let paddingMask: MLXArray

    /// ID mask, shape `[B, V]`. Variables with the same ID can attend to each other
    /// in space-wise attention layers.
    public let idMask: MLXArray

    /// Unix timestamps in seconds for each time step, shape `[B, V, T]`.
    /// Used for time-aware position encoding.
    public let timestampSeconds: MLXArray?

    /// Sampling interval in seconds between consecutive time steps, shape `[B, V]`.
    public let timeIntervalSeconds: MLXArray?

    /// Full initializer.
    public init(
        series: MLXArray,
        paddingMask: MLXArray,
        idMask: MLXArray,
        timestampSeconds: MLXArray? = nil,
        timeIntervalSeconds: MLXArray? = nil
    ) {
        self.series = series
        self.paddingMask = paddingMask
        self.idMask = idMask
        self.timestampSeconds = timestampSeconds
        self.timeIntervalSeconds = timeIntervalSeconds
    }

    /// Convenience initializer for univariate time series.
    ///
    /// Creates a single-batch, single-variable input with all values observed.
    /// - Parameter values: 1-D array of time series values.
    public static func univariate(_ values: MLXArray) -> TimeSeriesInput {
        let T = values.dim(0)
        return TimeSeriesInput(
            series: values.reshaped(1, 1, T),
            paddingMask: MLXArray.ones([1, 1, T]),
            idMask: MLXArray.zeros([1, 1], type: Int32.self)
        )
    }

    /// Convenience initializer from a Swift array of Floats.
    public static func univariate(_ values: [Float]) -> TimeSeriesInput {
        return univariate(MLXArray(values))
    }

    /// Left-pad the time series to a length that is a multiple of `patchSize`.
    ///
    /// - Parameter patchSize: Patch size to align to.
    /// - Returns: A new `TimeSeriesInput` with padded series and updated mask.
    public func padded(toPatchSize patchSize: Int) -> TimeSeriesInput {
        let T = series.dim(2)
        let remainder = T % patchSize
        if remainder == 0 {
            return self
        }

        let padLength = patchSize - remainder
        let B = series.dim(0)
        let V = series.dim(1)

        // Left-pad series with zeros
        let zeroPad = MLXArray.zeros([B, V, padLength], dtype: series.dtype)
        let paddedSeries = MLX.concatenated([zeroPad, series], axis: 2)

        // Left-pad mask with zeros (marking padded positions)
        let maskPad = MLXArray.zeros([B, V, padLength], dtype: paddingMask.dtype)
        let paddedMask = MLX.concatenated([maskPad, paddingMask], axis: 2)

        // Pad timestamps if present
        let paddedTimestamps: MLXArray?
        if let ts = timestampSeconds {
            let tsPad = MLXArray.zeros([B, V, padLength], dtype: ts.dtype)
            paddedTimestamps = MLX.concatenated([tsPad, ts], axis: 2)
        } else {
            paddedTimestamps = nil
        }

        return TimeSeriesInput(
            series: paddedSeries,
            paddingMask: paddedMask,
            idMask: idMask,
            timestampSeconds: paddedTimestamps,
            timeIntervalSeconds: timeIntervalSeconds
        )
    }
}
