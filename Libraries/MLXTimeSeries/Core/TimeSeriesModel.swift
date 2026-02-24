import Foundation
@preconcurrency import MLX
import MLXNN

/// Protocol for all time series forecasting models.
///
/// Each model (Toto, Chronos, TimesFM, Lag-Llama) conforms to this protocol,
/// enabling a unified loading and inference API via ``TimeSeriesModelFactory``.
public protocol TimeSeriesModel: Module {

    /// Run a forecast given preprocessed input.
    ///
    /// - Parameters:
    ///   - input: Preprocessed time series input.
    ///   - predictionLength: Number of future time steps to predict.
    ///   - caches: KV caches for autoregressive inference.
    /// - Returns: Forecast prediction containing mean, quantiles, and/or distribution params.
    func forecast(
        input: TimeSeriesInput,
        predictionLength: Int,
        caches: [TimeSeriesKVCache?]
    ) -> TimeSeriesPrediction

    /// Remap weight keys from the original framework's conventions to MLXNN conventions.
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray]

    /// Create fresh KV caches for this model's architecture.
    func newCaches() -> [TimeSeriesKVCache?]
}

/// Unified forecast output from any time series model.
public struct TimeSeriesPrediction: Sendable {
    /// Point forecast (mean), shape `[B, V, predictionLength]` or `[B, predictionLength]`.
    public let mean: MLXArray

    /// Optional quantile forecasts, shape `[B, V, predictionLength, nQuantiles]`.
    public let quantiles: MLXArray?

    /// Optional mixture distribution parameters (Toto-style).
    public let mixtureParams: MixtureParams?

    /// Number of predicted time steps.
    public let predictionLength: Int

    public init(
        mean: MLXArray,
        quantiles: MLXArray? = nil,
        mixtureParams: MixtureParams? = nil,
        predictionLength: Int
    ) {
        self.mean = mean
        self.quantiles = quantiles
        self.mixtureParams = mixtureParams
        self.predictionLength = predictionLength
    }
}
