import Foundation
import Hub
@preconcurrency import MLX
import MLXNN

/// Result of a Toto forecast.
public struct ForecastResult: Sendable {
    /// Point forecast (mean of the mixture distribution), shape `[B, V, predictionLength]`.
    public let mean: MLXArray

    /// Per-patch distribution parameters for the full forecast.
    public let params: MixtureParams

    /// Number of predicted time steps.
    public let predictionLength: Int
}

/// Public API for loading and running the Toto time series forecasting model.
///
/// Usage:
/// ```swift
/// let forecaster = try await TotoForecaster.loadFromHub(id: "Datadog/Toto-Open-Base-1.0")
/// let input = TimeSeriesInput.univariate(myData)
/// let forecast = forecaster.forecast(input: input, predictionLength: 256)
/// ```
public class TotoForecaster {

    /// The underlying Toto model.
    public let model: TotoModel

    /// Model configuration.
    public let config: TotoConfiguration

    /// Causal patch scaler for normalization.
    let scaler: CausalPatchScaler

    public init(model: TotoModel, config: TotoConfiguration) {
        self.model = model
        self.config = config
        self.scaler = CausalPatchScaler(patchSize: config.patchSize)
    }

    // MARK: - Loading

    /// Load a Toto model from a HuggingFace Hub repository.
    ///
    /// - Parameters:
    ///   - id: HuggingFace model ID (e.g. "Datadog/Toto-Open-Base-1.0").
    ///   - progressHandler: Optional callback for download progress.
    /// - Returns: A ready-to-use `TotoForecaster`.
    public static func loadFromHub(
        id: String,
        progressHandler: (@Sendable (Progress) -> Void)? = nil
    ) async throws -> TotoForecaster {
        let hub = HubApi()
        let repo = Hub.Repo(id: id)

        // Download model files
        let modelDirectory = try await hub.snapshot(
            from: repo,
            matching: ["*.safetensors", "config.json"],
            progressHandler: progressHandler ?? { _ in }
        )

        return try loadFromDirectory(modelDirectory)
    }

    /// Load a Toto model from a local directory containing config.json and safetensors files.
    ///
    /// - Parameter directory: Path to the model directory.
    /// - Returns: A ready-to-use `TotoForecaster`.
    public static func loadFromDirectory(_ directory: URL) throws -> TotoForecaster {
        // Load config
        let configURL = directory.appending(component: "config.json")
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(TotoConfiguration.self, from: configData)

        // Create model
        let model = TotoModel(config)

        // Load weights from all safetensors files
        var weights = [String: MLXArray]()
        let enumerator = FileManager.default.enumerator(
            at: directory, includingPropertiesForKeys: nil)!
        for case let url as URL in enumerator {
            if url.pathExtension == "safetensors" {
                let w = try loadArrays(url: url)
                for (key, value) in w {
                    weights[key] = value
                }
            }
        }

        // Sanitize weight keys (RMSNorm .scale -> .weight, filter RoPE buffers)
        weights = model.sanitize(weights: weights)

        // Cast to float16 for efficiency
        for (key, value) in weights {
            if value.dtype != .float16 {
                weights[key] = value.asType(.float16)
            }
        }

        // Load weights into model
        let parameters = ModuleParameters.unflattened(weights)
        try model.update(parameters: parameters, verify: [.all])
        eval(model)

        return TotoForecaster(model: model, config: config)
    }

    // MARK: - Inference

    /// Run autoregressive forecast on the input time series.
    ///
    /// - Parameters:
    ///   - input: Input time series data.
    ///   - predictionLength: Number of future time steps to predict.
    /// - Returns: Forecast result with mean predictions and distribution parameters.
    public func forecast(input: TimeSeriesInput, predictionLength: Int) -> ForecastResult {
        let patchSize = config.patchSize

        // Step 1: Pad input to multiple of patchSize
        let paddedInput = input.padded(toPatchSize: patchSize)
        let series = paddedInput.series
        let mask = paddedInput.paddingMask

        // Step 2: Normalize with causal patch scaler
        let (normalized, _, _, runLoc, runScale) = scaler(series, mask: mask)

        // Step 3: Patch embed
        let embedded = model.backbone.embed(normalized)
        // embedded: [B, V, nPatches, embedDim]

        // Step 4: Create KV caches
        let caches = makeTimeSeriesCaches(config)

        // Step 5: Process input through transformer (fills KV cache)
        var output = model.backbone(
            patchEmbedded: embedded,
            caches: caches,
            idMask: paddedInput.idMask
        )
        eval(output.hidden)

        // Step 6: Autoregressive loop for future patches
        let numFuturePatches = (predictionLength + patchSize - 1) / patchSize
        var allPredictions = [MLXArray]()

        // Use running stats at the last timestep (matches reference create_affine_transformed)
        let T = runLoc.dim(2)
        let denomLoc = runLoc[.ellipsis, (T - 1) ..< T]     // [B, V, 1]
        let denomScale = runScale[.ellipsis, (T - 1) ..< T]  // [B, V, 1]

        for _ in 0 ..< numFuturePatches {
            // Get mean prediction from the last patch's distribution
            // params shape: [B, V, nPatches, patchSize, kComponents]
            let nP = output.params.df.dim(-3)
            let lastPatchParams = MixtureParams(
                df: output.params.df[.ellipsis, (nP - 1), 0..., 0...],
                loc: output.params.loc[.ellipsis, (nP - 1), 0..., 0...],
                scale: output.params.scale[.ellipsis, (nP - 1), 0..., 0...],
                weights: output.params.weights[.ellipsis, (nP - 1), 0..., 0...]
            )
            // Mean prediction in normalized space: [B, V, patchSize]
            let predNormalized = lastPatchParams.mean()

            // Denormalize for output
            let predDenorm = predNormalized * denomScale + denomLoc
            allPredictions.append(predDenorm)

            // Feed the normalized prediction back directly — the model produced it
            // in normalized space, so it should go back without re-normalization.
            let B = predNormalized.dim(0)
            let V = predNormalized.dim(1)
            let nextInput = predNormalized.reshaped(B, V, 1, patchSize)
            let nextEmbedded = model.backbone.patchEmbed.projection(nextInput)

            output = model.backbone(
                patchEmbedded: nextEmbedded,
                caches: caches,
                idMask: paddedInput.idMask
            )
            eval(output.hidden)
        }

        // Step 7: Concatenate and trim predictions
        let fullPrediction = MLX.concatenated(allPredictions, axis: -1)
        let trimmed = fullPrediction[.ellipsis, 0 ..< predictionLength]

        return ForecastResult(
            mean: trimmed,
            params: output.params,
            predictionLength: predictionLength
        )
    }
}
