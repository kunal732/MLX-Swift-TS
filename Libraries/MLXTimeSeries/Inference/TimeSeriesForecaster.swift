import Foundation
import Hub
@preconcurrency import MLX
import MLXNN

/// Unified public API for loading and running any supported time series model.
///
/// Uses the type registry to dispatch to the correct model implementation
/// based on `model_type` in config.json.
///
/// Usage:
/// ```swift
/// // Load from a local directory (pre-converted)
/// let forecaster = try TimeSeriesForecaster.loadFromDirectory(url)
///
/// // Load from HuggingFace Hub
/// let forecaster = try await TimeSeriesForecaster.loadFromHub(id: "mlx-community/toto-4bit")
///
/// // Forecast
/// let input = TimeSeriesInput.univariate(myData)
/// let result = forecaster.forecast(input: input, predictionLength: 64)
/// ```
public class TimeSeriesForecaster {

    /// The underlying model (type-erased).
    public let model: any TimeSeriesModel

    /// Base configuration (model_type, quantization info).
    public let baseConfig: BaseConfiguration

    /// Model type string (e.g. "toto", "chronos").
    public var modelType: String { baseConfig.modelType }

    /// The registry used to resolve model types.
    private let registry: TSModelTypeRegistry

    public init(model: any TimeSeriesModel, baseConfig: BaseConfiguration) {
        self.model = model
        self.baseConfig = baseConfig
        self.registry = TimeSeriesTypeRegistry.shared
    }

    // MARK: - Loading

    /// Load a model from a HuggingFace Hub repository.
    ///
    /// - Parameters:
    ///   - id: HuggingFace model ID (e.g. "mlx-community/toto-4bit").
    ///   - registry: Model type registry. Defaults to the shared built-in registry.
    ///   - progressHandler: Optional callback for download progress.
    /// - Returns: A ready-to-use `TimeSeriesForecaster`.
    public static func loadFromHub(
        id: String,
        registry: TSModelTypeRegistry? = nil,
        progressHandler: (@Sendable (Progress) -> Void)? = nil
    ) async throws -> TimeSeriesForecaster {
        let hub = HubApi()
        let repo = Hub.Repo(id: id)

        let modelDirectory = try await hub.snapshot(
            from: repo,
            matching: ["*.safetensors", "config.json"],
            progressHandler: progressHandler ?? { _ in }
        )

        return try loadFromDirectory(modelDirectory, registry: registry)
    }

    /// Load a model from a local directory containing config.json and safetensors files.
    ///
    /// - Parameters:
    ///   - directory: Path to the model directory.
    ///   - registry: Model type registry. Defaults to the shared built-in registry.
    /// - Returns: A ready-to-use `TimeSeriesForecaster`.
    public static func loadFromDirectory(
        _ directory: URL,
        registry: TSModelTypeRegistry? = nil
    ) throws -> TimeSeriesForecaster {
        let reg = registry ?? TimeSeriesTypeRegistry.shared

        // Load config
        let configURL = directory.appending(component: "config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw TimeSeriesModelError.missingConfigFile(configURL)
        }
        let configData = try Data(contentsOf: configURL)

        // Create model via registry
        let (model, baseConfig) = try reg.load(configData: configData)

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

        guard !weights.isEmpty else {
            throw TimeSeriesModelError.weightLoadingFailed(
                "No safetensors files found in \(directory.path)")
        }

        // Sanitize weight keys
        weights = model.sanitize(weights: weights)

        // Determine target dtype
        let targetDtype: DType = .float16
        for (key, value) in weights {
            if value.dtype != targetDtype && value.ndim >= 2 {
                weights[key] = value.asType(targetDtype)
            }
        }

        // Load weights into model
        // Use .allModelKeysSet only — verifies every model parameter has a weight,
        // but silently ignores extra keys in the file (e.g. unused buffers like fft_norm).
        let parameters = ModuleParameters.unflattened(weights)
        try model.update(parameters: parameters, verify: [.allModelKeysSet])
        eval(model)

        return TimeSeriesForecaster(model: model, baseConfig: baseConfig)
    }

    // MARK: - Inference

    /// Run forecast on the input time series.
    ///
    /// - Parameters:
    ///   - input: Input time series data.
    ///   - predictionLength: Number of future time steps to predict.
    /// - Returns: Unified prediction result.
    public func forecast(
        input: TimeSeriesInput,
        predictionLength: Int
    ) -> TimeSeriesPrediction {
        let caches = model.newCaches()
        return model.forecast(
            input: input,
            predictionLength: predictionLength,
            caches: caches
        )
    }
}
