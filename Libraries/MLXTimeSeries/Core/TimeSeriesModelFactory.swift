import Foundation
@preconcurrency import MLX
import MLXNN

/// A type-erased model creator for the registry.
public struct TSModelCreator: Sendable {
    /// Create a model and load config from JSON data.
    public let create: @Sendable (Data) throws -> any TimeSeriesModel

    public init(create: @escaping @Sendable (Data) throws -> any TimeSeriesModel) {
        self.create = create
    }
}

/// Registry mapping `model_type` strings to Swift model constructors.
///
/// Usage:
/// ```swift
/// let (model, baseConfig) = try TimeSeriesTypeRegistry.shared.load(
///     configData: jsonData
/// )
/// ```
public final class TSModelTypeRegistry: @unchecked Sendable {
    private var creators: [String: TSModelCreator]

    public init(creators: [String: TSModelCreator] = [:]) {
        self.creators = creators
    }

    /// Register a new model type.
    public func register(_ modelType: String, creator: TSModelCreator) {
        creators[modelType] = creator
    }

    /// Create a model from config JSON data.
    ///
    /// - Parameter configData: Raw JSON data from config.json.
    /// - Returns: Tuple of (model instance, base configuration).
    public func load(configData: Data) throws -> (any TimeSeriesModel, BaseConfiguration) {
        let baseConfig = try JSONDecoder().decode(BaseConfiguration.self, from: configData)
        guard let creator = creators[baseConfig.modelType] else {
            throw TimeSeriesModelError.unsupportedModelType(baseConfig.modelType)
        }
        let model = try creator.create(configData)
        return (model, baseConfig)
    }
}

/// Errors from the model factory.
public enum TimeSeriesModelError: Error, LocalizedError {
    case unsupportedModelType(String)
    case missingConfigFile(URL)
    case weightLoadingFailed(String)

    public var errorDescription: String? {
        switch self {
        case .unsupportedModelType(let type):
            return "Unsupported model type: '\(type)'. Registered types: toto, chronos, chronos_v2, timesfm, lag_llama, flowstate, kairos, tirex"
        case .missingConfigFile(let url):
            return "Missing config.json at \(url.path)"
        case .weightLoadingFailed(let msg):
            return "Weight loading failed: \(msg)"
        }
    }
}

/// Convenience function to create a `TSModelCreator` for a given configuration and model type.
public func createTSModel<C: Decodable & Sendable, M: TimeSeriesModel>(
    _ configType: C.Type,
    _ modelInit: @escaping @Sendable (C) -> M
) -> TSModelCreator {
    TSModelCreator { data in
        let config = try JSONDecoder().decode(C.self, from: data)
        return modelInit(config)
    }
}

/// The shared registry with all built-in time series model types.
public enum TimeSeriesTypeRegistry {
    public static let shared: TSModelTypeRegistry = {
        let registry = TSModelTypeRegistry(creators: [
            "toto": createTSModel(TotoConfiguration.self) { TotoModel($0) },
            "chronos": createTSModel(ChronosConfiguration.self) { ChronosModel($0) },
            "chronos_v2": createTSModel(Chronos2Configuration.self) { Chronos2Model($0) },
            "timesfm": createTSModel(TimesFMConfiguration.self) { TimesFMModel($0) },
            "lag_llama": createTSModel(LagLlamaConfiguration.self) { LagLlamaModel($0) },
            "flowstate": createTSModel(FlowStateConfiguration.self) { FlowStateModel($0) },
            "kairos": createTSModel(KairosConfiguration.self) { KairosModel($0) },
            "tirex": createTSModel(TiRexConfiguration.self) { TiRexModel($0) },
        ])
        return registry
    }()
}
