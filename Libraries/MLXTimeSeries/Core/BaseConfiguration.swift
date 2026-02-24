import Foundation

/// Base configuration fields shared by all time series models.
///
/// Every converted model's config.json includes at minimum a `model_type` field.
/// This struct is used to peek at the model type before loading the full config.
public struct BaseConfiguration: Codable, Sendable {
    /// Model type identifier used by the registry (e.g. "toto", "chronos", "timesfm", "lag_llama").
    public let modelType: String

    /// Optional quantization parameters.
    public let quantization: QuantizationConfig?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case quantization
    }
}

/// Quantization configuration stored in config.json.
public struct QuantizationConfig: Codable, Sendable {
    public let bits: Int
    public let groupSize: Int

    enum CodingKeys: String, CodingKey {
        case bits
        case groupSize = "group_size"
    }
}
