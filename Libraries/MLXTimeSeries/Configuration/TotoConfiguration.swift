import Foundation

/// Configuration for the Toto time series forecasting model.
///
/// Decodes from the HuggingFace `config.json` for `Datadog/Toto-Open-Base-1.0`.
public struct TotoConfiguration: Codable, Sendable {
    public var patchSize: Int
    public var stride: Int
    public var embedDim: Int
    public var numLayers: Int
    public var numHeads: Int
    public var mlpHiddenDim: Int
    public var dropout: Float
    public var spacewiseEveryNLayers: Int
    public var spacewiseFirst: Bool
    public var kComponents: Int
    public var useMemoryEfficientAttention: Bool
    public var stabilizeWithGlobal: Bool
    public var scaleFactorExponent: Float

    public var headDim: Int { embedDim / numHeads }

    enum CodingKeys: String, CodingKey {
        case patchSize = "patch_size"
        case stride
        case embedDim = "embed_dim"
        case numLayers = "num_layers"
        case numHeads = "num_heads"
        case mlpHiddenDim = "mlp_hidden_dim"
        case dropout
        case spacewiseEveryNLayers = "spacewise_every_n_layers"
        case spacewiseFirst = "spacewise_first"
        case outputDistributionKwargs = "output_distribution_kwargs"
        case useMemoryEfficientAttention = "use_memory_efficient_attention"
        case stabilizeWithGlobal = "stabilize_with_global"
        case scaleFactorExponent = "scale_factor_exponent"
    }

    struct OutputDistributionKwargs: Codable {
        var kComponents: Int

        enum CodingKeys: String, CodingKey {
            case kComponents = "k_components"
        }
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        patchSize = try container.decode(Int.self, forKey: .patchSize)
        stride = try container.decode(Int.self, forKey: .stride)
        embedDim = try container.decode(Int.self, forKey: .embedDim)
        numLayers = try container.decode(Int.self, forKey: .numLayers)
        numHeads = try container.decode(Int.self, forKey: .numHeads)
        mlpHiddenDim = try container.decode(Int.self, forKey: .mlpHiddenDim)
        dropout = try container.decodeIfPresent(Float.self, forKey: .dropout) ?? 0.1
        spacewiseEveryNLayers = try container.decode(
            Int.self, forKey: .spacewiseEveryNLayers)
        spacewiseFirst =
            try container.decodeIfPresent(Bool.self, forKey: .spacewiseFirst) ?? false
        let kwargs = try container.decode(
            OutputDistributionKwargs.self, forKey: .outputDistributionKwargs)
        kComponents = kwargs.kComponents
        useMemoryEfficientAttention =
            try container.decodeIfPresent(Bool.self, forKey: .useMemoryEfficientAttention) ?? true
        stabilizeWithGlobal =
            try container.decodeIfPresent(Bool.self, forKey: .stabilizeWithGlobal) ?? true
        scaleFactorExponent =
            try container.decodeIfPresent(Float.self, forKey: .scaleFactorExponent) ?? 10.0
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(patchSize, forKey: .patchSize)
        try container.encode(stride, forKey: .stride)
        try container.encode(embedDim, forKey: .embedDim)
        try container.encode(numLayers, forKey: .numLayers)
        try container.encode(numHeads, forKey: .numHeads)
        try container.encode(mlpHiddenDim, forKey: .mlpHiddenDim)
        try container.encode(dropout, forKey: .dropout)
        try container.encode(spacewiseEveryNLayers, forKey: .spacewiseEveryNLayers)
        try container.encode(spacewiseFirst, forKey: .spacewiseFirst)
        try container.encode(
            OutputDistributionKwargs(kComponents: kComponents),
            forKey: .outputDistributionKwargs)
        try container.encode(useMemoryEfficientAttention, forKey: .useMemoryEfficientAttention)
        try container.encode(stabilizeWithGlobal, forKey: .stabilizeWithGlobal)
        try container.encode(scaleFactorExponent, forKey: .scaleFactorExponent)
    }

    /// Memberwise initializer for programmatic construction.
    public init(
        patchSize: Int = 64,
        stride: Int = 64,
        embedDim: Int = 768,
        numLayers: Int = 12,
        numHeads: Int = 12,
        mlpHiddenDim: Int = 3072,
        dropout: Float = 0.1,
        spacewiseEveryNLayers: Int = 12,
        spacewiseFirst: Bool = false,
        kComponents: Int = 24,
        useMemoryEfficientAttention: Bool = true,
        stabilizeWithGlobal: Bool = true,
        scaleFactorExponent: Float = 10.0
    ) {
        self.patchSize = patchSize
        self.stride = stride
        self.embedDim = embedDim
        self.numLayers = numLayers
        self.numHeads = numHeads
        self.mlpHiddenDim = mlpHiddenDim
        self.dropout = dropout
        self.spacewiseEveryNLayers = spacewiseEveryNLayers
        self.spacewiseFirst = spacewiseFirst
        self.kComponents = kComponents
        self.useMemoryEfficientAttention = useMemoryEfficientAttention
        self.stabilizeWithGlobal = stabilizeWithGlobal
        self.scaleFactorExponent = scaleFactorExponent
    }

    /// Returns true if the given layer index uses space-wise (cross-variate) attention.
    public func isSpacewise(layerIndex: Int) -> Bool {
        if spacewiseFirst {
            return layerIndex % spacewiseEveryNLayers == 0
        } else {
            return (layerIndex + 1) % spacewiseEveryNLayers == 0
        }
    }
}
