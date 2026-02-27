import Foundation
@preconcurrency import MLX
import MLXNN

/// The Toto backbone: patch embedding + transformer + unembed + distribution head.
///
/// Weight key path: `model.{patch_embed, transformer, unembed, output_distribution}.*`
public class TotoBackbone: Module {

    @ModuleInfo(key: "patch_embed") var patchEmbed: PatchEmbedding
    @ModuleInfo(key: "transformer") var transformer: TotoTransformer
    @ModuleInfo(key: "unembed") var unembed: Linear
    @ModuleInfo(key: "output_distribution") var outputDistribution: MixtureOfStudentTs

    let config: TotoConfiguration

    public init(_ config: TotoConfiguration) {
        self.config = config
        self._patchEmbed.wrappedValue = PatchEmbedding(
            patchSize: config.patchSize, stride: config.stride, embedDim: config.embedDim)
        self._transformer.wrappedValue = TotoTransformer(config)
        // unembed: embedDim -> embedDim * patchSize
        self._unembed.wrappedValue = Linear(
            config.embedDim, config.embedDim * config.patchSize)
        self._outputDistribution.wrappedValue = MixtureOfStudentTs(
            embedDim: config.embedDim, kComponents: config.kComponents)
    }

    /// Forward pass through the backbone.
    ///
    /// - Parameters:
    ///   - patchEmbedded: Patch-embedded input, shape `[B, V, nPatches, embedDim]`.
    ///   - caches: KV caches for each layer.
    ///   - idMask: ID mask for space-wise attention, shape `[B, V]`.
    /// - Returns: Distribution parameters for each time step in each patch.
    public func callAsFunction(
        patchEmbedded: MLXArray,
        caches: [TimeSeriesKVCache?],
        idMask: MLXArray?
    ) -> TotoBackboneOutput {
        // Transformer: [B, V, nPatches, embedDim]
        let hidden = transformer(patchEmbedded, caches: caches, idMask: idMask)

        // Unembed: [B, V, nPatches, embedDim] -> [B, V, nPatches, embedDim * patchSize]
        let unembedded = unembed(hidden)

        // Reshape: [B, V, nPatches, patchSize, embedDim]
        let shape = unembedded.shape
        let reshaped = unembedded.reshaped(
            shape[0], shape[1], shape[2], config.patchSize, config.embedDim)

        // Distribution head: [B, V, nPatches, patchSize, embedDim] -> MixtureParams
        let params = outputDistribution(reshaped)

        return TotoBackboneOutput(hidden: hidden, params: params)
    }

    /// Embed raw time series patches into the transformer space.
    public func embed(_ x: MLXArray) -> MLXArray {
        patchEmbed(x)
    }
}

/// Output of the TotoBackbone forward pass.
public struct TotoBackboneOutput: Sendable {
    /// Last hidden state from the transformer, shape `[B, V, nPatches, embedDim]`.
    public let hidden: MLXArray
    /// Distribution parameters for each time step.
    public let params: MixtureParams
}

/// Top-level Toto model wrapper.
///
/// The `model` property corresponds to `Toto.model` in PyTorch,
/// giving weight key paths like `model.patch_embed.projection.weight`.
public class TotoModel: Module, TimeSeriesModel {

    @ModuleInfo(key: "model") var backbone: TotoBackbone

    let config: TotoConfiguration

    public init(_ config: TotoConfiguration) {
        self.config = config
        self._backbone.wrappedValue = TotoBackbone(config)
    }

    // MARK: - TimeSeriesModel conformance

    public func forecast(
        input: TimeSeriesInput,
        predictionLength: Int,
        caches: [TimeSeriesKVCache?]
    ) -> TimeSeriesPrediction {
        let patchSize = config.patchSize
        let scaler = CausalPatchScaler(patchSize: patchSize)

        let paddedInput = input.padded(toPatchSize: patchSize)
        let series = paddedInput.series
        let mask = paddedInput.paddingMask

        let (normalized, _, _, runLoc, runScale) = scaler(series, mask: mask)
        let embedded = backbone.embed(normalized)

        var output = backbone(
            patchEmbedded: embedded, caches: caches, idMask: paddedInput.idMask)
        eval(output.hidden)

        let numFuturePatches = (predictionLength + patchSize - 1) / patchSize
        var allPredictions = [MLXArray]()

        // Use running stats at the last timestep (matches reference create_affine_transformed)
        let T = runLoc.dim(2)
        let denomLoc = runLoc[.ellipsis, (T - 1) ..< T]     // [B, V, 1]
        let denomScale = runScale[.ellipsis, (T - 1) ..< T]  // [B, V, 1]


        for _ in 0 ..< numFuturePatches {
            let nP = output.params.df.dim(-3)
            let lastPatchParams = MixtureParams(
                df: output.params.df[.ellipsis, (nP - 1), 0..., 0...],
                loc: output.params.loc[.ellipsis, (nP - 1), 0..., 0...],
                scale: output.params.scale[.ellipsis, (nP - 1), 0..., 0...],
                weights: output.params.weights[.ellipsis, (nP - 1), 0..., 0...]
            )
            let predNormalized = lastPatchParams.mean()
            let predDenorm = predNormalized * denomScale + denomLoc
            allPredictions.append(predDenorm)

            let B = predNormalized.dim(0)
            let V = predNormalized.dim(1)
            let nextInput = predNormalized.reshaped(B, V, 1, patchSize)
            let nextEmbedded = backbone.patchEmbed.projection(nextInput)

            output = backbone(
                patchEmbedded: nextEmbedded, caches: caches, idMask: paddedInput.idMask)
            eval(output.hidden)
        }

        let fullPrediction = MLX.concatenated(allPredictions, axis: -1)
        let trimmed = fullPrediction[.ellipsis, 0 ..< predictionLength]

        return TimeSeriesPrediction(
            mean: trimmed,
            mixtureParams: output.params,
            predictionLength: predictionLength
        )
    }

    public func newCaches() -> [TimeSeriesKVCache?] {
        makeTimeSeriesCaches(config)
    }

    // MARK: - Weight sanitization

    /// Remap weight keys from PyTorch conventions to MLXNN conventions.
    ///
    /// - RMSNorm: `.scale` → `.weight` (PyTorch uses `scale`, MLXNN uses `weight`)
    /// - Filter out non-persistent RoPE buffers
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = [String: MLXArray]()
        for (key, value) in weights {
            // Filter out non-persistent RoPE frequency buffers
            if key.contains("rotary_emb.freqs") || key.contains("rotary_emb.scale") {
                continue
            }

            var newKey = key

            // Remap RMSNorm parameter name: PyTorch `.scale` -> MLXNN `.weight`
            if newKey.hasSuffix(".norm1.scale") || newKey.hasSuffix(".norm2.scale") {
                newKey = String(newKey.dropLast("scale".count)) + "weight"
            }

            // Remap xFormers fused SwiGLU keys to our named keys:
            //   mlp.0.w12.{weight,bias} -> mlp.gate_up.{weight,bias}
            //   mlp.0.w3.{weight,bias}  -> mlp.down.{weight,bias}
            newKey = newKey
                .replacingOccurrences(of: ".mlp.0.w12.", with: ".mlp.gate_up.")
                .replacingOccurrences(of: ".mlp.0.w3.", with: ".mlp.down.")

            result[newKey] = value
        }
        return result
    }
}
