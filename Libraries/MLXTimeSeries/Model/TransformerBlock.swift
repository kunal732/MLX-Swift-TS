import Foundation
import MLX
import MLXNN

/// A single Toto transformer block with pre-norm residual connections.
///
/// Each block contains attention + SwiGLU MLP with RMSNorm.
/// Can operate in time-wise (causal) or space-wise (cross-variate) mode.
public class TotoTransformerBlock: Module {

    @ModuleInfo(key: "norm1") var norm1: RMSNorm
    @ModuleInfo(key: "norm2") var norm2: RMSNorm
    @ModuleInfo(key: "attention") var attention: TotoAttention
    @ModuleInfo(key: "mlp") var mlp: SwiGLUMLP

    let isSpacewise: Bool

    public init(_ config: TotoConfiguration, isSpacewise: Bool) {
        self.isSpacewise = isSpacewise
        self._norm1.wrappedValue = RMSNorm(dimensions: config.embedDim)
        self._norm2.wrappedValue = RMSNorm(dimensions: config.embedDim)
        self._attention.wrappedValue = TotoAttention(config, isSpacewise: isSpacewise)
        self._mlp.wrappedValue = SwiGLUMLP(
            embedDim: config.embedDim, mlpHiddenDim: config.mlpHiddenDim)
    }

    /// - Parameters:
    ///   - x: Input tensor of shape `[B, V, L, D]`.
    ///   - cache: Optional KV cache (time-wise layers only).
    ///   - idMask: Optional ID mask of shape `[B, V]` (space-wise layers only).
    ///   - rope: Optional RoPE+XPOS (time-wise layers only).
    /// - Returns: Output tensor of shape `[B, V, L, D]`.
    public func callAsFunction(
        _ x: MLXArray,
        cache: TimeSeriesKVCache?,
        idMask: MLXArray?,
        rope: RoPEXPOS?
    ) -> MLXArray {
        let B = x.dim(0)
        let V = x.dim(1)
        let L = x.dim(2)
        let D = x.dim(3)

        // Pre-norm + attention
        var h = norm1(x)

        if isSpacewise {
            // Space-wise: attend across variables
            // [B, V, L, D] -> [B, L, V, D] -> [B*L, V, D]
            h = h.transposed(0, 2, 1, 3).reshaped(B * L, V, D)

            // Build per-batch id mask for the space-wise attention
            // idMask: [B, V] -> repeat for each patch: [B*L, V]
            let expandedIdMask: MLXArray?
            if let idMask {
                // [B, V] -> [B, 1, V] -> broadcast to [B, L, V] -> [B*L, V]
                expandedIdMask = MLX.broadcast(
                    idMask.reshaped(B, 1, V),
                    to: [B, L, V]
                ).reshaped(B * L, V)
            } else {
                expandedIdMask = nil
            }

            h = attention(h, cache: nil, idMask: expandedIdMask, rope: nil)
            // [B*L, V, D] -> [B, L, V, D] -> [B, V, L, D]
            h = h.reshaped(B, L, V, D).transposed(0, 2, 1, 3)
        } else {
            // Time-wise: attend across time (patches)
            // [B, V, L, D] -> [B*V, L, D]
            h = h.reshaped(B * V, L, D)
            h = attention(h, cache: cache, idMask: nil, rope: rope)
            // [B*V, L, D] -> [B, V, L, D]
            h = h.reshaped(B, V, L, D)
        }

        let afterAttn = x + h

        // Pre-norm + MLP
        let mlpOut = mlp(norm2(afterAttn))
        return afterAttn + mlpOut
    }
}

/// The full transformer stack: 12 blocks (11 time-wise + 1 space-wise by default).
public class TotoTransformer: Module {

    let layers: [TotoTransformerBlock]

    /// RoPE with XPOS scaling (shared across all time-wise layers, not a Module).
    let rope: RoPEXPOS

    let config: TotoConfiguration

    public init(_ config: TotoConfiguration) {
        self.config = config
        self.rope = RoPEXPOS(dims: config.headDim)
        self.layers = (0 ..< config.numLayers).map { i in
            TotoTransformerBlock(config, isSpacewise: config.isSpacewise(layerIndex: i))
        }
    }

    /// - Parameters:
    ///   - x: Input tensor of shape `[B, V, nPatches, embedDim]`.
    ///   - caches: Array of KV caches, one per time-wise layer. Space-wise layers get `nil`.
    ///   - idMask: Optional ID mask of shape `[B, V]`.
    /// - Returns: Output tensor of shape `[B, V, nPatches, embedDim]`.
    public func callAsFunction(
        _ x: MLXArray,
        caches: [TimeSeriesKVCache?],
        idMask: MLXArray?
    ) -> MLXArray {
        var h = x
        for (i, layer) in layers.enumerated() {
            h = layer(h, cache: caches[i], idMask: idMask, rope: rope)
        }
        return h
    }
}
