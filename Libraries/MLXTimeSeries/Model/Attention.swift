import Foundation
import MLX
import MLXNN

/// Multi-head attention with fused QKV projection.
///
/// Supports two modes depending on the transformer layer type:
/// - **Time-wise** (causal): Attends across time steps with RoPE+XPOS and KV caching.
/// - **Space-wise** (cross-variate): Attends across variables with a block-diagonal ID mask.
public class TotoAttention: Module {

    @ModuleInfo(key: "wQKV") var wQKV: Linear
    @ModuleInfo(key: "wO") var wO: Linear

    let numHeads: Int
    let headDim: Int
    let scale: Float
    let isSpacewise: Bool

    public init(_ config: TotoConfiguration, isSpacewise: Bool) {
        let embedDim = config.embedDim
        self.numHeads = config.numHeads
        self.headDim = config.headDim
        self.scale = pow(Float(config.headDim), -0.5)
        self.isSpacewise = isSpacewise

        // Fused Q, K, V projection
        self._wQKV.wrappedValue = Linear(embedDim, 3 * embedDim)
        // Output projection
        self._wO.wrappedValue = Linear(embedDim, embedDim)
    }

    /// - Parameters:
    ///   - x: Input tensor of shape `[B', L, D]` where B' is the reshaped batch dimension.
    ///   - cache: Optional KV cache (time-wise layers only).
    ///   - idMask: Optional ID mask of shape `[B', V]` for block-diagonal attention (space-wise only).
    ///   - rope: Optional RoPE+XPOS embedding (time-wise layers only).
    /// - Returns: Output tensor of shape `[B', L, D]`.
    public func callAsFunction(
        _ x: MLXArray,
        cache: TimeSeriesKVCache?,
        idMask: MLXArray?,
        rope: RoPEXPOS?
    ) -> MLXArray {
        let Bp = x.dim(0)  // reshaped batch
        let L = x.dim(1)   // sequence length (nPatches or V)

        // Fused QKV projection and split
        let qkv = wQKV(x)  // [B', L, 3*D]
        let qkvParts = qkv.split(parts: 3, axis: -1)  // 3x [B', L, D]

        // Reshape to multi-head: [B', L, D] -> [B', nHeads, L, headDim]
        var q = qkvParts[0].reshaped(Bp, L, numHeads, headDim).transposed(0, 2, 1, 3)
        var k = qkvParts[1].reshaped(Bp, L, numHeads, headDim).transposed(0, 2, 1, 3)
        var v = qkvParts[2].reshaped(Bp, L, numHeads, headDim).transposed(0, 2, 1, 3)

        // Apply RoPE+XPOS (time-wise only)
        if !isSpacewise, let rope {
            let offset = cache?.offset ?? 0
            (q, k) = rope(queries: q, keys: k, offset: offset)
        }

        // Update KV cache (time-wise only)
        if let cache {
            let (cachedK, cachedV) = cache.update(keys: k, values: v)
            k = cachedK
            v = cachedV
        }

        // Build attention mask
        let mask = buildMask(queryLength: L, keyLength: k.dim(2), batchSize: Bp, idMask: idMask)

        // Scaled dot-product attention
        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask
        )

        // [B', nHeads, L, headDim] -> [B', L, D]
        let out = output.transposed(0, 2, 1, 3).reshaped(Bp, L, -1)
        return wO(out)
    }

    private func buildMask(
        queryLength: Int, keyLength: Int, batchSize: Int, idMask: MLXArray?
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if isSpacewise {
            // Block-diagonal mask: series with the same ID can attend to each other
            if let idMask {
                // idMask: [B', V] -> create [B', V, V] boolean mask
                let maskLeft = idMask.reshaped(batchSize, -1, 1)  // [B', V, 1]
                let maskRight = idMask.reshaped(batchSize, 1, -1)  // [B', 1, V]
                let blockMask = maskLeft .== maskRight  // [B', V, V]
                // Expand for heads: [B', 1, V, V]
                // Use additive mask: 0 for attend, -1e9 for masked
                let additiveMask = MLX.where(
                    blockMask.reshaped(batchSize, 1, queryLength, keyLength),
                    MLXArray(Float(0)),
                    MLXArray(Float(-1e9))
                )
                return .array(additiveMask)
            }
            // No mask: all variables attend to each other
            return .none
        } else {
            // Causal mask for time-wise attention
            if queryLength == 1 {
                return .none
            }
            return .causal
        }
    }
}
