import Foundation
@preconcurrency import MLX

/// Rotary Position Embeddings with XPOS scaling.
///
/// Manual implementation because `MLXFast.RoPE` doesn't support XPOS.
/// XPOS applies a position-dependent decay scaling to Q and K, improving
/// length generalization for the attention mechanism.
///
/// Reference: rotary-embedding-torch with `use_xpos=True`.
public struct RoPEXPOS: @unchecked Sendable {

    /// Inverse frequencies for the rotary embedding, shape `[halfDim]`.
    let invFreqs: MLXArray

    /// Per-dimension XPOS decay rates, shape `[halfDim]`.
    let xposScale: MLXArray

    /// Base for the XPOS power computation. Default 512.
    let scaleBase: Float

    /// Full head dimension (e.g. 64).
    let dims: Int

    /// Half of the head dimension.
    let halfDim: Int

    /// Create RoPE with XPOS scaling.
    ///
    /// - Parameters:
    ///   - dims: Head dimension (e.g. 64 for embed_dim=768, num_heads=12).
    ///   - theta: Base for inverse frequency computation. Default 10000.
    ///   - scaleBase: XPOS scale base. Default 512.
    public init(dims: Int, theta: Float = 10000.0, scaleBase: Float = 512.0) {
        self.dims = dims
        self.halfDim = dims / 2
        self.scaleBase = scaleBase

        // inv_freq = 1 / (theta ^ (2i / dims)) for i in 0..<halfDim
        let indicesArray = Array(stride(from: Float(0), to: Float(dims), by: 2))
        let indices = MLXArray(indicesArray)
        self.invFreqs = 1.0 / MLX.pow(MLXArray(theta), indices / Float(dims))

        // XPOS scale = (2i + 0.4 * dims) / (1.4 * dims)
        self.xposScale = (indices + 0.4 * Float(dims)) / (1.4 * Float(dims))
    }

    /// Apply RoPE with XPOS scaling to queries and keys.
    ///
    /// - Parameters:
    ///   - queries: Shape `[B, nHeads, L, headDim]`
    ///   - keys: Shape `[B, nHeads, L, headDim]`
    ///   - offset: Position offset (from KV cache). Default 0.
    /// - Returns: Tuple of (rotated queries with XPOS scale, rotated keys with inverse XPOS scale).
    public func callAsFunction(
        queries: MLXArray, keys: MLXArray, offset: Int = 0
    ) -> (MLXArray, MLXArray) {
        let seqLen = queries.dim(2)
        let posArray = (offset ..< (offset + seqLen)).map { Float($0) }
        let positions = MLXArray(posArray)

        // Compute rotation frequencies: outer(positions, invFreqs) -> [L, halfDim]
        let freqs = MLX.matmul(
            positions.reshaped(seqLen, 1),
            invFreqs.reshaped(1, halfDim)
        )  // [L, halfDim]

        let cosFreqs = MLX.cos(freqs)  // [L, halfDim]
        let sinFreqs = MLX.sin(freqs)  // [L, halfDim]

        // XPOS power: position / scaleBase (simplified, see explanation below)
        // The relative scaling Q_scale[i] / K_scale[j] = xposScale^((i-j)/scaleBase)
        // only depends on position difference, so centering doesn't affect attention.
        let power = positions / scaleBase  // [L]
        // scales: xposScale[d] ^ power[t] -> [L, halfDim]
        let scales = MLX.pow(
            xposScale.reshaped(1, halfDim),
            power.reshaped(seqLen, 1)
        )

        // Apply interleaved RoPE rotation
        let rotQ = applyRotaryEmb(queries, cos: cosFreqs, sin: sinFreqs)
        let rotK = applyRotaryEmb(keys, cos: cosFreqs, sin: sinFreqs)

        // Duplicate scales for interleaved pairs: [s0, s0, s1, s1, ...]
        // shape: [L, halfDim] -> [L, dims]
        let scalesFull = repeatInterleave(scales, count: 2)
        // Broadcast to [1, 1, L, dims]
        let s = scalesFull.reshaped(1, 1, seqLen, dims)

        let rotQScaled: MLXArray = rotQ * s
        let rotKScaled: MLXArray = rotK / s
        return (rotQScaled, rotKScaled)
    }

    /// Apply rotary embedding using the interleaved rotation convention.
    ///
    /// For pairs `(x0, x1), (x2, x3), ...`:
    ///   rotated = (x0*cos - x1*sin, x0*sin + x1*cos, x2*cos - x3*sin, ...)
    private func applyRotaryEmb(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let seqLen = x.dim(2)
        // Reshape cos/sin from [L, halfDim] to [1, 1, L, halfDim] for broadcasting
        let c = cos.reshaped(1, 1, seqLen, halfDim)
        let s = sin.reshaped(1, 1, seqLen, halfDim)

        // Split x into even and odd indices along last dim
        // x: [B, nHeads, L, dims] -> reshape to [B, nHeads, L, halfDim, 2]
        let shape = x.shape
        let xPairs = x.reshaped(shape[0], shape[1], shape[2], halfDim, 2)
        let x0 = xPairs[.ellipsis, 0]  // [B, nHeads, L, halfDim] - even indices
        let x1 = xPairs[.ellipsis, 1]  // [B, nHeads, L, halfDim] - odd indices

        // Apply rotation
        let out0 = x0 * c - x1 * s
        let out1 = x0 * s + x1 * c

        // Interleave back: [B, nHeads, L, halfDim, 2] -> [B, nHeads, L, dims]
        let stacked = MLX.stacked([out0, out1], axis: -1)  // [B, nHeads, L, halfDim, 2]
        return stacked.reshaped(shape)
    }

    /// Repeat-interleave along the last axis: [a, b, c] -> [a, a, b, b, c, c]
    private func repeatInterleave(_ x: MLXArray, count: Int) -> MLXArray {
        // x: [L, halfDim] -> expand to [L, halfDim, 1] -> broadcast to [L, halfDim, count]
        // -> reshape to [L, halfDim * count]
        let expanded = MLX.expandedDimensions(x, axis: -1)  // [L, halfDim, 1]
        let shape = expanded.shape
        let broadcastShape = [shape[0], shape[1], count]
        let repeated = MLX.broadcast(expanded, to: broadcastShape)
        return repeated.reshaped(x.dim(0), x.dim(-1) * count)
    }
}
