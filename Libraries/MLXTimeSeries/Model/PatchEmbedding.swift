import Foundation
import MLX
import MLXNN

/// Projects fixed-size patches of a time series into the transformer embedding space.
///
/// Input: `[B, V, T]` (batch, variables, time) → patches → `[B, V, nPatches, embedDim]`
public class PatchEmbedding: Module {

    @ModuleInfo(key: "projection") var projection: Linear

    let patchSize: Int
    let stride: Int

    public init(patchSize: Int, stride: Int, embedDim: Int) {
        self.patchSize = patchSize
        self.stride = stride
        self._projection.wrappedValue = Linear(patchSize, embedDim)
    }

    /// - Parameter x: Input tensor of shape `[B, V, T]` where T must be divisible by `patchSize`.
    /// - Returns: Embedded patches of shape `[B, V, nPatches, embedDim]`.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let V = x.dim(1)
        let T = x.dim(2)
        let nPatches = T / patchSize
        // [B, V, T] -> [B, V, nPatches, patchSize]
        let patches = x.reshaped(B, V, nPatches, patchSize)
        // [B, V, nPatches, patchSize] -> [B, V, nPatches, embedDim]
        return projection(patches)
    }
}
