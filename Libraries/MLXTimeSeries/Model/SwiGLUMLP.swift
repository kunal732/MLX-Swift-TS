import Foundation
import MLX
import MLXNN

/// SwiGLU feed-forward network matching PyTorch's Sequential(Linear, SwiGLU, Linear, Dropout).
///
/// Weight keys use numeric indices to match PyTorch's nn.Sequential:
/// - `0.weight/bias` → gate-up projection (embed_dim → 2 * mlp_hidden_dim)
/// - `2.weight/bias` → down projection (mlp_hidden_dim → embed_dim)
///
/// Indices 1 (SwiGLU activation) and 3 (Dropout) have no parameters.
public class SwiGLUMLP: Module {

    @ModuleInfo(key: "gate_up") var gateUp: Linear
    @ModuleInfo(key: "down") var down: Linear

    public init(embedDim: Int, mlpHiddenDim: Int, bias: Bool = true) {
        // gate_up projects to 2x hidden dim (split into gate and value)
        self._gateUp.wrappedValue = Linear(embedDim, 2 * mlpHiddenDim, bias: bias)
        // down projects from hidden dim back to embed dim
        self._down.wrappedValue = Linear(mlpHiddenDim, embedDim, bias: bias)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = gateUp(x)  // [..., 2 * mlpHiddenDim]
        let chunks = h.split(parts: 2, axis: -1)
        let gate = chunks[0]
        let value = chunks[1]
        return down(silu(gate) * value)
    }
}
