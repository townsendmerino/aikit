package decoder

import "fmt"

// gatedMLP runs one block's gated MLP for the current position and returns the
// output (caller applies the post-MLP norm + residual add). The gate/up/down
// structure is shared by GeGLU (Gemma) and SwiGLU (Llama/Mistral/Qwen); only
// the gate activation differs (Architecture.Act).
//
//	gate = GateProj·h            // [IntermediateDim]
//	up   = UpProj·h              // [IntermediateDim]
//	mid  = act(gate) ⊙ up        // [IntermediateDim]
//	out  = DownProj·mid          // [HiddenDim]
//
// No biases on any projection.
func gatedMLP(h []float32, lw *LayerWeights, arch *Architecture, be Backend) ([]float32, error) {
	inter, hidden := arch.IntermediateDim, arch.HiddenDim
	gate := make([]float32, inter)
	up := make([]float32, inter)
	lw.GateProj.matmul(be, h, gate, 1) // [1,inter] = h · GateProjᵀ
	lw.UpProj.matmul(be, h, up, 1)
	switch arch.Act {
	case ActGeluTanh:
		for i := range gate {
			gate[i] = geluTanh(gate[i]) * up[i]
		}
	case ActSiLU:
		for i := range gate {
			gate[i] = silu(gate[i]) * up[i]
		}
	default:
		return nil, fmt.Errorf("decoder: unsupported activation %d (have GeGLU/SwiGLU)", arch.Act)
	}
	out := make([]float32, hidden)
	lw.DownProj.matmul(be, gate, out, 1) // [1,hidden] = mid · DownProjᵀ
	return out, nil
}
