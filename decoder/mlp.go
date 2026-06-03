package decoder

// geGLU runs one block's GeGLU MLP for the current position and returns the
// output (caller applies the post-MLP norm + residual add).
//
//	gate = GateProj·h            // [IntermediateDim]
//	up   = UpProj·h              // [IntermediateDim]
//	mid  = geluTanh(gate) ⊙ up   // [IntermediateDim]   (gate is the activated branch)
//	out  = DownProj·mid          // [HiddenDim]
//
// geluTanh (rmsnorm.go) is the "gelu_pytorch_tanh" activation Gemma uses.
// No biases on any projection.
func geGLU(h []float32, lw *LayerWeights, cfg *Config, be Backend) ([]float32, error) {
	inter, hidden := cfg.IntermediateDim, cfg.HiddenDim
	gate := make([]float32, inter)
	up := make([]float32, inter)
	lw.GateProj.matmul(be, h, gate, 1) // [1,inter] = h · GateProjᵀ
	lw.UpProj.matmul(be, h, up, 1)
	for i := range gate {
		gate[i] = geluTanh(gate[i]) * up[i]
	}
	out := make([]float32, hidden)
	lw.DownProj.matmul(be, gate, out, 1) // [1,hidden] = mid · DownProjᵀ
	return out, nil
}
