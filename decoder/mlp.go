package decoder

import "fmt"

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
//
// SCAFFOLD: returns errNotImplemented; the math above is the whole M3 body.
func geGLU(h []float32, lw *LayerWeights, cfg *Config, be Backend) ([]float32, error) {
	_ = h
	_ = lw
	_ = cfg
	_ = be
	return nil, fmt.Errorf("decoder.geGLU: %w [M3]", errNotImplemented)
}
