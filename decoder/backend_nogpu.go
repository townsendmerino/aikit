//go:build !gpu

package decoder

import "fmt"

// newWebGPUBackend (default build): the WebGPU backend is compiled only under
// -tags gpu (it cgo-links wgpu-native). Without the tag, fall back to CPU with
// a note so `--backend webgpu` still runs — matching the encoder/gpu pattern.
func newWebGPUBackend() (Backend, error) {
	return &cpuBackend{}, fmt.Errorf("decoder: webgpu backend needs `-tags gpu`; using cpu")
}
