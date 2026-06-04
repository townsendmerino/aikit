package encoder

import (
	"fmt"
	"sync"
)

// Backend abstracts the one hot primitive the encoder forward pass is bound
// by — the big weight matmuls (Wqkv, OutProj, the MLP projections). Norms,
// RoPE, softmax and elementwise ops are cheap and stay on the CPU even when a
// GPU matmul backend is in use, which avoids a host↔device round-trip per
// layer.
//
// The default "cpu" backend is always registered (pure-Go SIMD matmul, no
// cgo). A WebGPU backend lives in the opt-in github.com/townsendmerino/goinfer/gpu
// module: importing it under `-tags gpu` calls RegisterBackend("webgpu", …) on
// init, so the encoder gains GPU acceleration WITHOUT ever importing
// github.com/cogentcore/webgpu (cgo) into aikit's dependency graph.
type Backend interface {
	// Name identifies the backend ("cpu", "webgpu").
	Name() string
	// MatmulBT computes dst[M,N] = a[M,K] · b[N,K]ᵀ — the PyTorch [out,in]
	// weight layout the safetensors checkpoints already store (so no
	// transpose copy), matching the encoder's matmulBT convention.
	MatmulBT(a, b, dst []float32, M, K, N int)
	// Close releases backend resources (GPU buffers, etc.). No-op on CPU.
	Close() error
}

var (
	backendMu       sync.RWMutex
	backendRegistry = map[string]func() (Backend, error){}
)

// RegisterBackend registers a named Backend factory. The goinfer/gpu module
// calls this from init() (under `-tags gpu`) to make "webgpu" available
// without aikit importing the cgo WebGPU implementation. Safe for concurrent
// use; a later registration of the same name replaces the earlier one.
func RegisterBackend(name string, factory func() (Backend, error)) {
	backendMu.Lock()
	defer backendMu.Unlock()
	backendRegistry[name] = factory
}

// NewBackend returns the named backend. "" and "cpu" always resolve to the
// pure-Go CPU backend. Other names resolve through the registry; "webgpu"
// falls back to CPU with an explanatory error (rather than hard-failing) when
// goinfer/gpu has not been imported, so a `--backend webgpu` flag still runs
// on a build without the GPU module.
func NewBackend(name string) (Backend, error) {
	switch name {
	case "", "cpu":
		return &cpuBackend{}, nil
	}
	backendMu.RLock()
	factory := backendRegistry[name]
	backendMu.RUnlock()
	if factory != nil {
		return factory()
	}
	if name == "webgpu" {
		return &cpuBackend{}, fmt.Errorf("encoder: webgpu backend not registered; import github.com/townsendmerino/goinfer/gpu and build `-tags gpu`; using cpu")
	}
	return nil, fmt.Errorf("encoder: unknown backend %q (have: cpu, webgpu)", name)
}

// cpuBackend dispatches the hot matmul to the encoder's pure-Go SIMD path
// (matmulBTInto: AVX2/NEON dot kernels parallelized across output columns).
type cpuBackend struct{}

func (*cpuBackend) Name() string { return "cpu" }

func (*cpuBackend) MatmulBT(a, b, dst []float32, M, K, N int) {
	matmulBTInto(a, b, dst, M, K, N)
}

func (*cpuBackend) Close() error { return nil }
