package decoder

import "fmt"

// Backend abstracts the one hot primitive the decoder forward pass is
// bound by — the big weight matmuls. Norms, RoPE, softmax and elementwise
// ops are cheap and stay on the CPU even when a GPU matmul backend is in
// use, which avoids a host↔device round-trip per layer.
//
// The seam exists so a WebGPU backend (github.com/cogentcore/webgpu, already
// in go.mod) can replace the CPU one for the larger checkpoints without the
// forward pass knowing. See docs/gemma-decoder-plan.md §5.
type Backend interface {
	// Name identifies the backend ("cpu", "webgpu").
	Name() string
	// MatmulBT computes dst[M,N] = a[M,K] · b[N,K]ᵀ, the PyTorch [out,in]
	// weight layout the safetensors checkpoints already store (so no
	// transpose copy), matching encoder/'s matmulBT convention.
	MatmulBT(a, b, dst []float32, M, K, N int)
	// Close releases backend resources (GPU buffers, etc.). No-op on CPU.
	Close() error
}

// NewBackend returns the named backend. "cpu" is always available; "webgpu"
// falls back to CPU with a returned note if no adapter is present so the
// demo never hard-fails on a headless machine.
func NewBackend(name string) (Backend, error) {
	switch name {
	case "", "cpu":
		return &cpuBackend{}, nil
	case "webgpu":
		// M9: real WGSL matmul kernel + weight upload. Until then, fall
		// back to CPU so --backend=webgpu still runs.
		return &cpuBackend{}, fmt.Errorf("decoder: webgpu backend not implemented yet (M9), using cpu: %w", errNotImplemented)
	default:
		return nil, fmt.Errorf("decoder: unknown backend %q (have: cpu, webgpu)", name)
	}
}

// cpuBackend is a correct, naive matmul so the Backend interface is
// exercised end-to-end by the scaffold. M7 swaps the body for the shared
// SIMD/row-parallel linalg lifted out of encoder/ (plan §1) — a one-line
// change here, the interface and all callers stay put.
type cpuBackend struct{}

func (*cpuBackend) Name() string { return "cpu" }

func (*cpuBackend) MatmulBT(a, b, dst []float32, M, K, N int) {
	// dst[i,j] = Σ_k a[i,k] * b[j,k]
	for i := 0; i < M; i++ {
		arow := a[i*K : i*K+K]
		drow := dst[i*N : i*N+N]
		for j := 0; j < N; j++ {
			brow := b[j*K : j*K+K]
			var s float32
			for k := 0; k < K; k++ {
				s += arow[k] * brow[k]
			}
			drow[j] = s
		}
	}
}

func (*cpuBackend) Close() error { return nil }
