//go:build gpu

package decoder

import (
	"fmt"
	"sync"

	"github.com/townsendmerino/aikit/encoder/gpu"
	"github.com/townsendmerino/aikit/internal/linalg"
)

// webgpuBackend runs MatmulBT on a WebGPU adapter (Vulkan / Metal / D3D12) via
// the shared encoder/gpu foundation. Built only under -tags gpu.
//
// Resident weights (M9): a decoder weight matrix is constant across every
// token, so the first MatmulBT for a given weight uploads it to a GPU storage
// buffer and caches the handle (keyed by the slice's backing pointer); every
// later token only uploads the tiny M=1 activation and reads the result back.
// This removes the catastrophic per-token re-upload of the weights (the LM
// head alone is the ~671 MB embedding). On any per-call GPU error it falls
// back to the CPU matmul, so results are always correct.
//
// Still naive beyond that: each matmul is its own synchronous dispatch +
// readback, so decode is latency-bound on per-matmul round-trips. Keeping the
// activations resident on-device across a layer's matmuls (and porting
// norms/rope/softmax to WGSL) is the remaining work for a GPU-fast decoder.
type webgpuBackend struct {
	ctx  *gpu.Context
	name string

	mu        sync.Mutex // encoder/gpu.Context is not goroutine-safe
	resident  map[*float32]*gpu.ResidentMatrix
	fallbacks int
}

func newWebGPUBackend() (Backend, error) {
	ctx, err := gpu.New()
	if err != nil {
		// No adapter (headless / no driver): fall back to CPU with a note.
		return &cpuBackend{}, fmt.Errorf("decoder: no WebGPU adapter (%v); using cpu", err)
	}
	return &webgpuBackend{
		ctx:      ctx,
		name:     "webgpu:" + ctx.Backend(),
		resident: make(map[*float32]*gpu.ResidentMatrix),
	}, nil
}

func (b *webgpuBackend) Name() string { return b.name }

// MatmulBT computes dst[M,N] = a · bMatᵀ, with bMat (the weight) uploaded once
// and reused. b's identity is its backing pointer — the decoder passes the same
// weight slice every token.
func (b *webgpuBackend) MatmulBT(a, bMat, dst []float32, M, K, N int) {
	b.mu.Lock()
	out, err := b.matmulLocked(a, bMat, M, K, N)
	b.mu.Unlock()
	if err != nil {
		b.fallbacks++
		linalg.MatmulBT(a, bMat, dst, M, K, N) // correctness over speed
		return
	}
	copy(dst, out)
}

// matmulLocked uploads (or reuses) the resident weight and dispatches. Caller
// holds b.mu.
func (b *webgpuBackend) matmulLocked(a, bMat []float32, M, K, N int) ([]float32, error) {
	if len(bMat) == 0 {
		return nil, fmt.Errorf("gpu: empty weight")
	}
	key := &bMat[0]
	rm := b.resident[key]
	if rm == nil {
		var err error
		if rm, err = b.ctx.UploadMatrix(bMat, N, K); err != nil {
			return nil, err
		}
		b.resident[key] = rm
	}
	return b.ctx.MatmulBTResident(a, rm, M)
}

func (b *webgpuBackend) Close() error {
	b.mu.Lock()
	for _, rm := range b.resident {
		rm.Release()
	}
	b.resident = nil
	b.mu.Unlock()
	b.ctx.Close()
	return nil
}
