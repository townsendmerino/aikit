package encoder

import (
	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
)

// LayerWeightsQ8 is the M8 int8-quantized per-layer bundle. Only the
// big linear projections are quantized — the LayerNorm parameters
// (Norm1W/B, Norm2W/B) are tiny ([D]=768 each) and stay f32 (no
// memory saving, plus LN is parity-sensitive: float-noise from
// quantizing γ/β here would compound across the 12 layers).
//
// Each big matrix is a weight-only Q8 linalg.WeightMat (per-row int8 +
// per-row f32 scales; reconstruction f32[i,j] ≈ int8[i*K+j] * scale[i]).
// The forward still drives the encoder's own baked-scale matmulBTQ8Into
// over the int8 codes/scales pulled via WeightMat.Int8() — storage is
// shared, the kernel stays the encoder's (it is numerically distinct
// from linalg.MatmulBTQ8: large-M dequant-once-into-scratch).
type LayerWeightsQ8 struct {
	Wqkv    linalg.WeightMat // [3*HiddenDim, HiddenDim]
	OutProj linalg.WeightMat // [HiddenDim, HiddenDim]
	Fc11    linalg.WeightMat // [IntermediateDim, HiddenDim]
	Fc12    linalg.WeightMat // [IntermediateDim, HiddenDim]
	Fc2     linalg.WeightMat // [HiddenDim, IntermediateDim]

	// LN weights stay f32 — small (768 each) so no memory saving, and
	// LN is the parity-sensitive op the f32 forward already uses f64
	// accumulators on; quantizing γ/β here introduces compounding noise
	// across the 12 layers for no benefit.
	Norm1W, Norm1B []float32
	Norm2W, Norm2B []float32
}

// WeightsQ8 mirrors Weights but with the 5 big linear projections per
// layer quantized to int8 + per-row scales. Embeddings (WordEmb) and
// emb_ln stay f32 — embedding rows feed an integer lookup, not a
// matmul, so quantizing them has no perf benefit; emb_ln is the same
// "tiny + parity-sensitive" exception as the per-layer LNs.
//
// Memory footprint: ~140 MB total vs Weights' ~547 MB (4× reduction
// for the linear-projection bulk; embeddings still cost ~92 MB).
type WeightsQ8 struct {
	Cfg          Config
	WordEmb      []float32 // [VocabSize, HiddenDim] — stays f32 (embedding lookup, not matmul)
	TokenTypeEmb []float32 // [TypeVocabSize, HiddenDim] — stays f32
	EmbLN_W      []float32 // [HiddenDim] — LN bias, parity-sensitive
	EmbLN_B      []float32

	Layers []LayerWeightsQ8

	// Retained for the underlying f32 weights' lifetime (slices alias
	// into the mmap region or heap buffer — same contract as Weights).
	st *embed.SafetensorsFile
}

// HeadDim convenience (mirrors Config.HeadDim).
func (w *WeightsQ8) HeadDim() int { return w.Cfg.HeadDim() }

// LoadWeightsQ8 reads the f32 checkpoint via the mmap path (M8) and
// quantizes the 5 big per-layer linear matrices to int8 at load time.
// Calibration is per-row symmetric (max(|row|) / 127).
//
// After this returns, the original f32 weight bytes are released via
// Close() on the safetensors handle — we hold int8 copies on the Go
// heap instead. The model footprint drops from ~547 MB (heap) +
// 547 MB (mmap shadow) to ~140 MB int8 + the small f32 tail.
func LoadWeightsQ8(dir string) (*WeightsQ8, error) {
	// Load the f32 weights first (via the mmap path).
	w, err := LoadWeights(dir)
	if err != nil {
		return nil, err
	}
	// Build the q8 bundle by quantizing each big projection.
	q := &WeightsQ8{
		Cfg:          w.Cfg,
		WordEmb:      cloneFloat32(w.WordEmb),
		TokenTypeEmb: cloneFloat32(w.TokenTypeEmb),
		EmbLN_W:      cloneFloat32(w.EmbLN_W),
		EmbLN_B:      cloneFloat32(w.EmbLN_B),
		Layers:       make([]LayerWeightsQ8, len(w.Layers)),
	}
	cfg := &w.Cfg
	for i, l := range w.Layers {
		lq := LayerWeightsQ8{
			Norm1W: cloneFloat32(l.Norm1W),
			Norm1B: cloneFloat32(l.Norm1B),
			Norm2W: cloneFloat32(l.Norm2W),
			Norm2B: cloneFloat32(l.Norm2B),
		}
		// Quantize big projections (rows = output dim, cols = input dim) to
		// weight-only Q8. linalg.QuantizeInt8 is bit-identical to the encoder's
		// quantizeRowsInt8 (same per-row symmetric max/127 round+clamp), so the
		// stored codes/scales — and the forward — are unchanged.
		lq.Wqkv = linalg.QuantizeInt8(l.Wqkv, 3*cfg.HiddenDim, cfg.HiddenDim, false)
		lq.OutProj = linalg.QuantizeInt8(l.OutProj, cfg.HiddenDim, cfg.HiddenDim, false)
		lq.Fc11 = linalg.QuantizeInt8(l.Fc11, cfg.IntermediateDim, cfg.HiddenDim, false)
		lq.Fc12 = linalg.QuantizeInt8(l.Fc12, cfg.IntermediateDim, cfg.HiddenDim, false)
		lq.Fc2 = linalg.QuantizeInt8(l.Fc2, cfg.HiddenDim, cfg.IntermediateDim, false)
		q.Layers[i] = lq
	}
	// Release the underlying mmap (the f32 weights are no longer needed —
	// we have int8 copies on the heap now).
	if w.st != nil {
		_ = w.st.Close()
	}
	return q, nil
}

func cloneFloat32(src []float32) []float32 {
	dst := make([]float32, len(src))
	copy(dst, src)
	return dst
}
