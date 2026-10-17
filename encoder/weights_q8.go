package encoder

import (
	"fmt"

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
// WordEmb is read from the checkpoint's mapping, not copied: it is only
// ever gathered by row, so a forward faults in the rows it uses and the
// table costs no heap. The weights therefore hold the checkpoint file open
// until Close.
//
// Memory footprint: ~110 MB of heap vs Weights' ~547 MB (4× reduction
// for the linear-projection bulk), plus the mapped pages of the
// word-embedding rows a workload touches.
type WeightsQ8 struct {
	// be is the compute backend for f32 matmuls; nil = the pure-Go path (default).
	be Backend

	Cfg          Config
	WordEmb      []float32 // [VocabSize, HiddenDim] — stays f32 (embedding lookup, not matmul)
	TokenTypeEmb []float32 // [TypeVocabSize, HiddenDim] — stays f32
	EmbLN_W      []float32 // [HiddenDim] — LN bias, parity-sensitive
	EmbLN_B      []float32

	Layers []LayerWeightsQ8

	// st is the checkpoint WordEmb is a view into (nil when the checkpoint was heap-read or widened). Held
	// until Close; see the type doc.
	st *embed.SafetensorsFile

	// ropeCache holds the rotary table across forwards instead of rebuilding it
	// per call (prefix-property cache; see Weights.ropeCache). Zero value ready.
	ropeCache ropeCache
}

// HeadDim convenience (mirrors Config.HeadDim).
func (w *WeightsQ8) HeadDim() int { return w.Cfg.HeadDim() }

// LoadWeightsQ8 reads the f32 checkpoint via the mmap path (M8) and
// quantizes the 5 big per-layer linear matrices to int8 at load time.
// Calibration is per-row symmetric (max(|row|) / 127).
//
// After this returns, the quantized projections are int8 copies on the Go
// heap and their f32 pages are released; the mapping itself stays open,
// because WordEmb is read from it (see WeightsQ8). Close releases it.
//
// Steady state was never the problem; the PEAK was. Quantizing reads every f32
// weight, so waiting until the end to Close left the entire checkpoint resident
// at once — measured at 727.6 MiB peak RSS to produce a 199.5 MiB model
// (TestQ8LoadFootprint). Each tensor is therefore released as soon as it has
// been quantized, which bounds the resident f32 set at roughly one layer instead
// of all of them. The pages are clean and file-backed, so this is advisory only:
// it can cost a re-fault, never wrong data.
func LoadWeightsQ8(dir string) (*WeightsQ8, error) {
	// Load the f32 weights first (via the mmap path).
	w, err := LoadWeights(dir)
	if err != nil {
		return nil, err
	}
	// The q8 forward quantizes the SwiGLU trio directly; a mixture-of-experts or
	// dense-GELU checkpoint has no fc11/fc12 to quantize. Refuse rather than
	// quantize nil tensors into a plausible-looking wrong model.
	if w.hasMoE() || !w.Cfg.gatedMLP() {
		w.closeFile()
		return nil, fmt.Errorf("encoder: q8 unsupported for this checkpoint (mixture-of-experts / non-gated MLP); use the f32 path")
	}
	// The Q8 attention path (selfAttentionQ8/…Batched) has no bias term and
	// LayerWeightsQ8 carries no bias fields, so a qkv_proj_bias checkpoint would
	// silently drop its attention biases on every Q8 path. Refuse rather than
	// mis-score (audit #3); the f32 path applies them correctly.
	if w.Cfg.QKVProjBias {
		w.closeFile()
		return nil, fmt.Errorf("encoder: q8 unsupported for a checkpoint with qkv_proj_bias (the Q8 attention path has no bias term); use the f32 path")
	}
	// Build the q8 bundle by quantizing each big projection. WordEmb stays a
	// view into the mapping (89 MiB on CodeRankEmbed that used to be cloned
	// onto the heap); the small embedding tensors are copied.
	q := &WeightsQ8{
		Cfg:          w.Cfg,
		WordEmb:      w.WordEmb,
		TokenTypeEmb: cloneFloat32(w.TokenTypeEmb),
		EmbLN_W:      cloneFloat32(w.EmbLN_W),
		EmbLN_B:      cloneFloat32(w.EmbLN_B),
		Layers:       make([]LayerWeightsQ8, len(w.Layers)),
		st:           w.st,
	}
	// The copied embedding tensors' mapped pages are dead weight from here.
	_ = w.releasePages( //nolint:errcheck // best-effort page drop; the heap copy is authoritative
		"embeddings.token_type_embeddings.weight",
		"emb_ln.weight", "emb_ln.bias",
	)
	cfg := &w.Cfg
	for i, l := range w.Layers {
		lq := LayerWeightsQ8{
			Norm1W: cloneFloat32(l.Norm1W),
			Norm1B: cloneFloat32(l.Norm1B),
			Norm2W: cloneFloat32(l.Norm2W),
			Norm2B: cloneFloat32(l.Norm2B),

			// Quantize big projections (rows = output dim, cols = input dim) to
			// weight-only Q8. linalg.QuantizeInt8 is bit-identical to the encoder's
			// quantizeRowsInt8 (same per-row symmetric max/127 round+clamp), so the
			// stored codes/scales — and the forward — are unchanged.
			Wqkv:    linalg.QuantizeInt8(l.Wqkv, 3*cfg.HiddenDim, cfg.HiddenDim, false),
			OutProj: linalg.QuantizeInt8(l.OutProj, cfg.HiddenDim, cfg.HiddenDim, false),
			Fc11:    linalg.QuantizeInt8(l.Fc11, cfg.IntermediateDim, cfg.HiddenDim, false),
			Fc12:    linalg.QuantizeInt8(l.Fc12, cfg.IntermediateDim, cfg.HiddenDim, false),
			Fc2:     linalg.QuantizeInt8(l.Fc2, cfg.HiddenDim, cfg.IntermediateDim, false)}
		q.Layers[i] = lq
		// Those five f32 tensors are now int8 + scales on the heap, and nothing
		// reads their mapped pages again. Drop them before touching layer i+1 so
		// only one layer's f32 side is resident at a time.
		_ = w.releasePages(layerQ8TensorNames(i)...) //nolint:errcheck // best-effort page drop; the heap copy is authoritative
	}
	return q, nil
}

// Close releases the checkpoint mapping WordEmb is read from. The weights
// must not be used afterwards. Calling Close again does nothing.
func (w *WeightsQ8) Close() error {
	if w == nil {
		return nil
	}
	st := w.st
	w.st, w.WordEmb = nil, nil
	if st == nil {
		return nil
	}
	return st.Close()
}

// layerQ8TensorNames lists the five tensors LoadWeightsQ8 quantizes for layer i.
// The names mirror buildWeightsFromSafetensors' gatedMLP branch — LoadWeightsQ8
// refuses any checkpoint that doesn't take it, so all five are guaranteed loaded.
// A drift between the two lists is caught by TestLayerQ8TensorNamesResolve rather
// than silently degrading into a no-op release.
func layerQ8TensorNames(i int) []string {
	pfx := fmt.Sprintf("encoder.layers.%d.", i)
	return []string{
		pfx + "attn.Wqkv.weight",
		pfx + "attn.out_proj.weight",
		pfx + "mlp.fc11.weight",
		pfx + "mlp.fc12.weight",
		pfx + "mlp.fc2.weight",
	}
}

// releasePages drops the mapped pages of tensors that have been consumed. It is
// advisory: an error means either a heap-backed file (nothing to release) or a
// name that doesn't resolve, and neither is a reason to fail a load that has
// otherwise succeeded. Tests assert the names resolve; production ignores it.
func (w *Weights) releasePages(names ...string) error {
	if w.st == nil {
		return nil
	}
	return w.st.ReleaseTensors(names...)
}

// closeFile releases w's checkpoint mapping on a load that is refusing it.
func (w *Weights) closeFile() {
	if w.st != nil {
		_ = w.st.Close() //nolint:errcheck // a refused load: the refusal is the error to report
	}
}

func cloneFloat32(src []float32) []float32 {
	dst := make([]float32, len(src))
	copy(dst, src)
	return dst
}
