package decoder

import (
	"fmt"
	"io/fs"
	"os"
	"path"
	"path/filepath"

	"github.com/townsendmerino/aikit/embed"
)

// LayerWeights bundles one decoder block's tensors. Matrices follow
// PyTorch's [out, in] row-major layout so MatmulBT is A·Bᵀ with no
// transpose copy, matching encoder.LayerWeights.
//
// Gemma 3 specifics reflected here: separate gate/up projections (GeGLU),
// pre- AND post-norms on both the attention and MLP sub-blocks, and
// optional QK-norm weights.
type LayerWeights struct {
	// Attention.
	QProj        []float32 // [NumHeads*HeadDim, HiddenDim]
	KProj        []float32 // [NumKVHeads*HeadDim, HiddenDim]
	VProj        []float32 // [NumKVHeads*HeadDim, HiddenDim]
	OProj        []float32 // [HiddenDim, NumHeads*HeadDim]
	QNorm        []float32 // [HeadDim] QK-norm on queries (Gemma 3)
	KNorm        []float32 // [HeadDim] QK-norm on keys (Gemma 3)
	PreAttnNorm  []float32 // [HiddenDim] input RMSNorm
	PostAttnNorm []float32 // [HiddenDim] RMSNorm after attention, before residual add

	// MLP (GeGLU).
	GateProj    []float32 // [IntermediateDim, HiddenDim]
	UpProj      []float32 // [IntermediateDim, HiddenDim]
	DownProj    []float32 // [HiddenDim, IntermediateDim]
	PreMLPNorm  []float32 // [HiddenDim]
	PostMLPNorm []float32 // [HiddenDim]
}

// Weights is the immutable per-checkpoint bundle. Embeddings are TIED:
// Embed doubles as the LM head (logits = h · Embedᵀ), so there is no
// separate output projection tensor.
type Weights struct {
	Cfg       Config
	Embed     []float32 // [VocabSize, HiddenDim] — input embedding AND tied LM head
	FinalNorm []float32 // [HiddenDim] RMSNorm before the LM head
	Layers    []LayerWeights

	st *embed.SafetensorsFile // retained so alias-backed slices stay valid
}

// LoadWeights reads config.json + model.safetensors from a real on-disk
// directory (the HF snapshot layout). The .safetensors blob is mmapped
// (not heap-copied) so the 270M's ~340 MB bf16 checkpoint stays in the OS
// page cache — same M8 path as encoder.LoadWeights.
//
// NOTE: this widens bf16/f16 weights to f32 on load (BFloat16sToF32 /
// Float16sToF32 allocate), which roughly doubles resident RAM vs keeping
// the tensors bf16. That's the M1 correctness-first choice; the
// half-the-RAM route is per-tile widen inside matmul.
// TODO(M8): bf16-resident matmul tiling (gemma-decoder-plan §2/§8) to
// drop the widen-on-load 2× memory cost for the 1B+ checkpoints.
//
// Use LoadWeightsFromFS for fs.FS-backed (MapFS, embed.FS) paths — that
// route stays heap-backed because fs.FS doesn't expose a file descriptor.
func LoadWeights(dir string) (*Weights, error) {
	cfg, err := loadConfig(os.DirFS(dir), "config.json")
	if err != nil {
		return nil, err
	}
	if err := cfg.ValidateAssumptions(); err != nil {
		return nil, err
	}
	st, err := embed.OpenSafetensorsMmap(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, fmt.Errorf("decoder: open safetensors: %w", err)
	}
	return buildWeightsFromSafetensors(cfg, st)
}

// LoadWeightsFromFS mirrors encoder.LoadWeightsFromFS: reads config.json +
// model.safetensors from fsys/dir, validates every tensor's shape against
// Cfg, and returns the populated bundle. Heap-backed (fs.ReadFile); use
// LoadWeights for the mmap path on a real directory.
func LoadWeightsFromFS(fsys fs.FS, dir string) (*Weights, error) {
	cfg, err := loadConfig(fsys, path.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	if err := cfg.ValidateAssumptions(); err != nil {
		return nil, err
	}
	st, err := embed.OpenSafetensorsFromFS(fsys, path.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, fmt.Errorf("decoder: open safetensors: %w", err)
	}
	return buildWeightsFromSafetensors(cfg, st)
}

// buildWeightsFromSafetensors fills a *Weights from an already-opened
// SafetensorsFile, shape-validating every tensor in gemma3TensorSchema
// against Cfg. Factored out so the heap (fs.FS) and mmap paths share one
// tensor-name + shape contract — a schema change is one edit, not two.
// Mirrors encoder.buildWeightsFromSafetensors.
func buildWeightsFromSafetensors(cfg *Config, st *embed.SafetensorsFile) (*Weights, error) {
	s := gemma3TensorSchema
	hd := cfg.HiddenDim
	qDim := cfg.NumHeads * cfg.HeadDim    // query projection rows (270M: 1024)
	kvDim := cfg.NumKVHeads * cfg.HeadDim // key/value projection rows (270M: 256)

	w := &Weights{Cfg: *cfg, st: st, Layers: make([]LayerWeights, cfg.NumLayers)}
	var err error

	// Tied embedding table (also the LM head) + final norm.
	if w.Embed, err = loadF32(st, s.Embed, []int{cfg.VocabSize, hd}); err != nil {
		return nil, err
	}
	if w.FinalNorm, err = loadF32(st, s.FinalNorm, []int{hd}); err != nil {
		return nil, err
	}

	for i := 0; i < cfg.NumLayers; i++ {
		l := &w.Layers[i]
		// Attention projections ([out, in] row-major, GQA: K/V are narrower).
		if l.QProj, err = loadF32(st, tensorName(i, s.QProj), []int{qDim, hd}); err != nil {
			return nil, err
		}
		if l.KProj, err = loadF32(st, tensorName(i, s.KProj), []int{kvDim, hd}); err != nil {
			return nil, err
		}
		if l.VProj, err = loadF32(st, tensorName(i, s.VProj), []int{kvDim, hd}); err != nil {
			return nil, err
		}
		if l.OProj, err = loadF32(st, tensorName(i, s.OProj), []int{hd, qDim}); err != nil {
			return nil, err
		}
		// QK-norm (Gemma 3): RMSNorm over the per-head dimension.
		if l.QNorm, err = loadF32(st, tensorName(i, s.QNorm), []int{cfg.HeadDim}); err != nil {
			return nil, err
		}
		if l.KNorm, err = loadF32(st, tensorName(i, s.KNorm), []int{cfg.HeadDim}); err != nil {
			return nil, err
		}
		// Pre/post norms around attention and MLP (all [HiddenDim]).
		if l.PreAttnNorm, err = loadF32(st, tensorName(i, s.PreAttnNorm), []int{hd}); err != nil {
			return nil, err
		}
		if l.PostAttnNorm, err = loadF32(st, tensorName(i, s.PostAttnNorm), []int{hd}); err != nil {
			return nil, err
		}
		// GeGLU MLP.
		if l.GateProj, err = loadF32(st, tensorName(i, s.GateProj), []int{cfg.IntermediateDim, hd}); err != nil {
			return nil, err
		}
		if l.UpProj, err = loadF32(st, tensorName(i, s.UpProj), []int{cfg.IntermediateDim, hd}); err != nil {
			return nil, err
		}
		if l.DownProj, err = loadF32(st, tensorName(i, s.DownProj), []int{hd, cfg.IntermediateDim}); err != nil {
			return nil, err
		}
		if l.PreMLPNorm, err = loadF32(st, tensorName(i, s.PreMLPNorm), []int{hd}); err != nil {
			return nil, err
		}
		if l.PostMLPNorm, err = loadF32(st, tensorName(i, s.PostMLPNorm), []int{hd}); err != nil {
			return nil, err
		}
	}
	return w, nil
}

// loadF32 fetches a tensor, shape-validates it against want, and decodes it
// to []float32 dispatching on DType: F32 is a zero-copy view; BF16/F16 are
// widened (allocating). One clear "shape %v != want %v" error otherwise.
// Mirrors encoder.loadF32, extended with the bf16/f16 dispatch.
func loadF32(st *embed.SafetensorsFile, name string, want []int) ([]float32, error) {
	t, err := st.Tensor(name)
	if err != nil {
		return nil, fmt.Errorf("decoder: tensor %q: %w", name, err)
	}
	if !shapeEqual(t.Shape, want) {
		return nil, fmt.Errorf("decoder: tensor %q shape %v != want %v", name, t.Shape, want)
	}
	var data []float32
	switch t.DType {
	case "F32":
		data, err = t.Float32s()
	case "BF16":
		data, err = t.BFloat16sToF32()
	case "F16":
		data, err = t.Float16sToF32()
	default:
		return nil, fmt.Errorf("decoder: tensor %q unsupported dtype %q (want F32/BF16/F16)", name, t.DType)
	}
	if err != nil {
		return nil, fmt.Errorf("decoder: tensor %q decode: %w", name, err)
	}
	return data, nil
}

func shapeEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// tensorName returns the HF safetensors key for a per-layer tensor. Kept in
// one place so the M1 loader and any future schema bump touch one function.
func tensorName(layer int, suffix string) string {
	return fmt.Sprintf("model.layers.%d.%s", layer, suffix)
}

// gemma3TensorSchema documents the keys the M1 loader will read. Referenced
// by name here (not used yet) so the schema is version-controlled alongside
// the struct it fills.
var gemma3TensorSchema = struct {
	Embed     string
	FinalNorm string
	// per-layer suffixes passed to tensorName(layer, suffix)
	QProj, KProj, VProj, OProj string
	QNorm, KNorm               string
	PreAttnNorm, PostAttnNorm  string
	GateProj, UpProj, DownProj string
	PreMLPNorm, PostMLPNorm    string
}{
	Embed:        "model.embed_tokens.weight",
	FinalNorm:    "model.norm.weight",
	QProj:        "self_attn.q_proj.weight",
	KProj:        "self_attn.k_proj.weight",
	VProj:        "self_attn.v_proj.weight",
	OProj:        "self_attn.o_proj.weight",
	QNorm:        "self_attn.q_norm.weight",
	KNorm:        "self_attn.k_norm.weight",
	PreAttnNorm:  "input_layernorm.weight",
	PostAttnNorm: "post_attention_layernorm.weight",
	GateProj:     "mlp.gate_proj.weight",
	UpProj:       "mlp.up_proj.weight",
	DownProj:     "mlp.down_proj.weight",
	PreMLPNorm:   "pre_feedforward_layernorm.weight",
	PostMLPNorm:  "post_feedforward_layernorm.weight",
}
