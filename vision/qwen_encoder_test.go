package vision

import (
	"encoding/json"
	"os"
	"testing"
)

// TestQwenVisionEncoder_parity gates the pure-Go Qwen2.5-VL vision tower against the
// HF Qwen2_5_VisionTransformerPretrainedModel golden (scripts/pin_qwen25vl_vision.py):
// same tiny checkpoint, same pixel_values + grid_thw, two stage-isolated cosines —
// the ViT pre-merge hidden and the merged image features — both ≥ 0.9999 (fp32).
// Asset-gated on testdata/qwen25vl-vision-tiny + the golden.
func TestQwenVisionEncoder_parity(t *testing.T) {
	const ckpt = "../testdata/qwen25vl-vision-tiny"
	if _, err := os.Stat(ckpt); err != nil {
		t.Skipf("no qwen25vl-vision-tiny checkpoint (%v); run scripts/pin_qwen25vl_vision.py", err)
	}
	raw, err := os.ReadFile("../testdata/qwen25vl_vision_golden.json")
	if err != nil {
		t.Skipf("no golden (%v)", err)
	}
	var g struct {
		GridTHW       [][3]int  `json:"grid_thw"`
		NPatches      int       `json:"n_patches"`
		NMerged       int       `json:"n_merged"`
		PixelValues   []float32 `json:"pixel_values"`
		VitHidden     []float32 `json:"vit_hidden"`     // [n_patches, hidden], WINDOW order (HF)
		ImageFeatures []float32 `json:"image_features"` // [n_merged, out_hidden], original order
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}

	enc, err := LoadQwenVisionEncoder(ckpt, false) // f32: bit-exact path
	if err != nil {
		t.Fatalf("LoadQwenVisionEncoder: %v", err)
	}

	// Stage 1: ViT pre-merge hidden. ForwardViT returns ORIGINAL patch order; HF's
	// last_hidden_state is WINDOW order, so de-window the golden to match.
	hidden := enc.Cfg.HiddenSize
	mergeUnit := enc.Cfg.SpatialMergeSize * enc.Cfg.SpatialMergeSize
	winIdx, _ := enc.windowIndex(g.GridTHW)
	goldVit := make([]float32, len(g.VitHidden))
	for win, orig := range winIdx {
		for u := 0; u < mergeUnit; u++ {
			d := (orig*mergeUnit + u) * hidden
			s := (win*mergeUnit + u) * hidden
			copy(goldVit[d:d+hidden], g.VitHidden[s:s+hidden])
		}
	}

	gotVit, err := enc.ForwardViT(g.PixelValues, g.GridTHW)
	if err != nil {
		t.Fatalf("ForwardViT: %v", err)
	}
	if len(gotVit) != len(goldVit) {
		t.Fatalf("vit hidden len %d != golden %d", len(gotVit), len(goldVit))
	}
	cosV, maxV := cosine(gotVit, goldVit)
	t.Logf("Qwen2.5-VL ViT pre-merge vs HF golden (f32): cosine=%.8f max abs diff=%.3e", cosV, maxV)
	if cosV < 0.9999 {
		t.Errorf("vit_hidden cosine %.8f < 0.9999 — block stack diverges from HF", cosV)
	}

	// Stage 2: merged image features (both in original patch order).
	got, err := enc.Forward(g.PixelValues, g.GridTHW)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if len(got) != len(g.ImageFeatures) {
		t.Fatalf("image_features len %d != golden %d", len(got), len(g.ImageFeatures))
	}
	cosM, maxM := cosine(got, g.ImageFeatures)
	t.Logf("Qwen2.5-VL merged features vs HF golden (f32): cosine=%.8f max abs diff=%.3e", cosM, maxM)
	if cosM < 0.9999 {
		t.Errorf("image_features cosine %.8f < 0.9999 — merger diverges from HF", cosM)
	}
}
