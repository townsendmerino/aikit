package vision

import (
	"compress/gzip"
	"encoding/json"
	"math"
	"os"
	"testing"
)

// TestQwen3Deepstack_tiny is G-S10b's tiny half (goinfer's docs/tasks/task-multimodal-support-2026-10.md, S10): a tiny
// Qwen3-VL tower WITH DeepStack (blocks 0 and 1), every LayerNorm randomised, against transformers' Qwen3VLVisionModel
// on two images (testdata/qwen3vl-vision-tiny, pinned by goinfer's scripts/pin_qwen3vl_vision_tiny.py). ForwardDeepstack's
// main rows and each DeepStack set at worst-row cosine >= 0.9999, and the planted defect, each DeepStack merger's
// post-shuffle LayerNorm dropped, must fall under it.
func TestQwen3Deepstack_tiny(t *testing.T) {
	const dir = "../testdata/qwen3vl-vision-tiny"
	f, err := os.Open(dir + "/golden.json.gz")
	if err != nil {
		t.Skipf("no fixture: %v", err)
	}
	defer f.Close()
	zr, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	var g struct {
		Grids       [][3]int    `json:"grids"`
		PixelValues []float32   `json:"pixel_values"`
		Merged      []float32   `json:"merged"`
		Deepstack   [][]float32 `json:"deepstack"`
		OutHidden   int         `json:"out_hidden"`
	}
	if err := json.NewDecoder(zr).Decode(&g); err != nil {
		t.Fatal(err)
	}
	e, err := LoadQwen3VisionEncoder(dir, false)
	if err != nil {
		t.Fatal(err)
	}
	worst := func(got, want []float32) float64 {
		if len(got) != len(want) {
			t.Fatalf("%d values, want %d", len(got), len(want))
		}
		w := 1.0
		d := g.OutHidden
		for r := range len(want) / d {
			var dot, na, nb float64
			for j := range d {
				x, y := float64(got[r*d+j]), float64(want[r*d+j])
				dot, na, nb = dot+x*y, na+x*x, nb+y*y
			}
			w = math.Min(w, dot/math.Sqrt(na*nb))
		}
		return w
	}
	merged, deep, err := e.ForwardDeepstack(g.PixelValues, g.Grids)
	if err != nil {
		t.Fatal(err)
	}
	if len(deep) != len(g.Deepstack) {
		t.Fatalf("%d DeepStack sets, HF %d", len(deep), len(g.Deepstack))
	}
	if w := worst(merged, g.Merged); w < 0.9999 {
		t.Errorf("main merged rows: worst cosine %.9f under 0.9999", w)
	} else {
		t.Logf("main merged rows: worst cosine %.9f", w)
	}
	plain, err := e.Forward(g.PixelValues, g.Grids)
	if err != nil {
		t.Fatal(err)
	}
	bitEqual(t, "Forward against ForwardDeepstack's main rows", plain, merged)
	for k := range deep {
		w := worst(deep[k], g.Deepstack[k])
		t.Logf("DeepStack set %d (block %d): worst cosine %.9f", k, e.Cfg.DeepstackVisualIndexes[k], w)
		if w < 0.9999 {
			t.Errorf("DeepStack set %d: worst cosine %.9f under 0.9999", k, w)
		}
	}
	// The planted defect: each DeepStack merger without its post-shuffle LayerNorm (fc1 -> GELU -> fc2 straight on the
	// shuffled rows), from the same tapped block outputs.
	h := map[int][]float32{}
	if _, err := e.forwardBlocks(g.PixelValues, g.Grids, func(li int, x []float32) { h[li] = append([]float32(nil), x...) }); err != nil {
		t.Fatal(err)
	}
	c := e.Cfg
	mh := c.HiddenSize * c.SpatialMergeSize * c.SpatialMergeSize
	for k, idx := range c.DeepstackVisualIndexes {
		m := &e.deepstack[k]
		groups := len(h[idx]) / mh
		mid := make([]float32, groups*mh)
		m.fc1w.MatmulBT(h[idx], mid, groups)
		addBias(mid, m.fc1b, groups, mh)
		geluErf(mid)
		out := make([]float32, groups*c.OutHiddenSize)
		m.fc2w.MatmulBT(mid, out, groups)
		addBias(out, m.fc2b, groups, c.OutHiddenSize)
		w := worst(out, g.Deepstack[k])
		t.Logf("planted: DeepStack set %d without its post-shuffle norm: worst cosine %.6f", k, w)
		if w >= 0.9999 {
			t.Errorf("planted defect left DeepStack set %d green (worst %.9f): the fixture cannot see a dropped norm", k, w)
		}
	}
}
