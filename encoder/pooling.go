package encoder

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path"
	"path/filepath"
)

// pooling selects how a sequence's per-token hidden states reduce to one
// embedding vector — a config-driven property read per model (from a
// sentence-transformers 1_Pooling/config.json), not assumed. Different embedders
// disagree: CodeRankEmbed and BGE pool CLS, MiniLM and most others mean.
//
// Only the reduction is parameterized here; the rest of the forward (positions,
// FFN) is still architecture-specific per loader.
type pooling string

const (
	// poolCLS takes the [CLS] token at position 0 — CodeRankEmbed, BGE, most
	// rerankers.
	poolCLS pooling = "cls"
	// poolMean averages the sequence's real tokens — the sentence-transformers
	// default (MiniLM, many embedders).
	poolMean pooling = "mean"
)

// poolingFromConfig reads the sentence-transformers pooling declaration at
// <dir>/1_Pooling/config.json (disk path) and returns the reduction mode — the
// BERT loader's entry point. See poolingFromBytes for the semantics.
func poolingFromConfig(dir string, fallback pooling) (pooling, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "1_Pooling", "config.json"))
	if err != nil {
		return fallback, nil // no ST pooling module — use the family default
	}
	return poolingFromBytes(raw, fallback)
}

// poolingFromFS is poolingFromConfig over an fs.FS — the Nomic/CodeRankEmbed
// loader's entry point (it reads via fs.FS so it can serve from an embed.FS).
func poolingFromFS(fsys fs.FS, dir string, fallback pooling) (pooling, error) {
	raw, err := fs.ReadFile(fsys, path.Join(dir, "1_Pooling", "config.json"))
	if err != nil {
		return fallback, nil
	}
	return poolingFromBytes(raw, fallback)
}

// poolingFromBytes parses a sentence-transformers 1_Pooling/config.json body.
// cls/mean map to the mode; an unsupported mode (max / mean-sqrt-len) is a hard
// error rather than a silent mispool — an embedder that pools the wrong way
// still returns plausible-looking vectors, exactly the failure the parity gates
// exist to catch. A file that sets none falls back to the loader's family
// default. (A missing file also falls back — handled by the callers above.)
func poolingFromBytes(raw []byte, fallback pooling) (pooling, error) {
	var pc struct {
		CLS      bool `json:"pooling_mode_cls_token"`
		Mean     bool `json:"pooling_mode_mean_tokens"`
		Max      bool `json:"pooling_mode_max_tokens"`
		MeanSqrt bool `json:"pooling_mode_mean_sqrt_len_tokens"`
	}
	if err := json.Unmarshal(raw, &pc); err != nil {
		return "", fmt.Errorf("encoder: parse 1_Pooling/config.json: %w", err)
	}
	switch {
	case pc.Max || pc.MeanSqrt:
		return "", fmt.Errorf("encoder: unsupported pooling mode (max/mean_sqrt_len) in 1_Pooling/config.json")
	case pc.CLS:
		return poolCLS, nil
	case pc.Mean:
		return poolMean, nil
	default:
		return fallback, nil
	}
}

// poolOne reduces ONE sequence's [L, D] hidden states (L real tokens, no padding)
// to a single D-vector per mode. The caller passes only the real tokens — for the
// batched path, the per-sequence sub-slice of length realLen — so mean needs no
// attention mask. mean accumulates in float64 (matching embed's pooling), then
// narrows; cls (the default, including the zero value) copies position 0.
func poolOne(seq []float32, L, D int, mode pooling) []float32 {
	out := make([]float32, D)
	if L == 0 {
		return out // degenerate (no tokens) — stable zero vector
	}
	if mode == poolMean {
		acc := make([]float64, D)
		for i := range L {
			row := seq[i*D : i*D+D]
			for j := range D {
				acc[j] += float64(row[j])
			}
		}
		inv := 1.0 / float64(L)
		for j := range out {
			out[j] = float32(acc[j] * inv)
		}
		return out
	}
	copy(out, seq[:D]) // poolCLS / zero value
	return out
}
