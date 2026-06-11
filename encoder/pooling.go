package encoder

// pooling selects how a sequence's per-token hidden states reduce to one
// embedding vector — an internal, config-driven seam toward supporting
// BERT-family models beyond CodeRankEmbed (which is CLS). cls is the default
// (and the zero value). Kept unexported until a loader that actually selects it
// (with a parity-pinned mean-pooled model) lands — see roadmap §2.5.
//
// Only the reduction is parameterized here; the rest of the forward (RoPE
// positions, SwiGLU MLP) is still NomicBert-specific. Learned absolute positions
// and a GELU FFN — the other axes a MiniLM/bge-class model needs — plus the
// loader that reads them, remain to be added and parity-pinned against a real
// model's golden fixture.
type pooling string

const (
	// poolCLS takes the [CLS] token at position 0 — CodeRankEmbed and most
	// rerankers. The default.
	poolCLS pooling = "cls"
	// poolMean averages the sequence's real tokens — the sentence-transformers
	// default (MiniLM, many embedders).
	poolMean pooling = "mean"
)

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
