package embed

import "math"

// l2NormEps avoids div-by-zero on degenerate (all-zero) vectors. Matches the
// convention used by sentence-transformers and a typical eps for float32.
const l2NormEps = 1e-12

// L2Normalize divides v by its L2 norm in place and returns it. If the norm is
// below l2NormEps (a degenerate / all-UNK input) v is zeroed and returned — a
// stable, safe value rather than NaN (the Python reference NaNs there; we return
// zero to keep downstream cosine well-behaved).
//
// This is the ONE L2-normalize the kit uses — the Model2Vec pooling path, the
// Matryoshka Truncate, and the encoder embedders (encoder.BERT.Embed) all route
// through it, so the degenerate-case behavior is pinned in a single place rather
// than diverging across three copies (audit #21).
//
// Precision: the sum-of-squares accumulator is float64 and each element is divided
// by the float64 norm before the cast to float32 — float32 accumulation here would
// compound the drift the float64 pooling path is careful to avoid.
func L2Normalize(v []float32) []float32 {
	var sq float64
	for _, x := range v {
		sq += float64(x) * float64(x)
	}
	norm := math.Sqrt(sq)
	if norm < l2NormEps {
		clear(v)
		return v
	}
	for i, x := range v {
		v[i] = float32(float64(x) / norm)
	}
	return v
}
