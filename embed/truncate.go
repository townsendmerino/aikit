package embed

// Truncate returns the first dim components of a Matryoshka embedding,
// L2-renormalized — a lower-dimensional embedding that trades a little fidelity
// for memory and faster scans. It composes with the int8 ann.FlatI8 index for a
// compounded reduction (e.g. 256→128 dims stored int8 is 8× smaller than the
// 256-d float32 original).
//
// This is only meaningful for embeddings trained with Matryoshka Representation
// Learning (MRL), where a leading prefix is itself a valid embedding (Nomic Embed
// and the OpenAI v3 models are the common examples). Truncating a non-MRL
// embedding degrades it — use TestMatryoshkaRecall-style measurement to confirm a
// given model survives truncation before relying on it.
//
// The input is not modified; a fresh slice is returned. dim is clamped to
// [0, len(v)].
func Truncate(v []float32, dim int) []float32 {
	if dim < 0 {
		dim = 0
	}
	if dim > len(v) {
		dim = len(v)
	}
	out := make([]float32, dim)
	copy(out, v[:dim])
	return l2Normalize(out)
}
