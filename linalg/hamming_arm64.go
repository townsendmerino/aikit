//go:build arm64

package linalg

// hammingNEON is the vectorized NEON kernel in hamming_arm64.s.
//
//go:noescape
func hammingNEON(q *uint64, codes *uint64, words int, n int, dst *uint16)

func hammingRows(q, codes []uint64, words, n int, dst []uint16) {
	if n == 0 || words == 0 {
		return
	}
	// Only the two specialized, single-pass entries in hamming_arm64.s
	// (words==4 dim 256, words==12 dim 768 — the real production shapes) are
	// dispatched here: measured ~3x faster than hammingRowsGeneric's own
	// words==4/12 fast paths (56µs vs 177µs at d256, 163µs vs 535µs at d768,
	// n=100k, M1 Pro — BenchmarkHammingRows/popcnt vs /generic), because NEON's
	// VCNT processes 16 bytes per instruction where Go's OnesCount64 handles 8.
	//
	// general_entry in the .s file (arbitrary words) is NOT dispatched: it
	// accumulates per-byte popcounts in an 8-bit NEON lane across words/2
	// iterations before widening, and overflows silently once words>=64
	// (dim>~4032) — confirmed: words=64 with maximally-differing input wraps
	// 4096 to 0. It has no other caller and is unverified performance-wise
	// besides, so anything outside the two proven shapes takes the portable,
	// overflow-free Go path instead.
	if words != 4 && words != 12 {
		hammingRowsGeneric(q, codes, words, n, dst)
		return
	}
	hammingNEON(&q[0], &codes[0], words, n, &dst[0])
}
