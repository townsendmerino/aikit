//go:build amd64

package linalg

// AVX-512 VNNI tier for the int8×int8 dot product, ahead of dotI8AVX2 in the
// dotI8 dispatch cascade (quant_i8_amd64.go), selected by cpufeat.HasAVX512VNNI
// (hand-rolled CPUID/XGETBV in internal/cpufeat, no x/sys dep).
//
// dotI8AVX2's own doc comment flags the obvious next lever and why it wasn't
// taken there: "VPMADDUBSW would remove the widening entirely, but u8×i8 pair
// sums can exceed int16 and it SATURATES; that route needs range-limited
// codes and belongs with the VNNI work, not here." VPDPBUSD is that work,
// and it sidesteps the hazard rather than working around it: it accumulates
// straight into int32 (no intermediate int16 stage to saturate), at the cost
// of being an asymmetric u8×s8 op, not s8×s8. See dotI8AVX512VNNI below for
// how the sign is recovered exactly.

// dotI8AVX512VNNI returns Σ a[i]*b[i] as int32 over the first n int8 pairs (n
// a multiple of 64 — the caller's contract, mirroring dotI8AVX2's "multiple of
// 16"; quant_i8_amd64.go's dotI8 peels this prefix and routes any remainder to
// the next tier down). Implemented in dot_i8_avx512vnni_amd64.s.
//
// VPDPBUSD computes an UNSIGNED×SIGNED dot (u8×s8 → int32, no saturation), but
// both operands here are signed. The fix is the standard offset trick, done
// entirely in-register: XOR each a-byte with 0x80 (bit-identical to +128 for a
// two's-complement int8, since a+128 ∈ [0,255] already covers the whole
// unsigned range) to get an unsigned view au, then
//
//	Σ au[i]*b[i] = Σ (a[i]+128)*b[i] = Σ a[i]*b[i] + 128·Σ b[i]
//
// so the true dot is the VPDPBUSD(au, b) result minus 128 times Σ b[i]. That
// correction sum is itself one more VPDPBUSD — dot(ones_u8, b) — rather than a
// separate widen-and-add pass, so the whole kernel is two VPDPBUSD per 64
// bytes plus the XOR, no extra reduction shape. Both accumulators are int32
// and integer addition doesn't reassociate, so — like dotI8AVX2 — the result
// is bit-identical to dotI8Scalar regardless of loop/chunk order; only the
// SIMD-vs-scalar boundary (n rounded down to a multiple of 64) can differ
// from a pure scalar sum, and it doesn't, because integer sums never do.
//
//go:noescape
func dotI8AVX512VNNI(a, b *int8, n int) int32
