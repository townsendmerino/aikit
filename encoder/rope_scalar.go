//go:build !goexperiment.simd

package encoder

// rotateHalfInto (default build) — byte-for-byte the loop applyRows (rope.go)
// had before rope_simd.go existed.
func rotateHalfInto(x1, x2, c, s []float32) {
	n := len(x1)
	if n == 0 {
		return
	}
	_ = x2[n-1]
	_ = c[n-1]
	_ = s[n-1]

	d := 0
	for ; d+3 < n; d += 4 {
		a0, b0, cd0, sd0 := x1[d+0], x2[d+0], c[d+0], s[d+0]
		a1, b1, cd1, sd1 := x1[d+1], x2[d+1], c[d+1], s[d+1]
		a2, b2, cd2, sd2 := x1[d+2], x2[d+2], c[d+2], s[d+2]
		a3, b3, cd3, sd3 := x1[d+3], x2[d+3], c[d+3], s[d+3]

		x1[d+0] = a0*cd0 - b0*sd0
		x2[d+0] = b0*cd0 + a0*sd0
		x1[d+1] = a1*cd1 - b1*sd1
		x2[d+1] = b1*cd1 + a1*sd1
		x1[d+2] = a2*cd2 - b2*sd2
		x2[d+2] = b2*cd2 + a2*sd2
		x1[d+3] = a3*cd3 - b3*sd3
		x2[d+3] = b3*cd3 + a3*sd3
	}
	for ; d < n; d++ {
		a, b := x1[d], x2[d]
		cd, sd := c[d], s[d]
		x1[d] = a*cd - b*sd
		x2[d] = b*cd + a*sd
	}
}
