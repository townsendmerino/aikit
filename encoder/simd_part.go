//go:build goexperiment.simd

package encoder

import "simd"

// maxF32Lanes bounds the vector width the part helpers stage through: 16 float32s is
// the widest Float32s the simd package builds (512 bits).
const maxF32Lanes = 16

// loadF32Part loads s, which is shorter than one vector, into the low lanes of a
// Float32s and zeroes the rest. It stages s through a stack buffer and never reads past
// len(s): simd.LoadFloat32sPart reads a whole vector on amd64, past the end of a short
// slice (TestElementwise_noReadPastShortTail).
func loadF32Part(s []float32) simd.Float32s {
	var probe simd.Float32s
	var buf [maxF32Lanes]float32
	copy(buf[:], s)
	return simd.LoadFloat32s(buf[:probe.Len()])
}

// storeF32Part writes v's low len(s) lanes to s, which is shorter than one vector, and
// nothing past it.
func storeF32Part(v simd.Float32s, s []float32) {
	var buf [maxF32Lanes]float32
	v.Store(buf[:v.Len()])
	copy(s, buf[:])
}
