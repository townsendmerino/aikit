package linalg

import (
	"math/rand/v2"
	"testing"
)

func int4SchemeErr(t *testing.T, scheme string, w []float32, rows, cols int) (float64, []byte, []float32) {
	t.Helper()
	prev := Int4WeightScheme()
	SetInt4WeightScheme(scheme)
	defer SetInt4WeightScheme(prev)
	q4, s := QuantizeGroupsInt4(w, rows, cols, 32)
	dq := make([]float32, cols)
	nG := (cols + 31) / 32
	var e float64
	for i := range rows {
		// The production dequantizer, so the candidates are proven format-compatible (negative
		// scales included), not just self-consistent.
		DequantizeRowInt4(q4[i*cols/2:(i+1)*cols/2], s[i*nG:(i+1)*nG], 32, cols, dq)
		for k := range cols {
			d := float64(w[i*cols+k] - dq[k])
			e += d * d
		}
	}
	return e, q4, s
}

// TestInt4Scheme_candidates: both candidates decode through DequantizeRowInt4 unchanged and beat the
// default max/7 rule on reconstruction error. "fullrange" produces code -8 (the default never does),
// and "mse" is never worse than "fullrange", whose scale is one of its grid points.
func TestInt4Scheme_candidates(t *testing.T) {
	r := rand.New(rand.NewPCG(11, 12))
	const rows, cols = 16, 256
	w := make([]float32, rows*cols)
	for i := range w {
		w[i] = float32(r.NormFloat64())
	}
	eDef, _, _ := int4SchemeErr(t, "", w, rows, cols)
	eFull, q4, _ := int4SchemeErr(t, "fullrange", w, rows, cols)
	eMSE, _, _ := int4SchemeErr(t, "mse", w, rows, cols)
	t.Logf("squared error: default %.4g, fullrange %.4g, mse %.4g", eDef, eFull, eMSE)
	if !(eFull < eDef) || !(eMSE <= eFull) {
		t.Errorf("want mse <= fullrange < default, got %.4g / %.4g / %.4g", eMSE, eFull, eDef)
	}
	hasMinus8 := false
	for _, b := range q4 {
		if b&0x0F == 0 || b>>4 == 0 {
			hasMinus8 = true
		}
	}
	if !hasMinus8 {
		t.Error("fullrange never produced code -8 (nibble 0)")
	}
}
