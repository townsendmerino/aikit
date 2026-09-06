package linalg

import (
	"math/rand"
	"testing"
)

func TestDotW4A8Uncentered_matchesScalar_exact(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for trial := range 200 {
		K := 1 + rng.Intn(400)
		group := 32
		nGroups := (K + group - 1) / group
		act := make([]int8, K)
		for i := range act {
			act[i] = int8(rng.Intn(255) - 128)
		}
		w := make([]float32, K)
		for i := range w {
			w[i] = float32(rng.NormFloat64())
		}
		packed, scales := QuantizeGroupsInt4(w, 1, K, group)
		sumAct := make([]int32, nGroups)
		SumActGroupsInto(sumAct, act, 1, K, group)

		got := dotW4A8UncenteredScalar(act, packed, scales, sumAct, group, K)
		want := dotW4A8Scalar(act, packed, scales, group, K)
		if got != want {
			t.Fatalf("trial %d K=%d: got %v want %v (bit mismatch)", trial, K, got, want)
		}
	}
}
