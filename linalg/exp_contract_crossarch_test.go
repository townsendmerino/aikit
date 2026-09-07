package linalg

import (
	"encoding/binary"
	"hash/fnv"
	"math"
	"math/rand/v2"
	"testing"
)

// contractGoldenInputs builds the fixed input set the cross-architecture golden
// is computed over. Deterministic by construction — a seeded PCG and fixed
// boundary values, no wall clock, no map iteration — so the same bytes are
// produced on every machine and every run.
func contractGoldenInputs() []float32 {
	var xs []float32
	for _, b := range []float64{0, 1, -1, 0.5, -0.5, 10, -10, 40, -40, 88, -87,
		expOverflowF32, expUnderflowF32, 0.625, -0.625, 9, -9, 4, -4} {
		for d := -8; d <= 8; d++ {
			xs = append(xs, float32(b)+float32(d)*0.0625)
		}
	}
	rng := rand.New(rand.NewPCG(0x60_1de, 0x9a71))
	for range 40_000 {
		xs = append(xs, float32(rng.NormFloat64()*20))
	}
	for len(xs)%8 != 0 { // a multiple of 8 keeps both kernels' main loops fed
		xs = append(xs, 0)
	}
	return xs
}

func contractGoldenHash(t *testing.T) uint64 {
	t.Helper()
	xs := contractGoldenInputs()
	h := fnv.New64a()
	var b [4]byte
	emit := func(v float32) {
		binary.LittleEndian.PutUint32(b[:], math.Float32bits(v))
		_, _ = h.Write(b[:])
	}
	out := make([]float32, len(xs))

	// Every public contract entry point, through whatever implementation this
	// build selected.
	ExpContractInto(out, xs)
	for _, v := range out {
		emit(v)
	}
	SiLUContractInto(out, xs)
	for _, v := range out {
		emit(v)
	}
	GELUTanhContractInto(out, xs)
	for _, v := range out {
		emit(v)
	}
	GELUContractInto(out, xs)
	for _, v := range out {
		emit(v)
	}
	// Softmax over fixed-size rows, so the pinned summation order is exercised
	// at several lengths including ones that are not multiples of the lane count.
	for _, n := range []int{7, 8, 16, 17, 1536} {
		if n <= len(xs) {
			SoftmaxRowContractInto(out[:n], xs[:n])
			for _, v := range out[:n] {
				emit(v)
			}
		}
	}
	return h.Sum64()
}

// contractGolden is the FNV-1a hash of every contract entry point's output over
// contractGoldenInputs. It is the direct proof of the property the whole S-06
// contract exists to provide: an M1 running NEON kernels and a Zen 2 running
// AVX2 kernels produce the SAME BITS, so a golden generated on one machine is
// valid on the other.
//
// Every other gate in this package proves a kernel matches its scalar oracle on
// the machine it ran on. Only this one compares ACROSS architectures, and it does
// it by comparing to a constant rather than to a second run — which is the only
// way, since no single process can execute both instruction sets.
//
// If this fails, do NOT re-baseline it. It means an implementation diverged, and
// the per-arch bit-identity tests will say which one.
const contractGolden = 0x28d4e904d9441147

func TestContractCrossArchGolden(t *testing.T) {
	got := contractGoldenHash(t)
	if got != contractGolden {
		t.Fatalf("contract output hash 0x%016x != golden 0x%016x — an implementation "+
			"diverged across architectures; the per-arch bit-identity tests will name it",
			got, contractGolden)
	}
}
