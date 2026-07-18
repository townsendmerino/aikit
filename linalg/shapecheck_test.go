package linalg

import (
	"strings"
	"testing"
)

// recoverPanic runs fn and returns the recovered panic value (nil if none). It
// runs fn on THIS goroutine, so it can only recover a panic that fires here —
// exactly the property M2 needs: a shape check that runs before the matmul's
// goroutine fan-out is recoverable; one that fires inside a worker is not.
func recoverPanic(fn func()) (msg string, panicked bool) {
	defer func() {
		if r := recover(); r != nil {
			panicked = true
			if s, ok := r.(string); ok {
				msg = s
			} else if e, ok := r.(error); ok {
				msg = e.Error()
			}
		}
	}()
	fn()
	return "", false
}

// TestMatmulShapeChecks_recoverableAcrossFanout (M2): a shape violation in the
// parallel matmuls must panic on the CALLER's goroutine, before the fan-out, so
// recover() catches it — not inside a spawned worker where it can't. We force
// the parallel path (threshold 0) so the check must pre-empt it.
func TestMatmulShapeChecks_recoverableAcrossFanout(t *testing.T) {
	old := parThreshold
	parThreshold = 0 // force parallelCols to fan out even for tiny shapes
	defer func() { parThreshold = old }()

	const M, K, N = 2, 8, 8
	a := make([]float32, M*K)
	bF := make([]float32, N*K)
	bQ := make([]int8, N*K)
	scales := make([]float32, N)
	good := make([]float32, M*N)
	short := make([]float32, M*N-1) // one element too short → OOB in a worker

	cases := []struct {
		name string
		call func()
	}{
		{"MatmulBT short dst", func() { MatmulBT(a, bF, short, M, K, N) }},
		{"MatmulBTQ8 short dst", func() { MatmulBTQ8(a, bQ, scales, short, M, K, N) }},
		{"MatmulBTW8A8 short dst", func() { MatmulBTW8A8(a, bQ, scales, short, M, K, N) }},
		{"MatmulBTQ8 short scales", func() { MatmulBTQ8(a, bQ, scales[:N-1], good, M, K, N) }},
		{"MatmulBTW4A8 group 0", func() {
			MatmulBTW4A8(a, make([]byte, N*((K+1)/2)), make([]float32, N), good, M, K, N, 0)
		}},
	}
	for _, c := range cases {
		msg, panicked := recoverPanic(c.call)
		if !panicked {
			t.Errorf("%s: expected a recoverable panic, got none", c.name)
			continue
		}
		if !strings.HasPrefix(msg, "linalg:") {
			t.Errorf("%s: panic %q, want a linalg: shape message", c.name, msg)
		}
	}
}

// TestShapeChecks_overflow: a dimension product that overflows int must be
// caught, not wrapped negative-and-small to slip a short slice past a length
// compare.
func TestShapeChecks_overflow(t *testing.T) {
	if got := mul(1<<40, 1<<40); got != -1 {
		t.Errorf("mul overflow: got %d, want -1 sentinel", got)
	}
	if got := mul(-1, 4); got != -1 {
		t.Errorf("mul negative: got %d, want -1", got)
	}
	if got := mul(1000, 1000); got != 1_000_000 {
		t.Errorf("mul normal: got %d, want 1000000", got)
	}

	// A WeightMat constructor with an overflowing shape must panic, not wrap.
	if _, p := recoverPanic(func() { WrapInt8(make([]int8, 4), make([]float32, 2), 1<<40, 1<<40, false) }); !p {
		t.Error("WrapInt8 with overflowing rows*cols: expected panic")
	}
	// Exact-length mismatch still rejected (unchanged behavior, overflow-safe).
	if _, p := recoverPanic(func() { WrapInt8(make([]int8, 5), make([]float32, 2), 2, 3, false) }); !p {
		t.Error("WrapInt8 wrong q8 length: expected panic")
	}
	// Negative dim rejected.
	if _, p := recoverPanic(func() { WrapF32(nil, -1, 4) }); !p {
		t.Error("WrapF32 negative dim: expected panic")
	}
	// A correctly-shaped wrap does NOT panic.
	if _, p := recoverPanic(func() { WrapInt8(make([]int8, 6), make([]float32, 2), 2, 3, false) }); p {
		t.Error("WrapInt8 valid shape: unexpected panic")
	}
}
