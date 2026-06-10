//go:build aikit_checks

package linalg

import (
	"strings"
	"testing"
)

// These run only under `-tags aikit_checks` — they prove the contract checks fire
// (and with a message naming the kernel) on misuse the production build trusts.
func TestChecks_fire(t *testing.T) {
	mustPanic(t, "MatmulBTW4A8", func() {
		// a too short for M*K (1*32).
		MatmulBTW4A8(make([]float32, 4), make([]byte, 16), make([]float32, 1), make([]float32, 1), 1, 32, 1, 32)
	})
	mustPanic(t, "MatmulBTQ4", func() {
		MatmulBTQ4(make([]float32, 32), make([]byte, 16), make([]float32, 1), make([]float32, 1), 1, 32, 1, 0) // group 0
	})
	mustPanic(t, "DequantizeRowInt4", func() {
		DequantizeRowInt4(make([]byte, 2), make([]float32, 1), 4, 8, make([]float32, 4)) // dst < cols
	})
	mustPanic(t, "DequantizeRowInt8", func() {
		DequantizeRowInt8(make([]int8, 8), 1, make([]float32, 4)) // dst < q
	})
}

// A correctly-shaped call must NOT panic under checks.
func TestChecks_passClean(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("clean call panicked: %v", r)
		}
	}()
	MatmulBTW4A8(make([]float32, 32), make([]byte, 16), make([]float32, 1), make([]float32, 1), 1, 32, 1, 32)
}

func mustPanic(t *testing.T, name string, f func()) {
	t.Helper()
	defer func() {
		r := recover()
		if r == nil {
			t.Errorf("%s: expected a contract-check panic, got none", name)
			return
		}
		if msg, ok := r.(string); ok && !strings.Contains(msg, name) {
			t.Errorf("%s: panic message %q does not name the kernel", name, msg)
		}
	}()
	f()
}
