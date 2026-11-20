//go:build unix

package linalg

import (
	"syscall"
	"testing"
	"time"
)

func processCPU(t *testing.T) time.Duration {
	t.Helper()
	var ru syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &ru); err != nil {
		t.Fatalf("getrusage: %v", err)
	}
	return time.Duration(ru.Utime.Nano() + ru.Stime.Nano())
}

// TestSpanPool_idleCPU: parked helpers burn no CPU. Over a quiet window after the pool has
// parked, the whole process's user+system time stays far below one core's share.
func TestSpanPool_idleCPU(t *testing.T) {
	checkOnce(t, 8192, 8, nil)
	waitParked(t)
	const window = 300 * time.Millisecond
	before := processCPU(t)
	time.Sleep(window)
	used := processCPU(t) - before
	if used > window/10 {
		t.Fatalf("idle process used %v CPU over %v with the pool parked", used, window)
	}
	t.Logf("idle CPU over %v: %v", window, used)
}
