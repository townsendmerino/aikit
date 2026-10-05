//go:build arm64 && windows

package linalg

import "syscall"

// Windows' time.Now/nanotime ticks coarsely (the first run of this A/B read ZERO elapsed for sub-millisecond blocks and produced NaN and Inf speedups on 88 of 150 cell lines), so the A/B reads the
// high-resolution performance counter instead.
var (
	abKernel32 = syscall.NewLazyDLL("kernel32.dll")
	abQPC      = abKernel32.NewProc("QueryPerformanceCounter")
	abQPCFreq  = abKernel32.NewProc("QueryPerformanceFrequency")
	abFreq     = func() int64 { var f int64; abQPCFreq.Call(uintptr(unsafePtr(&f))); return f }()
)

// abNow is monotonic nanoseconds from the performance counter.
func abNow() int64 {
	var c int64
	abQPC.Call(uintptr(unsafePtr(&c)))
	return c/abFreq*1_000_000_000 + (c%abFreq)*1_000_000_000/abFreq
}

func abClockName() string { return "QueryPerformanceCounter" }
