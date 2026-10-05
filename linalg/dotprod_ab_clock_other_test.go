//go:build arm64 && !windows

package linalg

import "time"

var abEpoch = time.Now()

// abNow is monotonic nanoseconds (time.Now carries a monotonic reading, and ticks at nanosecond resolution off Windows).
func abNow() int64 { return time.Since(abEpoch).Nanoseconds() }

func abClockName() string { return "time.Now (monotonic)" }
