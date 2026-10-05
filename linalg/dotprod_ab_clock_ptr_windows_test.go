//go:build arm64 && windows

package linalg

import "unsafe"

func unsafePtr(p *int64) unsafe.Pointer { return unsafe.Pointer(p) }
