//go:build !(linux || darwin)

package guardmem

import (
	"errors"
	"runtime"
)

// Supported is true where New maps guard pages.
const Supported = false

func pageSize() int { return 4096 }

func platform() string { return runtime.GOOS + "/" + runtime.GOARCH }

func mapGuarded(int) ([]byte, error) { return nil, errors.New("guard pages are not supported here") }

func unmap([]byte) error { return nil }
