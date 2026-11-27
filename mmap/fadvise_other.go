//go:build !(linux && (amd64 || arm64))

package mmap

import "os"

// DropFileCache is a no-op here: off Linux the page cache does not honor the advice for a read-only mapping's file (see
// Advise for darwin), and the other Linux architectures pass fadvise64's arguments differently (fadvise_linux64.go).
func DropFileCache(*os.File, int64, int64) error { return nil }
