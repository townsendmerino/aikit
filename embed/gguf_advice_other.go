//go:build !(linux && (amd64 || arm64))

package embed

import "os"

// openForAdvice keeps no file open where the page-cache advice is a no-op (mmap.DropFileCache): an open file would only
// hold a test's temporary directory on Windows.
func openForAdvice(string) *os.File { return nil }
