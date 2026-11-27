//go:build linux && (amd64 || arm64)

package embed

import "os"

// openForAdvice opens path for ReleaseTensors' page-cache advice, or returns nil where it cannot (the advice is then skipped).
func openForAdvice(path string) *os.File {
	f, err := os.Open(path)
	if err != nil {
		return nil
	}
	return f
}
