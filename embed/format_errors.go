package embed

import (
	"errors"
	"fmt"
)

// ErrFormat is returned (wrapped) by the loaders — OpenSafetensors* and OpenGGUF* —
// when the input is not a valid blob of that format: a bad magic, an unsupported
// version, or a truncated / malformed header. Test for it with errors.Is rather than
// string-matching:
//
//	g, err := embed.OpenGGUF(path)
//	if errors.Is(err, embed.ErrFormat) {
//		// not a usable GGUF file (corrupt, wrong magic, or an unsupported version)
//	}
//
// Per-tensor lookups (tensor-not-found, use-after-Close) are NOT wrapped with
// ErrFormat — only the file's own format errors are.
var ErrFormat = errors.New("embed: malformed or unsupported file format")

// errFormatf formats a blob-parse failure that wraps ErrFormat.
func errFormatf(format string, args ...any) error {
	return fmt.Errorf(format+": %w", append(args, ErrFormat)...)
}
