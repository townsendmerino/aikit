package ann

import (
	"errors"
	"fmt"
)

// ErrFormat is returned (wrapped) by every index-loading path — Load (HNSW),
// LoadFlatI8, and LoadFlatI8Mmap — when the input is not a valid serialized index:
// a bad magic, an unsupported format version, or a truncated / internally
// inconsistent blob. Test for it with errors.Is rather than string-matching:
//
//	idx, err := ann.LoadFlatI8(blob)
//	if errors.Is(err, ann.ErrFormat) {
//		// the bytes are not a usable FlatI8 index (corrupt, or a newer format)
//	}
//
// I/O failures from LoadFlatI8Mmap (open/mmap) are NOT wrapped with ErrFormat —
// only the blob's own format errors are.
var ErrFormat = errors.New("ann: malformed or unsupported serialized index")

// errFormatf formats a blob-parse failure that wraps ErrFormat.
func errFormatf(format string, args ...any) error {
	return fmt.Errorf(format+": %w", append(args, ErrFormat)...)
}
