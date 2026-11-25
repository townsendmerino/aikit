package guardmem

import (
	"strconv"
	"unsafe"
)

// unsafeSlice re-slices x to n elements with no bounds check, so a test can read past it.
func unsafeSlice[T any](x []T, n int) []T { return unsafe.Slice(unsafe.SliceData(x), n) }

// unsafeBefore returns the k elements that precede x in memory.
func unsafeBefore[T any](x []T, k int) []T {
	var zero T
	p := unsafe.Add(unsafe.Pointer(unsafe.SliceData(x)), -k*int(unsafe.Sizeof(zero)))
	return unsafe.Slice((*T)(p), k)
}

func itoa(n int) string { return strconv.Itoa(n) }
