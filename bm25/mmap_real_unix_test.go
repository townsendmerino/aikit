//go:build unix

package bm25

// mmapIsReal: mmap.MapReadOnly really maps on unix; elsewhere it heap-reads (see package mmap), so the tests that
// bound a mapped load's heap only apply here.
const mmapIsReal = true
