//go:build arm64

package linalg

import "testing"

func BenchmarkRepackW4A8Row4(b *testing.B) {
	const N, K, group = 1024, 4096, 32
	bpr := K / 2
	packed := make([]byte, N*bpr)
	for i := range packed {
		packed[i] = byte(i*13 + 7)
	}
	b.SetBytes(int64(N * bpr))
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		_ = RepackW4A8Row4(packed, N, K, group)
	}
}

func BenchmarkRepackW4A8SplitHalf(b *testing.B) {
	const N, K, group = 1024, 4096, 32
	bpr := K / 2
	packed := make([]byte, N*bpr)
	for i := range packed {
		packed[i] = byte(i*13 + 7)
	}
	b.SetBytes(int64(N * bpr))
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		_ = RepackW4A8SplitHalf(packed, N, K, group)
	}
}

func BenchmarkRepackInt4Row4Quad(b *testing.B) {
	const K = 4096
	bpr := K / 2
	src := make([]byte, 4*bpr)
	dst := make([]byte, 4*bpr)
	for i := range src {
		src[i] = byte(i*17 + 3)
	}
	b.SetBytes(int64(4 * bpr))
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		RepackInt4Row4Quad(dst, src, K)
	}
}

func BenchmarkRepackInt4SplitHalfRow(b *testing.B) {
	const K = 4096
	bpr := K / 2
	src := make([]byte, bpr)
	dst := make([]byte, bpr)
	for i := range src {
		src[i] = byte(i*17 + 3)
	}
	b.SetBytes(int64(bpr))
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		RepackInt4SplitHalfRow(dst, src, K)
	}
}
