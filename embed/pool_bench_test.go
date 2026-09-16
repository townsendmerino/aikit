package embed

import (
	"fmt"
	"testing"
)

func BenchmarkL2Normalize(b *testing.B) {
	for _, dim := range []int{256, 384, 768, 1024} {
		v := make([]float32, dim)
		for i := range v {
			v[i] = float32(i%100) * 0.1
		}
		b.Run(fmt.Sprintf("D%d", dim), func(b *testing.B) {
			b.SetBytes(int64(dim * 4))
			b.ReportAllocs()
			for b.Loop() {
				_ = L2Normalize(v)
			}
		})
	}
}
