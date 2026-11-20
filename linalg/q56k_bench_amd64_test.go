package linalg

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
)

// BenchmarkQ56KvsInt8: the M=1 Q5_K and Q6_K matmuls against the int8 W8A8 kind a consumer would
// otherwise widen them to, at Phi-3-mini's fused qkv (3072→9216) and down (8192→3072) shapes, serial
// and at goinfer's decode fan-out. SetBytes is each kind's own weight bytes; compare ns/op, which is
// what a decode token pays.
func BenchmarkQ56KvsInt8(b *testing.B) {
	r := rand.New(rand.NewPCG(75, 76))
	shapes := []struct {
		name string
		K, N int
	}{{"qkv3072x9216", 3072, 9216}, {"down8192x3072", 8192, 3072}}
	for _, sh := range shapes {
		K, N := sh.K, sh.N
		a := agRandMat(r, K)
		q5raw := q5kTestRows(r, N, K, false)
		q6raw := q6kTestRows(r, N, K, 0)
		q5, err := WrapQ5K(q5raw, N, K)
		if err != nil {
			b.Fatal(err)
		}
		q6, err := WrapQ6K(q6raw, N, K)
		if err != nil {
			b.Fatal(err)
		}
		w := make([]float32, N*K)
		q5.Row(0, w[:K])
		for n := 1; n < N; n++ {
			q5.Row(n, w[n*K:(n+1)*K])
		}
		i8 := QuantizeInt8(w, N, K, true)
		dst := make([]float32, N)
		for _, thr := range []int{math.MaxInt, 1 << 20} {
			mode := "serial"
			if thr != math.MaxInt {
				mode = "fanout"
			}
			for _, k := range []struct {
				name  string
				wm    *WeightMat
				bytes int
			}{{"q5k", &q5, len(q5raw)}, {"q6k", &q6, len(q6raw)}, {"int8w8a8", &i8, N*K + 4*N}} {
				b.Run(fmt.Sprintf("%s/%s/%s", sh.name, mode, k.name), func(b *testing.B) {
					var ws Workspace
					ws.SetThreshold(thr)
					b.SetBytes(int64(k.bytes))
					for range b.N {
						k.wm.MatmulBTInto(&ws, a, dst, 1)
					}
				})
			}
		}
	}
}
