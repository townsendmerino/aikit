package linalg_test

import (
	"fmt"

	"github.com/townsendmerino/aikit/linalg"
)

// MatmulBT computes dst[M,N] = a[M,K] · b[N,K]ᵀ — b is the PyTorch [out,in]
// weight layout the safetensors checkpoints store, so no transpose copy is
// needed. SIMD-accelerated (NEON/AVX2) under the hood.
func Example() {
	a := []float32{1, 2}             // 1×2
	b := []float32{1, 0, 0, 1, 1, 1} // 3×2 — rows [1,0] [0,1] [1,1]
	dst := make([]float32, 3)        // 1×3

	linalg.MatmulBT(a, b, dst, 1, 2, 3)
	fmt.Println(dst)
	// Output:
	// [1 2 3]
}
