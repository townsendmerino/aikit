//go:build aikit_checks

package linalg

import "fmt"

// Checked build (`-tags aikit_checks`): the quant-kernel contract checks validate
// the caller's shapes/arguments and panic with a precise message on violation.
// These contracts are otherwise trusted silently by the hot kernels — a bad shape
// or group=0 surfaces as a divide-by-zero or out-of-bounds read deep in an asm
// loop. Run the test suite with `-tags aikit_checks` (and the fuzzers) to exercise
// them; production builds (checks_off.go) compile these to no-ops.

const checksEnabled = true

func assertf(cond bool, format string, a ...any) {
	if !cond {
		panic("linalg/checks: " + fmt.Sprintf(format, a...))
	}
}

func checkDequantInt8(q []int8, dst []float32) {
	assertf(len(dst) >= len(q), "DequantizeRowInt8: dst len %d < q len %d", len(dst), len(q))
}

func checkDequantInt4(packed []byte, scales []float32, group, cols int, dst []float32) {
	assertf(group > 0, "DequantizeRowInt4: group must be > 0, got %d", group)
	assertf(cols >= 0, "DequantizeRowInt4: cols must be >= 0, got %d", cols)
	assertf(len(dst) >= cols, "DequantizeRowInt4: dst len %d < cols %d", len(dst), cols)
	assertf(len(packed) >= (cols+1)/2, "DequantizeRowInt4: packed len %d < ceil(cols/2) = %d", len(packed), (cols+1)/2)
	ng := (cols + group - 1) / group
	assertf(len(scales) >= ng, "DequantizeRowInt4: scales len %d < nGroups %d", len(scales), ng)
}

// checkGroupMatmul validates the shared shape contract of the group-quantized
// matmuls (MatmulBTW4A8, MatmulBTQ4): a is [M,K], the packed weights are [N, bytesPerRow],
// scales are [N, nGroups], dst is [M,N].
func checkGroupMatmul(name string, aLen int, packed []byte, scales []float32, dstLen, M, K, N, group int) {
	assertf(group > 0, "%s: group must be > 0, got %d", name, group)
	assertf(M >= 0 && K >= 0 && N >= 0, "%s: negative dim (M=%d K=%d N=%d)", name, M, K, N)
	assertf(aLen >= M*K, "%s: a len %d < M*K = %d", name, aLen, M*K)
	assertf(dstLen >= M*N, "%s: dst len %d < M*N = %d", name, dstLen, M*N)
	ng, bpr := groupsFor(K, group)
	assertf(len(packed) >= N*bpr, "%s: packed len %d < N*bytesPerRow = %d", name, len(packed), N*bpr)
	assertf(len(scales) >= N*ng, "%s: scales len %d < N*nGroups = %d", name, len(scales), N*ng)
}
