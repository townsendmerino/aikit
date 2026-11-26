package attn

import (
	"fmt"

	"github.com/townsendmerino/aikit/internal/kernreg"

	"github.com/townsendmerino/aikit/linalg"
)

var attnRows = []kernreg.Row{
	{Symbol: "avAcc64AVX32", Arch: "amd64", Dispatch: "avAcc64Blocks", Oracle: "TestAvAcc64AVX32_matchesGo"},
	{Symbol: "qkAcc64AVX4", Arch: "amd64", Dispatch: "qkAcc64KeysSplit", Oracle: "TestQkAcc64AVX16_matchesAVX4AndGo"},
	{Symbol: "qkAcc64AVX16", Arch: "amd64", Dispatch: "qkAcc64Keys16", Oracle: "TestQkAcc64AVX16_matchesAVX4AndGo"},
	{Symbol: "qkAcc64AVX16F16", Arch: "amd64", Dispatch: "qkAcc64Keys16F16", Oracle: "TestQkAcc64KeysF16_strideGuard"},
	{Symbol: "avAcc64AVX32F16", Arch: "amd64", Dispatch: "avAcc64BlocksF16", Oracle: "TestAcc64F16AVX_isLoadBearing"},
	{Symbol: "qkAcc64AVX4F16", Arch: "amd64", Dispatch: "qkAcc64KeysSplitF16", Oracle: "TestQkAcc64AVX4F16_matchesAVX4OnWidened"},
	{Symbol: "avAcc64NEON32", Arch: "arm64", Dispatch: "avAcc64Blocks", Oracle: "TestAVAcc64NEON32_matchesGo"},
	{Symbol: "qkAcc64NEON16", Arch: "arm64", Dispatch: "qkAcc64Keys", Oracle: "TestQKAcc64NEON16_matchesGo"},
	{Symbol: "qkAcc64NEON16F16", Arch: "arm64", Dispatch: "qkAcc64KeysF16", Oracle: "TestAcc64F16NEON_isLoadBearing"},
	{Symbol: "avAcc64NEON32F16", Arch: "arm64", Dispatch: "avAcc64BlocksF16", Oracle: "TestAcc64F16NEON_isLoadBearing"},
	{Symbol: "avAcc64NEON8G6", Arch: "arm64", Dispatch: "avAcc64GroupBlocks", Oracle: "TestAVAcc64NEON8G6_matchesGo"},
	{Symbol: "qkAcc64NEON8G6", Arch: "arm64", Dispatch: "qkAcc64GroupKeys", Oracle: "TestQKAcc64NEON8G6_matchesGo"},
}

// avShape is one call of an AV kernel: nKeys keys of a block of width dims, value rows
// stride elements apart.
type avShape struct{ nKeys, stride int }

// qkShape is one call of a QK kernel: blocks of keys of K = 4·k4 dims, key rows stride
// elements apart.
type qkShape struct{ k4, blocks, stride int }

// avShapes crosses key counts across the kernels' one-key loop with strides from the
// tightest (the block's own width) up to a 4 KiB multiple.
func avShapes(width int) []avShape {
	var out []avShape
	for _, nk := range []int{1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 33} {
		for _, st := range []int{width, width + 1, width + 8, 2 * width, 1024} {
			out = append(out, avShape{nk, st})
		}
	}
	return out
}

// qkShapes crosses head dims (k4, so K is a multiple of 4) with block counts and strides
// from the tightest (K itself) to K plus an odd pad.
func qkShapes(maxBlocks int) []qkShape {
	var out []qkShape
	for _, k4 := range []int{1, 2, 3, 4, 5, 8, 16, 17, 32} {
		for b := 1; b <= maxBlocks; b++ {
			for _, pad := range []int{0, 1, 3, 4} {
				out = append(out, qkShape{k4, b, 4*k4 + pad})
			}
		}
	}
	return out
}

// shapeIndices is 0..n-1: an adapter's Lengths when n indexes a list of shapes.
func shapeIndices(n int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = i
	}
	return out
}

// seqF16 is seqF32's values as binary16 bits.
func seqF16(n, seed int) []uint16 {
	x := make([]uint16, n)
	linalg.F32ToF16Slice(x, seqF32(n, seed))
	return x
}

// rowsLen is the elements a kernel reads from n rows of width elements, stride apart:
// the last row ends at its width, not at its stride.
func rowsLen(n, width, stride int) int {
	if n == 0 {
		return 0
	}
	return (n-1)*stride + width
}

func (s avShape) String() string { return fmt.Sprintf("nKeys=%d stride=%d", s.nKeys, s.stride) }
func (s qkShape) String() string {
	return fmt.Sprintf("k4=%d blocks=%d stride=%d", s.k4, s.blocks, s.stride)
}
