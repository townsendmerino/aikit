package linalg

import "github.com/townsendmerino/aikit/internal/kernreg"

// w4a8Rows: every W4A8 kernel reads, per 32-wide group, 32 int8 activations and 16
// packed weight bytes per weight row, plus one scale per weight row. The rows without a
// Dispatch are measurement probes that no production function calls.
var w4a8Rows = []kernreg.Row{
	{Symbol: "dotW4A8FoldAVX2", Arch: "amd64", Dispatch: "dotW4A8", Oracle: "TestW4A8_dotMatchesScalar"},
	{Symbol: "dotW4A8Fold2AccAVX2", Arch: "amd64", Undispatched: "a measurement arm: two chains sum in a different order, so not wired into dotW4A8's bit-identity-gated dispatch", Oracle: "TestDotW4A8Fold2AccAVX2_matchesOracle"},
	{Symbol: "dotW4A8SplitHalfAVX2", Arch: "amd64", Dispatch: "w4a8SplitHalfSpan", Oracle: "TestDotW4A8SplitHalfAVX2_matchesOracle"},
	{Symbol: "dotW4A8SplitHalfScaledAVX2", Arch: "amd64", Dispatch: "matmulBTW4A8SplitHalfGroupedQ", Oracle: "TestActGroup_splitHalfScaledKernelOddGroups"},
	{Symbol: "dotW4A8FoldAVX512VNNI", Arch: "amd64", Dispatch: "dotW4A8", Oracle: "TestAVX512VNNI_dotW4A8FoldAVX512VNNI_matchesScalar"},
	{Symbol: "cvtF16ToF32x8F16C", Arch: "amd64", Dispatch: "widenF16", Oracle: "TestWidenF16_exhaustive"},
	{Symbol: "dotW4A8FoldF16RowAVX2", Arch: "amd64", Dispatch: "dotW4A8F16Row", Oracle: "TestW4A8F16_everyPathMatchesF32"},
	{Symbol: "dotW4A8SplitHalfTile4RowAVX2", Arch: "amd64", Dispatch: "w4a8SplitHalfTileRows", Oracle: "TestMatmulBTW4A8SplitHalfTile_bitIdenticalToCanonical"},
	{Symbol: "dotW4A8Tile4RowAVX2", Arch: "amd64", Dispatch: "w4a8TileRows", Oracle: "TestW4A8TileRowsIndexHoisted_isBitIdentical"},

	{Symbol: "dotW4A8FoldSDOT", Arch: "arm64", Dispatch: "dotW4A8", Oracle: "TestW4A8_dotMatchesScalar"},
	{Symbol: "dotW4A8FoldSDOTv2", Arch: "arm64", Undispatched: "a measurement arm kept beside dotW4A8FoldSDOT for comparison", Oracle: "TestDotW4A8FoldSDOTv2_matchesOracle"},
	{Symbol: "dotW4A8SplitHalfSDOT", Arch: "arm64", Undispatched: "a measurement arm: the split-half layout alone, which dotW4A8's canonical-layout callers do not produce", Oracle: "TestDotW4A8SplitHalfSDOT_matchesOracle"},
	{Symbol: "dotW4A8FoldSDOT2Acc", Arch: "arm64", Undispatched: "a measurement arm: the two-chain probe of the SDOT fold", Oracle: "TestDotW4A8FoldSDOT2Acc_matchesOracle"},
	{Symbol: "dotW4A8FoldSDOT4Acc", Arch: "arm64", Undispatched: "a measurement arm: the four-chain probe of the SDOT fold", Oracle: "TestDotW4A8FoldSDOT4Acc_matchesOracle"},
	{Symbol: "dotW4A8SplitHalf2Acc", Arch: "arm64", Undispatched: "a measurement arm: two chains over the split-half layout", Oracle: "TestDotW4A8SplitHalf2Acc_matchesOracle"},
	{Symbol: "dotW4A8SplitHalf4Acc", Arch: "arm64", Undispatched: "a measurement arm: four chains over the split-half layout", Oracle: "TestDotW4A8SplitHalf4Acc_matchesOracle"},
	{Symbol: "dotW4A8SplitHalf4Row", Arch: "arm64", Dispatch: "w4a8Row4Span", Oracle: "TestDotW4A8SplitHalf4RowFold_bitIdenticalToSplitHalf4Row"},
	{Symbol: "dotW4A8SplitHalf4RowPrefetch", Arch: "arm64", Dispatch: "matmulBTW4A8Row4PrefetchInto", Oracle: "TestMatmulBTW4A8Row4PrefetchInto_bitIdenticalToRow4"},
	{Symbol: "dotW4A8SplitHalf4RowDeshared", Arch: "arm64", Dispatch: "matmulBTW4A8Row4DesharedInto", Oracle: "TestMatmulBTW4A8Row4DesharedInto_bitIdenticalToRow4"},
	{Symbol: "w4a8LaneCorrNeg8", Arch: "arm64", Dispatch: "matmulBTW4A8Row4Into", Oracle: "TestW4A8LaneCorrNeg8_matchesScalar"},
	{Symbol: "dotW4A8SplitHalf4RowFold", Arch: "arm64", Dispatch: "w4a8Row4Span", Oracle: "TestDotW4A8SplitHalf4RowFold_bitIdenticalToSplitHalf4Row"},
	{Symbol: "dotW4A8SplitHalf4RowFoldF16", Arch: "arm64", Dispatch: "w4a8Row4SpanF16", Oracle: "TestDotW4A8SplitHalf4RowFoldF16_matchesF32"},
	{Symbol: "dotW4A8Tile4RowSDOT", Arch: "arm64", Dispatch: "w4a8TileRows", Oracle: "TestW4A8Tile4RowSDOT_bitIdenticalToFoldSDOT"},
	{Symbol: "dotW4A8Row4Tile4x4", Arch: "arm64", Dispatch: "w4a8Row4TileSpan", Oracle: "TestMatmulBTW4A8Row4TileInto_bitIdenticalToCanonical"},
}

// w4a8Groups returns the group counts a W4A8 adapter sweeps: lo, lo+step, … up to 24
// groups, then each block-size boundary of a long row (31-33, 63-65) rounded to the
// kernel's multiple, and one 160-group (K = 5120) row.
func w4a8Groups(lo, step int) []int {
	var out []int
	for g := lo; g <= 24; g += step {
		out = append(out, g)
	}
	for _, g := range []int{31, 32, 33, 63, 64, 65, 160} {
		if g%step == 0 && g >= lo && g > 24 {
			out = append(out, g)
		}
	}
	return out
}

// w4a8Tile is one activation-tile shape: nGroups groups per row and four activation
// rows actPad bytes further apart than the row's own K.
type w4a8Tile struct{ nGroups, actPad int }

// w4a8TileShapes are the tile shapes swept: every group count of w4a8Groups(lo, 1) at
// the dispatcher's stride (actStride = K), and a few with rows padded apart, so the last
// row's end, not the stride, is what meets the guard.
func w4a8TileShapes(lo int) []w4a8Tile {
	var out []w4a8Tile
	for _, g := range w4a8Groups(lo, 1) {
		out = append(out, w4a8Tile{g, 0})
	}
	for _, g := range []int{1, 2, 7} {
		if g >= lo {
			out = append(out, w4a8Tile{g, 32}, w4a8Tile{g, 5})
		}
	}
	return out
}

// w4a8Indices returns 0..n-1, the Lengths of an adapter that sweeps a shape list.
func w4a8Indices(n int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = i
	}
	return out
}

// w4a8Row places one weight row's inputs for g groups: activations, packed bytes and
// f32 scales.
func w4a8Row(p *kernreg.Placer, g int) (act []int8, packed []byte, scales []float32) {
	return kernreg.Put(p, "act", seqI8(32*g, 1)), kernreg.Put(p, "packed", seqU8(16*g, 2)), kernreg.Put(p, "scales", seqF32(g, 3))
}

// w4a8SeqU16 is n binary16 values with exact small magnitudes, different for each seed.
func w4a8SeqU16(n, seed int) []uint16 {
	x := make([]uint16, n)
	for i := range x {
		x[i] = F32ToF16(float32((i*5+seed*3)%9 - 4))
	}
	return x
}
