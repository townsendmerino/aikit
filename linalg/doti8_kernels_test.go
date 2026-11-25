package linalg

import "github.com/townsendmerino/aikit/internal/kernreg"

// doti8Rows are the int8 dot kernels: the 1×1 dots, the per-32-scaled dots, the
// activation tiles and the eight-column dot.
var doti8Rows = []kernreg.Row{
	{Symbol: "dotI8NEON", Arch: "arm64", Dispatch: "dotI8", Oracle: "TestDotI8_matchesScalar"},
	{Symbol: "dotI8SDOT", Arch: "arm64", Dispatch: "dotI8", Oracle: "TestDotI8SDOT_matchesScalar"},
	{Symbol: "dotI8AVX512VNNI", Arch: "amd64", Dispatch: "dotI8", Oracle: "TestAVX512VNNI_dotI8AVX512VNNI_matchesScalar"},
	{Symbol: "dotI8Scaled32AVX2", Arch: "amd64", Dispatch: "dotI8Scaled32", Oracle: "TestDotI8Scaled32_matchesGo"},
	{Symbol: "dotI8Scaled32x2AVX2", Arch: "amd64", Dispatch: "w8a8GroupedSpan", Oracle: "TestDotI8Scaled32x2_matchesGo"},
	{Symbol: "dotI8Tile4x1AVX2", Arch: "amd64", Dispatch: "w8a8TileRect", Oracle: "TestW8A8TileVsSpanAB_amd64"},
	{Symbol: "dotI8Tile4x4", Arch: "arm64", Dispatch: "w8a8TileRect", Oracle: "TestW8A8TileVsSpanAB"},
	{Symbol: "dotI8x8AVX2", Arch: "amd64", Dispatch: "dotI8x8", Oracle: "TestDotI8x8_matchesScalar"},
}

// i8TileShape is one activation-tile call: n bytes per row, rows stride bytes apart.
type i8TileShape struct{ n, stride int }

// i8TileShapes reach the tile kernels' 16-byte step at one, two and many steps, with
// rows packed (stride = n) and with the K%16 remainder the dispatcher leaves between
// rows (stride = n + 1 … n + 15).
func i8TileShapes() []i8TileShape {
	var out []i8TileShape
	for _, n := range []int{16, 32, 48, 64, 1024} {
		for _, extra := range []int{0, 1, 15} {
			out = append(out, i8TileShape{n, n + extra})
		}
	}
	return out
}

// indexes returns 0 … n-1: the Lengths of an adapter whose n indexes a shape list.
func indexes(n int) []int { return upTo(1, n-1) }
