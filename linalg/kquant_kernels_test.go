package linalg

import "github.com/townsendmerino/aikit/internal/kernreg"

var kquantRows = []kernreg.Row{
	{Symbol: "dotQ4KAVX2", Arch: "amd64", Dispatch: "dotQ4K", Oracle: "TestQ4K_kernelMatchesGo"},
	{Symbol: "dotQ5KAVX2", Arch: "amd64", Dispatch: "dotQ5K", Oracle: "TestQ56K_kernelMatchesGo"},
	{Symbol: "dotQ6KAVX2", Arch: "amd64", Dispatch: "dotQ6K", Oracle: "TestQ56K_kernelMatchesGo"},
	{Symbol: "dotPartials16SDOT", Arch: "arm64", Dispatch: "dotPartials16", Oracle: "TestKQuantDotPartials_matchesScalar"},
}

// kquantBlocks is nSB super-blocks of blockB bytes from seqU8, with finite f16 scales
// written at each of the dOffs byte offsets (d, and dmin where the format has one), so the
// kernel's f32 arithmetic stays finite.
func kquantBlocks(nSB, blockB int, dOffs ...int) []byte {
	raw := seqU8(nSB*blockB, 3)
	for b := range nSB {
		for i, off := range dOffs {
			h := uint16(0x3C00 - 0x400*i) // 1.0, then 0.5
			raw[b*blockB+off], raw[b*blockB+off+1] = byte(h), byte(h>>8)
		}
	}
	return raw
}
