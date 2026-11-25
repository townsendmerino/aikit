package linalg

import "github.com/townsendmerino/aikit/internal/kernreg"

func kquantAdapters() []kernreg.Adapter {
	q4k := func() bool { return hasQ4KAVX2 }
	// n is nSB, the count of 256-weight super-blocks; the dispatchers pass nSB ≥ 1, and each
	// kernel tests nSB ≤ 0 before its loop.
	lengths := upTo(1, 9, 16, 17)
	// in is one row of nSB super-blocks of blockB bytes and its per-32 activations: 256
	// int8 codes, eight scales and (for the kinds with a minimum) eight scale·sum terms each.
	type in struct {
		row       []byte
		aq        []int8
		aS, asumf []float32
	}
	place := func(p *kernreg.Placer, nSB, blockB int, withMin bool, dOffs ...int) in {
		x := in{
			row: kernreg.Put(p, "row", kquantBlocks(nSB, blockB, dOffs...)),
			aq:  kernreg.Put(p, "aq", seqI8(nSB*qkK, 1)),
			aS:  kernreg.Put(p, "aS", seqF32(nSB*qkK/q4kGroup, 2)),
		}
		if withMin {
			x.asumf = kernreg.Put(p, "asumf", seqF32(nSB*qkK/q4kGroup, 3))
		}
		return x
	}
	return []kernreg.Adapter{
		{
			Symbol: "dotQ4KAVX2", Needs: q4k, Lengths: lengths,
			Call: func(p *kernreg.Placer, nSB int) any {
				x := place(p, nSB, q4kBlockB, true, 0, 2)
				return dotQ4KAVX2(ptr(x.row), ptr(x.aq), ptr(x.aS), ptr(x.asumf), nSB)
			},
		},
		{
			Symbol: "dotQ5KAVX2", Needs: q4k, Lengths: lengths,
			Call: func(p *kernreg.Placer, nSB int) any {
				x := place(p, nSB, q5kBlockB, true, 0, 2)
				return dotQ5KAVX2(ptr(x.row), ptr(x.aq), ptr(x.aS), ptr(x.asumf), nSB)
			},
		},
		{
			Symbol: "dotQ6KAVX2", Needs: q4k, Lengths: lengths,
			Call: func(p *kernreg.Placer, nSB int) any {
				x := place(p, nSB, q6kBlockB, false, 208)
				return dotQ6KAVX2(ptr(x.row), ptr(x.aq), ptr(x.aS), nSB)
			},
		},
	}
}
