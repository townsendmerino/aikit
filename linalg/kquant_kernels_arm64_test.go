package linalg

import "github.com/townsendmerino/aikit/internal/kernreg"

func kquantAdapters() []kernreg.Adapter {
	return []kernreg.Adapter{
		{
			Symbol: "dotPartials16SDOT", Needs: func() bool { return hasDotProd },
			// n is nsub, the count of 16-element sub-blocks; one SDOT each, no unroll.
			// The dispatcher passes nsub ≥ 1 (the kernel also returns at once on 0).
			Lengths: upTo(1, 18, 32, 33, 128, 129),
			Call: func(p *kernreg.Placer, nsub int) any {
				codes := kernreg.Put(p, "codes", seqI8(16*nsub, 1))
				qs := kernreg.Put(p, "qs", seqI8(16*nsub, 2))
				out := kernreg.Put(p, "out", make([]int32, nsub))
				dotPartials16SDOT(ptr(codes), ptr(qs), nsub, ptr(out))
				return out
			},
		},
	}
}
