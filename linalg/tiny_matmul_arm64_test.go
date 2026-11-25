package linalg

import "github.com/townsendmerino/aikit/internal/kernreg"

// tmArchExempt is empty on arm64: its Row4 entry points are swept here.
var tmArchExempt = map[string]string{}

// tmArchAdapters sweeps the arm64-only W4A8 4-row entry points. Their contract is M=1,
// group=32, N a multiple of 4 and K a multiple of 32 (the tile entry takes any M); the
// sweep crosses each anyway, and an explicit refusal is part of the result. The 4-row
// layout is a permutation of the canonical bytes, so any bytes of the right length are a
// valid weight.
func tmArchAdapters() []kernreg.Adapter {
	shapes := tmGroupShapes(tmG{tmMKN{1, 64, 8}, 32, 0}, []int{96, 127, 128, 129})
	inputs := func(p *kernreg.Placer, s tmG) (a []float32, w4 []byte, wS []float32, dst []float32) {
		bpr, ng := tmInt4Sizes(s.K, s.group)
		return kernreg.Put(p, "a", seqF32(s.M*s.K, 1)),
			kernreg.Put(p, "w4Row4", seqU8(s.N*bpr, 2)),
			kernreg.Put(p, "wScales4", tmScales(s.N*ng, 3)),
			kernreg.Put(p, "dst", make([]float32, s.M*s.N))
	}
	row4 := func(name string, mm func(ws *Workspace, a []float32, w4 []byte, wS, dst []float32, M, K, N, group int)) kernreg.Adapter {
		return tmGroupAdapter(name, shapes, func(p *kernreg.Placer, s tmG) any {
			a, w4, wS, dst := inputs(p, s)
			ws := tmWorkspace(s.actGroup)
			return tmResult(kernreg.Refused(func() { mm(ws, a, w4, wS, dst, s.M, s.K, s.N, s.group) }), dst)
		})
	}
	return []kernreg.Adapter{
		row4("MatmulBTW4A8Row4Into", MatmulBTW4A8Row4Into),
		row4("MatmulBTW4A8Row4TileInto", MatmulBTW4A8Row4TileInto),
		row4("MatmulBTW4A8Row4PrefetchInto", func(ws *Workspace, a []float32, w4 []byte, wS, dst []float32, M, K, N, g int) {
			MatmulBTW4A8Row4PrefetchInto(ws, a, w4, wS, dst, M, K, N, g, 256)
		}),
		tmGroupAdapter("MatmulBTW4A8Row4DesharedInto", shapes, func(p *kernreg.Placer, s tmG) any {
			bpr, ng := tmInt4Sizes(s.K, s.group)
			nq := s.N / 4
			a := kernreg.Put(p, "a", seqF32(s.M*s.K, 1))
			var w [4][]byte
			var sc [4][]float32
			for r := range 4 {
				w[r] = kernreg.Put(p, "w"+string(rune('0'+r)), seqU8(nq*bpr, 2+r))
				sc[r] = kernreg.Put(p, "s"+string(rune('0'+r)), tmScales(nq*ng, 3+r))
			}
			dst := kernreg.Put(p, "dst", make([]float32, s.M*s.N))
			ws := tmWorkspace(s.actGroup)
			r := kernreg.Refused(func() {
				MatmulBTW4A8Row4DesharedInto(ws, a, w[0], w[1], w[2], w[3], sc[0], sc[1], sc[2], sc[3], dst, s.M, s.K, s.N, s.group)
			})
			return tmResult(r, dst)
		}),
	}
}
