package linalg

import "github.com/townsendmerino/aikit/internal/kernreg"

func attnAdapters() []kernreg.Adapter {
	av := avShapes(32)
	av8 := avShapes(8)
	qk16 := qkShapes(3)
	qk8 := qkShapes(4)
	const g = 6
	avCall := func(s avShape, p *kernreg.Placer, f16 bool) any {
		scores := kernreg.Put(p, "scores["+s.String()+"]", seqF32(s.nKeys, 1))
		dst := kernreg.Put(p, "dst", make([]float32, 32))
		n := rowsLen(s.nKeys, 32, s.stride)
		if f16 {
			vals := kernreg.Put(p, "vals["+s.String()+"]", seqF16(n, 2))
			avAcc64NEON32F16(ptr(scores), ptr(vals), s.nKeys, 2*s.stride, ptr(dst))
		} else {
			vals := kernreg.Put(p, "vals["+s.String()+"]", seqF32(n, 2))
			avAcc64NEON32(ptr(scores), ptr(vals), s.nKeys, 4*s.stride, ptr(dst))
		}
		return dst
	}
	qkCall := func(s qkShape, p *kernreg.Placer, f16 bool) any {
		q := kernreg.Put(p, "q["+s.String()+"]", seqF32(4*s.k4, 1))
		dst := kernreg.Put(p, "dst", make([]float32, 16*s.blocks))
		n := rowsLen(16*s.blocks, 4*s.k4, s.stride)
		if f16 {
			rows := kernreg.Put(p, "rows["+s.String()+"]", seqF16(n, 2))
			qkAcc64NEON16F16(ptr(q), ptr(rows), 2*s.stride, s.k4, s.blocks, ptr(dst))
		} else {
			rows := kernreg.Put(p, "rows["+s.String()+"]", seqF32(n, 2))
			qkAcc64NEON16(ptr(q), ptr(rows), 4*s.stride, s.k4, s.blocks, ptr(dst))
		}
		return dst
	}
	return []kernreg.Adapter{
		{Symbol: "avAcc64NEON32", Lengths: shapeIndices(len(av)),
			Call: func(p *kernreg.Placer, i int) any { return avCall(av[i], p, false) }},
		{Symbol: "avAcc64NEON32F16", Lengths: shapeIndices(len(av)),
			Call: func(p *kernreg.Placer, i int) any { return avCall(av[i], p, true) }},
		{Symbol: "qkAcc64NEON16", Lengths: shapeIndices(len(qk16)),
			Call: func(p *kernreg.Placer, i int) any { return qkCall(qk16[i], p, false) }},
		{Symbol: "qkAcc64NEON16F16", Lengths: shapeIndices(len(qk16)),
			Call: func(p *kernreg.Placer, i int) any { return qkCall(qk16[i], p, true) }},
		{
			// G=6 queries' score rows (nKeys each, as avAcc64GroupBlocks lays them out)
			// against one 8-dim block of V; dst is six 8-float rows, hd apart (here 8+pad).
			Symbol: "avAcc64NEON8G6", Lengths: shapeIndices(len(av8)),
			Call: func(p *kernreg.Placer, i int) any {
				s := av8[i]
				scores := kernreg.Put(p, "scores["+s.String()+"]", seqF32(rowsLen(g, s.nKeys, s.nKeys), 1))
				vals := kernreg.Put(p, "vals["+s.String()+"]", seqF32(rowsLen(s.nKeys, 8, s.stride), 2))
				dstStride := 8 + i%3
				dst := kernreg.Put(p, "dst", make([]float32, rowsLen(g, 8, dstStride)))
				avAcc64NEON8G6(ptr(scores), 4*s.nKeys, ptr(vals), s.nKeys, 4*s.stride, ptr(dst), 4*dstStride)
				return dst
			},
		},
		{
			// G=6 query rows of K = 4·k4 (K apart, as qkAcc64GroupKeys lays them out) against
			// blocks of eight keys; dst is six rows of 8·blocks, 8·blocks apart.
			Symbol: "qkAcc64NEON8G6", Lengths: shapeIndices(len(qk8)),
			Call: func(p *kernreg.Placer, i int) any {
				s := qk8[i]
				k := 4 * s.k4
				nk := 8 * s.blocks
				q := kernreg.Put(p, "q["+s.String()+"]", seqF32(rowsLen(g, k, k), 1))
				rows := kernreg.Put(p, "rows["+s.String()+"]", seqF32(rowsLen(nk, k, s.stride), 2))
				dst := kernreg.Put(p, "dst", make([]float32, rowsLen(g, nk, nk)))
				qkAcc64NEON8G6(ptr(q), 4*k, ptr(rows), 4*s.stride, s.k4, s.blocks, ptr(dst), 4*nk)
				return dst
			},
		},
	}
}
