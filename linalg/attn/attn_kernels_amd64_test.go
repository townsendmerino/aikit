package attn

import (
	"github.com/townsendmerino/aikit/internal/cpufeat"
	"github.com/townsendmerino/aikit/internal/kernreg"
)

func attnAdapters() []kernreg.Adapter {
	avx2 := func() bool { return cpufeat.HasAVX2 }
	f16 := func() bool { return cpufeat.HasAVX2 && cpufeat.HasF16C }
	av := avShapes(32)
	qk4 := qkShapes(5)
	qk16 := qkShapes(3)
	avCall := func(s avShape, p *kernreg.Placer, f16 bool) any {
		scores := kernreg.Put(p, "scores["+s.String()+"]", seqF32(s.nKeys, 1))
		dst := kernreg.Put(p, "dst", make([]float32, 32))
		n := rowsLen(s.nKeys, 32, s.stride)
		if f16 {
			vals := kernreg.Put(p, "vals["+s.String()+"]", seqF16(n, 2))
			avAcc64AVX32F16(ptr(scores), ptr(vals), s.nKeys, 2*s.stride, ptr(dst))
		} else {
			vals := kernreg.Put(p, "vals["+s.String()+"]", seqF32(n, 2))
			avAcc64AVX32(ptr(scores), ptr(vals), s.nKeys, 4*s.stride, ptr(dst))
		}
		return dst
	}
	// keys is the keys per block: 4 or 16.
	qkCall := func(s qkShape, keys int, p *kernreg.Placer, f16 bool) any {
		q := kernreg.Put(p, "q["+s.String()+"]", seqF32(4*s.k4, 1))
		dst := kernreg.Put(p, "dst", make([]float32, keys*s.blocks))
		n := rowsLen(keys*s.blocks, 4*s.k4, s.stride)
		switch {
		case f16 && keys == 4:
			rows := kernreg.Put(p, "rows["+s.String()+"]", seqF16(n, 2))
			qkAcc64AVX4F16(ptr(q), ptr(rows), 2*s.stride, s.k4, s.blocks, ptr(dst))
		case f16:
			rows := kernreg.Put(p, "rows["+s.String()+"]", seqF16(n, 2))
			qkAcc64AVX16F16(ptr(q), ptr(rows), 2*s.stride, s.k4, s.blocks, ptr(dst))
		case keys == 4:
			rows := kernreg.Put(p, "rows["+s.String()+"]", seqF32(n, 2))
			qkAcc64AVX4(ptr(q), ptr(rows), 4*s.stride, s.k4, s.blocks, ptr(dst))
		default:
			rows := kernreg.Put(p, "rows["+s.String()+"]", seqF32(n, 2))
			qkAcc64AVX16(ptr(q), ptr(rows), 4*s.stride, s.k4, s.blocks, ptr(dst))
		}
		return dst
	}
	return []kernreg.Adapter{
		{Symbol: "avAcc64AVX32", Needs: avx2, Lengths: shapeIndices(len(av)),
			Call: func(p *kernreg.Placer, i int) any { return avCall(av[i], p, false) }},
		{Symbol: "avAcc64AVX32F16", Needs: f16, Lengths: shapeIndices(len(av)),
			Call: func(p *kernreg.Placer, i int) any { return avCall(av[i], p, true) }},
		{Symbol: "qkAcc64AVX4", Needs: avx2, Lengths: shapeIndices(len(qk4)),
			Call: func(p *kernreg.Placer, i int) any { return qkCall(qk4[i], 4, p, false) }},
		{Symbol: "qkAcc64AVX4F16", Needs: f16, Lengths: shapeIndices(len(qk4)),
			Call: func(p *kernreg.Placer, i int) any { return qkCall(qk4[i], 4, p, true) }},
		{Symbol: "qkAcc64AVX16", Needs: avx2, Lengths: shapeIndices(len(qk16)),
			Call: func(p *kernreg.Placer, i int) any { return qkCall(qk16[i], 16, p, false) }},
		{Symbol: "qkAcc64AVX16F16", Needs: f16, Lengths: shapeIndices(len(qk16)),
			Call: func(p *kernreg.Placer, i int) any { return qkCall(qk16[i], 16, p, true) }},
	}
}
