package linalg

import "github.com/townsendmerino/aikit/internal/kernreg"

// toArchExempt is empty on arm64: its arm64-only entry points have adapters here.
var toArchExempt = map[string]string{}

// toArchAdapters are the arm64-only row4 repack entry points.
func toArchAdapters() []kernreg.Adapter {
	quad := toN()
	return []kernreg.Adapter{
		toAd("repackInt4Row4Quad", quad, func(p *kernreg.Placer, s toShape) []any {
			n := 4 * (s.C / 2)
			dst := kernreg.Put(p, "dst", make([]byte, n))
			repackInt4Row4Quad(dst, kernreg.Put(p, "src", seqU8(n, 1)), s.C)
			return []any{dst}
		}),
		toAd("repackInt4Row4ScalesQuad", quad, func(p *kernreg.Placer, s toShape) []any {
			dst := kernreg.Put(p, "dst", make([]float32, 4*s.C))
			repackInt4Row4ScalesQuad(dst, kernreg.Put(p, "src", seqF32(4*s.C, 1)), s.C)
			return []any{dst}
		}),
		toAd("repackInt4Row4ScalesQuadF16", quad, func(p *kernreg.Placer, s toShape) []any {
			dst := kernreg.Put(p, "dst", make([]uint16, 4*s.C))
			repackInt4Row4ScalesQuadF16(dst, kernreg.Put(p, "src", F32ToF16Scales(seqF32(4*s.C, 1))), s.C)
			return []any{dst}
		}),
		toAd("RepackW4A8Row4", toRCG(), func(p *kernreg.Placer, s toShape) []any {
			packed, _ := toInt4(s)
			return []any{RepackW4A8Row4(kernreg.Put(p, "packed", packed), s.R, s.C, s.G)}
		}),
		toAd("repackW4A8Row4Deshared", toRCG(), func(p *kernreg.Placer, s toShape) []any {
			packed, _ := toInt4(s)
			b0, b1, b2, b3 := repackW4A8Row4Deshared(kernreg.Put(p, "packed", packed), s.R, s.C, s.G)
			return []any{b0, b1, b2, b3}
		}),
		toAd("repackW4A8Row4DesharedScales", toRCG(), func(p *kernreg.Placer, s toShape) []any {
			_, sc := toInt4(s)
			s0, s1, s2, s3 := repackW4A8Row4DesharedScales(kernreg.Put(p, "scales", sc), s.R, s.C, s.G)
			return []any{s0, s1, s2, s3}
		}),
		toAd("repackW4A8Row4Scales", toRCG(), func(p *kernreg.Placer, s toShape) []any {
			_, sc := toInt4(s)
			return []any{repackW4A8Row4Scales(kernreg.Put(p, "scales", sc), s.R, s.C, s.G)}
		}),
		toAd("RepackW4A8Row4ScalesF16", toRCG(), func(p *kernreg.Placer, s toShape) []any {
			_, sc := toInt4(s)
			return []any{RepackW4A8Row4ScalesF16(kernreg.Put(p, "scales", F32ToF16Scales(sc)), s.R, s.C, s.G)}
		}),
	}
}
