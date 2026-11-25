package linalg

import "github.com/townsendmerino/aikit/internal/kernreg"

func dequanti8Adapters() []kernreg.Adapter {
	return []kernreg.Adapter{{
		Symbol: "dequantI8NEON",
		// n a multiple of 16; the loop runs at least once (dequantRowInt8 calls it only
		// with n >= 16).
		Lengths: upTo(16, 160, 1008, 1024, 1040)[1:],
		Call: func(p *kernreg.Placer, n int) any {
			q := kernreg.Put(p, "q", seqI8(n, 1))
			dst := kernreg.Put(p, "dst", make([]float32, n))
			dequantI8NEON(ptr(q), ptr(dst), n, 0.03125)
			return dst
		},
	}}
}
