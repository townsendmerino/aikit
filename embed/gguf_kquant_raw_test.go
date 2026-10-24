package embed

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

// refDequantQ5K and refDequantQ6K are ggml's dequantize_row_q5_K and dequantize_row_q6_K written out per element, the
// reference the raw blocks are checked against.
func refDequantQ5K(blk []byte) []float32 {
	y := make([]float32, 256)
	d, dm := halfBitsToF32(uint16(blk[0])|uint16(blk[1])<<8), halfBitsToF32(uint16(blk[2])|uint16(blk[3])<<8)
	sm := func(j int) (float32, float32) {
		q := blk[4:16]
		if j < 4 {
			return float32(q[j] & 63), float32(q[j+4] & 63)
		}
		return float32(q[j+4]&15 | (q[j-4]>>6)<<4), float32(q[j+4]>>4 | (q[j]>>6)<<4)
	}
	qh, qs := blk[16:48], blk[48:176]
	for j := range 4 {
		sc0, m0 := sm(2 * j)
		sc1, m1 := sm(2*j + 1)
		for l := range 32 {
			lo, hi := float32(qs[32*j+l]&15), float32(qs[32*j+l]>>4)
			if qh[l]&(1<<(2*j)) != 0 {
				lo += 16
			}
			if qh[l]&(2<<(2*j)) != 0 {
				hi += 16
			}
			y[64*j+l] = d*sc0*lo - dm*m0
			y[64*j+32+l] = d*sc1*hi - dm*m1
		}
	}
	return y
}

func refDequantQ6K(blk []byte) []float32 {
	y := make([]float32, 256)
	d := halfBitsToF32(uint16(blk[208]) | uint16(blk[209])<<8)
	for n := range 2 {
		ql, qh, sc := blk[64*n:], blk[128+32*n:], blk[192+8*n:]
		for l := range 32 {
			is := l / 16
			q := [4]int{int(ql[l]&15 | (qh[l]&3)<<4), int(ql[l+32]&15 | (qh[l]>>2&3)<<4), int(ql[l]>>4 | (qh[l]>>4&3)<<4), int(ql[l+32]>>4 | (qh[l]>>6&3)<<4)}
			for t := range 4 {
				y[128*n+32*t+l] = d * float32(int8(sc[is+2*t])) * float32(q[t]-32)
			}
		}
	}
	return y
}

// TestQ5KQ6KRaw_matchesDequant: on real files, every Q5_K and Q6_K tensor's raw blocks, dequantized by the per-element
// reference, agree with RowDequantizer on three rows (within 1e-6 of the row's largest weight: the reference and the
// dequantizer may round d·sc·q differently); every tensor of another type reports ok=false. The files are Phi-3 mini's q4
// GGUF (Q5_K attn_qkv, Q6_K ffn_down and output) and Qwen2.5-Coder-0.5B q4_k_m (Q6_K); AIKIT_KQUANT_GGUF overrides them. With
// no file the test skips (a skip is not a pass).
func TestQ5KQ6KRaw_matchesDequant(t *testing.T) {
	var paths []string
	if p := os.Getenv("AIKIT_KQUANT_GGUF"); p != "" {
		paths = []string{p}
	} else if home, err := os.UserHomeDir(); err == nil {
		paths = []string{filepath.Join(home, "models", "phi3-mini-4k-gguf", "Phi-3-mini-4k-instruct-q4.gguf"),
			filepath.Join(home, "models", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")}
	}
	checked := map[string]int{}
	for _, path := range paths {
		if _, err := os.Stat(path); err != nil {
			t.Logf("no GGUF at %s", path)
			continue
		}
		g, err := OpenGGUFMmap(path)
		if err != nil {
			t.Fatal(err)
		}
		for _, name := range g.Names() {
			for _, k := range []struct {
				kind string
				typ  uint32
				bb   int
				get  func(string) ([]int, []byte, bool, error)
				ref  func([]byte) []float32
			}{{"Q5_K", ggmlTypeQ5_K, 176, g.Q5KRaw, refDequantQ5K}, {"Q6_K", ggmlTypeQ6_K, 210, g.Q6KRaw, refDequantQ6K}} {
				dims, raw, ok, err := k.get(name)
				if err != nil {
					t.Fatalf("%s: %v", name, err)
				}
				if ok != (g.tensors[name].typ == k.typ) {
					t.Fatalf("%s: %sRaw ok=%v for type %d", name, k.kind, ok, g.tensors[name].typ)
				}
				if !ok {
					continue
				}
				if len(dims) != 2 || dims[0]%256 != 0 || len(raw) != dims[1]*dims[0]/256*k.bb {
					t.Fatalf("%s: dims %v, %d bytes", name, dims, len(raw))
				}
				in, out := dims[0], dims[1]
				_, into, err := g.RowDequantizer(name)
				if err != nil {
					t.Fatal(err)
				}
				want := make([]float32, in)
				for _, r := range []int{0, out / 2, out - 1} {
					if err := into(r*in, want); err != nil {
						t.Fatal(err)
					}
					var got []float32
					rb := in / 256 * k.bb
					for b := range in / 256 {
						got = append(got, k.ref(raw[r*rb+b*k.bb:r*rb+(b+1)*k.bb])...)
					}
					var big float64
					for _, v := range want {
						big = max(big, math.Abs(float64(v)))
					}
					for i := range want {
						if math.Abs(float64(got[i]-want[i])) > 1e-6*big {
							t.Fatalf("%s row %d [%d]: raw %v, dequant %v", name, r, i, got[i], want[i])
						}
					}
				}
				checked[k.kind]++
			}
		}
		g.Close()
	}
	if checked["Q5_K"] == 0 || checked["Q6_K"] == 0 {
		t.Skipf("no Q5_K or no Q6_K tensor found (checked %v)", checked)
	}
	t.Logf("tensors agreeing with RowDequantizer: %v", checked)
}
