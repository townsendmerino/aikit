package encoder

import (
	"errors"
	"io/fs"
	"os"
	"strings"
	"testing"
)

// BenchmarkEncode_singleLong drives a single Model.Encode on real weights at a
// long-ish sequence — the latency path where per-head QKᵀ parallelism (§1.3)
// would matter. Run with -cpuprofile to read the fraction of time actually spent
// in the per-head attention loop vs the big weight matmuls:
//
//	go test ./encoder/ -run xxx -bench BenchmarkEncode_singleLong -cpuprofile /tmp/enc.prof
//	go tool pprof -top -nodecount=25 /tmp/enc.prof
//	go tool pprof -list selfAttention /tmp/enc.prof
func BenchmarkEncode_singleLong(b *testing.B) {
	if _, err := os.Stat(modelDir); errors.Is(err, fs.ErrNotExist) {
		b.Skipf("no model at %s", modelDir)
	}
	m, err := Load(modelDir)
	if err != nil {
		b.Fatal(err)
	}
	// A few hundred tokens of code — a realistic rerank passage length, long
	// enough that the L²·D attention term is non-trivial relative to L·D².
	text := strings.Repeat("func scale(x, y int) int { return x*y + add(x, y) } // helper line\n", 48)
	b.ResetTimer()
	for range b.N {
		if _, err := m.Encode(text, false); err != nil {
			b.Fatal(err)
		}
	}
}
