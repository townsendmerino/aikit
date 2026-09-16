package regex

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/chunk"
)

// loadFixture reads a real source file from this repo, relative to the repo
// root rather than to this package.
//
// Both fixture paths here were wrong, and both failed silently
// (perf-campaign item 1). `../../search/index.go` is a leftover `ken` path —
// aikit has no search/ directory. `../../../testdata/repo/…` resolves from
// <root>/chunk/regex to the PARENT OF THE REPO ROOT, so it could never have
// matched even in a complete checkout. b.Skipf turned both into passing
// benchmarks, leaving the default regex chunker with no live coverage at all.
//
// It now FAILS on a missing fixture: these are files checked into this
// repository, so absence means the benchmark is broken, not that the
// environment is unusual.
func loadFixture(b *testing.B, rel ...string) []byte {
	b.Helper()
	path := filepath.Join(append([]string{"..", ".."}, rel...)...)
	data, err := os.ReadFile(path)
	if err != nil {
		b.Fatalf("in-repo fixture %s is missing: %v", path, err)
	}
	return data
}

// typescriptFixture builds a representative TypeScript source. Unlike Go and
// Python there is no .ts file checked into this repository, and the chunker
// splits on syntax — so this is generated with real structure (imports,
// interfaces, classes with methods, exported functions) rather than borrowed
// from a corpus the chunker benchmarks do not otherwise depend on. Documented
// as synthetic because it is.
func typescriptFixture(n int) []byte {
	var b strings.Builder
	b.WriteString("import { Widget } from './widget';\nimport type { Config } from './config';\n\n")
	for i := range n {
		fmt.Fprintf(&b, `
export interface Shape%d {
  id: string;
  size: number;
}

export class Renderer%d {
  private cache: Map<string, Shape%d> = new Map();

  constructor(private readonly cfg: Config) {}

  public render(shape: Shape%d): string {
    if (this.cache.has(shape.id)) {
      return this.cache.get(shape.id)!.id;
    }
    this.cache.set(shape.id, shape);
    return shape.id;
  }
}

export function makeShape%d(id: string, size: number): Shape%d {
  return { id, size };
}
`, i, i, i, i, i, i)
	}
	return []byte(b.String())
}

func benchChunk(b *testing.B, lang string, source []byte) {
	c := New()
	b.SetBytes(int64(len(source)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := c.Chunk(source, lang, chunk.DefaultChunkSize)
		if err != nil {
			b.Fatalf("Chunk: %v", err)
		}
	}
}

func BenchmarkChunker_Go(b *testing.B) {
	benchChunk(b, "go", loadFixture(b, "ann", "hnsw.go"))
}

func BenchmarkChunker_TypeScript(b *testing.B) {
	benchChunk(b, "typescript", typescriptFixture(40))
}

func BenchmarkChunker_Python(b *testing.B) {
	benchChunk(b, "python", loadFixture(b, "scripts", "oracle", "encoder_model.py"))
}
