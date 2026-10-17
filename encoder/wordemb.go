package encoder

import (
	"fmt"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
)

// WordEmbPrecision is how a BERT or GTE model stores its word-embedding table (memory campaign item 9). The table is
// only ever gathered by row — one row per input token, never multiplied as a matrix — so a narrower storage costs one
// row's decode per token and nothing else, and it is 45–69% of these models' weights (bge-m3: 1.02 of 2.27 GB).
type WordEmbPrecision uint8

const (
	// WordEmbF32 keeps the checkpoint's table as loaded (for an F32 checkpoint, a view into its mapping). The default.
	WordEmbF32 WordEmbPrecision = iota
	// WordEmbF16 stores the table as IEEE binary16 on the heap: half the f32 bytes.
	WordEmbF16
	// WordEmbInt8 stores each row as symmetric int8 with one f32 scale (linalg.QuantizeRowInt8): about a quarter of
	// the f32 bytes.
	WordEmbInt8
)

func (p WordEmbPrecision) String() string {
	switch p {
	case WordEmbF32:
		return "f32"
	case WordEmbF16:
		return "f16"
	case WordEmbInt8:
		return "int8"
	}
	return fmt.Sprintf("WordEmbPrecision(%d)", uint8(p))
}

// LoadOptions are the load-time choices of LoadBERTWithOptions and LoadGTEWithOptions. The zero value loads exactly
// what LoadBERT / LoadGTE load.
type LoadOptions struct {
	// WordEmb is the word-embedding table's storage. Anything but WordEmbF32 changes the outputs slightly; see
	// WordEmbPrecision and docs/internal/task-memory-campaign.md (item 9) for the measured cosine against f32.
	WordEmb WordEmbPrecision
}

// wordTable is a word-embedding table stored at a precision other than f32 ([vocab, dim] row-major).
type wordTable struct {
	f16    []uint16
	q8     []int8
	scales []float32
	dim    int
}

// newWordTable converts an f32 table to p (not WordEmbF32).
func newWordTable(f32 []float32, vocab, dim int, p WordEmbPrecision) (*wordTable, error) {
	t := &wordTable{dim: dim}
	switch p {
	case WordEmbF16:
		t.f16 = make([]uint16, vocab*dim)
		linalg.F32ToF16Slice(t.f16, f32)
	case WordEmbInt8:
		t.q8 = make([]int8, vocab*dim)
		t.scales = make([]float32, vocab)
		for r := range vocab {
			t.scales[r] = linalg.QuantizeRowInt8(f32[r*dim:(r+1)*dim], t.q8[r*dim:(r+1)*dim])
		}
	default:
		return nil, fmt.Errorf("encoder: unknown word-embedding precision %v", p)
	}
	return t, nil
}

// rowInto writes token id's row, widened to f32, into dst[:dim].
func (t *wordTable) rowInto(id int, dst []float32) {
	d := t.dim
	if t.f16 != nil {
		linalg.F16ToF32Slice(dst[:d], t.f16[id*d:(id+1)*d])
		return
	}
	linalg.DequantizeRowInt8(t.q8[id*d:(id+1)*d], t.scales[id], dst[:d])
}

// bytes is the table's storage size.
func (t *wordTable) bytes() int { return 2*len(t.f16) + len(t.q8) + 4*len(t.scales) }

// wordEmbBytes is the size of the model's word-embedding table as stored: vocab × hidden × 4 for WordEmbF32 (bytes
// in the checkpoint's mapping, for an F32 checkpoint), less for the narrower precisions (bytes on the heap).
func (b *BERT) wordEmbBytes() int {
	if b.wordQ != nil {
		return b.wordQ.bytes()
	}
	return 4 * len(b.wordEmb)
}

// wordEmbBytes is BERT.wordEmbBytes for a GTE model.
func (g *GTE) wordEmbBytes() int {
	if g.wordQ != nil {
		return g.wordQ.bytes()
	}
	return 4 * len(g.wordEmb)
}

// LoadBERTWithOptions is LoadBERT with load-time choices; LoadOptions{} loads exactly what LoadBERT does. The
// table is converted from the loaded f32 one, whose mapped pages are then released (it is read once). SPLADE
// (LoadSPLADE) ties its decoder to the f32 table and keeps loading through LoadBERT.
func LoadBERTWithOptions(dir string, o LoadOptions) (*BERT, error) {
	if err := o.check(); err != nil {
		return nil, err
	}
	b, err := LoadBERT(dir)
	if err != nil || o.WordEmb == WordEmbF32 {
		return b, err
	}
	if b.wordQ, err = newWordTable(b.wordEmb, b.cfg.VocabSize, b.cfg.Hidden, o.WordEmb); err != nil {
		_ = b.Close() //nolint:errcheck // a failed load: the conversion error is the one to report
		return nil, err
	}
	b.wordEmb = nil
	releaseWordEmbPages(b.st)
	return b, nil
}

// LoadGTEWithOptions is LoadGTE with load-time choices; LoadOptions{} loads exactly what LoadGTE does. See
// LoadBERTWithOptions.
func LoadGTEWithOptions(dir string, o LoadOptions) (*GTE, error) {
	if err := o.check(); err != nil {
		return nil, err
	}
	g, err := LoadGTE(dir)
	if err != nil || o.WordEmb == WordEmbF32 {
		return g, err
	}
	if g.wordQ, err = newWordTable(g.wordEmb, g.cfg.VocabSize, g.cfg.Hidden, o.WordEmb); err != nil {
		_ = g.Close() //nolint:errcheck // a failed load: the conversion error is the one to report
		return nil, err
	}
	g.wordEmb = nil
	releaseWordEmbPages(g.st)
	return g, nil
}

func (o LoadOptions) check() error {
	if o.WordEmb > WordEmbInt8 {
		return fmt.Errorf("encoder: unknown word-embedding precision %v", o.WordEmb)
	}
	return nil
}

// releaseWordEmbPages drops the mapped pages of the f32 word-embedding table once it has been converted. Advisory
// (see SafetensorsFile.ReleaseTensors): the name is whichever of the BERT-family spellings the checkpoint uses.
func releaseWordEmbPages(st *embed.SafetensorsFile) {
	if st == nil {
		return
	}
	for _, p := range []string{"", "bert.", "roberta."} {
		if st.ReleaseTensors(p+"embeddings.word_embeddings.weight") == nil {
			return
		}
	}
}
