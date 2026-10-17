package encoder

import (
	"fmt"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
)

// WordEmbPrecision is how a BERT or GTE model holds its word-embedding table (memory campaign items 9 and 9b). The
// table is only ever gathered by row — one row per input token, never multiplied as a matrix — so it can stay in a
// narrow or mapped form and be widened one row at a time; it is 45–69% of these models' weights.
type WordEmbPrecision uint8

const (
	// WordEmbNative, the default, keeps the table in the checkpoint's own precision, read from the mapping: an F32
	// table is a view into it, and an F16 or BF16 table is a 16-bit view widened one row per token. Either way the
	// table adds no heap, and only the rows a workload touches become resident, as clean file pages. Outputs are
	// bit-identical to widening the whole table to f32 (the pre-v1.71.0 load). When 16-bit data is not 2-byte
	// aligned in the file, the load widens it to f32 on the heap instead, as before.
	WordEmbNative WordEmbPrecision = iota
	// WordEmbF16 copies the table onto the heap as IEEE binary16 (an F16 checkpoint's table is already that, and
	// stays a mapped view). Outputs match f32's to a cosine of 1.0000000 on every measured fixture.
	WordEmbF16
	// WordEmbInt8 copies the table onto the heap as symmetric int8 with one f32 scale per row
	// (linalg.QuantizeRowInt8), about a quarter of the f32 bytes. Min cosine against f32 0.99993–0.999997 measured.
	WordEmbInt8
)

func (p WordEmbPrecision) String() string {
	switch p {
	case WordEmbNative:
		return "native"
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
	// WordEmb is how the word-embedding table is held: WordEmbNative (the zero value), WordEmbF16 or WordEmbInt8.
	// See WordEmbPrecision; docs/internal/task-memory-campaign.md (items 9, 9b) has the measurements.
	WordEmb WordEmbPrecision
}

func (o LoadOptions) check() error {
	if o.WordEmb > WordEmbInt8 {
		return fmt.Errorf("encoder: unknown word-embedding precision %v", o.WordEmb)
	}
	return nil
}

// wordTable is a word-embedding table held other than as one f32 slice ([vocab, dim] row-major): a 16-bit view into
// the mapping (f16 or bf16, mapped), or a heap copy at f16 or int8.
type wordTable struct {
	f16    []uint16 // binary16 patterns
	bf16   []uint16 // bfloat16 patterns
	q8     []int8
	scales []float32
	dim    int
	mapped bool // f16/bf16 are views into the checkpoint's mapping, not heap
}

// mappedWordTable returns the table at name as a mapped 16-bit view when the checkpoint stores it as F16 or BF16 of
// shape [vocab, dim] and its data is 2-byte aligned in the file; nil otherwise, and the caller loads it as f32.
func mappedWordTable(st *embed.SafetensorsFile, name string, vocab, dim int) *wordTable {
	t, err := st.Tensor(name)
	if err != nil || (t.DType != "F16" && t.DType != "BF16") || len(t.Shape) != 2 || t.Shape[0] != vocab || t.Shape[1] != dim {
		return nil
	}
	u, err := t.Uint16s()
	if err != nil || len(u) != vocab*dim {
		return nil // not 2-byte aligned: the caller's widened copy is the fallback
	}
	wt := &wordTable{dim: dim, mapped: true}
	if t.DType == "F16" {
		wt.f16 = u
	} else {
		wt.bf16 = u
	}
	return wt
}

// newWordTable copies an f32 table onto the heap at p (WordEmbF16 or WordEmbInt8).
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
		return nil, fmt.Errorf("encoder: no heap table for word-embedding precision %v", p)
	}
	return t, nil
}

// rowInto writes token id's row, widened to f32, into dst[:dim]. The 16-bit widen is the scalar linalg.F16ToF32 /
// BF16ToF32, which decode every bit pattern exactly as embed's Float16sToF32 / BFloat16sToF32 do
// (TestWordTable_decodeMatchesEmbedExhaustively), so a mapped table gives the widened table's outputs bit for bit.
func (t *wordTable) rowInto(id int, dst []float32) {
	d := t.dim
	switch {
	case t.f16 != nil:
		for j, h := range t.f16[id*d : (id+1)*d] {
			dst[j] = linalg.F16ToF32(h)
		}
	case t.bf16 != nil:
		for j, h := range t.bf16[id*d : (id+1)*d] {
			dst[j] = linalg.BF16ToF32(h)
		}
	default:
		linalg.DequantizeRowInt8(t.q8[id*d:(id+1)*d], t.scales[id], dst[:d])
	}
}

// f32 returns the whole table widened to f32 (a heap copy), for converting a mapped 16-bit table to an opt-in form.
func (t *wordTable) f32(vocab int) []float32 {
	out := make([]float32, vocab*t.dim)
	for id := range vocab {
		t.rowInto(id, out[id*t.dim:(id+1)*t.dim])
	}
	return out
}

// heapBytes is the heap the table holds (0 for a mapped view).
func (t *wordTable) heapBytes() int {
	if t.mapped {
		return 0
	}
	return 2*len(t.f16) + 2*len(t.bf16) + len(t.q8) + 4*len(t.scales)
}

// wordEmbHeapBytes is the heap the model's word-embedding table holds: 0 for a mapped table (F32 view or 16-bit
// view), the copy's size otherwise (a widened f32 table, or an opt-in f16/int8 one).
func (b *BERT) wordEmbHeapBytes() int {
	if b.wordQ != nil {
		return b.wordQ.heapBytes()
	}
	if b.wordEmbMapped {
		return 0
	}
	return 4 * len(b.wordEmb)
}

// wordEmbHeapBytes is BERT.wordEmbHeapBytes for a GTE model.
func (g *GTE) wordEmbHeapBytes() int {
	if g.wordQ != nil {
		return g.wordQ.heapBytes()
	}
	if g.wordEmbMapped {
		return 0
	}
	return 4 * len(g.wordEmb)
}

// loadWordEmb loads the table at name for a model's loader: a mapped 16-bit view when native is set and the
// checkpoint allows it, else f32 through get (a view for an aligned F32 checkpoint, a widened heap copy otherwise).
// mapped reports whether the f32 slice is a view into the mapping.
func loadWordEmb(st *embed.SafetensorsFile, name string, vocab, dim int, native bool, get func(string, ...int) []float32) (f32 []float32, q *wordTable, mapped bool) {
	if native {
		if q := mappedWordTable(st, name, vocab, dim); q != nil {
			return nil, q, false
		}
	}
	if t, err := st.Tensor(name); err == nil && t.DType == "F32" {
		mapped = true // TensorF32 aliases an aligned F32 tensor
	}
	return get(name, vocab, dim), nil, mapped
}

// narrow converts a loaded table to the opt-in heap form p. f32 is the loaded f32 table (nil when q is a mapped
// view). It returns the new table, or q itself when q is already in that form.
func narrow(f32 []float32, q *wordTable, vocab, dim int, p WordEmbPrecision) (*wordTable, error) {
	if p == WordEmbF16 && q != nil && q.f16 != nil {
		return q, nil // an F16 checkpoint's mapped table is already f16
	}
	if f32 == nil {
		f32 = q.f32(vocab)
	}
	return newWordTable(f32, vocab, dim, p)
}

// LoadBERTWithOptions is LoadBERT with load-time choices; LoadOptions{} loads exactly what LoadBERT does. An opt-in
// heap table is converted from the loaded one, whose mapped pages are then released (it is read once). SPLADE
// (LoadSPLADE) ties its decoder to the table as a matrix and always loads it as f32.
func LoadBERTWithOptions(dir string, o LoadOptions) (*BERT, error) {
	if err := o.check(); err != nil {
		return nil, err
	}
	b, err := loadBERT(dir, true)
	if err != nil || o.WordEmb == WordEmbNative {
		return b, err
	}
	q, err := narrow(b.wordEmb, b.wordQ, b.cfg.VocabSize, b.cfg.Hidden, o.WordEmb)
	if err != nil {
		_ = b.Close() //nolint:errcheck // a failed load: the conversion error is the one to report
		return nil, err
	}
	if q != b.wordQ {
		b.wordQ, b.wordEmb, b.wordEmbMapped = q, nil, false
		releaseWordEmbPages(b.st)
	}
	return b, nil
}

// LoadGTEWithOptions is LoadGTE with load-time choices; LoadOptions{} loads exactly what LoadGTE does. See
// LoadBERTWithOptions.
func LoadGTEWithOptions(dir string, o LoadOptions) (*GTE, error) {
	if err := o.check(); err != nil {
		return nil, err
	}
	g, err := loadGTE(dir, true)
	if err != nil || o.WordEmb == WordEmbNative {
		return g, err
	}
	q, err := narrow(g.wordEmb, g.wordQ, g.cfg.VocabSize, g.cfg.Hidden, o.WordEmb)
	if err != nil {
		_ = g.Close() //nolint:errcheck // a failed load: the conversion error is the one to report
		return nil, err
	}
	if q != g.wordQ {
		g.wordQ, g.wordEmb, g.wordEmbMapped = q, nil, false
		releaseWordEmbPages(g.st)
	}
	return g, nil
}

// releaseWordEmbPages drops the mapped pages of the word-embedding table once it has been converted. Advisory
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
