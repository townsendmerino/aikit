package tokenizer

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"unicode/utf8"
)

var errNotImplemented = errors.New("tokenizer: not implemented (see docs/gemma-decoder-plan.md §9, M2)")

// spaceMarker is SentencePiece's "▁" (U+2581 LOWER ONE EIGHTH BLOCK), which
// Gemma's normalizer substitutes for every ASCII space before BPE.
const spaceMarker = "▁"

// SpecialTokens holds the Gemma chat/control token ids resolved from the
// vocab at load time. The generation loop needs BOS/EOS; the chat template
// needs the turn markers.
type SpecialTokens struct {
	BOS         int // <bos>
	EOS         int // <eos>
	Pad         int // <pad>
	StartOfTurn int // <start_of_turn>
	EndOfTurn   int // <end_of_turn>
}

// bigram is a comparable map key for the ordered BPE merge table: the pair
// (left, right) → its merge rank (lower = higher priority, merged first).
type bigram struct{ left, right string }

// Tokenizer is a loaded Gemma 3 byte-fallback BPE tokenizer.
//
// Despite the file name, Gemma 3 ships a *BPE* model (with an explicit ordered
// merge table) rather than a unigram one — we load it from the HF
// tokenizer.json. The pipeline mirrors HF `tokenizers` exactly so ids match
// the M2 golden: split out added/special tokens (longest match on the raw
// text), normalize ASCII space → ▁ in the gaps, then BPE each gap with
// per-rune byte-fallback for out-of-vocab runes.
type Tokenizer struct {
	vocab     map[string]int32 // piece → id
	idToPiece []string         // id → piece (len == vocab size)
	pairRank  map[bigram]int32 // BPE merge rank
	special   SpecialTokens

	byteFallback bool
	bytePiece    [256]string // b → "<0xNN>" piece (the byte-fallback tokens)
	byteToVal    map[int32]byte
	unkPiece     string
	unkID        int32

	added *addedTrie // added/special token surface forms → id
}

// Special returns the resolved special-token ids.
func (t *Tokenizer) Special() SpecialTokens { return t.special }

// --- tokenizer.json schema (only the fields we need) ---

type tokenizerJSON struct {
	AddedTokens []struct {
		ID      int32  `json:"id"`
		Content string `json:"content"`
	} `json:"added_tokens"`
	Model struct {
		Type         string           `json:"type"`
		ByteFallback bool             `json:"byte_fallback"`
		UnkToken     *string          `json:"unk_token"`
		Vocab        map[string]int32 `json:"vocab"`
		Merges       [][]string       `json:"merges"`
	} `json:"model"`
}

// Load reads a SentencePiece/BPE model. path may point directly at a
// tokenizer.json or at a directory containing one (e.g. the HF checkpoint
// dir). The legacy tokenizer.model (SP protobuf) is not supported — the HF
// tokenizer.json carries the same vocab plus the explicit merge table and
// pipeline, which is what we match for parity.
func Load(path string) (*Tokenizer, error) {
	jsonPath := path
	if fi, err := os.Stat(path); err == nil && fi.IsDir() {
		jsonPath = filepath.Join(path, "tokenizer.json")
	}
	raw, err := os.ReadFile(jsonPath)
	if err != nil {
		return nil, fmt.Errorf("tokenizer.Load: %w", err)
	}
	var tj tokenizerJSON
	if err := json.Unmarshal(raw, &tj); err != nil {
		return nil, fmt.Errorf("tokenizer.Load: parse %s: %w", jsonPath, err)
	}
	if tj.Model.Type != "BPE" {
		return nil, fmt.Errorf("tokenizer.Load: unsupported model type %q (want BPE)", tj.Model.Type)
	}
	if len(tj.Model.Vocab) == 0 {
		return nil, fmt.Errorf("tokenizer.Load: empty vocab in %s", jsonPath)
	}

	t := &Tokenizer{
		vocab:        tj.Model.Vocab,
		pairRank:     make(map[bigram]int32, len(tj.Model.Merges)),
		byteFallback: tj.Model.ByteFallback,
		byteToVal:    make(map[int32]byte, 256),
	}

	// id → piece, sized to the max id so every produced id is renderable.
	maxID := int32(-1)
	for _, id := range tj.Model.Vocab {
		if id > maxID {
			maxID = id
		}
	}
	t.idToPiece = make([]string, maxID+1)
	for piece, id := range tj.Model.Vocab {
		t.idToPiece[id] = piece
	}

	// Merge ranks: position in the list is the priority.
	for i, m := range tj.Model.Merges {
		if len(m) != 2 {
			return nil, fmt.Errorf("tokenizer.Load: merge %d has %d parts, want 2", i, len(m))
		}
		t.pairRank[bigram{m[0], m[1]}] = int32(i)
	}

	// Byte-fallback tokens: "<0x00>".."<0xFF>".
	for b := 0; b < 256; b++ {
		p := fmt.Sprintf("<0x%02X>", b)
		t.bytePiece[b] = p
		if id, ok := t.vocab[p]; ok {
			t.byteToVal[id] = byte(b)
		} else if t.byteFallback {
			return nil, fmt.Errorf("tokenizer.Load: byte_fallback set but %q missing from vocab", p)
		}
	}

	// Unk + resolved specials.
	t.unkPiece = "<unk>"
	if tj.Model.UnkToken != nil {
		t.unkPiece = *tj.Model.UnkToken
	}
	mustID := func(piece string) (int32, error) {
		id, ok := t.vocab[piece]
		if !ok {
			return 0, fmt.Errorf("tokenizer.Load: required token %q not in vocab", piece)
		}
		return id, nil
	}
	if t.unkID, err = mustID(t.unkPiece); err != nil {
		return nil, err
	}
	for _, r := range []struct {
		piece string
		dst   *int
	}{
		{"<bos>", &t.special.BOS}, {"<eos>", &t.special.EOS}, {"<pad>", &t.special.Pad},
		{"<start_of_turn>", &t.special.StartOfTurn}, {"<end_of_turn>", &t.special.EndOfTurn},
	} {
		id, err := mustID(r.piece)
		if err != nil {
			return nil, err
		}
		*r.dst = int(id)
	}

	// Added-vocabulary trie: every added-token surface form is matched
	// (longest-first) against the raw text before normalization/BPE.
	t.added = newAddedTrie()
	for _, a := range tj.AddedTokens {
		t.added.add(a.Content, a.ID)
	}

	return t, nil
}

// Encode turns text into token ids. If addBOS, prepend the BOS token (the
// generation prefill expects it for Gemma). Added/special tokens written
// literally in the text are recognized and emitted as their own ids.
func (t *Tokenizer) Encode(text string, addBOS bool) ([]int, error) {
	if t.vocab == nil {
		return nil, fmt.Errorf("tokenizer.Encode: %w", errors.New("tokenizer not loaded"))
	}
	var out []int32
	if addBOS {
		out = append(out, int32(t.special.BOS))
	}

	gapStart := 0
	i := 0
	flushGap := func(end int) {
		if end > gapStart {
			out = append(out, t.bpe(normalize(text[gapStart:end]))...)
		}
	}
	for i < len(text) {
		if id, n := t.added.match(text, i); n > 0 {
			flushGap(i)
			out = append(out, id)
			i += n
			gapStart = i
			continue
		}
		_, sz := utf8.DecodeRuneInString(text[i:])
		if sz == 0 {
			sz = 1 // defensive: never stall on invalid UTF-8
		}
		i += sz
	}
	flushGap(len(text))

	res := make([]int, len(out))
	for k, v := range out {
		res[k] = int(v)
	}
	return res, nil
}

// normalize applies Gemma's sole normalizer: replace every ASCII space with
// the ▁ marker. (Tabs, newlines and other whitespace are left as-is — the
// added-vocabulary split handles the newline-run tokens.)
func normalize(s string) string {
	return strings.ReplaceAll(s, " ", spaceMarker)
}

// bpe segments a normalized gap into ids: per-rune initial symbols (byte
// fallback for out-of-vocab runes), then greedy merge of the lowest-rank
// adjacent pair (leftmost on ties) until none remain.
func (t *Tokenizer) bpe(gap string) []int32 {
	if gap == "" {
		return nil
	}
	syms := make([]string, 0, len(gap))
	for _, r := range gap {
		s := string(r)
		if _, ok := t.vocab[s]; ok {
			syms = append(syms, s)
			continue
		}
		if t.byteFallback {
			for _, b := range []byte(s) {
				syms = append(syms, t.bytePiece[b])
			}
			continue
		}
		syms = append(syms, t.unkPiece)
	}

	for len(syms) >= 2 {
		const maxRank = int32(1<<31 - 1)
		bestRank := maxRank
		bestI := -1
		for i := 0; i+1 < len(syms); i++ {
			if r, ok := t.pairRank[bigram{syms[i], syms[i+1]}]; ok && r < bestRank {
				bestRank = r
				bestI = i
			}
		}
		if bestI < 0 {
			break
		}
		syms[bestI] += syms[bestI+1]
		syms = append(syms[:bestI+1], syms[bestI+2:]...)
	}

	ids := make([]int32, len(syms))
	for k, s := range syms {
		if id, ok := t.vocab[s]; ok {
			ids[k] = id
		} else {
			ids[k] = t.unkID // unreachable under byte fallback
		}
	}
	return ids
}

// Decode turns token ids back into text: render each piece (with ▁ → space),
// fusing runs of byte-fallback pieces back into their raw UTF-8 bytes. Special
// tokens render as their literal surface form (e.g. "<eos>") — the generation
// loop is responsible for stopping at EOS, not Decode.
func (t *Tokenizer) Decode(ids []int) (string, error) {
	var sb strings.Builder
	var pending []byte
	flush := func() {
		if len(pending) > 0 {
			sb.Write(pending)
			pending = pending[:0]
		}
	}
	for _, id := range ids {
		if id < 0 || id >= len(t.idToPiece) {
			return "", fmt.Errorf("tokenizer.Decode: id %d out of range [0,%d)", id, len(t.idToPiece))
		}
		if b, ok := t.byteToVal[int32(id)]; ok {
			pending = append(pending, b)
			continue
		}
		flush()
		sb.WriteString(strings.ReplaceAll(t.idToPiece[id], spaceMarker, " "))
	}
	flush()
	return sb.String(), nil
}

// DecodePiece decodes a single id to its display string — used for token
// streaming so the demo can print as it goes. A lone byte-fallback piece may
// be an incomplete UTF-8 sequence; callers that stream should buffer across
// calls (a demo concern, not the tokenizer's).
func (t *Tokenizer) DecodePiece(id int) (string, error) {
	return t.Decode([]int{id})
}
