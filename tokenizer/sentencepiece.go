package tokenizer

import (
	"errors"
	"fmt"
)

var errNotImplemented = errors.New("tokenizer: not implemented (see docs/gemma-decoder-plan.md §9, M2)")

// SpecialTokens holds the Gemma chat/control token ids resolved from the
// vocab at load time. The generation loop needs BOS/EOS; the chat template
// needs the turn markers.
type SpecialTokens struct {
	BOS          int // <bos>
	EOS          int // <eos>
	Pad          int // <pad>
	StartOfTurn  int // <start_of_turn>
	EndOfTurn    int // <end_of_turn>
}

// Tokenizer is a loaded SentencePiece model.
//
// SCAFFOLD: fields the real implementation will populate are noted; the
// methods return errNotImplemented.
type Tokenizer struct {
	// pieces   []string            // id → piece
	// scores   []float32           // unigram log-probs
	// byteToken [256]int           // byte-fallback ids
	special SpecialTokens
}

// Special returns the resolved special-token ids.
func (t *Tokenizer) Special() SpecialTokens { return t.special }

// Load reads a SentencePiece model. path may point at a tokenizer.model
// (SP protobuf) or a directory containing tokenizer.json (HF format) — M2
// decides which to support first (tokenizer.json is the easier JSON path).
func Load(path string) (*Tokenizer, error) {
	_ = path
	return nil, fmt.Errorf("tokenizer.Load: %w", errNotImplemented)
}

// Encode turns text into token ids. If addBOS, prepend the BOS token (the
// generation prefill expects it for Gemma).
func (t *Tokenizer) Encode(text string, addBOS bool) ([]int, error) {
	_ = text
	_ = addBOS
	return nil, fmt.Errorf("tokenizer.Encode: %w", errNotImplemented)
}

// Decode turns token ids back into text, stripping the ▁ whitespace marker
// and handling byte-fallback pieces.
func (t *Tokenizer) Decode(ids []int) (string, error) {
	_ = ids
	return "", fmt.Errorf("tokenizer.Decode: %w", errNotImplemented)
}

// DecodePiece decodes a single id to its display string — used for token
// streaming so the demo can print as it goes.
func (t *Tokenizer) DecodePiece(id int) (string, error) {
	_ = id
	return "", fmt.Errorf("tokenizer.DecodePiece: %w", errNotImplemented)
}
