// Package tokenizer implements the SentencePiece tokenizer Gemma 3 ships
// (a ~262k-token unigram/BPE model with byte-fallback). It is separate from
// embed.Tokenizer (WordPiece, for the Model2Vec/CodeRankEmbed encoders) —
// the algorithms and vocab format don't transfer.
//
// Status: SCAFFOLD. Load/Encode/Decode are stubs; the real work (M2 in
// docs/gemma-decoder-plan.md) is parsing the SentencePiece protobuf
// (tokenizer.model) or the HF tokenizer.json, the unigram Viterbi / BPE
// merge, byte-fallback for OOV, the ▁ whitespace marker, and Gemma's
// special tokens. Golden parity against HF's tokenizer is the M2 gate — a
// single-token drift silently degrades generation.
package tokenizer
