package embed

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"
)

// ──────────────────────────────────────────────────────────────────────────────
// Precompiled normalizer (SentencePiece charsmap)
// ──────────────────────────────────────────────────────────────────────────────
//
// XLM-R / bge-m3 / multilingual-e5 normalize with SentencePiece's "Precompiled"
// normalizer: a `precompiled_charsmap` blob holding a darts-clone double-array
// trie plus a pool of null-terminated normalized strings. Normalization iterates
// grapheme clusters (base + combining marks); each cluster (or, past 6 bytes, each
// char) is looked up via the trie's shortest-prefix match and replaced with its
// pool string (which may be empty — that's how control chars are deleted), or
// passed through unchanged when no rule matches. This mirrors HF spm_precompiled's
// normalize_string / transform exactly (see firstPrefix / normalize below) and
// reproduces its output byte-for-byte (verified in the norm oracle test against a
// per-codepoint sweep over U+0000..U+2FFFF plus combining sequences).
//
// Blob layout (little-endian): [u32 trieByteSize][trie u32 units][normalized pool].

type precompiled struct {
	array []uint32 // darts-clone double-array trie units
	pool  []byte   // normalized-string pool (null-terminated entries)
}

// newPrecompiled decodes a base64 precompiled_charsmap (as stored in
// tokenizer.json) into its trie + pool.
func newPrecompiled(b64 string) (*precompiled, error) {
	raw, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		return nil, fmt.Errorf("precompiled_charsmap: base64: %w", err)
	}
	if len(raw) < 4 {
		return nil, fmt.Errorf("precompiled_charsmap: too short (%d bytes)", len(raw))
	}
	trieBytes := int(binary.LittleEndian.Uint32(raw[:4]))
	if trieBytes%4 != 0 || 4+trieBytes > len(raw) {
		return nil, fmt.Errorf("precompiled_charsmap: bad trie size %d (total %d)", trieBytes, len(raw))
	}
	units := trieBytes / 4
	array := make([]uint32, units)
	off := 4
	for i := range units {
		array[i] = binary.LittleEndian.Uint32(raw[off : off+4])
		off += 4
	}
	return &precompiled{array: array, pool: raw[4+trieBytes:]}, nil
}

// darts-clone unit accessors (bit layout shared with SentencePiece / HF spm_precompiled).
func dartsHasLeaf(u uint32) bool { return (u>>8)&1 == 1 }
func dartsValue(u uint32) uint32 { return u & 0x7fffffff }
func dartsLabel(u uint32) uint32 { return u & (1<<31 | 0xff) }
func dartsOffset(u uint32) uint32 {
	return (u >> 10) << ((u & (1 << 9)) >> 6)
}

// firstPrefix walks the trie for key and returns the value of the FIRST (shortest)
// key that is a prefix of key. SentencePiece/HF `transform` deliberately takes
// results[0] — the shortest prefix match, not the longest ("Yes, this seems
// broken. No, I don't know why Google did this." — spm_precompiled). We reproduce
// that exactly. ok is false when no trie key prefixes key.
func (p *precompiled) firstPrefix(key []byte) (val uint32, ok bool) {
	if len(p.array) == 0 {
		return 0, false
	}
	nodePos := int(dartsOffset(p.array[0]))
	for i := 0; i < len(key); i++ {
		c := key[i]
		if c == 0 {
			break
		}
		nodePos ^= int(c)
		if nodePos < 0 || nodePos >= len(p.array) {
			return 0, false
		}
		unit := p.array[nodePos]
		if dartsLabel(unit) != uint32(c) {
			return 0, false
		}
		nodePos ^= int(dartsOffset(unit))
		if dartsHasLeaf(unit) {
			if nodePos < 0 || nodePos >= len(p.array) {
				return 0, false
			}
			return dartsValue(p.array[nodePos]), true
		}
	}
	return 0, false
}

// transform is HF spm_precompiled's transform: the shortest-prefix trie match's
// normalized replacement (which may be empty), or ("", false) if the chunk has no
// matching prefix. Note it replaces the WHOLE chunk with the (possibly shorter)
// match's value — bytes past the matched prefix are dropped, matching HF.
func (p *precompiled) transform(chunk []byte) (string, bool) {
	idx, ok := p.firstPrefix(chunk)
	if !ok {
		return "", false
	}
	if int(idx) >= len(p.pool) {
		return "", true
	}
	s := p.pool[idx:]
	if end := indexZero(s); end >= 0 {
		return string(s[:end]), true
	}
	return string(s), true
}

func indexZero(b []byte) int {
	for i, c := range b {
		if c == 0 {
			return i
		}
	}
	return -1
}

// isCombining reports whether r is a combining mark (Unicode Mn/Mc/Me). We use it
// to group a base scalar with its following marks into one cluster — the only
// multi-codepoint keys a SentencePiece NFKC charsmap holds are combining
// sequences, so this reproduces HF's UAX-29 grapheme walk for normalization
// without a segmentation dependency (verified byte-exact against the oracle).
func isCombining(r rune) bool {
	return unicode.In(r, unicode.Mn, unicode.Mc, unicode.Me)
}

// normalize reproduces HF spm_precompiled's normalize_string: iterate grapheme
// clusters (base + following combining marks); for a cluster under 6 bytes try to
// transform it whole, else transform each char and pass non-matching chars
// through unchanged.
func (p *precompiled) normalize(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for i := 0; i < len(s); {
		// Cluster = base rune + trailing combining marks.
		_, sz := utf8.DecodeRuneInString(s[i:])
		if sz == 0 {
			sz = 1
		}
		end := i + sz
		for end < len(s) {
			r, sz2 := utf8.DecodeRuneInString(s[end:])
			if !isCombining(r) {
				break
			}
			end += sz2
		}
		g := s[i:end]
		i = end

		if len(g) < 6 {
			if norm, ok := p.transform([]byte(g)); ok {
				b.WriteString(norm)
				continue
			}
		}
		for j := 0; j < len(g); {
			_, sz3 := utf8.DecodeRuneInString(g[j:])
			if sz3 == 0 {
				sz3 = 1
			}
			part := g[j : j+sz3]
			if norm, ok := p.transform([]byte(part)); ok {
				b.WriteString(norm)
			} else {
				b.WriteString(part)
			}
			j += sz3
		}
	}
	return b.String()
}

// ──────────────────────────────────────────────────────────────────────────────
// Unigram model (Viterbi)
// ──────────────────────────────────────────────────────────────────────────────
//
// Reproduces HF tokenizers' Unigram encode_optimized (which itself mirrors
// SentencePiece unigram_model.cc): a byte-position Viterbi over the vocab where
// each piece contributes its log-prob, unknown single chars cost
// min_score - K_UNK_PENALTY (10.0), and — with fuse_unk — consecutive unknown
// pieces collapse to one <unk>. byte_fallback is not used by XLM-R.

const unigramUnkPenalty = 10.0 // K_UNK_PENALTY in HF/SentencePiece

type unigram struct {
	piece2id map[string]int32
	scores   []float64 // score by id (vocab order)
	unkID    int32
	minScore float64
	maxBytes int // longest vocab piece in bytes — bounds the DP inner scan
	fuseUnk  bool
}

// viterbiIDs segments sentence (already normalized + metaspace-prefixed) into the
// max-log-prob path of vocab ids, matching HF encode_optimized exactly.
func (u *unigram) viterbiIDs(sentence string) []int32 {
	size := len(sentence)
	if size == 0 {
		return nil
	}
	unkScore := u.minScore - unigramUnkPenalty

	type node struct {
		id      int32
		score   float64
		startAt int
		reached bool
	}
	best := make([]node, size+1)
	best[0] = node{reached: true}

	for startAt := 0; startAt < size; {
		till := best[startAt].score
		reachedHere := best[startAt].reached
		_, mblen := utf8.DecodeRuneInString(sentence[startAt:])
		if mblen == 0 {
			mblen = 1
		}
		hasSingle := false
		if reachedHere {
			maxEnd := startAt + u.maxBytes
			if maxEnd > size {
				maxEnd = size
			}
			for end := startAt + 1; end <= maxEnd; end++ {
				id, ok := u.piece2id[sentence[startAt:end]]
				if !ok {
					continue
				}
				cand := u.scores[id] + till
				tn := &best[end]
				if !tn.reached || cand > tn.score {
					tn.reached = true
					tn.score = cand
					tn.startAt = startAt
					tn.id = id
				}
				if !hasSingle && end-startAt == mblen {
					hasSingle = true
				}
			}
			if !hasSingle {
				tn := &best[startAt+mblen]
				cand := unkScore + till
				if !tn.reached || cand > tn.score {
					tn.reached = true
					tn.score = cand
					tn.startAt = startAt
					tn.id = u.unkID
				}
			}
		}
		startAt += mblen
	}

	// Backtrack, then (fuse_unk) collapse runs of <unk> into one.
	var rev []int32
	for endsAt := size; endsAt > 0; {
		n := best[endsAt]
		if !n.reached { // defensive: unreachable tail (shouldn't happen)
			break
		}
		rev = append(rev, n.id)
		endsAt = n.startAt
	}
	ids := make([]int32, 0, len(rev))
	for i := len(rev) - 1; i >= 0; i-- {
		if u.fuseUnk && rev[i] == u.unkID && len(ids) > 0 && ids[len(ids)-1] == u.unkID {
			continue // fuse consecutive <unk>
		}
		ids = append(ids, rev[i])
	}
	return ids
}

// ──────────────────────────────────────────────────────────────────────────────
// Metaspace pre-tokenizer (WhitespaceSplit + ▁)
// ──────────────────────────────────────────────────────────────────────────────

const metaspace = '▁' // ▁ — SentencePiece's space marker

// metaspaceSplit reproduces the XLM-R / e5 pre_tokenizer Sequence[WhitespaceSplit,
// Metaspace(add_prefix_space=true)]: split the normalized text on Unicode
// whitespace (dropping it), then prepend ▁ to each resulting piece.
func metaspaceSplit(text string) []string {
	var out []string
	for _, field := range strings.FieldsFunc(text, unicode.IsSpace) {
		out = append(out, string(metaspace)+field)
	}
	return out
}

// metaspaceBare reproduces a BARE Metaspace(add_prefix_space=true) pre_tokenizer
// (bge-m3): replace every ASCII space with ▁, prepend a leading ▁ unless the text
// already begins with one, then split so each piece begins with ▁ (SentencePiece's
// MergedWithNext). Unlike metaspaceSplit it does NOT drop whitespace, so a lone ▁
// survives (e.g. a trailing space) — matching HF exactly (bge-m3 collapses runs of
// spaces in a preceding Replace normalizer, not here). Empty input yields nothing.
func metaspaceBare(text string) []string {
	if text == "" {
		return nil
	}
	var b strings.Builder
	b.Grow(len(text) + 3)
	for _, r := range text {
		if r == ' ' {
			b.WriteRune(metaspace)
		} else {
			b.WriteRune(r)
		}
	}
	s := b.String()
	if !strings.HasPrefix(s, string(metaspace)) {
		s = string(metaspace) + s
	}
	var out []string
	start := 0
	for i := 0; i < len(s); {
		r, sz := utf8.DecodeRuneInString(s[i:])
		if r == metaspace && i > start {
			out = append(out, s[start:i])
			start = i
		}
		i += sz
	}
	if start < len(s) {
		out = append(out, s[start:])
	}
	return out
}

// ──────────────────────────────────────────────────────────────────────────────
// Unigram tokenizer backend
// ──────────────────────────────────────────────────────────────────────────────

// unigramBackend is the SentencePiece/Unigram tokenizer (XLM-R family): the
// Precompiled normalizer, the Metaspace pre-tokenizer, the Unigram Viterbi model,
// added-token carve-out, and the TemplateProcessing specials. It plugs into the
// public embed.Tokenizer, which dispatches to it when tokenizer.json is Unigram.
type unigramBackend struct {
	norm            *precompiled
	replaces        []replaceRule // Replace normalizers applied after the charsmap (Sequence tail)
	whitespaceSplit bool          // pre_tokenizer splits on whitespace before Metaspace (XLM-R/e5) vs bare Metaspace (bge-m3)
	model           *unigram
	addedTokens     map[string]int32
	addedKeys       []string // longest-first carve-out scan order
	unkID           int32
	vocabSize       int
	prefixIDs       []int32 // TemplateProcessing single: specials before the sequence (<s>)
	suffixIDs       []int32 // ... and after (</s>)
}

// replaceRule is a Replace normalizer (regex pattern → literal content), e.g.
// bge-m3's " {2,}" → " " that collapses runs of spaces.
type replaceRule struct {
	re      *regexp.Regexp
	content string
}

// normalizeText runs the normalizer pipeline: the Precompiled charsmap, then any
// Replace rules, in Sequence order.
func (u *unigramBackend) normalizeText(text string) string {
	s := u.norm.normalize(text)
	for _, r := range u.replaces {
		s = r.re.ReplaceAllString(s, r.content)
	}
	return s
}

// preTokenize dispatches to the configured Metaspace variant.
func (u *unigramBackend) preTokenize(text string) []string {
	if u.whitespaceSplit {
		return metaspaceSplit(text)
	}
	return metaspaceBare(text)
}

// encode runs normalize → metaspace pre-tokenize → Unigram over the added-token
// carve-out, producing the BARE id sequence (no template specials).
func (u *unigramBackend) encode(text string) []int32 {
	if len(u.addedKeys) == 0 {
		return u.encodeSegment(text)
	}
	var out []int32
	var seg strings.Builder
	flush := func() {
		if seg.Len() > 0 {
			out = append(out, u.encodeSegment(seg.String())...)
			seg.Reset()
		}
	}
	for i := 0; i < len(text); {
		matched := ""
		for _, k := range u.addedKeys {
			if strings.HasPrefix(text[i:], k) {
				matched = k
				break
			}
		}
		if matched != "" {
			flush()
			out = append(out, u.addedTokens[matched])
			i += len(matched)
			continue
		}
		_, size := utf8.DecodeRuneInString(text[i:])
		if size == 0 {
			size = 1
		}
		seg.WriteString(text[i : i+size])
		i += size
	}
	flush()
	return out
}

func (u *unigramBackend) encodeSegment(text string) []int32 {
	normalized := u.normalizeText(text)
	var ids []int32
	for _, piece := range u.preTokenize(normalized) {
		ids = append(ids, u.model.viterbiIDs(piece)...)
	}
	return ids
}

// encodeWithSpecials wraps encode with the TemplateProcessing specials
// (prefixIDs ++ body ++ suffixIDs), truncating body from the right so the total
// is at most maxLen. For XLM-R this is <s> ++ body ++ </s>.
func (u *unigramBackend) encodeWithSpecials(text string, maxLen int) []int32 {
	fixed := len(u.prefixIDs) + len(u.suffixIDs)
	if maxLen < fixed {
		maxLen = fixed
	}
	body := u.encode(text)
	if len(body) > maxLen-fixed {
		body = body[:maxLen-fixed]
	}
	out := make([]int32, 0, len(body)+fixed)
	out = append(out, u.prefixIDs...)
	out = append(out, body...)
	out = append(out, u.suffixIDs...)
	return out
}

// ──────────────────────────────────────────────────────────────────────────────
// tokenizer.json → unigramBackend
// ──────────────────────────────────────────────────────────────────────────────

// unigramJSON is the tokenizer.json shape for a Unigram/SentencePiece tokenizer
// (the fields the Precompiled + Metaspace + Unigram + TemplateProcessing pipeline
// needs). Vocab is a list of [piece, score] pairs; entry index is the token id.
type unigramJSON struct {
	AddedTokens []struct {
		ID      int32  `json:"id"`
		Content string `json:"content"`
		Special bool   `json:"special"`
	} `json:"added_tokens"`
	Normalizer   json.RawMessage `json:"normalizer"`
	PreTokenizer json.RawMessage `json:"pre_tokenizer"`
	Model        struct {
		Type  string            `json:"type"`
		UnkID *int32            `json:"unk_id"`
		Vocab []json.RawMessage `json:"vocab"`
	} `json:"model"`
	PostProcessor struct {
		Type          string                       `json:"type"`
		Single        []map[string]templateElement `json:"single"`
		SpecialTokens map[string]struct {
			IDs []int32 `json:"ids"`
		} `json:"special_tokens"`
	} `json:"post_processor"`
}

type templateElement struct {
	ID string `json:"id"`
}

// isUnigramTokenizer peeks at model.type / normalizer.type to decide whether the
// tokenizer.json is a Unigram/SentencePiece one (handled here) vs WordPiece
// (handled by the base parser). XLM-R omits model.type, so the Precompiled
// normalizer is the reliable tell.
func isUnigramTokenizer(model, normalizer string) bool {
	return model == "Unigram" || (model == "" && normalizer == "Precompiled")
}

// parseUnigramTokenizer builds a unigramBackend from tokenizer.json bytes.
func parseUnigramTokenizer(data []byte) (*unigramBackend, error) {
	var raw unigramJSON
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("parse tokenizer.json (unigram): %w", err)
	}
	if raw.Model.UnkID == nil {
		return nil, fmt.Errorf("unigram: model.unk_id missing")
	}
	charsmap, replaces, err := buildNormalizer(raw.Normalizer)
	if err != nil {
		return nil, err
	}
	norm, err := newPrecompiled(charsmap)
	if err != nil {
		return nil, err
	}
	whitespaceSplit, err := preTokWhitespaceSplit(raw.PreTokenizer)
	if err != nil {
		return nil, err
	}

	// Vocab: [piece, score] pairs; index is the id.
	n := len(raw.Model.Vocab)
	piece2id := make(map[string]int32, n)
	scores := make([]float64, n)
	minScore := 0.0
	maxBytes := 1
	for i, entry := range raw.Model.Vocab {
		var pair []json.RawMessage
		if err := json.Unmarshal(entry, &pair); err != nil || len(pair) != 2 {
			return nil, fmt.Errorf("unigram: vocab[%d] not a [piece,score] pair", i)
		}
		var piece string
		var score float64
		if err := json.Unmarshal(pair[0], &piece); err != nil {
			return nil, fmt.Errorf("unigram: vocab[%d] piece: %w", i, err)
		}
		if err := json.Unmarshal(pair[1], &score); err != nil {
			return nil, fmt.Errorf("unigram: vocab[%d] score: %w", i, err)
		}
		piece2id[piece] = int32(i)
		scores[i] = score
		if score < minScore {
			minScore = score
		}
		if len(piece) > maxBytes {
			maxBytes = len(piece)
		}
	}

	added := make(map[string]int32, len(raw.AddedTokens))
	for _, at := range raw.AddedTokens {
		added[at.Content] = at.ID
	}
	addedKeys := make([]string, 0, len(added))
	for k := range added {
		addedKeys = append(addedKeys, k)
	}
	sort.Slice(addedKeys, func(i, j int) bool {
		if len(addedKeys[i]) != len(addedKeys[j]) {
			return len(addedKeys[i]) > len(addedKeys[j])
		}
		return addedKeys[i] < addedKeys[j]
	})

	prefixIDs, suffixIDs := templateSpecials(raw.PostProcessor.Single, raw.PostProcessor.SpecialTokens)

	return &unigramBackend{
		norm:            norm,
		replaces:        replaces,
		whitespaceSplit: whitespaceSplit,
		model: &unigram{
			piece2id: piece2id,
			scores:   scores,
			unkID:    *raw.Model.UnkID,
			minScore: minScore,
			maxBytes: maxBytes,
			fuseUnk:  true,
		},
		addedTokens: added,
		addedKeys:   addedKeys,
		unkID:       *raw.Model.UnkID,
		vocabSize:   n,
		prefixIDs:   prefixIDs,
		suffixIDs:   suffixIDs,
	}, nil
}

// buildNormalizer walks a tokenizer.json normalizer (a bare Precompiled, or a
// Sequence of them) and returns the single charsmap plus any Replace rules to run
// after it. It supports exactly the SentencePiece normalizers the multilingual
// embedders use — Precompiled and Replace (regex → literal) — and errors on any
// other type so an unsupported normalizer fails loudly (→ best-effort nil in the
// loader) rather than silently mis-normalizing.
func buildNormalizer(raw json.RawMessage) (charsmap string, replaces []replaceRule, err error) {
	var n struct {
		Type                string            `json:"type"`
		PrecompiledCharsmap string            `json:"precompiled_charsmap"`
		Normalizers         []json.RawMessage `json:"normalizers"`
		Pattern             struct {
			Regex  string `json:"Regex"`
			String string `json:"String"`
		} `json:"pattern"`
		Content string `json:"content"`
	}
	if err = json.Unmarshal(raw, &n); err != nil {
		return "", nil, fmt.Errorf("unigram: normalizer: %w", err)
	}
	switch n.Type {
	case "Precompiled":
		return n.PrecompiledCharsmap, nil, nil
	case "Replace":
		pat := n.Pattern.Regex
		if pat == "" { // a literal-string Replace: match it verbatim
			pat = regexp.QuoteMeta(n.Pattern.String)
		}
		re, cerr := regexp.Compile(pat)
		if cerr != nil {
			return "", nil, fmt.Errorf("unigram: Replace pattern %q: %w", pat, cerr)
		}
		return "", []replaceRule{{re: re, content: n.Content}}, nil
	case "Sequence":
		for _, sub := range n.Normalizers {
			cm, rs, serr := buildNormalizer(sub)
			if serr != nil {
				return "", nil, serr
			}
			if cm != "" {
				if charsmap != "" {
					return "", nil, fmt.Errorf("unigram: multiple Precompiled normalizers")
				}
				charsmap = cm
			}
			replaces = append(replaces, rs...)
		}
		if charsmap == "" {
			return "", nil, fmt.Errorf("unigram: no Precompiled normalizer in Sequence")
		}
		return charsmap, replaces, nil
	default:
		return "", nil, fmt.Errorf("unigram: unsupported normalizer.type %q", n.Type)
	}
}

// preTokWhitespaceSplit reports whether the pre_tokenizer splits on whitespace
// before Metaspace (XLM-R/e5 Sequence[WhitespaceSplit, Metaspace]) vs a bare
// Metaspace (bge-m3). Errors on any other shape.
func preTokWhitespaceSplit(raw json.RawMessage) (bool, error) {
	var p struct {
		Type          string `json:"type"`
		Pretokenizers []struct {
			Type string `json:"type"`
		} `json:"pretokenizers"`
	}
	if err := json.Unmarshal(raw, &p); err != nil {
		return false, fmt.Errorf("unigram: pre_tokenizer: %w", err)
	}
	switch p.Type {
	case "Metaspace":
		return false, nil
	case "Sequence":
		hasWS, hasMeta := false, false
		for _, pt := range p.Pretokenizers {
			switch pt.Type {
			case "WhitespaceSplit":
				hasWS = true
			case "Metaspace":
				hasMeta = true
			default:
				return false, fmt.Errorf("unigram: unsupported pre_tokenizer %q in sequence", pt.Type)
			}
		}
		if !hasMeta {
			return false, fmt.Errorf("unigram: pre_tokenizer sequence lacks Metaspace")
		}
		return hasWS, nil
	default:
		return false, fmt.Errorf("unigram: unsupported pre_tokenizer.type %q", p.Type)
	}
}

// templateSpecials reads a TemplateProcessing "single" template into the special
// ids that come before (prefix) and after (suffix) the sequence — e.g. for XLM-R,
// prefix=[<s>], suffix=[</s>]. Unknown/empty templates yield no specials.
func templateSpecials(single []map[string]templateElement, specials map[string]struct {
	IDs []int32 `json:"ids"`
}) (prefix, suffix []int32) {
	seenSeq := false
	for _, el := range single {
		if _, ok := el["Sequence"]; ok {
			seenSeq = true
			continue
		}
		st, ok := el["SpecialToken"]
		if !ok {
			continue
		}
		ids := specials[st.ID].IDs
		if seenSeq {
			suffix = append(suffix, ids...)
		} else {
			prefix = append(prefix, ids...)
		}
	}
	return prefix, suffix
}
