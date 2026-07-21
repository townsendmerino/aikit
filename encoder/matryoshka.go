package encoder

import "strings"

// Matryoshka (MRL) truncation registry — the exported twin of docs/embedder-coverage.md's
// Truncatable column, for callers that must decide whether shortening a vector is legitimate.
//
// Only models trained with Matryoshka Representation Learning may have their embeddings sliced.
// Doing it to any other model returns a unit-length vector that looks completely fine and simply
// RETRIEVES WORSE — a silent-wrong. That is measured, not asserted: in
// TestEmbedderCoverage_matryoshka, slicing multilingual-e5-base to a quarter width drops
// paraphrase-pair recall 1.00 → 0.80, while genuine MRL models hold their documented floor
// (nomic-embed-text-v1.5 1.00 → 0.90 at 768→64; nomic-embed-text-v2-moe 0.90 → 0.90 at 768→256).
//
// A serve layer honoring an OpenAI-style `dimensions` request therefore cannot do it blindly; this
// is the lookup that lets it refuse instead. Kept in lockstep with the coverage registry (and so
// with the published table) by TestMatryoshkaFloors_matchCoverage.
var matryoshkaFloors = map[string]int{
	"nomic-ai/nomic-embed-text-v1.5":   64,
	"nomic-ai/nomic-embed-text-v2-moe": 256,
}

// matryoshkaBare indexes the same rows by bare model name, since a serve layer usually knows a
// local directory ("…/models/nomic-embed-text-v1.5"), not the org-qualified HF id. Built in init;
// TestMatryoshkaFloors_matchCoverage asserts the bare names stay unambiguous.
var matryoshkaBare = map[string]int{}

func init() {
	for id, min := range matryoshkaFloors {
		matryoshkaBare[bareModelName(id)] = min
	}
}

func bareModelName(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	s = strings.TrimRight(s, "/")
	if i := strings.LastIndex(s, "/"); i >= 0 {
		return s[i+1:]
	}
	return s
}

// MatryoshkaFloor reports the smallest dimension model's embeddings may be truncated to.
//
// ok=false means the model is NOT known to be Matryoshka-trained, and its vectors must not be
// shortened at all — refuse the request rather than return a quietly degraded one. Unknown models
// report the same way on purpose: an embedder nobody measured is not one to truncate on a guess.
//
// model may be the HF id ("nomic-ai/nomic-embed-text-v1.5") or a path/directory whose last element
// is the model name ("/models/nomic-embed-text-v1.5/"); matching is case-insensitive.
func MatryoshkaFloor(model string) (min int, ok bool) {
	key := strings.TrimRight(strings.ToLower(strings.TrimSpace(model)), "/")
	if key == "" {
		return 0, false
	}
	if v, hit := matryoshkaFloors[key]; hit {
		return v, true
	}
	if v, hit := matryoshkaBare[bareModelName(key)]; hit {
		return v, true
	}
	return 0, false
}
