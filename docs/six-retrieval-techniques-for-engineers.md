# Six retrieval techniques, explained for an engineer

These are the six techniques from the bottom of the landscape table — the ones **not** yet
tried in ken. This doc explains each in plain engineering terms: the problem it solves, how it
works (with analogies to things you already know), what it costs, and how well it fits ken's
hard constraint (pure Go, no cgo, single static binary).

First, four concepts several of these techniques lean on. If you already have these, skip to the
techniques.

---

## Background: four concepts

**Embedding (a "meaning fingerprint").** An embedding model is a pure function
`text -> []float32` that always returns a fixed-length array (say 768 floats). It's trained so
that texts with *similar meaning* produce arrays that are *close together*. Think of it as a
fingerprint — but the opposite design goal from a hash. A hash is built to *scatter* (avoid
collisions); an embedding is built to *cluster* (similar things land near each other). You
compare two embeddings with **cosine similarity** — basically the dot product of the two
normalized arrays, a number from -1 to 1, where higher = more alike. This is the machinery that
lets the query "log in" match code that says `authenticate` despite zero shared characters. ken
uses a small, fast embedding model (potion) for this.

**Lexical vs semantic, and α.** ken scores every chunk two ways: a **lexical** score (BM25 —
classic keyword/term overlap, like a search engine) and a **semantic** score (embedding cosine
similarity). It blends them: `combined = α·semantic + (1−α)·lexical`. **α is just the blend
knob** — today it's a fixed value (0.3 for identifier-looking queries, 0.5 for natural-language
ones).

**Reranking / two-stage retrieval.** A standard pattern, and you already use its shape in
databases. Stage 1: a cheap, high-recall pass over the *whole* corpus to get a shortlist (say
top 50). Stage 2: an expensive, high-precision model re-scores *just those 50* and reorders
them. It's an index scan to narrow the field, then an expensive predicate on the survivors. ken
already does this — the CodeRankEmbed model is its stage-2 reranker.

**Bi-encoder vs cross-encoder.** Two ways a model can compare a query to a document:
- A **bi-encoder** embeds the query and the document *separately* into two fingerprints, then
  compares the fingerprints. You can compute every document's fingerprint *ahead of time* and
  store it. At query time you only fingerprint the query and do nearest-neighbor. Fast, scalable
  — but the query and document never actually "meet" inside the model. (ken's reranker is a
  bi-encoder.)
- A **cross-encoder** feeds the query and the document in *together*, as one combined input, and
  the model emits a single relevance score. Because they're processed jointly, it can reason
  about how *this* query relates to *this* document — far more precise. The catch: nothing can be
  precomputed. You must run the whole model once for *every* (query, document) pair, so you can
  only afford it on a shortlist.

Analogy for the last one: a bi-encoder compares two records by their precomputed checksums. A
cross-encoder runs a full, expensive `compare(a, b)` on each pair that examines both deeply —
you'd only ever call it on a handful of candidates, never the whole table.

---

## 5. Definition / symbol resolution

**The problem.** When someone searches for a specific symbol — `parseConfig`, `RateLimiter` —
text and embedding search treat that name as just another word to fuzzy-match. But code isn't
prose: a symbol usually has *exactly one definition* and a known set of references. That
structure is hard fact, and the fuzzy matchers throw it away.

**How it works.** At index time, parse the code (ken already does, via tree-sitter) and build a
symbol table: `name -> where it's defined`, plus `name -> everywhere it's referenced`. When a
query names a symbol, don't just fuzzy-match it — *resolve* it: jump straight to its definition,
and expand the result set with the definition and its references.

**Analogy.** This is "Go to Definition" and "Find All References" from your IDE, used as a search
signal. It's `ctags` / LSP, repurposed for ranking. You already trust it every day in your
editor precisely because it's exact and deterministic, not statistical.

**Cost & fit.** Needs a symbol index built from the AST at index time — cheap, deterministic, no
model, no API. **Excellent pure-Go fit;** the tree-sitter parsing infrastructure is already
shipped. Low risk, mostly plumbing.

---

## 6. Adaptive α / intent routing

**The problem.** That α blend knob is fixed, so it's a compromise. A query that is literally a
function name should lean almost entirely on lexical (exact) matching; a vague conceptual
question should lean on semantic. One fixed value can't be right for both.

**How it works.** Classify the query first — is it a bare symbol? a natural-language question? a
"where is X used" question? — and then pick the blend weight, or the whole retrieval strategy,
to match. "Routing" just means: different query shapes go down different code paths. The
classifier can be a handful of regexes and rules (cheap, and roughly what ken does today) or a
small learned model (still cheap).

**Analogy.** It's a query planner, or a strategy-pattern dispatch. A database optimizer looks at
the query shape and table statistics and chooses index-scan vs full-scan; this looks at the
query shape and chooses lexical vs semantic vs structural. Concretely it's a `switch` on query
type that dispatches to the best-suited handler instead of running one averaged pipeline for
everything.

**Cost & fit.** The heuristic version is nearly free; a learned classifier is still small. **Good
pure-Go fit** — this is mostly engineering and control flow, low risk. The main work is deciding
the query taxonomy and measuring that the routing actually helps rather than just adding
branches.

---

## 7. Multi-query fusion (RAG-fusion)

**The problem.** A single phrasing can miss documents that a different phrasing would have
caught. The user types "stop people hammering the API"; the code says `rate_limit`; a third
phrasing, "throttle requests," might have matched best of all. You're betting everything on one
wording.

**How it works.** Use a generative model to produce several paraphrases of the query, run a
separate search for each paraphrase, then merge the ranked result lists. The merge uses
Reciprocal Rank Fusion (RRF) — the *same* merge primitive ken already uses to combine its BM25
and semantic result lists. Documents that surface across multiple phrasings bubble to the top.

**Analogy.** Scatter-gather with synonym-expanded variants. You fan out N reworded versions of
the search, gather the result sets, and UNION them with a voting/scoring scheme so the documents
that show up repeatedly win.

**Cost & fit.** Two costs. The cheap part — the RRF merge — ken already has. The expensive part
is generating the paraphrases, which needs a **generative model running at query time**. That's
the same "run a text-generating model inside the tool" problem that doesn't fit pure-Go/no-cgo
well, and the campaign found this whole family expensive for thin gains. **Poor fit** as a
result — not because the idea is bad, but because the generation step is the costly piece and
it's exactly the piece pure-Go struggles with.

---

## 8. Query decomposition / step-back

Two related moves, both using a generative model to rewrite the query before searching.

**Decomposition.** Break a complex, multi-part query into sub-queries, search each, combine.
"How does auth interact with the rate limiter?" becomes search("auth") + search("rate limiter"),
results merged.

**Step-back.** Generate a more *abstract* version of the query first, retrieve on that to get the
lay of the land, then go specific. "Why does function X throw on null input?" steps back to "how
does X validate its inputs," retrieves that general area, then narrows.

**Analogy.** Decomposition is breaking a gnarly SQL query with several joins into CTEs /
subqueries and combining their results. Step-back is what you do by hand when dropped into
unfamiliar code: zoom out to understand the module before you drill into the offending line.

**Cost & fit.** Both need a generative model at query time to do the splitting/abstracting — same
decoder cost and same poor pure-Go fit as multi-query fusion (#7). Deprioritized for the same
reason. The *concept* (especially decomposition for genuinely multi-part questions) is sound; the
*delivery mechanism* is the expensive part that doesn't suit the constraint.

---

## 9. Late-interaction (ColBERT-style)

This is the most involved of the six, so it gets a bit more room.

**The problem.** The normal semantic approach squashes an entire chunk into *one* fingerprint.
That's lossy. A 50-line function collapses to 768 numbers, and a query term that strongly matches
one specific line gets averaged out against everything else in the chunk. Fine-grained matches
get diluted by the surrounding bulk.

**How it works.** Instead of one fingerprint per chunk, keep one fingerprint *per token* (per
word/sub-word). At query time, for each *query* token, find its best-matching token anywhere in
the document (this is called **MaxSim** — maximum similarity), and add up those best matches as
the document's score. The matching happens at the token level, computed *late* (at query time),
instead of being pre-squashed into a single vector. Hence "late interaction."

**Analogy.** Comparing two files by a single checksum (today's single-vector approach) versus
doing a token-by-token diff between them — except the diff is in *meaning-space*, so `auth` can
match `authenticate` without an exact string match. It's like keeping a per-token inverted index
of meaning-fingerprints and doing a fine-grained semantic comparison, rather than one coarse
whole-document fingerprint.

**Cost & fit.** The cost is storage and query work. Instead of 768 floats per chunk you store 768
floats × the number of tokens — a 200-token chunk costs ~200× the vector storage. That's the
"heavy index." Scoring is also more expensive (token-against-token instead of vector-against-
vector). It is **feasible in pure Go** — it's just a lot more vectors and a different scoring loop,
no cgo required — but it's a meaningful build and a much bigger index. The payoff is a genuinely
higher accuracy ceiling; ColBERT-family retrievers are known to be strong. Filed under "big
lift, real reward."

---

## 10. Cross-encoder rerank (CodeRankLLM)

**The problem.** ken's current reranker (CodeRankEmbed) is a *bi-encoder* — it fingerprints the
query and each candidate separately and compares the fingerprints. Fast and precomputable, but
the query and document never interact inside the model, so it can't reason about how this exact
query relates to this exact document.

**How it works.** A *cross-encoder* feeds the query and a candidate document in *together* and
emits one relevance score, with the two interacting throughout the model's computation. That lets
it catch fine relationships ("the query is about error handling; this document's try/except on
line 12 is exactly that"). Because nothing can be precomputed, you run the full model once per
(query, candidate) pair — so it only runs on the shortlist, as an even-more-precise final rerank
stage. CodeRankLLM is such a cross-encoder, built to pair with CodeRankEmbed, but it is
**LLM-scale** — much larger than the bi-encoder ken shipped.

**Analogy.** Same two-stage pattern ken already uses — cheap retrieve, then expensive rerank — but
swapping the final stage for an even more expensive, even more thorough `compare(query, doc)`
that you can only afford on the top handful of candidates. Bi-encoder = compare by precomputed
checksums; cross-encoder = full deep pairwise comparison, per pair.

**Cost & fit.** It needs an LLM-scale model at query time, which is the same wall as running a
generative model: too big and slow for pure-Go on a CPU, effectively needs cgo/GPU. **Poor fit**
for ken's constraint, despite being the highest-accuracy option on paper. It's the "buy more
accuracy with a much bigger model" lever, and that lever is the one ken has deliberately chosen
not to pull.

---

## Summary

| # | Technique | Needs a model? | Runs at | Pure-Go fit | One-line take |
|---|---|---|---|---|---|
| 5 | Definition / symbol resolution | No (AST) | Query time | **Excellent** | "Go to Definition" as a ranking signal; deterministic, infra already there |
| 6 | Adaptive α / intent routing | Tiny / heuristic | Query time | **Good** | A query planner that picks lexical vs semantic per query shape |
| 7 | Multi-query fusion | Yes (generative) | Query time | Poor | Scatter-gather over reworded queries; merge step is free, generation isn't |
| 8 | Query decomposition / step-back | Yes (generative) | Query time | Poor | Split or zoom-out the query first; same generation cost as #7 |
| 9 | Late-interaction (ColBERT) | No at query (heavy index) | Both | Feasible, heavy | Per-token fingerprints + token-level semantic diff; big index, real accuracy gain |
| 10 | Cross-encoder rerank | Yes (LLM-scale) | Query time | Poor | Deep `compare(query, doc)` per pair; most accurate, needs a big model |

**The pattern.** The techniques that fit ken cleanly (5 and 6) are the *deterministic, code-aware*
ones — they exploit structure ken already parses and need no model at all. The ones that fit
poorly (7, 8, 10) all share one trait: they need a **large or generative model running at query
time**, which is precisely what pure-Go/no-cgo makes expensive. Late-interaction (9) is the
interesting middle — no generation, no cgo, just a much bigger index and a different scoring loop,
buying a real accuracy gain for a real engineering cost.

So if you're deciding where to spend next, the highest-value, lowest-risk bets are the two
deterministic code-structure ones (symbol resolution, routing), with late-interaction as the
"worth it if we want a real accuracy step-up and can afford the index" option. The generative-query
techniques are good ideas wearing the wrong cost profile for this particular tool.
