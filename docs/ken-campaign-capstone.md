# ken retrieval campaign — capstone

One page over the detailed plans and explainers. The arc of what we built, what we tested, what
we learned, and where the frontier is. Detail lives in: `ken-rerank-plan.md`,
`ken-query-understanding-plan.md`, `ken-docside-expansion-plan.md`, the `m0…m0d-results.md`
memos, and the two explainers (`six-retrieval-techniques-for-engineers.md`,
`what-we-tried-and-what-happened.md`).

ken is a pure-Go, no-cgo, single-binary code search tool. Everything below was done without ever
adding a C dependency or a required external API.

## The arc

**1. Foundation — hybrid retrieval + a pure-Go neural reranker.** ken retrieves by blending
keyword search (BM25) with meaning-based search (embeddings), fusing the two ranked lists, and
applying code-aware boosts. On top of that we shipped a second-stage **neural reranker**
(CodeRankEmbed) entirely in pure Go — including hand-written ARM64 NEON assembly and int8
quantization for speed. This is the single biggest quality lever in the system, worth roughly
**15× any later refinement**. It is the thing that makes ken good.

**2. The question — can understanding the query do better?** With the reranker shipped, we asked
whether bridging the "vocabulary gap" (user asks in English, code is written in code) could add
more. We investigated it the disciplined way, and the discipline mattered more than any single
technique.

**3. The query-side investigation — and its honest close.** We tested three ways to improve the
*query*: **HyDE** (generate a fake answer and search with it — a real but thin gain, and it needs
a generative model ken can't run cheaply in pure Go), an **encoder keyword predictor** (guess the
missing code-words — a large proven ceiling, but the cheap predictor couldn't reach it; dead), and
**PRF** (re-search using the first results' keywords — a feedback loop that hurt the hard queries;
dead). Net: the query side is closed. Worth knowing, not worth building.

**4. The doc-side win.** We flipped the attack: instead of fixing the query on every search, fix
the *documents* once at index time. A simple **heuristic enrichment** — prepend each code chunk
with AST-derived facts (`func: authenticate | class: Session | calls: verify_token`) — **won**:
it matched HyDE's gain for *free* (pure-Go, no model, no API, computed from code ken already
parses), with a statistically significant +0.010 lift and almost no collateral damage. It ships.

## What made it rigorous (the transferable part)

- **Only recall matters.** Because the reranker re-scores its shortlist from scratch, the sole job
  of any add-on is to drag a missing right-answer *into* the shortlist; reshuffling is already
  solved. This one fact killed several plausible ideas on contact.
- **Measure the ceiling before building.** An "oracle" that cheats by reading the known answers
  told us the maximum any technique could achieve *before* we spent a week on the real version.
  It greenlit investigations and killed builds, both cheaply.
- **Distrust a flat benchmark.** Our first "nothing helps" result was a rigged test with the
  answers leaked into the documents. Fixing the benchmark revealed the real signal.
- **The constraint is a feature.** "Pure Go, no API" repeatedly forced us toward the cheap,
  durable, self-contained solution — and that solution kept winning (the reranker; heuristic
  enrichment) over the expensive, model-dependent ones.

## Where we are

A strong, self-contained, pure-Go retrieval system: hybrid search + a neural reranker doing the
heavy lifting, plus a free doc-side enrichment win banked on top. Honest limit: there's a
measured surface of ~40 "recoverable" hard queries, and the free winner reaches about a quarter of
it. The rest is still open — and it isn't reachable by more query-rewriting or bigger models
(both ruled out).

## The frontier

The signpost is in the winner itself: the *structural* slivers in the heuristic (`calls:`,
`class:`, `module:`) already pull real weight. That points at the next chapter — **code structure
as signal**: symbol/definition resolution and call-graph expansion, exploiting the parse tree ken
already builds. It's deterministic, pure-Go, and aimed squarely at the queries embeddings are
*worst* at ("where is this defined," "what calls this").

And structure opens a second front beyond retrieval quality: a structural index can answer a whole
class of questions **exactly**, not by fuzzy ranking — turning ken from "better fuzzy search" into
"fuzzy search **plus** exact structural navigation." That may be a larger product win than any
remaining relevance tuning, and it reuses the same index the structural-signal work builds.

**Next:** ship the heuristic enrichment (and the AST-extraction layer it forces), confirm it on a
casual-query benchmark, then build the structural index once and use it twice — as a retrieval
signal and as a new class of exact-answer capability.
