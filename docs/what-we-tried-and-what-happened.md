# What we tried, and what happened

Companion to *Six retrieval techniques, explained for an engineer*. That doc covered the
techniques we **haven't** tried. This one covers the ones we **did** — HyDE, the encoder
predictor, PRF, and heuristic enrichment — what each is, what happened when we benchmarked it,
and why. Written at the same level: experienced engineer, no AI background assumed.

(If you haven't read the companion's "four concepts" primer — embeddings, the α blend knob,
two-stage reranking, bi- vs cross-encoders — it's worth skimming first. This doc assumes them.)

---

## The shared problem: the vocabulary gap

Every technique here is attacking one thing. A user asks in English — "how do we stop people
hammering the API" — and the code is written in code: `RateLimiter`, `tokenBucket`,
`maxRequestsPerWindow`. There's almost no shared text, so keyword search (BM25) is blind to it,
and even meaning-based search only partly bridges it. Closing that gap — getting the right code
in front of a natural-language query — is the whole game.

There are two places you can attack it: change the **query** (make it look more like code) or
change the **documents** (make them look more like the query). Most of what we tried was
query-side; the thing that won was document-side.

---

## Three rules of the game (why the results came out as they did)

Before the techniques, three facts about ken's setup that determined every outcome. They're worth
understanding because they explain *why* good-sounding ideas failed and a humble one won.

**1. The reranker re-scores from scratch, so only *recall* matters.** ken already has a strong
second-stage reranker (a model called CodeRankEmbed) that takes the top ~50 candidates and
reorders them by relevance. Here's the key consequence: the reranker throws away the order it was
handed and re-scores those 50 from scratch. So a technique that merely *reshuffles* the top 50
gains nothing — the reranker overwrites the reshuffle. The *only* thing a technique can
contribute is **recall**: pulling a relevant document into the top-50 that wasn't there before.
Once it's in the shortlist, the reranker reliably floats it to the top on its own. So every
technique's real job reduces to: "get more right answers into the shortlist." Reshuffling is
already solved.

**2. The oracle: measure the ceiling before you build.** This is the smartest move in the whole
campaign and it's pure engineering discipline — like profiling before you optimize. Before
building a real predictor, we *cheated*: we looked at each query's known-correct answer, pulled
the distinctive keywords straight out of that answer, and injected them into the search. That
tells you the **ceiling** — "if our predictor were perfect, how much could this mechanism
possibly help?" If even the cheat doesn't move the needle, you stop immediately and never build
the real thing. We ran two versions: one allowing rare one-off keywords ("oracle-max"), and one
restricted to keywords common enough that a real predictor could plausibly guess them
("oracle-df5"). The second is the honest ceiling.

**3. The benchmark can lie to you.** Our first test set had the answers leaked into the documents
— each code chunk still contained its own docstring, and the test queries *were* those docstrings.
So the search was trivially easy, scored ~99%, and every technique looked useless because there
was no room to improve. It was like benchmarking a cache with a permanently warm hit rate. Once we
stripped the docstrings out, the real difficulty appeared and the techniques' true effects showed.
First lesson of the campaign: if a benchmark says nothing helps, suspect the benchmark.

With those in hand, here's what each technique did.

---

## HyDE — "write a fake answer, then search with that"

**What it is.** The insight: a *question* and the *code that answers it* sit far apart in
meaning-space, because they're phrased completely differently. So don't search with the question —
have a model generate a short, plausible-looking fake code snippet that would answer it, and
search with *that snippet's* fingerprint instead. The fake snippet doesn't have to be correct or
even runnable; it just has to be code-shaped with believable names, so it lands near the real code
in meaning-space.

**What happened.** The first benchmark said "useless" — but that was the rigged benchmark (rule 3).
On the fixed benchmark, it genuinely helped, modestly: of 40 recoverable queries, HyDE rescued
about 7. A real, measurable gain, but a small slice.

**Why it's not the answer.** Generating that fake snippet needs a *text-generating model running
inside ken at query time* — and ken's whole identity is a single, pure-Go, no-dependencies binary.
Running a generative language model in pure Go on a CPU is both a large build (you'd hand-implement
the model) and slow. Spending that to rescue ~7 queries doesn't pencil out.

**Verdict:** works, but thin — and the delivery mechanism is exactly the expensive thing the
project is built to avoid. The generative-model build was shelved.

---

## The encoder predictor — "guess the missing keywords (cheaply)"

**What it is.** Same goal as HyDE — bridge the vocabulary gap — but on the keyword side, and
*without* a generative model. The query says "log people in," the code says `authenticate`. So
predict the code-words the user didn't type and add them to the keyword search. The cheap,
pure-Go way to predict: give every identifier in the codebase a "meaning fingerprint" (averaged
from the fingerprints of the chunks it appears in), then at query time find the identifiers whose
fingerprint sits closest to the query's. No generation — just fingerprint comparison, which ken is
already fast at.

**What happened.** The oracle ceiling (rule 2) was *exciting*: a perfect keyword predictor could
rescue 40 of the missing queries — five times HyDE's reach. Big, promising surface. Then we built
the real predictor and benchmarked it: **dead.** It couldn't actually name the right keywords from
the query.

**Why.** The ceiling proved the *mechanism* works (inject the right keywords and recall jumps). It
did **not** prove the cheap predictor could *find* the right keywords — and it couldn't. Averaged
identifier fingerprints just don't separate cleanly enough to pick `authenticate` out of a
codebase from an English question.

**Verdict:** a large, proven ceiling with no cheap way to reach it. Dead for now — the surface is
real, but this particular bridge can't span it.

---

## PRF / RM3 — "trust the first results, then search again with their keywords"

**What it is.** The classic, zero-model idea from decades of search research. Run the search, take
the distinctive keywords from the top few results, add them to the query, and search again. The
assumption: the top results are probably on-topic, so borrowing their vocabulary sharpens the
query. No model at all — the cheapest thing on the menu.

**What happened.** Net-negative. 9 queries improved, 10 got worse. On balance it made retrieval
*worse*.

**Why.** It's a feedback loop, and feedback loops amplify whatever you feed them. On the *easy*
queries, the first results were already right — it didn't need help. On the *hard* queries — the
exact ones we wanted to fix — the first results were already *wrong*, so it harvested wrong
keywords and doubled down on the mistake. It helps where you don't need it and hurts where you do.

**Verdict:** dead. The cheapest idea, and the cheapness showed.

---

## Heuristic enrichment — "label each chunk with what it is" (the winner)

**What it is.** Stop trying to fix the *query*; fix the *documents* instead. At index time —
before anything is searched — prepend each code chunk with a one-line, auto-generated label of
plain facts about it, pulled straight from the parsed code (the syntax tree ken already builds):

```
# func: authenticate | class: Session | module: login | calls: verify_token, load_user
```

No model, no guessing — these are facts read off the AST. Now an English query mentioning
"session" or "authenticate" has real text to match against, and the chunk's meaning-fingerprint
shifts toward what the code is actually *about* rather than just its raw tokens.

**What happened.** It **won** the bake-off. It matched HyDE's gain — about the same ~7 of 40
queries — but for *free*: pure-Go, deterministic, no model, no API, computed entirely from code
ken already parses. Same benefit as the expensive options, essentially none of the cost.

**Why it wins.** It does the bridging **once, at index time, baked into the index** — the
precompute-vs-compute-on-read instinct any database engineer has. Every other technique paid its
cost *per query, forever*; this one pays once at indexing and reuses it on every search.

**The honest caveat.** It captured ~7 of the 40 recoverable queries — the same modest slice HyDE
got. So most of that 40-query surface is *still* unreached by anything practical we tried. The win
is "a free improvement that matches the expensive one," not "we solved the vocabulary gap."

**Verdict:** ship it. Best benefit-per-cost of everything tried, and it keeps ken pure and
self-contained.

---

## The one we held back: generated descriptions (doc2query)

There's a richer version of enrichment: instead of a mechanical AST label, have a capable model
*write* a sentence or two describing what each chunk does, and index that. It would likely bridge
more than the heuristic label. We deliberately **did not run it as a shippable option**, for two
reasons: it would cost real API money (~$20–25 per full index build), and — more important — it
would force every user to depend on a paid, proprietary model just to index their code, which
breaks the project's "single self-contained binary" promise. It's kept only as a research
*ceiling probe*: a way to learn how much a richer description *could* buy, never something users
would be required to run.

---

## Scoreboard

| Technique | Side | Needs a model? | Result | Verdict |
|---|---|---|---|---|
| **Heuristic enrichment** | Document | No (AST) | ~7 of 40, **free** | **Ship it** — the winner |
| HyDE | Query | Yes (generative) | ~7 of 40 | Works but thin; the model is the costly part — shelved |
| Encoder predictor | Query | No (fingerprints) | Dead (ceiling 40, reached ~0) | Real ceiling, no cheap bridge to it |
| PRF / RM3 | Query | No | Net-negative (9 up, 10 down) | Dead — feedback loop hurts the hard queries |
| Generated descriptions | Document | Yes (offline) | Not run (ceiling probe only) | Off-limits as a product — needs a paid API |
| *(reference) the oracle* | — | — | Ceiling = 40 of 40 | Not shippable; it cheats by looking at the answers |

For scale: the **reranker** ken already ships is worth roughly *fifteen times* any of these on its
own. These query- and document-side tricks are the fine-tuning on top of a system that already
works well — which is exactly why the gains are measured in single-digit queries, not landslides.

---

## What the campaign actually learned

- **Only recall matters here.** Because the reranker re-scores from scratch, the sole job of any
  add-on is to drag a missing right-answer into the top-50. That one fact killed several
  plausible ideas before they started.
- **Measure the ceiling before you build.** The oracle trick saved weeks — it greenlit the
  keyword-predictor *investigation* (ceiling of 40 looked great) and then the real benchmark
  killed the *build* (the cheap predictor couldn't reach it). Both were cheap to learn.
- **Distrust a benchmark that says nothing helps.** The first "everything is useless" result was
  a leaked-answer benchmark, not a real verdict.
- **Document-side beats query-side for a tool like this.** Fixing documents once at index time
  (precompute) beat fixing the query on every search (compute-on-read) — same gain, lower cost,
  and it kept the binary pure. That's the through-line, and it's the same instinct you'd have
  caching a hot computation.
- **The honest state:** there's a real ~40-query surface of recoverable searches. The free,
  shippable winner reaches about a sixth of it. The rest is still open — which is what the
  *untried* techniques in the companion doc (especially the deterministic, code-structure ones)
  are for.
