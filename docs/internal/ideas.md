# aikit ideas — what to build after the multi-model decoder lands

A backlog of "AI-cool" additions that fit aikit's ethos: **pure Go, no cgo,
parity-tested, independently importable, composes with what's already here.**
This is deliberately a menu, not a commitment — each entry says what it is, why
it fits aikit specifically, how hard it is, and what it composes with.

Context: aikit is the pure-Go **retrieval** toolkit (`topk`, `ann` flat+HNSW,
`bm25`, `fuse` RRF, `embed`, `encoder`, `chunk`, `linalg`); the **generation**
half (decoder / tokenizer) now lives in goinfer. This backlog is the aikit-side
retrieval + primitive ideas. (Written when both halves were one repo, so some
entries reference generation — those have moved to goinfer's backlog.)

Rough effort key: **S** = a few days, **M** = a week or two, **L** = a month+.

---

## Tier 1 — define what aikit *is*

### 1. `rag` — the end-to-end retrieval-augmented-generation pipeline  ·  M

**What.** A package that composes the pieces aikit already has into one
grounded-answer pipeline: chunk a corpus (`chunk`) → embed (`embed`) and index
(`ann` + `bm25`) → at query time retrieve from both, fuse the rankings (`fuse`
RRF) → rerank the survivors (`encoder`) → build a prompt with the top passages
and generate a cited answer (`decoder`). A small, opinionated `Pipeline` type
with swappable stages and a `Answer(query) (text, []Citation)` surface.

**Why it fits aikit specifically.** This is the one idea that makes the whole
library *more than the sum of its packages*. Today a consumer has to hand-wire
six packages and know how they interlock; the pipeline is the product, and
"local, pure-Go, no-service RAG with citations" is a thing essentially nothing
else offers. It also turns every other package into a load-bearing component
rather than an island, which is good pressure on their APIs.

**Design notes.** Keep retrieval, fusion, rerank, and generation as
interfaces so each stage is testable and replaceable (e.g. BM25-only, or
rerank-off). Citations come from tracking which chunk ids survived into the
prompt and (optionally) attributing generated spans back to passages via
attention or n-gram overlap. Ship a `demo/rag` CLI: point it at a folder, ask
a question, get a cited answer.

**Composes with.** Literally everything. **Risk:** prompt-format and
context-window management are fiddly; lean on the `decoder` chat template work.

---

### 2. Constrained / structured generation — guaranteed-valid output  ·  M

**Status: JSON shipped** (`constrain` package). The logit-mask seam
(`decoder.SamplingParams.LogitProcessor`) + a `Masker` over a byte-level
`Grammar`, with a streaming JSON grammar — a model that physically cannot emit
malformed JSON (`demo/gemma --json`; hard-invariant test vs `encoding/json`).
Still open below: a general GBNF/regex engine and a JSON-Schema → grammar
compiler on the same `Grammar` interface.

**What.** Decode under a constraint so the model *cannot* emit invalid output:
mask the logits at each step to only the tokens a grammar permits. Two layers:
a JSON-Schema → grammar compiler, and a general GBNF-style grammar engine
(regex and CFG). At each step the grammar's current state yields an allowed
token set; everything else gets `-inf` before sampling.

**Why it fits aikit.** Structured output is the feature people actually reach
for in production (function calls, JSON extraction, classification), and a
correct, dependency-free implementation in Go is rare. It sits right on top of
the existing `Sampler` — the logit vector is already in hand at each step. Great
demo value: "a 270M Gemma that physically cannot produce malformed JSON."

**Design notes.** The hard part is mapping grammar terminals to *token*
boundaries (a token may span a grammar boundary, or several tokens compose one
terminal) — precompute, per grammar state, the set of vocab tokens whose string
is a valid continuation, using a trie over the vocab. Cache aggressively; the
allowed-set computation must not dominate the per-token cost.

**Composes with.** `decoder` (logit masking in the sample loop), `tokenizer`
(vocab trie). Pairs beautifully with the `rag` pipeline for structured
extraction. **Effort** climbs to **L** if you want full CFG with good perf.

---

## Tier 2 — inference-time algorithms the forward pass unlocks

### 3. Speculative / assisted decoding  ·  M

**What.** Use a small fast model (Gemma 270M) as a *draft* that proposes k
tokens, then verify them in a single batched forward of the larger *target*
model, accepting the longest prefix that matches the target's own
distribution. 2–3× fewer target forwards for the same exact output
distribution (it's provably distribution-preserving, not an approximation).

**Why it fits aikit.** It's the canonical "cool inference trick," it's a
showcase for the multi-model work (draft and target can be different
checkpoints from the same family), and the speedup is real and measurable. The
`decoder` already separates `runLayers` from the LM head, which is most of the
batched-verify plumbing.

**Design notes.** Needs a *batched* forward over k candidate positions (the
verify step), so it leans on the M7 perf work and a KV-cache that can roll back
rejected tokens. Acceptance test: output distribution identical to plain
sampling at the same seed (the correctness gate).

**Composes with.** `decoder` (two models, shared sampler), the M7 batched
matmul. **Risk:** KV-cache rollback is the subtle part.

---

### 4. Logprobs, perplexity, and LLM-as-scorer  ·  S

**What.** Expose what the forward pass already computes: per-token log-probs,
sequence perplexity, and a `Score(prompt, continuation)` that returns the
total/average logprob of a continuation. Then an LLM-reranker that scores RAG
candidates by how well the model "expects" them.

**Why it fits aikit.** Almost free — the full logit vector is in hand at every
step; this is mostly API surface plus a softmax-logprob helper. It immediately
enables evaluation, candidate scoring, and a quality signal for the `rag`
pipeline. Cheap, high-utility, low-risk.

**Composes with.** `decoder`, `rag`, the eval harness (#9). **Effort: S** —
the lowest-cost item here with outsized leverage.

---

### 5. Sampler family expansion  ·  S

**What.** Add to the existing `Sampler`: **min-p** (keep tokens ≥ p·max_prob —
the current favorite over top-p), **Mirostat** (targets a perplexity
set-point), **contrastive search**, **beam search** (for tasks that want it),
and repetition controls (**repetition/frequency/presence penalties**, **DRY**,
**no-repeat-ngram**).

**Why it fits aikit.** Pure post-processing of the logit vector; each is a
small, independently-testable function alongside `topFilter`. Min-p and the
repetition penalties in particular noticeably improve small-model output, which
matters since aikit's sweet spot is small local models.

**Composes with.** `decoder.Sampler`. **Effort: S** per sampler.

---

### 6. KV-cache techniques — prefix caching, attention sinks, cache quant  ·  M

**What.** Three related upgrades to `KVCache`:
- **Prefix caching / sharing** — cache the KV for a fixed prefix (a system
  prompt, a RAG context) and reuse it across requests instead of re-prefilling.
- **StreamingLLM / attention sinks** — keep the first few tokens + a sliding
  window so generation continues coherently past the trained context length
  with bounded memory.
- **KV-cache quantization** — store cached K/V in int8 to roughly halve cache
  memory, the dominant cost at long context.

**Why it fits aikit.** Prefix caching is a direct, large win for the web GUI
(every chat turn re-uses the system prompt) and for `rag` (the retrieved
context is a long, reused prefix). Attention sinks are a genuinely clever,
low-code trick. All three are local-memory wins that matter precisely because
aikit targets laptops.

**Composes with.** `decoder`, the `demo/gemma-web` GUI, `rag`. **Risk:** prefix
caching interacts with position ids and the sliding window — careful invalidation.

---

## Tier 3 — retrieval-side depth

### 7. Late-interaction retrieval (ColBERT-style multi-vector)  ·  M

**What.** Instead of one vector per document, store per-token embeddings and
score a query against a document by summing each query token's max similarity to
any document token (MaxSim). Much stronger recall than single-vector,
especially for short or keyword-ish queries.

**Why it fits aikit.** It reuses `encoder` for token-level embeddings and slots
in as an alternative retriever behind the `rag` pipeline's retrieval interface.
It's a well-defined algorithm with a clear correctness story (MaxSim is exact),
and it differentiates aikit's retrieval from "yet another cosine index."

**Design notes.** Storage blows up (N tokens × dim per doc), so this wants the
quantized index (#8) underneath to be practical. Two-stage is natural: cheap
single-vector ANN to get candidates, MaxSim rerank on the survivors.

**Composes with.** `encoder`, `ann`, `rag`. **Risk:** memory; pair with #8.

### 8. Quantized / large-scale vector index — PQ, IVF, scalar quant  ·  M

**What.** Make `ann` scale past "fits in RAM as f32": **scalar quantization**
(int8 vectors, 4× smaller), **product quantization** (PQ — split the vector,
quantize subspaces, asymmetric distance), and **IVF** (inverted-file coarse
clustering so a query only scans a few cells). Optionally **binary embeddings**
+ Hamming rerank for a very fast first stage.

**Why it fits aikit.** The current flat + HNSW indexes are great up to a point;
PQ/IVF is what lets a laptop hold millions of vectors. It's classic,
well-specified, pure-Go-friendly numerical code with exact recall/latency
tradeoffs you can measure — squarely aikit's style.

**Composes with.** `ann`, `embed`, `rag`, late-interaction (#7). **Effort:** M,
more if you want the full IVF-PQ combo with good recall tuning.

### 9. Retrieval + generation eval harness  ·  S–M

**What.** A small `eval` package with the standard metrics: **nDCG, MRR,
recall@k, MAP** for retrieval; **perplexity, exact-match, ROUGE-ish overlap**
for generation; plus a runner that takes a labeled set and a pipeline and prints
a scoreboard.

**Why it fits aikit.** The project is already parity-obsessed (golden tests
everywhere); giving every feature a *quality* scoreboard, not just a
correctness one, is the natural extension. It makes #1, #7, and #8 comparable
instead of vibes-based, and it's the kind of thing a serious retrieval library
is expected to have.

**Composes with.** Everything; especially `rag`, `ann`, logprobs (#4).

---

## Tier 4 — classic ML primitives (cool, broadly useful, pure-Go)

### 10. Embedding-space clustering — k-means(++), mini-batch  ·  S

K-means/k-means++ over the vectors `embed`/`encoder` produce: semantic
clustering, topic discovery, IVF cell construction (#8), and "summarize a corpus
into N themes." Small, classic, and a building block several other ideas reuse.
**Composes with** `ann`, `embed`, IVF (#8).

### 11. Near-duplicate detection — SimHash / MinHash + LSH  ·  S

Fingerprint documents and find near-duplicates cheaply. Invaluable as a
*pre-indexing* cleaning step for `rag` corpora (dedup before you embed and
index), and a genuinely useful standalone utility. Pure bit-twiddling, very
Go-friendly. **Composes with** `chunk`, `rag`.

### 12. Dimensionality reduction + Matryoshka embeddings  ·  S–M

**PCA / random projection** for shrinking embeddings and 2-D visualization, plus
first-class support for **Matryoshka** embeddings (use the first k dims of a
model trained for it — instant 2–4× storage cut at a known quality cost).
**Composes with** `embed`, `ann`, #8.

---

## How these sequence

A reasonable order once multi-model lands, by leverage-per-effort:

1. **#4 logprobs/perplexity (S)** and **#5 sampler family (S)** — quick wins that
   strengthen `decoder` and unblock eval.
2. **#1 `rag` pipeline (M)** — the centerpiece; turns the library into a product.
3. **#9 eval harness (S–M)** and **#11 dedup (S)** — make `rag` measurable and its
   corpus clean.
4. **#2 constrained generation (M)** — the headline capability.
5. **#8 quantized index (M)** then **#7 late-interaction (M)** — retrieval depth at
   scale.
6. **#3 speculative decoding (M)** and **#6 KV-cache tricks (M)** — perf/UX, best
   after the M7 batched-matmul backend exists.
7. **#10 clustering / #12 dim-reduction (S)** — fill in as needed; several earlier
   items reuse them.

If only two ever get built: **#1 (`rag`)** because it defines what aikit is, and
**#2 (constrained generation)** because guaranteed-valid structured output is the
capability people most want and least often find in pure Go.

---

## Explicitly out of scope (for now)

Training / fine-tuning (aikit is inference-only by design), encoder-decoder and
non-transformer architectures (T5, Whisper, Mamba/SSM — different skeletons),
and anything requiring cgo or a GPU-only path (the WebGPU backend is the one
sanctioned accelerator, behind the existing `Backend` seam).
