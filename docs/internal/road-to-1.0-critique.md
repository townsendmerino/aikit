# Road to 1.0 — critique & punch-list (ken + aikit)

> Captured 2026-06-04. Source: external review of both repos. The engineering
> quality in both is well past most projects' 1.0 bar (ADRs for every decision,
> parity-gated numerics, honest limitation docs, clean TODO hygiene). The
> criticism below is **not about code quality** — it's about **product,
> audience, and focus**, which is the stuff that's easy to neglect when the code
> is this good.

---

## ken

### The core tension: it's defined as a faithful port
"Same algorithm as semble, Go runtime + token efficiency" is the stated pitch,
and it's also the ceiling. A port permanently chases its upstream — if semble
improves ranking, ken inherits a deficit. The genuinely differentiated thing in
the repo is **not** parity; it's the **embedded-corpus build pattern**
(ADR-016 / ADR-024): an SDK author `//go:embed`s their docs + model and ships a
single static MCP binary their users `brew install`. Python/semble can't do that
cleanly. That's a *product*. "semble but Go" is a *feature*. The README leads
with parity and buries the differentiator — for 1.0, flip that.

### Recall is the one thing that materially matters
82–91% recall@10 vs grep's ~99% is disclosed honestly, but disclosure doesn't
make it not-the-problem. For an agent, "1 in 5 relevant chunks silently missing"
is a correctness gap that "fall back to grep" papers over by pushing judgment
onto the agent. Every other axis (perf, languages, tools) is already closed.
Pushing recall@10 toward 95% is the only change that makes the *product* better
rather than more *complete*.

### Adoption friction is the real gap, not features
18 env vars, a 64MB model download (520MB for the reranker), a multi-step
`print-listen-script | psql` for the DB tier, 190–265MB demo binaries, no
Windows build, and GitHub-releases-only distribution (no brew/scoop tap). The
basic path works, but the surface a *new* user hits is large. None of this is a
code problem — it's the actual distance to adoption.

### Blunt summary for ken
The code is 1.0. The risk is shipping 1.0 to an audience of nearly nobody. Work
left is not engineering:
1. One recall push (82% → ~90%+).
2. A zero-config first run + real distribution (brew/scoop, smaller default
   footprint, ideally Windows).
3. At least one external proof point: ken-mcp in one agent's recommended config,
   or one SDK shipping an embedded `ken-mcp-docs` binary. Ship 1.0 *with* a named
   adopter, not before one.

---

## aikit

### Identity problem (the main criticism)
aikit is described as "the parts of ken another project could reuse" — that's
*ken's* point of view, not a user's. The packages span a 167-line `topk`
min-heap to a 4,300-line pure-Go multi-architecture LLM `decoder`
(Gemma/Qwen/Llama/Mistral/Mixtral/Mellum, GGUF/GPTQ/AWQ, int4/int8/bf16, MoE,
YaRN). Nobody's use case is "I need a heap selector *and* a Gemma runtime." It
reads as an extraction, not a product, and the **monolithic `go.mod` enforces the
worst version of that**: you can't `go get` one block, and `go build ./...`
drags in `cogentcore/webgpu` (cgo) and pre-1.0 `gotreesitter` even if you touch
neither.

### The decoder is the tail wagging the dog
It's the most impressive thing in either repo — a viable pure-Go, no-cgo local
LLM runtime — and it's buried in a utility grab-bag. It's also a treadmill: new
model families and quant formats land constantly, so its API "will keep moving."
Bundling it into aikit means aikit's 1.0 is hostage to the one package that can
never stop churning. Strong recommendation: the decoder + tokenizer + constrain
(the LLM runtime) deserve their own release cadence — at minimum their own
modules, arguably their own repo.

### No glue, no worked example
The packages obviously compose into a RAG pipeline
(`chunk → embed → ann + bm25 → fuse → encoder-rerank → topk`), but there's no
`rag` orchestrator and no end-to-end example — assembly is left to the reader.
Combined with no per-package README and no `examples/` dir, a user who wants
"just `bm25`" or "just `encoder`" has to read source. The single highest-value
addition is one end-to-end example that wires the whole pipeline; it sells every
package at once and answers "who is this for."

### Target user undefined
Real answer: someone building a Go RAG/search system, or wanting pure-Go local
inference with no Python/cgo. But it isn't packaged or marketed for that person.

---

## What caps each off for 1.0

### ken (work is distribution + recall, not code)
- [ ] Reposition around the embedded-docs-MCP pattern as the lead value prop.
- [ ] One recall campaign (82% → ~90%+) — the only product-moving change.
- [ ] Frictionless first run: smaller default footprint, real install channel
      (brew/scoop), ideally a Windows build.
- [ ] One external adopter / case study; ship 1.0 with it.

### aikit (work is focus + packaging) — DONE, ready to cut 1.0
- [x] **Independently consumable / optional deps isolated** — v0.4.0 split: LLM
      runtime → goinfer, `chunk/treesitter` its own module; core graph is
      `x/text`-only (CI cleanliness guard enforces it).
- [x] LLM runtime (decoder/tokenizer/constrain) on its own cadence — goinfer repo.
- [x] End-to-end RAG example (`examples/rag`) + per-package godoc `Example`s for
      the hard tier (chose `Example` tests over `doc.go` — compile-checked, render
      on pkg.go.dev).
- [x] **Hard tier frozen.** Tier membership decided: the retrieval core is the
      v1.0 compatibility guarantee; `linalg`, `encoder.Backend`, `ann.HNSW`,
      `encoder.Q8`, the mmap loader variant, and the concrete chunker structs are
      the **Experimental** tier (excluded from the 1.0 promise, may evolve).
      Verified backward-compatible across 0.4.x↔0.5.x with `apidiff` (zero
      incompatible changes) — the two-consecutive-minors bar is met.
- [ ] Cut `v1.0.0` (maintainer's go).

### One-sentence version
ken's gap to 1.0 is **distribution and recall, not code**; aikit's gap is
**focus** — independently-consumable modules, the decoder on its own cadence, and
an integrating example — before "1.0" means anything to a user.
