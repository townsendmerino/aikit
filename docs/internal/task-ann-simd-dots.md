# ann: route similarity dots through linalg SIMD (prompt for an aikit session)

> Ready-to-paste prompt. Context: drafted 2026-06-09. Follow-on to 58c947b
> (encoder scores·V vectorization) — same pattern, different package.

```
Follow-up to 58c947b (encoder scores·V vectorization): the same "scalar loop
where a SIMD kernel exists" pattern lives in the ann package, which does not
import linalg at all. Every similarity in both backends is a scalar float64
dot loop:

1. ann/flat.go:89 and :111 (both Query paths) — the brute-force scan dots q
   against EVERY indexed vector with a scalar loop: O(N·d) per query, 100% of
   Flat's query cost. This is the ideal customer for the batched kernels:
   linalg.Dot for the simple swap, Dot4x4/Dot8x4 to stream 4-8 index vectors
   per pass against the shared query (the same trick the W8A8 matmul uses).
2. ann/hnsw.go:124 sim() — the same scalar dot in HNSW's innermost loop,
   executed O(efSearch·M) times per query and per insert. linalg.Dot is a
   drop-in for the loop body.

DECISION NEEDED (analogous to the goinfer QKᵀ float64 question): both sites
accumulate in float64 and Hit.Score / sim() return float64; linalg.Dot is f32
accumulation + SIMD reordering. Consequences: near-tie orderings in ranked
results can flip (sub-ULP score differences). HNSW is approximate by contract
— accept silently. Flat advertises exact brute-force ranking — decide whether
f32 scores are acceptable (recommended: yes, vectors are f32 and unit-norm,
per-element error is bounded; document it) or keep a float64 scalar path
behind the existing API and add a fast path. State the choice in the commit.

VALIDATION: existing ann tests; plus a recall@k check old-vs-new on a real
embedding set (testdata or examples/rag corpus) — recall must be unchanged,
tie-order may differ. Benchmark before/after: Flat.Query at N=10k/100k,
d=768; HNSW search + build at the same scale. Expect roughly the DotF32-vs-
DotGo ratio from linalg's own benches.

CHECKED, NOT WORTH IT (skip): embed/model.go:244 weightedMeanPoolSafe is
O(L·d) single-pass pooling, not a hotspot; the scalar loops in
linalg/quant_w4a8*.go are the intentional generic fallback + group tails of
the SIMD kernels; examples/ are examples.
```
