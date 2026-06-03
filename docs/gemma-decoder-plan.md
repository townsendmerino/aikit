# aikit decoder plan — running Gemma 3 (the open Gemini) in pure Go

Status: **proposal / scaffold landed**. This mirrors the milestone +
golden-parity discipline of the rerank plan that produced `encoder/`. The
deliverable is a new `decoder/` package (autoregressive generation) plus a
`tokenizer/` package (SentencePiece) and a `demo/gemma` CLI. First target is
**Gemma 3 270M / 1B** running interactively on a laptop CPU, with the runtime
shaped so a **WebGPU backend** (the `github.com/cogentcore/webgpu` dep already
in `go.mod`) can slot in for the larger checkpoints later.

---

## 0. Why this is a new package, not an `encoder` extension

`encoder/` is a faithful, fast transformer forward pass — but it is an
**encoder**: bidirectional, single forward, embedding output. Gemma is a
**decoder**: causal, autoregressive, one-token-at-a-time with a growing KV
cache, and a vocabulary-sized softmax at the end. The shapes of the inner
math overlap; the control flow, memory model, and several layer primitives do
not. Forcing both into one package would mean a pile of `if causal` branches
through hot code. They stay siblings — exactly as `embed` and `encoder` are
siblings that share `safetensors.go` + the tokenizer surface.

### What carries over (reuse, don't rewrite)

| Asset (today in `encoder/` or `embed/`) | Reuse for the decoder |
|---|---|
| SIMD dot / matmul (`dot_arm64.s`, `dot_amd64.s`, `linalg.go`, row-parallel `parallel.go`) | The single largest win. Decode is matmul-bound; this is already NEON/AVX2-tuned. **Must be factored into a shared package** — see §1. |
| `rope.go` (NeoX RoPE, `rotate_half`, precomputed cos/sin) | Reusable; Gemma needs **two** tables (local θ=10k, global θ=1M) and the apply path must take a position offset for cached decode. |
| `embed/safetensors.go` (zero-copy header parse, mmap, finalizer) | Reusable as-is for the header; **needs a BF16/F16 decode path** (§2). Gemma ships bf16. |
| `scratch.go` + `sync.Pool` arena pattern | Reusable; the decode arena is smaller (L=1 on the hot step) but the KV cache replaces the per-call growth. |
| `quant.go` (per-row symmetric int8) | Reusable concept; extend to int4 group-quant for the 1B/4B memory story (§8). |
| `softmaxRow`, `silu` (`linalg.go`) | `softmaxRow` reused directly; add `gelu_tanh` for GeGLU. |
| Milestone + `scripts/pin_*.py` golden harness | Reused wholesale — this is how we keep parity honest (§10). |

### What is genuinely new

Causal attention + a **KV cache**; **grouped-query attention** (Gemma 3 270M
is 4 query heads / 1 KV head); **RMSNorm** with Gemma's `(1 + weight)` scaling
and pre+post norm placement; **GeGLU** (gelu-gated) MLP; **embedding scaling**
by `sqrt(hidden_dim)` and **tied** embeddings reused as the LM head; the
**262k SentencePiece tokenizer**; a **sampler** (greedy / temperature / top-k /
top-p) with EOS + stop handling; **sliding-window** attention on the local
layers; **QK-norm** (Gemma 3 dropped Gemma 2's logit soft-capping in favor of
query/key RMSNorm); and the **generation loop** that ties it together with a
streaming token channel.

---

## 1. Shared linalg — the prerequisite refactor

The SIMD matmul/dot lives **unexported inside `encoder/`** today. The decoder
needs it too, and a WebGPU backend needs a clean seam to substitute for it.

Plan: lift the compute primitives into `internal/linalg` (or a public
`linalg` if we want third parties to reuse it):

- `MatmulBT(a, b, dst []float32, M, K, N int)` and the blocked/parallel/tiled
  variants, plus the `dot*` asm and its `_arm64` / `_amd64` / `_other` build-tag
  trio.
- `SoftmaxRow`, `LayerNorm`, and the new `RMSNorm`, `GeluTanh`.

`encoder/` then imports it instead of owning it (a pure move + re-export; the
encoder's public API is unchanged, the package's "Hard, 1.0-committed" surface
in the README is untouched). This is mechanical but must land first because
everything below sits on it.

---

## 2. BF16 in the safetensors loader

`embed.Tensor` decodes `F32`/`F64`/`I64`. Gemma checkpoints are `BF16`
(and some are `F16`). Add to `embed/safetensors.go`:

- `func (t Tensor) BFloat16sToF32() ([]float32, error)` — widen bf16 → f32 by
  `uint32(b)<<16` into the float32 bit pattern (bf16 *is* the top 16 bits of
  f32, so this is exact and branch-free; subnormals/NaN/Inf come along for
  free).
- `func (t Tensor) Float16sToF32() ([]float32, error)` — proper f16 → f32
  (5-bit exponent rebias, subnormal handling).

Open question to settle at M1: **widen-on-load (2× RAM, simplest) vs keep bf16
resident and widen per-tile inside matmul (half the RAM, the route the int8
path already proves out).** For 270M either is fine; for 1B+ the resident-bf16
route matters. Recommend widen-on-load for M1 correctness, bf16-resident as a
later memory milestone alongside §8.

---

## 3. Gemma 3 architecture — the facts the loader/forward must pin

From the Gemma 3 release (validated against the HF `transformers` config and
the published technical report; see Sources at the bottom). The **270M**
config:

| Field | 270M | Notes |
|---|---|---|
| layers | 18 | |
| hidden (model) dim | 640 | |
| query heads | 4 | |
| KV heads | 1 | **GQA**, 4:1 |
| head dim | 256 | note: `heads*head_dim (1024) ≠ hidden (640)` — Gemma decouples them |
| MLP hidden | 2048 | GeGLU |
| vocab | 262 144 | SentencePiece; **tied** embeddings (~170M of the 270M params live here) |
| context | 32 768 | |
| local sliding window | 512 | |
| attention pattern | 5 local : 1 global | 270M = 15 local + 3 global layers |
| RoPE θ | local 10 000 / global 1 000 000 | **two tables** |
| norm | RMSNorm, pre **and** post on both attn and MLP | Gemma's `(1 + w)` scale |
| QK-norm | yes | RMSNorm on Q and K before attention |
| activation | `gelu_pytorch_tanh` | GeGLU |
| embedding scale | × `sqrt(hidden_dim)` | applied after embedding lookup |
| query scaling | `query_pre_attn_scalar` (256) | not `1/sqrt(head_dim)` by default |
| soft-capping | **none** in Gemma 3 | (Gemma 2 had it; 3 replaced it with QK-norm) |

The **1B** config differs in counts (≈26 layers, 1152 hidden, larger MLP) but
is the *same architecture* — so a config-driven loader handles both with no
code change, exactly as `encoder/`'s `Config` + `ValidateAssumptions` does.
`ValidateAssumptions` should fail loudly on anything the forward pass doesn't
implement (e.g. a checkpoint that still has soft-capping, or an interleaved
RoPE variant).

---

## 4. Package layout (scaffold landed in this change)

```
decoder/
  doc.go         package contract + the carry-over invariants
  config.go      Gemma3Config + ValidateAssumptions + loadConfig   [done: struct+validate]
  backend.go     Backend interface, CPU backend, WebGPU stub, registry   [done: iface+cpu naive]
  weights.go     Weights/LayerWeights + Load (BF16)                 [stub: returns NotImplemented]
  rmsnorm.go     RMSNorm with (1+w) scaling                         [done: small + correct]
  attention.go   causal GQA + sliding window + QK-norm + KV cache   [stub]
  mlp.go         GeGLU                                              [stub]
  kvcache.go     per-layer ring/append KV buffers                  [done: type + append]
  sampler.go     greedy / temp / top-k / top-p, EOS/stop           [done: greedy; top-k/p stub]
  model.go       Model.Load, Model.Generate (streaming), forward    [stub: wiring + NotImplemented]
tokenizer/
  doc.go
  sentencepiece.go   Load / Encode / Decode (262k byte-fallback BPE)  [DONE: M2, HF-exact]
  added.go           added/special-token trie (longest-match split)   [DONE: M2]
demo/gemma/
  main.go        CLI: flags, load, prompt, stream tokens to stdout  [done: wiring + NYI guard]
```

Everything compiles today; the stubs return
`errNotImplemented` referencing the milestone that fills them in, so the demo
builds and runs with an honest message until the forward pass lands.

---

## 5. The Backend seam (CPU now, WebGPU later)

`decoder.Backend` abstracts the few hot primitives the forward pass calls:

```go
type Backend interface {
    Name() string
    MatmulBT(a, b, dst []float32, M, K, N int) // dst[M,N] = a[M,K] · b[N,K]ᵀ
    Close() error
}
```

- `cpuBackend` — wraps the shared `linalg` matmul (§1). The scaffold ships a
  naive-but-correct version so the interface is exercised end-to-end; swapping
  in the SIMD/parallel path is a one-line change once §1 lands.
- `webgpuBackend` — documented stub. WebGPU shines exactly where CPU pure-Go
  hurts: the 1B/4B/27B matmuls. The seam keeps the forward pass identical; only
  the matmul provider changes. Weight upload to GPU buffers, a WGSL matmul
  kernel, and bf16/int8 staging are the real work, deferred to its own
  milestone. Selected via `--backend=webgpu` (falls back to CPU if the adapter
  is unavailable, so the demo never hard-fails on a headless box).

Keeping the interface tiny (one hot op) is deliberate: norms/rope/softmax are
cheap and stay on CPU even with a GPU matmul backend, which avoids a
chatty host↔device round-trip per layer.

---

## 6. Milestones (each ends green with a pinned golden)

- **M0 — parity harness.** `scripts/pin_gemma.py`: load Gemma 3 270M via HF
  `transformers`, dump config, a handful of tensor checksums, and the
  full logit vector for a fixed 8-token prompt at the first decode step into
  `testdata/gemma_golden.json`. This is the oracle for everything below.
- **M1 — loader.** BF16 decode (§2) + `Gemma3Config` + shape-validated weight
  load. Test: every tensor present with the expected shape; checksums match M0.
- **M2 — tokenizer.** ✅ **DONE 2026-06-02.** Byte-fallback **BPE** load (Gemma 3
  ships BPE, not unigram — see [`milestones/M2-tokenizer.md`](milestones/M2-tokenizer.md))
  + `Encode`/`Decode`; golden parity on a prompt set and verified id-for-id over
  215k+ adversarial inputs (every BMP codepoint) vs HF. ids match exactly, BOS
  included.
- **M3 — single-token forward.** ✅ **DONE 2026-06-02.** Embedding (×scale) →
  18 layers (RMSNorm sandwich, GQA, QK-norm, dual-base RoPE, GeGLU) → final norm
  → tied LM head. Matches the HF float32 oracle: **cosine 0.999999999999**,
  argmax identical (`' Paris'`). The correctness gate caught a real RoPE-base bug
  (`layer_types` vs null `sliding_window_pattern`) — see
  [`milestones/M3-forward.md`](milestones/M3-forward.md). Cache-based single-step
  path (lays M4 groundwork); naive backend (M7 perf).
- **M4 — KV cache + multi-token decode.** ✅ **DONE 2026-06-02.** Append K/V per
  step, attend over the cache, advance RoPE position. 48-token greedy
  continuation matches HF **id-for-id** and the decoded string matches end to
  end; EOS/stop wired (`isStop`, `Config.EOSIDs`). See
  [`milestones/M4-decode.md`](milestones/M4-decode.md).
- **M5 — sliding window.** ✅ **DONE 2026-06-02.** Local layers mask to the last
  512 keys; global layers see all. 748-token prompt matches HF (cosine 1−1e-11).
  Caught + fixed two latent windowing bugs (mutable position, off-by-one). See
  [`milestones/M5-window.md`](milestones/M5-window.md).
- **M6 — sampler + streaming.** ✅ **DONE 2026-06-02.** Temperature / top-k /
  top-p (seeded, reproducible), EOS/stop, and a streaming `demo/gemma` CLI that
  generates from a real checkpoint (UTF-8-correct byte-fallback streaming).
  Repetition penalty / chat template are open follow-ups. See
  [`milestones/M6-sampler.md`](milestones/M6-sampler.md).
- **M7 — perf.** Wire the SIMD/parallel `linalg` backend, scratch-pool the
  decode arena, profile tokens/sec on 270M and 1B. Target: interactive
  (>10 tok/s) on an M-series laptop for 270M.
- **M8 — memory / quant.** int8 (reuse `quant.go`) and int4 group-quant for the
  larger checkpoints; bf16-resident matmul tiling (§2). Makes 1B comfortable
  and 4B feasible on a laptop.
- **M9 — WebGPU backend (optional).** WGSL matmul kernel behind the §5 seam for
  the checkpoints CPU can't serve interactively.

A fresh checkout with no model assets present should `go test ./...` green with
the Gemma parity tests **skipped** (same convention `encoder/` uses).

---

## 7. Realistic targets (set expectations)

Pure-Go CPU on a laptop:

- **270M** — interactive, the demo's default. ~540 MB bf16 (or ~140 MB int8).
- **1B** — usable, a few tok/s f32; comfortable quantized. Good "real model" demo.
- **2B–4B** — feasible **only** quantized (int4) + mmap; slow but it runs.
- **12B / 27B** — not realistic in pure-Go CPU. These are the WebGPU-backend
  (M9) story, and even then a laptop GPU is the constraint, not the code.

Honesty here is the point: "run the open Gemini" means 270M/1B interactively
and the bigger ones as a quantized/GPU stretch — not 27B on a CPU.

---

## 8. Memory & quant notes

The 262k tied embedding table dominates the 270M's parameter count (~170M of
270M). int8 on just the matmul weights leaves embeddings as the floor; int4
group-quant (group size 32–128, per-group scale, the GPTQ/`Q4_K` shape) on the
embedding + projections is what unlocks the 1B/4B laptop story. `quant.go`'s
per-row symmetric int8 is the starting point and its round-trip test
(`≤ 1e-2` rel-L2) is the template for the int4 tests.

---

## 9. Tokenizer (the other big new piece)

Gemma uses a 262k **SentencePiece-derived BPE** model with byte-fallback —
**confirmed BPE** at M2 (explicit ~515k merge table, not unigram), shipped as
`tokenizer.model` (the SP protobuf) and/or `tokenizer.json`. M2 loads the
`tokenizer.json` (the explicit merges + pipeline make HF-exact parity
tractable). The
existing `embed.Tokenizer` is WordPiece and won't transfer. The new
`tokenizer` package needs: unigram/BPE merge logic, byte-fallback for OOV,
the Gemma special tokens (`<bos>`, `<eos>`, `<start_of_turn>`,
`<end_of_turn>`), and the `▁` whitespace marker decode. Parsing the SP
protobuf in pure Go is the bulk of the work; `tokenizer.json` (HF format) is an
easier JSON path if we accept that dependency shape. Golden parity (M2) against
HF's tokenizer is non-negotiable — a one-token tokenizer drift silently wrecks
generation quality.

---

## 10. Testing discipline (unchanged from `encoder/`)

Per-machine model assets, skip-clean when absent, committed JSON goldens
regenerated by a pinned Python script. The decode path adds one wrinkle:
generation is a *sequence*, so the golden stores the first-step logit vector
(deterministic, exact) **and** a greedy continuation string (catches cache /
position / masking bugs M3's single-step check can't see).

---

## Sources

- [Gemma 3 model card & config — Hugging Face `transformers` docs](https://huggingface.co/docs/transformers/model_doc/gemma3)
- [Gemma 3 270M specifications](https://apxml.com/models/gemma-3-270m)
- [Gemma 3 technical deep dive — architecture (local/global attention, dual RoPE, QK-norm)](https://namangoyal.com/blog/2025/gemma3/)
- [Gemma 3 270M from-scratch reference implementation](https://github.com/di37/gemma3-270M-tinystories-pytorch)
