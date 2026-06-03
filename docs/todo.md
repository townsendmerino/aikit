# aikit todo — what to do next (post-v0.2.0)

Ranked recommendation for the decoder follow-ups, grounded in
[`multi-model-plan.md`](multi-model-plan.md) §7–§8 and
[`milestones/G7-gguf.md`](milestones/G7-gguf.md) "Scope / next". Effort key:
**S** = a few days, **M** = a week or two, **L** = a month+.

---

## The four decoder follow-ups, ranked (value × effort × risk)

### 1. GGUF tokenizer + more K-quants (Q5_K/Q3_K) — **SPM tokenizer DONE** ⭐ · S

Closes the loop v0.2.0 opened. ✅ **`tokenizer.LoadGGUF` shipped** for the
SentencePiece byte-fallback family (`tokenizer.ggml.model == "llama"`:
Llama-2/Mistral/TinyLlama), HF-parity-gated on TinyLlama, and `demo/gemma` now
chats from a bare `.gguf` end-to-end. Still open, both deferred for lack of a
parity fixture (the repo's bar is no unvalidated dequant/tokenizer code):

- **Byte-level GGUF tokenizer** (`gpt2` family: Llama-3/Qwen/GPT-2) — same
  `modeByteLevel` machinery, knobs from `tokenizer.ggml.pre`; needs a committed
  byte-level GGUF to gate (testdata has only the SPM/llama TinyLlama GGUF).
- **More K-quants (Q5_K/Q3_K/IQ\*)** — each "a `dequant*` func + a size entry",
  but needs a Q5_K fixture or the Python `gguf` reference to validate.

### 2. Resident int8/int4 matmul (dequant-per-tile) — **int8 + mmap DONE** · M–L

Both planning docs flagged this as "the way to actually run big quantized models
in small memory." Two pieces shipped:

- ✅ **Streaming int8**: `Load(…, Quant:"int8")` quantizes each matmul tensor as
  it loads and frees the f32 before the next (safetensors, GPT-2, GGUF) — no
  whole-model f32 spike, so a checkpoint loads in ~¼ the RAM and a bare `.gguf`
  lands resident as int8 (`TestGGUF_int8_resident`: argmax + 0.9998 cosine vs
  f32). Reuses the `MatmulBTQ8` dequant-per-tile kernel.
- ✅ **mmap GGUF** (`embed.OpenGGUFMmap`): maps the `.gguf` instead of
  heap-reading it, so the raw quantized bytes are reclaimable page cache rather
  than a heap copy — removes the load-time peak that otherwise exceeded the int8
  steady state. Bonus: `tokenizer.LoadGGUF` no longer pages in the weights to
  read metadata (its test went ~0.5 s → ~0.03 s). Parse is bit-identical to the
  heap path.

Still open — **int4 group-quant** (≈⅛ f32, matches native Q4 footprint): the
streaming-quant load path is now in place, so this needs only its own
group-quantized `weightMat` variant (group-size 32–128, per-group scale, packed
nibbles) + a dequant-per-tile matmul kernel + a cosine-vs-f32 accuracy gate.
`MatmulBTQ8` and the `weightMat` seam are the template.

### 3. GPTQ / AWQ (safetensors-resident int4) — broadens coverage · M

"The other half of G7." Different packing (`qweight`/`qzeros`/`scales`/`g_idx`),
same dequant-to-f32 idea; the safetensors loader already handles the container.
Adds the HF-hosted int4 ecosystem. Coverage-breadth, not loop-closing — inherits
the same RAM caveat as #2 until it lands.

### 4. Shared-expert MoE + YaRN/longrope — most contained, lowest urgency · S–M

"A couple more `MoEConfig` knobs" on the G6 MoE base for Qwen-MoE/DeepSeek, plus
YaRN's mscale for long context. Cleanly scoped, but only pays off for those
specific families.

---

## Meta-note: the highest-value next thing may be none of the four

[`ideas.md`](ideas.md) Tier 1 argues the biggest wins *define what aikit is*
rather than deepen the decoder:

- **`rag`** — compose all packages into one cited-answer pipeline (the product).
- **Constrained / structured generation** — logit-masking on the existing
  `Sampler` so a small model *cannot* emit malformed JSON.

If the real question is "what's most impactful," `rag` ranks above all four
above. The four deepen the decoder; the Tier-1 ideas define the library.

---

## Decision

**Now:** #1, the **GGUF tokenizer** (+ a couple more K-quants if cheap) — small,
closes the v0.2.0 story, unblocks a clean "point at a `.gguf`, get a chat" demo.
**Next substantial push:** #2, resident int4.
