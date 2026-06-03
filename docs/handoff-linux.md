# Picking up on a fresh machine (Linux)

A launcher for continuing this work after cloning on another box — most
importantly an **amd64/Linux** box, where the things this repo *couldn't*
run on the arm64 Mac finally execute. This points at the detailed docs
rather than repeating them.

> Everything is on **`main`** now — a plain clone has it all:
>
> ```bash
> git clone https://github.com/townsendmerino/aikit.git && cd aikit
> ```

> **▶ Resume point (2026-06-03).** Done: Gemma decoder M1–M9 + the multi-model
> track **G0 (descriptor), G1 (sharded loader), G2 (Qwen3 + Llama dense), G3
> (byte-level BPE tokenizer, incl. Llama-3), G4 (RoPE scaling — linear + llama3 —
> + partial rotary)** — see the table in §3. **Qwen3-1.7B and Llama-3.2-1B both
> generate end-to-end in pure Go**; the `gemma-web` chat wires Qwen3 (ChatML +
> multi-EOS via `resolveEOSIDs`). The `llama` adapter is validated against
> TinyLlama-1.1B (Llama-2, cosine 1−3e-13) and Llama-3.2-1B (llama3 scaling
> factor 32, cosine 1−8e-13, tokenizer ids HF-identical). **Next up:** YaRN /
> longrope scaling (currently rejected — Phi-3-128k, long-context Qwen/Yi); a
> **Phi family adapter** (partial rotary is done at the RoPE layer; Phi adds
> parallel attention/MLP blocks + a fused QKV — close to GPT-2 plus rotary);
> **GPT-NeoX/Pythia/GPT-J** (LayerNorm + parallel blocks, on the G5 base);
> **Qwen-MoE/DeepSeek** (shared-expert MoE — a couple more knobs on G6's base);
> **YaRN/longrope** RoPE scaling (currently rejected); more **GGUF quant types**
> (Q5_K/Q3_K/IQ*), the **GGUF tokenizer**, and **GPTQ/AWQ** (the safetensors-
> resident half of G7). Pairing GGUF/quant with M8-style resident int8/int4
> (dequant per-tile in matmul) is what actually runs the big quantized models in
> small RAM. **Mistral, G5/GPT-2, G6/MoE, and G7/GGUF landed 2026-06-03.**
> Pure-Go, parity-validated families now: Gemma 3, Qwen3, Qwen2.5, Llama-2/3,
> Mistral, GPT-2, Mixtral (MoE) — plus **quantized GGUF** (Q8_0/Q4_0/Q4_K_M).
> Per-machine, NOT in git: `.venv/` (torch/transformers/tokenizers), the models
> under `testdata/` (gemma-3-270m, qwen3-1.7b, tinyllama-1.1b, llama3.2-1b,
> qwen2.5-0.5b, tinymistral-248m, gpt2, mixtral-tiny, tinyllama-gguf, *-sharded),
> and `docs/keepers.txt` (HF token — rotate + delete). Plan:
> `docs/multi-model-plan.md`.

## 0. Sanity: a fresh checkout is green

No model assets, no GPU, no Python needed for this:

```bash
go version                 # need 1.26+
go build ./...
go vet ./...
go test ./...              # model-dependent tests SKIP cleanly; everything else passes
```
If that's green you have a correct baseline. The skips are expected (encoder
parity, Gemma loader golden, GPU backend) — they light up once you add the
per-machine assets below.

## 1. ⭐ Validate the amd64 AVX2 path — highest priority

This is the one body of code that has **never executed**. The AVX2/FMA
kernels (`encoder/dot_amd64.{go,s}` — single-row `dotFMA`, register-blocked
`dotFMA4`/`dotFMA8`) and the CPUID detection were written, cross-compiled,
and `go vet`/`asmdecl`-checked on the Mac, but Apple Silicon can't run amd64
SIMD (not even under Rosetta). Your Linux box is the first real execution.

```bash
go test ./encoder/ -run 'AVX2|Dot' -v        # expect hasAVX2=true and all PASS
go test -race ./encoder/...                   # parallel matmul under the race detector
go test ./encoder/ -run XXX -bench 'Dot' -benchmem   # AVX2 vs scalar speedup
```
- **A SIGILL (illegal instruction)** rather than an assertion failure means a
  real asm bug — capture the test name and `grep flags /proc/cpuinfo | head -1`.
- `hasAVX2=false` means that CPU/OS lacks AVX2 (correct, just on the scalar
  fallback — use an AVX2 box, Haswell-2013+ or any modern cloud VM).

Full details + the deferred follow-ups (register-blocked kernel was
**implemented**; tuning, AVX-512, GPU resident buffers remain) are in
[`docs/cpu-acceleration.md`](cpu-acceleration.md).

## 2. Gemma M1 loader — golden parity ✅ VALIDATED 2026-06-02

The loader (BF16/F16 decode + shape-validated weight load) **passes against the
real `google/gemma-3-270m` checkpoint** — `TestLoadWeights_goldenChecksums`
loads the full `gemma3TensorSchema` and matches the Python oracle's
shape/dtype/checksums (≤1e-6 rel). The test **skips** when the checkpoint or
golden is absent; the steps below light it up on a fresh box.

```bash
# Python tooling for the golden (one-time). NOTE the install is split — see gotchas:
python3 -m venv .venv
.venv/bin/pip install huggingface_hub safetensors          # PyPI
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
# (transformers is NOT needed — pin_gemma.py imports only torch + safetensors.)

# gemma is a GATED repo: accept the license once at
#   https://huggingface.co/google/gemma-3-270m
# then authenticate this box (writes ~/.cache/huggingface/token):
.venv/bin/hf auth login                                    # paste an HF *read* token

# Get the checkpoint (~536 MB bf16, single-file) and regenerate the golden:
.venv/bin/python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('google/gemma-3-270m', local_dir='testdata/gemma-3-270m')"
.venv/bin/python scripts/pin_gemma.py        # writes testdata/gemma_golden.json

# Now the loader parity test runs instead of skipping:
go test ./decoder/ -run TestLoadWeights -v   # PASS = shapes + checksums match
```

**Setup gotchas hit on the Linux box (Python 3.14):**
- **Gated repo.** Plain download 401s with `GatedRepoError` until you accept the
  license *and* `hf auth login`. (huggingface_hub 1.x renamed the CLI `huggingface-cli`→`hf`.)
- **Split the pip install.** `pip install safetensors torch --index-url .../whl/cpu`
  fails — the torch CPU index has no `safetensors`, so the whole resolve aborts.
  Install safetensors from PyPI, torch from the CPU index, separately.
- **Python 3.14 is fine.** `torch 2.12.0+cpu` ships a `cp314` wheel. (A harmless
  "Failed to initialize NumPy" warning prints — pin_gemma.py doesn't use numpy.)
- `testdata/gemma-3-270m/` and `gemma_golden.json` is the committed oracle (~2.5 KB);
  the weights dir is gitignored.

Spec: [`docs/milestones/M1-loader.md`](milestones/M1-loader.md). Architecture
+ later milestones (M2 tokenizer, M3 forward pass, …): [`docs/gemma-decoder-plan.md`](gemma-decoder-plan.md).

**The previously-unverified assumption is now RESOLVED.** The tensor *names* in
`gemma3TensorSchema` (`decoder/weights.go`) — `q_norm`/`k_norm` existing,
single-file `model.safetensors` (not sharded) — are all confirmed: `LoadWeights`
validates every schema entry across all layers and returned no error. Shapes
(config-derived) match the real checkpoint exactly.

## 3. Where things stand

| Track | State |
|---|---|
| encoder AVX2 amd64 (+register-blocked kernel) | ✅ validated on Linux 2026-06-02 (Ryzen 7 3700X) — all tests pass, `-race` clean, ~6× single-row; see `cpu-acceleration.md` §A. Open: `Dot8x4` large-K crossover tuning |
| encoder intra-op parallel matmul | done, verified on arm64 |
| encoder/gpu WebGPU (`-tags gpu`) | foundation done; resident-buffer + tiled-kernel follow-ups deferred (loses to CPU until then) |
| ann HNSW, fuse RRF | done, verified |
| decoder M1 loader | ✅ golden parity validated 2026-06-02 against real gemma-3-270m checkpoint (§2); schema assumptions confirmed |
| decoder M2 tokenizer | ✅ done 2026-06-02 — byte-fallback BPE, HF-exact (215k+ inputs, 0 mismatch); see `milestones/M2-tokenizer.md` |
| decoder M3 forward pass | ✅ done 2026-06-02 — logits match HF f32 oracle, cosine 1−1e-12, argmax ' Paris'; see `milestones/M3-forward.md` |
| decoder M4 multi-token decode | ✅ done 2026-06-02 — 48-tok greedy continuation matches HF id-for-id + string; EOS wired; see `milestones/M4-decode.md` |
| decoder M5 sliding window | ✅ done 2026-06-02 — 748-tok prompt matches HF (cosine 1−1e-11); fixed 2 latent window bugs; see `milestones/M5-window.md` |
| decoder M6 sampler + streaming demo | ✅ done 2026-06-02 — temp/top-k/top-p sampling, `demo/gemma` generates from a real checkpoint; see `milestones/M6-sampler.md` |
| decoder M7 perf (shared SIMD linalg) | ✅ done 2026-06-02 — dot kernels lifted to `internal/linalg`, decoder matmul parallel; ~18 tok/s on Ryzen 7 3700X (>10 target); see `milestones/M7-perf.md` |
| decoder M8 int8 quant | ✅ done 2026-06-02 — `--quant int8`, argmax preserved (cosine 0.9996), weights 3.98× smaller; int4 + streaming-quant follow-ups; see `milestones/M8-quant.md` |
| decoder M9 WebGPU backend | ✅ done 2026-06-02 — `--backend webgpu` runs on RTX 2070 (Vulkan), resident weights, argmax parity; trails CPU for M=1 decode (latency-bound); see `milestones/M9-webgpu.md` |
| decoder G0 multi-model descriptor | ✅ done 2026-06-02 — forward pass reads `Architecture` (registry by `model_type`); Gemma goldens byte-identical. Toward running Llama/Qwen/etc. (multi-model-plan G1+); see `milestones/G0-descriptor.md` |
| decoder G1 sharded loader | ✅ done 2026-06-02 — `embed` mmaps + merges N shards (index.json); 3-shard 270m reproduces M1 checksums. Unblocks ≥7B/MoE. See `milestones/G1-sharded.md` |
| decoder G2 Qwen3 + Llama family | ✅ Qwen3 done 2026-06-02 (cosine 1−1e-12 vs HF, untied head). ✅ **Llama dense done 2026-06-03** — `llama` adapter (Qwen3 minus QK-norm, derived `head_dim`) + `llamaTensorSchema`; TinyLlama-1.1B cosine 1−3e-13 (argmax ' Paris'); scaled RoPE + `attention_bias` rejected loudly (G4 / later). See `milestones/G2-qwen3.md` |
| decoder G3 byte-level BPE tokenizer | ✅ done 2026-06-02 — pure-Go byte-level BPE (hand-written GPT-2 split regex + byte→rune map + `ignore_merges`), HF-exact on Qwen3 + **Llama-3** (id-for-id, 20-case golden; merge parser now also reads the older flat `["a b",…]` encoding); M2 Gemma byte-identical. **Qwen3-1.7B generates end-to-end in pure Go**; see `milestones/G3-tokenizer.md` |
| decoder G7 GGUF (quantized) | ✅ done 2026-06-03 — pure-Go GGUF reader (`embed/gguf.go`): container parse + F32/F16/Q8_0/Q4_0/Q4_K/Q6_K dequant; `.gguf` path self-describes (metadata→config) + un-permutes llama.cpp q/k RoPE. TinyLlama Q8_0/Q4_0/Q4_K_M cosine 0.99996/0.9944/0.9975 vs f32 (argmax ' Paris'). See `milestones/G7-gguf.md` |
| decoder G6 MoE (Mixtral) | ✅ done 2026-06-03 — router + sparse experts as a third `mlp()` variant (softmax→top-k→renorm→weighted SwiGLU experts); Mixtral-tiny (8x top-2) cosine 1−1e-13. Loader + int8 quant cover router/experts. See `milestones/G6-moe.md` |
| decoder G5 GPT-2 (LayerNorm class) | ✅ done 2026-06-03 — LayerNorm+bias, learned positions (no RoPE), non-gated GELU MLP, q/k/v + output bias, Conv1D transpose + fused c_attn split (dedicated `buildGPT2Weights`); GPT-2 small cosine 1−7e-14 (argmax ' the'), tokenizer + generation end-to-end in pure Go. See `milestones/G5-gpt2.md` |
| decoder Mistral family | ✅ done 2026-06-03 — llama + all-layer sliding window (Gemma M5 machinery); TinyMistral-248M cosine 1−4e-14 over a 67-tok prompt vs a 32-tok window. See `milestones/G2c-mistral.md` |
| decoder Qwen2/Qwen2.5 family | ✅ done 2026-06-03 — llama descriptor + `QKVBias` knob (q/k/v projection bias) + `qwen2TensorSchema`; Qwen2.5-0.5B matches HF cosine 1−1e-12 (argmax ' Paris'), tokenizer ids identical, generates end-to-end. See `milestones/G2b-qwen2.md` |
| decoder G4 RoPE scaling + partial rotary | ✅ done 2026-06-03 — `rope_scaling` (linear + llama3) baked into a precomputed inv-freq table at load; `partial_rotary_factor` wired through `applyRoPE`; YaRN/longrope/dynamic rejected loudly. **Llama-3.2-1B (llama3 factor 32) matches HF cosine 1−8e-13** + Go tokenizer ids HF-identical = first full Llama-3 pure-Go end-to-end. See `milestones/G4-rope-scaling.md` |
| demo/gemma-web | ✅ done — stdlib net/http + SSE chat GUI over the decoder. **Qwen3 chat wired 2026-06-03**: ChatML template (`<|im_start|>`/`<|im_end|>`) via `ChatStyle()` + multi-EOS stop (151645/151643) from `resolveEOSIDs` reading `generation_config.json`; verified end-to-end. `go run ./demo/gemma-web --model testdata/qwen3-1.7b` |

Reference docs: [`cpu-acceleration.md`](cpu-acceleration.md),
[`gemma-decoder-plan.md`](gemma-decoder-plan.md),
[`milestones/M1-loader.md`](milestones/M1-loader.md).

> Note: assistant *memory* files live under `~/.claude/` and do **not** travel
> with a clone — this doc is the portable handoff. (Local memory on the Mac
> records the GPU-scope-B-on-hold decision and that `cpu-acceleration.md` is a
> living doc; re-note those on the Linux box if you want them there.)
