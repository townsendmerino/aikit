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
| demo/gemma-web | ✅ done — stdlib net/http + SSE chat GUI over the decoder (`go run ./demo/gemma-web --model testdata/gemma-3-270m`) |

Reference docs: [`cpu-acceleration.md`](cpu-acceleration.md),
[`gemma-decoder-plan.md`](gemma-decoder-plan.md),
[`milestones/M1-loader.md`](milestones/M1-loader.md).

> Note: assistant *memory* files live under `~/.claude/` and do **not** travel
> with a clone — this doc is the portable handoff. (Local memory on the Mac
> records the GPU-scope-B-on-hold decision and that `cpu-acceleration.md` is a
> living doc; re-note those on the Linux box if you want them there.)
