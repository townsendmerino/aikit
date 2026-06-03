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

## 2. Gemma M1 loader — golden parity

The loader (BF16/F16 decode + shape-validated weight load) is done and its
test **skips** until the checkpoint + golden are present. To exercise it:

```bash
# Python tooling for the golden (one-time):
python3 -m venv .venv && . .venv/bin/activate
pip install torch safetensors transformers huggingface_hub

# Get the checkpoint (~340 MB bf16) and regenerate the golden:
huggingface-cli download google/gemma-3-270m --local-dir testdata/gemma-3-270m
.venv/bin/python scripts/pin_gemma.py        # writes testdata/gemma_golden.json

# Now the loader parity test runs instead of skipping:
go test ./decoder/ -run TestLoadWeights -v   # PASS = shapes + checksums match
```
Spec: [`docs/milestones/M1-loader.md`](milestones/M1-loader.md). Architecture
+ later milestones (M2 tokenizer, M3 forward pass, …): [`docs/gemma-decoder-plan.md`](gemma-decoder-plan.md).

**One unverified assumption** (couldn't be checked without the real
checkpoint): the tensor *names* in `gemma3TensorSchema` (`decoder/weights.go`)
— `q_norm`/`k_norm` existing, single-file `model.safetensors` (not sharded).
Shapes are config-derived and match the plan. If a name/shape is wrong the
loader fails with a precise `tensor %q ... shape %v != want %v` — that's the
signal to fix the schema.

## 3. Where things stand

| Track | State |
|---|---|
| encoder AVX2 amd64 (+register-blocked kernel) | written; **validate on Linux (§1)** |
| encoder intra-op parallel matmul | done, verified on arm64 |
| encoder/gpu WebGPU (`-tags gpu`) | foundation done; resident-buffer + tiled-kernel follow-ups deferred (loses to CPU until then) |
| ann HNSW, fuse RRF | done, verified |
| decoder M1 loader | done; **golden parity needs checkpoint (§2)** |
| decoder M2+ (tokenizer, forward, sampler) | scaffold only — stubs return errNotImplemented |

Reference docs: [`cpu-acceleration.md`](cpu-acceleration.md),
[`gemma-decoder-plan.md`](gemma-decoder-plan.md),
[`milestones/M1-loader.md`](milestones/M1-loader.md).

> Note: assistant *memory* files live under `~/.claude/` and do **not** travel
> with a clone — this doc is the portable handoff. (Local memory on the Mac
> records the GPU-scope-B-on-hold decision and that `cpu-acceleration.md` is a
> living doc; re-note those on the Linux box if you want them there.)
