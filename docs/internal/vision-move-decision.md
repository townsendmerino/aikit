# Decision: move goinfer's SigLIP/ViT vision encoder into aikit

**Status:** cleared — move proceeds. **Date:** 2026-06-12. Roadmap §2.9.

This is **capability acquisition**, not deduplication: the strategy questions
(below) come before any file move. They are answered here first, per the task's
"write these down before moving anything" gate.

## Phase 0 — gate

Stated trigger (§2.9): *launch feedback or an adopter asking for image/multimodal
retrieval.* The r/golang post has not yet generated that signal. **Francis sending
the task is the owner override** — recorded here and in the CHANGELOG/roadmap. The
strategy questions are answered regardless of the override, as required.

## Phase 1 — strategy decisions

### 1. Dependency-claim check — **PASS, no weakening**

aikit's promise (README:4): *"no cgo in the core (stdlib + `golang.org/x/text`
only)."* The risk was the image-decode path pulling `golang.org/x/image` or a cgo
codec.

Audit of `vision/preprocess.go`: the decode path imports **only stdlib** —
`image`, `image/draw`, and the registered `image/jpeg` + `image/png` decoders.
No `x/image`, no cgo. The package also reads the header (`image.DecodeConfig`) and
bounds pixel count *before* decoding, so a decompression bomb errors instead of
OOMing (the Track-2 security posture moves with it).

Across the whole package the external deps are exactly: stdlib, `aikit/embed`,
`aikit/linalg`. **vision introduces no new external dependency** — the lone
`x/text` continues to enter only via `embed`. The claim is preserved verbatim;
no decode quarantine (treesitter-style) is needed.

### 2. Scope line — encoder + preprocessing move; projector stays

**Moves to `aikit/vision`** (the encoder + its preprocessing/IO/GPU-seam):
`encoder.go`, `preprocess.go`, `load.go`, `qmat.go`, `resident.go`,
`gpu_export.go` (+ `encoder_test.go`, `preprocess_test.go`, `fuzz_test.go`).

**Stays in goinfer** (Gemma-specific connector + decoder glue):
- `projector.go` — `Gemma3MultiModalProjector` (vision last_hidden_state → image-token
  embeddings the LLM interleaves). Gemma-specific op order, pooling, RMSNorm.
- `gemma3_block.go` — the `<image_soft_token>` / BOI/EOI sentinels feeding the
  decoder's embed-by-vector seam. Pure Gemma prompt-assembly.

**Boundary is clean (abort criterion checked):**
- `vision/` has **zero** goinfer-internal imports (only stdlib + embed + linalg).
- No goinfer `decoder/` file imports `vision` — vision types do **not** leak into
  the decoder. The decoder seam takes *projected* features as plain `[]float32`.
- `Projector.Forward(visionHidden []float32) ([]float32, error)` references **no
  encoder type** — the projector↔encoder boundary is a `[]float32`. After the move
  goinfer's leftover imports `aikit/vision` only for the loader/types it actually
  uses; the connector itself needs nothing from the encoder's internals.
- The one shared-package helper the leftover uses is `openWeights` (load.go). load.go
  moves; goinfer's leftover gets a 10-line local `openWeights` over `aikit/embed`
  (already a goinfer dep), or aikit exports it. Minor, not tangled.

goinfer's leftover (projector + gemma3_block) is renamed `package multimodal` to
avoid two packages both named `vision` (goinfer aliases none today; the rename is
goinfer-internal and import-path-local).

### 3. What works without a text tower — **be honest in docs**

Day one, with the vision encoder in aikit:
- ✅ **image→image** similarity (embed two images, cosine).
- ✅ **image-as-document** in existing aikit pipelines (an image embedding is just
  a vector — index it in `ann`, fuse it, rerank around it).
- ❌ **text↔image** retrieval does **not** work. SigLIP pairs a vision tower with a
  SigLIP *text* tower; Gemma's pipeline uses the LLM for the text side, which aikit
  doesn't have. No SigLIP text tower exists in either repo.

The text tower is scoped as an **explicit follow-up** (roadmap, new gated item). It
is BERT-shaped, so `encoder`'s transformer machinery + the parity toolchain make it
tractable — but it is its own parity-pinned work (its own golden, its own pin
script). Trigger: someone actually needs text↔image, not just image→image.

### 4. Package name + tier

- **Package:** `vision` (in aikit). **Tier:** Experimental (outside the 1.0
  semver guarantee), same as encoder's newer surfaces.
- **DAG:** adds one edge, `vision → {embed, linalg}` — a new leaf consumer, no
  new external dependency, DAG stays shallow.
- **apidiff:** additive (a brand-new package; nothing existing changes).

## Phase 2 — the move (mechanics, executed after this doc)

1. Copy the eight files into `aikit/vision/` with attribution (shared NOTICE
   conventions). The import-free GPU seam (`GPUMat`/`GPULayer`/`GPUWeights`) moves
   intact and **stays import-free**.
2. Parity assets: `scripts/pin_siglip_vision.py` → aikit `scripts/`;
   `siglip_vision_golden.json` (74 KB of numbers, no image bytes) →
   aikit `testdata/`. The tiny `siglip-tiny` checkpoint stays model-gated
   (not committed); tests **skip-clean** when it is absent (CI has no model). A
   tiny deterministic pixel fixture covers preprocess without an image file.
3. goinfer: `go.work replace` for dev → validate the multimodal path green against
   aikit's copy → **then** delete goinfer's encoder copy, rename the leftover to
   `package multimodal`, point `gpu/vision_*.go` at `aikit/vision`, bump the pin on
   release.

**Acceptance:** parity unchanged (f32 cosine 1.0, int8 W8A8 ~0.9999 vs HF golden);
goinfer multimodal green against aikit's copy *before* goinfer's copy is deleted;
`-race`, cgo-free, windows cross-build, apidiff additive.

## Follow-ups unblocked by the move

- **`vision.qmat` → `linalg.WeightMat`** (the other in-flight unification). Once
  vision lands in aikit, this is a pure in-aikit refactor co-located with WeightMat
  — no longer a cross-repo step. Move qmat **verbatim** now (parity-preserving);
  migrate to WeightMat separately.
- **SigLIP text tower** — the gated item above; the thing that turns image→image
  into true text↔image retrieval.

## Abort criteria — not triggered

Image decode did **not** force `x/image`/cgo into core (stdlib only); the projector
boundary is a `[]float32`, not a tangle. Both quarantine bars (treesitter, Backend)
are met. Proceed.
