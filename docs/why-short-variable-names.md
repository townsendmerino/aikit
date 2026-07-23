# Why the short variable names?

aikit uses a lot of one- and two-letter locals — `M`, `N`, `K`, `q`/`k`/`v`,
`sb`, `ql`/`qh`, `b`/`off`/`pos`. This is deliberate, not sloppiness. This doc
explains the philosophy so a reviewer knows what to leave alone and what to
actually flag, because "make it longer" is usually the wrong fix.

The governing rule is not *long* — it's:

> **A name should be sized to its scope, consistent with its neighbours, and
> unambiguous in context.**

Short names satisfy that rule constantly; long names violate it about as often.

## Why short is right in the kernels

Most of `linalg`, `embed`'s dequant, and the `encoder`/`vision` forwards are a
transcription of math — a matmul, a RoPE rotation, a GGML quant formula. In that
setting the short letters *are* the domain vocabulary: `M/N/K` are the matmul
dimensions, `i/j/k` the indices, `q/k/v` attention, `alpha/beta` the BLAS
scalars, `ql`/`qh`/`qs`/`dl`/`ml` the GGML block fields. When the code mirrors
the reference notation one-to-one you can check it against the paper or the ggml
source by eye. Rename the contraction dimension `K` to `contractionDim` through a
tight kernel and you haven't helped anyone who knows linear algebra — you've
turned a clean transcription into something they have to mentally un-translate.

## Scope-proportionality

Go leans on this harder than most languages. One of the Go proverbs:

> The greater the distance between a name's declaration and its uses, the longer
> the name should be.

A variable born and consumed two lines later inside a four-line loop carries
almost no memory burden, so a descriptive name there is noise you read past every
time. An exported, package-level symbol is the opposite and earns a full name.
So `i` in a small loop and `WrapInt4` at the API boundary are *both* sized
correctly. Blanket-lengthening breaks that proportionality.

Related: don't restate the type. `func (f *SafetensorsFile)` doesn't need
`safetensorsFile`; `for _, op := range ops` doesn't need `operation`. The
context already says it.

## Long names have their own failure modes

Verbose names are not free:

- **Scannability drops.** Near-identical long names that share a prefix
  (`inputActivationRowBuffer` vs `inputActivationColBuffer`) are harder to tell
  apart at a glance — and easier to transpose — than `aRow`/`aCol`.
- **They lie louder when they drift.** A descriptive name makes a stronger
  claim, so a wrong one misleads more confidently. See the `bs` case below: a
  name that *reads* like a byte size but holds an element count is worse the more
  authoritative it sounds. The fix is accuracy, not length.

## So when *is* a short name wrong?

Length is never the defect on its own. Flag a name only when it hits one of
these — and the fix is "unambiguous," which is often still short:

1. **Ambiguous out of context.** `bs` — byte-slice or block-size? If a reader
   can't resolve it from the line, rename it (`blockElems`, `bScale`) — but a
   two-letter name that's unambiguous in its scope is fine.
2. **Overloaded within one file/function.** `h` meaning grid-*height* in one
   function and hidden-*state* in another (see `vision/qwen_encoder.go`) forces a
   double-take every time. Give the two distinct concepts distinct names
   (`gridH` vs `h`).
3. **Inverted against our own convention.** `hd` is head-*dimension* everywhere
   in the kit — using it as a head-*index* in one file (`bert.go`) reads
   backwards. Match the established name (`headIdx`).

Notice the fixes — `fn`, `blk`, `gridH`, `headIdx` — are still short. The bar is
clarity and consistency, not verbosity.

## The house vocabulary

These short names are established across the codebase; learn them once and read
everything. Reviewers should leave them as-is:

| Name(s) | Meaning |
|---|---|
| `M`, `N`, `K` | matmul dims: rows of A, cols of B, shared/contraction dim |
| `i`, `j`, `k` | loop indices (row, col, inner) |
| `q`, `k`, `v` | attention query / key / value |
| `hd`, `L`, `D` | head dimension, sequence length, hidden size |
| `alpha`, `beta` | BLAS-style scaling scalars |
| `ql`, `qh`, `qs`, `dl`, `ml` | GGML quant block fields (low/high nibbles, scales, mins) |
| `b`, `off`, `pos` | byte cursor, offset, position in a parser |
| `ef`, `M`, `mL`, `m0` | HNSW hyperparameters (paper notation) |

When in doubt, size the name to its scope and match its neighbours. Reach for a
longer name when the scope is wide or the meaning genuinely isn't recoverable
from context — not by default.
