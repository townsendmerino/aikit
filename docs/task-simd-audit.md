# Task: SIMD audit — is the CPU substrate using the hardware well? (2026-09-03)

> **BLUF.** At M=1 on the Mac, yes: the production W4A8 decode kernel
> (`dotW4A8SplitHalf4Row`) runs at 91–97% of its per-core issue ceiling, and the W8A8 kernel at
> 96% of its latency bound. What is *not* using the hardware well sits around the kernels, not
> inside them: (1) **there is no register-blocked GEMM for M>1 on either arch** — prefill and
> speculative verify call the single-row GEMV once per (activation row, weight row) and pay the
> nibble unpack and the scale fold M times, which is the mechanism behind "int8int8 prefill
> beats int4 by 25–33%" and most of the 2.98× CPU prefill gap to Ollama; (2) **decode at 6–8
> workers is fan-out-limited, not kernel-limited** — each core delivers 26–35% of what it does
> alone, on both boxes; (3) **the f64 attention kernels are pure Go** and a lane-per-output NEON
> port is bit-identical by construction, worth ~2× on QK and ~3× on P·V at depth 8k where they are
> ~75% of the token; (4) **the activation quantiser and the transcendentals are scalar and serial**
> on the calling goroutine, 8–10% of a short-context token. Everything proposed below keeps the
> bit-identity contract or says exactly where it does not; every measured negative in
> `docs/internal/perf-dead-ends.md` and goinfer's campaign records was read first and is not
> re-proposed.
>
> **Status: SCOPED 2026-09-03. Done so far (all 2026-09-03): the four no-hardware items —
> S-09.2, S-09.3, S-10.1, S-10.2; S-08.1 (the 3700X's bandwidth, and a 30% correction to
> `gpu-assessment.md` — **applied at the source 2026-09-03, goinfer `72df161`**); S-04 step 1 (the pure-Go AV
> accumulator blocks, 2.35× arm64 / 1.36–1.46× amd64); **S-01's arm64 half** — the M-invariance
> gate and then the register-blocked 4×4 tile, `dotW4A8Row4Tile4x4`, measured 2.88× on the
> prefill kernel; and **S-01b's arm64 half** — `dotI8Tile4x4`, 3.5–3.9× on W8A8 at M≥4 in every
> cache regime. Appendix B's SDOT-pipe question is measured and closed (four pipes, 3.98/cycle).
> See the annotation on each. Still open: S-01/S-01b's amd64 halves, the S-04 NEON ports,
> S-02/S-03/S-05/S-06/S-07, and the items needing hardware neither box has (I8MM, VNNI). S-09.1
> needs goinfer's decode bench in their repo.**
>
> **UPDATE 2026-09-03: S-01 and S-01b are COMPLETE and CLOSED on both arches, end to end.** The
> amd64 halves were built and measured on an idle 3700X — W4A8 1.65–1.90×, W8A8 1.17–1.44×. The
> W8A8 amd64 tile's first shape regressed 1.75× on streamed B and was replaced; see the S-01b
> annotation, which is the most useful thing in this document for whoever writes the next kernel.
> goinfer then measured the end-to-end prefill cell and **both pre-registered gates pass — 1.66×
> before/after and the int4/int8int8 inversion is gone.** S-09.1 is also done and refuted the
> 2026-08-11 result: fork/join is worth **1.70×** on decode, which confirms S-02's magnitude and
> leaves ~2.7× on the table that bandwidth does not explain. **Step 0 is now one item: S-02's
> per-shard timestamp harness.**
>
> **UPDATE 2026-09-03, later (Cowork): S-03, S-04 step 2 and S-07 are BUILT and gate-checked,
> unmeasured.** The NEON activation quantiser (`quant_act_arm64.s`), the NEON f64 ports of both
> attention kernels (`attn_acc64_arm64.s`) and the eight-column `q8Span` (`q8span_arm64.go`) are
> in the tree, each bit-identical to what it replaces by construction and by a raw-bit gate run on
> arm64 under qemu-user with DotProd on — see the annotation on each. The Mac benches are the next
> step (§5). The S-01 tiles were read back against their claims in the same pass: no defect, at
> 99% of a four-pipe issue ceiling, and the S-05 fold is the one lever left in them — 1.33×
> counted; see the S-01 read-back.
>
> **Original status: nothing started.** Static read of aikit `v1.31.0-5-gca817a4`
> (`linalg/`, 16 assembly files, the Go dispatch, the docs and bench records) and goinfer
> `30b168e` (the callers). Three independent reviewers (arm64 assembly, amd64 assembly, Go
> side + coupling), instruction counts by hand, every raw `WORD` encoding recomputed. Nothing
> was built or run — the review sandbox has go1.24 against `go 1.27`, no arm64, no GPU — so
> every throughput figure is either quoted from the two repos with its box label or derived
> from an instruction count with the arithmetic shown, and labelled as such. Supersedes nothing;
> extends `go127-simd-audit.md` (2026-08-20, in `~/tmcode/go127-simd-audit/`) — §4 reconciles.

**Boxes.** `apple-m1pro`: 6 P + 2 E cores, FEAT_DotProd (SDOT), **no FEAT_I8MM** (SMMLA is
M2+); STREAM triad 71.9 GB/s one thread, 121 at six. `nvidia-rtx2070s` host: Ryzen 7 3700X (Zen
2: AVX2+FMA, no VNNI, no AVX-512; 4.2 GHz implied by the FMA probe).

> **S-08.1 MEASURED 2026-09-03 — the 3700X's bandwidth gap is closed, and the standing estimate
> was 30% high.** Idle box (load 0.07), 480 MB arrays past the 32 MB L3, pthread, best-of-5:
>
> | probe | 1 thread | 2 | 4 | 8 | 16 |
> |---|--:|--:|--:|--:|--:|
> | **pure read** (weights are read-only — the decode-relevant one) | 22.6 | **30.5** | 30.0 | 28.9 | 27.5 |
> | **STREAM triad** (2R+1W, reported as 24 B/iter) | 19.4 | 21.2 | 20.6 | 19.8 | 19.3 |
>
> GB/s. Ceiling for reference: DDR4-3200 dual channel = 2 × 8 B × 3200 MT/s = **51.2 GB/s**, so
> reads reach ~60% of theoretical — unremarkable for Zen 2. DIMM speed is UNCONFIRMED (no sudo
> for `dmidecode`), so 51.2 is the assumed config, not a verified one.
>
> **Read, not triad, is the right anchor for a decode roofline.** Triad also writes, and unless
> the store is non-temporal the line is fetched for ownership first, so triad moves ~32 B of DRAM
> traffic per 24 B it reports — understating a read-only ceiling by ~1.33×. That reconciles the
> two: 21.2 triad ≈ 28 GB/s of actual traffic ≈ the 30.5 read figure.
>
> **This settles the two anchors this paragraph used to lean on.** Ollama's 27.6 GB/s of weights
> (P14) is **~90% of the measured read ceiling** — not anomalous, close to optimal. The
> "~40 GB/s" estimate in `gpu-assessment.md` is **too high by ~30%** and should be corrected to
> ~30 GB/s. ✅ **DONE — goinfer `72df161` carries the correction.**
>
> **It also sharpens §1's amd64 verdict rather than contradicting it.** "2–4 cores already exceed
> any plausible DRAM figure" is right, and tighter than stated: read bandwidth SATURATES AT TWO
> THREADS and then declines slightly. More decode workers cannot buy memory bandwidth on this
> box.
>
> A first attempt at the read probe reported 135 GB/s single-thread and 616 at 16 — the compiler
> had folded the reduction over a constant-filled array. It was caught by comparing against the
> 51.2 GB/s ceiling, the same discipline that caught `gpu`'s 8145 GB/s DtoD reading, and fixed
> with a data-dependent fill plus a volatile sink. A bandwidth probe without a ceiling assert is
> not a probe.

**Reference model** throughout: Qwen2.5-Coder-1.5B (hidden 1536, inter 8960, 28 layers, 12 q
heads / 2 kv heads, hd 128, vocab 151936); ~1.05 GB of int4 weights + scales per decode token.

---

## 1. Verdict by regime

| regime | share of a token | what runs | verdict | lever |
|---|---|---|---|---|
| **decode M=1, arm64** (Mac, 1.5B, depth ≤ 512) | ~85% weight stream | `dotW4A8SplitHalf4Row` (SDOT, 4 rows, f32 fold in register), W8A8 head via `dotI8SDOT` | **kernel at 91–97% of its issue ceiling per core; the 6-worker aggregate at 26–35% of six cores** | S-02 (fan-out), S-05 (9→7 µops), S-03 (batch entry) |
| **decode M=1, amd64** (3700X) | same | `dotW4A8FoldAVX2` (25 instr/32 MACs, 1 acc), `dotI8AVX2` (4 acc) | kernel at ~53% of its port floor but **2–4 cores already exceed any plausible DRAM figure**; 16 SMT shards per call | S-02, S-08 (measure the box first) |
| **prefill / verify M>1, both archs** | the whole prefill | `w4a8Span`: `for col { for row { dotW4A8(row, col) } }` — the M=1 GEMV per pair | ✅ **FIXED ON BOTH ARCHES 2026-09-03** — arm64 `dotW4A8Row4Tile4x4` **2.88×** (24.1 → 69.5 GMAC/s); amd64 `dotW4A8Tile4RowAVX2` **1.65–1.90×** (15.6 → 27.6). W8A8 too: arm64 3.5–3.9×, amd64 1.17–1.44×. | **S-01** (2.7× counted; 2.88× measured arm64, 1.78× amd64 — AVX2's 16 registers cap the tile at 4×1) |
| **decode attention, depth ≥ 2k** | 75% of the token at 8k | `MatmulQKAcc64` / `MatmulAVAcc64` — pure Go f64, 8 scalar chains / memory accumulators | QK at 80% of its scalar issue bound; AV at ~40% (accumulators in memory) | **S-04** (NEON f64 lane-per-output, bit-identical) |
| **Go-side serial work** | 8–10% at short context | `quantizeRowInt8Core` ×7 per layer (3 redundant), `silu` via f64 `math.Exp`, norms, fork/join | scalar, single goroutine, between parallel matmuls | S-03, S-06 |
| **weight-only `int8` mode** | every projection | `q8Span` → `dequantRowInt8` + `dotNEON4` (**one** FMLA chain) | 3.2 GMAC/s per core — the mechanism behind the LM head's old 11–13 GB/s | S-07 (bit-identical 8-column form) |
| **f32 GEMM** (prefill attention, f32 models) | attention share of prefill | `blockedFill` → `Dot2x8` / `dotFMA3x4` | ~95% of the measured FMLA peak on arm64; 55–64% on amd64 (no B-panel packing there) | S-08 (amd64 packing at power-of-two K) |

The instruction accounting behind each row is in §2; the sources of every measured figure are in
§6.

## 2. Kernel inventory — what executes per 32 int4 MACs (16 packed bytes)

**arm64** (per P-core, 4 SIMD pipes; counts per 16 weight bytes unless stated)

| kernel | selected when | instr / 32 MACs | SIMD µops | accumulators | binding resource | ceiling per core | measured | fraction |
|---|---|---:|---:|---|---|---|---|---|
| `dotW4A8SplitHalf4Row` (`dot_w4a8_arm64.s:444-532`) | M==1 and the tensor has a row4 layout (`weightmat_row4_arm64.go:48-54`; every 0.5B/1.5B projection qualifies; kind-4 `.giw` aliases it) | 12.75 (load 2.25, unpack 4, SDOT 2+MOVI, fold 2, loop 1.5) | 9 | 4 f32 chains, one per row | 9 µops / 4 pipes = 2.25 cy | 45.5 GMAC/s = 22.8 GB/s packed, 28.4 with scales | hot ≈44 GMAC/s; cold 42.4–42.7; 1 worker 25.9–26.2 GB/s | **91–97%** |
| same, 6 workers | — | — | — | — | not the kernel | 6 × 28.4 = 170 issue; 121 STREAM | 59.6–60.6 isolated; **44.9 in decode** | 35% / 26% |
| `dotW4A8FoldSDOT` (canonical, `:43-81`) | **every M>1 call**, non-row4 tensors, paged MoE | 17 (load 3, unpack 6 incl. ZIP1/ZIP2, SDOT 2+MOVI, fold 2, loop 3) | 11 | 1 f32 chain | FMLA latency ≈4 cy/group | 25.6 GMAC/s | 24.5–25.0 | 96–98% of the latency bound |
| `dotI8SDOT` (`dot_i8dp_arm64.s:22-76`, W8A8) | every W8A8 call at M<4, and the M%4/N%4 strips above it | 3.5 per 16 int8 MACs | 1 | 4 int32 | 1 SDOT / cy (4 acc × 4 cy) | 51 GMAC/s = 51 GB/s | 49.2 | 96% (64% of the load bound) |
| `dotI8Tile4x4` (`dot_i8_tile_arm64.s`, **added 2026-09-03**, S-01b) | every W8A8 call at M≥4, arm64 + DotProd | 1.5 per 16 int8 MACs (8 loads + 16 SDOT per 256) | 1 | **16 int32** | SDOT issue, **measured 3.98/cy** | 203.6 GMAC/s (`TestSDOTIssuePeak`) | 163.5 | **80%** |
| `dotNEON4` (`dot_arm64.s:25-45`, `dotF32`) | weight-only int8/int4 (`q8Span`, `q4Span`) | 5 per 4 f32 MACs | 1 chain | 1 | FMLA latency | 3.2 GMAC/s | LM head 11–13 GB/s over 6–8 cores | consistent |
| `dotNEON2x8` (`dot2x8_arm64.s`) | f32 GEMM row pairs (`blockedFill`) | 16 FMLA / 64 MACs | 16 | — | FMLA issue | 102 GFLOPS theo., 95.4 measured peak | ~42 GMAC/s | ~95% |
| `MatmulQKAcc64` (Go, `matmul_qk_acc64.go:21-67`) | decode attention QKᵀ | 26 instr / 8 f64 MACs (9 loads, 9 FCVT, 8 FMADD) | 17 FP | 8 f64 chains | FP issue | 6.0 GMAC/s | 4.8 | 80% |
| `MatmulAVAcc64` (Go, `matmul_av_acc64.go:33-52`) | decode attention P·V | ≈7 instr / MAC (`acc[d]` in memory: load, load, FCVT, FMADD, store, index) | — | hd in memory | memory round-trip per MAC | ~1.9 GMAC/s | 1.86 | at its own bound |
| `quantizeRowInt8Core` (Go, `quant.go:43-71`) | before every W8A8/W4A8 matmul, on the caller | two scalar passes | — | — | scalar chain, 4–10 cy/elem | — | 509k elem/token on the 1.5B | est. 0.6–1.6 ms/token |

Not wired (all harness-only, all measured): `FoldSDOTv2` (uncentered + f32 correction, 0.972×,
not bit-identical), `SplitHalfSDOT` (1.000×), `2Acc/4Acc` (1.4–1.75×, not bit-identical — a
row's fold split across chains), `SplitHalf4RowPrefetch` (1.000× at every distance),
`Deshared` (0.993×). `kquant.go`'s native Q4_K/Q6_K matmul (measured negative, `:5-10`). No
kernel detects or uses I8MM; UDOT is unused; no alignment is assumed anywhere (`LD1 .B16` cannot
fault; `.giw` aliases nibbles at an arbitrary byte offset, scales are heap-copied to 4-B alignment).

**amd64** (Zen 2, 4 × 256-bit FP pipes, `VPMADDWD` on one; per 32 MACs)

| kernel | selected when | instr / 32 MACs | FP µops | accumulators | port floor | ceiling per core | measured | fraction |
|---|---|---:|---:|---|---|---|---|---|
| `dotW4A8FoldAVX2` (`dot_w4a8_amd64.s:33-88`) | every W4A8 call, canonical layout (Zen 2 has no VNNI) | 25 (4 load, 9 unpack, 2 act widen, 3 MAC, 2 fold, 5 loop) | 16 (6 shuffle-class) | **1** f32 (`Y10`) | 4.2 cy/group | 32 GMAC/s = 16 GB/s packed | 17.1 hot / 16.2 cold | 53% |
| `dotW4A8SplitHalfAVX2` (`:243-292`) | M==1 with `GOINFER_W4A8_SPLITHALF=1` (default off) | 23 | 14 | 1 | 3.8 | 35 | 19.2 / 18.3 | 55% |
| `dotI8AVX2` (`dot_amd64.s:323-387`) | every W8A8 call | 10.5 | — | 4 int32 | 4.0 cy / 64 MACs | 67 GMAC/s | 48.3–51.3 | 72–76% |
| `dotW4A8FoldAVX512VNNI` / `dotI8AVX512VNNI` | AVX512F+BW+VNNI(+VL) hosts only — neither project box, not Intel client parts (VEX AVX-VNNI undetected) | 23 / 4 | — | 2 f32 / 4 | — | — | +1.23–1.28× on a cloud Xeon | — |
| `dotFMA3x4` / `dotFMA8` (f32 GEMM) | `blockedFill` | 10 / 11 | — | 12 / 8 | FMA issue | 67 / 54 GMAC/s | 43.2 (M32 GEMM) / 42.8 | 64% / 79% |
| `dotFMA` (single f32 dot) | `q8Span`, `q4Span`, `Dot` | 18 | — | 4 | 2 loads / FMA | 34 GMAC/s | — | — |

The W4A8 AVX2 kernel does **not** use `VPMADDUBSW`: both operands are sign-extended to int16
and multiplied with the signed×signed `VPMADDWD`, so no saturation question exists (|pair| ≤
2048). The measured 2Acc/4Acc null result (~0.5%) is real; the recorded *explanation* — "Zen 2
cracks every 256-bit op into two 128-bit µops, already at that floor" — is a Zen 1 property, and
the repo's own `dotI8AVX2` rate (16 vector instructions per 5.6–5.9 cycles) is impossible under
it. The set of observations (1Acc ≈ 2Acc ≈ 4Acc; −2 µops → +12%; −8 µops → +57%; idle FMA ports)
fits a kernel limited by how many ~24-cycle per-group dependency chains the FP scheduler holds
in flight, not by a port and not by the loop-carried chain (S-08 says what to measure).

## 3. Findings, ordered by prize

Each carries the mechanism, what to measure before believing it, and the bit-identity
consequence. "Bit-identical by construction" means every output's reduction is the same
sequence of floating-point operations as today — the property goinfer's `decode == batched
prefill == speculative verify` guarantees rest on.

### S-01 · Perf-major (prefill, both archs) — no register-blocked int4/int8 GEMM; M>1 is the M=1 GEMV per (row, column)

> **THE GATE IS IN, 2026-09-03 — and the audit's claim about it was right.** §5 step 1 calls the
> W4A8 M-invariance gate "the thing that makes the rest safe to ship"; it now exists, in
> `linalg/matmul_quant_mconsistent_test.go` (portable, both arches). The claim it rests on was
> verified rather than assumed: `TestMatmulBT_MConsistent` in `matmul_mconsistent_test.go` was
> the ONLY M-invariance gate in the package, and it covers f32 `MatmulBT` only.
>
> Three gates, because there are three distinct contracts and the existing tests pin none of them:
>
> | gate | pins |
> |---|---|
> | `TestMatmulBTW4A8_MConsistent` | canonical W4A8: row *i* of an M-row batch `==` that row alone at M=1 |
> | `TestMatmulBTW8A8_MConsistent` | the same for W8A8 — S-01b proposes an M-tile there too |
> | `TestWeightMatW4A8_MConsistentAcrossRow4Dispatch` | `WeightMat.MatmulBTW4A8Into`, which routes M=1 to the **row4** kernel and M>1 to the **canonical** one |
>
> **The third is the one that was actually missing**, and it is worth naming why. Two row4 gates
> already existed: `TestMatmulBTW4A8Row4Into_bitIdenticalToMatmulBTW4A8Into` pins row4 `==`
> canonical **at M=1**, and `TestWeightMat_RepackInt4Row4_dispatchMatchesCanonical` pins dispatch
> `==` canonical **at a fixed M**. Neither pins the *diagonal* — row4-at-M=1 against
> canonical-at-M>1 — which is exactly the pair speculative verify exercises (draft proposes at
> M=1 through row4, target verifies at M=K through canonical).
>
> **Sweep:** M ∈ {1,2,3,4,5,7,8,9} — chosen for the residues mod 4, since a 4×4 tile runs four
> activation rows at a time and hands 1/2/3 to a narrower remainder path. Shapes: both production
> projections (1536×8960 and 8960×1536, the row4-eligible ones), an N=6 case where row4 is
> rejected, a K=100 ragged-group case the row4 layout cannot represent, and a small shape that
> stays under the parallel threshold. The W4A8/W8A8 gates run each M twice — default threshold and
> a forced `SetThreshold(1)`/`SetWorkers(4)` fan-out — because a future M-tile can differ across
> that decision.
>
> **Verified failure-capable, not merely green** (the `analyzer-canaries.md` discipline). A canary
> that perturbs the M>1 result by **one ULP** — `Float32bits(x)+1` in `w4a8Span` and `w8a8Span` —
> makes all three gates fail at **every** shape, and the failure message reports `repacked=true`
> for the production shapes and `false` for the N=6/ragged ones, which is independent confirmation
> that both dispatch branches are genuinely being taken. Reverted; full `linalg` suite green.
>
> Today the property holds trivially — `w4a8Span` loops activation rows *inside* the column loop,
> so every output is its own independent reduction. That triviality is the point: the gate is
> written while the implementation still satisfies it by construction, so the tile is measured
> against a contract that predates it.
>
> **THE ARM64 TILE IS BUILT AND MEASURED, 2026-09-03 — 2.88×, bit-identical, and it beat the
> count.** `dotW4A8Row4Tile4x4` (`linalg/dot_w4a8_tile_arm64.s`) reduces FOUR activation rows
> against the four weight rows of one row4 quad per call — 16 outputs, 16 live f32 accumulators.
> Entry point `MatmulBTW4A8Row4TileInto`; `WeightMat.MatmulBTW4A8Into` now routes every M>1 to it
> when the tensor is row4-resident, and M=1 is untouched (decode keeps `dotW4A8SplitHalf4Row`).
>
> **Measured on `apple-m1pro`, K=1536 N=8960 (the gate/up projection), three interleaved paired
> passes, median** (`TestW4A8TileVsCanonicalAB`; the box was NOT idle — load ~2.3, ~0.4 cores
> busy — which matters little for the single-core arm and is why the parallel row is the softer
> number):
>
> | M | serial canonical | serial tile | | parallel canonical | parallel tile | |
> |--:|--:|--:|--:|--:|--:|--:|
> | 1 | 24.0 GMAC/s | 41.7 | 1.74× | 23.6 | 41.6 | 1.77× |
> | 2 | 24.1 | 42.3 | 1.75× | 79.8 | 131.9 | 1.65× |
> | 4 | 24.1 | **69.5** | **2.88×** | 91.4 | 221.0 | 2.42× |
> | 8 | 24.1 | 69.4 | 2.88× | 91.6 | 223.4 | 2.44× |
> | 16 | 24.1 | 69.2 | 2.87× | 95.7 | 234.3 | 2.45× |
> | 64 | 24.1 | 69.5 | 2.89× | 96.0 | 251.3 | **2.62×** |
>
> **Against the pre-registered gate — ship at ≥1.5× on the int4 prefill cell — this clears it by
> nearly 2×, and it clears the counted estimate too:** this section counted 2.7× and 21 MACs/cycle.
> Measured is 2.88× and 23.78 cycles per 32-k group against a modelled 24.0 (96 SIMD µops ÷ 4
> pipes at 3.228 GHz = 68.9 GMAC/s modelled, 69.5 measured — inside 1%). That is the FIRST time in
> this campaign an instruction-count argument has landed on the high side; the roofline record's
> note that such arguments overpromised three times is why the confidence line said "medium on
> 2.7×", and it is worth recording that the count was conservative here rather than optimistic.
>
> **It does NOT settle Appendix B's SDOT-pipe question, and the hedge attached to it was wrong.**
> Appendix B lists "whether SDOT issues on four pipes or two" as unconfirmed, with the note that
> two would halve S-01's gain. Two would not: the tile issues 32 SDOTs per group inside a 24-cycle
> window, and two pipes supply 48 slots in that window. The tile is bound by TOTAL SIMD µops (96,
> exactly filling 4 pipes × 24 cycles), not by SDOT issue, so the answer to the pipe question
> changes nothing here. The question stays open; the risk it was attached to does not exist.
>
> **Why M=1 and M=2 read lower (1.74–1.75×) is structural, not a shortfall.** Below M=4 the tile
> body never executes — those rows take the M%4 remainder path, which is the shipped
> `dotW4A8SplitHalf4Row` one activation row at a time. So that column is really row4-vs-canonical,
> and 41.7 GMAC/s reproduces the row4 kernel's own recorded 42.4–44. The canonical arm sits flat
> at 24.1 GMAC/s across every M, reproducing §2's recorded 24.5–25.0 for `dotW4A8FoldSDOT` to
> within 2% — so this is a same-box A/B against a baseline that re-derived its own published
> figure, not a comparison against another machine's records.
>
> **The parallel ratio being LOWER than the serial one (2.42–2.62 vs 2.88) is S-02 showing
> through**, and is the expected sign: making the kernel 2.9× faster does not make the fork/join
> 2.9× faster, so the fan-out's share of the call grows and the end-to-end ratio compresses. The
> tile therefore makes S-02 worth MORE than the audit costed it, not less.
>
> **Identity is checked at production shapes, not argued.**
> `TestMatmulBTW4A8Row4TileInto_bitIdenticalToCanonical` holds the tile against the canonical M>1
> path over M ∈ {1..9, 12, 13} (every residue mod 4, including the M<4 shapes where only the
> remainder path runs) at both production projections plus a single-group/single-quad shape and an
> odd quad count; `_parallelMatchesSerial` pins width-inertness at 1/2/3/4/8 workers;
> `_zeroActivationRow` pins the `aScale==0` shortcut in both the tiled and the remainder position.
> `TestWeightMatW4A8_MConsistentAcrossRow4Dispatch` — written before the kernel — now genuinely
> spans the M=1-kernel/M>1-tile boundary and passes. `go vet`, the full `linalg` suite, the
> `aikit_checks` contract build and the whole repo's tests are green.
>
> **S-01b IS ALSO DONE, 2026-09-03 — the W8A8 tile, 3.5–3.9×, and the `dotI8Cols8` failure mode
> did not reproduce.** `dotI8Tile4x4` (`linalg/dot_i8_tile_arm64.s`) reduces four activation rows
> against four weight rows into 16 int32 accumulators. It needs no layout change and no opt-in —
> `w8a8Span` now hands the largest 4×4 rectangle of every span to it on arm64, so **every** W8A8
> caller at M≥4 gets it. Bit-identity is free rather than argued: every partial is int32, integer
> addition is exact and associative, and the overflow envelope (|Σ| ≤ n·127²) is `dotI8SDOT`'s
> own, unchanged.
>
> **Measured, serial, one core, median of 3 interleaved passes** — the arms are the pre-tile span
> (`w8a8SpanRows`, literally the code that shipped before) and `w8a8Span`:
>
> | K | N | B size | M=4 | M=8 | M=16 |
> |--:|--:|--:|--:|--:|--:|
> | 1536 | 8960 | 13 MB resident — the prefill cell | 3.67× | 3.68× | 3.79× |
> | 768 | 8192 | 6 MB resident — *the shape `dotI8Cols8` was shipped on* | 3.62× | 3.76× | 3.87× |
> | 1536 | 18944 | 29 MB streamed — *`dotI8Cols8`'s worst case* | 3.74× | 3.74× | 3.83× |
> | 3584 | 18944 | 68 MB streamed — a 7B FFN | 2.08× | 3.47× | 3.66× |
> | 768 | 100000 | 73 MB streamed — aikit's ANN scan | 3.58× | 3.73× | 3.85× |
>
> **The grid straddling the LLC was the point, not decoration.** `w8a8Span`'s own comment records
> that `dotI8Cols8` was measured at ONE cache-resident shape, shipped on it, and lost badly
> wherever B streamed — it was deleted for that. The tile also advances multiple B streams, so the
> same trap was live. It did not spring: the streamed rows are as good as the resident ones. Two
> things differ from `dotI8Cols8` — four streams inside ONE contiguous 4·K block rather than eight
> scattered ones, and four times the arithmetic per weight byte because M≥4 rather than M=1.
>
> **One real cliff, characterised rather than left as an outlier.** At **M=4 only**, the ratio
> falls as K grows: a paired sweep at constant B (68 MB) gives 3.05× at K=3072, 2.05× at 3584,
> 2.01× at 4096, **1.69× at 5120** — while M=8 stays flat at 3.4–3.8× across the same K range.
> The mechanism is that M=4 is the one batch size with NO reuse in the tile's inner loop: each
> 4-weight-row block is read exactly once, so the tile needs ~32 GB/s of weight fetch to keep four
> SDOT pipes fed, against ~19 GB/s at M=8. Past K≈3072 the four large-stride streams stop
> delivering it. Worst cell measured is 1.69×, still above this section's ≥1.5× ship gate, and no
> cell anywhere regresses.
>
> **Appendix B's SDOT-pipe question is now ANSWERED: four.** `TestSDOTIssuePeak`
> (`sdot_peak_arm64.s`) runs 32 independent SDOTs per iteration over 16 accumulators with no
> memory traffic: **3.98 SDOT/cycle, 12.73 G SDOT/s, 203.6 GMAC/s** on one P-core — 99.5% of the
> 4-pipe ceiling at 3.2 GHz, and never below the 3-pipe ceiling even on a busy box. Two pipes was
> the hedge; it is excluded by a factor of two. The probe asserts the CEILING and not a floor, per
> the roofline record's rule, so a folded loop or a wrong clock constant fails it rather than
> being believed. Against that ceiling the W8A8 tile's 163.5 GMAC/s is **80% of pure SDOT issue**,
> which is a good fraction for a kernel that also runs 8 loads per 16 SDOTs.
>
> This supersedes the more cautious note above, which said the W4A8 tile did not settle the pipe
> question. It did not — 32 SDOTs in a 24-cycle window fit in two pipes — but the direct probe
> does.
>
> **BOTH amd64 HALVES ARE DONE, 2026-09-03, measured on an IDLE 3700X** (`nobara-pc`, load
> 0.00–0.07, the quiet box the campaign rules ask for). Serial, one core, median of three
> interleaved paired passes, against the pre-tile span as the do-nothing arm.
>
> **S-01 amd64 — `dotW4A8Tile4RowAVX2` (`dot_w4a8_tile_amd64.s`): 1.65–1.90×, everywhere.**
> The shape is ONE weight row × FOUR activation rows, not arm64's 4×4, because AVX2 has 16 YMM
> registers against NEON's 32 and sixteen live accumulators plus operands does not fit. Blocking
> the activation dimension alone is the part that matters anyway: the 10-instruction nibble unpack
> and the scale broadcast belong to the weight row and were being repeated per activation row.
> 13 instructions per 32 MACs per row against 25.
>
> | K | N | M=4 | M=8 | M=16 |
> |--:|--:|--:|--:|--:|
> | 1536 | 8960 (the prefill cell) | 1.78× | 1.75× | 1.73× |
> | 768 | 8192 | 1.85× | 1.83× | 1.77× |
> | 1536 | 18944 | 1.80× | 1.83× | 1.75× |
> | 3584 | 18944 | 1.73× | 1.70× | 1.65× |
> | 768 | 100000 | 1.90× | 1.85× | 1.79× |
>
> The prefill cell goes 15.6 → 27.6 GMAC/s. Pre-registered expectation was "roughly 2×, and the
> Zen 2 VPMADDWD floor does not bind"; measured 1.65–1.90×, so slightly under the count — the
> opposite sign from arm64, where the count was conservative.
>
> One correctness point that no measurement here could have found: **the tile is excluded on
> AVX-512 VNNI hosts.** `dotW4A8` prefers `dotW4A8FoldAVX512VNNI`, which folds through TWO f32
> accumulators where the AVX2 kernel uses one — a different summation order. An AVX2-based tile
> running at M>1 on a VNNI host while M=1 kept the VNNI kernel would make the result depend on M,
> which is exactly what `TestMatmulBTW4A8_MConsistent` forbids and what speculative verify rests
> on. Neither project box has VNNI, so this is excluded by construction rather than by test.
>
> **S-01b amd64 — and the first shape had to be thrown away.** The instruction-efficient shape is
> 4 activation × 2 weight rows: it fits the 16 YMM registers, costs 0.172 instructions per MAC
> against a 4×1's 0.203, and on cache-resident B it measured 1.34–1.52×. On streamed B it measured
> **0.70× and 0.57×** — a 1.75× REGRESSION:
>
> | K | N | B | 4×2 at M=4 | 4×1 at M=4 |
> |--:|--:|--|--:|--:|
> | 1536 | 8960 | 13 MB resident | 1.41× | 1.25× |
> | 768 | 8192 | 6 MB resident | 1.46× | 1.35× |
> | 1536 | 18944 | 29 MB streamed | **0.70×** | 1.40× |
> | 3584 | 18944 | 68 MB streamed | 1.17× | 1.44× |
> | 768 | 100000 | 73 MB streamed | **0.57×** | 1.32× |
>
> **That is `dotI8Cols8`'s failure, reproduced exactly** — a kernel that wins on cache-resident B
> and loses on streamed B because it interleaves weight streams where the span it replaces walks
> one. It is the third appearance of this trap in this one file's history, and the only reason it
> was caught is that the grid straddles the LLC; measured on the prefill cell alone it would have
> read as a clean 1.41× and shipped. **Note also that arm64's 4×4 tile advances FOUR weight
> streams and shows no such effect** — so "fewer streams" is not the rule; the M1's memory system
> tolerates what the 3700X's does not, and the rule is only ever "measure both regimes".
>
> The shipped kernel is therefore `dotI8Tile4x1AVX2`: four activation rows against ONE weight row,
> B walked exactly as before, **1.17–1.44× at every shape and every M with no regression
> anywhere.** This is the redo `w8a8Span`'s own comment asked for — "a form that does not trade
> the access pattern for it".
>
> **The ceiling was predicted correctly and it is low.** Written before the first run: VPMADDWD
> stays at one per 16 MACs however the loop is blocked, and it issues on a SINGLE Zen 2 port, so
> ~67 GMAC/s at 4.2 GHz binds; `dotI8AVX2` already sits at 48.3–51.3, i.e. 72–76% of it, capping
> any tile near 1.35×. Measured peak is 1.44×, a little above — the tile also removes three of
> every four Go↔asm transitions, which the instruction count does not see. **So S-01b's "2–4×" is
> an arm64 number and does not transfer:** SDOT does 16 MACs on four pipes, VPMADDWD does 16 MACs
> on one.
>
> All gates green on the 3700X: `go vet`, `linalg`, `-tags aikit_checks`, `go test -race ./linalg/`,
> and the whole repo's build/vet/test. Both tiles canary-verified there too — perturbing each by
> one ULP fails `TestMatmulBTW4A8_MConsistent` / `TestMatmulBTW8A8_MConsistent` at every shape,
> including the N-tail and ragged-K ones, so both remainder strips are genuinely exercised.
>
> **SIZED AT LARGE M, AND STEP 2 (M-BLOCKING) IS DEAD — 2026-09-05.** The read-back left open
> whether the activation panel is re-streamed from L2 per quad at large M, which is the only thing
> that would justify blocking the outer loop. `TestW4A8TilePanelTraffic` answers it with per-MAC
> cost against M, which is the discriminator (throughput against M is not — it rises for unrelated
> reasons). M1 Pro, `MatmulBTW4A8Row4TileInto`, best of 3:
>
> | shape | workers | M=8 | M=64 | M=128 | M=512 | panel at M=512 |
> |---|--:|--:|--:|--:|--:|--:|
> | K1536 N8960 | 1 | 14.01 | 13.83 | 13.79 | **13.75** | 0.75 MB |
> | K8960 N1536 | 1 | 13.80 | 13.62 | 13.56 | **13.63** | 4.38 MB |
>
> picoseconds per MAC. **Per-MAC cost does not rise with M — it falls slightly and flattens**, even
> at a 4.38 MB panel on the wide-K shape. So the panel is not being re-streamed at any cost worth
> paying, and M-blocking has nothing to collect. Dead, not deferred.
>
> **Throughput at large M, for the record:** single core 72.7 GMAC/s (K1536 N8960) and 73.4
> (K8960 N1536) at M=512 — consistent with the 69.5 measured at M=4. Six workers reach **394.7 and
> 380.3 GMAC/s, i.e. 5.4× and 5.2× the single core.** Prefill fan-out scales far better than
> decode's ~1.7×, which is S-02 seen from the other side: at M=512 each shard carries enough work
> that the ~92 µs wake stagger is amortised to nothing.
>
> **END-TO-END, AND THE OLD PEER ROW IS NOW SUPERSEDED BY A MEASUREMENT RATHER THAN A PREDICTION.**
> Both arms run on one box in one session against the same Ollama 0.32.5, 1.5B q4_K_M, n=4 fresh
> prefixes per cell, engines interleaved with a server restart, `--backend cpu`:
>
> | K | pre-tile (aikit v1.31.0) | post-tile (v1.34.0) | tile gain | vs Ollama, pre | vs Ollama, post |
> |--:|--:|--:|--:|--:|--:|
> | 512 | 67.6 tok/s | 141.7 | **2.10×** | 3.10× behind | 1.54× behind |
> | 1024 | 67.2 | 137.0 | 2.04× | 2.70× | 1.30× |
> | 2048 | 67.7 | 133.6 | 1.97× | 2.32× | 1.13× |
> | 3900 | 63.3 | 118.8 | 1.88× | 1.81× | **0.91× — AHEAD** |
>
> **The pre-tile arm reproduces the recorded stale row to within 4%** (measured 3.10× at K=512 and
> 1.81× at K=3900, against `cpu-peer-prefill-2026-09-01.md`'s 2.98× and 1.80×), which is what makes
> the two arms comparable: this is a same-box before/after, not a cross-session subtraction.
>
> **Kernel-to-end-to-end compression and what eats it.** The tile measures 2.88× single-core and
> delivers 1.88–2.10× end to end. S-02's fork/join is NOT the eater here — prefill fan-out scales
> 5.4× on six workers, per the panel probe above. What remains is the non-matmul prefill
> remainder, chiefly S-06's serial f32 transcendentals (this document estimates 7–25% of a prefill
> at the 3700X rate, less on the M1) plus attention, norms and the sampler. That is the residue,
> and it is goinfer-side.
>
> Caveat stated rather than buried: the two arms are STACK-level builds — a Sep-1 goinfer binary on
> aikit v1.31.0 against goinfer `3b20f74` on v1.34.0 — so they conflate the tile with whatever else
> changed in goinfer over those four days. The tile is the dominant known change and the kernel
> number is consistent with the end-to-end one, but this is not an aikit-isolated A/B. The box also
> carried ordinary desktop load (loadavg 3.3–3.8, recorded); both engines saw the same load, which
> is why the RATIO is the trustworthy quantity and the absolute tok/s is not.
>
> **S-01 IS CLOSED — goinfer measured the end-to-end prefill cell 2026-09-03 and BOTH pre-registered
> gates pass.** 1.5B q4_k_m, M1 Pro, quiet box, paired and interleaved with ROTATING ARM ORDER,
> n=5, spreads 0.9–5.6%:
>
> | arm | K=512 tok/s | K=3900 tok/s | marginal tok/s |
> |---|--:|--:|--:|
> | int4, new aikit (the change) | **115.4** | **103.5** | **101.9** |
> | int8int8, new aikit (the do-nothing arm) | 104.6 | 90.5 | 88.7 |
> | int4, pre-bump | 65.8 | 61.9 | 61.4 |
> | **new/old** | **1.754×** | **1.672×** | **1.659×** |
> | **int4 / int8int8** | 1.103× | 1.144× | 1.148× |
>
> **Gate 1 — ≥1.5× on the int4 prefill cell: PASSES at 1.66–1.75×.** The referent is before/after
> (int4-new vs int4-old), which is the convention every other decision rule in this document uses
> — S-03 says "before/after" in as many words, and S-02/S-04 quote bare ship ratios with no
> comparator arm. The "do-nothing arm: `--quant int8int8`" line names the comparator for gate 2 and
> for the product decision, not for gate 1. The `park below 1.2×` clause confirms it: on the
> int8int8 reading this rule would park a kernel that is a verified 1.66× improvement AND removed
> the inversion, which is plainly not what it was written to do.
>
> **Gate 2 — int4 ≥ int8int8 at every M: PASSES at 1.103× and 1.144×. THE INVERSION IS GONE.**
> This was S-01's actual thesis. "int8int8 prefill beats int4 by 25–33%" was the observation the
> whole finding was built on; int4 now wins at both depths. The mechanism named in §3 — that at
> M>1 int4's byte saving is served from cache anyway while its unpack ALU cost is paid M times —
> is confirmed by removing the unpack repetition and watching the ordering flip.
>
> **The 2.88× kernel win compresses to 1.66× end-to-end, as predicted and for the predicted
> reasons.** S-02's fork/join is untouched and S-06's serial f32 transcendentals (estimated 7–25%
> of a prefill) are still in the path. The compression is the residue those two findings name, not
> a shortfall in this one.
>
> **Bit-identity held across the bump, checked and not assumed:** goinfer's
> `TestForwardN_matchesSequential` is bit-identical across **19,447,808 logits** at K=128, and
> `TestMoEExpertMajor_bitIdentical` is green over 56 expert-major chunks. That is the cross-repo
> confirmation that `decode == batched prefill == speculative verify` survives a change that
> re-shaped how every M>1 matmul is computed.
>
> **A contamination catch worth propagating into the campaign rules.** goinfer's first pass ran
> S-09.1 concurrently with the prefill benchmark. Both numbers moved on clean re-takes:
> int4/int8int8 from 1.153× to **1.103×** (the contamination had flattered int4 by slowing the
> int8int8 arm) and S-09.1 from 1.35× to **1.70×**. Note the direction — contamination did not
> merely add noise, it biased the RATIO, because the two arms were not equally sensitive to the
> interference. "Quiet box" has to mean quiet of one's own other benchmarks too.
>
> **Still open in S-01:** nothing. A VNNI-host tile for both kernels remains S-08.3's
> file-do-not-build case.
>
> **goinfer needs no code change to get any of this**, which is worth stating because it was not
> true of the row4 layout when that landed. `decoder/weightmat.go:212` already calls
> `RepackInt4Row4`, and `serialize.go:1327` already takes the `WrapInt4Row4` path for kind-4
> `.giw`, so every eligible tensor is row4-resident and `WeightMat.MatmulBTW4A8Into` picks the
> tile up at M>1 on its own. The amd64 tiles hook `w4a8Span`/`w8a8Span` directly and need no
> opt-in at all. A dependency bump is the whole adoption.

> **READ-BACK 2026-09-03 (Cowork): the tile kernels audited against their own claims.** Both
> `dot_w4a8_tile_arm64.s` and `dot_i8_tile_arm64.s` were read instruction by instruction, every
> raw `WORD` recomputed (SDOT 0x4E809400, SCVTF 0x4E21D800, FADDP 0x6E20D400 — all correct), and
> the dispatch (`w4a8Row4TileSpan`, `w8a8TileRect`, the amd64 twins) checked for the seams where a
> tile goes wrong: the M%4 remainder rows, the N%4 / K%16 strips, the `aScale == 0` shortcut, the
> row-pointer setup, the quad-outer/block-inner loop order. **No defect found.** The W4A8 identity
> argument holds as written — per output, `MOVI, SDOT, SDOT, SCVTF, FMLA` in ascending group into
> its own accumulator, the same FADDP tree — and the W8A8 one is exact integer arithmetic. Two
> cosmetic notes only: `nGroups == 0` (K = 0) reaches `&blk[0]` on an empty slice and fails as an
> index panic rather than a shape error; and the "scripted WORD encodings" the header mentions have
> no script in the tree, so the next kernel in this family re-derives them (the ones below were
> checked by assembling for arm64 and running the `==` gates under qemu-user with DotProd on, which
> is the check that matters).
>
> **Where the headroom is, counted against the measurement.** The W4A8 tile issues 96 SIMD µops per
> 32-k group — 4 × {AND, USHR, SUB, SUB} for the unpack and 16 × {MOVI, SDOT, SDOT, SCVTF, FMLA} for
> the outputs — and the measured 69.5 GMAC/s is 21.5 MACs/cycle at 3.23 GHz, i.e. **23.8 cycles per
> group against 24 counted: the kernel is at 99% of a four-pipe issue ceiling, and `MOVI #0` is NOT
> a rename-time zero idiom** (if it were, 80 µops would have measured 25.6 MACs/cycle). So the only
> lever left inside the kernel is µops, and S-05 is exactly that lever, larger here than in the M=1
> kernel it was written for: fold the −8 centering into the SDOT accumulator's initial value. Per
> (activation row, group) precompute `corr = −8·laneSum(act)` once — two SDOTs of the activation
> halves against a vector of −8, K/32 × 16 bytes per row, shared by every quad — and in the tile
> replace `MOVI #0` with a load of `corr` and drop both `SUB` from the unpack. The int32 that
> reaches `SCVTF` is bit-for-bit today's (`Σ(nib−8)·act = Σ nib·act − 8·Σ act`, exact), so the f32
> fold and the identity gates are untouched. Count: **96 → 72 SIMD µops per group, 28 loads at
> ~1.6/cycle against 3 ports — 1.33× on the tile (69.5 → ~92 GMAC/s counted)**, and the same change
> takes the M=1 kernel from 9 to 6 µops per row-group. Pre-registered rule: `TestW4A8TileVsCanonicalAB`'s
> shape, ship at ≥ 1.2× single-core with every `==` gate green, park below 1.1×. Second, smaller:
> the four `LD1R` + `ADD` scale broadcasts per group can be one `LD1 .4S` + `FMLA` by element
> (scales are already interleaved `[s_r0 s_r1 s_r2 s_r3]`), off the load/ALU side — noise at four
> pipes, worth taking while the loop is open. Third, for verify widths: M=7 runs 4 rows through the
> tile and 3 through the single-activation kernel at 0.28 µops/MAC against the tile's 0.19; a 3×4
> remainder tile (12 accumulators, fits) is ~1.1× on the M=7 matmul term and no more — file, take
> only if verify is measured compute-bound at that width. Last, large-M: the activation panel is
> re-streamed per quad from L2 (4.6 MB at K=8960, M=512); an M-block outer loop is the textbook
> GEMM move and probably explains most of the 2.88× → 2.42–2.62× gap on the parallel dispatch, but
> measure the panel traffic before building it.
>
> **W8A8 tile:** 8 loads + 16 SDOT per 16-byte chunk, measured 163.5 of a 203.6 GMAC/s SDOT-issue
> ceiling (80%). Nothing structural to change — the accumulators clear latency, the loads clear the
> ports. A two-chunk unroll (V24–V31 are free in the loop) halves the loop and address-increment
> overhead per SDOT and is the one cheap experiment; expect ≤ 1.1× and do not chase further.

- **Where:** `quant.go:585-597` (`w4a8Span`), `quant.go:280-330` (`w8a8Span`);
  `weightmat_splithalf_amd64.go:67-73` and `weightmat_row4_arm64.go:48-54` route every M≠1 call
  to the canonical span; goinfer `decoder/weightmat.go` ("int4 weights run the W4A8 kernel at
  EVERY M").
- **Code:** `for j := range N { prow := w4[j*bpr:…]; for i := range M { dst[i,j] = dotW4A8(aq_i, prow, srow)·aScale_i } }`
  — the 16-byte weight load, the unpack (6 SIMD µops arm64 / 9 instructions amd64), the SDOT
  zero + scale fold (3 µops) and the ~18 ns Go↔asm transition execute once per activation row.
- **Mechanism, measured in-tree:** the unpack alone isolates at **1.57×** on amd64
  (goinfer `docs/measurements/aikit-w4a8-opsperbyte.md`), and on the Mac **int8int8 prefill is
  25–33% faster than int4** despite twice the bytes (goinfer
  `docs/measurements/cpu-peer-prefill-2026-09-01.md`) — the byte saving is served from cache at
  M>1 anyway while the ALU cost stays. At K=512 goinfer prefills at 68 tok/s ≈ 17 GMAC/s per
  core, ~17% of SDOT peak; Ollama's 204 tok/s ≈ 50%.
- **Proposal (new mechanism, not in any non-goals list):** a 4×4 tile kernel on the **existing
  row4 layout** — `dotW4A8Row4Tile4x4(act0..3, packed4, scales4, dst[16], nGroups)`: per
  32-k group unpack four weight rows once (4 LD1 + 16 SIMD µops), load four activation chunks,
  then 16 × (MOVI, SDOT, SDOT, SCVTF, FMLA) into 16 persistent f32 accumulators (register
  budget 31 of 32). SIMD µops per group: 96 for 512 MACs = **0.19/MAC vs 0.34 today**, and
  issue-bound instead of latency-bound: **21 MACs/cycle vs 8, counted 2.7×** (≈2× if SDOT issues
  on only two pipes). The row4 bytes already exist for every eligible tensor; M%4 remainder rows
  take the 4-row kernel one row at a time. amd64 twin: (a) unpack each weight row once per span
  into a K-byte int8 scratch (L1-resident) and run the unpack-free body per activation row, or
  (b) a 1-weight-row × 4-activation-row block: (15+36)/4 = 12.75 instructions per 32 MACs vs
  25. W8A8 (S-01b): `dotI8SDOT` is latency-bound at 1 SDOT/cycle (4 accumulators × 4-cycle
  latency — the measured 49.2 GMAC/s matches to 4%); an M-tile with 16 int32 accumulators is
  SDOT-issue-bound at 2–4× that, and even 8 accumulators in the single-row kernel is 1.5×.
- **Bit-identity:** preserved by construction on both — each output's per-group int32 is the
  same SDOT lane mapping, its f32 fold is one FMLA per group in ascending g into its own
  accumulator, the FADDP tree and `×aScale` are unchanged; W8A8 is exact integer arithmetic in
  any arrangement. Prove with `==` against `MatmulBTW4A8Into` across M ∈ {1..9}
  (`TestMatmulBTW4A8Row4Into_bitIdenticalToMatmulBTW4A8Into` is the shape) plus goinfer's
  `TestForwardN_matchesSequential` and `TestMoEExpertMajor_bitIdentical`. **There is no
  M-invariance gate for W4A8 today** (`TestMatmulBT_MConsistent` covers f32 only) — write it
  first; it is the thing that makes the rest safe to ship.
- **Decision rule, pre-registered:** `BenchmarkQ4vsQ8 W4A8/K1536_N8960` at M ∈ {4, 16, 64, 512}
  cold, both boxes, then `scripts/bench_peer_prefill.py --backend cpu` at K=512/3900 paired and
  interleaved. Ship at ≥1.5× on the int4 prefill cell and int4 ≥ int8int8 at every M; park
  below 1.2×; between → second mechanism. Do-nothing arm: `--quant int8int8`, which is the
  configuration the measurement says to use today.
- **Confidence:** high on the mechanism and the counts; medium on 2.7× (the roofline record
  notes instruction-count arguments overpromised here three times).

### S-02 · Perf-major (decode) — six workers deliver two cores' worth; the fan-out is the limiter on both boxes, not the kernel

- **Where:** `linalg.go:208-234` (`parallelSpawnCols`: one goroutine per equal static shard per
  call, shards rounded to 8 columns, `wg.Wait`); `workspace.go:62-64`; goinfer
  `decoder/weightmat.go:115` (`int4ParThreshold = 1<<20`), `decoder/attention.go:79-81` and
  `decoder/mlp.go:385-386` (q, k, v, gate, up issued as separate W4A8 calls).
- **Numbers (arm64):** one worker 25.9–26.2 GB/s (92% of the per-core port ceiling); six
  workers 59.6–60.6 isolated / **44.9 in real decode**; eight 58–62. Per-core throughput at six
  is 26–35% of the same core alone and ~50% of the 121 GB/s STREAM ceiling — neither the
  instruction count nor DRAM explains it. **amd64:** the cold kernel does 9.9–10.6 GB/s per
  core; the whole 8-core box streams ~19–21 GB/s (1.5B at 18.3 tok/s × 1.05 GB), i.e. two
  cores' worth, against Ollama's 27.6; `queue-performance.md:336-339` compared that
  whole-machine 11.7 GB/s figure against a *single-core* 10.65 and concluded "almost no
  composition overhead" — one core against eight. The +2.1% end-to-end from a +12% kernel
  (split-half) is the direct evidence the kernel is ≤ ~20% of the amd64 token at 8 cores.
- **Mechanism (two candidates, same remedies):** (a) the sequential goroutine wake chain —
  `newproc → wakep` wakes one spinning M at a time; a ~20 µs stagger per successive worker puts
  the 8th shard's start at ~140 µs and its finish at ~180 µs, which reproduces the ~190 µs a
  gate/up call takes at 45 GB/s while one P-core finishes its 1/8 shard in ~41 µs; (b) static
  equal shards across 6P+2E (Mac) or 16 SMT siblings sharing 8 FP units and two CCX L3s (Zen 2)
  — the slowest shard sets the barrier. Both are consistent with the June profile ("71%
  fork/join, 1.3× scaling on 8 cores", goinfer `docs/completed/perf-campaign.md:225-239`) and
  with `GOINFER_PAR_WIDTH=4` measuring +4.4% on the 1.5B. The spin-then-park pool is a
  **measured negative** (`perf-dead-ends.md` §8.1; goinfer Phase 3b: pool 64.1 vs spawn 67.6
  tok/s) and is not re-proposed; that negative does not cover shard skew, which the same entry
  leaves open ("dynamic work-stealing over column chunks").
> **MEASURED 2026-09-03 — step 0's last item is closed, hypothesis (a) wins, and BOTH remedies
> were compared before either was built.** `TestW4A8ForkJoinShardTiming` and
> `TestW4A8WorkPerBarrier` (`w4a8_forkjoin_probe_arm64_test.go`), M1 Pro, row4 kernel, K=1536
> N=8960, 8 distinct 8.2 MB matrices so nothing is cache-resident.
>
> **The mechanism is goroutine-wake stagger, not P/E-core shard skew.** Per-shard timestamps
> across one fan-out:
>
> | workers | start spread (max−min) | median shard duration | duration spread |
> |--:|--:|--:|--:|
> | 6 | **92.6 µs** | 57.7 µs | **1.07×** |
> | 8 | **98.3 µs** | 48.5 µs | 1.39× |
>
> The spread in START times is 1.6–2× a shard's entire duration while the durations themselves are
> nearly uniform. That is signature (a). Signature (b) would be the opposite — bunched starts,
> bimodal durations — and it is not what the machine shows. So the last worker spends more time
> waiting to be woken than it spends working, which is why the six-worker aggregate has been
> sitting at ~26–35% of six cores.
>
> **Work per barrier, which is a direct simulation of `MatmulBTW4A8Batch` rather than an analogy:**
>
> | matrices per fork/join | 1 worker | 6 workers | 8 workers |
> |--:|--:|--:|--:|
> | 1 (today) | 24.9 | 58.8 | 58.2 |
> | 2 (gate‖up) | 25.1 | 68.8 | 71.5 |
> | 3 (q‖k‖v) | 25.1 | 74.0 | 74.3 |
> | 8 | 25.1 | **87.4** | 75.7 |
>
> GB/s. **The single-worker column is the control that makes this readable: it is FLAT to within
> 1%** across the whole sweep, because with one worker there is no stagger to amortize. Without
> that row the six-worker climb could have been cache residency; with it, the effect is the
> barrier. Against this section's own pre-registered reading — "if GB/s climbs toward 100+ the
> barrier is the limiter, if it stays ~60 the memory system is" — it climbs.
>
> **Remedy (1), dynamic chunking, does NOT subsume remedy (2).** Measured in the same harness,
> workers taking quad blocks from an atomic counter, at ONE matrix per barrier (i.e. no API change
> and no batched caller): chunk=8 → 59.6 GB/s (1.013×), **chunk=32 → 66.8 (1.135×)**, chunk=128 →
> 64.4 (1.096×). So it captures roughly half of what three matrices per barrier gives (1.258×) and
> a third of what eight gives. Worth having — it is free and needs no caller cooperation — but it
> is complementary to the batch form, not a substitute for it. That question is why this section
> says measure first, and the answer would not have been guessable.
>
> **`MatmulBTW4A8Batch` therefore built, and measured against what actually ships** — N separate
> `MatmulBTW4A8Row4Into` calls, NOT the canonical kernel, because comparing against canonical would
> credit the batch with a layout win it did not earn:
>
> | workers | fusion | per-op (row4) | batch | |
> |--:|---|--:|--:|--:|
> | 6 | q‖k‖v | 56.1 µs | 50.2 µs | **1.118×** |
> | 6 | gate‖up | 287.3 µs | 237.7 µs | **1.209×** |
> | 8 | q‖k‖v | 56.1 µs | 49.3 µs | 1.137× |
> | 8 | gate‖up | 264.3 µs | 225.3 µs | 1.173× |
>
> **A design consequence worth recording, because the obvious API would have been a regression.**
> The natural signature — mirror `W8A8Op`, carry canonical `W4`/`Scales` only — would have taken
> goinfer's decode OFF the row4 kernel (~42 GB/s) and onto canonical (~24), a 1.75× kernel loss
> against a ~1.2× fan-out gain. Net worse, for exactly the caller the batch exists to serve. So
> `W4A8Op` carries optional `Row4`/`Row4Scales` and the span keeps the fast layout; unaligned
> column edges fall to canonical, which is a dispatch choice and never a numeric one because the
> two layouts are bit-identical.
>
> **What this does NOT establish: the end-to-end decode cell**, which is this section's actual
> ship gate (≥1.15×, park below 1.05×) and is goinfer's to measure. Arithmetically the per-layer
> saving is ~6 µs on q‖k‖v plus ~49 µs on gate‖up, so ~55 µs × 28 layers ≈ 1.5 ms of a ~25 ms
> token ≈ **1.06×** — which lands BETWEEN park and ship. That projection is a projection; the
> paired `bench_peer` cell decides, and if it lands short the honest move is dynamic chunking on
> top (independent, free) rather than reading the gate loosely.

- **Measure first (cheap, decisive):** per-shard `(start, end)` timestamps in
  `TestW4A8Item3ParallelAggregate` (`w4a8_item3_parallel_arm64_test.go:99-112`) — a stagger
  pattern is (a), a bimodal duration is (b); then the same harness with 8 matrices per fork/join
  (8× the work per barrier): if GB/s climbs toward 100+ the barrier is the limiter, if it stays
  ~60 the memory system is. On amd64, a 1..16-thread read-bandwidth probe of the 3700X first —
  nothing can be concluded there until a STREAM number exists.
- **Remedies (both output-partition-only, width-inert by the existing
  `TestParallelWidth_bitIdentical` contract):** (1) dynamic chunking — workers take quad blocks
  from an atomic counter (32 quads ≈ 96 KB at K=1536); under (a) the arithmetic gives ~110 µs
  instead of ~190 per gate/up call (1.7×); (2) `MatmulBTW4A8Batch` mirroring the existing
  `MatmulBTW8A8Batch` (`quant.go:351`): q‖k‖v in one fork/join and one quantisation, gate‖up in
  one → 5 → 3 barriers and 7 → 4 quantisations per layer; the W8A8 batch form measured 60 →
  66–68 tok/s when it shipped (`perf-campaign.md:213-216`). goinfer's `GOMAXPROCS` default of
  16 on the 3700X puts SMT siblings on one FP unit; `GOINFER_PAR_WIDTH=8` is a one-run check.
- **Decision rule:** the 1.5B decode cell (`bench_peer`, paired) on each box; ship at ≥1.15×,
  park below 1.05×.
- **Confidence:** high that the kernel is not the limiter (three independent measurements);
  medium on which candidate dominates — hence measure first.

### S-03 · Perf-minor (2–6% of a token) — the activation quantiser is scalar, serial, and run seven times per layer where four would do

> **DONE 2026-09-03 (Cowork) — the NEON quantiser, bit-identical, gate-checked.** `quant_act_arm64.s`:
> `FABS`/`FMAXNM` into four accumulators + `FMAXNMV` for the abs/max, then `FMUL` by the broadcast
> inv, `FCVTAS` (ties-away, math.Round's rule on the exact f64 widening of the f32 product),
> `SMIN`/`SMAX` ±127, `SQXTN`/`SQXTN2` narrow, one 16-byte store — 22 SIMD µops per 16 elements.
> `quantizeRowInt8Core` now dispatches through `maxAbsF32` / `quantizeRowScaled` (arm64 NEON, scalar
> elsewhere) with the scale arithmetic unchanged between them; the original body is kept verbatim as
> `quantizeRowInt8CoreScalar`, the oracle. `TestQuantizeRowInt8_bitIdenticalToScalar` (24 lengths ×
> 6 distributions, scale bits and every code) and `TestQuantizeRowInt8_corners` (NaN in body/tail/all,
> ±Inf, −0.0, denormals, exact .5 ties, saturating magnitudes, all-zero) pass on arm64 under
> qemu-user and natively on amd64. Not measured yet: `BenchmarkQuantizeRowInt8` (dispatched vs
> scalar, K=1536/8960) is in the tree for the Mac. The redundant three quantisations per layer are
> still S-02's batch entry. amd64 stays scalar (the `VMULPS` + `x+copysign(0.5,x)` form is open).

- **Where:** `quant.go:43-71` (`quantizeRowInt8Core`: an abs/max pass, then
  `int8(math.Round(float64(v*inv)))` with clamps), called on the calling goroutine before every
  fan-out from `MatmulBTW4A8Into:568-570`, `MatmulBTW4A8Row4Into` (`matmul_w4a8_row4_arm64.go:109`),
  `MatmulBTW8A8Batch:366-368`; no `.s`, no `simd` build on either arch.
- **Cost:** 28 × 7 quantisations = **509k elements per 1.5B token** (three of the seven
  re-quantise the same activation — q/k/v and gate/up have no W4A8 batch form); at an estimated
  4–10 cycles per element on the M1 that is 0.6–1.6 ms of a ~25 ms token; the June W8A8 profile
  put it at ~9% of CPU on the 0.5B (`perf-campaign.md:231`); on the 3700X `math.Round` alone
  measured 1.56–3.07 ns per element.
- **Fix, bit-identical:** NEON `FABS`+`FMAXNM` (4-wide, `FMAXV` at the end — max is exact),
  then `FMUL.4S` (the same f32 product as today), `FCVTAS.4S` (round-to-nearest-away on the f32
  value, which equals `math.Round` on its exact f64 widening), `SMIN/SMAX` clamp, `SQXTN`
  narrow — ≈0.3 cycles per element, ~50 µs per token. amd64: `VMULPS` + the `x+copysign(0.5,x)`
  truncate form that is exact here. Pin the edge cases the scalar defines (−0.0, NaN → 0 vs
  `int8(NaN)`, ±Inf) with the scalar as oracle. Then the W4A8 batch entry (S-02) removes the
  redundant three.
- **Decision rule:** `BenchmarkQuantizeRowInt8` per element before/after, and the 1.5B decode
  cell; the kernel is small enough that anything under 1.02× end-to-end is fine to keep for the
  fork/join count it also removes.
- **Confidence:** high on mechanism and identity; medium on the 2–6% (whether gc emitted
  `FCSEL` or branches for the abs/max is unverified).

### S-04 · Perf-major at long context — NEON f64 lane-per-output ports of the acc64 attention kernels, bit-identical

> **STEP 1 DONE 2026-09-03 — the pure-Go AV intermediate, and it beat its own prediction.**
> `MatmulAVAcc64` now blocks its output dims into 16 NAMED f64 accumulators with keys inner,
> so the accumulator lives in registers instead of the `acc` slice. No assembly.
>
> **Measured on `apple-m1pro`, three interleaved A/B passes (load ~4, hd=128, M=1):**
>
> | depth | old (memory acc) | new (16 registers) | |
> |--:|--:|--:|--:|
> | 130 | 9,210 ns | 3,848 ns | **2.39×** |
> | 2048 | 139,952 ns | 59,567 ns | **2.35×** |
> | 8192 | 569,935 ns | 241,997 ns | **2.35×** |
>
> Against this section's ~2× estimate for the Go intermediate and its ≥1.3× ship gate. The
> measured OLD figures reproduce the baselines quoted below to within 3% (9,210 vs 8,929;
> 569,935 vs 565,038), so this is a same-box A/B rather than a comparison against another
> machine's records — the failure mode the campaign rules exist to prevent.
>
> **Identity is checked, not argued:** `TestMatmulAVAcc64_exactMatchesStrided` compares against
> the independent strided kernel and passes. The mechanism is confirmed deterministically too —
> `go build -gcflags=-S` shows `FMADDD` 1 → 17 and `FCVTSD` 2 → 19 with accumulator spill stores
> **unchanged at 2**, i.e. sixteen accumulators resident in registers, none spilling.
>
> `BenchmarkMatmulAVAcc64` was added at the same time: this kernel had no local bench, which is
> why the figures below had to be quoted from goinfer's records at all.
>
> **Confirmed on amd64 too (idle 3700X, three interleaved passes):** depth130 10,489 → 7,710 ns
> (**1.36×**), depth2048 166,349 → 123,275 (**1.35×**), depth8192 715,851 → 491,217 (**1.46×**).
> The win transfers — it is pure Go — but is roughly 0.6× of arm64's, so the register pressure it
> relieves matters more on NEON. Both arches clear the ≥1.3× ship gate.
>
> **STEP 2 DONE 2026-09-03 (Cowork) — both NEON ports, bit-identical, gate-checked.**
> `attn_acc64_arm64.s`: `avAcc64NEON32` (32 dims per call, 16 f64 lane-pair accumulators, the score
> widened once per key with `FCVTSD`, the V row widened with `FCVTL`/`FCVTL2`, `FMLA` by element,
> `FCVTN` narrow at the end — ~1.1 SIMD µops/MAC against the Go block's ~3 instructions/MAC) and
> `qkAcc64NEON16` (16 keys per block, lane = key, `ZIP1`/`ZIP2` to pair keys per dim, four
> `FMLA`-by-element per d-quad in ascending d; ~1.3 FP µops/MAC against the 8-chain loop's ~2.1 and
> its 8-way latency bound). `MatmulAVAcc64` takes whole 32-dim blocks through the kernel and the
> Go 16-block/tail code finishes hd%32; `MatmulQKAcc64` takes N/16 blocks and the 8-chain loop
> finishes N%16 (K%4≠0 stays Go). Identity checked three ways: the existing
> `TestMatmulAVAcc64_exactMatchesStrided` / `TestMatmulQKAcc64_exactMatchesStrided` against the
> independent strided kernel (both green, including nKeys=8192); `TestAVAcc64NEON32_matchesGo` /
> `TestQKAcc64NEON16_matchesGo` calling the kernels directly on adversarial data (60 binades,
> random signs, raw-bit compare) over every residue; and `TestAttnAcc64NEON_mutationDetected`, which
> proves the oracle can fail. Counted: AV ~2.8× over the step-1 Go blocks (issue-bound at ~3.8
> MACs/cycle: 2 loads, 1 scalar widen, 16 converts, 16 FMLAs per key per 32 dims), QK ~1.7–2× over
> the 8-chain loop; both unmeasured until the Mac runs `BenchmarkMatmulAVAcc64` and the QK bench,
> depth 130/2048/8192, order-alternated, per this section's rule.
>
> **Still open in S-04:** the GQA-group V-widen sharing (a multi-query-row AV kernel: at M=2 with
> 24-dim blocks the register budget closes — 2×12 accumulators + 6 V + 2 score — and halves the
> convert cost; needs the goinfer caller to hand a KV group's query rows to one call), and the
> amd64 ports. The block size 16 is a guess the audit named, not a measured
> optimum — hd/16 = 8 passes over V, where the assembly form's 24 registers would give 3.

- **Where:** `matmul_qk_acc64.go:21-67` (8 keys as 8 named f64 accumulators; per d-step the
  arm64 compiler emits per key `LDR s` + `FCVTSD` + `FMADDD`), `matmul_av_acc64.go:33-52`
  (`acc[d] += w * float64(vrow[d])` with `acc` in memory: two loads, a convert, an FMADD, a
  store and index work per MAC). goinfer `decoder/forwardn.go:791`, `:893` — the only live
  decode-attention path.
- **Measured (aikit benches quoted in goinfer `task-attention-decode-cost.md` §A1):** QK 3,463
  ns at depth 130 / 219,500 at 8192 (≈4.8 GMAC/s, depth-independent); AV 8,929 / 565,038
  (≈1.86 GMAC/s). At depth 8192 attention is 85.2 ms of a ~110 ms token even with the 6-way
  head fan-out.
- **Why a port is bit-identical, and stronger than the code claims:** both operands are f32
  widened to f64, so every product is *exact* in f64 (24+24 ≤ 53 bits). A lane running
  `RN(acc + q_d·k_d)` in ascending d is therefore identical to the scalar `FMADDD` chain —
  and identical to an unfused `FMUL.2D`+`FADD.2D`, and to amd64's `MULSD`+`ADDSD`: these two
  kernels are the one place in the substrate that is already cross-arch identical. The existing
  `==` gates (`matmul_qk_acc64_test.go` over every nKeys residue mod 8;
  `TestMatmulAVAcc64_exactMatchesStrided`) are the acceptance test as they stand.
- **QK port:** lane j = key j. Without a layout change: per 2 keys × 4 d, two `LDR Q`, `ZIP1/ZIP2`
  (pair keys per d), `FCVTL/FCVTL2` ×4, `FMLA.2D` ×4 by element of a pre-widened q → 40 SIMD
  µops per 32 MACs vs 68 today: **≈1.7×**; with keys stored contiguous per d (a K-layout the
  cache could write at append time) it is `LDR D` + `FCVTL` + `FMLA.2D` per 2 MACs: **≈2.1×**.
- **AV port:** register-resident accumulators — 24 registers hold 48 dims, three passes for
  hd=128; per (key, 2 dims) `LDR D` + `FCVTL` + `FMLA.2D` by the widened score ≈1.05 µops/MAC
  vs ~7 instructions/MAC today. Bound ~6× when L1-resident; at depth 8192 the 4 MB V slice is
  re-read three times from L2/SLC (~120 µs) plus ~82 µs of compute → **≈3× (565 → ~180 µs per
  head-call)**. A pure-Go intermediate (16 named f64 accumulators per dim block, keys inner)
  gets ~2× with no assembly and the same identity argument — worth landing first because it
  needs no `.s` and settles the mechanism. Sharing the V widen across the six query heads of a
  GQA group (one worker owns a KV group) halves AV's convert+load cost again but re-shapes the
  head fan-out; second step.
- **Bound overall:** per head-call at depth 8192 ≈ 785 → ≈300 µs; attention 85 → ~33 ms; the
  token ~110 → ~58 ms (**≈1.9× at 8k**, ~1.1× at depth 128). DRAM is not the limiter (K+V read
  per token at 8k ≈ 470 MB ≈ 4–8 ms).
- **Decision rule:** the isolated QK/AV benches at depth 130/2048/8192, order-alternated
  best-of-3 as A1 did, then `bench_peer` at depth 2048 and 8192 paired (the harness re-prefills
  the prompt per completion — use the fixed-prefix protocol the campaign switched to). Ship at
  ≥1.5× on the depth-8192 attention component with every `==` gate green; the Go intermediate
  ships on its own at ≥1.3×.
- **Confidence:** high on identity; medium-high on the bounds (Firestorm port counts are from
  the same external table as the rest of §2).

### S-05 · Perf-minor (decode kernel, arm64) — fold the −8 centering into the SDOT accumulator's initial value: 9 → 7 SIMD µops per row-group, bit-identical

> **NOT STARTED, 2026-09-05, and deliberately so — the pre-registered decision rule said stop.**
> The CPU-prefill-remainder brief (goinfer `task-prefill-gap.md` §4 L4) gated S-05 in the tile on a
> peer measurement: *marginal ≤1.15× behind Ollama → CLOSE, no kernel work; ≥1.5× → build it.*
>
> Measured on the M1 Pro, 1.5B q4_K_M, both engines interleaved with a server restart, n=4 fresh
> prefixes per cell: the fit is `LINEAR_FIT_INVALID` on CPU exactly as the brief predicted (TTFT is
> superlinear in K, fitted overhead reads negative), so the local slopes are the number.
>
> | segment | goinfer | Ollama |
> |---|--:|--:|
> | 519 → 1031 | 132.7 tok/s | 148.1 |
> | 1031 → 2067 | 130.3 | 131.9 |
> | 2067 → 3919 | **105.8** | **82.3** |
>
> **Whole-curve marginal ratio 0.86× — goinfer is AHEAD of Ollama on the marginal, not behind.**
> That is not "within 1.15×", it is past parity, so the rule fires at its stopping end and S-05
> stays unbuilt. Before the tile the same measurement read 1.69× behind, which is what the gate was
> written to catch.
>
> **What is still true, for whoever picks this up later.** Nothing here refutes the mechanism: the
> counted 96 → 72 SIMD µops per group (1.33× on the tile) and 9 → 6 on the M=1 kernel stand
> unmeasured, the int32 identity argument stands, and the free scale-broadcast saving
> (four `LD1R + ADD` → one `LD1 .4S` + FMLA by element) is still sitting there. What changed is
> that there is no longer a prefill deficit to spend them on. If a decode-side need appears —
> S-02's remedies land and the kernel becomes the limiter again, or a slower arm64 part shows up —
> this is the first lever to reach for, and it is bit-identical so it carries no numerics risk.
>
> The one-ULP canary discipline and the raw-WORD cross-check the brief specifies were not exercised,
> because no assembly was written.

- **Where:** `dot_w4a8_arm64.s:462-473` (per row per group: `VSUB`, `VSUB`, `VMOVI $0`).
- **Mechanism:** per lane l the two SDOTs produce Σ over k ∈ {4l..4l+3} ∪ {16+4l..} of
  `(nib−8)·act`. Exactly, in integers (|values| ≪ 2³¹): `Σ(nib−8)·act = Σ nib·act − 8·Σ_lane act`.
  Precompute per token, per group, one 4-lane int32 vector `corr_g = −8·laneSum_g` (two SDOTs
  against a vector of 8s over the activation, K/32 groups), and replace `VMOVI $0, V16` with a
  load/`ORR` of `corr_g` (one µop either way). The two `VSUB.16B` per row vanish and the int32
  that reaches `SCVTF` is bit-for-bit what it is today, so the f32 fold is unchanged. Cost: one
  extra 16-byte load per group shared by four rows.
- **This is not the rejected v2 shape:** v2 applied `−8·Σ scale·sumAct` as a separate *f32* sum
  after the fold (not bit-identical, an extra pass, measured 0.972×); this applies the
  correction inside the exact int32 domain at zero extra µops. It is the "fold the correction
  into item 3's layout" retry the campaign doc leaves open. ggml's repack-time `nib ^ 0x88` +
  `SHL/SSHR` sign-extension is the alternative (3 µops for both halves instead of 4) but changes
  the row4 byte layout — a kind-4 version bump; the in-RAM GGUF repack could adopt it freely.
- **Expected:** up to 1.29× on the single-core issue-bound kernel; ~0 at six workers until S-02
  lands (the kernel is not the limiter there). A second, load-side saving in the same loop: the
  four rows' scales are adjacent (`[s_r0 s_r1 s_r2 s_r3]` per group) and could be one `LD1 .4S`
  with `FMLA` by element instead of four `LD1R` + four `ADD` — six fewer instructions per 51,
  off the scalar/load side rather than the SIMD pipes.
- **Decision rule:** the hot/cold single-call harness (`TestW4A8Row4ColdFix_warmIntact` shape)
  first; `==` vs `MatmulBTW4A8Into` as the gate; ship on any single-core win once S-02 makes it
  visible end-to-end, since it is free.
- **Confidence:** high on identity and the µop count; medium on 1.29× (`MOVI` may already be a
  zero idiom at rename — then the saving is still two µops).

### S-06 · Perf-minor decode / Perf-major prefill — the transcendentals are scalar f64 `math.Exp` on one goroutine, while `linalg`'s f32 SIMD family sits unused

- **Where:** goinfer `decoder/rmsnorm.go:79-118` (`silu`, `geluErf`, `geluTanh`, all f64
  "for parity"), the loops at `decoder/mlp.go:400-402` and `decoder/forwardn.go:585-587` (K×inter
  elements in one plain loop on the calling goroutine), every softmax (`decoder/attention.go:236/299`,
  `decoder/forwardn.go:829/861`, `decoder/fusedattn.go:126`); aikit `exp.go` / `exp_simd.go` (v1.15.0 scalar
  minimax family; v1.23.0/1.24.0 `simd` vectorisation behind `GOEXPERIMENT=simd`, ≤1 ULP vs
  `math.Exp`, softmax 5.2 → 2.1 ns/elem and SiLU 3.86 → 1.34 on the M1) — **called from nowhere
  in goinfer**.
- **Cost:** 251k `math.Exp` per decode token (SiLU over 8960 × 28) — ≤1 ms on the M1 (bounded by
  the 4.2 ms "everything else" bucket in the Gate-0 stub), ~3.8 ms of a 66 ms token on the
  3700X (15 ns/elem measured there); **128M per 512-token prefill and 980M at K=3900**, serial:
  ≤0.5 s of a 7.5 s prefill on the M1, ~1.9 s at the 3700X rate — 7–25% of a prefill that is
  2.98× behind. Softmax is ~0.3% at depth 128 and ~4% at 8k (already inside the head-parallel
  worker).
- **Two steps, the first free:** (1) split the elementwise loops across the existing worker pool
  — elementwise has no reduction, so it is bit-identical and costs nothing in goldens; the
  prefill share alone justifies it. (2) f32 `SiLUInto`/`SoftmaxRowScaledInto` — shifts every
  logit ≤4 ULP; `decode == prefill == verify` survives if the same function is used in both
  `mlp.go` and `forwardn.go`; arch-keyed goldens regenerate and the HF cosine gate re-runs. Step
  2 is the parity-class decision the August audit already named (its G4/G5); nothing about it
  changed except that the kernel now exists and is measured.
- **Decision rule:** stub the activation loop at K=512/3900 first (the share has never been
  measured directly — the Gate-0 stub bucketed it under "everything else"); ship step 1 on any
  win; step 2 needs the parity re-baseline and is a product decision (the same one as
  `--cpu-fast-attention`).
- **Confidence:** high on mechanism; medium on magnitude until the stub runs.

### S-07 · Perf-major for weight-only `int8` mode — `dotNEON4` has one FMLA chain: 1 MAC per cycle

> **DONE 2026-09-03 (Cowork) — the eight-column span, bit-identical, gate-checked.**
> `q8span_arm64.go`: `q8Span` widens eight weight rows into an 8×K scratch (`dequantRowInt8`, as
> before) and runs each activation row through `dot8ColsInto` → `dotNEON8x4`, whose per-row
> accumulator receives exactly `dotNEON4`'s lane sequence and whose fold is the same left-to-right
> `s0+s1+s2+s3`; the K%4 tail is the same `s += a[k]*b[k]` after the fold; the row scale multiplies
> last. Fewer than eight remaining columns, and K<4, take `q8SpanColumn`, which is the old body and
> the definition. The parallel path's per-worker `make([]float32, K)` is gone on every arch —
> `q8SpanScratchPool` (a `sync.Pool`) hands out the scratch, and
> `TestMatmulBTQ8Into_parallelScratchPooled` bounds allocations per call at workers+3.
> `TestQ8Span8Cols_bitIdenticalToColumnForm` (K%4 and K%16 tails, N<8, N%8, split column ranges,
> serial and forced-parallel `MatmulBTQ8Into`) and the older `TestQ8Span_bitIdenticalToScalarWiden`
> both pass on arm64 under qemu-user; amd64 keeps the column form (`dot8ColsInto` reduces
> differently there, so the identity argument does not transfer). `BenchmarkMatmulBTQ8_span` is in
> the tree for the Mac.

- **Where:** `dot_arm64.s:36-41`, `quant.go:130-144` (`q8Span`: per weight row `dequantRowInt8`
  then `dotF32` per activation row).
- **Mechanism:** a single accumulator at ~4-cycle FMLA latency → 3.2 GMAC/s per core = 3.2 GB/s
  of int8 weights; that is the arithmetic behind the LM head's 11–13 GB/s at 6–8 workers before
  it moved to W8A8 (`MatmulBTQ8` vs `MatmulBTQ8Into` identical at 12.6 GB/s — allocation was
  never the bottleneck, the chain was). `--quant int8` users still run every projection through
  it.
- **Fix, bit-identical to today's `MatmulBTQ8`:** dequantise eight weight rows into an 8×K
  scratch and call `dot8ColsInto` (`Dot8x4`) — its per-row lane partials are the identical
  lane-wise FMLA sequence to `dotNEON4`'s and the Go fold is the same left-to-right `s0+s1+s2+s3`
  (`dot_arm64.go:44`, `dot8cols.go:11`); eight independent chains are load-bound at ≈10.7
  MACs/cycle, ~4× per row including the dequant. Also fixes the per-worker `make([]float32, K)`
  per parallel call (~16 MB/token in `int8`/`int4Mix` modes).
- **Confidence:** high.

### S-08 · amd64 — five items that are measurement or contract before they are kernels

1. **Measure the box.** No STREAM-class number for the 3700X exists; every "DRAM-bound" claim
   about it is an estimate. A 1..16-thread read-bandwidth probe is a prerequisite for S-02 there.
2. **`GOAMD64` is unpinned, and v3 fuses the pure-Go accumulates.** `s += x*y` in
   `dot_acc64.go`, `matmul_qk_acc64.go`, `matmul_av_acc64.go`, the strided span and every Go
   scalar tail (`dot.go:24-26`, `matmul_blocked.go`, `rowblock_amd64.go`) compiles to
   `MULSD+ADDSD` at v1 and `VFMADD231SD` at v3 — a different bit pattern for every f32-tail
   partial sum on the same CPU from the same source. (The acc64 kernels are exempt — exact
   products — but the f32 tails are not.) Nothing pins the level in either repo's CI, `go.mod`
   or release docs. A 3-line startup self-test (`x := 1+2⁻²⁷; y := 1−2⁻²⁷; x*y−1` is 0 unfused,
   −2⁻⁵⁴ fused) mixed into goinfer's parity `deps_hash`, or `GOAMD64=v1` pinned in the release
   build and said so.
3. **The VNNI W4A8 kernel keeps the AVX2 unpack and doubles the fold** (23 instructions vs 25);
   a 2-group split layout with the −8Σact term as a per-group int32 accumulator init lands at
   ≈10.5 instructions per 32 MACs at YMM width (bit-identical to AVX2 with a lane permutation)
   or ≈6 at ZMM (not). Neither project box can measure it; file, do not build.
4. **VEX `AVX-VNNI` hosts (Intel client 12th–14th gen) are not detected** — they take the AVX2
   tier; the EVEX.256 kernel already has the right width and needs a `{vex}` twin. A detection
   gap, not a wrong result.
5. **The Zen 2 "double-pump" explanation of the 2Acc null result is wrong** (a Zen 1 property);
   the real constraint is unmeasured. Two-arm synthetic: +4 independent 1-cycle dummy ops per
   group vs +4 dependent ops on the chain — if the chain arm slows and the port arm does not,
   the scheduler-window model holds and the levers are chain-shortening ones (which is what the
   split-half and unpack-free results already suggest). Also: `blockedFill` has no B-panel packing
   on amd64 (`matmul_blocked.go:60-62`, "waits on §2.4"), so prefill P·V with nKeys a multiple of
   1024 aliases L1 sets — `MatmulBT` M=64, K=4096 vs 4000, N=128 on the 3700X is the one-run
   check.

### S-09 · Eng — three gates that vouch for less than they read as

1. **The "serial ties parallel, fork/join is net-neutral" A/B compared two parallel arms.**
   ✅ **DONE 2026-09-03, and the 2026-08-11 result is REFUTED.** goinfer re-ran it with
   `ws.SetThreshold(1<<62)` on a quiet box:

   | arm | tok/s (×4 runs) | excess goroutines |
   |---|---|--:|
   | serial | 51.4 / 52.3 / 52.4 / 52.2 | 5 |
   | parallel | 85.8 / 88.5 / 89.3 / 88.5 | 31–33 |

   **parallel / serial = 1.70×.** Fork/join is not net-neutral; the earlier "serial 54.77 vs
   parallel 54.34" (`perf-campaign.md:386-401`, from `queue-release.md:758-766`) was two parallel
   arms, exactly as this finding suspected — `GOINFER_PAR_THRESHOLD` set only the process global
   while decode had used the per-Workspace 300K override since 2026-08-01. The new serial arm is
   proven serial TWO independent ways rather than one: the goroutine count does not scale, and at
   `GOMAXPROCS=1` it is unchanged (42.6) while the parallel arm collapses onto it (43.3). The
   residual 5 goroutines appear in both arms at one core, so they are runtime, not matmul spawn.
   A first pass run concurrently with the prefill benchmark read 1.35× — see S-01's contamination
   note.

   **This CONFIRMS S-02 rather than unsettling it, and the direction matters.** S-02's premise is
   "each core delivers 26–35% of what it does alone". At 1.70× across ~6 workers that is **28% per
   worker — inside the predicted band**. What is refuted is only the corollary the old A/B was
   used for; the trace-derived ~1%/token scheduler latency still stands. S-02 now has a
   measurement under it instead of an inference.

   **And the headroom is real rather than a bandwidth ceiling** — the check worth doing before
   building any S-02 remedy. One worker runs at 26 GB/s, which is 91–97% of its PORT ceiling but
   only ~36% of the M1 Pro's 71.9 GB/s single-thread STREAM, so a lone worker is issue-bound, not
   memory-bound. Six workers at that rate would want 156 GB/s against a 121 GB/s triad ceiling, so
   bandwidth alone permits ~4.65× — and the read-only ceiling is higher still than triad's, by the
   same RFO argument S-08.1 made for the 3700X. Realized is 1.70×. Roughly **2.7× is being left on
   the table that memory bandwidth does not explain**, which is S-02's to collect.
2. **Zero-alloc is pinned only on the serial path.** ✅ **DONE 2026-09-03.**
   `TestW8A8Into_zeroAllocWhenReused` runs at 4.35M MACs, below the default threshold, so it
   only ever pinned the serial path. `TestW8A8Into_boundedAllocOnParallelPath` now forces the
   parallel path (`SetThreshold(1)`, `SetWorkers(4)`) and bounds it. **Measured: 4 allocs/op,
   flat across N=512 and N=4864** — comfortably under the `workers+2` = 6 bound. The gate
   asserts the bound AND that the count does not grow with N, which is the half that actually
   catches an S-07-class regression: zero would have been the wrong bar (the fan-out legitimately
   allocates bookkeeping per dispatch), so what is pinned is "per dispatch, not per column".
3. **The row4 bit-identity unit test covers K ≤ 640** (nGroups 1..20); production K is 1536 and
   8960. ✅ **DONE 2026-09-03, with a correction to this finding.** There are TWO row4
   bit-identity gates and this item conflates them. The production-level one,
   `TestMatmulBTW4A8Row4Into_bitIdenticalToMatmulBTW4A8Into`, **already ran `{1536, 8960}` and
   `{8960, 1536}`** — the exact production shapes — so the claim as written overstates the gap.
   The narrow one is the RAW KERNEL gate,
   `TestDotW4A8SplitHalf4Row_bitIdenticalToCanonical` (`w4a8_sdotv2_test.go`), which swept
   `nGroups 1..20` only. That now also runs nGroups 48 and 280 (K = 1536, 8960) and **passes
   bit-identical**, so the structural argument has a measurement under it at the shapes that
   ship.

### S-10 · Eng — hygiene

- `fma_issue_probe_test.go:146-149` says Go "deliberately does NOT auto-fuse `a*b+c`"; it does
  on arm64. ✅ **DONE 2026-09-03 — the finding is correct and the comment is now corrected in
  place.** Settled by counting, on the Mac: `go build -gcflags=-S ./linalg | grep -c 'FMADD[SD]'`
  → **59** (46 FMADDS, 13 FMADDD), while `grep -rn 'math\.FMA' linalg/*.go | grep -v _test` →
  **0**. Zero explicit calls, 59 fused instructions: every one is auto-fused from plain `a*b+c`.
  Sites include `dot.go`'s scalar tail (`sum += a[k] * b[k]`) and — the one that matters —
  `matmul_qk_acc64.go`, an f64 attention kernel whose bit-identity is load-bearing.
- `WrapInt4Row4` (`weightmat.go:177-191`) validates lengths but not `rows%4`/`cols%group`.
  ✅ **DONE 2026-09-03.** It now rejects the shape at wrap, matching the two `requireExactLen`
  calls beside it — a caller cannot recover from a malformed weight blob, and failing at load
  names the blob where failing at the first matmul named a kernel. `RepackInt4Row4` already
  declined exactly this shape, so `WrapInt4Row4` was the other door into the same field.
  `TestWrapInt4Row4_rejectsBadShapeAtWrap` asserts the message names `WrapInt4Row4` and the
  offending dims, so it distinguishes WHERE the failure happens rather than merely that one does;
  verified by removing the guard and watching it fail.
- Non-DotProd arm64 gets a fully scalar W4A8 (`quant_w4a8_arm64.go:119`; W8A8 has a
  `SMULL/SADALP` fallback, W4A8 does not) — irrelevant on Apple silicon, relevant for
  Graviton1/A72-class targets and the `dotprod_arm64_other.go` BSD/Windows assumption.
- `.giw` f32 scale arrays are heap-copied at load (goinfer `decoder/serialize.go:1150-1161`):
  4 B per 32 weights = 12.5% of the int4 bytes ≈ 130 MB for the 1.5B — the kind-4 table's
  "resident memory added ~0" line should say so.
- K-quant dequant on load is scalar (`kquant.go` mirrors `embed/gguf.go`); it is load-time
  only and parallel across layers, and does not matter.

## 4. What changed since the August Go 1.27 audit

`go127-simd-audit.md` (2026-08-20) concluded: the `.s` fleet is not replaceable in 1.27 (no
SDOT/UDOT/I8MM, no `VPDPBUSD` intrinsic); the opening is the transcendental layer (item 13);
goinfer's G1 (scores·V vectorised across d) and G2 (four-key scalar ILP) were the attention
items; G4/G5 (f32 silu/gelu) were parity-gated; G10 (RoPE table) bit-identical. Since then:

- **G2 landed as A1's 8-chain restructure** (4.4× measured) and is now at 80% of its scalar
  issue bound — the remaining lever is the lane-per-output port (S-04), which the August doc
  scoped as G1 for AV and which this audit extends to QK with the exact-product argument.
- **Item 13 shipped in aikit** (`exp.go`; `exp_simd.go` behind `GOEXPERIMENT=simd`) and is used
  by nothing in goinfer (S-06). The August doc's step-1 recommendation (adopt the scalar f32
  family first, behind a per-family re-baseline) still stands unchanged; the new information is
  the prefill share, which makes the free parallelisation step worth taking before the parity
  decision.
- **The M>1 GEMM gap (S-01) was not in the August scope** — it is an assembly-level item the
  1.27 packages cannot express (no SDOT), which is why the field's answer is hand-written
  interleaved kernels on both arches (ggml `q4_0_4x4`/`8x8` repack GEMM with `SDOT` by element,
  KleidiAI's `dotprod` GEMV and `i8mm` GEMM). On the M1 the reachable shape is the `dotprod`
  4×4 tile; the `SMMLA` GEMM (2× MACs per instruction) is where a further 2× sits behind hardware
  aikit does not have (M2+/Graviton3+ — add detection when a box exists, `HWCAP2_I8MM`).
- **The fan-out finding (S-02) supersedes the August doc's silence on composition**; the pool
  negative is unchanged, the shard-skew question is new.
- Verdicts unchanged: no porting of the `.s` fleet to `archsimd`; no waiting for VNNI
  intrinsics; layerNorm, hamming and the f64 pool gather stay settled.

## 5. Program — sequenced by prize, each independently droppable

| step | item | prize (counted, not measured) | numerics | prerequisite |
|---|---|---|---|---|
| 0 | ✅ **STEP 0 COMPLETE 2026-09-03.** S-09.1 re-run (1.70×, old result refuted); S-08.1 STREAM the 3700X; and the per-shard timestamp harness — wake stagger confirmed, dynamic chunking measured against batching | decided where the decode gap is: the barrier | none | none |
| 1 | S-01 W4A8 M-invariance gate ✅ **DONE**, the 4×4 tile on row4 (arm64) ✅ **DONE 2026-09-03, 2.88×**; the unpack-once span (amd64) still open | CPU prefill 2–2.7× at the kernel; the int4/int8int8 inversion; verify rounds cheaper on every spec path | bit-identical | the gate |
| 2 | S-02 remedy that step 0 selects (dynamic chunking and/or `MatmulBTW4A8Batch`) + ~~S-03 NEON quantiser~~ (**S-03 built 2026-09-03, unmeasured**) | decode 1.15–1.7× on the fan-out term; −3 barriers, −3 quantisations per layer | bit-identical | step 0 |
| 3 | ~~S-04 AV pure-Go accumulator blocks → NEON AV → NEON QK~~ (**all three built 2026-09-03; the NEON ports unmeasured**) | ~1.9× token at depth 8k; ~1.1× at 128 | bit-identical (exact products) | none |
| 4 | S-06 step 1 (parallelise the elementwise loops) | prefill 7–25% at the 3700X rate, less on M1 | bit-identical | the stub measurement |
| 5 | S-05 centering fold + the scale-vector load — **now in BOTH W4A8 kernels, 1.33× counted on the tile (see the S-01 read-back)**; ~~S-07 `q8Span` 8-column form~~ (**built 2026-09-03, unmeasured**) | single-core kernel up to 1.29×; weight-only int8 ~4× per row | bit-identical | S-02 (to be visible) |
| 6 | S-06 step 2 (f32 transcendentals), S-08.3 VNNI redesign, S-08.2 GOAMD64 pin | parity-class / hardware-gated | not bit-identical / n/a | product decision / a VNNI host |
| 7 | I8MM detection + `SMMLA` GEMM | 2× on step 1's prefill kernel | bit-identity needs its own argument (different lane grouping) | an M2+/Graviton3 box |

Every step names its do-nothing arm as today's shipped configuration and is measured paired and
interleaved on a quiet box per the campaign rules; a counted number in this doc is a hypothesis
until the row exists.

## 6. Sources for the measured figures

aikit: `linalg/w4a8_item3_parallel_arm64_test.go` (1/6/8-worker GB/s), `w4a8_opsperbyte_bench_arm64_test.go`
(hot/cold GMAC/s, unpack tax), `matmul_w4a8_row4_arm64_test.go` (row4 vs canonical, `==` gate),
`fmapeak_*` (95.4 GFLOPS M1; 135.6 GFLOPS 3700X), `dot_w4a8_2acc_amd64_test.go`,
`dot_w4a8_splithalf_amd64_test.go`, `dot_w4a8_avx512vnni_amd64_bench_test.go`, `acc64_test.go`,
`matmul_qk_acc64_test.go`, `matmul_av_acc64_test.go`, `matmul_mconsistent_test.go`,
`decode_perf_test.go`; `docs/internal/perf-dead-ends.md` (§2, §4.4, §8.1, §8.10, Group 2),
`docs/internal/measuring-performance.md`, `docs/internal/roofline-2026-08.md`,
`docs/internal/cpu-acceleration.md`, `docs/internal/perf-amdahl-apple-m1pro.md`, CHANGELOG
1.15.0/1.23.0/1.24.0/1.25.0/1.29.0. goinfer: `docs/task-w4a8-neon-bandwidth.md` (Gate 0, Gate 1,
probes 1+2, item-3 harness, end-to-end, non-goals), `docs/task-attention-decode-cost.md` (A0
split, A1 moves, depth curve), `docs/measurements/cpu-peer-prefill-2026-09-01.md`,
`docs/measurements/aikit-w4a8-opsperbyte.md`, `docs/completed/perf-campaign.md` (Phase 0/3b,
trace coda, W8A8 batch), `docs/completed/queue-performance.md` (P14), `docs/parity-coverage-policy.md`
(fused-site census), `docs/audit-2026-09-02.md` (P-04, P-07, P-08, L-09). Field shapes from
memory of ggml `ggml-cpu/arch/arm/quants.c`, `repack.cpp`, `ggml-cpu/arch/x86/quants.c` and
KleidiAI's `kai_matmul_clamp_f32_qai8dxp*_qsi4cxp*` families — structure certain, exact op
counts not.

---

## Appendix A — goinfer-side scalar inventory (what is not in aikit's kernels)

Per token, Qwen2.5-Coder-1.5B, Mac int4, ~25 ms at depth ~128 (39–41 tok/s). Measured figures
from the Gate-0 stub (26.5% non-matmul floor), the probe-1 split (norm 0.40 ms, attention
15.16 pre-A1), the A0 split (softmax 0.39 ms serial at depth 130, RoPE 0.05) and the A1 depth
curve (2.81 ms @128, 85.2 ms @8192).

| term | where | work per token | depth 128 | depth 8192 | SIMD today | vectorisable bit-identically? |
|---|---|---|---|---|---|---|
| dense W4A8 matmuls | aikit | ~1.05 GB | ~21 ms (85%) | ~21 ms (20%) | SDOT row4, 6w | n/a |
| attention QK/PV acc64 | aikit (pure Go) | 2·12·28·nKeys·128 MACs | ~2.4 ms (10%) | ~81 ms (75%) | scalar f64 chains | **yes** — lane per output (S-04) |
| softmax (max/exp/normalise, f64 `math.Exp`) | goinfer, inside the head worker | 336·nKeys exps | ~0.07 ms (0.3%) | ~4 ms (4%) | scalar | the sum is order-pinned; a lane-wise exp is not `math.Exp`-identical |
| SiLU (f64 `x/(1+exp(−x))`) | goinfer, main goroutine | 250,880 | ≤1 ms (≤4%), unmeasured | same | scalar, serial | elementwise: parallelising is identical; a lane-wise exp must reproduce the algorithm |
| activation int8 quantisation | aikit, calling goroutine | 509k elements | 0.6–1.6 ms (2.5–6%) | same | scalar | **yes** (S-03) |
| RMSNorm (f64 square-sum) | goinfer | 57 × 1536 | 0.40 ms (1.6%) | same | scalar | reduction — leave |
| RoPE | goinfer | 25k rotations + 1.8k cos/sin | 0.05 ms | same | scalar f64 | table is value-identical; leave |
| fork/join | aikit | 169 dispatches | ~0.3 ms (1.2%) | same | — | S-02 |
| sampler, residual adds, KV append | goinfer | — | <0.5 ms | same | partly | — |
| DeltaNet recurrence (hybrid families) | goinfer `decoder/deltanet.go:226-243` | per layer state walk | ~19% of a 35B token | — | scalar, stride-`hv` | loop interchange keeps order (audit P-07) |
| Mamba-2 projections (Granite/Nemotron) | goinfer `mamba2.go` | f32 `matvec` | unmeasured | — | f32 kernels at 4 B/weight | WeightMat → W8A8 is a precision decision (audit P-08) |

Reading: at short context the Go-side serial remainder is ≈8–10% of the token and the whole of
it is four small items (quantiser, SiLU, norms, fork/join); at long context the token is the
acc64 kernels, which are aikit's and pure Go. The 73–86% that is weight streaming is the SDOT
kernel at its per-core ceiling and the fan-out at a third of six cores.

## Appendix B — what was read, and what is unconfirmed

Read in full: every `.s` under `linalg/` (all 16; every raw `WORD` recomputed), every non-test
`.go` in `linalg/`, the tests that record numbers (listed in §6), `docs/internal/*.md`,
`docs/architecture.md`, CHANGELOG entries for W4A8/W8A8/acc64/exp/Workspace; goinfer's
`decoder/{weightmat,scratch,tune,forwardn,mlp,rmsnorm,attention,rope,fusedattn,serialize}.go` and
the campaign records above; `go127-simd-audit.md` and its `proto/`.

Unconfirmed (would take a run on the box): the Firestorm port/latency constants (4 SIMD pipes
for SDOT/SCVTF/FMLA alike, 3 loads/cycle) — the 91–97% fractions are the evidence they fit;
whether `MOVI #0` is a rename-time zero idiom; which of wake-chain stagger vs E-core/SMT
stragglers dominates S-02; whether gc emitted `FCSEL` or branches in the quantiser; the cost of
a line-straddling 16-byte load on Apple silicon (the `.giw` arbitrary-base question, expected ≤
a few % single-core, nil at six workers); ~~the 3700X's real bandwidth~~ (**measured, S-08.1**);
the Go version that introduced `GOAMD64>=v3` FMA fusion; ggml/KleidiAI op counts beyond kernel
shape; ~~whether SDOT issues on four pipes or two~~ — **MEASURED 2026-09-03: four.**
`TestSDOTIssuePeak` reads 3.98 SDOT/cycle (12.73 G SDOT/s, 203.6 GMAC/s) on one M1 Pro P-core,
99.5% of the 4-pipe ceiling and 1.99× the 2-pipe one. The "halves S-01's counted gain" note
attached to it was wrong twice over: the count did not need halving, and neither tile is
SDOT-issue-limited in the first place.
