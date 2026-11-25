# Roofline census: registration for the Ryzen 7 3700X and M1 Pro readings

Registered 2026-11-24, before either reading. The tool is `tools/roofline`; the rows are `linalg`'s `BenchmarkRooflineCensus`.

**Question.** Single core, which of the kernels perfgate or goinfer's kernelprint names sit furthest below this box's roof, at a
cache-resident shape and at a streamed shape? The answer ranks later speed work. It claims no speed, and nothing is built from it.

**Shapes.** Each kernel's largest operand is 128 KiB (resident) and 128 MiB (streamed). Matmuls run at M = 1, plus M = 32 for the
f32, W8A8 and repacked int4 tiles; the acc64 attention kernels run at head dim 128; the fused ViT attention has no streamed shape.
The table names each row's exact shape.

**Roofs**, each per core:
- f32 FMA peak: `measureFMAPeak`, measured. f64 is half of it.
- int8 dot peak: on the M1 Pro, the SDOT issue probe, measured. On the 3700X, 69.0 GMAC/s, cited from
  `docs/internal/roofline-2026-08.md` § 0 (VPMADDWD, the same box). There is no amd64 probe in the tree.
- streaming read: `bytes.IndexByte` over 512 MiB, best of five. Its cross-check: the same probe over 32 KiB must run at least twice
  as fast, or the probe is instruction-bound and the reading is INCONCLUSIVE.

A resident row's roof is its class's compute roof. A streamed row's roof is the lower of that and its MACs per operand byte times
the streaming roof. An elementwise row is held to the streaming roof alone.

**Lines.**
- **Laggard:** a streamed row below 0.45 of its roof, or a resident row with at least 8 MACs per byte below 0.25.
- **Parked:** a row between that and the line (0.50 streamed, 0.30 resident): listed, not queued.
- Resident rows under 8 MACs per byte are reported and not judged: no cache roof was measured.

**Run.** On each box, by day, under the exclusive timing lock, once the 1-minute load has held under 0.3 for two minutes:
from the aikit checkout at main's head, `python3 <goinfer>/scripts/timing_lock.py run --label roofline -- go run -C tools ./roofline`
(the lock is goinfer's). On the 3700X, add `--int8-roof 69.0 --int8-roof-source "docs/internal/roofline-2026-08.md § 0,
Ryzen 7 3700X"`. Three passes at 300 ms. Expected wall time: about 10 minutes, at most 15. Peak memory is under 1.5 GB.

**Stopping rule.** One reading per box. A reading started under load is void; it is re-run once and listed, never averaged.

**What each outcome means.**
- **PASS with laggards:** each laggard becomes one performance-queue entry with its measured gap, as the fraction and as the
  absolute GB/s or GMAC/s short of its bound.
- **PASS with none:** nothing is queued, and the table stands as the baseline the next reading is compared against, on the same
  box.
- **INCONCLUSIVE:** nothing is queued. The probe or the row it names is fixed, and the box is booked again.
