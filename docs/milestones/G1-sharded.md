# Milestone G1 — sharded safetensors loader

Parent plan: [`docs/multi-model-plan.md`](../multi-model-plan.md) §5.1, §7 (G1).
Touches: `embed/safetensors.go` (sharded opens), `decoder/weights.go` (wiring).

Status: **DONE & validated on Linux 2026-06-02.** A multi-shard copy of the 270M
checkpoint loads and reproduces the M1 sampled-tensor checksums; end-to-end
generation from the sharded dir matches the single-file output.

## Goal

`LoadWeights` opened one `model.safetensors`. Anything above ~2B params (Gemma 3
4B/12B/27B, every Llama ≥7B, all the MoE coders) ships as
`model-0000i-of-0000N.safetensors` + a `model.safetensors.index.json`
`weight_map` (tensor name → shard file). The plan calls this *"the #1 practical
blocker … do it first."* Without it the multi-model generalization is academic
for any model you'd actually want to run.

## What changed

- **`embed`** gained sharded opens that mmap (or heap-read) each shard once and
  **merge their tensor maps into a single `SafetensorsFile`** — so the decoder's
  `st.Tensor(name)` / `st.Close()` work unchanged whether the checkpoint is one
  file or twenty:
  - `OpenSafetensorsShardedMmap(indexPath)` — mmap path (the real loader).
  - `OpenSafetensorsShardedFromFS(fsys, indexPath)` — fs.FS/heap path.
  - `parseShardIndex` reads the `weight_map`; `mergeShards` folds every shard's
    tensors in and verifies every promised tensor resolved.
  - The `mmapped` field became `[][]byte` so `Close`/the finalizer munmap all
    shard regions; a shared `mmapReadOnly` helper backs single- and multi-file.
- **`decoder`** picks the path automatically: `openCheckpointMmap` /
  `openCheckpointFromFS` use the sharded open when `model.safetensors.index.json`
  is present, else the single `model.safetensors`. No API change.

## Validation

- **Mechanism** (`embed/safetensors_sharded_test.go`, committed, no fixtures):
  two tensors split across two shards resolve correctly via both the fs and mmap
  paths (incl. `Close` idempotency + munmap); empty `weight_map` and a
  missing-from-shard tensor error.
- **Real model** (`decoder/sharded_test.go`, skip-clean): the 270M split into 3
  shards by `scripts/shard_checkpoint.py` reproduces the **M1 sampled-tensor
  checksums** (shared `checkSampledChecksums` with the single-file test) — proof
  the cross-shard resolution yields byte-equal tensors. Tensors are spread
  round-robin so the layers genuinely straddle shards.
- **End to end:** `demo/gemma --model testdata/gemma-3-270m-sharded` →
  *"…Paris. It is the most visited city in the world."* — identical to the
  single-file run.
- `go build`/`vet`/`gofmt` clean; the existing encoder mmap usage is unaffected
  by the `mmapped` refactor (its tests pass).

## Notes / follow-ups

- The fixture (`scripts/shard_checkpoint.py` output) is per-machine, ~536 MB,
  gitignored — regenerate to run the real-model test. A genuinely-sharded
  download (Gemma 3 4B) would also exercise it.
- **Next:** G2 (Llama/Mistral/Qwen2 adapter + tensor schema + QKV bias + untied
  head) then G3 (byte-level BPE tokenizer) — the path to a real coding model
  (Qwen2.5-Coder). G1 unblocks the ≥7B / MoE checkpoints those scale to.
