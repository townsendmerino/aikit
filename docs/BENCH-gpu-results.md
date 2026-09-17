# GPU benchmark results

> **GENERATED** by `bench report` from `records.jsonl` — do not edit by hand.
> Re-generate after each periodic run (`docs/BENCH-gpu.md`). Cross-machine
> absolute numbers are never placed in adjacent columns — the CPU baselines are
> different chips; compare within a machine, or via the normalized summary.

aikit `unknown` · 76 records · 2 machine(s)

## Per-machine tables (apples-to-apples, same box)

### apple-m1pro Apple M1 Pro arm64 · GPU Apple M1 Pro (Apple)

| workload | shape | precision | cpu-arm64 q/s | cpu+metal q/s | speedup | recall@k | parity |
|---|---|---|--:|--:|--:|--:|:--:|
| ann.FlatI8.Query | N=10k dim=256 batch=1 k=10 | int8 | 6.8k queries/s | 3.0k queries/s | 0.44× | 0.9922 | ✅ |
| ann.FlatI8.Query | N=100k dim=256 batch=1 k=10 | int8 | 1.6k queries/s | 197.9 queries/s | 0.12× | 0.9781 | ✅ |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=1 k=10 | int8 | 7.0k queries/s | 1.8k queries/s | 0.25× | 1.0000 | ✅ |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=8 k=10 | int8 | 7.0k queries/s | 7.2k queries/s | 1.04× | 0.9875 | ✅ |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=64 k=10 | int8 | 6.9k queries/s | 19.7k queries/s | 2.86× | 0.9922 | ✅ |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=256 k=10 | int8 | 6.8k queries/s | 23.8k queries/s | 3.50× | 0.9883 | ✅ |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=1 k=10 | int8 | 2.1k queries/s | 206.1 queries/s | 0.10× | 1.0000 | ✅ |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=8 k=10 | int8 | 2.1k queries/s | 1.5k queries/s | 0.73× | 0.9750 | ✅ |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=64 k=10 | int8 | 1.7k queries/s | 3.5k queries/s | 2.11× | 0.9781 | ✅ |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=256 k=10 | int8 | 1.8k queries/s | 3.7k queries/s | 2.06× | 0.9754 | ✅ |
| vision.SigLIP.Forward | dim=512 patches=196 | int8 | 4.4 images/s | 6.1 images/s | 1.37× | 0.9999 | ✅ |
| vision.SigLIP.Forward | dim=768 patches=576 | int8 | 0.7 images/s | 1.1 images/s | 1.54× | 0.9999 | ✅ |

### nvidia amd64 · GPU NVIDIA GeForce RTX 2070 SUPER

| workload | shape | precision | cpu-amd64 q/s | cpu+cuda q/s | speedup | recall@k | parity |
|---|---|---|--:|--:|--:|--:|:--:|
| ann.FlatI8.Query | N=10k dim=256 batch=1 k=10 | int8 | 12.5k queries/s | 11.3k queries/s | 0.90× | 0.9922 | ✅ |
| ann.FlatI8.Query | N=100k dim=256 batch=1 k=10 | int8 | 1.3k queries/s | 4.4k queries/s | 3.32× | 0.9781 | ✅ |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=1 k=10 | int8 | 12.9k queries/s | 9.7k queries/s | 0.75× | 1.0000 | ✅ |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=8 k=10 | int8 | 29.6k queries/s | 27.5k queries/s | 0.93× | 0.9875 | ✅ |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=64 k=10 | int8 | 24.2k queries/s | 40.8k queries/s | 1.69× | 0.9922 | ✅ |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=256 k=10 | int8 | 29.2k queries/s | 45.6k queries/s | 1.56× | 0.9883 | ✅ |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=1 k=10 | int8 | 1.9k queries/s | 5.0k queries/s | 2.62× | 1.0000 | ✅ |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=8 k=10 | int8 | 4.8k queries/s | 20.6k queries/s | 4.27× | 0.9750 | ✅ |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=64 k=10 | int8 | 5.3k queries/s | 37.9k queries/s | 7.10× | 0.9781 | ✅ |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=256 k=10 | int8 | 5.3k queries/s | 53.5k queries/s | 10.02× | 0.9754 | ✅ |

## Normalized cross-platform summary (speedup over each box's own CPU)

The only honest all-backends view: absolute ms don't compare across machines, but
*speedup over the CPU each GPU ships next to* does — the decision-relevant number.

| workload | shape | precision | cpu+cuda ×vs-cpu | cpu+metal ×vs-cpu | cuda ×vs-cpu | metal ×vs-cpu |
|---|---|---|--:|--:|--:|--:|
| ann.FlatI8.Query | N=10k dim=256 batch=1 k=10 | int8 | — | — | 0.90× | 0.44× |
| ann.FlatI8.Query | N=100k dim=256 batch=1 k=10 | int8 | — | — | 3.32× | 0.12× |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=1 k=10 | int8 | 0.54× | 0.25× | 0.75× | 0.14× |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=8 k=10 | int8 | 0.80× | 1.50× | 0.93× | 1.04× |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=64 k=10 | int8 | 1.95× | 2.29× | 1.69× | 2.86× |
| ann.FlatI8.QueryBatch | N=10k dim=256 batch=256 k=10 | int8 | 1.78× | 3.68× | 1.56× | 3.50× |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=1 k=10 | int8 | 2.38× | 0.17× | 2.62× | 0.10× |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=8 k=10 | int8 | 2.17× | 1.05× | 4.27× | 0.73× |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=64 k=10 | int8 | 2.26× | 2.66× | 7.10× | 2.11× |
| ann.FlatI8.QueryBatch | N=100k dim=256 batch=256 k=10 | int8 | 2.35× | 2.42× | 10.02× | 2.06× |
| vision.SigLIP.Forward | dim=512 patches=196 | int8 | — | — | — | 1.37× |
| vision.SigLIP.Forward | dim=768 patches=576 | int8 | — | — | — | 1.54× |

## Dispatch thresholds (the crossover — the input to backend dispatch)

- **apple-m1pro/ann.FlatI8.Query (int8, N=100k)**: CPU wins at every measured batch (GPU never overtakes).
- **apple-m1pro/ann.FlatI8.Query (int8, N=10k)**: CPU wins at every measured batch (GPU never overtakes).
- **apple-m1pro/ann.FlatI8.QueryBatch (int8, N=100k)**: GPU overtakes CPU at **batch ≥ 64**.
- **apple-m1pro/ann.FlatI8.QueryBatch (int8, N=10k)**: GPU overtakes CPU at **batch ≥ 8**.
- **nvidia/ann.FlatI8.Query (int8, N=100k)**: GPU overtakes CPU at **batch ≥ 1**.
- **nvidia/ann.FlatI8.Query (int8, N=10k)**: CPU wins at every measured batch (GPU never overtakes).
- **nvidia/ann.FlatI8.QueryBatch (int8, N=100k)**: GPU overtakes CPU at **batch ≥ 1**.
- **nvidia/ann.FlatI8.QueryBatch (int8, N=10k)**: GPU overtakes CPU at **batch ≥ 64**.
