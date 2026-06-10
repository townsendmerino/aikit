# scripts/ — the parity oracle

The `pin_*.py` generators are aikit's parity oracle. Each loads a real model (via
PyTorch / sentence-transformers / transformers / model2vec) and dumps a golden
fixture into `testdata/`, which the Go tests then assert against bit-for-bit-ish.
This is design rule 3 (numerics are parity-pinned): every model-touching Go path —
`embed`, `encoder`, the `linalg` quant kernels — is checked against this reference,
so a port bug surfaces as a failing test, not a silent accuracy regression.

## Toolchain setup (roadmap §2.1)

CPU-only, runnable on a Mac. The venv is gitignored; `requirements.txt` pins the
versions that produced the committed goldens.

```sh
python3 -m venv .venv
.venv/bin/pip install -r scripts/requirements.txt
```

Verify it works (loads + embeds, ~90 MB download on first run):

```sh
.venv/bin/python -c "from sentence_transformers import SentenceTransformer as S; \
  print(S('sentence-transformers/all-MiniLM-L6-v2').encode('hi').shape)"   # (384,)
```

## Regenerating a golden

Each script writes a fixed `testdata/*.json` path; run from the repo root, e.g.:

```sh
.venv/bin/python scripts/pin_encoder.py     # → testdata/encoder_golden.json (CodeRankEmbed)
.venv/bin/python scripts/pin_inference.py   # → the Model2Vec embed golden
```

Models are fetched from the Hugging Face Hub on first run. GGUF dequant scripts
(`pin_iq_dequant.py`) additionally need `pip install gguf`.
