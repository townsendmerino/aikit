#!/usr/bin/env python3
"""pin_retrieval.py — golden fixture for minishlab/potion-retrieval-32M (roadmap
§2.4). This is the standard (non-quantized) Model2Vec format: the safetensors holds
only `embeddings` [vocab, dim] — no `mapping` or `weights` tensors — so token ids
index the embedding rows directly and pooling is a plain mean. Pins embed's
no-mapping path against StaticModel.encode.

Run from the repo root with the §2.1 toolchain:
    .venv/bin/python scripts/pin_retrieval.py
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODEL = REPO / "testdata" / "retrieval-model"
OUT = REPO / "testdata" / "retrieval_model_golden.json"

CASES = [
    "how do i parse json",
    "recursive directory walk that respects gitignore",
    "hello world",
    "",
    "x",
    "compute the sha256 hash of a file",
    "machine learning embeddings for semantic search",
    "!!! ??? ...",
]


def main() -> int:
    from model2vec import StaticModel

    m = StaticModel.from_pretrained(str(MODEL))
    out = {"model": "minishlab/potion-retrieval-32M", "cases": []}
    for text in CASES:
        v = m.encode(text)
        out["cases"].append({"text": text, "embedding": [round(float(x), 6) for x in v.tolist()]})
    out["dim"] = len(out["cases"][0]["embedding"])
    OUT.write_text(json.dumps(out))
    sys.stderr.write(f"[pin_retrieval] wrote {OUT} — {len(CASES)} cases, dim {out['dim']}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
