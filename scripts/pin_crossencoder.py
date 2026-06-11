#!/usr/bin/env python3
"""pin_crossencoder.py — golden for cross-encoder/ms-marco-MiniLM-L-6-v2 (roadmap
v3 §1). A BERT cross-encoder reranker: it scores a (query, document) pair jointly.
For curated pairs this dumps the input_ids + token_type_ids the model ate and the
classification logit(s) — the relevance score the Go CrossEncoder must reproduce.

Run from the repo root with the §2.1 toolchain:
    .venv/bin/python scripts/pin_crossencoder.py
"""
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODEL = REPO / "testdata" / "crossencoder-model"
OUT = REPO / "testdata" / "crossencoder_golden.json"

PAIRS = [
    ("how to parse json in python", "import json; json.loads(s) parses a JSON string into a dict"),
    ("how to parse json in python", "the mitochondria is the powerhouse of the cell"),
    ("what is the capital of france", "Paris is the capital and largest city of France"),
    ("what is the capital of france", "def add(a, b):\n    return a + b"),
    ("how do neural networks learn", "neural networks learn representations via gradient descent on a loss"),
    ("json", "python json parsing with the json module"),
]


def main() -> int:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL)).eval()
    out = {"model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
           "labels": int(model.config.num_labels), "cases": []}
    for q, d in PAIRS:
        enc = tok(q, d, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**enc).logits.squeeze(0)  # [labels]
        out["cases"].append({
            "query": q, "doc": d,
            "input_ids": [int(i) for i in enc["input_ids"][0].tolist()],
            "token_type_ids": [int(i) for i in enc["token_type_ids"][0].tolist()],
            "score": [round(float(x), 6) for x in logits.tolist()],
        })
    OUT.write_text(json.dumps(out))
    sys.stderr.write(f"[pin_crossencoder] wrote {OUT} — {len(PAIRS)} pairs, labels {out['labels']}, "
                     f"scores {[c['score'][0] for c in out['cases']]}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
