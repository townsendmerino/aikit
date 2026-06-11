#!/usr/bin/env python3
"""pin_splade.py — golden for naver/splade-cocondenser-ensembledistil (roadmap §2.3).
SPLADE expansion = BertForMaskedLM logits → log(1+ReLU) → max-pool over tokens, a
sparse vector over the BERT vocabulary. Dumps the nonzero (term, weight) pairs.

Run from the repo root with the §2.1 toolchain:
    .venv/bin/python scripts/pin_splade.py
"""
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODEL = REPO / "testdata" / "splade-model"
OUT = REPO / "testdata" / "splade_golden.json"
CASES = ["how do i parse json", "machine learning for semantic search",
         "the quick brown fox jumps", "x"]


def main() -> int:
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL))
    model = AutoModelForMaskedLM.from_pretrained(str(MODEL)).eval()
    out = {"model": "naver/splade-cocondenser-ensembledistil",
           "vocab": int(model.config.vocab_size), "cases": []}
    for text in CASES:
        enc = tok(text, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits  # [1, L, V]
        relu_log = torch.log1p(torch.relu(logits))
        mask = enc["attention_mask"].unsqueeze(-1)
        vec = (relu_log * mask).max(dim=1).values.squeeze(0)  # [V]
        nz = torch.nonzero(vec).squeeze(-1).tolist()
        out["cases"].append({
            "text": text,
            "input_ids": [int(i) for i in enc["input_ids"][0].tolist()],
            "terms": [int(i) for i in nz],
            "weights": [round(float(vec[i]), 6) for i in nz],
        })
    OUT.write_text(json.dumps(out))
    sys.stderr.write(f"[pin_splade] wrote {OUT} — {len(CASES)} cases, vocab {out['vocab']}, "
                     f"nnz {[len(c['terms']) for c in out['cases']]}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
