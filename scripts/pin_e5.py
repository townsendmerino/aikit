#!/usr/bin/env python3
"""pin_e5.py — golden fixture for intfloat/multilingual-e5-base, the CAPSTONE
parity oracle for aikit's multilingual embedder stack (docs/task-embedding-coverage.md
Phase 2).

Unlike xlm-roberta-base (a bare LM, forward-only), multilingual-e5-base is a real
sentence-transformers embedder — genuine XLM-R (model_type=xlm-roberta, so
posOff=2), a SentencePiece/Unigram tokenizer, and a mean-pooling head. It
therefore certifies the FULL stack against one reference: Unigram tokenizer +
position-id offset + mean pooling + forward, end to end, with a cosine gate.

Dumps, from the real model:
  - input_ids : the SentencePiece ids (<s> … </s>). Fed to the Go forward and
                compared against aikit's own Encode(text) for tokenizer parity.
  - hidden    : last_hidden_state [L, 768] flattened row-major, with L.
  - embedding : the mean-pooled, L2-normalized sentence embedding [768].

Run from the repo root:
    .venv/bin/python scripts/pin_e5.py
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "testdata" / "multilingual-e5-base"
OUT = REPO_ROOT / "testdata" / "e5_golden.json"

# e5 conventionally takes "query:"/"passage:" prefixes; we feed raw text (parity
# is about the forward, not the prompt convention) and lean multilingual to
# exercise the SentencePiece tokenizer + XLM-R offset together.
CASES = [
    "how do i parse json",
    "query: how do i parse json",  # e5 prompt shape
    "",  # degenerate: <s></s>
    "hello world",
    "x",
    "Bonjour le monde, ça va?",
    "Grüße aus München — Straße",
    "識別子を検索する",
    "Здравствуй, мир",
    "مرحبا بالعالم",
    "compute the sha256 hash of a file",
]


def main() -> int:
    from sentence_transformers import SentenceTransformer

    sys.stderr.write(f"[pin_e5] loading {MODEL_DIR} (CPU) ...\n")
    m = SentenceTransformer(str(MODEL_DIR), device="cpu")
    cfg = m[0].auto_model.config

    out = {
        "model": "intfloat/multilingual-e5-base",
        "hidden": cfg.hidden_size,
        "layers": cfg.num_hidden_layers,
        "model_type": cfg.model_type,
        "pad_token_id": cfg.pad_token_id,
        "pos_offset": cfg.pad_token_id + 1,
        "pooling": "mean",
        "cases": [],
    }
    for text in CASES:
        ids = m.tokenize([text])["input_ids"][0].tolist()
        hid = m.encode(text, output_value="token_embeddings", convert_to_numpy=True)
        emb = m.encode(text, normalize_embeddings=True, convert_to_numpy=True)
        out["cases"].append({
            "text": text,
            "input_ids": [int(i) for i in ids],
            "L": int(hid.shape[0]),
            "hidden": [round(float(x), 6) for x in hid.reshape(-1).tolist()],
            "embedding": [round(float(x), 6) for x in emb.tolist()],
        })

    OUT.write_text(json.dumps(out))
    sys.stderr.write(
        f"[pin_e5] wrote {OUT} — {len(CASES)} cases, dim {cfg.hidden_size}, "
        f"pooling=mean, pos_offset={out['pos_offset']}, {OUT.stat().st_size // 1024} KB\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
