#!/usr/bin/env python3
"""pin_nomic_embed.py — golden fixture for nomic-ai/nomic-embed-text-v1.5, the
parity oracle for aikit's MEAN-pooled nomic-bert (RoPE) embedder path
(docs/task-embedding-coverage.md Phase 1).

nomic-embed-text is the same nomic-bert architecture as CodeRankEmbed (RoPE,
SwiGLU FFN) but pools MEAN instead of CLS — so it certifies the Nomic loader's
declared-pooling change on the RoPE path (the path all-MiniLM/bge, being BERT,
don't exercise): LoadWeightsFromFS reads pooling_mode_mean_tokens from
1_Pooling/config.json and the forward must reproduce sentence-transformers'
mean-pooled vector.

Dumps, from the real model (raw text, no task prefix, so the Go forward eats the
same ids):
  - input_ids : WordPiece ids ([CLS] … [SEP]).
  - hidden    : last_hidden_state [L, 768] flattened row-major, with L.
  - embedding : the mean-pooled, L2-normalized sentence embedding [768].

Requires trust_remote_code (nomic-bert ships custom HF modeling). Run from root:
    .venv/bin/python scripts/pin_nomic_embed.py
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "testdata" / "nomic-embed"
OUT = REPO_ROOT / "testdata" / "nomic_embed_golden.json"

CASES = [
    "how do i parse json",
    "def add(a, b):\n    return a + b",
    "",  # degenerate: [CLS][SEP]
    "hello world",
    "x",  # single char
    "compute the sha256 hash of a file",
    "search_document: a passage about databases",  # nomic task-prefix shape
]


def main() -> int:
    from sentence_transformers import SentenceTransformer

    sys.stderr.write(f"[pin_nomic] loading {MODEL_DIR} (CPU, trust_remote_code) ...\n")
    m = SentenceTransformer(str(MODEL_DIR), device="cpu", trust_remote_code=True)
    cfg = m[0].auto_model.config

    out = {
        "model": "nomic-ai/nomic-embed-text-v1.5",
        "hidden": cfg.n_embd,
        "layers": cfg.n_layer,
        "heads": cfg.n_head,
        "act": cfg.activation_function,
        "rotary_base": cfg.rotary_emb_base,
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
        f"[pin_nomic] wrote {OUT} — {len(CASES)} cases, dim {cfg.n_embd}, "
        f"pooling=mean, {OUT.stat().st_size // 1024} KB\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
