#!/usr/bin/env python3
"""pin_bge.py — golden fixture for BAAI/bge-small-en-v1.5, the parity oracle for
aikit's CLS-pooled BERT embedder path (docs/task-embedding-coverage.md Phase 1).

bge-small is the same BERT architecture as all-MiniLM (learned ABSOLUTE positions,
GELU FFN, WordPiece) but pools the CLS token instead of mean — so it certifies the
declared-pooling work end-to-end: the Go loader reads pooling_mode_cls_token from
1_Pooling/config.json and must reproduce sentence-transformers' CLS-pooled vector.

For a few short curated cases this dumps, from the real model:
  - input_ids : the WordPiece ids the model ate ([CLS] … [SEP]). Fed directly to
                the Go forward, so tokenizer parity is out of scope here.
  - hidden    : last_hidden_state [L, 384] flattened row-major, with L.
  - embedding : the CLS-pooled, L2-normalized sentence embedding [384].

Run from the repo root:
    .venv/bin/python scripts/pin_bge.py
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "testdata" / "bge-small"
OUT = REPO_ROOT / "testdata" / "bge_golden.json"

# Short, varied cases (kept small — the per-token hidden dump stays compact).
CASES = [
    "how do i parse json",
    "def add(a, b):\n    return a + b",
    "",  # degenerate: [CLS][SEP]
    "hello world",
    "x",  # single char
    "compute the sha256 hash of a file",
    "!!! ??? ...",  # all punctuation
    "Represent this sentence for searching relevant passages",  # BGE query-ish
]


def main() -> int:
    from sentence_transformers import SentenceTransformer

    sys.stderr.write(f"[pin_bge] loading {MODEL_DIR} (CPU) ...\n")
    m = SentenceTransformer(str(MODEL_DIR), device="cpu")
    cfg = m[0].auto_model.config

    out = {
        "model": "BAAI/bge-small-en-v1.5",
        "hidden": cfg.hidden_size,
        "layers": cfg.num_hidden_layers,
        "heads": cfg.num_attention_heads,
        "intermediate": cfg.intermediate_size,
        "max_pos": cfg.max_position_embeddings,
        "type_vocab": cfg.type_vocab_size,
        "ln_eps": cfg.layer_norm_eps,
        "act": cfg.hidden_act,
        "pos": cfg.position_embedding_type,
        "pooling": "cls",
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
        f"[pin_bge] wrote {OUT} — {len(CASES)} cases, dim {cfg.hidden_size}, "
        f"pooling=cls, {OUT.stat().st_size // 1024} KB\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
