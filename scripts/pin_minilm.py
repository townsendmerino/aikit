#!/usr/bin/env python3
"""pin_minilm.py — golden fixture for sentence-transformers/all-MiniLM-L6-v2, the
parity oracle for aikit's MiniLM-class (BERT) encoder (roadmap §2.2).

MiniLM differs from CodeRankEmbed on three axes the Go forward must implement:
learned ABSOLUTE position embeddings (not RoPE), a GELU FFN (not SwiGLU), and mean
pooling (not CLS). For a few short curated cases this dumps, from the real model:

  - input_ids : the WordPiece ids the model ate ([CLS] … [SEP]). Fed directly to
                the Go forward, so tokenizer parity is out of scope here.
  - hidden    : the last_hidden_state [L, 384] (the full transformer forward — the
                axis-by-axis oracle), flattened row-major, with L.
  - embedding : the mean-pooled, L2-normalized sentence embedding [384] (the
                end-to-end product).

Run from the repo root with the §2.1 toolchain:
    .venv/bin/python scripts/pin_minilm.py
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "testdata" / "minilm-model"
OUT = REPO_ROOT / "testdata" / "minilm_golden.json"

# Short, varied cases — kept small so the per-token hidden-state dump stays compact
# (truncation/tokenizer behavior is pinned separately via the WordPiece tests).
CASES = [
    "how do i parse json",
    "def add(a, b):\n    return a + b",
    "",  # degenerate: [CLS][SEP]
    "こんにちは 世界",  # unicode
    "hello world",
    "x",  # single char
    "compute the sha256 hash of a file",
    "!!! ??? ...",  # all punctuation
]


def main() -> int:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    sys.stderr.write(f"[pin_minilm] loading {MODEL_DIR} (CPU) ...\n")
    m = SentenceTransformer(str(MODEL_DIR), device="cpu")
    cfg = m[0].auto_model.config

    out = {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "hidden": cfg.hidden_size,
        "layers": cfg.num_hidden_layers,
        "heads": cfg.num_attention_heads,
        "intermediate": cfg.intermediate_size,
        "max_pos": cfg.max_position_embeddings,
        "type_vocab": cfg.type_vocab_size,
        "ln_eps": cfg.layer_norm_eps,
        "act": cfg.hidden_act,
        "pos": cfg.position_embedding_type,
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
        f"[pin_minilm] wrote {OUT} — {len(CASES)} cases, dim {cfg.hidden_size}, "
        f"act={cfg.hidden_act} pos={cfg.position_embedding_type}, {OUT.stat().st_size // 1024} KB\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
