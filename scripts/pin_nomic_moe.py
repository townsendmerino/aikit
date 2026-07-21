#!/usr/bin/env python3
"""pin_nomic_moe.py — golden fixture for nomic-ai/nomic-embed-text-v2-moe, the
Bucket C (mixture-of-experts) capstone (docs/task-embedding-coverage.md Phase 4).

v2-moe is a nomic-bert (RoPE, post-norm) whose ODD layers replace the FFN with a
top-2-of-8 MoE, whose dense layers use a plain GELU fc1/fc2 (with biases, unlike
v1.5's SwiGLU), and whose tokenizer is XLM-R SentencePiece/Unigram. So this golden
gates every new piece at once: MoE routing, the dense GELU MLP, attention biases.

Dumps, from the real model:
  - input_ids : SentencePiece ids (<s> … </s>).
  - hidden    : last_hidden_state [L, 768] flattened row-major, with L.
  - embedding : the mean-pooled, L2-normalized sentence embedding [768].

Requires trust_remote_code (nomic-bert ships custom HF modeling). Run from root:
    .venv/bin/python scripts/pin_nomic_moe.py
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "testdata" / "nomic-moe"
OUT = REPO_ROOT / "testdata" / "nomic_moe_golden.json"

# Lean multilingual: routing is input-dependent, so varied scripts exercise
# different expert paths through the 6 MoE layers.
CASES = [
    "how do i parse json",
    "search_query: how do i parse json",
    "",
    "hello world",
    "x",
    "Bonjour le monde, ça va?",
    "識別子を検索する",
    "Здравствуй, мир",
    "compute the sha256 hash of a file",
]


def main() -> int:
    from sentence_transformers import SentenceTransformer

    sys.stderr.write(f"[pin_nomic_moe] loading {MODEL_DIR} (CPU, trust_remote_code) ...\n")
    m = SentenceTransformer(str(MODEL_DIR), device="cpu", trust_remote_code=True)
    cfg = m[0].auto_model.config

    out = {
        "model": "nomic-ai/nomic-embed-text-v2-moe",
        "hidden": cfg.n_embd,
        "layers": cfg.n_layer,
        "num_experts": cfg.num_experts,
        "moe_top_k": cfg.moe_top_k,
        "moe_every_n_layers": cfg.moe_every_n_layers,
        "act": cfg.activation_function,
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
        f"[pin_nomic_moe] wrote {OUT} — {len(CASES)} cases, dim {cfg.n_embd}, "
        f"experts={cfg.num_experts} top_k={cfg.moe_top_k}, {OUT.stat().st_size // 1024} KB\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
