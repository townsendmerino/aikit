#!/usr/bin/env python3
"""pin_bge_m3.py — golden fixture for BAAI/bge-m3, the CLS-pooled multilingual
capstone (docs/task-embedding-coverage.md Phase 2).

bge-m3 is the flagship multilingual retriever: a large XLM-R backbone (24 layers,
1024-dim, so posOff=2), a SentencePiece/Unigram tokenizer, and CLS pooling on the
dense head. It reuses the exact path multilingual-e5-base certified, but CLS
instead of mean and at 2x depth — so it independently exercises the CLS reduction
on the multilingual stack.

bge-m3 ships only pytorch_model.bin; convert it to model.safetensors first (see
the loader note in the certification commit) so aikit's LoadBERT can read it.

Dumps, from the real model:
  - input_ids : SentencePiece ids (<s> … </s>).
  - hidden    : last_hidden_state [L, 1024] flattened row-major, with L.
  - embedding : the CLS-pooled, L2-normalized dense embedding [1024].

Run from the repo root:
    .venv/bin/python scripts/pin_bge_m3.py
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "testdata" / "bge-m3"
OUT = REPO_ROOT / "testdata" / "bge_m3_golden.json"
ENC_OUT = REPO_ROOT / "testdata" / "bge_m3_encode_golden.json"

# Cosine cases: the sentence-transformers wrapper strips leading/trailing
# whitespace before encoding, so we keep inputs where the raw tokenizer and ST
# agree (realistic retrieval text). The Replace-collapse and leading cases stay —
# they exercise the bge-m3 normalizer/Metaspace and match either way.
CASES = [
    "how do i parse json",
    "",  # degenerate: <s></s>
    "hello world",
    "x",
    "Bonjour le monde, ça va?",
    "Grüße aus München — Straße",
    "識別子を検索する",
    "Здравствуй, мир",
    "مرحبا بالعالم",
    "search the knowledge base for relevant passages",
    "a  b  c",  # collapsed inner spaces (Replace normalizer)
    "  leading spaces",  # leading run
    "tab\tand\nnewline",  # non-space whitespace via the charsmap
]

# Tokenizer cases: id parity against the RAW tokenizer (tokenizer.json contract,
# not the ST wrapper) — includes the trailing-space case where a lone ▁ survives,
# which the ST wrapper strips but the raw tokenizer (and aikit) keep.
ENCODE_CASES = CASES + [
    "trailing spaces  ",
    "many    inner     spaces",
    "  both ends  ",
]


def main() -> int:
    from sentence_transformers import SentenceTransformer

    sys.stderr.write(f"[pin_bge_m3] loading {MODEL_DIR} (CPU) ...\n")
    m = SentenceTransformer(str(MODEL_DIR), device="cpu")
    cfg = m[0].auto_model.config

    out = {
        "model": "BAAI/bge-m3",
        "hidden": cfg.hidden_size,
        "layers": cfg.num_hidden_layers,
        "model_type": cfg.model_type,
        "pad_token_id": cfg.pad_token_id,
        "pos_offset": cfg.pad_token_id + 1,
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
        f"[pin_bge_m3] wrote {OUT} — {len(CASES)} cases, dim {cfg.hidden_size}, "
        f"pooling=cls, pos_offset={out['pos_offset']}, {OUT.stat().st_size // 1024} KB\n"
    )

    # Raw-tokenizer id oracle (the tokenizer.json contract aikit reproduces).
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    enc = {"model": "BAAI/bge-m3", "cases": [
        {"text": t, "input_ids": [int(i) for i in tk.encode(t).ids]} for t in ENCODE_CASES
    ]}
    ENC_OUT.write_text(json.dumps(enc))
    sys.stderr.write(f"[pin_bge_m3] wrote {ENC_OUT} — {len(ENCODE_CASES)} raw-tokenizer cases\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
