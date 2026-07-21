#!/usr/bin/env python3
"""pin_xlmr.py — forward-only golden for FacebookAI/xlm-roberta-base, the parity
oracle for aikit's XLM-R position-id-offset path (docs/task-embedding-coverage.md
Phase 1/2).

XLM-R numbers learned positions from padding_idx+1 (posOff=2 for pad_token_id=1),
unlike BERT which starts at 0 — a classic silent-wrong. This certifies aikit's
posOff handling against a real XLM-R forward: it dumps input_ids + last_hidden_state
(no pooling head — we compare the raw transformer output, where a wrong offset
shifts every position embedding and the hidden states diverge).

Tokenization is done here in Python (XLM-R uses a SentencePiece/Unigram tokenizer
aikit doesn't yet implement); the Go test feeds these input_ids to the forward
directly, so this validates the FORWARD + offset, not the tokenizer.

Run from the repo root:
    .venv/bin/python scripts/pin_xlmr.py
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "testdata" / "xlm-roberta-base"
OUT = REPO_ROOT / "testdata" / "xlmr_golden.json"

CASES = [
    "how do i parse json",
    "hello world",
    "x",
    "the quick brown fox",
    "Bonjour le monde",  # multilingual — XLM-R's point
    "識別子を検索する",  # non-Latin
]


def main() -> int:
    import torch
    from transformers import AutoModel, AutoTokenizer

    sys.stderr.write(f"[pin_xlmr] loading {MODEL_DIR} (CPU) ...\n")
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModel.from_pretrained(str(MODEL_DIR)).eval()

    out = {
        "model": "FacebookAI/xlm-roberta-base",
        "hidden": model.config.hidden_size,
        "layers": model.config.num_hidden_layers,
        "pad_token_id": model.config.pad_token_id,
        "pos_offset": model.config.pad_token_id + 1,
        "cases": [],
    }
    with torch.no_grad():
        for text in CASES:
            enc = tok(text, return_tensors="pt")
            ids = enc["input_ids"][0].tolist()
            hid = model(**enc).last_hidden_state[0]  # [L, hidden]
            out["cases"].append({
                "text": text,
                "input_ids": [int(i) for i in ids],
                "L": int(hid.shape[0]),
                "hidden": [round(float(x), 6) for x in hid.reshape(-1).tolist()],
            })

    OUT.write_text(json.dumps(out))
    sys.stderr.write(
        f"[pin_xlmr] wrote {OUT} — {len(CASES)} cases, dim {model.config.hidden_size}, "
        f"pos_offset={out['pos_offset']}, {OUT.stat().st_size // 1024} KB\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
