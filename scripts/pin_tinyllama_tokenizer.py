#!/usr/bin/env python3
"""pin_tinyllama_tokenizer.py — produce testdata/tinyllama_tokenizer_golden.json
from the TinyLlama (Llama-2 SentencePiece) tokenizer.

This is the SPM/byte-fallback oracle for the GGUF tokenizer work: TinyLlama is
the model we already ship as a quantized GGUF (testdata/tinyllama-gguf), and its
tokenizer.ggml.model is "llama" — the SentencePiece-style byte-fallback BPE.

Unlike the byte-level families (Qwen/Llama-3), the Llama-2 normalizer PREPENDS a
"▁" (the SentencePiece dummy prefix) before replacing spaces, so "Hello" tokenizes
as "▁Hello". The Go tokenizer's GGUF path must reproduce these ids id-for-id from
the GGUF metadata alone (tokens + merges + special-token ids), and Decode(ids)
must equal HF's rendering.

The same golden validates both the bare tokenizer.json load (tokenizer.Load) and
the bare-GGUF load (tokenizer.LoadGGUF) — they must agree with each other and HF.

Usage (from repo root):
    .venv/bin/python scripts/pin_tinyllama_tokenizer.py [MODEL_DIR]
    # MODEL_DIR defaults to testdata/tinyllama-1.1b
    # writes testdata/tinyllama_tokenizer_golden.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0 (Llama-2 SentencePiece tokenizer)"

# Prompts emphasize the SPM deltas: the ▁ dummy prefix, leading/trailing spaces,
# byte-fallback for out-of-vocab runes (emoji, rare CJK), digits, code, and the
# control tokens written literally.
PROMPTS = [
    "Hello world",
    " Hello",
    "  two  spaces",
    "trailing   ",
    "The quick brown fox jumps over the lazy dog.",
    "café 中文 — naïve façade",
    "𝕳ello",
    "emoji 🦄 and 🏳️‍🌈",
    "a\tb\nc\n\nd",
    "newline\n\n\nthree",
    "don't can't I'LL we've",
    "func main() { fmt.Println(\"hi\") }",
    "Number 1234567 and 56 and 8",
    "year 2024, pi 3.14159, 1000000",
    "x:=3; y==4 && z!=5",
    "<s>hi</s>",
    "<unk>",
    "Mixed CASE and    irregular   spacing",
    "",
]


def main() -> int:
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "testdata" / "tinyllama-1.1b"
    tok_path = model_dir / "tokenizer.json"
    out_path = REPO_ROOT / "testdata" / "tinyllama_tokenizer_golden.json"

    if not tok_path.exists():
        sys.stderr.write(
            f"[pin_tinyllama_tokenizer] missing {tok_path}\n"
            f"  hf download TinyLlama/TinyLlama-1.1B-Chat-v1.0 tokenizer.json tokenizer_config.json --local-dir {model_dir}\n"
        )
        return 1

    from tokenizers import Tokenizer

    tk = Tokenizer.from_file(str(tok_path))
    vocab = tk.get_vocab()

    def sid(piece: str):
        return vocab.get(piece, -1)

    cases = []
    for p in PROMPTS:
        with_special = tk.encode(p, add_special_tokens=True)
        no_special = tk.encode(p, add_special_tokens=False)
        cases.append({
            "text": p,
            "ids_bos": with_special.ids,
            "ids": no_special.ids,
            "tokens": no_special.tokens,
            "decode_bos": tk.decode(with_special.ids, skip_special_tokens=False),
            "decode": tk.decode(no_special.ids, skip_special_tokens=False),
        })
        sys.stderr.write(f"  {p!r:<46} -> {len(no_special.ids)} ids\n")

    payload = {
        "model_id": MODEL_ID,
        "note": (
            "SPM/byte-fallback oracle (Llama-2, TinyLlama). The normalizer "
            "prepends ▁ (SentencePiece dummy prefix) then replaces spaces with ▁; "
            "BPE is merge-rank with <0xNN> byte fallback. The Go GGUF tokenizer "
            "(tokenizer.LoadGGUF, built from the bare .gguf metadata) must "
            "reproduce ids id-for-id, and Decode(ids) must equal `decode`. "
            "(tokenizer.Load from tokenizer.json is Gemma-specific for this "
            "family, so it is not the path under test here.)"
        ),
        "special_tokens": {
            "bos": sid("<s>"),
            "eos": sid("</s>"),
            "unk": sid("<unk>"),
        },
        "cases": cases,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    sys.stderr.write(
        f"[pin_tinyllama_tokenizer] wrote {out_path.relative_to(REPO_ROOT)} "
        f"({out_path.stat().st_size/1024:.1f} KB)\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
