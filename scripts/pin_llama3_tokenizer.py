#!/usr/bin/env python3
"""pin_llama3_tokenizer.py — produce testdata/llama3_tokenizer_golden.json from
the Llama-3 tokenizer.

Second G3 byte-level oracle, alongside scripts/pin_qwen3_tokenizer.py. Llama-3
shares the byte-level BPE machinery but flips two pipeline knobs the Go
tokenizer reads from tokenizer.json:

  - digit-run cap: Llama-3 groups digits in runs of up to 3 (`\\p{N}{1,3}`)
    where Qwen takes one at a time (`\\p{N}`);
  - normalizer: Llama-3 has NONE (null) where Qwen normalizes NFC;

and, unlike Qwen, its post-processor DOES prepend a BOS (`<|begin_of_text|>`),
so `ids_bos` (add_special_tokens=True) differs from `ids`.

The tokenizer is gated under meta-llama, but non-gated mirrors (e.g.
NousResearch/Meta-Llama-3-8B) ship the byte-identical tokenizer.json. Point
MODEL_DIR at a dir holding tokenizer.json + tokenizer_config.json.

Usage (from repo root):
    .venv/bin/python scripts/pin_llama3_tokenizer.py [MODEL_DIR]
    # MODEL_DIR defaults to testdata/llama3-tokenizer
    # writes testdata/llama3_tokenizer_golden.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "meta-llama/Meta-Llama-3-8B (via non-gated mirror)"

# Prompts emphasize the Llama-3 deltas: multi-digit grouping, no NFC, BOS added.
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
    "Number 1234567 and 56 and 8",   # 3-digit grouping: 123|456|7
    "year 2024, pi 3.14159, 1000000",
    "x:=3; y==4 && z!=5",
    "<|begin_of_text|>hi<|end_of_text|>",  # literal special tokens
    "<|eot_id|>",
    "Mixed CASE and    irregular   spacing",
    "",
]


def main() -> int:
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "testdata" / "llama3-tokenizer"
    tok_path = model_dir / "tokenizer.json"
    out_path = REPO_ROOT / "testdata" / "llama3_tokenizer_golden.json"

    if not tok_path.exists():
        sys.stderr.write(
            f"[pin_llama3_tokenizer] missing {tok_path}\n"
            f"  hf download NousResearch/Meta-Llama-3-8B tokenizer.json tokenizer_config.json --local-dir {model_dir}\n"
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
            "G3 byte-level BPE oracle (Llama-3). Same byte-level core as Qwen but "
            "digit-run cap 3 (\\p{N}{1,3}) and NO normalizer; post_processor "
            "prepends <|begin_of_text|> so ids_bos != ids. The Go tokenizer must "
            "reproduce ids id-for-id (and ids_bos with BOS) and Decode(ids) must "
            "equal `decode`."
        ),
        "special_tokens": {
            "bos_begin_of_text": sid("<|begin_of_text|>"),
            "eos_end_of_text": sid("<|end_of_text|>"),
            "eot_id": sid("<|eot_id|>"),
        },
        "cases": cases,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    sys.stderr.write(
        f"[pin_llama3_tokenizer] wrote {out_path.relative_to(REPO_ROOT)} "
        f"({out_path.stat().st_size/1024:.1f} KB)\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
