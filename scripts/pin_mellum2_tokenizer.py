#!/usr/bin/env python3
"""pin_mellum2_tokenizer.py — produce testdata/mellum2_tokenizer_golden.json from
the real JetBrains/Mellum2-12B-A2.5B-Instruct tokenizer.

Oracle for the byte-level BPE tokenizer's Mellum2 pipeline (the "Mellum2 polish"
follow-up; see docs/todo.md). Sibling of pin_qwen3_tokenizer.py / pin_llama3_
tokenizer.py: a pinned dump the pure-Go tokenizer must reproduce id-for-id.

Mellum2's byte-level pipeline differs from Qwen/Llama-3 in one decisive way: its
pre_tokenizer is a Sequence whose FIRST stage is Digits{individual_digits:true},
running *before* the ByteLevel split. That isolates every digit into its own
pretoken, so a leading space never attaches to a digit — " 1" tokenizes as the
two pieces "Ġ" + "1", not the single piece "Ġ1" the bare GPT-2 regex would make.
There is no normalizer, ignore_merges is false, and the post_processor is plain
ByteLevel (bos_token <|endoftext|> is defined but not auto-prepended), so HF adds
no special tokens at encode time — ids_bos == ids. The Go tokenizer is validated
on the no-special `ids` column (the HF-faithful signal) plus the decode round-trip.

The digit-isolation prompts below are the ones that diverge from the GPT-2
default; keep them when extending the set so a regression fails loudly.

Usage (from repo root):
    .venv/bin/python scripts/pin_mellum2_tokenizer.py [MODEL_DIR]
    # MODEL_DIR defaults to testdata/mellum2-tokenizer
    # writes testdata/mellum2_tokenizer_golden.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "JetBrains/Mellum2-12B-A2.5B-Instruct"

# Prompts chosen to exercise the Digits{individual_digits} pre-split (spaces and
# letters adjacent to digit runs) on top of the usual byte-level edges.
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
    "don't can't I'LL we've",
    # Digit-isolation cases — where Mellum2 differs from the GPT-2 default:
    "Number 1234567 and 56 and 8",          # space + digit run -> Ġ then digits
    "year 2024, pi 3.14159, 1000000",       # digits mid-text and after space
    "x = 10; y = 200;",                      # code: space-equals-space-digits
    "v1.2.3-rc4 build 42",                   # version strings
    "id42name 7x 0b1010 0xFF",               # digits glued to letters
    "  3 spaces then 3",                     # leading spaces + lone digit
    "func main() { fmt.Println(\"hi\") }",   # code punctuation runs
    "<commit_before>diff --git a/1 b/2",     # special token + digits
    "<|endoftext|>",
    "",
]


def main() -> int:
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "testdata" / "mellum2-tokenizer"
    tok_path = model_dir / "tokenizer.json"
    out_path = REPO_ROOT / "testdata" / "mellum2_tokenizer_golden.json"

    if not tok_path.exists():
        sys.stderr.write(
            f"[pin_mellum2_tokenizer] missing {tok_path}\n"
            f"  hf download {MODEL_ID} tokenizer.json tokenizer_config.json --local-dir {model_dir}\n"
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
            "Byte-level BPE tokenizer oracle for Mellum2. The pre_tokenizer is "
            "Sequence[Digits{individual_digits:true}, ByteLevel], so each digit is "
            "isolated before the GPT-2 split — a leading space never attaches to a "
            "digit (' 1' -> 'Ġ' + '1'). No normalizer, ignore_merges=false. The "
            "post_processor is plain ByteLevel and add_bos_token is unset, so HF "
            "adds no special tokens (ids_bos == ids); the Go tokenizer is validated "
            "on the no-special `ids` column and Decode(ids) == `decode`."
        ),
        "special_tokens": {
            "endoftext": sid("<|endoftext|>"),
            "im_start": sid("<|im_start|>"),
            "im_end": sid("<|im_end|>"),
        },
        "cases": cases,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    sys.stderr.write(
        f"[pin_mellum2_tokenizer] wrote {out_path.relative_to(REPO_ROOT)} "
        f"({out_path.stat().st_size/1024:.1f} KB)\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
