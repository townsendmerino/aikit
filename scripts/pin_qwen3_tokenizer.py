#!/usr/bin/env python3
"""pin_qwen3_tokenizer.py — produce testdata/qwen3_tokenizer_golden.json from
the real Qwen/Qwen3-1.7B tokenizer.

Milestone G3 oracle for the pure-Go *byte-level* BPE tokenizer (the Qwen /
Llama-3 family; see docs/multi-model-plan.md §5.2 and
docs/milestones/G3-tokenizer.md). Sibling of scripts/pin_gemma_tokenizer.py
(M2, byte-fallback SentencePiece-style): a pinned dump the Go tokenizer must
reproduce id-for-id, so a one-token drift fails loudly in CI rather than
silently degrading generation.

Qwen3 ships an HF `tokenizers` byte-level BPE (151k vocab, ~151k merges) as
testdata/qwen3-1.7b/tokenizer.json. The pipeline differs from Gemma's on every
stage except the merge table: NFC normalize, a GPT-2 split regex pretokenizer,
a byte->unicode map (spaces become Ġ, newline Ċ, tab ĉ), `ignore_merges=true`,
and NO byte-fallback. It also adds NO special tokens at encode time (the
post_processor is plain ByteLevel; bos_token is null, add_bos_token=false), so
`ids_bos == ids` — the chat markers (<|im_start|>/<|im_end|>) come from the
chat template, not encode().

For a fixed prompt set chosen to exercise every edge of the pipeline we record:

  - ids with add_special_tokens=True (== ids for Qwen, kept for symmetry),
  - ids without special tokens,
  - the token *strings* (so a mismatch is human-readable),
  - decode(ids) with skip_special_tokens=False (the round-trip Go Decode matches).

Usage (from repo root):
    .venv/bin/python scripts/pin_qwen3_tokenizer.py [MODEL_DIR]
    # MODEL_DIR defaults to testdata/qwen3-1.7b
    # writes testdata/qwen3_tokenizer_golden.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "Qwen/Qwen3-1.7B"

# Prompts chosen to exercise every branch of the byte-level pipeline. Order is
# stable so the golden diff is readable when a prompt is added.
PROMPTS = [
    "Hello world",
    "Hello",
    " Hello",                       # leading space -> Ġ prefix attaches to word
    "Hello world ",                 # single trailing space
    "  two  spaces",                # space runs: leading Ġ then Ġword
    "trailing   ",                  # 3 trailing spaces -> one ĠĠĠ token
    "The quick brown fox jumps over the lazy dog.",
    "café 中文 — naïve façade",     # non-ASCII via byte-level (multi-byte UTF-8)
    "𝕳ello",                        # U+1D573 -> 4 byte-level chars, no byte-fallback
    "emoji 🦄 and 🏳️‍🌈",      # ZWJ emoji sequence, all byte-level
    "a\tb\nc\n\nd",                 # tab (ĉ prefix) vs newline runs (Ċ / ĊĊ)
    "newline\n\n\nthree",           # \n\n\n -> ĊĊĊ
    "don't can't I'LL we've",       # case-insensitive contraction split
    "func main() { fmt.Println(\"hi\") }",   # code: punctuation runs
    "Number 1234 and 56",           # digits split one-per-token
    "<|im_start|>user\nHi<|im_end|>",        # literal special tokens in text
    "<|endoftext|>",                # lone special token
    "x:=3; y==4 && z!=5",           # operator runs
    "Mixed CASE and    irregular   spacing",
    "",                             # empty -> []
]


def main() -> int:
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "testdata" / "qwen3-1.7b"
    tok_path = model_dir / "tokenizer.json"
    out_path = REPO_ROOT / "testdata" / "qwen3_tokenizer_golden.json"

    if not tok_path.exists():
        sys.stderr.write(
            f"[pin_qwen3_tokenizer] missing {tok_path}\n"
            f"  hf download {MODEL_ID} --local-dir {model_dir}\n"
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
            "G3 byte-level BPE tokenizer oracle. Qwen adds no special tokens at "
            "encode (post_processor is plain ByteLevel, bos null) so ids_bos==ids. "
            "The Go tokenizer must reproduce ids id-for-id and Decode(ids) must "
            "equal `decode`. NFC normalize, GPT-2 split regex, byte->unicode map, "
            "ignore_merges=true, no byte-fallback."
        ),
        "special_tokens": {
            "eos_im_end": sid("<|im_end|>"),
            "endoftext": sid("<|endoftext|>"),
            "im_start": sid("<|im_start|>"),
        },
        "cases": cases,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    sys.stderr.write(
        f"[pin_qwen3_tokenizer] wrote {out_path.relative_to(REPO_ROOT)} "
        f"({out_path.stat().st_size/1024:.1f} KB)\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
