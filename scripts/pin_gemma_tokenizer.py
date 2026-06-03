#!/usr/bin/env python3
"""pin_gemma_tokenizer.py — produce testdata/gemma_tokenizer_golden.json from
the real google/gemma-3-270m tokenizer.

Milestone M2 oracle for the pure-Go SentencePiece/BPE tokenizer (see
docs/gemma-decoder-plan.md §9 and docs/milestones/M2-tokenizer.md). Mirrors
scripts/pin_gemma.py: a pinned dump the Go tokenizer must reproduce id-for-id,
so a one-token drift fails loudly in CI rather than silently degrading
generation.

Gemma 3 ships a byte-fallback **BPE** tokenizer (262k vocab, ~515k merges) as
testdata/gemma-3-270m/tokenizer.json (HF `tokenizers` format). We load it with
the `tokenizers` library (Rust-backed, no torch needed) and, for a fixed set of
prompts chosen to exercise every edge of the pipeline, record:

  - ids with BOS (encode default: post_processor prepends <bos>),
  - ids without BOS (add_special_tokens=False),
  - the no-BOS token *strings* (so a mismatch is human-readable),
  - decode(ids_with_bos) rendering special tokens (skip_special_tokens=False),
  - decode(ids_no_bos) (the clean round-trip the Go Decode must match).

The prompt set deliberately covers: plain ASCII, leading/trailing/multiple
spaces (▁ runs), in-vocab multibyte unicode, byte-fallback (chars absent from
the vocab → <0xNN> UTF-8 bytes), tabs/newlines (added-token vs BPE split),
literal special/control tokens in text (added-vocabulary split), the chat
template markers, code, numbers, and the empty string.

Usage (from repo root):
    .venv/bin/python scripts/pin_gemma_tokenizer.py [MODEL_DIR]
    # MODEL_DIR defaults to testdata/gemma-3-270m
    # writes testdata/gemma_tokenizer_golden.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "google/gemma-3-270m"

# Prompts chosen to exercise every branch of the tokenizer pipeline. Order is
# stable so the golden diff is readable when a prompt is added.
PROMPTS = [
    "Hello world",
    "Hello",
    " Hello",                     # leading space -> ▁Hello (distinct id)
    "Hello world ",               # trailing space
    "  two  spaces",              # ▁ runs
    "The quick brown fox jumps over the lazy dog.",
    "café 中文 — naïve façade",   # in-vocab multibyte + punctuation
    "𝕳ello",                      # U+1D573 not in vocab -> byte fallback
    "emoji 🦄 and 🏳️‍🌈",    # one in-vocab, one ZWJ sequence (byte fallback parts)
    "a\tb\nc\n\nd",               # tab (BPE) vs newline runs (added tokens)
    "<bos>hi<eos>",               # literal special tokens in text
    "<start_of_turn>user\nHi<end_of_turn>",  # chat markers
    "func main() { fmt.Println(\"hi\") }",   # code
    "1234567890 +-*/= 3.14159",   # digits + symbols
    "Mixed CASE and    irregular   spacing",
    "",                           # empty -> [] (no-bos) / [bos] (with-bos)
]


def main() -> int:
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "testdata" / "gemma-3-270m"
    tok_path = model_dir / "tokenizer.json"
    out_path = REPO_ROOT / "testdata" / "gemma_tokenizer_golden.json"

    if not tok_path.exists():
        sys.stderr.write(
            f"[pin_gemma_tokenizer] missing {tok_path}\n"
            f"  huggingface-cli download {MODEL_ID} --local-dir {model_dir}\n"
        )
        return 1

    from tokenizers import Tokenizer

    tk = Tokenizer.from_file(str(tok_path))
    vocab = tk.get_vocab()

    def sid(piece: str) -> int:
        return vocab[piece]

    cases = []
    for p in PROMPTS:
        with_bos = tk.encode(p, add_special_tokens=True)
        no_bos = tk.encode(p, add_special_tokens=False)
        cases.append({
            "text": p,
            "ids_bos": with_bos.ids,
            "ids": no_bos.ids,
            "tokens": no_bos.tokens,
            "decode_bos": tk.decode(with_bos.ids, skip_special_tokens=False),
            "decode": tk.decode(no_bos.ids, skip_special_tokens=False),
        })
        sys.stderr.write(f"  {p!r:<46} -> {len(no_bos.ids)} ids\n")

    payload = {
        "model_id": MODEL_ID,
        "note": (
            "M2 tokenizer oracle. ids_bos is encode() with the post_processor "
            "(<bos> prepended); ids is add_special_tokens=False. The Go "
            "tokenizer must reproduce both id-for-id, and Decode(ids) must "
            "equal `decode`. Byte-fallback BPE, normalize space->U+2581."
        ),
        "special_tokens": {
            "bos": sid("<bos>"), "eos": sid("<eos>"), "pad": sid("<pad>"),
            "unk": sid("<unk>"),
            "start_of_turn": sid("<start_of_turn>"),
            "end_of_turn": sid("<end_of_turn>"),
        },
        "cases": cases,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    sys.stderr.write(
        f"[pin_gemma_tokenizer] wrote {out_path.relative_to(REPO_ROOT)} "
        f"({out_path.stat().st_size/1024:.1f} KB)\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
