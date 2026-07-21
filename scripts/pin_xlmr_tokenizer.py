#!/usr/bin/env python3
"""pin_xlmr_tokenizer.py — oracle fixtures for aikit's XLM-R (Unigram/SentencePiece)
tokenizer, the parity target for the pure-Go Unigram port.

Two goldens, both straight from HF `tokenizers` (the exact reference aikit must
reproduce — id-for-id):

  testdata/xlmr_norm_golden.json   — the Precompiled normalizer, probed per input:
      a broad sweep (every codepoint 0..0x2FFFF as a 1-char string, minus
      surrogates) plus combining-mark sequences and real multilingual lines.
      Each entry: {"in": <str>, "out": normalizer.normalize_str(in)}. This is the
      hard gate: the darts-clone charsmap walk must match byte-for-byte.

  testdata/xlmr_encode_golden.json — full text -> input_ids over varied lines
      (Latin, CJK, RTL, punctuation, whitespace, emoji), for the end-to-end gate.

Run from the repo root:
    .venv/bin/python scripts/pin_xlmr_tokenizer.py
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "testdata" / "xlm-roberta-base"
NORM_OUT = REPO_ROOT / "testdata" / "xlmr_norm_golden.json"
ENC_OUT = REPO_ROOT / "testdata" / "xlmr_encode_golden.json"

ENCODE_CASES = [
    "how do i parse json",
    "hello world",
    "x",
    "",
    "   leading and   collapsed   spaces  ",
    "the quick brown fox jumps over the lazy dog",
    "Bonjour le monde, ça va?",
    "Grüße aus München — Straße",  # umlaut + ß + em-dash
    "識別子を検索する",
    "北京欢迎你",
    "Здравствуй, мир",  # Cyrillic
    "مرحبا بالعالم",  # Arabic (RTL)
    "नमस्ते दुनिया",  # Devanagari (combining)
    "café CAFÉ Café",  # accents + case
    "ＦＵＬＬＷＩＤＴＨ ＡＢＣ",  # fullwidth (NFKC-folds)
    "égalité",  # decomposed accents (combining acute)
    "1+1=2  &  a<b>c",  # punctuation/symbols
    "tab\tnewline\nhere",  # control whitespace
    "🚀 rocket 😀 face",  # emoji
    "def add(a, b):\n    return a + b",  # code
    "\U000f0000\U000f0001",  # adjacent plane-15 PUA -> two <unk> that fuse to one
]


def main() -> int:
    from tokenizers import Tokenizer

    tk = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    norm = tk.normalizer

    # ---- normalizer oracle: per-codepoint sweep + curated multi-char cases ----
    # The committed golden is compact: EVERY codepoint whose normalization is
    # non-identity (the cases that actually exercise the charsmap) plus a
    # deterministic 1-in-SAMPLE identity sample (to guard pass-through). The full
    # U+0000..U+2FFFF sweep is validated locally; SAMPLE=1 reproduces it.
    SAMPLE = 41
    cases = []
    for cp in range(0, 0x30000):
        if 0xD800 <= cp <= 0xDFFF:  # surrogates: not scalar values
            continue
        s = chr(cp)
        out = norm.normalize_str(s)
        if s != out or cp % SAMPLE == 0:
            cases.append({"in": s, "out": out})
    # Combining sequences and short graphemes (the <6-byte whole-grapheme path).
    for s in [
        "é", "à", "ö", "ü", "ñ",  # base + combining
        "́", "̈",  # lone combining marks
        "ｶﾞ", "ﾊﾟ",  # halfwidth kana + combining sound marks
        "ﬀ", "ﬁ", "ﬂ",  # ligatures
        "①", "Ⅳ", "㍿", "㎡",  # circled/roman/square compat
        " ", " ", " ", "　",  # exotic whitespace
        "Ａ", "２", "！",  # fullwidth singles
    ]:
        cases.append({"in": s, "out": norm.normalize_str(s)})
    for line in ENCODE_CASES:
        cases.append({"in": line, "out": norm.normalize_str(line)})

    NORM_OUT.write_text(json.dumps({"model": "FacebookAI/xlm-roberta-base",
                                    "cases": cases}))
    sys.stderr.write(f"[pin_xlmr_tok] wrote {NORM_OUT} — {len(cases)} norm cases, "
                     f"{NORM_OUT.stat().st_size // 1024} KB\n")

    # ---- encode oracle: full text -> ids ----
    enc = []
    for line in ENCODE_CASES:
        ids = tk.encode(line).ids
        enc.append({"text": line, "input_ids": [int(i) for i in ids]})
    ENC_OUT.write_text(json.dumps({"model": "FacebookAI/xlm-roberta-base",
                                   "cases": enc}))
    sys.stderr.write(f"[pin_xlmr_tok] wrote {ENC_OUT} — {len(enc)} encode cases\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
