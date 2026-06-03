#!/usr/bin/env python3
"""pin_gemma_window.py — produce the M5 sliding-window parity oracle for the
pure-Go Gemma 3 decoder (docs/milestones/M5-window.md, plan §6 M5/§10).

M3/M4 only exercised short sequences (≤ ~54 positions), far below Gemma's
512-token sliding window, so local layers behaved identically to full causal
attention. M5 needs a prompt **past 512 tokens** so the window genuinely evicts
early keys on the local (sliding_attention) layers while the global
(full_attention) layers still see everything. A divergence between "attend the
last 512" and "attend all" then shows up in the logits — making this a real
test of the mask, not a tautology.

We build a long *varied* prompt (deterministic, seeded) so the evicted early
tokens actually differ from the recent ones, then run HF in float32 and dump
the last-position next-token logits (same compact + full-dump shape as
pin_gemma_forward.py). The Go test reads the token ids from the golden, so the
text generator only has to run here, once.

Output:
  - testdata/gemma_window_golden.json  (committed): ids, token count, argmax,
    top-32, stats, seeded sample, and which layers are local vs global.
  - testdata/gemma_window_full.json     (gitignored): every logit, for exact
    full-vector cosine.

Usage (from repo root):
    .venv/bin/python scripts/pin_gemma_window.py [MODEL_DIR]
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "google/gemma-3-270m"

N_SAMPLE = 256
SAMPLE_SEED = 1234
N_TOPK = 32
TEXT_SEED = 42
TARGET_TOKENS = 700  # comfortably past the 512 window


def build_long_text(tok) -> str:
    """Deterministic varied English, long enough to exceed the window."""
    rng = random.Random(TEXT_SEED)
    subj = ["The engineer", "A curious fox", "Our neighbor", "The old clock",
            "Every student", "The river", "A distant star", "The committee",
            "Her younger brother", "The painter", "A passing storm", "The library"]
    verb = ["measured", "questioned", "repaired", "ignored", "celebrated",
            "sketched", "abandoned", "discovered", "polished", "rewired",
            "translated", "counted"]
    obj = ["the ancient bridge", "three small lanterns", "a forgotten recipe",
           "the quarterly report", "every loose wire", "a field of barley",
           "the broken telescope", "several rival theories", "an empty notebook",
           "the winding staircase", "a handful of coins", "the morning schedule"]
    tail = ["before dawn", "without complaint", "in perfect silence",
            "against all advice", "near the harbor", "for the third time",
            "under heavy rain", "with great care", "by sheer luck",
            "across the valley", "until midnight", "on a whim"]
    parts = []
    i = 0
    while True:
        s = f"{rng.choice(subj)} {rng.choice(verb)} {rng.choice(obj)} {rng.choice(tail)}."
        parts.append(s)
        i += 1
        if i % 8 == 0:
            text = " ".join(parts)
            if len(tok(text, add_special_tokens=True)["input_ids"]) >= TARGET_TOKENS:
                return text


def main() -> int:
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "testdata" / "gemma-3-270m"
    if not (model_dir / "config.json").exists():
        sys.stderr.write(
            f"[pin_gemma_window] missing checkpoint under {model_dir}\n"
            f"  huggingface-cli download {MODEL_ID} --local-dir {model_dir}\n"
        )
        return 1

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.float32)
    model.eval()
    cfg = model.config

    text = build_long_text(tok)
    enc = tok(text, return_tensors="pt")
    ids = enc["input_ids"][0].tolist()
    sys.stderr.write(f"[pin_gemma_window] {len(ids)} tokens (window={cfg.sliding_window})\n")
    assert len(ids) > cfg.sliding_window, "prompt must exceed the sliding window"

    with torch.no_grad():
        out = model(**enc)
    logits = out.logits[0, -1, :].to(torch.float64)
    n = logits.numel()
    fl = logits.tolist()
    argmax = int(torch.argmax(logits).item())

    topv, topi = torch.topk(logits, N_TOPK)
    top_k = [[int(i), float(v)] for i, v in zip(topi.tolist(), topv.tolist())]
    s = float(logits.sum().item())
    ss = float((logits * logits).sum().item())

    g = torch.Generator().manual_seed(SAMPLE_SEED)
    idx = torch.randint(0, n, (N_SAMPLE * 2,), generator=g).tolist()
    seen, sample_idx = set(), []
    for i in idx:
        if i not in seen:
            seen.add(i)
            sample_idx.append(i)
        if len(sample_idx) == N_SAMPLE:
            break
    sample_idx.sort()
    sample = [[i, fl[i]] for i in sample_idx]

    layer_types = list(getattr(cfg, "layer_types", []))

    golden = {
        "model_id": MODEL_ID,
        "note": (
            "M5 sliding-window oracle. HF float32; next-token logits at the last "
            "position of a >512-token prompt, so local layers evict early keys "
            "while global layers attend all. argmax must match; sample/stats to "
            "small tol; full cosine in the gitignored gemma_window_full.json."
        ),
        "dtype": "float32",
        "n_tokens": len(ids),
        "sliding_window": cfg.sliding_window,
        "layer_types": layer_types,
        "ids": ids,
        "argmax": argmax,
        "argmax_token": tok.decode([argmax]),
        "vocab_size": n,
        "stats": {"n": n, "sum": s, "sum_sq": ss, "min": min(fl), "max": max(fl)},
        "top_k": top_k,
        "sample_seed": SAMPLE_SEED,
        "sample": sample,
    }
    gp = REPO_ROOT / "testdata" / "gemma_window_golden.json"
    with gp.open("w") as f:
        json.dump(golden, f, indent=2, allow_nan=False)
        f.write("\n")
    sys.stderr.write(
        f"[pin_gemma_window] wrote {gp.relative_to(REPO_ROOT)} ({gp.stat().st_size/1024:.1f} KB) "
        f"— argmax={argmax} ({tok.decode([argmax])!r})\n"
    )

    fp = REPO_ROOT / "testdata" / "gemma_window_full.json"
    with fp.open("w") as f:
        json.dump({"ids": ids, "argmax": argmax, "logits": fl}, f, allow_nan=False)
        f.write("\n")
    sys.stderr.write(
        f"[pin_gemma_window] wrote {fp.relative_to(REPO_ROOT)} "
        f"({fp.stat().st_size/1024/1024:.1f} MB, gitignored)\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
