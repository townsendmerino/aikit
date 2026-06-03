#!/usr/bin/env python3
"""pin_gemma_forward.py — produce the M3 forward-pass oracle for the pure-Go
Gemma 3 decoder (docs/milestones/M3-forward.md, plan §6 M3/§10).

Runs the real google/gemma-3-270m in **float32** (so the reference math is f32
end-to-end and the Go f32 forward can match it without bf16 rounding noise),
feeds a fixed BOS-prefixed prompt, and records the next-token logit vector at
the last position — the deterministic "first decode step" target.

Two outputs:
  - testdata/gemma_forward_golden.json  (committed, ~compact): the prompt + ids,
    argmax, top-32 (id, logit), full-vector stats (n/sum/sum_sq/min/max), and a
    seeded sample of 256 (index, value) pairs. A strong, durable CI gate.
  - testdata/gemma_forward_full.json     (gitignored, ~per-machine): every logit,
    so the Go test can compute an exact full-vector cosine when present.

The Go forward must reproduce the committed checks (argmax identical, top-k and
sampled values to a small tolerance, stats close) and, when the full dump is
present, cosine ≥ 1 − 1e-4 with an identical argmax.

Usage (from repo root):
    .venv/bin/python scripts/pin_gemma_forward.py [MODEL_DIR]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "google/gemma-3-270m"

PROMPT = "The capital of France is"
N_SAMPLE = 256
SAMPLE_SEED = 1234
N_TOPK = 32


def main() -> int:
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "testdata" / "gemma-3-270m"
    if not (model_dir / "config.json").exists():
        sys.stderr.write(
            f"[pin_gemma_forward] missing checkpoint under {model_dir}\n"
            f"  huggingface-cli download {MODEL_ID} --local-dir {model_dir}\n"
        )
        return 1

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype=torch.float32)
    model.eval()

    enc = tok(PROMPT, return_tensors="pt")  # add_special_tokens=True → BOS prepended
    ids = enc["input_ids"][0].tolist()
    sys.stderr.write(f"[pin_gemma_forward] prompt={PROMPT!r} ids={ids}\n")

    with torch.no_grad():
        out = model(**enc)
    logits = out.logits[0, -1, :].to(torch.float64)  # last position, next-token

    n = logits.numel()
    fl = logits.tolist()
    argmax = int(torch.argmax(logits).item())

    topv, topi = torch.topk(logits, N_TOPK)
    top_k = [[int(i), float(v)] for i, v in zip(topi.tolist(), topv.tolist())]

    s = float(logits.sum().item())
    ss = float((logits * logits).sum().item())

    # Seeded deterministic sample of indices (sorted, unique).
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

    golden = {
        "model_id": MODEL_ID,
        "note": (
            "M3 forward oracle. HF run in float32; next-token logits at the last "
            "position of a BOS-prefixed prompt. argmax must match; top_k/sample "
            "to small tol; stats close. Full vector (cosine) in the gitignored "
            "gemma_forward_full.json when regenerated."
        ),
        "dtype": "float32",
        "prompt": PROMPT,
        "ids": ids,
        "argmax": argmax,
        "argmax_token": tok.decode([argmax]),
        "vocab_size": n,
        "stats": {"n": n, "sum": s, "sum_sq": ss, "min": min(fl), "max": max(fl)},
        "top_k": top_k,
        "sample_seed": SAMPLE_SEED,
        "sample": sample,
    }
    gp = REPO_ROOT / "testdata" / "gemma_forward_golden.json"
    with gp.open("w") as f:
        json.dump(golden, f, indent=2, allow_nan=False)
        f.write("\n")
    sys.stderr.write(
        f"[pin_gemma_forward] wrote {gp.relative_to(REPO_ROOT)} "
        f"({gp.stat().st_size/1024:.1f} KB) — argmax={argmax} ({tok.decode([argmax])!r})\n"
    )

    # Full per-machine dump for exact cosine.
    fp = REPO_ROOT / "testdata" / "gemma_forward_full.json"
    with fp.open("w") as f:
        json.dump({"ids": ids, "argmax": argmax, "logits": fl}, f, allow_nan=False)
        f.write("\n")
    sys.stderr.write(
        f"[pin_gemma_forward] wrote {fp.relative_to(REPO_ROOT)} "
        f"({fp.stat().st_size/1024/1024:.1f} MB, gitignored)\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
