#!/usr/bin/env python3
"""pin_llama_forward.py — forward oracle for the pure-Go decoder running a Llama
dense checkpoint (multi-model-plan: the llama adapter). Mirrors
pin_qwen3_forward.py.

Validates the llama adapter (RMS no-offset, Pre2, SwiGLU, single-base RoPE,
NO QK-norm, untied head, derived head_dim) against HF. Runs TinyLlama-1.1B —
a Llama-2-architecture model small enough to run in float32 here — over a fixed
prompt and records the next-token logits at the last position. The golden also
records the HF token ids so the Go forward-parity test feeds identical ids and
isolates the forward pass from the tokenizer.

Outputs:
  - testdata/llama_forward_golden.json (committed): ids, argmax, top-32, stats,
    a seeded 256-index sample.
  - testdata/llama_forward_full.json (gitignored): every logit, for exact cosine.

Usage:  .venv/bin/python scripts/pin_llama_forward.py [MODEL_DIR]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# PROMPT override via env lets the Mistral check use a >sliding_window prompt so
# the window is actually exercised (TinyMistral's window is 32 tokens).
PROMPT = os.environ.get("PIN_PROMPT", "The capital of France is")
N_SAMPLE, SAMPLE_SEED, N_TOPK = 256, 1234, 32


def main() -> int:
    # argv: [MODEL_DIR] [GOLDEN_PREFIX]. Default is the TinyLlama (Llama-2)
    # adapter check; pass e.g. `testdata/llama3.2-1b llama32` for the llama3
    # rope-scaling check.
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "testdata" / "tinyllama-1.1b"
    prefix = sys.argv[2] if len(sys.argv) > 2 else "llama"
    if not (model_dir / "config.json").exists():
        sys.stderr.write(f"[pin_llama_forward] missing checkpoint under {model_dir}\n")
        return 1
    import json as _json
    MODEL_ID = _json.load(open(model_dir / "config.json")).get("_name_or_path", str(model_dir))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.float32)
    model.eval()

    enc = tok(PROMPT, return_tensors="pt")  # Llama default: prepends <s> (BOS)
    ids = enc["input_ids"][0].tolist()
    sys.stderr.write(f"[pin_llama_forward] prompt={PROMPT!r} ids={ids}\n")

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

    golden = {
        "model_id": MODEL_ID,
        "note": f"forward oracle for {MODEL_ID}. HF float32; next-token logits at "
                "the last position. ids are HF token ids (the Go test is "
                "tokenizer-independent). argmax must match; top_k/sample to small "
                f"tol; full cosine in the gitignored {prefix}_forward_full.json.",
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
    gp = REPO_ROOT / "testdata" / f"{prefix}_forward_golden.json"
    gp.write_text(json.dumps(golden, indent=2, allow_nan=False) + "\n")
    sys.stderr.write(f"[pin_llama_forward] wrote {gp.relative_to(REPO_ROOT)} "
                     f"({gp.stat().st_size/1024:.1f} KB) — argmax={argmax} ({tok.decode([argmax])!r})\n")

    fp = REPO_ROOT / "testdata" / f"{prefix}_forward_full.json"
    fp.write_text(json.dumps({"ids": ids, "argmax": argmax, "logits": fl}, allow_nan=False) + "\n")
    sys.stderr.write(f"[pin_llama_forward] wrote {fp.relative_to(REPO_ROOT)} "
                     f"({fp.stat().st_size/1024/1024:.1f} MB, gitignored)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
