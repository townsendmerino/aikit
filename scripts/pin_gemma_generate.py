#!/usr/bin/env python3
"""pin_gemma_generate.py — produce the M4 greedy-continuation oracle for the
pure-Go Gemma 3 decoder (docs/milestones/M4-decode.md, plan §6 M4/§10).

M3 proved one bit-faithful logit vector; M4 proves the KV-cache decode loop:
position advance, K/V append, and causal attention over the growing cache stay
correct across many steps. The oracle is HF's **greedy** continuation of a
fixed prompt for a fixed number of new tokens.

Determinism: we force exactly N_NEW tokens (min_new_tokens = max_new_tokens) so
EOS never truncates the sequence — the Go test reproduces the same N ids and
the same decoded string. Run in float32 to match the Go f32 forward (M3).

Output (committed, small): testdata/gemma_generate_golden.json with the prompt,
its ids, the N greedy continuation ids, and the decoded continuation text.

Usage (from repo root):
    .venv/bin/python scripts/pin_gemma_generate.py [MODEL_DIR]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "google/gemma-3-270m"

PROMPT = "The capital of France is"
N_NEW = 48


def main() -> int:
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "testdata" / "gemma-3-270m"
    if not (model_dir / "config.json").exists():
        sys.stderr.write(
            f"[pin_gemma_generate] missing checkpoint under {model_dir}\n"
            f"  huggingface-cli download {MODEL_ID} --local-dir {model_dir}\n"
        )
        return 1

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.float32)
    model.eval()

    enc = tok(PROMPT, return_tensors="pt")
    prompt_ids = enc["input_ids"][0].tolist()

    with torch.no_grad():
        out = model.generate(
            **enc,
            do_sample=False,            # greedy / argmax
            num_beams=1,
            max_new_tokens=N_NEW,
            min_new_tokens=N_NEW,       # force exactly N (suppress EOS truncation)
        )
    full = out[0].tolist()
    cont_ids = full[len(prompt_ids):]
    assert len(cont_ids) == N_NEW, f"got {len(cont_ids)} new tokens, want {N_NEW}"
    cont_text = tok.decode(cont_ids, skip_special_tokens=False)

    sys.stderr.write(f"[pin_gemma_generate] prompt={PROMPT!r}\n")
    sys.stderr.write(f"[pin_gemma_generate] continuation={cont_text!r}\n")

    golden = {
        "model_id": MODEL_ID,
        "note": (
            "M4 greedy-decode oracle. HF float32, greedy (do_sample=False), "
            "exactly N forced new tokens (min=max, EOS suppressed). The Go decode "
            "loop must reproduce continuation_ids id-for-id and Decode() must "
            "reproduce continuation_text."
        ),
        "dtype": "float32",
        "prompt": PROMPT,
        "prompt_ids": prompt_ids,
        "n_new": N_NEW,
        "continuation_ids": cont_ids,
        "continuation_text": cont_text,
    }
    gp = REPO_ROOT / "testdata" / "gemma_generate_golden.json"
    with gp.open("w") as f:
        json.dump(golden, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    sys.stderr.write(
        f"[pin_gemma_generate] wrote {gp.relative_to(REPO_ROOT)} ({gp.stat().st_size/1024:.1f} KB)\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
