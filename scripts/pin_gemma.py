#!/usr/bin/env python3
"""pin_gemma.py — produce testdata/gemma_golden.json from a real
google/gemma-3-270m checkpoint.

This is the Milestone M1 reference oracle for the Gemma 3 decoder loader
(see docs/milestones/M1-loader.md and docs/gemma-decoder-plan.md §6). It
mirrors scripts/pin_encoder.py: a pinned Python dump the pure-Go loader
must reproduce, so a transposed/mis-decoded tensor fails loudly in CI
rather than silently poisoning every later milestone.

What it records:
  - the full parsed config.json,
  - for a fixed handful of tensors (model.embed_tokens.weight,
    model.layers.0.self_attn.q_proj.weight, model.norm.weight): the
    safetensors shape + dtype, and a float64 checksum (sum and
    sum-of-squares) computed AFTER widening bf16 -> f32.

Why "after widening" matters: Gemma ships bf16. bf16 -> f32 is the exact
top-16-bits widen (embed.Tensor.BFloat16sToF32), so both sides see
identical f32 values; the only divergence is float64 summation ORDER
(torch's vectorized reduction vs Go's sequential loop), which stays far
below the test's 1e-6 relative bar.

Checksums are read straight from the on-disk model.safetensors (via
safe_open, one tensor at a time — embed_tokens alone is ~340 MB bf16, so
we never materialize the whole model) so they pin the exact bytes the Go
test's embed.OpenSafetensorsMmap will read.

Non-finite sums sanitize to JSON null (a real checkpoint should never hit
this; allow_nan=False below fails loudly if one slips through), the same
gotcha pin_encoder.py / pin_inference.py document.

Usage (from repo root):
    .venv/bin/python scripts/pin_gemma.py [MODEL_DIR]
    # MODEL_DIR defaults to testdata/gemma-3-270m
    # get it with:
    #   huggingface-cli download google/gemma-3-270m \\
    #       --local-dir testdata/gemma-3-270m
    # writes testdata/gemma_golden.json directly.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

MODEL_ID = "google/gemma-3-270m"

# The tensors the Go loader's weights_test.go cross-checks. One large
# bf16 matrix (the tied embedding — ~170M of the 270M params), one small
# attention projection, one 1-D norm vector: enough to catch a transposed
# matrix, a wrong dtype path, or a byte-order slip.
SAMPLED_TENSORS = [
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.norm.weight",
]


def read_safetensors_header(path: Path) -> dict:
    """Parse the safetensors header (exact stored shape + dtype) without a
    framework dependency: first 8 bytes are a little-endian uint64 header
    length, followed by that many JSON bytes."""
    with path.open("rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    return header


def checksum_f64(t) -> tuple[float, float, int]:
    """float64 (sum, sum_of_squares, n_elements) of a torch tensor, widened
    to f32 first (exact for bf16) then to f64 — matching the Go side's
    `sum += float64(v); sumSq += float64(v)*float64(v)` over the f32 values."""
    import torch

    f = t.to(torch.float32).reshape(-1).to(torch.float64)
    s = float(f.sum().item())
    ss = float((f * f).sum().item())
    return s, ss, f.numel()


def sanitize(x: float):
    return None if not math.isfinite(x) else x


def main() -> int:
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "testdata" / "gemma-3-270m"
    config_path = model_dir / "config.json"
    weights_path = model_dir / "model.safetensors"
    out_path = REPO_ROOT / "testdata" / "gemma_golden.json"

    # Dependency-free existence check first, so a missing checkpoint prints
    # clean guidance even without torch/safetensors installed.
    if not config_path.exists() or not weights_path.exists():
        sys.stderr.write(
            f"[pin_gemma] missing checkpoint under {model_dir} "
            f"(need config.json + model.safetensors).\n"
            f"  huggingface-cli download {MODEL_ID} --local-dir {model_dir}\n"
        )
        return 1

    from safetensors import safe_open

    sys.stderr.write(f"[pin_gemma] reading {model_dir} ...\n")
    sys.stderr.flush()

    config = json.loads(config_path.read_text())
    header = read_safetensors_header(weights_path)

    tensors_out: dict[str, dict] = {}
    with safe_open(str(weights_path), framework="pt") as f:
        for name in SAMPLED_TENSORS:
            if name not in header:
                sys.stderr.write(f"[pin_gemma] ERROR: tensor {name!r} not in checkpoint header\n")
                return 1
            meta = header[name]
            t = f.get_tensor(name)
            s, ss, n = checksum_f64(t)
            tensors_out[name] = {
                "shape": meta["shape"],
                "dtype": meta["dtype"],          # e.g. "BF16" — the Go side asserts this path
                "n": n,
                "sum": sanitize(s),
                "sum_sq": sanitize(ss),
            }
            sys.stderr.write(
                f"  {name:<48} {meta['dtype']:<5} shape={meta['shape']} "
                f"sum={s:.6e} sum_sq={ss:.6e}\n"
            )
            sys.stderr.flush()

    payload = {
        "model_id": MODEL_ID,
        "note": (
            "M1 loader oracle. Per-tensor shape/dtype from the safetensors "
            "header; sum and sum_sq are float64 reductions over the f32 "
            "values AFTER bf16->f32 widening (exact). The Go loader must "
            "reproduce shape exactly and the checksums to <=1e-6 relative."
        ),
        "checksum_method": "sum += f64(v); sum_sq += f64(v)*f64(v)  over widened-f32 values",
        "sampled_tensors": SAMPLED_TENSORS,
        "config": config,
        "tensors": tensors_out,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.write("\n")
    sys.stderr.write(
        f"[pin_gemma] wrote {out_path.relative_to(REPO_ROOT)} "
        f"({out_path.stat().st_size/1024:.1f} KB)\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
