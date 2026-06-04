#!/usr/bin/env python3
"""pin_iq_dequant.py — parity-gate the Go IQ4_NL / IQ4_XS dequant against the
canonical llama.cpp `gguf` Python reference.

The IQ* GGUF quants are codebook-based (a 4-bit code indexes a fixed non-linear
level table), so unlike the Q*_K block quants there is no widely-hosted small
model + f32 oracle to do a forward-parity check against. Instead we pin the
dequant kernel directly: construct deterministic raw super-blocks here, dequantize
them with `gguf.quants.dequantize` (the reference implementation), and record raw
bytes → expected float32. The Go test (TestIQDequant_matchesReference) feeds the
same raw bytes through dequantRange and must reproduce the reference values.

The f16 block scale is fixed to a few clean finite values so the outputs stay
well-scaled (random f16 bits would be mostly NaN/Inf); everything else — the 4-bit
codes and, for IQ4_XS, the 6-bit sub-block scales — is randomized to exercise the
whole codebook and the scale-bit assembly.

Usage (from repo root):
    .venv/bin/python scripts/pin_iq_dequant.py
    # writes testdata/iq_dequant_golden.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from gguf.quants import dequantize
from gguf.constants import GGMLQuantizationType as Q

REPO_ROOT = Path(__file__).resolve().parent.parent
CLEAN_D = [0.05, 0.1, -0.03, 0.2]  # finite f16 block scales


def f16le(x: float) -> bytes:
    return np.float16(x).view(np.uint16).astype("<u2").tobytes()


def build_iq4nl(rng: np.random.Generator, nblocks: int) -> bytes:
    out = bytearray()
    for b in range(nblocks):
        out += f16le(CLEAN_D[b % len(CLEAN_D)])      # d
        out += bytes(rng.integers(0, 256, 16, dtype=np.uint8).tolist())  # qs[16]
    return bytes(out)


def build_iq4xs(rng: np.random.Generator, nblocks: int) -> bytes:
    out = bytearray()
    for b in range(nblocks):
        out += f16le(CLEAN_D[b % len(CLEAN_D)])      # d
        out += bytes(rng.integers(0, 256, 2, dtype=np.uint8).tolist())   # scales_h (u16)
        out += bytes(rng.integers(0, 256, 4, dtype=np.uint8).tolist())   # scales_l[4]
        out += bytes(rng.integers(0, 256, 128, dtype=np.uint8).tolist()) # qs[128]
    return bytes(out)


def main() -> int:
    rng = np.random.default_rng(42)
    specs = [
        ("IQ4_NL", Q.IQ4_NL, 18, 32, build_iq4nl(rng, 4)),
        ("IQ4_XS", Q.IQ4_XS, 136, 256, build_iq4xs(rng, 3)),
    ]
    cases = []
    for name, qtype, blk, nper, raw in specs:
        nblocks = len(raw) // blk
        ref = np.asarray(dequantize(np.frombuffer(raw, dtype=np.uint8), qtype)).reshape(-1).astype(np.float32)
        assert np.isfinite(ref).all(), f"{name} produced non-finite reference"
        cases.append({
            "type": name,
            "ggml_type": int(qtype),
            "block_bytes": blk,
            "elems": nblocks * nper,
            "raw_hex": raw.hex(),
            "expected": [float(x) for x in ref],
        })
        sys.stderr.write(f"  {name:<7} {nblocks} blocks -> {ref.size} values, range [{ref.min():.4f}, {ref.max():.4f}]\n")

    out_path = REPO_ROOT / "testdata" / "iq_dequant_golden.json"
    payload = {
        "note": (
            "IQ4_NL / IQ4_XS dequant oracle from llama.cpp's gguf Python reference "
            "(gguf.quants.dequantize). Deterministic raw super-blocks (fixed clean "
            "f16 scale, randomized codes/sub-scales); the Go dequantRange must "
            "reproduce `expected` from `raw_hex`."
        ),
        "cases": cases,
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=1, allow_nan=False)
        f.write("\n")
    sys.stderr.write(f"[pin_iq_dequant] wrote {out_path.relative_to(REPO_ROOT)} ({out_path.stat().st_size/1024:.1f} KB)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
