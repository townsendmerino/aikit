#!/usr/bin/env python3
"""pin_iq_dequant.py — parity-gate the Go codebook and FP4 GGUF dequant (IQ4_NL,
IQ4_XS, IQ2_S, IQ3_S, NVFP4, IQ2_XXS, IQ2_XS, IQ3_XXS) against the canonical llama.cpp `gguf` Python reference.

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
    .venv/bin/python scripts/oracle/pin_iq_dequant.py
    # writes testdata/iq_dequant_golden.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from gguf.quants import dequantize
from gguf.constants import GGMLQuantizationType as Q

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
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


def build_grid_blocks(rng: np.random.Generator, nblocks: int, type_size: int) -> bytes:
    # IQ2_S / IQ3_S: a fixed clean f16 scale then random payload (grid indices,
    # sign bits, sub-scales) — the codebook lookup + sign unpack is exercised
    # regardless of which entries get hit.
    out = bytearray()
    for b in range(nblocks):
        out += f16le(CLEAN_D[b % len(CLEAN_D)])
        out += bytes(rng.integers(0, 256, type_size - 2, dtype=np.uint8).tolist())
    return bytes(out)


def build_nvfp4(rng: np.random.Generator, nrandom: int) -> bytes:
    # NVFP4 (64 elems, 36 bytes: 4 ue4m3 sub-block scales, then 8 nibble bytes per
    # sub-block). The first 64 blocks are exhaustive: their 256 scale bytes are
    # 0x00..0xFF in order (0x7F, 0x80 and 0xFF included), and every sub-block's
    # bytes are k | (k+8)<<4 for k = 0..7, so each of the 16 e2m1 codes meets each
    # scale byte once, and the low and high nibbles land in different halves of the
    # sub-block. Then `nrandom` fully random blocks.
    out = bytearray()
    sub = bytes(k | ((k + 8) << 4) for k in range(8))
    for b in range(64):
        out += bytes(range(4 * b, 4 * b + 4))
        out += sub * 4
    for _ in range(nrandom):
        out += bytes(rng.integers(0, 256, 36, dtype=np.uint8).tolist())
    return bytes(out)


def u32le(x: int) -> bytes:
    return int(x).to_bytes(4, "little")


def build_iq2xxs(rng: np.random.Generator, nrandom: int) -> bytes:
    # IQ2_XXS (66 bytes): d, then per 32-element sub-block a u32 of four grid bytes
    # and a u32 of four 7-bit sign indices and a 4-bit scale. The first 8 blocks use
    # grid entry k = 0..255 once each, sign index k%128 and scale (sub-block)%16.
    out = bytearray()
    for b in range(8):
        out += f16le(CLEAN_D[b % len(CLEAN_D)])
        for ib in range(8):
            ks = [32 * b + 4 * ib + l for l in range(4)]
            out += bytes(k % 256 for k in ks)
            out += u32le(sum((k % 128) << (7 * l) for l, k in enumerate(ks)) | ((8 * b + ib) % 16) << 28)
    out += build_grid_blocks(rng, nrandom, 66)
    return bytes(out)


def build_iq2xs(rng: np.random.Generator, nrandom: int) -> bytes:
    # IQ2_XS (74 bytes): d, qs[32] u16 (9-bit grid index | 7-bit sign index << 9),
    # scales[8]. The first 16 blocks use grid entry k = 0..511 once each with sign
    # index k%128; their scale bytes run through every value of both nibbles.
    out = bytearray()
    for b in range(16):
        out += f16le(CLEAN_D[b % len(CLEAN_D)])
        for l in range(32):
            k = 32 * b + l
            out += (k % 512 | (k % 128) << 9).to_bytes(2, "little")
        out += bytes(((8 * b + i) % 16) | ((15 - (8 * b + i)) % 16) << 4 for i in range(8))
    out += build_grid_blocks(rng, nrandom, 74)
    return bytes(out)


def build_iq3xxs(rng: np.random.Generator, nrandom: int) -> bytes:
    # IQ3_XXS (98 bytes): d, qs[64] grid bytes, then per 32-element sub-block a u32
    # of four 7-bit sign indices and a 4-bit scale. The first 4 blocks use grid
    # entry k = 0..255 once each; sign indices run k%128 and scales (sub-block)%16.
    out = bytearray()
    for b in range(4):
        out += f16le(CLEAN_D[b % len(CLEAN_D)])
        out += bytes(range(64 * b, 64 * b + 64))
        for ib in range(8):
            ks = [32 * b + 4 * ib + l for l in range(4)]
            out += u32le(sum((k % 128) << (7 * l) for l, k in enumerate(ks)) | ((8 * b + ib) % 16) << 28)
    out += build_grid_blocks(rng, nrandom, 98)
    return bytes(out)


def main() -> int:
    rng = np.random.default_rng(42)
    specs = [
        ("IQ4_NL", Q.IQ4_NL, 18, 32, build_iq4nl(rng, 4)),
        ("IQ4_XS", Q.IQ4_XS, 136, 256, build_iq4xs(rng, 3)),
        ("IQ2_S", Q.IQ2_S, 82, 256, build_grid_blocks(rng, 3, 82)),
        ("IQ3_S", Q.IQ3_S, 110, 256, build_grid_blocks(rng, 3, 110)),
        ("NVFP4", Q.NVFP4, 36, 64, build_nvfp4(rng, 4)),
        ("IQ2_XXS", Q.IQ2_XXS, 66, 256, build_iq2xxs(rng, 2)),
        ("IQ2_XS", Q.IQ2_XS, 74, 256, build_iq2xs(rng, 2)),
        ("IQ3_XXS", Q.IQ3_XXS, 98, 256, build_iq3xxs(rng, 2)),
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
            "IQ4_NL / IQ4_XS / IQ2_S / IQ3_S / NVFP4 / IQ2_XXS / IQ2_XS / IQ3_XXS dequant oracle from llama.cpp's gguf Python reference "
            "(gguf.quants.dequantize). Deterministic raw super-blocks (fixed clean "
            "f16 scale, randomized codes/sub-scales; NVFP4 also every ue4m3 scale byte "
            "against every e2m1 code; the IQ*_XXS / IQ2_XS cases every grid entry and sign index); the Go dequantRange must "
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
