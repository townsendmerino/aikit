#!/usr/bin/env python3
"""pin_mellum_rope.py — pin the YaRN RoPE reference for Mellum2's full-attention
layers (and the plain RoPE its sliding layers use), so the Go YaRN implementation
can be validated id-for-id without the 12B checkpoint.

The YaRN inv_freq + attention_factor are computed by replicating
transformers.modeling_rope_utils._compute_yarn_parameters VERBATIM (the version
in .venv was read line-for-line to write this), for the params in
JetBrains/Mellum2-12B-A2.5B-Thinking's config.json:

  full_attention:    rope_type=yarn, theta=500000, factor=16,
                     original_max_position_embeddings=8192, beta_fast=32,
                     beta_slow=1, attention_factor=1.2772588722239782
  sliding_attention: rope_type=default, theta=500000

head_dim 128 (partial_rotary_factor 1.0) → dim 128, 64 inverse frequencies.

Usage: .venv/bin/python scripts/pin_mellum_rope.py
       writes testdata/mellum_rope_golden.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

BASE = 500000.0
DIM = 128  # head_dim * partial_rotary_factor
FACTOR = 16.0
ORIG_MAX = 8192
BETA_FAST = 32.0
BETA_SLOW = 1.0
ATTENTION_FACTOR = 1.2772588722239782


def yarn_inv_freq():
    """Verbatim port of _compute_yarn_parameters' inv_freq computation (truncate=True)."""

    def find_correction_dim(num_rotations, dim, base, max_pos):
        return (dim * math.log(max_pos / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_pos):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_pos))  # truncate=True
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_pos))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp(mn, mx, n):
        if mn == mx:
            mx += 0.001
        return [min(max((i - mn) / (mx - mn), 0.0), 1.0) for i in range(n)]

    pos_freqs = [BASE ** ((2 * i) / DIM) for i in range(DIM // 2)]
    inv_extrap = [1.0 / p for p in pos_freqs]
    inv_interp = [1.0 / (FACTOR * p) for p in pos_freqs]
    low, high = find_correction_range(BETA_FAST, BETA_SLOW, DIM, BASE, ORIG_MAX)
    ramp = linear_ramp(low, high, DIM // 2)
    extrap_factor = [1.0 - r for r in ramp]
    inv_freq = [
        inv_interp[i] * (1 - extrap_factor[i]) + inv_extrap[i] * extrap_factor[i]
        for i in range(DIM // 2)
    ]
    return inv_freq, low, high


def plain_inv_freq():
    return [BASE ** (-(2 * i) / DIM) for i in range(DIM // 2)]


def main() -> int:
    inv_yarn, low, high = yarn_inv_freq()
    payload = {
        "note": (
            "YaRN RoPE reference for Mellum2 full-attention layers + plain RoPE for "
            "sliding layers. Port of transformers _compute_yarn_parameters (truncate=True). "
            "Go computeInvFreq(base, dim, yarn) must match 'full.inv_freq' and the plain "
            "table must match 'sliding.inv_freq'; the mscale is 'full.attention_factor'."
        ),
        "params": {
            "base": BASE, "dim": DIM, "factor": FACTOR, "original_max_position_embeddings": ORIG_MAX,
            "beta_fast": BETA_FAST, "beta_slow": BETA_SLOW,
            "correction_range_low": low, "correction_range_high": high,
        },
        "full": {"rope_type": "yarn", "attention_factor": ATTENTION_FACTOR, "inv_freq": inv_yarn},
        "sliding": {"rope_type": "default", "attention_factor": 1.0, "inv_freq": plain_inv_freq()},
    }
    # Cross-check the provided attention_factor against get_mscale(factor).
    expect_mscale = 0.1 * math.log(FACTOR) + 1.0
    assert abs(expect_mscale - ATTENTION_FACTOR) < 1e-12, (expect_mscale, ATTENTION_FACTOR)

    out = REPO_ROOT / "testdata" / "mellum_rope_golden.json"
    with out.open("w") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.write("\n")
    print(f"[pin_mellum_rope] correction range low={low} high={high}, mscale={ATTENTION_FACTOR}")
    print(f"[pin_mellum_rope] wrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
