#!/usr/bin/env python
"""Pin a tiny-random Qwen2.5-VL vision tower forward as an aikit parity golden — the
second ViT family (after SigLIP / pin_siglip_vision.py), dynamic-resolution.

Builds a SMALL random Qwen2_5_VLForConditionalGeneration (so the visual submodule
is the real Qwen2_5_VisionTransformerPretrainedModel + patch merger), runs its
`model.model.get_image_features(pixel_values, grid_thw)` at CPU float32, and dumps
BOTH stages:
  - last_hidden_state [n_patches, hidden]   — the ViT pre-merge output (block gate),
  - pooler_output[0]  [n_merged, out_hidden] — after the patch merger (merger gate).
The Go QwenVisionEncoder loads the SAME saved checkpoint + the SAME pixel_values +
grid_thw and must reproduce both (cosine ≥ 0.9999).

Unlike SigLIP (fixed 896×896 → 256 tokens, learned absolute pos, LayerNorm), this
exercises the Qwen2.5-VL deltas: pre-flattened patches + grid_thw (dynamic
resolution), 2D rotary, RMSNorm, windowed + full attention (fullatt_block_indexes),
a gated SiLU MLP, and the spatial-merge patch merger. Tiny → sub-second; no download.

    ~/.venv-vl/bin/python scripts/pin_qwen25vl_vision.py
    -> testdata/qwen25vl_vision_golden.json   (config + pixel_values + grid_thw + both stages)
    -> testdata/qwen25vl-vision-tiny/          (HF safetensors checkpoint of the same weights)
"""
import json
import os

import torch
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLTextConfig, Qwen2_5_VLVisionConfig)

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "qwen25vl_vision_golden.json")
CKPT = os.path.join(HERE, "..", "testdata", "qwen25vl-vision-tiny")

# Mirrors goinfer's tiny fixture (scripts/pin_qwen25vl_tiny.py): a 2-block ViT, one
# windowed + one full-attention block (fullatt_block_indexes=[1]), merge size 2.
TEXT = dict(
    vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    max_position_embeddings=128, rms_norm_eps=1e-6,
    rope_theta=10000.0, rope_scaling={"type": "mrope", "mrope_section": [4, 2, 2]},
)
VISION = dict(
    depth=2, hidden_size=32, intermediate_size=64, num_heads=2, in_chans=3,
    patch_size=14, spatial_merge_size=2, temporal_patch_size=2, out_hidden_size=64,
    window_size=112, fullatt_block_indexes=[1], hidden_act="silu",
)


def main():
    torch.manual_seed(0)
    cfg = Qwen2_5_VLConfig(
        text_config=Qwen2_5_VLTextConfig(**TEXT),
        vision_config=Qwen2_5_VLVisionConfig(**VISION),
        image_token_id=299, vision_start_token_id=298,
    )
    model = Qwen2_5_VLForConditionalGeneration(cfg).eval().to(torch.float32)
    vcfg = model.config.vision_config
    merge = vcfg.spatial_merge_size

    # One image, grid (t,h,w) in patch units; h,w multiples of merge for the merger.
    # h=4,w=6 with window_merger_size = 112//2//14 = 4 → the 2×3 merged grid spans
    # more than one window, so the windowed block's reorder is actually exercised.
    t, h, w = 1, 4, 6
    n_patches = t * h * w
    patch_dim = vcfg.in_chans * vcfg.temporal_patch_size * vcfg.patch_size * vcfg.patch_size
    n_merged = n_patches // (merge * merge)

    gen = torch.Generator().manual_seed(1)
    pixel_values = torch.randn(n_patches, patch_dim, generator=gen, dtype=torch.float32)
    grid_thw = torch.tensor([[t, h, w]], dtype=torch.long)

    with torch.no_grad():
        vis = model.model.get_image_features(pixel_values, grid_thw)
        vit_hidden = vis.last_hidden_state        # [n_patches, hidden]   (pre-merge)
        image_features = vis.pooler_output[0]     # [n_merged, out_hidden] (merged)

    golden = {
        "note": "tiny-random Qwen2.5-VL vision tower (get_image_features); CPU fp32",
        "vision_config": VISION,
        "grid_thw": [[t, h, w]],
        "n_patches": n_patches, "n_merged": n_merged,
        "pixel_values_shape": list(pixel_values.shape),
        "pixel_values": pixel_values.flatten().tolist(),
        "vit_hidden_shape": list(vit_hidden.shape),
        "vit_hidden": vit_hidden.reshape(-1).float().tolist(),
        "image_features_shape": list(image_features.shape),
        "image_features": image_features.reshape(-1).float().tolist(),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(golden, f)
    print(f"wrote {OUT}")
    print(f"  n_patches={n_patches} vit_hidden_shape={golden['vit_hidden_shape']}")
    print(f"  n_merged={n_merged} image_features_shape={golden['image_features_shape']}")

    model.save_pretrained(CKPT, safe_serialization=True)
    print(f"saved checkpoint -> {CKPT}  (model_type={cfg.model_type!r})")


if __name__ == "__main__":
    main()
