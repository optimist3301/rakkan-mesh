#!/usr/bin/env python3
"""Reference outputs of DeepSeek-V4.1-Flash's image encoder on its published
weights, for the Go engine's check (internal/llm/dsv41/real_test.go,
TestRealVision).

The unmodified vision.py (ViT and Aligner) with the weights of shard 1
encodes a seeded synthetic picture, prepared by the unmodified
image_processor.py.

Writes a safetensors file with:
  png [bytes] uint8: the picture
  grid [4] i64: n_vit_h, n_vit_w, n_llm_h, n_llm_w
  patches [n, 3, 14, 14] bf16: load_image's
  block{i} [n, 1024] bf16: the stream after encoder block i (0, 15, 31)
  vit [n, 1024] bf16: the encoder's output
  embeds [L, dim] bf16: the aligner's

usage: vision_check.py OUT.safetensors [--size W H]
Downloads (fetch.py) shard 1 (0.9 GiB).
"""
import argparse
import io
import os

import numpy as np
import torch

import fetch
import run

SHARD = "model-00001-of-00048.safetensors"


def picture(w, h, seed=20261104):
    """Smooth shapes and a little noise: closer to a photograph's statistics
    than noise is."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    img = np.stack([x / w * 255, y / h * 255, (x + y) / (w + h) * 255], -1)
    for _ in range(12):
        cx, cy, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(10, min(w, h) / 3)
        color = rng.uniform(0, 255, 3)
        mask = (x - cx) ** 2 + (y - cy) ** 2 < r * r
        img[mask] = img[mask] * 0.3 + color * 0.7
    img += rng.normal(0, 6, img.shape)
    buf = io.BytesIO()
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB").save(buf, "PNG")
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--size", type=int, nargs=2, default=(420, 300))
    a = ap.parse_args()

    run.setup()
    import image_processor as IP
    import vision as V
    from safetensors import safe_open
    from safetensors.torch import save_file

    import model as m

    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(os.cpu_count())
    args = run.load_args(m, os.path.join(run.UPSTREAM, "config.json"), max_batch_size=1)
    vit, aligner = V.ViT(args), V.Aligner(args)
    path = fetch.fetch(SHARD)
    with safe_open(path, "pt") as f:
        for mod, prefix in ((vit, "vision."), (aligner, "aligner.")):
            state = {k[len(prefix) :]: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)}
            missing, unexpected = mod.load_state_dict(state, strict=True)
            assert not missing and not unexpected, (missing, unexpected)

    png = picture(*a.size)
    patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = IP.load_image({"data": png}, args)
    out = {
        "png": torch.frombuffer(bytearray(png), dtype=torch.uint8),
        "grid": torch.tensor([n_vit_h, n_vit_w, n_llm_h, n_llm_w]),
        "patches": patches,
    }
    keep = (0, len(vit.blocks) // 2 - 1, len(vit.blocks) - 1)
    for i, block in enumerate(vit.blocks):
        if i in keep:
            block.register_forward_hook(lambda mod, inp, o, i=i: out.__setitem__(f"block{i}", o.detach().clone()))
    with torch.inference_mode():
        x = vit(patches, n_vit_h, n_vit_w)
        out["vit"] = x.clone()
        out["embeds"] = aligner(x, n_vit_h, n_vit_w).clone()
    save_file({k: v.contiguous() for k, v in out.items()}, a.out)
    print(f"{n_vit_h}x{n_vit_w} patches, {n_llm_h}x{n_llm_w} positions -> {a.out}")


if __name__ == "__main__":
    main()
