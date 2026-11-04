#!/usr/bin/env python3
"""Test vectors for the Go image preparation (internal/pil and dsv41's
image.go): what Pillow and the reference's image_processor.py make of
seeded images.

pil vectors (internal/pil/testdata/pillow.json.gz):

  decode   PNG files of every colour type and bit depth, with and without
           tRNS, written by hand (zlib, filter 0) or by Pillow, and the RGB
           pixels of Image.open(...).convert("RGB")
  resize   RGB images and Image.resize(size) (BICUBIC): up, down, one
           side each way, the vertical pass first (height shrinking more
           than twice the width), one-pixel sides
  pad      ImageOps.pad(image, size, color=(127, 127, 127))

dsv41 vectors (internal/llm/dsv41/testdata/images.json.gz):

  grid     plan_image_grid(width, height, args) for many sizes and args:
           (n_llm_h, n_llm_w, best_height, best_width)
  load     load_image(record, args) for PNG files: the grids and the sha256
           of the patches' bf16 bits, little-endian, with the first patch

Images are bytes base64, pixel arrays raw RGB base64.

usage: image_vectors.py PIL_OUT.json.gz DSV41_OUT.json.gz
"""
import base64
import gzip
import hashlib
import io
import json
import os
import random
import struct
import sys
import zlib
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "upstream", "inference"))
import image_processor as IP  # noqa: E402

rng = random.Random(20261104)


def b64(b):
    return base64.b64encode(bytes(b)).decode()


def png_chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def png(width, height, depth, ctype, samples, plte=None, trns=None):
    """A PNG of the given samples (a flat list in row order, ints of the
    bit depth), filter 0 on every row."""
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[ctype]
    raw = bytearray()
    per_row = width * channels
    for y in range(height):
        raw.append(0)
        row = samples[y * per_row : (y + 1) * per_row]
        if depth == 16:
            for v in row:
                raw += struct.pack(">H", v)
        elif depth == 8:
            raw += bytes(row)
        else:
            acc, bits = 0, 0
            for v in row:
                acc = (acc << depth) | v
                bits += depth
                if bits == 8:
                    raw.append(acc)
                    acc, bits = 0, 0
            if bits:
                raw.append(acc << (8 - bits))
    out = b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, depth, ctype, 0, 0, 0))
    if plte is not None:
        out += png_chunk(b"PLTE", bytes(plte))
    if trns is not None:
        out += png_chunk(b"tRNS", bytes(trns))
    return out + png_chunk(b"IDAT", zlib.compress(bytes(raw))) + png_chunk(b"IEND", b"")


def rand_samples(n, depth):
    top = (1 << depth) - 1
    # extremes often, as they are where conversions clip
    return [rng.choice((0, top, rng.randint(0, top))) for _ in range(n)]


def decode_cases():
    cases = []
    sizes = [(1, 1), (7, 3), (13, 9), (32, 17)]
    for ctype, depths in ((0, (1, 2, 4, 8, 16)), (2, (8, 16)), (3, (1, 2, 4, 8)), (4, (8, 16)), (6, (8, 16))):
        for depth in depths:
            for w, h in sizes:
                ch = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[ctype]
                plte = trns = None
                if ctype == 3:
                    n = 1 << depth
                    plte = [rng.randint(0, 255) for _ in range(3 * n)]
                    samples = [rng.randrange(n) for _ in range(w * h)]
                else:
                    samples = rand_samples(w * h * ch, depth)
                    if ctype == 0 and depth == 16:
                        # small values too: below 256 they pass, above clip
                        samples = [rng.choice((v, rng.randint(0, 300))) for v in samples]
                for with_trns in (False, True):
                    if with_trns:
                        if ctype == 3:
                            trns = [rng.randint(0, 255) for _ in range(rng.randint(1, len(plte) // 3))]
                        elif ctype == 0:
                            trns = struct.pack(">H", samples[0])
                        elif ctype == 2:
                            trns = struct.pack(">HHH", *samples[:3]) if depth == 16 else bytes([0, samples[0], 0, samples[1], 0, samples[2]])
                        else:
                            continue
                    data = png(w, h, depth, ctype, samples, plte, trns)
                    with Image.open(io.BytesIO(data)) as im:
                        mode = im.mode
                        rgb = im.convert("RGB")
                    cases.append(
                        {
                            "name": f"type{ctype}-{depth}bit-{w}x{h}{'-trns' if with_trns else ''}",
                            "mode": mode,
                            "png": b64(data),
                            "w": rgb.width,
                            "h": rgb.height,
                            "rgb": b64(rgb.tobytes()),
                        }
                    )
    # Pillow's own writer: other filters and zlib settings
    for mode in ("1", "L", "LA", "P", "RGB", "RGBA", "I;16"):
        im = random_rgb(23, 11).convert(mode if mode != "I;16" else "L")
        if mode == "I;16":
            im = Image.fromarray(np.asarray(im, dtype=np.uint16) * 3)  # mode I;16
        for optimize in (False, True):
            buf = io.BytesIO()
            im.save(buf, "PNG", optimize=optimize)
            data = buf.getvalue()
            with Image.open(io.BytesIO(data)) as im2:
                mode2 = im2.mode
                rgb = im2.convert("RGB")
            cases.append({"name": f"pillow-{mode}-{optimize}", "mode": mode2, "png": b64(data), "w": rgb.width, "h": rgb.height, "rgb": b64(rgb.tobytes())})
    return cases


def random_rgb(w, h, kind=None):
    kind = kind or rng.choice(("noise", "gradient", "edges"))
    if kind == "noise":
        a = np.array([rng.randint(0, 255) for _ in range(w * h * 3)], dtype=np.uint8)
    elif kind == "gradient":
        y, x = np.mgrid[0:h, 0:w]
        a = np.stack([(x * 255 // max(w - 1, 1)), (y * 255 // max(h - 1, 1)), ((x + y) * 7) % 256], -1).astype(np.uint8)
    else:  # hard 0/255 edges: the bicubic overshoot clips
        a = np.array([rng.choice((0, 255)) for _ in range(w * h * 3)], dtype=np.uint8)
    return Image.fromarray(a.reshape(h, w, 3), "RGB")


def resize_cases():
    cases = []
    pairs = [
        ((10, 10), (20, 20)),
        ((20, 20), (10, 10)),
        ((17, 13), (40, 5)),
        ((40, 30), (41, 29)),
        ((50, 60), (49, 10)),  # vertical first: the height shrinks most
        ((60, 50), (10, 49)),
        ((1, 1), (5, 7)),
        ((5, 7), (1, 1)),
        ((1, 30), (14, 28)),
        ((97, 3), (14, 14)),
        ((33, 33), (33, 12)),
        ((33, 33), (12, 33)),
        ((128, 96), (126, 98)),
        ((7, 200), (21, 63)),
    ]
    for (w, h), (W, H) in pairs:
        for kind in ("noise", "gradient", "edges"):
            im = random_rgb(w, h, kind)
            out = im.resize((W, H))
            cases.append({"w": w, "h": h, "in": b64(im.tobytes()), "W": W, "H": H, "out": b64(out.tobytes())})
    for _ in range(40):
        w, h = rng.randint(1, 90), rng.randint(1, 90)
        W, H = rng.randint(1, 90), rng.randint(1, 90)
        im = random_rgb(w, h)
        out = im.resize((W, H))
        cases.append({"w": w, "h": h, "in": b64(im.tobytes()), "W": W, "H": H, "out": b64(out.tobytes())})
    return cases


def pad_cases():
    cases = []
    for _ in range(30):
        w, h = rng.randint(1, 70), rng.randint(1, 70)
        W, H = rng.randint(1, 70), rng.randint(1, 70)
        if rng.random() < 0.2:
            W, H = w * 2, h * 2  # same ratio
        im = random_rgb(w, h)
        out = ImageOps.pad(im, (W, H), color=(127, 127, 127))
        cases.append({"w": w, "h": h, "in": b64(im.tobytes()), "W": W, "H": H, "out": b64(out.tobytes())})
    return cases


REAL = dict(
    vision_patch_size=14,
    vision_downsample_ratio=3,
    vision_max_n_token=1024,
    vision_min_pixels=295936,
    vision_max_wh_ratio=None,
)


def variants():
    yield REAL
    yield dict(REAL, vision_max_wh_ratio=2)
    yield dict(REAL, vision_min_pixels=0, vision_max_n_token=40)
    yield dict(REAL, vision_patch_size=4, vision_downsample_ratio=2, vision_max_n_token=30, vision_min_pixels=100)
    yield dict(REAL, vision_patch_size=4, vision_downsample_ratio=2, vision_max_n_token=12, vision_min_pixels=0, vision_max_wh_ratio=3)


def grid_cases():
    cases = []
    for args in variants():
        ns = SimpleNamespace(**args)
        sizes = [(1, 1), (1, 5000), (5000, 1), (544, 544), (543, 545), (14, 14), (100000, 3), (3, 100000)]
        sizes += [(rng.randint(1, 4000), rng.randint(1, 4000)) for _ in range(300)]
        sizes += [(rng.randint(1, 60), rng.randint(1, 60)) for _ in range(100)]
        for w, h in sizes:
            try:
                g = IP.plan_image_grid(w, h, ns)
            except AssertionError:
                g = None
            cases.append({"args": args, "w": w, "h": h, "grid": None if g is None else list(g)})
    return cases


def load_cases():
    cases = []
    sizes = [(40, 30), (13, 200), (300, 7), (544, 544), (600, 400), (31, 31)]
    for args in variants():
        ns = SimpleNamespace(**args)
        for w, h in sizes:
            # large noise compresses badly: large images are gradients
            im = random_rgb(w, h, None if w * h <= 4000 else "gradient")
            if rng.random() < 0.3:
                im = im.convert("RGBA")
            buf = io.BytesIO()
            im.save(buf, "PNG")
            data = buf.getvalue()
            patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = IP.load_image({"data": data}, ns)
            bits = patches.contiguous().view(-1).view(dtype=__import__("torch").int16).numpy().astype("<i2").tobytes()
            cases.append(
                {
                    "args": args,
                    "png": b64(data),
                    "grid": [n_vit_h, n_vit_w, n_llm_h, n_llm_w],
                    "sha256": hashlib.sha256(bits).hexdigest(),
                    "first": b64(bits[: patches.shape[1] * patches.shape[2] * patches.shape[3] * 2]),
                }
            )
    return cases


def write(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with gzip.open(path, "wt", compresslevel=9) as f:
        json.dump(obj, f, separators=(",", ":"))
    print(path, os.path.getsize(path), "bytes")


def main():
    import PIL

    pil_out, ds_out = sys.argv[1:3]
    write(pil_out, {"pillow": PIL.__version__, "decode": decode_cases(), "resize": resize_cases(), "pad": pad_cases()})
    write(ds_out, {"pillow": PIL.__version__, "grid": grid_cases(), "load": load_cases()})


if __name__ == "__main__":
    main()
