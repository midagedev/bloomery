#!/usr/bin/env python3
"""Write the fixed test images of the vision oracle into images/ beside this file.

The pixels are a pure function of the coordinates (integer hashing, no RNG), so the set can be
rebuilt anywhere with numpy and Pillow. The PNG bytes also depend on the zlib that writes them, so
the oracle does not trust a rebuild: dump_vision.py records each file's sha256, and the gates read
the committed files, not a regenerated set.

Every image is 8-bit RGB with no alpha and no palette, so the reference's `convert("RGB")` is the
identity and decoding is the only step before the resize plan.

Each size exercises one branch of the reference's preprocessing (`image_processor.load_image`):

  grad-448.png        448x448    below vision_min_pixels: grid 546x546, bicubic up in both axes
  checker-1036.png    1036x1036  a multiple of 14 inside the token budget: no resampling at all
  noise-546.png       546x546    the minimum grid itself: no resampling, photo-like content
  odd-777x513.png     777x513    not a multiple of 14: stretched to 784x518, no padding
  pad-600x451.png     600x451    below min pixels: grid 630x476, contain() to 630x474, pasted one row down
  wide-2400x1350.png  2400x1350  over the token budget: grid 1708x952, bicubic down to 1692x952, 8 px in
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent / "images"


def hash2(x, y, seed):
    """A 32-bit integer hash of lattice coordinates (lowbias32 over a mixed key)."""
    h = (x.astype(np.uint64) * np.uint64(0x9E3779B1) + y.astype(np.uint64) * np.uint64(0x85EBCA77)
         + np.uint64(seed) * np.uint64(0xC2B2AE3D)) & np.uint64(0xFFFFFFFF)
    h ^= h >> np.uint64(16)
    h = (h * np.uint64(0x7FEB352D)) & np.uint64(0xFFFFFFFF)
    h ^= h >> np.uint64(15)
    h = (h * np.uint64(0x846CA68B)) & np.uint64(0xFFFFFFFF)
    h ^= h >> np.uint64(16)
    return h


def value_noise(w, h, cell, seed):
    """Smooth noise in [0, 1): bilinear interpolation of hashed lattice values, `cell` pixels apart."""
    ys, xs = np.mgrid[0:h, 0:w]
    gx, gy = xs // cell, ys // cell
    fx = (xs % cell) / cell
    fy = (ys % cell) / cell
    fx = fx * fx * (3 - 2 * fx)
    fy = fy * fy * (3 - 2 * fy)

    def lat(dx, dy):
        return hash2(gx + dx, gy + dy, seed).astype(np.float64) / 2.0**32

    top = lat(0, 0) * (1 - fx) + lat(1, 0) * fx
    bot = lat(0, 1) * (1 - fx) + lat(1, 1) * fx
    return top * (1 - fy) + bot * fy


def to_u8(a):
    return np.clip(np.floor(a * 256.0), 0, 255).astype(np.uint8)


def gradient(w, h):
    ys, xs = np.mgrid[0:h, 0:w]
    r = xs / max(w - 1, 1)
    g = ys / max(h - 1, 1)
    b = (xs + ys) / max(w + h - 2, 1)
    return np.stack([r, g, 1 - b], axis=-1)


def checker(w, h, cell):
    ys, xs = np.mgrid[0:h, 0:w]
    on = ((xs // cell) + (ys // cell)) % 2
    tint = gradient(w, h)
    return np.where(on[..., None] == 1, 0.15 + 0.7 * tint, 0.9 - 0.7 * tint)


def photo(w, h):
    """Several octaves of value noise per channel plus a fine grain: edges, texture and flat areas."""
    chans = []
    for c in range(3):
        a = np.zeros((h, w))
        amp, total = 1.0, 0.0
        for octave, cell in enumerate((96, 48, 24, 12, 6)):
            a += amp * value_noise(w, h, cell, 17 * c + octave)
            total += amp
            amp *= 0.55
        a /= total
        ys, xs = np.mgrid[0:h, 0:w]
        grain = hash2(xs, ys, 1000 + c).astype(np.float64) / 2.0**32
        chans.append(0.9 * a + 0.1 * grain)
    return np.stack(chans, axis=-1)


def mixed(w, h):
    ys, xs = np.mgrid[0:h, 0:w]
    base = gradient(w, h)
    band = (xs * 3 // max(w, 1))[..., None]
    return np.where(band == 0, base, np.where(band == 1, checker(w, h, 9), photo(w, h)))


IMAGES = {
    "grad-448.png": lambda: gradient(448, 448),
    "checker-1036.png": lambda: checker(1036, 1036, 7),
    "noise-546.png": lambda: photo(546, 546),
    "odd-777x513.png": lambda: mixed(777, 513),
    "pad-600x451.png": lambda: mixed(600, 451),
    "wide-2400x1350.png": lambda: checker(2400, 1350, 25),
}


def main():
    OUT.mkdir(exist_ok=True)
    total = 0
    for name, make in IMAGES.items():
        img = Image.fromarray(to_u8(make()))
        assert img.mode == "RGB", img.mode
        path = OUT / name
        img.save(path, optimize=True)
        total += path.stat().st_size
        print(f"{name}\t{img.width}x{img.height}\t{path.stat().st_size} B")
    print(f"total\t{total} B")
    return 0 if total <= 2 * 1024 * 1024 else 1


if __name__ == "__main__":
    sys.exit(main())
