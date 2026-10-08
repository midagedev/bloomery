#!/usr/bin/env python3
"""Write the eight test images of the Clef-Flash image-input oracle into images/ beside this file.

The pixels are a pure function of the coordinates (integer arithmetic, no RNG), so the set can be
rebuilt anywhere with numpy and Pillow. The PNG bytes also depend on the zlib that writes them, so
nothing trusts a rebuild: dump_mtmd records each file's sha256 in set A's header, and the gates read
the committed files.

Every image is 8-bit RGB with no alpha and no palette, so decoding is the only step before the size
plan. Each size exercises one branch of llama.cpp mtmd's `qwen3vl_merger` preprocessing
(`calc_size_preserved_ratio` with align 32, min 8 tokens, max 4096 tokens, then the black PAD_CEIL
resize), with the aligned size computed by `plan()` below in f32, op for op:

  free-448x448.png      448x448     a multiple of 32 inside the token budget: copied, no resize, no pad
  free-640x480.png      640x480     the same
  free-1024x768.png     1024x768    the same
  min-96x64.png         96x64       below min_pixels: scaled up to 128x96 (12 tokens), bicubic up in both axes
  pad-1000x700.png      1000x700    aligns to 992x704: bicubic down to 992x695, centred on black, 4 rows above
  pad-777x513.png       777x513     aligns to 768x512: bicubic down to 768x508, 2 rows above
  max-2600x1800.png     2600x1800   above max_pixels: 2432x1696 (4028 tokens), bicubic down, no pad
  strip-1000x1.png      1000x1      aligns to 992x32 (31x1 tokens): one image row at row 15, black elsewhere

The content is smooth colour ramps, rings and coarse checkers, so each file compresses to a few
kilobytes and a resize still meets edges and every one of the 256 levels.
"""

import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent / "images"

PATCH, MERGE = 16, 2
ALIGN = PATCH * MERGE
MIN_TOKENS, MAX_TOKENS = 8, 4096
MIN_PIXELS = MIN_TOKENS * ALIGN * ALIGN
MAX_PIXELS = MAX_TOKENS * ALIGN * ALIGN

F32 = np.float32


def round_half_away(x):
    """std::round on an f32: half away from zero."""
    return int(math.floor(abs(float(x)) + 0.5)) * (1 if x >= 0 else -1)


def plan(w, h, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    """mtmd's `calc_size_preserved_ratio` (align 32), every operation in f32: (w_bar, h_bar)."""
    f = F32(ALIGN)

    def rnd(x):
        return round_half_away(F32(x) / f) * ALIGN

    def ceil(x):
        return int(math.ceil(float(F32(x) / f))) * ALIGN

    def floor(x):
        return int(math.floor(float(F32(x) / f))) * ALIGN

    w_bar, h_bar = max(ALIGN, rnd(w)), max(ALIGN, rnd(h))
    if max_pixels > 0 and h_bar * w_bar > max_pixels:
        beta = np.sqrt(F32(F32(h) * F32(w)) / F32(max_pixels))
        h_bar = max(ALIGN, floor(F32(h) / beta))
        w_bar = max(ALIGN, floor(F32(w) / beta))
    elif min_pixels > 0 and h_bar * w_bar < min_pixels:
        beta = np.sqrt(F32(min_pixels) / F32(F32(h) * F32(w)))
        h_bar = ceil(F32(h) * beta)
        w_bar = ceil(F32(w) * beta)
    return w_bar, h_bar


def pad_ceil(w, h, w_bar, h_bar):
    """The PAD_CEIL placement of mtmd's `img_tool::resize`: (new_w, new_h, off_x, off_y)."""
    if (w, h) == (w_bar, h_bar):
        return w, h, 0, 0
    scale = min(F32(w_bar) / F32(w), F32(h_bar) / F32(h))
    new_w = min(int(math.ceil(float(F32(w) * scale))), w_bar)
    new_h = min(int(math.ceil(float(F32(h) * scale))), h_bar)
    return new_w, new_h, (w_bar - new_w) // 2, (h_bar - new_h) // 2


def ramp(w, h):
    """Colour ramps: red along x, green along y, blue against red; every level 0..255 occurs."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.int64)
    r = xs * 255 // max(w - 1, 1)
    g = ys * 255 // max(h - 1, 1)
    b = 255 - r
    return np.stack([r, g, b], axis=-1)


def rings(w, h, step):
    """Concentric rings around the centre, `step` squared-pixels wide, tinted by ring parity."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.int64)
    cx, cy = w // 2, h // 2
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    band = (d2 // step) % 2
    base = ramp(w, h)
    return np.where(band[..., None] == 1, 255 - base, base)


def checker(w, h, cell):
    ys, xs = np.mgrid[0:h, 0:w].astype(np.int64)
    on = ((xs // cell) + (ys // cell)) % 2
    base = ramp(w, h)
    return np.where(on[..., None] == 1, 40 + base * 175 // 255, 215 - base * 175 // 255)


def mixed(w, h, cell, step):
    """A checker left, rings right, a ramp strip along the bottom: edges, curves and flat gradients."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.int64)
    img = np.where((xs < w // 2)[..., None], checker(w, h, cell), rings(w, h, step))
    return np.where((ys >= h - max(h // 8, 1))[..., None], ramp(w, h), img)


def strip(w):
    """One row: a ramp with a hard step every 100 pixels."""
    xs = np.arange(w, dtype=np.int64)
    r = xs * 255 // max(w - 1, 1)
    g = np.where((xs // 100) % 2 == 1, 255, 0)
    b = 255 - r
    return np.stack([r, g, b], axis=-1)[None, :, :]


IMAGES = {
    "free-448x448.png": lambda: mixed(448, 448, 28, 6000),
    "free-640x480.png": lambda: mixed(640, 480, 32, 9000),
    "free-1024x768.png": lambda: mixed(1024, 768, 64, 40000),
    "min-96x64.png": lambda: mixed(96, 64, 8, 600),
    "pad-1000x700.png": lambda: mixed(1000, 700, 50, 40000),
    "pad-777x513.png": lambda: mixed(777, 513, 27, 12000),
    "max-2600x1800.png": lambda: mixed(2600, 1800, 130, 600000),
    "strip-1000x1.png": lambda: strip(1000),
}


def main():
    OUT.mkdir(exist_ok=True)
    total = 0
    for name, make in IMAGES.items():
        a = np.clip(make(), 0, 255).astype(np.uint8)
        img = Image.fromarray(a)
        assert img.mode == "RGB", img.mode
        w, h = img.size
        stem = name[: -len(".png")]
        assert stem.endswith(f"{w}x{h}"), (name, w, h)
        path = OUT / name
        img.save(path, optimize=True)
        total += path.stat().st_size
        w_bar, h_bar = plan(w, h)
        nw, nh, ox, oy = pad_ceil(w, h, w_bar, h_bar)
        print(f"{name}\t{w}x{h}\t{path.stat().st_size} B\t-> {w_bar}x{h_bar} ({w_bar // ALIGN}x{h_bar // ALIGN} tokens)"
              f" resized {nw}x{nh} at ({ox},{oy})")
    print(f"total\t{total} B")
    return 0 if total < 200 * 1024 else 1


if __name__ == "__main__":
    sys.exit(main())
