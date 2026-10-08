#!/usr/bin/env python3
"""Set A against an independent path: Pillow's decode, bicubic resize and paste, numpy's f32 normalize.

    tools/ref/clefvis/check_preproc.py --set DIR --images DIR
    tools/ref/clefvis/check_preproc.py --self-test

dump_mtmd's preproc set (`# clefvis preproc`) holds, per test image, the graph input of mtmd's `qwen3vl_merger` tower:
the image after mtmd's own decode (stb_image), `calc_size_preserved_ratio`, PAD_CEIL resize on black and
`(u8/255 - 0.5)/0.5`, channel-planar f32. This tool recomputes each from the PNG with code that shares nothing with
mtmd: the size plan of make_images.py (f32, op for op), Pillow's BICUBIC resize (the algorithm mtmd ports), the
PAD_CEIL placement, and numpy's f32 arithmetic. A value that differs is named by image and count; the exit code is 1
then. The size each `# image` line states must equal the plan's, and each PNG's sha256 the manifest's.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_images  # noqa: E402


def normalize(rgb: np.ndarray) -> np.ndarray:
    """mtmd's `from_u8` then `normalize` (mean = std = 0.5), every operation in f32: u8 [h, w, 3] -> f32 [h, w, 3]."""
    x = rgb.astype(np.float32) / np.float32(255.0)
    return (x - np.float32(0.5)) / np.float32(0.5)


def planar(img: np.ndarray) -> np.ndarray:
    """[h, w, 3] -> [3, h, w], the layout of the graph input."""
    return np.ascontiguousarray(img.transpose(2, 0, 1))


def expected_input(rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """The planar f32 tower input for a decoded u8 [h, w, 3] image, and its (w_bar, h_bar)."""
    from PIL import Image

    h, w, _ = rgb.shape
    w_bar, h_bar = make_images.plan(w, h)
    if (w, h) == (w_bar, h_bar):
        canvas = rgb
    else:
        nw, nh, ox, oy = make_images.pad_ceil(w, h, w_bar, h_bar)
        small = np.asarray(Image.fromarray(rgb).resize((nw, nh), Image.BICUBIC))
        canvas = np.zeros((h_bar, w_bar, 3), dtype=np.uint8)
        canvas[oy : oy + nh, ox : ox + nw] = small
    return planar(normalize(canvas)), (w_bar, h_bar)


def parse_images(text: str) -> list[dict[str, str]]:
    """The `# image` lines of a set's manifest, by the names its `# image columns` line gives."""
    cols: list[str] = []
    rows = []
    for line in text.splitlines():
        f = line.split("\t")
        if f[0] == "# image columns":
            cols = f[1].split(" ")
        elif f[0] == "# image":
            if len(f) - 1 != len(cols):
                raise SystemExit(f"check_preproc: an image line of {len(f) - 1} fields, the columns line names {len(cols)}")
            rows.append(dict(zip(cols, f[1:])))
    if not rows:
        raise SystemExit("check_preproc: the manifest has no # image lines")
    return rows


def compare(got: np.ndarray, want: np.ndarray) -> tuple[int, float]:
    """(count of values whose f32 bits differ, the largest absolute difference)."""
    if got.shape != want.shape:
        raise SystemExit(f"check_preproc: shapes {got.shape} and {want.shape}")
    diff = int(np.count_nonzero(got.view(np.uint32) != want.view(np.uint32)))
    return diff, float(np.max(np.abs(got.astype(np.float64) - want.astype(np.float64))))


def check_set(set_dir: Path, images: Path) -> bool:
    from PIL import Image

    text = (set_dir / "MANIFEST.tsv").read_text()
    if "# clefvis\tpreproc" not in text:
        raise SystemExit(f"check_preproc: {set_dir} is not a preproc set")
    ok = True
    for row in parse_images(text):
        name = row["name"]
        png = images / f"{name}.png"
        sha = hashlib.sha256(png.read_bytes()).hexdigest()
        if sha != row["png_sha256"]:
            print(f"FAIL {name}: the PNG has sha256 {sha}, the set was dumped from {row['png_sha256']}")
            ok = False
            continue
        rgb = np.asarray(Image.open(png).convert("RGB"))
        want, (w_bar, h_bar) = expected_input(rgb)
        if (int(row["best_w"]), int(row["best_h"])) != (w_bar, h_bar):
            print(f"FAIL {name}: mtmd sized {row['best_w']}x{row['best_h']}, the plan says {w_bar}x{h_bar}")
            ok = False
            continue
        if hashlib.sha256(rgb.tobytes()).hexdigest() != row["rgb8_sha256"]:
            print(f"FAIL {name}: Pillow's decode differs from stb_image's (rgb8 sha256)")
            ok = False
        got = np.fromfile(set_dir / f"{name}_inp_raw.0.f32", dtype="<f4").reshape(3, h_bar, w_bar)
        diff, maxd = compare(got, want)
        print(f"{'ok  ' if diff == 0 else 'FAIL'} {name}: {w_bar}x{h_bar}, {diff} of {got.size} values differ, max abs {maxd:.3g}")
        ok = ok and diff == 0
    return ok


def self_test() -> bool:
    ok = True

    def expect(name: str, got: Any, want: Any) -> None:
        nonlocal ok
        if got != want:
            print(f"FAIL {name}: got {got!r}, want {want!r}")
            ok = False

    n = normalize(np.array([[[0, 255, 128]]], dtype=np.uint8))
    expect("normalize ends", n[0, 0, :2].tolist(), [-1.0, 1.0])
    expect("normalize middle", abs(float(n[0, 0, 2]) - (2 * 128 / 255 - 1)) < 1e-7, True)
    expect("normalize is f32", str(n.dtype), "float32")
    expect("planar", planar(np.arange(12, dtype=np.float32).reshape(2, 2, 3)).shape, (3, 2, 2))
    expect("plan 96x64", make_images.plan(96, 64), (128, 96))
    expect("plan 1000x700", make_images.plan(1000, 700), (992, 704))
    expect("plan 1000x1", make_images.plan(1000, 1), (992, 32))
    expect("plan 777x513", make_images.plan(777, 513), (768, 512))
    expect("plan 2600x1800", make_images.plan(2600, 1800), (2432, 1696))
    expect("plan free", [make_images.plan(w, h) for w, h in ((448, 448), (640, 480), (1024, 768))], [(448, 448), (640, 480), (1024, 768)])
    expect("pad 1000x700", make_images.pad_ceil(1000, 700, 992, 704), (992, 695, 0, 4))
    expect("pad 1000x1", make_images.pad_ceil(1000, 1, 992, 32), (992, 1, 0, 15))
    expect("pad 96x64", make_images.pad_ceil(96, 64, 128, 96), (128, 86, 0, 5))
    # the same input to bits: a resize-free image is copied, so the input is the normalized pixels
    rgb = (np.arange(128 * 128 * 3) % 251).astype(np.uint8).reshape(128, 128, 3)
    got, size = expected_input(rgb)
    expect("copy path size", size, (128, 128))
    expect("copy path values", bool(np.array_equal(got, planar(normalize(rgb)))), True)
    # a resize puts the image on black: the border rows of a 1x… strip are the normalized black, -1
    strip = np.full((1, 1000, 3), 200, dtype=np.uint8)
    got, size = expected_input(strip)
    expect("strip size", size, (992, 32))
    expect("strip black rows", float(got[0, 0, 0]), -1.0)
    expect("strip content row", float(got[0, 15, 5]), float(normalize(np.array([[[200, 0, 0]]], dtype=np.uint8))[0, 0, 0]))
    expect("compare equal", compare(got, got.copy()), (0, 0.0))
    other = got.copy()
    other[0, 0, 0] = np.float32(0.0)
    expect("compare differs", compare(got, other)[0], 1)
    rows = parse_images("# image columns\tname w\n# image\ta\t3\n# image\tb\t4\n")
    expect("parse", [r["name"] for r in rows], ["a", "b"])
    print("check_preproc self-test: " + ("ok" if ok else "FAILED"))
    return ok


def main(argv: list[str]) -> int:
    if argv[:1] == ["--self-test"]:
        return 0 if self_test() else 1
    import argparse

    p = argparse.ArgumentParser(prog="check_preproc.py")
    p.add_argument("--set", required=True)
    p.add_argument("--images", required=True)
    a = p.parse_args(argv)
    return 0 if check_set(Path(a.set), Path(a.images)) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
