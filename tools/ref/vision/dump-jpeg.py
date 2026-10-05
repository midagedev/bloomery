#!/usr/bin/env python3
"""JPEG fixtures for the decoder test: JPEG files cut from the oracle images, and the reference's decode of each.

The reference opens an image with Pillow, `Image.open(path).convert("RGB")`; for a JPEG that is
libjpeg(-turbo)'s ISLOW IDCT, its fancy upsampling and its table colour conversion. This script crops
the images under images/ (make_images.py), encodes each crop with this Pillow at a quality, a chroma
subsampling and a scan mode, and decodes the file it wrote the way the reference does. The test in
crates/vision (`image::tests`) decodes the same files with our decoder and compares, pixel for pixel.

Writes into --out:
  <name>.jpg      the JPEG file
  <name>.rgb      Image.open(<name>.jpg).convert("RGB") as [h, w, 3] u8
  <name>.ycc      the same decode with the colour conversion left out (`draft("YCbCr")`: the IDCT and the
                  upsampling only), [h, w, 3] u8; 3-component files only
  refuse-*.jpg    files our decoder must refuse by name: a CMYK file, and an RGB-coded one (Adobe transform
                  0) when this Pillow can write one
  MANIFEST.tsv    Pillow and libjpeg versions, one `jpeg` row per fixture and one `refuse` row per refusal
                  file, each with the sha256 of its files, `# complete` last

Every decode is done twice and must give the same bytes, and the progressive twin of a baseline fixture
must decode to the same pixels as it (a progressive file carries the same coefficients). The JPEG bytes
depend on the libjpeg that writes them, so the test reads the committed files and their sha256, never a
rebuilt set.

`just dump-ref-jpeg` runs it on the box, with the torch environment's python and the system Pillow (the
one dump_vision.py uses), and brings the set back whole into crates/vision/tests/fixtures/jpeg/.
"""

import argparse
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Pillow is not in the torch environment; the system one is appended after it, as dump_vision.py does,
# so the decoder these pixels come from is the oracle set's.
SYSTEM_PACKAGES = "/usr/lib/python3/dist-packages"

# name, source image, crop (x, y, w, h), mode, subsampling (Pillow: 0 4:4:4, 1 4:2:2, 2 4:2:0), quality, progressive.
# The 61x45 crops end inside an MCU in both axes for every subsampling (16x16, 16x8, 8x8). The tiny ones
# are smaller than one MCU.
FIXTURES = [
    ("noise-q75-420", "noise-546.png", (200, 200, 61, 45), "RGB", 2, 75, False),
    ("noise-q75-420-prog", "noise-546.png", (200, 200, 61, 45), "RGB", 2, 75, True),
    ("noise-q95-444", "noise-546.png", (200, 200, 61, 45), "RGB", 0, 95, False),
    ("noise-q50-422", "noise-546.png", (200, 200, 61, 45), "RGB", 1, 50, False),
    ("checker-q75-420", "checker-1036.png", (500, 500, 61, 45), "RGB", 2, 75, False),
    ("checker-q90-444-prog", "checker-1036.png", (500, 500, 61, 45), "RGB", 0, 90, True),
    ("grad-q90-420", "grad-448.png", (100, 100, 61, 45), "RGB", 2, 90, False),
    ("mixed-q85-420", "odd-777x513.png", (230, 200, 61, 45), "RGB", 2, 85, False),
    ("noise-q85-grey", "noise-546.png", (200, 200, 61, 45), "L", 0, 85, False),
    ("tiny-1x1-420", "noise-546.png", (10, 10, 1, 1), "RGB", 2, 75, False),
    ("tiny-3x2-420", "noise-546.png", (10, 10, 3, 2), "RGB", 2, 75, False),
    ("tiny-17x9-420", "noise-546.png", (10, 10, 17, 9), "RGB", 2, 75, False),
]
# A progressive fixture and the baseline one it must decode identically to.
TWINS = {"noise-q75-420-prog": "noise-q75-420"}
SAMPLING = {0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    sys.path.append(SYSTEM_PACKAGES)

    import numpy as np
    import PIL
    from PIL import Image, JpegImagePlugin, features

    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    stale = sorted(p.name for p in out.iterdir())
    if stale:
        sys.exit(f"dump-jpeg: {out} is not empty ({', '.join(stale[:4])}…): the set is written whole or not at all")

    def decode(path, ycc):
        """The reference's decode of one file, as bytes; with `ycc`, libjpeg's YCbCr output instead."""
        with Image.open(path) as im:
            if ycc:
                im.draft("YCbCr", None)
                assert im.mode == "YCbCr", im.mode
                return np.asarray(im).tobytes()
            return np.asarray(im.convert("RGB")).tobytes()

    rows, refuse = [], []
    pixels = {}
    for name, src, (x, y, w, h), mode, sub, quality, progressive in FIXTURES:
        with Image.open(HERE / "images" / src) as im:
            assert im.mode == "RGB", f"{src}: {im.mode}"
            crop = im.crop((x, y, x + w, y + h)).convert(mode)
        jpg = out / f"{name}.jpg"
        crop.save(jpg, "JPEG", quality=quality, subsampling=sub, progressive=progressive)
        with Image.open(jpg) as back:
            got_mode, got_size = back.mode, back.size
            got_prog = bool(back.info.get("progressive", 0))
            got_sub = JpegImagePlugin.get_sampling(back) if mode == "RGB" else -1
        if (got_mode, got_size, got_prog) != (mode, (w, h), progressive) or (mode == "RGB" and got_sub != sub):
            sys.exit(f"dump-jpeg: {name}: wrote {got_mode} {got_size} progressive={got_prog} subsampling={got_sub}")
        rgb = decode(jpg, ycc=False)
        if rgb != decode(jpg, ycc=False):
            sys.exit(f"dump-jpeg: {name}: two decodes of one file differ")
        if len(rgb) != w * h * 3:
            sys.exit(f"dump-jpeg: {name}: {len(rgb)} RGB bytes for {w}x{h}")
        (out / f"{name}.rgb").write_bytes(rgb)
        pixels[name] = rgb
        ycc_sha = "-"
        if mode == "RGB":
            ycc = decode(jpg, ycc=True)
            if ycc != decode(jpg, ycc=True) or len(ycc) != w * h * 3:
                sys.exit(f"dump-jpeg: {name}: the YCbCr decode is not reproducible or not {w}x{h}x3")
            (out / f"{name}.ycc").write_bytes(ycc)
            ycc_sha = sha256(out / f"{name}.ycc")
        kind = ("progressive" if progressive else "baseline") + " " + (
            SAMPLING[sub] if mode == "RGB" else "grey")
        rows.append((name, src, x, y, w, h, kind, quality, sha256(jpg), sha256(out / f"{name}.rgb"), ycc_sha))
        print(f"{name:<22} {kind:<20} q{quality:<3} {w}x{h}  {jpg.stat().st_size} B")
    for prog, base in TWINS.items():
        if pixels[prog] != pixels[base]:
            sys.exit(f"dump-jpeg: {prog} and its baseline twin {base} decode to different pixels")

    with Image.open(HERE / "images" / "noise-546.png") as im:
        crop = im.crop((200, 200, 216, 216))
    cmyk = out / "refuse-cmyk.jpg"
    crop.convert("CMYK").save(cmyk, "JPEG", quality=90)
    with Image.open(cmyk) as back:
        assert back.mode == "CMYK", back.mode
    refuse.append(("refuse-cmyk.jpg", "CMYK (4 components)", sha256(cmyk)))
    rgbcoded = out / "refuse-rgbcoded.jpg"
    # Pillow ignores a save option it does not know, so the file is checked, not the call.
    crop.save(rgbcoded, "JPEG", quality=90, keep_rgb=True)
    with Image.open(rgbcoded) as back:
        transform = back.info.get("adobe_transform")
    if transform == 0:
        refuse.append(("refuse-rgbcoded.jpg", "RGB-coded (Adobe transform 0)", sha256(rgbcoded)))
    else:
        rgbcoded.unlink()
        print(f"dump-jpeg: this Pillow does not write an RGB-coded JPEG (adobe_transform {transform}); no such fixture")

    with open(out / "MANIFEST.tsv", "w") as f:
        f.write("# fixtures\ttools/ref/vision/dump-jpeg.py\n")
        f.write(f"# pillow\t{PIL.__version__}\t{Path(PIL.__file__).parent}\n")
        f.write(f"# libjpeg\t{features.version_codec('jpg')}\tlibjpeg-turbo {features.version('libjpeg_turbo')}\n")
        f.write(f"# numpy\t{np.__version__}\n")
        f.write("# decode\tImage.open(f).convert(\"RGB\") (.rgb); Image.open(f) with draft(\"YCbCr\", None) (.ycc)\n")
        f.write("# jpeg columns\tname source x y w h kind quality jpg_sha256 rgb_sha256 ycc_sha256\n")
        for r in rows:
            f.write("jpeg\t" + "\t".join(map(str, r)) + "\n")
        f.write("# refuse columns\tfile what sha256\n")
        for r in refuse:
            f.write("refuse\t" + "\t".join(r) + "\n")
        f.write("# twin columns\tprogressive baseline\n")
        for prog, base in TWINS.items():
            f.write(f"twin\t{prog}\t{base}\n")
        f.write(f"# complete\t{len(rows)}\t{len(refuse)}\n")
    total = sum(p.stat().st_size for p in out.iterdir())
    print(f"dump-jpeg: {len(rows)} fixtures, {len(refuse)} refusal files, {total} B, Pillow {PIL.__version__}, "
          f"libjpeg {features.version_codec('jpg')} (libjpeg-turbo {features.version('libjpeg_turbo')})")


if __name__ == "__main__":
    main()
