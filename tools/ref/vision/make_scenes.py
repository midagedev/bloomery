#!/usr/bin/env python3
"""Write the scene images of the V4.1 vision oracle's fork comparison into scenes/ beside this file.

Three images a person would send, each asking the model for something else to see:

  chart-bars.png     800x560   a bar chart drawn here: four labelled bars, a title, a value axis (read numbers)
  screen-term.png    1000x500  a terminal window drawn here: monospace lines of a shell session (read text)
  photo-toaster.png  960x661   a photograph: an antique toaster with a catalogue label and a grey card (see an object)
  grad-448.png       448x448   a copy of images/grad-448.png: dump_vision.py's full-tap image, which every set holds

The photograph is NPS's, public domain (NPGallery; Wikimedia Commons, "Now sought after as a "vintage"
collector's piece, this beautifully designed chrome Westinghouse Turnover Toaster was once
(10967a7f-1dd8-b71b-0bb0-0dd91871041e).jpg", its 960 px rendition), decoded by Pillow and stored as 8-bit RGB PNG;
`--photo JPEG` names that file. The chart and the terminal are drawn with Pillow in the system fonts this script
names, so another machine draws other pixels: the committed files are the images, and dump_vision.py records each
file's sha256.
"""

import argparse
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
OUT = HERE / "scenes"
SANS = "/System/Library/Fonts/Helvetica.ttc"
MONO = "/System/Library/Fonts/Menlo.ttc"

BARS = [("North", 42), ("South", 27), ("East", 35), ("West", 18)]

TERMINAL = [
    "$ ls -l reports/",
    "total 12",
    "-rw-r--r-- 1 ana staff 2048 Mar  3 09:14 budget-2024.csv",
    "-rw-r--r-- 1 ana staff 5120 Mar  3 09:20 summary.txt",
    "$ wc -l reports/summary.txt",
    "87 reports/summary.txt",
    "$ grep -c ERROR app.log",
    "14",
    "$ echo done",
    "done",
]


def chart(path):
    w, h = 800, 560
    im = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(im)
    title = ImageFont.truetype(SANS, 30)
    label = ImageFont.truetype(SANS, 24)
    small = ImageFont.truetype(SANS, 20)
    d.text((w // 2, 34), "Apples sold per region (crates)", font=title, fill=(20, 20, 20), anchor="mm")
    x0, y0, x1, y1 = 110, 90, 760, 470
    top = 50
    for v in range(0, top + 1, 10):
        y = y1 - (y1 - y0) * v / top
        d.line([(x0, y), (x1, y)], fill=(225, 225, 225), width=1)
        d.text((x0 - 12, y), str(v), font=small, fill=(60, 60, 60), anchor="rm")
    d.line([(x0, y0), (x0, y1), (x1, y1)], fill=(40, 40, 40), width=2)
    slot = (x1 - x0) / len(BARS)
    colors = [(66, 114, 196), (237, 125, 49), (112, 173, 71), (165, 105, 189)]
    for i, ((name, v), c) in enumerate(zip(BARS, colors)):
        cx = x0 + slot * (i + 0.5)
        yt = y1 - (y1 - y0) * v / top
        d.rectangle([cx - 50, yt, cx + 50, y1], fill=c)
        d.text((cx, yt - 16), str(v), font=label, fill=(20, 20, 20), anchor="mm")
        d.text((cx, y1 + 24), name, font=label, fill=(20, 20, 20), anchor="mm")
    d.text((40, (y0 + y1) // 2), "crates", font=small, fill=(60, 60, 60), anchor="mm")
    im.save(path, optimize=True)


def terminal(path):
    w, h = 1000, 500
    im = Image.new("RGB", (w, h), (30, 30, 30))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, w, 36], fill=(58, 58, 58))
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([16 + 26 * i, 11, 30 + 26 * i, 25], fill=c)
    bar = ImageFont.truetype(SANS, 18)
    d.text((w // 2, 18), "ana@laptop: ~/project", font=bar, fill=(210, 210, 210), anchor="mm")
    mono = ImageFont.truetype(MONO, 22)
    for i, line in enumerate(TERMINAL):
        fill = (120, 220, 120) if line.startswith("$") else (230, 230, 230)
        d.text((20, 52 + 43 * i), line, font=mono, fill=fill)
    im.save(path, optimize=True)


def photo(src, path):
    with Image.open(src) as im:
        rgb = im.convert("RGB")
    if rgb.size != (960, 661):
        raise SystemExit(f"make_scenes: {src} is {rgb.size}, the 960 px rendition is 960x661")
    rgb.save(path, optimize=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", type=Path, required=True, help="the NPS photograph's 960 px JPEG")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    chart(OUT / "chart-bars.png")
    terminal(OUT / "screen-term.png")
    photo(a.photo, OUT / "photo-toaster.png")
    shutil.copyfile(HERE / "images" / "grad-448.png", OUT / "grad-448.png")
    for p in sorted(OUT.glob("*.png")):
        with Image.open(p) as im:
            print(f"{p.name}\t{im.size[0]}x{im.size[1]}\t{im.mode}\t{p.stat().st_size} B")


if __name__ == "__main__":
    main()
