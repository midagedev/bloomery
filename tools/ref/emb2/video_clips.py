#!/usr/bin/env python3
"""Prepare constant-rate decoded RGB clips and selected PNGs for the video oracle."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    clips = []
    for name, seconds, fps, width, height, first in (
        ("clip_1s24", 1, 24, 200, 200, 11),
        ("clip_7s30", 7, 30, 384, 256, 14),
        ("clip_16s24", 16, 24, 256, 384, 11),
    ):
        out = a.out / name
        out.mkdir()
        indices = list(range(first, seconds * fps, fps))
        selected = set(indices)
        digest = hashlib.sha256()
        images = []
        with (out / "frames.rgb").open("wb") as stream:
            for index in range(seconds * fps):
                image = Image.new("RGB", (width, height), (index % 256, index // 256, 127))
                x = index % width
                image.paste((255, 255, 255), (x, 0, x + 1, height))
                raw = image.tobytes()
                stream.write(raw)
                digest.update(raw)
                if index in selected:
                    png = out / f"frame_{index}.png"
                    image.save(png)
                    images.append(str(png.resolve()))
        clips.append({
            "name": name,
            "source_fps": [fps, 1],
            "source_frames": seconds * fps,
            "width": width,
            "height": height,
            "rgb": str((out / "frames.rgb").resolve()),
            "sha256": digest.hexdigest(),
            "indices": indices,
            "images": images,
        })
    manifest = a.out / "clips.json"
    manifest.write_text(json.dumps(clips, indent=2) + "\n")
    (a.out / "clips.tsv").write_text("".join(
        "\t".join(map(str, [c["name"], *c["source_fps"], c["width"], c["height"],
                            c["source_frames"], c["rgb"], c["sha256"],
                            ",".join(map(str, c["indices"]))])) + "\n"
        for c in clips
    ))
    print(f"video_clips: {len(clips)} decoded clips into {manifest}")


if __name__ == "__main__":
    main()
