#!/usr/bin/env python3
"""The V4.1 vision oracle: the official checkpoint's own preprocessing, ViT and aligner, dumped per image.

Runs the reference code itself, `inference/image_processor.py` and `inference/vision.py` of
deepseek-ai/DeepSeek-V4.1-Flash, on the files under --images, and writes one set into --out:

  <image>.rgb.u8        [best_h, best_w, 3] u8  the resized and padded image load_image normalizes
  <image>.patches.bf16  [n_vit, 588] bf16       load_image's patch tensor (c, py, px inside a patch)
  <image>.vit.bf16      [n_vit, 1024] bf16      ViT output after its final RMSNorm
  <image>.aligner.bf16  [n_llm_h * n_llm_w, 5120] bf16  aligner rows in reading order
  <image>.types.i32     [n_tokens] i32          image_token_types (IMAGE_START 0, IMAGE 1, NEW_LINE 2, END 3)
  <image>.ids.i32       [n_tokens] i32          the span's input_ids: image_token_id at every position
  delims.bf16           [3, 5120] bf16          image_start, image_end, image_newline as merge casts them
  plans.tsv             the resize plan and the pad geometry of the images and of a table of sizes
  MANIFEST.tsv          where the set came from, one row per image and per file, `# complete` last

Nothing in the set depends on the time or the host beyond what MANIFEST names, so two runs are
byte-identical files (dump-vision.sh runs it twice and compares before installing).

The model is built the way generate.py builds it: torch.set_default_dtype(torch.bfloat16) before the
modules, on the card, RMSNorm gains in f32 as vision.py declares them, then the card as the default
device. model.py is not imported (it
needs tilelang for the text path); its ViT/Aligner construction is those two lines of vision.py.

The pad geometry (the size contain() resizes to and where pad() pastes it) is read off the reference
rather than recomputed: a black image of the same size goes through the same ImageOps.pad call, and
the bounding box of the pixels that are not the grey fill is where the resized image landed.
Bicubic weights are normalized, so black stays exactly 0.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

# The checkpoint this oracle is of: the HF revision, the LFS sha256 of the two shards it reads and the
# git blob ids of the three reference files (HF tree API at that revision). A tree that differs is refused.
REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"
SHARDS = {
    "model-00001-of-00048.safetensors": "886aebdafa08cc27bbae2165ed35bdfe0de9370bf88c1411283c155c6ae4ff89",
    "model-00002-of-00048.safetensors": "4320066fc6958e5bc01d8c3feba79b7454b59f0f4b7299ab7145ed44bbf4ecec",
}
CODE = {
    "inference/vision.py": "77af0bdef9f4649edebac2876cd1df50a147ce98",
    "inference/image_processor.py": "a50311ece615b72be2c9d4b649a05619c27681e7",
    "inference/config.json": "7a915cc69e21abbc6d7fb939cef5d09c72aa0d24",
}
# Pillow is not in the torch environment; the system one is appended after it, so numpy and torch
# still come from that environment.
SYSTEM_PACKAGES = "/usr/lib/python3/dist-packages"

# Sizes whose plan is dumped besides the images': both collapse branches of solve_resize_ratio, the
# min-pixels upscale, a multiple of 14 and its neighbours, the budget edge, common camera sizes.
SIZES = [
    (1, 1), (13, 13), (14, 14), (15, 15), (448, 448), (543, 543), (544, 544), (545, 545),
    (546, 546), (547, 547), (1035, 1035), (1036, 1036), (1037, 1037), (800, 600), (1920, 1080),
    (1080, 1920), (3840, 2160), (4000, 3000), (3000, 4000), (2000, 500), (500, 2000),
    (10, 6000), (6000, 10), (1, 700), (8000, 1), (1, 8000),
]


def git_blob(path):
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--mmproj", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    inference = a.ckpt / "inference"
    sys.path.insert(0, str(inference))
    sys.path.append(SYSTEM_PACKAGES)

    import numpy as np
    import PIL
    import torch
    from PIL import Image, ImageOps
    from safetensors import safe_open

    import image_processor as ip
    import vision

    for rel, want in CODE.items():
        got = git_blob(a.ckpt / rel)
        if got != want:
            sys.exit(f"dump_vision: {rel} has git blob {got}, revision {REVISION} has {want}")
    shard_sha = {}
    for name, want in SHARDS.items():
        got = sha256_file(a.ckpt / name)
        if got != want:
            sys.exit(f"dump_vision: {name} has sha256 {got}, revision {REVISION} has {want}")
        shard_sha[name] = got
    mmproj_sha = sha256_file(a.mmproj)

    with open(inference / "config.json") as f:
        args = SimpleNamespace(**json.load(f))

    if not torch.cuda.is_available():
        sys.exit("dump_vision: no CUDA device visible")
    dev = torch.device("cuda")
    torch.set_default_dtype(torch.bfloat16)
    with dev:
        vit = vision.ViT(args)
        aligner = vision.Aligner(args)
    shard1 = a.ckpt / "model-00001-of-00048.safetensors"
    with safe_open(str(shard1), "pt", device="cuda") as f:
        keys = list(f.keys())
        vit_sd = {k[len("vision."):]: f.get_tensor(k) for k in keys if k.startswith("vision.")}
        al_sd = {k[len("aligner."):]: f.get_tensor(k) for k in keys if k.startswith("aligner.")}
    vit.load_state_dict(vit_sd, strict=True)
    aligner.load_state_dict(al_sd, strict=True)
    vit.eval()
    aligner.eval()
    with safe_open(str(a.ckpt / "model-00002-of-00048.safetensors"), "pt", device="cpu") as f:
        delims = [f.get_tensor(k) for k in ("image_start", "image_end", "image_newline")]
    delim_dtype = str(delims[0].dtype)
    # generate.py sets the default device after loading; vision.get_vision_cos_sin builds its tables there.
    torch.set_default_device("cuda")

    out = a.out
    out.mkdir(parents=True, exist_ok=False)
    files = []

    def write(name, kind, arr, dtype, shape):
        data = arr.tobytes()
        (out / name).write_bytes(data)
        files.append((name, kind, dtype, "x".join(map(str, shape)), len(data), hashlib.md5(data).hexdigest()))

    def bf16_bytes(t):
        t = t.detach().to(torch.bfloat16).contiguous().cpu()
        return t.view(torch.int16).numpy().astype("<i2")

    def geometry(w, h, best_w, best_h):
        """contain()'s size and pad()'s paste offset, read off ImageOps.pad on a black image."""
        padded = np.asarray(
            ImageOps.pad(Image.new("RGB", (w, h), (0, 0, 0)), (best_w, best_h), color=(127, 127, 127))
        )
        inside = np.argwhere((padded != 127).any(axis=-1))
        (y0, x0), (y1, x1) = inside.min(axis=0), inside.max(axis=0)
        return int(x1 - x0 + 1), int(y1 - y0 + 1), int(x0), int(y0)

    write("delims.bf16", "delims", bf16_bytes(torch.stack(delims)), "bf16", (3, args.dim))

    plan_rows = []
    image_rows = []
    for path in sorted(a.images.glob("*.png")):
        stem = path.stem
        raw = path.read_bytes()
        with Image.open(path) as src:
            if src.mode != "RGB":
                sys.exit(f"dump_vision: {path.name} is {src.mode}, the set is 8-bit RGB only")
            w, h = src.size
        patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = ip.load_image({"url": str(path)}, args)
        plan = ip.plan_image_grid(w, h, args)
        assert plan[:2] == (n_llm_h, n_llm_w), (plan, n_llm_h, n_llm_w)
        _, _, best_h, best_w = plan
        # The u8 image load_image normalized, by the same calls; it must normalize to its patches.
        with Image.open(path) as src:
            rgb = np.asarray(ImageOps.pad(src.convert("RGB"), (best_w, best_h), color=(127, 127, 127)))
        x = torch.from_numpy(rgb.astype(np.float32)).permute(2, 0, 1) / 255
        x = ((x - 0.5) / 0.5).to(torch.bfloat16)
        p = args.vision_patch_size
        again = x.reshape(3, n_vit_h, p, n_vit_w, p).permute(1, 3, 0, 2, 4).reshape(n_vit_h * n_vit_w, 3, p, p)
        assert torch.equal(again.view(torch.int16), patches.view(torch.int16)), stem
        types = ip.image_token_types(n_llm_h, n_llm_w)
        n_tok = types.numel()
        assert n_tok == ip.num_image_tokens(n_llm_h, n_llm_w)
        with torch.inference_mode():
            feats = vit(patches.to(dev), n_vit_h, n_vit_w)
            rows = aligner(feats, n_vit_h, n_vit_w)
        assert tuple(feats.shape) == (n_vit_h * n_vit_w, args.vision_dim), feats.shape
        assert tuple(rows.shape) == (n_llm_h * n_llm_w, args.dim), rows.shape
        n_vit = n_vit_h * n_vit_w
        write(f"{stem}.rgb.u8", "rgb", rgb.astype(np.uint8), "u8", (best_h, best_w, 3))
        write(f"{stem}.patches.bf16", "patches", bf16_bytes(patches.reshape(n_vit, -1)), "bf16", (n_vit, 3 * p * p))
        write(f"{stem}.vit.bf16", "vit", bf16_bytes(feats), "bf16", (n_vit, args.vision_dim))
        write(f"{stem}.aligner.bf16", "aligner", bf16_bytes(rows), "bf16", (n_llm_h * n_llm_w, args.dim))
        write(f"{stem}.types.i32", "types", types.cpu().numpy().astype("<i4"), "i32", (n_tok,))
        ids = np.full(n_tok, args.image_token_id, dtype="<i4")
        write(f"{stem}.ids.i32", "ids", ids, "i32", (n_tok,))
        geo = geometry(w, h, best_w, best_h)
        image_rows.append((path.name, hashlib.sha256(raw).hexdigest(), w, h, best_w, best_h, n_vit_h, n_vit_w,
                           n_llm_h, n_llm_w, n_tok, *geo))
        plan_rows.append(("image", w, h, n_llm_h, n_llm_w, best_h, best_w, n_tok, *geo))
    if not image_rows:
        sys.exit(f"dump_vision: no *.png under {a.images}")

    for w, h in SIZES:
        n_llm_h, n_llm_w, best_h, best_w = ip.plan_image_grid(w, h, args)
        plan_rows.append(("size", w, h, n_llm_h, n_llm_w, best_h, best_w, ip.num_image_tokens(n_llm_h, n_llm_w),
                          *geometry(w, h, best_w, best_h)))
    with open(out / "plans.tsv", "w") as f:
        f.write("# kind\tw\th\tn_llm_h\tn_llm_w\tbest_h\tbest_w\tn_tokens\tresized_w\tresized_h\toff_x\toff_y\n")
        for r in plan_rows:
            f.write("\t".join(map(str, r)) + "\n")

    props = torch.cuda.get_device_properties(dev)
    with open(out / "MANIFEST.tsv", "w") as f:
        f.write(f"# oracle\tdump_vision.py\n")
        f.write(f"# checkpoint\tdeepseek-ai/DeepSeek-V4.1-Flash@{REVISION}\t{a.ckpt}\n")
        for name, sha in shard_sha.items():
            f.write(f"# shard\t{name}\tsha256\t{sha}\n")
        for rel, blob in CODE.items():
            f.write(f"# code\t{rel}\tgit-blob\t{blob}\n")
        f.write(f"# torch\t{torch.__version__}\tcuda {torch.version.cuda}\n")
        f.write(f"# numpy\t{np.__version__}\n")
        f.write(f"# pillow\t{PIL.__version__}\t{Path(PIL.__file__).parent}\n")
        f.write(f"# device\tcuda\t{props.name}\tsm_{props.major}{props.minor}\n")
        f.write("# dtype\tbfloat16 default, RMSNorm gains f32 (generate.py, vision.py)\n")
        f.write(f"# bf16_reduced_precision_reduction\t{torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}\n")
        f.write(f"# delims\timage_start image_end image_newline\tcheckpoint {delim_dtype}\n")
        f.write(f"# mmproj\t{a.mmproj}\tsha256\t{mmproj_sha}\n")
        f.write(f"# image_token_id\t{args.image_token_id}\n")
        f.write("# image columns\tname sha256 w h best_w best_h n_vit_h n_vit_w n_llm_h n_llm_w n_tokens "
                "resized_w resized_h off_x off_y\n")
        for r in image_rows:
            f.write("image\t" + "\t".join(map(str, r)) + "\n")
        f.write("# file columns\tname kind dtype shape bytes md5\n")
        for r in files:
            f.write("file\t" + "\t".join(map(str, r)) + "\n")
        f.write(f"# complete\t{len(image_rows)}\t{len(files)}\t{len(plan_rows)}\n")
    print(f"dump_vision: {len(image_rows)} images, {len(files)} files, {len(plan_rows)} plans, device {props.name}")


if __name__ == "__main__":
    main()
