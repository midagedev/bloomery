#!/usr/bin/env python3
"""The V4.1 vision oracle: the official checkpoint's own preprocessing, ViT and aligner, dumped per image.

Runs the reference code itself, `inference/image_processor.py` and `inference/vision.py` of
deepseek-ai/DeepSeek-V4.1-Flash, on the files under --images, and writes one set into --out:

  <image>.rgb.u8        [best_h, best_w, 3] u8  the resized and padded image load_image normalizes
  <image>.patches.bf16  [n_vit, 588] bf16       load_image's patch tensor (c, py, px inside a patch)
  <image>.embed.bf16    [n_vit, 1024] bf16      patch embedding (PatchEmbed.proj) — the input of block 0; the ViT
                                                adds no position here, its 2D RoPE turns q and k inside each block
  <image>.blk<i>.bf16   [n_vit, 1024] bf16      the residual stream after block i: blocks 0, 1, 15 and 31, and every
                                                block of the full-tap image (FULL_TAP_IMAGE)
  <image>.vit.bf16      [n_vit, 1024] bf16      ViT output after its final RMSNorm
  <image>.aligner.bf16  [n_llm_h * n_llm_w, 5120] bf16  aligner rows in reading order
  <image>.types.i32     [n_tokens] i32          image_token_types (IMAGE_START 0, IMAGE 1, NEW_LINE 2, END 3)
  <image>.ids.i32       [n_tokens] i32          the span's input_ids: image_token_id at every position
  delims.bf16           [3, 5120] bf16          image_start, image_end, image_newline as merge casts them

The full-tap image also gets block 0 op by op and the aligner's two linears, each the output of one
module of vision.py (a forward hook; a pre-hook for a module's input), so a gate that fails on block 0
names the op:
  <image>.blk0.norm1.bf16 [n_vit, 1024]   Block.norm1                <image>.blk0.qkv.bf16  [n_vit, 3072]  Attention.wqkv
  <image>.blk0.qrot.bf16  [n_vit, 1024]   q after apply_rotary       <image>.blk0.krot.bf16 [n_vit, 1024]  k after it
  <image>.blk0.sdpa.bf16  [n_vit, 1024]   the input of Attention.wo  <image>.blk0.attn.bf16 [n_vit, 1024]  Attention
  <image>.blk0.norm2.bf16 [n_vit, 1024]   Block.norm2                <image>.blk0.w1.bf16   [n_vit, 5632]  MLP.w1
  <image>.blk0.act.bf16   [n_vit, 2816]   the input of MLP.w2        <image>.blk0.mlp.bf16  [n_vit, 1024]  MLP
  <image>.aligner.w1.bf16 [n_llm, 5120]   Aligner.w1                 <image>.aligner.h.bf16 [n_llm, 5120]  input of Aligner.w2
qrot and krot are recomputed from the qkv tap by vision.py's own get_vision_cos_sin and apply_rotary on
the same tensors, which is the computation Attention.forward runs.

MANIFEST also records, for the full-tap image, how the reference rounds (the premise of the gates'
bands): `# sdpa` names every SDPA backend that reproduces the unrestricted call's output bit for bit on
block 0 (and those that refuse the 3-D call), and each `# probe` row counts the values of one op's
output that differ from the exactly rounded result of the same inputs — the op computed in float64 on
the card, rounded to bf16 to nearest even in float64 — with the largest distance in bf16 steps. `# sensitivity` rows run the reference against itself with
one-ulp noise in the full-tap image's patch embedding (SENSITIVITY_RATE of its values), per tap: how
far the network itself carries a difference of the size a correct kernel makes at its first op.
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
# The blocks whose output every image dumps, and the image that dumps every block plus block 0 op by op.
TAP_BLOCKS = (0, 1, 15, 31)
FULL_TAP_IMAGE = "grad-448"
# The fraction of patch-embedding values whose last mantissa bit the sensitivity probe flips: the
# fraction a correct kernel's patch GEMM differs from torch's in, measured.
SENSITIVITY_RATE = 1.5e-4
# The full-tap image's op-level taps, in file order.
SUB_TAPS = ("blk0.norm1", "blk0.qkv", "blk0.qrot", "blk0.krot", "blk0.sdpa", "blk0.attn", "blk0.norm2",
            "blk0.w1", "blk0.act", "blk0.mlp", "aligner.w1", "aligner.h")

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


def attach_taps(vit, aligner, full):
    """Forward hooks that keep a copy of each tapped tensor; returns (store, handles)."""
    store = {}
    handles = []

    def out_hook(name):
        def hook(_mod, _args, out):
            store[name] = out.detach().clone()
        return hook

    def in_hook(name):
        def hook(_mod, args):
            store[name] = args[0].detach().clone()
        return hook

    handles.append(vit.patch_embed.register_forward_hook(out_hook("embed")))
    for i, blk in enumerate(vit.blocks):
        if full or i in TAP_BLOCKS:
            handles.append(blk.register_forward_hook(out_hook(f"blk{i}")))
    handles.append(vit.norm.register_forward_hook(out_hook("vit")))
    if full:
        b0 = vit.blocks[0]
        handles += [
            b0.norm1.register_forward_hook(out_hook("blk0.norm1")),
            b0.attn.wqkv.register_forward_hook(out_hook("blk0.qkv")),
            b0.attn.wo.register_forward_pre_hook(in_hook("blk0.sdpa")),
            b0.attn.register_forward_hook(out_hook("blk0.attn")),
            b0.norm2.register_forward_hook(out_hook("blk0.norm2")),
            b0.mlp.w1.register_forward_hook(out_hook("blk0.w1")),
            b0.mlp.w2.register_forward_pre_hook(in_hook("blk0.act")),
            b0.mlp.register_forward_hook(out_hook("blk0.mlp")),
            aligner.w1.register_forward_pre_hook(in_hook("aligner.x")),
            aligner.w1.register_forward_hook(out_hook("aligner.w1")),
            aligner.w2.register_forward_pre_hook(in_hook("aligner.h")),
            aligner.w2.register_forward_hook(out_hook("aligner.out")),
        ]
    return store, handles


def bf16_steps(np, got, exact):
    """Per value, |got - round_bf16(exact)| in bf16 steps at the exact value's binade (float64)."""
    exact = exact.astype(np.float64)
    _, e = np.frexp(exact)
    step = np.ldexp(1.0, e - 8)  # 8 significant bits
    rounded = np.rint(exact / step) * step
    return np.abs(got.astype(np.float64) - rounded) / step


def probe_rows(np, torch, F, vit, aligner, store, patches, eps):
    """How the reference rounds, on the full-tap image: which SDPA backend ran, and per op the values
    that differ from the exactly rounded result of the op's own (bf16) inputs."""
    rows = []
    b0 = vit.blocks[0]
    n = patches.shape[0]
    f64 = torch.float64

    def row(op, k, got, exact):
        g = got.float().cpu().numpy()
        steps = bf16_steps(np, g, exact.cpu().numpy())
        rows.append((op, k, g.size, int((steps > 0).sum()), f"{float(steps.max()):g}"))

    def linear(mod, x):
        y = x.to(f64) @ mod.weight.to(f64).T
        return y + mod.bias.to(f64) if mod.bias is not None else y

    def rms(mod, x):
        x = x.to(f64)
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * mod.weight.to(f64)

    row("patch_embed", patches.flatten(1).shape[1], store["embed"], linear(vit.patch_embed.proj, patches.flatten(1)))
    row("blk0.norm1", 1024, store["blk0.norm1"], rms(b0.norm1, store["embed"]))
    row("blk0.wqkv", 1024, store["blk0.qkv"], linear(b0.attn.wqkv, store["blk0.norm1"]))
    q = store["blk0.qrot"].view(n, 16, 64).transpose(0, 1)
    k = store["blk0.krot"].view(n, 16, 64).transpose(0, 1)
    v = store["blk0.qkv"].chunk(3, dim=-1)[2].reshape(n, 16, 64).transpose(0, 1)
    s = (q.to(f64) @ k.to(f64).transpose(1, 2)) * 64 ** -0.5
    o = (torch.softmax(s, dim=-1) @ v.to(f64)).transpose(0, 1).reshape(n, -1)
    row("blk0.sdpa", n, store["blk0.sdpa"], o)
    row("blk0.wo", 1024, store["blk0.attn"], linear(b0.attn.wo, store["blk0.sdpa"]))
    row("blk0.norm2", 1024, store["blk0.norm2"], rms(b0.norm2, store["embed"] + store["blk0.attn"]))
    row("blk0.w1", 1024, store["blk0.w1"], linear(b0.mlp.w1, store["blk0.norm2"]))
    g, u = store["blk0.w1"].to(f64).chunk(2, dim=-1)
    row("blk0.silu_mul", 1, store["blk0.act"], g * torch.sigmoid(g) * u)
    row("blk0.w2", 2816, store["blk0.mlp"], linear(b0.mlp.w2, store["blk0.act"]))
    row("aligner.w1", 9216, store["aligner.w1"], linear(aligner.w1, store["aligner.x"]))
    row("aligner.gelu", 1, store["aligner.h"], F.gelu(store["aligner.w1"].to(f64)))
    row("aligner.w2", 5120, store["aligner.out"], linear(aligner.w2, store["aligner.h"]))

    from torch.nn.attention import SDPBackend, sdpa_kernel
    base = F.scaled_dot_product_attention(q, k, v)
    sdpa = []
    for name in ("MATH", "FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION"):
        try:
            with sdpa_kernel(getattr(SDPBackend, name)):
                out = F.scaled_dot_product_attention(q, k, v)
            same = torch.equal(out.view(torch.int16), base.view(torch.int16))
            sdpa.append(f"{name.lower()} {'same' if same else 'differs'}")
        except RuntimeError:
            sdpa.append(f"{name.lower()} refused")
    return rows, sdpa


def tap_stats(torch, got, ref):
    """max|d|/max|ref|, rms(d)/rms(ref) and the fraction of values that differ, of two bf16 tensors."""
    g, r = got.double(), ref.double()
    d = (g - r).abs()
    return (float(d.max() / r.abs().max()), float((d.square().sum() / r.square().sum()).sqrt()),
            float((got.view(torch.int16) != ref.view(torch.int16)).double().mean()))


def sensitivity_rows(torch, vision, vit, aligner, store, n_h, n_w, rate):
    """The reference's own amplification of one-ulp noise: the patch embedding with the last
    mantissa bit of a fraction `rate` of its values flipped (a fixed CPU generator picks them), run
    through the official blocks, final norm and aligner, each tap against the unperturbed run."""
    g = torch.Generator().manual_seed(0)
    embed = store["embed"]
    mask = (torch.rand(tuple(embed.shape), generator=g, device="cpu", dtype=torch.float32) < rate).to(embed.device)
    bits = embed.view(torch.int16)
    x = torch.where(mask, bits ^ 1, bits).view(torch.bfloat16)
    rows = [("embed", *tap_stats(torch, x, embed))]
    cos, sin = vision.get_vision_cos_sin(n_h, n_w, vit.rope_dim, vit.rope_theta)
    for i, blk in enumerate(vit.blocks):
        x = blk(x, cos, sin)
        rows.append((f"blk{i}", *tap_stats(torch, x, store[f"blk{i}"])))
    feats = vit.norm(x)
    rows.append(("vit", *tap_stats(torch, feats, store["vit"])))
    rows.append(("aligner", *tap_stats(torch, aligner(feats, n_h, n_w), store["aligner.out"])))
    return rows


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
    import torch.nn.functional as F
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
    probes, sdpa, sens = None, None, None
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
        full = stem == FULL_TAP_IMAGE
        store, handles = attach_taps(vit, aligner, full)
        with torch.inference_mode():
            feats = vit(patches.to(dev), n_vit_h, n_vit_w)
            rows = aligner(feats, n_vit_h, n_vit_w)
            for h_ in handles:
                h_.remove()
            if full:
                cos, sin = vision.get_vision_cos_sin(n_vit_h, n_vit_w, vit.rope_dim, vit.rope_theta)
                n_q = n_vit_h * n_vit_w
                heads, hd = args.vision_n_heads, args.vision_dim // args.vision_n_heads
                q, k, _ = store["blk0.qkv"].chunk(3, dim=-1)
                store["blk0.qrot"] = vision.apply_rotary(q.reshape(n_q, heads, hd), cos, sin).reshape(n_q, -1)
                store["blk0.krot"] = vision.apply_rotary(k.reshape(n_q, heads, hd), cos, sin).reshape(n_q, -1)
                probes, sdpa = probe_rows(np, torch, F, vit, aligner, store, patches.to(dev),
                                          vit.blocks[0].norm1.eps)
                sens = sensitivity_rows(torch, vision, vit, aligner, store, n_vit_h, n_vit_w, SENSITIVITY_RATE)
        assert tuple(feats.shape) == (n_vit_h * n_vit_w, args.vision_dim), feats.shape
        assert tuple(rows.shape) == (n_llm_h * n_llm_w, args.dim), rows.shape
        n_vit = n_vit_h * n_vit_w
        write(f"{stem}.rgb.u8", "rgb", rgb.astype(np.uint8), "u8", (best_h, best_w, 3))
        write(f"{stem}.patches.bf16", "patches", bf16_bytes(patches.reshape(n_vit, -1)), "bf16", (n_vit, 3 * p * p))
        write(f"{stem}.embed.bf16", "embed", bf16_bytes(store["embed"]), "bf16", (n_vit, args.vision_dim))
        for i in range(args.vision_n_layers):
            if f"blk{i}" in store:
                write(f"{stem}.blk{i}.bf16", f"blk{i}", bf16_bytes(store[f"blk{i}"]), "bf16", (n_vit, args.vision_dim))
        if full:
            for tap in SUB_TAPS:
                t = store[tap]
                write(f"{stem}.{tap}.bf16", tap, bf16_bytes(t), "bf16", tuple(t.shape))
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
    if probes is None:
        sys.exit(f"dump_vision: no {FULL_TAP_IMAGE}.png under {a.images}, the full-tap image")

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
        f.write(f"# taps\tblocks {' '.join(map(str, TAP_BLOCKS))} of every image; every block and "
                f"{' '.join(SUB_TAPS)} of {FULL_TAP_IMAGE}\n")
        f.write(f"# sdpa\t{FULL_TAP_IMAGE} block 0, backends against the unrestricted call\t" + "\t".join(sdpa) + "\n")
        f.write(f"# sensitivity columns\ttap max_rel rms_rel differ\tthe reference against itself with the last "
                f"mantissa bit of {SENSITIVITY_RATE:g} of {FULL_TAP_IMAGE}'s patch embedding flipped\n")
        for r in sens:
            f.write("# sensitivity\t" + r[0] + "\t" + "\t".join(f"{v:.3e}" for v in r[1:]) + "\n")
        f.write("# probe columns\top K values differ_from_exact_rounding max_bf16_steps\n")
        for r in probes:
            f.write("# probe\t" + "\t".join(map(str, r)) + "\n")
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
