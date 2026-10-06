#!/usr/bin/env python3
"""The V4.1 vision fork comparison's cases: the ids and image rows both engines are fed (visref.sh runs it).

For each row of the case list (name, image, question) it renders one user turn holding the image and the question
with the official template (`encoding/encoding.py` of the checkpoint, chat mode: `<bos><User><image>\\n\\n<question>
<Assistant></think>`), tokenizes it with the checkpoint's tokenizer the way `inference/generate.py` does
(`tokenizer.encode`), and expands the one image placeholder into the image's span as `image_processor.
prepare_vl_inputs` does: the span's positions all hold `image_token_id`, its rows are the vision set's — `image_start`,
the aligner rows in reading order with `image_newline` after each grid row, `image_end` (`delims.bf16`,
`<image>.aligner.bf16`, `<image>.types.i32`). Three cases per row:

  <name>        the image case
  <name>-text   the same ids with the span's positions removed (the text-only control)
  <name>-prose  the span's positions holding as many ids of the prose corpus instead (the control at the image's depth)

`cases` writes into --out: cases.tsv (`name n_ids span_at span_len gen kind image question`, the first five what
visref_fork reads), and per case <name>.ids.i32, and for an image case <name>.rows.bf16 and <name>.types.i32. The
official files are refused unless they are revision dba1be0a's (git blob ids, HF tree API at that revision).

`manifest` writes the set's MANIFEST.tsv once visref_fork has written each case's answer and logits beside its
inputs: the identity lines (the model file, the fork's commit, the vision set and its checkpoint), the case rows
with the fork's `case` lines (answer length, eog, logits rows), one file row per file with its md5, `# complete`
last. It refuses a case the fork's log does not report, or whose files are missing.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

CODE = {
    "encoding/encoding.py": "92a67eab2a67d924a6a8279bc0f90e3f504ac40e",
    "tokenizer.json": "6a15814dd25c934028034531da744c689f87ff21",
    "tokenizer_config.json": "f3dad388a2bbfd6a8605bd02754acd86d9ca5112",
    "inference/config.json": "7a915cc69e21abbc6d7fb939cef5d09c72aa0d24",
}
KINDS = {0: "start", 1: "image", 2: "newline", 3: "end"}


def git_blob(path):
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def manifest(set_dir):
    """The vision set's file rows by name, and its # lines by key."""
    files, heads = {}, {}
    cols = None
    for line in (set_dir / "MANIFEST.tsv").read_text().splitlines():
        f = line.split("\t")
        if f[0] == "# file columns":
            cols = ["row"] + f[1].split(" ")
        elif f[0] == "file":
            row = dict(zip(cols, f))
            files[row["name"]] = row
        elif f[0].startswith("# "):
            heads[f[0][2:]] = f[1:]
    if "complete" not in heads:
        sys.exit(f"visref_cases: {set_dir} has no # complete trailer")
    return files, heads


def cases(a):
    import numpy as np
    from transformers import AutoTokenizer

    for rel, want in CODE.items():
        got = git_blob(a.ckpt / rel)
        if got != want:
            sys.exit(f"visref_cases: {rel} has git blob {got}, revision dba1be0a has {want}")
    sys.path.insert(0, str(a.ckpt / "encoding"))
    import encoding

    cfg = json.loads((a.ckpt / "inference" / "config.json").read_text())
    image_id, dim = cfg["image_token_id"], cfg["dim"]
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    files, _ = manifest(a.vision_set)
    prose = [int(t) for t in a.prose.read_text().split()]

    def bf16(name, rows):
        row = files[name]
        data = np.fromfile(a.vision_set / name, dtype="<u2")
        if data.size * 2 != int(row["bytes"]) or hashlib.md5(data.tobytes()).hexdigest() != row["md5"]:
            sys.exit(f"visref_cases: {name} does not match its file row")
        return data.reshape(rows, dim)

    delims = bf16("delims.bf16", 3)
    a.out.mkdir(parents=True, exist_ok=False)
    out = []
    for line in a.cases.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, image, question = line.split("\t")
        msg = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"{image}.png"}},
                                             {"type": "text", "text": question}]}]
        prompt = encoding.encode_messages(msg, thinking_mode="chat")
        ids = tok.encode(prompt)
        at = [i for i, t in enumerate(ids) if t == image_id]
        if len(at) != 1:
            sys.exit(f"visref_cases: {name}: {len(at)} image placeholders in {prompt!r}")
        types = np.fromfile(a.vision_set / f"{image}.types.i32", dtype="<i4")
        aligner = bf16(f"{image}.aligner.bf16", int((types == 1).sum()))
        rows, k = [], 0
        for t in types:
            if t == 1:
                rows.append(aligner[k])
                k += 1
            else:
                rows.append(delims[{0: 0, 2: 2, 3: 1}[int(t)]])
        if types[0] != 0 or types[-1] != 3 or k != aligner.shape[0]:
            sys.exit(f"visref_cases: {image}.types.i32 is not a span of {aligner.shape[0]} aligner rows")
        s, n = at[0], len(types)
        full = ids[:s] + [image_id] * n + ids[s + 1:]
        cases = [
            (name, full, s, n, "image", image),
            (f"{name}-text", ids[:s] + ids[s + 1:], 0, 0, "text", "-"),
            (f"{name}-prose", ids[:s] + prose[:n] + ids[s + 1:], 0, 0, "prose", "-"),
        ]
        for cname, cids, cat, clen, kind, img in cases:
            np.asarray(cids, dtype="<i4").tofile(a.out / f"{cname}.ids.i32")
            if clen:
                np.stack(rows).astype("<u2").tofile(a.out / f"{cname}.rows.bf16")
                types.astype("<i4").tofile(a.out / f"{cname}.types.i32")
            out.append((cname, len(cids), cat, clen, a.gen, kind, img, question))
        print(f"visref_cases: {name}: {len(full)} ids, span {s}+{n} ({' '.join(f'{KINDS[t]} {int((types == t).sum())}' for t in KINDS)})")
    with open(a.out / "cases.tsv", "w") as f:
        f.write("# name\tn_ids\tspan_at\tspan_len\tgen\tkind\timage\tquestion\n")
        for r in out:
            f.write("\t".join(map(str, r)) + "\n")
    for rel, blob in CODE.items():
        print(f"visref_cases: {rel} git-blob {blob}")


def write_manifest(a):
    d = a.set
    head = {}
    fork = {}
    for line in a.fork_log.read_text().splitlines():
        f = line.split()
        if line.startswith("visref_fork: n_vocab"):
            head = dict(zip(f[1::2], f[2::2]))
        elif f[:1] == ["case"]:
            fork[f[1]] = dict(zip(f[2::2], f[3::2]))
    if "n_vocab" not in head or "n_embd_inp" not in head:
        sys.exit(f"visref_cases: {a.fork_log} has no n_vocab line")
    vision_files, vision_heads = manifest(a.vision_set)
    checkpoint = vision_heads["checkpoint"][0]
    rows = []
    for line in (d / "cases.tsv").read_text().splitlines():
        if line.startswith("#"):
            continue
        name, n_ids, at, n, gen, kind, image, question = line.split("\t")
        if name not in fork:
            sys.exit(f"visref_cases: the fork's log reports no case {name}")
        r = fork[name]
        rows.append((name, kind, image, n_ids, at, n, gen, r["answer"], r["eog"], r["logits"], question))
    files = []
    for p in sorted(d.iterdir()):
        if p.name in ("MANIFEST.tsv",) or p.suffix not in (".i32", ".bf16", ".f32", ".txt", ".tsv", ".log"):
            continue
        data = p.read_bytes()
        kind = p.name.split(".", 1)[1] if "." in p.name else "-"
        files.append((p.name, kind, len(data), hashlib.md5(data).hexdigest()))
    with open(d / "MANIFEST.tsv", "w") as f:
        f.write("# oracle\tvisref_fork.cpp\ttools/ref/vision/visref.sh\n")
        f.write(f"# model\t{a.model}\n")
        f.write(f"# build\t{a.build}\thttps://github.com/smalinin/llama.cpp\n")
        f.write("# arch\tdeepseek41\n")
        f.write(f"# rows\t{a.vision_set}\t{checkpoint}\n")
        for rel, blob in CODE.items():
            f.write(f"# code\t{rel}\tgit-blob\t{blob}\n")
        f.write(f"# image_token_id\t{vision_heads['image_token_id'][0]}\n")
        f.write(f"# n_vocab\t{head['n_vocab']}\n")
        f.write(f"# n_embd\t{head['n_embd_inp']}\n")
        f.write(f"# keep\t{a.keep}\n")
        f.write(f"# placement\t{a.placement}\n")
        f.write(f"# device\t{a.device}\n")
        f.write("# case columns\tname kind image n_ids span_at span_len gen answer eog logits question\n")
        for r in rows:
            f.write("case\t" + "\t".join(map(str, r)) + "\n")
        f.write("# file columns\tname kind dtype shape bytes md5\n")
        for name, kind, size, md5 in files:
            dtype = {"i32": "i32", "bf16": "bf16", "f32": "f32"}.get(name.rsplit(".", 1)[-1], "text")
            f.write(f"file\t{name}\t{kind}\t{dtype}\t-\t{size}\t{md5}\n")
        f.write(f"# complete\t{len(rows)}\t{len(files)}\n")
    print(f"visref_cases: {d}/MANIFEST.tsv: {len(rows)} cases, {len(files)} files")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cases")
    c.add_argument("--ckpt", required=True, type=Path)
    c.add_argument("--vision-set", required=True, type=Path)
    c.add_argument("--cases", required=True, type=Path)
    c.add_argument("--prose", required=True, type=Path)
    c.add_argument("--gen", required=True, type=int)
    c.add_argument("--out", required=True, type=Path)
    m = sub.add_parser("manifest")
    m.add_argument("--set", required=True, type=Path)
    m.add_argument("--fork-log", required=True, type=Path)
    m.add_argument("--vision-set", required=True, type=Path)
    m.add_argument("--model", required=True)
    m.add_argument("--build", required=True)
    m.add_argument("--keep", required=True, type=int)
    m.add_argument("--placement", required=True)
    m.add_argument("--device", required=True)
    a = ap.parse_args()
    cases(a) if a.cmd == "cases" else write_manifest(a)


if __name__ == "__main__":
    main()
