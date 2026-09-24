#!/usr/bin/env python3
"""Generate crates/gguf/src/iq_tables.rs from ggml's sources, or check it.

The i-quant codebooks (iq2xs_grid, iq3xxs_grid, ksigns_iq2xs, kmask_iq2xs,
kvalues_iq4nl) are data, not code: this script is their one transcription.
It reads them from the ggml tree the reference harnesses link ($IK), writes
them as Rust `const` arrays, and prints each table's md5 over its
little-endian bytes. The generated file carries those md5 values; `--check`
re-extracts from the header on the box and fails unless the committed file
is byte-identical to a fresh generation, so the Rust tables are pinned to
the header, not to themselves.

Usage (from the repo root):
  gen-iq-tables.py [--check] [--ik DIR] [--also DIR ...] [--out FILE]

  --ik DIR    the ggml tree to read (default $IK, else /home/user/ik_llama.cpp)
  --also DIR  another ggml tree whose tables must be identical (e.g. mainline)
  --out FILE  the Rust file (default crates/gguf/src/iq_tables.rs)
  --check     compare instead of write; exit 1 on any difference

Sources inside a tree: ggml/src/ggml-common.h for the GGML_TABLE_BEGIN
blocks; kvalues_iq4nl is a GGML_TABLE in mainline's header but a
`static const int8_t kvalues_iq4nl[16]` in ik's ggml/src/ggml-quants.c, so
both spellings are read.
"""

import argparse
import hashlib
import os
import re
import sys

# (C name, Rust name, C element type, count)
TABLES = [
    ("kmask_iq2xs", "KMASK_IQ2XS", "uint8_t", 8),
    ("ksigns_iq2xs", "KSIGNS_IQ2XS", "uint8_t", 128),
    ("iq2xs_grid", "IQ2XS_GRID", "uint64_t", 512),
    ("iq3xxs_grid", "IQ3XXS_GRID", "uint32_t", 256),
    ("kvalues_iq4nl", "KVALUES_IQ4NL", "int8_t", 16),
]

RUST_TY = {"uint8_t": "u8", "uint32_t": "u32", "uint64_t": "u64", "int8_t": "i8"}
BYTES = {"uint8_t": 1, "uint32_t": 4, "uint64_t": 8, "int8_t": 1}


def strip_comments(s):
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    return re.sub(r"//[^\n]*", "", s)


def parse_values(body):
    vals = []
    for tok in body.replace("\n", " ").split(","):
        tok = tok.strip()
        if tok:
            vals.append(int(tok, 0))
    return vals


def extract(tree, cname, ctype, n):
    """The table's values from `tree`, and the file:line it came from."""
    for rel in ("ggml/src/ggml-common.h", "ggml/src/ggml-quants.c"):
        path = os.path.join(tree, rel)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        pats = [
            r"GGML_TABLE_BEGIN\(\s*%s\s*,\s*%s\s*,\s*%d\s*\)(.*?)GGML_TABLE_END\(\)"
            % (re.escape(ctype), cname, n),
            r"static\s+const\s+%s\s+%s\s*\[\s*%d\s*\]\s*=\s*\{(.*?)\};"
            % (re.escape(ctype), cname, n),
        ]
        for pat in pats:
            m = re.search(pat, text, flags=re.S)
            if m:
                vals = parse_values(strip_comments(m.group(1)))
                if len(vals) != n:
                    sys.exit(f"{path}: {cname} has {len(vals)} values, want {n}")
                line = text.count("\n", 0, m.start()) + 1
                return vals, f"{rel}:{line}"
    sys.exit(f"{tree}: table {cname}[{n}] of {ctype} not found")


def le_bytes(vals, ctype):
    w = BYTES[ctype]
    signed = ctype.startswith("int")
    return b"".join(v.to_bytes(w, "little", signed=signed) for v in vals)


def lit(v, ctype):
    if ctype == "int8_t":
        return str(v)
    if ctype == "uint8_t":
        return str(v)
    if ctype == "uint32_t":
        h = f"{v:08x}"
        return f"0x{h[:4]}_{h[4:]}"
    h = f"{v:016x}"
    return f"0x{h[:4]}_{h[4:8]}_{h[8:12]}_{h[12:]}"


def per_line(ctype):
    return {"uint8_t": 16, "int8_t": 16, "uint32_t": 6, "uint64_t": 3}[ctype]


def render(tables):
    out = []
    out.append("//! The i-quant codebooks, transcribed from ggml by `tools/ref/gen-iq-tables.py`.")
    out.append("//!")
    out.append("//! Generated; do not edit. `just gate-1-1` regenerates this file from the header")
    out.append("//! the reference harnesses link and fails on any difference. The md5 after each")
    out.append("//! table's source is over its little-endian bytes, as the script prints it.")
    out.append("")
    for cname, rname, ctype, n, vals, src, md5 in tables:
        out.append(f"/// `{cname}` ({src}), md5 {md5}.")
        out.append("#[rustfmt::skip]")
        out.append(f"pub const {rname}: [{RUST_TY[ctype]}; {n}] = [")
        k = per_line(ctype)
        for i in range(0, n, k):
            out.append("    " + ", ".join(lit(v, ctype) for v in vals[i : i + k]) + ",")
        out.append("];")
        out.append("")
    return "\n".join(out).rstrip("\n") + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--ik", default=os.environ.get("IK", "/home/user/ik_llama.cpp"))
    ap.add_argument("--also", action="append", default=[])
    ap.add_argument("--out", default="crates/gguf/src/iq_tables.rs")
    a = ap.parse_args()

    tables = []
    ok = True
    for cname, rname, ctype, n in TABLES:
        vals, src = extract(a.ik, cname, ctype, n)
        md5 = hashlib.md5(le_bytes(vals, ctype)).hexdigest()
        print(f"{cname:14} {src:28} md5 {md5}  ({a.ik})")
        for other in a.also:
            ov, osrc = extract(other, cname, ctype, n)
            omd5 = hashlib.md5(le_bytes(ov, ctype)).hexdigest()
            same = "same" if ov == vals else "DIFFERS"
            print(f"{cname:14} {osrc:28} md5 {omd5}  ({other}) {same}")
            ok &= ov == vals
        tables.append((cname, rname, ctype, n, vals, src, md5))
    if not ok:
        sys.exit("iq tables: the trees disagree")

    text = render(tables)
    if a.check:
        with open(a.out, encoding="utf-8") as f:
            have = f.read()
        if have != text:
            print(f"iq tables: {a.out} differs from a fresh generation from {a.ik}")
            sys.exit(1)
        print(f"iq tables: {a.out} matches {a.ik}")
    else:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
