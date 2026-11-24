#!/usr/bin/env python3
"""Generate crates/gguf/src/iq_tables.rs from ggml's sources, or check it.

The i-quant codebooks (iq2xxs_grid, iq2xs_grid, iq3xxs_grid, iq3s_grid, ksigns_iq2xs, kmask_iq2xs,
kvalues_iq4nl, iq2s_grid, iq1s_grid) are data, not code: this script is their one transcription.

Usage (from the repo root):

  gen-iq-tables.py --check [--out FILE]
      hold each table of the Rust file to the md5 its doc line records. Reads no tree: this is
      what `just gate-1-1` runs, and no gate reads a reference tree.
  gen-iq-tables.py --check --ik DIR [--also DIR ...] [--out FILE]
      read the tables from DIR too, and fail unless the file is byte-identical to a fresh
      generation from it: onboarding, the way a table earns a new md5 line.
  gen-iq-tables.py --ik DIR [--also DIR ...] [--out FILE]
      write the Rust file from DIR's sources.

  --ik DIR    the ggml tree to read; there is no default, a tree is read only when one is named
  --also DIR  another ggml tree whose tables must be identical (e.g. mainline); needs --ik
  --out FILE  the Rust file (default crates/gguf/src/iq_tables.rs)
  --check     compare instead of write; exit 1 on any difference

Without --ik each table is read back from the Rust file by the shape render() writes (its doc
line, then the const and its literals) and hashed over its little-endian bytes. A table that is
missing, declared with another Rust type or count, holds a literal this script cannot read, or
hashes to something other than its doc line's md5 is refused by name. The recorded md5 is only as
good as its last --ik check: a value and its md5 line edited together pass, which is why the
onboarding check compares the whole file with the header's own tables.

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
    ("iq2xxs_grid", "IQ2XXS_GRID", "uint64_t", 256),
    ("iq2xs_grid", "IQ2XS_GRID", "uint64_t", 512),
    ("iq3xxs_grid", "IQ3XXS_GRID", "uint32_t", 256),
    ("iq3s_grid", "IQ3S_GRID", "uint32_t", 512),
    ("kvalues_iq4nl", "KVALUES_IQ4NL", "int8_t", 16),
    ("iq2s_grid", "IQ2S_GRID", "uint64_t", 1024),
    # The header spells the count NGRID_IQ1S (2048); the length check below holds it.
    ("iq1s_grid", "IQ1S_GRID", "uint64_t", 2048),
]

RUST_TY = {"uint8_t": "u8", "uint32_t": "u32", "uint64_t": "u64", "int8_t": "i8"}
BYTES = {"uint8_t": 1, "uint32_t": 4, "uint64_t": 8, "int8_t": 1}

# The shape render() gives a table in the Rust file, as read_tables() reads it back.
DOC_LINE = re.compile(r"^/// `(?P<cname>[a-z0-9_]+)` \((?P<src>[^)]+)\), md5 (?P<md5>[0-9a-f]{32})\.$")
CONST_LINE = re.compile(r"^pub const (?P<rname>[A-Z0-9_]+): \[(?P<ty>u8|u32|u64|i8); (?P<n>[0-9]+)\] = \[$")


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
            r"GGML_TABLE_BEGIN\(\s*%s\s*,\s*%s\s*,\s*(?:%d|NGRID_IQ1S)\s*\)(.*?)GGML_TABLE_END\(\)"
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
    if ctype in ("int8_t", "uint8_t"):
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
    out.append("//! Generated; do not edit. `just gate-1-1` holds each table to the md5 its doc line")
    out.append("//! records, over its little-endian bytes as the script prints it: the digest of the")
    out.append("//! last extraction from the header the line names. The script's `--ik DIR` reads a tree")
    out.append("//! (onboarding only) and fails unless this file equals the fresh generation.")
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


def read_tables(path):
    """The tables of the Rust file at `path`, as render() wrote them: {C name: (Rust name, Rust type,
    count, values, source, md5)}. A doc line, a const or a value line of another shape is a named exit."""
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().split("\n")
    except OSError as e:
        sys.exit(f"iq tables: cannot read {path}: {e.strerror}")
    out = {}
    i = 0
    while i < len(lines):
        doc = DOC_LINE.match(lines[i])
        if doc is None:
            if lines[i].startswith("/// `") or lines[i].startswith("pub const "):
                sys.exit(f"{path}:{i + 1}: a table's doc line or const of another shape: {lines[i]}")
            i += 1
            continue
        cname = doc["cname"]
        const = CONST_LINE.match(lines[i + 2]) if i + 2 < len(lines) else None
        if const is None or lines[i + 1] != "#[rustfmt::skip]":
            sys.exit(f"{path}:{i + 1}: the doc line of {cname} is not followed by #[rustfmt::skip] and its const")
        vals = []
        i += 3
        while i < len(lines) and lines[i] != "];":
            if not (lines[i].startswith("    ") and lines[i].endswith(",")):
                sys.exit(f"{path}:{i + 1}: a value line of {cname} of another shape: {lines[i]}")
            for tok in lines[i][4:-1].split(","):
                try:
                    vals.append(int(tok, 0))
                except ValueError:
                    sys.exit(f"{path}:{i + 1}: {cname} holds a literal this script cannot read: {tok.strip()}")
            i += 1
        if i >= len(lines):
            sys.exit(f"{path}:{i}: the const {cname} does not end with ];")
        out[cname] = (const["rname"], const["ty"], int(const["n"]), vals, doc["src"], doc["md5"])
        i += 1
    return out


def check_recorded(path):
    """Every table of TABLES is in the Rust file with its Rust type and count and hashes to the md5 its
    doc line records. A table missing, extra or of another type or count is a named exit; a table that
    hashes to something else is printed and ends the run with exit 1, named."""
    have = read_tables(path)
    bad = []
    for cname, rname, ctype, n in TABLES:
        t = have.pop(cname, None)
        if t is None:
            sys.exit(f"iq tables: {path} has no table {cname}")
        rust_name, ty, declared, vals, src, recorded = t
        if rust_name != rname or ty != RUST_TY[ctype] or declared != n:
            sys.exit(f"iq tables: {path}: {cname} is {rust_name}: [{ty}; {declared}], want {rname}: [{RUST_TY[ctype]}; {n}]")
        if len(vals) != n:
            sys.exit(f"iq tables: {path}: {cname} holds {len(vals)} values, want {n}")
        try:
            md5 = hashlib.md5(le_bytes(vals, ctype)).hexdigest()
        except OverflowError:
            sys.exit(f"iq tables: {path}: a value of {cname} does not fit {ctype}")
        if md5 == recorded:
            print(f"{cname:14} {src:28} md5 {md5}  as recorded")
        else:
            print(f"{cname:14} {src:28} md5 {recorded}  DIFFERS: its values hash to {md5}")
            bad.append(cname)
    if have:
        sys.exit(f"iq tables: {path} holds tables this script does not transcribe: {sorted(have)}")
    if bad:
        sys.exit(f"iq tables: {path}: {len(bad)} table(s) differ from their recorded md5: {', '.join(bad)}")
    print(f"iq tables: {path}: {len(TABLES)} tables, each at its recorded md5")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--ik", help="the ggml tree to read; no default, a tree is read only when one is named")
    ap.add_argument("--also", action="append", default=[])
    ap.add_argument("--out", default="crates/gguf/src/iq_tables.rs")
    a = ap.parse_args()
    if a.also and not a.ik:
        ap.error("--also names a second tree to hold against --ik's: pass --ik DIR")
    if not a.check and not a.ik:
        ap.error("nothing to check or write: --check holds --out to its recorded md5s, --ik DIR writes --out from a tree")
    if a.check and not a.ik:
        check_recorded(a.out)
        return

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
