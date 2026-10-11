#!/usr/bin/env python3
"""A flat importance matrix for the tensors a fixture variant retypes to a very-low-bit quant.

ik's llama-quantize refuses IQ2_XXS, IQ2_XS, IQ2_S, IQ1_S and IQ1_M for any tensor that has no importance
matrix entry (src/llama-quantize.cpp, "Missing importance matrix ... bailing out"). A fixture variant needs
the type, not the quality: its gates compare the engine with ik on the same file. This writes ik's legacy
imatrix file (examples/quantize/quantize.cpp load_imatrix: i32 entries, then per entry i32 name length, the
name, i32 ncall, i32 nval, nval f32) with every weight 1.0, one entry per such tensor, nval =
ne[0] * ne[2] (a row's width times the expert count, the layout the quantizer reads an expert stack's
entry in; a 2-D tensor has ne[2] = 1).

  flat-imatrix.py SOURCE_FIRST_SHARD CUSTOM_Q OUT   write OUT; print the entries to stderr; exit 1 when
                                                    CUSTOM_Q retypes nothing to such a type (no file)
  flat-imatrix.py --check-row OUT TABLE TAG         exit 65 unless the variant row of TAG in TABLE names the matrix OUT
                                                    (`imatrix=sha256:<first 16 hex of its bytes>` in the why column), or,
                                                    when OUT does not exist, names none
  flat-imatrix.py --self-test

CUSTOM_Q is the `--custom-q` argument `fixture variant-rules` prints: `^<escaped name>$=<type>` rules joined
by commas. Every shard of SOURCE's `<stem>-0000N-of-0000M.gguf` set is read for its tensor dims.

The matrix is part of the variant's identity: its bytes shape the quantized tensors, and the file's own
`quantize.imatrix.*` keys name only a path. The variant's row in tools/fixture-variants.tsv carries the matrix's id,
`--check-row` refuses a requant whose matrix is not the one the row names, and the row's text is in the key of every item
that runs under the variant, so a changed matrix is an edited row and a new fixture.
"""

import glob
import hashlib
import os
import re
import struct
import sys
import tempfile

LOW_BIT = {"iq2_xxs", "iq2_xs", "iq2_s", "iq1_s", "iq1_m"}
SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "B", 10: "Q", 11: "q", 12: "d"}


class Short(Exception):
    pass


class Buf:
    def __init__(self, b):
        self.b, self.p = b, 0

    def take(self, fmt):
        n = struct.calcsize(fmt)
        if self.p + n > len(self.b):
            raise Short
        v = struct.unpack_from("<" + fmt, self.b, self.p)
        self.p += n
        return v[0]

    def string(self):
        n = self.take("Q")
        if self.p + n > len(self.b):
            raise Short
        s = self.b[self.p:self.p + n].decode("utf-8")
        self.p += n
        return s

    def skip(self, t):
        if t in SCALAR:
            self.take(SCALAR[t])
        elif t == 8:
            self.string()
        elif t == 9:
            et, n = self.take("I"), self.take("Q")
            for _ in range(n):
                self.skip(et)
        else:
            raise ValueError(f"gguf value type {t}")


def tensor_dims(path):
    """{name: dims} of one GGUF file, read from its header alone."""
    n = 8 << 20
    with open(path, "rb") as f:
        while True:
            f.seek(0)
            b = f.read(n)
            try:
                bf = Buf(b)
                if b[:4] != b"GGUF":
                    raise ValueError(f"{path}: not a GGUF file")
                bf.p = 4
                bf.take("I")
                nt, nkv = bf.take("Q"), bf.take("Q")
                for _ in range(nkv):
                    bf.string()
                    bf.skip(bf.take("I"))
                out = {}
                for _ in range(nt):
                    name = bf.string()
                    dims = [bf.take("Q") for _ in range(bf.take("I"))]
                    bf.take("I")
                    bf.take("Q")
                    out[name] = dims
                return out
            except Short:
                if len(b) < n:
                    raise ValueError(f"{path}: the header ends inside a record")
                n *= 2


def shards(first):
    m = re.match(r"^(.*)-00001-of-(\d{5})\.gguf$", first)
    if not m:
        raise ValueError(f"{first} is not a first shard named <stem>-00001-of-0000M.gguf")
    files = sorted(glob.glob(f"{m.group(1)}-?????-of-{m.group(2)}.gguf"))
    if len(files) != int(m.group(2)):
        raise ValueError(f"{first}: found {len(files)} shards, the name says {m.group(2)}")
    return files


def low_bit_rules(custom_q):
    """[(tensor name, type)] of the rules whose type needs an importance matrix."""
    out = []
    for rule in custom_q.split(","):
        pat, _, ty = rule.rpartition("=")
        if not (pat.startswith("^") and pat.endswith("$")):
            raise ValueError(f"custom-q rule {rule!r}: not ^name$=type")
        if ty in LOW_BIT:
            out.append((pat[1:-1].replace("\\.", "."), ty))
    return out


def write(entries, out_path):
    """entries: [(name, nval)]; one flat entry each."""
    b = bytearray(struct.pack("<i", len(entries)))
    for name, nval in entries:
        nb = name.encode()
        b += struct.pack("<i", len(nb)) + nb + struct.pack("<ii", 1, nval) + struct.pack("<f", 1.0) * nval
    tmp = f"{out_path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(b)
    os.replace(tmp, out_path)


ROW_KEY = "imatrix="


def matrix_id(path):
    with open(path, "rb") as f:
        return "sha256:" + hashlib.sha256(f.read()).hexdigest()[:16]


def check_row(out_path, table, tag):
    """None when the row of `tag` names the matrix at `out_path` (its id after `imatrix=` in the why column) or, with no
    file there, names none; else the refusal text."""
    why = None
    with open(table, encoding="utf-8") as f:
        for ln in f.read().splitlines():
            cols = ln.split("\t")
            if len(cols) == 6 and not ln.startswith("#") and cols[0] == tag:
                why = cols[5]
    if why is None:
        return f"{table} has no row {tag}"
    named = [w[len(ROW_KEY):] for w in why.split() if w.startswith(ROW_KEY)]
    if not os.path.exists(out_path):
        return f"row {tag} names the importance matrix {named}, but its map needs none" if named else None
    want = matrix_id(out_path)
    if named != [want]:
        return f"row {tag} names the importance matrix {named or 'none'}; the one written is {ROW_KEY}{want}: put that in its why column"
    return None


def main(first, custom_q, out_path):
    rules = low_bit_rules(custom_q)
    if not rules:
        print("flat-imatrix.py: custom-q retypes nothing to a very-low-bit type", file=sys.stderr)
        return 1
    dims = {}
    for s in shards(first):
        dims.update(tensor_dims(s))
    entries = []
    for name, ty in rules:
        if name not in dims:
            raise ValueError(f"custom-q names {name}, which the source does not hold")
        d = dims[name]
        entries.append((name, d[0] * (d[2] if len(d) > 2 else 1)))
    write(entries, out_path)
    for name, nval in entries:
        print(f"flat-imatrix.py: {name} nval {nval}", file=sys.stderr)
    return 0


def self_test():
    def gguf(tensors):
        b = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 1)
        key = b"general.architecture"
        b += struct.pack("<Q", len(key)) + key + struct.pack("<I", 8) + struct.pack("<Q", 3) + b"foo"
        for name, dims in tensors:
            nb = name.encode()
            b += struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(dims))
            b += b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", 0, 0)
        return b

    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "m-00001-of-00002.gguf")
        c = os.path.join(d, "m-00002-of-00002.gguf")
        open(a, "wb").write(gguf([("blk.0.ffn_gate_exps.weight", [256, 8, 4]), ("blk.0.attn_q.weight", [256, 8])]))
        open(c, "wb").write(gguf([("blk.1.ffn_up_exps.weight", [512, 8, 3])]))
        q = "^blk\\.0\\.ffn_gate_exps\\.weight$=iq2_xs,^blk\\.0\\.attn_q\\.weight$=q4_K,^blk\\.1\\.ffn_up_exps\\.weight$=iq2_xxs"
        out = os.path.join(d, "im.dat")
        assert main(a, q, out) == 0
        data = open(out, "rb").read()
        n = struct.unpack_from("<i", data, 0)[0]
        assert n == 2, n
        p, got = 4, []
        for _ in range(n):
            ln = struct.unpack_from("<i", data, p)[0]
            name = data[p + 4:p + 4 + ln].decode()
            ncall, nval = struct.unpack_from("<ii", data, p + 4 + ln)
            vals = struct.unpack_from(f"<{nval}f", data, p + 12 + ln)
            assert ncall == 1 and set(vals) == {1.0}
            got.append((name, nval))
            p += 12 + ln + 4 * nval
        assert p == len(data)
        assert got == [("blk.0.ffn_gate_exps.weight", 256 * 4), ("blk.1.ffn_up_exps.weight", 512 * 3)], got
        # a rule set with no very-low-bit type writes nothing
        assert main(a, "^blk\\.0\\.attn_q\\.weight$=q4_K", os.path.join(d, "none")) == 1
        assert not os.path.exists(os.path.join(d, "none"))
        # the row names the matrix it is given: its id, and no other, and none for a map that needs none
        table = os.path.join(d, "t.tsv")
        want = ROW_KEY + matrix_id(out)

        def row(why):
            open(table, "w").write(f"# c\nv1\tfoo\ttrunk\tffn_gate_exps=iq2_xs\tgate-x\t{why}\n")

        row(f"small proof {want} and more")
        assert check_row(out, table, "v1") is None
        row("small proof")
        assert want in check_row(out, table, "v1"), check_row(out, table, "v1")
        row(f"small proof {ROW_KEY}sha256:0000000000000000")
        assert want in check_row(out, table, "v1")
        row(f"{want} {want}")
        assert check_row(out, table, "v1") is not None
        assert "no row v2" in check_row(out, table, "v2")
        none = os.path.join(d, "none")
        row("small proof")
        assert check_row(none, table, "v1") is None
        row(f"small proof {want}")
        assert "needs none" in check_row(none, table, "v1")
        # a tensor the source lacks is refused by name
        try:
            main(a, "^blk\\.9\\.ffn_up_exps\\.weight$=iq2_xs", os.path.join(d, "x"))
        except ValueError as e:
            assert "blk.9.ffn_up_exps.weight" in str(e)
        else:
            raise AssertionError("a missing tensor was accepted")
        # a shard set with a missing member is refused
        os.remove(c)
        try:
            shards(a)
        except ValueError as e:
            assert "found 1 shards" in str(e)
        else:
            raise AssertionError("a short shard set was accepted")
    print("flat-imatrix.py self-test ok")
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        sys.exit(self_test())
    if sys.argv[1:2] == ["--check-row"]:
        if len(sys.argv) != 5:
            print(__doc__, file=sys.stderr)
            sys.exit(64)
        refusal = check_row(*sys.argv[2:])
        if refusal:
            print(f"flat-imatrix.py: {refusal}", file=sys.stderr)
            sys.exit(65)
        sys.exit(0)
    if len(sys.argv) != 4:
        print(__doc__, file=sys.stderr)
        sys.exit(64)
    sys.exit(main(*sys.argv[1:]))
