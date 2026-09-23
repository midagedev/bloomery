#!/usr/bin/env python3
"""Verify an oracle set's integer twins against its f32 files.

    tools/ref/check-int-twins.py [<set dir>...]      # default: $BLOOMERY_DATA/ref
    tools/ref/check-int-twins.py --self-test

dump_ref writes every integer tensor twice: cast to f32 in the file the gates read, and exact
in a twin (`.i32`/`.i64`, one `int` manifest row per twin file). The f32 file is exact only up
to 2^24 in magnitude, so the twin is what backs an exact-match gate — and only this check says
the two hold the same tensor. Per set it requires:

  - a complete manifest (the `# complete` trailer);
  - for every integer-typed `tensor`/`input` row, its flat twin, and a logical twin exactly
    when the row has a logical f32 file; for every `int` row, the row it twins;
  - each twin file's size = count x width, count = ne0*ne1*ne2*ne3 of its row, and the `int`
    row's exact sum (wrapping 64-bit, signed) and largest magnitude recomputed from the file;
  - element by element, the f32 file of the same layout holds exactly the twin value rounded
    to binary32 — round-to-nearest-even computed in integer arithmetic, which is C's cast.
    Up to 2^24 in magnitude that is plain equality; above it, it is the cast the dumper made,
    and those elements are counted in the ok line: they are where the f32 file alone cannot
    back an exact gate;
  - an `inp_tokens` input, when the set has one, equal to the `# tokens` header — past its first
    `# prefill` ids in a decode-step set, whose graph evaluated only the step.

Exit 0 with one ok line per set, 1 with every failure listed. --self-test builds small sets in
a temp directory around 2^24 and 384,006,168 (V4.1's largest engram row id) and requires green
on the good one and red on each broken variant — the branch the V2-Lite set, whose ids are all
below 64, can never reach.
"""
import os
import random
import struct
import sys
import tempfile

WIDTH = {"i32": 4, "i64": 8}
INT_TYPES = {"i8", "i16", "i32", "i64"}
EXACT = 1 << 24


def rne_f32_bits(v: int) -> int:
    """binary32 bits of the integer v under round-to-nearest-even, i.e. C's (float)v."""
    if v == 0:
        return 0
    sign = 0x80000000 if v < 0 else 0
    m = -v if v < 0 else v
    e = m.bit_length() - 1  # m is in [2^e, 2^(e+1))
    if e <= 23:
        mant = m << (23 - e)
    else:
        shift = e - 23
        mant = m >> shift
        rem = m - (mant << shift)
        half = 1 << (shift - 1)
        if rem > half or (rem == half and mant & 1):
            mant += 1
            if mant == 1 << 24:  # the carry reached the next binade
                mant >>= 1
                e += 1
    if e > 127:
        return sign | 0x7F800000
    return sign | ((e + 127) << 23) | (mant & 0x7FFFFF)


def safe_name(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_")


def read_manifest(path):
    header, rows, ints, complete = {}, {}, [], False
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if line.startswith("#"):
                key, _, val = line[2:].partition("\t")
                complete |= key == "complete"
                header.setdefault(key, val)
                continue
            f_ = line.split("\t")
            if f_[0] in ("tensor", "input"):
                rows[(f_[0], f_[1], int(f_[2]))] = {
                    "type": f_[3],
                    "count": int(f_[4]) * int(f_[5]) * int(f_[6]) * int(f_[7]),
                    "logical": int(f_[12]) if len(f_) >= 13 else 0,
                    "line": n,
                }
            elif f_[0] == "int":
                if len(f_) != 12:
                    raise ValueError(f"{path}:{n}: int row has {len(f_)} fields, want 12")
                ints.append({
                    "key": (f_[3], f_[1], int(f_[2])), "type": f_[4], "twin": f_[5], "layout": f_[6],
                    "count": int(f_[7]), "bytes": int(f_[8]), "sum": int(f_[9]), "absmax": int(f_[10]),
                    "file": f_[11], "line": n,
                })
    return header, rows, ints, complete


def check_set(d: str):
    """(failures, ok summary) for the set at d."""
    fails = []
    man = os.path.join(d, "MANIFEST.tsv")
    try:
        header, rows, ints, complete = read_manifest(man)
    except (OSError, ValueError) as e:
        return [f"cannot read the manifest: {e}"], ""
    if not complete:
        fails.append("no `# complete` trailer — the dump that wrote this set did not finish")

    seen = set()
    elements = above = 0
    values = {}
    for r in ints:
        where = f"MANIFEST.tsv:{r['line']} {r['file']}"
        kind, name, occ = r["key"]
        row = rows.get(r["key"])
        if row is None:
            fails.append(f"{where}: twins {kind} {name}/{occ}, which has no row")
            continue
        if row["type"] not in INT_TYPES or r["type"] != row["type"]:
            fails.append(f"{where}: twin of a {row['type']} row claims type {r['type']}")
            continue
        want_twin = "i64" if row["type"] == "i64" else "i32"
        stem = f"{safe_name(name)}.{occ}" + (".input" if kind == "input" else "")
        suffix = ".logical." if r["layout"] == "logical" else "."
        if r["twin"] != want_twin or r["file"] != stem + suffix + want_twin or r["layout"] not in ("flat", "logical"):
            fails.append(f"{where}: want {stem}{suffix}{want_twin} ({want_twin}, {r['layout']})")
            continue
        seen.add((r["key"], r["layout"]))
        if r["count"] != row["count"] or r["bytes"] != r["count"] * WIDTH[want_twin]:
            fails.append(f"{where}: count {r['count']} / bytes {r['bytes']}, row says {row['count']} elements")
            continue
        try:
            raw = open(os.path.join(d, r["file"]), "rb").read()
            f32 = open(os.path.join(d, stem + suffix + "f32"), "rb").read()
        except OSError as e:
            fails.append(f"{where}: {e}")
            continue
        n = r["count"]
        if len(raw) != r["bytes"] or len(f32) != 4 * n:
            fails.append(f"{where}: {len(raw)} B twin / {len(f32)} B f32 for {n} elements")
            continue
        vals = struct.unpack(f"<{n}{'q' if want_twin == 'i64' else 'i'}", raw)
        bits = struct.unpack(f"<{n}I", f32)
        total = sum(vals) & 0xFFFFFFFFFFFFFFFF
        total = total - (1 << 64) if total >> 63 else total
        if total != r["sum"] or max((abs(v) for v in vals), default=0) != r["absmax"]:
            fails.append(f"{where}: sum/absmax recomputed from the file differ from the int row")
        bad = [i for i in range(n) if bits[i] != rne_f32_bits(vals[i])]
        if bad:
            i = bad[0]
            got = struct.unpack("<f", struct.pack("<I", bits[i]))[0]
            fails.append(f"{where}: {len(bad)} of {n} elements differ from the f32 file; "
                         f"first at {i}: twin {vals[i]}, f32 {got!r}")
        elements += n
        above += sum(1 for v in vals if abs(v) > EXACT)
        values[(r["key"], r["layout"])] = vals

    for key, row in rows.items():
        if row["type"] not in INT_TYPES:
            continue
        for layout in ("flat", "logical") if row["logical"] else ("flat",):
            if (key, layout) not in seen:
                fails.append(f"MANIFEST.tsv:{row['line']}: {key[0]} {key[1]}/{key[2]} ({row['type']}) "
                             f"has no valid {layout} twin")

    tokens_note = "no inp_tokens input"
    tok = values.get((("input", "inp_tokens", 0), "flat"))
    if tok is not None:
        want = [int(t) for t in header.get("tokens", "").split(",") if t]
        prefill = int(header.get("prefill", "0"))
        if list(tok) != want[prefill:]:
            fails.append(f"inp_tokens {list(tok)} != # tokens {want}" if not prefill else
                         f"inp_tokens {list(tok)} != # tokens past the prefill of {prefill}: {want[prefill:]}")
        tokens_note = (f"inp_tokens = # tokens ({len(want)} ids)" if not prefill else
                       f"inp_tokens = # tokens past the prefill of {prefill} ({len(want) - prefill} of {len(want)} ids)")
    n_in = sum(1 for r in ints if r["key"][0] == "input")
    ok = (f"{len(ints)} twins ({len(ints) - n_in} of tensors, {n_in} of inputs), {elements} elements, "
          f"{above} above 2^24; {tokens_note}")
    return fails, ok


def run(dirs) -> int:
    rc = 0
    for d in dirs:
        fails, ok = check_set(d)
        if fails:
            rc = 1
            for f in fails:
                print(f"check-int-twins: {d}: {f}", file=sys.stderr)
            print(f"check-int-twins: {d}: RED ({len(fails)} failure(s))", file=sys.stderr)
        else:
            print(f"check-int-twins: {d}: ok — {ok}")
    return rc


# ------------------------------------------------------------------------------ self-test

def write_set(d, tensors, tokens, drop_int=(), f32_patch=None, prefill=None):
    """A minimal set: tensors = [(kind, name, type, flat values, logical values or None)]."""
    lines = ["# dump_ref — check-int-twins self-test", "# model\tself-test",
             "# tokens\t" + ",".join(map(str, tokens))]
    if prefill is not None:
        lines.append(f"# prefill\t{prefill}")
    for kind, name, ty, flat, logical in tensors:
        stem = f"{safe_name(name)}.0" + (".input" if kind == "input" else "")
        twin = "i64" if ty == "i64" else "i32"
        layouts = [("flat", flat, ".")] + ([("logical", logical, ".logical.")] if logical else [])
        lines.append("\t".join(map(str, [kind, name, 0, ty, len(flat), 1, 1, 1, 4 * len(flat), 0.0,
                                        "NONE", 0 if logical else 1, 1 if logical else 0, "-", "-"])))
        for layout, vals, suffix in layouts:
            bits = [rne_f32_bits(v) for v in vals]
            if f32_patch and f32_patch[0] == (name, layout):
                bits[f32_patch[1]] = f32_patch[2]
            open(os.path.join(d, stem + suffix + "f32"), "wb").write(struct.pack(f"<{len(bits)}I", *bits))
            open(os.path.join(d, stem + suffix + twin), "wb").write(
                struct.pack(f"<{len(vals)}{'q' if twin == 'i64' else 'i'}", *vals))
            if (name, layout) in drop_int:
                continue
            total = sum(vals) & 0xFFFFFFFFFFFFFFFF
            total = total - (1 << 64) if total >> 63 else total
            lines.append("\t".join(map(str, ["int", name, 0, kind, ty, twin, layout, len(vals),
                                            len(vals) * WIDTH[twin], total, max(abs(v) for v in vals),
                                            stem + suffix + twin])))
    lines.append("# complete\t1\t0")
    open(os.path.join(d, "MANIFEST.tsv"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


def self_test() -> int:
    # The integer rounding against Python's own double -> float cast, which is exact below 2^53.
    rng = random.Random(7)
    probe = [0, 1, -1, EXACT - 1, EXACT, EXACT + 1, EXACT + 2, EXACT + 3, 384_006_168, 384_016_682,
             (1 << 31) - 1, -(1 << 31), (1 << 53) - 1] + [rng.randrange(-(1 << 53), 1 << 53) for _ in range(20000)]
    for v in probe:
        if rne_f32_bits(v) != struct.unpack("<I", struct.pack("<f", float(v)))[0]:
            print(f"self-test: rne_f32_bits({v}) disagrees with the C cast", file=sys.stderr)
            return 1
    ids = [EXACT + 1, 3, -5, 384_006_168]            # 2^24 + 1 rounds to 2^24 (tie, even)
    rows64 = [(1 << 60) + 1, -(1 << 31), 7]
    good = [("tensor", "ids", "i32", ids, None),
            ("tensor", "rows64 (view)", "i64", rows64, rows64[::-1]),
            ("input", "inp_tokens", "i32", [5, 6], None)]
    step_up = rne_f32_bits(EXACT + 2)                # the next binary32 above 2^24
    cases = [
        ("good set", {}, False),
        ("f32 one step off above 2^24", {"f32_patch": (("ids", "flat"), 0, step_up)}, True),
        ("missing flat twin row", {"drop_int": {("ids", "flat")}}, True),
        ("missing logical twin row", {"drop_int": {("rows64 (view)", "logical")}}, True),
        ("tokens header differs", {"tokens": [5, 7]}, True),
        ("decode step: inp_tokens = the ids past the prefill", {"tokens": [9, 5, 6], "prefill": 1}, False),
        ("decode step: prefill line one short", {"tokens": [9, 5, 6], "prefill": 0}, True),
        ("decode step: prefill line missing", {"tokens": [9, 5, 6]}, True),
    ]
    for label, kw, want_red in cases:
        with tempfile.TemporaryDirectory() as d:
            write_set(d, good, kw.pop("tokens", [5, 6]), **kw)
            fails, _ = check_set(d)
            if bool(fails) != want_red:
                print(f"self-test: {label}: want {'red' if want_red else 'green'}, got {fails or 'green'}",
                      file=sys.stderr)
                return 1
            print(f"self-test: {label}: {'red' if fails else 'green'} as required"
                  + (f" ({fails[0]})" if fails else ""))
    print(f"self-test: ok ({len(probe)} rounding probes, {len(cases)} sets)")
    return 0


def main() -> int:
    args = sys.argv[1:]
    if args == ["--self-test"]:
        return self_test()
    if any(a.startswith("-") for a in args):
        print(__doc__, file=sys.stderr)
        return 2
    if not args:
        args = [os.path.join(os.environ.get("BLOOMERY_DATA") or "/root/bloomery-data", "ref")]
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
