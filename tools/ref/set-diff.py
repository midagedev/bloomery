#!/usr/bin/env python3
"""Two node-dump sets of one model against each other, node by node: an oracle tree's own floor
against the set another tree dumped at the same ids, position and flags (round candref's d1 of the
candidate tree against the db517b69 d1).

    set-diff.py A B [--layers LO-HI] [--top K]
    set-diff.py --self-test

Both sides are set directories, read through tools/bloomery/manifest.py by column name; each must
carry its `# complete` trailer. A tensor row is paired with the row of the same name and occurrence on
the other side, and compared when both have the same op, type and shape (a graph that names another
node alike is not compared): its logical twin when both
sides have one, else its plain file. Per pair: bit-identical or not, the largest absolute distance
and that distance over the largest |B| of the row (non-finite values must sit in the same places and
be equal; one that does not counts as a mismatch of the row). Integer twins (`int` rows of a tensor)
are compared exactly, and a `lid_top_k-*` list also as a set, whose order the top-k does not fix.

Printed: one line per layer (the name's `-<layer>` suffix; rows without one are layer -1) with the
pairs compared, the bit-identical ones, the worst row by relative distance, and the top-k lists whose
sets differ; then the K worst rows overall; the rows each side has alone, counted; the first layer
whose top-k set differs; and a verdict line. --layers keeps the layers in LO..HI (and -1).

Exit: 0 every compared pair bit-identical and every top-k set equal, 1 otherwise, 2 a side that is not
a complete set or a file whose size is not its row's.
"""
import array
import importlib.util
import math
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "manifest", os.path.join(HERE, "..", "bloomery", "manifest.py"))
manifest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(manifest)

LAYER = re.compile(r"-(\d+)(?: |$)")


class DiffError(ValueError):
    pass


def layer_of(name):
    m = LAYER.search(name)
    return int(m.group(1)) if m else -1


def stem(name):
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_")


def side(path):
    m = manifest.read(path)
    if m.complete is None:
        raise DiffError(f"{path}: no # complete trailer")
    tensors = {(r["name"], r.int("occurrence")): r for r in m.rows("tensor")}
    ints = {(r["name"], r.int("occurrence")): r for r in m.rows("int") if r["of"] == "tensor"}
    return m, tensors, ints


def read_array(f, code, want):
    a = array.array(code)
    with open(f, "rb") as fh:
        raw = fh.read()
    if len(raw) != want * a.itemsize:
        raise DiffError(f"{f}: {len(raw)} bytes, its row holds {want} values of {a.itemsize}")
    a.frombytes(raw)
    if sys.byteorder != "little":
        a.byteswap()
    return raw, a


def values(m, row, logical):
    occ = row.int("occurrence")
    f = os.path.join(m.dir, f"{stem(row['name'])}.{occ}{'.logical' if logical else ''}.f32")
    want = 1
    for ne in ("ne0", "ne1", "ne2", "ne3"):
        want *= row.int(ne)
    return read_array(f, "f", want)


def int_values(m, row):
    f = os.path.join(m.dir, row["file"])
    return read_array(f, "i" if row["twin"] == "i32" else "q", row.int("count"))[1]


def compare(a, b):
    """(identical, max abs distance, that over max |b|, non-finite mismatch) of two (bytes, array)
    sides."""
    if a[0] == b[0]:
        return True, 0.0, 0.0, False
    bad = False
    da = scale = 0.0
    for x, y in zip(a[1], b[1]):
        fx, fy = math.isfinite(x), math.isfinite(y)
        if fx != fy or (not fx and x != y):
            bad = True
        elif fx:
            da = max(da, abs(x - y))
            scale = max(scale, abs(y))
    return False, da, (da / scale if scale > 0 else float("inf") if da > 0 else 0.0), bad


def diff(pa, pb, lo=None, hi=None, top=10, out=sys.stdout):
    ma, ta, ia = side(pa)
    mb, tb, ib = side(pb)
    keep = lambda key: lo is None or layer_of(key[0]) == -1 or lo <= layer_of(key[0]) <= hi
    common = [k for k in ta if k in tb and keep(k)]
    per = {}
    worst = []
    shape_skip = 0
    for key in common:
        ra, rb = ta[key], tb[key]
        shape = lambda r: (r["op"], r["type"], r["ne0"], r["ne1"], r["ne2"], r["ne3"])
        if shape(ra) != shape(rb):
            shape_skip += 1
            continue
        logical = ra.get("logical") == "1" and rb.get("logical") == "1"
        same, da, rel, bad = compare(values(ma, ra, logical), values(mb, rb, logical))
        L = layer_of(key[0])
        p = per.setdefault(L, {"n": 0, "same": 0, "worst": (0.0, None), "topk": []})
        p["n"] += 1
        p["same"] += int(same and not bad)
        if not same:
            r = float("inf") if bad else rel
            worst.append((r, da, key[0]))
            if r > p["worst"][0] or p["worst"][1] is None:
                p["worst"] = (r, key[0])
    topk_bad = []
    int_bad = 0
    for key in ia:
        if key not in ib or not keep(key):
            continue
        a, b = int_values(ma, ia[key]), int_values(mb, ib[key])
        L = layer_of(key[0])
        p = per.setdefault(L, {"n": 0, "same": 0, "worst": (0.0, None), "topk": []})
        if key[0].startswith("lid_top_k-"):
            sa, sb = set(a.tolist()), set(b.tolist())
            if sa != sb:
                p["topk"].append(f"{key[0]} symdiff {len(sa ^ sb)}")
                topk_bad.append(L)
        elif a != b:
            int_bad += 1
            p["topk"].append(f"{key[0]} ints differ")
    for L in sorted(per):
        p = per[L]
        w = p["worst"]
        wtxt = f" worst {w[1]} rel {w[0]:.3e}" if w[1] else ""
        tk = f" | {'; '.join(p['topk'])}" if p["topk"] else ""
        print(f"layer {L}: {p['n']} pairs, {p['same']} bit-identical{wtxt}{tk}", file=out)
    worst.sort(reverse=True)
    for r, da, name in worst[:top]:
        print(f"  worst: {name} rel {r:.3e} abs {da:.3e}", file=out)
    only_a = sum(1 for k in ta if k not in tb and keep(k))
    only_b = sum(1 for k in tb if k not in ta and keep(k))
    n = sum(p["n"] for p in per.values())
    same = sum(p["same"] for p in per.values())
    first = min(topk_bad) if topk_bad else None
    print(f"rows only in A {only_a}, only in B {only_b}, same name and another op or shape {shape_skip}",
          file=out)
    print(f"first layer whose top-k set differs: {first if first is not None else 'none'}", file=out)
    ok = same == n and not topk_bad and int_bad == 0
    print(f"set-diff: {n} pairs, {same} bit-identical, {len(topk_bad)} top-k sets differ, "
          f"{int_bad} integer rows differ — {'IDENTICAL' if ok else 'DIFFER'}", file=out)
    return ok


def self_test():
    import io
    with tempfile.TemporaryDirectory(prefix="set-diff-") as t:
        def write(d, vals, ids):
            os.makedirs(d)
            lines = ["# dump_ref — test", "# kind\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top\t"
                     "contig\tlogical\tsrc0\tsrc1",
                     "# int\tname\toccurrence\tof\ttype\ttwin\tlayout\tcount\tbytes\tsum\tabsmax\tfile"]
            array.array("f", vals).tofile(open(os.path.join(d, "l_out-3.0.f32"), "wb"))
            lines.append(f"tensor\tl_out-3\t0\tf32\t{len(vals)}\t1\t1\t1\t{4 * len(vals)}\t0\tADD\t1\t0\t-\t-")
            array.array("f", ids).tofile(open(os.path.join(d, "lid_top_k-2.0.f32"), "wb"))
            array.array("i", ids).tofile(open(os.path.join(d, "lid_top_k-2.0.i32"), "wb"))
            lines.append(f"tensor\tlid_top_k-2\t0\ti32\t{len(ids)}\t1\t1\t1\t{4 * len(ids)}\t0\tCONT\t1\t0\t-\t-")
            lines.append(f"int\tlid_top_k-2\t0\ttensor\ti32\ti32\tflat\t{len(ids)}\t{4 * len(ids)}\t0\t0\t"
                         "lid_top_k-2.0.i32")
            lines.append("# complete\t2\t0")
            open(os.path.join(d, "MANIFEST.tsv"), "w").write("\n".join(lines) + "\n")
        write(f"{t}/a", [1.0, 2.0, float("-inf")], [3, 1, 2])
        write(f"{t}/same", [1.0, 2.0, float("-inf")], [3, 1, 2])
        write(f"{t}/order", [1.0, 2.0, float("-inf")], [1, 2, 3])
        write(f"{t}/off", [1.0, 2.0000002, float("-inf")], [1, 2, 3])
        write(f"{t}/ids", [1.0, 2.0, float("-inf")], [1, 2, 4])
        write(f"{t}/inf", [1.0, 2.0, 0.0], [3, 1, 2])
        sink = io.StringIO()
        fails = []
        if not diff(f"{t}/a", f"{t}/same", out=sink):
            fails.append("identical sets read as different")
        # The top-k order differs, its set does not: the list's own file differs bit for bit.
        sink = io.StringIO()
        diff(f"{t}/a", f"{t}/order", out=sink)
        if ", 0 top-k sets differ" not in sink.getvalue():
            fails.append(f"a reordered top-k counted as another set: {sink.getvalue()}")
        sink = io.StringIO()
        if diff(f"{t}/a", f"{t}/off", out=sink) or "l_out-3 rel" not in sink.getvalue():
            fails.append(f"a one-ulp row not named: {sink.getvalue()}")
        sink = io.StringIO()
        if diff(f"{t}/a", f"{t}/ids", out=sink) or "first layer whose top-k set differs: 2" not in sink.getvalue():
            fails.append(f"a top-k of another set not caught: {sink.getvalue()}")
        sink = io.StringIO()
        if diff(f"{t}/a", f"{t}/inf", out=sink) or "rel inf" not in sink.getvalue():
            fails.append(f"a -inf against a finite value not a mismatch: {sink.getvalue()}")
        os.remove(f"{t}/same/MANIFEST.tsv")
        open(f"{t}/same/MANIFEST.tsv", "w").write("# kind\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\t"
                                                  "sum\top\tcontig\tlogical\tsrc0\tsrc1\n")
        try:
            diff(f"{t}/a", f"{t}/same", out=io.StringIO())
            fails.append("a set without its trailer was read")
        except DiffError:
            pass
    for f in fails:
        print(f"self-test FAIL: {f}", file=sys.stderr)
    print(f"set-diff self-test: {'ok' if not fails else 'FAIL'}")
    return 0 if not fails else 1


def main(argv):
    if argv[1:] == ["--self-test"]:
        return self_test()
    args = argv[1:]
    lo = hi = None
    top = 10
    pos = []
    i = 0
    while i < len(args):
        if args[i] == "--layers" and i + 1 < len(args):
            lo, _, hi = args[i + 1].partition("-")
            lo, hi = int(lo), int(hi)
            i += 2
        elif args[i] == "--top" and i + 1 < len(args):
            top = int(args[i + 1])
            i += 2
        else:
            pos.append(args[i])
            i += 1
    if len(pos) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        return 0 if diff(pos[0], pos[1], lo, hi, top) else 1
    except (DiffError, manifest.ManifestError) as e:
        print(f"set-diff: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
