#!/usr/bin/env python3
"""The logits rows a generate binary's `--rows FILE` writes, read and compared.

`--rows FILE` (crates/gpu-gates/src/generate.rs, `Diag`) appends, run after run, the rows `--top2 K` reads: each row
the vocabulary's f32 logits, little-endian, as the model made them, with no header. Each run's rows are followed by a
`rows` record (`rows=K n=<entries a row> bytes= path=`, read through records.py), whose `n` is this tool's --n.

  rows.py diff A B --n N [--a-from I] [--b-from J] [--count K]
        rows I.. of A against rows J.. of B (K rows; default every row both hold from there), a line a row:
        `bit-equal`, or the entries that differ (a signed zero against its twin counts), the largest |a - b| and its
        id, and the RMS of a - b over the row. Exit 0 when every row compared is bit-equal, 1 otherwise.
  rows.py picks FILE --n N [--skip a,b]
        each row's pick, one a line: its largest entry with the --skip ids passed over (an `ignore eos` record's
        ids), first of equals — the token the run took from the row; a NaN is refused by name.
  rows.py --self-test

A file whose size is not a whole number of rows of N entries, a row index past the file, and a NaN in a row a pick
reads are refused by name (exit 2).
"""
import array
import math
import os
import sys


def fail(msg):
    print(f"rows.py: {msg}", file=sys.stderr)
    sys.exit(2)


def rows_of(path, n):
    """The file's rows, each `n` entries as raw bytes."""
    size = os.path.getsize(path)
    width = 4 * n
    if n <= 0 or size % width:
        fail(f"{path}: {size} bytes is not a whole number of rows of {n} f32")
    with open(path, "rb") as f:
        raw = f.read()
    return [raw[i:i + width] for i in range(0, size, width)]


def floats(b):
    a = array.array("f")
    a.frombytes(b)
    if sys.byteorder != "little":
        a.byteswap()
    return a


def diff(x, y):
    """`bit-equal`, or the differing entries, the largest |x - y| with its id, and the RMS."""
    if x == y:
        return "bit-equal"
    a, b = floats(x), floats(y)
    cnt, mx, at, sq = 0, 0.0, -1, 0.0
    for i, (u, v) in enumerate(zip(a, b)):
        if u != v or math.copysign(1, u) != math.copysign(1, v) or (u != u) != (v != v):
            cnt += 1
            d = abs(u - v)
            if d == d:
                sq += d * d
                if d > mx:
                    mx, at = d, i
    return f"differs: {cnt} of {len(a)} entries, max {mx:.6g} at id {at}, rms {math.sqrt(sq / len(a)):.6g}"


def pick(b, skip, where):
    """The largest entry with `skip` passed over, first of equals."""
    best, at = -math.inf, -1
    for i, v in enumerate(floats(b)):
        if v != v:
            fail(f"{where}: NaN at id {i}")
        if i in skip:
            continue
        if v > best:
            best, at = v, i
    return at


def flag(args, name, default):
    if name not in args:
        return default
    i = args.index(name)
    if i + 1 >= len(args):
        fail(f"{name} needs a value")
    v = args[i + 1]
    del args[i:i + 2]
    return v


def cmd_diff(args):
    n = int(flag(args, "--n", "0"))
    fa, fb = int(flag(args, "--a-from", "0")), int(flag(args, "--b-from", "0"))
    count = flag(args, "--count", None)
    if len(args) != 2:
        fail("diff takes two files: " + __doc__)
    a, b = rows_of(args[0], n), rows_of(args[1], n)
    if fa > len(a) or fb > len(b):
        fail(f"--a-from {fa} / --b-from {fb} past {len(a)} / {len(b)} rows")
    k = min(len(a) - fa, len(b) - fb) if count is None else int(count)
    if fa + k > len(a) or fb + k > len(b):
        fail(f"{k} rows from {fa} and {fb} pass the files' {len(a)} and {len(b)} rows")
    equal = True
    for r in range(k):
        d = diff(a[fa + r], b[fb + r])
        equal &= d == "bit-equal"
        print(f"row A{fa + r} B{fb + r}: {d}")
    print(f"{k} rows: {'every row bit-equal' if equal else 'NOT bit-equal'}")
    return 0 if equal else 1


def cmd_picks(args):
    n = int(flag(args, "--n", "0"))
    skip = {int(s) for s in flag(args, "--skip", "").split(",") if s}
    if len(args) != 1:
        fail("picks takes one file: " + __doc__)
    for r, row in enumerate(rows_of(args[0], n)):
        print(f"row {r}: {pick(row, skip, f'{args[0]} row {r}')}")
    return 0


def self_test():
    import subprocess
    import tempfile

    def row(*vals):
        a = array.array("f", vals)
        if sys.byteorder != "little":
            a.byteswap()
        return a.tobytes()

    assert diff(row(1, 2), row(1, 2)) == "bit-equal"
    assert diff(row(1, 2), row(1, 2.5)) == "differs: 1 of 2 entries, max 0.5 at id 1, rms 0.353553", diff(
        row(1, 2), row(1, 2.5))
    assert diff(row(0.0, 1), row(-0.0, 1)).startswith("differs: 1 of 2"), "a signed zero is a bit difference"
    assert pick(row(1, 9, 5), set(), "x") == 1 and pick(row(1, 9, 5), {1}, "x") == 2
    assert pick(row(3, 3), set(), "x") == 0, "first of equals"
    me = os.path.abspath(__file__)
    with tempfile.TemporaryDirectory() as t:
        pa, pb, bad = (os.path.join(t, x) for x in ("a", "b", "bad"))
        open(pa, "wb").write(row(1, 9, 5) + row(4, 2, 7))
        open(pb, "wb").write(row(4, 2, 7) + row(4, 2, 7.5))
        open(bad, "wb").write(row(1, 2) + b"\0")

        def run(*a):
            p = subprocess.run([sys.executable, me, *a], capture_output=True, text=True)
            return p.returncode, p.stdout, p.stderr

        rc, out, _ = run("diff", pa, pb, "--n", "3", "--a-from", "1", "--count", "1")
        assert rc == 0 and "row A1 B0: bit-equal" in out, out
        rc, out, _ = run("diff", pa, pb, "--n", "3", "--a-from", "1", "--b-from", "1")
        assert rc == 1 and "row A1 B1: differs: 1 of 3" in out and "NOT bit-equal" in out, out
        rc, out, _ = run("picks", pa, "--n", "3", "--skip", "1")
        assert rc == 0 and out.splitlines() == ["row 0: 2", "row 1: 2"], out
        rc, _, err = run("diff", pa, bad, "--n", "2")
        assert rc == 2 and "is not a whole number of rows" in err, err
        rc, _, err = run("diff", pa, pb, "--n", "3", "--count", "3")
        assert rc == 2 and "pass the files'" in err, err
    print("rows.py self-test: PASS")
    return 0


def main(argv):
    if argv[1:] == ["--self-test"]:
        return self_test()
    if len(argv) < 2 or argv[1] not in ("diff", "picks"):
        fail(__doc__)
    args = argv[2:]
    return cmd_diff(args) if argv[1] == "diff" else cmd_picks(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
