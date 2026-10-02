#!/usr/bin/env python3
"""Per-position distance between two sets of hidden-state rows: clef_hidden's output against a
hidden-qwen35 reference set (tools/ref/hidden.sh), or two sets against each other (mainline on the
CUDA backend against its CPU twin, the oracle's own floor).

    hidden-diff.py A B [--top K] [--with C]
                                     each side a raw f32 file (`[positions][width]`, clef_hidden --out)
                                     or a set directory (its MANIFEST.tsv's `result_norm` row, read
                                     through tools/bloomery/manifest.py); the width is the set's ne0,
                                     5120 when both sides are raw files
    hidden-diff.py --self-test

Per position t: rel = ||A_t - B_t|| / ||B_t|| and the max abs error. Printed: the positions, the
quantiles of rel (median, p90, p99, max), how many positions sit above 0.12 and 0.32 (the gate's
median and worst bands, gate_clef_hidden), the rel median of each quarter of the prompt (an error
that grows with position reads as a rising row), and the K worst positions with their rel, max abs
and ||B_t||. `--with C` also reads C against B and prints how many of the positions above 0.32
for A are above it for C too: a tail both sides share belongs to the prompt and the rule, not to
either side's engine. A side whose length is not a whole number of rows, or two sides of different lengths,
is refused by name (exit 2).
"""
import importlib.util
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "manifest", os.path.join(HERE, "..", "bloomery", "manifest.py"))
manifest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(manifest)

RAW_WIDTH = 5120
BANDS = (0.12, 0.32)


class DiffError(ValueError):
    pass


def side(path):
    """The rows of one side, `[positions, width]` f32, and its width's source."""
    if os.path.isdir(path):
        m = manifest.read(path)
        rows = [r for r in m.rows("tensor") if r["name"] == "result_norm"]
        if len(rows) != 1:
            raise DiffError(f"{path}: {len(rows)} result_norm rows, one wanted")
        width = int(rows[0]["ne0"])
        data = np.fromfile(os.path.join(path, "result_norm.0.f32"), dtype="<f4")
    else:
        width = RAW_WIDTH
        data = np.fromfile(path, dtype="<f4")
    if data.size == 0 or data.size % width:
        raise DiffError(f"{path}: {data.size} values are not whole rows of {width}")
    return data.reshape(-1, width)


def distances(a, b):
    """Per-position rel L2 of a against b (on b's norm) and max abs error, in f64."""
    if a.shape != b.shape:
        raise DiffError(f"the sides hold {a.shape} and {b.shape} values")
    d = a.astype(np.float64) - b.astype(np.float64)
    den = np.sqrt((b.astype(np.float64) ** 2).sum(axis=1))
    rel = np.sqrt((d ** 2).sum(axis=1)) / np.maximum(den, np.finfo(np.float64).tiny)
    rel[~np.isfinite(rel)] = np.inf
    return rel, np.abs(d).max(axis=1), den


def report(a, b, top):
    rel, mx, den = distances(a, b)
    n = rel.size
    q = np.quantile(rel, [0.5, 0.9, 0.99, 1.0])
    lines = [f"positions {n}  rel median {q[0]:.4e}  p90 {q[1]:.4e}  p99 {q[2]:.4e}  max {q[3]:.4e}"]
    lines.append("  above " + "  ".join(f"{t}: {int((rel > t).sum())}" for t in BANDS))
    quarters = [rel[i * n // 4:(i + 1) * n // 4] for i in range(4)]
    lines.append("  quarter medians " + " ".join(f"{np.median(x):.4e}" for x in quarters if x.size))
    for t in np.argsort(-rel, kind="stable")[:top]:
        lines.append(f"  pos {t}: rel {rel[t]:.4e}  max abs {mx[t]:.4e}  |B| {den[t]:.4e}")
    return lines


def overlap(a, c, b, cut=BANDS[1]):
    """The positions above `cut` for a against b and for c against b, and how many both hold."""
    ra = set(np.flatnonzero(distances(a, b)[0] > cut).tolist())
    rc = set(np.flatnonzero(distances(c, b)[0] > cut).tolist())
    return [f"  above {cut}: A {len(ra)}, C {len(rc)}, both {len(ra & rc)}"]


def self_test():
    rng = np.random.default_rng(7)
    b = rng.standard_normal((8, 4)).astype("<f4")
    a = b.copy()
    a[5] *= 2.0
    rel, mx, _ = distances(a, b)
    assert np.allclose(rel[5], 1.0) and np.all(rel[[0, 1, 2, 3, 4, 6, 7]] == 0.0), rel
    assert np.isclose(mx[5], np.abs(b[5]).max()), mx
    c = b.copy()
    c[[5, 6]] *= 3.0
    assert overlap(a, c, b) == ["  above 0.32: A 1, C 2, both 1"], overlap(a, c, b)
    lines = report(a, b, 1)
    assert lines[2].startswith("  quarter medians") and lines[3].startswith("  pos 5: rel 1.0000e+00"), lines
    with tempfile.TemporaryDirectory() as t:
        raw = os.path.join(t, "x.f32")
        np.zeros(RAW_WIDTH + 1, dtype="<f4").tofile(raw)
        try:
            side(raw)
            raise AssertionError("a ragged file was read")
        except DiffError as e:
            assert "not whole rows" in str(e), e
        d = os.path.join(t, "set")
        os.mkdir(d)
        b.tofile(os.path.join(d, "result_norm.0.f32"))
        with open(os.path.join(d, "MANIFEST.tsv"), "w") as fh:
            fh.write("# kind\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top\tcontig\tlogical\tsrc0\tsrc1\n")
            fh.write("tensor\tresult_norm\t0\tf32\t4\t8\t1\t1\t128\t0\tRMS_NORM\t1\t0\t-\t-\n")
            fh.write("# complete\t1\t0\n")
        assert side(d).shape == (8, 4)
        try:
            distances(side(d), b[:7])
            raise AssertionError("sides of two lengths were compared")
        except DiffError as e:
            assert "hold" in str(e), e
    print("hidden-diff self-test: ok")
    return 0


def main(argv):
    if argv == ["--self-test"]:
        return self_test()
    top, args, other = 10, [], None
    it = iter(argv)
    for a in it:
        if a == "--top":
            top = int(next(it))
        elif a == "--with":
            other = next(it)
        else:
            args.append(a)
    if len(args) != 2:
        print(__doc__, file=sys.stderr)
        return 64
    try:
        a, b = side(args[0]), side(args[1])
        lines = report(a, b, top)
        if other is not None:
            lines += overlap(a, side(other), b)
        for line in lines:
            print(line)
    except (DiffError, manifest.ManifestError, OSError) as e:
        print(f"hidden-diff: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
