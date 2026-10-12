#!/usr/bin/env python3
"""The `[loop]` floor of a depth runner's decode row: how few distinct ids a generated text of
n ids may hold before the row is read as a decode loop.

A paged plan's cost of a token depends on its routing, so it depends on the text. The runner's
`distinct_tokens` column counts the distinct ids of the steps an arm generated; a greedy run that
fell into a loop holds a few (93 of 1024 on a paged plan's plain arm) where the same prompt's
text holds hundreds (412 of 1024 on its drafted arm), and a speed read across such a pair measures
two different texts. The floor says which side of that a row is on.

Derivation (no number is written down here): the floor for window n is the 1st percentile of
distinct/n over every n-id window of the real corpora, `corpus-prose.ids` and `corpus-code.ids`
(one id a line), pooled. Real text, by the windows' own spread, falls under the floor one window
in a hundred; a loop falls far under it. The percentile is the lower nearest rank: with W
windows sorted ascending, the window at index ceil(W / 100) - 1. The window sizes are whole, so
the floor is a whole count of distinct ids, `distinct=`, and a row is a loop when it holds fewer
(`floor=` is that count over n, for the eye): the comparison is on integers, so a row at the
floor's own window is never tagged by a rounded ratio. A file shorter than n has no window; no
window at all is refused by name. The runner calls this once per window size, on the corpora its
profile holds, and prints the floor beside the files and the window count it came from.

    loop-floor.py --n N FILE [FILE...]     prints `floor=<f> distinct=<k> windows=<W> n=<N> files=<K>`
    loop-floor.py --self-test
"""

import math
import sys
from collections import Counter

PERCENT = 1


class Refused(Exception):
    """A refusal by name: the file or the window size this tool cannot answer for."""


def read_ids(path):
    """The ids of `path`, one a line; a line that is no id names its file and number."""
    ids = []
    with open(path, encoding="ascii") as f:
        for i, line in enumerate(f, 1):
            text = line.rstrip("\n")
            if not text.isascii() or not text.isdigit():
                raise Refused(f"line {i} of {path} ({text!r}) is no id: the file is one id a line")
            ids.append(int(text))
    return ids


def window_counts(ids, n):
    """The distinct ids of every n-id window of `ids`, in order: a sliding count, one step a window."""
    if n < 1:
        raise Refused(f"a window of {n} ids")
    if len(ids) < n:
        return []
    held = Counter(ids[:n])
    out = [len(held)]
    for i in range(n, len(ids)):
        gone = ids[i - n]
        held[gone] -= 1
        if held[gone] == 0:
            del held[gone]
        held[ids[i]] += 1
        out.append(len(held))
    return out


def percentile_floor(counts, percent=PERCENT):
    """The lower nearest rank of `counts` at `percent`: ascending, index ceil(W * percent / 100) - 1."""
    if not counts:
        raise Refused("no window: every file is shorter than the window")
    ranked = sorted(counts)
    return ranked[max(0, math.ceil(len(ranked) * percent / 100) - 1)]


def floor_of(paths, n):
    """(floor in distinct ids, windows) over the pooled windows of the files in `paths`."""
    counts = []
    for p in paths:
        counts.extend(window_counts(read_ids(p), n))
    return percentile_floor(counts), len(counts)


def self_test():
    import os
    import tempfile

    fails = []

    def eq(name, got, want):
        if got != want:
            fails.append(f"{name}: got {got!r}, want {want!r}")

    def refuses(name, fn, text):
        try:
            fn()
        except Refused as e:
            if text not in str(e):
                fails.append(f"{name}: refused with {e!s}, want {text!r}")
        else:
            fails.append(f"{name}: not refused")

    # The sliding count equals a recount of each window.
    ids = [1, 2, 1, 3, 3, 3, 4, 1, 2, 2]
    for n in (1, 3, 4, 10):
        want = [len(set(ids[i:i + n])) for i in range(len(ids) - n + 1)]
        eq(f"sliding n={n}", window_counts(ids, n), want)
    eq("shorter than the window", window_counts([1, 2], 3), [])
    refuses("zero window", lambda: window_counts(ids, 0), "a window of 0 ids")
    # The lower nearest rank at 1 %: 100 windows read the smallest, 101 the second smallest, 200 the second.
    eq("rank 100", percentile_floor(list(range(100, 0, -1))), 1)
    eq("rank 101", percentile_floor(list(range(1, 102))), 2)
    eq("rank 200", percentile_floor(list(range(1, 201))), 2)
    eq("rank 1", percentile_floor([7]), 7)
    refuses("no ratios", lambda: percentile_floor([]), "no window")
    # Files pooled: a 300-id file of a 3-id loop (3 distinct in every window of 10) and a 300-id file of
    # all-distinct ids (10) pool to 582 windows; the lowest 1 % (6 of them) are the loop's, so the floor is
    # 3, and an all-distinct file alone has floor 10.
    with tempfile.TemporaryDirectory() as d:
        loop = os.path.join(d, "loop.ids")
        flat = os.path.join(d, "flat.ids")
        bad = os.path.join(d, "bad.ids")
        short = os.path.join(d, "short.ids")
        with open(loop, "w") as f:
            f.write("".join(f"{i % 3}\n" for i in range(300)))
        with open(flat, "w") as f:
            f.write("".join(f"{i}\n" for i in range(300)))
        with open(bad, "w") as f:
            f.write("1\n2\nx3\n")
        with open(short, "w") as f:
            f.write("1\n2\n")
        eq("pooled", floor_of([loop, flat], 10), (3, 582))
        eq("flat alone", floor_of([flat], 10), (10, 291))
        eq("a short file adds no window", floor_of([flat, short], 10), (10, 291))
        refuses("not an id", lambda: floor_of([bad], 2), "line 3 of")
        refuses("all short", lambda: floor_of([short], 10), "no window")
        rc = main(["--n", "10", loop, flat], out=lambda s: None)
        eq("main rc", rc, 0)
        lines = []
        main(["--n", "10", loop, flat], out=lines.append)
        eq("main line", lines, ["floor=0.3000 distinct=3 windows=582 n=10 files=2"])
    print(f"self-test: {'FAIL' if fails else 'ok'} ({len(fails)} failures)")
    for f in fails:
        print(f"self-test FAIL: {f}", file=sys.stderr)
    return 1 if fails else 0


def main(argv, out=print):
    if argv == ["--self-test"]:
        return self_test()
    if len(argv) < 3 or argv[0] != "--n" or not argv[1].isdigit():
        print("usage: loop-floor.py --n N FILE [FILE...] | --self-test", file=sys.stderr)
        return 64
    n = int(argv[1])
    try:
        floor, windows = floor_of(argv[2:], n)
    except (Refused, OSError, ValueError) as e:
        print(f"loop-floor.py: {e}", file=sys.stderr)
        return 2
    out(f"floor={floor / n:.4f} distinct={floor} windows={windows} n={n} files={len(argv) - 2}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
