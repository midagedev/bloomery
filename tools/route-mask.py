#!/usr/bin/env python3
"""Make a BLOOMERY_ROUTE_MASK sidecar from routing censuses.

    tools/route-mask.py CENSUS [CENSUS ...] --keep-fraction F [-o MASK]
    tools/route-mask.py --self-test

A census is the file `BLOOMERY_ROUTE_CENSUS` makes (crates/runtime/src/routedial.rs `CensusTable`):

    # route-census v1
    n_layer 43
    n_expert 512
    n_used 10
    <layer> <expert> <picks> <weight_sum>        one line a cell with a pick

Several censuses (one per prompt set, or per run) are added cell by cell; their shapes must match.
Per layer an expert's score is its pick count times its mean gate weight (the weight sum of a
picked expert: picks * (weight_sum / picks)). The layer keeps the ceil(F * n_expert) experts of the
highest score, never fewer than n_used, and bans the rest; a tie goes to the expert with more
picks, then the lower id. A layer with no pick in any census is not a routed layer and gets no
line. The mask is one line a banned layer, `<layer> <id> <id> ...`, ids ascending, the sidecar's
format (`RouteMask::parse`). The kept pick-mass share of each layer (the kept experts' share of
its weight sum) and of its picks goes to stderr, then the summary line.
"""

import argparse
import math
import sys
import tempfile
from pathlib import Path

TITLE = "# route-census v1"


class CensusError(Exception):
    pass


class Census:
    """The cells of one or more censuses, summed."""

    def __init__(self, n_layer, n_expert, n_used):
        self.n_layer, self.n_expert, self.n_used = n_layer, n_expert, n_used
        self.picks = {}
        self.weight = {}

    def shape(self):
        return (self.n_layer, self.n_expert, self.n_used)

    def add(self, layer, expert, picks, weight):
        key = (layer, expert)
        self.picks[key] = self.picks.get(key, 0) + picks
        self.weight[key] = self.weight.get(key, 0.0) + weight

    def merge(self, other):
        if self.shape() != other.shape():
            raise CensusError(f"a census of shape {other.shape()} merged into one of {self.shape()}")
        for key, p in other.picks.items():
            self.add(key[0], key[1], p, other.weight[key])


def parse(text, name="census"):
    lines = [(i + 1, l.strip()) for i, l in enumerate(text.splitlines())]
    if not lines or lines[0][1] != TITLE:
        raise CensusError(f"{name}: the first line is not {TITLE!r}")
    shape = []
    for i, key in enumerate(("n_layer", "n_expert", "n_used"), start=1):
        if i >= len(lines):
            raise CensusError(f"{name}: no {key} line")
        n, line = lines[i]
        words = line.split()
        if len(words) != 2 or words[0] != key or not words[1].isdigit():
            raise CensusError(f"{name} line {n}: expected `{key} <number>`")
        shape.append(int(words[1]))
    c = Census(*shape)
    for n, line in lines[4:]:
        if not line or line.startswith("#"):
            continue
        w = line.split()
        if len(w) != 4:
            raise CensusError(f"{name} line {n}: expected `<layer> <expert> <picks> <weight_sum>`")
        try:
            layer, expert, picks, weight = int(w[0]), int(w[1]), int(w[2]), float(w[3])
        except ValueError:
            raise CensusError(f"{name} line {n}: a field that is not a number") from None
        if not (0 <= layer < c.n_layer and 0 <= expert < c.n_expert) or picks < 0:
            raise CensusError(f"{name} line {n}: cell ({layer}, {expert}) outside {c.n_layer}x{c.n_expert}")
        c.add(layer, expert, picks, weight)
    return c


def make_mask(c, keep_fraction):
    """The banned ids per layer (ascending) and the per-layer shares: layer -> (banned, mass, picks)."""
    if not 0.0 < keep_fraction <= 1.0:
        raise CensusError(f"--keep-fraction {keep_fraction}: a fraction above 0 up to 1")
    keep_n = min(c.n_expert, max(c.n_used, math.ceil(keep_fraction * c.n_expert)))
    layers = sorted({layer for layer, _ in c.picks})
    out = {}
    for layer in layers:
        cells = []
        for e in range(c.n_expert):
            p = c.picks.get((layer, e), 0)
            w = c.weight.get((layer, e), 0.0)
            mean = w / p if p else 0.0
            cells.append((p * mean, p, e))
        cells.sort(key=lambda t: (-t[0], -t[1], t[2]))
        kept, dropped = cells[:keep_n], cells[keep_n:]
        total = sum(t[0] for t in cells)
        total_p = sum(t[1] for t in cells)
        mass = sum(t[0] for t in kept) / total if total > 0 else 1.0
        share = sum(t[1] for t in kept) / total_p if total_p else 1.0
        out[layer] = (sorted(t[2] for t in dropped), mass, share)
    return out


def render(mask, sources, keep_fraction):
    lines = [f"# route-mask keep={keep_fraction} from {', '.join(sources)}"]
    for layer, (banned, _, _) in sorted(mask.items()):
        if banned:
            lines.append(f"{layer} " + " ".join(str(e) for e in banned))
    return "\n".join(lines) + "\n"


def report(mask, keep_fraction):
    rows = []
    for layer, (banned, mass, share) in sorted(mask.items()):
        rows.append(f"layer {layer}: banned {len(banned)} kept weight-mass {mass:.4f} kept picks {share:.4f}")
    if mask:
        masses = [m for _, m, _ in mask.values()]
        rows.append(
            f"keep {keep_fraction}: {len(mask)} layers, banned {sum(len(b) for b, _, _ in mask.values())} experts; "
            f"kept weight-mass min {min(masses):.4f} mean {sum(masses) / len(masses):.4f}"
        )
    return rows


def self_test():
    # The census text the Rust writer makes (routedial.rs `a_census_round_trips_and_merges`), by hand.
    text = f"{TITLE}\nn_layer 3\nn_expert 8\nn_used 2\n0 1 5 2.5\n0 2 1 0.5\n2 7 1 0.1\n"
    c = parse(text)
    assert c.shape() == (3, 8, 2) and c.picks[(0, 1)] == 5 and c.weight[(0, 1)] == 2.5
    # A merge adds cell by cell and refuses another shape.
    d = parse(text)
    c.merge(d)
    assert c.picks[(0, 1)] == 10 and c.weight[(0, 2)] == 1.0
    try:
        c.merge(Census(3, 9, 2))
    except CensusError:
        pass
    else:
        raise AssertionError("a merge of another shape must be refused")
    # Each malformed census is refused by name.
    for bad in (
        "n_layer 3\n",
        f"{TITLE}\nn_layer 3\nn_expert 8\n",
        f"{TITLE}\nn_layer 3\nn_expert 8\nn_used 2\n0 1 5\n",
        f"{TITLE}\nn_layer 3\nn_expert 8\nn_used 2\n0 8 1 1.0\n",
        f"{TITLE}\nn_layer 3\nn_expert 8\nn_used 2\n0 1 x 1.0\n",
    ):
        try:
            parse(bad)
        except CensusError:
            pass
        else:
            raise AssertionError(f"{bad!r} must be refused")
    # Layer 0: experts 0..7 with picks 8,7,...,1 at a mean weight 0.1; keep 0.5 keeps 4, bans 4..7.
    cells = "".join(f"0 {e} {8 - e} {(8 - e) * 0.1}\n" for e in range(8))
    c = parse(f"{TITLE}\nn_layer 2\nn_expert 8\nn_used 2\n{cells}")
    m = make_mask(c, 0.5)
    assert list(m) == [0], "only the layer with a pick gets a line"
    banned, mass, share = m[0]
    assert banned == [4, 5, 6, 7], banned
    assert abs(share - (8 + 7 + 6 + 5) / 36) < 1e-12, share
    assert abs(mass - (8 + 7 + 6 + 5) / 36) < 1e-9, mass
    # The floor: never fewer than n_used experts kept.
    assert make_mask(c, 0.01)[0][0] == [2, 3, 4, 5, 6, 7]
    # Keep everything bans nothing, and the mask text has no line for it.
    assert make_mask(c, 1.0)[0][0] == []
    assert render(make_mask(c, 1.0), ["x"], 1.0).count("\n") == 1
    # A tie goes to the expert with more picks, then the lower id.
    t = parse(f"{TITLE}\nn_layer 1\nn_expert 4\nn_used 1\n0 0 1 1.0\n0 1 2 2.0\n0 2 2 2.0\n")
    assert make_mask(t, 0.5)[0][0] == [0, 3]
    # The sidecar is one line, ids ascending (RouteMask::parse's format).
    assert render(m, ["a.txt"], 0.5).splitlines()[1] == "0 4 5 6 7"
    for bad in (0.0, 1.5, -1.0):
        try:
            make_mask(c, bad)
        except CensusError:
            pass
        else:
            raise AssertionError(f"--keep-fraction {bad} must be refused")
    # The command line: two censuses added, the mask written to a file.
    with tempfile.TemporaryDirectory() as d:
        a, b, out = (Path(d) / n for n in ("a.txt", "b.txt", "mask.txt"))
        a.write_text(f"{TITLE}\nn_layer 1\nn_expert 4\nn_used 1\n0 0 1 1.0\n0 1 1 1.0\n")
        b.write_text(f"{TITLE}\nn_layer 1\nn_expert 4\nn_used 1\n0 0 3 3.0\n0 3 1 1.0\n")
        assert main([str(a), str(b), "--keep-fraction", "0.5", "-o", str(out)]) == 0
        assert out.read_text().splitlines()[1] == "0 2 3", out.read_text()
    print("route-mask: self-test ok")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("censuses", nargs="*", type=Path)
    ap.add_argument("--keep-fraction", type=float)
    ap.add_argument("-o", "--out", type=Path, help="the mask file (default: stdout)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.censuses or a.keep_fraction is None:
        ap.error("a census and --keep-fraction are required")
    try:
        merged = None
        for path in a.censuses:
            c = parse(path.read_text(), str(path))
            if merged is None:
                merged = c
            else:
                merged.merge(c)
        mask = make_mask(merged, a.keep_fraction)
    except (CensusError, OSError) as e:
        print(f"route-mask: {e}", file=sys.stderr)
        return 2
    text = render(mask, [str(p) for p in a.censuses], a.keep_fraction)
    if a.out:
        a.out.write_text(text)
    else:
        sys.stdout.write(text)
    for row in report(mask, a.keep_fraction):
        print(row, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
