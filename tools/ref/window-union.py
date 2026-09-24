#!/usr/bin/env python3
"""How many host expert reads a k-row pass shares, from router sets and a placement.

    tools/ref/window-union.py <set> [<set>...] --n-l <spec> [--hot <file>] [--k 2,3,4,6]
                              [--ranges 0-1,2-9,10-19,20-29,30-39]
    tools/ref/window-union.py --self-test

Each <set> is a router_trace directory (tools/ref/router-coverage.py's docstring has the format); a set
whose manifest has no `# complete` trailer is refused.

The placement says which (layer, expert) pairs a card holds; every other selected pair is a host slot.
--n-l is the plan's per-layer card count, either explicit (`2-21:64,22-39:63`; a layer not named keeps
0) or `spread:<card experts>@<first>-<last>`, which replays crates/model/src/placement.rs `spread` —
one more expert per eligible layer in ascending order, cycling — at an equal cost per expert. The plan
line `generate_ds41` prints (`card_experts=`, `n_l=<min>..<max> on <n> layers`) is the check on it.
--hot is a router-hotlist.py file: layer l's card keeps the first n_l ids of its line (rank order);
without it, the id prefix [0, n_l), as the placement does without BLOOMERY_HOT_LIST.

A window is k consecutive positions of one chunk (a window across a chunk edge crosses a context reset
and is skipped). Per window and layer, U = |union of the rows' host experts| and S = the sum of the
rows' host slots; a pass that reads each host expert once reads U, one call per row reads S.
Per set and k:

    predict  the ratio if every position routed independently with the set's own marginals: per layer
             Σ_host (1 − (1 − p_e)^k) over k Σ_host p_e, p_e = count_e / tokens; pooled over layers
    ratio    per window (Σ_l U) / (Σ_l S) — one pass's host reads against k single-row passes' — its
             mean, p10 and p90 over windows; `pooled` is Σ U / Σ S over every window
    slots    mean host slots per row (Σ_l over one row) and mean union per window

then, per layer range, the pooled ratio per k. The last block (`all`) pools every set's windows.
"""
import importlib.util
import os
import sys
import tempfile
from array import array

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("router_coverage", os.path.join(_here, "router-coverage.py"))
rc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rc)

DEFAULT_K = (2, 3, 4, 6)
DEFAULT_RANGES = "0-1,2-9,10-19,20-29,30-39"


def parse_range(text):
    a, sep, b = text.partition("-")
    lo = int(a)
    hi = int(b) if sep else lo
    if hi < lo:
        raise rc.SetError(f"range {text!r} runs backwards")
    return lo, hi


def parse_n_l(spec, layers):
    """Per-layer card counts from an explicit spec or the spread rule."""
    n_l = {l: 0 for l in layers}
    if spec.startswith("spread:"):
        total, _, span = spec[len("spread:"):].partition("@")
        lo, hi = parse_range(span)
        eligible = [l for l in layers if lo <= l <= hi]
        left = int(total)
        if not eligible:
            raise rc.SetError(f"--n-l {spec!r}: no eligible layer in {lo}-{hi}")
        while left > 0:
            for l in eligible:
                if left == 0:
                    break
                n_l[l] += 1
                left -= 1
        return n_l
    for part in spec.split(","):
        span, _, n = part.partition(":")
        lo, hi = parse_range(span)
        for l in range(lo, hi + 1):
            if l not in n_l:
                raise rc.SetError(f"--n-l names layer {l}, which the set does not trace")
            n_l[l] = int(n)
    return n_l


def read_hot(path):
    lists = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            layer, ids = line.rstrip("\n").split("\t")
            lists[int(layer)] = [int(e) for e in ids.split(",")]
    return lists


def card_sets(layers, n_l, n_expert, hot):
    """Per layer, the ids the card holds."""
    out = {}
    for l in layers:
        n = n_l[l]
        if n > n_expert:
            raise rc.SetError(f"layer {l}: n_l {n} is above n_expert {n_expert}")
        if hot is None:
            out[l] = set(range(n))
        else:
            ids = hot.get(l, [])
            if len(ids) < n:
                raise rc.SetError(f"layer {l}: the hot list holds {len(ids)} ids, the plan keeps {n}")
            out[l] = set(ids[:n])
    return out


def predict(counts, tokens, card, k):
    """Expected (union, slots) of k independent rows on one layer's host experts."""
    union = slots = 0.0
    for e, c in enumerate(counts):
        if e in card or c == 0:
            continue
        p = c / tokens
        union += 1.0 - (1.0 - p) ** k
        slots += k * p
    return union, slots


def pct(sorted_v, q):
    return sorted_v[min(len(sorted_v) - 1, int(q * (len(sorted_v) - 1) + 0.5))]


def measure(s, card, ks):
    """Per k: per-window (U, S) summed over layers, and per-layer Σ U, Σ S."""
    T, K, E = s["tokens"], s["n_used"], s["n_expert"]
    chunk = int(s["header"].get("chunk", T)) or T
    starts = {k: [w for w in range(T - k + 1) if w // chunk == (w + k - 1) // chunk] for k in ks}
    win = {k: (array("l", [0]) * len(starts[k]), array("l", [0]) * len(starts[k])) for k in ks}
    per_layer = {k: {} for k in ks}
    row_slots = {}
    for l in s["layers"]:
        ids = rc.read_topk(s, l)
        bit = [0 if e in card[l] else 1 << e for e in range(E)]
        masks = [0] * T
        pops = array("l", [0]) * (T + 1)  # prefix sums of per-row host slots
        for t in range(T):
            m = 0
            for e in ids[t * K:(t + 1) * K]:
                m |= bit[e]
            masks[t] = m
            pops[t + 1] = pops[t] + m.bit_count()
        row_slots[l] = pops[T] / T
        for k in ks:
            wu, ws = win[k]
            su = ss = 0
            for i, w in enumerate(starts[k]):
                m = 0
                for t in range(w, w + k):
                    m |= masks[t]
                u = m.bit_count()
                sl = pops[w + k] - pops[w]
                wu[i] += u
                ws[i] += sl
                su += u
                ss += sl
            per_layer[k][l] = (su, ss)
    return win, per_layer, row_slots


def summary(win_k):
    wu, ws = win_k
    ratios = sorted(u / sl for u, sl in zip(wu, ws) if sl)
    return {
        "windows": len(wu),
        "mean": sum(ratios) / len(ratios) if ratios else float("nan"),
        "p10": pct(ratios, 0.10) if ratios else float("nan"),
        "p90": pct(ratios, 0.90) if ratios else float("nan"),
        "pooled": sum(wu) / sum(ws) if sum(ws) else float("nan"),
        "union": sum(wu) / len(wu) if wu else float("nan"),
    }


def report(set_dirs, spec, hot_path, ks, ranges):
    sets = [rc.read_manifest(d) for d in set_dirs]
    hot = read_hot(hot_path) if hot_path else None
    ranges = [parse_range(r) for r in ranges.split(",")]
    print(f"# window-union — n_l {spec}, card list {hot_path or 'id prefix'}, k {','.join(map(str, ks))}")
    pooled_all = {k: ([], []) for k in ks}
    layers_all = {k: {} for k in ks}
    slots_all = {}
    for s in sets:
        n_l = parse_n_l(spec, s["layers"])
        card = card_sets(s["layers"], n_l, s["n_expert"], hot)
        counts = rc.read_counts(s)
        name = os.path.basename(os.path.normpath(s["dir"]))
        print(f"\n## {name}: {s['tokens']} tokens, chunk {s['header'].get('chunk', '-')}, "
              f"{len(s['layers'])} layers, card {sum(n_l.values())} experts "
              f"(n_l {min(n_l.values())}..{max(n_l.values())}), model {s['header'].get('model_file', '?')}, "
              f"build {s['header'].get('build', '?')}")
        win, per_layer, row_slots = measure(s, card, ks)
        print(f"host slots per row {sum(row_slots.values()):.1f} of {len(s['layers']) * s['n_used']}")
        print("k\twindows\tpredict\tmean\tp10\tp90\tpooled\tunion/window\tslots/window")
        for k in ks:
            pu = ps = 0.0
            for l in s["layers"]:
                u, sl = predict(counts[l], s["tokens"], card[l], k)
                pu += u
                ps += sl
            m = summary(win[k])
            print(f"{k}\t{m['windows']}\t{pu / ps:.3f}\t{m['mean']:.3f}\t{m['p10']:.3f}\t{m['p90']:.3f}\t"
                  f"{m['pooled']:.3f}\t{m['union']:.1f}\t{sum(win[k][1]) / len(win[k][1]):.1f}")
            pooled_all[k][0].extend(win[k][0])
            pooled_all[k][1].extend(win[k][1])
            for l, (u, sl) in per_layer[k].items():
                a = layers_all[k].setdefault(l, [0, 0])
                a[0] += u
                a[1] += sl
        for l, v in row_slots.items():
            slots_all.setdefault(l, []).append(v)
        print_ranges(per_layer, row_slots, ks, ranges)
    if len(sets) > 1:
        print(f"\n## all: {len(sets)} sets")
        print("k\twindows\tmean\tp10\tp90\tpooled")
        for k in ks:
            m = summary(pooled_all[k])
            print(f"{k}\t{m['windows']}\t{m['mean']:.3f}\t{m['p10']:.3f}\t{m['p90']:.3f}\t{m['pooled']:.3f}")
        mean_slots = {l: sum(v) / len(v) for l, v in slots_all.items()}
        print_ranges({k: {l: tuple(v) for l, v in layers_all[k].items()} for k in ks}, mean_slots, ks, ranges)


def print_ranges(per_layer, row_slots, ks, ranges):
    print("layers\t" + "\t".join(f"k={k}" for k in ks) + "\tslots/row")
    for lo, hi in ranges:
        cells = []
        for k in ks:
            u = sum(v[0] for l, v in per_layer[k].items() if lo <= l <= hi)
            sl = sum(v[1] for l, v in per_layer[k].items() if lo <= l <= hi)
            cells.append(f"{u / sl:.3f}" if sl else "-")
        slots = sum(v for l, v in row_slots.items() if lo <= l <= hi)
        print(f"{lo}-{hi}\t" + "\t".join(cells) + f"\t{slots:.1f}")


def main(argv):
    if argv == ["--self-test"]:
        return self_test()
    sets, spec, hot, ks, ranges = [], None, None, DEFAULT_K, DEFAULT_RANGES
    it = iter(argv)
    for a in it:
        if a == "--n-l":
            spec = next(it)
        elif a == "--hot":
            hot = next(it)
        elif a == "--k":
            ks = tuple(int(x) for x in next(it).split(","))
        elif a == "--ranges":
            ranges = next(it)
        elif a.startswith("-"):
            sys.exit(f"unknown flag {a}")
        else:
            sets.append(a)
    if not sets or spec is None or any(k < 1 for k in ks):
        sys.exit(__doc__)
    try:
        report(sets, spec, hot, ks, ranges)
    except rc.SetError as e:
        sys.exit(f"window-union: {e}")
    return 0


def _fake_set(root, name, rows_by_layer, n_expert, n_used, chunk=None):
    d = os.path.join(root, name)
    os.makedirs(d)
    T = len(next(iter(rows_by_layer.values())))
    with open(os.path.join(d, "MANIFEST.tsv"), "w", encoding="utf-8") as f:
        f.write("# router_trace — test\n# model_file\tm.gguf\n# build\tb0\n")
        f.write(f"# tokens\t{T}\n# n_expert\t{n_expert}\n# n_expert_used\t{n_used}\n")
        if chunk:
            f.write(f"# chunk\t{chunk}\n")
        for l in rows_by_layer:
            f.write(f"layer\t{l}\tx\tx\t{T}\t0\t0\ttopk-{l}.u16\n")
        f.write(f"# complete\t{T}\t{len(rows_by_layer)}\n")
    with open(os.path.join(d, "counts.tsv"), "w", encoding="utf-8") as f:
        f.write("# layer\texpert\tcount\n")
        for l, rows in rows_by_layer.items():
            c = [0] * n_expert
            for r in rows:
                for e in r:
                    c[e] += 1
            for e in range(n_expert):
                f.write(f"{l}\t{e}\t{c[e]}\n")
            a = array("H", [e for r in rows for e in r])
            if sys.byteorder != "little":
                a.byteswap()
            with open(os.path.join(d, f"topk-{l}.u16"), "wb") as g:
                g.write(a.tobytes())
    return d


def self_test():
    """Synthetic sets with known answers: sharing, no sharing, the card, chunk edges, the prediction."""
    E, K = 16, 2
    with tempfile.TemporaryDirectory() as root:
        # every row the same two experts, nothing on the card: a k-row pass reads 1/k of k passes
        same = rc.read_manifest(_fake_set(root, "same", {0: [(3, 7)] * 8}, E, K))
        card = card_sets(same["layers"], parse_n_l("0:0", same["layers"]), E, None)
        win, per_layer, slots = measure(same, card, (2, 4))
        assert abs(summary(win[2])["mean"] - 1 / 2) < 1e-12 and abs(summary(win[4])["pooled"] - 1 / 4) < 1e-12
        assert slots[0] == 2.0 and per_layer[4][0] == (2 * 5, 8 * 5), per_layer[4][0]
        # every row its own experts: no sharing, the ratio is 1
        apart = rc.read_manifest(_fake_set(root, "apart", {0: [(2 * t, 2 * t + 1) for t in range(8)]}, E, K))
        win, _, _ = measure(apart, {0: set()}, (3,))
        assert summary(win[3])["mean"] == 1.0 and summary(win[3])["p10"] == 1.0
        # the shared expert 0 sits on the card (hot list first id): what is left does not share
        rows = [(0, t + 1) for t in range(8)]
        shared = rc.read_manifest(_fake_set(root, "shared", {0: rows}, E, K))
        hot_path = os.path.join(root, "hot.txt")
        with open(hot_path, "w", encoding="utf-8") as f:
            f.write("# router-hotlist\n0\t0,5,6\n")
        on_card = card_sets(shared["layers"], parse_n_l("0:1", shared["layers"]), E, read_hot(hot_path))
        assert on_card == {0: {0}}, on_card
        win, _, slots = measure(shared, on_card, (2,))
        assert summary(win[2])["mean"] == 1.0 and slots[0] == 1.0
        win, _, _ = measure(shared, {0: set()}, (2,))  # expert 0 on the host: 3 of 4 slots per pair
        assert summary(win[2])["mean"] == 3 / 4
        # windows do not cross a chunk edge: 8 tokens in chunks of 4 hold 2 x 3 windows of 2, not 7
        chunked = rc.read_manifest(_fake_set(root, "chunked", {0: [(3, 7)] * 8}, E, K, chunk=4))
        win, _, _ = measure(chunked, {0: set()}, (2, 4))
        assert summary(win[2])["windows"] == 6 and summary(win[4])["windows"] == 2
        # the prediction: 16 experts each picked once in 8 rows, p = 1/8: 16 (1 - (7/8)^2) over 2 x 2
        u, sl = predict([1] * E, 8, set(), 2)
        assert abs(u / sl - 16 * (1 - (7 / 8) ** 2) / 4) < 1e-12 and abs(sl - 4.0) < 1e-12
        # the spread rule: 2414 over layers 2..39 is 64 on 2..21, 63 on 22..39, 0 on 0 and 1
        n_l = parse_n_l("spread:2414@2-39", list(range(40)))
        assert [n_l[l] for l in (0, 1, 2, 21, 22, 39)] == [0, 0, 64, 64, 63, 63] and sum(n_l.values()) == 2414
        assert n_l == parse_n_l("2-21:64,22-39:63", list(range(40)))
        try:
            card_sets([0], {0: 4}, E, read_hot(hot_path))
            raise AssertionError("a hot list shorter than n_l was accepted")
        except rc.SetError:
            pass
    print("window-union: self-test ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
