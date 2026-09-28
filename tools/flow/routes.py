#!/usr/bin/env python3
"""The V4.1 prompt's routing at the id prefix, from router sets: the flow model's real-text arms.

    tools/flow/routes.py build [--data DIR] [--sets prose,code,korean] [--groups 1,2,8]
    tools/flow/routes.py --self-test

build reads each router set (tools/ref/router-coverage.py's readers: a set without its `# complete`
trailer is refused) under DIR (default $BLOOMERY_DATA/router; the box's sets, or a copy of them) and
writes routes-idprefix.tsv beside this file, which tools/flow/ds41_prefill.py reads for the prompts
`prose`, `code` and `korean` without a hot list. The card holds each layer's id prefix [0, n_l): n_l is
placement (a)'s, the plan record's `card_experts` spread over its last `n_l_layers` layers one expert a
layer in ascending order, cycling (crates/model/src/placement.rs `spread`; the plan is
plans/ds41-p512-ced-on.rec).

A group of G batches is a window of G x 512 consecutive positions of the set, windows back to back from
position 0 over the set's whole chunks. Per window and layer:
  host  the host experts the window routes to, in the order a causal pick at the group's first route
        sees them: by their count in the window's first 512 positions, then by the whole window's, ties
        to the lower id; each at the window's own count
  card  the card experts [0, n_l), by the window's count, ties to the lower id
Per (set, G, layer, list) the file holds, rank by rank, the sum over the windows of that rank's count
(an integer; a rank a window does not reach adds 0), and the window count: the flow model's columns
per 512 positions at a rank are total / (windows x G).

Python 3 standard library only.
"""
import importlib.util
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "routes-idprefix.tsv")
PLAN = os.path.join(HERE, "plans", "ds41-p512-ced-on.rec")
SETS = ("prose", "code", "korean")
GROUPS = (1, 2, 8)
BATCH = 512          # body/prefill.rs T_MAX: one batch's positions


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rc = _load("router_coverage", os.path.join(HERE, "..", "ref", "router-coverage.py"))
records = _load("records", os.path.join(HERE, "..", "bloomery", "records.py"))


class RouteError(Exception):
    pass


def spread(total, layers):
    """crates/model/src/placement.rs `spread`: one more expert a layer in ascending order, cycling."""
    n = {l: 0 for l in layers}
    left = total
    while left > 0:
        for l in layers:
            if left == 0:
                break
            n[l] += 1
            left -= 1
    return n


def plan_n_l(n_layers, path=PLAN):
    """Placement (a)'s card experts per layer, from the engine's plan record."""
    rec = records.first(records.read(path), "plan")
    if rec is None:
        raise RouteError(f"{path}: no `plan` record (just records-refresh writes it)")
    first = n_layers - rec["n_l_layers"]
    n = spread(rec["card_experts"], list(range(first, n_layers)))
    got = [n[l] for l in range(first, n_layers)]
    want = [int(x) for x in str(rec["n_l"]).split("..")]
    if [min(got), max(got)] != [want[0], want[-1]]:
        raise RouteError(f"{path}: spreading {rec['card_experts']} over {rec['n_l_layers']} layers gives "
                         f"n_l {min(got)}..{max(got)}, the plan prints {rec['n_l']}")
    return [n.get(l, 0) for l in range(n_layers)]


def windows(n_tok, chunk, size):
    """The windows of `size` positions back to back from 0 over the set's whole chunks."""
    end = n_tok // chunk * chunk
    return [(t, t + size) for t in range(0, end - size + 1, size)]


def counts(ids, n_expert, n_used, t0, t1):
    c = [0] * n_expert
    for e in ids[t0 * n_used:t1 * n_used]:
        c[e] += 1
    return c


def build_set(s, n_l, groups, batch=BATCH):
    """{G: (windows, [host totals per layer], [card totals per layer])} of one router set."""
    E, K = s["n_expert"], s["n_used"]
    chunk = s["header"].get("chunk")
    if not chunk:
        raise RouteError(f"{s['dir']}: the manifest names no `chunk`")
    if len(n_l) != len(s["layers"]) or s["layers"] != list(range(len(n_l))):
        raise RouteError(f"{s['dir']}: layers {s['layers'][0]}..{s['layers'][-1]}, the plan places {len(n_l)}")
    out = {}
    ids = {l: rc.read_topk(s, l) for l in s["layers"]}
    for g in groups:
        wins = windows(s["tokens"], int(chunk), g * batch)
        if not wins:
            raise RouteError(f"{s['dir']}: no window of {g} x {batch} positions in {s['tokens']} tokens")
        host, card = [], []
        for l in s["layers"]:
            h_acc, c_acc = [0] * E, [0] * n_l[l]
            for t0, t1 in wins:
                c = counts(ids[l], E, K, t0, t1)
                c0 = counts(ids[l], E, K, t0, t0 + batch)
                hs = sorted((e for e in range(n_l[l], E) if c[e] > 0), key=lambda e: (-c0[e], -c[e], e))
                cd = sorted(range(n_l[l]), key=lambda e: (-c[e], e))
                for r, e in enumerate(hs):
                    h_acc[r] += c[e]
                for r, e in enumerate(cd):
                    c_acc[r] += c[e]
            host.append([x for x in h_acc if x > 0])
            card.append(c_acc)
        out[g] = (len(wins), host, card)
    return out


def text(built, sets, n_l, groups, argv):
    lines = [f"# tools/flow/routes.py {' '.join(argv)}: the id-prefix routing of the V4.1 prompt, placement (a)"
             " (read by tools/flow/ds41_prefill.py; the format is routes.py's docstring)",
             "# n_l\t" + ",".join(map(str, n_l)),
             "# groups\t" + ",".join(map(str, groups))]
    for name, s in sets:
        h = s["header"]
        lines.append(f"# set\t{name}\tmodel\t{h.get('model', '?')}\tbuild\t{h.get('build', '?')}\tids_md5\t"
                     f"{h.get('ids_md5', '?')}\ttokens\t{s['tokens']}\tchunk\t{h.get('chunk', '?')}")
    lines.append("# set\tG\twindows\tlayer\tlist\ttotals")
    for name, _ in sets:
        for g in groups:
            nw, host, card = built[name][g]
            for l in range(len(n_l)):
                lines.append(f"{name}\t{g}\t{nw}\t{l}\thost\t{','.join(map(str, host[l]))}")
                if n_l[l]:
                    lines.append(f"{name}\t{g}\t{nw}\t{l}\tcard\t{','.join(map(str, card[l]))}")
    return "\n".join(lines) + "\n"


def cmd_build(argv):
    data = os.path.join(os.environ["BLOOMERY_DATA"], "router") if os.environ.get("BLOOMERY_DATA") else None
    names, groups = list(SETS), list(GROUPS)
    it = iter(argv)
    for a in it:
        if a == "--data":
            data = next(it, None)
        elif a == "--sets":
            names = next(it, "").split(",")
        elif a == "--groups":
            groups = [int(x) for x in next(it, "").split(",")]
        else:
            raise RouteError(f"build: no argument {a!r}")
    if not data:
        raise RouteError("build: no --data and no $BLOOMERY_DATA: name the directory the router sets are in")
    sets = [(n, rc.read_manifest(os.path.join(data, n))) for n in names]
    n_l = plan_n_l(len(sets[0][1]["layers"]))
    built = {n: build_set(s, n_l, groups) for n, s in sets}
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text(built, sets, n_l, groups,
                     ["build", "--sets", ",".join(names), "--groups", ",".join(map(str, groups))]))
    print(f"{os.path.relpath(OUT)}: {len(names)} sets x groups {','.join(map(str, groups))}, "
          f"n_l {min(x for x in n_l if x)}..{max(n_l)} on {sum(1 for x in n_l if x)} layers")
    return 0


def read(path=OUT):
    """{(set, G): (windows, [host totals per layer], [card totals per layer])}, and the file's n_l."""
    if not os.path.exists(path):
        raise RouteError(f"no {path}: tools/flow/routes.py build writes it")
    n_l, out = None, {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("# n_l\t"):
                n_l = [int(x) for x in line.split("\t")[1].split(",")]
            if line.startswith("#") or not line:
                continue
            name, g, nw, l, kind, totals = line.split("\t")
            key = (name, int(g))
            if n_l is None:
                raise RouteError(f"{path}: a row before the `# n_l` line")
            if key not in out:
                out[key] = (int(nw), [[] for _ in n_l], [[] for _ in n_l])
            out[key][1 if kind == "host" else 2][int(l)] = [int(x) for x in totals.split(",") if x]
    return out, n_l


def self_test():
    """A synthetic set with known totals: the causal order, the card cut, the window edges."""
    wu = _load("window_union", os.path.join(HERE, "..", "ref", "window-union.py"))
    with tempfile.TemporaryDirectory() as root:
        # one layer, 8 experts, top-2, batch 2, chunk 4: windows of G 1 are [0,2) [2,4) [4,6) [6,8), the tail
        # 8..9 is no whole chunk. n_l 2: experts 0, 1 are the card's.
        rows = [(0, 2), (3, 4), (2, 3), (3, 1), (5, 0), (5, 6), (7, 6), (7, 6), (2, 3), (2, 3)]
        s = rc.read_manifest(wu._fake_set(root, "t", {0: rows}, 8, 2, chunk=4))
        g1 = build_set(s, [2], (1, 2), batch=2)
        nw, host, card = g1[1]
        # window [0,2): host 2, 3, 4 once each (ties to the id); card 0 (1), 1 (0)
        # [2,4): 2 1, 3 2 -> 3, 2; card 1 (1), 0 (0). [4,6): 5 2, 6 1 -> 5, 6; card 0 (1). [6,8): 7 2, 6 2 -> 6, 7
        assert nw == 4 and host == [[1 + 2 + 2 + 2, 1 + 1 + 1 + 2, 1]], (nw, host)
        assert card == [[1 + 1 + 1 + 0, 0]], card
        # G 2: windows [0,4) [4,8). The first: its first batch sees 2, 3, 4 once each (ties to the id), the
        # window 3 three times, 2 twice, 4 once -> P0 order 3, 2, 4 at 3, 2, 1. The second: first batch 5 (2),
        # 6 (1); window 6 (3), 7 (2), 5 (2) -> 5, 6, 7 at 2, 3, 2.
        nw, host, card = g1[2]
        assert nw == 2 and host == [[3 + 2, 2 + 3, 1 + 2]], (nw, host)
        assert plan_n_l(40)[:3] == [0, 0, 71] and sum(plan_n_l(40)) == 2668, plan_n_l(40)
        assert spread(5, [2, 3]) == {2: 3, 3: 2}
        try:
            build_set(rc.read_manifest(wu._fake_set(root, "nochunk", {0: rows}, 8, 2)), [2], (1,), batch=2)
            raise AssertionError("a set that names no chunk was accepted")
        except RouteError as e:
            assert "chunk" in str(e), e
    print("routes: self-test ok")
    return 0


def main(argv):
    if argv == ["--self-test"]:
        return self_test()
    try:
        if argv[:1] == ["build"]:
            return cmd_build(argv[1:])
    except (RouteError, rc.SetError) as e:
        print(f"routes.py: {e}", file=sys.stderr)
        return 1
    print(__doc__, file=sys.stderr)
    return 64


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
