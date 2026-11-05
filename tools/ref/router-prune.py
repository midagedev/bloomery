#!/usr/bin/env python3
"""How much of the expert tail a text mix never needs, from router sets (tools/ref/router_trace.cpp).

    tools/ref/router-prune.py replay [--data DIR] [--learn prose,code] [--eval prose,code,threads,korean]
                                     [--cuts 25,40,50,60] [--per-layer]
    tools/ref/router-prune.py rank --out FILE [--data DIR] [--learn prose,code]
    tools/ref/router-prune.py --self-test

A set is a directory router_trace wrote under DIR (default $BLOOMERY_DATA/router): the reader is
tools/ref/router-coverage.py's, reached through tools/ref/router-rank.py, which also owns the rank file's
writer. This tool adds no reader and no second writer of either.

The learning slice is every context of a learn set before its held-out split except the first
(`router-coverage.py held_split` marks the split: the chunk boundary nearest T / 2): contexts 1 up to the
split. Context 0 is left out because the card lab (`lab_prune_ds41`) scores the first context of the prose
and code id files, so the mask it cuts never saw the text it is scored on. The eval positions of every
set, learn sets included, are its held-out half, from the split to the end.

replay  Per set in --eval and per layer, the share of the layer's selections (tokens x n_expert_used) that
        the coldest X percent of its experts take, for X in --cuts, the experts ranked by the learn sets'
        pooled counts in the slice above (hottest first, ties to the lower id: `router-coverage.py hot`).
        Printed: the mean over layers (every layer has the same number of selections), the least and the
        most any layer gives up, and the uniform share X percent; then, per set, how many of a layer's
        experts the set's whole trace picks fewer than 1, 10 and 100 times. The coldest cut(X) experts are
        cut(X) = floor(n_expert * X / 100 + 1/2), the count `lab_prune_ds41` cuts. A learn set's rows are
        held out (the slice never holds the eval half); a set outside --learn is out of distribution.
        --per-layer adds the share of every layer.
rank    Writes the rank file of the learn slice to FILE through router-rank.py's `rank` (a set directory per
        learn set holding the slice, in a temporary directory), the input of `lab_prune_ds41 --rank` and of
        BLOOMERY_EXPERT_RANK: the order replay ranks by, which is checked against the file written.
"""
import importlib.util
import io
import os
import sys
import tempfile
from array import array
from contextlib import redirect_stderr, redirect_stdout

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("router_rank", os.path.join(_here, "router-rank.py"))
rr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rr)
rc = rr.rc

SetError = rr.SetError


class UsageError(SetError):
    pass


DEFAULT_LEARN = ("prose", "code")
DEFAULT_EVAL = ("prose", "code", "threads", "korean")
DEFAULT_CUTS = (25, 40, 50, 60)
RARE = (1, 10, 100)


def cut_count(n_expert, pct):
    """The experts a cut of `pct` percent removes: floor(n_expert * pct / 100 + 1/2), in integers."""
    return (n_expert * pct * 2 + 100) // 200


def default_data():
    root = os.environ.get("BLOOMERY_DATA")
    if not root:
        raise SetError("replay: no --data and BLOOMERY_DATA is unset")
    return os.path.join(root, "router")


def learn_span(s):
    """The learn slice of a learn set, [first context's end, held-out split): tokens."""
    chunk = int(s["header"].get("chunk", 0) or 0)
    split = rc.held_split(s)
    if chunk <= 0 or chunk >= s["tokens"] or split <= chunk:
        raise SetError(
            f"{s['dir']}: no learn slice: it needs a `# chunk` line and a held-out split past the first "
            f"context (chunk {chunk}, tokens {s['tokens']}, split {split})")
    return chunk, split


def open_sets(names, data):
    """The sets `names` under `data`, read by the common reader, each holding its verified counts."""
    sets = []
    for name in names:
        s = rc.read_manifest(os.path.join(data, name))
        s["counts"] = rc.read_counts(s)
        sets.append(s)
    return sets


def shared_layers(sets):
    first = sets[0]["layers"]
    for s in sets:
        if s["layers"] != first:
            raise SetError(f"{s['dir']} holds layers {s['layers']}, {sets[0]['dir']} {first}")
    return list(first)


def learn_orders(learn_sets):
    """Per layer the experts of the learn slice's pooled counts, hottest first, ties to the lower id."""
    layers = shared_layers(learn_sets)
    n_expert = learn_sets[0]["n_expert"]
    for s in learn_sets:
        if (s["n_expert"], s["n_used"]) != (n_expert, learn_sets[0]["n_used"]):
            raise SetError(f"{s['dir']}: {s['n_expert']} experts, {learn_sets[0]['n_expert']} in the first learn set")
    orders = {}
    for layer in layers:
        pooled = [0] * n_expert
        for s in learn_sets:
            t0, t1 = learn_span(s)
            ids = rc.read_topk(s, layer)
            for e, c in enumerate(rc.count(ids, n_expert, t0, t1, s["n_used"])):
                pooled[e] += c
        orders[layer] = rc.hot(pooled, n_expert)
    return orders


def cold_share(order, counts, n_cut):
    """The share of `counts`' selections that the `n_cut` coldest experts of `order` take."""
    total = sum(counts)
    if total == 0:
        raise SetError("a layer with no selection in the eval positions")
    return sum(counts[e] for e in order[len(order) - n_cut:]) / total


def eval_shares(s, orders, cuts):
    """Per cut, per layer, the cold share on the set's held-out half."""
    t0, t1, K, E = rc.held_split(s), s["tokens"], s["n_used"], s["n_expert"]
    out = {x: {} for x in cuts}
    for layer, order in orders.items():
        if layer not in s["layers"]:
            raise SetError(f"{s['dir']} has no layer {layer}")
        counts = rc.count(rc.read_topk(s, layer), E, t0, t1, K)
        for x in cuts:
            out[x][layer] = cold_share(order, counts, cut_count(E, x))
    return out


def rare_counts(s, layers):
    """Per threshold, per layer, the experts the set's whole trace picks fewer than that many times."""
    return {t: {l: sum(1 for c in rr.layer_counts(s, l) if c < t) for l in layers} for t in RARE}


def write_slice(s, t0, t1, dst):
    """A set directory holding tokens t0..t1 of `s`, every layer, in router_trace's shape, so the one rank
    writer reads it as it reads a trace."""
    os.mkdir(dst)
    T, K, E = t1 - t0, s["n_used"], s["n_expert"]
    layers = s["layers"]
    with open(os.path.join(dst, "counts.tsv"), "w") as f:
        f.write("# layer\texpert\tcount\n")
        for l in layers:
            ids = rc.read_topk(s, l)[t0 * K:t1 * K]
            le = array("H", ids)
            if sys.byteorder != "little":
                le.byteswap()
            with open(os.path.join(dst, f"topk-{l}.u16"), "wb") as g:
                g.write(le.tobytes())
            for e, c in enumerate(rc.count(ids, E, 0, T, K)):
                f.write(f"{l}\t{e}\t{c}\n")
    h = s["header"]
    with open(os.path.join(dst, "MANIFEST.tsv"), "w") as f:
        f.write(f"# router_trace — tokens {t0}..{t1} of {os.path.basename(s['dir'])} (router-prune.py slice)\n")
        f.write(f"# tokens\t{T}\n# n_expert\t{E}\n# n_expert_used\t{K}\n")
        for key in ("model", "model_file", "build", "ids"):
            f.write(f"# {key}\t{h.get(key, '-')}\n")
        f.write("# layer\tlayer\tsource\tproducer\ttokens\tid_sum\tignored\tfile\n")
        for l in layers:
            f.write(f"layer\t{l}\tslice\trouter-prune\t{T}\t0\t0\ttopk-{l}.u16\n")
        f.write(f"# complete\t{T}\t{len(layers)}\n")


def write_rank(learn_sets, out):
    """The learn slice's rank file, through router-rank.py's writer; returns its header line."""
    with tempfile.TemporaryDirectory() as tmp:
        names = []
        for s in learn_sets:
            t0, t1 = learn_span(s)
            name = rr.set_name(s["dir"])
            write_slice(s, t0, t1, os.path.join(tmp, name))
            names.append(name)
        return rr.rank(names, tmp, None, out)


def replay(data, learn, evals, cuts, per_layer):
    names = list(dict.fromkeys(list(learn) + list(evals)))
    by_name = dict(zip(names, open_sets(names, data)))
    learn_sets = [by_name[n] for n in learn]
    orders = learn_orders(learn_sets)
    layers = sorted(orders)
    E, K = learn_sets[0]["n_expert"], learn_sets[0]["n_used"]
    spans = ", ".join(f"{n} {learn_span(by_name[n])[0]}..{learn_span(by_name[n])[1]}" for n in learn)
    print(f"router-prune replay [derived]: n_expert {E} n_used {K}, {len(layers)} layers {layers[0]}..{layers[-1]}")
    print(f"  learn slice (tokens): {spans}; eval = each set's held-out half")
    print("  cut(X) experts: " + ", ".join(f"X={x}% -> {cut_count(E, x)}" for x in cuts))
    rows = {}
    for n in evals:
        s = by_name[n]
        rows[n] = eval_shares(s, orders, cuts)
    print()
    head = f"{'set':<9}{'kind':<5}{'positions':>10}" + "".join(f"{f'X={x}%':>22}" for x in cuts)
    print("share of a layer's slots taken by its coldest X% of experts (mean over layers, [min..max layer])")
    print(head)
    for n in evals:
        s = by_name[n]
        kind = "in" if n in learn else "out"
        cells = ""
        for x in cuts:
            v = [rows[n][x][l] for l in layers]
            cells += f"{100 * sum(v) / len(v):>9.2f}% [{100 * min(v):5.2f}..{100 * max(v):5.2f}]".rjust(22)
        print(f"{n:<9}{kind:<5}{s['tokens'] - rc.held_split(s):>10}{cells}")
    print(f"{'uniform':<9}{'':<5}{'':>10}" + "".join(f"{x:>21.2f}%" for x in cuts))
    order = {x: sorted(evals, key=lambda n: sum(rows[n][x][l] for l in layers)) for x in cuts}
    print()
    for x in cuts:
        print(f"  predicted KLD order at X={x}% (ascending cold share): " + " < ".join(order[x]))
    print()
    uniform = by_name[evals[0]]["tokens"] * K // E
    print(f"experts a layer's whole trace picks fewer than N times (mean over layers, of {E}; uniform mean "
          f"{uniform} picks an expert)")
    print(f"{'set':<9}{'tokens':>8}" + "".join(f"{f'<{t}':>12}" for t in RARE) + f"{'layers with a <1':>20}")
    for n in evals:
        s = by_name[n]
        rare = rare_counts(s, layers)
        cells = "".join(f"{sum(rare[t].values()) / len(layers):>12.1f}" for t in RARE)
        never = sum(1 for l in layers if rare[1][l] > 0)
        print(f"{n:<9}{s['tokens']:>8}{cells}{never:>20}")
    if per_layer:
        print()
        print("per layer, cold share % at each X, per eval set")
        print(f"{'layer':>5}" + "".join(f"{f'{n} X={x}':>16}" for n in evals for x in cuts))
        for l in layers:
            print(f"{l:>5}" + "".join(f"{100 * rows[n][x][l]:>16.2f}" for n in evals for x in cuts))
    return orders


def parse_args(argv):
    if not argv or argv[0] not in ("replay", "rank"):
        raise UsageError("usage: router-prune.py replay|rank ... | --self-test")
    cmd, opts, flags = argv[0], {}, set()
    it = iter(argv[1:])
    valued = {"--data", "--learn", "--eval", "--cuts", "--out"}
    for a in it:
        if a in valued:
            if a in opts:
                raise SetError(f"{cmd}: {a} supplied more than once")
            v = next(it, None)
            if v is None:
                raise SetError(f"{cmd}: {a} takes a value")
            opts[a] = v
        elif a == "--per-layer":
            flags.add(a)
        else:
            raise SetError(f"{cmd}: unknown argument {a}")
    if cmd == "rank" and "--out" not in opts:
        raise SetError("rank: --out FILE names where the rank is written")
    if cmd == "replay" and "--out" in opts:
        raise SetError("replay: --out is rank's")
    if cmd == "rank" and ("--eval" in opts or "--cuts" in opts or flags):
        raise SetError("rank: --eval, --cuts and --per-layer are replay's")
    learn = tuple(opts["--learn"].split(",")) if "--learn" in opts else DEFAULT_LEARN
    evals = tuple(opts["--eval"].split(",")) if "--eval" in opts else DEFAULT_EVAL
    cuts = tuple(int(c) for c in opts["--cuts"].split(",")) if "--cuts" in opts else DEFAULT_CUTS
    if any(not 0 < c < 100 for c in cuts):
        raise SetError("replay: a cut is a percent between 0 and 100, exclusive")
    for what, names in (("--learn", learn), ("--eval", evals)):
        if not names or "" in names or len(set(names)) != len(names):
            raise SetError(f"{cmd}: {what} names no set, an empty one or a set twice")
    return cmd, opts.get("--data"), learn, evals, cuts, "--per-layer" in flags, opts.get("--out")


def run(argv):
    cmd, data, learn, evals, cuts, per_layer, out = parse_args(argv)
    data = data or default_data()
    if cmd == "replay":
        replay(data, learn, evals, cuts, per_layer)
        return 0
    learn_sets = open_sets(learn, data)
    head = write_rank(learn_sets, out)
    _, written = rr.read_rank(out)
    orders = learn_orders(learn_sets)
    if written != orders:
        raise SetError(f"rank: {out} is not the order replay ranks by")
    print(f"router-prune: wrote {out}")
    print(head)
    return 0


def self_test():
    """Synthetic sets with known picks: the cut counts, the slice, the shares, the rare counts, the rank file
    through the common writer, and each refusal."""
    assert [cut_count(384, x) for x in (25, 40, 50, 60)] == [96, 154, 192, 230]
    assert [cut_count(8, x) for x in (25, 50, 60)] == [2, 4, 5]
    E, K, C = 8, 2, 4

    def make(tmp, name, picks, complete=True, chunk=C):
        d = os.path.join(tmp, name)
        os.mkdir(d)
        T = len(next(iter(picks.values())))
        with open(os.path.join(d, "counts.tsv"), "w") as f:
            f.write("# layer\texpert\tcount\n")
            for l, rows in picks.items():
                c = [0] * E
                for r in rows:
                    for e in r:
                        c[e] += 1
                for e in range(E):
                    f.write(f"{l}\t{e}\t{c[e]}\n")
                with open(os.path.join(d, f"topk-{l}.u16"), "wb") as g:
                    g.write(array("H", [e for r in rows for e in r]).tobytes())
        with open(os.path.join(d, "MANIFEST.tsv"), "w") as f:
            f.write(f"# router_trace — self-test\n# tokens\t{T}\n# n_expert\t{E}\n# n_expert_used\t{K}\n")
            f.write(f"# model\t/m/m-00001-of-00002.gguf\n# model_file\tm.gguf\n# build\tb\n# ids\t{d}/ids\n# chunk\t{chunk}\n")
            f.write("# layer\tlayer\tsource\tproducer\ttokens\tid_sum\tignored\tfile\n")
            for l in picks:
                f.write(f"layer\t{l}\tVIEW-of-ARGSORT\tp\t{T}\t0\t0\ttopk-{l}.u16\n")
            if complete:
                f.write(f"# complete\t{T}\t{len(picks)}\n")
        return d

    # T = 16, chunk 4: the held-out split is 8, the learn slice tokens 4..8, the eval half 8..16.
    # Layer 2's learn rows all pick 5 and 1, and its context 0 picks 7 and 6: the order is 1, 5 (a tie, the
    # lower id first), then 0, 2, 3, 4, 6, 7 by id. Its eval rows pick (6, 7) twice, (0, 5) four times and
    # (1, 5) twice.
    def rows(learn_row, eval_rows):
        return [[7, 6]] * 4 + [learn_row] * 4 + eval_rows
    ev2 = [[6, 7]] * 2 + [[0, 5]] * 4 + [[1, 5]] * 2
    ev3 = [[0, 1]] * 8
    with tempfile.TemporaryDirectory() as tmp:
        a = make(tmp, "a", {2: rows([5, 1], ev2), 3: rows([0, 1], ev3)})
        b = make(tmp, "b", {2: rows([5, 3], ev2), 3: rows([0, 1], ev3)})
        thin = make(tmp, "thin", {2: [[0, 1]] * 4}, chunk=4)
        # the learn slice excludes context 0, whose picks (7 and 6) would otherwise rank first
        sa = open_sets(["a"], tmp)
        assert learn_span(sa[0]) == (4, 8)
        order = learn_orders(sa)
        assert order[2] == [1, 5, 0, 2, 3, 4, 6, 7], order[2]  # 1 and 5 tie at 4: the lower id first
        assert order[3] == [0, 1, 2, 3, 4, 5, 6, 7], order[3]
        # two learn sets pool: a gives 5:4 1:4, b gives 5:4 3:4, so 5 leads, then 1, 3 (4 each, lower id first)
        pooled = learn_orders(open_sets(["a", "b"], tmp))
        assert pooled[2][:3] == [5, 1, 3], pooled[2]
        # shares: layer 2's cold X=25% is cut(8, 25) = 2 experts: the last two of its order, 6 and 7
        sh = eval_shares(sa[0], order, (25, 50))
        assert abs(sh[25][2] - 4 / 16) < 1e-12, sh[25]  # (6, 7) twice: 4 of 16 selections
        # X = 50%: the last four, 3 4 6 7: still only those 4 selections
        assert abs(sh[50][2] - 4 / 16) < 1e-12, sh[50]
        assert sh[25][3] == 0.0, sh[25]  # layer 3's eval picks only 0 and 1, both hot
        # rare counts over the whole trace: layer 2 never picks 2, 3 or 4; layer 3 never picks 2, 3, 4 or 5 and
        # picks 6 and 7 four times each, so six of its experts are picked fewer than ten times
        r = rare_counts(sa[0], [2, 3])
        assert r[1][2] == 3 and r[1][3] == 4, r[1]
        assert r[10][3] == 6, r[10]
        # the rank file: the common writer's, equal to the order replay ranks by
        out = os.path.join(tmp, "rank.tsv")
        head = write_rank(open_sets(["a", "b"], tmp), out)
        assert head == "# router-rank\tn_expert=8\tn_used=2\tsets=a,b\tpositions=8", head
        meta, layers = rr.read_rank(out)
        assert layers == pooled, (layers, pooled)
        assert [f for f in os.listdir(tmp) if ".tmp." in f] == [], "the rank writer left a temp file"
        # replay runs end to end
        with redirect_stdout(io.StringIO()) as shown:
            replay(tmp, ("a",), ("a", "b"), (25, 50), True)
        text = shown.getvalue()
        assert "predicted KLD order at X=25%" in text and "uniform" in text, text
        # refusals, each by name
        for args, why in (
            (["replay", "--data", tmp, "--learn", "thin", "--eval", "thin"], "no learn slice"),
            (["replay", "--data", tmp, "--learn", "nosuch", "--eval", "nosuch"], "no manifest"),
        ):
            with redirect_stderr(io.StringIO()):
                try:
                    run(args)
                except SetError as e:
                    assert why in str(e), (why, str(e))
                    continue
            raise AssertionError(f"expected a refusal naming {why!r}")
        cut = make(tmp, "cut", {2: rows([5, 1], ev2)}, complete=False)
        try:
            open_sets(["cut"], tmp)
        except SetError as e:
            assert "complete" in str(e), str(e)
        else:
            raise AssertionError("an unfinished set was read")
        assert cut and thin
    for bad in (["replay", "--cuts", "0"], ["replay", "--cuts", "100"], ["rank"], ["replay", "--out", "x"],
                ["rank", "--out", "x", "--per-layer"], ["replay", "--learn", "a,a"], ["replay", "--eval", ""],
                ["replay", "--data"], ["replay", "--data", "a", "--data", "b"], ["x"], []):
        try:
            parse_args(bad)
        except SetError:
            continue
        raise AssertionError(f"{bad} was taken")
    print("router-prune --self-test: ok")
    return 0


def main(argv):
    if argv[:1] == ["--self-test"]:
        return self_test()
    try:
        return run(argv)
    except UsageError as e:
        print(f"router-prune: {e}", file=sys.stderr)
        return 64
    except SetError as e:
        print(f"router-prune: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
