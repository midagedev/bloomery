#!/usr/bin/env python3
"""A hot list whose card prefix is one card's share of a two-card split of each layer's hot union.

    tools/ref/hotlist-split.py IN OUT --pick cold|hot|every:K --tier N --plan a [--card-experts C]
    tools/ref/hotlist-split.py --self-test

IN is a router-hotlist.py file (crates/model/src/placement/hot_list.rs has the format): per layer the
routed expert ids in rank order, and a plan's card keeps each layer's first n_l. In a two-card split the
union of a layer is its first n_l + N ranks: the A6000 keeps n_l of them and the second card (the tier)
keeps N. The pick names which positions of the union (0-based, hottest first) go to the tier:

    cold      positions n_l .. n_l + N - 1, the union's cold end: OUT is IN
    hot       positions 0 .. N - 1
    every:K   positions K - 1, 2K - 1, ..., NK - 1 (K >= 2): every K-th rank, the first N of them

OUT keeps IN's header lines and adds a `# split` provenance line; each layer's line is the A6000's
positions in IN's order, then every other position in IN's order (the tier's picks, then the ranks past
the union). So a one-card plan that reads OUT keeps exactly the A6000 set of the split on its card, and
the tier's experts fall to the host: the one-card proxy of the split's A6000 shadow.

--plan a is plan (a)'s per-layer count: the expert rule of crates/model/src/placement.rs `spread` (one
more expert per eligible layer in ascending order, cycling, from zero), replayed at the card total over
the eligible layers. The total and the first eligible layer are the flow model's measured rows
`n_card_total` and `card_first_layer` (tools/flow/constants.tsv, read by column name); --card-experts
replaces the total. The eligible layers run from the first to IN's last layer; layers below the first
keep no card expert and pass through unchanged. The count depends on the run's ctx_max and the card's
dense bytes, which this tool cannot see: a run of OUT prints its `plan` record (card_experts, n_l
min..max, n_l_layers), and a reader holds it to this tool's summary line:

    hotlist-split: wrote OUT pick=<p> tier=<N> plan=a card_experts=<C> n_l=<lo>..<hi> n_l_layers=<L>

Refused by name (exit 2), never written: an IN that is not a hot list (hot_list.rs's rules), a layer
between the first eligible one and IN's last that IN does not list, a layer that lists fewer than
n_l + N ids, an every:K whose N-th pick falls past the union ((K - 1) N > n_l), and OUT naming IN.
Exit 64 on a bad command line.
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CONSTANTS = os.path.join(HERE, "..", "flow", "constants.tsv")


class SplitError(Exception):
    pass


def parse_hot(text, where):
    """(header lines as printed, n_expert, {layer: [ids]}) under hot_list.rs's rules."""
    header, layers = [], {}
    n_expert = order = None
    for no, line in enumerate(text.splitlines(), 1):
        at = f"{where}:{no}"
        if not line.strip():
            continue
        if line.startswith("#"):
            key, _, value = line[1:].lstrip().partition("\t")
            key = key.strip()
            if key == "n_expert":
                try:
                    n_expert = int(value.strip())
                except ValueError:
                    raise SplitError(f"{at}: n_expert {value!r} is not a count")
            elif key == "order":
                order = value.strip()
            header.append(line)
            continue
        if n_expert is None:
            raise SplitError(f"{at}: a layer line before `# n_expert`")
        layer, tab, ids = line.partition("\t")
        if not tab:
            raise SplitError(f"{at}: not `layer<TAB>ids`")
        try:
            layer = int(layer.strip())
            row = [int(s) for s in ids.split(",") if s.strip()]
        except ValueError as e:
            raise SplitError(f"{at}: {e}")
        if len(set(row)) != len(row):
            raise SplitError(f"{at}: layer {layer} lists an id twice")
        if any(not 0 <= e < n_expert for e in row):
            raise SplitError(f"{at}: layer {layer} lists an id outside 0..{n_expert - 1}")
        if layer in layers:
            raise SplitError(f"{at}: layer {layer} appears twice")
        layers[layer] = row
    if n_expert is None:
        raise SplitError(f"{where}: no `# n_expert` line")
    if order != "rank":
        raise SplitError(f"{where}: `# order` is {order!r}, not 'rank'")
    return header, n_expert, layers


def spread(total, eligible):
    """plan_rule's per-layer counts from zero: one more on each eligible layer in ascending order, cycling."""
    if not eligible:
        raise SplitError("no eligible layer to spread the card's experts over")
    q, r = divmod(total, len(eligible))
    return {l: q + (1 if i < r else 0) for i, l in enumerate(sorted(eligible))}


def constant(name):
    """A row's `value` from the flow model's constants table, by column name."""
    with open(CONSTANTS, encoding="utf-8") as f:
        rows = [l.rstrip("\n").split("\t") for l in f if l.strip() and not l.startswith("#")]
    head = rows[0]
    col = {c: i for i, c in enumerate(head)}
    for row in rows[1:]:
        if row[col["name"]] == name:
            return float(row[col["value"]])
    raise SplitError(f"{CONSTANTS}: no row {name!r}")


def plan_counts(plan, layers, card_experts=None):
    """Per layer of `layers`, the plan's card count; 0 below the first eligible layer."""
    if plan != "a":
        raise SplitError(f"--plan {plan!r}: only plan (a), the one-card plan, is replayed")
    first = int(constant("card_first_layer"))
    total = int(constant("n_card_total")) if card_experts is None else card_experts
    last = max(layers)
    missing = [l for l in range(first, last + 1) if l not in layers]
    if missing:
        raise SplitError(f"plan (a) keeps experts on layers {first}..{last}; the list lacks layer(s) {missing}")
    n_l = spread(total, list(range(first, last + 1)))
    return {l: n_l.get(l, 0) for l in layers}


def tier_positions(pick, n_l, tier):
    """The union positions the tier takes."""
    if pick == "cold":
        return list(range(n_l, n_l + tier))
    if pick == "hot":
        return list(range(tier))
    k = int(pick.split(":", 1)[1])
    if (k - 1) * tier > n_l:
        raise SplitError(f"every:{k} with tier {tier}: the {tier}-th pick, position {k * tier - 1}, is past "
                         f"the union of {n_l + tier}")
    return [k * j + k - 1 for j in range(tier)]


def split_layer(ids, n_l, tier, pick, layer):
    """The layer's ids with the A6000's union positions first, the rest after them, each in input order."""
    if n_l == 0:
        return list(ids)
    if len(ids) < n_l + tier:
        raise SplitError(f"layer {layer} lists {len(ids)} ids; the split's union is n_l {n_l} + tier {tier}")
    to_tier = set(tier_positions(pick, n_l, tier))
    card = [e for p, e in enumerate(ids[:n_l + tier]) if p not in to_tier]
    rest = [e for p, e in enumerate(ids) if p in to_tier or p >= n_l + tier]
    return card + rest


def check_pick(pick):
    if pick in ("cold", "hot"):
        return pick
    if pick.startswith("every:"):
        k = pick.split(":", 1)[1]
        if k.isdigit() and int(k) >= 2:
            return pick
    raise ValueError(f"--pick {pick!r} is cold, hot or every:K with K >= 2")


def split_text(text, where, pick, tier, plan, n_l_of):
    """OUT's text, the per-layer counts it was written for, and its summary."""
    header, _, layers = parse_hot(text, where)
    n_l = n_l_of(layers)
    lines = list(header)
    held = [n for n in n_l.values() if n > 0]
    summary = (f"pick={pick} tier={tier} plan={plan} card_experts={sum(held)} n_l={min(held, default=0)}..{max(held, default=0)} "
               f"n_l_layers={len(held)}")
    lines.append(f"# split\t{summary} from={where}")
    for l in sorted(layers):
        lines.append(f"{l}\t" + ",".join(str(e) for e in split_layer(layers[l], n_l[l], tier, pick, l)))
    return "\n".join(lines) + "\n", n_l, summary


def write(src, dst, pick, tier, plan, card_experts):
    if os.path.realpath(src) == os.path.realpath(dst):
        raise SplitError(f"OUT {dst} is IN: the input list is never rewritten")
    with open(src, encoding="utf-8") as f:
        text = f.read()
    out, _, summary = split_text(text, src, pick, tier, plan, lambda layers: plan_counts(plan, layers, card_experts))
    d = os.path.dirname(os.path.abspath(dst))
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".hotlist-split.")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(out)
    os.replace(tmp, dst)
    print(f"hotlist-split: wrote {dst} {summary}")


def self_test():
    fails = []

    def check(name, got, want):
        if got != want:
            fails.append(f"{name}: got {got!r}, want {want!r}")

    def refused(name, fn):
        try:
            fn()
        except SplitError:
            return
        fails.append(f"{name}: not refused")

    ids = {0: [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
           1: [3, 7, 1, 9, 0, 11, 2, 5, 4, 8, 6, 10],
           2: [5, 0, 10, 2, 8, 1, 6, 9, 3, 11, 7, 4]}
    text = "# router-hotlist\n# n\t12\n# n_expert\t12\n# order\trank\n" + "".join(
        f"{l}\t{','.join(map(str, v))}\n" for l, v in ids.items())
    # Layer 0 below the first eligible layer; 7 card experts over layers 1..2: 4 and 3.
    counts = lambda layers: {0: 0, **spread(7, [1, 2])}
    check("spread 7 over 1..2", spread(7, [1, 2]), {1: 4, 2: 3})
    check("spread 2668 over 2..39", sorted(set(spread(2668, list(range(2, 40))).items()))[:9],
          [(l, 71) for l in range(2, 10)] + [(10, 70)])
    cold, n_l, summary = split_text(text, "t", "cold", 2, "a", counts)
    check("cold is the identity", "".join(l + "\n" for l in cold.splitlines() if not l.startswith("# split\t")), text)
    check("cold layers", parse_hot(cold, "c")[2], ids)
    check("summary", summary, "pick=cold tier=2 plan=a card_experts=7 n_l=3..4 n_l_layers=2")
    two = parse_hot(split_text(text, "t", "every:2", 2, "a", counts)[0], "e")[2]
    # Layer 1 (n_l 4, union 6): the tier takes positions 1 and 3; the card keeps 0, 2, 4, 5.
    check("every:2 layer 1", two[1], [3, 1, 0, 11, 7, 9, 2, 5, 4, 8, 6, 10])
    # Layer 2 (n_l 3, union 5): the tier takes 1 and 3; the card keeps 0, 2, 4.
    check("every:2 layer 2", two[2], [5, 10, 8, 0, 2, 1, 6, 9, 3, 11, 7, 4])
    check("every:2 layer 0 unchanged", two[0], ids[0])
    hot = parse_hot(split_text(text, "t", "hot", 2, "a", counts)[0], "h")[2]
    check("hot layer 1", hot[1], [1, 9, 0, 11, 3, 7, 2, 5, 4, 8, 6, 10])
    check("hot layer 2", hot[2], [10, 2, 8, 5, 0, 1, 6, 9, 3, 11, 7, 4])
    for name, got in (("every:2", two), ("hot", hot)):
        for l in (1, 2):
            check(f"{name} layer {l} is a permutation", sorted(got[l]), sorted(ids[l]))
    check("every:3 positions", tier_positions("every:3", 4, 2), [2, 5])
    refused("every:4 past the union", lambda: tier_positions("every:4", 3, 2))
    refused("a short layer", lambda: split_text(text.replace("2\t5,0,10,2,8,1,", "2\t5,0,10,2,", 1)
                                                .replace(",6,9,3,11,7,4\n", "\n"), "t", "every:2", 2, "a", counts))
    refused("a missing layer", lambda: plan_counts("a", {0: [], 2: [], 4: []}, 4))
    a = plan_counts("a", {l: [] for l in range(40)})
    check("plan (a) from the constants table", (a[1], a[2], a[9], a[10], a[39], sum(a.values())), (0, 71, 71, 70, 70, 2668))
    refused("plan b", lambda: plan_counts("b", {2: []}, 4))
    refused("a repeated id", lambda: parse_hot("# n_expert\t4\n# order\trank\n2\t1,1\n", "r"))
    refused("no order line", lambda: parse_hot("# n_expert\t4\n2\t1\n", "r"))
    d = tempfile.mkdtemp(prefix="hotlist-split.")
    src = os.path.join(d, "in.txt")
    with open(src, "w", encoding="utf-8") as f:
        f.write(text)
    refused("OUT is IN", lambda: write(src, src, "every:2", 2, "a", 7))
    for p in os.listdir(d):
        os.remove(os.path.join(d, p))
    os.rmdir(d)
    for p in ("cold", "hot", "every:2", "every:9"):
        check_pick(p)
    for p in ("every:1", "every:x", "warm"):
        try:
            check_pick(p)
            fails.append(f"--pick {p} accepted")
        except ValueError:
            pass
    for f in fails:
        print(f"hotlist-split self-test: FAIL {f}", file=sys.stderr)
    if fails:
        return 1
    print("hotlist-split self-test: ok")
    return 0


def main(argv):
    if argv == ["--self-test"]:
        return self_test()
    usage = "usage: hotlist-split.py IN OUT --pick cold|hot|every:K --tier N --plan a [--card-experts C] | --self-test"
    pos, opts = [], {}
    it = iter(argv)
    for a in it:
        if a in ("--pick", "--tier", "--plan", "--card-experts"):
            v = next(it, None)
            if v is None or a in opts:
                print(usage, file=sys.stderr)
                return 64
            opts[a] = v
        elif a.startswith("-"):
            print(usage, file=sys.stderr)
            return 64
        else:
            pos.append(a)
    if len(pos) != 2 or not {"--pick", "--tier", "--plan"} <= opts.keys():
        print(usage, file=sys.stderr)
        return 64
    try:
        pick = check_pick(opts["--pick"])
        tier = int(opts["--tier"])
        if tier < 1:
            raise ValueError(f"--tier {tier} is not a positive count")
        ce = opts.get("--card-experts")
        ce = None if ce is None else int(ce)
        if ce is not None and ce < 1:
            raise ValueError(f"--card-experts {ce} is not a positive count")
    except ValueError as e:
        print(f"hotlist-split: {e}; {usage}", file=sys.stderr)
        return 64
    try:
        write(pos[0], pos[1], pick, tier, opts["--plan"], ce)
    except (SplitError, OSError) as e:
        print(f"hotlist-split: refused: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
