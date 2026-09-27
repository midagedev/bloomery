#!/usr/bin/env python3
"""The v41-pplb card's term, from generate_ds41's per-layer-batch records.

  pplb.py LOG...      every arm of each log (a generate_ds41 output, or a depth runner's log that echoes its
                      arms' `fed` and `stat prefill` lines), then the mean per prompt length over every arm
  pplb.py --self-test

An arm opens at its `fed` record and holds its `stat prefill split` and, under BLOOMERY_STEP_STATS=1, a
`stat prefill lb` record per layer-batch with that layer-batch's card and serve times (`card_out_ms`,
`card_in_ms` where it ran a block, `union_ms`, `wait_ms`). Every value is read through
tools/bloomery/records.py by kind and field. Every arm of the log counts, a runner's warm-up or discard arm
too (BLOOMERY_AB_WARMUP, the blocks order): the card's sitting runs the rotate order, which has none.

The term, per batch b: over its layer-0 and layer-1 layer-batches that ran a block,
max(0, union_ms - card_in_ms of that layer-batch - card_out_ms of the route enqueued ahead of its serve) —
the host serve's time the card has no queued work for. The route ahead of a serve: a group runs its
layer-batches layer by layer, each over the group's batches (the engine's Tally order, the order the lb
records print in), and in a group of two or more batches the next layer-batch's route is enqueued after this
one's shadow and before its serve; the group's last layer-batch and every layer-batch of a group of one have
none. A group's size is the count of layer-0 records it opens with (the last group of a call can be short).
Each batch's row names the route it paired with each serve.

Beside it, layer 2 minus layer 3 card_out_ms at batch 30 against batch 2 (the indexer layers' route against
the batch index).

A log is refused by name (exit 2) when an arm has no per-layer-batch times, its records break the group
order, or their sums miss the split line's by more than the engine's check allows plus the two lines'
printed rounding. Exit 64: usage.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bloomery"))
import records  # noqa: E402 - the engine's record lines, read by kind and field

# The engine's own check (body::PromptCounts::check_split, SPLIT_SUM_MS), and half a printed digit per
# record: lb times print two decimals, the split's one.
ENGINE_MS = 0.1
LB_ROUND_MS = 0.005
SPLIT_ROUND_MS = 0.05
SERVED_LAYERS = (0, 1)
DIFF_LAYERS = (2, 3)
DIFF_BATCHES = (2, 30)


class Refused(Exception):
    pass


def arms(recs):
    """The records of each arm: from a `fed` record to the next."""
    out = []
    for r in recs:
        if r.kind == "fed":
            out.append([r])
        elif out:
            out[-1].append(r)
    return out


def groups(lbs):
    """The lb records cut into their groups, each [(layer, [record per batch])], checked against the group
    order; refused by name where they break it."""
    if not lbs:
        return []
    n_layers = max(r["layer"] for r in lbs) + 1
    out, k = [], 0
    while k < len(lbs):
        g = 0
        while k + g < len(lbs) and lbs[k + g]["layer"] == 0:
            g += 1
        b0 = lbs[k]["b"]
        if g == 0 or k + g * n_layers > len(lbs):
            raise Refused(f"record {k} (b={lbs[k]['b']} layer={lbs[k]['layer']}) opens no group of {n_layers} "
                          f"layers, each over the group's batches")
        layers = []
        for i in range(n_layers):
            row = lbs[k + i * g:k + (i + 1) * g]
            want = [(b0 + j, i) for j in range(g)]
            got = [(r["b"], r["layer"]) for r in row]
            if got != want:
                raise Refused(f"the group from record {k} reads (b, layer) {got} where its order puts {want}")
            layers.append((i, row))
        out.append(layers)
        k += g * n_layers
    return out


def need(r, name):
    if name not in r:
        raise Refused(f"`stat prefill lb` b={r['b']} layer={r['layer']} has no {name}: a binary before the "
                      f"per-layer-batch times, or a run without card timing (BLOOMERY_STEP_STATS=1)")
    return r[name]


def check_sums(split, lbs):
    """The lb times summed against the split line's, within the engine's check and the printed rounding."""
    if split is None:
        raise Refused("no `stat prefill split` record in the arm")
    if "card_out_ms" not in split:
        raise Refused("the split line was not timed on the card (card=untimed): run under BLOOMERY_STEP_STATS=1")
    tol = ENGINE_MS + SPLIT_ROUND_MS + LB_ROUND_MS * len(lbs)
    for name in ("card_out_ms", "card_in_ms", "union_ms", "wait_ms"):
        total = sum(r[name] if name in r else (0.0 if name == "card_in_ms" and not r["block"] else need(r, name))
                    for r in lbs)
        if abs(total - split[name]) > tol:
            raise Refused(f"the lb records' {name} sums to {total:.2f}, the split line's is {split[name]:.1f}: "
                          f"more than {tol:.2f} ms apart")


def arm_terms(recs):
    """One arm: (P, group lever, [(batch, term, pairing text)], {batch: layer-2 minus layer-3 card_out})."""
    fed = recs[0]
    split = records.first(recs, "stat_prefill_split")
    lbs = records.of_kind(recs, "stat_prefill_lb")
    if not lbs:
        raise Refused("no `stat prefill lb` record: a batched run under BLOOMERY_STEP_STATS=1 prints them")
    for r in lbs:
        need(r, "card_out_ms")
    check_sums(split, lbs)
    per, diff = {}, {}
    for layers in groups(lbs):
        g = len(layers[0][1])
        order = [r for _, row in layers for r in row]
        for x, r in enumerate(order):
            if r["layer"] in DIFF_LAYERS:
                sign = 1 if r["layer"] == DIFF_LAYERS[0] else -1
                diff[r["b"]] = diff.get(r["b"], 0.0) + sign * r["card_out_ms"]
            if r["layer"] not in SERVED_LAYERS or not r["block"]:
                continue
            ahead = order[x + 1] if g >= 2 and x + 1 < len(order) else None
            out = ahead["card_out_ms"] if ahead is not None else 0.0
            shadow = need(r, "card_in_ms")
            idle = max(0.0, need(r, "union_ms") - shadow - out)
            who = f"b{ahead['b']} L{ahead['layer']}" if ahead is not None else "none"
            text = (f"L{r['layer']}: union {r['union_ms']:.2f} - in {shadow:.2f} - out({who}) {out:.2f} "
                    f"= {idle:.2f}")
            t, parts = per.get(r["b"], (0.0, []))
            per[r["b"]] = (t + idle, parts + [text])
    rows = [(b, t, " | ".join(parts)) for b, (t, parts) in sorted(per.items())]
    return fed["ids"], split["group"], rows, diff


def report(paths):
    by_p = {}
    for path in paths:
        for k, recs in enumerate(arms(records.read(path))):
            try:
                P, group, rows, diff = arm_terms(recs)
            except Refused as e:
                raise Refused(f"{path}: arm {k}: {e}") from e
            print(f"{path}: arm {k}: P {P}, group lever {group}, {len(rows)} batches with a layer-0/1 serve")
            for b, t, text in rows:
                print(f"  b {b:3d}  term {t:7.2f} ms  {text}")
            mean = sum(t for _, t, _ in rows) / len(rows) if rows else float("nan")
            print(f"  arm mean {mean:.2f} ms over {len(rows)} batches")
            d = [diff.get(b) for b in DIFF_BATCHES]
            if None in d:
                have = ", ".join(str(b) for b in DIFF_BATCHES if b in diff)
                print(f"  layer {DIFF_LAYERS[0]} - layer {DIFF_LAYERS[1]} card_out: batches {DIFF_BATCHES} not both "
                      f"in the arm (has {have or 'neither'})")
            else:
                print(f"  layer {DIFF_LAYERS[0]} - layer {DIFF_LAYERS[1]} card_out: batch {DIFF_BATCHES[0]} {d[0]:.2f}, "
                      f"batch {DIFF_BATCHES[1]} {d[1]:.2f} ms; {DIFF_BATCHES[1]} minus {DIFF_BATCHES[0]} "
                      f"{d[1] - d[0]:+.2f} ms")
            by_p.setdefault(P, []).extend(t for _, t, _ in rows)
    if not by_p:
        raise Refused(f"no arm (no `fed` record) in {', '.join(paths)}")
    for P, ts in sorted(by_p.items()):
        print(f"P {P}: term mean {sum(ts) / len(ts):.2f} ms over {len(ts)} batches of every arm")


def self_test():
    """A call of three batches over four layers under the group lever 2 (a group of two, then one of one),
    its times chosen so every pairing gives its own value; then the refusals."""
    import tempfile

    def lb(b, layer, block, out, shadow, union, wait):
        s = f"stat prefill lb b={b} layer={layer} block={'true' if block else 'false'} entries_route=9 " \
            f"entries_shadow={3 if block else 0} card_out_ms={out:.2f}"
        if block:
            s += f" card_in_ms={shadow:.2f}"
        return s + f" union_ms={union:.2f} wait_ms={wait:.2f}"

    # (b, layer): card_out, card_in, union. Layers 0 and 1 serve long unions; layer 3 of batch 2 has no block.
    t = {}
    for b in range(3):
        for layer in range(4):
            t[(b, layer)] = (10.0 + b + layer * 0.5, 4.0, 60.0 if layer < 2 else 5.0)
    order = [(b, layer) for layer in range(4) for b in (0, 1)] + [(2, layer) for layer in range(4)]

    def log(lines_of=lambda k: True, split_shift=0.0, extra=()):
        lines = ["fed ids=1536 first=[1, 2, 3, 4] last=[5, 6, 7, 8] depth_sequence_from=0"]
        tot = {"card_out_ms": 0.0, "card_in_ms": 0.0, "union_ms": 0.0, "wait_ms": 0.0}
        for k in order:
            out, shadow, union = t[k]
            block = k != (2, 3)
            tot["card_out_ms"] += out
            tot["card_in_ms"] += shadow if block else 0.0
            tot["union_ms"] += union if block else 0.0
            tot["wait_ms"] += 0.25 if block else 0.0
            if lines_of(k):
                lines.append(lb(k[0], k[1], block, out, shadow, union if block else 0.0, 0.25 if block else 0.0))
        lines.insert(1, f"stat prefill split group=2 batches=3 layer_batches=11 prologue_ms=1.0 chain_ms=900.0 "
                        f"union_ms={tot['union_ms'] + split_shift:.1f} wait_ms={tot['wait_ms']:.1f} enqueue_ms=1.0 "
                        f"copy_ms=1.0 union_lb=1.00 wait_lb=1.00 wait_first_lb=1.00 enqueue_lb=1.00 copy_lb=1.00 "
                        f"entries_route=9.0 entries_shadow=3.0 excluded_lb=0.0 card_out_ms={tot['card_out_ms']:.1f} "
                        f"card_in_ms={tot['card_in_ms']:.1f} card_proj_ms=1.0 card_out_lb=1.00 card_in_lb=1.00 "
                        f"card_proj_lb=1.00")
        lines.extend(extra)
        f = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False)
        f.write("\n".join(lines) + "\n")
        f.close()
        return f.name

    fails = []

    def check(name, ok, detail=""):
        print(f"{'ok' if ok else 'FAIL'} {name}{': ' + detail if detail and not ok else ''}")
        if not ok:
            fails.append(name)

    path = log()
    recs = records.read(path)
    check("every line is read whole", all(not r.loose for r in recs))
    P, group, rows, diff = arm_terms(arms(recs)[0])
    got = {b: round(tm, 6) for b, tm, _ in rows}
    # Group {0, 1}: (0,L0) -> route (1,L0) 11.0; (1,L0) -> (0,L1) 10.5; (0,L1) -> (1,L1) 11.5;
    # (1,L1) -> (0,L2) 11.0. Group {2}: no route ahead.
    want = {0: (60 - 4 - 11.0) + (60 - 4 - 11.5), 1: (60 - 4 - 10.5) + (60 - 4 - 11.0), 2: (60 - 4) * 2}
    check("the term pairs each serve with the route ahead of it", got == want, f"{got} != {want}")
    check("the pairing is named", "out(b1 L0)" in rows[0][2] and "out(none)" in rows[2][2], rows[0][2])
    check("P and the lever", (P, group) == (1536, 2), f"{(P, group)}")
    check("layer 2 minus layer 3", round(diff[1], 6) == -0.5, f"{diff}")
    os.unlink(path)

    def refused(name, path, pat):
        try:
            arm_terms(arms(records.read(path))[0])
        except Refused as e:
            check(name, pat in str(e), str(e))
        else:
            check(name, False, "not refused")
        os.unlink(path)

    refused("a dropped card_out breaks the sum", log(lines_of=lambda k: k != (1, 2)), "sums to")
    refused("a shifted split breaks the sum", log(split_shift=5.0), "union_ms sums to")
    order[0], order[2] = order[2], order[0]
    refused("a group that opens past layer 0", log(), "opens no group")
    order[0], order[2] = order[2], order[0]
    order[2], order[3] = order[3], order[2]
    refused("an lb out of the group order", log(), "where its order puts")
    order[2], order[3] = order[3], order[2]
    untimed = log()
    with open(untimed) as f:
        text = f.read()
    with open(untimed, "w") as f:
        f.write("\n".join(ln.split(" card_out_ms=")[0] if ln.startswith("stat prefill lb") else ln
                          for ln in text.splitlines()) + "\n")
    refused("records without times", untimed, "has no card_out_ms")
    print(f"self-test: {'ok' if not fails else 'FAIL'} ({len(fails)} failures)")
    return 0 if not fails else 1


def main(argv):
    if argv == ["--self-test"]:
        return self_test()
    if not argv or any(a.startswith("-") for a in argv):
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 64
    try:
        report(argv)
    except Refused as e:
        print(f"pplb.py: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
