#!/usr/bin/env python3
"""Qwen3.8's MTP window at a width chosen per round from the draft's confidence, replayed from a log.

    tools/flow/q38width.py LOG [LOG...] [--step MS] [--x MS] [--walk MS] [--head-rows LIST]
    tools/flow/q38width.py --self-test

LOG is generate_qwen3moe's output under BLOOMERY_MTP_WINDOWS=1: one `mtp window` record a drafted pass
(its pass number, the target's position, the proposal's ids, each id's probability p_j among the draft
head's rows, how many ids the target kept), read through tools/bloomery/records.py by kind and field. A
log of several arms restarts the pass numbers at each arm; each arm is a block of its own and the pooled
row sums them.

The window's cost is `step + x * (rows - 1)`: one target step, and x for each verify row past the first,
its draft walk included (--step 17.49 and --x 7.29 ms by default, the q38seed sitting's step and its
per-row term, docs/cards/q38head-ab.card's sources). --walk W takes W of each x as the walk, which the
chain runs whether its row is verified or not (the chain reads back once, after its last walk): a
truncated row then saves x - W, not x. Default 0, the brief's model.

Rows printed, for each block:
  fixed d    every window truncated to its first d drafts (d = 1, 2, 3): positions a window (1 + the kept
             drafts, at most d), rows a window, ms a window and tok/s, and the ratio to the width the log ran
             at (every draft verified, d = 3).
  tau t      the rule "stop the chain before the first draft with p_j < t", t = 0.50, 0.55 ... 0.95:
             a window keeps its leading drafts while p_j >= t; positions 1 + min(kept, leading), rows
             1 + leading. The same columns.
  head N     with --head-rows LIST (tools/ref/draft-vocab.py's list): the log's windows with a kept draft
             whose id the list does not hold, and every kept draft after it, rejected — what a head of
             those rows would keep, at most. It counts only the loss: a draft the full head missed and the
             list's argmax would have hit is not in the log.
  accept     the share of drafts the target kept, by p_j decile, all drafts pooled: how well p_j ranks.

Each window is replayed on its own: a shorter window moves every later window's start, and with it the
p_j and the kept counts the log holds, so the tau and fixed rows are [derived] under that independence.

Refused, each by name (exit 2): a log with no `mtp window` record, a window of no ids or of more than
three, a p list of another length, a p_j outside (0, 1], a kept count past the ids, a list without its
`# bloomery mtp-head-rows 1` line. Exit 64: usage.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bloomery"))
import records  # noqa: E402 - the engine's record lines, read by kind and field

BIN = "generate_qwen3moe"
KIND = "mtp_window"
STEP_MS = 17.49
X_MS = 7.29
WIDTH = 3
TAUS = [round(0.50 + 0.05 * k, 2) for k in range(10)]
ROWS_FORMAT = "# bloomery mtp-head-rows 1"


class Refused(Exception):
    pass


def windows(path):
    """Each arm's windows [(p list, kept count, ids)], an arm opening where the pass numbers restart."""
    arms, last = [], None
    for r in records.of_kind(records.read(path, BIN), KIND):
        ids, accepted, n = r["ids"], r["accepted"], r["window"]
        try:
            p = [float(x) for x in r["p"]]
        except ValueError as e:
            raise Refused(f"{path}: window {n}: a p that is not a number ({r['p']})") from e
        if not 1 <= len(ids) <= WIDTH:
            raise Refused(f"{path}: window {n} holds {len(ids)} ids; a window holds 1 to {WIDTH}")
        if len(p) != len(ids):
            raise Refused(f"{path}: window {n}: {len(p)} probabilities for {len(ids)} ids")
        if not all(0.0 < x <= 1.0 for x in p):
            raise Refused(f"{path}: window {n}: a p_j outside (0, 1]: {p}")
        if accepted > len(ids):
            raise Refused(f"{path}: window {n} kept {accepted} of {len(ids)} ids")
        if last is None or n <= last:
            arms.append([])
        arms[-1].append((p, accepted, ids))
        last = n
    if not arms:
        raise Refused(f"{path}: no `mtp window` record (run generate_qwen3moe with BLOOMERY_MTP_WINDOWS=1)")
    return arms


def head_rows(path):
    try:
        lines = open(path, encoding="utf-8").read().splitlines()
    except OSError as e:
        raise Refused(f"{path}: {e.strerror}") from e
    if not lines or not lines[0].startswith(ROWS_FORMAT):
        raise Refused(f"{path}: no `{ROWS_FORMAT}` line first")
    return {int(x) for x in lines[1:]}


def replay(ws, cut, step, x, walk):
    """(positions, rows, ms) a window, the window cut to `cut(p, accepted, ids)` = (leading drafts, kept)."""
    pos = rows = ms = 0.0
    for p, accepted, ids in ws:
        lead, kept = cut(p, accepted, ids)
        pos += 1 + kept
        rows += 1 + lead
        ms += step + x * lead + walk * (len(p) - lead)
    n = len(ws)
    return pos / n, rows / n, ms / n


def fixed(d):
    return lambda p, a, ids: (min(d, len(p)), min(a, d))


def tau(t):
    def cut(p, a, ids):
        lead = next((j for j, v in enumerate(p) if v < t), len(p))
        return lead, min(a, lead)
    return cut


def listed(rows):
    def cut(p, a, ids):
        kept = next((j for j in range(a) if ids[j] not in rows), a)
        return len(p), kept
    return cut


def block(name, ws, step, x, walk, rows):
    out = [f"## {name}: {len(ws)} windows, step {step} ms, x {x} ms, walk {walk} ms"]
    base = replay(ws, fixed(WIDTH), step, x, walk)
    base_rate = base[0] / base[2]

    def row(label, cut):
        pos, r, ms = replay(ws, cut, step, x, walk)
        rate = pos / ms
        out.append(f"{label:10s} positions={pos:.3f} rows={r:.3f} ms={ms:.2f} tok/s={1e3 * rate:.2f} "
                   f"ratio={rate / base_rate:.4f}")
        return rate / base_rate

    for d in range(1, WIDTH + 1):
        row(f"fixed {d}", fixed(d))
    best = max(((row(f"tau {t:.2f}", tau(t)), t) for t in TAUS))
    out.append(f"best tau {best[1]:.2f}: {100 * (best[0] - 1):+.2f} % over fixed {WIDTH} [derived]")
    if rows is not None:
        row(f"head {len(rows)}", listed(rows))
    return out


def deciles(ws):
    buckets = [[0, 0] for _ in range(10)]
    for p, accepted, _ in ws:
        for j, v in enumerate(p):
            b = buckets[min(9, int(v * 10))]
            b[0] += 1
            b[1] += j < accepted
    return [f"accept p [{k / 10:.1f}, {(k + 1) / 10:.1f}): drafts={n} kept={kept / n:.3f}"
            for k, (n, kept) in enumerate(buckets) if n]


def run(paths, step, x, walk, rows_path):
    rows = head_rows(rows_path) if rows_path else None
    allw, out = [], []
    for path in paths:
        for i, ws in enumerate(windows(path)):
            out += block(f"{os.path.basename(path)} arm {i}", ws, step, x, walk, rows)
            allw += ws
    out += block("pooled", allw, step, x, walk, rows)
    out += deciles(allw)
    return out


def main(argv):
    if argv == ["--self-test"]:
        return self_test()
    paths, step, x, walk, rows = [], STEP_MS, X_MS, 0.0, None
    try:
        i = 0
        while i < len(argv):
            a = argv[i]
            if a in ("--step", "--x", "--walk", "--head-rows"):
                if i + 1 >= len(argv):
                    raise ValueError(f"{a} takes a value")
                v = argv[i + 1]
                if a == "--head-rows":
                    rows = v
                else:
                    f = float(v)
                    if not f >= 0:
                        raise ValueError(f"{a} {v}: a time in ms, 0 or more")
                    step, x, walk = (f, x, walk) if a == "--step" else (step, f, walk) if a == "--x" else (step, x, f)
                i += 2
            elif a.startswith("--"):
                raise ValueError(f"unknown flag {a}")
            else:
                paths.append(a)
                i += 1
        if not paths:
            raise ValueError("no log")
        if walk > x:
            raise ValueError(f"--walk {walk} is more than --x {x}: the walk is part of a row's cost")
    except ValueError as e:
        print(f"q38width: {e}\n{__doc__}", file=sys.stderr)
        return 64
    try:
        print("\n".join(run(paths, step, x, walk, rows)))
    except Refused as e:
        print(f"q38width: {e}", file=sys.stderr)
        return 2
    return 0


def self_test():
    import tempfile
    ok = True

    def expect(what, cond):
        nonlocal ok
        print(f"{'ok  ' if cond else 'FAIL'} {what}")
        ok = ok and cond

    def line(n, pos, ids, p, a):
        return (f"mtp window window={n} pos={pos} ids=[{','.join(map(str, ids))}] "
                f"p=[{','.join(f'{v:.5f}' for v in p)}] accepted={a}")

    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "a.log")
        # Arm 0: two windows. W1 p .9 .8 .4, kept 3; W2 p .9 .3 .9, kept 1. Arm 1 (pass numbers restart): W1
        # p .6, kept 0 (a one-id proposal near the context's end).
        with open(log, "w") as f:
            f.write("\n".join([
                "step 0 511 13 (a line of no kind)",
                line(1, 512, [5, 6, 7], [0.9, 0.8, 0.4], 3),
                "mtp summary proposals=2 kept=[0, 1, 0, 1] positions=6 passes=2 tok/s(positions)=1.00",
                line(3, 516, [8, 9, 10], [0.9, 0.3, 0.9], 1),
                line(1, 4096, [11], [0.6], 0),
            ]) + "\n")
        arms = windows(log)
        expect("two arms, by the pass numbers' restart", [len(a) for a in arms] == [2, 1])
        expect("fields", arms[0][0] == ([0.9, 0.8, 0.4], 3, [5, 6, 7]))
        ws = arms[0]
        # fixed 3: positions (4 + 2)/2 = 3, rows 4, ms 10 + 3 = 13 at step 10, x 1.
        expect("fixed 3", replay(ws, fixed(3), 10, 1, 0) == (3.0, 4.0, 13.0))
        # fixed 1: positions (2 + 2)/2 = 2, rows 2, ms 11.
        expect("fixed 1", replay(ws, fixed(1), 10, 1, 0) == (2.0, 2.0, 11.0))
        # tau .5: W1 leads 2 (p .4 < .5), kept min(3, 2) = 2 -> 3 positions, 3 rows; W2 leads 1, kept 1 -> 2, 2.
        expect("tau 0.50", replay(ws, tau(0.5), 10, 1, 0) == (2.5, 2.5, 11.5))
        # --walk .5: W1 pays .5 for its cut walk, W2 1.0 for its two.
        expect("tau 0.50 with the walk paid", replay(ws, tau(0.5), 10, 1, 0.5) == (2.5, 2.5, 12.25))
        # head rows {5, 6, 8}: W1's kept 5, 6 then 7 not held -> kept 2; W2 kept 8 -> 1.
        expect("head rows", replay(ws, listed({5, 6, 8}), 10, 1, 0) == (2.5, 4.0, 13.0))
        rows = os.path.join(d, "rows.txt")
        with open(rows, "w") as f:
            f.write(f"{ROWS_FORMAT} vocab=12 rows=3 vocab_sha256=00\n5\n6\n8\n")
        expect("head rows read", head_rows(rows) == {5, 6, 8})
        out = run([log], 10, 1, 0, rows)
        expect("the pooled block's best tau", any(o.startswith("best tau") for o in out)
               and out[-1].startswith("accept p [0.9, 1.0): drafts=3 kept=0.667"))
        expect("main exits 0", main([log, "--step", "10", "--x", "1", "--head-rows", rows]) == 0)
        expect("main: --walk past --x is a usage error", main([log, "--x", "1", "--walk", "2"]) == 64)

        def refused(what, text):
            bad = os.path.join(d, "bad.log")
            with open(bad, "w") as f:
                f.write(text + "\n")
            try:
                windows(bad)
            except Refused as e:
                expect(f"{what}: {e}", True)
                return
            expect(f"{what}: not refused", False)

        refused("no window", "step 1 2 3")
        refused("a p list of another length", line(1, 9, [1, 2], [0.5], 0))
        refused("a p_j of 0", line(1, 9, [1], [0.0], 0))
        refused("a p_j past 1", line(1, 9, [1], [1.5], 0))
        refused("kept past the ids", line(1, 9, [1], [0.5], 2))
        refused("four ids", line(1, 9, [1, 2, 3, 4], [0.5] * 4, 0))
        with open(rows, "w") as f:
            f.write("5\n6\n")
        try:
            head_rows(rows)
            expect("a list without its format line: not refused", False)
        except Refused as e:
            expect(f"a list without its format line: {e}", True)
    print("q38width self-test:", "ok" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
