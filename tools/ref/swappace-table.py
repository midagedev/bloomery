#!/usr/bin/env python3
"""The reading side of tools/ref/swappace-diag.sh: per run and per arm, the residency cycle's table.
Reads generate_ds41 logs only, through tools/bloomery/records.py by kind and field; starts nothing.

    swappace-table.py [--steps A:B] [LABEL=]LOG...
    swappace-table.py --self-test

A log's records are split into arms at its `arm` records (a log with none is one arm, 0). Per arm:

  steps     the `time step` and `stat step` records with A <= i < B (default 20:96), warm ones left out,
            grouped by i mod 4: the median step ms, the median leg_us (as ms), the median gap_p50_us and
            the largest gap_max_us; then the steps' mean ms and the cycle excess, the sum of the four
            mod medians less four times the least of them (the residency cycle is 4 steps: the least mod
            is the step that stages nothing).
  passes    the `residency pass` records of a decode step (pass=step) with A <= boundary < B, grouped by
            boundary mod 4 — a boundary's report covers the pass before it; boundaries and step indices
            are not matched here: the median stage_us, stage_in_us, window_wait_us, prepare_us, the sum
            of stage_out_bytes, and the sums of late and forced. A field the log's binary did not print
            (an older binary, read loose) is `-`.

Exit status: 0; 1 a log with no step record in the window; 64 a usage error.
"""
import os
import statistics
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bloomery"))
import records  # noqa: E402 - the record reader, beside this tree's tools

MODS = 4
STEPS = (20, 96)
PASS_MEDIANS = ("stage_us", "stage_in_us", "window_wait_us", "prepare_us")
PASS_SUMS = ("stage_out_bytes", "late", "forced")


def arms(recs):
    """The records per arm, in order: [(arm index, records)]."""
    out, cur, i = [], [], 0
    for r in recs:
        if r.kind == "arm":
            if cur or out:
                out.append((i, cur))
            cur, i = [], r["i"]
        else:
            cur.append(r)
    out.append((i, cur))
    return [(i, rs) for i, rs in out if rs]


def med(xs):
    return statistics.median(xs) if xs else None


def show(v, fmt="{:.1f}"):
    return "-" if v is None else fmt.format(v)


def arm_table(recs, lo, hi):
    """The table's rows of one arm: (lines, whether it had a step in the window)."""
    ms = {r["i"]: r["ms"] for r in records.of_kind(recs, "time_step") if not r.get("warm")}
    st = {r["i"]: r for r in records.of_kind(recs, "stat_step") if not r.get("warm")}
    steps = sorted(i for i in ms if lo <= i < hi)
    passes = [r for r in records.of_kind(recs, "residency_pass")
              if r.get("pass") == "step" and lo <= r.get("boundary", -1) < hi]
    lines = []
    mod_ms = []
    lines.append("  mod   step_ms  leg_ms  gap_p50_us  gap_max_us | passes  "
                 + "  ".join(PASS_MEDIANS) + "  " + "  ".join(PASS_SUMS))
    for m in range(MODS):
        at = [i for i in steps if i % MODS == m]
        s_ms = med([ms[i] for i in at])
        mod_ms.append(s_ms)
        legs = [st[i]["leg_us"] / 1000 for i in at if i in st and "leg_us" in st[i]]
        p50 = [st[i]["gap_p50_us"] for i in at if i in st and "gap_p50_us" in st[i]]
        gmax = [st[i]["gap_max_us"] for i in at if i in st and "gap_max_us" in st[i]]
        ps = [p for p in passes if p["boundary"] % MODS == m]
        cells = [show(med([p[f] for p in ps if f in p]), "{:.0f}") for f in PASS_MEDIANS]
        cells += [show(sum(p[f] for p in ps) if ps and all(f in p for p in ps) else None, "{:.0f}")
                  for f in PASS_SUMS]
        lines.append(f"  {m}   {show(s_ms, '{:8.2f}')}  {show(med(legs), '{:6.2f}')}  "
                     f"{show(med(p50), '{:10.1f}')}  {show(max(gmax) if gmax else None, '{:10.1f}')} | "
                     f"{len(ps):6d}  " + "  ".join(cells))
    known = [x for x in mod_ms if x is not None]
    mean = statistics.fmean([ms[i] for i in steps]) if steps else None
    excess = sum(known) - MODS * min(known) if len(known) == MODS else None
    head = (f"steps {lo}..{hi} (n={len(steps)}): mean {show(mean, '{:.2f}')} ms, cycle excess "
            f"{show(excess, '{:.2f}')} ms (the mod medians' sum less {MODS} x the least)")
    return [head] + lines, bool(steps)


def table(logs, lo, hi):
    """Every log's table as lines; the count of arms with no step in the window."""
    out, empty = [], 0
    for label, path in logs:
        for i, recs in arms(records.read(path)):
            rows, any_step = arm_table(recs, lo, hi)
            empty += not any_step
            out.append(f"table {label} arm {i}: {rows[0]}")
            out.extend(rows[1:])
    return out, empty


def parse_args(argv):
    lo, hi = STEPS
    logs = []
    it = iter(argv)
    for a in it:
        if a == "--steps":
            v = next(it, None)
            try:
                lo, hi = (int(x) for x in (v or "").split(":"))
            except ValueError:
                raise SystemExit(f"swappace-table: --steps takes A:B, not {v!r}") from None
        elif "=" in a:
            label, path = a.split("=", 1)
            logs.append((label, path))
        else:
            logs.append((os.path.basename(a), a))
    return logs, lo, hi


def render(kind, values):
    """A line of `kind` as the binary prints it, `values` by field name (a flag: True), every other
    field a zero of its type."""
    out = [kind.head]
    for p in kind.parts:
        name = p.get("key", p.get("pos"))
        if name is not None:
            ty = p["ty"]
            v = values.get(name, 0 if ty != "word" else "w")
            text = f"{float(v):.{ty.split('.')[1]}f}" if ty.startswith("f64") else str(v)
            out.append(f" {name}={text}" if "key" in p else text)
        elif "flag" in p:
            out.append(f" {p['flag']}" if values.get(p["flag"]) else "")
        else:
            out.append(p["lit"])
    return "".join(out)


def self_test():
    schema = records.load()
    k = schema.by_name
    lines = [render(k["arm"], {"i": 0, "arms": 2, "feed": "prose", "ids": 64, "n": 96})]
    # arm 0: steps 0..11 at 20 ms, mod 0 at 30; a planning pass every 4 boundaries
    for i in range(12):
        lines.append(render(k["time_step"], {"i": i, "ms": 30.0 if i % 4 == 0 else 20.0}))
        lines.append(render(k["stat_step"], {"i": i, "leg_us": 14000.0 + i, "gap_p50_us": 200.0,
                                               "gap_max_us": 300.0 + i}))
    for b in range(12):
        lines.append(render(k["residency_pass"], {
            "pass": "step", "boundary": b, "stage_us": 1000 * (b % 4), "stage_in_us": 900 * (b % 4),
            "stage_out_bytes": 5 if b % 4 == 1 else 0, "window_wait_us": 7, "prepare_us": 3,
            "late": 1 if b == 5 else 0, "forced": 0}))
    lines.append(render(k["residency_pass"], {"pass": "prompt", "boundary": 2, "late": 9}))
    lines.append(render(k["time_step"], {"i": 3, "warm": True, "ms": 99.0}))
    # arm 1: one step, outside the 0:12 window
    lines.append(render(k["arm"], {"i": 1, "arms": 2, "feed": "prose", "ids": 64, "n": 96}))
    lines.append(render(k["time_step"], {"i": 40, "ms": 21.0}))
    # an older binary's pass, without the staging window's fields: read loose, its cells `-`
    old = render(k["residency_pass"], {"pass": "step", "boundary": 1})
    old = " ".join(w for w in old.split(" ") if not w.startswith(("stage_in_us=", "stage_out_bytes=",
                                                                   "window_wait_us=")))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "run.log")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\nnot a record\n")
        out, empty = table([("t", path)], 0, 12)
        assert empty == 1, (empty, out)
        head0 = next(x for x in out if x.startswith("table t arm 0:"))
        assert "(n=12)" in head0 and "mean 22.50 ms" in head0 and "cycle excess 10.00 ms" in head0, head0
        rows = out[out.index(head0) + 2:out.index(head0) + 2 + MODS]
        assert rows[0].split()[:5] == ["0", "30.00", "14.00", "200.0", "308.0"], rows[0]
        # mod 1: boundaries 1, 5, 9 — stage_us 1000, stage_in 900, wait 7, prepare 3, out 15, late 1
        assert rows[1].split("|")[1].split() == ["3", "1000", "900", "7", "3", "15", "1", "0"], rows[1]
        assert any(x.startswith("table t arm 1:") and "(n=0)" in x for x in out), out
        path2 = os.path.join(d, "old.log")
        with open(path2, "w") as f:
            f.write(render(k["time_step"], {"i": 1, "ms": 20.0}) + "\n" + old + "\n")
        out2, _ = table([("o", path2)], 0, 4)
        assert out2[3].split("|")[1].split() == ["1", "0", "-", "-", "0", "-", "0", "0"], out2
    print("swappace-table: self-test ok")
    return 0


def main(argv):
    if argv == ["--self-test"]:
        return self_test()
    logs, lo, hi = parse_args(argv)
    if not logs:
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 64
    out, empty = table(logs, lo, hi)
    print("\n".join(out))
    return 1 if empty else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
