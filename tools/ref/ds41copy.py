#!/usr/bin/env python3
"""The copy stream beside the V4.1 decode step under adaptive residency, under Nsight.

  ds41copy.py --self-test
      the copy-stream cut held to a synthetic trace (two streams, kernels, a join gap, H2D
      chunks over each); prints `self-test: ok` or the failures (exit 1)

Called by tools/ref/nsys-ds41.sh's decode-form analyze (and so by its `--analyze` on a saved
sqlite), with the replay windows and the engine stream's host-wait gaps its own cut already
found; this module owns the other side of the river:

  - the engine stream is the stream the graph replays run on: the streamId of kernels with a
    graphNodeId, which only a replay's kernels carry (the batch's eager kernels have none);
  - every other stream with activity inside the windows is tabled as a copy stream — the
    residency machine's (`crates/gpu/src/host/swap.rs`), which stages a flip's expert parts
    in pinned host memory as H2D chunks, copies a staged r8 part to a scratch part (D2D) and
    unpacks it back into its slot (`ds41_r8_q3k`, the inverse of `qdot::repack_q3k_r8`);
    a copy that carries a graph node is a replay's own, engine work wherever it ran, and is
    counted beside the engine stream's copies instead;
  - per step (replay window) each H2D's busy time is split by what the engine stream was
    doing under it: over its host-wait gaps (the wait in front of each layer's
    `ds41_ffn_post*`, where the step waits on the host tier — copy time there is free), over
    its kernels (copy time the card could have spent on the step), or over neither (other
    gaps). An interval is clipped to its window, so a chunk that straddles a boundary bills
    each step its own part;
  - the per-step means grouped by step mod 4: the rule plans its flips at every fourth
    boundary (LIVE_DELAY, `crates/gpu-deepseek41/src/swap.rs`), so a copy-heavy step repeats
    on that cycle.

The table prints only when the run log carries a `residency host` record (the lever was on);
with the lever off the one line printed names that, and any activity on a non-engine stream
is named but not tabled (a DSpark draft's card, for one, is not the copy stream).

µs are the trace's. Windows and gaps arrive as Python objects from nsys-ds41.sh's analyze
(no JSON on disk); the self-test builds the same sqlite tables the trace reader queries
(CUPTI_ACTIVITY_KIND_KERNEL / MEMCPY / MEMSET, the column names nsys exports).
"""
import bisect
import contextlib
import io
import os
import sqlite3
import statistics
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "bloomery"))
import records  # noqa: E402 - the record reader, beside this tree's tools
from ds41pp import MEMCPY_KIND, Trace  # noqa: E402 - the trace tables' one reader


def residency_word(runlog):
    """The run's `residency host` word (the lever was on), or None."""
    try:
        recs = records.read(runlog)
    except OSError:
        return None
    for x in recs:
        if x.kind == "residency_host":
            return x["residency"]
    return None


class Union:
    """Disjoint intervals, for the length of their intersection with a query interval."""

    def __init__(self, iv):
        merged = []
        for s, e in sorted(iv):
            if e <= s:
                continue
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        self.iv = merged
        self.starts = [s for s, _ in merged]

    def overlap(self, lo, hi):
        """The total length of the intersection of [lo, hi) with the intervals."""
        if hi <= lo or not self.starts:
            return 0.0
        b = bisect.bisect_left(self.starts, hi)      # the first interval that starts at or past hi
        a = bisect.bisect_right(self.starts, lo) - 1  # the last that starts at or before lo
        tot = 0.0
        if a >= 0:
            s, e = self.iv[a]
            tot += max(0.0, min(hi, e) - lo)
        for i in range(max(a + 1, 0), b):            # start inside (lo, hi); the end may pass it
            s, e = self.iv[i]
            tot += min(e, hi) - s
        return tot


def engine_streams(tr):
    """The streams the graph replays run on (kernels with a graph node)."""
    return sorted({k["stream"] for k in tr.kernels if k["graph"] is not None})


def _blank(dev):
    return dict(dev=dev, h2d=[0, 0, 0.0, 0.0, 0.0, 0.0], d2d=[0, 0, 0.0], kern=[0, 0.0],
                by_name={}, other={})


def window_rows(tr, engine, windows, joins):
    """Per window (step), per stream that is not the engine's: the H2D/D2D/kernel numbers
    (H2D: count, bytes, busy µs, busy over the engine stream's host-wait gaps, over its
    kernels, over neither); and the engine side's own copies in `context` for the note line."""
    eng_k = sorted((k["s"], k["e"]) for k in tr.kernels if k["stream"] in engine)
    eng_starts = [s for s, _ in eng_k]
    rows = {}
    context = dict(kind={"H2D": [0, 0, 0.0], "D2H": [0, 0, 0.0], "D2D": [0, 0, 0.0]},
                   graph=0, other={})
    for r, t0, t1 in windows:
        hi = t1 if t1 is not None else float("inf")
        lo_i = max(bisect.bisect_right(eng_starts, t0) - 1, 0)
        while lo_i > 0 and eng_k[lo_i - 1][1] > t0:  # an earlier overlapping one may straddle t0
            lo_i -= 1
        hi_i = bisect.bisect_left(eng_starts, hi)
        kern_u = Union([(max(s, t0), min(e, hi)) for s, e in eng_k[lo_i:hi_i] if e > t0 and s < hi])
        gap_u = Union(joins.get(r, []))
        rows[r] = {}
        for act in tr.copies:
            s, e = max(act["s"], t0), min(act["e"], hi)
            if e <= s:
                continue
            dur = (e - s) / 1e3
            # A straddling activity's count and bytes bill the window its start falls in, so
            # they sum to the run's totals; its time is clipped to each window it touches.
            starts_here = t0 <= act["s"] < hi
            if act["graph"] is not None or act["stream"] in engine:
                if not starts_here:
                    continue
                if act["stream"] not in engine:
                    # a replay's own copy, wherever the graph put it
                    context["graph"] += 1
                elif act["name"] in context["kind"]:
                    c = context["kind"][act["name"]]
                    c[0] += 1
                    c[1] += act["bytes"] or 0
                    c[2] += dur
                else:
                    context["other"][act["name"]] = context["other"].get(act["name"], 0) + 1
                continue
            d = rows[r].setdefault(act["stream"], _blank(act["dev"]))
            if act["name"] == "H2D":
                in_gap, in_kern = gap_u.overlap(s, e) / 1e3, kern_u.overlap(s, e) / 1e3
                d["h2d"][2] += dur
                d["h2d"][3] += in_gap
                d["h2d"][4] += in_kern
                d["h2d"][5] += dur - in_gap - in_kern
                if starts_here:
                    d["h2d"][0] += 1
                    d["h2d"][1] += act["bytes"] or 0
            elif act["name"] == "D2D":
                d["d2d"][2] += dur
                if starts_here:
                    d["d2d"][0] += 1
                    d["d2d"][1] += act["bytes"] or 0
            else:
                if starts_here:
                    d["other"][act["name"]] = d["other"].get(act["name"], 0) + 1
        for k in tr.kernels:
            if k["stream"] in engine or k["graph"] is not None or not (k["e"] > t0 and k["s"] < hi):
                continue
            s, e = max(k["s"], t0), min(k["e"], hi)
            d = rows[r].setdefault(k["stream"], _blank(k["dev"]))
            d["kern"][0] += 1
            d["kern"][1] += (e - s) / 1e3
            n = d["by_name"].setdefault(k["name"], [0, 0.0])
            n[0] += 1
            n[1] += (e - s) / 1e3
    return rows, context


COLS = (f"  {'step':>6s} {'mod':>3s} {'H2D n':>6s} {'H2D B':>10s} {'busy':>8s} {'in gap':>8s} "
        f"{'in kern':>8s} {'rest':>7s} | {'D2D n':>5s} {'D2D B':>10s} {'D2D':>7s} | "
        f"{'kern n':>6s} {'kern µs':>8s}")


def tables(path, runlog, windows, joins, top=24):
    """Print the copy-stream tables for the windows (nsys-ds41.sh's analyze calls this); the
    per-step rows, or None when the run carries no residency or no graph kernels."""
    word = residency_word(runlog)
    tr = Trace(path)
    engine = engine_streams(tr)
    rows, context = window_rows(tr, engine, windows, joins)
    streams = sorted({s for r in rows for s in rows[r]})
    if word is None:
        what = ""
        if streams:
            what = (f"; the activity on non-engine stream(s) {', '.join(str(s) for s in streams)} inside "
                    f"the windows is not tabled (not the residency machine's copy stream)")
        print(f"[copy-stream] no copy-stream table: the run log carries no `residency host` record "
              f"(no run log was given, or BLOOMERY_RESIDENCY was off at the run){what}")
        return None
    if not engine:
        print("[copy-stream] no copy-stream table: no kernel in the trace carries a graph node, so no "
              "stream is the engine's")
        return None
    eng_n = sum(1 for k in tr.kernels if k["graph"] is not None)
    print(f"[copy-stream] residency {word} (the run log's `residency host` record); the engine stream is "
          f"{', '.join(str(s) for s in engine)} — the stream of the {eng_n} graph kernels, the replays'; "
          f"every other stream with activity inside the windows is tabled below as the copy stream")
    side = ", ".join(f"{v[0]} {k} ({v[1]} B, {v[2]:.1f} µs)" for k, v in context["kind"].items() if v[0])
    bits = [f"on the engine stream: {side or 'no copies'}"]
    if context["other"]:
        bits.append("beside them: " + ", ".join(f"{v} {k}" for k, v in sorted(context["other"].items())))
    if context["graph"]:
        bits.append(f"{context['graph']} with a graph node (a replay's own, on another stream)")
    print(f"    copies inside the windows {'; '.join(bits)}")
    if not streams:
        print("    no activity on any other stream inside the windows: the rule copied nothing the "
              "tabled steps cover")
        return {}
    for st in streams:
        steps_of = {r: rows[r][st] for r in rows if st in rows[r]}
        dev = next(iter(steps_of.values()))["dev"]
        print()
        print(f"=== copy stream {st} (device {dev}) per step (µs; busy = the activity's own time clipped "
              f"to the step's window; in gap = over the engine stream's host-wait gaps, in kern = over "
              f"its kernels)")
        print(COLS)
        for r, d in steps_of.items():
            h, dd, kn = d["h2d"], d["d2d"], d["kern"]
            print(f"  {r:6d} {r % 4:3d} {h[0]:6d} {h[1]:10d} {h[2]:8.1f} {h[3]:8.1f} {h[4]:8.1f} "
                  f"{h[5]:7.1f} | {dd[0]:5d} {dd[1]:10d} {dd[2]:7.1f} | {kn[0]:6d} {kn[1]:8.1f}")
        print()
        print(f"=== copy stream {st} per step mod 4 (the rule's flip cycle), the means of the steps above")
        print(f"  {'mod':>3s} {'steps':>5s} {'H2D n':>6s} {'H2D B':>10s} {'busy':>8s} {'in gap':>8s} "
              f"{'in kern':>8s} {'rest':>7s} | {'D2D n':>5s} {'D2D B':>10s} {'D2D':>7s} | "
              f"{'kern n':>6s} {'kern µs':>8s}")
        for m in sorted({r % 4 for r in steps_of}):
            sel = [r for r in steps_of if r % 4 == m]
            h = [statistics.fmean(steps_of[r]["h2d"][j] for r in sel) for j in range(6)]
            dd = [statistics.fmean(steps_of[r]["d2d"][j] for r in sel) for j in range(3)]
            kn = [statistics.fmean(steps_of[r]["kern"][j] for r in sel) for j in range(2)]
            print(f"  {m:3d} {len(sel):5d} {h[0]:6.1f} {h[1]:10.0f} {h[2]:8.1f} {h[3]:8.1f} {h[4]:8.1f} "
                  f"{h[5]:7.1f} | {dd[0]:5.1f} {dd[1]:10.0f} {dd[2]:7.1f} | {kn[0]:6.1f} {kn[1]:8.1f}")
        by_name = {}
        for r, d in steps_of.items():
            for name, (c, us) in d["by_name"].items():
                e = by_name.setdefault(name, [0, 0.0])
                e[0] += c
                e[1] += us
        if by_name:
            n = len(steps_of)
            print()
            print(f"=== copy stream {st} kernels by name (per step, mean over the {n} step{'' if n == 1 else 's'} "
                  f"above)")
            print(f"  {'kernel':44s} {'n/step':>7s} {'µs/step':>9s} {'µs/n':>8s}")
            for name, (c, us) in sorted(by_name.items(), key=lambda x: -x[1][1])[:top]:
                print(f"  {name[:44]:44s} {c / n:7.2f} {us / n:9.1f} {us / c:8.2f}")
        other = {}
        for r, d in steps_of.items():
            for name, c in d["other"].items():
                other[name] = other.get(name, 0) + c
        if other:
            print(f"    other activity on stream {st} inside the windows: "
                  + ", ".join(f"{v} {k}" for k, v in sorted(other.items())))
    return rows


# ---- the self-test ----

ENG, CPY = 7, 9
MS = 1_000_000.0  # the times below are written in ms units


class Synth:
    """Two streams, graph kernels (the engine stream's replays), an eager kernel, H2D chunks
    (one over the join gap, one over a kernel, one straddling a window's end, one before every
    window), a D2D, an unpack kernel and a memset on the copy stream, and an H2D on the engine
    stream: every number the tables print is written by hand beside it."""

    def __init__(self):
        self.names, self.kernels, self.copies, self.sets = {}, [], [], []
        self.corr = 0

    def sid(self, s):
        return self.names.setdefault(s, len(self.names) + 1)

    def kern(self, s, e, name, stream, graph=None, dev=0):
        self.corr += 1
        self.kernels.append((s, e, self.sid(name), self.corr, stream, dev, 1, 1, 1, 256, 1, 1, graph))

    def copy(self, s, e, kind, size, stream, dev=0, graph=None):
        self.corr += 1
        self.copies.append((s, e, kind, size, self.corr, stream, dev, graph))

    def memset(self, s, e, size, stream, dev=0):
        self.corr += 1
        self.sets.append((s, e, size, self.corr, stream, dev, None))

    def write(self, path):
        db = sqlite3.connect(path)
        db.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
        db.executemany("INSERT INTO StringIds VALUES (?, ?)", [(v, k) for k, v in self.names.items()])
        db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start, end, shortName, correlationId, "
                   "streamId, deviceId, gridX, gridY, gridZ, blockX, blockY, blockZ, graphNodeId)")
        db.executemany(f"INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES ({','.join('?' * 13)})",
                       self.kernels)
        db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start, end, copyKind, bytes, "
                   "correlationId, streamId, deviceId, graphNodeId)")
        db.executemany(f"INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES ({','.join('?' * 8)})",
                       self.copies)
        db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET (start, end, bytes, correlationId, "
                   "streamId, deviceId, graphNodeId)")
        db.executemany(f"INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET VALUES ({','.join('?' * 7)})",
                       self.sets)
        # ds41pp's Trace reads the API table too (nsys exports it with -t cuda); empty here.
        db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start, end, nameId, correlationId, "
                   "globalTid)")
        db.commit()
        db.close()


def synth():
    """The synthetic trace: (its writer, the windows, the joins, the expected per-step rows)."""
    s = Synth()
    # The prompt's eager kernel, before every window, on the engine stream.
    s.kern(0.90 * MS, 0.95 * MS, "ds41_glue_embed_q3k", ENG)
    # Step 100, window [1.0, 2.0) ms: handoff, a card kernel, the post 450 µs later (the join
    # gap 1.15-1.6), and the head near the end.
    s.kern(1.00 * MS, 1.05 * MS, "ds41_ffn_handoff", ENG, graph=101)
    s.kern(1.05 * MS, 1.15 * MS, "ds41_card_gate", ENG, graph=102)
    s.kern(1.60 * MS, 1.70 * MS, "ds41_ffn_post", ENG, graph=103)
    s.kern(1.95 * MS, 1.98 * MS, "ds41_head_logits", ENG, graph=104)
    # Step 101, window [2.0, 3.0) ms; join gap 2.15-2.6.
    s.kern(2.00 * MS, 2.05 * MS, "ds41_ffn_handoff", ENG, graph=201)
    s.kern(2.05 * MS, 2.15 * MS, "ds41_card_gate", ENG, graph=202)
    s.kern(2.60 * MS, 2.65 * MS, "ds41_ffn_post", ENG, graph=203)
    s.kern(2.90 * MS, 2.95 * MS, "ds41_head_logits", ENG, graph=204)
    # Step 102, the last window (no end): one kernel, no join.
    s.kern(3.00 * MS, 3.10 * MS, "ds41_ffn_handoff", ENG, graph=301)
    # The copy stream: an H2D before every window (not counted), then per window the chunks.
    s.copy(0.95 * MS, 0.99 * MS, 1, 4096, CPY)
    # Step 100: A over the gap (300 µs), B over the post kernel (100 of 150), C straddling the
    # window's end (100 of 200 kept, 30 of it over the head kernel).
    s.copy(1.20 * MS, 1.50 * MS, 1, 2 * 1024 * 1024, CPY)
    s.copy(1.60 * MS, 1.75 * MS, 1, 1024 * 1024, CPY)
    s.copy(1.90 * MS, 2.10 * MS, 1, 3 * 1024 * 1024, CPY)
    s.copy(1.55 * MS, 1.60 * MS, 8, 512, CPY)
    s.kern(1.50 * MS, 1.55 * MS, "ds41_r8_q3k", CPY)
    s.memset(1.98 * MS, 1.99 * MS, 256, CPY)
    # Step 101: D over the two kernels and into the gap (150 + 50 of 200).
    s.copy(2.00 * MS, 2.20 * MS, 1, 1024 * 1024, CPY)
    s.kern(2.20 * MS, 2.30 * MS, "ds41_r8_q3k", CPY)
    s.copy(2.50 * MS, 2.55 * MS, 3, 128, CPY)  # a kind the reader names, not tables
    # Step 102: one H2D half over the kernel, half over nothing (no next window).
    s.copy(3.05 * MS, 3.20 * MS, 1, 1024, CPY)
    # The engine stream's own H2D inside a window (the context line), and a copy that carries a
    # graph node on another stream (a replay's own, not the copy stream's).
    s.copy(1.70 * MS, 1.72 * MS, 1, 4096, ENG)
    s.copy(1.80 * MS, 1.82 * MS, 1, 2048, 11, graph=105)
    windows = [(100, 1.00 * MS, 2.00 * MS), (101, 2.00 * MS, 3.00 * MS), (102, 3.00 * MS, None)]
    joins = {100: [(1.15 * MS, 1.60 * MS)], 101: [(2.15 * MS, 2.60 * MS)], 102: []}
    want = {
        100: dict(h2d=[3, 6 * 1024 * 1024, 550.0, 300.0, 130.0, 120.0], d2d=[1, 512, 50.0],
                  kern=[1, 50.0], other={"memset": 1}),
        101: dict(h2d=[1, 1024 * 1024, 300.0, 50.0, 250.0, 0.0], d2d=[0, 0, 0.0],
                  kern=[1, 100.0], other={"copy3": 1}),
        102: dict(h2d=[1, 1024, 150.0, 0.0, 50.0, 100.0], d2d=[0, 0, 0.0],
                  kern=[0, 0.0], other={}),
    }
    return s, windows, joins, want


RUNLOG_ON = ("load resident_bytes=1 shadow=host 0 unified_addressing=1 ctx=4096 layers=40 top_k=512 "
             "mode=graph place=a pin_main=on pinned=true prefill=batch ced=on group=1 in 1.0 s "
             "(runtime value)\n"
             "residency host residency=mid-p40-s1 pinned=63 churn_experts=10 churn_bytes=1000 "
             "headroom=100 headroom_after=90\n")
RUNLOG_OFF = ("load resident_bytes=1 shadow=host 0 unified_addressing=1 ctx=4096 layers=40 top_k=512 "
              "mode=graph place=a pin_main=on pinned=true prefill=batch ced=on group=1 in 1.0 s "
              "(runtime value)\n")


def cmd_self_test(argv):
    fails, checks = [], 0

    def check(ok, what):
        nonlocal checks
        checks += 1
        if not ok:
            fails.append(what)

    def close(a, b, what):
        check(abs(a - b) < 0.05, f"{what}: {a:.3f}, expected {b:.3f}")

    s, windows, joins, want = synth()
    with tempfile.TemporaryDirectory() as d:
        db, log = os.path.join(d, "syn.sqlite"), os.path.join(d, "run.txt")
        s.write(db)
        with open(log, "w") as f:
            f.write(RUNLOG_ON)
        tr = Trace(db)
        check(engine_streams(tr) == [ENG], f"engine streams {engine_streams(tr)}, the graph kernels' [7]")
        rows, context = window_rows(tr, [ENG], windows, joins)
        check(sorted(rows) == [100, 101, 102], f"windows tabled {sorted(rows)}")
        check(context["kind"]["H2D"] == [1, 4096, 20.0], f"engine-stream H2D context "
                                                         f"{context['kind']['H2D']}, expected [1, 4096, 20.0]")
        check(context["graph"] == 1, f"graph-node copies {context['graph']}, expected 1 (not tabled)")
        for r, exp in want.items():
            got = rows[r][CPY]
            check(got["h2d"][0] == exp["h2d"][0], f"step {r}: H2D count {got['h2d'][0]}, expected {exp['h2d'][0]}")
            check(got["h2d"][1] == exp["h2d"][1], f"step {r}: H2D bytes {got['h2d'][1]}, expected {exp['h2d'][1]}")
            for j, name in enumerate(("busy", "in gap", "in kern", "rest"), start=2):
                close(got["h2d"][j], exp["h2d"][j], f"step {r}: H2D {name}")
            check(got["d2d"][0] == exp["d2d"][0] and got["d2d"][1] == exp["d2d"][1],
                  f"step {r}: D2D {got['d2d'][:2]}, expected {exp['d2d'][:2]}")
            close(got["d2d"][2], exp["d2d"][2], f"step {r}: D2D busy")
            check(got["kern"][0] == exp["kern"][0], f"step {r}: copy-stream kernel count {got['kern'][0]}")
            close(got["kern"][1], exp["kern"][1], f"step {r}: copy-stream kernel µs")
            check(got["other"] == exp["other"], f"step {r}: other activity {got['other']}, expected {exp['other']}")
            check(11 not in rows[r], f"step {r}: a graph-node copy's stream was tabled as a copy stream")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            got = tables(db, log, windows, joins)
        text = out.getvalue()
        check(got is not None, "tables refused a residency run log")
        check("residency mid-p40-s1" in text, "the tables do not name the residency word")
        check("engine stream is 7" in text, "the tables do not name the engine stream")
        check("copy stream 9" in text, "the tables do not name the copy stream")
        check("1 H2D (4096 B, 20.0 µs)" in text, "the engine stream's own copies are missing")
        check("1 with a graph node" in text, "the graph-node copy is not named beside them")
        # The mod-4 means: one step a class here, so each row is that step's values.
        for m, r in ((0, 100), (1, 101), (2, 102)):
            line = next((x for x in text.splitlines() if x.split() and x.split()[0] == str(m)), None)
            check(line is not None, f"no mod {m} row in the means table")
            if line:
                close(float(line.split()[4]), want[r]["h2d"][2], f"mod {m} mean busy")
                close(float(line.split()[5]), want[r]["h2d"][3], f"mod {m} mean in gap")
        # The lever off: one line naming it, no tables, and the other stream named but not tabled.
        with open(log, "w") as f:
            f.write(RUNLOG_OFF)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            got = tables(db, log, windows, joins)
        text = out.getvalue()
        check(got is None and "no `residency host` record" in text and "stream(s) 9" in text,
              "the lever-off line does not name the refusal and the non-engine stream")
    for f in fails:
        print(f"self-test FAIL: {f}")
    print(f"self-test: {'ok' if not fails else 'FAIL'} ({checks} checks, {len(fails)} failures)")
    return 1 if fails else 0


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--self-test":
        raise SystemExit(cmd_self_test(sys.argv[2:]))
    print(__doc__)
    raise SystemExit(64)


if __name__ == "__main__":
    main()
