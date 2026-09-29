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
  - the unpack hold, printed when the copy stream runs the swap's r8 unpack (a kernel whose
    name starts `ds41_r8_q3k`, the plain and the group form): per unpack kernel its launches
    against the stage card's engine kernels (how many start during one launch, how long after
    a launch's end the next one waits), and the stage-card layer periods (one handoff kernel
    end to the next, whatever form the engine launched it in) split by (heavy/light step, the
    step's copy-stream H2D bytes) x (an unpack overlaps the period's longest engine gap) — the
    wait an unpack holds open. The stage card is the unpack kernels' device, and the steps are
    bounded by the graph `argmax_fault` kernel's ends; a trace missing the handoff or the
    step-end kernel names it and skips only the layer split.

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
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "bloomery"))
import records  # noqa: E402 - the record reader, beside this tree's tools
from ds41pp import MEMCPY_KIND, Trace  # noqa: E402 - the trace tables' one reader

# The unpack-hold cut's kernel names and classes, one block: the swap's r8 unpack (the plain and
# the group form), the routed layer's handoff to the host tier, the kernel whose graph launch ends
# a step. A step is heavy when its copy-stream H2D passes HEAVY_H2D_BYTES, and the first
# WARMUP_STEPS steps are the load's tail, before the rule's first flip.
UNPACK_PREFIX = "ds41_r8_q3k"
STEP_END_KERNEL = "argmax_fault"
WARMUP_STEPS = 10
HEAVY_H2D_BYTES = 60 * 1000 * 1000
# The routed layer's handoff kernel, every form the engine launches it in: the plain one (six
# slots a token on V4.1) and its eight- and ten-slot and multi-column forms
# (gpu/src/host/handoff.rs `enqueue_handoff`), and the two-card tier card's
# (gpu-deepseek41/src/chain/ffn/tier.rs). A prefix, not a list: which form runs is the engine's
# shape and placement choice, and nsys-ds41.sh's join cut reads this same definition.
HANDOFF_PREFIX = "ds41_ffn_handoff"


def is_handoff(name):
    """Whether `name` is a routed layer's handoff kernel (the family `HANDOFF_PREFIX` names)."""
    return name.startswith(HANDOFF_PREFIX)


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

    def minus(self, other):
        """These intervals minus `other`'s, as a Union: what `overlap` counts over the parts of
        this union the other one does not cover."""
        out = []
        for s, e in self.iv:
            cur = s
            for gs, ge in other.iv:
                if ge <= cur or gs >= e:
                    continue
                if gs > cur:
                    out.append((cur, min(gs, e)))
                cur = max(cur, ge)
                if cur >= e:
                    break
            if cur < e:
                out.append((cur, e))
        return Union(out)


def engine_streams(tr):
    """The (deviceId, streamId) pairs the graph replays run on (kernels with a graph node)."""
    return sorted({(k["dev"], k["stream"]) for k in tr.kernels if k["graph"] is not None})


def _blank(dev):
    return dict(dev=dev, h2d=[0, 0, 0.0, 0.0, 0.0, 0.0], d2d=[0, 0, 0.0], kern=[0, 0.0],
                by_name={}, other={})


def window_rows(tr, engine, windows, joins):
    """Per window (step), per stream that is not the engine's: the H2D/D2D/kernel numbers
    (H2D: count, bytes, busy µs, busy over the engine stream's host-wait gaps, over its
    kernels, over neither); and the engine side's own copies in `context` for the note line."""
    eng_k = sorted((k["s"], k["e"]) for k in tr.kernels if (k["dev"], k["stream"]) in engine)
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
        # The two classes must be disjoint: under a residency the engine streams run eager
        # kernels between the replays and under two cards another card's kernel can sit inside
        # a host-wait gap, and the gap wins — the copy's time there hides the step's wait.
        kern_off_gap = kern_u.minus(gap_u)
        rows[r] = {}
        for act in tr.copies:
            s, e = max(act["s"], t0), min(act["e"], hi)
            if e <= s:
                continue
            dur = (e - s) / 1e3
            # A straddling activity's count and bytes bill the window its start falls in, so
            # they sum to the run's totals; its time is clipped to each window it touches.
            starts_here = t0 <= act["s"] < hi
            if act["graph"] is not None or (act["dev"], act["stream"]) in engine:
                if not starts_here:
                    continue
                if (act["dev"], act["stream"]) not in engine:
                    # a replay's own copy, wherever the graph put it
                    context["graph"] += 1
                elif act["name"] in context["kind"]:
                    c = context["kind"][act["name"]]
                    c[0] += 1
                    c[1] += act["bytes"] or 0
                    # the whole copy, not its clipped part: this window is the only one that
                    # bills it, and a copy that runs past the window's end keeps its tail
                    c[2] += (act["e"] - act["s"]) / 1e3
                else:
                    context["other"][act["name"]] = context["other"].get(act["name"], 0) + 1
                continue
            d = rows[r].setdefault((act["dev"], act["stream"]), _blank(act["dev"]))
            if act["name"] == "H2D":
                in_gap, in_kern = gap_u.overlap(s, e) / 1e3, kern_off_gap.overlap(s, e) / 1e3
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
            if (k["dev"], k["stream"]) in engine or k["graph"] is not None or not (k["e"] > t0 and k["s"] < hi):
                continue
            s, e = max(k["s"], t0), min(k["e"], hi)
            d = rows[r].setdefault((k["dev"], k["stream"]), _blank(k["dev"]))
            # A straddling kernel's launch counts once, in the window its start falls in (as a
            # straddling copy's does); its time is clipped to each window it touches.
            starts_here = t0 <= k["s"] < hi
            if starts_here:
                d["kern"][0] += 1
            d["kern"][1] += (e - s) / 1e3
            n = d["by_name"].setdefault(k["name"], [0, 0.0])
            if starts_here:
                n[0] += 1
            n[1] += (e - s) / 1e3
    return rows, context


def _rank(vals, idx):
    """sorted(vals)[idx] — the p10/p90 rank cut over the samples themselves."""
    return sorted(vals)[idx]


def unpack_hold(tr, engine):
    """The copy stream's r8 unpack against the stage card's engine kernels (the docstring's
    unpack-hold bullet). None when no copy-stream kernel's name starts with the unpack prefix;
    else a dict: stage_dev (the unpack kernels' device), names ({kernel name: [launches,
    median µs, p90 µs, grid, block, engine starts during a launch (median, max), first engine
    start after a launch's end (median, p10, p90 µs), None when none follows]}), split
    ({(class, overlapped): [n, median period µs, median longest gap µs, median engine busy µs]}
    over the four classes, the medians None at n = 0), hold_us (the heavy steps' overlapped −
    clean median period, None when a side is empty) and missing (the named lines why a part is
    absent, empty when all of it printed)."""
    unpack = sorted((k for k in tr.kernels
                     if k["name"].startswith(UNPACK_PREFIX) and k["graph"] is None
                     and (k["dev"], k["stream"]) not in engine), key=lambda k: k["s"])
    if not unpack:
        return None
    res = dict(stage_dev=None, names={}, split={}, hold_us=None, missing=[])
    devs = {k["dev"] for k in unpack}
    if len(devs) != 1:
        res["missing"].append(f"the unpack kernels run on devices {sorted(devs)}, not one stage card: "
                              "the unpack-hold table is skipped")
        return res
    dev = res["stage_dev"] = devs.pop()
    eng = sorted((k["s"], k["e"]) for k in tr.kernels if k["graph"] is not None and k["dev"] == dev)
    es = [s for s, _ in eng]
    for name in sorted({k["name"] for k in unpack}):
        ks = [k for k in unpack if k["name"] == name]
        durs = [(k["e"] - k["s"]) / 1e3 for k in ks]
        during = [bisect.bisect_left(es, k["e"]) - bisect.bisect_left(es, k["s"]) for k in ks]
        after = []
        for k in ks:
            j = bisect.bisect_left(es, k["e"])
            if j < len(eng):
                after.append((eng[j][0] - k["e"]) / 1e3)
        grid, block = Counter((k["grid"], k["block"]) for k in ks).most_common(1)[0][0]
        res["names"][name] = [
            len(ks), statistics.median(durs), _rank(durs, 9 * len(durs) // 10), grid, block,
            statistics.median(during), max(during),
            statistics.median(after) if after else None,
            _rank(after, len(after) // 10) if after else None,
            _rank(after, 9 * len(after) // 10) if after else None]
    ends = sorted(k["e"] for k in tr.kernels
                  if k["graph"] is not None and k["dev"] == dev and k["name"] == STEP_END_KERNEL)
    d0 = sorted((k for k in tr.kernels if k["graph"] is not None and k["dev"] == dev),
                key=lambda k: k["s"])
    ho = [i for i, k in enumerate(d0) if is_handoff(k["name"])]
    if not ends:
        res["missing"].append(f"no {STEP_END_KERNEL} graph kernel on device {dev}: the layer split is skipped")
    if not ho:
        res["missing"].append(f"no {HANDOFF_PREFIX}* graph kernel on device {dev}: the layer split is skipped")
    if res["missing"]:
        return res
    step_h2d = Counter()
    for a in tr.copies:
        if a["name"] != "H2D" or a["graph"] is not None or a["dev"] != dev \
                or (a["dev"], a["stream"]) in engine:
            continue
        i = bisect.bisect_left(ends, a["s"])
        if i < len(ends):
            step_h2d[i] += a["bytes"] or 0
    ui = [(k["s"], k["e"]) for k in unpack]
    buckets = {}
    for a, b in zip(ho, ho[1:]):
        s, e = d0[a]["e"], d0[b]["e"]
        step = bisect.bisect_left(ends, s)
        if step < WARMUP_STEPS or step >= len(ends) or step != bisect.bisect_left(ends, e):
            continue
        gaps = [(d0[j + 1]["s"] - d0[j]["e"], d0[j]["e"], d0[j + 1]["s"]) for j in range(a, b)]
        if not gaps:
            continue
        g, gs, ge = max(gaps)
        busy = sum(d0[j]["e"] - d0[j]["s"] for j in range(a + 1, b + 1))
        overlapped = any(us < ge and ue > gs for us, ue in ui)
        cl = "heavy" if step_h2d[step] > HEAVY_H2D_BYTES else "light"
        buckets.setdefault((cl, overlapped), []).append(((e - s) / 1e3, g / 1e3, busy / 1e3))
    for key in (("heavy", True), ("heavy", False), ("light", True), ("light", False)):
        v = buckets.get(key, [])
        med = [statistics.median(x[i] for x in v) for i in range(3)] if v else [None] * 3
        res["split"][key] = [len(v)] + med
    hy, hn = res["split"][("heavy", True)], res["split"][("heavy", False)]
    if hy[0] and hn[0]:
        res["hold_us"] = hy[1] - hn[1]
    return res


def _print_unpack_hold(hold):
    """The unpack-hold table's lines (nothing when unpack_hold found no unpack kernels)."""
    if hold is None:
        return

    def num(v, spec):
        return format(v, spec) if v is not None else "-"

    where = f", device {hold['stage_dev']}" if hold["stage_dev"] is not None else ""
    print()
    print(f"=== unpack hold (the copy stream's r8 unpack against the stage card's engine kernels{where})")
    for m in hold["missing"]:
        print(f"    {m}")
    if hold["names"]:
        print(f"  {'unpack kernel':30s} {'launch':>7s} {'med µs':>8s} {'p90 µs':>8s} {'grid':>18s} "
              f"{'during med':>10s} {'during max':>10s} {'after med':>9s} {'after p10':>9s} {'after p90':>9s}")
        for name, r in sorted(hold["names"].items()):
            grid = f"{r[3][0]}x{r[3][1]}x{r[3][2]}|{r[4][0]}x{r[4][1]}x{r[4][2]}"
            print(f"  {name[:30]:30s} {r[0]:7d} {r[1]:8.1f} {r[2]:8.1f} {grid:>18s} "
                  f"{r[5]:10.1f} {r[6]:10d} {num(r[7], '9.1f')} {num(r[8], '9.1f')} {num(r[9], '9.1f')}")
    if hold["split"]:
        print()
        print("  stage-card layer periods, one handoff end to the next, by step class x whether an "
              "unpack overlaps the period's longest engine gap")
        print(f"  {'class':24s} {'n':>5s} {'period µs':>10s} {'longest gap µs':>14s} {'engine busy µs':>14s}")
        for cl, ov in (("heavy", True), ("heavy", False), ("light", True), ("light", False)):
            n, per, gap, busy = hold["split"][(cl, ov)]
            lab = f"{cl}, {'unpack in the gap' if ov else 'clean'}"
            print(f"  {lab:24s} {n:5d} {num(per, '10.1f')} {num(gap, '14.1f')} {num(busy, '14.1f')}")
    if hold["hold_us"] is not None:
        print(f"  unpack hold: overlapped - clean layer period = {hold['hold_us']:.1f} µs (heavy steps)")


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
          f"{', '.join(str(s) for s in engine)} — the (deviceId, streamId) of the {eng_n} graph kernels, "
          f"the replays'; every other stream with activity inside the windows is tabled below as the "
          f"copy stream")
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
    for dev, st in streams:
        steps_of = {r: rows[r][(dev, st)] for r in rows if (dev, st) in rows[r]}
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
    _print_unpack_hold(unpack_hold(tr, engine))
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

    def kern(self, s, e, name, stream, graph=None, dev=0, grid=(1, 1, 1), block=(256, 1, 1)):
        self.corr += 1
        self.kernels.append((s, e, self.sid(name), self.corr, stream, dev,
                             grid[0], grid[1], grid[2], block[0], block[1], block[2], graph))

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


def synth_hold(handoff="ds41_ffn_handoff_tier"):
    """The unpack-hold synthetic: twelve steps bounded by `argmax_fault` ends (the first ten the
    load's tail, skipped), a heavy step with two layers — the first period's longest engine gap
    held open by an unpack launch that spans the starts of two engine kernels, the second clean —
    and a light clean step. `handoff` is the handoff kernel's name: the tier card's (plan (b′))
    or the plain one (plan (a)), the same cut read for both. A tier-card kernel with a graph
    node runs on (device 1, stream 9): the copy stream's streamId under an engine stream's, so
    keying the streams by streamId alone (the mutant this trace must fail) would call the copy
    stream an engine stream and lose the unpack table whole."""
    s = Synth()
    for i in range(12):
        t = i * MS
        s.kern(t + 0.10 * MS, t + 0.15 * MS, "ds41_card_gate", CPY, graph=1000 + i, dev=1)
        s.kern(t + 0.98 * MS, t + 0.99 * MS, "argmax_fault", ENG, graph=100 + i)
    # Step 10, heavy (a 70 MB H2D on the copy stream): period 1 has its 200 µs gap [10.06, 10.26]
    # overlapped by the unpack [10.005, 10.265] (two engine starts under it: 10.010, 10.260; the
    # next engine start, 10.310, waits 45 µs after its end); period 2 is clean.
    s.kern(10.00 * MS, 10.01 * MS, handoff, ENG, graph=210)
    s.kern(10.01 * MS, 10.06 * MS, "ds41_card_gate", ENG, graph=211)
    s.kern(10.26 * MS, 10.31 * MS, "ds41_card_gate", ENG, graph=212)
    s.kern(10.31 * MS, 10.32 * MS, handoff, ENG, graph=213)
    s.kern(10.32 * MS, 10.36 * MS, "ds41_card_gate", ENG, graph=214)
    s.kern(10.46 * MS, 10.50 * MS, "ds41_card_gate", ENG, graph=215)
    s.kern(10.50 * MS, 10.51 * MS, handoff, ENG, graph=216)
    s.kern(10.005 * MS, 10.265 * MS, "ds41_r8_q3k", CPY, grid=(4950, 1, 1))
    s.copy(10.60 * MS, 10.70 * MS, 1, 70 * 1000 * 1000, CPY)
    # Step 11, light (10 MB) and clean; a group unpack outside its gaps changes no class (the
    # first engine start after its end, the step's argmax at 11.98, waits 380 µs).
    s.kern(11.00 * MS, 11.01 * MS, handoff, ENG, graph=310)
    s.kern(11.01 * MS, 11.06 * MS, "ds41_card_gate", ENG, graph=311)
    s.kern(11.16 * MS, 11.21 * MS, "ds41_card_gate", ENG, graph=312)
    s.kern(11.21 * MS, 11.22 * MS, handoff, ENG, graph=313)
    s.kern(11.50 * MS, 11.60 * MS, "ds41_r8_q3k_groups", CPY, grid=(64, 1, 1))
    s.copy(11.60 * MS, 11.70 * MS, 1, 10 * 1000 * 1000, CPY)
    return s


def synth_bare():
    """No handoff and no step-end kernel: the per-name rows still print (no engine kernel follows
    the unpack, its after-stats are `-`), and the layer split is skipped by name."""
    s = Synth()
    s.kern(1.00 * MS, 1.05 * MS, "ds41_card_gate", ENG, graph=1)
    s.kern(2.00 * MS, 2.10 * MS, "ds41_r8_q3k", CPY)
    return s


def synth_edges():
    """The window edges: a copy-stream kernel straddling the boundary (one launch, its time
    split), an engine-stream copy that runs past the window's end (its context µs whole), and
    an eager engine kernel inside a join gap (the H2D split's classes disjoint, the gap first)."""
    s = Synth()
    s.kern(1.00 * MS, 1.05 * MS, "ds41_ffn_handoff", ENG, graph=101)
    s.kern(1.05 * MS, 1.15 * MS, "ds41_card_gate", ENG, graph=102)
    s.kern(1.60 * MS, 1.70 * MS, "ds41_ffn_post", ENG, graph=103)
    s.kern(1.95 * MS, 1.97 * MS, "ds41_head_logits", ENG, graph=104)
    s.kern(2.00 * MS, 2.05 * MS, "ds41_ffn_handoff", ENG, graph=201)
    s.kern(2.90 * MS, 2.95 * MS, "ds41_head_logits", ENG, graph=202)
    s.kern(1.97 * MS, 2.03 * MS, "ds41_stage_thing", CPY)
    s.copy(1.90 * MS, 2.10 * MS, 1, 4096, ENG)
    s.kern(1.30 * MS, 1.40 * MS, "ds41_boundary_work", ENG)
    s.copy(1.25 * MS, 1.55 * MS, 1, 1024, CPY)
    return s


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
        check(engine_streams(tr) == [(0, ENG)],
              f"engine streams {engine_streams(tr)}, the graph kernels' [(0, 7)]")
        rows, context = window_rows(tr, [(0, ENG)], windows, joins)
        check(sorted(rows) == [100, 101, 102], f"windows tabled {sorted(rows)}")
        check(context["kind"]["H2D"] == [1, 4096, 20.0], f"engine-stream H2D context "
                                                         f"{context['kind']['H2D']}, expected [1, 4096, 20.0]")
        check(context["graph"] == 1, f"graph-node copies {context['graph']}, expected 1 (not tabled)")
        for r, exp in want.items():
            got = rows[r][(0, CPY)]
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
            check((0, 11) not in rows[r], f"step {r}: a graph-node copy's stream was tabled as a copy stream")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            got = tables(db, log, windows, joins)
        text = out.getvalue()
        check(got is not None, "tables refused a residency run log")
        check("residency mid-p40-s1" in text, "the tables do not name the residency word")
        check("engine stream is (0, 7)" in text, "the tables do not name the engine stream")
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
        check(got is None and "no `residency host` record" in text and "stream(s) (0, 9)" in text,
              "the lever-off line does not name the refusal and the non-engine stream")
        # The unpack hold: the whole-trace table, its values written by hand beside the synth.
        db2, log2 = os.path.join(d, "hold.sqlite"), os.path.join(d, "hold.txt")
        synth_hold().write(db2)
        with open(log2, "w") as f:
            f.write(RUNLOG_ON)
        tr2 = Trace(db2)
        eng2 = engine_streams(tr2)
        check(eng2 == [(0, ENG), (1, CPY)],
              f"engine streams {eng2}, the graph kernels' [(0, 7), (1, 9)]")
        check((1, CPY) in eng2 and (0, CPY) not in eng2,
              "the copy stream's id no longer collides with a tier-card engine stream's — keyed by "
              "streamId alone engine_streams returns {7, 9} and the checks above fail (the mutant)")
        hold = unpack_hold(tr2, eng2)
        check(hold is not None, "no unpack kernel found on the copy stream (the collision ate it)")
        check(hold["stage_dev"] == 0 and hold["missing"] == [], f"stage card {hold}")
        r = hold["names"]["ds41_r8_q3k"]
        check(r[0] == 1, f"plain unpack launches {r[0]}, expected 1")
        close(r[1], 260.0, "plain unpack median µs")
        close(r[2], 260.0, "plain unpack p90 µs")
        check(r[3] == (4950, 1, 1) and r[4] == (256, 1, 1), f"plain unpack grid/block {r[3]} {r[4]}")
        check(r[5] == 2 and r[6] == 2, f"engine starts during the unpack (med, max) {r[5]} {r[6]}, expected 2 2")
        close(r[7], 45.0, "first engine start after the unpack's end, median")
        close(r[8], 45.0, "first engine start after the unpack's end, p10")
        close(r[9], 45.0, "first engine start after the unpack's end, p90")
        g = hold["names"]["ds41_r8_q3k_groups"]
        check(g[0] == 1 and g[3] == (64, 1, 1), f"group unpack launches/grid {g[0]} {g[3]}")
        close(g[1], 100.0, "group unpack median µs")
        check(g[5] == 0 and g[6] == 0, f"engine starts during the group unpack {g[5]} {g[6]}, expected 0 0")
        close(g[7], 380.0, "first engine start after the group unpack's end, median")
        hy, hn = hold["split"][("heavy", True)], hold["split"][("heavy", False)]
        ly, ln = hold["split"][("light", True)], hold["split"][("light", False)]
        check(hy[0] == 1 and hn[0] == 1 and ln[0] == 1, f"split counts {hy[0]} {hn[0]} {ly[0]} {ln[0]}")
        close(hy[1], 310.0, "heavy overlapped median period µs")
        close(hy[2], 200.0, "heavy overlapped median longest gap µs")
        close(hy[3], 110.0, "heavy overlapped median engine busy µs")
        close(hn[1], 190.0, "heavy clean median period µs")
        close(hn[2], 100.0, "heavy clean median longest gap µs")
        close(hn[3], 90.0, "heavy clean median engine busy µs")
        close(ln[1], 210.0, "light clean median period µs")
        check(ly == [0, None, None, None], f"light overlapped row {ly}, expected the explicit empty")
        close(hold["hold_us"], 120.0, "unpack hold µs (heavy overlapped - clean)")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            tables(db2, log2, [], {})
        text = out.getvalue()
        check("unpack hold: overlapped - clean layer period = 120.0 µs (heavy steps)" in text,
              "the printed summary line is missing or wrong")
        check("ds41_r8_q3k_groups" in text and "4950x1x1|256x1x1" in text, "the per-name rows are missing")
        check("heavy, unpack in the gap" in text and "light, clean" in text, "the split rows are missing")
        # A trace with no handoff and no step-end kernel: names table yes, layer split by name.
        db3, log3 = os.path.join(d, "bare.sqlite"), os.path.join(d, "bare.txt")
        synth_bare().write(db3)
        with open(log3, "w") as f:
            f.write(RUNLOG_ON)
        tr3 = Trace(db3)
        hold3 = unpack_hold(tr3, engine_streams(tr3))
        check(hold3 is not None and hold3["stage_dev"] == 0, "the bare trace's unpack kernel was not found")
        b = hold3["names"]["ds41_r8_q3k"]
        check(b[0] == 1 and b[7] is None and b[8] is None and b[9] is None,
              f"the bare trace's after-stats {b[7:]} expected None (no engine kernel follows)")
        check(hold3["missing"] == [f"no {STEP_END_KERNEL} graph kernel on device 0: the layer split is skipped",
                                   f"no {HANDOFF_PREFIX}* graph kernel on device 0: the layer split is skipped"],
              f"the bare trace's missing lines {hold3['missing']}")
        check(hold3["split"] == {} and hold3["hold_us"] is None, "the bare trace printed a layer split")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            tables(db3, log3, [], {})
        check("ds41_r8_q3k" in out.getvalue() and "the layer split is skipped" in out.getvalue(),
              "the bare trace's printed table does not name the skip")
        # The same hold cut over the plain handoff name (plan (a)'s trace): the family reads both
        # forms, and the split is the tier trace's.
        db4, log4 = os.path.join(d, "plain.sqlite"), os.path.join(d, "plain.txt")
        synth_hold("ds41_ffn_handoff").write(db4)
        with open(log4, "w") as f:
            f.write(RUNLOG_ON)
        tr4 = Trace(db4)
        hold4 = unpack_hold(tr4, engine_streams(tr4))
        check(hold4 is not None and hold4["missing"] == [],
              f"the plain-handoff trace's layer split was skipped: {hold4['missing'] if hold4 else None}")
        if hold4 is not None and not hold4["missing"]:
            for key in (("heavy", True), ("heavy", False), ("light", False)):
                check(hold4["split"][key][0] == hold["split"][key][0],
                      f"the plain-handoff trace's {key} count {hold4['split'][key][0]}, "
                      f"the tier trace's {hold['split'][key][0]}")
            close(hold4["hold_us"], hold["hold_us"], "the plain-handoff trace's unpack hold µs")
        # The window edges: a straddling copy-stream kernel counts its launch once, an
        # engine-stream copy keeps the tail that runs past the window's end, and the H2D
        # split's classes stay disjoint when an eager kernel sits inside a join gap.
        db5 = os.path.join(d, "edges.sqlite")
        synth_edges().write(db5)
        tr5 = Trace(db5)
        rows5, ctx5 = window_rows(tr5, engine_streams(tr5),
                                  [(100, 1.00 * MS, 2.00 * MS), (101, 2.00 * MS, None)],
                                  {100: [(1.15 * MS, 1.60 * MS)]})
        e100, e101 = rows5[100][(0, CPY)], rows5[101][(0, CPY)]
        check(e100["kern"][0] == 1 and e101["kern"][0] == 0,
              f"a straddling kernel's launch count {e100['kern'][0]} + {e101['kern'][0]}, expected 1")
        check(e100["by_name"]["ds41_stage_thing"][0] == 1 and e101["by_name"]["ds41_stage_thing"][0] == 0,
              f"a straddling kernel's by-name count {e100['by_name']} / {e101['by_name']}, expected 1 launch")
        close(e100["kern"][1] + e101["kern"][1], 60.0, "the straddling kernel's clipped time sums to its whole")
        check(ctx5["kind"]["H2D"] == [1, 4096, 200.0],
              f"an engine-stream copy that runs past the window's end: {ctx5['kind']['H2D']}, "
              f"expected [1, 4096, 200.0] (its whole duration)")
        h = e100["h2d"]
        check(h[0] == 1 and h[1] == 1024, f"the edges trace's H2D count/bytes {h[:2]}")
        close(h[2], 300.0, "the edges trace's H2D busy")
        close(h[3], 300.0, "the edges trace's H2D in gap (an eager kernel inside the gap is gap time)")
        close(h[4], 0.0, "the edges trace's H2D in kern (none outside the gap)")
        close(h[5], 0.0, "the edges trace's H2D rest (never negative)")
        check(h[5] >= -0.001, f"the H2D rest is negative: {h[5]}")
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
