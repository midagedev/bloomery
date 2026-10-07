#!/usr/bin/env python3
"""Expert residency replayed over router sets: which card set serves how many routed slots.

    tools/ref/router-residency.py hit <family> [<set>...] [--cap plan|n<N>|<f>] [--seed <seed>]
                                  [--rule mid|strata|knee] [--every N] [--link-gbps G]
                                  [--policies static,adaptive,lru,belady] [--json PATH]
    tools/ref/router-residency.py hit <family> --streams S1,S2[,...] [--offset K] [--turn K] [--cap ...]
                                  [--seed <seed>] [--rule ...] [--every N] [--link-gbps G]
                                  [--policies static,adaptive,belady] [--json PATH]
    tools/ref/router-residency.py hit <family> [<set>...] --window N [--prompt P] [--open M|all
                                  [--stage K] [--open-from all|last:N]] [--seed <seed>] [--d D]
                                  [--copies K] [--json PATH]
    tools/ref/router-residency.py gen <family> <trace> [--prompt P] [--window N] [--short skip] [--split held|learn]
                                  [--seed <seed>]
                                  [--open M|all] [--stage K] [--open-from all|last:N] [--d D]
                                  [--copies K] [--json PATH]
    tools/ref/router-residency.py fixture --out PATH [--passes N] [--lcg SEED]
    tools/ref/router-residency.py --self-test

Every command takes --data DIR: the directory the router sets live in. It has no default; a set named
without --data is refused by name. A set is a router_trace directory, read through tools/ref/router-coverage.py
(read_manifest over tools/bloomery/manifest.py, read_topk), which refuses a set without its
`# complete` trailer. A <set> is a name under --data or a directory path.

One token of a set is one decode step (a pass of one row), on the family's eligible layers only
(FAMILY below). A hit is a routed slot whose expert is on the card when the pass starts. Sets are
split by position: seeds learn on [0, SPLIT) and every replay evaluates on [SPLIT, tokens).

Seeds (per layer, a ranked list; the card holds its first n_l ids):
    in        the set's own [0, SPLIT)            insample  the set's own [SPLIT, tokens)
    cross     the family's other sets, whole      pooled    every family set's [0, SPLIT)
    prefix    ids 0, 1, 2, ...
Ranks are hottest first, ties to the lower id.

hit      Without --window, the continuous replay over the eval half, one markdown row per policy:
           static    the seed's card set, never changed
           adaptive  the swap rule (below; label = the rule's name), Strata's evict-then-admit: the
                     victim leaves at the decision, the admitted expert serves from the first pass
                     after its copy over a serialized link of --link-gbps GB/s (0 = no link) ends
           lru       one capacity sum(n_l) shared by the layers; fill=all admits every miss,
                     fill=budget at most step_ms x pin GB/s of expert bytes a token
           belady    per layer, MIN with bypass and free swaps: the ceiling for a demand policy
         and the prices at --pin-gbps / --page-gbps. Columns as ideaverify's v1 table.
         With --streams S1,S2,... (B of them, 1..8, each a set of the family's table), the interleaved
         replay: B sessions share one card, pass t carries token t of every stream (B rows a pass), and
         the rule counts every row of the pass under the same decay. A set named once is its whole eval
         half; a set named m > 1 times is m windows of it, copy j at eval positions [j K, (j + 1) K) with
         K = --offset (by default floor(eval / m): disjoint, adjacent windows, the "similar topic" arm);
         --offset 0 makes every copy the whole eval half (identical streams, each pass counted m times);
         m K past the eval half is refused, and so is --offset with no set named twice. Streams of
         unequal lengths are truncated to the shortest, and a `truncated` line says so: refusing would
         rule out every arm that mixes a windowed set with a whole one, and only passes that carry all B
         rows measure the merged count. `--turn K` takes the card in turns instead: stream 0 runs K
         passes (one token each), then stream 1 K, round robin until every window is consumed — every
         pass one row of one stream, which the rule's cadence counts (K 1 is token interleave, K 64
         today's time-slicing; B = 1 is the plain hit), the rows carry turn=K, and a `tok` column is per
         one-row pass. Seeds: in = the streams' learn halves summed, insample = their
         eval windows summed, cross = the family's sets that are no stream (refused when there is none),
         pooled and prefix as above. Rows: `pooled` (all B rows), `sK` (stream K inside the merged
         replay), and per stream `alone sK` (that stream replayed by itself from its own seed, B = 1),
         then a `merged` line per arm: pooled minus the alone mean, and the host cost of the difference
         a pass at host_gbps [derived]. `tok` columns are per pass here; host GB a pass counts missed
         slots, and `unique` counts each missed (layer, expert) of a pass once (the host reads it once).
         belady has no per-stream split: its pooled row is MIN over the interleaved rows in order (it
         may change the set between the rows of a pass, so it is above any pass-granular policy), and
         its alone rows are each stream's own ceiling. lru is refused here: it has no rows-a-pass form.
         B = 1 prints the plain hit's rows, every column, with the streams' columns beside them.
         With --window N, the timed-window replay: the eval half cut into requests of P prompt then
         N decode tokens; each request starts from the seed after a reset with counts at zero and
         runs the rule with admit-then-flip (--spares S slots a layer, 1 by default; the victim stays
         live until its replacement is), live at max(boundary + d, the step a budget of --copies copies
         a step reaches it); flips live at a boundary land before it plans (--spares-per-pass: S new
         flips a boundary, in flight not counted, planned before the landing, as window4). --open M
         adds the opening reshuffle before decode step 0: per layer the prompt's most-used
         non-resident experts against the least-used residents (whole-prompt counts, ties to the
         seed's rank), kept while in > out, the M largest gains over all layers; --open-from last:N
         pairs by the counts of the prompt's last N positions only (the streaming design's opening:
         the prompt call's last group leaves its pick on the card), `all` (the default) by the whole
         prompt's; --stage 0 copies them inside the prompt call (live at step 0), --stage K from step 0
         at K a step. The rule then counts from the whole prompt's decayed counts either way. Prints the static arm and the
         asked arm: hit, the mean of six 16-step blocks, swaps a token, opening swaps a request.
gen      The router-gen replay: <trace> holds prompt + generation in one context per manifest
         `chunk`; positions [0, P) of each context are the prompt (--prompt P). A route trace with a
         contexts.tsv (crates/gpu/src/host/route_trace.rs, read by tools/bloomery/route_trace.py)
         gives each request's context and its own P (the `prompt` column) instead, and --prompt is
         refused beside it. A context shorter than P + N is refused by name; with --short skip it gets
         no window arms (a `skipped K of M` line lists them) and still counts in the steady rule.
         --split held|learn scores only that split's requests of a contexts.tsv (the steady rule still
         walks every position); without it a `split:` line says both are scored, and --split on a
         trace with no contexts.tsv is refused.
         Over the window [P, P + N) of each context (N = --window, 96):
           (a) static    the seed's card set
           (b) adaptive  the rule from zero counts, reset to the seed at P
           (c) open      the opening reshuffle over [0, P) (M = --open, the family's default; by the
                         last N of it under --open-from last:N), then the rule
         plus (a) over the whole generation [P, end) and the steady rule (continuous from the seed
         over every position of the trace, hit over [P, end) of each context), then the verdict
         line of adaptres §6: (c)-(a) >= 10 points -> R5 with R3; < 3 -> drop R5; steady < 70 % ->
         hold R3; 3 <= (c)-(a) < 10 has no rule there and says so.
fixture  A synthetic trace (a 32-bit LCG, so the file is the same on every numpy) and the rule's
         flips over it, as JSON: header (command, the boundary and order contract, tie counts),
         params, seed, trace (layers x passes x rows x ids), kept (rows counted per pass), flips
         (boundary, layer, out, in, live_at), and cap_case: the same seed and trace at cap 2 and
         spares 2, its header counting the boundaries the cap bound (refused at 0). Passes carry 1-3
         rows and count only the first kept. --pinned P: every layer's first P seed ids are never a
         victim (params.pinned), and the file is refused unless pinning moves some flip. --away A:
         each layer's ids of A are on another device, never admitted and never a victim
         (params.away); A is a JSON file holding a list of id lists, one a layer, or that list
         written inline as ids comma-separated and layers '/'-separated; an away id in its layer's
         seed is refused, and so is a file in which no flip moves.

The rule (every, cap, margin, min_count, decay, clock): every `every` kept rows (clock `kept`, the
clock the engine runs: a boundary plans when the ended passes have kept `every` rows past the last
plan's, the rows past `every` carry, and a boundary plans once at most) or at every `every`-th
boundary (clock `pass`), per layer the missing experts by decayed count (descending, ties to the
lower id) pair rank for rank with the residents by
count (ascending, ties to the lower id) while count_in >= min_count and count_in >= count_out +
margin; the pairs go by gain descending, then layer ascending, then in id ascending, at most `cap`
over all layers (and under flip at most --spares in flight a layer); then every count decays by
`decay`. The copies live at a boundary land before it plans (crates/runtime's swaprule). Presets:
    mid     every 4, cap 24, margin 3, min_count 2, decay 0.9    (the adopted setting; clock kept)
    strata  every 4, cap 96, margin 1.5, min_count 2, decay 0.7  (Strata's default; clock pass, its
            source plans every 4 verify windows)
    knee    every 16, cap 24, margin 6, min_count 2, decay 0.9   (clock kept)

Needs numpy (the Mac's python3 has it; check-recipes runs the self-test there).
"""
import argparse
import hashlib
import heapq
import importlib.util
import io
import json
import math
import os
import sys
import tempfile
from collections import OrderedDict
from contextlib import redirect_stderr, redirect_stdout

try:
    import numpy as np
except ImportError:
    sys.exit("router-residency: needs numpy (the replays are numpy array code): python3 -m pip install numpy")

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("router_coverage", os.path.join(_here, "router-coverage.py"))
rc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rc)
_spec = importlib.util.spec_from_file_location("window_union", os.path.join(_here, "window-union.py"))
wu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wu)
_spec = importlib.util.spec_from_file_location("route_trace", os.path.join(_here, "..", "bloomery", "route_trace.py"))
rt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rt)

SPLIT = 24576  # 12 whole chunks of 2048: learn on [0, SPLIT), evaluate on [SPLIT, tokens)

# Model facts the replays price with. plan_total / plan_n: the card experts of V4.1 plan (a) (spread over
# the eligible layers), GLM glmcard and the Qwen3.8 card plan; step_ms: the decode step the link is timed
# against [derived: V4.1 from 39.6 tok/s, GLM 40 ms, q38 53.85 tok/s at depth 512 —
# rig-log log/2026-09-29.md#q38prose-pp]; open_m: gen's default opening cap (V4.1's victims come back from
# NVMe, GLM's and q38's whole routed sets are in host memory).
# q38's eligible layers are the ones whose routed stacks a card expert kernel reads (`card_routed`: the
# Q4_K gate·up and the Q5_1 down) — every layer but the host-only 2, 4, 30, 46, 47, pinned in
# crates/model/tests/qwen4exp_meta.rs (HOST_ONLY); plan_total is the A6000 card plan's 12,841 experts at
# ctx 4,096 (299 on 27 layers, 298 on 16 — the same spread `placement.rs` makes); expert_bytes
# 3,072,000 = gate 921,600 + up 921,600 + down 1,228,800. Its engine traces are the sets
# `generate_qwen3moe --prefill step` writes under BLOOMERY_ROUTE_TRACE (all 48 layers, `call` rows per
# prompt, a `chunk` header line per arm); no corpus set exists yet, so `sets` is empty and a set is always
# named.
FAMILY = {
    "v41": dict(sets=("prose", "code", "korean"), n_expert=384, n_used=6, expert_bytes=16_773_120,
                eligible=list(range(2, 40)), plan_total=2668, step_ms=25.3, seeds=("cross", "in"),
                open_m=400),
    "glm": dict(sets=("glm5next-prose",), n_expert=288, n_used=8, expert_bytes=15_204_352,
                eligible=list(range(3, 11)) + list(range(13, 44)), plan_n=67, step_ms=40.0,
                seeds=("in", "prefix"), open_m=None),
    "q38": dict(sets=(), n_expert=512, n_used=10, expert_bytes=3_072_000,
                eligible=[l for l in range(48) if l not in (2, 4, 30, 46, 47)], plan_total=12_841,
                step_ms=18.6, seeds=("in", "prefix"), open_m=None),
}
RULES = {
    "mid": dict(every=4, cap=24, margin=3.0, min_count=2.0, decay=0.9),
    "strata": dict(every=4, cap=96, margin=1.5, min_count=2.0, decay=0.7, clock="pass"),
    "knee": dict(every=16, cap=24, margin=6.0, min_count=2.0, decay=0.9),
}
B_PIN, B_PAGE = 25.0, 14.4  # GB/s: the rates the recorded replays priced with (A6000: 26.28 / 21.16)
HOST_GBPS = 121.0  # GB/s: the host expert read rate b5prof measured (docs/plan-ledger.md B12), prices a host miss
MAX_STREAMS = 8  # hit --streams: the most sessions one card is replayed for
COPIES = 30  # the window replays' copy budget, copies a decode step
WINDOW_B = 16  # the block the window replays average over


class ToolError(Exception):
    pass


class Rule:
    def __init__(self, name, **over):
        if name not in RULES:
            raise ToolError(f"no rule {name!r}: the presets are {', '.join(RULES)}")
        p = dict(RULES[name])
        p.update({k: v for k, v in over.items() if v is not None})
        self.name = name
        self.every, self.cap, self.margin = int(p["every"]), int(p["cap"]), float(p["margin"])
        self.min_count, self.decay = float(p["min_count"]), float(p["decay"])
        self.clock = p.get("clock") or "kept"
        if self.every < 1 or self.cap < 0:
            raise ToolError(f"rule {name}: every {self.every} and cap {self.cap} must be >= 1 and >= 0")
        if self.clock not in ("kept", "pass"):
            raise ToolError(f"rule {name}: clock {self.clock!r} is not 'kept' or 'pass'")

    def params(self):
        return dict(every=self.every, clock=self.clock, cap=self.cap, margin=self.margin,
                    min_count=self.min_count, decay=self.decay)

    def text(self):
        return (f"{self.name} (every {self.every} {'kept rows' if self.clock == 'kept' else 'passes'}, "
                f"cap {self.cap}, margin {self.margin:g}, "
                f"min_count {self.min_count:g}, decay {self.decay:g})")


# --- sets and seeds -------------------------------------------------------------------------------


def family(name):
    if name not in FAMILY:
        raise ToolError(f"family {name!r} is not in the table: {', '.join(FAMILY)}")
    return FAMILY[name]


def set_dir(data, name):
    if os.path.isdir(name) and os.path.isfile(os.path.join(name, "MANIFEST.tsv")):
        return name
    if data is None:
        raise ToolError(f"router set {name!r} is not a set directory, and no --data DIR names where sets live")
    d = os.path.join(data, name)
    if not os.path.isdir(d):
        raise ToolError(f"no router set {name!r}: {d} is not a directory (--data {data})")
    return d


class Set:
    def __init__(self, path, fam=None):
        self.meta = rc.read_manifest(path)
        self.dir = path
        self.name = os.path.basename(os.path.normpath(path))
        self.T = self.meta["tokens"]
        self.E = self.meta["n_expert"]
        self.K = self.meta["n_used"]
        self.layers = self.meta["layers"]
        chunk = self.meta["header"].get("chunk")
        self.chunk = int(chunk) if chunk else None
        self._ids = {}
        if fam is not None:
            F = family(fam)
            if (self.E, self.K) != (F["n_expert"], F["n_used"]):
                raise ToolError(f"{path}: {self.E} experts x top-{self.K}, family {fam} is "
                                f"{F['n_expert']} x top-{F['n_used']}")
            missing = [l for l in F["eligible"] if l not in self.layers]
            if missing:
                raise ToolError(f"{path}: family {fam}'s eligible layers {missing} are not in the set")

    def ids(self, layer):
        if layer not in self._ids:
            if layer not in self.layers:
                raise ToolError(f"{self.dir}: no layer {layer}")
            a = np.frombuffer(rc.read_topk(self.meta, layer), dtype=np.uint16).astype(np.int32)
            a = a.reshape(self.T, self.K)
            if a.size and a.max() >= self.E:
                raise ToolError(f"{self.dir} layer {layer}: id {a.max()} >= n_expert {self.E}")
            for row in a[:: max(1, self.T // 997)]:
                if len(set(row.tolist())) != self.K:
                    raise ToolError(f"{self.dir} layer {layer}: a token selects an expert twice")
            self._ids[layer] = a
        return self._ids[layer]

    def stack(self, layers, t0=0, t1=None):
        """[T', L, K] ids of the given layers over tokens [t0, t1)."""
        t1 = self.T if t1 is None else t1
        return np.stack([self.ids(l)[t0:t1] for l in layers], axis=1)

    def counts(self, layer, t0=0, t1=None):
        t1 = self.T if t1 is None else t1
        return np.bincount(self.ids(layer)[t0:t1].ravel(), minlength=self.E)


def hit_eval_stack(s, F):
    """hit's eval half [SPLIT, tokens) of the family's eligible layers, refused by name for a set that
    ends at or before SPLIT — an engine decode trace is that short; gen replays those by their contexts."""
    if s.T <= SPLIT:
        raise ToolError(f"{s.dir}: {s.T} tokens; hit evaluates [{SPLIT}, tokens), which it holds none "
                        "of — gen replays a trace this short by its contexts")
    return s.stack(F["eligible"], SPLIT, s.T)


def rank(counts):
    """Hottest first, ties to the lower id (router-coverage.py's `hot`)."""
    return np.lexsort((np.arange(len(counts)), -np.asarray(counts)))


def spread(total, eligible):
    """crates/model/src/placement.rs `spread`: one more expert per eligible layer in order, cycling."""
    n = {l: 0 for l in eligible}
    left = total
    while left > 0:
        for l in eligible:
            if left == 0:
                break
            n[l] += 1
            left -= 1
    return n


def n_plan(fam):
    F = family(fam)
    if "plan_total" in F:
        n = spread(F["plan_total"], F["eligible"])
        return [n[l] for l in F["eligible"]]
    return [F["plan_n"]] * len(F["eligible"])


def n_cap(fam, label, E):
    L = len(family(fam)["eligible"])
    if label == "plan":
        return n_plan(fam)
    try:
        n = int(label[1:]) if label.startswith("n") else int(round(float(label) * E))
    except ValueError:
        raise ToolError(f"--cap {label!r}: plan, n<count> or a fraction of n_expert") from None
    if not 0 <= n <= E:
        raise ToolError(f"--cap {label!r}: {n} is outside 0..{E}")
    return [n] * L


def seed_lists(fam, s, name, data):
    """Per eligible layer, the ranked list of seed `name` for set s (evaluated on [SPLIT, T))."""
    F = family(fam)
    lay = F["eligible"]
    if name == "in":
        return [rank(s.counts(l, 0, SPLIT)) for l in lay]
    if name == "insample":
        return [rank(s.counts(l, SPLIT, s.T)) for l in lay]
    if name == "prefix":
        return [np.arange(s.E) for _ in lay]
    if name in ("cross", "pooled"):
        names = [n for n in F["sets"] if name == "pooled" or n != s.name]
        if not names:
            raise ToolError(f"seed cross: family {fam} has no set other than {s.name}")
        others = [s if n == s.name else Set(set_dir(data, n), fam) for n in names]
        if name == "cross":
            return [rank(sum(o.counts(l) for o in others)) for l in lay]
        return [rank(sum(o.counts(l, 0, SPLIT) for o in others)) for l in lay]
    raise ToolError(f"no seed {name!r}: in, insample, cross, pooled or prefix")


def seed_resident(lists, n_l, E):
    res = np.zeros((len(lists), E), dtype=bool)
    for i, ids in enumerate(lists):
        take = np.asarray(ids[: n_l[i]], dtype=np.int64)
        if len(take) < n_l[i]:
            raise ToolError(f"seed layer {i}: {len(take)} ids for n_l {n_l[i]}")
        res[i, take] = True
    return res


# --- policies -------------------------------------------------------------------------------------


def static_hits(X, lists, n_l):
    """X [T, L, K]; lists[i] ranked ids for layer i; n_l[i] capacity. Returns hits per layer."""
    T, L, K = X.shape
    hits = np.zeros(L, dtype=np.int64)
    for i in range(L):
        E = max(int(X[:, i].max()) + 1, len(lists[i]))
        res = np.zeros(E, dtype=bool)
        res[np.asarray(lists[i][: n_l[i]], dtype=np.int64)] = True
        hits[i] = res[X[:, i]].sum()
    return hits


class Link:
    """A serialized copy stream at `gbps`, timed in ms against a decode step of step_ms: ready(b) is the
    first pass a copy issued at boundary b (after pass b - 1) serves."""

    def __init__(self, bytes_per_copy, gbps, step_ms):
        self.copy_ms = bytes_per_copy / (gbps * 1e6) if gbps else 0.0
        self.step_ms = step_ms
        self.free_ms = 0.0
        self.busy_ms = 0.0
        self.delay_steps = 0

    def ready(self, b):
        now_ms, now_step = b * self.step_ms, b - 1
        if self.copy_ms == 0.0:
            return now_step + 1
        start = max(self.free_ms, now_ms)
        self.free_ms = start + self.copy_ms
        self.busy_ms += self.copy_ms
        ready = max(now_step + 1, math.ceil(self.free_ms / self.step_ms - 1e-9))
        self.delay_steps += ready - now_step
        return ready


class Budget:
    """K copies a pass in FIFO order (None: no budget); ready(b) is the pass the copy has landed by."""

    def __init__(self, k):
        self.k = k
        self.t = 0.0

    def ready(self, b):
        if self.k is None:
            return b
        start = max(self.t, float(b))
        self.t = start + 1.0 / self.k
        return math.ceil(self.t - 1e-9)


def pass_swaps(counts, resident, pend_in, pend_out, n_l, rule, per_layer_cap, ties=None, in_flight=None,
               pinned=None, away=None):
    """The rule's pairs (layer, in, out, gain) at one boundary, in the order they are issued. A layer takes
    at most per_layer_cap, less its flips still in flight (in_flight: layer -> count) when given. pinned
    [L, E] bool: ids never a victim. away [L, E] bool: ids never admitted (they are never resident, so
    never a victim either)."""
    L, E = counts.shape
    taken = resident | pend_in if away is None else resident | pend_in | away
    cand = np.where(taken, -1.0, counts)
    keep = resident & ~pend_out if pinned is None else resident & ~pend_out & ~pinned
    vict = np.where(keep, counts, np.inf)
    m = min(max(n_l), E)
    cidx = np.argsort(-cand, axis=1, kind="stable")[:, :m]
    vidx = np.argsort(vict, axis=1, kind="stable")[:, :m]
    cval = np.take_along_axis(cand, cidx, axis=1)
    vval = np.take_along_axis(vict, vidx, axis=1)
    ok = (cval >= rule.min_count) & (cval >= vval + rule.margin) & np.isfinite(vval)
    li, jj = np.nonzero(ok)
    if not li.size:
        return []
    gain = cval[li, jj] - vval[li, jj]
    order = np.argsort(-gain, kind="stable")
    out, used, blocked = [], dict(in_flight or {}), set()
    for o in order:
        i = int(li[o])
        if per_layer_cap is not None and used.get(i, 0) >= per_layer_cap:
            if (in_flight or {}).get(i, 0) >= per_layer_cap:
                blocked.add(i)
            continue
        if len(out) >= rule.cap:
            if ties is not None:
                ties["cap_bound"] = ties.get("cap_bound", 0) + 1  # the cap cut a pair the spares let through
            break
        used[i] = used.get(i, 0) + 1
        out.append((i, int(cidx[i, jj[o]]), int(vidx[i, jj[o]]), float(gain[o])))
        if ties is not None:
            ties["in_id"] += int((cand[i] == cval[i, jj[o]]).sum() > 1)
            ties["out_id"] += int((vict[i] == vval[i, jj[o]]).sum() > 1)
    if ties is not None:
        ties["blocked_in_flight"] = ties.get("blocked_in_flight", 0) + len(blocked)
        gains = [(g, i) for i, _, _, g in out]
        ties["across_layers"] += sum(1 for g, i in gains if any(g2 == g and i2 != i for g2, i2 in gains))
    return out


class Replay:
    def __init__(self, hits_layer, per_row, swaps, plans):
        self.hits_layer, self.per_row, self.swaps, self.plans = hits_layer, per_row, swaps, plans


def replay(X, resident0, n_l, E, rule=None, *, sem="flip", spares=1, in_flight_cap=True, d=1, link=None,
           counts0=None, record=None, passes=None, on_flip=None, ties=None, pinned=None, away=None):
    """The card set over X [rows, L, K]. passes: None (one row a pass, counted) or (r0, r1, kept) per pass;
    every row of a pass is served by the set in force at its start, its first `kept` rows are counted.
    rule None is the static set. sem "hole": the victim leaves at the decision; "flip": it stays live
    until the admitted expert is: a layer has at most `spares` flips planned and not yet live. A flip
    decided at boundary b (after pass b - 1) is live from pass max(b + d, link.ready(b)); at a boundary
    the flips live there land first, then the rule plans, so one live at b is no longer in flight when b
    plans. in_flight_cap False is the adaptres window scripts' order: at most `spares` new flips a layer
    a boundary however many are in flight, and the rule plans before the flips live at b land. pinned
    [L, E] bool: residents never a victim (crates/runtime's pinned seed ranks). away [L, E] bool: ids on
    another device, never admitted (crates/runtime's away experts)."""
    L = X.shape[1]
    ar = np.arange(L)[:, None]
    resident = resident0.copy()
    counts = np.zeros((L, E)) if counts0 is None else counts0.copy()
    pend_in = np.zeros((L, E), dtype=bool)
    pend_out = np.zeros((L, E), dtype=bool)
    heap = []
    seq = swaps = plans = 0
    since = 0  # kept rows the ended passes have kept past the last plan's (clock kept)
    hits = np.zeros(L, dtype=np.int64)
    per_row = np.zeros(X.shape[0])
    per_layer_cap = spares if sem == "flip" else None
    flying = {} if sem == "flip" and in_flight_cap else None

    def land(b):
        while heap and heap[0][0] <= b:
            _, _, i, e_in, e_out = heapq.heappop(heap)
            pend_in[i, e_in] = False
            resident[i, e_in] = True
            if sem == "flip":
                pend_out[i, e_out] = False
                resident[i, e_out] = False
            if flying is not None:
                flying[i] -= 1

    npass = X.shape[0] if passes is None else len(passes)
    for p in range(npass):
        land(p)
        r0, r1, kept = (p, p + 1, 1) if passes is None else passes[p]
        for r in range(r0, r1):
            sel = X[r]
            hr = resident[ar, sel]
            if record is not None:
                record[r] = hr
            hits += hr.sum(axis=1)
            per_row[r] = hr.mean()
            if rule is not None and r < r0 + kept:
                counts[ar, sel] += 1.0
        if rule is None:
            continue
        b = p + 1
        if rule.clock == "pass":
            due = b % rule.every == 0
        else:
            since += kept
            due = since >= rule.every
            if due:
                # the rows past `every` carry to the next plan; a boundary plans once at most
                since %= rule.every
        if not due:
            continue
        plans += 1
        if in_flight_cap:
            land(b)
        for i, e_in, e_out, g in pass_swaps(counts, resident, pend_in, pend_out, n_l, rule, per_layer_cap, ties,
                                            {k: v for k, v in flying.items() if v} if flying is not None else None,
                                            pinned, away):
            live = max(b + d, link.ready(b) if link is not None else b)
            pend_in[i, e_in] = True
            if sem == "hole":
                resident[i, e_out] = False
            else:
                pend_out[i, e_out] = True
            heapq.heappush(heap, (live, seq, i, e_in, e_out))
            if flying is not None:
                flying[i] = flying.get(i, 0) + 1
            seq += 1
            swaps += 1
            if on_flip is not None:
                on_flip(b, i, e_out, e_in, live)
        counts *= rule.decay
    return Replay(hits, per_row, swaps, plans)


def adaptive(X, seed, n_l, E, rule, link=None, record=None):
    """The continuous replay (ideaverify V1): evict-then-admit, live when the copy has landed."""
    r = replay(X, seed_resident(seed, n_l, E), n_l, E, rule, sem="hole", d=0, link=link, record=record)
    return r.hits_layer, r.swaps


def global_lru(X, seed, n_l, *, fill="all", budget=None):
    """FreeToken's global LRU: one capacity sum(n_l) shared by the layers. fill="all" admits every
    miss (it then serves the next access); fill="budget" admits at most `budget` misses a token, in
    layer order. Returns (hits per layer, fills)."""
    T, L, K = X.shape
    cache = OrderedDict()
    for i in range(L):
        for e in reversed(list(seed[i][: n_l[i]])):
            cache[(i, int(e))] = None
    for key in list(cache)[::-1]:
        cache.move_to_end(key, last=False)
    C = sum(n_l)
    hits = np.zeros(L, dtype=np.int64)
    fills = 0
    Xl = X.tolist()
    for s in range(T):
        left = budget if fill == "budget" else None
        row = Xl[s]
        for i in range(L):
            for e in row[i]:
                key = (i, e)
                if key in cache:
                    hits[i] += 1
                    cache.move_to_end(key)
                elif left is None or left > 0:
                    cache[key] = None
                    fills += 1
                    if left is not None:
                        left -= 1
                    if len(cache) > C:
                        cache.popitem(last=False)
    return hits, fills


def belady(X, seed, n_l):
    """Per layer, Belady's MIN with bypass and free swaps: the ceiling for any demand policy at n_l."""
    T, L, K = X.shape
    hits = np.zeros(L, dtype=np.int64)
    admits = 0
    for i in range(L):
        cap = n_l[i]
        seq = X[:, i, :]
        nxt = np.full((T, K), T, dtype=np.int64)  # next use (step) of the expert selected at (s, k)
        last = {}
        for s in range(T - 1, -1, -1):
            for k in range(K):
                e = int(seq[s, k])
                nxt[s, k] = last.get(e, T)
            for k in range(K):
                last[int(seq[s, k])] = s
        first = {}
        for s in range(T - 1, -1, -1):
            for k in range(K):
                first[int(seq[s, k])] = s
        res = {}
        heap = []
        for e in seed[i][:cap]:
            e = int(e)
            nu = first.get(e, T)
            res[e] = nu
            heapq.heappush(heap, (-nu, e))
        seql = seq.tolist()
        nxtl = nxt.tolist()
        for s in range(T):
            row, nrow = seql[s], nxtl[s]
            missed = []
            for k in range(K):
                e = row[k]
                if e in res:
                    hits[i] += 1
                    res[e] = nrow[k]
                    heapq.heappush(heap, (-nrow[k], e))
                else:
                    missed.append((nrow[k], e))
            for nu, e in missed:
                if cap == 0:
                    continue
                if len(res) < cap:
                    res[e] = nu
                    heapq.heappush(heap, (-nu, e))
                    admits += 1
                    continue
                while True:
                    fnu, fe = heap[0]
                    if fe in res and res[fe] == -fnu:
                        break
                    heapq.heappop(heap)
                if -fnu > nu:
                    heapq.heappop(heap)
                    del res[fe]
                    res[e] = nu
                    heapq.heappush(heap, (-nu, e))
                    admits += 1
    return hits, admits


# --- the timed window and the opening reshuffle ---------------------------------------------------


def opening(Xp, seed, n_l, E, M, rule, last=None):
    """The opening reshuffle over prompt Xp: (seed card set, the prompt's decayed counts, pairs
    (gain, layer, in, out) largest gain first, at most M). last N: the pairs rank and gain by the counts of
    the prompt's last N positions; the decayed counts stay the whole prompt's."""
    L = Xp.shape[1]
    ar = np.arange(L)[:, None]
    resident = np.zeros((L, E), dtype=bool)
    prior = np.zeros((L, E))
    for i in range(L):
        idx = np.asarray(seed[i], dtype=np.int64)
        resident[i, idx[: n_l[i]]] = True
        prior[i, idx] = (E - np.arange(len(idx))) / (E + 1.0)  # < 1: breaks count ties by seed rank
    raw = np.zeros((L, E))
    dec = np.zeros((L, E))
    if last is not None and not 1 <= last <= Xp.shape[0]:
        raise ToolError(f"--open-from last:{last} on a prompt of {Xp.shape[0]} positions")
    first = 0 if last is None else Xp.shape[0] - last
    for t in range(Xp.shape[0]):
        if t >= first:
            raw[ar, Xp[t]] += 1.0
        dec[ar, Xp[t]] += 1.0
        if (t + 1) % rule.every == 0:
            dec *= rule.decay
    sc = raw + prior
    pairs = []
    for i in range(L):
        cin = [e for e in np.argsort(-np.where(resident[i], -np.inf, sc[i]), kind="stable") if not resident[i, e]]
        cout = [e for e in np.argsort(np.where(resident[i], sc[i], np.inf), kind="stable") if resident[i, e]]
        for a, b in zip(cin, cout):
            if raw[i, a] <= raw[i, b]:
                break
            pairs.append((raw[i, a] - raw[i, b], i, a, b))
    pairs.sort(key=lambda p: -p[0])
    return resident, dec, pairs[:M] if M is not None else pairs


def window_arm(Xp, Xd, seed, n_l, E, rule, *, arm, M=None, stage=0, d=1, copies=COPIES, spares=1, per_pass=False,
               last=None):
    """One timed request: (hits per decode step, rule swaps, opening swaps). arm: static, zero (the rule
    from zero counts) or open (the opening reshuffle, then the rule from the prompt's decayed counts)."""
    res0 = seed_resident(seed, n_l, E)
    kw = dict(sem="flip", spares=spares, in_flight_cap=not per_pass, d=d)
    if arm == "static":
        return replay(Xd, res0, n_l, E).per_row, 0, 0
    if arm == "zero":
        r = replay(Xd, res0, n_l, E, rule, link=Budget(copies), **kw)
        return r.per_row, r.swaps, 0
    if arm != "open":
        raise ToolError(f"no window arm {arm!r}")
    resident, dec, pairs = opening(Xp, seed, n_l, E, M, rule, last)
    L = Xd.shape[1]
    ar = np.arange(L)[:, None]
    res = resident.copy()
    if stage == 0:
        for _, i, a, b in pairs:
            res[i, a] = True
            res[i, b] = False
        r = replay(Xd, res, n_l, E, rule, link=Budget(copies), counts0=dec, **kw)
        return r.per_row, r.swaps, len(pairs)
    live = {}
    for j, p in enumerate(pairs):
        live.setdefault(math.ceil((j + 1) / stage), []).append(p)
    start = max(live) if live else 0
    hits = np.zeros(Xd.shape[0])
    counts = dec.copy()
    for s in range(min(start, Xd.shape[0])):
        for _, i, a, b in live.get(s, []):
            res[i, a] = True
            res[i, b] = False
        sel = Xd[s]
        hits[s] = res[ar, sel].mean()
        counts[ar, sel] += 1.0
        if (s + 1) % rule.every == 0:
            counts *= rule.decay
    for _, i, a, b in live.get(start, []):
        res[i, a] = True
        res[i, b] = False
    sw = 0
    if start < Xd.shape[0]:
        r = replay(Xd[start:], res, n_l, E, rule, link=Budget(copies), counts0=counts, **kw)
        hits[start:] = r.per_row
        sw = r.swaps
    return hits, sw, len(pairs)


def window_requests(X, P, N, lists, n_l, E, rule, arm, M=None, stage=0, d=1, copies=COPIES, spares=1, per_pass=False,
                    last=None):
    """X [T, L, K] cut into requests of P prompt then N decode tokens, each from the seed: the mean hit per
    decode step over the requests, its 16-step blocks, rule swaps a token, opening swaps a request."""
    R = P + N
    nreq = X.shape[0] // R
    if nreq == 0:
        raise ToolError(f"{X.shape[0]} tokens to replay, one request is {R}")
    per = np.zeros(N)
    sw = op = 0
    for r in range(nreq):
        h, a, b = window_arm(X[r * R: r * R + P], X[r * R + P: (r + 1) * R], lists, n_l, E, rule,
                             arm=arm, M=M, stage=stage, d=d, copies=copies, spares=spares, per_pass=per_pass,
                             last=last)
        per += h
        sw += a
        op += b
    per /= nreq
    return dict(nreq=nreq, hit=float(per.mean()), b16=[float(per[i:i + WINDOW_B].mean()) for i in range(0, N, WINDOW_B)],
                swaps_tok=sw / (nreq * N), open=op / nreq)


# --- gen: prompt + generation in one context --------------------------------------------------------


def gen_values(X, contexts, P, N, seed, n_l, E, rule, M, stage, d, copies, spares=1, per_pass=False,
               skipped=None, last=None, scored=None):
    """Per context (start, end) of X [T, L, K]: (a), (b), (c) over [start + P, start + P + N), (a) over
    [start + P, end); and the steady rule's hit over every [start + P, end). A context (start, end, p)
    carries its own P. A context shorter than P + N is refused by name, unless `skipped` is a list: then
    it has no row and (index, length, P + N) goes into the list, and it still counts in the steady hit.
    With `scored` a set of context indices, the others get no row and no share of the steady hit; the
    steady rule still runs over every position."""
    rows = []
    res0 = seed_resident(seed, n_l, E)
    contexts = [c if len(c) == 3 else (c[0], c[1], P) for c in contexts]
    counted = [scored is None or c in scored for c in range(len(contexts))]
    for c, (t0, t1, P) in enumerate(contexts):
        if not counted[c]:
            continue
        if t1 - t0 < P + N and skipped is not None:
            skipped.append((c, t1 - t0, P + N))
            continue
        if t1 - t0 < P + N:
            raise ToolError(f"context {c} holds {t1 - t0} positions, shorter than prompt {P} + window {N}")
        Xp, Xw = X[t0:t0 + P], X[t0 + P:t0 + P + N]
        a = window_arm(Xp, Xw, seed, n_l, E, rule, arm="static")[0].mean()
        kw = dict(d=d, copies=copies, spares=spares, per_pass=per_pass)
        b = window_arm(Xp, Xw, seed, n_l, E, rule, arm="zero", **kw)[0].mean()
        ch, _, op = window_arm(Xp, Xw, seed, n_l, E, rule, arm="open", M=M, stage=stage, last=last, **kw)
        a_all = replay(X[t0 + P:t1], res0, n_l, E).per_row.mean()
        rows.append(dict(context=c, a=float(a), b=float(b), c=float(ch.mean()), open=op, a_all=float(a_all)))
    steady = replay(X, res0, n_l, E, rule, sem="flip", spares=spares, in_flight_cap=not per_pass, d=d,
                    link=Budget(copies)).per_row
    gen_rows = np.concatenate([steady[t0 + P:t1] for (t0, t1, P), k in zip(contexts, counted) if k])
    return rows, float(gen_rows.mean())


def verdict(a, c, steady):
    """adaptres §6: (c)-(a) >= 10 points -> R5 with R3; < 3 -> drop R5; steady adaptive < 70 % -> hold R3."""
    x = 100.0 * (c - a)
    hold = 100.0 * steady < 70.0
    if x >= 10.0:
        text = "R5 alone, hold R3" if hold else "R5 with R3"
    elif x < 3.0:
        text = "drop R5, hold R3" if hold else "drop R5"
    else:
        text = "between the bands (3 <= x < 10): no rule in adaptres §6" + (", hold R3" if hold else "")
    return f"(c)-(a) = {x:+.1f} points, steady adaptive {100.0 * steady:.1f} % -> {text}"


# --- fixture ----------------------------------------------------------------------------------------


class Lcg:
    def __init__(self, seed):
        self.x = seed & 0xFFFFFFFF

    def below(self, n):
        self.x = (1664525 * self.x + 1013904223) & 0xFFFFFFFF
        return (self.x >> 8) % n


FIXTURE_CONTRACT = {
    "boundary": "passes are numbered from 0; boundary b follows pass b-1; the first `kept` rows of a pass add 1.0 each "
                "to their ids' counts (f64), every row of a pass is served by the set in force at its start; at every "
                "boundary b the flips with live_at = b land first, then, when the passes ended since the seed have kept "
                "`every` rows past the last plan's (params.clock `kept`: the rows past `every` carry, so the carry "
                "is the kept total mod `every` and a boundary plans once at most), the rule plans and every "
                "count is multiplied by decay; a flip planned at b is live from pass live_at = b + d (d = 0: it lands "
                "at b after the plan); a flip is in flight from its boundary until it lands, so one with live_at = b "
                "is no longer in flight when boundary b plans",
    "order": "per layer: candidates = ids neither resident nor pending in, by count descending then id ascending; "
             "victims = residents not pending out, by count ascending then id ascending; the j-th candidate pairs "
             "with the j-th victim while count_in >= min_count and count_in >= count_out + margin; pairs go by gain "
             "(count_in - count_out) descending, then layer ascending, then in id ascending; a layer takes a pair "
             "only while its flips in flight (earlier boundaries' not yet landed, and this one's) number fewer than "
             "`spares`; at most `cap` in all a boundary; flip: the victim stays live until live_at",
    "cases": "the top level is one plan case; `cap_case` is another over the same seed and trace at cap 2 and "
             "spares 2; each case's header counts `cap_bound`, the boundaries where the cap cut a pair the spares "
             "limit let through (the generator refuses a cap_case with none)",
}
CAP_CASE = dict(cap=2, spares=2)
PINNED_CONTRACT = "params.pinned: a layer's first pinned[l] seed ids are never victims; `pinned_moved` counts the " \
                  "unpinned rule's flips that are not in `flips`"
AWAY_CONTRACT = "params.away: per layer the ids on another device, never admitted and never a victim; `away_moved` " \
                "counts the flips of the rule with none away that are not in `flips`"
OPEN_CONTRACT = {
    "open": "the opening reshuffle over a whole prompt: `counts` are the prompt's undecayed per-layer counts (every "
            "row); the card set is the seed's first `capacity` ids; per layer candidates = non-resident ids by count "
            "descending, then seed rank ascending, then id ascending; victims = resident ids by count ascending, "
            "then seed rank descending (the worst-ranked seed leaves first), then id ascending; the j-th candidate "
            "pairs with the j-th victim while count_in > count_out, gain = count_in - count_out; pairs go by gain "
            "descending, then layer ascending, then j ascending; `flips` are the first m, `flips_all` all of them",
}


def lcg_perm(g, n):
    p = list(range(n))
    for i in range(n - 1, 0, -1):
        j = g.below(i + 1)
        p[i], p[j] = p[j], p[i]
    return p


def lcg_rows(g, rows, K, E, hot, paired):
    """rows of K distinct ids, 3 in 4 from `hot`; paired: ids come as even-odd pairs (equal counts)."""
    out = []
    for _ in range(rows):
        row = []
        while len(row) < K:
            e = hot[g.below(len(hot))] if g.below(4) < 3 else g.below(E)
            if paired:
                e &= ~1
            if e not in row:
                row += [e, e + 1] if paired else [e]
        out.append(row)
    return out


def make_fixture(passes, lcg_seed, n_expert=64, top=6, cap_n=16, d=8, spares=1, rule_name="mid", pinned=0,
                 away=None):
    """Three layers: layer 1 relabels layer 0 (+16: equal gains across layers), layer 2 routes ids in even-odd
    pairs (equal counts within a layer); passes of 1-3 rows, the first 1..rows kept. The rule at `spares`,
    then `cap_case`: the same seed and trace at CAP_CASE's cap and spares. pinned > 0: every layer's first
    `pinned` seed ids are never victims (params.pinned); the header counts `pinned_moved`, the flips of the
    unpinned rule that are not the pinned rule's. away (a list of id lists, one a layer): those ids are never
    admitted nor a victim (params.away); the header counts `away_moved`, the flips of the rule with none away
    that are not this rule's."""
    if d < 0 or spares < 1 or not 0 <= pinned <= cap_n:
        raise ToolError(f"--d {d}, --spares {spares} and --pinned {pinned}: d >= 0, spares >= 1, "
                        f"0 <= pinned <= {cap_n}")
    E, K, g, L = n_expert, top, Lcg(lcg_seed), 3
    trace = [[] for _ in range(L)]
    kept = []
    for p in range(passes):
        rows = 1 + g.below(3)
        kept.append(1 + g.below(rows))
        base = (p // 50) * 7 % E
        rs = lcg_rows(g, rows, K, E, [(base + j) % E for j in range(10)], False)
        trace[0].append(rs)
        trace[1].append([[(e + 16) % E for e in row] for row in rs])
        trace[2].append(lcg_rows(g, rows, K, E, [(base + 32 + 2 * j) % E for j in range(5)], True))
    seed = [list(range(cap_n)), [(e + 16) % E for e in range(cap_n)], list(range(32, 32 + cap_n))]
    X, pz = [], []
    for p in range(passes):
        r0 = len(X)
        for r in range(len(trace[0][p])):
            X.append([trace[l][p][r] for l in range(L)])
        pz.append((r0, len(X), kept[p]))
    X = np.asarray(X, dtype=np.int64)
    res0 = np.zeros((L, E), dtype=bool)
    pin = np.zeros((L, E), dtype=bool)
    for l in range(L):
        res0[l, seed[l]] = True
        pin[l, seed[l][:pinned]] = True
    far = None
    if away is not None:
        if len(away) != L:
            raise ToolError(f"--away names {len(away)} layers; the fixture has {L}")
        far = np.zeros((L, E), dtype=bool)
        for l, ids in enumerate(away):
            for e in ids:
                if not 0 <= e < E:
                    raise ToolError(f"--away: layer {l} id {e} is not one of the {E} experts")
                if far[l, e]:
                    raise ToolError(f"--away: layer {l} names id {e} twice")
                if res0[l, e]:
                    raise ToolError(f"--away: layer {l} id {e} is in its layer's seed")
                far[l, e] = True

    def case(rule, s, p, off=far):
        flips = []
        ties = {"across_layers": 0, "in_id": 0, "out_id": 0, "blocked_in_flight": 0, "cap_bound": 0}
        replay(X, res0, [cap_n] * L, E, rule, sem="flip", spares=s, d=d, link=Budget(None), passes=pz,
               on_flip=lambda b, i, o, n, live: flips.append({"boundary": b, "layer": i, "out": o, "in": n,
                                                              "live_at": live}),
               ties=ties, pinned=pin if p else None, away=off)
        cap_bound = ties.pop("cap_bound")
        params = dict(rule.params(), spares=s, d=d, n_expert=E, top_k=K, capacity=[cap_n] * L)
        if p:
            params["pinned"] = [p] * L
        if off is not None:
            params["away"] = [sorted(int(e) for e in np.nonzero(off[l])[0]) for l in range(L)]
        return {"header": dict(ties=ties, cap_bound=cap_bound), "params": params, "seed": seed, "trace": trace,
                "kept": kept, "flips": flips}
    fx = case(Rule(rule_name), spares, pinned)
    if pinned:
        free = case(Rule(rule_name), spares, 0)["flips"]
        fx["header"] = dict(pinned=PINNED_CONTRACT, pinned_moved=sum(1 for f in free if f not in fx["flips"]),
                            **fx["header"])
    if far is not None:
        free = case(Rule(rule_name), spares, pinned, None)["flips"]
        fx["header"] = dict(away=AWAY_CONTRACT, away_moved=sum(1 for f in free if f not in fx["flips"]),
                            **fx["header"])
    fx["header"] = dict(FIXTURE_CONTRACT, **fx["header"])
    fx["cap_case"] = case(Rule(rule_name, cap=CAP_CASE["cap"]), CAP_CASE["spares"], pinned)
    return fx


def make_open_fixture(lcg_seed, prompt=48, m=11, n_expert=64, top=6, cap_n=16):
    """The opening reshuffle: seeds are full rankings (LCG permutations; layer 1 relabels layer 0's), the prompt
    routes mostly outside each seed's card set, so many residents keep count 0 and tie."""
    E, K, g, L = n_expert, top, Lcg(lcg_seed), 3
    s0 = lcg_perm(g, E)
    seed = [s0, [(e + 16) % E for e in s0], lcg_perm(g, E)]
    trace = [[] for _ in range(L)]
    rows0 = lcg_rows(g, prompt, K, E, s0[cap_n - 4: cap_n + 8], False)
    trace[0] = rows0
    trace[1] = [[(e + 16) % E for e in row] for row in rows0]
    trace[2] = lcg_rows(g, prompt, K, E, seed[2][cap_n - 4: cap_n + 8], True)
    Xp = np.asarray([[trace[l][t] for l in range(L)] for t in range(prompt)], dtype=np.int64)
    n_l = [cap_n] * L
    resident, _, pairs_all = opening(Xp, seed, n_l, E, None, Rule("mid"))
    counts = np.zeros((L, E), dtype=np.int64)
    for t in range(prompt):
        counts[np.arange(L)[:, None], Xp[t]] += 1

    def flips_of(pairs):
        return [{"layer": int(i), "out": int(b), "in": int(a), "gain": int(gn)} for gn, i, a, b in pairs]
    ties = {"across_layers": 0, "in_count": 0, "out_count": 0, "zero_count_residents": 0}
    for gn, i, a, b in pairs_all:
        ties["across_layers"] += int(any(g2 == gn and i2 != i for g2, i2, _, _ in pairs_all))
        ties["in_count"] += int(((counts[i] == counts[i, a]) & ~resident[i]).sum() > 1)
        ties["out_count"] += int(((counts[i] == counts[i, b]) & resident[i]).sum() > 1)
    ties["zero_count_residents"] = int(sum(((counts[i] == 0) & resident[i]).sum() for i in range(L)))
    ties["cut_in_gain_tie"] = int(0 < m < len(pairs_all) and pairs_all[m - 1][0] == pairs_all[m][0])
    params = dict(m=m, n_expert=E, top_k=K, capacity=n_l, prompt_rows=prompt)
    return {"header": dict(OPEN_CONTRACT, ties=ties, pairs_all=len(pairs_all)), "params": params, "seed": seed,
            "prompt": trace, "counts": counts.tolist(), "flips": flips_of(pairs_all[:m]),
            "flips_all": flips_of(pairs_all)}


def fixture_text(fx, command):
    fx["header"] = dict({"command": command}, **fx["header"])
    parts = [f'"{k}": {json.dumps(fx[k], separators=(",", ":"), ensure_ascii=False)}' for k in fx]
    return "{\n" + ",\n".join(parts) + "\n}\n"


# --- output -----------------------------------------------------------------------------------------


def v1_row(fam, s, cap_label, n_l, policy, seedname, label, hits, swaps, link, B_pin, B_page, tokens=None, rows=1,
           name=None):
    """One policy's row. tokens: passes replayed (the eval half by default); rows: rows a pass, which the hit
    divides by and host_gb_tok multiplies by (so with rows > 1 the `tok` columns are per pass)."""
    F = family(fam)
    L = len(F["eligible"])
    Tev = s.T - SPLIT if tokens is None else tokens
    S = F["expert_bytes"]
    h = hits.sum() / (Tev * rows * L * s.K)
    row = dict(fam=fam, set=s.name if name is None else name, cap=cap_label, f=sum(n_l) / (L * s.E),
               n_l=f"{min(n_l)}..{max(n_l)}", policy=policy, label=label, seed=seedname, tokens=Tev, layers=L, hit=h,
               hit_layer=(hits / (Tev * rows * s.K)).tolist(), swaps_tok=swaps / Tev, mb_tok=swaps / Tev * S / 1e6,
               ms_pin=swaps / Tev * S / (B_pin * 1e6), ms_page=swaps / Tev * S / (B_page * 1e6),
               host_gb_tok=(1 - h) * L * s.K * S * rows / 1e9)
    if link is not None and link.copy_ms:
        row["link_util"] = link.busy_ms / (Tev * F["step_ms"])
        row["delay_steps"] = link.delay_steps / max(swaps, 1)
    return row


def v1_line(r):
    return (f"| {r['fam']} | {r['set']} | {r['cap']} {r['n_l']} ({r['f']:.3f}) | {r['label']} | {100 * r['hit']:.1f} | "
            f"{r['swaps_tok']:.2f} | {r['mb_tok']:.0f} | {r['ms_pin']:.2f} | {r['ms_page']:.2f} | "
            f"{r.get('link_util', float('nan')):.3f} | {r.get('delay_steps', float('nan')):.2f} | {r['host_gb_tok']:.2f} |")


def window_line(r):
    arm = r["arm"] + (f" M={'all' if r['M'] is None else r['M']} K={r['K']}" if r["arm"] == "open" else "")
    return (f"{r['fam']} {r['set']} seed={r['seed']} P={r['P']} N={r['N']} {arm}: hit {100 * r['hit']:.1f} "
            f"b16 {' '.join(f'{100 * x:.0f}' for x in r['b16'])} sw {r['swaps_tok']:.2f} open {r['open']:.0f} "
            f"(n={r['nreq']})")


# --- commands ---------------------------------------------------------------------------------------


def spares_text(a):
    if a.spares < 1:
        raise ToolError(f"--spares {a.spares}: at least 1")
    return f"spares {a.spares} " + ("a boundary (in flight not counted, plan before land)" if a.spares_per_pass
                                    else "in flight a layer (land before plan)")


def parse_open(text):
    if text is None:
        return False, None
    if text == "all":
        return True, None
    try:
        m = int(text)
    except ValueError:
        raise ToolError(f"--open {text!r}: a count or `all`") from None
    if m < 0:
        raise ToolError(f"--open {m}: a count >= 0")
    return True, m


def parse_open_from(text, P, what="the prompt's positions"):
    """--open-from: None for `all` (the whole prompt), N for `last:N` with 1 <= N <= P."""
    if text is None or text == "all":
        return None
    head, sep, n = text.partition(":")
    try:
        last = int(n) if head == "last" and sep else None
    except ValueError:
        last = None
    if last is None:
        raise ToolError(f"--open-from {text!r}: `all` or `last:N`")
    if not 1 <= last <= P:
        raise ToolError(f"--open-from {text}: N must be within 1..{P}, {what}")
    return last


def stream_windows(fam, names, offset, data):
    """--streams: (sets by name, streams [(name, set, t0, t1)] in eval positions, truncation line or None)."""
    F = family(fam)
    if not names or any(not n for n in names):
        raise ToolError("--streams: a comma-separated list of set names, none empty")
    if len(names) > MAX_STREAMS:
        raise ToolError(f"--streams: {len(names)} streams, at most {MAX_STREAMS}")
    for n in names:
        if n not in F["sets"]:
            raise ToolError(f"--streams: {n!r} is not a set of family {fam} "
                            f"({', '.join(F['sets']) if F['sets'] else 'its table names no set'})")
    if offset is not None and offset < 0:
        raise ToolError(f"--offset {offset}: 0 or more")
    if offset is not None and all(names.count(n) == 1 for n in names):
        raise ToolError(f"--offset {offset}: it places the copies of a set named more than once, and every stream "
                        "names another set")
    sets, streams, seen = {}, [], {}
    for n in names:
        if n not in sets:
            sets[n] = Set(set_dir(data, n), fam)
            if sets[n].T <= SPLIT:
                raise ToolError(f"{sets[n].dir}: {sets[n].T} tokens; hit evaluates [{SPLIT}, tokens), which it holds "
                                "none of")
        s, m, j = sets[n], names.count(n), seen.get(n, 0)
        seen[n] = j + 1
        N = s.T - SPLIT
        K = N if m == 1 else N // m if offset is None else offset
        if m > 1 and K == 0 and offset is None:
            raise ToolError(f"--streams: {n} named {m} times, and its eval half holds {N} positions")
        if m * K > N:
            raise ToolError(f"--offset {K}: {m} windows of {n} reach eval position {m * K}, past its eval half of {N}")
        streams.append((n, s, 0, N) if K == 0 else (n, s, j * K, (j + 1) * K))
    W = min(t1 - t0 for _, _, t0, t1 in streams)
    note = None
    if any(t1 - t0 != W for _, _, t0, t1 in streams):
        note = (f"truncated to the shortest stream, {W} passes: " +
                ", ".join(f"s{i} {n} {t1 - t0} -> {W}" for i, (n, _, t0, t1) in enumerate(streams) if t1 - t0 != W))
        streams = [(n, s, t0, t0 + W) for n, s, t0, _ in streams]
    return sets, streams, note


def stream_seed(fam, streams, name, data, sets):
    """Per eligible layer, the ranked list of seed `name` for the streams sharing one card."""
    F = family(fam)
    lay = F["eligible"]
    if name == "in":
        return [rank(sum(s.counts(l, 0, SPLIT) for _, s, _, _ in streams)) for l in lay]
    if name == "insample":
        return [rank(sum(s.counts(l, SPLIT + t0, SPLIT + t1) for _, s, t0, t1 in streams)) for l in lay]
    if name == "prefix":
        return [np.arange(F["n_expert"]) for _ in lay]
    if name in ("cross", "pooled"):
        mine = {n for n, _, _, _ in streams}
        names = [n for n in F["sets"] if name == "pooled" or n not in mine]
        if not names:
            raise ToolError(f"seed cross: every set of family {fam} is a stream")
        for n in names:
            if n not in sets:
                sets[n] = Set(set_dir(data, n), fam)
        if name == "cross":
            return [rank(sum(sets[n].counts(l) for n in names)) for l in lay]
        return [rank(sum(sets[n].counts(l, 0, SPLIT) for n in names)) for l in lay]
    raise ToolError(f"no seed {name!r}: in, insample, cross, pooled or prefix")


def unique_misses(X, rec, B, E):
    """Missed (pass, layer, expert) triples of X [W B, L, K] (B rows a pass), each counted once."""
    W, L = X.shape[0] // B, X.shape[1]
    key = (np.repeat(np.arange(W, dtype=np.int64), B)[:, None, None] * L + np.arange(L)[None, :, None]) * E + X
    return int(np.unique(key[~rec]).size)


def streams_replay(fam, streams, seeds, pol, rule, link_gbps, a, data, sets, extra, turn=None):
    """The replay of `streams` on one card: rows (pooled first, then one a stream for static and adaptive)
    for every policy and seed. turn None is the merged replay (a pass carries one token of every stream);
    turn K is the turn replay: stream 0 runs K passes, then stream 1 K, round robin, every pass one row of
    one stream."""
    F = family(fam)
    B, s0 = len(streams), streams[0][1]
    W = streams[0][3] - streams[0][2]
    n_l = n_cap(fam, a.cap, s0.E)
    S, step = F["expert_bytes"], F["step_ms"]
    Xs = np.stack([s.stack(F["eligible"], SPLIT + t0, SPLIT + t1) for _, s, t0, t1 in streams], axis=1)
    if turn is None:
        X = Xs.reshape(W * B, Xs.shape[2], Xs.shape[3])
        passes, npass, rows_pass, whose = [(p * B, (p + 1) * B, B) for p in range(W)], W, B, None
    else:
        at = np.asarray([(s, t) for j0 in range(0, W, turn) for s in range(B)
                         for t in range(j0, min(j0 + turn, W))], dtype=np.int64)
        X = Xs[at[:, 1], at[:, 0]]
        passes, npass, rows_pass, whose = None, B * W, 1, at[:, 0]
    joined = "+".join(n for n, _, _, _ in streams)
    rows = []

    def emit(policy, sd, label, hits, swaps, rec=None, link=None, ties=None, plans=None):
        r = v1_row(fam, s0, a.cap, n_l, policy, sd, label, hits, swaps, link, a.pin_gbps, a.page_gbps,
                   tokens=npass, rows=rows_pass, name=joined)
        r.update(role="pooled", streams=joined, rows_pass=rows_pass)
        if turn is not None:
            r["turn"] = turn
        if ties is not None:
            # plans: the replay's own planning boundaries (a kept-row clock over
            # B-row passes plans B times as often as a pass clock while B <= every)
            r["cap_bound"] = ties["cap_bound"] / max(plans if plans is not None else npass // rule.every, 1)
        if rec is not None:
            u = unique_misses(X, rec, rows_pass, s0.E)
            r["host_gb_unique"] = u * S / npass / 1e9
        rows.append(r)
        if rec is None or B == 1:
            return
        for j, (n, s, t0, t1) in enumerate(streams):
            h = rec.reshape(W, B, rec.shape[1], rec.shape[2])[:, j] if whose is None else rec[whose == j]
            q = v1_row(fam, s, a.cap, n_l, policy, sd, label, h.sum(axis=(0, 2)).astype(np.int64), swaps, link,
                       a.pin_gbps, a.page_gbps, tokens=W)
            q.update(role=f"s{j}", streams=joined, rows_pass=rows_pass, window=[SPLIT + t0, SPLIT + t1])
            if turn is not None:
                q["turn"] = turn
            rows.append(q)

    lists = {sd: stream_seed(fam, streams, sd, data, sets) for sd in seeds}
    if "static" in pol and not a.seed:
        lists["insample"] = stream_seed(fam, streams, "insample", data, sets)
    L, K = X.shape[1], X.shape[2]
    if "static" in pol:
        for sd in lists:
            rec = np.zeros((W * B, L, K), dtype=bool)
            r = replay(X, seed_resident(lists[sd], n_l, s0.E), n_l, s0.E, record=rec, passes=passes)
            emit("static", sd, f"static[{sd}]", r.hits_layer, 0, rec)
    if "adaptive" in pol:
        for sd in seeds:
            link = Link(S, link_gbps, step) if link_gbps else None
            rec = np.zeros((W * B, L, K), dtype=bool)
            ties = {"across_layers": 0, "in_id": 0, "out_id": 0, "cap_bound": 0}
            r = replay(X, seed_resident(lists[sd], n_l, s0.E), n_l, s0.E, rule, sem="hole", d=0, link=link,
                       record=rec, passes=passes, ties=ties)
            emit("adaptive", sd, f"{rule.name}[{sd}{',' + extra if extra else ''}]", r.hits_layer, r.swaps, rec, link,
                 ties, plans=r.plans)
    if "belady" in pol:
        h, adm = belady(X, lists[seeds[0]], n_l)
        emit("belady", seeds[0], f"belady[{seeds[0]}]", h, adm)
    return rows


def streams_line(r):
    host = r["host_gb_tok"]
    uniq = r.get("host_gb_unique", float("nan"))
    streams = r["streams"] + (f" turn={r['turn']}" if "turn" in r else "")
    return (f"| {r['fam']} | {streams} | {r['role']} | {r['cap']} {r['n_l']} ({r['f']:.3f}) | {r['label']} | "
            f"{100 * r['hit']:.2f} | {r['swaps_tok']:.2f} | {r['mb_tok']:.0f} | {r['ms_pin']:.2f} | "
            f"{r.get('link_util', float('nan')):.3f} | {r.get('delay_steps', float('nan')):.2f} | "
            f"{100 * r.get('cap_bound', float('nan')):.1f} | {host:.3f} | {uniq:.3f} | {1e3 * uniq / HOST_GBPS:.2f} |")


def cmd_streams(a, out):
    F = family(a.family)
    if a.sets:
        raise ToolError(f"--streams names the sets; drop the positional sets {' '.join(a.sets)}")
    if a.window is not None or a.open is not None or a.prompt is not None:
        raise ToolError("--streams is the continuous replay: drop --window, --open and --prompt")
    if a.turn is not None and a.turn < 1:
        raise ToolError(f"--turn {a.turn}: at least 1 pass a stream")
    pol = (a.policies or "static,adaptive,belady").split(",")
    bad = [p for p in pol if p not in ("static", "adaptive", "belady")]
    if bad:
        raise ToolError(f"--policies with --streams: no policy {', '.join(bad)} (static, adaptive, belady; lru has "
                        "no rows-a-pass form)")
    names = a.streams.split(",")
    sets, streams, note = stream_windows(a.family, names, a.offset, a.data)
    B = len(streams)
    rule = Rule(a.rule, every=a.every)
    link_gbps = B_PIN if a.link_gbps is None else a.link_gbps
    extra = ",".join(x for x in (f"every={a.every}" if a.every is not None else "",
                                 f"gbps={a.link_gbps:g}" if a.link_gbps is not None else "") if x)
    seeds = [a.seed] if a.seed else list(F["seeds"])
    skipped = []
    if not a.seed and "cross" in seeds and all(n in {m for m, _, _, _ in streams} for n in F["sets"]):
        seeds.remove("cross")
        skipped.append("seed cross: every set of the family is a stream, so it has no cross seed; not run")
    out.write(f"# router-residency hit --streams: data={a.data} family={a.family} B={B}"
              + (f" turn {a.turn}" if a.turn is not None else "") + f" rule {rule.text()} "
              f"link {link_gbps:g} GB/s, step {F['step_ms']:g} ms, host {HOST_GBPS:g} GB/s\n")
    out.write("# streams: " + "  ".join(f"s{j} {n} [{SPLIT + t0}, {SPLIT + t1})" for j, (n, _, t0, t1) in
                                       enumerate(streams)) + "\n")
    for line in ([note] if note else []) + skipped:
        out.write(line + "\n")
    out.write("| fam | streams | row | cap (f) | policy | hit % | swaps/pass | MB/pass | ms/pass pin | link util | "
              "admit delay (steps) | cap-bound % | host GB/pass (slots) | host GB/pass (unique) | host ms/pass @"
              f"{HOST_GBPS:g} |\n")
    out.write("|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    rows = streams_replay(a.family, streams, seeds, pol, rule, link_gbps, a, a.data, sets, extra, turn=a.turn)
    for r in rows:
        out.write(streams_line(r) + "\n")
    out.flush()
    if B == 1:
        return rows
    alone = []
    for j, st in enumerate(streams):
        for r in streams_replay(a.family, [st], seeds, pol, rule, link_gbps, a, a.data, sets, extra,
                                turn=a.turn):
            r["role"] = f"alone s{j}"
            r["window"] = [SPLIT + st[2], SPLIT + st[3]]
            alone.append(r)
            out.write(streams_line(r) + "\n")
        out.flush()
    L, K, S = len(F["eligible"]), F["n_used"], F["expert_bytes"]
    for r in rows:
        if r["role"] != "pooled":
            continue
        mine = [q for q in alone if (q["policy"], q["label"]) == (r["policy"], r["label"])]
        if len(mine) != B:
            raise ToolError(f"{r['label']}: {len(mine)} alone rows for {B} streams")
        mean = sum(q["hit"] for q in mine) / B
        slots_ms = (mean - r["hit"]) * B * L * K * S / (HOST_GBPS * 1e6)  # B rows: a merged pass, a turn round
        sum_u = (sum(q["host_gb_unique"] for q in mine)
                 if "host_gb_unique" in r and all("host_gb_unique" in q for q in mine) else None)
        r.update(alone_mean=mean, delta=r["hit"] - mean, extra_ms_slots=slots_ms)
        head = "merged" if a.turn is None else f"turn {a.turn}"
        unit = "pass" if r["rows_pass"] == B else f"{B} passes"
        text = (f"{head} {r['streams']} {r['label']}: pooled {100 * r['hit']:.2f} alone mean {100 * mean:.2f} "
                f"delta {100 * (r['hit'] - mean):+.2f} points; extra host {slots_ms:+.2f} ms/{unit} at {HOST_GBPS:g} GB/s "
                "priced by slots [derived]")
        if sum_u is not None:
            text += (f"; unique host {1e3 * r['host_gb_unique'] * B / r['rows_pass'] / HOST_GBPS:.2f} ms/{unit} "
                     f"against {1e3 * sum_u / HOST_GBPS:.2f} for the {B} streams in passes of their own [derived]")
        out.write(text + "\n")
    return rows + alone


def cmd_hit(a, out):
    F = family(a.family)
    if a.streams is not None:
        return cmd_streams(a, out)
    if a.offset is not None:
        raise ToolError("--offset needs --streams: it places the windows of a set named twice")
    if a.turn is not None:
        raise ToolError("--turn needs --streams: it cuts the streams' replay into turns of K passes")
    if a.open is not None and a.window is None:
        raise ToolError("--open needs --window: the opening reshuffle is a timed-window replay")
    if a.stage is not None and a.open is None:
        raise ToolError("--stage needs --open")
    if a.open_from is not None and a.open is None:
        raise ToolError("--open-from needs --open")
    if a.window is None and a.prompt is not None:
        raise ToolError("--prompt needs --window (hit) — gen takes its own --prompt")
    names = a.sets or list(F["sets"])
    rule = Rule(a.rule, every=a.every)
    rows = []
    if a.window is not None:
        if a.window < 1:
            raise ToolError(f"--window {a.window}: at least 1")
        has_open, M = parse_open(a.open)
        P = 512 if a.prompt is None else a.prompt
        last = parse_open_from(a.open_from, P)
        seedname = a.seed or "pooled"
        out.write(f"# router-residency hit --window: data={a.data} family={a.family} rule {rule.text()} flip {spares_text(a)} "
                  f"d {a.d} copies/step {a.copies}" + (f" open from the last {last} prompt positions" if last else "")
                  + "\n")
        for n in names:
            s = Set(set_dir(a.data, n), a.family)
            n_l = n_cap(a.family, a.cap, s.E)
            X = hit_eval_stack(s, F)
            lists = seed_lists(a.family, s, seedname, a.data)
            for arm in ("static", "open" if has_open else "zero"):
                r = dict(fam=a.family, set=s.name, seed=seedname, P=P, N=a.window, arm=arm,
                         M=M if arm == "open" else None, K=(a.stage or 0) if arm == "open" else None, d=a.d,
                         copies=a.copies)
                if last is not None and arm == "open":
                    r["open_from"] = f"last:{last}"
                r.update(window_requests(X, P, a.window, lists, n_l, s.E, rule, arm, M, a.stage or 0, a.d, a.copies,
                                         a.spares, a.spares_per_pass, last))
                rows.append(r)
                out.write(window_line(r) + "\n")
                out.flush()
    else:
        pol = (a.policies or "static,adaptive,lru,belady").split(",")
        bad = [p for p in pol if p not in ("static", "adaptive", "lru", "belady")]
        if bad:
            raise ToolError(f"--policies: no policy {', '.join(bad)} (static, adaptive, lru, belady)")
        link_gbps = B_PIN if a.link_gbps is None else a.link_gbps
        extra = ",".join(x for x in (f"every={a.every}" if a.every is not None else "",
                                     f"gbps={a.link_gbps:g}" if a.link_gbps is not None else "") if x)
        out.write(f"# router-residency hit: data={a.data} family={a.family} eval [{SPLIT}, tokens) rule {rule.text()} "
                  f"link {link_gbps:g} GB/s, priced at {a.pin_gbps:g} / {a.page_gbps:g} GB/s\n")
        out.write("| fam | set | cap (f) | policy | hit % | swaps/tok | MB/tok | ms/tok pin | ms/tok page | link util | "
                  "admit delay (steps) | host GB/tok (eligible) |\n")
        out.write("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for n in names:
            s = Set(set_dir(a.data, n), a.family)
            n_l = n_cap(a.family, a.cap, s.E)
            X = hit_eval_stack(s, F)
            seeds = [a.seed] if a.seed else list(F["seeds"])
            lists = {sd: seed_lists(a.family, s, sd, a.data) for sd in seeds}
            if "static" in pol and not a.seed:
                lists["insample"] = seed_lists(a.family, s, "insample", a.data)
            S, step = F["expert_bytes"], F["step_ms"]

            def emit(policy, sd, label, hits, swaps, link=None):
                r = v1_row(a.family, s, a.cap, n_l, policy, sd, label, hits, swaps, link, a.pin_gbps, a.page_gbps)
                rows.append(r)
                out.write(v1_line(r) + "\n")
                out.flush()

            if "static" in pol:
                for sd in lists:
                    emit("static", sd, f"static[{sd}]", static_hits(X, lists[sd], n_l), 0)
            if "adaptive" in pol:
                for sd in seeds:
                    link = Link(S, link_gbps, step) if link_gbps else None
                    h, sw = adaptive(X, lists[sd], n_l, s.E, rule, link=link)
                    emit("adaptive", sd, f"{rule.name}[{sd}{',' + extra if extra else ''}]", h, sw, link)
            if "lru" in pol:
                for fill in ("all", "budget"):
                    budget = int(step * a.pin_gbps * 1e6 // S) if fill == "budget" else None
                    h, f = global_lru(X, lists[seeds[0]], n_l, fill=fill, budget=budget)
                    emit("lru", seeds[0], f"lru[{seeds[0]},fill={fill}]", h, f)
            if "belady" in pol:
                h, adm = belady(X, lists[seeds[0]], n_l)
                emit("belady", seeds[0], f"belady[{seeds[0]}]", h, adm)
    return rows


def cmd_gen(a, out):
    F = family(a.family)
    spares_text(a)
    d = set_dir(a.data, a.trace)
    try:
        requests, _ = rt.read_contexts(d)
    except (rt.TraceError, OSError) as e:
        raise ToolError(str(e)) from None
    if requests is not None and a.prompt is not None:
        raise ToolError(f"{d}: contexts.tsv gives each context its prompt; --prompt is refused beside it")
    if requests is None and a.split is not None:
        raise ToolError(f"{d}: no contexts.tsv, so no learn/held split for --split {a.split}")
    if requests is None and (a.prompt is None or a.prompt < 1):
        raise ToolError("gen needs --prompt P >= 1: the positions [0, P) of each context are the prompt")
    N = 96 if a.window is None else a.window
    s = Set(d, a.family)
    if requests is not None:
        contexts = [(r["first"], r["end"], r["prompt"]) for r in requests]
        shape = "from contexts.tsv prompt=per context"
    elif s.chunk is None:
        raise ToolError(f"{s.dir}: the manifest names no `chunk`, so the contexts are unknown")
    else:
        contexts = [(t0, min(t0 + s.chunk, s.T), a.prompt) for t0 in range(0, s.T, s.chunk)]
        shape = f"x {s.chunk} prompt={a.prompt}"
    # bounded by the contexts that get window arms (--short skip drops the others before the opening runs)
    armed = [p for t0, t1, p in contexts if t1 - t0 >= p + N] or [p for _, _, p in contexts]
    last = parse_open_from(a.open_from, min(armed),
                           "the prompt's positions" if requests is None else "the shortest prompt in contexts.tsv")
    if a.open is None:
        M = F["open_m"]
    else:
        M = parse_open(a.open)[1]
    stage = a.stage or 0
    n_l = n_plan(a.family)
    seedname = a.seed or "pooled"
    if seedname in ("in", "insample", "cross"):
        raise ToolError(f"gen: seed {seedname} would learn on the trace itself or cut it at {SPLIT}; "
                        "use pooled or prefix")
    if seedname == "pooled" and s.name in F["sets"]:
        raise ToolError(f"gen: {s.name} is one of family {a.family}'s corpus sets, whose first halves build the pooled "
                        "seed: the seed would have seen the trace; use prefix")
    lists = seed_lists(a.family, s, seedname, a.data)
    rule = Rule(a.rule)
    X = s.stack(F["eligible"])
    skipped = [] if a.short == "skip" else None
    scored = None
    if requests is not None and a.split is not None:
        scored = {i for i, r in enumerate(requests) if r["split"] == a.split}
        if not scored:
            raise ToolError(f"{d}: no request of split {a.split}")
    if requests is None:
        split_line = None
    elif scored is None:
        split_line = "split: none given, the learn and the held requests both scored"
    else:
        split_line = (f"split: {a.split}, {len(scored)} of {len(requests)} requests scored; the steady rule walks "
                      "every position")
    rows, steady = gen_values(X, contexts, a.prompt, N, lists, n_l, s.E, rule, M, stage, a.d, a.copies, a.spares,
                              a.spares_per_pass, skipped, last, scored)
    if not rows:
        raise ToolError(f"every one of the {len(contexts)} contexts is shorter than its P + window {N}")
    kept = [contexts[r["context"]] for r in rows]
    out.write(f"# router-residency gen: data={a.data} trace={s.dir} family={a.family} tokens={s.T} contexts="
              f"{len(contexts)} {shape} window={N}\n")
    out.write(f"# seed {seedname}, n_l {min(n_l)}..{max(n_l)} on {len(n_l)} layers, rule {rule.text()}, flip {spares_text(a)}, "
              f"d {a.d}, copies/step {a.copies}; open M={'all' if M is None else M} stage {stage}"
              + (f" from the last {last} prompt positions" if last else "") + "\n")
    if split_line is not None:
        out.write(split_line + "\n")
    if skipped is not None:
        lengths = ", ".join(f"{c}:{n}<{w}" for c, n, w in skipped) or "none"
        out.write(f"skipped {len(skipped)} of {len(contexts)} contexts shorter than P+N (lengths context:positions<P+N "
                  f"{lengths}); the steady rule still runs over them\n")
    for r, (t0, t1, P) in zip(rows, kept):
        own = "" if requests is None else f"  prompt {P} of {t1 - t0}"
        out.write(f"context {r['context']}: (a) static {100 * r['a']:.1f}  (b) adaptive {100 * r['b']:.1f}  "
                  f"(c) open {100 * r['c']:.1f} ({r['open']} opening swaps)  static over [P, end) {100 * r['a_all']:.1f}"
                  f"{own}\n")
    mean = {k: float(np.mean([r[k] for r in rows])) for k in ("a", "b", "c")}
    a_all = float(np.mean(np.concatenate([np.full(t1 - t0 - P, r["a_all"])
                                           for (t0, t1, P), r in zip(kept, rows)])))
    out.write(f"mean (n={len(rows)}): (a) {100 * mean['a']:.1f}  (b) {100 * mean['b']:.1f}  (c) {100 * mean['c']:.1f}  "
              f"static over [P, end) {100 * a_all:.1f}  steady {100 * steady:.1f}  "
              f"(b)-(a) {100 * (mean['b'] - mean['a']):+.1f}  (c)-(a) {100 * (mean['c'] - mean['a']):+.1f} points\n")
    out.write("steady = the rule continuous from the seed over every position of the trace, hit over [P, end) "
              "of each context\n")
    out.write("verdict: " + verdict(mean["a"], mean["c"], steady) + "\n")
    res = dict(contexts=rows, mean=mean, a_all=a_all, steady=steady, verdict=verdict(mean["a"], mean["c"], steady))
    if requests is not None:
        res["split"] = a.split or "both"
    if skipped is not None:
        res["skipped"] = [dict(context=c, positions=n, need=w) for c, n, w in skipped]
    if last is not None:
        res["open_from"] = f"last:{last}"
    return res


def away_lists(spec):
    """--away's layers of ids: a JSON file holding a list of id lists, or `ids,…/ids,…` inline (an empty layer
    is an empty field)."""
    if os.path.isfile(spec):
        with open(spec, encoding="utf-8") as f:
            lists = json.load(f)
        if not (isinstance(lists, list) and all(isinstance(l, list) and all(isinstance(e, int) for e in l)
                                                for l in lists)):
            raise ToolError(f"--away {spec}: the file holds no list of id lists")
        return lists
    try:
        return [[int(e) for e in field.split(",") if e.strip()] for field in spec.split("/")]
    except ValueError:
        raise ToolError(f"--away {spec!r}: neither a file nor ids comma-separated, layers '/'-separated") from None


def cmd_fixture(a, out, argv):
    if a.spares_per_pass:
        raise ToolError("the fixture is written with the in-flight cap on: drop --spares-per-pass")
    if a.open:
        fx = make_open_fixture(a.lcg)
        ties = fx["header"]["ties"]
        if 0 in (ties["across_layers"], ties["in_count"], ties["out_count"], ties["zero_count_residents"],
                 ties["cut_in_gain_tie"]):
            raise ToolError(f"the open fixture holds no tie or no cut to check ({ties}, {fx['header']['pairs_all']} "
                            "pairs); change --lcg")
    else:
        if a.passes < 1:
            raise ToolError(f"--passes {a.passes}: at least 1")
        away = away_lists(a.away) if a.away is not None else None
        fx = make_fixture(a.passes, a.lcg, d=a.d, spares=a.spares, pinned=a.pinned, away=away)
        ties = fx["header"]["ties"]
        if a.pinned and fx["header"]["pinned_moved"] == 0:
            raise ToolError(f"--pinned {a.pinned} moves no flip, so the file cannot tell pinning from none; "
                            "change --pinned, --lcg or --passes")
        if away is not None and fx["header"]["away_moved"] == 0:
            raise ToolError(f"--away {a.away} moves no flip, so the file cannot tell it from none away; "
                            "change --away, --lcg or --passes")
        if away is not None and any(f["in"] in away[f["layer"]] for f in fx["flips"] + fx["cap_case"]["flips"]):
            raise ToolError("a flip admits an away id: the replay broke its own contract")
        if a.passes >= 100 and 0 in (ties["across_layers"], ties["in_id"], ties["blocked_in_flight"]):
            raise ToolError(f"the fixture's flips hold no tie or no in-flight block to check ({ties}); change --lcg")
        if fx["cap_case"]["header"]["cap_bound"] == 0:
            raise ToolError(f"the fixture's cap_case (cap {CAP_CASE['cap']}, spares {CAP_CASE['spares']}) never meets "
                            "its cap, so it cannot tell the cap from none; change --lcg or --passes")
    text = fixture_text(fx, "tools/ref/router-residency.py " + " ".join(argv))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(text)
    extra = ""
    if not a.open:
        cc = fx["cap_case"]
        extra = (f", cap {fx['header']['cap_bound']} bound; cap_case {len(cc['flips'])} flips, ties {cc['header']['ties']}, "
                 f"cap {cc['header']['cap_bound']} bound")
    out.write(f"fixture: {a.out}: {len(fx['flips'])} flips, ties {ties}{extra}, "
              f"md5 {hashlib.md5(text.encode()).hexdigest()}\n")
    return fx


def parser():
    p = argparse.ArgumentParser(prog="router-residency.py", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd")

    def common(q):
        q.add_argument("--data")
        q.add_argument("--seed")
        q.add_argument("--rule", default="mid")
        q.add_argument("--d", type=int, default=1)
        q.add_argument("--spares", type=int, default=1)
        q.add_argument("--spares-per-pass", action="store_true")
        q.add_argument("--copies", type=int, default=COPIES)
        q.add_argument("--open")
        q.add_argument("--stage", type=int)
        q.add_argument("--open-from")
        q.add_argument("--window", type=int)
        q.add_argument("--prompt", type=int)
        q.add_argument("--json")

    h = sub.add_parser("hit")
    h.add_argument("family")
    h.add_argument("sets", nargs="*")
    common(h)
    h.add_argument("--cap", default="plan")
    h.add_argument("--every", type=int)
    h.add_argument("--link-gbps", type=float)
    h.add_argument("--pin-gbps", type=float, default=B_PIN)
    h.add_argument("--page-gbps", type=float, default=B_PAGE)
    h.add_argument("--policies", help="static,adaptive,lru,belady by default; static,adaptive,belady with --streams")
    h.add_argument("--streams", help="S1,S2,...: B interleaved sessions on one card, each a set of the family; a set "
                                     "named m > 1 times is m windows of its eval half, copy j at [j K, (j + 1) K)")
    h.add_argument("--offset", type=int, help="K for --streams' windows of a repeated set: floor(eval / m) by default "
                                              "(disjoint), 0 = every copy the whole eval half")
    h.add_argument("--turn", type=int, help="K passes a stream runs before the next takes the card (needs --streams): "
                                            "1 is token interleave, 64 today's time-slicing")
    g = sub.add_parser("gen")
    g.add_argument("family")
    g.add_argument("trace")
    common(g)
    g.add_argument("--short", choices=("skip",))
    g.add_argument("--split", choices=("held", "learn"))
    f = sub.add_parser("fixture")
    f.add_argument("--out", required=True)
    f.add_argument("--passes", type=int, default=400)
    f.add_argument("--lcg", type=int, default=1)
    f.add_argument("--d", type=int, default=8)
    f.add_argument("--spares", type=int, default=1)
    f.add_argument("--spares-per-pass", action="store_true")
    f.add_argument("--open", action="store_true")
    f.add_argument("--pinned", type=int, default=0)
    f.add_argument("--away")
    return p


def main(argv, out=None):
    out = sys.stdout if out is None else out
    if argv == ["--self-test"]:
        return self_test()
    if argv[:1] == ["--fixture"]:
        argv = ["fixture"] + argv[1:]
    a = parser().parse_args(argv)
    if a.cmd is None:
        sys.stderr.write(__doc__)
        return 2
    try:
        if a.cmd == "hit":
            rows = cmd_hit(a, out)
        elif a.cmd == "gen":
            rows = cmd_gen(a, out)
        else:
            cmd_fixture(a, out, argv)
            return 0
    except (ToolError, rc.SetError) as e:
        sys.stderr.write(f"router-residency: {e}\n")
        return 1
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1)
    return 0


# --- self-test --------------------------------------------------------------------------------------


def X_of(rows):
    """rows: list of per-step lists of per-layer id tuples -> [T, L, K]."""
    return np.asarray(rows, dtype=np.int64)


def case_static():
    X = X_of([[(0, 1)], [(2, 3)], [(0, 3)]])  # one layer, K=2: capacity 1 holding 0, selected at steps 0 and 2
    assert static_hits(X, [[0, 1, 2, 3]], [1]).tolist() == [2]
    assert static_hits(X, [[3, 0, 1, 2]], [2]).tolist() == [4]


def case_belady():
    # cap 1, K=1, seq a b a b a b seeded with a: MIN with bypass keeps a at every b (a's next use is sooner)
    X = X_of([[(0,)], [(1,)], [(0,)], [(1,)], [(0,)], [(1,)]])
    h, adm = belady(X, [[0, 1]], [1])
    assert h.tolist() == [3] and adm == 0, (h, adm)  # last b: both never used again -> tie, bypass
    X = X_of([[(0,)], [(5,)], [(0,)]])  # a never-again miss must not evict a resident used again
    h, adm = belady(X, [[0]], [1])
    assert h.tolist() == [2] and adm == 0, (h, adm)


def case_lru():
    # 2 layers, n_l (1, 1) -> global 2: layer 0 misses every step and takes the slots in LRU order
    X = X_of([[(1,), (7,)], [(2,), (7,)], [(1,), (7,)]])
    h, f = global_lru(X, [[0], [7]], [1, 1])
    assert h.tolist() == [0, 3] and f == 3, (h, f)
    h, f = global_lru(X, [[0], [7]], [1, 1], fill="budget", budget=0)
    assert h.tolist() == [0, 3] and f == 0


def case_adaptive_link():
    # one layer, E=4, cap 1 seeded with 0; 2 used every step; pass every 2, no decay: after step 1 2 has 2,
    # 0 has 0 -> swap (2 >= 0 + 1.5), serves from step 2
    X = X_of([[(2,)]] * 6)
    h, sw = adaptive(X, [[0, 1, 2, 3]], [1], 4, Rule("strata", every=2, decay=1.0))
    assert h.tolist() == [4] and sw == 1, (h, sw)
    # used once a pass with decay 0 never reaches the margin -> no swap
    X = X_of([[(2,)], [(3,)]] * 3)
    h, sw = adaptive(X, [[0, 1, 2, 3]], [1], 4, Rule("strata", every=2, decay=0.0))
    assert h.tolist() == [0] and sw == 0, (h, sw)
    # the link: a copy of 25 ms at step 10 ms issued after step 1 (t = 20 ms) ends at 45 -> serves step 5
    X = X_of([[(2,)]] * 8)
    link = Link(25_000_000, 1.0, 10.0)
    h, sw = adaptive(X, [[0, 1, 2, 3]], [1], 4, Rule("strata", every=2, decay=1.0), link=link)
    assert h.tolist() == [3] and sw == 1, (h, sw)  # steps 5, 6, 7


def case_adaptive_margin():
    # pass every 4, no decay: 2 used 3 times, the resident 0 once -> 3 >= 1 + 1.5 swaps; margin 3 must not
    X = X_of([[(2,)], [(2,)], [(2,)], [(0,)], [(2,)], [(2,)]])
    h, sw = adaptive(X, [[0, 1, 2, 3]], [1], 4, Rule("strata", every=4, margin=1.5, decay=1.0))
    assert sw == 1 and h.tolist() == [3], (h, sw)  # step 3 (0) and steps 4-5 (2)
    h, sw = adaptive(X, [[0, 1, 2, 3]], [1], 4, Rule("strata", every=4, margin=3.0, decay=1.0))
    assert sw == 0 and h.tolist() == [1], (h, sw)


def case_adaptive_min_count():
    # 2 uses clear margin 1.5 over an unused resident but not a threshold of 3
    X = X_of([[(2,)], [(2,)], [(2,)]])
    h, sw = adaptive(X, [[0, 1, 2, 3]], [1], 4, Rule("strata", every=2, min_count=3.0, decay=1.0))
    assert sw == 0, (h, sw)


def case_adaptive_cap():
    # two layers both want a swap, cap 1 admits the larger margin (layer 1: 3 uses against 2)
    X = X_of([[(2,), (2,)], [(2,), (2,)], [(1,), (2,)], [(3,), (3,)]])
    h, sw = adaptive(X, [[0, 1, 2, 3], [0, 1, 2, 3]], [1, 1], 4, Rule("strata", every=3, cap=1, decay=1.0))
    assert sw == 1 and h.tolist() == [0, 0], (h, sw)
    X = X_of([[(2,), (2,)], [(2,), (2,)], [(1,), (2,)], [(3,), (2,)]])
    h, sw = adaptive(X, [[0, 1, 2, 3], [0, 1, 2, 3]], [1, 1], 4, Rule("strata", every=3, cap=1, decay=1.0))
    assert sw == 1 and h.tolist() == [0, 1], (h, sw)


def case_lru_is_global():
    # layer 0 alternates 1, 2; layer 1's seed 9 is never used: a shared capacity of 3 keeps 1, 2 and 5
    X = X_of([[(1,), (5,)], [(2,), (5,)], [(1,), (5,)], [(2,), (5,)]])
    h3, _ = global_lru(X, [[0], [9]], [2, 1])
    assert h3.tolist() == [2, 3], h3
    # layer 1 reuses 9 and never 8: the shared capacity 3 moves 8's slot to layer 0, which then holds 1
    # and 2 (a per-layer LRU at n_l 1 would never hit on layer 0)
    X = X_of([[(1,), (9,)], [(2,), (9,)], [(1,), (9,)], [(2,), (9,)], [(1,), (9,)]])
    h, _ = global_lru(X, [[0], [9, 8]], [1, 2])
    assert h.tolist() == [3, 5], h


def case_window():
    # one layer, E=4, top-1, n_l 1 seeded with 0; mid rule. Decode 2 2 2 2 0 2 2 2: at boundary 4 the count
    # of 2 is 4 >= 0 + 3 -> flip 2 in, 0 out, live at max(4 + 1, ceil(4 + 1/30)) = 5. Flip: 0 stays live
    # through step 4 (a hit), 2 serves steps 5-7. The prompt (3s) is not counted by the zero arm.
    rule = Rule("mid")
    Xp = X_of([[(3,)]] * 4)
    Xd = X_of([[(2,)]] * 4 + [[(0,)]] + [[(2,)]] * 3)
    h, sw, op = window_arm(Xp, Xd, [[0, 1, 2, 3]], [1], 4, rule, arm="zero")
    assert h.tolist() == [0, 0, 0, 0, 1, 1, 1, 1] and sw == 1 and op == 0, (h, sw, op)
    h, sw, _ = window_arm(Xp, Xd, [[0, 1, 2, 3]], [1], 4, rule, arm="static")
    assert h.tolist() == [0, 0, 0, 0, 1, 0, 0, 0] and sw == 0, h
    # the requests: two of P 4 + N 8 (the tail of 3 is no request), each from the seed with zero counts
    X = np.concatenate([Xp, Xd, Xp, Xd, Xp[:3]])
    w = window_requests(X, 4, 8, [[0, 1, 2, 3]], [1], 4, rule, "zero")
    assert (w["nreq"], w["hit"], w["swaps_tok"], w["b16"]) == (2, 0.5, 1 / 8, [0.5]), w


def case_open():
    # one layer, E=6, top-1, n_l 2 seeded [0, 4, ...]. Prompt: 3 x5, 0 x32, 1 x4. Whole-prompt counts put 3
    # (5) over 1 (4); decayed counts (x0.9 every 4) put 1 (3.7) over 3 (4.6 x 0.9^9 = 1.78). Victim: 4
    # (count 0; 0 has 32). M 1: 3 in, 4 out, gain 5. Decode 1 1 1 1: 1 is not on the card -> no hit (the
    # rule's flip of 1 at boundary 4 is live at step 5, past the window).
    rule = Rule("mid")
    seed = [[0, 4, 1, 2, 3, 5]]
    Xp = X_of([[(3,)]] * 5 + [[(0,)]] * 32 + [[(1,)]] * 4)
    Xd = X_of([[(1,)]] * 4)
    _, dec, pairs = opening(Xp, seed, [2], 6, 1, rule)
    assert [(float(g), i, int(a), int(b)) for g, i, a, b in pairs] == [(5.0, 0, 3, 4)], pairs
    assert abs(dec[0, 1] - 3.7) < 1e-12 and abs(dec[0, 3] - 4.6 * 0.9 ** 9) < 1e-12, dec
    h, sw, op = window_arm(Xp, Xd, seed, [2], 6, rule, arm="open", M=1)
    assert h.tolist() == [0, 0, 0, 0] and op == 1 and sw == 1, (h, sw, op)
    # staged at 1 a step: the pair is live from step 1; the seed card set serves step 0
    h, _, _ = window_arm(Xp, X_of([[(4,)], [(3,)], [(4,)], [(3,)]]), seed, [2], 6, rule, arm="open", M=1, stage=1)
    assert h.tolist() == [1, 1, 0, 1], h
    # ties go to the seed rank, not the id: seed [5, 0, 1, 2, 3, 4], n_l 3 -> {5, 0, 1} all at count 0; the
    # prompt uses 3 and 4 twice each. Candidates tie at 2: 3 (rank 4) before 4 (rank 5); victims tie at 0:
    # the worst-ranked resident 1 leaves first, then 0.
    _, _, pairs = opening(X_of([[(3,)], [(4,)], [(3,)], [(4,)]]), [[5, 0, 1, 2, 3, 4]], [3], 6, None, rule)
    assert [(float(g), i, int(a), int(b)) for g, i, a, b in pairs] == [(2.0, 0, 3, 1), (2.0, 0, 4, 0)], pairs


def case_open_last():
    # case_open's prompt and seed, the pairs by the last 4 positions (1 x4): 1 (4 uses) goes in, and of the
    # residents 0 and 4 (no use there) the worse seed rank, 4, leaves: gain 4. The whole prompt's pair was 3
    # for 4. The decayed counts stay the whole prompt's. Decode 1 1 1 1 then hits every step (whole prompt: none).
    rule = Rule("mid")
    seed = [[0, 4, 1, 2, 3, 5]]
    Xp = X_of([[(3,)]] * 5 + [[(0,)]] * 32 + [[(1,)]] * 4)
    Xd = X_of([[(1,)]] * 4)
    _, dec_all, _ = opening(Xp, seed, [2], 6, 1, rule)
    _, dec, pairs = opening(Xp, seed, [2], 6, 1, rule, last=4)
    assert [(float(g), i, int(a), int(b)) for g, i, a, b in pairs] == [(4.0, 0, 1, 4)], pairs
    assert (dec == dec_all).all(), (dec, dec_all)
    h, _, op = window_arm(Xp, Xd, seed, [2], 6, rule, arm="open", M=1, last=4)
    assert h.tolist() == [1, 1, 1, 1] and op == 1, (h, op)
    # a window past the prompt is refused, not read as the whole prompt
    try:
        opening(Xp, seed, [2], 6, 1, rule, last=Xp.shape[0] + 1)
    except ToolError as e:
        assert "on a prompt of 41 positions" in str(e), e
    else:
        raise AssertionError("--open-from past the prompt was taken")
    # last:P is the whole prompt
    assert opening(Xp, seed, [2], 6, 1, rule, last=Xp.shape[0])[2] == opening(Xp, seed, [2], 6, 1, rule)[2]
    assert parse_open_from(None, 8) is None and parse_open_from("all", 8) is None and parse_open_from("last:8", 8) == 8
    for bad, why in (("last:0", "within 1..8"), ("last:9", "within 1..8"), ("first:4", "`all` or `last:N`"),
                     ("last:x", "`all` or `last:N`")):
        try:
            parse_open_from(bad, 8)
        except ToolError as e:
            assert why in str(e), (bad, e)
        else:
            raise AssertionError(f"--open-from {bad} was taken")


def case_gen():
    # one context of 16: prompt 3 x8, generation 2 2 2 2 0 2 2 2 (the window, N 8). Seed card set {0}.
    # (a) static: step 4 only, 1/8. (b) from zero counts: 1/2 (case_window). (c) the opening puts 3 (8 uses)
    # in for 0; the rule then starts from the prompt's decayed 3 (6.84): 2's 4 uses never clear 6.84 + 3,
    # so no hit, 0. Steady over all 16 from {0}: 3 in at boundary 4 (live 5), 2 never clears 3's count:
    # no hit in [8, 16).
    rule = Rule("mid")
    X = X_of([[(3,)]] * 8 + [[(2,)]] * 4 + [[(0,)]] + [[(2,)]] * 3)
    rows, steady = gen_values(X, [(0, 16)], 8, 8, [[0, 1, 2, 3]], [1], 4, rule, None, 0, 1, COPIES)
    r = rows[0]
    assert (r["a"], r["b"], r["c"], r["open"], r["a_all"], steady) == (0.125, 0.5, 0.0, 1, 0.125, 0.0), (r, steady)
    try:
        gen_values(X, [(0, 16)], 9, 8, [[0, 1, 2, 3]], [1], 4, rule, None, 0, 1, COPIES)
        raise AssertionError("a context shorter than P + N was accepted")
    except ToolError as e:
        assert "shorter than prompt 9 + window 8" in str(e), e
    assert verdict(0.60, 0.75, 0.78).endswith("-> R5 with R3"), verdict(0.60, 0.75, 0.78)
    assert verdict(0.60, 0.62, 0.78).endswith("-> drop R5")
    assert verdict(0.60, 0.65, 0.78).endswith("no rule in adaptres §6")
    assert verdict(0.60, 0.75, 0.65).endswith("-> R5 alone, hold R3")


def case_gen_contexts():
    # A route trace of two requests of unequal lengths, 10 positions with prompt 4 and 7 with prompt 3:
    # gen takes each context and its P from contexts.tsv, the same values gen_values gives the triples,
    # and refuses --prompt beside the sidecar.
    with tempfile.TemporaryDirectory() as root:
        rows = {l: [tuple((7 * t + l + 50 * j) % 384 for j in range(6)) for t in range(17)] for l in range(40)}
        d = wu._fake_set(root, "chat", rows, 384, 6)
        path = os.path.join(d, "MANIFEST.tsv")
        with open(path, encoding="utf-8") as f:
            text = f.read()
        calls = "# call\tindex\tfirst\tprompt_call\tend\tpos0\ncall\t0\t0\t3\t10\t0\ncall\t1\t10\t2\t17\t0\n"
        with open(path, "w", encoding="utf-8") as f:
            f.write(text.replace("# complete", calls + "# complete"))
        ctx = [dict(request=0, first=0, end=10, prompt=4, prompt_call=3, prompt_ids=4, cache=0, generated=7,
                    stop="eos", prompt_id="a", genre="ko", split="learn"),
               dict(request=1, first=10, end=17, prompt=3, prompt_call=2, prompt_ids=3, cache=0, generated=5,
                    stop="limit", prompt_id="b", genre="en", split="held")]
        rt._write_contexts(d, ctx)
        js = os.path.join(root, "gen.json")
        out = io.StringIO()
        argv = ["gen", "v41", d, "--data", root, "--seed", "prefix", "--window", "2", "--json", js]
        with redirect_stderr(io.StringIO()) as err:
            code = main(argv, out)
        assert code == 0, err.getvalue()
        text = out.getvalue()
        assert "contexts=2 from contexts.tsv prompt=per context window=2" in text, text
        assert "prompt 4 of 10" in text and "prompt 3 of 7" in text, text
        with open(js, encoding="utf-8") as f:
            got = json.load(f)
        s = Set(d, "v41")
        n_l = n_plan("v41")
        want, steady = gen_values(s.stack(family("v41")["eligible"]), [(0, 10, 4), (10, 17, 3)], None, 2,
                                  seed_lists("v41", s, "prefix", root), n_l, 384, Rule("mid"),
                                  family("v41")["open_m"], 0, 1, COPIES)
        assert got["contexts"] == want and got["steady"] == steady, (got, want)
        err = io.StringIO()
        with redirect_stderr(err), redirect_stdout(io.StringIO()):
            code = main(argv[:-2] + ["--prompt", "4"])
        assert code == 1 and "--prompt is refused beside it" in err.getvalue(), err.getvalue()
        # --open-from is bounded by the shortest prompt in contexts.tsv (3), not by a --prompt it has none of
        err = io.StringIO()
        with redirect_stderr(err), redirect_stdout(io.StringIO()):
            code = main(argv[:-2] + ["--open-from", "last:4"])
        assert code == 1 and "within 1..3, the shortest prompt in contexts.tsv" in err.getvalue(), err.getvalue()
        # last:3 reaches the opening of every context: the (c) arms are gen_values' with last 3, which on context
        # 0 (prompt 4) differ from the whole prompt's
        js3 = os.path.join(root, "gen3.json")
        with redirect_stderr(io.StringIO()) as e3, redirect_stdout(io.StringIO()) as o3:
            code = main(argv[:-2] + ["--open-from", "last:3", "--json", js3])
        assert code == 0 and "from the last 3 prompt positions" in o3.getvalue(), e3.getvalue()
        with open(js3, encoding="utf-8") as f:
            got3 = json.load(f)
        want3, _ = gen_values(s.stack(family("v41")["eligible"]), [(0, 10, 4), (10, 17, 3)], None, 2,
                              seed_lists("v41", s, "prefix", root), n_l, 384, Rule("mid"),
                              family("v41")["open_m"], 0, 1, COPIES, last=3)
        assert got3["contexts"] == want3 and want3 != want and got3["open_from"] == "last:3", (got3, want3)


def case_gen_split():
    # The two requests of case_gen_contexts, 0 learn and 1 held. --split held scores request 1 alone: its
    # window row as gen_values gives it, and the steady hit over its [P, end) with the rule walked over all
    # 17 positions; no --split says both are scored; --split on a chunk trace is refused.
    with tempfile.TemporaryDirectory() as root:
        rows = {l: [tuple((7 * t + l + 50 * j) % 384 for j in range(6)) for t in range(17)] for l in range(40)}
        d = wu._fake_set(root, "chat", rows, 384, 6)
        path = os.path.join(d, "MANIFEST.tsv")
        with open(path, encoding="utf-8") as f:
            text = f.read()
        calls = "# call\tindex\tfirst\tprompt_call\tend\tpos0\ncall\t0\t0\t3\t10\t0\ncall\t1\t10\t2\t17\t0\n"
        with open(path, "w", encoding="utf-8") as f:
            f.write(text.replace("# complete", calls + "# complete"))
        rt._write_contexts(d, [dict(request=0, first=0, end=10, prompt=4, prompt_call=3, prompt_ids=4, cache=0,
                                    generated=7, stop="eos", prompt_id="a", genre="ko", split="learn"),
                               dict(request=1, first=10, end=17, prompt=3, prompt_call=2, prompt_ids=3, cache=0,
                                    generated=5, stop="limit", prompt_id="b", genre="en", split="held")])
        argv = ["gen", "v41", d, "--data", root, "--window", "2"]

        def gen(extra, seed="prefix"):
            out, err = io.StringIO(), io.StringIO()
            js = os.path.join(root, "gen.json")
            with redirect_stderr(err):
                code = main(argv + ["--seed", seed, "--json", js] + extra, out)
            if code != 0:
                return None, err.getvalue()
            with open(js, encoding="utf-8") as f:
                return json.load(f), out.getvalue()

        got, text = gen([])
        assert "split: none given, the learn and the held requests both scored" in text and got["split"] == "both", text
        got, text = gen(["--split", "held"])
        assert "split: held, 1 of 2 requests scored" in text and "context 0:" not in text, text
        s = Set(d, "v41")
        n_l = n_plan("v41")
        seed = seed_lists("v41", s, "prefix", root)
        X = s.stack(family("v41")["eligible"])
        rule = Rule("mid")
        want, _ = gen_values(X, [(10, 17, 3)], None, 2, seed, n_l, 384, rule, family("v41")["open_m"], 0, 1, COPIES)
        strip = lambda r: {k: v for k, v in r.items() if k != "context"}
        assert [r["context"] for r in got["contexts"]] == [1], got["contexts"]
        assert [strip(r) for r in got["contexts"]] == [strip(r) for r in want], (got["contexts"], want)
        per = replay(X, seed_resident(seed, n_l, 384), n_l, 384, rule, sem="flip", spares=1, in_flight_cap=True, d=1,
                     link=Budget(COPIES)).per_row
        assert got["steady"] == float(per[13:17].mean()) and got["split"] == "held", got
        got, text = gen(["--split", "learn"])
        assert [r["context"] for r in got["contexts"]] == [0] and got["steady"] == float(per[4:10].mean()), got

        c = wu._fake_set(root, "chunked", rows, 384, 6)
        err = io.StringIO()
        with redirect_stderr(err), redirect_stdout(io.StringIO()):
            code = main(["gen", "v41", c, "--data", root, "--prompt", "4", "--split", "held"])
        assert code == 1 and "no contexts.tsv, so no learn/held split" in err.getvalue(), err.getvalue()


def case_gen_short():
    # Three requests, the middle one 3 positions (prompt 2): shorter than P + N = 4 at window 2. Without
    # --short it is refused by name; with --short skip the line names it, the two long contexts keep the
    # arms they have alone, and the steady hit still runs over all three.
    with tempfile.TemporaryDirectory() as root:
        rows = {l: [tuple((5 * t + 2 * l + 50 * j) % 384 for j in range(6)) for t in range(20)] for l in range(40)}
        d = wu._fake_set(root, "chat", rows, 384, 6)
        path = os.path.join(d, "MANIFEST.tsv")
        with open(path, encoding="utf-8") as f:
            text = f.read()
        calls = ("# call\tindex\tfirst\tprompt_call\tend\tpos0\ncall\t0\t0\t3\t10\t0\n"
                 "call\t1\t10\t1\t13\t0\ncall\t2\t13\t2\t20\t0\n")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text.replace("# complete", calls + "# complete"))
        spans = [(0, 10, 4), (10, 13, 2), (13, 20, 3)]
        rt._write_contexts(d, [dict(request=i, first=t0, end=t1, prompt=p, prompt_call=p - 1, prompt_ids=p, cache=0,
                                    generated=t1 - t0 - p + 1, stop="eos", prompt_id=f"p{i}", genre="ko", split="learn")
                               for i, (t0, t1, p) in enumerate(spans)])
        argv = ["gen", "v41", d, "--data", root, "--seed", "prefix", "--window", "2"]
        err = io.StringIO()
        with redirect_stderr(err), redirect_stdout(io.StringIO()):
            code = main(argv)
        assert code == 1 and "context 1 holds 3 positions, shorter than prompt 2 + window 2" in err.getvalue(), \
            err.getvalue()
        js = os.path.join(root, "gen.json")
        out = io.StringIO()
        with redirect_stderr(io.StringIO()) as e2:
            code = main(argv + ["--short", "skip", "--json", js], out)
        assert code == 0, e2.getvalue()
        text = out.getvalue()
        assert "skipped 1 of 3 contexts shorter than P+N (lengths context:positions<P+N 1:3<4)" in text, text
        assert "context 1:" not in text and "mean (n=2)" in text, text
        with open(js, encoding="utf-8") as f:
            got = json.load(f)
        s = Set(d, "v41")
        n_l = n_plan("v41")
        seed = seed_lists("v41", s, "prefix", root)
        X = s.stack(family("v41")["eligible"])
        rule = Rule("mid")
        alone, _ = gen_values(X, [spans[0], spans[2]], None, 2, seed, n_l, 384, rule, family("v41")["open_m"], 0, 1,
                              COPIES)
        strip = lambda r: {k: v for k, v in r.items() if k != "context"}
        assert [r["context"] for r in got["contexts"]] == [0, 2], got["contexts"]
        assert [strip(r) for r in got["contexts"]] == [strip(r) for r in alone], (got["contexts"], alone)
        per = replay(X, seed_resident(seed, n_l, 384), n_l, 384, rule, sem="flip", spares=1, in_flight_cap=True, d=1,
                     link=Budget(COPIES)).per_row
        want = float(np.concatenate([per[t0 + p:t1] for t0, t1, p in spans]).mean())
        assert got["steady"] == want and got["skipped"] == [dict(context=1, positions=3, need=4)], got
        # --open-from is bounded by the contexts that keep their arms: the skipped one's prompt 2 does not refuse
        # last:3, which the two kept prompts (4 and 3) hold; last:4 is past the kept prompt 3
        with redirect_stderr(io.StringIO()) as e3, redirect_stdout(io.StringIO()):
            code = main(argv + ["--short", "skip", "--open-from", "last:3"])
        assert code == 0, e3.getvalue()
        with redirect_stderr(io.StringIO()) as e4, redirect_stdout(io.StringIO()):
            code = main(argv + ["--short", "skip", "--open-from", "last:4"])
        assert code == 1 and "within 1..3" in e4.getvalue(), e4.getvalue()


def q38_engine_set(root, name, rows, chunk=None, complete=True, n_expert=512):
    """A set in the engine route trace's shape (crates/gpu/src/host/route_trace.rs): header lines by
    key, `layer` rows with a slots column, a `call` row per prompt, `# chunk` when given and
    slots-<layer>.u8 beside the topk files (their kinds unread here, so all H)."""
    d = os.path.join(root, name)
    os.makedirs(d)
    T = len(next(iter(rows.values())))
    K = len(next(iter(rows.values()))[0])
    L = sorted(rows)
    for l, rs in rows.items():
        np.asarray(rs, dtype="<u2").tofile(os.path.join(d, f"topk-{l}.u16"))
        with open(os.path.join(d, f"slots-{l}.u8"), "wb") as f:
            f.write(b"H" * (T * K))
    lines = [
        "# router_trace — bloomery engine: the routed ids of every position the engine ran, in step "
        "order, as the host step port read them",
        "# model\t/models/Qwen3.8-Flash-Next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
        "# model_file\tQwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
        "# arch\tqwen4exp",
        "# build\tgenerate_qwen3moe",
        "# engine\tbloomery (not an ik build)",
        f"# tokens\t{T}",
        f"# n_expert\t{n_expert}",
        f"# n_expert_used\t{K}",
        f"# n_layer\t{len(L)}",
        "# feed\tone step per position: a prompt call's ids one step each (call rows)",
        "# slots\tslots-<layer>.u8 beside topk-<layer>.u16: C stage card, T tier card, H host",
        "# place\ta",
        "# experts\tcard",
        "# prefill\tstep",
    ]
    if chunk:
        lines.append(f"# chunk\t{chunk}")
    lines.append("# layer\tlayer\ttokens\tfile\tslots")
    lines += [f"layer\t{l}\t{T}\ttopk-{l}.u16\tslots-{l}.u8" for l in L]
    lines.append("# call\tindex\tfirst\tprompt_call\tend\tpos0")
    lines.append(f"call\t0\t0\t{min(8, T)}\t{T}\t0")
    lines.append(f"# positions\t{T}")
    if complete:
        lines.append(f"# complete\t{T}\t{len(L)}")
    with open(os.path.join(d, "MANIFEST.tsv"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return d


def case_q38_family():
    # The q38 row's facts (Qwen3.8-Flash-Next: 512 x top-10, the 43 card layers of 48, the card plan's
    # 12,841 experts spread 299 on 27 layers and 298 on 16) and the engine trace shape its sets come
    # in: a set the engine wrote (a slots column, a call row, a chunk line) opens as family q38 and
    # gen replays it by its chunk, the values gen_values gives.
    F = family("q38")
    assert (F["n_expert"], F["n_used"], F["expert_bytes"]) == (512, 10, 3_072_000)
    assert F["eligible"] == [l for l in range(48) if l not in (2, 4, 30, 46, 47)]
    n_l = n_plan("q38")
    assert n_l == [299] * 27 + [298] * 16 and sum(n_l) == 12_841
    assert n_l == list(spread(12_841, F["eligible"]).values())
    with tempfile.TemporaryDirectory() as root:
        rows = {l: [tuple((7 * t + l + 50 * j) % 512 for j in range(10)) for t in range(17)]
                for l in range(48)}
        d = q38_engine_set(root, "q38trace", rows, chunk=17)
        s = Set(d, "q38")
        assert s.chunk == 17 and s.layers == list(range(48)) and s.T == 17
        js = os.path.join(root, "gen.json")
        out = io.StringIO()
        with redirect_stderr(io.StringIO()) as err:
            code = main(["gen", "q38", d, "--data", root, "--seed", "prefix", "--prompt", "8",
                         "--window", "4", "--json", js], out)
        assert code == 0, err.getvalue()
        text = out.getvalue()
        assert "contexts=1 x 17 prompt=8 window=4" in text and "n_l 298..299 on 43 layers" in text, text
        with open(js, encoding="utf-8") as f:
            got = json.load(f)
        X = s.stack(F["eligible"])
        want, steady = gen_values(X, [(0, 17, 8)], 8, 4, seed_lists("q38", s, "prefix", root),
                                  n_l, 512, Rule("mid"), None, 0, 1, COPIES)
        assert got["contexts"] == want and got["steady"] == steady, (got["contexts"], want)
        # refusals: a set that lacks an eligible layer, and one of another experts x top-k
        short = q38_engine_set(root, "q38short", {l: rows[l] for l in range(48) if l != 5}, chunk=17)
        narrow = q38_engine_set(
            root, "q38narrow",
            {l: [tuple((7 * t + l + 50 * j) % 256 for j in range(10)) for t in range(17)]
             for l in range(48)}, n_expert=256)
        for path, why in ((short, "family q38's eligible layers [5] are not in the set"),
                          (narrow, "256 experts x top-10, family q38 is 512 x top-10"),
                          (d, "hit evaluates [24576, tokens)")):
            err = io.StringIO()
            with redirect_stderr(err), redirect_stdout(io.StringIO()):
                code = main(["hit", "q38", path, "--data", root])
            assert code == 1 and why in err.getvalue(), (path, err.getvalue())


def case_fixture():
    # one layer, E=4, n_l 1 seeded {0}; each pass two rows, 2 then 1, only the first kept: 1 is never
    # counted (counted, it would tie 2 at 4 and win by the lower id). Boundary 4: 2 in, 0 out, live 5.
    rule = Rule("mid")
    X = X_of([[(2,)], [(1,)]] * 8)
    passes = [(2 * p, 2 * p + 2, 1) for p in range(8)]
    flips = []
    replay(X, np.array([[True, False, False, False]]), [1], 4, rule, passes=passes, link=Budget(None),
           on_flip=lambda b, i, o, n, live: flips.append((b, i, o, n, live)))
    assert flips == [(4, 0, 0, 2, 5)], flips
    # the file: the same bytes for the same command (pinned), and the tie counts the order is checked with
    fx = make_fixture(40, 1)
    text = fixture_text(fx, "tools/ref/router-residency.py fixture --out x --passes 40")
    assert json.loads(text)["header"]["command"].endswith("--passes 40")
    digest = hashlib.md5(text.encode()).hexdigest()
    assert digest == FIXTURE_40_MD5, digest
    big = make_fixture(400, 1)
    assert min(big["header"]["ties"].values()) > 0, big["header"]["ties"]
    assert big["header"]["cap_bound"] == 0 < big["cap_case"]["header"]["cap_bound"], big["cap_case"]["header"]
    # pinned: layer 0 seeded {0, 1}, 0 pinned; 2 hot, then 3: without the pin 0 (no use) leaves first,
    # with it 1 does, and the second flip finds only 2 (just admitted) and 0 (pinned) -> 2 leaves for 3.
    X = X_of([[(2,)]] * 4 + [[(3,)]] * 12)
    res = np.array([[True, True, False, False]])
    pin = np.array([[True, False, False, False]])
    for p, want in ((None, [(4, 0, 0, 2, 12), (12, 0, 1, 3, 20)]), (pin, [(4, 0, 1, 2, 12), (12, 0, 2, 3, 20)])):
        flips = []
        replay(X, res, [2], 4, rule, d=8, link=Budget(None), pinned=p,
               on_flip=lambda b, i, o, n, live: flips.append((b, i, o, n, live)))
        assert flips == want, (p is not None, flips)
    fp = make_fixture(400, 1, pinned=8)
    assert fp["params"]["pinned"] == [8, 8, 8] and fp["header"]["pinned_moved"] > 0, fp["header"]
    assert all(f["out"] not in fp["seed"][f["layer"]][:8] for f in fp["flips"] + fp["cap_case"]["flips"])
    # away: the same trace with 2 away: it is never admitted, and 3 goes in at the first boundary it clears
    # the margin (8), not after 2's flip lands.
    far = np.array([[False, False, True, False]])
    flips = []
    replay(X, res, [2], 4, rule, d=8, link=Budget(None), away=far,
           on_flip=lambda b, i, o, n, live: flips.append((b, i, o, n, live)))
    assert flips == [(8, 0, 0, 3, 16)], flips
    fa = make_fixture(400, 1, away=[[24], [40], [62]])
    assert fa["params"]["away"] == [[24], [40], [62]] and fa["header"]["away_moved"] > 0, fa["header"]
    assert all(f["in"] not in fa["params"]["away"][f["layer"]] for f in fa["flips"] + fa["cap_case"]["flips"])
    digest = hashlib.md5(fixture_text(fa, "tools/ref/router-residency.py fixture --away 24/40/62 --out x")
                         .encode()).hexdigest()
    assert digest == AWAY_FIXTURE_MD5, digest
    for bad, why in (([[0], [], []], "in its layer's seed"), ([[24, 24], [], []], "twice"), ([[64], [], []], "experts"),
                     ([[24], [40]], "layers")):
        try:
            make_fixture(40, 1, away=bad)
        except ToolError as e:
            assert why in str(e), (bad, e)
        else:
            raise AssertionError(f"--away {bad} was taken")
    assert away_lists("24,25//62") == [[24, 25], [], [62]]
    fo = make_open_fixture(1)
    digest = hashlib.md5(fixture_text(fo, "tools/ref/router-residency.py fixture --open --out x").encode()).hexdigest()
    assert digest == OPEN_FIXTURE_MD5, digest
    assert min(fo["header"]["ties"].values()) > 0, fo["header"]["ties"]


FIXTURE_40_MD5 = "267b9a38beeb1f472a38285660045978"
OPEN_FIXTURE_MD5 = "1791eb13350ebdac4eb72e234e68ec13"
AWAY_FIXTURE_MD5 = "eee3dd7afdaae47a14533799e21ed0c0"


def case_in_flight():
    # one layer, E=4, n_l 2 seeded {0, 1}, mid rule, d 8. Boundary 4: 2 (4 uses) in for 0, live 12. Boundary 8:
    # 3 clears 1 by the margin, but the layer's one spare is in flight -> blocked. Boundary 12: the flip live
    # at 12 lands first and frees the spare: 3 in for 1, live 20. Counted per boundary only, 3 goes in at 8.
    rule = Rule("mid")
    X = X_of([[(2,)]] * 4 + [[(3,)]] * 12)
    res = np.array([[True, True, False, False]])
    for cap, want in ((True, [(4, 0, 0, 2, 12), (12, 0, 1, 3, 20)]), (False, [(4, 0, 0, 2, 12), (8, 0, 1, 3, 16)])):
        flips = []
        replay(X, res, [2], 4, rule, d=8, in_flight_cap=cap, link=Budget(None),
               on_flip=lambda b, i, o, n, live: flips.append((b, i, o, n, live)))
        assert flips == want, (cap, flips)


def case_land_then_plan():
    # crates/runtime's flips_land_before_the_pass_plans: one layer, E=4, n_l 1 seeded {0}, every 2, d 2, decay 0,
    # steps 2 2 3 3. Boundary 2: 2 in for 0, live 4. Boundary 4: that flip lands first, so 2 is resident, the
    # spare is free and 3 (2 uses) goes in for 2 (0 uses), live 6. Planned before the landing (the window
    # scripts' order) 0 is still leaving and 2 still pending: no victim, no second flip.
    rule = Rule("strata", every=2, decay=0.0)
    X = X_of([[(2,)], [(2,)], [(3,)], [(3,)]])
    for cap, want in ((True, [(2, 0, 0, 2, 4), (4, 0, 2, 3, 6)]), (False, [(2, 0, 0, 2, 4)])):
        flips = []
        replay(X, np.array([[True, False, False, False]]), [1], 4, rule, d=2, in_flight_cap=cap, link=Budget(None),
               on_flip=lambda b, i, o, n, live: flips.append((b, i, o, n, live)))
        assert flips == want, (cap, flips)


def case_refusals():
    with tempfile.TemporaryDirectory() as root:
        d = wu._fake_set(root, "tiny", {0: [(0, 1)] * 8}, 4, 2)
        cases = [
            (["hit", "v41", "nosuch", "--data", root], "no router set 'nosuch'"),
            (["hit", "qwen9", "--data", root], "family 'qwen9' is not in the table"),
            (["hit", "v41", "prose", "--data", root, "--open", "400"], "--open needs --window"),
            (["hit", "v41", "prose", "--data", root, "--window", "96", "--open-from", "last:512"],
             "--open-from needs --open"),
            (["hit", "v41", d, "--data", root], "384 x top-6"),
            (["gen", "v41", d, "--data", root], "gen needs --prompt"),
            (["hit", "v41", "prose"], "no --data DIR"),
        ]
        os.makedirs(os.path.join(root, "nomanifest"))
        cases.append((["hit", "v41", "nomanifest", "--data", root], "no manifest"))
        for argv, want in cases:
            err = io.StringIO()
            with redirect_stderr(err), redirect_stdout(io.StringIO()):
                code = main(argv)
            assert code != 0 and want in err.getvalue(), (argv, code, err.getvalue())


def fixture_sets(root, specs):
    """v41-shaped router sets from the fixture's trace: per (name, lcg seed, eval positions), make_fixture(400)'s
    rows in pass order, tiled to SPLIT + eval positions; v41 layer i routes fixture layer i % 3's ids plus
    64 ((i // 3) % 6), so every layer is distinct and every id is below 384."""
    for name, lcg, ev in specs:
        trace = make_fixture(400, lcg)["trace"]
        seq = np.asarray([[row for p in trace[l] for row in p] for l in range(3)], dtype=np.int64)  # [3, R, 6]
        T = SPLIT + ev
        seq = seq[:, np.arange(T) % seq.shape[1]]
        d = os.path.join(root, name)
        os.makedirs(d)
        with open(os.path.join(d, "MANIFEST.tsv"), "w", encoding="utf-8") as f:
            f.write("# router_trace — test\n# model\t/m/m.gguf\n# model_file\tm.gguf\n# build\tb0\n")
            f.write(f"# tokens\t{T}\n# n_expert\t384\n# n_expert_used\t6\n")
            f.write("# layer\tlayer\tsource\tproducer\ttokens\tid_sum\tignored\tfile\n")
            for l in range(40):
                f.write(f"layer\t{l}\tx\tx\t{T}\t0\t0\ttopk-{l}.u16\n")
            f.write(f"# complete\t{T}\t40\n")
        for l in range(40):
            (seq[l % 3] + 64 * ((l // 3) % 6)).astype("<u2").tofile(os.path.join(d, f"topk-{l}.u16"))


def hit_json(argv, root):
    js = os.path.join(root, "hit.json")
    out, err = io.StringIO(), io.StringIO()
    with redirect_stderr(err):
        code = main(argv + ["--data", root, "--json", js], out)
    assert code == 0, (argv, err.getvalue())
    with open(js, encoding="utf-8") as f:
        return json.load(f), out.getvalue()


def case_streams_one():
    # B = 1 through --streams is the plain hit, every column of every row (static over cross, in and insample,
    # the adaptive rule over cross and in, belady), with the streams' columns beside them and no other row.
    with tempfile.TemporaryDirectory() as root:
        fixture_sets(root, (("prose", 1, 240), ("code", 2, 200), ("korean", 3, 240)))
        plain, _ = hit_json(["hit", "v41", "prose", "--cap", "n16", "--policies", "static,adaptive,belady"], root)
        merged, text = hit_json(["hit", "v41", "--streams", "prose", "--cap", "n16"], root)
        assert len(plain) == 6 and len(merged) == len(plain), (len(plain), len(merged))
        for p, m in zip(plain, merged):
            assert m["role"] == "pooled" and m["rows_pass"] == 1, m
            assert {k: m[k] for k in p} == p, (p, m)
        assert "s0 prose [24576, 24816)" in text and "alone" not in text, text


def case_streams_twice():
    # B = 2 of one fixture set at offset 0: both streams are the whole eval half, so every pass counts each id
    # twice. Doubled counts against margin 3 and min_count 2 are the single stream's counts against 1.5 and 1
    # (decay and the gain order scale with them, exactly in binary), and the kept-row clock's every 4 over
    # passes of 2 kept rows plans every 2 passes, so the single stream's twin runs every 2 — so each stream's
    # adaptive row is that replay's, and differs from the stream alone (the case tells two counts from one).
    with tempfile.TemporaryDirectory() as root:
        fixture_sets(root, (("prose", 1, 240),))
        rows, text = hit_json(["hit", "v41", "--streams", "prose,prose", "--offset", "0", "--cap", "n16", "--seed", "in",
                               "--policies", "static,adaptive"], root)
        assert "s0 prose [24576, 24816)  s1 prose [24576, 24816)" in text, text
        by = {(r["policy"], r["role"]): r for r in rows}
        s = Set(os.path.join(root, "prose"), "v41")
        F = family("v41")
        X = hit_eval_stack(s, F)
        n_l = [16] * len(F["eligible"])
        seed = seed_lists("v41", s, "in", root)
        r = replay(X, seed_resident(seed, n_l, 384), n_l, 384, Rule("mid", every=2, margin=1.5, min_count=1.0),
                   sem="hole", d=0, link=Link(F["expert_bytes"], B_PIN, F["step_ms"]))
        want = r.hits_layer.sum() / (X.shape[0] * X.shape[1] * X.shape[2])
        for pol in ("static", "adaptive"):
            s0, s1, pooled = by[(pol, "s0")], by[(pol, "s1")], by[(pol, "pooled")]
            strip = lambda q: {k: v for k, v in q.items() if k != "role"}
            assert strip(s0) == strip(s1) and pooled["hit"] == s0["hit"], (s0, s1, pooled)
        assert by[("adaptive", "s0")]["hit"] == want and by[("adaptive", "s0")]["swaps_tok"] == r.swaps / X.shape[0], \
            (by[("adaptive", "s0")], want, r.swaps)
        assert by[("adaptive", "s0")]["hit"] != by[("adaptive", "alone s0")]["hit"], by[("adaptive", "alone s0")]
        assert by[("static", "s0")]["hit"] == by[("static", "alone s0")]["hit"]
        assert "merged prose+prose mid[in]: pooled" in text, text


def case_streams_windows():
    # A set named twice is two adjacent disjoint windows of its eval half by default, K = floor(240 / 2); unequal
    # lengths are truncated to the shortest and the line names each cut stream.
    with tempfile.TemporaryDirectory() as root:
        fixture_sets(root, (("prose", 1, 240), ("code", 2, 200)))
        rows, text = hit_json(["hit", "v41", "--streams", "prose,prose", "--cap", "n16", "--seed", "in",
                               "--policies", "static"], root)
        assert "s0 prose [24576, 24696)  s1 prose [24696, 24816)" in text, text
        assert [r["window"] for r in rows if r["role"] in ("s0", "s1")] == [[24576, 24696], [24696, 24816]], rows
        rows, text = hit_json(["hit", "v41", "--streams", "prose,code", "--cap", "n16", "--seed", "in",
                               "--policies", "static"], root)
        assert "truncated to the shortest stream, 200 passes: s0 prose 240 -> 200" in text, text
        assert {r["tokens"] for r in rows} == {200}, rows


def case_streams_refusals():
    with tempfile.TemporaryDirectory() as root:
        fixture_sets(root, (("prose", 1, 240), ("code", 2, 240), ("korean", 3, 240)))
        cases = [
            (["hit", "v41", "--streams", "prose,threads"], "'threads' is not a set of family v41"),
            (["hit", "v41", "--streams", ",".join(["prose"] * 9)], "9 streams, at most 8"),
            (["hit", "v41", "--streams", "prose,prose", "--offset", "121"], "past its eval half of 240"),
            (["hit", "v41", "--streams", "prose,code", "--offset", "5"], "every stream names another set"),
            (["hit", "v41", "--streams", "prose", "--policies", "lru"], "lru has no rows-a-pass form"),
            (["hit", "v41", "prose", "--streams", "prose"], "drop the positional sets"),
            (["hit", "v41", "prose", "--offset", "3"], "--offset needs --streams"),
            (["hit", "v41", "--streams", "prose,code,korean", "--seed", "cross"], "every set of family v41 is a stream"),
            (["hit", "q38", "--streams", "prose"], "is not a set of family q38 (its table names no set)"),
        ]
        for argv, want in cases:
            err = io.StringIO()
            with redirect_stderr(err), redirect_stdout(io.StringIO()):
                code = main(argv + ["--data", root, "--cap", "n16"])
            assert code == 1 and want in err.getvalue(), (argv, code, err.getvalue())


def case_turn_refusals():
    # C1: --turn is a --streams modifier — without --streams refused by name, and so is K < 1
    with tempfile.TemporaryDirectory() as root:
        fixture_sets(root, (("prose", 1, 240),))
        cases = [
            (["hit", "v41", "prose", "--turn", "3"], "--turn needs --streams"),
            (["hit", "v41", "--streams", "prose", "--turn", "0"], "--turn 0: at least 1 pass a stream"),
            (["hit", "v41", "--streams", "prose", "--turn", "-2"], "--turn -2: at least 1 pass a stream"),
        ]
        for argv, want in cases:
            err = io.StringIO()
            with redirect_stderr(err), redirect_stdout(io.StringIO()):
                code = main(argv + ["--data", root, "--cap", "n16"])
            assert code == 1 and want in err.getvalue(), (argv, code, err.getvalue())


def case_turn_one_stream():
    # C3: --turn K with one stream is the plain hit on that set's window — the turn order over one stream
    # is the token order, and one row a pass is the plain hit's own pass, every number bit for bit
    with tempfile.TemporaryDirectory() as root:
        fixture_sets(root, (("prose", 1, 240), ("code", 2, 200), ("korean", 3, 240)))
        plain, _ = hit_json(["hit", "v41", "prose", "--cap", "n16", "--policies", "static,adaptive,belady"], root)
        turned, text = hit_json(["hit", "v41", "--streams", "prose", "--turn", "7", "--cap", "n16"], root)
        assert len(plain) == len(turned) == 6, (len(plain), len(turned))
        for p, m in zip(plain, turned):
            assert m["role"] == "pooled" and m["rows_pass"] == 1 and m["turn"] == 7, m
            assert {k: m[k] for k in p} == p, (p, m)  # every plain column, bit for bit
            assert {"role", "streams", "rows_pass", "turn"} <= set(m), m
        assert "B=1 turn 7" in text and "turn=7" in text, text


def case_turn_blocks():
    # C5: two streams whose hot layers (2..5) route disjoint 6-id sets, card n6 = one stream's set. The
    # pooled `in` seed sums both learn halves, whose counts tie, so the card starts at {0..5} (the lower
    # ids). Under --turn 1 the rule counts both streams alike every window: pb's candidates tie pa's
    # residents forever, never clear strata's margin, and pb runs its whole window at 0/6 on those
    # layers. Under --turn W pa's whole block runs first (pb's ids count 0: no flip), then pb's, where
    # 64..69 clear the margin over the victims' decayed counts at boundary w_full and serve 6/6 from
    # pass 4 w_full of pb's block.
    W, L, HOT = 240, 38, 4
    with tempfile.TemporaryDirectory() as root:
        for name, hot in (("prose", (0, 1, 2, 3, 4, 5)), ("code", (64, 65, 66, 67, 68, 69))):
            T = SPLIT + W
            d = os.path.join(root, name)
            os.makedirs(d)
            with open(os.path.join(d, "MANIFEST.tsv"), "w", encoding="utf-8") as f:
                f.write("# router_trace — test\n# model\t/m/m.gguf\n# model_file\tm.gguf\n# build\tb0\n")
                f.write(f"# tokens\t{T}\n# n_expert\t384\n# n_expert_used\t6\n")
                f.write("# layer\tlayer\tsource\tproducer\ttokens\tid_sum\tignored\tfile\n")
                for l in range(40):
                    f.write(f"layer\t{l}\tx\tx\t{T}\t0\t0\ttopk-{l}.u16\n")
                f.write(f"# complete\t{T}\t40\n")
            for l in range(40):
                row = hot if 2 <= l <= 5 else (0, 1, 2, 3, 4, 5)
                np.asarray([row] * T, dtype="<u2").tofile(os.path.join(d, f"topk-{l}.u16"))
        argv = ["hit", "v41", "--streams", "prose,code", "--cap", "n6", "--seed", "in", "--policies", "adaptive",
                "--rule", "strata", "--link-gbps", "0"]
        # the rule's own arithmetic on the fixture, not a run: pa's block takes 0..5's counts to their
        # decay steady state (4 a window, x0.7 at the boundary); in pb's block 64..69's accumulated count
        # clears the victims' decayed one plus strata's margin 1.5 first at boundary w_full
        v = 0.0
        for _ in range(W // 4):
            v = (v + 4.0) * 0.7
        c, w_full = 0.0, None
        for w in range(1, W // 4 + 1):
            if c + 4.0 >= v + 1.5:
                w_full = w
                break
            c = (c + 4.0) * 0.7
            v *= 0.7
        assert w_full == 3, w_full
        t1, text1 = hit_json(argv + ["--turn", "1"], root)
        tw, textw = hit_json(argv + ["--turn", str(W)], root)
        by1 = {(r["policy"], r["role"]): r for r in t1}
        byw = {(r["policy"], r["role"]): r for r in tw}
        slots, a_hits = 2 * W * L * 6, W * L * 6
        b1 = W * (L - HOT) * 6                               # turn 1: pb never gains a hot-layer slot
        bw = W * (L - HOT) * 6 + HOT * 6 * (W - 4 * w_full)   # turn W: 0/6 for 4 w_full passes, then 6/6
        assert by1[("adaptive", "pooled")]["hit"] == (a_hits + b1) / slots, by1[("adaptive", "pooled")]
        assert byw[("adaptive", "pooled")]["hit"] == (a_hits + bw) / slots, byw[("adaptive", "pooled")]
        assert byw[("adaptive", "pooled")]["hit"] > by1[("adaptive", "pooled")]["hit"]
        assert by1[("adaptive", "s0")]["hit"] == byw[("adaptive", "s0")]["hit"] == 1.0
        assert by1[("adaptive", "s1")]["hit"] == b1 / (W * L * 6), by1[("adaptive", "s1")]
        assert byw[("adaptive", "s1")]["hit"] == bw / (W * L * 6), byw[("adaptive", "s1")]
        assert by1[("adaptive", "pooled")]["swaps_tok"] == 0.0, by1[("adaptive", "pooled")]
        assert byw[("adaptive", "pooled")]["swaps_tok"] == HOT * 6 / (2 * W), byw[("adaptive", "pooled")]
        assert "B=2 turn 1" in text1 and f"B=2 turn {W}" in textw, (text1, textw)
        assert "turn 1 prose+code strata[in,gbps=0]: pooled" in text1, text1


CASES = [case_static, case_belady, case_lru, case_adaptive_link, case_adaptive_margin, case_adaptive_min_count,
         case_adaptive_cap, case_lru_is_global, case_in_flight, case_land_then_plan, case_window, case_open,
         case_open_last, case_gen, case_gen_contexts, case_gen_split, case_gen_short, case_q38_family,
         case_fixture, case_refusals, case_streams_one, case_streams_twice, case_streams_windows,
         case_streams_refusals, case_turn_refusals, case_turn_one_stream, case_turn_blocks]


def self_test():
    failed = []
    for case in CASES:
        try:
            case()
        except AssertionError as e:
            failed.append(case.__name__)
            print(f"FAIL {case.__name__}: {e!r}", file=sys.stderr)
    if failed:
        print(f"router-residency: self-test FAILED: {', '.join(failed)}", file=sys.stderr)
        return 1
    print(f"router-residency: self-test ok ({len(CASES)} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
