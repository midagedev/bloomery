#!/usr/bin/env python3
"""Decode residency policies replayed over router sets: which selected experts a card holds, step by step.

    tools/ref/router-residency.py replay <set> --n-l <spec> [--policy static|strata|lfu|lru|belady]
                                  [--seed in|insample|prefix|cross|hot:<file>] [--cross <set>...]
                                  [--split auto|<tokens>] [--score LO-HI] [--buckets a-b,c-d,...]
                                  [--every 4] [--cap 24] [--margin 3] [--decay 0.9] [--min-count 2]
                                  [--expert-bytes B] [--step-ms S] [--gbps 26.28] [--pageable-gbps 21.16]
                                  [--crossings 3] [--fill all|budget]
    tools/ref/router-residency.py --self-test

A set is a router_trace directory (tools/ref/router-coverage.py reads it and refuses one without its
`# complete` trailer). One token is one decode step. The steps [split, T) are replayed in order and scored;
the lists a policy starts from are learned on [0, split). `--split auto` (the default) is the chunk boundary
nearest T / 2 (router-coverage.py's held_split); `--split 0` replays and scores every token, for a set
whose card list comes from elsewhere (`--seed prefix` or `hot:`). Residency carries across chunk edges,
as a server's does across requests.

The card's per-layer capacity n_l is window-union.py's `--n-l` (explicit, or `spread:<n>@<a>-<b>`, the
placement's spread rule); a layer without a count holds nothing and is not scored. A hit is a selected
slot whose expert is resident when the step starts; hit % is over the scored layers' selections.

Seeds (the list each layer starts from, hottest first): `in` its counts on [0, split); `insample` its
counts on the scored steps (the in-sample bound); `prefix` ids 0.. (the placement without a hot list);
`cross` the summed counts of the `--cross` sets (every token of each); `hot:<file>` a
tools/ref/router-hotlist.py file. Every set named must come from one model file (the manifests'
`# model` lines, the first shard's full path): two quantizations of one model share shard names.

Policies:
  static   the seed's first n_l, never moved
  strata   Strata's swap rule: every `--every` steps, per layer pair the hottest missing experts
           (decayed count >= --min-count) with the least-used residents, swap when count >= victim +
           --margin, at most --cap swaps a pass over all layers (largest margin first), then multiply
           every count by --decay. The defaults are the setting ideaverify adopted ("mid"). An admitted
           expert serves from the first step that starts after its copy ends; the victim leaves at once
  lfu      the same rule every step with no threshold or margin, at most the link's budget a step
           (floor(step_ms x gbps / expert_bytes)), decay 0.7^(1/4) a step
  lru      one capacity sum(n_l) shared by every layer; `--fill all` admits every miss, `budget` at most
           the link's budget a token, in layer order (FreeToken's global LRU)
  belady   per layer, Belady's MIN with bypass and free swaps: the ceiling of any demand policy at n_l

The link model: one copy stream of --expert-bytes at --gbps, serialized (strata and lfu wait on it; lru
and belady do not). Every swap is priced: MB a token, ms a token at the pinned and the pageable rates,
the pinned link's share of a --step-ms step, and host DRAM GB/s at --crossings passes a copied byte (3:
this box refuses to register the file mapping, so a copy is staged — tools/ref/dma-dram-share.sh
measures the passes). --expert-bytes and --step-ms have no default: they are the model's.

`--score LO-HI` scores only the positions LO..HI (inclusive) inside each chunk; `--buckets` prints the hit
per position bucket inside a chunk. Both need a per-slot record: static, strata and lfu.
"""
import heapq
import importlib.util
import math
import os
import sys
import tempfile
from collections import OrderedDict

try:
    import numpy as np
except ImportError:
    sys.exit("router-residency: numpy is required (the Mac's python3 has it; the box's system python3 "
             "does not)")

_here = os.path.dirname(os.path.abspath(__file__))


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_here, file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rc = _load("router_coverage", "router-coverage.py")
wu = _load("window_union", "window-union.py")

# Strata's rule at the setting ideaverify adopted (every 4 steps, cap 24, margin 3, decay 0.9; the
# threshold stayed at 2).
MID = dict(every=4, min_count=2.0, margin=3.0, cap=24, decay=0.9)
PINNED_GBPS = 26.28    # A6000 pinned H2D, docs/facts.md
PAGEABLE_GBPS = 21.16  # A6000 pageable H2D, the same line
CROSSINGS = 3.0
POLICIES = ("static", "strata", "lfu", "lru", "belady")
RECORDED = ("static", "strata", "lfu")


def rank(counts):
    """Hottest first, ties to the lower id (router-coverage.py's `hot`)."""
    counts = np.asarray(counts)
    return np.lexsort((np.arange(len(counts)), -counts))


def static_hits(X, lists, n_l, record=None):
    """X [T, L, K]; lists[i] ranked ids for layer i; n_l[i] capacity. Hits per layer."""
    T, L, K = X.shape
    hits = np.zeros(L, dtype=np.int64)
    for i in range(L):
        E = max(int(X[:, i].max()) + 1, len(lists[i]))
        res = np.zeros(E, dtype=bool)
        res[np.asarray(lists[i][: n_l[i]], dtype=np.int64)] = True
        hr = res[X[:, i]]
        if record is not None:
            record[:, i] = hr
        hits[i] = hr.sum()
    return hits


class Link:
    """A serialized copy stream at gbps; issue() returns the step from which a copy issued now serves."""

    def __init__(self, bytes_per_copy, gbps, step_ms):
        self.copy_ms = bytes_per_copy / (gbps * 1e6) if gbps else 0.0
        self.step_ms = step_ms
        self.free_ms = 0.0
        self.busy_ms = 0.0

    def issue(self, now_ms, now_step):
        if self.copy_ms == 0.0:
            return now_step + 1
        start = max(self.free_ms, now_ms)
        self.free_ms = start + self.copy_ms
        self.busy_ms += self.copy_ms
        return max(now_step + 1, math.ceil(self.free_ms / self.step_ms - 1e-9))


def adaptive(X, seed_lists, n_l, E, *, every=MID["every"], min_count=MID["min_count"], margin=MID["margin"],
             cap=MID["cap"], decay=MID["decay"], link=None, record=None):
    """Strata's swap rule (module doc). Returns (hits per layer, swaps)."""
    T, L, K = X.shape
    ar = np.arange(L)[:, None]
    resident = np.zeros((L, E), dtype=bool)
    for i in range(L):
        resident[i, np.asarray(seed_lists[i][: n_l[i]], dtype=np.int64)] = True
    inflight = np.zeros((L, E), dtype=bool)
    counts = np.zeros((L, E))
    pending = []  # heap of (ready step, layer, expert)
    hits = np.zeros(L, dtype=np.int64)
    swaps = 0
    width = min(max(n_l), E)
    for s in range(T):
        while pending and pending[0][0] <= s:
            _, i, e = heapq.heappop(pending)
            inflight[i, e] = False
            resident[i, e] = True
        sel = X[s]
        hr = resident[ar, sel]
        if record is not None:
            record[s] = hr
        hits += hr.sum(axis=1)
        counts[ar, sel] += 1.0
        if (s + 1) % every:
            continue
        cand = np.where(resident | inflight, -1.0, counts)
        vict = np.where(resident, counts, np.inf)
        cidx = np.argsort(-cand, axis=1, kind="stable")[:, :width]
        vidx = np.argsort(vict, axis=1, kind="stable")[:, :width]
        cval = np.take_along_axis(cand, cidx, axis=1)
        vval = np.take_along_axis(vict, vidx, axis=1)
        ok = (cval >= min_count) & (cval >= vval + margin) & np.isfinite(vval)
        li, jj = np.nonzero(ok)
        if li.size:
            order = np.argsort(-(cval[li, jj] - vval[li, jj]), kind="stable")[:cap]
            now_ms = (s + 1) * (link.step_ms if link else 1.0)
            for o in order:
                i, j = li[o], jj[o]
                resident[i, vidx[i, j]] = False
                inflight[i, cidx[i, j]] = True
                ready = link.issue(now_ms, s) if link else s + 1
                heapq.heappush(pending, (ready, i, cidx[i, j]))
                swaps += 1
        counts *= decay
    return hits, swaps


def global_lru(X, seed_lists, n_l, *, fill="all", budget=None):
    """FreeToken's global LRU (module doc). Returns (hits per layer, fills)."""
    T, L, K = X.shape
    cache = OrderedDict()
    for i in range(L):
        for e in reversed(list(seed_lists[i][: n_l[i]])):
            cache[(i, int(e))] = None
    for key in list(cache)[::-1]:
        cache.move_to_end(key, last=False)
    C = sum(n_l)
    hits = np.zeros(L, dtype=np.int64)
    fills = 0
    for row in X.tolist():
        left = budget if fill == "budget" else None
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


def belady(X, seed_lists, n_l):
    """Per layer, Belady's MIN with bypass and free swaps. Returns (hits per layer, admits)."""
    T, L, K = X.shape
    hits = np.zeros(L, dtype=np.int64)
    admits = 0
    for i in range(L):
        cap = n_l[i]
        seq = X[:, i, :].tolist()
        nxt = [[T] * K for _ in range(T)]  # the next step that selects the expert of (s, k)
        last, first = {}, {}
        for s in range(T - 1, -1, -1):
            for k in range(K):
                nxt[s][k] = last.get(seq[s][k], T)
            for k in range(K):
                last[seq[s][k]] = s
                first[seq[s][k]] = s
        res, heap = {}, []
        for e in seed_lists[i][:cap]:
            e = int(e)
            res[e] = first.get(e, T)
            heapq.heappush(heap, (-res[e], e))
        for s in range(T):
            missed = []
            for k in range(K):
                e = seq[s][k]
                if e in res:
                    hits[i] += 1
                    res[e] = nxt[s][k]
                    heapq.heappush(heap, (-nxt[s][k], e))
                else:
                    missed.append((nxt[s][k], e))
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


def price(swaps, steps, expert_bytes, step_ms, gbps=PINNED_GBPS, pageable=PAGEABLE_GBPS, crossings=CROSSINGS):
    """The link model's columns for `swaps` copies over `steps` steps."""
    per_step = swaps * expert_bytes / steps  # bytes a step
    ms = per_step / (gbps * 1e6)
    return {"swaps": swaps / steps, "mb": per_step / 1e6, "ms_pinned": ms,
            "ms_pageable": per_step / (pageable * 1e6), "link": ms / step_ms,
            "dram_gbps": crossings * per_step / (step_ms * 1e6)}


def read_ids(s, layer):
    a = np.asarray(rc.read_topk(s, layer), dtype=np.int64)
    if a.size and a.max() >= s["n_expert"]:
        raise rc.SetError(f"{s['dir']}: layer {layer} names expert {a.max()} of {s['n_expert']}")
    return a.reshape(s["tokens"], s["n_used"])


def same_model(sets):
    models = {s["header"].get("model") for s in sets}
    if len(models) != 1 or None in models:
        raise rc.SetError("the sets come from different model files or name none: "
                          + ", ".join(f"{s['dir']}: {s['header'].get('model')}" for s in sets))


def parse_span(text, flag):
    a, sep, b = text.partition("-")
    try:
        lo, hi = int(a), int(b) if sep else int(a)
    except ValueError:
        raise rc.SetError(f"{flag} {text!r} is not LO-HI") from None
    if lo < 0 or hi < lo:
        raise rc.SetError(f"{flag} {text!r} runs backwards or below 0")
    return lo, hi


def seed_lists(kind, s, layers, split, cross, X_eval):
    if kind == "in":
        if split == 0:
            raise rc.SetError("--seed in learns on [0, split) and --split is 0: name another seed")
        return [rank(np.bincount(read_ids(s, l)[:split].ravel(), minlength=s["n_expert"])) for l in layers]
    if kind == "insample":
        return [rank(np.bincount(X_eval[:, i].ravel(), minlength=s["n_expert"])) for i in range(len(layers))]
    if kind == "prefix":
        return [np.arange(s["n_expert"]) for _ in layers]
    if kind == "cross":
        if not cross:
            raise rc.SetError("--seed cross needs --cross <set>")
        out = []
        for l in layers:
            c = np.zeros(s["n_expert"], dtype=np.int64)
            for o in cross:
                if l not in o["layers"] or o["n_expert"] != s["n_expert"]:
                    raise rc.SetError(f"{o['dir']}: no layer {l} of {s['n_expert']} experts")
                c += np.bincount(read_ids(o, l).ravel(), minlength=s["n_expert"])
            out.append(rank(c))
        return out
    if kind.startswith("hot:"):
        hot = wu.read_hot(kind[4:])
        missing = [l for l in layers if l not in hot]
        if missing:
            raise rc.SetError(f"{kind[4:]} has no line for layers {missing}")
        return [np.asarray(hot[l]) for l in layers]
    raise rc.SetError(f"--seed {kind!r}: in, insample, prefix, cross or hot:<file>")


def run(s, layers, n_l, policy, lists, X, opts, record=None):
    """(hits per layer, swaps) of one policy over X."""
    link = None
    if policy in ("strata", "lfu", "lru"):
        if opts["expert_bytes"] is None or opts["step_ms"] is None:
            raise rc.SetError(f"--policy {policy} moves experts: --expert-bytes and --step-ms price the link")
        link = Link(opts["expert_bytes"], opts["gbps"], opts["step_ms"])
    budget = None
    if link is not None:
        budget = int(opts["step_ms"] * opts["gbps"] * 1e6 // opts["expert_bytes"])
    if policy == "static":
        return static_hits(X, lists, n_l, record), 0
    if policy == "strata":
        kw = {k: opts[k] for k in MID}
        return adaptive(X, lists, n_l, s["n_expert"], link=link, record=record, **kw)
    if policy == "lfu":
        return adaptive(X, lists, n_l, s["n_expert"], every=1, min_count=1e-9, margin=1e-9, cap=budget,
                        decay=0.7 ** 0.25, link=link, record=record)
    if policy == "lru":
        return global_lru(X, lists, n_l, fill=opts["fill"], budget=budget if opts["fill"] == "budget" else None)
    return belady(X, lists, n_l)


def replay(set_dir, argv_opts):
    o = argv_opts
    s = rc.read_manifest(set_dir)
    cross = [rc.read_manifest(d) for d in o["cross"]]
    same_model([s] + cross)
    n_l_all = wu.parse_n_l(o["n_l"], s["layers"])
    layers = [l for l in s["layers"] if n_l_all[l] > 0]
    if not layers:
        raise rc.SetError(f"--n-l {o['n_l']} gives no layer of the set a card expert")
    n_l = [n_l_all[l] for l in layers]
    T = s["tokens"]
    split = rc.held_split(s) if o["split"] == "auto" else int(o["split"])
    if not 0 <= split < T:
        raise rc.SetError(f"--split {split} is outside [0, {T})")
    X = np.stack([read_ids(s, l)[split:] for l in layers], axis=1)
    lists = seed_lists(o["seed"], s, layers, split, cross, X)
    chunk = int(s["header"].get("chunk", 0) or 0) or T
    pos = (np.arange(split, T) % chunk)
    want_record = o["score"] is not None or o["buckets"]
    if want_record and o["policy"] not in RECORDED:
        raise rc.SetError(f"--score and --buckets need a per-slot record: {', '.join(RECORDED)}")
    record = np.zeros(X.shape, dtype=bool) if want_record else None
    hits, swaps = run(s, layers, n_l, o["policy"], lists, X, o, record)
    steps, K = X.shape[0], s["n_used"]
    name = os.path.basename(os.path.normpath(set_dir))
    print(f"# router-residency {name}: policy {o['policy']}, seed {o['seed']}, n_l {o['n_l']} "
          f"({sum(n_l)} card experts on {len(layers)} layers), steps [{split}, {T}) of chunk {chunk}, "
          f"model {s['header'].get('model', '?')}")
    if o["policy"] == "strata":
        print("# strata " + " ".join(f"{k}={o[k]}" for k in MID))
    if record is not None and o["score"] is not None:
        lo, hi = o["score"]
        keep = (pos >= lo) & (pos <= hi)
        if not keep.any():
            raise rc.SetError(f"--score {lo}-{hi} keeps no step")
        hit = record[keep].sum() / (keep.sum() * len(layers) * K)
        print(f"hit {100 * hit:.1f} % over positions {lo}-{hi} of each chunk ({int(keep.sum())} steps)")
    else:
        print(f"hit {100 * hits.sum() / (steps * len(layers) * K):.1f} % over {steps} steps")
    if o["buckets"]:
        print("bucket\tsteps\thit %")
        for lo, hi in o["buckets"]:
            keep = (pos >= lo) & (pos <= hi)
            n = int(keep.sum())
            cell = f"{100 * record[keep].sum() / (n * len(layers) * K):.1f}" if n else "-"
            print(f"{lo}-{hi}\t{n}\t{cell}")
    if o["policy"] != "static" and o["expert_bytes"] is not None and o["step_ms"] is not None:
        p = price(swaps, steps, o["expert_bytes"], o["step_ms"], o["gbps"], o["pageable_gbps"], o["crossings"])
        print(f"swaps/token {p['swaps']:.2f}  MB/token {p['mb']:.1f}  ms/token {p['ms_pinned']:.2f} @ "
              f"{o['gbps']} GB/s, {p['ms_pageable']:.2f} @ {o['pageable_gbps']}  link {100 * p['link']:.0f} % "
              f"of {o['step_ms']} ms  DRAM {p['dram_gbps']:.1f} GB/s at {o['crossings']:g} passes")
    return 0


def parse(argv):
    o = dict(MID, policy="strata", seed="in", cross=[], split="auto", score=None, buckets=[], n_l=None,
             expert_bytes=None, step_ms=None, gbps=PINNED_GBPS, pageable_gbps=PAGEABLE_GBPS,
             crossings=CROSSINGS, fill="all")
    num = {"--every": ("every", int), "--cap": ("cap", int), "--margin": ("margin", float),
           "--decay": ("decay", float), "--min-count": ("min_count", float),
           "--expert-bytes": ("expert_bytes", int), "--step-ms": ("step_ms", float), "--gbps": ("gbps", float),
           "--pageable-gbps": ("pageable_gbps", float), "--crossings": ("crossings", float)}
    sets = []
    it = iter(argv)
    for a in it:
        if a.startswith("--"):
            v = next(it, None)
            if v is None:
                raise rc.SetError(f"{a} takes a value")
            if a in num:
                key, conv = num[a]
                try:
                    o[key] = conv(v)
                except ValueError:
                    raise rc.SetError(f"{a} {v!r} is not a number") from None
            elif a == "--policy":
                if v not in POLICIES:
                    raise rc.SetError(f"--policy {v!r}: one of {', '.join(POLICIES)}")
                o["policy"] = v
            elif a == "--seed":
                o["seed"] = v
            elif a == "--cross":
                o["cross"].append(v)
            elif a == "--split":
                if v != "auto" and not v.isdigit():
                    raise rc.SetError(f"--split {v!r}: auto or a token count")
                o["split"] = v
            elif a == "--score":
                o["score"] = parse_span(v, "--score")
            elif a == "--buckets":
                o["buckets"] = [parse_span(b, "--buckets") for b in v.split(",")]
            elif a == "--n-l":
                o["n_l"] = v
            elif a == "--fill":
                if v not in ("all", "budget"):
                    raise rc.SetError(f"--fill {v!r}: all or budget")
                o["fill"] = v
            else:
                raise rc.SetError(f"unknown flag {a}")
        else:
            sets.append(a)
    if len(sets) != 1 or o["n_l"] is None:
        raise rc.SetError("replay takes one set and --n-l")
    if o["every"] < 1 or o["cap"] < 0 or not 0.0 <= o["decay"] <= 1.0 or o["gbps"] <= 0 or o["crossings"] <= 0:
        raise rc.SetError("--every >= 1, --cap >= 0, --decay in [0, 1], --gbps and --crossings > 0")
    return sets[0], o


def main(argv):
    if argv == ["--self-test"]:
        return self_test()
    if argv[:1] != ["replay"]:
        print(__doc__, file=sys.stderr)
        return 64
    try:
        set_dir, o = parse(argv[1:])
        return replay(set_dir, o)
    except rc.SetError as e:
        print(f"router-residency: {e}", file=sys.stderr)
        return 1


def _X(rows):
    """rows: per step, per layer, a tuple of ids -> [T, L, K]."""
    return np.asarray(rows, dtype=np.int64)


def test_static():
    # one layer, K 2, E 4: capacity 1 holding expert 0, which steps 0 and 2 select
    X = _X([[(0, 1)], [(2, 3)], [(0, 3)]])
    assert static_hits(X, [[0, 1, 2, 3]], [1]).tolist() == [2]
    assert static_hits(X, [[3, 0, 1, 2]], [2]).tolist() == [4]


def test_belady():
    # cap 1, K 1, a b a b a b seeded with a: every b misses and is bypassed (a's next use is sooner or
    # never-again ties) — 3 hits, no admit
    X = _X([[(0,)], [(1,)], [(0,)], [(1,)], [(0,)], [(1,)]])
    h, adm = belady(X, [[0, 1]], [1])
    assert h.tolist() == [3] and adm == 0, (h, adm)
    # a miss never used again must not evict a resident that is
    h, adm = belady(_X([[(0,)], [(5,)], [(0,)]]), [[0]], [1])
    assert h.tolist() == [2] and adm == 0, (h, adm)


def test_lru():
    # two layers, n_l (1, 1), global capacity 2: layer 0 misses every step and takes the LRU slot,
    # layer 1's 7 stays; with a budget of 0 nothing is filled
    X = _X([[(1,), (7,)], [(2,), (7,)], [(1,), (7,)]])
    h, f = global_lru(X, [[0], [7]], [1, 1])
    assert h.tolist() == [0, 3] and f == 3, (h, f)
    h, f = global_lru(X, [[0], [7]], [1, 1], fill="budget", budget=0)
    assert h.tolist() == [0, 3] and f == 0


def test_adaptive():
    # one layer, E 4, cap 1 seeded with 0; expert 2 every step, a pass every 2 steps, no decay: after
    # step 1 its count 2 >= 0 + 1.5, so it swaps in and serves from step 2
    X = _X([[(2,)]] * 6)
    h, sw = adaptive(X, [[0, 1, 2, 3]], [1], 4, every=2, min_count=2, margin=1.5, cap=96, decay=1.0)
    assert h.tolist() == [4] and sw == 1, (h, sw)
    # decay 0 and one use a pass never reach the threshold 2
    X = _X([[(2,)], [(3,)]] * 3)
    h, sw = adaptive(X, [[0, 1, 2, 3]], [1], 4, every=2, min_count=2, margin=1.5, cap=96, decay=0.0)
    assert h.tolist() == [0] and sw == 0, (h, sw)
    # the link: a copy of 25 ms issued after step 1 (t 20 ms, steps of 10 ms) ends at 45 ms, so the
    # expert serves from step 5: steps 5, 6, 7 hit
    X = _X([[(2,)]] * 8)
    h, sw = adaptive(X, [[0, 1, 2, 3]], [1], 4, every=2, margin=1.5, decay=1.0, link=Link(25_000_000, 1.0, 10.0))
    assert h.tolist() == [3] and sw == 1, (h, sw)


def test_adaptive_margin_and_threshold():
    # a pass every 4 steps, no decay: expert 2 used 3 times, the resident 0 once — 3 >= 1 + 1.5 swaps,
    # margin 3 does not
    X = _X([[(2,)], [(2,)], [(2,)], [(0,)], [(2,)], [(2,)]])
    h, sw = adaptive(X, [[0, 1, 2, 3]], [1], 4, every=4, margin=1.5, decay=1.0)
    assert sw == 1 and h.tolist() == [3], (h, sw)
    h, sw = adaptive(X, [[0, 1, 2, 3]], [1], 4, every=4, margin=3.0, decay=1.0)
    assert sw == 0 and h.tolist() == [1], (h, sw)
    # the threshold: 2 uses clear margin 1.5 over an unused resident but not min_count 3
    h, sw = adaptive(_X([[(2,)], [(2,)], [(2,)]]), [[0, 1, 2, 3]], [1], 4, every=2, min_count=3.0, margin=1.5,
                     decay=1.0)
    assert sw == 0, (h, sw)
    # the cap: both layers want a swap, cap 1 takes the larger margin (layer 1: 3 uses against 2)
    X = _X([[(2,), (2,)], [(2,), (2,)], [(1,), (2,)], [(3,), (3,)]])
    h, sw = adaptive(X, [[0, 1, 2, 3], [0, 1, 2, 3]], [1, 1], 4, every=3, margin=1.5, cap=1, decay=1.0)
    assert sw == 1 and h.tolist() == [0, 0], (h, sw)
    X = _X([[(2,), (2,)], [(2,), (2,)], [(1,), (2,)], [(3,), (2,)]])
    h, sw = adaptive(X, [[0, 1, 2, 3], [0, 1, 2, 3]], [1, 1], 4, every=3, margin=1.5, cap=1, decay=1.0)
    assert sw == 1 and h.tolist() == [0, 1], (h, sw)


def test_lru_is_global():
    # layer 0 alternates 1 and 2, layer 1 reuses 9 and never 8: the shared capacity 3 moves 8's slot to
    # layer 0, which then holds both (a per-layer LRU at n_l 1 would never hit there)
    X = _X([[(1,), (5,)], [(2,), (5,)], [(1,), (5,)], [(2,), (5,)]])
    h3, _ = global_lru(X, [[0], [9]], [2, 1])
    assert h3.tolist() == [2, 3], h3
    X = _X([[(1,), (9,)], [(2,), (9,)], [(1,), (9,)], [(2,), (9,)], [(1,), (9,)]])
    h, _ = global_lru(X, [[0], [9, 8]], [1, 2])
    assert h.tolist() == [3, 5], h


def test_link_price_and_score():
    # 2 swaps of 1 MB over 4 steps of 10 ms: 0.5 MB a token, 0.5 ms at 1 GB/s, 5 % of the step, and
    # 3 x 0.5 MB / 10 ms = 0.15 GB/s of DRAM
    p = price(2, 4, 1_000_000, 10.0, gbps=1.0, pageable=0.5, crossings=3.0)
    assert (p["swaps"], p["mb"], p["ms_pinned"], p["ms_pageable"]) == (0.5, 0.5, 0.5, 1.0), p
    assert abs(p["link"] - 0.05) < 1e-12 and abs(p["dram_gbps"] - 0.15) < 1e-12, p
    # the end to end replay on a synthetic set: chunks of 4, 12 tokens, split auto = 4 (the boundary
    # nearest 6, ties low); static on seed prefix n_l 1 holds expert 0, which positions 0 and 1 of each
    # chunk select
    rows = [(0, 1) if t % 4 < 2 else (2, 3) for t in range(12)]
    with tempfile.TemporaryDirectory() as root:
        d = wu._fake_set(root, "s", {0: rows}, 4, 2, chunk=4)
        s = rc.read_manifest(d)
        assert rc.held_split(s) == 4
        set_dir, o = parse([d, "--n-l", "0:1", "--policy", "static", "--seed", "prefix", "--score", "0-1"])
        saved, sys.stdout = sys.stdout, open(os.devnull, "w")
        try:
            assert replay(set_dir, o) == 0
        finally:
            sys.stdout.close()
            sys.stdout = saved
        X = np.stack([read_ids(s, 0)[4:]], axis=1)
        rec = np.zeros(X.shape, dtype=bool)
        static_hits(X, [np.arange(4)], [1], rec)
        pos = np.arange(4, 12) % 4
        assert rec[pos <= 1].mean() == 0.5 and rec[pos >= 2].sum() == 0  # 1 of K 2 at positions 0-1
        try:
            same_model([s, dict(s, header=dict(s["header"], model="/other/m.gguf"))])
            raise AssertionError("sets of two model files were accepted")
        except rc.SetError:
            pass


def self_test():
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("router-residency: self-test ok (7 cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
