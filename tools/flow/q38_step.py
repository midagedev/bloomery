#!/usr/bin/env python3
"""Qwen3.8's decode step, MTP window and prompt ubatch as per-resource timelines, by machine, backtested.

The evaluator the discovery search scores ideas with: decode tok/s and prompt tok/s from the machine's resource
rates and the plan, its structure the code's, its constants rows of q38-constants.tsv (kind measured, derived or
assumed, each with its conditions, its source and the backtest row it was anchored on, which is then not scored
against it), its errors the residuals of `--backtest`.

The sync structure (crates/, read 2026-10-09 at b3878dc5). One row's decode step is the walk (1, 1, Step)
(runtime/src/sched.rs:105-139 step_nth): per layer the back of layer l-1, then the front and the shadow of layer l, on
the one card stream. Front (arch/qwen3moe/program38.rs:1495-1507 Step38::front): the two gated mixes, the mixer (GDN
or QSA), the router, the handoff launch that writes the normed row and the ids into the host-mapped page, and the go
(host/step.rs:430-443 go_batch: system barrier, layer word, barrier, generation add). Shadow (program38.rs:1509-1513):
the card leg over the layer's card experts (Card38::enqueue), then the shared expert. Back (program38.rs:1517-1521 ->
:728-737 step_back): the wait (host/step.rs:451-459, op_wait_geq on the row's counter) and the combine
(shared_add), which reads the host's sum through the mapping. The host pool spins on the generation word and serves
each layer in chain order (host/mod.rs:2286 serve_captured_of; host/step.rs:880 take_go, :922 signal). So a layer's
wall is front + max(shadow, go -> host service -> signal) + back, and the layers add: the next front reads this
back's output. Nothing overlaps across layers for one row; only the pair pass (2, 1, Step) runs two rows one layer
apart (out of scope here). A verify of w rows is (1, w, Step) through the same port's Cols(w) chain
(program38.rs:1580-1615 Verify38; host/mod.rs:34-39): one handoff of w columns, one go, one union over every
column's host experts (each distinct expert read once), one wait. On a tier layer (place bp) the 3090 serves its
slots behind its own go and wait (program38.rs:708-719, :740-747), in the same shadow.

A prompt's ubatch walk is (1, U, Batch) (sched.rs:145-170 batch_nth at one unit: shadow, serve, back, then the next
front, which reads this back). Front (arch/qwen3moe/wide38.rs:2031-2055): the layer to its router over U rows, then
the D2H of the activations and routed ids. Shadow (wide38.rs:2068-2084): under host streaming the shared expert, then
the pick (wide38.rs:1919-1975 stream_pick: the host waits for the download, counts, and SwapMachine::call_pick
admits host experts into the card pool, the copies staged by the copier thread(s) and landed in batches), then the
card route, each GEMM batch behind its landing batch. Serve (host/batch.rs:1209 serve_port): the calling thread's
host union over the remaining host experts' columns. Back (wide38.rs:2091-2110): the upload of the host sums and the
combine. A layer's wall is front + down + max(card chain, host chain) + upload + back, the card chain the copies
then the route's tail, the host chain the pick, the copies a backlog bound leaves ahead of the union, and the union
(the walk the engine prices with: runtime/src/xsplit.rs:234-291 admit_walk; v0.2.7's rule: every count past
floor = floor(m*) + 1, wide38.rs:1824-1833 at v0.2.7).

Units: us, ms, bytes, GB/s (1e9 B/s). Per-layer numbers are means over the 48 layers. A tok/s carries its
conditions: machine, depth or P, the hit or host slots, the window (w, E).

    --machine NAME      box-3090 (default: the primary target, the box's 3090 beside its host), box-3090-lowhost
                        (the same with the host's expert threads capped: the weak-host shape the box can run),
                        box-a6000, common-5090 (a direction check, mostly datasheet-derived); every row of the
                        machine's prefix is a term
    --config NAME       the preset's configuration (CONFIGS): box-3090 server (default: a lone 3090's plan a,
                        residency, xstream split, the draft), gate (--place gate's defaults), readme (the README row)
    --set NAME=VALUE    override a term or a constant (repeatable, after the configuration's); unknown names refused
    (no mode)           the timeline and tok/s at decode depths short, 3.8K, 29.6K and prompts P 512, 4096
    --ceilings          per token, the bytes and time each resource must move; the perfect-overlap and serial
                        bounds; the Belady hits on the prose trace; today against them, and where the gap sits
    --backtest          the recorded outcomes B1-B9 against the model, the anchors listed, the pass bar
    --tornado           today's prediction at each banded constant's ends, assumed first, anchors re-solved
    --explain           every term of the timeline with the constants it reads
    --replay DIR        the trace replays the derived prose rows come from (numpy; tools/ref/router-residency.py)
    --self-test         the arithmetic of a hand-worked layer, the refusals, the presets' consistency

Scoring (written before the first backtest). An effect row (a ratio between two arms of one sitting) passes when its
sign is right and |predicted effect| / |measured effect| lies in [1/1.5, 1.5]; a zero row (its measured interval
holds 0, or the record says "unchanged") passes when the predicted effect lies inside that interval. An absolute row
(tok/s) passes when predicted / measured lies in [1/1.5, 1.5]; its error is printed beside. The pass bar: every
non-anchor row's sign right, and the magnitude inside the factor on at least 70 % of them.

Refused by name (exit 2): a constants file whose row has another cell count, a name twice, a kind outside measured /
derived / assumed, or a value outside its own band; an unknown machine or configuration; a --set of a name no row or
term carries, or a value that is not a finite number; a hit outside [0, 1]; a window of w outside 1..4 or E outside
1..w; a prompt of P < 1; host threads past the machine's. Exit 64: usage.

Python 3 standard library only; --replay alone loads numpy through tools/ref/router-residency.py.
"""

import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONSTANTS = os.path.join(HERE, "q38-constants.tsv")
RR = os.path.join(HERE, "..", "ref", "router-residency.py")

KINDS = ("measured", "derived", "assumed")
MACHINES = {"box-3090": "3090.", "box-3090-lowhost": "3090.", "box-a6000": "a6000.", "common-5090": "5090."}
DEFAULT_MACHINE = "box-3090"
# A preset that is another's rows with terms set: the weak-host emulation the box can run (docs/plan-triage.md G5),
# the host's expert threads capped (host_threads; what it moves: host_scale()).
PRESET_SETS = {"box-3090-lowhost": ["host_threads=12"]}
# The configurations a preset runs today, by name (--config), the first its default. Each is a list of --set terms.
CONFIGS = {
    "box-3090": {
        "server": ("a lone 3090's server: plan a (serve_seats/qwen38.rs:424-427), mid-p0-s1, xstream split, the MTP "
                   "draft (its 933 experts off the 4,269)", []),
        "gate": ("--place gate's defaults: residency off (levers/src/lib.rs:1459-1475), xstream off "
                 "(gpu-gates/src/bin/shared/xstream38.rs:42-45), the draft off: the plain prompt path, a static card set",
                 ["mtp=0", "hit=0.1907", "flips_tok=0", "pick=0", "ring=0"]),
        "readme": ("the README row (rig-log 10-06#num3090, 0.2.1 b83f1269): --place gate, mid-p32-s1 set, the draft "
                   "off, the plain prompt path (no stream at gate), the 0.2.1 plan's card experts [derived]",
                   ["mtp=0", "hit=0.6450", "pick=0", "ring=0", "pool_pinned=32", "card_experts=3428",
                    "pf_front_extra_tok_ms=0.0537"]),
    },
    "box-a6000": {"server": ("plan a's serve defaults: mid-p0-s1, xstream split, the MTP draft", [])},
    "common-5090": {"server": ("the user's server at 0.2.8's defaults: plan a, mid-p0-s1, xstream split, the draft", [])},
}
CONFIGS["box-3090-lowhost"] = CONFIGS["box-3090"]
DEPTHS = {"short": 128, "3.8K": 3800, "29.6K": 29600}
PROMPTS = (512, 4096)


class Refused(Exception):
    pass


# ============================================================================ constants

class Const:
    __slots__ = ("name", "value", "lo", "hi", "unit", "kind", "conditions", "source", "anchor", "note")

    def __init__(self, cells):
        (self.name, value, lo, hi, self.unit, self.kind, self.conditions, self.source,
         self.anchor, self.note) = cells
        try:
            self.value, self.lo, self.hi = float(value), float(lo), float(hi)
        except ValueError:
            raise Refused(f"q38-constants.tsv: {self.name}: a value, lo or hi that is not a number") from None


def load_constants(path=CONSTANTS):
    out, header = {}, None
    try:
        f = open(path, encoding="utf-8")
    except OSError as e:
        raise Refused(f"{path}: {e.strerror}") from None
    with f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            cells = line.rstrip("\n").split("\t")
            if header is None:
                header = cells
                if len(header) != 10:
                    raise Refused(f"{path}: the header has {len(header)} cells, the format 10")
                continue
            if len(cells) != len(header):
                raise Refused(f"{path}: {cells[0]!r} has {len(cells)} cells, the header {len(header)}")
            c = Const(cells)
            if c.name in out:
                raise Refused(f"{path}: {c.name} twice")
            if c.kind not in KINDS:
                raise Refused(f"{path}: {c.name}: kind {c.kind!r} is not one of {', '.join(KINDS)}")
            if not (c.lo <= c.value <= c.hi):
                raise Refused(f"{path}: {c.name}: value {c.value} outside its band {c.lo}..{c.hi}")
            out[c.name] = c
    if header is None:
        raise Refused(f"{path}: no header row")
    return out


# The terms a scenario sets that are not constants of a machine: today's defaults (0.2.8 under --place a, the serve
# defaults), each a number so that --set states an idea. Documented in --explain.
TERMS = {
    "mtp": (1.0, "1: the MTP draft (its card bytes leave the plan, the decode is windows); 0: plain steps"),
    "w": (4.0, "verify rows a window (the draft's three ids and the anchor row)"),
    "E": (-1.0, "positions a window; -1 takes the machine's genre row (E_prose or E_chat)"),
    "chat": (0.0, "1: the chat genre's E and hit band rows instead of prose's"),
    "hit": (-1.0, "card share of a row's routed slots at decode; -1 from host_slots, else the replay row"),
    "host_slots": (-1.0, "host slots a decoded row (48 layers x 10 - card slots); -1 from the hit"),
    "tier_experts": (0.0, "experts on an expert-tier card (place bp: the 3090)"),
    "tier_hit": (-1.0, "the tier's share of a row's slots; -1 from tier_experts over the routed set"),
    "host_only_layers": (0.0, "layers whose routed experts all stay on the host (5 before 2026-10-01's q38hol)"),
    "head_rows": (65536.0, "the draft head's rows (the shipped list); 248320 the full head"),
    "pick": (2.0, "prompt pick: 0 none (0.2.5), 1 v0.2.6/0.2.7's floor rule, 2 the walk (0.2.8), 3 admit mode's floor 32 (place bp)"),
    "copier_threads": (4.0, "the pick's staging threads (1 before 0.2.8's pickrate)"),
    "pool_pinned": (0.0, "card experts a layer the residency pins (no admit replaces them): mid-p0-s1 today"),
    "flips_tok": (-1.0, "residency flips a decoded token; -1 the machine's steady row"),
    "restore_ms": (0.0, "the #3 fix's pool return before a prompt's first token, ms"),
    "miss_card_share": (0.0, "a share s of a layer's host misses the card reads over PCIe instead (G2's mechanism)"),
    "layer_overlap": (1.0, "1: max(card shadow, host leg) inside a layer (the code); 0: their sum"),
    "prompt_corr": (0.0, "the pool's share at a prompt's hottest ranks on entry: 0 uncorrelated (an id prefix, another prompt)"),
    "ub_carry": (-1.0, "the share of a ubatch's admits still hot in the next ubatch of one prompt; -1 the prose row"),
    "ring": (1.0, "1: split's ring streams past the pool's admits (0.2.6..0.2.8 under split); 0: off (admit mode, place bp, 0.2.5)"),
    "ring_half": (97.0, "the stream's ring slots a unit (xstream_half on the A6000's load lines)"),
    "pf_front_extra_tok_ms": (0.0, "A6000 front ms a row a walk a build lacks (0.2.5/0.2.6: lever 3, ~0.22 s at 4096)"),
    "host_threads": (-1.0, "the host's expert threads; -1 the machine's host_threads_base (a cap: host_scale())"),
    "pf_overlap": (0.0, "a prompt layer's serial card part S (front, down, up, back) hidden under its max I: wall = S + I - s min(S, I) (0 the code)"),
}


def params(consts, machine, sets=()):
    """The term values for `machine`: common rows, the machine's rows (prefix stripped), TERMS, then `sets`."""
    if machine not in MACHINES:
        raise Refused(f"no machine {machine!r}: the presets are {', '.join(MACHINES)}")
    pre = MACHINES[machine]
    sets = list(PRESET_SETS.get(machine, [])) + list(sets)
    others = [p for p in MACHINES.values() if p != pre]
    p = {}
    for name, c in consts.items():
        if any(name.startswith(o) for o in others):
            continue
        p[name[len(pre):] if name.startswith(pre) else name] = c.value
    for k, (v, _) in TERMS.items():
        p.setdefault(k, v)
    for s in sets:
        if "=" not in s:
            raise Refused(f"--set {s!r}: NAME=VALUE")
        k, v = s.split("=", 1)
        if k not in p:
            raise Refused(f"--set {k}: no term or constant of that name on {machine}")
        try:
            x = float(v)
        except ValueError:
            raise Refused(f"--set {k}={v}: not a number") from None
        if not math.isfinite(x):
            raise Refused(f"--set {k}={v}: not a finite number")
        p[k] = x
    p["_machine"] = machine
    return p


def config_sets(machine, config=None):
    """The --set terms of `machine`'s configuration `config` (its first when None), refused by name."""
    table = CONFIGS[machine]
    name = config or next(iter(table))
    if name not in table:
        raise Refused(f"{machine} has no configuration {name!r}: {', '.join(table)}")
    return name, list(table[name][1])


def const_of(consts, machine, name):
    """The row a term name resolves to on `machine` (its machine row first)."""
    return consts.get(MACHINES[machine] + name) or consts.get(name)


# ============================================================================ the trace replays (numpy)

REPLAY_N = (3336, 4269, 5482, 6415, 6810, 11569, 12567, 18694, 19803)  # card (+ tier) experts the plans hold
DISPERSION_N = (3336, 4269, 5482, 12567)  # the presets' plans: the window and dispersion rows
REPLAY_RANKS = (1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 320, 384, 448, 512)


def replay_main(trace):
    """The derived prose rows: hits by policy and card count, the window's distinct-expert factors, the prompt
    units' count profiles, the per-layer miss dispersion. Prints `row NAME VALUE` lines and the commands' facts."""
    try:
        import importlib.util
        import numpy as np
    except ModuleNotFoundError:
        raise Refused("--replay needs numpy (tools/ref/router-residency.py reads the trace with it)") from None
    spec = importlib.util.spec_from_file_location("rr", RR)
    rr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rr)
    s = rr.Set(trace)
    if (s.E, s.K) != (512, 10) or s.T < 4096:
        raise Refused(f"{trace}: {s.T} tokens of {s.E} experts top-{s.K}; the q38 replay wants 512 x top-10 and 4096+")
    L = list(range(48))
    X = s.stack(L)  # [T, 48, 10]: every layer, not rr.family('q38')'s 43 (the plan holds all 48 since q38hol)
    T = X.shape[0]
    calls = [(c * 768, c * 768 + 512, c * 768 + 768) for c in range(T // 768)]
    dec = np.concatenate([np.arange(a, b) for _, a, b in calls]) if calls else np.arange(T)
    print(f"# replay of {trace}: {T} rows, 48 layers, {len(calls)} contexts of 512 prompt + 256 decode rows")
    seed = [np.arange(512) for _ in L]
    rule = rr.Rule("mid")

    def row(name, v):
        print(f"row {name}\t{v:.4f}")

    for N in REPLAY_N:
        n_l = [rr.spread(N, L)[l] for l in L]
        res0 = rr.seed_resident(seed, n_l, 512)
        st = rr.static_hits(X, seed, n_l).sum() / X[:, :, :].size
        rec = np.zeros(X.shape, dtype=bool)
        r = rr.replay(X, res0, n_l, 512, rule, sem="flip", spares=1, d=2, record=rec)
        mid_all = rec.mean()
        mid_dec = rec[dec].mean()
        bel, _ = rr.belady(X, seed, n_l)
        bel = bel.sum() / X.size
        row(f"hit_static@{N}", st)
        row(f"hit_mid@{N}", mid_dec)
        row(f"hit_belady@{N}", bel)
        print(f"#   N {N}: n_l {min(n_l)}..{max(n_l)}; mid over all rows {mid_all:.4f}, decode rows {mid_dec:.4f}; "
              f"swaps {r.swaps} over {T} rows")
        if N == 4269:
            # The README 3090 row's word, mid-p32-s1: each layer's 32 lowest seed ids never a victim.
            pinned = np.zeros((48, 512), dtype=bool)
            pinned[:, :32] = True
            rec32 = np.zeros(X.shape, dtype=bool)
            rr.replay(X, res0, n_l, 512, rule, sem="flip", spares=1, d=2, record=rec32, pinned=pinned)
            row(f"hit_midp32@{N}", rec32[dec].mean())
        if N in DISPERSION_N:
            miss = ~rec[dec]                               # [Td, 48, 10]
            per_layer = miss.sum(axis=2).astype(float)     # misses a layer a row
            p = per_layer.mean() / 10
            row(f"miss_dispersion@{N}", per_layer.var() / (10 * p * (1 - p)))
            Xd = X[dec]
            for w in (2, 3, 4):
                d_all = d_miss = n_miss = 0.0
                n_win = 0
                for c0 in range(0, Xd.shape[0] - w + 1, w):
                    for li in range(48):
                        ids = Xd[c0:c0 + w, li, :]
                        m = miss[c0:c0 + w, li, :]
                        d_all += len(set(ids.ravel().tolist()))
                        mi = ids[m]
                        d_miss += len(set(mi.tolist()))
                        n_miss += mi.size
                        n_win += 1
                row(f"union_all_w{w}@{N}", d_all / (n_win * 10 * w))
                row(f"union_miss_w{w}@{N}", d_miss / max(n_miss, 1))
    for U, spans in ((512, [(a, a + 512) for a, _, _ in calls]), (4096, [(0, 4096)])):
        prof = np.zeros(512)
        k = 0
        for a, b in spans:
            for li in L:
                c = np.bincount(X[a:b, li].ravel(), minlength=512)
                prof += np.sort(c)[::-1]
                k += 1
        prof /= k
        for rnk in REPLAY_RANKS:
            row(f"prose_c{U}_r{rnk}", prof[rnk - 1])
        print(f"#   U {U}: {k} unit-layers; touched experts a layer {np.mean(prof > 0.5) * 512:.1f}")
    # The pool's carry between two units of one prompt: per context and layer, the k hottest experts of the
    # prompt's first 256 rows and the share of the next 256 rows' columns they take, scaled between k random
    # experts' share (0) and the next half's own k hottest (1).
    for kk in (32, 64, 96):
        num = den = 0.0
        for a, _, _ in calls:
            for li in L:
                c1 = np.bincount(X[a:a + 256, li].ravel(), minlength=512)
                c2 = np.bincount(X[a + 256:a + 512, li].ravel(), minlength=512)
                tot = c2.sum()
                cap = c2[np.argsort(-c1, kind="stable")[:kk]].sum() / tot
                num += cap - kk / 512
                den += np.sort(c2)[::-1][:kk].sum() / tot - kk / 512
        row(f"prose_carry_k{kk}", num / den)
    return 0


# ============================================================================ distributions

def betabinom(n, p, disp):
    """P(k), k = 0..n, of a layer's n slots landing k on one side with mean share p, overdispersed by `disp`
    (variance over the binomial's, the replay's miss_dispersion row): a beta-binomial of intra-class correlation
    rho = (disp - 1) / (n - 1); disp 1 is the binomial."""
    if not 0.0 <= p <= 1.0:
        raise Refused(f"a slot share {p} outside [0, 1]")
    if n == 0 or p in (0.0, 1.0):
        return [1.0 if k == round(n * p) else 0.0 for k in range(n + 1)]
    rho = 0.0 if n <= 1 else max(0.0, (disp - 1.0) / (n - 1))
    if rho <= 1e-9:
        return [math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n + 1)]
    rho = min(rho, 0.999)
    a, b = p * (1 / rho - 1), (1 - p) * (1 / rho - 1)
    lb = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    out = []
    for k in range(n + 1):
        lk = math.lgamma(k + a) + math.lgamma(n - k + b) - math.lgamma(n + a + b)
        out.append(math.comb(n, k) * math.exp(lk - lb))
    s = sum(out)
    return [x / s for x in out]


# ============================================================================ the decode step and the MTP window

def card_lat(p):
    """The A6000-anchored latency parts (us, ms for out) of a layer's card work, by subtraction of each anchor's
    byte time at the A6000's rate; a machine scales them by its lat_ratio."""
    bw = p["bw_a6000_gbs"] * 1e3
    return dict(front=p["front_us_a6000"] - p["dense_front_b"] / bw,
                shared=p["shared_us_a6000"] - p["shared_b"] / bw,
                leg=p["leg_launch_us_a6000"],
                back=p["back_us_a6000"],
                out=p["step_out_ms_a6000"] * 1e3 - p["head_b"] / bw)


def attn_b(p, depth):
    """A QSA layer's K/V and pooled-key bytes for one query at `depth`, averaged over the 48 layers."""
    sel = min(depth, p["qsa_select"])
    return p["n_qsa"] / p["n_layer"] * (p["kv_b_pos"] * sel + p["idx_b_pos"] * depth)


def host_scale(p):
    """The host's expert threads capped at T below the base its W and c ran at: (W's factor, c's factor). A column's
    compute spreads over the threads (c x base / T); an expert's bytes stream at min(host DRAM, T x core_dram_gbs),
    so W rises only once the capped threads cannot fill the channels."""
    base = p["host_threads_base"]
    T = p["host_threads"] if p["host_threads"] > 0 else base
    if T > base:
        raise Refused(f"host_threads {T:g} past the machine's {base:g}")
    full = min(p["host_dram_gbs"], base * p["core_dram_gbs"])
    return full / min(p["host_dram_gbs"], T * p["core_dram_gbs"]), base / T


def host_w_us(p):
    """One host expert at one column, the pool's whole DRAM rate on it: the measured W on the box, bytes over the
    pool's rate elsewhere; scaled by a thread cap."""
    w = p["host_w_us"] if p.get("host_w_us", 0) > 0 else p["expert_b"] / (p["host_dram_gbs"] * p["host_w_eff"] * 1e3)
    return w * host_scale(p)[0]


def union_costs(p):
    """The host union's (W, c) at the machine's threads."""
    fw, fc = host_scale(p)
    return p["union_W_us"] * fw, p["union_c_us"] * fc


def slot_shares(p, chat=None):
    """(card, tier, host) shares of a routed slot at decode on the card layers."""
    if p["host_slots"] >= 0:
        cards = p["n_layer"] - p["host_only_layers"]
        h = 1.0 - (p["host_slots"] - 10 * p["host_only_layers"]) / (10 * cards)
    elif p["hit"] >= 0:
        h = p["hit"]
    else:
        chat = p["chat"] if chat is None else chat
        h = p["hit_chat"] if chat else p["hit_prose"]
    if not 0.0 <= h <= 1.0:
        raise Refused(f"a hit of {h:.3f} outside [0, 1] (host_slots {p['host_slots']})")
    t = p["tier_hit"] if p["tier_hit"] >= 0 else min(1 - h, p["tier_experts"] / (p["n_expert"] * p["n_layer"]))
    t = max(0.0, min(t, 1.0 - h))
    return h, t, 1.0 - h - t


def layer_parts(p, depth, w=1):
    """The card's per-layer work of a w-row verify (w = 1 the step), us: front, shared, leg launches, the bytes
    time of one card expert, back; and the step's outside part, us."""
    bw = p["card_bw_gbs"] * 1e3
    lat, r = card_lat(p), p["lat_ratio"]
    grow = 1.0 + p["verify_row_frac"] * (w - 1)
    front = (p["dense_front_b"] + attn_b(p, depth) * w - attn_b(p, 6)) / bw + lat["front"] * r * grow
    shared = p["shared_b"] / bw + lat["shared"] * r * grow
    back = lat["back"] * r * grow + p["layer_extra_us"]
    out = p["head_b"] / bw + lat["out"] * r * grow + (p["tier_out_us"] if p["tier_experts"] > 0 else 0.0)
    return dict(front=front, shared=shared, leg=lat["leg"] * r, slot=p["expert_b"] / bw, back=back, out=out)


def host_leg_us(p, k_host, cols=1.0):
    """The host's leg for k distinct host experts at `cols` columns each: the go and signal latency, the service's
    fixed part, each expert's max(W, c cols); a share miss_card_share of them the card reads instead."""
    if k_host <= 0:
        return p["handoff_us"]
    k = k_host * (1.0 - p["miss_card_share"])
    return p["handoff_us"] + p["host_svc_us"] + k * max(host_w_us(p), union_costs(p)[1] * cols)


def layer_wall(p, parts, k_card, k_tier, k_host, cols=1.0, card_layer=True):
    """One layer's wall (us) and its resource split: front + max(card shadow, host leg, tier leg) + back."""
    shadow = parts["shared"]
    if card_layer:
        shadow += parts["leg"] + k_card * parts["slot"]
    pcie = k_host * p["miss_card_share"] * p["expert_b"] / (p["pcie_h2d_gbs"] * 1e3)
    shadow += pcie
    host = host_leg_us(p, k_host, cols)
    tier = p["tier_fixed_us"] + k_tier * p["tier_slot_us"] if k_tier > 0 else 0.0
    inner = max(shadow, host, tier) if p["layer_overlap"] >= 0.5 else shadow + host + tier
    return parts["front"] + inner + parts["back"], dict(shadow=shadow, host=host, tier=tier, pcie=pcie)


def step(p, depth, w=1, hit=None):
    """A decode step (w = 1) or a w-row verify: the layers' expected walls over the slot distribution, the outside
    part, the residency flips' cost on the step's path. Returns ms and the per-layer timeline (us)."""
    if not 1 <= w <= 4:
        raise Refused(f"a verify of {w} rows: the window is 1..4 rows")
    h, t, hh = slot_shares(p) if hit is None else (hit, 0.0, 1.0 - hit)
    parts = layer_parts(p, depth, w)
    disp = p["miss_dispersion"]
    if w == 1:
        n, cols = 10, 1.0
        q_host, q_tier = hh, t
    else:
        # The window's distinct experts a layer: 10 w u_all of them, a host one with the share that keeps the
        # misses' own (smaller) repeat, each read once at 1/u_miss columns.
        u_all, u_miss = p[f"union_all_w{w}"], p[f"union_miss_w{w}"]
        n = max(1, round(10 * w * u_all))
        q_host = min(1.0, hh * u_miss / u_all)
        q_tier = min(1.0 - q_host, t * u_all / u_all)
        cols = 1.0 / u_miss
    pk = betabinom(n, q_host, disp)
    tl = dict(front=parts["front"], shared=parts["shared"], back=parts["back"], shadow=0.0, host=0.0, tier=0.0,
              pcie=0.0, wall=0.0, host_slots=0.0)
    cards = int(p["n_layer"] - p["host_only_layers"])
    tot = 0.0
    for kh, pr in enumerate(pk):
        if pr < 1e-12:
            continue
        rest = n - kh
        kt = rest * q_tier / (1 - q_host) if q_host < 1 else 0.0
        kc = rest - kt
        wall, r = layer_wall(p, parts, kc, kt, kh, cols)
        tot += pr * wall
        for k in ("shadow", "host", "tier", "pcie"):
            tl[k] += pr * r[k]
        tl["host_slots"] += pr * kh
    only = 0.0
    if p["host_only_layers"] > 0:
        k_only = n * (1.0 if w == 1 else p[f"union_miss_w{w}"] / p[f"union_all_w{w}"]) if w > 1 else 10
        only, _ = layer_wall(p, parts, 0, 0, min(k_only, n), cols, card_layer=False)
    layers_us = cards * tot + p["host_only_layers"] * only
    tl["wall"] = layers_us / p["n_layer"]
    flips = p["flips_tok"] if p["flips_tok"] >= 0 else p["flips_steady"]
    flip_us = flips * p["flip_cpu_us"] * p["flip_crit"] * (1 if w == 1 else 1)
    ms = (layers_us + parts["out"] + flip_us) / 1e3
    tl.update(out=parts["out"], flip=flip_us, step_ms=ms, hit=h, tier_share=t)
    return ms, tl


def draft_ms(p, w):
    """The MTP draft's chain before a w-row verify: w - 1 walks (the refresh, then its own), each its layer's dense
    bytes, its ten experts and the head (the shipped row list or the full head), and one readback."""
    bw = p["card_bw_gbs"] * 1e3
    head = p["head_rows"] * p["hidden"] * 34 / 32
    walk = (p["mtp_dense_b"] + 10 * p["mtp_expert_b"] + head) / bw + p["mtp_walk_lat_us"] * p["lat_ratio"]
    return ((w - 1) * walk + p["mtp_readback_us"]) / 1e3


def decode(p, depth):
    """Decode tok/s at `depth`: plain steps, or MTP windows of w rows keeping E positions; and the timeline."""
    if p["mtp"] < 0.5:
        ms, tl = step(p, depth)
        return 1e3 / ms, dict(tl, window_ms=ms, E=1.0, w=1, draft_ms=0.0)
    w = int(p["w"])
    E = p["E"] if p["E"] > 0 else (p["E_chat"] if p["chat"] >= 0.5 else p["E_prose"])
    if not 1.0 <= E <= w:
        raise Refused(f"E {E} positions a window outside 1..{w}")
    ms, tl = step(p, depth, w)
    d = draft_ms(p, w)
    win = ms + d + p["window_host_us"] / 1e3
    return E * 1e3 / win, dict(tl, window_ms=win, E=E, w=w, draft_ms=d)


# ============================================================================ the prompt ubatch

def profile(p, U):
    """The unit's sorted routed counts over the 512 experts of a layer (hottest first), from the prose rows at
    U 512 and 4096: log-log between the listed ranks, linear in U between the two units, proportional below 512."""
    def at(u):
        pts = [(r, p[f"prose_c{u}_r{r}"]) for r in RANKS]
        out = []
        for r in range(1, 513):
            for (r0, c0), (r1, c1) in zip(pts, pts[1:]):
                if r0 <= r <= r1:
                    if c0 > 0 and c1 > 0:
                        f = math.log(r / r0) / math.log(r1 / r0)
                        out.append(math.exp(math.log(c0) + f * (math.log(c1) - math.log(c0))))
                    else:
                        out.append(c0 + (c1 - c0) * (r - r0) / (r1 - r0))
                    break
        return out
    lo, hi = at(512), at(4096)
    if U <= 512:
        return [c * U / 512 for c in lo]
    f = (U - 512) / (4096 - 512)
    return [a + f * (b - a) for a, b in zip(lo, hi)]


RANKS = (1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 320, 384, 448, 512)


def split_sets(prof, n_card, n_tier, n_hot):
    """A layer's counts on the host, the card pool and the tier: n_hot of the card's experts at the unit's hottest
    ranks (the pool's carry), the rest of the card's and the tier's spread evenly over the other ranks (an id-prefix
    seed is uncorrelated with a prompt's heat)."""
    E = len(prof)
    n_hot = max(0, min(n_hot, n_card))
    hot = list(range(n_hot))
    rest = list(range(n_hot, E))

    def spread(ranks, n):
        if n <= 0:
            return []
        st = len(ranks) / n
        return sorted({ranks[min(len(ranks) - 1, int((i + 0.5) * st))] for i in range(n)})
    card = hot + spread(rest, n_card - n_hot)
    cset = set(card)
    left = [r for r in range(E) if r not in cset]
    tier = spread(left, n_tier)
    tset = set(tier)
    host = [r for r in left if r not in tset]
    return [prof[r] for r in host], [prof[r] for r in card], [prof[r] for r in tier]


def cost_union(counts, W, c):
    """The host union's wall over experts of these expected counts: W a listed expert (touched with probability
    1 - e^-m) plus c a column — the additive form q0's per-layer union walls fit within 6 % at P 512 and 4096
    (specs/worker2/pickrate-impl/c1_band.py 'union model vs record'), where the engine's own price, max(W, c m)
    (xsplit.rs:249), reads them 13..27 % low."""
    return sum(W * (1 - math.exp(-m)) + c * m for m in counts)


def admits_walk(p, host, U, card_cols):
    """xsplit.rs admit_walk at the engine's own constants (body38.rs XSTREAM_COSTS, the tail per card pick) and the
    machine's lane probe: (admits, backlog bound)."""
    Wp, cp, fp = p["walk_W_us"], p["walk_c_us"], p["walk_f_us"]
    b = p["expert_b_q4"]
    listed = sum(1 - math.exp(-m) for m in host)
    cols = sum(host)
    s = Wp * listed / (Wp * listed + cp * cols) if cols > 0 else 0.0
    beside = p["lane_gbs"] * 1e3 * (1 - s)
    copy_us = b / beside
    tail = p["walk_tail_us_col"] * card_cols
    gate = math.ceil(copy_us / ((Wp + cp) * p["top_k"]))
    if U < gate:
        return 0, 0
    ranked = sorted(host, reverse=True)
    left = sum(max(Wp, cp * m) for m in ranked if m >= 0.5)
    card, a = tail, 0
    for m in ranked:
        if m < 0.5:
            break
        if card + copy_us > left - max(Wp, cp * m):
            break
        card += copy_us
        left -= max(Wp, cp * m)
        a += 1
    if a == 0:
        return 0, 0
    bound = math.ceil(max(left - tail, 0.0) * beside / b)
    return a, int(min(max(bound, min(p["walk_queue_floor"], a)), a))


def floor_v027(p, U):
    """v0.2.7's pick floor (wide38.rs:1824 pick_floor at v0.2.7): floor(m*) + 1 at its costs row (c 7.0, W 23.36,
    the lane alone), STREAM_FLOOR below the unit width m_min."""
    b = p["expert_b_q4"]
    mstar = (b / (p["lane_gbs"] * 1e3) + p["walk_f_us"]) / (p["v027_c_us"] - p["walk_k_us"])
    if U < mstar * p["n_expert"] / p["top_k"]:
        return p["stream_floor"]
    return math.floor(mstar) + 1


def pick(p, host, pool, U, card_cols):
    """The prompt call's admits a layer (swaprule.rs:1137 call_pick: host experts past the floor, hottest first,
    each over the coldest pool resident it beats), the backlog bound, and the victims' counts."""
    mode = int(p["pick"])
    if mode == 0 or not pool:
        return 0, 0, []
    if mode == 2:
        a_walk, bound = admits_walk(p, host, U, card_cols)
        if a_walk == 0:
            return 0, 0, []
        floor = sorted(host, reverse=True)[a_walk - 1]
    else:
        floor = floor_v027(p, U) if mode == 1 else p["stream_floor"]
        bound = p["v027_backlog"]
    cand = sorted((m for m in host if m >= floor), reverse=True)
    vict = sorted(pool)
    a = 0
    for m, v in zip(cand, vict):
        if m <= v:
            break
        a += 1
    return a, min(bound, a), vict[:a]


def copy_rates(p, share, union_gbs):
    """(beside the union, alone) staging rates, GB/s. Beside: the threads' rate or the link, less the union's
    weight-listing burst share of its window (xsplit.rs union_burst_share, the engine's own term), capped by the host
    DRAM the union leaves at three crossings a copied byte (the source read, the pinned write, the DMA read).
    Alone: the threads' rate or the load's lane probe."""
    n = p["copier_threads"]
    room = max(p["host_dram_gbs"] - union_gbs, 1.0) / 3
    beside = min(min(n * p["copy_thread_beside_gbs"], p["pcie_h2d_gbs"]) * (1 - share), room)
    alone = min(n * p["copy_thread_alone_gbs"], p["lane_gbs"], p["host_dram_gbs"] / 3)
    return beside, alone


def ring_stream(p, left, U, admits, card_cols):
    """The ring past the pick's admits (xstream.rs:865-926 XStream::layer): the rule's set (xsplit.rs:467
    stream_tail: the unit clears m_min, an expert streams when max(W, c m) > b/r + f + k m at the beside rate r)
    cut at the balance (xstream.rs:1430 balance_cut: the lane's copies, the admits' first, and the card's columns
    against the union the host keeps) and at the ring's half; priced at the build's costs row. The streamed
    experts' expected counts, hottest first."""
    if p["ring"] < 0.5:
        return []
    v027 = int(p["pick"]) == 1
    cp = p["v027_c_us"] if v027 else p["walk_c_us"]
    Wp = p["v027_W_us"] if v027 else p["walk_W_us"]
    fp, kp = p["walk_f_us"], p["walk_k_us"]
    listed = sum(1 - math.exp(-m) for m in left)
    cols = sum(left)
    s = Wp * listed / (Wp * listed + cp * cols) if cols > 0 else 0.0
    copy_us = p["expert_b_q4"] / (p["lane_gbs"] * 1e3 * (1 - s))
    if U < (copy_us + fp) / (cp - kp) * p["n_expert"] / p["top_k"]:
        return []

    def host_of(m):
        return max(Wp, cp * m)
    cand = sorted((m for m in left if m >= 0.5 and host_of(m) > copy_us + fp + kp * m), reverse=True)
    union = cp * (cols - sum(cand)) + sum(host_of(m) for m in cand)
    lane = admits * copy_us + kp * card_cols
    take = 0
    for m in cand:
        card = copy_us + fp + kp * m
        if lane + card > union - host_of(m):
            break
        lane += card
        union -= host_of(m)
        take += 1
    return cand[: min(take, int(p["ring_half"]))]


def prompt_layer(p, U, pos0, a_prev=0):
    """One layer of a ubatch of U rows starting at position pos0, after a ubatch of the same prompt that admitted
    a_prev experts (0 for the first): (wall ms, timeline dict)."""
    fr = p["front_ratio"]
    rows = sum(min(q, p["qsa_select"]) for q in range(pos0, pos0 + U)) if U < 20000 else 0
    front = ((p["pf_front_tok_ms"] + p["pf_front_extra_tok_ms"]) * U + p["pf_front_att_ms"] * rows) / p["n_layer"] / fr
    down = U * p["pf_down_b_tok"] / (p["pcie_d2h_gbs"] * 1e6)
    up = U * p["hidden"] * 4 / (p["pcie_h2d_gbs"] * 1e6) + p["pf_up_fixed_ms"]
    back = U * p["pf_back_tok_ms"] / fr
    draft = p["draft_experts"] if p["mtp"] >= 0.5 else 0.0
    n_card = int(round((p["card_experts"] - draft) / p["n_layer"]))
    n_tier = int(round(p["tier_experts"] / p["n_layer"]))
    prof = profile(p, U)
    carry = p["ub_carry"] if p["ub_carry"] >= 0 else p["prose_carry_k64"]
    n_hot = max(round(p["prompt_corr"] * n_card), round(carry * a_prev))
    host, card, tier = split_sets(prof, n_card, n_tier, n_hot)
    pool = sorted(card)[: max(0, n_card - int(p["pool_pinned"]))]
    card_cols = sum(card)
    a, bound, victims = pick(p, host, pool, U, card_cols)
    host_left = sorted(host, reverse=True)[a:] + victims
    streamed = ring_stream(p, host_left, U, a, U * p["top_k"] - sum(host_left))
    if streamed:
        host_left = sorted(host_left, reverse=True)[len(streamed):]
    W, c = union_costs(p)
    union = cost_union(host_left, W, c) / 1e3
    listed = sum(1 - math.exp(-m) for m in host_left)
    cols = sum(host_left)
    share = W * listed / (W * listed + c * cols) if cols > 0 else 0.0
    union_gbs = listed * p["expert_b"] / max(union, 1e-9) / 1e6
    beside, alone = copy_rates(p, share, union_gbs)
    b = p["expert_b"]
    ahead = (a - bound) * b / (alone * 1e6)
    beside_ms = bound * b / (beside * 1e6)
    if streamed:
        # the ring's lane (four fill threads) beside the pick's copier: one link and one DRAM between them
        lane = copy_rates(dict(p, copier_threads=p["copier_threads"] + 4), share, union_gbs)[0]
        beside_ms = (bound + len(streamed)) * b / (lane * 1e6)
    union *= 1 + p["union_copy_slow"] * (beside if bound else 0.0)
    adm_cols = sum(sorted(host, reverse=True)[:a])
    tail = p["walk_tail_us_col"] * (card_cols + adm_cols - sum(victims) + sum(streamed)) / 1e3 / fr
    card_chain = ahead + beside_ms + tail
    pick_ms = (p["pf_pick0_ms"] + p["pf_pick1_ms"] * U) if int(p["pick"]) else 0.0
    host_chain = p["pf_host0_ms"] + pick_ms + ahead + union
    tier_chain = 0.0
    if n_tier:
        tier_chain = p["pf_tier_col_us"] * sum(tier) / 1e3 + 2 * U * p["hidden"] * 4 / (p["tier_pcie_gbs"] * 1e6)
    inner = max(card_chain, host_chain, tier_chain)
    serial = front + down + up + back
    wall = serial + inner - p["pf_overlap"] * min(serial, inner)
    return wall, dict(front=front, down=down, card=card_chain, host=host_chain, tier=tier_chain, union=union,
                      copies=ahead + beside_ms, tail=tail, up=up, back=back, admits=a, bound=bound, streamed=len(streamed),
                      host_cols=cols, listed=listed, share=share, copy_gbs=beside, alone_gbs=alone, union_gbs=union_gbs,
                      card_bound=card_chain > host_chain, wall=wall, pick=pick_ms)


def prompt(p, P):
    """A prompt of P positions in ubatches of at most ubatch_max: tok/s and the per-ubatch layer timelines."""
    if P < 1:
        raise Refused(f"a prompt of {P} positions")
    U_max = int(p["ubatch_max"])
    ubs, pos, a_prev = [], 0, 0
    while pos < P:
        U = min(U_max, P - pos)
        wall, tl = prompt_layer(p, U, pos, a_prev)
        ubs.append((U, pos, wall, tl))
        pos += U
        a_prev = tl["admits"]
    walk = sum(p["n_layer"] * w for _, _, w, _ in ubs)
    fr = p["front_ratio"]
    extra = P * (p["pf_prologue_tok_ms"] + p["pf_other_tok_ms"])
    store = P * p["draft_store_tok_ms"] / fr if p["mtp"] >= 0.5 else 0.0
    total = walk + extra + store + p["restore_ms"]
    return P * 1e3 / total, dict(walk_ms=walk, extra_ms=extra, store_ms=store, restore_ms=p["restore_ms"],
                                 total_ms=total, ubatches=ubs)


# ============================================================================ anchors solved

# The anchored rows the model solves rather than reads: (row, machine, the anchor's terms, the metric, the target).
ANCHORS = {
    "verify_row_frac": ("box-a6000", ["mtp=1", "w=4", "E=3.6", "head_rows=65536", "host_slots=131.75",
                                      "host_only_layers=0"], ("window_ms", 512), 36.13),
    "layer_extra_us": ("common-5090", ["mtp=0", "host_slots=45.5", "flips_tok=0.87"], ("decode", 29600), 60.4),
    "union_c_us": ("common-5090", ["mtp=0", "pick=0", "ring=0", "pf_front_extra_tok_ms=0.0537"], ("prompt", 3800), 433.0),
}


def metric(p, what, arg):
    if what == "decode":
        return decode(p, arg)[0]
    if what == "window_ms":
        return decode(p, arg)[1]["window_ms"]
    if what == "prompt":
        return prompt(p, arg)[0]
    raise Refused(f"no metric {what!r}")


def solve(p, name, what, arg, target, lo, hi):
    """Bisection of term `name` in [lo, hi] for metric == target (monotone either way)."""
    def f(x):
        return metric(dict(p, **{name: x}), what, arg) - target
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        raise Refused(f"anchor {name}: the target {target} lies outside the metric over {lo}..{hi} "
                      f"({flo + target:.2f}..{fhi + target:.2f})")
    for _ in range(60):
        mid = (lo + hi) / 2
        fm = f(mid)
        if (fm < 0) == (flo < 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return (lo + hi) / 2


def calibrate(consts, machine, sets=()):
    """The machine's terms with every anchored row re-solved at its anchor (the user's --set of constants
    applies to the solve, its scenario terms do not)."""
    const_sets = [s for s in sets if s.split("=", 1)[0] not in TERMS]
    common = [s for s in const_sets if s.split("=", 1)[0] in consts]
    solved = {}
    for name, (m, extra, (what, arg), target) in ANCHORS.items():
        # A common row's anchor is solved for every machine, a machine row's for its own preset only.
        if name in consts:
            pass
        elif MACHINES[m] != MACHINES[machine]:
            continue
        q = params(consts, m, (const_sets if MACHINES[m] == MACHINES[machine] else common) + extra)
        q.update({k: v for k, v in solved.items() if k in consts})
        c = const_of(consts, m, name)
        solved[name] = solve(q, name, what, arg, target, c.lo, c.hi)
    p = params(consts, machine, sets)
    for k, v in solved.items():
        if not any(s.split("=", 1)[0] == k for s in sets):
            p[k] = v
    return p, solved


# ============================================================================ the report

def fmt_decode(p, depth):
    tps, tl = decode(p, depth)
    w = tl["w"]
    cards = p["n_layer"] - p["host_only_layers"]
    host_b = tl["host_slots"] * cards * p["expert_b"]
    dram_ms = host_b / (p["host_dram_gbs"] * 1e6)
    card_b = p["n_layer"] * (p["dense_front_b"] + p["shared_b"]) + p["head_b"] + (10 * (w if w == 1 else
                                                                                           w * p[f"union_all_w{w}"]) - tl["host_slots"]) * cards * p["expert_b"]
    unit = "window" if w > 1 else "step"
    what = "distinct host experts a layer a window" if w > 1 else "host slots a layer"
    return [f"decode {p['_machine']} depth {depth}: {tps:.1f} tok/s  ("
            + (f"w {w}, E {tl['E']:.2f}, window {tl['window_ms']:.2f} ms" if w > 1 else f"step {tl['step_ms']:.2f} ms")
            + f"; hit {tl['hit']:.3f}{', tier %.3f' % tl['tier_share'] if tl['tier_share'] else ''}, {what} "
            f"{tl['host_slots']:.2f})",
            f"  a layer, us: card front {tl['front']:.1f} | max(card shadow {tl['shadow']:.1f}, host leg {tl['host']:.1f}"
            + (f", tier {tl['tier']:.1f}" if tl["tier"] else "") + f") | back {tl['back']:.1f} -> wall {tl['wall']:.1f};"
            f" outside the layers {tl['out'] / 1e3:.2f} ms, flips {tl['flip'] / 1e3:.2f} ms"
            + (f", the draft {tl['draft_ms']:.2f} ms" if w > 1 else ""),
            f"  a {unit}, ms: card chain {(p['n_layer'] * (tl['front'] + tl['shadow'] + tl['back']) + tl['out']) / 1e3 + tl['draft_ms']:.2f}"
            f" (its bytes {card_b / (p['card_bw_gbs'] * 1e6) + (tl['draft_ms'] and 0):.2f} at {p['card_bw_gbs']:g} GB/s) | host cores"
            f" {p['n_layer'] * tl['host'] / 1e3:.2f} | host DRAM {dram_ms:.2f} | PCIe {p['n_layer'] * tl['pcie'] / 1e3:.2f}"
            f" | NVMe 0 -> wall {tl['window_ms']:.2f}: a sum over layers of front + max(shadow, host) + back"]


def fmt_prompt(p, P):
    tps, d = prompt(p, P)
    lines = [f"prompt {p['_machine']} P {P}: {tps:.0f} tok/s ({d['total_ms']:.0f} ms: walk {d['walk_ms']:.0f}, "
             f"prologue+other {d['extra_ms']:.0f}, draft store {d['store_ms']:.0f}, pool return {d['restore_ms']:.0f})"]
    for U, pos0, wall, t in d["ubatches"][:2] + (d["ubatches"][-1:] if len(d["ubatches"]) > 2 else []):
        lines.append(f"  ubatch U {U} @ {pos0}, a layer ms: front {t['front']:.2f} + down {t['down']:.2f} + max(card "
                     f"{t['card']:.2f} [copies {t['copies']:.2f}, tail {t['tail']:.2f}], host {t['host']:.2f} [pick "
                     f"{t['pick']:.2f}, union {t['union']:.2f}]{', tier %.2f' % t['tier'] if t['tier'] else ''}) + up "
                     f"{t['up']:.2f} + back {t['back']:.2f} = {wall:.2f}")
        lines.append(f"    admits {t['admits']} (backlog bound {t['bound']}){', streamed %d' % t['streamed'] if t['streamed'] else ''}, "
                     f"host columns {t['host_cols']:.0f} over {t['listed']:.0f} experts; copy {t['copy_gbs']:.1f} GB/s beside "
                     f"(share {t['share']:.2f}), union DRAM {t['union_gbs']:.1f} GB/s; {'card' if t['card_bound'] else 'host'}-bound")
    if len(d["ubatches"]) > 3:
        lines.append(f"  ({len(d['ubatches'])} ubatches; the middle ones as the last but their position)")
    return lines


def report(p):
    cfg = p.get("_config", "")
    others = [c for c in CONFIGS[p["_machine"]] if c != cfg]
    out = [f"# {p['_machine']}, configuration {cfg}: {CONFIGS[p['_machine']][cfg][0] if cfg else ''}",
           f"# (0.2.8; every term overridable with --set{'; other configurations: ' + ', '.join(others) if others else ''})"]
    for name, depth in DEPTHS.items():
        out += fmt_decode(p, depth)
    for P in PROMPTS:
        out += fmt_prompt(p, P)
    return out


# ============================================================================ the backtest

def with_sets(p, sets):
    """A copy of the term values `p` with `sets` applied, refused as params() refuses."""
    q = dict(p)
    for s in sets:
        k, v = s.split("=", 1)
        if k not in q:
            raise Refused(f"--set {k}: no term or constant of that name on {p['_machine']}")
        q[k] = float(v)
    return q


A6 = "box-a6000"
B3 = "box-3090"
F5 = "common-5090"
# Conditions shared by rows (A6000 plan a at ctx 4352; 0.2.7 is the v0.2.7 floor rule, one copier; 0.2.5 had no pick
# and no ring; from 0.2.6 the unset residency pins no seed expert (mid-p0-s1, eb18766f) unless the host room moves
# the pin, as on q38bpbug-ab's loads (mid-p130-s1)).
V027 = ["pick=1", "copier_threads=1"]
V026_5090 = ["pick=1", "copier_threads=1", "pf_front_extra_tok_ms=0.0537"]
V025 = ["pick=0", "ring=0", "pf_front_extra_tok_ms=0.0537"]
CH16 = ["card_experts=12502"]           # ctx 16384 with the draft: 11,569 + its 933
BP16 = ["card_experts=12391", "tier_experts=7236", "pick=3", "ring=0"]
ST40 = ["card_experts=12213"]           # ctx 40960 with the draft: 235 a layer x 48 + 933 [derived]

# (id, machine, kind, metric, arg, A sets, B sets, measured, lo, hi, anchor, source)
# kind: effect (B against A, %), zero (B against A inside lo..hi, %), abs (A's tok/s; window_ms: A's ms).
ROWS = [
    ("Amtp", A6, "abs", "window_ms", 560, ["mtp=1", "w=4", "E=3.6", "head_rows=65536", "host_slots=131.75",
                                            "host_only_layers=0"], None, 36.13, 0, 0, "verify_row_frac",
     "rig-log 10-01#q38head-ab pass 36.13 ms (65,536 rows, P 512); 527 host slots a pass (10-01#q38hol-ab)"),
    ("Amtp.4096", A6, "abs", "window_ms", 4144, ["mtp=1", "w=4", "E=3.07", "head_rows=65536", "host_slots=61.5",
                                                  "host_only_layers=0"], None, 34.51, 0, 0, "",
     "rig-log 10-01#q38head-ab pass 34.51 ms (65,536 rows, P 4096); 246 host slots a pass (q38hol-ab)"),
    ("Adec.21", A6, "abs", "decode", 4144, ["mtp=0", "host_slots=21", "flips_tok=0.87"], None, 62.93, 0, 0, "",
     "specs/leader/ppgap/func-arm2-stepstats.log: plain step 15.89 ms after P 4096 at 21 host slots (functional)"),
    ("B1a", A6, "effect", "prompt", 4096, V027, ["pick=2", "copier_threads=1"], 14.04, 13.1, 15.0, "",
     "docs/cards/pickrule-ab.card: the walk against v0.2.7's floor, pp4096, draft on"),
    ("B1b", A6, "effect", "prompt", 512, V027, ["pick=2", "copier_threads=1"], 4.1, 2.0, 6.2, "",
     "pickrule-ab sitting: pp512 1,017.69 against 977.82"),
    ("B2a", A6, "effect", "prompt", 4096, ["copier_threads=1"], ["copier_threads=4"], 4.56, 4.31, 4.81, "",
     "specs/worker2/pickrate-impl/box/c1ab.log; docs/cards/pickrate-c1-ab.card"),
    ("B2b", A6, "zero", "prompt", 512, ["copier_threads=1"], ["copier_threads=4"], 1.92, -3.58, 7.42, "",
     "c1ab.log pp512 +1.92 +- 5.50 ('not slower'): a copier under the union's shadow"),
    ("B3a", A6, "effect", "prompt", 4096, V027, [], 19.3, 17.2, 21.4, "",
     "rig-log 10-09#rel028-ab: pp4096 1,673 against 1,403"),
    ("B3b", A6, "effect", "prompt", 512, V027, [], 7.3, 3.9, 10.7, "", "rig-log 10-09#rel028-ab: pp512"),
    ("B3c", A6, "zero", "decode", 560, V027, [], -1.0, -7.0, 5.0, "", "rel028-ab decode 0.99 +- 0.06 (P 512)"),
    ("B3d", A6, "zero", "decode", 4144, V027, [], 2.0, -2.0, 6.0, "", "rel028-ab decode 1.02 +- 0.04 (P 4096)"),
    ("B3e", A6, "abs", "prompt", 4096, [], None, 1673, 0, 0, "", "rel028-ab: 0.2.8 pp4096"),
    ("B3f", A6, "abs", "prompt", 512, [], None, 1026, 0, 0, "", "rel028-ab: 0.2.8 pp512"),
    ("B3g", A6, "abs", "decode", 4144, ["E=3.10"], None, 90.5, 0, 0, "", "rel028-ab: 0.2.8 decode after P 4096 (E 3.10)"),
    ("B4a", A6, "effect", "decode", 560, ["mtp=0", "hit=0.713"],
     ["mtp=0", "hit=0.713", "tier_experts=7236", "tier_hit=0.217"], 3.53, 2.72, 4.34, "",
     "docs/cards/q38bpbug-ab.card: bp/a 1.0353 +- 0.0081, draft off, P 512; slots from its legs (79.1 a, 2.2 tier)"),
    ("B4b", A6, "effect", "prompt", 512, ["mtp=0", "pool_pinned=130"] + V027,
     ["mtp=0", "pool_pinned=130", "copier_threads=1", "pick=3", "ring=0", "tier_experts=7236"], 38.1, 34.2, 42.0, "",
     "q38bpbug-ab.card: pp512 bp/a 1.3811 +- 0.0388 (a's floor STREAM_FLOOR below m_min; bp admit mode)"),
    ("B4c", A6, "abs", "decode", 560, ["mtp=0", "hit=0.713"], None, 61.72, 0, 0, "", "q38bpbug-ab: a 61.72"),
    ("B4d", A6, "abs", "prompt", 512, ["mtp=0", "pool_pinned=130", "copier_threads=1", "pick=3", "ring=0",
                                         "tier_experts=7236"],
     None, 1427.77, 0, 0, "", "q38bpbug-ab: bp pp512 1,427.77"),
    ("B5a", A6, "effect", "decode", 29600, ["mtp=0", "host_slots=156.6", "flips_tok=6.0"],
     ["mtp=0", "host_slots=42.4", "flips_tok=0.87"], 11.1, 8.0, 14.3, "",
     "notes-final.md (56.6 -> 62.9); slots and flips specs/leader/longgap/report2.md §2"),
    ("B5b", A6, "effect", "decode", 3800, ["mtp=0", "host_slots=139", "flips_tok=5.84"],
     ["mtp=0", "host_slots=42.4", "flips_tok=1.23"], 4.5, 2.0, 7.0, "", "notes-final.md (60.5 -> 63.2); report2 §2"),
    ("B6a", A6, "abs", "prompt", 5536, CH16 + ["restore_ms=870"], None, 1294, 0, 0, "",
     "notes-final.md release check: a, long tape 5,536, pool return 0.87 s"),
    ("B6b", A6, "abs", "prompt", 538, CH16 + ["restore_ms=210"], None, 741, 0, 0, "", "release check: a, code 538"),
    ("B6c", A6, "abs", "prompt", 5536, BP16 + ["restore_ms=560"], None, 1451, 0, 0, "", "release check: bp, long"),
    ("B6d", A6, "abs", "prompt", 538, BP16 + ["restore_ms=140"], None, 979, 0, 0, "", "release check: bp, code"),
    ("B6e", A6, "abs", "decode", 5600, CH16 + ["hit=0.9195", "E=2.87"], None, 74.0, 0, 0, "", "release check: a, long"),
    ("B6f", A6, "abs", "decode", 600, CH16 + ["hit=0.9195", "E=3.37"], None, 96.7, 0, 0, "", "release check: a, code"),
    ("B6g", A6, "abs", "decode", 128, CH16 + ["hit=0.9195", "E=3.13"], None, 90.8, 0, 0, "", "release check: a, short"),
    ("B6h", A6, "abs", "decode", 5600, BP16 + ["hit=0.9174", "tier_hit=0.056", "E=2.78"], None, 81.7, 0, 0, "",
     "release check: bp, long (hits: the prose replay at 11,458 and 18,694)"),
    ("B6i", A6, "abs", "decode", 600, BP16 + ["hit=0.9174", "tier_hit=0.056", "E=3.40"], None, 106, 0, 0, "",
     "release check: bp, code"),
    ("B6j", A6, "abs", "decode", 128, BP16 + ["hit=0.9174", "tier_hit=0.056", "E=3.21"], None, 98.8, 0, 0, "",
     "release check: bp, short"),
    ("B6k", A6, "abs", "prompt", 28300, ST40 + ["restore_ms=345"], None, 1766, 0, 0, "",
     "release check, #3 stream ctx 40960: a ~28.3K 1,757-1,775 (return 0.34-0.35 s)"),
    ("B6l", A6, "abs", "prompt", 3650, ST40 + ["restore_ms=210"], None, 1854, 0, 0, "",
     "#3 stream: a 3.5K-3.8K 1,836-1,872 (return 0.19-0.22 s)"),
    ("B7.a", F5, "abs", "decode", 29600, ["mtp=0", "host_slots=45.5", "flips_tok=0.87"], None, 60.4, 0, 0,
     "layer_extra_us", "issue3-thread.md: 0.2.6 MTP off BLOOMERY_XSTREAM=off after ~29.6K (45.5 slots [derived])"),
    ("B7.b", F5, "abs", "prompt", 3800, ["mtp=0"] + V025, None, 433, 0, 0, "union_c_us",
     "issue1-comments.md: 0.2.5 MTP off prefill 4K 433"),
    ("B7.c", F5, "abs", "prompt", 29600, ["mtp=0"] + V025, None, 416, 0, 0, "", "issue1: 0.2.5 MTP off 32K 416"),
    ("B7.d", F5, "abs", "prompt", 3800, V025, None, 407, 0, 0, "", "issue1: 0.2.5 MTP on 4K 407"),
    ("B7.e", F5, "abs", "prompt", 29600, V025, None, 390, 0, 0, "", "issue1: 0.2.5 MTP on 32K 390"),
    ("B7.f", F5, "abs", "prompt", 3800, ["mtp=0"] + V026_5090, None, 1537, 0, 0, "",
     "issue3-thread.md: 0.2.6 MTP off ~3.8K 1,537"),
    ("B7.g", F5, "abs", "prompt", 29600, ["mtp=0"] + V026_5090, None, 1982, 0, 0, "", "issue3: 0.2.6 MTP off ~29.6K 1,982"),
    ("B7.h", F5, "abs", "prompt", 3800, V026_5090, None, 1473, 0, 0, "", "issue3: 0.2.6 D0a (MTP on) 1,473"),
    ("B7.i", F5, "abs", "prompt", 29600, V026_5090, None, 1875, 0, 0, "", "issue3: 0.2.6 D0a 1,875"),
    ("B7.j", F5, "abs", "prompt", 29600, V025, None, 390, 0, 0, "", "issue3: 0.2.6 XSTREAM=off MTP on ~29.6K 390"),
    ("B7.k", F5, "abs", "prompt", 29600, ["pick=1", "copier_threads=1"], None, 2030, 0, 0, "",
     "issue3: 0.2.7 MTP on ~29.6K 2,023 / 2,036"),
    ("B7.l", F5, "abs", "decode", 128, ["mtp=0", "hit=0.8825", "flips_tok=0.87"], None, 46.1, 0, 0, "",
     "issue1: 0.2.5 MTP off short 46.1 (hit: chat 0.86 at 5,482 scaled by the replay's misses to 6,415)"),
    ("B7.m", F5, "abs", "decode", 3800, ["mtp=0", "hit=0.8825", "flips_tok=0.87"], None, 45.7, 0, 0, "", "issue1: 4K 45.7"),
    ("B7.n", F5, "abs", "decode", 29600, ["mtp=0", "hit=0.8825", "flips_tok=0.87"], None, 46.2, 0, 0, "", "issue1: 32K 46.2"),
    ("B7.o", F5, "abs", "decode", 128, ["chat=1", "E=2.38"], None, 48.8, 0, 0, "", "issue1: 0.2.5 MTP on short 48.8 (acceptance 0.46)"),
    ("B7.p", F5, "abs", "decode", 3800, ["chat=1", "E=2.38"], None, 53.0, 0, 0, "", "issue1: 4K 53.0"),
    ("B7.q", F5, "abs", "decode", 29600, ["chat=1", "E=2.38"], None, 47.4, 0, 0, "", "issue1: 32K 47.4"),
    ("B7.r", F5, "abs", "decode", 128, ["mtp=0", "host_slots=45.5", "flips_tok=2"], None, 58.2, 0, 0, "",
     "issue3: 0.2.6 MTP off short 58.2 (slots and flips: report2's replay of their stream)"),
    ("B7.s", F5, "abs", "decode", 3800, ["mtp=0", "host_slots=119.5", "flips_tok=5.84"], None, 48.3, 0, 0, "",
     "issue3: 0.2.6 MTP off ~3.8K 48.3"),
    ("B7.t", F5, "abs", "decode", 29600, ["mtp=0", "host_slots=141.5", "flips_tok=6.0"], None, 40.25, 0, 0, "",
     "issue3: 0.2.6 MTP off ~29.6K 41.6 / 38.9"),
    ("B8a", A6, "effect", "decode", 560, ["mtp=0", "host_slots=235.8", "host_only_layers=5", "flips_tok=0"],
     ["mtp=1", "E=3.46", "head_rows=248320", "host_slots=253", "host_only_layers=5", "flips_tok=0"], 51.9, 45.0, 59.0, "",
     "rig-log 09-30#q38mtp-speed after the fix (52.77 -> 80.18), residency off; E 09-30#q38res-hit"),
    ("B8b", A6, "effect", "decode", 4144, ["mtp=0", "host_slots=235.8", "host_only_layers=5", "flips_tok=0"],
     ["mtp=1", "E=3.06", "head_rows=248320", "host_slots=253", "host_only_layers=5", "flips_tok=0"], 31.1, 25.0, 37.0, "",
     "09-30#q38mtp-speed (53.17 -> 69.72)"),
    ("B8c", F5, "effect", "decode", 128, ["mtp=0", "hit=0.8825", "flips_tok=0.87"], ["chat=1", "E=2.38"], 5.9, 0, 0, "",
     "issue1: 0.2.5 chat MTP gain short (46.1 -> 48.8)"),
    ("B8d", F5, "effect", "decode", 3800, ["mtp=0", "hit=0.8825", "flips_tok=0.87"], ["chat=1", "E=2.38"], 16.0, 0, 0, "",
     "issue1: 4K (45.7 -> 53.0)"),
    ("B8e", F5, "effect", "decode", 29600, ["mtp=0", "hit=0.8825", "flips_tok=0.87"], ["chat=1", "E=2.38"], 2.6, 0, 0, "",
     "issue1: 32K (46.2 -> 47.4)"),
    ("B8f", F5, "effect", "decode", 29600, ["mtp=0", "host_slots=45.5", "flips_tok=0.87"],
     ["host_slots=54.2", "E=2.44"], 9.6, 0, 0, "",
     "issue3: XSTREAM=off, MTP width fixed 66.2 against off 60.4 (slots 42.4 x the replay's misses 6,810 -> 5,482)"),
    ("B9a", A6, "effect", "decode", 4144, ["mtp=0", "host_slots=180", "host_only_layers=5"],
     ["mtp=0", "host_slots=77", "host_only_layers=5"], 7.3, 4.0, 10.6, "",
     "rig-log 09-30#q38seed-ab plain P 4096 (slots 180 -> 77; naive slots x W reads +16 %)"),
    ("B9b", A6, "zero", "decode", 560, ["mtp=0", "host_slots=181", "host_only_layers=5"],
     ["mtp=0", "host_slots=136", "host_only_layers=5"], 5.6, -0.3, 11.5, "", "09-30#q38seed-ab plain P 512 +5.6 +- 5.9"),
    # Added after the first backtest, before it was run (the advisor's measured-zero row): the triple's stage 1, the
    # costs row alone under v0.2.7's floor rule, predicted -239..-143 ms by its round.
    ("B9c", A6, "zero", "prompt", 4096, ["mtp=0", "pick=1", "copier_threads=1"],
     ["mtp=0", "pick=1", "copier_threads=1", "v027_c_us=3.67", "v027_W_us=19.95"], 0.1, -3.0, 3.3, "",
     "specs/worker2/pickrule-impl/triple-report.md: stage 1 alone mean -3 ms (+95 / -100) on s0b's 3,112.8"),
    # The primary target (the 2026-10-09 amendment): the README's 3090 row.
    ("B10", B3, "abs", "prompt", 512, CONFIGS["box-3090"]["readme"][1], None, 486.0, 0, 0, "",
     "rig-log 10-06#num3090 / README.md 'Numbers', the 'One RTX 3090 24 GB' table: pp512 486.0, --place gate, mid-p32-s1, draft off, 0.2.1 b83f1269 "
     "(its decode row, two requests 60.71, is the pair pass: out of scope)"),
]


def row_value(p, metric_name, arg):
    return metric(p, metric_name, arg)


def backtest(consts, out=print):
    cal = {m: calibrate(consts, m)[0] for m in MACHINES}
    res = []
    for (rid, m, kind, met, arg, A, B, meas, lo, hi, anchor, src) in ROWS:
        pa = with_sets(cal[m], A)
        va = row_value(pa, met, arg)
        if kind == "abs":
            pred = va
            ratio = pred / meas
            sign_ok = True
            mag_ok = 1 / 1.5 <= ratio <= 1.5
            shown = (f"{pred:9.2f}", f"{meas:9.2f}", f"{ratio:6.3f}")
        else:
            vb = row_value(with_sets(cal[m], B), met, arg)
            pred = (vb / va - 1) * 100
            if kind == "zero":
                sign_ok = mag_ok = lo <= pred <= hi
                ratio = float("nan")
                shown = (f"{pred:+8.2f}%", f"[{lo:+.1f},{hi:+.1f}]", "  zero")
            else:
                sign_ok = (pred > 0) == (meas > 0) and pred != 0
                ratio = abs(pred) / abs(meas)
                mag_ok = sign_ok and 1 / 1.5 <= ratio <= 1.5
                shown = (f"{pred:+8.2f}%", f"{meas:+8.2f}%", f"{ratio:6.3f}")
        res.append((rid, m, kind, met, arg, anchor, sign_ok, mag_ok, shown, src))
    out(f"{'row':9s} {'machine':11s} {'kind':6s} {'metric':>14s} {'predicted':>10s} {'measured':>14s} {'ratio':>6s}  verdict")
    for rid, m, kind, met, arg, anchor, sign_ok, mag_ok, shown, src in res:
        v = f"anchor ({anchor})" if anchor else ("pass" if sign_ok and mag_ok else
                                                 ("FAIL sign" if not sign_ok else "FAIL x1.5"))
        out(f"{rid:9s} {m:11s} {kind:6s} {met + ' ' + str(arg):>14s} {shown[0]:>10s} {shown[1]:>14s} {shown[2]:>6s}  {v}")
    scored = [r for r in res if not r[5]]
    n = len(scored)
    signs = sum(1 for r in scored if r[6])
    mags = sum(1 for r in scored if r[7])
    ok = signs == n and mags >= 0.7 * n
    out(f"scored {n} rows (anchors {len(res) - n} not scored): sign right {signs}/{n}, inside x1.5 {mags}/{n} "
        f"({100 * mags / max(n, 1):.0f} %); the bar (every sign, >= 70 %): {'PASS' if ok else 'FAIL'}")
    by = {}
    for r in scored:
        key = (r[1], r[3])
        a, b = by.get(key, (0, 0))
        by[key] = (a + 1, b + int(r[6] and r[7]))
    out("by machine and metric: " + "; ".join(f"{m} {met} {b}/{a}" for (m, met), (a, b) in sorted(by.items())))
    return ok, res


# ============================================================================ the ceilings

def decode_ceiling(p, depth, hit):
    """Per token: each resource's bytes and time for one decoded token at card hit `hit` (the plain step, and the
    window's share when the draft is on), ms."""
    q = dict(p, hit=hit, host_slots=-1.0)
    E = 1.0
    w = 1
    if q["mtp"] >= 0.5:
        w = int(q["w"])
        E = q["E"] if q["E"] > 0 else (q["E_chat"] if q["chat"] >= 0.5 else q["E_prose"])
    h, t, hh = slot_shares(q)
    cards = q["n_layer"] - q["host_only_layers"]
    if w == 1:
        k_dist = 10.0
        k_host = 10 * hh
    else:
        k_dist = 10 * w * q[f"union_all_w{w}"]
        k_host = 10 * w * hh * q[f"union_miss_w{w}"]
    card_b = (q["n_layer"] * (q["dense_front_b"] + q["shared_b"]) + q["head_b"] + attn_b(q, depth) * q["n_layer"] * w
              + cards * (k_dist - k_host) * q["expert_b"])
    if w > 1:
        card_b += (w - 1) * (q["mtp_dense_b"] + 10 * q["mtp_expert_b"] + q["head_rows"] * q["hidden"] * 34 / 32)
    host_b = cards * k_host * q["expert_b"] + q["host_only_layers"] * k_dist * q["expert_b"]
    pcie_b = q["n_layer"] * w * (2 * q["hidden"] * 4 + 160)
    card_ms = card_b / (q["card_bw_gbs"] * 1e6) / E
    dram_ms = host_b / (q["host_dram_gbs"] * q["host_w_eff"] * 1e6) / E
    pcie_ms = pcie_b / (q["pcie_h2d_gbs"] * 1e6) / E
    return dict(card=(card_b / E, card_ms), dram=(host_b / E, dram_ms), pcie=(pcie_b / E, pcie_ms))


def prompt_ceiling(p, U):
    """One ubatch of U rows a layer at the machine's true costs, ms: per resource its demand at the split of the
    host's experts between the copy link and the host union (hottest first, the pool at the unit's hottest ranks:
    the oracle) that minimizes the largest demand (the perfect-overlap bound), and the split that minimizes their
    sum (the serial bound). Card SMs: the front, the back and the route over every card column."""
    n_card = int(round((p["card_experts"] - (p["draft_experts"] if p["mtp"] >= 0.5 else 0)) / p["n_layer"]))
    prof = profile(p, U)
    host, card, _ = split_sets(prof, n_card, 0, n_card)
    rows = sum(min(q, p["qsa_select"]) for q in range(U))
    fr = p["front_ratio"]
    front = (p["pf_front_tok_ms"] * U + p["pf_front_att_ms"] * rows) / p["n_layer"] / fr + U * p["pf_back_tok_ms"] / fr
    b = p["expert_b"]
    hs = sorted(host, reverse=True)
    best = serial = None
    for j in range(len(hs) + 1):
        rest = hs[j:]
        d = dict(card=front + p["walk_k_us"] * (sum(card) + sum(hs[:j])) / 1e3 / fr,
                 link=j * b / (p["pcie_h2d_gbs"] * 1e6),
                 union=cost_union(rest, *union_costs(p)) / 1e3,
                 dram=(3 * j * b + sum(1 - math.exp(-m) for m in rest) * b) / (p["host_dram_gbs"] * 1e6))
        mx = max(d.values())
        sm = d["card"] + d["link"] + d["union"]
        if best is None or mx < best[0]:
            best = (mx, j, d)
        if serial is None or sm < serial[0]:
            serial = (sm, j, d)
    return best, serial


def ceilings(p, out=print):
    m = p["_machine"]
    out(f"# {m}: ceilings per token (decode at depth 3.8K; prompt by its 4096- or 512-row ubatch)")
    N = int(p["card_experts"] - (p["draft_experts"] if p["mtp"] >= 0.5 else 0))
    tps_today, tl = decode(p, 3800)
    hits = [("today (prose mid replay)", tl["hit"])]
    for N0 in sorted({5482, 12567} | ({N} if f"hit_belady_{N}" in p else set())):
        hits.append((f"Belady at {N0} (prose){' *' if N0 == N else ''}", p[f"hit_belady_{N0}"]))
    hits.append(("every expert on the card", 1.0))
    E = tl["E"]
    out(f"decode, the draft {'on (w %d, E %.2f)' % (tl['w'], E) if tl['w'] > 1 else 'off'}; "
        f"today's model {tps_today:.1f} tok/s at hit {tl['hit']:.3f} ({N} card experts)")
    out(f"  (* this plan's own card experts)")
    out(f"  {'residency':26s} {'card GB':>8s} {'card ms':>8s} {'DRAM GB':>8s} {'DRAM ms':>8s} {'PCIe MB':>8s} "
        f"{'PCIe ms':>8s} {'overlap tok/s':>14s} {'serial tok/s':>13s}")
    for name, h in hits:
        c = decode_ceiling(p, 3800, h)
        mx = max(v[1] for v in c.values())
        sm = sum(v[1] for v in c.values())
        out(f"  {name:26s} {c['card'][0] / 1e9:8.3f} {c['card'][1]:8.3f} {c['dram'][0] / 1e9:8.3f} {c['dram'][1]:8.3f} "
            f"{c['pcie'][0] / 1e6:8.3f} {c['pcie'][1]:8.3f} {1e3 / mx:14.1f} {1e3 / sm:13.1f}")
    c = decode_ceiling(p, 3800, tl["hit"])
    bound_res = max(c, key=lambda k: c[k][1])
    win = tl["window_ms"] / E
    card_chain = (p["n_layer"] * (tl["front"] + tl["shadow"] + tl["back"]) + tl["out"] + tl["draft_ms"] * 1e3) / 1e3 / E
    exposed = max(p["n_layer"] * (tl["wall"] - tl["front"] - tl["shadow"] - tl["back"]) / 1e3 / E, 0.0)
    host_leg = p["n_layer"] * tl["host"] / 1e3 / E
    out(f"  today {win:.2f} ms a token against the overlap bound's {c[bound_res][1]:.2f} ({bound_res}): the card chain "
        f"{card_chain:.2f} ms (its bytes {c['card'][1]:.2f}, latency {card_chain - c['card'][1]:.2f}), the host leg "
        f"{host_leg:.2f} ms of which {exposed:.2f} past the card's shadow, flips {tl['flip'] / 1e3 / E:.2f}")
    if bound_res == "card":
        why = (f"the card's latency chain: {card_chain - c['card'][1]:.2f} ms a token of launches and syncs over its "
               f"bytes' {c['card'][1]:.2f}")
    else:
        why = (f"the card's front and back, serial with a host leg near its DRAM bound ({host_leg:.2f} against "
               f"{c['dram'][1]:.2f} ms): the leg overlaps only the shadow, so {card_chain - tl['shadow'] * p['n_layer'] / 1e3 / E:.2f} ms "
               f"of card chain adds to it")
    out(f"  verdict (decode): the gap sits on {why}")
    for P in PROMPTS:
        U = min(P, int(p["ubatch_max"]))
        (mx, j, d), (sm, js, ds) = prompt_ceiling(p, U)
        tps, rec = prompt(p, P)
        t = rec["ubatches"][0][3]
        k = p["n_layer"] / U

        def tok(x):
            return 1e3 / (x * k + P * (p["pf_prologue_tok_ms"] + p["pf_other_tok_ms"]) / P)
        out(f"prompt P {P}: a layer of {U} rows at the oracle split ({j} host experts copied): card SMs {d['card']:.2f}, "
            f"link {d['link']:.2f}, host union {d['union']:.2f}, host DRAM {d['dram']:.2f} ms; NVMe 0")
        out(f"  bounds: perfect overlap {tok(mx):.0f} tok/s ({max(d, key=d.get)}), serial {tok(sm):.0f} tok/s "
            f"({js} copied); today's model {tps:.0f} tok/s, a layer {t['wall']:.2f} ms = card front, down, up, back "
            f"{t['front'] + t['down'] + t['up'] + t['back']:.2f} + max(card chain {t['card']:.2f}, host chain {t['host']:.2f})")
        inner = max(t["host"], t["card"])
        excess = inner - max(d["link"], d["union"], d["dram"])
        gap = t["wall"] - mx
        side = "host chain" if t["host"] >= t["card"] else "card chain"
        where = (f"the {side}'s excess over the oracle split's ({excess:.2f})" if excess >= gap - excess else
                 f"the code's serial order, the card's front and back before and after the max ({gap - excess:.2f})")
        out(f"  verdict (prompt {P}): the gap {gap:.2f} ms a layer = the {side} past the oracle split's inner demand "
            f"{excess:.2f} + the card's front and back serial with the max, which a perfect overlap hides, "
            f"{gap - excess:.2f}; it sits mostly on {where}")


# ============================================================================ the tornado

def swept_rows(consts, machine):
    """The machine's rows with a band to sweep: every assumed row, then every derived or measured one, but the
    anchored rows calibrate() solves (their band is the solver's range)."""
    pre = MACHINES[machine]
    rows = []
    for name, c in consts.items():
        if c.lo == c.hi or any(name.startswith(o) for o in MACHINES.values() if o != pre):
            continue
        short = name[len(pre):] if name.startswith(pre) else name
        if short in ANCHORS:
            continue
        rows.append((KINDS.index(c.kind) != 2, name, short))
    return [(n, s) for _, n, s in sorted(rows, key=lambda r: r[0])]


def tornado(consts, machine=F5, csets=(), out=print):
    csets = list(csets)
    base, _ = calibrate(consts, machine, csets)
    metrics = [("decode", d) for d in DEPTHS.values()] + [("prompt", P) for P in PROMPTS]
    vals0 = [metric(base, w, a) for w, a in metrics]
    out(f"# {machine}: today's prediction at each banded constant's ends (assumed first, then derived and measured),"
        f" the anchors re-solved each time; sorted by the widest move within each kind")
    out(f"{'constant':28s} {'kind':8s} {'band end':>14s} " + " ".join(f"{w[:3]} {a:>6}" for w, a in metrics))
    out(f"{'(today)':28s} {'':8s} {'':>14s} " + " ".join(f"{v:10.1f}" for v in vals0))
    rows = []
    for name, short in swept_rows(consts, machine):
        c = consts[name]
        ends = []
        for end in (c.lo, c.hi):
            try:
                p, _ = calibrate(consts, machine, csets + [f"{short}={end}"])
                ends.append((end, [metric(p, w, a) for w, a in metrics], ""))
            except Refused as e:
                ends.append((end, [float("nan")] * len(metrics), str(e)))
        spread = max((abs(x / y - 1) for _, v, _ in ends for x, y in zip(v, vals0) if x == x), default=0.0)
        rows.append((c.kind != "assumed", -spread, short, c, ends))
    for _, _, short, c, ends in sorted(rows, key=lambda r: (r[0], r[1])):
        for end, v, why in ends:
            tag = ("lo " if end == c.lo else "hi ") + format(end, "g")
            cells = " ".join(f"{(x / y - 1) * 100:+9.1f}%" if x == x else "      n/a" for x, y in zip(v, vals0))
            out(f"{short:28s} {c.kind:8s} {tag:>14s} {cells}{'  (' + why + ')' if why else ''}")
    return rows


# ============================================================================ --explain

FORMULAS = """\
decode, a layer (us), the plain step (w = 1) or a w-row verify:
  front  = (dense_front_b + attn_b(depth) w - attn_b(6)) / card_bw + lat.front x lat_ratio x grow
  shadow = shared_b / card_bw + lat.shared x lat_ratio x grow + leg_launch x lat_ratio + k_card x expert_b / card_bw
           + k_host x miss_card_share x expert_b / pcie_h2d
  host   = handoff + host_svc + k_host (1 - miss_card_share) max(W, union_c x cols)   [W = host_w_us or
           expert_b / (host_dram x host_w_eff)]
  tier   = tier_fixed + k_tier x tier_slot                                             [place bp]
  back   = lat.back x lat_ratio x grow + layer_extra_us
  wall   = front + max(shadow, host, tier) + back      (layer_overlap 0: their sum)
  lat.X  = the A6000 anchor X less its bytes at bw_a6000;   grow = 1 + verify_row_frac (w - 1)
  k_*: the layer's slots on each side, a beta-binomial over the 10 (or the window's 10 w union_all) distinct
  experts at the host share, overdispersed by miss_dispersion; a window's host expert is read once at
  1 / union_miss columns
step  = sum over layers + out (head_b / card_bw + lat.out x grow) + flips x flip_cpu_us x flip_crit
window = step(w) + draft (w - 1 walks of mtp_dense_b + 10 mtp_expert_b + head_rows x 2720 B, + mtp_walk_lat)
         + mtp_readback + window_host;   tok/s = E / window
prompt, a layer of a ubatch of U rows (ms):
  front = (pf_front_tok x U + pf_front_att x sum min(q, qsa_select)) / 48 / front_ratio;  down/up its bytes at
  pcie; the pool: card_experts / 48 less the draft's, n_hot of them at the unit's hottest ranks (prompt_corr,
  ub_carry x the last ubatch's admits); pick (0..3) over the host experts, each admit pairing the coldest
  unpinned resident; the ring (stream_tail + balance_cut at the build's costs, at most ring_half)
  card chain = admits past the backlog bound alone + the bound and ring beside the union + the route's tail
  host chain = pf_host0 + pick + the alone copies + union (W x listed + union_c x cols) x (1 + copy_slow x GB/s)
  wall = S + I - pf_overlap x min(S, I), S = front + down + up + back, I = max(card, host, tier);
  tok/s = P / (sum of walls + P x (prologue + other) + draft store + restore)"""


def explain(consts, machine, out=print):
    p, solved = calibrate(consts, machine)
    out(FORMULAS)
    out("")
    out(f"# terms a scenario sets (--set NAME=VALUE), today's default on {machine}")
    for k, (v, doc) in TERMS.items():
        out(f"  {k:24s} {v:>10g}  {doc}")
    out("")
    out(f"# constants on {machine} (kind, band, source); anchors re-solved: "
        + ", ".join(f"{k}={v:.4g}" for k, v in solved.items()))
    pre = MACHINES[machine]
    for name, c in consts.items():
        if any(name.startswith(o) for o in MACHINES.values() if o != pre):
            continue
        short = name[len(pre):] if name.startswith(pre) else name
        val = solved.get(short, c.value)
        out(f"  {short:28s} {val:>12.6g} [{c.lo:g}, {c.hi:g}] {c.unit:22s} {c.kind:8s} {c.anchor or '-':16s} {c.source[:90]}")


# ============================================================================ --self-test

def self_test():
    import tempfile
    fails = []

    def check(cond, what):
        if not cond:
            fails.append(what)

    def refused(fn, what, needle):
        try:
            fn()
        except Refused as e:
            check(needle in str(e), f"{what}: refused with {e!r}, wants {needle!r}")
            return
        fails.append(f"{what}: not refused")

    consts = load_constants()
    # The presets are consistent: a machine row on one machine is a row on the other; every band holds its value
    # (load_constants refuses otherwise); every assumed row has a band to sweep.
    prefixes = sorted(set(MACHINES.values()))
    for a in prefixes:
        for b in prefixes:
            for name in consts:
                if a != b and name.startswith(a):
                    check(b + name[len(a):] in consts, f"preset: {name} has no {b} twin")
    for m in MACHINES:
        for cname in CONFIGS[m]:
            params(consts, m, config_sets(m, cname)[1])
    for name, c in consts.items():
        check(c.kind != "assumed" or c.lo < c.hi, f"preset: assumed {name} has no band")
        check(c.source.strip() != "", f"preset: {name} has no source")
    # A hand-worked layer: the A6000's latency parts at 674 GB/s, 8 card slots and 2 host slots, depth 6.
    p = params(consts, "box-a6000", ["mtp=0", "host_slots=96"])
    parts = layer_parts(p, 6)
    slot = 3133867 / 674e3                                     # 4.6496 us
    check(abs(parts["slot"] - slot) < 1e-6, f"hand layer: slot {parts['slot']} against {slot}")
    check(abs(parts["front"] - 194.0) < 1e-6, f"hand layer: front {parts['front']} against 194 (the anchor)")
    wall, r = layer_wall(p, parts, 8, 0, 2)
    shadow = 24.0 + 14.0 + 8 * slot                            # 75.197
    host = 4.0 + 8.0 + 2 * 23.36                               # 58.72
    check(abs(r["shadow"] - shadow) < 1e-6 and abs(r["host"] - host) < 1e-6,
          f"hand layer: shadow {r['shadow']} host {r['host']} against {shadow}, {host}")
    check(abs(wall - (194.0 + shadow + 30.5)) < 1e-6, f"hand layer: wall {wall} against {194.0 + shadow + 30.5}")
    wall0, _ = layer_wall(dict(p, layer_overlap=0.0), parts, 8, 0, 2)
    check(abs(wall0 - (194.0 + shadow + host + 30.5)) < 1e-6, f"hand layer: serial wall {wall0}")
    # Three host slots put the host leg past the shadow: 12 + 3 x 23.36 = 82.08 against 24 + 14 + 7 x 4.6496.
    wall3, r3 = layer_wall(p, parts, 7, 0, 3)
    check(abs(wall3 - (194.0 + 82.08 + 30.5)) < 1e-6, f"hand layer: host-bound wall {wall3}")
    # The beta-binomial: sums to 1, mean n p, the binomial at dispersion 1, wider past it.
    for disp in (1.0, 2.38):
        pk = betabinom(10, 0.2, disp)
        mean = sum(k * x for k, x in enumerate(pk))
        var = sum((k - mean) ** 2 * x for k, x in enumerate(pk))
        check(abs(sum(pk) - 1) < 1e-9 and abs(mean - 2.0) < 1e-6, f"betabinom({disp}): sum {sum(pk)} mean {mean}")
        check(abs(var / (10 * 0.2 * 0.8) - disp) < 1e-6, f"betabinom({disp}): variance ratio {var / 1.6}")
    # The prompt's profile rebuilds its rows and holds a unit's columns.
    for U in (512, 4096):
        prof = profile(p, U)
        for rk in RANKS:
            check(abs(prof[rk - 1] - p[f"prose_c{U}_r{rk}"]) < 1e-6, f"profile U {U} rank {rk}")
        check(abs(sum(prof) / (U * 10) - 1) < 0.08, f"profile U {U}: {sum(prof):.0f} columns against {U * 10}")
    h, c, t = split_sets(profile(p, 512), 100, 50, 20)
    check(len(h) == 362 and len(c) == 100 and len(t) == 50, f"split_sets: {len(h)}, {len(c)}, {len(t)}")
    check(abs(sum(c[:20]) - sum(profile(p, 512)[:20])) < 1e-9, "split_sets: n_hot at the hottest ranks")
    # The anchors: each re-solve lands inside its row's band, reproduces its target, and equals the stored value.
    for m in MACHINES:
        q, solved = calibrate(consts, m)
        for name, x in solved.items():
            am, extra, (what, arg), target = ANCHORS[name]
            got = metric(with_sets(dict(params(consts, am, extra), **{k: v for k, v in solved.items()
                                                                   if am == m or k == "verify_row_frac"}),
                                   []), what, arg)
            check(abs(got / target - 1) < 0.005, f"anchor {name} on {m}: {got:.3f} against {target}")
            stored = const_of(consts, am, name).value
            check(abs(x / stored - 1) < 0.01, f"anchor {name}: re-solved {x:.4g}, stored {stored:.4g} (write the re-solved value into its q38-constants.tsv row)")
    # Refusals by name.
    # The thread cap: 12 of 32 threads spread c over 32/12 of the time; W rises only past the channels' rate
    # (147.7 GB/s against 12 x 12 GB/s = 144).
    fw, fc = host_scale(params(consts, "box-3090-lowhost"))
    check(abs(fc - 32 / 12) < 1e-12 and abs(fw - 147.7 / 144.0) < 1e-12, f"host_scale: {fw}, {fc}")
    check(host_scale(params(consts, "box-3090")) == (1.0, 1.0), "host_scale: the uncapped preset")
    refused(lambda: host_scale(params(consts, "box-3090", ["host_threads=40"])), "threads past base", "past the machine")
    refused(lambda: config_sets("box-3090", "nope"), "unknown config", "has no configuration")
    refused(lambda: params(consts, "rtx-4090"), "unknown machine", "no machine")
    refused(lambda: params(consts, "box-a6000", ["nope=1"]), "unknown --set", "no term or constant")
    refused(lambda: params(consts, "box-a6000", ["hit=x"]), "--set not a number", "not a number")
    refused(lambda: params(consts, "box-a6000", ["hit=inf"]), "--set inf", "not a finite number")
    refused(lambda: params(consts, "box-a6000", ["hit"]), "--set without =", "NAME=VALUE")
    refused(lambda: decode(params(consts, "box-a6000", ["mtp=0", "hit=1.2"]), 128), "hit past 1", "outside [0, 1]")
    refused(lambda: decode(params(consts, "box-a6000", ["w=5"]), 128), "w 5", "1..4 rows")
    refused(lambda: decode(params(consts, "box-a6000", ["E=4.5"]), 128), "E past w", "outside 1..4")
    refused(lambda: prompt(params(consts, "box-a6000"), 0), "P 0", "a prompt of 0")
    head = "name\tvalue\tlo\thi\tunit\tkind\tconditions\tsource\tanchor\tnote\n"
    good = "x\t1\t0\t2\tu\tmeasured\tc\ts\t\t\n"
    bad = {"cells": "x\t1\t0\t2\tu\tmeasured\tc\ts\t\n", "twice": good + good,
           "kind": good.replace("measured", "guessed"), "band": "x\t3\t0\t2\tu\tmeasured\tc\ts\t\t\n",
           "number": "x\tone\t0\t2\tu\tmeasured\tc\ts\t\t\n"}
    needles = {"cells": "cells", "twice": "twice", "kind": "is not one of", "band": "outside its band",
               "number": "not a number"}
    with tempfile.TemporaryDirectory() as d:
        for case, body in bad.items():
            path = os.path.join(d, case + ".tsv")
            with open(path, "w", encoding="utf-8") as f:
                f.write(head + body)
            refused(lambda path=path: load_constants(path), f"constants {case}", needles[case])
        path = os.path.join(d, "ok.tsv")
        with open(path, "w", encoding="utf-8") as f:
            f.write(head + good)
        check(load_constants(path)["x"].value == 1.0, "constants: a good row")
    # The mechanisms move their terms the right way: a share read by the card lowers a host-bound layer's host
    # leg; the serial layer is no faster than the overlapped one; a higher hit is no slower.
    q = params(consts, "box-a6000", ["mtp=0", "host_slots=200"])
    # A share s of k host slots read by the card over PCIe: the host leg keeps k (1 - s) of them, the shadow pays
    # k s x expert_b / pcie_h2d (119.25 us an expert at 26.28 GB/s, past its 23.36 us on the host).
    w4, r4 = layer_wall(dict(p, miss_card_share=0.5), parts, 6, 0, 4)
    check(abs(r4["host"] - (12.0 + 2 * 23.36)) < 1e-6 and abs(r4["pcie"] - 2 * 3133867 / 26.28e3) < 1e-6,
          f"mechanism: miss_card_share host {r4['host']} pcie {r4['pcie']}")
    check(decode(dict(q, layer_overlap=0.0), 128)[0] < decode(q, 128)[0], "mechanism: layer_overlap")
    check(decode(dict(q, host_slots=50), 128)[0] > decode(q, 128)[0], "mechanism: host_slots")
    # pf_overlap hides the serial card part under the layer's max: s 1 leaves max(S, I).
    q = params(consts, "box-3090")
    w0, t0 = prompt_layer(q, 4096, 0)
    w1, _ = prompt_layer(dict(q, pf_overlap=1.0), 4096, 0)
    S = t0["front"] + t0["down"] + t0["up"] + t0["back"]
    inner = max(t0["card"], t0["host"], t0["tier"])
    check(abs(w0 - (S + inner)) < 1e-9 and abs(w1 - max(S, inner)) < 1e-9, f"pf_overlap: {w0}, {w1}, {S}, {inner}")
    check("numpy" not in sys.modules, "the self-test loaded numpy")
    for f_ in fails:
        print(f"FAIL {f_}")
    print(f"q38_step self-test: {'ok' if not fails else str(len(fails)) + ' failed'}")
    return 1 if fails else 0


# ============================================================================ main

class Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        print(f"q38_step.py: {message}", file=sys.stderr)
        sys.exit(64)


def main(argv=None):
    ap = Parser(description=__doc__.split("\n")[0])
    ap.add_argument("--machine", default=None, help=f"one of {', '.join(MACHINES)} (default {DEFAULT_MACHINE})")
    ap.add_argument("--config", default=None, help="the preset's configuration by name (default its first)")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--ceilings", action="store_true")
    g.add_argument("--backtest", action="store_true")
    g.add_argument("--tornado", action="store_true")
    g.add_argument("--explain", action="store_true")
    g.add_argument("--replay", metavar="DIR")
    g.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    try:
        if a.self_test:
            return self_test()
        if a.replay:
            return replay_main(a.replay)
        consts = load_constants()
        machine = a.machine or DEFAULT_MACHINE
        if machine not in MACHINES:
            raise Refused(f"no machine {machine!r}: the presets are {', '.join(MACHINES)}")
        cname, csets = config_sets(machine, a.config)
        if a.backtest:
            ok, _ = backtest(consts)
            return 0 if ok else 1
        if a.tornado:
            tornado(consts, machine, csets)
            return 0
        if a.explain:
            explain(consts, machine)
            return 0
        p, solved = calibrate(consts, machine, csets + a.set)
        p["_config"] = cname
        if a.ceilings:
            ceilings(p)
            return 0
        for line in report(p):
            print(line)
        return 0
    except Refused as e:
        print(f"q38_step.py: refused: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
