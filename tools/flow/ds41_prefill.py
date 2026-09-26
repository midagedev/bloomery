#!/usr/bin/env python3
"""The V4.1 prompt batch as one executable timeline model (A6000, plan (a)).

Resources, per layer-batch (one layer over one prompt batch of <= 512 positions):
  host thread  issue of the route and shadow calls, time blocked in the driver launch queue,
               the wait for the route's D2H, the copy of x (before hostserve), the union
  card stream  the route (engram, latent-only chunks, attention chunks, router tail), then the
               shadow under the union, the upload and join
  PCIe         the route's D2H of x and the upload of the host sums (in the card FIFO); the
               streamed experts on a copy engine
  host DRAM    the union's expert reads; the ring fill (3 crossings) while streaming

The card route is a sum of kernel terms measured under nsys in the afe86d5 sitting (rig-log
09-26#v41-prefill-nsys); the four attention projections follow their grid: launch + iterations x
sum over waves of max(the kernel's measured per-iteration latency L at m = 8, each unit's demand at
the wave's resident blocks). The launch queue holds Q activities (kernels and copies); event records
and stream waits cost the host a call and take no slot. The union is the kernel sum t(m) plus its
causes (join tails, wakes, stash, zero fill, scan, preparation, per flow) plus one named residual.

Today the host issues a layer-batch's route and shadow, waits for the route and then runs the union,
so the wall is a sum (cardroute-design-report.md:142-153; body/prefill.rs:1175-1265). The card runs the
route while the host issues it (an eager stream, prefill.rs:877-882), so the issue sits inside the
route window unless the launch queue is full; the batch's prologue ends in a synchronous H2D
(prefill.rs:1041-1043) and is serial. The G scheduler (layer-first over G
batches, hoststream-design-report.md:37-49) issues batch b+1's route before batch b's union, which is
a max once the host is not blocked in the queue. Host streaming adds a PCIe pipe and a ring, and its
k per layer is the resource-balance choice (hoststream-design-report.md:92).

Configuration knobs beyond the recorded commits: `b1` (the projections per 128-token sub-block,
chain/attn/batch.rs:778-800), `tile` (cardtile: the grouped shadow per (card expert, 8-column tile) item),
`imma` (the IMMA grouped GEMM in its place), `G` with `wrap` (the layer-first scheduler that issues
route(i + 1) before union(i), across the layer boundary too, cardnext-design-report.md 1.1), `timed`
(the card-timing marks a BLOOMERY_STEP_STATS=1 run records, as queue events) and `_clk` (the card's
SM-bound kernels at an SM clock ratio: the prose prompt ran its route clk_cardbound slower).

Every value is a row of constants.tsv or a count transcribed from the code (path:line).

    --backtest         the recorded sittings, predicted against measured, rc != 0 on a red row that names
                       no model term (BLAME, TERMS); a named red row prints RED and its term
    --predict STEP     the ladder (now, B1, B1+T, B1+T+G, +stream, +h3tile-b, +B4, or all), with bands
    --self-test        units and identities
    --explain ROW      every term of a backtest row or a ladder step, with its constants
    --uncalibrated     the uncalibrated terms and the runner command that calibrates each
    --cells            B1, B1+T, B1+G, B1+T+G per cell with bands, and the IMMA shadow over T at G 1 and G 2

Python 3 standard library only; the t table is tools/gpu-ab.py's.
"""

import argparse
import importlib.util
import math
import os
import sys
from bisect import bisect_right
from functools import lru_cache

HERE = os.path.dirname(os.path.abspath(__file__))
CONSTANTS = os.path.join(HERE, "constants.tsv")
PROSE_COUNTS = os.path.join(HERE, "prose-counts.tsv")
GPU_AB = os.path.join(HERE, "..", "gpu-ab.py")


# ============================================================================ constants

class Const:
    __slots__ = ("name", "value", "lo", "hi", "unit", "kind", "conditions", "source", "anchor", "note")

    def __init__(self, cells):
        (self.name, value, lo, hi, self.unit, self.kind, self.conditions, self.source,
         self.anchor, self.note) = cells
        self.value, self.lo, self.hi = float(value), float(lo), float(hi)


def load_constants(path=CONSTANTS):
    out, header = {}, None
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            cells = line.rstrip("\n").split("\t")
            if header is None:
                header = cells
                continue
            if len(cells) != len(header):
                raise SystemExit(f"constants.tsv: {cells[0]!r} has {len(cells)} cells, the header {len(header)}")
            c = Const(cells)
            if c.name in out:
                raise SystemExit(f"constants.tsv: {c.name} twice")
            if not c.source.strip():
                raise SystemExit(f"constants.tsv: {c.name} has no source")
            if c.kind not in ("measured", "derived", "assumed"):
                raise SystemExit(f"constants.tsv: {c.name} has kind {c.kind!r}")
            out[c.name] = c
    return out


C = load_constants()


def _load_t975():
    """tools/gpu-ab.py's two-sided 95 % t quantile, as tools/ref/card.py loads it."""
    try:
        spec = importlib.util.spec_from_file_location("gpu_ab", GPU_AB)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.t975
    except Exception as e:  # noqa: BLE001 - any load failure is the same named refusal
        raise SystemExit(f"ds41_prefill.py: the t table ({GPU_AB}, t975) does not load: {e}")


T975 = _load_t975()


class Params(dict):
    """The constant values of one evaluation; `log` collects every name the model reads."""

    def __init__(self, values, log=None):
        super().__init__(values)
        self.log = log

    def __getitem__(self, k):
        if self.log is not None:
            self.log.add(k)
        return dict.__getitem__(self, k)

    def but(self, **kw):
        q = Params(self, self.log)
        dict.update(q, kw)
        return q


def central(log=None):
    return Params({k: c.value for k, c in C.items()}, log)


# ============================================================================ geometry

L = 40              # every layer routed (docs/plan-ledger.md:701)
CHUNK = 8           # crates/gpu-deepseek41/src/body/prefill.rs:72 (CHUNK = HC_MAX_TOKENS)
T_MAX = 512         # body/prefill.rs:69 (T_MAX = UNION_MAX_COLS)
RING = 128          # the window ring's rows, body.rs:2165-2167; body/ced.rs:364-386 pins the triangle with 128
TOP_K = 512         # attention.indexer.top_k (crates/model/src/arch/deepseek41/hparams.rs:185-187)
N_EMBD = 5120
X_BYTES = N_EMBD * 4  # one column of x or of the host sums, f32
N_USED = 6          # crates/gpu-deepseek41/src/router.rs:63
N_EXPERT = 384      # router.rs:60
# The served layer table, hparams.rs:1265-1283: ratio 0 on 0-1, 2 on 2-19, 1 on 20-39; compressors
# and index keys on 2, 8, 14 (gated) and 20; indexer queries on 2, 8, 14, 20, 24, 28, 32, 36.
# Engram tables on blk.1 and blk.14 (docs/facts.md:11).
SRC_GATED = frozenset({2, 8, 14})
SRC_PLAIN = frozenset({20})
SOURCES = SRC_GATED | SRC_PLAIN
IDX_ONLY = frozenset({24, 28, 32, 36})
INDEXERS = SOURCES | IDX_ONLY
ENGRAM = frozenset({1, 14})
EVERY = SOURCES     # body/ced.rs:302-316: every = compressor || index_keys


def ratio(l):
    return 0 if l < 2 else (2 if l < 20 else 1)


def batches(P):
    """body/prefill.rs:137-146: ceil(P / T_MAX) batches, the first P mod k of them one longer."""
    k = -(-P // T_MAX)
    out, p = [], 0
    for j in range(k):
        n = P // k + (1 if j < P % k else 0)
        out.append((p, p + n))
        p += n
    return out


def chunk_cuts(b, u):
    """body/prefill.rs:466-476: chunks cut at every multiple of CHUNK."""
    out, p, end = [], b, b + u
    while p < end:
        nxt = min((p // CHUNK + 1) * CHUNK, end)
        out.append((p, nxt))
        p = nxt
    return out


SUB_CHUNKS = 16     # chain/attn/batch.rs:51 (SUB_CHUNKS): B1's projections run per sub-block of <= 16 chunks


def sub_blocks(chunks):
    """chain/attn/batch.rs:78-111 (SubBlocks): a short chunk alone, each run of whole chunks as runs of
    16, then 8, 4, 2 and 1 chunks. `chunks` is [(position, m)]; returns lists of them."""
    out, k, n = [], 0, len(chunks)
    while k < n:
        if chunks[k][1] == CHUNK:
            run = 0
            while k + run < n and chunks[k + run][1] == CHUNK and run < SUB_CHUNKS:
                run += 1
            s = 1 << (run.bit_length() - 1)
        else:
            s = 1
        out.append(chunks[k:k + s])
        k += s
    return out


def ced_need(first, end, starts, on):
    """body/ced.rs:256-296: (part, full) per layer, walked from the last layer down."""
    if not on:
        return [(first, first)] * L

    def floor(x):
        x = max(x, first)
        batch = max((s for s in starts if s <= x), default=first)
        return max(x - x % CHUNK, batch, first)

    out = [None] * L
    read_from = end - 1
    for i in reversed(range(L)):
        full = floor(read_from)
        part = first if i in EVERY else floor(min(max(full + 1 - RING, 0), max(end - RING, 0)))
        out[i] = (part, full)
        read_from = part
    return out


class LB:
    """One layer of one batch: its chunks by mode (body/prefill.rs:891-1041)."""
    __slots__ = ("b", "l", "full", "part", "engram", "T", "u", "first_layer")

    def __init__(self, b, l, full, part, engram, u, first_layer):
        self.b, self.l, self.full, self.part, self.engram = b, l, full, part, engram
        self.T = sum(m for _, m in full)
        self.u, self.first_layer = u, first_layer


@lru_cache(maxsize=None)
def prompt_plan(P, ced_on):
    """((start, u, (LB per layer)) per batch) for a prompt of P ids from position 0."""
    runs = batches(P)
    need = ced_need(0, P, [s for s, _ in runs], ced_on)
    plan = []
    for j, (s, e) in enumerate(runs):
        cuts = chunk_cuts(s, e - s)
        lbs = []
        for l in range(L):
            part_from, full_from = need[l]
            full = [(a, z - a) for a, z in cuts if a >= full_from]
            part = [(a, z - a) for a, z in cuts if part_from <= a < full_from]
            engram = sorted(part + full) if l in ENGRAM else []
            lbs.append(LB(j, l, full, part, engram, e - s, l == 0))
        plan.append((s, e - s, tuple(lbs)))
    return tuple(plan)


def block_positions(P, ced_on):
    return sum(lb.T for _, _, lbs in prompt_plan(P, ced_on) for lb in lbs)


def served_lbs(P, ced_on):
    return sum(1 for _, _, lbs in prompt_plan(P, ced_on) for lb in lbs if lb.T)


# ============================================================================ queue entries
#
# An activity (a kernel or a copy) takes a slot of the driver's launch queue and card time; an event
# record or a stream wait costs the host a call and takes no slot (the nsys sitting's plateau is
# 1,068 activities, 1,326 calls up to the event). The counts below match the trace's per-layer
# N_r/N_s/ev columns (a normal layer 1,350 / 517 / 257).

def attn_kernels(l):
    """Kernels of one staged attention chunk (chain/attn.rs:1430-1685): hc_pre, norm_quant, joint qkv
    gemv + 2 transposes, q_a norm, q_b gemv + transpose, rope tail, kv append, seg, merge, commit,
    rope back, quantize, wo_a, quantize, wo_b + transpose, hc_post = 20; the indexer adds 2 and a
    source's compressor and index key 8 (7 ungated) (docs/research/cardroute-design-report.md:66-77)."""
    return 30 if l in SRC_GATED else 29 if l in SRC_PLAIN else 22 if l in IDX_ONLY else 20


FORK_EVENTS = 4                # gpu/src/graph.rs:251-253, :283-286: record + wait, twice (chain/attn.rs:1493, :1671)
ROUTE_NQ = 1                   # chain/ffn/batch.rs:1350-1368: the route's norm_quant, per chunk
TAIL_BULK_ACTS = 6             # batch.rs:1370-1399 scores + pick + places; batch.rs:1105-1122 three D2H
TAIL_EVENTS = 1                # batch.rs:1105-1122: the event the host waits on
TAIL_PREBULK_ACTS = 3          # three D2H; router and places ran per chunk (d37136c batch.rs:815-866)
POST_ACTS = 2                  # batch.rs:1155-1164 upload, batch.rs:1769-1838 join
HEAD_ACTS = 7                  # the trace: 7 activities after the last join
SHADOW_EXPERT_CHUNK = 8        # batch.rs:1593-1766: hc_pre, norm, 2 dtod, shexp gate-up, q8, down, transpose
SHADOW_EXPERT_BLOCK = 5        # batch.rs:1449-1593: buckets, grouped gate-up, quantize_sel, grouped down, card_acc
SHADOW_NOCARD_CHUNK = 5        # the trace, layers 0-1 (no card expert): N_s 321 = 64 x 5 + 1
SHADOW_NOCARD_BLOCK = 1
SHADOW_SLOT_CHUNK = 10         # cardroute-design-report.md:73 (per slot: gu, quantize_sel, q4k_sel, acc + shexp)


def part_acts(l):
    """chain/attn.rs:1336-1424, a latent-only chunk: norm, qkv gemv, transpose, kv append, and a
    source layer's compressor and index key."""
    return 4 + (8 if l in SRC_GATED else 7 if l in SRC_PLAIN else 0)


def engram_acts(m):
    """chain/glue/batch.rs:190-297: an engram_rows launch per token, quantize, wkv gemv, transpose,
    key norm, gate, fold."""
    return m + 6


def full_acts(l):
    return attn_kernels(l) + ROUTE_NQ


def src_sub_acts(l):
    """A source layer's projection launches in a B1 sub-block (chain/attn/batch.rs:871-909): the gated
    compressors' joint projection and its two token-major copies (3), the ungated one's kv projection
    and its copy (2). The rest of the source stays in the chunk loop (src_chunk_acts)."""
    return 3 if l in SRC_GATED else 2 if l in SRC_PLAIN else 0


def src_chunk_acts(l):
    """A source layer's chunk launches (chain/attn/batch.rs:1015-1034, attn.rs:2436-2499): the pool (or
    the ratio-1 rows), then the index key's quantize, gemv, token-major copy and key (1 + 4)."""
    return 5 if l in SOURCES else 0


def b1_chunk_acts(l):
    """B1 (chain/attn/batch.rs:979-1213): only the ring-ordered launches stay in the attention chunk loop
    (kv append, seg, merge, commit, the source's and the indexer's); the route's per-chunk norm_quant is
    the FFN's (batch.rs:1350-1368) and stays."""
    return 4 + ROUTE_NQ + src_chunk_acts(l) + (2 if l in INDEXERS else 0)


B1_SB_ACTS = 14                # chain/attn/batch.rs:833-954, :1219-1262: a block sub-block's sub_pre (8) + sub_post (6)
B1_SB_SMALL = 10               # of them the small ones (norms, quantizes, token-major copies, ropes)
B1_PART_SB_ACTS = 3            # attn/batch.rs:778-787: a latent-only sub-block's norm_quant, qkv, latent copy
B1_FORK_ACTS = 2               # attn/batch.rs:753-822: HC_PRE on the fork, HC_POST after the join
TIMED_SB_EVENTS = 4            # attn/batch.rs:789-799: with card timing, a mark pair around sub_pre and sub_post
TIMED_PART_SB_EVENTS = 2       # attn/batch.rs:779-781: around a latent-only sub_pre
TIMED_ROUTE_EVENTS = 2         # body/prefill.rs:1110, :1211/:1228: the layer's start and the route's end
TIMED_SHADOW_EVENTS = 1        # body/prefill.rs:1246: the shadow's end


def batch_acts(s, u):
    """body/prefill.rs:1088-1101 (queue::GATHER, queue::embed): per batch before layer 0, a words gather a
    chunk and an embedding broadcast a token."""
    return len(chunk_cuts(s, u)) + u


# ============================================================================ routing

class Routing:
    """Per layer: host experts and card experts as (columns per 512 tokens, count), hottest first."""

    def __init__(self, name, host, card):
        self.name, self.host, self.card = name, tuple(host), tuple(card)

    def host_slots_tok(self, l):
        return sum(lam * n for lam, n in self.host[l]) / 512.0

    def card_slots_tok(self, l):
        return sum(lam * n for lam, n in self.card[l]) / 512.0

    def n_host(self, l):
        return sum(n for _, n in self.host[l])

    def n_card(self, l):
        return sum(n for _, n in self.card[l])

    def borrow(self, per_layer):
        """The coldest `per_layer` card experts of each layer join the host set for the prompt."""
        host, card = [], []
        for l in range(L):
            left, kept, moved = per_layer, [], []
            for lam, n in reversed(self.card[l]):
                take = min(n, left)
                left -= take
                if take > 0:
                    moved.append((lam, take))
                if n - take > 0:
                    kept.append((lam, n - take))
            card.append(tuple(sorted(kept, key=lambda g: -g[0])))
            host.append(tuple(sorted(tuple(self.host[l]) + tuple(moved), key=lambda g: -g[0])))
        return Routing(self.name + "+borrow", host, card)


def routing_lcg(p, hot=True):
    """The timing runner's lcg prompt: per layer one host rate and one card rate (lcg is uniform to
    first order, hoststream-recal-report.md:160). Layers below card_first_layer hold no card expert
    (plan (a)), so all six slots of a token go to the host there; the recorded mean host rate
    (s_host_lcg with the hot list, s_host_lcg_nohot without) sets the other layers'."""
    s = p["s_host_lcg"] if hot else p["s_host_lcg_nohot"]
    n0 = int(p["card_first_layer"])
    nc = p["n_card_total"] / (L - n0)
    sl = (L * s - N_USED * n0) / (L - n0)
    host, card = [], []
    for l in range(L):
        if l < n0:
            host.append(((N_USED * 512.0 / N_EXPERT, float(N_EXPERT)),))
            card.append(())
        else:
            nh = N_EXPERT - nc
            host.append(((sl * 512.0 / nh, nh),))
            card.append((((N_USED - sl) * 512.0 / nc, nc),))
    return Routing("lcg" if hot else "lcg-nohot", host, card)


@lru_cache(maxsize=None)
def _prose_counts():
    rows = [[0] * N_EXPERT for _ in range(L)]
    with open(PROSE_COUNTS, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            l, e, n = map(int, line.split())
            rows[l][e] = n
    return tuple(tuple(r) for r in rows)


def routing_prose_trace(p):
    """prose-in: each layer's card set is the trace's own top n_l (hoststream-recal-report.md:30-33),
    the in-sample (optimistic) arm; zero-count host experts are never touched. Every layer keeps n_l
    card experts (the trace's curve), unlike plan (a)'s host-only layers 0-1. The trace's own
    identities (the self-test) and a comparison arm; the prose prompt is routing_prose."""
    n_l, ntok = int(p["prose_n_l"]), p["prose_ntok"]
    host, card = [], []
    for l in range(L):
        cols = sorted((c * 512.0 / ntok for c in _prose_counts()[l]), reverse=True)
        card.append(tuple((x, 1.0) for x in cols[:n_l]))
        host.append(tuple((x, 1.0) for x in cols[n_l:] if x > 0))
    return Routing("prose-in", host, card)


def routing_prose(p):
    """prose: the prose prompt the lease feeds (corpus-prose.ids, the first 512) under plan (a) and hot
    list 384, from the trace's per-layer rank curve. Three facts the trace alone does not give:
    layers below card_first_layer hold no card expert (plan (a)); the hot list is learned from other
    sets too, so the card holds the trace's ranks [s, s + n_l) instead of [0, n_l) — the prompt's s
    hottest experts stay on the host (prose_swap, fixed by the prompt's host-slot count); and a
    512-token window routes burstier than the 50,000-token average, so its host columns land on a
    fraction phi of the host experts at rates / phi (prose_phi, fixed by the prompt's union). The card
    list keeps the trace's rates (its per-slot cost does not read the spread; T's tiles would read
    fewer, so T is priced conservatively)."""
    n_l, ntok = int(p["prose_n_l"]), p["prose_ntok"]
    n0, sw, phi = int(p["card_first_layer"]), p["prose_swap"], p["prose_phi"]
    i, f = int(sw), sw - int(sw)
    host, card = [], []
    for l in range(L):
        cols = sorted((c * 512.0 / ntok for c in _prose_counts()[l]), reverse=True)
        w = [0.0] * len(cols)
        if l >= n0:
            for r in range(i + 1, min(i + n_l, len(cols))):
                w[r] = 1.0
            w[i] += 1.0 - f
            if i + n_l < len(cols):
                w[i + n_l] += f
        card.append(tuple((cols[r], w[r]) for r in range(len(cols)) if w[r] > 1e-12))
        host.append(tuple((cols[r] / phi, (1.0 - w[r]) * phi) for r in range(len(cols))
                          if 1.0 - w[r] > 1e-12 and cols[r] > 0))
    return Routing("prose", host, card)


def routing(p, name, hot=True):
    if name == "lcg":
        return routing_lcg(p, hot)
    if name == "prose":
        return routing_prose(p)
    if name == "prose-in":
        return routing_prose_trace(p)
    raise SystemExit(f"ds41_prefill.py: no routing {name!r} (lcg, prose, prose-in)")


# ============================================================================ the host union

@lru_cache(maxsize=None)
def ecost(T, lam, a, c, w):
    """(E[max(W, a + c m); m >= 1] in us, P(m >= 1)) for m ~ Binomial(T, lam / 512): the union's
    t(m) with Binomial column counts (docs/plan.md:76)."""
    if T <= 0 or lam <= 0:
        return 0.0, 0.0
    q = min(lam / 512.0, 1.0)
    if q >= 1.0:
        return max(w, a + c * T), 1.0
    mu = T * q
    mmax = min(T, int(mu + 12 * math.sqrt(mu) + 30))
    lq, l1q, lg = math.log(q), math.log1p(-q), math.lgamma
    s = 0.0
    for m in range(1, mmax + 1):
        s += math.exp(lg(T + 1) - lg(m + 1) - lg(T - m + 1) + m * lq + (T - m) * l1q) * max(w, a + c * m)
    return s, -math.expm1(T * l1q)


@lru_cache(maxsize=None)
def union_table(hostlist, T, a, c, w):
    """Prefix sums over the host list, hottest first: counts, ms, touched experts, slots."""
    cn, cms, cact, cs, per = [0.0], [0.0], [0.0], [0.0], []
    for lam, n in hostlist:
        e, act = ecost(T, round(lam, 6), a, c, w)
        s = T * lam / 512.0
        per.append((e / 1000.0, act, s))
        cn.append(cn[-1] + n)
        cms.append(cms[-1] + n * e / 1000.0)
        cact.append(cact[-1] + n * act)
        cs.append(cs[-1] + n * s)
    return cn, cms, cact, cs, per


def union_sum(tab, skip=0.0):
    """(ms, slots, touched) of the host experts after the hottest `skip` (streamed) ones."""
    cn, cms, cact, cs, per = tab
    if skip <= 0:
        return cms[-1], cs[-1], cact[-1]
    if skip >= cn[-1]:
        return 0.0, 0.0, 0.0
    i = bisect_right(cn, skip) - 1
    f = skip - cn[i]
    e, act, s = per[i]
    return cms[-1] - cms[i] - f * e, cs[-1] - cs[i] - f * s, cact[-1] - cact[i] - f * act


def union_kernel(p, cfg):
    k = cfg.get("union", "union")
    if k == "hosttile":
        return p["a_hosttile"], p["c_hosttile"]
    if k == "r8":
        return p["a_r8"], p["c_r8"]
    return p["a_union"], p["c_union"]


GU_SLOT_BYTES = 2 * 2304 * 4 + 2664   # the combine re-reads a slot's gate/up (f32) and writes qc (moe.rs D3)


def cause_terms(p, flow, K, slots, touched, T, f, a, c):
    """The union's time beyond its kernel sum K, by cause (ms), for one call of T columns.

    chunks (before 9626c7f, moe.rs serve_planned on 9626c7f^): x quantized once, then per chunk of 8
    host experts a gate/up row dispatch, a down pre-pass and a down row dispatch, a stash of the
    downs (not after the last chunk), and one sum: 2 + 3 n_c + (n_c - 1) dispatches (161 at 40
    chunks). Each row dispatch ends in a join tail, f of the last participant's steal block, and a
    block is lane / steal_blocks, so the tails are f x K / steal_blocks. Serial on the calling
    thread: the set_cols zero fill per chunk, the non-finite scan of x, the chunk preparation.
    five (9626c7f, moe.rs:1347-1490): quantize x, gate/up, combine, down, sum (two for a call of <= 8
    columns); the row dispatches' blocks are capped at union_block_rows rows, so the two tails are f
    of a 144-row block each; the combine re-reads the gate/up slab; the scan is a pool pass."""
    wide = T > 8
    if flow == "five":
        cost = a + c * (slots / touched)                       # one expert at its mean m, 32 threads
        share = p["gu_row_share"]
        trow = share * 32 * cost / 4608 + (1 - share) * 32 * cost / 5120
        return dict(tail=f * p["union_block_rows"] * trow / 1000.0,
                    wake=(5 if wide else 2) * p["wake_us"] / 1000.0,
                    t6=slots * GU_SLOT_BYTES / (p["t6_gbs"] * 1e6) if wide else 0.0,
                    scan=p["scan_pool_ms"] * T / 512.0)
    n_c = max(1, math.ceil(touched / 8.0 - 1e-9))
    n_disp = (2 + 3 * n_c if wide else 2 * n_c) + (n_c - 1)
    return dict(tail=f * K / p["steal_blocks"],
                wake=n_disp * p["wake_us"] / 1000.0,
                smalltail=n_c * p["smalltail_us"] / 1000.0,
                stash=slots * X_BYTES * (n_c - 1) / n_c / (p["stash_gbs"] * 1e6),
                zfill=n_c * p["zfill_chunk_bytes"] / (p["zfill_gbs"] * 1e6),
                scan=T * X_BYTES / (p["scan_gbs"] * 1e6),
                prep=n_c * p["prep_us"] / 1000.0)


def _layer_calls(rt, P, ced, a, c, w):
    for _, _, lbs in prompt_plan(P, ced):
        for lb in lbs:
            if lb.T:
                K, s, t = union_sum(union_table(rt.host[lb.l], lb.T, a, c, w))
                if t > 1e-12:
                    yield lb, K, s, t


def union_anchor(p, anchors):
    """f and X_u for these constants.

    f, the join-tail share of a steal block: the uniondispatch A/B at P 512 (no hot list, CED on, one
    lease) measured the chunk flow minus the five dispatches; X_u is the same per slot in both arms,
    so that difference is f x (tail coefficients) + the other cause terms, linear in f.
    X_u, the union's named residual: its anchor row minus the kernel sum and the chunk flow's causes,
    per host slot ("slot") or per kernel ms ("prop")."""
    a, c, w = p["a_union"], p["c_union"], p["w"]
    coef = rest = 0.0
    n = 0
    for lb, K, s, t in _layer_calls(routing_lcg(p, False), 512, True, a, c, w):
        o1, o0 = cause_terms(p, "chunks", K, s, t, lb.T, 1.0, a, c), cause_terms(p, "chunks", K, s, t, lb.T, 0.0, a, c)
        n1, n0 = cause_terms(p, "five", K, s, t, lb.T, 1.0, a, c), cause_terms(p, "five", K, s, t, lb.T, 0.0, a, c)
        coef += (sum(o1.values()) - sum(o0.values())) - (sum(n1.values()) - sum(n0.values()))
        rest += sum(o0.values()) - sum(n0.values())
        n += 1
    f = (p["ud_delta_512"] - rest / n) / (coef / n)
    name = anchors["union"]
    # (P, CED, hot list, flow) of the anchor row: S13b CED off and S14's slot arm ran the chunk flow with the hot
    # list; the uniondispatch lease's new arm ran today's five dispatches without it
    P, ced, hot, flow = {"anchor_union_s13b": (512, False, True, "chunks"), "anchor_union_s14s": (512, True, True, "chunks"),
                         "anchor_union_ud512": (512, True, False, "five")}[name]
    Ks = sl = ex = 0.0
    m = 0
    for lb, K, s, t in _layer_calls(routing_lcg(p, hot), P, ced, a, c, w):
        Ks += K
        sl += s
        ex += sum(cause_terms(p, flow, K, s, t, lb.T, f, a, c).values())
        m += 1
    left = p[name] * m - Ks - ex
    mode = anchors.get("resid_mode", "slot")
    return dict(f=f, mode=mode, x=left / sl if mode == "slot" else left / Ks, anchor=name)


def union_call(p, cfg, rt, l, T, U, skip=0.0):
    """One union call (ms) over T columns of layer l, the hottest `skip` host experts excluded."""
    zero = dict(ms=0.0, raw=0.0, resid=0.0, expl=0.0, x=0.0, slots=0.0, touched=0.0, bytes=0.0, parts={})
    if T <= 0:
        return zero
    a, c = union_kernel(p, cfg)
    K, slots, touched = union_sum(union_table(rt.host[l], T, a, c, p["w"]), skip)
    if touched <= 1e-12:
        return zero
    if U.get("off"):
        parts, x = {}, 0.0
    else:
        parts = cause_terms(p, cfg.get("flow", "chunks"), K, slots, touched, T, U["f"], a, c)
        x = U["x"] * (slots if U["mode"] == "slot" else K)
    expl = sum(parts.values())
    res = expl + x + (p["unionreal_cut"] if cfg.get("unionreal_cut") else 0.0)
    return dict(ms=K + res, raw=K, resid=res, expl=expl, x=x, slots=slots, touched=touched,
                bytes=touched * p["expert_bytes_host"], parts=parts)


# ============================================================================ card kernels: the projections

# name: rows, K, blocks/SM constant, the measured launch (layer 3 chunk 32, m = 8), ncu tag, long K
KERN = {
    "qkv": (1792, 5120, "q3k_bps", "proj_qkv_us", "qkv", True),
    "q_b": (32768, 1280, "q3k_bps", "proj_qb_us", "qb", False),
    "wo_a": (8192, 4096, "heads_bps", "proj_woa_us", "woa", False),
    "wo_b": (5120, 8192, "q3k_bps", "proj_wob_us", "wob", True),
}
QB_ROWS, QB_ROWS_IDX = 32768, 36864   # 64 heads x 512; joined with the indexer's query (cardroute:84)
WEIGHT_BYTES_BLOCK_STEP = 8 * 2 * 110  # 8 rows x two Q3_K super-blocks a warp-iteration


def qb_rows(l):
    return QB_ROWS_IDX if l in INDEXERS else QB_ROWS


def kgeom(p, name, rows, groups=1):
    """(resident blocks an SM, iterations, blocks, full waves, blocks of the partial wave)."""
    _, K, bps, _, _, _ = KERN[name]
    r = int(p[bps])
    iters = -(-(K // 256) // 2)
    blocks = -(-rows // 8) * groups
    full, tail = divmod(blocks, int(p["n_sm"]) * r)
    return r, iters, blocks, full, tail


def lat_meas(p, name):
    """The kernel's per-iteration latency at m = 8 (us): its measured launch less the launch
    intercept, over iterations x waves of its own grid (a partial wave counts whole)."""
    r, iters, blocks, full, tail = kgeom(p, name, KERN[name][0])
    return (p[KERN[name][3]] - p["gemv_launch"]) / (iters * (full + (1 if tail else 0)))


def demand_us(p, name, r, groups=1):
    """Each unit's demand of one wave-iteration at r resident blocks an SM (us): issue, LSU, L1TEX
    from the ncu counts, DRAM for the weights (shared by the column groups of a grouped launch)."""
    tag = KERN[name][4]
    f = p["f_sm"] * 1e3
    d = {u: r * p[f"dmd_{tag}_{u}"] / f for u in ("iss", "lsu", "l1")}
    d["dram"] = r * WEIGHT_BYTES_BLOCK_STEP / groups / (p["bw_card"] * 1e3 / p["n_sm"])
    return d


def lsu_eff(p):
    """q_b's measured step against its LSU demand at 6 blocks: the busiest unit's efficiency at full
    residency."""
    L_qb = lat_meas(p, "q_b")
    return demand_us(p, "q_b", int(p["q3k_bps"]))["lsu"] / L_qb if L_qb > 0 else 1.0


def launch_us(p, name, rows, m=8, groups=1, grouped=False, detail=None):
    """One launch of the m-column core over `groups` column groups (us): launch + iterations x
    (full waves x step at full residency + the partial wave's step). A step is max(L, the busiest
    unit's demand at the wave's residency); L is the kernel's measured m = 8 latency, moved by
    full_res_lat for a full-residency wave (+1: the long-K q3k launches at wo_b's wave-1 reading,
    their partial waves at qkv's; -1: a grouped launch at its busiest unit / lsu_eff)."""
    r, iters, blocks, full, tail = kgeom(p, name, rows, groups)
    L_k = lat_meas(p, name)
    lf = lt = L_k
    mode = p["full_res_lat"]
    if KERN[name][5] and mode > 0:
        L_qkv, L_wob = lat_meas(p, "qkv"), lat_meas(p, "wo_b")
        lf = L_k + mode * ((2 * L_wob - L_qkv) - L_k)
        lt = L_k + mode * (L_qkv - L_k)
    dem_f = demand_us(p, name, r, groups)
    if grouped and mode < 0:
        lf = L_k + (-mode) * (max(dem_f.values()) / lsu_eff(p) - L_k)
    if m < 8:
        s = (m - 1) / 7.0
        lf = p["gemv_lat_iter"] + (lf - p["gemv_lat_iter"]) * s
        lt = p["gemv_lat_iter"] + (lt - p["gemv_lat_iter"]) * s
    step_f = max(lf, max(dem_f.values()))
    rt = -(-tail // int(p["n_sm"])) if tail else 0
    dem_t = demand_us(p, name, rt, groups)
    step_t = max(lt, max(dem_t.values())) if tail else 0.0
    t = p["gemv_launch"] + iters * (full * step_f + (1 if tail else 0) * step_t)
    if detail is not None:
        detail.update(r=r, iters=iters, blocks=blocks, waves=blocks / (int(p["n_sm"]) * r), full=full, tail=tail,
                      L=L_k, lf=lf, step_f=step_f, step_t=step_t, dem=dem_f, t=t)
    return t


def proj_chunk(p, l, m):
    return (launch_us(p, "qkv", 1792, m) + launch_us(p, "q_b", qb_rows(l), m) + launch_us(p, "wo_a", 8192, m)
            + launch_us(p, "wo_b", 5120, m))


def proj_sb(p, l, sb):
    """B1's four projections over one sub-block (us): a launch each over its chunks as column groups in
    one grid (q3k_gemv_groups: a row stretch's groups are consecutive blocks); a short chunk alone is one
    group of m columns."""
    g, m = (len(sb), 8) if sb[0][1] == CHUNK else (1, sb[0][1])
    return (launch_us(p, "qkv", 1792, m, g, True) + launch_us(p, "q_b", qb_rows(l), m, g, True)
            + launch_us(p, "wo_a", 8192, m, g, True) + launch_us(p, "wo_b", 5120, m, g, True))


def proj_b1(p, l, chunks):
    """B1 (2f45a79): the projections per sub-block of the block's chunks (chain/attn/batch.rs:788-800)."""
    return sum(proj_sb(p, l, sb) for sb in sub_blocks(chunks))


def full_chunks(T):
    """A block of T positions from a multiple of CHUNK, as [(position, m)]."""
    return [(a, z - a) for a, z in chunk_cuts(0, T)]


def proj_macs(l):
    return 1792 * 5120 + qb_rows(l) * 1280 + 8192 * 4096 + 5120 * 8192


# ============================================================================ card kernels: attention and the rest

def keys_at(l, pos):
    """Keys a query at `pos` reads (the window ring and the selected compressed rows), and the
    compressed rows visible to it (what the indexer scans)."""
    r = ratio(l)
    vis = (pos + 1) // r if r else 0
    return min(pos + 1, RING) + min(vis, TOP_K), vis


K6 = sum(keys_at(x, 5)[0] for x in range(2, L)) / (L - 2)
K1024 = sum(keys_at(x, 1023)[0] for x in range(2, L)) / (L - 2)


@lru_cache(maxsize=None)
def chunk_keys(l, pos0, m):
    ks = [keys_at(l, pos0 + t) for t in range(m)]
    return sum(k for k, _ in ks) / m, sum(v for _, v in ks)


# layer 2's chunks at P 512, where attn_seg_l2 and idx_chunk were measured
KREF_L2 = sum(chunk_keys(2, s, 8)[0] for s in range(0, 512, 8)) / 64
VISREF_L2 = sum(chunk_keys(2, s, 8)[1] for s in range(0, 512, 8)) / 64


def attn_seg(p, l, pos, m):
    """seg over a chunk (us): 40 blocks a token at 1 block/SM (docs/plan-ledger.md:1176, A2); a block's
    time is layer 2's prompt-batch measurement moved along the decode slope in keys."""
    slope = (p["attn_seg_d1024"] - p["attn_seg_d6"]) / (K1024 - K6)
    waves = -(-40 * m // int(p["n_sm"]))
    blk = (p["attn_seg_l2"] - p["gemv_launch"]) / 4.0
    keys, _ = chunk_keys(l, pos, m)
    return p["gemv_launch"] + waves * (blk + slope * (keys - KREF_L2))


def src_us(p, l):
    if l in SRC_GATED:
        return p["src_chunk"]
    if l in SRC_PLAIN:
        return p["src_chunk"] - p["src_gemv_pair"] / 2.0
    return 0.0


def special_chunk(p, l, pos, m):
    t = src_us(p, l)
    if l in INDEXERS:
        t += p["idx_chunk"] + (chunk_keys(l, pos, m)[1] - VISREF_L2) * p["idx_row_ns"] / 1000.0
    return t


def route_lb(p, cfg, lb):
    """Card time (ms), activities, events and terms of a layer-batch's route: everything the host
    issues before it waits for the route's D2H (body/prefill.rs:1106-1230). With B1 the projections
    and their small launches run per sub-block (chain/attn/batch.rs:778-800), the latent-only part's
    too; `timed` adds the card-timing marks (events) a BLOOMERY_STEP_STATS=1 run records; `_clk`
    stretches the card's kernels (not the D2H) by an SM clock ratio."""
    b1, b4 = cfg.get("b1") or cfg.get("b4"), cfg.get("b4")
    bulk = cfg.get("route", "bulk") == "bulk"
    timed = cfg.get("timed", False)
    l, g, sk = lb.l, p["gap_act"] / 1000.0, p["small_k"]
    terms, acts, ev = {}, 0, (TIMED_ROUTE_EVENTS if timed else 0)
    eg = 0.0
    for _, m in lb.engram:
        a = engram_acts(m)
        eg += p["engram_chunk"] * m / 8.0 / 1000.0 + a * g
        acts += a
    terms["engram"] = eg
    pt = pp_ = 0.0
    if b1 and lb.part:
        for sb in sub_blocks(lb.part):
            a = B1_PART_SB_ACTS + src_sub_acts(l)
            gs, ms = (len(sb), 8) if sb[0][1] == CHUNK else (1, sb[0][1])
            pp_ += (launch_us(p, "qkv", 1792, ms, gs, True) + 2 * sk) / 1000.0 + a * g
            acts += a
            ev += TIMED_PART_SB_EVENTS if timed else 0
            for _ in sb:
                a = 1 + src_chunk_acts(l)
                pt += (p["kv_append"] + src_us(p, l)) / 1000.0 + a * g
                acts += a
    else:
        for _, m in lb.part:
            a = part_acts(l)
            pt += (p["part_chunk"] + src_us(p, l)) / 1000.0 + a * g
            acts += a
    terms["part"] = pt
    terms["part_proj"] = pp_
    proj = attn = small = ovl = spec = nq = gaps = 0.0
    nfull = len(lb.full)
    for pos, m in lb.full:
        attn += (attn_seg(p, l, pos, m) + p["attn_other"]) / 1000.0
        spec += special_chunk(p, l, pos, m) / 1000.0
        nq += p["route_nq"] / 1000.0
        if b1:
            a, e = b1_chunk_acts(l), 0
        else:
            proj += proj_chunk(p, l, m) / 1000.0
            small += p["small_chunk"] / 1000.0
            ovl -= p["fork_overlap"] / 1000.0
            a, e = full_acts(l), FORK_EVENTS
        if not bulk:
            a += m + 1                     # a one-token router launch per token and a places launch
        acts += a
        ev += e
        gaps += a * g
    if b1 and nfull:
        T = lb.T
        sbs = sub_blocks(lb.full)
        if b4:
            proj = 2.0 * proj_macs(l) * T / (p["gemm_tops_proj"] * 1e12) * 1e3
        else:
            proj = proj_b1(p, l, lb.full) / 1000.0
        # HC_PRE stays on the fork branch beside the sub-blocks' launches (hidden, as in the loop)
        small = max((B1_SB_SMALL * len(sbs) + B1_FORK_ACTS) * sk / 1000.0,
                    p["small_bytes_tok"] * T / (p["bw_card"] * 1e9) * 1e3)
        a = (B1_SB_ACTS + src_sub_acts(l)) * len(sbs) + B1_FORK_ACTS
        acts += a
        ev += FORK_EVENTS + (TIMED_SB_EVENTS * len(sbs) if timed else 0)
        gaps += a * g
    terms.update(proj=proj, attn=attn, small=small, overlap=ovl, special=spec, route_nq=nq, gaps=gaps)
    tail = d2h = 0.0
    if nfull:
        T = lb.T
        d2h = T * X_BYTES / (p["d2h_gbs"] * 1e6)
        if bulk:
            tail = ((-(-T // 8)) * p["router_tile"] + p["pick"] + p["places"]) / 1000.0 + TAIL_BULK_ACTS * g
            acts += TAIL_BULK_ACTS
        else:
            tail = (T * p["router_tok"] + nfull * p["places"]) / 1000.0 + TAIL_PREBULK_ACTS * g
            acts += TAIL_PREBULK_ACTS
        ev += TAIL_EVENTS
    terms["tail"] = tail
    clk = cfg.get("_clk", 1.0)
    if clk != 1.0:
        terms = {k: v * clk for k, v in terms.items()}
    terms["d2h"] = d2h
    return sum(terms.values()), acts, ev, terms


def shadow_lb(p, cfg, lb, rt):
    """Card time (ms), activities and events of a layer-batch's shadow (batch.rs:1402-1766), under the
    union. Expert arm: the per-chunk kernels as traced and the grouped gate-up and down per card slot;
    slot arm: the same per-chunk kernels and each slot's expert bytes (derived, not traced). `tile` (T,
    cardtile) prices the grouped pair per (card expert, 8-column tile) item at t_tile_T; `imma` prices
    it as the IMMA grouped GEMM (gemm_q3k/gemm_q4k): imma_lb a layer-batch of 512 columns, in
    proportion below, at least the card experts' bytes once. `_clk` stretches the SM-bound parts (the per-chunk kernels, T's issue-bound
    items), not the per-slot grouped pair, which reads its weights from DRAM once a slot."""
    if not lb.full:
        return 0.0, 0, 0
    g, sk, bw = p["gap_act"] / 1000.0, p["small_k"], p["bw_card"] * 1e9
    clk = cfg.get("_clk", 1.0)
    arm = cfg.get("arm", "expert") if cfg.get("route", "bulk") == "bulk" else "slot"
    l = lb.l
    card = rt.n_card(l) > 0
    cs_tok = rt.card_slots_tok(l)
    t, acts = 0.0, 0
    for _, m in lb.full:
        if arm == "expert":
            k, a = (p["shadow_chunk"], SHADOW_EXPERT_CHUNK) if card else (p["shadow_chunk_nocard"], SHADOW_NOCARD_CHUNK)
            k *= clk
        else:
            k = p["shadow_chunk"] + m * cs_tok * (p["expert_gu_bytes"] + p["expert_down_bytes"]) / bw * 1e6 + 2 * sk
            a = SHADOW_SLOT_CHUNK
        t += k / 1000.0 + a * g
        acts += a
    if arm == "expert":
        if card:
            if cfg.get("imma"):
                floor = rt.n_card(l) * (p["expert_gu_bytes"] + p["expert_down_bytes"]) / bw * 1e6
                grouped = max(floor, p["imma_lb"] * 1000.0 * lb.T / T_MAX)
            elif cfg.get("tile"):
                grouped = card_tiles(rt, l, lb.T) * p["t_tile_T"] * clk
            elif cfg.get("_tile_us"):
                grouped = card_tiles(rt, l, lb.T) * cfg["_tile_us"]
            else:
                grouped = lb.T * cs_tok * p["shadow_slot"]
            t += (p["shadow_lb"] * clk + grouped) / 1000.0 + SHADOW_EXPERT_BLOCK * g
            acts += SHADOW_EXPERT_BLOCK
        else:
            t += SHADOW_NOCARD_BLOCK * g
            acts += SHADOW_NOCARD_BLOCK
    return t, acts, (TIMED_SHADOW_EVENTS if cfg.get("timed") else 0)


@lru_cache(maxsize=None)
def tiles_of(T, lam):
    """E[ceil(m / 8)] for m ~ Binomial(T, lam / 512): the grouped kernels' 8-column tiles of one card expert."""
    if T <= 0 or lam <= 0:
        return 0.0
    q = min(lam / 512.0, 1.0)
    if q >= 1.0:
        return float(-(-T // 8))
    mu = T * q
    mmax = min(T, int(mu + 12 * math.sqrt(mu) + 30))
    lq, l1q, lg = math.log(q), math.log1p(-q), math.lgamma
    return sum(math.exp(lg(T + 1) - lg(m + 1) - lg(T - m + 1) + m * lq + (T - m) * l1q) * -(-m // 8)
               for m in range(1, mmax + 1))


def card_tiles(rt, l, T):
    return sum(n * tiles_of(T, round(lam, 6)) for lam, n in rt.card[l])


def shadow_tile_us(p):
    """The grouped kernels' cost per (card expert, 8-column tile), from the same trace mean shadow_slot was
    read from (lcg, hot list 384, P 512 CED on, layers 2-39) [derived]."""
    rt = routing_lcg(p, True)
    slots = tiles = 0.0
    for _, _, lbs in prompt_plan(512, True):
        for lb in lbs:
            if lb.full and rt.n_card(lb.l) > 0:
                slots += lb.T * rt.card_slots_tok(lb.l)
                tiles += card_tiles(rt, lb.l, lb.T)
    return p["shadow_slot"] * slots / tiles


def post_card(p, lb):
    """The upload of the host sums and the join (batch.rs:1155-1164, :1769-1838), card ms."""
    return lb.T * X_BYTES / (p["post_h2d_gbs"] * 1e6) + (lb.T * p["post_batch_tok"]) / 1000.0 \
        + POST_ACTS * p["gap_act"] / 1000.0


# ============================================================================ the launch queue

def queue_split(A_r, A_s, E, C_r, C_s, Q, t_i):
    """Closed form of one burst: the host issues A_r route activities (and E events) then A_s shadow
    activities and waits for the route. The queue holds Q activities, so the host is blocked until the
    card has consumed A_r + A_s - Q of them (cardroute-design-report.md:285-296). Units of C_r; t_i
    per call (activity or event) in the same unit."""
    N = A_r + A_s
    K = N - Q
    if K <= 0 or A_r == 0:
        blocked = 0.0
    elif K <= A_r:
        blocked = K * C_r / A_r
    else:
        blocked = C_r + (K - A_r) * (C_s / A_s if A_s else 0.0)
    enqueue = max((N + E) * t_i, blocked)
    return enqueue, max(0.0, C_r - enqueue)


class Card:
    """One card stream, a FIFO of jobs (cumulative activities before, activities, start, end)."""
    __slots__ = ("free", "jobs", "total")

    def __init__(self):
        self.free, self.jobs, self.total = 0.0, [], 0.0

    def copy(self):
        c = Card()
        c.free, c.jobs, c.total = self.free, list(self.jobs), self.total
        return c

    def add(self, acts, dur, avail, issue_end, end_fn=None):
        start = max(self.free, avail)
        end = max(end_fn(start) if end_fn else start + dur, issue_end)
        self.jobs.append((self.total, acts, start, end))
        self.total += acts
        self.free = end
        return start, end

    def consumed_at(self, K):
        """When the card has consumed K activities (cumulative); drops the jobs before them."""
        if K <= 0:
            return -math.inf
        jobs = self.jobs
        i = 0
        while i < len(jobs) - 1 and jobs[i][0] + jobs[i][1] < K:
            i += 1
        if i:
            del jobs[:i]
        cb, n, s, e = jobs[0]
        return s + (e - s) * min(1.0, (K - cb) / n) if n else e


class State:
    __slots__ = ("t", "issued", "card", "pcie", "slots", "fills")

    def __init__(self, ring=0):
        self.t, self.issued, self.card = 0.0, 0.0, Card()
        self.pcie, self.slots, self.fills = 0.0, [0.0] * ring, []

    def copy(self):
        s = State()
        s.t, s.issued, s.card = self.t, self.issued, self.card.copy()
        s.pcie, s.slots, s.fills = self.pcie, list(self.slots), list(self.fills)
        return s

    def take(self, o):
        self.t, self.issued, self.card, self.pcie, self.slots, self.fills = o.t, o.issued, o.card, o.pcie, o.slots, o.fills


def issue(p, st, jobs, Q):
    """The host issues `jobs` [(activities, events, card ms[, end_fn])] back to back, t_issue a call,
    blocked while the queue holds Q activities. Returns (host ms spent, [(start, end)] per job)."""
    t_i = p["t_issue"] / 1000.0
    t0, calls, acts, placed = st.t, 0.0, 0.0, []
    for job in jobs:
        a, e, dur = job[0], job[1], job[2]
        fn = job[3] if len(job) > 3 else None
        placed.append(st.card.add(a, dur, t0 + calls * t_i, t0 + (calls + min(a + e, 1)) * t_i, fn))
        calls += a + e
        acts += a
    st.issued += acts
    t_end = t0 + calls * t_i
    if st.issued > Q:
        t_end = max(t_end, st.card.consumed_at(st.issued - Q))
    st.t = t_end
    return t_end - t0, placed


# ============================================================================ DRAM under the fill

def dram_stretch(p, s0, compute, byts, fills):
    """Union duration (ms) when its kernel model says `compute` ms and it reads `byts` of DRAM, while
    the ring fill takes fill_crossings x PCIe of DRAM inside the `fills` intervals
    (hoststream-design-report.md:124): inside them the union runs at the fraction of its own speed the
    remaining DRAM allows, and at most 1 - fill_steal (the fill threads on its cores' SMT siblings,
    hoststream-design-report.md:50). Outside them its own model, W floor included, holds."""
    if not fills or compute <= 0:
        return compute
    rate = byts / 1e6 / compute                   # GB/s the union reads at on its own
    room = p["dram_eff"] - p["fill_crossings"] * p["pcie_pinned"]
    phi = min(1.0 - p["fill_steal"], max(room, 1e-3) / rate) if rate > 0 else 1.0 - p["fill_steal"]
    left, t = compute, s0
    for f0, f1 in sorted(fills):
        if f1 <= t:
            continue
        if f0 > t:
            if f0 - t >= left:
                return t + left - s0
            left -= f0 - t
            t = f0
        if (f1 - t) * phi >= left:
            return t + left / phi - s0
        left -= (f1 - t) * phi
        t = f1
    return t + left - s0


# ============================================================================ one layer, one prompt

def hot_lams(hostlist, k):
    out, left = [], k
    for lam, n in hostlist:
        if left <= 0:
            break
        take = min(n, left)
        whole = int(take + 1e-9)
        out.extend([lam] * whole)
        if take - whole > 1e-9:
            out.append(lam * (take - whole))
        left -= take
    return out


def run_layer(p, cfg, lbs, rt, work, U, Q, st, k, ring, fills_guess):
    """Host and card through one layer over the group's batches (G = len(lbs)), k host experts
    streamed through a ring of `ring` slots. Mutates `st`; returns per-lb records and stream info."""
    G = len(lbs)
    l = lbs[0].l
    routes, shadows = work
    served = [lb.T > 0 for lb in lbs]
    rec = [dict(lb=lb, enqueue=0.0, wait=0.0, copy=0.0, union=0.0, union_raw=0.0, union_resid=0.0,
                union_expl=0.0, union_x=0.0, union_dram=0.0, slots=0.0, touched=0.0, card_out=routes[b][0],
                card_in=shadows[b][0] if served[b] else 0.0, acts_r=routes[b][1], ev_r=routes[b][2],
                acts_s=shadows[b][1] if served[b] else 0, ev_s=shadows[b][2] if served[b] else 0,
                terms=routes[b][3], parts={})
           for b, lb in enumerate(lbs)]
    info = dict(fills=[], start=None, end=None, pcie_end=None, k=k, gemm=0.0)
    stream = None
    if k > 0:
        lams = hot_lams(rt.host[l], k)
        cols = sum(lb.T for lb in lbs)
        gem = [(p["stream_gfix"] + p["stream_gcol"] * cols * lam / 512.0) / 1000.0 for lam in lams]
        info["gemm"] = sum(gem)
        pms = p["expert_bytes_host"] / (p["pcie_pinned"] * 1e9) * 1e3
        layer_start = st.t

        def end_fn(start):
            # the fill of a layer starts with the layer and runs whenever a ring slot is free; the GEMM
            # of expert i needs its arrival, the group's last route and shadow, and GEMM i - 1
            a_prev, e_prev, fl = max(st.pcie, layer_start), start, []
            for i, g in enumerate(gem):
                j = i % ring
                a0 = max(a_prev, st.slots[j])
                a = a0 + pms
                if fl and abs(fl[-1][1] - a0) < 1e-9:
                    fl[-1] = (fl[-1][0], a)
                else:
                    fl.append((a0, a))
                e_prev = max(a, e_prev) + g
                st.slots[j] = e_prev
                a_prev = a
            st.pcie = a_prev
            info.update(fills=fl, start=start, end=e_prev, pcie_end=a_prev)
            return e_prev

        stream = (2 + math.ceil(len(gem) / ring), 0, 0.0, end_fn)
    fills = [f for f in st.fills] + list(fills_guess or [])
    ev = [None] * G
    first = [(routes[0][1], routes[0][2], routes[0][0])]
    if G == 1:
        if stream:                         # the streamed GEMMs drain the ring before the shadow runs
            first.append(stream)
        if served[0]:
            first.append((shadows[0][1], shadows[0][2], shadows[0][0]))
    e, placed = issue(p, st, first, Q)
    rec[0]["enqueue"] += e
    ev[0] = placed[0][1]
    deferred = []
    for b in range(G):
        if G > 1:
            burst, at_next = [], None
            if stream and b == G - 1:
                burst.append(stream)
            if served[b]:
                burst.append((shadows[b][1], shadows[b][2], shadows[b][0]))
            if b + 1 < G:
                at_next = len(burst)
                burst.append((routes[b + 1][1], routes[b + 1][2], routes[b + 1][0]))
            if burst:
                e, placed = issue(p, st, burst, Q)
                rec[b]["enqueue"] += e
                if at_next is not None:
                    ev[b + 1] = placed[at_next][1]
        if not served[b]:
            continue
        lb, r = lbs[b], rec[b]
        w0 = st.t
        st.t = max(st.t, ev[b])
        r["wait"] = st.t - w0
        if cfg.get("copy"):
            r["copy"] = lb.T * X_BYTES / (p["copy_gbs"] * 1e6)
            st.t += r["copy"]
        u = union_call(p, cfg, rt, l, lb.T, U, skip=k)
        dur = dram_stretch(p, st.t, u["ms"], u["bytes"], fills) if k else u["ms"]
        r.update(union=dur, union_raw=u["raw"], union_resid=u["resid"], union_expl=u["expl"], union_x=u["x"],
                 slots=u["slots"], touched=u["touched"], union_dram=dur - u["ms"], parts=u["parts"])
        st.t += dur
        post = (POST_ACTS, 0, post_card(p, lb))
        if stream:
            deferred.append(post)          # the joins read the streamed experts' sums
        else:
            r["enqueue"] += issue(p, st, [post], Q)[0]
    if deferred:
        issue(p, st, deferred, Q)
    if stream:
        st.fills = [f for f in st.fills if f[1] > st.t] + info["fills"]
    return rec, info


def layer_work(p, cfg, lbs, rt):
    return ([route_lb(p, cfg, lb) for lb in lbs], [shadow_lb(p, cfg, lb, rt) for lb in lbs])


def _close(a, b):
    return len(a) == len(b) and all(abs(x0 - y0) < 1e-6 and abs(x1 - y1) < 1e-6
                                    for (x0, x1), (y0, y1) in zip(a, b))


def settle_layer(p, cfg, lbs, rt, work, U, Q, st, k, ring):
    """run_layer to a fixed point of its fill intervals: a union sees the fill it overlaps, and in a
    group the fill is issued after the first unions."""
    guess = []
    for _ in range(8):
        trial = st.copy()
        rec, info = run_layer(p, cfg, lbs, rt, work, U, Q, trial, k, ring, guess)
        if not k or _close(info["fills"], guess):
            break
        guess = info["fills"]
    return trial, rec, info


def best_k(p, cfg, lbs, rt, work, U, Q, st, ring):
    """The layer's streamed count that minimizes its wall: the resource-balance rule of
    hoststream-design-report.md:92, a load-time constant per (layer, T)."""
    n = rt.n_host(lbs[0].l)
    memo = {}

    def wall(k):
        if k not in memo:
            s, _, _ = settle_layer(p, cfg, lbs, rt, work, U, Q, st, k, ring)
            memo[k] = max(s.t, s.card.free)
        return memo[k]

    cands = sorted(set([0.0] + [float(x) for x in range(8, int(n) + 1, 8)] + [float(n)]))
    best = min(cands, key=lambda k: (wall(k), k))
    for k in range(max(0, int(best) - 7), min(int(n), int(best) + 7) + 1):
        if wall(float(k)) < wall(best) - 1e-9:
            best = float(k)
    return best


def resolve_cfg(cfg, P):
    """The per-prompt choices of a configuration: G (G = 'auto' is every batch, at most 8) and the ring
    (128 borrowed from P >= borrow_min_p, else the unborrowed 8; docs/plan-triage.md:43)."""
    c = dict(cfg)
    nb = len(batches(P))
    g = c.get("G", 1)
    c["G"] = min(8, nb) if g == "auto" else g
    if c.get("stream"):
        if P >= C["borrow_min_p"].value or c.get("force_borrow"):
            c["ring"], c["borrow"] = int(C["ring_borrow"].value), True
        else:
            c["ring"], c["borrow"] = int(C["ring_small"].value), False
    else:
        c["ring"], c["borrow"] = 0, False
    return c


AGG = ("union", "wait", "enqueue", "copy", "card_out", "card_in", "acts_r", "ev_r", "acts_s", "ev_s", "union_raw",
       "union_resid", "union_expl", "union_x", "union_dram", "slots", "touched")
OVER_ALL = ("enqueue", "card_out", "acts_r", "ev_r", "acts_s", "ev_s")


def new_rec(lb, route, shadow):
    return dict(lb=lb, enqueue=0.0, wait=0.0, copy=0.0, union=0.0, union_raw=0.0, union_resid=0.0, union_expl=0.0,
                union_x=0.0, union_dram=0.0, slots=0.0, touched=0.0, card_out=route[0],
                card_in=shadow[0] if lb.T else 0.0, acts_r=route[1], ev_r=route[2], acts_s=shadow[1] if lb.T else 0,
                ev_s=shadow[2] if lb.T else 0, terms=route[3], parts={})


def run_group_wrap(p, cfg, grp, rt, U, Q, st, recs, binds, busy):
    """One group of G >= 2 batches through the wrap order (cardnext-design-report.md section 1.1): the
    group's (layer, batch) items layer-first, and for each item i the host issues [shadow(i), route(i + 1)],
    waits for route(i)'s D2H, runs union(i), then issues its upload and join. route(l + 1, b) reads
    join(l, b), issued G items earlier, so the stream's FIFO keeps every read after its write; the
    stock G (run_layer) issues a layer's first route only after the previous layer's last post."""
    t_i = p["t_issue"] / 1000.0
    seq = [x[2][l] for l in range(L) for x in grp]
    work = [(route_lb(p, cfg, lb), shadow_lb(p, cfg, lb, rt)) for lb in seq]
    ev = [None] * len(seq)
    e, placed = issue(p, st, [(work[0][0][1], work[0][0][2], work[0][0][0])], Q)
    ev[0] = placed[0][1]
    layer_t0, lay_host, lay_card = max(st.t, st.card.free), 0.0, 0.0
    first = e
    for i, lb in enumerate(seq):
        route, shadow = work[i]
        r = new_rec(lb, route, shadow)
        r["enqueue"] += first
        first = 0.0
        burst, at_next = [], None
        if lb.T:
            burst.append((shadow[1], shadow[2], shadow[0]))
        if i + 1 < len(seq):
            at_next = len(burst)
            nxt = work[i + 1][0]
            burst.append((nxt[1], nxt[2], nxt[0]))
        e, placed = issue(p, st, burst, Q)
        if at_next is not None:
            ev[i + 1] = placed[at_next][1]
        if lb.T:
            r["enqueue"] += e
            w0 = st.t
            st.t = max(st.t, ev[i])
            r["wait"] = st.t - w0
            u = union_call(p, cfg, rt, lb.l, lb.T, U)
            r.update(union=u["ms"], union_raw=u["raw"], union_resid=u["resid"], union_expl=u["expl"], union_x=u["x"],
                     slots=u["slots"], touched=u["touched"], parts=u["parts"])
            st.t += u["ms"]
            r["enqueue"] += issue(p, st, [(POST_ACTS, 0, post_card(p, lb))], Q)[0]
        else:
            first = e                      # the next item's route was issued here: its enqueue
        host = r["union"] + (r["acts_r"] + r["ev_r"] + r["acts_s"] + r["ev_s"] + POST_ACTS * (lb.T > 0)) * t_i
        card = r["card_out"] + r["card_in"] + (post_card(p, lb) if lb.T else 0.0)
        busy["host"] += host
        busy["card"] += card
        busy["dram"] += r["touched"] * p["expert_bytes_host"] / 1e9 / p["dram_eff"] * 1e3
        lay_host += host * (lb.T > 0)
        lay_card += card
        recs.append(r)
        if (i + 1) % len(grp) == 0:        # the layer's last batch
            if any(x.T for x in seq[i + 1 - len(grp):i + 1]):
                wall_l = st.t - layer_t0
                top = max((lay_host, "host"), (lay_card, "card"))
                binds.append("serial" if wall_l > 1.10 * top[0] else top[1])
            layer_t0, lay_host, lay_card = st.t, 0.0, 0.0


def run_prompt(p, cfg, P, rt, U, kfix=None):
    """One prompt of P ids: per-lb records, the wall (ms) and pp (tok/s)."""
    cfg = resolve_cfg(cfg, P)
    if cfg.get("wrap") and cfg.get("stream"):
        raise SystemExit("ds41_prefill.py: the wrap order is not modelled with host streaming")
    plan = prompt_plan(P, cfg.get("ced", True))
    G, ring = cfg["G"], cfg["ring"]
    if cfg["borrow"]:
        rt = rt.borrow(p["ring_borrow"] / L)
    Q = p["q_act"]
    st = State(ring)
    recs, ks, binds = [], {}, []
    batch_enq = batch_n = 0.0
    busy = dict(host=0.0, card=0.0, pcie=0.0, dram=0.0)
    pms = p["expert_bytes_host"] / (p["pcie_pinned"] * 1e9) * 1e3
    t_i = p["t_issue"] / 1000.0
    for gi in range(0, len(plan), G):
        grp = plan[gi:gi + G]
        for _, u, _ in grp:                  # a batch's prologue ends in a synchronous H2D
            st.t = max(st.t, st.card.free) + u * p["prologue_tok"] / 1000.0
        for s0, u, _ in grp:                 # then its gathers and embeddings, before layer 0
            a = batch_acts(s0, u)
            e, _ = issue(p, st, [(a, 0, a * (p["small_k"] + p["gap_act"]) / 1000.0)], Q)
            batch_enq += e
            batch_n += a
        if cfg.get("wrap") and len(grp) >= 2:
            run_group_wrap(p, cfg, grp, rt, U, Q, st, recs, binds, busy)
            continue
        for l in range(L):
            lbs = [x[2][l] for x in grp]
            work = layer_work(p, cfg, lbs, rt)
            k = 0.0
            if ring and any(lb.T for lb in lbs):
                k = kfix[(gi, l)] if kfix is not None else best_k(p, cfg, lbs, rt, work, U, Q, st, ring)
            t0 = max(st.t, st.card.free)
            st2, rec, info = settle_layer(p, cfg, lbs, rt, work, U, Q, st, k, ring)
            st.take(st2)
            ks[(gi, l)] = k
            recs.extend(rec)
            host = sum(r["union"] + r["copy"] + (r["acts_r"] + r["ev_r"] + r["acts_s"] + r["ev_s"] + POST_ACTS) * t_i for r in rec)
            card = sum(r["card_out"] + r["card_in"] + (post_card(p, r["lb"]) if r["lb"].T else 0.0) for r in rec)
            card += info["gemm"]                    # the stream's GEMMs; its waits on PCIe are not card work
            busy["host"] += host
            busy["card"] += card
            busy["pcie"] += k * pms
            busy["dram"] += sum(r["touched"] * p["expert_bytes_host"] for r in rec) / 1e9 / p["dram_eff"] * 1e3 \
                + k * pms * p["fill_crossings"] * p["pcie_pinned"] / p["dram_eff"]
            if any(lb.T for lb in lbs):
                wall_l = max(st.t, st.card.free) - t0
                top = max((host, "host"), (card, "card"), (k * pms, "pcie"))
                binds.append("serial" if wall_l > 1.10 * top[0] else top[1])
    issue(p, st, [(HEAD_ACTS, 0, p["head"])], Q)
    wall = max(st.t, st.card.free)
    reup = p["borrow_reupload"] if cfg["borrow"] else 0.0
    wall += reup
    served = [r for r in recs if r["lb"].T]
    n = max(1, len(served))
    # the stat line's per-layer-batch values are sums over every layer of every batch over the served
    # ones (body/prefill.rs:414-456, :985-990): a layer with no block still issues and runs its route
    agg = {k2: sum(r[k2] for r in (recs if k2 in OVER_ALL else served)) / n for k2 in AGG}
    agg["enqueue"] += batch_enq / n
    agg["card_proj"] = sum(r["terms"]["proj"] + r["terms"]["small"] + r["terms"]["part_proj"] for r in recs) / n
    # the stat line's entries_route + entries_shadow: the layers' calls, the posts, the batches' gathers
    agg["entries"] = agg["acts_r"] + agg["ev_r"] + agg["acts_s"] + agg["ev_s"] + (POST_ACTS * len(served) + batch_n) / n
    agg["route"] = agg["wait"] + agg["enqueue"]
    agg["nonunion"] = (wall - sum(r["union"] for r in served)) / n
    agg["prologue"] = sum(u for _, u, _ in plan) * p["prologue_tok"] / 1000.0
    agg["host_slots"] = sum(r["slots"] for r in served)
    busy["host"] += agg["prologue"] + batch_enq
    return dict(wall=wall, pp=P / wall * 1000.0, recs=recs, agg=agg, n_lb=len(served), ks=ks, binds=binds,
                cfg=cfg, reupload=reup, busy=busy, duty=busy["card"] / wall)


# ============================================================================ configurations

CONFIGS = {
    # what each recorded commit ran: route "prebulk" = one-token router launches and a places launch
    # per chunk (d37136c batch.rs:815-866) and the per-slot shadow; copy = the union's copy of x before
    # hostserve (c58cb37); the union kernel (hosttile before h1fold, 5685a15); flow = the union's
    # dispatch flow (chunks before 9626c7f, five after); hot = the hot list 384 (else the id prefix)
    "ds41batch": dict(commit="43cd107", route="prebulk", union="hosttile", unionreal_cut=True, copy=True, ced=False,
                      flow="chunks"),
    "S13": dict(commit="e4d0aae", route="prebulk", copy=True, flow="chunks"),
    "S13b": dict(commit="672ffab", route="prebulk", copy=True, flow="chunks"),
    "S14pre": dict(commit="d37136c", route="prebulk", copy=True, flow="chunks"),
    "S14": dict(commit="9d61a13", route="bulk", copy=True, flow="chunks"),
    "S15": dict(commit="de3a086", route="bulk", copy=True, flow="chunks"),
    "nsys": dict(commit="afe86d5", route="bulk", copy=False, flow="chunks"),
    "UDbase": dict(commit="af929ae", route="bulk", copy=False, flow="chunks", hot=False),
    "UD": dict(commit="9626c7f", route="bulk", copy=False, flow="five", hot=False),
    "now": dict(commit="9626c7f", route="bulk", copy=False, flow="five"),
    # the B1 lease (09-26#b1-pp-ab): main 0bcee2c with B1 and the tree before it (74fe84c, the uniondispatch
    # flow), no hot list, BLOOMERY_STEP_STATS=1 (timed: the card-timing marks are queue events); the prose
    # calibration ran the same B1 binary with hot list 384
    "B1": dict(commit="0bcee2c", route="bulk", copy=False, flow="five", b1=True, hot=False, timed=True),
    "B1base": dict(commit="74fe84c", route="bulk", copy=False, flow="five", hot=False, timed=True),
    "B1prose": dict(commit="0bcee2c", route="bulk", copy=False, flow="five", b1=True, timed=True),
}
DEFAULT_ANCHORS = dict(union="anchor_union_s13b", resid_mode="slot")


def evaluate(p, cfg, P, rname="lcg", anchors=None, kfix=None):
    """A prompt with the union's f and X_u re-derived for these constants."""
    anchors = anchors or DEFAULT_ANCHORS
    U = union_anchor(p, anchors)
    if anchors.get("shadow") == "tile":
        cfg = dict(cfg, _tile_us=shadow_tile_us(p))
    r = run_prompt(p, cfg, P, routing(p, rname, cfg.get("hot", True)), U, kfix)
    r["U"] = U
    return r


# ============================================================================ the ruler

def tol_abs(p, n):
    """A prediction from other leases' constants: the between-lease move of one bench binary
    (rig-log 09-25#h1fold-union, the larger of 4.3 % and 1.6 %) and the row's own same-binary noise
    as a known-sigma z interval, 1.96 x 0.6 % / sqrt(n), for every n (AGENTS.md 'Know the ruler')."""
    return math.hypot(p["between_lease"], 1.96 * p["sd_same_binary"] / math.sqrt(n))


def tol_ratio(p, n):
    """A same-lease ratio of two arms' means over n rounds each: t(2n - 2) x 0.6 % x sqrt(2 / n), which
    is AGENTS.md's +-1.0 % at four rounds and +-0.8 % at six. One round has no t: None, and the row
    prints 'no interval (one round)' unscored."""
    if n <= 1:
        return None
    return T975(2 * n - 2) * p["sd_same_binary"] * math.sqrt(2.0 / n)


# ============================================================================ backtest rows

# (id, config, P, overrides, quantity, measured, rounds, kind, source)
# kind: abs = against other leases' constants; ratio = same lease, two arms; info = printed, not scored
UDS = "09-26#uniondispatch-ab (ud-sit depth.log, rounds"
B1S = "09-26#b1-pp-ab (b1pp, b1pp2 run.log; clean rounds"
PRS = "09-26#b1-pp-ab (cardin run.log;"
ROWS = [
    ("B.pp512", "ds41batch", 512, {}, "pp", 91.2, 2, "abs", "09-25#ds41batch-pp: 90.95, 91.46"),
    ("B.pp4096", "ds41batch", 4096, {}, "pp", 89.2, 2, "abs", "89.40, 89.02"),
    ("B.union512", "ds41batch", 512, {}, "union", 93.49, 2, "abs", "3,761.5 / 3,718.0 ms over 40"),
    ("B.union4096", "ds41batch", 4096, {}, "union", 94.79, 2, "abs", "30,276.5 / 30,386.3 ms over 320"),
    ("S13.pp512.on", "S13", 512, {}, "pp", 109.89, 2, "abs", "09-25#v41-prefill-resit"),
    ("S13.pp4096.on", "S13", 4096, {}, "pp", 152.83, 2, "abs", ""),
    ("S13.pp512.off", "S13", 512, {"ced": False}, "pp", 104.66, 2, "abs", "BLOOMERY_CED=off"),
    ("S13.pp4096.off", "S13", 4096, {"ced": False}, "pp", 102.56, 2, "abs", ""),
    ("S13.ratio512.on_off", "S13", 512, {"vs": ("S13", {"ced": False})}, "ratio", 1.0499, 2, "ratio", "+-0.0086"),
    ("S13.ratio4096.on_off", "S13", 4096, {"vs": ("S13", {"ced": False})}, "ratio", 1.4902, 2, "ratio", "+-0.0641"),
    ("S13.ratio512.oxcpu", "S13", 512, {"vs": ("S13", {})}, "ratio", 1.0016, 2, "ratio",
     "oxcpu is no model term: 1 by construction"),
    ("S13b.pp512.on", "S13b", 512, {}, "pp", 110.69, 1, "abs", "09-25#v41-prefill-resit-b"),
    ("S13b.pp512.off", "S13b", 512, {"ced": False}, "pp", 105.79, 1, "abs", ""),
    ("S13b.union.on", "S13b", 512, {}, "union", 71.8, 1, "abs", "2,872.8 ms / 40"),
    ("S13b.union.off", "S13b", 512, {"ced": False}, "union", 76.1, 1, "abs", "3,044.3 ms / 40"),
    ("S13b.union.preox", "S13b", 512, {}, "union", 73.3, 1, "abs", "oxcpu's pre binary; no model term"),
    ("S13b.nonunion.on", "S13b", 512, {}, "nonunion", 43.8, 1, "abs", "(4,626 - 2,873) / 40"),
    ("S13b.slots.on", "S13b", 512, {}, "host_slots", 90629, 1, "info", "a routing count, no timing ruler"),
    ("S13b.slots.off", "S13b", 512, {"ced": False}, "host_slots", 96525, 1, "info", ""),
    ("S14.pp512.expert", "S14", 512, {"cold": 0.5}, "pp", 121.38, 2, "abs", "09-25#v41-prefill-s14"),
    ("S14.pp512.slot", "S14", 512, {"arm": "slot"}, "pp", 122.59, 2, "abs", ""),
    ("S14.pp512.pre", "S14pre", 512, {}, "pp", 111.30, 2, "abs", "d37136c, the pre-landing binary"),
    ("S14.pp4096.expert", "S14", 4096, {"cold": 0.5}, "pp", 169.39, 2, "abs", ""),
    ("S14.pp4096.slot", "S14", 4096, {"arm": "slot"}, "pp", 170.82, 2, "abs", ""),
    ("S14.pp4096.pre", "S14pre", 4096, {}, "pp", 155.81, 2, "abs", ""),
    ("S14.ratio512.expert_pre", "S14", 512, {"cold": 0.5, "vs": ("S14pre", {})}, "ratio", 1.0906, 2, "ratio",
     "rounds 1.0845, 1.0966"),
    ("S14.ratio4096.expert_pre", "S14", 4096, {"cold": 0.5, "vs": ("S14pre", {})}, "ratio", 1.0872, 2, "ratio",
     "1.0756, 1.0988"),
    ("S14.ratio512.expert_slot", "S14", 512, {"cold": 0.5, "vs": ("S14", {"arm": "slot"})}, "ratio", 0.99015, 2,
     "ratio", "0.9796, 1.0007"),
    ("S14.ratio4096.expert_slot", "S14", 4096, {"cold": 0.5, "vs": ("S14", {"arm": "slot"})}, "ratio", 0.99165, 2,
     "ratio", "0.9855, 0.9978"),
    ("S14.p512.expert.union", "S14", 512, {}, "union", 72.3, 1, "abs", "stat prefill split, round 2"),
    ("S14.p512.expert.wait", "S14", 512, {}, "wait", 11.7, 1, "abs", ""),
    ("S14.p512.expert.enqueue", "S14", 512, {}, "enqueue", 18.8, 1, "abs", ""),
    ("S14.p512.expert.route", "S14", 512, {}, "route", 30.5, 1, "abs", "wait + enqueue"),
    ("S14.p512.expert.nonunion", "S14", 512, {}, "nonunion", 32.3, 1, "abs", ""),
    ("S14.p512.slot.union", "S14", 512, {"arm": "slot"}, "union", 72.4, 1, "abs", ""),
    ("S14.p512.slot.wait", "S14", 512, {"arm": "slot"}, "wait", 9.2, 1, "abs", ""),
    ("S14.p512.slot.enqueue", "S14", 512, {"arm": "slot"}, "enqueue", 21.2, 1, "abs", ""),
    ("S14.p512.slot.route", "S14", 512, {"arm": "slot"}, "route", 30.4, 1, "abs", ""),
    ("S14.p512.slot.nonunion", "S14", 512, {"arm": "slot"}, "nonunion", 32.3, 1, "abs", ""),
    ("S14.p512.pre.union", "S14pre", 512, {}, "union", 72.4, 1, "abs", ""),
    ("S14.p512.pre.nonunion", "S14pre", 512, {}, "nonunion", 42.3, 1, "abs", ""),
    ("S14.p4096.expert.union", "S14", 4096, {}, "union", 73.9, 1, "abs", ""),
    ("S14.p4096.expert.wait", "S14", 4096, {}, "wait", 12.1, 1, "abs", ""),
    ("S14.p4096.expert.enqueue", "S14", 4096, {}, "enqueue", 20.8, 1, "abs", ""),
    ("S14.p4096.expert.route", "S14", 4096, {}, "route", 32.9, 1, "abs", "wait + enqueue"),
    ("S14.p4096.expert.nonunion", "S14", 4096, {}, "nonunion", 35.1, 1, "abs", ""),
    ("S14.p4096.pre.union", "S14pre", 4096, {}, "union", 74.4, 1, "abs", ""),
    ("S14.p4096.pre.nonunion", "S14pre", 4096, {}, "nonunion", 45.4, 1, "abs", ""),
    ("S14.prologue512", "S14", 512, {}, "prologue", 40.6, 1, "abs", "round 2"),
    ("S14.prologue4096", "S14", 4096, {}, "prologue", 308.3, 1, "abs", "round 2"),
    ("S15.expert.enqueue", "S15", 384, {}, "enqueue", 9.02, 1, "abs", "09-26#v41-cardroute-queue"),
    ("S15.expert.wait", "S15", 384, {}, "wait", 14.07, 1, "abs", ""),
    ("S15.expert.card_out", "S15", 384, {}, "card_out", 22.50, 1, "abs", ""),
    ("S15.expert.card_in", "S15", 384, {}, "card_in", 16.63, 1, "abs",
     "an event interval (the model: kernel sum + idle per activity)"),
    ("S15.expert.union", "S15", 384, {}, "union", 58.35, 1, "info", "[cpu-busy 1/1]: not used"),
    ("S15.slot.enqueue", "S15", 384, {"arm": "slot"}, "enqueue", 10.85, 1, "abs", ""),
    ("S15.slot.wait", "S15", 384, {"arm": "slot"}, "wait", 12.25, 1, "abs", ""),
    ("S15.slot.card_out", "S15", 384, {"arm": "slot"}, "card_out", 22.52, 1, "abs", ""),
    ("S15.slot.card_in", "S15", 384, {"arm": "slot"}, "card_in", 19.94, 1, "abs", "s15.log stat line"),
    ("S15.slot.union", "S15", 384, {"arm": "slot"}, "union", 55.44, 1, "abs", "clean"),
    ("S15.slot.pp", "S15", 384, {"arm": "slot"}, "pp", 120.1, 1, "abs", "clean"),
    ("S15.diff.enqueue", "S15", 384, {"arm": "slot", "vs": ("S15", {})}, "diff:enqueue", 1.83, 1, "info",
     "slot - expert"),
    ("S15.diff.wait", "S15", 384, {"arm": "slot", "vs": ("S15", {})}, "diff:wait", -1.82, 1, "info", ""),
    # uniondispatch, one lease, no hot list: base = main af929ae (chunk flow), new = 9626c7f (five);
    # the new arm's round 1 was the lease's first process at each P (prologue 114.2 / 908.3 ms)
    ("UD.base512.union", "UDbase", 512, {}, "union", 77.07, 3, "abs", UDS + " 76.97, 77.68, 76.57)"),
    ("UD.base512.wait", "UDbase", 512, {}, "wait", 11.63, 3, "abs", "11.60, 11.65, 11.64"),
    ("UD.base512.enqueue", "UDbase", 512, {}, "enqueue", 18.64, 3, "abs", "18.58, 18.67, 18.66"),
    ("UD.base512.pp", "UDbase", 512, {}, "pp", 118.04, 3, "abs", "118.26, 117.32, 118.54"),
    ("UD.new512.union", "UD", 512, {}, "union", 66.73, 3, "abs", "66.83, 67.17, 66.20"),
    ("UD.new512.wait", "UD", 512, {}, "wait", 11.64, 3, "abs", "11.57, 11.67, 11.67"),
    ("UD.new512.enqueue", "UD", 512, {}, "enqueue", 18.67, 3, "abs", "18.53, 18.75, 18.73"),
    ("UD.new512.pp", "UD", 512, {"cold": 1 / 3}, "pp", 129.63, 3, "abs", "128.25 (cold), 129.67, 130.98"),
    ("UD.base4096.union", "UDbase", 4096, {}, "union", 77.59, 3, "abs", "77.40, 77.81, 77.56"),
    ("UD.base4096.wait", "UDbase", 4096, {}, "wait", 12.06, 3, "abs", "12.06, 12.05, 12.08"),
    ("UD.base4096.enqueue", "UDbase", 4096, {}, "enqueue", 20.69, 3, "abs", "20.70, 20.67, 20.71"),
    ("UD.base4096.pp", "UDbase", 4096, {}, "pp", 166.25, 3, "abs", "166.52, 165.97, 166.26"),
    ("UD.new4096.union", "UD", 4096, {}, "union", 68.14, 3, "abs", "67.84, 68.73, 67.86"),
    ("UD.new4096.wait", "UD", 4096, {}, "wait", 12.07, 3, "abs", "12.05, 12.06, 12.09"),
    ("UD.new4096.enqueue", "UD", 4096, {}, "enqueue", 20.73, 3, "abs", "20.70, 20.72, 20.78"),
    ("UD.new4096.pp", "UD", 4096, {"cold": 1 / 3}, "pp", 180.02, 3, "abs", "177.68 (cold), 180.46, 181.92"),
    ("UD.delta512", "UD", 512, {"vs": ("UDbase", {})}, "diff:union", -10.34, 3, "info", "per round -10.14, -10.51, -10.37"),
    ("UD.ratio512", "UD", 512, {"cold": 1 / 3, "vs": ("UDbase", {})}, "ratio", 1.098, 3, "ratio",
     "+-0.030; rounds 1.0845, 1.1053, 1.1049"),
    ("UD.ratio4096", "UD", 4096, {"cold": 1 / 3, "vs": ("UDbase", {})}, "ratio", 1.083, 3, "ratio",
     "+-0.035; rounds 1.0670, 1.0873, 1.0942"),
    # B1, two leases in a row, no hot list, lcg, BLOOMERY_STEP_STATS=1: main 0bcee2c (B1; generate_ds41
    # f0dddb6f7667, rebuilt 75dadf84c752 for the second lease) and 74fe84c (791dec6236ff). Four clean rounds
    # (b1pp r2, r3; b1pp2 r1, r2); b1pp round 1 is void: base P 512 and ours P 4096 [cpu-busy] (a build),
    # ours P 512 the lease's first process (prologue 108.3 ms). Stat values are the four rows' means.
    ("B1.pp512", "B1", 512, {}, "pp", 144.63, 4, "abs", B1S + " 144.21, 143.46, 145.47, 145.37; void 142.49)"),
    ("B1.pp4096", "B1", 4096, {}, "pp", 200.95, 4, "abs", "201.42, 200.63, 201.56, 200.19; void 197.54 [cpu-busy]"),
    ("B1.base.pp512", "B1base", 512, {}, "pp", 130.56, 4, "abs", "131.08, 130.04, 130.79, 130.34; void 129.88 [cpu-busy]"),
    ("B1.base.pp4096", "B1base", 4096, {}, "pp", 182.36, 4, "abs", "183.12, 183.14, 181.28, 181.89; void 182.17"),
    ("B1.ratio512", "B1", 512, {"vs": ("B1base", {})}, "ratio", 1.1077, 4, "ratio",
     "+-0.011; rounds 1.1002, 1.1032, 1.1122, 1.1153"),
    ("B1.ratio4096", "B1", 4096, {"vs": ("B1base", {})}, "ratio", 1.1020, 4, "ratio",
     "+-0.011; rounds 1.0999, 1.0955, 1.1119, 1.1006"),
    ("B1.p512.card_proj", "B1", 512, {}, "card_proj", 12.13, 4, "abs", "12.19, 12.11, 12.10, 12.12"),
    ("B1.p512.card_out", "B1", 512, {}, "card_out", 19.61, 4, "abs", "19.70, 19.56, 19.59, 19.59"),
    ("B1.p512.card_in", "B1", 512, {}, "card_in", 18.75, 4, "abs", "18.79, 18.80, 18.69, 18.70"),
    ("B1.p512.union", "B1", 512, {}, "union", 67.01, 4, "abs", "67.19, 67.78, 66.51, 66.57"),
    ("B1.p512.wait", "B1", 512, {}, "wait", 16.56, 4, "abs", "16.67, 16.50, 16.54, 16.53"),
    ("B1.p512.enqueue", "B1", 512, {}, "enqueue", 3.81, 4, "abs", "3.79, 3.82, 3.81, 3.82"),
    ("B1.p512.entries", "B1", 512, {}, "entries", 983.4, 4, "info", "entries_route 506.4 + entries_shadow 477.0"),
    ("B1.p4096.card_proj", "B1", 4096, {}, "card_proj", 12.24, 4, "abs", "12.27, 12.24, 12.20, 12.23"),
    ("B1.p4096.card_out", "B1", 4096, {}, "card_out", 21.90, 4, "abs", "21.96, 21.89, 21.84, 21.89"),
    ("B1.p4096.card_in", "B1", 4096, {}, "card_in", 19.20, 4, "abs", "19.27, 19.22, 19.11, 19.20"),
    ("B1.p4096.union", "B1", 4096, {}, "union", 68.43, 4, "abs", "68.16, 68.58, 68.18, 68.78"),
    ("B1.p4096.wait", "B1", 4096, {}, "wait", 18.02, 4, "abs", "18.07, 18.02, 17.96, 18.01"),
    ("B1.p4096.enqueue", "B1", 4096, {}, "enqueue", 4.62, 4, "abs", "4.63, 4.61, 4.62, 4.62"),
    ("B1.p4096.entries", "B1", 4096, {}, "entries", 1024.9, 4, "info", "549.5 + 475.4 over 220 served (320 run)"),
    ("B1.base.p512.card_out", "B1base", 512, {}, "card_out", 29.69, 4, "abs", "29.68, 29.72, 29.63, 29.72"),
    ("B1.base.p512.card_in", "B1base", 512, {}, "card_in", 18.43, 4, "abs", "18.43, 18.47, 18.38, 18.45"),
    ("B1.base.p512.union", "B1base", 512, {}, "union", 66.47, 4, "abs", "66.09, 66.83, 66.34, 66.60"),
    ("B1.base.p512.wait", "B1base", 512, {}, "wait", 11.68, 4, "abs", "11.68, 11.69, 11.67, 11.69"),
    ("B1.base.p512.enqueue", "B1base", 512, {}, "enqueue", 18.76, 4, "abs", "18.75, 18.79, 18.72, 18.78"),
    ("B1.base.p4096.card_out", "B1base", 4096, {}, "card_out", 32.24, 4, "abs", "32.31, 32.22, 32.21, 32.23"),
    ("B1.base.p4096.card_in", "B1base", 4096, {}, "card_in", 18.96, 4, "abs", "18.98, 18.97, 18.95, 18.95"),
    ("B1.base.p4096.union", "B1base", 4096, {}, "union", 67.55, 4, "abs", "67.07, 67.15, 68.19, 67.80"),
    ("B1.base.p4096.wait", "B1base", 4096, {}, "wait", 12.11, 4, "abs", "12.13, 12.10, 12.10, 12.10"),
    ("B1.base.p4096.enqueue", "B1base", 4096, {}, "enqueue", 20.85, 4, "abs", "20.89, 20.83, 20.82, 20.84"),
    ("B1.diff512.card_out", "B1", 512, {"vs": ("B1base", {})}, "diff:card_out", -10.08, 4, "info", "B1 - base"),
    # the prose calibration: the same B1 binary (75dadf84c752), hot list 384, the prose prompt
    # (corpus-prose.ids, the first 512), one prompt, the lease's only process; the card was busy ~0.91 of the
    # wall, and its route ran clk_cardbound slower than lcg's (the row's own condition)
    ("B1.prose.pp", "B1prose", 512, {"rt": "prose", "clk": "cardbound"}, "pp", 171.64, 1, "abs",
     PRS + " time prompt 2,982.96 ms; prologue 81.3 of it, the model's 39.6)"),
    ("B1.prose.pp.clk1", "B1prose", 512, {"rt": "prose"}, "pp", 171.64, 1, "info", "the same row at lcg's clock"),
    ("B1.prose.card_out", "B1prose", 512, {"rt": "prose", "clk": "cardbound"}, "card_out", 22.05, 1, "abs", ""),
    ("B1.prose.card_proj", "B1prose", 512, {"rt": "prose", "clk": "cardbound"}, "card_proj", 13.72, 1, "abs", ""),
    ("B1.prose.card_in", "B1prose", 512, {"rt": "prose", "clk": "cardbound"}, "card_in", 45.28, 1, "abs",
     "the grouped pair per card slot: 63,203 card slots (6 x 19,232 - 52,189)"),
    ("B1.prose.union", "B1prose", 512, {"rt": "prose", "clk": "cardbound"}, "union", 36.42, 1, "abs", ""),
    ("B1.prose.wait", "B1prose", 512, {"rt": "prose", "clk": "cardbound"}, "wait", 30.49, 1, "abs",
     "the shadow's excess over the union, layers 0-1's host-only union, the route"),
    ("B1.prose.enqueue", "B1prose", 512, {"rt": "prose", "clk": "cardbound"}, "enqueue", 5.62, 1, "abs", ""),
    ("B1.prose.slots", "B1prose", 512, {"rt": "prose"}, "host_slots", 52189, 1, "info", "a routing count, no timing ruler"),
    ("B1.prose.prologue", "B1prose", 512, {"rt": "prose"}, "prologue", 81.3, 1, "info",
     "no model term: the prompt's first process in its lease (eng_cold 2,741; lcg's warm 44.3)"),
]

# rows a constant in use was taken from: printed as 'anchor', not scored
ANCHOR_ROWS = {
    "anchor_union_s13b": {"S13b.union.off"},
    "anchor_union_s14s": {"S14.p512.slot.union"},
    "anchor_union_ud512": {"UD.new512.union"},
    "s_host_lcg": {"S13b.slots.off"},
    "prologue_tok": {"S14.prologue512", "S14.prologue4096"},
    "ud_delta_512": {"UD.delta512", "UD.ratio512"},
    "full_res_lat": {"B1.p512.card_proj"},
    "t_issue": {"B1.p512.enqueue"},
    "clk_cardbound": {"B1.prose.card_out"},
    "prose_swap": {"B1.prose.slots"},
    "prose_phi": {"B1.prose.union"},
}

# the term that breaks a red row, named after reading its terms (--explain <row>, the diagnostics below)
TERMS = {
    "router_tok": "the pre-ds41bulk one-token router launch, 15.1 us derived from ds41router's prediction"
                  " (plan-ledger.md:1167); S14's pre - post non-union difference implies ~21 us (21.56 was measured"
                  " before ds41router); only pre-ds41bulk rows read it",
    "union-T": "S15's clean P 384 union sits 4.9 ms over its kernel sum, less than the join tails alone (f 0.48 x K / 4 ="
               " 6.1 ms): the cause model does not scale to it; the kernel sum at m-bar 5.4 (t(m) fitted at m 8 and 16, the"
               " W floor) or the lease (between-lease 4.3 %) - the CED-off T sweep decides",
    "slot-shadow": "the slot arm's shadow is derived (the traced per-chunk kernels + each slot's expert bytes at bw_card),"
                   " not traced; S15's card_in reads it 2.4 ms longer at P 384 (the per-slot kernels are latency-bound"
                   " m-column launches, not DRAM-bound)",
    "union-duty": "the same union code reads 0.54 (P 512) and 0.88 (P 4096) ms a layer-batch longer in the B1 arm than in"
                  " the base arm of the same lease (67.01 / 66.47, 68.43 / 67.55), where the host spends a larger share of"
                  " the wall in the union (P 4096: 75 % against 67 %); the model's union is a function of the routing"
                  " alone, so it gives B1 the base's union and reads B1's P 4096 gain 1.2 % high. A host clock under"
                  " the higher all-core duty is the candidate (CPU frequency is not in the witness lines)",
}
BLAME = dict({r: "router_tok" for r in ("S13.pp4096.on", "S13.pp4096.off", "S13b.nonunion.on", "S14.ratio512.expert_pre",
                                        "S14.ratio4096.expert_pre", "S14.p512.pre.nonunion", "S14.p4096.pre.nonunion",
                                        "S14.pp4096.pre", "B.pp4096")},
             **{r: "union-T" for r in ("S15.slot.union", "S15.slot.pp")},
             **{"S15.slot.card_in": "slot-shadow", "B1.ratio4096": "union-duty"})


def cold_pp(p, res, P, frac):
    """A recorded pp whose rounds include `frac` cold rounds (the lease's first process at that P): the
    mean of cold and warm rounds."""
    extra = p["prologue_cold_extra_512"] if P <= 512 else p["prologue_cold_extra_4096"]
    return (frac * P / (res["wall"] + extra) + (1 - frac) * P / res["wall"]) * 1000.0


def row_cfg(p, cname, over):
    """A row's configuration and routing: `rt` names the routing (lcg unless given), `clk` = "cardbound"
    runs the card at clk_cardbound (the row's own measured condition)."""
    over = dict(over)
    rname = over.pop("rt", "lcg")
    if over.pop("clk", None) == "cardbound":
        over["_clk"] = p["clk_cardbound"]
    return dict(CONFIGS[cname], **over), rname


def predict_row(p, row, anchors=None):
    rid, cname, P, over, q, meas, n, kind, _ = row
    over = dict(over)
    cold = over.pop("cold", 0.0)
    vs = over.pop("vs", None)
    cfg, rname = row_cfg(p, cname, over)
    res = evaluate(p, cfg, P, rname, anchors)
    pp = cold_pp(p, res, P, cold) if cold else res["pp"]
    if q == "ratio":
        ocfg, orn = row_cfg(p, *vs)
        other = evaluate(p, ocfg, P, orn, anchors)
        return pp / other["pp"], res
    if q.startswith("diff:"):
        term = q.split(":")[1]
        ocfg, orn = row_cfg(p, *vs)
        other = evaluate(p, ocfg, P, orn, anchors)
        return res["agg"][term] - other["agg"][term], res
    if q == "pp":
        return pp, res
    return res["agg"][q], res


def scored(rid, kind, q, skip, n=2):
    return rid not in skip and kind != "info" and not q.startswith("diff:") and not (kind == "ratio" and n <= 1)


IN_USE = ("anchor_union_s13b", "s_host_lcg", "prologue_tok", "ud_delta_512", "full_res_lat", "t_issue", "clk_cardbound",
          "prose_swap", "prose_phi")


def backtest(verbose=True, out_rows=None):
    p = central()
    skip = set().union(*(ANCHOR_ROWS[k] for k in IN_USE))
    out, red = [], []
    hdr = f"{'row':28} {'kind':5} {'measured':>10} {'predicted':>10} {'err %':>7} {'tol %':>6}  verdict"
    out += [hdr, "-" * len(hdr)]
    for row in ROWS:
        rid, cname, P, over, q, meas, n, kind, src = row
        pred, _ = predict_row(p, row)
        err = (pred - meas) / abs(meas) * 100.0
        tol = float("nan")
        if rid in skip:
            verdict = "anchor"
        elif kind == "ratio" and n <= 1:
            verdict = "no interval (one round)"
        elif not scored(rid, kind, q, skip, n):
            verdict = "info"
        else:
            tol = tol_ratio(p, n) if kind == "ratio" else tol_abs(p, n)
            verdict = "ok" if abs(err) <= tol else "RED"
            if verdict == "RED":
                red.append(rid)
        if out_rows is not None:
            out_rows[rid] = (meas, pred, err, verdict)
        f = "10.4f" if kind == "ratio" else "10.2f"
        line = f"{rid:28} {kind:5} {meas:{f}} {pred:{f}} {err:+7.2f} {tol:6.2f}  {verdict}"
        if verdict == "RED":
            line += f"  <- {BLAME.get(rid, 'UNNAMED')}"
        out.append(line)
    out += ["", "per served layer-batch (ms), model central [derived]; acts = activities (queue slots), ev = events:",
            f"{'config':12} {'P':>5} {'union':>7} {'kernel':>7} {'cause':>6} {'X_u':>5} {'wait':>6} {'enq':>6} {'copy':>5}"
            f" {'route':>6} {'shadow':>6} {'A_r':>5} {'ev':>4} {'A_s':>4} {'nonun':>6} {'lb':>4} {'pp':>7}"]
    for key, cname, P, over in (("ds41batch", "ds41batch", 512, {}), ("ds41batch", "ds41batch", 4096, {}),
                                ("S13 on", "S13", 512, {}), ("S13 on", "S13", 4096, {}),
                                ("S13 off", "S13", 512, {"ced": False}), ("S13 off", "S13", 4096, {"ced": False}),
                                ("S14 expert", "S14", 512, {}), ("S14 slot", "S14", 512, {"arm": "slot"}),
                                ("S14 pre", "S14pre", 512, {}), ("S14 expert", "S14", 4096, {}),
                                ("S14 pre", "S14pre", 4096, {}), ("S15 expert", "S15", 384, {}),
                                ("S15 slot", "S15", 384, {"arm": "slot"}), ("nsys", "nsys", 512, {}),
                                ("UD base", "UDbase", 512, {}), ("UD new", "UD", 512, {}),
                                ("UD base", "UDbase", 4096, {}), ("UD new", "UD", 4096, {}),
                                ("now", "now", 512, {}), ("now", "now", 4096, {}),
                                ("B1 base", "B1base", 512, {}), ("B1", "B1", 512, {}), ("B1 base", "B1base", 4096, {}),
                                ("B1", "B1", 4096, {}), ("B1 prose", "B1prose", 512, {"rt": "prose", "clk": "cardbound"})):
        cfg, rname = row_cfg(p, cname, over)
        r = evaluate(p, cfg, P, rname)
        a = r["agg"]
        out.append(f"{key:12} {P:5d} {a['union']:7.2f} {a['union_raw']:7.2f} {a['union_expl']:6.2f} {a['union_x']:5.2f}"
                   f" {a['wait']:6.2f} {a['enqueue']:6.2f} {a['copy']:5.2f} {a['card_out']:6.2f} {a['card_in']:6.2f}"
                   f" {a['acts_r']:5.0f} {a['ev_r']:4.0f} {a['acts_s']:4.0f} {a['nonunion']:6.2f} {r['n_lb']:4d} {r['pp']:7.2f}")
    U = union_anchor(p, DEFAULT_ANCHORS)
    flo = union_anchor(p.but(ud_delta_512=C["ud_delta_512"].lo), DEFAULT_ANCHORS)["f"]
    fhi = union_anchor(p.but(ud_delta_512=C["ud_delta_512"].hi), DEFAULT_ANCHORS)["f"]
    xud = union_anchor(p, dict(DEFAULT_ANCHORS, union="anchor_union_ud512"))["x"]
    out += ["", f"union [derived]: f {U['f']:.3f} ({flo:.3f}-{fhi:.3f} over the difference's 95 % interval)"
                f"; X_u anchored on the uniondispatch new arm instead: {xud * 1000:.3f} us a slot"]
    out += [f"  f is the join-tail share of a steal block, from the uniondispatch A/B's {p['ud_delta_512']:.2f} ms at"
            f" P 512; X_u {U['x'] * 1000:.3f} us per host slot is the named residual, anchor {U['anchor']}"]
    if TERMS:
        out += ["", "named terms of the red rows:"] + [f"  {k}: {v}" for k, v in TERMS.items()]
    out += ["", "diagnostics [derived; the constants stay as recorded]:"]
    out += ["  " + d for d in diagnostics(p)]
    out.append(f"tolerances: abs n=1 {tol_abs(p, 1):.2f} %, n=2 {tol_abs(p, 2):.2f} %, n=3 {tol_abs(p, 3):.2f} %;"
               f" ratio n=1 none (one round: unscored), n=2 {tol_ratio(p, 2):.2f} %, n=3 {tol_ratio(p, 3):.2f} %"
               f" (n=4 {tol_ratio(p, 4):.2f}, n=6 {tol_ratio(p, 6):.2f})")
    ns = sum(1 for r in ROWS if scored(r[0], r[7], r[4], skip, r[6]))
    unnamed = [r for r in red if r not in BLAME]
    out.append(f"scored rows {ns}, ok {ns - len(red)}, red {len(red)} (named {len(red) - len(unnamed)},"
               f" unnamed {len(unnamed)}{': ' + ', '.join(unnamed) if unnamed else ''})")
    if verbose:
        print("\n".join(out))
    return unnamed


def fit(f, target, lo, hi):
    """The x in [lo, hi] where the monotone f(x) crosses target (bisection); a diagnostic, never a
    constant the model then reads."""
    up = f(hi) > f(lo)
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if (f(mid) < target) == up:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# the trace's per-layer host wait (h-wait, ms; under the profiler) at P 512, hot list 384, afe86d5
NSYS_WAIT = {0: 16.30, 1: 16.31, 2: 9.89, 3: 11.83, 14: 9.80, 20: 10.17, 36: 14.52, 37: 11.16, 38: 5.90, 39: 0.62}
# the trace's layer-2 projections per chunk (us): an indexer layer (q_b 36,864 rows) and qkv's 228 blocks
NSYS_L2 = dict(qkv=29.62, q_b=82.90, wo_a=105.32, wo_b=123.67)


def diagnostics(p):
    out = []
    # the grid law on grids it was not read from
    l2 = dict(qkv=launch_us(p, "qkv", 1824), q_b=launch_us(p, "q_b", QB_ROWS_IDX), wo_a=launch_us(p, "wo_a", 8192),
              wo_b=launch_us(p, "wo_b", 5120))
    out.append("grid law at layer 2 (qkv 228 blocks, q_b 4,608), us: " + ", ".join(
        f"{k} {l2[k]:.1f}/{NSYS_L2[k]} ({(l2[k] - NSYS_L2[k]) / NSYS_L2[k] * 100:+.1f} %)" for k in NSYS_L2))
    r = evaluate(p, CONFIGS["nsys"], 512)
    served = [x for x in r["recs"] if x["lb"].T]
    tm = {}
    for x in served:
        for k, v in x["terms"].items():
            tm[k] = tm.get(k, 0.0) + v / len(served)
    out.append(f"the nsys sitting's 40-lb means, model/trace (ms): projections {tm['proj']:.2f}/19.641, attention"
               f" {tm['attn']:.2f}/4.162, small {tm['small']:.2f}/3.883, source+indexer {tm['special']:.3f}/0.325, engram"
               f" {tm['engram']:.2f}/1.137 (+gaps), route window {r['agg']['card_out']:.2f}/30.29, shadow"
               f" {r['agg']['card_in']:.2f}/21.646 (+gaps), A_r {r['agg']['acts_r']:.0f}/1,394, ev {r['agg']['ev_r']:.0f}/241,"
               f" A_s {r['agg']['acts_s']:.0f}/476")
    waits = {x["lb"].l: x["wait"] for x in served}
    out.append("the queue per layer, model/trace host wait (ms, the trace under the profiler): " + ", ".join(
        f"L{l} {waits[l]:.2f}/{w}" for l, w in NSYS_WAIT.items()))
    out.append(f"  wait/enqueue at the 40-lb mean: {r['agg']['wait']:.2f}/{r['agg']['enqueue']:.2f} against the"
               f" trace's stat line 11.76/19.34 (profiled)")
    # the union's causes
    U = union_anchor(p, DEFAULT_ANCHORS)
    for cname in ("UDbase", "UD"):
        a = evaluate(p, CONFIGS[cname], 512)
        parts = {}
        for x in a["recs"]:
            for k, v in x["parts"].items():
                parts[k] = parts.get(k, 0.0) + v / a["n_lb"]
        out.append(f"union causes, {cname} P 512 (ms a layer-batch): kernel {a['agg']['union_raw']:.2f}, "
                   + ", ".join(f"{k} {v:.2f}" for k, v in parts.items()) + f", X_u {a['agg']['union_x']:.2f}")
    # the prose prompt's shadow: routing, not the per-slot cost
    prow = next(x for x in ROWS if x[0] == "B1.prose.card_in")
    per_slot, per_tile = predict_row(p, prow)[0], predict_row(p, prow, dict(DEFAULT_ANCHORS, shadow="tile"))[0]
    pin = evaluate(p, dict(CONFIGS["B1prose"], _clk=p["clk_cardbound"]), 512, "prose-in")["agg"]["card_in"]
    out.append(f"prose card_in 45.28 measured: per slot {per_slot:.2f} with the prompt's routing (63,203 card slots), per"
               f" (expert, tile) {per_tile:.2f}; per slot with the trace's own top 70 on all 40 layers (prose-in, the"
               f" 53.0 the round assumed) {pin:.2f}")
    b1 = evaluate(p, CONFIGS["B1"], 512)["agg"]
    out.append(f"the route window (B1 lcg P 512): enqueue + wait {b1['enqueue'] + b1['wait']:.2f} against card_out"
               f" {b1['card_out']:.2f} (measured 20.37 / 19.61): the host issues while the card runs the route")
    s15 = evaluate(p, dict(CONFIGS["S15"], arm="slot"), 384)["agg"]
    tail = U["f"] / p["steal_blocks"] * s15["union_raw"]
    out.append(f"union-T: S15's clean union 55.44 ms at P 384 leaves {55.44 - s15['union_raw']:.2f} ms over the kernel sum"
               f" {s15['union_raw']:.2f}; the join tails alone at f {U['f']:.2f} are {tail:.2f} and the chunk flow's causes"
               f" {s15['union_expl']:.2f}, X_u {s15['union_x']:.2f} (predicted {s15['union']:.2f})")
    # pre-bulk router
    rt = []
    for P, pre, post in ((512, 42.3, 32.3), (4096, 45.4, 35.1)):
        base = evaluate(p, CONFIGS["S14"], P)["agg"]["nonunion"]
        v = fit(lambda x: evaluate(p.but(router_tok=x), CONFIGS["S14pre"], P)["agg"]["nonunion"] - base, pre - post,
                5.0, 40.0)
        rt.append(f"P {P} {v:.1f} us")
    out.append("router_tok: the one-token router launch S14's pre - post non-union difference (10.0, 10.3 ms) implies: "
               + ", ".join(rt) + f" (constant {p['router_tok']:.1f}, derived; 21.56 measured before ds41router)")
    rtp = p.but(router_tok=21.0)
    closes = []
    for rid in ("S13.pp4096.on", "S13.pp4096.off", "S13b.nonunion.on", "S14.ratio512.expert_pre", "S14.ratio4096.expert_pre",
                "S14.p512.pre.nonunion", "S14.p4096.pre.nonunion"):
        row = next(x for x in ROWS if x[0] == rid)
        v, _ = predict_row(rtp, row)
        closes.append(f"{rid} {(v - row[5]) / abs(row[5]) * 100:+.1f} %")
    out.append("  with router_tok 21.0 those rows would read: " + ", ".join(closes))
    s4 = [(q, evaluate(p, CONFIGS["S14"], 4096)["agg"][q], m) for q, m in (("union", 73.9), ("route", 32.9),
                                                                         ("nonunion", 35.1))]
    out.append("P dependence: S14 at P 4096 reads " + ", ".join(f"{q} {v:.2f}/{m} ({(v - m) / m * 100:+.1f} %)"
                                                                  for q, v, m in s4)
               + f"; CED off P 4096/512 {evaluate(p, dict(CONFIGS['S13'], ced=False), 4096)['pp'] / evaluate(p, dict(CONFIGS['S13'], ced=False), 512)['pp']:.3f}"
               " against S13's 102.56/104.66 = 0.980")
    return out


# ============================================================================ the ladder

STEPS = [
    # (name, overrides on "now", what it is, decisions (low, in band, high))
    ("now", {}, "main 9626c7f: the five-dispatch union (uniondispatch) on sitting 15's card route, hot list 384", None),
    ("B1", {"b1": True}, "the four attention projections per 128-token sub-block (ds41proj, landed 2f45a79)", None),
    ("B1+T", {"b1": True, "tile": True},
     "+ cardtile: the grouped shadow as (card expert, row tile, 8-column tile) items, bit for bit"
     " (cardnext-design-report.md section 2.3)", None),
    ("B1+T+G", {"b1": True, "tile": True, "G": 2, "wrap": True},
     "+ prefillgroup: the layer-first scheduler with the cross-layer wrap over pairs of batches (section 1.1)", None),
    ("+stream", {"b1": True, "tile": True, "G": "auto", "stream": True, "force_borrow": True},
     "+ host streaming through a 128-slot ring borrowed from cold card experts for the prompt (stock G <= 8, no wrap)",
     None),
    ("+h3tile-b", {"b1": True, "tile": True, "G": "auto", "stream": True, "force_borrow": True, "union": "r8"},
     "+ the r8 host tile alone (h3tile-b-design-report.md:236-256; R1 landed as uniondispatch 9626c7f and is in now)", None),
    ("+B4", {"b4": True, "tile": True, "G": "auto", "stream": True, "force_borrow": True, "union": "r8"},
     "+ int8 GEMM projections (B4; the prefill = step bits stay on the off arm)", None),
]
COMPARE = [
    ("B1+G", {"b1": True, "G": 2, "wrap": True}, "G before T: the wrap scheduler over pairs of batches on B1 alone"),
    ("B1+T+G8", {"b1": True, "tile": True, "G": "auto", "wrap": True}, "the wrap over up to 8 batches instead of 2"),
    ("B1+T+G stock", {"b1": True, "tile": True, "G": 2}, "G 2 in the stock layer-first order (no wrap)"),
    ("B1+IMMA", {"b1": True, "imma": True}, "the IMMA grouped GEMM in T's place, G 1 (float sum order moves)"),
    ("B1+IMMA+G", {"b1": True, "imma": True, "G": 2, "wrap": True}, "the IMMA grouped GEMM in T's place, G 2 wrap"),
    ("G without B1", {"G": "auto"}, "the G scheduler with one host thread and no B1"),
    ("+stream R8", {"b1": True, "tile": True, "G": "auto", "stream": True},
     "the triage rule instead: no borrowing below P 1024, the unborrowed 8-slot ring"),
    ("h3tile-b on now", {"union": "r8"}, "the r8 host tile alone on today's flow (no B1, G 1)"),
]
MEASURED = {
    "B1": "measured 09-26#b1-pp-ab on the lcg prompt without the hot list: 144.6 / 201.0 (the B1 rows; the model"
          " reads 147.8 / 208.2 there), and on the prose prompt with it: 171.6 at P 512, where the card was busy 0.93"
          " of the wall and its route ran clk_cardbound slower (175.8 with the ratio, 183.8 without)",
}
DECISIONS = {
    "now": ("today's flow is off the model (the route or the union moved): re-sit one arm with BLOOMERY_STEP_STATS=1"
            " before a step reads from it",
            "the ladder starts here; on the prose prompt the card already binds most layers (the shadow's grouped"
            " kernels, 45.3 ms a layer-batch measured after B1 against a 36.4 ms union): for a prose prompt the next"
            " lever is the grouped shadow, not the host",
            "same as low"),
    "B1+T": ("the prose P 512 wall still carries the shadow: read card_in (predicted 17.8 [15.5-19.9] ms a layer-batch"
             " at lcg's clock); above ~22 the tile items run below e 0.45 (registers, occupancy) — ptxas -v and ncu the GT kernel, split"
             " gate and up (cardnext 2.3) before G",
             "build prefillgroup (G): after T the prose P 4096 card (40 ms a layer-batch) and host (42) balance and"
             " only the wrap overlaps them; lcg moves 0 at both P (the shadow was already under the union)",
             "the route ran faster than the card-bound clock allows (card duty fell to ~0.66): G at P 4096 next, the"
             " IMMA shadow is worth 0 at G 1"),
    "B1+T+G": ("wait_lb near the route (the exchange set shared: F4) or enqueue_lb above ~8 (the queue): fix the"
               " scheduler before anything else",
               "lcg P 4096 is host-bound (the union 63 of a 70 ms layer-batch): h3tile-b or streaming next; prose P 4096"
               " is card-bound on most layers (route 21 + shadow 18 + posts): the route (B4) and the per-chunk shadow"
               " (7 ms) are the card's next terms, the IMMA shadow +4 %",
               "the union runs faster overlapped than alone: re-derive k(T) before streaming"),
    "+stream": ("the fill takes more DRAM than 3 x 26.3 GB/s under a 126 GB/s ceiling (fill_crossings 4 costs 16 at P 4096):"
                " the 5-minute DRAM probe before building, then a smaller k(T)",
                "the card binds 26 of 40 layers at P 4096 (route 24 + shadow 21 + stream GEMMs): a card lever (the grouped"
                " shadow, B4) is worth more there than h3tile-b (+5.5 %); at P 512 the host still binds",
                "the card terms are smaller than derived (stream_gfix, stream_gcol): calibrate them, then h3tile-b"),
    "+h3tile-b": ("the r8 tile's bench gain did not survive the five-dispatch flow: time it alone on today's flow first"
                  " (164 / 231 predicted, the comparison row 'h3tile-b on now')",
                  "at P 4096 the card binds: B4 or the grouped shadow next; at P 512 the route before the union is the"
                  " serial term",
                  "the host sits at the W floor at P 512: only card levers remain"),
    "+B4": ("the int8 projection GEMM runs under the MoE rate: the Q3_K GEMM bench arm before building",
            "host and card balance at P 4096 (8.1 / 8.2 s busy): the DRAM/PCIe terms and the grouped shadow are next",
            "the stream GEMMs and the shadow are cheaper than derived: a bigger ring or G"),
}

VARY = [k for k, c in C.items() if c.lo != c.hi and k not in ("between_lease", "sd_same_binary")]
# The grouped shadow per (expert, tile) was a fourth alternative until the prose prompt's card_in: per slot
# it reads 45.6 against 45.28 measured, per tile 39.3 (-13 %); on lcg the two sit within 3 % of each
# other, so only the prose row separates them (diagnostics print both).
STRUCT = (("resid_mode=prop", dict(DEFAULT_ANCHORS, resid_mode="prop")),
          ("union anchor=S14 slot", dict(DEFAULT_ANCHORS, union="anchor_union_s14s")),
          ("union anchor=UD new 512", dict(DEFAULT_ANCHORS, union="anchor_union_ud512")))


def step_cfg(over):
    return dict(CONFIGS["now"], **over)


@lru_cache(maxsize=None)
def duty_ends():
    """The card duty (busy / wall) of the two rows clk_cardbound was read between, at the central constants
    and the lcg clock: B1 lcg P 512 (no slowdown) and the B1 prose prompt (clk_cardbound)."""
    p = central()
    c1, r1 = row_cfg(p, "B1", {})
    c2, r2 = row_cfg(p, "B1prose", {"rt": "prose"})
    return evaluate(p, c1, 512, r1)["duty"], evaluate(p, c2, 512, r2)["duty"]


def clk_at(p, duty):
    """The card-bound clock ratio at a run's card duty: 1 at B1 lcg's duty, clk_cardbound at the prose
    prompt's, linear between (the shape is assumed; the ends are measured)."""
    lo, hi = duty_ends()
    w = min(1.0, max(0.0, (duty - lo) / (hi - lo)))
    return 1.0 + (p["clk_cardbound"] - 1.0) * w


def band_of(cfg, P, rname):
    """Central pp with k searched per layer; one-at-a-time lo/hi of every varying constant the run reads
    and the structural alternatives (k held, f and X_u re-derived at every point); the band is
    central -/+ their quadrature, the corners every constant at its adverse/favourable end."""
    log = set()
    pc = central(log)
    c0 = evaluate(pc, cfg, P, rname)
    p = central()
    kfix = c0["ks"] if c0["cfg"]["ring"] else None
    base = c0["pp"]
    devs = []
    for name in VARY:
        if name not in log:
            continue
        d = [evaluate(p.but(**{name: v}), cfg, P, rname, None, kfix)["pp"] - base for v in (C[name].lo, C[name].hi)]
        devs.append((name, d[0], d[1]))
    for label, anc in STRUCT:
        d = evaluate(p, cfg, P, rname, anc, kfix)["pp"] - base
        devs.append((label, d, d))
    clk = clk_at(p, c0["duty"])
    if clk > 1.0 + 1e-9:
        d = evaluate(p, dict(cfg, _clk=clk), P, rname, None, kfix)["pp"] - base
        devs.append((f"card-bound clock x{clk:.3f}", d, d))
    down = math.sqrt(sum(min(0.0, a, b) ** 2 for _, a, b in devs))
    up = math.sqrt(sum(max(0.0, a, b) ** 2 for _, a, b in devs))
    lo_set, hi_set = {}, {}
    for name, a, b in devs:
        if name in C:
            lo_set[name], hi_set[name] = (C[name].lo, C[name].hi) if a <= b else (C[name].hi, C[name].lo)
    worst = evaluate(p.but(**lo_set), cfg, P, rname, None, kfix)["pp"]
    best = evaluate(p.but(**hi_set), cfg, P, rname, None, kfix)["pp"]
    devs.sort(key=lambda x: -max(abs(x[1]), abs(x[2])))
    return dict(pp=base, run=c0, lo=base - down, hi=base + up, worst=worst, best=best, devs=devs, read=sorted(log))


def bind_summary(res):
    out = {}
    for b in res["binds"]:
        out[b] = out.get(b, 0) + 1
    return " ".join(f"{k} {v}" for k, v in sorted(out.items()))


def predict(which):
    todo = [s for s in STEPS if which in ("all", s[0])]
    comp = COMPARE if which == "all" else [c for c in COMPARE if c[0] == which]
    if not todo and not comp:
        raise SystemExit(f"--predict: no step {which!r}; steps: all, " + ", ".join(s[0] for s in STEPS))
    print("pp tok/s [derived] @ A6000 300 W, plan (a), hot list 384, CED on, CARD_EXPERTS expert, the lcg prompt or the"
          " prose prompt (corpus-prose.ids; routing calibrated on its B1 row), from main 9626c7f.")
    print("band = central -/+ quadrature of every read constant's lo/hi and three structural alternatives (resid_mode prop,"
          " X_u anchored on the S14 slot arm or on the uniondispatch lease's new arm (today's code, no hot list: it sits 2 %"
          " under the model)) and, where the card is busier than"
          " on B1 lcg, the card-bound clock at the cell's duty (clk_cardbound); corner = all constants at their adverse /"
          " favourable ends together; k per layer is searched at the central constants and held across the band.")
    prev = {}
    for name, over, what, _ in todo:
        cfg = step_cfg(over)
        print(f"\n== {name}: {what}")
        for rname in ("lcg", "prose"):
            for P in (512, 4096):
                b = band_of(cfg, P, rname)
                r = b["run"]
                a = r["agg"]
                d0 = prev.get((rname, P))
                gain = f"  ({(b['pp'] / d0 - 1) * 100:+.1f} % on the step before)" if d0 else ""
                print(f"  {rname:8} P {P:4d}: {b['pp']:7.1f}  band {b['lo']:7.1f}-{b['hi']:7.1f}  corner"
                      f" {b['worst']:7.1f}-{b['best']:7.1f}{gain}")
                ring = r["cfg"]["ring"]
                extra = ""
                if ring:
                    ks = list(r["ks"].values())
                    extra = (f"; ring {ring}{' borrowed, re-upload ' + format(r['reupload'], '.0f') + ' ms' if r['cfg']['borrow'] else ''},"
                             f" k layer 0 {r['ks'].get((0, 0), 0):.0f}, mean {sum(ks) / len(ks):.0f}")
                print(f"      G {r['cfg']['G']}{extra}; per lb: union {a['union']:.1f} (DRAM +{a['union_dram']:.1f}),"
                      f" wait {a['wait']:.1f}, enqueue {a['enqueue']:.1f}, route card {a['card_out']:.1f}, shadow"
                      f" {a['card_in']:.1f}, A_r {a['acts_r']:.0f} + ev {a['ev_r']:.0f} + A_s {a['acts_s']:.0f};"
                      f" layers bound: {bind_summary(r)}")
                bz = r["busy"]
                print(f"      busy over the prompt (s): host {bz['host'] / 1e3:.2f}, card {bz['card'] / 1e3:.2f}, PCIe"
                      f" {bz['pcie'] / 1e3:.2f}, DRAM {bz['dram'] / 1e3:.2f}; wall {r['wall'] / 1e3:.2f} ="
                      f" {r['wall'] / max(bz.values()):.2f} x the busiest")
                top = ", ".join(f"{n} {lo:+.1f}/{hi:+.1f}" for n, lo, hi in b["devs"][:4])
                print(f"      depends most on (pp at lo/hi): {top}")
                if r["cfg"]["borrow"]:
                    nv = evaluate(central().but(borrow_reupload=C["borrow_reupload_nvme"].value), cfg, P, rname, None,
                                  r["ks"])["pp"]
                    print(f"      with the re-upload read back from NVMe (sources dropped by DONTNEED): {nv:.1f}")
                prev[(rname, P)] = b["pp"]
        if name == "now":
            print("  variants of today's flow (central; CED off = BLOOMERY_CED=off, slot = BLOOMERY_CARD_EXPERTS=slot,"
                  " no hot list = the uniondispatch lease's condition, measured 129.63 / 180.02):")
            for label, vo in (("CED off", {"ced": False}), ("slot arm", {"arm": "slot"}), ("no hot list", {"hot": False})):
                vals = [f"{rn} P {P} {evaluate(central(), dict(cfg, **vo), P, rn)['pp']:.1f}"
                        for rn in (("lcg",) if label == "no hot list" else ("lcg", "prose")) for P in (512, 4096)]
                print(f"      {label:11}: " + ", ".join(vals))
        if name in MEASURED:
            print(f"  {MEASURED[name]}")
        dec = DECISIONS.get(name)
        if dec:
            for label, text in zip(("low", "in band", "high"), dec):
                print(f"  if {label}: {text}")
    for name, over, what in comp:
        cfg = step_cfg(over)
        print(f"\n-- comparison, {name}: {what}")
        for rname in ("lcg", "prose"):
            for P in (512, 4096):
                r = evaluate(central(), cfg, P, rname)
                a = r["agg"]
                print(f"  {rname:8} P {P:4d}: {r['pp']:7.1f}  G {r['cfg']['G']} ring {r['cfg']['ring']}; per lb union"
                      f" {a['union']:.1f}, wait {a['wait']:.1f}, enqueue {a['enqueue']:.1f}, route card {a['card_out']:.1f},"
                      f" shadow {a['card_in']:.1f}; layers bound: {bind_summary(r)}")


# ============================================================================ the cells of the next two levers

CELL_STEPS = (("B1", {"b1": True}), ("B1+T", {"b1": True, "tile": True}), ("B1+G", {"b1": True, "G": 2, "wrap": True}),
              ("B1+T+G", {"b1": True, "tile": True, "G": 2, "wrap": True}))
CELL_MEASURED = {("lcg", 512): "144.6 (no hot list)", ("lcg", 4096): "201.0 (no hot list)", ("prose", 512): "171.6",
                 ("prose", 4096): "-"}


def imma_over_t(cfg_t, cfg_i, P, rname):
    """pp(IMMA shadow) / pp(T shadow) - 1 at the central constants, and its range over imma_lb and t_tile_T at
    their ends and the card-bound clock at each arm's duty."""
    p = central()
    out = []
    for il in (C["imma_lb"].lo, C["imma_lb"].value, C["imma_lb"].hi):
        for tt in (C["t_tile_T"].lo, C["t_tile_T"].value, C["t_tile_T"].hi):
            for clk in (False, True):
                q = p.but(imma_lb=il, t_tile_T=tt)
                rs = []
                for cfg in (cfg_t, cfg_i):
                    r = evaluate(q, cfg, P, rname)
                    if clk:
                        r = evaluate(q, dict(cfg, _clk=clk_at(q, r["duty"])), P, rname)
                    rs.append(r["pp"])
                out.append(rs[1] / rs[0] - 1.0)
    c = evaluate(p, cfg_i, P, rname)["pp"] / evaluate(p, cfg_t, P, rname)["pp"] - 1.0
    return c, min(out), max(out)


def cells():
    """The next two levers per cell (lcg or prose, P 512 or 4096), with bands, and what the IMMA shadow is worth
    over T's at G 1 and at G 2 [derived]."""
    print("pp tok/s [derived] @ A6000 300 W, plan (a), hot list 384, CED on; bands as --predict. B1 measured: lcg"
          " without the hot list (09-26#b1-pp-ab), prose with it.")
    print(f"{'cell':12} {'B1 meas':>18} {'B1':>22} {'B1+T':>22} {'B1+G':>22} {'B1+T+G':>22} {'IMMA/T G1':>18} {'IMMA/T G2':>18}")
    for rname in ("lcg", "prose"):
        for P in (512, 4096):
            cols = []
            for name, over in CELL_STEPS:
                b = band_of(step_cfg(over), P, rname)
                cols.append(f"{b['pp']:6.1f} [{b['lo']:5.1f}-{b['hi']:5.1f}]")
            g1 = imma_over_t(step_cfg({"b1": True, "tile": True}), step_cfg({"b1": True, "imma": True}), P, rname)
            g2 = imma_over_t(step_cfg({"b1": True, "tile": True, "G": 2, "wrap": True}),
                             step_cfg({"b1": True, "imma": True, "G": 2, "wrap": True}), P, rname)
            im = [f"{c * 100:+5.1f} [{lo * 100:+4.1f},{hi * 100:+4.1f}]" for c, lo, hi in (g1, g2)]
            print(f"{rname + ' ' + str(P):12} {CELL_MEASURED[(rname, P)]:>18} " + " ".join(f"{c:>22}" for c in cols)
                  + " " + " ".join(f"{c:>18}" for c in im))
    print("\nper served layer-batch (ms, central): the host thread's issue, wait and union; the card's route and shadow;"
          " busy = the union and the issue calls (host), route + shadow + posts (card); the layers' binding resource")
    for rname in ("lcg", "prose"):
        for P in (512, 4096):
            for name, over in CELL_STEPS:
                r = evaluate(central(), step_cfg(over), P, rname)
                a, n = r["agg"], r["n_lb"]
                print(f"  {rname:5} P {P:4d} {name:7}: issue {a['enqueue']:5.1f}, wait {a['wait']:5.1f}, union {a['union']:5.1f} |"
                      f" route {a['card_out']:5.1f}, shadow {a['card_in']:5.1f} | busy host {r['busy']['host'] / n:5.1f},"
                      f" card {r['busy']['card'] / n:5.1f}, wall {r['wall'] / n:5.1f} (duty {r['duty']:.2f}); layers: {bind_summary(r)}")


# ============================================================================ explain

def b1_table(p, l, T, label):
    """B1's four launches at layer l over T columns, one row a kernel [derived]: the geometry of the first
    sub-block, the time summed over the sub-blocks; beside them the chunk loop's same work (T / 8
    launches at m = 8)."""
    sbs = sub_blocks(full_chunks(T))
    g = len(sbs[0])
    rows = [f"  B1 at layer {l}, T {T} ({len(sbs)} sub-blocks, {g} column groups in the first), {label}; us [derived]:",
            f"    {'kernel':5} {'rows x K':>12} {'regs':>4} {'r':>2} {'blocks':>7} {'waves':>7} {'it':>3} {'L':>5}"
            f" {'issue':>5} {'LSU':>5} {'L1TEX':>5} {'DRAM':>5} {'step':>5} {'B1 us':>8} {'loop us':>8}"]
    tb = tl = 0.0
    for name, nrows in (("qkv", 1792), ("q_b", qb_rows(l)), ("wo_a", 8192), ("wo_b", 5120)):
        d = {}
        launch_us(p, name, nrows, 8, g, True, d)
        t = sum(launch_us(p, name, nrows, 8, len(sb), True) for sb in sbs)
        loop = len(full_chunks(T)) * launch_us(p, name, nrows, 8)
        tb += t
        tl += loop
        regs = 80 if name == "wo_a" else 40
        dm = d["dem"]
        rows.append(f"    {name:5} {nrows:>6}x{KERN[name][1]:<5} {regs:>4} {d['r']:>2} {d['blocks']:>7} {d['waves']:7.2f}"
                    f" {d['iters']:>3} {d['L']:5.2f} {dm['iss']:5.2f} {dm['lsu']:5.2f} {dm['l1']:5.2f} {dm['dram']:5.2f}"
                    f" {d['step_f']:5.2f} {t:8.0f} {loop:8.0f}")
    rows.append(f"    sum {tb:.0f} us against the loop's {tl:.0f} ({(tb - tl) / 1000:+.2f} ms)")
    return rows


def explain(target):
    log = set()
    p = central(log)
    row = next((r for r in ROWS if r[0] == target), None)
    step = next((s for s in STEPS + [c + (None,) for c in COMPARE] if s[0] == target), None)
    if row:
        rid, cname, P, over, q, meas, n, kind, src = row
        over = {k: v for k, v in over.items() if k not in ("cold", "vs")}
        cfg, rname = row_cfg(p, cname, over)
        print(f"{rid}: {q} measured {meas} ({src}); {cname} {cfg}, P {P}, {rname}")
        runs = [(rname, P, cfg)]
        for key in BLAME.get(rid, "").split(", "):
            if key:
                print(f"  red, named: {key}: {TERMS[key]}")
    elif step:
        cfg = step_cfg(step[1])
        print(f"{step[0]}: {step[2]}; {cfg}")
        runs = [(rn, P, cfg) for rn in ("lcg", "prose") for P in (512, 4096)]
    else:
        raise SystemExit(f"--explain: {target!r} is neither a backtest row nor a ladder step")
    for rname, P, cfg in runs:
        r = evaluate(p, cfg, P, rname)
        a = r["agg"]
        served = [x for x in r["recs"] if x["lb"].T]
        n = len(served)
        tsum = {}
        for x in served:
            for k, v in x["terms"].items():
                tsum[k] = tsum.get(k, 0.0) + v / n
        print(f"\n  {rname}, P {P}: wall {r['wall']:.1f} ms, pp {r['pp']:.2f}; {n} served layer-batches, "
              f"{block_positions(P, cfg.get('ced', True))} block positions, G {r['cfg']['G']}, ring {r['cfg']['ring']}")
        print("  route card terms per served layer-batch (ms): "
              + ", ".join(f"{k} {v:.3f}" for k, v in tsum.items()) + f" = {a['card_out']:.3f}")
        tc = a["card_out"] / a["acts_r"] * 1000 if a["acts_r"] else 0.0
        e, w = queue_split(a["acts_r"], a["acts_s"], a["ev_r"], a["card_out"], a["card_in"], p["q_act"],
                           p["t_issue"] / 1000.0)
        print(f"  queue: A_r {a['acts_r']:.0f} + ev {a['ev_r']:.0f}, A_s {a['acts_s']:.0f}, Q {p['q_act']:.0f} activities,"
              f" t_c {tc:.2f} us an activity, t_i {p['t_issue']:.2f} us a call; closed form at the means: enqueue {e:.2f},"
              f" wait {w:.2f}; per-layer simulation: enqueue {a['enqueue']:.2f}, wait {a['wait']:.2f}")
        parts = {}
        for x in served:
            for k, v in x["parts"].items():
                parts[k] = parts.get(k, 0.0) + v / n
        print(f"  union a layer-batch: kernel {a['union_raw']:.2f} + causes {a['union_expl']:.2f} ("
              + ", ".join(f"{k} {v:.2f}" for k, v in parts.items()) + f") + X_u {a['union_x']:.2f} + DRAM"
              f" {a['union_dram']:.2f} = {a['union']:.2f} ms; {a['slots']:.0f} host slots, {a['touched']:.1f} experts"
              f" touched; f {r['U']['f']:.3f}, X_u {r['U']['mode']} {r['U']['x']:.5f}")
        print(f"  shadow {a['card_in']:.2f}, copy {a['copy']:.2f}, non-union {a['nonunion']:.2f} ms a layer-batch;"
              f" prologue {a['prologue']:.1f} ms a prompt; layers bound: {bind_summary(r)}")
        lb = next((x["lb"] for x in served if len(x["lb"].full) >= 8 and x["lb"].l == 3),
                  next((x["lb"] for x in served if len(x["lb"].full) >= 8), served[0]["lb"]))
        if cfg.get("b1") or cfg.get("b4"):
            for line in b1_table(p, 3, 512, "a normal layer"):
                print(line)
            Tm = round(sum(x["lb"].T for x in served) / n)
            for line in b1_table(p, 3, Tm, "the served mean T"):
                print(line)
        pos, m = lb.full[len(lb.full) // 2]
        print(f"  one chunk of layer {lb.l} at position {pos} (m {m}), us [derived]:")
        for name, rows in (("qkv", 1792), ("q_b", qb_rows(lb.l)), ("wo_a", 8192), ("wo_b", 5120)):
            d = {}
            t = launch_us(p, name, rows, m, 1, False, d)
            dm = d["dem"]
            print(f"    {name:5} {rows}x{KERN[name][1]}: {t:6.1f} = launch + {d['iters']} iterations x ({d['full']} full"
                  f" waves at step {d['step_f']:.2f} + {'one partial at ' + format(d['step_t'], '.2f') if d['tail'] else 'no partial'});"
                  f" L {d['L']:.2f}; demand at r {d['r']}: issue {dm['iss']:.2f} LSU {dm['lsu']:.2f} L1TEX {dm['l1']:.2f}"
                  f" DRAM {dm['dram']:.2f}")
        print(f"    seg {attn_seg(p, lb.l, pos, m):.1f} ({chunk_keys(lb.l, pos, m)[0]:.0f} keys), attn_other"
              f" {p['attn_other']:.1f}, small {p['small_chunk']:.1f} - fork overlap {p['fork_overlap']:.1f}, special"
              f" {special_chunk(p, lb.l, pos, m):.1f}, route_nq {p['route_nq']:.2f}, idle {full_acts(lb.l) * p['gap_act']:.1f}")
    print("\n  constants read:")
    for name in sorted(log):
        c = C[name]
        print(f"    {name:24} {c.value:>11g} [{c.lo:g}, {c.hi:g}] {c.unit:18} {c.kind:9} {c.source}")


# ============================================================================ uncalibrated terms

SWEEP = ("BLOOMERY_AB_ROUNDS=1 BLOOMERY_BOX_ENV='BLOOMERY_HOT_LIST=/root/bloomery-data/router/hotlist-384.txt"
         " BLOOMERY_STEP_STATS=1 BLOOMERY_CED=off BLOOMERY_LEASE_CARD=docs/cards/<slug>.card' just depth-gpu-ds41 128 256 384 512")


def uncal_rows(p):
    """(term, what the model assumes now, the cheapest runner command, expected [derived], box minutes [derived])."""
    sw = []
    for P in (128, 256, 384, 512):
        cfg = dict(CONFIGS["now"], ced=False)
        a = evaluate(p, cfg, P)["agg"]
        b = evaluate(p, cfg, P, "lcg", dict(DEFAULT_ANCHORS, resid_mode="prop"))["agg"]
        sw.append(f"P {P}: union {a['union']:.1f} (prop {b['union']:.1f}), wait {a['wait']:.1f}, enqueue {a['enqueue']:.1f},"
                  f" card_in {a['card_in']:.1f}")
    t = evaluate(p, step_cfg({"b1": True, "tile": True}), 512, "prose")["agg"]
    tlo = evaluate(p.but(t_tile_T=C["t_tile_T"].lo), step_cfg({"b1": True, "tile": True}), 512, "prose")["agg"]
    thi = evaluate(p.but(t_tile_T=C["t_tile_T"].hi), step_cfg({"b1": True, "tile": True}), 512, "prose")["agg"]
    return [
        ("the card-bound clock (clk_cardbound)", "the route and the per-chunk shadow 1.124x slower when the card is busy"
         " most of the wall (the prose prompt against lcg, same binary); central predictions leave it out, the band carries"
         " it at the cell's duty", "the SM clock sampled through one prose prompt (nvidia-smi --query-gpu=clocks.sm"
         " -lms 20 beside generate_ds41 under the lease; no runner arm yet), or the card-timing stat line of the T A/B",
         "~1,810 MHz under the prose prompt against ~2,030 on lcg (2,030 / 1.124), if the clock is the cause", "3"),
        ("t_tile_T (cardtile)", "44.97 us a (card expert, 8-column tile) item, from cardnext's lcg 5.3 ms a layer-batch",
         "the T landing A/B's stat line on the prose prompt (card_in)", f"card_in {t['card_in']:.1f} ms a layer-batch"
         f" (band {tlo['card_in']:.1f}-{thi['card_in']:.1f})", "0 on top of the T A/B"),
        ("union T-scaling (union-T)", f"X_u {union_anchor(p, DEFAULT_ANCHORS)['x'] * 1000:.2f} us a host slot on top of the"
         " causes; S15 P 384 reads below the kernel sum's slope", SWEEP, "; ".join(sw) + " (ms a layer-batch, CED off so"
         " T = P)", "5: four arms, one round"),
        ("small_bytes_tok (B1)", "780 kB a token for B1's batch-wide small kernels (HC_PRE on the branch)",
         "the nsys prefill form after B1 lands", "0.4-0.8 ms a layer-batch at T 512", "5 (warm load, one prompt)"),
        ("dram_eff, fill_crossings, fill_steal", "126 GB/s ceiling, 3 crossings of 26.3 GB/s, 4 % of the union's cores",
         "the DRAM probe: bench_v41_host union5:4x8u0.125 beside two fill threads and a 26 GB/s pinned reader"
         " (hoststream-design-report.md:144-146; the bench has no such arm yet)",
         "the fill holds >= 26.3 GB/s and the union slows 4-10 %", "5, no GPU"),
        ("stream_gcol, stream_gfix", "3.0 us a column and 28 us an expert", "just bench-gpu-kernels with a grouped arm"
         " at ~60 columns an expert (no such arm yet)", "20-40 us + 1.5-5 us a column; the card binds at G8", "3"),
        ("prose routing past 512 tokens (prose_swap, prose_phi)", "the P 512 prompt's card capture and host spread,"
         " carried to every batch of a longer prompt", "a prose prompt of 4,096 ids through time-gate.sh with"
         " BLOOMERY_STEP_STATS=1 (union_host_slots and union_lb per batch)", "union"
         f" {evaluate(p, dict(CONFIGS['B1prose'], _clk=p['clk_cardbound']), 4096, 'prose')['agg']['union']:.1f} ms a"
         " layer-batch at P 4096", "5"),
        ("router_tok, copy_gbs", "15.1 us, 15 GB/s: only the pre-ds41bulk / pre-hostserve rows read them",
         "none: both paths are gone from main", "S14 implies router_tok ~20 us", "0"),
    ]


def uncalibrated():
    p = central()
    for t in uncal_rows(p):
        print(f"* {t[0]}\n    now: {t[1]}\n    run: {t[2]}\n    expect [derived]: {t[3]}\n    box [derived]: {t[4]}")
    print("\nassumed constants (constants.tsv kind = assumed):")
    for k, c in C.items():
        if c.kind == "assumed":
            print(f"  {k:18} {c.value:g} [{c.lo:g}, {c.hi:g}] {c.unit}: {c.conditions}")


# ============================================================================ self-test

ZERO_CARD = ("gemv_launch", "gemv_lat_iter", "proj_qkv_us", "proj_qb_us", "proj_woa_us", "proj_wob_us", "attn_seg_l2",
             "attn_seg_d6", "attn_seg_d1024", "attn_other", "small_chunk", "fork_overlap", "route_nq", "src_chunk",
             "src_gemv_pair", "idx_chunk", "idx_row_ns", "part_chunk", "engram_chunk", "router_tile", "pick", "places",
             "gap_act", "small_k", "router_tok", "head", "t_issue", "small_bytes_tok", "shadow_chunk", "shadow_chunk_nocard",
             "shadow_lb", "shadow_slot", "post_batch_tok") + tuple(
    f"dmd_{k}_{u}" for k in ("qkv", "qb", "woa", "wob") for u in ("iss", "lsu", "l1"))


def self_test():
    p = central()
    fails = []

    def check(name, ok, detail=""):
        print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f": {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    bad = [k for k, c in C.items() if not (c.lo <= c.value <= c.hi)]
    check("every constant: lo <= value <= hi, a source, a kind", not bad, ", ".join(bad))
    check("CED at P 512: 19,232 of 20,480 block positions",
          (block_positions(512, True), block_positions(512, False)) == (19232, 20480),
          f"{block_positions(512, True)}, {block_positions(512, False)}")
    check("CED at P 4096: 106,400 of 163,840 (body/ced.rs:386)",
          (block_positions(4096, True), block_positions(4096, False)) == (106400, 163840),
          f"{block_positions(4096, True)}, {block_positions(4096, False)}")
    need = ced_need(0, 4096, [s for s, _ in batches(4096)], True)
    ok = all(need[39 - j] == (4096 - (136 + 128 * j), 4096 - (8 + 128 * j)) for j in range(19))
    ok = ok and need[20] == (0, 4096 - 2440) and all(need[l] == (0, 0) for l in range(20))
    check("the triangle body/ced.rs:362-387 pins (layers 39 - j, 20, 0-19)", ok)
    check("P 4096 serves 220 layer-batches (S14's 73.9 + 35.1 ms x 220 = its wall)", served_lbs(4096, True) == 220,
          str(served_lbs(4096, True)))
    check("CED slot ratio 19,232/20,480 = S13b's 90,629/96,525 (0.1 %)",
          abs(19232 / 20480 - 90629 / 96525) / (90629 / 96525) < 1e-3, f"{19232 / 20480:.5f}, {90629 / 96525:.5f}")
    rn = evaluate(p, CONFIGS["nsys"], 512)
    a = rn["agg"]
    check("the trace's queue entries at P 512 (40-lb means): A_r 1,394, ev 241, A_s 476, 1 %",
          abs(a["acts_r"] - 1394) / 1394 < 0.01 and abs(a["ev_r"] - 241) / 241 < 0.01 and abs(a["acts_s"] - 476) / 476 < 0.01,
          f"{a['acts_r']:.1f}, {a['ev_r']:.1f}, {a['acts_s']:.1f}")
    per = {x["lb"].l: x for x in rn["recs"]}
    check("a normal layer's entries = the trace's layer 3: 1,350 / 257 / 517",
          (per[3]["acts_r"], per[3]["ev_r"], per[3]["acts_s"]) == (1350, 257, 517),
          f"{per[3]['acts_r']}, {per[3]['ev_r']}, {per[3]['acts_s']}")
    check("layer 14 (engram + gated source) = the trace's 2,886 activities; layer 0 (no card expert) N_s 321",
          per[14]["acts_r"] == 2886 and per[0]["acts_s"] == 321, f"{per[14]['acts_r']}, {per[0]['acts_s']}")
    rs = evaluate(p, dict(CONFIGS["S14"], arm="slot"), 512)
    r384 = evaluate(p, CONFIGS["S15"], 384)
    # cardroute:337 counted 8 shadow entries a chunk on every layer; the trace has 5 on layers 0-1 (48 chunks each
    # at P 384, and 1 instead of 5 a layer-batch): 371 - 2 x (3 x 48 + 4) / 40 = 363.6
    check("P 384 calls ~ 1,243 route (cardroute:337, calls = activities + events), 363.6 shadow, 1 %",
          abs(r384["agg"]["acts_r"] + r384["agg"]["ev_r"] - 1243) / 1243 < 0.01
          and abs(r384["agg"]["acts_s"] - 363.6) / 363.6 < 0.01,
          f"{r384['agg']['acts_r'] + r384['agg']['ev_r']:.1f}, {r384['agg']['acts_s']:.1f}")
    rb1 = evaluate(p, step_cfg({"b1": True}), 512)
    ab1 = rb1["agg"]["acts_r"] + rb1["agg"]["acts_s"]
    check("B1's burst is under Q (activities)", ab1 < p["q_act"], f"{ab1:.0f} < {p['q_act']:.0f}")
    lay3 = proj_chunk(p, 3, 8)
    check("the grid law reproduces the trace's layer-3 chunk (318.2 us: 28.3 + 70.8 + 100.0 + 119.1)",
          abs(lay3 - 318.2) < 0.05, f"{lay3:.2f}")
    check("full_res_lat +1 moves no chunk-loop projection (wo_b's two waves sum to the same)",
          abs(proj_chunk(p.but(full_res_lat=1.0), 3, 8) - lay3) < 1e-6)
    Nr, Ns, E, Cr, Cs, t_i = 1350.0, 517.0, 257.0, 29.36, 10.0, 0.0032
    for Q in (1000.0, 1068.0, 1500.0):
        e, w = queue_split(Nr, Ns, E, Cr, Cs, Q, t_i)
        if Nr + Ns > Q > Ns and e > (Nr + Ns + E) * t_i:
            check(f"queue, Q {Q:.0f}: enqueue + wait = C_r; wait = (Q - A_s) t_c; enqueue = (A - Q) t_c",
                  abs(e + w - Cr) < 1e-9 and abs(w - (Q - Ns) * Cr / Nr) < 1e-9 and abs(e - (Nr + Ns - Q) * Cr / Nr) < 1e-9)
        else:
            check(f"queue, Q {Q:.0f}: unblocked, enqueue = calls x t_i", abs(e - (Nr + Ns + E) * t_i) < 1e-12)
    ew, ww = queue_split(Nr, Ns, E, Cr, Cs, 1068.0, t_i)
    check("queue at the trace's constants: wait (1,068 - 517) x 21.75 us = 11.98 ms (trace 12.12 at its t_c)",
          abs(ww - 551 * Cr / Nr) < 1e-9, f"{ww:.2f}")
    st = State()
    ee, placed = issue(p, st, [(Nr, E, Cr), (Ns, 0, Cs)], 1068.0)
    wsim = max(0.0, placed[0][1] - st.t)
    ce, cw = queue_split(Nr, Ns, E, Cr, Cs, 1068.0, p["t_issue"] / 1000.0)
    check("simulated burst = closed form within one activity", abs(ee - ce) < Cr / Nr + 1e-6 and abs(wsim - cw) < Cr / Nr + 1e-6,
          f"enqueue {ee:.3f}/{ce:.3f}, wait {wsim:.3f}/{cw:.3f}")
    r = evaluate(p, CONFIGS["S14"], 512)
    tot = sum(x["enqueue"] + x["wait"] + x["copy"] + x["union"] for x in r["recs"] if x["lb"].T)
    rest = r["wall"] - tot - r["agg"]["prologue"]
    check("G1: wall = prologue + sum(enqueue + wait + copy + union) + posts + head (< 1 %)",
          0 <= rest < 0.01 * r["wall"], f"remainder {rest:.2f} of {r['wall']:.0f} ms")
    g8 = step_cfg({"b1": True, "G": "auto"})
    cz = p.but(w=0.0, a_union=0.0, c_union=0.0, t_issue=0.0, prologue_tok=0.0)
    z1 = run_prompt(cz, g8, 4096, routing_lcg(cz), dict(off=True))
    card = sum(x["card_out"] + x["card_in"] + (post_card(cz, x["lb"]) if x["lb"].T else 0.0) for x in z1["recs"]) + cz["head"]
    check("no host cost: the G8 wall is the card's sum", abs(z1["wall"] - card) / card < 0.01, f"{z1['wall']:.1f} vs {card:.1f}")
    hz = p.but(bw_card=1e12, pcie_pinned=1e12, d2h_gbs=1e12, post_h2d_gbs=1e12, **{k: 0.0 for k in ZERO_CARD})
    U = union_anchor(p, DEFAULT_ANCHORS)
    z2 = run_prompt(hz, g8, 4096, routing_lcg(hz), U)
    host = sum(x["union"] for x in z2["recs"]) + sum(u for _, u, _ in prompt_plan(4096, True)) * hz["prologue_tok"] / 1000
    check("no card cost: the G8 wall is the host's sum", abs(z2["wall"] - host) / host < 0.01, f"{z2['wall']:.1f} vs {host:.1f}")
    check("the ruler: ratio +-1.0 % at 4 rounds and +-0.8 % at 6 (AGENTS.md; t from tools/gpu-ab.py)",
          abs(tol_ratio(p, 4) - 1.0) < 0.05 and abs(tol_ratio(p, 6) - 0.8) < 0.05 and tol_ratio(p, 1) is None,
          f"{tol_ratio(p, 4):.3f}, {tol_ratio(p, 6):.3f}, n=1 {tol_ratio(p, 1)}")
    pr = routing_prose_trace(p)
    hc = sum(pr.host_slots_tok(l) for l in range(L)) / L * 512
    sh = [sum(sum(lam for lam, _ in pr.host[l][:k]) / (pr.host_slots_tok(l) * 512) for l in range(L)) / L
          for k in (40, 80, 120, 160)]
    tgt = [p["prose_share40"], p["prose_share80"], p["prose_share120"], p["prose_share160"]]
    check("prose-in: 1,011.2 host columns per 512 tokens, shares 0.353/0.595/0.760/0.871 (hoststream-recal:48-49)",
          abs(hc - p["prose_host_cols"]) / p["prose_host_cols"] < 0.001 and all(abs(x - y) < 0.0015 for x, y in zip(sh, tgt)),
          f"{hc:.1f}; " + " ".join(f"{x:.3f}" for x in sh))
    card_cols = sum(sum(lam for lam, _ in pr.card[l]) for l in range(L)) / L / p["prose_n_l"]
    check("prose-in: card experts ~ 29.4 columns per 512 tokens", abs(card_cols - p["prose_card_cols"]) < 0.1,
          f"{card_cols:.2f}")
    lc = routing_lcg(p)
    mean_s = sum(lc.host_slots_tok(l) for l in range(L)) / L
    check("lcg: the per-layer rates keep the recorded mean 4.7131 host slots a token-layer, layers 0-1 all host",
          abs(mean_s - p["s_host_lcg"]) < 1e-9 and lc.n_card(0) == 0 and abs(lc.host_slots_tok(0) - 6) < 1e-9,
          f"{mean_s:.4f}")
    e1, act = ecost(512, 7.6, 30.2, 22.2, 126.4)
    check("E[max(W, a + c m)] >= a + c E[m], P(active) ~ 1 at m 7.6", e1 >= 30.2 + 22.2 * 7.6 - 1e-9 and act > 0.999,
          f"{e1:.1f} >= {30.2 + 22.2 * 7.6:.1f}")
    ro = evaluate(p, dict(CONFIGS["S13b"], ced=False), 512)
    check("the union anchor reproduces itself (S13b CED off = 76.1)", abs(ro["agg"]["union"] - 76.1) < 0.02,
          f"{ro['agg']['union']:.3f}")
    ua, ub = evaluate(p, CONFIGS["UD"], 512)["agg"]["union"], evaluate(p, CONFIGS["UDbase"], 512)["agg"]["union"]
    check("f reproduces the uniondispatch A/B's union difference at P 512 (10.34)", abs(ub - ua - 10.34) < 0.02,
          f"{ub - ua:.3f}")
    old = cause_terms(p, "chunks", 60.0, 2413.0, 317.3, 512, 0.3, 30.2, 22.2)
    n_c = 40
    check("the chunk flow's dispatches at 317.3 host experts: 161 (1 + 40 x 3 + 39 + 1)",
          abs(old["wake"] * 1000 / p["wake_us"] - 161) < 1e-9, f"{old['wake'] * 1000 / p['wake_us']:.0f}, {n_c} chunks")
    fl = [(10.0, 20.0)]
    d = dram_stretch(p, 0.0, 30.0, 5.35e9, fl)
    rate = 5.35e9 / 1e6 / 30.0
    phi = min(1 - p["fill_steal"], (p["dram_eff"] - p["fill_crossings"] * p["pcie_pinned"]) / rate)
    check("DRAM: a union through a fill window loses (1 - phi) of the window", abs(d - (30.0 + 10.0 * (1 - phi))) < 1e-6,
          f"{d:.3f} ms, phi {phi:.3f}")
    sb = [len(x) for x in sub_blocks([(8 * i, 8) for i in range(64)])], [len(x) for x in sub_blocks(
        [(8 * i, 8) for i in range(49)])], [len(x) for x in sub_blocks([(0, 5)] + [(5 + 8 * i, 8) for i in range(15)])]
    check("B1's sub-blocks (chain/attn/batch.rs:78-111): 64 chunks 16x4, 49 16,16,16,1, a short first chunk alone",
          sb == ([16] * 4, [16, 16, 16, 1], [1, 8, 4, 2, 1]), str(sb))
    for P, want in ((512, 983.4), (4096, 1024.9)):
        cfg, rn = row_cfg(p, "B1", {})
        got = evaluate(p, cfg, P, rn)["agg"]["entries"]
        check(f"B1's calls at P {P} = the stat line's entries_route + entries_shadow ({want}), 0.1 %",
              abs(got - want) / want < 1e-3, f"{got:.1f}")
    cfg, rn = row_cfg(p, "B1prose", {"rt": "prose"})
    got = evaluate(p, cfg, 512, rn)["agg"]["host_slots"]
    check("the prose routing reproduces its anchor: 52,189 host slots at P 512 (0.1 %)", abs(got - 52189) / 52189 < 1e-3,
          f"{got:.0f}")
    w1 = evaluate(p, step_cfg({"b1": True, "G": 1, "wrap": True}), 4096)["pp"]
    s1 = evaluate(p, step_cfg({"b1": True}), 4096)["pp"]
    check("the wrap order at G 1 is today's order", abs(w1 - s1) < 1e-9, f"{w1:.3f} vs {s1:.3f}")
    print(f"self-test: {len(fails)} failed")
    return fails


# ============================================================================ main

def main():
    ap = argparse.ArgumentParser(description="V4.1 prompt-batch flow model (see the module docstring)")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--predict", metavar="STEP")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--explain", metavar="ROW")
    ap.add_argument("--uncalibrated", action="store_true")
    ap.add_argument("--cells", action="store_true", help="B1, B1+T, B1+G, B1+T+G per cell and the IMMA shadow over T")
    args = ap.parse_args()
    rc = 0
    if args.self_test:
        rc |= 1 if self_test() else 0
    if args.backtest:
        rc |= 1 if backtest() else 0
    if args.predict:
        predict(args.predict)
    if args.explain:
        explain(args.explain)
    if args.uncalibrated:
        uncalibrated()
    if args.cells:
        cells()
    if not any((args.self_test, args.backtest, args.predict, args.explain, args.uncalibrated, args.cells)):
        ap.print_help()
    return rc


if __name__ == "__main__":
    sys.exit(main())
