#!/usr/bin/env python3
"""How concentrated MoE routing is, per layer, from router sets (tools/ref/router_trace.cpp).

    tools/ref/router-coverage.py coverage <set> [<set>...] [--n 16,32,...] [--layers 3-10,13-43]
    tools/ref/router-coverage.py locality <set> [<set>...]
    tools/ref/router-coverage.py transfer <set A> <set B> [--n 64]
    tools/ref/router-coverage.py compare <set A> <set B> [--tokens N]
    tools/ref/router-coverage.py oracle <set> <oracle set dir>
    tools/ref/router-coverage.py bursts <set> [<set>...] [--window 512] [--n-l <spec>] [--layers 2-39]
    tools/ref/router-coverage.py predict <set> [--n 4,8,16,32] [--layers 2-39] [--positions all|gen]
                                 [--genre G[,G...]] [--tiers CARD,POOL ...]
    tools/ref/router-coverage.py --self-test

A set is a directory router_trace wrote under $BLOOMERY_DATA/router/: counts.tsv, topk-<layer>.u16
(tokens x n_expert_used ids, little-endian u16, token-major) and MANIFEST.tsv. Every command
refuses a set whose manifest has no `# complete` trailer.

coverage  Per layer: the share of the layer's selections (tokens x n_expert_used) that its n hottest
          experts receive, for n in N, under the uniform share n/n_expert; the same over all layers
          (the mean of the layer shares — every layer has the same number of selections); the top-1
          expert's share of tokens (at most 100 %: an expert is picked once per token at most) and
          the Gini coefficient of the counts (0 = uniform). N is --n's comma-separated counts, or
          N_LIST's up to the expert count. --layers keeps only the listed layers (ranges `a-b`
          inclusive), rows and the `all` mean alike; a listed layer the set lacks is refused. Two
          tables:
            in    the top-n list and the share come from the same tokens. Biased up: the top n of noisy
                  counts are partly the lucky ones.
            held  the top-n list from the tokens before the held-out split, the share on the tokens after
                  it — what a placement fixed before the traffic would serve, and the noise floor for
                  `transfer`. The split is the chunk boundary nearest T / 2 (held_split), so no context
                  is cut in two; a set without a `chunk` line, or of one chunk, splits at T / 2.
transfer  Per layer, with each set's hot-n list learned on all its tokens: the overlap |hot(A) & hot(B)|,
          A's list measured on B's selections beside B's own list on B (in-sample) and B's held-out
          share (split as coverage's), and the same the other way round. Does a list learned on one stream serve another?
compare   Two traces of the same tokens (another schedule, ubatch or backend), token by token, per
          layer: the ids equal in rank order, the ids shared as sets, the tokens whose selection is the
          same set, and the first layer where any token's set differs. --tokens N compares the first N
          tokens of two sets of other lengths, traced from the same ids file in the same chunk size
          (another model file's trace of the same text, say).
bursts    Per window of W consecutive positions inside one chunk (--window, default 512; a window across a
          chunk edge crosses a context reset and is skipped), how bursty the routing of the host experts
          is — the experts the card does not hold (--n-l as tools/ref/window-union.py reads it: each
          layer's id prefix; without --n-l every expert is a host expert). Invalid placements and repeated
          --window, --n-l or --layers options are named errors, not a replacement placement. Two tables:
            m     per window and host expert, m = the window's selections of it, over m̄ = W x
                  n_expert_used / n_expert (every expert's mean): p50, p90, p99 and max of m / m̄ over the
                  (window, host expert) pairs the window touches, and the share of the host slots taken
                  by experts with m >= 2m̄, 3m̄, 4m̄ and by the window's hottest tenth of host experts
            phi   per window, the fraction phi of the host experts that reproduces the window's count of
                  touched host experts when each routes at rate lambda / phi, lambda = its set-wide count
                  x W / tokens: sum over host experts of phi (1 - (1 - min(lambda / (phi W), 1))^W), solved
                  by bisection on (0, 1]; phi 1 is routing that follows the set-wide rates (tools/flow's
                  prose_phi rows are this quantity fitted from a union's time). Mean, p10, p90 and window 0
predict   How well one token's routing at layer l names its routing at layer l + 1, the premise of reading
          layer l + 1's misses ahead. Per predicted layer m (--layers lists them; m - 1 must be in the set)
          and per held-out position, the m - 1 ids rank m's experts and recall@n is the share of m's
          selections inside the first n (N = --n, 4,8,16,32 by default). Learn and held-out positions: a
          route trace's contexts.tsv requests by their split (learn: every position; held: every position,
          or with --positions gen only [first + prompt, end), the decode steps; --genre a,b keeps only the
          held requests of those genres), else held_split as coverage's. Rankers, learned on the learn
          positions:
            freq      m's hottest experts, the same list at every position (coverage's held-out hot-n)
            cooccur   score(e') = sum over m - 1's ids e of P(e' at m | e at m - 1), ties to m's count and
                      then the lower id; `cooccur in-sample` fits and scores on the held-out positions
          A ceiling row: n / n_expert_used while n < n_expert_used. The first table's rows are means over
          the layers (every layer has the same selections); a table per layer follows. Each --tiers
          CARD,POOL adds a table of where a selection is read from: the card (CARD: p<C>, ids 0..C-1, the
          id range a plan places, or n<C>, m's C hottest learn experts), else the pool in host memory,
          which holds the card (POOL: n<N>, the card and the hottest other experts up to N, or lru<N>, the
          card and lru_hits' LRU of the other N - C, seeded with the hottest of them; each position is
          judged on the pool as it stood before the position, then its misses are admitted), else NVMe.
          Rows: the card misses and the RAM ones (pool, not card) with the off-card experts ranked, the
          NVMe ones (not in the pool) with the experts out of the pool ranked (a resident is never read
          ahead); each with its share of selections, recall@n pooled over layers and `used`, the hits over
          the reads started (n a position and layer, fewer when fewer experts are candidates). Needs
          numpy, imported by this command only (the self-test calls it).
oracle    Per layer, the set's ids against the oracle set's integer twin of ffn_moe_topk-<layer>,
          integer for integer in rank order, and as sets. The twin is the one the oracle manifest's
          `tensor` row points at: the logical twin when the row marks a view (its flat file is the
          argsort's head, the first n_expert_used x n_tokens values of token 0's sorted row, not each
          token's top-k), else the flat one. The set's tokens, model file and ik build must be the
          oracle's: the model file is the `# model` line, the first shard's full path (two quantizations
          of one model share their shard names, so `# model_file`, the basename, cannot tell them
          apart); a manifest without it or `# build`, or naming another, is refused. Exit 1 on any
          difference.
"""
import importlib.util
import io
import os
import struct
import sys
import tempfile
from array import array
from collections import OrderedDict
from contextlib import redirect_stdout

_spec = importlib.util.spec_from_file_location(
    "manifest", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bloomery", "manifest.py"))
manifest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(manifest)

N_LIST = (16, 32, 48, 64, 96, 128, 192)


class SetError(Exception):
    pass


def read_manifest(set_dir):
    """A router set's header and layers, its rows read by their column lines' names
    (tools/bloomery/manifest.py)."""
    path = os.path.join(set_dir, "MANIFEST.tsv")
    if not os.path.isfile(path):
        raise SetError(f"no manifest at {path}")
    try:
        m = manifest.read(path)
    except manifest.ManifestError as e:
        raise SetError(str(e)) from None
    if not (m.title or "").startswith("# router_trace"):
        raise SetError(f"{path} is not a router set manifest")
    if m.complete is None:
        raise SetError(f"{path} has no `# complete` trailer: the run that wrote it did not finish")
    header = m.header
    return {
        "dir": set_dir,
        "header": header,
        "layers": [r.int("layer") for r in m.rows("layer")],
        "tokens": int(header["tokens"]),
        "n_expert": int(header["n_expert"]),
        "n_used": int(header["n_expert_used"]),
    }


def read_counts(s):
    counts = {l: [0] * s["n_expert"] for l in s["layers"]}
    with open(os.path.join(s["dir"], "counts.tsv"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            l, e, c = (int(x) for x in line.split("\t"))
            counts[l][e] = c
    for l, c in counts.items():
        if sum(c) != s["tokens"] * s["n_used"]:
            raise SetError(f"{s['dir']}: layer {l} counts sum to {sum(c)}, not tokens x n_expert_used")
    return counts


def read_topk(s, layer):
    a = array("H")
    path = os.path.join(s["dir"], f"topk-{layer}.u16")
    with open(path, "rb") as f:
        a.frombytes(f.read())
    if sys.byteorder != "little":
        a.byteswap()
    if len(a) != s["tokens"] * s["n_used"]:
        raise SetError(f"{path} holds {len(a)} ids, not tokens x n_expert_used = {s['tokens'] * s['n_used']}")
    K = s["n_used"]
    for t in range(s["tokens"]):
        if len(set(a[t * K:(t + 1) * K])) != K:
            raise SetError(f"{path}: token {t} selects an expert twice")
    return a


def count(ids, n_expert, t0, t1, n_used):
    c = [0] * n_expert
    for e in ids[t0 * n_used:t1 * n_used]:
        c[e] += 1
    return c


def hot(counts, n):
    """The n hottest experts, ties broken by the lower id so the list is reproducible."""
    return sorted(range(len(counts)), key=lambda e: (-counts[e], e))[:n]


def share(counts, experts):
    total = sum(counts)
    return sum(counts[e] for e in experts) / total if total else 0.0


def gini(counts):
    x = sorted(counts)
    n, total = len(x), sum(x)
    if total == 0:
        return 0.0
    return 2.0 * sum((i + 1) * v for i, v in enumerate(x)) / (n * total) - (n + 1) / n


def pct(v):
    return f"{100.0 * v:.1f}"


def held_split(s):
    """The held-out split: the chunk boundary nearest T / 2 (ties to the lower one), so the learning and
    the held-out halves are whole contexts; T / 2 when the set has no `chunk` line or no boundary inside
    it (one context)."""
    T = s["tokens"]
    chunk = int(s["header"].get("chunk", 0) or 0)
    if chunk <= 0 or chunk >= T:
        return T // 2
    bounds = range(chunk, T, chunk)
    return min(bounds, key=lambda b: (abs(b - T // 2), b))


def lru_hits(ids, n_used, capacity, reset_tokens=0, seed=(), skip=(), before=None):
    """Replay ranked selections; hits refresh recency and misses insert then evict.
    The second count checks membership before any selection of that token is inserted.
    The cache starts holding `seed` (least recent first); ids in `skip` never enter it and count as neither;
    `before`, a list, gets each token's cache members before the token."""
    cache = OrderedDict((e, None) for e in seed)
    hits, start_hits = 0, 0
    for p in range(0, len(ids), n_used):
        if reset_tokens and (p // n_used) % reset_tokens == 0:
            cache.clear()
        row = [e for e in ids[p:p + n_used] if e not in skip]
        if before is not None:
            before.append(list(cache))
        start_hits += sum(e in cache for e in row)
        for e in row:
            if e in cache:
                hits += 1
                cache.move_to_end(e)
            else:
                cache[e] = None
                if len(cache) > capacity:
                    cache.popitem(last=False)
    return hits, start_hits


def locality(set_dirs):
    for d in set_dirs:
        s = read_manifest(d)
        counts = read_counts(s)
        E, T, K = s["n_expert"], s["tokens"], s["n_used"]
        chunk = int(s["header"].get("chunk", 0) or 0)
        half = held_split(s)
        if E < 128 or K < 1 or T < 2 or not s["layers"]:
            raise SetError(f"{d}: locality needs >=128 experts, positive top-k, >=2 tokens and a routed layer")
        rows = []
        for layer in s["layers"]:
            ids = read_topk(s, layer)
            for p in range(0, len(ids), K):
                if any(e >= E for e in ids[p:p + K]):
                    raise SetError(f"{d}: layer {layer} token {p // K} has an expert id not below n_expert {E}")
            first = count(ids, E, 0, half, K)
            second = count(ids, E, half, T, K)
            if [a + b for a, b in zip(first, second)] != counts[layer]:
                raise SetError(f"{d}: layer {layer} counts.tsv does not match topk-{layer}.u16")
            h64, s64 = lru_hits(ids, K, 64)
            h128, s128 = lru_hits(ids, K, 128)
            cold64, _ = lru_hits(ids, K, 64, chunk)
            cold128, _ = lru_hits(ids, K, 128, chunk)
            total = T * K
            rows.append((layer, [h64 / total, h128 / total, share(second, hot(first, 32)),
                                 s64 / total, s128 / total, cold64 / total, cold128 / total]))
        print(f"## {d}\n")
        print(f"{T} tokens, {len(rows)} layers, {E} experts, top-{K}, chunk {chunk}. "
              f"Static top-32 learns on [0, {half}) and is evaluated on [{half}, {T}).\n")
        print("LRU starts empty, follows rank order, and persists across context resets. "
              "Start columns check the cache before the token; cold columns reset it at every context. "
              "All columns are percent of routed selections.\n")
        print("| layer | LRU-64 | LRU-128 | static-32 held | start-64 | start-128 | cold-64 | cold-128 |")
        print("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for layer, values in rows:
            print(f"| {layer} | " + " | ".join(pct(v) for v in values) + " |")
        print("| all | " + " | ".join(pct(sum(v[i] for _, v in rows) / len(rows)) for i in range(7)) + " |\n")


def coverage(set_dirs, asked=None, keep=None):
    for d in set_dirs:
        s = read_manifest(d)
        counts = read_counts(s)
        E, T, K = s["n_expert"], s["tokens"], s["n_used"]
        # N_LIST's counts up to E by default; an asked count past E is refused.
        n_list = tuple(n for n in N_LIST if n <= E) if asked is None else asked
        if not all(0 < n <= E for n in n_list):
            raise SetError(f"{d}: --n {','.join(map(str, n_list))} is not within 1..{E}")
        h = s["header"]
        half = held_split(s)
        print(f"## {d}\n")
        print(f"{T} tokens of {h.get('ids', '?')} (md5 {h.get('ids_md5', '?')}), chunk {h.get('chunk', '?')}, "
              f"{len(s['layers'])} layers, {E} experts, top-{K}. held = top-n list from tokens [0, {half}), "
              f"share on [{half}, {T}).\n")
        rows_in, rows_held = [], []
        if keep is not None and not set(keep) <= set(s["layers"]):
            raise SetError(f"{d}: --layers {sorted(set(keep) - set(s['layers']))} not in the set")
        for l in s["layers"]:
            if keep is not None and l not in keep:
                continue
            c = counts[l]
            ids = read_topk(s, l)
            first, second = count(ids, E, 0, half, K), count(ids, E, half, T, K)
            if [x + y for x, y in zip(first, second)] != c:
                raise SetError(f"{d}: layer {l} counts.tsv does not match topk-{l}.u16")
            order = hot(c, E)
            order_first = hot(first, E)
            # in-sample rows carry the top-1 share of tokens and the Gini ahead of the hot-n shares
            rows_in.append((l, [max(c) / T, gini(c)] + [share(c, order[:n]) for n in n_list]))
            rows_held.append((l, [share(second, order_first[:n]) for n in n_list]))
        n_cols = " | ".join(f"n={n}" for n in n_list)
        uniform = " | ".join(pct(n / E) for n in n_list)
        tables = (("in-sample", "top-1 % of tokens | Gini | ", f"{pct(K / E)} | 0.000 | ", 2, rows_in),
                  ("held-out", "", "", 0, rows_held))
        for title, extra_head, extra_uniform, n_extra, rows in tables:
            def cells(v):
                return " | ".join(f"{x:.3f}" if i == 1 and n_extra else pct(x) for i, x in enumerate(v))
            print(f"### hot-n share of selections, {title} (%)\n")
            print(f"| layer | {extra_head}{n_cols} |")
            print("|---:|" + "---:|" * (n_extra + len(n_list)))
            print(f"| uniform | {extra_uniform}{uniform} |")
            for l, v in rows:
                print(f"| {l} | {cells(v)} |")
            mean = [sum(r[1][i] for r in rows) / len(rows) for i in range(len(rows[0][1]))]
            print(f"| all | {cells(mean)} |\n")


def layer_list(text):
    """`3-10,13` as [3, ..., 10, 13]; a range whose end is before its start is refused."""
    out = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        lo, hi = int(lo), int(hi or lo)
        if hi < lo:
            raise SetError(f"--layers {part}: the end is before the start")
        out.extend(range(lo, hi + 1))
    return out


def transfer(dir_a, dir_b, n):
    a, b = read_manifest(dir_a), read_manifest(dir_b)
    if (a["n_expert"], a["n_used"]) != (b["n_expert"], b["n_used"]) or a["layers"] != b["layers"]:
        raise SetError("the two sets route over different experts or layers")
    ca, cb = read_counts(a), read_counts(b)
    E, K = a["n_expert"], a["n_used"]

    def held(s, l):
        ids, T, half = read_topk(s, l), s["tokens"], held_split(s)
        return share(count(ids, E, half, T, K), hot(count(ids, E, 0, half, K), n))

    print(f"## hot-{n} transfer: A = {dir_a} ({a['tokens']} tokens), B = {dir_b} ({b['tokens']} tokens)\n")
    print(f"Shares are of the evaluated set's selections, %. Uniform: {pct(n / E)}.\n")
    print(f"| layer | overlap of hot-{n} | A-list on B | B-list on B (in) | B held-out | "
          "B-list on A | A-list on A (in) | A held-out |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    sums = [0.0] * 7
    for l in a["layers"]:
        ha, hb = hot(ca[l], n), hot(cb[l], n)
        row = [len(set(ha) & set(hb)), share(cb[l], ha), share(cb[l], hb), held(b, l),
               share(ca[l], hb), share(ca[l], ha), held(a, l)]
        sums = [x + y for x, y in zip(sums, row)]
        print(f"| {l} | {row[0]} | " + " | ".join(pct(v) for v in row[1:]) + " |")
    k = len(a["layers"])
    print(f"| all | {sums[0] / k:.1f} | " + " | ".join(pct(v / k) for v in sums[1:]) + " |\n")


def agreement(a, b, K):
    """(ids equal in rank order, ids shared as sets, tokens whose selection is the same set)."""
    ranked = sum(1 for x, y in zip(a, b) if x == y)
    shared = same = 0
    for t in range(len(a) // K):
        sa, sb = set(a[t * K:(t + 1) * K]), set(b[t * K:(t + 1) * K])
        shared += len(sa & sb)
        same += sa == sb
    return ranked, shared, same


def compare(dir_a, dir_b, tokens=None):
    """Token by token over both sets, or over their first `tokens` tokens: a prefix is the same
    tokens at the same positions only when both were traced from the same ids file in the same chunk
    size, so both are required. Returns the first layer with a differing set, or None."""
    a, b = read_manifest(dir_a), read_manifest(dir_b)
    if (a["n_used"], a["layers"]) != (b["n_used"], b["layers"]):
        raise SetError("the two sets differ in top-k or layers")
    if tokens is None:
        if a["tokens"] != b["tokens"]:
            raise SetError(f"the two sets differ in token count ({a['tokens']} vs {b['tokens']}); "
                           "--tokens N compares a common prefix")
        T = a["tokens"]
    else:
        if tokens < 1 or tokens > min(a["tokens"], b["tokens"]):
            raise SetError(f"--tokens {tokens} is not in 1..{min(a['tokens'], b['tokens'])}, the shorter set")
        if a["header"].get("chunk") != b["header"].get("chunk"):
            raise SetError(f"--tokens needs one chunk size: A `{a['header'].get('chunk')}`, "
                           f"B `{b['header'].get('chunk')}` put the prefix at other positions")
        T = tokens
    if a["header"].get("ids_md5") != b["header"].get("ids_md5"):
        raise SetError("the two sets were traced from different ids files")
    K = a["n_used"]
    print(f"## compare: A = {dir_a}, B = {dir_b} ({T} tokens, top-{K})\n")
    for key in ("model", "build", "schedule", "n_ubatch", "flags"):
        print(f"- {key}: A `{a['header'].get(key)}`, B `{b['header'].get(key)}`")
    print("\n| layer | ids equal, rank order % | ids shared as sets % | tokens with the same set % |")
    print("|---:|---:|---:|---:|")
    first_diff = None
    for l in a["layers"]:
        ranked, shared, same = agreement(read_topk(a, l)[:T * K], read_topk(b, l)[:T * K], K)
        if same < T and first_diff is None:
            first_diff = l
        print(f"| {l} | {pct(ranked / (T * K))} | {pct(shared / (T * K))} | {pct(same / T)} |")
    print(f"\nfirst layer with a differing set: {first_diff if first_diff is not None else 'none'}\n")
    return first_diff


def window_union():
    """tools/ref/window-union.py, loaded by path when a command needs a placement (it loads this file)."""
    spec = importlib.util.spec_from_file_location(
        "window_union", os.path.join(os.path.dirname(os.path.abspath(__file__)), "window-union.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def solve_phi(lams, window, touched):
    """The phi in (0, 1] at which sum over lams of phi (1 - (1 - min(lam / (phi window), 1))^window) equals
    touched; the sum rises with phi, so bisection. A window that touches more than phi 1 predicts reads 1."""
    def f(phi):
        return sum(phi * (1.0 - (1.0 - min(lam / (phi * window), 1.0)) ** window) for lam in lams)
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if f(mid) > touched:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def burst_stats(s, card, window, keep=None):
    """Per window of `window` positions inside one chunk: the m / m̄ samples of the touched host experts,
    the slot shares by m threshold and of the hottest tenth, and phi."""
    T, K, E = s["tokens"], s["n_used"], s["n_expert"]
    chunk = int(s["header"].get("chunk", 0) or 0) or T
    starts = [w for w in range(0, T - window + 1, window) if w // chunk == (w + window - 1) // chunk]
    if not starts:
        raise SetError(f"{s['dir']}: no window of {window} positions inside a chunk of {chunk}")
    mbar = window * K / E
    layers = [l for l in s["layers"] if keep is None or l in keep]
    counts = read_counts(s)
    per_window = [[0] * E for _ in starts]  # scratch reused per layer
    touched = [0] * len(starts)
    slots = [0] * len(starts)
    thresh = {2: [0] * len(starts), 3: [0] * len(starts), 4: [0] * len(starts)}
    top_tenth = [0] * len(starts)
    ratios = []
    lams = []
    for l in layers:
        host = [e for e in range(E) if e not in card.get(l, ())]
        lams.extend(counts[l][e] * window / T for e in host if counts[l][e])
        ids = read_topk(s, l)
        for i, w in enumerate(starts):
            m = per_window[i]
            for e in range(E):
                m[e] = 0
            for e in ids[w * K:(w + window) * K]:
                m[e] += 1
            hm = sorted((m[e] for e in host if m[e]), reverse=True)
            touched[i] += len(hm)
            slots[i] += sum(hm)
            for x, acc in thresh.items():
                acc[i] += sum(v for v in hm if v >= x * mbar)
            top_tenth[i] += sum(hm[:max(1, len(host) // 10)]) if hm else 0
            ratios.extend(v / mbar for v in hm)
    phis = [solve_phi(lams, window, touched[i]) for i in range(len(starts))]
    return {"windows": len(starts), "mbar": mbar, "layers": len(layers), "ratios": sorted(ratios),
            "slots": slots, "thresh": thresh, "top_tenth": top_tenth, "phi": phis, "touched": touched}


def bursts(set_dirs, window, spec=None, keep=None):
    wu = window_union() if spec is not None else None
    for d in set_dirs:
        s = read_manifest(d)
        if keep is not None and not set(keep) <= set(s["layers"]):
            raise SetError(f"{d}: --layers {sorted(set(keep) - set(s['layers']))} not in the set")
        if spec is not None:
            try:
                n_l = wu.parse_n_l(spec, s["layers"], s["n_expert"])
            except wu.rc.SetError as e:
                raise SetError(str(e)) from None
            card = wu.card_sets(s["layers"], n_l)
        else:
            card = {l: set() for l in s["layers"]}
        b = burst_stats(s, card, window, keep)
        r = b["ratios"]
        total = sum(b["slots"])

        def q(v):
            return r[min(len(r) - 1, int(v * (len(r) - 1) + 0.5))] if r else float("nan")

        def mean_share(acc):
            v = [a / sl for a, sl in zip(acc, b["slots"]) if sl]
            return sum(v) / len(v) if v else float("nan")
        phis = sorted(b["phi"])
        print(f"## {d}: {b['windows']} windows of {window} inside chunks of {s['header'].get('chunk', '-')}, "
              f"{b['layers']} layers, card {'n_l ' + spec if spec else 'none'}, m̄ {b['mbar']:.2f}\n")
        print("| host slots/window, layers summed | p50 m/m̄ | p90 | p99 | max | slots m>=2m̄ | >=3m̄ | >=4m̄ | hottest tenth |")
        print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        print(f"| {total / b['windows']:.0f} | {q(0.5):.2f} | {q(0.9):.2f} | {q(0.99):.2f} | "
              f"{(r[-1] if r else float('nan')):.1f} | {pct(mean_share(b['thresh'][2]))} | "
              f"{pct(mean_share(b['thresh'][3]))} | {pct(mean_share(b['thresh'][4]))} | "
              f"{pct(mean_share(b['top_tenth']))} |\n")
        print("| phi mean | p10 | p90 | window 0 | touched host experts/window |")
        print("|---:|---:|---:|---:|---:|")
        print(f"| {sum(phis) / len(phis):.3f} | {phis[int(0.1 * (len(phis) - 1) + 0.5)]:.3f} | "
              f"{phis[int(0.9 * (len(phis) - 1) + 0.5)]:.3f} | {b['phi'][0]:.3f} | "
              f"{sum(b['touched']) / b['windows']:.0f} |\n")


def route_trace():
    """tools/bloomery/route_trace.py, loaded by path when a command reads a route trace's sidecars."""
    spec = importlib.util.spec_from_file_location(
        "route_trace", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bloomery", "route_trace.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def predict_spans(s, positions, genre):
    """(learn spans, held-out spans, a line naming them): contexts.tsv's splits when the set has one, else
    held_split's halves."""
    rt = route_trace()
    try:
        rows, _ = rt.read_contexts(s["dir"])
    except rt.TraceError as e:
        raise SetError(str(e)) from None
    T = s["tokens"]
    if rows is None:
        if positions != "all" or genre is not None:
            raise SetError(f"{s['dir']}: --positions gen and --genre read a contexts.tsv, and the set has none")
        half = held_split(s)
        return [(0, half)], [(half, T)], f"learn [0, {half}), held-out [{half}, {T})"
    learn = [(r["first"], r["end"]) for r in rows if r["split"] == "learn"]
    held = [(r["first"] + (r["prompt"] if positions == "gen" else 0), r["end"]) for r in rows
            if r["split"] == "held" and (genre is None or r["genre"] in genre.split(","))]
    held = [(a, b) for a, b in held if b > a]
    if not learn or not held:
        raise SetError(f"{s['dir']}: no learn request or no held request{' of genre ' + genre if genre else ''}")
    return learn, held, (f"learn {len(learn)} requests ({sum(b - a for a, b in learn)} positions), held-out "
                         f"{len(held)} requests{' of genre ' + genre if genre else ''} "
                         f"({sum(b - a for a, b in held)} {'decode ' if positions == 'gen' else ''}positions)")


def tier_spec(text):
    """`--tiers CARD,POOL` as ((kind, n), (kind, n)): CARD p<C> or n<C>, POOL n<N> or lru<N>."""
    card, _, pool = text.partition(",")

    def one(t, kinds):
        for k in kinds:
            if t.startswith(k) and t[len(k):].isdigit():
                return k, int(t[len(k):])
        raise SetError(f"--tiers {text!r}: CARD is p<C> or n<C>, POOL n<N> or lru<N>")
    c, p = one(card, ("p", "n")), one(pool, ("lru", "n"))
    if p[1] < c[1]:
        raise SetError(f"--tiers {text!r}: the pool holds the card, so N >= C")
    return c, p


def predict(set_dir, n_list, keep=None, positions="all", genre=None, tiers=()):
    try:
        import numpy as np
    except ImportError:
        raise SetError("predict needs numpy: python3 -m pip install numpy") from None
    s = read_manifest(set_dir)
    E, K, T = s["n_expert"], s["n_used"], s["tokens"]
    if not all(0 < n <= E for n in n_list):
        raise SetError(f"{set_dir}: --n {','.join(map(str, n_list))} is not within 1..{E}")
    if any(p[1] > E for _, p in tiers) or len(set(tiers)) != len(tiers):
        raise SetError(f"{set_dir}: a --tiers pool past the expert count {E}, or a --tiers given twice")
    targets = [m for m in s["layers"] if m - 1 in s["layers"]] if keep is None else keep
    bad = [m for m in targets if m not in s["layers"] or m - 1 not in s["layers"]]
    if bad or not targets:
        raise SetError(f"{set_dir}: --layers {bad}: a predicted layer and the one before it must be in the set")
    learn, held, said = predict_spans(s, positions, genre)
    if not all(b > a for a, b in learn + held):
        raise SetError(f"{set_dir}: an empty learn or held-out span ({said})")
    ids = {}

    def layer(l):
        if l not in ids:
            a = np.frombuffer(read_topk(s, l), dtype=np.uint16).astype(np.int64).reshape(T, K)
            if a.size and a.max() >= E:
                raise SetError(f"{set_dir}: layer {l}: id {a.max()} >= n_expert {E}")
            ids[l] = a
        return ids[l]

    def rows(a, spans):
        return np.concatenate([a[t0:t1] for t0, t1 in spans])

    def places(score, freq, exclude=None):
        """[P, E]: each expert's place in the ranking (0 is first) by score, then count, then the lower id;
        excluded experts go last."""
        P = len(score)
        if exclude is not None:
            score = np.where(exclude, -np.inf, score)
        order = np.lexsort((np.broadcast_to(np.arange(E), (P, E)), np.broadcast_to(-freq, (P, E)), -score), axis=-1)
        out = np.empty_like(order)
        out[np.arange(P)[:, None], order] = np.arange(E)
        return out

    def cooccur(prev_learn, cur_learn, prev_eval):
        c = np.zeros((E, E))
        np.add.at(c, (prev_learn[:, :, None], cur_learn[:, None, :]), 1)
        p = c / np.maximum(np.bincount(prev_learn.ravel(), minlength=E), 1)[:, None]
        return sum(p[prev_eval[:, i]] for i in range(K))

    def card_pool(freq, b_e, spec):
        """[P, E] residency of the card and of the pool (which holds the card) before each held-out position."""
        (ck, cn), (pk, pn) = spec
        P = len(b_e)
        rank = np.array(hot(freq.tolist(), E))
        card = np.zeros(E, dtype=bool)
        card[np.arange(cn) if ck == "p" else rank[:cn]] = True
        rest = rank[~card[rank]]
        room = pn - cn
        pool = np.broadcast_to(card, (P, E)).copy()
        if pk == "n":
            pool[:, rest[:room]] = True
            return np.broadcast_to(card, (P, E)), pool
        held_lru = []
        lru_hits(b_e.ravel().tolist(), K, room, seed=rest[:room][::-1].tolist(),
                 skip=set(np.nonzero(card)[0].tolist()), before=held_lru)
        for t, members in enumerate(held_lru):
            pool[t, members] = True
        return np.broadcast_to(card, (P, E)), pool

    print(f"## {set_dir}: predict layer m from layer m - 1\n")
    print(f"{T} positions, {E} experts, top-{K}, predicted layers {targets[0]}..{targets[-1]} ({len(targets)}). "
          f"{said}.\n")
    table = {"freq": [], "cooccur": [], "cooccur in-sample": []}
    acc = {t: {} for t in tiers}
    P = 0
    for m in targets:
        a, b = layer(m - 1), layer(m)
        a_l, b_l, a_e, b_e = rows(a, learn), rows(b, learn), rows(a, held), rows(b, held)
        P = len(b_e)
        freq = np.bincount(b_l.ravel(), minlength=E)
        zero = np.zeros((P, E))
        score = cooccur(a_l, b_l, a_e)
        got = {"freq": np.take_along_axis(places(zero, freq), b_e, axis=1),
               "cooccur": np.take_along_axis(places(score, freq), b_e, axis=1),
               "cooccur in-sample": np.take_along_axis(
                   places(cooccur(a_e, b_e, a_e), np.bincount(b_e.ravel(), minlength=E)), b_e, axis=1)}
        for k, v in got.items():
            table[k].append([float((v < n).mean()) for n in n_list])
        for spec in tiers:
            card, pool = card_pool(freq, b_e, spec)
            on_card = np.take_along_axis(card, b_e, axis=1)
            in_pool = np.take_along_axis(pool, b_e, axis=1)
            classes = (("card miss", ~on_card, card), ("RAM", ~on_card & in_pool, card), ("NVMe", ~in_pool, pool))
            for name, sel, ex in classes:
                for ranker, sc in (("freq", zero), ("cooccur", score)):
                    pl = np.take_along_axis(places(sc, freq, ex), b_e, axis=1)[sel]
                    cand = E - ex.sum(axis=1)
                    row = acc[spec].setdefault((name, ranker), [0, [0] * len(n_list), [0] * len(n_list)])
                    row[0] += int(sel.sum())
                    for i, n in enumerate(n_list):
                        row[1][i] += int((pl < n).sum())
                        row[2][i] += int(np.minimum(cand, n).sum())
    L = len(targets)
    n_cols = " | ".join(f"n={n}" for n in n_list)
    print(f"### recall@n of all selections (%)\n\n| ranker | {n_cols} |\n|---|" + "---:|" * len(n_list))
    print(f"| ceiling | {' | '.join(pct(min(n, K) / K) for n in n_list)} |")
    for k, v in table.items():
        print(f"| {k} | {' | '.join(pct(sum(r[i] for r in v) / L) for i in range(len(n_list)))} |")
    print("\n| layer m | " + " | ".join(f"{k} n={n}" for k in table for n in n_list) + " |")
    print("|---:|" + "---:|" * (len(table) * len(n_list)))
    for j, m in enumerate(targets):
        print(f"| {m} | " + " | ".join(pct(table[k][j][i]) for k in table for i in range(len(n_list))) + " |")
    print()
    for spec, rows_ in acc.items():
        (ck, cn), (pk, pn) = spec
        sel = P * L * K
        print(f"### card {ck}{cn}, pool {pk}{pn}: recall@n of the misses (%)\n")
        print("Candidates are the experts off the card (card miss, RAM) or out of the pool (NVMe). `used` is the "
              f"class's hits over the reads started, n a position and layer or every candidate when fewer. "
              f"{P} positions x {L} layers.\n")
        print("| class | % of selections | ranker | " + " | ".join(f"n={n} | used" for n in n_list) + " |")
        print("|---|---:|---|" + "---:|" * (2 * len(n_list)))
        for (name, ranker), (miss, hits, reads) in rows_.items():
            print(f"| {name} | {pct(miss / sel)} | {ranker} | " + " | ".join(
                f"{pct(h / miss) if miss else '-'} | {pct(h / r) if r else '-'}" for h, r in zip(hits, reads)) + " |")
        print()


def read_oracle(oracle_dir):
    """An oracle set's header, its `tensor` rows by (name, occurrence) and its integer twins of
    tensors by (name, occurrence, layout), read by their column lines' names."""
    try:
        m = manifest.read(os.path.join(oracle_dir, "MANIFEST.tsv"))
    except manifest.ManifestError as e:
        raise SetError(str(e)) from None
    if m.complete is None:
        raise SetError(f"{m.path} has no `# complete` trailer")
    tensors = {(r["name"], r.int("occurrence")): r for r in m.rows("tensor")}
    ints = {(r["name"], r.int("occurrence"), r["layout"]): r for r in m.rows("int") if r["of"] == "tensor"}
    return m.header, tensors, ints


def oracle(set_dir, oracle_dir):
    s = read_manifest(set_dir)
    header, tensors, ints = read_oracle(oracle_dir)
    K, T = s["n_used"], s["tokens"]
    for key in ("model", "build"):
        ours, theirs = s["header"].get(key), header.get(key)
        if ours is None or theirs is None:
            where = set_dir if ours is None else oracle_dir
            raise SetError(f"{where}: its manifest has no `# {key}` line, so what it was traced from is unknown")
        if ours != theirs:
            raise SetError(f"the set and the oracle name another `# {key}`: set {ours!r}, oracle {theirs!r}")
    problems = []
    want = [int(x) for x in header.get("tokens", "").split(",") if x]
    try:
        with open(s["header"]["ids"], encoding="utf-8") as f:
            got = [int(x) for x in f.read().split()][:T]
    except OSError as e:
        got = None
        problems.append(f"cannot read the set's ids file: {e}")
    if got is not None and got != want:
        problems.append(f"tokens: set {got}, oracle {want}")
    if problems:
        for p in problems:
            print(f"oracle: {p}")
        print("oracle: FAIL — the set and the oracle are not the same run's inputs")
        return 1

    total = equal = 0
    bad_layers = []
    for l in s["layers"]:
        name = f"ffn_moe_topk-{l}"
        row = tensors.get((name, 0))
        if row is None:
            print(f"layer {l:2d}: the oracle has no {name}")
            bad_layers.append(l)
            continue
        ne0, ne1 = row.int("ne0"), row.int("ne1")
        layout = "logical" if row["logical"] == "1" else "flat"
        twin = ints.get((name, 0, layout))
        if twin is None or (ne0, ne1) != (K, T):
            print(f"layer {l:2d}: oracle {name} is [{ne0}, {ne1}] with no {layout} twin; the set is [{K}, {T}]")
            bad_layers.append(l)
            continue
        with open(os.path.join(oracle_dir, twin["file"]), "rb") as f:
            ref = list(struct.unpack(f"<{ne0 * ne1}i", f.read()))
        ours = list(read_topk(s, l))
        diff = [t for t in range(T) if ours[t * K:(t + 1) * K] != ref[t * K:(t + 1) * K]]
        n_eq, n_shared, _ = agreement(ours, ref, K)
        total += len(ref)
        equal += n_eq
        line = f"layer {l:2d}: {n_eq}/{len(ref)} ids equal, {n_shared}/{len(ref)} shared as sets  " \
               f"(oracle: {row['op']} of {row['src0']}, contig {row['contig']}, {twin['file']})"
        if diff:
            t = diff[0]
            line += f"  DIFF at token {t}: ours {ours[t * K:(t + 1) * K]} oracle {ref[t * K:(t + 1) * K]}"
            bad_layers.append(l)
        print(line)
    ok = not bad_layers and total == equal
    print(f"total: {len(s['layers'])} layers, {equal}/{total} ids equal, "
          f"{len(bad_layers)} layer(s) differ — {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def self_test():
    """Synthetic sets with known answers: the math, the refusals and the oracle comparison."""
    E, K = 8, 2
    MODEL = "/models/q3/m-00001-of-00002.gguf"
    # Refreshing a hit must evict the least recent expert, rather than the first inserted.
    assert lru_hits([0, 1, 0, 2, 0, 1], 1, 2) == (2, 2)
    # A miss can evict an expert that was present at the start of this token.
    assert lru_hits([0, 1, 2, 1], 2, 2) == (1, 1)
    assert lru_hits([0, 1, 2, 0], 2, 2) == (0, 1)
    assert lru_hits([0, 1, 0, 1], 2, 2, 1) == (0, 0)
    with tempfile.TemporaryDirectory() as tmp:
        def make(name, rows_by_layer, complete=True, model=MODEL, chunk=None, n_expert=E):
            d = os.path.join(tmp, name)
            os.mkdir(d)
            T = len(next(iter(rows_by_layer.values())))
            with open(os.path.join(d, "counts.tsv"), "w") as f:
                f.write("# layer\texpert\tcount\n")
                for l, rows in rows_by_layer.items():
                    c = [0] * n_expert
                    for r in rows:
                        for e in r:
                            c[e] += 1
                    for e in range(n_expert):
                        f.write(f"{l}\t{e}\t{c[e]}\n")
                    with open(os.path.join(d, f"topk-{l}.u16"), "wb") as g:
                        g.write(array("H", [e for r in rows for e in r]).tobytes())
            with open(os.path.join(d, "MANIFEST.tsv"), "w") as f:
                f.write("# router_trace — self-test\n# tokens\t%d\n# n_expert\t%d\n# n_expert_used\t%d\n"
                        % (T, n_expert, len(next(iter(rows_by_layer.values()))[0])))
                if model is not None:
                    f.write(f"# model\t{model}\n")
                if chunk:
                    f.write(f"# chunk\t{chunk}\n")
                f.write("# model_file\tm.gguf\n# build\tb\n# ids\t%s\n" % os.path.join(d, "ids"))
                f.write("# layer\tlayer\tsource\tproducer\ttokens\tid_sum\tignored\tfile\n")
                for l in rows_by_layer:
                    f.write(f"layer\t{l}\tVIEW-of-ARGSORT\tp\t{T}\t0\t0\ttopk-{l}.u16\n")
                if complete:
                    f.write(f"# complete\t{T}\t{len(rows_by_layer)}\n")
            with open(os.path.join(d, "ids"), "w") as f:
                f.write("\n".join(str(t) for t in range(T)) + "\n")
            return d

        uniform = [[(2 * t) % E, (2 * t + 1) % E] for t in range(8)]  # every expert twice
        s = read_manifest(make("u", {0: uniform}))
        c = read_counts(s)[0]
        assert c == [2] * E and gini(c) == 0.0, c
        assert share(c, hot(c, 2)) == 2 / E
        skew = [[0, 1]] * 6 + [[2, 3], [4, 5]]  # experts 0, 1 take 12 of 16 selections
        c = read_counts(read_manifest(make("s", {0: skew})))[0]
        assert share(c, hot(c, 2)) == 12 / 16 and hot(c, 2) == [0, 1], (c, hot(c, 2))
        assert abs(gini([0] * 7 + [1]) - 7 / 8) < 1e-12
        assert agreement([1, 2, 3, 4], [2, 1, 3, 5], K) == (1, 3, 1)
        # the held-out split lands on the chunk boundary nearest T / 2, never inside a context
        assert held_split({"tokens": 50000, "header": {"chunk": "2048"}}) == 24576
        assert held_split({"tokens": 10, "header": {"chunk": "4"}}) == 4
        assert held_split({"tokens": 12, "header": {"chunk": "4"}}) == 4  # 4 and 8 tie around 6: the lower
        assert held_split({"tokens": 10, "header": {}}) == 5
        assert held_split({"tokens": 10, "header": {"chunk": "16"}}) == 5
        local = make("local", {0: [[0, 1]] * 4 + [[126, 127]] * 4}, n_expert=128, chunk=4)
        buf = io.StringIO()
        with redirect_stdout(buf):
            locality([local])
        assert "| 0 | 75.0 | 75.0 | 0.0 |" in buf.getvalue(), buf.getvalue()
        # A counts file of the right size cannot hide duplicate ids or inconsistent contents.
        for name, bad_rows in (("duplicate", [[0, 0]] * 2), ("mismatch", [[0, 1]] * 2)):
            bad_local = make(name, {0: bad_rows}, n_expert=128)
            if name == "mismatch":
                with open(os.path.join(bad_local, "topk-0.u16"), "wb") as f:
                    f.write(array("H", [2, 3, 2, 3]).tobytes())
            try:
                locality([bad_local])
            except SetError:
                pass
            else:
                raise AssertionError(f"locality accepted {name}")
        # the reader itself refuses a token that selects one expert twice, naming it, whatever counts.tsv says,
        # so every caller (locality, predict, router-residency, router-rank) gets only distinct ids
        dup = make("dup", {0: [[0, 1], [1, 1], [2, 3]]})
        try:
            read_topk(read_manifest(dup), 0)
        except SetError as e:
            assert "token 1 selects an expert twice" in str(e), e
        else:
            raise AssertionError("read_topk accepted a token that selects an expert twice")
        assert list(read_topk(read_manifest(make("dup-free", {0: [[0, 1], [1, 2], [2, 3]]})), 0)) == [0, 1, 1, 2, 2, 3]
        # bursts: routing that follows the set-wide rates everywhere reads phi 1 — 16 experts over 8
        # windows of 4 rows, every window every expert twice (K 8)
        flat = [[(t * 8 + j) % 16 for j in range(8)] for t in range(32)]
        b = burst_stats(read_manifest(make("flat", {0: flat}, n_expert=16, chunk=16)), {0: set()}, 4)
        assert b["windows"] == 8 and abs(b["mbar"] - 2.0) < 1e-12, b["windows"]
        assert all(abs(p - 1.0) < 1e-9 for p in b["phi"]), b["phi"]
        assert b["ratios"] == [1.0] * 128 and all(v == 0 for v in b["thresh"][2])
        # a burst: each window of 4 rows picks the same two experts, a new pair per window; each
        # expert's set-wide rate is 0.5 a window, so phi solves 16 phi (1 - (1 - 1 / (8 phi))^4) = 2:
        # phi 1/8, and every touched expert takes m = 4 = 8 m̄ (m̄ = 4 x 2 / 16)
        burst = [[2 * (t // 4), 2 * (t // 4) + 1] for t in range(32)]
        b = burst_stats(read_manifest(make("burst", {0: burst}, n_expert=16)), {0: set()}, 4)
        assert all(abs(p - 0.125) < 1e-9 for p in b["phi"]), b["phi"]
        assert b["ratios"] == [8.0] * 16 and b["thresh"][4] == b["slots"], (b["ratios"], b["thresh"])
        # the cap: an expert whose rate is past phi W reads 1, not a power of a negative base — one at
        # 8 a window and 16 at 0.5 over W 4 touching 2 solve phi + 16 phi = 2 with every rate capped
        assert abs(solve_phi([8.0] + [0.5] * 16, 4, 2.0) - 2 / 17) < 1e-9
        # experts 0 and 1 on the card: window 0 touches no host expert, window 1 two
        b = burst_stats(read_manifest(make("burst2", {0: burst}, n_expert=16)), {0: {0, 1}}, 4)
        assert b["touched"][0] == 0 and b["touched"][1] == 2
        try:
            read_manifest(make("x", {0: uniform}, complete=False))
            raise AssertionError("an incomplete set was accepted")
        except SetError:
            pass

        # the oracle comparison: the logical twin is the right one, and one flipped id is a FAIL
        o = os.path.join(tmp, "oracle")
        os.mkdir(o)
        rows = [[3, 5], [1, 0], [7, 2]]
        T = len(rows)
        flat = [3, 5, 6, 4, 1, 0]  # a view's flat head: token 0's sorted row, not each token's top-k
        with open(os.path.join(o, "ffn_moe_topk-0.0.logical.i32"), "wb") as f:
            f.write(struct.pack(f"<{T * K}i", *[e for r in rows for e in r]))
        with open(os.path.join(o, "ffn_moe_topk-0.0.i32"), "wb") as f:
            f.write(struct.pack(f"<{T * K}i", *flat))
        with open(os.path.join(o, "MANIFEST.tsv"), "w") as f:
            f.write(f"# model\t{MODEL}\n# model_file\tm.gguf\n# build\tb\n# tokens\t0,1,2\n")
            f.write("# kind\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top\tcontig\tlogical"
                    "\tsrc0\tsrc1\n")
            f.write("# int\tname\toccurrence\tof\ttype\ttwin\tlayout\tcount\tbytes\tsum\tabsmax\tfile\n")
            f.write(f"tensor\tffn_moe_topk-0\t0\ti32\t{K}\t{T}\t1\t1\t24\t0\tVIEW\t0\t1\tp (sort)\t-\n")
            f.write("int\tffn_moe_topk-0\t0\ttensor\ti32\ti32\tflat\t6\t24\t0\t6\tffn_moe_topk-0.0.i32\n")
            f.write("int\tffn_moe_topk-0\t0\ttensor\ti32\ti32\tlogical\t6\t24\t0\t7\tffn_moe_topk-0.0.logical.i32\n")
            f.write("# complete\t1\t0\n")
        devnull = open(os.devnull, "w")
        saved, sys.stdout = sys.stdout, devnull
        try:
            good = oracle(make("og", {0: rows}), o)
            bad = oracle(make("ob", {0: [[3, 5], [0, 1], [7, 2]]}), o)
            # identity is the first shard's full path: another file whose shards share the basename, or a
            # set that states no path, is refused by name
            for name, model in (("other", "/models/q3-requant/m-00001-of-00002.gguf"), ("nomodel", None)):
                try:
                    oracle(make(name, {0: rows}, model=model), o)
                    raise AssertionError(f"{name}: a set of another model file was accepted")
                except SetError as e:
                    assert "model" in str(e), e
            # the report commands run end to end on a two-layer set
            two = make("two", {0: uniform, 3: skew})
            coverage([two])
            coverage([two], (1, E))
            coverage([two], (1,), layer_list("3"))
            assert layer_list("0-2,5") == [0, 1, 2, 5]
            try:
                coverage([two], (1,), [1])
                raise AssertionError("a layer the set lacks was accepted")
            except SetError:
                pass
            try:
                coverage([two], (E + 1,))
                raise AssertionError("a hot-n past the expert count was accepted")
            except SetError:
                pass
            transfer(two, two, 2)
            assert compare(two, two) is None
            # a prefix: the same first four tokens, then a differing tail and a longer set
            head = [[0, 1], [2, 3], [4, 5], [6, 7]]
            short = make("short", {0: head + [[0, 1]] * 2}, chunk=4)
            long_ = make("long", {0: head + [[7, 6], [5, 4], [3, 2]]}, chunk=4)
            other = make("chunk2", {0: head + [[0, 1]] * 3}, chunk=2)
            assert compare(short, long_, 4) is None
            assert compare(short, long_, 5) == 0
            for args, why in (((short, long_), "unequal token counts without --tokens"),
                              ((short, long_, 7), "--tokens past the shorter set"),
                              ((short, other, 4), "two chunk sizes")):
                try:
                    compare(*args)
                    raise AssertionError(f"compare accepted {why}")
                except SetError:
                    pass
            bursts([two], 2)
            # the held-out split sits on a chunk edge: 10 tokens in chunks of 4 split at 4, not at 5. Tokens
            # 0-3 touch every expert once (the hot-2 list ties to {0, 1}); token 4 onward all pick {2, 3}.
            # Split at 4, the list {0, 1} serves none of the rest; split at 5 it would learn {2, 3} and serve all.
            edge = make("edge", {0: [[0, 1], [2, 3], [4, 5], [6, 7]] + [[2, 3]] * 6}, chunk=4)
            buf = io.StringIO()
            with redirect_stdout(buf):
                coverage([edge], (2,))
            held_rows = buf.getvalue().split("held-out")[1]
            assert "| 0 | 0.0 |" in held_rows and "share on [4, 10)" in buf.getvalue(), buf.getvalue()
            # predict: layer 1 picks the pair after layer 0's, so co-occurrence names it; the counts are flat,
            # so frequency's list is ids 0, 1 (ties to the lower id) and serves only the pair 0, 1. Card {0, 1},
            # pool {0..3}: the held-out pairs {2, 3}, {4, 5}, {6, 7} miss the card, {2, 3} is RAM, {4..7} NVMe
            pairs = [[2 * (t % 4), 2 * (t % 4) + 1] for t in range(16)]
            nxt = make("next", {0: pairs, 1: [[(a + 2) % E, (b + 2) % E] for a, b in pairs]})
            buf = io.StringIO()
            with redirect_stdout(buf):
                predict(nxt, (1, 2), tiers=[tier_spec("n2,n4")])
            out = buf.getvalue()
            for row in ("| ceiling | 50.0 | 100.0 |", "| freq | 12.5 | 25.0 |", "| cooccur | 50.0 | 100.0 |",
                        "| card miss | 75.0 | freq | 16.7 | 25.0 | 33.3 | 25.0 |",
                        "| card miss | 75.0 | cooccur | 50.0 | 75.0 | 100.0 | 75.0 |",
                        "| RAM | 25.0 | cooccur | 50.0 | 25.0 | 100.0 | 25.0 |",
                        "| NVMe | 50.0 | freq | 25.0 | 25.0 | 50.0 | 25.0 |",
                        "| NVMe | 50.0 | cooccur | 50.0 | 50.0 | 100.0 | 50.0 |"):
                assert row in out, (row, out)
            # the id-prefix card p2 is {0, 1} here too; pool n7 leaves expert 7 the one NVMe candidate, so each
            # position starts one read, not two: it serves both NVMe selections (7 in {6, 7}) of 8 reads
            buf = io.StringIO()
            with redirect_stdout(buf):
                predict(nxt, (2,), tiers=[tier_spec("p2,n7")])
            assert "| NVMe | 12.5 | cooccur | 100.0 | 25.0 |" in buf.getvalue(), buf.getvalue()
            # pool lru4: card {0, 1} and an LRU of two seeded {2, 3}; the held pairs {2, 3} (RAM, at token 8),
            # then {4, 5}, {6, 7}, {2, 3}, {4, 5}, {6, 7} each find the LRU holding the pair before them (NVMe)
            buf = io.StringIO()
            with redirect_stdout(buf):
                predict(nxt, (1, 2), tiers=[tier_spec("n2,lru4")])
            for row in ("| RAM | 12.5 | cooccur | 50.0 | 12.5 | 100.0 | 12.5 |",
                        "| NVMe | 62.5 | cooccur | 50.0 | 62.5 | 100.0 | 62.5 |"):
                assert row in buf.getvalue(), (row, buf.getvalue())
            twice = make("twice", {0: pairs, 1: [[a, a] for a, _ in pairs]})
            for d, kw, why in ((nxt, {"positions": "gen"}, "--positions gen without a contexts.tsv"),
                               (nxt, {"tiers": [tier_spec("n2,n4")] * 2}, "one --tiers twice"),
                               (twice, {}, "a position that selects an expert twice")):
                try:
                    predict(d, (2,), **kw)
                    raise AssertionError(f"predict accepted {why}")
                except SetError:
                    pass
            try:
                tier_spec("n4,n2")
                raise AssertionError("a pool smaller than the card was accepted")
            except SetError:
                pass
        finally:
            sys.stdout = saved
            devnull.close()
        assert good == 0 and bad == 1, (good, bad)
    print("router-coverage: self-test ok")
    return 0


def main(argv):
    if argv[:1] == ["--self-test"]:
        return self_test()
    try:
        if len(argv) >= 2 and argv[0] == "locality":
            locality(argv[1:])
            return 0
        if len(argv) >= 2 and argv[0] == "coverage":
            sets, opts = [], {}
            it = iter(argv[1:])
            for a in it:
                if a in ("--n", "--layers"):
                    opts[a] = next(it, "")
                else:
                    sets.append(a)
            asked = tuple(int(n) for n in opts["--n"].split(",")) if "--n" in opts else None
            keep = layer_list(opts["--layers"]) if "--layers" in opts else None
            coverage(sets, asked, keep)
            return 0
        if len(argv) in (3, 5) and argv[0] == "transfer":
            n = int(argv[4]) if len(argv) == 5 and argv[3] == "--n" else 64
            transfer(argv[1], argv[2], n)
            return 0
        if argv[:1] == ["compare"] and len(argv) in (3, 5):
            if len(argv) == 5 and argv[3] != "--tokens":
                raise SetError(f"compare: unknown flag {argv[3]}")
            compare(argv[1], argv[2], int(argv[4]) if len(argv) == 5 else None)
            return 0
        if len(argv) == 3 and argv[0] == "oracle":
            return oracle(argv[1], argv[2])
        if len(argv) >= 2 and argv[0] == "predict":
            sets, opts, tiers = [], {}, []
            it = iter(argv[1:])
            for a in it:
                if a in ("--n", "--layers", "--positions", "--genre", "--tiers"):
                    v = next(it, None)
                    if v is None:
                        raise SetError(f"{a} takes a value")
                    if a == "--tiers":
                        tiers.append(tier_spec(v))
                    else:
                        opts[a] = v
                elif a.startswith("-"):
                    raise SetError(f"predict: unknown flag {a}")
                else:
                    sets.append(a)
            if len(sets) != 1 or opts.get("--positions", "all") not in ("all", "gen"):
                raise SetError("predict: one set, and --positions all or gen")
            predict(sets[0], tuple(int(n) for n in opts.get("--n", "4,8,16,32").split(",")),
                    layer_list(opts["--layers"]) if "--layers" in opts else None, opts.get("--positions", "all"),
                    opts.get("--genre"), tiers)
            return 0
        if len(argv) >= 2 and argv[0] == "bursts":
            sets, opts = [], {}
            it = iter(argv[1:])
            for a in it:
                if a in ("--window", "--n-l", "--layers"):
                    if a in opts:
                        raise SetError(f"{a} supplied more than once")
                    v = next(it, None)
                    if v is None:
                        raise SetError(f"{a} takes a value")
                    opts[a] = v
                elif a.startswith("-"):
                    raise SetError(f"bursts: unknown flag {a}")
                else:
                    sets.append(a)
            window = int(opts.get("--window", 512))
            if window < 1 or not sets:
                raise SetError("bursts: a set and a window of at least one position")
            keep = layer_list(opts["--layers"]) if "--layers" in opts else None
            bursts(sets, window, opts.get("--n-l"), keep)
            return 0
    except SetError as e:
        print(f"router-coverage: {e}", file=sys.stderr)
        return 1
    print(__doc__, file=sys.stderr)
    return 64


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
