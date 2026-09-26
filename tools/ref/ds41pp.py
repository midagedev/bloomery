#!/usr/bin/env python3
"""The V4.1 prompt batch under Nsight: one owner of the batch's cut for both profilers.

  ds41pp.py tables <sqlite> <P> <n> <run log> [--layer L] [--blocked-us U]
      the nsys prefill form's tables (tools/ref/nsys-ds41.sh, BLOOMERY_NSYS_FORM=prefill, and its
      --analyze): the prompt window, the layer-batches, their kernel terms, the card and host
      timelines and the launch queue
  ds41pp.py ncu-plan <sqlite> <P> <run log> <plan out> [--layer L]
      the ncu ds41pp form's target (tools/ref/ncu-gpu.sh, BLOOMERY_NCU_FORM=ds41pp): the four
      projections of one full chunk, the launch skip that reaches them and the shapes they must have
  ds41pp.py ncu-summary <csv> <plan> <model>
      the shape check of the profiled launches, then each unit's demand next to the block-step

The trace is `generate_ds41 --depth P -n N --time` under `nsys profile -t cuda --cuda-graph-trace=node`
(the prefill form's command). What it holds and how it is cut:
  - the prompt window: every kernel with no graph node (the batch runs eagerly and refuses a capture;
    the capture before it executes nothing), from the first to the first graph replay, which is the
    first generated step's;
  - the batches and chunks: body/prefill.rs `batches` (ceil(P / T_MAX) runs of near-equal length) and
    `chunks` (cut at multiples of CHUNK); checked against the trace — one `gather_pairs` a chunk per
    batch, one `ds41_hc_post` a full chunk per layer-batch;
  - a layer-batch: a layer's block in one batch, the launches from the previous one's join to its own
    join (`ds41_ffn_post_batch*`); its route ends at its `ds41_ffn_places`, the route's three copies to
    the host and their event; the shadow runs from there to the serve's event sync; the post is the
    host sums' copy and the join. Which layer each is comes from the run's `stat prefill ced=` line
    (each layer's block and latent starts), and the count must equal the `stat prefill split` line's
    `layer_batches`, or no table;
  - inside the attention, a full chunk runs from its fork (the event record and wait before
    `ds41_hc_pre`) to its `ds41_hc_post`; the launches before the first full chunk are the engram step
    (its rows, gate and fold) and the latent-only chunks.
Order is launch order: the main thread's API calls by start time, each tied to its card activity by
correlation id — the fork stream's HC_PRE runs beside the main stream, so card start order is not it.

Microseconds here are not records: the API trace adds host cost to every call, and ncu serializes and
replays kernels at fixed clocks. The tables are shares, counts and cycle ratios.

Exit status: 0 with tables; 3 a named refusal (a boundary, a skip or a shape that does not match);
64 a usage error.
"""
import bisect
import csv
import re
import sqlite3
import statistics
import struct
import sys
from collections import Counter, defaultdict

# body/prefill.rs: T_MAX = UNION_MAX_COLS (model::moe), CHUNK = HC_MAX_TOKENS (gpu-deepseek41 hc.rs).
T_MAX = 512
CHUNK = 8
# The queue's plateau: a call within FULL_MARGIN activities of the most in flight, after which at
# least FULL_LONG_SHARE of the calls block.
FULL_MARGIN = 16
FULL_LONG_SHARE = 0.1

PROLOGUE = {"gather_pairs", "ds41_glue_embed", "ds41_glue_embed_q3k"}
ENGRAM_ROWS = {"ds41_glue_engram_rows", "ds41_glue_engram_rows_q3k"}
JOIN = {"ds41_ffn_post_batch", "ds41_ffn_post_batch_streams"}
PLACES = "ds41_ffn_places"
HC_PRE = "ds41_hc_pre"
HC_POST = "ds41_hc_post"
FOLD = "ds41_hc_fold"
PROJ_ROW = "q3k_gemv"
PROJ_HEADS = "ds41_q3k_gemv_heads_mcol"
ROPE = "ds41_rope_tail"
ATTN = {"ds41_attn_seg", "ds41_attn_seg_sel", "ds41_attn_seg_stage", "ds41_attn_seg_sel_stage",
        "ds41_attn_merge", "ds41_ring_commit", "ds41_kv_norm_rope_append", "kv_norm_rope_append"}
SMALL = {"norm_quant", "rms_norm", "q3k_quantize_q8_1", "ds41_rows_to_tokens",
         ROPE, HC_PRE, "ds41_hc_pre_f32", HC_POST}
SOURCE = {"ds41_comp_pool", "ds41_comp_rows", "ds41_index_key", "ds41_indexer_score", "ds41_indexer_topk"}
ROUTE_TAIL = {"ds41_router", "ds41_router_scores", "ds41_router_pick", PLACES}
# The terms of a layer-batch, in print order. `projections` are the chunk's four (qkv, q_b, wo_a,
# wo_b); a chunk's other q3k_gemv (a compressor's joined kv·gate) goes to `source + indexer`.
TERMS = ("projections", "attention", "small", "source + indexer", "latent-only chunks", "engram",
         "route norm_quant", "route tail", "other")
ROLES = ("qkv", "q_b", "wo_a", "wo_b")

MEMCPY_KIND = {1: "H2D", 2: "D2H", 8: "D2D"}
ENQUEUE = re.compile(r"^cu(LaunchKernel|LaunchCooperativeKernel|Memcpy\w*Async|Memset\w*Async|"
                     r"EventRecord|StreamWaitEvent|StreamBatchMemOp|StreamWaitValue|StreamWriteValue|"
                     r"LaunchHostFunc|GraphLaunch)")
SYNC = re.compile(r"^cu(EventSynchronize|StreamSynchronize|CtxSynchronize)")


def refuse(msg):
    print(msg)
    raise SystemExit(3)


# ---- the run's own lines ----

def read_runlog(path):
    r = {"lines": []}
    try:
        text = open(path).read().splitlines()
    except OSError:
        text = []
    for line in text:
        m = re.match(r"time prompt n=(\d+) ms=([0-9.]+) tok/s=\S+ passes=(\d+) kind=(\w+)", line)
        if m:
            r["p_n"], r["p_ms"], r["passes"], r["kind"] = int(m[1]), float(m[2]), int(m[3]), m[4]
            r["lines"].append(line)
        m = re.match(r"load .*\blayers=(\d+)\b.*\bprefill=(\w+)", line)
        if m:
            r["n_layer"], r["prefill"] = int(m[1]), m[2]
            r["lines"].append(line)
        m = re.match(r"stat prefill split batches=(\d+) layer_batches=(\d+) .*union_lb=([0-9.]+) "
                     r"wait_lb=([0-9.]+) enqueue_lb=([0-9.]+) copy_lb=([0-9.]+)", line)
        if m:
            r["batches"], r["layer_batches"] = int(m[1]), int(m[2])
            r["lb"] = {"union": float(m[3]), "wait": float(m[4]), "enqueue": float(m[5]), "copy": float(m[6])}
            r["lines"].append(line)
        m = re.match(r"stat prefill ced=.*\bfirst=(\d+) end=(\d+) .*full_from=\[([0-9,]*)\] part_from=\[([0-9,]*)\]", line)
        if m:
            r["ced_first"], r["ced_end"] = int(m[1]), int(m[2])
            r["full_from"] = [int(x) for x in m[3].split(",") if x]
            r["part_from"] = [int(x) for x in m[4].split(",") if x]
            r["lines"].append(line)
        if line.startswith(("plan ", "capture ", "fed ", "step 0 ", "SMOKE ")):
            r["lines"].append(line)
        if line.startswith("plan "):
            r["plan"] = line
    return r


def batches(first, n):
    """body/prefill.rs `batches`: ceil(n / T_MAX) runs, the first n mod k one position longer."""
    k = -(-n // T_MAX)
    out, p = [], first
    for j in range(k):
        ln = n // k + (1 if j < n % k else 0)
        out.append((p, p + ln))
        p += ln
    return out


def chunks(b, e):
    """body/prefill.rs `chunks`: cut at every multiple of CHUNK."""
    out, p = [], b
    while p < e:
        nxt = min((p // CHUNK + 1) * CHUNK, e)
        out.append((p, nxt))
        p = nxt
    return out


def batch_plan(run):
    """Per layer-batch in launch order: (batch, layer, full chunks, latent chunks, the full chunks'
    lengths), from the batch cut, the chunk cut and each layer's block and latent starts."""
    need = ("full_from", "part_from", "ced_first", "ced_end", "n_layer", "batches", "layer_batches")
    missing = [k for k in need if k not in run]
    if missing:
        refuse(f"[boundary] the run log lacks {', '.join(missing)} (`load`, `stat prefill ced=`, "
               f"`stat prefill split` lines): the layer-batches cannot be named, no table")
    full, part, n_layer = run["full_from"], run["part_from"], run["n_layer"]
    if len(full) != n_layer or len(part) != n_layer:
        refuse(f"[boundary] the ced line names {len(full)} block and {len(part)} latent starts for "
               f"{n_layer} layers: no table")
    runs = batches(run["ced_first"], run["ced_end"] - run["ced_first"])
    if len(runs) != run["batches"]:
        refuse(f"[boundary] ceil(P / {T_MAX}) = {len(runs)} batches, the run's stat line says "
               f"{run['batches']}: T_MAX moved, no table")
    plan = []
    for bi, (b, e) in enumerate(runs):
        cuts = chunks(b, e)
        for layer in range(n_layer):
            f = [c for c in cuts if c[0] >= full[layer]]
            p = [c for c in cuts if part[layer] <= c[0] < full[layer]]
            if f:
                plan.append({"batch": bi, "layer": layer, "full": len(f), "part": len(p),
                             "m": [c[1] - c[0] for c in f], "cuts": len(cuts)})
    if len(plan) != run["layer_batches"]:
        refuse(f"[boundary] the ced line gives {len(plan)} layer-batches, the stat line says "
               f"{run['layer_batches']}: no table")
    return plan, runs


# ---- the trace ----

class Trace:
    def __init__(self, path):
        db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        names = dict(db.execute("SELECT id, value FROM StringIds"))

        def short(n):
            return names.get(n, str(n)).split("(")[0]

        self.kernels = [dict(s=s, e=e, name=short(n), corr=c, stream=st, dev=dv, grid=(gx, gy, gz),
                             block=(bx, by, bz), graph=g)
                        for s, e, n, c, st, dv, gx, gy, gz, bx, by, bz, g in db.execute(
                            "SELECT start, end, shortName, correlationId, streamId, deviceId, gridX, gridY, "
                            "gridZ, blockX, blockY, blockZ, graphNodeId FROM CUPTI_ACTIVITY_KIND_KERNEL")]
        self.copies = []
        for table, kind in (("CUPTI_ACTIVITY_KIND_MEMCPY", "copyKind"), ("CUPTI_ACTIVITY_KIND_MEMSET", "0")):
            try:
                rows = db.execute(f"SELECT start, end, {kind}, bytes, correlationId, streamId, deviceId, "
                                  f"graphNodeId FROM {table}").fetchall()
            except sqlite3.OperationalError:
                rows = []
            for s, e, k, b, c, st, dv, g in rows:
                name = MEMCPY_KIND.get(k, "memset" if table.endswith("MEMSET") else f"copy{k}")
                self.copies.append(dict(s=s, e=e, name=name, bytes=b, corr=c, stream=st, dev=dv, graph=g))
        self.api = [dict(s=s, e=e, name=names.get(n, str(n)), corr=c, tid=t)
                    for s, e, n, c, t in db.execute(
                        "SELECT start, end, nameId, correlationId, globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME "
                        "ORDER BY start")]
        self.gpu_by_corr = defaultdict(list)
        for a in self.kernels:
            self.gpu_by_corr[a["corr"]].append(("kernel", a))
        for a in self.copies:
            self.gpu_by_corr[a["corr"]].append((a["name"], a))


def prompt_window(tr, run):
    """The eager kernels (the batch) and the first replay's: refusals when the trace is not a prompt."""
    eager = sorted((k for k in tr.kernels if k["graph"] is None), key=lambda k: k["s"])
    graph = sorted((k for k in tr.kernels if k["graph"] is not None), key=lambda k: k["s"])
    if not eager:
        refuse("[boundary] no kernel outside a graph: this trace holds no prompt batch (a decode-form "
               "trace, or BLOOMERY_PREFILL=steps), no table")
    if run.get("kind") != "batch" or run.get("prefill") != "batch":
        refuse(f"[boundary] the run fed its prompt as kind={run.get('kind')} (load prefill="
               f"{run.get('prefill')}): the prefill form reads the batch feed only, no table")
    after = [k for k in graph if k["s"] > eager[-1]["s"]]
    if not after:
        refuse("[boundary] no graph replay after the prompt's kernels: the window has no end "
               "(the form runs -n >= 2 --mode graph), no table")
    if any(k["s"] < eager[-1]["s"] for k in graph):
        refuse("[boundary] a graph replay runs before the prompt's last eager kernel: not one prompt "
               "window, no table")
    return eager, after[0]


def entries(tr, eager, replay):
    """The main thread's API calls from the prompt's first launch to the first replay's launch."""
    by_corr = {a["corr"]: a for a in tr.api}
    first = by_corr.get(eager[0]["corr"])
    rl = by_corr.get(replay["corr"])
    if first is None or rl is None:
        refuse("[boundary] the prompt's first kernel or the first replay has no API call in the trace "
               "(the CUDA API trace is off?), no table")
    tid = first["tid"]
    tids = Counter(by_corr[k["corr"]]["tid"] for k in eager if k["corr"] in by_corr)
    if len(tids) != 1:
        refuse(f"[boundary] the prompt's kernels come from {len(tids)} threads: the launch order is "
               f"not one thread's, no table")
    out = []
    for a in tr.api:
        if a["tid"] != tid or a["s"] < first["s"] or a["s"] >= rl["s"]:
            continue
        acts = tr.gpu_by_corr.get(a["corr"], [])
        act = acts[0][1] if acts else None
        kname = act["name"] if act is not None else None
        out.append(dict(api=a["name"], s=a["s"], e=a["e"], corr=a["corr"], act=act,
                        kernel=kname if acts and acts[0][0] == "kernel" else None,
                        copy=acts[0][0] if acts and acts[0][0] != "kernel" else None,
                        enq=bool(ENQUEUE.match(a["name"])), sync=bool(SYNC.match(a["name"]))))
    return out


def cut_layer_batches(en, plan, runs):
    """Split the window's entries into the batches' prologues, the layer-batches and the head."""
    joins = [i for i, x in enumerate(en) if x["kernel"] in JOIN]
    places = [i for i, x in enumerate(en) if x["kernel"] == PLACES]
    print(f"[boundary] kernel {PLACES}: {len(places)} launches, joins ({'/'.join(sorted(JOIN))}) "
          f"{len(joins)}, expected layer_batches = {len(plan)} (the run's stat and ced lines) -> "
          f"{'OK' if len(places) == len(joins) == len(plan) else 'MISMATCH'}")
    if not (len(places) == len(joins) == len(plan)):
        refuse("    no table: the layer-batch cut does not match the run")
    lbs, start = [], 0
    for k, (p, j) in enumerate(zip(places, joins)):
        if not start <= p < j:
            refuse(f"[boundary] layer-batch {k}: its places launch is not before its join, no table")
        end = j
        if end + 1 < len(en) and en[end + 1]["kernel"] == "ds41_tap_means":
            end += 1
        lbs.append(dict(plan[k], lo=start, places=p, join=j, hi=end))
        start = end + 1
    # A batch's prologue: its chunks' gathers and embeddings before its first layer.
    first_of = {}
    for k, lb in enumerate(lbs):
        first_of.setdefault(lb["batch"], k)
    for bi, k in first_of.items():
        lb = lbs[k]
        g = [i for i in range(lb["lo"], lb["places"]) if en[i]["kernel"] in PROLOGUE]
        gathers = sum(1 for i in g if en[i]["kernel"] == "gather_pairs")
        if gathers != lb["cuts"]:
            refuse(f"[boundary] batch {bi}: {gathers} gather_pairs launches, the chunk cut gives "
                   f"{lb['cuts']} chunks (CHUNK moved?), no table")
        lb["prologue"] = (lb["lo"], g[-1] + 1)
        lb["lo"] = g[-1] + 1
    head = (lbs[-1]["hi"] + 1, len(en))
    return lbs, head


def phases(en, lb):
    """Tag the layer-batch's entries: engram, latent, full chunk k, route, d2h, shadow, sync, post."""
    lo, p, j, hi = lb["lo"], lb["places"], lb["join"], lb["hi"]
    posts = [i for i in range(lo, p) if en[i]["kernel"] == HC_POST]
    if len(posts) != lb["full"]:
        refuse(f"[boundary] layer {lb['layer']} batch {lb['batch']}: {len(posts)} {HC_POST} launches, "
               f"the ced line gives {lb['full']} full chunks, no table")
    route0 = posts[-1] + 1
    tag = {}
    # Full chunks: from the fork before each HC_PRE to its HC_POST.
    pres = [i for i in range(lo, route0) if en[i]["kernel"] == HC_PRE]
    if len(pres) != lb["full"]:
        refuse(f"[boundary] layer {lb['layer']} batch {lb['batch']}: {len(pres)} {HC_PRE} launches in "
               f"the attention for {lb['full']} full chunks, no table")
    chunk_lo = []
    for k, i in enumerate(pres):
        s = i
        while s - 1 >= lo and en[s - 1]["api"].startswith(("cuEventRecord", "cuStreamWaitEvent")) \
                and (k == 0 or s - 1 > posts[k - 1]):
            s -= 1
        chunk_lo.append(s)
    for k in range(lb["full"]):
        for i in range(chunk_lo[k], posts[k] + 1):
            tag[i] = ("chunk", k)
    # Before the first full chunk: the engram step (rows ... fold, chunk after chunk), then latent parts.
    i, engram_end = lo, lo
    if lo < chunk_lo[0] and en[lo]["kernel"] in ENGRAM_ROWS:
        last_fold = None
        for q in range(lo, chunk_lo[0]):
            if en[q]["kernel"] == FOLD:
                last_fold = q
        if last_fold is None:
            refuse(f"[boundary] layer {lb['layer']}: an engram step with no {FOLD}, no table")
        engram_end = last_fold + 1
    for i in range(lo, chunk_lo[0]):
        tag[i] = ("engram", 0) if i < engram_end else ("latent", 0)
    for i in range(route0, p + 1):
        tag[i] = ("route", 0)
    # After places: the route's copies to the host and their event, then the shadow up to the sync.
    q = p + 1
    while q < j and (en[q]["copy"] == "D2H" or (en[q]["api"].startswith("cuEventRecord")
                                                 and q > p + 1 and en[q - 1]["copy"] == "D2H")):
        tag[q] = ("d2h", 0)
        q += 1
    d2h = [i for i in range(p + 1, q) if en[i]["copy"] == "D2H"]
    if len(d2h) != 3:
        refuse(f"[boundary] layer {lb['layer']} batch {lb['batch']}: {len(d2h)} copies to the host after "
               f"{PLACES}, the route hands off three, no table")
    syncs = [i for i in range(q, j) if en[i]["sync"]]
    if not syncs:
        refuse(f"[boundary] layer {lb['layer']} batch {lb['batch']}: no event sync between the route and "
               f"the join (the serve's wait), no table")
    s = syncs[0]
    for i in range(q, s):
        tag[i] = ("shadow", 0)
    tag[s] = ("sync", 0)
    for i in range(s + 1, hi + 1):
        tag[i] = ("post", 0)
    lb.update(tag=tag, chunk_lo=chunk_lo, posts=posts, route0=route0, d2h=d2h, sync=s,
              engram=(lo, engram_end), latent=(engram_end, chunk_lo[0]))
    h2d = [i for i in range(s + 1, hi + 1) if en[i]["copy"] == "H2D"]
    lb["h2d"] = h2d[0] if h2d else None
    return lb


def chunk_roles(en, lb, k):
    """The four projections of full chunk k: wo_a is its heads launch, wo_b the q3k_gemv after it,
    q_b the last q3k_gemv before its first rope tail, qkv the q3k_gemv before q_b. Others: None."""
    idx = [i for i in range(lb["chunk_lo"][k], lb["posts"][k] + 1)]
    heads = [i for i in idx if en[i]["kernel"] == PROJ_HEADS]
    rope = [i for i in idx if en[i]["kernel"] == ROPE]
    gem = [i for i in idx if en[i]["kernel"] == PROJ_ROW]
    if len(heads) != 1 or not rope:
        return None
    wob = [i for i in gem if i > heads[0]]
    qb = [i for i in gem if i < rope[0]]
    if not wob or len(qb) < 2:
        return None
    return {"qkv": qb[-2], "q_b": qb[-1], "wo_a": heads[0], "wo_b": wob[0]}


def term_of(en, lb, i, roles):
    ph, k = lb["tag"].get(i, ("?", 0))
    name = en[i]["kernel"] or en[i]["copy"]
    if ph == "chunk":
        if name in (PROJ_ROW, PROJ_HEADS):
            r = roles.get(k)
            if r is not None and i in r.values():
                return "projections"
            return "source + indexer" if name == PROJ_ROW else "projections"
        if name in ATTN:
            return "attention"
        if name in SMALL:
            return "small"
        if name in SOURCE:
            return "source + indexer"
        return "other"
    if ph == "latent":
        return "latent-only chunks"
    if ph == "engram":
        return "engram"
    if ph == "route":
        if name == "norm_quant":
            return "route norm_quant"
        return "route tail" if name in ROUTE_TAIL else "other"
    if ph == "d2h":
        return "route tail"
    return ph


class Busy:
    """The card's activity intervals, for the length of their union inside a window."""

    def __init__(self, iv):
        self.iv = sorted(iv)
        self.starts = [s for s, _ in self.iv]
        self.longest = max((e - s for s, e in self.iv), default=0)

    def union(self, lo, hi):
        """The length of the union of the intervals clipped to [lo, hi)."""
        a = bisect.bisect_left(self.starts, lo - self.longest)
        b = bisect.bisect_left(self.starts, hi)
        tot, cur_s, cur_e = 0, None, None
        for s, e in self.iv[a:b]:
            s, e = max(s, lo), min(e, hi)
            if s >= e:
                continue
            if cur_e is None or s > cur_e:
                if cur_e is not None:
                    tot += cur_e - cur_s
                cur_s, cur_e = s, e
            else:
                cur_e = max(cur_e, e)
        if cur_e is not None:
            tot += cur_e - cur_s
        return tot


def measure(en, lbs, head, busy_of, blocked_ns):
    """Per layer-batch: the terms, the card windows and the host and queue numbers."""
    for n_, lb in enumerate(lbs):
        roles = {}
        for k in range(lb["full"]):
            r = chunk_roles(en, lb, k)
            if r is not None:
                roles[k] = r
        lb["roles"] = roles
        lb["irregular"] = lb["full"] - len(roles)
        terms = defaultdict(float)
        by_kernel = defaultdict(lambda: [0, 0.0, ""])
        for i in range(lb["lo"], lb["hi"] + 1):
            a = en[i]["act"]
            if a is None:
                continue
            t = term_of(en, lb, i, roles)
            d = (a["e"] - a["s"]) / 1e6
            name = en[i]["kernel"] or en[i]["copy"]
            terms[t] += d
            key = (t, name)
            by_kernel[key][0] += 1
            by_kernel[key][1] += d
        lb["terms"], lb["by_kernel"] = dict(terms), dict(by_kernel)
        acts = [en[i]["act"] for i in range(lb["lo"], lb["hi"] + 1) if en[i]["act"] is not None]
        t0 = min(a["s"] for a in acts)
        t_route = max(en[i]["act"]["e"] for i in lb["d2h"])
        t_h2d = en[lb["h2d"]]["act"]["s"] if lb["h2d"] is not None else t_route
        nxt = lbs[n_ + 1] if n_ + 1 < len(lbs) else None
        if nxt is not None:
            t_next = min(en[i]["act"]["s"] for i in range(nxt["lo"], nxt["hi"] + 1) if en[i]["act"] is not None)
        else:
            ha = [en[i]["act"]["s"] for i in range(head[0], head[1]) if en[i]["act"] is not None]
            t_next = min(ha) if ha else max(a["e"] for a in acts)
        wall = (t_route - t0) / 1e6
        busy = busy_of.union(t0, t_route) / 1e6
        ksum = sum(v for t, v in terms.items() if t in TERMS)
        shadow_sum = terms.get("shadow", 0.0)
        gap = (t_h2d - t_route) / 1e6
        gap_busy = busy_of.union(t_route, t_h2d) / 1e6
        post = (t_next - t_h2d) / 1e6
        post_busy = busy_of.union(t_h2d, t_next) / 1e6
        lb["card"] = dict(t0=t0, wall=wall, busy=busy, ksum=ksum, idle=wall - busy, overlap=ksum - busy,
                          gap=gap, shadow=shadow_sum, gap_busy=gap_busy, gap_idle=gap - gap_busy,
                          post=post, post_busy=post_busy)
        # Host: the main thread's calls. Enqueue = everything but the wait and the union's span.
        s = lb["sync"]
        first_api = en[lb["lo"]]["s"]
        sync = en[s]
        h2d_api = en[lb["h2d"]]["s"] if lb["h2d"] is not None else sync["e"]
        post_end = en[lb["hi"]]["e"]
        lb["host"] = dict(enqueue=((sync["s"] - first_api) + (post_end - h2d_api)) / 1e6,
                          wait=(sync["e"] - sync["s"]) / 1e6, union=(h2d_api - sync["e"]) / 1e6,
                          before_sync=(sync["s"] - first_api) / 1e6)
        # The queue. In flight at a call: the activity-bearing calls issued before it minus the
        # activities finished by its start (event records and stream waits leave no activity, so
        # they are counted apart). The queue is full from the first call at the plateau, when the
        # calls after it block often (the driver frees slots in batches, so a full queue shows as a
        # long call every few calls, not every call); a long call before it is driver jitter.
        routed = max(i for i in range(lb["lo"], s) if lb["tag"].get(i, ("?",))[0] in ("d2h", "route"))
        q_ent = [i for i in range(lb["lo"], s) if en[i]["enq"]]
        q_act = [i for i in q_ent if en[i]["act"] is not None]
        n_r = sum(1 for i in q_act if i <= routed)
        n_s = len(q_act) - n_r
        n_p = sum(1 for i in range(s + 1, lb["hi"] + 1) if en[i]["enq"])
        dur = [en[i]["e"] - en[i]["s"] for i in q_ent]
        long_ = [d > blocked_ns for d in dur]
        ends = sorted(en[i]["act"]["e"] for i in q_act)
        inflight, issued = [], 0
        for i in q_ent:
            inflight.append(issued - bisect.bisect_right(ends, en[i]["s"]))
            if en[i]["act"] is not None:
                issued += 1
        q_max = max(inflight) if inflight else 0
        full = next((k for k, v in enumerate(inflight) if v >= q_max - FULL_MARGIN), len(q_ent))
        after = long_[full:]
        saturated = len(after) >= 20 and sum(after) >= FULL_LONG_SHARE * len(after)
        upto = full if saturated else len(q_ent)
        acts_before = sum(1 for i in q_ent[:upto] if en[i]["act"] is not None)
        span = en[q_ent[upto - 1]]["s"] - en[q_ent[0]]["s"] if upto > 1 else 0
        lb["queue"] = dict(n_r=n_r, n_s=n_s, n_ev=len(q_ent) - len(q_act), n_p=n_p, api_sum=sum(dur) / 1e6,
                           q=q_max if saturated else None, full_act=acts_before + 1 if saturated else None,
                           blocked=sum(after) if saturated else 0,
                           blocked_sum=sum(d for d, b in zip(dur[full:], after) if b) / 1e6 if saturated else 0.0,
                           stray=sum(long_[:upto]),
                           h_api=statistics.median(dur[:upto]) / 1e3 if upto else None,
                           h=span / 1e3 / (acts_before - 1) if acts_before > 1 else None,
                           t_c=wall * 1e3 / n_r if n_r else None,
                           enq_wall=(en[q_ent[-1]]["e"] - en[q_ent[0]]["s"]) / 1e6 if q_ent else 0.0,
                           d0=(t0 - en[q_ent[0]]["s"]) / 1e6 if q_ent else 0.0)
        # The model at the measured Q, h and t_c: the queue fills at activity Q / (1 - h / t_c); from
        # there the host issues at the card's pace, so its last call returns when the card has done
        # N - Q activities, and the serve's wait is the route's remaining Q - N_s.
        qd = lb["queue"]
        if saturated and qd["h"] and qd["t_c"] and qd["h"] < qd["t_c"]:
            qd["pred_fill"] = q_max / (1 - qd["h"] / qd["t_c"])
            qd["pred_enq"] = qd["d0"] + (n_r + n_s - q_max) * qd["t_c"] / 1e3
            qd["pred_wait"] = (q_max - n_s) * qd["t_c"] / 1e3
        else:
            qd["pred_fill"] = qd["pred_enq"] = qd["pred_wait"] = None


def fmt_ms(v):
    return f"{v:8.3f}"


def mean_of(xs):
    """The mean of the defined values (None and NaN left out); NaN when none is defined."""
    xs = [x for x in xs if x is not None and x == x]
    return statistics.fmean(xs) if xs else float("nan")


def fmt_opt(v, spec=".0f"):
    """A value, or `none` where it is undefined (no layer-batch had it)."""
    return "none" if v is None or v != v else format(v, spec)


def cmd_tables(argv):
    if len(argv) < 4:
        print(__doc__)
        raise SystemExit(64)
    path, P, n, runlog = argv[0], int(argv[1]), int(argv[2]), argv[3]
    layer_pick = int(opt(argv, "--layer", "2"))
    blocked_ns = float(opt(argv, "--blocked-us", "8")) * 1e3
    run = read_runlog(runlog)
    for line in run["lines"]:
        print("    " + line)
    if P < 9:
        refuse(f"[boundary] P = {P}: the prefill form reads prompts of at least 9 positions, no table")
    tr = Trace(path)
    eager, replay = prompt_window(tr, run)
    if run.get("p_n") != P:
        refuse(f"[boundary] the run's `time prompt` row is n={run.get('p_n')}, the form asked for P={P}: no table")
    plan, runs = batch_plan(run)
    en = entries(tr, eager, replay)
    dev = eager[0]["dev"]
    busy_of = Busy([(k["s"], k["e"]) for k in tr.kernels if k["dev"] == dev] +
                   [(c["s"], c["e"]) for c in tr.copies if c["dev"] == dev])
    lbs, head = cut_layer_batches(en, plan, runs)
    for lb in lbs:
        phases(en, lb)
    measure(en, lbs, head, busy_of, blocked_ns)
    apis = Counter(x["api"] for x in en)
    print(f"    window API calls {len(en)}: " + ", ".join(f"{k} {v}" for k, v in apis.most_common(12)))
    if not any(k.startswith("cuEventRecord") for k in apis):
        print("    (no cuEventRecord or cuStreamWaitEvent in the API trace: the fork, join and route events "
              "are not in the entry counts below)")

    w0, w1 = eager[0]["s"], replay["s"]
    wall = (w1 - w0) / 1e6
    ksum = sum(k["e"] - k["s"] for k in eager) / 1e6
    print()
    print(f"µs and ms under nsys are not numbers of record (the API trace adds host cost to every call); "
          f"they are shares, counts and ratios. P = {P}, {len(runs)} batch(es), {len(lbs)} layer-batches.")
    print(f"=== prompt window: first eager kernel to the first replay {wall:.1f} ms, time prompt "
          f"ms={run.get('p_ms', float('nan')):.1f}; eager kernels {len(eager)}, kernel sum {ksum:.1f} ms")
    hk = [en[i] for i in range(*head) if en[i]["act"] is not None]
    if hk:
        print(f"    head (after the last join): {len(hk)} activities, "
              f"{sum(x['act']['e'] - x['act']['s'] for x in hk) / 1e6:.3f} ms")

    # The per-layer table.
    print()
    print("=== per layer-batch (ms; card from the trace's activities, host from the main thread's calls)")
    print(f"  {'b':>2s} {'lay':>3s} {'full':>4s} {'lat':>3s} {'irr':>3s} | {'route':>7s} {'kern':>7s} {'busy':>7s} "
          f"{'idle':>6s} {'ovl':>5s} | {'gap':>7s} {'shadow':>7s} {'g-idle':>7s} | {'post':>6s} | "
          f"{'h-enq':>7s} {'h-wait':>7s} {'h-union':>7s} | {'N_r':>5s} {'N_s':>4s} {'ev':>4s} | {'Q':>5s} "
          f"{'fill':>5s} {'blk':>4s} {'stray':>5s} | {'t_c us':>6s}")
    for lb in lbs:
        c, h, q = lb["card"], lb["host"], lb["queue"]
        print(f"  {lb['batch']:2d} {lb['layer']:3d} {lb['full']:4d} {lb['part']:3d} {lb['irregular']:3d} | "
              f"{c['wall']:7.2f} {c['ksum']:7.2f} {c['busy']:7.2f} {c['idle']:6.2f} {c['overlap']:5.2f} | "
              f"{c['gap']:7.2f} {c['shadow']:7.2f} {c['gap_idle']:7.2f} | {c['post']:6.3f} | "
              f"{h['enqueue']:7.2f} {h['wait']:7.2f} {h['union']:7.2f} | {q['n_r']:5d} {q['n_s']:4d} {q['n_ev']:4d} | "
              f"{fmt_opt(q['q'], 'd'):>5s} {fmt_opt(q['full_act'], 'd'):>5s} {q['blocked']:4d} {q['stray']:5d} | "
              f"{fmt_opt(q['t_c'], '.2f'):>6s}")
    print("  (route = layer start to the route's last copy to the host; kern = kernel sum in it; busy = the "
          "union of every activity's interval in it, both streams; idle = route - busy; ovl = kern - busy, the "
          "fork stream's overlap; gap = that copy to the host sums' copy, shadow its kernel sum, g-idle the card "
          "idle in it; post = the host sums' copy to the next layer's first activity; h-enq = the host's time "
          "outside the wait and the union; N_r, N_s = the route's and the shadow's calls that carry an activity "
          "(kernels, copies), ev = event records and stream waits; Q = the most activities in flight, when the "
          "queue fills (none: it never did); fill = the activity (1-based) at which it filled; blk = calls "
          "longer than the threshold after that, stray = before; t_c = route / N_r)")

    sel = [lb for lb in lbs if 2 <= lb["layer"] <= 39]
    one = next((lb for lb in lbs if lb["layer"] == layer_pick), None)
    if one is None:
        refuse(f"[boundary] no layer-batch of layer {layer_pick}: no table")
    print()
    print(f"=== kernel table: layer {one['layer']} of batch {one['batch']} ({one['full']} full chunks, "
          f"{one['part']} latent-only, {one['irregular']} chunks without the four projections), ms")
    print(f"  {'term':20s} {'kernel':34s} {'launches':>8s} {'total ms':>9s} {'µs/launch':>9s} {'µs/chunk':>9s}")
    for t in TERMS + ("shadow", "post"):
        rows = sorted(((k[1], v) for k, v in one["by_kernel"].items() if k[0] == t), key=lambda x: -x[1][1])
        if not rows:
            continue
        for name, (c, d, _) in rows:
            print(f"  {t:20s} {name:34s} {c:8d} {d:9.3f} {1e3 * d / c:9.2f} {1e3 * d / max(one['full'], 1):9.2f}")
        tot = sum(v[1] for _, v in rows)
        print(f"  {t + ' (sum)':20s} {'':34s} {sum(v[0] for _, v in rows):8d} {tot:9.3f} {'':9s} "
              f"{1e3 * tot / max(one['full'], 1):9.2f}")
    c = one["card"]
    print(f"  route window {c['wall']:.3f} = kernel sum {c['ksum']:.3f} - fork overlap {c['overlap']:.3f} "
          f"+ card idle {c['idle']:.3f}; union gap {c['gap']:.3f} (shadow {c['shadow']:.3f}, idle "
          f"{c['gap_idle']:.3f}); post {c['post']:.3f}")
    if one["roles"]:
        print("  the four projections, per chunk (µs, mean over the chunks that have them):")
        for r in ROLES:
            ds = [(en[v[r]]["act"]["e"] - en[v[r]]["act"]["s"]) / 1e3 for v in one["roles"].values()]
            g = en[next(iter(one["roles"].values()))[r]]["act"]["grid"]
            print(f"    {r:5s} grid {g[0]:5d}  {statistics.fmean(ds):8.2f} µs (min {min(ds):.2f}, max {max(ds):.2f})")

    for label, group in ((f"layers 2-39 of every batch ({len(sel)} layer-batches)", sel),
                         (f"all {len(lbs)} layer-batches", lbs)):
        print()
        print(f"=== mean per layer-batch over {label}, ms")
        for t in TERMS + ("shadow", "post"):
            v = mean_of([lb["terms"].get(t, 0.0) for lb in group])
            print(f"  {t:22s} {v:8.3f}")
        cm = {k: mean_of([lb["card"][k] for lb in group]) for k in ("wall", "ksum", "busy", "idle", "overlap",
                                                                    "gap", "shadow", "gap_idle", "post")}
        print(f"  route window {cm['wall']:.3f} = kernel sum {cm['ksum']:.3f} - fork overlap "
              f"{cm['overlap']:.3f} + card idle (the gaps) {cm['idle']:.3f}")
        print(f"  union gap {cm['gap']:.3f}: shadow kernels {cm['shadow']:.3f}, card idle {cm['gap_idle']:.3f}; "
              f"post {cm['post']:.3f}; full chunks {mean_of([lb['full'] for lb in group]):.1f}, latent-only "
              f"{mean_of([lb['part'] for lb in group]):.1f}")

    # The queue table.
    print()
    print(f"=== the queue from the host side, mean per layer-batch (enqueue calls: {ENQUEUE.pattern[3:60]}...; "
          f"blocked = a call longer than {blocked_ns / 1e3:.1f} µs)")
    for label, group in (("layers 2-39", sel), ("all", lbs)):
        full_g = [lb for lb in group if lb["queue"]["q"] is not None]
        q = {k: mean_of([lb["queue"][k] for lb in group])
             for k in ("n_r", "n_s", "n_ev", "n_p", "api_sum", "blocked", "blocked_sum", "stray", "h_api", "h",
                       "t_c", "enq_wall", "d0")}
        qf = {k: mean_of([lb["queue"][k] for lb in full_g])
              for k in ("q", "full_act", "pred_fill", "pred_enq", "pred_wait", "enq_wall")}
        wf = mean_of([lb["host"]["wait"] for lb in full_g])
        h = {k: mean_of([lb["host"][k] for lb in group]) for k in ("enqueue", "wait", "union", "before_sync")}
        print(f"  [{label}] N_r {q['n_r']:.0f}  N_s {q['n_s']:.0f}  events {q['n_ev']:.0f}  post {q['n_p']:.0f}; "
              f"the queue fills in {len(full_g)} of {len(group)} layer-batches, at Q {fmt_opt(qf['q'])} activities "
              f"in flight, from activity {fmt_opt(qf['full_act'])}; blocked calls after that {q['blocked']:.0f} "
              f"({q['blocked_sum']:.2f} ms), stray long calls before {q['stray']:.1f}")
        print(f"      host: an unblocked call {fmt_opt(q['h_api'], '.2f')} µs in the driver, one activity every "
              f"{fmt_opt(q['h'], '.2f')} µs issued (h); card: {fmt_opt(q['t_c'], '.2f')} µs a route activity (t_c), "
              f"first activity {q['d0']:.2f} ms after the first call; enqueue calls' wall {q['enq_wall']:.2f} ms, "
              f"their API sum {q['api_sum']:.2f} ms")
        print(f"      host: before the sync {h['before_sync']:.2f}, wait {h['wait']:.2f}, union+copy "
              f"{h['union']:.2f}, enqueue (all but wait and union) {h['enqueue']:.2f}")
        if full_g:
            print(f"      the queue model at the measured Q, h, t_c (the layer-batches that fill): fills at "
                  f"Q / (1 - h / t_c) = {fmt_opt(qf['pred_fill'])} (measured {fmt_opt(qf['full_act'])}); "
                  f"enqueue wall d0 + (N_r + N_s - Q) t_c = {fmt_opt(qf['pred_enq'], '.2f')} (measured "
                  f"{fmt_opt(qf['enq_wall'], '.2f')}); wait (Q - N_s) t_c = {fmt_opt(qf['pred_wait'], '.2f')} "
                  f"(measured {fmt_opt(wf, '.2f')})")
        else:
            print("      the queue never filled")
    if "lb" in run:
        s = run["lb"]
        print(f"  the run's stat line, per layer-batch: enqueue {s['enqueue']:.2f}, wait {s['wait']:.2f}, "
              f"union {s['union']:.2f}, copy {s['copy']:.2f} (host wall clock, under the profiler)")
    return 0


def opt(argv, flag, default):
    return argv[argv.index(flag) + 1] if flag in argv else default


# ---- ncu ----

def cmd_ncu_plan(argv):
    if len(argv) < 4:
        print(__doc__)
        raise SystemExit(64)
    path, P, runlog, out = argv[0], int(argv[1]), argv[2], argv[3]
    want = opt(argv, "--layer", None)
    run = read_runlog(runlog)
    tr = Trace(path)
    eager, replay = prompt_window(tr, run)
    if run.get("p_n") != P:
        refuse(f"[skip] the trace's run fed n={run.get('p_n')}, the form asked for P={P}: no target")
    plan, runs = batch_plan(run)
    en = entries(tr, eager, replay)
    lbs, head = cut_layer_batches(en, plan, runs)
    for lb in lbs:
        phases(en, lb)
    pick = None
    for lb in lbs:
        if lb["layer"] < 2 or (want is not None and lb["layer"] != int(want)):
            continue
        roles = {k: chunk_roles(en, lb, k) for k in range(lb["full"])}
        plain = all(r is not None for r in roles.values()) and lb["engram"][0] == lb["engram"][1]
        srcs = any(en[i]["kernel"] in SOURCE for i in range(lb["lo"], lb["route0"]))
        if want is not None or (plain and not srcs):
            pick = (lb, roles)
            break
    if pick is None:
        refuse(f"[skip] no layer-batch of layer {want if want is not None else '>= 2'} whose full chunks all "
               f"carry the four projections (no engram, no compressor or indexer kernels): no target")
    lb, roles = pick
    k = lb["full"] // 2
    r = roles.get(k)
    if r is None:
        refuse(f"[skip] layer {lb['layer']} chunk {k}: the four projections are not all in it: no target")
    m = lb["m"][k]
    if m != CHUNK:
        refuse(f"[skip] layer {lb['layer']} full chunk {k} holds {m} positions, the form profiles m = {CHUNK}: no target")
    names = {PROJ_ROW, PROJ_HEADS}
    order = sorted((x for x in en if x["kernel"] in names and x["act"]["graph"] is None), key=lambda x: x["s"])
    at = {x["corr"]: i for i, x in enumerate(order)}
    lo, hi = at[en[r["qkv"]]["corr"]], at[en[r["wo_b"]]["corr"]]
    role_of = {en[v]["corr"]: role for role, v in r.items()}
    with open(out, "w") as f:
        f.write(f"trace={path}\nP={P}\nbatch={lb['batch']}\nlayer={lb['layer']}\nchunk={k}\nm={m}\n")
        f.write(f"regex=^({PROJ_ROW}|{PROJ_HEADS})$\nskip={lo}\ncount={hi - lo + 1}\n")
        for i in range(lo, hi + 1):
            x = order[i]
            g, b = x["act"]["grid"], x["act"]["block"]
            f.write(f"launch {i - lo} {role_of.get(x['corr'], 'other')} {x['kernel']} "
                    f"{g[0]},{g[1]},{g[2]} {b[0]},{b[1]},{b[2]}\n")
        if run.get("plan"):
            f.write(f"plan_line={run['plan']}\n")
    print(f"[skip] layer {lb['layer']} (batch {lb['batch']}), full chunk {k} of {lb['full']} (m = {m}): matching "
          f"launches of the prompt before its qkv {lo} -> --launch-skip {lo} --launch-count {hi - lo + 1}; the "
          f"skip counts the eager {PROJ_ROW}/{PROJ_HEADS} launches in launch order, which the eager ncu run "
          f"repeats (no capture there)")
    for line in open(out):
        if line.startswith("launch "):
            print("    " + line.rstrip())
    return 0


def gguf_tensors(first):
    """name -> (dims, type) over every shard of a split GGUF (tensor infos only)."""
    m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", first)
    paths = [first] if m is None else [f"{first[:m.start()]}-{i:05d}-of-{m[2]}.gguf" for i in range(1, int(m[2]) + 1)]
    fixed = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    out = {}
    for p in paths:
        with open(p, "rb") as f:
            if f.read(4) != b"GGUF":
                refuse(f"[shape] {p} is not a GGUF file: no table")
            f.read(4)
            n_t, n_kv = struct.unpack("<QQ", f.read(16))

            def string():
                (ln,) = struct.unpack("<Q", f.read(8))
                return f.read(ln)

            def skip(t):
                if t in fixed:
                    f.read(fixed[t])
                elif t == 8:
                    string()
                elif t == 9:
                    et, cnt = struct.unpack("<IQ", f.read(12))
                    if et in fixed:
                        f.read(fixed[et] * cnt)
                    else:
                        for _ in range(cnt):
                            skip(et)
                else:
                    refuse(f"[shape] {p}: a metadata value of type {t}: no table")

            for _ in range(n_kv):
                string()
                (t,) = struct.unpack("<I", f.read(4))
                skip(t)
            for _ in range(n_t):
                name = string().decode()
                (nd,) = struct.unpack("<I", f.read(4))
                dims = struct.unpack(f"<{nd}Q", f.read(8 * nd))
                ty, _off = struct.unpack("<IQ", f.read(12))
                out[name] = (dims, ty)
    return out


# The rows and the row length K of a role's launch in layer l, from the file's tensors (the joined
# launches of chain/attn.rs `join_groups`: q_a + kv [+ indexer.proj], q_b [+ indexer.attn_q_b]).
PARTS = {
    "qkv": ("blk.{l}.attn_q_a.weight", "blk.{l}.attn_kv.weight", "blk.{l}.indexer.proj.weight"),
    "q_b": ("blk.{l}.attn_q_b.weight", "blk.{l}.indexer.attn_q_b.weight"),
    "wo_a": ("blk.{l}.attn_output_a.weight",),
    "wo_b": ("blk.{l}.attn_output_b.weight",),
}
Q3_K = 11


def role_shape(ts, role, layer):
    rows, k = 0, None
    for i, pat in enumerate(PARTS[role]):
        name = pat.format(l=layer)
        if name not in ts:
            if i == 0:
                refuse(f"[shape] the file has no {name}: no table")
            continue
        dims, ty = ts[name]
        if ty != Q3_K:
            refuse(f"[shape] {name} is ggml type {ty}, the m-column walk reads Q3_K: no table")
        if k is not None and dims[0] != k:
            continue
        k = dims[0]
        rows += dims[1]
    return rows, k


def cmd_ncu_summary(argv):
    if len(argv) < 3:
        print(__doc__)
        raise SystemExit(64)
    path, planf, model = argv[0], argv[1], argv[2]
    plan, launches = {}, []
    for line in open(planf):
        line = line.rstrip("\n")
        if line.startswith("launch "):
            _, i, role, name, g, b = line.split(" ")
            launches.append(dict(role=role, name=name, grid=tuple(int(x) for x in g.split(",")),
                                 block=tuple(int(x) for x in b.split(","))))
        elif "=" in line:
            key, v = line.split("=", 1)
            plan[key] = v
    rows = list(csv.reader(open(path)))
    hdr = next((i for i, r in enumerate(rows) if r and r[0] == "ID"), None)
    if hdr is None:
        refuse(f"[shape] {path} holds no ncu CSV header: no table")
    col = {n: i for i, n in enumerate(rows[hdr])}
    need = ("ID", "Kernel Name", "Block Size", "Grid Size", "Metric Name", "Metric Unit", "Metric Value")
    if any(n not in col for n in need):
        refuse(f"[shape] the CSV lacks one of {need}: no table")
    per = {}
    for r in rows[hdr + 1:]:
        # A metric row ends at its value; a rule row (the sections' analysis) carries no metric name.
        if len(r) <= col["Metric Value"] or not r[col["Metric Name"]]:
            continue
        rid = int(r[col["ID"]])
        d = per.setdefault(rid, dict(name=r[col["Kernel Name"]].split("(")[0],
                                     block=tuple(int(x) for x in re.findall(r"\d+", r[col["Block Size"]])),
                                     grid=tuple(int(x) for x in re.findall(r"\d+", r[col["Grid Size"]])),
                                     m={}))
        try:
            v = float(r[col["Metric Value"]].replace(",", ""))
        except ValueError:
            continue
        d["m"][(r[col["Metric Name"]], r[col["Metric Unit"]])] = v
    ids = sorted(per)
    count = int(plan["count"])
    if len(ids) != count:
        refuse(f"[shape] MISMATCH: {len(ids)} profiled launches, the plan names {count} (skip {plan['skip']}): no table")
    bad = []
    for i, rid in enumerate(ids):
        got, exp = per[rid], launches[i]
        if (got["name"], got["grid"], got["block"]) != (exp["name"], exp["grid"], exp["block"]):
            bad.append(f"launch {i}: profiled {got['name']} grid {got['grid']} block {got['block']}, expected "
                       f"{exp['role']} {exp['name']} grid {exp['grid']} block {exp['block']}")
    if bad:
        print(f"[shape] MISMATCH against the plan (skip {plan['skip']}, count {count}):")
        for b in bad:
            print("    " + b)
        refuse("    no table: the skip did not land on the chunk the trace named")
    ts = gguf_tensors(model)
    layer, m = int(plan["layer"]), int(plan["m"])

    def met(d, name, unit=None):
        vs = [v for (n, u), v in d["m"].items() if n == name and (unit is None or u == unit)]
        return vs[0] if vs else None

    out = []
    for i, rid in enumerate(ids):
        role = launches[i]["role"]
        if role not in ROLES:
            continue
        d = per[rid]
        rows_, k = role_shape(ts, role, layer)
        n_sb = k // 256
        iters = -(-n_sb // 2)
        if -(-rows_ // 8) != d["grid"][0]:
            refuse(f"[shape] {role}: the file gives {rows_} rows (grid {-(-rows_ // 8)}), the launch has grid "
                   f"{d['grid'][0]}: no table")
        ld = met(d, "smsp__inst_executed_op_global_ld.sum")
        if ld is None:
            refuse(f"[shape] {role}: no smsp__inst_executed_op_global_ld.sum in the CSV: the column count "
                   f"cannot be read, no table")
        # The m-column walk loads 8 weight words and, per column, two u64 code slots and one f32 block
        # scale each warp-iteration (cores.rs q3k_sb_decode and q3k_row_dot_cols_span; the sm_86 SASS
        # issues 16 LDG.E + 16 LDG.E.64 at m = 8).
        m_seen = (ld / (rows_ * iters) - 8) / 3
        if abs(m_seen - m) > 0.01:
            refuse(f"[shape] {role}: {ld:.0f} global loads over {rows_} rows x {iters} iterations is m = "
                   f"{m_seen:.3f} columns, the plan's chunk has m = {m}: no table")
        out.append((role, d, rows_, k, iters, m_seen))
    print(f"--- shapes OK: layer {layer} chunk {plan['chunk']}, m = {m} (read back from each launch's global "
          f"loads), launch-skip {plan['skip']}")
    print("--- per launch: cycles are SM cycles at ncu's fixed clocks; demand = the unit's work over the "
          "block-steps at its peak rate (issue 1 a cycle a scheduler; alu, fmaheavy, fmalite 16 lanes a "
          "scheduler = 2 cycles a warp-instruction; xu 4 lanes = 8; L1TEX data pipe 1 wavefront a cycle an "
          "SM; lsu 1 warp-instruction per 2 cycles an SM, ncu's own peak for pipe_lsu), and where ncu counts "
          "it, the pipe's own active cycles; a block-step is one iteration of a "
          "block's 8 warps; dp4a (IDP.4A) issues on fmaheavy with IMAD and IMUL (the Profiling Guide's "
          "pipeline table), the other integer ops on alu; not numbers of record")
    for role, d, rows_, k, iters, m_seen in out:
        n_sm = met(d, "# SMs") or met(d, "launch__sm_count")
        if n_sm is None:
            refuse(f"[shape] {role}: the CSV has no # SMs (LaunchStats): no table")
        g = d["grid"][0]
        bs = g * iters                          # block-steps of the launch
        elapsed = met(d, "sm__cycles_elapsed.avg")
        active = met(d, "sm__cycles_active.avg")
        step = elapsed * n_sm / bs
        dem = []

        def pipe(label, metric, cost, per_smsp=True):
            v = met(d, metric)
            if v is None:
                dem.append((label, None, metric))
                return
            dem.append((label, v * cost / ((4 if per_smsp else 1) * bs), metric))

        def busy(label, metric):
            # A per-scheduler (or per-L1TEX) active-cycle mean: as SM cycles a block-step, v x n_SM / bs.
            v = met(d, metric)
            dem.append((label, None if v is None else v * n_sm / bs, metric))

        pipe("issue", "smsp__inst_executed.sum", 1)
        pipe("alu (count x 2)", "sm__inst_executed_pipe_alu.sum", 2)
        busy("alu (pipe active)", "smsp__pipe_alu_cycles_active.avg")
        pipe("fmaheavy (count x 2)", "sm__inst_executed_pipe_fmaheavy.sum", 2)
        busy("fmaheavy (pipe active)", "smsp__pipe_fmaheavy_cycles_active.avg")
        pipe("fmalite (count x 2)", "sm__inst_executed_pipe_fmalite.sum", 2)
        busy("fmalite (pipe active)", "smsp__pipe_fmalite_cycles_active.avg")
        busy("fma (pipe active)", "smsp__pipe_fma_cycles_active.avg")
        pipe("xu (count x 8)", "sm__inst_executed_pipe_xu.sum", 8)
        pipe("lsu (count x 2, a SM)", "sm__inst_executed_pipe_lsu.sum", 2, per_smsp=False)
        pipe("L1TEX data pipe", "l1tex__data_pipe_lsu_wavefronts.sum", 1, per_smsp=False)
        busy("LSU writeback (active)", "l1tex__lsu_writeback_active.avg")
        dram = met(d, "dram__throughput.avg.pct_of_peak_sustained_elapsed")
        if dram is not None:
            dem.append(("DRAM (share of the step)", dram / 100 * step, "dram__throughput.avg.pct_of_peak_sustained_elapsed"))
        res = [met(d, n) for n in ("Block Limit Registers", "Block Limit Shared Mem", "Block Limit Warps",
                                   "Block Limit SM")]
        resident = min(v for v in res if v is not None) if any(v is not None for v in res) else float("nan")
        print(f"  [{role}] {d['name']} grid {g} x {d['block'][0]}: {rows_} rows, K {k} ({n_sb} super-blocks, "
              f"{iters} iterations), m {m_seen:.2f}; {n_sm:.0f} SMs, {g / n_sm:.2f} blocks an SM ({resident:.0f} "
              f"resident, {met(d, 'Waves Per SM') or float('nan'):.2f} waves); {bs / n_sm:.1f} block-steps an SM")
        print(f"    block-step {step:8.1f} cycles (elapsed {elapsed:.0f}, active {active:.0f} an SM; achieved "
              f"occupancy {met(d, 'sm__warps_active.avg.pct_of_peak_sustained_active') or float('nan'):.1f} %, "
              f"issue-active {met(d, 'smsp__issue_active.avg.pct_of_peak_sustained_active') or float('nan'):.1f} %)")
        freq, dur = met(d, "SM Frequency"), met(d, "Duration")
        print(f"    {fmt_opt(freq and freq / 1e9, '.2f')} GHz SM clock, {fmt_opt(dur and dur / 1e3, '.1f')} µs under "
              f"ncu (clocks fixed, caches flushed before each pass); {fmt_opt(met(d, 'Registers Per Thread'))} "
              f"registers a thread; resident-block limits: registers {fmt_opt(met(d, 'Block Limit Registers'))}, "
              f"shared {fmt_opt(met(d, 'Block Limit Shared Mem'))}, warps {fmt_opt(met(d, 'Block Limit Warps'))}")
        for label, v, metric in dem:
            if v is None:
                print(f"    {label:26s} {'NOT REPORTED':>10s}   {metric}")
            else:
                print(f"    {label:26s} {v:10.1f} cycles  {100 * v / step:5.1f} % of the step   {metric}")
        top = max((x for x in dem if x[1] is not None), key=lambda x: x[1], default=None)
        if top:
            print(f"    busiest {top[0]} at {100 * top[1] / step:.1f} % of the block-step")
        per_wi = {}
        for key in ("smsp__inst_executed.sum", "sm__inst_executed_pipe_alu.sum", "sm__inst_executed_pipe_fmaheavy.sum",
                    "sm__inst_executed_pipe_fmalite.sum", "sm__inst_executed_pipe_xu.sum",
                    "sm__inst_executed_pipe_lsu.sum", "sm__inst_executed_pipe_cbu.sum",
                    "sm__inst_executed_pipe_adu.sum", "sm__inst_executed_pipe_uniform.sum",
                    "smsp__inst_executed_op_global_ld.sum", "l1tex__lsuin_requests.sum",
                    "l1tex__data_pipe_lsu_wavefronts.sum"):
            v = met(d, key)
            if v is not None:
                per_wi[key] = v / (rows_ * iters)
        print("    per warp-iteration: " + ", ".join(f"{k.split('__')[1].replace('inst_executed_', '').replace('.sum', '')} "
                                                  f"{v:.1f}" for k, v in per_wi.items()))
        pcts = [(n.replace("sm__inst_executed_pipe_", "").split(".")[0], met(d, n)) for n in (
            "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active",
            "sm__inst_executed_pipe_fmaheavy.avg.pct_of_peak_sustained_active",
            "sm__inst_executed_pipe_fmalite.avg.pct_of_peak_sustained_active",
            "sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active",
            "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active")]
        print("    ncu's own pipe shares (% of peak, active cycles): " +
              ", ".join(f"{n} {v:.1f}" for n, v in pcts if v is not None) +
              f"; L1TEX {met(d, 'l1tex__throughput.avg.pct_of_peak_sustained_active') or float('nan'):.1f} %")
        st = sorted(((n.replace("smsp__warp_issue_stalled_", "").replace("_per_warp_active.pct", ""), v)
                     for (n, _), v in d["m"].items() if n.startswith("smsp__warp_issue_stalled_")
                     and n.endswith("_per_warp_active.pct")), key=lambda x: -x[1])
        if not st:
            refuse(f"[shape] {role}: no smsp__warp_issue_stalled_*_per_warp_active.pct in the CSV: no stall table")
        print("    stalls (% of the active warps' cycles): " + ", ".join(f"{n} {v:.1f}" for n, v in st[:6]))
    return 0


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("tables", "ncu-plan", "ncu-summary"):
        print(__doc__)
        raise SystemExit(64)
    cmd = {"tables": cmd_tables, "ncu-plan": cmd_ncu_plan, "ncu-summary": cmd_ncu_summary}[sys.argv[1]]
    raise SystemExit(cmd(sys.argv[2:]))


if __name__ == "__main__":
    main()
