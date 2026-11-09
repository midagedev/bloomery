#!/usr/bin/env python3
"""The kernel-pick model against the forced-candidate bench (docs/research/kernel-picks.md section 3).

  kernel-model.py report --bench LOG... [--scan LOG] [--sm-mhz N] [--c-node US] [--band 0.10] [--dram-gbs N]
    --dram-gbs replaces the card row's DRAM rate (a datasheet figure the caller names)
  kernel-model.py --self-test

`bench_picks --time` prints `kcal card`, `kcal pair` and `kcal single` rows (crates/gpu-gates/src/bin/bench_picks.rs).
This tool computes section 3's per-candidate terms from each row's shape, the card row (SMs, DRAM rate, L2 bytes, the slab
entry's driver-queried wave, the int8 MMA rate) and the entries' occupancy from a `ptx-scan` table (--scan; without one the
sm_86 table of the design is used), then prints per pair the predicted and the measured winner and ratio, and a summary.

    t_c = max( B_c / (R_dram * s), Q_c / R_l2, I_c / R_issue, W_c * L_c ) + N_c * c_node        (section 3)

Two models share the terms and differ in the constants:
  M0  the constants the design names: s = 0.85 and c_node, nothing else; the terms it has no constant for (L2 requests,
      issue, the chain) are absent, so an MXFP8 pair of equal bytes and launches is a tie, and a tie keeps the incumbent.
  M1  M0 plus the section 4 R3 constants fitted on the bench itself (the L2 request rate, the issue rate, the chain per
      step, each as a multiple of its datasheet-derived value, and the read share): in-sample and k-fold by shape group.
A pick is the incumbent (the rule the bench printed in `rule=`) unless the other candidate is predicted faster by more than
the band; the measured winner is the faster mean, a `level` pair when the gap is inside the A/A arm's spread.
Every number printed is computed from the rows read; a constant that is a guess is named `[guess]` where it is set.
"""
import math
import re
import shlex
import sys

# --- the design's named constants, and the guesses beside them --------------------------------------------------
READ_SHARE = 0.85  # section 3, the upper end of the 0.69..0.85 span
C_NODE_US = 0.852  # section 3, the A6000's graph node [measured: rig-log docs/bloomery/plan.md 「비용 모델 (디코드 스텝)」, row `c_node`]
BAND = 0.10  # section 3 rule 3
# Section 3.2: L2 requests a row-step (64 values) a tile, plain and slab.
Q8_REQ = {"plain": 3.0, "slab": 1.25}
# [guess] warp instructions a warp issues a step: a + b * columns, from reading the walks (decode, one load and four fused
# multiply-adds a column, the staged walk's barrier and stage copy); the Q8_0 GEMM's per block-step, a few more for the slab.
IPW = {"mxfp8_gemv": (28, 6), "plain": (28, 6), "staged": (31, 6), "staged_wide": (31, 6)}
Q8_IPB = {"plain": 60.0, "slab": 64.0}
# [guess] datasheet-derived rates the fitted multiples scale: warp instructions a clock an SM, FP32 FMAs a clock an SM, one
# L2 request a clock a slice with a slice a 2 SMs.
ISSUE_PER_SM_CLK = 4.0
FMA_PER_SM_CLK = 128.0
STEP_CHAIN_NS = 60.0  # [guess] one dependent step of a block
ENTRY = {
    ("mxfp8", "mxfp8_gemv"): "mxfp8_gemv",
    ("mxfp8", "plain"): "mxfp8_gemv_mcol",
    ("mxfp8", "staged"): "mxfp8_gemv_mcol_staged",
    ("mxfp8", "staged_wide"): "mxfp8_gemv_mcol_staged_wide",
    ("q8_0p", "plain"): "gemm_q8_0p",
    ("q8_0p", "slab"): "gemm_q8_0p_slab",
    ("f32", "gemv"): "f32_gemv",
    ("f32", "tile"): "f32_tile_gemm",
    ("bf16", "gemv"): "bf16_gemv",
    ("bf16", "tile"): "bf16_tile_gemm",
}
# The design's sm_86 table (ptx-scan --no-jit of bloomery-serve, section 3): regs, smem, blocks an SM.
SM86 = {
    "mxfp8_gemv": (39, 0, 6), "mxfp8_gemv_mcol": (72, 0, 3), "mxfp8_gemv_mcol_staged": (48, 4096, 5),
    "mxfp8_gemv_mcol_staged_wide": (47, 8192, 5), "gemm_q8_0p": (106, 33312, 2), "gemm_q8_0p_slab": (114, 33312, 2),
    "f32_gemv": (64, 0, 4), "bf16_gemv": (60, 0, 4), "f32_tile_gemm": (214, 49152, 1), "bf16_tile_gemm": (202, 49152, 1),
}
GROUP = 16  # gemm32.rs GEMM32_GROUP
RASTER_BYTES = 8 << 20  # gemm32.rs GEMM32_RASTER_BYTES
TILE_COLS = 64
TILE_ROWS = 128


# --- reading ---------------------------------------------------------------------------------------------------
def parse_kv(line):
    """`kcal <kind> k=v ...` as (kind, dict of str)."""
    words = shlex.split(line)
    kind, d = words[1], {}
    for w in words[2:]:
        if "=" in w:
            k, v = w.split("=", 1)
            d[k] = v
    return kind, d


def read_bench(paths):
    """cards and rows of the logs; a row remembers the card row before it."""
    cards, rows, sm_mhz = [], [], None
    for p in paths:
        card = None
        for line in open(p, errors="replace"):
            if line.startswith("kcal card"):
                _, card = parse_kv(line.strip())
                cards.append(card)
            elif line.startswith(("kcal pair", "kcal single")):
                kind, d = parse_kv(line.strip())
                if card is None:
                    raise SystemExit(f"kernel-model: {p}: a `kcal {kind}` row before any `kcal card` row")
                rows.append((kind, d, card))
            m = re.search(r"clocks: sm=\d+ max=(\d+) MHz", line)
            if m and sm_mhz is None:
                sm_mhz = float(m.group(1))
    return cards, rows, sm_mhz


def read_scan(path):
    """entry -> (regs, smem, blocks an SM) from a ptx-scan table."""
    out, cols = {}, None
    for line in open(path, errors="replace"):
        w = line.split()
        if w and w[0] == "entry":
            cols = {c: i for i, c in enumerate(w)}
        elif cols and len(w) == len(cols) and w[0] in SM86:
            try:
                out[w[0]] = (int(w[cols["regs"]]), int(w[cols["smem"]]), int(w[cols["blk/SM(static)"]]))
            except ValueError:
                pass
    return out


# --- the terms -------------------------------------------------------------------------------------------------
def f(d, key):
    return float(d[key])


def binom_tiles(tokens, top_k, experts, col_tile=TILE_COLS):
    """Expected (tiles, hit experts) of a route of `tokens` tokens choosing `top_k` of `experts`: the tiles an expert is
    ceil(n_e / 64) for n_e ~ Binomial(tokens, top_k / experts)."""
    p = top_k / experts
    if experts == 1:
        return math.ceil(tokens / col_tile), 1.0
    lp, lq = math.log(p), math.log1p(-p)
    pmf, tail = [], 0.0
    for n in range(tokens + 1):
        lg = math.lgamma(tokens + 1) - math.lgamma(n + 1) - math.lgamma(tokens - n + 1)
        pmf.append(math.exp(lg + n * lp + (tokens - n) * lq))
    e_tiles = 0.0
    for n, pr in enumerate(pmf):
        e_tiles += pr * math.ceil(n / col_tile)
    hit = 1.0 - pmf[0]
    return experts * e_tiles, experts * hit


def terms(row, name, card, ent, sm_mhz):
    """The §3 terms of candidate `name` for a row: bytes, requests, warp instructions, FMAs, waves, chain steps, launches,
    and the extra DRAM bytes of a slab refetch; all [derived] from the shape."""
    fam = row["family"]
    rows_, k = int(row["rows"]), int(row["k"])
    if fam == "mxfp8" and name == "plain":
        k = int(row.get("k_plain", k))  # a proxy pair's plain walk runs at its own k
    sms = int(card["sms"])
    regs, smem, blk = ent[ENTRY[(fam, name)]]
    t = {"blk": blk, "launches": 1, "req": 0.0, "winstr": 0.0, "fma": 0.0, "mma_macs": 0.0}
    if fam == "mxfp8":
        m = int(row["m"])
        blocks = math.ceil(rows_ / 8)
        steps = k / 128
        a, b = IPW[name]
        t.update(
            bytes=rows_ * k / 32 * 33 + 4 * rows_ * m + 4 * m * k,
            req=(rows_ * k / 32 * 33 + blocks * 4 * m * k) / 32,
            winstr=blocks * 8 * steps * (a + b * m),
            waves=math.ceil(blocks / (sms * blk)),
            steps=steps,
        )
    elif fam == "q8_0p":
        experts, slots = int(row["experts"]), int(row["slots"])
        tokens = int(row["tokens"])
        top_k = slots // tokens
        if experts == 1:
            tiles, hit = math.ceil(slots / TILE_COLS), 1.0
        else:
            # An expert's slot list is Binomial(tokens, top_k / experts) long whether its slots read their own columns
            # or a token's shared one.
            tiles, hit = binom_tiles(tokens, top_k, experts)
        row_tiles = math.ceil(rows_ / TILE_ROWS)
        steps = k / 64
        stack = rows_ * (k + k / 16)  # one expert's planes
        group = GROUP if (name == "slab" and stack >= RASTER_BYTES) else 1
        wave_blocks = sms * blk
        resident_tiles = max(1, wave_blocks // row_tiles)
        tiles_e = max(1.0, tiles / max(hit, 1.0))
        l2 = int(card["l2_bytes"])
        if stack <= l2 // 2:
            fetches = 1.0
        else:
            fetches = math.ceil(tiles_e / max(resident_tiles, group))
        cols = slots if row.get("per_slot") == "true" else tokens
        act = cols * k * 72 / 64
        extra = 0.0
        if group > 1:
            g_act = group * TILE_COLS * k * 72 / 64
            if g_act > l2 // 2:
                extra = act * (row_tiles - 1)  # the group's columns are refetched by every slab
        t.update(
            bytes=hit * stack * fetches + act + extra + 4 * slots * rows_,
            req=row_tiles * tiles * TILE_ROWS * steps * Q8_REQ[name],
            winstr=row_tiles * tiles * 8 * steps * Q8_IPB[name],
            mma_macs=rows_ * slots * k,
            waves=math.ceil(row_tiles * tiles / wave_blocks),
            steps=steps,
        )
    else:  # f32 / bf16: the gemv repeated against the tile
        m = int(row["m"])
        w_bytes = rows_ * k * (4 if fam == "f32" else 2)
        l2 = int(card["l2_bytes"])
        if name == "gemv":
            launches = math.ceil(m / 8)
            blocks = math.ceil(rows_ / 8)
            t.update(
                launches=launches,
                bytes=w_bytes * (launches if w_bytes > l2 // 2 else 1) + 4 * m * k + 4 * rows_ * m,
                req=launches * w_bytes / 32,
                winstr=launches * blocks * 8 * (k / 128) * 40,
                fma=rows_ * m * k,
                waves=launches * math.ceil(blocks / (sms * blk)),
                steps=k / 128,
            )
        else:
            grid = math.ceil(rows_ / 32) * math.ceil(m / 32)
            t.update(
                bytes=w_bytes * (math.ceil(m / 32) if w_bytes > l2 // 2 else 1) + 4 * m * k + 4 * rows_ * m,
                req=grid * (32 + min(m, 32)) * k * 4 / 16 / 4,
                winstr=0.0,
                fma=rows_ * m * k,
                waves=math.ceil(grid / (sms * blk)),
                steps=k / 64,
            )
    return t


def predict(row, name, card, ent, sm_mhz, theta, c_node):
    """µs of candidate `name`; theta = None is M0, else the fitted multiples (s, q, i, c)."""
    t = terms(row, name, card, ent, sm_mhz)
    sms = int(card["sms"])
    clk = sm_mhz * 1e-3
    dram_rate = f(card, "dram_b_per_us")
    s, q, i, c = (READ_SHARE, 0, 0, 0) if theta is None else theta
    parts = [t["bytes"] / (dram_rate * s)]
    if theta is not None:
        parts.append(q * t["req"] / (sms / 2 * clk * 1e3))
        issue = t["winstr"] / (sms * ISSUE_PER_SM_CLK * clk * 1e3) + t["fma"] / (sms * FMA_PER_SM_CLK * clk * 1e3)
        mma = t["mma_macs"] * 2 / f(card, "int8_ops_per_us") if "int8_ops_per_us" in card else 0.0
        parts.append(i * (issue + mma))
        parts.append(c * t["waves"] * t["steps"] * STEP_CHAIN_NS * 1e-3)
    return max(parts) + t["launches"] * c_node


# --- measured --------------------------------------------------------------------------------------------------
PROXY_SLACK = 0.03  # a proxy pair compares two shapes (k and k - 128): per-step times agree to a few percent at best


def is_proxy(d):
    return d["family"] == "mxfp8" and d.get("k_plain", d["k"]) != d["k"]


def per_step(d, name):
    """Steps (128 values) a candidate's launch walks: 1 for a pair on one shape, so times compare as measured; a proxy
    pair compares per step."""
    if not is_proxy(d):
        return 1.0
    return (int(d["k_plain"]) if name == "plain" else int(d["k"])) / 128


def measured(kind, d):
    a, b = d["a"], d["b"]
    if kind == "single" or b == "-":
        return None
    t_a, t_b = f(d, f"{a}_us_mean") / per_step(d, a), f(d, f"{b}_us_mean") / per_step(d, b)
    ratio = f(d, f"{b}_over_{a}") if not is_proxy(d) else t_b / t_a
    aa = f(d, f"aa_over_{a}")
    sd = f(d, f"aa_over_{a}_sd")
    thr = max(0.01, abs(aa - 1) + 2 * sd / math.sqrt(f(d, "rounds"))) + (PROXY_SLACK if is_proxy(d) else 0.0)
    return {"ratio": ratio, "thr": thr, "level": abs(ratio - 1) <= thr, "t_a": t_a, "t_b": t_b}


def pick(pa, pb, a, b, rule, band):
    """The incumbent unless the other candidate is predicted faster by more than the band."""
    inc = rule if rule in (a, b) else a
    other = b if inc == a else a
    t_inc, t_oth = (pa, pb) if inc == a else (pb, pa)
    return other if t_oth < t_inc * (1 - band) else inc


def fit(rows, ent, sm_mhz, c_node, folds):
    """Fit (s, q, i, c) of one arch by coordinate descent on the log error of every candidate time, k-fold by shape
    group. Returns (in-sample theta, {row index: theta of its fold})."""
    def loss(theta, idx):
        e = 0.0
        for j in idx:
            d, card = rows[j][1], rows[j][2]
            for name, key in ((d["a"], "a"), (d["b"], "b")):
                if name == "-":
                    continue
                tm = f(d, f"{name}_us_mean")
                e += math.log(max(predict(d, name, card, ent, sm_mhz, theta, c_node), 1e-3) / tm) ** 2
        return e

    def descend(idx):
        theta = [READ_SHARE, 1.0, 1.0, 1.0]
        step = 2.0
        for _ in range(40):
            moved = False
            for p in range(4):
                for mult in (step, 1 / step):
                    cand = list(theta)
                    cand[p] = min(max(theta[p] * mult, 1e-3), 1.0) if p == 0 else max(theta[p] * mult, 1e-3)
                    if loss(cand, idx) < loss(theta, idx) - 1e-12:
                        theta, moved = cand, True
            if not moved:
                step = step**0.5
                if step < 1.02:
                    break
        return tuple(theta)

    idx = [j for j, r in enumerate(rows) if r[0] == "pair"]
    groups = sorted({rows[j][1]["family"] + ":" + rows[j][1]["shape"] for j in idx})
    full = descend(idx)
    fold_of = {g: n % folds for n, g in enumerate(groups)}
    by_fold = {}
    for n in range(folds):
        train = [j for j in idx if fold_of[rows[j][1]["family"] + ":" + rows[j][1]["shape"]] != n]
        by_fold[n] = descend(train) if train else full
    return full, {j: by_fold[fold_of[rows[j][1]["family"] + ":" + rows[j][1]["shape"]]] for j in idx}


# --- the report ------------------------------------------------------------------------------------------------
def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(math.ceil(q * len(xs))) - 1)] if xs else float("nan")


def classify(d, card):
    """R1 / R2 of section 4, or empty."""
    if d["family"] == "mxfp8" and int(d["m"]) <= 4:
        return "R1"  # the plain-against-staged walk at two to four columns
    if d["family"] == "q8_0p":
        grid, wave = int(d["grid"]), int(card["slab_wave"])
        per_expert = int(d["slots"]) / int(d["experts"])
        if 0.5 <= grid / wave <= 4 and per_expert <= 2 * TILE_COLS:
            return "R2"
    return ""


def report(rows, ent, sm_mhz, c_node, band, out=print, folds=5):
    arches = {}
    for r in rows:
        arches.setdefault(f"{r[2]['name']} cc{r[2]['cc']}", []).append(r)
    for arch, rs in arches.items():
        out(f"== {arch}: {sum(1 for r in rs if r[0] == 'pair')} pairs, {sum(1 for r in rs if r[0] == 'single')} single arms")
        theta, by_row = fit(rs, ent, sm_mhz, c_node, folds)
        out(f"M1 fit (s, L2 req x, issue x, chain x): " + " ".join(f"{x:.3f}" for x in theta))
        out(f"{'label':34} {'rule':11} {'M0':11} {'M1':11} {'meas':11} {'pred/meas':>9} {'meas t_b/t_a':>12} {'cls':3}")
        stats = {k: {"hit": 0, "loss": [], "n": 0} for k in ("rule", "M0", "M1-fit", "M1-cv")}
        rat, absr, levels = [], [], 0
        # pairs whose rule loses by more than the band, and whether each model's pick is the measured winner there
        big = {"n": 0, "M0": 0, "M1-cv": 0, "list": []}
        by_fam = {}
        for j, (kind, d, card) in enumerate(rs):
            if kind != "pair":
                continue
            ms = measured(kind, d)
            a, b, rule = d["a"], d["b"], d["rule"]
            t0 = {n: predict(d, n, card, ent, sm_mhz, None, c_node) / per_step(d, n) for n in (a, b)}
            t1 = {n: predict(d, n, card, ent, sm_mhz, theta, c_node) / per_step(d, n) for n in (a, b)}
            t1c = {n: predict(d, n, card, ent, sm_mhz, by_row[j], c_node) / per_step(d, n) for n in (a, b)}
            w_meas = a if ms["t_a"] <= ms["t_b"] else b
            picks = {
                "rule": rule if rule in (a, b) else None,
                "M0": pick(t0[a], t0[b], a, b, rule, band),
                "M1-fit": pick(t1[a], t1[b], a, b, rule, band),
                "M1-cv": pick(t1c[a], t1c[b], a, b, rule, band),
            }
            levels += ms["level"]
            for key, p in picks.items():
                if p is None:
                    continue
                s = stats[key]
                fam = by_fam.setdefault((d["family"], key), [0, 0])
                fam[1] += 1
                fam[0] += p == w_meas or ms["level"]
                s["n"] += 1
                tp = ms["t_a"] if p == a else ms["t_b"]
                best = min(ms["t_a"], ms["t_b"])
                if p == w_meas or ms["level"]:
                    s["hit"] += 1
                else:
                    s["loss"].append(tp / best)
            if rule in (a, b) and not ms["level"] and rule != w_meas:
                lost = (ms["t_a"] if rule == a else ms["t_b"]) / min(ms["t_a"], ms["t_b"])
                if lost > 1 + band:
                    big["n"] += 1
                    big["M0"] += picks["M0"] == w_meas
                    big["M1-cv"] += picks["M1-cv"] == w_meas
                    big["list"].append(f"{d['label']} {lost:.3f}x")
            pr = t1c[b] / t1c[a]
            rat.append(pr / ms["ratio"])
            absr += [t1c[a] / ms["t_a"], t1c[b] / ms["t_b"]]
            out(f"{d['label']:34} {rule:11} {picks['M0']:11} {picks['M1-cv']:11} {('level' if ms['level'] else w_meas):11} "
                f"{pr / ms['ratio']:9.3f} {ms['ratio']:12.4f} {classify(d, card):3}")
        out(f"-- pairs level inside the A/A spread: {levels}")
        for key, s in stats.items():
            loss = s["loss"]
            out(f"-- {key:7} hit {s['hit']}/{s['n']} = {s['hit'] / max(s['n'], 1):.3f}; wrong {len(loss)}: "
                f"mean loss {sum(loss) / len(loss) if loss else 1.0:.4f}x, worst {max(loss) if loss else 1.0:.4f}x, "
                f"{sum(1 for x in loss if x > 1.03)} lose more than 3 %")
        for fam in sorted({k[0] for k in by_fam}):
            out(f"-- {fam:6} hit " + ", ".join(f"{key} {by_fam[(fam, key)][0]}/{by_fam[(fam, key)][1]}" for key in stats if (fam, key) in by_fam))
        out(f"-- M1-cv predicted/measured pair ratio: median {pct(rat, .5):.3f} p90 {pct(rat, .9):.3f}; "
            f"candidate time: median {pct(absr, .5):.3f} p90 {pct(absr, .9):.3f}")
        out(f"-- rule loses by more than the band ({band:.0%}) in {big['n']} pairs, M0 picks the winner in {big['M0']}, "
            f"M1-cv in {big['M1-cv']}: " + "; ".join(big["list"]))
        for cls in ("R1", "R2"):
            sel = [r for r in rs if r[0] == "pair" and classify(r[1], r[2]) == cls]
            out(f"-- {cls}: {len(sel)} pairs")


# --- self-test -------------------------------------------------------------------------------------------------
def fake_row(label, a, b, rule, ratio, aa=1.0, sd=0.002, **kw):
    d = {"label": label, "family": "mxfp8", "shape": "s", "rows": "10240", "k": "2560", "experts": "1", "slots": "4",
         "m": "4", "a": a, "b": b, "rule": rule, "rounds": "9", f"{a}_us_mean": "100", f"{b}_us_mean": str(100 * ratio),
         f"{b}_over_{a}": str(ratio), f"aa_over_{a}": str(aa), f"aa_over_{a}_sd": str(sd)}
    d.update(kw)
    return d


def self_test():
    card = {"name": "T", "cc": "8.6", "sms": "84", "l2_bytes": str(6 << 20), "dram_b_per_us": "768000", "int8_ops_per_us": "4e8",
            "slab_wave": "168"}
    # parsing: a quoted name with spaces, and the kind
    kind, d = parse_kv('kcal card name="NVIDIA RTX A6000" cc=8.6 sms=84')
    assert kind == "card" and d["name"] == "NVIDIA RTX A6000" and d["sms"] == "84", (kind, d)
    # measured: a ratio outside the A/A spread is a winner, one inside is level
    assert not measured("pair", fake_row("x", "plain", "staged", "staged", 0.90))["level"]
    assert measured("pair", fake_row("x", "plain", "staged", "staged", 1.004))["level"]
    assert measured("pair", fake_row("x", "plain", "staged", "staged", 0.97, aa=1.03))["level"]  # the A/A arm moved 3 %
    # the band: inside it the incumbent stays, outside the faster candidate is taken
    assert pick(100, 95, "plain", "staged", "plain", 0.10) == "plain"
    assert pick(100, 85, "plain", "staged", "plain", 0.10) == "staged"
    assert pick(85, 100, "plain", "staged", "plain", 0.10) == "plain"
    assert pick(100, 85, "plain", "staged", "-", 0.10) == "staged"  # no incumbent among the candidates: a is it
    # M0 has equal bytes and launches for the MXFP8 pair: a tie, so the incumbent
    row = fake_row("x", "plain", "staged", "staged", 0.9)
    t_a = predict(row, "plain", card, SM86, 1800.0, None, C_NODE_US)
    t_b = predict(row, "staged", card, SM86, 1800.0, None, C_NODE_US)
    assert t_a == t_b, (t_a, t_b)
    # a proxy pair (plain at k - 128 against the engine's walk at k) compares per step: 19 steps against 20 here
    px = fake_row("x", "plain", "staged", "staged", 1.0, k="2560", k_plain="2432")
    mp = measured("pair", px)
    assert abs(mp["ratio"] - (100 / 20) / (100 / 19)) < 1e-9 and mp["thr"] >= PROXY_SLACK, mp
    assert per_step(px, "plain") == 19 and per_step(px, "staged") == 20 and per_step(row, "plain") == 1.0
    # the DRAM floor of 10240x2560 MXFP8 at four columns on 768 GB/s and 0.85: 27 MB plus x and y over 652.8 B/us
    want = (10240 * 2560 / 32 * 33 + 4 * 10240 * 4 + 4 * 4 * 2560) / (768000 * 0.85) + C_NODE_US
    assert abs(t_a - want) < 1e-9, (t_a, want)
    # a repeated gemv is more launches and, past the L2, more bytes than the tile: the tile wins at m = 512
    fr = {"family": "f32", "shape": "512x2560", "rows": "512", "k": "2560", "experts": "1", "slots": "512", "m": "512"}
    assert predict(fr, "gemv", card, SM86, 1800.0, None, C_NODE_US) > predict(fr, "tile", card, SM86, 1800.0, None, C_NODE_US)
    # the Q8_0 raster: a stack over the L2 is fetched once per resident-tile group by the plain order and once per 16 tiles
    # by the slab. At 10240 rows (80 row tiles) the resident wave holds 4 tiles, so with the 24 MiB L2 the slab fetches a
    # quarter as often; with a 6 MiB L2 its group's columns (7 MB at K 6144) are larger than the L2 and every slab
    # refetches them, so it loses
    qr = {"family": "q8_0p", "shape": "out", "rows": "10240", "k": "6144", "experts": "1", "slots": "2048", "tokens": "2048",
          "per_slot": "true", "m": "2048"}
    card24 = dict(card, l2_bytes=str(24 << 20))
    assert predict(qr, "slab", card24, SM86, 1800.0, None, C_NODE_US) < predict(qr, "plain", card24, SM86, 1800.0, None, C_NODE_US)
    assert predict(qr, "slab", card, SM86, 1800.0, None, C_NODE_US) > predict(qr, "plain", card, SM86, 1800.0, None, C_NODE_US)
    # the expected route: tokens x top_k slots over the experts, about the mean tiles when every expert gets many slots
    tiles, hit = binom_tiles(4096, 10, 512)
    assert 512 * 1.0 <= tiles <= 512 * 3.0 and hit > 511, (tiles, hit)
    # the classes: a four-column MXFP8 pair is R1; a Q8_0 pair near one wave with one tile an expert is R2
    assert classify(row, card) == "R1"
    q2 = dict(qr, grid="170", slots="64", experts="1", tokens="64")
    assert classify(q2, card) == "R2"
    # a report over two fake pairs prints a hit-rate line for the incumbent
    lines = []
    rows = [("pair", fake_row("p1", "plain", "staged", "staged", 0.80, shape="s1"), card),
            ("pair", fake_row("p2", "plain", "staged", "plain", 1.00, shape="s2", m="5"), card)]
    report(rows, SM86, 1800.0, C_NODE_US, BAND, out=lines.append, folds=2)
    text = "\n".join(lines)
    assert "-- rule    hit 2/2" in text, text  # staged is the rule and wins p1; p2 is level, which no pick loses
    print("kernel-model self-test: ok")


def main(argv):
    if argv[1:] == ["--self-test"]:
        self_test()
        return 0
    if len(argv) < 3 or argv[1] != "report":
        print(__doc__)
        return 64
    bench, scan, sm_mhz, c_node, band, dram = [], None, None, C_NODE_US, BAND, None
    it = iter(argv[2:])
    for a in it:
        if a == "--bench":
            bench.append(next(it))
        elif a == "--scan":
            scan = next(it)
        elif a == "--sm-mhz":
            sm_mhz = float(next(it))
        elif a == "--c-node":
            c_node = float(next(it))
        elif a == "--band":
            band = float(next(it))
        elif a == "--dram-gbs":
            dram = float(next(it))
        else:
            print(f"kernel-model: unknown argument {a}", file=sys.stderr)
            return 64
    cards, rows, witness_mhz = read_bench(bench)
    for c in cards:
        if dram is not None:
            c["dram_b_per_us"] = str(dram * 1e3)  # the datasheet rate in place of the card row's
        elif "dram_b_per_us" not in c:
            print("kernel-model: the card row prices no DRAM rate (rates_unpriced); pass --dram-gbs", file=sys.stderr)
            return 64
    ent = dict(SM86)
    if scan:
        ent.update(read_scan(scan))
    mhz = sm_mhz or witness_mhz
    if mhz is None:
        print("kernel-model: no SM clock: the log has no `clocks: sm=.. max=.. MHz` witness line; pass --sm-mhz", file=sys.stderr)
        return 64
    report(rows, ent, mhz, c_node, band)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
