#!/usr/bin/env python3
"""The expected cells of `nodes_scale` (docs/contrib/dgx-spark.md §9.9), derived apart from the planner.

What it derives. For each synthetic shape the gate plans (§9.7's terms for V4.1, GLM-5.3-Flash and Qwen3.8, A low and
high), on N = 1 to 4 Sparks (a pair, a ring of three, four on a switch), at λ 25 and 70 µs, under the gate's ruler and a
4 % one, and at S₁ 1,000 and 3,000 tok/s: the nodes the plan uses and why it leaves each other node idle (§9.8 criteria
1, 2 and 7), the routed ids the pools cannot hold, in all and per node (criterion 1), the decode step, node 0's messages
in the layer that reaches the most remotes (criterion 6) and the busiest NIC with whether criterion 5 fires. From the
cells it prints §9.8's "where each fires" table, §9.7's (a) rows and §9.9's per-node paging at N = 2, at the gate's inputs.

How, so that the cells do not repeat the planner's method (crates/placement/src/placement/nodes.rs):
- The split is §9.2's rule in real shares: node 0 takes the share at which its mean window equals a remote's, the
  remotes share the rest equally, and each node is held to its pool's room where the pools hold every id, or keeps at
  least its room where they do not. Bisection on node 0's share; the planner instead places whole ids one at a time.
- A layer's windows are expectations over the count tuples of a token's k slots, whose probabilities are built one slot
  at a time: each slot moves a tuple's probability to the tuples one count higher, times the node's share (§9.7: "each
  routed slot lands on node j with the ownership probability"). The planner sums closed-form multinomials instead.
- Floats with the four basic operations only, so every host writes the same file.

The formulas and their sources:
- A layer (§9.7 (a), "the window is the max over the nodes of `Sh + slots·e` (node 0) or `λ + hop + posts + rows
  ingress + slots·e` (a remote)"): node 0 reads the shared bytes and its slots' experts; a remote that a slot reaches
  waits for the round trip (λ, plus the hop on a switch), a post for each other remote reached and the rows of the
  other remotes' slots through node 0's NIC, then reads its slots' experts; the layer waits for the slowest. The step
  is A plus every layer (§9.7 (a): "t = A/BW + max(...)/BW").
- Criterion 2 (§9.8): a plan's link share is its step less the step of the same nodes split with no link term; a node
  is used where the share stays within a tenth of the step.
- Criterion 7 (§9.8): a node past the pools that already hold every id is used where it gains at least twice the ruler.
  Criterion 1: a node the pools need is used; a remote whose own bytes pass its pool holds nothing and is idle for its
  pool; node 0's bytes past its pool refuse every plan.
- Criterion 5 (§9.8, "busy at most 50 % of the decode step and of the predicted prompt batch"): a node's NIC is busy, a
  direction, by the rows of the slots on the remotes, one row of n_embd values a slot (§9.7's "rows ingress"): node 0
  takes them all in, a remote sends its own. The handoffs go the other way, one a remote a token reaches, and never
  outnumber the rows, since a remote reached holds at least one slot. The prompt batch runs at
  §9.7's Amdahl term at the split: S₁ times the one-node token's bytes, A + Sh + R, over A and the slowest node's FFN
  bytes, node 0's with Sh. At an even split with no Sh that is 9.7's `1/(1−φ+φ/N)`. The planner states the same
  definition; it is shared on purpose, as the criterion it pins.
- Criterion 6 (§9.8, "at most 2·min(k, N−1) for (a)"): node 0 posts a handoff and takes the rows of each remote, so
  the layer that reaches the most remotes carries 2 × min(k, remotes used). Shared with the planner on purpose.

What it takes as given: the rows of crates/placement/tests/nodes_scale_inputs.tsv, among them each node's bytes beside
the routed ids (fixed0, fixed1), which today's card plan sets and the gate reads back from it, and the shapes' rounding
to whole Q4_K rows of n_embd values (`shape`, the gate's `Shape::bytes`: the shapes are the gate's inputs).

Tolerances and edges. The planner splits whole ids, so each value carries the largest change one id moved makes:
between node 0 and any single remote either way, and between two remotes, within each node's bounds (`*_tol`; a node's
paged ids carry one id). A cell whose NIC band holds the 50 % line, whose link share's band holds the tenth, or whose
gain's band holds twice the ruler is named in `edge` (`nic`, `link:<node>:<ppm>:<tol ppm>`, `gain:<node>:<ppm>:<tol
ppm>`), and its side is the derivation's value; the gate pins each edge's side by name and holds the planner's value to
the band besides.

    nodes_scale.py              print the cells
    nodes_scale.py --write      write crates/placement/tests/nodes_scale_cells.tsv
    nodes_scale.py --check      exit 1 when that file differs from the cells, or the doc's tables from --table
    nodes_scale.py --table      §9.8's "where each fires", §9.7's (a) rows and §9.9's N = 2 paging
    nodes_scale.py --self-test  hand-worked cases, the refusals, and --check

Refused by name (exit 2): an inputs row of another cell count, a name twice, a kind outside measured / derived /
assumed, a value that is not a whole number in ASCII digits, a missing or unknown row, an input outside its domain
(top-k outside 1 to the experts, a ruler outside 1 to 1000 permille, a rate, a pool, layers or a row size of 0, a term
under one row), and node 0's bytes past its pool. Exit 64: usage.

Python 3 standard library only.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.join(HERE, "..", "..", "crates", "placement", "tests")
INPUTS = os.path.join(TESTS, "nodes_scale_inputs.tsv")
CELLS = os.path.join(TESTS, "nodes_scale_cells.tsv")
DOC = os.path.join(HERE, "..", "..", "docs", "contrib", "dgx-spark.md")

KINDS = ("measured", "derived", "assumed")
MACHINE = ("read", "nic", "pool", "lambda_lo", "lambda_hi", "hop", "post", "row_value", "vocab", "ruler",
           "ruler_wide", "s1_lo", "s1_hi")
TERMS = ("layers", "n_embd", "experts", "top_k", "expert", "shared_lo", "shared_hi", "serial_lo", "serial_hi",
         "fixed0_lo", "fixed0_hi", "fixed1")
COLUMNS = ("model", "a", "n", "lambda_us", "ruler", "s1", "used", "idle", "paged", "paged_nodes", "step_ns", "step_tol",
           "nic", "nic_tol", "fires", "msgs", "edge")
NS = 1e9
# Each band's slack in whole units (ns, ‰, ppm): the band is truncated (under 1), the value rounded to the nearest (½),
# and the planner's value rounded down from sums of whole picoseconds (under 1 and a fraction): under 3, so 2 whole.
SLACK = 2


class Refused(Exception):
    pass


def parse(text, path):
    """The inputs file's rows as (name to int, models in file order)."""
    rows, models, header = {}, [], None
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        cells = line.split("\t")
        if header is None:
            header = cells
            if header != ["name", "value", "unit", "kind", "source"]:
                raise Refused(f"{path}: the header is {header}, not name, value, unit, kind, source")
            continue
        if len(cells) != 5:
            raise Refused(f"{path}: {cells[0]!r} has {len(cells)} cells, the header 5")
        name, value, _, kind, _ = cells
        if name in rows:
            raise Refused(f"{path}: {name} twice")
        if kind not in KINDS:
            raise Refused(f"{path}: {name}: kind {kind!r} is not one of {', '.join(KINDS)}")
        if not (value.isascii() and value.isdigit()):
            raise Refused(f"{path}: {name}: {value!r} is not a whole number")
        rows[name] = int(value)
        model = name.split("/")[0] if "/" in name else None
        if model is not None and model not in models:
            models.append(model)
    if header is None:
        raise Refused(f"{path}: no header row")
    return rows, models


def inputs(text, path):
    """The inputs as (machine, models): name to int, and model to term to int, models in file order."""
    rows, models = parse(text, path)
    machine = {}
    for k in MACHINE:
        if k not in rows:
            raise Refused(f"{path}: no {k} row")
        machine[k] = rows[k]
    terms = {}
    for m in models:
        terms[m] = {}
        for t in TERMS:
            if f"{m}/{t}" not in rows:
                raise Refused(f"{path}: no {m}/{t} row")
            terms[m][t] = rows[f"{m}/{t}"]
    known = set(MACHINE) | {f"{m}/{t}" for m in models for t in TERMS}
    for name in rows:
        if name not in known:
            raise Refused(f"{path}: {name} is no input")
    domain(machine, terms, path)
    return machine, terms


def domain(mc, terms, path):
    """Refuses an input the model is undefined at, before anything is computed from it."""
    for k in ("read", "nic", "pool", "row_value", "s1_lo", "s1_hi"):
        if mc[k] == 0:
            raise Refused(f"{path}: {k} is 0")
    for k in ("ruler", "ruler_wide"):
        if not 1 <= mc[k] <= 1000:
            raise Refused(f"{path}: {k} {mc[k]} is outside 1 to 1000 permille")
    for m, t in terms.items():
        for k in ("layers", "experts"):
            if t[k] == 0:
                raise Refused(f"{path}: {m}/{k} is 0")
        if not 1 <= t["top_k"] <= t["experts"]:
            raise Refused(f"{path}: {m}/top_k {t['top_k']} is outside 1 to its {t['experts']} experts")
        if t["n_embd"] < 256:
            raise Refused(f"{path}: {m}/n_embd {t['n_embd']} is under one Q4_K block of 256 values")
        row = t["n_embd"] // 256 * 144
        if t["expert"] < row:
            raise Refused(f"{path}: {m}/expert {t['expert']} B is under one row of {row} B")
        for end in ("lo", "hi"):
            if t[f"serial_{end}"] // row < mc["vocab"]:
                raise Refused(f"{path}: {m}/serial_{end} holds fewer rows than the head's {mc['vocab']}")


def load():
    try:
        with open(INPUTS, encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        raise Refused(f"{INPUTS}: {e.strerror}") from None
    return inputs(text, INPUTS)


def shape(mc, t, high):
    """The bytes the gate's shape lays out, in whole Q4_K rows of n_embd values: A (its attention, the head and one
    embedding row), a layer's Sh and one expert; and the rest of the case."""
    row = t["n_embd"] // 256 * 144
    layers = t["layers"]
    end = "hi" if high else "lo"
    attn = (t[f"serial_{end}"] // row - mc["vocab"]) // layers
    return {
        "layers": layers, "experts": t["experts"], "k": t["top_k"], "n_embd": t["n_embd"],
        "serial": (attn * layers + mc["vocab"] + 1) * row,
        "shared": t[f"shared_{end}"] // row // layers * row,
        "expert": t["expert"] // row * row,
        "fixed0": t[f"fixed0_{end}"], "fixed1": t["fixed1"],
    }


def falls(s, k):
    """The probability of each count tuple of a token's `k` slots on nodes of shares `s`, one slot at a time."""
    dist = {(0,) * len(s): 1.0}
    for _ in range(k):
        nxt = {}
        for c, p in dist.items():
            for j, x in enumerate(s):
                if x == 0.0:
                    continue
                d = c[:j] + (c[j] + 1,) + c[j + 1:]
                nxt[d] = nxt.get(d, 0.0) + p * x
        dist = nxt
    return dist


class Link:
    """A remote's link terms a layer, in seconds: the round trip, a further post, one slot's rows through a NIC."""

    def __init__(self, rt, post, row):
        self.rt, self.post, self.row = rt, post, row


def layer(s, sh, e, k, link):
    """A layer at shares `s` (node 0 first), in seconds: the expected slowest window, and each node's mean window."""
    n = len(s)
    slowest, mean = 0.0, [0.0] * n
    for c, p in falls(s, k).items():
        reached = sum(1 for cj in c[1:] if cj > 0)
        remote = sum(c[1:])
        w = [sh + c[0] * e]
        for cj in c[1:]:
            w.append(link.rt + link.post * (reached - 1) + link.row * (remote - cj) + cj * e if cj > 0 else 0.0)
        slowest += p * max(w)
        for j in range(n):
            mean[j] += p * w[j]
    return slowest, mean


class Plan:
    def __init__(self, s, step, tol):
        self.s, self.step, self.tol = s, step, tol


class Case:
    def __init__(self, mc, sh, n, lam):
        self.mc, self.sh = mc, sh
        read = mc["read"]
        self.A = sh["serial"] / read
        self.Sh = sh["shared"] / read
        self.e = sh["expert"] / read
        self.id = sh["layers"] * sh["expert"]
        if sh["fixed0"] > mc["pool"]:
            raise Refused(f"node 0's {sh['fixed0']} B beside the routed ids pass its pool of {mc['pool']} B")
        self.rooms = ((mc["pool"] - sh["fixed0"]) // self.id, max(0, mc["pool"] - sh["fixed1"]) // self.id)
        self.pool_idle = sh["fixed1"] > mc["pool"]
        hop = mc["hop"] if n == 4 else 0
        self.link = Link((lam + hop) / NS, mc["post"] / NS, mc["row_value"] * sh["n_embd"] / mc["nic"])
        self.bare = Link(0.0, 0.0, 0.0)

    def room(self, m):
        return self.rooms[0] + m * self.rooms[1]

    def limits(self, m):
        """Each used node's share's bounds with `m` remotes: up to its room where the pools hold every id, from it
        where they do not."""
        E = self.sh["experts"]
        r = [min(1.0, self.rooms[0] / E)] + [min(1.0, self.rooms[1] / E)] * m
        if self.room(m) >= E:
            return [(0.0, x) for x in r]
        return [(x, 1.0) for x in r]

    def shares(self, s0, m):
        return [s0] + [(1.0 - s0) / m] * m if m else [1.0]

    def balance(self, m, link):
        """§9.2's rule: node 0's share where its mean window equals a remote's, within the bounds."""
        if m == 0:
            return 1.0
        b = self.limits(m)
        lo, hi = max(b[0][0], 1.0 - m * b[1][1]), min(b[0][1], 1.0 - m * b[1][0])
        k = self.sh["k"]

        def diff(s0):
            _, mean = layer(self.shares(s0, m), self.Sh, self.e, k, link)
            return mean[0] - mean[1]

        if diff(lo) >= 0:
            return lo
        if diff(hi) <= 0:
            return hi
        for _ in range(64):
            mid = (lo + hi) / 2
            if diff(mid) < 0:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def step(self, s, link):
        slowest, _ = layer(s, self.Sh, self.e, self.sh["k"], link)
        return self.A + self.sh["layers"] * slowest

    def moves(self, s):
        """The shares one id away from `s`: one id from any used node to any other, within each node's bounds."""
        d = 1.0 / self.sh["experts"]
        b = self.limits(len(s) - 1)
        out = []
        for i in range(len(s)):
            for j in range(len(s)):
                if i == j or s[i] - d < b[i][0] or s[j] + d > b[j][1]:
                    continue
                t = list(s)
                t[i] -= d
                t[j] += d
                out.append(t)
        return out

    def plan(self, m, link):
        """The split of `m` remotes: its shares, its step and the step's one-id tolerance."""
        s = self.shares(self.balance(m, link), m)
        t = self.step(s, link)
        return Plan(s, t, max([abs(self.step(x, link) - t) for x in self.moves(s)], default=0.0))

    def nic(self, s, t, s1):
        """The busiest node's NIC busy share, a fraction, of the plan at shares `s` and step `t`."""
        sh = self.sh
        k, L, m = sh["k"], sh["layers"], len(s) - 1
        if m == 0:
            return 0.0
        rows = [0.0] * (m + 1)
        for c, p in falls(s, k).items():
            for j in range(1, m + 1):
                rows[0] += p * c[j]
                rows[j] += p * c[j]
        row = self.mc["row_value"] * sh["n_embd"]
        A, Sh, R = sh["serial"], sh["shared"] * L, k * self.id
        ffn = max([Sh + s[0] * R] + [x * R for x in s[1:]])
        rate = s1 * (A + Sh + R) / (A + ffn)
        busiest = 0.0
        for j in range(m + 1):
            b = L * row * rows[j] / self.mc["nic"]
            busiest = max(busiest, b / t, b * rate)
        return busiest


def ppm(x):
    return round(x * 1e6)


def busiest(c, p, m, s1):
    """The plan's busiest NIC share, its one-id tolerance, and whether its band holds the 50 % line."""
    busy = c.nic(p.s, p.step, s1)
    tol = max([abs(c.nic(x, c.step(x, c.link), s1) - busy) for x in c.moves(p.s)], default=0.0)
    return busy, tol, m > 0 and abs(busy - 0.5) <= tol


def cell(mc, sh, n, lam, ruler):
    """The plan of shape `sh` on `n` nodes at round trip `lam` ns: the case, m remotes used, idle [(node, why)], the
    edges of the node choice, and the plan."""
    c = Case(mc, sh, n, lam)
    E = sh["experts"]
    m, idle, edges = 0, [], []
    now = c.plan(0, c.link)
    for j in range(1, n):
        if c.pool_idle:
            idle.append((j, "pool"))
            continue
        nxt = c.plan(m + 1, c.link)
        if c.room(m) < E and c.rooms[1] > 0:
            m, now = m + 1, nxt
            continue
        bare = c.plan(m + 1, c.bare)
        share = (nxt.step - bare.step) / nxt.step
        share_tol = (nxt.tol + bare.tol) / nxt.step
        if abs(share - 0.1) <= share_tol:
            edges.append(f"link:{j}:{ppm(share)}:{int(share_tol * 1e6) + SLACK}")
        if share > 0.1:
            idle.append((j, "link"))
            continue
        gain = (now.step - nxt.step) / now.step
        gain_tol = (now.tol + nxt.tol) / now.step
        if abs(gain - 2 * ruler / 1000) <= gain_tol:
            edges.append(f"gain:{j}:{ppm(gain)}:{int(gain_tol * 1e6) + SLACK}")
        if gain < 2 * ruler / 1000:
            idle.append((j, "gain"))
            continue
        m, now = m + 1, nxt
    return c, m, idle, edges, now


def derive(mc, terms):
    """Every cell, in the gate's order: model, A low then high, (ruler, S₁), λ, N."""
    out = []
    pairs = ((mc["ruler"], mc["s1_lo"]), (mc["ruler"], mc["s1_hi"]), (mc["ruler_wide"], mc["s1_hi"]))
    for model, t in terms.items():
        for high in (False, True):
            sh = shape(mc, t, high)
            for ruler, s1 in pairs:
                for lam in (mc["lambda_lo"], mc["lambda_hi"]):
                    for n in range(1, 5):
                        try:
                            c, m, idle, edges, p = cell(mc, sh, n, lam, ruler)
                        except Refused as e:
                            raise Refused(f"{model} A {'high' if high else 'low'}: {e}") from None
                        busy, busy_tol, on_line = busiest(c, p, m, s1)
                        if on_line:
                            edges.insert(0, "nic")
                        E = sh["experts"]
                        paging = c.room(m) < E
                        rooms = [c.rooms[0]] + [c.rooms[1]] * m
                        per = [round(x * E - r) if paging else 0 for x, r in zip(p.s, rooms)] + [0] * (n - 1 - m)
                        out.append({
                            "model": model, "a": "high" if high else "low", "n": n, "lambda_us": lam // 1000,
                            "ruler": ruler, "s1": s1, "used": m + 1,
                            "idle": ",".join(f"{j}:{w}" for j, w in idle) or "-",
                            "paged": max(0, E - c.room(m)), "paged_nodes": ",".join(map(str, per)),
                            "step_ns": round(p.step * NS), "step_tol": int(p.tol * NS) + SLACK,
                            "nic": round(busy * 1000), "nic_tol": int(busy_tol * 1000) + SLACK,
                            "fires": int(m > 0 and busy > 0.5), "msgs": 2 * min(sh["k"], m),
                            "edge": ",".join(edges) or "-",
                        })
    return out


def text(cells):
    lines = ["# nodes_scale's expected cells (docs/contrib/dgx-spark.md §9.9), written by tools/flow/nodes_scale.py "
             "--write from nodes_scale_inputs.tsv; do not edit.",
             "\t".join(COLUMNS)]
    for c in cells:
        lines.append("\t".join(str(c[k]) for k in COLUMNS))
    return "\n".join(lines) + "\n"


def span(values, fmt):
    lo, hi = fmt(min(values)), fmt(max(values))
    return lo if lo == hi else f"{lo} to {hi}"


def blocks(mc, terms, cells):
    """The doc's derived tables: §9.8's criteria, §9.7's (a) rows and §9.9's N = 2 paging."""
    at = {(c["model"], c["a"], c["n"], c["lambda_us"], c["ruler"], c["s1"]): c for c in cells}
    lo, hi = mc["lambda_lo"] // 1000, mc["lambda_hi"] // 1000
    r, s1_lo, s1_hi = mc["ruler"], mc["s1_lo"], mc["s1_hi"]

    def where(model, pick, rate=s1_hi):
        """The cells `pick` names, by N; an N alone is every λ and A at it."""
        hits = []
        for n in range(1, 5):
            each = [(lam, [a for a in ("low", "high") if pick(at[(model, a, n, lam, r, rate)])]) for lam in (lo, hi)]
            if all(len(ends) == 2 for _, ends in each):
                hits.append(f"N = {n}")
                continue
            hits += [f"N = {n}, λ {lam}, A {' and '.join(ends)}" for lam, ends in each if ends]
        return "; ".join(hits) or "never"

    fires = [f"| Model | Decode step (ms) at the gate's inputs, N = 1 → 2 → 3 → 4, λ {lo} / {hi}, A low "
             f"| Criterion 2 fires | Criterion 7 fires | Criterion 5 fires, S₁ {s1_lo:,} | Criterion 5 fires, "
             f"S₁ {s1_hi:,} |", "|---|---|---|---|---|---|"]
    for model in terms:
        steps = []
        for n in range(1, 5):
            a, b = (at[(model, "low", n, lam, r, s1_hi)]["step_ns"] / 1e6 for lam in (lo, hi))
            steps.append(f"{a:.1f}" if n == 1 else f"{a:.1f} / {b:.1f}")
        fires.append(f"| {model} | {' → '.join(steps)} | {where(model, lambda c: 'link' in c['idle'])} "
                     f"| {where(model, lambda c: 'gain' in c['idle'])} "
                     f"| {where(model, lambda c: c['fires'] == 1, rate=s1_lo)} "
                     f"| {where(model, lambda c: c['fires'] == 1)} |")

    star = ["| (a) star at the gate's inputs | N | Nodes used | Decode, one request (tok/s) | Prompt (× S₁) "
            "| Node 0's NIC ingress cap (tok/s) | Link bytes a token (MB) | Messages a token |",
            "|---|---|---|---|---|---|---|---|"]
    for model, t in terms.items():
        for n in (2, 4):
            cs = [at[(model, a, n, lam, r, s1_hi)] for a in ("low", "high") for lam in (lo, hi)]
            used, decode, prompt, cap, link, msgs = [], [], [], [], [], []
            for a in ("low", "high"):
                sh = shape(mc, t, a == "high")
                for lam in (lo, hi):
                    c, m, _, _, p = cell(mc, sh, n, lam * 1000, r)
                    k, L = sh["k"], sh["layers"]
                    A, Sh, R = sh["serial"], sh["shared"] * L, k * c.id
                    used.append(m + 1)
                    decode.append(NS / round(p.step * NS))
                    prompt.append((A + Sh + R) / (A + max([Sh + p.s[0] * R] + [x * R for x in p.s[1:]])))
                    row = mc["row_value"] * sh["n_embd"]
                    remote = 1.0 - p.s[0]
                    cap.append(mc["nic"] / (L * k * remote * row) if m else 0.0)
                    reached = 0.0
                    for cc, q in falls(p.s, k).items():
                        reached += q * sum(1 for x in cc[1:] if x > 0)
                    link.append(L * row * (reached + k * remote) / 1e6)
                    msgs.append(2 * L * reached)
            paged = [x["paged"] for x in cs]
            note = f" (pool only: pages {span(paged, str)} ids)" if max(paged) else ""
            star.append(f"| {model} | {n} | {span(used, str)} | {span(decode, lambda v: f'{v:.1f}')}{note} "
                        f"| {span(prompt, lambda v: f'{v:.2f}')} | {span(cap, lambda v: f'{v:,.0f}')} "
                        f"| {span(link, lambda v: f'{v:.2f}')} | {span(msgs, lambda v: f'{v:.0f}')} |")
    paging = ["| N = 2 at the gate's inputs | A | λ (µs) | Paged ids, total | Paged ids on node 0 | Paged ids on node 1 |",
              "|---|---|---|---|---|---|"]
    for model in terms:
        for a in ("low", "high"):
            for lam in (lo, hi):
                c = at[(model, a, 2, lam, r, s1_hi)]
                per = " | ".join(c["paged_nodes"].split(","))
                paging.append(f"| {model} | {a} | {lam} | {c['paged']} | {per} |")
    return [fires, star, paging]


def table(mc, terms, cells):
    return "\n\n".join("\n".join(b) for b in blocks(mc, terms, cells))


def doc_differs(bs, text):
    """The first line of each block of `bs` that the doc `text` does not carry as printed, alone under its header."""
    doc = text.split("\n")
    off = []
    for b in bs:
        heads = [i for i, line in enumerate(doc) if line == b[0]]
        if len(heads) != 1 or doc[heads[0]:heads[0] + len(b)] != b or heads[0] + len(b) < len(doc) and \
                doc[heads[0] + len(b)].startswith("|"):
            off.append(b[0])
    return off


def check():
    try:
        with open(CELLS, encoding="utf-8") as f:
            have = f.read()
    except OSError as e:
        print(f"nodes_scale: {CELLS}: {e.strerror}", file=sys.stderr)
        return 1
    machine, terms = load()
    cells = derive(machine, terms)
    rc = 0
    if have != text(cells):
        print(f"nodes_scale: {CELLS} is not the derivation of {INPUTS}: run tools/flow/nodes_scale.py --write",
              file=sys.stderr)
        rc = 1
    try:
        with open(DOC, encoding="utf-8") as f:
            doc = f.read()
    except OSError as e:
        print(f"nodes_scale: {DOC}: {e.strerror}", file=sys.stderr)
        return 1
    for head in doc_differs(blocks(machine, terms, cells), doc):
        print(f"nodes_scale: {DOC} does not carry the --table block {head!r} as printed", file=sys.stderr)
        rc = 1
    return rc


def self_test():
    ok = True

    def expect(what, cond):
        nonlocal ok
        print(f"{'ok  ' if cond else 'FAIL'} {what}")
        ok = ok and cond

    def near(a, b):
        return abs(a - b) <= 1e-12 * max(1.0, abs(b))

    # Two slots at shares (1/2, 1/2): both on node 0 a quarter, one each a half, both remote a quarter.
    d = falls([0.5, 0.5], 2)
    expect("two slots on a pair: 1/4, 1/2, 1/4", d == {(2, 0): 0.25, (1, 1): 0.5, (0, 2): 0.25})
    d = falls([0.25, 0.25, 0.25, 0.25], 10)
    expect("ten slots on four nodes: 286 tuples, one in all, (10, 0, 0, 0) at 4^-10",
           len(d) == 286 and near(sum(d.values()), 1.0) and near(d[(10, 0, 0, 0)], 1 / 1048576))
    # (2, 1, 1) at shares (1/2, 1/4, 1/4): 12 orders, each (1/2)² × 1/4 × 1/4 = 1/64.
    expect("the tuple (2, 1, 1) at (1/2, 1/4, 1/4)", near(falls([0.5, 0.25, 0.25], 4)[(2, 1, 1)], 12 / 64))

    # Sh 1, e 2, rt 4. Both on node 0 (1/4): 1 + 2e = 5; one each (1/2): max(1 + e, rt + e) = 6; both remote (1/4):
    # max(1, rt + 2e) = 8. Expected 1.25 + 3 + 2 = 6.25. Node 0's mean window: 1 + e at one slot on average, 3; the
    # remote's: (1/2) × 6 + (1/4) × 8 = 5.
    slowest, mean = layer([0.5, 0.5], 1.0, 2.0, 2, Link(4.0, 0.0, 0.0))
    expect("a pair's layer: the expected slowest window", near(slowest, 6.25))
    expect("a pair's layer: the mean windows", near(mean[0], 3.0) and near(mean[1], 5.0))
    # Three nodes at (0, 1/2, 1/2), two slots: one on each remote (1/2) posts once more and takes the other's rows
    # ahead: rt + post + row + e = 4 + 1 + 0.5 + 2 = 7.5; both on one remote (1/2): rt + 2e = 8. Expected 7.75.
    slowest, _ = layer([0.0, 0.5, 0.5], 0.0, 2.0, 2, Link(4.0, 1.0, 0.5))
    expect("a ring's layer: a post and the rows ahead of a second remote", near(slowest, 7.75))

    mc = {"read": 1, "nic": 1, "pool": 100, "lambda_lo": 0, "lambda_hi": 0, "hop": 0, "post": 0, "row_value": 4,
          "vocab": 0, "ruler": 10, "ruler_wide": 40, "s1_lo": 1, "s1_hi": 1}
    sh = {"layers": 1, "experts": 4, "k": 1, "n_embd": 256, "serial": 0, "shared": 0, "expert": 10,
          "fixed0": 70, "fixed1": 60}
    c = Case(mc, sh, 2, 0)
    # Rooms (100 - 70) / 10 = 3 and (100 - 60) / 10 = 4: one remote holds every id; with no link and no shared
    # bytes the windows balance at half the ids each, inside both rooms.
    expect("the rooms", c.rooms == (3, 4) and not c.pool_idle)
    expect("no link, no shared bytes: half each", near(c.balance(1, c.bare), 0.5))
    expect("the bounds where the pools hold every id", c.limits(1) == [(0.0, 0.75), (0.0, 1.0)])
    # A shared read of 3 on node 0 against one slot of 10: s0 × 10 + 3 = (1 - s0) × 10 at s0 = 0.35. One slot: on
    # node 0 (0.35) the window is 3 + 10, on the remote (0.65) max(3, 10): a step of 4.55 + 6.5 = 11.05. One id moved
    # (a quarter): node 0 at 0.1 gives 1.3 + 9 = 10.3, at 0.6 gives 7.8 + 4 = 11.8, so the tolerance is 0.75.
    c.Sh = 3.0
    p = c.plan(1, c.bare)
    expect("plan(): node 0's shared bytes move ids to the remote, and its one-id tolerance",
           near(p.s[0], 0.35) and near(p.step, 11.05) and near(p.tol, 0.75))
    c2 = Case(mc, dict(sh, fixed0=90, fixed1=90), 2, 0)
    # Rooms 1 and 1 of 4 ids: the pools page 2, and each node keeps at least its room.
    expect("paging: each node at least its room", c2.room(1) == 2 and c2.limits(1) == [(0.25, 1.0), (0.25, 1.0)])
    c3 = Case(mc, dict(sh, fixed1=95), 2, 0)
    expect("a room-0 remote has paging bounds and fits its fixed-byte budget",
           c3.rooms == (3, 0) and not c3.pool_idle and c3.limits(1) == [(0.75, 1.0), (0.0, 1.0)])
    _, m, idle, _, _ = cell(mc, dict(sh, fixed1=95), 2, 0, 10)
    expect("a remote of room 0 must gain twice the ruler", m == 0 and idle == [(1, "gain")])
    _, m, idle, _, _ = cell(mc, dict(sh, fixed1=101), 3, 0, 10)
    expect("a remote whose own bytes pass its pool is idle for its pool", m == 0 and idle == [(1, "pool"), (2, "pool")])
    try:
        Case(mc, dict(sh, fixed0=101), 1, 0)
        expect("node 0's bytes past its pool: not refused", False)
    except Refused as e:
        expect(f"node 0's bytes past its pool: {e}", "node 0" in str(e))
    # Four nodes on a switch: the hop joins λ, a ring of three has none. One slot, every id on node 1: a layer is
    # λ + hop + e = 25 + 1.5 + 10 s at λ 25 s, so a step of A 2 + 36.5.
    sw = dict(mc, hop=1_500_000_000, lambda_lo=25_000_000_000)
    c4 = Case(sw, dict(sh, serial=2), 4, sw["lambda_lo"])
    expect("the switch's hop at N = 4", near(c4.step([0.0, 1.0, 0.0, 0.0], c4.link), 38.5)
           and near(Case(sw, sh, 3, sw["lambda_lo"]).link.rt, 25.0))
    # NIC, node 0 at s0 = 1/2, k = 1, one layer, rows of 4 × 256 = 1,024 B: a remote slot half the time, so 512 B a
    # token. Decode at a step of 1,024 s on a NIC of 1 B/s: half busy. The prompt at S₁ 1 with no A or Sh runs
    # 10 / 5 = 2 tokens a second over the slowest node's FFN: 1,024 B a second, twice the NIC.
    expect("the NIC: the prompt's share where it dominates", near(c.nic([0.5, 0.5], 1024.0, 1), 1024.0))
    expect("the NIC: the decode's share where it dominates", near(c.nic([0.5, 0.5], 1024.0, 1e-9), 0.5))
    # Three nodes at (1/2, 1/4, 1/4): node 0 takes every remote slot's rows, 512 B a token, each remote its own
    # quarter, 256 B. Decode at a step of 1,024 s: node 0 half busy.
    expect("the NIC: node 0 takes every remote's rows", near(Case(mc, sh, 3, 0).nic([0.5, 0.25, 0.25], 1024.0, 1e-9),
                                                             0.5))
    # Sh of 10 on node 0 against R = 10: node 0's FFN is 10 + 5, the remote's 5, so the prompt runs 20 / 15 tokens a
    # second at S₁ 1 and node 0 takes 512 B a token: 682.67 B a second.
    cs = Case(mc, dict(sh, shared=10), 2, 0)
    expect("the NIC: the prompt's slowest FFN is node 0's with Sh", near(cs.nic([0.5, 0.5], 1e9, 1), 512 * 20 / 15))
    # On a NIC of 10^6 B/s the same split's prompt at S₁ 488.28125 is busy one half (1,024 µB a token of S₁); node 0
    # at a quarter of the ids takes 768 µB at the same rate, so the band holds the line: an edge.
    cn = Case(dict(mc, nic=10 ** 6), sh, 2, 0)
    busy, tol, on = busiest(cn, cn.plan(1, cn.link), 1, 488.28125)
    expect("the NIC on its half within one id: an edge", near(busy, 0.5) and tol > 0.3 and on)
    # A gain on twice an integer ruler. A = 248 and two slots of 1 on one node: a step of 250. A second node takes one
    # slot of two half the time, so the window falls to 1.5: a gain of 0.2 %, twice a ruler of 1 permille; one id
    # moved (of two) moves the step by 0.5. The node is named an edge and used, the line not under it.
    gm = dict(mc, nic=10 ** 12, pool=10 ** 6)
    gs = dict(sh, experts=2, k=2, expert=1, serial=248, fixed0=0, fixed1=0)
    _, m, idle, edges, _ = cell(gm, gs, 2, 0, 1)
    expect("a gain on twice an integer ruler is an edge, on the used side", edges == ["gain:1:2000:2002"] and m == 1)
    # A = 123, one id of a hundred: the gain is 0.5 / 125 = 0.4 %, between a ruler of 3 permille and twice it, and one
    # id moved (0.01 of the share) moves the step by 0.0002: no edge, and the node is idle.
    _, m, idle, edges, _ = cell(gm, dict(gs, experts=100, serial=123), 2, 0, 3)
    expect("a gain under twice the ruler and over it leaves the node idle", (m, idle, edges) == (0, [(1, "gain")], []))
    # One slot, two ids, A 3.4, e 1 and a round trip of 2: the split balances at s0 = 3/4 (s0 = (1 - s0) × 3), so the
    # step is 3.4 + 1.5 against 3.4 + 1 with no link: a link share of 0.5 / 4.9, past the tenth. One id moved puts
    # s0 at 1/4 and the step at 3.4 + 2.5, so the band holds the tenth: an edge, and the node idle.
    lm = dict(gm, read=10, lambda_lo=2 * 10 ** 9)
    _, m, idle, edges, _ = cell(lm, dict(gs, k=1, expert=10, serial=34), 2, lm["lambda_lo"], 10)
    expect("a link share on the tenth within one id: an edge, on its side", (m, idle) == (0, [(1, "link")])
           and edges == [f"link:1:{round(0.5 / 4.9 * 1e6)}:{int(1.0 / 4.9 * 1e6) + 2}"])
    cm = Case(mc, dict(sh, fixed0=0, fixed1=0), 3, 0)
    moves = cm.moves([0.5, 0.25, 0.25])
    expect("one id moved: node 0 to and from each remote, and between the remotes",
           len(moves) == 6 and [0.5, 0.0, 0.5] in moves and [0.5, 0.5, 0.0] in moves)

    machine, terms = load()
    cells = derive(machine, terms)
    by = {(x["model"], x["a"], x["n"], x["lambda_us"], x["ruler"], x["s1"]): x for x in cells}
    glm2 = [x for x in cells if x["model"] == "GLM" and x["n"] == 2]
    expect("GLM held whole on two Sparks at every A, lambda, ruler and prompt rate",
           len(glm2) == 12 and all((x["paged"], x["paged_nodes"], x["used"]) == (0, "0,0", 2) for x in glm2))
    v41 = [x for x in cells if x["model"] == "V4.1" and x["n"] == 2]
    want = {("low", 25): (62, "11,51"), ("low", 70): (62, "28,34"),
            ("high", 25): (67, "0,67"), ("high", 70): (67, "0,67")}
    expect("V4.1's two-node paging pinned per node at every A, lambda, ruler and prompt rate",
           len(v41) == 12 and all((x["paged"], x["paged_nodes"]) == want[(x["a"], x["lambda_us"])] for x in v41))
    expect("V4.1 is whole on three Sparks", all(x["paged"] == 0 for x in cells if x["model"] == "V4.1" and x["n"] == 3))
    expect("the N = 2 paging table is included",
           len(blocks(machine, terms, cells)) == 3)
    q4 = by[("Qwen3.8", "low", 4, 70, machine["ruler"], machine["s1_hi"])]
    expect("Qwen3.8 on four switched Sparks at λ 70 leaves nodes idle for the link", "link" in q4["idle"])
    expect("V4.1 on four Sparks: a handoff and its rows for each of three remotes, none on one",
           by[("V4.1", "low", 4, 25, machine["ruler"], machine["s1_hi"])]["msgs"] == 6
           and by[("V4.1", "low", 1, 25, machine["ruler"], machine["s1_hi"])]["msgs"] == 0)
    expect("each cell's paged ids per node sum to its paged ids",
           all(sum(map(int, x["paged_nodes"].split(","))) == x["paged"] for x in cells))

    with open(INPUTS, encoding="utf-8") as f:
        good = f.read()
    lines = good.split("\n")
    first = next(i for i, x in enumerate(lines) if x.startswith("read\t"))

    def row(name, value):
        """The inputs with row `name`'s value replaced."""
        return "\n".join(f"{name}\t{value}\t" + x.split("\t", 2)[2] if x.startswith(name + "\t") else x
                         for x in lines)

    def refused(what, body, reason):
        try:
            inputs(body, "in.tsv")
        except Refused as e:
            expect(f"{what}: {e}", reason in str(e))
            return
        expect(f"{what}: not refused", False)

    def accepted(what, body):
        try:
            inputs(body, "in.tsv")
            expect(what, True)
        except Refused as e:
            expect(f"{what}: refused, {e}", False)

    accepted("the inputs as they are", good)
    refused("a row of four cells", "\n".join(lines[:first] + [lines[first].rsplit("\t", 1)[0]] + lines[first + 1:]),
            "has 4 cells")
    refused("a name twice", "\n".join(lines[:first + 1] + [lines[first]] + lines[first + 1:]), "read twice")
    refused("a kind of no kind", "\n".join(lines[:first] + [lines[first].replace("\tmeasured\t", "\tguessed\t")]
                                           + lines[first + 1:]), "kind 'guessed'")
    refused("a value not whole", row("read", "2.5"), "not a whole number")
    refused("a value in other digits", row("read", "٢"), "not a whole number")
    refused("a missing row", "\n".join(lines[:first] + lines[first + 1:]), "no read row")
    refused("a row that is no input", good + "spare\t1\tx\tassumed\tx\n", "spare is no input")
    refused("top-k 0", row("GLM/top_k", 0), "GLM/top_k 0")
    refused("top-k past the experts", row("GLM/top_k", 289), "GLM/top_k 289")
    accepted("top-k at the experts", row("GLM/top_k", 288))
    refused("a ruler of 0", row("ruler", 0), "ruler 0")
    refused("a ruler past 1000", row("ruler_wide", 1001), "ruler_wide 1001")
    accepted("rulers of 1 and 1000", row("ruler", 1).replace("ruler_wide\t40\t", "ruler_wide\t1000\t"))
    refused("a read rate of 0", row("read", 0), "read is 0")
    refused("a NIC of 0", row("nic", 0), "nic is 0")
    refused("a prompt rate of 0", row("s1_lo", 0), "s1_lo is 0")
    refused("no layers", row("V4.1/layers", 0), "V4.1/layers is 0")
    refused("a row width under a block", row("Qwen3.8/n_embd", 255), "Qwen3.8/n_embd 255")
    accepted("a row width of one block", row("Qwen3.8/n_embd", 256))
    refused("an expert under a row", row("Qwen3.8/expert", 1439), "Qwen3.8/expert 1439")
    try:
        derive(*inputs(row("GLM/fixed0_lo", machine["pool"] + 1), "in.tsv"))
        expect("node 0's bytes past its pool in the inputs: not refused", False)
    except Refused as e:
        expect(f"node 0's bytes past its pool in the inputs: {e}", str(e).startswith("GLM A low: node 0's"))
    bs = [["| a | b |", "|---|---|", "| 1 | 2 |"]]
    expect("a doc that carries a block as printed", doc_differs(bs, "x\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\ny") == [])
    expect("a doc whose block differs in a cell, has a row more, or lacks it",
           all(doc_differs(bs, d) == ["| a | b |"] for d in ("| a | b |\n|---|---|\n| 1 | 3 |\n",
                                                               "| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n", "x\n")))
    expect("the cells file and the doc's tables are the derivation's", check() == 0)
    print(f"self-test: {'ok' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv):
    if len(argv) > 1 or (argv and argv[0] not in ("--write", "--check", "--table", "--self-test")):
        print(f"nodes_scale: usage\n{__doc__}", file=sys.stderr)
        return 64
    try:
        if argv == ["--self-test"]:
            return self_test()
        if argv == ["--check"]:
            return check()
        machine, terms = load()
        cells = derive(machine, terms)
    except Refused as e:
        print(f"nodes_scale: {e}", file=sys.stderr)
        return 2
    if argv == ["--write"]:
        with open(CELLS, "w", encoding="utf-8") as f:
            f.write(text(cells))
    elif argv == ["--table"]:
        print(table(machine, terms, cells))
    else:
        sys.stdout.write(text(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
