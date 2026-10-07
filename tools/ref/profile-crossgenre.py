#!/usr/bin/env python3
"""The residency profile seed's cross-genre gate: a profile learned on one genre, seeding another.

    tools/ref/profile-crossgenre.py gate <family> --data DIR [--json PATH]
    tools/ref/profile-crossgenre.py contexts <set dir> --genres G:N[,G:N...]
    tools/ref/profile-crossgenre.py smoke --out DIR
    tools/ref/profile-crossgenre.py --self-test

The engine seeds each layer's card with the residency profile's top n_l experts (count descending, id
ascending; crates/placement/src/placement/profile.rs) in place of the id prefix. For every ordered
pair (A, B) of the family's genres, at each of its card shares, `gate` learns A's profile (the
undecayed per-(layer, expert) count of the router's picks over A's learn part) and scores it on B's
score part against the id prefix, in three arms:

    static   the seed's card, never changed, over B's whole score part (chat: each held reply)
    tokN     the residency rule from the seed, counts at zero, over B's first N tokens (chat: each
             held reply's first min(N, reply)) — mid's parameters every 4 tokens, unpinned, one spare,
             a flip landing d trace tokens after its boundary (the token clock in a one-row trace)

The hit is the share of routed picks the card serves; the cell is diff = seed - prefix, in points. A
cell fails when -diff is past its pair's pinned loss plus 2 SE of its own noise:

    chat B          the paired per-request diffs over the held requests, SE = SD / sqrt(n)
    corpus static   the 512-token block diffs folded into blocks of 2048 (their lag-1 correlation
                    is ~0.6, so 512-token blocks understate the SE), SE = bootstrap SD of the means
    corpus tokN     the same cold window swept over the score part (96 every 512, 672 every 1024),
                    SE = SD / sqrt(windows)

The pinned loss is 0 for every pair but the ones PINS names (the measured worst pair). Every replay
is router-residency.py's own machinery (Rule, replay, seed_resident, rank, n_plan, spread, Set),
imported, never re-implemented. A family with fewer than two genres is refused by name. Exit 0 when
every cell holds, 1 when one fails, 2 on a refusal.

`contexts` writes a chat set's contexts.tsv (tools/bloomery/route_trace.py's columns) from its call
rows, for a set the engine wrote one call a prompt (generate_qwen3moe --arm, one arm a held prompt of
tools/ref/chat-ids.sh's corpora): --genres names the calls' genres in order and how many each, and
each genre's calls alternate learn, held, learn, ... so both parts hold every genre. A file already
there, or counts that do not sum to the calls, are refused by name.

`smoke --out DIR` writes the synthetic pair the self-test reads (tools/ref/data/profile-smoke):
genre `beta` routes the low ids the id prefix holds, genre `alpha` is a mix of its own experts and
a third of beta's, so alpha's profile on beta is below the prefix beyond noise — the red path.
"""
import argparse
import importlib.util
import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout

try:
    import numpy as np
except ImportError:
    sys.exit("profile-crossgenre: needs numpy (router-residency.py's replays): python3 -m pip install numpy")

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("router_residency", os.path.join(_here, "router-residency.py"))
rr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rr)

SMOKE = os.path.join(_here, "data", "profile-smoke")
BOOT_N = 10000
NOISE_SEED = 20261007

# Each family's genres and their parts. corpus: a set split at `half` (learn [0, half), score
# [half, T)); chat: a route-trace-chat.py set whose contexts.tsv splits its requests learn / held.
# d: the flip's landing delay in trace tokens, round(2E) for the genre's measured E (V4.1 prose
# band 3.3, chat 2.3 — the resclock report's values). shares: the plan's n_l, or a card share of
# n_expert x eligible layers spread as the planner spreads it. block: the static noise's small block.
FAMILIES = {
    "v41": dict(rr="v41", corpora=("prose", "code", "korean"), chat="d2-chat", half=25000,
                d_corpus=7, d_chat=5, shares=(("18", "plan"), ("22.3", 5482 / 24576)), block=512,
                arms=(("static", None, None, None), ("tok96", 96, 512, 48), ("tok672", 672, 1024, 24))),
    "q38": dict(rr="q38", corpora=("q38-prose-trace-09300532",), chat=None, half=3072,
                d_corpus=7, d_chat=5, shares=(("plan", "plan"),), block=128,
                arms=(("static", None, None, None), ("tok96", 96, 128, 16))),
    "smoke": dict(rr=None, n_expert=16, n_used=2, eligible=[0, 1], corpora=("alpha", "beta"), chat=None,
                  half=2048, d_corpus=2, d_chat=2, shares=(("25", 0.25),), block=32,
                  arms=(("static", None, None, None), ("tok96", 96, 128, 16))),
}

# PIN(2026-10-07): a mixed chat profile seeding the code corpus loses to the id prefix: the profxgenre
# replay (V4.1, 16 learn->score pairs, the leader's specs/leader/profxgenre/report.md §3, §5,
# pairs.json) measured chat->code at -3.1 static / -2.2 after 96 tokens / -0.7 after 672 at the 18 %
# share and -3.3 / -2.1 / -0.7 at 22.3 %, 6-30x past its noise; the cause is the genre mix in one file
# (chat's code requests alone give +11.8). The user re-pinned "never below the id prefix" to "never
# worse than the measured worst-pair loss": every other pair cleared the prefix (+6...+58 static).
PINS = {
    ("v41", "d2-chat", "code", "18"): {"static": 3.1, "tok96": 2.2, "tok672": 0.7},
    ("v41", "d2-chat", "code", "22.3"): {"static": 3.3, "tok96": 2.1, "tok672": 0.7},
}


class GateError(Exception):
    pass


def family(name):
    if name not in FAMILIES:
        raise GateError(f"family {name!r} is not in the table: {', '.join(FAMILIES)}")
    f = FAMILIES[name]
    genres = list(f["corpora"]) + ([f["chat"]] if f["chat"] else [])
    if len(genres) < 2:
        raise GateError(
            f"family {name} has {len(genres)} genre ({', '.join(genres)}): a cross-genre pair needs two — "
            "dump a second genre's route trace (BLOOMERY_ROUTE_TRACE) and add it to FAMILIES")
    return f


def facts(f):
    """(n_expert, n_used, eligible layers) of the family."""
    if f["rr"]:
        F = rr.family(f["rr"])
        return F["n_expert"], F["n_used"], F["eligible"]
    return f["n_expert"], f["n_used"], f["eligible"]


def shares(f, E, lay):
    out = []
    for label, s in f["shares"]:
        if s == "plan":
            out.append((label, rr.n_plan(f["rr"])))
        else:
            card = round(s * E * len(lay))
            n = rr.spread(card, lay)
            out.append((label, [n[l] for l in lay]))
    return out


def open_set(f, data, name):
    E, K, lay = facts(f)
    s = rr.Set(rr.set_dir(data, name), f["rr"])
    if (s.E, s.K) != (E, K) or any(l not in s.layers for l in lay):
        raise GateError(f"{s.dir}: {s.E} x top-{s.K} over layers {s.layers}; the family is {E} x top-{K} "
                        f"over {lay}")
    return s


def load(f, data):
    """Each genre: its learn rows (the profile's) and its score parts."""
    E, K, lay = facts(f)
    G = {}
    for g in f["corpora"]:
        s = open_set(f, data, g)
        if s.T <= f["half"]:
            raise GateError(f"{s.dir}: {s.T} tokens, none past the learn half {f['half']}")
        X = s.stack(lay)
        G[g] = dict(kind="corpus", learn=X[:f["half"]], score=X[f["half"]:])
    if f["chat"]:
        s = open_set(f, data, f["chat"])
        X = s.stack(lay)
        reqs, _ = rr.rt.read_contexts(s.dir)
        learn = [r for r in reqs if r["split"] == "learn"]
        held = [r for r in reqs if r["split"] == "held"]
        if not learn or not held:
            raise GateError(f"{s.dir}: {len(learn)} learn and {len(held)} held requests; the gate needs both")
        mask = np.zeros(s.T, dtype=bool)
        for r in learn:
            mask[r["first"]:r["end"]] = True
        replies = [X[r["first"] + r["prompt"]:r["end"]] for r in held]
        G[f["chat"]] = dict(kind="chat", learn=X[mask], replies=[w for w in replies if len(w)])
    return G


def tally(Xt, E):
    """[T, L, K] picks -> [L, E] undecayed counts: the engine's profile of those rows."""
    return np.stack([np.bincount(Xt[:, i].ravel(), minlength=E) for i in range(Xt.shape[1])])


def static_per_token(lists, n_l, E, Xt):
    res = rr.seed_resident(lists, n_l, E)
    return res[np.arange(len(lists))[None, :, None], Xt].mean(axis=(1, 2))


def rule_hit(lists, n_l, E, Xt, d):
    res0 = rr.seed_resident(lists, n_l, E)
    r = rr.replay(Xt, res0, n_l, E, rr.Rule("mid"), sem="flip", spares=1, d=d, link=rr.Budget(None))
    return float(r.per_row.mean())


def cells(f, G, n_l, rng):
    """Every (A, B, arm) cell at one share: hit, prefix, diff and SE in points."""
    E, _, lay = facts(f)
    pfx = [np.arange(E) for _ in lay]
    prof = {g: [rr.rank(c) for c in tally(G[g]["learn"], E)] for g in G}
    out = []
    cache = {}

    def per(lists, B, arm):
        """The arm's samples on B: per request (chat), per token (corpus static), per window and the
        head window (corpus tok)."""
        g = G[B]
        name, N, step, nwin = arm
        if g["kind"] == "chat":
            d = f["d_chat"]
            if N is None:
                return [float(static_per_token(lists, n_l, E, w).mean()) for w in g["replies"]]
            return [rule_hit(lists, n_l, E, w[:N], d) for w in g["replies"]]
        X = g["score"]
        if N is None:
            return static_per_token(lists, n_l, E, X)
        d = f["d_corpus"]
        return [rule_hit(lists, n_l, E, X[j * step:j * step + N], d) for j in range(nwin)]

    for B in G:
        for arm in f["arms"]:
            key = (B, arm[0])
            if key not in cache:
                cache[key] = per(pfx, B, arm)
            p = cache[key]
            for A in G:
                s = per(prof[A], B, arm)
                if G[B]["kind"] == "chat":
                    diffs = np.asarray(s) - np.asarray(p)
                    hit, base = float(np.mean(s)), float(np.mean(p))
                    se = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs)))
                elif arm[1] is None:
                    hit, base = float(np.mean(s)), float(np.mean(p))
                    b = f["block"]
                    n = len(s) // b // 4 * 4
                    blocks = (np.asarray(s) - np.asarray(p))[:n * b].reshape(n // 4, 4 * b).mean(axis=1)
                    boot = rng.choice(blocks, size=(BOOT_N, len(blocks)), replace=True).mean(axis=1)
                    se = float(np.std(boot, ddof=1))
                else:
                    # the arm's own value is its head window; the sweep prices its noise
                    hit, base = s[0], p[0]
                    diffs = np.asarray(s) - np.asarray(p)
                    se = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs)))
                out.append(dict(A=A, B=B, arm=arm[0], hit=100 * hit, prefix=100 * base,
                                diff=100 * (hit - base), se=100 * se))
    return out


def judge(fam, share, c, pins):
    pin = pins.get((fam, c["A"], c["B"], share), {}).get(c["arm"], 0.0)
    c["pin"] = pin
    c["ok"] = -c["diff"] <= pin + 2 * c["se"]
    return c


def gate(fam, data, out, pins=None, json_path=None):
    pins = PINS if pins is None else pins
    f = family(fam)
    E, K, lay = facts(f)
    G = load(f, data)
    rng = np.random.default_rng(NOISE_SEED)
    rows = []
    for label, n_l in shares(f, E, lay):
        cs = [judge(fam, label, c, pins) for c in cells(f, G, n_l, rng)]
        for c in cs:
            c["share"] = label
        rows.extend(cs)
        arms = [a[0] for a in f["arms"]]
        print(f"# {fam} share {label}: n_l {min(n_l)}..{max(n_l)} = {sum(n_l)} of {E * len(lay)} "
              f"({100 * sum(n_l) / (E * len(lay)):.1f} %); diff seed - prefix in points, +-2SE, pin",
              file=out)
        print("| B | A | " + " | ".join(arms) + " |", file=out)
        print("|---|---|" + "---:|" * len(arms), file=out)
        for B in G:
            for A in G:
                r = [c for c in cs if c["A"] == A and c["B"] == B]
                text = []
                for c in r:
                    mark = "" if c["ok"] else " FAIL"
                    pin = f" pin {c['pin']:.1f}" if c["pin"] else ""
                    text.append(f"{c['diff']:+.1f} +-{2 * c['se']:.1f}{pin}{mark}")
                print(f"| {B} | {A} | " + " | ".join(text) + " |", file=out)
    bad = [c for c in rows if not c["ok"]]
    if json_path:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(dict(family=fam, rows=rows, boot_n=BOOT_N, noise_seed=NOISE_SEED), fh, indent=1)
    for c in bad:
        print(f"FAIL {fam} share {c['share']} {c['A']} -> {c['B']} {c['arm']}: diff {c['diff']:+.2f} past the "
              f"pinned loss {c['pin']:.1f} + 2SE {2 * c['se']:.2f}", file=out)
    print(f"profile-crossgenre {fam}: {len(rows)} cells, {len(bad)} failing", file=out)
    return 1 if bad else 0


# --- the synthetic pair ---------------------------------------------------------------------------


def smoke_rows(genre, T, E=16, K=2):
    """Deterministic routing of the smoke genres over T tokens on layers 0 and 1: beta draws from ids
    0..5, alpha from ids 8..15 two tokens in three and from beta's ids the third — a mixed history."""
    g = rr.Lcg(7 if genre == "alpha" else 11)
    rows = {0: [], 1: []}
    for t in range(T):
        for l in (0, 1):
            pool = list(range(6)) if genre == "beta" or (genre == "alpha" and t % 3 == 2) else list(range(8, E))
            first = pool[g.below(len(pool))]
            rest = [i for i in pool if i != first]
            rows[l].append((first, rest[g.below(len(rest))]))
    return rows


def cmd_smoke(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for genre in ("alpha", "beta"):
        d = os.path.join(out_dir, genre)
        if os.path.isdir(d):
            for name in os.listdir(d):
                os.remove(os.path.join(d, name))
            os.rmdir(d)
        rr.q38_engine_set(out_dir, genre, smoke_rows(genre, 4096), n_expert=16)
    return 0


def cmd_contexts(set_dir, genres):
    calls, _ = rr.rt.read_calls(set_dir)
    runs = []
    for part in genres.split(","):
        g, _, n = part.partition(":")
        if not g or not n.isdigit():
            raise GateError(f"--genres {genres!r}: G:N[,G:N...]")
        runs.extend([g] * int(n))
    if len(runs) != len(calls):
        raise GateError(f"{set_dir}: {len(calls)} call rows, --genres names {len(runs)}")
    path = os.path.join(set_dir, "contexts.tsv")
    if os.path.exists(path):
        raise GateError(f"{path} is there already")
    seen = {}
    lines = ["# route-trace contexts — profile-crossgenre.py contexts: one call a prompt, each genre's calls "
             "learn and held in turn", "# columns\t" + "\t".join(rr.rt.CONTEXT_COLUMNS)]
    for i, (c, g) in enumerate(zip(calls, runs)):
        j = seen.get(g, 0)
        seen[g] = j + 1
        prompt = c["prompt_call"]
        row = dict(request=i, first=c["first"], end=c["end"], prompt=prompt, prompt_call=prompt,
                   prompt_ids=prompt, cache=0, generated=c["end"] - c["first"] - prompt, stop="length",
                   prompt_id=f"chat-{g}:{j}", genre=g, split="learn" if j % 2 == 0 else "held")
        lines.append("\t".join(str(row[k]) for k in rr.rt.CONTEXT_COLUMNS))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    rr.rt.read_contexts(set_dir)
    return 0


def main(argv, out=None):
    out = out or sys.stdout
    if argv == ["--self-test"]:
        return self_test()
    p = argparse.ArgumentParser(prog="profile-crossgenre.py")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate")
    g.add_argument("family")
    g.add_argument("--data", required=True)
    g.add_argument("--json")
    c = sub.add_parser("contexts")
    c.add_argument("set")
    c.add_argument("--genres", required=True)
    s = sub.add_parser("smoke")
    s.add_argument("--out", required=True)
    a = p.parse_args(argv)
    try:
        if a.cmd == "smoke":
            return cmd_smoke(a.out)
        if a.cmd == "contexts":
            return cmd_contexts(a.set, a.genres)
        return gate(a.family, a.data, out, json_path=a.json)
    except (GateError, rr.ToolError, rr.rc.SetError, rr.rt.TraceError) as e:
        print(f"profile-crossgenre: {e}", file=sys.stderr)
        return 2


def self_test():
    # The committed pair is the generator's output, byte for byte.
    with tempfile.TemporaryDirectory() as root:
        cmd_smoke(root)
        for genre in ("alpha", "beta"):
            for name in sorted(os.listdir(os.path.join(root, genre))):
                with open(os.path.join(root, genre, name), "rb") as a, \
                        open(os.path.join(SMOKE, genre, name), "rb") as b:
                    assert a.read() == b.read(), f"{SMOKE}/{genre}/{name} is not what `smoke` writes"
    # The red path: alpha's mixed profile on beta is below the prefix past its noise, with no pin.
    buf = io.StringIO()
    code = gate("smoke", SMOKE, buf, pins={})
    text = buf.getvalue()
    assert code == 1, text
    fails = [l for l in text.splitlines() if l.startswith("FAIL ")]
    assert fails and all(" alpha -> beta " in l for l in fails), text
    assert "| beta | beta | +" in text and "| alpha | alpha | +" in text, text
    # The pin: the pair's measured loss pinned, the gate holds; a pin short of it does not.
    loss = {}
    for line in fails:
        arm = line.split(" alpha -> beta ")[1].split(":")[0]
        loss[arm] = -float(line.split("diff ")[1].split(" ")[0])
    held = gate("smoke", SMOKE, io.StringIO(), pins={("smoke", "alpha", "beta", "25"): loss})
    assert held == 0, loss
    short = {k: v - 1.0 for k, v in loss.items()}
    assert gate("smoke", SMOKE, io.StringIO(), pins={("smoke", "alpha", "beta", "25"): short}) == 1
    # A family of one genre is refused by name; the v41 pins name the measured worst pair alone.
    FAMILIES["one"] = dict(FAMILIES["smoke"], corpora=("alpha",))
    try:
        err = io.StringIO()
        with redirect_stderr(err), redirect_stdout(io.StringIO()):
            code = main(["gate", "one", "--data", SMOKE])
    finally:
        del FAMILIES["one"]
    assert code == 2 and "family one has 1 genre (alpha)" in err.getvalue(), err.getvalue()
    # contexts: a set of five calls, two genres in turn learn and held; a count that does not sum is refused
    with tempfile.TemporaryDirectory() as root:
        rows = {l: [(t % 16, (t + 1) % 16) for t in range(40)] for l in (0, 1)}
        d = rr.q38_engine_set(root, "calls", rows, n_expert=16)
        with open(os.path.join(d, "MANIFEST.tsv"), encoding="utf-8") as fh:
            text = fh.read()
        five = "\n".join(f"call\t{i}\t{8 * i}\t3\t{8 * i + 8}\t0" for i in range(5))
        with open(os.path.join(d, "MANIFEST.tsv"), "w", encoding="utf-8") as fh:
            fh.write(text.replace("call\t0\t0\t8\t40\t0", five))
        err = io.StringIO()
        with redirect_stderr(err):
            assert main(["contexts", d, "--genres", "code:2,ko:2"]) == 2 and "5 call rows" in err.getvalue()
        assert main(["contexts", d, "--genres", "code:2,ko:3"]) == 0
        ctx, _ = rr.rt.read_contexts(d)
        assert [(r["genre"], r["split"], r["generated"]) for r in ctx] == [
            ("code", "learn", 5), ("code", "held", 5), ("ko", "learn", 5), ("ko", "held", 5), ("ko", "learn", 5)], ctx
    assert set(PINS) == {("v41", "d2-chat", "code", "18"), ("v41", "d2-chat", "code", "22.3")}
    print("profile-crossgenre self-test: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
