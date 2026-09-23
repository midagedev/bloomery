#!/usr/bin/env python3
"""What a free draft would accept on a token stream, offline, if the target were the text itself.

    tools/ref/draft-accept.py <ids> [<ids>...] [--tokens N] [--chunk C]
    tools/ref/draft-accept.py --self-test

<ids> is a token stream, one decimal token id per line ($BLOOMERY_DATA/engram/corpus-<name>.ids).
The first N tokens are read (default 50000); a file with fewer is refused, and so is a line that is
not a non-negative integer. With --chunk C the stream is cut into independent contexts of C tokens
and every draft's index is reset at each boundary; without it the N tokens are one document.

Every draft proposes one token for position i from the prefix of its context, [start, i). Positions
are every i with at least one token of prefix: N minus the number of contexts. The index is updated
with the n-grams ending at i - 1 and their follower t[i] only after position i is scored, so a
draft never sees the token it is proposing.

Drafts:
  lookup-recent   llama.cpp `lookup` shape: for n = 3, 2, 1, the most recent earlier occurrence of
                  the last n tokens; propose the token that followed it. No match at any n: no
                  proposal.
  lookup-frequent the same chain, but the follower seen most often after that n-gram (ties go to
                  the one seen most recently).
  markov1         argmax_t count(prev, t) over the prefix: order-1 Markov, learned causally, ties to
                  the most recent. The shape of a bigram draft head, as an ideal upper bound. It is
                  lookup-frequent with the chain cut to n = 1.
k=2 (lookup-recent only): where position i was proposed from a match whose follower sits at j, the
second draft token is t[j + 1] (the drafted token itself when j + 1 = i). The table counts it only
where position i was accepted and i + 1 lies in the context: a2 = accepted / those positions.

The target is the corpus text, human-written: the numbers are how predictable the domain is to the
draft, not the engine's acceptance, whose target is the model's greedy continuation.
"""
import os
import sys
import tempfile

NS = (3, 2, 1)
CAVEAT = ("target = the corpus text itself (human text), so this is the domain's predictability by "
          "the draft, not the engine's acceptance (target = the model's greedy continuation)")


class InputError(Exception):
    pass


def read_ids(path, n_tokens):
    out = []
    with open(path, encoding="ascii") as f:
        for lineno, line in enumerate(f, 1):
            if len(out) == n_tokens:
                break
            s = line.strip()
            if not s.isdigit():
                raise InputError(f"{path}:{lineno}: not a token id: {line.rstrip()!r}")
            out.append(int(s))
    if len(out) < n_tokens:
        raise InputError(f"{path} holds {len(out)} tokens, fewer than the {n_tokens} asked for")
    return out


class Tally:
    __slots__ = ("positions", "proposed", "accepted", "by_n", "k2_pos", "k2_prop", "k2_acc")

    def __init__(self):
        self.positions = self.proposed = self.accepted = 0
        self.by_n = {n: [0, 0] for n in NS}  # n -> [proposed, accepted]
        self.k2_pos = self.k2_prop = self.k2_acc = 0

    def add(self, other):
        self.positions += other.positions
        self.proposed += other.proposed
        self.accepted += other.accepted
        for n in NS:
            self.by_n[n][0] += other.by_n[n][0]
            self.by_n[n][1] += other.by_n[n][1]
        self.k2_pos += other.k2_pos
        self.k2_prop += other.k2_prop
        self.k2_acc += other.k2_acc


def score_recent(t):
    """lookup-recent over one context, with the k=2 continuation."""
    tal = Tally()
    idx = {n: {} for n in NS}  # n -> {ngram: index of its most recent follower}
    T = len(t)
    for i in range(1, T):
        tal.positions += 1
        j = None
        used = 0
        for n in NS:
            if i >= n:
                j = idx[n].get(tuple(t[i - n:i]))
                if j is not None:
                    used = n
                    break
        if j is not None:
            d1 = t[j]
            tal.proposed += 1
            tal.by_n[used][0] += 1
            if d1 == t[i]:
                tal.accepted += 1
                tal.by_n[used][1] += 1
                if i + 1 < T:
                    tal.k2_pos += 1
                    d2 = t[j + 1] if j + 1 < i else d1
                    tal.k2_prop += 1
                    if d2 == t[i + 1]:
                        tal.k2_acc += 1
        for n in NS:
            if i >= n:
                idx[n][tuple(t[i - n:i])] = i
    return tal


def score_frequent(t, ns):
    """Most-frequent follower along the chain ns; markov1 is ns = (1,)."""
    tal = Tally()
    # n -> {ngram: [counts {token: count}, best token, best count]}
    idx = {n: {} for n in ns}
    for i in range(1, len(t)):
        tal.positions += 1
        d1 = None
        used = 0
        for n in ns:
            if i >= n:
                e = idx[n].get(tuple(t[i - n:i]))
                if e is not None:
                    d1 = e[1]
                    used = n
                    break
        if d1 is not None:
            tal.proposed += 1
            tal.by_n[used][0] += 1
            if d1 == t[i]:
                tal.accepted += 1
                tal.by_n[used][1] += 1
        y = t[i]
        for n in ns:
            if i >= n:
                key = tuple(t[i - n:i])
                e = idx[n].get(key)
                if e is None:
                    idx[n][key] = [{y: 1}, y, 1]
                else:
                    c = e[0].get(y, 0) + 1
                    e[0][y] = c
                    if c >= e[2]:  # ties go to the most recent follower
                        e[1] = y
                        e[2] = c
    return tal


DRAFTS = (
    ("lookup-recent", score_recent),
    ("lookup-frequent", lambda t: score_frequent(t, NS)),
    ("markov1", lambda t: score_frequent(t, (1,))),
)


def contexts(t, chunk):
    if not chunk:
        return [t]
    return [t[s:s + chunk] for s in range(0, len(t), chunk)]


def run(stream, chunk):
    """{draft: Tally} over every context of the stream."""
    out = {}
    for name, fn in DRAFTS:
        tot = Tally()
        for ctx in contexts(stream, chunk):
            tot.add(fn(ctx))
        out[name] = tot
    return out


def frac(a, b):
    return f"{a / b:.3f}" if b else "-"


def print_tables(results, n_tokens, chunk):
    ctx = f"chunks of {chunk} tokens, index reset at each boundary" if chunk else "one document, no resets"
    print(f"First {n_tokens} tokens of each stream; {ctx}. positions = tokens with >= 1 token of "
          f"prefix in their context.\n")
    for name, _ in DRAFTS:
        print(f"### {name} — {CAVEAT}\n")
        print("| corpus | positions | proposed | accepted | acc/proposed | acc/positions |")
        print("|---|---:|---:|---:|---:|---:|")
        for corpus, res in results:
            r = res[name]
            print(f"| {corpus} | {r.positions} | {r.proposed} | {r.accepted} | "
                  f"{frac(r.accepted, r.proposed)} | {frac(r.accepted, r.positions)} |")
        print()
    for name in ("lookup-recent", "lookup-frequent"):
        print(f"### {name}, by the n that matched — acc/positions is that n's share of the total; {CAVEAT}\n")
        print("| corpus | " + " | ".join(f"n={n} prop | n={n} acc/prop | n={n} acc/pos" for n in NS) + " |")
        print("|---|" + "---:|" * (3 * len(NS)))
        for corpus, res in results:
            r = res[name]
            cells = []
            for n in NS:
                p, a = r.by_n[n]
                cells += [str(p), frac(a, p), frac(a, r.positions)]
            print(f"| {corpus} | " + " | ".join(cells) + " |")
        print()
    print(f"### lookup-recent, k=2 — position 2 given position 1 accepted; {CAVEAT}\n")
    print("| corpus | pos-1 accepted, pos 2 in context | pos-2 accepted | a2 = pos-2 accepted / those |"
          " expected tokens per step 1 + a1 + a1*a2 (a1 = acc/positions) |")
    print("|---|---:|---:|---:|---:|")
    for corpus, res in results:
        r = res["lookup-recent"]
        a1 = r.accepted / r.positions if r.positions else 0.0
        a2 = r.k2_acc / r.k2_pos if r.k2_pos else 0.0
        print(f"| {corpus} | {r.k2_pos} | {r.k2_acc} | {frac(r.k2_acc, r.k2_pos)} | {1 + a1 + a1 * a2:.3f} |")
    print()


def corpus_name(path):
    base = os.path.basename(path)
    if base.startswith("corpus-") and base.endswith(".ids"):
        return base[len("corpus-"):-len(".ids")]
    return base


def self_test():
    # A repeated 7-token pattern of distinct ids, 10 repeats. Position 7 has no proposal (p6 has
    # no follower yet); 8 proposes by n=1, 9 by n=2, 10 on by n=3, and every one is right.
    pat = [11, 12, 13, 14, 15, 16, 17]
    t = pat * 10
    N = len(t)
    for name, fn in DRAFTS:
        r = fn(t)
        assert r.positions == N - 1, (name, r.positions)
        assert r.proposed == N - 8 and r.accepted == N - 8, (name, r.proposed, r.accepted)
    r = score_recent(t)
    assert r.by_n[1] == [1, 1] and r.by_n[2] == [1, 1] and r.by_n[3] == [N - 10, N - 10], r.by_n
    assert r.k2_pos == N - 9 and r.k2_acc == N - 9, (r.k2_pos, r.k2_acc)

    # recent vs frequent: after "1" the followers go 2, 2, 3; at the final "1" recent says 3,
    # frequent says 2, and the next token is 2. Distinct separators keep n=2, 3 from matching there.
    t = [1, 2, 90, 1, 2, 91, 1, 3, 92, 1, 2]
    mk, rc = score_frequent(t, (1,)), score_recent(t)
    # markov1: 4 (1 -> 2, ok), 5 (2 -> 90, miss), 7 (1 -> 2, miss), 10 (1: 2 twice, 3 once -> 2, ok).
    assert (mk.proposed, mk.accepted) == (4, 2), (mk.proposed, mk.accepted)
    # recent: 4 (n=1 -> 2, ok), 5 (n=2 (1,2) -> 90, miss), 7 (n=1 -> 2, miss), 10 (n=1 -> 3, miss).
    assert (rc.proposed, rc.accepted) == (4, 1), (rc.proposed, rc.accepted)
    assert rc.by_n[2] == [1, 0] and rc.by_n[1] == [3, 1], rc.by_n

    # k=2 continuation when the match's follower is the drafted token itself: a run of one id.
    t = [5] * 6
    r = score_recent(t)
    # position 1: no index yet; 2..5 proposed and right; k=2 at 2, 3, 4 (5 has no i+1).
    assert (r.proposed, r.accepted, r.k2_pos, r.k2_acc) == (4, 4, 3, 3), (r.proposed, r.k2_pos)

    # Chunk resets: the pattern stream in chunks of 14 = two repeats per chunk; each chunk alone is
    # the first case with N = 14.
    t = pat * 10
    res = run(t, 14)
    for name, _ in DRAFTS:
        assert res[name].positions == 5 * 13 and res[name].accepted == 5 * (14 - 8), (name, res[name].accepted)

    # Input refusals, through the real reader.
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "corpus-t.ids")
        with open(p, "w") as f:
            f.write("1\n2\n3\n")
        assert read_ids(p, 2) == [1, 2]
        for bad, n in (("1\n2\n3\n", 4), ("1\n-2\n3\n", 3), ("1\nx\n", 2)):
            with open(p, "w") as f:
                f.write(bad)
            try:
                read_ids(p, n)
            except InputError:
                pass
            else:
                raise AssertionError(f"accepted {bad!r} for {n} tokens")
        assert corpus_name(p) == "t"
    print("draft-accept: self-test ok")


def main(argv):
    if argv[:1] == ["--self-test"]:
        self_test()
        return 0
    paths, n_tokens, chunk = [], 50000, 0
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--tokens", "--chunk"):
            if i + 1 >= len(argv) or not argv[i + 1].isdigit() or int(argv[i + 1]) < 2:
                print(f"draft-accept: {a} takes an integer >= 2", file=sys.stderr)
                return 64
            if a == "--tokens":
                n_tokens = int(argv[i + 1])
            else:
                chunk = int(argv[i + 1])
            i += 2
        elif a.startswith("-"):
            print(__doc__, file=sys.stderr)
            return 64
        else:
            paths.append(a)
            i += 1
    if not paths:
        print(__doc__, file=sys.stderr)
        return 64
    results = []
    try:
        for p in paths:
            results.append((corpus_name(p), run(read_ids(p, n_tokens), chunk)))
    except (InputError, OSError) as e:
        print(f"draft-accept: {e}", file=sys.stderr)
        return 1
    print_tables(results, n_tokens, chunk)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
