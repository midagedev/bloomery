#!/usr/bin/env python3
"""What a free draft would accept on a token stream, offline, if the target were the text itself.

    tools/ref/draft-accept.py <ids> [<ids>...] [--tokens N] [--chunk C]
                              [--corpus <ids> [--corpus-skip K] [--corpus-tokens M] [--corpus-doc D]]
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
  sam             exllamav3's SAM draft (util/build_sam.md): the longest suffix of the context that
                  occurs with a follower, in the live history (the context's own earlier tokens, a
                  match of at most HIST_MAX tokens) or in a fixed corpus (a suffix automaton over
                  --corpus, unbounded); on equal lengths the history wins, and a corpus match whose
                  follower lies past its document's end supplies nothing (the next shorter suffix is
                  tried). Only with --corpus. `sam-history` is the same rule with no corpus: what the
                  corpus adds is sam - sam-history.
The corpus is held out: it is read from --corpus starting at token K (--corpus-skip, so a stream's
tail can serve its own head), at most M tokens (--corpus-tokens, default 1,000,000), cut into
documents of D tokens (--corpus-doc; default: the whole read is one document). A corpus that holds
more than OVERLAP_MAX of a scored stream's 16-grams is refused by name: it would score the text
against itself (prose-all's head is the prose set, draft-accept-e5-report.md section 4).
k=2 (lookup-recent and sam): where position i was proposed from a match whose follower sits at j, the
second draft token is t[j + 1] (the drafted token itself when j + 1 = i; for a corpus match the corpus
token after the follower, none past the document's end). The table counts it only where position i was
accepted and i + 1 lies in the context: a2 = accepted / those positions.

The target is the corpus text, human-written: the numbers are how predictable the domain is to the
draft, not the engine's acceptance, whose target is the model's greedy continuation.
"""
import os
import sys
import tempfile

NS = (3, 2, 1)
SOURCES = ("history", "corpus")
HIST_MAX = 64          # the longest history match the sam arm looks for, in tokens
OVERLAP_GRAM = 16
OVERLAP_MAX = 0.01     # a corpus holding more of a scored stream's 16-grams than this is not held out
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
    __slots__ = ("positions", "proposed", "accepted", "by_n", "by_src", "k2_pos", "k2_prop", "k2_acc")

    def __init__(self):
        self.positions = self.proposed = self.accepted = 0
        self.by_n = {n: [0, 0] for n in NS}  # n -> [proposed, accepted]
        self.by_src = {src: [0, 0] for src in SOURCES}  # sam: where the match was found
        self.k2_pos = self.k2_prop = self.k2_acc = 0

    def add(self, other):
        self.positions += other.positions
        self.proposed += other.proposed
        self.accepted += other.accepted
        for n in NS:
            self.by_n[n][0] += other.by_n[n][0]
            self.by_n[n][1] += other.by_n[n][1]
        for src in SOURCES:
            self.by_src[src][0] += other.by_src[src][0]
            self.by_src[src][1] += other.by_src[src][1]
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


class Corpus:
    """A suffix automaton over the corpus documents, each followed by its own separator (a negative id,
    which no token equals), so no match crosses a document and a follower past a document's end is a
    separator. first[v] is the end position of the first occurrence of state v's strings."""

    def __init__(self, docs):
        self.text = []
        for d, doc in enumerate(docs):
            self.text.extend(doc)
            self.text.append(-1 - d)
        nxt, link, length, first = [{}], [-1], [0], [-1]
        last = 0
        for pos, c in enumerate(self.text):
            cur = len(nxt)
            nxt.append({})
            link.append(-1)
            length.append(length[last] + 1)
            first.append(pos)
            p = last
            while p != -1 and c not in nxt[p]:
                nxt[p][c] = cur
                p = link[p]
            if p == -1:
                link[cur] = 0
            else:
                q = nxt[p][c]
                if length[p] + 1 == length[q]:
                    link[cur] = q
                else:
                    clone = len(nxt)
                    nxt.append(dict(nxt[q]))
                    link.append(link[q])
                    length.append(length[p] + 1)
                    first.append(first[q])
                    while p != -1 and nxt[p].get(c) == q:
                        nxt[p][c] = clone
                        p = link[p]
                    link[q] = link[cur] = clone
            last = cur
        self.nxt, self.link, self.length, self.first = nxt, link, length, first

    def step(self, v, n, c):
        """The matching state after appending token c to a match (v, n): the longest suffix that occurs."""
        while v and c not in self.nxt[v]:
            v = self.link[v]
            n = self.length[v]
        if c in self.nxt[v]:
            return self.nxt[v][c], n + 1
        return 0, 0

    def follower(self, v, n):
        """(match length, corpus position of the follower) for the longest suffix of the match (v, n)
        whose first occurrence has a follower in its document; (0, None) when none has."""
        while v:
            j = self.first[v] + 1
            if self.text[j] >= 0:
                return n, j
            v = self.link[v]
            n = self.length[v]
        return 0, None

    def after(self, j):
        """The corpus token after position j, or None past its document."""
        return self.text[j + 1] if self.text[j + 1] >= 0 else None


class History:
    """The live history's n-grams up to HIST_MAX, by a rolling hash of each suffix: ngram -> the index of
    its most recent follower. Updated with position i's n-grams only after position i is scored."""
    MOD = (1 << 61) - 1
    BASE = 1_000_003

    def __init__(self):
        self.idx = {}
        self.pw = [1]
        for _ in range(HIST_MAX):
            self.pw.append(self.pw[-1] * self.BASE % self.MOD)
        self.pre = [0]   # prefix hashes of the context

    def push(self, tok):
        self.pre.append((self.pre[-1] * self.BASE + tok + 1) % self.MOD)

    def h(self, a, b):
        """Hash of t[a:b] (0 <= a <= b <= pushed)."""
        return (self.pre[b] - self.pre[a] * self.pw[b - a]) % self.MOD

    def longest(self, i):
        """(length, follower index) of the longest suffix of t[:i] seen earlier with a follower."""
        best = (0, None)
        for n in range(1, min(HIST_MAX, i) + 1):
            j = self.idx.get((n, self.h(i - n, i)))
            if j is None:
                break
            best = (n, j)
        return best

    def learn(self, i):
        """Record the n-grams ending at i - 1 with their follower at i."""
        for n in range(1, min(HIST_MAX, i) + 1):
            self.idx[(n, self.h(i - n, i))] = i


def score_sam(t, corpus):
    """The sam arm over one context: the longer of the history and corpus matches, the history on a tie;
    corpus None is sam-history. A history match of length n also has every shorter suffix, so the scan
    stops at the first n with no entry."""
    tal = Tally()
    hist = History()
    v = n = 0
    T = len(t)
    hist.push(t[0])
    if corpus is not None:
        v, n = corpus.step(0, 0, t[0])
    for i in range(1, T):
        tal.positions += 1
        hn, hj = hist.longest(i)
        cn, cj = corpus.follower(v, n) if corpus is not None else (0, None)
        d1 = d2 = src = None
        if hn and hn >= min(cn, HIST_MAX):
            d1, src = t[hj], "history"
            d2 = t[hj + 1] if hj + 1 < i else d1
        elif cn:
            d1, src = corpus.text[cj], "corpus"
            d2 = corpus.after(cj)
        if d1 is not None:
            tal.proposed += 1
            tal.by_src[src][0] += 1
            if d1 == t[i]:
                tal.accepted += 1
                tal.by_src[src][1] += 1
                if i + 1 < T:
                    tal.k2_pos += 1
                    if d2 is not None:
                        tal.k2_prop += 1
                        if d2 == t[i + 1]:
                            tal.k2_acc += 1
        hist.learn(i)
        hist.push(t[i])
        if corpus is not None:
            v, n = corpus.step(v, n, t[i])
    return tal


def corpus_docs(t, doc):
    return [t[s:s + doc] for s in range(0, len(t), doc)] if doc else [t]


def overlap(stream, corpus, g=OVERLAP_GRAM):
    """The share of the stream's g-grams (distinct) that the corpus holds."""
    grams = {tuple(stream[i:i + g]) for i in range(len(stream) - g + 1)}
    if not grams:
        return 0.0
    seen = set()
    for doc in corpus_docs_of(corpus):
        for i in range(len(doc) - g + 1):
            k = tuple(doc[i:i + g])
            if k in grams:
                seen.add(k)
    return len(seen) / len(grams)


def corpus_docs_of(corpus):
    docs, cur = [], []
    for x in corpus.text:
        if x < 0:
            docs.append(cur)
            cur = []
        else:
            cur.append(x)
    return docs


DRAFTS = (
    ("lookup-recent", score_recent),
    ("lookup-frequent", lambda t: score_frequent(t, NS)),
    ("markov1", lambda t: score_frequent(t, (1,))),
)


def contexts(t, chunk):
    if not chunk:
        return [t]
    return [t[s:s + chunk] for s in range(0, len(t), chunk)]


def run(stream, chunk, corpus=None):
    """{draft: Tally} over every context of the stream; the sam arms with a corpus only (a Corpus)."""
    out = {}
    arms = list(DRAFTS)
    if corpus is not None:
        arms += [("sam-history", lambda t: score_sam(t, None)), ("sam", lambda t: score_sam(t, corpus))]
    for name, fn in arms:
        tot = Tally()
        for ctx in contexts(stream, chunk):
            tot.add(fn(ctx))
        out[name] = tot
    return out


def arm_names(results):
    return [name for name in results[0][1]] if results else [name for name, _ in DRAFTS]


def frac(a, b):
    return f"{a / b:.3f}" if b else "-"


def print_tables(results, n_tokens, chunk, corpus_line=None):
    ctx = f"chunks of {chunk} tokens, index reset at each boundary" if chunk else "one document, no resets"
    print(f"First {n_tokens} tokens of each stream; {ctx}. positions = tokens with >= 1 token of "
          f"prefix in their context.\n")
    if corpus_line:
        print(corpus_line + "\n")
    for name in arm_names(results):
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
    if "sam" in arm_names(results):
        print(f"### sam, by where the match was found — acc/positions is that source's share; {CAVEAT}\n")
        print("| corpus | " + " | ".join(f"{s} prop | {s} acc/prop | {s} acc/pos" for s in SOURCES) + " |")
        print("|---|" + "---:|" * (3 * len(SOURCES)))
        for corpus, res in results:
            r = res["sam"]
            cells = []
            for src in SOURCES:
                p, a = r.by_src[src]
                cells += [str(p), frac(a, p), frac(a, r.positions)]
            print(f"| {corpus} | " + " | ".join(cells) + " |")
        print()
    for name in [n for n in ("lookup-recent", "sam-history", "sam") if n in arm_names(results)]:
        print(f"### {name}, k=2 — position 2 given position 1 accepted; {CAVEAT}\n")
        print("| corpus | pos-1 accepted, pos 2 in context | pos-2 accepted | a2 = pos-2 accepted / those |"
              " expected tokens per step 1 + a1 + a1*a2 (a1 = acc/positions) |")
        print("|---|---:|---:|---:|---:|")
        for corpus, res in results:
            r = res[name]
            a1 = r.accepted / r.positions if r.positions else 0.0
            a2 = r.k2_acc / r.k2_pos if r.k2_pos else 0.0
            print(f"| {corpus} | {r.k2_pos} | {r.k2_acc} | {frac(r.k2_acc, r.k2_pos)} | {1 + a1 + a1 * a2:.3f} |")
        print()


def read_span(path, skip, n):
    """Tokens [skip, skip + n) of an ids file (fewer if the file ends first; none is refused)."""
    out = []
    with open(path, encoding="ascii") as f:
        for lineno, line in enumerate(f, 1):
            if lineno <= skip:
                continue
            if len(out) == n:
                break
            s = line.strip()
            if not s.isdigit():
                raise InputError(f"{path}:{lineno}: not a token id: {line.rstrip()!r}")
            out.append(int(s))
    if not out:
        raise InputError(f"{path} holds no token past {skip}")
    return out


def load_corpus(path, skip, n, doc):
    t = read_span(path, skip, n)
    c = Corpus(corpus_docs(t, doc))
    docs = len(corpus_docs(t, doc))
    return c, (f"sam corpus: {path} tokens [{skip}, {skip + len(t)}), {docs} document(s) of "
               f"{doc or len(t)} tokens, history matches up to {HIST_MAX} tokens")


def held_out(path, stream, corpus):
    share = overlap(stream, corpus)
    if share > OVERLAP_MAX:
        raise InputError(f"the corpus holds {share:.1%} of {path}'s {OVERLAP_GRAM}-grams (at most {OVERLAP_MAX:.0%}): "
                         f"it is not held out from the scored text — pick another file or a --corpus-skip past it")


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

    # sam: the corpus supplies a pattern the history has never seen (and its k=2 continuation).
    pat = list(range(21, 29))
    t = [1, 2, 3] + pat
    c = Corpus([pat + [99]])
    r, h = score_sam(t, c), score_sam(t, None)
    assert (r.proposed, r.accepted) == (7, 7) and r.by_src["corpus"] == [7, 7], (r.proposed, r.accepted, r.by_src)
    assert (h.proposed, h.accepted) == (0, 0), (h.proposed, h.accepted)
    assert (r.k2_pos, r.k2_prop, r.k2_acc) == (6, 6, 6), (r.k2_pos, r.k2_prop, r.k2_acc)
    # sam: on equal lengths the history wins. At i = 5 both match [5, 6]; the history says 7 (right),
    # the corpus 8. i = 1 (corpus 6, right), 2 (corpus 8, wrong), 4 (tie, both 6, right).
    r = score_sam([5, 6, 7, 5, 6, 7], Corpus([[5, 6, 8]]))
    assert (r.proposed, r.accepted) == (4, 3), (r.proposed, r.accepted)
    assert r.by_src == {"history": [2, 2], "corpus": [2, 1]}, r.by_src
    # sam: a corpus continuation stops at its document's end ([40, 41] then [42, 43]: after 41 nothing).
    r2, r1 = score_sam([40, 41, 42], Corpus([[40, 41], [42, 43]])), score_sam([40, 41, 42], Corpus([[40, 41, 42, 43]]))
    assert (r2.proposed, r2.accepted, r1.proposed, r1.accepted) == (1, 1, 2, 2), (r2.proposed, r1.proposed)
    assert corpus_docs(list(range(5)), 2) == [[0, 1], [2, 3], [4]]
    # the suffix automaton's match is the longest suffix that occurs, against a brute force.
    import random
    rng = random.Random(7)
    for _ in range(200):
        text = [rng.randrange(3) for _ in range(rng.randrange(1, 30))]
        c = Corpus([text])
        q = [rng.randrange(3) for _ in range(rng.randrange(1, 20))]
        v = n = 0
        for i, x in enumerate(q):
            v, n = c.step(v, n, x)
            want = max((L for L in range(0, i + 2)
                        if any(text[j:j + L] == q[i + 1 - L:i + 1] for j in range(len(text) - L + 1))), default=0)
            assert n == want, (text, q, i, n, want)
    # held out: a corpus that holds the scored text is refused by name, a disjoint one is not.
    t = [rng.randrange(1000) for _ in range(400)]
    try:
        held_out("t", t, Corpus([t[100:300]]))
    except InputError:
        pass
    else:
        raise AssertionError("a corpus holding the scored text was accepted")
    held_out("t", t, Corpus([[rng.randrange(1000, 2000) for _ in range(400)]]))
    print("draft-accept: self-test ok")


def main(argv):
    if argv[:1] == ["--self-test"]:
        self_test()
        return 0
    paths, n_tokens, chunk = [], 50000, 0
    cpath, cskip, ctokens, cdoc = None, 0, 1_000_000, 0
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--tokens", "--chunk", "--corpus-tokens", "--corpus-doc"):
            if i + 1 >= len(argv) or not argv[i + 1].isdigit() or int(argv[i + 1]) < 2:
                print(f"draft-accept: {a} takes an integer >= 2", file=sys.stderr)
                return 64
            v = int(argv[i + 1])
            if a == "--tokens":
                n_tokens = v
            elif a == "--chunk":
                chunk = v
            elif a == "--corpus-tokens":
                ctokens = v
            else:
                cdoc = v
            i += 2
        elif a == "--corpus-skip":
            if i + 1 >= len(argv) or not argv[i + 1].isdigit():
                print(f"draft-accept: {a} takes an integer >= 0", file=sys.stderr)
                return 64
            cskip = int(argv[i + 1])
            i += 2
        elif a == "--corpus":
            if i + 1 >= len(argv):
                print(f"draft-accept: {a} takes a path", file=sys.stderr)
                return 64
            cpath = argv[i + 1]
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
    if cpath is None and (cskip or cdoc or ctokens != 1_000_000):
        print("draft-accept: --corpus-skip, --corpus-tokens and --corpus-doc need --corpus", file=sys.stderr)
        return 64
    results = []
    try:
        corpus, line = None, None
        if cpath is not None:
            corpus, line = load_corpus(cpath, cskip, ctokens, cdoc)
        for p in paths:
            stream = read_ids(p, n_tokens)
            if corpus is not None:
                held_out(p, stream, corpus)
            results.append((corpus_name(p), run(stream, chunk, corpus)))
    except (InputError, OSError) as e:
        print(f"draft-accept: {e}", file=sys.stderr)
        return 1
    print_tables(results, n_tokens, chunk, line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
