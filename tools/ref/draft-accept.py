#!/usr/bin/env python3
"""What a free draft would accept on a token stream, offline, if the target were the text itself.

    tools/ref/draft-accept.py <ids> [<ids>...] [--tokens N] [--chunk C] [--recall GRAMS:W[:NMIN]]
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
recall-longest  `runtime::Recall` (crates/runtime/src/recall.rs), the multi-token draft: of the tracked n-gram
                  lengths GRAMS, the longest whose last-n tokens occurred earlier (n from the largest down to
                  the smallest, the most recent occurrence), proposes the W tokens that followed that
                  occurrence, the context extended with the proposal itself where the follower run reaches
                  the context's end (a match at distance p repeats with period p). The match is then widened
                  backward over the tokens that still agree, to RECALL_CAP, for its length `ell`. A proposal
                  is made only where ell >= NMIN. `--recall GRAMS:W[:NMIN]` scores one setting after the tables,
                  GRAMS a list (`3,6,12`, what `Recall::new(3)` tracks) or a range (`1-16`), NMIN the least
                  `ell` that proposes (default: the smallest of GRAMS), and prints the pass walk: a pass at
                  position i verifies the proposal and keeps 1 + r tokens, r the leading drafts equal to the
                  stream; with no proposal it keeps 1. Its self-test cases are the ones `recall.rs` pins.
The corpus is held out: it is read from --corpus starting at token K (--corpus-skip, so a stream's
tail can serve its own head), at most M tokens (--corpus-tokens, default 1,000,000), cut into
documents of D tokens (--corpus-doc; default: the whole read is one document). A corpus that holds
more than OVERLAP_MAX of a scored stream's 16-grams is refused by name: it would score the text
against itself (prose-all's head is the prose set, https://github.com/midagedev/rig-log/blob/main/docs/bloomery/research/draft-accept-e5-report.md section 4).
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
RECALL_CAP = 32        # the longest match length recall-longest reports, in tokens
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


class Longest:
    """score_longest's result: one entry per scored position (the token being proposed for), in order.
    ell[i] is the match length of the proposal at i (0: no match at any tracked n), r[i] the leading
    drafts equal to the stream, m[i] the drafts the stream could check (the proposal's width, cut where
    the stream ends before the bonus token)."""
    __slots__ = ("grams", "w", "fill", "ell", "r", "m")

    def __init__(self, grams, w, fill):
        self.grams, self.w, self.fill = grams, w, fill
        self.ell, self.r, self.m = [], [], []

    def tally(self, nmin, w=None):
        """(positions, proposed, accepted drafts, histogram of r over the proposed) with the proposal
        made only where ell >= nmin and cut to its first w tokens (w <= the scored width)."""
        w = self.w if w is None else w
        if w > self.w:
            raise InputError(f"a proposal of {w} tokens from a scan of width {self.w}")
        hist = [0] * (w + 1)
        proposed = accepted = 0
        for ell, r in zip(self.ell, self.r):
            if ell >= nmin and ell > 0:
                proposed += 1
                accepted += min(r, w)
                hist[min(r, w)] += 1
        return len(self.ell), proposed, accepted, hist

    def walk(self, nmin, w=None):
        """The pass walk over the scored span: (passes, drafted passes, tokens, accepted drafts). A pass at
        position i with a proposal keeps 1 + r tokens (r cut to w), one without keeps 1."""
        w = self.w if w is None else w
        if w > self.w:
            raise InputError(f"a proposal of {w} tokens from a scan of width {self.w}")
        i, n = 0, len(self.ell)
        passes = drafted = tokens = hits = 0
        while i < n:
            passes += 1
            if self.ell[i] >= nmin and self.ell[i] > 0:
                r = min(self.r[i], w)
                drafted += 1
                hits += r
            else:
                r = 0
            tokens += 1 + r
            i += 1 + r
        return passes, drafted, tokens, hits


def score_longest(t, grams, w, scored_from=1, fill=True):
    """recall-longest over one context: the proposal at every position i >= max(scored_from, 1) from the
    context t[:i]. The index of each tracked n-gram length holds the follower of its most recent
    occurrence, updated with the n-gram ending at i - 1 only after position i is proposed, so a proposal
    never sees the token it is proposing. With fill the proposal is always w tokens: where the follower
    run reaches the context's end the proposal repeats itself with the match's period; without it the
    proposal stops at the end of the context (the match's distance)."""
    if w < 1:
        raise InputError("a recall proposal of no token")
    if not grams or any(n < 1 for n in grams) or len(set(grams)) != len(grams):
        raise InputError(f"tracked n-gram lengths are distinct and at least 1, not {list(grams)}")
    down = sorted(grams, reverse=True)
    idx = {n: {} for n in down}
    out = Longest(tuple(sorted(grams)), w, fill)
    T = len(t)
    for i in range(1, T):
        j, used = None, 0
        for n in down:
            if i >= n:
                j = idx[n].get(tuple(t[i - n:i]))
                if j is not None:
                    used = n
                    break
        if i >= scored_from:
            if j is None:
                out.ell.append(0)
                out.r.append(0)
                out.m.append(0)
            else:
                ell = used
                while ell < RECALL_CAP and j - ell - 1 >= 0 and t[j - ell - 1] == t[i - ell - 1]:
                    ell += 1
                width = w if fill else min(w, i - j)
                d = []
                for k in range(width):
                    s = j + k
                    d.append(t[s] if s < i else d[s - i])
                m = min(width, T - 1 - i)
                r = 0
                while r < m and d[r] == t[i + r]:
                    r += 1
                out.ell.append(ell)
                out.r.append(r)
                out.m.append(m)
        for n in down:
            if i >= n:
                idx[n][tuple(t[i - n:i])] = i
    return out


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


def parse_recall(spec):
    """(grams, w, nmin) of `GRAMS:W[:NMIN]`: GRAMS is `a,b,c` or `a-b` (every length from a to b)."""
    parts = spec.split(":")
    try:
        if len(parts) not in (2, 3):
            raise ValueError
        g = parts[0]
        grams = list(range(int(g.split("-")[0]), int(g.split("-")[1]) + 1)) if "-" in g else [int(x) for x in g.split(",")]
        w = int(parts[1])
        nmin = int(parts[2]) if len(parts) == 3 else min(grams)
    except (ValueError, IndexError):
        raise InputError(f"--recall takes GRAMS:W[:NMIN] (GRAMS `3,6,12` or `1-16`), not {spec!r}") from None
    if w < 1 or not grams or min(grams) < 1 or nmin < 1:
        raise InputError(f"--recall: tracked lengths, W and NMIN are at least 1, not {spec!r}")
    return grams, w, nmin


def recall_row(stream, chunk, grams, w, nmin):
    """The recall-longest numbers over every context of the stream: positions, proposed, accepted
    drafts, the histogram of r over the proposed, and the pass walk (passes, drafted, tokens)."""
    pos = prop = acc = passes = drafted = tokens = 0
    hist = [0] * (w + 1)
    for ctx in contexts(stream, chunk):
        if len(ctx) < 2:
            continue
        res = score_longest(ctx, grams, w)
        p, q, a, h = res.tally(nmin)
        pa, dr, tk, _ = res.walk(nmin)
        pos, prop, acc = pos + p, prop + q, acc + a
        passes, drafted, tokens = passes + pa, drafted + dr, tokens + tk
        hist = [x + y for x, y in zip(hist, h)]
    return pos, prop, acc, hist, passes, drafted, tokens


def print_recall(results, grams, w, nmin, chunk):
    print(f"### recall-longest GRAMS={','.join(map(str, grams))} W={w} NMIN={nmin} — {CAVEAT}\n")
    print("| corpus | positions | proposed | proposed/positions | accepted drafts/proposed | r histogram "
          "(r = 0..W over the proposed) | passes | drafted/passes | tokens/pass |")
    print("|---|---:|---:|---:|---:|---|---:|---:|---:|")
    for corpus, row in results:
        pos, prop, acc, hist, passes, drafted, tokens = row
        print(f"| {corpus} | {pos} | {prop} | {frac(prop, pos)} | {frac(acc, prop)} | "
              f"{' / '.join(map(str, hist))} | {passes} | {frac(drafted, passes)} | {frac(tokens, passes)} |")
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


def lcg_stream(n):
    """The stream `crates/runtime/src/lookup.rs` and `recall.rs` score in their tests: a u32 LCG over eleven ids
    with a five-token phrase spliced in."""
    s, out = 1, []
    while len(out) < n:
        s = (s * 1_664_525 + 1_013_904_223) & 0xFFFFFFFF
        x = s >> 16
        if x % 7 == 0:
            out += [40, 41, 42, 43, 44]
        else:
            out.append(x % 11)
    return out[:n]


def recall_self_test():
    # A repeated 7-token pattern: position 7 has no proposal, every later one is right until the stream
    # ends: 62 proposals, 58 of four right, then 3, 2, 1, 0 (the stream's last positions cut the check).
    pat = [11, 12, 13, 14, 15, 16, 17]
    r = score_longest(pat * 10, [1, 2, 3], 4)
    assert r.tally(1) == (69, 62, 238, [1, 1, 1, 1, 58]), r.tally(1)
    assert r.walk(1) == (20, 13, 69, 49), r.walk(1)
    # A run of one id: the match's follower is the context's own end, so the proposal repeats itself
    # (the context extended with the proposal); cut at the context's end it would hold one id only.
    r = score_longest([5] * 6, [1], 3)
    assert r.tally(1) == (5, 4, 6, [1, 1, 1, 1]) and r.walk(1) == (2, 1, 5, 3), (r.tally(1), r.walk(1))
    r = score_longest([5] * 6, [1], 3, fill=False)
    assert r.tally(1) == (5, 4, 3, [1, 3, 0, 0]), r.tally(1)
    # The match length is the tracked n-gram widened backward over the ids that still agree: the 3-gram
    # (1, 2, 3) of the stream's end sits after a 7 both times, so the match is 4 long; it proposes only
    # where ell >= NMIN, and the stream's end cuts the check (m).
    t = [7, 1, 2, 3, 4, 7, 1, 2, 3, 4]
    r = score_longest(t, [3], 2, scored_from=9)
    assert (r.ell, r.r, r.m) == ([4], [0], [0]), (r.ell, r.r, r.m)
    t = [7, 1, 2, 3, 4, 7, 1, 2, 3, 4, 7, 1]
    r = score_longest(t, [3], 2, scored_from=10)
    assert (r.ell, r.r, r.m) == ([5, 6], [1, 0], [1, 0]), (r.ell, r.r, r.m)
    assert r.tally(6)[1] == 1 and r.tally(7)[1] == 0 and r.tally(5)[1] == 2
    # A match widens backward to RECALL_CAP ids and no further: a 70-id block repeated for its first 60.
    t = list(range(200, 270)) + list(range(200, 261))
    r = score_longest(t, [3], 2, scored_from=130)
    assert r.ell == [32] and RECALL_CAP == 32, (r.ell, RECALL_CAP)
    # The longest tracked n decides, not the most recent occurrence of a shorter one: the tail (8, 1, 2) was
    # followed by 3, the 2-gram (1, 2) last by 5, and the stream goes on with 3.
    t = [8, 1, 2, 3, 6, 1, 2, 5, 6, 8, 1, 2, 3, 6]
    r = score_longest(t, [2, 3], 1, scored_from=12)
    assert (r.ell, r.r, r.m) == ([3, 4], [1, 0], [1, 0]), (r.ell, r.r, r.m)
    r = score_longest(t, [2], 1, scored_from=12)
    assert (r.ell, r.r, r.m) == ([2, 4], [0, 0], [1, 0]), (r.ell, r.r, r.m)
    # Refusals by name.
    for bad in ([], [0], [2, 2], ):
        try:
            score_longest([1, 2, 3], bad, 2)
        except InputError:
            pass
        else:
            raise AssertionError(f"grams {bad} accepted")
    try:
        score_longest([1, 2, 3], [1], 0)
    except InputError:
        pass
    else:
        raise AssertionError("a proposal of no token was accepted")
    # The same 4000-token stream recall.rs scores (`matches_draft_accept_py`): one setting a case.
    t = lcg_stream(4000)
    for grams, w, want_tally, want_walk in (
        ([3, 6, 12], 4, (3999, 2798, 3394, [1377, 415, 351, 343, 312]), (2624, 1423, 3999, 1375)),
        ([2, 4, 8], 8, (3999, 3850, 4116, [2232, 477, 359, 355, 338, 64, 6, 4, 15]), (2386, 2237, 3999, 1613)),
        ([4, 8, 16], 2, (3999, 1942, 1873, [815, 381, 746]), (3125, 1068, 3999, 874)),
    ):
        r = score_longest(t, grams, w)
        assert r.tally(grams[0]) == want_tally, (grams, w, r.tally(grams[0]))
        assert r.walk(grams[0]) == want_walk, (grams, w, r.walk(grams[0]))
    # the CLI's spec
    assert parse_recall("3,6,12:4") == ([3, 6, 12], 4, 3) and parse_recall("1-4:8:2") == ([1, 2, 3, 4], 8, 2)
    for bad in ("3,6", "x:4", "3:0", "3:4:0", "0:4", "3-:4"):
        try:
            parse_recall(bad)
        except InputError:
            pass
        else:
            raise AssertionError(f"--recall {bad!r} accepted")
    # the row the CLI prints, against the scorer's own numbers; a chunk resets the context
    row = recall_row(lcg_stream(4000), 0, [3, 6, 12], 4, 3)
    assert row == (3999, 2798, 3394, [1377, 415, 351, 343, 312], 2624, 1423, 3999), row
    row = recall_row(pat * 10, 14, [1, 2, 3], 4, 1)
    assert row[0] == 5 * 13 and row[1] == 5 * 6, row


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
    recall_self_test()
    print("draft-accept: self-test ok")


def main(argv):
    if argv[:1] == ["--self-test"]:
        self_test()
        return 0
    paths, n_tokens, chunk = [], 50000, 0
    cpath, cskip, ctokens, cdoc = None, 0, 1_000_000, 0
    recall = None
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
        elif a == "--recall":
            if i + 1 >= len(argv):
                print(f"draft-accept: {a} takes GRAMS:W[:NMIN]", file=sys.stderr)
                return 64
            try:
                recall = parse_recall(argv[i + 1])
            except InputError as e:
                print(f"draft-accept: {e}", file=sys.stderr)
                return 64
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
    results, recalls = [], []
    try:
        corpus, line = None, None
        if cpath is not None:
            corpus, line = load_corpus(cpath, cskip, ctokens, cdoc)
        for p in paths:
            stream = read_ids(p, n_tokens)
            if corpus is not None:
                held_out(p, stream, corpus)
            results.append((corpus_name(p), run(stream, chunk, corpus)))
            if recall is not None:
                recalls.append((corpus_name(p), recall_row(stream, chunk, *recall)))
    except (InputError, OSError) as e:
        print(f"draft-accept: {e}", file=sys.stderr)
        return 1
    print_tables(results, n_tokens, chunk, line)
    if recall is not None:
        print_recall(recalls, *recall, chunk)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
