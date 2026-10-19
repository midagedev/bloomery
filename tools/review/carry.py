#!/usr/bin/env python3
"""Shingler across a rebase: an approval carries to a head whose commits are the patches the bot approved.

    tools/review/carry.py decide      --repo OWNER/NAME --head SHA --prev SHA --prev-review FILE --base-ref REF
                                      --ledger VERSION --tools ID --out DIR [--prev-pr N]            (gh)
    tools/review/carry.py supersedes  --repo OWNER/NAME --pr-json FILE                               (gh)
    tools/review/carry.py reviewer-id --root DIR
    tools/review/carry.py compare     --repo OWNER/NAME --head SHA --prev SHA --base-ref REF         (gh)
    tools/review/carry.py --self-test

Why: a review is tied to a head sha, so after a rebase or a replacement pull request (`Supersedes #N`) the bot read
the whole patch again, found new things in code it had approved, and each round took 30 to 45 minutes. What the bot
approved is a series of patches, and a rebase that keeps every patch keeps what was approved.

`decide` compares the new head's commits with the last reviewed head's and prints `<mode>: <why>`, writing DIR/carry.json
(and DIR/prev-review.md and DIR/interdiff.patch for the modes that use them). The modes, by the first rule that holds:

  full       the review of the reviewed head has no marker, was made under other reviewer instructions or tools
             (`tools=`, carry.py reviewer-id) or another slag ledger (`ledger=`), a series cannot be read, holds a
             merge commit, or the interdiff would be larger than the pull request. The bot reads everything.
  carry      the series are the same: for every position the patch-id and the subject are equal, and that review
             approved (`verdict=CLEAN event=APPROVE`). No model runs; `post.py carry` writes the approval.
  interdiff  anything else: the bot reads the commits that changed, were added or dropped, as a diff of the two patches
             (DIR/interdiff.patch), with that review as prev-review.md.

What the series rule means:
  - The patch-id is `git patch-id --verbatim`'s definition: every changed line, the context lines around it and
    whitespace count, blob ids and line numbers do not. Whitespace counts because a justfile, a workflow or Python
    file changes meaning with it.
  - A patch whose context moved (the base changed within its context lines) is never carried, whatever else holds: the
    context is part of what a hunk means, so it is reviewed. The pair is marked `context only` when its + and - lines
    are equal, so the reviewer reads the hunk in place and asks one question: did the base's change alter what it does.
  - A commit added, dropped, split, squashed, reordered or reworded is another series. A new commit is never covered by
    a carried verdict: it is an interdiff.
  - Only an approval carries. A MUST_FIX never does, nor a CLEAN review that only commented (a guarded path, a freeze):
    those go to `full`, so a rebase never turns them into an approval.
  - post.py recomputes the event on the current files and freeze: a release freeze opened since still holds a
    `needs-gpu` change, and a guarded path still needs a person.
  The honest boundary: a carry covers each patch's own lines and the lines around them. What the base changed elsewhere
  is covered by CI on the new head and by the landing gates; carry.json lists the files the base changed in the window
  that the pull request changes too.

`supersedes` reads a pull request's body for a `Supersedes #N` line and prints `ok <head> <N>` when #N is closed, not
merged, by the same author, from the same head repository, and the bot reviewed it (the head is its last review's); else
`none <why>`, and nothing when the body has no such line. `compare` prints how two heads' commits pair up, with no review
check. `reviewer-id` prints the digest of .github/review and tools/review under DIR: what a carried review ran under.
"""

import argparse
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import NamedTuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from post import footer, parse_footer  # noqa: E402

NET_S = 120  # each gh or git call: past it the step reads as unanswered, never as a hang
SHA_RE = re.compile(r"[0-9a-f]{40}")


class Unavailable(Exception):
    """What git or gh was asked could not be answered; the decision falls back to a full review and says why."""


class Commit(NamedTuple):
    sha: str
    subject: str
    patch: str  # the diff alone: the mail header, the diffstat and the signature are cut
    parents: int


def run(cmd, cwd=None, stdin=None):
    try:
        p = subprocess.run(cmd, cwd=cwd, input=stdin, capture_output=True, text=True, timeout=NET_S)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise Unavailable(f"{' '.join(cmd[:3])}: {e}")
    if p.returncode:
        raise Unavailable(f"{' '.join(cmd[:4])} ended {p.returncode}: {p.stderr.strip()[:200]}")
    return p.stdout


def gh_text(path, accept=None, paginate=False, jq=None):
    cmd = ["gh", "api"] + (["--paginate"] if paginate else []) + (["--jq", jq] if jq else [])
    return run(cmd + (["-H", f"Accept: {accept}"] if accept else []) + [path])


# ---------------------------------------------------------------------------------------------------------------------
# patches

def diff_part(mail):
    """The diff of a format-patch mail (git's or the API's patch media type), without the signature trailer."""
    m = re.search(r"^diff --git ", mail, re.M)
    return re.sub(r"\n-- \n[^\n]*\n*\Z", "\n", mail[m.start():]) if m else ""


def patch_id(diff):
    """What makes two patches the same patch: the diff without its blob ids (`index` lines) and its positions (hunk
    headers), hashed. This is `git patch-id --verbatim`'s definition, computed here so no process runs per commit:
    every changed line, every context line and whitespace count, line numbers do not."""
    if not diff.strip():
        return ""
    keep = ["@@" if ln.startswith("@@") else ln for ln in diff.split("\n") if not ln.startswith("index ")]
    return hashlib.sha256("\n".join(keep).encode()).hexdigest()


def edit_lines(diff):
    """The lines of a diff that edit: its +/- lines and file headers, not context, hunk numbers or index lines."""
    return [ln for ln in diff.split("\n") if ln and not ln.startswith((" ", "@@", "index ", "\\"))]


def shown(diff):
    """A diff as the interdiff compares it: hunk numbers and index lines dropped, so a patch that only moved reads as
    itself; the function text after a hunk header stays, it tells the reader where the hunk sits."""
    out = []
    for ln in diff.split("\n"):
        if ln.startswith("index "):
            continue
        m = re.match(r"@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@(.*)", ln)
        out.append("@@" + m.group(1) if m else ln)
    return out


def pr_files(commits):
    names = set()
    for c in commits:
        for a, b in re.findall(r"^diff --git a/(.+?) b/(.+)$", c.patch, re.M):
            names.update((a, b))
    return names


# ---------------------------------------------------------------------------------------------------------------------
# where the commits come from: the GitHub API in the workflow, git in `just pre-review` and the tests

class GhSource:
    """compare/<base>...<sha> lists a head's own commits whether or not it is still a branch tip; commits/<sha> with
    the patch media type gives each one's mail."""

    def __init__(self, repo, base_ref, call=None):
        self.repo, self.base_ref, self.call = repo, base_ref, call or gh_text

    def series(self, sha):
        jq = ('. as $c | .commits[] | {mb: $c.merge_base_commit.sha, total: $c.total_commits, sha: .sha, '
              'parents: (.parents | length), subject: (.commit.message | split("\\n")[0])}')
        text = self.call(f"repos/{self.repo}/compare/{self.base_ref}...{sha}?per_page=100", paginate=True, jq=jq)
        rows = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
        if not rows:
            raise Unavailable(f"{sha[:8]} has no commits beyond {self.base_ref}")
        if len(rows) != rows[0]["total"] or rows[-1]["sha"] != sha:
            raise Unavailable(f"compare listed {len(rows)} of {rows[0]['total']} commits of {sha[:8]}")
        return [Commit(r["sha"], r["subject"],
                       diff_part(self.call(f"repos/{self.repo}/commits/{r['sha']}", accept="application/vnd.github.patch")),
                       r["parents"]) for r in rows], rows[0]["mb"]

    def window_files(self, a, b):
        return self.call(f"repos/{self.repo}/compare/{a}...{b}?per_page=100", paginate=True,
                         jq=".files[].filename").splitlines()


class GitSource:
    def __init__(self, base_ref, cwd=None):
        self.base_ref, self.cwd = base_ref, cwd

    def git(self, *args):
        return run(["git", *args], cwd=self.cwd)

    def series(self, sha):
        mb = self.git("merge-base", self.base_ref, sha).strip()
        log = self.git("log", "--reverse", "-p", "-M", "--binary", "--src-prefix=a/", "--dst-prefix=b/",
                       "--format=%x00%H%x1f%P%x1f%s", f"{mb}..{sha}")
        commits = []
        for rec in log.split("\x00")[1:]:  # a merge commit shows no diff
            head, _, rest = rec.partition("\n")
            h, parents, subject = head.split("\x1f", 2)
            commits.append(Commit(h, subject, diff_part(rest), len(parents.split())))
        if not commits:
            raise Unavailable(f"{sha[:8]} has no commits beyond {self.base_ref}")
        return commits, mb

    def window_files(self, a, b):
        return self.git("diff", "--name-only", a, b).splitlines()


# ---------------------------------------------------------------------------------------------------------------------
# the decision

def pair_up(old, new, old_ids, new_ids):
    """[(kind, old index | None, new index | None)]: one entry per new commit in order, then the dropped old ones.
    A new commit pairs first with an old one of the same patch-id, then of the same subject. Kinds: same, reworded
    (the patch, not the subject), context (the + and - lines, not the context around them), changed, added, dropped."""
    match, used = [None] * len(new), set()
    for equal in (lambda i, j: old_ids[i] == new_ids[j], lambda i, j: old[i].subject == new[j].subject):
        for j in range(len(new)):
            for i in range(len(old)):
                if match[j] is None and i not in used and equal(i, j):
                    match[j] = i
                    used.add(i)
    entries = []
    for j, i in enumerate(match):
        if i is None:
            kind = "added"
        elif old_ids[i] == new_ids[j]:
            kind = "same" if old[i].subject == new[j].subject else "reworded"
        elif edit_lines(old[i].patch) == edit_lines(new[j].patch):
            kind = "context"
        else:
            kind = "changed"
        entries.append((kind, i, j))
    return entries + [("dropped", i, None) for i in range(len(old)) if i not in used]


def same_series(entries):
    """Every commit the same patch and subject at the same position, none added or dropped."""
    return all(k == "same" and i == j for k, i, j in entries)


def reordered(entries):
    seq = [i for _, i, j in entries if i is not None and j is not None]
    return seq != sorted(seq)


def interdiff(entries, old, new):
    """The text the reviewer reads: the pairs that changed as a diff of the two patches (a line starting + or - is a
    patch line the new patch gained or lost), the added commits' patches and the dropped ones'."""
    lines = []
    same = [j + 1 for k, _, j in entries if k == "same"]
    if same:
        lines.append(f"Unchanged (same patch and subject; not shown): new commits {', '.join(map(str, same))} of {len(new)}.")
    if reordered(entries):
        lines.append("The unchanged commits are in another order than at the reviewed head.")
    for kind, i, j in entries:
        if kind == "same":
            continue
        o, n = (old[i] if i is not None else None), (new[j] if j is not None else None)
        if kind == "dropped":
            lines += ["", f"=== dropped: prev {i + 1}/{len(old)} {o.sha[:8]} {o.subject} (not in the new series)"] + o.patch.splitlines()
        elif kind == "added":
            lines += ["", f"=== added: new {j + 1}/{len(new)} {n.sha[:8]} {n.subject}"] + n.patch.splitlines()
        elif kind == "reworded":
            lines += ["", f"=== reworded: new {j + 1}/{len(new)} {n.sha[:8]}: same patch; the subject was `{o.subject}`, now `{n.subject}`"]
        else:
            note = ("context only: the + and - lines are equal and the lines around them moved (the base changed under "
                    "the patch). Read the hunk in PR_TREE: does the base's change alter what it does?"
                    if kind == "context" else "the changed lines differ: an edit, or a rebase conflict resolved by hand.")
            lines += ["", f"=== {kind}: new {j + 1}/{len(new)} {n.sha[:8]} (was prev {i + 1}/{len(old)} {o.sha[:8]}) {n.subject}",
                      note]
            lines += list(difflib.unified_diff(shown(o.patch), shown(n.patch), f"prev {o.sha[:8]}", f"new {n.sha[:8]}",
                                               lineterm="", n=3))
    return "\n".join(lines) + "\n"


def summary(entries):
    count = {}
    for k, _, _ in entries:
        count[k] = count.get(k, 0) + 1
    parts = [f"{count[k]} {k}" for k in ("same", "reworded", "context", "changed", "added", "dropped") if k in count]
    return ", ".join(parts) + ("; reordered" if reordered(entries) else "")


def decide(prev_body, prev_head, head, src, tools, ledger):
    """{"mode": carry | interdiff | full, "why": ..., ...}; `interdiff` holds the text for the interdiff mode."""
    d = {"head": head, "prev_head": prev_head}

    def out(mode, why, **kw):
        d.update(mode=mode, why=why, **kw)
        return d

    if prev_head == head:
        return out("full", "the reviewed head is this head")
    f = parse_footer(prev_body)
    if not f or f.get("head_sha") != prev_head:
        return out("full", f"the review of {prev_head[:8]} has no marker to read")
    for key, now in (("tools", tools), ("ledger", ledger)):
        if f.get(key) != now:
            return out("full", f"{key} is {now}, was {f.get(key) or 'not recorded'} at that review: it was made under "
                               "other reviewer instructions")
    try:
        old, old_base = src.series(prev_head)
        new, new_base = src.series(head)
        if any(c.parents != 1 for c in old + new):
            return out("full", "a series holds a merge commit; commits are compared one to one")
    except Unavailable as e:
        return out("full", f"the patches could not be read: {e}")
    entries = pair_up(old, new, [patch_id(c.patch) for c in old], [patch_id(c.patch) for c in new])
    d["entries"] = [(k, i, j) for k, i, j in entries]
    approved = f.get("verdict") == "CLEAN" and f.get("event") == "APPROVE"
    if same_series(entries):
        if not approved:
            return out("full", f"the review of {prev_head[:8]} was {f.get('verdict')}/{f.get('event') or 'no event'}, "
                               "not an approval: nothing to carry")
        moved, note = None, ""
        if old_base != new_base:
            try:
                moved = sorted(set(src.window_files(old_base, new_base)) & pr_files(new))
            except Unavailable as e:
                moved, note = [], f"The base moved, and the files it changed could not be listed ({e}); CI and the landing gates cover them."
        return out("carry", f"{len(new)} commit{'s' if len(new) != 1 else ''}, every patch-id and subject equal to "
                            f"{prev_head[:8]}'s", commits=len(new), main_files=moved, main_note=note)
    text = interdiff(entries, old, new)
    if len(text.splitlines()) > sum(len(c.patch.splitlines()) for c in new) + 10:
        return out("full", f"the interdiff ({summary(entries)}) is larger than the pull request: read it whole")
    return out("interdiff", summary(entries), interdiff=text)


# ---------------------------------------------------------------------------------------------------------------------
# a replacement pull request, and the reviewer's identity

def bot_reviews(call, repo, n):
    text = call(f"repos/{repo}/pulls/{n}/reviews?per_page=100", paginate=True,
                jq='.[] | select(.user.login == "github-actions[bot]") | {body}')
    return [json.loads(ln).get("body") or "" for ln in text.splitlines() if ln.strip()]


def supersedes(pr, repo, call=gh_text):
    """(head, number, why): the last head the bot reviewed on the pull request `pr` says it supersedes. head is None
    with why "" when the body has no `Supersedes #N` line, and with the reason when it has one and no carry follows."""
    found = sorted(set(re.findall(r"^[ \t]*supersedes[ \t]+#(\d+)\b", pr.get("body") or "", re.M | re.I)))
    if not found:
        return None, None, ""
    if len(found) > 1:
        return None, None, f"the body supersedes several pull requests (#{', #'.join(found)}); a carry follows one"
    n = int(found[0])
    if n == pr["number"]:
        return None, None, "a pull request cannot supersede itself"
    try:
        old = json.loads(call(f"repos/{repo}/pulls/{n}"))
        bodies = bot_reviews(call, repo, n)
    except (Unavailable, ValueError) as e:
        return None, None, f"#{n} could not be read ({e})"

    def repo_of(p):
        return ((p.get("head") or {}).get("repo") or {}).get("full_name")

    if old.get("state") != "closed" or old.get("merged"):
        return None, None, f"#{n} is {'merged' if old.get('merged') else old.get('state')}, not closed unmerged"
    if (old.get("user") or {}).get("login") != (pr.get("user") or {}).get("login"):
        return None, None, f"#{n} is by another author"
    if not repo_of(old) or repo_of(old) != repo_of(pr):
        return None, None, f"#{n} is from another head repository"
    for body in reversed(bodies):
        f = parse_footer(body)
        if f and SHA_RE.fullmatch(f["head_sha"]):
            return f["head_sha"], n, ""
    return None, None, f"the bot has no review of #{n}"


def reviewer_id(root):
    """A digest of what a review runs under that the PR cannot change: .github/review and tools/review."""
    h = hashlib.sha256()
    for sub in (".github/review", "tools/review"):
        top = os.path.join(root, sub)
        if not os.path.isdir(top):
            raise ValueError(f"{top} is not a directory")
        for dirpath, dirs, files in os.walk(top):
            dirs[:] = sorted(x for x in dirs if x != "__pycache__")
            for name in sorted(x for x in files if not x.endswith(".pyc")):
                path = os.path.join(dirpath, name)
                h.update(os.path.relpath(path, root).encode() + b"\0" + open(path, "rb").read() + b"\0")
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------------------------------------------------

def self_test():
    fails, cases = [], []

    def expect(ok, what):
        cases.append(what)
        print(("ok   " if ok else "FAIL ") + what)
        if not ok:
            fails.append(what)

    LEDGER, TOOLS = "0a1b2c3d", "0123456789abcdef"
    diff = ("diff --git a/x.rs b/x.rs\nindex 1111111..2222222 100644\n--- a/x.rs\n+++ b/x.rs\n"
            "@@ -10,3 +10,3 @@ fn old_name() {\n a\n-b\n+c\n d\n")
    moved = diff.replace("1111111..2222222", "3333333..4444444").replace("-10,3 +10,3", "-40,3 +41,3").replace("old_name", "renamed")
    expect(patch_id(diff) == patch_id(moved) and patch_id(diff) != "",
           "the patch-id ignores blob ids, hunk positions and the function text after a hunk header")
    expect(len({patch_id(diff), patch_id(diff.replace(" a\n", " x\n")), patch_id(diff.replace("+c", "+c ")),
                patch_id(diff.replace("b/x.rs\n@@", "b/y.rs\n@@")), patch_id("")}) == 5,
           "the patch-id sees a changed context line, whitespace, a path, and an empty patch")
    with tempfile.TemporaryDirectory() as d:
        env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull}

        def sh(*c, cwd=None):
            return subprocess.run(["git", *c], cwd=cwd or repo, check=True, capture_output=True, text=True, env=env).stdout.strip()

        def wr(path, text):
            with open(os.path.join(repo, path), "w") as fh:
                fh.write(text)

        def edit(path, n, text):
            p = os.path.join(repo, path)
            lines = open(p).read().split("\n")
            lines[n - 1] = text
            wr(path, "\n".join(lines))

        def commit(subject):
            sh("add", "-A")
            sh("commit", "-qm", subject)
            return sh("rev-parse", "HEAD")

        def branch(name, start):
            sh("checkout", "-q", "-B", name, start)

        repo = os.path.join(d, "r")
        os.makedirs(repo)
        sh("init", "-q", "-b", "main")
        sh("config", "user.email", "t@t")
        sh("config", "user.name", "t")
        wr("f.txt", "".join(f"line {i}\n" for i in range(1, 41)))
        wr("justfile", "".join(f"recipe {i}\n" for i in range(1, 31)))
        commit("base")
        # the pull request: three independent commits on main
        branch("topic", "main")
        edit("f.txt", 20, "twenty")
        commit("f: line 20")
        wr("n.txt", "".join(f"new {i}\n" for i in range(1, 31)))
        commit("n: a new file")
        edit("justfile", 10, "recipe ten")
        P = commit("justfile: recipe 10")
        # the base moves away from the pull request's lines, in a file it also changes
        branch("main", "main")
        edit("f.txt", 5, "five")
        commit("main: f line 5")
        edit("justfile", 28, "recipe 28!")
        commit("main: justfile line 28")
        wr("m.txt", "m\n")
        commit("main: m.txt")
        branch("moved", "main")
        src = GitSource("main", repo)

        def rebased(name="rebased", onto="main"):
            branch(name, "topic")
            sh("rebase", "-q", onto)
            return sh("rev-parse", "HEAD")

        def prev(verdict="CLEAN", event="APPROVE", tools=TOOLS, ledger=LEDGER, head=None):
            return (f"**Verdict: {verdict}**: x\n\n"
                    + footer(head or P, verdict, "", "full", ledger, "1", event, tools) + "\n")

        def decision(head, body=None, tools=TOOLS, ledger=LEDGER, source=None, prev_head=None):
            return decide(prev() if body is None else body, prev_head or P, head, source or src, tools, ledger)

        # 1. a pure rebase: every patch-id equal; the base's other changes to the same files are listed
        H = rebased()
        r = decision(H)
        expect(r["mode"] == "carry" and r["commits"] == 3 and r["main_files"] == ["f.txt", "justfile"],
               f"a pure rebase carries: {r['mode']} {r['why']} {r.get('main_files')}")
        # 2. nothing moved: a carry that lists no window
        branch("same-base", "topic")
        sh("commit", "-q", "--amend", "--no-edit", "--date=2020-01-01T00:00:00")
        sh("update-ref", "refs/heads/at-base", sh("merge-base", "main", "topic"))
        r = decide(prev(), P, sh("rev-parse", "HEAD"), GitSource("at-base", repo), TOOLS, LEDGER)
        expect(r["mode"] == "carry" and r["main_files"] is None,
               f"a head on the same base with the same patches carries and lists no window: {r['mode']} {r.get('main_files')}")
        # 3. one patch changed
        branch("amended", "topic")
        sh("reset", "-q", "--hard", "HEAD~2")
        edit("f.txt", 20, "twenty, amended")
        sh("commit", "-qa", "--amend", "--no-edit")
        sh("cherry-pick", "topic~1", "topic")
        r = decision(sh("rev-parse", "HEAD"))
        expect(r["mode"] == "interdiff" and [e[0] for e in r["entries"]] == ["changed", "same", "same"]
               and "twenty, amended" in r["interdiff"] and "+recipe ten" not in r["interdiff"],
               f"one changed patch is an interdiff of that pair only: {r['mode']} {r['why']}")
        # 4. a commit added
        branch("added", "topic")
        wr("extra.txt", "x\n")
        commit("extra: a new commit")
        r = decision(sh("rev-parse", "HEAD"))
        expect(r["mode"] == "interdiff" and r["entries"][-1][0] == "added" and "+x" in r["interdiff"],
               f"a new commit is never covered: {r['mode']} {r['why']}")
        # 5. a commit dropped
        branch("dropped", "topic")
        sh("reset", "-q", "--hard", "HEAD~1")
        r = decision(sh("rev-parse", "HEAD"))
        expect(r["mode"] == "interdiff" and r["entries"][-1][0] == "dropped" and "not in the new series" in r["interdiff"],
               f"a dropped commit is an interdiff: {r['mode']} {r['why']}")
        # 6. a reordered commit
        branch("reordered", "main~3")
        sh("cherry-pick", "topic~1", "topic~2", "topic")
        r = decision(sh("rev-parse", "HEAD"), source=GitSource("main~3", repo))
        expect(r["mode"] == "interdiff" and "reordered" in r["why"] and "another order" in r["interdiff"],
               f"a reordered commit is not a carry: {r['mode']} {r['why']}")
        # 7. a conflict resolved by hand
        branch("conflicting", "main~3")
        edit("f.txt", 20, "twenty on main")
        commit("main-side: f line 20")
        sh("update-ref", "refs/heads/cmain", "HEAD")
        branch("resolved", "topic")
        p = subprocess.run(["git", "rebase", "cmain"], cwd=repo, capture_output=True, text=True, env=env)
        wr("f.txt", "".join(f"line {i}\n" for i in range(1, 41)))
        edit("f.txt", 20, "resolved by hand")
        sh("add", "f.txt")
        subprocess.run(["git", "-c", "core.editor=true", "rebase", "--continue"], cwd=repo, check=True, capture_output=True, env=env)
        r = decide(prev(), P, sh("rev-parse", "HEAD"), GitSource("cmain", repo), TOOLS, LEDGER)
        expect(p.returncode != 0 and r["mode"] == "interdiff" and [e[0] for e in r["entries"]] == ["changed", "same", "same"]
               and "resolved by hand" in r["interdiff"], f"a conflict resolution is a changed pair: {r['mode']} {r['why']}")
        # 8. context only: the base changed a line within a hunk's three context lines
        branch("nearmain", "main~3")
        edit("f.txt", 17, "seventeen")
        commit("main: f line 17")
        branch("near", "topic")
        sh("rebase", "-q", "nearmain")
        r = decide(prev(), P, sh("rev-parse", "HEAD"), GitSource("nearmain", repo), TOOLS, LEDGER)
        expect(r["mode"] == "interdiff" and r["entries"][0][0] == "context" and "context only" in r["interdiff"]
               and "+ seventeen" in r["interdiff"], f"a patch whose context moved is reviewed, not carried: {r['mode']} {r['why']}")
        # 9. whitespace is part of the patch
        branch("space", "topic")
        sh("reset", "-q", "--hard", "HEAD~1")
        edit("n.txt", 1, "new 1  ")
        sh("commit", "-qa", "--amend", "--no-edit")
        sh("cherry-pick", "topic")
        r = decision(sh("rev-parse", "HEAD"))
        expect(r["mode"] == "interdiff" and r["entries"][1][0] == "changed", f"a whitespace change is a changed patch: {r['mode']} {r['why']}")
        # 10. a reworded subject
        branch("reworded", "topic")
        sh("commit", "-q", "--amend", "-m", "justfile: recipe ten")
        r = decision(sh("rev-parse", "HEAD"))
        expect(r["mode"] == "interdiff" and r["entries"][2][0] == "reworded" and "reworded" in r["interdiff"],
               f"a reworded commit is not the same series: {r['mode']} {r['why']}")
        # 11. a merge commit in the series
        branch("merged", "topic")
        branch("side", "main")
        wr("side.txt", "s\n")
        commit("side: a file")
        sh("checkout", "-q", "merged")
        sh("merge", "-q", "--no-ff", "-m", "merge side", "side")
        r = decision(sh("rev-parse", "HEAD"))
        expect(r["mode"] == "full" and "merge commit" in r["why"], f"a merge commit is read whole: {r['mode']} {r['why']}")
        # 12. everything rewritten: the interdiff is larger than the pull request
        branch("squashed", "main")
        sh("merge", "-q", "--squash", "topic")
        sh("commit", "-qm", "all of it at once")
        r = decision(sh("rev-parse", "HEAD"))
        expect(r["mode"] == "full" and "larger than the pull request" in r["why"], f"a squash is a full review: {r['mode']} {r['why']}")
        # 13. the previous verdict decides what carries: a MUST_FIX or a comment never turns into an approval
        for what, body in (("MUST_FIX", prev("MUST_FIX", "REQUEST_CHANGES")), ("a CLEAN that only commented", prev("CLEAN", "COMMENT"))):
            r = decision(H, body)
            expect(r["mode"] == "full" and "not an approval" in r["why"], f"a pure rebase after {what} does not carry: {r['mode']} {r['why']}")
        # 14. the reviewer must be the one that reviewed
        for what, kw in (("tools", {"tools": "ffffffffffffffff"}), ("ledger", {"ledger": "other"})):
            r = decision(H, **kw)
            expect(r["mode"] == "full" and what in r["why"], f"another {what} is a full review: {r['mode']} {r['why']}")
        old_marker = "**Verdict: CLEAN**: x\n<!-- bloomery-review head_sha=" + P + " verdict=CLEAN must_fix=0 warn=0 mode=full -->\n"
        for what, body in (("a review before tools= and event=", old_marker), ("no marker (jq's null)", "null\n"),
                           ("a marker for another head", prev(head="c" * 40))):
            r = decision(H, body)
            expect(r["mode"] == "full", f"{what} is a full review: {r['mode']} {r['why']}")
        r = decision(P)
        expect(r["mode"] == "full" and "is this head" in r["why"], f"an unchanged head needs no decision: {r['why']}")
        r = decision("0" * 40)
        expect(r["mode"] == "full" and "could not be read" in r["why"], f"an unreadable head is a named full review: {r['why']}")

        # 15. through the API: patches arrive as mail (diffstat, signature), a replacement pull request's reviewed head
        # is a head no branch holds any more, a failed call is a named full review
        def fake_call(fail=()):
            def call(path, accept=None, paginate=False, jq=None):
                if any(x in path for x in fail):
                    raise Unavailable(f"gh api {path} ended 1: HTTP 404")
                m = re.match(r"repos/o/r/compare/(\S+?)\.\.\.([0-9a-f]+)\?", path)
                if m and jq and "commits" in jq:
                    base, sha = m.groups()
                    mb = sh("merge-base", base, sha)
                    revs = sh("rev-list", "--reverse", "--parents", f"{mb}..{sha}").splitlines()
                    return "".join(json.dumps({"mb": mb, "total": len(revs), "sha": r.split()[0], "parents": len(r.split()) - 1,
                                               "subject": sh("log", "-1", "--format=%s", r.split()[0])}) + "\n" for r in revs)
                if m:
                    return sh("diff", "--name-only", m.group(1), m.group(2)) + "\n"
                m = re.match(r"repos/o/r/commits/([0-9a-f]+)$", path)
                assert m and accept == "application/vnd.github.patch", path
                return sh("format-patch", "-1", "--stdout", m.group(1)) + "\n"
            return call

        r = decide(prev(), P, H, GhSource("o/r", "main", fake_call()), TOOLS, LEDGER)
        expect(r["mode"] == "carry" and r["main_files"] == ["f.txt", "justfile"], f"through the API, a pure rebase carries: {r['mode']} {r['why']}")
        r = decide(prev(), P, H, GhSource("o/r", "main", fake_call(fail=("commits/" + H,))), TOOLS, LEDGER)
        expect(r["mode"] == "full" and "HTTP 404" in r["why"], f"a failed patch call is named: {r['mode']} {r['why']}")
        for what, fail in (("the reviewed head's", f"...{P}?"), ("the new head's", f"...{H}?")):
            r = decide(prev(), P, H, GhSource("o/r", "main", fake_call(fail=(fail,))), TOOLS, LEDGER)
            expect(r["mode"] == "full" and "HTTP 404" in r["why"], f"a failed compare of {what} commits is a named full review: {r['why']}")
        # the window listing failing does not stop a carry; it says so
        bad_window = fake_call(fail=(f"{sh('merge-base', 'main', P)}...{sh('merge-base', 'main', H)}",))
        r = decide(prev(), P, H, GhSource("o/r", "main", bad_window), TOOLS, LEDGER)
        expect(r["mode"] == "carry" and "could not be listed" in r["main_note"], f"an unlisted window is named in the carry: {r['mode']} {r.get('main_note')}")

        # 16. Supersedes: the same author, a closed unmerged pull request from the same repository, a bot review
        def pulls(**over):
            n = {"state": "closed", "merged": False, "user": {"login": "me"}, "head": {"repo": {"full_name": "me/fork"}}}
            n.update(over)
            return n

        def sup_call(old, bodies):
            def call(path, accept=None, paginate=False, jq=None):
                if path.endswith("/reviews?per_page=100"):
                    return "".join(json.dumps({"body": b}) + "\n" for b in bodies)
                return json.dumps(old)
            return call

        newpr = {"number": 9, "body": "Replaces the rebased one.\n\nSupersedes #5.\n", "user": {"login": "me"},
                 "head": {"repo": {"full_name": "me/fork"}}}
        good = [prev(head="1" * 40), prev(head="2" * 40)]
        got = supersedes(newpr, "o/r", sup_call(pulls(), good))
        expect(got == ("2" * 40, 5, ""), f"Supersedes names the last head the bot reviewed on #5: {got}")
        for what, old, bodies, want in (
                ("a merged pull request", pulls(merged=True), good, "merged"),
                ("an open pull request", pulls(state="open"), good, "open"),
                ("another author", pulls(user={"login": "you"}), good, "another author"),
                ("another head repository", pulls(head={"repo": {"full_name": "you/fork"}}), good, "head repository"),
                ("a deleted head repository", pulls(head={"repo": None}), good, "head repository"),
                ("no bot review", pulls(), ["a person's words"], "no review"),
                ("no reviews at all", pulls(), [], "no review")):
            got = supersedes(newpr, "o/r", sup_call(old, bodies))
            expect(got[0] is None and want in got[2], f"Supersedes refuses {what}: {got}")
        for what, body, want in (("a body with no line", "just a body", ""), ("two pull requests", "Supersedes #5\nSupersedes #6", "several"),
                                 ("itself", "Supersedes #9", "itself")):
            got = supersedes({**newpr, "body": body}, "o/r", sup_call(pulls(), good))
            expect(got[0] is None and want in got[2] and (want or got[2] == ""), f"Supersedes with {what}: {got}")
        def gone(*a, **k):
            raise Unavailable("gh api x ended 1: HTTP 404")

        got = supersedes(newpr, "o/r", gone)
        expect(got[0] is None and "could not be read" in got[2], f"Supersedes with a failed call is named: {got}")
        # the replacement pull request's first decision starts from the head the bot reviewed on the superseded one
        sup_head, _, _ = supersedes(newpr, "o/r", sup_call(pulls(), [prev(head=P)]))
        r = decide(prev(), sup_head, rebased("replacement"), GhSource("o/r", "main", fake_call()), TOOLS, LEDGER)
        expect(sup_head == P and r["mode"] == "carry", f"a replacement carries from the superseded pull request's head: {r['mode']} {r['why']}")

        # 17. the reviewer's identity moves with what a review runs under, and with nothing else
        root = os.path.join(d, "tree")
        for rel, text in ((".github/review/reviewer.md", "a"), ("tools/review/post.py", "b")):
            os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
            open(os.path.join(root, rel), "w").write(text)
        a0 = reviewer_id(root)
        os.makedirs(os.path.join(root, "tools/review/__pycache__"))
        open(os.path.join(root, "tools/review/__pycache__/post.cpython-312.pyc"), "w").write("junk")
        open(os.path.join(root, "README.md"), "w").write("outside")
        expect(re.fullmatch(r"[0-9a-f]{16}", a0) and reviewer_id(root) == a0, "reviewer-id ignores bytecode and files outside both directories")
        open(os.path.join(root, "tools/review/post.py"), "w").write("c")
        a1 = reviewer_id(root)
        open(os.path.join(root, ".github/review/reviewer.md"), "w").write("d")
        expect(a1 != a0 and reviewer_id(root) != a1, "reviewer-id moves with a tool and with an instruction file")
        shutil.rmtree(os.path.join(root, ".github/review"))
        try:
            reviewer_id(root)
            expect(False, "reviewer-id of a tree without .github/review was accepted")
        except ValueError:
            expect(True, "reviewer-id of a tree without .github/review is a named error")

    print(f"carry.py self-test: {'ok' if not fails else 'FAIL'} ({len(cases)} cases, {len(fails)} failed)")
    return 1 if fails else 0


# ---------------------------------------------------------------------------------------------------------------------

def cmd_decide(a):
    try:
        body = open(a.prev_review, encoding="utf-8").read()
    except OSError:
        body = ""
    d = decide(body, a.prev, a.head, GhSource(a.repo, a.base_ref), a.tools, a.ledger)
    d["prev_pr"] = a.prev_pr
    d["entries"] = [list(e) for e in d.get("entries", [])]
    os.makedirs(a.out, exist_ok=True)
    text = d.pop("interdiff", None)
    if d["mode"] != "full":
        shutil.copyfile(a.prev_review, os.path.join(a.out, "prev-review.md"))
    if text is not None:
        with open(os.path.join(a.out, "interdiff.patch"), "w", encoding="utf-8") as f:
            f.write(text)
    with open(os.path.join(a.out, "carry.json"), "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1)
    print(f"{d['mode']}: {d['why']}")
    return 0


def cmd_supersedes(a):
    pr = json.load(open(a.pr_json, encoding="utf-8"))
    head, n, why = supersedes(pr, a.repo)
    if head:
        print(f"ok {head} {n}")
    elif why:
        print(f"none {why}")
    return 0


def cmd_compare(a):
    src = GhSource(a.repo, a.base_ref)
    old, new = src.series(a.prev)[0], src.series(a.head)[0]
    entries = pair_up(old, new, [patch_id(c.patch) for c in old], [patch_id(c.patch) for c in new])
    for k, i, j in entries:
        print(f"{k:9} new {'-' if j is None else j + 1:>2}  prev {'-' if i is None else i + 1:>2}  "
              f"{(new[j] if j is not None else old[i]).subject[:80]}")
    print(f"same series: {same_series(entries)}; {summary(entries)}")
    return 0


def main(argv):
    if argv == ["--self-test"]:
        return self_test()
    ap = argparse.ArgumentParser(prog="tools/review/carry.py", description="Shingler's verdict across a rebase")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("decide")
    for k in ("--repo", "--head", "--prev", "--prev-review", "--base-ref", "--ledger", "--tools", "--out"):
        s.add_argument(k, required=True)
    s.add_argument("--prev-pr", type=int)
    s = sub.add_parser("supersedes")
    s.add_argument("--repo", required=True)
    s.add_argument("--pr-json", required=True)
    s = sub.add_parser("reviewer-id")
    s.add_argument("--root", required=True)
    s = sub.add_parser("compare")
    for k in ("--repo", "--head", "--prev", "--base-ref"):
        s.add_argument(k, required=True)
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 64 if e.code else 0
    try:
        if a.cmd == "reviewer-id":
            print(reviewer_id(a.root))
            return 0
        return {"decide": cmd_decide, "supersedes": cmd_supersedes, "compare": cmd_compare}[a.cmd](a)
    except (ValueError, Unavailable) as e:
        print(f"carry.py {a.cmd}: {e}", file=sys.stderr)
        return 70


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
