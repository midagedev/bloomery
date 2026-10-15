#!/usr/bin/env python3
"""Shingler's reviewer on this machine, before a push: the PR review's own instructions, catalog and ledger, run through
the base tree's tools/review/run-claude.sh under the caller's own Claude login (`just pre-review`).

    tools/review/local.py [--base REF] [--full] [--minutes N] [--body FILE] [--no-pre-checks]
    tools/review/local.py --self-test

What it reviews is the committed HEAD, what a push sends; a dirty tree is named and left out. The trees come from
`git archive`, BASE_TREE at REF (default origin/main, fetched first) and PR_TREE at HEAD, so no checkout hook adds a
file to either. CTX holds what .github/workflows/review.yml's "Build the review context" step writes, built from git
instead of the pull request API: diff.patch (BASE...HEAD), files.txt (the PR's files, its format), pr.json (the same
keys; number 0 and the head's subject when the branch has no open pull request), open-issues.txt and ledger.md (read
with gh; an empty file and a named line when gh cannot), device-dirs.txt (post.py device), diagnostics.md (dryer and
crapper at the commits review.yml pins, installed once where the workflow installs them; one line saying they did not
run when they cannot be had). When the branch's pull request already has a Shingler review of an ancestor of HEAD, the run is incremental as the
workflow's is: the diff since that head, and that review as prev-review.md.

By default only the reviewer runs: the validator only removes findings, and here a false one costs a read, not a
round trip. --full runs the validator after it, as the workflow does. Then `post.py check` reads the output as it
does before posting. Nothing is posted and the workflow is not touched; review.yml stays the review of record.

Before the reviewer, the text-only checks a push needs and nobody runs by hand run on PR_TREE when the branch touches
their inputs (PRE_CHECKS: tools/check-levers.sh, about 10 s, for tools/, the justfile, .cargo/ or crates/); a red one
ends the run 1 by name, before the reviewer's cost. --no-pre-checks skips them. The failure it blocks: a lever or
marker name that only a hand-run check-levers caught.

Two pre-reviews run at once; a third waits on a flock slot. device-dirs.txt is cached by the base sha, and the
summary's context time starts after the pre-checks.

`claude --version` runs first on a real review. A binary that is on PATH and does not answer is a broken install.

Prints the verdict, the must-fix titles and where the output is (~/.cache/bloomery/pre-review/<branch>-<sha8>/).
A CLEAN review also prints one line for the pull request body, with the full HEAD sha.
A base that lacks commits origin/main has is behind it and is refused by name before the reviewer runs.
A missing origin/main is not that case. A failed ancestry walk stops 70.
Exit: 0 CLEAN; 1 MUST_FIX or a red pre-check; 75 no claude CLI, a broken install, or a role past --minutes (default 10); 64 usage,
or a base behind origin/main; 65 output post.py refused
or a review with no verdict; 70 a context step or a git command that failed (named).
The failure it blocks: a pull request whose first Shingler review is a MUST_FIX, each fold another 10-27 minute round.
"""

import argparse
import fcntl
import contextlib
import io
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from post import MARKER, VERDICT_RE  # noqa: E402
from run import MINUTES, TIMEOUT, launch, produced  # noqa: E402
sys.path.insert(0, os.path.dirname(HERE))
from recipes import RecipeError, archive  # noqa: E402

REPO = "midagedev/bloomery"
# (script, the path prefixes that run it): a text-only check of PR_TREE, run when the branch changes one of them
PRE_CHECKS = (("tools/check-levers.sh", ("tools/", "justfile", ".cargo/", "crates/")),)
NET_S = 60  # each gh or git fetch call: past it the step reads as unanswered, never as a hang
MARKER_RE = re.compile(rf"<!-- {re.escape(MARKER)} head_sha=([0-9a-f]{{40}})")
# The files the workflow's context step leaves in CTX (pr-full.json is removed there); prev-review.md only on a re-review.
CTX_FILES = ("diff.patch", "files.txt", "pr.json", "open-issues.txt", "ledger.md", "device-dirs.txt", "diagnostics.md")
CTX_RE_REVIEW = ("prev-review.md",)
TRANSIENT = ("pr-full.json",)
PR_KEYS = ("number", "title", "body", "author", "repo", "base_sha", "head_sha", "tree_sha", "mode", "ledger", "run")
STATUS = {"A": "added", "D": "removed", "M": "modified", "R": "renamed", "C": "copied", "T": "changed"}
NO_DIAG = "The diagnostics did not run on this machine (no dryer or crapper); no dryer or crapper results for this review."


class Stop(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code, self.msg = code, msg


def git(*args):
    p = subprocess.run(["git", *args], capture_output=True, text=True)
    if p.returncode:
        raise Stop(70, f"pre-review: git {' '.join(args)} failed: {p.stderr.strip()[:200]}")
    return p.stdout


def git_ok(*args):
    """stdout of a git call, or None when git refuses it. A clone with no origin/main is not refused for that."""
    p = subprocess.run(["git", *args], capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else None


def gh_json(*args):
    """gh's JSON answer, or None when gh cannot answer (no gh, no login, no network)."""
    if not shutil.which("gh"):
        return None
    try:
        p = subprocess.run(["gh", *args], capture_output=True, text=True, timeout=NET_S)
        if p.returncode:
            return None
        return json_pages(p.stdout) if "--paginate" in args else json.loads(p.stdout)
    except (subprocess.TimeoutExpired, ValueError):
        return None


def json_pages(text):
    """`gh api --paginate` prints one JSON document per page (`--slurp` is newer than the gh some machines carry):
    each is decoded in turn, and list pages are joined."""
    dec, i, out = json.JSONDecoder(), 0, []
    while True:
        while i < len(text) and text[i].isspace():
            i += 1
        if i == len(text):
            return out
        page, i = dec.raw_decode(text, i)
        if not isinstance(page, list):
            raise ValueError("a paginated gh page is not a list")
        out += page


def files_txt(name_status, numstat):
    """files.txt as the workflow writes it from the pull request's files: `<status>\t+<added>\t-<deleted>\t<path>`,
    a rename under its new path with its own counts, a binary file's counts 0. The inputs are `git diff -z -M
    --name-status` and `-z -M --numstat`: NUL-separated, a rename's two paths as two fields."""
    counts, f, i = {}, numstat.split("\0"), 0
    while i < len(f) - 1:
        a, d, path = f[i].split("\t", 2)
        if path == "":  # a rename or copy: the two paths follow
            path, i = f[i + 2], i + 3
        else:
            i += 1
        counts[path] = (0 if a == "-" else int(a), 0 if d == "-" else int(d))
    out, f, i = [], name_status.split("\0"), 0
    while i < len(f) - 1:
        st = f[i]
        if st[0] in "RC":
            path, i = f[i + 2], i + 3
        else:
            path, i = f[i + 1], i + 2
        a, d = counts.get(path, (0, 0))
        out.append(f"{STATUS.get(st[0], 'modified')}\t+{a}\t-{d}\t{path}")
    return "\n".join(out) + ("\n" if out else "")


def last_review(pr_number):
    """(head sha, body) of the last Shingler review of the pull request, or None."""
    reviews = gh_json("api", "--paginate", f"repos/{REPO}/pulls/{pr_number}/reviews")
    if not isinstance(reviews, list):
        return None
    for r in reversed(reviews):
        m = MARKER_RE.search(r.get("body") or "")
        if (r.get("user") or {}).get("login") == "github-actions[bot]" and m:
            return m.group(1), r["body"]
    return None


def is_ancestor(a, b):
    return subprocess.run(["git", "merge-base", "--is-ancestor", a, b], capture_output=True).returncode == 0


def build_ctx(ctx, base_tree, base_sha, head, args, say):
    """CTX as review.yml's context step leaves it; returns pr.json's dict."""
    merge_base = git("merge-base", base_sha, head).strip()
    branch = git("rev-parse", "--abbrev-ref", "HEAD").strip()
    pr = gh_json("pr", "view", branch, "--repo", REPO, "--json", "number,title,body,author,headRefOid,state")
    if pr and pr.get("state") != "OPEN":
        pr = None
    mode, diff_from = "full", merge_base
    if pr:
        prev = last_review(pr["number"])
        if prev and prev[0] != head and is_ancestor(prev[0], head):
            mode, diff_from = "incremental", prev[0]
            with open(os.path.join(ctx, "prev-review.md"), "w") as f:
                f.write(prev[1])
    with open(os.path.join(ctx, "diff.patch"), "w") as f:
        f.write(git("diff", diff_from, head) if mode == "incremental" else git("diff", f"{base_sha}...{head}"))
    with open(os.path.join(ctx, "files.txt"), "w") as f:
        f.write(files_txt(git("diff", "-z", "-M", "--name-status", f"{base_sha}...{head}"),
                          git("diff", "-z", "-M", "--numstat", f"{base_sha}...{head}")))
    issues = gh_json("api", "--paginate", f"repos/{REPO}/issues?labels=review-bot&state=open&per_page=100")
    with open(os.path.join(ctx, "open-issues.txt"), "w") as f:
        if isinstance(issues, list):
            f.write("".join(f"#{i['number']} {i['title']}\n" for i in issues))
        else:
            say("pre-review: gh could not list the bot's open issues; open-issues.txt is empty")
    device_dirs(base_tree, os.path.join(ctx, "device-dirs.txt"), base_sha)
    ledger = read_ledger(base_tree, os.path.join(ctx, "ledger.md"))
    if ledger is None:
        say("pre-review: the slag ledger could not be read (gh); ledger.md is empty")
        open(os.path.join(ctx, "ledger.md"), "w").close()
        ledger = ""
    diag(ctx, base_tree, base_sha, head, say)
    subject = git("log", "-1", "--format=%s", head).strip()
    body = open(args.body).read() if args.body else ""
    meta = {
        "number": pr["number"] if pr else 0,
        "title": pr["title"] if pr else subject,
        "body": (pr.get("body") or "") if pr and not args.body else body,
        "author": ((pr or {}).get("author") or {}).get("login") or git("config", "user.name").strip(),
        "repo": REPO, "base_sha": merge_base, "head_sha": head, "tree_sha": base_sha,
        "mode": mode, "ledger": ledger, "run": "local",
    }
    with open(os.path.join(ctx, "pr.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def device_dirs(base_tree, out, base_sha):
    """post.py device, copied from the cache for this base sha when that file is already there."""
    cache = os.path.expanduser(f"~/.cache/bloomery-review/device-{base_sha}")
    if os.path.isfile(cache):
        shutil.copy(cache, out)
        return
    # No toolchain override: rustup reads rust-toolchain.toml. install.sh installs only that channel, not stable.
    p = subprocess.run([sys.executable, os.path.join(base_tree, "tools/review/post.py"), "device", "--root", base_tree,
                        "--out", out], capture_output=True, text=True)
    if p.returncode:
        raise Stop(70, f"pre-review: post.py device failed: {p.stderr.strip()[:200]}")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    tmp = f"{cache}.{os.getpid()}"
    shutil.copy(out, tmp)
    os.replace(tmp, cache)


def review_slot(say=lambda m: None):
    """One of two flock slots. A third pre-review polls both and takes the first that frees."""
    d = os.path.expanduser("~/.cache/bloomery-review")
    os.makedirs(d, exist_ok=True)
    fds = [open(os.path.join(d, f"slot{i}"), "a") for i in (0, 1)]
    said = False
    while True:
        for f in fds:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return f
            except BlockingIOError:
                continue
        if not said:
            say("pre-review: both review slots are busy; waiting")
            said = True
        time.sleep(0.05)


def read_ledger(base_tree, out):
    """The ledger's version (ledger.py prints it), the lessons written to out; None when gh cannot read it. ledger.py
    reads GH_TOKEN, which a plain `gh auth login` does not set: it comes from `gh auth token` then, into the child's
    environment only."""
    env = dict(os.environ)
    if not env.get("GH_TOKEN") and shutil.which("gh"):
        try:
            t = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=NET_S)
        except subprocess.TimeoutExpired:
            return None
        if t.returncode == 0 and t.stdout.strip():
            env["GH_TOKEN"] = t.stdout.strip()
    try:
        p = subprocess.run([sys.executable, os.path.join(base_tree, "tools/review/ledger.py"), "accepted", "--repo", REPO,
                            "--out", out], capture_output=True, text=True, env=env, timeout=NET_S)
    except subprocess.TimeoutExpired:
        return None
    return p.stdout.strip() if p.returncode == 0 else None


UB = "~/.cache/bloomery-review/ub"  # where review.yml installs dryer and crapper; the same place here


def ub_pins(base_tree):
    """The dryer and crapper commits review.yml's install step pins (its DRYER_SHA and CRAPPER_SHA), or None."""
    try:
        text = open(os.path.join(base_tree, ".github/workflows/review.yml"), encoding="utf-8").read()
    except OSError:
        return None
    pins = [re.search(rf"\b{k}_SHA: ([0-9a-f]{{40}})\b", text) for k in ("DRYER", "CRAPPER")]
    return (pins[0][1], pins[1][1]) if all(pins) else None


def install_ub(root, pins):
    """review.yml's install step, here: each tool at its pinned commit, one venv, built in place (a venv's scripts name
    its absolute path, so it cannot be moved); the pins file is written last and marks it complete."""
    shutil.rmtree(root, ignore_errors=True)
    for name, sha in zip(("dryer", "crapper"), pins):
        d = os.path.join(root, name)
        for c in (["git", "init", "-q", d], ["git", "-C", d, "fetch", "-q", "--depth", "1", f"https://github.com/unclebob/{name}", sha],
                  ["git", "-C", d, "checkout", "-q", "FETCH_HEAD"]):
            subprocess.run(c, check=True, capture_output=True, timeout=NET_S)
    subprocess.run([sys.executable, "-m", "venv", os.path.join(root, "venv")], check=True, capture_output=True, timeout=NET_S)
    subprocess.run([os.path.join(root, "venv/bin/pip"), "install", "-q", os.path.join(root, "dryer"), os.path.join(root, "crapper")],
                   check=True, capture_output=True, timeout=3 * NET_S)
    with open(os.path.join(root, "pins"), "w") as f:
        f.write(" ".join(pins) + "\n")


def ub_bins(base_tree, say, root=None, install=install_ub):
    """dryer and crapper at review.yml's pins, installed once (about 20 s) when missing or at other pins; None when
    they cannot be had (the reason is printed). The failure it blocks: a duplicate only Shingler's dryer reported."""
    root = os.path.expanduser(root or UB)
    pins = ub_pins(base_tree)
    if pins is None:
        say("pre-review: review.yml names no DRYER_SHA/CRAPPER_SHA; the diagnostics do not run")
        return None
    bins = tuple(os.path.join(root, "venv/bin", t) for t in ("dryer", "crapper"))
    try:
        have = open(os.path.join(root, "pins")).read().split()
    except OSError:
        have = []
    if have != list(pins) or not all(os.access(b, os.X_OK) for b in bins):
        say(f"pre-review: installing dryer {pins[0][:8]} and crapper {pins[1][:8]} under {root} (once)")
        try:
            install(root, pins)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
            say(f"pre-review: the dryer/crapper install failed ({e}); the diagnostics do not run")
            return None
    return bins


def diag(ctx, base_tree, base_sha, head, say):
    """diagnostics.md from dryer and crapper; the shared cache is held under one lock (a sibling file, which the
    install's rmtree leaves) from the pin check through the tools' use, so two runs at other pins never mix it."""
    root = os.path.expanduser(UB)
    os.makedirs(os.path.dirname(root), exist_ok=True)
    with open(root + ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        out = os.path.join(ctx, "diagnostics.md")
        bins = ub_bins(base_tree, say, root=root)
        if not bins:
            with open(out, "w") as f:
                f.write(NO_DIAG + "\n")
            return
        dryer, crapper = bins
        changed = os.path.join(ctx, ".changed")
        with open(changed, "w") as f:
            f.write(git("diff", "--name-only", f"{base_sha}...{head}"))
        p = subprocess.run([sys.executable, os.path.join(base_tree, "tools/review/diagnostics.py"), "--base", base_tree,
                            "--pr", os.path.join(os.path.dirname(ctx), "pr"), "--changed", changed,
                            "--diff", os.path.join(ctx, "diff.patch"), "--out", out, "--dryer", dryer, "--crapper", crapper],
                           capture_output=True, text=True)
        os.unlink(changed)
        if p.returncode or not os.path.isfile(out):
            say("pre-review: diagnostics.py failed; diagnostics.md says so")
            with open(out, "w") as f:
                f.write(NO_DIAG + "\n")


def run_role(role, d, minutes, say):
    """One role through the base tree's run-claude.sh, by the workflow's own launcher (run.py): its own process group,
    stopped past the bound or when we are; its output in trace/<role>.log, whose summary line is printed. Its rc, TIMEOUT
    past the bound. A role that ends 0 without its output (run.py's produced: the validator's line in review.md, the
    reviewer's verdict and decision.txt) ends the run 65 by name."""
    MINUTES[role] = minutes
    paths = [os.path.join(d, x) for x in ("base", "pr", "ctx", "out", "trace")]
    with tempfile.TemporaryDirectory() as snap:
        before = os.path.join(snap, "out")
        shutil.copytree(paths[3], before)
        rc = launch("claude", role, paths, dict(os.environ, REVIEW_SETTING_SOURCES="project"))
        if rc == 0:
            reason = produced(role, paths[3], before)
            if reason:
                raise Stop(65, f"pre-review: the {role} ended 0 without its output: {reason}")
    log = os.path.join(d, "trace", f"{role}.log")
    if os.path.exists(log):
        for line in open(log, encoding="utf-8", errors="replace"):
            if line.startswith(f"{role}: "):
                say(line.rstrip("\n"))
    return rc


def claude_ok():
    """`claude --version` must answer. On PATH and silent or red is a broken install, not a missing binary."""
    try:
        p = subprocess.run(["claude", "--version"], capture_output=True, text=True,
                           stdin=subprocess.DEVNULL, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise Stop(75, f"pre-review: claude --version failed ({type(e).__name__}): broken install; nothing was reviewed") from e
    if p.returncode != 0 or not (p.stdout + p.stderr).strip():
        raise Stop(75, f"pre-review: claude --version failed (rc {p.returncode}): broken install; nothing was reviewed")


def pre_checks(pr_tree, paths, say):
    """PRE_CHECKS whose prefixes a changed path starts with, on PR_TREE; the first red one stops the run 1, named."""
    for script, prefixes in PRE_CHECKS:
        if not any(p.startswith(prefixes) for p in paths):
            continue
        t0 = time.time()
        r = subprocess.run(["bash", os.path.join(pr_tree, script)], cwd=pr_tree, capture_output=True, text=True,
                           stdin=subprocess.DEVNULL)
        if r.returncode:
            tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-8:])
            raise Stop(1, f"pre-review: {script} is red on this head (rc {r.returncode}); fix it before the push:\n{tail}")
        say(f"pre-review: {script} ok ({time.time() - t0:.0f} s)")


def must_fix_titles(body):
    """The must-fix items' titles (`#### <n>. <title>` under `### Must fix`)."""
    m = re.search(r"^### Must fix\s*$(.*?)(?=^### |\Z)", body, re.M | re.S)
    return re.findall(r"^#### (.+)$", m.group(1), re.M) if m else []


def review(args, stubbed=False, say=lambda m: print(m, file=sys.stderr)):
    slot = None if stubbed else review_slot(say)  # held until this review returns
    if not stubbed and not shutil.which("claude"):
        raise Stop(75, "pre-review: no claude CLI on PATH; nothing was reviewed")
    if not stubbed:
        claude_ok()
    try:
        fetched = subprocess.run(["git", "fetch", "-q", "origin"], capture_output=True, timeout=NET_S).returncode == 0
    except subprocess.TimeoutExpired:
        fetched = False
    if not fetched:
        say(f"pre-review: git fetch origin failed or took over {NET_S} s; {args.base} is as this clone last saw it")
    head = git("rev-parse", "HEAD").strip()
    base_sha = git("rev-parse", args.base).strip()
    main_sha = git_ok("rev-parse", "--verify", "origin/main")
    if main_sha and base_sha != main_sha:
        behind = git("rev-list", "--count", f"{base_sha}..{main_sha}").strip()
        if behind != "0":
            noun = "commit" if behind == "1" else "commits"
            raise Stop(64, f"pre-review: base {args.base} ({base_sha[:8]}) is behind origin/main ({main_sha[:8]}): "
                           f"origin/main has {behind} {noun} the base does not")
    if git("status", "--porcelain", "--untracked-files=no").strip():
        say("pre-review: the tree has uncommitted changes; what is reviewed is the committed HEAD, what a push sends")
    branch = re.sub(r"[^A-Za-z0-9._-]", "_", git("rev-parse", "--abbrev-ref", "HEAD").strip())
    keep = os.path.expanduser(f"~/.cache/bloomery/pre-review/{branch}-{head[:8]}")
    shutil.rmtree(keep, ignore_errors=True)
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="pre-review.") as d:
        try:
            archive(base_sha, os.path.join(d, "base"), cwd=os.getcwd())
            archive(head, os.path.join(d, "pr"), cwd=os.getcwd())
        except RecipeError as e:
            raise Stop(70, f"pre-review: {e}") from e
        if args.no_pre_checks:
            say("pre-review: pre-checks skipped (--no-pre-checks)")
        else:
            pre_checks(os.path.join(d, "pr"), git("diff", "--name-only", "-z", f"{base_sha}...{head}").split("\0"), say)
        t_pre = time.time()
        for x in ("ctx", "out", "trace"):
            os.makedirs(os.path.join(d, x))
        try:
            meta = build_ctx(os.path.join(d, "ctx"), os.path.join(d, "base"), base_sha, head, args, say)
            t_ctx = time.time() - t_pre
            roles = ("reviewer", "validator") if args.full else ("reviewer",)
            for role in roles:
                rc = run_role(role, d, args.minutes, say)
                if rc == TIMEOUT:
                    raise Stop(75, f"pre-review: the {role} ran past {args.minutes} min; no verdict (push is your call); see {keep}")
                if rc:
                    raise Stop(70, f"pre-review: the {role} ended rc {rc}; its transcript is in {keep}/trace")
        finally:
            # every path out keeps what ran, a failed or stopped role's transcript included
            os.makedirs(os.path.dirname(keep), exist_ok=True)
            for x in ("ctx", "out", "trace"):
                shutil.copytree(os.path.join(d, x), os.path.join(keep, x), dirs_exist_ok=True)
        usages = [os.path.join(d, "trace", f"{r}.usage.json") for r in roles]
        p = subprocess.run([sys.executable, os.path.join(d, "base/tools/review/post.py"), "check", "--out", os.path.join(d, "out"),
                            "--diff", os.path.join(d, "ctx/diff.patch"), "--pr-json", os.path.join(d, "ctx/pr.json"),
                            "--files", os.path.join(d, "ctx/files.txt"), "--device", os.path.join(d, "ctx/device-dirs.txt"),
                            "--base", os.path.join(d, "base"), "--secret-env", "ANTHROPIC_API_KEY",
                            *[a for u in usages if os.path.isfile(u) for a in ("--usage", u)]],
                           capture_output=True, text=True)
        for x in ("ctx", "out", "trace"):
            shutil.copytree(os.path.join(d, x), os.path.join(keep, x), dirs_exist_ok=True)
        if p.returncode:
            raise Stop(65, f"pre-review: post.py check refused the output (rc {p.returncode}): {p.stderr.strip()[:200]}; see {keep}")
        body = open(os.path.join(d, "out/review.md")).read()
    m = VERDICT_RE.search(body)
    if not m:
        raise Stop(65, f"pre-review: the review has no verdict line; see {keep}/out/review.md")
    titles = must_fix_titles(body)
    print(f"pre-review: {m.group(1)} — {meta['mode']} review of {head[:8]} against {args.base} "
          f"({'reviewer + validator' if args.full else 'reviewer'}, {time.time() - t0:.0f} s of which context {t_ctx:.0f} s)")
    for t in titles[:12]:
        print(f"  must fix: {t}")
    print(f"  review: {keep}/out/review.md")
    if m.group(1) == "CLEAN":
        print(f"pre-review: PR body: Pre-review CLEAN at {head}.")
    return 1 if m.group(1) == "MUST_FIX" else 0


# ---------------------------------------------------------------------------------------------------------------------

def self_test():
    fails = []

    def expect(ok, what):
        print(("ok   " if ok else "FAIL ") + what)
        if not ok:
            fails.append(what)

    # the context matches the workflow's: the files its step leaves, and pr.json's keys (read from review.yml itself)
    wf = open(os.path.join(HERE, "../../.github/workflows/review.yml")).read()
    step = wf[wf.index("- name: Build the review context"):wf.index("- name: Logins")]
    written = set(re.findall(r"ctx/([a-z0-9._-]+)", step)) - set(TRANSIENT)
    expect(written == set(CTX_FILES) | set(CTX_RE_REVIEW),
           f"ctx files are review.yml's: workflow {sorted(written)}, here {sorted(set(CTX_FILES) | set(CTX_RE_REVIEW))}")
    m = re.search(r"'\{number, title, body, author: \.user\.login, repo: \$repo, base_sha: \.base\.sha, head_sha: \.head\.sha,"
                  r"\s*tree_sha: \$tree, mode: \$mode, ledger: \$ledger, run: \$run\}'", step)
    expect(m is not None, "pr.json's keys are the workflow's jq object (number … run)")
    # files.txt: GitHub's status words, a rename under its new path, binary counts 0
    with tempfile.TemporaryDirectory() as t:
        sh = lambda *c: subprocess.run(c, cwd=t, check=True, capture_output=True, text=True).stdout
        sh("git", "init", "-q")
        sh("git", "config", "user.email", "t@t")
        sh("git", "config", "user.name", "t")
        os.makedirs(os.path.join(t, "x"))
        open(os.path.join(t, "x/a.py"), "w").write("".join(f"line {i}\n" for i in range(20)))
        open(os.path.join(t, "old.rs"), "w").write("o\n" * 9)
        open(os.path.join(t, "img.png"), "wb").write(b"\0\1\2")
        sh("git", "add", "-A")
        sh("git", "commit", "-qm", "a")
        os.rename(os.path.join(t, "x/a.py"), os.path.join(t, "x/b.py"))
        open(os.path.join(t, "x/b.py"), "a").write("one more\n")
        os.unlink(os.path.join(t, "old.rs"))
        open(os.path.join(t, "new.rs"), "w").write("n\n" * 3)
        open(os.path.join(t, "img.png"), "wb").write(b"\0\3")
        sh("git", "add", "-A")
        sh("git", "commit", "-qm", "b")
        got = files_txt(sh("git", "diff", "-z", "-M", "--name-status", "HEAD~1", "HEAD"),
                        sh("git", "diff", "-z", "-M", "--numstat", "HEAD~1", "HEAD"))
    expect(sorted(got.splitlines()) == ["added\t+3\t-0\tnew.rs", "modified\t+0\t-0\timg.png", "removed\t+0\t-9\told.rs",
                                        "renamed\t+1\t-0\tx/b.py"],
           f"files.txt from git's own output: a rename under its new path with its counts, a binary 0: {got!r}")
    # must-fix titles from the body format reviewer.md asks for
    body = "**Verdict: MUST_FIX**: x\n\n### Must fix\n#### 1. One\ntext\n#### 2. Two\n\n### Checked\n#### not this\n"
    expect(must_fix_titles(body) == ["1. One", "2. Two"], f"must-fix titles: {must_fix_titles(body)}")
    expect(must_fix_titles("**Verdict: CLEAN**: ok\n\n### Checked\n") == [], "a clean review has no must-fix titles")
    # dryer and crapper: review.yml's pins, installed only when missing or at other pins
    with tempfile.TemporaryDirectory() as t:
        expect(ub_pins(os.path.normpath(os.path.join(HERE, "../.."))) is not None, "review.yml's install step names both pins")
        os.makedirs(os.path.join(t, "base/.github/workflows"))
        open(os.path.join(t, "base/.github/workflows/review.yml"), "w").write("DRYER_SHA: " + "a" * 40 + "\nCRAPPER_SHA: " + "b" * 40 + "\n")
        calls = []

        def fake_install(root, pins):
            calls.append(pins)
            os.makedirs(os.path.join(root, "venv/bin"), exist_ok=True)
            for b in ("dryer", "crapper"):
                open(os.path.join(root, "venv/bin", b), "w").close()
                os.chmod(os.path.join(root, "venv/bin", b), 0o755)
            open(os.path.join(root, "pins"), "w").write(" ".join(pins) + "\n")

        ub = os.path.join(t, "ub")
        got = ub_bins(os.path.join(t, "base"), lambda m: None, ub, fake_install)
        expect(got is not None and calls == [("a" * 40, "b" * 40)], f"missing tools are installed at the pins: {calls}")
        ub_bins(os.path.join(t, "base"), lambda m: None, ub, fake_install)
        expect(len(calls) == 1, f"tools at the pins are not installed again: {calls}")
        open(os.path.join(t, "base/.github/workflows/review.yml"), "w").write("DRYER_SHA: " + "c" * 40 + "\nCRAPPER_SHA: " + "b" * 40 + "\n")
        # the tools stay in place: only the pin comparison can call for the install
        got = ub_bins(os.path.join(t, "base"), lambda m: None, ub, fake_install)
        expect(got is not None and len(calls) == 2 and calls[-1][0] == "c" * 40, f"a moved pin installs again: {calls}")

        # the cache's lock: a run that finds it held waits for it before it looks at the pins
        lock_root = os.path.join(t, "ublock")
        seen = []
        g = globals()
        saved_ub, saved_bins = g["UB"], g["ub_bins"]
        g["UB"] = lock_root
        g["ub_bins"] = lambda base_tree, say, root=None: seen.append(time.time()) or None
        try:
            os.makedirs(os.path.join(t, "ctx"), exist_ok=True)
            with open(lock_root + ".lock", "a") as held:
                fcntl.flock(held, fcntl.LOCK_EX)
                th = threading.Thread(target=diag, args=(os.path.join(t, "ctx"), os.path.join(t, "base"), "a", "b", lambda m: None))
                th.start()
                time.sleep(0.5)
                waited = seen == []
            th.join(10)
            expect(waited and len(seen) == 1, f"diag waits for the cache's lock, then runs: waited {waited}, ran {len(seen)}")
        finally:
            g["UB"], g["ub_bins"] = saved_ub, saved_bins

        def broken(root, pins):
            raise subprocess.CalledProcessError(128, "git")

        said = []
        expect(ub_bins(os.path.join(t, "base"), said.append, os.path.join(t, "ub2"), broken) is None and any("failed" in m for m in said),
               f"a failed install is named and the diagnostics skip: {said}")
    # pre-checks: run only for a path under their prefixes; a red one stops the run 1 naming the script and its output
    with tempfile.TemporaryDirectory() as t:
        os.makedirs(os.path.join(t, "tools"))
        lev = os.path.join(t, "tools/check-levers.sh")
        open(lev, "w").write("echo 'check-levers: SOME_NAME is no row'; exit 3\n")
        said = []
        pre_checks(t, ["a.txt", "docs/x.md"], said.append)
        expect(said == [], f"a branch outside the prefixes runs no pre-check: {said}")
        try:
            pre_checks(t, ["a.txt", "tools/box.sh"], said.append)
            got = None
        except Stop as e:
            got = e
        expect(got is not None and got.code == 1 and "tools/check-levers.sh is red" in got.msg and "SOME_NAME is no row" in got.msg,
               f"a red check-levers stops the run 1 by name with its output: {got and (got.code, got.msg)}")
        open(lev, "w").write("exit 0\n")
        pre_checks(t, ["crates/gpu/src/lib.rs"], said.append)
        expect(len(said) == 1 and said[0].startswith("pre-review: tools/check-levers.sh ok"), f"a green one is printed: {said}")
    # claude --version: a binary that exits red is a broken install (75), and the message names it
    with tempfile.TemporaryDirectory() as t:
        fake = os.path.join(t, "claude")
        open(fake, "w").write("#!/bin/sh\necho broken >&2\nexit 1\n")
        os.chmod(fake, 0o755)
        saved_path = os.environ.get("PATH")
        os.environ["PATH"] = t + os.pathsep + (saved_path or "")
        try:
            try:
                claude_ok()
                got = None
            except Stop as e:
                got = e
            expect(got is not None and got.code == 75 and "broken install" in got.msg,
                   f"a claude that fails --version ends 75 naming a broken install: {got and (got.code, got.msg)}")
            open(fake, "w").write("#!/bin/sh\necho 2.0.0\nexit 0\n")
            os.chmod(fake, 0o755)
            try:
                claude_ok()
                ok_ver = True
            except Stop as e:
                ok_ver = False
            expect(ok_ver, "a claude that answers --version is a healthy install")
            for body in ("#!/bin/sh\nexit 0\n", "#!/bin/sh\nprintf ' \\n\\n'\nexit 0\n"):
                open(fake, "w").write(body)
                os.chmod(fake, 0o755)
                try:
                    claude_ok()
                    got = None
                except Stop as e:
                    got = e
                expect(got is not None and got.code == 75 and "broken install" in got.msg,
                       f"a claude that exits 0 with no version text ends 75 naming a broken install: {got and (got.code, got.msg)}")
        finally:
            if saved_path is None:
                os.environ.pop("PATH", None)
            else:
                os.environ["PATH"] = saved_path
    # gh's paginated answer is several JSON documents: every page is read; a credential lookup that times out is the
    # named empty-ledger fallback, not a traceback
    with tempfile.TemporaryDirectory() as t:
        fake = os.path.join(t, "gh")
        marker = "<!-- bloomery-review head_sha=" + "a" * 40 + " verdict=CLEAN"
        # gh prints the pages back to back, one JSON document each, as 2.46 does (no --slurp)
        pages = "".join(json.dumps(pg) for pg in ([{"user": {"login": "someone"}, "body": "x"}],
                                                  [{"user": {"login": "github-actions[bot]"}, "body": marker}])) \
            + "\n" + json.dumps([{"user": {"login": "z"}, "body": "y"}])
        open(fake, "w").write('#!/usr/bin/env bash\ncase "$*" in\n'
                              f'  "auth token") sleep 5 ;;\n  *--slurp*) echo "unknown flag: --slurp" >&2; exit 1 ;;\n  *--paginate*) printf %s \'{pages}\' ;;\n  *) echo \'[{{"number":1}}]\' ;;\nesac\n')
        os.chmod(fake, 0o755)
        saved_env = {k: os.environ.get(k) for k in ("PATH", "GH_TOKEN")}
        saved_net = globals()["NET_S"]
        os.environ["PATH"] = t + os.pathsep + os.environ["PATH"]
        os.environ.pop("GH_TOKEN", None)
        try:
            got = gh_json("api", "--paginate", "x")
            expect(isinstance(got, list) and len(got) == 3, f"a paginated gh answer is read page by page: {got}")
            expect(last_review(7) == ("a" * 40, marker), f"the last Shingler review is found across pages: {last_review(7)}")
            expect(gh_json("pr", "view") == [{"number": 1}], "an unpaginated gh answer is parsed as before")
            globals()["NET_S"] = 1
            expect(read_ledger(t, os.path.join(t, "ledger.md")) is None, "a timed-out credential lookup is the empty-ledger fallback")
        finally:
            globals()["NET_S"] = saved_net
            for k, v in saved_env.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    # two pre-reviews at once: a third waits until one slot frees
    with tempfile.TemporaryDirectory() as t:
        home = os.environ.get("HOME")
        os.environ["HOME"] = t
        held = []
        try:
            held = [review_slot(), review_slot()]
            seen = []
            said = []

            def third():
                review_slot(said.append)
                seen.append(1)

            th = threading.Thread(target=third)
            th.start()
            time.sleep(0.4)
            waited = seen == []
            held.pop().close()  # slot 1 frees; slot 0 stays held
            th.join(5)
            expect(waited and seen == [1] and not th.is_alive()
                   and said == ["pre-review: both review slots are busy; waiting"],
                   f"a third pre-review takes the free slot, not only slot 0, and says it is waiting: "
                   f"waited {waited}, ran {seen}, alive {th.is_alive()}, said {said}")
        finally:
            for f in held:
                f.close()
            if home is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = home
    # device dirs are cached by the base sha: the second call at one sha does not run post.py again
    with tempfile.TemporaryDirectory() as t:
        home = os.environ.get("HOME")
        os.environ["HOME"] = t
        ran = []
        envs = []
        real_run = subprocess.run

        def wrapped(argv, *a, **k):
            if len(argv) > 1 and str(argv[1]).endswith("post.py"):
                ran.append(1)
                envs.append((k.get("env") or {}).get("RUSTUP_TOOLCHAIN"))
                open(argv[argv.index("--out") + 1], "w").write("crates/gpu/\n")
                return subprocess.CompletedProcess(argv, 0, "", "")
            return real_run(argv, *a, **k)

        subprocess.run = wrapped
        real_replace = os.replace
        try:
            outs = [os.path.join(t, n) for n in ("a", "b", "c")]
            published = []

            def wrap_replace(src, dst):
                published.append(open(src).read())
                return real_replace(src, dst)

            os.replace = wrap_replace
            device_dirs(t, outs[0], "sha-a")
            device_dirs(t, outs[1], "sha-a")
            device_dirs(t, outs[2], "sha-b")
            expect(ran == [1, 1] and open(outs[1]).read() == "crates/gpu/\n" and published == ["crates/gpu/\n", "crates/gpu/\n"],
                   f"device dirs run once per base sha and replace the cache whole: ran {len(ran)}, published {published}")
            expect(envs == [None, None],
                   f"post.py device does not force RUSTUP_TOOLCHAIN: {envs}")
        finally:
            os.replace = real_replace
            subprocess.run = real_run
            if home is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = home
    # a scratch repo end to end with a stub runner: CLEAN 0, MUST_FIX 1, a slow runner 75, no verdict 65
    with tempfile.TemporaryDirectory() as t:
        repo = os.path.join(t, "r")
        os.makedirs(repo)
        sh = lambda *c: subprocess.run(c, cwd=repo, check=True, capture_output=True)
        sh("git", "init", "-q", "-b", "main")
        sh("git", "config", "user.email", "t@t")
        sh("git", "config", "user.name", "t")
        here_root = os.path.normpath(os.path.join(HERE, "../.."))
        for rel in ("tools/review/post.py", "tools/review/ledger.py", "tools/review/usage.py",
                    "tools/recipes.py", "justfile", "Cargo.toml"):
            src = os.path.join(here_root, rel)
            if os.path.exists(src):
                os.makedirs(os.path.dirname(os.path.join(repo, rel)), exist_ok=True)
                shutil.copy(src, os.path.join(repo, rel))
        # the base tree's run-claude.sh, which run.launch starts, hands the role to the case's stub
        stub = os.path.join(t, "stub.sh")
        open(os.path.join(repo, "tools/review/run-claude.sh"), "w").write('#!/usr/bin/env bash\nexec bash "$PRE_REVIEW_TEST_STUB" "$@"\n')
        open(os.path.join(repo, "a.txt"), "w").write("a\n")
        sh("git", "add", "-A")
        sh("git", "commit", "-qm", "base")
        sh("git", "branch", "basebr")
        sh("git", "checkout", "-qb", "topic")
        open(os.path.join(repo, "a.txt"), "w").write("a\nb\n")
        sh("git", "commit", "-qam", "topic: one line")

        def runner(text, sleep=0):
            open(stub, "w").write("#!/usr/bin/env bash\nout=$5\n" + (f"sleep {sleep}\n" if sleep else "")
                                  + f"cat > \"$out/review.md\" <<'EOF'\n{text}\nEOF\n"
                                  + "case \"$(head -1 \"$out/review.md\")\" in *MUST_FIX*) echo MUST_FIX ;; *) echo CLEAN ;; esac > \"$out/decision.txt\"\n")

        cwd = os.getcwd()
        os.chdir(repo)
        g = globals()
        saved = {k: g[k] for k in ("gh_json", "device_dirs", "read_ledger")}
        # no network and no cargo here: gh answers nothing, the device list and the ledger are fixed
        g["gh_json"] = lambda *a: None
        g["device_dirs"] = lambda base_tree, out, base_sha: open(out, "w").write("crates/gpu\n")
        def fake_ledger(base_tree, out):
            open(out, "w").close()
            return "v0"

        g["read_ledger"] = fake_ledger
        home = os.environ.get("HOME")
        os.environ["HOME"] = os.path.join(t, "home")
        os.environ["PRE_REVIEW_TEST_STUB"] = stub
        # origin/main at the base: the branch's base is not behind it. A later case moves it ahead.
        sh("git", "update-ref", "refs/remotes/origin/main", "basebr")
        try:
            def go(text, sleep=0, minutes=10):
                a = argparse.Namespace(base="basebr", full=False, minutes=minutes, body=None, no_pre_checks=False)
                msgs = []
                try:
                    runner(text, sleep)
                    return review(a, stubbed=True, say=msgs.append), msgs
                except Stop as e:
                    return e.code, msgs + [e.msg]

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc, _ = go("**Verdict: CLEAN**: fine\n\n### Checked\nall")
            expect(rc == 0, f"a CLEAN review ends 0 (got {rc})")
            expect("$" not in buf.getvalue(), f"accounting that is not available is not printed as a cost: {buf.getvalue()!r}")
            head_sha = git("rev-parse", "HEAD").strip()
            expect(f"pre-review: PR body: Pre-review CLEAN at {head_sha}." in buf.getvalue().splitlines(),
                   f"a CLEAN review prints one PR-body line with the full HEAD sha: {buf.getvalue()!r}")
            sh("git", "checkout", "-q", "main")
            open(os.path.join(repo, "m.txt"), "w").write("m\n")
            sh("git", "add", "m.txt")
            sh("git", "commit", "-qm", "main moves")
            sh("git", "update-ref", "refs/remotes/origin/main", "HEAD")
            sh("git", "checkout", "-q", "topic")
            rc, msgs = go("**Verdict: CLEAN**: should not run\n\n### Checked\nall")
            expect(rc == 64 and any("is behind origin/main" in m for m in msgs),
                   f"a base behind origin/main is refused by name before the review: {rc} {msgs}")
            sh("git", "update-ref", "refs/remotes/origin/main", "basebr")
            # A commit whose parent object is gone: rev-parse still names it, rev-list cannot walk it.
            tree = subprocess.run(["git", "rev-parse", "HEAD^{tree}"], cwd=repo, capture_output=True,
                                   check=True, text=True).stdout.strip()
            parent = subprocess.run(["git", "commit-tree", tree, "-m", "parent"], cwd=repo,
                                     capture_output=True, check=True, text=True).stdout.strip()
            child = subprocess.run(["git", "commit-tree", tree, "-p", parent, "-m", "child"], cwd=repo,
                                    capture_output=True, check=True, text=True).stdout.strip()
            os.remove(os.path.join(repo, ".git", "objects", parent[:2], parent[2:]))
            sh("git", "update-ref", "refs/remotes/origin/main", child)
            rc, msgs = go("**Verdict: CLEAN**: should not run\n\n### Checked\nall")
            expect(rc == 70 and any("rev-list" in m for m in msgs),
                   f"a failed ancestry walk stops 70: {rc} {msgs}")
            sh("git", "update-ref", "refs/remotes/origin/main", "basebr")
            keep = os.path.join(t, "home/.cache/bloomery/pre-review/topic-" + git("rev-parse", "HEAD").strip()[:8])
            ctx_files = set(os.listdir(os.path.join(keep, "ctx")))
            expect(ctx_files == set(CTX_FILES), f"the kept ctx holds the workflow's files: {sorted(ctx_files)}")
            pj = json.load(open(os.path.join(keep, "ctx/pr.json")))
            expect(tuple(pj) == PR_KEYS and pj["mode"] == "full" and pj["run"] == "local", f"pr.json: {pj}")
            expect(open(os.path.join(keep, "ctx/files.txt")).read() == "modified\t+1\t-0\ta.txt\n", "files.txt of the branch")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc, _ = go("**Verdict: MUST_FIX**: no\n\n### Must fix\n#### 1. Broken\n")
            expect(rc == 1 and "PR body:" not in buf.getvalue(),
                   f"a MUST_FIX review ends 1 and prints no PR-body line (got {rc}): {buf.getvalue()!r}")
            pidf = os.path.join(t, "child.pid")
            open(stub, "w").write(f"#!/usr/bin/env bash\n( sleep 30 & echo $! > {pidf}; wait )\n")
            a = argparse.Namespace(base="basebr", full=False, minutes=0.03, body=None, no_pre_checks=False)
            try:
                rc, msg = review(a, stubbed=True, say=lambda m: None), ""
            except Stop as e:
                rc, msg = e.code, e.msg
            time.sleep(0.2)
            child = int(open(pidf).read())
            alive = subprocess.run(["kill", "-0", str(child)], capture_output=True).returncode == 0
            kept = os.path.isdir(os.path.join(t, "home/.cache/bloomery/pre-review/topic-" + git("rev-parse", "HEAD").strip()[:8], "trace"))
            expect(kept, "a role stopped past --minutes still leaves its kept trace")
            expect(rc == 75 and "past" in msg and not alive,
                   f"a role past --minutes ends 75 named and its whole process group is gone: rc {rc}, child alive {alive}")
            # --full: a validator that ends 0 and leaves the review as the reviewer wrote it is no validation
            runner("**Verdict: CLEAN**: fine\n\n### Checked\nall")
            a = argparse.Namespace(base="basebr", full=True, minutes=10, body=None, no_pre_checks=False)
            try:
                rc, msg = review(a, stubbed=True, say=lambda m: None), ""
            except Stop as e:
                rc, msg = e.code, e.msg
            expect(rc == 65 and "validator ended 0 without its output" in msg, f"--full with a validator that wrote nothing ends 65: {rc} {msg}")
            for bad in ("nan", "inf", "0", "-1"):
                expect(main(["--minutes", bad]) == 64, f"--minutes {bad} is refused 64 before any role")
            rc, msgs = go("no verdict here")
            expect(rc in (65,) , f"a review with no verdict ends 65: {rc} {msgs[-1:]}")
            # SIGTERM to local.py itself stops the role's process group (run.launch's cleanup runs on the exception)
            open(stub, "w").write(f"#!/usr/bin/env bash\n( sleep 30 & echo $! > {pidf}; wait )\n")
            os.unlink(pidf)
            wrapper = (f"import shutil, sys\nsys.path.insert(0, {HERE!r})\nimport local as L\nreal = shutil.which\n"
                       "shutil.which = lambda n, *a: '/x' if n == 'claude' else real(n, *a)\n"
                       "L.claude_ok = lambda: None\n"
                       "L.gh_json = lambda *a: None\nL.device_dirs = lambda b, o, s: open(o, 'w').write('crates/gpu\\n')\n"
                       "L.read_ledger = lambda b, o: (open(o, 'w').close(), 'v0')[1]\n"
                       "sys.exit(L.main(['--base', 'basebr', '--minutes', '5']))\n")
            proc = subprocess.Popen([sys.executable, "-c", wrapper], cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    env={**os.environ, "HOME": t})
            for _ in range(200):
                if os.path.exists(pidf) and open(pidf).read().strip():
                    break
                time.sleep(0.1)
            child = int(open(pidf).read())
            proc.send_signal(signal.SIGTERM)
            try:
                proc.communicate(timeout=30)
                ended = True
            except subprocess.TimeoutExpired:
                proc.kill()
                ended = False
            time.sleep(0.2)
            alive = subprocess.run(["kill", "-0", str(child)], capture_output=True).returncode == 0
            expect(ended and not alive, f"SIGTERM to local.py ends it and the role's process group: ended {ended}, child alive {alive}")
            if alive:
                os.kill(child, signal.SIGKILL)
            # --no-pre-checks: a red text check on this head does not run, and the review still ends
            open(os.path.join(repo, "tools/check-levers.sh"), "w").write("echo 'check-levers: SOME_NAME is no row'; exit 3\n")
            sh("git", "add", "-A")
            sh("git", "commit", "-qm", "topic: a red levers check")
            a = argparse.Namespace(base="basebr", full=False, minutes=10, body=None, no_pre_checks=False)
            try:
                rc, msg = review(a, stubbed=True, say=lambda m: None), ""
            except Stop as e:
                rc, msg = e.code, e.msg
            expect(rc == 1 and "tools/check-levers.sh is red" in msg,
                   f"a red pre-check still stops the run when the flag is off: {rc} {msg}")
            runner("**Verdict: CLEAN**: fine\n\n### Checked\nall")
            a.no_pre_checks = True
            skipped = []
            try:
                rc = review(a, stubbed=True, say=skipped.append)
            except Stop as e:
                rc = e.code
            expect(rc == 0 and any("pre-checks skipped" in m for m in skipped),
                   f"--no-pre-checks skips a red pre-check and the review ends 0: {rc} {skipped}")
            # context seconds start after the pre-checks: the clock, not a sleep, places that boundary
            from types import SimpleNamespace
            from unittest.mock import patch

            open(os.path.join(repo, "tools/check-levers.sh"), "w").write("exit 0\n")
            sh("git", "add", "-A")
            sh("git", "commit", "-qm", "topic: a timing pre-check")
            runner("**Verdict: CLEAN**: fine\n\n### Checked\nall")
            buf = io.StringIO()
            a = argparse.Namespace(base="basebr", full=False, minutes=10,
                                   body=None, no_pre_checks=False)
            ticks = iter((0, 0, 10, 10, 11, 11))
            fake_time = SimpleNamespace(time=lambda: next(ticks))
            with patch.dict(g, time=fake_time), contextlib.redirect_stdout(buf):
                rc = review(a, stubbed=True, say=lambda m: None)
            ctx_s = re.search(r"context (\d+) s", buf.getvalue())
            expect(rc == 0 and ctx_s is not None and int(ctx_s.group(1)) == 1,
                   f"context time starts after pre-checks: {buf.getvalue()!r}")
        finally:
            os.chdir(cwd)
            g.update(saved)
            os.environ.pop("PRE_REVIEW_TEST_STUB", None)
            if home is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = home
    print(f"pre-review: self-test {'ok' if not fails else 'FAIL'} ({len(fails)} failed)")
    return 1 if fails else 0


def main(argv):
    if argv == ["--self-test"]:
        return self_test()
    ap = argparse.ArgumentParser(prog="tools/review/local.py", description="Shingler's reviewer on this machine, before a push")
    ap.add_argument("--base", default="origin/main")
    ap.add_argument("--full", action="store_true", help="the validator too, as the workflow runs it")
    ap.add_argument("--minutes", type=float, default=10, help="the bound per role (default 10)")
    ap.add_argument("--body", help="a file with the pull request body the reviewer reads (default: the open PR's, else none)")
    ap.add_argument("--no-pre-checks", action="store_true", help="skip the text-only checks before the reviewer")
    try:
        args = ap.parse_args(argv)
    except SystemExit as e:
        return 64 if e.code else 0
    if not math.isfinite(args.minutes) or args.minutes <= 0:
        print("pre-review: --minutes must be finite and positive", file=sys.stderr)
        return 64
    try:
        signal.signal(signal.SIGTERM, signal.default_int_handler)
        return review(args)
    except Stop as e:
        print(e.msg, file=sys.stderr)
        return e.code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
