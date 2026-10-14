#!/usr/bin/env python3
"""Shingler's reviewer on this machine, before a push: the PR review's own instructions, catalog and ledger, run through
the base tree's tools/review/run-claude.sh under the caller's own Claude login (`just pre-review`).

    tools/review/local.py [--base REF] [--full] [--minutes N] [--body FILE]
    tools/review/local.py --self-test

What it reviews is the committed HEAD, what a push sends; a dirty tree is named and left out. The trees come from
`git archive`, BASE_TREE at REF (default origin/main, fetched first) and PR_TREE at HEAD, so no checkout hook adds a
file to either. CTX holds what .github/workflows/review.yml's "Build the review context" step writes, built from git
instead of the pull request API: diff.patch (BASE...HEAD), files.txt (the PR's files, its format), pr.json (the same
keys; number 0 and the head's subject when the branch has no open pull request), open-issues.txt and ledger.md (read
with gh; an empty file and a named line when gh cannot), device-dirs.txt (post.py device), diagnostics.md (dryer and
crapper when tools/review/diagnostics.py finds them where the workflow installs them, else one line saying they did not
run). When the branch's pull request already has a Shingler review of an ancestor of HEAD, the run is incremental as the
workflow's is: the diff since that head, and that review as prev-review.md.

By default only the reviewer runs: the validator only removes findings, and here a false one costs a read, not a
round trip. --full runs the validator after it, as the workflow does. Then `post.py check` reads the output as it
does before posting. Nothing is posted and the workflow is not touched; review.yml stays the review of record.

Prints the verdict, the must-fix titles and where the output is (~/.cache/bloomery/pre-review/<branch>-<sha8>/).
Exit: 0 CLEAN; 1 MUST_FIX; 75 no claude CLI, or a role past --minutes (default 10); 64 usage; 65 output post.py refused
or a review with no verdict; 70 a context step that failed (named).
The failure it blocks: a pull request whose first Shingler review is a MUST_FIX, each fold another 10-27 minute round.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from post import VERDICT_RE  # noqa: E402
from run import MINUTES, TIMEOUT, launch  # noqa: E402

REPO = "midagedev/bloomery"
NET_S = 60  # each gh or git fetch call: past it the step reads as unanswered, never as a hang
MARKER_RE = re.compile(r"<!-- bloomery-review head_sha=([0-9a-f]{40})")
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


def git(*args, cwd=None):
    p = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if p.returncode:
        raise Stop(70, f"pre-review: git {' '.join(args)} failed: {p.stderr.strip()[:200]}")
    return p.stdout


def gh_json(*args):
    """gh's JSON answer, or None when gh cannot answer (no gh, no login, no network)."""
    if not shutil.which("gh"):
        return None
    try:
        p = subprocess.run(["gh", *args], capture_output=True, text=True, timeout=NET_S)
    except subprocess.TimeoutExpired:
        return None
    if p.returncode:
        return None
    try:
        return json.loads(p.stdout)
    except ValueError:
        return None


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
    device_dirs(base_tree, os.path.join(ctx, "device-dirs.txt"))
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


def device_dirs(base_tree, out):
    p = subprocess.run([sys.executable, os.path.join(base_tree, "tools/review/post.py"), "device", "--root", base_tree,
                        "--out", out], capture_output=True, text=True)
    if p.returncode:
        raise Stop(70, f"pre-review: post.py device failed: {p.stderr.strip()[:200]}")


def read_ledger(base_tree, out):
    """The ledger's version (ledger.py prints it), the lessons written to out; None when gh cannot read it. ledger.py
    reads GH_TOKEN, which a plain `gh auth login` does not set: it comes from `gh auth token` then, into the child's
    environment only."""
    env = dict(os.environ)
    if not env.get("GH_TOKEN") and shutil.which("gh"):
        t = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=NET_S)
        if t.returncode == 0 and t.stdout.strip():
            env["GH_TOKEN"] = t.stdout.strip()
    try:
        p = subprocess.run([sys.executable, os.path.join(base_tree, "tools/review/ledger.py"), "accepted", "--repo", REPO,
                            "--out", out], capture_output=True, text=True, env=env, timeout=NET_S)
    except subprocess.TimeoutExpired:
        return None
    return p.stdout.strip() if p.returncode == 0 else None


def diag(ctx, base_tree, base_sha, head, say):
    ub = os.path.expanduser("~/.cache/bloomery-review/ub/venv/bin")
    dryer, crapper = os.path.join(ub, "dryer"), os.path.join(ub, "crapper")
    out = os.path.join(ctx, "diagnostics.md")
    if not (os.access(dryer, os.X_OK) and os.access(crapper, os.X_OK)):
        with open(out, "w") as f:
            f.write(NO_DIAG + "\n")
        return
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
    past the bound."""
    MINUTES[role] = minutes
    paths = [os.path.join(d, x) for x in ("base", "pr", "ctx", "out", "trace")]
    rc = launch("claude", role, paths, dict(os.environ, REVIEW_SETTING_SOURCES="project"))
    log = os.path.join(d, "trace", f"{role}.log")
    if os.path.exists(log):
        for line in open(log, encoding="utf-8", errors="replace"):
            if line.startswith(f"{role}: "):
                say(line.rstrip("\n"))
    return rc


def must_fix_titles(body):
    """The must-fix items' titles (`#### <n>. <title>` under `### Must fix`)."""
    m = re.search(r"^### Must fix\s*$(.*?)(?=^### |\Z)", body, re.M | re.S)
    return re.findall(r"^#### (.+)$", m.group(1), re.M) if m else []


def unpack(sha, dest):
    os.makedirs(dest)
    a = subprocess.Popen(["git", "archive", sha], stdout=subprocess.PIPE)
    t = subprocess.run(["tar", "-x", "-C", dest], stdin=a.stdout)
    a.stdout.close()
    if a.wait() or t.returncode:
        raise Stop(70, f"pre-review: git archive {sha[:12]} could not be unpacked")


def review(args, stubbed=False, say=lambda m: print(m, file=sys.stderr)):
    if not stubbed and not shutil.which("claude"):
        raise Stop(75, "pre-review: no claude CLI on PATH; nothing was reviewed")
    try:
        fetched = subprocess.run(["git", "fetch", "-q", "origin"], capture_output=True, timeout=NET_S).returncode == 0
    except subprocess.TimeoutExpired:
        fetched = False
    if not fetched:
        say(f"pre-review: git fetch origin failed or took over {NET_S} s; {args.base} is as this clone last saw it")
    head = git("rev-parse", "HEAD").strip()
    base_sha = git("rev-parse", args.base).strip()
    if git("status", "--porcelain", "--untracked-files=no").strip():
        say("pre-review: the tree has uncommitted changes; what is reviewed is the committed HEAD, what a push sends")
    branch = re.sub(r"[^A-Za-z0-9._-]", "_", git("rev-parse", "--abbrev-ref", "HEAD").strip())
    keep = os.path.expanduser(f"~/.cache/bloomery/pre-review/{branch}-{head[:8]}")
    shutil.rmtree(keep, ignore_errors=True)
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="pre-review.") as d:
        unpack(base_sha, os.path.join(d, "base"))
        unpack(head, os.path.join(d, "pr"))
        for x in ("ctx", "out", "trace"):
            os.makedirs(os.path.join(d, x))
        try:
            meta = build_ctx(os.path.join(d, "ctx"), os.path.join(d, "base"), base_sha, head, args, say)
            t_ctx = time.time() - t0
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
    cost = 0.0
    for r in roles:
        try:
            cost += json.load(open(os.path.join(keep, "trace", f"{r}.usage.json"))).get("cost_usd") or 0.0
        except (OSError, ValueError):
            pass
    titles = must_fix_titles(body)
    print(f"pre-review: {m.group(1)} — {meta['mode']} review of {head[:8]} against {args.base} "
          f"({'reviewer + validator' if args.full else 'reviewer'}, {time.time() - t0:.0f} s of which context {t_ctx:.0f} s, ${cost:.2f})")
    for t in titles[:12]:
        print(f"  must fix: {t}")
    print(f"  review: {keep}/out/review.md")
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
        g["device_dirs"] = lambda base_tree, out: open(out, "w").write("crates/gpu\n")
        def fake_ledger(base_tree, out):
            open(out, "w").close()
            return "v0"

        g["read_ledger"] = fake_ledger
        home = os.environ.get("HOME")
        os.environ["HOME"] = os.path.join(t, "home")
        os.environ["PRE_REVIEW_TEST_STUB"] = stub
        try:
            def go(text, sleep=0, minutes=10):
                a = argparse.Namespace(base="basebr", full=False, minutes=minutes, body=None)
                msgs = []
                try:
                    runner(text, sleep)
                    return review(a, stubbed=True, say=msgs.append), msgs
                except Stop as e:
                    return e.code, msgs + [e.msg]

            rc, _ = go("**Verdict: CLEAN**: fine\n\n### Checked\nall")
            expect(rc == 0, f"a CLEAN review ends 0 (got {rc})")
            keep = os.path.join(t, "home/.cache/bloomery/pre-review/topic-" + git("rev-parse", "HEAD").strip()[:8])
            ctx_files = set(os.listdir(os.path.join(keep, "ctx")))
            expect(ctx_files == set(CTX_FILES), f"the kept ctx holds the workflow's files: {sorted(ctx_files)}")
            pj = json.load(open(os.path.join(keep, "ctx/pr.json")))
            expect(tuple(pj) == PR_KEYS and pj["mode"] == "full" and pj["run"] == "local", f"pr.json: {pj}")
            expect(open(os.path.join(keep, "ctx/files.txt")).read() == "modified\t+1\t-0\ta.txt\n", "files.txt of the branch")
            rc, _ = go("**Verdict: MUST_FIX**: no\n\n### Must fix\n#### 1. Broken\n")
            expect(rc == 1, f"a MUST_FIX review ends 1 (got {rc})")
            pidf = os.path.join(t, "child.pid")
            open(stub, "w").write(f"#!/usr/bin/env bash\n( sleep 30 & echo $! > {pidf}; wait )\n")
            a = argparse.Namespace(base="basebr", full=False, minutes=0.03, body=None)
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
            rc, msgs = go("no verdict here")
            expect(rc in (65,) , f"a review with no verdict ends 65: {rc} {msgs[-1:]}")
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
    try:
        args = ap.parse_args(argv)
    except SystemExit as e:
        return 64 if e.code else 0
    try:
        return review(args)
    except Stop as e:
        print(e.msg, file=sys.stderr)
        return e.code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
