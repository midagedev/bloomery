#!/usr/bin/env python3
"""Area and kind labels on every issue and pull request (.github/workflows/labels.yml).

    tools/review/labels.py run --repo OWNER/NAME [--number N | --state open|all] [--root DIR] [--dry-run]  (GH_TOKEN)
    tools/review/labels.py --self-test

A pull request's area labels come from the paths it changes, never from its tree: an open one's `pulls/<n>/files`, a
merged one's merge commit against its first parent (a closed pull request's file list runs from a base that may be
stale, so it can hold a whole stack). A merge only adds labels: a rebase merge has no merge commit, so its diff is its
last commit's and can leave out what the earlier ones changed. A pull request closed unmerged is skipped. An issue's
area labels come from the prefix of its title (`serve: …`, `tools, model: …`): each comma-separated word is a crate
directory (CRATE_AREA), an alias (ALIASES) or a path (path_labels). A prefix that names a concept rather than a place
(`paged`, `MTP`, `prefill`) gets no area. A crate CRATE_AREA lacks is `gpu` when its name starts with `gpu-` or it
reaches cuda-core, else it gets no area and the output names it. `dgx-spark` and `measure` come from the title of
either, and `dgx-spark` also from SPARK_PATHS; `bug` and `hardware` from an issue title in its template's form.
`new-model` is the template's alone: a `model:` title cannot tell a new architecture from work on crates/model.
`tracking` is defined and set by a person.

The card label follows Shingler (tools/review/post.py) on an open pull request whose file list is complete and whose
device crates are known (post.py device_dirs, tools/recipes.py pure-crates on the trusted tree). One that changes a
device crate gets `needs-gpu` when it has neither `gpu-evidence` nor `gpu-verified`, even after a person took it off:
a card check is recorded by `gpu-verified`. A `needs-gpu` this bot put on comes off when no device path is left or
`gpu-evidence` is on, as apply_labels does. `gpu-evidence` and `gpu-verified` are never touched, and nothing is
decided about a card label on a closed pull request, so the two converge in any order.

Every other label in MANAGED is added unless a person took it off (its last `unlabeled` event is not BOT's), and is
removed only when its last `labeled` event is BOT's, so a person's label stays. BOT is the login of the workflow's
GITHUB_TOKEN, which Shingler's jobs share; the script therefore writes only inside GitHub Actions and is a dry run
elsewhere. Each item's labels, title and events are read again after its files and device crates, right before the
writes. Every label in LABELS that the repository lacks is created; an existing label's colour and description are not
changed. --dry-run sends only GET requests and prints what it would change.
"""

import argparse
import io
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from post import CARD_LABELS, Refused, api, apply_labels, device_dirs  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BOT = "github-actions[bot]"
CRATE_AREA = {
    "app": "gpu", "gpu": "gpu", "gpu-deepseek41": "gpu", "gpu-gates": "gpu", "gpu-glm5next": "gpu",
    "gpu-mimo2": "gpu", "gpu-vision": "gpu",
    "gguf": "model", "hf": "model", "jinja": "model", "model": "model", "models": "model", "tokenizer": "model",
    "decision": "host", "engram": "host", "engram-lab": "host", "levers": "host", "qdot": "host", "refset": "host",
    "runtime": "host", "sampler": "host", "threads": "host", "vision": "host",
    "placement": "placement",
    "serve": "serve",
    "oxide-ice-unroll": "tools",  # a cuda-oxide reproducer outside the workspace
}
# Files a change updates beside its code (ratchets, ledgers, generated schemas, the lock): they name no area.
COMPANIONS = ("Cargo.lock", "THIRD_PARTY_NOTICES.md", "tools/unsafe-ratchet.txt", "tools/lint-ratchet.txt",
              "tools/levers-direct.txt", "tools/spawn-allow.txt", "tools/gate-paths.tsv", "tools/defaults-twins.tsv",
              "tools/bloomery/schema/", "tools/flow/plans/", "tools/ref/ptx-shapes.tsv")
TOOLS = ("tools/", ".github/", ".cargo/", ".config/", "fuzz/", "justfile", "Cargo.toml", "rust-toolchain.toml",
         "deny.toml", "clippy.toml")
SPARK_PATHS = ("docs/contrib/dgx-spark.md", "docs/research/sparkdec.md")
ALIASES = {
    "dgx-spark": "dgx-spark", "measure": "measure", "측정": "measure", "experiment": "measure",
    "bug": "bug", "hardware": "hardware",
    "docs": "documentation", "readme": "documentation", "contributing": "documentation",
    "tools": "tools", "justfile": "tools", "cargo": "tools", "ci": "tools", "review": "tools", "fuzz": "tools",
    "recipes": "tools", "github": "tools", "release": "tools", "box.sh": "tools",
}
PR_TITLE_KINDS = {"dgx-spark", "measure"}
LABELS = {
    "gpu": ("fbca04", "The card engine: crates/gpu*, crates/app"),
    "host": ("c2e0c6",
             "Host crates: qdot, threads, engram, engram-lab, runtime, sampler, levers, decision, vision, refset"),
    "model": ("d4c5f9", "The CPU engine and model files: crates/model, models, hf, gguf, tokenizer, jinja"),
    "serve": ("f9d0c4", "The HTTP server: crates/serve"),
    "placement": ("fef2c0", "Placement planning: crates/placement"),
    "tools": ("ededed", "Tools, recipes, CI and the toolchain: tools/, justfile, .github/, Cargo"),
    "documentation": ("0075ca", "Improvements or additions to documentation"),
    "dgx-spark": ("76b900", "The DGX Spark (aarch64, GB10) port"),
    "measure": ("006b75", "A measurement or an experiment"),
    "bug": ("d73a4a", "Something isn't working"),
    "hardware": ("0e8a16", "bloomery on a card, CPU or machine we have not run"),
    "tracking": ("0052cc", "A status issue kept open and updated as the work moves"),
    **CARD_LABELS,
}
MANAGED = set(CRATE_AREA.values()) | {"documentation"} | set(ALIASES.values())
COMPARE_CAP = 300  # the compare API lists at most this many files, on its first page
PAUSE = 1.0  # seconds between the writes of two items in a backfill, below GitHub's secondary rate limit
PREFIX_RE = re.compile(r"^\s*([^:]{1,40}):")


def crate_area(name, dirs=()):
    """A crate directory's area, or None."""
    if name in CRATE_AREA:
        return CRATE_AREA[name]
    return "gpu" if name.startswith("gpu-") or f"crates/{name}/" in dirs else None


def path_labels(path, dirs=()):
    """The labels one changed path names."""
    out = {"dgx-spark"} if path in SPARK_PATHS else set()
    if path.startswith("crates/"):
        area = crate_area(path.split("/")[1], dirs)
        out |= {area} if area else set()
    elif path.startswith(COMPANIONS) or path.startswith("docs/cards/") and path.endswith(".card"):
        pass
    elif path.startswith(".github/"):
        out.add("tools")
    elif path.startswith("docs/") or path.endswith(".md"):
        out.add("documentation")
    elif path.startswith(TOOLS):
        out.add("tools")
    return out


def title_labels(title):
    """(the labels a title's prefix names, the prefix words that name none)."""
    m = PREFIX_RE.match(title)
    out, unmapped = set(), []
    for word in (m.group(1).split(",") if m else []):
        w = re.sub(r"\s+(tests|테스트)$", "", re.sub(r"\(.*\)$", "", word.strip().lower()))
        hit = ALIASES.get(w) or crate_area(w)
        got = {hit} if hit else path_labels(w) if "/" in w or w.endswith(".md") else set()
        out |= got
        if not got:
            unmapped.append(w)
    return out, unmapped


def wanted(title, paths, is_pr, dirs=()):
    """(labels, unmapped words): a pull request by its paths and the title's kinds (and the crates that name no area),
    an issue by its title (and the prefix words that name none)."""
    kinds, unmapped = title_labels(title)
    if not is_pr:
        return kinds, unmapped
    crates = sorted({p.split("/")[1] for p in paths if p.count("/") > 1 and p.startswith("crates/")
                     and not crate_area(p.split("/")[1], dirs)})
    return set().union(*(path_labels(p, dirs) for p in paths)) | (kinds & PR_TITLE_KINDS), crates


def decide(have, want, events, card):
    """(add, remove, kept off) on one issue or pull request. card: whether an open pull request changes a device
    crate, or None when that is not known or the item is no open pull request (the card labels stay as they are)."""
    last = {}
    for e in events:
        if e.get("event") in ("labeled", "unlabeled") and e.get("label"):
            last[(e["event"], e["label"]["name"].lower())] = (e.get("actor") or {}).get("login")
    have = {h.lower() for h in have}
    off = {n for (ev, n), who in last.items() if ev == "unlabeled" and who != BOT}
    add = set(want) - have - off
    remove = {n for n in (MANAGED & have) - set(want) if last.get(("labeled", n)) == BOT}
    if card and not have & {"needs-gpu", "gpu-evidence", "gpu-verified"}:
        add.add("needs-gpu")
    if ("needs-gpu" in have and last.get(("labeled", "needs-gpu")) == BOT
            and (card is False or card and "gpu-evidence" in have)):
        remove.add("needs-gpu")
    return sorted(add), sorted(remove), sorted((set(want) - have) & off)


def pages(call, url, token):
    page = 1
    while True:
        got = call("GET", f"{url}{'&' if '?' in url else '?'}per_page=100&page={page}", token)
        yield from got
        if len(got) < 100:
            return
        page += 1


def pr_files(call, base, n, token):
    """(files, state, None or why the list may be short) of pull request n; files is None when it is skipped."""
    pr = call("GET", f"{base}/pulls/{n}", token)
    if pr["state"] == "open":
        files = list(pages(call, f"{base}/pulls/{n}/files", token))
        short = len(files) != pr["changed_files"]
        return files, "open", short and f"{len(files)} of {pr['changed_files']} files listed"
    if not pr.get("merged"):
        return None, "closed", None
    sha = pr["merge_commit_sha"]
    parent = call("GET", f"{base}/commits/{sha}", token)["parents"][0]["sha"]
    files = call("GET", f"{base}/compare/{parent}...{sha}", token)["files"]
    return files, "merged", len(files) >= COMPARE_CAP and f"the merge diff lists {COMPARE_CAP} files, its cap"


def label_exists(e):
    """Whether an HTTP 422 on a label's creation says it exists."""
    try:
        body = json.loads(e.read() or b"{}")
    except ValueError:
        return False
    return any(isinstance(x, dict) and x.get("code") == "already_exists" for x in body.get("errors") or [])


def run(repo, token, number=None, state="open", root=ROOT, dry=False, call=api, dirs=None):
    """Label one issue or pull request (number) or every one in state; returns the printed lines."""
    base = f"https://api.github.com/repos/{repo}"
    lines, n_add, n_rm, wrote = [], 0, 0, False

    def note(line):  # one line of plain text: a title cannot end the summary's code fence
        lines.append(" ".join(line.split()).replace("`", "'"))
    have_repo = {x["name"].lower() for x in pages(call, f"{base}/labels", token)}
    for name, (color, desc) in LABELS.items():
        if name.lower() in have_repo:
            continue
        note(f"create label {name}")
        if not dry:
            try:
                call("POST", f"{base}/labels", token, {"name": name, "color": color, "description": desc})
            except urllib.error.HTTPError as e:
                if e.code != 422 or not label_exists(e):
                    raise
    items = [call("GET", f"{base}/issues/{number}", token)] if number else pages(
        call, f"{base}/issues?state={state}&sort=created&direction=asc", token)
    for listed in items:
        n, is_pr = listed["number"], "pull_request" in listed
        files, kind, short, card = [], "issue", None, None
        if is_pr:
            files, kind, short = pr_files(call, base, n, token)
            if files is None:
                note(f"#{n} pr closed unmerged: skipped | {listed['title'][:90]}")
                continue
            if kind == "open" and not short:
                if dirs is None:
                    try:
                        dirs = device_dirs(root)
                    except Refused as e:
                        dirs = []
                        note(f"device crates unknown, no card label decided: {e}")
                card = any(f["filename"].startswith(tuple(dirs)) for f in files) if dirs else None
        # A rename's old path names an area too; the card label reads `filename` alone, as Shingler's files.txt does.
        paths = [f["filename"] for f in files] + [f["previous_filename"] for f in files if f.get("previous_filename")]
        events = list(pages(call, f"{base}/issues/{n}/events", token))
        it = call("GET", f"{base}/issues/{n}", token)  # the labels and title as they are now, after the slow reads
        want, unmapped = wanted(it["title"], paths, is_pr, dirs or ())
        have = [x["name"] for x in it.get("labels", [])]
        add, remove, kept = decide(have, want, events, card if it["state"] == "open" else None)
        held = short or (kind == "merged" and "merged")
        remove = [] if held else remove
        if (add or remove) and not dry:
            if wrote:
                time.sleep(PAUSE)
            wrote = True
            if add:
                call("POST", f"{base}/issues/{n}/labels", token, {"labels": add})
            for name in remove:
                try:
                    call("DELETE", f"{base}/issues/{n}/labels/{urllib.parse.quote(name)}", token)
                except urllib.error.HTTPError as e:
                    if e.code != 404:  # 404: someone took it off since the read
                        raise
        n_add, n_rm = n_add + len(add), n_rm + len(remove)
        note(f"#{n} {'pr ' + kind if is_pr else 'issue ' + it['state']}: "
             + (" ".join([f"+{a}" for a in add] + [f"-{r}" for r in remove]) or "unchanged")
             + f" | has {','.join(sorted(have)) or '-'}"
             + (f" | kept off by a person: {','.join(kept)}" if kept else "")
             + (f" | {'crate' if is_pr else 'prefix'} not mapped: {','.join(unmapped)}" if unmapped else "")
             + (f" | {held}: nothing removed, no card label decided" if held else "")
             + f" | {it['title'][:90]}")
    note(f"labels: {n_add} added, {n_rm} removed{' (dry run: nothing sent)' if dry else ''}")
    return lines


def self_test():
    crates = os.path.join(ROOT, "crates")
    stale = [d for d in CRATE_AREA if not os.path.exists(os.path.join(crates, d, "Cargo.toml"))]
    assert not stale, f"CRATE_AREA names crates that are gone: {stale}"
    assert MANAGED | {"tracking"} | set(CARD_LABELS) == set(LABELS), set(LABELS) ^ MANAGED
    for path, want in (("crates/gpu-gates/src/bin/x.rs", {"gpu"}), ("crates/engram-lab/Cargo.toml", {"host"}),
                       ("crates/gguf/RESULTS.md", {"model"}), ("crates/gpu-new/src/a.rs", {"gpu"}),
                       ("crates/newbie/src/a.rs", set()), ("docs/contrib/dgx-spark.md", {"documentation", "dgx-spark"}),
                       ("README.md", {"documentation"}), ("docs/cards/README.md", {"documentation"}),
                       ("docs/cards/b1-pp.card", set()), (".github/review/reviewer.md", {"tools"}),
                       (".config/nextest.toml", {"tools"}), ("Cargo.toml", {"tools"}), ("Cargo.lock", set()),
                       ("tools/unsafe-ratchet.txt", set()), ("tools/recipes.py", {"tools"}), ("LICENSE", set())):
        assert path_labels(path) == want, (path, path_labels(path))
    assert wanted("x", ["crates/newbie/a.rs", "crates/newdev/a.rs"], True, ["crates/newdev/"]) == ({"gpu"}, ["newbie"])
    for title, want, unmapped in (
            ("serve 테스트: mock::Hold", {"serve"}, []), ("tools, model: two reds", {"tools", "model"}, []),
            ("Model: MiniCPM-V", {"model"}, []), ("측정: Qwen3.8", {"measure"}, []),
            ("측정 러너: ik", set(), ["측정 러너"]), ("experiment: a lever", {"measure"}, []),
            ("gpu-gates: x", {"gpu"}, []), ("tools/ref: lcpp", {"tools"}, []), ("ci(review): x", {"tools"}, []),
            ("box.sh: x", {"tools"}, []), ("serve seats: x", set(), ["serve seats"]), ("paged: x", set(), ["paged"]),
            ("bug: x", {"bug"}, []), ("dgx-spark: P0", {"dgx-spark"}, []), ("no prefix here", set(), []),
            ("0.3.0 numbers for the README at serving defaults: x", set(), [])):
        assert title_labels(title) == (want, unmapped), (title, title_labels(title))
    assert wanted("dgx-spark: P0", ["crates/qdot/src/a.rs", "Cargo.lock"], True) == ({"host", "dgx-spark"}, [])

    def ev(event, name, who):
        return {"event": event, "label": {"name": name}, "actor": {"login": who}}
    mine, theirs = [ev("labeled", "needs-gpu", BOT)], [ev("labeled", "needs-gpu", "m")]
    # A person's removal holds and a person's label stays, except that needs-gpu follows the paths both ways.
    for have, want, events, card, out in (
            ([], {"tools"}, [ev("labeled", "tools", BOT), ev("unlabeled", "tools", "m")], None, ([], [], ["tools"])),
            (["tools", "serve"], {"gpu"}, [ev("labeled", "tools", BOT), ev("labeled", "serve", "m")], None,
             (["gpu"], ["tools"], [])),
            (["gpu-verified"], set(), [], True, ([], [], [])),
            ([], set(), mine + [ev("unlabeled", "needs-gpu", "m")], True, (["needs-gpu"], [], [])),
            (["needs-gpu"], set(), mine, False, ([], ["needs-gpu"], [])),  # Shingler skipped it; no device path left
            (["needs-gpu", "gpu-evidence"], set(), mine, True, ([], ["needs-gpu"], [])),
            (["needs-gpu", "gpu-evidence"], set(), mine, None, ([], [], [])),
            (["needs-gpu"], set(), theirs, False, ([], [], [])), (["needs-gpu"], set(), mine, None, ([], [], []))):
        assert decide(have, want, events, card) == out, (have, events, card, decide(have, want, events, card))

    def github(s):
        """One item, #5, as GitHub holds it. s["hooks"][path] runs once before that GET: another run in between."""
        def call(method, url, token, payload=None):
            path = url.split("/repos/o/r/")[1].split("?")[0]
            s.get("hooks", {}).pop(path, lambda: None)()
            item = {"number": 5, "title": s["title"], "state": s["pr"]["state"] if s["pr"] else "open",
                    "labels": [{"name": x} for x in s["labels"]], **({"pull_request": {}} if s["pr"] else {})}
            if method == "POST" and path == "issues/5/labels":
                s["labels"].update(payload["labels"])
                s["events"].extend(ev("labeled", x, BOT) for x in payload["labels"])
            elif method == "DELETE":
                name = urllib.parse.unquote(path.rsplit("/", 1)[1])
                s["labels"].discard(name)
                s["events"].append(ev("unlabeled", name, BOT))
            elif method == "POST":
                return {}
            elif path == "issues":
                return [dict(item, title=s.get("listed", s["title"]), labels=[])]
            return {"issues/5": item, "issues/5/labels": item["labels"], "issues/5/events": list(s["events"]),
                    "pulls/5": s["pr"], "pulls/5/files": s.get("files"), "commits/m": {"parents": [{"sha": "p"}]},
                    "compare/p...m": {"files": s.get("merged")}}.get(path, [])
        return call

    def pr(state="open", n=1, merged=False):
        return {"state": state, "changed_files": n, "merged": merged, "merge_commit_sha": "m"}
    gpu_file = {"filename": "crates/gpu/src/a.rs"}

    # With Shingler's apply_labels (L this script, B the bot): the same labels in any order, and when B runs between
    # L's first reads and its writes (R). A rename out of a device crate: Shingler sets no card label, nor may L.
    renamed = {"filename": "crates/runtime/src/a.rs", "previous_filename": "crates/gpu/src/a.rs"}
    for verdict, changed, final in (("gpu-evidence", gpu_file, {"gpu", "gpu-evidence"}),
                                    ("needs-gpu", gpu_file, {"gpu", "needs-gpu"}),
                                    (None, renamed, {"gpu", "host"})):
        for order in ("LB", "BL", "LBL", "LBLBL", "R"):
            s = {"title": "gpu: x", "labels": set(), "events": [], "pr": pr(), "files": [changed]}
            call = github(s)
            with tempfile.TemporaryDirectory() as out:
                with open(os.path.join(out, "labels.json"), "w") as f:
                    json.dump({"set": verdict}, f)
                s["hooks"] = {"pulls/5/files": lambda: apply_labels(out, "o/r", "5", "t", call)} if order == "R" else {}
                for step in order:
                    if step in "LR":
                        run("o/r", "t", 5, call=call, dirs=["crates/gpu/"])
                    else:
                        apply_labels(out, "o/r", "5", "t", call)
            assert s["labels"] == final, (verdict, order, s["labels"])

    # Each: the item, the run's device crates, the labels it ends with, why removal was held (None: not held).
    stale = {"needs-gpu", "serve"}
    for s, dirs, final, held in (
            ({"title": "tools: old", "pr": None}, None, {"tools"}, None),
            ({"listed": "tools: old", "title": "serve: new", "pr": None}, None, {"serve"}, None),  # edited in a backfill
            ({"title": "x", "pr": pr(), "files": [{"filename": "tools/a.sh"}]}, [], {"tools", "needs-gpu"}, None),
            ({"title": "x", "pr": pr(), "files": [{"filename": "tools/a.sh"}]}, None, {"tools"}, None),
            ({"title": "x", "pr": pr(n=2), "files": [gpu_file]}, None, {"gpu"} | stale, "1 of 2 files listed"),
            ({"title": "x", "pr": pr("closed", merged=True), "files": [gpu_file], "merged": [{"filename": "tools/a"}]},
             None, {"tools"} | stale, "merged"),
            ({"title": "x", "pr": pr("closed"), "files": [gpu_file]}, None, stale, None)):
        s.update(labels=set(stale) if s["pr"] else set(), events=[ev("labeled", x, BOT) for x in stale])
        lines = run("o/r", "t", None, "all", call=github(s), dirs=dirs if dirs is not None else ["crates/gpu/"])
        assert s["labels"] == final, (s, s["labels"])
        notes = [line for line in lines if "nothing removed" in line]
        assert len(notes) == bool(held) and all(f"| {held}: nothing removed" in x for x in notes), (held, lines)

    def refuse(code):
        def call(method, url, token, payload=None):
            if method == "POST":
                body = json.dumps({"errors": [{"resource": "Label", "code": code, "field": "name"}]}).encode()
                raise urllib.error.HTTPError(url, 422, "Validation Failed", {}, io.BytesIO(body))
            return []
        return call
    run("o/r", "t", call=refuse("already_exists"))
    try:
        run("o/r", "t", call=refuse("invalid"))
        raise AssertionError("a 422 other than already_exists passed")
    except urllib.error.HTTPError:
        pass
    sent = []
    run("o/r", "t", None, "all", dry=True, dirs=["crates/gpu/"],
        call=lambda m, u, t, p=None: sent.append(m) or ({"number": 1, "title": "serve: x", "state": "open",
                                                         "labels": []} if u.endswith("/issues/1") else
                                                        [{"number": 1}] if "/issues?" in u else []))
    assert set(sent) == {"GET"}, sent
    print("labels.py self-test: ok")


def main(argv):
    if argv[1:] == ["--self-test"]:
        self_test()
        return 0
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--number", type=int)
    ap.add_argument("--state", choices=["open", "all"], default="open")
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv[1:])
    token = os.environ.get("GH_TOKEN", "")
    if not token:
        print("labels: GH_TOKEN is not set", file=sys.stderr)
        return 64
    dry = a.dry_run or os.environ.get("GITHUB_ACTIONS") != "true"
    if dry and not a.dry_run:
        print(f"labels: a dry run outside GitHub Actions, where the labels this writes would not be {BOT}'s")
    for line in run(a.repo, token, a.number, a.state, a.root, dry):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
