#!/usr/bin/env python3
"""board.py — the status board: one Markdown body for one pinned GitHub issue, rewritten by a script, no LLM.

  board.py                       print the body (what --publish would send)
  board.py --publish ISSUE       replace that issue's body (gh issue edit ISSUE --body-file -)
  board.py --self-test           the renderer, the privacy filter and the publish path on synthetic state, no network

The body opens with a piece of ASCII art that follows the board's state, then the sections, each with its own "as of":
Release, Trains, Batches, Workers, Contributor PRs and issues, Recently landed. The state files it reads (formats in
tools/board/README.md) live under --dir, default ~/.cache/bloomery/board: release.json and trains/*.json (the lead),
batch/*.jsonl (tools/gate-batch.sh), workers/<role>.md (the sessions). The PR and issue lists come from `gh`, the landed
list from `git log origin/main` in --tree (the checkout holding this file).

A missing state directory is an empty section. A source that fails (gh, git, an unreadable or malformed state file) is
a named line in its section and a non-zero exit after the rest is rendered and published: the board shows its own gaps.
The body goes through the privacy filter (PRIVACY) line by line before it is printed or published, the same path for
both, and ends with the count of lines it withheld.

Exit: 0 ok; 1 a source failed, the body is over GitHub's size limit, or the publish failed; 64 usage.
"""

import argparse
import dataclasses
import datetime as dt
import io
import json
import os
import re
import subprocess
import sys
import tempfile

STATES = ("building", "static", "batch", "ab", "landing", "landed")
QUIET_S = 3 * 3600  # a worker file or a batch that has been quiet this long is shown as such
BLOOM_S = 3600  # a train landed this recently puts the header art in bloom
RELEASE_STATES = ("collecting", "union-batch", "user-pass", "tagged")
RELEASE_GROUPS = (  # item state -> the group's heading, in the order shown
    ("landed", "Landed"),
    ("in-train", "In a train now"),
    ("pending", "Pending (with ETA)"),
    ("deferred", "Deferred to the next release"),
)
DROP_S = 24 * 3600  # a landed train and an ended or quiet batch drop off after a day
COMMENT_DAYS = 7  # an issue is listed while its last non-maintainer comment is this recent
WORKER_LINES = 6
SUBJECT_MAX = 100
LOG_N = 10
LIST_MAX = {"prs": 40, "issues": 20, "red": 15, "batches": 10, "pieces": 30}
BODY_MAX = 60000  # GitHub's limit is 65536 characters
DEFAULT_MAINTAINERS = ("midagedev",)
GH_TIMEOUT_S = 60
GH_LIMIT = 100  # gh's page for each list; a list that fills it may be cut

# The privacy filter: a rendered line matching any of these is dropped and counted. Blocking: the body is never
# printed or published unfiltered. Addresses of the workstation's network, host names, home paths, tokens.
PRIVACY = (
    ("lan", r"192\.168\."),
    ("tailnet-ip", r"\b100\.\d+\.\d+\.\d+"),
    ("tailnet-name", r"\.ts\.net"),
    ("admin", r"\badmin\b"),
    ("host-sihyung", r"sihyung"),
    ("host-dopc", r"dopc"),
    ("mac-home", r"/Users/[^ ]+"),
    ("root-path", r"/root/"),
    ("linux-home", r"/home/[^ ]+"),
    ("hf-token", r"hf_[A-Za-z0-9]{10,}"),
    ("gh-token", r"ghp_"),
    ("token-assign", r"(?i)\btoken\b\s*[:=]"),
    ("bearer", r"(?i)bearer\s+\S"),
)
PRIVACY_RES = tuple((name, re.compile(rx)) for name, rx in PRIVACY)


def withhold(lines, patterns=PRIVACY_RES):
    """(the lines no pattern matches, [(the first pattern's name, the line)] for each line dropped)."""
    kept, dropped = [], []
    for ln in lines:
        hit = next((name for name, rx in patterns if rx.search(ln)), None)
        (dropped.append((hit, ln)) if hit else kept.append(ln))
    return kept, dropped


# ----------------------------------------------------------------------------------------------
# the header art
# ----------------------------------------------------------------------------------------------
# A bloomery is an iron-smelting furnace, and a bloom is also a flower. The first three pieces follow the board's state
# (mood()); idle days rotate through IDLE by date. Plain ASCII, at most ART_ROWS lines and ART_COLS columns, no backtick
# (it sits in a fenced block), so it lines up on a phone.
ART_COLS, ART_ROWS = 56, 10
PIECES = {name: tuple(art.strip("\n").split("\n")) for name, art in {
    "furnace": r"""
        *    .    '     *    .
    .        \  .  |  '  /       *
      '     *  \  \ | /  /    .
                 .-^-^-.
                /  ) (  \       ___
               |  (   )  |=====<   >
               |   ) (   |      \_/
               |  .---.  |
              _|_/_____\_|_
             [_____________]
""",
    "bloom": r"""
         _
       _(_)_          wWWWw
      (_)@(_)         (___)
        (_)\    _       |
           \ _(_)_      |
            \(_)@(_)    |
              (_)/      |
        ________|______/____
       ~~~~~~~~~~~~~~~~~~~~~~
""",
    "rain": r"""
            .--.
         .-(    ).
        (___.__)__)
         ' ' ' ' '
          ' ' ' '
            \ | /
             \|/
              |
        ~~~~~~~~~~~~~~
""",
    "anvil": r"""
               ___________
          ____/___________\
         '-.__           _/
              |  |||||  |
              |  |||||  |
             _|__|||||__|_
            |_____________|
          ====================
""",
    "bellows": r"""
            ______
          .'      '.
         /  ~~  ~~  \=======
        |   ~~~~~~   |-----==>
         \  ~~  ~~  /=======
          '.______.'
           //    \\
          //      \\
""",
    "lump": r"""
              .  *  .
          *  .-~~~~~-.  .
           ( @ @@ @@ @ )
         .  \ @@ @ @@ /  *
             '-.___.-'
          ~~~~~~~~~~~~~~~~~
            a bloom of iron
""",
    "sleepy": r"""
                       z
                    z
            .-"-.    Z
           |  _  |
           | (_) |    _.._
           |     |   (    )
         __|_____|__  '--'
        [___________]
""",
}.items()}
IDLE = ("anvil", "bellows", "lump", "sleepy")


# ----------------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------------
def one_line(s):
    return " ".join(str(s).split())


def cut(s, n):
    s = one_line(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def ascii_cut(s, n):
    """One line of plain ASCII (anything else becomes `?`) of at most n columns."""
    s = one_line(s).encode("ascii", "replace").decode()
    return s if len(s) <= n else s[: n - 3] + "..."


def code(s):
    """A code span nothing inside can close or turn into a mention, a link or markup (titles come from outsiders)."""
    return "`" + one_line(s).replace("`", "'") + "`"


def local(t):
    return t.astimezone()


def clock(t, now):
    t, now = local(t), local(now)
    return t.strftime("%H:%M") if t.date() == now.date() else t.strftime("%m-%d %H:%M")


def parse_ts(s):
    t = dt.datetime.fromisoformat(s)
    return t if t.tzinfo else t.astimezone()


def mtime(path):
    return dt.datetime.fromtimestamp(os.stat(path).st_mtime).astimezone()


def run(argv, cwd=None, stdin=None, env=None):
    """(rc, stdout, stderr); a missing program and a timeout are rc 127 and 124 with the reason on stderr."""
    try:
        p = subprocess.run(argv, cwd=cwd, input=stdin, capture_output=True, text=True, timeout=GH_TIMEOUT_S,
                           env={**os.environ, **env} if env else None)
    except FileNotFoundError:
        return 127, "", f"{argv[0]}: not found"
    except subprocess.TimeoutExpired:
        return 124, "", f"{argv[0]}: no answer in {GH_TIMEOUT_S} s"
    return p.returncode, p.stdout, p.stderr


def files(dirpath, suffix):
    try:
        return sorted(os.path.join(dirpath, n) for n in os.listdir(dirpath) if n.endswith(suffix))
    except FileNotFoundError:
        return []


# ----------------------------------------------------------------------------------------------
# the state files
# ----------------------------------------------------------------------------------------------
@dataclasses.dataclass
class Train:
    name: str
    branch: str
    base: str
    pieces: list
    state: str
    note: str
    at: dt.datetime  # the file's mtime: when the lead last wrote it


def read_trains(root, now):
    """(trains, problems): the lead's trains/*.json; a landed train drops off a day after its last write."""
    trains, problems = [], []
    for path in files(os.path.join(root, "trains"), ".json"):
        base = os.path.basename(path)
        try:
            with open(path, encoding="utf-8") as fh:
                j = json.load(fh)
            if not isinstance(j, dict):
                raise ValueError("not a JSON object")
            for k in ("name", "state", "pieces"):
                if k not in j:
                    raise ValueError(f"no \"{k}\"")
            if j["state"] not in STATES:
                raise ValueError(f"state \"{j['state']}\" is none of {'|'.join(STATES)}")
            if not isinstance(j["pieces"], list) or not all(isinstance(p, dict) and "what" in p for p in j["pieces"]):
                raise ValueError("pieces is not a list of {what, owner, pr}")
            for p in j["pieces"]:
                if p.get("pr") is not None and not isinstance(p["pr"], int):
                    raise ValueError("a piece's pr is neither a number nor null")
            tr = Train(
                str(j["name"]), str(j.get("branch", "")), str(j.get("base", "")), j["pieces"], j["state"],
                str(j.get("note", "")), mtime(path),
            )
        except (OSError, ValueError) as e:  # json.JSONDecodeError is a ValueError
            problems.append(f"trains/{base}: {one_line(e)}")
            continue
        if tr.state == "landed" and (now - tr.at).total_seconds() > DROP_S:
            continue
        trains.append(tr)
    trains.sort(key=lambda t: t.at, reverse=True)
    return trains, problems


@dataclasses.dataclass
class Batch:
    name: str  # the log directory's name, the jsonl's stem
    start: dict
    items: list
    done: dict
    stop: dict
    last: dt.datetime  # the newest event's time


def read_batches(root, now):
    """(batches, problems): the newest run in each batch/*.jsonl. A file may hold several runs (one log directory name
    reused); the last `start` opens the one shown. Runs quiet for a day are dropped."""
    out, problems = [], []
    for path in files(os.path.join(root, "batch"), ".jsonl"):
        name = os.path.basename(path)[: -len(".jsonl")]
        start, items, done, stop, last, bad = {}, [], {}, {}, None, 0
        try:
            with open(path, encoding="utf-8") as fh:
                for raw in fh:
                    if not raw.strip():
                        continue
                    try:
                        ev = json.loads(raw)
                        t = parse_ts(ev["ts"])
                        kind = ev["ev"]
                    except (ValueError, KeyError, TypeError):
                        bad += 1
                        continue
                    last = t
                    if kind == "start":
                        start, items, done, stop = ev, [], {}, {}
                    elif kind == "item":
                        items.append(ev)
                    elif kind == "done":
                        done = ev
                    elif kind == "stop":
                        stop = ev
        except OSError as e:
            problems.append(f"batch/{name}.jsonl: {one_line(e)}")
            continue
        if bad:
            problems.append(f"batch/{name}.jsonl: {bad} unreadable event line(s)")
        if last is None or (now - last).total_seconds() > DROP_S:
            continue
        out.append(Batch(name, start, items, done, stop, last))
    out.sort(key=lambda b: b.last, reverse=True)
    return out, problems


@dataclasses.dataclass
class Release:
    version: str
    cut: str
    state: str
    items: list
    at: dt.datetime  # the file's mtime
    start: int  # the cut window's start as minutes of the day, or None when the text has no window


CUT_WINDOW = re.compile(r"(\d{1,2}):(\d{2})\s*[–—-]\s*\d{1,2}:\d{2}")


def minutes(hhmm):
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", str(hhmm))
    if not m or int(m[1]) > 24 or int(m[2]) > 59:
        raise ValueError(f"\"{hhmm}\" is not HH:MM")
    return int(m[1]) * 60 + int(m[2])


def read_release(root):
    """(release or None, problems): release.json, the lead's plan for the next release. No file is no release."""
    path = os.path.join(root, "release.json")
    if not os.path.exists(path):
        return None, []
    try:
        with open(path, encoding="utf-8") as fh:
            j = json.load(fh)
        if not isinstance(j, dict):
            raise ValueError("not a JSON object")
        for k in ("version", "cut", "state", "items"):
            if k not in j:
                raise ValueError(f"no \"{k}\"")
        if j["state"] not in RELEASE_STATES:
            raise ValueError(f"state \"{j['state']}\" is none of {'|'.join(RELEASE_STATES)}")
        if not isinstance(j["items"], list):
            raise ValueError("items is not a list")
        states = [g for g, _ in RELEASE_GROUPS]
        for n, it in enumerate(j["items"], 1):
            if not isinstance(it, dict) or "what" not in it:
                raise ValueError(f"item {n} is not an object with \"what\"")
            if it.get("state") not in states:
                raise ValueError(f"item {n}: state \"{it.get('state')}\" is none of {'|'.join(states)}")
            if it.get("eta") is not None:
                minutes(it["eta"])
        m = CUT_WINDOW.search(str(j["cut"]))
        if not m:
            raise ValueError(f"cut \"{j['cut']}\" holds no HH:MM–HH:MM window")
        return Release(str(j["version"]), str(j["cut"]), j["state"], j["items"], mtime(path), minutes(f"{m[1]}:{m[2]}")), []
    except (OSError, ValueError) as e:
        return None, [f"release.json: {one_line(e)}"]


def late(rel, it):
    """An item still on its way (in a train or pending) whose ETA is later in the day than the cut window's start."""
    return it["state"] in ("in-train", "pending") and it.get("eta") is not None and minutes(it["eta"]) > rel.start


def overdue(it, now):
    """An item still on its way (in a train or pending) whose ETA, a time of today, is already behind the clock."""
    n = local(now)
    return it["state"] in ("in-train", "pending") and it.get("eta") is not None and minutes(it["eta"]) < n.hour * 60 + n.minute


def read_workers(root, now):
    """([(role, mtime, stale, lines, lines cut)], problems): workers/<role>.md, at most WORKER_LINES lines each."""
    out, problems = [], []
    for path in files(os.path.join(root, "workers"), ".md"):
        role = os.path.basename(path)[: -len(".md")]
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                lines = [ln.rstrip() for ln in fh.read().splitlines() if ln.strip()]
            at = mtime(path)
        except OSError as e:
            problems.append(f"workers/{role}.md: {one_line(e)}")
            continue
        out.append((role, at, (now - at).total_seconds() > QUIET_S, lines[:WORKER_LINES], max(0, len(lines) - WORKER_LINES)))
    return out, problems


# ----------------------------------------------------------------------------------------------
# gh and git
# ----------------------------------------------------------------------------------------------
def repo_slug(tree):
    """OWNER/NAME from the tree's origin remote, or None."""
    rc, out, _ = run(["git", "-C", tree, "remote", "get-url", "origin"])
    m = re.search(r"github\.com[:/]([^/\s]+/[^/\s]+?)(?:\.git)?\s*$", out) if rc == 0 else None
    return m.group(1) if m else None


def gh_json(gh, repo, args):
    """(parsed JSON, None) or (None, the one-line reason)."""
    rc, out, err = run([gh, *args, "--repo", repo])
    if rc != 0:
        print(f"board.py: gh {' '.join(args[:2])}: {one_line(err)[:200]}", file=sys.stderr)  # the log, never the body
        return None, f"gh {' '.join(args[:2])} failed (exit {rc})"
    try:
        return json.loads(out), None
    except ValueError:
        return None, f"gh {' '.join(args[:2])} printed no JSON"


def login(user):
    return (user or {}).get("login", "")


def contributor_issues(issues, maintainers, now, skip):
    """[(issue, the newest comment of a non-maintainer, non-bot author within COMMENT_DAYS)], most recent first."""
    out = []
    for it in issues:
        if it.get("number") == skip:
            continue
        newest = None
        for c in it.get("comments") or []:
            who = c.get("author") or {}
            if who.get("is_bot") or who.get("login") in maintainers or not who.get("login"):
                continue
            t = parse_ts(c["createdAt"].replace("Z", "+00:00"))
            if (now - t).total_seconds() <= COMMENT_DAYS * 86400 and (newest is None or t > newest[0]):
                newest = (t, who["login"])
        if newest:
            out.append((it, newest))
    out.sort(key=lambda p: p[1][0], reverse=True)
    return out


def git_fetch(tree):
    """None, or the reason: origin/main brought up to date. Best effort — a failure leaves the log as the last fetch had it."""
    rc, _, err = run(["git", "-C", tree, "fetch", "-q", "origin", "main"], env={"GIT_TERMINAL_PROMPT": "0"})
    if rc == 0:
        return None
    print(f"board.py: git fetch: {one_line(err)[:200]}", file=sys.stderr)  # the log, never the body
    return f"git fetch origin main failed (exit {rc}): the list is as of the last fetch that worked"


def git_log(tree):
    """([(hash, committed at, subject)], None) or (None, the reason): origin/main as this checkout has it."""
    rc, out, _ = run(["git", "-C", tree, "log", "origin/main", f"-{LOG_N}", "--format=%h%x09%cI%x09%s"])
    if rc != 0:
        return None, f"git log origin/main failed (exit {rc})"
    rows = [ln.split("\t", 2) for ln in out.splitlines() if ln.strip()]
    return [(h, parse_ts(t), s) for h, t, s in rows], None


# ----------------------------------------------------------------------------------------------
# one collected board
# ----------------------------------------------------------------------------------------------
@dataclasses.dataclass
class Board:
    trains: list
    batches: list
    workers: list
    release: Release  # None when there is no release.json (or it is unreadable: a problem)
    prs: list  # None when unavailable
    issues: list  # None when unavailable
    log: list  # None when unavailable
    problems: dict  # source -> [what failed]: shown in its section, and exit 1
    gh_off: bool = False


def collect(a, now):
    trains, p1 = read_trains(a.dir, now)
    batches, p2 = read_batches(a.dir, now)
    workers, p3 = read_workers(a.dir, now)
    release, p4 = read_release(a.dir)
    problems = {"trains": p1, "batches": p2, "workers": p3, "release": p4, "gh": [], "log": []}
    prs = issues = None
    if not a.no_gh:
        repo = a.repo or repo_slug(a.tree)
        if repo is None:
            problems["gh"].append("no OWNER/NAME: the tree has no github origin remote and --repo is not given")
        else:
            prs, e1 = gh_json(a.gh, repo, ["pr", "list", "--state", "open", "--limit", str(GH_LIMIT), "--json",
                                           "number,title,author,url,isDraft,updatedAt,reviewDecision"])
            issues, e2 = gh_json(a.gh, repo, ["issue", "list", "--state", "open", "--limit", str(GH_LIMIT), "--json",
                                              "number,title,author,url,updatedAt,comments"])
            problems["gh"] += [e for e in (e1, e2) if e]
            problems["gh"] += [f"gh {what} list returned {GH_LIMIT} rows, its limit: older ones are not shown"
                               for what, rows in (("pr", prs), ("issue", issues)) if rows is not None and len(rows) >= GH_LIMIT]
            if issues is not None:
                issues = contributor_issues(issues, a.maintainer, now, int(a.board_issue) if a.board_issue else None)
    e = git_fetch(a.tree)
    if e:
        problems["log"].append(e)
    log, e = git_log(a.tree)
    if e:
        problems["log"].append(e)
    return Board(trains, batches, workers, release, prs, issues, log, problems, a.no_gh)


# ----------------------------------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------------------------------
def batch_counts(b):
    """(done, green, red [(name, rc, lane, secs)], other {skip|deferred|yield: n})."""
    green, red, other = 0, [], {}
    for it in b.items:
        rc = it.get("rc")
        if rc == 0:
            green += 1
        elif isinstance(rc, int):
            red.append((it.get("name", "?"), rc, it.get("lane", "?"), it.get("secs", 0)))
        else:
            other[str(rc)] = other.get(str(rc), 0) + 1
    return len(b.items), green, red, other


def batch_status(b, now):
    if b.done:
        return "ended"
    if b.stop:
        return f"stopped (signal) at {clock(parse_ts(b.stop['ts']), now)}"
    if (now - b.last).total_seconds() > QUIET_S:
        return f"no DONE, last event {clock(b.last, now)}"
    return "running"


def batch_progress(b, now):
    done, green, red, other = batch_counts(b)
    total = b.start.get("total", "?")
    parts = [f"{green} green", *(f"{n} {k}" for k, n in sorted(other.items())), f"{len(red)} red"]
    s = f"{done}/{total} done ({', '.join(parts)})"
    if not b.done and b.start.get("predicted_wall") is not None and "ts" in b.start:
        end = parse_ts(b.start["ts"]) + dt.timedelta(seconds=int(b.start["predicted_wall"]))
        s += f", predicted end {clock(end, now)}"
    return s


def done_line(ev):
    secs = ("wall", "laneA", "laneB", "laneX")
    skip = ("ev", "ts", "deferred_list")
    return "DONE " + " ".join(f"{k}={v}{'s' if k in secs else ''}" for k, v in ev.items() if k not in skip)


def stage(n, trains, pr):
    for t in trains:
        if any(p.get("pr") == n for p in t.pieces):
            return f"in train {t.name} ({t.state})"
    if pr.get("isDraft"):
        return "waiting (draft)"
    if pr.get("reviewDecision") == "CHANGES_REQUESTED":
        return "waiting (changes requested)"
    return "not yet picked"


def mood(b, now):
    """(the header piece's name, its caption): a red in the newest batch is a rain cloud, a batch running is the furnace,
    an all-green newest batch or a train landed within BLOOM_S is the flower, else IDLE rotates by date."""
    def label(x):  # the matching train's name, else the batch's tree
        t = next((t for t in b.trains if t.branch and x.start.get("branch") == t.branch), None)
        return t.name if t else (x.start.get("tree") or x.name)

    cur = b.batches[0] if b.batches else None
    if cur:
        done, green, red, _ = batch_counts(cur)
        total, recent = cur.start.get("total", "?"), (now - cur.last).total_seconds() <= QUIET_S
        if red and recent:
            return "rain", f"{label(cur)}: {len(red)} red ({', '.join(n for n, *_ in red[:3])}), {green}/{total} green"
    for x in b.batches:
        if batch_status(x, now) == "running":
            return "furnace", f"{label(x)}: {batch_counts(x)[1]}/{x.start.get('total', '?')} green, running"
    if cur and cur.done and not red and recent:
        return "bloom", f"{label(cur)}: {done}/{total} done, 0 red"
    landed = [t for t in b.trains if t.state == "landed" and (now - t.at).total_seconds() <= BLOOM_S]
    if landed:
        return "bloom", f"{landed[0].name} landed {clock(landed[0].at, now)}"
    return IDLE[local(now).date().toordinal() % len(IDLE)], "quiet: nothing running"


def release_lines(b, now):
    rel = b.release
    L = ["## Release", ""]
    L += [f"- problem: {p}" for p in b.problems["release"]] + ([""] if b.problems["release"] else [])
    if rel is None:
        return L + ["_No release planned._", ""]
    counts = {g: sum(1 for it in rel.items if it["state"] == g) for g, _ in RELEASE_GROUPS}
    L += [f"### {code(rel.version)}: {rel.state}", "",
          f"cut window {one_line(rel.cut)}; plan updated {clock(rel.at, now)}; "
          + ", ".join(f"{counts[g]} {h.split(' (')[0].lower()}" for g, h in RELEASE_GROUPS), ""]
    for g, heading in RELEASE_GROUPS:
        L += [f"#### {heading} ({counts[g]})", ""]
        for it in (it for it in rel.items if it["state"] == g):
            eta = f", ETA {it['eta']}" if it.get("eta") else ""
            flag = ("" if not overdue(it, now) else ", **overdue**") + ("" if not late(rel, it) else ", **late for the cut**")
            note = f" — {one_line(it['note'])}" if it.get("note") else ""
            L.append(f"- {one_line(it['what'])} ({one_line(it.get('owner', '?'))}){eta}{flag}{note}")
        L += [] if counts[g] else ["_None._", ""]
        L += [""] if counts[g] else []
    return L


def more(n, shown):
    return [f"- … and {n - shown} more"] if n > shown else []


def render(b, now):
    when = local(now).strftime("%Y-%m-%d %H:%M %z")
    piece, caption = mood(b, now)
    L = ["```text", *PIECES[piece], ascii_cut(caption, ART_COLS), "```", "",
         "_Rewritten every 10 minutes by `tools/board/board.py`: state files, `gh` and `git log` only, no LLM. "
         f"Rendered {when}._", ""]
    L += release_lines(b, now)

    def problems(key):
        return [*(f"- problem: {p}" for p in b.problems[key]), *([""] if b.problems[key] else [])]

    # Trains
    L += ["## Trains", ""]
    stamp = max((t.at for t in b.trains), default=None)
    L += [f"_as of {clock(stamp, now)} (the newest train file)_" if stamp else "_no train files_", ""]
    L += problems("trains")
    if not b.trains:
        L += ["_No trains in flight._", ""]
    for t in b.trains:
        L += [f"### {code(t.name)}: {t.state}", f"updated {clock(t.at, now)}; branch {code(t.branch or '-')} on base {code(t.base or '-')}"
              + (f"; {one_line(t.note)}" if t.note else ""), ""]
        for p in t.pieces[: LIST_MAX["pieces"]]:
            pr = f" #{p['pr']}" if p.get("pr") else ""
            L.append(f"- {one_line(p['what'])} ({one_line(p.get('owner', '?'))}){pr}")
        L += more(len(t.pieces), LIST_MAX["pieces"])
        mine = next((x for x in b.batches if t.branch and x.start.get("branch") == t.branch), None)
        if mine:
            L.append(f"- batch {code(mine.name)}: {batch_status(mine, now)}, {batch_progress(mine, now)}")
        L.append("")

    # Batches
    L += ["## Batches", ""]
    stamp = max((x.last for x in b.batches), default=None)
    L += [f"_as of {clock(stamp, now)} (the newest event)_" if stamp else "_no batch files_", ""]
    L += problems("batches")
    if not b.batches:
        L += ["_No batch in the last day._", ""]
    for x in b.batches[: LIST_MAX["batches"]]:
        _, _, red, _ = batch_counts(x)
        what = [f"tree {code(x.start['tree'])}" if x.start.get("tree") else None,
                f"branch {code(x.start['branch'])}" if x.start.get("branch") else None,
                f"{x.start['tier']} tier" if x.start.get("tier") else None,
                f"{x.start['mode']} ledger" if x.start.get("mode") not in (None, "none") else None]
        L.append(f"- {code(x.name)}: {batch_status(x, now)}, {batch_progress(x, now)}"
                 + (f" — {', '.join(w for w in what if w)}" if any(what) else ""))
        for name, rc, lane, secs in red[: LIST_MAX["red"]]:
            L.append(f"  - red: {code(name)} rc {rc}, lane {lane}, {secs} s")
        if len(red) > LIST_MAX["red"]:
            L.append(f"  - … and {len(red) - LIST_MAX['red']} more red")
        if x.done:
            L.append(f"  - {code(done_line(x.done))}")
    if b.batches:
        L += more(len(b.batches), LIST_MAX["batches"]) + [""]

    # Workers
    L += ["## Workers", "", "_each file's own time; quiet for 3 hours reads as stale_", ""]
    L += problems("workers")
    if not b.workers:
        L += ["_No worker files._", ""]
    for role, at, stale, lines, cutn in b.workers:
        L += [f"### {code(role)}: {'stale since' if stale else 'updated'} {clock(at, now)}", ""]
        L += [f"- {one_line(ln)}" for ln in lines]
        if cutn:
            L.append(f"- … {cutn} more line(s) not shown")
        L.append("")

    # Contributor PRs and issues
    L += ["## Contributor PRs and issues", ""]
    L += [f"_as of {clock(now, now)} (queried now)_", ""]
    L += problems("gh")
    if b.gh_off:
        L += ["_Not queried (--no-gh)._", ""]
    else:
        L += ["### Open PRs", ""]
        if b.prs is None:
            L += ["_unavailable (above)_", ""]
        elif not b.prs:
            L += ["_No open PRs._", ""]
        else:
            for pr in sorted(b.prs, key=lambda p: -p["number"])[: LIST_MAX["prs"]]:
                L.append(f"- #{pr['number']} {code(cut(pr['title'], SUBJECT_MAX))} by {code(login(pr.get('author')) or '?')}: "
                         f"{stage(pr['number'], b.trains, pr)}, updated {clock(parse_ts(pr['updatedAt'].replace('Z', '+00:00')), now)}")
            L += more(len(b.prs), LIST_MAX["prs"]) + [""]
        L += [f"### Issues with a contributor comment in the last {COMMENT_DAYS} days", ""]
        if b.issues is None:
            L += ["_unavailable (above)_", ""]
        elif not b.issues:
            L += ["_None._", ""]
        else:
            for it, (t, who) in b.issues[: LIST_MAX["issues"]]:
                L.append(f"- #{it['number']} {code(cut(it['title'], SUBJECT_MAX))}: last comment by {code(who)} {clock(t, now)}")
            L += more(len(b.issues), LIST_MAX["issues"]) + [""]

    # Recently landed
    L += ["## Recently landed", ""]
    stamp = max((t for _, t, _ in b.log), default=None) if b.log else None
    L += [f"_as of {clock(now, now)}; origin/main as this checkout has it, newest commit {clock(stamp, now)}_" if stamp
          else f"_as of {clock(now, now)}_", ""]
    L += problems("log")
    if b.log is not None and not b.log:
        L += ["_No commits._"]
    for h, _, subject in b.log or []:
        L.append(f"- {h} {code(cut(subject, SUBJECT_MAX))}")
    return L


def finish(lines, patterns=PRIVACY_RES):
    """(the body, the lines withheld): the lines the privacy filter keeps, then the count it withheld. A body over the
    limit is an error."""
    kept, dropped = withhold(lines, patterns)
    body = "\n".join([*kept, "", f"{len(dropped)} lines withheld", ""])
    if len(body) > BODY_MAX:
        raise ValueError(f"the body is {len(body)} characters, over the {BODY_MAX} this board allows (GitHub's limit is 65536)")
    return body, dropped


# ----------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------
def parser():
    ap = argparse.ArgumentParser(prog="board.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--publish", metavar="ISSUE", help="replace this issue's body instead of printing it")
    ap.add_argument("--board-issue", metavar="ISSUE", help="the board's own issue, left out of the issue list (--publish implies it; "
                    "a preview names it to print what --publish would send)")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--dir", default=os.path.expanduser("~/.cache/bloomery/board"), help="the state directory")
    ap.add_argument("--tree", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
                    help="the checkout whose origin/main the landed list reads (default: the one holding this file)")
    ap.add_argument("--repo", help="OWNER/NAME for gh (default: the tree's origin remote)")
    ap.add_argument("--gh", default="gh", help="the gh program")
    ap.add_argument("--maintainer", action="append", help="a login whose comments do not list an issue (default midagedev)")
    ap.add_argument("--no-gh", action="store_true", help="skip the PR and issue lists")
    ap.add_argument("--explain", action="store_true", help="print each withheld line and the pattern that took it to stderr")
    return ap


def parse(argv):
    a = parser().parse_args(argv)
    a.maintainer = tuple(a.maintainer or DEFAULT_MAINTAINERS)
    a.board_issue = a.board_issue or a.publish
    return a


def main(argv, now=None):
    a = parse(argv)
    for flag, v in (("--publish", a.publish), ("--board-issue", a.board_issue)):
        if v is not None and not v.isdigit():
            print(f"board.py: {flag} takes an issue number", file=sys.stderr)
            return 64
    now = now or dt.datetime.now().astimezone()
    board = collect(a, now)
    rc = 0
    for ps in board.problems.values():
        for p in ps:
            print(f"board.py: {p}", file=sys.stderr)
            rc = 1
    try:
        body, dropped = finish(render(board, now))
    except ValueError as e:
        print(f"board.py: {e}", file=sys.stderr)
        return 1
    if dropped:
        names = sorted({n for n, _ in dropped})
        print(f"board.py: {len(dropped)} lines withheld ({', '.join(names)})", file=sys.stderr)
        for n, ln in dropped if a.explain else ():
            print(f"board.py: withheld ({n}): {ln}", file=sys.stderr)
    if a.publish is None:
        sys.stdout.write(body)
        return rc
    repo = a.repo or repo_slug(a.tree)
    if repo is None:
        print("board.py: --publish needs OWNER/NAME: the tree has no github origin remote and --repo is not given", file=sys.stderr)
        return 1
    prc, _, err = run([a.gh, "issue", "edit", a.publish, "--repo", repo, "--body-file", "-"], stdin=body)
    if prc != 0:
        print(f"board.py: gh issue edit {a.publish} failed (exit {prc}): {one_line(err)[:200]}", file=sys.stderr)
        return 1
    return rc


# ----------------------------------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------------------------------
class Cases:
    def __init__(self):
        self.n = self.bad = 0

    def check(self, name, cond, detail=""):
        self.n += 1
        if cond:
            print(f"ok {name}")
        else:
            self.bad += 1
            print(f"FAIL {name}: {detail}")


# One line per privacy pattern. Addresses are assembled so this file holds none of them whole.
PRIVACY_SAMPLES = {
    "lan": "host 192" + ".168.1.20 is up",
    "tailnet-ip": "node 100" + ".64.0.7 joined",
    "tailnet-name": "box.tail1234" + ".ts" + ".net answers",
    "admin": "log in as admin now",
    "host-sihyung": "ran on sihyung last night",
    "host-dopc": "the dopc host",
    "mac-home": "see /Users" + "/someone/repo/x",
    "root-path": "file /root" + "/bloomery-gate.lock",
    "linux-home": "at /home" + "/someone/x",
    "hf-token": "key hf" + "_abcdefghij12 leaked",
    "gh-token": "key gh" + "p_abcdef leaked",
    "token-assign": "my TOKEN: abc is here",
    "bearer": "Authorization: Bearer abcdef",
}
CLEAN = (
    "- #12 `a plain title` by `someone`: not yet picked",
    "the administrator and 100 tok/s and 1.100.2 and home/x",
    "serve: named 400s for token limits, /tokenize content, unserved samplers (#24-#26)",
    "- #24 `serve: a token limit outside -1..=i32::MAX is a 400 naming the field`",
    "the tokens, the Tokenizer and a token budget of 4096",
)


def utime(path, now, age_s):
    t = (now - dt.timedelta(seconds=age_s)).timestamp()
    os.utime(path, (t, t))


def write(path, text, now=None, age_s=0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    if now:
        utime(path, now, age_s)


def ev(now, age_s, kind, **kw):
    return json.dumps({"ev": kind, "ts": (now - dt.timedelta(seconds=age_s)).isoformat(timespec="seconds"), **kw})


def stub_gh(path, state):
    """A gh that answers `pr list` and `issue list` from files under state, records `issue edit`, and fails the
    subcommand named in state/fail."""
    write(path, f"""#!{sys.executable}
import json, os, sys
state = {state!r}
sub = " ".join(sys.argv[1:3])
fail = open(state + "/fail").read().split() if os.path.exists(state + "/fail") else []
if sub.replace(" ", "-") in fail:
    sys.exit(1)
if sub == "issue edit":
    json.dump({{"argv": sys.argv[1:], "body": sys.stdin.read()}}, open(state + "/edit.json", "w"))
elif sub in ("pr list", "issue list"):
    sys.stdout.write(open(state + "/" + sub.split()[0] + ".json").read())
else:
    sys.exit(2)
""")
    os.chmod(path, 0o755)


def git(tree, *args):
    env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
    subprocess.run(["git", "-C", tree, *args], check=True, capture_output=True, env=env)


def capture(fn, *args):
    """(fn's result, what it wrote to stdout, what it wrote to stderr)."""
    out, err = io.StringIO(), io.StringIO()
    real = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err
    try:
        rc = fn(*args)
    finally:
        sys.stdout, sys.stderr = real
    return rc, out.getvalue(), err.getvalue()


def self_test():
    c = Cases()
    now = dt.datetime(2026, 10, 11, 15, 0, 0).astimezone()

    # The privacy filter: one case per pattern, and the case is red with the filter off (an identity filter keeps the line).
    c.check("privacy: every pattern has a case and no case is orphaned", set(PRIVACY_SAMPLES) == {n for n, _ in PRIVACY}, "pattern and sample names differ")
    for name, _ in PRIVACY:
        line = PRIVACY_SAMPLES[name]
        kept, dropped = withhold([line])
        kept_off, dropped_off = withhold([line], ())
        c.check(f"privacy: {name} is withheld, and kept with the filter off", kept == [] and dropped == [(name, line)] and kept_off == [line] and dropped_off == [],
                f"on: kept={kept} dropped={dropped}; off: kept={kept_off} dropped={dropped_off}")
    kept, dropped = withhold(list(CLEAN))
    c.check("privacy: lines near a pattern but outside it are kept", kept == list(CLEAN) and dropped == [], f"kept={kept} dropped={dropped}")
    c.check("privacy: \"token limits\" and \"tokenize\" in a title are not withheld",
            withhold(["- #25 `serve: /tokenize takes content`", "- #24 `serve: a token limit outside -1..=i32::MAX is a 400`"])[1] == [], "withheld")
    body, _ = finish(["- keep this", PRIVACY_SAMPLES["lan"], PRIVACY_SAMPLES["token-assign"], "- and this"])
    c.check("privacy: the body ends with the count and holds neither line", body.endswith("\n2 lines withheld\n") and "192" not in body
            and "TOKEN" not in body and "keep this" in body and "and this" in body, body)
    try:
        finish(["x" * (BODY_MAX + 1)])
        c.check("size: a body over the limit is an error", False, "no error")
    except ValueError as e:
        c.check("size: a body over the limit is an error", "over the" in str(e), str(e))

    with tempfile.TemporaryDirectory(prefix="board-test-") as tmp:
        root, gh_state, tree = (os.path.join(tmp, d) for d in ("board", "gh-state", "tree"))
        os.makedirs(gh_state)
        os.makedirs(tree)
        args = ["--dir", root, "--tree", tree, "--repo", "o/n", "--gh", os.path.join(tmp, "gh"), "--publish", "99"]
        stub_gh(os.path.join(tmp, "gh"), gh_state)

        # Absent state: every section says so; a gh that answers nothing is an empty list.
        write(os.path.join(gh_state, "pr.json"), "[]")
        write(os.path.join(gh_state, "issue.json"), "[]")
        origin = os.path.join(tmp, "origin.git")
        git(tmp, "init", "-q", "--bare", "-b", "main", origin)
        git(tree, "init", "-q", "-b", "main")
        write(os.path.join(tree, "f"), "1")
        git(tree, "add", "f")
        git(tree, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "x" * 150)
        git(tree, "remote", "add", "origin", origin)
        git(tree, "push", "-q", "origin", "main")
        b = collect(parse(args[:-2]), now)
        text = finish(render(b, now))[0]
        c.check("empty: no state directory renders each section empty", all(s in text for s in (
            "_No release planned._", "_No trains in flight._", "_No batch in the last day._", "_No worker files._", "_No open PRs._", "_None._")), text)
        c.check("landed: a subject is cut to 100 characters", ("x" * 99 + "…") in text and "x" * 100 not in text, text)

        # Trains, with a batch matched by branch.
        write(os.path.join(root, "trains/t3.json"), json.dumps({
            "name": "train3", "branch": "train3", "base": "abc1234", "state": "batch", "note": "on the lanes",
            "pieces": [{"what": "kv pool", "owner": "kvpool", "pr": 41}, {"what": "spark docs", "owner": "sparkdoc", "pr": None}]}), now, 600)
        write(os.path.join(root, "trains/old.json"), json.dumps({"name": "train2", "branch": "t2", "base": "b", "state": "landed", "pieces": []}), now, DROP_S + 60)
        write(os.path.join(root, "trains/fresh.json"), json.dumps({"name": "train4", "branch": "t4", "base": "b", "state": "landed", "pieces": []}), now, 3600)
        write(os.path.join(root, "trains/bad.json"), "{not json", now, 0)
        write(os.path.join(root, "trains/odd.json"), json.dumps({"name": "x", "state": "sleeping", "pieces": []}), now, 0)
        write(os.path.join(root, "batch/20261011-140000.jsonl"), "\n".join([
            ev(now, 3600, "start", total=5, tier="fixture", lanes=2, mode="round", predicted_wall=7200, tree="bloomery-train3", branch="train3"),
            ev(now, 3000, "item", name="gate-a", rc=0, secs=100, lane="A"),
            ev(now, 2900, "item", name="gate-b", rc=1, secs=200, lane="B"),
            ev(now, 2800, "item", name="gate-c", rc="skip", secs=0, lane="A")]) + "\n", now, 2800)
        write(os.path.join(root, "batch/corrupt.jsonl"), ev(now, 100, "start", total=1, tier="real", lanes=2, mode="none", predicted_wall=10,
                                                            tree="c", branch="cb") + "\nthis is not json\n", now, 100)
        write(os.path.join(root, "batch/20261011-100000.jsonl"), "\n".join([
            ev(now, 20000, "start", total=1, tier="real", lanes=2, mode="lead", predicted_wall=100, tree="bloomery", branch="main"),
            ev(now, 19900, "item", name="gate-z", rc=0, secs=90, lane="A"),
            ev(now, 19890, "done", total=1, red=0, wall=95, laneA=90, laneB=0)]) + "\n", now, 19890)
        write(os.path.join(root, "batch/20261011-090000.jsonl"), "\n".join([
            ev(now, 30000, "start", total=2, tier="real", lanes=2, mode="lead", predicted_wall=100, tree="bloomery", branch="m"),
            ev(now, 29990, "stop")]) + "\n", now, 29990)
        write(os.path.join(root, "batch/20261011-080000.jsonl"), ev(now, 40000, "start", total=2, tier="real", lanes=2, mode="lead",
                                                                    predicted_wall=100, tree="bloomery", branch="q") + "\n", now, 40000)
        write(os.path.join(root, "batch/ancient.jsonl"), ev(now, DROP_S + 100, "start", total=1) + "\n", now, DROP_S + 100)
        write(os.path.join(root, "batch/reused.jsonl"), "\n".join([
            ev(now, 900, "start", total=9, tier="fixture", lanes=2, mode="none", predicted_wall=60, tree="r", branch="rb"),
            ev(now, 800, "item", name="first", rc=0, secs=1, lane="A"),
            ev(now, 400, "start", total=2, tier="fixture", lanes=2, mode="none", predicted_wall=60, tree="r", branch="rb"),
            ev(now, 300, "item", name="second", rc=0, secs=1, lane="A")]) + "\n", now, 300)
        write(os.path.join(root, "workers/kvpool.md"), "round: kv pool\nETA 16:30\nwaits on: train3 batch\n", now, 600)
        write(os.path.join(root, "workers/sparkdoc.md"), "round: spark docs\n", now, QUIET_S + 3600)
        write(os.path.join(root, "workers/long.md"), "".join(f"line {i}\n" for i in range(9)), now, 60)
        write(os.path.join(gh_state, "pr.json"), json.dumps([
            {"number": 41, "title": "kv pool `@mention` <b>x</b>", "author": {"login": "kvpool"}, "isDraft": False, "updatedAt": "2026-10-11T05:00:00Z", "reviewDecision": ""},
            {"number": 42, "title": "a draft", "author": {"login": "ext1"}, "isDraft": True, "updatedAt": "2026-10-10T05:00:00Z", "reviewDecision": ""},
            {"number": 43, "title": "needs work", "author": {"login": "ext2"}, "isDraft": False, "updatedAt": "2026-10-10T05:00:00Z", "reviewDecision": "CHANGES_REQUESTED"},
            {"number": 44, "title": "ready, unpicked", "author": {"login": "ext3"}, "isDraft": False, "updatedAt": "2026-10-10T05:00:00Z", "reviewDecision": "APPROVED"}]))
        recent, old = "2026-10-09T05:00:00Z", "2026-09-20T05:00:00Z"
        write(os.path.join(gh_state, "issue.json"), json.dumps([
            {"number": 7, "title": "ext question", "author": {"login": "ext1"}, "comments": [
                {"author": {"login": "midagedev"}, "createdAt": recent}, {"author": {"login": "ext1"}, "createdAt": recent}]},
            {"number": 8, "title": "only maintainer", "author": {"login": "ext1"}, "comments": [{"author": {"login": "midagedev"}, "createdAt": recent}]},
            {"number": 9, "title": "old contributor comment", "author": {"login": "ext1"}, "comments": [{"author": {"login": "ext1"}, "createdAt": old}]},
            {"number": 10, "title": "bot comment", "author": {"login": "ext1"}, "comments": [{"author": {"login": "dependabot", "is_bot": True}, "createdAt": recent}]},
            {"number": 99, "title": "the board itself", "author": {"login": "midagedev"}, "comments": [{"author": {"login": "ext9"}, "createdAt": recent}]}]))
        b = collect(parse(args), now)
        text = finish(render(b, now))[0]
        for name, want in (
            ("trains: a train shows its state, branch and pieces with PR numbers", ["### `train3`: batch", "branch `train3` on base `abc1234`", "- kv pool (kvpool) #41", "- spark docs (sparkdoc)"]),
            ("trains: a fresh landed train stays", ["`train4`: landed"]),
            ("trains: the train shows the batch matched by its branch", ["- batch `20261011-140000`: running, 3/5 done (1 green, 1 skip, 1 red), predicted end 16:00"]),
            ("trains: an unreadable and an invalid file are named", ["- problem: trains/bad.json:", "- problem: trains/odd.json: state \"sleeping\" is none of"]),
            ("batches: red items are named with their rc and lane", ["  - red: `gate-b` rc 1, lane B, 200 s", "tree `bloomery-train3`, branch `train3`, fixture tier, round ledger"]),
            ("batches: a bad event line is a named problem", ["- problem: batch/corrupt.jsonl: 1 unreadable event line(s)", "`corrupt`: running, 0/1 done"]),
            ("batches: an ended batch shows its DONE line", ["`20261011-100000`: ended, 1/1 done (1 green, 0 red)", "`DONE total=1 red=0 wall=95s laneA=90s laneB=0s`"]),
            ("batches: a stopped batch says so", ["`20261011-090000`: stopped (signal) at"]),
            ("batches: a batch with no DONE and no event for hours says when it last spoke", ["`20261011-080000`: no DONE, last event"]),
            ("batches: only the last run of a reused log name counts", ["`reused`: running, 1/2 done"]),
            ("workers: a fresh file shows its lines under its time", ["### `kvpool`: updated 14:50", "- ETA 16:30"]),
            ("workers: a file quiet for 3 hours is stale since its time", ["### `sparkdoc`: stale since 11:00"]),
            ("workers: lines past six are counted, not shown", ["- line 5", "- … 3 more line(s) not shown"]),
            ("prs: a train piece's PR is in its train", ["#41 `kv pool '@mention' <b>x</b>` by `kvpool`: in train train3 (batch)"]),
            ("prs: a draft and a changes-requested PR are waiting, the rest not yet picked",
             ["#42 `a draft` by `ext1`: waiting (draft)", "#43 `needs work` by `ext2`: waiting (changes requested)", "#44 `ready, unpicked` by `ext3`: not yet picked"]),
            ("issues: a non-maintainer comment in the week lists the issue", ["#7 `ext question`: last comment by `ext1`"])):
            miss = [w for w in want if w not in text]
            c.check(name, not miss, f"missing {miss}")
        for name, absent in (("trains: an old landed train drops off", "`train2`"),
                             ("batches: a batch quiet for a day drops off", "ancient"),
                             ("issues: a maintainer-only comment does not list", "only maintainer"),
                             ("issues: an old contributor comment does not list", "old contributor comment"),
                             ("issues: a bot comment does not list", "bot comment"),
                             ("issues: the board's own issue is not listed", "the board itself"),
                             ("prs: a title carries no live mention or markup (a code span, backticks turned to quotes)", "`@mention`")):
            c.check(name, absent not in text, f"{absent} is in the body")

        # The privacy filter inside the full path: a worker line with a home path is withheld and counted.
        for f in ("trains/bad.json", "trains/odd.json", "batch/corrupt.jsonl"):  # the problem fixtures: they make the exit 1
            os.remove(os.path.join(root, f))
        write(os.path.join(root, "workers/leaky.md"), "round: x\nsee /Users" + "/someone/repo for the log\n", now, 60)
        text = finish(render(collect(parse(args), now), now))[0]
        c.check("privacy: a leaking worker line is gone from the full body and counted", "someone" not in text and text.endswith("\n1 lines withheld\n"), text[-200:])
        _, _, err = capture(main, args[:-2] + ["--explain"], now)
        c.check("privacy: stderr counts what was withheld and --explain names the line and its pattern",
                "1 lines withheld (mac-home)" in err and "withheld (mac-home): - see /Users" in err and "withheld (mac-home): " not in capture(main, args[:-2], now)[2], err)
        os.remove(os.path.join(root, "workers/leaky.md"))

        # Print mode and publish mode send the same body, through gh with the issue number, the repo and the body on stdin.
        rc_print, printed, _ = capture(main, args[:-2] + ["--board-issue", "99"], now)
        rc_pub, _, _ = capture(main, args, now)
        sent = json.load(open(os.path.join(gh_state, "edit.json")))
        c.check("publish: gh issue edit gets the issue, the repo and the body on stdin", sent["argv"] == ["issue", "edit", "99", "--repo", "o/n", "--body-file", "-"], sent["argv"])
        c.check("publish: the body sent is the body printed", rc_print == 0 and rc_pub == 0 and sent["body"] == printed and sent["body"].endswith("lines withheld\n"),
                f"rc {rc_print} {rc_pub}")
        c.check("publish: the issue-number argument is checked", capture(main, ["--publish", "abc", "--dir", root, "--no-gh"], now)[0] == 64, "no 64")

        # A source that fails is a named line in its section and exit 1, and the rest is still published.
        full = json.loads(open(os.path.join(gh_state, "pr.json")).read())
        write(os.path.join(gh_state, "pr-full.json"), json.dumps([dict(full[0], number=1000 + i) for i in range(GH_LIMIT)]))
        os.rename(os.path.join(gh_state, "pr.json"), os.path.join(gh_state, "pr-small.json"))
        os.rename(os.path.join(gh_state, "pr-full.json"), os.path.join(gh_state, "pr.json"))
        rc, out, _ = capture(main, args[:-2], now)
        c.check("gh limit: a list that fills gh's page is a named problem and exit 1", rc == 1 and f"- problem: gh pr list returned {GH_LIMIT} rows" in out, f"rc {rc}")
        os.rename(os.path.join(gh_state, "pr-small.json"), os.path.join(gh_state, "pr.json"))
        os.remove(os.path.join(gh_state, "edit.json"))
        write(os.path.join(gh_state, "fail"), "pr-list")
        rc, _, err = capture(main, args, now)
        sent = json.load(open(os.path.join(gh_state, "edit.json")))
        c.check("gh failure: the section names it, the exit is 1, the other sections still go out",
                rc == 1 and "- problem: gh pr list failed (exit 1)" in sent["body"] and "`train3`" in sent["body"] and "ext question" in sent["body"]
                and "gh pr list failed" in err, f"rc {rc} {err}")
        write(os.path.join(gh_state, "fail"), "issue-edit")
        c.check("gh failure: a failed issue edit is exit 1", capture(main, args, now)[0] == 1, "not 1")
        rc, _, err = capture(main, ["--dir", root, "--tree", tree, "--gh", os.path.join(tmp, "missing-gh"), "--repo", "o/n"], now)
        c.check("gh failure: no gh program is exit 1 naming it, not a crash", rc == 1 and "gh pr list failed (exit 127)" in err, f"rc {rc} {err}")
        # git fetch before the log: a commit another clone pushed shows, and a fetch that fails is a named problem over the stale log.
        git(tmp, "clone", "-q", origin, os.path.join(tmp, "other"))
        write(os.path.join(tmp, "other", "g"), "2")
        git(os.path.join(tmp, "other"), "add", "g")
        git(os.path.join(tmp, "other"), "-c", "commit.gpgsign=false", "commit", "-q", "-m", "pushed from another clone")
        git(os.path.join(tmp, "other"), "push", "-q", "origin", "main")
        rc, out, err = capture(main, ["--dir", root, "--tree", tree, "--no-gh"], now)
        c.check("fetch: a commit pushed after this checkout's last fetch is in Recently landed", rc == 0 and "pushed from another clone" in out, f"rc {rc} {err}")
        os.rename(origin, origin + ".gone")
        rc, out, err = capture(main, ["--dir", root, "--tree", tree, "--no-gh"], now)
        c.check("fetch: a failed fetch is a named problem, exit 1, over the log as the last fetch left it",
                rc == 1 and "- problem: git fetch origin main failed" in out and "pushed from another clone" in out and "git fetch:" in err, f"rc {rc} {err}")
        os.rename(origin + ".gone", origin)
        rc, out, err = capture(main, ["--dir", root, "--tree", os.path.join(tmp, "no-such-tree"), "--no-gh"], now)
        c.check("git failure: a tree with no origin/main is exit 1 and the body says so", rc == 1 and "- problem: git log origin/main failed" in out, f"rc {rc} {err}")

        # The release plan (release.json): each group, the late flag, and the files that are wrong.
        rroot = os.path.join(tmp, "rel")
        rargs = ["--dir", rroot, "--tree", tree, "--no-gh"]
        item = lambda what, state, eta=None, note="": {"what": what, "owner": "o", "state": state, "eta": eta, "note": note}
        write(os.path.join(rroot, "release.json"), json.dumps({"version": "0.3.0", "cut": "2026-10-11 22:00–24:00 KST", "state": "collecting", "items": [
            item("landed-a", "landed"), item("landed-b", "landed", "23:59"),
            item("train-c", "in-train", "17:30", "train1"), item("train-d", "in-train", "22:30"),
            item("pend-e", "pending", "19:00"), item("pend-f", "pending", "23:00", "may slip"), item("pend-g", "pending", "22:00"),
            item("def-h", "deferred", None, "0.3.1"),
            item("train-o", "in-train", "14:00"), item("pend-p", "pending", "14:59"), item("pend-q", "pending", "15:00"), item("landed-r", "landed", "10:00")]}), now, 300)
        text = finish(render(collect(parse(rargs), now), now))[0]
        rel = text[text.index("## Release"):text.index("## Trains")]
        row = lambda what: next((ln for ln in rel.splitlines() if ln.startswith(f"- {what} ")), "")
        c.check("release: the version, state, cut window and counts head the section",
                "### `0.3.0`: collecting" in rel and "cut window 2026-10-11 22:00–24:00 KST; plan updated 14:55; 3 landed, 3 in a train now, 5 pending, 1 deferred" in rel, rel)
        c.check("release: the four groups are headed with their counts, in order",
                [h for h in ("#### Landed (3)", "#### In a train now (3)", "#### Pending (with ETA) (5)", "#### Deferred to the next release (1)") if h in rel] ==
                [ln for ln in rel.splitlines() if ln.startswith("#### ")], rel)
        c.check("release: a landed item is under Landed with no ETA", rel.index("- landed-a") < rel.index("#### In a train now"), rel)
        c.check("release: an item in a train shows its ETA and note", "- train-c (o), ETA 17:30 — train1" in rel and rel.index("- train-c") < rel.index("#### Pending"), rel)
        c.check("release: a pending item shows its ETA and note", "- pend-f (o), ETA 23:00, **late for the cut** — may slip" in rel, rel)
        c.check("release: a deferred item is under Deferred with its note", "- def-h (o) — 0.3.1" in rel and rel.index("- def-h") > rel.index("#### Deferred"), rel)
        c.check("release: an in-train item whose ETA is past the window's start is late", "late for the cut" in row("train-d"), row("train-d"))
        c.check("release: a pending item whose ETA is past the window's start is late", "late for the cut" in row("pend-f"), row("pend-f"))
        c.check("release: an ETA before the start, or equal to it, is not late", all("late" not in row(w) for w in ("train-c", "pend-e", "pend-g")), rel)
        c.check("release: a landed item is never late, whatever its ETA says", "late" not in row("landed-b") and "late" not in row("landed-a"), rel)
        c.check("release: exactly the two late items are flagged", rel.count("late for the cut") == 2, rel)
        c.check("release: an item in a train or pending whose ETA is before now is overdue", "- train-o (o), ETA 14:00, **overdue**" in rel and "- pend-p (o), ETA 14:59, **overdue**" in rel, rel)
        c.check("release: an ETA at or after now is not overdue", all("overdue" not in row(w) for w in ("pend-q", "train-c", "pend-e", "pend-f", "pend-g")), rel)
        c.check("release: a landed item is never overdue, whatever its ETA says", "overdue" not in row("landed-r") and "overdue" not in row("landed-b"), rel)
        c.check("release: exactly the two overdue items are flagged, and an overdue one is not late", rel.count("**overdue**") == 2 and "late" not in row("train-o"), rel)
        good = open(os.path.join(rroot, "release.json")).read()
        for name, mutate, want in (
            ("a cut with no window", lambda j: j.update(cut="after lunch"), "- problem: release.json: cut \"after lunch\" holds no HH:MM–HH:MM window"),
            ("an unknown release state", lambda j: j.update(state="cooking"), "- problem: release.json: state \"cooking\" is none of"),
            ("an item in no group", lambda j: j["items"].append(item("x", "dreaming")), "- problem: release.json: item 13: state \"dreaming\" is none of"),
            ("an ETA that is no time", lambda j: j["items"].append(item("x", "pending", "soon")), "- problem: release.json: \"soon\" is not HH:MM")):
            j = json.loads(good)
            mutate(j)
            write(os.path.join(rroot, "release.json"), json.dumps(j))
            bd = collect(parse(rargs), now)
            text = finish(render(bd, now))[0]
            c.check(f"release: {name} is a named problem, no group is guessed", want in text and "_No release planned._" in text and "#### Landed" not in text and bd.problems["release"], text[:400])
        write(os.path.join(rroot, "release.json"), "{broken")
        c.check("release: a broken file is a named problem", collect(parse(rargs), now).problems["release"][0].startswith("release.json: "), "no problem")
        os.remove(os.path.join(rroot, "release.json"))
        c.check("release: no file is no release, and no problem", collect(parse(rargs), now).problems["release"] == [], "a problem")

        # The header art: its shape, and which piece each state picks.
        c.check("art: five to seven pieces", 5 <= len(PIECES) <= 7, len(PIECES))
        for name, art in PIECES.items():
            shape = [ln for ln in art if len(ln) > ART_COLS or any(not 0x20 <= ord(ch) < 0x7F or ch == "`" for ch in ln) or ln != ln.rstrip()]
            c.check(f"art: {name} is at most {ART_ROWS} lines of {ART_COLS} printable ASCII columns, no backtick, no trailing space",
                    len(art) <= ART_ROWS and not shape, f"{len(art)} lines, bad: {shape}")
        c.check("art: no line of any piece trips the privacy filter", withhold([ln for a in PIECES.values() for ln in a])[1] == [], "a piece line is withheld")
        c.check("art: a caption is plain ASCII within the width", ascii_cut("tr\u00e4in " + "x" * 100, ART_COLS) == "tr?in " + "x" * 47 + "..." and len(ascii_cut("tr\u00e4in " + "x" * 100, ART_COLS)) == ART_COLS, "")
        ago = lambda sec: now - dt.timedelta(seconds=sec)
        batch = lambda last, rcs, total=5, done=None, name="b0", branch="b": Batch(
            name, {"ts": ago(last + 600).isoformat(), "total": total, "predicted_wall": 3600, "tree": "tr", "branch": branch},
            [{"ev": "item", "name": f"g{i}", "rc": rc, "secs": 1, "lane": "A"} for i, rc in enumerate(rcs)], done or {}, {}, ago(last))
        landed = lambda age, name="train4": Train(name, "t4", "b", [], "landed", "", ago(age))
        board = lambda trains=(), batches=(): Board(list(trains), list(batches), [], None, None, None, None, {k: [] for k in ("trains", "batches", "workers", "release", "gh", "log")}, True)
        for name, bd, want, cap in (
            ("a batch running with no red is the furnace", board(batches=[batch(60, [0, 0])]), "furnace", "tr: 2/5 green, running"),
            ("a red in the newest batch is the rain cloud, even while it runs", board(batches=[batch(60, [0, 1])]), "rain", "tr: 1 red (g1), 1/5 green"),
            ("a red beats another batch that is running", board(batches=[batch(30, [2]), batch(90, [0])]), "rain", "tr: 1 red (g0), 0/5 green"),
            ("an all-green batch that just ended is the flower", board(batches=[batch(300, [0, 0], total=2, done={"ev": "done"})]), "bloom", "tr: 2/2 done, 0 red"),
            ("a train landed within the hour is the flower", board(trains=[landed(1800)]), "bloom", "train4 landed 14:30"),
            ("a train landed two hours ago is idle", board(trains=[landed(7200)]), None, "quiet: nothing running"),
            ("a batch that ended green five hours ago is idle", board(batches=[batch(5 * 3600, [0], total=1, done={"ev": "done"})]), None, "quiet: nothing running"),
            ("a red from five hours ago is idle", board(batches=[batch(5 * 3600, [1], total=1, done={"ev": "done"})]), None, "quiet: nothing running"),
            ("nothing at all is idle", board(), None, "quiet: nothing running")):
            piece, caption = mood(bd, now)
            c.check(f"art: {name}", (piece == want if want else piece in IDLE) and caption == cap and piece in PIECES, f"{piece!r} {caption!r}")
        c.check("art: a train's name captions a batch on its branch", mood(board(trains=[Train("train3", "b", "x", [], "batch", "", now)], batches=[batch(60, [0])]), now)[1] == "train3: 1/5 green, running", "")
        days = [mood(board(), now + dt.timedelta(days=d))[0] for d in range(len(IDLE))]
        c.check("art: idle rotates through every resting piece by date, one piece a day", sorted(days) == sorted(IDLE) and mood(board(), now)[0] == mood(board(), now + dt.timedelta(hours=1))[0], days)
        text = finish(render(collect(parse(rargs), now), now))[0]
        block = text.split("```")[1].splitlines()
        c.check("art: the body opens with the fenced art and one caption line, above the Release section",
                text.startswith("```text\n") and 2 <= len(block) <= ART_ROWS + 2 and max(len(ln) for ln in block) <= ART_COLS and text.index("```text") < text.index("## Release"), text[:300])

    print(f"self-test: {'FAIL' if c.bad else 'ok'} ({c.n} cases, {c.bad} failed)")
    return 1 if c.bad else 0


if __name__ == "__main__":
    if "--self-test" in sys.argv[1:]:
        sys.exit(self_test())
    sys.exit(main(sys.argv[1:]))
