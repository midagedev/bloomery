#!/usr/bin/env python3
"""Check and post the review bot's output (.github/workflows/review.yml).

    tools/review/post.py check  --out DIR --diff FILE --pr-json FILE [--files FILE] [--device FILE] [--freeze FILE]
                                [--base DIR] [--secret-env NAME]... [--secret-file FILE]... [--usage FILE]...
    tools/review/post.py device --root DIR --out FILE
    tools/review/post.py freeze --repo OWNER/NAME --out FILE                                             (GH_TOKEN)
    tools/review/post.py issues --out DIR --repo OWNER/NAME --pr N
    tools/review/post.py post   --out DIR --repo OWNER/NAME --pr N
    tools/review/post.py labels --out DIR --repo OWNER/NAME --pr N
    tools/review/post.py carry  --carry FILE --prev-review FILE --out DIR
    tools/review/post.py --self-test

`check` reads what the reviewer wrote to DIR (review.md, decision.txt, optional comments.json) and rewrites it in
place into the shape `post` sends: the decision follows the body's verdict line, inline comments that do not sit on a
line the diff adds are dropped (GitHub refuses the whole review for one), at most MAX_COMMENTS remain, the body is cut
to MAX_BODY characters, and the footer marker that the next run reads (head sha, verdict, counts, mode, the event posted) is appended, with
the slag ledger's version, the workflow run and the reviewer's identity when pr.json carries them (`ledger=`, `run=`:
the weekly learner reads the run's artifact back, tools/review/signals.py; `tools=`: tools/review/carry.py reviewer-id,
which a later carry compares), and a
second marker carrying each --usage file's accounting (tools/review/usage.py: turns, cost, context, tool calls) kept
in DIR/usage.txt so a later check without the files keeps it. Issue candidates (issues.json: defects outside the
diff) are kept when well formed, at most MAX_ISSUES, and with --base only when every evidence path exists there. It
refuses (exit 3, nothing written) when the output carries a credential: the value of each --secret-env variable, every
string of 16 or more characters in each --secret-file JSON file (a Codex login's auth.json: its tokens, as the run left
them), or a token-shaped string. A missing or unreadable output is exit 65.

`issues` files each candidate as an issue labelled ISSUE_LABEL, unless an issue with that label and the same title
exists (open or closed), and writes the numbers to DIR/filed.txt.

`check` also decides the review's event (OUT/event.txt; a later check keeps a COMMENT): REQUEST_CHANGES for a
MUST_FIX verdict; APPROVE for a CLEAN one when decision.txt agrees, the validator ran (its line is in review.md), and
no file of the pull request (--files, the `pulls/<n>/files` list, else the diff's paths) is under APPROVE_GUARD, the
paths that change the bot itself, CI or the toolchain and so need a person; else COMMENT, with a line saying why the
CLEAN review did not approve. With no approvals required, a change request does not lock the merge button; it says
"fix this first", and the bot's next review of a new head replaces it.
Card labels: `device` writes the directories of the crates that reach cuda-core (tools/recipes.py pure-crates, run
on the trusted tree). When the pull request changes a file under one, `check` sets the bot's card label in
OUT/labels.json: `gpu-evidence` when the reviewer's gpu.json says the pull request carries a gate run on a card
(CONTRIBUTING's evidence list), else `needs-gpu`, and adds a line saying so. A run on the author's card counts
as verified (CONTRIBUTING trusts it); `gpu-verified` is a maintainer's own run, and the bot never adds or removes it. While a release freeze is on (an open
issue labelled FREEZE_LABEL opened by the owner; `freeze` writes its number to FILE), a `needs-gpu` pull request is
not approved: a merge would change the tree under the release batch. `labels` applies OUT/labels.json.
`carry` writes into DIR the review of a carried approval (tools/review/carry.py decided `carry`: the pull request's
commits are the patches the bot approved at an earlier head, so no reviewer ran): the body, decision CLEAN, the card
evidence the carried review had, no comments and no issues. It refuses by name unless the carried review's own marker
says it approved (`verdict=CLEAN event=APPROVE`), so a MUST_FIX or a COMMENT never turns into an approval here. `check`
then decides the event as for any review, with every guard recomputed on the current files and freeze; the one
difference is that no validator line is expected (pr.json `mode` is `carry`, with the head it carries from in
`carried_from`).
`post` sends DIR as one pull request review with that event, using
GH_TOKEN, with a line naming the issues `issues` filed. When GitHub refuses the inline comments (422), it posts the
body alone and says so in the body.
"""

import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

MARKER = "bloomery-review"
# Shingler: the worker who hammers the slag out of a fresh bloom. The name is for people; MARKER stays as it is, since
# the learner finds past reviews by it.
SIGNATURE = ("<sub>🔨 Shingler, bloomery's review bot: approves a clean pull request and requests changes on a must-fix "
             "([how it works](https://github.com/{repo}/blob/main/.github/review/README.md))</sub>")
MAX_BODY = 12000
MAX_COMMENTS = 10
CARD_LABELS = {
    "needs-gpu": ("d93f0b", "Changes code a card must check, and carries no gate run on a card"),
    "gpu-evidence": ("bfe5bf", "Verified on the author's card: the pull request carries the gate run and its output"),
}
FREEZE_LABEL = "release-freeze"
APPROVE_GUARD = (".github/", "tools/review/", "AGENTS.md", "CLAUDE.md", "CONTRIBUTING.md", "CONTRIBUTING.ko.md",
                 "rust-toolchain.toml", "deny.toml", ".cargo/")
MAX_ISSUES = 2
ISSUE_LABEL = "review-bot"
VERDICTS = ("MUST_FIX", "CLEAN")
TOKEN_RE = re.compile(
    r"sk-ant-[A-Za-z0-9_-]{16,}"
    r"|gh[pousr]_[A-Za-z0-9]{30,}"
    r"|github_pat_[A-Za-z0-9_]{30,}"
    r"|\bsk-(?:proj|svcacct|admin)-[A-Za-z0-9_-]{20,}"
    r"|\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]*"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
)
VERDICT_RE = re.compile(r"^\*\*Verdict:\s*(MUST_FIX|CLEAN)\*\*", re.M)
VALIDATOR_LINE = "<!-- validator:"


class Refused(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code = code


def added_lines(diff_text):
    """{path: set of new-file line numbers the diff adds} from a unified diff."""
    out, path, line = {}, None, 0
    for raw in diff_text.splitlines():
        if raw.startswith("+++ "):
            name = raw[4:].strip()
            path = None if name == "/dev/null" else name[2:] if name.startswith("b/") else name
            continue
        if raw.startswith("--- ") or raw.startswith("diff --git"):
            continue
        m = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", raw)
        if m:
            line = int(m.group(1))
            continue
        if path is None or line == 0:
            continue
        if raw.startswith("+"):
            out.setdefault(path, set()).add(line)
            line += 1
        elif raw.startswith("-") or raw.startswith("\\"):
            pass
        else:
            line += 1
    return out


def secrets_in(texts, secret_values):
    """Names of what leaked: 'env:<NAME>' for a secret's value, 'token-shape' for a token-shaped string."""
    found = []
    for name, value in secret_values:
        if value and len(value) >= 8 and any(value in t for t in texts):
            found.append(name if ":" in name else "env:" + name)
    if any(TOKEN_RE.search(t) for t in texts):
        found.append("token-shape")
    return found


def unsafe_entries(d):
    """The paths under d, relative to it, that are neither a regular file nor a directory: a symbolic link (listed,
    not followed), a FIFO, a socket, a device."""
    bad = []
    for root, dirs, files in os.walk(d):
        for name in dirs + files:
            p = os.path.join(root, name)
            mode = os.lstat(p).st_mode
            if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                bad.append(os.path.relpath(p, d))
    return sorted(bad)


def secret_file_values(path):
    """[(name, value)] for every string of 16 or more characters in a JSON file, at any depth; [] when it is missing."""
    try:
        text = open(path, encoding="utf-8").read()
    except OSError:
        return []
    try:
        stack, found = [json.loads(text)], []
    except ValueError:
        return [("file:" + os.path.basename(path), text.strip())]
    while stack:
        v = stack.pop()
        if isinstance(v, dict):
            stack.extend(v.values())
        elif isinstance(v, list):
            stack.extend(v)
        elif isinstance(v, str) and len(v) >= 16:
            found.append(("file:" + os.path.basename(path), v))
    return found


def keep_comments(comments, added):
    """The inline comments GitHub will accept, at most MAX_COMMENTS, and how many were dropped."""
    kept, dropped = [], 0
    for c in comments:
        ok = (
            isinstance(c, dict)
            and isinstance(c.get("path"), str)
            and isinstance(c.get("line"), int)
            and isinstance(c.get("body"), str)
            and c["body"].strip()
        )
        lines = added.get(c.get("path"), set()) if ok else set()
        start = c.get("start_line", c.get("line")) if ok else None
        if ok and not isinstance(start, int):
            ok = False
        if ok and all(n in lines for n in range(start, c["line"] + 1)) and start <= c["line"]:
            item = {"path": c["path"], "line": c["line"], "side": "RIGHT", "body": c["body"]}
            if start != c["line"]:
                item["start_line"] = start
                item["start_side"] = "RIGHT"
            kept.append(item)
        else:
            dropped += 1
    dropped += max(0, len(kept) - MAX_COMMENTS)
    return kept[:MAX_COMMENTS], dropped


def keep_issues(items, base):
    """Well-formed issue candidates, at most MAX_ISSUES, and how many were dropped."""
    kept, dropped = [], 0
    for it in items if isinstance(items, list) else []:
        ev = it.get("evidence") if isinstance(it, dict) else None
        ok = (
            isinstance(it, dict)
            and isinstance(it.get("title"), str) and 0 < len(it["title"].strip()) <= 120
            and isinstance(it.get("body"), str) and 0 < len(it["body"].strip()) <= 4000
            and isinstance(ev, list) and ev
            and all(isinstance(e, dict) and isinstance(e.get("path"), str) and isinstance(e.get("line"), int)
                    and ".." not in e["path"].split("/") and not e["path"].startswith("/") for e in ev)
        )
        if ok and base is not None:
            ok = all(os.path.isfile(os.path.join(base, e["path"])) for e in ev)
        if ok:
            kept.append({"title": it["title"].strip(), "body": it["body"].strip(),
                         "evidence": [{"path": e["path"], "line": e["line"]} for e in ev]})
        else:
            dropped += 1
    dropped += max(0, len(kept) - MAX_ISSUES)
    return kept[:MAX_ISSUES], dropped


def footer(head, verdict, body, mode, ledger=None, run=None, event=None, tools=None, carried_from=None):
    """The marker the next run reads. `event` is what was posted (a carry needs an APPROVE), `tools` is carry.py's
    reviewer-id, `carried_from` the reviewed head a `carry` review stands on."""
    must = len(re.findall(r"^#### \d+\.", body, re.M))
    warn = len(re.findall(r"^⚠", body, re.M))
    tail = (f" carried_from={carried_from}" if carried_from else "") + (f" event={event}" if event else "")
    tail += (f" ledger={ledger}" if ledger else "") + (f" run={run}" if run and str(run).isdigit() else "")
    tail += f" tools={tools}" if tools and re.fullmatch(r"[0-9a-f]{16}", str(tools)) else ""
    return f"<!-- {MARKER} head_sha={head} verdict={verdict} must_fix={must} warn={warn} mode={mode}{tail} -->"


def parse_footer(text):
    """{key: value} of the last `<!-- bloomery-review ... -->` marker in text, or None when there is none."""
    found = re.findall(rf"<!-- {MARKER} ([^>]*?) -->", text or "")
    if not found:
        return None
    f = dict(kv.split("=", 1) for kv in found[-1].split() if "=" in kv)
    return f if f.get("head_sha") and f.get("verdict") else None


def usage_marker(paths):
    parts = []
    for p in paths:
        try:
            u = json.load(open(p, encoding="utf-8"))
        except (OSError, ValueError):
            parts.append(f"{os.path.basename(p)}: unreadable")
            continue
        tools = ",".join(f"{k}:{v}" for k, v in (u.get("tools") or {}).items()) or "none"
        cost = u.get("cost_usd")
        parts.append(f"{u.get('role')}: turns={u.get('num_turns')} cost_usd={cost if cost is None else round(cost, 4)} "
                     f"ctx_max={u.get('context_max')} ctx_sum={u.get('context_sum')} "
                     f"out={(u.get('tokens') or {}).get('output')} tools={tools} denied={u.get('denied')} "
                     f"engine={u.get('engine', 'claude')}")
    return "<!-- bloomery-usage " + "; ".join(parts) + " -->" if parts else ""


def pr_paths(files_path, diff):
    """The pull request's file paths: from the `pulls/<n>/files` list (`status\t+a\t-d\tpath` lines) when given,
    else from the diff's headers (an incremental diff holds only the new commits)."""
    if files_path and os.path.exists(files_path):
        return [ln.split("\t")[-1] for ln in open(files_path, encoding="utf-8").read().splitlines() if ln.strip()]
    return sorted(set(re.findall(r"^\+\+\+ b/(\S+)", diff, re.M) + re.findall(r"^--- a/(\S+)", diff, re.M)))


def device_dirs(root):
    """`crates/<dir>/` for every workspace crate whose dependency closure reaches cuda-core."""
    root = os.path.abspath(root)
    r = subprocess.run([sys.executable, os.path.join(root, "tools", "recipes.py"), "pure-crates"], cwd=root,
                       capture_output=True, text=True)
    names = {ln.split()[1] for ln in r.stdout.splitlines()
             if ln.startswith("rejected") and "reaches the device root" in ln}
    dirs = []
    for d in sorted(os.listdir(os.path.join(root, "crates"))):
        toml = os.path.join(root, "crates", d, "Cargo.toml")
        m = re.search(r'^name\s*=\s*"([^"]+)"', open(toml).read(), re.M) if os.path.exists(toml) else None
        if m and m.group(1) in names:
            dirs.append(f"crates/{d}/")
    if r.returncode or not dirs:
        raise Refused(70, f"device: tools/recipes.py pure-crates gave rc={r.returncode} and {len(dirs)} device "
                          f"crate(s): {r.stderr.strip()[:300]}")
    return dirs


def card_label(paths, dirs, out):
    """(label or None, the line for the review) from the changed paths and the reviewer's gpu.json."""
    touched = [p for p in paths if p.startswith(tuple(dirs))]
    if not dirs or not touched:
        return None, None
    gp = os.path.join(out, "gpu.json")
    evidence = False
    if os.path.exists(gp):
        try:
            g = json.load(open(gp, encoding="utf-8"))
        except ValueError as e:
            raise Refused(65, f"gpu.json is not JSON: {e}")
        if not isinstance(g, dict) or not isinstance(g.get("evidence"), bool):
            raise Refused(65, 'gpu.json is not {"evidence": true|false, ...}')
        evidence = g["evidence"]
    where = ", ".join(touched[:3]) + (" …" if len(touched) > 3 else "")
    if evidence:
        return "gpu-evidence", (f"Card check: `gpu-evidence`. This pull request changes {where} and carries a gate run "
                                "on the author's card, which counts as verified.")
    return "needs-gpu", (f"Card check: `needs-gpu`. This pull request changes {where} and carries no gate run on a "
                         "card (CONTRIBUTING's evidence list); a maintainer adds `gpu-verified` after one.")


def event_for(verdict, decision, body, paths, carried=False):
    """("APPROVE" | "REQUEST_CHANGES" | "COMMENT", why a CLEAN review does not approve or None). A carried review
    (`post.py carry`) had no model run, so it has no validator line; every other guard applies to it as to any."""
    if verdict == "MUST_FIX":
        return "REQUEST_CHANGES", None
    if decision != verdict:
        return "COMMENT", "decision.txt disagreed with the verdict line"
    if VALIDATOR_LINE not in body and not carried:
        return "COMMENT", "the validator did not finish"
    guarded = [p for p in paths if p.startswith(APPROVE_GUARD)]
    if guarded:
        return "COMMENT", (f"this pull request changes {', '.join(guarded[:3])}{' …' if len(guarded) > 3 else ''}, "
                           "which changes the bot, CI or the toolchain and needs a person's approval")
    return "APPROVE", None


def check(out, diff_path, pr_json, secret_env, usage_paths=(), base=None, files_path=None, device_path=None,
          freeze_path=None, secret_files=()):
    bad = unsafe_entries(out)
    if bad:
        raise Refused(3, f"the review output holds what is not a regular file ({', '.join(bad)}); nothing is posted")
    try:
        body = open(os.path.join(out, "review.md"), encoding="utf-8").read()
        decision = open(os.path.join(out, "decision.txt"), encoding="utf-8").read().strip()
        pr = json.load(open(pr_json, encoding="utf-8"))
        diff = open(diff_path, encoding="utf-8", errors="replace").read()
    except (OSError, ValueError) as e:
        raise Refused(65, f"unreadable review output: {e}")
    comments_path = os.path.join(out, "comments.json")
    comments_text = ""
    comments = []
    if os.path.exists(comments_path):
        comments_text = open(comments_path, encoding="utf-8").read()
        try:
            comments = json.loads(comments_text or "[]")
        except ValueError as e:
            raise Refused(65, f"comments.json is not JSON: {e}")
        if not isinstance(comments, list):
            raise Refused(65, "comments.json is not a JSON array")
    issues_path = os.path.join(out, "issues.json")
    issues_text = open(issues_path, encoding="utf-8").read() if os.path.exists(issues_path) else ""
    try:
        issues = json.loads(issues_text or "[]")
    except ValueError as e:
        raise Refused(65, f"issues.json is not JSON: {e}")
    leaked = secrets_in([body, decision, comments_text, issues_text],
                        [(n, os.environ.get(n, "")) for n in secret_env]
                        + [kv for p in secret_files for kv in secret_file_values(p)])
    if leaked:
        raise Refused(3, "the review output carries a credential (" + ", ".join(leaked) + "); nothing is posted")
    if not body.strip():
        raise Refused(65, "review.md is empty")
    m = VERDICT_RE.search(body)
    if not m:
        raise Refused(65, "review.md has no '**Verdict: MUST_FIX|CLEAN**' line")
    verdict = m.group(1)
    notes = []
    if decision != verdict:
        notes.append(f"decision.txt said {decision or 'nothing'}; the body's verdict line ({verdict}) is used")
    body = re.sub(rf"(\n*<!-- (?:{MARKER}|bloomery-usage) [^>]*-->)+\s*$", "", body.rstrip())
    body = re.sub(r"\n*<sub>🔨 Shingler[^\n]*</sub>\s*$", "", body)  # a second check (the post job)
    kept, dropped = keep_comments(comments, added_lines(diff))
    if dropped:
        notes.append(f"{dropped} inline suggestion(s) were dropped: not on lines this diff adds, or over the cap")
    issues, idropped = keep_issues(issues, base)
    if idropped:
        notes.append(f"{idropped} issue candidate(s) were dropped: malformed, evidence missing, or over the cap")
    cap = MAX_BODY - len(SIGNATURE) - 100  # the signature still fits under the cap
    if len(body) > cap:
        body = body[:cap].rsplit("\n", 1)[0] + "\n\n(The review was cut at the length cap.)"
    if notes:
        body += "\n\n<sub>" + " · ".join(notes) + "</sub>"
    usage_txt = os.path.join(out, "usage.txt")
    carried = pr.get("mode") == "carry"
    marker = "" if carried else usage_marker(usage_paths)  # a carry ran no model: it has no accounting to keep
    if marker:
        with open(usage_txt, "w", encoding="utf-8") as f:
            f.write(marker + "\n")
    elif os.path.exists(usage_txt):
        marker = open(usage_txt, encoding="utf-8").read().strip()
    paths = pr_paths(files_path, diff)
    if carried and not re.fullmatch(r"[0-9a-f]{40}", pr.get("carried_from") or ""):
        raise Refused(65, "pr.json says mode carry without the head it carries from (carried_from)")
    event, why = event_for(verdict, decision, body, paths, carried)
    dirs = open(device_path, encoding="utf-8").read().split() if device_path and os.path.exists(device_path) else []
    label, card_line = card_label(paths, dirs, out)
    with open(os.path.join(out, "labels.json"), "w", encoding="utf-8") as f:
        json.dump({"set": label}, f)
    freeze = open(freeze_path, encoding="utf-8").read().strip() if freeze_path and os.path.exists(freeze_path) else ""
    if event == "APPROVE" and label == "needs-gpu" and freeze:
        event, why = "COMMENT", (f"a release freeze is on ({freeze}), and a merge of a `needs-gpu` change now would "
                                 "change the tree under the release batch")
    if card_line and "\n<sub>Card check:" not in body:
        body += f"\n\n<sub>{card_line}</sub>"
    # A second check never raises the event: the first one saw the reviewer's own decision.txt, which it then rewrote.
    ev_path = os.path.join(out, "event.txt")
    if os.path.exists(ev_path) and open(ev_path, encoding="utf-8").read().strip() == "COMMENT":
        event = "COMMENT"
    if "\n<sub>Not approved:" in body:
        why = None
    if why:
        body += f"\n\n<sub>Not approved: {why}.</sub>"
    body += "\n\n" + SIGNATURE.format(repo=pr.get("repo") or "midagedev/bloomery")
    body += "\n\n" + footer(pr["head_sha"], verdict, body, pr.get("mode", "full"), pr.get("ledger"), pr.get("run"),
                            event, pr.get("tools"), pr.get("carried_from") if carried else None)
    if marker:
        body += "\n" + marker
    with open(os.path.join(out, "review.md"), "w", encoding="utf-8") as f:
        f.write(body + "\n")
    with open(os.path.join(out, "decision.txt"), "w", encoding="utf-8") as f:
        f.write(verdict + "\n")
    with open(os.path.join(out, "event.txt"), "w", encoding="utf-8") as f:
        f.write(event + "\n")
    with open(os.path.join(out, "comments.json"), "w", encoding="utf-8") as f:
        json.dump(kept, f, indent=1)
    with open(issues_path, "w", encoding="utf-8") as f:
        json.dump(issues, f, indent=1)
    with open(os.path.join(out, "head_sha.txt"), "w", encoding="utf-8") as f:
        f.write(pr["head_sha"] + "\n")
    with open(os.path.join(out, "base_sha.txt"), "w", encoding="utf-8") as f:
        # Issue evidence is verified in BASE_TREE, so its links point at the commit that tree was checked out at.
        f.write((pr.get("tree_sha") or pr.get("base_sha", "")) + "\n")
    return verdict, len(kept), notes


CARD_LINE_RE = re.compile(r"<sub>Card check: `(needs-gpu|gpu-evidence)`")


def carry_review(carry_path, prev_path, out):
    """Write OUT for a carried approval: the files `check` reads from a reviewer, from carry.json (tools/review/carry.py
    decide) and the review it carries. The carried review's own marker must say it approved; carry.py decides that
    too, and this refusal keeps a wiring mistake from posting an approval that never was one."""
    try:
        carry = json.load(open(carry_path, encoding="utf-8"))
        prev = open(prev_path, encoding="utf-8").read()
    except (OSError, ValueError) as e:
        raise Refused(65, f"unreadable input: {e}")
    f = parse_footer(prev)
    if carry.get("mode") != "carry":
        raise Refused(65, f"carry.json says mode {carry.get('mode')!r}")
    if not f or f.get("head_sha") != carry.get("prev_head"):
        raise Refused(65, "the carried review has no marker for the head carry.json names")
    if f.get("verdict") != "CLEAN" or f.get("event") != "APPROVE":
        raise Refused(65, f"the carried review was {f.get('verdict')}/{f.get('event') or 'no event'}, not an "
                          "approval; only an approval carries")
    card = CARD_LINE_RE.search(prev)
    prev8, head8, n = carry["prev_head"][:8], carry["head"][:8], carry.get("commits", 0)
    moved = carry.get("main_files")
    if carry.get("main_note"):
        base_line = carry["main_note"]
    elif moved:
        base_line = ("The base moved and also changed " + ", ".join(f"`{x}`" for x in moved[:10])
                     + (" …" if len(moved) > 10 else "") + ", which this pull request changes too. Each of its hunks "
                     "still applies with the same context lines.")
    elif moved is not None:
        base_line = "The base moved, and changed none of the files this pull request changes."
    else:
        base_line = "The base did not move."
    body = (f"**Verdict: CLEAN**: carried from `{prev8}`, not re-reviewed.\n\n"
            f"The approval of `{prev8}` stands for `{head8}`. This pull request's {n} commit{'s' if n != 1 else ''} are "
            "the same patches in the same order: `git patch-id --verbatim` (the changed lines, the lines around them "
            "and whitespace) and the subject are equal for every commit. That review ran under the reviewer "
            "instructions, tools and slag ledger that review this head, and it approved.\n\n"
            f"{base_line}\n\n"
            "A carry covers each patch's own lines and the lines around them. What the base changed elsewhere is "
            "covered by CI on this head and by the landing gates.\n")
    os.makedirs(out, exist_ok=True)
    for name, text in (("review.md", body), ("decision.txt", "CLEAN\n"), ("comments.json", "[]"),
                       ("issues.json", "[]"), ("gpu.json", json.dumps({"evidence": bool(card and card.group(1) == "gpu-evidence")}))):
        with open(os.path.join(out, name), "w", encoding="utf-8") as fh:
            fh.write(text)
    return body


def api(method, url, token, payload=None):
    req = urllib.request.Request(
        url,
        data=None if payload is None else json.dumps(payload).encode(),
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def issue_body(it, repo, base_sha, pr):
    links = "\n".join(f"- [{e['path']}:{e['line']}](https://github.com/{repo}/blob/{base_sha}/{e['path']}#L{e['line']})"
                      for e in it["evidence"])
    return (f"{it['body']}\n\nEvidence (at the base of #{pr}):\n{links}\n\n"
            f"<sub>Filed by Shingler, the review bot, while reviewing #{pr}; the defect is outside that pull request.</sub>\n"
            f"<!-- {MARKER}-issue pr={pr} -->")


def file_issues(out, repo, pr, token, call=api):
    items = json.load(open(os.path.join(out, "issues.json"), encoding="utf-8"))
    filed = []
    if items:
        base_sha = open(os.path.join(out, "base_sha.txt"), encoding="utf-8").read().strip()
        existing = {}
        page = 1
        while True:
            got = call("GET", f"https://api.github.com/repos/{repo}/issues?labels={ISSUE_LABEL}&state=all"
                              f"&per_page=100&page={page}", token)
            for i in got:
                existing[i["title"].strip().lower()] = i["number"]
            if len(got) < 100:
                break
            page += 1
        for it in items:
            n = existing.get(it["title"].lower())
            if n is not None:
                filed.append(f"existing #{n}")
                continue
            r = call("POST", f"https://api.github.com/repos/{repo}/issues", token,
                     {"title": it["title"], "body": issue_body(it, repo, base_sha, pr), "labels": [ISSUE_LABEL]})
            filed.append(f"#{r['number']}")
    with open(os.path.join(out, "filed.txt"), "w", encoding="utf-8") as f:
        f.write(", ".join(filed) + ("\n" if filed else ""))
    return filed


def post(out, repo, pr, token):
    body = open(os.path.join(out, "review.md"), encoding="utf-8").read()
    filed_path = os.path.join(out, "filed.txt")
    filed = open(filed_path, encoding="utf-8").read().strip() if os.path.exists(filed_path) else ""
    if filed:
        body = body.replace(f"<!-- {MARKER} ", f"Filed outside this PR: {filed}\n\n<!-- {MARKER} ", 1)
    head = open(os.path.join(out, "head_sha.txt"), encoding="utf-8").read().strip()
    comments = json.load(open(os.path.join(out, "comments.json"), encoding="utf-8"))
    url = f"https://api.github.com/repos/{repo}/pulls/{pr}/reviews"
    ev_path = os.path.join(out, "event.txt")
    event = open(ev_path, encoding="utf-8").read().strip() if os.path.exists(ev_path) else "COMMENT"
    if event not in ("APPROVE", "REQUEST_CHANGES", "COMMENT"):
        raise Refused(65, f"event.txt says {event!r}; only APPROVE, REQUEST_CHANGES or COMMENT is posted")
    payload = {"commit_id": head, "event": event, "body": body, "comments": comments}
    try:
        r = api("POST", url, token, payload)
    except urllib.error.HTTPError as e:
        if e.code != 422 or not comments:
            raise
        payload["comments"] = []
        payload["body"] = body.replace(
            f"<!-- {MARKER} ", "<sub>GitHub refused the inline suggestions; they are left out.</sub>\n\n"
            f"<!-- {MARKER} ", 1)
        r = api("POST", url, token, payload)
    return r.get("html_url", "")


def freeze_issue(repo, token, call=None):
    """`#<n>` of an open issue labelled FREEZE_LABEL that the repository owner opened, or ""."""
    call = call or api
    got = call("GET", f"https://api.github.com/repos/{repo}/issues?labels={FREEZE_LABEL}&state=open&per_page=20",
               token)
    owner = repo.split("/")[0]
    hits = [i["number"] for i in got if (i.get("user") or {}).get("login") == owner]
    return f"#{min(hits)}" if hits else ""


def apply_labels(out, repo, pr, token, call=None):
    """Set the bot's one card label (or none) on the pull request; never touch any other label."""
    call = call or api
    want = (json.load(open(os.path.join(out, "labels.json"), encoding="utf-8")) or {}).get("set")
    if want is not None and want not in CARD_LABELS:
        raise Refused(65, f"labels.json sets {want!r}; the bot sets only {', '.join(CARD_LABELS)}")
    have = {x["name"] for x in call("GET", f"https://api.github.com/repos/{repo}/issues/{pr}/labels", token)}
    done = []
    for name in CARD_LABELS:
        if name == want and name not in have:
            try:
                color, desc = CARD_LABELS[name]
                call("POST", f"https://api.github.com/repos/{repo}/labels", token,
                     {"name": name, "color": color, "description": desc})
            except urllib.error.HTTPError as e:
                if e.code != 422:  # 422: the label exists
                    raise
            call("POST", f"https://api.github.com/repos/{repo}/issues/{pr}/labels", token, {"labels": [name]})
            done.append(f"+{name}")
        elif name != want and name in have:
            call("DELETE", f"https://api.github.com/repos/{repo}/issues/{pr}/labels/{name}", token)
            done.append(f"-{name}")
    return done


def self_test():
    assert footer("h", "CLEAN", "", "full").endswith("mode=full -->")
    assert footer("h", "CLEAN", "", "full", "0a1b2c").endswith("mode=full ledger=0a1b2c -->")
    assert footer("h", "CLEAN", "", "full", "0a1b2c", "123").endswith("ledger=0a1b2c run=123 -->")
    assert footer("h", "CLEAN", "", "full", None, "1 x").endswith("mode=full -->")
    assert footer("h", "CLEAN", "", "carry", "L", "5", "APPROVE", "0123456789abcdef", "c" * 40).endswith(
        f"mode=carry carried_from={'c' * 40} event=APPROVE ledger=L run=5 tools=0123456789abcdef -->")
    assert "tools=" not in footer("h", "CLEAN", "", "full", tools="not-a-digest")
    assert parse_footer("x\n" + footer("h" * 40, "CLEAN", "", "full", "L", "5", "APPROVE", "0123456789abcdef")) == {
        "head_sha": "h" * 40, "verdict": "CLEAN", "must_fix": "0", "warn": "0", "mode": "full", "event": "APPROVE",
        "ledger": "L", "run": "5", "tools": "0123456789abcdef"}
    assert parse_footer("no marker") is None and parse_footer("<!-- bloomery-review-issue pr=5 -->") is None
    assert "event" not in parse_footer("<!-- bloomery-review head_sha=ab verdict=CLEAN must_fix=0 warn=0 mode=full -->")
    diff = (
        "diff --git a/crates/x/src/a.rs b/crates/x/src/a.rs\n--- a/crates/x/src/a.rs\n+++ b/crates/x/src/a.rs\n"
        "@@ -10,3 +10,4 @@ fn f() {\n ctx\n-old\n+new1\n+new2\n ctx2\n"
        "diff --git a/tools/n.sh b/tools/n.sh\n--- /dev/null\n+++ b/tools/n.sh\n@@ -0,0 +1,2 @@\n+a\n+b\n"
        "diff --git a/gone.rs b/gone.rs\n--- a/gone.rs\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n"
    )
    got = added_lines(diff)
    assert got == {"crates/x/src/a.rs": {11, 12}, "tools/n.sh": {1, 2}}, got

    kept, dropped = keep_comments(
        [
            {"path": "crates/x/src/a.rs", "line": 11, "body": "b"},
            {"path": "crates/x/src/a.rs", "line": 12, "start_line": 11, "body": "b"},
            {"path": "crates/x/src/a.rs", "line": 10, "body": "context line, not added"},
            {"path": "crates/x/src/a.rs", "line": 13, "start_line": 11, "body": "runs past the hunk"},
            {"path": "nope.rs", "line": 1, "body": "b"},
            {"path": "tools/n.sh", "line": "1", "body": "line is a string"},
            "not an object",
        ],
        got,
    )
    assert [(c["line"], c.get("start_line")) for c in kept] == [(11, None), (12, 11)], kept
    assert dropped == 5, dropped
    many, over = keep_comments([{"path": "tools/n.sh", "line": 1, "body": "x"}] * 12, got)
    assert len(many) == MAX_COMMENTS and over == 2

    assert secrets_in(["fine"], [("K", "")]) == []
    assert secrets_in(["key sk-ant-api03-abcdefghijklmnopqrst"], []) == ["token-shape"]
    assert secrets_in(["x MYSECRETVALUE y"], [("K", "MYSECRETVALUE")]) == ["env:K"]
    assert secrets_in(["ghp_" + "a" * 36], []) == ["token-shape"]
    assert secrets_in(["Bearer eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkw.sig"], []) == ["token-shape"]
    jwt = ".".join(["eyJhbGciOiJSUzI1NiJ9", "eyJzdWIiOiIxMjM0NTY3ODkw", "c2lnbmF0dXJl"])
    assert TOKEN_RE.sub("-", f"a {jwt} b") == "a - b"
    assert secrets_in(["k sk-proj-" + "a" * 30, "a disk-usage-reporting-tool-v2"], []) == ["token-shape"]
    assert secrets_in(["a disk-usage-reporting-tool-v2 and task-runner-for-every-build"], []) == []

    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "out")
        os.mkdir(out)
        dp, pj = os.path.join(d, "diff.patch"), os.path.join(d, "pr.json")
        open(dp, "w").write(diff)
        json.dump({"head_sha": "abc123", "mode": "full"}, open(pj, "w"))

        def write(body, decision, comments=None):  # a fresh review job's output
            if os.path.exists(os.path.join(out, "event.txt")):
                os.remove(os.path.join(out, "event.txt"))
            open(os.path.join(out, "review.md"), "w").write(body)
            open(os.path.join(out, "decision.txt"), "w").write(decision)
            cp = os.path.join(out, "comments.json")
            if comments is None:
                if os.path.exists(cp):
                    os.remove(cp)
            else:
                open(cp, "w").write(comments)

        body = "**Verdict: MUST_FIX**: one.\n\n### Must fix\n#### 1. t\nx\n\n### Non-blocking\n⚠ a\n⚠ b\n"
        write(body, "CLEAN", json.dumps([{"path": "tools/n.sh", "line": 2, "body": "s"},
                                         {"path": "tools/n.sh", "line": 9, "body": "s"}]))
        verdict, n, notes = check(out, dp, pj, [])
        assert (verdict, n) == ("MUST_FIX", 1) and len(notes) == 2, (verdict, n, notes)
        posted = open(os.path.join(out, "review.md")).read()
        assert posted.rstrip().endswith(
            "<!-- bloomery-review head_sha=abc123 verdict=MUST_FIX must_fix=1 warn=2 mode=full event=REQUEST_CHANGES -->"), posted
        assert open(os.path.join(out, "decision.txt")).read() == "MUST_FIX\n"
        # A second check (the post job repeats it) keeps one footer and one signature.
        check(out, dp, pj, [])
        again = open(os.path.join(out, "review.md")).read()
        assert again.count("<!-- bloomery-review") == 1 and again.count("🔨 Shingler") == 1, again
        assert open(os.path.join(out, "event.txt")).read() == "REQUEST_CHANGES\n"

        # APPROVE only for a CLEAN verdict the decision agrees with, after the validator, off the guarded paths.
        clean = "**Verdict: CLEAN**: fine.\n\n⚠ a\n<!-- validator: removed=0 downgraded=0 upgraded=0 -->\n"
        fp = os.path.join(d, "files.txt")
        for body_, decision, files, want, why in (
                (clean, "CLEAN", "modified\t+1\t-0\ttools/n.sh\n", "APPROVE", None),
                (clean, "MUST_FIX", "modified\t+1\t-0\ttools/n.sh\n", "COMMENT", "decision.txt disagreed"),
                (clean.split("<!--")[0], "CLEAN", "modified\t+1\t-0\ttools/n.sh\n", "COMMENT", "validator"),
                (clean, "CLEAN", "modified\t+1\t-0\ttools/n.sh\nmodified\t+1\t-0\t.github/workflows/ci.yml\n",
                 "COMMENT", ".github/workflows/ci.yml"),
                (clean, "CLEAN", None, "APPROVE", None)):
            write(body_, decision)
            if files is None:
                os.remove(fp)
            else:
                open(fp, "w").write(files)
            check(out, dp, pj, [], files_path=fp if files is not None else None)
            got = open(os.path.join(out, "event.txt")).read().strip()
            text = open(os.path.join(out, "review.md")).read()
            assert got == want and ("Not approved:" in text) == bool(why) and (not why or why in text), (got, text)
            check(out, dp, pj, [], files_path=fp if files is not None else None)
            assert open(os.path.join(out, "review.md")).read() == text  # a second check changes nothing
        assert pr_paths(None, "--- a/x.rs\n+++ b/x.rs\n--- /dev/null\n+++ b/.github/y\n") == [".github/y", "x.rs"]

        # The workflow passes a root relative to the directory above the tree.
        tree = os.path.join(d, "tree")
        os.makedirs(os.path.join(tree, "tools"))
        os.makedirs(os.path.join(tree, "crates", "foo"))
        open(os.path.join(tree, "tools", "recipes.py"), "w").write('print("rejected foo reaches the device root")\n')
        open(os.path.join(tree, "crates", "foo", "Cargo.toml"), "w").write('name = "foo"\n')
        here = os.getcwd()
        os.chdir(d)
        try:
            assert device_dirs("tree") == ["crates/foo/"]
        finally:
            os.chdir(here)

        # Card labels: only for a change under a device crate; a freeze holds a needs-gpu approval.
        dev, frz = os.path.join(d, "device.txt"), os.path.join(d, "freeze.txt")
        open(dev, "w").write("crates/gpu/\ncrates/app/\n")
        gpu_files = "modified\t+1\t-0\tcrates/gpu/src/a.rs\n"
        for files, gpu, freeze, want_label, want_event in (
                ("modified\t+1\t-0\tcrates/sampler/src/a.rs\n", None, "", None, "APPROVE"),
                (gpu_files, None, "", "needs-gpu", "APPROVE"),
                (gpu_files, '{"evidence": true, "card": "RTX 4090"}', "#99", "gpu-evidence", "APPROVE"),
                (gpu_files, '{"evidence": false}', "#99", "needs-gpu", "COMMENT")):
            write(clean, "CLEAN")
            open(fp, "w").write(files)
            gj = os.path.join(out, "gpu.json")
            if gpu is None:
                if os.path.exists(gj):
                    os.remove(gj)
            else:
                open(gj, "w").write(gpu)
            open(frz, "w").write(freeze)
            for _ in range(2):  # the post job repeats the check
                check(out, dp, pj, [], files_path=fp, device_path=dev, freeze_path=frz)
            text = open(os.path.join(out, "review.md")).read()
            assert json.load(open(os.path.join(out, "labels.json"))) == {"set": want_label}, files
            assert open(os.path.join(out, "event.txt")).read().strip() == want_event, (want_event, text)
            assert text.count("Card check:") == (1 if want_label else 0), text
            assert ("release freeze is on (#99)" in text) == (want_event == "COMMENT"), text
        open(os.path.join(out, "gpu.json"), "w").write('{"evidence": "yes"}')
        try:
            check(out, dp, pj, [], files_path=fp, device_path=dev)
            raise AssertionError("a malformed gpu.json was accepted")
        except Refused as e:
            assert e.code == 65
        os.remove(os.path.join(out, "gpu.json"))
        lc = []

        def lfake(method, url, token, payload=None):
            lc.append((method, url.split("/repos/o/r/")[1], payload))
            return [{"name": "gpu-evidence"}, {"name": "gpu-verified"}] if method == "GET" else {}
        json.dump({"set": "needs-gpu"}, open(os.path.join(out, "labels.json"), "w"))
        assert apply_labels(out, "o/r", "5", "t", lfake) == ["+needs-gpu", "-gpu-evidence"], lc
        assert not any("gpu-verified" in u for _, u, _ in lc[1:]), lc
        json.dump({"set": "gpu-verified"}, open(os.path.join(out, "labels.json"), "w"))
        try:
            apply_labels(out, "o/r", "5", "t", lfake)
            raise AssertionError("the bot set a maintainer's label")
        except Refused:
            pass
        assert freeze_issue("o/r", "t", lambda *a: [{"number": 4, "user": {"login": "mallory"}},
                                                     {"number": 9, "user": {"login": "o"}}]) == "#9"
        assert freeze_issue("o/r", "t", lambda *a: [{"number": 4, "user": {"login": "mallory"}}]) == ""
        write(body, "CLEAN")
        check(out, dp, pj, [])

        up = os.path.join(d, "reviewer.usage.json")
        json.dump({"role": "reviewer", "num_turns": 7, "cost_usd": 0.51234, "context_max": 90000,
                   "context_sum": 400000, "tokens": {"output": 9000}, "tools": {"Read": 20, "Grep": 9},
                   "denied": 0}, open(up, "w"))
        check(out, dp, pj, [], [up])
        check(out, dp, pj, [])  # the post job has no usage files; usage.txt keeps the marker
        posted = open(os.path.join(out, "review.md")).read()
        assert posted.count("<!-- bloomery-usage reviewer: turns=7 cost_usd=0.5123 ctx_max=90000") == 1, posted
        assert posted.count("<!-- bloomery-review") == 1 and "tools=Read:20,Grep:9" in posted, posted
        assert "denied=0 engine=claude -->" in posted, posted

        # A carried approval (tools/review/carry.py): no model ran, so there is no validator line; every other guard
        # is recomputed on the current files and freeze, and only an approval carries.
        ph, nh = "a" * 40, "b" * 40
        cout = os.path.join(d, "cout")
        cj, pv, cpr = os.path.join(d, "carry.json"), os.path.join(d, "prev.md"), os.path.join(d, "cpr.json")
        cfreeze = os.path.join(d, "cfreeze.txt")

        def prev_review(verdict="CLEAN", event="APPROVE", head=ph, card="gpu-evidence"):
            return (f"**Verdict: {verdict}**: fine.\n\n<sub>Card check: `{card}`. x</sub>\n\n"
                    + footer(head, verdict, "", "full", "0a1b2c", "9", event, "0123456789abcdef") + "\n")

        def carry_json(**kw):
            json.dump({"mode": "carry", "prev_head": ph, "head": nh, "commits": 3, "main_files": ["justfile"], **kw},
                      open(cj, "w"))

        def carried_pr(**kw):
            json.dump({"head_sha": nh, "mode": "carry", "carried_from": ph, "ledger": "0a1b2c", "run": "10",
                       "tools": "0123456789abcdef", **kw}, open(cpr, "w"))

        open(pv, "w").write(prev_review())
        carry_json()
        carried_pr()
        carry_review(cj, pv, cout)
        assert open(os.path.join(cout, "decision.txt")).read() == "CLEAN\n"
        assert json.load(open(os.path.join(cout, "gpu.json"))) == {"evidence": True}
        assert json.load(open(os.path.join(cout, "comments.json"))) == [] == json.load(open(os.path.join(cout, "issues.json")))
        for files, frz, want in (("modified\t+1\t-0\ttools/n.sh\n", "", "APPROVE"),
                                 ("modified\t+1\t-0\ttools/n.sh\nmodified\t+1\t-0\t.github/review/reviewer.md\n", "", "COMMENT"),
                                 ("modified\t+1\t-0\tcrates/gpu/src/a.rs\n", "#99", "APPROVE")):
            open(fp, "w").write(files)
            open(cfreeze, "w").write(frz)
            for _ in range(2):  # the post job repeats the check
                check(cout, dp, cpr, [], [os.path.join(d, "no-such.usage.json")], files_path=fp, device_path=dev, freeze_path=cfreeze)
            text = open(os.path.join(cout, "review.md")).read()
            assert "bloomery-usage" not in text, text
            assert open(os.path.join(cout, "event.txt")).read().strip() == want, (files, text)
            assert text.count("carried from `aaaaaaaa`") == 1 and "validator" not in text, text
            assert text.rstrip().endswith(f"mode=carry carried_from={ph} event={want} ledger=0a1b2c run=10 "
                                          "tools=0123456789abcdef -->"), text
            os.remove(os.path.join(cout, "event.txt"))
        # the carried review's card evidence is inherited, so a freeze holds only a needs-gpu one
        open(pv, "w").write(prev_review(card="needs-gpu"))
        carry_review(cj, pv, cout)
        assert json.load(open(os.path.join(cout, "gpu.json"))) == {"evidence": False}
        open(cfreeze, "w").write("#99")
        check(cout, dp, cpr, [], files_path=fp, device_path=dev, freeze_path=cfreeze)
        assert open(os.path.join(cout, "event.txt")).read().strip() == "COMMENT"
        assert "release freeze is on (#99)" in open(os.path.join(cout, "review.md")).read()
        for what, prev, kw in (("a MUST_FIX review", prev_review("MUST_FIX", "REQUEST_CHANGES"), {}),
                               ("a CLEAN review that only commented", prev_review("CLEAN", "COMMENT"), {}),
                               ("an old marker with no event", "**Verdict: CLEAN**\n<!-- bloomery-review head_sha=" + ph
                                + " verdict=CLEAN must_fix=0 warn=0 mode=full -->\n", {}),
                               ("a review of another head", prev_review(head="c" * 40), {}),
                               ("no marker", "**Verdict: CLEAN**: fine.\n", {}),
                               ("a decision that is not a carry", prev_review(), {"mode": "interdiff"})):
            open(pv, "w").write(prev)
            carry_json(**kw)
            try:
                carry_review(cj, pv, os.path.join(d, "cout2"))
                raise AssertionError(f"{what} was carried")
            except Refused as e:
                assert e.code == 65, (what, e)
        assert not os.path.exists(os.path.join(d, "cout2")), "a refused carry left output behind"
        carried_pr(carried_from="")
        try:
            check(cout, dp, cpr, [], files_path=fp)
            raise AssertionError("mode carry without carried_from was accepted")
        except Refused as e:
            assert e.code == 65
        open(pv, "w").write(prev_review())
        carry_json()
        assert main(["post.py", "carry", "--carry", cj, "--prev-review", pv, "--out", os.path.join(d, "cout3")]) == 0
        assert main(["post.py", "carry", "--carry", cj, "--prev-review", os.path.join(d, "nope.md"), "--out", d]) == 65
        assert main(["post.py", "carry", "--carry", cj]) == 64

        base = os.path.join(d, "base")
        os.makedirs(os.path.join(base, "crates"))
        open(os.path.join(base, "crates", "a.rs"), "w").write("x")
        good = {"title": "gguf: a NaN block quantizes to code 0", "body": "b", "evidence": [{"path": "crates/a.rs", "line": 3}]}
        open(os.path.join(out, "issues.json"), "w").write(json.dumps([
            good, {**good, "title": "missing file", "evidence": [{"path": "crates/nope.rs", "line": 1}]},
            {**good, "evidence": [{"path": "../etc/passwd", "line": 1}]}, {**good, "title": ""},
            {**good, "title": "third"}, {**good, "title": "fourth"}]))
        check(out, dp, pj, [], base=base)
        assert [i["title"] for i in json.load(open(os.path.join(out, "issues.json")))] == [good["title"], "third"]

        calls = []

        def fake(method, url, token, payload=None):
            calls.append((method, url, payload))
            if method == "GET":
                return [{"title": "Third", "number": 7}]
            return {"number": 41}

        assert file_issues(out, "o/r", "5", "t", fake) == ["#41", "existing #7"]
        assert open(os.path.join(out, "filed.txt")).read() == "#41, existing #7\n"
        assert calls[1][2]["labels"] == [ISSUE_LABEL] and "<!-- bloomery-review-issue pr=5 -->" in calls[1][2]["body"]
        open(os.path.join(out, "issues.json"), "w").write("[]")
        assert file_issues(out, "o/r", "5", "t", fake) == [] and len(calls) == 2

        # post sends the event check decided, and refuses any other.
        open(os.path.join(out, "head_sha.txt"), "w").write("abc123\n")
        real_api, g = api, globals()
        g["api"] = fake
        try:
            for ev in ("APPROVE", "REQUEST_CHANGES", "COMMENT"):
                open(os.path.join(out, "event.txt"), "w").write(ev + "\n")
                post(out, "o/r", "5", "t")
                assert calls[-1][0] == "POST" and calls[-1][2]["event"] == ev, calls[-1]
            open(os.path.join(out, "event.txt"), "w").write("DISMISS\n")
            try:
                post(out, "o/r", "5", "t")
                raise AssertionError("an unknown event was posted")
            except Refused:
                pass
        finally:
            g["api"] = real_api

        write("**Verdict: CLEAN**: fine.\nsk-ant-api03-abcdefghijklmnopqrstuv\n", "CLEAN")
        try:
            check(out, dp, pj, [])
            raise AssertionError("a token-shaped string was not refused")
        except Refused as e:
            assert e.code == 3

        os.environ["POST_SELFTEST_SECRET"] = "s3cr3t-value-123"
        write("**Verdict: CLEAN**: fine.\n", "CLEAN", '[{"path":"tools/n.sh","line":1,"body":"s3cr3t-value-123"}]')
        try:
            check(out, dp, pj, ["POST_SELFTEST_SECRET"])
            raise AssertionError("a secret's value in comments.json was not refused")
        except Refused as e:
            assert e.code == 3

        # A Codex login's auth.json: every long string in it is refused, whatever the key; a short one is not a secret.
        auth = os.path.join(d, "auth.json")
        json.dump({"auth_mode": "chatgpt", "tokens": {"refresh_token": "refresh-refresh-refresh",
                                                      "account_id": "acct-1"}}, open(auth, "w"))
        assert [v for _, v in secret_file_values(auth)] == ["refresh-refresh-refresh"]
        assert secret_file_values(os.path.join(d, "none.json")) == []
        write("**Verdict: CLEAN**: fine.\nrefresh-refresh-refresh\n", "CLEAN")
        try:
            check(out, dp, pj, [], secret_files=[auth])
            raise AssertionError("an auth.json token in review.md was not refused")
        except Refused as e:
            assert e.code == 3 and "file:auth.json" in str(e), e

        write("**Verdict: CLEAN**: fine.\n", "CLEAN")
        os.symlink(auth, os.path.join(out, "notes.md"))
        try:
            check(out, dp, pj, [])
            raise AssertionError("a symbolic link in the output was not refused")
        except Refused as e:
            assert e.code == 3 and "notes.md" in str(e), e
        os.remove(os.path.join(out, "notes.md"))

        write("No verdict line.\n", "CLEAN")
        try:
            check(out, dp, pj, [])
            raise AssertionError("a body without a verdict line was not refused")
        except Refused as e:
            assert e.code == 65

        write("**Verdict: CLEAN**: long.\n" + ("line\n" * 4000), "CLEAN")
        check(out, dp, pj, [])
        assert len(open(os.path.join(out, "review.md")).read()) < MAX_BODY + 400
    print("post.py self-test: ok")


def main(argv):
    if argv[1:] == ["--self-test"]:
        self_test()
        return 0
    if len(argv) < 2 or argv[1] not in ("check", "device", "freeze", "issues", "post", "labels", "carry"):
        print(__doc__, file=sys.stderr)
        return 64
    opts, secret_env, secret_files, usage, rest = {}, [], [], [], argv[2:]
    while rest:
        k = rest.pop(0)
        if not rest:
            print(f"{k} needs a value", file=sys.stderr)
            return 64
        v = rest.pop(0)
        if k == "--secret-env":
            secret_env.append(v)
        elif k == "--secret-file":
            secret_files.append(v)
        elif k == "--usage":
            usage.append(v)
        else:
            opts[k] = v
    try:
        if argv[1] == "check":
            verdict, n, notes = check(opts["--out"], opts["--diff"], opts["--pr-json"], secret_env, usage,
                                      opts.get("--base"), opts.get("--files"), opts.get("--device"),
                                      opts.get("--freeze"), secret_files)
            event = open(os.path.join(opts["--out"], "event.txt"), encoding="utf-8").read().strip()
            label = json.load(open(os.path.join(opts["--out"], "labels.json"), encoding="utf-8")).get("set")
            print(f"check: verdict={verdict} event={event} card={label or '-'} inline={n}"
                  + "".join(f"\n  note: {x}" for x in notes))
        elif argv[1] == "device":
            dirs = device_dirs(opts["--root"])
            open(opts["--out"], "w", encoding="utf-8").write("\n".join(dirs) + "\n")
            print("device:", " ".join(dirs))
        elif argv[1] == "carry":
            carry_review(opts["--carry"], opts["--prev-review"], opts["--out"])
            print(f"carry: wrote {opts['--out']}")
        else:
            token = os.environ.get("GH_TOKEN", "")
            if not token:
                print(f"{argv[1]}: GH_TOKEN is not set", file=sys.stderr)
                return 64
            if argv[1] == "freeze":
                f = freeze_issue(opts["--repo"], token)
                open(opts["--out"], "w", encoding="utf-8").write(f + "\n" if f else "")
                print("freeze:", f or "off")
            elif argv[1] == "labels":
                print("labels:", " ".join(apply_labels(opts["--out"], opts["--repo"], opts["--pr"], token)) or "unchanged")
            elif argv[1] == "issues":
                print("issues:", ", ".join(file_issues(opts["--out"], opts["--repo"], opts["--pr"], token)) or "none")
            else:
                print("posted:", post(opts["--out"], opts["--repo"], opts["--pr"], token))
    except KeyError as e:
        print(f"missing option {e}", file=sys.stderr)
        return 64
    except Refused as e:
        print(f"{argv[1]}: {e}", file=sys.stderr)
        return e.code
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
