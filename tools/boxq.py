#!/usr/bin/env python3
"""boxq.py — the box queue's client, the idle-fill list's reader and the guard on the client's exit codes.

  boxq.py submit --card C --class K --expected-s N --name TAG [--owner O] [--env K=V …] -- CMD…
                                                   post a job to signalbox; its id on stdout
  boxq.py attach ID [--rc-dir DIR]                 follow its out and err, mirror its rc, exit with it
  boxq.py run …                                    submit, then attach (the id goes to stderr: stdout is the job's out)
  boxq.py cancel ID                                post the cancel, print the job's status
  boxq.py fill-list   [--file F] [--justfile J]   print tools/box-fill.tsv's rows, each checked against the justfile
  boxq.py codes-check [--root DIR]                 no other tool or recipe produces exit codes 80, 81, 83, 84, 85
  boxq.py --self-test [CASE …]                     --list-cases names the cases

Client. Every request is one `ssh <host> curl …` to the daemon on the box's loopback; a POST's JSON body goes on ssh's
stdin (no job text ever lands on a command line), a GET has no body and no stdin. CMD… is the job's `bash -c` text:
one word is the text as given, several words are shell-quoted and joined. A job's out and err follow as two
`ssh <host> tail -n +1 -F <path>` children whose stdout and stderr are ours; each remote tail also ends when its ssh's
stdin closes, so a killed client leaves none on the box.
The daemon is polled every BOXQ_POLL_S; a job seen terminal is given one more interval (at most 2 s) for the tails to
flush, then the tails are stopped (a job that was never seen running has its out and err read whole instead, no flush
to guess at) and `<rc-dir>/<id>.rc` (default $BOXQ_RC_DIR, else ~/.cache/bloomery/boxq) is written atomically with
the exit code. A queued job's wait is counted from the job's `submitted` (ISO; no zone means this machine's, which the
box shares), else from the client's first sight of it.
The box's rc is the authority: for done and failed the exit code is the job's rc, so
80, 83 and 85 from the job's command prefix pass through unchanged; a job cancelled by someone else exits with its rc,
else 81, with `boxq: <id> cancelled elsewhere` on stderr; 143 is only our own trapped TERM. A mirror that cannot be
written is a stderr line; the exit code stays the job's.
INT or TERM during run or attach posts the cancel and keeps polling until the job is terminal (at most
BOXQ_CANCEL_WAIT_S), then exits 130 or 143; a job not terminal by then, or a second signal, exits 81 naming the job.
Three polls in a row that cannot reach the daemon exit 84 and leave the job as it is.

  BOXQ_HOST   ssh host (default $BLOOMERY_BOX, else ws)      BOXQ_URL            the daemon (http://127.0.0.1:8030)
  BOXQ_SSH    the ssh command (default ssh)                  BOXQ_POLL_S         poll interval, seconds (5)
  BOXQ_NOTE_S seconds between queue-wait lines (60)          BOXQ_CANCEL_WAIT_S  wait after a cancel (150)

fill-list: a row names a justfile recipe (or `cmd:<command>`), the card lane it holds, where its predicted minutes come
from (`times:<item>` or `given:<minutes>`) and a reason. A row whose recipe the justfile does not hold, an unknown lane,
a malformed minutes source, a duplicate or a row with no reason is refused by name (rc 65); the recipe's justfile
groups are printed beside it, read from the justfile and kept nowhere else.

codes-check: the box queue's client keeps 80 (stale tree), 81 (lost), 83 (dir owned by another tree), 84 (consumer not
alive) and 85 (a job's dir lock not taken). 82 is retired and never reused. The check refuses any script under tools/
or the justfile that produces one of these codes (`exit N`, `sys.exit(N)`, `return N`, `flock -E N`, `SystemExit(N)`);
a script that only tests a code, or names it in a comment, is no collision. The codes also stay clear of those tools/
and the justfile already name.

Exit: the job's rc (run, attach); 0 ok; 64 usage or no such job; 65 a fill row or a code collision, named; 69 an
unreadable file; 81 lost, or still running after the cancel; 84 the daemon is not alive or answers outside the wire;
130/143 a trapped INT/TERM; a daemon refusal exits with the rc it names.
"""

import argparse
import contextlib
import datetime
import http.server
import io
import json
import os
import re
import select
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time

USAGE, DATA, ROOT_BAD = 64, 65, 69
OWN_CODES = (80, 81, 83, 84, 85)
# What tools/ and the justfile already name; the collision check holds the queue's codes apart from these.
OTHER_CODES = frozenset(range(64, 76)) | {77, 78, 97, 124, 137}
LANES = ("3090", "a6000", "both", "any", "none")
CLASSES = ("lead", "round", "sitting", "fill")
STATUSES = ("queued", "running", "done", "failed", "cancelled", "lost")
TERMINAL = ("done", "failed", "cancelled", "lost")
LOST, DOWN = 81, 84
INT_RC, TERM_RC = 130, 143
POLL_FAILS = 3  # polls in a row that cannot reach the daemon before the attach gives up
FLUSH_MAX_S = 2.0  # the longest the tails get, after a terminal status, to read what the job wrote last
NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
ENV_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class Refusal(Exception):
    def __init__(self, code, name, msg):
        super().__init__(msg)
        self.code, self.name, self.msg = code, name, msg


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise Refusal(USAGE, "usage", message)


def here():
    return os.path.dirname(os.path.abspath(__file__))


def justfile_recipes(path):
    """{recipe: [groups]} from a justfile's text: a recipe is a column-0 name whose first unquoted ':' is not ':='."""
    try:
        with open(path) as f:
            lines = f.read().splitlines()
    except OSError as e:
        raise Refusal(ROOT_BAD, "no-justfile", f"{path}: {e}")
    out, attrs = {}, []
    for line in lines:
        if not line or line[0] in " \t":
            continue
        s = line.rstrip()
        if s.startswith("#"):
            continue
        m = re.match(r"^\[(.*)\]$", s)
        if m:
            attrs += re.findall(r"group\s*[(:]\s*['\"]([^'\"]+)['\"]", m.group(1))
            continue
        quote, colon = None, -1
        for i, ch in enumerate(s):
            if quote:
                quote = None if ch == quote else quote
            elif ch in "'\"`":
                quote = ch
            elif ch == ":":
                colon = i
                break
        name = re.match(r"^@?([A-Za-z_][A-Za-z0-9_-]*)", s)
        if (name and colon > 0 and s[colon:colon + 2] != ":="
                and name.group(1) not in ("set", "alias", "export", "import", "mod", "unexport")):
            out[name.group(1)] = attrs
        attrs = []
    return out


def read_fill(path, jf):
    recipes = justfile_recipes(jf)
    try:
        with open(path) as f:
            text = f.read().splitlines()
    except OSError as e:
        raise Refusal(ROOT_BAD, "no-fill-file", f"{path}: {e}")
    rows, seen = [], set()
    for n, line in enumerate(text, 1):
        if not line.strip() or line.startswith("#"):
            continue
        cols = line.split("\t")
        where = f"{path}:{n}"
        if len(cols) != 4:
            raise Refusal(DATA, "bad-fill-row", f"{where}: {len(cols)} columns, the file has 4 (item, lane, minutes, why)")
        item, lane, src, why = cols
        if item in seen:
            raise Refusal(DATA, "bad-fill-row", f"{where}: {item} appears twice")
        seen.add(item)
        if not item.startswith("cmd:") and item not in recipes:
            raise Refusal(DATA, "bad-fill-row", f"{where}: {item} is not a recipe of {jf}")
        if lane not in LANES:
            raise Refusal(DATA, "bad-fill-row", f"{where}: lane {lane!r} is not one of {', '.join(LANES)}")
        m = re.match(r"^(times:\S+|given:([0-9]+(?:\.[0-9]+)?))$", src)
        if not m:
            raise Refusal(DATA, "bad-fill-row", f"{where}: minutes source {src!r} is not times:<item> or given:<minutes>")
        if item.startswith("cmd:") and not m.group(2):
            raise Refusal(DATA, "bad-fill-row", f"{where}: a cmd: row has no recipe to take times from: use given:<minutes>")
        if not why.strip():
            raise Refusal(DATA, "bad-fill-row", f"{where}: no reason")
        rows.append({"item": item, "lane": lane, "src": src, "why": why,
                     "groups": recipes.get(item, [])})
    return rows


def cmd_fill_list(a):
    for r in read_fill(a.file or os.path.join(here(), "box-fill.tsv"), a.justfile or os.path.join(here(), "..", "justfile")):
        print(f"{r['item']}\tlane={r['lane']}\tgroups={','.join(r['groups']) or '-'}\t{r['src']}")

PRODUCE_RES = [re.compile(p) for p in (
    r"(?<![\w$.-])exit\s+(8[01345])\b", r"\bexit\(\s*(8[01345])\s*\)", r"(?<![\w$-])return\s+(8[01345])\b",
    r"\s-E\s*(8[01345])\b", r"\bSystemExit\(\s*(8[01345])\b", r"\bos\._exit\(\s*(8[01345])\b",
)]


def cmd_codes_check(a):
    base = a.root or os.path.join(here(), "..")
    bad = []
    files = [os.path.join(base, "justfile")]
    for dp, _, fns in os.walk(os.path.join(base, "tools")):
        files += [os.path.join(dp, f) for f in fns if f.endswith((".sh", ".py")) or "." not in f]
    for p in sorted(files):
        if os.path.abspath(p) == os.path.abspath(__file__) or not os.path.isfile(p):
            continue
        try:
            with open(p, errors="replace") as f:
                for n, line in enumerate(f, 1):
                    if line.lstrip().startswith("#"):
                        continue
                    for rx in PRODUCE_RES:
                        for m in rx.finditer(line):
                            bad.append(f"{os.path.relpath(p, base)}:{n}: produces {m.group(1)}")
        except OSError as e:
            raise Refusal(ROOT_BAD, "unreadable", f"{p}: {e}")
    if bad or set(OWN_CODES) & OTHER_CODES or len(set(OWN_CODES)) != len(OWN_CODES):
        raise Refusal(DATA, "exit-code-collision", "\n".join(bad) or "the tool's own codes overlap the reserved set")
    print("boxq codes 80, 81, 83-85: no other tool or recipe produces them")


# ---- the queue client ---------------------------------------------------------------------------------------------

class Down(Refusal):
    """The daemon cannot be reached, or answers outside the wire: client rc 84."""

    def __init__(self, name, msg):
        super().__init__(DOWN, name, msg)


def say(msg):
    print(msg, file=sys.stderr, flush=True)


def env_secs(env, name, default, lo):
    raw = env.get(name)
    if raw in (None, ""):
        return default
    try:
        v = float(raw)
    except ValueError:
        v = float("nan")
    if not v >= lo:
        raise Refusal(USAGE, "usage", f"{name} is seconds >= {lo:g}, got {raw!r}")
    return v


def refusal(st, obj, path):
    """The exception for a reply that is not the success the caller wanted."""
    text = obj.get("error") if isinstance(obj.get("error"), str) else json.dumps(obj)[:200]
    rc = obj.get("rc")
    if st == 404 and path != "/jobs":
        return Refusal(USAGE, "no-such-job", f"{path}: {text}")
    if 400 <= st < 500 and isinstance(rc, int) and not isinstance(rc, bool) and 1 <= rc <= 255:
        return Refusal(rc, "refused", text)
    return Down("bad-reply", f"{path}: status {st}: {text}")


class Queue:
    """The daemon's loopback HTTP, one `ssh <host> curl` per request, and the tails' ssh children."""

    def __init__(self, env):
        self.host = env.get("BOXQ_HOST") or env.get("BLOOMERY_BOX") or "ws"
        self.ssh = shlex.split(env.get("BOXQ_SSH") or "ssh")
        self.url = (env.get("BOXQ_URL") or "http://127.0.0.1:8030").rstrip("/")
        self.poll_s = env_secs(env, "BOXQ_POLL_S", 5.0, 0.01)
        self.note_s = env_secs(env, "BOXQ_NOTE_S", 60.0, 0.0)
        self.cancel_wait_s = env_secs(env, "BOXQ_CANCEL_WAIT_S", 150.0, 0.0)

    def spawn(self, remote, **kw):
        # Own session: a Ctrl-C reaches this client only, which stops its children itself.
        try:
            return subprocess.Popen(self.ssh + [self.host, remote], start_new_session=True, **kw)
        except OSError as e:
            raise Down("no-ssh", f"{self.ssh[0]}: {e}")

    def request(self, method, path, body=None):
        """(status, JSON object) of one request."""
        words = ["curl", "-sS", "--max-time", "20", "-X", method]
        if body is not None:  # only a POST has a body, on stdin
            words += ["-H", "Content-Type: application/json", "-H", "Expect:", "--data-binary", "@-"]
        remote = " ".join(shlex.quote(w) for w in words + ["-w", r"\n%{http_code}", self.url + path])
        try:
            p = self.spawn(remote, stdin=subprocess.DEVNULL if body is None else subprocess.PIPE,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                out, err = p.communicate(None if body is None else body.encode(), timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
                p.communicate()
                raise Down("daemon-down", f"{method} {path}: no reply in 30 s")
        except OSError as e:
            raise Down("daemon-down", f"{method} {path}: {e}")
        if p.returncode != 0:
            why = (err.decode(errors="replace").strip().splitlines() or ["no message"])[0]
            raise Down("daemon-down", f"{method} {path}: ssh {self.host} rc {p.returncode}: {why}")
        raw, _, code = out.rpartition(b"\n")
        try:
            st, obj = int(code), json.loads(raw)
        except ValueError:
            raise Down("bad-reply", f"{method} {path}: not JSON and a status: {out[:120]!r}")
        if not isinstance(obj, dict):
            raise Down("bad-reply", f"{method} {path}: the reply is not a JSON object: {raw[:120]!r}")
        return st, obj

    def call(self, method, path, body=None, ok=(200,)):
        st, obj = self.request(method, path, body)
        if st not in ok:
            raise refusal(st, obj, path)
        return obj

    def health(self):
        st, obj = self.request("GET", "/health")
        if st != 200 or obj.get("ok") is not True:
            raise Down("daemon-down", f"GET /health: status {st}, {json.dumps(obj)[:120]}")

    def job(self, jid):
        job = self.call("GET", f"/jobs/{jid}")
        if job.get("status") not in STATUSES:
            raise Down("bad-reply", f"job {jid}: status {job.get('status')!r} is not one of {', '.join(STATUSES)}")
        return job

    def cancel(self, jid):
        return self.call("POST", f"/jobs/{jid}/cancel", "{}", ok=(200, 202))


class Trap:
    """INT and TERM as a list and a wake-up byte: the wait between polls selects on the pipe, so a signal ends it."""

    def __init__(self):
        self.sigs = []
        self.r, self.w = os.pipe()
        os.set_blocking(self.r, False)
        os.set_blocking(self.w, False)
        self.old = {}

    def __enter__(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            self.old[sig] = signal.signal(sig, self.on_signal)
        return self

    def __exit__(self, *exc):
        for sig, old in self.old.items():
            signal.signal(sig, old)
        os.close(self.r)
        os.close(self.w)

    def on_signal(self, signum, _frame):
        self.sigs.append(signum)
        with contextlib.suppress(OSError):
            os.write(self.w, b"x")

    def wait(self, secs):
        select.select([self.r], [], [], secs)
        with contextlib.suppress(OSError):
            os.read(self.r, 64)

    def rc(self):
        return INT_RC if self.sigs and self.sigs[0] == signal.SIGINT else TERM_RC


def tail_cmd(path):
    # The remote tail ends when ssh's stdin does: killing the ssh client alone leaves a `tail -F` on the box for good.
    return f"tail -n +1 -F {shlex.quote(path)} & t=$!; read x; kill $t"


def box_path(job, key):
    path = job.get(key)
    return path if isinstance(path, str) and path.startswith("/") else None


def spawn_all(q, cmds, fds):
    """One ssh child per remote command, its stdout on the matching fd and its stderr on ours."""
    kids = []
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        for cmd, fd in zip(cmds, fds):
            kids.append(q.spawn(cmd, stdin=subprocess.PIPE, stdout=fd, stderr=2))
    except Down:
        unfollow(kids)
        raise
    return kids


def follow(q, job, fds=(1, 2)):
    """The job's out and err as two tail children writing to fds."""
    paths = [box_path(job, key) for key in ("out", "err")]
    if None in paths:
        raise Down("bad-reply", f"job {job.get('id')}: out and err are {paths!r}, a running job has absolute box paths")
    return spawn_all(q, [tail_cmd(p) for p in paths], fds)


def dump(q, job, trap, fds=(1, 2)):
    """A job that finished before any tail ran: its files are final, so each is read whole and the children end by
    themselves; a file the job never wrote, or a path it never got, is skipped."""
    pairs = [(box_path(job, key), fd) for key, fd in zip(("out", "err"), fds)]
    kids = spawn_all(q, [f"[ ! -e {shlex.quote(p)} ] || exec cat {shlex.quote(p)}" for p, _ in pairs if p],
                     [fd for p, fd in pairs if p])
    while any(k.poll() is None for k in kids) and not trap.sigs:
        trap.wait(0.1)
    unfollow(kids)


def unfollow(kids, grace=3.0):
    """Stop the tails: close their stdin (the remote side ends), terminate and then kill a ssh that does not follow."""
    for k in kids:
        with contextlib.suppress(OSError):
            k.stdin.close()
    end = time.monotonic() + grace
    for k in kids:
        try:
            k.wait(max(0.05, end - time.monotonic()))
        except subprocess.TimeoutExpired:
            k.terminate()
            try:
                k.wait(2)
            except subprocess.TimeoutExpired:
                k.kill()
                k.wait()
    kids.clear()


def eta_text(eta):
    if isinstance(eta, (int, float)) and not isinstance(eta, bool) and eta >= 0:
        return f"eta {round(eta)} s"
    return "eta unknown"


def submitted_age(job):
    """Seconds since the job's `submitted` (ISO; without a zone it is read in this machine's, which the box shares),
    or None when it is missing, not ISO, or in the future."""
    raw = job.get("submitted")
    if not isinstance(raw, str):
        return None
    try:
        t = datetime.datetime.fromisoformat(raw[:-1] + "+00:00" if raw.endswith("Z") else raw)
    except ValueError:
        return None
    age = (datetime.datetime.now(t.tzinfo) - t).total_seconds()
    return age if age >= 0 else None


def outcome(jid, job):
    """The exit code of a terminal job."""
    rc = job.get("rc")
    if job["status"] == "lost":
        return LOST
    if isinstance(rc, int) and not isinstance(rc, bool) and 0 <= rc <= 255:
        return rc
    if job["status"] == "cancelled":
        return LOST
    raise Down("bad-reply", f"job {jid} is {job['status']} with rc {rc!r}")


def mirror(rc_dir, jid, rc):
    """<rc-dir>/<id>.rc, written whole or not at all."""
    tmp = os.path.join(rc_dir, f".{jid}.rc.{os.getpid()}")
    try:
        os.makedirs(rc_dir, exist_ok=True)
        with open(tmp, "w") as f:
            f.write(f"{rc}\n")
        os.replace(tmp, os.path.join(rc_dir, f"{jid}.rc"))
    except OSError as e:
        say(f"boxq: {jid}: the rc mirror {rc_dir}/{jid}.rc was not written: {e}")
        with contextlib.suppress(OSError):
            os.unlink(tmp)


def attach_job(q, jid, rc_dir, trap):
    """Follow the job to a terminal status, mirror and return its exit code."""
    kids, fails, last = [], 0, None
    queued_at, next_note, reason, cancel_at = None, 0.0, None, None
    try:
        while True:
            job = None
            try:
                job = q.job(jid)
                fails = 0
            except Down as e:
                fails += 1
                if fails >= POLL_FAILS:
                    raise Down(e.name, f"job {jid} is left as it is: {e.msg}")
                say(f"boxq: {jid}: poll {fails}/{POLL_FAILS} failed: {e.msg}")
            if job is not None:
                last, now = job, time.monotonic()
                if job["status"] == "queued":
                    reason = job.get("wait_reason")
                    queued_at = now if queued_at is None else queued_at
                    if now >= next_note:
                        say(f"boxq: {jid} waiting: {reason or 'none'}, {eta_text(job.get('eta_s'))}")
                        next_note = now + q.note_s
                else:
                    if queued_at is not None:
                        age = submitted_age(job)
                        age = now - queued_at if age is None else age
                        say(f"boxq: {jid} queued {round(age)} s (wait_reason {reason or 'none'})")
                    if not kids and job["status"] not in TERMINAL:
                        kids = follow(q, job)
                    queued_at = None
                if job["status"] in TERMINAL:
                    break
            if trap.sigs and cancel_at is None:
                say(f"boxq: {jid}: signal, cancelling")
                q.cancel(jid)
                cancel_at = time.monotonic()
            if cancel_at is not None and (len(trap.sigs) > 1 or time.monotonic() - cancel_at >= q.cancel_wait_s):
                st = last["status"] if last else "unknown"
                raise Refusal(LOST, "cancel-pending", f"job {jid} is still {st} after the cancel; check it on the box")
            trap.wait(q.poll_s)
        if kids:
            trap.wait(min(q.poll_s, FLUSH_MAX_S))  # the tails read what the job wrote last
            unfollow(kids)
        else:
            dump(q, last, trap)
        if cancel_at is None and last["status"] == "cancelled":
            say(f"boxq: {jid} cancelled elsewhere")
        rc = trap.rc() if cancel_at is not None else outcome(jid, last)
        mirror(rc_dir, jid, rc)
        return rc
    finally:
        unfollow(kids)


def default_owner():
    return os.path.basename(os.path.dirname(here()))


def job_body(a):
    words = a.words
    if not words or not "".join(words).strip():
        raise Refusal(USAGE, "usage", "the job's command goes after --")
    owner = a.owner or default_owner()
    for flag, v in (("--name", a.name), ("--owner", owner)):
        if not NAME_RE.match(v):
            raise Refusal(USAGE, "usage", f"{flag} is [A-Za-z0-9._-]+, got {v!r}")
    if a.expected_s < 1:
        raise Refusal(USAGE, "usage", f"--expected-s is whole seconds >= 1, got {a.expected_s}")
    env = {}
    for kv in a.env:
        k, eq, v = kv.partition("=")
        if not eq or not ENV_RE.match(k):
            raise Refusal(USAGE, "usage", f"--env is NAME=VALUE, got {kv!r}")
        env[k] = v
    return {"kind": "command", "argv": words[0] if len(words) == 1 else shlex.join(words), "env": env, "owner": owner,
            "name": a.name, "card": a.card, "job_class": a.job_class, "expected_s": a.expected_s, "wait": 0}


def submit_job(q, body, trap=None):
    """Post the job; its id."""
    say(f"boxq: {body['name']} class={body['job_class']} card={body['card']} predicted {body['expected_s']} s")
    q.health()
    if trap and trap.sigs:
        raise Refusal(trap.rc(), "interrupted", "before the job was posted")
    jid = q.call("POST", "/jobs", json.dumps(body), ok=(201,)).get("id")
    if not (isinstance(jid, str) and NAME_RE.match(jid)):
        raise Down("bad-reply", f"POST /jobs: the job has no usable id: {jid!r}")
    return jid


def job_id(a):
    if not NAME_RE.match(a.id):
        raise Refusal(USAGE, "usage", f"a job id is [A-Za-z0-9._-]+, got {a.id!r}")
    return a.id


def rc_dir_of(a):
    return a.rc_dir or os.environ.get("BOXQ_RC_DIR") or os.path.expanduser("~/.cache/bloomery/boxq")


def cmd_submit(a):
    body = job_body(a)
    print(submit_job(Queue(os.environ), body), flush=True)


def cmd_attach(a):
    q, jid = Queue(os.environ), job_id(a)
    with Trap() as trap:
        return attach_job(q, jid, rc_dir_of(a), trap)


def cmd_run(a):
    body, q = job_body(a), Queue(os.environ)
    with Trap() as trap:
        jid = submit_job(q, body, trap)
        say(f"boxq: {body['name']} is job {jid}")
        return attach_job(q, jid, rc_dir_of(a), trap)


def cmd_cancel(a):
    print(Queue(os.environ).cancel(job_id(a)).get("status"), flush=True)


def parser():
    p = Parser(prog="boxq.py", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd")
    for name in ("submit", "run"):
        s = sub.add_parser(name)
        s.add_argument("--card", required=True, choices=LANES)
        s.add_argument("--class", dest="job_class", required=True, choices=CLASSES)
        s.add_argument("--expected-s", dest="expected_s", required=True, type=int)
        s.add_argument("--name", required=True)
        s.add_argument("--owner")
        s.add_argument("--env", action="append", default=[])
        if name == "run":
            s.add_argument("--rc-dir")
    s = sub.add_parser("attach")
    s.add_argument("id")
    s.add_argument("--rc-dir")
    sub.add_parser("cancel").add_argument("id")
    s = sub.add_parser("fill-list")
    s.add_argument("--file")
    s.add_argument("--justfile")
    sub.add_parser("codes-check").add_argument("--root")
    return p


HANDLERS = {"submit": cmd_submit, "attach": cmd_attach, "run": cmd_run, "cancel": cmd_cancel,
            "fill-list": cmd_fill_list, "codes-check": cmd_codes_check}


def main(argv):
    try:
        if argv and argv[0] == "--self-test":
            return self_test(argv[1:])
        if argv and argv[0] == "--list-cases":
            print("\n".join(c.__name__[5:].replace("_", "-") for c in CASES))
            return 0
        head, words = argv, None
        if argv[:1] in (["submit"], ["run"]) and "--" in argv:
            head, words = argv[:argv.index("--")], argv[argv.index("--") + 1:]
        a = parser().parse_args(head)
        if not a.cmd:
            raise Refusal(USAGE, "usage", f"a subcommand is needed: {', '.join(HANDLERS)}")
        a.words = words
        return HANDLERS[a.cmd](a) or 0
    except Refusal as r:
        print(f"boxq: {r.name}: {r.msg}", file=sys.stderr)
        return r.code
    except KeyboardInterrupt:
        print("boxq: interrupted", file=sys.stderr)
        return INT_RC


# ---- self-test ----------------------------------------------------------------------------------------------------

class Fake:
    """The daemon's side of the wire in a thread: it records each request and serves job j1 from a script of poll
    states, the last state repeating. `post` and `health` are (status, object) replies that override the defaults;
    `cancel_to` is the status a cancelled job shows (None: the cancel is ignored); `slow` seconds pass before health
    answers; `gate` (a callable) holds the job running until it is true; `late` is a pair of byte strings appended to
    the out and err files 0.1 s after the first terminal poll was answered, the way a job's last output trails its
    status."""

    def __init__(self, files, states, health=(200, {"ok": True}), post=None, cancel_to="cancelled", slow=0.0, gate=None,
                 late=None):
        self.requests, self.polls, self.cancelled, self.slow = [], 0, False, slow
        self.files, self.states, self.health, self.post, self.cancel_to = files, states, health, post, cancel_to
        self.gate, self.late = gate, late
        fake = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def serve(self):
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                st, obj = fake.handle(self.command, self.path, body, self.headers.get("Content-Type"))
                data = json.dumps(obj).encode()
                self.send_response(st)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_GET = do_POST = serve

        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()

    def job(self, **kw):
        job = {"id": "j1", "status": "queued", "rc": None, "wait_reason": None, "eta_s": None, "out": self.files[0],
               "err": self.files[1], "rc_file": "/box/j1/rc", "resolved_card": "a6000"}
        job.update(kw)
        return job

    def peek(self):
        if self.cancelled and self.cancel_to:
            return self.job(status=self.cancel_to)
        return self.job(**self.states[min(self.polls, len(self.states) - 1)])

    def poll(self):
        job = self.peek()
        if job["status"] in TERMINAL and not self.cancelled:
            if self.gate and not self.gate():
                return self.job(status="running")
            if self.late:
                timer = threading.Timer(0.1, self.append_late, [self.late])
                timer.daemon = True
                timer.start()
                self.late = None
        self.polls += 1
        return job

    def append_late(self, late):
        for path, data in zip(self.files, late):
            with contextlib.suppress(OSError), open(path, "ab") as f:
                f.write(data)

    def handle(self, method, path, body, ctype):
        self.requests.append((method, path, body, ctype))
        if (method, path) == ("GET", "/health"):
            time.sleep(self.slow)
            return self.health
        if (method, path) == ("POST", "/jobs"):
            return self.post or (201, self.job(**self.states[0]))
        if path == "/jobs/j1" and method == "GET":
            return 200, self.poll()
        if path == "/jobs/j1/cancel" and method == "POST":
            self.cancelled = True
            return 200, self.peek()
        return 404, {"error": "no such job"}

    def cancels(self):
        return [r for r in self.requests if r[1] == "/jobs/j1/cancel"]

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


JOB_ARGS = ("--card", "a6000", "--class", "round", "--expected-s", "90", "--name", "tag-1")


class Fix:
    """One case's world: a temp dir and a helper that runs a command in this process; for the client cases also a fake
    daemon, an ssh stub that runs its last argument locally (so `curl` and `tail -F` meet the fake and its files), and
    the client as a child process. Every pid a case leaves behind is in a file here, and cleanup signals only those."""

    def __init__(self, tmp):
        self.tmp, self.fake, self.procs = tmp, None, []
        self.out, self.err = os.path.join(tmp, "job.out"), os.path.join(tmp, "job.err")
        self.rc_dir, self.ssh = os.path.join(tmp, "rc"), os.path.join(tmp, "ssh")
        self.ssh_log = os.path.join(tmp, "ssh.log")
        self.tail_pids, self.hung_pids = os.path.join(tmp, "tail.pids"), os.path.join(tmp, "hung.pid")

    def call(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = main(list(argv))
        return rc, out.getvalue(), err.getvalue()

    def script(self, path, body):
        with open(path, "w") as f:
            f.write("#!/bin/sh\n" + body)
        os.chmod(path, 0o755)

    def stubs(self):
        if os.path.exists(self.ssh):
            return
        for tool in ("curl", "tail"):
            check(shutil.which(tool), f"the client cases need {tool} on PATH")
        bindir, q = os.path.join(self.tmp, "bin"), shlex.quote
        os.makedirs(bindir)
        self.script(os.path.join(bindir, "tail"),
                    f'echo $$ >> {q(self.tail_pids)}\nexec {q(shutil.which("tail"))} "$@"\n')
        self.script(self.ssh, f'printf "%s\\n" "$*" >> {q(self.ssh_log)}\nfor last; do :; done\n'
                              f'PATH={q(bindir)}:$PATH\nexport PATH\nexec sh -c "$last"\n')
        os.makedirs(self.rc_dir, exist_ok=True)

    def files(self, out=b"", err=b""):
        for path, data in ((self.out, out), (self.err, err)):
            with open(path, "wb") as f:
                f.write(data)

    def serve(self, states, **kw):
        if self.fake:
            self.fake.close()
        if not os.path.exists(self.out):
            self.files()
        self.fake = Fake((self.out, self.err), states, **kw)
        return self.fake

    def env(self, **extra):
        self.stubs()
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("BOXQ_") and k != "BLOOMERY_BOX"}
        env.update(BOXQ_SSH=self.ssh, BOXQ_HOST="testbox", BOXQ_POLL_S="0.2", BOXQ_CANCEL_WAIT_S="10")
        if self.fake:
            env["BOXQ_URL"] = self.fake.url
        for k, v in extra.items():
            env.pop(k, None) if v is None else env.__setitem__(k, v)
        return env

    def start(self, *argv, **extra):
        p = subprocess.Popen([sys.executable, os.path.abspath(__file__), *argv], env=self.env(**extra),
                             stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.procs.append(p)
        return p

    def finish(self, p, secs=40):
        try:
            out, err = p.communicate(timeout=secs)
        except subprocess.TimeoutExpired:
            p.kill()
            p.communicate()
            raise AssertionError(f"the client did not end in {secs} s")
        return p.returncode, out, err

    def client(self, *argv, **extra):
        return self.finish(self.start(*argv, **extra))

    def run_args(self, *extra, words=("true",)):
        return ("run", *JOB_ARGS, "--rc-dir", self.rc_dir, *extra, "--", *words)

    def rc_file(self, jid="j1"):
        try:
            with open(os.path.join(self.rc_dir, f"{jid}.rc")) as f:
                return f.read()
        except OSError:
            return None

    def rm_rc(self, jid="j1"):
        with contextlib.suppress(FileNotFoundError):
            os.unlink(os.path.join(self.rc_dir, f"{jid}.rc"))

    def pids(self, path):
        try:
            with open(path) as f:
                return [int(x) for x in f.read().split()]
        except OSError:
            return []

    def wait_pids(self, path, n, secs=10):
        end = time.monotonic() + secs
        while len(self.pids(path)) < n:
            check(time.monotonic() < end, f"{n} pids did not appear in {os.path.basename(path)}")
            time.sleep(0.02)

    def gone(self, path, n=None, secs=3):
        """The pids a stub recorded (n of them, when n is given) are all gone; the record restarts."""
        pids = self.pids(path)
        check(n is None or len(pids) == n, f"{os.path.basename(path)} holds {len(pids)} pids, {n} expected")
        end = time.monotonic() + secs
        while any(alive(p) for p in pids):
            check(time.monotonic() < end, f"pids still alive {secs} s on: {[p for p in pids if alive(p)]}")
            time.sleep(0.02)
        open(path, "w").close()

    def wait_requests(self, fake, n, which=None, secs=10):
        end = time.monotonic() + secs
        while len(which() if which else fake.requests) < n:
            check(time.monotonic() < end, f"the daemon saw fewer than {n} of the requests it waited for")
            time.sleep(0.02)

    def wait_polls(self, fake, n):
        self.wait_requests(fake, n, lambda: [r for r in fake.requests if r[:2] == ("GET", "/jobs/j1")])

    def cleanup(self):
        for p in self.procs:
            if p.poll() is None:
                p.kill()
                p.communicate()
        for path in (self.tail_pids, self.hung_pids):
            for pid in self.pids(path):
                with contextlib.suppress(ProcessLookupError):
                    os.kill(pid, signal.SIGKILL)
        if self.fake:
            self.fake.close()


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def case_fill_list(fx):
    jf = os.path.join(fx.tmp, "justfile")
    with open(jf, "w") as f:
        f.write("set shell := ['bash', '-c']\nX := 'a:b'\nalias w := weekly-a\n\n[group('solo')]\n# c\n[group('v41-load')]\n"
                "weekly-a:\n    echo hi:there\n\nweekly-b *ARGS:\n    true\nname arg='x:y' *R: dep\n    true\nexport Y := 'z'\n")
    check(justfile_recipes(jf) == {"weekly-a": ["solo", "v41-load"], "weekly-b": [], "name": []},
          f"the justfile parse read {justfile_recipes(jf)}")
    ff = os.path.join(fx.tmp, "fill.tsv")

    def run(*rows):
        with open(ff, "w") as f:
            f.write("# header\n" + "\n".join("\t".join(r) for r in rows) + "\n")
        return fx.call("fill-list", "--file", ff, "--justfile", jf)

    good = ("weekly-a", "both", "times:weekly-a", "why")
    rc, out, err = run(good, ("weekly-b", "any", "given:10", "why"))
    check(rc == 0 and "groups=solo,v41-load" in out and out.count("\n") == 2, f"good rows: {rc} {out!r} {err!r}")
    for bad, name in ((("nope", "any", "given:5", "w"), "a recipe the justfile lacks"),
                      (("weekly-a", "moon", "given:5", "w"), "an unknown lane"),
                      (("weekly-a", "any", "five", "w"), "a bad minutes source"),
                      (("cmd:true", "any", "times:x", "w"), "a cmd: row on times"),
                      (("weekly-a", "any", "given:5", ""), "no reason"),
                      (("weekly-a", "any", "given:5"), "three columns")):
        rc, _, err = run(bad)
        check(rc == DATA and "bad-fill-row" in err, f"{name} must be 65 by name, got {rc} {err!r}")
    check(run(good, good)[0] == DATA, "a duplicate row is refused")
    check(run(("weekly-a", "any", "given:45", "w"))[0] == 0, "a row over 30 min parses: the minutes bound nothing")


def case_codes(fx):
    check(not set(OWN_CODES) & OTHER_CODES and len(set(OWN_CODES)) == 5 and OWN_CODES == (80, 81, 83, 84, 85),
          "the codes 80, 81, 83-85 overlap the reserved set or each other")
    d = os.path.join(fx.tmp, "repo")
    os.makedirs(os.path.join(d, "tools"))
    for n, text in (("justfile", "r:\n    exit 64\n"), ("tools/a.sh", "# exit 80 in a comment\nexit 75\n"),
                    ("tools/b.py", "sys.exit(97)\n")):
        with open(os.path.join(d, n), "w") as f:
            f.write(text)
    rc, out, err = fx.call("codes-check", "--root", d)
    check(rc == 0, f"a tree producing 64, 75 and 97 is clean: {rc} {err!r}")
    for bad in ("exit 80", "sys.exit(83)", "return 84", "flock -s -n -E 85 f true", "(exit 81)", "raise SystemExit(80)"):
        with open(os.path.join(d, "tools", "c.sh"), "w") as f:
            f.write(f"x\n{bad}\n")
        rc, out, err = fx.call("codes-check", "--root", d)
        check(rc == DATA and "tools/c.sh:2" in err, f"a tool producing '{bad}' must be 65: {rc} {err!r}")
    with open(os.path.join(d, "tools", "c.sh"), "w") as f:
        f.write("exit 82\n")
    check(fx.call("codes-check", "--root", d)[0] == 0, "82 is no longer the queue's: another tool may produce it")
    with open(os.path.join(d, "tools", "c.sh"), "w") as f:
        f.write('[ "$rc" = 80 ] && case $rc in 83) echo x;; esac\n')
    check(fx.call("codes-check", "--root", d)[0] == 0, "a tool that tests a code, not produces it, is not a collision")

def case_submit_body(fx):
    fake = fx.serve([{"status": "queued"}])
    long_cmd = "cd ~/repo/x && just gate-y " + "z" * 3000
    rc, out, err = fx.client("submit", *JOB_ARGS, "--owner", "w3b1", "--env", "A=1", "--env", "B=x=y", "--", long_cmd)
    check(rc == 0 and out == b"j1\n", f"submit prints the id and nothing else: {rc} {out!r} {err!r}")
    check(err == b"boxq: tag-1 class=round card=a6000 predicted 90 s\n", f"the predicted-wall line: {err!r}")
    check([r[:2] for r in fake.requests] == [("GET", "/health"), ("POST", "/jobs")],
          f"health, then the post: {fake.requests}")
    body = json.loads(fake.requests[1][2])
    want = {"kind": "command", "argv": long_cmd, "env": {"A": "1", "B": "x=y"}, "owner": "w3b1", "name": "tag-1",
            "card": "a6000", "job_class": "round", "expected_s": 90, "wait": 0}
    check(body == want and "priority" not in body, f"the body is the wire's fields and no priority: {body}")
    check(fake.requests[1][3] == "application/json", "the body is sent as application/json")
    log = open(fx.ssh_log).read().splitlines()
    check(log[1] == f"testbox curl -sS --max-time 20 -X POST -H 'Content-Type: application/json' -H Expect: "
                    f"--data-binary @- -w '\\n%{{http_code}}' {fake.url}/jobs", f"the ssh command of the post: {log[1]!r}")
    check(log[0] == f"testbox curl -sS --max-time 20 -X GET -w '\\n%{{http_code}}' {fake.url}/health",
          f"a GET has no body and no stdin: {log[0]!r}")
    check(fake.requests[0][2] == b"" and fake.requests[0][3] is None, f"the daemon saw a GET with a body: {fake.requests[0]}")
    check("just gate-y" not in "\n".join(log), "the job text is on an ssh command line")
    rc, out, err = fx.client("submit", *JOB_ARGS, "--", "just", "gate-y", "two words")
    body = json.loads(fake.requests[-1][2])
    tree = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    check(body["argv"] == "just gate-y 'two words'" and body["env"] == {} and body["owner"] == tree,
          f"several words are shell-quoted, the owner defaults to the tree {tree}: {body}")
    for env, host in (({"BOXQ_HOST": None, "BLOOMERY_BOX": "bx"}, "bx"), ({"BOXQ_HOST": None}, "ws")):
        fx.client("submit", *JOB_ARGS, "--", "true", **env)
        check(open(fx.ssh_log).read().splitlines()[-1].startswith(f"{host} curl"), f"the host defaults to {host}")
    posts = len(fake.requests)
    for bad, why in ((JOB_ARGS, "no -- command"), ((*JOB_ARGS, "--"), "an empty command"),
                     ((*JOB_ARGS, "--", " "), "a blank command"),
                     (("--card", "moon", *JOB_ARGS[2:], "--", "true"), "an unknown card"),
                     ((*JOB_ARGS[:2], "--class", "media", *JOB_ARGS[4:], "--", "true"), "a class the routes own"),
                     ((*JOB_ARGS[:4], "--expected-s", "0", *JOB_ARGS[6:], "--", "true"), "expected-s 0"),
                     ((*JOB_ARGS[:6], "--name", "a b", "--", "true"), "a name with a space"),
                     ((*JOB_ARGS, "--env", "NOEQ", "--", "true"), "an env without =")):
        rc, out, err = fx.client("submit", *bad)
        check(rc == USAGE and out == b"", f"{why} must be usage (64): {rc} {err!r}")
    check(len(fake.requests) == posts, "a refused command line reached the daemon")


def case_run_done(fx):
    out, err, late = b"o1\nbin\xff\x00end\r\n", b"e1\nwarn\n", (b"late-out\n", b"late-err\n")
    for status, rc in (("done", 0), ("failed", 3)):
        fx.files(out, err)
        fx.serve([{"status": "running"}, {"status": status, "rc": rc}], late=late,
                 gate=lambda: len(fx.pids(fx.tail_pids)) >= 2)
        got, so, se = fx.client(*fx.run_args(), BOXQ_POLL_S="0.5")
        check(got == rc, f"the exit code is the job's rc {rc}: {got} {se!r}")
        check(so == out + late[0], f"stdout is the job's out and what trailed its status, byte for byte: {so!r}")
        check(se.count(err + late[1]) == 1 and out not in se, f"the job's err goes to stderr once, its out not: {se!r}")
        check(fx.rc_file() == f"{rc}\n" and os.listdir(fx.rc_dir) == ["j1.rc"],
              f"the rc mirror: {fx.rc_file()!r} {os.listdir(fx.rc_dir)}")
        fx.gone(fx.tail_pids, 2)
        fx.rm_rc()


def case_lost(fx):
    fx.serve([{"status": "running"}, {"status": "running"}, {"status": "lost"}])
    got, so, se = fx.client(*fx.run_args())
    check(got == 81 and fx.rc_file() == "81\n", f"a lost job exits 81 and mirrors it: {got} {fx.rc_file()!r} {se!r}")


def case_health_down(fx):
    fake = fx.serve([], health=(503, {"error": "starting"}))
    for health, why in (((503, {"error": "starting"}), "a 503"), ((200, {"ok": False}), "ok false"),
                        ((200, {}), "no ok")):
        fake.health = health
        rc, out, err = fx.client("submit", *JOB_ARGS, "--", "true")
        check(rc == 84 and b"daemon-down" in err, f"health {why} must be 84: {rc} {err!r}")
        check(not [r for r in fake.requests if r[0] == "POST"], f"a job was posted past a failed health ({why})")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        closed = f"http://127.0.0.1:{sock.getsockname()[1]}"
    rc, out, err = fx.client("submit", *JOB_ARGS, "--", "true", BOXQ_URL=closed)
    check(rc == 84 and b"daemon-down" in err, f"a daemon that is not there must be 84: {rc} {err!r}")
    rc, out, err = fx.client("attach", "j1", "--rc-dir", fx.rc_dir, BOXQ_URL=closed)
    check(rc == 84 and b"poll 2/3 failed" in err and b"left as it is" in err and fx.rc_file() is None,
          f"an attach that cannot poll gives up after three, mirrors nothing: {rc} {err!r}")


def case_refusal(fx):
    fake = fx.serve([], post=(409, {"error": "a sitting is draining the box", "rc": 75}))
    rc, out, err = fx.client("submit", *JOB_ARGS, "--", "true")
    check(rc == 75 and out == b"" and b"a sitting is draining the box" in err,
          f"a 4xx exits with its rc and prints its text: {rc} {err!r}")
    check(b"predicted 90 s" in err, f"the predicted wall is printed before the post: {err!r}")
    for post, why in (((400, {"error": "bad", "rc": 0}), "a refusal that names rc 0"),
                      ((500, {"error": "boom", "rc": 70}), "a 5xx"),
                      ((201, {"status": "queued"}), "a created job with no id"),
                      ((200, {"id": "j1"}), "a 200 where the wire says 201")):
        fake.post = post
        rc, out, err = fx.client("submit", *JOB_ARGS, "--", "true")
        check(rc == 84 and out == b"" and b"bad-reply" in err,
              f"{why} is the daemon off the wire, 84: {rc} {out!r} {err!r}")
    rc, out, err = fx.client("attach", "j9", "--rc-dir", fx.rc_dir)
    check(rc == 64 and b"no-such-job" in err, f"a 404 is 64 no such job: {rc} {err!r}")
    for state, why in (({"status": "claimed"}, "claimed"),
                       ({"status": "done", "rc": None}, "a done job without an rc")):
        fake.states = [state]
        rc, out, err = fx.client("attach", "j1", "--rc-dir", fx.rc_dir)
        check(rc == 84 and b"bad-reply" in err and fx.rc_file() is None,
              f"{why} is named, 84, mirrors nothing: {rc} {err!r}")


def case_passthrough(fx):
    for rc in (80, 83, 85):
        fx.serve([{"status": "running"}, {"status": "failed", "rc": rc}])
        got, so, se = fx.client(*fx.run_args())
        check(got == rc and fx.rc_file() == f"{rc}\n",
              f"the job's rc {rc} passes through unchanged: {got} {fx.rc_file()!r} {se!r}")


def case_queued_line(fx):
    queued = {"status": "queued", "wait_reason": "card-busy"}
    states = [queued, queued, {**queued, "eta_s": 12}, {"status": "running"}, {"status": "running"},
              {"status": "done", "rc": 0}]
    fx.serve(states)
    rc, so, se = fx.client(*fx.run_args())
    lines = se.decode().splitlines()
    waits = [l for l in lines if l.startswith("boxq: j1 waiting")]
    check(rc == 0 and waits == ["boxq: j1 waiting: card-busy, eta unknown"],
          f"one wait line a minute, eta unknown for null: {waits} {rc}")
    moved = [l for l in lines if re.fullmatch(r"boxq: j1 queued \d+ s \(wait_reason card-busy\)", l)]
    check(len(moved) == 1, f"one line when the job leaves the queue: {lines}")
    fx.serve(states)
    rc, so, se = fx.client(*fx.run_args(), BOXQ_NOTE_S="0")
    waits = [l for l in se.decode().splitlines() if l.startswith("boxq: j1 waiting")]
    check(waits == ["boxq: j1 waiting: card-busy, eta unknown"] * 2 + ["boxq: j1 waiting: card-busy, eta 12 s"],
          f"a line every poll at 0 s: {waits}")


def case_int_cancel(fx):
    for sig, want in ((signal.SIGINT, 130), (signal.SIGTERM, 143)):
        fx.files(b"o\n", b"e\n")
        fake = fx.serve([{"status": "running"}])
        p = fx.start(*fx.run_args())
        fx.wait_pids(fx.tail_pids, 2)
        p.send_signal(sig)
        got, so, se = fx.finish(p)
        cancels = fake.cancels()
        check(got == want and b"elsewhere" not in se, f"{sig.name} exits {want}, the cancel is ours: {got} {se!r}")
        check(len(cancels) == 1 and cancels[0][0] == "POST" and cancels[0][2] == b"{}",
              f"one POST cancel with {{}}: {cancels}")
        check(fx.rc_file() == f"{want}\n", f"the mirror holds {want}: {fx.rc_file()!r}")
        fx.gone(fx.tail_pids, 2)
        fx.rm_rc()
    fake = fx.serve([{"status": "queued", "wait_reason": "card-busy"}])
    p = fx.start(*fx.run_args())
    fx.wait_polls(fake, 2)
    p.send_signal(signal.SIGTERM)
    got, so, se = fx.finish(p)
    check(got == 143 and fx.pids(fx.tail_pids) == [] and len(fake.cancels()) == 1,
          f"a queued job is cancelled too, with no tail to stop: {got} {se!r} {fake.requests}")
    fake = fx.serve([{"status": "running"}], slow=1.0)
    p = fx.start(*fx.run_args())
    fx.wait_requests(fake, 1)
    p.send_signal(signal.SIGINT)
    got, so, se = fx.finish(p)
    check(got == 130 and not [r for r in fake.requests if r[:2] == ("POST", "/jobs")] and b"interrupted" in se,
          f"a signal before the post posts nothing: {got} {se!r} {fake.requests}")
    fx.rm_rc()
    fake = fx.serve([{"status": "running"}], cancel_to=None)
    p = fx.start(*fx.run_args(), BOXQ_CANCEL_WAIT_S="1")
    fx.wait_pids(fx.tail_pids, 2)
    t0 = time.monotonic()
    p.send_signal(signal.SIGINT)
    got, so, se = fx.finish(p)
    check(got == 81 and b"j1" in se and time.monotonic() - t0 >= 1.0,
          f"a job still running after the wait: 81 naming it: {got} {se!r}")
    check(fx.rc_file() is None, "a job that is not terminal gets no rc mirror")
    fx.gone(fx.tail_pids, 2)
    fake = fx.serve([{"status": "running"}], cancel_to=None)
    p = fx.start(*fx.run_args(), BOXQ_CANCEL_WAIT_S="100")
    fx.wait_pids(fx.tail_pids, 2)
    p.send_signal(signal.SIGINT)
    fx.wait_requests(fake, 1, fake.cancels)
    p.send_signal(signal.SIGINT)
    got, so, se = fx.finish(p, 10)
    check(got == 81, f"a second signal ends the wait: {got} {se!r}")


class RecTrap:
    """A Trap that never sleeps and keeps the waits it was asked for."""

    def __init__(self):
        self.sigs, self.waits = [], []

    def wait(self, secs):
        self.waits.append(secs)


def case_flush_cap(fx):
    fx.serve([{"status": "running"}, {"status": "done", "rc": 0}])
    for poll, flush in (("5", 2.0), ("1", 1.0)):
        trap = RecTrap()
        rc = attach_job(Queue(fx.env(BOXQ_POLL_S=poll)), "j1", fx.rc_dir, trap)
        check(rc == 0 and trap.waits == [float(poll), flush],
              f"the flush after a terminal status is min(poll, 2 s): poll {poll} s waited {trap.waits}")
        fx.gone(fx.tail_pids)  # the waits are instant, so a tail may be stopped before it recorded its pid
        fx.fake.polls = 0


def case_queued_age(fx):
    now, sec = datetime.datetime.now(), datetime.timedelta(seconds=1)
    kst = datetime.timezone(datetime.timedelta(hours=9))
    for raw, lo, hi in (((now - 100 * sec).isoformat(timespec="seconds"), 100, 104),
                        ((now - 50 * sec).astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), 50, 54),
                        ((datetime.datetime.now(kst) - 30 * sec).isoformat(), 30, 34)):
        age = submitted_age({"submitted": raw})
        check(age is not None and lo <= age <= hi, f"the age of {raw} is about {lo} s: {age}")
    for raw in (None, 7, "not a time", (now + 60 * sec).isoformat()):
        check(submitted_age({"submitted": raw}) is None, f"{raw!r} is no usable submitted time")
    check(submitted_age({}) is None, "a job without submitted has no age")
    submitted = (datetime.datetime.now() - 100 * sec).isoformat(timespec="seconds")
    fx.serve([{"status": "queued", "wait_reason": "card-busy", "submitted": submitted},
              {"status": "running", "submitted": submitted}, {"status": "running", "submitted": submitted},
              {"status": "done", "rc": 0, "submitted": submitted}])
    rc, so, se = fx.client(*fx.run_args())
    m = re.search(rb"queued (\d+) s \(wait_reason", se)
    check(rc == 0 and m and 100 <= int(m.group(1)) <= 120, f"the queued seconds come from `submitted`: {rc} {se!r}")


def case_cancelled_elsewhere(fx):
    for state, want in (({"status": "cancelled", "rc": 137}, 137), ({"status": "cancelled", "rc": None}, 81)):
        fx.serve([{"status": "running"}, {"status": "running"}, state])
        got, so, se = fx.client(*fx.run_args())
        check(got == want and b"boxq: j1 cancelled elsewhere\n" in se and fx.rc_file() == f"{want}\n",
              f"a job cancelled by someone else exits {want} and says so: {got} {se!r} {fx.rc_file()!r}")
        fx.gone(fx.tail_pids, 2)
        fx.rm_rc()


def case_attach_env(fx):
    fx.files(b"out\n", b"")
    fx.serve([{"status": "done", "rc": 0}])
    home = os.path.join(fx.tmp, "home")
    for env, where in (({"BOXQ_RC_DIR": os.path.join(fx.tmp, "envrc")}, "envrc"),
                       ({"BOXQ_RC_DIR": None, "HOME": home}, "home/.cache/bloomery/boxq")):
        got, so, se = fx.client("attach", "j1", **env)
        mirrored = os.path.join(fx.tmp, where, "j1.rc")
        check(got == 0 and so == b"out\n" and os.path.exists(mirrored) and open(mirrored).read() == "0\n",
              f"attach to a finished job prints its out and mirrors into {where}: {got} {so!r} {se!r}")
    check(fx.pids(fx.tail_pids) == [], "a job that finished before the attach is read whole, with no tail -F")
    for path in (fx.out, fx.err):
        os.unlink(path)
    fx.serve([{"status": "cancelled"}])
    got, so, se = fx.client("attach", "j1", BOXQ_RC_DIR=os.path.join(fx.tmp, "envrc"))
    check(got == 81 and so == b"" and se == b"boxq: j1 cancelled elsewhere\n" and fx.pids(fx.tail_pids) == [],
          f"a cancelled job that never wrote a file: 81 and one line: {got} {so!r} {se!r}")


def case_mirror(fx):
    path = os.path.join(fx.rc_dir, "j1.rc")
    os.makedirs(fx.rc_dir, exist_ok=True)
    with open(path, "w") as f:
        f.write("9\n")
    seen, real = [], os.replace

    def spy(src, dst):
        with open(src) as f:
            seen.append((f.read(), open(dst).read()))
        real(src, dst)

    os.replace = spy
    try:
        mirror(fx.rc_dir, "j1", 5)
    finally:
        os.replace = real
    check(seen == [("5\n", "9\n")], f"the new rc is whole in a temp file while the old stands, then renamed: {seen}")
    check(open(path).read() == "5\n" and os.listdir(fx.rc_dir) == ["j1.rc"],
          f"the mirror after: {os.listdir(fx.rc_dir)}")
    blocked = os.path.join(path, "x")
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        mirror(blocked, "j2", 7)
    check("was not written" in err.getvalue(), f"a mirror that cannot be written is a stderr line: {err.getvalue()!r}")


def case_config(fx):
    q = Queue({})
    got = (q.host, q.ssh, q.url, q.poll_s, q.note_s, q.cancel_wait_s)
    check(got == ("ws", ["ssh"], "http://127.0.0.1:8030", 5.0, 60.0, 150.0), f"the defaults: {got}")
    q = Queue({"BLOOMERY_BOX": "bx", "BOXQ_SSH": "ssh -o BatchMode=yes", "BOXQ_URL": "http://h:1/", "BOXQ_POLL_S": "2"})
    check((q.host, q.ssh, q.url, q.poll_s) == ("bx", ["ssh", "-o", "BatchMode=yes"], "http://h:1", 2.0), "the env's values")
    check(Queue({"BOXQ_HOST": "h", "BLOOMERY_BOX": "bx"}).host == "h", "BOXQ_HOST beats BLOOMERY_BOX")
    for name, bad in (("BOXQ_POLL_S", "0"), ("BOXQ_POLL_S", "x"), ("BOXQ_POLL_S", "nan"),
                      ("BOXQ_CANCEL_WAIT_S", "-1"), ("BOXQ_NOTE_S", "-1")):
        try:
            Queue({name: bad})
        except Refusal as r:
            check(r.code == USAGE and name in r.msg, f"{name}={bad} is a usage error naming it: {r.code} {r.msg}")
        else:
            raise AssertionError(f"{name}={bad} was taken")


def case_cancel_cmd(fx):
    fake = fx.serve([{"status": "running"}])
    rc, out, err = fx.client("cancel", "j1")
    check(rc == 0 and out == b"cancelled\n" and fake.requests[-1] == ("POST", "/jobs/j1/cancel", b"{}", "application/json"),
          f"cancel posts {{}} and prints the status: {rc} {out!r} {err!r} {fake.requests}")
    rc, out, err = fx.client("cancel", "j9")
    check(rc == 64, f"cancel of an unknown job is 64: {rc} {err!r}")


def case_tails_stop(fx):
    fx.files(b"x\n", b"y\n")
    q, job, null = Queue(fx.env()), {"id": "j1", "out": fx.out, "err": fx.err}, os.open(os.devnull, os.O_WRONLY)
    kids = follow(q, job, (null, null))
    fx.wait_pids(fx.tail_pids, 2)
    unfollow(kids)
    check(not kids, "unfollow empties the list")
    fx.gone(fx.tail_pids, 2)
    fx.script(os.path.join(fx.tmp, "hung"), f"echo $$ >> {shlex.quote(fx.hung_pids)}\nexec sleep 60\n")
    q.ssh = [os.path.join(fx.tmp, "hung")]
    kids = follow(q, job, (null, null))
    fx.wait_pids(fx.hung_pids, 2)
    unfollow(kids, grace=0.3)
    fx.gone(fx.hung_pids, 2)
    os.close(null)


CASES = [case_fill_list, case_codes, case_submit_body, case_run_done, case_lost, case_health_down, case_refusal,
         case_passthrough, case_queued_line, case_int_cancel, case_attach_env, case_mirror, case_config, case_cancel_cmd,
         case_tails_stop, case_flush_cap, case_queued_age, case_cancelled_elsewhere]


def self_test(names):
    known = {c.__name__[5:].replace("_", "-"): c for c in CASES}
    unknown = [n for n in names if n not in known]
    if unknown:
        raise Refusal(USAGE, "usage", f"no such case: {', '.join(unknown)}")
    failed = []
    for name, fn in known.items():
        if names and name not in names:
            continue
        with tempfile.TemporaryDirectory(prefix="boxq-selftest-") as tmp:
            fx = Fix(tmp)
            try:
                fn(fx)
                print(f"ok   {name}")
            except Exception as e:  # a case's red line is its message; a traceback only for a non-assertion
                failed.append(name)
                print(f"FAIL {name}: {e if isinstance(e, AssertionError) else repr(e)}")
            finally:
                fx.cleanup()
    if failed:
        print(f"boxq self-test: {len(failed)} red: {', '.join(failed)}")
        return 1
    print(f"boxq self-test: {len(names) or len(known)} cases ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
