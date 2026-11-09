#!/usr/bin/env python3
"""The functional phase of a release sitting's cell: what a user's first requests check, on a live server.

    tools/release/pass_checks.py [URL] [--pid N] [--stderr FILE] [--dir DIR] [--tag TEXT] [--timeout S]
                                 [--max-tokens N] [--long-repeat N] [--client toktape [--toktape-bin PATH]]
    tools/release/pass_checks.py off-reason --stderr FILE
    tools/release/pass_checks.py --self-test

The server is the cell's `bloomery-serve`, already listening. URL, --pid (the server's process), --stderr (its
stderr) and --dir (where each request and its reply are kept, one JSON a check) default to
$BLOOMERY_SESSION_URL, _PID, _STDERR and _DIR. Every request is a greedy chat turn (temperature 0, --max-tokens
128, thinking off) on /v1/chat/completions, one at a time and in this order, each check printing
`check <name>: PASS|FAIL|SKIP <detail>`:

  pass_short                   a short prompt is answered
  pass_long                    a prompt of at least 4096 tokens, counted by /tokenize before it is sent (--long-repeat
                               repeats of one sentence; a count under 4096 is a FAIL whose detail starts `harness:`
                               and names it, and the prompt is not sent), is answered
  second_request_equals_first  pass_short's request replayed, after the long one, answers the same text (a server that
                               moves experts: the same ids, or a split at a near tie; see below)
  off_reason                   only when the server is not up when the checks end: its refusal read from --stderr

With --client toktape the request of each of those three checks is `toktape record`'s (the flags are the release
pass's), tagged <tag>-short, <tag>-long and <tag>-short2, and the check reads the reply from the tape toktape wrote
under --dir/tapes: the check sends nothing itself, so a prompt is prefilled once and the tape that is the release's
record is the very request the check judged. The tape must hold one request, without an error, whose last message
is the check's prompt, and whose completion equals the text of its streamed tokens (the reasoning ones apart);
anything else is a FAIL naming it. toktape ends other than 0 -> FAIL naming its exit status.

A server that moves experts is read from its own records, not from a family list: the word of its `residency lever`
(else `residency unset`, else `residency host`) record in --stderr is `off` or a `mid-p<P>-s<S>` word, and any word but `off`
moves experts between the card and the host while it serves. Its replay runs on another expert map than the first reply
did, and the host and the card sum one expert in different orders (designed rounding, AGENTS.md "Performance first,
accuracy opt-in"), so a near-tie argmax may flip. Both requests then carry `logprobs` and `top_logprobs: 2`, and the check
passes when the two replies' token ids are the same, or when, at the first differing position, the first reply's top-2 gap
plus the replay's (a flip moves the pair's gap by at least both) is within SPLIT_GAP_BAND (the texts after the split are not
compared); a larger sum is a FAIL naming the position, both tokens and both gaps. Under --client toktape the tapes hold no logprobs, so the check sends a direct pair of the short request
(after the first tape, and after the long one) and judges that pair; the tapes are recorded and checked as ever. A server
that moves no expert (word `off`, or no residency record, which the detail names) keeps the exact text comparison.

A reply's text is `content`, or `reasoning_content` when content is empty; both empty is a FAIL. Anything but HTTP
200 is a FAIL. A FAIL does not stop the next check. After every check the server is probed (/health, and --pid
alive and not a zombie); a server that is not up skips every check left with `server not up: <why>`, the why
carrying its refusal. The last line is `SESSION rc=<n>`: 1 when any check is a FAIL, else 0 (and the exit status).
A line `session-plan: place=.. ctx_max=.. load_ctx=..` before the checks reads the server's `plan` and `load`
records from --stderr (tools/bloomery/records.py), the key the cell ran at.

An off cell is a server that never listened and said why. Its checks are all SKIP and `off_reason` is PASS with the
refusal text; SESSION rc is 0, because a named refusal is an outcome the release notes list, and the caller that
expects a cell to run sees `check off_reason:` and decides. A server that is down with no refusal line past the
start banner and the records is `off_reason: FAIL`. The refusal is the last line `bloomery-serve*: <text>` that is
neither the start banner nor a record, with the lines after it that are not records (`caused by:` ...); the process
prints it last, whatever the log holds before. `off-reason --stderr FILE` prints that text (exit 1 and no output when
the file has none).

Exit status: SESSION rc (0 or 1); 2 a named refusal (a file or the toktape binary is missing); 64 a usage error.
--self-test runs every case against tools/ref/stub-llama-server.py on a port of its own, then each mutant of the
code (the behaviour of the pass scripts this replaces, one at a time) and requires its case to fail.
"""
import argparse
import collections
import contextlib
import gzip
import http.client
import http.server
import importlib.util
import io
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import types
import urllib.parse
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUB = HERE.parent / "ref" / "stub-llama-server.py"

SHORT_PROMPT = "Reply with one short sentence about the weather."
LONG_SENTENCE = "Local model serving keeps the weights on the card and the host. "
LONG_MIN_TOKENS = 4096
CHAT_PATH = "/v1/chat/completions"
PROBE_TIMEOUT = 5.0
HEAD = re.compile(r"^bloomery-serve[\w-]*: ")
BANNER = re.compile(r"^bloomery-serve[\w-]*: a first start on a new card\b")
# The binaries whose records a server's stderr can hold; `plan` and `load` are the generate_ds41 kinds.
RECORD_BINS = ("generate_ds41", "bloomery-serve-ds41", "bloomery-serve-qwen38")
REASON_LINES = 8
# The run file `toktape record` writes: gzip'd JSON (or plain), `schema` first; this reader knows one.
TAPE_SCHEMA = 1
# The residency words a load runs by (crates/levers `residency_word`): `off`, or `mid-p<P>-s<S>` with an optional
# `-unhosted`. The kinds that state the word, most resolved first: the lever's record (V4.1, Qwen3.8 and GLM-5.3 set), the
# `residency unset` record of Qwen3.8 and the GLM seat unset, and the churn pool's `residency host` (printed only where the
# word is not `off`).
RESIDENCY_WORD = re.compile(r"^(off|mid-p\d+-s\d+(-unhosted)?)$")
RESIDENCY_KINDS = ("residency_lever", "residency_unset", "residency_host")
LOGPROBS_TOP = 2
# The widest shift, in nats, between a replay's two top-2 gaps at its split on a moving expert map. The host (Q8_2_X4, 32-value
# activation blocks) and the card (q8_1, 128-value blocks) round a moved expert's activation differently, so two expert maps
# give one id's logit apart by sigma = 0.35; a gap is the difference of two logits, so it moves by sqrt(2) * 0.35 = 0.495, and a
# flip is excused at three of that. A flip needs the pair's gap to move by the first reply's gap g1 plus the replay's g2.
# PIN(2026-11-04): 3 * sqrt(2) * 0.35 = 1.485 [derived] with V4.1's sigma (crates/gpu-gates/src/prompts.rs SIGMA_LOGIT_V41),
# which is borrowed: GLM-5.3's own sigma is not measured. The GLM-5.3 seat's data (rounds glmrep and resbits, [measured]: gap
# shift rms 0.19 and p95 0.37 over 768 same-prefix position pairs; g1 + g2 at ten splits at most 0.77) sits inside it.
SPLIT_GAP_BAND = 1.485
SIGMA_LOGIT = 0.35


class CheckFail(Exception):
    pass


class CheckSkip(Exception):
    pass


class Unreachable(Exception):
    pass


def load_records():
    spec = importlib.util.spec_from_file_location("bloomery_records", HERE.parent / "bloomery" / "records.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


records = load_records()
SCHEMAS = {}


def schema(bin_name):
    if bin_name not in SCHEMAS:
        SCHEMAS[bin_name] = records.load(bin_name)
    return SCHEMAS[bin_name]


def head(text, n=200):
    t = " ".join(str(text).split())
    return t if len(t) <= n else t[: n - 3] + "..."


def request(base, method, path, body, timeout):
    """(status, body text) of one HTTP request; Unreachable when no answer comes."""
    u = urllib.parse.urlsplit(base)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    try:
        conn.request(method, path, body=None if body is None else json.dumps(body).encode(),
                     headers={"Content-Type": "application/json"})
        r = conn.getresponse()
        return r.status, r.read().decode("utf-8", "replace")
    except (OSError, http.client.HTTPException) as e:
        raise Unreachable(str(e) or type(e).__name__)
    finally:
        conn.close()


def alive(pid):
    """Whether process `pid` runs: a zombie is not running."""
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except (PermissionError, ValueError):
        return True
    try:
        stat = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return True
    return bool(stat.strip()) and not stat.strip().startswith("Z")


def chat_body(prompt, max_tokens, top_logprobs=0):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0,
            "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
    if top_logprobs:
        body.update(logprobs=True, top_logprobs=top_logprobs)
    return body


def choose_text(content, reasoning):
    """(text, field): the content, or the reasoning when content is empty; both empty is a CheckFail."""
    text = content if content.strip() else reasoning
    if not text.strip():
        raise CheckFail("both content and reasoning_content are empty")
    return text, "content" if content.strip() else "reasoning_content"


def reply_text(obj):
    """(text, field) of a chat reply: content, or reasoning_content when content is empty; both empty is a CheckFail."""
    msg = ((obj.get("choices") or [{}])[0]).get("message") or {}
    return choose_text(msg.get("content") or "", msg.get("reasoning_content") or "")


First = collections.namedtuple("First", "body text rows why")
Residency = collections.namedtuple("Residency", "word moves note err")


def residency_word(text):
    """The residency word the load ran by, read from the server's own records in `text` (its stderr): the `residency lever`
    record, else `residency unset`, else `residency host`; None when it printed none."""
    lines = text.splitlines()
    found = {}
    for b in RECORD_BINS:
        for r in records.read(lines, schema=schema(b)):
            if r.kind in RESIDENCY_KINDS and "residency" in r.fields:
                found.setdefault(r.kind, r["residency"])
    return next((found[k] for k in RESIDENCY_KINDS if k in found), None)


def logprob_rows(obj, who):
    """The rows of a chat reply's choices[0].logprobs.content as (id, token, [(id, token, logprob), ...]); CheckFail when
    the reply has none or a row without an id."""
    content = (((obj.get("choices") or [{}])[0]).get("logprobs") or {}).get("content")
    if not isinstance(content, list) or not content:
        raise CheckFail(f"{who} carries no logprobs.content although top_logprobs {LOGPROBS_TOP} was asked")
    rows = []
    for i, e in enumerate(content):
        try:
            tops = [(int(t["id"]), t.get("token", ""), float(t["logprob"])) for t in e["top_logprobs"]]
            rows.append((int(e["id"]), e.get("token", ""), tops))
        except (KeyError, TypeError, ValueError):
            raise CheckFail(f"{who}: logprobs.content[{i}] has no id, or a top_logprobs entry has no id or logprob")
    return rows


def split_index(a, b):
    """The index of the first token id two replies' rows differ at; None when their ids are the same."""
    for i in range(min(len(a), len(b))):
        if a[i][0] != b[i][0]:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def top2_gap(rows, i):
    """Row i's top-2 gap (the best logprob less the next, in nats); None when there is no such row or it names one token."""
    if i >= len(rows) or len(rows[i][2]) < 2:
        return None
    best = sorted((t[2] for t in rows[i][2]), reverse=True)
    return best[0] - best[1]


def split_verdict(first, replay, note):
    """The detail of a replay that is the first reply or splits from it at a near tie; CheckFail for a split above the band.
    A flip moves the pair's gap by at least the first reply's top-2 gap g1 plus the replay's g2 (a lower bound on the
    reversed gap), so g1 + g2 is held to the band; a reply that ended at the split has no gap there and counts 0."""
    i = split_index(first, replay)
    if i is None:
        return f"the replay of pass_short has the same {len(first)} ids ({note})"

    def tok(rows):
        return f"{rows[i][1]!r} (id {rows[i][0]})" if i < len(rows) else "<end of reply>"

    g1, g2 = top2_gap(first, i), top2_gap(replay, i)
    where = f"the replay splits at token {i} of {len(first)}: first {tok(first)}, replay {tok(replay)}"
    if g1 is None and g2 is None:
        raise CheckFail(f"{where}; neither reply has a top-2 row there to tell a near tie from a leak ({note})")
    total = (g1 or 0.0) + (g2 or 0.0)
    gaps = " + ".join(f"{g:.4f}" if g is not None else "none" for g in (g1, g2))
    if total > SPLIT_GAP_BAND:
        raise CheckFail(f"{where}; the first reply's and the replay's top-2 gaps {gaps} = {total:.4f} are above the band "
                        f"{SPLIT_GAP_BAND}, so not a near tie ({note})")
    return (f"{where}; the first reply's and the replay's top-2 gaps {gaps} = {total:.4f} are within the band {SPLIT_GAP_BAND}, "
            f"the texts after it are not compared ({note})")


def read_tape(path):
    """The parsed run file `toktape record` wrote at `path`; CheckFail when it is missing, unreadable or of another schema."""
    try:
        raw = Path(path).read_bytes()
    except OSError as e:
        raise CheckFail(f"toktape wrote no tape at {path}: {e.strerror}")
    try:
        raw = gzip.decompress(raw)
    except OSError:
        pass
    try:
        obj = json.loads(raw)
    except ValueError:
        raise CheckFail(f"the tape {path} is no JSON")
    if not isinstance(obj, dict) or obj.get("schema") != TAPE_SCHEMA:
        raise CheckFail(f"the tape {path} has schema {obj.get('schema') if isinstance(obj, dict) else None!r}, "
                        f"this reader knows {TAPE_SCHEMA}")
    return obj


def tape_reply(obj, prompt):
    """(text, field, evaluated) of the one request a tape holds, which must be the check's: CheckFail on a tape of
    another shape, with an error, of another prompt, or whose completion is not its streamed tokens' text."""
    reqs = obj.get("requests") or []
    if len(reqs) != 1:
        raise CheckFail(f"the tape holds {len(reqs)} requests, want 1")
    rec = reqs[0]
    if rec.get("error"):
        raise CheckFail(f"the tape's request failed: {head(rec['error'])}")
    p = rec.get("prompt") or {}
    msgs = p.get("messages") or [{}]
    sent = msgs[-1].get("content")
    if sent != prompt:
        raise CheckFail(f"the tape recorded another prompt than the check's: {head(sent, 60)!r}, want {head(prompt, 60)!r}")
    toks = rec.get("tokens") or []
    completion = p.get("completion") or ""
    streamed = "".join(t.get("text", "") for t in toks if not t.get("r"))
    if completion != streamed:
        raise CheckFail(f"the tape's completion ({len(completion)} chars) is not the text of its streamed tokens "
                        f"({len(streamed)} chars)")
    reasoning = "".join(t.get("text", "") for t in toks if t.get("r"))
    text, field = choose_text(completion, reasoning)
    return text, field, (rec.get("timings") or {}).get("prompt_n")


def refusal_of(text):
    """What a server that ended printed last as its reason: the last `bloomery-serve*: <text>` line that is neither
    the start banner nor a record, and the non-record lines after it; None when the log has none."""
    lines = text.splitlines()
    schemas = [schema(b) for b in RECORD_BINS]

    def is_rec(line):
        return any(sc.read_line(line) is not None for sc in schemas)

    start = None
    for i, line in enumerate(lines):
        if HEAD.match(line) and not BANNER.match(line) and not is_rec(line):
            start = i
    if start is None:
        return None
    block = [HEAD.sub("", lines[start], count=1).strip()]
    for line in lines[start + 1:]:
        if line.strip() and not is_rec(line):
            block.append(line.strip())
    return head(" / ".join(block[:REASON_LINES]), 500)


class Session:
    def __init__(self, a):
        self.a = a
        self.fails = 0
        self.dead = None          # why the server is not up, once it is known not to be
        self.first = None         # First(body, text, logprob rows, why they are missing) of pass_short; the body is None under --client toktape
        self.residency = Residency(None, False, "residency not read", None)

    def say(self, name, status, detail=""):
        if status == "FAIL":
            self.fails += 1
        print(f"check {name}: {status}" + (f" {head(detail, 400)}" if detail else ""), flush=True)

    def stderr_text(self):
        try:
            return Path(self.a.stderr).read_text(errors="replace") if self.a.stderr else None
        except OSError:
            return None

    def save(self, name, body, status, text):
        if self.a.dir:
            (Path(self.a.dir) / f"{name}.json").write_text(
                json.dumps({"request": body, "status": status, "response": text[:20000]}, indent=1))

    def info(self):
        text = self.stderr_text()
        if text is None:
            return
        recs = records.read(text.splitlines(), schema=schema("generate_ds41"))
        plan, load = records.first(recs, "plan"), records.first(recs, "load")
        parts = []
        if plan is not None:
            parts += [f"place={plan.get('place')}", f"ctx_max={plan.get('ctx_max')}"]
        if load is not None:
            parts.append(f"load_ctx={load.get('ctx')}")
        if parts:
            print("session-plan: " + " ".join(parts), flush=True)

    def read_residency(self):
        """Whether the server moves experts, from the residency word its stderr records: Residency(word, moves, note, err),
        the note naming the fact for a check's detail and err a word this check does not know."""
        text = self.stderr_text()
        if text is None:
            return Residency(None, False, "no --stderr, so no residency record was read", None)
        word = residency_word(text)
        if word is None:
            return Residency(None, False, "no residency record in the --stderr", None)
        if not RESIDENCY_WORD.match(word):
            return Residency(word, False, f"residency {word}",
                             f"harness: the residency record names {word!r}, none of off or mid-p<P>-s<S>[-unhosted]")
        return Residency(word, word != "off", f"residency {word}", None)

    def probe_down(self):
        """None while the server answers /health and its process runs; else why it is down."""
        why = None
        if self.a.pid and not alive(self.a.pid):
            why = f"process {self.a.pid} has exited"
        else:
            try:
                status, _ = request(self.a.url, "GET", "/health", None, PROBE_TIMEOUT)
                if status == 200:
                    return None
                why = f"/health answered HTTP {status}"
            except Unreachable as e:
                why = f"/health: {e}"
        text = self.stderr_text()
        said = refusal_of(text) if text is not None else None
        return f"{why}; its last words: {said}" if said else why

    def post_chat_obj(self, name, body):
        """The parsed reply of one chat request; CheckFail on a status other than 200 or a reply that is no JSON."""
        status, text = request(self.a.url, "POST", CHAT_PATH, body, self.a.timeout)
        self.save(name, body, status, text)
        if status != 200:
            raise CheckFail(f"HTTP {status}: {head(text)}")
        try:
            return json.loads(text)
        except ValueError:
            raise CheckFail(f"the reply is not JSON: {head(text)}")

    def post_chat(self, name, body):
        """(text, field) of one chat request; CheckFail on a status other than 200, a reply that is no JSON, or no text."""
        return reply_text(self.post_chat_obj(name, body))

    def asked_reply(self, name, body, who):
        """(text, field, rows, why) of a chat request that asked for logprobs: the rows are None, with the reason in `why`,
        when the reply carries none; CheckFail as `post_chat` for a reply with no text."""
        obj = self.post_chat_obj(name, body)
        text, field = reply_text(obj)
        try:
            return text, field, logprob_rows(obj, who), None
        except CheckFail as e:
            return text, field, None, str(e)

    def tokenized(self, text):
        st, reply = request(self.a.url, "POST", "/tokenize", {"content": text, "add_special": False}, self.a.timeout)
        if st != 200:
            raise CheckFail(f"harness: /tokenize answered HTTP {st}: {head(reply)}")
        try:
            return len(json.loads(reply)["tokens"])
        except (ValueError, KeyError, TypeError):
            raise CheckFail(f"harness: /tokenize's reply has no tokens list: {head(reply)}")

    def record(self, kind, suffix, prompt):
        """One `toktape record` of `prompt`, and (text, field, evaluated, name) of the reply its tape holds."""
        a = self.a
        name = f"{a.tag}-{kind}{suffix}"
        prompts = Path(a.dir) / f"{name}.jsonl"
        prompts.write_text(json.dumps({"name": kind, "prompt": prompt}, ensure_ascii=False) + "\n")
        tapes = Path(a.dir) / "tapes"
        cmd = [a.toktape_bin, "record", "--url", a.url, "--temp", "0", "--no-think", "--endpoint", "chat",
               "-n", str(a.max_tokens), "--for", "0", "--wait", "60s", "--host-label", "none",
               "--prompts", str(prompts), "--out", str(tapes), "--tag", name, "-o", "json", "--quiet"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
        except subprocess.TimeoutExpired:
            raise CheckFail(f"toktape did not end within {a.timeout:g} s")
        (Path(a.dir) / f"{name}.json").write_text(r.stdout)
        if r.returncode != 0:
            raise CheckFail(f"toktape exit {r.returncode}: {head(r.stderr or r.stdout)}")
        try:
            run_id = json.loads(r.stdout)["id"]
        except (ValueError, KeyError, TypeError):
            raise CheckFail(f"toktape's summary names no run id: {head(r.stdout)}")
        text, field, evaluated = tape_reply(read_tape(tapes / f"{run_id}.toktape"), prompt)
        return text, field, evaluated, name

    def tape_detail(self, name, evaluated, text, field):
        return f"tape {name}, {evaluated} prompt tokens evaluated, {len(text)} chars from {field}"

    def pass_short(self):
        top = LOGPROBS_TOP if self.residency.moves else 0
        if self.a.client == "toktape":
            text, field, evaluated, name = self.record("short", "", SHORT_PROMPT)
            rows = why = None
            if top:
                # A tape holds no logprobs: the pair the replay check judges is this direct request and the one after the long
                # prompt; a failure of it is that check's, not pass_short's (the short prompt was answered).
                try:
                    _, _, rows, why = self.asked_reply("pass_short_probe", chat_body(SHORT_PROMPT, self.a.max_tokens, top),
                                                       "the direct request after pass_short's tape")
                except CheckFail as e:
                    why = f"the direct request after pass_short's tape failed: {e}"
            self.first = First(None, text, rows, why)
            return self.tape_detail(name, evaluated, text, field)
        body = chat_body(SHORT_PROMPT, self.a.max_tokens, top)
        if top:
            text, field, rows, why = self.asked_reply("pass_short", body, "pass_short's reply")
        else:
            text, field = self.post_chat("pass_short", body)
            rows = why = None
        self.first = First(body, text, rows, why)
        return f"{len(text)} chars from {field}"

    def pass_long(self):
        prompt = (LONG_SENTENCE * self.a.long_repeat).strip()
        tokens = self.tokenized(prompt)
        if tokens < LONG_MIN_TOKENS:
            raise CheckFail(f"harness: the long prompt is {tokens} tokens by /tokenize, under {LONG_MIN_TOKENS}")
        if self.a.client == "toktape":
            text, field, evaluated, name = self.record("long", "", prompt)
            return f"{tokens} prompt tokens, " + self.tape_detail(name, evaluated, text, field)
        text, field = self.post_chat("pass_long", chat_body(prompt, self.a.max_tokens))
        return f"{tokens} prompt tokens, {len(text)} chars from {field}"

    def second_request_equals_first(self):
        if self.first is None:
            raise CheckSkip("pass_short gave no reply to replay")
        if self.residency.err:
            raise CheckFail(self.residency.err)
        if self.residency.moves:
            return self.replay_of_a_moving_server()
        body, want = self.first.body, self.first.text
        if body is None:
            text, _, _, _ = self.record("short", "2", SHORT_PROMPT)
        else:
            text, _ = self.post_chat("second_request_equals_first", body)
        if text != want:
            raise CheckFail(f"the replay's text differs: first {head(want, 60)!r}, replay {head(text, 60)!r} "
                            f"({self.residency.note}, so the replay is held exact)")
        return f"the replay of pass_short has the same {len(text)} chars"

    def replay_of_a_moving_server(self):
        """The replay on a server that moves experts: its ids the first reply's, or a split at a near tie (`split_verdict`)."""
        first, note = self.first, self.residency.note
        if first.why:
            raise CheckFail(f"{first.why} ({note} and moves experts, so the replay is judged by logprobs)")
        if first.body is None:
            body = chat_body(SHORT_PROMPT, self.a.max_tokens, LOGPROBS_TOP)
            _, _, rows, why = self.asked_reply("second_request_probe", body, "the direct request after the long prompt")
            tape, _, _, _ = self.record("short", "2", SHORT_PROMPT)
            tapes = "agree" if tape == first.text else "differ"
            note += f", experts move; tapes {tapes}, direct pair judged"
        else:
            _, _, rows, why = self.asked_reply("second_request_equals_first", first.body, "the replay")
            note += ", experts move"
        if why:
            raise CheckFail(f"{why} ({note})")
        return split_verdict(first.rows, rows, note)

    def off_reason(self):
        if not self.a.stderr:
            self.say("off_reason", "FAIL", "harness: no --stderr, so the server's refusal cannot be read")
            return
        text = self.stderr_text()
        said = refusal_of(text) if text is not None else None
        if said:
            self.say("off_reason", "PASS", said)
        else:
            self.say("off_reason", "FAIL", f"no refusal line in {self.a.stderr} (the start banner and records are no reason)")

    def steps(self):
        return [("pass_short", self.pass_short), ("pass_long", self.pass_long),
                ("second_request_equals_first", self.second_request_equals_first)]

    def run(self):
        self.info()
        self.residency = self.read_residency()
        self.dead = self.probe_down()
        for name, fn in self.steps():
            if self.dead:
                self.say(name, "SKIP", f"server not up: {self.dead}")
                continue
            try:
                self.say(name, "PASS", fn())
            except CheckFail as e:
                self.say(name, "FAIL", str(e))
            except CheckSkip as e:
                self.say(name, "SKIP", str(e))
            except Unreachable as e:
                self.say(name, "FAIL", f"no answer: {e}")
            self.dead = self.probe_down()
        if self.dead is not None:
            self.off_reason()
        rc = 1 if self.fails else 0
        print(f"SESSION rc={rc}", flush=True)
        return rc


class Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        print(f"pass_checks.py: {message}", file=sys.stderr)
        sys.exit(64)


def refuse(msg):
    print(f"pass_checks.py: {msg}", file=sys.stderr)
    sys.exit(2)


def main(argv, env=None):
    env = os.environ if env is None else env
    if argv[:1] == ["off-reason"]:
        ap = Parser(prog="pass_checks.py off-reason")
        ap.add_argument("--stderr", required=True)
        a = ap.parse_args(argv[1:])
        try:
            said = refusal_of(Path(a.stderr).read_text(errors="replace"))
        except OSError as e:
            refuse(f"cannot read {a.stderr}: {e}")
        if said:
            print(said)
        return 0 if said else 1
    ap = Parser(prog="pass_checks.py")
    ap.add_argument("url", nargs="?", default=env.get("BLOOMERY_SESSION_URL"))
    ap.add_argument("--pid", default=env.get("BLOOMERY_SESSION_PID"))
    ap.add_argument("--stderr", default=env.get("BLOOMERY_SESSION_STDERR"))
    ap.add_argument("--dir", default=env.get("BLOOMERY_SESSION_DIR"))
    ap.add_argument("--tag", default="pass")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--long-repeat", type=int, default=450)
    ap.add_argument("--client", choices=("http", "toktape"), default="http")
    ap.add_argument("--toktape-bin")
    a = ap.parse_args(argv)
    if not a.url:
        ap.error("no URL: give one, or set BLOOMERY_SESSION_URL")
    if not re.match(r"^http://[^/:]+(:\d+)?/?$", a.url):
        ap.error(f"the URL {a.url!r} is not http://host[:port]")
    if a.client == "toktape":
        if not a.dir:
            ap.error("--client toktape needs --dir (the tapes and their prompt files go there)")
        a.toktape_bin = a.toktape_bin or shutil.which("toktape")
        if not a.toktape_bin:
            refuse("--client toktape: toktape is not on PATH (--toktape-bin names it)")
    if a.dir:
        Path(a.dir).mkdir(parents=True, exist_ok=True)
    if a.stderr and not Path(a.stderr).is_file():
        refuse(f"--stderr {a.stderr} is not a file")
    return Session(a).run()


# ---- self-test ----

class Stub:
    """tools/ref/stub-llama-server.py as a child: its stdout read for the listening line, its stderr in a file."""

    def __init__(self, tmp, env, args):
        e = dict(os.environ)
        e["TMPDIR"] = str(tmp)
        e.update(env)
        self.err = Path(tmp) / "stub.err"
        self.errf = open(self.err, "wb")
        self.proc = subprocess.Popen([sys.executable, "-I", str(STUB), "--port", "0", *args], stdout=subprocess.PIPE,
                                     stderr=self.errf, env=e, text=True)
        self.port = None
        timer = threading.Timer(30, self.proc.kill)
        timer.start()
        for line in self.proc.stdout:
            m = re.search(r"listening on http://127\.0\.0\.1:(\d+)", line)
            if m:
                self.port = int(m.group(1))
                break
        timer.cancel()
        if self.port is None:
            self.proc.wait()

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
        self.proc.wait()
        self.proc.stdout.close()
        self.errf.close()


SCRATCH = []


def scratch(prefix):
    """A temporary directory the self-test removes when it ends."""
    d = Path(tempfile.mkdtemp(prefix=prefix))
    SCRATCH.append(d)
    return d


def closed_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def parse(lines):
    by, order, rc = {}, [], None
    for line in lines:
        m = re.match(r"check (\w+): (PASS|FAIL|SKIP)(?: (.*))?$", line)
        if m:
            by[m.group(1)] = (m.group(2), m.group(3) or "")
            order.append(m.group(1))
        m = re.match(r"SESSION rc=(\d+)$", line)
        if m:
            rc = int(m.group(1))
    return by, order, rc


class Run:
    """One session of `main` against a fresh stub child."""

    def __init__(self, main, stub_env=None, stub_args=(), argv=(), env_style=False, pid=True):
        self.tmp = scratch("pass-checks-st.")
        self.stub = Stub(self.tmp, stub_env or {}, stub_args)
        try:
            port = self.stub.port or closed_port()
            url = f"http://127.0.0.1:{port}"
            pid_s = str(self.stub.proc.pid)
            if env_style:
                env = {"BLOOMERY_SESSION_URL": url, "BLOOMERY_SESSION_PID": pid_s,
                       "BLOOMERY_SESSION_STDERR": str(self.stub.err), "BLOOMERY_SESSION_DIR": str(self.tmp / "out")}
                args = list(argv)
            else:
                env = {}
                args = [url, "--stderr", str(self.stub.err), "--dir", str(self.tmp / "out")] + list(argv)
                if pid:
                    args += ["--pid", pid_s]
            args += ["--timeout", "20"]
            self.rc, self.lines, self.raised = call(main, args, env)
            self.by, self.order, self.rc_line = parse(self.lines)
        finally:
            self.stub.close()

    def chats(self):
        f = self.tmp / "stub-srv-chats"
        return [x.split() for x in f.read_text().splitlines()] if f.exists() else []


def call(main, argv, env):
    out = io.StringIO()
    rc, raised = None, None
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            rc = main(argv, env)
    except SystemExit as e:
        rc = e.code
    except Exception as e:
        raised = f"{type(e).__name__}: {e}"
    return rc, out.getvalue().splitlines(), raised


CASES = {}


def case(fn):
    CASES[fn.__name__.replace("_", "-")] = fn
    return fn


def verdicts(run, **want):
    """Problems of `run` against `name=STATUS` wishes, and that the run did not raise."""
    out = [f"raised {run.raised}"] if run.raised else []
    for name, status in want.items():
        got = run.by.get(name, ("missing", ""))[0]
        if got != status:
            out.append(f"{name} is {got}, want {status}")
    return out


def rc_problem(r, want):
    return [] if r.rc_line == want and r.rc == want else [f"SESSION rc {r.rc_line}, exit {r.rc}, want {want}"]


@case
def healthy_server_through_the_environment(main):
    r = Run(main, {"STUB_SRV_PLAN": "16384"}, ["--place", "a"], env_style=True)
    p = verdicts(r, pass_short="PASS", pass_long="PASS", second_request_equals_first="PASS") + rc_problem(r, 0)
    if r.order != ["pass_short", "pass_long", "second_request_equals_first"]:
        p.append(f"checks {r.order}: an off_reason or a tape on a server that is up")
    if "session-plan: place=a ctx_max=16384 load_ctx=16384" not in r.lines:
        p.append(f"no session-plan line from the plan and load records: {r.lines[:2]}")
    ch = r.chats()
    if len(ch) != 3 or ch[0][2] != ch[2][2] or ch[0][2] == ch[1][2] or {c[1] for c in ch} != {"128"}:
        p.append(f"the requests are not short, long, the short again at 128 tokens: {ch}")
    kept = sorted(f.name for f in (r.tmp / "out").glob("*.json"))
    if kept != ["pass_long.json", "pass_short.json", "second_request_equals_first.json"]:
        p.append(f"the requests kept in --dir: {kept}")
    return p


@case
def reasoning_only_reply(main):
    r = Run(main, {"STUB_SRV_CHAT_MODE": "reasoning"})
    p = verdicts(r, pass_short="PASS", pass_long="PASS", second_request_equals_first="PASS") + rc_problem(r, 0)
    if "reasoning_content" not in r.by.get("pass_short", ("", ""))[1]:
        p.append("pass_short's detail does not name reasoning_content")
    return p


@case
def both_empty_reply(main):
    p = []
    for mode in ("empty", "null"):
        r = Run(main, {"STUB_SRV_CHAT_MODE": mode})
        p += [f"{mode}: {x}" for x in verdicts(r, pass_short="FAIL", pass_long="FAIL", second_request_equals_first="SKIP")
              + rc_problem(r, 1)]
    return p


@case
def long_prompt_under_4096_tokens(main):
    r = Run(main, {"STUB_SRV_TOKENIZE_N": "4061"})
    p = verdicts(r, pass_short="PASS", pass_long="FAIL", second_request_equals_first="PASS") + rc_problem(r, 1)
    detail = r.by.get("pass_long", ("", ""))[1]
    if not detail.startswith("harness:") or "4061" not in detail:
        p.append(f"pass_long's detail is not a `harness:` line naming 4061: {detail!r}")
    if len(r.chats()) != 2:
        p.append(f"the long prompt was sent anyway: {len(r.chats())} chat requests, want 2")
    r = Run(main, {"STUB_SRV_TOKENIZE_N": "4096"})
    p += [f"4096: {x}" for x in verdicts(r, pass_long="PASS") + rc_problem(r, 0)]
    return p


@case
def second_request_500(main):
    r = Run(main, {"STUB_SRV_FAIL_AT": "3"})
    p = verdicts(r, pass_short="PASS", pass_long="PASS", second_request_equals_first="FAIL") + rc_problem(r, 1)
    if "500" not in r.by.get("second_request_equals_first", ("", ""))[1]:
        p.append("second_request_equals_first's detail does not name the 500")
    return p


@case
def a_fail_does_not_stop_the_next_check(main):
    r = Run(main, {"STUB_SRV_FAIL_AT": "2"})
    p = verdicts(r, pass_short="PASS", pass_long="FAIL", second_request_equals_first="PASS") + rc_problem(r, 1)
    if r.order[:3] != ["pass_short", "pass_long", "second_request_equals_first"]:
        p.append(f"order {r.order}")
    return p


@case
def a_dead_server_skips_the_rest_naming_why(main):
    r = Run(main, {"STUB_SRV_DIE_AT": "2"})
    p = verdicts(r, pass_short="PASS", pass_long="FAIL", second_request_equals_first="SKIP", off_reason="PASS")
    p += rc_problem(r, 1)
    for name in ("second_request_equals_first", "off_reason"):
        if "engine fault at request 2" not in r.by.get(name, ("", ""))[1]:
            p.append(f"{name}'s detail does not name the server's last words")
    return p


@case
def a_replay_that_differs_fails(main):
    r = Run(main, {"STUB_SRV_DRIFT": "1"})
    return verdicts(r, pass_short="PASS", pass_long="PASS", second_request_equals_first="FAIL") + rc_problem(r, 1)


REFUSAL = "--ctx 16384 over 1 slot(s) asks 16384 positions a slot, and the card holds 11502 (stub)"


@case
def an_off_cell_reads_the_refusal_not_the_banner(main):
    r = Run(main, {"STUB_SRV_BANNER": "1", "STUB_SRV_PLAN": "16384", "STUB_SRV_REFUSE": REFUSAL}, ["--place", "a"])
    p = verdicts(r, pass_short="SKIP", pass_long="SKIP", second_request_equals_first="SKIP", off_reason="PASS")
    p += rc_problem(r, 0)
    if r.by.get("off_reason", ("", ""))[1] != REFUSAL:
        p.append(f"off_reason's detail is {r.by.get('off_reason', ('', ''))[1]!r}, want the refusal text")
    if "session-plan: place=a ctx_max=16384" not in r.lines:
        p.append(f"the refused server's plan record is not read, or its load is: {r.lines[:1]}")
    for line in r.lines:
        if "first start" in line or "place unset" in line or "plan place" in line:
            p.append(f"a banner or record line in the output: {line!r}")
    return p


@case
def an_off_cell_with_only_the_banner_is_unnamed(main):
    r = Run(main, {"STUB_SRV_BANNER": "1", "STUB_SRV_EXIT": "1"})
    p = verdicts(r, pass_short="SKIP", off_reason="FAIL") + rc_problem(r, 1)
    if "first start" in r.by.get("off_reason", ("", ""))[1]:
        p.append("the banner is given as the reason")
    return p


# A stand-in for `toktape record`: it POSTs the prompt of its --prompts line to the stub server's chat endpoint, writes the
# run file (gzip'd JSON, schema 1) at <out>/<id>.toktape with the reply as the recorder keeps it (the completion, and the
# streamed tokens with the reasoning ones flagged), and prints the summary. FAKE_TOKTAPE_MODE (comma-separated) bends the tape:
# otherprompt, splitdiff (tokens that are not the completion), error (a request error, exit 0), two (two requests), schema2,
# plain (not gzip'd); FAKE_TOKTAPE_RC ends it with that status and no tape; every call's argv is appended to FAKE_TOKTAPE_LOG.
FAKE_TOKTAPE = r"""#!@PY@
import gzip, http.client, json, os, re, sys, urllib.parse
argv = sys.argv[1:]
if os.environ.get("FAKE_TOKTAPE_LOG"):
    with open(os.environ["FAKE_TOKTAPE_LOG"], "a") as f:
        f.write(" ".join(argv) + "\n")
def opt(name):
    return argv[argv.index(name) + 1]
modes = os.environ.get("FAKE_TOKTAPE_MODE", "").split(",")
rc = int(os.environ.get("FAKE_TOKTAPE_RC") or 0)
if rc:
    print(json.dumps({"error": {"code": "streams"}}))
    sys.exit(rc)
prompt = json.loads(open(opt("--prompts")).readline())["prompt"]
u = urllib.parse.urlsplit(opt("--url"))
body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": int(opt("-n")), "temperature": 0,
        "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
try:
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=30)
    conn.request("POST", "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    resp = conn.getresponse()
    status, raw = resp.status, resp.read()
except (OSError, http.client.HTTPException):
    print(json.dumps({"error": {"code": "unreachable"}}))
    sys.exit(2)
if status // 100 != 2:
    print(json.dumps({"error": {"code": "streams"}}))
    sys.exit(3)
msg = json.loads(raw)["choices"][0]["message"]
content, reasoning = msg.get("content") or "", msg.get("reasoning_content") or ""
toks = [{"i": i, "text": w} for i, w in enumerate(re.findall(r"\S+\s*", content))]
toks += [{"i": len(toks) + i, "text": w, "r": True} for i, w in enumerate(re.findall(r"\S+\s*", reasoning))]
if "splitdiff" in modes:
    toks.append({"i": len(toks), "text": " and more"})
shown = "something else" if "otherprompt" in modes else prompt
rec = {"index": 0, "prompt": {"messages": [{"role": "user", "content": shown}], "completion": content, "max_tokens": 128},
       "tokens": toks, "timings": {"prompt_n": len(prompt.split())}}
if "error" in modes:
    rec["error"] = "stream cut"
tape = {"schema": 2 if "schema2" in modes else 1, "summary": {}, "requests": [rec, rec] if "two" in modes else [rec]}
run_id = opt("--tag") + "-" + str(os.getpid())
os.makedirs(opt("--out"), exist_ok=True)
data = json.dumps(tape).encode()
with open(os.path.join(opt("--out"), run_id + ".toktape"), "wb") as f:
    f.write(data if "plain" in modes else gzip.compress(data))
print(json.dumps({"id": run_id, "timings": {"prompt_n": len(prompt.split())}}))
"""


def tape_run(main, stub_env=None, mode="", rc=0, extra=()):
    """A session with --client toktape through the stand-in; its toktape calls are in `.calls`."""
    d = scratch("pass-checks-tt.")
    fake, log = d / "toktape", d / "argv"
    fake.write_text(FAKE_TOKTAPE.replace("@PY@", sys.executable))
    fake.chmod(0o755)
    env = {"FAKE_TOKTAPE_LOG": str(log), "FAKE_TOKTAPE_MODE": mode, "FAKE_TOKTAPE_RC": str(rc)}
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        r = Run(main, stub_env, argv=["--client", "toktape", "--toktape-bin", str(fake), "--tag", "rel", *extra])
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    r.calls = log.read_text().splitlines() if log.exists() else []
    return r


THREE = ["pass_short", "pass_long", "second_request_equals_first"]


@case
def toktape_client_checks_are_the_tapes(main):
    r = tape_run(main, {"STUB_SRV_PLAN": "16384"})
    p = verdicts(r, pass_short="PASS", pass_long="PASS", second_request_equals_first="PASS") + rc_problem(r, 0)
    if r.order != THREE:
        p.append(f"checks {r.order}: want the three, with no tape check of their own")
    ch = r.chats()
    if len(ch) != 3 or ch[0][2] != ch[2][2] or ch[0][2] == ch[1][2] or {c[1] for c in ch} != {"128"}:
        p.append(f"the requests are not the three tapes' (short, long, the short again at 128 tokens, none of the check's own): {ch}")
    flags = (r"record --url http://127\.0\.0\.1:\d+ --temp 0 --no-think --endpoint chat -n 128 --for 0 --wait 60s "
             r"--host-label none --prompts \S+ --out \S+ --tag rel-%s -o json --quiet$")
    if len(r.calls) != 3 or not all(re.match(flags % tag, c) for c, tag in zip(r.calls, ("short", "long", "short2"))):
        p.append(f"toktape's argv is not the release pass's, short, long, short2: {r.calls}")
    kept = sorted(f.name for f in (r.tmp / "out").iterdir())
    want = sorted(["rel-short.json", "rel-long.json", "rel-short2.json", "rel-short.jsonl", "rel-long.jsonl",
                   "rel-short2.jsonl", "tapes"])
    if kept != want:
        p.append(f"kept in --dir: {kept}, want {want}")
    if len(list((r.tmp / "out" / "tapes").glob("*.toktape"))) != 3:
        p.append("the three tapes are not in --dir/tapes")
    detail = r.by.get("pass_long", ("", ""))[1]
    if not re.match(r"\d+ prompt tokens, tape rel-long, \d+ prompt tokens evaluated, \d+ chars from content$", detail):
        p.append(f"pass_long's detail does not name its tape and the tokens: {detail!r}")
    r = tape_run(main, {"STUB_SRV_CHAT_MODE": "reasoning"}, mode="plain")
    p += [f"reasoning, plain tape: {x}" for x in verdicts(r, pass_short="PASS", pass_long="PASS", second_request_equals_first="PASS")]
    if "reasoning_content" not in r.by.get("pass_short", ("", ""))[1]:
        p.append("a reasoning-only tape: pass_short's detail does not name reasoning_content")
    for mode in ("empty", "null"):
        r = tape_run(main, {"STUB_SRV_CHAT_MODE": mode})
        p += [f"{mode}: {x}" for x in verdicts(r, pass_short="FAIL", pass_long="FAIL", second_request_equals_first="SKIP")
              + rc_problem(r, 1)]
    return p


@case
def toktape_client_failures_name_their_check(main):
    p = []
    r = tape_run(main, rc=3)
    p += [f"toktape exit 3: {x}" for x in verdicts(r, pass_short="FAIL", pass_long="FAIL", second_request_equals_first="SKIP")
          + rc_problem(r, 1)]
    if "toktape exit 3" not in r.by.get("pass_long", ("", ""))[1]:
        p.append("pass_long's detail does not name toktape's exit status")
    r = tape_run(main, {"STUB_SRV_DRIFT": "1"})
    p += [f"drift: {x}" for x in verdicts(r, pass_short="PASS", pass_long="PASS", second_request_equals_first="FAIL")
          + rc_problem(r, 1)]
    r = tape_run(main, {"STUB_SRV_FAIL_AT": "3"})
    p += [f"500 at the replay: {x}" for x in verdicts(r, pass_short="PASS", pass_long="PASS", second_request_equals_first="FAIL")
          + rc_problem(r, 1)]
    if "toktape exit 3" not in r.by.get("second_request_equals_first", ("", ""))[1]:
        p.append("500 at the replay: the detail does not name toktape's exit status")
    r = tape_run(main, {"STUB_SRV_TOKENIZE_N": "4061"})
    p += [f"4061: {x}" for x in verdicts(r, pass_short="PASS", pass_long="FAIL", second_request_equals_first="PASS")
          + rc_problem(r, 1)]
    if len(r.calls) != 2 or len(r.chats()) != 2:
        p.append(f"4061: the long prompt was recorded anyway: {len(r.calls)} toktape calls, {len(r.chats())} chat requests, want 2")
    r = tape_run(main, {"STUB_SRV_DIE_AT": "2"})
    p += [f"dead at the long one: {x}" for x in verdicts(r, pass_short="PASS", pass_long="FAIL",
                                                         second_request_equals_first="SKIP", off_reason="PASS")
          + rc_problem(r, 1)]
    if "engine fault at request 2" not in r.by.get("second_request_equals_first", ("", ""))[1]:
        p.append("dead at the long one: the skip does not name the server's last words")
    return p


@case
def toktape_tape_of_another_shape_fails(main):
    p = []
    for mode, word in (("otherprompt", "another prompt"), ("splitdiff", "streamed tokens"), ("error", "request failed"),
                       ("two", "2 requests"), ("schema2", "schema")):
        r = tape_run(main, mode=mode)
        p += [f"{mode}: {x}" for x in verdicts(r, pass_short="FAIL") + rc_problem(r, 1)]
        if word not in r.by.get("pass_short", ("", ""))[1]:
            p.append(f"{mode}: pass_short's detail does not say '{word}': {r.by.get('pass_short')}")
    return p


# The replay cases of a server that moves experts. The stub child answers no logprobs, so these run against Scripted: a chat
# server in this process whose k-th chat reply is script[k] = (text, rows), the rows going out as choices[0].logprobs.content
# when the request asked for logprobs and rows is not None.
class Scripted:
    def __init__(self, script):
        outer = self
        self.script, self.bodies = list(script), []

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def reply(self, code, obj):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                self.reply(200 if self.path == "/health" else 404, {"status": "ok"})

            def do_POST(self):
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/tokenize":
                    return self.reply(200, {"tokens": list(range(5000))})
                outer.bodies.append(req)
                k = len(outer.bodies) - 1
                text, rows = outer.script[k] if k < len(outer.script) else ("past the script", None)
                choice = {"index": 0, "finish_reason": "length",
                          "message": {"role": "assistant", "content": text, "reasoning_content": ""}}
                if rows is not None and req.get("logprobs"):
                    choice["logprobs"] = {"content": [
                        {"id": i, "token": t, "logprob": top[0][2],
                         "top_logprobs": [{"id": a, "token": b, "logprob": c} for a, b, c in top]} for i, t, top in rows]}
                self.reply(200, {"choices": [choice], "usage": {}})

        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.srv.daemon_threads = True
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


def rows_of(ids, gaps=None):
    """Rows for token ids `ids`: row i names the id and id + 100000, `gaps[i]` nats apart (3.0 where none is given)."""
    gaps = gaps or {}
    return [(t, f"t{t}", [(t, f"t{t}", -0.1), (t + 100000, f"u{t}", -0.1 - gaps.get(i, 3.0))]) for i, t in enumerate(ids)]


GLM_UNSET = """residency unset residency=mid-p0-s1 why=unset: --place a or bp read=MemAvailable room=260623720448 plan_room=260086849536 mem_left=82845655040
residency host residency=mid-p0-s1 hosted=hosted pinned=0 churn_experts=850 churn_bytes=13036814336 headroom=86135177216 headroom_after=73098362880 host_figure=nominal-A6000-workstation host_usable=270071001088
"""
LEVER_MID = "residency lever residency=mid-p0-s1 why=place\n"
HOST_ONLY = GLM_UNSET.splitlines(True)[1]
LEVER_OFF = "residency lever residency=off why=set\n"
LEVER_BAD = "residency lever residency=banana why=set\n"
NO_RESIDENCY = "plan place=a card=Stub ctx_max=16384 card_experts=1 (1 B) host_experts=0 (0 B) host_shadow=0 B n_l=1 on 1 layers\n"
SPLIT_AT = 2           # the token index the near-tie replays split at
IDS = [11, 12, 13, 14, 15]


def tie_script(gap, back=0.02, ids=IDS, at=SPLIT_AT, long_text="the long reply"):
    """The replies of short, long, short: the replay takes the runner-up at index `at`; the first reply's top-2 gap there is
    `gap` and the replay's is `back`."""
    first = rows_of(ids, {at: gap})
    runner = ids[at] + 100000
    swapped = [(runner, f"u{ids[at]}", [(runner, f"u{ids[at]}", -0.1), (ids[at], f"t{ids[at]}", -0.1 - back)])]
    replay = first[:at] + swapped + rows_of([x + 200000 for x in ids[at + 1:]])
    return [("short reply " + " ".join(f"t{i}" for i in ids), first), (long_text, None),
            ("short reply " + " ".join(f"t{i}" for i in ids[:at]) + f" u{ids[at]} later", replay)]


def scripted_run(main, script, stderr=None, argv=(), toktape=None):
    """One session of `main` against a Scripted server; `stderr` is the text of the --stderr file (None: no such flag).
    `toktape` (a mode string) puts the fake toktape on the run."""
    tmp = scratch("pass-checks-sc.")
    srv = Scripted(script)
    try:
        args = [srv.url, "--dir", str(tmp / "out"), "--timeout", "20"] + list(argv)
        if stderr is not None:
            (tmp / "err").write_text(stderr)
            args += ["--stderr", str(tmp / "err")]
        saved = {}
        if toktape is not None:
            fake, log = tmp / "toktape", tmp / "argv"
            fake.write_text(FAKE_TOKTAPE.replace("@PY@", sys.executable))
            fake.chmod(0o755)
            env = {"FAKE_TOKTAPE_LOG": str(log), "FAKE_TOKTAPE_MODE": toktape, "FAKE_TOKTAPE_RC": "0"}
            saved = {k: os.environ.get(k) for k in env}
            os.environ.update(env)
            args += ["--client", "toktape", "--toktape-bin", str(fake), "--tag", "rel"]
        try:
            rc, lines, raised = call(main, args, {})
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        r = types.SimpleNamespace(rc=rc, lines=lines, raised=raised, bodies=list(srv.bodies), tmp=tmp)
        r.by, r.order, r.rc_line = parse(lines)
        return r
    finally:
        srv.close()


def asked(body):
    return body.get("logprobs") is True and body.get("top_logprobs") == LOGPROBS_TOP


def detail(r, name="second_request_equals_first"):
    return r.by.get(name, ("missing", ""))[1]


@case
def a_moving_seat_that_splits_at_a_near_tie_passes(main):
    r = scripted_run(main, tie_script(0.05), GLM_UNSET)
    p = verdicts(r, pass_short="PASS", pass_long="PASS", second_request_equals_first="PASS") + rc_problem(r, 0)
    d = detail(r)
    for want in (f"splits at token {SPLIT_AT} of {len(IDS)}", "'t13' (id 13)", "'u13' (id 100013)", "gaps 0.0500 + 0.0200 = 0.0700",
                 "residency mid-p0-s1", "texts after it are not compared"):
        if want not in d:
            p.append(f"the detail lacks {want!r}: {d!r}")
    if [asked(b) for b in r.bodies] != [True, False, True]:
        p.append(f"the short requests ask logprobs and the long one does not: {r.bodies}")
    if {b.get("max_tokens") for b in r.bodies} != {128}:
        p.append(f"max_tokens drifted: {[b.get('max_tokens') for b in r.bodies]}")
    return p


@case
def a_moving_seat_with_the_same_ids_passes(main):
    ids = IDS
    script = [("same", rows_of(ids)), ("long", None), ("same", rows_of(ids))]
    r = scripted_run(main, script, GLM_UNSET)
    p = verdicts(r, pass_short="PASS", pass_long="PASS", second_request_equals_first="PASS") + rc_problem(r, 0)
    if f"the same {len(ids)} ids" not in detail(r):
        p.append(f"the detail does not say the ids are the same: {detail(r)!r}")
    return p


@case
def a_moving_seat_that_splits_above_the_band_fails(main):
    r = scripted_run(main, tie_script(2.0), GLM_UNSET)
    p = verdicts(r, pass_short="PASS", pass_long="PASS", second_request_equals_first="FAIL") + rc_problem(r, 1)
    d = detail(r)
    for want in (f"token {SPLIT_AT} of {len(IDS)}", "'t13' (id 13)", "'u13' (id 100013)", "gaps 2.0000 + 0.0200 = 2.0200",
                 f"band {1.485}"):
        if want not in d:
            p.append(f"the detail lacks {want!r}: {d!r}")
    # The band's edge, through the check, in g1 + g2: 1.46 + 0.02 splits, 1.47 + 0.02 does not.
    for gap, status in ((1.46, "PASS"), (1.47, "FAIL")):
        e = scripted_run(main, tie_script(gap), GLM_UNSET)
        p += [f"gap {gap}: {x}" for x in verdicts(e, second_request_equals_first=status)]
    return p


@case
def a_moving_seat_whose_first_gap_is_small_and_replay_gap_large_fails(main):
    # A flip moves the pair's gap by g1 + g2: a replay that is sure of the other token (2.0) is not a near tie.
    r = scripted_run(main, tie_script(0.05, back=2.0), GLM_UNSET)
    p = verdicts(r, pass_short="PASS", pass_long="PASS", second_request_equals_first="FAIL") + rc_problem(r, 1)
    d = detail(r)
    if "gaps 0.0500 + 2.0000 = 2.0500" not in d or "above the band" not in d:
        p.append(f"the detail does not hold the sum of both gaps to the band: {d!r}")
    p += verdicts(scripted_run(main, tie_script(0.05, back=0.4), GLM_UNSET), second_request_equals_first="PASS")
    return p


@case
def a_moving_seat_whose_first_reply_ended_at_the_split_reads_the_replay_gap(main):
    short = IDS[:3]
    script = [("short", rows_of(short)), ("long", None), ("longer", rows_of(IDS, {3: 0.04}))]
    r = scripted_run(main, script, GLM_UNSET)
    p = verdicts(r, second_request_equals_first="PASS") + rc_problem(r, 0)
    d = detail(r)
    if "<end of reply>" not in d or "gaps none + 0.0400 = 0.0400" not in d:
        p.append(f"the detail does not read the replay's row at the end of the first reply: {d!r}")
    script[2] = ("longer", rows_of(IDS, {3: 2.0}))
    p += verdicts(scripted_run(main, script, GLM_UNSET), second_request_equals_first="FAIL")
    return p


@case
def a_seat_that_moves_no_expert_keeps_the_exact_comparison(main):
    r = scripted_run(main, tie_script(0.05), LEVER_OFF)
    p = verdicts(r, pass_short="PASS", pass_long="PASS", second_request_equals_first="FAIL") + rc_problem(r, 1)
    d = detail(r)
    if "the replay's text differs" not in d or "residency off" not in d:
        p.append(f"the detail is not the exact comparison naming the word: {d!r}")
    if any(b.get("logprobs") or b.get("top_logprobs") for b in r.bodies):
        p.append(f"a seat that moves no expert was asked for logprobs: {r.bodies}")
    same = [("same", rows_of(IDS)), ("long", None), ("same", rows_of(IDS))]
    p += verdicts(scripted_run(main, same, LEVER_OFF), second_request_equals_first="PASS")
    return p


@case
def a_server_with_no_residency_record_is_held_exact_and_says_so(main):
    p = []
    for text, said in ((NO_RESIDENCY, "no residency record in the --stderr"), (None, "no --stderr, so no residency record was read")):
        r = scripted_run(main, tie_script(0.05), text)
        p += [f"{said}: {x}" for x in verdicts(r, pass_short="PASS", second_request_equals_first="FAIL") + rc_problem(r, 1)]
        if said not in detail(r):
            p.append(f"the FAIL does not name the missing fact ({said!r}): {detail(r)!r}")
        if any(b.get("logprobs") for b in r.bodies):
            p.append(f"{said}: logprobs were asked of a server with no residency fact")
    return p


@case
def the_residency_word_comes_from_the_servers_records(main):
    p = []
    for name, text, status in (("lever mid", LEVER_MID, "PASS"), ("unset and host", GLM_UNSET, "PASS"),
                               ("host only", HOST_ONLY, "PASS"), ("lever off", LEVER_OFF, "FAIL"),
                               ("lever off before a host line", LEVER_OFF + HOST_ONLY, "FAIL")):
        r = scripted_run(main, tie_script(0.05), text)
        p += [f"{name}: {x}" for x in verdicts(r, second_request_equals_first=status)]
    return p


@case
def a_moving_seat_with_no_logprobs_in_a_reply_fails_by_name(main):
    p = []
    for name, script in (("first", [("a", None), ("long", None), ("a", rows_of(IDS))]),
                         ("replay", [("a", rows_of(IDS)), ("long", None), ("a", None)])):
        r = scripted_run(main, script, GLM_UNSET)
        p += [f"{name}: {x}" for x in verdicts(r, pass_short="PASS", second_request_equals_first="FAIL") + rc_problem(r, 1)]
        if "logprobs" not in detail(r):
            p.append(f"{name}: the FAIL does not name logprobs: {detail(r)!r}")
    return p


@case
def an_unknown_residency_word_is_a_harness_fail(main):
    r = scripted_run(main, tie_script(0.05), LEVER_BAD)
    p = verdicts(r, pass_short="PASS", second_request_equals_first="FAIL") + rc_problem(r, 1)
    if not detail(r).startswith("harness:") or "banana" not in detail(r):
        p.append(f"the FAIL is not a harness line naming the word: {detail(r)!r}")
    return p


@case
def the_toktape_client_on_a_moving_seat_judges_a_direct_pair(main):
    # The tapes (toktape's own requests, no logprobs) are chats 0, 2 and 4; the direct pair is chats 1 and 3.
    p = []
    for name, tied, want in (("near tie", tie_script(0.05), "PASS"), ("far split", tie_script(2.0), "FAIL")):
        script = [("tape one", None), tied[0], (tied[1][0], None), tied[2], ("tape two", None)]
        r = scripted_run(main, script, GLM_UNSET, toktape="")
        p += [f"{name}: {x}" for x in verdicts(r, pass_short="PASS", pass_long="PASS", second_request_equals_first=want)
              + rc_problem(r, 0 if want == "PASS" else 1)]
        if [asked(b) for b in r.bodies] != [False, True, False, True, False]:
            p.append(f"{name}: the direct requests are not chats 1 and 3: {[asked(b) for b in r.bodies]}")
        if want == "PASS" and ("tapes differ" not in detail(r) or "direct pair" not in detail(r)):
            p.append(f"{name}: the detail does not say the tapes differ and the direct pair is judged: {detail(r)!r}")
        kept = sorted(f.name for f in (r.tmp / "out").glob("*.json"))
        if "pass_short_probe.json" not in kept or "second_request_probe.json" not in kept:
            p.append(f"{name}: the direct pair is not kept in --dir: {kept}")
        if len(list((r.tmp / "out" / "tapes").glob("*.toktape"))) != 3:
            p.append(f"{name}: the three tapes are not in --dir/tapes")
    return p


@case
def the_band_is_its_derivation(main):
    g = main.__globals__
    band, sigma = g["SPLIT_GAP_BAND"], g["SIGMA_LOGIT"]
    p = []
    if abs(band - 3 * math.sqrt(2) * sigma) > 5e-4:
        p.append(f"SPLIT_GAP_BAND {band} is not 3 * sqrt(2) * SIGMA_LOGIT {sigma} = {3 * math.sqrt(2) * sigma:.4f}")
    src = (HERE.parent.parent / "crates" / "gpu-gates" / "src" / "prompts.rs").read_text()
    m = re.search(r"pub const SIGMA_LOGIT_V41: f64 = ([0-9.]+);", src)
    if not m or float(m.group(1)) != sigma:
        p.append(f"SIGMA_LOGIT {sigma} is not prompts.rs's SIGMA_LOGIT_V41 ({m.group(1) if m else 'not found'})")
    return p


OFF_LOG = """bloomery-serve: a first start on a new card can compile the GPU code for it (tens of seconds); later starts take seconds
place unset place=bp why=set tier_experts=0 break_even=0 basis=none
bloomery-serve-ds41: place=bp ctx=65536 listening on http://127.0.0.1:18080
bloomery-serve: tier card 3090 (tier 0) holds no expert
  caused by: the cards before it hold every routed expert
"""
LISTENED_THEN_PANICKED = """bloomery-serve: a first start on a new card can compile the GPU code for it (tens of seconds); later starts take seconds
bloomery-serve-ds41: place=bp ctx=65536 listening on http://127.0.0.1:18080
thread 'main' panicked at crates/serve/src/engine.rs:1:1
"""


@case
def off_reason_command_reads_the_refusal(main):
    tmp = scratch("pass-checks-or.")
    (tmp / "refused").write_text(OFF_LOG)
    (tmp / "panicked").write_text(LISTENED_THEN_PANICKED)
    p = []
    for name, want_rc, want_out in (
            ("refused", 0, ["tier card 3090 (tier 0) holds no expert / caused by: the cards before it hold every routed expert"]),
            ("panicked", 1, [])):
        rc, lines, raised = call(main, ["off-reason", "--stderr", str(tmp / name)], {})
        if raised or rc != want_rc or lines != want_out:
            p.append(f"{name}: exit {rc}, output {lines}, raised {raised}; want {want_rc}, {want_out}")
    return p


@case
def a_server_down_with_no_stderr_is_a_harness_fail(main):
    rc, lines, raised = call(main, [f"http://127.0.0.1:{closed_port()}", "--timeout", "5"], {})
    by, _, rc_line = parse(lines)
    detail = by.get("off_reason", ("missing", ""))[1]
    p = [f"raised {raised}"] if raised else []
    if by.get("off_reason", ("",))[0] != "FAIL" or not detail.startswith("harness:") or rc != 1 or rc_line != 1:
        p.append(f"off_reason {by.get('off_reason')}, exit {rc}, SESSION rc {rc_line}")
    return p


@case
def usage_errors_exit_64_and_a_missing_file_2(main):
    p = []
    for argv in ([], ["http://127.0.0.1:1", "--client", "toktape"], ["http://127.0.0.1:1", "--bogus"]):
        rc, _, raised = call(main, argv, {})
        if rc != 64:
            p.append(f"{argv}: exit {rc} {raised or ''}, want 64")
    rc, _, _ = call(main, ["http://127.0.0.1:1", "--stderr", "/nonexistent/stderr"], {})
    if rc != 2:
        p.append(f"a --stderr that is no file: exit {rc}, want 2")
    cli = subprocess.run([sys.executable, "-I", str(Path(__file__).resolve())], capture_output=True, text=True,
                         env={"PATH": os.environ.get("PATH", "")})
    if cli.returncode != 64 or "no URL" not in cli.stderr:
        p.append(f"the script with no URL: exit {cli.returncode}, stderr {cli.stderr!r}")
    return p


def run_cases(main, only=None):
    problems = {}
    for name, fn in CASES.items():
        if only and name not in only:
            continue
        got = fn(main)
        if got:
            problems[name] = got
    return problems


# Each mutant puts back, in the code above this line, what one of the pass scripts this replaces did; the case named
# beside it must fail on it. An anchor that no longer matches fails the self-test by name.
MUTANTS = (
    ("takes any HTTP status", (("if status != 200:", "if False:", 1),), "second-request-500"),
    ("reads content only", (("text = content if content.strip() else reasoning", "text = content", 1),),
     "reasoning-only-reply"),
    ("sends a long prompt of any size", (("if tokens < LONG_MIN_TOKENS:", "if False:", 1),),
     "long-prompt-under-4096-tokens"),
    ("stops at the first FAIL", (("if self.dead:", "if self.dead or self.fails:", 1),),
     "a-fail-does-not-stop-the-next-check"),
    ("never finds the server dead", (("self.dead = self.probe_down()", "self.dead = None", 2),),
     "a-dead-server-skips-the-rest-naming-why"),
    ("reads the log's last five lines as the reason",
     (("def refusal_of(text):", 'def refusal_of(text):\n    return head(" ".join(text.splitlines()[-5:]), 400)\n\n\ndef refusal_of_whole(text):', 1),),
     "an-off-cell-reads-the-refusal-not-the-banner"),
    ("takes a record for the refusal", (("not BANNER.match(line) and not is_rec(line)", "not BANNER.match(line)", 1),), "off-reason-command-reads-the-refusal"),
    ("compares no text", (("if text != want:", "if False:", 1),), "a-replay-that-differs-fails"),
    ("ends 0 on a FAIL", (("rc = 1 if self.fails else 0", "rc = 0", 1),), "second-request-500"),
    ("sends the long prompt itself as well as recording it",
     (('text, field, evaluated, name = self.record("long", "", prompt)',
       'self.post_chat("pass_long", chat_body(prompt, self.a.max_tokens))\n            '
       'text, field, evaluated, name = self.record("long", "", prompt)', 1),),
     "toktape-client-checks-are-the-tapes"),
    ("takes a tape of another prompt", (("if sent != prompt:", "if False:", 1),), "toktape-tape-of-another-shape-fails"),
    ("trusts the completion over the tokens", (("if completion != streamed:", "if False:", 1),),
     "toktape-tape-of-another-shape-fails"),
    ("ignores the error of a tape's request", (('if rec.get("error"):', "if False:", 1),),
     "toktape-tape-of-another-shape-fails"),
    ("takes a tape of any number of requests", (("if len(reqs) != 1:", "if False:", 1),),
     "toktape-tape-of-another-shape-fails"),
    ("reads a tape of any schema", (('or obj.get("schema") != TAPE_SCHEMA:', ":", 1),),
     "toktape-tape-of-another-shape-fails"),
    ("passes a toktape that ended other than 0", (("if r.returncode != 0:", "if False:", 1),),
     "toktape-client-failures-name-their-check"),
    ("judges the replay of a server that moves experts by exact text",
     (("        if self.residency.moves:\n            return self.replay_of_a_moving_server()",
       "        if False:\n            return self.replay_of_a_moving_server()", 1),),
     "a-moving-seat-that-splits-at-a-near-tie-passes"),
    ("passes a split at any gap", (("if total > SPLIT_GAP_BAND:", "if False:", 1),),
     "a-moving-seat-that-splits-above-the-band-fails"),
    ("holds the first reply's gap alone to the band", (("total = (g1 or 0.0) + (g2 or 0.0)", "total = g1 or 0.0", 1),),
     "a-moving-seat-whose-first-gap-is-small-and-replay-gap-large-fails"),
    ("reads a split as the same ids", (("if a[i][0] != b[i][0]:", "if False:", 1),),
     "a-moving-seat-that-splits-above-the-band-fails"),
    ("takes every server for one that moves experts",
     (('return Residency(word, word != "off", f"residency {word}", None)',
       'return Residency(word, True, f"residency {word}", None)', 1),),
     "a-seat-that-moves-no-expert-keeps-the-exact-comparison"),
    ("asks every server for logprobs", (("top = LOGPROBS_TOP if self.residency.moves else 0", "top = LOGPROBS_TOP", 1),),
     "a-seat-that-moves-no-expert-keeps-the-exact-comparison"),
    ("takes a server with no residency record for one that moves experts",
     (('return Residency(None, False, "no residency record in the --stderr", None)',
       'return Residency(None, True, "no residency record in the --stderr", None)', 1),),
     "a-server-with-no-residency-record-is-held-exact-and-says-so"),
    ("reads only the lever record for the residency word",
     (('RESIDENCY_KINDS = ("residency_lever", "residency_unset", "residency_host")',
       'RESIDENCY_KINDS = ("residency_lever",)', 1),),
     "the-residency-word-comes-from-the-servers-records"),
    ("takes any residency word", (("if not RESIDENCY_WORD.match(word):", "if False:", 1),),
     "an-unknown-residency-word-is-a-harness-fail"),
    ("judges no direct pair under toktape",
     (("rows = why = None\n            if top:", "rows = why = None\n            if False:", 1),),
     "the-toktape-client-on-a-moving-seat-judges-a-direct-pair"),
    ("re-pins the band without its derivation", (("SPLIT_GAP_BAND = 1.485", "SPLIT_GAP_BAND = 3.0", 1),),
     "the-band-is-its-derivation"),
)


def mutant_main(pairs):
    text = Path(__file__).read_text()
    cut = text.index("# Each mutant puts back")
    src, rest = text[:cut], text[cut:]
    for old, new, count in pairs:
        if src.count(old) != count:
            raise AssertionError(f"mutant anchor {old!r} is in the code {src.count(old)} times, want {count}")
        src = src.replace(old, new)
    src += rest
    ns = {"__file__": str(Path(__file__).resolve()), "__name__": "pass_checks_mutant"}
    exec(compile(src, "<mutant>", "exec"), ns)
    return ns["main"]


def self_test():
    try:
        return self_test_run()
    finally:
        for d in SCRATCH:
            shutil.rmtree(d, ignore_errors=True)


def self_test_run():
    problems = run_cases(main)
    for name, p in problems.items():
        for x in p:
            print(f"FAIL {name}: {x}")
    if problems:
        print(f"pass_checks self-test: {len(problems)} of {len(CASES)} cases FAILED")
        return 1
    bad = 0
    for what, pairs, name in MUTANTS:
        red = run_cases(mutant_main(pairs), only={name}).get(name)
        print(f"mutant {what}: {name} {'red: ' + head(red[0], 110) if red else 'GREEN'}")
        bad += not red
    if bad:
        print(f"pass_checks self-test: {bad} of {len(MUTANTS)} mutants left their case green")
        return 1
    print(f"pass_checks self-test: {len(CASES)} cases ok, {len(MUTANTS)} mutants red")
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        sys.exit(self_test())
    sys.exit(main(sys.argv[1:]))
