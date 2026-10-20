#!/usr/bin/env python3
"""Schemajump (any model answering a schema) against Clef-Flash: one SystemOne request to each engine, and the
tables of a sitting.

    tools/ref/clef/sjcmp.py convert [--suite en|ko] [--id ID]
    tools/ref/clef/sjcmp.py run [--clef NAME=URL] [--qwen NAME=URL ...]
                                [--suite FILE ...] --rounds N --rounds-long N --out DIR [--model NAME]
    tools/ref/clef/sjcmp.py check-suites [--suite FILE ...]
    tools/ref/clef/sjcmp.py table [--data DIR] ROWS [ROWS ...]
    tools/ref/clef/sjcmp.py --self-test

What it is. The user asked for schemajump — any model answering a schema the way Clef-Flash does — measured
against Clef-Flash on the time to a complete answer and on agreement (the design, `/Users/hckim/orca/briefs/
w2300/par/sjfast-files/design.md` §4). The arms are servers this program posts to: the Clef decide seat
(`POST /v1/systemone`), a generative seat with the schema as a `response_format` (`POST /v1/chat/
completions`, kind `qwen`: any schema seat, the Qwen3.6 one and V4.1 alike, each with its jump arm and its step
arm).

The request conversion (§4.2). A SystemOne question becomes a property of a JSON schema: a choice a string
enum in Clef's sorted order, a score an integer enum `0..n-1`, a noul a boolean. The prompt is Clef's own
schema text (`crates/decision/src/encode.rs`'s `SCHEMA FIELDS` block) in the system message and the state in
the user message, with one instruction line, thinking off, `cache_prompt` off and greedy. The answer comes back
as a JSON object and is mapped back to Clef's top per question (`e2e.py`'s `top`: a noul's yes/no, a choice's
key, a score's level index).

The call-count check (§4.2). A constrained request's engine calls are deterministic: the program the server
compiles from the schema fixes every id but the picks. The program is rebuilt here from the same converted
schema, tokenized on the server's own `/tokenize`, and walked for the answer the model gave; the calls it
predicts must be the ones the server reports in `timings.constrained` (`next_calls`, `prefill_calls`), or the
run stops by name. The server's counts are of the calls after the prompt (`crates/serve/src/genloop.rs`: the
prompt's own calls and the step of its last id are `prompt_ms`'s), so the prediction excludes the prompt's
closing step; a server that counts it is accepted and the run says which convention it saw. A server that
reports `next_after_calls` is one with the jump levers (the answer's opening ids ride the prompt's call, each
run after a decision is one call that also steps its last id, the ids after the last decision are taken with
no call and the stop is the vocabulary's end of sequence): its plan is `plan_levers`, and it must reproduce
`next_calls`, `prefill_calls`, `next_after_calls`, `lead`, `spans` and `tail`. A server that reports the old
keys only gets the old plan.

The gold (`<suite>.gold.jsonl` beside a suite file). One line a request, `{"id", "gold": {question id: option
id}, "why", "source": {"url", "quote"}}` (a noul's label is `true`/`false` or `yes`/`no`; `why` and `source` are
the labeller's notes and take no part in the score). Every arm's picks are scored against it (`correct/N` per
suite); a gold line that names a request the suite lacks, a question the request lacks or a word that is not an
option of the question, a key other than those four, and a request with no gold line, is a named error
(`check-suites` reads them all and the shell runs it before the lease).

The bodies. Beside `rows.json` in the out dir, `bodies/<arm>/<suite>-<id>.json` holds each arm's answer to each
request's first counted round (not the dropped one) exactly as the server sent it: the Clef arm's `/v1/systemone`
body, a generative arm's chat completion; the row names its file under `body`. The files are written after the last
round, from text kept in memory, so nothing but that bookkeeping sits between two timed POSTs. Then each generative
arm whose `GET /props` lists `/v1/systemone` under `routes` is asked every request once more on that route, untimed
(the Clef arm's body is already one), and the bodies go to `bodies/<arm>-systemone/<suite>-<id>.json`; a non-200
answer is saved too, its status in `<that file>.status`, and `[bodies] <arm>: systemone n ok / m` counts the 200s. A
server without the route (or with no readable `/props`) is a named line and no error; a server that stops answering
ends its pass by name. The pass is made after `rows.json` is written, so a failure in it never costs the
measurement. The directories are made before the first POST: an out dir that cannot hold them ends the run before
the rounds.

The tables (`table`). The client wall — the POST to the last byte, the user's time to answer — p50 and p95 per
suite and per bucket (short: a prompt under 1,000 ids; long: the two long requests), beside the server's own
split and the host-prep term (`wall - (prompt_ms + predicted_ms)`, `head_ms` for the Clef arm); the tops
against the official BF16 rows and against the Clef arm served; and the jump arm against the step arm
(`SAME n/n`).

The suites are `tools/ref/clef/suite.jsonl` (English, suite `en`) and `suite-ko.jsonl` (Korean, suite `ko`), 15
requests, 31 questions, unless `--suite FILE` names others (a suite is named by its file: `suite-hard.jsonl` is
`hard`); `suite-upstream.jsonl` and `edge.jsonl` are wire checks (the conversion of every row of them is in the
self-test). The official rows are the box's `<data>/clef/flash/ref/reference.jsonl` and `ref-ko/reference.jsonl`,
for `en` and `ko` alone.

`--self-test` is pure: no server, no card, no clock. It covers the conversion of every question type of the
four suites, the back-mapping, the rotation, the call plan and the count-mismatch refusal, the stats on a fixed
sample, and the bodies (first counted round, the SystemOne pass's route test, order and failures) against servers
that are objects logging their requests.
"""

from __future__ import annotations

import http.client
import json
import math
import os
import re
import statistics
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

import e2e  # the suite table, `top` (a question's top), the official rows

SUITES = e2e.SUITES
WIRE_SUITES = ("tools/ref/clef/suite-upstream.jsonl", "tools/ref/clef/edge.jsonl")

# Clef's own words (crates/decision/src/encode.rs, crates/decision/src/request.rs).
CLEF_SYSTEM_PROMPT = (
    "Read the complete state and schema. Decide every field jointly. Each answer must be exactly one of that "
    "field's allowed options."
)
NOUL_TRUE = "The proposition is true or the answer is yes."
NOUL_FALSE = "The proposition is false or the answer is no."

# The instruction line the Qwen prompt carries beside Clef's schema text (the design's §2.1: the Clef schema
# text re-used, the state alone in the user message, plus one instruction line).
QSYS = (
    "Read the state and the schema fields. Decide every field. Reply with one JSON object: one key per field ID, "
    "each value exactly one of that field's allowed options."
)

# A noul's gold may be spelled in Clef's top space (`yes`, `no`, `e2e.py`'s `top`) as well as by its option ids.
NOUL_YES_NO = {"yes": "true", "no": "false"}
# A gold line's keys: the labels, and the labeller's notes (the reason, and where the answer was taken from).
GOLD_KEYS = frozenset(("id", "gold", "why", "source"))

# A prompt under this many ids is the short bucket (the design's §4.2: the 13 requests under 1,000 ids).
SHORT_LIMIT = 1000
# The jump arm's fewest ids a run is fed as a call (crates/serve/src/genloop.rs's MIN_FILL): a shorter run is
# stepped, a call of one id costs what the step it replaces costs.
MIN_FILL = 2
# The request's bound (seconds).
REQUEST_BOUND = 300


class Stop(SystemExit):
    """A named refusal: the run ends with the reason on stderr."""


def fail(why: str) -> None:
    """Stop the run by name."""
    raise Stop(f"sjcmp: {why}")


# ---------------------------------------------------------------- the request conversion


def render(v) -> str:
    """Clef's text for one request value (`crates/decision/src/render.rs`): a string passes through, anything else
    is Python's `json.dumps` with the compact separators and the keys sorted (the encoder's own writer)."""
    if isinstance(v, str):
        return v
    return json.dumps(v, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def json_inner(s: str) -> str:
    """A JSON string's body as the server writes it (`crates/serve/src/constrain/schema.rs`'s `json_inner`): the
    string's escapes, a `<` as `\\u003c`."""
    return json.dumps(s, ensure_ascii=False)[1:-1].replace("<", "\\u003c")


def options(q: dict) -> list[tuple[str, object]]:
    """Clef's options of one question (`crates/decision/src/request.rs`'s `options_with`, `Rules::CLEF`): a noul's
    `true` then `false` (the release's sentences, the criteria's when they give one), a choice's criteria sorted
    by key (code point order), a score's levels by index."""
    kind, crit = q["type"], q.get("criteria")
    if kind == "noul":
        over = crit if isinstance(crit, dict) else {}
        out = []
        for key, default in (("true", NOUL_TRUE), ("false", NOUL_FALSE)):
            out.append((key, over[key] if key in over else default))
        return out
    if kind == "choice":
        return sorted(crit.items(), key=lambda kv: kv[0])
    return [(str(i), v) for i, v in enumerate(crit)]


def check_request(req: dict, where: str) -> None:
    """A suite row's shape: the questions' types, the criteria they need and the ids the program writes as
    property names. A row that does not hold is a named error, never a default."""
    if not isinstance(req, dict):
        fail(f"{where}: the row is not an object")
    if "state" not in req:
        fail(f"{where}: the row has no 'state'")
    questions = req.get("questions")
    if not isinstance(questions, dict) or not questions:
        fail(f"{where}: the row has no questions")
    for qid, q in questions.items():
        if not isinstance(qid, str) or not qid:
            fail(f"{where}: a question's id is empty")
        if not isinstance(q, dict):
            fail(f"{where}: the question '{qid}' is not an object")
        kind = q.get("type")
        if kind not in ("noul", "choice", "score"):
            fail(f"{where}: the question '{qid}' has the type {kind!r}, not noul, choice or score")
        crit = q.get("criteria")
        if kind == "choice" and not (isinstance(crit, dict) and crit):
            fail(f"{where}: the choice '{qid}' has no criteria")
        if kind == "score" and not (isinstance(crit, list) and crit):
            fail(f"{where}: the score '{qid}' has no criteria")
        if kind == "noul" and crit is not None and not isinstance(crit, dict):
            fail(f"{where}: the noul '{qid}' has criteria that are not an object")


def schema_text(req: dict) -> str:
    """Clef's `SCHEMA FIELDS` block, as `crates/decision/src/encode.rs` writes it: the fields in the request's
    order, each with its id, its type, its instruction and its options."""
    out = "\n\nSCHEMA FIELDS:\n"
    for qi, (qid, q) in enumerate(req["questions"].items()):
        ins = q.get("instructions")
        ins = qid if ins is None or ins == "" else render(ins)
        out += f"\nFIELD {qi + 1}\nID: {qid}\nTYPE: {q['type']}\nINSTRUCTION: {ins}\nALLOWED OPTIONS:\n"
        for oi, (oid, desc) in enumerate(options(q)):
            sem = {"option_id": oid}
            if desc is not None:
                sem["description"] = desc
            out += f"OPTION {oi + 1}: {render(sem)}\n"
        out += "END FIELD\n"
    return out


def json_schema(req: dict) -> dict:
    """The schema the answer follows: a question id a property, a choice a string enum in Clef's sorted order, a
    score an integer enum `0..n-1`, a noul a boolean."""
    props = {}
    for qid, q in req["questions"].items():
        if q["type"] == "choice":
            props[qid] = {"type": "string", "enum": [k for k, _ in options(q)]}
        elif q["type"] == "score":
            props[qid] = {"type": "integer", "enum": list(range(len(options(q))))}
        else:
            props[qid] = {"type": "boolean"}
    return {"type": "object", "properties": props, "required": list(props)}


def chat_body(req: dict, model: str | None = None) -> dict:
    """The converted request: a generative seat's `POST /v1/chat/completions` body."""
    body = {
        "messages": [
            {"role": "system", "content": QSYS + schema_text(req)},
            {"role": "user", "content": "STATE:\n" + render(req["state"])},
        ],
        "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": json_schema(req)}},
        "chat_template_kwargs": {"enable_thinking": False},
        "cache_prompt": False,
    }
    if model:
        body["model"] = model
    return body


def option_ids(q: dict) -> list[str]:
    """A question's option ids, in Clef's order."""
    return [oid for oid, _ in options(q)]


def pick_of_top(q: dict, top: str) -> str:
    """A top in Clef's top space (a noul's yes/no, a choice's key, a score's level index) as the option id the
    gold labels name (a noul's `true`/`false`)."""
    if q["type"] == "noul":
        return "true" if top == "yes" else "false"
    return top


def clef_body(req: dict) -> dict:
    """The Clef arm's body: the SystemOne request as it stands (its `id` is the runner's, not the server's)."""
    body = {k: v for k, v in req.items() if k != "id"}
    body.setdefault("model", "clef-flash")
    return body


def program_parts(req: dict) -> list[tuple[str, object]]:
    """The answer program as the server's compile sees it (`crates/serve/src/constrain/schema.rs`'s object and
    enumeration): the forced texts and the picks, in order."""
    parts: list[tuple[str, object]] = [("T", "{")]
    for i, (qid, q) in enumerate(req["questions"].items()):
        sep = "" if i == 0 else ", "
        parts.append(("T", f'{sep}"{json_inner(qid)}": '))
        if q["type"] == "choice":
            parts += [("T", '"'), ("P", [k for k, _ in options(q)]), ("T", '"')]
        elif q["type"] == "score":
            parts += [("P", [str(n) for n in range(len(options(q)))])]
        else:
            parts += [("P", ["true", "false"])]
    parts.append(("T", "}"))
    return parts


def merged(parts: list[tuple[str, object]]) -> list[tuple[str, object]]:
    """The server's `merged` (`crates/serve/src/constrain/program.rs`): the neighbouring forced texts are joined,
    so each is tokenized whole."""
    out: list[tuple[str, object]] = []
    for part in parts:
        if part[0] == "T" and out and out[-1][0] == "T":
            out[-1] = ("T", out[-1][1] + part[1])
        elif part[0] == "T" and part[1] == "":
            continue
        else:
            out.append(part)
    return out


# ---------------------------------------------------------------- the answer back-mapping


def picks_from_answer(req: dict, obj: dict) -> list[int]:
    """The answer's picks: the index of the chosen option of every question, in the request's order. A value that
    is not one of the options is a named error."""
    picks = []
    for qid, q in req["questions"].items():
        if qid not in obj:
            fail(f"the answer has no '{qid}'")
        v = obj[qid]
        if q["type"] == "noul":
            if not isinstance(v, bool):
                fail(f"the answer's '{qid}' is {v!r}, not a boolean")
            # The program's noul pick lists `true` first (`program_parts`, `options`).
            picks.append(0 if v else 1)
        elif q["type"] == "choice":
            keys = [k for k, _ in options(q)]
            if v not in keys:
                fail(f"the answer's '{qid}' is {v!r}, not one of {keys}")
            picks.append(keys.index(v))
        else:
            levels = len(options(q))
            if not isinstance(v, int) or isinstance(v, bool) or not 0 <= v < levels:
                fail(f"the answer's '{qid}' is {v!r}, not a level index in 0..{levels - 1}")
            picks.append(v)
    return picks


def tops_from_answer(req: dict, obj: dict) -> dict[str, str]:
    """The answer's top per question, in Clef's top space (`e2e.py`'s `top`): a noul's yes/no, a choice's key, a
    score's level index."""
    tops = {}
    for qid, q in req["questions"].items():
        v = obj[qid]
        if q["type"] == "noul":
            tops[qid] = "yes" if v else "no"
        elif q["type"] == "choice":
            tops[qid] = v
        else:
            tops[qid] = str(v)
    return tops


def tops_from_answers(resp: dict, req: dict, where: str) -> dict[str, str]:
    """The Clef arm's top per question, from its `answers`."""
    answers = resp.get("answers")
    if not isinstance(answers, dict):
        fail(f"{where}: the response has no answers")
    tops = {}
    for qid in req["questions"]:
        if qid not in answers:
            fail(f"{where}: the response has no answer to '{qid}'")
        tops[qid] = e2e.top(answers[qid])[0]
    return tops


# ---------------------------------------------------------------- the call plan


def trie_path(seqs: list[list[int]], k: int) -> list[tuple[int, bool]]:
    """The chosen option's path through the options' trie: an id and whether the node holds more than one option
    (the server reads no row at a forced id, nor at a trie's only child: `Walk::fixed_next`)."""
    path: list[tuple[int, bool]] = []
    alive = list(range(len(seqs)))
    for depth, tid in enumerate(seqs[k]):
        kids = {seqs[a][depth] for a in alive if len(seqs[a]) > depth}
        path.append((tid, len(kids) > 1))
        alive = [a for a in alive if len(seqs[a]) > depth and seqs[a][depth] == tid]
    return path


def stream(parts_ids: list[tuple[str, object]], picks: list[int]) -> list[tuple[int, bool]]:
    """The answer's tokens as (id, a decision row) a token: the forced texts' ids and the chosen option's path."""
    toks: list[tuple[int, bool]] = []
    pi = 0
    for kind, body in parts_ids:
        if kind == "F":
            toks += [(t, False) for t in body]
        else:
            toks += trie_path(body, picks[pi])
            pi += 1
    return toks


def plan_jump(toks: list[tuple[int, bool]]) -> list[tuple[str, int]]:
    """The jump arm's engine calls after the prompt (`crates/serve/src/genloop.rs`'s `plan_span` and `fill`): the
    prompt's closing step, then a run of fixed ids fed as one call (the id just taken and the run but its last
    id) and the step that closes it, or a step an id where the run is shorter than `MIN_FILL`."""
    calls: list[tuple[str, int]] = []
    n = len(toks)
    dec = [d for _, d in toks]
    calls.append(("next", 1))  # close(): the prompt's last id's step
    i = 0
    while i < n:
        j = i + 1
        while j < n and not dec[j]:
            j += 1
        run = j - (i + 1)
        if run >= MIN_FILL:
            calls.append(("prefill", run))
            calls.append(("next", 1))
            i = j
        else:
            calls.append(("next", 1))
            i += 1
    return calls


def plan_step(toks: list[tuple[int, bool]]) -> list[tuple[str, int]]:
    """The step arm's engine calls after the prompt: one step an id, the stop's row among them."""
    return [("next", 1)] * (len(toks) + 1)


# The counts a server with the jump levers reports in `timings.constrained` beside `next_calls` and
# `prefill_calls` (`crates/serve/src/genloop.rs`'s `Jump`).
LEVER_KEYS = ("next_calls", "prefill_calls", "next_after_calls", "lead", "spans", "tail")


def plan_levers(toks: list[tuple[int, bool]]) -> dict[str, int]:
    """The jump arm's engine calls after the levers, as the counts the server reports (`Gen::take_lead`,
    `plan_span`, `fill`, `tail_id`): the ids before the first decision are the lead and ride the prompt's call,
    which steps its last id (the prompt's own, so not counted); each run of fixed ids between two decisions is
    one `next_after` of the decision's id and the run (`spans`, `span_ids`), or, shorter than `MIN_FILL`, one step
    an id; the ids after the last decision are the tail, the stop with them, taken with no call. Nothing decided
    is all tail but its first id, which the prompt's step picks."""
    out = dict.fromkeys((*LEVER_KEYS, "span_ids"), 0)
    decisions = [i for i, (_, dec) in enumerate(toks) if dec]
    if not decisions:
        out["tail"] = len(toks)
        return out
    out["lead"] = decisions[0]
    for a, b in zip(decisions, decisions[1:]):
        run = b - a - 1
        if run >= MIN_FILL:
            out["next_after_calls"] += 1
            out["spans"] += 1
            out["span_ids"] += run + 1
        else:
            out["next_calls"] += run + 1
    out["tail"] = len(toks) - decisions[-1]
    return out


def step_levers(toks: list[tuple[int, bool]]) -> dict[str, int]:
    """The step arm's counts under the same keys: one step an id, nothing else."""
    out = dict.fromkeys((*LEVER_KEYS, "span_ids"), 0)
    out["next_calls"] = len(toks)
    return out


def plan_counts(calls: list[tuple[str, int]]) -> tuple[int, int]:
    """A plan's (steps after the prompt, calls): the prompt's closing step is the prompt's own, so it is out."""
    nexts = sum(1 for c in calls if c[0] == "next")
    prefills = sum(1 for c in calls if c[0] == "prefill")
    return nexts - 1, prefills


# ---------------------------------------------------------------- the stats


def percentile(xs: list[float], p: float) -> float:
    """The p-th percentile of `xs` (the linear interpolation between the closest ranks, numpy's default)."""
    ys = sorted(xs)
    if not ys:
        fail("a percentile of no samples")
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return ys[int(k)]
    return ys[lo] + (ys[hi] - ys[lo]) * (k - lo)


def arm_rows(rows: list[dict]) -> dict[str, list[dict]]:
    """The rows of every arm, in the order the arms first appear."""
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["arm"], []).append(r)
    return out


def suite_order(rows: list[dict]) -> list[str]:
    """The suites the rows hold, in the order they first appear."""
    out: list[str] = []
    for r in rows:
        if r["suite"] not in out:
            out.append(r["suite"])
    return out


def gold_counts(arm: str, rs: list[dict], suite: str, gold: dict[str, dict[str, str]]) -> tuple[int, int, list[str]]:
    """An arm's picks in a suite against its gold: (right, questions asked, the wrong ones as lines). One answer a
    request, the first counted round's. A request of the gold that the arm has no row for, and a row with no
    picks, are named errors: a score over fewer questions would pass for a lower one."""
    first: dict[str, dict] = {}
    for r in rs:
        if r["arm"] == arm and r["suite"] == suite and not r["dropped"]:
            first.setdefault(r["id"], r)
    right = n = 0
    wrong: list[str] = []
    for rid, labels in gold.items():
        r = first.get(rid)
        if r is None:
            fail(f"table: the arm {arm} has no counted row of {suite}/{rid}, which the gold names")
        if "picks" not in r:
            fail(f"table: the arm {arm}'s row of {suite}/{rid} has no picks, so the gold cannot score it")
        for qid, want in labels.items():
            n += 1
            got = r["picks"].get(qid)
            if got == want:
                right += 1
                continue
            wrong.append(f"{suite}/{rid}/{qid} (gold {want}, ours {got})")
    return right, n, wrong


def stats_table(rows: list[dict], suites: dict | None = None) -> list[str]:
    """The wall table: per arm, per suite and per bucket, the client wall's p50 and p95, the server's own split
    and the host-prep term beside them, and, for a suite with gold labels (`suites`, the rows file's), the
    picks right of the questions asked."""
    lines = ["=== per arm: the client wall (POST to the last byte) and the server's own split ==="]
    suites = suites or {}
    for arm, rs in arm_rows(rows).items():
        kind = rs[0]["kind"]
        groups: list[tuple[str, list[dict]]] = []
        for suite in suite_order(rs):
            g = [r for r in rs if r["suite"] == suite and not r["dropped"]]
            if g:
                groups.append((f"suite {suite}", g))
        for bucket in ("short", "long"):
            g = [r for r in rs if r["bucket"] == bucket and not r["dropped"]]
            if g:
                groups.append((f"bucket {bucket}", g))
        for label, g in groups:
            walls = [r["wall_ms"] for r in g]
            server = [r["server_ms"] for r in g if r["server_ms"] is not None]
            prep = [r["host_ms"] for r in g if r["host_ms"] is not None]
            parts = [
                f"n {len(g)}",
                f"p50 {percentile(walls, 50):.1f} ms",
                f"p95 {percentile(walls, 95):.1f} ms",
            ]
            if server:
                parts.append(f"server {statistics.median(server):.1f} ms")
            if prep:
                parts.append(f"host prep {statistics.median(prep):.1f} ms")
            gold = (suites.get(label[len("suite "):]) or {}).get("gold") if label.startswith("suite ") else None
            if gold:
                right, asked, _ = gold_counts(arm, rs, label[len("suite "):], gold)
                parts.append(f"correct {right}/{asked}")
            lines.append(f"arm {arm} ({kind}) | {label} | " + " | ".join(parts))
    return lines


def gold_table(rows: list[dict], suites: dict) -> list[str]:
    """Every arm's picks against the gold labels, a line an arm and suite with gold: the right of the asked and
    the wrong ones. Empty when no suite has gold."""
    lines: list[str] = []
    for arm, rs in arm_rows(rows).items():
        for suite in suite_order(rs):
            gold = (suites.get(suite) or {}).get("gold")
            if not gold:
                continue
            right, asked, wrong = gold_counts(arm, rs, suite, gold)
            lines.append(
                f"arm {arm} ({rs[0]['kind']}) vs the gold of suite {suite} | {right}/{asked} | wrong: "
                + ("; ".join(wrong) if wrong else "—")
            )
    return ["=== gold: the picks against the labels ==="] + lines if lines else []


def agreement_tables(rows: list[dict], refs: dict) -> list[str]:
    """The agreement tables: every generative arm's tops against the official BF16 rows and against the Clef arm
    served, and the jump arm against the step arm."""
    lines = ["=== agreement ==="]
    arms = arm_rows(rows)
    clef = next((a for a in arms if arms[a][0]["kind"] == "clef"), None)
    served = {}
    if clef:
        for r in arms[clef]:
            if not r["dropped"]:
                served[(r["suite"], r["id"])] = r["tops"]
    generative = [a for a in arms if arms[a][0]["kind"] != "clef"]
    for arm in generative:
        rs = [r for r in arms[arm] if not r["dropped"]]
        # one answer a request: the first counted round holds it.
        answers: dict[tuple[str, str], dict[str, str]] = {}
        for r in rs:
            answers.setdefault((r["suite"], r["id"]), r["tops"])
        if refs:
            n = agree = 0
            en = [0, 0]
            ko = [0, 0]
            misses = []
            for (suite, rid), tops in answers.items():
                ref = refs.get((suite, rid))
                if not ref:
                    continue
                for qid, top in tops.items():
                    want = e2e.top(ref["answers"][qid])[0]
                    n += 1
                    ok = top == want
                    agree += ok
                    if suite in ("en", "ko"):
                        (en if suite == "en" else ko)[0] += ok
                        (en if suite == "en" else ko)[1] += 1
                    if not ok:
                        misses.append(f"{suite}/{rid}/{qid} (official {want}, ours {top})")
            lines.append(
                f"arm {arm} vs the official BF16 tops | {agree}/{n} | English {en[0]}/{en[1]} | "
                f"Korean {ko[0]}/{ko[1]} | misses: " + ("; ".join(misses) if misses else "—")
            )
        if served:
            n = agree = 0
            misses = []
            for (suite, rid), tops in answers.items():
                other = served.get((suite, rid))
                if not other:
                    continue
                for qid, top in tops.items():
                    n += 1
                    ok = top == other[qid]
                    agree += ok
                    if not ok:
                        misses.append(f"{suite}/{rid}/{qid} (served {other[qid]}, ours {top})")
            lines.append(
                f"arm {arm} vs arm {clef} (served) | {agree}/{n} | misses: "
                + ("; ".join(misses) if misses else "—")
            )
    # the jump arm against the step arm: the same request's answers.
    for jump, step in jump_step_pairs(rows):
        got = {}
        for r in arms[jump]:
            if not r["dropped"]:
                got.setdefault((r["suite"], r["id"]), r["tops"])
        other = {}
        for r in arms[step]:
            if not r["dropped"]:
                other.setdefault((r["suite"], r["id"]), r["tops"])
        n = same = 0
        for key, tops in got.items():
            if key not in other:
                continue
            n += 1
            same += tops == other[key]
        line = f"{jump} vs {step} | SAME {same}/{n}"
        exact = exactness([r for r in arms[jump] if not r["dropped"]], [r for r in arms[step] if not r["dropped"]])
        if exact:
            line += " | " + " | ".join(exact)
        lines.append(line)
    return lines


def stop_class_same(a: list[int], b: list[int], stops: list[int]) -> bool:
    """Two answers' ids are the same but for the stop: equal ids, or equal but the last, which both are stop
    ids. The jump arm names the vocabulary's end of sequence where the step arm picks its stop from a row."""
    if a == b:
        return True
    return bool(a) and bool(b) and len(a) == len(b) and a[:-1] == b[:-1] and a[-1] in stops and b[-1] in stops


def exactness(jump_rows: list[dict], step_rows: list[dict]) -> list[str]:
    """What a jump arm's answer shares with the step arm's beyond the picks, a request's first counted round
    each: the text, the finish reason and the usage exact, and, for rows that carry the generated ids (`ids`,
    with the `stop_ids` the vocabulary names), the ids but for the stop's class. Empty for rows that carry none
    of them (the chat route returns text, a finish reason and usage; only `/completion` returns ids)."""
    first_j: dict[tuple[str, str], dict] = {}
    first_s: dict[tuple[str, str], dict] = {}
    for rows, into in ((jump_rows, first_j), (step_rows, first_s)):
        for r in rows:
            into.setdefault((r["suite"], r["id"]), r)
    keys = [k for k in first_j if k in first_s and "text" in first_j[k] and "text" in first_s[k]]
    if not keys:
        return []
    out = []
    for field, label in (("text", "TEXT"), ("finish_reason", "FINISH"), ("usage", "USAGE")):
        out.append(f"{label} {sum(first_j[k].get(field) == first_s[k].get(field) for k in keys)}/{len(keys)}")
    with_ids = [k for k in keys if first_j[k].get("ids") is not None and first_s[k].get("ids") is not None]
    if with_ids:
        same = sum(stop_class_same(first_j[k]["ids"], first_s[k]["ids"], first_j[k].get("stop_ids") or []) for k in with_ids)
        out.append(f"IDS {same}/{len(with_ids)}")
    return out


def jump_step_pairs(rows: list[dict]) -> list[tuple[str, str]]:
    """The (jump, step) arm pairs to compare. An arm's role is the server's own report (`timings.constrained.arm`,
    kept in the row's `arm_kind`), else its name ending in `jump` or `step`; a jump arm pairs with the step arm
    of its seat, the name before that word (`ds41-jump` with `ds41-step`; the shell's `jump` and `step` are one
    seat), whichever window each ran in. The arms of no role are paired in the runs' order, the first run's
    against the next one's. A jump arm with no step arm stays alone."""
    gens: dict[str, dict] = {}
    for r in rows:
        if r["kind"] != "clef":
            gens.setdefault(r["arm"], {"run": r.get("run", 0), "kind": r.get("arm_kind")})

    def role_of(arm: str, info: dict) -> str | None:
        if info["kind"] in ("jump", "step"):
            return info["kind"]
        low = arm.lower()
        return next((role for role in ("jump", "step") if low.endswith(role)), None)

    def seat_of(arm: str) -> str:
        low = arm.lower()
        for role in ("jump", "step"):
            if low.endswith(role):
                return arm[: len(arm) - len(role)]
        return ""

    roles = {a: role_of(a, i) for a, i in gens.items()}
    pairs: list[tuple[str, str]] = []
    used: set[str] = set()
    for arm, role in roles.items():
        if role != "jump" or arm in used:
            continue
        other = next((a for a, r in roles.items() if r == "step" and a not in used and seat_of(a) == seat_of(arm)), None)
        if other:
            pairs.append((arm, other))
            used.update((arm, other))
    rest = sorted((a for a, r in roles.items() if r is None and a not in used), key=lambda a: gens[a]["run"])
    for i in range(0, len(rest) - 1, 2):
        pairs.append((rest[i], rest[i + 1]))
    return pairs


# ---------------------------------------------------------------- the servers


class Client:
    """One HTTP connection to a server, kept between the requests: the wall is the user's time to answer, and a
    new connection a request would add its setup to it. A connection the server closed while it was idle is
    opened again and the request sent once more (the retry's wall is the measurement: the failed attempt spent
    no answer)."""

    def __init__(self, url: str, name: str):
        u = urlsplit(url)
        if u.scheme not in ("", "http"):
            fail(f"the server at {url} is not an http one")
        self.url = url
        self.name = name
        self.host = u.hostname or "127.0.0.1"
        self.port = u.port or 80
        self.base = u.path.rstrip("/")
        self.conn: http.client.HTTPConnection | None = None

    def _open(self) -> None:
        self.conn = http.client.HTTPConnection(self.host, self.port, timeout=REQUEST_BOUND)

    def close(self) -> None:
        if self.conn is not None:
            try:
                self.conn.close()
            except OSError:
                pass
            self.conn = None

    def _exchange(self, method: str, path: str, data: bytes | None, headers: dict) -> tuple[int, str, float]:
        """One request and its answer: the status, the body's text and the wall in ms (the request to the last
        byte)."""
        for attempt in (1, 2):
            if self.conn is None:
                self._open()
            t0 = time.perf_counter()
            try:
                self.conn.request(method, self.base + path, body=data, headers=headers)
                resp = self.conn.getresponse()
                raw = resp.read().decode()
                wall = (time.perf_counter() - t0) * 1000.0
                return resp.status, raw, wall
            except (http.client.HTTPException, OSError) as e:
                self.close()
                if attempt == 2:
                    fail(f"{self.name} at {self.url}: {path} failed: {e}")
        fail(f"{self.name} at {self.url}: {path} failed")

    def post(self, path: str, body: dict) -> tuple[int, str, float]:
        """The POST's answer: the status, the body's text and the wall in ms (the POST to the last byte)."""
        data = json.dumps(body).encode()
        return self._exchange("POST", path, data, {"Content-Type": "application/json", "Content-Length": str(len(data))})

    def get(self, path: str) -> tuple[int, str]:
        """The GET's answer: the status and the body's text. Only the untimed passes read with it."""
        status, raw, _ = self._exchange("GET", path, None, {})
        return status, raw


class Arm:
    """One arm: a name, a kind (clef or qwen) and the server it posts to."""

    def __init__(self, name: str, url: str, kind: str):
        self.name = name
        self.kind = kind
        self.url = url
        self.client = Client(url, name)
        self.ids_cache: dict[tuple[str, str], list[tuple[str, object]]] = {}

    def tokenize(self, text: str) -> list[int]:
        """The server's own ids of a text (`POST /tokenize`): the program's pieces are tokenized on the server
        that compiles them."""
        status, raw, _ = self.client.post("/tokenize", {"content": text})
        if status != 200:
            fail(f"{self.name}: /tokenize answered {status}: {raw[:200]}")
        body = json.loads(raw)
        ids = body.get("tokens")
        if not isinstance(ids, list):
            fail(f"{self.name}: /tokenize answered without tokens: {raw[:200]}")
        return [int(i) for i in ids]

    def program_ids(self, suite: str, req: dict) -> list[tuple[str, object]]:
        """The program's ids: each forced text and each option tokenized on its own (the server's compile)."""
        key = (suite, req["id"])
        if key not in self.ids_cache:
            out: list[tuple[str, object]] = []
            for kind, body in merged(program_parts(req)):
                if kind == "T":
                    out.append(("F", self.tokenize(str(body))))
                else:
                    out.append(("P", [self.tokenize(str(o)) for o in body]))
            self.ids_cache[key] = out
        return self.ids_cache[key]


def check_counts(arm: Arm, suite: str, req: dict, obj: dict, reported: dict) -> str:
    """The engine calls the program makes against the ones the server reports: a mismatch is a named error. The
    server counts the calls after the prompt, so the prediction leaves the prompt's closing step out; a server
    that counts it is accepted, and the run says which convention it saw. A server that reports
    `next_after_calls` has the jump levers and is held to `plan_levers` (or `step_levers`) on every count of
    `LEVER_KEYS`, with no convention to choose."""
    where = f"{suite}/{req['id']}"
    kind = reported.get("arm")
    if kind not in ("jump", "step"):
        fail(f"{where}: the server reports the arm {kind!r}, not jump or step")
    toks = stream(arm.program_ids(suite, req), picks_from_answer(req, obj))
    if "next_after_calls" in reported:
        want = plan_levers(toks) if kind == "jump" else step_levers(toks)
        for key in LEVER_KEYS:
            if reported.get(key) != want[key]:
                fail(
                    f"{where}: the server reports {key} {reported.get(key)}, the {kind} arm's program makes "
                    f"{want[key]} (the plan: { {k: want[k] for k in LEVER_KEYS} })"
                )
        return "the jump levers"
    calls = plan_step(toks) if kind == "step" else plan_jump(toks)
    want_next, want_prefill = plan_counts(calls)
    got_next = reported.get("next_calls")
    got_prefill = reported.get("prefill_calls")
    if got_next == want_next:
        convention = "after the prompt"
    elif got_next == want_next + 1:
        convention = "the prompt's closing step counted"
    else:
        fail(
            f"{where}: the server reports next_calls {got_next}, the program makes {want_next} "
            f"(the calls: {calls})"
        )
    if got_prefill != want_prefill:
        fail(
            f"{where}: the server reports prefill_calls {got_prefill}, the program makes {want_prefill} "
            f"(the calls: {calls})"
        )
    return convention


# ---------------------------------------------------------------- the run


def load_suite(path: Path) -> list[dict]:
    """A suite's requests, in the file's order."""
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        req = json.loads(line)
        where = f"{path.name}/{req.get('id', '?')}"
        check_request(req, where)
        out.append(req)
    return out


def suite_name(path: str) -> str:
    """A suite file's name in the tables: `en` and `ko` for the two files the official rows are of, else the
    file's stem without a leading `suite-` (`suite-hard.jsonl` is `hard`)."""
    for name, known, _ in SUITES:
        if path == known:
            return name
    stem = Path(path).stem
    return stem[len("suite-"):] if stem.startswith("suite-") and len(stem) > len("suite-") else stem


def gold_path(path: Path) -> Path:
    """The gold file beside a suite file: `suite-hard.jsonl` has `suite-hard.gold.jsonl`."""
    return path.with_suffix(".gold.jsonl")


def load_gold(gpath: Path, name: str, reqs: list[dict]) -> dict[str, dict[str, str]]:
    """A suite's gold labels, request id -> question id -> option id. Every line is checked against the suite: a
    request it names that the suite lacks, a question the request lacks, a word that is not an option of the
    question, a request named twice, a request with no line and a key other than `GOLD_KEYS` are named errors;
    `why` and `source` are the labeller's notes and are not read."""
    by_id = {r["id"]: r for r in reqs}
    out: dict[str, dict[str, str]] = {}
    for n, line in enumerate(gpath.read_text().splitlines(), 1):
        if not line.strip():
            continue
        where = f"{gpath.name}:{n}"
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            fail(f"{where}: the gold line is no JSON: {e}")
        if not isinstance(row, dict) or "id" not in row or not isinstance(row.get("gold"), dict):
            fail(f"{where}: a gold line is {{\"id\", \"gold\": {{question id: option id}}}}")
        unknown = sorted(set(row) - GOLD_KEYS)
        if unknown:
            fail(f"{where}: the gold line has the key(s) {unknown}; a line is {sorted(GOLD_KEYS)} and nothing else")
        if "source" in row and not isinstance(row["source"], dict):
            fail(f"{where}: the gold 'source' is {{\"url\", \"quote\"}}, not {type(row['source']).__name__}")
        rid = row["id"]
        if rid not in by_id:
            fail(f"{where}: the gold names the id '{rid}', which the suite {name} does not hold")
        if rid in out:
            fail(f"{where}: the gold names the id '{rid}' twice")
        questions = by_id[rid]["questions"]
        labels: dict[str, str] = {}
        for qid, opt in row["gold"].items():
            if qid not in questions:
                fail(f"{where}: the gold of '{rid}' names the question '{qid}', which the request lacks")
            if isinstance(opt, bool):
                opt = "true" if opt else "false"
            elif isinstance(opt, int):
                opt = str(opt)
            elif questions[qid]["type"] == "noul" and opt in NOUL_YES_NO:
                opt = NOUL_YES_NO[opt]
            ids = option_ids(questions[qid])
            if opt not in ids:
                fail(f"{where}: the gold of '{rid}' gives '{qid}' the option {opt!r}, not one of {ids}")
            labels[qid] = opt
        out[rid] = labels
    missing = [rid for rid in by_id if rid not in out]
    if missing:
        fail(f"{gpath.name}: no gold line for the request(s) {missing} of the suite {name}")
    return out


def load_suites(paths: list[str] | None = None) -> list[dict]:
    """The suites to run: {name, path, reqs, gold} a file, in the order named (the two default files when none
    is). A path is relative to the repository unless absolute; a file that is not there, two files of one name
    and a request id used twice in one file are named errors. `gold` is `None` for a suite with no gold file."""
    paths = list(paths) if paths else [p for _, p, _ in SUITES]
    out: list[dict] = []
    for path in paths:
        full = Path(path) if os.path.isabs(path) else ROOT / path
        if not full.is_file():
            fail(f"no suite file at {path}")
        name = suite_name(path)
        if any(s["name"] == name for s in out):
            fail(f"two suite files are named '{name}': {path} and {next(s for s in out if s['name'] == name)['path']}")
        reqs = load_suite(full)
        ids = [r["id"] for r in reqs]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            fail(f"{full.name}: the request id(s) {dup} are used twice")
        gfile = gold_path(full)
        gold = load_gold(gfile, name, reqs) if gfile.is_file() else None
        out.append({"name": name, "path": path, "reqs": reqs, "gold": gold})
    return out


def suite_requests(suites: list[dict]) -> list[tuple[str, dict]]:
    """The timed requests of the suites, a suite after a suite in their files' order."""
    return [(s["name"], req) for s in suites for req in s["reqs"]]


def check_suites(argv: list[str]) -> int:
    """`check-suites [--suite FILE ...]`: every suite and gold file read and checked, one line each."""
    paths = []
    i = 0
    while i < len(argv):
        if argv[i] != "--suite" or i + 1 >= len(argv):
            fail(f"check-suites takes --suite FILE, got '{argv[i]}'")
        paths.append(argv[i + 1])
        i += 2
    for s in load_suites(paths):
        questions = sum(len(r["questions"]) for r in s["reqs"])
        gold = "no gold file" if s["gold"] is None else f"gold of {sum(len(g) for g in s['gold'].values())} questions"
        print(f"[suites] {s['name']} ({s['path']}): {len(s['reqs'])} requests, {questions} questions, {gold}", flush=True)
    return 0


def post_one(arm: Arm, suite: str, req: dict, rnd: int, dropped: bool, model: str | None) -> dict:
    """One request to one arm: the wall, the server's own split, the tops and the call-count check. The row also
    holds the answer's text as received under `_raw`, which `measure` takes out (`take_raw`) before the row is kept."""
    where = f"{suite}/{req['id']} on arm {arm.name}"
    if arm.kind == "clef":
        status, raw, wall = arm.client.post("/v1/systemone", clef_body(req))
    else:
        status, raw, wall = arm.client.post("/v1/chat/completions", chat_body(req, model))
    if status != 200:
        fail(f"{where}: the server answered {status}: {raw[:300]}")
    try:
        resp = json.loads(raw)
    except json.JSONDecodeError as e:
        fail(f"{where}: the answer is no JSON: {e}: {raw[:200]}")
    timings = resp.get("timings") or {}
    arm_kind = None
    calls = None
    extra: dict = {}
    if arm.kind == "clef":
        tops = tops_from_answers(resp, req, where)
        server_ms = None
        if timings.get("prompt_ms") is not None and timings.get("head_ms") is not None:
            server_ms = float(timings["prompt_ms"]) + float(timings["head_ms"])
    else:
        try:
            choice = resp["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError):
            fail(f"{where}: the answer has no choices[0].message.content: {raw[:300]}")
        server_ms = None
        if timings.get("prompt_ms") is not None and timings.get("predicted_ms") is not None:
            server_ms = float(timings["prompt_ms"]) + float(timings["predicted_ms"])
        extra = {"text": content, "finish_reason": choice.get("finish_reason"), "usage": resp.get("usage")}
        try:
            obj = json.loads(content)
        except (json.JSONDecodeError, TypeError) as e:
            fail(f"{where}: the content is no JSON: {e}: {str(content)[:200]}")
        if not isinstance(obj, dict):
            fail(f"{where}: the content is not a JSON object: {content[:200]}")
        picks_from_answer(req, obj)
        tops = tops_from_answer(req, obj)
        c = timings.get("constrained")
        if isinstance(c, dict):
            extra["convention"] = check_counts(arm, suite, req, obj, c)
            arm_kind = c["arm"]
            calls = {k: v for k, v in c.items() if isinstance(v, int) and not isinstance(v, bool)}
    picks = {qid: pick_of_top(req["questions"][qid], top) for qid, top in tops.items()}
    host_ms = wall - server_ms if server_ms is not None else None
    return {
        "arm": arm.name,
        "kind": arm.kind,
        "arm_kind": arm_kind,
        "suite": suite,
        "id": req["id"],
        "round": rnd,
        "dropped": dropped,
        "wall_ms": wall,
        "server_ms": server_ms,
        "host_ms": host_ms,
        "prompt_n": timings.get("prompt_n"),
        "prompt_ms": timings.get("prompt_ms"),
        "head_ms": timings.get("head_ms"),
        "predicted_ms": timings.get("predicted_ms"),
        "calls": calls,
        "tops": tops,
        "picks": picks,
        **extra,
        "_raw": raw,
    }


def take_raw(row: dict) -> str:
    """The answer's text a row of `post_one` carries, taken out: a row is kept without it."""
    return row.pop("_raw")


def bucket_of(rows: list[dict]) -> dict[tuple[str, str], str]:
    """Each request's bucket: short (a prompt under `SHORT_LIMIT` ids) or long, from the prompt length the arms
    reported."""
    out: dict[tuple[str, str], str] = {}
    for r in rows:
        key = (r["suite"], r["id"])
        n = r.get("prompt_n")
        if n is None:
            continue
        out.setdefault(key, "short" if int(n) < SHORT_LIMIT else "long")
    return out


# The route the Clef arm posts to and the generative arms' untimed SystemOne pass posts to.
SYSTEMONE = "/v1/systemone"
# A file name the bodies' paths are made of: an arm, a suite or a request id.
FILE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def body_rel(arm: str, suite: str, rid: str, dir_suffix: str = "") -> str:
    """A body's path relative to the out dir: `bodies/<arm><dir_suffix>/<suite>-<id>.json`."""
    return f"bodies/{arm}{dir_suffix}/{suite}-{rid}.json"


def check_body_names(arms: list[Arm], reqs: list[tuple[str, dict]]) -> None:
    """The bodies' files are named from the arm, the suite and the request id, so each must be a plain file name,
    no two directories may be one (an arm named like another's `-systemone` directory) and no two requests may
    share a file. A named error before the first POST, never a file written over another."""
    for arm in arms:
        if not FILE_NAME.match(arm.name):
            fail(f"the arm name '{arm.name}' is no plain file name: the bodies are saved under it")
    dirs = [a.name for a in arms] + [f"{a.name}-systemone" for a in arms if a.kind != "clef"]
    dup = sorted({d for d in dirs if dirs.count(d) > 1})
    if dup:
        fail(f"the bodies' directories {dup} are named twice (an arm named like another's '-systemone' directory)")
    files = []
    for suite, req in reqs:
        for part in (suite, req["id"]):
            if not FILE_NAME.match(str(part)):
                fail(f"'{part}' ({suite}/{req['id']}) is no plain file name: the bodies are saved under it")
        files.append(body_rel("x", suite, req["id"]))
    dup = sorted({f for f in files if files.count(f) > 1})
    if dup:
        fail(f"two requests share a body file name: {[Path(d).name for d in dup]}")


def make_dir(path: Path) -> None:
    """A directory made, or a named error."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        fail(f"cannot make {path}: {e}")


def save(path: Path, data: bytes) -> None:
    """A file written exactly, or a named error."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    except OSError as e:
        fail(f"cannot write {path}: {e}")


def arm_routes(arm: Arm) -> tuple[list[str] | None, str]:
    """The routes the arm's server lists in `GET /props` (`routes`, a list of paths; none for a server that has no
    such key), or None and why `/props` could not be read."""
    try:
        status, raw = arm.client.get("/props")
    except Stop as e:
        return None, str(e).removeprefix("sjcmp: ")
    if status != 200:
        return None, f"GET /props answered {status}"
    try:
        props = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"GET /props is no JSON: {e}"
    routes = props.get("routes", []) if isinstance(props, dict) else None
    if not isinstance(routes, list) or not all(isinstance(r, str) for r in routes):
        return None, "GET /props has no list of paths under routes"
    return routes, ""


def systemone_bodies(arm: Arm, reqs: list[tuple[str, dict]], out: Path) -> None:
    """A generative arm's SystemOne bodies, untimed: when the server lists `/v1/systemone` in its `/props`' routes,
    each request of each suite is posted once to it and the body saved as received under
    `bodies/<arm>-systemone/<suite>-<id>.json`. A non-200 answer is saved too, its status in `<that file>.status`.
    Only the 200s count as ok. A server without the route, or whose `/props` cannot be read, is a named line, not
    an error; so is a server that stops answering (the pass ends there and the rest count as not ok). The pass
    makes no row and touches none: it runs after the rounds, when the timed regions are over."""
    routes, why = arm_routes(arm)
    if routes is None:
        print(f"[bodies] {arm.name}: the server lists no {SYSTEMONE}; skipped ({why})", flush=True)
        return
    if SYSTEMONE not in routes:
        print(f"[bodies] {arm.name}: the server lists no {SYSTEMONE}; skipped", flush=True)
        return
    ok = 0
    for suite, req in reqs:
        where = f"{suite}/{req['id']}"
        try:
            status, raw, _ = arm.client.post(SYSTEMONE, clef_body(req))
        except Stop as e:
            print(f"[bodies] {arm.name}: {where}: {str(e).removeprefix('sjcmp: ')}; the pass ends here", flush=True)
            break
        path = out / body_rel(arm.name, suite, req["id"], "-systemone")
        save(path, raw.encode())
        if status == 200:
            ok += 1
        else:
            save(path.with_name(path.name + ".status"), f"{status}\n".encode())
            print(f"[bodies] {arm.name}: {where}: {SYSTEMONE} answered {status}", flush=True)
    print(f"[bodies] {arm.name}: systemone {ok} ok / {len(reqs)}", flush=True)


def run(argv: list[str]) -> int:
    """The run: the arms, the suites, then `measure`."""
    opts = parse_run_args(argv)
    arms = [Arm(name, url, kind) for kind, name, url in opts["arms"]]
    if not arms:
        fail("no arms: --clef NAME=URL and --qwen NAME=URL name the servers")
    names = [a.name for a in arms]
    if len(set(names)) != len(names):
        fail(f"the arms' names are not distinct: {names}")
    return measure(arms, load_suites(opts["suites"]), opts)


def measure(arms: list[Arm], suites: list[dict], opts: dict) -> int:
    """The rounds, the requests, the rows and the bodies. The timed regions are `post_one`'s POSTs alone; between
    them the loop only keeps the first counted round's answer text in memory. Past the last round: the rows, then
    the kept bodies as files (`bodies/<arm>/<suite>-<id>.json`, the row naming its file under `body`), then each
    generative arm's untimed SystemOne pass. The bodies' directories are made before the first POST."""
    names = [a.name for a in arms]
    reqs = suite_requests(suites)
    check_body_names(arms, reqs)
    out = Path(opts["out"])
    for arm in arms:
        make_dir(out / "bodies" / arm.name)
    rows: list[dict] = []
    held: dict[tuple[str, str, str], str] = {}
    # The first post of every request to every arm: the first-request state, dropped. It also says how long each
    # prompt is, so the buckets are known before the rounds.
    for suite, req in reqs:
        for arm in arms:
            row = post_one(arm, suite, req, 0, True, opts["model"])
            take_raw(row)
            rows.append(row)
    buckets = bucket_of(rows)
    for suite, req in reqs:
        buckets.setdefault((suite, req["id"]), "short")
    for r in rows:
        r.setdefault("bucket", buckets.get((r["suite"], r["id"]), "short"))
    short = [(s, r) for s, r in reqs if buckets[(s, r["id"])] == "short"]
    long = [(s, r) for s, r in reqs if buckets[(s, r["id"])] != "short"]
    print(f"[config] arms={names} rounds={opts['rounds']} rounds_long={opts['rounds_long']} "
          f"short={len(short)} long={len(long)}", flush=True)
    rot = 0
    for bucket, group, rounds in (("short", short, opts["rounds"]), ("long", long, opts["rounds_long"])):
        for _ in range(rounds):
            for suite, req in group:
                for k in range(len(arms)):
                    arm = arms[(k + rot) % len(arms)]
                    row = post_one(arm, suite, req, rot + 1, False, opts["model"])
                    raw = take_raw(row)
                    row["bucket"] = bucket
                    key = (arm.name, suite, req["id"])
                    if key not in held:
                        held[key] = raw
                        row["body"] = body_rel(*key)
                    rows.append(row)
                    print(
                        f"r{row['round']} {arm.name} {suite} {req['id']} | wall {row['wall_ms']:.1f} ms | "
                        f"prompt_n {row['prompt_n']} | tops "
                        + " ".join(f"{k}={v}" for k, v in row["tops"].items()),
                        flush=True,
                    )
            rot += 1
    for arm in arms:
        arm.client.close()
    # The timed regions are over. The rows go first: they are the measurement.
    meta = {s["name"]: {"path": s["path"], "gold": s["gold"]} for s in suites}
    (out / "rows.json").write_text(json.dumps({"arms": names, "suites": meta, "rows": rows}, indent=1))
    print(f"[rows] {out / 'rows.json'}", flush=True)
    for (arm_name, suite, rid), raw in held.items():
        save(out / body_rel(arm_name, suite, rid), raw.encode())
    print(f"[bodies] timed: {len(held)} bodies of {len(arms)} arms under {out / 'bodies'}", flush=True)
    for arm in arms:
        if arm.kind != "clef":
            systemone_bodies(arm, reqs, out)
    return 0


def parse_run_args(argv: list[str]) -> dict:
    """The run's arguments: the arms, the rounds and the out dir."""
    opts: dict = {"arms": [], "suites": [], "rounds": 20, "rounds_long": 8, "out": ".", "model": None}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--clef", "--qwen"):
            if i + 1 >= len(argv):
                fail(f"{a} needs a NAME=URL")
            spec = argv[i + 1]
            if "=" not in spec:
                fail(f"{a} needs a NAME=URL, got '{spec}'")
            name, url = spec.split("=", 1)
            opts["arms"].append((a[2:], name, url))
            i += 2
            continue
        if a == "--suite":
            if i + 1 >= len(argv):
                fail("--suite needs a file")
            opts["suites"].append(argv[i + 1])
            i += 2
            continue
        if a in ("--rounds", "--rounds-long", "--out", "--model"):
            if i + 1 >= len(argv):
                fail(f"{a} needs a value")
            key = a[2:].replace("-", "_")
            opts[key] = argv[i + 1]
            i += 2
            continue
        fail(f"'{a}' is not an argument of `run`")
    for key in ("rounds", "rounds_long"):
        try:
            opts[key] = int(opts[key])
        except ValueError:
            fail(f"--{key.replace('_', '-')} is a whole number, got '{opts[key]}'")
        if opts[key] < 1:
            fail(f"--{key.replace('_', '-')} is at least 1, got {opts[key]}")
    return opts


# ---------------------------------------------------------------- the table


def table(argv: list[str]) -> int:
    """The tables of the runs' rows."""
    data = Path(os.environ.get("BLOOMERY_DATA", "/root/bloomery-data"))
    files: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--data":
            if i + 1 >= len(argv):
                fail("table: --data needs a value")
            data = Path(argv[i + 1])
            i += 2
            continue
        files.append(argv[i])
        i += 1
    if not files:
        fail("table: the rows' files are the arguments")
    rows: list[dict] = []
    suites: dict = {}
    for n, path in enumerate(files):
        p = Path(path)
        if not p.exists():
            fail(f"table: no rows at {p}")
        body = json.loads(p.read_text())
        for name, meta in (body.get("suites") or {}).items():
            if name in suites and suites[name] != meta:
                fail(f"table: the suite '{name}' differs between the rows' files ({suites[name]['path']} and {meta['path']})")
            suites[name] = meta
        for r in body["rows"]:
            if "bucket" not in r:
                fail(f"table: a row of {p} has no bucket (the rows are the runner's own)")
            r["run"] = n
            rows.append(r)
    # The official rows are of the English and Korean suites: a sitting of other suites needs none.
    official = any(r["suite"] in ("en", "ko") for r in rows)
    missing = [str(data / refp) for _, _, refp in SUITES if official and not (data / refp).exists()]
    refs = {} if missing or not official else e2e.official_rows(data)
    if missing:
        print(f"=== agreement (the official rows are not at {', '.join(missing)}: the agreement against them is not shown) ===")
    for line in stats_table(rows, suites):
        print(line)
    for line in agreement_tables(rows, refs):
        print(line)
    for line in gold_table(rows, suites):
        print(line)
    return 0


# ---------------------------------------------------------------- the convert


def convert(argv: list[str]) -> int:
    """One converted request: the chat body the generative seat answers."""
    suite = "en"
    want = None
    i = 0
    while i < len(argv):
        if argv[i] in ("--suite", "--id"):
            if i + 1 >= len(argv):
                fail(f"{argv[i]} needs a value")
            if argv[i] == "--suite":
                suite = argv[i + 1]
            else:
                want = argv[i + 1]
            i += 2
            continue
        fail(f"'{argv[i]}' is not an argument of `convert`")
    if suite not in ("en", "ko"):
        fail(f"--suite is en or ko, got '{suite}'")
    for name, path, _ in SUITES:
        if name != suite:
            continue
        for req in load_suite(ROOT / path):
            if want and req["id"] != want:
                continue
            print(json.dumps(chat_body(req), ensure_ascii=False, indent=2))
            return 0
    fail(f"convert: no request '{want}' in the {suite} suite")


# ---------------------------------------------------------------- the self-test


def self_test() -> None:
    """The self-test: pure, no server, no clock. Every case was red before the code it checks."""
    n = 0
    fails = 0

    def ok(what: str) -> None:
        nonlocal n
        n += 1
        print(f"ok {what}")

    def bad(what: str, why: str) -> None:
        nonlocal n, fails
        n += 1
        fails += 1
        print(f"FAIL {what}: {why}")

    def check(what: str, got, want) -> None:
        if got == want:
            ok(what)
        else:
            bad(what, f"got {got!r}, want {want!r}")

    # --- the conversion: every question type of the four suites.
    kinds = set()
    for path in [ROOT / p for _, p, _ in SUITES] + [ROOT / p for p in WIRE_SUITES]:
        if not path.exists():
            bad(f"the suite {path.name} is in the tree", "it is not")
            continue
        for req in load_suite(path):
            kinds.update(q["type"] for q in req["questions"].values())
            body = chat_body(req)
            schema = body["response_format"]["json_schema"]["schema"]
            props = schema["properties"]
            if list(props) != list(req["questions"]):
                bad(f"{path.name}/{req['id']}", "the properties are not the question ids in order")
                continue
            for qid, q in req["questions"].items():
                prop = props[qid]
                if q["type"] == "choice":
                    want = {"type": "string", "enum": sorted(q["criteria"])}
                elif q["type"] == "score":
                    want = {"type": "integer", "enum": list(range(len(q["criteria"])))}
                else:
                    want = {"type": "boolean"}
                if prop != want:
                    bad(f"{path.name}/{req['id']}/{qid}", f"the property is {prop}, want {want}")
            if body["cache_prompt"] is not False:
                bad(f"{path.name}/{req['id']}", "cache_prompt is not off")
            if body["chat_template_kwargs"]["enable_thinking"] is not False:
                bad(f"{path.name}/{req['id']}", "thinking is not off")
            sysmsg = body["messages"][0]["content"]
            if not sysmsg.startswith(QSYS) or "SCHEMA FIELDS:" not in sysmsg:
                bad(f"{path.name}/{req['id']}", "the system message is not the schema text")
            if body["messages"][1]["content"] != "STATE:\n" + render(req["state"]):
                bad(f"{path.name}/{req['id']}", "the user message is not the state")
    check("the conversion covers every question type", sorted(kinds), ["choice", "noul", "score"])

    # a choice's enum is in Clef's sorted order, a score's is 0..n-1, a noul's is a boolean.
    req = {
        "id": "t",
        "state": "s",
        "questions": {
            "zeta": {"type": "choice", "criteria": {"b": "B", "a": "A"}},
            "level": {"type": "score", "criteria": ["low", "high"]},
            "flag": {"type": "noul"},
        },
    }
    schema = json_schema(req)
    check("a choice's enum is sorted", schema["properties"]["zeta"], {"type": "string", "enum": ["a", "b"]})
    check("a score's enum is 0..n-1", schema["properties"]["level"], {"type": "integer", "enum": [0, 1]})
    check("a noul is a boolean", schema["properties"]["flag"], {"type": "boolean"})
    check("the schema's fields are the question ids", list(schema["properties"]), ["zeta", "level", "flag"])

    # the schema text is Clef's own (crates/decision/src/encode.rs).
    text = schema_text(req)
    check(
        "the schema text is Clef's",
        text,
        "\n\nSCHEMA FIELDS:\n"
        "\nFIELD 1\nID: zeta\nTYPE: choice\nINSTRUCTION: zeta\nALLOWED OPTIONS:\n"
        'OPTION 1: {"description":"A","option_id":"a"}\n'
        'OPTION 2: {"description":"B","option_id":"b"}\n'
        "END FIELD\n"
        "\nFIELD 2\nID: level\nTYPE: score\nINSTRUCTION: level\nALLOWED OPTIONS:\n"
        'OPTION 1: {"description":"low","option_id":"0"}\n'
        'OPTION 2: {"description":"high","option_id":"1"}\n'
        "END FIELD\n"
        "\nFIELD 3\nID: flag\nTYPE: noul\nINSTRUCTION: flag\nALLOWED OPTIONS:\n"
        'OPTION 1: {"description":"The proposition is true or the answer is yes.","option_id":"true"}\n'
        'OPTION 2: {"description":"The proposition is false or the answer is no.","option_id":"false"}\n'
        "END FIELD\n",
    )

    # a malformed row is refused by name.
    for bad_row, want in (
        ({"state": "s", "questions": {"q": {"type": "bool"}}}, "not noul, choice or score"),
        ({"state": "s", "questions": {}}, "no questions"),
        ({"questions": {"q": {"type": "noul"}}}, "no 'state'"),
        ({"state": "s", "questions": {"q": {"type": "choice"}}}, "no criteria"),
        ({"state": "s", "questions": {"q": {"type": "score", "criteria": []}}}, "no criteria"),
        ({"state": "s", "questions": {"": {"type": "noul"}}}, "id is empty"),
    ):
        try:
            check_request(bad_row, "the row")
        except Stop as e:
            if want in str(e):
                ok(f"a malformed row is refused ({want})")
            else:
                bad("a malformed row is refused", f"{e} does not name {want}")
        else:
            bad("a malformed row is refused", f"{bad_row} passed")

    # --- the back-mapping.
    tops = tops_from_answer(req, {"zeta": "b", "level": 1, "flag": True})
    check("the back-mapping", tops, {"zeta": "b", "level": "1", "flag": "yes"})
    check("a noul's no", tops_from_answer(req, {"zeta": "a", "level": 0, "flag": False})["flag"], "no")
    for wrong, want in (
        ({"zeta": "c", "level": 0, "flag": True}, "not one of"),
        ({"zeta": "a", "level": 9, "flag": True}, "not a level index"),
        ({"zeta": "a", "level": 0, "flag": "yes"}, "not a boolean"),
        ({"zeta": "a", "level": 0}, "no 'flag'"),
    ):
        try:
            picks_from_answer(req, wrong)
        except Stop as e:
            if want in str(e):
                ok(f"a wrong answer is refused ({want})")
            else:
                bad("a wrong answer is refused", f"{e} does not name {want}")
        else:
            bad("a wrong answer is refused", f"{wrong} passed")
    # PIN(2026-10-20): a noul's `true` is pick 0, the program's first option (`program_parts` lists `true` then
    # `false`); the plan walked the `false` path for a `true` answer, which counts the same ids only when both
    # words are one id. Was [1, 1, 1].
    check("the picks of an answer", picks_from_answer(req, {"zeta": "b", "level": 1, "flag": True}), [1, 1, 0])

    # --- the rotation (ab-decode's rule: arm k of round r is arm (k + r - 1) mod n).
    check("the rotation of round 1", [(k + 0) % 3 for k in range(3)], [0, 1, 2])
    check("the rotation of round 2", [(k + 1) % 3 for k in range(3)], [1, 2, 0])
    check("the rotation of round 3", [(k + 2) % 3 for k in range(3)], [2, 0, 1])
    check("the rotation of two arms", [(k + 1) % 2 for k in range(2)], [1, 0])

    # --- the call plan and the count check (a byte tokenizer: one id a byte).
    def bytes_tok(text: str) -> list[int]:
        return [ord(c) for c in text]

    ko = {
        "id": "ko-06",
        "state": "s",
        "questions": {
            "intent": {"type": "choice", "criteria": {"refund": "R", "tracking": "T", "cancel": "C", "other": "O"}},
            "angry": {"type": "noul"},
        },
    }
    parts = merged(program_parts(ko))
    ids = []
    for kind, body in parts:
        if kind == "T":
            ids.append(("F", bytes_tok(str(body))))
        else:
            ids.append(("P", [bytes_tok(str(o)) for o in body]))
    toks = stream(ids, [0, 0])  # "cancel" and "true"
    check("the program's ids", len(toks), 12 + 6 + 12 + 4 + 1)
    calls = plan_jump(toks)
    check(
        "the jump arm's calls",
        calls,
        [("next", 1), ("prefill", 11), ("next", 1), ("prefill", 17), ("next", 1), ("prefill", 4), ("next", 1)],
    )
    check("the jump arm's counts after the prompt", plan_counts(calls), (3, 3))
    step = plan_step(toks)
    check("the step arm's calls", len(step), len(toks) + 1)
    check("the step arm's counts after the prompt", plan_counts(step), (len(toks), 0))

    # a mismatch is refused by name.
    class FakeArm:
        name = "fake"

        def program_ids(self, suite, req):
            return ids

    try:
        check_counts(FakeArm(), "en", ko, {"intent": "cancel", "angry": True},
                     {"arm": "jump", "next_calls": 99, "prefill_calls": 3})
    except Stop as e:
        if "next_calls 99" in str(e):
            ok("a count mismatch is refused")
        else:
            bad("a count mismatch is refused", str(e))
    else:
        bad("a count mismatch is refused", "it passed")
    try:
        check_counts(FakeArm(), "en", ko, {"intent": "cancel", "angry": True},
                     {"arm": "jump", "next_calls": 3, "prefill_calls": 9})
    except Stop as e:
        if "prefill_calls 9" in str(e):
            ok("a prefill mismatch is refused")
        else:
            bad("a prefill mismatch is refused", str(e))
    else:
        bad("a prefill mismatch is refused", "it passed")
    try:
        check_counts(FakeArm(), "en", ko, {"intent": "cancel", "angry": True},
                     {"arm": "walk", "next_calls": 3, "prefill_calls": 3})
    except Stop as e:
        if "not jump or step" in str(e):
            ok("an unknown arm is refused")
        else:
            bad("an unknown arm is refused", str(e))
    else:
        bad("an unknown arm is refused", "it passed")
    # a server that counts the prompt's closing step is accepted, and the run says so.
    class ConvArm:
        name = "fake"

        def program_ids(self, suite, req):
            return ids

    conv = check_counts(ConvArm(), "en", ko, {"intent": "cancel", "angry": True},
                        {"arm": "jump", "next_calls": 4, "prefill_calls": 3})
    check("the closing step's convention is named", conv, "the prompt's closing step counted")

    # --- the stats on a fixed sample.
    sample = [10.0, 20.0, 30.0, 40.0]
    check("the p50 of a sample", percentile(sample, 50), 25.0)
    check("the p95 of a sample", percentile(sample, 95), 38.5)
    check("the p50 of one sample", percentile([7.0], 50), 7.0)
    check("the p95 of a sample of two", percentile([1.0, 3.0], 95), 2.9)
    rows = [
        {"arm": "A", "kind": "clef", "suite": "en", "id": "r1", "round": 1, "dropped": False, "wall_ms": 100.0,
         "server_ms": 90.0, "host_ms": 10.0, "bucket": "short", "tops": {"q": "a"}},
        {"arm": "A", "kind": "clef", "suite": "en", "id": "r1", "round": 2, "dropped": False, "wall_ms": 200.0,
         "server_ms": 190.0, "host_ms": 10.0, "bucket": "short", "tops": {"q": "a"}},
        {"arm": "A", "kind": "clef", "suite": "ko", "id": "r2", "round": 1, "dropped": False, "wall_ms": 300.0,
         "server_ms": 290.0, "host_ms": 10.0, "bucket": "long", "tops": {"q": "b"}},
        {"arm": "A", "kind": "clef", "suite": "en", "id": "r1", "round": 0, "dropped": True, "wall_ms": 999.0,
         "server_ms": 1.0, "host_ms": 1.0, "bucket": "short", "tops": {"q": "a"}},
    ]
    lines = stats_table(rows)
    check("the stats' rows", len(lines), 5)
    check(
        "the stats' short bucket",
        any("bucket short" in l and "p50 150.0 ms" in l and "p95 195.0 ms" in l for l in lines),
        True,
    )
    check("the dropped row is out", any("999.0" in l for l in lines), False)

    # --- the agreement tables.
    refs = {("en", "r1"): {"answers": {"q": {"type": "choice", "choice": "a", "probabilities": {"a": 0.9}}}}}
    qrows = [
        {"arm": "A", "kind": "clef", "suite": "en", "id": "r1", "round": 1, "dropped": False, "wall_ms": 1.0,
         "server_ms": 1.0, "host_ms": 0.0, "bucket": "short", "tops": {"q": "a"}},
        {"arm": "B", "kind": "qwen", "arm_kind": "jump", "suite": "en", "id": "r1", "round": 1, "dropped": False,
         "wall_ms": 1.0, "server_ms": 1.0, "host_ms": 0.0, "bucket": "short", "tops": {"q": "a"}},
    ]
    out = agreement_tables(qrows, refs)
    check("the agreement with the official rows", any("1/1" in l for l in out), True)
    check("the agreement with the served arm", any("vs arm A (served) | 1/1" in l for l in out), True)

    # the jump arm against the step arm: the same request's answers.
    srows = [
        {"arm": "jump", "kind": "qwen", "arm_kind": "jump", "suite": "en", "id": "r1", "round": 1,
         "dropped": False, "wall_ms": 1.0, "server_ms": 1.0, "host_ms": 0.0, "bucket": "short",
         "tops": {"q": "a"}},
        {"arm": "step", "kind": "qwen", "arm_kind": "step", "suite": "en", "id": "r1", "round": 1,
         "dropped": False, "wall_ms": 1.0, "server_ms": 1.0, "host_ms": 0.0, "bucket": "short",
         "tops": {"q": "a"}},
        {"arm": "step", "kind": "qwen", "arm_kind": "step", "suite": "en", "id": "r1", "round": 2,
         "dropped": False, "wall_ms": 1.0, "server_ms": 1.0, "host_ms": 0.0, "bucket": "short",
         "tops": {"q": "b"}},
    ]
    out = agreement_tables(srows, {})
    check("the jump arm against the step arm", any("SAME 1/1" in l for l in out), True)
    # the arms' names pair them too (the shell's windows).
    named = [dict(r, arm=r["arm"] if r["kind"] == "clef" else ("jump" if r["arm"] == "jump" else "step"),
                  arm_kind=None) for r in srows]
    out = agreement_tables(named, {})
    check("the arms' names pair them", any("SAME 1/1" in l for l in out), True)

    # --- the arm's own top space (e2e.py's `top`, imported, not copied).
    check("the imported top of a noul", e2e.top({"type": "noul", "noul": 0.7})[0], "yes")
    check("the imported top of a choice", e2e.top({"type": "choice", "choice": "b", "probabilities": {"b": 0.6}})[0], "b")
    check("the suite table is e2e's", [s for s, _, _ in SUITES], ["en", "ko"])

    # --- the official rows are e2e's reader; `table` names their absence and shows no agreement against them.
    import contextlib
    import io
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        rows_path = Path(tmp) / "rows.json"
        rows_path.write_text(json.dumps({"rows": qrows}))
        empty = Path(tmp) / "no-data"
        empty.mkdir()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = table(["--data", str(empty), str(rows_path)])
        check("the table without the official rows", rc, 0)
        check("the table names the absent official rows",
              f"the official rows are not at {empty / SUITES[0][2]}, {empty / SUITES[1][2]}" in buf.getvalue(), True)
        check("the table shows no agreement against absent rows", "vs the official BF16 tops" in buf.getvalue(), False)
        for _, _, refp in SUITES:
            (empty / refp).parent.mkdir(parents=True)
            (empty / refp).write_text(json.dumps({"id": "r1", "response": refs[("en", "r1")]}) + "\n")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = table(["--data", str(empty), str(rows_path)])
        check("the table with the official rows read by e2e's reader", rc, 0)
        check("the table shows the agreement against them", "vs the official BF16 tops" in buf.getvalue(), True)

    # --- the cases of the hard suite, the gold and the lever-aware plan. Each runs guarded: a
    # case that cannot run (a function missing, a named refusal it did not expect) is its own FAIL line.
    import contextlib
    import io
    import tempfile

    def guarded(what: str, fn) -> None:
        try:
            fn()
        except (Exception, Stop) as e:  # noqa: BLE001 - a self-test case reports any way out
            bad(what, f"{type(e).__name__}: {e}")

    def refused(what: str, want: str, fn) -> None:
        """`fn` ends in a named refusal that holds `want`."""
        try:
            fn()
        except Stop as e:
            if want in str(e):
                ok(what)
            else:
                bad(what, f"{e} does not name {want!r}")
        except Exception as e:  # noqa: BLE001
            bad(what, f"{type(e).__name__}: {e}")
        else:
            bad(what, "it passed")

    # The fixture suite: two requests, a choice and a noul in the first, a score in the second.
    fix_rows = [
        {"id": "f-1", "state": "A customer wants the money back.",
         "questions": {"intent": {"type": "choice", "criteria": {"refund": "R", "cancel": "C", "other": "O"}},
                       "angry": {"type": "noul"}}},
        {"id": "f-2", "state": "Fine.", "questions": {"level": {"type": "score", "criteria": ["low", "mid", "high"]}}},
    ]
    fix_gold = [{"id": "f-1", "gold": {"intent": "refund", "angry": "true"}, "why": "x"},
                {"id": "f-2", "gold": {"level": "1"}, "why": "y"}]

    def write_suite(tmp: Path, rows, gold=None, name="suite-fix") -> str:
        path = tmp / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        if gold is not None:
            gold_path(path).write_text("".join(json.dumps(g) + "\n" for g in gold))
        return str(path)

    # --- suites by name and the gold beside them.
    def case_suite_names():
        check("a default suite keeps its name", suite_name("tools/ref/clef/suite.jsonl"), "en")
        check("the Korean suite keeps its name", suite_name("tools/ref/clef/suite-ko.jsonl"), "ko")
        check("a suite is named by its file", suite_name("tools/ref/clef/suite-hard.jsonl"), "hard")
        check("a file without the prefix is its stem", suite_name("/x/y/mine.jsonl"), "mine")
        check("no suite named loads the two default files", [s["name"] for s in load_suites()], ["en", "ko"])
        check("the gold file is beside the suite", gold_path(Path("a/suite-hard.jsonl")).name, "suite-hard.gold.jsonl")

    guarded("suites by name", case_suite_names)

    def case_suite_gold():
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            path = write_suite(t, fix_rows, fix_gold)
            suites = load_suites([path])
            check("a suite with a gold file loads it", suites[0]["gold"],
                  {"f-1": {"intent": "refund", "angry": "true"}, "f-2": {"level": "1"}})
            check("the suite is named by its file", suites[0]["name"], "fix")
            check("the suite's requests", [r["id"] for r in suites[0]["reqs"]], ["f-1", "f-2"])
            check("a suite with no gold file has none",
                  load_suites([write_suite(t, fix_rows, None, "suite-bare")])[0]["gold"], None)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = check_suites(["--suite", path])
            check("check-suites reads a suite and its gold", (rc, "fix (" in buf.getvalue(), "gold of 3 questions" in buf.getvalue()),
                  (0, True, True))
            # every way a gold file can disagree with its suite is a named error.
            refused("a gold line naming an id the suite lacks is refused", "the gold names the id 'f-9', which the suite g1 does not hold",
                    lambda: load_suites([write_suite(t, fix_rows, fix_gold + [{"id": "f-9", "gold": {"q": "a"}}], "suite-g1")]))
            refused("a gold line naming a question the request lacks is refused", "the question 'mood', which the request lacks",
                    lambda: load_suites([write_suite(t, fix_rows, [{"id": "f-1", "gold": {"intent": "refund", "mood": "x"}}, fix_gold[1]], "suite-g2")]))
            refused("a gold line naming a word that is no option is refused", "gives 'intent' the option 'tracking'",
                    lambda: load_suites([write_suite(t, fix_rows, [{"id": "f-1", "gold": {"intent": "tracking"}}, fix_gold[1]], "suite-g3")]))
            refused("a request with no gold line is refused", "no gold line for the request(s) ['f-2']",
                    lambda: load_suites([write_suite(t, fix_rows, fix_gold[:1], "suite-g4")]))
            refused("a request named twice in the gold is refused", "names the id 'f-1' twice",
                    lambda: load_suites([write_suite(t, fix_rows, fix_gold + fix_gold[:1], "suite-g5")]))
            refused("a missing suite file is refused", "no suite file at", lambda: load_suites([str(t / "nosuch.jsonl")]))
            refused("two files of one name are refused", "are named 'fix'",
                    lambda: load_suites([path, str(t / "suite-fix.jsonl")]))
            # a score and a noul may be written as a number and a boolean in the gold.
            loose = [{"id": "f-1", "gold": {"intent": "refund", "angry": True}}, {"id": "f-2", "gold": {"level": 2}}]
            check("a boolean and a number are the option ids they spell",
                  load_suites([write_suite(t, fix_rows, loose, "suite-loose")])[0]["gold"],
                  {"f-1": {"intent": "refund", "angry": "true"}, "f-2": {"level": "2"}})

    guarded("the gold beside a suite", case_suite_gold)

    def case_gold_notes():
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            src = {"url": "https://example.org/policy", "quote": "Refunds are paid within 14 days."}
            noted = [dict(fix_gold[0], source=src), dict(fix_gold[1], source=src)]
            check("a gold line's source is accepted and takes no part in the labels",
                  load_suites([write_suite(t, fix_rows, noted, "suite-src")])[0]["gold"],
                  {"f-1": {"intent": "refund", "angry": "true"}, "f-2": {"level": "1"}})
            check("a line with no why is as good",
                  load_suites([write_suite(t, fix_rows, [{"id": "f-1", "gold": fix_gold[0]["gold"]}, fix_gold[1]], "suite-nowhy")])[0]["gold"]["f-1"],
                  {"intent": "refund", "angry": "true"})
            refused("any other key on a gold line is refused", "has the key(s) ['note']",
                    lambda: load_suites([write_suite(t, fix_rows, [dict(fix_gold[0], note="x"), fix_gold[1]], "suite-key")]))
            refused("a source that is no object is refused", "the gold 'source' is",
                    lambda: load_suites([write_suite(t, fix_rows, [dict(fix_gold[0], source="http://x"), fix_gold[1]], "suite-srcstr")]))

    guarded("the gold's notes", case_gold_notes)

    def case_gold_yes_no():
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            yn = [{"id": "f-1", "gold": {"intent": "refund", "angry": "yes"}}, {"id": "f-2", "gold": {"level": "1"}}]
            check("a noul's label in Clef's top space is its option id",
                  load_suites([write_suite(t, fix_rows, yn, "suite-yn")])[0]["gold"]["f-1"], {"intent": "refund", "angry": "true"})
            no = [{"id": "f-1", "gold": {"angry": "no"}}, fix_gold[1]]
            check("and the other one", load_suites([write_suite(t, fix_rows, no, "suite-no")])[0]["gold"]["f-1"], {"angry": "false"})
            refused("a noul's label that is neither is refused", "gives 'angry' the option 'maybe'",
                    lambda: load_suites([write_suite(t, fix_rows, [{"id": "f-1", "gold": {"angry": "maybe"}}, fix_gold[1]], "suite-mb")]))
            refused("yes is no option of a choice", "gives 'intent' the option 'yes'",
                    lambda: load_suites([write_suite(t, fix_rows, [{"id": "f-1", "gold": {"intent": "yes"}}, fix_gold[1]], "suite-yc")]))

    guarded("a noul's gold in top space", case_gold_yes_no)

    def case_hard_suite():
        path = "tools/ref/clef/suite-hard.jsonl"
        if not (ROOT / path).is_file():
            print(f"note: {path} is not in this tree, so its own cases did not run")
            return
        (hard,) = load_suites([path])
        check("the hard suite is named by its file", hard["name"], "hard")
        check("the hard suite has a gold line for every request", sorted(hard["gold"]), sorted(r["id"] for r in hard["reqs"]))
        check("its gold is of every question it asks",
              all(sorted(hard["gold"][r["id"]]) == sorted(r["questions"]) for r in hard["reqs"]), True)
        for r in hard["reqs"]:
            if list(chat_body(r)["response_format"]["json_schema"]["schema"]["properties"]) != list(r["questions"]):
                bad(f"hard/{r['id']}", "the schema's properties are not the question ids in order")
        ok("every hard request converts to a schema of its question ids in order")

    guarded("the hard suite", case_hard_suite)

    # --- the gold scores every arm's picks, one answer a request.
    def case_gold_scores():
        def row(arm, kind, rid, picks, rnd=1, dropped=False):
            return {"arm": arm, "kind": kind, "suite": "fix", "id": rid, "round": rnd, "dropped": dropped,
                    "wall_ms": 10.0, "server_ms": 9.0, "host_ms": 1.0, "bucket": "short", "tops": {}, "picks": picks}

        good = {"f-1": {"intent": "refund", "angry": "true"}, "f-2": {"level": "1"}}
        wrong = {"f-1": {"intent": "cancel", "angry": "true"}, "f-2": {"level": "1"}}
        rows = []
        for rid in ("f-1", "f-2"):
            rows.append(row("G", "qwen", rid, good[rid]))
            rows.append(row("W", "qwen", rid, wrong[rid]))
            rows.append(row("W", "qwen", rid, good[rid], rnd=2))  # a later round does not change the answer
            rows.append(row("G", "qwen", rid, wrong[rid], rnd=0, dropped=True))
        meta = {"fix": {"path": "x", "gold": {"f-1": {"intent": "refund", "angry": "true"}, "f-2": {"level": "1"}}}}
        check("a right pick scores", gold_counts("G", rows, "fix", meta["fix"]["gold"])[:2], (3, 3))
        right, asked, bad_ones = gold_counts("W", rows, "fix", meta["fix"]["gold"])
        check("a wrong pick does not", (right, asked), (2, 3))
        check("the wrong pick is named", bad_ones, ["fix/f-1/intent (gold refund, ours cancel)"])
        check("an arm is scored on its own rows alone", gold_counts("G", rows, "fix", meta["fix"]["gold"])[2], [])
        lines = stats_table(rows, meta)
        check("each arm's suite line gains correct/N",
              [l for l in lines if l.startswith("arm G") and "suite fix" in l and "correct 3/3" in l] != [], True)
        check("the wrong arm's line", any(l.startswith("arm W") and "suite fix" in l and "correct 2/3" in l for l in lines), True)
        check("a bucket line carries no score", any("bucket short" in l and "correct" in l for l in lines), False)
        check("no gold, no score", any("correct" in l for l in stats_table(rows)), False)
        table_lines = gold_table(rows, meta)
        check("the gold table names the wrong pick",
              any(l.startswith("arm W (qwen) vs the gold of suite fix | 2/3") and "gold refund, ours cancel" in l for l in table_lines), True)
        check("the gold table lists the right arm clean", any(l.startswith("arm G (qwen)") and l.endswith("| 3/3 | wrong: —") for l in table_lines), True)
        check("no gold, no table", gold_table(rows, {"fix": {"path": "x", "gold": None}}), [])
        refused("a row with no picks cannot be scored", "has no picks",
                lambda: gold_counts("G", [{k: v for k, v in rows[0].items() if k != "picks"}], "fix", meta["fix"]["gold"]))
        refused("an arm with no row of a gold request is refused", "no counted row of fix/f-2",
                lambda: gold_counts("G", [rows[0]], "fix", meta["fix"]["gold"]))

    guarded("the gold scores the picks", case_gold_scores)

    # --- a row of each kind, from a canned server.
    class CannedClient:
        def __init__(self, replies):
            self.replies = replies
            self.sent = []

        def post(self, path, body):
            self.sent.append((path, body))
            if path == "/tokenize":
                return 200, json.dumps({"tokens": [ord(c) for c in body["content"]]}), 1.0
            status, resp = self.replies[path]
            return status, json.dumps(resp), 12.5

    def case_server_error():
        # a server that does not answer is an error, never a row.
        arm = Arm("q", "http://127.0.0.1:1", "qwen")
        arm.client = CannedClient({"/v1/chat/completions": (500, {"error": "x"})})
        refused("a server error is no scored answer", "the server answered 500", lambda: post_one(arm, "fix", fix_rows[0], 1, False, None))
        refused("a --chat flag is no argument any more", "'--chat' is not an argument", lambda: parse_run_args(["--chat", "c=http://x"]))

    guarded("a server error", case_server_error)

    # --- the lever-aware call plan.
    ko06 = {"id": "ko-06", "state": "s",
            "questions": {"intent": {"type": "choice", "criteria": {"refund": "R", "tracking": "T", "cancel": "C", "other": "O"}},
                          "angry": {"type": "noul"}}}
    # ko-06's ids as a real vocabulary cuts them: the opening `{"intent": "` 4 ids, the four words a single id, the
    # closing `", "angry": ` 3 ids, `true` and `false` a single id, `}` one.
    ko06_ids = [("F", [1, 2, 3, 4]), ("P", [[10], [11], [12], [13]]), ("F", [20, 21, 22]), ("P", [[30], [31]]), ("F", [40])]
    # a request with a score question: the opening 4 ids, a 2-id run before the score, a 1-id run before the noul.
    sc_ids = [("F", [1, 2, 3, 4]), ("P", [[10], [11], [12], [13]]), ("F", [20, 21]), ("P", [[30], [31], [32]]),
              ("F", [40]), ("P", [[50], [51]]), ("F", [60])]

    def case_levers():
        toks = stream(ko06_ids, [0, 0])
        check("the reference counts of ko-06 after the levers",
              {k: v for k, v in plan_levers(toks).items() if k != "span_ids"},
              {"next_calls": 0, "prefill_calls": 0, "next_after_calls": 1, "lead": 4, "spans": 1, "tail": 2})
        check("ko-06's span feeds the decision's id and the run", plan_levers(toks)["span_ids"], 4)
        check("ko-06's other picks make the same counts", plan_levers(stream(ko06_ids, [3, 1])), plan_levers(toks))
        # the score request: a 2-id run is fed, a 1-id run is stepped (the decision's step and the fixed id's).
        toks = stream(sc_ids, [0, 1, 0])
        check("a request with a score question",
              plan_levers(toks),
              {"next_calls": 2, "prefill_calls": 0, "next_after_calls": 1, "lead": 4, "spans": 1, "span_ids": 3, "tail": 2})
        # every id a decision: nothing is fed, every pick is stepped.
        check("two decisions in a row are one step",
              plan_levers(stream([("F", [1]), ("P", [[10, 11], [10, 12], [20, 21]]), ("F", [9])], [0])),
              {"next_calls": 1, "prefill_calls": 0, "next_after_calls": 0, "lead": 1, "spans": 0, "span_ids": 0, "tail": 2})
        # no decision at all: the first id is the prompt's row's, the rest is the tail, the stop with it.
        check("a program that decides nothing is all tail but its first id",
              plan_levers(stream([("F", [1, 2, 3]), ("P", [[7]]), ("F", [9])], [0])),
              {"next_calls": 0, "prefill_calls": 0, "next_after_calls": 0, "lead": 0, "spans": 0, "span_ids": 0, "tail": 5})
        check("the step arm steps every id", step_levers(stream(ko06_ids, [0, 0])),
              {"next_calls": 10, "prefill_calls": 0, "next_after_calls": 0, "lead": 0, "spans": 0, "span_ids": 0, "tail": 0})

    guarded("the lever-aware plan", case_levers)

    def case_lever_check():
        class IdsArm:
            name = "fake"

            def program_ids(self, suite, req):
                return ko06_ids

        ref = {"arm": "jump", "next_calls": 0, "prefill_calls": 0, "next_after_calls": 1, "lead": 4, "spans": 1, "tail": 2}
        ans = {"intent": "cancel", "angry": True}
        check("a lever server's reference counts are held", check_counts(IdsArm(), "en", ko06, ans, ref), "the jump levers")
        for key, val in (("tail", 3), ("lead", 0), ("spans", 2), ("next_after_calls", 0), ("next_calls", 1), ("prefill_calls", 1)):
            refused(f"a lever server's {key} is held", f"the server reports {key} {val}", lambda key=key, val=val:
                    check_counts(IdsArm(), "en", ko06, ans, dict(ref, **{key: val})))
        step = {"arm": "step", "next_calls": 10, "prefill_calls": 0, "next_after_calls": 0, "lead": 0, "spans": 0, "tail": 0}
        check("a lever server's step arm is held", check_counts(IdsArm(), "en", ko06, ans, step), "the jump levers")
        refused("a lever server's step arm that took the jump's tail is refused", "the server reports tail 2",
                lambda: check_counts(IdsArm(), "en", ko06, ans, dict(step, tail=2)))
        # the old keys alone get the old plan, whatever the new plan says.
        want_next, want_prefill = plan_counts(plan_jump(stream(ko06_ids, [0, 0])))
        old = {"arm": "jump", "next_calls": want_next, "prefill_calls": want_prefill}
        check("a server with the old keys gets the old plan", check_counts(IdsArm(), "en", ko06, ans, old), "after the prompt")
        refused("and is held to it", f"the server reports next_calls 0, the program makes {want_next}",
                lambda: check_counts(IdsArm(), "en", ko06, ans, dict(old, next_calls=0)))

    guarded("the lever check", case_lever_check)

    def case_lever_row():
        # the whole path with a byte tokenizer: ko-06's ids are one a byte, so the counts are the byte program's.
        arm = Arm("jump", "http://127.0.0.1:1", "qwen")
        constrained = {"arm": "jump", "spans": 1, "span_ids": 18, "decisions": 2, "free_steps": 0, "next_calls": 0,
                       "prefill_calls": 0, "next_after_calls": 1, "lead": 12, "tail": 5}
        resp = {"choices": [{"message": {"role": "assistant", "content": '{"intent": "cancel", "angry": true}'},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 36},
                "timings": {"prompt_n": 50, "prompt_ms": 30.0, "predicted_ms": 5.0, "constrained": constrained}}
        arm.client = CannedClient({"/v1/chat/completions": (200, resp)})
        r = post_one(arm, "en", ko06, 1, False, None)
        check("a lever row keeps the server's kind and the convention", (r["arm_kind"], r["convention"]), ("jump", "the jump levers"))
        check("a lever row keeps every count", (r["calls"]["lead"], r["calls"]["tail"], r["calls"]["next_after_calls"]), (12, 5, 1))
        check("a lever row's picks", r["picks"], {"intent": "cancel", "angry": "true"})
        arm = Arm("jump", "http://127.0.0.1:1", "qwen")
        arm.client = CannedClient({"/v1/chat/completions": (200, dict(resp, timings=dict(resp["timings"], constrained=dict(constrained, tail=4))))})
        refused("a lever row with a count off is refused by name", "the server reports tail 4", lambda: post_one(arm, "en", ko06, 1, False, None))

    guarded("a lever row", case_lever_row)

    # --- the pairs: a jump arm pairs with the step arm of its seat.
    def case_pairs():
        def r(arm, kind, arm_kind, run):
            return {"arm": arm, "kind": kind, "arm_kind": arm_kind, "run": run, "suite": "s", "id": "x"}

        rows = [r("A", "clef", None, 0), r("jump", "qwen", "jump", 0),
                r("A", "clef", None, 1), r("step", "qwen", "step", 1),
                r("ds41-jump", "qwen", "jump", 2)]
        check("the jump arm pairs with the step arm of another window, the V4.1 jump arm with none yet",
              jump_step_pairs(rows), [("jump", "step")])
        check("a V4.1 jump arm pairs with its own step arm", jump_step_pairs(rows + [r("ds41-step", "qwen", "step", 3)]),
              [("jump", "step"), ("ds41-jump", "ds41-step")])
        check("arms of no role are paired in the runs' order", jump_step_pairs([r("B", "qwen", None, 0), r("C", "qwen", None, 1)]),
              [("B", "C")])

    guarded("the arm pairs", case_pairs)

    # --- the jump arm against the step arm, beyond the picks: text, finish reason, usage, and the stop's class.
    def case_exactness():
        def row(arm, kind, text, finish="stop", usage=None, ids=None, stops=None):
            return {"arm": arm, "kind": "qwen", "arm_kind": kind, "suite": "en", "id": "r1", "round": 1, "dropped": False,
                    "wall_ms": 1.0, "server_ms": 1.0, "host_ms": 0.0, "bucket": "short", "tops": {"q": "a"},
                    "text": text, "finish_reason": finish, "usage": usage or {"completion_tokens": 7},
                    "ids": ids, "stop_ids": stops}

        same = [row("jump", "jump", "{}"), row("step", "step", "{}")]
        line = next(l for l in agreement_tables(same, {}) if l.startswith("jump vs step"))
        check("a jump and a step answer that agree to the byte", line, "jump vs step | SAME 1/1 | TEXT 1/1 | FINISH 1/1 | USAGE 1/1")
        differ = [row("jump", "jump", "{}"), row("step", "step", "{ }", finish="length", usage={"completion_tokens": 8})]
        line = next(l for l in agreement_tables(differ, {}) if l.startswith("jump vs step"))
        check("text, finish reason and usage are each held exact", line,
              "jump vs step | SAME 1/1 | TEXT 0/1 | FINISH 0/1 | USAGE 0/1")
        with_ids = [row("jump", "jump", "{}", ids=[5, 6, 2], stops=[2, 3]), row("step", "step", "{}", ids=[5, 6, 3])]
        line = next(l for l in agreement_tables(with_ids, {}) if l.startswith("jump vs step"))
        check("ids are held but for the stop's class", line.endswith("| IDS 1/1"), True)
        check("the same ids", stop_class_same([1, 2, 3], [1, 2, 3], [3]), True)
        check("a stop id for another stop id", stop_class_same([1, 2, 3], [1, 2, 9], [3, 9]), True)
        check("a changed id before the stop", stop_class_same([1, 2, 3], [1, 5, 9], [3, 9]), False)
        check("a last id that is no stop", stop_class_same([1, 2, 3], [1, 2, 9], [3]), False)
        check("a different length", stop_class_same([1, 2, 3], [1, 2], [3]), False)

    guarded("the exact compare", case_exactness)

    # --- the table of a sitting of the hard suite: the gold in the rows' file, the lines, no official rows needed.
    def case_table_gold():
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)

            def row(arm, kind, rid, picks):
                return {"arm": arm, "kind": kind, "arm_kind": None, "suite": "fix", "id": rid, "round": 1, "dropped": False,
                        "wall_ms": 5.0, "server_ms": 4.0, "host_ms": 1.0, "bucket": "short", "tops": {}, "picks": picks}

            rows = [row("A", "clef", "f-1", {"intent": "refund", "angry": "true"}), row("A", "clef", "f-2", {"level": "0"}),
                    row("d", "qwen", "f-1", {"intent": "refund", "angry": "true"}), row("d", "qwen", "f-2", {"level": "1"})]
            body = {"arms": ["A", "d"], "rows": rows,
                    "suites": {"fix": {"path": "x", "gold": {"f-1": {"intent": "refund", "angry": "true"}, "f-2": {"level": "1"}}}}}
            (t / "rows.json").write_text(json.dumps(body))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = table(["--data", str(t / "no-data"), str(t / "rows.json")])
            out = buf.getvalue()
            check("the table of a hard-suite sitting", rc, 0)
            check("it needs no official rows", "the official rows are not at" in out, False)
            check("it scores the Clef arm", "arm A (clef) | suite fix" in out and "correct 2/3" in out, True)
            check("it scores a generative arm", any(l.startswith("arm d (qwen) | suite fix") and l.endswith("correct 3/3")
                                                    for l in out.splitlines()), True)
            check("it lists the wrong picks", "arm A (clef) vs the gold of suite fix | 2/3 | wrong: fix/f-2/level (gold 1, ours 0)" in out, True)

    guarded("the table with gold", case_table_gold)

    # --- the bodies: servers that log every request, in one list shared by the arms, in the order they came.
    class RecClient:
        """A server for `measure`: the answers of the real ones in their own text (indented, a trailing newline,
        the request's nth answer), every request logged. `props` is the `/props` object (None: a 404);
        `systemone` the status the SystemOne route answers a generative arm; `die_after` the SystemOne answers
        before the server stops answering."""

        def __init__(self, name, log, props=None, systemone=200, die_after=None):
            self.name, self.log, self.props, self.systemone, self.die_after = name, log, props, systemone, die_after
            self.nth: dict = {}
            self.answers: dict = {}
            self.passed = 0

        def get(self, path):
            self.log.append((self.name, "GET", path))
            if self.props is None:
                return 404, "not found\n"
            return 200, json.dumps(self.props)

        def close(self):
            pass

        def post(self, path, body):
            self.log.append((self.name, "POST", path))
            timed = path == "/v1/chat/completions" or (path == "/v1/systemone" and not self.log_pass_started())
            if path == "/v1/systemone" and not timed:
                if self.die_after is not None and self.passed >= self.die_after:
                    fail("the server stopped answering")
                self.passed += 1
                if self.systemone != 200:
                    return self.systemone, "no such route\n", 1.0
            key = (path, body["messages"][1]["content"] if path == "/v1/chat/completions" else body["state"])
            self.nth[key] = self.nth.get(key, 0) + 1
            if path == "/v1/chat/completions":
                props = body["response_format"]["json_schema"]["schema"]["properties"]
                obj = {qid: (True if p["type"] == "boolean" else p["enum"][0]) for qid, p in props.items()}
                resp = {"nth": self.nth[key], "choices": [{"message": {"role": "assistant", "content": json.dumps(obj)},
                                                           "finish_reason": "stop"}],
                        "timings": {"prompt_n": 1500 if "Fine." in key[1] else 260, "prompt_ms": 50.0, "predicted_ms": 25.0}}
            else:
                answers = {}
                for qid, q in body["questions"].items():
                    if q["type"] == "choice":
                        keys = sorted(q["criteria"])
                        answers[qid] = {"type": "choice", "choice": keys[0], "probabilities": {k: 0.5 for k in keys}}
                    elif q["type"] == "score":
                        answers[qid] = {"type": "score", "probabilities": {str(i): 0.5 for i in range(len(q["criteria"]))}}
                    else:
                        answers[qid] = {"type": "noul", "noul": 0.9}
                resp = {"nth": self.nth[key], "model": "clef", "answers": answers,
                        "timings": {"prompt_n": 1500 if "Fine." in key[1] else 300, "prompt_ms": 70.0, "head_ms": 20.0}}
            raw = json.dumps(resp, indent=3, ensure_ascii=False) + "\n"
            self.answers.setdefault(key, []).append(raw)
            return 200, raw, 5.0

        def log_pass_started(self):
            # the arm's pass began when it read /props: every systemone POST after that GET is the pass's.
            return any(e[:3] == (self.name, "GET", "/props") for e in self.log)

    def measure_fix(tmp, arms_spec, rounds=2, rounds_long=1):
        """`measure` over the fixture suite on RecClient arms: arms_spec is [(name, kind, RecClient kwargs)]."""
        log: list = []
        arms = []
        for name, kind, kw in arms_spec:
            arm = Arm(name, "http://127.0.0.1:1", kind)
            arm.client = RecClient(name, log, **kw)
            arms.append(arm)
        suites = [{"name": "fix", "path": "x", "reqs": fix_rows, "gold": None}]
        opts = {"rounds": rounds, "rounds_long": rounds_long, "out": str(tmp / "out"), "model": None}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = measure(arms, suites, opts)
        return tmp / "out", log, arms, buf.getvalue(), rc

    def case_bodies_timed():
        with tempfile.TemporaryDirectory() as tmp:
            out, log, arms, text, rc = measure_fix(Path(tmp), [("A", "clef", {}), ("jump", "qwen", {})])
            check("a run with bodies ends clean", rc, 0)
            data = json.loads((out / "rows.json").read_text())
            rows = data["rows"]
            check("a row holds no raw answer", any("_raw" in r for r in rows), False)
            for arm in arms:
                for rid, state in (("f-1", fix_rows[0]["state"]), ("f-2", fix_rows[1]["state"])):
                    answers = next(v for k, v in arm.client.answers.items() if state in k[1] or k[1] == state)
                    path = out / "bodies" / arm.name / f"fix-{rid}.json"
                    got = path.read_text() if path.exists() else None
                    # the 1st answer is the dropped round's, the 2nd the first counted round's: the file is the 2nd,
                    # byte for byte.
                    check(f"the first counted round's body of {arm.name}/{rid} is written as received", got, answers[1])
                    check(f"it is not the dropped round's of {arm.name}/{rid}", got != answers[0], True)
                    mine = [r for r in rows if r["arm"] == arm.name and r["id"] == rid]
                    named = [r for r in mine if "body" in r]
                    check(f"one row of {arm.name}/{rid} names a body", len(named), 1)
                    check(f"it is the first counted round's of {arm.name}/{rid}",
                          (named[0]["dropped"], named[0]["round"], named[0]["body"]),
                          (False, min(r["round"] for r in mine if not r["dropped"]), f"bodies/{arm.name}/fix-{rid}.json"))
                    check(f"the dropped row of {arm.name}/{rid} names none", any("body" in r for r in mine if r["dropped"]), False)
            check("no more files than requests and arms", sorted(p.name for p in (out / "bodies").rglob("*.json")),
                  ["fix-f-1.json", "fix-f-1.json", "fix-f-2.json", "fix-f-2.json"])
            check("the timed bodies are counted in one line", "[bodies] timed: 4 bodies of 2 arms under " in text, True)

    guarded("the timed bodies", case_bodies_timed)

    def pass_start(log):
        return next(i for i, e in enumerate(log) if e[1:] == ("GET", "/props"))

    def case_bodies_pass():
        props = {"routes": ["/v1/chat/completions", "/v1/systemone"]}
        with tempfile.TemporaryDirectory() as tmp:
            out, log, arms, text, rc = measure_fix(Path(tmp), [("A", "clef", {}), ("jump", "qwen", {"props": props})])
            first = pass_start(log)
            tail = log[first:]
            check("the pass starts with /props", tail[0], ("jump", "GET", "/props"))
            check("only the pass's requests follow it", set(tail[1:]), {("jump", "POST", "/v1/systemone")})
            check("after the rounds: every timed request is before the first /props",
                  sum(1 for e in log[:first] if e[1] == "POST"), 10)
            check("the arm's systemone POSTs are all after it", [i for i, e in enumerate(log[:first]) if e == ("jump", "POST", "/v1/systemone")], [])
            check("the Clef arm is not asked for /props", [e for e in log if e[0] == "A" and e[1] == "GET"], [])
            check("the pass posts each request once", len(tail) - 1, 2)
            answers = [v[0] for k, v in arms[1].client.answers.items() if k[0] == "/v1/systemone"]
            got = sorted(p.read_text() for p in (out / "bodies" / "jump-systemone").glob("*.json"))
            check("the systemone bodies are saved as received", got, sorted(answers))
            check("the systemone files are named by suite and id",
                  sorted(p.name for p in (out / "bodies" / "jump-systemone").iterdir()), ["fix-f-1.json", "fix-f-2.json"])
            check("the Clef arm has no systemone directory", (out / "bodies" / "A-systemone").exists(), False)
            check("the summary line", "[bodies] jump: systemone 2 ok / 2" in text, True)
            check("the pass is after the rows", text.index("[rows]") < text.index("[bodies] jump:"), True)
            check("the rows hold no systemone body", any("systemone" in json.dumps(r) for r in json.loads((out / "rows.json").read_text())["rows"]), False)
        for props, line in (
            ({"routes": ["/v1/chat/completions"]}, "[bodies] jump: the server lists no /v1/systemone; skipped\n"),
            ({"build_info": "x"}, "[bodies] jump: the server lists no /v1/systemone; skipped\n"),
            (None, "[bodies] jump: the server lists no /v1/systemone; skipped (GET /props answered 404)\n"),
            ({"routes": "/v1/systemone"}, "[bodies] jump: the server lists no /v1/systemone; skipped (GET /props has no list of paths under routes)\n"),
        ):
            with tempfile.TemporaryDirectory() as tmp:
                out, log, arms, text, rc = measure_fix(Path(tmp), [("A", "clef", {}), ("jump", "qwen", {"props": props})])
                check(f"props {props}: skipped by name, not an error", (rc, line in text), (0, True))
                check(f"props {props}: no systemone POST of the generative arm",
                      [e for e in log if e == ("jump", "POST", "/v1/systemone")], [])
                check(f"props {props}: no systemone directory", (out / "bodies" / "jump-systemone").exists(), False)

    guarded("the untimed SystemOne pass", case_bodies_pass)

    def case_bodies_pass_failures():
        props = {"routes": ["/v1/systemone"]}
        with tempfile.TemporaryDirectory() as tmp:
            out, log, arms, text, rc = measure_fix(Path(tmp), [("A", "clef", {}), ("jump", "qwen", {"props": props, "systemone": 404})])
            check("a 404 is not raised", rc, 0)
            d = out / "bodies" / "jump-systemone"
            check("a 404's body is saved", sorted(p.name for p in d.glob("*.json")), ["fix-f-1.json", "fix-f-2.json"])
            check("a 404's body is as received", (d / "fix-f-1.json").read_text(), "no such route\n")
            check("a 404's status is in its side file", (d / "fix-f-1.json.status").read_text(), "404\n")
            check("the side files are the non-200s", sorted(p.name for p in d.glob("*.status")), ["fix-f-1.json.status", "fix-f-2.json.status"])
            check("a 404 is counted", "[bodies] jump: systemone 0 ok / 2" in text, True)
            check("a 404 is named", "[bodies] jump: fix/f-1: /v1/systemone answered 404" in text, True)
            check("the rows are written", (out / "rows.json").is_file(), True)
        with tempfile.TemporaryDirectory() as tmp:
            out, log, arms, text, rc = measure_fix(Path(tmp), [("A", "clef", {}), ("jump", "qwen", {"props": props, "die_after": 1})])
            check("a server that stops answering ends the pass, not the run", rc, 0)
            check("and the pass says where", "[bodies] jump: fix/f-2: " in text and "the pass ends here" in text, True)
            check("and counts what it got", "[bodies] jump: systemone 1 ok / 2" in text, True)
            check("a body written before it stays", sorted(p.name for p in (out / "bodies" / "jump-systemone").iterdir()), ["fix-f-1.json"])
            check("the timed rows and bodies are intact", ((out / "rows.json").is_file(), len(list((out / "bodies" / "jump").glob("*.json")))), (True, 2))
        with tempfile.TemporaryDirectory() as tmp:
            out, log, arms, text, rc = measure_fix(Path(tmp), [("jump", "qwen", {"props": props}), ("step", "qwen", {"props": props})])
            check("each generative arm has its own pass line",
                  ("[bodies] jump: systemone 2 ok / 2" in text, "[bodies] step: systemone 2 ok / 2" in text), (True, True))

    guarded("the SystemOne pass's failures", case_bodies_pass_failures)

    def case_bodies_names():
        def arm(name, kind="qwen"):
            return Arm(name, "http://127.0.0.1:1", kind)

        reqs = [("fix", fix_rows[0])]
        refused("an arm named with a slash is refused", "is no plain file name", lambda: check_body_names([arm("a/b")], reqs))
        refused("an arm named .. is refused", "is no plain file name", lambda: check_body_names([arm("..")], reqs))
        refused("an arm named like another's systemone directory is refused", "named twice",
                lambda: check_body_names([arm("x"), arm("x-systemone")], reqs))
        check("the Clef arm has no systemone directory to collide with", check_body_names([arm("x", "clef"), arm("x-systemone")], reqs), None)
        refused("a request id with a slash is refused", "is no plain file name",
                lambda: check_body_names([arm("a")], [("fix", dict(fix_rows[0], id="a/b"))]))
        refused("two requests that share a file name are refused", "share a body file name",
                lambda: check_body_names([arm("a")], [("a", dict(fix_rows[0], id="b-c")), ("a-b", dict(fix_rows[0], id="c"))]))
        # the bodies' directories are made before the first POST: a run that cannot write them posts nothing.
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "out").write_text("a file where the out dir should be")
            log: list = []
            a = arm("a")
            a.client = RecClient("a", log)
            refused("an out dir that cannot be made ends the run by name", "cannot make",
                    lambda: measure(
                        [a], [{"name": "fix", "path": "x", "reqs": fix_rows, "gold": None}],
                        {"rounds": 1, "rounds_long": 1, "out": str(Path(tmp) / "out"), "model": None}))
            check("and it posted nothing", log, [])

    guarded("the bodies' names", case_bodies_names)

    if fails:
        print(f"sjcmp self-test: {fails} of {n} checks FAILED")
        raise SystemExit(1)
    print(f"sjcmp self-test: {n} checks ok")


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        return 0
    if len(sys.argv) >= 2 and sys.argv[1] == "run":
        return run(sys.argv[2:])
    if len(sys.argv) >= 3 and sys.argv[1] == "table":
        return table(sys.argv[2:])
    if len(sys.argv) >= 2 and sys.argv[1] == "check-suites":
        return check_suites(sys.argv[2:])
    if len(sys.argv) >= 2 and sys.argv[1] == "convert":
        return convert(sys.argv[2:])
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
