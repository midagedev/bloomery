#!/usr/bin/env python3
"""lev's /v1/systemone oracle: llama.cpp mainline's server answering the suite, with every prompt's ids.

    tools/ref/lev_ref.py run --model FILE --suite FILE --edge FILE --out DIR --tree DIR [--port 8092]
    tools/ref/lev_ref.py table DIR|FILE
    tools/ref/lev_ref.py --self-test

ik_llama.cpp has no decision server, so the oracle is mainline's `llama-server` at the commit that added it
(`LEV_LCPP_COMMIT` of tools/ref/models/qwen35.sh, PR #29818) built by tools/ref/build-lcpp-lev.sh with
tools/ref/lev/lcpp-ids.patch: a logging-only patch that makes the server print, once per decision task, a
`bloomery-ids n <N>: <id> <id> ...` line (the server's own `prompt token` debug block is commented out). The
server itself is the oracle for every number: the prompt the template renders, its tokens, the labels, the
temperatures, the softmax and the body. `run` refuses a tree that is not that commit with exactly that patch
applied, and records the pair as `lcpp_build` in every row; the gate (crates/decision/tests/lev.rs) refuses a
dump whose `lcpp_build` is not the one it pins.

`run` also writes `<out>/labels.json`: the label codes (`A`..`Z`, then `AA`..`ZZ`, the first 255 that the server's
own /tokenize, with no special tokens added or parsed, gives as one token) with their ids, the list
`server-decision.cpp` builds at start (`labels`, `label_texts`).

`run` starts the server on FILE (`-np 1`, every layer on the card in view: run it under `BLOOMERY_CARD=a6000
tools/box.sh`, as tools/ref/clef_ref.py is; the server is this script's child, stopped by its own pid, written
to `<out>/server.pid`), POSTs each request of the suite and the edge file to /v1/systemone as written (the line
from `"model"` on: the `id` is ours, the server would ignore it), and writes `<out>/reference.jsonl`, one line a
request: `id`, `lcpp_build`, `tasks` (one entry a prompt, in the server's order: question id, variant, `ids`),
`response` (the body, keys in the server's order, `timings` left out) and `wall_ms`. The tasks of a request are
its questions in order, each question twice when it is a choice of two options or more (lev shows those options
in two orders, `server-decision.cpp` n_variants) and once otherwise. The prompts' ids are taken from the log by
task id, and `usage.input_tokens` must equal their count: a request where it does not, a log with another number
of tasks than the plan, or a body that is not 200, stops the run by name.

`table` prints one Markdown row a question of a dump: its top option, that option's probability and the margin
to the runner-up (the top of a noul is yes when its p >= 0.5).
"""

from __future__ import annotations

import hashlib
import json
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PATCH = HERE / "lev" / "lcpp-ids.patch"
LOG_LINE = re.compile(r"task (\d+) \| bloomery-ids n (\d+): ([0-9 ]*)$")


# ---- pure helpers (the self-test pins them) ----

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise SystemExit(f"lev_ref: {path}:{number} is an empty line")
        rows.append(json.loads(line))
    ids = [row.get("id") for row in rows]
    if any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(ids):
        raise SystemExit(f"lev_ref: {path} needs a unique string id on every row, got {ids}")
    return rows


def request_text(line: str) -> str:
    """The request as the server is given it: the line from its `"model"` key on, which drops the leading `id`."""
    at = line.find('"model"')
    if at < 0:
        raise SystemExit(f"lev_ref: a request line holds no \"model\" key: {line[:60]!r}")
    return "{" + line[at:]


def n_variants(question: dict[str, Any]) -> int:
    """lev shows the options of a choice of two or more in two orders (server-decision.cpp n_variants)."""
    if question.get("type") == "choice" and len(question.get("criteria") or {}) > 1:
        return 2
    return 1


def plan(request: dict[str, Any]) -> list[tuple[str, int]]:
    """The (question id, variant) of every prompt the server builds for a request, in its order."""
    return [(qid, v) for qid, q in request["questions"].items() for v in range(n_variants(q))]


MAX_LABELS = 255


def label_codes() -> list[str]:
    """`A`..`Z` then `AA`..`ZZ`, the codes server-decision.cpp draws the labels from."""
    letters = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    return letters + [a + b for a in letters for b in letters]


def pick_labels(tokenize: Any) -> list[dict[str, Any]]:
    """The first MAX_LABELS codes `tokenize(code)` gives as exactly one token, each with its id."""
    out: list[dict[str, Any]] = []
    for code in label_codes():
        tokens = tokenize(code)
        if len(tokens) == 1 and len(out) < MAX_LABELS:
            out.append({"code": code, "id": tokens[0]})
    return out


def scrape(log: str) -> list[tuple[int, list[int]]]:
    """The (task id, ids) of every `bloomery-ids` line of a server log, in task id order. A line whose count is
    not the number of ids it lists is a refusal."""
    out = []
    for line in log.splitlines():
        m = LOG_LINE.search(line.rstrip())
        if not m:
            continue
        ids = [int(x) for x in m.group(3).split()]
        if len(ids) != int(m.group(2)):
            raise SystemExit(f"lev_ref: task {m.group(1)} claims {m.group(2)} ids and lists {len(ids)}")
        out.append((int(m.group(1)), ids))
    out.sort(key=lambda t: t[0])
    return out


def top(answer: dict[str, Any]) -> tuple[str, float, float]:
    """A question's top option, its probability and the margin to the runner-up. A noul's options are yes
    (p = `noul`) and no (1 - p)."""
    if answer["type"] == "noul":
        p = answer["noul"]
        return ("yes" if p >= 0.5 else "no"), max(p, 1 - p), abs(2 * p - 1)
    probs = answer["probabilities"]
    ranked = sorted(probs.items(), key=lambda kv: -kv[1])
    return ranked[0][0], ranked[0][1], ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.0)


# ---- the box side ----

def git(tree: Path, *args: str) -> str:
    done = subprocess.run(["git", "-c", f"safe.directory={tree}", "-C", str(tree), *args],
                          capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f"lev_ref: git {' '.join(args)} in {tree}: {done.stderr.strip()}")
    return done.stdout


def lcpp_build(tree: Path, commit: str) -> str:
    """`<commit>+ids-patch:<sha256 of the patch>` of a tree that is exactly that commit with the patch applied."""
    head = git(tree, "rev-parse", "HEAD").strip()
    if head != commit:
        raise SystemExit(f"lev_ref: {tree} is at {head}, the oracle is {commit}")
    patch = PATCH.read_bytes()
    want = sorted(re.findall(r"^\+\+\+ b/(\S+)", patch.decode(), re.M))
    changed = sorted(git(tree, "diff", "--name-only").split())
    if changed != want:
        raise SystemExit(f"lev_ref: {tree} has tracked changes {changed}, the patch's are {want}")
    done = subprocess.run(["git", "-c", f"safe.directory={tree}", "-C", str(tree), "apply", "--check", "--reverse", "-"],
                          input=patch, capture_output=True)
    if done.returncode != 0:
        raise SystemExit(f"lev_ref: {tree}'s changes are not {PATCH.name}: {done.stderr.decode().strip()}")
    return f"{commit}+ids-patch:{hashlib.sha256(patch).hexdigest()}"


def post(url: str, text: str) -> tuple[int, str]:
    req = urllib.request.Request(url, data=text.encode("utf-8"), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


def tokenize(port: int, text: str) -> list[int]:
    """The server's /tokenize with no special tokens added and none parsed (`common_tokenize(vocab, text, false,
    false)`)."""
    body = json.dumps({"content": text, "add_special": False, "parse_special": False})
    status, answer = post(f"http://127.0.0.1:{port}/tokenize", body)
    if status != 200:
        raise SystemExit(f"lev_ref: /tokenize {text!r}: HTTP {status}: {answer[:200]}")
    return json.loads(answer)["tokens"]


def wait_health(port: int, proc: subprocess.Popen, within: float) -> None:
    end = time.monotonic() + within
    while time.monotonic() < end:
        if proc.poll() is not None:
            raise SystemExit(f"lev_ref: the server exited ({proc.returncode}) before it listened")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(1)
    raise SystemExit(f"lev_ref: the server did not listen within {within:.0f} s")


def run(args: Any) -> int:
    commit = None
    for line in (HERE / "models" / "qwen35.sh").read_text().splitlines():
        if line.startswith("LEV_LCPP_COMMIT="):
            commit = line.split("=", 1)[1].strip()
    if not commit:
        raise SystemExit("lev_ref: models/qwen35.sh names no LEV_LCPP_COMMIT")
    tree = Path(args.tree)
    build = lcpp_build(tree, commit)
    server = tree / "build" / "bin" / "llama-server"
    if not server.is_file():
        raise SystemExit(f"lev_ref: no {server} (build it: tools/ref/build-lcpp-lev.sh)")
    model = Path(args.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(Path(args.suite)) + read_jsonl(Path(args.edge))
    lines = {json.loads(line)["id"]: line
             for path in (args.suite, args.edge) for line in Path(path).read_text(encoding="utf-8").splitlines()}
    log_path = out / "server.log"
    cmd = [str(server), "-m", str(model), "--host", "127.0.0.1", "--port", str(args.port), "-np", "1",
           "-c", "16384", "-ngl", "99", "-a", model.name, "--no-ui"]
    print(f"lev_ref: {' '.join(cmd)}", flush=True)
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        (out / "server.pid").write_text(f"{proc.pid}\n")
        try:
            wait_health(args.port, proc, 600)
            labels = pick_labels(lambda code: tokenize(args.port, code))
            (out / "labels.json").write_text(json.dumps({"labels": labels}) + "\n")
            print(f"lev_ref: {len(labels)} labels, {labels[0]} .. {labels[-1]}", flush=True)
            reference = []
            for request in rows:
                rid = request["id"]
                tasks = plan(request)
                before = len(scrape(log_path.read_text()))
                t = time.perf_counter()
                status, text = post(f"http://127.0.0.1:{args.port}/v1/systemone", request_text(lines[rid]))
                wall = (time.perf_counter() - t) * 1e3
                if status != 200:
                    raise SystemExit(f"lev_ref: {rid}: HTTP {status}: {text[:300]}")
                body = json.loads(text)
                # the log is written by the server's own thread: give the last line a moment
                deadline = time.monotonic() + 10
                while True:
                    seen = scrape(log_path.read_text())[before:]
                    if len(seen) >= len(tasks) or time.monotonic() > deadline:
                        break
                    time.sleep(0.2)
                if len(seen) != len(tasks):
                    raise SystemExit(f"lev_ref: {rid}: the log holds {len(seen)} prompts, the plan {len(tasks)}")
                total = sum(len(ids) for _, ids in seen)
                if body["usage"]["input_tokens"] != total:
                    raise SystemExit(f"lev_ref: {rid}: usage.input_tokens {body['usage']['input_tokens']} != "
                                     f"{total} ids of the log's {len(seen)} prompts")
                if list(body["answers"]) != list(request["questions"]):
                    raise SystemExit(f"lev_ref: {rid}: answers {list(body['answers'])} do not match the questions")
                body.pop("timings", None)
                reference.append({
                    "id": rid,
                    "lcpp_build": build,
                    "tasks": [{"question": q, "variant": v, "ids": ids} for (q, v), (_, ids) in zip(tasks, seen)],
                    "response": body,
                    "wall_ms": round(wall, 3),
                })
                print(f"ref {rid} prompts={len(tasks)} n={total} wall_ms={wall:.0f}", flush=True)
        finally:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
    with open(out / "reference.jsonl", "w", encoding="utf-8") as f:
        for row in reference:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"lev_ref: {len(reference)} requests -> {out / 'reference.jsonl'} ({build})")
    print_table(reference)
    return 0


def print_table(rows: list[dict[str, Any]]) -> None:
    print("| id | prompts | n | question | top option | p | margin |")
    print("|---|---|---|---|---|---|---|")
    for row in rows:
        n = sum(len(t["ids"]) for t in row["tasks"])
        for qid, answer in row["response"]["answers"].items():
            option, p, margin = top(answer)
            print(f"| {row['id']} | {len(row['tasks'])} | {n} | {qid} | {option} | {p:.4f} | {margin:.4f} |")


def table(args: Any) -> int:
    path = Path(args.path)
    if path.is_dir():
        path = path / "reference.jsonl"
    print_table([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()])
    return 0


# ---- the self-test (no server, no box) ----

def self_test() -> bool:
    ok = True

    def expect(name: str, got: Any, want: Any) -> None:
        nonlocal ok
        if got != want:
            print(f"FAIL {name}: got {got!r}, want {want!r}")
            ok = False

    expect("request text drops the id", request_text('{"id": "a-1", "model": "lev", "state": 1}'), '{"model": "lev", "state": 1}')
    expect("n_variants: two-option choice", n_variants({"type": "choice", "criteria": {"a": 1, "b": 2}}), 2)
    expect("n_variants: one-option choice", n_variants({"type": "choice", "criteria": {"a": 1}}), 1)
    expect("n_variants: score and noul", [n_variants({"type": "score", "criteria": [1, 2]}), n_variants({"type": "noul"})], [1, 1])
    request = {"questions": {"d": {"type": "choice", "criteria": {"x": 1, "y": 2}}, "u": {"type": "score", "criteria": ["a", "b"]},
                             "o": {"type": "noul"}, "s": {"type": "choice", "criteria": {"only": 1}}}}
    expect("plan", plan(request), [("d", 0), ("d", 1), ("u", 0), ("o", 0), ("s", 0)])
    log = ("I slot update_slots: id  0 | task 7 | bloomery-ids n 3: 10 20 30 \n"
           "noise\n"
           "I slot update_slots: id  0 | task 5 | bloomery-ids n 2: 1 2 \n")
    expect("scrape sorts by task id", scrape(log), [(5, [1, 2]), (7, [10, 20, 30])])
    try:
        scrape("x | task 1 | bloomery-ids n 3: 1 2 \n")
        expect("scrape refuses a short list", "accepted", "SystemExit")
    except SystemExit as e:
        expect("scrape refuses a short list", "claims 3 ids and lists 2" in str(e), True)
    expect("top of a noul", top({"type": "noul", "noul": 0.25}), ("no", 0.75, 0.5))
    option, p, margin = top({"type": "choice", "probabilities": {"a": 0.2, "b": 0.7, "c": 0.1}})
    expect("top of a choice", (option, round(p, 6), round(margin, 6)), ("b", 0.7, 0.5))
    expect("top of a one-option choice", top({"type": "choice", "probabilities": {"a": 1.0}}), ("a", 1.0, 1.0))
    patch = PATCH.read_text()
    codes = label_codes()
    expect("label codes", (len(codes), codes[0], codes[25], codes[26], codes[27], codes[-1]), (702, "A", "Z", "AA", "AB", "ZZ"))
    # a vocabulary that holds "AA" as two tokens: it is skipped, and the later codes follow
    got = pick_labels(lambda c: [100 + len(c)] if c != "AA" else [1, 2])
    expect("labels skip a split code", (len(got), got[26]["code"]), (255, "AB"))
    expect("labels are capped", pick_labels(lambda c: [7])[-1]["code"], codes[254])
    expect("the patch names one file", re.findall(r"^\+\+\+ b/(\S+)", patch, re.M), ["tools/server/server-context.cpp"])
    expect("the patch prints the ids line the scraper reads", "bloomery-ids n %d: %s" in patch, True)
    return ok


def main(argv: list[str]) -> int:
    import argparse

    if argv == ["--self-test"]:
        good = self_test()
        print("lev_ref self-test: " + ("ok" if good else "FAILED"))
        return 0 if good else 1
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--suite", required=True)
    r.add_argument("--edge", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--port", type=int, default=8092)
    r.add_argument("--tree", required=True)
    t = sub.add_parser("table")
    t.add_argument("path")
    args = p.parse_args(argv)
    return run(args) if args.cmd == "run" else table(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
