#!/usr/bin/env python3
"""Clef end to end: one model file through `bloomery_serve_clef` against the official BF16 answers.

    tools/ref/clef/e2e.py run --model FILE --out DIR [--port 8091] [--bin PATH] [--head PATH] [--data DIR]
    tools/ref/clef/e2e.py table LOG [LOG ...]
    tools/ref/clef/e2e.py --self-test

`run` starts the server on FILE with the release's head (`--head`, default
/models/clef-flash/hf/joint_head.safetensors), POSTs every request of `tools/ref/clef/suite.jsonl` (English)
and `suite-ko.jsonl` (Korean) that has an official row (`<data>/clef/flash/ref/reference.jsonl` and
`ref-ko/reference.jsonl`, `--data` default $BLOOMERY_DATA or /root/bloomery-data), and stops the server by
the pid captured at its spawn (written to `<out>/serve.pid`; signalled only while `/proc/<pid>/comm` names
the server). It writes `<out>/rows.json` (each request's response, the official one and the wall) and
`<out>/serve.log` (the server's stderr), and prints one line a question then `TOPS a/n`: the log `table`
reads. A server that does not print its listening line, or a response with no answers, stops the run by name
(rc 1) after the server is stopped.

A question's top is the answer the body picks: a noul's yes when its p >= 0.5, a choice's `choice`, else the
most probable option; its p is that answer's probability. |dp| is |p(ours, our top) - p(official, its top)|,
the same on a question whose tops agree.

`table` prints one Markdown row a log: the model file, tops agreeing over all questions, the English and
Korean split, max and mean |dp|, prefill tok/s on the longest prompt of the Korean suite (prompt_n /
prompt_ms of its timings), the median head_ms over the requests, and each miss as question (official p,
ours). The size column is left to the caller (the log does not hold it).
"""

from __future__ import annotations

import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SUITES = (("en", "tools/ref/clef/suite.jsonl", "clef/flash/ref/reference.jsonl"),
          ("ko", "tools/ref/clef/suite-ko.jsonl", "clef/flash/ref-ko/reference.jsonl"))
HEAD = "/models/clef-flash/hf/joint_head.safetensors"


def top(a: dict) -> tuple[str, float]:
    """The body's answer to one question and its probability (module doc)."""
    t = a["type"]
    if t == "noul":
        return ("yes" if a["noul"] >= 0.5 else "no"), a["noul"]
    if t == "choice":
        return a["choice"], a["probabilities"][a["choice"]]
    pr = a["probabilities"]
    k = max(pr, key=pr.get)
    return k, pr[k]


def lines(rows: list[dict]) -> list[str]:
    """One log line a question of `rows` (`run`'s rows.json), then `TOPS a/n`. A response with no answers
    raises by name."""
    out, agree, n = [], 0, 0
    for r in rows:
        resp = r["resp"]
        if "answers" not in resp:
            raise SystemExit(f"e2e: {r['id']} has no answers: {json.dumps(resp)[:300]}")
        tm = resp.get("timings", {})
        for q, ref in r["ref"]["answers"].items():
            ours = resp["answers"].get(q)
            if ours is None:
                raise SystemExit(f"e2e: {r['id']} has no answer to {q}")
            to, po = top(ours)
            tr, pr = top(ref)
            n += 1
            agree += to == tr
            out.append(f"{r['suite']} {r['id']} {q}\t{to}/{tr}\t{'ok' if to == tr else 'MISS'}\t"
                       f"p {po:.4f}/{pr:.4f}\t|dp| {abs(po - pr):.4f}\tprompt_n {tm.get('prompt_n')}\t"
                       f"prompt_ms {tm.get('prompt_ms')}\thead_ms {tm.get('head_ms')}")
    out.append(f"TOPS {agree}/{n}")
    return out


LINE = re.compile(r"^(?P<suite>en|ko) (?P<id>\S+) (?P<q>.+?)\t(?P<to>[^/\t]*)/(?P<tr>[^\t]*)\t(?P<ok>ok|MISS)\t"
                  r"p (?P<po>[\d.]+)/(?P<pr>[\d.]+)\t\|dp\| (?P<dp>[\d.]+)\tprompt_n (?P<n>\d+)\t"
                  r"prompt_ms (?P<ms>[\d.]+)\thead_ms (?P<head>[\d.]+)$")


def summary(log: str) -> dict:
    """The table's fields of one log's text (module doc). A log with no question line, or whose TOPS line
    disagrees with its lines, raises by name."""
    qs, model, tops = [], None, None
    for l in log.splitlines():
        m = LINE.match(l)
        if m:
            qs.append(m.groupdict())
        elif l.startswith("TOPS "):
            tops = l.split()[1]
        elif "(model " in l:
            model = l.split("(model ", 1)[1].split(",", 1)[0].rstrip(")")
    if not qs or tops is None:
        raise SystemExit("e2e table: no question lines or no TOPS line in the log")
    ok = [q["ok"] == "ok" for q in qs]
    if f"{sum(ok)}/{len(qs)}" != tops:
        raise SystemExit(f"e2e table: TOPS {tops} but the lines agree {sum(ok)}/{len(qs)}")
    split = {s: (sum(o for q, o in zip(qs, ok) if q["suite"] == s), sum(q["suite"] == s for q in qs))
             for s in ("en", "ko")}
    dps = [float(q["dp"]) for q in qs]
    per_req = {(q["suite"], q["id"]): q for q in qs}.values()
    ko = [q for q in per_req if q["suite"] == "ko"]
    if not ko:
        raise SystemExit("e2e table: no Korean request, so no long prompt to read prefill from")
    long = max(ko, key=lambda q: int(q["n"]))
    return {
        "model": model or "?",
        "tops": tops,
        "en": split["en"],
        "ko": split["ko"],
        "dp_max": max(dps),
        "dp_mean": sum(dps) / len(dps),
        "pp_n": int(long["n"]),
        "pp": int(long["n"]) / float(long["ms"]) * 1000.0,
        "head_ms": statistics.median(float(q["head"]) for q in per_req),
        "misses": [(q["q"], float(q["pr"]), float(q["po"]), q["tr"], q["to"])
                   for q, o in zip(qs, ok) if not o],
    }


def row(s: dict) -> str:
    misses = "; ".join(f"{q} (official {tr} {pr:.3f}, ours {to} {po:.3f})" for q, pr, po, tr, to in s["misses"])
    return (f"| {s['model']} | | {s['tops']} | {s['en'][0]}/{s['en'][1]} | {s['ko'][0]}/{s['ko'][1]} | "
            f"{s['dp_max']:.4f} | {s['dp_mean']:.4f} | {s['pp']:,.0f} (n {s['pp_n']}) | {s['head_ms']:.1f} | "
            f"{misses or '—'} |")


def stop(p: subprocess.Popen) -> None:
    """SIGTERM the server spawned as `p` while its comm still names it; SIGKILL after 30 s."""
    try:
        comm = Path(f"/proc/{p.pid}/comm").read_text().strip()
    except OSError:
        comm = None
    if comm and comm.startswith("bloomery_serve"):
        os.kill(p.pid, signal.SIGTERM)
        try:
            p.wait(30)
        except subprocess.TimeoutExpired:
            os.kill(p.pid, signal.SIGKILL)
            p.wait()
    print(f"server stopped (pid {p.pid}, comm {comm}, rc {p.poll()})", flush=True)


def run(argv: list[str]) -> int:
    a = dict(zip(argv[::2], argv[1::2]))
    if len(argv) % 2 or not {"--model", "--out"} <= a.keys() or a.keys() - {"--model", "--out", "--port", "--bin",
                                                                              "--head", "--data"}:
        raise SystemExit(__doc__)
    out = Path(a["--out"])
    out.mkdir(parents=True, exist_ok=True)
    port = int(a.get("--port", "8091"))
    binp = a.get("--bin", str(ROOT / "target/release/bloomery_serve_clef"))
    data = Path(a.get("--data", os.environ.get("BLOOMERY_DATA", "/root/bloomery-data")))
    refs = {}
    for suite, _, refp in SUITES:
        for l in open(data / refp):
            r = json.loads(l)
            refs[(suite, r["id"])] = r["response"]
    log = open(out / "serve.log", "w")
    t0 = time.time()
    p = subprocess.Popen([binp, "--model", a["--model"], "--head", a.get("--head", HEAD), "--port", str(port)],
                         stdout=subprocess.PIPE, stderr=log, text=True, start_new_session=True)
    (out / "serve.pid").write_text(f"{p.pid}\n")
    try:
        first = p.stdout.readline() if p.stdout else ""
        print(f"open {time.time() - t0:.1f}s: {first.strip()}", flush=True)
        if "listening" not in first:
            log.flush()
            print((out / "serve.log").read_text()[-4000:])
            raise SystemExit("e2e: the server did not start")
        rows = []
        for suite, path, _ in SUITES:
            for l in open(ROOT / path):
                req = json.loads(l)
                rid = req.pop("id")
                if (suite, rid) not in refs:
                    continue
                hr = urllib.request.Request(f"http://127.0.0.1:{port}/v1/systemone", data=json.dumps(req).encode(),
                                            headers={"Content-Type": "application/json"})
                t1 = time.time()
                try:
                    resp = json.loads(urllib.request.urlopen(hr, timeout=300).read())
                except urllib.error.HTTPError as e:
                    resp = {"error": e.code, "body": e.read().decode()[:500]}
                rows.append({"suite": suite, "id": rid, "resp": resp, "ref": refs[(suite, rid)],
                             "wall_ms": (time.time() - t1) * 1000})
        (out / "rows.json").write_text(json.dumps(rows))
        for l in lines(rows):
            print(l, flush=True)
    finally:
        stop(p)
    return 0


def self_test() -> None:
    noul = {"type": "noul", "noul": 0.7}
    choice = {"type": "choice", "choice": "b", "probabilities": {"a": 0.3, "b": 0.6, "c": 0.1}}
    multi = {"type": "multi", "probabilities": {"x": 0.2, "y": 0.5}}
    assert top(noul) == ("yes", 0.7) and top({"type": "noul", "noul": 0.2}) == ("no", 0.2)
    assert top(choice) == ("b", 0.6) and top(multi) == ("y", 0.5)

    def req(suite, rid, n, ms, head, answers, ref):
        return {"suite": suite, "id": rid, "ref": {"answers": ref},
                "resp": {"answers": answers, "timings": {"prompt_n": n, "prompt_ms": ms, "head_ms": head}}}
    rows = [
        req("en", "a-1", 300, 60.0, 30.0, {"q1": noul, "q2": choice},
            {"q1": {"type": "noul", "noul": 0.68}, "q2": {**choice, "choice": "a"}}),
        req("ko", "b-1", 4067, 900.0, 20.0, {"부서": noul}, {"부서": {"type": "noul", "noul": 0.71}}),
        req("ko", "b-2", 250, 50.0, 25.0, {"x": multi}, {"x": multi}),
    ]
    log = "open 1.0s: bloomery_serve_clef: listening on http://127.0.0.1:1 (model m-Q3_K_M.gguf, quant Q3_K_M)\n"
    log += "\n".join(lines(rows))
    s = summary(log)
    assert s["model"] == "m-Q3_K_M.gguf", s
    assert s["tops"] == "3/4" and s["en"] == (1, 2) and s["ko"] == (2, 2), s
    assert abs(s["dp_max"] - 0.3) < 1e-9 and abs(s["dp_mean"] - (0.02 + 0.3 + 0.01 + 0.0) / 4) < 1e-9, s
    assert s["pp_n"] == 4067 and abs(s["pp"] - 4067 / 0.9) < 1e-6, s
    assert s["head_ms"] == 25.0, s
    assert s["misses"] == [("q2", 0.3, 0.6, "a", "b")], s
    assert row(s).startswith("| m-Q3_K_M.gguf | | 3/4 | 1/2 | 2/2 | 0.3000 |"), row(s)
    # A response with no answers and a log whose TOPS disagrees with its lines stop by name.
    for bad, want in (([{"suite": "en", "id": "e", "resp": {"error": 500}, "ref": {"answers": {}}}], "no answers"),):
        try:
            lines(bad)
        except SystemExit as e:
            assert want in str(e), e
        else:
            raise AssertionError("a response with no answers passed")
    try:
        summary(log.replace("TOPS 3/4", "TOPS 4/4"))
    except SystemExit as e:
        assert "TOPS 4/4" in str(e), e
    else:
        raise AssertionError("a TOPS line that disagrees with its lines passed")
    print("e2e self-test ok")


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        return 0
    if len(sys.argv) >= 2 and sys.argv[1] == "run":
        return run(sys.argv[2:])
    if len(sys.argv) >= 3 and sys.argv[1] == "table":
        print("| file | size (GB) | tops /31 | English /16 | Korean /15 | max abs dp | mean abs dp | "
              "prefill tok/s | head_ms median | misses (question: official, ours) |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for f in sys.argv[2:]:
            print(row(summary(Path(f).read_text())))
        return 0
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
