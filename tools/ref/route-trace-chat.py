#!/usr/bin/env python3
"""The chat route trace: one bloomery-serve-ds41 process (one load) writes a route trace
(crates/gpu/src/host/route_trace.rs) of the model's greedy replies to a list of chat prompts.

    tools/ref/route-trace-chat.py run --server BIN --out DIR --prompts FILE[,FILE...]
                                      [--place a] [--n-predict 256] [--bound 1800]
    tools/ref/route-trace-chat.py join DIR
    tools/ref/route-trace-chat.py --self-test

run    DIR is an absolute path or one under the repository's target/: a path elsewhere in the tree is
       refused by name, since tools/box.sh's sync (rsync --delete) removes what it does not carry.
       Starts `timeout --kill-after=10 BOUND BIN --host 127.0.0.1 --port 0 --place PLACE --cache-ram 0`
       with BLOOMERY_ROUTE_TRACE=DIR and BLOOMERY_PREFILL=steps (the trace records the step feed; the
       server creates DIR at main and refuses an existing one), its stderr in DIR.serve.log, and waits for
       its `listening` record (tools/bloomery/records.py). Then, for every prompt row of the files in file
       order: POST /apply-template (one user message), /tokenize of that text (add_special false: the ids
       /v1/chat/completions runs), and /completion of those ids — temperature 0 (the engine's argmax),
       n_predict N, cache_prompt false (each request from an empty cache, so no request keeps the template
       head of the one before), return_tokens. One line a request into DIR/requests.tsv as it returns.
       Then TERM to the timeout's pid (recorded at spawn; the timeout passes it on), KILL 30 s later if it
       is still up. The server never finishes its trace (a TERM runs no drop), so once it has exited
       with every request answered the driver seals the set (tools/bloomery/route_trace.py seal: the
       files cut to the manifest's positions, `# complete` written), then `join`. Exit 1 with the
       reason on any failure, after stopping the server; a set the driver did not seal stays without
       `# complete`, which every router set reader refuses.
join   DIR/contexts.tsv from DIR/requests.tsv and the set's call rows (tools/bloomery/route_trace.py
       read_calls), refused by name unless every request agrees with the serve contract
       (crates/serve/src/engine.rs): cut or reset to `cache` k, a prompt call of ids[k..n-1], a step on
       ids[n-1], then a step on every generated token but the last. So per request: its call's pos0 = k,
       prompt_call = n - 1 - k, end - first = prompt_call + max(generated, 1); and the set holds one call a
       request.
       A context's `prompt` is n - k: the positions whose input is a prompt id (the last one a step, the
       prompt call's positions frozen for an expert cache are `prompt_call`).

A prompt file is `id<TAB>domain or genre<TAB>split<TAB>text` rows under `#` lines (tools/ref/data/
d2-prompts-*.tsv). The genre: `code` stays `code`, any other Korean domain is `ko`, `en` stays `en`.
"""
import http.client
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import importlib.util

_here = os.path.dirname(os.path.abspath(__file__))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_here, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rt = _load("route_trace", "../bloomery/route_trace.py")

ROOT = os.path.realpath(os.path.join(_here, "..", ".."))

REQUEST_COLUMNS = ("request", "prompt_id", "domain", "genre", "split", "prompt_ids", "cache", "prompt_n",
                   "generated", "stop")


class DriverError(Exception):
    pass


def read_prompts(paths):
    """Every prompt row of `paths` in order: id, domain (column 2 as written), genre, split, text."""
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                fields = line.split("\t")
                if len(fields) != 4 or not all(fields):
                    raise DriverError(f"{p}:{n}: not id, domain, split, text")
                pid, domain, split, text = fields
                if split not in ("learn", "held"):
                    raise DriverError(f"{p}:{n}: split {split!r} is not learn or held")
                genre = domain if domain in ("code", "en") else "ko"
                rows.append(dict(prompt_id=pid, domain=domain, genre=genre, split=split, text=text))
    ids = [r["prompt_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise DriverError("a prompt id repeats across the files")
    return rows


def read_requests(set_dir):
    path = os.path.join(set_dir, "requests.tsv")
    out, build = [], "unknown"
    with open(path, encoding="utf-8") as f:
        names = None
        for n, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if line.startswith("# columns\t"):
                names = line.split("\t")[1:]
                continue
            if line.startswith("# build\t"):
                build = line.split("\t", 1)[1]
                continue
            if not line or line.startswith("#"):
                continue
            if names is None or len(line.split("\t")) != len(names):
                raise DriverError(f"{path}:{n}: a row the column line does not read")
            r = dict(zip(names, line.split("\t")))
            for k in ("request", "prompt_ids", "cache", "prompt_n", "generated"):
                r[k] = int(r[k])
            out.append(r)
    return out, build


def join(set_dir):
    """contexts.tsv of the set in `set_dir` (the module doc's contract); returns its rows."""
    try:
        calls, tokens = rt.read_calls(set_dir)
    except rt.TraceError as e:
        raise DriverError(str(e)) from None
    reqs, build = read_requests(set_dir)
    if len(reqs) != len(calls):
        raise DriverError(f"{set_dir}: {len(reqs)} requests, the trace holds {len(calls)} prompt calls")
    rows = []
    for i, (q, c) in enumerate(zip(reqs, calls)):
        n, k, g = q["prompt_ids"], q["cache"], q["generated"]
        want = dict(pos0=k, prompt_call=n - 1 - k)
        got = dict(pos0=c["pos0"], prompt_call=c["prompt_call"])
        if q["request"] != i or got != want or q["prompt_n"] != n - k:
            raise DriverError(f"request {i} ({q['prompt_id']}): the call row {c}, the request {q}: want {want} "
                              f"and prompt_n = n - k")
        # The step on ids[n-1] runs even for a budget of 0; after it, every generated token but the last.
        if c["end"] - c["first"] != c["prompt_call"] + max(g, 1):
            raise DriverError(f"request {i} ({q['prompt_id']}): {c['end'] - c['first']} positions, want prompt "
                              f"call {c['prompt_call']} + {g} generated (the last one never run)")
        rows.append(dict(request=i, first=c["first"], end=c["end"], prompt=n - k, prompt_call=c["prompt_call"],
                         prompt_ids=n, cache=k, generated=g, stop=q["stop"], prompt_id=q["prompt_id"],
                         genre=q["genre"], split=q["split"], domain=q["domain"]))
    cols = list(rt.CONTEXT_COLUMNS) + ["domain"]
    text = ["# route-trace contexts — one row per request (tools/ref/route-trace-chat.py join)",
            f"# build\t{build}", f"# positions\t{tokens}", "# columns\t" + "\t".join(cols)]
    text += ["\t".join(str(r[c]) for c in cols) for r in rows]
    tmp = os.path.join(set_dir, "contexts.tsv.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(text) + "\n")
    os.replace(tmp, os.path.join(set_dir, "contexts.tsv"))
    try:
        rt.read_contexts(set_dir)
    except rt.TraceError as e:
        raise DriverError(f"the contexts written do not read back: {e}") from None
    return rows


class Server:
    def __init__(self, port):
        self.port = port

    def call(self, method, path, body=None, timeout=900):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            c.request(method, path, body=None if body is None else json.dumps(body),
                      headers={"Content-Type": "application/json"})
            r = c.getresponse()
            data = r.read()
        finally:
            c.close()
        if r.status != 200:
            raise DriverError(f"{method} {path}: HTTP {r.status}: {data[:400]!r}")
        return json.loads(data)


def listening_port(log, records):
    try:
        with open(log, encoding="utf-8", errors="replace") as f:
            rec = records.first(records.read(f.read().splitlines(), "bloomery-serve-ds41"), "listening")
    except OSError:
        return None
    if rec is None:
        return None
    return int(str(rec["addr"]).rsplit(":", 1)[1])


def out_paths(out, root=ROOT):
    """The trace directory and its server log for `--out`, refused by name when either lands in the
    repository tree `root` outside its target/ (the sync removes it)."""
    out = os.path.realpath(out)
    log = out + ".serve.log"
    target = os.path.join(root, "target")
    for p in (out, log):
        inside = os.path.commonpath([p, root]) == root
        kept = os.path.commonpath([p, target]) == target and p != target
        if inside and not kept:
            raise DriverError(f"--out {out}: {p} is inside the repository tree {root}, whose next box sync "
                              f"deletes it; give an absolute path outside it or one under {target}/")
    return out, log


def run(a):
    records = _load("records", "../bloomery/records.py")
    prompts = read_prompts(a.prompts.split(","))
    out, log = out_paths(a.out)
    if os.path.exists(out):
        raise DriverError(f"{out} exists: the trace takes a new directory")
    env = dict(os.environ, BLOOMERY_ROUTE_TRACE=out, BLOOMERY_PREFILL="steps")
    cmd = ["timeout", "--kill-after=10", str(a.bound), a.server, "--host", "127.0.0.1", "--port", "0",
           "--place", a.place, "--cache-ram", "0"]
    print(f"route-trace-chat: {len(prompts)} prompts; {' '.join(cmd)} (stderr {log})", flush=True)
    with open(log, "w", encoding="utf-8") as lf:
        p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    print(f"route-trace-chat: server timeout pid {p.pid}", flush=True)
    t0 = time.monotonic()
    try:
        port = None
        while port is None:
            if p.poll() is not None:
                raise DriverError(f"the server exited {p.returncode} before it listened (see {log})")
            if time.monotonic() - t0 > a.bound:
                raise DriverError(f"the server did not listen within {a.bound} s")
            time.sleep(2)
            port = listening_port(log, records)
        load_s = time.monotonic() - t0
        s = Server(port)
        version = (s.call("GET", "/props").get("engine") or {}).get("version", "unknown")
        print(f"route-trace-chat: listening on {port} after {load_s:.0f} s; engine {version}", flush=True)
        req_path = os.path.join(out, "requests.tsv")
        with open(req_path, "w", encoding="utf-8") as rf:
            rf.write(f"# route-trace-chat requests\n# build\t{version}\n# columns\t" + "\t".join(REQUEST_COLUMNS) + "\n")
            for i, q in enumerate(prompts):
                t = time.monotonic()
                text = s.call("POST", "/apply-template",
                              {"messages": [{"role": "user", "content": q["text"]}]})["prompt"]
                ids = s.call("POST", "/tokenize", {"content": text, "add_special": False})["tokens"]
                r = s.call("POST", "/completion", {"prompt": ids, "temperature": 0, "n_predict": a.n_predict,
                                                   "cache_prompt": False, "return_tokens": True, "stream": False})
                tim = r.get("timings") or {}
                gen = r.get("tokens")
                if not isinstance(gen, list) or tim.get("predicted_n") != len(gen):
                    raise DriverError(f"request {i}: tokens {gen!r:.80} and timings {tim} disagree")
                stop = str(r.get("stop_type", "?")) + ("+truncated" if r.get("truncated") else "")
                row = dict(request=i, prompt_id=q["prompt_id"], domain=q["domain"], genre=q["genre"],
                           split=q["split"], prompt_ids=len(ids), cache=tim.get("cache_n"),
                           prompt_n=tim.get("prompt_n"), generated=len(gen), stop=stop)
                rf.write("\t".join(str(row[c]) for c in REQUEST_COLUMNS) + "\n")
                rf.flush()
                print(f"request {i} {q['prompt_id']} {q['genre']}: {len(ids)} prompt ids, {len(gen)} generated, "
                      f"{stop}, {time.monotonic() - t:.1f} s", flush=True)
    finally:
        stop_server(p)
    try:
        print(f"route-trace-chat: {rt.seal(out)}", flush=True)
    except rt.TraceError as e:
        raise DriverError(str(e)) from None
    rows = join(out)
    pos = sum(r["end"] - r["first"] for r in rows)
    print(f"route-trace-chat: {len(rows)} requests, {pos} positions, prompt {sum(r['prompt'] for r in rows)}, "
          f"wall {time.monotonic() - t0:.0f} s; contexts.tsv written", flush=True)
    return 0


def stop_server(p):
    """TERM to the timeout started here (it passes it on); KILL 30 s later if it is still up."""
    if p.poll() is not None:
        return
    p.send_signal(signal.SIGTERM)
    try:
        p.wait(timeout=30)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()
    print(f"route-trace-chat: server stopped ({p.returncode})", flush=True)


def self_test():
    root = "/r/bloomery"
    for out, ok in (("/r/bloomery/d2", False), ("/r/bloomery/tools/x", False), ("/r/bloomery/target", False),
                    ("/r/bloomery/target/d2", True), ("/r/other/d2", True), ("/r/bloomery-x/d2", True)):
        try:
            got = out_paths(out, root)
            assert ok and got == (out, out + ".serve.log"), (out, got)
        except DriverError as e:
            assert not ok and "inside the repository tree /r/bloomery" in str(e), (out, str(e))
    rows = read_prompts([os.path.join(_here, "data", "d2-prompts-ko.tsv"),
                         os.path.join(_here, "data", "d2-prompts-en-code.tsv")])
    genres = [r["genre"] for r in rows]
    assert len(rows) == 48 and rows[0]["prompt_id"] == "ko-01" and rows[-1]["prompt_id"] == "code-14", rows[-1]
    assert (genres.count("ko"), genres.count("en"), genres.count("code")) == (16, 14, 18), genres
    assert genres[20:34] == ["en"] * 14 and genres[34:] == ["code"] * 14
    with tempfile.TemporaryDirectory() as root:
        d = os.path.join(root, "t")
        # two requests: 5 prompt ids (call of 4, the 5th a step) + 2 generated (1 run); 4 ids + 1 generated
        rt._write_set(d, [(0, 4, 6), (6, 3, 10)], 10)
        calls_lines = open(os.path.join(d, "MANIFEST.tsv"), encoding="utf-8").read()

        def reqs(edit=None):
            q = [dict(request=0, prompt_id="ko-01", domain="writing", genre="ko", split="learn", prompt_ids=5,
                      cache=0, prompt_n=5, generated=2, stop="limit"),
                 dict(request=1, prompt_id="en-01", domain="en", genre="en", split="learn", prompt_ids=4, cache=0,
                      prompt_n=4, generated=1, stop="eos")]
            if edit:
                q[edit[0]].update(edit[1])
            with open(os.path.join(d, "requests.tsv"), "w", encoding="utf-8") as f:
                f.write("# build\tv0\n# columns\t" + "\t".join(REQUEST_COLUMNS) + "\n")
                for r in q:
                    f.write("\t".join(str(r[c]) for c in REQUEST_COLUMNS) + "\n")

        reqs()
        got = join(d)
        assert [(r["first"], r["end"], r["prompt"], r["prompt_call"]) for r in got] == [(0, 6, 5, 4), (6, 10, 4, 3)]
        back, header = rt.read_contexts(d)
        assert header["build"] == "v0" and back[1]["genre"] == "en" and back[0]["domain"] == "writing", back
        for edit, want in [((0, dict(prompt_ids=6, prompt_n=6)), "want {'pos0': 0, 'prompt_call': 5}"),
                           ((0, dict(generated=3)), "want prompt call 4 + 3 generated"),
                           ((1, dict(cache=1, prompt_n=3)), "want {'pos0': 1, 'prompt_call': 2}")]:
            reqs(edit)
            try:
                join(d)
                raise AssertionError(f"accepted: {want}")
            except DriverError as e:
                assert want in str(e), (want, str(e))
        with open(os.path.join(d, "requests.tsv"), "a", encoding="utf-8") as f:
            f.write("\t".join(["2", "x", "en", "en", "held", "3", "0", "3", "1", "eos"]) + "\n")
        try:
            join(d)
            raise AssertionError("a request with no call row was joined")
        except DriverError as e:
            assert "3 requests, the trace holds 2 prompt calls" in str(e), e
        assert calls_lines.startswith("# router_trace")
    print("route-trace-chat: self-test ok")
    return 0


def main(argv):
    if argv == ["--self-test"]:
        return self_test()
    import argparse
    ap = argparse.ArgumentParser(prog="route-trace-chat.py")
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run")
    r.add_argument("--server", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--prompts", required=True)
    r.add_argument("--place", default="a")
    r.add_argument("--n-predict", type=int, default=256)
    r.add_argument("--bound", type=int, default=1800)
    j = sub.add_parser("join")
    j.add_argument("dir")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "run":
            return run(a)
        if a.cmd == "join":
            join(a.dir)
            return 0
    except (DriverError, OSError, ValueError) as e:
        print(f"route-trace-chat: {e}", file=sys.stderr)
        return 1
    ap.print_help(sys.stderr)
    return 64


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
