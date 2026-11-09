#!/usr/bin/env python3
"""k31-calib-gen.py — conversations written by Kimi K3.1 itself, the calibration corpus of the K3.1 imatrix.

    python3 -I tools/ref/k31-calib-gen.py --seeds tools/ref/k31-calib-seeds.jsonl --out conversations.jsonl [--budget-usd 3.50]
    python3 -I tools/ref/k31-calib-gen.py --self-test

Each seed (tools/ref/k31-calib-seeds.jsonl, one JSON object per line: id, lang en|ko|code, kind, turns, optional system, tools,
tool_results, reasoning, max_tokens) becomes one conversation through OpenRouter's chat API, model moonshotai/kimi-k3.1-pro on the
official endpoint (provider order ["Moonshot AI"], no fallbacks: the released mxfp4 weights). Every user turn is one request; the model's
earlier answers are fed back; a tool call is answered from the seed's `tool_results` (a string, or a map path -> text for read_file)
and the model then answers again. Reasoning is requested at effort medium for a seed with `"reasoning": true` and disabled otherwise;
the returned reasoning text is stored per assistant message.

The key is OPENROUTER_API_KEY from ~/.config/outsource/credentials, read here and never printed, logged or written.

Budget: usage.cost of every response is summed; a seed is started only while spent + (in-flight seeds x the dearest seed so far) stays
under --budget-usd, and a conversation that meets the budget between two of its requests is dropped, not written. Retries: 429, 5xx and
network errors back off 2, 4, 8, 16 s (4 retries), then the run stops by name; any other 4xx stops the run by name.

Output: one JSON line per finished conversation, appended to --out: id, lang, kind, code_lang, thinking, system, messages (OpenAI
chat form; assistant messages carry `reasoning`), tools, cost, provider, requests, finish_reasons, truncated. A run that stopped is
continued with --resume (seeds whose id is already in --out are skipped); an existing --out without --resume is refused.
Exit: 0 finished or stopped at the budget; 2 a named stop (a 4xx, retries exhausted); 64 a usage error.
"""
import argparse
import concurrent.futures
import contextlib
import io
import json
import os
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "moonshotai/kimi-k3.1-pro"
PROVIDER = {"order": ["Moonshot AI"], "allow_fallbacks": False}
CREDENTIALS = "~/.config/outsource/credentials"
BACKOFF_S = (2, 4, 8, 16)
MAX_TOOL_ROUNDS = 3
MIN_SEED_COST = 0.04  # the in-flight reserve until a seed has finished


class StopRun(Exception):
    """A named reason the whole run stops (a 4xx, retries exhausted); the conversation in hand is dropped."""


class BudgetHit(Exception):
    """The budget was met between two requests of one conversation: it is dropped."""


def read_key(path=CREDENTIALS):
    for line in open(os.path.expanduser(path), encoding="utf-8"):
        k, _, v = line.strip().partition("=")
        if k == "OPENROUTER_API_KEY" and v:
            return v.strip().strip("\"'")
    raise SystemExit(f"k31-calib-gen: no OPENROUTER_API_KEY in {path}")


def http_transport(body, key, timeout=600):
    """One POST; returns (status, payload dict or None, error text). Never echoes a header."""
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(), method="POST",
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r), ""
    except urllib.error.HTTPError as e:
        try:
            text = e.read().decode("utf-8", "replace")
        except Exception:
            text = ""
        return e.code, None, text
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        return 0, None, f"{type(e).__name__}"


class Run:
    def __init__(self, transport, key, budget, sleep=time.sleep, workers=4, verbose=True):
        self.transport, self.key, self.budget, self.sleep, self.workers = transport, key, budget, sleep, workers
        self.lock = threading.Lock()
        self.spent = 0.0
        self.requests = 0
        self.providers = {}
        self.max_seed_cost = 0.0
        self.stop = None  # the named reason the run stops, set once

    def say(self, text):
        print(text.replace(self.key, "***") if self.key else text, flush=True)

    def request(self, body):
        """One request with the retry policy; returns the response payload."""
        attempt = 0
        while True:
            with self.lock:
                if self.stop:
                    raise StopRun(self.stop)
                if self.spent >= self.budget:
                    raise BudgetHit()
            status, payload, err = self.transport(body, self.key)
            if status == 200 and payload and payload.get("choices"):
                usage = payload.get("usage") or {}
                with self.lock:
                    self.spent += float(usage.get("cost") or 0.0)
                    self.requests += 1
                    n = self.requests
                    prov = payload.get("provider") or "?"
                    self.providers[prov] = self.providers.get(prov, 0) + 1
                    spent = self.spent
                if n % 10 == 0:
                    self.say(f"k31-calib-gen: {n} requests, cost ${spent:.4f} of ${self.budget:.2f}")
                return payload
            if status == 200:
                err = "no choices in the response"
            if status in (0, 429) or status >= 500 or status == 200:
                if attempt >= len(BACKOFF_S):
                    raise StopRun(f"retries exhausted after {attempt + 1} attempts (last status {status}: {err[:160]})")
                self.sleep(BACKOFF_S[attempt])
                attempt += 1
                continue
            raise StopRun(f"HTTP {status} is not retried: {err[:300]}")

    def conversation(self, seed):
        """The seed's whole conversation as a result row, or None when it is dropped (an empty answer)."""
        thinking = bool(seed.get("reasoning"))
        messages = []
        if seed.get("system"):
            messages.append({"role": "system", "content": seed["system"]})
        tools = seed.get("tools") or None
        cost0, finish, truncated, requests = 0.0, [], False, 0
        for turn in seed["turns"]:
            messages.append({"role": "user", "content": turn})
            for _round in range(MAX_TOOL_ROUNDS + 1):
                body = {"model": MODEL, "provider": PROVIDER, "usage": {"include": True},
                        "messages": [wire(m) for m in messages], "max_tokens": int(seed.get("max_tokens", 1500)),
                        "reasoning": {"effort": "medium"} if thinking else {"enabled": False}}
                if tools:
                    body["tools"] = tools
                payload = self.request(body)
                requests += 1
                cost0 += float((payload.get("usage") or {}).get("cost") or 0.0)
                choice = payload["choices"][0]
                msg = choice.get("message") or {}
                finish.append(choice.get("finish_reason"))
                out = {"role": "assistant", "content": msg.get("content") or ""}
                if thinking and msg.get("reasoning"):
                    out["reasoning"] = msg["reasoning"]
                calls = msg.get("tool_calls") or []
                if calls:
                    out["tool_calls"] = [{"id": c.get("id") or f"call_{i}", "type": "function",
                                          "function": {"name": c["function"]["name"],
                                                       "arguments": c["function"].get("arguments") or "{}"}}
                                         for i, c in enumerate(calls)]
                messages.append(out)
                if choice.get("finish_reason") == "length":
                    truncated = True
                if not out["content"] and not calls:
                    return None
                if not calls or truncated:
                    break
                for c in out["tool_calls"]:
                    messages.append({"role": "tool", "tool_call_id": c["id"], "tool": c["function"]["name"],
                                     "content": tool_result(seed, c)})
            if truncated:
                break
        if messages[-1].get("tool_calls") or messages[-1]["role"] != "assistant":
            return None  # ended on an unanswered tool round
        return {"id": seed["id"], "lang": seed["lang"], "kind": seed["kind"], "code_lang": seed.get("code_lang"),
                "thinking": thinking, "system": seed.get("system"), "messages": messages, "tools": tools,
                "cost": round(cost0, 6), "provider": self.provider_of(), "requests": requests,
                "finish_reasons": finish, "truncated": truncated}

    def provider_of(self):
        with self.lock:
            return max(self.providers, key=self.providers.get) if self.providers else None


def wire(m):
    """The request form of a stored message: no reasoning text, no renderer-only keys."""
    out = {k: m[k] for k in ("role", "content", "tool_calls", "tool_call_id") if k in m}
    return out


def tool_result(seed, call):
    name = call["function"]["name"]
    res = (seed.get("tool_results") or {}).get(name)
    if isinstance(res, dict):
        try:
            path = json.loads(call["function"]["arguments"]).get("path", "")
        except (ValueError, AttributeError):
            path = ""
        return res.get(path, json.dumps({"error": f"no such file: {path}"}))
    if res is None:
        return json.dumps({"error": f"tool {name} is not available"})
    return res


def generate(args, transport, key, sleep=time.sleep):
    seeds = [json.loads(l) for l in open(args.seeds, encoding="utf-8") if l.strip()]
    done = set()
    if os.path.exists(args.out) and os.path.getsize(args.out):
        if not args.resume:
            print(f"k31-calib-gen: {args.out} exists and is not empty: --resume continues it", file=sys.stderr)
            return 64
        done = {json.loads(l)["id"] for l in open(args.out, encoding="utf-8") if l.strip()}
    todo = [s for s in seeds if s["id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    run = Run(transport, key, args.budget_usd, sleep=sleep, workers=args.workers)
    written, dropped = 0, 0
    out_lock = threading.Lock()
    out = open(args.out, "a", encoding="utf-8")

    def work(seed):
        try:
            row = run.conversation(seed)
        except BudgetHit:
            return seed["id"], "budget"
        except StopRun as e:
            with run.lock:
                run.stop = run.stop or str(e)
            return seed["id"], "stop"
        if row is None:
            return seed["id"], "dropped"
        with out_lock:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
        with run.lock:
            run.max_seed_cost = max(run.max_seed_cost, row["cost"])
        return seed["id"], "ok"

    pending = set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        it = iter(todo)
        exhausted = False
        while True:
            while not exhausted and len(pending) < run.workers:
                with run.lock:
                    reserve = max(run.max_seed_cost * 1.2, MIN_SEED_COST) * (len(pending) + 1)
                    ok = run.spent + reserve <= run.budget and not run.stop
                if not ok:
                    exhausted = True
                    break
                try:
                    seed = next(it)
                except StopIteration:
                    exhausted = True
                    break
                pending.add(ex.submit(work, seed))
            if not pending:
                break
            finished, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for f in finished:
                _id, what = f.result()
                written += what == "ok"
                dropped += what == "dropped"
    out.close()
    provs = ",".join(f"{k}:{v}" for k, v in sorted(run.providers.items())) or "none"
    why = run.stop or ("budget" if len(todo) and written + dropped < len(todo) else "all seeds done")
    run.say(f"k31-calib-gen: done: requests={run.requests} conversations={written} dropped={dropped} "
            f"total_cost=${run.spent:.4f} providers={provs} stop={why}")
    return 2 if run.stop else 0


# ---------------------------------------------------------------- self-test

def _fake(script):
    """A transport from a function (body, n) -> (status, payload, err); the key must arrive, never in the body."""
    calls = []

    def t(body, key):
        assert key == "sk-or-v1-SELFTESTKEY", "the key must reach the transport"
        assert "sk-or-v1-SELFTESTKEY" not in json.dumps(body), "the key must never be in the request body"
        calls.append(body)
        return script(body, len(calls))
    t.calls = calls
    return t


def _ok(content, cost=0.1, tool_calls=None, reasoning=None, finish="stop"):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    if reasoning:
        msg["reasoning"] = reasoning
    return 200, {"provider": "Moonshot AI", "choices": [{"finish_reason": finish, "message": msg}],
                 "usage": {"cost": cost}}, ""


def self_test():
    key = "sk-or-v1-SELFTESTKEY"
    tmp = tempfile.mkdtemp(prefix="k31gen-")
    seeds = tmp + "/seeds.jsonl"
    seed_rows = [{"id": f"a{i}", "lang": "en", "kind": "qa", "turns": ["hi"], "max_tokens": 50} for i in range(10)]
    seed_rows.append({"id": "m0", "lang": "ko", "kind": "multi", "turns": ["하나", "둘"], "reasoning": True})
    seed_rows.append({"id": "t0", "lang": "en", "kind": "tool", "turns": ["weather?"],
                      "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}],
                      "tool_results": {"get_weather": "{\"t\":18}"}})
    with open(seeds, "w") as f:
        for s in seed_rows:
            f.write(json.dumps(s) + "\n")

    def ns(out, **kw):
        d = dict(seeds=seeds, out=out, budget_usd=3.5, resume=False, workers=1, limit=0)
        d.update(kw)
        return argparse.Namespace(**d)

    def go(args, transport, sleep=lambda s: None):
        buf, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
            rc = generate(args, transport, key, sleep)
        return rc, buf.getvalue() + err.getvalue()

    def rows(path):
        return [json.loads(l) for l in open(path)] if os.path.exists(path) else []

    # 1. the budget stops the run cleanly, keeping what was written, and never overshoots
    out = tmp + "/o1.jsonl"
    rc, log = go(ns(out, budget_usd=0.35), _fake(lambda b, n: _ok("a", 0.1)))
    r = rows(out)
    assert rc == 0 and 1 <= len(r) < len(seed_rows), (rc, len(r), log)
    assert sum(x["cost"] for x in r) <= 0.35 + 1e-9, "the written conversations cost over the budget"
    assert "stop=budget" in log and "total_cost=" in log, log

    # 2. resume skips what is done; a second run without --resume is refused
    out = tmp + "/o2.jsonl"
    go(ns(out, limit=3), _fake(lambda b, n: _ok("a", 0.01)))
    assert len(rows(out)) == 3
    rc, _ = go(ns(out), _fake(lambda b, n: _ok("a", 0.01)))
    assert rc == 64, "an existing --out without --resume must be refused"
    rc, _ = go(ns(out, resume=True), _fake(lambda b, n: _ok("a", 0.01)))
    ids = [x["id"] for x in rows(out)]
    assert rc == 0 and len(ids) == len(set(ids)) == len(seed_rows), ids

    # 3. 429 then 5xx retry with backoff 2, 4, then succeed; reasoning is requested per seed and stored
    out, sleeps = tmp + "/o3.jsonl", []
    state = {"n": 0}

    def flaky(b, n):
        state["n"] += 1
        if state["n"] == 1:
            return 429, None, "slow down"
        if state["n"] == 2:
            return 503, None, "busy"
        return _ok("ok", 0.01, reasoning="think" if b["reasoning"] != {"enabled": False} else None)
    rc, _ = go(ns(out, limit=1), _fake(flaky), sleeps.append)
    assert rc == 0 and sleeps == [2, 4] and len(rows(out)) == 1, (rc, sleeps)
    t = _fake(lambda b, n: _ok("ok", 0.01, reasoning="r"))
    out = tmp + "/o3b.jsonl"
    go(ns(out, resume=False), t)
    by = {x["id"]: x for x in rows(out)}
    assert "reasoning" in by["m0"]["messages"][1] and by["m0"]["thinking"] and not by["a0"]["thinking"]
    efforts = [c["reasoning"] for c in t.calls]
    assert {"effort": "medium"} in efforts and {"enabled": False} in efforts, efforts
    assert [m["role"] for m in by["m0"]["messages"]] == ["user", "assistant", "user", "assistant"]
    assert "reasoning" not in json.dumps(t.calls[-1]["messages"]), "stored reasoning must not be sent back"

    # 4. retries exhausted stops the run by name (rc 2); a plain 4xx stops without retrying
    out, sleeps = tmp + "/o4.jsonl", []
    rc, log = go(ns(out), _fake(lambda b, n: (503, None, "down")), sleeps.append)
    assert rc == 2 and sleeps == [2, 4, 8, 16] and "retries exhausted" in log, (rc, sleeps, log)
    out, sleeps = tmp + "/o5.jsonl", []
    rc, log = go(ns(out), _fake(lambda b, n: (401, None, "bad key " + key)), sleeps.append)
    assert rc == 2 and sleeps == [] and "HTTP 401" in log, (rc, sleeps, log)

    # 5. the key never reaches the output or the log, even when the server echoes it
    assert key not in log, log
    for name in os.listdir(tmp):
        assert key not in open(os.path.join(tmp, name)).read(), name

    # 6. a tool seed: tool call, the seed's result, the final answer
    def tool_script(b, n):
        if not any(m["role"] == "tool" for m in b["messages"]):
            return _ok("", 0.01, tool_calls=[{"id": "c1", "type": "function", "function":
                                              {"name": "get_weather", "arguments": "{\"city\":\"Seoul\"}"}}])
        return _ok("It is 18 degrees.", 0.01)
    out = tmp + "/o6.jsonl"
    go(ns(out), _fake(tool_script))
    t0 = {x["id"]: x for x in rows(out)}["t0"]
    assert [m["role"] for m in t0["messages"]] == ["user", "assistant", "tool", "assistant"], t0["messages"]
    assert t0["messages"][2]["content"] == "{\"t\":18}" and t0["tools"], t0

    # 7. an answer cut at the token cap is kept and marked; an empty one is dropped
    out = tmp + "/o7.jsonl"
    go(ns(out, limit=1), _fake(lambda b, n: _ok("partial", 0.01, finish="length")))
    assert rows(out)[0]["truncated"] is True
    out = tmp + "/o8.jsonl"
    go(ns(out, limit=1), _fake(lambda b, n: _ok("", 0.01, finish="length")))
    assert rows(out) == []
    print("k31-calib-gen self-test: ok")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", default="tools/ref/k31-calib-seeds.jsonl")
    ap.add_argument("--out")
    ap.add_argument("--budget-usd", type=float, default=3.50)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="at most this many seeds in this run")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.out:
        ap.error("--out FILE is required")
    return generate(a, http_transport, read_key())


if __name__ == "__main__":
    sys.exit(main())
