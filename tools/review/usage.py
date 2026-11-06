#!/usr/bin/env python3
"""Token and tool accounting for one review bot run, from Claude Code's stream-json transcript (the claude and glm
engines) or Codex's JSONL events.

    tools/review/usage.py ROLE TRANSCRIPT.ndjson OUT.json [glm|claude|codex]
    tools/review/usage.py --self-test

Writes OUT.json and prints one line. The fields are what a later optimization needs: the engine, how the run ended,
its turns, cost and wall time; tokens by kind (input, cache read, cache write, output); the context each request
carried (max and sum: every turn re-sends it, so the sum is what the run paid for); tool calls by tool, denials, and
the files read most often with how many times; the last error message, whole (tools/review/run.py removes
credentials from it before it shortens it). A Codex stream (`codex exec --json`) reports only the turn's summed tokens
and no cost (a ChatGPT login has no per-run price), so its requests, context_max, turns, cost and reads are None; its
tools are the item types, and a command the sandbox refused counts as a denial. Without ENGINE, a stream with Codex's
thread or turn events is Codex's. A glm stream (Claude Code on z.ai) streams every request's usage as zeros and
holds the run's totals only in its result, so its context_max is None and its context_sum the result's total, and its
cost_usd is None: the CLI's figure is its own price list applied to a model it does not know, and the plan is billed in
prompts, not tokens. Exit 1 when the transcript has no result event (Codex: no completed turn) or the
result is an error; OUT.json is written either way.
"""

import collections
import json
import re
import sys
import tempfile


def events(lines):
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            ev = json.loads(raw)
        except ValueError:
            continue
        if isinstance(ev, dict):
            yield ev


def summarize_codex(role, evs):
    tools = collections.Counter()
    denied = 0
    done = failed = False
    usage, error = {}, None
    for ev in evs:
        kind = ev.get("type")
        item = ev.get("item") or {}
        if kind == "item.completed" and item.get("type") not in (None, "agent_message", "reasoning", "todo_list",
                                                                 "error"):
            tools[item["type"]] += 1
            refused = r"Operation not permitted|Permission denied|Read-only file system"
            if item["type"] == "command_execution" and item.get("exit_code") != 0 and re.search(
                    refused, item.get("aggregated_output") or ""):
                denied += 1
        elif kind == "item.completed" and item.get("type") == "error" and re.match(
                r"Configured service tier `.*` is not advertised", item.get("message") or "", flags=re.DOTALL):
            # Codex drops a tier the model does not offer and runs on at the default one (REVIEW_CODEX_TIER).
            failed, error = True, item["message"]
        elif kind == "turn.completed":
            done, usage = True, ev.get("usage") or {}
        elif kind == "turn.failed":
            failed, error = True, (ev.get("error") or {}).get("message")
        elif kind == "error":
            error = ev.get("message")
    total, cached = usage.get("input_tokens"), usage.get("cached_input_tokens") or 0
    return {
        "role": role,
        "engine": "codex",
        "subtype": "error" if failed else "success" if done else "no-result",
        "is_error": failed or not done,
        "error": error or None,
        "num_turns": None,
        "duration_ms": None,
        "cost_usd": None,
        "tokens": {
            "input": None if total is None else total - cached,
            "cache_read": usage.get("cached_input_tokens"),
            "cache_write": usage.get("cache_write_input_tokens"),
            "output": usage.get("output_tokens"),
            "reasoning": usage.get("reasoning_output_tokens"),
        },
        "requests": None,
        "context_max": None,
        "context_sum": total,
        "tools": dict(tools.most_common()),
        "tool_calls": sum(tools.values()),
        "denied": denied,
        "reads_top": [],
        "reads_distinct": None,
    }


def summarize(role, lines, engine=None):
    evs = list(events(lines))
    if engine == "codex" or engine is None and any(e.get("type") in ("thread.started", "turn.started") for e in evs):
        return summarize_codex(role, evs)
    tools = collections.Counter()
    reads = collections.Counter()
    denied = 0
    contexts = []
    result = None
    seen_msgs = set()
    for ev in evs:
        kind = ev.get("type")
        if kind == "assistant":
            msg = ev.get("message", {})
            mid = msg.get("id")
            u = msg.get("usage") or {}
            # One API response can arrive as several events (one per content block) with the same id and usage.
            if mid not in seen_msgs:
                seen_msgs.add(mid)
                contexts.append(u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0)
                                + u.get("cache_creation_input_tokens", 0))
            for block in msg.get("content") or []:
                if block.get("type") == "tool_use":
                    name = block.get("name", "?")
                    tools[name] += 1
                    inp = block.get("input") or {}
                    if name == "Read" and inp.get("file_path"):
                        reads[inp["file_path"]] += 1
        elif kind == "user":
            for block in (ev.get("message", {}).get("content") or []):
                if isinstance(block, dict) and block.get("type") == "tool_result" and block.get("is_error"):
                    text = json.dumps(block.get("content", ""))
                    if "permission" in text.lower() or "denied" in text.lower():
                        denied += 1
        elif kind == "result":
            result = ev
    r = result or {}
    u = r.get("usage") or {}
    is_error = bool(r.get("is_error", result is None))
    total = sum(u.get(k) or 0 for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
    unreported = bool(contexts) and not any(contexts) and total > 0
    return {
        "role": role,
        "engine": engine or "claude",
        "subtype": r.get("subtype", "no-result"),
        "is_error": is_error,
        "error": (str(r.get("result") or "") or None) if is_error else None,
        "num_turns": r.get("num_turns"),
        "duration_ms": r.get("duration_ms"),
        "cost_usd": None if engine == "glm" else r.get("total_cost_usd"),
        "tokens": {
            "input": u.get("input_tokens"),
            "cache_read": u.get("cache_read_input_tokens"),
            "cache_write": u.get("cache_creation_input_tokens"),
            "output": u.get("output_tokens"),
        },
        "requests": len(contexts),
        "context_max": None if unreported else max(contexts, default=0),
        "context_sum": total if unreported else sum(contexts),
        "tools": dict(tools.most_common()),
        "tool_calls": sum(tools.values()),
        "denied": denied,
        "reads_top": [{"path": p, "count": n} for p, n in reads.most_common(15)],
        "reads_distinct": len(reads),
    }


def line(s):
    tools = ",".join(f"{k}:{v}" for k, v in s["tools"].items()) or "none"
    cost = s["cost_usd"]
    return (f"{s['role']}: {s['engine']} {s['subtype']} turns={s['num_turns']} "
            f"requests={s['requests']} "
            f"cost_usd={cost if cost is None else round(cost, 4)} ctx_max={s['context_max']} ctx_sum={s['context_sum']} "
            f"out={s['tokens']['output']} tools={tools} denied={s['denied']} reads={s['reads_distinct']}")


def self_test():
    ev = [
        {"type": "system", "subtype": "init"},
        {"type": "assistant", "message": {"id": "m1", "usage": {"input_tokens": 10, "cache_read_input_tokens": 1000,
                                                                "cache_creation_input_tokens": 50},
                                          "content": [{"type": "tool_use", "name": "Read",
                                                       "input": {"file_path": "/a.rs"}}]}},
        # The same response's second block: same id, counted once for context.
        {"type": "assistant", "message": {"id": "m1", "usage": {"input_tokens": 10, "cache_read_input_tokens": 1000,
                                                                "cache_creation_input_tokens": 50},
                                          "content": [{"type": "tool_use", "name": "Grep", "input": {}}]}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "is_error": True,
                                                  "content": "Permission to read /etc/x was denied"}]}},
        {"type": "assistant", "message": {"id": "m2", "usage": {"input_tokens": 5, "cache_read_input_tokens": 2000},
                                          "content": [{"type": "tool_use", "name": "Read",
                                                       "input": {"file_path": "/a.rs"}},
                                                      {"type": "text", "text": "done"}]}},
        {"type": "result", "subtype": "success", "is_error": False, "num_turns": 3, "duration_ms": 1200,
         "total_cost_usd": 0.0123, "usage": {"input_tokens": 15, "cache_read_input_tokens": 3000,
                                             "cache_creation_input_tokens": 50, "output_tokens": 77}},
    ]
    s = summarize("reviewer", [json.dumps(e) for e in ev] + ["not json"])
    assert s["requests"] == 2 and s["context_max"] == 2005 and s["context_sum"] == 1060 + 2005, s
    assert s["tools"] == {"Read": 2, "Grep": 1} and s["tool_calls"] == 3 and s["denied"] == 1, s
    assert s["reads_top"] == [{"path": "/a.rs", "count": 2}] and s["tokens"]["output"] == 77, s
    assert "tools=Read:2,Grep:1" in line(s) and "cost_usd=0.0123" in line(s), line(s)
    none = summarize("validator", [])
    assert none["subtype"] == "no-result" and none["is_error"], none
    cx = [{"type": "thread.started", "thread_id": "t"}, {"type": "turn.started"},
          {"type": "item.completed", "item": {"type": "agent_message", "text": "x"}},
          {"type": "item.completed", "item": {"type": "command_execution", "exit_code": 0, "aggregated_output": "a"}},
          {"type": "item.completed", "item": {"type": "command_execution", "exit_code": 1,
                                              "aggregated_output": "cat: /h/auth.json: Operation not permitted"}},
          {"type": "item.completed", "item": {"type": "file_change", "status": "completed"}},
          {"type": "turn.completed", "usage": {"input_tokens": 1000, "cached_input_tokens": 800,
                                               "cache_write_input_tokens": 0, "output_tokens": 50}}]
    c = summarize("reviewer", [json.dumps(e) for e in cx])
    assert (c["engine"], c["subtype"], c["is_error"], c["cost_usd"]) == ("codex", "success", False, None), c
    assert c["tools"] == {"command_execution": 2, "file_change": 1} and c["denied"] == 1, c
    assert c["tokens"]["input"] == 200 and c["context_sum"] == 1000 and "codex success" in line(c), c
    cf = summarize("reviewer", [json.dumps(e) for e in cx[:2] + [
        {"type": "turn.failed", "error": {"message": "You've hit your usage limit."}}]])
    assert cf["is_error"] and cf["subtype"] == "error" and "usage limit" in cf["error"], cf
    tier = {"type": "error", "message": "Configured service tier `bo\ngus` is not advertised as supported for model "
                                        "`m` and will be omitted from requests."}
    ct = summarize("reviewer", [json.dumps(e) for e in cx[:2] + [{"type": "item.completed", "item": tier}] + cx[2:]])
    assert ct["is_error"] and ct["subtype"] == "error" and "`bo\ngus`" in ct["error"], ct
    # z.ai through Claude Code: every request's usage streams as zeros, the totals are in the result, and the cost is
    # the CLI's guess. The same stream under the claude label keeps that cost; real per-request usage keeps its numbers.
    zeros = [{"type": "assistant", "message": {"id": f"m{i}", "usage": {"input_tokens": 0, "output_tokens": 0},
                                               "content": [{"type": "text", "text": "x"}]}} for i in (1, 2)]
    zres = {"type": "result", "subtype": "success", "is_error": False, "num_turns": 2, "total_cost_usd": 0.0487,
            "usage": {"input_tokens": 9533, "cache_read_input_tokens": 1344, "cache_creation_input_tokens": 0,
                      "output_tokens": 16}}
    z = summarize("reviewer", [json.dumps(e) for e in zeros + [zres]], "glm")
    assert (z["engine"], z["cost_usd"], z["context_max"], z["context_sum"], z["requests"]) == ("glm", None, None, 10877, 2), z
    assert "glm success" in line(z) and "cost_usd=None ctx_max=None ctx_sum=10877" in line(z), line(z)
    zc = summarize("reviewer", [json.dumps(e) for e in zeros + [zres]], "claude")
    assert zc["cost_usd"] == 0.0487 and zc["context_sum"] == 10877, zc
    assert summarize("reviewer", [json.dumps(e) for e in ev], "glm")["context_max"] == 2005
    # A Codex run that died before its first event is still Codex's.
    assert summarize("reviewer", ["Reading additional input"], "codex")["engine"] == "codex"
    with tempfile.NamedTemporaryFile("w", suffix=".ndjson", delete=False) as f:
        f.write("\n".join(json.dumps(e) for e in ev))
    out = f.name + ".json"
    assert main(["usage.py", "reviewer", f.name, out]) == 0
    assert json.load(open(out))["num_turns"] == 3 and json.load(open(out))["engine"] == "claude"
    assert main(["usage.py", "reviewer", f.name, out, "glm"]) == 0 and json.load(open(out))["engine"] == "glm"
    assert main(["usage.py", "reviewer", f.name, out, "gpt"]) == 64
    print("usage.py self-test: ok")


def main(argv):
    if argv[1:] == ["--self-test"]:
        self_test()
        return 0
    if len(argv) not in (4, 5) or argv[4:] not in ([], ["glm"], ["claude"], ["codex"]):
        print(__doc__, file=sys.stderr)
        return 64
    try:
        lines = open(argv[2], encoding="utf-8", errors="replace").readlines()
    except OSError:
        lines = []
    s = summarize(argv[1], lines, argv[4] if len(argv) == 5 else None)
    with open(argv[3], "w", encoding="utf-8") as f:
        json.dump(s, f, indent=1)
    print(line(s))
    return 1 if s["is_error"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
