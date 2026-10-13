#!/usr/bin/env python3
"""Token and tool accounting for one review bot run, from Claude Code's stream-json transcript.

    tools/review/usage.py ROLE TRANSCRIPT.ndjson OUT.json
    tools/review/usage.py --self-test

Writes OUT.json and prints one line. The fields are what a later optimization needs: how the run ended, its turns,
cost and wall time; tokens by kind (input, cache read, cache write, output); the context each request carried (max
and sum: every turn re-sends it, so the sum is what the run paid for); tool calls by tool, denials, and the files
read most often with how many times. Exit 1 when the transcript has no result event or the result is an error;
OUT.json is written either way.
"""

import collections
import json
import sys
import tempfile


def summarize(role, lines):
    tools = collections.Counter()
    reads = collections.Counter()
    denied = 0
    contexts = []
    result = None
    seen_msgs = set()
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            ev = json.loads(raw)
        except ValueError:
            continue
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
    return {
        "role": role,
        "subtype": r.get("subtype", "no-result"),
        "is_error": bool(r.get("is_error", result is None)),
        "num_turns": r.get("num_turns"),
        "duration_ms": r.get("duration_ms"),
        "cost_usd": r.get("total_cost_usd"),
        "tokens": {
            "input": u.get("input_tokens"),
            "cache_read": u.get("cache_read_input_tokens"),
            "cache_write": u.get("cache_creation_input_tokens"),
            "output": u.get("output_tokens"),
        },
        "requests": len(contexts),
        "context_max": max(contexts, default=0),
        "context_sum": sum(contexts),
        "tools": dict(tools.most_common()),
        "tool_calls": sum(tools.values()),
        "denied": denied,
        "reads_top": [{"path": p, "count": n} for p, n in reads.most_common(15)],
        "reads_distinct": len(reads),
    }


def line(s):
    tools = ",".join(f"{k}:{v}" for k, v in s["tools"].items()) or "none"
    cost = s["cost_usd"]
    return (f"{s['role']}: {s['subtype']} turns={s['num_turns']} requests={s['requests']} "
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
    with tempfile.NamedTemporaryFile("w", suffix=".ndjson", delete=False) as f:
        f.write("\n".join(json.dumps(e) for e in ev))
    out = f.name + ".json"
    assert main(["usage.py", "reviewer", f.name, out]) == 0
    assert json.load(open(out))["num_turns"] == 3
    print("usage.py self-test: ok")


def main(argv):
    if argv[1:] == ["--self-test"]:
        self_test()
        return 0
    if len(argv) != 4:
        print(__doc__, file=sys.stderr)
        return 64
    try:
        lines = open(argv[2], encoding="utf-8", errors="replace").readlines()
    except OSError:
        lines = []
    s = summarize(argv[1], lines)
    with open(argv[3], "w", encoding="utf-8") as f:
        json.dump(s, f, indent=1)
    print(line(s))
    return 1 if s["is_error"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
