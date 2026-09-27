"""Reference renders of the GLM-5.3-Flash chat template for crates/serve/tests/fixtures/glm5-renders.json.

usage (on the box, jinja2 3.1.2):
    python3 render_glm5.py glm5-chat-template.jinja > glm5-renders.json

The environment is ik_llama.cpp tests/test-jinja.cpp's py_script, line for line: SandboxedEnvironment with
trim_blocks, lstrip_blocks and the loopcontrols extension; `tojson` is json.dumps with ensure_ascii=False,
indent=None, separators=None, sort_keys=False as defaults; globals strftime_now and raise_exception.

The variables are the ones bloomery-serve's render_chat (crates/serve/src/api.rs) passes for a body: messages
(a content array flattened to its text parts joined by "\n"), add_generation_prompt (the body's boolean, else
true), bos_token and eos_token (the mock engine's texts), tools and reasoning_effort when present and not null,
and chat_template_kwargs merged on top.

Every object in a body is written with its keys sorted, at every level: bloomery's template engine iterates
object keys in sorted order (serde_json's map) where jinja2 keeps insertion order, so a body in sorted order is
one on which the two agree.
"""
import json
import sys
from datetime import datetime

import jinja2
import jinja2.ext as jinja2_ext
from jinja2.sandbox import SandboxedEnvironment

BOS = "<｜begin▁of▁sentence｜>"
EOS = "<｜end▁of▁sentence｜>"

env = SandboxedEnvironment(
    trim_blocks=True,
    lstrip_blocks=True,
    extensions=[jinja2_ext.loopcontrols],
)


def raise_exception(message):
    raise jinja2.exceptions.TemplateError(message)


env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(
    x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys
)
env.globals["strftime_now"] = lambda format: datetime.now().strftime(format)
env.globals["raise_exception"] = raise_exception

WEATHER = {
    "function": {
        "description": "Current weather for a city.",
        "name": "get_weather",
        "parameters": {
            "properties": {
                "days": {"description": "How many days ahead.", "type": "integer"},
                "location": {"description": "City name, e.g. Seoul.", "type": "string"},
            },
            "required": ["location"],
            "type": "object",
        },
        "strict": True,
    },
    "type": "function",
}
TIME = {
    "function": {
        "description": "The time now.",
        "name": "get_time",
        "parameters": {"properties": {}, "type": "object"},
    },
    "type": "function",
}
NOTE = {
    "function": {
        "description": "Save a note.",
        "name": "save_note",
        "parameters": {
            "properties": {
                "filter": {"type": "object"},
                "text": {"type": "string"},
            },
            "type": "object",
        },
    },
    "type": "function",
}


def call(id_, name, arguments):
    return {"function": {"arguments": arguments, "name": name}, "id": id_, "type": "function"}


CASES = [
    ("system and user", {"messages": [
        {"content": "You are terse.", "role": "system"},
        {"content": "Hi", "role": "user"},
    ]}),
    ("three turns", {"messages": [
        {"content": "What is the capital of France?", "role": "user"},
        {"content": "Paris.", "role": "assistant"},
        {"content": "And of Italy?", "role": "user"},
    ]}),
    ("past reasoning kept, clear_thinking absent", {"messages": [
        {"content": "Hi", "role": "user"},
        {"content": "Hello!", "reasoning_content": "The user greets.", "role": "assistant"},
        {"content": "Again?", "role": "user"},
    ]}),
    ("past reasoning cleared, clear_thinking true", {"chat_template_kwargs": {"clear_thinking": True}, "messages": [
        {"content": "Hi", "role": "user"},
        {"content": "Hello!", "reasoning_content": "The user greets.", "role": "assistant"},
        {"content": "Again?", "role": "user"},
    ]}),
    ("think span in the answer, no generation prompt", {"add_generation_prompt": False, "messages": [
        {"content": "2+2?", "role": "user"},
        {"content": "<think>Add them.</think>\n\n4", "role": "assistant"},
    ]}),
    ("reasoning_effort low", {"messages": [{"content": "Hi", "role": "user"}], "reasoning_effort": "low"}),
    ("reasoning_effort outside the list", {"messages": [{"content": "Hi", "role": "user"}], "reasoning_effort": "medium"}),
    ("tools and a user turn", {"messages": [{"content": "Weather in Seoul?", "role": "user"}], "tools": [WEATHER, TIME]}),
    ("two calls, their results in the other order", {"messages": [
        {"content": "Weather and time in Seoul?", "role": "user"},
        {"content": "Checking both.", "role": "assistant", "tool_calls": [
            call("call_a", "get_weather", {"days": 3, "location": "Seoul"}),
            call("call_b", "get_time", {}),
        ]},
        {"content": "12:00", "role": "tool", "tool_call_id": "call_b"},
        {"content": "{\"temp\": 21}", "role": "tool", "tool_call_id": "call_a"},
    ], "tools": [WEATHER, TIME]}),
    ("a tool result without an id", {"messages": [
        {"content": "Time?", "role": "user"},
        {"content": "", "role": "assistant", "tool_calls": [call("call_c", "get_time", {})]},
        {"content": "12:00", "role": "tool"},
        {"content": "Thanks.", "role": "user"},
    ], "tools": [TIME]}),
    ("non-ASCII and nested arguments", {"add_generation_prompt": False, "messages": [
        {"content": "Note it.", "role": "user"},
        {"content": "", "role": "assistant", "tool_calls": [
            call("call_d", "save_note", {"filter": {"n": [1, 2], "q": "서울 ☃"}, "text": "a<b & \"c\""}),
        ]},
    ], "tools": [NOTE]}),
    ("string arguments", {"messages": [
        {"content": "Time?", "role": "user"},
        {"content": "", "role": "assistant", "tool_calls": [call("call_e", "get_time", "{}")]},
    ], "tools": [TIME]}),
]


def variables(body):
    msgs = []
    for m in body["messages"]:
        m = dict(m)
        if isinstance(m.get("content"), list):
            m["content"] = "\n".join(
                p["text"] for p in m["content"] if p.get("type") == "text" and isinstance(p.get("text"), str)
            )
        msgs.append(m)
    agp = body.get("add_generation_prompt")
    v = {
        "messages": msgs,
        "add_generation_prompt": agp if isinstance(agp, bool) else True,
        "bos_token": BOS,
        "eos_token": EOS,
    }
    for k in ("tools", "reasoning_effort"):
        if body.get(k) is not None:
            v[k] = body[k]
    if isinstance(body.get("chat_template_kwargs"), dict):
        v.update(body["chat_template_kwargs"])
    return v


def main():
    source = open(sys.argv[1], encoding="utf-8").read()
    template = env.from_string(source)
    cases = []
    for name, body in CASES:
        # The body as the test sends it: keys sorted at every level.
        body = json.loads(json.dumps(body, sort_keys=True))
        case = {"name": name, "body": body}
        try:
            case["prompt"] = template.render(**variables(body))
        except Exception as e:  # jinja2 raised: the case records its error
            case["error"] = f"{type(e).__name__}: {e}"
        cases.append(case)
    out = {
        "template": None,
        "reference": None,
        "cases": cases,
    }
    sys.stdout.write(json.dumps(out, ensure_ascii=False, indent=2) + "\n")


main()
