"""Reference renders of the Solar-Open2 chat template for crates/serve/tests/fixtures/solar-renders.json.

usage (jinja2 3.1.x; the chat template is the file's own, `tokenizer.chat_template`, byte for byte Upstage's
chat_template.jinja):
    python3 render_solar.py solar-chat-template.jinja > solar-renders.json

The environment is ik_llama.cpp tests/test-jinja.cpp's py_script, line for line: SandboxedEnvironment with
trim_blocks, lstrip_blocks and the loopcontrols extension; `tojson` is json.dumps with ensure_ascii=False,
indent=None, separators=None, sort_keys=False as defaults; globals strftime_now and raise_exception.

The variables are the ones bloomery-serve's render_chat (crates/serve/src/api.rs) passes for a body: messages
(a content array flattened to its text parts joined by "\n"), add_generation_prompt (the body's boolean, else
true), bos_token and eos_token (the mock engine's texts), tools and reasoning_effort when present and not null,
and chat_template_kwargs merged on top. The one rule the server applies on top is thinking_vars' for a template
with an effort knob and no thinking switch (Solar's): reasoning_effort "none" reaches the template as "low".

Bodies keep their keys in the order written here: the template walks a tool call's arguments with `|items`, so
the order is part of the render, and bloomery's template engine keeps a body's insertion order (serde_json's
`preserve_order`), as jinja2 does. `provider_system_prompt` is not covered: it prints today's date.
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
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name, e.g. Seoul."},
                "days": {"type": "integer", "description": "How many days ahead."},
            },
            "required": ["location"],
        },
    },
}
TIME = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "The time now.",
        "parameters": {"type": "object", "properties": {}},
    },
}
NOTE = {
    "type": "function",
    "function": {
        "name": "save_note",
        "description": "Save a note.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}, "filter": {"type": "object"}},
        },
    },
}
SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "temp": {"type": "number"}},
            "required": ["city"],
        },
    },
}
HI = [{"role": "user", "content": "Hi"}]


def call(id_, name, arguments):
    return {"id": id_, "type": "function", "function": {"name": name, "arguments": arguments}}


def effort(name, **body):
    return (name, {"messages": HI, **body})


TURNS = [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!", "reasoning_content": "The user greets."},
    {"role": "user", "content": "Again?"},
    {"role": "assistant", "content": "Hello again.", "reasoning_content": "The user repeats."},
    {"role": "user", "content": "One more."},
]

CASES = [
    ("system and user", {"messages": [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Hi"},
    ]}),
    ("a user turn alone", {"messages": HI}),
    ("two system messages", {"messages": [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Hi"},
        {"role": "system", "content": "Answer in Korean."},
    ]}),
    ("content parts flattened", {"messages": [
        {"role": "system", "content": [{"type": "text", "text": "Be brief."}, {"type": "text", "text": "No lists."}]},
        {"role": "user", "content": [{"type": "text", "text": "서울 날씨? ☃ \"quoted\" <b>"}]},
    ]}),
    ("three turns, past reasoning dropped", {"messages": TURNS}),
    ("three turns, interleaved named", {"chat_template_kwargs": {"think_render_option": "interleaved"}, "messages": TURNS}),
    ("three turns, preserved", {"chat_template_kwargs": {"think_render_option": "preserved"}, "messages": TURNS}),
    ("an unknown think_render_option renders no span", {"chat_template_kwargs": {"think_render_option": "off"}, "messages": TURNS}),
    ("reasoning after the last user turn is kept", {"messages": [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4", "reasoning_content": "Add them."},
    ], "add_generation_prompt": False}),
    ("reasoning field wins over reasoning_content", {"chat_template_kwargs": {"think_render_option": "preserved"}, "messages": [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4", "reasoning": "first", "reasoning_content": "second"},
        {"role": "user", "content": "3+3?"},
    ]}),
    ("empty and null reasoning, preserved", {"chat_template_kwargs": {"think_render_option": "preserved"}, "messages": [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b", "reasoning_content": ""},
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "d", "reasoning_content": None},
        {"role": "user", "content": "e"},
        {"role": "assistant", "content": "f"},
    ], "add_generation_prompt": False}),
    ("assistant first, no user turn", {"messages": [
        {"role": "assistant", "content": "Hello", "reasoning_content": "Greet."},
        {"role": "assistant", "content": "Again", "reasoning_content": "Repeat."},
    ], "add_generation_prompt": False}),
    ("no generation prompt", {"messages": HI, "add_generation_prompt": False}),
    effort("reasoning_effort low", reasoning_effort="low"),
    effort("reasoning_effort minimal", reasoning_effort="minimal"),
    effort("reasoning_effort medium", reasoning_effort="medium"),
    effort("reasoning_effort high", reasoning_effort="high"),
    effort("reasoning_effort xhigh", reasoning_effort="xhigh"),
    effort("reasoning_effort max", reasoning_effort="max"),
    effort("reasoning_effort none is sent as low", reasoning_effort="none"),
    effort("kwargs reasoning_effort medium", chat_template_kwargs={"reasoning_effort": "medium"}),
    effort("kwargs reasoning_effort low", chat_template_kwargs={"reasoning_effort": "low"}),
    effort("kwargs reasoning_effort unknown", chat_template_kwargs={"reasoning_effort": "turbo"}),
    ("tools and a user turn", {"messages": [{"role": "user", "content": "Weather in Seoul?"}], "tools": [WEATHER, TIME]}),
    ("system, tools and a user turn", {"messages": [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Weather in Seoul?"},
    ], "tools": [WEATHER]}),
    ("tools and a json_schema response_format", {"chat_template_kwargs": {"response_format": SCHEMA}, "messages": [
        {"role": "user", "content": "Weather in Seoul?"},
    ], "tools": [WEATHER]}),
    ("a json_object response_format alone", {"chat_template_kwargs": {"response_format": {"type": "json_object"}}, "messages": HI}),
    ("a json_schema response_format alone", {"chat_template_kwargs": {"response_format": SCHEMA}, "messages": HI}),
    ("two calls, their results in the other order", {"messages": [
        {"role": "user", "content": "Weather and time in Seoul?"},
        {"role": "assistant", "content": "Checking both.", "reasoning_content": "Two lookups.", "tool_calls": [
            call("call_a", "get_weather", {"location": "Seoul", "days": 3}),
            call("call_b", "get_time", {}),
        ]},
        {"role": "tool", "tool_call_id": "call_b", "content": "12:00"},
        {"role": "tool", "tool_call_id": "call_a", "content": "{\"temp\": 21}"},
    ], "tools": [WEATHER, TIME]}),
    ("a tool result with no matching call", {"messages": [
        {"role": "user", "content": "Time?"},
        {"role": "assistant", "content": "", "tool_calls": [call("call_a", "get_time", {})]},
        {"role": "tool", "tool_call_id": "call_zzz", "content": "orphan"},
        {"role": "tool", "tool_call_id": "call_a", "content": "12:00"},
        {"role": "user", "content": "Thanks."},
    ], "tools": [TIME]}),
    ("tool results after no assistant call", {"messages": [
        {"role": "user", "content": "Time?"},
        {"role": "tool", "tool_call_id": "x1", "content": "one"},
        {"role": "tool", "tool_call_id": "x2", "content": "two"},
        {"role": "user", "content": "Thanks."},
    ]}),
    ("two tool blocks", {"messages": [
        {"role": "user", "content": "Go."},
        {"role": "assistant", "tool_calls": [call("c1", "get_time", {})]},
        {"role": "tool", "tool_call_id": "c1", "content": "12:00"},
        {"role": "assistant", "content": "ok", "tool_calls": [call("c2", "get_time", {}), call("c3", "get_time", {})]},
        {"role": "tool", "tool_call_id": "c3", "content": "13:00"},
        {"role": "tool", "tool_call_id": "c2", "content": "12:30"},
    ], "tools": [TIME]}),
    ("arguments of every kind, in written order", {"messages": [
        {"role": "user", "content": "Note it."},
        {"role": "assistant", "content": "", "tool_calls": [
            call("call_d", "save_note", {
                "text": "a<b & \"c\"\n서울 ☃",
                "filter": {"q": "서울 ☃", "n": [1, 2.5, None, True]},
                "count": 3,
                "ratio": 0.5,
                "flag": False,
                "nothing": None,
                "list": ["x", {"y": 1}],
                "alpha": "bare",
            }),
        ]},
    ], "tools": [NOTE], "add_generation_prompt": False}),
    ("a call with no arguments", {"messages": [
        {"role": "user", "content": "Time?"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_time"}}]},
    ], "tools": [TIME], "add_generation_prompt": False}),
    ("string arguments", {"messages": [
        {"role": "user", "content": "Time?"},
        {"role": "assistant", "content": "", "tool_calls": [call("call_e", "get_time", "{}")]},
    ], "tools": [TIME]}),
    ("a tool result without an id", {"messages": [
        {"role": "user", "content": "Time?"},
        {"role": "assistant", "content": "", "tool_calls": [call("call_c", "get_time", {})]},
        {"role": "tool", "content": "12:00"},
    ], "tools": [TIME]}),
    ("a call without an id", {"messages": [
        {"role": "user", "content": "Time?"},
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "get_time", "arguments": {}}}]},
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
    if body.get("tools") is not None:
        v["tools"] = body["tools"]
    if body.get("reasoning_effort") is not None:
        # thinking_vars: a template with an effort knob and no thinking switch gets "none" as its lowest effort.
        e = body["reasoning_effort"]
        v["reasoning_effort"] = "low" if e == "none" else e
    if isinstance(body.get("chat_template_kwargs"), dict):
        v.update(body["chat_template_kwargs"])
    return v


def main():
    source = open(sys.argv[1], encoding="utf-8").read()
    template = env.from_string(source)
    cases = []
    for name, body in CASES:
        case = {"name": name, "body": body}
        try:
            case["prompt"] = template.render(**variables(body))
        except Exception as e:  # jinja2 raised: the case records its error
            case["error"] = f"{type(e).__name__}: {e}"
        cases.append(case)
    out = {
        "template": "solar-chat-template.jinja",
        "reference": (
            f"jinja2 {jinja2.__version__} (Python {sys.version.split()[0]}), the HF chat-template environment of "
            "ik_llama.cpp tests/test-jinja.cpp (SandboxedEnvironment with trim_blocks, lstrip_blocks and the "
            "loopcontrols extension; tojson = json.dumps with ensure_ascii=False, indent=None, separators=None, "
            "sort_keys=False as defaults; globals strftime_now and raise_exception); each case rendered with the "
            "variables bloomery-serve's /apply-template passes (messages with a content array flattened to its text "
            "parts, add_generation_prompt default true, the mock's bos_token and eos_token, tools and "
            "reasoning_effort when not null, chat_template_kwargs merged; reasoning_effort none sent as low); a "
            "body keeps its keys in written order; a case jinja2 refused carries its error in place of a prompt. "
            "Script: crates/serve/tests/fixtures/render_solar.py, run as "
            "`python3 render_solar.py solar-chat-template.jinja > solar-renders.json`"
        ),
        "cases": cases,
    }
    sys.stdout.write(json.dumps(out, ensure_ascii=False, indent=2) + "\n")


main()
