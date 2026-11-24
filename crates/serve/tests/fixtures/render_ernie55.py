"""Reference renders of ERNIE 5.5 Flash Preview's chat protocol for crates/serve/tests/fixtures/ernie55-renders.json.

usage:
    python3 -I render_ernie55.py <vendor dir> <tokenizer.json> > ernie55-renders.json

`<vendor dir>` holds a copy of the checkpoint folder's `encoding_ernie.py` (baidu/ERNIE-5.5-Flash-Preview), the
vendor's Python renderer: it is imported from there, never run from the model folder. `<tokenizer.json>` is the
folder's own (the fixture cut's is the same file): its `added_tokens` name the ids the renderer's control segments
become. The vendor writes no Jinja template, so this is not a jinja2 render: each case is the vendor's
`render_chat` on the request as bloomery-serve hands a template one (crates/serve/src/api.rs, render_chat): a
content array flattened to its text parts joined by "\n", a `developer` role mapped to `system`, the OpenAI
`reasoning_effort` as the renderer's keyword (the server sends "none" as enable_thinking false), and
chat_template_kwargs merged on top as the renderer's other keywords (enable_thinking, options, system_settings,
agent_storage, examples, sort_keys). Every object in a body is written with its keys sorted, at every level, as the
other fixtures are.

A case holds `prompt`, the vendor's text, and `runs` (where the server's ids are the vendor's): the token structure of its segments, as the vendor's
`encode_segments` splits them (tokenization_ernie.py) before the BPE runs: a string is an ordinary text run, a number
an added token's id. The test holds bloomery-serve's `/apply-template` ids to the runs, each ordinary run encoded as
text with no token matching. A case the vendor refuses holds `error` (its exception) and `refusal`, the words the
server's 400 names it by. `vendor_differs` marks a case whose prompt text is the vendor's but whose ids are not (the
reason in the field); it carries no runs.
"""
import importlib
import json
import os
import re
import sys

MISSING = object()
EFFORTS = ("low", "medium", "high")

WEATHER = {
    "function": {
        "description": "Current weather for a city.",
        "name": "get_weather",
        "parameters": {
            "properties": {
                "days": {"description": "How many days ahead.", "type": "integer"},
                "location": {"description": "City name, e.g. Seoul.", "type": "string"},
                "unit": {"default": "celsius", "description": "celsius or fahrenheit.", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
            "type": "object",
        },
    },
    "type": "function",
}
TIME = {
    "function": {"description": "The time now.", "name": "get_time", "parameters": {"properties": {}, "type": "object"}},
    "type": "function",
}
NOTE = {
    "function": {
        "description": "Save a note.",
        "name": "save_note",
        "parameters": {
            "properties": {"filter": {"type": "object"}, "text": {"type": "string"}, "tone": {"type": ["string", "null"]}},
            "required": ["text"],
            "type": "object",
        },
    },
    "type": "function",
}
# Numbers as Python's json.dumps spells them: an exponent, a float with a fraction, integers, a negative.
KNOBS = {
    "function": {
        "description": "A knob.\nWith a newline and \"quotes\".",
        "name": "set_knob",
        "parameters": {
            "properties": {
                "gain": {"default": 1.5, "maximum": 2, "minimum": -0.25, "type": "number"},
                "rate": {"default": 1e-05, "type": "number"},
                "size": {"type": "integer"},
                "tags": {"items": {"type": "string"}, "type": "array"},
                "on": {"type": "boolean"},
            },
            "required": ["gain"],
            "type": "object",
        },
    },
    "type": "function",
}
# A tool the way the protocol also takes one: no `function` wrapper.
BARE = {"description": "No wrapper.", "name": "bare", "parameters": {"properties": {}, "type": "object"}, "strict": False}


def call(id_, name, arguments):
    return {"function": {"arguments": arguments, "name": name}, "id": id_, "type": "function"}


def user(text):
    return {"content": text, "role": "user"}


def system(text):
    return {"content": text, "role": "system"}


def assistant(text, **more):
    return {"content": text, "role": "assistant", **more}


def tool(text, id_=None):
    m = {"content": text, "role": "tool"}
    if id_ is not None:
        m["tool_call_id"] = id_
    return m


# (name, body, keep_order): keep_order leaves the body's own key order (a case about sort_keys).
CASES = [
    ("user only", {"messages": [user("Hi")]}),
    ("system and user", {"messages": [system("You are terse."), user("Hi")]}),
    ("three turns", {"messages": [
        user("What is the capital of France?"), assistant("Paris."), user("And of Italy?"),
    ]}),
    ("a system message with no content", {"messages": [{"role": "system"}, user("Hi")]}),
    ("a system message in the middle is a turn", {"messages": [
        user("First."), system("Second."), user("Third."),
    ]}),
    ("no generation prompt ends at the last turn", {"messages": [user("Hi"), assistant("Hello!")], "add_generation_prompt": False}),
    ("no generation prompt after a user turn", {"messages": [user("Hi")], "add_generation_prompt": False}),
    ("reasoning_content of an assistant turn", {"messages": [
        user("Hi"), assistant("Hello!", reasoning_content="The user greets."),
        user("Again?"), assistant("Hello again!", reasoning_content="  "),  # whitespace only: not written
    ]}),
    ("thoughts before reasoning_content", {"messages": [
        user("Hi"), assistant("Hello!", reasoning="unused", reasoning_content="second", thoughts="first"),
    ]}),
    ("inline think tags in an assistant's content", {"messages": [
        user("Hi"), assistant("<ernie:think>The user greets.</ernie:think>Hello!"),
    ]}),
    ("inline think tags: text before the open tag is dropped, the first close ends the reasoning", {"messages": [
        user("Hi"),
        assistant("dropped<ernie:think>kept<ernie:think>still reasoning</ernie:think>first answer<ernie:think>x</ernie:think>last answer"),
    ]}),
    ("an assistant's content with a close tag and no open tag", {"messages": [
        user("Hi"), assistant("the reasoning</ernie:think>the answer"),
    ]}),
    ("thinking off: enable_thinking false in the kwargs", {
        "chat_template_kwargs": {"enable_thinking": False}, "messages": [user("Hi")],
    }),
    ("thinking off: options thinking_mode close", {
        "chat_template_kwargs": {"options": {"thinking_mode": "close"}}, "messages": [user("Hi")],
    }),
    ("thinking off: options thinking_mode false", {
        "chat_template_kwargs": {"options": {"thinking_mode": "false"}}, "messages": [user("Hi")],
    }),
    ("thinking off: enable_thinking 0 in the kwargs", {"chat_template_kwargs": {"enable_thinking": 0}, "messages": [user("Hi")]}),
    ("thinking off: reasoning_effort none on the body", {"messages": [user("Hi")], "reasoning_effort": "none"}),
    ("thinking off drops an assistant's reasoning and reads no effort", {
        "chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "max"},
        "messages": [user("Hi"), assistant("Hello!", reasoning_content="The user greets."), assistant("<ernie:think>inline</ernie:think>Hi")],
    }),
    ("enable_thinking true in the kwargs", {"chat_template_kwargs": {"enable_thinking": True}, "messages": [user("Hi")]}),
    ("reasoning_effort low on the body", {"messages": [user("Hi")], "reasoning_effort": "low"}),
    ("reasoning_effort medium on the body", {"messages": [user("Hi")], "reasoning_effort": "medium"}),
    ("reasoning_effort high on the body", {"messages": [user("Hi")], "reasoning_effort": "high"}),
    ("reasoning_effort through the kwargs", {"chat_template_kwargs": {"reasoning_effort": "low"}, "messages": [user("Hi")]}),
    ("tools offered", {"messages": [user("Weather in Seoul?")], "tools": [WEATHER, TIME]}),
    ("tools and a system turn", {"messages": [system("You help."), user("Weather in Seoul?")], "tools": [WEATHER, BARE]}),
    ("tools with parallel_tool_calls false in the options", {
        "chat_template_kwargs": {"options": {"parallel_tool_calls": False}}, "messages": [user("Weather?")], "tools": [WEATHER],
    }),
    ("tools with parallel_tool_calls the string false in the options", {
        "chat_template_kwargs": {"options": {"parallel_tool_calls": "false"}}, "messages": [user("Weather?")], "tools": [WEATHER],
    }),
    ("tools with parallel_tool_calls true in the options", {
        "chat_template_kwargs": {"options": {"parallel_tool_calls": True}}, "messages": [user("Weather?")], "tools": [WEATHER],
    }),
    ("no tools: parallel_tool_calls writes nothing", {
        "chat_template_kwargs": {"options": {"parallel_tool_calls": True}}, "messages": [user("Weather?")],
    }),
    ("numbers and nesting in a tool schema", {"messages": [user("Set the gain.")], "tools": [KNOBS]}),
    ("an assistant tool call and its result", {"messages": [
        user("Weather in Seoul?"),
        assistant("", reasoning_content="One call.", tool_calls=[call("call_a", "get_weather", {"location": "Seoul", "days": 3})]),
        tool("21 C", "call_a"),
        user("Thanks."),
    ], "tools": [WEATHER]}),
    ("two calls and two results: the tool group", {"messages": [
        user("Weather and time in Seoul?"),
        assistant("", tool_calls=[
            call("call_a", "get_weather", {"location": "Seoul"}),
            call("call_b", "get_time", {}),
        ]),
        tool("12:00", "call_b"),
        tool("{\"temp\": 21, \"sky\": \"맑음\"}", "call_a"),
    ], "tools": [WEATHER, TIME]}),
    ("tool results with no tools offered", {"messages": [
        user("Time?"), assistant("", tool_calls=[call("call_c", "get_time", {})]), tool("12:00"), user("Thanks."),
    ]}),
    ("a tool message first", {"messages": [tool("orphan"), user("Thanks.")]}),
    ("a null assistant content with tool calls", {"messages": [
        user("Time?"), {"content": None, "role": "assistant", "tool_calls": [call("call_d", "get_time", {})]},
    ], "tools": [TIME]}),
    ("tool call arguments as a JSON string", {"messages": [
        user("Weather?"), assistant("", tool_calls=[call("call_e", "get_weather", "{\"location\": \"Seoul\", \"days\": 3}")]),
    ], "tools": [WEATHER]}),
    ("a bare tool call: no function wrapper", {"messages": [
        user("Time?"), assistant("", tool_calls=[{"name": "get_time", "arguments": {"zone": 1}}]),
    ], "tools": [TIME]}),
    ("nested and typed arguments", {"messages": [
        user("Note it."),
        assistant("", tool_calls=[call("call_f", "save_note", {
            "filter": {"n": [1, 2], "q": "서울 ☃", "ok": True, "x": None, "f": 1.5},
            "text": "a<b & \"c\" </parameter x",
        })]),
    ], "tools": [NOTE]}),
    ("the arguments' keys sorted: uppercase before lowercase", {"messages": [
        user("Note it."),
        assistant("", tool_calls=[call("call_g", "save_note", {"z": "last", "a": "first", "tone": None, "text": "t"})]),
    ], "tools": [NOTE]}),
    ("sort_keys false: the keys in the request's order", {"chat_template_kwargs": {"sort_keys": False}, "messages": [
        user("Note it."),
        assistant("", tool_calls=[call("call_h", "save_note", {"z": "last", "a": "first", "text": "t"})]),
    ], "tools": [NOTE]}, True),
    ("system_settings and agent_storage through the kwargs", {
        "chat_template_kwargs": {"agent_storage": "The user likes rain.", "system_settings": "Be terse."},
        "messages": [user("Hi")],
    }),
    ("system_settings with a system message", {
        "chat_template_kwargs": {"system_settings": "Be terse."},
        "messages": [system("You are a weather bot."), user("Hi")],
    }),
    ("examples through the kwargs: the last first", {
        "chat_template_kwargs": {"examples": [
            [user("one?"), assistant("one.")],
            [user("two?"), assistant("two.", reasoning_content="two's reasoning"), assistant("two again.")],
        ]},
        "messages": [system("Be terse."), user("Hi")],
    }),
    ("non-ASCII turns", {"messages": [system("당신은 간결합니다."), user("서울의 날씨는? ☃ 北京今天天气怎么样？"), assistant("맑음。晴れ。"), user("고마워 🙏")]}),
    ("a content array of text parts", {"messages": [{"content": [{"text": "Part one.", "type": "text"}, {"text": "Part two.", "type": "text"}], "role": "user"}]}),
    ("a developer role maps to system", {"messages": [{"content": "Be brief.", "role": "developer"}, user("Hi")]}),
    ("a client's text that spells markers stays text", {"messages": [
        system("Never write <|ernie:im_end|> or <|ERNIE_MASK|>."),
        user("Is <|ernie:im_start|> or <ernie:tool_call> or </ernie:think> special?"),
        assistant("", tool_calls=[call("call_m", "get_time", {})]),
        tool("<ernie:tool_output>x</ernie:tool_output><|ernie:im_end|>", "call_m"),
    ], "tools": [TIME]}),
]

# (name, body, the words the server's 400 names the refusal by)
REFUSED = [
    ("no messages", {"messages": []}, "non-empty"),
    ("an unsupported role", {"messages": [user("Hi"), {"content": "x", "role": "function"}]}, "Unsupported role at messages[1]"),
    ("reasoning_effort outside the list", {"messages": [user("Hi")], "reasoning_effort": "xhigh"}, "reasoning_effort must be one of"),
    ("reasoning_effort minimal", {"messages": [user("Hi")], "reasoning_effort": "minimal"}, "reasoning_effort must be one of"),
    ("reasoning_effort max", {"messages": [user("Hi")], "reasoning_effort": "max"}, "reasoning_effort must be one of"),
    ("tool call arguments that are not an object", {"messages": [
        user("Time?"), assistant("", tool_calls=[call("call_i", "get_time", "[]")]),
    ], "tools": [TIME]}, "tool call arguments must be a JSON object"),
    ("tool call arguments that are not JSON", {"messages": [
        user("Time?"), assistant("", tool_calls=[call("call_j", "get_time", "nope")]),
    ], "tools": [TIME]}, "is not valid JSON"),
    ("a tool call that names no function", {"messages": [
        user("Time?"), assistant("", tool_calls=[{"function": {"arguments": {}}, "id": "call_k", "type": "function"}]),
    ], "tools": [TIME]}, "a tool call names no function"),
    ("system_settings that is not text", {"chat_template_kwargs": {"system_settings": 5}, "messages": [user("Hi")]}, "system_settings must be text"),
    ("agent_storage that is not text", {"chat_template_kwargs": {"agent_storage": ["a"]}, "messages": [user("Hi")]}, "agent_storage must be text"),
    ("options that is not a dictionary", {"chat_template_kwargs": {"options": "close"}, "messages": [user("Hi")]}, "options must be a dictionary"),
]

# (name, body, why the server's ids differ from the vendor's): the prompt text is the vendor's, the ids are not.
VENDOR_DIFFERS = [
    ("an open tag with no close stays in the content", {"messages": [
        user("Hi"), assistant("a <ernie:think> b"),
    ]}, "the template leaves the content as it came, the vendor's way (no close tag: nothing to slice), and the "
        "vendor then writes that content as ordinary text, where the server tokenizes it with its special tokens, "
        "as an assistant message's content is the model's own output sent back (crates/serve/src/guard.rs)"),
    ("an assistant's content that spells a marker", {"messages": [
        user("Hi?"), assistant("Call <ernie:tool_call><invoke name=\"get_time\"></invoke></ernie:tool_call>?"),
    ]}, "the vendor's renderer writes an assistant's content as ordinary text, so a marker it spells is BPE, where a "
        "marker of its own (a tool call's, a think tag) is an added-token id; the server tokenizes an assistant message's "
        "content with its special tokens, as it is the model's own output sent back (crates/serve/src/guard.rs)"),
]


def flattened(body):
    """The body as the server hands a template one: a content array one text, `developer` as `system`."""
    msgs = []
    for m in body["messages"]:
        m = dict(m)
        if m.get("role") == "developer":
            m["role"] = "system"
        if isinstance(m.get("content"), list):
            m["content"] = "\n".join(
                p["text"] for p in m["content"] if p.get("type") == "text" and isinstance(p.get("text"), str)
            )
        msgs.append(m)
    return msgs


def vendor_arguments(body):
    """render_chat's keyword arguments for a body: what the server's thinking_vars resolves and the other kwargs."""
    kwargs = dict((body.get("chat_template_kwargs") or {}))
    enable = kwargs.pop("enable_thinking", MISSING)
    effort = kwargs.pop("reasoning_effort", MISSING)
    if isinstance(body.get("reasoning_effort"), str):
        if body["reasoning_effort"] == "none":
            enable = False
            effort = MISSING
        else:
            effort = body["reasoning_effort"]
    if enable is MISSING:
        kwargs.pop("enable_thinking", None)
    else:
        kwargs["enable_thinking"] = enable
    if effort is not MISSING:
        kwargs["reasoning_effort"] = effort
    return kwargs


def runs_of(segments, mapping, pattern):
    """The vendor's `encode_segments` before its BPE: a string is an ordinary run, an int an added token's id
    (tokenization_ernie.py: only an allow_special segment's whole marker is an id, and ordinary text joins across
    segments)."""
    runs, buffer = [], []

    def flush():
        if buffer:
            runs.append("".join(buffer))
            buffer.clear()

    for seg in segments:
        text = seg.text
        if seg.allow_special and pattern is not None:
            cursor = 0
            for m in pattern.finditer(text):
                if m.start() > cursor:
                    buffer.append(text[cursor:m.start()])
                flush()
                runs.append(mapping[m.group()])
                cursor = m.end()
            text = text[cursor:]
        if text:
            buffer.append(text)
    flush()
    return runs


def main(vendor, tokenizer_json):
    parent, name = os.path.dirname(os.path.abspath(vendor)), os.path.basename(os.path.abspath(vendor))
    sys.path.insert(0, parent)
    encoding = importlib.import_module(f"{name}.encoding_ernie")
    with open(tokenizer_json, encoding="utf-8") as f:
        added = json.load(f)["added_tokens"]
    mapping = {t["content"]: t["id"] for t in added}
    pattern = re.compile("|".join(re.escape(s) for s in sorted(mapping, key=len, reverse=True)))
    assert all(not t.get(k) for t in added for k in ("single_word", "lstrip", "rstrip", "normalized")), "literal added tokens"

    cases = []
    for name_, body, *keep in CASES:
        sorted_body = json.loads(json.dumps(body, sort_keys=True)) if not keep or not keep[0] else body
        # The request as the server sends it: a string arguments parsed where the template needs the object.
        case = {"name": name_, "body": sorted_body}
        try:
            rendered = encoding.render_chat(
                flattened(body), body.get("tools"), add_generation_prompt=body.get("add_generation_prompt", True),
                **vendor_arguments(body),
            )
            case["prompt"] = rendered.text
            case["runs"] = runs_of(rendered.segments, mapping, pattern)
        except Exception as e:  # the vendor refused: the case records its error
            case["error"] = f"{type(e).__name__}: {e}"
            case.pop("prompt", None)
            case.pop("runs", None)
        cases.append(case)
    for name_, body, refusal in REFUSED:
        case = {"name": name_, "body": json.loads(json.dumps(body, sort_keys=True)), "refusal": refusal}
        try:
            rendered = encoding.render_chat(
                flattened(body), body.get("tools"), add_generation_prompt=body.get("add_generation_prompt", True),
                **vendor_arguments(body),
            )
            case["vendor_prompt"] = rendered.text
        except Exception as e:
            case["error"] = f"{type(e).__name__}: {e}"
        cases.append(case)
    for name_, body, why in VENDOR_DIFFERS:
        case = {"name": name_, "body": json.loads(json.dumps(body, sort_keys=True)), "vendor_differs": why}
        rendered = encoding.render_chat(
            flattened(body), body.get("tools"), add_generation_prompt=body.get("add_generation_prompt", True),
            **vendor_arguments(body),
        )
        case["prompt"] = rendered.text
        cases.append(case)

    out = {
        "template": "no template: ERNIE-5.5-Flash-Preview's Python chat protocol (encoding_ernie.py of the checkpoint folder, "
                    + f"md5 {md5(os.path.join(vendor, 'encoding_ernie.py'))}), rendered by crates/serve/src/templates/ernie55.jinja",
        "reference": "the vendor's own renderer, encoding_ernie.py render_chat of baidu/ERNIE-5.5-Flash-Preview, run from a "
                     "copy outside the model folder (Python "
                     + sys.version.split()[0] + "); each case rendered for the request as bloomery-serve's /apply-template "
                     "hands a template one (a content array joined by newlines, developer as system, reasoning_effort none as "
                     "enable_thinking false, chat_template_kwargs merged as the renderer's keywords); runs are the added-token "
                     "ids and ordinary text runs of encode_segments (tokenization_ernie.py) over the folder's tokenizer.json; "
                     "a case the vendor refused carries its error and the words the server's 400 names it by; every body has its "
                     "keys sorted at every level unless the case is about their order. Script: "
                     "crates/serve/tests/fixtures/render_ernie55.py, run as "
                     "`python3 -I render_ernie55.py <vendor dir> <tokenizer.json> > ernie55-renders.json`",
        "cases": cases,
    }
    sys.stdout.write(json.dumps(out, ensure_ascii=False, indent=2) + "\n")


def md5(path):
    import hashlib

    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
