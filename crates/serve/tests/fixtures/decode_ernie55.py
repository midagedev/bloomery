"""Reference parses of ERNIE 5.5 Flash Preview's completions for crates/serve/tests/fixtures/ernie55-parses.json.

usage:
    python3 -I decode_ernie55.py <vendor dir> > ernie55-parses.json

`<vendor dir>` holds a copy of the checkpoint folder's `encoding_ernie.py` and `decoding_ernie.py` (baidu/
ERNIE-5.5-Flash-Preview): the vendor's renderer and decoder, imported from there, never run from the model folder.
Each case is a model output as the server's parser sees it: the text after the generation prompt, the stop token taken
off. The vendor's `parse_completion` reads it as a string with the request's tools (the prompt's own `<ernie:think>`
put back in front of it, as `rendered` makes the decoder do, and the `<|ernie:im_end|>` a whole output ends on put
back after it), in its default argument mode (`schema`: each parameter's text is the string it reads as, or the JSON
it parses as, whichever the tool's schema admits; two readings that both pass are refused as ambiguous). `thinking`
false is the prompt that closes the think span (enable_thinking false). A `cut` output is one the token limit ended
inside the think span or the tool-call block: the vendor reads it with allow_incomplete.

A case holds `want` (reasoning, content and the calls: each name and its arguments as JSON) or `error` (the vendor's
exception: a malformed output the server must refuse too) or `cut_want` (what the vendor reads of a cut output). The
tools are render_ernie55.py's.
"""
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_ernie55 as r  # noqa: E402  (the tools, shared with the renders)

TOOLS = [r.WEATHER, r.TIME, r.NOTE, r.KNOBS]

ON = "<ernie:think>"
CLOSE = "</ernie:think>"
BLOCK = "<ernie:tool_call>"
BLOCK_END = "</ernie:tool_call>"


def invoke(name, **params):
    return f'<invoke name="{name}">' + "".join(
        f'<parameter name="{k}">{v}</parameter>' for k, v in params.items()
    ) + "</invoke>"


def block(*invokes):
    return BLOCK + "\n" + "".join(i + "\n" for i in invokes) + BLOCK_END


WEATHER = block(invoke("get_weather", location="Seoul", days=3))
TIME = block(invoke("get_time"))

# (name, thinking, output, tools, cut)
CASES = [
    ("an answer", True, "The user greets." + CLOSE + "Hello!", TOOLS, False),
    ("an empty think span", True, CLOSE + "Hello!", TOOLS, False),
    ("an answer with no tools offered", True, "Why." + CLOSE + "Hello!", None, False),
    ("no think close, cut by the limit", True, "still thinking", TOOLS, True),
    ("one call", True, "Need the weather." + CLOSE + WEATHER, TOOLS, False),
    ("one call, no tools offered", True, CLOSE + block(invoke("get_time")), None, False),
    ("two calls in one block", True, "Both." + CLOSE + "Checking both." + block(invoke("get_weather", location="Seoul"), invoke("get_time")), TOOLS, False),
    ("text before the calls", True, "Hi." + CLOSE + "Let me check." + WEATHER, TOOLS, False),
    ("whitespace between the elements", True, CLOSE + BLOCK + "\n\t" + invoke("get_time") + "\n  \n" + BLOCK_END + "\n", TOOLS, False),
    ("more whitespace in an opening tag", True, CLOSE + block('<invoke\n  name="get_time"></invoke>'), TOOLS, False),
    ("a string value with quotes, an angle bracket and a near-miss close", True, CLOSE + block(
        invoke("save_note", text='a<b & "c" </parameter x </invoke> y'),
    ), TOOLS, False),
    ("a string value with newlines and non-ASCII", True, CLOSE + block(
        invoke("save_note", text="첫 줄\n둘째 줄 ☃\n"),
    ), TOOLS, False),
    ("JSON-valued arguments", True, CLOSE + block(invoke(
        "set_knob", gain="1.5", rate="0.1", size="-3", tags='["a", "b"]', on="true", **{},
    )), TOOLS, False),
    ("a JSON object argument", True, CLOSE + block(
        invoke("save_note", filter='{"n": [1, 2], "q": "서울 ☃", "ok": true, "x": null}', text="t"),
    ), TOOLS, False),
    ("a string-typed value that reads as a number", True, CLOSE + block(invoke("get_weather", location="123")), TOOLS, False),
    ("a string-typed value that reads as JSON", True, CLOSE + block(invoke("get_weather", location='{"a": 1}')), TOOLS, False),
    ("a parameter the schema does not name: text", True, CLOSE + block(invoke("get_weather", location="Seoul", extra="just words")), TOOLS, False),
    ("a call with no parameter", True, CLOSE + TIME, TOOLS, False),
    ("no whitespace at all", True, CLOSE + BLOCK + invoke("get_time") + BLOCK_END, TOOLS, False),
    ("text after the block's close: whitespace only", True, CLOSE + TIME + "\n\n", TOOLS, False),
    ("the think span closed and nothing else", True, CLOSE, TOOLS, False),
    ("thinking off: the output is content", False, "Hello!", TOOLS, False),
    ("thinking off: a call", False, "Sure." + TIME, TOOLS, False),
    # Both refuse.
    ("a duplicate parameter", True, CLOSE + block(
        '<invoke name="get_weather"><parameter name="location">Seoul</parameter><parameter name="location">Oslo</parameter></invoke>',
    ), TOOLS, False),
    ("a parameter never closed", True, CLOSE + BLOCK + '<invoke name="get_weather"><parameter name="location">Seoul' + BLOCK_END, TOOLS, False),
    ("an invoke never closed", True, CLOSE + BLOCK + '<invoke name="get_weather">' + BLOCK_END, TOOLS, False),
    ("a block with no invoke", True, CLOSE + BLOCK + BLOCK_END, TOOLS, False),
    ("text where an invoke belongs", True, CLOSE + BLOCK + "hello" + BLOCK_END, TOOLS, False),
    ("an opening tag without a name", True, CLOSE + BLOCK + "<invoke></invoke>" + BLOCK_END, TOOLS, False),
    ("an opening tag with an empty name", True, CLOSE + BLOCK + '<invoke name=""></invoke>' + BLOCK_END, TOOLS, False),
    ("text where a parameter belongs", True, CLOSE + BLOCK + '<invoke name="get_weather"><function=x></invoke>' + BLOCK_END, TOOLS, False),
    ("an integer-typed value that is not JSON", True, CLOSE + block(invoke("get_weather", location="Seoul", days="three")), TOOLS, False),
    ("a boolean-typed word", True, CLOSE + block(invoke("set_knob", gain="1", on="True")), TOOLS, False),
    ("the block's close inside a value", True, CLOSE + BLOCK + '<invoke name="get_weather"><parameter name="location">Se' + BLOCK_END + '</parameter></invoke>' + BLOCK_END, TOOLS, False),
    ("a name with a quote cut short", True, CLOSE + BLOCK + '<invoke name="get"time"></invoke>' + BLOCK_END, TOOLS, False),
    # A cut inside the tool-call block: the vendor reads what it can with allow_incomplete.
    ("cut inside a call after its name", True, CLOSE + BLOCK + '<invoke name="get_weather"><parameter name="location">Se', TOOLS, True),
    ("cut right after the block's opening", True, CLOSE + BLOCK, TOOLS, True),
]


def main(vendor):
    parent, name = os.path.dirname(os.path.abspath(vendor)), os.path.basename(os.path.abspath(vendor))
    sys.path.insert(0, parent)
    encoding = importlib.import_module(f"{name}.encoding_ernie")
    decoding = importlib.import_module(f"{name}.decoding_ernie")
    assert decoding.THINK == ON and decoding.CALL == BLOCK and decoding.CALL_END == BLOCK_END

    cases = []
    for name_, thinking, output, tools, cut in CASES:
        tools = None if tools is None else json.loads(json.dumps(tools))
        rendered = encoding.render_chat(
            [{"content": "hi", "role": "user"}], tools=tools, add_generation_prompt=True, enable_thinking=thinking,
        )
        case = {"name": name_, "thinking": thinking, "output": output, "tools": tools, "cut": cut}
        try:
            parsed = decoding.parse_completion(
                output if cut else output + decoding.END, rendered=rendered, tools=tools, allow_incomplete=cut,
            )
        except Exception as e:  # the vendor refuses: a malformed output
            assert not cut, (name_, e)
            case["error"] = f"{type(e).__name__}: {e}"
            cases.append(case)
            continue
        calls = [
            {"arguments": json.loads(c["function"]["arguments"]), "name": c["function"]["name"]}
            for c in parsed["tool_calls"]
        ]
        want = {"calls": calls, "content": parsed["content"], "reasoning": parsed["reasoning_content"]}
        if cut:
            want["complete"] = parsed["complete"]
            case["cut_want"] = want
        else:
            assert parsed["complete"], (name_, parsed)
            case["want"] = want
        cases.append(case)

    out = {
        "reference": "the vendor's own decoder, decoding_ernie.py parse_completion of baidu/ERNIE-5.5-Flash-Preview "
                     "(argument_mode schema, the default), run from a copy outside the model folder (Python "
                     + sys.version.split()[0]
                     + "); each output is the text after the generation prompt the vendor's renderer writes for one user "
                     "message (`thinking` false: enable_thinking false), read with the prompt's own think tag and the end "
                     "token put back; a `cut` output is read with allow_incomplete. The tools are "
                     "ernie55-renders.json's generator's (render_ernie55.py). Script: crates/serve/tests/fixtures/"
                     "decode_ernie55.py, run as `python3 -I decode_ernie55.py <vendor dir> > ernie55-parses.json`",
        "cases": cases,
    }
    sys.stdout.write(json.dumps(out, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
