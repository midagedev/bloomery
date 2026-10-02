#!/usr/bin/env python3
"""The official Clef answers: the release's own Python systemone() on the A6000, with its hidden states.

    tools/ref/clef_ref.py run --snapshot DIR --revision SHA --suite FILE --edge FILE --out DIR [--reps 5]
    tools/ref/clef_ref.py encode --snapshot DIR --edge FILE --out DIR
    tools/ref/clef_ref.py --self-test

`run` imports the snapshot's own `joint_schema_model.py` (sys.path) and loads the release with
`load_release_model(snapshot, device="cuda:0")` (BF16, the model as shipped). Device 0 must be the
A6000 (`BLOOMERY_CARD=a6000 tools/box.sh` leaves only that card visible); any other card is refused by
name before the load. Per request of the suite it writes one line of `<out>/reference.jsonl`:
  - `id`, `input_ids`, `state_render` (the state as the encoder renders it);
  - `questions`: per question `question_id`, `question_type`, `question_span`, `option_spans`, `option_ids`;
  - `logits_bf16`: the shipped head's logits from the systemone() call itself, cast to float;
  - `response`: that call's systemone() body;
  - `logits_f32_head`: the same hidden states and lexical rows through a float32 copy of the head, on the card;
  - `logits_f64_head`: the same through a float64 copy of the head on the CPU (the referee), read from the
    two dump files below, so the referee is reproducible from them alone;
  - `wall_ms`: after the capture call (the warm-up), `--reps` timed systemone() calls between
    torch.cuda.synchronize() pairs: `median`, `min`, `all`; `same_answers` says whether every timed call
    returned the captured body;
  - `facts`: torch, transformers and CUDA versions, the card, the attention implementation, whether `fla`
    and `causal_conv1d` import, every warning transformers logged (and those that name a fallback), the
    head file's stored dtype, and the snapshot revision.
and two files: `<id>.hidden.f32` (`last_hidden_state`, float32, row-major [n_tokens, hidden]) and
`<id>.lexical.f32` (the output embedding rows of every id inside the option spans, float32, questions in
order, options in order, positions in order) with `<id>.lexical.ids` (those ids, one per line).
`--revision` must equal the commit the snapshot's download metadata records (refused otherwise).

`encode` (and `run`, first) writes `<out>/edge.jsonl`: the edge requests' `input_ids`, spans and
`state_render`, with no model call.

`render` and `question_options` below are the release's two pure functions, copied so the self-test can
pin them without torch; `run` and `encode` check that the snapshot's own functions give the same output on
every request, and stop by name when they do not. A request whose `usage.input_tokens` differs from its
`input_ids` length, a question type the release does not know, or an answer set the head did not score
stops by name too.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


# ---- the release's pure functions (joint_schema_model.py), pinned by the self-test ----

def render(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def question_options(question: dict[str, Any]) -> list[tuple[str, Any]]:
    question_type = str(question["type"])
    if question_type == "noul":
        criteria = {
            "true": "The proposition is true or the answer is yes.",
            "false": "The proposition is false or the answer is no.",
        }
        criteria.update(question.get("criteria") or {})
        return [(key, criteria[key]) for key in ("true", "false")]
    if question_type == "choice":
        return sorted((str(key), value) for key, value in question["criteria"].items())
    return [(str(index), value) for index, value in enumerate(question["criteria"])]


# ---- pure helpers ----

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise SystemExit(f"clef_ref: {path}:{number} is an empty line")
        rows.append(json.loads(line))
    ids = [row.get("id") for row in rows]
    if any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(ids):
        raise SystemExit(f"clef_ref: {path} needs a unique string id on every row, got {ids}")
    return rows


def lexical_order(input_ids: list[int], spans: list[list[tuple[int, int]]]) -> tuple[list[int], list[int]]:
    """The ids inside the option spans in span order, and input_ids remapped so that each position inside
    a span indexes its own row of that list (positions outside every span map to 0 and are never read)."""
    ids: list[int] = []
    remapped = [0] * len(input_ids)
    seen: set[int] = set()
    for question in spans:
        for start, end in question:
            if not 0 <= start < end <= len(input_ids):
                raise SystemExit(f"clef_ref: option span ({start}, {end}) outside {len(input_ids)} input ids")
            for position in range(start, end):
                if position in seen:
                    raise SystemExit(f"clef_ref: position {position} sits in two option spans")
                seen.add(position)
                remapped[position] = len(ids)
                ids.append(input_ids[position])
    return ids, remapped


def softmax(values: list[float]) -> list[float]:
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def max_dp(a: list[list[float]], b: list[list[float]]) -> float:
    """Max |Δp| over every option of every question, each question's logits through its own softmax."""
    if [len(q) for q in a] != [len(q) for q in b]:
        raise SystemExit(f"clef_ref: logit shapes differ: {[len(q) for q in a]} vs {[len(q) for q in b]}")
    return max(abs(x - y) for qa, qb in zip(a, b) for x, y in zip(softmax(qa), softmax(qb)))


def question_rows(encoded: Any) -> list[dict[str, Any]]:
    return [
        {
            "question_id": q.question_id,
            "question_type": q.question_type,
            "question_span": list(q.question_span),
            "option_spans": [list(s) for s in q.option_spans],
            "option_ids": list(q.option_ids),
        }
        for q in encoded.questions
    ]


def check_pure(release: Any, request: dict[str, Any]) -> None:
    """The snapshot's render/question_options against the pinned copies on one request."""
    pairs = [("state", request["state"])]
    pairs += [(f"{qid}.instructions", q.get("instructions") or str(qid)) for qid, q in request["questions"].items()]
    for name, value in pairs:
        if release.render(value) != render(value):
            raise SystemExit(f"clef_ref: {request['id']}: the release's render differs from the pinned copy on {name}")
    for qid, q in request["questions"].items():
        if release.question_options(q) != question_options(q):
            raise SystemExit(f"clef_ref: {request['id']}: the release's question_options differs on {qid}")


# ---- the box side (torch, the release module) ----

def import_release(snapshot: Path) -> Any:
    source = snapshot / "joint_schema_model.py"
    if not source.is_file():
        raise SystemExit(f"clef_ref: no joint_schema_model.py in {snapshot}")
    sys.path.insert(0, str(snapshot))
    import joint_schema_model

    if Path(joint_schema_model.__file__).resolve() != source.resolve():
        raise SystemExit(f"clef_ref: imported {joint_schema_model.__file__}, not {source}")
    return joint_schema_model


def snapshot_revision(snapshot: Path) -> str:
    """The commit hf download recorded for config.json (its metadata file's first line)."""
    meta = snapshot / ".cache" / "huggingface" / "download" / "config.json.metadata"
    if not meta.is_file():
        raise SystemExit(f"clef_ref: no download metadata at {meta}: not an `hf download --local-dir` snapshot")
    return meta.read_text().splitlines()[0].strip()


def encode_edges(release: Any, tokenizer: Any, processor: Any, edge: Path, out: Path) -> None:
    rows = read_jsonl(edge)
    with open(out / "edge.jsonl", "w", encoding="utf-8") as f:
        for request in rows:
            check_pure(release, request)
            encoded = release.encode_record(tokenizer, request, max_length=16384, processor=processor)
            row = {
                "id": request["id"],
                "input_ids": list(encoded.input_ids),
                "state_render": render(request["state"]),
                "questions": question_rows(encoded),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"edge {request['id']} n={len(encoded.input_ids)}", flush=True)


def run(args: Any) -> int:
    import importlib.util
    import logging
    import statistics
    import time

    import numpy as np
    import torch

    snapshot = Path(args.snapshot)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    revision = snapshot_revision(snapshot)
    if revision != args.revision:
        raise SystemExit(f"clef_ref: the snapshot records revision {revision}, not {args.revision}")
    release = import_release(snapshot)

    if args.command == "encode":
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(snapshot)
        encode_edges(release, processor.tokenizer, processor, Path(args.edge), out)
        return 0

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit(f"clef_ref: needs exactly one visible card (the A6000), got {torch.cuda.device_count()}")
    name = torch.cuda.get_device_name(0)
    if "A6000" not in name:
        raise SystemExit(f"clef_ref: device 0 is {name!r}, not the A6000 (run under BLOOMERY_CARD=a6000 tools/box.sh)")

    warnings: list[str] = []

    class Catch(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            warnings.append(record.getMessage())

    logging.getLogger("transformers").addHandler(Catch(level=logging.WARNING))

    from safetensors import safe_open
    from safetensors.torch import load_file
    import transformers

    t0 = time.perf_counter()
    model, processor = release.load_release_model(snapshot, device="cuda:0")
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    print(f"loaded in {load_s:.1f} s on {name}", flush=True)
    tokenizer = processor.tokenizer
    encode_edges(release, tokenizer, processor, Path(args.edge), out)

    head_path = snapshot / "joint_head.safetensors"
    with safe_open(head_path, framework="pt") as f:
        stored = sorted({str(f.get_slice(k).get_dtype()) for k in f.keys()})
    head_config = json.loads((snapshot / "joint_head_config.json").read_text())
    state = load_file(head_path)
    head32 = release.JointSchemaHead(**head_config)
    head32.load_state_dict(state, strict=True)
    head32 = head32.to(device="cuda:0", dtype=torch.float32).eval()
    head64 = release.JointSchemaHead(**head_config)
    head64.load_state_dict(state, strict=True)
    head64 = head64.to(dtype=torch.float64).eval()
    # The CPU referee shares the box with other rounds' builds: the thread cap llama-quantize gets too.
    torch.set_num_threads(16)

    base = model.language_model
    out_weight = base.get_output_embeddings().weight.detach()
    facts = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "card": name,
        "attn_implementation": base.config.text_config._attn_implementation
        if hasattr(base.config, "text_config") else base.config._attn_implementation,
        "fla": importlib.util.find_spec("fla") is not None,
        "causal_conv1d": importlib.util.find_spec("causal_conv1d") is not None,
        "head_stored_dtype": stored,
        "backbone_dtype": str(next(base.parameters()).dtype),
        "revision": revision,
        "load_s": round(load_s, 2),
    }

    captured: dict[str, Any] = {}

    def pre(_module: Any, inputs: tuple) -> None:
        captured["hidden"] = inputs[0].detach().clone()

    def post(_module: Any, _inputs: tuple, output: Any) -> None:
        captured["logits"] = [[t.detach().float().cpu().tolist() for t in record] for record in output]

    lines = []
    for request in read_jsonl(Path(args.suite)):
        rid = request["id"]
        check_pure(release, request)
        encoded = release.encode_record(tokenizer, request, max_length=16384, processor=processor)
        input_ids = list(encoded.input_ids)

        hooks = [model.head.register_forward_pre_hook(pre), model.head.register_forward_hook(post)]
        captured.clear()
        response = release.systemone(model, processor, request)
        for hook in hooks:
            hook.remove()
        if response["usage"]["input_tokens"] != len(input_ids):
            raise SystemExit(f"clef_ref: {rid}: usage.input_tokens {response['usage']['input_tokens']} != {len(input_ids)} ids")
        if set(response["answers"]) != {q.question_id for q in encoded.questions}:
            raise SystemExit(f"clef_ref: {rid}: answers {sorted(response['answers'])} do not match the questions")
        hidden = captured["hidden"]
        if tuple(hidden.shape) != (1, len(input_ids), head_config["hidden_size"]):
            raise SystemExit(f"clef_ref: {rid}: hidden {tuple(hidden.shape)} for {len(input_ids)} ids")
        logits_bf16 = captured["logits"][0]

        walls = []
        same = True
        for _ in range(args.reps):
            torch.cuda.synchronize()
            t = time.perf_counter()
            again = release.systemone(model, processor, request)
            torch.cuda.synchronize()
            walls.append((time.perf_counter() - t) * 1e3)
            same = same and again == response

        hidden32 = hidden[0].float()
        hidden32.cpu().numpy().astype(np.float32).tofile(out / f"{rid}.hidden.f32")
        lex_ids, remapped = lexical_order(input_ids, [list(q.option_spans) for q in encoded.questions])
        rows32 = out_weight[torch.tensor(lex_ids, device=out_weight.device)].float()
        rows32.cpu().numpy().astype(np.float32).tofile(out / f"{rid}.lexical.f32")
        (out / f"{rid}.lexical.ids").write_text("".join(f"{i}\n" for i in lex_ids))

        records = [encoded]
        mask = torch.ones((1, len(input_ids)), dtype=torch.long)
        ids_remapped = torch.tensor([remapped], dtype=torch.long)
        with torch.inference_mode():
            l32 = head32(hidden32.unsqueeze(0), ids_remapped.to("cuda:0"), mask.to("cuda:0"), records, rows32)[0]
            hidden64 = torch.from_numpy(np.fromfile(out / f"{rid}.hidden.f32", dtype=np.float32)).double()
            rows64 = torch.from_numpy(np.fromfile(out / f"{rid}.lexical.f32", dtype=np.float32)).double()
            hidden64 = hidden64.view(1, len(input_ids), head_config["hidden_size"])
            rows64 = rows64.view(len(lex_ids), head_config["hidden_size"])
            l64 = head64(hidden64, ids_remapped, mask, records, rows64)[0]
        logits_f32 = [t.cpu().tolist() for t in l32]
        logits_f64 = [t.tolist() for t in l64]

        line = {
            "id": rid,
            "input_ids": input_ids,
            "state_render": render(request["state"]),
            "questions": question_rows(encoded),
            "logits_bf16": logits_bf16,
            "logits_f32_head": logits_f32,
            "logits_f64_head": logits_f64,
            "response": response,
            "wall_ms": {
                "median": round(statistics.median(walls), 3),
                "min": round(min(walls), 3),
                "all": [round(w, 3) for w in walls],
            },
            "same_answers": same,
            "facts": dict(
                facts,
                fallback_warnings=sorted({w for w in warnings if "fall" in w.lower()}),
                transformers_warnings=sorted(set(warnings)),
            ),
        }
        lines.append(line)
        print(
            f"ref {rid} n={len(input_ids)} wall_ms median={line['wall_ms']['median']} min={line['wall_ms']['min']}"
            f" dp_f32={max_dp(logits_f32, logits_f64):.3e} dp_bf16={max_dp(logits_bf16, logits_f64):.3e} same={same}",
            flush=True,
        )

    with open(out / "reference.jsonl", "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print_table(lines)
    for w in sorted(set(warnings)):
        print(f"transformers warning: {w}", flush=True)
    return 0


def print_table(lines: list[dict[str, Any]]) -> None:
    print("| id | n | question | top option | p | max dp f32 | max dp bf16 |")
    print("|---|---|---|---|---|---|---|")
    for line in lines:
        dp32 = max_dp(line["logits_f32_head"], line["logits_f64_head"])
        dp16 = max_dp(line["logits_bf16"], line["logits_f64_head"])
        for q, logits in zip(line["questions"], line["logits_bf16"]):
            p = softmax(logits)
            top = max(range(len(p)), key=p.__getitem__)
            print(f"| {line['id']} | {len(line['input_ids'])} | {q['question_id']} | {q['option_ids'][top]} | "
                  f"{p[top]:.4f} | {dp32:.2e} | {dp16:.2e} |")


# ---- the self-test (no torch) ----

def self_test() -> bool:
    ok = True

    def expect(name: str, got: Any, want: Any) -> None:
        nonlocal ok
        if got != want:
            print(f"FAIL {name}: got {got!r}, want {want!r}")
            ok = False

    expect("render str", render("a\tb"), "a\tb")
    expect("render sorted compact", render({"b": 1, "a": [True, None]}), '{"a":[true,null],"b":1}')
    expect("render float exponent", render({"x": 1e16, "y": 1.5e-07, "z": 1250.0}), '{"x":1e+16,"y":1.5e-07,"z":1250.0}')
    expect("render non-ascii", render({"고객": "환불 😡"}), '{"고객":"환불 😡"}')
    expect("render bigint", render({"n": 2**64 + 1, "m": -(10**30)}),
           '{"m":-1000000000000000000000000000000,"n":18446744073709551617}')
    expect("render control", render({"log": "a\u0001b\tc\nd\u001f"}), '{"log":"a\\u0001b\\tc\\nd\\u001f"}')
    expect("render key order is code point", render({"b": 0, "B": 0, "_": 0, "가": 0}), '{"B":0,"_":0,"b":0,"가":0}')

    expect("noul default", question_options({"type": "noul"}), [
        ("true", "The proposition is true or the answer is yes."),
        ("false", "The proposition is false or the answer is no.")])
    expect("noul override true", question_options({"type": "noul", "criteria": {"true": "Approved."}}), [
        ("true", "Approved."), ("false", "The proposition is false or the answer is no.")])
    expect("choice sorted", [k for k, _ in question_options(
        {"type": "choice", "criteria": {"zeta": 1, "alpha": 2, "Mid": 3, "beta": 4, "_misc": 5}})],
        ["Mid", "_misc", "alpha", "beta", "zeta"])
    expect("score indexed", question_options({"type": "score", "criteria": ["lo", "hi"]}), [("0", "lo"), ("1", "hi")])

    ids, remapped = lexical_order([9, 8, 7, 6, 5, 4], [[(1, 3)], [(4, 6), (3, 4)]])
    expect("lexical ids", ids, [8, 7, 5, 4, 6])
    expect("lexical remap", remapped, [0, 0, 1, 4, 2, 3])
    try:
        lexical_order([1, 2, 3], [[(0, 2), (1, 3)]])
        expect("lexical overlap refused", "accepted", "refused")
    except SystemExit:
        pass
    expect("max_dp zero", max_dp([[1.0, 2.0]], [[1.0, 2.0]]), 0.0)
    expect("max_dp", round(max_dp([[0.0, 0.0]], [[0.0, math.log(3.0)]]), 12), 0.25)

    here = Path(__file__).resolve().parent / "clef"
    suite = read_jsonl(here / "suite.jsonl")
    expect("suite ids", [r["id"] for r in suite], [
        "route-01", "invoice-02", "tool-03", "guard-04", "review-05", "ko-06", "sentiment-07", "long-08"])
    edge = {r["id"]: r for r in read_jsonl(here / "edge.jsonl")}
    expect("edge count", len(edge), 6)
    needles = {
        "edge-float-exp": ["1e+16", "1.5e-07"],
        "edge-korean-emoji": ["고객", "😡"],
        "edge-bigint": ["123456789012345678901234567890", "-18446744073709551617"],
        "edge-control-chars": ["\\u0001", "\\t", "\\n"],
    }
    for rid, words in needles.items():
        state = render(edge[rid]["state"]) if rid in edge else ""
        for word in words:
            expect(f"{rid} renders {word}", word in state, True)
    unsorted = edge.get("edge-choice-unsorted", {"questions": {}})["questions"]
    expect("edge choice keys out of order", any(
        q["type"] == "choice" and list(q["criteria"]) != sorted(q["criteria"]) for q in unsorted.values()), True)
    override = edge.get("edge-noul-criteria", {"questions": {}})["questions"]
    expect("edge noul overrides true", any(
        q["type"] == "noul" and "true" in (q.get("criteria") or {}) for q in override.values()), True)
    for rid, request in list(edge.items()) + [(r["id"], r) for r in suite]:
        for qid, q in request["questions"].items():
            expect(f"{rid}.{qid} type", q["type"] in ("noul", "choice", "score"), True)
            question_options(q)
    print("clef_ref self-test: " + ("ok" if ok else "FAILED"))
    return ok


def main(argv: list[str]) -> int:
    if argv[:1] == ["--self-test"]:
        return 0 if self_test() else 1
    import argparse

    parser = argparse.ArgumentParser(prog="clef_ref.py")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "encode"):
        p = sub.add_parser(command)
        p.add_argument("--snapshot", required=True)
        p.add_argument("--revision", required=True)
        p.add_argument("--edge", required=True)
        p.add_argument("--out", required=True)
        if command == "run":
            p.add_argument("--suite", required=True)
            p.add_argument("--reps", type=int, default=5)
    args = parser.parse_args(argv)
    if args.command == "run" and args.reps < 1:
        raise SystemExit("clef_ref: --reps is at least 1")
    return run(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
