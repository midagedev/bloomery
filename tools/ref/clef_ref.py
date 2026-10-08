#!/usr/bin/env python3
"""The official Clef answers: the release's own Python systemone() on the A6000, with its hidden states.

    tools/ref/clef_ref.py run --snapshot DIR --revision SHA --suite FILE --edge FILE --out DIR [--reps 5]
    tools/ref/clef_ref.py encode --snapshot DIR --edge FILE --out DIR
    tools/ref/clef_ref.py images --snapshot DIR --revision SHA --requests FILE --images-dir DIR --out DIR [--encode-only]
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

`images` runs the release on requests that carry images (tools/ref/clefvis/requests.jsonl, the Clef image-input oracle's
set D, `just dump-ref-clefvis --ids | --ref-d`):

    tools/ref/clef_ref.py images --snapshot DIR --revision SHA --requests FILE --images-dir DIR --out DIR
                                 [--encode-only] [--reps 1] [--preproc-set DIR]

A request's `images` are file stems under --images-dir (PNG, opened as RGB); `state_tokens: N` in place of a `state`
makes a deterministic filler text that the release's tokenizer counts as exactly N ids. It first prints
`type(processor.image_processor).__name__` and writes `<out>/images.jsonl` and `<out>/<id>.ids` (the ids one per line,
what dump_mtmd's hidden sets read): per request the `input_ids`, the image-pad spans, the processor's `image_grid_thw`,
a summary of its `pixel_values` and, with --preproc-set (dump_mtmd's preproc set), the patches compared with the
mtmd-preprocessed image of the same name. A request whose image the processor would resize (a grid other than
(1, h/16, w/16)) is refused: set D runs on resize-free images, where the HF rule and llama.cpp's coincide. With
--encode-only that is all (no card). Without it the A6000 answers each request as `run` does (same files) and writes
`<id>.tower.f32`, the vision tower's merged output rows ([tokens, 4096] f32), and each request's facts.

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


# ---- images (set D): pure helpers, pinned by the self-test ----

IMAGE_PAD_TOKEN = "<|image_pad|>"
FILLER_WORDS = (
    "table river window garden paper silver morning engine harbor letter market bridge forest lantern copper valley "
    "meadow signal ribbon anchor orchard pillow quartz saddle thunder velvet whistle yellow zephyr candle dragon "
    "falcon glacier hammer island jungle kettle ladder mirror needle ocean pepper quiver rocket spider tunnel umbrella "
    "violin walnut anvil basket cactus dolphin ember feather granite hazel iris jasmine kernel lemon maple nectar olive"
).split()


def filler_text(n_words: int, seed: int = 20261008) -> str:
    """A deterministic text of n_words plain words: a linear congruential walk over FILLER_WORDS."""
    x = seed
    words = []
    for _ in range(n_words):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        words.append(FILLER_WORDS[(x >> 8) % len(FILLER_WORDS)])
    return " ".join(words)


def filler_state(count: Any, n_tokens: int) -> str:
    """The longest filler text `count` (text -> number of ids) puts at n_tokens or fewer, which must be exactly n_tokens."""
    lo, hi = 0, n_tokens + 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(filler_text(mid)) <= n_tokens:
            lo = mid
        else:
            hi = mid - 1
    text = filler_text(lo)
    if count(text) != n_tokens:
        raise SystemExit(f"clef_ref: no filler text of exactly {n_tokens} ids (the nearest has {count(text)})")
    return text


def image_spans(ids: list[int], pad_id: int) -> list[tuple[int, int]]:
    """(start, length) of every maximal run of pad_id in ids."""
    spans = []
    i = 0
    while i < len(ids):
        if ids[i] != pad_id:
            i += 1
            continue
        j = i
        while j < len(ids) and ids[j] == pad_id:
            j += 1
        spans.append((i, j - i))
        i = j
    return spans


def check_resize_free(rid: str, sizes: list[tuple[int, int]], grid: list[list[int]]) -> list[int]:
    """The processor's image_grid_thw against the images' (w, h): each must be (1, h/16, w/16), the proof that the image
    went through unresized. Returns the merged token count of each image ((h/16)(w/16)/4)."""
    if len(sizes) != len(grid):
        raise SystemExit(f"clef_ref: {rid}: {len(sizes)} images, the processor returned {len(grid)} grids")
    counts = []
    for (w, h), (t, gh, gw) in zip(sizes, grid):
        if w % 32 or h % 32 or (t, gh, gw) != (1, h // 16, w // 16):
            raise SystemExit(
                f"clef_ref: {rid}: a {w}x{h} image became grid {(t, gh, gw)}, want {(1, h // 16, w // 16)}: "
                "the processor resized it, and set D runs on resize-free images only"
            )
        counts.append(gh * gw // 4)
    return counts


def merge_order_patches(raw: Any, gh: int, gw: int) -> Any:
    """A planar f32 image [3, gh*16, gw*16] as the HF processor lays the patches out: [gh*gw, 3, 2, 16, 16] in merge order
    (rows (by, bx, dy, dx)), each patch (c, t, py, px) with the two temporal halves equal."""
    import numpy as np

    a = raw.reshape(3, gh // 2, 2, 16, gw // 2, 2, 16)  # c, by, dy, py, bx, dx, px
    a = a.transpose(1, 4, 2, 5, 0, 3, 6)  # by, bx, dy, dx, c, py, px
    a = a.reshape(gh * gw, 3, 1, 16, 16)
    return np.ascontiguousarray(np.repeat(a, 2, axis=2))


def pixel_diff(pv: Any, raw: Any, gh: int, gw: int) -> dict[str, Any]:
    """HF's pixel_values rows of one image ([gh*gw, 1536]) against the mtmd-preprocessed image: how many of the values are
    bit-equal and the largest difference."""
    import numpy as np

    want = merge_order_patches(raw, gh, gw).reshape(gh * gw, 1536).astype(np.float32)
    got = np.asarray(pv, dtype=np.float32).reshape(gh * gw, 1536)
    return {
        "values": int(got.size),
        "bit_equal": int(np.count_nonzero(got.view(np.uint32) == want.view(np.uint32))),
        "max_abs": float(np.max(np.abs(got.astype(np.float64) - want.astype(np.float64)))),
    }


def pick_tower_rows(output: Any) -> Any:
    """The merged rows of a vision tower's forward output: `pooler_output` of a model output, else the first of a tuple."""
    rows = getattr(output, "pooler_output", None)
    if rows is None and isinstance(output, (tuple, list)):
        rows = output[0]
    if rows is None:
        rows = output
    return rows


def materialize(row: dict[str, Any], images_dir: Path, tokenizer: Any) -> dict[str, Any]:
    """A request line as the release's functions take it: the state text (a filler of `state_tokens` ids when asked) and
    PIL images in place of the stems."""
    from PIL import Image

    request = {k: v for k, v in row.items() if k not in ("images", "state_tokens")}
    if "state_tokens" in row:
        request["state"] = filler_state(lambda t: len(tokenizer(t, add_special_tokens=False).input_ids), int(row["state_tokens"]))
    request["images"] = [Image.open(images_dir / f"{stem}.png").convert("RGB") for stem in row.get("images", [])]
    return request


def encode_images(release: Any, processor: Any, requests: Path, images_dir: Path, out: Path, preproc: Path | None) -> list[dict[str, Any]]:
    """Encode every request with the release's processor: <out>/images.jsonl and <out>/<id>.ids. Returns the materialized
    requests."""
    import hashlib

    import numpy as np

    tokenizer = processor.tokenizer
    print(f"image_processor: {type(processor.image_processor).__name__}", flush=True)
    pad_id = tokenizer.convert_tokens_to_ids(IMAGE_PAD_TOKEN)
    done = []
    with open(out / "images.jsonl", "w", encoding="utf-8") as f:
        for row in read_jsonl(requests):
            request = materialize(row, images_dir, tokenizer)
            check_pure(release, request)
            encoded = release.encode_record(tokenizer, request, max_length=16384, processor=processor)
            ids = list(encoded.input_ids)
            media = encoded.media or {}
            grid = media["image_grid_thw"].tolist() if "image_grid_thw" in media else []
            counts = check_resize_free(row["id"], [im.size for im in request["images"]], grid)
            spans = image_spans(ids, pad_id)
            if [n for _, n in spans] != counts:
                raise SystemExit(f"clef_ref: {row['id']}: image-pad runs {[n for _, n in spans]}, the grids say {counts}")
            (out / f"{row['id']}.ids").write_text("".join(f"{i}\n" for i in ids))
            pixels = []
            if grid:
                pv = media["pixel_values"].float().numpy()
                at = 0
                for stem, (_, gh, gw) in zip(row["images"], grid):
                    n = gh * gw
                    entry = {"image": stem, "rows": n, "sha256": hashlib.sha256(np.ascontiguousarray(pv[at : at + n]).tobytes()).hexdigest()}
                    if preproc is not None:
                        raw = np.fromfile(preproc / f"{stem}_inp_raw.0.f32", dtype="<f4").reshape(3, gh * 16, gw * 16)
                        entry["vs_mtmd"] = pixel_diff(pv[at : at + n], raw, gh, gw)
                    pixels.append(entry)
                    at += n
            line = {
                "id": row["id"],
                "images": row.get("images", []),
                "image_processor": type(processor.image_processor).__name__,
                "input_ids": ids,
                "image_spans": spans,
                "image_grid_thw": grid,
                "pixel_values": pixels,
                "state_render": render(request["state"]),
                "questions": question_rows(encoded),
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            print(f"images {row['id']} n={len(ids)} spans={spans} grid={grid}", flush=True)
            for entry in pixels:
                if "vs_mtmd" in entry:
                    print(f"  pixels {entry['image']}: {entry['vs_mtmd']}", flush=True)
            done.append(request)
    return done


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
    images_mode = args.command == "images"
    if images_mode:
        from transformers import AutoProcessor

        requests = encode_images(
            release, AutoProcessor.from_pretrained(snapshot), Path(args.requests), Path(args.images_dir), out,
            Path(args.preproc_set) if args.preproc_set else None)
        if args.encode_only:
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
    if images_mode:
        facts_extra = {"image_processor": type(processor.image_processor).__name__}
    else:
        facts_extra = {}
        encode_edges(release, tokenizer, processor, Path(args.edge), out)
        requests = read_jsonl(Path(args.suite))

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
        **facts_extra,
    }

    captured: dict[str, Any] = {}

    def pre(_module: Any, inputs: tuple) -> None:
        captured["hidden"] = inputs[0].detach().clone()

    def post(_module: Any, _inputs: tuple, output: Any) -> None:
        captured["logits"] = [[t.detach().float().cpu().tolist() for t in record] for record in output]

    lines = []
    for request in requests:
        rid = request["id"]
        check_pure(release, request)
        encoded = release.encode_record(tokenizer, request, max_length=16384, processor=processor)
        input_ids = list(encoded.input_ids)

        hooks = [model.head.register_forward_pre_hook(pre), model.head.register_forward_hook(post)]
        tower: list[Any] = []

        def vis(_module: Any, _inputs: tuple, output: Any) -> None:
            tower.append(pick_tower_rows(output).detach().float().cpu().numpy())

        if request.get("images"):
            visual = getattr(getattr(base, "model", None), "visual", None)
            if visual is None:
                raise SystemExit(f"clef_ref: {rid}: the backbone has no model.visual to hook")
            hooks.append(visual.register_forward_hook(vis))
        captured.clear()
        response = release.systemone(model, processor, request)
        for hook in hooks:
            hook.remove()
        if request.get("images") and not tower:
            raise SystemExit(f"clef_ref: {rid}: the vision tower did not run")
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
        tower_info: dict[str, Any] = {}
        if tower:
            rows = np.concatenate(tower, axis=0).astype(np.float32)
            rows.tofile(out / f"{rid}.tower.f32")
            tower_info = {"tower": {"shape": list(rows.shape), "calls": len(tower), "file": f"{rid}.tower.f32"}}

        line = {
            "id": rid,
            "input_ids": input_ids,
            "state_render": render(request["state"]),
            "questions": question_rows(encoded),
            "logits_bf16": logits_bf16,
            "logits_f32_head": logits_f32,
            "logits_f64_head": logits_f64,
            "response": response,
            **tower_info,
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
    # images (set D)
    expect("image_spans", image_spans([1, 9, 9, 2, 9, 3], 9), [(1, 2), (4, 1)])
    expect("image_spans none", image_spans([1, 2, 3], 9), [])
    expect("resize-free", check_resize_free("r", [(448, 448), (640, 480)], [[1, 28, 28], [1, 30, 40]]), [196, 300])
    for bad in ([(1000, 700)], [(448, 448), (640, 480)], [(448, 448)]):
        grids = {1: [[1, 44, 62]], 2: [[1, 28, 28]], 3: [[1, 27, 28]]}[len(bad) if bad != [(448, 448)] else 3]
        try:
            check_resize_free("r", bad, grids)
            expect(f"resize refused {bad}", "accepted", "refused")
        except SystemExit:
            pass
    words = lambda text: len(text.split())  # noqa: E731
    state = filler_state(words, 50)
    expect("filler count", words(state), 50)
    expect("filler deterministic", filler_state(words, 50), state)
    expect("filler prefix grows", filler_text(7).startswith(filler_text(3)), True)
    try:
        filler_state(lambda text: 2 * words(text), 51)
        expect("filler odd count refused", "accepted", "refused")
    except SystemExit:
        pass

    import numpy as np

    raw = np.random.default_rng(7).standard_normal((3, 32, 64)).astype(np.float32)
    pv = merge_order_patches(raw, 2, 4)
    expect("patches shape", pv.shape, (8, 3, 2, 16, 16))
    for by in range(1):
        for bx in range(2):
            for dy in range(2):
                for dx in range(2):
                    patch = pv[(by * 2 + bx) * 4 + dy * 2 + dx]
                    y0, x0 = (by * 2 + dy) * 16, (bx * 2 + dx) * 16
                    for t in range(2):
                        expect(f"patch ({by},{bx},{dy},{dx}) t{t}", bool(np.array_equal(patch[:, t], raw[:, y0 : y0 + 16, x0 : x0 + 16])), True)
    expect("pixel_diff equal", pixel_diff(pv.reshape(8, 1536), raw, 2, 4), {"values": 8 * 1536, "bit_equal": 8 * 1536, "max_abs": 0.0})
    other = pv.reshape(8, 1536).copy()
    other[3, 5] += np.float32(0.25)
    expect("pixel_diff counts", (pixel_diff(other, raw, 2, 4)["bit_equal"], pixel_diff(other, raw, 2, 4)["max_abs"]), (8 * 1536 - 1, 0.25))

    class WithPooler:
        pooler_output = "rows"

    expect("tower rows pooler", pick_tower_rows(WithPooler()), "rows")
    expect("tower rows tuple", pick_tower_rows(("a", "b")), "a")
    expect("tower rows plain", pick_tower_rows("t"), "t")
    print("clef_ref self-test: " + ("ok" if ok else "FAILED"))
    return ok


def main(argv: list[str]) -> int:
    if argv[:1] == ["--self-test"]:
        return 0 if self_test() else 1
    import argparse

    parser = argparse.ArgumentParser(prog="clef_ref.py")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "encode", "images"):
        p = sub.add_parser(command)
        p.add_argument("--snapshot", required=True)
        p.add_argument("--revision", required=True)
        p.add_argument("--out", required=True)
        if command != "images":
            p.add_argument("--edge", required=True)
        if command == "run":
            p.add_argument("--suite", required=True)
            p.add_argument("--reps", type=int, default=5)
        if command == "images":
            p.add_argument("--requests", required=True)
            p.add_argument("--images-dir", required=True)
            p.add_argument("--encode-only", action="store_true")
            p.add_argument("--preproc-set", default=None)
            p.add_argument("--reps", type=int, default=1)
    args = parser.parse_args(argv)
    if args.command in ("run", "images") and args.reps < 1:
        raise SystemExit("clef_ref: --reps is at least 1")
    return run(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
