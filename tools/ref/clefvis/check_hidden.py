#!/usr/bin/env python3
"""Set C, C' and C'' against an independent walk of the prompt: the M-RoPE positions, the ids and the rows.

    tools/ref/clefvis/check_hidden.py --set DIR --ids FILE [--against DIR]...
    tools/ref/clefvis/check_hidden.py --self-test

dump_mtmd's prompt sets (`# clefvis hidden|prose|bf16rows`) hold `result_norm` of every position and `mrope_pos`, the
(t, y, x) the decode was fed (mtmd's helper batches for an image, the text rule for the rest). This tool recomputes the
positions from the ids and the `# span` lines alone, by the rule of design §4.2: a text row at position p is (p, p, p)
and advances p by 1; an image of an nx by ny grid starting at p has row i at (p, p + i // nx, p + i % nx) and advances p
by max(nx, ny); a prose span is text. It shares no code with the harness. It also checks the ids file (its sha256 and
count, the image-pad id at every image span and nowhere else) and that `result_norm` is finite with no all-zero row.
`--against DIR` compares the rows before the first span with another prompt set of the same ids: causal attention makes
them the same computation, so they must be bit-identical, and the rows from the first span on are summarized (the
largest relative difference, as rms of the difference over rms of the reference). Exit 1 on any failure.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np

N_EMBD = 4096


def header_lines(text: str, key: str) -> list[list[str]]:
    """The fields after `# <key>` of every header line that opens with it."""
    return [line.split("\t")[1:] for line in text.splitlines() if line.split("\t")[0] == f"# {key}"]


def spans_of(text: str) -> list[dict[str, int | str]]:
    """The `# span` lines by the names of the `# span columns` line."""
    cols = header_lines(text, "span columns")
    if len(cols) != 1:
        raise SystemExit(f"check_hidden: {len(cols)} `# span columns` lines, want one")
    names = cols[0][0].split(" ")
    out = []
    for f in header_lines(text, "span"):
        if len(f) != len(names):
            raise SystemExit(f"check_hidden: a span line of {len(f)} fields, the columns line names {len(names)}")
        row: dict[str, int | str] = {}
        for n, v in zip(names, f):
            row[n] = v if n == "image" else int(v)
        out.append(row)
    return out


def expected_positions(n_ids: int, spans: list[dict[str, int | str]], prose: bool) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """[n_ids, 3] (t, y, x) by the rule, and each span's (start, end) position."""
    pos = np.zeros((n_ids, 3), dtype=np.int64)
    p = 0
    i = 0
    bounds = []
    for s in spans:
        at, ln = int(s["at"]), int(s["len"])
        if at < i or at + ln > n_ids:
            raise SystemExit(f"check_hidden: span {s['index']} at {at} len {ln} outside the {n_ids} ids after {i}")
        while i < at:
            pos[i] = (p, p, p)
            i += 1
            p += 1
        start = p
        if prose:
            for _ in range(ln):
                pos[i] = (p, p, p)
                i += 1
                p += 1
        else:
            nx, ny = int(s["nx"]), int(s["ny"])
            if nx * ny != ln:
                raise SystemExit(f"check_hidden: span {s['index']} is {nx}x{ny} for {ln} ids")
            for k in range(ln):
                pos[i] = (start, start + k // nx, start + k % nx)
                i += 1
            p = start + max(nx, ny)
        bounds.append((start, p))
    while i < n_ids:
        pos[i] = (p, p, p)
        i += 1
        p += 1
    return pos, bounds


def rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(a.astype(np.float64)))))


def check_set(set_dir: Path, ids_file: Path) -> tuple[bool, dict[str, Any]]:
    text = (set_dir / "MANIFEST.tsv").read_text()
    kinds = header_lines(text, "clefvis")
    if len(kinds) != 1 or kinds[0][0] not in ("hidden", "prose", "bf16rows"):
        raise SystemExit(f"check_hidden: {set_dir} is not a prompt set ({kinds})")
    kind = kinds[0][0]
    ok = True

    def fail(msg: str) -> None:
        nonlocal ok
        print(f"FAIL {set_dir.name}: {msg}")
        ok = False

    ids = [int(x) for x in ids_file.read_text().split()]
    n_tokens = int(header_lines(text, "tokens_count")[0][0])
    if len(ids) != n_tokens:
        fail(f"the ids file holds {len(ids)} ids, the set says {n_tokens}")
    sha = hashlib.sha256(ids_file.read_bytes()).hexdigest()
    if sha != header_lines(text, "tokens_file_sha256")[0][0]:
        fail(f"the ids file has sha256 {sha}, the set was dumped from {header_lines(text, 'tokens_file_sha256')[0][0]}")
    pad = int(header_lines(text, "image_pad_id")[0][0])
    spans = spans_of(text)
    in_span = np.zeros(len(ids), dtype=bool)
    for s in spans:
        in_span[int(s["at"]) : int(s["at"]) + int(s["len"])] = True
    ida = np.array(ids)
    if kind != "prose":
        if not np.all(ida[in_span] == pad):
            fail("an id inside an image span is not the image-pad id")
    if np.any(ida[~in_span] == pad):
        fail("an image-pad id outside every span")
    want, bounds = expected_positions(len(ids), spans, kind == "prose")
    got = np.fromfile(set_dir / "mrope_pos.0.i32", dtype="<i4").reshape(len(ids), 3)
    bad = np.flatnonzero(np.any(got != want, axis=1))
    if bad.size:
        i = int(bad[0])
        fail(f"{bad.size} of {len(ids)} positions differ; the first is index {i}: fed {got[i].tolist()}, the rule says {want[i].tolist()}")
    for s, (start, end) in zip(spans, bounds):
        if (int(s["start_pos"]), int(s["end_pos"])) != (start, end):
            fail(f"span {s['index']}: the set says positions {s['start_pos']}..{s['end_pos']}, the rule says {start}..{end}")
    rows = np.fromfile(set_dir / "result_norm.0.f32", dtype="<f4").reshape(len(ids), N_EMBD)
    if not np.all(np.isfinite(rows)):
        fail("result_norm holds a non-finite value")
    if np.any(np.all(rows == 0.0, axis=1)):
        fail("result_norm holds an all-zero row")
    info = {"kind": kind, "ids": len(ids), "spans": [(int(s["at"]), int(s["len"])) for s in spans], "end_pos": bounds[-1][1] if bounds else len(ids), "rows": rows}
    print(f"{'ok  ' if ok else 'FAIL'} {set_dir.name}: {kind}, {len(ids)} ids, {len(spans)} span(s), positions end at {info['end_pos']}")
    return ok, info


def compare(a: dict[str, Any], b: dict[str, Any], na: str, nb: str) -> bool:
    """Rows before the first span of two prompt sets of the same ids: bit-identical; the rest summarized."""
    if a["ids"] != b["ids"] or a["spans"] != b["spans"]:
        print(f"FAIL {na} vs {nb}: not the same prompt ({a['ids']} ids {a['spans']}, {b['ids']} ids {b['spans']})")
        return False
    first = a["spans"][0][0] if a["spans"] else a["ids"]
    ra, rb = a["rows"], b["rows"]
    same = int(np.count_nonzero(ra[:first].view(np.uint32) != rb[:first].view(np.uint32)))
    tail = ""
    if first < a["ids"]:
        d = ra[first:] - rb[first:]
        tail = f"; rows {first}.. rms relative difference {rms(d) / max(rms(ra[first:]), 1e-30):.3g}"
    print(f"{'ok  ' if same == 0 else 'FAIL'} {na} vs {nb}: {same} of {first * N_EMBD} values differ before the first span{tail}")
    return same == 0


def self_test() -> bool:
    ok = True

    def expect(name: str, got: Any, want: Any) -> None:
        nonlocal ok
        if got != want:
            print(f"FAIL {name}: got {got!r}, want {want!r}")
            ok = False

    spans = [{"index": 0, "image": "g", "at": 2, "len": 6, "nx": 3, "ny": 2, "n_pos": 3}]
    pos, bounds = expected_positions(10, spans, False)
    expect("text before", pos[:2].tolist(), [[0, 0, 0], [1, 1, 1]])
    expect("image rows", pos[2:8].tolist(), [[2, 2, 2], [2, 2, 3], [2, 2, 4], [2, 3, 2], [2, 3, 3], [2, 3, 4]])
    expect("text resumes at start + max(nx, ny)", pos[8:].tolist(), [[5, 5, 5], [6, 6, 6]])
    expect("bounds", bounds, [(2, 5)])
    tall = [{"index": 0, "image": "g", "at": 0, "len": 4, "nx": 1, "ny": 4, "n_pos": 4}]
    pos, bounds = expected_positions(5, tall, False)
    expect("tall image", pos[:4].tolist(), [[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0]])
    expect("tall resumes", pos[4].tolist(), [4, 4, 4])
    pos, bounds = expected_positions(8, spans, True)
    expect("prose span is text", pos[:, 0].tolist(), [0, 1, 2, 3, 4, 5, 6, 7])
    expect("prose span is text on every axis", bool(np.all(pos == pos[:, :1])), True)
    expect("prose bounds", bounds, [(2, 8)])
    none, bounds = expected_positions(3, [], False)
    expect("no span", (none.tolist(), bounds), ([[0, 0, 0], [1, 1, 1], [2, 2, 2]], []))
    try:
        expected_positions(4, [{"index": 0, "image": "g", "at": 1, "len": 5, "nx": 5, "ny": 1, "n_pos": 5}], False)
        expect("a span past the ids is refused", False, True)
    except SystemExit:
        pass
    try:
        expected_positions(8, [{"index": 0, "image": "g", "at": 1, "len": 5, "nx": 2, "ny": 2, "n_pos": 2}], False)
        expect("a grid that is not its length is refused", False, True)
    except SystemExit:
        pass
    text = "# span columns\tindex image at len\n# span\t0\tg\t2\t3\n"
    expect("span parse", spans_of(text), [{"index": 0, "image": "g", "at": 2, "len": 3}])
    a = {"ids": 4, "spans": [(2, 2)], "rows": np.arange(4 * N_EMBD, dtype=np.float32).reshape(4, N_EMBD)}
    b = {"ids": 4, "spans": [(2, 2)], "rows": a["rows"].copy()}
    b["rows"][3, 0] += 1.0
    expect("compare equal before the span", compare(a, b, "a", "b"), True)
    b["rows"][1, 5] += 1.0
    expect("compare names a difference before the span", compare(a, b, "a", "b"), False)
    print("check_hidden self-test: " + ("ok" if ok else "FAILED"))
    return ok


def main(argv: list[str]) -> int:
    if argv[:1] == ["--self-test"]:
        return 0 if self_test() else 1
    import argparse

    p = argparse.ArgumentParser(prog="check_hidden.py")
    p.add_argument("--set", required=True)
    p.add_argument("--ids", required=True)
    p.add_argument("--against", action="append", default=[])
    a = p.parse_args(argv)
    ok, info = check_set(Path(a.set), Path(a.ids))
    for other in a.against:
        ok2, info2 = check_set(Path(other), Path(a.ids))
        ok = ok2 and ok
        ok = compare(info, info2, Path(a.set).name, Path(other).name) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
