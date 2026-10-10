#!/usr/bin/env python3
"""Every `bloomery-serve --hf <repo>:<quant>` row of the README's Use table is checked against what Hugging Face serves now: the quant's shard 1 is range-read and its `general.architecture` goes through a mirror of the serve router, because a file's architecture string can differ from the module name and only the file the hub serves today shows it.

    tools/hf-arch-check.py --live         read the hub, check every row (network: huggingface.co only)
    tools/hf-arch-check.py --live --box [--box-hash]
                                          also compare each row's files with the box copies the gates load
    tools/hf-arch-check.py --self-test    offline: the mirror equals the Rust source, and the checks' own tests
    tools/hf-arch-check.py box-facts [--hash] [--grep FILE STRING]... <paths...>
                                          what --live --box runs on the box

--live reads README.md's Use table (the rows are the table's, never a list of this tool's), the listing of each
repo (`api/models/<repo>/tree/main?recursive=true`), picks the quant's set as the hf crate does, and range-reads
shard 1's header with curl: 1 MiB, then doubled by fetching only the next slice, at most 64 MiB a row. It prints
one line per row:

    hf-arch <seat> <repo>:<quant> shard1=<path> arch=<arch> route=<ok|REFUSED: ...> embd=<type> exps=<types> types=<ok|printed|REFUSED: ...>

and a summary `hf-arch rows=<n> ok=<n> red=<n>`. route=ok holds when the router's seat for the architecture is the row's
seat (`Model::serves`; for the decide row one of `serve::decide::pick`'s paths) and the model reader takes the
architecture (`model::arch::spec`). embd and exps are the ggml types of `token_embd.weight` and of the routed-expert
tensors in the headers of the set's shards (a split set's later shards are range-read for theirs, within the same
64 MiB), exps by role. types is judged for the decide seat alone, whose backbone body keeps one list per site kind
(`crates/gpu/src/arch/qwen3moe/body35.rs`: ok, or REFUSED naming the first tensor and its list); every other seat
decides its weight types in several places, so types=printed names them and nothing is judged from the types.

--live --box runs `box-facts` on the box through tools/box.sh (it syncs this tree to the box and builds nothing) for
every file of every row's set at the paths `BOX_FILES` names, and compares each with the hub's listing: absent, size, then
the sha256 from the file's `.verified` marker, else from this machine's cache (path, size, mtime_ns), else the file stays
`unhashed` and the run prints the one hash command with the bytes it reads and its wall. `--box-hash` hashes those files on
the box with O_DIRECT reads into page-aligned buffers, never through the page cache and with no buffered fallback. Per file:

    hf-arch-box <seat> <repo>:<quant> file=<box path> box=<n>B live=<n>B digest=<marker|cache|hash|unhashed> verdict=<ok|unhashed|excepted: ...|FAIL: ...>

and a summary `hf-arch-box files=<n> ok=<n> fail=<n> unhashed=<n> excepted=<n>`. A box file may carry a named exception that
expires (`BOX_EXCEPTIONS`): while its expiry file on the box (`box-facts --grep`) does not hold its string, a file at exactly
the two named sizes is `excepted: <reason>` and not a FAIL, counted on its own in `excepted=`, and the day the file holds the
string it is a FAIL naming the exception.

--self-test greps the Rust sources the mirror copies (`Model::serves`, `Arch::from_name`, `spec`, the decide rows'
constants, `GgmlType::from_u32`) and fails, naming every name that is in one and not the other, or a source it cannot
parse; it also runs the checks on synthetic GGUF headers, the README parser and the hf crate's own `names` examples.

Exit status: 0 every row green (and, with --box, no file FAIL); 1 a row red (every row is checked first) or a self-test failed; 64 a usage error;
65 a README or a source this tool cannot read, by name; 69 the network or the hub (curl 5, 6, 7, 28, 35, 52, 55, 56,
or HTTP 429 and 5xx) refused, one named line, never a skipped row. An HTTP 404 or 401 on a row's own repo or file is
a red row: the command the README gives does not work.
"""
import collections
import concurrent.futures
import hashlib
import importlib.util
import json
import mmap
import os
import re
import shlex
import struct
import subprocess
import sys
import tempfile
import traceback
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = "README.md"
HUB = "https://huggingface.co"
MIB = 1 << 20
FIRST_READ = 1 * MIB
ROW_BUDGET = 64 * MIB
CONNECT_TIMEOUT = "15"
FETCH_TIMEOUT = "300"


class Refuse(Exception):
    """A named refusal of an input (65), the network (69) or a usage error (64)."""

    def __init__(self, code, msg):
        super().__init__(msg)
        self.code = code


Entry = collections.namedtuple("Entry", "path size check digest")  # check: "sha256" (LFS) or "git-sha1"; digest: hex


class Red(Exception):
    """A row the router, the hub or the file refuses: one red line, not an end of the run."""


# ------------------------------------------------------------------------------------------------
# The mirror of the serve router. Each block cites its Rust source; --self-test greps the source and
# fails on any name in one and not the other.

# crates/gpu-gates/src/bin/bloomery_serve.rs:Model::serves — the architecture strings each generative seat
# serves. The glm seat reads its names through Arch::from_name (SERVES_VIA_FROM_NAME).
SERVES = {
    "ds41": ("deepseek41", "deepseek4"),
    "qwen38": ("qwen4exp",),
    "qwen3": ("qwen3moe", "qwen35moe"),
    "mimo2": ("mimo2",),
}
SERVES_VIA_FROM_NAME = {"glm": "Glm5next"}
# crates/gpu-gates/src/bin/bloomery_serve.rs:Model::ALL and Model::word — the generative seats in the order
# the refusal of an unserved file lists them; crates/serve/src/decide.rs:WORD is the decide seat's word.
SEAT_ORDER = ("ds41", "qwen38", "glm", "qwen3", "mimo2")
DECIDE_WORD = "decide"
# crates/model/src/arch/mod.rs:Arch::from_name — variant -> the strings that name it.
FROM_NAME = {
    "Deepseek2": ("deepseek2",),
    "Deepseek41": ("deepseek41", "deepseek4"),
    "Qwen3moe": ("qwen3moe",),
    "Qwen35moe": ("qwen35moe",),
    "Glm5next": ("glm5next", "glm5-next"),
    "Mimo2": ("mimo2",),
}
# crates/model/src/arch/mod.rs:spec — the architectures the model reader takes (QWEN35_BODY is `qwen35` and `clef`).
SPEC = ("deepseek41", "deepseek4", "qwen3moe", "qwen35moe", "qwen4exp", "qwen35", "clef", "glm5next", "glm5-next",
        "mimo2")
# crates/gpu-gates/src/bin/shared/serve_seats/decide.rs:ROWS with the constants of
# crates/decision/src/release/{clef,lev}.rs.
DECIDE_ROWS = (
    {"name": "clef", "backbones": ("qwen35", "clef"), "in_file_arch": ("clef",), "in_file_decision": (),
     "head_repo": "Cloudflare/clef-flash", "quant_repo": "bartowski/Cloudflare_clef-flash-GGUF:Q5_K_M",
     "unserved": ()},
    {"name": "lev", "backbones": ("qwen35",), "in_file_arch": (), "in_file_decision": ("lev",),
     "head_repo": None, "quant_repo": None, "unserved": ()},
)
# crates/gguf/src/quant.rs:GgmlType::from_u32 — the ids the enum names. EXTRA_TYPES are ggml's names of ids the
# enum leaves Unknown (the doc of GgmlType names them); an id in neither prints as `type<id>`.
GGML_TYPES = {
    0: "F32", 1: "F16", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S",
    23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0",
    35: "TQ2_0", 39: "MXFP4",
}
EXTRA_TYPES = {2: "Q4_0", 3: "Q4_1", 9: "Q8_1", 15: "Q8_K"}
# The decide seat's backbone body has one owner list of the weight types it launches: crates/gpu/src/arch/qwen3moe/body35.rs
# (its `const` lists of SiteTy, read at each site by `ty_of` and `routed_ty`). The lists are read from that source each run
# (body35_lists, SiteTy::Q3K being Q3_K through crates/gpu/src/site.rs:SiteTy::of_ggml), so a type added to one is followed.
# BODY35_ARCHS are the architectures that body reads (crates/model/src/arch/mod.rs:QWEN35_BODY); BODY35_SITES the tensors
# each list is read for, by the `ty_of`/`routed_ty` calls of `plan_gqa`, `plan_delta`, `plan_ffn` and `open`; --self-test
# holds how many calls read each list. A routed layer's stacks and shared expert are read against FFN_READ, their admission
# being the card kernel table's (crates/model/src/arch/coverage.rs:qwen35_gate_up/qwen35_down/qwen35_shared); a qwen35 file
# has no routed layer.
# The head site is `crates/gpu/src/weights.rs:head_tensor`.
BODY35_RS = "crates/gpu/src/arch/qwen3moe/body35.rs"
BODY35_ARCHS = ("qwen35", "clef")
BODY35_SITES = (
    (r"blk\.\d+\.attn_q\.weight", "PROJ"),
    (r"blk\.\d+\.attn_k\.weight", "PROJ"),
    (r"blk\.\d+\.attn_v\.weight", "PROJ"),
    (r"blk\.\d+\.attn_output\.weight", "PROJ"),
    (r"blk\.\d+\.attn_qkv\.weight", "PROJ"),
    (r"blk\.\d+\.attn_gate\.weight", "PROJ"),
    (r"blk\.\d+\.ssm_out\.weight", "PROJ"),
    (r"blk\.\d+\.ffn_gate\.weight", "PROJ"),
    (r"blk\.\d+\.ffn_up\.weight", "PROJ"),
    (r"blk\.\d+\.ffn_down\.weight", "PROJ"),
    (r"blk\.\d+\.ssm_beta\.weight", "BETA_ALPHA"),
    (r"blk\.\d+\.ssm_alpha\.weight", "BETA_ALPHA"),
    (r"blk\.\d+\.ffn_(?:gate|up|down)_exps\.weight", "FFN_READ"),
    (r"blk\.\d+\.ffn_(?:gate|up|down)_shexp\.weight", "FFN_READ"),
)
# crates/gpu-gates/src/bin/shared/serve_seats/decide.rs:BODIES — the one backbone body the decide seat opens, by the
# predicate that selects it (`is_qwen35_body`: BODY35_ARCHS).
DECIDE_BODY_PREDICATES = ("is_qwen35_body",)
BODY35_EMBED_TENSOR = "token_embd.weight"
BODY35_HEAD_TENSOR = "output.weight"
# The other seats have no one owner list: each decides in several places, (file, anchor) pairs with the line found by
# the anchor. A type there is printed and never judged.
TYPE_PLACES = {
    "ds41": (("crates/gpu-deepseek41/src/dense.rs", "GgmlType::Q3_K => Ok(Dense::Q3K(t))"),
             ("crates/gpu-deepseek41/src/chain/glue.rs", "let embd_ok = matches!(embd.ty()"),
             ("crates/placement/src/placement.rs", "pub fn of(ty: GgmlType) -> Option<CardFormat>"),
             ("crates/qdot/src/lib.rs", "pub fn supports(w: GgmlType) -> bool")),
    "glm": (("crates/model/src/arch/glm5next/place.rs", "pub fn card_routed(ty: GgmlType)"),
            ("crates/gpu-glm5next/src/ffn.rs", "GgmlType::Q4_K => kq.enqueue_gate_up_q4k"),
            ("crates/gpu-glm5next/src/body.rs", "if table.ty() != GgmlType::Q8_0"),
            ("crates/qdot/src/lib.rs", "pub fn supports(w: GgmlType) -> bool")),
    "qwen38": (("crates/model/src/arch/qwen35moe/place.rs", "pub fn card_routed(ty: GgmlType)"),
               ("crates/placement/src/placement/qwen38_cards.rs", "fn card_routed(ty: GgmlType)"),
               ("crates/gpu/src/arch/qwen3moe/program38.rs",
                "let (qs, d) = q8(c.w, &model::arch::qwen35moe::names::token_embd())?;"),
               ("crates/qdot/src/lib.rs", "pub fn supports(w: GgmlType) -> bool")),
    "qwen3": (("crates/gpu/src/arch/qwen3moe/body.rs", "kq_site(w, &g.attn_q, q, h, &[SiteTy::Q4K])?;"),
              ("crates/gpu/src/head.rs", "fn head_out_w(w: &Weights)"),
              ("crates/model/src/arch/qwen3moe/place.rs", "pub fn card_routed(ty: GgmlType)"),
              ("crates/qdot/src/lib.rs", "pub fn supports(w: GgmlType) -> bool")),
    "mimo2": (("crates/model/src/arch/mimo2/place.rs", "let plan = placement::plan_host_routed("),
              ("crates/gpu-mimo2/src/body.rs", "if table.ty() != GgmlType::Q8_0"),
              ("crates/qdot/src/lib.rs", "pub fn supports(w: GgmlType) -> bool")),
}
SEAT_WORDS = SEAT_ORDER + (DECIDE_WORD,)


def seat_names(seat):
    """The architecture strings of the generative `seat`."""
    if seat in SERVES_VIA_FROM_NAME:
        return FROM_NAME[SERVES_VIA_FROM_NAME[seat]]
    return SERVES[seat]


def serves(seat, arch):
    return arch in seat_names(seat)


def decide_pick(arch, decision, repo, card):
    """crates/serve/src/decide.rs:pick with no --model word and no head flag, as `bloomery-serve` calls it
    (bloomery_serve.rs:head_of): ("refused", text), ("decide", row) or None for the generative seats. `card(repo)`
    is the repo the model card says `repo` quantizes (None for no such card); it is called only where pick reads it."""
    for row in DECIDE_ROWS:
        for unserved_arch, why in row["unserved"]:
            if unserved_arch == arch:
                return ("refused", f"a {arch} file is {why}")
    by_arch = any(arch in r["in_file_arch"] for r in DECIDE_ROWS)
    if not by_arch and decision is not None:
        row = next((r for r in DECIDE_ROWS if arch in r["backbones"] and decision in r["in_file_decision"]), None)
        if row is None:
            return ("refused", f"a {arch} file whose {arch}.decision.type is {decision}: this server serves no such "
                               f"decision model; it serves {served_in_file()}")
        return ("decide", row)
    row = next((r for r in DECIDE_ROWS if arch in r["in_file_arch"]), None)
    if row is not None:
        return ("decide", row)
    backbone = any(arch in r["backbones"] for r in DECIDE_ROWS)
    generative = any(serves(s, arch) for s in SEAT_ORDER)
    if generative or not backbone:
        return None
    base = card(repo)
    row = next((r for r in DECIDE_ROWS if r["head_repo"] is not None and base == r["head_repo"]
                and arch in r["backbones"]), None)
    if row is not None:
        return ("decide", row)
    return None


def served_in_file():
    """crates/serve/src/decide.rs:served_in_file."""
    types = [d for r in DECIDE_ROWS for d in r["in_file_decision"]]
    archs = [a for r in DECIDE_ROWS for a in r["in_file_arch"]]
    return f"{', '.join(types)} by decision type and {', '.join(archs)} by architecture"


def no_head(arch):
    """The start of crates/serve/src/decide.rs:no_head's refusal, with the rows' head repos."""
    heads = ", ".join(f"{r['name']} (head repo {r['head_repo']}; --hf {r['quant_repo']})"
                      for r in DECIDE_ROWS if r["head_repo"] is not None)
    return (f"a {arch} file with no head: a decision model is named by its head; --hf <repo>[:<quant>] of a repo "
            f"whose model card names a row's head repo as the model it quantizes seats it: {heads}")


def route(arch, decision, repo, card, spec=None):
    """The seat `bloomery-serve --hf <repo>` seats a first shard of architecture `arch` in
    (bloomery_serve.rs:run: head_of, then seat_of_file), and whether the model reader takes it.
    `spec` is the architectures the reader takes (SPEC). -> (seat word, None) or (None, refusal text)."""
    spec = SPEC if spec is None else spec
    if arch is None:
        return None, ("the first shard names no architecture; the seats are --model "
                      + ", ".join(SEAT_WORDS))
    picked = decide_pick(arch, decision, repo, card)
    if picked is not None and picked[0] == "refused":
        return None, picked[1]
    if picked is not None:
        seat = DECIDE_WORD
    elif any(arch in r["backbones"] for r in DECIDE_ROWS):
        return None, no_head(arch)
    else:
        seat = next((s for s in SEAT_ORDER if serves(s, arch)), None)
        if seat is None:
            return None, f"a {arch} file, which no seat serves; the seats are --model " + ", ".join(SEAT_WORDS)
    if arch not in spec:
        return None, (f"the seat {seat} takes a {arch} file, which model::arch::spec does not read: "
                      f"UnknownArchitecture({arch})")
    return seat, None


# ------------------------------------------------------------------------------------------------
# The hf crate's picking, mirrored from crates/hf/src/lib.rs.

def ascii_lower(s):
    return "".join(c.lower() if "A" <= c <= "Z" else c for c in s)


def alnum(c):
    return c.isascii() and c.isalnum()


def is_model(name):
    """crates/hf/src/lib.rs:is_model (llama.cpp's gguf_filename_is_model)."""
    return name.endswith(".gguf") and not any(w in name for w in
                                              ("mmproj", "imatrix", "mtp-", "eagle3-", "dflash-", "dspark-"))


def split_of(stem):
    """crates/hf/src/lib.rs:split_of: (prefix, index, count) of a `-NNNNN-of-NNNNN` stem, else None."""
    if "-of-" not in stem:
        return None
    head, count = stem.rsplit("-of-", 1)
    if "-" not in head:
        return None
    prefix, index = head.rsplit("-", 1)

    def five(s):
        return len(s) == 5 and all("0" <= c <= "9" for c in s)

    if not five(index) or not five(count) or not prefix:
        return None
    return prefix, int(index), int(count)


def tag_of(name):
    """crates/hf/src/lib.rs:tag_of (llama.cpp's re_tag), upper-cased; empty when there is none."""
    base = name.rsplit("/", 1)[-1]
    tail = 0
    for c in reversed(base):
        if alnum(c) or c == "_":
            tail += 1
        else:
            break
    at = len(base) - tail
    if tail == 0 or at == 0 or base[at - 1] not in "-.":
        return ""
    return "".join(c.upper() if "a" <= c <= "z" else c for c in base[at:])


def gguf_sets(repo, entries):
    """crates/hf/src/lib.rs:gguf_sets: the GGUF model sets of `entries` (Entry), in the listing's order of their
    first file; a split set with a missing or doubled shard is a Red."""
    sets = []
    for e in entries:
        if not is_model(e.path.rsplit("/", 1)[-1]):
            continue
        stem = e.path[:-len(".gguf")]
        parts = split_of(stem)
        name, count = (parts[0], parts[2]) if parts else (stem, 1)
        hit = next((s for s in sets if s["name"] == name and s["count"] == count and count > 1), None)
        if hit is not None:
            hit["files"].append(e)
        else:
            sets.append({"name": name, "tag": tag_of(name), "files": [e], "count": count})
    for s in sets:
        if s["count"] == 1:
            continue

        def index(f):
            p = split_of(f.path[:-5])
            return p[1] if p else 0

        s["files"].sort(key=index)
        have = [index(f) for f in s["files"]]
        missing = [i for i in range(1, s["count"] + 1) if i not in have]
        if missing or len(have) != s["count"]:
            raise Red(f"{repo}: the split set {s['name']} of {s['count']} shards lacks shard(s) {missing}")
    return sets


def names(quant, name):
    """crates/hf/src/lib.rs:names: whether `quant` names the set `name` (case-insensitive; a run of the file name
    that starts it or follows `-`, `.` or `_`, and ends it or is followed by `-` or `.`)."""
    base = ascii_lower(name.rsplit("/", 1)[-1])
    q = ascii_lower(quant)
    if not q:
        return False
    start = 0
    while True:
        at = base.find(q, start)
        if at < 0:
            return False
        end = at + len(q)
        left = at == 0 or base[at - 1] in "-._"
        right = end == len(base) or base[end] in "-."
        if left and right:
            return True
        start = at + 1


def shown(s):
    return f"{s['name']} ({s['count']} shards)" if s["count"] > 1 else s["name"]


def pick(repo, sets, quant):
    """crates/hf/src/lib.rs:pick: the one set `quant` names; no set, no match and two matches are a Red."""
    if not sets:
        raise Red(f"{repo} holds no GGUF model file")
    hits = [s for s in sets if names(quant, s["name"])]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        tags = ", ".join(s["tag"] or s["name"] for s in sets)
        raise Red(f'{repo}: no GGUF set carries the quant "{quant}"; its quants: {tags}')
    raise Red(f'{repo}: the quant "{quant}" matches {len(hits)} sets: {", ".join(shown(s) for s in hits)}')


def parse_listing(repo, text):
    """crates/hf/src/lib.rs:listing: the files (Entry) of one page of the tree listing, a digest the LFS sha256 or
    the git blob sha1; directories left out; anything this does not read is a Red."""
    def bad(why):
        return Red(f"{repo}: the API's listing is not one this reads: {why}")

    try:
        items = json.loads(text)
    except ValueError as e:
        raise bad(str(e))
    if not isinstance(items, list):
        raise bad("the answer is not a list")
    out = []
    for item in items:
        ty = item.get("type") if isinstance(item, dict) else None
        if ty == "directory":
            continue
        if ty != "file":
            raise bad(f"an entry of type {ty!r}: {item}")
        path, size = item.get("path"), item.get("size")
        if not isinstance(path, str):
            raise bad(f"a file with no path: {item}")
        if not isinstance(size, int) or isinstance(size, bool):
            raise bad(f"{path} has no size")
        lfs = item.get("lfs")
        if lfs is not None:
            if not isinstance(lfs, dict) or lfs.get("size") != size:
                raise bad(f"{path}: its LFS size is not its size {size}")
            digest = hex_of(lfs.get("oid"), 64)
            if digest is None:
                raise bad(f"{path}: its LFS oid is not a sha256")
            out.append(Entry(path, size, "sha256", digest))
        else:
            digest = hex_of(item.get("oid"), 40)
            if digest is None:
                raise bad(f"{path}: its oid is not a git sha1")
            out.append(Entry(path, size, "git-sha1", digest))
    return out


def hex_of(v, length):
    """crates/hf/src/lib.rs:hex_of: `v` as lowercase hex of `length` digits, else None."""
    if isinstance(v, str) and len(v) == length and all(c in "0123456789abcdefABCDEF" for c in v):
        return v.lower()
    return None


def next_link(headers):
    """crates/hf/src/lib.rs:next_link: the `rel="next"` URL of a response's Link header, if one."""
    for line in headers.splitlines():
        if ":" not in line:
            continue
        name, value = line.split(":", 1)
        if name.strip().lower() != "link":
            continue
        for link in value.split(","):
            parts = link.split(";")
            if any(p.strip().replace(" ", "") == 'rel="next"' for p in parts[1:]):
                return parts[0].strip().lstrip("<").rstrip(">")
    return None


def parse_repo(text):
    """crates/hf/src/lib.rs:RepoRef::parse: (repo, quant or None), or None for a string it refuses."""
    repo, _, quant = text.partition(":")
    has_quant = ":" in text

    def word(p):
        return bool(p) and p not in (".", "..") and all(alnum(c) or c in "-_." for c in p)

    if "/" not in repo:
        return None
    owner, name = repo.split("/", 1)
    if not word(owner) or not word(name) or (has_quant and not word(quant)):
        return None
    return repo, (quant if has_quant else None)


def card_quantizes(text):
    """crates/hf/src/card.rs:quantizes: the one `base_model` of the front matter whose `base_model_relation` is
    `quantized`, else None."""
    lines = text.lstrip("﻿").splitlines()
    if not lines or lines[0].rstrip() != "---":
        return None
    base, relation, in_base = [], None, False

    def unquote(v):
        v = v.strip()
        for q in ('"', "'"):
            if len(v) >= 2 and v.startswith(q) and v.endswith(q):
                return v[1:-1]
        return v

    for line in lines[1:]:
        line = line.rstrip()
        if line == "---":
            break
        stripped = line.lstrip()
        item = stripped[2:] if stripped.startswith("- ") else None
        if in_base and item is not None:
            base.append(unquote(item))
            continue
        in_base = False
        if line[:1].isspace():
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        key = key.strip()
        if key == "base_model" and not value:
            in_base = True
        elif key == "base_model":
            if value.startswith("[") and value.endswith("]"):
                base.extend(unquote(v) for v in value[1:-1].split(","))
            else:
                base.append(unquote(value))
        elif key == "base_model_relation":
            relation = unquote(value)
    if relation == "quantized" and len(base) == 1 and base[0]:
        return base[0]
    return None


# ------------------------------------------------------------------------------------------------
# The README's Use table.

def use_rows(text):
    """The rows of the first table under `## Use`: (seat word, repo, quant, line number). Every data row must hold
    `bloomery-serve --hf <repo>:<quant>` in its last cell; a table with no row, or a row that does not, is a
    Refuse(65) naming the line."""
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.rstrip() == "## Use"), None)
    if start is None:
        raise Refuse(65, f"{README}: no `## Use` heading, so no table of rows to check")
    table = []
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("## "):
            break
        if lines[i].lstrip().startswith("|"):
            table.append((i + 1, lines[i]))
        elif table:
            break
    if len(table) < 3:
        raise Refuse(65, f"{README}: the Use table under `## Use` has {len(table)} lines, not a header, a "
                         "separator and rows")

    def cells(line):
        body = line.strip()
        body = body[1:] if body.startswith("|") else body
        body = body[:-1] if body.endswith("|") else body
        return [c.strip() for c in re.split(r"(?<!\\)\|", body)]

    if not all(re.fullmatch(r":?-+:?", c) for c in cells(table[1][1])):
        raise Refuse(65, f"{README}:{table[1][0]}: the second line of the Use table is no separator row")
    rows = []
    for no, line in table[2:]:
        c = cells(line)
        if len(c) < 3:
            raise Refuse(65, f"{README}:{no}: a Use row of {len(c)} cells; want the seat, what it serves and the command")
        word = c[0].strip("`")
        if word not in SEAT_WORDS:
            raise Refuse(65, f"{README}:{no}: the seat word {word!r} is none of {', '.join(SEAT_WORDS)}")
        m = re.search(r"`bloomery-serve --hf (\S+)`", c[-1])
        if not m:
            raise Refuse(65, f"{README}:{no}: the last cell of the {word} row holds no `bloomery-serve --hf "
                             f"<repo>:<quant>`: {c[-1]!r}")
        ref = parse_repo(m.group(1))
        if ref is None or ref[1] is None:
            raise Refuse(65, f"{README}:{no}: the --hf text {m.group(1)!r} of the {word} row is not "
                             "<owner>/<name>:<quant> (each part of letters, digits, '-', '_' and '.')")
        rows.append((word, ref[0], ref[1], no))
    return rows


# ------------------------------------------------------------------------------------------------
# The hub, by curl.

# crates/hf/src/fetch.rs:network_failure and hub_unavailable.
NETWORK_CODES = (5, 6, 7, 28, 35, 52, 55, 56)


def curl_why(code, stderr):
    """crates/hf/src/fetch.rs:curl_why."""
    text = stderr.strip()
    status = http_status(stderr)
    if status in (401, 403):
        return f"{text} (the repo may be gated)"
    if status == 404:
        return f"{text} (the repo or the file is not there)"
    return text or f"curl exit {code}"


def http_status(stderr):
    m = re.search(r"returned error: (\d+)", stderr)
    return int(m.group(1)) if m else None


class Hub:
    """The hub's reads. curl, as crates/hf/src/fetch.rs: Python's own TLS can fail on a machine curl works on."""

    def __init__(self, scratch):
        self.scratch = scratch
        self.requests = 0
        self.bytes = 0

    def _curl(self, what, args):
        cmd = ["curl", "--fail", "--location", "--silent", "--show-error", "--connect-timeout", CONNECT_TIMEOUT,
               "--max-time", FETCH_TIMEOUT] + args
        self.requests += 1
        try:
            done = subprocess.run(cmd, capture_output=True)
        except FileNotFoundError:
            raise Refuse(69, "curl is not on PATH; the hub is read with it")
        err = done.stderr.decode("utf-8", "replace")
        if done.returncode == 0:
            return done.stdout
        status = http_status(err)
        if done.returncode in NETWORK_CODES or (done.returncode == 22 and status is not None
                                                and (status == 429 or 500 <= status <= 599)):
            raise Refuse(69, f"curl {what}: {curl_why(done.returncode, err)} (the network or the hub is down, "
                             "not a row's refusal)")
        raise Red(f"curl {what}: {curl_why(done.returncode, err)}")

    def listing(self, repo):
        """Every file of `repo`'s main branch, each page followed to its last."""
        url = f"{HUB}/api/models/{repo}/tree/main?recursive=true"
        headers = os.path.join(self.scratch, "listing.headers")
        out = []
        while True:
            body = self._curl(f"listing {url}", ["--dump-header", headers, url])
            self.bytes += len(body)
            with open(headers, "r", errors="replace") as f:
                head = f.read()
            out.extend(parse_listing(repo, body.decode("utf-8", "replace")))
            url = next_link(head)
            if url is None:
                return out

    def range(self, repo, path, start, end):
        """The bytes [start, end) of `path` of `repo`: one range read, checked to be a 206 of that length (a
        server that sends the whole file is refused before the body, by --max-filesize)."""
        want = end - start
        url = f"{HUB}/{repo}/resolve/main/{urllib.parse.quote(path)}"
        part = os.path.join(self.scratch, "range.part")
        out = self._curl(f"range {start}-{end - 1} of {url}",
                         ["-r", f"{start}-{end - 1}", "--max-filesize", str(want + 1024), "-o", part,
                          "-w", "%{http_code} %{size_download}", url])
        code, _, got = out.decode().partition(" ")
        with open(part, "rb") as f:
            data = f.read()
        os.unlink(part)
        if code != "206" or int(got) != want or len(data) != want:
            raise Red(f"{repo}: the range {start}-{end - 1} of {path} answered HTTP {code} with {len(data)} bytes, "
                      f"not a 206 of {want}")
        self.bytes += want
        return data

    def card(self, repo, entries):
        """The repo `repo`'s model card says it quantizes, or None (crates/gpu-gates/src/model_file.rs:
        quantized_from: no README.md in the listing, or no such front matter, is None)."""
        size = {e.path: e.size for e in entries}.get("README.md")
        if size is None:
            return None
        data = self.range(repo, "README.md", 0, min(size, FIRST_READ))
        return card_quantizes(data.decode("utf-8", "replace"))


# ------------------------------------------------------------------------------------------------
# One row.

_GGUF = None


def gguf_ranges():
    """tools/ref/gguf-ranges.py, the one header parser, loaded by path."""
    global _GGUF
    if _GGUF is None:
        path = os.path.join(ROOT, "tools", "ref", "gguf-ranges.py")
        if not os.path.isfile(path):
            raise Refuse(65, f"{path}: the header parser this tool reuses is not there")
        spec = importlib.util.spec_from_file_location("gguf_ranges", path)
        _GGUF = importlib.util.module_from_spec(spec)
        keep, sys.dont_write_bytecode = sys.dont_write_bytecode, True
        try:
            spec.loader.exec_module(_GGUF)
        finally:
            sys.dont_write_bytecode = keep
    return _GGUF


class NeedMore(Exception):
    """The header runs past the bytes read."""


def parse_header(local, size):
    """(meta, tensors) of the GGUF header in the file `local`, a prefix of a file of `size` bytes. A header that
    ends past the prefix is NeedMore; any other refusal of the parser is a Red naming it."""
    g = gguf_ranges()
    try:
        meta, tensors, _, _, _ = g.header(local, size=size)
    except g.Refusal as e:
        text = str(e).replace(local, "the header")
        if e.code == 65 and "the header ends at byte" in text:
            raise NeedMore(text)
        raise Red(f"the header does not parse: {text}")
    return meta, tensors


def read_header(fetch, local, size, budget):
    """Read the header of a file of `size` bytes through `fetch(start, end)`: FIRST_READ bytes, then the next slice
    of as many again until the header parses, never more than `budget` bytes in all. -> (meta, tensors, bytes read)
    or (None, None, bytes read) when it is still unfinished at the budget."""
    have, want = 0, min(FIRST_READ, size, budget)
    with open(local, "wb"):
        pass
    while True:
        if want > have:
            data = fetch(have, want)
            with open(local, "ab") as f:
                f.write(data)
            have = want
        try:
            meta, tensors = parse_header(local, size)
            return meta, tensors, have
        except NeedMore as e:
            if have >= size:
                raise Red(f"the file has {size} bytes and its header runs past them: {e}")
            if have >= budget:
                return None, None, have
            want = min(have * 2, size, budget)


def meta_str(meta, key):
    v = meta.get(key)
    return v.decode("utf-8", "replace") if isinstance(v, bytes) else v


EXPS = re.compile(r"blk\.\d+\.ffn_(gate|up|down|gate_up)_exps(?:\.weight)?")
ROLE_ORDER = ("gate", "up", "down", "gate_up")


def type_name(ty):
    return GGML_TYPES.get(ty) or EXTRA_TYPES.get(ty) or f"type{ty}"


def type_facts(tensors_by_shard):
    """(embd, exps) as printed: `token_embd.weight`'s type and the routed-expert types by role, over the shards'
    tensor lists read; each None where no list holds one."""
    embd, exps = set(), {}
    for tensors in tensors_by_shard:
        for name, ty, _, _ in tensors:
            if name == "token_embd.weight":
                embd.add(type_name(ty))
            m = EXPS.fullmatch(name)
            if m:
                exps.setdefault(m.group(1), set()).add(type_name(ty))
    e = "/".join(sorted(embd)) if embd else None
    x = ",".join(f"{r}:{'/'.join(sorted(exps[r]))}" for r in ROLE_ORDER if r in exps) if exps else None
    return e, x


def anchor_line(path, anchor, what):
    """The line of `path` (relative to the repo root) that holds `anchor`, as `path:line`."""
    try:
        with open(os.path.join(ROOT, path), encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError as e:
        raise Refuse(65, f"{path}: the place {what} cannot be read: {e}")
    at = next((i + 1 for i, l in enumerate(lines) if anchor in l), None)
    if at is None:
        raise Refuse(65, f"{path}: no line with `{anchor}`, the place {what}")
    return f"{path}:{at}"


def places(seat):
    """`file:line` of each place the seat's weight types are decided in."""
    return [anchor_line(p, a, f"the types of the {seat} seat are decided in") for p, a in TYPE_PLACES[seat]]


def body35_lists_of(name, head):
    """The lists of body35.rs the tensor `name` is read against, `head` being the head tensor's name."""
    out = [lst for rx, lst in BODY35_SITES if re.fullmatch(rx, name)]
    if name == BODY35_EMBED_TENSOR:
        out.append("EMBED")
    if name == head:
        out.append("HEAD_TY")
    return out


_BODY35 = None


def body35_lists(lists=None):
    """The lists of body35.rs, {list name: [ggml type names]}, read from the source; `lists` stands in for the source
    (the self-test's). A source this cannot read, or a list a site reads that it does not hold, is a Refuse(65)."""
    global _BODY35
    if lists is None:
        if _BODY35 is None:
            try:
                _BODY35 = src_body35(Src(BODY35_RS, read_source(BODY35_RS)), Src(SITE_RS, read_source(SITE_RS)))[0]
            except SourceError as e:
                raise Refuse(65, f"the type lists of the decide seat's body cannot be read: {e}")
        lists = _BODY35
    wanted = {lst for _, lst in BODY35_SITES} | {"EMBED", "HEAD_TY"}
    if wanted - set(lists):
        raise Refuse(65, f"{BODY35_RS}: no list {sorted(wanted - set(lists))}, which a site of this tool's table reads")
    return lists


def body35_types(tensors_by_shard, lists=None):
    """The decide seat's backbone body against the tensors of the headers read: (refusal or None, tensors checked).
    The refusal names the first tensor, in file order, whose type is outside the list its site reads."""
    lists = body35_lists(lists)
    seen = {}
    for tensors in tensors_by_shard:
        for name, ty, _, _ in tensors:
            seen[name] = ty
    head = BODY35_HEAD_TENSOR if (BODY35_HEAD_TENSOR in seen or BODY35_EMBED_TENSOR not in seen) \
        else BODY35_EMBED_TENSOR
    checked = 0
    for name, ty in seen.items():
        for lst in body35_lists_of(name, head):
            checked += 1
            if type_name(ty) not in lists[lst]:
                where = anchor_line(BODY35_RS, f"const {lst}: &[SiteTy]", "the list is declared in")
                return (f"{name} is {type_name(ty)}; {where} {lst} takes {', '.join(lists[lst])}"), checked
    return None, checked


def check_row(row, hub, scratch, picked=None):
    """One README row: (stdout line, green). The hub's own refusals of the row's repo are a red line; the network's
    are a Refuse(69). The set the quant picked is `picked["files"]` (Entry list) once it is."""
    seat, repo, quant, _ = row
    unread = "unread"
    shard1, arch = "none", unread
    head = f"hf-arch {seat} {repo}:{quant}"
    spent = [0]

    def fetch(p):
        def go(start, end):
            spent[0] += end - start
            return hub.range(repo, p, start, end)
        return go

    try:
        entries = hub.listing(repo)
        chosen = pick(repo, gguf_sets(repo, entries), quant)
        if picked is not None:
            picked["files"] = chosen["files"]
        shard1, size = chosen["files"][0].path, chosen["files"][0].size
        local = os.path.join(scratch, "header.bin")
        meta, tensors, n = read_header(fetch(shard1), local, size, ROW_BUDGET)
        if meta is None:
            return (f"{head} shard1={shard1} arch={unread} route=REFUSED: shard 1's header is unfinished at "
                    f"{n // MIB} MiB, so its architecture is unread embd=not-reached exps=not-reached "
                    f"types=not-reached"), False
        arch = meta_str(meta, "general.architecture")
        decision = meta_str(meta, f"{arch}.decision.type") if arch else None
        seat_of, why = route(arch, decision, repo, lambda r: hub.card(r, entries))
        if why is None and seat_of != seat:
            why = f"the router seats a {arch} file in the {seat_of} seat, and the README row says {seat}"
        shown_route = "ok" if why is None else f"REFUSED: {why}"
    except Red as e:
        return (f"{head} shard1={shard1} arch={arch} route=REFUSED: {e} embd={unread} exps={unread} "
                f"types={unread}"), False
    lists, tail, types, green = [tensors], "", None, why is None
    for e in chosen["files"][1:]:
        p, s = e.path, e.size
        if ROW_BUDGET - spent[0] <= 0:
            tail = "+unread-shards"
            break
        try:
            _, t2, _ = read_header(fetch(p), local, s, ROW_BUDGET - spent[0])
        except Red as e:
            types, green = f"REFUSED: {p}: {e}", False
            break
        if t2 is None:
            tail = "+unread-shards"
            break
        lists.append(t2)
    embd, exps = type_facts(lists)
    if types is None and arch in BODY35_ARCHS:
        if tail:
            types = "not-reached (a later shard's header is unread)"
        else:
            refusal, n = body35_types(lists)
            if refusal:
                types, green = f"REFUSED: {refusal}", False
            else:
                types = (f"ok ({n} site tensors against the lists at "
                         f"{anchor_line(BODY35_RS, 'const PROJ: &[SiteTy]', 'the lists are declared in')})")
    elif types is None:
        types = "printed (no one owner: " + ", ".join(places(seat)) + ")"
    return (f"{head} shard1={shard1} arch={arch or '<missing>'} route={shown_route} embd={embd or 'not-found'}{tail} "
            f"exps={exps or 'none'}{tail} types={types}"), green


# ------------------------------------------------------------------------------------------------
# The box's copies of the files the gates load, beside the hub's: size, and the digest where one is known.

# Each seat's gate files on the box: (the path of the set's first shard, or of its file, and the source file that
# names it). --self-test and --live --box grep that file for the exact path, so a path that moved is a refusal. A
# split set's other shards are the hub's shard names in the same directory.
BOX_FILES = {
    "ds41": (("/models/DeepSeek-V4.1-Flash-Q3_K_M/DeepSeek-V4.1-Flash-Q3_K_M-00001-of-00009.gguf",
              "tools/ref/models/deepseek41.sh"),),
    "glm": (("/models/GLM-5.3-Flash-UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf",
             "tools/ref/models/glm5next.sh"),),
    "qwen38": (("/models/Qwen3.8-Flash-Next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
                "tools/ref/models/qwen4exp.sh"),),
    "qwen3": (("/models/Qwen3-30B-A3B/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf", "tools/ref/models/qwen3moe.sh"),),
    "mimo2": (("/models/MiMo-V2.6-Flash-MOPD/MiMo-V2.6-Flash-MOPD-MXFP4-00001-of-00002.gguf",
               "tools/ref/models/mimo2.sh"),),
    "decide": (("/models/clef-flash/Cloudflare_clef-flash-Q5_K_M.gguf", "justfile"),
               ("/root/models/clef-flash/Cloudflare_clef-flash-Q5_K_M.gguf", "justfile")),
}
# The box's ik tree, spelled in two pieces so that this file, which check-recipes runs, does not read as a script of a
# reference tree (tools/recipes.py IK_READ matches the whole path), which would make check-recipes skippable.
IK_TREE = "/home/" + "user/ik-idxkey"
# A box file that stays at the box's bytes on purpose, by name, until its expiry: a file of `box_size` bytes on the box
# against `live_size` on the hub is `excepted: <reason>` instead of a size FAIL while `expiry` (a box file and a string)
# is absent from that file; the day the file holds the string it is a FAIL naming `name`, ending with `then`. `cites`
# are (source file, the text it must hold): --self-test and --live --box grep each, the expiry file lies under the first
# text, and the key is a BOX_FILES path of `seat`, so a cited path that moved is a refusal.
BoxException = collections.namedtuple("BoxException", "seat name box_size live_size reason expiry then cites")
BOX_EXCEPTIONS = {
    "/models/GLM-5.3-Flash-UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf": BoxException(
        seat="glm", name="the GLM shard 1 exception", box_size=9429859, live_size=9429984,
        reason="ik, the GLM oracle, cannot open glm5-next (its llama-arch.cpp names glm5next) and the GLM reference sets "
               "are bound to this shard's path; the glm5-next file is staged at /models/GLM-5.3-Flash-UD-Q4_K_XL-glm5dash/",
        expiry=(IK_TREE + "/src/llama-arch.cpp", "glm5-next"),
        then="refresh the copy and the GLM reference sets",
        cites=(("tools/ref/models/glm5next.sh", IK_TREE),)),
}
GREP_STATES = ("found", "absent", "missing")
BOX_CACHE = os.path.join("~", ".cache", "bloomery", "hf-arch-box.tsv")
# What a hash run reads at: the drive's sequential read rate measured on this machine (rig-log's offload-bandwidth
# entry) and one core's sha256 rate measured on the box (the hasher's reads overlap its hashing, so a hash run is
# the slower of the two).
NVME_GBPS = 6.12
SHA256_GBPS = 2.0
HASH_CHUNK = 8 * MIB
PAGE = 4096


def box_table_problems(read=None):
    """BOX_FILES entries whose source file does not hold the path: messages, empty when each is where it says."""
    out = []
    for seat, entries in BOX_FILES.items():
        for path, source in entries:
            try:
                text = (read or read_source)(source)
            except SourceError as e:
                out.append(f"{seat}: {e}")
                continue
            if path not in text:
                out.append(f"{seat}: {source} no longer holds the path {path}")
    return out


def exception_problems(read=None):
    """BOX_EXCEPTIONS entries that do not hold: messages, empty when each key is a BOX_FILES path of its seat, each cited
    source holds its text and the expiry file lies under the first cited text."""
    out = []
    for path, ex in BOX_EXCEPTIONS.items():
        if path not in [p for p, _ in BOX_FILES.get(ex.seat, ())]:
            out.append(f"{ex.seat}: the exception's path {path} is no BOX_FILES path of the seat")
        for source, cited in ex.cites:
            try:
                text = (read or read_source)(source)
            except SourceError as e:
                out.append(f"{ex.seat}: {e}")
                continue
            if cited not in text:
                out.append(f"{ex.seat}: {source} no longer holds {cited}")
        if not ex.expiry[0].startswith(ex.cites[0][1] + "/"):
            out.append(f"{ex.seat}: the expiry file {ex.expiry[0]} is not under {ex.cites[0][1]}")
    return out


def exception_verdict(path, box_size, live_size, greps):
    """The named exception's verdict for a box file of `box_size` bytes against a live one of `live_size`: None unless
    `path` has one and the sizes are exactly the two it names, then `excepted: <reason>` while the expiry file on the
    box does not hold its string, else the FAIL naming why it cannot stand. `greps` is the box's answer per expiry."""
    ex = BOX_EXCEPTIONS.get(path)
    if ex is None or (box_size, live_size) != (ex.box_size, ex.live_size):
        return None
    file, string = ex.expiry
    if greps is None or (file, string) not in greps:
        raise Refuse(69, f"the box was not asked whether {file} holds {string}, so {ex.name} cannot be judged")
    if greps[(file, string)] == "missing":
        return f"FAIL: {file} is not on the box: the exception cannot be judged"
    if greps[(file, string)] == "found":
        return f"FAIL: {ex.name} expired: {file} now names {string}; {ex.then}"
    return f"excepted: {ex.reason}"


def odirect_sha256(path, size, o_direct=getattr(os, "O_DIRECT", None), opener=os.open):
    """The sha256 of the first `size` bytes of `path`, read with O_DIRECT into page-aligned (anonymous mmap) buffers
    and never through the page cache: a buffered read of a model set evicts the pages a running gate maps. A drive or
    a file system that refuses O_DIRECT is a Refuse(69); there is no buffered fallback. Reads and hashing overlap
    (two buffers, one reader thread)."""
    if o_direct is None:
        raise Refuse(69, f"{path}: this platform has no O_DIRECT, and a hash is never read through the page cache")
    try:
        fd = opener(path, os.O_RDONLY | o_direct)
    except OSError as e:
        raise Refuse(69, f"{path}: O_DIRECT is refused ({e.strerror}); there is no buffered fallback")
    try:
        def read_at(offset, view):
            try:
                return os.preadv(fd, [view], offset)
            except OSError as e:
                raise Refuse(69, f"{path}: the O_DIRECT read at byte {offset} failed ({e.strerror})")

        return hash_chunks(read_at, size, HASH_CHUNK, path)
    finally:
        os.close(fd)


def hash_chunks(read_at, size, chunk, what):
    """The sha256 of `size` bytes `read_at(offset, view) -> n` fills into page-aligned buffers of `chunk` bytes, a
    thread reading the next chunk while the last is hashed. A short read that leaves the next offset off a page, and a
    file that ends before `size`, are a Refuse(69) naming `what`."""
    h = hashlib.sha256()
    if size == 0:
        return h.hexdigest()
    bufs = [mmap.mmap(-1, chunk), mmap.mmap(-1, chunk)]
    views = [memoryview(b) for b in bufs]
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            off, i = 0, 0
            pending = pool.submit(read_at, 0, views[0])
            while off < size:
                n = pending.result()
                if n <= 0:
                    raise Refuse(69, f"{what}: the file ends at byte {off}, before its {size}")
                take = min(n, size - off)
                nxt = off + n
                if nxt < size:
                    if n % PAGE:
                        raise Refuse(69, f"{what}: a read of {n} bytes at byte {off} leaves the next read off a page")
                    pending = pool.submit(read_at, nxt, views[1 - i])
                h.update(views[i][:take])
                off, i = nxt, 1 - i
    finally:
        for v in views:
            v.release()
        for b in bufs:
            b.close()
    return h.hexdigest()


BOX_FACTS_USAGE = "usage: hf-arch-check.py box-facts [--hash] [--grep FILE STRING]... <paths...>"


def box_facts_args(argv):
    """`box-facts`'s (hash flag, [(file, string)] of its --grep options, paths); an option it does not take, a --grep
    with fewer than two values and no path are a Refuse(64)."""
    hashing, greps, paths, i = False, [], [], 0
    while i < len(argv):
        if argv[i] == "--hash":
            hashing, i = True, i + 1
        elif argv[i] == "--grep" and i + 2 < len(argv):
            greps.append((argv[i + 1], argv[i + 2]))
            i += 3
        elif argv[i].startswith("-"):
            raise Refuse(64, BOX_FACTS_USAGE)
        else:
            paths.append(argv[i])
            i += 1
    if not paths:
        raise Refuse(64, BOX_FACTS_USAGE)
    return hashing, greps, paths


def grep_state(file, string):
    """`found` or `absent` for `string` in `file`, `missing` for a file that is not there; any other read error is a
    Refuse(69), never `missing`."""
    try:
        with open(file, "rb") as f:
            data = f.read()
    except FileNotFoundError:
        return "missing"
    except OSError as e:
        raise Refuse(69, f"{file}: cannot be read ({e.strerror})")
    return "found" if string.encode() in data else "absent"


def cmd_box_facts(argv, out=None):
    """`box-facts [--hash] [--grep FILE STRING]... <paths...>`, run on the box: one `box-fact` line per path (size,
    mtime_ns, the contents of its `.verified` marker, the sha256 under --hash), one `box-grep` line per --grep (the file,
    the string, `found`, `absent` or `missing`), then `box-fact-end <n paths>`. A path that is not there is ABSENT."""
    out = out or sys.stdout
    hashing, greps, paths = box_facts_args(argv)
    for path in paths:
        try:
            st = os.stat(path)
        except FileNotFoundError:
            print(f"box-fact\t{path}\tABSENT", file=out)
            continue
        except OSError as e:
            print(f"box-fact\t{path}\tERROR\t{e.strerror}", file=out)
            continue
        try:
            with open(path + ".verified", encoding="utf-8", errors="replace") as f:
                marker = f.read().strip() or "-"
        except OSError:
            marker = "-"
        digest = odirect_sha256(path, st.st_size) if hashing else "-"
        print(f"box-fact\t{path}\t{st.st_size}\t{st.st_mtime_ns}\t{marker}\t{digest}", file=out, flush=True)
    for file, string in greps:
        print(f"box-grep\t{file}\t{string}\t{grep_state(file, string)}", file=out, flush=True)
    print(f"box-fact-end\t{len(paths)}", file=out)


def parse_box_facts(text, paths):
    """The `box-fact` lines of the box's answer for `paths`: {path: None (absent) | dict(size, mtime_ns, marker,
    sha256) | str (an error)}. An answer that lacks its end line or a path is a Refuse(69)."""
    return parse_box_answer(text, paths)[0]


def parse_box_answer(text, paths, greps=()):
    """The box's answer for `paths` and `greps` ([(file, string)]): (parse_box_facts' dict, {(file, string): found |
    absent | missing}). A `box-fact` or `box-grep` line this tool cannot read, a `box-grep` for a pair that was not
    asked, an answer that lacks its end line, a path or a pair: a Refuse(69). Lines of other kinds are skipped."""
    facts, found, ended = {}, {}, None
    for line in text.splitlines():
        parts = line.split("\t")
        if parts[0] == "box-fact-end" and len(parts) == 2:
            ended = parts[1]
        elif parts[0] == "box-grep":
            if len(parts) != 4 or parts[3] not in GREP_STATES or (parts[1], parts[2]) not in greps:
                raise Refuse(69, f"the box's answer holds a line this tool cannot read: {line!r}")
            found[(parts[1], parts[2])] = parts[3]
        elif parts[0] == "box-fact" and len(parts) >= 3:
            if parts[2] == "ABSENT":
                facts[parts[1]] = None
            elif parts[2] == "ERROR":
                facts[parts[1]] = "cannot stat: " + "\t".join(parts[3:])
            elif len(parts) == 6 and parts[2].isdigit() and parts[3].isdigit():
                facts[parts[1]] = {"size": int(parts[2]), "mtime_ns": int(parts[3]),
                                   "marker": None if parts[4] == "-" else parts[4],
                                   "sha256": None if parts[5] == "-" else parts[5]}
            else:
                raise Refuse(69, f"the box's answer holds a line this tool cannot read: {line!r}")
    if ended != str(len(paths)) or set(facts) != set(paths) or set(found) != set(greps):
        raise Refuse(69, f"the box's answer is incomplete: {len(facts)} of {len(paths)} paths, {len(found)} of "
                         f"{len(set(greps))} greps, end line {ended!r}")
    return facts, found


def box_runner(argv):
    """The box's `box-facts` stdout for `argv`, through tools/box.sh (it syncs this tree to its remote directory,
    BLOOMERY_REMOTE or the tree's own, and builds nothing). A box that does not answer is a Refuse(69)."""
    cmd = "python3 tools/hf-arch-check.py box-facts " + " ".join(shlex.quote(a) for a in argv)
    done = subprocess.run([os.path.join(ROOT, "tools", "box.sh"), cmd], capture_output=True, text=True)
    if done.returncode != 0:
        tail = " | ".join((done.stderr.strip().splitlines() or ["no output"])[-3:])
        raise Refuse(69, f"tools/box.sh exit {done.returncode}"
                         f"{' (the box is busy: contention, not a failure)' if done.returncode == 75 else ''}: {tail}")
    return done.stdout


def cache_load(path):
    """The digests a hash run left on this machine: {(box path, size, mtime_ns): sha256}."""
    out = {}
    try:
        with open(os.path.expanduser(path), encoding="utf-8") as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) == 4 and p[1].isdigit() and p[2].isdigit():
                    out[(p[0], int(p[1]), int(p[2]))] = p[3]
    except OSError:
        pass
    return out


def cache_store(path, entries):
    """`entries` added to the digest cache at `path`."""
    path = os.path.expanduser(path)
    have = cache_load(path)
    have.update(entries)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for (p, size, mtime), digest in sorted(have.items()):
            f.write(f"{p}\t{size}\t{mtime}\t{digest}\n")


def box_paths(seat, live_files):
    """[(box set path, [(live Entry, box path)] or a refusal text)] for `seat`: the hub's shard names in the
    directory of each box path of BOX_FILES. A box path whose file name is not the hub's first file's is the refusal."""
    out = []
    for first, _ in BOX_FILES[seat]:
        d, name = os.path.split(first)
        head = live_files[0].path.rsplit("/", 1)[-1]
        if name != head:
            out.append((first, f"the box names {name}, the hub's first file is {head}"))
        else:
            out.append((first, [(e, f"{d}/{e.path.rsplit('/', 1)[-1]}") for e in live_files]))
    return out


def compare_file(live, path, fact, cached, hashed, greps=None):
    """One live file against the box's `fact`: (digest source, verdict). The verdict is `ok`, `unhashed` (sizes equal,
    no digest known), `excepted: ...` (sizes unequal, as BOX_EXCEPTIONS names them, its expiry not reached) or `FAIL: ...`
    naming both values. `cached` is a digest from this machine's cache, `hashed` one a hash run just read, `greps` the
    box's answer per expiry."""
    if fact is None:
        return "-", "FAIL: absent on the box"
    if isinstance(fact, str):
        return "-", f"FAIL: {fact}"
    if fact["size"] != live.size:
        return "-", (exception_verdict(path, fact["size"], live.size, greps)
                     or f"FAIL: size box={fact['size']} live={live.size}")
    if live.check != "sha256":
        return "-", "unhashed"
    source, digest = "unhashed", None
    if fact["marker"] is not None:
        digest = hex_of(fact["marker"], len(live.digest))
        if digest is None:
            return "marker", f"FAIL: the .verified marker {fact['marker']!r} is not a {len(live.digest)}-digit digest"
        source = "marker"
    elif fact["sha256"] is not None:
        digest, source = fact["sha256"], "hash"
    elif cached is not None:
        digest, source = cached, "cache"
    elif hashed is not None:
        digest, source = hashed, "hash"
    if digest is None:
        return "unhashed", "unhashed"
    if digest != live.digest:
        return source, f"FAIL: digest box={digest} live={live.digest}"
    return source, "ok"


def hash_wall(nbytes):
    """Seconds to read and hash `nbytes` on the box, one file at a time: the slower of the drive and one core [derived]."""
    return nbytes / (min(NVME_GBPS, SHA256_GBPS) * 1e9)


def box_compare(picked, runner, cache_path, hash_misses, out):
    """Compare each picked set ({row: (seat, repo, quant, [Entry])}) with the box's files. Prints one `hf-arch-box`
    line per file and a summary; -> ({row: green}, fail count)."""
    problems = box_table_problems() + exception_problems()
    if problems:
        raise Refuse(65, "the box paths of the gates have moved: " + "; ".join(problems))
    plan, paths = {}, []
    for key, (seat, repo, quant, files) in picked.items():
        plan[key] = box_paths(seat, files) if seat in BOX_FILES else [(None, f"no box file is named for the {seat} seat")]
        for _, pairs in plan[key]:
            if not isinstance(pairs, str):
                paths += [bp for _, bp in pairs]
    asks = [ex.expiry for path, ex in BOX_EXCEPTIONS.items() if path in paths]
    facts, greps = {}, {}
    if paths:
        facts, greps = parse_box_answer(runner(paths + [a for file, string in asks for a in ("--grep", file, string)]),
                                        paths, asks)
    cache = cache_load(cache_path)
    hashed = {}
    for attempt in (0, 1):
        results = []
        for key, sets in plan.items():
            seat, repo, quant, files = picked[key]
            for first, pairs in sets:
                if isinstance(pairs, str):
                    results.append((key, first or "-", None, "-", f"FAIL: {pairs}"))
                    continue
                for e, bp in pairs:
                    f = facts[bp]
                    c = cache.get((bp, f["size"], f["mtime_ns"])) if isinstance(f, dict) else None
                    h = hashed.get((bp, f["size"], f["mtime_ns"])) if isinstance(f, dict) else None
                    src, verdict = compare_file(e, bp, f, c, h, greps)
                    results.append((key, bp, e, src, verdict))
        misses = [(bp, e) for _, bp, e, _, v in results if v == "unhashed" and e is not None]
        if attempt == 1 or not hash_misses or not misses:
            break
        total = sum(e.size for _, e in misses)
        print(f"hf-arch-box hash: {len(misses)} files, {total} bytes to read; wall ~ {hash_wall(total):.0f} s "
              f"[derived: bytes / min(NVMe {NVME_GBPS} GB/s, one core's sha256 {SHA256_GBPS} GB/s), one file at a time]",
              file=out, flush=True)
        answer = parse_box_facts(runner(["--hash"] + [bp for bp, _ in misses]), [bp for bp, _ in misses])
        new = {}
        for bp, f in answer.items():
            if isinstance(f, dict) and f["sha256"] is not None:
                new[(bp, f["size"], f["mtime_ns"])] = f["sha256"]
        hashed.update(new)
        cache_store(cache_path, new)
    green = {key: True for key in picked}
    counts = {"ok": 0, "fail": 0, "unhashed": 0, "excepted": 0}
    for key, bp, e, src, verdict in results:
        seat, repo, quant, _ = picked[key]
        kind = ("ok" if verdict == "ok" else "unhashed" if verdict == "unhashed"
                else "excepted" if verdict.startswith("excepted: ") else "fail")
        counts[kind] += 1
        green[key] = green[key] and kind != "fail"
        box = facts.get(bp)
        bsize = box["size"] if isinstance(box, dict) else "-"
        lsize = e.size if e is not None else "-"
        print(f"hf-arch-box {seat} {repo}:{quant} file={bp} box={bsize}B live={lsize}B digest={src} verdict={verdict}",
              file=out)
    print(f"hf-arch-box files={sum(counts.values())} ok={counts['ok']} fail={counts['fail']} "
          f"unhashed={counts['unhashed']} excepted={counts['excepted']}", file=out)
    misses = [(bp, e) for _, bp, e, _, v in results if v == "unhashed" and e is not None]
    if misses:
        total = sum(e.size for _, e in misses)
        print(f"hf-arch-box unhashed: {len(misses)} files, {total} bytes; `--live --box --box-hash` reads them with O_DIRECT, "
              f"wall ~ {hash_wall(total):.0f} s [derived: bytes / min(NVMe {NVME_GBPS} GB/s, one core's sha256 "
              f"{SHA256_GBPS} GB/s), one file at a time]; the paths: python3 tools/hf-arch-check.py box-facts --hash "
              + " ".join(bp for bp, _ in misses), file=out)
    return green, counts["fail"]


def live(readme_path=None, text=None, box=False, box_hash=False, hub_factory=None, runner=None, cache_path=BOX_CACHE):
    if text is None:
        path = readme_path or os.path.join(ROOT, README)
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except OSError as e:
            raise Refuse(65, f"{path}: {e.strerror}")
    rows = use_rows(text)
    green_rows, picked = [], {}
    with tempfile.TemporaryDirectory() as scratch:
        hub = (hub_factory or Hub)(scratch)
        for row in rows:
            before = (hub.requests, hub.bytes)
            got = {}
            line, green = check_row(row, hub, scratch, got)
            print(line, flush=True)
            print(f"hf-arch: {row[1]}:{row[2]} {hub.requests - before[0]} requests, {hub.bytes - before[1]} bytes",
                  file=sys.stderr, flush=True)
            green_rows.append(green)
            if "files" in got:
                picked[len(green_rows) - 1] = (row[0], row[1], row[2], got["files"])
    if box:
        boxed, _ = box_compare(picked, runner or box_runner, cache_path, box_hash, sys.stdout)
        green_rows = [g and boxed.get(i, True) for i, g in enumerate(green_rows)]
    ok = sum(green_rows)
    print(f"hf-arch rows={len(rows)} ok={ok} red={len(rows) - ok}")
    return 0 if ok == len(rows) else 1


# ------------------------------------------------------------------------------------------------
# The Rust sources the mirror copies, read by grep (a small scanner, not a Rust parser): a form it cannot read is
# a SourceError naming the file and line, never a pass.

SERVE_RS = "crates/gpu-gates/src/bin/bloomery_serve.rs"
ARCH_RS = "crates/model/src/arch/mod.rs"
DECIDE_SERVE_RS = "crates/serve/src/decide.rs"
DECIDE_SEAT_RS = "crates/gpu-gates/src/bin/shared/serve_seats/decide.rs"
QUANT_RS = "crates/gguf/src/quant.rs"
SITE_RS = "crates/gpu/src/site.rs"
RELEASE_RS = {"clef": "crates/decision/src/release/clef.rs", "lev": "crates/decision/src/release/lev.rs"}


class SourceError(Exception):
    pass


def mask(src):
    """`src` with comments and the insides of string and char literals blanked, newlines and length kept, so an
    offset in it is an offset in `src`."""
    out = list(src)
    n = len(src)

    def blank(a, b):
        for k in range(a, b):
            if out[k] != "\n":
                out[k] = " "

    i = 0
    while i < n:
        c = src[i]
        if src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            blank(i, j)
            i = j
        elif src.startswith("/*", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if src.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif src.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            blank(i, j)
            i = j
        elif c == "r" and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] == "_")) and re.match(r'r#*"', src[i:i + 12]):
            hashes = len(re.match(r"r(#*)", src[i:]).group(1))
            close = '"' + "#" * hashes
            j = src.find(close, i + 2 + hashes)
            j = n if j < 0 else j
            blank(i + 2 + hashes, j)
            i = j + len(close)
        elif c == '"':
            j = i + 1
            while j < n and src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            blank(i + 1, j)
            i = j + 1
        elif c == "'":
            if src[i + 1:i + 2] == "\\":
                j = src.find("'", i + 2)
                j = n if j < 0 else j
                blank(i + 1, j)
                i = j + 1
            elif src[i + 2:i + 3] == "'":
                blank(i + 1, i + 2)
                i += 3
            else:
                i += 1
        else:
            i += 1
    return "".join(out)


class Src:
    """One Rust source: its text, its masked text and the position helpers."""

    def __init__(self, path, text):
        self.path, self.text = path, text
        self.m = mask(text)

    def line(self, at):
        return self.text.count("\n", 0, at) + 1

    def err(self, at, why):
        return SourceError(f"{self.path}:{self.line(at)}: {why}")

    def close(self, at):
        """The offset of the bracket that closes the one at `at`."""
        depth = 0
        for k in range(at, len(self.m)):
            ch = self.m[k]
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
                if depth == 0:
                    return k
        raise self.err(at, "an opening bracket with no close")

    def fn_bodies(self, name):
        """(start, end) of the inside of each body of a `fn <name>`."""
        out = []
        for hit in re.finditer(r"\bfn\s+" + re.escape(name) + r"\b", self.m):
            k = self.m.find("(", hit.end())
            k = self.close(k) if k >= 0 else -1
            if k < 0:
                continue
            b = re.compile(r"[{;]").search(self.m, k)
            if b and self.m[b.start()] == "{":
                out.append((b.start() + 1, self.close(b.start())))
        return out

    def one_body(self, name):
        bodies = self.fn_bodies(name)
        if len(bodies) != 1:
            raise SourceError(f"{self.path}: {len(bodies)} bodies of `fn {name}`, want exactly 1")
        return bodies[0]

    def match_arms(self, lo, hi, head):
        """The top-level arms of the first `match` of `m[lo:hi]` whose head matches the regex `head`:
        [((pattern lo, hi), (expr lo, hi))]."""
        hit = re.compile(head).search(self.m, lo, hi)
        if not hit:
            raise self.err(lo, f"no `match` with a head like /{head}/")
        a = self.m.find("{", hit.start())
        end = self.close(a)
        arms, depth, i, seg = [], 0, a + 1, a + 1
        while i < end:
            ch = self.m[i]
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            elif depth == 0 and self.m.startswith("=>", i):
                pat = (seg, i)
                j = i + 2
                while self.m[j].isspace():
                    j += 1
                if self.m[j] == "{":
                    k = self.close(j)
                    expr, j = (j, k + 1), k + 1
                    while j < end and self.m[j].isspace():
                        j += 1
                    if j < end and self.m[j] == ",":
                        j += 1
                else:
                    d, k = 0, j
                    while k < end:
                        c2 = self.m[k]
                        if c2 in "([{":
                            d += 1
                        elif c2 in ")]}":
                            d -= 1
                        elif c2 == "," and d == 0:
                            break
                        k += 1
                    expr, j = (j, k), (k + 1 if k < end else k)
                arms.append((pat, expr))
                seg, i = j, j
                continue
            i += 1
        return arms

    def split_top(self, lo, hi, sep):
        """The spans of m[lo:hi] split on the top-level separator `sep`."""
        spans, depth, start, i = [], 0, lo, lo
        while i < hi:
            ch = self.m[i]
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            elif depth == 0 and self.m.startswith(sep, i):
                spans.append((start, i))
                start = i + len(sep)
                i = start
                continue
            i += 1
        spans.append((start, hi))
        return spans

    def literal(self, lo, hi):
        """The value of the string literal that is all of m[lo:hi], else None."""
        t = self.m[lo:hi].strip()
        if re.fullmatch(r'"[ \t]*"', t):
            a = lo + self.m[lo:hi].index('"')
            return self.text[a + 1:a + len(t) - 1]
        return None

    def consts(self):
        """name -> str or [str] or [(str, str)] of the `const`s that are string literals or arrays of them."""
        out = {}
        for hit in re.finditer(r"\bconst\s+(\w+)\s*:", self.m):
            eq = self.m.find("=", hit.end())
            end = self.split_top(eq + 1, len(self.m), ";")[0][1]
            val = self.value(eq + 1, end)
            if val is not None:
                out[hit.group(1)] = val
        return out

    def value(self, lo, hi):
        lit = self.literal(lo, hi)
        if lit is not None:
            return lit
        t = self.m[lo:hi].strip()
        if not t.startswith("&["):
            return None
        a = lo + self.m[lo:hi].index("[")
        b = self.close(a)
        items = []
        for x, y in self.split_top(a + 1, b, ","):
            if not self.m[x:y].strip():
                continue
            s = self.literal(x, y)
            if s is not None:
                items.append(s)
                continue
            t2 = self.m[x:y].strip()
            if t2.startswith("("):
                p = x + self.m[x:y].index("(")
                q = self.close(p)
                parts = [self.literal(u, v) for u, v in self.split_top(p + 1, q, ",") if self.m[u:v].strip()]
                if len(parts) == 2 and None not in parts:
                    items.append(tuple(parts))
                    continue
            return None
        return items


def names_of(src, lo, hi, consts):
    """The strings of an alternation `"a" | CONST | "b"` in src.m[lo:hi]: literals, and identifiers resolved
    through `consts` (a str or [str])."""
    out = []
    for a, b in src.split_top(lo, hi, "|"):
        lit = src.literal(a, b)
        if lit is not None:
            out.append(lit)
            continue
        ident = src.m[a:b].strip()
        if not re.fullmatch(r"[A-Za-z_][\w:]*", ident):
            raise src.err(a, f"an alternative this tool cannot read: {ident!r}")
        val = consts.get(ident.rsplit("::", 1)[-1])
        if isinstance(val, str):
            out.append(val)
        elif isinstance(val, list) and all(isinstance(v, str) for v in val):
            out.extend(val)
        else:
            raise src.err(a, f"the constant {ident} is not a string or a list of strings this tool resolves")
    return out


def src_seat_words(serve, consts_decide):
    """(Model variant -> seat word from `Model::word`, the order of `Model::ALL`, the decide WORD)."""
    words = {}
    for lo, hi in serve.fn_bodies("word"):
        for pat, expr in serve.match_arms(lo, hi, r"\bmatch\s+self\b"):
            m = re.fullmatch(r"\s*Model::(\w+)\s*", serve.m[pat[0]:pat[1]])
            lit = serve.literal(*expr)
            if m and lit is not None:
                words[m.group(1)] = lit
    if not words:
        raise SourceError(f"{serve.path}: no `Model::<Variant> => \"<word>\"` arms in a `fn word`")
    hit = re.search(r"\bconst\s+ALL\s*:\s*\[Model;\s*\d+\]\s*=\s*\[([^\]]*)\]", serve.m)
    if not hit:
        raise SourceError(f"{serve.path}: no `const ALL: [Model; N] = [...]`")
    order = re.findall(r"Model::(\w+)", hit.group(1))
    if "WORD" not in consts_decide or not isinstance(consts_decide["WORD"], str):
        raise SourceError(f"{DECIDE_SERVE_RS}: no `const WORD: &str`")
    return words, order, consts_decide["WORD"]


def src_serves(serve, arch_consts, words):
    """Model::serves: ({seat word: [names]}, {seat word: Arch variant it reads through Arch::from_name})."""
    lo, hi = serve.one_body("serves")
    direct, via = {}, {}
    for pat, (a, b) in serve.match_arms(lo, hi, r"\bmatch\s+self\b"):
        m = re.fullmatch(r"\s*Model::(\w+)\s*", serve.m[pat[0]:pat[1]])
        if not m or m.group(1) not in words:
            raise serve.err(pat[0], "a `serves` arm that is not `Model::<Variant>` of the seat words")
        word = words[m.group(1)]
        expr = serve.m[a:b].strip()
        if expr.startswith("matches!("):
            p = serve.m.index("(", a)
            q = serve.close(p)
            first, rest = serve.split_top(p + 1, q, ",")[:2]
            head = serve.m[first[0]:first[1]].strip()
            tail = serve.m[rest[0]:q]
            if head == "arch":
                direct[word] = names_of(serve, rest[0], q, arch_consts)
            elif head == "Arch::from_name(arch)":
                v = re.fullmatch(r"\s*Ok\(Arch::(\w+)\)\s*", tail)
                if not v:
                    raise serve.err(rest[0], f"a from_name arm this tool cannot read: {tail.strip()!r}")
                via[word] = v.group(1)
            else:
                raise serve.err(first[0], f"a matches! head this tool cannot read: {head!r}")
        else:
            alts = []
            for x, y in serve.split_top(a, b, "||"):
                t = re.fullmatch(r"\s*arch\s*==\s*(.*?)\s*", serve.m[x:y])
                if not t:
                    raise serve.err(x, f"a serves arm this tool cannot read: {serve.m[x:y].strip()!r}")
                alts += names_of(serve, x + serve.m[x:y].index("==") + 2, y, arch_consts)
            direct[word] = alts
    return direct, via


def src_from_name(arch):
    """Arch::from_name: {variant: [names]}."""
    lo, hi = arch.one_body("from_name")
    consts = arch.consts()
    out = {}
    for pat, (a, b) in arch.match_arms(lo, hi, r"\bmatch\s+name\b"):
        if re.fullmatch(r"\s*[a-z_]\w*\s*", arch.m[pat[0]:pat[1]]):
            continue
        v = re.fullmatch(r"\s*Ok\(Arch::(\w+)\)\s*", arch.m[a:b])
        if not v:
            raise arch.err(a, f"a from_name arm this tool cannot read: {arch.m[a:b].strip()!r}")
        out.setdefault(v.group(1), []).extend(names_of(arch, pat[0], pat[1], consts))
    return out


def src_spec(arch):
    """spec: the architecture names its arms read, with `is_qwen35_body` resolved through QWEN35_BODY."""
    lo, hi = arch.one_body("spec")
    consts = arch.consts()
    out = []
    for pat, _ in arch.match_arms(lo, hi, r"\bmatch\s+split\.architecture\(\)"):
        text = arch.m[pat[0]:pat[1]].strip()
        some = re.fullmatch(r"Some\((.*)\)", text, re.S)
        if re.fullmatch(r"[a-z_]\w*", text):
            continue
        if re.fullmatch(r"Some\(\w+\)\s+if\s+is_qwen35_body\(\w+\)", text):
            body = consts.get("QWEN35_BODY")
            if not isinstance(body, list):
                raise arch.err(pat[0], "QWEN35_BODY is not a list of strings")
            out.extend(body)
        elif some:
            open_at = arch.m.index("Some(", pat[0]) + len("Some")
            out.extend(names_of(arch, open_at + 1, arch.close(open_at), consts))
        else:
            raise arch.err(pat[0], f"a spec arm this tool cannot read: {text!r}")
    return out


def src_decide_rows(seat, releases):
    """The decide rows: [{name, backbones, in_file_arch, in_file_decision, head_repo, quant_repo, unserved}]."""
    hit = re.search(r"\bconst\s+ROWS\b[^=]*=\s*&\[", seat.m)
    if not hit:
        raise SourceError(f"{seat.path}: no `const ROWS: ... = &[`")
    lo = hit.end() - 1
    hi = seat.close(lo)
    rows = []
    for a, b in seat.split_top(lo + 1, hi, ","):
        body = seat.m[a:b]
        if not body.strip():
            continue
        if not re.match(r"\s*Row\s*\{", body):
            raise seat.err(a, f"a row this tool cannot read: {body.strip()[:40]!r}")
        o = a + body.index("{")
        c = seat.close(o)
        fields = {}
        for x, y in seat.split_top(o + 1, c, ","):
            f = re.match(r"\s*(\w+)\s*:", seat.m[x:y])
            if f:
                fields[f.group(1)] = (x + f.end(), y)
        rows.append(row_of(seat, fields, releases, a))
    if not rows:
        raise SourceError(f"{seat.path}: no rows in ROWS")
    return rows


def release_value(seat, lo, hi, releases):
    """The value of `clef::BACKBONES`-style paths and of `&[]` in seat.m[lo:hi]."""
    t = seat.m[lo:hi].strip()
    if t == "&[]":
        return []
    m = re.fullmatch(r"(\w+)::(\w+)", t)
    if not m or m.group(1) not in releases:
        raise seat.err(lo, f"a field value this tool cannot read: {t!r}")
    val = releases[m.group(1)].get(m.group(2))
    if val is None:
        raise seat.err(lo, f"{t} is not a string or list constant of {RELEASE_RS[m.group(1)]}")
    return val


def row_of(seat, fields, releases, at):
    for need in ("name", "backbones", "in_file", "head", "unserved"):
        if need not in fields:
            raise seat.err(at, f"a Row with no `{need}:` field")
    one = lambda k: release_value(seat, *fields[k], releases)
    in_arch, in_decision = [], []
    lo, hi = fields["in_file"]
    p = seat.m.index("[", lo)
    for x, y in seat.split_top(p + 1, seat.close(p), ","):
        t = seat.m[x:y].strip()
        if not t:
            continue
        m = re.fullmatch(r"InFile::(Arch|Decision)\((.*)\)", t, re.S)
        if not m:
            raise seat.err(x, f"an in_file item this tool cannot read: {t!r}")
        v = release_value(seat, x + seat.m[x:y].index("(") + 1, y - (len(seat.m[x:y]) - len(seat.m[x:y].rstrip())) - 1,
                          releases)
        (in_arch if m.group(1) == "Arch" else in_decision).append(v)
    head, quant = None, None
    h = seat.m[fields["head"][0]:fields["head"][1]].strip()
    if h != "None":
        if not h.startswith("Some(HeadRepo"):
            raise seat.err(fields["head"][0], f"a head field this tool cannot read: {h[:40]!r}")
        o = seat.m.index("{", fields["head"][0])
        sub = {}
        for x, y in seat.split_top(o + 1, seat.close(o), ","):
            f = re.match(r"\s*(\w+)\s*:", seat.m[x:y])
            if f:
                sub[f.group(1)] = (x + f.end(), y)
        head = release_value(seat, *sub["repo"], releases)
        quant = release_value(seat, *sub["quant_repo"], releases)
    uns = one("unserved")
    return {"name": one("name"), "backbones": tuple(one("backbones")), "in_file_arch": tuple(in_arch),
            "in_file_decision": tuple(in_decision), "head_repo": head, "quant_repo": quant,
            "unserved": tuple(uns)}


def src_types(quant):
    """GgmlType::from_u32: {id: variant name}."""
    lo, hi = quant.one_body("from_u32")
    out = {}
    for pat, (a, b) in quant.match_arms(lo, hi, r"\bmatch\s+v\b"):
        t = quant.m[pat[0]:pat[1]].strip()
        if not t.isdigit():
            continue
        v = re.fullmatch(r"GgmlType::(\w+)", quant.m[a:b].strip())
        if not v:
            raise quant.err(a, f"a from_u32 arm this tool cannot read: {quant.m[a:b].strip()!r}")
        out[int(t)] = v.group(1)
    return out


def src_site_names(site):
    """SiteTy::of_ggml: {SiteTy variant: ggml type name}."""
    lo, hi = site.one_body("of_ggml")
    out = {}
    for pat, (a, b) in site.match_arms(lo, hi, r"\bmatch\s+ty\b"):
        p = re.fullmatch(r"\s*GgmlType::(\w+)\s*", site.m[pat[0]:pat[1]])
        if not p:
            continue
        v = re.fullmatch(r"\s*Some\(SiteTy::(\w+)\)\s*", site.m[a:b])
        if not v:
            raise site.err(a, f"an of_ggml arm this tool cannot read: {site.m[a:b].strip()!r}")
        out[v.group(1)] = p.group(1)
    return out


def src_body35(body, site):
    """body35.rs: ({list name: [ggml type names]}, {list name: how many `ty_of`/`routed_ty` calls read it})."""
    names = src_site_names(site)
    lists = {}
    for hit in re.finditer(r"\bconst\s+(\w+)\s*:\s*&\[SiteTy\]\s*=\s*&\[", body.m):
        lo = hit.end() - 1
        items = re.findall(r"SiteTy::(\w+)", body.m[lo:body.close(lo)])
        for v in items:
            if v not in names:
                raise body.err(lo, f"SiteTy::{v} is no arm of {SITE_RS}:SiteTy::of_ggml")
        lists[hit.group(1)] = [names[v] for v in items]
    calls = {}
    for hit in re.finditer(r"(?<!fn )\b(?:ty_of|routed_ty)\(", body.m):
        open_at = hit.end() - 1
        args = [sp for sp in body.split_top(open_at + 1, body.close(open_at), ",") if body.m[sp[0]:sp[1]].strip()]
        last = body.m[args[-1][0]:args[-1][1]].strip()
        if re.fullmatch(r"[A-Z][A-Z_0-9]*", last):
            calls[last] = calls.get(last, 0) + 1
        elif not re.fullmatch(r"[a-z_]\w*", last):
            raise body.err(args[-1][0], f"a site call whose list this tool cannot read: {last!r}")
    return lists, calls


def src_bodies(seat):
    """The predicates of the decide seat's `BODIES`."""
    hit = re.search(r"\bconst\s+BODIES\b[^=]*=\s*&\[", seat.m)
    if not hit:
        raise SourceError(f"{seat.path}: no `const BODIES: ... = &[`")
    lo = hit.end() - 1
    return re.findall(r"\(\s*(\w+)\s*,", seat.m[lo + 1:seat.close(lo)])


def diff_sets(label, mirror, source, where):
    """Every name in one of the two and not the other, as messages."""
    out = [f"{label}: {n!r} is in {where} and not in the mirror" for n in sorted(set(source) - set(mirror))]
    out += [f"{label}: {n!r} is in the mirror and not in {where}" for n in sorted(set(mirror) - set(source))]
    return out


def diff_maps(label, mirror, source, where):
    """diff_sets over the keys, then over each key's value (a list of names)."""
    out = diff_sets(f"{label} keys", mirror, source, where)
    for k in sorted(set(mirror) & set(source)):
        mv, sv = mirror[k], source[k]
        if isinstance(mv, (list, tuple, set)):
            out += diff_sets(f"{label} {k}", mv, sv, where)
        elif mv != sv:
            out.append(f"{label} {k}: the mirror has {mv!r} and {where} has {sv!r}")
    return out


def mirror_vs_source(read):
    """Every difference between the mirror and the Rust sources `read(path)` returns, as messages; a SourceError
    where a source cannot be read."""
    serve, arch = Src(SERVE_RS, read(SERVE_RS)), Src(ARCH_RS, read(ARCH_RS))
    decide_serve = Src(DECIDE_SERVE_RS, read(DECIDE_SERVE_RS))
    seat = Src(DECIDE_SEAT_RS, read(DECIDE_SEAT_RS))
    quant = Src(QUANT_RS, read(QUANT_RS))
    releases = {k: Src(v, read(v)).consts() for k, v in RELEASE_RS.items()}
    words, order, decide_word = src_seat_words(serve, decide_serve.consts())
    out = []
    word_of = lambda v: words.get(v, "?" + v)
    out += diff_sets("seat order (Model::ALL)", SEAT_ORDER, [word_of(v) for v in order], SERVE_RS)
    if [word_of(v) for v in order] != list(SEAT_ORDER):
        out.append(f"seat order: the mirror has {list(SEAT_ORDER)} and {SERVE_RS} has {[word_of(v) for v in order]}")
    if decide_word != DECIDE_WORD:
        out.append(f"decide word: the mirror has {DECIDE_WORD!r} and {DECIDE_SERVE_RS} has {decide_word!r}")
    direct, via = src_serves(serve, arch.consts(), words)
    out += diff_maps(f"{SERVE_RS}:Model::serves", SERVES, direct, SERVE_RS)
    out += diff_maps(f"{SERVE_RS}:Model::serves via Arch::from_name", SERVES_VIA_FROM_NAME, via, SERVE_RS)
    out += diff_maps(f"{ARCH_RS}:Arch::from_name", FROM_NAME, src_from_name(arch), ARCH_RS)
    out += diff_sets(f"{ARCH_RS}:spec", SPEC, src_spec(arch), ARCH_RS)
    rows = {r["name"]: r for r in src_decide_rows(seat, releases)}
    out += diff_sets("decide rows", [r["name"] for r in DECIDE_ROWS], list(rows), DECIDE_SEAT_RS)
    for mine in DECIDE_ROWS:
        theirs = rows.get(mine["name"])
        if theirs is None:
            continue
        for k in ("backbones", "in_file_arch", "in_file_decision", "unserved"):
            out += diff_sets(f"decide row {mine['name']} {k}", mine[k], theirs[k], DECIDE_SEAT_RS)
        for k in ("head_repo", "quant_repo"):
            if mine[k] != theirs[k]:
                out.append(f"decide row {mine['name']} {k}: the mirror has {mine[k]!r} and {DECIDE_SEAT_RS} has "
                           f"{theirs[k]!r}")
    consts = arch.consts()
    out += diff_sets(f"{ARCH_RS}:QWEN35_BODY against BODY35_ARCHS", BODY35_ARCHS, consts.get("QWEN35_BODY", []), ARCH_RS)
    out += diff_sets(f"{DECIDE_SEAT_RS}:BODIES predicates", DECIDE_BODY_PREDICATES, src_bodies(seat), DECIDE_SEAT_RS)
    _, calls = src_body35(Src(BODY35_RS, read(BODY35_RS)), Src(SITE_RS, read(SITE_RS)))
    mine = {}
    for _, lst in BODY35_SITES:
        mine[lst] = mine.get(lst, 0) + 1
    mine["EMBED"], mine["HEAD_TY"] = 1, 1
    out += diff_maps(f"{BODY35_RS} sites per list", mine, calls, BODY35_RS)
    ids = src_types(quant)
    out += diff_maps(f"{QUANT_RS}:GgmlType::from_u32", {str(k): v for k, v in GGML_TYPES.items()},
                     {str(k): v for k, v in ids.items()}, QUANT_RS)
    out += [f"{QUANT_RS}:GgmlType::from_u32: the id {k} is named {ids[k]} there and is one of the mirror's "
            "unnamed ids (EXTRA_TYPES)" for k in sorted(set(EXTRA_TYPES) & set(ids))]
    return out


def read_source(path):
    try:
        with open(os.path.join(ROOT, path), encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        raise SourceError(f"{path}: {e.strerror}")


# ------------------------------------------------------------------------------------------------
# The self-test: offline, on the sources, on synthetic GGUF headers and on the hf crate's own examples.

def kv_string(key, value):
    k, v = key.encode(), value.encode()
    return struct.pack("<Q", len(k)) + k + struct.pack("<I", 8) + struct.pack("<Q", len(v)) + v


def kv_strings(key, count, width):
    """A string array of `count` strings of `width` bytes: a tokenizer's size."""
    k = key.encode()
    out = struct.pack("<Q", len(k)) + k + struct.pack("<I", 9) + struct.pack("<I", 8) + struct.pack("<Q", count)
    item = struct.pack("<Q", width) + b"t" * width
    return out + item * count


def kv_scalars(key, type_id, count):
    """A scalar array KV of `count` zeroed values of GGUF type `type_id` (5, int32): a tokenizer's token-type array."""
    k = key.encode()
    step = struct.calcsize(gguf_ranges().SCALAR[type_id])
    return (struct.pack("<Q", len(k)) + k + struct.pack("<I", 9) + struct.pack("<I", type_id) +
            struct.pack("<Q", count) + b"\0" * (count * step))


def synth(path, arch, tensors=(), extra=(), vocab=0):
    """A GGUF v3 file at `path`: general.architecture, `extra` ready-packed KVs, a token array of `vocab` strings
    and `tensors` ((name, type id, dims, bytes))."""
    kvs = [kv_string("general.architecture", arch)] + list(extra)
    if vocab:
        kvs.append(kv_strings("tokenizer.ggml.tokens", vocab, 24))
    gguf_ranges().write_gguf(path, kvs, list(tensors))
    with open(path, "rb") as f:
        return f.read()


class FakeHub:
    """check_row's `hub`: repos held in memory, each {path: bytes}, with the cards by repo."""

    def __init__(self, repos, cards=None):
        self.repos, self.cards = repos, cards or {}
        self.requests = self.bytes = 0
        self.card_asked = []

    def listing(self, repo):
        return [Entry(p, len(b), "sha256", hashlib.sha256(b).hexdigest()) for p, b in self.repos[repo].items()]

    def range(self, repo, path, start, end):
        data = self.repos[repo][path][start:end]
        if len(data) != end - start:
            raise Red(f"{repo}: the range {start}-{end - 1} of {path} is past its {len(self.repos[repo][path])} bytes")
        self.requests += 1
        self.bytes += len(data)
        return data

    def card(self, repo, entries):
        self.card_asked.append(repo)
        return self.cards.get(repo)


def self_test():
    ok = True

    def check(name, cond, detail=""):
        """`cond` is a value, or a function returning one that may raise: an error is a FAIL naming it."""
        nonlocal ok
        if callable(cond):
            try:
                cond = cond()
            except Exception as e:
                cond, detail = False, f"{type(e).__name__}: {e}"
        print(("ok " if cond else "FAIL ") + name + ("" if cond else f": {detail}"))
        ok = ok and bool(cond)

    def raises(fn, exc, needle):
        try:
            fn()
        except exc as e:
            return needle in str(e), str(e)
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"
        return False, "no refusal"

    # -- the mirror equals the Rust source
    try:
        diffs = mirror_vs_source(read_source)
        check("the mirror equals the router, reader and decide-row sources", not diffs, "; ".join(diffs))
    except SourceError as e:
        check("the mirror equals the router, reader and decide-row sources", False, f"a source cannot be parsed: {e}")

    def doctored(path, old, new):
        def read(p):
            text = read_source(p)
            if p == path:
                if old not in text:
                    raise SourceError(f"{p}: the self-test's doctoring text is gone: {old!r}")
                text = text.replace(old, new)
            return text
        return read

    for name, path, old, new, want in (
        ("a name added to Arch::from_name's glm arm is named", ARCH_RS,
         '"glm5next" | GLM5_NEXT_LLAMA_CPP =>', '"glm5next" | GLM5_NEXT_LLAMA_CPP | "glm5-nxt" =>', "glm5-nxt"),
        ("a name added to a Model::serves arm is named", SERVE_RS,
         'Model::Qwen38 => arch == "qwen4exp",', 'Model::Qwen38 => matches!(arch, "qwen4exp" | "qwen4expx"),',
         "qwen4expx"),
        ("a name added to spec's arms is named", ARCH_RS, 'Some("mimo2") =>', 'Some("mimo2" | "mimo3") =>', "mimo3"),
        ("a name added to a decide row's backbones is named", RELEASE_RS["lev"],
         'pub const BACKBONES: &[&str] = &["qwen35"];', 'pub const BACKBONES: &[&str] = &["qwen35", "qwen36"];',
         "qwen36"),
        ("a name dropped from the source is named", ARCH_RS, '"glm5next" | GLM5_NEXT_LLAMA_CPP =>', '"glm5next" =>',
         "glm5-next"),
        ("an id added to GgmlType::from_u32 is named", QUANT_RS, "            39 => GgmlType::MXFP4,\n",
         "            39 => GgmlType::MXFP4,\n            40 => GgmlType::NVFP4,\n", "'40'"),
        ("a new site read against a body35 list is named", BODY35_RS,
         "v_ty: ty_of(file, &v, kv, h, PROJ)?,", "v_ty: ty_of(file, &v, kv, h, PROJ)?, x_ty: ty_of(file, &v, kv, h, PROJ)?,",
         "sites per list PROJ"),
        ("a second backbone body of the decide seat is named", DECIDE_SEAT_RS,
         "&[(is_qwen35_body, open_qwen35)]", "&[(is_qwen35_body, open_qwen35), (is_other_body, open_other)]",
         "is_other_body"),
        ("a name added to QWEN35_BODY is named", ARCH_RS, 'pub const QWEN35_BODY: &[&str] = &["qwen35", "clef"];',
         'pub const QWEN35_BODY: &[&str] = &["qwen35", "clef", "qwen36"];', "qwen36"),
    ):
        try:
            got = mirror_vs_source(doctored(path, old, new))
            check(name, any(want in d for d in got), f"{got}")
        except SourceError as e:
            check(name, False, str(e))
    def followed():
        text = read_source(BODY35_RS)
        old = "const EMBED: &[SiteTy] = &["
        if old not in text:
            raise SourceError(f"{BODY35_RS}: the self-test's doctoring text is gone: {old!r}")
        doc = Src(BODY35_RS, text.replace(old, old + "SiteTy::Q3K,", 1))
        lists = src_body35(doc, Src(SITE_RS, read_source(SITE_RS)))[0]
        pair = lambda ty: [[("token_embd.weight", ty, [], 0), ("output.weight", 14, [], 0)]]
        return (body35_types(pair(11), lists)[0] is None and "Q3_K" in lists["EMBED"]
                and body35_types(pair(1), lists)[0] is not None)

    check("a type added to a body35 list is followed from the source, not mirrored", followed)
    try:
        mirror_vs_source(doctored(SERVE_RS, 'Model::Qwen3 => matches!(arch, "qwen3moe" | "qwen35moe"),',
                                  'Model::Qwen3 => arch.starts_with("qwen3"),'))
        check("a serves arm in a form this tool cannot read fails by file and line", False, "no failure")
    except SourceError as e:
        check("a serves arm in a form this tool cannot read fails by file and line",
              f"{SERVE_RS}:" in str(e) and "cannot read" in str(e), str(e))
    check("a missing source fails", raises(lambda: mirror_vs_source(lambda p: read_source(p + ".gone")),
                                           SourceError, "No such file")[0])
    try:
        places_ok = {seat: places(seat) for seat in TYPE_PLACES}
        check("every place the seats' types are decided in is found by its anchor",
              all(len(v) == len(TYPE_PLACES[s]) for s, v in places_ok.items()), str(places_ok))
    except Refuse as e:
        check("every place the seats' types are decided in is found by its anchor", False, str(e))

    # -- synthetic headers through check_row
    with tempfile.TemporaryDirectory() as tmp:
        def row_line(seat, arch, tensors=(), extra=(), vocab=0, quant="Q4_K_M", cards=None, files=None, repo="own/m-GGUF"):
            path = os.path.join(tmp, "m.gguf")
            blob = synth(path, arch, tensors, extra, vocab)
            fs = files if files is not None else {f"m-{quant}.gguf": blob}
            hub = FakeHub({repo: fs}, cards)
            line, green = check_row((seat, repo, quant, 0), hub, tmp)
            return line, green, hub

        small = [("token_embd.weight", 11, [256, 2], b"\0" * 8), ("blk.0.ffn_gate_exps.weight", 12, [256, 2, 2], b"\0" * 8),
                 ("blk.0.ffn_down_exps.weight", 14, [256, 2, 2], b"\0" * 8)]
        line, green, hub = row_line("glm", "glm5-next", small)
        check("a glm5-next file routes to the glm seat", green and "arch=glm5-next route=ok" in line, line)
        check("a tensor-type line parses", "embd=Q3_K exps=gate:Q4_K,down:Q6_K " in line, line)
        line, green, _ = row_line("glm", "glm5nextx", small)
        check("a glm5nextx file is refused by name",
              not green and "arch=glm5nextx route=REFUSED: a glm5nextx file, which no seat serves; the seats are "
              "--model ds41, qwen38, glm, qwen3, mimo2, decide" in line, line)
        line, green, _ = row_line("glm", "deepseek41", small)
        check("a file the router seats elsewhere is refused for the row's seat",
              not green and "in the ds41 seat, and the README row says glm" in line, line)
        line, green, _ = row_line("qwen38", "qwen4exp", [("token_embd.weight", 140, [256], b"\0" * 8)])
        check("a type id outside the table prints as type<id>", green and "embd=type140 exps=none " in line, line)
        line, green, _ = row_line("qwen3", "qwen3moe", [("blk.0.attn_q.weight", 0, [8, 2], b"\0" * 8)])
        check("no token_embd and no experts in the headers read are printed by name",
              "embd=not-found exps=none " in line, line)
        line, green, _ = row_line("ds41", "deepseek4", small)
        check("a deepseek4 file routes to the ds41 seat", green and "arch=deepseek4 route=ok" in line, line)
        line, green, _ = row_line("glm", "glm5next", small)
        check("a glm5next file routes to the glm seat", green, line)
        line, green, _ = row_line("qwen3", "deepseek2", small)
        check("an architecture no seat serves is refused by name, before the reader's verdict",
              not green and "a deepseek2 file, which no seat serves" in line, line)
        line, green, _ = row_line("mimo2", "mimo2", small)
        check("a mimo2 file routes to the mimo2 seat", green and "arch=mimo2 route=ok" in line, line)
        # a split set: the experts are in shard 2
        p1, p2 = os.path.join(tmp, "s1.gguf"), os.path.join(tmp, "s2.gguf")
        b1 = synth(p1, "glm5-next", [small[0]])
        b2 = synth(p2, "glm5-next", small[1:])
        line, green, hub = row_line("glm", "glm5-next", files={"UD-Q4_K_XL/m-UD-Q4_K_XL-00001-of-00002.gguf": b1,
                                                            "UD-Q4_K_XL/m-UD-Q4_K_XL-00002-of-00002.gguf": b2},
                                    quant="UD-Q4_K_XL")
        check("a split set's later shards are read for their expert types",
              green and "embd=Q3_K exps=gate:Q4_K,down:Q6_K " in line and "shard1=UD-Q4_K_XL/m-UD-Q4_K_XL-00001" in line,
              line)
        line, green, _ = row_line("glm", "glm5-next", files={"x.gguf": b""}, quant="Q9_9")
        check("a quant no set names is a red row naming the quants",
              not green and "no GGUF set carries the quant" in line and "shard1=none" in line, line)

        # type ids drawn from the body's own lists, so these rows do not pin what the lists hold
        lists = body35_lists()
        tid = {v: k for k, v in GGML_TYPES.items()}
        inside = lambda lst: tid[lists[lst][0]]
        outside = lambda lst: next(k for k, v in sorted(GGML_TYPES.items()) if v not in lists[lst])
        both = lambda a, b: next((tid[t] for t in lists[a] if t not in lists[b]), outside(b))
        dense = [("token_embd.weight", inside("EMBED"), [256, 2], b"\0" * 8),
                 ("output.weight", inside("HEAD_TY"), [256, 2], b"\0" * 8),
                 ("blk.0.attn_q.weight", inside("PROJ"), [256, 2], b"\0" * 8),
                 ("blk.0.ssm_beta.weight", inside("BETA_ALPHA"), [8, 2], b"\0" * 8),
                 ("blk.0.ffn_down.weight", inside("PROJ"), [32, 2], b"\0" * 8),
                 ("blk.0.attn_norm.weight", 0, [8], b"\0" * 8)]
        line, green, _ = row_line("decide", "clef", dense)
        check("a clef file whose sites read their lists is types=ok", green and "types=ok (5 site tensors" in line
              and "body35.rs:" in line, line)

        def swapped(name, ty):
            return [(n, ty if n == name else t, d, b) for n, t, d, b in dense]

        bad = outside("EMBED")
        line, green, _ = row_line("decide", "clef", swapped("token_embd.weight", bad))
        check("a token_embd outside EMBED is a red row naming the list", not green and
              f"types=REFUSED: token_embd.weight is {type_name(bad)}; crates/gpu/src/arch/qwen3moe/body35.rs:" in line
              and f"EMBED takes {', '.join(lists['EMBED'])}" in line, line)
        bad = outside("HEAD_TY")
        line, green, _ = row_line("decide", "qwen35", swapped("output.weight", bad))
        check("a head outside HEAD_TY is a red row naming the head list", not green and
              f"output.weight is {type_name(bad)}" in line and f"HEAD_TY takes {', '.join(lists['HEAD_TY'])}" in line, line)
        bad = outside("PROJ")
        line, green, _ = row_line("decide", "clef", swapped("blk.0.attn_q.weight", bad))
        check("a projection outside PROJ is a red row", not green and f"blk.0.attn_q.weight is {type_name(bad)}" in line, line)
        bad = outside("BETA_ALPHA")
        line, green, _ = row_line("decide", "clef", swapped("blk.0.ssm_beta.weight", bad))
        check("a beta outside BETA_ALPHA is a red row", not green and f"blk.0.ssm_beta.weight is {type_name(bad)}" in line,
              line)
        tied = both("EMBED", "HEAD_TY")
        line, green, _ = row_line("decide", "clef", [("token_embd.weight", tied, [256, 2], b"\0" * 8)] + dense[2:])
        check("a tied head is read against HEAD_TY as the token embedding", not green and
              f"token_embd.weight is {type_name(tied)}; crates/gpu/src/arch/qwen3moe/body35.rs:" in line
              and "HEAD_TY takes" in line, line)
        line, green, _ = row_line("qwen3", "qwen3moe", swapped("token_embd.weight", bad))
        check("a seat with no one owner list prints its types and the places",
              green and "types=printed (no one owner: crates/gpu/src/arch/qwen3moe/body.rs:" in line, line)

        # -- the header read: a short read asks for more bytes
        path = os.path.join(tmp, "big.gguf")
        big = synth(path, "glm5-next", small, extra=[kv_scalars("tokenizer.ggml.token_type", 5, 300000)], vocab=60000)
        check("the self-test's big header is past the first read", len(big) > FIRST_READ + MIB // 4, str(len(big)))
        local = os.path.join(tmp, "prefix.bin")
        with open(local, "wb") as f:
            f.write(big[:FIRST_READ])
        got = raises(lambda: parse_header(local, len(big)), NeedMore, "the header ends at byte")
        check("a truncated header asks for more bytes, also inside a scalar array longer than the prefix", got[0], got[1])
        reads = []

        def fetch(a, b):
            reads.append((a, b))
            return big[a:b]

        def sliced():
            meta, tensors, n = read_header(fetch, local, len(big), ROW_BUDGET)
            return (meta is not None and reads[0] == (0, FIRST_READ)
                    and reads[1] == (FIRST_READ, min(2 * FIRST_READ, len(big)))
                    and sum(b - a for a, b in reads) == n and len(tensors) == len(small)
                    and meta_str(meta, "general.architecture") == "glm5-next")

        check("the header read fetches only the next slice, ends when it parses and reads the architecture", sliced,
              f"{reads}")
        check("a header unfinished at the budget is (None, None, bytes read), not a failure",
              lambda: read_header(lambda a, b: big[a:b], local, len(big), FIRST_READ) == (None, None, FIRST_READ))
        bad = b"GGUX" + big[4:FIRST_READ]
        got = raises(lambda: read_header(lambda a, b: bad[a:b], local, len(big), ROW_BUDGET), Red, "is not a GGUF file")
        check("a header that is not a GGUF file is a refusal by name, not a request for more", got[0], got[1])
        strings = synth(os.path.join(tmp, "strings.gguf"), "glm5-next", small, vocab=60000)
        got = raises(lambda: read_header(lambda a, b: strings[a:b], local, FIRST_READ, ROW_BUDGET), Red, "runs past them")
        check("a file that ends inside its header is a refusal, not a request for more", got[0], got[1])

    # -- the hub's failures, through a stub curl on PATH: the network's are a Refuse(69), a row's own 404 a red line
    with tempfile.TemporaryDirectory() as tmp:
        stub = os.path.join(tmp, "curl")
        with open(stub, "w") as f:
            f.write('#!/bin/sh\necho "$STUB_ERR" >&2\nexit "$STUB_RC"\n')
        os.chmod(stub, 0o755)
        saved = dict(os.environ)

        def stubbed(rc, err, path=tmp):
            os.environ.update({"PATH": path, "STUB_RC": str(rc), "STUB_ERR": err})
            return Hub(tmp)

        try:
            for rc, err, why in ((7, "curl: (7) Failed to connect", "refused"), (6, "curl: (6) Could not resolve host", "no host"),
                                 (28, "curl: (28) Operation timed out", "timeout"),
                                 (22, "curl: (22) The requested URL returned error: 503", "busy hub"),
                                 (22, "curl: (22) The requested URL returned error: 429", "rate limit")):
                got = raises(lambda: stubbed(rc, err).listing("o/r"), Refuse, "the network or the hub is down")
                check(f"curl {why} is a named refusal that ends the run", got[0] and got[1].startswith("curl listing"), got[1])

            def status():
                try:
                    stubbed(7, "x").listing("o/r")
                except Refuse as e:
                    return e.code == 69
                return False

            check("a network refusal carries exit status 69", status)

            def not_found():
                line, green = check_row(("glm", "o/gone", "Q4_K_M", 0), stubbed(
                    22, "curl: (22) The requested URL returned error: 404"), tmp)
                return (not green and "shard1=none arch=unread route=REFUSED: curl listing" in line
                        and "is not there" in line)

            check("an HTTP 404 on a row's repo is a red row, not a network refusal", not_found)
            got = raises(lambda: stubbed(0, "", path=os.path.join(tmp, "empty")).listing("o/r"), Refuse, "curl is not on PATH")
            check("a machine with no curl is a named refusal", got[0], got[1])
        finally:
            os.environ.clear()
            os.environ.update(saved)

    # -- the decide row's paths
    def asked(arch, decision=None, card=None):
        calls = []

        def cb(r):
            calls.append(r)
            return card
        return route(arch, decision, "o/r", cb), calls

    (seat, why), calls = asked("clef")
    check("a clef-layout file seats the decide row and reads no card", seat == "decide" and why is None and not calls,
          f"{seat} {why} {calls}")
    (seat, why), calls = asked("qwen35", None, "Cloudflare/clef-flash")
    check("a qwen35 file whose card names the head repo seats the decide row", seat == "decide" and calls == ["o/r"],
          f"{seat} {why} {calls}")
    (seat, why), calls = asked("qwen35", None, None)
    check("a qwen35 file with no card naming a head repo is refused for its head",
          seat is None and "a qwen35 file with no head" in why, str(why))
    (seat, why), _ = asked("qwen35", "lev")
    check("a qwen35 file whose decision type is lev seats the decide row", seat == "decide" and why is None, str(why))
    (seat, why), _ = asked("qwen35", "kev")
    check("a decision type no row serves is refused by name", seat is None and "decision.type is kev" in why, str(why))
    (seat, why), calls = asked("glm5-next")
    check("a generative file reads no card", seat == "glm" and not calls, f"{seat} {calls}")
    (seat, why) = route("glm5-next", None, "o/r", lambda r: None, spec=tuple(a for a in SPEC if a != "glm5-next"))
    check("an architecture the router seats and the reader does not take is refused by name",
          seat is None and "the seat glm takes a glm5-next file, which model::arch::spec does not read" in why, str(why))
    (seat, why), _ = asked(None)
    check("a header with no architecture is refused by name", seat is None and "names no architecture" in why, str(why))

    # -- the README's Use table
    good = ("## Use\n\n| Seat | Serves | One command |\n|---|---|---|\n"
            "| `glm` | [GLM](https://x/y) | `bloomery-serve --hf unsloth/GLM-5.3-Flash-GGUF:UD-Q4_K_XL` |\n"
            "| `decide` | a, b | `bloomery-serve --hf bartowski/Cloudflare_clef-flash-GGUF:Q5_K_M` |\n\n## Numbers\n")
    check("a good Use table parses", use_rows(good) == [("glm", "unsloth/GLM-5.3-Flash-GGUF", "UD-Q4_K_XL", 5),
                                                         ("decide", "bartowski/Cloudflare_clef-flash-GGUF", "Q5_K_M", 6)],
          str(use_rows(good)))
    for name, text, needle in (
        ("a row with no --hf command is refused", good.replace("`bloomery-serve --hf unsloth/GLM-5.3-Flash-GGUF:UD-Q4_K_XL`",
                                                               "`bloomery-serve -m x.gguf`"), "holds no `bloomery-serve --hf"),
        ("a row whose --hf has no quant is refused", good.replace(":UD-Q4_K_XL`", "`"), "is not <owner>/<name>:<quant>"),
        ("a row whose --hf is malformed is refused", good.replace("unsloth/GLM", "unsloth/a/GLM"),
         "is not <owner>/<name>:<quant>"),
        ("a row with an unknown seat word is refused", good.replace("`glm`", "`glx`"), "the seat word 'glx'"),
        ("a README with no Use heading is refused", good.replace("## Use", "## Using"), "no `## Use` heading"),
        ("a table with no rows is refused", good.split("| `glm`")[0], "not a header, a separator and rows"),
    ):
        got = raises(lambda: use_rows(text), Refuse, needle)
        check(name, got[0], got[1])
    try:
        with open(os.path.join(ROOT, README), encoding="utf-8") as f:
            rows = use_rows(f.read())
        check("the README's own Use table parses into rows of known seats", len(rows) >= 1, str(rows))
    except (OSError, Refuse) as e:
        check("the README's own Use table parses into rows of known seats", False, str(e))

    # -- the hf crate's picking, on its own fixtures and examples (crates/hf/src/lib.rs tests)
    def fixture(name):
        with open(os.path.join(ROOT, "crates", "hf", "tests", "fixtures", name), encoding="utf-8") as f:
            return f.read()

    try:
        clef, split = "bartowski/Cloudflare_clef-flash-GGUF", "bartowski/Qwen_Qwen3-235B-A22B-Instruct-2507-GGUF"
        sets = gguf_sets(clef, parse_listing(clef, fixture("bartowski-clef-flash-gguf.json")))
        check("the clef listing is 22 sets and no mmproj or imatrix", len(sets) == 22 and all(
            "mmproj" not in s["name"] and "imatrix" not in s["name"] for s in sets), str(len(sets)))
        check("bf16 names the model and not mmproj-bf16", pick(clef, sets, "bf16")["name"] == "Cloudflare_clef-flash-bf16")
        wrong = []
        for q, name in (("Q3_K_S", "Cloudflare_clef-flash-Q3_K_S"), ("q5_k_m", "Cloudflare_clef-flash-Q5_K_M"),
                        ("Q6_K", "Cloudflare_clef-flash-Q6_K"), ("Q6_K_L", "Cloudflare_clef-flash-Q6_K_L"),
                        ("IQ4_XS", "Cloudflare_clef-flash-IQ4_XS")):
            s = pick(clef, sets, q)
            if s["name"] != name or len(s["files"]) != 1:
                wrong.append((q, s["name"]))
        check("a tag picks one set case-insensitively; Q6_K does not name Q6_K_L", not wrong, str(wrong))
        check("Q4_K_M does not name IQ4_K_M and Q6_K does not name Q6_K_L",
              not names("Q4_K_M", "x-IQ4_K_M") and not names("Q6_K", "x-Q6_K_L") and names("Q4_K_M", "d/x-Q4_K_M"))
        got = raises(lambda: pick(clef, sets, "Q7_K"), Red, "no GGUF set carries the quant")
        check("a quant no set names is refused with its candidates", got[0] and "Q3_K_S" in got[1], got[1])
        got = raises(lambda: pick(clef, sets, "K_M"), Red, "matches 3 sets")
        check("K_M names three sets and is refused", got[0], got[1])
        got = raises(lambda: pick(clef, sets, "Q4_K"), Red, "no GGUF set carries the quant")
        check("a bare Q4_K names no set: _M after it is not a separator", got[0], got[1])
        ssets = gguf_sets(split, parse_listing(split, fixture("bartowski-qwen3-235b-a22b-instruct-2507-gguf.json")))
        s = pick(split, ssets, "Q4_K_M")
        want = [f"Qwen_Qwen3-235B-A22B-Instruct-2507-Q4_K_M/Qwen_Qwen3-235B-A22B-Instruct-2507-Q4_K_M-{i:05d}-of-00004.gguf"
                for i in range(1, 5)]
        check("a split set is one set of every shard in order", s["count"] == 4 and s["tag"] == "Q4_K_M"
              and [f.path for f in s["files"]] == want and sum(f.size for f in s["files"]) ==
              39_856_336_448 + 39_847_230_560 * 2 + 23_096_107_744, str(s))
        q2 = pick(split, ssets, "Q2_K")
        check("Q2_K does not name Q2_K_L's shards", len(q2["files"]) == 3 and all("Q2_K_L" not in f.path for f in q2["files"]))
        broken = fixture("bartowski-qwen3-235b-a22b-instruct-2507-gguf.json").replace(
            "Q4_K_M-00003-of-00004.gguf", "Q4_K_M-00003-of-00004.gguf.bak")
        got = raises(lambda: gguf_sets(split, parse_listing(split, broken)), Red, "lacks shard(s) [3]")
        check("a split set missing a shard is refused", got[0], got[1])
    except (OSError, Red) as e:
        check("the hf crate's picking examples", False, str(e))
    check("a file whose name ends .gguf_file is no model, and mmproj, imatrix and draft files are none",
          not is_model("a-Q4.gguf_file") and not is_model("mmproj-x.gguf") and not is_model("x-imatrix.gguf")
          and not is_model("mtp-x.gguf") and is_model("x.gguf"))
    check("tag_of and split_of read llama.cpp's forms", tag_of("d/GLM-5.3-Flash-UD-Q4_K_XL") == "Q4_K_XL"
          and tag_of("noseparator") == "" and split_of("p-00001-of-00006") == ("p", 1, 6)
          and split_of("p-0001-of-00006") is None and split_of("-00001-of-00006") is None)
    bad_repos = ["", "noslash", "a/b/c", "/b", "a/", "a/b:", "a/b:c:d", "a/..", "a b/c", "a/b:Q4 K"]
    check("a repo string parses as the hf crate's RepoRef::parse does",
          parse_repo("bartowski/Cloudflare_clef-flash-GGUF:Q5_K_M") == ("bartowski/Cloudflare_clef-flash-GGUF", "Q5_K_M")
          and parse_repo("Cloudflare/clef-flash") == ("Cloudflare/clef-flash", None)
          and all(parse_repo(b) is None for b in bad_repos), str([b for b in bad_repos if parse_repo(b)]))
    check("a listing's Link header gives the next page", next_link(
        'content-type: x\r\nLink: <https://h/api?cursor=a>; rel="next", <https://h/api>; rel="first"\r\n')
          == "https://h/api?cursor=a" and next_link("content-type: x\r\n") is None)
    got = raises(lambda: parse_listing("o/r", '{"a": 1}'), Red, "the answer is not a list")
    check("a listing that is not a list is refused", got[0], got[1])
    card = ("---\nquantized_by: bartowski\nlicense: apache-2.0\nbase_model: Cloudflare/clef-flash\ntags:\n- clef\n"
            "- qwen3.5\nbase_model_relation: quantized\n---\n\n## Llamacpp\nbase_model: Other/x\n")
    cases = [
        (card, "Cloudflare/clef-flash"),
        ("---\nbase_model:\n  - 'Org/a'\nbase_model_relation: \"quantized\"\n---\n", "Org/a"),
        ("---\nbase_model: [Org/a]\nbase_model_relation: quantized\n---\n", "Org/a"),
        ("---\nbase_model: Org/a\nbase_model_relation: finetune\n---\n", None),
        ("---\nbase_model:\n- Org/a\n- Org/b\nbase_model_relation: quantized\n---\n", None),
        ("---\nbase_model: Org/a\n---\n", None),
        ("---\nlicense: mit\n---\nbase_model: Org/a\nbase_model_relation: quantized\n", None),
        ("base_model: Org/a\nbase_model_relation: quantized\n", None),
    ]
    check("a model card names the model its repo quantizes, as the hf crate's card.rs reads it",
          all(card_quantizes(t) == w for t, w in cases), str([(t, card_quantizes(t), w) for t, w in cases
                                                              if card_quantizes(t) != w]))
    check("a ggml type id has its name", type_name(11) == "Q3_K" and type_name(2) == "Q4_0" and type_name(140) == "type140")
    check("a listing's LFS sha256 is read, and a malformed one is refused", lambda: (
        parse_listing("o/r", json.dumps([{"type": "file", "path": "a.gguf", "size": 5, "oid": "a" * 40,
                                           "lfs": {"oid": "B" * 64, "size": 5}}])) == [Entry("a.gguf", 5, "sha256", "b" * 64)]
        and parse_listing("o/r", json.dumps([{"type": "file", "path": "r.md", "size": 5, "oid": "c" * 40}]))
        == [Entry("r.md", 5, "git-sha1", "c" * 40)]
        and raises(lambda: parse_listing("o/r", json.dumps([{"type": "file", "path": "a", "size": 5,
                                                              "lfs": {"oid": "xyz", "size": 5}}])), Red,
                   "its LFS oid is not a sha256")[0]))
    box_checks(check, raises)
    return ok


def box_checks(check, raises):
    """The self-test of the box side: the table, the facts, the hasher, the cache and the compare, offline."""
    problems = box_table_problems()
    check("every box path of the gates is where its source says", not problems, "; ".join(problems))
    moved = box_table_problems(lambda src: read_source(src).replace(BOX_FILES["glm"][0][0], "/models/elsewhere.gguf"))
    check("a box path that moved out of its source is named", any("glm" in m and BOX_FILES["glm"][0][0] in m for m in moved),
          str(moved))
    gpath = BOX_FILES["glm"][0][0]
    gex = BOX_EXCEPTIONS[gpath]
    gfile, gstr = gex.expiry
    problems = exception_problems()
    check("every exception is a BOX_FILES path of its seat, cites a source that holds its text, and expires in that tree",
          not problems, "; ".join(problems))
    cited_source, cited_text = gex.cites[0]
    moved = exception_problems(lambda src: read_source(src).replace(cited_text, "/elsewhere/ik"))
    check("an exception's cited text that moved out of its source is named", any(
        "glm" in m and cited_source in m and cited_text in m for m in moved), str(moved))
    saved_files, saved_ex = BOX_FILES["glm"], dict(BOX_EXCEPTIONS)
    BOX_FILES["glm"] = (("/models/elsewhere.gguf", "justfile"),)
    try:
        moved = exception_problems()
    finally:
        BOX_FILES["glm"] = saved_files
    check("an exception whose path is no BOX_FILES path of its seat is named", any(gpath in m and "no BOX_FILES path" in m
                                                                                for m in moved), str(moved))
    BOX_EXCEPTIONS[gpath] = gex._replace(expiry=("/elsewhere/llama-arch.cpp", gstr))
    try:
        moved = exception_problems()
    finally:
        BOX_EXCEPTIONS.clear()
        BOX_EXCEPTIONS.update(saved_ex)
    check("an expiry file outside the cited tree is named", any("/elsewhere/llama-arch.cpp is not under" in m for m in moved),
          str(moved))
    with tempfile.TemporaryDirectory() as tmp:
        data = os.urandom(3 * HASH_CHUNK // 8 * 5 + 777)
        f1, f2 = os.path.join(tmp, "m.gguf"), os.path.join(tmp, "n.gguf")
        for path in (f1, f2):
            with open(path, "wb") as f:
                f.write(data)
        with open(f1 + ".verified", "w") as f:
            f.write(hashlib.sha256(data).hexdigest() + "\n")
        import io
        out = io.StringIO()
        cmd_box_facts([f1, f2, os.path.join(tmp, "gone.gguf")], out)
        facts = parse_box_facts(out.getvalue(), [f1, f2, os.path.join(tmp, "gone.gguf")])
        st = os.stat(f1)
        check("box-facts prints size, mtime and marker, and a path that is not there is absent", facts == {
            f1: {"size": len(data), "mtime_ns": st.st_mtime_ns, "marker": hashlib.sha256(data).hexdigest(), "sha256": None},
            f2: {"size": len(data), "mtime_ns": os.stat(f2).st_mtime_ns, "marker": None, "sha256": None},
            os.path.join(tmp, "gone.gguf"): None}, str(facts))
        check("box-facts without --hash hashes nothing", "box-fact-end\t3" in out.getvalue() and "\t-\n" in out.getvalue())
        got = raises(lambda: parse_box_facts("box-fact\tx\tABSENT\n", ["x"]), Refuse, "answer is incomplete")
        check("a box answer with no end line is refused", got[0], got[1])
        got = raises(lambda: parse_box_facts("box-fact\tx\t12\n", ["x"]), Refuse, "cannot read")
        check("a box answer with a line this tool cannot read is refused", got[0], got[1])
        got = raises(lambda: cmd_box_facts(["--bogus"], io.StringIO()), Refuse, "usage")
        check("box-facts with a flag it does not take is a usage error", got[0], got[1])

        # the hasher: the loop over fake reads, page-aligned buffers, the refusals
        import ctypes
        chunk, seen = 2 * PAGE, []

        def reader_of(blob, short=None):
            def read_at(offset, view):
                seen.append(ctypes.addressof(ctypes.c_char.from_buffer(view)) % PAGE)
                n = min(len(view), len(blob) - offset) if short is None else min(short, len(blob) - offset)
                view[:n] = blob[offset:offset + n]
                return n
            return read_at

        sizes = (0, 1, chunk - 1, chunk, chunk + 1, 5 * chunk + 777)
        check("the hasher equals sha256 over sizes around the chunk", all(
            hash_chunks(reader_of(data[:n]), n, chunk, "t") == hashlib.sha256(data[:n]).hexdigest() for n in sizes))
        check("the hasher reads into page-aligned buffers", seen and all(a == 0 for a in seen), str(set(seen)))
        got = raises(lambda: hash_chunks(reader_of(data[:5 * chunk]), 6 * chunk, chunk, "t"), Refuse, "the file ends at byte")
        check("a file that ends before its size is refused by name", got[0], got[1])
        got = raises(lambda: hash_chunks(reader_of(data[:5 * chunk], short=PAGE + 1), 5 * chunk, chunk, "t"), Refuse,
                     "off a page")
        check("a short read that leaves the next read off a page is refused", got[0], got[1])
        flags = []

        def refusing(path, fl):
            flags.append(fl)
            raise OSError(22, "Invalid argument")

        got = raises(lambda: odirect_sha256(f1, len(data), o_direct=0x4000, opener=refusing), Refuse,
                     "O_DIRECT is refused (Invalid argument); there is no buffered fallback")
        check("an O_DIRECT refusal is named and has no buffered fallback", got[0] and flags and flags[0] & 0x4000, got[1])
        got = raises(lambda: odirect_sha256(f1, len(data), o_direct=None), Refuse, "no O_DIRECT")
        check("a platform with no O_DIRECT is refused, not read through the cache", got[0], got[1])
        if getattr(os, "O_DIRECT", None) is not None:
            try:
                direct = odirect_sha256(f1, len(data))
            except Refuse as e:  # a file system that refuses O_DIRECT (tmpfs): the named refusal
                direct = str(e)
            check("the O_DIRECT hash equals sha256, or the file system's refusal is named",
                  direct == hashlib.sha256(data).hexdigest() or "O_DIRECT is refused" in direct, direct)

        # the cache
        cp = os.path.join(tmp, "cache.tsv")
        cache_store(cp, {("/b/x", 5, 7): "a" * 64})
        cache_store(cp, {("/b/y", 6, 8): "b" * 64})
        check("the digest cache keeps what was stored and misses on another mtime", lambda: (
            cache_load(cp) == {("/b/x", 5, 7): "a" * 64, ("/b/y", 6, 8): "b" * 64} and ("/b/x", 5, 9) not in cache_load(cp)
            and cache_load(os.path.join(tmp, "none.tsv")) == {}))

        # the compare, per file
        d = "d" * 64
        ent = Entry("m-00001.gguf", 100, "sha256", d)
        fact = lambda size=100, marker=None, sha=None: {"size": size, "mtime_ns": 5, "marker": marker, "sha256": sha}
        cases = (
            ("a size that differs is a FAIL naming both sizes", compare_file(ent, "/b/m", fact(size=99), None, None),
             ("-", "FAIL: size box=99 live=100")),
            ("an absent box file is a FAIL by name", compare_file(ent, "/b/m", None, None, None), ("-", "FAIL: absent on the box")),
            ("the marker's digest is the box's digest", compare_file(ent, "/b/m", fact(marker=d), "e" * 64, None), ("marker", "ok")),
            ("a marker digest that differs is a FAIL naming both", compare_file(ent, "/b/m", fact(marker="e" * 64), None, None),
             ("marker", f"FAIL: digest box={'e' * 64} live={d}")),
            ("a malformed marker is a FAIL", compare_file(ent, "/b/m", fact(marker="zz"), None, None),
             ("marker", "FAIL: the .verified marker 'zz' is not a 64-digit digest")),
            ("a cached digest is used when there is no marker", compare_file(ent, "/b/m", fact(), d, None), ("cache", "ok")),
            ("a cached digest that differs is a FAIL", compare_file(ent, "/b/m", fact(), "e" * 64, None),
             ("cache", f"FAIL: digest box={'e' * 64} live={d}")),
            ("no marker and no cache is unhashed", compare_file(ent, "/b/m", fact(), None, None), ("unhashed", "unhashed")),
            ("a digest a hash run read is compared", compare_file(ent, "/b/m", fact(), None, d), ("hash", "ok")),
            ("a hash run's digest that differs is a FAIL", compare_file(ent, "/b/m", fact(sha="e" * 64), None, None),
             ("hash", f"FAIL: digest box={'e' * 64} live={d}")),
            ("a stat error is a FAIL", compare_file(ent, "/b/m", "cannot stat: x", None, None), ("-", "FAIL: cannot stat: x")),
        )
        for name, got, want in cases:
            check(name, got == want, f"{got} != {want}")

        # the named exception, per file, at the real table's sizes
        def verdict_of(box_size, state, live_size=9429984, path=gpath):
            live_ent = Entry(path.rsplit("/", 1)[-1], live_size, "sha256", d)
            return compare_file(live_ent, path, fact(size=box_size), None, None, {(gfile, gstr): state})[1]

        check("the exception applies at the two exact sizes with its string absent from the expiry file",
              verdict_of(9429859, "absent") == "excepted: " + gex.reason, verdict_of(9429859, "absent"))
        check("a box size one byte off is the size FAIL", verdict_of(9429858, "absent") == "FAIL: size box=9429858 live=9429984",
              verdict_of(9429858, "absent"))
        check("a live size one byte off is the size FAIL", verdict_of(9429859, "absent", live_size=9429985)
              == "FAIL: size box=9429859 live=9429985", verdict_of(9429859, "absent", live_size=9429985))
        check("the string in the expiry file is the expired FAIL naming the file and what to refresh", verdict_of(
            9429859, "found") == f"FAIL: {gex.name} expired: {gfile} now names glm5-next; refresh the copy and the GLM "
            "reference sets", verdict_of(9429859, "found"))
        check("an expiry file that is not on the box is a FAIL, not an exception", verdict_of(9429859, "missing")
              == f"FAIL: {gfile} is not on the box: the exception cannot be judged", verdict_of(9429859, "missing"))
        check("the same two sizes at another path keep the size FAIL", verdict_of(
            9429859, "absent", path="/models/other.gguf") == "FAIL: size box=9429859 live=9429984")
        check("a box file at the live size is no exception: ok by its marker, the expiry file read or not", compare_file(
            Entry("m", 9429984, "sha256", d), gpath, fact(size=9429984, marker=d), None, None, None) == ("marker", "ok"))
        got = raises(lambda: compare_file(Entry("m", 9429984, "sha256", d), gpath, fact(size=9429859), None, None, None),
                     Refuse, "was not asked whether")
        check("an exception judged with no answer for its expiry is a refusal, not the size FAIL", got[0], got[1])

        # box-facts --grep: found, absent, missing; a read error is a refusal, a pair not asked is refused
        needle_file = os.path.join(tmp, "arch.cpp")
        with open(needle_file, "w") as f:
            f.write('{ LLM_ARCH_GLM5NEXT, "glm5next" },\n')
        gone = os.path.join(tmp, "none.cpp")
        asks = [(needle_file, "glm5next"), (needle_file, "glm5-next"), (gone, "glm5-next")]
        out = io.StringIO()
        args = [f1] + [a for pair in asks for a in ("--grep",) + pair]
        cmd_box_facts(args, out)
        facts, states = parse_box_answer(out.getvalue(), [f1], asks)
        check("box-facts --grep says found, absent and missing, beside the facts of its paths", states == {
            asks[0]: "found", asks[1]: "absent", asks[2]: "missing"} and f1 in facts and out.getvalue().rstrip().endswith(
            "box-fact-end\t1"), str(states))
        got = raises(lambda: cmd_box_facts([f1, "--grep", tmp, "x"], io.StringIO()), Refuse, "cannot be read")
        check("a --grep file that cannot be read is a refusal, not missing", got[0], got[1])
        got = raises(lambda: cmd_box_facts([f1, "--grep", needle_file], io.StringIO()), Refuse, "usage")
        check("--grep with one value is a usage error", got[0], got[1])
        got = raises(lambda: parse_box_answer("box-grep\tf\ts\tfound\nbox-fact-end\t0\n", [], []), Refuse, "cannot read")
        check("a box answer for a grep that was not asked is refused", got[0], got[1])
        got = raises(lambda: parse_box_answer("box-grep\tf\ts\tmaybe\nbox-fact-end\t0\n", [], [("f", "s")]), Refuse,
                     "cannot read")
        check("a box answer with a grep state outside found, absent, missing is refused", got[0], got[1])
        got = raises(lambda: parse_box_answer("box-fact-end\t0\n", [], [("f", "s")]), Refuse, "answer is incomplete")
        check("a box answer that lacks an asked grep is incomplete", got[0], got[1])
        check("a hash run's wall is its bytes over the slower of the drive and one core",
              abs(hash_wall(10 * min(NVME_GBPS, SHA256_GBPS) * 1e9) - 10) < 1e-9)
        check("the box paths pair the hub's shard names with each box path's directory", lambda: box_paths("decide", [
            Entry("Cloudflare_clef-flash-Q5_K_M.gguf", 1, "sha256", d)]) == [
            ("/models/clef-flash/Cloudflare_clef-flash-Q5_K_M.gguf", [(Entry("Cloudflare_clef-flash-Q5_K_M.gguf", 1, "sha256", d),
             "/models/clef-flash/Cloudflare_clef-flash-Q5_K_M.gguf")]),
            ("/root/models/clef-flash/Cloudflare_clef-flash-Q5_K_M.gguf", [(Entry("Cloudflare_clef-flash-Q5_K_M.gguf", 1,
             "sha256", d), "/root/models/clef-flash/Cloudflare_clef-flash-Q5_K_M.gguf")])])
        check("a box file named other than the hub's first file is a refusal for the set", lambda: box_paths("decide", [
            Entry("Cloudflare_clef-flash-Q5_K_M-v2.gguf", 1, "sha256", d)])[0][1].startswith("the box names"))

        # the whole flow: two rows, a fake hub and a fake box
        body = body35_lists()
        sites = [("token_embd.weight", next(k for k, v in GGML_TYPES.items() if v == body["EMBED"][0]), [256, 2], b"\0" * 8),
                 ("output.weight", next(k for k, v in GGML_TYPES.items() if v == body["HEAD_TY"][0]), [256, 2], b"\0" * 8)]

        def blob(name, arch):
            return synth(os.path.join(tmp, name), arch, sites)

        glm_files = {f"UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-{i:05d}-of-00006.gguf": blob("g.gguf", "glm5-next")
                     for i in range(1, 7)}
        clef_file = {"Cloudflare_clef-flash-Q5_K_M.gguf": blob("c.gguf", "clef")}
        hub = lambda scratch: FakeHub({"unsloth/GLM-5.3-Flash-GGUF": glm_files, "bartowski/Cloudflare_clef-flash-GGUF": clef_file})
        readme = ("## Use\n\n| Seat | Serves | One command |\n|---|---|---|\n"
                  "| `glm` | x | `bloomery-serve --hf unsloth/GLM-5.3-Flash-GGUF:UD-Q4_K_XL` |\n"
                  "| `decide` | y | `bloomery-serve --hf bartowski/Cloudflare_clef-flash-GGUF:Q5_K_M` |\n")
        box = {}
        for name, b in glm_files.items():
            box[f"/models/GLM-5.3-Flash-UD-Q4_K_XL/{name.rsplit('/', 1)[-1]}"] = (len(b), hashlib.sha256(b).hexdigest())
        for root in ("/models", "/root/models"):
            b = clef_file["Cloudflare_clef-flash-Q5_K_M.gguf"]
            box[f"{root}/clef-flash/Cloudflare_clef-flash-Q5_K_M.gguf"] = (len(b), hashlib.sha256(b).hexdigest())

        def runner_of(table, hashed=None, asked=None, greps=None):
            """The box's `box-facts` over `table`; each --grep is answered by `greps` ({(file, string): state}), else absent."""
            def run(argv):
                if asked is not None:
                    asked.append(argv)
                lines, paths, asks, i = [], [], [], 0
                while i < len(argv):
                    if argv[i] == "--grep":
                        asks.append((argv[i + 1], argv[i + 2]))
                        i += 3
                    else:
                        paths += [] if argv[i] == "--hash" else [argv[i]]
                        i += 1
                for path in paths:
                    if path not in table:
                        lines.append(f"box-fact\t{path}\tABSENT")
                        continue
                    size, digest = table[path]
                    sha = hashed.get(path, "-") if "--hash" in argv and hashed else "-"
                    marker = digest if hashed is None else "-"
                    lines.append(f"box-fact\t{path}\t{size}\t77\t{marker}\t{sha}")
                lines += [f"box-grep\t{f}\t{s}\t{(greps or {}).get((f, s), 'absent')}" for f, s in asks]
                return "\n".join(lines + [f"box-fact-end\t{len(paths)}"]) + "\n"
            return run

        def run_live(table, cache="flow.tsv", **kw):
            import contextlib
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                rc = live(text=readme, box=True, hub_factory=hub, cache_path=os.path.join(tmp, cache), **kw)
            return rc, out.getvalue()

        saved = BOX_FILES["glm"]
        BOX_FILES["glm"] = (("/models/elsewhere.gguf", "justfile"),)
        try:
            got = raises(lambda: run_live(box, runner=runner_of(box)), Refuse, "the box paths of the gates have moved")
        finally:
            BOX_FILES["glm"] = saved
        check("--live --box refuses by name a box path that moved out of its source", got[0], got[1])
        rc, text = run_live(box, runner=runner_of(box))
        check("a box that equals the hub is green: every file ok by its marker", rc == 0 and "hf-arch-box files=8 ok=8 fail=0 unhashed=0" in text
              and "hf-arch rows=2 ok=2 red=0" in text and text.count("digest=marker verdict=ok") == 8, text)
        short = dict(box)
        k = "/models/GLM-5.3-Flash-UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf"
        short[k] = (short[k][0] - 125, short[k][1])
        gone = "/root/models/clef-flash/Cloudflare_clef-flash-Q5_K_M.gguf"
        del short[gone]
        rc, text = run_live(short, runner=runner_of(short))
        check("a size that differs and an absent file are red rows naming both values", rc == 1 and (
            f"file={k} box={short[k][0]}B live={box[k][0]}B digest=- verdict=FAIL: size box={short[k][0]} live={box[k][0]}") in text
              and f"file={gone} box=-B live=" in text and "verdict=FAIL: absent on the box" in text
              and "hf-arch rows=2 ok=0 red=2" in text, text)
        nomark = dict(box)
        rc, text = run_live(nomark, runner=runner_of(nomark, hashed={}))
        check("a box with no digest is unhashed, and the run prints the hash command and its estimate", rc == 0
              and "digest=unhashed verdict=unhashed" in text and "hf-arch-box unhashed: 8 files" in text
              and "box-facts --hash /models/GLM-5.3-Flash-UD-Q4_K_XL/" in text and "wall ~" in text, text)
        asked = []
        hashes = {p: v[1] for p, v in box.items()}
        rc, text = run_live(box, runner=runner_of(box, hashed=hashes, asked=asked), box_hash=True)
        check("--box-hash hashes the misses after printing the estimate, and compares", lambda: rc == 0 and text.index(
            "hf-arch-box hash:") < text.index("verdict=ok") and text.count("digest=hash verdict=ok") == 8
              and any("--hash" in a for a in asked), text)
        rc, text = run_live(box, runner=runner_of(box, hashed={}))
        check("the digests a hash run read are in the cache the next run uses", rc == 0 and text.count(
            "digest=cache verdict=ok") == 8 and "unhashed:" not in text, text)
        stale = dict(hashes)
        stale["/models/GLM-5.3-Flash-UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00002-of-00006.gguf"] = "f" * 64
        rc, text = run_live(box, cache="stale.tsv", runner=runner_of(box, hashed=stale), box_hash=True)
        check("a digest that differs after a hash run is a red row naming both", rc == 1 and "verdict=FAIL: digest box=" + "f" * 64 in text, text)

        # the named exception in the whole flow: the real entry's sizes patched to the fake shard's (the same 125 bytes
        # short), restored in the finally
        n = box[k][0]
        kept = dict(BOX_EXCEPTIONS)
        BOX_EXCEPTIONS[k] = kept[k]._replace(box_size=n - 125, live_size=n)
        held = dict(box)
        held[k] = (n - 125, box[k][1])
        grep_key = (gfile, gstr)

        def try_live(greps, asked=None, **kw):
            try:
                return run_live(held, cache="exc.tsv", runner=runner_of(held, greps=greps, asked=asked), **kw)
            except Refuse as e:
                return None, f"Refuse: {e}"

        try:
            asked = []
            rc, text = try_live(None, asked)
            check("--live --box asks the expiry in its one box call, and an excepted file leaves its row green and counted",
                  rc == 0 and len(asked) == 1 and asked[0][-3:] == ["--grep", gfile, gstr] and f"file={k} box={n - 125}B live={n}B "
                  f"digest=- verdict=excepted: {gex.reason}" in text and "hf-arch-box files=8 ok=7 fail=0 unhashed=0 excepted=1" in text
                  and "hf-arch rows=2 ok=2 red=0" in text, f"{asked} {text}")
            rc, text = try_live({grep_key: "found"})
            check("an expiry file that holds its string is a red row naming the exception", rc == 1 and (
                f"verdict=FAIL: {gex.name} expired: {gfile} now names glm5-next; ") in text and "fail=1 unhashed=0 excepted=0" in text
                  and "hf-arch rows=2 ok=1 red=1" in text, text)
            rc, text = try_live({grep_key: "missing"})
            check("an expiry file that is not on the box is a red row", rc == 1 and (
                f"verdict=FAIL: {gfile} is not on the box: the exception cannot be judged") in text, text)
            BOX_EXCEPTIONS[k] = BOX_EXCEPTIONS[k]._replace(cites=(("tools/ref/models/glm5next.sh", "/elsewhere/ik"),))
            rc, text = try_live(None)
            check("--live --box refuses by name an exception whose cited text moved out of its source", rc is None and (
                "the box paths of the gates have moved" in text and "/elsewhere/ik" in text), text)
        finally:
            BOX_EXCEPTIONS.clear()
            BOX_EXCEPTIONS.update(kept)


def main(argv):
    try:
        if argv[:1] == ["box-facts"]:
            cmd_box_facts(argv[1:])
            return 0
        if argv[:1] == ["--live"] and set(argv[1:]) <= {"--box", "--box-hash"} and (
                "--box-hash" not in argv or "--box" in argv):
            return live(box="--box" in argv, box_hash="--box-hash" in argv)
        if argv == ["--self-test"]:
            try:
                return 0 if self_test() else 1
            except Exception as e:
                at = traceback.extract_tb(e.__traceback__)[-1]
                print(f"FAIL the self-test stopped on {type(e).__name__}: {e} (line {at.lineno}, in {at.name})")
                return 1
        raise Refuse(64, "usage: hf-arch-check.py --live [--box [--box-hash]] | --self-test | box-facts [--hash] [--grep FILE STRING]... <paths...>")
    except Refuse as e:
        print(f"hf-arch-check.py: {e}", file=sys.stderr)
        return e.code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
