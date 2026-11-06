#!/usr/bin/env python3
"""Tensors out of a GGUF set and tensors into one, the keys and the tensor bytes carried as they are.

    tools/ref/gguf-splice.py strip --in A --out B --drop GLOB [--drop GLOB ...] [options]
    tools/ref/gguf-splice.py add   --in A --from C --take GLOB [--take GLOB ...] --out D [--arch NEW] [--kv KEY=TYPE:VALUE ...] [options]
    tools/ref/gguf-splice.py list  --in A [--tsv] [--gguf-py MODE]
    tools/ref/gguf-splice.py check-shape --in A --names TSV [--types]
    tools/ref/gguf-splice.py check-rules --rules FILE --table TSV [--in A] [--partial] [--retype Q6_K=Q4_K] [--flag-types NAME=type,...]
    tools/ref/gguf-splice.py compare --a A --b B [--no-data] [--tensors-only]
    tools/ref/gguf-splice.py plan --log DRYRUN.log --out TSV
    tools/ref/gguf-splice.py classes (--tsv TSV | --in A) [--md] [--field CLASS]
    tools/ref/gguf-splice.py fill --template T --out O [--set KEY=VALUE ...] [--file KEY=PATH ...]
    tools/ref/gguf-splice.py --self-test

Built for the Kimi K3.1 pipeline (docs/cards/k31-*.card): the K3.1 checkpoint's DSA layers carry an indexer that mainline llama.cpp's
kimi-k3 graph does not know, and its loader refuses a file with tensors the architecture does not create. `strip` writes the K3-shaped
file (`--drop 'blk.*.indexer.*'`), the imatrix and the quantizer run on it, and `add` puts the indexer tensors back into the quantized
set (`--take 'blk.*.indexer.*' --arch kimi-k31`). The tool is generic: a GGUF file or a split set in, a GGUF file or a split set out.

A set is named by its first shard (`-00001-of-000NN.gguf`; a file without `split.count`, or with 1, is one file under any name). The
shard count stays the input's and every tensor stays in its shard (the tensor-to-shard map is kept; added tensors go to the last shard,
`--into N` picks another). Tensors keep their order and data, so a tensor's bytes are the source's, copied in 32 MiB chunks: nothing
but one chunk is ever in memory, and the input is read once, hashed as it is read. Every metadata value is copied as its raw bytes, in
its order. What changes, and only that:
  - `split.tensors.count` (the total mainline's loader checks against the tensors it found) in every shard that is written;
  - with `--arch NEW`: `general.architecture` and every `OLD.` key of every shard that has them take the new spelling (OLD is what the
    file says; a file whose shards disagree, or that has none, is refused);
  - with `--kv KEY=TYPE:VALUE` (add only; the key in its final spelling, after `--arch`): a key appended to the first shard. TYPE is
    u8 i8 u16 i16 u32 i32 u64 i64 f32 f64 bool or str, or one of the numeric ones with `[]` for an array (`u32[]:3,7,11`). A key the
    file already has with another value is refused unless `--kv-replace`; with the same value it is a no-op.
Every tensor is dropped, taken or kept whole; a pattern (fnmatch on the tensor name, `*` matches dots too) that matches nothing is
refused by name, so is a taken tensor whose name the base set already has, a take of nothing, and a strip of every tensor.

Options:
  --link       a shard whose tensors and keys do not change is hard-linked from the input instead of copied. The first shard is
               written whenever the tensor total moves (it carries `split.tensors.count`). A linked shard keeps its own
               `split.tensors.count`: mainline reads only the first shard's, and the tool does not rewrite a file it links.
  --consume    strip only: the input is consumed to fit a disk that cannot hold two sets. Shards are written one at a time, the first
               shard last, each is verified, renamed into place, and then its input shard is deleted (an unchanged shard is renamed,
               never copied). A killed run is resumed by the same command: a shard whose input is gone and whose output is there is
               read from the output and skipped; an output whose input is still there (the kill fell between rename and delete) is
               held to its planned header, and its input deleted; any other existing output is refused (73). The finished outputs of
               a failed run are kept. An add does not need it: --link rewrites only the receiving shard and the first.
  --verify M   data (default): the written shard is read back, every tensor's sha256 compared with the input's, and the header
               compared with the one the tool meant to write; header: the header comparison alone.
  --gguf-py M  auto (default), required, off: read every written shard back through gguf-py's GGUFReader and compare tensor names,
               shapes, types, sizes and offsets and the metadata keys with this tool's own reading. gguf-py is the oracle tree's
               ($LCPP/gguf-py, default /home/user/llama.cpp-kimik3; read only: no bytecode is written there); auto skips it,
               naming that it did, where the directory or numpy is missing; required refuses (66) then.
  --no-space-check  skip the free-space refusal (the tool refuses, 74, when the output filesystem has less than the shards it
               writes, or than the largest one with --consume).

`list` prints each shard's tensor count, data start and size, then a table of the tensor types (count and bytes), or with --tsv one
line `name TAB type TAB dims-joined-by-x TAB shard` per tensor, the form of crates/model/tests/kimi_k3_tensors.tsv. `check-shape`
holds the set's tensor names and dims to such a table (with --types, the types too; without, a stripped K3.1 file is MXFP4 and BF16
where the K3 file is IQ1_S and Q6_K): a missing or extra name, or a different shape, is refused by name before a 17-minute load
finds it. `check-rules` holds a llama-quantize `--tensor-type-file` to a table: it applies llama-quantize's own decisions (which
tensors it quantizes at all, then the first rule whose regex the lowercased name matches) to every tensor of the table and refuses
a tensor with no rule, a rule whose type is not the table's, and a tensor the quantizer leaves alone (it keeps the source's type)
whose table type is not F32; with --in, the same for a real input: a kept tensor must already have the table's type there.
`compare` holds two sets to each other: every tensor's name, type, dims and sha256, and every key but the split.* ones
(--tensors-only: the tensors alone, for a set whose architecture word and keys were renamed).

`plan` reads the log of `llama-quantize --dry-run` (the tool's own plan of every tensor's target type and size, from the input's
shapes, no data read) into the same name TAB type TAB dims TAB bytes table: the type a quantized tensor will have is the one after
`->`, an untouched tensor keeps the source's; the count of tensors it finds must be the log's own `[i/N]` total. `check-shape --types`
then holds the quantized output to that plan. `classes` sums a table (or a set's real tensors) into the classes a placement is
sized by: experts (`_exps.`), embedding, head, indexer, small F32 tensors (norms, scores, state) and the trunk (the rest); `--md`
prints a markdown table, `--field CLASS` the class's bytes alone. `fill` writes a template with every `@KEY@` replaced (a `--file`
value is the file's text); a `@KEY@` left in the output, or a key never used, is refused by name.

The output is written once: an existing output shard is refused (73). Shards are written to `<name>.tmp.<pid>`, verified and renamed,
the first shard last, so a set without its first shard is an unfinished one; a failed run removes what it wrote (except --consume).

Exit status: 0; 2 a file missing or unreadable; 64 a usage error; 65 a malformed or inconsistent input, a refusal by name, or an
output that differs from what was meant (verification); 66 gguf-py required and missing; 73 an output exists; 74 no space.
"""
import argparse
import contextlib
import errno
import fnmatch
import hashlib
import importlib.util
import io
import os
import re
import shutil
import struct
import sys
import tempfile
import time

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(file, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


Z = _load("gguf-respell.py", "gguf_respell")  # raw header reading and packing, the one place that walks a header's bytes
R = Z.R  # the header parser and the test writer (gguf-ranges.py), the instance respell holds, so the refusals are one class
Refusal = R.Refusal
CHUNK = 32 << 20

TYPE_NAME = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K",
             12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
             20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
             29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4", 41: "Q1_0", 42: "Q2_0"}
KV_TYPES = {"u8": 0, "i8": 1, "u16": 2, "i16": 3, "u32": 4, "i32": 5, "f32": 6, "bool": 7, "str": 8, "u64": 10, "i64": 11, "f64": 12}
INTS = {"u8", "i8", "u16", "i16", "u32", "i32", "u64", "i64"}
DEFAULT_LCPP = "/home/user/llama.cpp-kimik3"


NAME_TO_ID = {v.lower(): k for k, v in TYPE_NAME.items()}

# llama-quantize's tensor_allows_quantization (src/llama-quant.cpp at the pinned commit 78feec062), the part a name decides:
# a tensor is quantized only when it has two or more dims (a trailing 1 is no dim: ggml_n_dims) and its name ends with "weight"
# and holds none of these. A tensor it does not quantize keeps its source type, whatever a --tensor-type rule says.
NEVER_QUANTIZED = ("_norm.weight", "ffn_gate_inp.weight", "ffn_gate_tid2eid.weight", "engram_gate.weight", "altup", "laurel",
                   "per_layer_model_proj", "ssm_conv1d", "shortconv.conv.weight", "conv2d_spatial.weight", "conv_dw",
                   "indexer.k_proj.weight", "indexer.q_proj.weight", "time_mix_", "attn_rel_b.weight", ".position_embd", "sam.pos_embd",
                   "sam.neck.", "sam.net_", ".rel_pos", ".patch_embd", ".patch_merger", "a.rvq.codebook", "mm.a.code_embd")


def n_dims(dims):
    n = len(dims)
    while n > 1 and dims[n - 1] == 1:
        n -= 1
    return n


def quantizer_touches(name, dims):
    return (n_dims(dims) >= 2 and name.endswith("weight") and not any(x in name for x in NEVER_QUANTIZED)
            and not name.startswith(("a.conv1d.", "a.gen.wav.")) and name not in ("position_embd.weight", "token_types.weight"))


def parse_rules(path):
    """A --tensor-type-file as llama-quantize reads it: whitespace-separated `regex=type` tokens, the regex lowercased,
    the type name matched without regard to case. Returns [(compiled regex, regex text, type id)]."""
    try:
        with open(path) as f:
            toks = f.read().split()
    except OSError as e:
        raise Refusal(2, f"{path}: {e.strerror}") from e
    rules = []
    for tok in toks:
        pat, eq, ty = tok.partition("=")
        if not pat or not eq or not ty:
            raise Refusal(65, f"{path}: {tok!r} is not regex=type")
        if ty.lower() not in NAME_TO_ID:
            raise Refusal(65, f"{path}: {tok!r}: llama-quantize knows no ggml type {ty!r}")
        try:
            rules.append((re.compile(pat.lower()), pat.lower(), NAME_TO_ID[ty.lower()]))
        except re.error as e:
            raise Refusal(65, f"{path}: {tok!r}: the regex does not compile ({e})") from e
    if not rules:
        raise Refusal(65, f"{path}: no rule")
    return rules


def first_rule(rules, name):
    for rx, text, ty in rules:
        if rx.search(name):
            return text, ty
    return None


def pad(n, align):
    return (n + align - 1) // align * align


def say(msg):
    print(f"gguf-splice: {msg}", flush=True)


def human(n):
    return f"{n / 1e9:.2f} GB" if n >= 1e8 else f"{n / 1e6:.2f} MB"


# ---- reading ----

class Tensor:
    """One tensor of a file: its info as stored, its data offset (relative to the data start) and size."""
    __slots__ = ("name", "ty", "dims", "off", "nb")

    def __init__(self, name, ty, dims, off, nb):
        self.name, self.ty, self.dims, self.off, self.nb = name, ty, dims, off, nb


class Shard:
    """One GGUF file read: the metadata as raw (key, type, value bytes) in file order, the tensors in file order with sizes, the
    data start and the file size. A file whose tensors are not laid out the way mainline's reader requires (in info order,
    each at the end of the one before, padded to the alignment) is refused: this tool writes that layout and nothing else."""

    def __init__(self, path, input_path=None):
        self.path = path
        self.input_path = input_path or path  # where this shard is (or, once consumed, was) in the input set
        self.consumed = input_path is not None  # read from the output set: the input shard is already gone
        self.meta, tensors, self.start, self.size, self.align = R.header(path)
        self.version, self.kvs, self.infos, start, size, align = Z.raw_header(path)
        if (start, size, align) != (self.start, self.size, self.align):
            raise Refusal(65, f"{path}: the two header readers disagree on the data start, the size or the alignment")
        self.tensors = []
        pos = 0
        for name, ty, dims, off in tensors:
            nb = R.nbytes(path, name, ty, dims)
            if off != pos:
                raise Refusal(65, f"{path}: tensor {name} is at data offset {off}, mainline's reader requires {pos} (the tensors in "
                                  "info order, each at the end of the one before, padded to the alignment)")
            self.tensors.append(Tensor(name, ty, dims, off, nb))
            pos += pad(nb, self.align)
        # The last tensor's own padding may be missing (some writers stop at the last byte); a tensor cut short, or a whole
        # alignment unit of extra bytes after the tensors, is not a file this tool reproduces.
        data = self.size - self.start
        last_end = self.tensors[-1].off + self.tensors[-1].nb if self.tensors else 0
        if data < last_end:
            raise Refusal(65, f"{path}: the data region is {data} bytes, its tensors end at {last_end}: the file ends inside a tensor")
        if data - pos >= self.align:
            raise Refusal(65, f"{path}: {data - pos} bytes after the last tensor's padded end ({pos} bytes of tensors, {data} in the region)")

    def value(self, key):
        return self.meta.get(key)


def load_set(first, out_first=None):
    """The shards of the set `first` opens, in order, each read; tensor names unique across the set. With `out_first`
    (a --consume run), an input shard that is gone and whose output shard is there is read from the output: it is
    already done, and its tensors are the ones the run keeps."""
    first = os.path.abspath(first)
    head = Shard(first)
    paths = R.shard_paths(first, head.meta)
    outs = out_paths(out_first, len(paths)) if out_first else None
    shards = [head]
    for i, p in enumerate(paths[1:], 1):
        if outs and not os.path.exists(p) and os.path.exists(outs[i]):
            shards.append(Shard(outs[i], input_path=p))
        else:
            shards.append(Shard(p))
    seen = {}
    for i, s in enumerate(shards):
        if len(shards) > 1 and s.meta.get("split.no") != i:
            raise Refusal(65, f"{s.path}: split.no {s.meta.get('split.no')}, expected {i}")
        for t in s.tensors:
            if t.name in seen:
                raise Refusal(65, f"{s.path}: tensor {t.name} is also in {seen[t.name]}")
            seen[t.name] = s.path
    return shards


# ---- metadata edits (raw values) ----

def kv_get(kvs, key):
    for k, t, raw in kvs:
        if k == key:
            return t, raw
    return None


def with_split_total(kvs, total, where):
    """`kvs` with split.tensors.count set to `total`, in the type the file holds it in; kvs unchanged where the key is absent."""
    out = []
    for k, t, raw in kvs:
        if k == b"split.tensors.count":
            if t not in (2, 3, 4, 5, 10, 11):
                raise Refusal(65, f"{where}: split.tensors.count is not an integer (type {t})")
            try:
                raw = struct.pack(R.SCALAR[t], total)
            except struct.error as e:
                raise Refusal(65, f"{where}: split.tensors.count {total} does not fit its type ({e})") from e
        out.append((k, t, raw))
    return out


def arch_of(shard):
    for k, t, raw in shard.kvs:
        if k == b"general.architecture":
            if t != 8:
                raise Refusal(65, f"{shard.path}: general.architecture is not a string")
            return raw[8:].decode()
    return None


def renamed(kvs, old, new, where):
    ob, nb = old.encode(), new.encode()
    out = []
    for k, t, raw in kvs:
        if k == b"general.architecture":
            raw = struct.pack("<Q", len(nb)) + nb
        elif k.startswith(ob + b"."):
            k = nb + k[len(ob):]
        out.append((k, t, raw))
    names = [k for k, _, _ in out]
    dup = sorted({k for k in names if names.count(k) > 1})
    if dup:
        raise Refusal(65, f"{where}: the renamed keys collide: {[d.decode() for d in dup]}")
    return out


def parse_kv(text):
    """`KEY=TYPE:VALUE` as (key bytes, type id, raw value bytes). Refuses (64) by what is wrong with it."""
    key, eq, rest = text.partition("=")
    tname, colon, val = rest.partition(":")
    if not key or not eq or not colon or not tname:
        raise Refusal(64, f"--kv wants KEY=TYPE:VALUE, got {text!r}")
    arr = tname.endswith("[]")
    base = tname[:-2] if arr else tname
    if base not in KV_TYPES or (arr and base == "str"):
        raise Refusal(64, f"--kv {key}: type {tname!r} is not one of {', '.join(sorted(KV_TYPES))} (arrays: the numeric ones and bool, with [])")
    t = KV_TYPES[base]

    def one(v):
        try:
            if base in INTS:
                return struct.pack(R.SCALAR[t], int(v, 0))
            if base in ("f32", "f64"):
                return struct.pack(R.SCALAR[t], float(v))
            if base == "bool":
                if v.lower() not in ("true", "false", "1", "0"):
                    raise ValueError(v)
                return struct.pack("<?", v.lower() in ("true", "1"))
            vb = v.encode()
            return struct.pack("<Q", len(vb)) + vb
        except (ValueError, struct.error) as e:
            raise Refusal(64, f"--kv {key}: {v!r} is not a {base} ({e})") from e

    if arr:
        items = [x for x in val.split(",") if x != ""] if val else []
        return key.encode(), 9, struct.pack("<IQ", t, len(items)) + b"".join(one(x) for x in items)
    return key.encode(), t, one(val)


def apply_kv(kvs, edits, replace, where):
    out = list(kvs)
    notes = []
    for k, t, raw in edits:
        at = [i for i, (kk, _, _) in enumerate(out) if kk == k]
        if not at:
            out.append((k, t, raw))
            notes.append(f"--kv {k.decode()}: added")
        elif out[at[0]][1] == t and out[at[0]][2] == raw:
            notes.append(f"--kv {k.decode()}: the file already has it, the same value")
        elif replace:
            out[at[0]] = (k, t, raw)
            notes.append(f"--kv {k.decode()}: replaced")
        else:
            raise Refusal(65, f"{where}: --kv {k.decode()}: the key exists with another value (--kv-replace replaces it)")
    return out, notes


# ---- the plan ----

class Out:
    """A tensor of an output shard and where its bytes come from."""
    __slots__ = ("name", "ty", "dims", "nb", "src", "src_off", "sha")

    def __init__(self, name, ty, dims, nb, src, src_off):
        self.name, self.ty, self.dims, self.nb, self.src, self.src_off, self.sha = name, ty, dims, nb, src, src_off, None


def info_bytes(tensors, align):
    off, out = 0, b""
    for t in tensors:
        nb = t.name.encode()
        out += struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(t.dims))
        out += b"".join(struct.pack("<Q", d) for d in t.dims) + struct.pack("<IQ", t.ty, off)
        off += pad(t.nb, align)
    return out


def header_of(version, kvs, tensors, align):
    return Z._pack(version, len(tensors), kvs, info_bytes(tensors, align), align)


class Planned:
    """One output shard: its source shard, its tensors, its metadata, the header bytes, and whether it is linked."""

    def __init__(self, idx, src, tensors, kvs, link):
        self.idx, self.src, self.tensors, self.kvs, self.link = idx, src, tensors, kvs, link
        self.header = header_of(src.version, kvs, tensors, src.align)
        self.data = sum(pad(t.nb, src.align) for t in tensors)
        # a linked shard is the input's own file, with whatever it ends in
        self.size = src.size if link else len(self.header) + self.data


def out_paths(out_first, count):
    out_first = os.path.abspath(out_first)
    if count == 1:
        return [out_first]
    m = R.SPLIT.search(out_first)
    if not m or int(m.group(1)) != 1 or int(m.group(2)) != count:
        raise Refusal(64, f"--out {out_first}: the input is {count} shards and the count stays, so the output is named -00001-of-{count:05d}.gguf")
    return [out_first[: m.start()] + f"-{i:05d}-of-{count:05d}.gguf" for i in range(1, count + 1)]


def match_names(names, patterns, flag, where, allow_empty=False):
    """The names the patterns select, each pattern holding at least one; refused by name otherwise. `allow_empty`: a
    resumed --consume run, whose finished shards no longer hold what the pattern dropped from them."""
    picked = set()
    for p in patterns:
        hit = [n for n in names if fnmatch.fnmatchcase(n, p)]
        if not hit and not allow_empty:
            raise Refusal(65, f"{flag} {p!r} matches no tensor of {where}")
        picked.update(hit)
    return picked


def plan(shards, drops, extra_set, takes, arch, kv_edits, kv_replace, into, link, from_first=None):
    """The output shards: `drops` removed, `takes` of `extra_set` appended to shard `into` (0-based), the keys edited. Returns
    (planned list, notes)."""
    notes = []
    resumed = [s for s in shards if s.consumed]
    names = [t.name for s in shards for t in s.tensors]
    live_names = [t.name for s in shards if not s.consumed for t in s.tensors]
    dropped = match_names(live_names, drops, "--drop", shards[0].path, allow_empty=bool(resumed)) if drops else set()
    if len(dropped) == len(live_names) and not resumed and drops:
        raise Refusal(65, "--drop selects every tensor of the set: nothing would be left")
    if resumed:
        notes.append(f"resumed: shard(s) {', '.join(str(shards.index(s) + 1) for s in resumed)} are already done (read from the output)")
    kept = set(names) - dropped
    extras = []
    if takes:
        from_names = [t.name for s in extra_set for t in s.tensors]
        taken = match_names(from_names, takes, "--take", extra_set[0].path)
        for s in extra_set:
            for t in s.tensors:
                if t.name in taken:
                    if t.name in kept:
                        raise Refusal(65, f"--take: tensor {t.name} is already in {shards[0].path}")
                    extras.append((s, t))
        if not extras:
            raise Refusal(65, "--take selects no tensor")
        if len({t.name for _, t in extras}) != len(extras):
            raise Refusal(65, f"--take: a tensor name appears in two shards of {extra_set[0].path}")
    total = len(kept) + len(extras)
    if into is not None and not 0 <= into < len(shards):
        raise Refusal(64, f"--into {into + 1}: the set has {len(shards)} shard(s)")
    target = len(shards) - 1 if into is None else into
    # The architecture the file says; every shard that holds one must agree.
    old = None
    if arch:
        for s in shards:
            a = arch_of(s)
            if a is None:
                continue
            if old not in (None, a):
                raise Refusal(65, f"{s.path}: general.architecture {a!r}, another shard says {old!r}")
            old = a
        if old is None:
            raise Refusal(65, f"--arch {arch}: no shard has a general.architecture")
        if old == arch:
            raise Refusal(64, f"--arch {arch} is the name the file already has")
    out = []
    for i, s in enumerate(shards):
        if s.consumed:
            out.append(Planned(i, s, [Out(t.name, t.ty, t.dims, t.nb, s.path, s.start + t.off) for t in s.tensors], list(s.kvs), False))
            continue
        tensors = [Out(t.name, t.ty, t.dims, t.nb, s.path, s.start + t.off) for t in s.tensors if t.name in kept]
        if i == target:
            tensors += [Out(t.name, t.ty, t.dims, t.nb, es.path, es.start + t.off) for es, t in extras]
        changed_tensors = len(tensors) != len(s.tensors) or i == target and bool(extras)
        kvs = list(s.kvs)
        if arch and (arch_of(s) is not None or any(k.startswith(old.encode() + b".") for k, _, _ in kvs)):
            kvs = renamed(kvs, old, arch, s.path)
        if kv_edits and i == 0:
            kvs, n = apply_kv(kvs, kv_edits, kv_replace, s.path)
            notes += n
        kv_changed = kvs != list(s.kvs)
        # A shard that is linked keeps its own split.tensors.count; every shard that is written carries the new total.
        unchanged = link and not changed_tensors and not kv_changed and (i != 0 or total == len(names))
        if not unchanged:
            kvs = with_split_total(kvs, total, s.path)
        out.append(Planned(i, s, tensors, kvs, unchanged))
    if arch:
        notes.append(f"--arch {arch}: general.architecture and the {old}.* keys renamed")
    return out, notes


# ---- verification ----

def hash_range(path, off, n):
    h = hashlib.sha256()
    with open(path, "rb", buffering=0) as f:
        f.seek(off)
        buf = bytearray(min(CHUNK, max(n, 1)))
        view = memoryview(buf)
        left = n
        while left:
            k = f.readinto(view[: min(left, len(buf))])
            if not k:
                raise Refusal(65, f"{path}: ends inside a tensor ({left} bytes short)")
            h.update(view[:k])
            left -= k
    return h.hexdigest()


def cmp_hash(path, off, n):
    """compare's own seam onto hash_range, so the self-test can break compare's hashing without breaking the write-back verification."""
    return hash_range(path, off, n)


def drop_cache(path):
    """Ask the kernel to forget a file's pages: a terabyte copy would evict whatever else the box had cached."""
    if hasattr(os, "posix_fadvise"):
        try:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except OSError:
            pass


def gguf_py(mode):
    """The gguf module of $LCPP/gguf-py, or None where mode is auto and it is not there. No bytecode is written into that tree."""
    if mode == "off":
        return None
    lcpp = os.environ.get("LCPP") or DEFAULT_LCPP
    d = os.path.join(lcpp, "gguf-py")
    why = None
    if not os.path.isdir(os.path.join(d, "gguf")):
        why = f"no {d}/gguf (LCPP names a llama.cpp tree)"
    else:
        sys.dont_write_bytecode = True
        if d not in sys.path:
            sys.path.insert(0, d)
        try:
            import gguf  # noqa: PLC0415
            return gguf
        except ImportError as e:
            why = f"gguf-py does not import ({e})"
    if mode == "required":
        raise Refusal(66, f"--gguf-py required: {why}")
    say(f"gguf-py cross-check skipped: {why}")
    return None


def cross_check(gg, path, tensors, align):
    """Read `path` through gguf-py and hold it to this tool's own reading: the tensors and the key names."""
    try:
        rd = gg.GGUFReader(path)
    except Exception as e:  # noqa: BLE001 - whatever gguf-py raises is the verdict
        raise Refusal(65, f"{path}: gguf-py's GGUFReader refuses it: {e}") from e
    mine = R.header(path)
    names = [t.name for t in rd.tensors]
    if names != [t[0] for t in mine[1]]:
        raise Refusal(65, f"{path}: gguf-py reads the tensors {names[:3]}..., this tool reads {[t[0] for t in mine[1]][:3]}...")
    for t, (name, ty, dims, off) in zip(rd.tensors, mine[1]):
        want = (tuple(dims), ty, R.nbytes(path, name, ty, dims), mine[2] + off)
        got = (tuple(int(x) for x in t.shape), int(t.tensor_type), int(t.n_bytes), int(t.data_offset))
        if got != want:
            raise Refusal(65, f"{path}: tensor {name}: gguf-py reads (shape, type, bytes, offset) {got}, this tool {want}")
    keys = {k for k in rd.fields if not k.startswith("GGUF.")}
    if keys != set(mine[0]):
        raise Refusal(65, f"{path}: gguf-py's keys differ from this tool's: {sorted(keys ^ set(mine[0]))}")
    for k in ("general.architecture", "general.alignment", "split.no", "split.count", "split.tensors.count"):
        if k in rd.fields:
            v = rd.fields[k].contents()
            w = mine[0][k]
            if (v.encode() if isinstance(v, str) else v) != w:
                raise Refusal(65, f"{path}: key {k}: gguf-py reads {v!r}, this tool {w!r}")


def verify_shard(path, planned, mode, gg):
    """Hold the written `path` to what was meant: the header bytes, the layout, every tensor's bytes (mode data), gguf-py."""
    got = Shard(path)
    want_kvs, want_infos = planned.kvs, info_bytes(planned.tensors, planned.src.align)
    if got.kvs != want_kvs:
        have, wnt = {k: (t, v) for k, t, v in got.kvs}, {k: (t, v) for k, t, v in want_kvs}
        diff = sorted(k.decode("utf-8", "replace") for k in set(have) ^ set(wnt) | {k for k in have if k in wnt and have[k] != wnt[k]})
        raise Refusal(65, f"{path}: the keys differ from the ones meant: {diff or 'their order'}")
    if got.infos != want_infos or [t.name for t in got.tensors] != [t.name for t in planned.tensors]:
        raise Refusal(65, f"{path}: the tensor names, shapes, types or offsets differ from the ones meant")
    if got.size != planned.size:
        raise Refusal(65, f"{path}: {got.size} bytes, {planned.size} meant")
    if mode == "data":
        for t, o in zip(got.tensors, planned.tensors):
            h = hash_range(path, got.start + t.off, t.nb)
            if h != o.sha:
                raise Refusal(65, f"{path}: tensor {t.name}: sha256 {h[:16]} on disk, {o.sha[:16]} as read from {o.src}")
    if gg is not None:
        cross_check(gg, path, got.tensors, got.align)


# ---- writing ----

def copy_shard(planned, tmp):
    """Write `tmp`: the header, then each tensor's bytes from its source, padded; every tensor's sha256 as it is copied."""
    align = planned.src.align
    handles = {}
    buf = bytearray(CHUNK)
    view = memoryview(buf)
    try:
        with open(tmp, "wb") as out:
            out.write(planned.header)
            for t in planned.tensors:
                f = handles.get(t.src)
                if f is None:
                    f = handles[t.src] = open(t.src, "rb", buffering=0)
                f.seek(t.src_off)
                h = hashlib.sha256()
                left = t.nb
                while left:
                    k = f.readinto(view[: min(left, CHUNK)])
                    if not k:
                        raise Refusal(2, f"{t.src}: ends inside tensor {t.name}")
                    h.update(view[:k])
                    out.write(view[:k])
                    left -= k
                out.write(b"\0" * (pad(t.nb, align) - t.nb))
                t.sha = h.hexdigest()
            out.flush()
            os.fsync(out.fileno())
    finally:
        for f in handles.values():
            f.close()


def run(planned, shards, out_first, consume, verify, gg, space_check):
    """Write the planned set. Returns the output paths."""
    n = len(shards)
    outs = out_paths(out_first, n)
    ins = [s.input_path for s in shards]
    clash = [o for o in outs if o in ins]
    if clash:
        raise Refusal(64, f"--out names a shard of the input ({clash[0]}): the output set is written beside the input, never over it")
    out_dir = os.path.dirname(outs[0])
    if not os.path.isdir(out_dir):
        raise Refusal(2, f"{out_dir} is not a directory")
    order = list(range(1, n)) + [0] if consume else list(range(n))
    # What is already there: an output is never overwritten; with --consume, a shard whose input is gone and whose output is there is done.
    done = set()
    for i in range(n):
        if not os.path.lexists(outs[i]):
            continue
        if not consume:
            raise Refusal(73, f"{outs[i]} exists: an output shard is written once (remove it, or pick another --out)")
        if os.path.exists(ins[i]):
            # A run that died between the rename of an output (renamed only after it was verified) and the delete of its input.
            try:
                verify_shard(outs[i], planned[i], "header", gg)
            except Refusal as e:
                raise Refusal(73, f"{outs[i]} exists and is not the shard this run would write ({e}): remove it, or pick another --out") from e
            os.unlink(ins[i])
            say(f"shard {i + 1}/{n}: output present and its header verified, the input it was written from deleted")
        done.add(i)
    writes = [p for p in planned if p.idx not in done and not p.link]
    if space_check and writes:
        need = max(p.size for p in writes) if consume else sum(p.size for p in writes)
        free = os.statvfs(out_dir).f_bavail * os.statvfs(out_dir).f_frsize
        if free < need:
            raise Refusal(74, f"{out_dir}: {free} bytes free, the shards to write need {need}" + ("" if consume else " (--consume writes one shard at a time)"))
    made, tmps = [], []
    t_all = time.time()
    try:
        for i in order:
            p = planned[i]
            if i in done:
                m = R.header(outs[i])
                if n > 1 and (m[0].get("split.no") != i or m[0].get("split.count") != n):
                    raise Refusal(65, f"{outs[i]}: the finished output of shard {i + 1} has split.no {m[0].get('split.no')}, split.count {m[0].get('split.count')}")
                say(f"shard {i + 1}/{n}: done before (output present, input gone): skipped")
                continue
            t0 = time.time()
            if p.link:
                how = None
                try:
                    if consume:
                        os.rename(ins[i], outs[i])
                        how = "renamed from the input"
                    else:
                        os.link(ins[i], outs[i])
                        how = "hard-linked from the input"
                except OSError as e:
                    if e.errno != errno.EXDEV:  # another filesystem: copy it
                        raise Refusal(2, f"{outs[i]}: cannot link from {ins[i]}: {e.strerror}") from e
                if how:
                    made.append(outs[i])
                    verify_shard(outs[i], p, "header", gg)
                    say(f"shard {i + 1}/{n}: {how} ({human(os.path.getsize(outs[i]))})")
                    continue
                p = Planned(i, p.src, p.tensors, p.src.kvs, False)
            tmp = f"{outs[i]}.tmp.{os.getpid()}"
            tmps.append(tmp)
            copy_shard(p, tmp)
            t1 = time.time()
            verify_shard(tmp, p, verify, gg)
            os.rename(tmp, outs[i])
            tmps.remove(tmp)
            made.append(outs[i])
            drop_cache(outs[i])
            if consume:
                os.unlink(ins[i])
            else:
                drop_cache(ins[i])
            say(f"shard {i + 1}/{n}: {human(p.size)} written in {t1 - t0:.1f} s ({p.size / max(t1 - t0, 1e-9) / 1e9:.2f} GB/s), "
                f"verified ({verify}{', gguf-py' if gg is not None else ''}) in {time.time() - t1:.1f} s{', input deleted' if consume else ''}")
    except BaseException:
        for t in tmps:
            if os.path.lexists(t):
                os.unlink(t)
        if not consume:
            for o in made:
                if os.path.lexists(o):
                    os.unlink(o)
            if made or tmps:
                say(f"removed what this run wrote ({len(made)} shard(s), {len(tmps)} unfinished)")
        else:
            say("--consume: the finished shards are kept; run the same command again to resume")
        raise
    say(f"set of {n} shard(s) at {outs[0]} in {time.time() - t_all:.1f} s")
    return outs


# ---- commands ----

def cmd_transform(a, strip):
    if a.consume and not strip:
        raise Refusal(64, "--consume is for strip: an add rewrites only the receiving shard and the first one, so --link already costs no more than "
                          "those two shards of disk")
    if not strip and a.arch and "." in a.arch:
        raise Refusal(64, "--arch is a name without a dot")
    if a.consume and not os.path.exists(a.inp):
        if os.path.exists(a.out):
            say(f"--consume: the input's first shard is gone and {a.out} is there: the run is complete (a first shard is the last one a run finishes)")
            return 0
        raise Refusal(2, f"{a.inp}: no such file")
    shards = load_set(a.inp, a.out if a.consume else None)
    extra = load_set(a.src) if not strip else None
    edits = [parse_kv(x) for x in (a.kv or [])]
    into = None if a.into is None else a.into - 1
    gg = gguf_py(a.gguf_py)
    planned, notes = plan(shards, a.drop or [], extra, a.take or [], a.arch, edits, a.kv_replace, into, a.link or a.consume)
    names_in = sum(len(s.tensors) for s in shards)
    names_out = sum(len(p.tensors) for p in planned)
    say(f"{'strip' if strip else 'add'}: {len(shards)} shard(s), {names_in} tensors in, {names_out} out")
    for n in notes:
        say(n)
    if a.dry_run:
        for p in planned:
            say(f"shard {p.idx + 1}: {len(p.src.tensors)} -> {len(p.tensors)} tensors, {'linked' if p.link else 'written'}, {human(p.size)}")
        return 0
    run(planned, shards, a.out, a.consume, a.verify, gg, not a.no_space_check)
    return 0


def tsv_lines(shards):
    for i, s in enumerate(shards):
        for t in s.tensors:
            yield f"{t.name}\t{TYPE_NAME.get(t.ty, 'type%d' % t.ty)}\t{'x'.join(str(d) for d in t.dims)}\t{i + 1}"


def cmd_list(a):
    shards = load_set(a.inp)
    gg = gguf_py(a.gguf_py)
    if gg is not None:
        for s in shards:
            cross_check(gg, s.path, s.tensors, s.align)
    if a.tsv:
        for line in tsv_lines(shards):
            print(line)
        return 0
    by = {}
    for i, s in enumerate(shards):
        print(f"shard {i + 1}/{len(shards)} {s.path} tensors={len(s.tensors)} data_start={s.start} bytes={s.size}")
        for t in s.tensors:
            c = by.setdefault(TYPE_NAME.get(t.ty, "type%d" % t.ty), [0, 0])
            c[0] += 1
            c[1] += t.nb
    arch = arch_of(shards[0])
    print(f"architecture={arch} tensors={sum(c[0] for c in by.values())} bytes={sum(s.size for s in shards)}"
          + (f" gguf-py={'cross-checked' if gg is not None else 'not used'}"))
    for name, (c, b) in sorted(by.items(), key=lambda kv: -kv[1][1]):
        print(f"  {name:8s} {c:6d} tensors {b:16d} B")
    return 0


def read_tsv(path):
    """name -> (type name, dims text) of a name TAB type TAB dims [TAB shard] table."""
    try:
        with open(path) as f:
            rows = [ln.rstrip("\n").split("\t") for ln in f if ln.strip()]
    except OSError as e:
        raise Refusal(2, f"{path}: {e.strerror}") from e
    if not rows or any(len(r) < 3 for r in rows):
        raise Refusal(65, f"{path}: not a name TAB type TAB dims table")
    names = [r[0] for r in rows]
    if len(set(names)) != len(names):
        raise Refusal(65, f"{path}: a tensor name appears twice")
    return {r[0]: (r[1], r[2]) for r in rows}


def tname(ty):
    return TYPE_NAME.get(ty, "type%d" % ty)


def norm_dims(text):
    """`7168x1` and `7168` are one shape: ggml pads a tensor's dims with 1s, a log prints all four."""
    d = [x for x in text.split("x") if x != ""]
    while len(d) > 1 and d[-1] == "1":
        d.pop()
    return "x".join(d)


_PLAN_ROW = re.compile(r"^\[\s*(\d+)/\s*(\d+)\] (\S+)\s+- \[([^\]]*)\], type =\s*(\S+), size =\s*([0-9.]+) MiB"
                       r"(?: ->\s*([0-9.]+) MiB \((\S+)\))?\s*$")
_TYPE_BY_UPPER = {v.upper(): v for v in TYPE_NAME.values()}


def plan_type(spelling, where):
    ty = _TYPE_BY_UPPER.get(spelling.upper())
    if ty is None:
        raise Refusal(65, f"{where}: the type '{spelling}' is none llama-quantize's log names here")
    return ty


def plan_rows(text, where):
    """The rows (name, type, dims, bytes) of a `llama-quantize --dry-run` log, in its order."""
    rows, total = [], None
    for ln in text.splitlines():
        m = _PLAN_ROW.match(ln)
        if not m:
            continue
        idx, tot, name, dims, src, a, b, new = m.groups()
        total = int(tot)
        d = norm_dims("x".join(x.strip() for x in dims.split(",")))
        rows.append((name, plan_type(new if new else src, where), d, round(float(b if b else a) * 1048576)))
    if not rows:
        raise Refusal(65, f"{where}: no tensor line of a llama-quantize --dry-run log (`[ i/N] name - [dims], type = T, size = ...`)")
    if len(rows) != total or len({r[0] for r in rows}) != len(rows):
        raise Refusal(65, f"{where}: {len(rows)} tensor lines ({len({r[0] for r in rows})} distinct names), the log says {total}")
    return rows


def cmd_plan(a):
    try:
        with open(a.log) as f:
            text = f.read()
    except OSError as e:
        raise Refusal(2, f"{a.log}: {e.strerror}") from e
    rows = plan_rows(text, a.log)
    with open(a.out, "w") as f:
        for name, ty, dims, nb in rows:
            f.write(f"{name}\t{ty}\t{dims}\t{nb}\n")
    say(f"plan: {len(rows)} tensors, {sum(r[3] for r in rows)} B planned, written to {a.out}")
    return 0


CLASSES = ("experts", "embedding", "head", "indexer", "trunk", "small f32")


def classify(name, ty):
    if "_exps." in name:
        return "experts"
    if name == "token_embd.weight":
        return "embedding"
    if name == "output.weight":
        return "head"
    if ".indexer." in name:
        return "indexer"
    return "small f32" if ty == "F32" else "trunk"


def class_sums(rows):
    """class -> [tensors, bytes, {type: count}] of (name, type, bytes) rows."""
    out = {c: [0, 0, {}] for c in CLASSES}
    for name, ty, nb in rows:
        c = out[classify(name, ty)]
        c[0] += 1
        c[1] += nb
        c[2][ty] = c[2].get(ty, 0) + 1
    return out


def cmd_classes(a):
    if bool(a.tsv) == bool(a.inp):
        raise Refusal(64, "classes takes --tsv TABLE or --in SET, one of them")
    if a.tsv:
        try:
            with open(a.tsv) as f:
                raw = [ln.rstrip("\n").split("\t") for ln in f if ln.strip()]
        except OSError as e:
            raise Refusal(2, f"{a.tsv}: {e.strerror}") from e
        if not raw or any(len(r) < 4 or not r[3].isdigit() for r in raw):
            raise Refusal(65, f"{a.tsv}: not a name TAB type TAB dims TAB bytes table (`plan` writes one)")
        rows = [(r[0], r[1], int(r[3])) for r in raw]
    else:
        rows = [(t.name, tname(t.ty), t.nb) for s in load_set(a.inp) for t in s.tensors]
    sums = class_sums(rows)
    if a.field:
        if a.field not in sums:
            raise Refusal(64, f"--field {a.field}: one of {', '.join(CLASSES)}")
        print(sums[a.field][1])
        return 0
    total = sum(c[1] for c in sums.values())
    if a.md:
        print("| class | tensors | types | GB |\n|---|---|---|---|")
    for cls in CLASSES:
        n, nb, ty = sums[cls]
        if n == 0:
            continue
        types = ", ".join(f"{k} {v}" for k, v in sorted(ty.items(), key=lambda kv: -kv[1]))
        print(f"| {cls} | {n} | {types} | {nb / 1e9:.2f} |" if a.md else f"  {cls:10s} {n:6d} tensors {nb / 1e9:9.2f} GB  {types}")
    print(f"| total | {sum(c[0] for c in sums.values())} | | {total / 1e9:.2f} |" if a.md else f"  {'total':10s} {sum(c[0] for c in sums.values()):6d} tensors {total / 1e9:9.2f} GB")
    return 0


_HOLE = re.compile(r"@[A-Z][A-Z0-9_]*@")


def fill_text(text, values):
    """`text` with every @KEY@ of `values` replaced; a hole left, or a key without one, is refused by name."""
    used = set()

    def sub(m):
        key = m.group(0)[1:-1]
        if key not in values:
            return m.group(0)
        used.add(key)
        return values[key]

    out = _HOLE.sub(sub, text)
    left = sorted(set(_HOLE.findall(out)))
    if left:
        raise Refusal(65, f"the template has {len(left)} placeholder(s) no value was given for: {', '.join(left[:6])}")
    unused = sorted(set(values) - used)
    if unused:
        raise Refusal(65, f"{len(unused)} value(s) given for a placeholder the template does not have: {', '.join(unused[:6])}")
    return out


def cmd_fill(a):
    try:
        with open(a.template) as f:
            text = f.read()
    except OSError as e:
        raise Refusal(2, f"{a.template}: {e.strerror}") from e
    values = {}
    for kind, items in (("--set", a.set or []), ("--file", a.file or [])):
        for item in items:
            key, sep, val = item.partition("=")
            if not sep or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
                raise Refusal(64, f"{kind} {item}: KEY=VALUE with KEY like A_B1")
            if key in values:
                raise Refusal(64, f"{kind} {key}: given twice")
            if kind == "--file":
                try:
                    with open(val) as f:
                        val = f.read().rstrip("\n")
                except OSError as e:
                    raise Refusal(2, f"{val}: {e.strerror}") from e
            values[key] = val
    out = fill_text(text, values)
    if os.path.exists(a.out):
        raise Refusal(73, f"{a.out} exists")
    with open(a.out, "w") as f:
        f.write(out if out.endswith("\n") else out + "\n")
    say(f"fill: {a.out}, {len(values)} placeholder(s) filled")
    return 0


def cmd_check_shape(a):
    shards = load_set(a.inp)
    want = read_tsv(a.names)
    have = {t.name: (tname(t.ty), "x".join(str(d) for d in t.dims)) for s in shards for t in s.tensors}
    missing = sorted(set(want) - set(have))
    extra = sorted(set(have) - set(want))
    wrong = sorted(n for n in set(want) & set(have) if norm_dims(want[n][1]) != norm_dims(have[n][1]))
    typed = sorted(n for n in set(want) & set(have) if a.types and want[n][0] != have[n][0])
    if missing or extra or wrong or typed:
        for label, items in (("missing from the file", missing), ("not in the table", extra)):
            if items:
                print(f"gguf-splice: {len(items)} tensor(s) {label}: {', '.join(items[:6])}{' ...' if len(items) > 6 else ''}", file=sys.stderr)
        if wrong:
            print(f"gguf-splice: {len(wrong)} tensor(s) with another shape, e.g. {wrong[0]}: file {have[wrong[0]][1]}, table {want[wrong[0]][1]}", file=sys.stderr)
        if typed:
            print(f"gguf-splice: {len(typed)} tensor(s) with another type, e.g. {typed[0]}: file {have[typed[0]][0]}, table {want[typed[0]][0]}", file=sys.stderr)
        raise Refusal(65, f"{a.inp}: not the table's {'shape and types' if a.types else 'shape'} ({len(have)} tensors, the table {len(want)}): mainline's loader would refuse it")
    say(f"check-shape: {len(have)} tensors, names and dims{' and types' if a.types else ''} equal the table's {a.names}")
    return 0


def parse_type_pairs(text, what):
    """`A=B,C=D` as a list of (A, B); every B (and, for --retype, every A) must be a type llama-quantize knows."""
    out = []
    for tok in (text or "").split(","):
        if not tok:
            continue
        k, eq, v = tok.partition("=")
        if not k or not eq or v.lower() not in NAME_TO_ID:
            raise Refusal(64, f"{what}: {tok!r} is not NAME=type with a type llama-quantize knows")
        out.append((k, v))
    return out


def cmd_check_rules(a):
    """The rules against a reference table (and, with --in, a real input): derived from llama-quantize's own decisions.
    --retype FROM=TO maps the table's types before the comparison (the tight map is the K3 table's Q6_K trunk as Q4_K);
    --flag-types NAME=type names the tensors a flag types (--token-embedding-type, --output-tensor-type): no rule may match
    them and the table's type is not what they get."""
    rules = parse_rules(a.rules)
    table = read_tsv(a.table)
    partial = getattr(a, "partial", False)
    retype = {k.upper(): tname(NAME_TO_ID[v.lower()]) for k, v in parse_type_pairs(getattr(a, "retype", None), "--retype")}
    flagged = {k: tname(NAME_TO_ID[v.lower()]) for k, v in parse_type_pairs(getattr(a, "flag_types", None), "--flag-types")}
    problems, quantized, kept, by_rule, to_global, by_flag = [], 0, 0, {}, 0, 0
    for name in flagged:
        if name not in table:
            problems.append(f"--flag-types names {name}, which is not in the table")
    for name, (ty, dims) in table.items():
        d = [int(x) for x in dims.split("x")]
        if quantizer_touches(name, d):
            quantized += 1
            hit = first_rule(rules, name)
            if name in flagged:
                by_flag += 1
                if hit is not None:
                    problems.append(f"{name}: a flag types it ({flagged[name]}) and rule {hit[0]} matches it too")
                continue
            if hit is None:
                to_global += 1
                if not partial:
                    problems.append(f"{name}: the quantizer quantizes it and no rule matches, so it would get the global type")
            else:
                by_rule[hit[0]] = by_rule.get(hit[0], 0) + 1
                want = retype.get(ty.upper(), ty)
                if not partial and tname(hit[1]) != want:
                    problems.append(f"{name}: rule {hit[0]} gives {tname(hit[1])}, the table has {ty}" + (f" (wanted {want} by --retype)" if want != ty else ""))
        else:
            kept += 1
            if ty != "F32":
                problems.append(f"{name}: the quantizer leaves it alone (it keeps its source type) and the table has {ty}, not F32")
    unused = [text for _, text, _ in rules if text not in by_rule]
    if unused:
        problems.append(f"{len(unused)} rule(s) match no tensor the quantizer touches: {', '.join(unused[:4])}")
    if a.inp:
        have = {t.name: (tname(t.ty), "x".join(str(x) for x in t.dims)) for s in load_set(a.inp) for t in s.tensors}
        for name in sorted(set(table) - set(have)):
            problems.append(f"{name}: missing from {a.inp}")
        for name in sorted(set(have) - set(table)):
            problems.append(f"{name}: in {a.inp}, not in the table")
        for name, (ty, dims) in table.items():
            if name in have:
                if norm_dims(have[name][1]) != norm_dims(dims):
                    problems.append(f"{name}: dims {have[name][1]} in {a.inp}, {dims} in the table")
                d = [int(x) for x in dims.split("x")]
                if not quantizer_touches(name, d) and have[name][0] != ty:
                    problems.append(f"{name}: the quantizer will not touch it and the input has {have[name][0]}, the table {ty}: the output would keep {have[name][0]}")
    if problems:
        for pr in problems[:25]:
            print(f"gguf-splice: {pr}", file=sys.stderr)
        raise Refusal(65, f"{a.rules}: {len(problems)} problem(s) against {a.table}" + (f" and {a.inp}" if a.inp else ""))
    say(f"check-rules: {len(table)} tensors: {quantized - to_global - by_flag} quantized by a rule ({', '.join(f'{k}: {v}' for k, v in sorted(by_rule.items()))}), "
        + (f"{by_flag} typed by a flag, " if by_flag else "")
        + (f"{to_global} left to the global type, " if partial else "")
        + f"{kept} kept as the source has them (all F32 in the table)" + (f"; {a.inp} has the kept ones' types" if a.inp else ""))
    return 0


def cmd_compare(a):
    A, B = load_set(a.a), load_set(b := a.b)

    def table(shards):
        out = {}
        for s in shards:
            for t in s.tensors:
                out[t.name] = (t.ty, tuple(t.dims), t.nb, None if a.no_data else cmp_hash(s.path, s.start + t.off, t.nb))
        return out

    def keys(shards):
        out = {}
        for s in shards:
            for k, t, raw in s.kvs:
                if not k.startswith(b"split."):
                    if out.get(k, (t, raw)) != (t, raw):
                        raise Refusal(65, f"{s.path}: key {k.decode()} differs between the shards of one set")
                    out[k] = (t, raw)
        return out

    ta, tb = table(A), table(B)
    problems = [f"tensor {n} is only in {a.a if n in ta else a.b}" for n in sorted(set(ta) ^ set(tb))]
    for n in sorted(set(ta) & set(tb)):
        if ta[n] != tb[n]:
            what = "type" if ta[n][0] != tb[n][0] else "dims" if ta[n][1] != tb[n][1] else "bytes"
            problems.append(f"tensor {n}: {what} differ ({tname(ta[n][0])} {ta[n][1]} vs {tname(tb[n][0])} {tb[n][1]})")
    ka, kb = ({}, {}) if a.tensors_only else (keys(A), keys(B))
    problems += [f"key {k.decode()} is only in {a.a if k in ka else a.b}" for k in sorted(set(ka) ^ set(kb))]
    problems += [f"key {k.decode()} has another value" for k in sorted(set(ka) & set(kb)) if ka[k] != kb[k]]
    if problems:
        for pr in problems[:20]:
            print(f"gguf-splice: {pr}", file=sys.stderr)
        raise Refusal(65, f"{a.a} and {b} differ ({len(problems)} point(s))")
    say(f"compare: {len(ta)} tensors{'' if a.no_data else ' (sha256 of every byte)'} and {'no keys (--tensors-only)' if a.tensors_only else f'{len(ka)} keys'} equal, "
        f"split.* keys aside; {len(A)} shard(s) against {len(B)}")
    return 0


# ---- the self-test ----

def _blob(name, nb):
    return hashlib.shake_128(name.encode()).digest(nb)


def _t(name, ty, dims):
    return (name, ty, dims, _blob(name, R.nbytes("fixture", name, ty, dims)))


def _fixture(d, name="a"):
    """A three-shard set in K3's layout: shard 1 the metadata (a string array, the alignment, the split keys, a total of 10) and no
    tensor; shards 2 and 3 tensors (8 in all), with indexer tensors (BF16 and F32) in both."""
    kv_s, kv_u = R._kv_string, R._kv_scalar
    arr = struct.pack("<Q", 5) + b"tokns" + struct.pack("<I", 9) + struct.pack("<IQ", 8, 2)
    arr += struct.pack("<Q", 1) + b"a" + struct.pack("<Q", 3) + b"bcd"
    split = lambda no: [kv_u("split.no", 2, no), kv_u("split.count", 2, 3), kv_u("split.tensors.count", 5, 8)]
    os.makedirs(os.path.join(d, name))
    base = os.path.join(d, name, "m-0000{}-of-00003.gguf")
    R.write_gguf(base.format(1), [kv_s("general.architecture", "kimi-k3"), kv_u("kimi-k3.block_count", 4, 6), arr,
                                  kv_u("kimi-k3.rope.scale", 6, 0.5), kv_u("general.alignment", 4, 32)] + split(0), [])
    R.write_gguf(base.format(2), split(1), [
        _t("token_embd.weight", 14, [256, 4]), _t("blk.0.attn_q.weight", 8, [64, 3]),
        _t("blk.3.indexer.proj.weight", 30, [32, 4]), _t("blk.3.indexer.k_norm.weight", 0, [32]),
        _t("blk.3.indexer.k_norm.bias", 0, [32])])
    R.write_gguf(base.format(3), split(2), [
        _t("blk.7.ffn_up_exps.weight", 19, [256, 2, 3]), _t("blk.7.indexer.attn_k.weight", 30, [16, 8]),
        _t("output.weight", 14, [256, 2])])
    return base.format(1)


def _indexer_file(d):
    """A one-file GGUF of the indexer tensors of the fixture's layer 3 and 7 and one tensor that is not an indexer's."""
    kv_s = R._kv_string
    path = os.path.join(d, "indexer.gguf")
    R.write_gguf(path, [kv_s("general.architecture", "kimi-k3"), R._kv_scalar("general.alignment", 4, 32)], [
        _t("blk.3.indexer.proj.weight", 30, [32, 4]), _t("blk.3.indexer.k_norm.weight", 0, [32]),
        _t("blk.3.indexer.k_norm.bias", 0, [32]), _t("blk.7.indexer.attn_k.weight", 30, [16, 8]),
        _t("blk.7.not_indexer.weight", 0, [8])])
    return path


def _tensors_by_name(first):
    out = {}
    for s in load_set(first):
        with open(s.path, "rb") as f:
            for t in s.tensors:
                f.seek(s.start + t.off)
                out[t.name] = (t.ty, tuple(t.dims), f.read(t.nb))
    return out


def _kvs_all(first):
    return [s.kvs for s in load_set(first)]


def _checks():
    """Every self-test clause, printed; True when all hold."""
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(f"  {'ok  ' if cond else 'FAIL'} {name}" + ("" if cond else f": {detail}"))
        ok &= bool(cond)

    def refused(name, fn, code, frag):
        try:
            fn()
            check(name, False, "no refusal")
        except Refusal as e:
            check(name, e.code == code and frag in str(e), f"rc {e.code}: {e}")

    def cli(*argv):
        return main([str(x) for x in argv])

    sys.dont_write_bytecode = True
    gg = gguf_py("auto")
    gmode = "auto" if gg is not None else "off"
    with tempfile.TemporaryDirectory() as d:
        A = _fixture(d)
        C = _indexer_file(d)
        IDX = "blk.*.indexer.*"
        names_a = {t.name for s in load_set(A) for t in s.tensors}
        # strip
        B = os.path.join(d, "b", "m-00001-of-00003.gguf")
        os.makedirs(os.path.dirname(B))
        check("strip exits 0", cli("strip", "--in", A, "--out", B, "--drop", IDX, "--gguf-py", gmode) == 0)
        tb, ta = _tensors_by_name(B), _tensors_by_name(A)
        check("strip drops exactly the indexer tensors and keeps the others' bytes", set(tb) == {n for n in names_a if ".indexer." not in n}
              and all(tb[n] == ta[n] for n in tb), sorted(tb))
        ka, kb = _kvs_all(A), _kvs_all(B)
        total = lambda kvs: struct.unpack("<i", dict((k, v) for k, _, v in kvs).get(b"split.tensors.count", b"\xff\xff\xff\xff"))[0]
        check("strip keeps every key but split.tensors.count, which is the new total in every shard",
              all([x for x in kb[i] if x[0] != b"split.tensors.count"] == [x for x in ka[i] if x[0] != b"split.tensors.count"]
                  and total(kb[i]) == 4 for i in range(3)), [total(k) for k in kb])
        check("strip leaves the shard count and the tensor-to-shard map",
              [len(s.tensors) for s in load_set(B)] == [0, 2, 2] and [t.name for t in load_set(B)[1].tensors] == ["token_embd.weight", "blk.0.attn_q.weight"])
        # add: the round trip
        D = os.path.join(d, "d", "m-00001-of-00003.gguf")
        os.makedirs(os.path.dirname(D))
        check("add exits 0", cli("add", "--in", B, "--from", C, "--take", IDX, "--out", D, "--gguf-py", gmode) == 0)
        td, kd = _tensors_by_name(D), _kvs_all(D)
        check("strip then add: every tensor of A is in D with its type, shape and bytes", td == ta, sorted(set(td) ^ set(ta)))
        check("strip then add: every key of every shard is A's, byte for byte", kd == ka, [total(k) for k in kd])
        check("the added tensors are in the last shard, after its own", [t.name for t in load_set(D)[2].tensors][:2] == ["blk.7.ffn_up_exps.weight", "output.weight"]
              and len(load_set(D)[2].tensors) == 2 + 4 and len(load_set(D)[1].tensors) == 2)
        # --arch, --kv
        E = os.path.join(d, "e", "m-00001-of-00003.gguf")
        os.makedirs(os.path.dirname(E))
        check("add --arch --kv exits 0", cli("add", "--in", B, "--from", C, "--take", IDX, "--out", E, "--arch", "kimi-k31",
                                            "--kv", "kimi-k31.attention.indexer.top_k=u32:2048",
                                            "--kv", "kimi-k31.attention.indexer.layers=u32[]:3,7,11", "--kv", "bloomery.note=str:a:b",
                                            "--gguf-py", gmode) == 0)
        me = R.header(E)[0]
        check("--arch renames general.architecture and every old key, only those",
              me.get("general.architecture") == b"kimi-k31" and me.get("kimi-k31.block_count") == 6 and me.get("kimi-k31.rope.scale") == 0.5
              and not [k for k in me if k.startswith("kimi-k3.")] and "tokns" in me and me["general.alignment"] == 32, sorted(me))
        check("--kv appends a number, an array and a string with a colon in it to the first shard",
              me.get("kimi-k31.attention.indexer.top_k") == 2048 and me.get("bloomery.note") == b"a:b"
              and kv_get(Shard(E).kvs, b"kimi-k31.attention.indexer.layers")[1] == struct.pack("<IQ", 4, 3) + struct.pack("<3I", 3, 7, 11))
        check("the tensors are untouched by the rename", _tensors_by_name(E) == ta)
        # --link
        L = os.path.join(d, "l", "m-00001-of-00003.gguf")
        os.makedirs(os.path.dirname(L))
        check("--link add exits 0", cli("add", "--in", B, "--from", C, "--take", IDX, "--out", L, "--link", "--gguf-py", gmode) == 0)
        ino = lambda p: os.stat(p).st_ino
        sb = [os.path.join(d, "b", f"m-0000{i}-of-00003.gguf") for i in (1, 2, 3)]
        sl = [os.path.join(d, "l", f"m-0000{i}-of-00003.gguf") for i in (1, 2, 3)]
        check("--link: the untouched shard is the input's inode, the first and the receiving shard are written",
              ino(sl[1]) == ino(sb[1]) and ino(sl[0]) != ino(sb[0]) and ino(sl[2]) != ino(sb[2]), [ino(p) for p in sl])
        check("--link: the set reads as D's tensors, and the first shard carries the total", _tensors_by_name(L) == ta and total(_kvs_all(L)[0]) == 8)
        # --consume (strip), killed after an output was renamed and before its input was deleted, then resumed by the same command
        K1 = os.path.join(d, "k1")
        os.makedirs(K1)
        Ac = _fixture(d, "ac")
        Kc = os.path.join(K1, "m-00001-of-00003.gguf")
        real_unlink = os.unlink

        def dying_unlink(p, *a, **k):
            if os.path.basename(p) == "m-00003-of-00003.gguf" and os.path.dirname(p).endswith("ac"):
                raise OSError("simulated kill between the rename of shard 3's output and the delete of its input")
            return real_unlink(p, *a, **k)

        os.unlink = dying_unlink
        try:
            cli("strip", "--in", Ac, "--out", Kc, "--drop", IDX, "--consume", "--gguf-py", gmode)
            check("--consume: a kill after an output is renamed stops the run", False, "no kill")
        except OSError:
            check("--consume: a kill after an output is renamed stops the run", True)
        finally:
            os.unlink = real_unlink
        left = sorted(os.listdir(os.path.join(d, "ac")))
        check("--consume: the killed run kept its two finished outputs, deleted the finished input and kept the others",
              sorted(os.listdir(K1)) == ["m-00002-of-00003.gguf", "m-00003-of-00003.gguf"] and left == ["m-00001-of-00003.gguf", "m-00003-of-00003.gguf"], (os.listdir(K1), left))
        check("--consume: the same command resumes (shard 2 read from its output, shard 3's input deleted after its header check, shard 1 written) and leaves no input",
              cli("strip", "--in", Ac, "--out", Kc, "--drop", IDX, "--consume", "--gguf-py", gmode) == 0 and os.listdir(os.path.join(d, "ac")) == [], os.listdir(os.path.join(d, "ac")))
        check("--consume: the output set equals the plain strip's, keys and tensors", _tensors_by_name(Kc) == tb and [x for x in _kvs_all(Kc)] == kb)
        check("--consume: the command again, with every input gone, says the run is complete", cli("strip", "--in", Ac, "--out", Kc, "--drop", IDX, "--consume") == 0)
        # an output that is not the one this run would write is refused, not trusted
        K2 = os.path.join(d, "k2")
        os.makedirs(K2)
        Ad = _fixture(d, "ad")
        with open(os.path.join(K2, "m-00002-of-00003.gguf"), "wb") as f:
            f.write(b"not a gguf file")
        check("--consume: a foreign file in the output's place is refused and no input is deleted",
              cli("strip", "--in", Ad, "--out", os.path.join(K2, "m-00001-of-00003.gguf"), "--drop", IDX, "--consume") != 0 and len(os.listdir(os.path.join(d, "ad"))) == 3)
        refused("--consume on an add is refused", lambda: cmd_transform(argparse.Namespace(inp=B, src=C, out=os.path.join(d, "q", "m-00001-of-00003.gguf"), drop=None, take=[IDX], arch=None,
                kv=None, kv_replace=False, into=None, link=False, consume=True, verify="data", gguf_py="off", no_space_check=False, dry_run=False), False), 64, "--consume is for strip")
        # single file
        one = os.path.join(d, "one.gguf")
        R.write_gguf(one, [R._kv_string("general.architecture", "kimi-k3"), R._kv_scalar("kimi-k3.block_count", 4, 6)],
                     [_t("blk.0.attn_q.weight", 8, [64, 3]), _t("blk.3.indexer.proj.weight", 30, [32, 4])], trailer=0)
        one_b, one_d = os.path.join(d, "one-b.gguf"), os.path.join(d, "one-d.gguf")
        check("a single file strips and adds back (round trip, keys byte-equal)",
              cli("strip", "--in", one, "--out", one_b, "--drop", IDX, "--gguf-py", gmode) == 0
              and cli("add", "--in", one_b, "--from", C, "--take", "blk.3.indexer.proj.weight", "--out", one_d, "--gguf-py", gmode) == 0
              and _tensors_by_name(one_d) == _tensors_by_name(one) and Shard(one_d).kvs == Shard(one).kvs)
        # refusals, each by name
        refused("a --drop that matches nothing is refused", lambda: cmd_transform(argparse.Namespace(inp=A, src=None, out=os.path.join(d, "r1", "m-00001-of-00003.gguf"), drop=["blk.*.nope.*"], take=None, arch=None, kv=None, kv_replace=False, into=None,
                                                                    link=False, consume=False, verify="data", gguf_py="off", no_space_check=False, dry_run=False), True),
                65, "--drop 'blk.*.nope.*' matches no tensor")
        refused("a --take that matches nothing is refused", lambda: plan(load_set(B), [], load_set(C), ["zzz*"], None, [], False, None, False), 65, "--take 'zzz*' matches no tensor")
        refused("a take of a name the set already has is refused (collision)", lambda: plan(load_set(A), [], load_set(C), [IDX], None, [], False, None, False), 65, "already in")
        refused("a strip of every tensor is refused", lambda: plan(load_set(A), ["*"], None, [], None, [], False, None, False), 65, "nothing would be left")
        refused("an existing output is refused", lambda: run(*plan(load_set(A), [IDX], None, [], None, [], False, None, False)[:1], load_set(A), os.path.join(d, "b", "m-00001-of-00003.gguf"), False, "data", None, True), 73, "exists")
        refused("an output named as one of the input's shards is refused", lambda: run(*plan(load_set(A), [IDX], None, [], None, [], False, None, False)[:1], load_set(A), A, False, "data", None, True), 64, "never over it")
        refused("an output name that does not carry the shard count is refused", lambda: out_paths(os.path.join(d, "x.gguf"), 3), 64, "-00001-of-00003.gguf")
        refused("--into outside the set is refused", lambda: plan(load_set(B), [], load_set(C), [IDX], None, [], False, 7, False), 64, "--into 8")
        refused("--arch of the name the file has is refused", lambda: plan(load_set(B), [], load_set(C), [IDX], "kimi-k3", [], False, None, False), 64, "already has")
        refused("--arch on a file with no architecture is refused", lambda: plan([_noarch(d)], [], load_set(C), [IDX], "x", [], False, None, False), 65, "no shard has a general.architecture")
        refused("a --kv that disagrees with the file is refused", lambda: apply_kv(Shard(E).kvs, [parse_kv("kimi-k31.attention.indexer.top_k=u32:1")], False, "E"), 65, "another value")
        check("a --kv with the same value is a no-op and --kv-replace replaces",
              apply_kv(Shard(E).kvs, [parse_kv("kimi-k31.attention.indexer.top_k=u32:2048")], False, "E")[0] == Shard(E).kvs
              and kv_get(apply_kv(Shard(E).kvs, [parse_kv("kimi-k31.attention.indexer.top_k=u32:1")], True, "E")[0], b"kimi-k31.attention.indexer.top_k")[1] == struct.pack("<I", 1))
        refused("a malformed --kv is refused", lambda: parse_kv("novalue"), 64, "KEY=TYPE:VALUE")
        refused("a --kv of an unknown type is refused", lambda: parse_kv("k=u99:1"), 64, "not one of")
        refused("a --kv number out of its type's range is refused", lambda: parse_kv("k=u8:300"), 64, "not a u8")
        # a tensor type the table does not know is refused when the file is read
        bad = os.path.join(d, "badtype.gguf")
        R.write_gguf(bad, [R._kv_string("general.architecture", "x")], [("w.weight", 0, [4], b"\0" * 16)])
        with open(bad, "rb") as f:
            raw = f.read()
        i = raw.index(b"w.weight") + 8 + 4 + 8  # name, n_dims, one dim -> the type field
        with open(bad, "wb") as f:
            f.write(raw[:i] + struct.pack("<I", 99) + raw[i + 4:])
        refused("a tensor type outside the size table is refused by name", lambda: Shard(bad), 65, "type id 99")
        # a layout mainline would refuse (a hole between two tensors)
        hole = os.path.join(d, "hole.gguf")
        R.write_gguf(hole, [R._kv_string("general.architecture", "x")], [("a.weight", 0, [8], b"\1" * 32), ("b.weight", 0, [8], b"\2" * 32)])
        with open(hole, "rb") as f:
            raw = f.read()
        j = raw.index(b"b.weight") + 8 + 4 + 8 + 4  # its offset field
        with open(hole, "wb") as f:
            f.write(raw[:j] + struct.pack("<Q", 64) + raw[j + 8:] + b"\0" * 32)
        refused("a hole between tensors (mainline's reader refuses it) is refused", lambda: Shard(hole), 65, "mainline's reader requires")
        # verification FAIL-first: a flipped data byte, a dropped key, a changed tensor-info byte
        shards_b = load_set(B)
        pl_b, _ = plan(shards_b, [], load_set(C), [IDX], None, [], False, None, False)
        copy_shard(pl_b[2], os.path.join(d, "v.gguf"))
        verify_shard(os.path.join(d, "v.gguf"), pl_b[2], "data", gg)
        for what, mut, frag in (
            ("a flipped data byte", lambda b, st: b[:-40] + bytes([b[-40] ^ 1]) + b[-39:], "sha256"),
            ("a changed tensor name byte", lambda b, st: b.replace(b"blk.7.ffn_up_exps", b"blk.7.ffn_up_expz", 1), "tensor names, shapes, types or offsets"),
            ("a shortened file", lambda b, st: b[:-32], "bytes"),
        ):
            with open(os.path.join(d, "v.gguf"), "rb") as f:
                raw = f.read()
            with open(os.path.join(d, "w.gguf"), "wb") as f:
                f.write(mut(raw, 0))
            try:
                verify_shard(os.path.join(d, "w.gguf"), pl_b[2], "data", gg)
                check(f"verification refuses {what}", False, "no refusal")
            except Refusal as e:
                check(f"verification refuses {what}", e.code == 65 and (frag in str(e) or "tile" in str(e) or "fill" in str(e) or "ends" in str(e)), str(e))
        check("a refused run leaves no output, no temporary file", not [f for f in os.listdir(os.path.join(d)) if ".tmp." in f] and not os.path.exists(os.path.join(d, "r1")))
        # check-shape
        tsv = os.path.join(d, "shape.tsv")
        with open(tsv, "w") as f:
            f.write("\n".join(tsv_lines(load_set(A))) + "\n")
        check("check-shape passes the set its own table", cli("check-shape", "--in", A, "--names", tsv) == 0)
        refused("check-shape refuses a set with a tensor the table lacks, by name", lambda: cmd_check_shape(argparse.Namespace(inp=B, names=tsv, types=False)), 65, "not the table's shape")
        # check-shape --types
        refused("check-shape --types refuses a tensor whose type is not the table's", lambda: cmd_check_shape(argparse.Namespace(inp=A, names=_retyped(d, tsv), types=True)), 65, "shape and types")
        check("check-shape --types passes the set its own table", cli("check-shape", "--in", A, "--names", tsv, "--types") == 0)
        # compare
        Df = os.path.join(d, "f")
        shutil.copytree(os.path.join(d, "d"), Df)
        s3 = load_set(os.path.join(Df, "m-00001-of-00003.gguf"))[2]
        with open(s3.path, "r+b") as f:
            f.seek(s3.start + s3.tensors[0].off + 3)
            byte = f.read(1)
            f.seek(s3.start + s3.tensors[0].off + 3)
            f.write(bytes([byte[0] ^ 1]))
        Dc = os.path.join(Df, "m-00001-of-00003.gguf")
        check("compare: a strip then add equals the original, every tensor by sha256 and every key but split.*", cli("compare", "--a", A, "--b", D) == 0)
        check("compare refuses a copy with one flipped tensor byte", cli("compare", "--a", A, "--b", Dc) == 65)
        check("compare --no-data compares headers only (the flipped copy passes: what the flag is)", cli("compare", "--a", A, "--b", Dc, "--no-data") == 0)
        check("compare refuses sets with other tensors", cli("compare", "--a", A, "--b", B) == 65)
        check("compare refuses a set whose keys were renamed, and --tensors-only compares the tensors alone", cli("compare", "--a", A, "--b", E) == 65
              and cli("compare", "--a", A, "--b", E, "--tensors-only") == 0)
        # check-rules
        rt = os.path.join(d, "rules.tbl")
        with open(rt, "w") as f:
            f.write("\n".join(["blk.0.attn_q.weight\tQ6_K\t64x3", "blk.0.attn_norm.weight\tF32\t64", "blk.0.ffn_up_exps.weight\tIQ1_S\t256x2x3",
                               "blk.1.ffn_up_exps.weight\tIQ2_XXS\t256x2x3", "blk.0.ssm_beta.weight\tF32\t512x2",
                               "blk.0.ssm_conv1d_q.weight\tF32\t4x1x256", "blk.0.ffn_gate_inp.weight\tF32\t512x8", "token_embd.weight\tQ6_K\t512x100"]) + "\n")
        good = ["blk\\.1\\.ffn_up_exps\\.weight=iq2_xxs", "ffn_(down|gate|up)_exps\\.weight=iq1_s", "ssm_beta\\.weight=f32", "\\.attn_q\\.weight$=Q6_K", "^token_embd\\.weight$=q6_k"]

        def rules_file(name, lines):
            path = os.path.join(d, name)
            with open(path, "w") as f:
                f.write("\n".join(lines) + "\n")
            return path

        check("check-rules passes the rules that give the table's types", cli("check-rules", "--rules", rules_file("r.ok", good), "--table", rt) == 0)
        refused("check-rules refuses a quantizable tensor no rule matches (it would get the global type)", lambda: cmd_check_rules(argparse.Namespace(rules=rules_file("r.nobeta", good[:2] + good[3:]), table=rt, inp=None)), 65, "problem")
        refused("check-rules refuses a rule order that lets a general rule shadow a layer's (the first match wins)", lambda: cmd_check_rules(argparse.Namespace(rules=rules_file("r.order", [good[1], good[0]] + good[2:]), table=rt, inp=None)), 65, "problem")
        refused("check-rules refuses a rule that matches no tensor", lambda: cmd_check_rules(argparse.Namespace(rules=rules_file("r.extra", good + ["nothing_like_this=q8_0"]), table=rt, inp=None)), 65, "problem")
        check("check-rules --partial passes a rule set that leaves the trunk to the global type", cmd_check_rules(argparse.Namespace(rules=rules_file("r.part", good[1:3]), table=rt, inp=None, partial=True)) == 0)
        refused("check-rules --partial still refuses a rule that matches no tensor", lambda: cmd_check_rules(argparse.Namespace(rules=rules_file("r.part2", good[1:3] + ["nothing_like_this=q8_0"]), table=rt, inp=None, partial=True)), 65, "problem")
        # --retype and --flag-types: the tight map is the table's Q6_K trunk as Q4_K, with the embedding typed by a flag
        def cli_rc(fn):  # a refusal is a red clause, not a crash, in a mutant run
            try:
                return fn()
            except Refusal as e:
                return e.code

        tight = good[:3] + ["\\.attn_q\\.weight$=q4_k"]
        ns = lambda name, lines, **kw: argparse.Namespace(rules=rules_file(name, lines), table=rt, inp=None, **kw)  # noqa: E731
        check("check-rules --retype --flag-types passes the tight rules (trunk Q6_K read as Q4_K, the embedding typed by a flag)",
              cli_rc(lambda: cmd_check_rules(ns("r.t1", tight, retype="Q6_K=Q4_K", flag_types="token_embd.weight=q4_k"))) == 0)
        refused("check-rules --retype refuses a rule that still gives the table's own type", lambda: cmd_check_rules(ns("r.t2", good[:4], retype="Q6_K=Q4_K", flag_types="token_embd.weight=q4_k")), 65, "problem")
        refused("check-rules without --flag-types refuses the embedding no rule matches", lambda: cmd_check_rules(ns("r.t3", tight, retype="Q6_K=Q4_K")), 65, "problem")
        refused("check-rules refuses a tensor a flag types and a rule matches too", lambda: cmd_check_rules(ns("r.t4", tight + ["^token_embd\\.weight$=q4_k"], retype="Q6_K=Q4_K", flag_types="token_embd.weight=q4_k")), 65, "problem")
        refused("check-rules --flag-types refuses a name the table does not have", lambda: cmd_check_rules(ns("r.t5", tight, retype="Q6_K=Q4_K", flag_types="token_embd.weight=q4_k,output.weight=q6_k")), 65, "problem")
        refused("check-rules --retype refuses a type name llama-quantize does not know", lambda: cmd_check_rules(ns("r.t6", tight, retype="Q6_K=Q9_9")), 64, "NAME=type")
        refused("check-rules refuses a type llama-quantize does not know", lambda: parse_rules(rules_file("r.badtype", ["a=q9_9"])), 65, "knows no ggml type")
        refused("check-rules refuses a regex that does not compile", lambda: parse_rules(rules_file("r.badrx", ["(a=q8_0"])), 65, "does not compile")
        with open(rt, "a") as f:
            f.write("blk.0.ssm_a_norm.weight\tQ6_K\t64x64\n")
        refused("check-rules refuses a tensor the quantizer leaves alone whose table type is not F32", lambda: cmd_check_rules(argparse.Namespace(rules=rules_file("r.ok2", good), table=rt, inp=None)), 65, "problem")
        tin = os.path.join(d, "in.tbl")
        with open(tin, "w") as f:
            f.write("blk.0.attn_q.weight\tQ6_K\t64x3\nblk.9.attn_q.weight\tQ6_K\t64x3\n")
        refused("check-rules --in refuses a table tensor the input lacks", lambda: cmd_check_rules(argparse.Namespace(rules=rules_file("r.q", [good[3]]), table=tin, inp=one_b)), 65, "problem")
        # list
        check("list reads a set and the tsv form equals the table's", cli("list", "--in", A, "--gguf-py", gmode) == 0 and cli("list", "--in", A, "--tsv") == 0)
        # the command line
        check("the command line refuses a missing --drop and a missing --take", cli("strip", "--in", A, "--out", os.path.join(d, "n1")) == 64
              and cli("add", "--in", B, "--from", C, "--out", os.path.join(d, "n2")) == 64)
        check("--dry-run writes nothing", cli("strip", "--in", A, "--out", os.path.join(d, "dr", "m-00001-of-00003.gguf"), "--drop", IDX, "--dry-run") == 0 and not os.path.exists(os.path.join(d, "dr")))
        # plan, classes, fill (the tight map's tooling: llama-quantize's own --dry-run plan, the sizes a placement is sized by, the card)
        log = os.path.join(d, "dry.log")
        with open(log, "w") as f:
            f.write("llama_quantize: calculating quantization size for 'x' as Q4_K_M using 1 threads\n"
                    "llama_tensor_get_type: blk.0.ssm_beta.weight - applying manual override: q4_K -> f32\n"
                    "[   1/6] output.weight                        - [  512,  1000,     1,     1], type =   q6_K, size =    0.977 MiB\n"
                    "[   2/6] token_embd.weight                    - [  512,  1000,     1,     1], type =   bf16, size =     0.98 MiB ->     0.28 MiB (q4_K)\n"
                    "[   3/6] blk.0.ffn_up_exps.weight             - [  256,   128,     8,     1], type =  mxfp4, size =     0.12 MiB ->     0.04 MiB (iq1_s)\n"
                    "[   4/6] blk.0.attn_norm.weight               - [  512,     1,     1,     1], type =    f32, size =    0.002 MiB\n"
                    "[   5/6] blk.0.attn_q.weight                  - [  512,   256,     1,     1], type =   bf16, size =     0.25 MiB ->     0.07 MiB (q4_K)\n"
                    "[   6/6] blk.3.indexer.attn_k.weight          - [  512,     8,     1,     1], type =   bf16, size =     0.01 MiB\n"
                    "llama_model_quantize_impl: quant size  = 1.4 MiB\n")
        ptsv = os.path.join(d, "plan.tsv")
        check("plan exits 0", cli("plan", "--log", log, "--out", ptsv) == 0)
        with open(ptsv) as f:
            prow = [ln.rstrip("\n").split("\t") for ln in f]
        check("plan: the type is the one after the arrow, an untouched tensor keeps the source's, dims lose their padding 1s",
              [(r[0], r[1], r[2]) for r in prow] == [("output.weight", "Q6_K", "512x1000"), ("token_embd.weight", "Q4_K", "512x1000"),
                                                     ("blk.0.ffn_up_exps.weight", "IQ1_S", "256x128x8"), ("blk.0.attn_norm.weight", "F32", "512"),
                                                     ("blk.0.attn_q.weight", "Q4_K", "512x256"), ("blk.3.indexer.attn_k.weight", "BF16", "512x8")], prow)
        check("plan: the bytes are the log's MiB", [int(r[3]) for r in prow][:2] == [round(0.977 * 1048576), round(0.28 * 1048576)])
        with open(log + "2", "w") as f:
            f.write("[   1/3] a.weight - [  4,  4,     1,     1], type =    f32, size =    0.001 MiB\n")
        refused("plan refuses a log with fewer tensor lines than its own total", lambda: cmd_plan(argparse.Namespace(log=log + "2", out=os.path.join(d, "p2.tsv"))), 65, "the log says 3")
        with open(log + "3", "w") as f:
            f.write("[   1/1] a.weight - [  4,  4,     1,     1], type =    q9_x, size =    0.001 MiB\n")
        refused("plan refuses a type it does not know", lambda: cmd_plan(argparse.Namespace(log=log + "3", out=os.path.join(d, "p3.tsv"))), 65, "none llama-quantize")
        refused("plan refuses a log with no tensor line", lambda: plan_rows("nothing here\n", "x"), 65, "no tensor line")
        csum = class_sums([(r[0], r[1], int(r[3])) for r in prow])
        check("classes: experts, embedding, head, indexer, small f32 and the trunk, each with its bytes",
              [csum[c][0] for c in CLASSES] == [1, 1, 1, 1, 1, 1] and csum["trunk"][2] == {"Q4_K": 1} and csum["small f32"][2] == {"F32": 1}
              and csum["experts"][1] == round(0.04 * 1048576), csum)
        check("classes --field prints one class's bytes and --md a markdown table", cli("classes", "--tsv", ptsv, "--field", "trunk") == 0 and cli("classes", "--tsv", ptsv, "--md") == 0)
        check("classes --in reads a real set", cli("classes", "--in", A) == 0)
        refused("classes refuses both inputs and neither", lambda: cmd_classes(argparse.Namespace(tsv=ptsv, inp=A, md=False, field=None)), 64, "one of them")
        check("check-shape treats 7168x1 and 7168 as one shape", norm_dims("7168x1") == norm_dims("7168") == "7168" and norm_dims("4x1x256") == "4x1x256")
        tpl = os.path.join(d, "t.md.in")
        with open(tpl, "w") as f:
            f.write("---\nname: @NAME@\n---\n@BODY@ and @NAME@ again\n")
        body = os.path.join(d, "body.txt")
        with open(body, "w") as f:
            f.write("line one\nline two\n")
        card = os.path.join(d, "card.md")
        check("fill replaces every placeholder, a --file value by its text", cli("fill", "--template", tpl, "--out", card, "--set", "NAME=kimi", "--file", f"BODY={body}") == 0
              and open(card).read() == "---\nname: kimi\n---\nline one\nline two and kimi again\n")
        refused("fill refuses a placeholder no value was given for", lambda: cmd_fill(argparse.Namespace(template=tpl, out=os.path.join(d, "c2.md"), set=["NAME=x"], file=None)), 65, "@BODY@")
        refused("fill refuses a value the template has no placeholder for", lambda: cmd_fill(argparse.Namespace(template=tpl, out=os.path.join(d, "c3.md"), set=["NAME=x", "BODY=y", "EXTRA=z"], file=None)), 65, "EXTRA")
        refused("fill refuses an output that exists", lambda: cmd_fill(argparse.Namespace(template=tpl, out=card, set=["NAME=x", "BODY=y"], file=None)), 73, "exists")
        print(f"  gguf-py: {'cross-checked every written shard' if gg is not None else 'not present here: the cross-check leg did not run'}")
    return ok


def _corrupt_copy(real):
    def copy(planned, tmp):
        real(planned, tmp)
        if planned.tensors:  # one byte inside the last tensor's data
            with open(tmp, "r+b") as f:
                f.seek(len(planned.header) + planned.data - 40)
                b = f.read(1)
                f.seek(len(planned.header) + planned.data - 40)
                f.write(bytes([b[0] ^ 1]))
    return copy


def _plan_source_type(real_rows):
    """plan_rows as a tool that reads the `type =` field and ignores the arrow would give it (the mutant of the plan clause)."""
    def rows(text, where):
        return [(n, plan_type(m.group(5), where), d, nb) for (n, _t, d, nb), m in
                zip(real_rows(text, where), (_PLAN_ROW.match(ln) for ln in text.splitlines() if _PLAN_ROW.match(ln)))]
    return rows


def self_test():
    """The clauses, then FAIL-first: each mutant of the tool (a patched function) must turn the same clauses red."""
    ok = _checks()
    real = {k: globals()[k] for k in ("with_split_total", "match_names", "verify_shard", "copy_shard", "renamed", "pad", "info_bytes",
                                      "first_rule", "quantizer_touches", "cmp_hash", "plan_rows", "fill_text")}
    mutants = [
        ("split.tensors.count left as read", {"with_split_total": lambda kvs, total, where: kvs}),
        ("a pattern that matches nothing accepted", {"match_names": lambda names, patterns, flag, where, allow_empty=False: real["match_names"](names, patterns, flag, where, True)}),
        ("the write-back verification off", {"verify_shard": lambda *a, **k: None}),
        ("a data byte corrupted in the copy, verification off", {"copy_shard": _corrupt_copy(real["copy_shard"]), "verify_shard": lambda *a, **k: None}),
        ("--arch renaming only general.architecture", {"renamed": lambda kvs, old, new, where: [(k, t, struct.pack("<Q", len(new)) + new.encode() if k == b"general.architecture" else v)
                                                                                                    for k, t, v in kvs]}),
        ("tensor offsets padded to 16 instead of the alignment, verification off", {"info_bytes": lambda tensors, align: real["info_bytes"](tensors, 16), "verify_shard": lambda *a, **k: None}),
        ("the last matching rule winning instead of the first", {"first_rule": lambda rules, name: next(((t, y) for _, t, y in reversed(rules) if re.search(
            next(r for r, tt, _ in rules if tt == t).pattern, name)), None)}),
        ("every tensor counted as quantizable", {"quantizer_touches": lambda name, dims: True}),
        ("tensor bytes not hashed by compare", {"cmp_hash": lambda path, off, n: "0"}),
        ("plan keeping the source type of a converted tensor", {"plan_rows": _plan_source_type(real["plan_rows"])}),
        ("fill leaving an unfilled placeholder in the output", {"fill_text": lambda text, values: _HOLE.sub(lambda m: values.get(m.group(0)[1:-1], m.group(0)), text)}),
    ]
    for name, patch in mutants:
        globals().update(patch)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
                held = _checks()
        except Exception as e:  # noqa: BLE001 - a crash is red too
            held, buf = False, io.StringIO(f"  FAIL crashed: {type(e).__name__}: {e}\n")
        finally:
            globals().update(real)
        reds = [ln.strip() for ln in buf.getvalue().splitlines() if ln.startswith("  FAIL")]
        print(f"  {'ok  ' if not held and reds else 'FAIL'} FAIL-first: with {name}, {len(reds)} clause(s) go red" + (f" (first: {reds[0][5:90]})" if reds else ""))
        ok &= not held and bool(reds)
    return ok


def _retyped(d, tsv):
    """A copy of table `tsv` whose first row has another type."""
    out = os.path.join(d, "retyped.tbl")
    with open(tsv) as f:
        rows = f.read().splitlines()
    first = rows[0].split("\t")
    first[1] = "F16" if first[1] != "F16" else "F32"
    with open(out, "w") as f:
        f.write("\n".join(["\t".join(first)] + rows[1:]) + "\n")
    return out


def _noarch(d):
    p = os.path.join(d, "noarch.gguf")
    R.write_gguf(p, [R._kv_scalar("general.alignment", 4, 32)], [_t("w.weight", 0, [4])])
    return Shard(p)


# ---- the command line ----

class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise Refusal(64, message)


def build_parser():
    p = _Parser(prog="gguf-splice.py", add_help=True)
    sub = p.add_subparsers(dest="cmd")

    def common(s):
        s.add_argument("--in", dest="inp", required=True, help="the input: a file, or a split set's first shard")
        s.add_argument("--gguf-py", choices=("auto", "required", "off"), default="auto")

    def writer(s):
        s.add_argument("--out", required=True, help="the output file, or the output set's first shard")
        s.add_argument("--link", action="store_true")
        s.add_argument("--consume", action="store_true")
        s.add_argument("--verify", choices=("data", "header"), default="data")
        s.add_argument("--no-space-check", action="store_true")
        s.add_argument("--dry-run", action="store_true", help="print the plan; write nothing")

    s = sub.add_parser("strip")
    common(s)
    writer(s)
    s.add_argument("--drop", action="append", required=True)
    s = sub.add_parser("add")
    common(s)
    writer(s)
    s.add_argument("--from", dest="src", required=True, help="where the tensors come from: a file, or a split set's first shard")
    s.add_argument("--take", action="append", required=True)
    s.add_argument("--arch")
    s.add_argument("--kv", action="append")
    s.add_argument("--kv-replace", action="store_true")
    s.add_argument("--into", type=int, help="the 1-based shard that receives the tensors (default: the last)")
    s = sub.add_parser("list")
    common(s)
    s.add_argument("--tsv", action="store_true")
    s = sub.add_parser("check-shape")
    s.add_argument("--in", dest="inp", required=True)
    s.add_argument("--names", required=True, help="a name TAB type TAB dims table (crates/model/tests/kimi_k3_tensors.tsv)")
    s.add_argument("--types", action="store_true", help="hold the types to the table's too")
    s = sub.add_parser("check-rules")
    s.add_argument("--rules", required=True, help="a llama-quantize --tensor-type-file")
    s.add_argument("--table", required=True, help="the reference table the rules must reproduce")
    s.add_argument("--in", dest="inp", help="a real input set whose kept tensors must already have the table's types")
    s.add_argument("--retype", help="FROM=TO,...: the table's FROM types are compared as TO (the tight map is the K3 table with its Q6_K trunk as Q4_K)")
    s.add_argument("--flag-types", help="NAME=type,...: tensors a flag types (token_embd.weight=q4_k): no rule may match them")
    s.add_argument("--partial", action="store_true", help="the rules need not cover every quantizable tensor or give the table's types (the global type does the rest); each must still match a tensor")
    s = sub.add_parser("compare")
    s.add_argument("--a", required=True)
    s.add_argument("--b", required=True)
    s.add_argument("--no-data", action="store_true", help="names, types, dims and keys only; no tensor bytes read")
    s.add_argument("--tensors-only", action="store_true", help="the tensors only: no key is compared (a renamed architecture word)")
    s = sub.add_parser("plan")
    s.add_argument("--log", required=True, help="the log of llama-quantize --dry-run")
    s.add_argument("--out", required=True)
    s = sub.add_parser("classes")
    s.add_argument("--tsv", help="a table `plan` wrote")
    s.add_argument("--in", dest="inp", help="a set's first shard: its real tensors")
    s.add_argument("--md", action="store_true")
    s.add_argument("--field", help="print this class's bytes only")
    s = sub.add_parser("fill")
    s.add_argument("--template", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--set", action="append")
    s.add_argument("--file", action="append")
    return p


def main(argv):
    try:
        if argv[:1] == ["--self-test"]:
            return 0 if self_test() else 1
        a = build_parser().parse_args(argv)
        if a.cmd is None:
            raise Refusal(64, "usage: gguf-splice.py strip|add|list|check-shape|check-rules|compare|plan|classes|fill ... | --self-test (see the file's docstring)")
        if a.cmd == "strip":
            a.take, a.src, a.arch, a.kv, a.kv_replace, a.into = None, None, None, None, False, None
            return cmd_transform(a, True)
        if a.cmd == "add":
            a.drop = None
            return cmd_transform(a, False)
        if a.cmd == "list":
            return cmd_list(a)
        if a.cmd == "check-rules":
            return cmd_check_rules(a)
        if a.cmd == "compare":
            return cmd_compare(a)
        if a.cmd == "plan":
            return cmd_plan(a)
        if a.cmd == "classes":
            return cmd_classes(a)
        if a.cmd == "fill":
            return cmd_fill(a)
        return cmd_check_shape(a)
    except Refusal as e:
        print(f"gguf-splice.py: {e}", file=sys.stderr)
        return e.code
    except SystemExit as e:
        return 64 if e.code else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
