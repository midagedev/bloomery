#!/usr/bin/env python3
"""A V4.1 GGUF set with its Engram in llama.cpp's layout, written from a set in the per-site layout.

    tools/ref/engram-relayout.py relayout <first shard> <out dir> [--table-type q6_K] [--kv-type q6_K] [--per-site]
    tools/ref/engram-relayout.py twin <fixture dir> [--per-site]
    tools/ref/engram-relayout.py check <twin first shard> --source <first shard>
    tools/ref/engram-relayout.py --self-test

The per-site layout (bloomery's fixture writer, ik's converter) keeps a table per Engram site,
`blk.L.engram_embd.weight` [width, rows], the gains `blk.L.engram_k.weight` and `blk.L.engram_q.weight`, the projection
`blk.L.engram_wkv.weight`, and the hash under `deepseek41.engram.*`, each site's bucket offsets counted from its own
table's row 0. llama.cpp's converter (conversion/deepseek41.py) writes the same model as:
  - one `per_layer_token_embd.weight` [width, sum of the sites' rows]: every site's rows back to back, in layer order;
  - `blk.L.engram_gate.weight`, F32: the two gains folded, q * k, each product rounded to f32 once;
  - `blk.L.engram_kv.weight`: engram_wkv under that name;
  - the hash under `deepseek41.ple.*` (layers as u32, ngram_size, heads_per_ngram, layer_multipliers, head_vocab_sizes,
    and head_offsets counted from the one table's row 0), `deepseek41.embedding_length_per_layer_input` (the key
    length), `deepseek41.engram.pad_token_id` and `deepseek41.engram.token_map`.

`relayout` writes that layout of the set the first shard opens into <out dir>, which must not exist (73): one file per
source shard, `<stem>-engram-upstream-NNNNN-of-MMMMM.gguf`. The concatenated table takes the first site's place in its
shard, each gate the place of its site's engram_k, each kv its engram_wkv's; the other per-site tensors go, every other
tensor and key is carried byte for byte, and `split.tensors.count` is the new count in every shard. The carrier shard
gains `bloomery.relayout.source` (the source's first shard, symlinks resolved), `bloomery.relayout.source_header_sha256`
(the sha256 of that shard's header, every byte before its data) and `bloomery.relayout.types` (`tables=<type> kv=<type>`).
The gains are read at F32, F16, BF16 or Q3_K (ggml's dequantization; every value is exact in f32), the gate is
q * k in f32. The rows keep their type and bits unless `--table-type q6_K` or `--kv-type q6_K` asks for Q6_K: a Q3_K
block is then written as the Q6_K block with the same d, the scales sc - 32 and the codes v + 32. A Q3_K value is
(d * (sc - 32)) * v with d an f16, sc 6 bits and v in -4..3, the Q6_K value (d * s) * (code - 32): the same factors, and
each product exact in f32 (at most 11 + 6 + 3 significant bits), so every reader decodes the same f32 values from
either block. A row of any other type with Q6_K asked is refused by name: that would be a requantization, which this
tool does not do. Before <out dir> appears, the copy is held to the source: every carried tensor's bytes (sha256 of
both ranges), every site's rows at their base row of the one table, the gates recomputed, a sample of every transcoded
tensor's blocks decoded on both sides, the keys, and the shards' sizes against their headers.

`relayout --per-site` writes the copy's per-site counterpart instead: the set stays in the per-site layout (its tensors,
names and keys are the source's), and carries the numbers the llama.cpp layout's copy carries, so a reader of either
layout computes the same bits from the two. Each site's table and wkv are written at the type the copy's are (the same
blocks), `engram_q` is the gate (the same bytes, from the one fold) and `engram_k` is F32 ones: a key gain of ones is
what a reader of the folded layout applies (`v * 1` is `v`). The shards are `<stem>-engram-per-site-NNNNN-of-MMMMM.gguf`,
the record's types end in ` gains=ones,gate`, and the hold is the copy's, each site's table at its own name.

`twin <fixture dir>` prints where a fixture's upstream twin lives: `<parent of the fixture root>/fixture-variants/
engram-upstream/<fixture dir's name>`, beside the variants under a name that is no tag of tools/fixture-variants.tsv
(a tag is lower-case letters and digits), so no BLOOMERY_FIXTURE_VARIANT selects it. With `--per-site`, where its
per-site counterpart lives: the twin's directory's `per-site`, which holds no tag either.

`check <twin first shard> --source <first shard>` holds a twin to the source it was made from: the recorded source is
that file, and its header's sha256 is the recorded one; otherwise 65, naming the stale record.

Exit status: 0; 2 a file missing or unreadable; 64 a usage error; 65 a malformed source, one already in llama.cpp's
layout, a set this tool does not place, or a copy that differs from its source; 73 an existing <out dir>.
"""
import hashlib
import importlib.util
import os
import random
import re
import shutil
import struct
import sys
import tempfile

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("gguf_respell", os.path.join(_here, "gguf-respell.py"))
W = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(W)  # raw_header and the header packer; W.R is gguf-ranges.py, the one header parser
R = W.R
Refusal = R.Refusal

ARCH = "deepseek41"
CONCAT = "per_layer_token_embd.weight"
F32, F16, Q3_K, Q6_K, BF16 = 0, 1, 11, 14, 30
TYPE_NAMES = {F32: "F32", F16: "F16", Q3_K: "Q3_K", Q6_K: "Q6_K", BF16: "BF16"}
Q3_BYTES, Q6_BYTES, QK = 110, 210, 256
CHUNK = 16 << 20
SAMPLE = 64  # transcoded blocks decoded on both sides, per transcoded range
# The per-site keys this tool reads (the canonical spelling), each to its llama.cpp spelling, in the converter's order.
# `num_embeddings` is the sites' row counts, which the tables' shapes carry: checked and dropped.
KEYS = (
    ("engram.layer_ids", "ple.layers"),
    ("engram.max_ngram_size", "ple.ngram_size"),
    ("engram.head_count", "ple.heads_per_ngram"),
    ("engram.key_length", "embedding_length_per_layer_input"),
    ("engram.multipliers", "ple.layer_multipliers"),
    ("engram.offsets", "ple.head_offsets"),
    ("engram.primes", "ple.head_vocab_sizes"),
    ("engram.pad_id", "engram.pad_token_id"),
    ("engram.token_map", "engram.token_map"),
)
DROPPED = ("engram.num_embeddings",)
ROLES = ("engram_embd", "engram_k", "engram_q", "engram_wkv")
SITE_TENSOR = re.compile(r"blk\.(\d+)\.(engram_[a-z_]+)\.weight$")
RECORD = ("bloomery.relayout.source", "bloomery.relayout.source_header_sha256", "bloomery.relayout.types")
TWIN = "engram-upstream"
PER_SITE = "engram-per-site"
PER_SITE_DIR = "per-site"
ONE = struct.pack("<f", 1.0)


def full(suffix):
    return f"{ARCH}.{suffix}"


# ---- values ----

def _array(path, key, raw):
    """(element type, values) of an array value of scalar elements."""
    et, cnt = struct.unpack_from("<IQ", raw, 0)
    if et not in R.SCALAR:
        raise Refusal(65, f"{path}: {key} is an array of type {et}, not of numbers")
    fmt = R.SCALAR[et][1:]
    return et, list(struct.unpack_from("<" + fmt * cnt, raw, 12))


def _pack_array(et, values):
    return struct.pack("<IQ", et, len(values)) + struct.pack("<" + R.SCALAR[et][1:] * len(values), *values)


def _scalar(path, key, t, raw):
    if t not in R.SCALAR or t in (6, 12, 7):
        raise Refusal(65, f"{path}: {key} has value type {t}, not an integer")
    return struct.unpack(R.SCALAR[t], raw)[0]


def _f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _f16(b):
    return struct.unpack("<e", bytes(b))[0]


def q3k_scales(s12):
    """The 16 sub-block scales of a Q3_K block, sc - 32 (ggml's kmask unpacking of the 12 bytes)."""
    out = []
    for j in range(16):
        low = s12[j] & 15 if j < 8 else s12[j - 8] >> 4
        high = (s12[8 + (j & 3)] >> (2 * (j >> 2))) & 3
        out.append((low | high << 4) - 32)
    return out


def q3k_values(blk):
    """ggml's dequantize_row_q3_K of one block: value i = (d * (sc - 32)) * (q - 4 * !h), each product an f32."""
    d, sc, out = _f16(blk[108:110]), q3k_scales(blk[96:108]), []
    for i in range(QK):
        n, j, lane = i >> 7, (i >> 5) & 3, i & 31
        q = (blk[32 + 32 * n + lane] >> (2 * j)) & 3
        h = (blk[lane] >> (4 * n + j)) & 1
        out.append(_f32(_f32(d * sc[i >> 4]) * (q - (0 if h else 4))))
    return out


def q6k_values(blk):
    """ggml's dequantize_row_q6_K of one block: value i = (d * sc) * (code - 32), each product an f32."""
    d, out = _f16(blk[208:210]), []
    for i in range(QK):
        h, q, lane = i >> 7, (i >> 5) & 3, i & 31
        ql, qh = blk[64 * h + 32 * (q & 1) + lane], blk[128 + 32 * h + lane]
        code = ((ql >> (4 * (q >> 1))) & 15) | (((qh >> (2 * q)) & 3) << 4)
        sc = struct.unpack("<b", bytes([blk[192 + (i >> 4)]]))[0]
        out.append(_f32(_f32(d * sc) * (code - 32)))
    return out


def _lane_tables():
    """For one lane of one half (n, L): the Q3_K qs byte and the hmask nibble (bits 4n..4n+3) give the four values
    v = q - 4 * !h of j = 0..3 (value 128n + 32j + L); the Q6_K block keeps code = v + 32 in ql[64n + L] (j 0 low, 2
    high), ql[64n + 32 + L] (j 1 low, 3 high) and qh[32n + L] (2 bits a j)."""
    lo, hi, qh = bytearray(4096), bytearray(4096), bytearray(4096)
    for key in range(4096):
        b, nib = key >> 4, key & 15
        codes = [((b >> (2 * j)) & 3) + 28 + 4 * ((nib >> j) & 1) for j in range(4)]
        lo[key] = (codes[0] & 15) | (codes[2] & 15) << 4
        hi[key] = (codes[1] & 15) | (codes[3] & 15) << 4
        qh[key] = sum(((c >> 4) & 3) << (2 * j) for j, c in enumerate(codes))
    return bytes(lo), bytes(hi), bytes(qh)


LANE_LO, LANE_HI, LANE_QH = _lane_tables()


def q3k_to_q6k(blk):
    """The Q6_K block that decodes to the Q3_K block's values (the module doc's derivation)."""
    hm, qs = blk[0:32], blk[32:96]
    k0 = [qs[i] << 4 | (hm[i] & 15) for i in range(32)]
    k1 = [qs[32 + i] << 4 | hm[i] >> 4 for i in range(32)]
    ql = bytes(LANE_LO[k] for k in k0) + bytes(LANE_HI[k] for k in k0) + bytes(LANE_LO[k] for k in k1) + bytes(LANE_HI[k] for k in k1)
    qh = bytes(LANE_QH[k] for k in k0) + bytes(LANE_QH[k] for k in k1)
    sc = bytes((s & 0xFF) for s in q3k_scales(blk[96:108]))
    return ql + qh + sc + bytes(blk[108:110])


def gain_values(path, name, ty, blob):
    """A gain tensor's f32 values: F32, F16, BF16 or Q3_K."""
    if ty == F32:
        return list(struct.unpack(f"<{len(blob) // 4}f", blob))
    if ty == F16:
        return list(struct.unpack(f"<{len(blob) // 2}e", blob))
    if ty == BF16:
        return [struct.unpack("<f", b"\0\0" + blob[i:i + 2])[0] for i in range(0, len(blob), 2)]
    if ty == Q3_K:
        out = []
        for o in range(0, len(blob), Q3_BYTES):
            out += q3k_values(blob[o:o + Q3_BYTES])
        return out
    raise Refusal(65, f"{path}: gain {name} has type id {ty}; the fold reads F32, F16, BF16 or Q3_K")


def fold(path, name, q, k):
    """The converter's gate: q * k, each product rounded to f32 once (a product of two f32 values is exact in a double)."""
    try:
        return struct.pack(f"<{len(q)}f", *(a * b for a, b in zip(q, k)))
    except OverflowError as e:
        raise Refusal(65, f"{path}: the fold of {name} passes the f32 range") from e


# ---- the source ----

class Shard:
    def __init__(self, path):
        self.path = path
        self.version, self.kvs, self.infos, self.start, self.size, self.align = W.raw_header(path)
        _, self.tensors, _, _, _ = R.header(path)
        with open(path, "rb") as f:
            self.head = f.read(self.start)
        for name, ty, dims, off in self.tensors:
            if self.start + off + R.nbytes(path, name, ty, dims) > self.size:
                raise Refusal(65, f"{path}: tensor {name} runs past the file's end")

    def kv(self, key):
        for k, t, raw in self.kvs:
            if k == key.encode():
                return t, raw
        return None


class Site:
    def __init__(self, layer):
        self.layer = layer
        self.t = {}  # role -> (shard index, name, type, dims, offset)


def read_source(first):
    """The shards, the carrier's index, the sites in layer order, the key values and the tables' (width, type)."""
    first = os.path.realpath(first)
    meta = R.header(first)[0]
    arch = meta.get("general.architecture")
    if arch != ARCH.encode():
        raise Refusal(65, f"{first}: general.architecture is {arch!r}, not {ARCH!r}")
    shards = [Shard(p) for p in R.shard_paths(first, meta)]
    carriers = [i for i, s in enumerate(shards) if s.kv(full("engram.layer_ids"))]
    for s in shards:
        for k, _, _ in s.kvs:
            if k.decode("utf-8", "replace").startswith(full("ple.")):
                raise Refusal(65, f"{s.path}: {k.decode()} is a key of llama.cpp's layout: the set is not in the per-site layout")
        for name, *_ in s.tensors:
            if name == CONCAT:
                raise Refusal(65, f"{s.path}: {CONCAT} is in the set: it is in llama.cpp's layout already")
    if len(carriers) != 1:
        raise Refusal(65, f"{first}: {len(carriers)} shards hold {full('engram.layer_ids')}, not one")
    c = carriers[0]
    known = {full(a) for a, _ in KEYS} | {full(a) for a in DROPPED}
    for k, _, _ in shards[c].kvs:
        key = k.decode("utf-8", "replace")
        if key.startswith(full("engram.")) and key not in known:
            raise Refusal(65, f"{shards[c].path}: {key} is a per-site Engram key this tool does not place")
    vals = {}
    for a, _ in KEYS:
        got = shards[c].kv(full(a))
        if got is None:
            raise Refusal(65, f"{shards[c].path}: no {full(a)}")
        vals[a] = got
    path = shards[c].path
    layers = _array(path, "layer_ids", vals["engram.layer_ids"][1])[1]
    if not layers or any(x < 0 for x in layers) or sorted(set(layers)) != layers:
        raise Refusal(65, f"{path}: {full('engram.layer_ids')} {layers} is not ascending non-negative layer ids")
    sites = {L: Site(L) for L in layers}
    for si, s in enumerate(shards):
        for name, ty, dims, off in s.tensors:
            m = SITE_TENSOR.match(name)
            if not m:
                if "engram" in name:
                    raise Refusal(65, f"{s.path}: tensor {name} is an Engram tensor this tool does not place")
                continue
            L, role = int(m.group(1)), m.group(2)
            if role not in ROLES:
                raise Refusal(65, f"{s.path}: tensor {name} is an Engram tensor this tool does not place")
            if L not in sites:
                raise Refusal(65, f"{s.path}: tensor {name} is on layer {L}, which {full('engram.layer_ids')} {layers} does not list")
            sites[L].t[role] = (si, name, ty, dims, off)
    for L, site in sites.items():
        for role in ROLES:
            if role not in site.t:
                raise Refusal(65, f"{first}: no blk.{L}.{role}.weight, and {full('engram.layer_ids')} lists layer {L}")
    return shards, c, [sites[L] for L in layers], vals


def plan_tables(path, sites, vals):
    """Each site's (rows, base row) of the one table, the absolute offsets, the width and the table type."""
    key_len = _scalar(path, "key_length", *vals["engram.key_length"])
    primes = _array(path, "primes", vals["engram.primes"][1])[1]
    offsets = _array(path, "offsets", vals["engram.offsets"][1])[1]
    n = len(sites)
    if len(primes) % n or len(offsets) != len(primes) or not primes:
        raise Refusal(65, f"{path}: {len(primes)} primes and {len(offsets)} offsets are not the same whole number for each of {n} sites")
    per = len(primes) // n
    widths, types, bases, absolute, base = set(), set(), [], [], 0
    for e, site in enumerate(sites):
        _, name, ty, dims, _ = site.t["engram_embd"]
        if len(dims) != 2:
            raise Refusal(65, f"{path}: table {name} has {len(dims)} dimensions, not 2")
        widths.add(dims[0])
        types.add(ty)
        mine = primes[e * per:(e + 1) * per]
        run = 0
        for i, p in enumerate(mine):
            if offsets[e * per + i] != run:
                raise Refusal(65, f"{path}: {full('engram.offsets')}[{e * per + i}] is {offsets[e * per + i]}; site {e} "
                                  f"(layer {site.layer}) counts its buckets from its own row 0, so it is {run}")
            absolute.append(base + run)
            run += p
        if run != dims[1]:
            raise Refusal(65, f"{path}: table {name} has {dims[1]} rows; site {e}'s buckets cover {run}")
        bases.append(base)
        base += run
    if len(widths) != 1 or len(types) != 1:
        raise Refusal(65, f"{path}: the sites' tables are not one width and one type (widths {sorted(widths)}, types {sorted(types)}): "
                          "one table cannot hold them")
    width, ty = widths.pop(), types.pop()
    if width != key_len:
        raise Refusal(65, f"{path}: the tables' rows are {width} wide, and {full('engram.key_length')} is {key_len}")
    if "engram.num_embeddings" in DROPPED:
        pass
    return bases, base, absolute, width, ty


def upstream_keys(path, vals, absolute):
    """The carrier's llama.cpp-spelled keys, in the converter's order: [(key bytes, type, raw)]."""
    out = []
    for a, b in KEYS:
        t, raw = vals[a]
        if a == "engram.layer_ids":
            layers = _array(path, a, raw)[1]
            if any(x >= 1 << 32 for x in layers):
                raise Refusal(65, f"{path}: a layer id of {layers} passes u32")
            t, raw = 9, _pack_array(4, layers)
        elif a == "engram.offsets":
            t, raw = 9, _pack_array(10, absolute)
        out.append((full(b).encode(), t, raw))
    return out


# ---- the copy ----

class Out:
    """One output tensor: name, type, dims, byte count, and how its bytes are made."""

    def __init__(self, name, ty, dims, nbytes, parts):
        self.name, self.ty, self.dims, self.nbytes, self.parts = name, ty, dims, nbytes, parts


def _copy(src, off, n, out, h=None):
    with open(src, "rb") as f:
        f.seek(off)
        while n:
            b = f.read(min(CHUNK, n))
            if not b:
                raise Refusal(2, f"{src}: ends before byte {off + n}")
            out.write(b)
            if h is not None:
                h.update(b)
            n -= len(b)


def _transcode(src, off, n, out):
    with open(src, "rb") as f:
        f.seek(off)
        step = Q3_BYTES * 4096
        while n:
            b = f.read(min(step, n))
            if not b or len(b) % Q3_BYTES:
                raise Refusal(2, f"{src}: ends inside a block")
            out.write(b"".join(q3k_to_q6k(b[o:o + Q3_BYTES]) for o in range(0, len(b), Q3_BYTES)))
            n -= len(b)


def _part_bytes(part):
    kind, src, off, n = part[:4]
    return n if kind == "copy" else (n // Q3_BYTES * Q6_BYTES if kind == "q6" else len(part[4]))


def _row_type(path, name, ty, want, dims):
    """The type `name` is written at, and whether its blocks are transcoded."""
    if want is None or want == ty:
        return ty, False
    if ty != Q3_K or want != Q6_K:
        raise Refusal(65, f"{path}: {name} is {TYPE_NAMES.get(ty, ty)}; Q6_K is written only from Q3_K, an exact transcode, "
                          "and anything else would be a requantization")
    if dims[0] % QK:
        raise Refusal(65, f"{path}: {name} rows of {dims[0]} are not whole blocks of {QK}")
    return Q6_K, True


def plan_shards(shards, c, sites, vals, table_type, kv_type, per_site=False):
    """Per shard, the output tensors in the source's data order; and the carrier's new key list (None: the source's,
    for the per-site counterpart)."""
    path = shards[c].path
    bases, rows, absolute, width, tty = plan_tables(path, sites, vals)
    first_table = sites[0].t["engram_embd"]
    out_ty, tq6 = _row_type(path, CONCAT, tty, table_type, [width])
    parts = []
    for site in sites:
        si, name, ty, dims, off = site.t["engram_embd"]
        n = R.nbytes(shards[si].path, name, ty, dims)
        parts.append(("q6" if tq6 else "copy", shards[si].path, shards[si].start + off, n))
    table = Out(CONCAT, out_ty, [width, rows], sum(_part_bytes(p) for p in parts), parts)
    by_name = {}
    if per_site:
        for site, part in zip(sites, parts):
            _, name, _, dims, _ = site.t["engram_embd"]
            by_name[name] = Out(name, out_ty, list(dims), _part_bytes(part), [part])
    for site in sites:
        si_k, k_name, k_ty, k_dims, k_off = site.t["engram_k"]
        si_q, q_name, q_ty, q_dims, q_off = site.t["engram_q"]
        if k_dims != q_dims:
            raise Refusal(65, f"{path}: {k_name} {k_dims} and {q_name} {q_dims} are not one shape")
        kb = _read(shards[si_k], k_name, k_ty, k_dims, k_off)
        qb = _read(shards[si_q], q_name, q_ty, q_dims, q_off)
        gate = fold(path, f"blk.{site.layer}.engram_gate.weight",
                    gain_values(path, q_name, q_ty, qb), gain_values(path, k_name, k_ty, kb))
        si_w, w_name, w_ty, w_dims, w_off = site.t["engram_wkv"]
        kv_ty, kq6 = _row_type(path, w_name, w_ty, kv_type, w_dims)
        n = R.nbytes(shards[si_w].path, w_name, w_ty, w_dims)
        part = ("q6" if kq6 else "copy", shards[si_w].path, shards[si_w].start + w_off, n)
        if per_site:
            ones = ONE * (len(gate) // 4)
            by_name[k_name] = Out(k_name, F32, list(k_dims), len(ones), [("bytes", None, 0, 0, ones)])
            by_name[q_name] = Out(q_name, F32, list(q_dims), len(gate), [("bytes", None, 0, 0, gate)])
            by_name[w_name] = Out(w_name, kv_ty, list(w_dims), _part_bytes(part), [part])
            continue
        by_name[k_name] = Out(f"blk.{site.layer}.engram_gate.weight", F32, list(k_dims), len(gate), [("bytes", None, 0, 0, gate)])
        by_name[q_name] = None
        by_name[w_name] = Out(f"blk.{site.layer}.engram_kv.weight", kv_ty, list(w_dims), _part_bytes(part), [part])
        by_name[site.t["engram_embd"][1]] = table if site.t["engram_embd"] == first_table else None
    plan = []
    for s in shards:
        outs = []
        for name, ty, dims, off in sorted(s.tensors, key=lambda t: t[3]):
            if name in by_name:
                if by_name[name] is not None:
                    outs.append(by_name[name])
            else:
                n = R.nbytes(s.path, name, ty, dims)
                outs.append(Out(name, ty, list(dims), n, [("copy", s.path, s.start + off, n)]))
        plan.append(outs)
    return plan, None if per_site else upstream_keys(path, vals, absolute), (bases, rows, absolute, width)


def _read(shard, name, ty, dims, off):
    n = R.nbytes(shard.path, name, ty, dims)
    with open(shard.path, "rb") as f:
        f.seek(shard.start + off)
        b = f.read(n)
    if len(b) != n:
        raise Refusal(2, f"{shard.path}: {name} ends past the file")
    return b


def _infos(outs, align):
    blob, offs, pos = b"", [], 0
    for o in outs:
        pos += (-pos) % align
        nb = o.name.encode()
        blob += struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(o.dims)) + b"".join(struct.pack("<Q", d) for d in o.dims)
        blob += struct.pack("<IQ", o.ty, pos)
        offs.append(pos)
        pos += o.nbytes
    return blob, offs


def new_kvs(shard, is_carrier, ukeys, n_tensors, record):
    """`shard`'s keys with split.tensors.count the new count and, in the carrier, the per-site Engram keys replaced
    in place by the llama.cpp spellings (kept when `ukeys` is None) and the record appended."""
    out, placed = [], False
    canon = {full(a).encode() for a, _ in KEYS} | {full(a).encode() for a in DROPPED}
    for k, t, raw in shard.kvs:
        if k in (r.encode() for r in RECORD):
            raise Refusal(65, f"{shard.path}: already carries {k.decode()}; a relaid copy is not relaid again")
        if k == b"split.tensors.count":
            raw = struct.pack(R.SCALAR[t], n_tensors)
        if is_carrier and ukeys is not None and k in canon:
            if not placed:
                out += ukeys
                placed = True
            continue
        out.append((k, t, raw))
    if is_carrier:
        out += [W._string_kv(RECORD[i], v) for i, v in enumerate(record)]
    return out


def header_sha(path):
    start = R.header(path)[2]
    with open(path, "rb") as f:
        return hashlib.sha256(f.read(start)).hexdigest()


def out_name(src, stem_tag=TWIN):
    base = os.path.basename(src)
    m = R.SPLIT.search(base)
    return base[: m.start()] + f"-{stem_tag}" + base[m.start():] if m else base[:-5] + f"-{stem_tag}.gguf"


def relayout(first, out, table_type=None, kv_type=None, per_site=False):
    first = os.path.realpath(first)
    out = os.path.abspath(out)
    if os.path.lexists(out):
        raise Refusal(73, f"{out} exists; a relaid copy is written once")
    if not os.path.isdir(os.path.dirname(out)):
        raise Refusal(2, f"{os.path.dirname(out)} is not a directory")
    shards, c, sites, vals = read_source(first)
    plan, ukeys, tables = plan_shards(shards, c, sites, vals, table_type, kv_type, per_site)
    n_tensors = sum(len(p) for p in plan)
    table_names = {s.t["engram_embd"][1] for s in sites} if per_site else {CONCAT}
    kv_tail = ".engram_wkv.weight" if per_site else ".engram_kv.weight"
    types = f"tables={TYPE_NAMES.get(plan_tables(shards[c].path, sites, vals)[4], '?')}->" \
            f"{TYPE_NAMES[next(o for p in plan for o in p if o.name in table_names).ty]} " \
            f"kv={'+'.join(sorted({TYPE_NAMES.get(o.ty, str(o.ty)) for p in plan for o in p if o.name.endswith(kv_tail)}))}" \
            + (" gains=ones,gate" if per_site else "")
    record = (first, header_sha(first), types)
    os.mkdir(out)
    written = []
    try:
        for i, (s, outs) in enumerate(zip(shards, plan)):
            dst = os.path.join(out, out_name(s.path, PER_SITE if per_site else TWIN))
            infos, offs = _infos(outs, s.align)
            head = W._pack(s.version, len(outs), new_kvs(s, i == c, ukeys, n_tensors, record), infos, s.align)
            with open(dst, "wb") as f:
                f.write(head)
                pos = 0
                for o, at in zip(outs, offs):
                    f.write(b"\0" * (at - pos))
                    for part in o.parts:
                        kind, src, off, n = part[:4]
                        if kind == "copy":
                            _copy(src, off, n, f)
                        elif kind == "q6":
                            _transcode(src, off, n, f)
                        else:
                            f.write(part[4])
                    pos = at + o.nbytes
            written.append(dst)
        hold(shards, plan, written, sites, tables, per_site)
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise
    return written, tables, types


def _sha_range(path, off, n):
    h = hashlib.sha256()
    _copy(path, off, n, _Sink(), h)
    return h.hexdigest()


class _Sink:
    def write(self, b):
        pass


def _sample(n_blocks, seed):
    rnd = random.Random(seed)
    picks = {0, n_blocks - 1}
    while len(picks) < min(SAMPLE, n_blocks):
        picks.add(rnd.randrange(n_blocks))
    return sorted(picks)


def hold(shards, plan, written, sites, tables, per_site=False):
    """The copy held to its source (the module doc); a difference refuses by name (65)."""
    for dst, outs in zip(written, plan):
        meta, tensors, start, size, _ = R.header(dst)
        got = {name: (ty, dims, off) for name, ty, dims, off in tensors}
        if [t[0] for t in tensors] != [o.name for o in outs]:
            raise Refusal(65, f"{dst}: the tensors written are not the planned ones")
        end = 0
        for o in outs:
            ty, dims, off = got[o.name]
            if ty != o.ty or list(dims) != o.dims or R.nbytes(dst, o.name, ty, dims) != o.nbytes:
                raise Refusal(65, f"{dst}: {o.name} is written as type {ty} {dims}, planned {o.ty} {o.dims}")
            at = start + off
            end = max(end, at + o.nbytes)
            for part in o.parts:
                kind, src, soff, n = part[:4]
                if kind == "copy":
                    if _sha_range(src, soff, n) != _sha_range(dst, at, n):
                        raise Refusal(65, f"{dst}: {o.name}'s bytes at {at} differ from {src}'s at {soff}")
                elif kind == "q6":
                    with open(src, "rb") as fs, open(dst, "rb") as fd:
                        for b in _sample(n // Q3_BYTES, o.name):
                            fs.seek(soff + b * Q3_BYTES)
                            fd.seek(at + b * Q6_BYTES)
                            a, z = fs.read(Q3_BYTES), fd.read(Q6_BYTES)
                            if struct.pack("<256f", *q3k_values(a)) != struct.pack("<256f", *q6k_values(z)):
                                raise Refusal(65, f"{dst}: {o.name} block {b} does not decode to {src}'s block at {soff + b * Q3_BYTES}")
                    n = n // Q3_BYTES * Q6_BYTES
                else:
                    with open(dst, "rb") as fd:
                        fd.seek(at)
                        if fd.read(n or len(part[4])) != part[4]:
                            raise Refusal(65, f"{dst}: {o.name}'s bytes are not the fold")
                    n = len(part[4])
                at += n
        if end != size:
            raise Refusal(65, f"{dst}: the data ends at {end}, the file at {size}")
    bases, rows, _, width = tables
    if per_site:
        for site, base, end in zip(sites, bases, bases[1:] + [rows]):
            name = site.t["engram_embd"][1]
            got = [o for outs in plan for o in outs if o.name == name]
            if len(got) != 1 or got[0].dims != [width, end - base]:
                raise Refusal(65, f"site layer {site.layer}'s table is not written once at its {end - base} rows")
        return
    concat = [o for outs in plan for o in outs if o.name == CONCAT]
    if len(concat) != 1 or concat[0].dims != [width, rows]:
        raise Refusal(65, "the one table is not written once at the sites' rows")
    row_bytes = concat[0].nbytes // rows
    at = 0
    for site, base, part in zip(sites, bases, concat[0].parts):
        if at != base * row_bytes:
            raise Refusal(65, f"site layer {site.layer} starts at byte {at} of {CONCAT}, its base row {base} is byte {base * row_bytes}")
        at += _part_bytes(part)


def twin(fixture_dir, per_site=False):
    d = os.path.abspath(fixture_dir).rstrip("/")
    name, root = os.path.basename(d), os.path.dirname(d)
    if not name or not root or root == "/":
        raise Refusal(64, f"{fixture_dir} is not <fixture root>/<fixture dir>")
    at = os.path.join(os.path.dirname(root), "fixture-variants", TWIN, name)
    return os.path.join(at, PER_SITE_DIR) if per_site else at


def check(twin_first, source):
    src = os.path.realpath(source)
    shard = Shard(twin_first)
    rec = {}
    for i, key in enumerate(RECORD):
        got = shard.kv(key)
        if got is None or got[0] != 8:
            raise Refusal(65, f"{twin_first}: no {key}: not a copy this tool wrote")
        rec[key] = got[1][8:].decode()
    if rec[RECORD[0]] != src:
        raise Refusal(65, f"{twin_first} was made from {rec[RECORD[0]]}, not {src}: make it again (just fixture-engram-upstream)")
    sha = header_sha(src)
    if rec[RECORD[1]] != sha:
        raise Refusal(65, f"{twin_first} was made from a header of {src} with sha256 {rec[RECORD[1]]}; the file's is {sha} now: "
                          "make it again (just fixture-engram-upstream)")
    return rec


# ---- the self-test ----

def _kv(key, t, raw):
    kb = key.encode()
    return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", t) + raw


def _u32(key, v):
    return _kv(key, 4, struct.pack("<I", v))


def _arr(key, et, values):
    return _kv(key, 9, _pack_array(et, values))


def _q3_block(rnd):
    b = bytearray(rnd.randrange(256) for _ in range(108))
    b += struct.pack("<e", rnd.choice((1.0, -0.5, 3.0e-3, 6.1e-5, 2.0 ** -20, 1000.0)) * rnd.random())
    return bytes(b)


def _source(d, rnd, primes=(3, 5, 7, 11, 2, 3, 5, 7), offsets=None, table_ty=Q3_K, extra=(), keys=None, rows=None):
    """A two-shard per-site set: layers 1 and 3, two heads, ngram size 3 (four buckets a site); site 1's table, gains
    (Q3_K) and kv in shard 1, site 3's (gains BF16) in shard 2."""
    os.makedirs(d)
    first = os.path.join(d, "m-00001-of-00002.gguf")
    second = os.path.join(d, "m-00002-of-00002.gguf")
    rows = rows or (sum(primes[:4]), sum(primes[4:]))
    if offsets is None:
        offsets = [0, 3, 8, 15, 0, 2, 5, 10]
    blocks = lambda n: b"".join(_q3_block(rnd) for _ in range(n))

    def table(n):
        return blocks(n) if table_ty == Q3_K else bytes(rnd.randrange(256) for _ in range(n * 1024))

    bf16 = lambda n: b"".join(struct.pack("<f", rnd.uniform(-2, 2))[2:] for _ in range(n))
    split = lambda no: [_kv("split.no", 2, struct.pack("<H", no)), _kv("split.count", 2, struct.pack("<H", 2)),
                        _kv("split.tensors.count", 5, struct.pack("<i", 10))]
    kvs = [_kv("general.architecture", 8, struct.pack("<Q", len(ARCH)) + ARCH.encode()), _u32(full("block_count"), 4)]
    kvs += keys if keys is not None else [
        _u32(full("engram.head_count"), 2), _u32(full("engram.key_length"), 256), _u32(full("engram.max_ngram_size"), 3),
        _arr(full("engram.layer_ids"), 5, [1, 3]), _arr(full("engram.multipliers"), 10, [11, 13, 17, 19, 23, 29]),
        _arr(full("engram.primes"), 10, list(primes)), _arr(full("engram.offsets"), 10, offsets),
        _arr(full("engram.token_map"), 5, [0, 1, 2, 1]), _u32(full("engram.pad_id"), 2)]
    kvs += [_kv("bloomery.fixture.version", 4, struct.pack("<I", 1))]
    t1 = [("token_embd.weight", F32, [8, 3], bytes(rnd.randrange(256) for _ in range(96))),
          ("blk.1.engram_embd.weight", table_ty, [256, rows[0]], table(rows[0])),
          ("blk.1.engram_k.weight", Q3_K, [256, 2], blocks(2)),
          ("blk.1.engram_q.weight", Q3_K, [256, 2], blocks(2)),
          ("blk.1.engram_wkv.weight", Q3_K, [512, 3], blocks(6)),
          ("blk.2.attn_norm.weight", F32, [16], bytes(range(64)))] + list(extra)
    t2 = [("blk.3.engram_k.weight", BF16, [256, 2], bf16(512)),
          ("blk.3.engram_embd.weight", table_ty, [256, rows[1]], table(rows[1])),
          ("blk.3.engram_q.weight", BF16, [256, 2], bf16(512)),
          ("blk.3.engram_wkv.weight", Q3_K, [512, 3], blocks(6)),
          ("output.weight", F32, [8, 2], bytes(rnd.randrange(256) for _ in range(64)))]
    R.write_gguf(first, kvs + split(0), t1)
    R.write_gguf(second, split(1), t2)
    return first


def _tensor(path, want):
    _, tensors, start, _, _ = R.header(path)
    for name, ty, dims, off in tensors:
        if name == want:
            with open(path, "rb") as f:
                f.seek(start + off)
                return ty, dims, f.read(R.nbytes(path, name, ty, dims))
    return None


def self_test():
    ok = True

    def check_(name, cond, detail=""):
        nonlocal ok
        print(f"  {'ok  ' if cond else 'FAIL'} {name}" + ("" if cond else f": {detail}"))
        ok &= bool(cond)

    def refused(what, fn, code, frag):
        try:
            fn()
            check_(f"refuses {what}", False, "no refusal")
        except Refusal as e:
            check_(f"refuses {what}", e.code == code and frag in str(e), f"{e.code} {e}")

    rnd = random.Random(7)
    # Every (qs byte, hmask nibble) lane and every scale, and random blocks: the Q6_K block decodes to the Q3_K values.
    lanes, bad = [], 0
    for blk in range(64):
        b = bytearray(_q3_block(rnd))
        for n in (0, 1):
            for lane in range(32):
                key = blk * 64 + n * 32 + lane
                b[32 + 32 * n + lane] = key >> 4
                b[lane] = (b[lane] & (0xF0 if n == 0 else 0x0F)) | ((key & 15) << (4 * n))
        lanes.append(bytes(b))
    for s in range(4):
        b = bytearray(_q3_block(rnd))
        packed = [16 * s + j for j in range(16)]
        b[96:108] = bytes([(packed[j] & 15) | (packed[j + 8] & 15) << 4 for j in range(8)] +
                          [sum(((packed[4 * i + j] >> 4) & 3) << (2 * i) for i in range(4)) for j in range(4)])
        lanes.append(bytes(b))
        check_(f"Q3_K scales {16 * s}..{16 * s + 15} unpack as packed", q3k_scales(b[96:108]) == [p - 32 for p in packed],
               q3k_scales(b[96:108]))
    for blk in lanes + [_q3_block(rnd) for _ in range(200)]:
        bad += struct.pack("<256f", *q3k_values(blk)) != struct.pack("<256f", *q6k_values(q3k_to_q6k(blk)))
    check_("every lane, scale and 200 random Q3_K blocks decode to the same f32 bits as Q6_K", bad == 0, f"{bad} blocks differ")
    mutant = bytearray(q3k_to_q6k(lanes[0]))
    mutant[192:208] = bytes(16)
    check_("a Q6_K block with its scales gone decodes otherwise", q6k_values(bytes(mutant)) != q3k_values(lanes[0]))
    check_("the twin of <root>/v41 is <parent>/fixture-variants/engram-upstream/v41",
           twin("/m/fixtures/v41") == "/m/fixture-variants/engram-upstream/v41", twin("/m/fixtures/v41"))

    with tempfile.TemporaryDirectory() as d:
        first = _source(os.path.join(d, "src"), random.Random(1))
        out = os.path.join(d, "up")
        written, tables, types = relayout(first, out)
        s1, s2 = written
        meta, tensors, *_ = R.header(s1)
        names = [t[0] for t in tensors] + [t[0] for t in R.header(s2)[1]]
        check_("the tensors: one table at site 1's place, gates at the k's, kvs at the wkv's, the q's and site 3's table gone",
               names == ["token_embd.weight", CONCAT, "blk.1.engram_gate.weight", "blk.1.engram_kv.weight", "blk.2.attn_norm.weight",
                         "blk.3.engram_gate.weight", "blk.3.engram_kv.weight", "output.weight"], names)
        ty, dims, blob = _tensor(s1, CONCAT)
        a = _tensor(first, "blk.1.engram_embd.weight")[2]
        b = _tensor(os.path.join(d, "src", "m-00002-of-00002.gguf"), "blk.3.engram_embd.weight")[2]
        check_("the table is site 1's rows then site 3's, Q3_K bits copied", (ty, dims, blob) == (Q3_K, [256, 43], a + b), (ty, dims))
        for L, shard, kty in ((1, first, "Q3_K"), (3, os.path.join(d, "src", "m-00002-of-00002.gguf"), "BF16")):
            q = _tensor(shard, f"blk.{L}.engram_q.weight")
            k = _tensor(shard, f"blk.{L}.engram_k.weight")
            gate = _tensor(s1 if L == 1 else s2, f"blk.{L}.engram_gate.weight")
            qv, kv = gain_values("q", "q", q[0], q[2]), gain_values("k", "k", k[0], k[2])
            want = struct.pack("<512f", *(struct.unpack("<f", struct.pack("<f", x * y))[0] for x, y in zip(qv, kv)))
            check_(f"site {L}'s gate is q * k in f32 ({kty} gains)", gate[:2] == (F32, [256, 2]) and gate[2] == want)
            check_(f"site {L}'s kv is its wkv, renamed",
                   _tensor(s1 if L == 1 else s2, f"blk.{L}.engram_kv.weight") == _tensor(shard, f"blk.{L}.engram_wkv.weight"))
        _, kvs, *_ = W.raw_header(s1)
        keys = {k.decode(): (t, raw) for k, t, raw in kvs}
        check_("the hash under llama.cpp's spelling, the per-site keys gone",
               _array("t", "l", keys[full("ple.layers")][1]) == (4, [1, 3])
               and _array("t", "o", keys[full("ple.head_offsets")][1]) == (10, [0, 3, 8, 15, 26, 28, 31, 36])
               and _array("t", "p", keys[full("ple.head_vocab_sizes")][1])[1] == [3, 5, 7, 11, 2, 3, 5, 7]
               and meta[full("ple.ngram_size")] == 3 and meta[full("ple.heads_per_ngram")] == 2
               and meta[full("embedding_length_per_layer_input")] == 256 and meta[full("engram.pad_token_id")] == 2
               and full("engram.token_map") in keys and not [k for k in keys if k.startswith(full("engram.")) and
                                                              k not in (full("engram.pad_token_id"), full("engram.token_map"))],
               sorted(keys))
        check_("split.tensors.count is the new count in both shards, the fixture keys kept",
               meta["split.tensors.count"] == 8 and R.header(s2)[0]["split.tensors.count"] == 8 and meta["bloomery.fixture.version"] == 1)
        check_("the record names the source and its header", check(s1, first)[RECORD[2]] == "tables=Q3_K->Q3_K kv=Q3_K")
        check_("the copies tile their files", R.tensors_of(s1)[1] == 2, R.tensors_of(s1))
        refused("an existing output directory", lambda: relayout(first, out), 73, "exists")
        refused("a copy in llama.cpp's layout as a source", lambda: relayout(s1, os.path.join(d, "again")), 65, "llama.cpp's layout")
        q6 = os.path.join(d, "q6")
        written6, _, types6 = relayout(first, q6, table_type=Q6_K, kv_type=Q6_K)
        ty, dims, blob = _tensor(written6[0], CONCAT)
        same = all(q3k_values(a[i * 110:(i + 1) * 110] if i < 26 else b[(i - 26) * 110:(i - 25) * 110]) ==
                   q6k_values(blob[i * 210:(i + 1) * 210]) for i in range(43))
        check_("--table-type q6_K: every row decodes to the source's values", ty == Q6_K and dims == [256, 43] and same, (ty, dims))
        kv6 = _tensor(written6[0], "blk.1.engram_kv.weight")
        wkv = _tensor(first, "blk.1.engram_wkv.weight")
        check_("--kv-type q6_K: the kv decodes to the wkv's values", kv6[0] == Q6_K and all(
            q3k_values(wkv[2][i * 110:(i + 1) * 110]) == q6k_values(kv6[2][i * 210:(i + 1) * 210]) for i in range(6)))
        check_("the record names the types", types6 == "tables=Q3_K->Q6_K kv=Q6_K", types6)
        # The per-site counterpart: the source's layout and keys, the copy's numbers.
        ps = relayout(first, os.path.join(d, "ps"), table_type=Q6_K, kv_type=Q6_K, per_site=True)[0]
        src_names = [t[0] for t in R.header(first)[1]] + [t[0] for t in R.header(os.path.join(d, "src", "m-00002-of-00002.gguf"))[1]]
        ps_names = [t[0] for t in R.header(ps[0])[1]] + [t[0] for t in R.header(ps[1])[1]]
        check_("--per-site: the source's tensors, in its shards and order", ps_names == src_names, ps_names)
        check_("--per-site: shards named engram-per-site",
               [os.path.basename(x) for x in ps] == ["m-engram-per-site-00001-of-00002.gguf", "m-engram-per-site-00002-of-00002.gguf"], ps)
        for L, at, n in ((1, 0, 26), (3, 26, 17)):
            sh, up = ps[0] if L == 1 else ps[1], written6[0] if L == 1 else written6[1]
            gate = _tensor(up, f"blk.{L}.engram_gate.weight")
            check_(f"--per-site: site {L}'s engram_q is the copy's gate, bytes for bytes",
                   _tensor(sh, f"blk.{L}.engram_q.weight") == gate, _tensor(sh, f"blk.{L}.engram_q.weight")[:2])
            check_(f"--per-site: site {L}'s engram_k is F32 ones",
                   _tensor(sh, f"blk.{L}.engram_k.weight") == (F32, [256, 2], struct.pack("<512f", *[1.0] * 512)))
            check_(f"--per-site: site {L}'s wkv is the copy's kv, bytes for bytes",
                   _tensor(sh, f"blk.{L}.engram_wkv.weight") == _tensor(up, f"blk.{L}.engram_kv.weight"))
            check_(f"--per-site: site {L}'s table is the copy's rows {at}..{at + n}, bytes for bytes",
                   _tensor(sh, f"blk.{L}.engram_embd.weight") == (Q6_K, [256, n], _tensor(written6[0], CONCAT)[2][at * 210:(at + n) * 210]))
        ps_meta = R.header(ps[0])[0]
        src_meta = R.header(first)[0]
        check_("--per-site: the per-site keys kept, no llama.cpp key",
               all(ps_meta[k] == v for k, v in src_meta.items() if k.startswith(full("engram.")))
               and not [k for k in ps_meta if k.startswith(full("ple."))] and ps_meta["split.tensors.count"] == len(src_names),
               sorted(ps_meta))
        check_("--per-site: the record names the types and the gains", check(ps[0], first)[RECORD[2]] ==
               "tables=Q3_K->Q6_K kv=Q6_K gains=ones,gate", check(ps[0], first)[RECORD[2]])
        refused("a per-site counterpart as a source", lambda: relayout(ps[0], os.path.join(d, "ps-again")), 65, "not relaid again")
        check_("the per-site counterpart of <root>/v41 is in its twin's directory, under per-site",
               twin("/m/fixtures/v41", per_site=True) == "/m/fixture-variants/engram-upstream/v41/per-site")
        # The record goes stale when the source's header moves.
        with open(first, "r+b") as f:
            raw = f.read()
            at = raw.index(b"bloomery.fixture.version") + len("bloomery.fixture.version") + 4
            f.seek(at)
            f.write(struct.pack("<I", 2))
        refused("a twin whose source header moved", lambda: check(s1, first), 65, "make it again")
        refused("a twin of another source", lambda: check(s1, s2), 65, "was made from")
        for what, kw, frag in (
                ("a table whose rows are not its buckets'", dict(rows=(27, 17)), "buckets cover"),
                ("offsets counted across the sites", dict(offsets=[0, 3, 8, 15, 26, 28, 31, 36]), "counts its buckets from its own row 0"),
                ("an Engram tensor it does not place", dict(extra=[("blk.1.engram_scale.weight", F32, [4], bytes(16))]), "does not place"),
                ("a missing per-site key", dict(keys=[_arr(full("engram.layer_ids"), 5, [1, 3])]), "no deepseek41.engram.")):
            src = _source(os.path.join(d, what.replace(" ", "_").replace("'", "")), random.Random(2), **kw)
            refused(what, lambda: relayout(src, os.path.join(d, what.replace(" ", "-") + "-out")), 65, frag)
        f32_src = _source(os.path.join(d, "f32table"), random.Random(3), table_ty=F32)
        refused("Q6_K from an F32 table (a requantization)", lambda: relayout(f32_src, os.path.join(d, "f32out"), table_type=Q6_K),
                65, "requantization")
        check_("a refused copy leaves no output directory", not os.path.exists(os.path.join(d, "f32out")))
    return ok


def main(argv):
    try:
        if argv[:1] == ["--self-test"]:
            ok = self_test()
            print(f"engram-relayout self-test: {'PASS' if ok else 'FAIL'}")
            return 0 if ok else 1
        if argv[:1] == ["relayout"] and len(argv) >= 3:
            _, first, out, *rest = argv
            want = {"--table-type": None, "--kv-type": None}
            per_site = False
            while rest:
                if rest[0] == "--per-site" and not per_site:
                    per_site, rest = True, rest[1:]
                    continue
                if rest[0] not in want or len(rest) < 2 or rest[1] != "q6_K":
                    raise Refusal(64, f"relayout takes --table-type q6_K, --kv-type q6_K and --per-site, got {rest[:2]}")
                want[rest[0]] = Q6_K
                rest = rest[2:]
            written, (bases, rows, _, width), types = relayout(first, out, want["--table-type"], want["--kv-type"], per_site)
            for p in written:
                print(f"relayout: {p}")
            what = f"{len(bases)} per-site tables" if per_site else f"{CONCAT} [{width}, {rows}]"
            print(f"relayout: {what}, sites at base rows {bases}; {types}")
            return 0
        if argv[:1] == ["twin"] and len(argv) in (2, 3) and argv[2:] in ([], ["--per-site"]):
            print(twin(argv[1], argv[2:] == ["--per-site"]))
            return 0
        if argv[:1] == ["check"] and len(argv) == 4 and argv[2] == "--source":
            rec = check(argv[1], argv[3])
            print(f"check: {argv[1]} is the twin of {rec[RECORD[0]]} as it stands ({rec[RECORD[2]]})")
            return 0
        raise Refusal(64, "usage:\n" + "\n".join(__doc__.split("\n\n")[1].splitlines()))
    except Refusal as e:
        print(f"engram-relayout.py: {e}", file=sys.stderr)
        return e.code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
