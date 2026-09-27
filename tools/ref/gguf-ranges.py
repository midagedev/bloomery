#!/usr/bin/env python3
"""The file ranges a reference engine keeps on the host, read from a GGUF header, and a preheat of them.

    tools/ref/gguf-ranges.py host <first shard> --n-cpu-moe K [--ngl N] [--out FILE]
    tools/ref/gguf-ranges.py preheat FILE [--threads T] [--chunk-mib C]
    tools/ref/gguf-ranges.py fixture <path>
    tools/ref/gguf-ranges.py --self-test

`host` reads the header of the first shard and of every shard its `split.count` names (the
`-NNNNN-of-MMMMM.gguf` siblings; header bytes only, no tensor data) and selects what llama.cpp and ik
place on the host under `-ngl N --n-cpu-moe K`:
  - the routed experts of layers 0..K-1: every tensor whose name matches
    `blk\\.<i>\\.ffn_(up|down|gate|gate_up)_(ch|)exps`, i < K — the buffer-type overrides
    llama-bench's --n-cpu-moe adds (common/common.h LLM_FFN_EXPS_REGEX, llm_add_n_cpu_ffn_overrides;
    ik's llama-bench makes the same list);
  - token_embd.weight: both engines put the input embedding in the host input context.
Leading dense layers (GLM's 0-2: no expert tensor of the file names them) add nothing: the override
list matches nothing there. The first layer with expert tensors is f; every layer f..K-1 must have
them, and the header's <arch>.leading_dense_block_count, when it has one, must be f — otherwise the
set is refused (65) by name.
It leaves out the engram tables (their loaders mark them lazy: a few rows a token, read through the
mapping) and everything the engine uploads to the card, which it reads once at load, before its timer.
N below block_count + 1 puts whole layers or the output head on the host, a set this rule does not
model: refused (64). One line on stdout:
    host K=<k> layers=<block_count> tensors=<n> bytes=<b> ranges=<r> shards=<s> exps_first=<B> exps_last=<B> dense=<f>
exps_first/exps_last are layer f's and layer K-1's expert bytes (0 for a layer outside the set), dense
the leading dense layers' count f. With --out, the ranges as
`<path>\\t<offset>\\t<length>` lines, contiguous tensors merged, in file order.

`preheat` reads every range of FILE into the page cache: pread into a reused buffer per thread, the data
discarded, T threads (default 16) over chunks of C MiB (default 16). One line on stdout:
    bytes=<b> s=<seconds> gbps=<b / s / 1e9>
A short read (the file is shorter than its header says) is an error, not a smaller total.

`fixture <path>` writes the self-test's two-shard GGUF set at <path> (the first shard's name ends in
-00001-of-00002.gguf): the depth runner's stub test preheats it.

Every tensor's size comes from its type's block size and bytes per block (gguf-py `constants.py`,
GGML_QUANT_SIZES, the ids mainline and ik share); a type not in the table is refused by name. Each
shard's tensors are then checked against their offsets: sorted by offset, every tensor ends at or
before the next one starts and the gap is less than the alignment, and the last ends within an
alignment of the file's end. A header whose sizes do not tile its data is refused (65), never read as
a smaller set.

Exit status: 0; 2 a file missing or unreadable; 64 a usage error or a flag set this rule does not
model; 65 a malformed or inconsistent header.
"""
import os
import re
import struct
import sys
import tempfile
import threading
import time

# (block size in elements, bytes per block): gguf-py constants.py GGML_QUANT_SIZES, the type ids
# mainline and ik share.
QUANT = {
    0: (1, 4),       # F32
    1: (1, 2),       # F16
    2: (32, 18),     # Q4_0
    3: (32, 20),     # Q4_1
    6: (32, 22),     # Q5_0
    7: (32, 24),     # Q5_1
    8: (32, 34),     # Q8_0
    9: (32, 36),     # Q8_1
    10: (256, 84),   # Q2_K
    11: (256, 110),  # Q3_K
    12: (256, 144),  # Q4_K
    13: (256, 176),  # Q5_K
    14: (256, 210),  # Q6_K
    15: (256, 292),  # Q8_K
    16: (256, 66),   # IQ2_XXS
    17: (256, 74),   # IQ2_XS
    18: (256, 98),   # IQ3_XXS
    19: (256, 50),   # IQ1_S
    20: (32, 18),    # IQ4_NL
    21: (256, 110),  # IQ3_S
    22: (256, 82),   # IQ2_S
    23: (256, 136),  # IQ4_XS
    24: (1, 1),      # I8
    25: (1, 2),      # I16
    26: (1, 4),      # I32
    27: (1, 8),      # I64
    28: (1, 8),      # F64
    29: (256, 56),   # IQ1_M
    30: (1, 2),      # BF16
    39: (32, 17),    # MXFP4
}
# Metadata value types (GGUF v2/v3): scalar struct formats; 8 is a string, 9 an array.
SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
EXPS = re.compile(r"blk\.(\d+)\.ffn_(up|down|gate|gate_up)_(ch|)exps")
SPLIT = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")
DEFAULT_ALIGNMENT = 32


class Refusal(Exception):
    """A named refusal: the message and the exit status."""

    def __init__(self, code, msg):
        super().__init__(msg)
        self.code = code


def header(path):
    """(metadata, tensors, data_start, file_size) of one GGUF file; tensors are (name, type, dims, offset)."""
    try:
        f = open(path, "rb")
    except OSError as e:
        raise Refusal(2, f"{path}: {e.strerror}") from e
    with f:
        size = os.fstat(f.fileno()).st_size

        def take(n):
            b = f.read(n)
            if len(b) != n:
                raise Refusal(65, f"{path}: the header ends at byte {f.tell()}, inside a field")
            return b

        def u32():
            return struct.unpack("<I", take(4))[0]

        def u64():
            return struct.unpack("<Q", take(8))[0]

        def string():
            n = u64()
            if n > size:
                raise Refusal(65, f"{path}: a string of {n} bytes in a file of {size}")
            return take(n)

        def value(t):
            if t in SCALAR:
                fmt = SCALAR[t]
                return struct.unpack(fmt, take(struct.calcsize(fmt)))[0]
            if t == 8:
                return string()
            if t == 9:
                et, cnt = u32(), u64()
                if et in SCALAR:
                    step = struct.calcsize(SCALAR[et])
                    if cnt * step > size:
                        raise Refusal(65, f"{path}: an array of {cnt} values in a file of {size}")
                    f.seek(cnt * step, 1)
                else:
                    for _ in range(cnt):
                        value(et)
                return None
            raise Refusal(65, f"{path}: a metadata value of type {t}")

        if take(4) != b"GGUF":
            raise Refusal(65, f"{path} is not a GGUF file")
        version = u32()
        if version not in (2, 3):
            raise Refusal(65, f"{path}: GGUF version {version} (this reader knows 2 and 3)")
        n_tensors, n_kv = u64(), u64()
        meta = {}
        for _ in range(n_kv):
            k = string().decode("utf-8", "replace")
            meta[k] = value(u32())
        tensors = []
        for _ in range(n_tensors):
            name = string().decode("utf-8", "replace")
            nd = u32()
            dims = [u64() for _ in range(nd)]
            tensors.append((name, u32(), dims, u64()))
        align = meta.get("general.alignment", DEFAULT_ALIGNMENT)
        if not isinstance(align, int) or align <= 0 or align & (align - 1):
            raise Refusal(65, f"{path}: general.alignment {align!r} is not a power of two")
        start = (f.tell() + align - 1) // align * align
        return meta, tensors, start, size, align


def nbytes(path, name, ty, dims):
    """A tensor's data bytes from its type and shape."""
    if ty not in QUANT:
        raise Refusal(65, f"{path}: tensor {name} has type id {ty}, which this reader's table does not have")
    bs, ts = QUANT[ty]
    n = 1
    for d in dims:
        n *= d
    if not dims or dims[0] % bs:
        raise Refusal(65, f"{path}: tensor {name} rows of {dims[0] if dims else 0} are not whole blocks of {bs}")
    return n // bs * ts


def shard_paths(first, meta):
    """Every shard of the set the first shard opens, in order."""
    count = meta.get("split.count")
    if count is None:
        return [first]
    m = SPLIT.search(first)
    if not m or int(m.group(1)) != 1 or int(m.group(2)) != count:
        raise Refusal(65, f"{first}: split.count {count}, but the name is not the first of -NNNNN-of-{count:05d}.gguf")
    return [first[: m.start()] + f"-{i:05d}-of-{count:05d}.gguf" for i in range(1, count + 1)]


def tensors_of(first):
    """(block_count, shards, [(path, name, abs offset, bytes)], leading_dense_block_count or None) over the
    whole split set, sizes checked."""
    meta, _, _, _, _ = header(first)
    arch = meta.get("general.architecture")
    arch = arch.decode() if isinstance(arch, bytes) else arch
    layers = meta.get(f"{arch}.block_count")
    if not isinstance(layers, int):
        raise Refusal(65, f"{first}: no {arch}.block_count")
    dense = meta.get(f"{arch}.leading_dense_block_count")
    if dense is not None and not isinstance(dense, int):
        raise Refusal(65, f"{first}: {arch}.leading_dense_block_count {dense!r} is not an integer")
    shards = shard_paths(first, meta)
    out = []
    for i, p in enumerate(shards):
        m, ts, start, size, align = header(p)
        if len(shards) > 1 and m.get("split.no") != i:
            raise Refusal(65, f"{p}: split.no {m.get('split.no')}, expected {i}")
        rows = sorted((off, name, nbytes(p, name, ty, dims)) for name, ty, dims, off in ts)
        for j, (off, name, n) in enumerate(rows):
            if off % align:
                raise Refusal(65, f"{p}: tensor {name} at data offset {off}, not a multiple of {align}")
            end = off + n
            nxt = rows[j + 1][0] if j + 1 < len(rows) else size - start
            if end > nxt or nxt - end >= align:
                raise Refusal(65, f"{p}: tensor {name} is {n} B at {off}, but the next starts at {nxt}: "
                                  "the sizes do not tile the data")
            out.append((p, name, start + off, n))
    return layers, len(shards), out, dense


def host_set(first, k, ngl):
    """The --n-cpu-moe K host set: (layers, shards, [(path, name, offset, bytes)], exps bytes per layer, f),
    f the first layer with routed experts. Layers 0..f-1 are the leading dense layers: no expert tensor
    of the file names them, so the override list matches nothing there and they add nothing to the set.
    Every layer f..K-1 must have experts; the header's leading_dense_block_count, when it has one, must
    be f."""
    layers, shards, ts, dense = tensors_of(first)
    if k < 0 or k > layers:
        raise Refusal(64, f"--n-cpu-moe {k} outside 0..{layers} (block_count)")
    if ngl is not None and ngl < layers + 1:
        raise Refusal(64, f"-ngl {ngl} < block_count + 1 = {layers + 1}: whole layers or the head on the host, "
                          "a set this rule does not model")
    have = sorted({int(m.group(1)) for m in (EXPS.search(t[1]) for t in ts) if m})
    if k and not have:
        raise Refusal(65, f"{first}: no expert tensors in the file, so --n-cpu-moe {k} places nothing")
    f = have[0] if have else layers
    if dense is not None and dense != f:
        raise Refusal(65, f"{first}: leading_dense_block_count {dense}, but the first layer with expert "
                          f"tensors is {f}")
    sel, per = [], {}
    for p, name, off, n in ts:
        m = EXPS.search(name)
        if (m and int(m.group(1)) < k) or name == "token_embd.weight":
            sel.append((p, name, off, n))
            if m:
                per[int(m.group(1))] = per.get(int(m.group(1)), 0) + n
    missing = [i for i in range(f, k) if i not in per]
    if missing:
        raise Refusal(65, f"{first}: no expert tensors in layer(s) {','.join(map(str, missing))} after the "
                          f"first expert layer {f}, inside --n-cpu-moe {k}: a dense layer that is not "
                          "leading, a set this rule does not model")
    return layers, shards, sel, per, f


def merge(sel):
    """Contiguous (path, offset, length) ranges in file order."""
    out = []
    for p, _, off, n in sorted(sel, key=lambda t: (t[0], t[2])):
        if out and out[-1][0] == p and out[-1][1] + out[-1][2] >= off:
            last = out[-1]
            out[-1] = (p, last[1], max(last[1] + last[2], off + n) - last[1])
        else:
            out.append((p, off, n))
    return out


def cmd_host(argv):
    first, k, ngl, out = None, None, None, None
    it = iter(argv)
    for a in it:
        if a == "--n-cpu-moe":
            k = int(next(it))
        elif a == "--ngl":
            ngl = int(next(it))
        elif a == "--out":
            out = next(it)
        elif first is None and not a.startswith("-"):
            first = a
        else:
            raise Refusal(64, f"host: unexpected argument {a!r}")
    if first is None or k is None:
        raise Refusal(64, "usage: gguf-ranges.py host <first shard> --n-cpu-moe K [--ngl N] [--out FILE]")
    layers, shards, sel, per, dense = host_set(first, k, ngl)
    ranges = merge(sel)
    total = sum(n for _, _, _, n in sel)
    if sum(r[2] for r in ranges) != total:
        raise Refusal(65, f"{first}: selected tensors overlap ({total} B selected, {sum(r[2] for r in ranges)} B of ranges)")
    if out:
        with open(out, "w") as f:
            for p, off, n in ranges:
                f.write(f"{p}\t{off}\t{n}\n")
    print(f"host K={k} layers={layers} tensors={len(sel)} bytes={total} ranges={len(ranges)} shards={shards} "
          f"exps_first={per.get(dense, 0)} exps_last={per.get(k - 1, 0)} dense={dense}")


def preheat(ranges, threads, chunk):
    """Read every range once; (bytes, seconds)."""
    work = []
    for p, off, n in ranges:
        for o in range(off, off + n, chunk):
            work.append((p, o, min(chunk, off + n - o)))
    lock = threading.Lock()
    pos = [0]
    done = [0]
    errors = []
    fds = {}
    try:
        for p in {w[0] for w in work}:
            fds[p] = os.open(p, os.O_RDONLY)
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(fds[p], 0, 0, os.POSIX_FADV_SEQUENTIAL)
    except OSError as e:
        for fd in fds.values():
            os.close(fd)
        raise Refusal(2, f"{e.filename}: {e.strerror}") from e

    def run():
        buf = bytearray(chunk)
        view = memoryview(buf)
        got = 0
        while not errors:
            with lock:
                if pos[0] == len(work):
                    break
                p, off, n = work[pos[0]]
                pos[0] += 1
            at = 0
            while at < n:
                try:
                    r = os.preadv(fds[p], [view[: n - at]], off + at)
                except OSError as e:
                    errors.append(f"{p} at {off + at}: {e.strerror}")
                    return
                if r == 0:
                    errors.append(f"{p}: ends before byte {off + n} (a short read at {off + at})")
                    return
                at += r
            got += n
        with lock:
            done[0] += got

    t0 = time.monotonic()
    pool = [threading.Thread(target=run) for _ in range(max(1, threads))]
    for t in pool:
        t.start()
    for t in pool:
        t.join()
    dt = time.monotonic() - t0
    for fd in fds.values():
        os.close(fd)
    if errors:
        raise Refusal(2, "preheat: " + errors[0])
    return done[0], dt


def read_ranges(path):
    ranges = []
    try:
        with open(path) as f:
            for i, line in enumerate(f, 1):
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 3 or not parts[1].isdigit() or not parts[2].isdigit():
                    raise Refusal(65, f"{path}:{i}: not `<path>\\t<offset>\\t<length>`")
                ranges.append((parts[0], int(parts[1]), int(parts[2])))
    except OSError as e:
        raise Refusal(2, f"{path}: {e.strerror}") from e
    return ranges


def cmd_preheat(argv):
    path, threads, chunk_mib = None, 16, 16
    it = iter(argv)
    for a in it:
        if a == "--threads":
            threads = int(next(it))
        elif a == "--chunk-mib":
            chunk_mib = int(next(it))
        elif path is None and not a.startswith("-"):
            path = a
        else:
            raise Refusal(64, f"preheat: unexpected argument {a!r}")
    if path is None or threads < 1 or chunk_mib < 1:
        raise Refusal(64, "usage: gguf-ranges.py preheat FILE [--threads T >= 1] [--chunk-mib C >= 1]")
    ranges = read_ranges(path)
    want = sum(r[2] for r in ranges)
    got, dt = preheat(ranges, threads, chunk_mib << 20)
    if got != want:
        raise Refusal(65, f"preheat read {got} B of {want}")
    print(f"bytes={got} s={dt:.2f} gbps={got / dt / 1e9 if dt > 0 else 0:.2f}")


# ---- the fixture and the self-test ----

def _kv_string(k, v):
    kb, vb = k.encode(), v.encode()
    return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 8) + struct.pack("<Q", len(vb)) + vb


def _kv_scalar(k, t, v):
    kb = k.encode()
    return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", t) + struct.pack(SCALAR[t], v)


def write_gguf(path, kvs, tensors, align=32, trailer=0):
    """A GGUF v3 file: kvs are ready-packed pairs, tensors (name, type, dims, data bytes) laid out in order."""
    info, data = b"", b""
    for name, ty, dims, blob in tensors:
        pad = (-len(data)) % align
        data += b"\0" * pad
        nb = name.encode()
        info += struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(dims))
        info += b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<I", ty) + struct.pack("<Q", len(data))
        data += blob
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)) + b"".join(kvs) + info
    head += b"\0" * ((-len(head)) % align)
    with open(path, "wb") as f:
        f.write(head + data + b"\0" * trailer)


def fixture(first):
    """The two-shard set: shard 1 holds token_embd, layer 0's experts and a card tensor; shard 2 layer 1's
    experts (Q4_K) and the output. block_count 2."""
    m = SPLIT.search(first)
    if not m or m.group(1) != "00001" or m.group(2) != "00002":
        raise Refusal(64, "fixture: the path ends in -00001-of-00002.gguf")
    second = first[: m.start()] + "-00002-of-00002.gguf"
    base = [_kv_string("general.architecture", "tst"), _kv_scalar("tst.block_count", 4, 2),
            _kv_scalar("split.count", 2, 2)]
    t1 = [("token_embd.weight", 0, [8, 3], b"\1" * 96),
          ("blk.0.ffn_gate_exps.weight", 11, [256, 2, 3], b"\2" * 660),
          ("blk.0.ffn_down_exps.weight", 11, [256, 2, 3], b"\3" * 660),
          ("blk.0.attn_q.weight", 1, [16, 4], b"\4" * 128)]
    t2 = [("blk.1.ffn_up_exps.weight", 12, [256, 1, 3], b"\5" * 432),
          ("blk.1.ffn_gate_up_exps.weight", 12, [256, 1, 2], b"\6" * 288),
          ("output.weight", 0, [8, 2], b"\7" * 64)]
    write_gguf(first, base + [_kv_scalar("split.no", 2, 0)], t1, trailer=5)
    write_gguf(second, base + [_kv_scalar("split.no", 2, 1)], t2)
    return first, second


def glm_fixture(path, dense_key=2, exps_layers=(2, 3, 4)):
    """A one-file GLM-shaped model: block_count 5, layers 0-1 dense (their FFN is plain ffn_up/down/gate),
    routed experts (Q4_K) and shared experts (ffn_up_shexp, not an override target) in exps_layers, the
    last of them standing for GLM's NextN block. dense_key is the header's leading_dense_block_count,
    None for no key."""
    kvs = [_kv_string("general.architecture", "glmt"), _kv_scalar("glmt.block_count", 4, 5)]
    if dense_key is not None:
        kvs.append(_kv_scalar("glmt.leading_dense_block_count", 4, dense_key))
    ts = [("token_embd.weight", 0, [8, 3], b"\1" * 96)]
    for i in range(5):
        ts.append((f"blk.{i}.attn_q.weight", 1, [16, 2], b"\4" * 64))
        if i in exps_layers:
            ts.append((f"blk.{i}.ffn_up_shexp.weight", 1, [16, 2], b"\3" * 64))
            ts += [(f"blk.{i}.ffn_{w}_exps.weight", 12, [256, 1, 2], bytes([16 + i]) * 288)
                   for w in ("gate", "up", "down")]
        else:
            ts += [(f"blk.{i}.ffn_{w}.weight", 1, [16, 2], b"\2" * 64) for w in ("gate", "up", "down")]
    write_gguf(path, kvs, ts)
    return path


def self_test():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(("ok " if cond else "FAIL ") + name + ("" if cond else f": {detail}"))
        ok = ok and cond

    with tempfile.TemporaryDirectory() as d:
        first, second = fixture(os.path.join(d, "m-00001-of-00002.gguf"))
        layers, shards, sel, per, _ = host_set(first, 1, 999)
        names = sorted(n for _, n, _, _ in sel)
        check("K=1 selects token_embd and layer 0's experts",
              names == ["blk.0.ffn_down_exps.weight", "blk.0.ffn_gate_exps.weight", "token_embd.weight"], names)
        check("K=1 bytes", sum(n for *_, n in sel) == 96 + 660 + 660, sum(n for *_, n in sel))
        _, _, sel2, per2, _ = host_set(first, 2, None)
        check("K=2 adds layer 1's up and gate_up (the second shard)",
              per2 == {0: 1320, 1: 720} and sum(n for *_, n in sel2) == 96 + 1320 + 720, (per2, sel2))
        check("K=2 reads both shards", {p for p, *_ in sel2} == {first, second})
        rs = merge(sel2)
        with open(first, "rb") as f:
            blob = f.read()
        p0 = [r for r in rs if r[0] == first]
        # token_embd (96 B) and the gate experts (660 B) touch; the down experts start after 12 B of padding.
        check("shard 1: token_embd and the gate experts merge, the down experts after the padding do not",
              [n for _, _, n in p0] == [756, 660] and p0[1][1] == p0[0][1] + 768, p0)
        _, off, n = p0[0]
        check("the ranges hold token_embd, then layer 0's gate and down experts",
              blob[off:off + 96] == b"\1" * 96 and blob[off + 96:off + 756] == b"\2" * 660
              and blob[p0[1][1]:p0[1][1] + 660] == b"\3" * 660, p0)
        _, _, sel0, _, _ = host_set(first, 0, None)
        check("K=0 is token_embd alone", [n for _, n, _, _ in sel0] == ["token_embd.weight"], sel0)
        for args, code, what in (((first, 3, None), 64, "K past block_count"),
                                 ((first, 1, 2), 64, "-ngl below block_count + 1")):
            try:
                host_set(*args)
                check(f"refuses {what}", False, "no refusal")
            except Refusal as e:
                check(f"refuses {what}", e.code == code, (e.code, str(e)))
        # GLM's shape: leading dense layers hold no expert tensor, so they add nothing and do not count
        # against K; a dense layer after the first expert layer, or a header whose
        # leading_dense_block_count names another layer, is refused.
        glm = glm_fixture(os.path.join(d, "glm.gguf"))
        try:
            layers5, _, selg, perg, dense = host_set(glm, 4, None)
            namesg = sorted(n for _, n, _, _ in selg)
            check("GLM K=4: token_embd and layers 2-3's experts, no dense or shared-expert tensor",
                  dense == 2 and layers5 == 5 and perg == {2: 864, 3: 864} and namesg == sorted(
                      ["token_embd.weight"] + [f"blk.{i}.ffn_{w}_exps.weight" for i in (2, 3)
                                               for w in ("gate", "up", "down")]), (dense, perg, namesg))
        except Refusal as e:
            check("GLM K=4: token_embd and layers 2-3's experts, no dense or shared-expert tensor", False,
                  (e.code, str(e)))
        try:
            _, _, selg5, perg5, _ = host_set(glm, 5, None)
            check("GLM K=5 adds the last block's experts", sorted(perg5) == [2, 3, 4]
                  and sum(n for *_, n in selg5) == 96 + 3 * 864, perg5)
        except Refusal as e:
            check("GLM K=5 adds the last block's experts", False, (e.code, str(e)))
        try:
            _, _, selg1, _, _ = host_set(glm, 1, None)
            check("GLM K=1, inside the dense lead, is token_embd alone",
                  [n for _, n, _, _ in selg1] == ["token_embd.weight"], selg1)
        except Refusal as e:
            check("GLM K=1, inside the dense lead, is token_embd alone", False, (e.code, str(e)))
        try:
            _, _, selk, perk, densek = host_set(glm_fixture(os.path.join(d, "glm-nokey.gguf"), None), 3, None)
            check("GLM with no leading_dense_block_count key: the lead read from the tensors",
                  densek == 2 and perk == {2: 864}, (densek, perk))
        except Refusal as e:
            check("GLM with no leading_dense_block_count key: the lead read from the tensors", False,
                  (e.code, str(e)))
        for path, k, what, frag in (
                (glm_fixture(os.path.join(d, "gap.gguf"), None, (2, 4)), 5,
                 "a dense layer after the first expert layer", "layer(s) 3 after the first expert layer 2"),
                (glm_fixture(os.path.join(d, "key.gguf"), 1), 4,
                 "a leading_dense_block_count the tensors contradict", "leading_dense_block_count 1"),
                (glm_fixture(os.path.join(d, "dense.gguf"), None, ()), 2,
                 "K > 0 on a file with no expert tensor", "no expert tensors in the file")):
            try:
                host_set(path, k, None)
                check(f"refuses {what}", False, "no refusal")
            except Refusal as e:
                check(f"refuses {what}", e.code == 65 and frag in str(e), (e.code, str(e)))
        got, _ = preheat(rs, 3, 64)
        check("preheat reads every byte of the ranges", got == sum(r[2] for r in rs), got)
        out = os.path.join(d, "r.tsv")
        with open(out, "w") as f:
            for p, o, n in rs:
                f.write(f"{p}\t{o}\t{n}\n")
        check("the ranges file reads back", read_ranges(out) == rs)
        try:
            preheat([(second, 0, os.path.getsize(second) + 10)], 2, 64)
            check("a short read is refused", False, "no refusal")
        except Refusal as e:
            check("a short read is refused", e.code == 2 and "ends before" in str(e), str(e))
        # A header whose sizes do not tile its data: a Q3_K tensor declared one block larger than its bytes.
        bad = os.path.join(d, "bad.gguf")
        write_gguf(bad, [_kv_string("general.architecture", "tst"), _kv_scalar("tst.block_count", 4, 1)],
                   [("blk.0.ffn_up_exps.weight", 11, [256, 1, 2], b"\0" * 110), ("x", 0, [8], b"\0" * 32)])
        with open(bad, "r+b") as f:
            raw = f.read()
            i = raw.index(b"blk.0.ffn_up_exps.weight") + len("blk.0.ffn_up_exps.weight") + 4 + 16
            f.seek(i)
            f.write(struct.pack("<Q", 3))
        try:
            tensors_of(bad)
            check("sizes that do not tile the data are refused", False, "no refusal")
        except Refusal as e:
            check("sizes that do not tile the data are refused", e.code == 65 and "tile" in str(e), str(e))
        unk = os.path.join(d, "unk.gguf")
        write_gguf(unk, [_kv_string("general.architecture", "tst"), _kv_scalar("tst.block_count", 4, 1)],
                   [("w", 140, [256], b"\0" * 144)])
        try:
            tensors_of(unk)
            check("an unknown type is refused by name", False, "no refusal")
        except Refusal as e:
            check("an unknown type is refused by name", e.code == 65 and "type id 140" in str(e), str(e))
    return ok


def main(argv):
    try:
        if argv[:1] == ["--self-test"]:
            return 0 if self_test() else 1
        if argv[:1] == ["host"]:
            cmd_host(argv[1:])
        elif argv[:1] == ["preheat"]:
            cmd_preheat(argv[1:])
        elif argv[:1] == ["fixture"] and len(argv) == 2:
            fixture(argv[1])
        else:
            raise Refusal(64, __doc__.split("\n\n")[1])
    except Refusal as e:
        print(f"gguf-ranges.py: {e}", file=sys.stderr)
        return e.code
    except (ValueError, StopIteration) as e:
        print(f"gguf-ranges.py: bad argument ({e or 'missing value'})", file=sys.stderr)
        return 64
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
