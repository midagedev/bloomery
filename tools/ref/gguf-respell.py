#!/usr/bin/env python3
"""A copy of a GGUF set whose architecture takes another spelling, the tensor data untouched.

    tools/ref/gguf-respell.py respell <first shard> <out dir> --from OLD --to NEW [--set NEWKEY=OLDVAL:NEWVAL ...]
    tools/ref/gguf-respell.py --self-test

`respell` writes, for each shard of the set the first shard opens (`split.count`), one file of the same name in <out dir>:
  - a shard with no `OLD.` key and no `general.architecture` is a symlink to the input shard (the set's later shards
    usually hold only `split.*`);
  - any other shard is rewritten: `general.architecture` (which must read OLD) becomes NEW, every `OLD.<key>` becomes
    `NEW.<key>`, and three keys are appended: `bloomery.respell.source` (the first shard's path as given, made absolute),
    `bloomery.respell.source_sha256` (the sha256 of the input shard this file was cut from) and `bloomery.respell.from`
    (OLD). Every other key keeps its order and its value bytes. Each `--set` is the one value change that is not a name:
    a u32 key, named by its new spelling, that must read OLDVAL in the source (`glm5-next.expert_shared_count=1:4`); a
    fourth key, `bloomery.respell.set`, records every set as `key=OLDVAL->NEWVAL`, comma-joined. A `--set` that no
    rewritten shard holds, or whose key is not a u32 reading OLDVAL, is refused by name. Tensor offsets are relative to the data start, so the
    tensor-info block is copied byte for byte and the data region is copied after the header's new padding.
OLD may equal NEW (a copy that keeps the architecture and only changes u32 values): at least one `--set` is then required,
and the three records are appended as for any copy (`bloomery.respell.from` reads the unchanged name):
`respell <file> <out dir> --from qwen35moe --to qwen35moe --set qwen35moe.block_count=41:40 --set qwen35moe.nextn_predict_layers=1:0`
cuts the declared next-token layer off a header whose tensors never held it.
The copy is held to its source before it is moved into place, and anything else that differs is refused by name (65): the
same keys apart from the respelled ones and the three appended, the same tensor names, shapes, types and offsets, and the
same data bytes (the sha256 of the data region of both files, the output's read back from disk). The output directory
must not exist; the files go into `<out dir>.tmp.<pid>`, which is renamed when every shard is done and removed on a refusal.

Exit status: 0; 2 a file missing or unreadable; 64 a usage error; 65 a malformed or inconsistent input or a copy that
differs from its source.
"""
import hashlib
import importlib.util
import os
import shutil
import struct
import sys
import tempfile

_spec = importlib.util.spec_from_file_location("gguf_ranges", os.path.join(os.path.dirname(os.path.abspath(__file__)), "gguf-ranges.py"))
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)  # the one header parser: header(), shard_paths(), Refusal, SCALAR and the test writer
Refusal = R.Refusal
CHUNK = 16 << 20
ADDED = ("bloomery.respell.source", "bloomery.respell.source_sha256", "bloomery.respell.from", "bloomery.respell.set")


def _walk_string(buf, pos):
    (n,) = struct.unpack_from("<Q", buf, pos)
    end = pos + 8 + n
    if end > len(buf):
        raise Refusal(65, "a string runs past the header")
    return end


def _walk_value(buf, pos, t):
    """The offset after one metadata value of type t that starts at pos."""
    if t in R.SCALAR:
        return pos + struct.calcsize(R.SCALAR[t])
    if t == 8:
        return _walk_string(buf, pos)
    if t == 9:
        et, cnt = struct.unpack_from("<IQ", buf, pos)
        pos += 12
        if et in R.SCALAR:
            return pos + cnt * struct.calcsize(R.SCALAR[et])
        for _ in range(cnt):
            pos = _walk_value(buf, pos, et)
        return pos
    raise Refusal(65, f"a metadata value of type {t}")


def raw_header(path):
    """(version, kvs, infos, start, size, align): kvs the [(key bytes, type, raw value bytes)] in file order, infos the
    tensor-info block's raw bytes, start the data region's offset, size the file's size."""
    _, tensors, start, size, align = R.header(path)
    with open(path, "rb") as f:
        buf = f.read(start)
    try:
        version, n_tensors, n_kv = struct.unpack_from("<IQQ", buf, 4)
        pos, kvs = 24, []
        for _ in range(n_kv):
            k0 = pos
            pos = _walk_string(buf, pos)
            key = buf[k0 + 8 : pos]
            (t,) = struct.unpack_from("<I", buf, pos)
            v0 = pos + 4
            pos = _walk_value(buf, v0, t)
            kvs.append((key, t, buf[v0:pos]))
        i0 = pos
        for _ in range(n_tensors):
            pos = _walk_string(buf, pos)
            (nd,) = struct.unpack_from("<I", buf, pos)
            pos += 4 + nd * 8 + 4 + 8
    except (struct.error, Refusal) as e:
        raise Refusal(65, f"{path}: the header does not walk ({e})") from e
    if pos > start or any(buf[pos:start]):
        raise Refusal(65, f"{path}: the header ends at {pos}, and the bytes up to the data start {start} are not zero padding")
    if len(tensors) != n_tensors:
        raise Refusal(65, f"{path}: the header parser and the walker disagree on the tensor count")
    return version, kvs, buf[i0:pos], start, size, align


def _string_kv(key, value):
    kb, vb = key.encode(), value.encode()
    return (kb, 8, struct.pack("<Q", len(vb)) + vb)


def _pack(version, n_tensors, kvs, infos, align):
    head = b"GGUF" + struct.pack("<IQQ", version, n_tensors, len(kvs))
    for key, t, raw in kvs:
        head += struct.pack("<Q", len(key)) + key + struct.pack("<I", t) + raw
    head += infos
    return head + b"\0" * ((-len(head)) % align)


def _hash_range(path, start, hasher):
    with open(path, "rb") as f:
        f.seek(start)
        buf = bytearray(CHUNK)
        view = memoryview(buf)
        while True:
            n = f.readinto(buf)
            if not n:
                return
            hasher.update(view[:n])


def needs_respell(kvs, old):
    return any(k == b"general.architecture" or k.startswith(old.encode() + b".") for k, _, _ in kvs)


def respelled(path, kvs, old, new, source, sha, sets=(), applied=None):
    """The key list of the copy of `kvs`: the old name renamed, the `sets` (NEWKEY, OLDVAL, NEWVAL) applied, the records
    appended. `applied`, when given, collects the keys of `sets` this file held. Refusals name the key."""
    ob, nb = old.encode(), new.encode()
    out = []
    todo = {k.encode(): (o, n) for k, o, n in sets}
    for k, t, raw in kvs:
        if k in (a.encode() for a in ADDED):
            raise Refusal(65, f"{path}: already carries {k.decode()}; a respelled copy is not respelled again")
        if k == b"general.architecture":
            want = struct.pack("<Q", len(ob)) + ob
            if t != 8 or raw != want:
                raise Refusal(65, f"{path}: general.architecture is not the string {old!r}")
            raw = struct.pack("<Q", len(nb)) + nb
        elif k.startswith(ob + b"."):
            k = nb + k[len(ob) :]
        if k in todo:
            o, n = todo[k]
            if t != 4 or struct.unpack("<I", raw)[0] != o:
                raise Refusal(65, f"{path}: --set {k.decode()}: the key is not a u32 reading {o} (type {t}, value bytes {raw.hex()})")
            raw = struct.pack("<I", n)
            if applied is not None:
                applied.add(k.decode())
        out.append((k, t, raw))
    names = [k for k, _, _ in out]
    dup = sorted({k for k in names if names.count(k) > 1})
    if dup:
        raise Refusal(65, f"{path}: the respelled keys collide: {[d.decode() for d in dup]}")
    record = ",".join(f"{k}={o}->{n}" for k, o, n in sets)
    return out + [_string_kv(ADDED[0], source), _string_kv(ADDED[1], sha), _string_kv(ADDED[2], old)] + (
        [_string_kv(ADDED[3], record)] if sets else [])


def check_copy(src, dst, old, new, source, sha, data_sha, kvs_want):
    """Hold the rewritten `dst` to its `src`: keys, tensor infos, data bytes. Refuses by name (65)."""
    version, kvs, infos, start, size, align = raw_header(dst)
    s_version, s_kvs, s_infos, s_start, s_size, s_align = raw_header(src)
    if version != s_version or align != s_align:
        raise Refusal(65, f"{dst}: the version or the alignment differs from {src}")
    if kvs != kvs_want:
        have, want = {k: (t, v) for k, t, v in kvs}, {k: (t, v) for k, t, v in kvs_want}
        diff = sorted(k.decode("utf-8", "replace") for k in set(have) ^ set(want) | {k for k in have if k in want and have[k] != want[k]})
        raise Refusal(65, f"{dst}: the keys differ from the source's respelled keys: {diff or 'their order'}")
    if infos != s_infos:
        raise Refusal(65, f"{dst}: the tensor names, shapes, types or offsets differ from {src}")
    if size - start != s_size - s_start:
        raise Refusal(65, f"{dst}: the data region is {size - start} bytes, the source's {s_size - s_start}")
    h = hashlib.sha256()
    _hash_range(dst, start, h)
    if h.hexdigest() != data_sha:
        raise Refusal(65, f"{dst}: the data region's sha256 {h.hexdigest()} differs from the source's {data_sha}")


def rewrite(src, dst, old, new, source, sets=(), applied=None):
    """Write `dst` from `src` and hold it to the source. The input is read once: the pass that copies the data region
    hashes it (the whole file's sha256 is the record, the region's the check). The record's length is fixed (64 hex
    characters), so the header is written with a placeholder and written again, the same length, once the sha is known."""
    version, kvs, infos, start, size, align = raw_header(src)
    n_tensors = len(R.header(src)[1])
    placeholder = "0" * 64
    head = _pack(version, n_tensors, respelled(src, kvs, old, new, source, placeholder, sets, applied), infos, align)
    whole, data = hashlib.sha256(), hashlib.sha256()
    buf = bytearray(CHUNK)
    view = memoryview(buf)
    with open(src, "rb") as f, open(dst, "wb") as out:
        out.write(head)
        whole.update(f.read(start))
        pos = start
        while pos < size:
            n = f.readinto(buf)
            if not n:
                raise Refusal(2, f"{src}: ends at {pos}, the header says {size}")
            whole.update(view[:n])
            data.update(view[:n])
            out.write(view[:n])
            pos += n
        sha = whole.hexdigest()
        want = respelled(src, kvs, old, new, source, sha, sets)
        final = _pack(version, n_tensors, want, infos, align)
        if len(final) != len(head):
            raise Refusal(65, f"{dst}: the header changed length when the sha256 was written")
        out.seek(0)
        out.write(final)
    check_copy(src, dst, old, new, source, sha, data.hexdigest(), want)
    return sha


def respell(first, out, old, new, sets=()):
    if old == new and not sets:
        raise Refusal(64, "--from and --to are the same name: a copy that changes nothing but the records needs a --set")
    first = os.path.abspath(first)
    out = os.path.abspath(out)
    if os.path.lexists(out):
        raise Refusal(64, f"{out} exists; a respelled copy is written once")
    if not os.path.isdir(os.path.dirname(out)):
        raise Refusal(2, f"{os.path.dirname(out)} is not a directory")
    meta = R.header(first)[0]
    shards = R.shard_paths(first, meta)
    tmp = f"{out}.tmp.{os.getpid()}"
    os.mkdir(tmp)
    done, applied = [], set()
    try:
        for p in shards:
            _, kvs, *_ = raw_header(p)
            dst = os.path.join(tmp, os.path.basename(p))
            if needs_respell(kvs, old):
                sha = rewrite(p, dst, old, new, first, sets, applied)
                done.append(f"{os.path.basename(p)} rewritten (source sha256 {sha})")
            else:
                os.symlink(p, dst)
                done.append(f"{os.path.basename(p)} -> {p}")
        missing = sorted(k for k, _, _ in sets if k not in applied)
        if missing:
            raise Refusal(65, f"--set {missing}: no rewritten shard holds the key")
        os.rename(tmp, out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return done


# ---- the self-test ----

def _fixture(d, name, old):
    """A three-shard set: shard 1 the architecture keys, an array of strings and two tensors; shard 2 `split.*` only;
    shard 3 `general.architecture` and no `OLD.` key."""
    kv_s, kv_u = R._kv_string, R._kv_scalar
    arr = struct.pack("<Q", 5) + b"tokns" + struct.pack("<I", 9) + struct.pack("<IQ", 8, 2)
    arr += struct.pack("<Q", 1) + b"a" + struct.pack("<Q", 3) + b"bcd"
    split = lambda no: [kv_u("split.no", 2, no), kv_u("split.count", 2, 3)]
    sets = os.path.join(d, name)
    os.mkdir(sets)
    base = os.path.join(sets, "m-0000{}-of-00003.gguf")
    blob = bytes(range(256)) * 3
    R.write_gguf(base.format(1), [kv_s("general.architecture", old), kv_u(f"{old}.block_count", 4, 7), arr,
                                  kv_u(f"{old}.rope.scale", 6, 0.5), kv_u("general.alignment", 4, 32)] + split(0)
                 + [kv_u("bloomery.fixture.seed", 4, 1)],
                 [("a.weight", 8, [32, 3], blob[:102]), ("b.weight", 1, [8], blob[:16])], trailer=7)
    R.write_gguf(base.format(2), split(1), [("c.weight", 0, [4], blob[:16])])
    R.write_gguf(base.format(3), [kv_s("general.architecture", old)] + split(2), [("d.weight", 0, [2], blob[:8])])
    return base.format(1)


def self_test():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(f"  {'ok  ' if cond else 'FAIL'} {name}" + ("" if cond else f": {detail}"))
        ok &= bool(cond)

    with tempfile.TemporaryDirectory() as d:
        first = _fixture(d, "src", "oldarch")
        out = os.path.join(d, "out")
        done = respell(first, out, "oldarch", "old-arch")
        s1, s2, s3 = (os.path.join(out, f"m-0000{i}-of-00003.gguf") for i in (1, 2, 3))
        check("the output directory appears whole, no tmp left", os.path.isdir(out) and not [e for e in os.listdir(d) if ".tmp." in e], os.listdir(d))
        check("a shard with no old key is a symlink to the input shard",
              os.path.islink(s2) and os.readlink(s2) == os.path.join(d, "src", "m-00002-of-00003.gguf"), done)
        check("a shard holding only general.architecture is rewritten", os.path.isfile(s3) and not os.path.islink(s3), done)
        meta, tensors, start, size, align = R.header(s1)
        src_meta, src_tensors, src_start, src_size, _ = R.header(first)
        check("general.architecture and every old key renamed",
              meta["general.architecture"] == b"old-arch" and meta["old-arch.block_count"] == 7 and meta["old-arch.rope.scale"] == 0.5
              and not [k for k in meta if k.startswith("oldarch.")], sorted(meta))
        check("the other keys are carried (array, alignment, fixture, split)", "tokns" in meta and meta["bloomery.fixture.seed"] == 1
              and meta["split.count"] == 3 and meta["general.alignment"] == 32, sorted(meta))
        check("the respell record names the source shard 1 and its sha256",
              meta["bloomery.respell.source"] == first.encode() and meta["bloomery.respell.from"] == b"oldarch"
              and meta["bloomery.respell.source_sha256"] == hashlib.sha256(open(first, "rb").read()).hexdigest().encode(), meta)
        check("tensors equal, data region bytes equal, trailer kept", tensors == src_tensors
              and open(s1, "rb").read()[start:] == open(first, "rb").read()[src_start:] and size - start == src_size - src_start,
              (start, src_start))
        check("every record of the third shard is its own source's", R.header(s3)[0]["bloomery.respell.source"] == first.encode()
              and R.header(s3)[0]["bloomery.respell.source_sha256"]
              == hashlib.sha256(open(os.path.join(d, "src", "m-00003-of-00003.gguf"), "rb").read()).hexdigest().encode())
        # refusals, each by name
        for what, fn, frag in (
            ("an existing output directory", lambda: respell(first, out, "oldarch", "old-arch"), "exists"),
            ("an architecture that is not the old spelling", lambda: respell(first, os.path.join(d, "o2"), "wrongarch", "x"), "not the string 'wrongarch'"),
            ("a respelled copy respelled again", lambda: respell(s1, os.path.join(d, "o3"), "old-arch", "x"), "already carries"),
        ):
            try:
                fn()
                check(f"refuses {what}", False, "no refusal")
            except Refusal as e:
                check(f"refuses {what}", frag in str(e), str(e))
        check("a refusal leaves no tmp directory", not [e for e in os.listdir(d) if ".tmp." in e], os.listdir(d))
        # --set: one u32 value change, recorded and held by the copy's check
        out2 = os.path.join(d, "out2")
        respell(first, out2, "oldarch", "old-arch", [("old-arch.block_count", 7, 9)])
        m2 = R.header(os.path.join(out2, "m-00001-of-00003.gguf"))[0]
        check("a --set changes the one value and records it", m2["old-arch.block_count"] == 9
              and m2["bloomery.respell.set"] == b"old-arch.block_count=7->9" and m2["old-arch.rope.scale"] == 0.5, m2)
        check("without a --set the record key is absent", "bloomery.respell.set" not in meta, sorted(meta))
        for what, sets, frag in (
            ("a --set of a key that is not a u32", [("old-arch.rope.scale", 1, 2)], "not a u32 reading 1 (type 6"),
            ("a --set whose old value is not the file's", [("old-arch.block_count", 8, 9)], "not a u32 reading 8"),
            ("a --set of a key no shard holds", [("old-arch.absent", 1, 2)], "no rewritten shard holds the key"),
        ):
            try:
                respell(first, os.path.join(d, "o4"), "oldarch", "old-arch", sets)
                check(f"refuses {what}", False, "no refusal")
            except Refusal as e:
                check(f"refuses {what}", e.code == 65 and frag in str(e), str(e))
        check("a refused --set leaves no output and no tmp directory", not os.path.exists(os.path.join(d, "o4"))
              and not [e for e in os.listdir(d) if ".tmp." in e], os.listdir(d))
        check("the command line takes a --set and refuses a malformed one",
              main(["respell", first, os.path.join(d, "o5"), "--from", "oldarch", "--to", "old-arch", "--set", "old-arch.block_count=7:9"]) == 0
              and main(["respell", first, os.path.join(d, "o6"), "--from", "oldarch", "--to", "old-arch", "--set", "block_count=7:9"]) == 64
              and main(["respell", first, os.path.join(d, "o6"), "--from", "oldarch", "--to", "old-arch", "--set", "old-arch.block_count=7:7"]) == 64)
        # OLD == NEW: the same architecture, u32 values changed (the MiniCPM next-token cut)
        out3 = os.path.join(d, "out3")
        respell(first, out3, "oldarch", "oldarch", [("oldarch.block_count", 7, 5)])
        m3, t3, st3, sz3, _ = R.header(os.path.join(out3, "m-00001-of-00003.gguf"))
        check("a same-name copy changes the --set value, keeps the architecture, records source, from and set",
              m3["general.architecture"] == b"oldarch" and m3["oldarch.block_count"] == 5 and m3["oldarch.rope.scale"] == 0.5
              and m3["bloomery.respell.from"] == b"oldarch" and m3["bloomery.respell.set"] == b"oldarch.block_count=7->5"
              and m3["bloomery.respell.source"] == first.encode() and t3 == src_tensors
              and open(os.path.join(out3, "m-00001-of-00003.gguf"), "rb").read()[st3:] == open(first, "rb").read()[src_start:], m3)
        for what, fn, frag in (
            ("a same-name respell with no --set", lambda: respell(first, os.path.join(d, "o7"), "oldarch", "oldarch"), "needs a --set"),
        ):
            try:
                fn()
                check(f"refuses {what}", False, "no refusal")
            except Refusal as e:
                check(f"refuses {what}", e.code == 64 and frag in str(e), str(e))
        check("the command line takes --from == --to with a --set and refuses it without one",
              main(["respell", first, os.path.join(d, "o8"), "--from", "oldarch", "--to", "oldarch", "--set", "oldarch.block_count=7:5"]) == 0
              and main(["respell", first, os.path.join(d, "o9"), "--from", "oldarch", "--to", "oldarch"]) == 64
              and not os.path.exists(os.path.join(d, "o9")))
        # FAIL-first of the copy's check: a flipped data byte, a changed tensor offset, a dropped key
        src_version, src_kvs, *_ = raw_header(first)
        want = respelled(first, src_kvs, "oldarch", "old-arch", first, hashlib.sha256(open(first, "rb").read()).hexdigest())
        data_sha = hashlib.sha256(open(first, "rb").read()[src_start:]).hexdigest()
        sha = hashlib.sha256(open(first, "rb").read()).hexdigest()
        check_copy(first, s1, "oldarch", "old-arch", first, sha, data_sha, want)
        set_first = os.path.join(out2, "m-00001-of-00003.gguf")
        want_set = respelled(first, src_kvs, "oldarch", "old-arch", first, sha, [("old-arch.block_count", 7, 9)])
        check_copy(first, set_first, "oldarch", "old-arch", first, sha, data_sha, want_set)
        try:
            check_copy(first, s1, "oldarch", "old-arch", first, sha, data_sha, want_set)
            check("the copy's check refuses a copy without the set the run named", False, "no refusal")
        except Refusal as e:
            check("the copy's check refuses a copy without the set the run named", e.code == 65 and "old-arch.block_count" in str(e), str(e))
        for what, mutate, frag in (
            ("a flipped data byte", lambda b: b[:-20] + bytes([b[-20] ^ 1]) + b[-19:], "data region's sha256"),
            ("a changed tensor-info byte", lambda b: b.replace(b"a.weight", b"z.weight", 1), "tensor names, shapes, types or offsets"),
            ("a changed value of a carried key", lambda b: b.replace(b"bloomery.fixture.seed\x04\x00\x00\x00\x01", b"bloomery.fixture.seed\x04\x00\x00\x00\x02", 1), "bloomery.fixture.seed"),
        ):
            bad = os.path.join(d, "bad.gguf")
            with open(s1, "rb") as f:
                raw = f.read()
            with open(bad, "wb") as f:
                f.write(mutate(raw))
            try:
                check_copy(first, bad, "oldarch", "old-arch", first, sha, data_sha, want)
                check(f"the copy's check refuses {what}", False, "no refusal")
            except Refusal as e:
                check(f"the copy's check refuses {what}", e.code == 65 and frag in str(e), str(e))
    return ok


def main(argv):
    try:
        if argv[:1] == ["--self-test"]:
            return 0 if self_test() else 1
        if argv[:1] == ["respell"] and len(argv) >= 7 and argv[3] == "--from" and argv[5] == "--to":
            _, first, out, _, old, _, new, *rest = argv
            if not old or not new or "." in old + new:
                raise Refusal(64, "--from and --to are names without a dot")
            sets = []
            while rest:
                if rest[0] != "--set" or len(rest) < 2:
                    raise Refusal(64, f"respell: unexpected argument {rest[0]!r}")
                key, _, vals = rest[1].partition("=")
                o, _, n = vals.partition(":")
                if not key.startswith(new + ".") or not o.isdigit() or not n.isdigit() or int(n) >= 1 << 32 or int(o) == int(n):
                    raise Refusal(64, f"--set wants <{new}.key>=<old u32>:<new u32>, two different values, got {rest[1]!r}")
                sets.append((key, int(o), int(n)))
                rest = rest[2:]
            if old == new and not sets:
                raise Refusal(64, "--from and --to are the same name: give a --set, or two different names")
            for line in respell(first, out, old, new, sets):
                print(f"respell: {line}")
            print(f"respell: {out}")
        else:
            raise Refusal(64, __doc__.split("\n\n")[0] + "\n" + "\n".join(__doc__.split("\n\n")[1].splitlines()[:2]))
    except Refusal as e:
        print(f"gguf-respell.py: {e}", file=sys.stderr)
        return e.code
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
