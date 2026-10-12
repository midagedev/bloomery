#!/usr/bin/env python3
"""Census of the GGUF files users download, by header read only: which ggml type each tensor of a set holds.

    tools/type-census.py OUT_DIR [--stubs DIR] REPO...
    tools/type-census.py pin MANIFEST REPOS_FILE
    tools/type-census.py build ROOT MANIFEST
    tools/type-census.py --self-test    offline: the parse, the grouping, the refusals, the stubs, the manifest's rules
                                        and the build on synthetic headers

For each REPO (`owner/name`) it lists the files of the main branch, groups them into sets (the hf crate's split
shards together; `mmproj` files left out, every other .gguf kept: a draft head's file is a file users download), and
reads each shard's header by ranged fetches (curl, tools/bloomery/hf.py); no tensor byte is read. It writes
OUT_DIR/<owner>__<name>__<set>.json for each set:

    repo, set, shards (the files, in order), sizes (file -> bytes), kv (the `general.*` keys and the layer and expert
    counts), top (a tensor outside the blocks -> [type, dims]), blk (the tensor name without `blk.<i>.` -> type ->
    the sorted layers)

and prints one line per set:

    <repo> <set> arch=<general.architecture> tensors=<n> shards=<n>

Type names are tools/bloomery/hf.py's ggml names. A tensor type id the table does not name fails its set
(`ggml type id <n> in <file>: not in the type table`), never a placeholder name.

--stubs DIR writes, for each shard of a set that read whole, its header bytes up to the tensor data (padded to
`general.alignment`) to DIR/<repo>/<file>, then truncates the file to the shard's real size: a sparse file the GGUF
reader opens with every tensor in place and every tensor byte zero, for plan-only probes.

`pin` resolves the commit each repo of REPOS_FILE (one `owner/name` a line, `#` comments and blank lines skipped)
points to, lists the repo at that commit and reads the header of every file a manifest holds (the census's files but
an importance matrix; a draft head's file is its own set) at that commit. It writes MANIFEST whole or not at all, a
tab-separated table of the columns in MANIFEST_COLUMNS, one row per file, sorted by (org, repo, file). The table is
the contract with the reader of the stubs (crates/refset), and check_rows holds its rules: header_bytes is the tensor
data offset, header_sha256 the sha256 of the real file's first header_bytes bytes as the hub sent them, file_sha256 the
hub's LFS sha256 of the whole file.

`build` rebuilds the stubs from MANIFEST into ROOT, a directory that does not exist or is empty: for each row the
first header_bytes bytes of the file at its revision, checked against header_sha256, written at
ROOT/<org>/<repo>/<file> and extended (sparse) to file_bytes, then read back and checked again. A set that fails
leaves no file of its own behind, the other sets go on, and the run ends with 1.

A set that fails prints `<repo> <set>: FAILED <reason>` and writes nothing; a repo that cannot be listed prints
`<repo>: listing failed: <reason>`. Exit status: 0 everything read (or built); 1 a set, a listing or an output name
failed; 64 a usage error; 65 an input refused by name (a manifest that breaks a rule, a ROOT that is not new).
"""
import collections
import concurrent.futures
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import re
import struct
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_by_path(name, *rel):
    """A module of this tree, loaded by path (tools/ has no packages), leaving no bytecode behind."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, *rel))
    mod = importlib.util.module_from_spec(spec)
    keep, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = keep
    return mod


hf = load_by_path("bloomery_hf", "tools", "bloomery", "hf.py")

POOL = 8
HEADER_BUDGET = 64 * hf.MIB  # a header past this is an error, not a longer wait
KV_SUFFIXES = ("expert_count", "expert_used_count", "block_count", "nextn_predict_layers", "leading_dense_block_count")
BLOCK = re.compile(r"blk\.(\d+)\.(.*)")
NAME = r"[A-Za-z0-9][A-Za-z0-9._-]*"
REPO = re.compile(f"{NAME}/{NAME}")
USAGE = ("usage: type-census.py OUT_DIR [--stubs DIR] REPO... | pin MANIFEST REPOS_FILE | build ROOT MANIFEST | "
         "--self-test")
CHUNK = 16 * hf.MIB  # the most one build read asks for


def census_file(name):
    """The files a census reads: every .gguf but a multimodal projector (the hf crate's `is_model` also drops the
    draft and importance-matrix files, which are files users download)."""
    return name.endswith(".gguf") and "mmproj" not in name.lower()


def manifest_file(name):
    """The files a manifest holds: the census's files but an importance matrix (a draft head's file stays, as its own
    set)."""
    return census_file(name) and "imatrix" not in name.lower()


def kv_of(meta):
    """The metadata a census keeps: `general.*` and the layer and expert counts, a string under 200 characters or a
    scalar; an array is left out."""
    kv = {}
    for k, v in meta.items():
        if not (k.startswith("general.") or k.endswith(KV_SUFFIXES)):
            continue
        if isinstance(v, bytes):
            v = v.decode("utf-8", "replace")
            if len(v) >= 200:
                continue
        elif v is None:
            continue
        kv[k] = v
    return kv


def grouped(tensors):
    """(top, blk) of [(name, type name, dims)]: a tensor outside the blocks -> [type, dims]; the rest keyed by the
    name without `blk.<i>.`, then by type, then the sorted layers."""
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    top = {}
    for name, ty, dims in tensors:
        m = BLOCK.match(name)
        if m:
            by[m.group(2)][ty].append(int(m.group(1)))
        else:
            top[name] = [ty, dims]
    return top, {k: {ty: sorted(ls) for ty, ls in v.items()} for k, v in sorted(by.items())}


def data_offset(local, size):
    """The offset of the tensor data of a file of `size` bytes whose first bytes are in `local`: the header and its
    padding to `general.alignment`."""
    return hf.gguf_ranges().header(local, size=size)[2]


def stub_head(local, size):
    """The header bytes of the prefix file `local` (of a file of `size` bytes) up to the tensor data, zero-padded to
    `general.alignment`."""
    start = data_offset(local, size)
    with open(local, "rb") as f:
        head = f.read(start)
    return head + bytes(start - len(head))


def header_prefix(fetch, local, size):
    """The real bytes [0, tensor data offset) of a file of `size` bytes, read through `fetch(start, end)`: the header
    as hf.read_header reads it, then the padding to `general.alignment` when those reads stopped short of it. Nothing
    is synthesized. A file with no tensor (the metadata-only first shard of a split set) ends where its header does,
    short of the padding its data offset would add: its prefix is the whole file."""
    meta, tensors, got = hf.read_header(fetch, local, size, HEADER_BUDGET)
    if meta is None:
        raise hf.Red(f"the header is unfinished at {got} bytes, the most this reads")
    start = data_offset(local, size)
    if start > size and not tensors:
        start = size
    if start > size:
        raise hf.Red(f"the header and its padding take {start} bytes, more than the file's {size}")
    with open(local, "rb") as f:
        head = f.read(start)
    if len(head) < start:
        head += fetch(len(head), start)
    return head


def stub_path(stubs, repo, path):
    """DIR/<repo>/<file>; a hub path that is not a plain relative one is a Red."""
    parts = path.split("/")
    if path.startswith("/") or any(p in ("", ".", "..") for p in parts):
        raise hf.Red(f"{repo}: the file path {path!r} is not a plain relative path, so no stub is written for it")
    return os.path.join(stubs, repo, *parts)


def write_stub(fn, head, size):
    """`head` at the start of `fn`, the file extended to `size` bytes (sparse, every other byte zero)."""
    if len(head) > size:
        raise hf.Red(f"{fn}: the header and its padding take {len(head)} bytes, more than the file's {size}")
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    with open(fn, "wb") as w:
        w.write(head)
        w.truncate(size)


def out_name(repo, name):
    return f"{repo.replace('/', '__')}__{os.path.basename(name)}.json"


def read_shard(hub, repo, entry, scratch, want_head):
    """(kv, [(tensor name, type name, dims)], stub head or None) of one shard, its header read through `hub`."""
    path, size = entry.path, entry.size
    local = os.path.join(scratch, "header.bin")
    try:
        meta, tensors, got = hf.read_header(lambda a, b: hub.range(repo, path, a, b), local, size, HEADER_BUDGET)
        if meta is None:
            raise hf.Red(f"the header is unfinished at {got} bytes, the most this reads")
        named = [(name, hf.ggml_type_name(ty, path), dims) for name, ty, dims, _ in tensors]
        return kv_of(meta), named, (stub_head(local, size) if want_head else None)
    except hf.Red as e:
        if path in str(e):
            raise
        raise hf.Red(f"{path}: {e}") from e


def census_set(make_hub, repo, s, out, stubs):
    """Read every shard of the set `s` (hf.gguf_sets' dict), write its JSON (and stubs), return its printed line."""
    files = s["files"]
    targets = [stub_path(stubs, repo, e.path) for e in files] if stubs is not None else []
    kv, tensors, heads = {}, [], []
    with tempfile.TemporaryDirectory() as scratch:
        hub = make_hub(scratch)
        for e in files:
            k, t, head = read_shard(hub, repo, e, scratch, stubs is not None)
            kv.update(k)
            tensors += t
            heads.append(head)
    for fn, head, e in zip(targets, heads, files):
        write_stub(fn, head, e.size)
    top, blk = grouped(tensors)
    rec = {"repo": repo, "set": s["name"], "shards": [e.path for e in files], "sizes": {e.path: e.size for e in files},
           "kv": kv, "top": top, "blk": blk}
    with open(os.path.join(out, out_name(repo, s["name"])), "w") as f:
        json.dump(rec, f, indent=1)
    return (f"{repo} {os.path.basename(s['name'])} arch={kv.get('general.architecture')} tensors={len(tensors)} "
            f"shards={len(files)}")


def one_line(e):
    return " ".join(str(e).split())


def run(out, stubs, repos, make_hub=hf.Hub):
    """Census `repos` into `out`; the exit status."""
    os.makedirs(out, exist_ok=True)
    hf.gguf_ranges()  # loaded here, before the pool's threads ask
    failed, jobs, claimed = 0, [], {}
    with tempfile.TemporaryDirectory() as scratch:
        hub = make_hub(scratch)
        for repo in repos:
            try:
                sets = hf.gguf_sets(repo, hub.listing(repo), census_file)
                if not sets:
                    raise hf.Red(f"{repo} holds no GGUF file this reads")
            except (hf.Red, hf.Refuse) as e:
                print(f"{repo}: listing failed: {one_line(e)}", flush=True)
                failed += 1
                continue
            for s in sets:
                fn = out_name(repo, s["name"])
                if fn in claimed:
                    print(f"{repo} {s['name']}: FAILED its output file {fn} is the one of the set {claimed[fn]}",
                          flush=True)
                    failed += 1
                    continue
                claimed[fn] = s["name"]
                jobs.append((repo, s))
    with concurrent.futures.ThreadPoolExecutor(POOL) as ex:
        futs = {ex.submit(census_set, make_hub, repo, s, out, stubs): (repo, s["name"]) for repo, s in jobs}
        for fu in concurrent.futures.as_completed(futs):
            try:
                print(fu.result(), flush=True)
            except (hf.Red, hf.Refuse, OSError) as e:
                print(f"{futs[fu][0]} {futs[fu][1]}: FAILED {one_line(e)}", flush=True)
                failed += 1
            except Exception as e:
                print(f"{futs[fu][0]} {futs[fu][1]}: FAILED {type(e).__name__}: {one_line(e)}", flush=True)
                failed += 1
    return 1 if failed else 0


# ------------------------------------------------------------------------------------------------
# The manifest, tools/census/stubs.tsv: the contract with the reader of the stubs (crates/refset).

MANIFEST_COLUMNS = ("org", "repo", "revision", "set", "file", "file_bytes", "file_sha256", "header_bytes",
                    "header_sha256")
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
DECIMAL = re.compile(r"0|[1-9][0-9]*")
CONTROL = re.compile(r"[\x00-\x1f\x7f\\]")


# The rules of check_rows, each the name a refusal carries.
RULES = ("fields", "text", "path", "set", "revision", "digest", "size", "duplicate", "shards", "directory",
         "set-revision", "rows")


def rule(name, where, why):
    """A manifest input that breaks the rule `name`: refused by that name (65)."""
    if name not in RULES:
        raise ValueError(f"{name} is not a rule of the manifest")
    raise hf.Refuse(65, f"census manifest rule {name}: {where}: {why}")


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def set_of(file):
    """(set, shard index, shard count) of a file path: the basename minus `-NNNNN-of-NNNNN.gguf` (five digits each),
    else minus `.gguf`; index and count are None for a file with no such suffix. The path ends in `.gguf`."""
    stem = file.rsplit("/", 1)[-1][:-len(".gguf")]
    parts = hf.split_of(stem)
    return parts if parts else (stem, None, None)


def dir_of(file):
    return file.rsplit("/", 1)[0] if "/" in file else ""


def set_groups(rows):
    """The rows by set identity (org, repo, directory, set), each set at its first row's place."""
    groups = {}
    for r in rows:
        groups.setdefault((r["org"], r["repo"], dir_of(r["file"]), r["set"]), []).append(r)
    return groups


def set_name(key):
    org, repo, d, name = key
    return f"{org}/{repo} {d + '/' if d else ''}{name}"


def check_rows(rows):
    """Refuse by the rule's name a list of manifest rows (dicts of text, one key per column) that breaks the contract;
    `pin` runs it before it writes and `build` after it reads. RULES names them; `rows` is a manifest with no row."""
    if not rows:
        rule("rows", "the manifest", "it holds no row")
    seen = set()
    for n, r in enumerate(rows, 1):
        if sorted(r) != sorted(MANIFEST_COLUMNS):
            rule("fields", f"row {n}", f"its fields are {sorted(r)}, not {sorted(MANIFEST_COLUMNS)}")
        where = f"row {n} ({r['org']}/{r['repo']} {r['file']})"
        for col, text in r.items():
            if CONTROL.search(text):
                rule("text", where, f"{col} holds a control character or a backslash")
        for col in ("org", "repo"):
            if not re.fullmatch(NAME, r[col]):
                rule("path", where, f"{col} {r[col]!r} is not a plain name (no leading /, no .. component)")
        file = r["file"]
        if file.startswith("/"):
            rule("path", where, "the file has a leading /")
        if not file.endswith(".gguf") or file.rsplit("/", 1)[-1] == ".gguf":
            rule("path", where, "the file is not a name ending in .gguf")
        if any(c in ("", ".", "..") for c in file.split("/")):
            rule("path", where, "the file has an empty, . or .. component")
        want = set_of(file)[0]
        if r["set"] != want:
            rule("set", where, f"the set is {r['set']!r}, the basename minus its shard suffix is {want!r}")
        if not HEX40.fullmatch(r["revision"]):
            rule("revision", where, f"the revision {r['revision']!r} is not 40 lowercase hex digits")
        for col in ("file_sha256", "header_sha256"):
            if not HEX64.fullmatch(r[col]):
                rule("digest", where, f"{col} {r[col]!r} is not 64 lowercase hex digits")
        for col in ("file_bytes", "header_bytes"):
            if not DECIMAL.fullmatch(r[col]):
                rule("size", where, f"{col} {r[col]!r} is not a decimal integer")
        file_bytes, header_bytes = int(r["file_bytes"]), int(r["header_bytes"])
        if not 0 < header_bytes <= file_bytes:
            rule("size", where, f"header_bytes {header_bytes} is not within 1..={file_bytes} (file_bytes)")
        key = (r["org"], r["repo"], file)
        if key in seen:
            rule("duplicate", where, "another row has the same org, repo and file")
        seen.add(key)
    groups = set_groups(rows)
    shards_of = collections.defaultdict(list)  # (org, repo, set) -> (directory, index, count) of each suffixed row
    for (org, repo, d, name), members in groups.items():
        for r in members:
            _, index, count = set_of(r["file"])
            if count is not None:
                shards_of[(org, repo, name)].append((d, index, count))
    for key, members in groups.items():
        where = f"set {set_name(key)}"
        revisions = sorted({r["revision"] for r in members})
        if len(revisions) > 1:
            rule("set-revision", where, f"its rows name the revisions {revisions}; a set has one")
        suffixed = [set_of(r["file"])[1:] for r in members if set_of(r["file"])[1] is not None]
        if not suffixed:
            continue
        if len(suffixed) != len(members):
            rule("shards", where, "it holds a file with a shard suffix and a file without one")
        counts = sorted({c for _, c in suffixed})
        if len(counts) > 1:
            rule("shards", where, f"its shards name the counts {counts}; a set has one")
        count = counts[0]
        have = sorted(i for i, _ in suffixed)
        need = list(range(1, count + 1))
        if have != need:
            missing = [i for i in need if i not in have]
            apart = sorted({d for d, i, c in shards_of[key[:2] + (key[3],)] if d != key[2] and c == count and i in missing})
            if apart:
                rule("directory", where, f"shards {missing} are in {apart}, not in {key[2] or 'the repo root'}; the "
                                         "shards of a set share one directory")
            rule("shards", where, f"its shards are {have} of {count}; a set has exactly 1..={count}")


def manifest_text(rows):
    lines = ["\t".join(MANIFEST_COLUMNS)] + ["\t".join(r[c] for c in MANIFEST_COLUMNS) for r in rows]
    return "\n".join(lines) + "\n"


def write_manifest(path, rows):
    """`rows` sorted by (org, repo, file), checked, and written to `path` whole or not at all (a temp file beside it,
    then a rename); the sorted rows."""
    rows = sorted(rows, key=lambda r: (r["org"], r["repo"], r["file"]))
    check_rows(rows)
    fd, tmp = tempfile.mkstemp(prefix=".stubs-", suffix=".tmp", dir=os.path.dirname(os.path.abspath(path)))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(manifest_text(rows))
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise
    return rows


def read_manifest(path):
    """The rows (dicts of text) of the manifest at `path`, after check_rows; the columns are read by name."""
    try:
        with open(path, encoding="utf-8", newline="") as f:
            text = f.read()
    except OSError as e:
        raise hf.Refuse(65, f"{path}: {e.strerror}") from e
    except UnicodeDecodeError as e:
        rule("text", path, f"it is not UTF-8: {e}")
    if not text.endswith("\n"):
        rule("fields", path, "the last line has no newline (a truncated file)")
    lines = text[:-1].split("\n")
    head = lines[0].split("\t")
    if sorted(head) != sorted(MANIFEST_COLUMNS):
        rule("fields", path, f"the header line has the columns {head}, not {list(MANIFEST_COLUMNS)} in any order")
    rows = []
    for n, line in enumerate(lines[1:], 2):
        cells = line.split("\t")
        if len(cells) != len(head):
            rule("fields", f"{path}:{n}", f"{len(cells)} fields, the header line has {len(head)}")
        rows.append(dict(zip(head, cells)))
    check_rows(rows)
    return rows


def read_repos(path):
    """The repos of a repos file: one `owner/name` a line, `#` comments and blank lines skipped."""
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError as e:
        raise hf.Refuse(65, f"{path}: {e.strerror}") from e
    repos = []
    for n, line in enumerate(lines, 1):
        text = line.split("#", 1)[0].strip()
        if not text:
            continue
        if not REPO.fullmatch(text):
            raise hf.Refuse(65, f"{path}:{n}: {text!r} is not <owner>/<name>")
        if text in repos:
            raise hf.Refuse(65, f"{path}:{n}: {text} is listed twice")
        repos.append(text)
    if not repos:
        raise hf.Refuse(65, f"{path}: no repo")
    return repos


def pin_row(make_hub, repo, revision, entry):
    """The manifest row of one file of `repo` at `revision`, its header read through a hub of its own."""
    with tempfile.TemporaryDirectory() as scratch:
        hub = make_hub(scratch)
        try:
            head = header_prefix(lambda a, b: hub.range(repo, entry.path, a, b, revision=revision),
                                 os.path.join(scratch, "header.bin"), entry.size)
        except hf.Red as e:
            if entry.path in str(e):
                raise
            raise hf.Red(f"{entry.path}: {e}") from e
    org, name = repo.split("/")
    return {"org": org, "repo": name, "revision": revision, "set": set_of(entry.path)[0], "file": entry.path,
            "file_bytes": str(entry.size), "file_sha256": entry.digest, "header_bytes": str(len(head)),
            "header_sha256": sha256_hex(head)}


def pin(manifest, repos_file, make_hub=hf.Hub):
    """Write `manifest` from the repos of `repos_file`, each read at the commit its main branch points to now; the
    exit status. Any repo or file that fails ends the run with 1 and no manifest."""
    started = time.monotonic()
    repos = read_repos(repos_file)
    hf.gguf_ranges()  # loaded here, before the pool's threads ask
    failed, jobs = 0, []
    with tempfile.TemporaryDirectory() as scratch:
        hub = make_hub(scratch)
        for repo in repos:
            try:
                revision = hub.main_sha(repo)
                sets = hf.gguf_sets(repo, hub.listing(repo, revision=revision), manifest_file)
                files = [e for s in sets for e in s["files"]]
                if not files:
                    raise hf.Red(f"{repo} holds no GGUF file this reads")
                for e in files:
                    if e.check != "sha256":
                        raise hf.Red(f"{repo}: the listing gives {e.path} no LFS sha256 (a {e.check}), so the "
                                     "manifest has no file_sha256 for it")
            except (hf.Red, hf.Refuse) as e:
                print(f"{repo}: listing failed: {one_line(e)}", flush=True)
                failed += 1
                continue
            print(f"{repo} revision={revision} files={len(files)}", flush=True)
            jobs += [(repo, revision, e) for e in files]
    rows = []
    with concurrent.futures.ThreadPoolExecutor(POOL) as ex:
        futs = {ex.submit(pin_row, make_hub, repo, revision, e): (repo, e.path) for repo, revision, e in jobs}
        for fu in concurrent.futures.as_completed(futs):
            try:
                rows.append(fu.result())
            except (hf.Red, hf.Refuse, OSError) as e:
                print(f"{futs[fu][0]} {futs[fu][1]}: FAILED {one_line(e)}", flush=True)
                failed += 1
            except Exception as e:
                print(f"{futs[fu][0]} {futs[fu][1]}: FAILED {type(e).__name__}: {one_line(e)}", flush=True)
                failed += 1
    if failed:
        print(f"pin: {failed} failed; {manifest} is not written", flush=True)
        return 1
    rows = write_manifest(manifest, rows)
    groups = set_groups(rows)
    print(f"pin: {len(rows)} rows, {len(groups)} sets ({sum(len(m) > 1 for m in groups.values())} of several "
          f"shards), header_bytes {sum(int(r['header_bytes']) for r in rows)}, {len(repos)} repos, "
          f"{time.monotonic() - started:.0f} s", flush=True)
    return 0


def fetch_prefix(fetch, n):
    """The bytes [0, n) through `fetch(start, end)`, in reads of at most CHUNK bytes."""
    out = bytearray()
    while len(out) < n:
        out += fetch(len(out), min(n, len(out) + CHUNK))
    return bytes(out)


def build_stub(hub, root, r, written):
    """The stub of the row `r` under `root`; every path it creates is added to `written` first."""
    org, name, file, revision = r["org"], r["repo"], r["file"], r["revision"]
    n, size, want = int(r["header_bytes"]), int(r["file_bytes"]), r["header_sha256"]
    head = fetch_prefix(lambda a, b: hub.range(f"{org}/{name}", file, a, b, revision=revision), n)
    got = sha256_hex(head)
    if got != want:
        raise hf.Red(f"{org}/{name} {file} at {revision}: its first {n} bytes have sha256 {got}, the manifest has "
                     f"{want}")
    fn = stub_path(root, f"{org}/{name}", file)
    if os.path.lexists(fn):
        raise hf.Red(f"{fn} is already there")
    written.append(fn)
    write_stub(fn, head, size)
    with open(fn, "rb") as f:
        back = sha256_hex(f.read(n))
    if os.path.getsize(fn) != size or back != want:
        raise hf.Red(f"{org}/{name} {file}: the stub written has {os.path.getsize(fn)} bytes and a first-{n}-bytes "
                     f"sha256 of {back}; the manifest has {size} bytes and {want}")


def build_set(make_hub, root, members):
    """The stubs of every row of one set; when one fails, the files this set wrote are removed and the error raised."""
    written = []
    try:
        with tempfile.TemporaryDirectory() as scratch:
            hub = make_hub(scratch)
            for r in members:
                build_stub(hub, root, r, written)
    except BaseException:
        for fn in written:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(fn)
        raise


def build(root, manifest, make_hub=hf.Hub):
    """Rebuild the stubs of `manifest` into the new directory `root`; the exit status."""
    rows = read_manifest(manifest)
    if os.path.lexists(root) and (not os.path.isdir(root) or os.listdir(root)):
        raise hf.Refuse(65, f"{root}: ROOT exists and is not an empty directory; a rebuild goes to a new directory")
    os.makedirs(root, exist_ok=True)
    groups = set_groups(rows)
    failed = 0
    with concurrent.futures.ThreadPoolExecutor(POOL) as ex:
        futs = {ex.submit(build_set, make_hub, root, members): key for key, members in groups.items()}
        for fu in concurrent.futures.as_completed(futs):
            label = set_name(futs[fu])
            try:
                fu.result()
                print(f"{label} built files={len(groups[futs[fu]])}", flush=True)
            except (hf.Red, hf.Refuse, OSError) as e:
                print(f"{label}: FAILED {one_line(e)}", flush=True)
                failed += 1
            except Exception as e:
                print(f"{label}: FAILED {type(e).__name__}: {one_line(e)}", flush=True)
                failed += 1
    print(f"build: {len(groups) - failed} of {len(groups)} sets built into {root}", flush=True)
    return 1 if failed else 0


# ------------------------------------------------------------------------------------------------
# The offline test: synthetic headers, an in-memory hub, a stub curl.

def kv_string(key, value):
    k, v = key.encode(), value.encode()
    return struct.pack("<Q", len(k)) + k + struct.pack("<I", 8) + struct.pack("<Q", len(v)) + v


def kv_u32(key, value):
    k = key.encode()
    return struct.pack("<Q", len(k)) + k + struct.pack("<I", 4) + struct.pack("<I", value)


def kv_strings(key, values):
    k = key.encode()
    body = b"".join(struct.pack("<Q", len(v)) + v for v in (s.encode() for s in values))
    return struct.pack("<Q", len(k)) + k + struct.pack("<I", 9) + struct.pack("<I", 8) + struct.pack("<Q", len(values)) + body


def tensor_infos_len(tensors):
    return sum(8 + len(name.encode()) + 4 + 8 * len(dims) + 4 + 8 for name, _, dims, _ in tensors)


class FakeHub:
    """The hub's reads, over files held in memory: {repo: {path: bytes}}. A path in `plain` has a git blob sha1 for its
    digest, as a file outside LFS; a path in `flip` is served with its first byte changed. `log` holds each read as
    (kind, repo, revision, a, b)."""

    def __init__(self, repos, plain=(), flip=()):
        self.repos, self.plain, self.flip, self.log = repos, set(plain), set(flip), []

    @staticmethod
    def sha_of(repo):
        return hashlib.sha1(repo.encode()).hexdigest()

    def main_sha(self, repo):
        self.log.append(("main_sha", repo, None, None, None))
        return self.sha_of(repo)

    def listing(self, repo, revision="main"):
        self.log.append(("listing", repo, revision, None, None))
        return [hf.Entry(p, len(b), "git-sha1", hashlib.sha1(b).hexdigest()) if p in self.plain
                else hf.Entry(p, len(b), "sha256", sha256_hex(b)) for p, b in self.repos[repo].items()]

    def range(self, repo, path, start, end, revision="main"):
        self.log.append(("range", repo, revision, start, end))
        data = self.repos[repo][path][start:end]
        if len(data) != end - start:
            raise hf.Red(f"{repo}: the range {start}-{end - 1} of {path} is past its {len(self.repos[repo][path])} bytes")
        if path in self.flip and start == 0:
            data = bytes([data[0] ^ 1]) + data[1:]
        return data


# A curl that answers the hub's reads from fixed text and logs each URL it is asked for to $STUB_LOG: the model info
# (o/nosha has no sha), a listing of two pages (the first has a Link header), and a ranged read that, as the hub's xet
# bridge, refuses a range under 1 KiB.
STUB_CURL = r"""#!/bin/sh
hdr=""; out=""; url=""; range=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dump-header) hdr="$2" ;;
    -o) out="$2" ;;
    -r) range="$2" ;;
  esac
  url="$1"
  shift
done
echo "$url" >> "$STUB_LOG"
case "$url" in
  */resolve/*)
    n=$(( ${range#*-} - ${range%-*} + 1 ))
    # The xet bridge answers a range under 1 KiB with the whole file, which curl's --max-filesize stops (63).
    [ "$n" -ge 1024 ] || exit 63
    { printf 'abcd'; head -c $((n - 4)) /dev/zero; } > "$out"; printf '206 %s' "$n" ;;
  https://h/next) : > "$hdr"; echo '[{"type":"file","path":"d/b.gguf","size":5,"oid":"@GIT@"},{"type":"directory","path":"d"}]' ;;
  */tree/*) printf 'Link: <https://h/next>; rel="next"\r\n' > "$hdr"; echo '[{"type":"file","path":"a.gguf","size":3,"lfs":{"oid":"@LFS@","size":3}}]' ;;
  */o/nosha) echo '{"id":"o/nosha"}' ;;
  *) echo '{"sha":"@SHA@"}' ;;
esac
"""


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

    g = hf.gguf_ranges()
    align = 64
    kvs = [kv_string("general.architecture", "tst"), kv_u32("general.alignment", align), kv_u32("tst.block_count", 12),
           kv_u32("tst.expert_count", 8), kv_u32("tst.nextn_predict_layers", 1), kv_u32("tst.context_length", 1024),
           kv_string("general.name", "x" * 200), kv_string("tokenizer.ggml.model", "gpt2"),
           kv_strings("general.tags", ["a", "b"])]
    blob = b"\7" * 8
    layers = [("blk.10.attn_q.weight", 12), ("blk.2.attn_q.weight", 12), ("blk.2.ffn_down.weight", 14),
              ("blk.10.ffn_down.weight", 14), ("blk.2.ffn_up.weight", 12), ("blk.10.ffn_up.weight", 14)]
    tensors = [("token_embd.weight", 0, [8, 3], blob)] + [(n, ty, [256, 2], blob) for n, ty in layers]
    want_kv = {"general.architecture": "tst", "general.alignment": align, "tst.block_count": 12,
               "tst.expert_count": 8, "tst.nextn_predict_layers": 1}
    want_top = {"token_embd.weight": ["F32", [8, 3]]}
    want_blk = {"attn_q.weight": {"Q4_K": [2, 10]}, "ffn_down.weight": {"Q6_K": [2, 10]},
                "ffn_up.weight": {"Q4_K": [2], "Q6_K": [10]}}

    with tempfile.TemporaryDirectory() as tmp:
        def gguf_file(name, kv, ts):
            path = os.path.join(tmp, name)
            g.write_gguf(path, kv, ts, align=align)
            with open(path, "rb") as f:
                return f.read()

        data = gguf_file("good.gguf", kvs, tensors)
        tell = 4 + 4 + 8 + 8 + sum(len(k) for k in kvs) + tensor_infos_len(tensors)
        entry = hf.Entry("d/good.gguf", len(data), "sha256", "0" * 64)
        hub = FakeHub({"o/r": {"d/good.gguf": data}})

        def shard(e=entry, want_head=True, h=hub):
            return read_shard(h, "o/r", e, tmp, want_head)

        got_kv, got_tensors, head = shard()
        check("the kv keeps general.*, the layer and expert counts and a short string, and drops the rest",
              got_kv == want_kv and list(got_kv) == list(want_kv), str(got_kv))
        check("a long string, an array and a key outside the filter are not kept",
              "general.name" not in got_kv and "general.tags" not in got_kv and "tst.context_length" not in got_kv)
        check("each tensor is read with its ggml type name, in file order",
              [(n, t) for n, t, _ in got_tensors] == [("token_embd.weight", "F32")] + [
                  (n, {12: "Q4_K", 14: "Q6_K"}[ty]) for n, ty in layers], str(got_tensors))
        top, blk = grouped(got_tensors)
        check("a tensor outside the blocks is top, keyed by its name with [type, dims]", top == want_top, str(top))
        check("blk is keyed by the name without blk.<i>., then type, then the layers sorted as numbers, keys sorted",
              blk == want_blk and list(blk) == sorted(blk) and blk["attn_q.weight"]["Q4_K"] == [2, 10], str(blk))

        bad = gguf_file("bad.gguf", kvs, tensors + [("blk.3.attn_k.weight", 77, [256, 2], blob)])
        bad_entry = hf.Entry("d/bad.gguf", len(bad), "sha256", "0" * 64)
        check("a type id the table does not name is a named error, never a placeholder",
              lambda: _raises(lambda: shard(bad_entry, h=FakeHub({"o/r": {"d/bad.gguf": bad}})), hf.Red,
                              "ggml type id 77 in d/bad.gguf: not in the type table"))
        check("the type table names every id gguf-ranges.py sizes, one name each",
              set(g.QUANT) <= set(hf.TYPE_NAMES) and len(set(hf.TYPE_NAMES.values())) == len(hf.TYPE_NAMES),
              str(sorted(set(g.QUANT) - set(hf.TYPE_NAMES))))

        # -- the stub
        check("the stub head is the header up to the aligned tensor data, GGUF first",
              head[:4] == b"GGUF" and len(head) % align == 0 and tell <= len(head) < tell + align
              and head == data[:len(head)] and len(head) < len(data), f"{len(head)} {tell} {len(data)}")
        short = os.path.join(tmp, "prefix.bin")
        with open(short, "wb") as f:
            f.write(data[:tell])
        check("a prefix that ends inside the padding is padded with zeros to the tensor data",
              lambda: stub_head(short, len(data)) == head, str(len(head)))
        stub = os.path.join(tmp, "stubs", "o", "r", "d", "good.gguf")
        write_stub(stub, head, len(data))
        with open(stub, "rb") as f:
            raw = f.read()
        check("the stub has the shard's size, the head, then zeros", len(raw) == len(data) and raw[:len(head)] == head
              and not any(raw[len(head):]) and os.path.getsize(stub) == len(data), f"{len(raw)} {len(data)}")
        m2, t2, start, size2, _ = g.header(stub)
        check("the GGUF reader opens the stub with every tensor in place", (m2, t2) == g.header(os.path.join(tmp, "good.gguf"))[:2]
              and start == len(head) and size2 == len(data))
        check("a head longer than the file is refused", lambda: _raises(lambda: write_stub(stub, head, len(head) - 1),
                                                                         hf.Red, "more than the file's"))
        check("a hub path that climbs out of the stub directory is refused",
              _raises(lambda: stub_path(tmp, "o/r", "../x.gguf"), hf.Red, "not a plain relative path")
              and _raises(lambda: stub_path(tmp, "o/r", "/x.gguf"), hf.Red, "not a plain relative path")
              and stub_path(tmp, "o/r", "d/x.gguf") == os.path.join(tmp, "o/r/d/x.gguf"))

        # -- the sets and the whole run, over an in-memory hub
        second = gguf_file("two.gguf", [kv_string("general.architecture", "tst"), kv_u32("tst.block_count", 12)],
                       [("blk.11.attn_q.weight", 12, [256, 2], blob), ("output.weight", 8, [8, 3], blob)])
        repos = {
            "o/r": {"m-00001-of-00002.gguf": data, "m-00002-of-00002.gguf": second, "solo.gguf": data,
                    "mtp-draft.gguf": data, "mmproj-m.gguf": data, "README.md": b"x"},
            "o/bad": {"bad.gguf": bad, "ok.gguf": data},
            "o/dup": {"a/x.gguf": data, "b/x.gguf": data},
            "o/none": {"README.md": b"x"},
        }
        fake = FakeHub(repos)
        check("the census keeps a draft head's file and leaves an mmproj file out",
              census_file("mtp-x.gguf") and census_file("dflash-x.gguf") and census_file("imatrix.gguf")
              and not census_file("mmproj-x.gguf") and not census_file("MMPROJ-x.gguf") and not census_file("x.bin"))

        def census(repo_list, stubs=None):
            out = os.path.join(tmp, "out")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = run(out, stubs, repo_list, lambda scratch: fake)
            return rc, sorted(buf.getvalue().splitlines()), out

        rc, lines, out = census(["o/r"], os.path.join(tmp, "st"))
        check("a run prints one line per set: the split set once, the draft head's file kept, mmproj and README left out",
              rc == 0 and lines == sorted([
                  "o/r m arch=tst tensors=9 shards=2", "o/r solo arch=tst tensors=7 shards=1",
                  "o/r mtp-draft arch=tst tensors=7 shards=1"]), f"{rc} {lines}")
        rec = json.load(open(os.path.join(out, "o__r__m.json")))
        check("its JSON has census's keys, the shards in order, the sizes, and the shards' tensors grouped together",
              list(rec) == ["repo", "set", "shards", "sizes", "kv", "top", "blk"] and rec["set"] == "m"
              and rec["shards"] == ["m-00001-of-00002.gguf", "m-00002-of-00002.gguf"]
              and rec["sizes"] == {"m-00001-of-00002.gguf": len(data), "m-00002-of-00002.gguf": len(second)}
              and rec["top"] == {"token_embd.weight": ["F32", [8, 3]], "output.weight": ["Q8_0", [8, 3]]}
              and rec["blk"]["attn_q.weight"] == {"Q4_K": [2, 10, 11]} and rec["kv"]["tst.block_count"] == 12,
              json.dumps(rec)[:300])
        check("--stubs wrote a sparse-size stub of every shard at DIR/<repo>/<file>", all(
            os.path.getsize(os.path.join(tmp, "st", "o", "r", f)) == n for f, n in (
                ("m-00001-of-00002.gguf", len(data)), ("m-00002-of-00002.gguf", len(second)), ("solo.gguf", len(data)))))
        rc, lines, out = census(["o/bad"], os.path.join(tmp, "st2"))
        check("a set with an unnamed type id fails by name, ends the run with 1, and writes no JSON and no stub",
              rc == 1 and "o/bad bad: FAILED ggml type id 77 in bad.gguf: not in the type table" in lines
              and "o/bad ok arch=tst tensors=7 shards=1" in lines and not os.path.exists(os.path.join(out, "o__bad__bad.json"))
              and not os.path.exists(os.path.join(tmp, "st2", "o", "bad", "bad.gguf")), f"{rc} {lines}")
        rc, lines, _ = census(["o/dup"])
        check("two sets that would write one output file fail the second by name",
              rc == 1 and len(lines) == 2 and any("FAILED its output file o__dup__x.json is the one of the set" in l
                                                  for l in lines), f"{rc} {lines}")
        rc, lines, _ = census(["o/none"])
        check("a repo with no GGUF file is a named failure",
              rc == 1 and lines == ["o/none: listing failed: o/none holds no GGUF file this reads"], f"{rc} {lines}")

        # -- the hub's URL forms and its pages, through a stub curl that logs each URL it is asked for
        sha, lfs_oid, git_oid = "a" * 40, "b" * 64, "c" * 40
        stubdir, curl_log = os.path.join(tmp, "bin"), os.path.join(tmp, "curl.log")
        os.makedirs(stubdir)
        with open(os.path.join(stubdir, "curl"), "w") as f:
            f.write(STUB_CURL.replace("@SHA@", sha).replace("@LFS@", lfs_oid).replace("@GIT@", git_oid))
        os.chmod(os.path.join(stubdir, "curl"), 0o755)
        saved = dict(os.environ)
        try:
            os.environ.update({"PATH": os.pathsep.join([stubdir, "/usr/bin", "/bin"]), "STUB_LOG": curl_log})
            stub_hub = hf.Hub(tmp)
            got_sha = stub_hub.main_sha("o/r")
            no_sha = _raises(lambda: stub_hub.main_sha("o/nosha"), hf.Red, "o/nosha: the API's model info has no commit sha")
            listed = stub_hub.listing("o/r", revision=sha)
            pinned = stub_hub.range("o/r", "d/x y.gguf", 0, 4, revision=sha)
            stub_hub.range("o/r", "d/x.gguf", 0, 4)
        finally:
            os.environ.clear()
            os.environ.update(saved)
        with open(curl_log) as f:
            asked = f.read().split()
        check("the model info's sha is the commit main points to; an answer with none is a named error",
              got_sha == sha and no_sha)
        check("a listing of two pages is joined, files only, each with the digest its kind of file has",
              listed == [hf.Entry("a.gguf", 3, "sha256", lfs_oid), hf.Entry("d/b.gguf", 5, "git-sha1", git_oid)],
              str(listed))
        check("a read of four bytes asks the hub's shortest served range and returns its own four",
              pinned == b"abcd", repr(pinned[:16]))
        check("a read names its revision in the one URL form, and a read with no revision names main", pinned == b"abcd"
              and asked == ["https://huggingface.co/api/models/o/r", "https://huggingface.co/api/models/o/nosha",
                            f"https://huggingface.co/api/models/o/r/tree/{sha}?recursive=true", "https://h/next",
                            f"https://huggingface.co/o/r/resolve/{sha}/d/x%20y.gguf",
                            "https://huggingface.co/o/r/resolve/main/d/x.gguf"], str(asked))

        # -- the manifest: which files it holds, then pin over an in-memory hub
        check("a manifest holds the census's files but an importance matrix; a draft head's file is its own set",
              manifest_file("m.gguf") and manifest_file("mtp-x.gguf") and manifest_file("dflash-x.gguf")
              and not manifest_file("imatrix_unsloth.gguf") and not manifest_file("mmproj-x.gguf")
              and not manifest_file("m.bin") and set_of("d/mtp-x.gguf") == ("mtp-x", None, None))
        check("a set is the basename minus a five-digit shard suffix, else minus .gguf",
              set_of("d/m-00001-of-00002.gguf") == ("m", 1, 2) and set_of("m-1-of-2.gguf") == ("m-1-of-2", None, None)
              and set_of("a-of-b-00002-of-00003.gguf") == ("a-of-b", 2, 3) and dir_of("d/e/m.gguf") == "d/e"
              and dir_of("m.gguf") == "")
        pin_repos = {
            "o/r": {"m-00001-of-00002.gguf": data, "m-00002-of-00002.gguf": second, "solo.gguf": data, "d/q.gguf": data,
                    "mtp-draft.gguf": data, "dflash-x.gguf": data, "imatrix_unsloth.gguf": data,
                    "mmproj-m.gguf": data, "README.md": b"x"},
            "p/s": {"s.gguf": data},
        }
        repos_file = os.path.join(tmp, "repos.txt")
        with open(repos_file, "w") as f:
            f.write("# the repos\n\np/s   # a comment after one\no/r\n")

        def quiet(fn):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rv = fn()
            return rv, buf.getvalue().splitlines()

        manifest = os.path.join(tmp, "stubs.tsv")
        pin_hub = FakeHub(pin_repos)
        rc, lines = quiet(lambda: pin(manifest, repos_file, lambda scratch: pin_hub))
        rows = read_manifest(manifest)
        files = [(r["org"], r["repo"], r["file"]) for r in rows]
        check("pin writes one row per file the census keeps, sorted by (org, repo, file)", rc == 0 and files == [
            ("o", "r", "d/q.gguf"), ("o", "r", "dflash-x.gguf"), ("o", "r", "m-00001-of-00002.gguf"),
            ("o", "r", "m-00002-of-00002.gguf"), ("o", "r", "mtp-draft.gguf"), ("o", "r", "solo.gguf"),
            ("p", "s", "s.gguf")], f"{rc} {files}")
        with open(manifest) as f:
            text = f.read()
        check("the table is its columns in order on a header line, a tab between cells, a newline ends every line",
              text.startswith("\t".join(MANIFEST_COLUMNS) + "\n") and text.endswith("\n") and "\r" not in text
              and all(len(l.split("\t")) == len(MANIFEST_COLUMNS) for l in text.splitlines()))
        check("the sets are the basename's, with the draft heads' files as their own sets and the two shards as one",
              sorted(set_groups(rows)) == sorted([("o", "r", "d", "q"), ("o", "r", "", "dflash-x"), ("o", "r", "", "m"),
                                                  ("o", "r", "", "mtp-draft"), ("o", "r", "", "solo"),
                                                  ("p", "s", "", "s")])
              and len(set_groups(rows)[("o", "r", "", "m")]) == 2)
        want_start = data_offset(os.path.join(tmp, "good.gguf"), len(data))
        want_start2 = data_offset(os.path.join(tmp, "two.gguf"), len(second))
        by_file = {r["file"]: r for r in rows if r["org"] == "o"}
        check("a row has the file's size and LFS sha256, the tensor data offset, and the sha256 of the real bytes before it",
              all(by_file[f]["file_bytes"] == str(len(b)) and by_file[f]["file_sha256"] == sha256_hex(b)
                  for f, b in pin_repos["o/r"].items() if f in by_file)
              and by_file["solo.gguf"]["header_bytes"] == str(want_start)
              and by_file["solo.gguf"]["header_sha256"] == sha256_hex(data[:want_start])
              and by_file["m-00002-of-00002.gguf"]["header_bytes"] == str(want_start2)
              and by_file["m-00002-of-00002.gguf"]["header_sha256"] == sha256_hex(second[:want_start2]),
              str(by_file["solo.gguf"]))
        reads = [e for e in pin_hub.log if e[0] in ("listing", "range")]
        check("every read of the run names the commit its repo's main pointed to at the start, and none names main",
              reads and all(e[2] == FakeHub.sha_of(e[1]) for e in reads)
              and {r["revision"] for r in rows if r["org"] == "o"} == {FakeHub.sha_of("o/r")}
              and [e[1] for e in pin_hub.log if e[0] == "main_sha"] == ["p/s", "o/r"], str(pin_hub.log[:4]))
        check("pin prints each repo's revision and a summary of the rows, sets, header bytes and wall",
              "o/r revision=" + FakeHub.sha_of("o/r") + " files=6" in lines
              and any(l.startswith("pin: 7 rows, 6 sets (1 of several shards), header_bytes ") for l in lines),
              str(lines))

        # the reads stop short of the padding: the rest of it is fetched, never synthesized
        short_hub = FakeHub({"o/r": {"solo.gguf": data}})
        short_manifest, one_repo = os.path.join(tmp, "short.tsv"), os.path.join(tmp, "one-repo.txt")
        with open(one_repo, "w") as f:
            f.write("o/r\n")
        keep, hf.FIRST_READ = hf.FIRST_READ, tell
        try:
            rc, _ = quiet(lambda: pin(short_manifest, one_repo, lambda scratch: short_hub))
        finally:
            hf.FIRST_READ = keep
        short_rows = read_manifest(short_manifest)
        check("a header whose reads end inside its padding gets the padding by a read at the pinned revision",
              rc == 0 and short_rows[0]["header_sha256"] == sha256_hex(data[:want_start])
              and ("range", "o/r", FakeHub.sha_of("o/r"), tell, want_start) in short_hub.log
              and tell < want_start, str(short_hub.log))

        # a file with no tensor ends at its header, short of the padding its data offset would add; a file with tensors
        # that ends short of it is cut
        bare = gguf_file("bare.gguf", kvs, [])
        bare_end = 4 + 4 + 8 + 8 + sum(len(k) for k in kvs)
        check("the synthetic metadata-only file is padded past its header, which its cut copy leaves out",
              len(bare) > bare_end and data_offset(os.path.join(tmp, "bare.gguf"), len(bare)) == len(bare))
        cut_manifest = os.path.join(tmp, "cut.tsv")
        rc, lines = quiet(lambda: pin(cut_manifest, one_repo, lambda scratch: FakeHub({"o/r": {"meta.gguf": bare[:bare_end]}})))
        cut_row = read_manifest(cut_manifest)[0]
        check("a file with no tensor, ending at its header, has the whole file for its prefix",
              rc == 0 and cut_row["header_bytes"] == cut_row["file_bytes"] == str(bare_end)
              and cut_row["header_sha256"] == sha256_hex(bare[:bare_end]), f"{rc} {cut_row}")
        rc, lines = quiet(lambda: pin(cut_manifest, one_repo, lambda scratch: FakeHub({"o/r": {"cut.gguf": data[:tell]}})))
        check("a file with tensors that ends short of its data offset is a named failure",
              rc == 1 and any("cut.gguf: FAILED cut.gguf: the header and its padding take " in l and
                              f"more than the file's {tell}" in l for l in lines), f"{rc} {lines}")

        # pin refuses, by name, what it cannot pin, and then writes nothing and leaves the table that was there
        with open(os.path.join(tmp, "old.tsv"), "w") as f:
            f.write("old\n")
        for what, repo_files, plain, want_text in (
                ("a file the listing gives no LFS sha256", {"s.gguf": data}, {"s.gguf"}, "no LFS sha256 (a git-sha1)"),
                ("a file whose header does not parse", {"s.gguf": data, "junk.gguf": b"not a gguf at all"}, (),
                 "junk.gguf: FAILED"),
                ("a repo with no file to keep", {"README.md": b"x"}, (), "holds no GGUF file this reads")):
            rc, lines = quiet(lambda: pin(os.path.join(tmp, "old.tsv"), repos_file,
                                          lambda scratch: FakeHub({"o/r": repo_files, "p/s": {"s.gguf": data}}, plain)))
            with open(os.path.join(tmp, "old.tsv")) as f:
                kept = f.read()
            check(f"pin over {what} ends with 1, names it and does not touch the table",
                  rc == 1 and kept == "old\n" and any(want_text in l for l in lines) and lines[-1].startswith("pin: ")
                  and "is not written" in lines[-1], f"{rc} {kept!r} {lines}")
        for body, want_text in (("o/r\no/r\n", "is listed twice"), ("o/r/x\n", "is not <owner>/<name>"),
                                ("# nothing\n", "no repo"), ("../x/y\n", "is not <owner>/<name>")):
            with open(os.path.join(tmp, "bad-repos.txt"), "w") as f:
                f.write(body)
            check(f"a repos file with {body.strip()!r} is refused by name",
                  _raises(lambda: read_repos(os.path.join(tmp, "bad-repos.txt")), hf.Refuse, want_text))

        # -- the rules, each with a row that breaks it and nothing else
        def mutant(edit):
            m = [dict(r) for r in rows]
            edit(m)
            return m

        def pick(m, name):
            return next(r for r in m if r["file"] == name)

        shard1, shard2 = "m-00001-of-00002.gguf", "m-00002-of-00002.gguf"

        def copy_of(m, name, **fields):
            m.append({**pick(m, name), **fields})

        def drop(m, name):
            m.remove(pick(m, name))

        mutants = [
            ("fields", "a row with a field the columns do not have", lambda m: m[0].update(extra="x")),
            ("fields", "a row without a field", lambda m: m[0].pop("set")),
            ("text", "a tab in a cell", lambda m: pick(m, "solo.gguf").update(file="so\tlo.gguf")),
            ("text", "a carriage return in a cell", lambda m: pick(m, "solo.gguf").update(file="solo.gguf\r")),
            ("path", "a file that does not end in .gguf", lambda m: pick(m, "solo.gguf").update(file="solo.bin")),
            ("path", "a file that is only .gguf", lambda m: pick(m, "solo.gguf").update(file=".gguf")),
            ("path", "a file with a .. component", lambda m: pick(m, "solo.gguf").update(file="d/../solo.gguf")),
            ("path", "a file with a leading /", lambda m: pick(m, "solo.gguf").update(file="/solo.gguf")),
            ("path", "a file with an empty component", lambda m: pick(m, "d/q.gguf").update(file="d//q.gguf")),
            ("path", "an org that is a .. component", lambda m: pick(m, "solo.gguf").update(org="..")),
            ("path", "an org with a leading /", lambda m: pick(m, "solo.gguf").update(org="/o")),
            ("path", "a repo that is a .. component", lambda m: pick(m, "solo.gguf").update(repo="..")),
            ("path", "a repo with a leading /", lambda m: pick(m, "solo.gguf").update(repo="/r")),
            ("path", "a repo with a / inside", lambda m: pick(m, "solo.gguf").update(repo="r/x")),
            ("set", "a set that is not the basename minus its suffix", lambda m: pick(m, "solo.gguf").update(set="solo2")),
            ("set", "a shard whose set keeps its suffix", lambda m: pick(m, shard2).update(set="m-00002-of-00002")),
            ("set", "a set that carries its directory", lambda m: pick(m, "d/q.gguf").update(set="d/q")),
            ("revision", "a revision in upper case", lambda m: pick(m, "solo.gguf").update(revision="A" * 40)),
            ("revision", "a revision of 39 digits", lambda m: pick(m, "solo.gguf").update(revision="a" * 39)),
            ("revision", "a branch name for a revision", lambda m: pick(m, "solo.gguf").update(revision="main")),
            ("digest", "a file digest in upper case", lambda m: pick(m, "solo.gguf").update(file_sha256="A" * 64)),
            ("digest", "a header digest of 63 digits", lambda m: pick(m, "solo.gguf").update(header_sha256="0" * 63)),
            ("size", "a header of 0 bytes", lambda m: pick(m, "solo.gguf").update(header_bytes="0")),
            ("size", "a header longer than its file",
             lambda m: pick(m, "solo.gguf").update(header_bytes=str(int(pick(m, "solo.gguf")["file_bytes"]) + 1))),
            ("size", "a size in hex", lambda m: pick(m, "solo.gguf").update(header_bytes="0x10")),
            ("size", "a size with a leading zero", lambda m: pick(m, "solo.gguf").update(file_bytes="0100")),
            ("size", "a negative size", lambda m: pick(m, "solo.gguf").update(file_bytes="-5")),
            ("duplicate", "two rows for one file", lambda m: copy_of(m, "solo.gguf")),
            ("shards", "a set without its second shard", lambda m: drop(m, shard2)),
            ("shards", "a shard past the count", lambda m: pick(m, shard2).update(file="m-00003-of-00002.gguf")),
            ("shards", "a shard 0", lambda m: pick(m, shard2).update(file="m-00000-of-00002.gguf")),
            ("shards", "shards under two counts", lambda m: pick(m, shard2).update(file="m-00002-of-00003.gguf")),
            ("shards", "a suffixed and an unsuffixed file of one set",
             lambda m: copy_of(m, "solo.gguf", file="m.gguf", set="m")),
            ("directory", "a second shard in another directory",
             lambda m: pick(m, shard2).update(file="x/" + shard2)),
            ("set-revision", "the shards of a set at two revisions",
             lambda m: pick(m, shard2).update(revision="1" * 40)),
            ("rows", "no row", lambda m: m.clear()),
        ]
        check("the rows pin wrote pass every rule", lambda: check_rows(rows) is None)
        for name, label, edit in mutants:
            check(f"rule {name} refuses {label}", lambda: _raises(lambda: check_rows(mutant(edit)), hf.Refuse,
                                                                  f"census manifest rule {name}: "))
        check("every rule the check names has a mutant above", {m[0] for m in mutants} == set(RULES),
              str(set(RULES) ^ {m[0] for m in mutants}))

        # -- reading and writing the table
        def refused(label, text, name):
            with open(os.path.join(tmp, "m.tsv"), "w", newline="") as f:
                f.write(text)
            check(f"a table with {label} is refused by rule {name}",
                  lambda: _raises(lambda: read_manifest(os.path.join(tmp, "m.tsv")), hf.Refuse,
                                  f"census manifest rule {name}: "))

        lines_ = text.splitlines()
        refused("a last line with no newline", text[:-1], "fields")
        refused("a column missing", "\n".join([lines_[0].rsplit("\t", 1)[0]] + [l.rsplit("\t", 1)[0] for l in lines_[1:]])
                + "\n", "fields")
        refused("a column twice", "\n".join([lines_[0] + "\tset"] + [l + "\tx" for l in lines_[1:]]) + "\n", "fields")
        refused("a row with a cell fewer", "\n".join(lines_[:2] + [lines_[2].rsplit("\t", 1)[0]] + lines_[3:]) + "\n",
                "fields")
        refused("carriage returns in its rows", lines_[0] + "\n" + "\r\n".join(lines_[1:]) + "\r\n", "text")
        refused("a header line and no row", lines_[0] + "\n", "rows")
        with open(os.path.join(tmp, "m.tsv"), "wb") as f:
            f.write(b"\xff\xfe\n")
        check("a table that is not UTF-8 is refused by rule text", lambda: _raises(
            lambda: read_manifest(os.path.join(tmp, "m.tsv")), hf.Refuse, "census manifest rule text: "))
        check("a table that is not there is refused by name",
              lambda: _raises(lambda: read_manifest(os.path.join(tmp, "no-such.tsv")), hf.Refuse, "no-such.tsv: "))
        swapped = os.path.join(tmp, "swapped.tsv")
        wrote = write_manifest(swapped, list(reversed(rows)))
        cols = lines_[0].split("\t")
        with open(os.path.join(tmp, "m.tsv"), "w") as f:  # the same table, its columns in another order
            f.write("\n".join("\t".join(c[cols.index(n)] for n in reversed(cols)) for c in
                              (l.split("\t") for l in lines_)) + "\n")
        check("write sorts the rows and the table reads back as the rows; the columns are read by name, in any order",
              wrote == rows and read_manifest(swapped) == rows and read_manifest(os.path.join(tmp, "m.tsv")) == rows)
        broken = [dict(r) for r in rows]
        pick(broken, "solo.gguf")["revision"] = "A" * 40
        target = os.path.join(tmp, "target.tsv")
        with open(target, "w") as f:
            f.write("old\n")
        refused_write = _raises(lambda: write_manifest(target, broken), hf.Refuse, "census manifest rule revision")
        with open(target) as f:
            left = f.read()
        check("a table that breaks a rule is not written: the table that was there stays and no temp file does",
              refused_write and left == "old\n" and not [n for n in os.listdir(tmp) if n.startswith(".stubs-")],
              str(os.listdir(tmp)))

        # -- build, over an in-memory hub: sparse files of the real length, the real header, nothing else
        wide = [dict(r) for r in rows]
        pick(wide, "solo.gguf")["file_bytes"] = str(8 * hf.MIB)
        wide_manifest = os.path.join(tmp, "wide.tsv")
        write_manifest(wide_manifest, wide)
        probe = os.path.join(tmp, "probe")
        with open(probe, "wb") as f:
            f.truncate(8 * hf.MIB)
        holes = os.stat(probe).st_blocks * 512 < hf.MIB
        root = os.path.join(tmp, "stubs-root")
        build_hub = FakeHub(pin_repos)
        rc, lines = quiet(lambda: build(root, wide_manifest, lambda scratch: build_hub))

        def stub_ok(r, base):
            fn = os.path.join(base, r["org"], r["repo"], r["file"])
            n = int(r["header_bytes"])
            with open(fn, "rb") as f:
                got = f.read(n)
                rest = f.read()
            return (os.path.getsize(fn) == int(r["file_bytes"]) and sha256_hex(got) == r["header_sha256"]
                    and rest.count(0) == len(rest))

        check("build writes, for every row, a stub of file_bytes whose first header_bytes bytes have header_sha256 and "
              "whose other bytes are zero", rc == 0 and all(stub_ok(r, root) for r in wide) and "build: 6 of 6 sets built into "
              + root in lines, f"{rc} {lines}")
        big = os.stat(os.path.join(root, "o", "r", "solo.gguf"))
        check("a stub is sparse" if holes else "a stub is sparse (this filesystem has no holes: not checked)",
              big.st_size == 8 * hf.MIB and (not holes or big.st_blocks * 512 < hf.MIB), str(big))
        ranges = sorted((e[1], e[2], e[3], e[4]) for e in build_hub.log if e[0] == "range")
        check("build reads each file once, from byte 0 to header_bytes, at the row's revision, and lists nothing",
              ranges == sorted((f"{r['org']}/{r['repo']}", r["revision"], 0, int(r["header_bytes"])) for r in wide)
              and not [e for e in build_hub.log if e[0] != "range"], str(ranges))
        before = len(build_hub.log)
        check("a ROOT that holds anything is refused by name, before any read",
              lambda: _raises(lambda: build(root, wide_manifest, lambda scratch: build_hub), hf.Refuse,
                              "ROOT exists and is not an empty directory") and len(build_hub.log) == before)
        check("a ROOT that is a file is refused by name",
              lambda: _raises(lambda: build(probe, wide_manifest, lambda scratch: build_hub), hf.Refuse,
                              "ROOT exists and is not an empty directory"))
        empty = os.path.join(tmp, "empty-root")
        os.makedirs(empty)
        rc, _ = quiet(lambda: build(empty, wide_manifest, lambda scratch: build_hub))
        check("an empty ROOT is taken", rc == 0 and all(stub_ok(r, empty) for r in wide))

        bad_root = os.path.join(tmp, "bad-root")
        before = len(build_hub.log)
        write_manifest(os.path.join(tmp, "good2.tsv"), wide)
        with open(os.path.join(tmp, "good2.tsv")) as f:
            good_text = f.read()
        with open(os.path.join(tmp, "bad.tsv"), "w") as f:
            f.write(good_text.replace(by_file["solo.gguf"]["revision"], "A" * 40))
        check("a table that breaks a rule is refused before ROOT is made or a byte is read", lambda: _raises(
            lambda: build(bad_root, os.path.join(tmp, "bad.tsv"), lambda scratch: build_hub), hf.Refuse,
            "census manifest rule revision") and not os.path.exists(bad_root) and len(build_hub.log) == before)

        flip_root = os.path.join(tmp, "flip-root")
        flipped = bytes([second[0] ^ 1]) + second[1:want_start2]
        rc, lines = quiet(lambda: build(flip_root, wide_manifest, lambda scratch: FakeHub(pin_repos, flip={shard2})))
        failing = [l for l in lines if "FAILED" in l]
        r2 = by_file[shard2]
        check("a fetched prefix with another digest fails its set by name with org, repo, file and both digests",
              rc == 1 and len(failing) == 1 and failing[0].startswith("o/r m: FAILED ")
              and all(t in failing[0] for t in ("o/r", shard2, r2["header_sha256"], sha256_hex(flipped), r2["revision"])),
              f"{rc} {lines}")
        check("the failed set leaves no file behind, the other sets are built and the run ends with 1",
              not os.path.exists(os.path.join(flip_root, "o", "r", shard1)) and not os.path.exists(
                  os.path.join(flip_root, "o", "r", shard2))
              and all(os.path.getsize(os.path.join(flip_root, r["org"], r["repo"], r["file"])) == int(r["file_bytes"])
                      for r in wide if r["file"] not in (shard1, shard2)) and "build: 5 of 6 sets built into " + flip_root
              in lines, f"{lines}")

        reread_root = os.path.join(tmp, "reread-root")
        real_write = write_stub

        def write_other_bytes(fn, head, size):
            real_write(fn, bytes([head[0] ^ 1]) + head[1:], size)

        globals()["write_stub"] = write_other_bytes
        try:
            rc, lines = quiet(lambda: build(reread_root, wide_manifest, lambda scratch: FakeHub(pin_repos)))
        finally:
            globals()["write_stub"] = real_write
        check("a stub that reads back with another digest fails its set by name, and leaves no file",
              rc == 1 and len([l for l in lines if "the stub written has" in l]) == 6
              and not [n for _, _, fs in os.walk(reread_root) for n in fs], f"{rc} {lines}")

        # -- the command line
        errs = io.StringIO()
        with contextlib.redirect_stderr(errs):
            usage = [main(["pin", "only-one"]), main(["build"]), main(["pin", "-x", "y"]), main(["build", "a", "b", "c"])]
            missing = main(["build", os.path.join(tmp, "r"), os.path.join(tmp, "absent.tsv")])
        check("pin and build with the wrong arguments are usage errors (64); an absent table is an input refusal (65)",
              usage == [64, 64, 64, 64] and missing == 65 and "absent.tsv" in errs.getvalue(), f"{usage} {missing}")
    return ok


def _raises(fn, exc, text):
    try:
        fn()
    except exc as e:
        return text in str(e)
    return False


def manifest_main(mode, args):
    """The `pin` and `build` commands: their arguments, and the exit status of what they refuse."""
    if len(args) != 2 or any(a.startswith("-") for a in args):
        print(USAGE, file=sys.stderr)
        return 64
    try:
        return pin(*args) if mode == "pin" else build(*args)
    except hf.Refuse as e:
        print(f"type-census.py {mode}: {one_line(e)}", file=sys.stderr)
        return e.code
    except OSError as e:
        print(f"type-census.py {mode}: {one_line(e)}", file=sys.stderr)
        return 1


def main(argv):
    if argv == ["--self-test"]:
        try:
            return 0 if self_test() else 1
        except Exception as e:
            print(f"FAIL the self-test stopped on {type(e).__name__}: {e}")
            return 1
    if argv and argv[0] in ("pin", "build"):
        return manifest_main(argv[0], argv[1:])
    args, stubs = list(argv), None
    if "--stubs" in args:
        at = args.index("--stubs")
        if at + 1 >= len(args):
            print(USAGE, file=sys.stderr)
            return 64
        stubs = args[at + 1]
        del args[at:at + 2]
    if len(args) < 2 or any(a.startswith("-") for a in args):
        print(USAGE, file=sys.stderr)
        return 64
    out, repos = args[0], args[1:]
    for r in repos:
        if not REPO.fullmatch(r) or ".." in r.split("/"):
            print(f"type-census.py: {r!r} is not <owner>/<name>", file=sys.stderr)
            return 64
    return run(out, stubs, repos)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
