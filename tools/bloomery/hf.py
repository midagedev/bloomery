"""The Hugging Face hub's reads, for the tools that look at the GGUF files users download: the listing of a repo, its
GGUF sets, a ranged fetch, and the header of a GGUF file read by ranged fetches. Every read is curl's (as
crates/hf/src/fetch.rs): Python's own TLS can fail on a machine curl works on. A read names its revision (a branch
or a commit sha; `main` unless the caller pins one), and the three URL forms of the hub are built here and nowhere
else.

tools/ has no packages: a script imports this file by path, as tools/hf-arch-check.py and tools/type-census.py do.

    info_url / tree_url / resolve_url    the model info, the listing and a file of a repo at a revision
    Hub(scratch).main_sha(repo)          the commit sha `main` points to now
    Hub(scratch).listing(repo, revision) the files of the repo at the revision (main), as Entry
    Hub(scratch).range(repo, path, a, b, revision) the bytes [a, b) of one file at the revision
    gguf_sets(repo, entries)             the GGUF sets of a listing (the hf crate's grouping)
    read_header(fetch, local, size, n)   a file's metadata and tensor list, ranged fetches until it parses
    ggml_type_name(id, file)             the ggml name of a tensor type id; an id the table lacks is a Red

Refuse is a refusal that ends a run with its code (the network 69, an input 65, a usage error 64); Red is one
row's or one file's refusal, which the caller prints and goes on from. The header parser is
tools/ref/gguf-ranges.py, the one parser of the tree, loaded by path.
"""
import collections
import importlib.util
import json
import os
import re
import subprocess
import sys
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HUB = "https://huggingface.co"
MIB = 1 << 20
FIRST_READ = 1 * MIB
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


# The ggml_type names (the spelling of ggml's enum, which the Rust side lower-cases in GgmlType::name):
# crates/gguf/src/quant.rs:GgmlType::from_u32, the ids the enum names. An id not in it has no name here.
GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K",
    12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
    20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
    29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4", 41: "Q1_0", 42: "Q2_0",
    2000: "F8E4M3_ROWS", 2001: "MXFP8", 2002: "NVFP4_ROWS", 2003: "FP8_B128",
    2004: "MLX_A4", 2005: "MLX_A8", 2006: "MLX_A4H", 2007: "MLX_A8H",
}


def ggml_type_name(ty, file):
    """The ggml name of the tensor type id `ty` read from `file`; an id the table does not name is a Red."""
    name = GGML_TYPES.get(ty)
    if name is None:
        raise Red(f"ggml type id {ty} in {file}: not in the type table")
    return name


# ------------------------------------------------------------------------------------------------
# The hf crate's grouping of a listing into GGUF sets, mirrored from crates/hf/src/lib.rs.

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


def gguf_sets(repo, entries, model=is_model):
    """crates/hf/src/lib.rs:gguf_sets: the GGUF model sets of `entries` (Entry), in the listing's order of their
    first file; a split set with a missing or doubled shard is a Red. `model` says which file names (the last
    path component) are model files: the hf crate's `is_model`, or a census's own rule."""
    sets = []
    for e in entries:
        if not model(e.path.rsplit("/", 1)[-1]):
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


# ------------------------------------------------------------------------------------------------
# The hub's URL forms: the one owner, so a pinned read and a read of main differ by the revision alone.

def info_url(repo):
    """The model info of `repo` (its `sha` is the commit `main` points to now)."""
    return f"{HUB}/api/models/{repo}"


def tree_url(repo, revision):
    """The recursive listing of `repo` at `revision`."""
    return f"{HUB}/api/models/{repo}/tree/{urllib.parse.quote(revision, safe='')}?recursive=true"


def resolve_url(repo, revision, path):
    """The bytes of `path` of `repo` at `revision`."""
    return f"{HUB}/{repo}/resolve/{urllib.parse.quote(revision, safe='')}/{urllib.parse.quote(path)}"


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


# The shortest range read the hub serves as a range: its xet bridge answers a range of a few hundred bytes of a large
# file with the whole file (HTTP 200), so a shorter read asks this much and keeps its own bytes.
MIN_RANGE = 4096


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

    def main_sha(self, repo):
        """The commit sha `main` of `repo` points to now, from the model info; an answer without one is a Red."""
        url = info_url(repo)
        body = self._curl(f"model info {url}", [url])
        self.bytes += len(body)
        try:
            sha = hex_of(json.loads(body.decode("utf-8", "replace")).get("sha"), 40)
        except (ValueError, AttributeError):
            sha = None
        if sha is None:
            raise Red(f"{repo}: the API's model info has no commit sha this reads")
        return sha

    def listing(self, repo, revision="main"):
        """Every file of `repo` at `revision`, each page followed to its last."""
        url = tree_url(repo, revision)
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

    def range(self, repo, path, start, end, revision="main"):
        """The bytes [start, end) of `path` of `repo` at `revision`: one range read of at least MIN_RANGE bytes (the
        hub's xet bridge answers a shorter range with the whole file), checked to be a 206 holding them (a server
        that sends the whole file is refused before the body, by --max-filesize); a range past the end of the file
        comes back short, which is fine as long as it holds [start, end)."""
        want = end - start
        ask = max(want, MIN_RANGE)
        url = resolve_url(repo, revision, path)
        part = os.path.join(self.scratch, "range.part")
        out = self._curl(f"range {start}-{start + ask - 1} of {url}",
                         ["-r", f"{start}-{start + ask - 1}", "--max-filesize", str(ask + 1024), "-o", part,
                          "-w", "%{http_code} %{size_download}", url])
        code, _, got = out.decode().partition(" ")
        with open(part, "rb") as f:
            data = f.read()
        os.unlink(part)
        if code != "206" or int(got) != len(data) or len(data) < want:
            raise Red(f"{repo}: the range {start}-{start + ask - 1} of {path} answered HTTP {code} with {len(data)} "
                      f"bytes, not a 206 holding the {want} asked for")
        self.bytes += len(data)
        return data[:want]


# ------------------------------------------------------------------------------------------------
# A GGUF file's header.

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
