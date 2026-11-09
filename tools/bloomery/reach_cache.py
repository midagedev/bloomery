"""The reach's per-file facts, kept by content, and the snapshot guard of a key run.

`Reach.facts` parses a file once per (text, cfg(test)) and that parse is a pure function of those two and of the
analysis code, so its result is stored under the digest of exactly that: the entry is shared by every tree, and a tree
whose files changed pays the parse of those files only. An entry is hydrated only through the fixed schema below (no
code objects, pickle, dynamic imports or arbitrary object construction) and is checked against its own key and
payload digest; a damaged entry is named on stderr and recomputed. The store is bounded: `prune` removes the
least-recently-used entries (by mtime, refreshed on a hit older than an hour) once the directory holds more than
CAP_BYTES, down to KEEP_BYTES, at the exit of a process that stored something.

`prepare` runs the structural walk of a key request under observation and `current` re-checks it, so source that moves
while a key is assembled is refused rather than keyed.
"""
from __future__ import annotations

import atexit
import contextlib
import dataclasses
import functools
import hashlib
import json
import os
import stat
import sys
import tempfile
import time

_REACH_LOADED_CODE = sys._getframe().f_code

VERSION = 3
CAP_BYTES = 1 << 30
KEEP_BYTES = CAP_BYTES * 3 // 4
TOUCH_AFTER = 3600
STATS = {"hits": 0, "live": 0, "misses": 0, "stores": 0, "corrupt": 0, "evicted": 0, "off": 0}
_LIVE: dict[tuple[str, bool], object] = {}
_BODIES: dict[str, object] = {}


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _bytes(path):
    with open(path, "rb") as fh:
        before = os.fstat(fh.fileno())
        data = fh.read()
        after = os.fstat(fh.fileno())
    if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise OSError("file changed during reach snapshot: " + path)
    return data


@contextlib.contextmanager
def _paths():
    """Share parent resolution within one snapshot, then check every parent again before using it."""
    parents = {}

    def resolve(path):
        parent, name = os.path.split(path)
        if name in ("", ".", ".."):
            return os.path.realpath(path)
        try:
            linked = stat.S_ISLNK(os.lstat(path).st_mode)
        except OSError:
            linked = False
        if linked:
            return os.path.realpath(path)
        if parent not in parents:
            parents[parent] = os.path.realpath(parent)
        return os.path.join(parents[parent], name)

    yield resolve
    for parent, real in parents.items():
        if os.path.realpath(parent) != real:
            raise OSError("source parent changed during reach snapshot: " + parent)


def _file(path, cache, realpath):
    return [realpath(path), cache.sha(path)]


def _corpus(tree):
    """Enumerate source additions too, including excluded crates and symlinked source referents.

    Cargo's output directory and git's object database are not source owners. Reads from either are still
    covered by the observed-read provenance. Directory entries (including symlinks) catch new target roots.
    """
    cache = tree._reach_hashes
    cache.t0_ns = time.time_ns()
    rows = []
    with _paths() as realpath:
        for base, dirs, files in os.walk(tree.root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in (".git", "target", "__pycache__"))
            relbase = os.path.relpath(base, tree.root)
            rows.append([relbase, dirs[:], sorted(files)])
            for name in sorted(files):
                if name.endswith((".rs", ".toml", ".py")) or name == "Cargo.lock":
                    path = os.path.join(base, name)
                    rows.append([os.path.relpath(path, tree.root), _file(path, cache, realpath)])
            # Walk symlinked source directories separately, following each real directory only once.
            for name in dirs[:]:
                path = os.path.join(base, name)
                if os.path.islink(path):
                    rows.append([os.path.relpath(path, tree.root), _linked(path, cache, realpath)])
    return _digest(_json(rows))


def _linked(root, cache, realpath):
    rows, seen = [], set()
    def visit(path, rel):
        real = realpath(path)
        if os.path.isdir(path):
            if real in seen:
                rows.append([rel, real, "cycle"])
                return
            seen.add(real)
            names = sorted(os.listdir(path))
            rows.append([rel, real, names])
            for name in names:
                visit(os.path.join(path, name), rel + "/" + name)
        elif os.path.isfile(path):
            rows.append([rel, _file(path, cache, realpath)])
        else:
            rows.append([rel, real, "absent"])
    visit(root, "")
    return rows


@functools.lru_cache(maxsize=16)
def _disk_code(data, filename, optimize):
    """Memoize compilation by exact bytes, never file timestamps or a prior disk identity."""
    return compile(data, filename, "exec", dont_inherit=True, optimize=optimize)


def _code_identity(module, error):
    """Bind disk bytes to the code actually imported, never reconstruct code from cache data."""
    loaded = module._REACH_LOADED_CODE
    path = module.__file__
    data = _bytes(path)
    message = "analysis code changed after import: " + path + "; request a fresh process"
    try:
        disk = _disk_code(data, loaded.co_filename, sys.flags.optimize)
    except (SyntaxError, ValueError, OverflowError) as exc:
        raise error(message) from exc
    # Code equality compares bytecode, constants, names and line tables recursively and ignores runtime
    # specialization. Marshal output depends on object sharing, and its older formats cannot hold every
    # constant the compiler folds (slices).
    if disk != loaded:
        raise error(message)
    return [os.path.realpath(path), _digest(data)]


def _identity(tree):
    module = sys.modules[type(tree).__module__]
    code = [_code_identity(module, module.RecipeError),
            _code_identity(sys.modules[__name__], module.RecipeError)]
    fixture = sys.modules.get("bloomery.fixture_dirs")
    if fixture is not None:
        code.append(_code_identity(fixture, module.RecipeError))
    return _digest(_json([VERSION, sys.version, tree.root, code, tree.metadata,
                         {n: dataclasses.asdict(p) for n, p in tree.packages.items()}]))


def _observed(tree, saw):
    rows = []
    with _paths() as realpath:
        for (kind, rel), was in sorted(saw.items()):
            path = os.path.join(tree.root, rel)
            if kind == "r":
                # Tree.read uses text-mode universal newlines; the digest still covers the raw bytes.
                data = _bytes(path)
                text = data.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
                if text != was[0]:
                    raise OSError("source changed during reach analysis: " + rel)
                rows.append([kind, rel, [realpath(path), _digest(data)]])
            else:
                now = os.path.isfile(path) if kind == "f" else os.path.isdir(path)
                if now != was:
                    raise OSError("source probe changed during reach analysis: " + rel)
                rows.append([kind, rel, [now, realpath(path)]])
    return rows


def _holds(tree, rows):
    if not isinstance(rows, list):
        return False
    with _paths() as realpath:
        for row in rows:
            if not isinstance(row, list) or len(row) != 3:
                return False
            kind, rel, was = row
            if kind not in ("r", "f", "d") or not isinstance(rel, str):
                return False
            path = os.path.join(tree.root, rel)
            if kind == "r":
                if _file(path, tree._reach_hashes, realpath) != was:
                    return False
            elif [(os.path.isfile(path) if kind == "f" else os.path.isdir(path)), realpath(path)] != was:
                return False
    return True


def current(reach):
    try:
        return (_identity(reach.tree) == reach._key_identity and _corpus(reach.tree) == reach._key_snapshot
                and _holds(reach.tree, reach._key_observed))
    except (OSError, ValueError):
        return False


# ---- the facts store ----

_STR = frozenset({str})
_UNIT_FIELDS = ("kind", "ctx", "name", "line", "sites", "idents", "macro_calls", "includes", "literals", "impl",
                "names", "methods", "test")
_HEAD_FIELDS = ("params", "self_head", "self_args", "trait", "trait_args", "text")
_SCHEMA_OK: set[int] = set()
_PRUNE_ARMED = []


def store_dir():
    return os.path.join(os.path.expanduser("~"), ".cache", "bloomery", "reach-facts")


def _check_schema(module):
    """The codec is written against these fields; a record that grew one fails here, by name, not as a stale entry."""
    if id(module) in _SCHEMA_OK:
        return
    for cls, names in ((module.Unit, _UNIT_FIELDS), (module.ImplHead, _HEAD_FIELDS)):
        if tuple(f.name for f in dataclasses.fields(cls)) != names:
            raise module.RecipeError(f"reach_cache: {cls.__name__} fields changed; update the facts codec ({names})")
    _SCHEMA_OK.add(id(module))


def _facts_code(reach):
    """The digest of the analysis code that parses (recipes.py and this file by their bytes, never their paths, so
    trees with equal code share entries), or None when the code on disk is not the code running: the store is then
    off for this process, and `prepare` refuses the run by name."""
    code = reach.__dict__.get("_facts_code")
    if code is None:
        module = sys.modules[type(reach).__module__]
        try:
            parts = [_code_identity(module, module.RecipeError)[1], _code_identity(sys.modules[__name__], module.RecipeError)[1]]
            code = _digest(_json([VERSION, sys.version, parts]))
        except (OSError, module.RecipeError):
            code = ""
        reach.__dict__["_facts_code"] = code
    return code or None


def _facts_key(code, test, text):
    sha = hashlib.sha256()
    sha.update(code.encode() + (b"\0t\0" if test else b"\0f\0"))
    sha.update(text.encode("utf-8", "surrogatepass"))
    return sha.hexdigest()


def _strs(value):
    if type(value) is not list or not _STR.issuperset(map(type, value)):
        raise ValueError("invalid string list")
    return value


def _opt_path(value):
    return None if value is None else tuple(_strs(value))


def _path_list(value):
    if type(value) is not list:
        raise ValueError("invalid path list")
    return tuple([tuple(_strs(p)) for p in value])


def _enc_facts(value):
    heads, at = [], {}

    def head(h):
        if h is None:
            return -1
        if h not in at:
            at[h] = len(heads)
            heads.append([sorted(h.params), None if h.self_head is None else list(h.self_head),
                          [list(a) for a in h.self_args], None if h.trait is None else list(h.trait),
                          [list(a) for a in h.trait_args], h.text])
        return at[h]

    units = [[u.kind, list(u.ctx), u.name, u.line, [list(s) for s in u.sites], sorted(u.idents), sorted(u.macro_calls),
              list(u.includes), list(u.literals), head(u.impl), sorted(u.names), sorted(u.methods), u.test]
             for u in value.units[:value.uses]]
    return {"u": units, "h": heads, "b": [[list(c), list(p), n, pub] for c, p, n, pub in value.binds],
            "x": sorted(value.exported)}


def _dec_head(row, module):
    if type(row) is not list or len(row) != len(_HEAD_FIELDS):
        raise ValueError("invalid impl head")
    params, self_head, self_args, trait, trait_args, text = row
    if type(text) is not str:
        raise ValueError("invalid impl head text")
    return module.ImplHead(frozenset(_strs(params)), _opt_path(self_head), _path_list(self_args), _opt_path(trait),
                           _path_list(trait_args), text)


def _dec_facts(body, module):
    if type(body) is not dict or set(body) != {"u", "h", "b", "x"}:
        raise ValueError("invalid facts schema")
    if type(body["h"]) is not list or type(body["u"]) is not list or type(body["b"]) is not list:
        raise ValueError("invalid facts lists")
    heads = [_dec_head(h, module) for h in body["h"]]
    units = []
    for row in body["u"]:
        if type(row) is not list or len(row) != len(_UNIT_FIELDS):
            raise ValueError("invalid unit")
        kind, ctx, name, line, sites, idents, calls, incs, lits, impl, names, methods, test = row
        if (type(kind) is not str or (name is not None and type(name) is not str) or type(line) is not int
                or type(test) is not bool or type(impl) is not int or not -1 <= impl < len(heads)):
            raise ValueError("invalid unit field")
        units.append(module.Unit(kind, tuple(_strs(ctx)), name, line, _path_list(sites), frozenset(_strs(idents)),
                                 frozenset(_strs(calls)), tuple(_strs(incs)), tuple(_strs(lits)),
                                 heads[impl] if impl >= 0 else None, frozenset(_strs(names)),
                                 frozenset(_strs(methods)), test))
    binds = []
    for row in body["b"]:
        if type(row) is not list or len(row) != 4 or type(row[2]) is not str or type(row[3]) is not bool:
            raise ValueError("invalid use leaf")
        binds.append((tuple(_strs(row[0])), tuple(_strs(row[1])), row[2], row[3]))
    return module.RustFacts(units, binds, frozenset(_strs(body["x"])))


def _damaged(path, why):
    STATS["corrupt"] += 1
    print(f"reach cache: {path}: {why}; recomputed", file=sys.stderr)
    with contextlib.suppress(OSError):
        os.unlink(path)


def _verify(head, key, body):
    """Whether this header, under this key, vouches for these payload bytes; raises the reason when it does not."""
    if type(head) is not dict or set(head) != {"v", "key", "digest"} or head["v"] != VERSION:
        raise ValueError("foreign version or header")
    if head["key"] != key:
        raise ValueError("another entry's key")
    if head["digest"] != _digest(body):
        raise ValueError("payload digest mismatch")


def _load(path, key, module):
    """The entry's facts, or None for none or a damaged one (named, and removed so the recomputation replaces it)."""
    try:
        st = os.lstat(path)
    except OSError:
        return None  # no entry, or no usable store (a HOME that is not a directory): a miss either way
    try:
        if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid() or st.st_mode & 0o077:
            raise ValueError("not a private regular file")
        with open(path, "rb") as fh:
            raw = fh.read()
        line, sep, body = raw.partition(b"\n")
        head = json.loads(line)
        if not sep:
            raise ValueError("no header line")
        _verify(head, key, body)
        # equal payloads are one parse: a file whose two cfg(test) views read alike is stored twice
        value = _BODIES.get(head["digest"])
        if value is None:
            value = _BODIES[head["digest"]] = _dec_facts(json.loads(body), module)
    except (OSError, ValueError, TypeError, KeyError, IndexError, RecursionError) as exc:
        _damaged(path, str(exc) or type(exc).__name__)
        return None
    if time.time() - st.st_mtime > TOUCH_AFTER:
        with contextlib.suppress(OSError):
            os.utime(path)
    return value


def _store(path, key, value):
    """Write the entry; a store that cannot be written (read-only HOME, full disk) is a miss, never a failure."""
    body = json.dumps(_enc_facts(value), separators=(",", ":")).encode()
    data = json.dumps({"v": VERSION, "key": key, "digest": _digest(body)}, separators=(",", ":")).encode() + b"\n" + body
    shard, tmp = os.path.dirname(path), None
    try:
        root = os.path.dirname(shard)
        os.makedirs(root, mode=0o700, exist_ok=True)
        st = os.lstat(root)
        if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid() or st.st_mode & 0o077:
            return
        os.makedirs(shard, mode=0o700, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=shard, suffix=".tmp", delete=False) as fh:
            tmp = fh.name
            fh.write(data)
        os.replace(tmp, path)
        tmp = None
    except OSError:
        return
    finally:
        if tmp is not None:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
    STATS["stores"] += 1
    if not _PRUNE_ARMED:
        _PRUNE_ARMED.append(atexit.register(prune))


def prune(cap=CAP_BYTES, keep=KEEP_BYTES, root=None):
    """Past `cap` bytes of entries, remove the least recently used (oldest mtime) until `keep` remain, and any
    temporary file older than an hour; the number of entries removed."""
    root = root or store_dir()
    now = time.time()
    entries, total = [], 0
    try:
        shards = [e.path for e in os.scandir(root) if e.is_dir(follow_symlinks=False)]
    except OSError:
        return 0
    for shard in shards:
        try:
            files = list(os.scandir(shard))
        except OSError:
            continue
        for e in files:
            try:
                st = e.stat(follow_symlinks=False)
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            if e.name.endswith(".tmp"):
                if now - st.st_mtime > 3600:
                    with contextlib.suppress(OSError):
                        os.unlink(e.path)
            elif e.name.endswith(".json"):
                entries.append((st.st_mtime, e.path, st.st_size))
                total += st.st_size
    if total <= cap:
        return 0
    entries.sort()
    removed = 0
    for _, path, size in entries:
        if total <= keep:
            break
        try:
            os.unlink(path)
        except OSError:
            continue
        total -= size
        removed += 1
    STATS["evicted"] += removed
    return removed


def facts(reach, text, test, compute):
    """`compute(text, test)` (the file's RustFacts), served from the store when an entry of these exact bytes under
    this analysis code exists. The caller has read `text` through the tree, so the read is observed whatever serves
    the parse."""
    live = _LIVE.get((text, test))
    if live is not None:
        STATS["live"] += 1
        return live
    code = _facts_code(reach)
    if code is None:
        STATS["off"] += 1
        return compute(text, test)
    module = sys.modules[type(reach).__module__]
    _check_schema(module)
    key = _facts_key(code, test, text)
    path = os.path.join(store_dir(), key[:2], key + ".json")
    value = _load(path, key, module)
    if value is None:
        STATS["misses"] += 1
        value = compute(text, test)
        _store(path, key, value)
    else:
        STATS["hits"] += 1
    _LIVE[(text, test)] = value
    return value


def prepare(reach):
    """Materialize the whole structural graph (every target root, test views included) under observation, so every
    read of the analysis is recorded, and bind that record to the tree for `current`. Closure, environment and
    architecture decisions stay lazy."""
    tree = reach.tree
    module = sys.modules[type(reach).__module__]
    identity, corpus = _identity(tree), _corpus(tree)
    # Reach's eager initializer may have walked before this snapshot. Those unprotected local walks must not
    # pass on to Graph's later key-input discovery.
    tree._walk_cache.clear()
    tree.unresolved.clear()
    saw = {}
    tree._saw.append(saw)
    try:
        # Reinitialize the same object, not its Tree or Graph owner, under observation. This
        # discards eager module maps and captures all discovery reads before materialization.
        type(reach).__init__(reach, tree)
        for pkg in tree.packages.values():
            for target in pkg.targets:
                for test in (False, True):
                    rk = (pkg.name, target.src, test)
                    reach.mods(rk)
                    reach.own_rules(rk)
                    for f in sorted(tree.tree_of(target.src, test)[0]):
                        for hit in reach.units(rk, f):
                            reach.node(hit)
            if pkg.lib() is not None:
                rk = (pkg.name, pkg.lib().src, False)
                for hit in reach.units(rk, reach.doctests(pkg.name)):
                    reach.node(hit)
        reach.rules()
        reach.dispatches()
        i = 0
        while i < len(reach.nodes):
            reach.edges(i)
            i += 1
    finally:
        tree._saw.pop()
    observed = _observed(tree, saw)
    if _identity(tree) != identity or _corpus(tree) != corpus or not _holds(tree, observed):
        raise module.RecipeError("source tree changed during reach analysis")
    reach._key_identity, reach._key_snapshot, reach._key_observed = identity, corpus, observed


def main(argv):
    root = store_dir()
    if argv == ["stats"]:
        count = size = 0
        for base, _, names in os.walk(root):
            for name in names:
                if name.endswith(".json"):
                    count += 1
                    size += os.lstat(os.path.join(base, name)).st_size
        print(f"{root}: {count} entries, {size} bytes (cap {CAP_BYTES}, kept down to {KEEP_BYTES})")
        return 0
    if argv == ["prune"]:
        print(f"{root}: {prune()} entries removed")
        return 0
    print("usage: reach_cache.py stats | prune", file=sys.stderr)
    return 64


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
