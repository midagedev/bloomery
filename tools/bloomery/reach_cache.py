"""Private JSON snapshots of recipes.Reach's structural graph (never per-recipe reach decisions).

The source corpus is content-addressed independently of git, and observed Tree reads/probes are validated
again before publication and key output. Hydration accepts only the fixed Reach schema below; no code objects,
pickle, dynamic imports or arbitrary object construction enter the cache protocol.
"""
from __future__ import annotations

import dataclasses
import contextlib
import hashlib
import functools
import json
import os
import stat
import sys
import tempfile
import time
import types
import typing

_REACH_LOADED_CODE = sys._getframe().f_code

VERSION = 2
STATE = ("idents", "lib_root", "lib_roots", "_mods", "_bases", "_macros", "_exported", "_text",
         "_traits", "ids", "nodes", "adj", "file_root", "builds", "_rules", "_own_rules",
         "_dispatch", "_rows_nodes", "_enum_nodes", "_family_dirs", "_facts")
_SCALARS = (str, int, bool)
_CONTAINERS = {"list": list, "tuple": tuple, "set": set, "frozenset": frozenset}


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
        if name in ("", ".", "..") or os.path.islink(path):
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


def _encode(value):
    if value is None or type(value) in _SCALARS:
        return value
    if isinstance(value, (list, tuple, set, frozenset)):
        return [type(value).__name__, [v if v is None or type(v) in _SCALARS else _encode(v) for v in value]]
    if isinstance(value, dict):
        return ["dict", [[k if k is None or type(k) in _SCALARS else _encode(k),
                          v if v is None or type(v) in _SCALARS else _encode(v)] for k, v in value.items()]]
    name = type(value).__name__
    if name == "RustFacts":
        return [name, _encode([list(value.units[:value.uses]), list(value.binds), value.exported])]
    if name == "RuleSet":
        return [name, _encode(value.rules)]
    if name in ("Unit", "ImplHead", "Dispatch"):
        return [name, {f: v if (v := getattr(value, f)) is None or type(v) in _SCALARS else _encode(v)
                       for f in _record_types(type(value))}]
    raise ValueError("unsupported Reach field " + name)


def _decode(value, module):
    if value is None or type(value) in _SCALARS:
        return value
    if not isinstance(value, list) or len(value) != 2 or not isinstance(value[0], str):
        raise ValueError("invalid Reach value")
    name, body = value
    if name in _CONTAINERS:
        if not isinstance(body, list):
            raise ValueError("invalid Reach container")
        return _CONTAINERS[name](v if v is None or type(v) in _SCALARS else _decode(v, module) for v in body)
    if name == "dict":
        if not isinstance(body, list):
            raise ValueError("invalid Reach map")
        out = {}
        for row in body:
            if not isinstance(row, list) or len(row) != 2:
                raise ValueError("invalid Reach map entry")
            key, val = (_decode(v, module) for v in row)
            if key in out:
                raise ValueError("duplicate Reach map entry")
            out[key] = val
        return out
    if name == "RustFacts":
        args = _decode(body, module)
        if not isinstance(args, list) or len(args) != 3:
            raise ValueError("invalid Rust facts")
        return module.RustFacts(*args)
    if name == "RuleSet":
        return module.RuleSet(_decode(body, module))
    if name in ("Unit", "ImplHead", "Dispatch"):
        cls = getattr(module, name)
        if not isinstance(body, dict) or set(body) != {f.name for f in dataclasses.fields(cls)}:
            raise ValueError("invalid Reach record schema")
        return cls(**{k: _decode(v, module) for k, v in body.items()})
    raise ValueError("foreign Reach record")


@functools.cache
def _record_types(cls):
    return typing.get_type_hints(cls)


@functools.cache
def _typed(annotation):
    """Compile the fixed schema once; every value still gets strict container/record checks."""
    origin, args = typing.get_origin(annotation), typing.get_args(annotation)
    if origin in (typing.Union, types.UnionType):
        checks = tuple(_typed(a) for a in args)
        return lambda value: any(check(value) for check in checks)
    if origin in (list, set, frozenset) or (origin is tuple and len(args) == 2 and args[1] is Ellipsis):
        element = args[0]
        if element in _SCALARS:
            return lambda value: type(value) is origin and all(type(v) is element for v in value)
        check = _typed(element)
        return lambda value: type(value) is origin and all(check(v) for v in value)
    if origin is dict:
        key, item = (_typed(a) for a in args)
        return lambda value: type(value) is dict and all(key(k) and item(v) for k, v in value.items())
    if origin is tuple:
        checks = tuple(_typed(a) for a in args)
        return lambda value: (type(value) is tuple and len(value) == len(checks)
                              and all(check(v) for v, check in zip(value, checks)))
    if dataclasses.is_dataclass(annotation):
        fields = tuple((k, _typed(a)) for k, a in _record_types(annotation).items())
        return lambda value: type(value) is annotation and all(check(getattr(value, k)) for k, check in fields)
    return lambda value: type(value) is annotation


def _validate(state, tree, module):
    """Validate node/root/unit identity and every edge/rule/dispatch index before touching a live Reach."""
    if not isinstance(state, dict) or set(state) != set(STATE):
        raise ValueError("invalid Reach state schema")
    roots = {(p.name, t.src, test) for p in tree.packages.values() for t in p.targets for test in (False, True)}
    root = tuple[str, str, bool]
    hit = tuple[root, str, int]
    mod = tuple[str, ...]
    schemas = {
        "idents": dict[str, set[str]], "lib_root": dict[str, root], "lib_roots": set[root],
        "_mods": dict[root, dict[mod, str]], "_bases": dict[root, dict[str, list[mod]]],
        "_macros": dict[root, dict[str, list[tuple[str, int]]]],
        "_exported": dict[str, dict[str, list[tuple[str, int]]]], "_text": dict[str, str],
        "_traits": dict[tuple, list[tuple[hit, frozenset[str], str]]],
        "ids": dict[hit, int], "nodes": list[hit], "adj": list[list[tuple[int, str]]],
        "file_root": dict[str, root], "builds": dict[str, str],
        "_rules": module.RuleSet, "_own_rules": dict[root, module.RuleSet],
        "_dispatch": dict[root, module.Dispatch], "_rows_nodes": dict[int, module.Dispatch],
        "_enum_nodes": frozenset[int], "_family_dirs": tuple[str, ...],
        "_facts": dict[tuple[str, bool], module.RustFacts],
    }
    for key, annotation in schemas.items():
        if not _typed(annotation)(state[key]):
            raise ValueError("invalid Reach field type: " + key)
    for (f, test), facts in state["_facts"].items():
        if f not in state["_text"] or not _typed(tuple[module.Unit, ...])(facts.units):
            raise ValueError("invalid Reach fact provenance")
        if not _typed(tuple[tuple[mod, mod, str, bool], ...])(facts.binds) or not _typed(frozenset[str])(facts.exported):
            raise ValueError("invalid Reach import facts")
    for f, text in state["_text"].items():
        if not f.endswith("/<doctests>.rs"):
            with open(os.path.join(tree.root, f), encoding="utf-8", errors="replace") as fh:
                if fh.read() != text:
                    raise ValueError("invalid Reach source provenance")
    nodes, adj, ids = state["nodes"], state["adj"], state["ids"]
    if not isinstance(nodes, list) or not isinstance(adj, list) or len(nodes) != len(adj):
        raise ValueError("invalid Reach adjacency")
    if ids != {h: i for i, h in enumerate(nodes)} or len(ids) != len(nodes):
        raise ValueError("invalid Reach node index")
    def index(i):
        if type(i) is not int or not 0 <= i < len(nodes):
            raise ValueError("invalid Reach node reference")
    for rk, f, k in nodes:
        if rk not in roots or not isinstance(f, str) or type(k) is not int or k < -1:
            raise ValueError("invalid Reach node identity")
        if k >= 0:
            facts = state["_facts"].get((f, rk[2]))
            if not isinstance(facts, module.RustFacts) or k >= len(facts.units):
                raise ValueError("invalid Reach unit index")
    for edges in adj:
        if not isinstance(edges, list):
            raise ValueError("partial Reach graph")
        for i, why in edges:
            index(i)
            if not isinstance(why, str):
                raise ValueError("invalid Reach edge")
    for rk, mods in state["_mods"].items():
        if rk not in roots or not isinstance(mods, dict) or mods.get(()) != rk[1]:
            raise ValueError("invalid Reach module root")
    for rs in [state["_rules"], *state["_own_rules"].values()]:
        if not isinstance(rs, module.RuleSet):
            raise ValueError("invalid Reach rules")
        for i, f, why, groups, name in rs.rules:
            index(i)
            if nodes[i][1] != f or not isinstance(why, str) or (name is not None and not isinstance(name, str)):
                raise ValueError("invalid Reach rule identity")
            for group in groups:
                for j in group:
                    index(j)
    for rk, dispatch in state["_dispatch"].items():
        if rk not in roots or not isinstance(dispatch, module.Dispatch):
            raise ValueError("invalid Reach dispatch")
        index(dispatch.node)
        for i in dispatch.enumerators:
            index(i)
        for i, rows in dispatch.entries.items():
            index(i)
            if any(type(r) is not int or not 0 <= r < len(dispatch.rows) for r in rows):
                raise ValueError("invalid Reach dispatch row")
    if state["_rows_nodes"] != {d.node: d for d in state["_dispatch"].values()}:
        raise ValueError("invalid Reach dispatch index")
    if any(rk not in roots for rk in state["lib_roots"] | set(state["_own_rules"])):
        raise ValueError("invalid Reach root index")
    # Reconstructible indexes must agree with their canonical owners, not merely be in range.
    expected_libs = {p.name: (p.name, p.lib().src, False) for p in tree.packages.values() if p.lib() is not None}
    if state["lib_root"] != expected_libs or state["lib_roots"] != set(expected_libs.values()):
        raise ValueError("invalid Reach library root index")
    expected_files = {}
    for rk in expected_libs.values():
        for f in state["_mods"][rk].values():
            expected_files.setdefault(f, rk)
    if state["file_root"] != expected_files:
        raise ValueError("invalid Reach file owner index")
    for rk, by in state["_bases"].items():
        expected = {}
        for path, f in state["_mods"][rk].items():
            if not path or state["_mods"][rk].get(path[:-1]) != f:
                expected.setdefault(f, []).append(path)
        if by != expected:
            raise ValueError("invalid Reach module base index")
    for rk, names in state["_macros"].items():
        expected = {}
        for f in sorted(set(state["_mods"][rk].values())):
            for name, indexes in state["_facts"][(f, rk[2])].macro_named.items():
                expected.setdefault(name, []).extend((f, i) for i in indexes)
        if names != expected:
            raise ValueError("invalid Reach macro index")
    for pkg, names in state["_exported"].items():
        rk = expected_libs.get(pkg)
        expected = {} if rk is None else {
            name: [(f, i) for f, i in hits if name in state["_facts"][(f, rk[2])].exported]
            for name, hits in state["_macros"][rk].items()}
        if names != expected:
            raise ValueError("invalid Reach exported macro index")
    for values in state["_traits"].values():
        if any(h not in ids for h, _, _ in values):
            raise ValueError("invalid Reach trait unit index")
    if state["_enum_nodes"] != frozenset(i for d in state["_dispatch"].values() for i in d.enumerators):
        raise ValueError("invalid Reach enumerator index")


def prepare(reach):
    tree = reach.tree
    module = sys.modules[type(reach).__module__]
    identity, corpus = _identity(tree), _corpus(tree)
    # Reach's eager initializer may have walked before this snapshot. Neither a miss nor a hit
    # may pass those unprotected local walks on to Graph's later key-input discovery.
    tree._walk_cache.clear()
    tree.unresolved.clear()
    namespace = _digest(tree.root.encode())
    directory = os.path.join(os.path.expanduser("~"), ".cache", "bloomery", "reach", namespace)
    path = os.path.join(directory, "graph.json")
    try:
        st = os.lstat(path)
        if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid() or st.st_mode & 0o077:
            raise ValueError("non-private Reach cache")
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        if (not isinstance(payload, dict) or set(payload) != {"version", "identity", "corpus", "observed", "state", "digest"}
                or payload["version"] != VERSION or payload["identity"] != identity or payload["corpus"] != corpus
                or payload["digest"] != _digest(_json(payload["state"])) or not _holds(tree, payload["observed"])):
            raise ValueError("stale Reach cache")
        state = _decode(payload["state"], module)
        _validate(state, tree, module)
        # Complete validation precedes hydration; partial or foreign data cannot affect a live graph.
        if _identity(tree) != identity or _corpus(tree) != corpus or not _holds(tree, payload["observed"]):
            raise module.RecipeError("source tree changed during reach cache validation")
        reach.__dict__.update(state)
        # Resolver results are a recomputable per-analysis memo, not structural graph state.
        # Discard any eager results made before this validated snapshot was installed.
        reach._resolved.clear()
        reach._key_identity, reach._key_snapshot, reach._key_observed = identity, corpus, payload["observed"]
        return
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, RecursionError):
        pass
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
    state = {k: getattr(reach, k) for k in STATE}
    _validate(state, tree, module)
    encoded = _encode(state)
    payload = {"version": VERSION, "identity": identity, "corpus": corpus, "observed": observed,
               "state": encoded, "digest": _digest(_json(encoded))}
    reach._key_identity, reach._key_snapshot, reach._key_observed = identity, corpus, observed
    # A read-only HOME is a cache miss, not a key failure. Computation above remains authoritative.
    temporary = None
    try:
        os.makedirs(directory, mode=0o700, exist_ok=True)
        ds = os.lstat(directory)
        if not stat.S_ISDIR(ds.st_mode) or ds.st_uid != os.getuid() or ds.st_mode & 0o077:
            return
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, delete=False) as fh:
            temporary = fh.name
            # Keep publication in the C encoder, not dump's millions of per-token Python writes.
            fh.write(json.dumps(payload, separators=(",", ":")))
            fh.flush()
            os.fsync(fh.fileno())
        reach.validate()
        os.replace(temporary, path)
        temporary = None
    except OSError:
        pass
    finally:
        if temporary is not None:
            os.unlink(temporary)
