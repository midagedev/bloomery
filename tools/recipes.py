#!/usr/bin/env python3
"""The justfile's recipes as cargo targets and input files — one parser for check-recipes and affected.

Runs on the Mac: it reads the justfile through `just --dump --dump-format json` (just's own parser),
the workspace through `cargo metadata --format-version 1 --no-deps --offline --locked` (no build, no
network, never rewrites Cargo.lock), and the sources as text. It builds nothing and runs no gate.

    python3 tools/recipes.py check            # the target checks of `just check-recipes`
    python3 tools/recipes.py targets [RECIPE]  # recipe -> cargo targets, scripts, input count
    python3 tools/recipes.py affected [BASE | A..B] [--no-box] [--all-recipes]
    python3 tools/recipes.py why FILE...       # every recipe a file selects, with the chain
    python3 tools/recipes.py key --manifest F [--ledger L] ITEM...  # the green ledger's key per item
    python3 tools/recipes.py box-manifest      # on the box, through tools/box.sh: the key's box part
    python3 tools/recipes.py --self-test

`affected` prints the gate-* recipes whose inputs a diff touches; nothing runs. BASE (default
`main`) compares the working tree, uncommitted and untracked files included, with the merge base of
BASE and HEAD — a round's tree is never committed. `A..B` compares two commits exactly; both sides
are read from `git archive` copies, so the graph is the one those commits had.

A recipe's inputs are:
  - the files of every cargo target it builds or tests: the target's own module tree (`mod`,
    `#[path]`, `include_str!`/`include_bytes!`/`include!`), its package's lib, the libs of its
    workspace dependencies — feature-aware, so an optional dependency counts only when the recipe's
    `--features` enable it, dev-dependencies only for test targets — each package's Cargo.toml and
    build script, and repository paths named by a whole string literal (`"../../tools/ref/prompts.tsv"`
    joined to CARGO_MANIFEST_DIR is read at run time and is in no dep-info);
  - on the box's last build, the bin's top-level dep-info (`target/release/<bin>.d`, which lists every
    local source file of the binary) — a cross-check of the walk, and a union with it;
  - the scripts the recipe text names (`tools/**`, `crates/*/tools/**`) and, transitively, the repository
    files those scripts name;
  - for a box recipe, tools/box.sh and what it sources on every command: tools/ref/ref-paths.sh,
    tools/ref/models/deepseek41.sh and the recipe's profile (`BLOOMERY_MODEL=x` on the line, else
    deepseek2);
  - for a cargo recipe, Cargo.toml, Cargo.lock, rust-toolchain.toml, .cargo/config.toml, and
    .cargo/cuda-oxide.toml when the recipe runs `cargo oxide`;
  - its own text in the justfile (a recipe whose text changed is selected), and its dependencies'
    inputs.

Honest expectation: a change in crates/gpu or crates/model selects every GPU gate — that is the
crate graph's true answer (a one-line enum change in the fault word selects them all, and it should).
The saving is on leaf changes (serve, tokenizer, sampler, one gate binary, a runner, docs); the larger
value is the list itself: no forgotten gate, and a gate left out is a printed record, not a judgment.
What this layer cannot see: data under $BLOOMERY_DATA and the ik trees (not in git), and a card
dependence (BLOOMERY_GATE_CARD). The box's dep-info can be stale or missing; the walk does not depend
on it, and the notes say which targets had one and how old it is.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import difflib
import fcntl
import glob
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GATE_PREFIX = "gate-"
# The checks the lead's batches always run; the diff does not pick them.
ALWAYS = ["check-recipes", "check-rustflags", "check-comments", "check-arch", "check", "fmt-check", "lint"]
CARGO_GLOBALS = ["Cargo.toml", "Cargo.lock", "rust-toolchain.toml", ".cargo/config.toml"]
OXIDE_GLOBALS = [".cargo/cuda-oxide.toml"]
BOX_GLOBALS = ["tools/box.sh", "tools/ref/ref-paths.sh", "tools/ref/models/deepseek41.sh"]
DEFAULT_PROFILE = "deepseek2"
# Runners that execute target/release/<first argument>.
BIN_RUNNERS = ("tools/gpu-gate.sh", "tools/host-gate.sh", "tools/ref/time-gate.sh")
BUILD_SUBS = {"build", "test", "run", "check", "clippy", "bench", "doc", "rustc"}


class RecipeError(Exception):
    """A recipe, a tree or a tool the parser cannot read: a named failure, never a guess."""


# ----------------------------------------------------------------------------------------------
# justfile
# ----------------------------------------------------------------------------------------------


@dataclass
class Recipe:
    name: str
    lines: list[str]
    deps: list[str]
    text: str  # canonical form of params + deps + body; equal text means an unchanged recipe
    line: int = 0  # 1-based header line in the justfile (0 when not found)


def _render_expr(expr) -> str:
    if isinstance(expr, list) and len(expr) == 2 and expr[0] == "variable":
        return expr[1]
    return "…"


def load_justfile(path: str) -> dict[str, Recipe]:
    if shutil.which("just") is None:
        raise RecipeError("`just` is not on PATH — recipes.py reads the justfile through `just --dump`")
    proc = subprocess.run(
        ["just", "--justfile", path, "--working-directory", os.path.dirname(path) or ".", "--dump", "--dump-format", "json"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RecipeError(f"`just --dump` failed on {path} (exit {proc.returncode}): {proc.stderr.strip()}")
    dump = json.loads(proc.stdout)
    with open(path, encoding="utf-8") as fh:
        header_lines = fh.read().split("\n")
    recipes: dict[str, Recipe] = {}
    for name, rec in dump["recipes"].items():
        lines = []
        for frags in rec["body"]:
            out = []
            for frag in frags:
                if isinstance(frag, str):
                    out.append(frag)
                else:
                    out.append("{{" + " ".join(_render_expr(e) for e in frag) + "}}")
            lines.append("".join(out))
        deps = [d["recipe"] for d in rec["dependencies"]]
        text = json.dumps({"p": rec["parameters"], "d": rec["dependencies"], "b": rec["body"], "a": rec["attributes"]}, sort_keys=True)
        recipes[name] = Recipe(name=name, lines=lines, deps=deps, text=text)
    header = re.compile(r"^@?([A-Za-z0-9_-]+)(?:\s[^:]*)?\s*:(?!=)")
    for i, ln in enumerate(header_lines, 1):
        m = header.match(ln)
        if m and m.group(1) in recipes and recipes[m.group(1)].line == 0:
            recipes[m.group(1)].line = i
    return recipes


# ----------------------------------------------------------------------------------------------
# shell text -> simple commands
# ----------------------------------------------------------------------------------------------

_PUNCT = set(";&|()<>")
_REDIRECTS = {">", ">>", "<", "<<", "<<<", ">&", "<&", "&>", ">|", "&>>"}
_KEYWORDS = {"if", "then", "else", "elif", "fi", "do", "done", "while", "until", "for", "case", "esac", "{", "}", "!", "function", "select"}
_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def shell_words(text: str) -> list[str]:
    lex = shlex.shlex(text, posix=True, punctuation_chars=";&|()<>")
    lex.whitespace_split = True
    lex.commenters = ""
    try:
        return list(lex)
    except ValueError as err:
        raise RecipeError(f"cannot split shell text ({err}): {text[:120]}") from err


def simple_commands(words: list[str]) -> list[list[str]]:
    cmds: list[list[str]] = []
    cur: list[str] = []
    skip = False
    for w in words:
        if skip:
            skip = False
            continue
        if w and all(c in _PUNCT for c in w):
            if w in _REDIRECTS:
                skip = True
                continue
            if cur:
                cmds.append(cur)
            cur = []
            continue
        if not cur and w in _KEYWORDS:
            continue
        cur.append(w)
    if cur:
        cmds.append(cur)
    return cmds


def strip_wrappers(cmd: list[str]) -> tuple[dict[str, str], list[str]]:
    """Leading assignments and wrappers (timeout, env, exec, nohup, time) off a simple command."""
    env: dict[str, str] = {}
    i = 0
    while i < len(cmd):
        w = cmd[i]
        if _ASSIGN.match(w):
            k, v = w.split("=", 1)
            env[k] = v
            i += 1
        elif w == "timeout":
            i += 1
            while i < len(cmd) and cmd[i].startswith("-"):
                i += 1
            i += 1  # the duration
        elif w in ("env", "exec", "nohup", "time", "command", "nice"):
            i += 1
        else:
            break
    return env, cmd[i:]


# ----------------------------------------------------------------------------------------------
# cargo invocations
# ----------------------------------------------------------------------------------------------

_CARGO_VALUE = {
    "-p", "--package", "--features", "-F", "--bin", "--test", "--example", "--bench", "--target", "--profile",
    "-j", "--jobs", "--manifest-path", "--message-format", "--color", "--target-dir", "-Z", "--config", "--exclude",
}
_CARGO_FLAG = {
    "--release", "-r", "--lib", "--doc", "--bins", "--tests", "--examples", "--benches", "--all-targets",
    "--workspace", "--all", "--no-default-features", "--all-features", "--locked", "--offline", "--frozen",
    "-q", "--quiet", "-v", "-vv", "--verbose", "--no-run", "--no-fail-fast", "--keep-going", "--future-incompat-report",
}


@dataclass
class Invocation:
    sub: str
    oxide: bool
    packages: list[str] = field(default_factory=list)
    features: list[str] = field(default_factory=list)
    selectors: list[tuple[str, str | None]] = field(default_factory=list)
    workspace: bool = False
    no_default: bool = False
    all_features: bool = False
    all_targets: bool = False
    raw: str = ""


def parse_cargo(args: list[str], raw: str, via_gate_sh: bool = False) -> Invocation | None:
    """`cargo …` words after `cargo` (or tools/gate.sh's arguments). None for a non-build subcommand."""
    if not args:
        raise RecipeError(f"bare `cargo` in: {raw}")
    oxide = False
    if via_gate_sh:
        sub = "test"
        rest = list(args)
        if rest and rest[0] == "--oxide":
            oxide = True
            rest = rest[1:]
    elif args[0] == "oxide":
        oxide = True
        if len(args) < 2:
            raise RecipeError(f"bare `cargo oxide` in: {raw}")
        sub = args[1]
        rest = args[2:]
        if "--" in rest:
            rest = rest[rest.index("--") + 1 :]
        else:
            rest = []
    else:
        sub = args[0]
        rest = args[1:]
    if sub not in BUILD_SUBS:
        return None
    if "--" in rest:
        rest = rest[: rest.index("--")]
    inv = Invocation(sub=sub, oxide=oxide, raw=raw)
    i = 0
    while i < len(rest):
        w = rest[i]
        opt, val = w, None
        if w.startswith("--") and "=" in w:
            opt, val = w.split("=", 1)
        if opt in _CARGO_VALUE:
            if val is None:
                i += 1
                if i >= len(rest):
                    raise RecipeError(f"`{opt}` without a value in: {raw}")
                val = rest[i]
            if opt in ("-p", "--package"):
                inv.packages.append(val)
            elif opt in ("--features", "-F"):
                inv.features.extend(f for f in re.split(r"[,\s]+", val) if f)
            elif opt == "--bin":
                inv.selectors.append(("bin", val))
            elif opt == "--test":
                inv.selectors.append(("test", val))
            elif opt == "--example":
                inv.selectors.append(("example", val))
            elif opt == "--bench":
                inv.selectors.append(("bench", val))
        elif opt in _CARGO_FLAG:
            if opt == "--lib":
                inv.selectors.append(("lib", None))
            elif opt == "--doc":
                inv.selectors.append(("doc", None))
            elif opt == "--bins":
                inv.selectors.append(("bins", None))
            elif opt == "--tests":
                inv.selectors.append(("tests", None))
            elif opt == "--examples":
                inv.selectors.append(("examples", None))
            elif opt == "--benches":
                inv.selectors.append(("benches", None))
            elif opt == "--all-targets":
                inv.all_targets = True
            elif opt in ("--workspace", "--all"):
                inv.workspace = True
            elif opt == "--no-default-features":
                inv.no_default = True
            elif opt == "--all-features":
                inv.all_features = True
        elif w.startswith("-"):
            raise RecipeError(f"cargo option `{w}` is not known to tools/recipes.py (add it to _CARGO_VALUE or _CARGO_FLAG): {raw}")
        # a positional word is a test-name filter (`cargo test <filter>`); it selects no target
        i += 1
    return inv


# ----------------------------------------------------------------------------------------------
# recipe -> commands
# ----------------------------------------------------------------------------------------------

_PATHLIKE = re.compile(r"(?:^|(?<=[\s\"'=:(]))(?:\./)?((?:tools|crates)/[A-Za-z0-9_./-]*[A-Za-z0-9_-])")
# Inside a script a repository path usually hangs off a root variable (`"$ROOT/crates/…"`, `"$HERE/tools/…"`);
# every hit is kept only when it names a file of the tree.
_SCRIPT_PATH = re.compile(r"(?:^|(?<=[\s\"'=:(/]))(?:\./)?((?:tools|crates)/[A-Za-z0-9_./-]*[A-Za-z0-9_-])")
_SCRIPT_EXT = (".sh", ".py")


@dataclass
class RecipeCommands:
    box: bool = False
    env: dict[str, str] = field(default_factory=dict)
    invocations: list[Invocation] = field(default_factory=list)
    runs: list[str] = field(default_factory=list)  # target/release binaries executed
    scripts: list[str] = field(default_factory=list)  # repository scripts executed or sourced
    paths: list[str] = field(default_factory=list)  # every repository-looking path the text names


def _norm(p: str) -> str:
    p = p[2:] if p.startswith("./") else p
    return os.path.normpath(p)


def _classify(cmd: list[str], rc: RecipeCommands, raw: str) -> None:
    env, cmd = strip_wrappers(cmd)
    if not cmd:
        return
    head, args = cmd[0], cmd[1:]
    if head == "cargo":
        inv = parse_cargo(args, raw)
        if inv is not None:
            rc.invocations.append(inv)
        return
    script = None
    sargs: list[str] = []
    if head in ("bash", "sh", "source", ".") and args:
        script, sargs = args[0], args[1:]
    elif head in ("python3", "python") and args:
        script, sargs = args[0], args[1:]
    elif re.match(r"^(\./)?(tools|crates)/", head):
        script, sargs = head, args
    elif re.match(r"^(\./)?target/release/[^/]+$", head):
        rc.runs.append(head.rsplit("/", 1)[1])
        return
    if script is None:
        return
    s = _norm(script)
    if re.match(r"^(tools|crates)/", s):
        rc.scripts.append(s)
    if s == "tools/gate.sh":
        inv = parse_cargo(sargs, raw, via_gate_sh=True)
        if inv is not None:
            rc.invocations.append(inv)
    elif s in BIN_RUNNERS and sargs:
        rc.runs.append(sargs[0])


def recipe_commands(recipe: Recipe) -> RecipeCommands:
    rc = RecipeCommands()
    for line in recipe.lines:
        stripped = line.lstrip("@-").strip()
        if not stripped or stripped.startswith("#"):
            continue
        local = shell_words(stripped)
        for cmd in simple_commands(local):
            env, rest = strip_wrappers(cmd)
            if rest and _norm(rest[0]) == "tools/box.sh":
                rc.box = True
                rc.env.update(env)
                remote = " ".join(rest[1:])
                for p in _PATHLIKE.findall(remote):
                    rc.paths.append(_norm(p))
                for rcmd in simple_commands(shell_words(remote)):
                    _classify(rcmd, rc, remote)
            else:
                for p in _PATHLIKE.findall(" ".join(cmd)):
                    rc.paths.append(_norm(p))
                _classify(cmd, rc, stripped)
    return rc


# ----------------------------------------------------------------------------------------------
# the workspace: packages, targets, features
# ----------------------------------------------------------------------------------------------


@dataclass
class TargetMeta:
    kind: str  # lib | bin | test | example | bench | build
    name: str
    src: str  # relative to the tree root
    required: list[str]


@dataclass
class Dep:
    key: str  # the name the manifest uses (rename or package name)
    pkg: str
    kind: str  # normal | dev | build
    optional: bool
    features: list[str]
    default: bool


@dataclass
class Package:
    name: str
    dir: str
    manifest: str
    features: dict[str, list[str]]
    deps: list[Dep]
    targets: list[TargetMeta]

    def lib(self) -> TargetMeta | None:
        return next((t for t in self.targets if t.kind == "lib"), None)

    def find(self, kind: str, name: str) -> TargetMeta | None:
        return next((t for t in self.targets if t.kind == kind and t.name == name), None)


def cargo_metadata(root: str) -> dict:
    if shutil.which("cargo") is None:
        raise RecipeError("`cargo` is not on PATH — recipes.py reads the workspace through `cargo metadata`")
    proc = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps", "--offline", "--locked"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RecipeError(f"`cargo metadata` failed in {root} (exit {proc.returncode}): {proc.stderr.strip()[-600:]}")
    return json.loads(proc.stdout)


def _kind_of(kinds: list[str]) -> str:
    for k in kinds:
        if k in ("lib", "rlib", "dylib", "cdylib", "staticlib", "proc-macro"):
            return "lib"
        if k == "custom-build":
            return "build"
        if k in ("bin", "test", "example", "bench"):
            return k
    raise RecipeError(f"unknown cargo target kind {kinds}")


class Tree:
    """One checkout of the repository: its packages and the files each target reads."""

    def __init__(self, root: str, meta: dict | None = None):
        self.root = os.path.realpath(root)
        meta = meta if meta is not None else cargo_metadata(self.root)
        wroot = os.path.realpath(meta["workspace_root"])
        members = set(meta["workspace_members"])
        self.packages: dict[str, Package] = {}
        for p in meta["packages"]:
            if p["id"] not in members:
                continue
            pdir = os.path.relpath(os.path.dirname(os.path.realpath(p["manifest_path"])), wroot)
            deps = []
            for d in p["dependencies"]:
                deps.append(
                    Dep(
                        key=d.get("rename") or d["name"],
                        pkg=d["name"],
                        kind=d.get("kind") or "normal",
                        optional=bool(d.get("optional")),
                        features=list(d.get("features") or []),
                        default=bool(d.get("uses_default_features", True)),
                    )
                )
            targets = [
                TargetMeta(
                    kind=_kind_of(t["kind"]),
                    name=t["name"],
                    src=os.path.relpath(os.path.realpath(t["src_path"]), wroot),
                    required=list(t.get("required-features") or []),
                )
                for t in p["targets"]
            ]
            self.packages[p["name"]] = Package(
                name=p["name"],
                dir=pdir,
                manifest=os.path.join(pdir, "Cargo.toml"),
                features={k: list(v) for k, v in p["features"].items()},
                deps=deps,
                targets=targets,
            )
        self._walk_cache: dict[str, tuple[set[str], list[tuple[str, str]]]] = {}
        self.unresolved: list[str] = []
        self.depinfo: dict[str, tuple[set[str], float]] = {}  # bin name -> (files, mtime)

    # ---------------- file walk ----------------

    def exists(self, rel: str) -> bool:
        return os.path.isfile(os.path.join(self.root, rel))

    def isdir(self, rel: str) -> bool:
        return os.path.isdir(os.path.join(self.root, rel))

    def read(self, rel: str) -> str:
        with open(os.path.join(self.root, rel), encoding="utf-8", errors="replace") as fh:
            return fh.read()

    _MOD = re.compile(r"^(\s*)(?:pub(?:\s*\([^)]*\))?\s+)?mod\s+(?:r#)?([A-Za-z_][A-Za-z0-9_]*)\s*;")
    _INLINE = re.compile(r"^(\s*)(?:pub(?:\s*\([^)]*\))?\s+)?mod\s+(?:r#)?([A-Za-z_][A-Za-z0-9_]*)\s*\{\s*$")
    _PATH = re.compile(r"#\[\s*path\s*=\s*\"([^\"]+)\"\s*\]")
    _INCLUDE = re.compile(r"\binclude(?:_str|_bytes)?!\s*\(\s*\"([^\"]+)\"\s*\)")
    _LITERAL = re.compile(r"\"(/?(?:\.\./)*(?:tools|crates|docs)/[A-Za-z0-9_./-]+)\"")

    def walk(self, src: str) -> tuple[set[str], list[tuple[str, str]]]:
        """The module tree rooted at `src`: its files, and (file, literal) repository-path literals."""
        files: set[str] = set()
        literals: list[tuple[str, str]] = []
        self._walk(src, os.path.dirname(src), files, literals)
        return files, literals

    def _walk(self, rel: str, moddir: str, files: set[str], literals: list[tuple[str, str]]) -> None:
        if rel in files:
            return
        files.add(rel)
        fdir = os.path.dirname(rel)
        pending_path: str | None = None
        inline: list[tuple[int, str]] = []
        attr_depth = 0  # open brackets of a multi-line attribute (`#[allow(\n…\n)]`)
        for line in self.read(rel).split("\n"):
            s = line.strip()
            if not s or s.startswith("//"):
                continue
            if attr_depth > 0:
                attr_depth += s.count("[") - s.count("]")
                continue
            indent = len(line) - len(line.lstrip())
            if s == "}" and inline and inline[-1][0] == indent:
                inline.pop()
                continue
            for lit in self._LITERAL.findall(line):
                literals.append((rel, lit))
            for inc in self._INCLUDE.findall(line):
                p = os.path.normpath(os.path.join(fdir, inc))
                if self.exists(p):
                    files.add(p)
                else:
                    self.unresolved.append(f"{rel}: include of {inc} ({p} not found)")
            pm = self._PATH.search(line)
            m = self._MOD.match(line)
            im = self._INLINE.match(line)
            if pm and not m:
                pending_path = pm.group(1)
                continue
            if m:
                name = m.group(2)
                path_attr = pm.group(1) if pm else pending_path
                pending_path = None
                inner = [n for (i, n) in inline if i < indent]
                if path_attr is not None:
                    base = os.path.join(fdir, *inner) if inner else fdir
                    child = os.path.normpath(os.path.join(base, path_attr))
                    if not self.exists(child):
                        self.unresolved.append(f"{rel}: #[path = \"{path_attr}\"] mod {name} ({child} not found)")
                        continue
                    # a #[path] file is a mod-rs file: its children live beside it
                    self._walk(child, os.path.dirname(child), files, literals)
                    continue
                base = os.path.join(moddir, *inner) if inner else moddir
                cand = [os.path.normpath(os.path.join(base, name + ".rs")), os.path.normpath(os.path.join(base, name, "mod.rs"))]
                hit = next((c for c in cand if self.exists(c)), None)
                if hit is None:
                    self.unresolved.append(f"{rel}: mod {name} (neither {cand[0]} nor {cand[1]})")
                    continue
                child_dir = os.path.dirname(hit) if hit.endswith("mod.rs") else hit[: -len(".rs")]
                self._walk(hit, child_dir, files, literals)
                continue
            if im:
                inline.append((len(im.group(1)), im.group(2)))
                pending_path = None
                continue
            if s.startswith("#["):
                attr_depth = max(0, s.count("[") - s.count("]"))
                continue
            pending_path = None

    def tree_of(self, src: str) -> tuple[set[str], list[tuple[str, str]]]:
        if src not in self._walk_cache:
            self._walk_cache[src] = self.walk(src)
        return self._walk_cache[src]

    # ---------------- features and dependencies ----------------

    def feature_closure(self, pkg: Package, requested: list[str], default: bool = True, all_features: bool = False) -> tuple[set[str], set[str], list[str]]:
        """(enabled features, enabled optional dependency keys, unknown feature names)."""
        opt_keys = {d.key for d in pkg.deps if d.optional}
        feats: set[str] = set()
        deps_on: set[str] = set()
        unknown: list[str] = []
        stack = list(pkg.features) if all_features else list(requested)
        if default and "default" in pkg.features:
            stack.append("default")
        if all_features:
            deps_on |= opt_keys
        while stack:
            f = stack.pop()
            if f in feats:
                continue
            if f in pkg.features:
                feats.add(f)
                for v in pkg.features[f]:
                    if v.startswith("dep:"):
                        deps_on.add(v[4:])
                    elif "/" in v:
                        d = v.split("/", 1)[0]
                        if not d.endswith("?"):
                            deps_on.add(d)
                    else:
                        stack.append(v)
            elif f in opt_keys:
                deps_on.add(f)
            else:
                unknown.append(f)
        return feats, deps_on, unknown

    def lib_files(self, name: str) -> dict[str, str]:
        """The files a dependent sees of package `name`: its lib tree, manifest and build script."""
        pkg = self.packages[name]
        out: dict[str, str] = {pkg.manifest: f"{name} Cargo.toml"}
        lib = pkg.lib()
        if lib is not None:
            files, lits = self.tree_of(lib.src)
            for f in files:
                out.setdefault(f, f"lib {name}")
            for f in self._resolve_literals(pkg, lits):
                out.setdefault(f, f"lib {name} (path literal)")
        for t in pkg.targets:
            if t.kind == "build":
                files, _ = self.tree_of(t.src)
                for f in files:
                    out.setdefault(f, f"build script {name}")
        return out

    def _resolve_literals(self, pkg: Package, lits: list[tuple[str, str]]) -> list[str]:
        out = []
        for _, lit in lits:
            if lit.startswith("/") or lit.startswith("../"):
                p = os.path.normpath(os.path.join(pkg.dir, lit.lstrip("/")))
            else:
                p = os.path.normpath(lit)
            if self.exists(p):
                out.append(p)
            elif self.isdir(p):
                out.append(p.rstrip("/") + "/")
        return out

    def closure(self, pkg: Package, features: list[str], dev: bool, default: bool = True, all_features: bool = False) -> dict[str, str]:
        """Workspace packages a target of `pkg` links: name -> the edge that brought it in."""
        _, deps_on, _ = self.feature_closure(pkg, features, default, all_features)
        seen: dict[str, str] = {}
        stack: list[tuple[str, list[str], bool, str]] = []
        for d in pkg.deps:
            if d.pkg not in self.packages:
                continue
            if d.kind == "dev" and not dev:
                continue
            if d.optional and d.key not in deps_on:
                continue
            stack.append((d.pkg, d.features, d.default, f"{pkg.name} -> {d.pkg}"))
        while stack:
            name, feats, dflt, why = stack.pop()
            if name in seen:
                continue
            seen[name] = why
            q = self.packages[name]
            _, q_on, _ = self.feature_closure(q, feats, dflt)
            for d in q.deps:
                if d.pkg not in self.packages or d.kind == "dev":
                    continue
                if d.optional and d.key not in q_on:
                    continue
                stack.append((d.pkg, d.features, d.default, f"{why} -> {d.pkg}"))
        return seen

    def target_files(self, pkg_name: str, kind: str, name: str | None, features: list[str], default: bool = True, all_features: bool = False) -> dict[str, str]:
        """file -> origin for one target. kind: lib bin test example bench doctest libtest."""
        pkg = self.packages[pkg_name]
        out: dict[str, str] = {}
        dev = kind in ("test", "bench", "doctest", "libtest", "example")
        if kind in ("lib", "doctest", "libtest"):
            own = pkg.lib()
            label = {"lib": "lib", "doctest": "doctests", "libtest": "lib tests"}[kind]
        else:
            own = pkg.find(kind, name or "")
            label = f"{kind} {name}"
        if own is None:
            raise RecipeError(f"{pkg_name} has no {kind} target {name or ''}".rstrip())
        files, lits = self.tree_of(own.src)
        for f in files:
            out[f] = f"{pkg_name} {label}"
        for f in self._resolve_literals(pkg, lits):
            out.setdefault(f, f"{pkg_name} {label} (path literal)")
        for f, why in self.lib_files(pkg_name).items():
            out.setdefault(f, why)
        for dep, why in self.closure(pkg, features, dev, default, all_features).items():
            for f, fwhy in self.lib_files(dep).items():
                out.setdefault(f, f"{fwhy} ({why})")
        if kind == "bin" and name in self.depinfo:
            dfiles, _ = self.depinfo[name]
            if own.src in dfiles:
                for f in dfiles:
                    if f not in out and (self.exists(f)):
                        out[f] = f"dep-info of {name}"
        return out

    # ---------------- invocation -> targets ----------------

    def resolve(self, inv: Invocation) -> tuple[list[tuple[str, str, str | None, list[str]]], list[str]]:
        """[(package, kind, name, features)], [errors]. Test kinds: libtest for `--lib`, doctest for `--doc`."""
        errors: list[str] = []
        out: list[tuple[str, str, str | None, list[str]]] = []
        pkgs = inv.packages or sorted(self.packages)
        test = inv.sub in ("test", "bench")
        for pname in pkgs:
            if pname not in self.packages:
                errors.append(f"-p {pname}: no such workspace package")
                continue
            pkg = self.packages[pname]
            feats = []
            for f in inv.features:
                if "{{" in f:
                    continue  # a recipe parameter; resolved only at run time
                if "/" in f:
                    fp, ff = f.split("/", 1)
                    if fp not in self.packages:
                        errors.append(f"--features {f}: no workspace package {fp}")
                    elif fp == pname:
                        feats.append(ff)
                    elif ff not in self.packages[fp].features:
                        errors.append(f"--features {f}: {fp} has no feature {ff}")
                else:
                    feats.append(f)
            enabled, _, unknown = self.feature_closure(pkg, feats, not inv.no_default, inv.all_features)
            if inv.packages:
                for u in unknown:
                    errors.append(f"--features {u}: {pname} has no such feature")

            def req_ok(t: TargetMeta) -> bool:
                return inv.all_features or set(t.required) <= enabled

            def add(kind: str, t: TargetMeta | None, name: str | None) -> None:
                out.append((pname, kind, name, feats))

            sels = list(inv.selectors)
            if inv.all_targets:
                sels = [("lib", None), ("bins", None), ("tests", None), ("examples", None), ("benches", None)]
                if test:
                    sels.append(("doc", None))
            if not sels:
                if test:
                    sels = [("lib", None), ("bins", None), ("tests", None), ("examples", None), ("doc", None)]
                elif inv.sub == "run":
                    bins = [t for t in pkg.targets if t.kind == "bin"]
                    if len(bins) != 1:
                        errors.append(f"cargo run -p {pname} names no --bin and the package has {len(bins)} bins")
                        continue
                    sels = [("bin", bins[0].name)]
                else:
                    sels = [("lib", None), ("bins", None)]
                implicit = True
            else:
                implicit = False
            for kind, name in sels:
                if kind == "lib":
                    lib = pkg.lib()
                    if lib is None:
                        if not implicit and not inv.all_targets:
                            errors.append(f"--lib: {pname} has no lib target")
                        continue
                    add("libtest" if test else "lib", lib, None)
                elif kind == "doc":
                    if pkg.lib() is None:
                        if not implicit and not inv.all_targets:
                            errors.append(f"--doc: {pname} has no lib target")
                        continue
                    add("doctest", pkg.lib(), None)
                elif kind in ("bin", "test", "example", "bench"):
                    if name is not None and "{{" in name:
                        continue  # a recipe parameter; resolved only at run time
                    t = pkg.find(kind, name or "")
                    if t is None:
                        have = sorted(x.name for x in pkg.targets if x.kind == kind)
                        near = difflib.get_close_matches(name or "", have, n=3, cutoff=0.8)
                        errors.append(f"--{kind} {name}: {pname} has no such {kind} target" + (f" (near: {', '.join(near)})" if near else ""))
                        continue
                    if not req_ok(t):
                        errors.append(f"--{kind} {name}: needs features {t.required}, the recipe enables {sorted(enabled)}")
                        continue
                    add(kind, t, name)
                elif kind in ("bins", "tests", "examples", "benches"):
                    k = kind[:-1]
                    for t in pkg.targets:
                        if t.kind == k and req_ok(t):
                            add(k, t, t.name)
                    if kind == "tests" and test and pkg.lib() is not None and not implicit:
                        add("libtest", pkg.lib(), None)
        return out, errors


# ----------------------------------------------------------------------------------------------
# recipe inputs
# ----------------------------------------------------------------------------------------------


@dataclass
class RecipeInputs:
    files: dict[str, str]  # file -> origin
    prefixes: dict[str, str]  # directory prefix "a/b/" -> origin
    targets: list[tuple[str, str, str | None, list[str]]]
    errors: list[str]
    commands: RecipeCommands

    def match(self, path: str) -> str | None:
        if path in self.files:
            return self.files[path]
        for p, why in self.prefixes.items():
            if path.startswith(p):
                return why
        return None


class Graph:
    def __init__(self, tree: Tree, recipes: dict[str, Recipe]):
        self.tree = tree
        self.recipes = recipes
        self._inputs: dict[str, RecipeInputs] = {}
        self._script_cache: dict[str, dict[str, str]] = {}

    def script_closure(self, rel: str) -> dict[str, str]:
        """Repository files a script names, transitively (comments skipped)."""
        if rel in self._script_cache:
            return self._script_cache[rel]
        out: dict[str, str] = {}
        self._script_cache[rel] = out
        stack = [rel]
        seen: set[str] = set()
        while stack:
            s = stack.pop()
            if s in seen or not self.tree.exists(s):
                continue
            seen.add(s)
            out.setdefault(s, f"script {rel}" if s != rel else "script")
            c_like = s.endswith((".cpp", ".h", ".c", ".cc", ".hpp"))
            sdir = os.path.dirname(s)
            for line in self.tree.read(s).split("\n"):
                st = line.strip()
                if not st:
                    continue
                if c_like:
                    if st.startswith("//"):
                        continue
                elif st.startswith("#"):
                    continue
                for p in _SCRIPT_PATH.findall(line):
                    p = _norm(p)
                    if self.tree.exists(p):
                        stack.append(p)
                for b in re.findall(r"([A-Za-z0-9_.-]+\.(?:sh|py|tsv|cpp|h|txt|json|jinja))\b", line):
                    p = os.path.normpath(os.path.join(sdir, b))
                    if self.tree.exists(p):
                        stack.append(p)
        for s in seen:
            out.setdefault(s, f"script {rel}")
        return out

    def inputs(self, name: str, _stack: tuple[str, ...] = ()) -> RecipeInputs:
        if name in self._inputs:
            return self._inputs[name]
        if name in _stack:
            raise RecipeError(f"recipe dependency cycle: {' -> '.join(_stack + (name,))}")
        recipe = self.recipes[name]
        errors: list[str] = []
        try:
            rc = recipe_commands(recipe)
        except RecipeError as err:
            ri = RecipeInputs({}, {}, [], [str(err)], RecipeCommands())
            self._inputs[name] = ri
            return ri
        files: dict[str, str] = {}
        prefixes: dict[str, str] = {}
        targets: list[tuple[str, str, str | None, list[str]]] = []
        for inv in rc.invocations:
            tl, errs = self.tree.resolve(inv)
            errors.extend(errs)
            for pname, kind, tname, feats in tl:
                targets.append((pname, kind, tname, feats))
                label = f"{kind} {tname}" if tname else kind
                for f, why in self.tree.target_files(pname, kind, tname, feats, not inv.no_default, inv.all_features).items():
                    if f.endswith("/"):
                        prefixes.setdefault(f, f"{label} <- {why}")
                    else:
                        files.setdefault(f, f"{label} <- {why}" if not why.startswith(f"{pname} {label}") else label)
            for g in CARGO_GLOBALS:
                files.setdefault(g, "cargo (every build)")
            if inv.oxide:
                for g in OXIDE_GLOBALS:
                    files.setdefault(g, "cargo oxide")
        if rc.box:
            for g in BOX_GLOBALS:
                files.setdefault(g, "tools/box.sh (every box command)")
            profile = rc.env.get("BLOOMERY_MODEL", DEFAULT_PROFILE)
            files.setdefault(f"tools/ref/models/{profile}.sh", f"profile {profile}")
        for s in rc.scripts + rc.paths:
            if self.tree.exists(s):
                for f, why in self.script_closure(s).items():
                    files.setdefault(f, why)
            elif self.tree.isdir(s):
                prefixes.setdefault(s.rstrip("/") + "/", "named directory")
            elif s in rc.scripts or s.endswith(_SCRIPT_EXT):
                errors.append(f"names {s}, which is not in the tree")
        built = {t for (_, k, t, _) in targets if k == "bin"}
        for dep in recipe.deps:
            if dep not in self.recipes:
                errors.append(f"depends on {dep}, which is not a recipe")
                continue
            di = self.inputs(dep, _stack + (name,))
            for f, why in di.files.items():
                files.setdefault(f, f"{why} (via {dep})")
            for p, why in di.prefixes.items():
                prefixes.setdefault(p, f"{why} (via {dep})")
            built |= {t for (_, k, t, _) in di.targets if k == "bin"}
        for b in rc.runs:
            if "{{" in b or b.startswith("$"):
                continue
            if b not in built:
                errors.append(f"runs target/release/{b}, which the recipe does not build")
        ri = RecipeInputs(files, prefixes, targets, errors, rc)
        self._inputs[name] = ri
        return ri


# ----------------------------------------------------------------------------------------------
# check
# ----------------------------------------------------------------------------------------------


def check(tree: Tree, recipes: dict[str, Recipe], prefix: str = GATE_PREFIX) -> list[str]:
    graph = Graph(tree, recipes)
    problems: list[str] = []
    for name in sorted(recipes, key=lambda n: recipes[n].line):
        ri = graph.inputs(name)
        where = f"justfile:{recipes[name].line} {name}"
        for e in ri.errors:
            problems.append(f"{where}: {e}")
        if name.startswith(prefix) and not ri.targets:
            problems.append(f"{where}: a gate recipe that builds or tests no cargo target")
    for u in sorted(set(tree.unresolved)):
        problems.append(f"source walk: {u}")
    return problems


# ----------------------------------------------------------------------------------------------
# dep-info from the box
# ----------------------------------------------------------------------------------------------


def parse_depinfo(text: str) -> tuple[str, set[str], str]:
    """(target path, dependency paths, root) of one .d file; root is the part before /target/."""
    joined = re.sub(r"\\\n", " ", text)
    for line in joined.split("\n"):
        if not line.strip() or line.startswith("#"):
            continue
        m = re.match(r"^((?:[^:\\]|\\.)+):(?:\s(.*))?$", line)
        if not m:
            continue
        target = m.group(1).replace("\\ ", " ")
        deps = {d.replace("\\ ", " ") for d in re.split(r"(?<!\\)\s+", m.group(2) or "") if d}
        root = target.split("/target/", 1)[0] if "/target/" in target else ""
        return target, deps, root
    raise RecipeError("a dep-info file with no rule")


def fetch_depinfo(host: str, remote: str) -> list[tuple[str, float, str]]:
    script = (
        f"cd {remote}/target || exit 3; "
        "for f in release/*.d release/deps/*.d release/build/bloomery-*/*.d debug/*.d debug/deps/*.d; do "
        '[ -f "$f" ] || continue; printf "\\n==> %s %s\\n" "$f" "$(stat -c %Y "$f")"; cat "$f"; done'
    )
    try:
        proc = subprocess.run(["ssh", "-o", "BatchMode=yes", host, script], capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired as err:
        raise RecipeError(f"reading dep-info over ssh {host} timed out; --no-box skips it") from err
    if proc.returncode != 0:
        raise RecipeError(f"reading dep-info over ssh {host}:{remote}/target failed (exit {proc.returncode}): {proc.stderr.strip()[-300:]}; --no-box skips it")
    out = []
    for chunk in proc.stdout.split("\n==> ")[1:]:
        head, _, body = chunk.partition("\n")
        path, mtime = head.rsplit(" ", 1)
        out.append((path, float(mtime), body))
    return out


def attach_depinfo(tree: Tree, entries: list[tuple[str, float, str]]) -> list[str]:
    """Top-level bin dep-info (release/<bin>.d) onto the tree; returns notes."""
    notes = []
    bins = {t.name for p in tree.packages.values() for t in p.targets if t.kind == "bin"}
    for path, mtime, body in entries:
        base = os.path.basename(path)[: -len(".d")]
        if "/" in path.replace("release/", "", 1).replace("debug/", "", 1):
            continue  # deps/ and build/ units: crate-level, not a bin's closure
        if base not in bins:
            continue
        try:
            _, deps, root = parse_depinfo(body)
        except RecipeError:
            notes.append(f"dep-info {path} has no rule")
            continue
        rel = {os.path.relpath(d, root) for d in deps if root and d.startswith(root + "/")}
        old = tree.depinfo.get(base)
        if old is None or mtime > old[1]:
            tree.depinfo[base] = (rel, mtime)
    return notes


# ----------------------------------------------------------------------------------------------
# affected
# ----------------------------------------------------------------------------------------------


def git(*args: str, cwd: str = ROOT) -> str:
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RecipeError(f"git {' '.join(args)} failed (exit {proc.returncode}): {proc.stderr.strip()}")
    return proc.stdout


def archive(rev: str, into: str) -> str:
    os.makedirs(into, exist_ok=True)
    arch = subprocess.Popen(["git", "archive", "--format=tar", rev], cwd=ROOT, stdout=subprocess.PIPE)
    tar = subprocess.run(["tar", "-x", "-C", into], stdin=arch.stdout, capture_output=True)
    arch.stdout.close()
    rc = arch.wait()
    if rc != 0 or tar.returncode != 0:
        raise RecipeError(f"git archive {rev} | tar failed (git {rc}, tar {tar.returncode}): {tar.stderr.decode(errors='replace')[-300:]}")
    return into


@dataclass
class Side:
    tree: Tree
    recipes: dict[str, Recipe]
    graph: Graph


def make_side(root: str) -> Side:
    tree = Tree(root)
    recipes = load_justfile(os.path.join(tree.root, "justfile"))
    return Side(tree, recipes, Graph(tree, recipes))


@dataclass
class Selection:
    recipe: str
    files: list[str]
    why: str


def select(changed: list[str], a: Side | None, b: Side, prefix: str) -> tuple[list[Selection], list[str], list[str]]:
    """(selections in justfile order, unmapped files, notes)."""
    names = [n for n in sorted(b.recipes, key=lambda n: b.recipes[n].line) if n.startswith(prefix)]
    hits: dict[str, list[tuple[str, str]]] = {n: [] for n in names}
    mapped: set[str] = set()
    notes: list[str] = []
    for f in changed:
        if f == "justfile":
            continue
        side = b if b.tree.exists(f) or a is None else a
        for n in names:
            if n not in side.recipes:
                continue
            why = side.graph.inputs(n).match(f)
            if why is not None:
                hits[n].append((f, why))
                mapped.add(f)
    if "justfile" in changed:
        for n in names:
            old = a.recipes.get(n) if a is not None else None
            if old is None or old.text != b.recipes[n].text:
                hits[n].append(("justfile", "new recipe" if old is None else "recipe text"))
                mapped.add("justfile")
    sels = []
    for n in names:
        if hits[n]:
            ranked = sorted(hits[n], key=lambda h: _distance(h[1]))
            sels.append(Selection(n, [f for f, _ in ranked], ranked[0][1]))
    unmapped = []
    for f in changed:
        if f in mapped:
            continue
        side = b if b.tree.exists(f) or a is None else a
        hint = "no gate-* recipe text changed" if f == "justfile" else orphan_hint(side, f)
        unmapped.append(f + (f"  ({hint})" if hint else ""))
    return sels, unmapped, notes


def _distance(why: str) -> int:
    """How far an origin is from the recipe's own target: own files first, globals last."""
    if why in ("recipe text", "new recipe"):
        return 0
    if why.startswith(("cargo", "tools/box.sh", "profile ")):
        return 90
    if why.startswith("script"):
        return 50
    return (10 if "<-" in why else 0) + 10 * why.count("->")


def orphan_hint(side: Side, f: str) -> str:
    """For a source file no gate reads: the cargo targets that do read it."""
    users = []
    for p in side.tree.packages.values():
        for t in p.targets:
            if t.kind == "build":
                continue
            kind = "libtest" if t.kind == "lib" else t.kind
            try:
                files = side.tree.target_files(p.name, kind, None if t.kind == "lib" else t.name, list(p.features), True, True)
            except RecipeError:
                continue
            if f in files:
                users.append(f"{t.kind} {t.name}")
    if not users:
        return ""
    return "read by " + ", ".join(sorted(set(users))[:4]) + (" …" if len(users) > 4 else "") + " — no gate-* recipe builds it"


def changed_files(spec: str) -> tuple[list[str], str | None, str | None]:
    """(files, A rev, B rev or None for the working tree)."""
    if ".." in spec:
        a, b = spec.split("..", 1)
        a = a or "HEAD"
        b = b or "HEAD"
        files = git("diff", "--name-only", "--no-renames", a, b).split()
        return sorted(set(files)), a, b
    base = git("merge-base", spec, "HEAD").strip()
    files = git("diff", "--name-only", "--no-renames", base).split()
    files += git("ls-files", "--others", "--exclude-standard").split()
    return sorted(set(files)), base, None


def cmd_affected(args: argparse.Namespace) -> int:
    changed, a_rev, b_rev = changed_files(args.base)
    prefix = "" if args.all_recipes else GATE_PREFIX
    tmp = tempfile.mkdtemp(prefix="recipes-affected-")
    try:
        b = make_side(archive(b_rev, os.path.join(tmp, "b")) if b_rev else ROOT)
        a = make_side(archive(a_rev, os.path.join(tmp, "a"))) if a_rev else None
        notes: list[str] = []
        if not args.no_box:
            entries = fetch_depinfo(args.box, args.depinfo_remote)
            notes += attach_depinfo(b.tree, entries)
            if a is not None:
                attach_depinfo(a.tree, entries)
        sels, unmapped, more = select(changed, a, b, prefix)
        notes += more
        total = sum(1 for n in b.recipes if n.startswith(prefix))
        rng = f"{a_rev[:9]}..{b_rev[:9]}" if b_rev else f"{a_rev[:9]}..(working tree)"
        print(f"affected: {rng} — {len(changed)} changed files, {len(sels)} of {total} {prefix or ''}recipes selected")
        width = max([len(s.recipe) for s in sels] + [10])
        for s in sels:
            more_n = f" (+{len(s.files) - 1})" if len(s.files) > 1 else ""
            print(f"  {s.recipe:<{width}}  {s.files[0]}{more_n}  [{s.why}]")
        print("recipes: " + " ".join(s.recipe for s in sels))
        print("always: " + " ".join(ALWAYS))
        print(f"unmapped ({len(unmapped)}):")
        for u in unmapped:
            print(f"  {u}")
        if not args.no_box:
            di = b.tree.depinfo
            if di:
                lo = min(m for _, m in di.values())
                hi = max(m for _, m in di.values())
                notes.append(
                    f"dep-info: {len(di)} bins from {args.box}:{args.depinfo_remote}/target, "
                    f"{_ts(lo)} … {_ts(hi)}; test targets have none there (release/deps is read when present) — static walk only"
                )
            sel_bins = sorted({t for s in sels for (_, k, t, _) in b.graph.inputs(s.recipe).targets if k == "bin"})
            by_time: dict[str, list[str]] = {}
            for t in sorted(sel_bins, key=lambda t: di.get(t, (None, 0.0))[1]):
                if t in di:
                    by_time.setdefault(_ts(di[t][1]), []).append(t)
            for ts, ts_bins in by_time.items():
                notes.append(f"dep-info of {' '.join(ts_bins)} is from {ts}")
            missing = [t for t in sel_bins if t not in di]
            if missing:
                notes.append(f"no dep-info for {len(missing)} selected bins (static walk only): {' '.join(missing)}")
            extra = depinfo_only(b)
            if extra:
                notes.append(f"dep-info names {len(extra)} files the static walk does not (stale dep-info or a walk gap): " + " ".join(extra[:6]))
            elif di:
                notes.append("dep-info cross-check: every existing file a bin's dep-info names is in the static walk")
        else:
            notes.append("dep-info not read (--no-box): static walk only")
        print("notes:")
        for n in notes:
            print(f"  {n}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


def depinfo_only(side: Side) -> list[str]:
    """Files a bin's dep-info lists that exist in the tree but the walk alone did not find."""
    extra = set()
    for p in side.tree.packages.values():
        for t in p.targets:
            if t.kind != "bin" or t.name not in side.tree.depinfo:
                continue
            dfiles, _ = side.tree.depinfo[t.name]
            saved = side.tree.depinfo.pop(t.name)
            try:
                walk = side.tree.target_files(p.name, "bin", t.name, list(p.features), True, True)
            finally:
                side.tree.depinfo[t.name] = saved
            for f in dfiles:
                if f not in walk and side.tree.exists(f):
                    extra.add(f"{t.name}:{f}")
    return sorted(extra)


def _ts(t: float) -> str:
    return datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M")


def cmd_targets(args: argparse.Namespace) -> int:
    side = make_side(ROOT)
    names = args.recipes or [n for n in sorted(side.recipes, key=lambda n: side.recipes[n].line) if n.startswith(GATE_PREFIX)]
    for n in names:
        if n not in side.recipes:
            raise RecipeError(f"no recipe {n}")
        ri = side.graph.inputs(n)
        tl = ", ".join(f"{p}:{k}{(' ' + t) if t else ''}{('[' + ','.join(f) + ']') if f else ''}" for p, k, t, f in ri.targets)
        print(f"{n}\t{len(ri.files)} files\t{tl}\tscripts: {' '.join(sorted(set(ri.commands.scripts))) or '-'}\truns: {' '.join(ri.commands.runs) or '-'}")
        for e in ri.errors:
            print(f"  error: {e}")
    return 0


def cmd_why(args: argparse.Namespace) -> int:
    side = make_side(ROOT)
    names = [n for n in sorted(side.recipes, key=lambda n: side.recipes[n].line) if args.all_recipes or n.startswith(GATE_PREFIX)]
    for f in args.files:
        f = _norm(f)
        hits = [(n, side.graph.inputs(n).match(f)) for n in names]
        hits = [(n, w) for n, w in hits if w]
        print(f"{f}: {len(hits)} recipes")
        for n, w in hits:
            print(f"  {n}  [{w}]")
        if not hits:
            hint = orphan_hint(side, f)
            if hint:
                print(f"  ({hint})")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    tree = Tree(ROOT)
    recipes = load_justfile(args.justfile or os.path.join(ROOT, "justfile"))
    problems = check(tree, recipes)
    for p in problems:
        print(f"check-recipes: {p}", file=sys.stderr)
    return 1 if problems else 0


# ----------------------------------------------------------------------------------------------
# the green ledger's input key (tools/gate-batch.sh --ledger)
# ----------------------------------------------------------------------------------------------
#
# An item's key is the sha256 of labelled parts, one per line, in a fixed order:
#   v         the key format
#   recipe    the text of the recipe and of every recipe it depends on (just's own parse: a comment
#             edit in the justfile does not move it), and the justfile's settings and assignments
#   scope     `closure` — the files below come from the walk `affected` uses (crate graph, module
#             trees, the scripts the recipe names and what they name; cfg-ignoring, so a superset) —
#             or `tree`, every file tools/box.sh ships, with the reason: a command run on the Mac, a
#             cargo subcommand the parser does not model (`cargo fmt`), a script that walks the tree
#   file      path relative to the tree, content sha256 (a link: its target)
#   global    Cargo.toml, Cargo.lock, rust-toolchain.toml, .cargo/*, the cuda-oxide pin rev
#   item      ARGS, the item's full BLOOMERY_BOX_ENV (the caller's, then the item's, then the lane's
#             card), the card that env forces
#   mac-env   the Mac-side variables tools/box.sh carries or reads (BLOOMERY_REMOTE is not one: it
#             names a track's directory, not an input), and the versions of just and python3
#   box       every line of the box manifest (`box-manifest`, fetched once per batch)
# Not in it: the binary (its paths are per track) and the commit.

KEY_VERSION = "bloomery-gate-key 1"
MANIFEST_VERSION = "box-manifest 1"
MAC_ENV = ("BLOOMERY_BOX", "BLOOMERY_CARD", "BLOOMERY_DATA", "BLOOMERY_MODEL", "BLOOMERY_REF_MODEL", "BLOOMERY_V41_MODEL")
KEY_GLOBALS = ["Cargo.toml", "Cargo.lock", "rust-toolchain.toml", ".cargo/config.toml", ".cargo/cuda-oxide.toml"]
# tools/box.sh's rsync excludes (target/ and .git/ at any depth; *.ptx, *.ll, .oxide-artifacts/ at the
# root), plus per-checkout noise no gate reads that would split the key by checkout: a worktree's .git
# file, .DS_Store, __pycache__/.
TREE_SKIP_DIRS = frozenset({"target", ".git", "__pycache__"})
TREE_SKIP_FILES = frozenset({".git", ".DS_Store"})
SCRIPT_EXTS = (".sh", ".py", ".bash")
# $BLOOMERY_DATA entries left out of the manifest: only the timing runners write them
# (tools/ref/nsys-gpu.sh, nsys-ds41.sh, ncu-gpu.sh), and no gate reads them — the self-test fails when
# a gate-* recipe's inputs name one.
DATA_UNREAD = {
    "nsys": "profiler reports of the timing runners (tools/ref/nsys-gpu.sh, nsys-ds41.sh); no gate reads them",
    "ncu": "profiler reports of a timing runner (tools/ref/ncu-gpu.sh); no gate reads them",
}
DATA_UNREAD_NAMED = re.compile(r"BLOOMERY_DATA\}?\"?/(?:nsys|ncu)\b|join\(\"(?:nsys|ncu)\"\)|\"(?:nsys|ncu)/")
# A cargo selector whose value is a recipe parameter (`--features {{FEATURES}}`, `--bin {{BIN}}`): just
# resolves it when the recipe runs, the closure never sees it (Graph.inputs skips it), so a ledger key
# cannot name the crates the build pulls in.
CARGO_PARAM = re.compile(r"(?:--features|--bin|--test|--example|--package|-p|-F)(?:=|\s+)\S*\{\{[^}]*\}\}")

# A reference tree (ik, llama.cpp, mistral.rs) is outside the manifest: a recipe that reads one is
# never skipped. The profiles and ref-paths.sh only define these names for every box command. The home
# path is spelled with a class so that this file, which check-recipes runs, does not match itself.
IK_READ = re.compile(r"\$\{?(?:IK|IKBIN|LCPP|LCPPBIN|MRS|MRSBIN)\b|/home/[u]ser/")
IK_DEFINERS = ("tools/ref/ref-paths.sh", "tools/ref/models/")
HOME_LITERAL = re.compile(r"\"/home/")
# A walk of the tree, which no closure bounds: the recipe is keyed on the whole tree.
TREE_WALK = re.compile(
    r"\bfind\s+\"?(?:\$\{?(?:ROOT|HERE)\}?\"?(?:/\S*)?|\.|crates|tools|docs)(?=[\s\"/]|$)"
    r"|\bgrep\s+(?:-[A-Za-z]*r[A-Za-z]*|--recursive)\b"
    r"|\bgit\s+(?:ls-files|grep)\b"
    r"|\bos\.walk\(|\bglob\.glob\(|\.rglob\("
    r"|(?:\$\{?(?:ROOT|HERE)\}?\"?/|(?<![\w./$-])(?:crates|tools|docs)/)[^\s'\"]*[*?{\[]"
)
ITEM_RE = re.compile(r"^([A-Za-z0-9_-]+)(?:@([^:]*))?(?::(.*))?$")
ANY_CARD = "BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any}"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
MODEL_LITERAL = re.compile(r"/models/[A-Za-z0-9._+-]+(?:/[A-Za-z0-9._+-]+)*")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def shipped_files(root: str) -> list[str]:
    """Every file (or link) tools/box.sh ships from `root`, relative and sorted, minus TREE_SKIP_*."""
    out: list[str] = []
    for dp, dns, fns in os.walk(root):
        rel = os.path.relpath(dp, root)
        keep = []
        for d in sorted(dns):
            if d in TREE_SKIP_DIRS or (rel == "." and d == ".oxide-artifacts"):
                continue
            if os.path.islink(os.path.join(dp, d)):
                out.append(os.path.normpath(os.path.join(rel, d)))  # shipped as a link, not walked
                continue
            keep.append(d)
        dns[:] = keep
        for f in fns:
            if f in TREE_SKIP_FILES or (rel == "." and f.endswith((".ptx", ".ll"))):
                continue
            out.append(os.path.normpath(os.path.join(rel, f)))
    return sorted(out)


def command_shape(recipe: Recipe) -> tuple[list[str], list[str]]:
    """(commands the recipe runs on the Mac other than tools/box.sh, box cargo subcommands the parser
    does not model). Either one means the recipe's reads are not bounded by its closure."""
    mac: list[str] = []
    unmodeled: list[str] = []
    for line in recipe.lines:
        stripped = line.lstrip("@-").strip()
        if not stripped or stripped.startswith("#"):
            continue
        for cmd in simple_commands(shell_words(stripped)):
            _, rest = strip_wrappers(cmd)
            if not rest:
                continue
            if _norm(rest[0]) != "tools/box.sh":
                mac.append(" ".join(rest[:2]))
                continue
            for rcmd in simple_commands(shell_words(" ".join(rest[1:]))):
                _, r = strip_wrappers(rcmd)
                if r and r[0] == "cargo":
                    try:
                        inv = parse_cargo(r[1:], " ".join(r))
                    except RecipeError:
                        inv = None
                    if inv is None:
                        unmodeled.append(" ".join(r[:2]))
    return mac, unmodeled


def exec_closure(tree: Tree, start: list[str]) -> list[str]:
    """The scripts a recipe runs: `start` (what its text runs or sources) and, transitively, the scripts
    a shell script among them names. A python script is a leaf: the paths it names are data it reads
    (recipes.py names every runner as a string), not scripts it runs."""
    seen: list[str] = []
    stack = [s for s in start if tree.exists(s)]
    while stack:
        s = stack.pop()
        if s in seen:
            continue
        seen.append(s)
        if not s.endswith((".sh", ".bash")):
            continue
        sdir = os.path.dirname(s)
        for line in tree.read(s).split("\n"):
            st = line.strip()
            if not st or st.startswith("#"):
                continue
            for q in _SCRIPT_PATH.findall(line):
                q = _norm(q)
                if q.endswith(SCRIPT_EXTS) and tree.exists(q):
                    stack.append(q)
            for b in re.findall(r"([A-Za-z0-9_.-]+\.(?:sh|py|bash))\b", line):
                q = os.path.normpath(os.path.join(sdir, b))
                if tree.exists(q):
                    stack.append(q)
    return sorted(seen)


def recipe_closure(recipes: dict[str, Recipe], name: str) -> list[str]:
    seen: list[str] = []
    stack = [name]
    while stack:
        n = stack.pop()
        if n in seen or n not in recipes:
            continue
        seen.append(n)
        stack.extend(recipes[n].deps)
    return sorted(seen)


def scan_lines(text: str, pat: re.Pattern, where: str, numbered: bool = True) -> str | None:
    """The first non-comment line of `text` that `pat` matches, as `where[:line] (match)`."""
    if not pat.search(text):
        return None
    for i, line in enumerate(text.split("\n"), 1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        m = pat.search(line)
        if m:
            return f"{where}:{i} ({m.group(0).strip()})" if numbered else f"{where} ({m.group(0).strip()})"
    return None


_SCAN_MEMO: dict[tuple, str | None] = {}
_JUST_VERSION: list[str] = []


def scan_file(root: str, rel: str, pat: re.Pattern) -> str | None:
    """scan_lines over a tree file, memoized on its stat (the self-test keys many copies of one tree)."""
    p = os.path.join(root, rel)
    st = os.stat(p)
    k = (p, st.st_size, st.st_mtime_ns, pat.pattern)
    if k not in _SCAN_MEMO:
        with open(p, encoding="utf-8", errors="replace") as fh:
            _SCAN_MEMO[k] = scan_lines(fh.read(), pat, rel)
    return _SCAN_MEMO[k]


def just_version() -> str:
    if not _JUST_VERSION:
        _JUST_VERSION.append(subprocess.run(["just", "--version"], capture_output=True, text=True).stdout.strip() or "unknown")
    return _JUST_VERSION[0]


def oxide_rev(text: str) -> str:
    """The cuda-oxide rev tools/box.sh reads: the [patch] section's cuda-device rev, else the first one."""
    sec = re.search(r'^\[patch\."https://github\.com/NVlabs/cuda-oxide\.git"\]\s*$(.*?)(?=^\[workspace|\Z)', text, re.M | re.S)
    for body in ([sec.group(1)] if sec else []) + [text]:
        m = re.search(r'^cuda-device = .*rev = "([0-9a-f]+)"', body, re.M)
        if m:
            return m.group(1)
    return "absent"


def load_manifest(path: str | None) -> tuple[list[str], dict[str, str], str | None]:
    """(content lines, header fields, error) of a box manifest file."""
    if not path:
        return [], {}, "no box manifest was given (--manifest)"
    try:
        with open(path, encoding="utf-8") as fh:
            lines = [ln for ln in fh.read().split("\n") if ln.strip()]
    except OSError as err:
        return [], {}, f"cannot read {path}: {err.strerror}"
    if not lines or not lines[0].startswith("# " + MANIFEST_VERSION):
        return [], {}, f"{path} is not a {MANIFEST_VERSION} file (first line {(lines[:1] or [''])[0][:60]!r})"
    if lines[-1] != "# end":
        return [], {}, f"{path} has no `# end` line: the box manifest is truncated"
    head = dict(kv.split("=", 1) for kv in lines[0].split()[3:] if "=" in kv)
    return [ln for ln in lines if not ln.startswith("#")], head, None


def part_id(line: str) -> tuple[str, ...]:
    """A part's identity for a diff: label and name (label, kind and name for a manifest line)."""
    f = line.split("\t")
    return tuple(f[:3]) if f[0] == "box" else tuple(f[:2])


class KeyContext:
    """Keys for the items of one batch: one tree, one box manifest, one caller environment."""

    def __init__(self, side: Side, manifest_path: str | None, box_env: str = "", environ: dict[str, str] | None = None, manifest_error: str | None = None, settings: dict | None = None):
        self.side = side
        self.box_env = box_env
        env = dict(os.environ) if environ is None else environ
        self.env = env
        self.mac = [f"mac-env\t{v}\t{env[v] if v in env else '<unset>'}" for v in MAC_ENV]
        self.mac += [f"mac-tool\tjust\t{just_version()}", f"mac-tool\tpython3\t{platform.python_version()}"]
        self.manifest, self.manifest_head, self.manifest_error = load_manifest(manifest_path)
        if manifest_error:
            self.manifest_error = f"the box manifest failed: {manifest_error}"
        self.settings = settings if settings is not None else justfile_settings(os.path.join(side.tree.root, "justfile"))
        self._sha: dict[str, str] = {}
        self._tree: list[str] | None = None

    def tree_files(self) -> list[str]:
        if self._tree is None:
            self._tree = shipped_files(self.side.tree.root)
        return self._tree

    def file_part(self, rel: str) -> str:
        p = os.path.join(self.side.tree.root, rel)
        if os.path.islink(p):
            return f"link\t{rel}\t{os.readlink(p)}"
        if rel not in self._sha:
            self._sha[rel] = sha256_file(p)
        return f"file\t{rel}\t{self._sha[rel]}"

    def globals(self) -> list[str]:
        root = self.side.tree.root
        names = list(KEY_GLOBALS)
        cdir = os.path.join(root, ".cargo")
        if os.path.isdir(cdir):
            names += sorted(f".cargo/{f}" for f in os.listdir(cdir) if f".cargo/{f}" not in names)
        out = []
        for g in names:
            p = os.path.join(root, g)
            out.append(f"global\t{g}\t{sha256_file(p) if os.path.isfile(p) else 'absent'}")
        toml = os.path.join(root, "Cargo.toml")
        out.append(f"global\toxide-rev\t{oxide_rev(self.side.tree.read('Cargo.toml')) if os.path.isfile(toml) else 'absent'}")
        return out

    def scope(self, names: list[str], ri: RecipeInputs) -> tuple[str, str]:
        st = self.settings.get("settings", {})
        if st.get("dotenv_load") or st.get("dotenv_filename") or st.get("dotenv_path"):
            return "tree", "the justfile loads a dotenv file"
        recipes = self.side.recipes
        for n in names:
            mac, unmodeled = command_shape(recipes[n])
            if mac:
                return "tree", f"{n} runs `{mac[0]}` on the Mac; its reads are not modeled"
            if unmodeled:
                return "tree", f"{n} runs `{unmodeled[0]}`, a cargo subcommand the parser does not model"
            hit = scan_lines("\n".join(recipes[n].lines), TREE_WALK, f"justfile:{recipes[n].line} {n}", numbered=False)
            if hit:
                return "tree", f"{hit} walks the tree"
        for f in self.run_scripts(names):
            hit = scan_file(self.side.tree.root, f, TREE_WALK)
            if hit:
                return "tree", f"{hit} walks the tree"
        return "closure", f"{len(ri.files)} files, {len(ri.prefixes)} directories"

    def run_scripts(self, names: list[str]) -> list[str]:
        """The scripts the recipes `names` run on the box or the Mac (exec_closure), box.sh's own
        sourced files included for a box recipe."""
        start: list[str] = []
        for n in names:
            rc = self.side.graph.inputs(n).commands
            start += rc.scripts
            if rc.box:
                start += BOX_GLOBALS + [f"tools/ref/models/{rc.env.get('BLOOMERY_MODEL', DEFAULT_PROFILE)}.sh"]
        return exec_closure(self.side.tree, start)

    def never(self, names: list[str], ri: RecipeInputs, envs: list[str]) -> str | None:
        recipes = self.side.recipes
        for n in names:
            for ln in recipes[n].lines:
                hit = CARGO_PARAM.search(ln)
                if hit:
                    return f"justfile:{recipes[n].line} {n}: `{hit.group(0)}` takes its value from a recipe parameter, so the key cannot see the crates the build pulls in"
        for n in names:
            if "tools/gate-batch.sh" in self.side.graph.inputs(n).commands.scripts:
                return f"{n} runs tools/gate-batch.sh: a batch inside a batch, whose items' inputs are not this item's"
        for n in names:
            hit = scan_lines("\n".join(recipes[n].lines), IK_READ, f"justfile:{recipes[n].line} {n}", numbered=False)
            if hit:
                return f"{hit} reads a reference tree, which is outside the key"
        for f in self.run_scripts(names):
            if f.startswith(IK_DEFINERS):
                continue
            hit = scan_file(self.side.tree.root, f, IK_READ)
            if hit:
                return f"{hit} reads a reference tree, which is outside the key"
        for f in sorted(ri.files):
            if f.endswith(".rs") and self.side.tree.exists(f):
                hit = scan_file(self.side.tree.root, f, HOME_LITERAL)
                if hit:
                    return f"{hit} reads a reference tree, which is outside the key"
        # tools/gpu-gate.sh's `any` picks a card at run time (the idle A6000, else the 3090): with no card
        # forced (--lanes 1) the key cannot name the card a green ran on.
        forced = any(e.partition("=")[0] == "BLOOMERY_GATE_CARD" for e in self.box_env.split() + envs)
        if not forced:
            for n in names:
                if any(ANY_CARD in ln for ln in recipes[n].lines):
                    return f"{n} calls tools/gpu-gate.sh with the `any` card and no card is forced (--lanes 1): the card is picked at run time"
        # A path in an env entry is read by the gate but not by the box manifest: the manifest walks
        # $BLOOMERY_DATA and the model directories as the caller's env names them, never an item's.
        for e in envs:
            k, _, v = e.partition("=")
            if v.startswith("/"):
                return f"the item's env {k}={v} names a path the box manifest does not read"
        data = self.manifest_head.get("data", "")
        for e in self.box_env.split():
            k, _, v = e.partition("=")
            if v.startswith("/") and not v.startswith("/models/") and not (data and (v == data or v.startswith(data.rstrip("/") + "/"))):
                return f"BLOOMERY_BOX_ENV's {k}={v} names a path the box manifest does not read"
        return None

    def parts(self, item: str) -> tuple[list[str], str | None, str | None, str]:
        """(parts, error, never-skip reason, scope summary) of one item."""
        m = ITEM_RE.match(item)
        if not m:
            return [], f"malformed item {item!r} (NAME[@K=V,…][:ARGS])", None, ""
        name, env, args = m.groups()
        recipes, graph, tree = self.side.recipes, self.side.graph, self.side.tree
        if name not in recipes:
            return [], f"{name!r} is not a recipe in the justfile", None, ""
        envs = [e for e in (env or "").split(",") if e]
        bad = [e for e in envs if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", e)]
        if bad:
            return [], f"env entry {bad[0]!r} is not K=V", None, ""
        try:
            argv = shlex.split(args) if args else []
        except ValueError as err:
            return [], f"ARGS {args!r}: {err}", None, ""
        if self.manifest_error:
            return [], self.manifest_error, None, ""
        try:
            ri = graph.inputs(name)
        except (OSError, RecipeError) as err:
            return [], f"{name}: its inputs cannot be read: {err}", None, ""
        if ri.errors:
            return [], f"{name}: {ri.errors[0]}", None, ""
        names = recipe_closure(recipes, name)
        try:
            return self._parts(name, names, ri, envs, argv)
        except OSError as err:
            return [], f"{name}: an input cannot be read: {err}", None, ""

    def _parts(self, name: str, names: list[str], ri: RecipeInputs, envs: list[str], argv: list[str]) -> tuple[list[str], str | None, str | None, str]:
        recipes, tree = self.side.recipes, self.side.tree
        scope, why = self.scope(names, ri)
        never = self.never(names, ri, envs)
        eff = self.box_env.split() + envs
        # A path in ARGS is read by the gate: an absolute one is outside the manifest (never-skip), a tree
        # path joins the key's files. The same for a tree path in an env value.
        named: set[str] = set()
        for w in argv + [e.partition("=")[2] for e in eff]:
            for v in w.split(","):
                v = v.split("=", 1)[-1]
                if v.startswith("/") and w in argv and never is None:
                    never = f"ARGS {v} names a path the box manifest does not read"
                elif "/" in v and not v.startswith("/"):
                    rel = _norm(v)
                    if tree.exists(rel):
                        named.add(rel)
                    elif tree.isdir(rel):
                        named |= {f for f in self.tree_files() if f.startswith(rel.rstrip("/") + "/")}
        parts = [f"v\t{KEY_VERSION}"]
        parts += [f"recipe\t{n}\t{sha256_text(recipes[n].text)}" for n in names]
        parts.append(f"recipe\t<justfile>\t{sha256_text(json.dumps(self.settings, sort_keys=True))}")
        parts.append(f"scope\t{scope}\t{why if scope == 'tree' else 'closure'}")
        if scope == "tree":
            files = set(self.tree_files())
        else:
            files = set(ri.files)
            for p in ri.prefixes:
                files |= {f for f in self.tree_files() if f.startswith(p)}
        files |= named
        files -= set(KEY_GLOBALS)
        if scope == "closure":
            for u in tree.unresolved:
                if u.split(": ", 1)[0] in files:
                    return [], f"source walk: {u}", None, ""
        for rel in sorted(files):
            try:
                parts.append(self.file_part(rel))
            except OSError as err:
                return [], f"cannot read {rel}: {err.strerror}", None, ""
        parts += self.globals()
        card = "none"
        for e in eff:
            k, _, v = e.partition("=")
            if k == "BLOOMERY_GATE_CARD":
                card = v
        parts += [f"item\targs\t{json.dumps(argv)}", f"item\tboxenv\t{' '.join(eff)}", f"item\tcard\t{card}"]
        parts += self.mac
        parts += [f"box\t{ln}" for ln in self.manifest]
        summary = f"scope=tree ({why})" if scope == "tree" else f"scope=closure ({why})"
        return parts, None, never, summary


def justfile_settings(path: str) -> dict:
    """The justfile's settings, assignments and unexports (just's own parse), for the key."""
    proc = subprocess.run(["just", "--justfile", path, "--working-directory", os.path.dirname(path) or ".", "--dump", "--dump-format", "json"], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RecipeError(f"`just --dump` failed on {path} (exit {proc.returncode}): {proc.stderr.strip()}")
    dump = json.loads(proc.stdout)
    return {k: dump.get(k) for k in ("settings", "assignments", "unexports")}


def item_key(parts: list[str]) -> str:
    return sha256_text("\n".join(parts) + "\n")


@dataclass
class LedgerRecord:
    key: str
    recipe: str
    item: str
    commit: str
    date: str
    tree: str


class Ledger:
    """The green ledger, read-only here: `key recipe item commit date tree`, appended by gate-batch.sh."""

    def __init__(self, path: str):
        self.path = path
        self.by_key: dict[str, LedgerRecord] = {}
        self.by_item: dict[tuple[str, str], LedgerRecord] = {}
        self.by_recipe: dict[str, LedgerRecord] = {}
        self.bad = 0
        if not os.path.exists(path):
            return
        with open(path, encoding="utf-8", errors="replace") as fh:
            for ln in fh:
                f = ln.rstrip("\n").split("\t")
                if len(f) != 6 or not HEX64.match(f[0]):
                    self.bad += 1
                    continue
                rec = LedgerRecord(*f)
                self.by_key[rec.key] = rec
                self.by_item[(rec.recipe, rec.item)] = rec
                self.by_recipe[rec.recipe] = rec

    def parts_of(self, key: str) -> list[str] | None:
        p = os.path.join(self.path + ".parts", key + ".parts")
        try:
            with open(p, encoding="utf-8") as fh:
                return [ln for ln in fh.read().split("\n") if ln]
        except OSError:
            return None

    def why(self, name: str, item: str, parts: list[str]) -> str:
        """Why an item that has no green record at its key runs: what moved since its last green."""
        rec = self.by_item.get((name, item))
        which = "this item"
        if rec is None:
            rec = self.by_recipe.get(name)
            which = f"{name} as {rec.item}" if rec else ""
        if rec is None:
            return f"no green record of {name}"
        head = f"no green record at this key; {which} was green at {rec.commit} {rec.date}"
        old = self.parts_of(rec.key)
        if old is None:
            return head + " (its parts were not kept)"
        cur = {part_id(p): p for p in parts}
        was = {part_id(p): p for p in old}
        moved = [k for k in cur if k in was and was[k] != cur[k]] + [k for k in cur if k not in was] + [k for k in was if k not in cur]
        shown = "; ".join(" ".join(k) for k in moved[:4])
        return head + f"; moved since: {shown}" + (f" (+{len(moved) - 4})" if len(moved) > 4 else "")


def cmd_key(args: argparse.Namespace) -> int:
    side = make_side(ROOT)
    ctx = KeyContext(side, args.manifest, args.box_env or "", manifest_error=args.manifest_error)
    ledger = Ledger(args.ledger) if args.ledger else None
    if ledger is not None and ledger.bad:
        print(f"recipes.py key: {args.ledger}: {ledger.bad} malformed line(s) ignored (they cannot skip an item)", file=sys.stderr)
    if args.parts_dir:
        os.makedirs(args.parts_dir, exist_ok=True)
    for i, item in enumerate(args.items):
        parts, err, never, summary = ctx.parts(item)
        if err is not None:
            print(f"-\t{item}\terror\t{err}".replace("\n", " "))
            continue
        key = item_key(parts)
        if args.parts_dir:
            with open(os.path.join(args.parts_dir, f"{i}.parts"), "w", encoding="utf-8") as fh:
                fh.write("\n".join(parts) + "\n")
        name = ITEM_RE.match(item).group(1)
        if never is not None:
            status, detail = "never", never
        elif ledger is None:
            status, detail = "key", summary
        elif args.rerun:
            status, detail = "rerun", "--rerun: runs whatever the ledger holds"
        elif key in ledger.by_key:
            r = ledger.by_key[key]
            status, detail = "skip", f"green-at={r.commit} {r.date} tree={r.tree}"
        else:
            status, detail = "run", ledger.why(name, item, parts)
        print(f"{key}\t{item}\t{status}\t{detail}".replace("\n", " "))
        if args.show_parts:
            for p in parts:
                print(f"  {p}")
    return 0


# ---------------- the box manifest (runs on the box, through tools/box.sh) ----------------


class HashCache:
    """sha256 of files keyed by (size, mtime, ctime, inode, device), kept between batches on the box:
    a file rewritten with the same bytes (gate-1-1 and gate-tokenizer rewrite theirs on every run) keeps
    its hash, and only files whose stat moved are read again. A file changed within the last 2 s is
    hashed but not cached (its next write may land inside the same timestamp)."""

    RACY_NS = 2_000_000_000

    def __init__(self, path: str):
        self.path = path
        self.entries: dict[str, tuple] = {}
        self.dirty = False
        self.hashed = 0
        self.hashed_bytes = 0
        self.t0_ns = time.time_ns()
        self._fd = -1

    def __enter__(self) -> HashCache:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._fd = os.open(self.path + ".lock", os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(self._fd, fcntl.LOCK_EX)
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8", errors="replace") as fh:
                for ln in fh:
                    f = ln.rstrip("\n").split("\t")
                    if len(f) == 7 and HEX64.match(f[0]):
                        self.entries[f[6]] = (f[0], int(f[1]), int(f[2]), int(f[3]), int(f[4]), int(f[5]))
        return self

    def __exit__(self, et, ev, tb) -> None:
        try:
            if et is None and self.dirty:
                tmp = f"{self.path}.tmp.{os.getpid()}"
                with open(tmp, "w", encoding="utf-8") as fh:
                    for p, e in sorted(self.entries.items()):
                        if "\t" not in p and "\n" not in p:
                            fh.write("\t".join(map(str, e)) + "\t" + p + "\n")
                os.replace(tmp, self.path)
        finally:
            os.close(self._fd)

    @staticmethod
    def ident(st: os.stat_result) -> tuple:
        return (st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino, st.st_dev)

    def cached(self, path: str, st: os.stat_result) -> str | None:
        e = self.entries.get(path)
        return e[0] if e is not None and e[1:] == self.ident(st) else None

    def store(self, path: str, st: os.stat_result, digest: str) -> None:
        self.dirty = True
        self.hashed += 1
        self.hashed_bytes += st.st_size
        if max(st.st_mtime_ns, st.st_ctime_ns) > self.t0_ns - self.RACY_NS:
            self.entries.pop(path, None)
        else:
            self.entries[path] = (digest, *self.ident(st))

    def forget_under(self, root: str, seen: set[str]) -> None:
        pre = root.rstrip("/") + "/"
        for p in [p for p in self.entries if p.startswith(pre) and p not in seen]:
            del self.entries[p]
            self.dirty = True

    def sha(self, path: str) -> str:
        st = os.stat(path)
        got = self.cached(path, st)
        if got is None:
            got, st = hash_stable(path, st)
            self.store(path, st, got)
        return got


def hash_stable(path: str, st: os.stat_result) -> tuple[str, os.stat_result]:
    """sha256 of a file whose stat did not move while it was read (retried twice), and that stat."""
    for _ in range(3):
        h = hashlib.sha256()
        with open(path, "rb", buffering=0) as fh:
            try:
                os.posix_fadvise(fh.fileno(), 0, 0, os.POSIX_FADV_NOREUSE)
            except (AttributeError, OSError):
                pass
            buf = bytearray(8 << 20)
            view = memoryview(buf)
            while True:
                n = fh.readinto(buf)
                if not n:
                    break
                h.update(view[:n])
            after = os.fstat(fh.fileno())
        if HashCache.ident(after) == HashCache.ident(st):
            return h.hexdigest(), st
        st = os.stat(path)
    raise RecipeError(f"box-manifest: {path} kept changing while it was hashed — something writes $BLOOMERY_DATA now; retry")


def _run(cmd: list[str], **kw) -> str:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, **kw)
    except (OSError, subprocess.TimeoutExpired) as err:
        raise RecipeError(f"box-manifest: {cmd[0]} could not run: {err}") from err
    if proc.returncode != 0:
        raise RecipeError(f"box-manifest: {' '.join(cmd[:3])} failed (exit {proc.returncode}): {proc.stderr.strip()[-300:]}")
    return proc.stdout


def data_manifest(root: str, cache: HashCache, workers: int) -> tuple[list[tuple], int]:
    """One line per top-level entry of $BLOOMERY_DATA: a digest over (path, size, content sha256) of every
    file under it — a link by its target text, and its target's content when that is a file. The
    DATA_UNREAD entries are named, not read."""
    lines: list[tuple] = []
    rows: dict[str, list[tuple[str, str]]] = {}  # top -> [(sort key, digest line)]
    files: dict[str, int] = {}
    size: dict[str, int] = {}
    todo: list[tuple[str, str, str, os.stat_result]] = []  # (top, rel, abspath, stat) not in the cache
    seen: set[str] = set()

    def add_file(top: str, rel: str, st: os.stat_result, digest: str) -> None:
        rows[top].append((rel, f"{rel}\t{st.st_size}\t{digest}"))
        files[top] += 1
        size[top] += st.st_size

    for top in sorted(os.listdir(root)):
        if top in DATA_UNREAD:
            lines.append(("data-skip", top + "/", DATA_UNREAD[top]))
            continue
        tp = os.path.join(root, top)
        rows[top], files[top], size[top] = [], 0, 0
        is_dir = os.path.isdir(tp) and not os.path.islink(tp)
        walk = os.walk(tp) if is_dir else [(root, [], [top])]
        for dp, dns, fns in walk:
            dns.sort()
            for d in [d for d in dns if os.path.islink(os.path.join(dp, d))]:
                rel = os.path.relpath(os.path.join(dp, d), root)
                rows[top].append((rel + "\0", f"{rel}\t->\t{os.readlink(os.path.join(dp, d))}"))
            for f in sorted(fns):
                ap = os.path.join(dp, f)
                rel = os.path.relpath(ap, root)
                lst = os.lstat(ap)
                st = lst
                if stat.S_ISLNK(lst.st_mode):
                    rows[top].append((rel + "\0", f"{rel}\t->\t{os.readlink(ap)}"))
                    try:
                        st = os.stat(ap)
                    except FileNotFoundError:
                        continue  # a dangling link: its target text is the record
                if not stat.S_ISREG(st.st_mode):
                    if not stat.S_ISLNK(lst.st_mode):
                        rows[top].append((rel, f"{rel}\t{stat.filemode(lst.st_mode)}"))
                    continue
                seen.add(ap)
                got = cache.cached(ap, st)
                if got is None:
                    todo.append((top, rel, ap, st))
                else:
                    add_file(top, rel, st, got)
    if todo:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            for (top, rel, ap, _), (digest, st) in zip(todo, ex.map(lambda t: hash_stable(t[2], t[3]), todo)):
                cache.store(ap, st, digest)
                add_file(top, rel, st, digest)
    cache.forget_under(root, seen)
    for top in rows:
        rows[top].sort()
        tp = os.path.join(root, top)
        name = top + ("/" if os.path.isdir(tp) and not os.path.islink(tp) else "")
        lines.append(("data", name, sha256_text("\n".join(ln for _, ln in rows[top]) + "\n"), files[top], size[top]))
    lines.sort(key=lambda t: (t[0], t[1]))
    return lines, sum(files.values())


PROFILE_PATHS = 'unset BLOOMERY_REF_MODEL; source "$1" > /dev/null || exit 3; for v in $(compgen -v); do case "${!v}" in /models/*) printf "%s\\n" "${!v}" ;; esac; done'


def model_dirs() -> set[str]:
    """The model directories a gate can open: every /models path the profiles define (under their own
    defaults), the box command's environment names, or the tree names literally — the directory of each,
    so a split set's shards all count."""
    paths: set[str] = set()
    env = {k: v for k, v in os.environ.items() if k != "BLOOMERY_REF_MODEL"}
    for prof in sorted(glob.glob(os.path.join(ROOT, "tools/ref/models/*.sh"))):
        out = _run(["bash", "-c", PROFILE_PATHS, "bash", prof], env=env)
        paths |= {ln for ln in out.splitlines() if ln.startswith("/models/")}
    paths |= {v for v in os.environ.values() if v.startswith("/models/")}
    for rel in shipped_files(ROOT):
        if rel == "justfile" or rel.endswith((".rs", ".sh", ".py", ".toml", ".tsv")):
            with open(os.path.join(ROOT, rel), encoding="utf-8", errors="replace") as fh:
                paths |= set(MODEL_LITERAL.findall(fh.read()))
    # A model directory is /models/<name>: never /models itself, whose listing moves whenever any
    # model is fetched (a literal naming a file right under /models is its own entry).
    return {"/".join(p.split("/")[:3]) for p in paths if p.count("/") >= 2}


def cmd_box_manifest(args: argparse.Namespace) -> int:
    t0 = time.time()
    for lock in args.lease or []:
        r = subprocess.run(["flock", "-n", "-E", "75", lock, "true"])
        if r.returncode == 75:
            print(f"box-manifest: the timing lease {lock} is held — a sitting must not see the manifest's reads", file=sys.stderr)
            return 75
        if r.returncode != 0:
            raise RecipeError(f"box-manifest: cannot test the lease {lock} (flock exit {r.returncode})")
    data = os.environ.get("BLOOMERY_DATA", "")
    if not data or not os.path.isdir(data):
        raise RecipeError(f"box-manifest: BLOOMERY_DATA={data!r} is not a directory (run it through tools/box.sh, which exports it)")
    rows: list[tuple] = []
    with HashCache(os.path.expanduser(args.cache)) as cache:
        env_file = os.path.expanduser("~/bloomery-env.sh")
        rows.append(("env", "bloomery-env.sh", sha256_file(env_file) if os.path.isfile(env_file) else "absent"))
        rows.append(("os", "kernel", platform.release()))
        for ln in _run(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,vbios_version", "--format=csv,noheader"]).splitlines():
            if ln.strip():
                rows.append(("gpu", *[x.strip() for x in ln.split(",")]))
        rows.append(("tool", "rustc", "; ".join(_run(["rustc", "-vV"], cwd=ROOT).split("\n")).strip("; ")))
        llc = os.environ.get("CUDA_OXIDE_LLC", "")
        rows.append(("tool", "llc", "; ".join(ln.strip() for ln in _run([llc, "--version"]).splitlines() if "version" in ln.lower()) if llc else "unset"))
        ptxas = os.path.join(os.environ.get("CUDA_TOOLKIT_PATH", "/usr/local/cuda"), "bin", "ptxas")
        rows.append(("tool", "ptxas", "; ".join(_run([ptxas, "--version"]).strip().splitlines()[-2:])))
        co = shutil.which("cargo-oxide")
        rows.append(("tool", "cargo-oxide", cache.sha(co) if co else "absent"))
        be = os.environ.get("CUDA_OXIDE_BACKEND", "")
        if be and os.path.isfile(be):
            srev = os.path.join(os.path.dirname(be), "source-rev.txt")
            rev_text = "absent"
            if os.path.isfile(srev):
                with open(srev, encoding="utf-8", errors="replace") as fh:
                    rev_text = fh.read().strip() or "empty"
            rows.append(("backend", os.path.basename(os.path.dirname(be)), cache.sha(be), rev_text))
        else:
            rows.append(("backend", "absent", be or "unset"))
        drows, nfiles = data_manifest(data, cache, args.workers)
        rows += drows
        for d in sorted(model_dirs()):
            if os.path.isfile(d):
                st = os.stat(d)
                rows.append(("model", d, st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino))
                continue
            if not os.path.isdir(d):
                rows.append(("model-dir", d, "absent"))
                continue
            for f in sorted(os.listdir(d)):
                p = os.path.join(d, f)
                lst = os.lstat(p)
                if stat.S_ISDIR(lst.st_mode):
                    rows.append(("model", p, "dir"))
                    continue
                link = os.readlink(p) if stat.S_ISLNK(lst.st_mode) else ""
                try:
                    st = os.stat(p)
                except FileNotFoundError:
                    rows.append(("model", p, "->", link, "dangling"))
                    continue
                rows.append(("model", p, st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino) + (("->", link) if link else ()))
        took = time.time() - t0
        head = (
            f"# {MANIFEST_VERSION} host={platform.node()} remote={ROOT} data={data} took={took:.2f}s "
            f"data_files={nfiles} hashed={cache.hashed} hashed_bytes={cache.hashed_bytes} cache={cache.path}"
        )
    print(head)
    for r in rows:
        print("\t".join(str(x).replace("\t", " ").replace("\n", " ") for x in r))
    print("# end")
    return 0


# ----------------------------------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------------------------------


def side_at(root: str, meta: dict) -> Side:
    tree = Tree(root, meta)
    recipes = load_justfile(os.path.join(tree.root, "justfile"))
    return Side(tree, recipes, Graph(tree, recipes))


def key_self_test(expect, real: Side) -> None:
    """The ledger key: the detectors on the real tree, then a scratch copy where each input class is
    changed in turn — the same inputs give the same key, each change moves exactly the keys that read it."""
    meta = cargo_metadata(ROOT)
    settings = justfile_settings(os.path.join(ROOT, "justfile"))
    ctx = KeyContext(real, None, settings=settings)
    gates = [n for n in real.recipes if n.startswith(GATE_PREFIX)]
    never, tree_scoped = set(), set()
    for n in gates:
        ri = real.graph.inputs(n)
        names = recipe_closure(real.recipes, n)
        if ctx.never(names, ri, ["BLOOMERY_GATE_CARD=3090"]):  # a lane's card forced: the card rule is not asked here
            never.add(n)
        if ctx.scope(names, ri)[0] == "tree":
            tree_scoped.add(n)
        for f in sorted(ri.files):
            if real.tree.exists(f) and not f.startswith("tools/ref/models/"):
                hit = scan_file(real.tree.root, f, DATA_UNREAD_NAMED)
                expect(hit is None, f"{n} reads {hit}: an entry the box manifest leaves out (DATA_UNREAD) — take it out of DATA_UNREAD")
    expect(never == {"gate-1-1", "gate-tokenizer"}, f"never-skip gate recipes: {sorted(never)}")
    expect(
        tree_scoped <= never,
        f"skippable gate-* recipes keyed on the whole tree: {sorted(tree_scoped - never)} — a script in the closure walks the tree",
    )
    for n in ("check", "fmt-check", "check-recipes", "check-comments", "check-arch", "check-rustflags"):
        expect(ctx.scope(recipe_closure(real.recipes, n), real.graph.inputs(n))[0] == "tree", f"{n} is not keyed on the whole tree")
    expect(ctx.scope(["lint"], real.graph.inputs("lint"))[0] == "closure", "lint (clippy, modeled) is keyed on the whole tree")
    for s in ("tools/check-comments.sh", "tools/check-arch.sh", "crates/tokenizer/tools/oracle.sh"):
        expect(scan_lines(real.tree.read(s), TREE_WALK, s) is not None, f"the walk detector misses {s}")
    quiet = ["tools/box.sh", "tools/gate.sh", "tools/gpu-gate.sh", "tools/host-gate.sh", "tools/ref/ref-paths.sh", "tools/ptx-spill-check.sh", "tools/ptx-scan.sh"]
    quiet += sorted(os.path.relpath(p, ROOT) for p in glob.glob(os.path.join(ROOT, "tools/ref/models/*.sh")))
    for s in quiet:
        hit = scan_lines(real.tree.read(s), TREE_WALK, s)
        expect(hit is None, f"the walk detector flags {hit}: every gate that runs it would be keyed on the whole tree")

    with tempfile.TemporaryDirectory(prefix="recipes-keytest-") as tmp:
        roots = []
        for sub in ("a", "b"):
            r = os.path.join(tmp, sub)
            for rel in shipped_files(ROOT):
                src, dst = os.path.join(ROOT, rel), os.path.join(r, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if os.path.islink(src):
                    os.symlink(os.readlink(src), dst)
                else:
                    shutil.copyfile(src, dst)
            roots.append(r)
        mf = os.path.join(tmp, "manifest.txt")
        manifest = [
            f"# {MANIFEST_VERSION} host=selftest data=/root/bloomery-data",
            "env\tbloomery-env.sh\t" + "a" * 64,
            "os\tkernel\t6.8.0",
            "gpu\t0\tNVIDIA GeForce RTX 3090\tGPU-0\t615.71.09\t94.02",
            "tool\trustc\trustc 1.95.0-nightly",
            "backend\tb9847e95\t" + "b" * 64 + "\tb9847e95",
            "data\tref/\t" + "c" * 64 + "\t10\t100",
            "model\t/models/small/x.gguf\t1\t2\t3\t4",
            "# end",
        ]

        def write_manifest(lines: list[str]) -> None:
            with open(mf, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")

        write_manifest(manifest)
        items = {
            "gpu": "gate-gpu-q4k-sel@BLOOMERY_GATE_CARD=3090",
            "cpu": "gate-sampler",
            "mac": "check-comments",
            "fmt": "fmt-check",
            "args": "gate-gpu-e2e@BLOOMERY_GATE_CARD=a6000:--x",
        }
        sa, sb = side_at(roots[0], meta), side_at(roots[1], meta)

        def keys(side: Side, box_env: str = "", environ: dict | None = None, manifest_path: str | None = mf, only: dict | None = None, fresh: bool = False) -> dict[str, str]:
            st = justfile_settings(os.path.join(side.tree.root, "justfile")) if fresh else settings
            c = KeyContext(side, manifest_path, box_env, {} if environ is None else environ, settings=st)
            out = {}
            for k, it in (only or items).items():
                parts, err, _, _ = c.parts(it)
                out[k] = item_key(parts) if err is None else "error: " + err
            return out

        base = keys(sa)
        expect(not any(v.startswith("error") for v in base.values()), f"base keys: {base}")
        expect(keys(sa) == base, "the same inputs gave another key")
        expect(keys(sb) == base, "a second checkout at another path gives other keys (a path is not relative)")

        def moved(after: dict[str, str]) -> set[str]:
            return {k for k in after if after[k] != base[k]}

        def edit(rel: str, fn, side_fresh: bool = False) -> set[str]:
            p = os.path.join(roots[0], rel)
            with open(p, encoding="utf-8") as fh:
                old = fh.read()
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(fn(old))
            try:
                return moved(keys(side_at(roots[0], meta) if side_fresh else sa, fresh=side_fresh))
            finally:
                with open(p, "w", encoding="utf-8") as fh:
                    fh.write(old)

        def append(text: str):
            return lambda s: s + text

        m = edit("crates/gpu-gates/src/bin/gate_q4k_sel.rs", append("\n// key probe\n"))
        expect(m == {"gpu", "mac", "fmt"}, f"gate_q4k_sel.rs moves {sorted(m)}, not gpu mac fmt")
        m = edit("tools/gpu-gate.sh", append("\n# key probe\n"))
        expect(m == {"gpu", "args", "mac", "fmt"}, f"tools/gpu-gate.sh moves {sorted(m)}")
        m = edit("docs/plan.md", append("\nkey probe\n"))
        expect(m == {"mac", "fmt"}, f"docs/plan.md moves {sorted(m)}, not only the whole-tree keys")
        cpu_lib = sorted(f for f in sa.graph.inputs("gate-sampler").files if f.startswith("crates/sampler/src/"))
        m = edit(cpu_lib[0], append("\n// key probe\n"))
        expect({"cpu", "mac", "fmt"} <= m and "gpu" not in m, f"{cpu_lib[0]} moves {sorted(m)}")
        for g in KEY_GLOBALS:
            m = edit(g, append("\n# key probe\n"))
            expect(m == set(items), f"{g} moves {sorted(m)}, not every key")
        two = {k: items[k] for k in ("gpu", "mac")}
        m = edit("justfile", append("\n# key probe\n"), side_fresh=True)
        expect(m == {"mac", "fmt"}, f"a justfile comment moves {sorted(m)}")
        m = edit("justfile", lambda s: s.replace("bash tools/gpu-gate.sh gate_q4k_sel'", "bash tools/gpu-gate.sh gate_q4k_sel '", 1), side_fresh=True)
        expect("gpu" in m and "cpu" not in m and "args" not in m, f"gate-gpu-q4k-sel's recipe text moves {sorted(m)}")

        one = lambda it: keys(sa, only={"x": it})["x"]  # noqa: E731
        expect(one("gate-gpu-q4k-sel@BLOOMERY_GATE_CARD=3090,X=1") != base["gpu"], "an item env entry does not move the key")
        expect(one("gate-gpu-q4k-sel@BLOOMERY_GATE_CARD=a6000") != base["gpu"], "the lane's card does not move the key")
        expect(one("gate-gpu-q4k-sel") != base["gpu"], "no card (--lanes 1) gives the 3090 lane's key")
        expect(one("gate-gpu-e2e@BLOOMERY_GATE_CARD=a6000:--y") != base["args"], "ARGS do not move the key")
        expect(moved(keys(sa, box_env="BLOOMERY_X=1")) == set(items), "the caller's BLOOMERY_BOX_ENV does not move every key")
        expect(moved(keys(sa, environ={"BLOOMERY_DATA": "/x"})) == set(items), "the Mac's BLOOMERY_DATA does not move every key")
        for i in range(1, len(manifest) - 1):
            mut = list(manifest)
            mut[i] = mut[i] + "x"
            write_manifest(mut)
            m = moved(keys(sa, only=two))
            expect(m == set(two), f"manifest line {manifest[i].split(chr(9))[0]} moves {sorted(m)}")
        mut = list(manifest)
        mut[0] = mut[0] + " took=9s"
        write_manifest(mut)
        expect(not moved(keys(sa, only=two)), "the manifest's comment line moves a key")
        write_manifest(manifest[:-1])
        expect(all(v.startswith("error") for v in keys(sa).values()), "a truncated manifest (no `# end`) keyed an item")
        write_manifest(manifest)
        expect(all(v.startswith("error") for v in keys(sa, manifest_path=os.path.join(tmp, "absent")).values()), "a missing manifest keyed an item")
        c = KeyContext(sa, mf, "", {}, manifest_error="ssh failed", settings=settings)
        expect(c.parts(items["gpu"])[1] is not None, "a failed manifest fetch keyed an item")
        c = KeyContext(sa, mf, "", {}, settings=settings)
        expect(c.parts("gate gpu")[1] is not None and c.parts("gate-nope")[1] is not None, "a malformed or unknown item was keyed")
        expect(c.parts("gate-sampler@BLOOMERY_REF_MODEL=/models/small/x.gguf")[2] is not None, "an item env naming a path was skippable")
        c2 = KeyContext(sa, mf, "BLOOMERY_KLD_FILE=/root/x.kld", {}, settings=settings)
        expect(c2.parts(items["cpu"])[2] is not None, "a caller env path outside data and /models was skippable")
        c3 = KeyContext(sa, mf, "BLOOMERY_DATA=/root/bloomery-data", {}, settings=settings)
        expect(c3.parts(items["cpu"])[2] is None, "the data directory itself made an item never-skip")
        expect(c.parts("gate-1-1")[2] is not None and c.parts("gate-tokenizer")[2] is not None, "an ik-reading gate was skippable")
        expect(c.parts("smoke")[2] is not None and c.parts("check-recipes")[2] is None, "smoke (a batch in a batch) or check-recipes has the wrong never-skip")
        expect(c.parts("gate-gpu-e2e")[2] is not None, "an `any`-card recipe with no card forced was skippable")
        expect(c.parts("gate-gpu-e2e@BLOOMERY_GATE_CARD=a6000:--in /root/x.bin")[2] is not None, "an absolute path in ARGS was skippable")
        argpath = "gate-gpu-e2e@BLOOMERY_GATE_CARD=a6000:--in docs/plan.md"
        plan = os.path.join(roots[0], "docs/plan.md")
        with open(plan, encoding="utf-8") as fh:
            plan_text = fh.read()
        before = one(argpath)
        with open(plan, "w", encoding="utf-8") as fh:
            fh.write(plan_text + "\nkey probe\n")
        after = one(argpath)
        with open(plan, "w", encoding="utf-8") as fh:
            fh.write(plan_text)
        expect(not before.startswith("error") and before != after, "a tree file named in ARGS is not in the key")
        expect(c.parts("gate-gpu-e2e@BLOOMERY_GATE_CARD=a6000")[2] is None and c.parts("gate-gpu-q4k-sel")[2] is None, "a card-determined item was never-skip")
        expect(c.parts("ptx-scan@BLOOMERY_GATE_CARD=a6000:generate_ds41 --features gpu,deepseek41")[2] is not None,
               "an item whose cargo selector is a recipe parameter was skippable")
        # a module file the walk reaches through `mod` (not a target root): gone, the item is an error
        roots_src = {t.src for pkg in sa.tree.packages.values() for t in pkg.targets}
        for which in ("cpu", "gpu"):
            name = items[which].split("@")[0]
            mods = sorted(f for f in sa.graph.inputs(name).files if f.endswith(".rs") and f not in roots_src and f.startswith("crates/"))
            if mods:
                break
        victim = os.path.join(roots[0], mods[0])
        hold = victim + ".held"
        os.rename(victim, hold)
        try:
            gone = keys(side_at(roots[0], meta), only={which: items[which]})
            expect(gone[which].startswith("error"), f"with {mods[0]} missing, {items[which]} was keyed: {gone[which]}")
        finally:
            os.rename(hold, victim)




def self_test() -> int:
    fails: list[str] = []

    def expect(cond: bool, what: str) -> None:
        if not cond:
            fails.append(what)

    # shell splitting and the four command shapes
    rc = recipe_commands(
        Recipe(
            "x",
            [
                "BLOOMERY_MODEL=qwen3moe ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_a --bin gate_b && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_a {{ARGS}} && bash tools/gate.sh --oxide -p bloomery-gpu --release --lib -- --include-ignored && timeout --kill-after=10 300 bash crates/tokenizer/tools/oracle.sh && S=$(. tools/ref/models/deepseek41.sh && printf %s \"$X\") && cargo build --release -p bloomery-model --bin markov-accept > $D/x.log 2> $D/y.err'"
            ],
            [],
            "",
        )
    )
    expect(rc.box and rc.env.get("BLOOMERY_MODEL") == "qwen3moe", "box line env")
    kinds = [(i.sub, i.oxide, i.packages, i.selectors, i.features) for i in rc.invocations]
    expect(kinds[0] == ("build", True, ["bloomery-gpu-gates"], [("bin", "gate_a"), ("bin", "gate_b")], ["gpu"]), f"oxide build parse {kinds[:1]}")
    expect(kinds[1] == ("test", True, ["bloomery-gpu"], [("lib", None)], []), f"gate.sh --oxide parse {kinds[1:2]}")
    expect(kinds[2] == ("build", False, ["bloomery-model"], [("bin", "markov-accept")], []), f"plain build parse {kinds[2:3]}")
    expect(rc.runs == ["gate_a"], f"runner binaries {rc.runs}")
    expect("crates/tokenizer/tools/oracle.sh" in rc.scripts and "tools/ref/models/deepseek41.sh" in rc.scripts, f"scripts {rc.scripts}")
    try:
        parse_cargo(["build", "--frobnicate"], "cargo build --frobnicate")
        fails.append("unknown cargo option accepted")
    except RecipeError:
        pass

    # the real tree
    side = make_side(ROOT)
    tree, recipes, graph = side.tree, side.recipes, side.graph
    gates = [n for n in recipes if n.startswith(GATE_PREFIX)]
    expect(len(gates) >= 80, f"only {len(gates)} gate-* recipes parsed")
    for n in gates:
        expect(bool(graph.inputs(n).targets), f"{n} yields no target")
    problems = check(tree, recipes)
    expect(not problems, "check on the real justfile: " + "; ".join(problems[:5]))
    expect(not tree.unresolved, f"unresolved modules: {tree.unresolved[:3]}")

    def sel(f: str) -> set[str]:
        return {n for n in gates if graph.inputs(n).match(f)}

    one = sel("crates/gpu-gates/src/bin/gate_p1.rs")
    expect(one == {"gate-gpu-p1"}, f"gate_p1.rs selects {sorted(one)}")
    shared = sel("crates/gpu-gates/src/bin/shared/ds41_shadow.rs")
    expect({"gate-gpu-ds41-step", "gate-gpu-ds41-skew"} <= shared and "gate-gpu-e2e" not in shared, f"shared/ds41_shadow.rs selects {sorted(shared)}")
    samp = sel("crates/sampler/src/lib.rs")
    expect("gate-sampler" in samp and "gate-ops" not in samp and "gate-gpu-e2e" not in samp, f"sampler lib selects {sorted(samp)}")
    ds41 = sel("crates/gpu-deepseek41/src/lib.rs")
    expect(ds41 and not any("qwen3moe" in n for n in ds41) and "gate-gpu-e2e" not in ds41, f"gpu-deepseek41 lib selects qwen3/e2e: {sorted(ds41)[:6]}")
    prompts = sel("tools/ref/prompts.tsv")
    expect({"gate-prompts", "gate-gpu-e2e", "gate-gpu-hybrid"} <= prompts, f"prompts.tsv selects {sorted(prompts)}")
    common = sel("crates/model/tests/common/asserts.rs")
    expect("gate-attn" in common and "gate-kv" not in common, f"tests/common/asserts.rs selects {sorted(common)}")
    fixture = sel("crates/serve/tests/fixtures/qwen3-renders.json")
    expect(fixture == {"gate-serve"}, f"serve fixture selects {sorted(fixture)}")
    spill = sel("tools/ref/ptx-shapes.tsv")
    expect(spill == {"gate-ptx-spill"}, f"ptx-shapes.tsv selects {sorted(spill)}")
    qprof = sel("tools/ref/models/qwen3moe.sh")
    expect("gate-gpu-qwen3moe-e2e" in qprof and "gate-gpu-e2e" not in qprof, f"qwen3moe profile selects {sorted(qprof)[:5]}")
    expect(sel("docs/plan.md") == set(), "docs/plan.md is selected")
    cases = sel("crates/tokenizer/tests/cases.txt")
    expect(cases == {"gate-tokenizer"}, f"tokenizer cases.txt (read by oracle.sh) selects {sorted(cases)}")
    expect(len(sel("tools/box.sh")) == len(gates), "tools/box.sh does not select every gate")

    # FAIL-first on a mutated justfile: a typo'd --bin, --test, -p, feature and runner name
    with open(os.path.join(ROOT, "justfile"), encoding="utf-8") as fh:
        text = fh.read()
    # the anchors are found in the text, not spelled here, so a justfile edit does not rot them
    mutations = []
    for pat, typo in [
        (r"--bin (gate_\w+) &&", lambda w: w + "x"),
        (r"--test (\w+) --", lambda w: w[::-1]),
        (r"-p (bloomery-[a-z0-9-]+) --test", lambda w: w + "x"),
        (r"--features (deepseek41|vision) --release --bin", lambda w: w + "x"),
        (r"tools/gpu-gate\.sh (gate_\w+)'", lambda w: w + "x"),
    ]:
        m = re.search(pat, text)
        if m is None:
            fails.append(f"no mutation anchor for {pat}")
            continue
        bad = typo(m.group(1))
        mutations.append((m.start(1), m.end(1), bad))
    with tempfile.TemporaryDirectory(prefix="recipes-selftest-") as tmp:
        for start, end, bad in mutations:
            p = os.path.join(tmp, "justfile")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(text[:start] + bad + text[end:])
            probs = check(Tree(ROOT), load_justfile(p))
            expect(any(bad in x for x in probs), f"mutation {bad} not caught: {probs[:2]}")

    # a changed recipe text selects that recipe and no other
    import copy

    b2 = copy.copy(side)
    b2.recipes = dict(recipes)
    r = copy.copy(recipes["gate-sampler"])
    r.text = r.text + " "
    b2.recipes["gate-sampler"] = r
    sels, unmapped, _ = select(["justfile"], side, b2, GATE_PREFIX)
    expect([x.recipe for x in sels] == ["gate-sampler"] and not unmapped, f"recipe-text change selects {[x.recipe for x in sels]}")
    sels, _, _ = select(["crates/gpu-gates/src/bin/gate_p1.rs", "docs/plan.md"], side, side, GATE_PREFIX)
    expect([x.recipe for x in sels] == ["gate-gpu-p1"], f"select() on gate_p1.rs gives {[x.recipe for x in sels]}")

    # dep-info parsing
    t, deps, root = parse_depinfo("/r/b/target/release/g: /r/b/crates/a\\ b.rs /r/b/crates/c.rs\n")
    expect(t.endswith("/g") and root == "/r/b" and "/r/b/crates/a b.rs" in deps, "dep-info parse")

    # the green ledger's key
    key_self_test(expect, side)

    for f in fails:
        print(f"self-test FAIL: {f}", file=sys.stderr)
    print(f"self-test: {'FAIL' if fails else 'ok'} ({len(gates)} gate recipes, {len(fails)} failures)")
    return 1 if fails else 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--self-test", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    c = sub.add_parser("check")
    c.add_argument("--justfile")
    t = sub.add_parser("targets")
    t.add_argument("recipes", nargs="*")
    a = sub.add_parser("affected")
    a.add_argument("base", nargs="?", default="main")
    a.add_argument("--no-box", action="store_true", help="do not read the box's dep-info")
    a.add_argument("--all-recipes", action="store_true", help="every recipe, not only gate-*")
    a.add_argument("--box", default=os.environ.get("BLOOMERY_BOX", "ws"))
    a.add_argument("--depinfo-remote", default=os.environ.get("BLOOMERY_DEPINFO_REMOTE", "~/repo/bloomery"))
    w = sub.add_parser("why")
    w.add_argument("files", nargs="+")
    w.add_argument("--all-recipes", action="store_true")
    k = sub.add_parser("key", help="each ITEM's ledger key: key<TAB>item<TAB>status<TAB>detail")
    k.add_argument("items", nargs="+", metavar="ITEM", help="NAME[@K=V,…][:ARGS], as tools/gate-batch.sh runs it (its lane's card in the env)")
    k.add_argument("--manifest", help="the box manifest (box-manifest's output)")
    k.add_argument("--manifest-error", help="the manifest fetch failed with this message: every item is an error")
    k.add_argument("--box-env", default=os.environ.get("BLOOMERY_BOX_ENV", ""), help="the caller's BLOOMERY_BOX_ENV")
    k.add_argument("--ledger", help="the ledger file: status skip | run | rerun | never | error instead of key")
    k.add_argument("--rerun", action="store_true", help="with --ledger: no item skips")
    k.add_argument("--parts-dir", help="write each item's labelled parts to DIR/<index>.parts")
    k.add_argument("--parts", dest="show_parts", action="store_true", help="print each item's labelled parts")
    b = sub.add_parser("box-manifest", help="on the box, through tools/box.sh: the key's box part")
    b.add_argument("--lease", action="append", help="a timing lease lock; held, the manifest refuses (exit 75)")
    b.add_argument("--cache", default="~/.cache/bloomery/sha256-cache.tsv", help="the stat-keyed sha256 cache")
    b.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    try:
        if args.self_test:
            return self_test()
        if args.cmd == "check":
            return cmd_check(args)
        if args.cmd == "targets":
            return cmd_targets(args)
        if args.cmd == "affected":
            return cmd_affected(args)
        if args.cmd == "why":
            return cmd_why(args)
        if args.cmd == "key":
            return cmd_key(args)
        if args.cmd == "box-manifest":
            return cmd_box_manifest(args)
        ap.print_help()
        return 64
    except RecipeError as err:
        print(f"recipes.py: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
