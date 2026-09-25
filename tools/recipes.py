#!/usr/bin/env python3
"""The justfile's recipes as cargo targets and input files — one parser for check-recipes and affected.

Runs on the Mac: it reads the justfile through `just --dump --dump-format json` (just's own parser),
the workspace through `cargo metadata --format-version 1 --no-deps --offline --locked` (no build, no
network, never rewrites Cargo.lock), and the sources as text. It builds nothing and runs no gate.

    python3 tools/recipes.py check            # the target checks of `just check-recipes`
    python3 tools/recipes.py targets [RECIPE]  # recipe -> cargo targets, scripts, input count
    python3 tools/recipes.py affected [BASE | A..B] [--no-box] [--all-recipes]
    python3 tools/recipes.py why FILE...       # every recipe a file selects, with the chain
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
import datetime
import difflib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
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
# self-test
# ----------------------------------------------------------------------------------------------


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
        ap.print_help()
        return 64
    except RecipeError as err:
        print(f"recipes.py: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
