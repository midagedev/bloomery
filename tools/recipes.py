#!/usr/bin/env python3
"""The justfile's recipes as cargo targets and input files — one parser for check-recipes and affected.

Runs on the Mac: it reads the justfile through `just --dump --dump-format json` (just's own parser),
the workspace through `cargo metadata --format-version 1 --no-deps --offline --locked` (no build, no
network, never rewrites Cargo.lock), and the sources as text. It builds nothing and runs no gate.

    python3 tools/recipes.py check            # the target checks of `just check-recipes`, and the plan files records-refresh writes (presence)
    python3 tools/recipes.py targets [RECIPE]  # recipe -> cargo targets, scripts, input count
    python3 tools/recipes.py affected [BASE | A..B] [--no-box] [--all-recipes] [--narrow [--scan BASE_LOG NEW_LOG]...]
    python3 tools/recipes.py scan-set [REV]    # the bins whose ptx-scans cover every kernel carrier: bin<TAB>features<TAB>carriers (just narrow scans them)
    python3 tools/recipes.py why FILE...       # every recipe a file selects, with the chain
    python3 tools/recipes.py reach FILE... [--brief] [--row]  # the gates whose targets can run the file's code, each with a chain
    python3 tools/recipes.py key --manifest F [--ledger L [--round-ledger R]] ITEM...  # the green ledger's key per item
    python3 tools/recipes.py box-manifest      # on the box, through tools/box.sh: the key's box part
    python3 tools/recipes.py box-command RECIPE  # the recipe's box.sh command, verbatim (tools/mac-check.sh derives from it)
    python3 tools/recipes.py opens [RECIPE...] [--explain]  # per recipe, whether a target it builds or runs reaches the fixture fence (tools/gate-batch.sh's real-file deferral)
    python3 tools/recipes.py pure-crates [--names]  # the crates tools/mac-check.sh test runs natively on the Mac, each rejected one with its reason
    python3 tools/recipes.py host-nightly [--excluded]  # the units tools/nightly/run.sh runs on a Linux host with no card and no model files: name<TAB>command
    python3 tools/recipes.py combos           # every build shape a recipe compiles, one cargo check command each (tools/mac-check.sh combos)
    python3 tools/recipes.py orphan-tests      # every #[test] no gate-* or lab-* recipe runs on the box
    python3 tools/recipes.py --self-test

`affected` prints the gate-* recipes whose inputs a diff touches; nothing runs. BASE (default
`main`) compares the working tree, uncommitted and untracked files included, with the merge base of
BASE and HEAD — a round's tree is never committed. `A..B` compares two commits exactly; both sides
are read from `git archive` copies, so the graph is the one those commits had.

A recipe's inputs are:
  - the files of every cargo target it builds or tests: the target's own module tree (`mod`,
    `#[path]`, `include_str!`/`include_bytes!`/`include!`), its package's lib, the libs of its
    workspace dependencies — feature-aware, so an optional dependency counts only when the recipe's
    `--features` enable it, dev-dependencies only for test targets — each package's Cargo.toml (read
    by table: its shared tables and the recipe's own targets' [[bin]]/[[test]]/[[bench]]/[[example]]
    entries, so another target's entry selects nothing; `manifest_view`) and build script, and
    repository paths named by a whole string literal (`"../../tools/ref/prompts.tsv"`
    joined to CARGO_MANIFEST_DIR is read at run time and is in no dep-info) — except a literal that
    resolves to a `.rs` file of a workspace package: source is reached by compilation (the module
    walk of whatever compiles it), and a whole-string `.rs` literal in this tree is documentation of
    an owner (levers' InPlace rows), never a run-time read. A `.rs` literal outside every package
    (an excluded crate) is still an input. A module declared under
    `#[cfg(test)]` (or `cfg(all(…, test, …))`), and whatever it includes or names, belongs to the test
    builds of its own target — a lib's test target (`--lib` under cargo test, `--tests`,
    `--all-targets`), a test, a bench, and a bin's or example's own tree — never to the lib as a
    dependency sees it;
  - on the box's last build, the bin's top-level dep-info (`target/release/<bin>.d`, which lists every
    local source file of the binary) — a cross-check of the walk, and a union with it;
  - the scripts the recipe text names (`tools/**`, `crates/*/tools/**`) and, transitively, the repository
    files those scripts name, and every file under a tree directory a shell script walks with `find`
    from its root variable (`find "$ROOT/docs" -name '*.md'`: the whole directory, whatever the filter);
  - for a box recipe, tools/box.sh and what it sources on every command: tools/ref/ref-paths.sh,
    tools/ref/models/deepseek41.sh and the recipe's profile (`BLOOMERY_MODEL=x` on the line, else
    deepseek2);
  - for a cargo recipe, Cargo.toml, Cargo.lock, rust-toolchain.toml, .cargo/config.toml, and
    .cargo/cuda-oxide.toml when the recipe runs `cargo oxide`;
  - its own text in the justfile (a recipe whose text changed is selected), and its dependencies'
    inputs.

Honest expectation: a change in crates/gpu or crates/model selects every GPU gate — that is the
crate graph's true answer (a one-line enum change in the fault word selects them all, and it should).
The saving is on leaf changes (serve, tokenizer, sampler, one gate binary, a runner, docs — which no gate
reads: the tokenizer oracle's Korean text is gathered by its dump, not by the gate); the larger
value is the list itself: no forgotten gate, and a gate left out is a printed record, not a judgment.
What this layer cannot see: data under $BLOOMERY_DATA and the ik trees (not in git), and a card
dependence (BLOOMERY_GATE_CARD). The box's dep-info can be stale or missing; the walk does not depend
on it, and the notes say which targets had one and how old it is.

Pure crates (`pure-crates`). A workspace crate is pure when `cargo test -p <crate>` builds and runs on aarch64-apple-darwin, and the
rule below decides it from the tree without building anything. Two conditions, both over every feature
of the crate (its optional dependencies enabled, `cfg(feature = …)` read as unknown), so the verdict
does not depend on which feature set a recipe picks:

  1. No device root in its dependency closure. The closure is the crate's normal, build and dev
     dependencies, the workspace ones' normal and build dependencies transitively (an optional one of a
     workspace dependency counts as enabled), and every external package's dependencies as Cargo.lock
     resolves them. The roots are read from Cargo.lock: every package whose source is the cuda-oxide git
     repository (NVIDIA/cuda-rust, or our fork's midagedev/cuda-oxide spelling), and `cuda-core` and
     `cuda-bindings` by name, from any source (the shared host crates, crates.io or that repository). The
     bloomery device crates (bloomery-gpu, -gpu-deepseek41, -gpu-vision) declare cuda-device and
     cuda-host themselves, so they are the first hop of the chain the reason prints, not a second list.
  2. No code the Mac cannot build or run, in the files `cargo test -p <crate>` compiles: the crate's own
     targets (lib with its test modules, bins, tests, examples, doctests, build script) and the libs of
     its dependencies as a dependent sees them. What counts is what the crates use:
     `std::arch::x86_64` / `core::arch::x86_64`, `#[target_feature(…)]`, `is_x86_feature_detected!`,
     `std::os::linux`, the Linux-only libc items the tree calls (cpu_set_t and CPU_*, sched_*affinity,
     posix_fadvise, sync_file_range, RUSAGE_THREAD, renameat2, prctl, MADV_POPULATE_*, the huge-page
     flags), memmap2's Linux-only advice (PopulateRead/Write, HugePage, NoHugePage), and a "/proc/" or
     "/sys/" path literal (the file does not exist on the Mac, so the code that reads it fails there).
     Comments are not code. An item or statement under a `#[cfg(…)]` that is false on
     aarch64-apple-darwin (target_arch, target_os, target_family, target_vendor, target_env,
     target_pointer_width, target_endian, unix, windows) is left out, up to its end — the `;` or `,` at
     its own depth, its closing brace, or the close of the block around it, whichever comes first. A
     file whose `#![cfg(…)]` is false is left out whole. `cfg(test)` is unknown in the crate's own files
     (both sides are built) and false in a dependency's (its test modules are not built). Any other
     predicate is unknown, and an unknown one leaves the item in.

The rule errs one way: whatever it cannot read stays in, so a crate is rejected with a line to look at,
never selected on a guess. A file module declared under a target cfg (`#[cfg(target_arch = "x86_64")]
mod avx;`) is still read — the module walk does not evaluate target cfgs. Two gaps go the other way,
and both fail loudly: a doc comment's example is blanked as a comment but compiled by `cargo test` as a
doctest, and a Linux-only name missing from the list above is not seen. Either way the crate is
selected and its native build fails in `just mac-test`, which names it — a rule bug to fix here, never
an exception list.

Host nightly (`host-nightly`). The units of the nightly run on a Linux machine that is not the box (tools/nightly/run.sh): what a
plain `cargo test` runs on x86_64 Linux with no card and no model file. One unit per workspace crate that reaches no device root
(the same Cargo.lock reading as `pure-crates`' first condition, over every feature, so a new crate joins the set by being a member
and is never left out by a list), plus the device-linked crates in HOST_DEVICE_LINKED — each with the reason its default-feature
tests open no card and build as host code. The Mac's second condition (x86_64 and Linux-only code) does not apply: the host is
x86_64 Linux. A unit runs `tools/gate.sh --release --locked -p <crate> --no-fail-fast`, so the bound and the exit code have their one
owner; `#[ignore]`d tests (the `hw_` gates, which need a card, the model files or `$BLOOMERY_DATA`) stay ignored. A non-ignored test
that needs such a file is HOST_TEST_SKIP's, by exact libtest name and reason, and `--excluded` prints every exclusion; the rest
of the units are every `tools/check-*.sh` minus HOST_CHECK_SKIP, then HOST_EXTRA_UNITS (the recipes.py self-test). A skip that names no `fn` of its crate, a crate outside the set or a script that is gone is an
error of `check` and of this command, never a silent no-op.

Orphan tests (`orphan-tests`). Every `#[test]` fn of every workspace target (lib, bins, tests/*.rs) is run
on the box by some cargo test call of a gate-* or lab-* recipe (a lab-* recipe is a lab crate's own test
runner; `just mac-test` runs on the Mac and is not one). A call runs a test when all hold:
  1. it runs the test's target — `--lib` for a lib's tests, `--bin N`, `--test N`, no selector for all
     three kinds (doctests are not #[test] fns; example and bench tests run only with `test = true`);
  2. every cfg on the test's path — the file's `#![cfg]`, each module's `#[cfg]`, the fn's own — holds on
     x86_64-unknown-linux-gnu with cfg(test) on and the call's features: its `--features`, the defaults
     unless `--no-default-features`, their closure;
  3. its name filter, if any, matches the libtest name (the module path from the target's root, then the fn:
     a substring, or equality under `--exact`), and no `--skip` does;
  4. its ignore flags reach it: an `#[ignore]`d test needs `--ignored` or `--include-ignored`, and
     `--ignored` alone runs no other.
The check prints `path:line name — why` per test no call runs, then a count, and exits 1; 0 when there is
none. What it cannot read exits 2, named by file and line, never a pass: a cfg it cannot parse or whose key
it does not know (target_feature, debug_assertions — the profile decides it), a `cfg_attr` carrying cfg,
ignore, path or test, a test attribute from a macro crate (`#[tokio::test]`), a test attribute inside a
macro or any other non-module item, a module-level `include!`, a test argument from a recipe parameter, a
libtest option it does not know. Two gaps it does not see: tests a macro defined elsewhere expands into
(its invocation carries no test attribute), and anything outside the workspace members.

Build shapes (`combos`). Every cargo build a recipe runs on the box compiles one package under one feature set,
in one of two modes: a plain build, or a test build (cfg(test): a lib's or a bin's unit tests under
`cargo test`, a `--test` or `--bench` target under any subcommand). A shape is (package, mode, the package's
enabled features — the closure, defaults included — and any `dep/feature` words for other packages); the
recipes that share a shape share one command, `cargo check -p <package> [--profile test] [--features …]` over
the union of their targets, so each shape is type-checked the way the box compiles it. One package a command:
two in one invocation would unify a shared dependency's features and hide a feature one of them lacks. A
shape that another covers (the same package with more features) is still its own command — a
`cfg(not(feature))` or an optional dependency makes them different code. Named, never silent, and left out:
the `check` and `lint` recipes (`tools/mac-check.sh check` and `lint` run their commands); an invocation
whose package, features or targets come from a recipe parameter (`{{…}}`: known at run time only); a
doctest (rustdoc compiles it and cargo check has no doctest mode — the lib it imports is checked in the
package's plain shape). The profile is not part of a shape: no source in the tree reads
`cfg(debug_assertions)`, and `cargo check` builds no code whose optimisation level matters. `--base` scopes
the list for a round's loop (a shape whose inputs — shape_inputs — hold no file changed since the base, or
whose input key is green in the `--ledger`, skips with why; a run's combo lines gain the key as a fourth
field, which `tools/mac-check.sh combos --ledger` appends to the ledger green), so a loop that edited one
crate checks only the shapes that read it; without the flags the list is the full one the lead's landing uses.
"""

from __future__ import annotations

import argparse
import bisect
import concurrent.futures
import datetime
import difflib
import fcntl
import functools
import glob
import hashlib
import importlib.util
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
from collections.abc import Iterable
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GATE_PREFIX = "gate-"
# A gate taken out of the landing batch: `just weekly` runs every one, and `just affected` names one only when a
# changed file matches one of its trigger rows in tools/gate-paths.tsv, or its own recipe text changed.
WEEKLY_PREFIX = "weekly-"
# The checks the lead's batches always run; the diff does not pick them.
ALWAYS = ["check-recipes", "check-rustflags", "check-comments", "check-levers", "check-arch", "check", "fmt-check", "lint"]
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
    groups: tuple[str, ...] = ()  # its `[group('…')]` attributes, the scheduling classes of tools/gate-batch.sh


def _render_expr(expr) -> str:
    if isinstance(expr, list) and len(expr) == 2 and expr[0] == "variable":
        return expr[1]
    return "…"


def join_continued(name: str, lines: list[str]) -> list[str]:
    """A recipe's lines as just runs them: `just --dump` gives a line that ends in a backslash and the
    next as two body lines, and just runs them as one, the backslash dropped and the next line's
    leading whitespace with it (inside quotes too). A backslash on the recipe's last line continues
    nothing and is refused by name."""
    out: list[str] = []
    acc: str | None = None
    for ln in lines:
        if acc is not None:
            ln = acc + ln.lstrip()
            acc = None
        if ln.endswith("\\"):
            acc = ln[:-1]
            continue
        out.append(ln)
    if acc is not None:
        raise RecipeError(f"recipe {name}: its last line ends in a backslash, a continuation of nothing: {acc[-120:]}\\")
    return out


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
        lines = join_continued(name, lines)
        deps = [d["recipe"] for d in rec["dependencies"]]
        text = json.dumps({"p": rec["parameters"], "d": rec["dependencies"], "b": rec["body"], "a": rec["attributes"]}, sort_keys=True)
        groups = tuple(a["group"] for a in rec["attributes"] if isinstance(a, dict) and "group" in a)
        recipes[name] = Recipe(name=name, lines=lines, deps=deps, text=text, groups=groups)
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
    names: list[str] = field(default_factory=list)  # positional test-name filters before `--` (`cargo test <filter>`)
    test_args: list[str] = field(default_factory=list)  # the test binary's arguments, after `--`


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
    test_args: list[str] = []
    if "--" in rest:
        test_args = rest[rest.index("--") + 1 :]
        rest = rest[: rest.index("--")]
    inv = Invocation(sub=sub, oxide=oxide, raw=raw, test_args=test_args)
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
        else:
            # a positional word is a test-name filter (`cargo test <filter>`); it selects no target
            inv.names.append(w)
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
# A shell script's `find` rooted at a directory of the tree under its root variable (`find "$ROOT/docs"
# -name '*.md'`) reads files no path in the script names: the directory is an input, a prefix. A bare
# `find crates … -newer` (timing-card.sh's staleness probe) reads mtimes, not contents, and is not one.
_SCRIPT_WALK = re.compile(r"\bfind\s+\"?\$\{?(?:ROOT|HERE)\}?\"?/((?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_-][A-Za-z0-9_.-]*)\"?(?=[\s);|&]|$)")
_SCRIPT_NAME = re.compile(r"([A-Za-z0-9_.-]+\.(?:sh|py|tsv|cpp|h|txt|json|jinja))\b")


@functools.cache
def _script_refs(text: str, c_like: bool, shell: bool) -> tuple[tuple[tuple[str, ...], ...], ...]:
    """What each non-comment line of a script names, as three tuples of one entry a line: its repository
    paths (normalized), its `find` roots (a shell script's only) and its bare file names, each as the
    patterns read them and before the tree says which exist. A pure function of the text, cached by it:
    every recipe's closure reads the same scripts again."""
    paths: list[tuple[str, ...]] = []
    walks: list[tuple[str, ...]] = []
    names: list[tuple[str, ...]] = []
    for line in text.split("\n"):
        st = line.strip()
        if not st or st.startswith("//" if c_like else "#"):
            continue
        paths.append(tuple(_norm(p) for p in _SCRIPT_PATH.findall(line)))
        walks.append(tuple(_norm(d) for d in _SCRIPT_WALK.findall(line)) if shell else ())
        names.append(tuple(_SCRIPT_NAME.findall(line)))
    return tuple(paths), tuple(walks), tuple(names)


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
        # A repository file on a Mac-side `<` is read by the command (a script over ssh's stdin:
        # `tools/box.sh 'bash -s' < tools/box-gc.sh`); simple_commands drops redirect targets.
        for w, nxt in zip(local, local[1:]):
            if w == "<" and re.match(r"^(\./)?(tools|crates)/", nxt):
                p = _norm(nxt)
                rc.paths.append(p)
                if p.endswith(_SCRIPT_EXT):
                    rc.scripts.append(p)
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


class BoxCommandError(RecipeError):
    """A recipe whose box command cannot be printed as it is: `box-command` exits 64 on it."""


_BOX_LINE = re.compile(r"^(?:\./)?tools/box\.sh '([^']*)'$")


def box_command(recipe: Recipe) -> str:
    """The single-quoted argument of the recipe's one `tools/box.sh '…'` line, verbatim: tools/mac-check.sh
    derives its cargo call from it, so the recipe stays the one owner of its command. Refused by name: no
    box.sh line, more than one, an environment prefix on the call (it would not reach the Mac), an argument
    that is not one single-quoted word ending the line, a `{{…}}` parameter left in it."""
    found: list[str] = []
    for line in recipe.lines:
        stripped = line.lstrip("@-").strip()
        if not stripped or stripped.startswith("#"):
            continue
        for cmd in simple_commands(shell_words(stripped)):
            env, rest = strip_wrappers(cmd)
            if not rest or _norm(rest[0]) != "tools/box.sh":
                continue
            if env or rest[0] != cmd[0]:
                raise BoxCommandError(f"{recipe.name}: its box.sh call carries a prefix ({' '.join(cmd[: len(cmd) - len(rest)])}) that would not reach the Mac")
            m = _BOX_LINE.match(stripped)
            if m is None:
                raise BoxCommandError(f"{recipe.name}: its box.sh line is not `./tools/box.sh '<command>'` alone on the line: {stripped[:120]}")
            found.append(m.group(1))
    if not found:
        raise BoxCommandError(f"{recipe.name}: no tools/box.sh line")
    if len(found) > 1:
        raise BoxCommandError(f"{recipe.name}: {len(found)} tools/box.sh lines, not one")
    if "{{" in found[0]:
        raise BoxCommandError(f"{recipe.name}: its box command takes a parameter ({{{{…}}}}), which has no value outside just")
    if not found[0].strip():
        raise BoxCommandError(f"{recipe.name}: its box command is empty")
    return found[0]


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


# ---- a package manifest, read by table ----
#
# A target's build sees its package manifest's shared tables — everything but the target arrays below:
# [package], [lib], the dependency tables, [features], [lints], [profile*], [target.*] and any table
# not named here — and its own entry in one of the target arrays, matched by `name` or by a `path` that
# resolves to the target's source (an entry that claims another target's file is in that target's view).
# Another target's entry is not: adding a [[bin]] moves only the recipes that build it. An entry with
# no string `name`, or a target key that is not an array of tables, stays shared. Selection
# (`TableReads`) and the ledger key (`KeyContext._parts`) read the view through `manifest_view` alone;
# a manifest some input reads whole (a script, a path literal, `include_str!`, a `find`, dep-info) is
# a file for that recipe, never a view (`ManifestUse.whole`).
TARGET_TABLES = ("bin", "test", "bench", "example")
Entry = tuple[str, str, str]  # (kind, name, src) of a target whose entry a view keeps
_TOML_MEMO: dict[tuple[str, int, int], dict] = {}


def read_manifest(root: str, rel: str) -> dict:
    """`rel` parsed with tomllib, memoized on its stat. A manifest that does not parse is a named error,
    never a fallback to the whole file."""
    import tomllib

    p = os.path.join(root, rel)
    st = os.stat(p)
    k = (p, st.st_size, st.st_mtime_ns)
    if k not in _TOML_MEMO:
        with open(p, "rb") as fh:
            try:
                _TOML_MEMO[k] = tomllib.load(fh)
            except tomllib.TOMLDecodeError as err:
                raise RecipeError(f"{rel} does not parse as TOML ({err}): a package manifest is read by table") from err
    return _TOML_MEMO[k]


def _target_array(key: str, value) -> bool:
    return key in TARGET_TABLES and isinstance(value, list) and all(isinstance(e, dict) for e in value)


def _keeps(e: dict, kind: str, pkg_dir: str, wants: frozenset[Entry] | None) -> bool:
    """Whether a view of `wants` (None: every entry) keeps target-array entry `e` of `kind`."""
    name = e.get("name")
    if wants is None or not isinstance(name, str):
        return True
    path = e.get("path")
    src = os.path.normpath(os.path.join(pkg_dir, path)) if isinstance(path, str) else None
    return any((k == kind and n == name) or src == s for k, n, s in wants)


def canon(value) -> str:
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def manifest_view(doc: dict, pkg_dir: str, wants: frozenset[Entry] | None) -> dict:
    """The tables of `doc` a build of the targets `wants` sees (None: the whole document): the shared
    tables, and of each target array the entries it keeps, in canonical order (an array's order is not
    an input)."""
    out = {k: v for k, v in doc.items() if not _target_array(k, v)}
    for k in TARGET_TABLES:
        v = doc.get(k)
        if _target_array(k, v):
            kept = sorted((e for e in v if _keeps(e, k, pkg_dir, wants)), key=canon)
            if kept:
                out[k] = kept
    return out


def _by_name(entries: list[dict] | None) -> dict[str, str]:
    """A target array's entries by name ("" for the nameless), each name's entries in canonical form."""
    acc: dict[str, list[str]] = {}
    for e in entries or []:
        n = e.get("name")
        acc.setdefault(n if isinstance(n, str) else "", []).append(canon(e))
    return {n: canon(sorted(es)) for n, es in acc.items()}


def view_moves(da: dict | None, db: dict | None, pkg_dir: str, wants: frozenset[Entry] | None) -> list[tuple[str, bool]]:
    """What moved between two parses of a manifest in the view of `wants`, one (label, own) per table or
    entry — `[features]`, `[[bin]] x (added)` — where `own` is a named entry of a target array, not a shared
    table. Empty when the view did not move; a side with no file is the whole file."""
    if da is None or db is None:
        return [("the whole file (" + ("added" if da is None else "removed") + ")", False)] if da is not db else []
    va, vb = manifest_view(da, pkg_dir, wants), manifest_view(db, pkg_dir, wants)
    out: list[tuple[str, bool]] = []
    for k in sorted(set(va) | set(vb)):
        x, y = va.get(k), vb.get(k)
        if canon(x) == canon(y):
            continue
        if k in TARGET_TABLES and all(v is None or _target_array(k, v) for v in (x, y)):
            nx, ny = _by_name(x), _by_name(y)
            before = len(out)
            for n in sorted(set(nx) | set(ny)):
                if nx.get(n) == ny.get(n):
                    continue
                how = " (added)" if n not in nx else " (removed)" if n not in ny else ""
                out.append((f"[[{k}]] {n}{how}", True) if n else (f"[[{k}]] an entry with no name{how}", False))
            if len(out) == before:
                out.append((f"[[{k}]]", False))
        else:
            out.append((f"[{k}]", False))
    return out


@dataclass
class ManifestUse:
    """How a recipe's targets read the package manifests of their closure: `views` maps a manifest to
    the entries its view keeps (empty: a dependent's view, the shared tables alone), and `whole` holds
    the manifests some input reads as a file."""

    views: dict[str, set[Entry]] = field(default_factory=dict)
    whole: set[str] = field(default_factory=set)

    def merge(self, other: "ManifestUse") -> None:
        for m, es in other.views.items():
            self.views.setdefault(m, set()).update(es)
        self.whole |= other.whole

    def entries(self, rel: str) -> frozenset[Entry] | None:
        """The entries of `rel`'s view; None when the recipe reads `rel` whole, or not at all."""
        if rel in self.whole or rel not in self.views:
            return None
        return frozenset(self.views[rel])


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


@dataclass
class ModuleTree:
    """One target root's module tree (`Tree.walk`): its files and (file, literal) repository-path
    literals, and apart from them the ones only a module under `#[cfg(test)]` reaches."""

    files: set[str]
    literals: list[tuple[str, str]]
    test_files: set[str]
    test_literals: list[tuple[str, str]]
    mods: dict[tuple[str, ...], str] = field(default_factory=dict)  # module path below the root -> its file, inline modules too
    test_mods: dict[tuple[str, ...], str] = field(default_factory=dict)


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
        self.manifest_paths = {p.manifest for p in self.packages.values()}
        # `dir/` of every workspace package: a .rs literal under one of these is source, not data
        # (`_resolve_literals`); a .rs outside them (an excluded crate) stays an input.
        self.pkg_dirs = tuple(sorted(p.dir + "/" for p in self.packages.values() if p.dir not in ("", ".")))
        self._walk_cache: dict[str, ModuleTree] = {}
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
    _CFG = re.compile(r"^#\[\s*cfg\s*\((.*)\)\s*\]$")

    @staticmethod
    def cfg_test(attr: str) -> bool:
        """Whether a one-line `#[cfg(…)]` holds in test builds only: `test`, or `all(…)` with `test`
        among its own items. Any other predicate (`any(test, …)`, `not(…)`, a feature) does not, and a
        multi-line attribute is not read: the item stays in every build."""
        m = Tree._CFG.match(attr)
        if not m:
            return False
        pred = m.group(1).strip()
        if pred == "test":
            return True
        a = re.match(r"^all\s*\((.*)\)$", pred)
        if not a:
            return False
        items, depth, cur = [], 0, ""
        for ch in a.group(1):
            if ch == "," and depth == 0:
                items.append(cur.strip())
                cur = ""
                continue
            depth += (ch == "(") - (ch == ")")
            cur += ch
        items.append(cur.strip())
        return "test" in items

    def walk(self, src: str) -> ModuleTree:
        """The module tree rooted at `src`. A module declared under a test-only `cfg` (`cfg_test`),
        file or inline, and what it declares, includes and names go to the test side; a file reached
        both ways is on the build side."""
        acc = ModuleTree(set(), [], set(), [])
        self._walk(src, os.path.dirname(src), acc, False)
        acc.test_files -= acc.files
        for m in set(acc.test_mods) & set(acc.mods):
            del acc.test_mods[m]
        return acc

    def _walk(self, rel: str, moddir: str, acc: ModuleTree, test: bool, modpath: tuple[str, ...] = ()) -> None:
        (acc.test_mods if test else acc.mods).setdefault(modpath, rel)
        if rel in acc.files or (test and rel in acc.test_files):
            return
        (acc.test_files if test else acc.files).add(rel)
        fdir = os.path.dirname(rel)
        pending_path: str | None = None
        pending_test = False  # a test-only `#[cfg]` since the last item: the next module is test-only
        inline: list[tuple[int, str, bool]] = []  # (indent, name, test-only)
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
            here = inline[-1][2] if inline else test
            for lit in self._LITERAL.findall(line):
                (acc.test_literals if here else acc.literals).append((rel, lit))
            for inc in self._INCLUDE.findall(line):
                p = os.path.normpath(os.path.join(fdir, inc))
                if self.exists(p):
                    (acc.test_files if here else acc.files).add(p)
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
                child_test = here or pending_test
                pending_path = None
                pending_test = False
                inner = [n for (i, n, _) in inline if i < indent]
                hit, child_dir, err = self.mod_file(rel, moddir, inner, name, path_attr)
                if hit is None:
                    self.unresolved.append(err)
                    continue
                self._walk(hit, child_dir, acc, child_test, modpath + tuple(inner) + (name,))
                continue
            if im:
                inner = [n for (i, n, _) in inline if i < indent]
                (acc.test_mods if here or pending_test else acc.mods).setdefault(modpath + tuple(inner) + (im.group(2),), rel)
                inline.append((len(im.group(1)), im.group(2), here or pending_test))
                pending_path = None
                pending_test = False
                continue
            if s.startswith("#["):
                pending_test = pending_test or self.cfg_test(s)
                attr_depth = max(0, s.count("[") - s.count("]"))
                continue
            pending_path = None
            pending_test = False

    def mod_file(self, rel: str, moddir: str, inner: list[str], name: str, path_attr: str | None) -> tuple[str | None, str, str]:
        """(file, its module directory, "") of `mod name;` declared in `rel`, whose children live in
        `moddir`, inside the inline modules `inner` — or (None, "", why) when no file is there. The
        one resolver of a file module for both walks (`_walk`, `tests_of`)."""
        fdir = os.path.dirname(rel)
        if path_attr is not None:
            base = os.path.join(fdir, *inner) if inner else fdir
            child = os.path.normpath(os.path.join(base, path_attr))
            if not self.exists(child):
                return None, "", f"{rel}: #[path = \"{path_attr}\"] mod {name} ({child} not found)"
            # a #[path] file is a mod-rs file: its children live beside it
            return child, os.path.dirname(child), ""
        base = os.path.join(moddir, *inner) if inner else moddir
        cand = [os.path.normpath(os.path.join(base, name + ".rs")), os.path.normpath(os.path.join(base, name, "mod.rs"))]
        hit = next((c for c in cand if self.exists(c)), None)
        if hit is None:
            return None, "", f"{rel}: mod {name} (neither {cand[0]} nor {cand[1]})"
        return hit, os.path.dirname(hit) if hit.endswith("mod.rs") else hit[: -len(".rs")], ""

    def tree_of(self, src: str, test: bool = False) -> tuple[set[str], list[tuple[str, str]]]:
        """`src`'s module tree as one build of it sees it: a test build (`test`) with its test-only
        modules, any other build without them."""
        if src not in self._walk_cache:
            self._walk_cache[src] = self.walk(src)
        w = self._walk_cache[src]
        if test:
            return w.files | w.test_files, w.literals + w.test_literals
        return w.files, w.literals

    def mods_of(self, src: str, test: bool = False) -> dict[tuple[str, ...], str]:
        """`src`'s modules as `tree_of` sees them: module path below the root -> the file that holds it."""
        self.tree_of(src)
        w = self._walk_cache[src]
        return {**w.mods, **w.test_mods} if test else w.mods

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
        out: dict[str, str] = {self.packages[name].manifest: f"{name} Cargo.toml"}
        for f, why in self._lib_reads(name).items():
            out.setdefault(f, why)
        return out

    def _lib_reads(self, name: str) -> dict[str, str]:
        """lib_files but the manifest: what package `name`'s lib and build script read as files."""
        pkg = self.packages[name]
        out: dict[str, str] = {}
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
                if p.endswith(".rs") and p.startswith(self.pkg_dirs):
                    # a workspace package's source, not data: what compiles it reaches it through the
                    # module walk, and nothing in this tree opens a .rs path at run time (levers'
                    # InPlace rows name files as documentation). A run-time reader of one is a new
                    # kind of read this rule must learn by name, in its own round.
                    continue
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

    def target_files(self, pkg_name: str, kind: str, name: str | None, features: list[str], default: bool = True, all_features: bool = False, use: ManifestUse | None = None) -> dict[str, str]:
        """file -> origin for one target. kind: lib bin test example bench doctest libtest. The
        target's own test-only modules count for every kind but `lib` and `doctest` (rustdoc sets no
        cfg(test)); for a bin or an example that is also its plain build, an over-selection confined to
        its own recipes. Every lib it links, its own package's included, is seen without them. `use`,
        when given, gains how the target reads each manifest of its closure: by table — its own entry in
        its package's, the shared tables of every other — or whole, when a file walk reaches one."""
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
        files, lits = self.tree_of(own.src, test=kind not in ("lib", "doctest"))
        for f in files:
            out[f] = f"{pkg_name} {label}"
        for f in self._resolve_literals(pkg, lits):
            out.setdefault(f, f"{pkg_name} {label} (path literal)")
        for f, why in self.lib_files(pkg_name).items():
            out.setdefault(f, why)
        linked = self.closure(pkg, features, dev, default, all_features)
        for dep, why in linked.items():
            for f, fwhy in self.lib_files(dep).items():
                out.setdefault(f, f"{fwhy} ({why})")
        read: set[str] = set(files) | set(self._resolve_literals(pkg, lits))
        if kind == "bin" and name in self.depinfo:
            dfiles, _ = self.depinfo[name]
            if own.src in dfiles:
                read |= dfiles
                for f in dfiles:
                    if f not in out and (self.exists(f)):
                        out[f] = f"dep-info of {name}"
        if use is not None:
            mine = use.views.setdefault(pkg.manifest, set())
            if kind in TARGET_TABLES:
                mine.add((kind, own.name, own.src))
            for q in [pkg_name, *linked]:
                use.views.setdefault(self.packages[q].manifest, set())
                read |= set(self._lib_reads(q))
            use.whole |= read & self.manifest_paths
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
            # features from a recipe parameter are known at run time only, where cargo refuses a bin
            # whose required features they lack
            param = any("{{" in f for f in inv.features)

            def req_ok(t: TargetMeta) -> bool:
                return inv.all_features or param or set(t.required) <= enabled

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
                    # `--tests` (and `--all-targets`) compile the lib's test target under check, clippy
                    # and build as well as under test
                    if kind == "tests" and pkg.lib() is not None and not implicit:
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
    manifests: ManifestUse = field(default_factory=ManifestUse)  # how the targets read their package manifests

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
        """Repository files a script names, transitively (comments skipped), and as `dir/` each tree
        directory a shell script walks with `find`."""
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
            sdir = os.path.dirname(s)
            paths, walks, names = _script_refs(self.tree.read(s), s.endswith((".cpp", ".h", ".c", ".cc", ".hpp")), s.endswith((".sh", ".bash")))
            for line_paths, line_walks, line_names in zip(paths, walks, names):
                for p in line_paths:
                    if self.tree.exists(p):
                        stack.append(p)
                for d in line_walks:
                    if d != "." and self.tree.isdir(d):
                        out.setdefault(d + "/", f"find in {s}" if s == rel else f"find in {s} (script {rel})")
                for b in line_names:
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
        use = ManifestUse()
        for inv in rc.invocations:
            tl, errs = self.tree.resolve(inv)
            errors.extend(errs)
            for pname, kind, tname, feats in tl:
                targets.append((pname, kind, tname, feats))
                label = f"{kind} {tname}" if tname else kind
                for f, why in self.tree.target_files(pname, kind, tname, feats, not inv.no_default, inv.all_features, use).items():
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
                    if f.endswith("/"):
                        prefixes.setdefault(f, why)
                    else:
                        files.setdefault(f, why)
                        if f in self.tree.manifest_paths:
                            use.whole.add(f)
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
            use.merge(di.manifests)
            built |= {t for (_, k, t, _) in di.targets if k == "bin"}
        for b in rc.runs:
            if "{{" in b or b.startswith("$"):
                continue
            if b not in built:
                errors.append(f"runs target/release/{b}, which the recipe does not build")
        # a manifest that is a cargo global, or lies under a directory a script walks, is read whole
        use.whole |= {m for m in self.tree.manifest_paths if m in CARGO_GLOBALS or any(m.startswith(p) for p in prefixes)}
        ri = RecipeInputs(files, prefixes, targets, errors, rc, use)
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
        if name.startswith((prefix, WEEKLY_PREFIX)) and not ri.targets:
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


class TableReads:
    """The package manifests of one diff, each parsed on both sides, and what moved in each recipe's view
    of them (manifest_view). One reader for every place a changed file picks recipes (select, narrow); the
    key reads the same view. `hit` records each answer, which `lines` prints."""

    def __init__(self, changed: list[str], a: Side | None, b: Side):
        self.a, self.b = a, b
        self.docs: dict[str, tuple[dict | None, dict | None]] = {}
        for f in changed:
            if a is not None and (f in b.tree.manifest_paths or f in a.tree.manifest_paths):
                self.docs[f] = (self._doc(a, f), self._doc(b, f))
        self.seen: dict[str, dict[str, list[tuple[str, bool]] | None]] = {f: {} for f in self.docs}

    @staticmethod
    def _doc(side: Side, f: str) -> dict | None:
        return read_manifest(side.tree.root, f) if side.tree.exists(f) else None

    def moves(self, n: str, f: str) -> list[tuple[str, bool]] | None:
        """What moved in recipe `n`'s view of manifest `f`, its own entries taken from both sides (an entry
        renamed, or a target a required-features edit lets in or out of a `--bins`, is in one side's list);
        None when `f` is not a manifest of the diff or `n` reads it whole on either side (the file rule)."""
        if f not in self.docs:
            return None
        wants: set[Entry] = set()
        read = False
        for side in (self.a, self.b):
            if side is None or n not in side.recipes:
                continue
            mu = side.graph.inputs(n).manifests
            if f in mu.whole:
                self.seen[f][n] = None
                return None
            if f in mu.views:
                read = True
                wants |= mu.views[f]
        if not read:
            return None
        da, db = self.docs[f]
        mv = view_moves(da, db, os.path.dirname(f), frozenset(wants))
        self.seen[f][n] = mv
        return mv

    def hit(self, n: str, f: str, side: Side) -> tuple[str | None, bool | None]:
        """(why recipe `n` reads changed file `f` on `side`, with what moved in its view — None when it does
        not read it or its view did not move; whether every move is an entry of its own targets — None when
        `f` is read as a file). A manifest of the diff is read on A too: a recipe whose B build no longer
        reaches it (a renamed entry, a target a required-features edit leaves out) read it before."""
        why = side.graph.inputs(n).match(f)
        if why is None and f in self.docs and self.a is not None and n in self.a.recipes:
            why = self.a.graph.inputs(n).match(f)
        if why is None:
            return None, None
        mv = self.moves(n, f)
        if mv is None:
            return why, None
        if not mv:
            return None, None
        return f"{why}: {', '.join(m for m, _ in mv)}", all(own for _, own in mv)

    def lines(self) -> list[str]:
        """The debugging block of `affected`: per manifest, what changed by table and the recipes each change
        selects, then the recipes that read it whole."""
        out = []
        for f, (da, db) in self.docs.items():
            seen = self.seen[f]
            by_table = {n: mv for n, mv in seen.items() if mv is not None}
            whole = sorted(n for n, mv in seen.items() if mv is None)
            out.append(f"  {f}: {len(by_table)} recipes read it by table, {len(whole)} whole")
            for label, _ in view_moves(da, db, os.path.dirname(f), None) or [("nothing (comments or layout only)", False)]:
                sel = [n for n, mv in by_table.items() if any(m == label for m, _ in mv)]
                every = len(sel) == len(by_table) and len(sel) > 3
                out.append(f"    {label} -> " + (f"every reader ({len(sel)})" if every else " ".join(sel) if sel else "no recipe"))
            if whole:
                out.append("    read whole -> " + " ".join(whole))
        return out


def select(changed: list[str], a: Side | None, b: Side, prefix: str, tables: TableReads | None = None) -> tuple[list[Selection], list[str], list[str]]:
    """(selections in justfile order, unmapped files, notes). A manifest is read by table (`TableReads`)."""
    names = [n for n in sorted(b.recipes, key=lambda n: b.recipes[n].line) if n.startswith(prefix)]
    hits: dict[str, list[tuple[str, str]]] = {n: [] for n in names}
    mapped: set[str] = set()
    notes: list[str] = []
    tables = tables if tables is not None else TableReads(changed, a, b)
    for f in changed:
        if f == "justfile":
            continue
        side = b if b.tree.exists(f) or a is None else a
        for n in names:
            if n not in side.recipes:
                continue
            why, _ = tables.hit(n, f, side)
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
        if f == "justfile":
            hint = "no gate-* recipe text changed"
        elif any(mv is not None for mv in tables.seen.get(f, {}).values()):
            hint = "read by table: no recipe's view of it moved (manifests:)"
        else:
            hint = orphan_hint(side, f)
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
    if args.scan and not args.narrow:
        raise RecipeError("--scan needs --narrow")
    # Every scan log and the table are read before the diff: one the tool cannot vouch for is refused first.
    scans = [(read_scan(x), read_scan(y)) for x, y in args.scan or []]
    rows = load_gate_paths() if args.narrow else []
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
        tables = TableReads(changed, a, b)
        sels, unmapped, more = select(changed, a, b, prefix, tables)
        notes += more
        trows, tnote = trigger_rows(b)
        if tnote:
            notes.append(tnote)
        trig = weekly_triggers(changed, a, b, trows) if prefix == GATE_PREFIX else []
        named = {f for t in trig for f in t.files}
        unmapped = [u for u in unmapped if u.split("  (")[0] not in named]
        n_weekly = sum(1 for n in b.recipes if n.startswith(WEEKLY_PREFIX))
        weekly = f", {len(trig)} of {n_weekly} weekly-recipes by a trigger" if prefix == GATE_PREFIX else ""
        total = sum(1 for n in b.recipes if n.startswith(prefix))
        rng = f"{a_rev[:9]}..{b_rev[:9]}" if b_rev else f"{a_rev[:9]}..(working tree)"
        nar = narrow(changed, a, b, scans, rows, tables) if args.narrow else None
        narrowed = nar is not None and not nar.full
        if narrowed:
            print(f"affected: {rng} — {len(changed)} changed files, {len(nar.picks)} of {total} {prefix or ''}recipes selected "
                  f"(narrowed from {len(sels)} by ptx-scan){weekly}")
        else:
            print(f"affected: {rng} — {len(changed)} changed files, {len(sels)} of {total} {prefix or ''}recipes selected{weekly}")
        if nar is not None:
            for r in nar.full:
                print(f"narrow: full list — {r}")
            for v in nar.verdicts:
                print(f"narrow: ptx-scan {v.line()}")
            for line in nar.files:
                print(f"narrow: file {line}")
        if narrowed:
            width = max([len(n) for n in nar.picks] + [len(t.recipe) for t in trig] + [10])
            for n, why in nar.picks.items():
                print(f"  {n:<{width}}  {why}")
            print_triggers(trig, width)
            print("recipes: " + " ".join([*nar.picks, *(t.recipe for t in trig)]))
            print("always: " + " ".join(ALWAYS))
            let_out = "; ".join(v.line() for v in nar.verdicts)
            out = [s.recipe for s in sels if s.recipe not in nar.picks]
            print(f"left out ({len(out)}), each by ptx-scan {let_out}:")
            for n in out:
                print(f"  - {n}  (ptx-scan {let_out})")
        else:
            width = max([len(s.recipe) for s in sels] + [len(t.recipe) for t in trig] + [10])
            for s in sels:
                more_n = f" (+{len(s.files) - 1})" if len(s.files) > 1 else ""
                print(f"  {s.recipe:<{width}}  {s.files[0]}{more_n}  [{s.why}]")
            print_triggers(trig, width)
            print("recipes: " + " ".join([*(s.recipe for s in sels), *(t.recipe for t in trig)]))
            print("always: " + " ".join(ALWAYS))
        print(f"unmapped ({len(unmapped)}):")
        for u in unmapped:
            print(f"  {u}")
        if tables.docs:
            print(f"manifests ({len(tables.docs)}), read by table:")
            for line in tables.lines():
                print(line)
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


def print_triggers(trig: list["Trigger"], width: int) -> None:
    for t in trig:
        more_n = f" (+{len(t.files) - 1})" if len(t.files) > 1 else ""
        print(f"  {t.recipe:<{width}}  {t.files[0]}{more_n}  [{t.why}]")


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


def cmd_reach(args: argparse.Namespace) -> int:
    """`why`'s files through the derivation (Reach): the gate-* recipes whose targets can run code of each file through
    a dependency edge, each with one chain, and the readers the derivation leaves out by count. `why` lists every
    recipe that links a file (the crate-grain bound narrow() reads); `reach` is the subset a row must name."""
    side = make_side(ROOT)
    tree, graph = side.tree, side.graph
    gates = [n for n in sorted(side.recipes, key=lambda n: side.recipes[n].line) if n.startswith(GATE_PREFIX)]
    union: dict[str, None] = {}
    for f in args.files:
        f = _norm(f)
        reads = [(n, graph.inputs(n).match(f)) for n in gates]
        reads = [(n, w) for n, w in reads if w]
        own = [n for n, w in reads if _own_origin(w)]
        dep = [n for n, w in reads if not _own_origin(w)]
        if not tree.exists(f):
            print(f"{f}: not a file of the tree")
            continue
        if not reads:
            print(f"{f}: no gate-* recipe reads it")
            continue
        if f in tree.manifest_paths or f in CARGO_GLOBALS or f in BOX_GLOBALS:
            print(f"{f}: read by table or as a global ({len(dep)} recipes through a dependency): the derivation places no "
                  f"manifest or global; the rows do not map it")
            continue
        got = dep_reach(tree, graph, gates, f)
        held = [n for n in gates if n in got and n not in NARROW_ALWAYS]
        print(f"{f}: {len(held)} gates reach it through a dependency, of {len(dep)} that link it through one; "
              f"{len(own)} read it as their own target's (narrow() picks those without a row)")
        for n in held:
            union[n] = None
            print(f"  {n}" + ("" if args.brief else f"  {reach_chain(tree, graph, n, got[n])}"))
        left = [n for n in dep if n not in got and n not in NARROW_ALWAYS]
        if left:
            print(f"  not reached ({len(left)}): {' '.join(left)}")
    if args.row:
        names = [n for n in gates if n in union]
        glob = args.files[0] if len(args.files) == 1 else "{" + ",".join(_norm(f) for f in args.files) + "}"
        print(f"{glob}\t{' '.join(names) if names else '-'}\t({len(names)} gates)")
    return 0


def cmd_box_command(args: argparse.Namespace) -> int:
    recipes = load_justfile(os.path.join(ROOT, "justfile"))
    if args.recipe not in recipes:
        print(f"recipes.py box-command: no recipe {args.recipe}", file=sys.stderr)
        return 64
    try:
        print(box_command(recipes[args.recipe]))
    except BoxCommandError as err:
        print(f"recipes.py box-command: {err}", file=sys.stderr)
        return 64
    return 0


# ----------------------------------------------------------------------------------------------
# affected --narrow: ptx-scan logs and the host-path table
# ----------------------------------------------------------------------------------------------
# The rule (AGENTS.md, `just affected`): when `just ptx-scan` of every bin a change reaches equals
# the base, no kernel moved, and the landing batch is the gates that run the changed host path plus
# the static checks. The "add" class is the same with the base plus exactly the new entries.
#
# A log of `just ptx-scan <bin> --no-jit` (banner `jit=skipped(no-jit)`, the two JIT cells the word
# `skipped(no-jit)`) proves the first rule only: a pair with one, equal in entries, rows without the JIT
# columns and digests, narrows as an identical pair does, and its verdict says the JIT was skipped. Any
# other result of such a pair is refused by name (scan_verdict): the add and occupancy classes read the
# JIT columns.
#
# A scan log is the text `just ptx-scan <bin>` prints. tools/ref/ptx-canon.py's parse_scan is the one
# reader of its entry table and its md5 block; read_scan below adds what that reader does not see —
# the banner's fields and the `ptx-scan: modN bundle=<crate>` lines — and refuses a log it cannot
# vouch for by name: a failed or filtered scan, two scans in one log, a table row with no digest or a
# digest with no row.
#
# A changed file selects, under --narrow:
#   - the recipes whose own target reads it: the target's own module tree (a bin, a test file, a
#     lib's own tests, a `#[path]` module a bin includes), a script the recipe names and what that
#     script names, a path literal, the recipe's own text in the justfile;
#   - when some gate reads it through a dependency edge (a workspace lib a target links, a crate
#     manifest, a cargo global, box.sh's profile files): the recipes of the rows of
#     tools/gate-paths.tsv whose glob matches it, all of them. No row: the full list, naming the file.
#     A row whose recipes are `*`: the full list, naming the row;
#   - a file under docs/, a `.card` or a `*.md` needs no row (not a host path);
#   - the gates of a kernel carrier no scan pair covers, whenever the change touches that carrier's
#     closure: a device lib whose bundle is in no pair's banner, or a bin with kernels of its own
#     unless a pair is the scan of that very bin and names the package's bundle (carrier_covered).
# Then the static checks (ALWAYS), gate-ptx-spill when the PTX or its own pins can have moved (a
# scan pair not identical, or a changed file the recipe reads), and every recipe the diff adds.
#
# The same table holds the weekly tier's triggers. A `weekly-*` name in a row is a trigger: a changed file
# the row's glob matches names that recipe, with or without --narrow and whatever the scans say, and so
# does a change of the recipe's own text (weekly_triggers). Only a row's gate-* names map a file for the
# rule above: a file that only trigger rows match, read by a gate through a dependency, still keeps the
# full list.

GATE_PATHS = "tools/gate-paths.tsv"
PTX_CANON = "tools/ref/ptx-canon.py"
# The banner fields two scans of a pair must agree on: another digest rule, assembler, arch, JIT
# capability or driver makes the columns incomparable, which is not the same as a kernel that moved.
# The JIT columns follow the card's compute capability, not its name (both cards here are sm_86 under
# one driver, measured equal entry by entry): a pair scanned on either card of one capability
# compares, so the name (jit-card) is never a compared field. A `ptx-scan.sh --no-jit` scan names
# `jit=skipped(no-jit)` instead of the jit-* fields and holds no JIT reading, so a pair with one
# compares neither the jit-* fields nor the two JIT columns (scan_verdict).
SCAN_SAME = ("method", "ptxas-version", "arch", "jit-cc", "jit-cuda")
NOT_HOST = re.compile(r"^docs/|\.card$|\.md$")
# In a narrowed list only when the PTX or its own pins can have moved (narrow() owns the rule): the
# spill ratchet over the scanned bins.
NARROW_ALWAYS = ["gate-ptx-spill"]
KERNEL_ATTR = re.compile(r"#\s*\[\s*(?:kernel|cuda_module)\b")


@functools.cache
def _ptx_canon():
    """tools/ref/ptx-canon.py as a module (its file name is not an identifier)."""
    import importlib.util

    path = os.path.join(ROOT, PTX_CANON)
    spec = importlib.util.spec_from_file_location("ptx_canon", path)
    if spec is None or spec.loader is None:
        raise RecipeError(f"cannot load {path}, the ptx-scan logs' reader")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@dataclass
class ScanLog:
    path: str
    bin: str
    fields: dict[str, str]  # the banner's key=value words, and `method` from the md5 block
    bundles: list[str]
    header: list[str]
    rows: dict[str, list[str]]
    md5: dict[str, tuple[str, str]]
    nojit: bool = False  # a `--no-jit` scan: its JIT cells are the word `skipped(no-jit)`, not readings


_SCAN_BANNER = re.compile(r"^ptx-scan bin=\S")
_SCAN_BUNDLE = re.compile(r"^ptx-scan: mod[0-9]+ bundle=(\S+) bytes=[0-9]+$")


def read_scan(path: str) -> ScanLog:
    pc = _ptx_canon()
    try:
        method, header, rows, md5, nojit = pc.parse_scan(path)
    except pc.Refused as err:
        raise RecipeError(f"scan log {path}: {err}") from err
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().split("\n")
    banners = [ln for ln in lines if _SCAN_BANNER.match(ln)]
    if len(banners) != 1:
        raise RecipeError(f"scan log {path}: {len(banners)} `ptx-scan bin=` banners — one log holds one scan")
    fields = dict(w.split("=", 1) for w in banners[0].split()[1:] if "=" in w)
    if not nojit and "jit-card" not in fields and "jit-cc" not in fields:
        raise RecipeError(f"scan log {path}: the banner names no JIT reading (jit-card, jit-cc) and no jit=skipped(no-jit) — "
                          "which columns the JIT cells hold is unknown")
    if "filter" in fields:
        raise RecipeError(f"scan log {path}: a scan filtered to entries containing {fields['filter']!r} holds part of the table — scan the whole binary")
    bundles = [m.group(1) for m in map(_SCAN_BUNDLE.match, lines) if m]
    if not bundles:
        raise RecipeError(f"scan log {path}: no `ptx-scan: modN bundle=` line — which crates' kernels the scan read is unknown")
    if set(rows) != set(md5):
        odd = sorted(set(rows) ^ set(md5))
        raise RecipeError(f"scan log {path}: {len(odd)} entries in only one of the table and the md5 block ({' '.join(odd[:4])}) — a cut or mixed log")
    fields["method"] = method
    return ScanLog(path, os.path.basename(fields["bin"]), fields, bundles, header, rows, md5, nojit)


@dataclass
class ScanVerdict:
    bin: str
    kind: str  # identical | added | moved
    entries: list[str]  # added: the new entries; moved: the removed or changed ones
    detail: str
    bundles: list[str]

    def line(self) -> str:
        what = f" {' '.join(self.entries)}" if self.entries else ""
        return f"{self.bin}: {self.kind}{what}{f' ({self.detail})' if self.detail else ''}"


def _compared_rows(log: ScanLog, nojit: bool) -> dict[str, list[str]]:
    """The rows a pair compares: with a no-jit scan in the pair, the two JIT columns are cut from both sides."""
    if not nojit:
        return log.rows
    cut = _ptx_canon().JIT_COLUMNS
    keep = [i for i, h in enumerate(log.header[1:]) if h not in cut]
    return {e: [cells[i] for i in keep] for e, cells in log.rows.items()}


def scan_verdict(base: ScanLog, new: ScanLog) -> ScanVerdict:
    """identical, added (every base row and digest unchanged, only new entries) or moved (a base row or
    digest changed, or an entry removed). A pair of two binaries or of two scan setups is refused.

    A pair with a `--no-jit` scan compares neither the jit-* banner fields nor the two JIT columns (a
    skipped cell is no reading and equals nothing). It proves a move only: identical entries, rows and
    digests, said in the verdict's detail. Any other result is refused by name, since the add and the
    occupancy classes read the JIT columns."""
    if base.bin != new.bin:
        raise RecipeError(f"scan pair {base.path} {new.path}: bin {base.bin} against bin {new.bin} — a pair is two scans of one binary")
    nojit = base.nojit or new.nojit
    if nojit:
        same = tuple(k for k in SCAN_SAME if not k.startswith("jit-"))
    else:
        # A banner older than jit-cc names only the card: two such logs compare as before, by the card's
        # name; one of each is refused, since the old side's capability is unknown.
        olds = ("jit-cc" not in base.fields) + ("jit-cc" not in new.fields)
        if olds == 1:
            raise RecipeError(f"scan pair {base.path} {new.path}: one banner names no jit-cc (an older ptx-scan) — "
                              "rescan it with this tree's tools before comparing")
        same = SCAN_SAME if olds == 0 else tuple("jit-card" if k == "jit-cc" else k for k in SCAN_SAME)
    for k in same:
        if base.fields.get(k) != new.fields.get(k):
            raise RecipeError(f"scan pair {base.path} {new.path}: {k}={base.fields.get(k)} against {k}={new.fields.get(k)} — "
                              "rescan both under one toolchain and card capability before comparing")
    if base.header != new.header:
        raise RecipeError(f"scan pair {base.path} {new.path}: the tables have other columns")
    brows, nrows = _compared_rows(base, nojit), _compared_rows(new, nojit)
    if sorted(base.bundles) != sorted(new.bundles):
        v = ScanVerdict(base.bin, "moved", [], f"bundles {' '.join(base.bundles)} -> {' '.join(new.bundles)}", new.bundles)
    else:
        removed = sorted(set(brows) - set(nrows))
        changed = sorted(e for e in set(brows) & set(nrows) if brows[e] != nrows[e] or base.md5[e] != new.md5[e])
        added = sorted(set(nrows) - set(brows))
        if removed or changed:
            detail = "; ".join(x for x in (f"{len(removed)} removed" if removed else "", f"{len(changed)} changed" if changed else "") if x)
            v = ScanVerdict(base.bin, "moved", removed + changed, detail, new.bundles)
        else:
            v = ScanVerdict(base.bin, "added" if added else "identical", added, "", new.bundles)
    if not nojit:
        return v
    which = "both" if base.nojit and new.nojit else "base" if base.nojit else "new"
    if v.kind != "identical":
        raise RecipeError(f"scan pair {base.path} {new.path}: {v.line()} in a pair with a --no-jit scan ({which}) — "
                          "rescan without --no-jit: the add and occupancy classes need the JIT columns")
    v.detail = f"JIT skipped: --no-jit {which}"
    return v


@dataclass
class PathRow:
    line: int
    glob: str
    recipes: list[str]  # ["*"]: every gate
    why: str
    pattern: re.Pattern
    exceptions: dict[str, str] = field(default_factory=dict)  # gate the derivation says reaches a file of the row -> why it is left out

    def every(self) -> bool:
        return self.recipes == ["*"]

    def gates(self) -> list[str]:
        """The names that map a file for narrow(): the row's gate-* recipes, or `*`."""
        return [n for n in self.recipes if not n.startswith(WEEKLY_PREFIX)]

    def weeklies(self) -> list[str]:
        """The row's triggers: weekly-* recipes it names by a change of a file it matches."""
        return [n for n in self.recipes if n.startswith(WEEKLY_PREFIX)]


def glob_alternatives(glob: str) -> list[str]:
    """`a/{b,c/**}` as `a/b` and `a/c/**`; braces do not nest."""
    m = re.search(r"\{([^{}]*)\}", glob)
    if m is None:
        if re.search(r"[{}]", glob):
            raise RecipeError(f"{GATE_PATHS}: the glob {glob} has an unmatched brace")
        return [glob]
    return [x for alt in m.group(1).split(",") for x in glob_alternatives(glob[: m.start()] + alt + glob[m.end():])]


def glob_regex(glob: str) -> re.Pattern:
    """`**` any run of characters, `/` included; `*` any run without `/`; `?` one character but `/`;
    `{a,b}` either."""
    return re.compile("(?:" + "|".join(f"(?:{_glob_one(g)})" for g in glob_alternatives(glob)) + r")\Z")


def _glob_one(glob: str) -> str:
    out, i = [], 0
    while i < len(glob):
        if glob.startswith("**", i):
            out.append(".*")
            i += 2
        elif glob[i] == "*":
            out.append("[^/]*")
            i += 1
        elif glob[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(glob[i]))
            i += 1
    return "".join(out)


_EXCEPT_CLAUSE = re.compile(r"^except ((?:gate-[A-Za-z0-9_-]+ )*gate-[A-Za-z0-9_-]+): (\S.*)$")


def parse_exceptions(text: str) -> dict[str, str]:
    """A row's fourth column: `except GATE…: reason`, clauses joined by ` | `; gate -> reason. Anything else, or a
    gate named twice, is a named error."""
    out: dict[str, str] = {}
    for clause in text.split(" | "):
        m = _EXCEPT_CLAUSE.match(clause)
        if m is None:
            raise RecipeError(f"not `except gate-… gate-…: reason`: {clause[:80]!r}")
        for n in m.group(1).split(" "):
            if n in out:
                raise RecipeError(f"{n} named twice in the exceptions")
            out[n] = m.group(2)
    return out


def load_gate_paths(root: str = ROOT) -> list[PathRow]:
    """The table: `glob<TAB>recipes<TAB>why[<TAB>exceptions]`, recipes space-separated or the literal `*`, the
    exceptions as parse_exceptions reads them; `#` lines and blank lines skipped. Anything else is a named error."""
    path = os.path.join(root, GATE_PATHS)
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError as err:
        raise RecipeError(f"{GATE_PATHS}: {err.strerror}") from err
    rows: list[PathRow] = []
    for n, ln in enumerate(text.split("\n"), 1):
        if not ln.strip() or ln.startswith("#"):
            continue
        f = ln.split("\t")
        if len(f) not in (3, 4) or not all(x.strip() == x and x for x in f):
            raise RecipeError(f"{GATE_PATHS}:{n}: not glob<TAB>recipes<TAB>why[<TAB>exceptions] with non-empty fields: {ln[:120]!r}")
        names = f[1].split(" ")
        if "*" in names and names != ["*"]:
            raise RecipeError(f"{GATE_PATHS}:{n}: `*` stands alone — it means every gate")
        if len(set(names)) != len(names):
            raise RecipeError(f"{GATE_PATHS}:{n}: a recipe named twice")
        if f[0].startswith("/") or ".." in f[0].split("/"):
            raise RecipeError(f"{GATE_PATHS}:{n}: the glob {f[0]} is not a path relative to the tree")
        try:
            pattern = glob_regex(f[0])
            exceptions = parse_exceptions(f[3]) if len(f) == 4 else {}
        except RecipeError as err:
            raise RecipeError(f"{GATE_PATHS}:{n}: {err}") from err
        both = sorted(set(exceptions) & set(names))
        if both:
            raise RecipeError(f"{GATE_PATHS}:{n}: {both[0]} is both in the row and left out of it")
        if exceptions and names == ["*"]:
            raise RecipeError(f"{GATE_PATHS}:{n}: a `*` row leaves no gate out")
        rows.append(PathRow(n, f[0], names, f[2], pattern, exceptions))
    if not rows:
        raise RecipeError(f"{GATE_PATHS}: no rows")
    return rows


_OWN_LIB_TESTS = re.compile(r"^(?:libtest <- \S+ lib tests|doctest <- \S+ doctests)(?: \(path literal\))?$")


def _own_origin(why: str) -> bool:
    """Whether a hit's origin is the recipe's own target or its own text, not a dependency edge: a lib's
    own tests read the lib as their own tree (Graph.inputs labels them `libtest <- <pkg> lib tests`)."""
    inner = re.sub(r" \(via [^)]*\)$", "", why)
    if _OWN_LIB_TESTS.match(inner):
        return True
    if "<-" in inner or inner.startswith(("cargo", "tools/box.sh", "profile ", "dep-info")):
        return False
    return True


# ---- which module paths a target's own files name ----

def _use_paths(stmt: str) -> list[tuple[str, ...]]:
    """The paths of one `use` tree (`a::b::{c, d::{e as f}, self, *}`), each a tuple of segments."""
    return [p for p, _ in _use_bindings(stmt)]


def _use_bindings(stmt: str) -> list[tuple[tuple[str, ...], str]]:
    """(path, the name it binds) of each leaf of one `use` tree: the `as` alias, else the last segment (`self`
    binds its parent's); a glob leaf ends in `*` and binds `*`."""
    toks = re.findall(r"::|[{},*]|(?:r#)?[A-Za-z_][A-Za-z0-9_]*", stmt)

    def tree(i: int, prefix: tuple[str, ...]) -> tuple[list[tuple[tuple[str, ...], str]], int]:
        segs = list(prefix)
        alias = ""
        while i < len(toks):
            t = toks[i]
            if t == "{":
                out: list[tuple[tuple[str, ...], str]] = []
                i += 1
                while i < len(toks) and toks[i] != "}":
                    if toks[i] == ",":
                        i += 1
                        continue
                    sub, i = tree(i, tuple(segs))
                    out += sub
                return out, i + 1
            if t in (",", "}"):
                break
            if t == "as":
                alias = toks[i + 1] if i + 1 < len(toks) else ""
                i += 2
                continue
            if t != "::" and t != "self":
                segs.append(t[2:] if t.startswith("r#") else t)
            i += 1
        return [(tuple(segs), alias or (segs[-1] if segs else ""))], i

    return [(p, n) for p, n in tree(0, ())[0] if p]


_USE_STMT = re.compile(r"\buse\s+([^;]*);")
_PATH_EXPR = re.compile(r"(?<![A-Za-z0-9_:])((?:[A-Za-z_][A-Za-z0-9_]*\s*::\s*)+[A-Za-z_][A-Za-z0-9_]*)")


@functools.cache
def named_paths(text: str) -> frozenset[tuple[str, ...]]:
    """Every module path a Rust file's code names: its `use` trees expanded, and its path expressions.
    Comments and string contents do not count."""
    _, shape = rust_lex(text)
    out: set[tuple[str, ...]] = set()
    for m in _USE_STMT.finditer(shape):
        out.update(_use_paths(m.group(1)))
    for m in _PATH_EXPR.finditer(shape):
        out.add(tuple(re.split(r"\s*::\s*", m.group(1))))
    return frozenset(out)


def names_module(paths: frozenset[tuple[str, ...]], module: tuple[str, ...], children: frozenset[str] = frozenset()) -> bool:
    """A path at or under `module` that does not go on into one of `children` (a submodule with a file of its
    own, which a row for the parent's file alone does not cover), or a glob import of `module` or a module
    above it."""
    k = len(module)
    return any(
        (p[:k] == module and not (len(p) > k and p[k] in children)) or (p[-1] == "*" and module[: len(p) - 1] == p[:-1])
        for p in paths
    )


def module_of(tree: Tree, glob: str) -> tuple[str, tuple[str, ...], frozenset[str]] | None:
    """(package, module path below its lib, the submodules the glob leaves out) of a glob that names one lib
    module: `…/src/x.rs` or `…/src/x/mod.rs` (the module's own file: its submodules' files are left out),
    `…/src/x/**`, `…/src/x/*.rs`; None for any other glob (a crate root, a bin, a data file)."""
    for p in tree.packages.values():
        lib = p.lib()
        if lib is None:
            continue
        srcdir = os.path.dirname(lib.src) + "/"
        if not glob.startswith(srcdir):
            continue
        rest = glob[len(srcdir):]
        children: frozenset[str] = frozenset()
        for tail in ("/**", "/*.rs", "/*"):
            if rest.endswith(tail):
                rest = rest[: -len(tail)]
                break
        else:
            if not rest.endswith(".rs"):
                return None
            rest = rest[:-3]
            if rest.endswith("/mod"):
                rest = rest[:-4]
            sub = os.path.join(tree.root, srcdir, rest)
            if os.path.isdir(sub):
                children = frozenset(
                    e[:-3] if e.endswith(".rs") else e
                    for e in os.listdir(sub)
                    if e != "mod.rs" and (e.endswith(".rs") or os.path.isdir(os.path.join(sub, e)))
                )
        if not rest or rest == "lib" or re.search(r"[*?]", rest):
            return None
        return p.name, tuple(rest.split("/")), children
    return None


def crate_idents(tree: Tree, pkg: str) -> set[str]:
    """The names a dependent's code calls package `pkg` by: its lib's name and every rename of it."""
    lib = tree.packages[pkg].lib()
    out = {lib.name.replace("-", "_")} if lib is not None else set()
    for p in tree.packages.values():
        out |= {d.key.replace("-", "_") for d in p.deps if d.pkg == pkg and d.key != d.pkg}
    return out


# ---- which gates reach a file: the item graph across the workspace ----

_UNIT_HEAD = re.compile(
    r"(?:pub\s*(?:\([^)]*\))?\s*)?(?:(?:unsafe|async|const|default|safe)\s+|extern\s+(?:\"[^\"]*\"\s*)?)*"
    r"(fn|struct|enum|union|trait|impl|mod|type|const|static|use|macro_rules\s*!|extern\s+crate)(?![A-Za-z0-9_])\s*(?:r#)?([A-Za-z_][A-Za-z0-9_]*)?"
)
_SEMI_KINDS = frozenset({"use", "const", "static", "type", "extern crate"})
_DOC_RUST = frozenset({"rust", "compile_fail", "should_panic", "no_run", "allow_fail", "test_harness"})
_ITEM_DEF = re.compile(r"\b(?:fn|struct|enum|union|trait|type|const|static)\s+(?:r#)?([A-Za-z_][A-Za-z0-9_]*)")
_FN_NAME = re.compile(r"\bfn\s+(?:r#)?([A-Za-z_][A-Za-z0-9_]*)")
_MACRO_EXPORT = re.compile(r"#\s*\[\s*macro_export\b[^\]]*\]\s*macro_rules\s*!\s*(?:r#)?([A-Za-z_][A-Za-z0-9_]*)")
_MACRO_CALL = re.compile(r"(?<![A-Za-z0-9_:])([A-Za-z_][A-Za-z0-9_]*)\s*!\s*[(\[{]")
_WORD = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z_][A-Za-z0-9_]*")
_PUB_BEFORE = re.compile(r"\bpub\b(?:\s*\([^)]*\))?\s*$")
_TYPE_PATH = re.compile(r"(?<![A-Za-z0-9_'])((?:[A-Za-z_][A-Za-z0-9_]*\s*::\s*)*[A-Za-z_][A-Za-z0-9_]*)")
_TYPE_HEAD = re.compile(r"\s*!?\s*(?:&\s*(?:'[A-Za-z_][A-Za-z0-9_]*\s*)?(?:mut\s+)?)?(?:dyn\s+)?((?:[A-Za-z_][A-Za-z0-9_]*\s*::\s*)*[A-Za-z_][A-Za-z0-9_]*)")
_NOT_A_TYPE = frozenset({"dyn", "mut", "const", "for", "impl", "where", "Self", "self", "crate", "super", "unsafe", "extern", "fn", "as"})


@dataclass(frozen=True)
class ImplHead:
    """One `impl` block's header: its generic parameters, the self type's head path and the paths in its
    arguments, the trait's head path and the paths in its arguments, as written."""

    params: frozenset[str]
    self_head: tuple[str, ...] | None
    self_args: tuple[tuple[str, ...], ...]
    trait: tuple[str, ...] | None
    trait_args: tuple[tuple[str, ...], ...]
    text: str


@dataclass(frozen=True)
class Unit:
    """One item of a module (an inline module's items are units of their own, `ctx` below the file's module), or
    one associated item of an impl block (`assoc`, with the block's header in `impl`; an impl with no items is one
    unit named None): the paths, identifiers and macros its code names, the header's included. `kind` is the
    item's keyword, `anon` for an item-level macro call or anything else no keyword starts; `names` are the items an
    `anon` unit's text declares (`thread_local! { static X … }`), `methods` a trait's fn names."""

    kind: str
    ctx: tuple[str, ...]
    name: str | None
    line: int
    sites: tuple[tuple[str, ...], ...]
    idents: frozenset[str]
    macro_calls: frozenset[str]
    includes: tuple[str, ...]
    literals: tuple[str, ...]
    impl: ImplHead | None
    names: frozenset[str] = frozenset()
    methods: frozenset[str] = frozenset()
    test: bool = False  # under a test attribute (`#[test]`, `#[rstest]`, `#[test_case]`): an entry of a test harness


class RustFacts:
    """What the reach reads from one file: its units, its `use` leaves (ctx, path, bound name, pub; a glob ends in
    `*`), and its `#[macro_export]` macros (items of the crate root)."""

    def __init__(self, units: list[Unit], binds: list[tuple[tuple[str, ...], tuple[str, ...], str, bool]], exported: frozenset[str]):
        # the last unit stands for the file's `use` declarations: a path that passes through one of its re-exports
        # reaches it, since a change of the leaf changes the item the path names
        self.uses = len(units)
        self.units = tuple(units) + (Unit("use", (), None, 0, (), frozenset(), frozenset(), (), (), None),)
        self.binds = tuple(binds)
        self.scoped: dict[tuple[tuple[str, ...], str], list[tuple[str, ...]]] = {}
        for c, p, n, _ in self.binds:
            if p[-1] != "*":
                self.scoped.setdefault((c, n), []).append(p)
        self.exported = exported
        self.named: dict[tuple[tuple[str, ...], str], list[int]] = {}
        self.macro_named: dict[str, list[int]] = {}
        for i, u in enumerate(self.units):
            for name in ([u.name] if u.name is not None and u.kind != "assoc" else []) + sorted(u.names):
                self.named.setdefault((u.ctx, name), []).append(i)
            if u.kind == "macro_rules!" and u.name is not None:
                self.macro_named.setdefault(u.name, []).append(i)
        self.bound = frozenset(n for _, p, n, _ in self.binds if p[-1] != "*")
        self.globs = tuple((c, p) for c, p, _, _ in self.binds if p[-1] == "*")
        self.anon = tuple(i for i, u in enumerate(self.units) if u.kind == "anon")
        self.traits = tuple(i for i, u in enumerate(self.units) if u.kind == "trait")


def _angle_close(text: str, at: int) -> int | None:
    """The index of the `>` closing the `<` at `at`, every bracket kind counted (`->` and `=>` are not brackets)."""
    depth = 0
    for j in range(at, len(text)):
        c = text[j]
        if c in "<([{":
            depth += 1
        elif c in ")]}" or (c == ">" and text[j - 1] not in "-="):
            depth -= 1
        if depth == 0:
            return j
    return None


def _split_angles(text: str) -> list[str]:
    """`text` split at its commas outside every bracket kind, `<…>` included."""
    out, depth, start = [], 0, 0
    for j, c in enumerate(text):
        if c in "<([{":
            depth += 1
        elif c in ")]}" or (c == ">" and j and text[j - 1] not in "-="):
            depth -= 1
        elif c == "," and depth == 0:
            out.append(text[start:j])
            start = j + 1
    out.append(text[start:])
    return [x.strip() for x in out if x.strip()]


def _type_names(text: str, params: frozenset[str]) -> tuple[tuple[str, ...] | None, tuple[tuple[str, ...], ...]]:
    """(head path, the other paths) of a type or trait as an impl header writes it; a generic parameter, a lifetime
    or a keyword is no path, and a head that is one leaves the head None."""

    def split(p: str) -> tuple[str, ...]:
        return tuple(re.split(r"\s*::\s*", p))

    def path(p: tuple[str, ...]) -> bool:
        # `crate::`, `self::` and `super::` open a path; alone, they and the other keywords name no type
        if len(p) > 1 and p[0] in ("crate", "self", "super"):
            return True
        return p[0] not in _NOT_A_TYPE and not (len(p) == 1 and p[0] in params)

    paths = [p for p in (split(m.group(1)) for m in _TYPE_PATH.finditer(text)) if path(p)]
    head = None
    m = _TYPE_HEAD.match(text)
    if m and path(split(m.group(1))):
        head = split(m.group(1))
    rest, dropped = [], False
    for p in paths:
        if p == head and not dropped:
            dropped = True
            continue
        rest.append(p)
    return head, tuple(rest)


def _impl_head(shape: str, i: int, end: int) -> ImplHead:
    """The header of the impl item whose `impl` keyword ends at `i` (the item ends at `end`)."""
    while i < end and shape[i].isspace():
        i += 1
    params: set[str] = set()
    if i < end and shape[i] == "<":
        close = _angle_close(shape, i)
        if close is not None:
            for item in _split_angles(shape[i + 1 : close]):
                if item.startswith("'"):
                    continue
                w = re.match(r"(?:const\s+)?([A-Za-z_][A-Za-z0-9_]*)", item)
                if w:
                    params.add(w.group(1))
            i = close + 1
    stop, depth = i, 0
    while stop < end:
        c = shape[stop]
        if c in "<([":
            depth += 1
        elif c in ")]" or (c == ">" and shape[stop - 1] not in "-="):
            depth -= 1
        elif depth == 0 and (c in "{;" or (shape.startswith("where", stop) and not _IDENT_CH.match(shape[stop - 1]) and not _IDENT_CH.match(shape[stop + 5 : stop + 6] or " "))):
            break
        stop += 1
    hdr = shape[i:stop]
    cut, depth = None, 0
    for k, c in enumerate(hdr):
        if c in "<([":
            depth += 1
        elif c in ")]" or (c == ">" and k and hdr[k - 1] not in "-="):
            depth -= 1
        elif depth == 0 and hdr.startswith("for", k) and (k == 0 or not _IDENT_CH.match(hdr[k - 1])) and not _IDENT_CH.match(hdr[k + 3 : k + 4] or " "):
            if not hdr[k + 3 :].lstrip().startswith("<"):
                cut = k
                break
    p = frozenset(params)
    if cut is None:
        th, ta = None, ()
        sh, sa = _type_names(hdr, p)
    else:
        th, ta = _type_names(hdr[:cut], p)
        sh, sa = _type_names(hdr[cut + 3 :], p)
    return ImplHead(p, sh, sa, th, ta, " ".join(("impl " + hdr).split())[:96])


def _unit_end(shape: str, i: int, end: int, semi: bool) -> int:
    """Where the item starting at `i` ends: after its `;` at its own depth, or (unless `semi`: a use, const,
    static or type, which a `;` ends) after the brace block opened at its own depth."""
    depth = 0
    for j in range(i, end):
        c = shape[j]
        if c == "{" and depth == 0 and not semi:
            close = _close(shape, j)
            return end if close is None else min(close + 1, end)
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == ";" and depth == 0:
            return j + 1
    return end


def _module_units(shape: str, a: int, b: int, ctx: tuple[str, ...], out: list, spans: list) -> None:
    """The items of the module region [a, b) of a file's shape: (kind, ctx, name, start, end, impl header, under a
    test attribute) into `out`, each inline module's region (start, end, ctx) into `spans` and its items under its
    name."""
    i = a
    test = False
    while i < b:
        while i < b and shape[i].isspace():
            i += 1
        if i >= b:
            break
        c = shape[i]
        if c == ";":
            i += 1
            continue
        if c == "#":
            j = i + 1
            while j < b and shape[j] in " !\t\n":
                j += 1
            close = _close(shape, j) if j < b and shape[j] == "[" else None
            test = test or (close is not None and _TEST_ATTR.match(shape, i, close + 1) is not None)
            i = (close + 1) if close is not None else i + 1
            continue
        start = len(out)
        m = _UNIT_HEAD.match(shape, i, b)
        if m is None:
            end = max(_unit_end(shape, i, b, False), i + 1)
            out.append(("anon", ctx, None, i, end, None, test))
            i, test = end, False
            continue
        kind = re.sub(r"\s+", " ", m.group(1)).replace(" !", "!")
        name = m.group(2)
        if kind == "mod" and name is not None:
            j = m.end()
            while j < b and shape[j].isspace():
                j += 1
            if j < b and shape[j] == "{":
                close = _close(shape, j)
                close = b - 1 if close is None else close
                spans.append((j, close, ctx + (name,)))
                _module_units(shape, j + 1, close, ctx + (name,), out, spans)
                i = close + 1
            else:
                i = _unit_end(shape, i, b, True)
            test = False
            continue
        if kind == "impl":
            j, depth, open_at = m.end(1), 0, None
            while j < b:
                c = shape[j]
                if c == "{" and depth == 0:
                    open_at = j
                    break
                if c in "([":
                    depth += 1
                elif c in ")]":
                    depth -= 1
                elif c == ";" and depth == 0:
                    break
                j += 1
            if open_at is None:
                i = j + 1
                continue
            close = _close(shape, open_at)
            close = b - 1 if close is None else close
            hdr = (m.end(1), i, open_at)
            _assoc_units(shape, open_at + 1, close, ctx, hdr, out)
            if len(out) == start:
                out.append(("assoc", ctx, None, i, close + 1, hdr, False))
            i, test = close + 1, False
            continue
        end = _unit_end(shape, i, b, kind in _SEMI_KINDS)
        if kind != "use":
            out.append((kind, ctx, name if kind != "mod" and name != "_" else None, i, end, None, test))
        i, test = max(end, i + 1), False


def _assoc_units(shape: str, a: int, b: int, ctx: tuple[str, ...], hdr: tuple[int, int, int], out: list) -> None:
    """The associated items of the impl body [a, b): ("assoc", ctx, name, start, end, hdr) each; a macro call there
    has no name."""
    i = a
    while i < b:
        while i < b and shape[i].isspace():
            i += 1
        if i >= b:
            break
        if shape[i] == ";":
            i += 1
            continue
        if shape[i] == "#":
            j = i + 1
            while j < b and shape[j] in " !\t\n":
                j += 1
            close = _close(shape, j) if j < b and shape[j] == "[" else None
            i = (close + 1) if close is not None else i + 1
            continue
        m = _UNIT_HEAD.match(shape, i, b)
        semi = m is not None and re.sub(r"\s+", " ", m.group(1)) in _SEMI_KINDS
        end = max(_unit_end(shape, i, b, semi), i + 1)
        out.append(("assoc", ctx, m.group(2) if m is not None and m.group(2) != "_" else None, i, end, hdr, False))
        i = end


@functools.cache
def rust_facts(text: str, test: bool) -> RustFacts:
    """The facts the reach reads from one file, as a build with cfg(test) = `test` compiles it (build_view)."""
    code, shape = build_view(text, test)
    raw: list = []
    spans: list = []
    _module_units(shape, 0, len(shape), (), raw, spans)

    def ctx(pos: int) -> tuple[str, ...]:
        best: tuple[str, ...] = ()
        for a, b, c in spans:
            if a <= pos < b and len(c) > len(best):
                best = c
        return best

    binds = []
    body = list(shape)
    for m in _USE_STMT.finditer(shape):
        pub = bool(_PUB_BEFORE.search(shape, max(0, m.start() - 48), m.start()))
        c = ctx(m.start())
        for path, name in _use_bindings(m.group(1)):
            binds.append((c, path, name, pub))
        _blank(body, m.start(), m.end())
    rest = "".join(body)
    starts = [r[3] for r in raw]

    def owner(pos: int) -> int | None:
        k = bisect.bisect_right(starts, pos) - 1
        return k if k >= 0 and raw[k][3] <= pos < raw[k][4] else None

    sites: list[list] = [[] for _ in raw]
    calls: list[set] = [set() for _ in raw]
    incs: list[list] = [[] for _ in raw]
    lits: list[list] = [[] for _ in raw]
    for m in _PATH_EXPR.finditer(rest):
        k = owner(m.start())
        if k is not None:
            sites[k].append(tuple(re.split(r"\s*::\s*", m.group(1))))
    for m in _MACRO_CALL.finditer(rest):
        k = owner(m.start())
        if k is not None and m.group(1) != "macro_rules":
            calls[k].add(m.group(1))
    for pat, acc in ((Tree._INCLUDE, incs), (Tree._LITERAL, lits)):
        for m in pat.finditer(code):
            k = owner(m.start())
            if k is not None:
                acc[k].append(m.group(1))
    heads: dict[tuple[int, int, int], tuple[ImplHead, tuple, frozenset[str]]] = {}
    units = []
    for k, (kind, c, name, a, b, hdr, test) in enumerate(raw):
        hsites: tuple = ()
        hidents: frozenset[str] = frozenset()
        head = None
        if hdr is not None:
            if hdr not in heads:
                kw_end, h0, h1 = hdr
                heads[hdr] = (_impl_head(shape, kw_end, h1),
                              tuple(tuple(re.split(r"\s*::\s*", x.group(1))) for x in _PATH_EXPR.finditer(rest, h0, h1)),
                              frozenset(_WORD.findall(rest, h0, h1)))
            head, hsites, hidents = heads[hdr]
        text = shape[a:b]
        units.append(Unit(kind, c, name, shape.count("\n", 0, a) + 1, tuple(sites[k]) + hsites,
                          frozenset(_WORD.findall(rest, a, b)) | hidents, frozenset(calls[k]), tuple(incs[k]), tuple(lits[k]), head,
                          frozenset(_ITEM_DEF.findall(text)) if kind == "anon" else frozenset(),
                          frozenset(_FN_NAME.findall(text)) if kind == "trait" else frozenset(), test))
    return RustFacts(units, binds, frozenset(_MACRO_EXPORT.findall(shape)))


RootKey = tuple[str, str, bool]  # (package, target root file, cfg(test))
Hit = tuple[RootKey, str, int]  # (the root a file is compiled in, the file, the unit's index; -1: a data file)


class RuleSet:
    """Rule 1's associated items and their indexes: by a unit their header names, by the name a reached unit must
    say, and the items that need neither."""

    def __init__(self, rules: list[tuple[int, str, str, list[list[int]], str | None]]):
        self.rules = rules
        self.by_node: dict[int, list[int]] = {}
        self.by_name: dict[str, list[int]] = {}
        for i, (_, _, _, groups, name) in enumerate(rules):
            for g in groups:
                for x in g:
                    self.by_node.setdefault(x, []).append(i)
            if name is not None:
                self.by_name.setdefault(name, []).append(i)
        self.free = [i for i, r in enumerate(rules) if not r[3] and r[4] is None]
        self.names = frozenset(self.by_name)


class Reach:
    """Which gates' targets can run code of a file: the item graph across the workspace, closed from each gate
    target's own items. A unit is one item of a module (a fn, a type, a trait, a const or static, a macro, an
    item-level macro call) or one associated item of an impl block. A gate reaches file F when one of its targets'
    own units reaches a unit of F by a chain of these edges, every file on it linked by the target (Graph.inputs,
    feature-aware at crate grain):

    - a path. A unit's code names a path (a path expression `a::b::c`, a `name!` macro by path, an identifier a
      `use` leaf of its file binds, an identifier its own module defines) that resolves to item N of module M: N is
      reached when M defines it; when M only imports it (a `use`, a re-export), the import is followed instead and
      M is a namespace on the way. A path's first segment resolves as rustc's: `crate`, `self`, `super`, a child
      module, a name a `use` leaf of the file binds, a workspace crate's name (crate_idents); any other (std, an
      external crate, a local variable, a generic) names nothing of the workspace. A name M neither defines nor
      imports by name (an item a macro writes) reaches every unit of M. A path that ends at a module is a
      namespace, and a `mod` declaration no edge: declaring a module runs none of its code.
    - a method. Method syntax names no path but always the method's name: a trait in scope in a unit's file (a
      private `use` of it, its own definition, a glob-imported module's) is reached by a unit that names one of the
      trait's fns, and an associated item by the rule for impls below.
    - an item-level macro call (`thread_local!`, a macro that writes fns) is reached with every unit of its file,
      and answers to the names its text declares.

    The six cases the table's rule has to decide:
    1. Trait and inherent impls. An associated item is reached when its impl's self type is reached (a value of the
       type exists only where some reached code names the type) and, when that type is generic over concrete
       workspace types (`impl GpuModel<Body>`), each of those types too (the impl runs only on that instantiation:
       the qwen3moe row's stated form), and some reached unit names the item (`x.m()`, `T::m`, `Self::C`) — except
       in an impl of a trait outside the workspace (Drop, Display, Deref, From, an operator), whose items std calls
       without naming them. An impl for a generic parameter or a foreign type (a blanket impl) is reached with its
       trait the same way, and one with neither side in the workspace whenever its file is linked. A trait's
       default methods are in the trait's unit, which the method rule above reaches.
    2. Generics. A generic fn runs in the binary that monomorphizes it, and its caller names it: the path edge
       covers it, and nothing is added for monomorphization.
    3. `#[cfg(test)]`. Items and modules under cfg(test) exist only in their own lib's test build: a dependent's
       view of a lib blanks them (build_view), and only the lib's own libtest target reads them.
    4. Constants, statics and types. Naming one reaches its unit even when the value folds into the caller and no
       code of it runs: a change of it changes the caller.
    5. Macros. A `name!` call reaches the `macro_rules! name` units of the same root (textual scope, `#[macro_use]`
       modules included); a `#[macro_export]` macro is an item of its crate's root, so a path to it or a `use` of
       it reaches it; the macro's body is that unit's code, and its `$crate::` paths resolve in its crate.
    6. Run-time dispatch by value (an `Arch` match). Every arm's path is named, so every arm is reached: the static
       reach over-selects there, and a row that leaves such a gate out says so as a named exception
       (gate_paths_problems).

    The seeds: a test harness (a lib's own tests, a test file) runs its `#[test]` fns, and its own impls follow rule 1
    in its own root; any other target (a bin, `cargo test --bin` too) and a test file with no `#[test]` fn is
    reached whole (every unit of its own module tree); a doctest run reaches what its examples name; a build script
    is reached by every gate that reaches its crate's code. Unclear cases resolve wide: a name a glob import could
    bring takes every candidate, a file reached in two module contexts both, a name read in a comment-free code text
    whether or not it is the method's receiver."""

    def __init__(self, tree: Tree):
        self.tree = tree
        self.idents: dict[str, set[str]] = {}
        self.lib_root: dict[str, RootKey] = {}
        for p in tree.packages.values():
            lib = p.lib()
            if lib is None:
                continue
            self.lib_root[p.name] = (p.name, lib.src, False)
            for ident in crate_idents(tree, p.name):
                self.idents.setdefault(ident, set()).add(p.name)
        self.lib_roots = set(self.lib_root.values())
        self._mods: dict[RootKey, dict[tuple[str, ...], str]] = {}
        self._bases: dict[RootKey, dict[str, list[tuple[str, ...]]]] = {}
        self._macros: dict[RootKey, dict[str, list[tuple[str, int]]]] = {}
        self._exported: dict[str, dict[str, list[tuple[str, int]]]] = {}
        self._text: dict[str, str] = {}
        self._resolved: dict[tuple, list] = {}
        self._active: set[tuple] = set()
        self._cut = 0
        # the item graph, numbered as it is discovered
        self.ids: dict[Hit, int] = {}
        self.nodes: list[Hit] = []
        self.adj: list[list[tuple[int, str]] | None] = []
        self._gates: dict[tuple[str, str], dict[int, tuple[int, str]]] = {}
        self._gate_files: dict[tuple[str, str], dict[str, int]] = {}
        self.file_root: dict[str, RootKey] = {}
        for rk in self.lib_root.values():
            for f in self.mods(rk).values():
                self.file_root.setdefault(f, rk)
        self.builds = {t.src: p.name for p in tree.packages.values() for t in p.targets if t.kind == "build"}
        self._traits: dict[tuple, list[tuple[Hit, frozenset[str], str]]] = {}
        self._rules: RuleSet | None = None
        self._own_rules: dict[RootKey, RuleSet] = {}

    def text(self, f: str) -> str:
        if f not in self._text:
            self._text[f] = self.tree.read(f)
        return self._text[f]

    def facts(self, rk: RootKey, f: str) -> RustFacts:
        return rust_facts(self.text(f), rk[2])

    def mods(self, rk: RootKey) -> dict[tuple[str, ...], str]:
        if rk not in self._mods:
            self._mods[rk] = self.tree.mods_of(rk[1], rk[2])
        return self._mods[rk]

    def bases(self, rk: RootKey, f: str) -> list[tuple[str, ...]]:
        """The module paths whose file module `f` is in root `rk` (an inline module of `f` is below one of them)."""
        if rk not in self._bases:
            mods = self.mods(rk)
            by: dict[str, list[tuple[str, ...]]] = {}
            for m, g in mods.items():
                if not m or mods.get(m[:-1]) != g:
                    by.setdefault(g, []).append(m)
            self._bases[rk] = by
        return self._bases[rk].get(f, [])

    def node(self, h: Hit) -> int:
        i = self.ids.get(h)
        if i is None:
            i = self.ids[h] = len(self.nodes)
            self.nodes.append(h)
            self.adj.append(None)
        return i

    def doctests(self, pkg: str) -> str:
        """The pseudo file of package `pkg`'s doctests: every Rust code block of its lib's `///` and `//!` comments
        (hidden `# ` lines included), each as the body of a fn, so a doctest run reaches what its examples name."""
        path = f"{self.tree.packages[pkg].dir}/<doctests>.rs"
        if path not in self._text:
            blocks: list[str] = []
            rk = self.lib_root.get(pkg)
            for f in sorted(set(self.mods(rk).values())) if rk is not None else []:
                cur: list[str] | None = None
                for line in self.text(f).split("\n"):
                    m = re.match(r"\s*//[/!] ?(.*)$", line)
                    if m is None:
                        cur = None
                        continue
                    body = m.group(1)
                    fence = re.match(r"\s*```\s*(.*)$", body)
                    if fence is not None:
                        if cur is None:
                            info = {t.strip() for t in fence.group(1).split(",") if t.strip()}
                            rust = all(t in _DOC_RUST or re.match(r"^(?:edition\d+|E\d+)$", t) for t in info)
                            cur = [] if rust else ["<skip>"]
                        else:
                            if cur[:1] != ["<skip>"]:
                                blocks.append("fn __doctest() {\n" + "\n".join(cur) + "\n}\n")
                            cur = None
                        continue
                    if cur is not None:
                        cur.append(re.sub(r"^\s*# ?", "", body) if re.match(r"^\s*#(?: |$)", body) else body)
            self._text[path] = "".join(blocks)
        return path

    def units(self, rk: RootKey, f: str) -> list[Hit]:
        """Every unit of file `f` in root `rk`; a file with no Rust items (a data file, an empty module) as -1."""
        if not f.endswith(".rs"):
            return [(rk, f, -1)]
        n = len(self.facts(rk, f).units)
        return [(rk, f, i) for i in range(n)] or [(rk, f, -1)]

    def macros(self, rk: RootKey) -> dict[str, list[tuple[str, int]]]:
        """`macro_rules!` name -> its (file, unit) in root `rk`."""
        if rk not in self._macros:
            out: dict[str, list[tuple[str, int]]] = {}
            for f in sorted(set(self.mods(rk).values())):
                for name, idx in self.facts(rk, f).macro_named.items():
                    out.setdefault(name, []).extend((f, i) for i in idx)
            self._macros[rk] = out
        return self._macros[rk]

    def exported(self, pkg: str) -> dict[str, list[tuple[str, int]]]:
        """`#[macro_export]` name -> its (file, unit) in package `pkg`'s lib: items of its crate root."""
        if pkg not in self._exported:
            out: dict[str, list[tuple[str, int]]] = {}
            rk = self.lib_root.get(pkg)
            if rk is not None:
                for name, at in self.macros(rk).items():
                    out[name] = [(f, i) for f, i in at if name in self.facts(rk, f).exported]
            self._exported[pkg] = out
        return self._exported[pkg]

    # -- paths --

    def _once(self, key: tuple, fn) -> list:
        """`fn()` memoized by `key`; a key met again inside its own computation (a cycle of imports) gives nothing
        there, and a result computed under such a cut is not kept."""
        if key in self._resolved:
            return self._resolved[key]
        if key in self._active:
            self._cut += 1
            return []
        self._active.add(key)
        cut = self._cut
        try:
            out = fn()
        finally:
            self._active.discard(key)
        if self._cut == cut:
            self._resolved[key] = out
        return out

    def resolve(self, rk: RootKey, mod: tuple[str, ...], path: tuple[str, ...], strict: bool = False, skip: str = "") -> list[Hit]:
        """The units `path`, named in module `mod` of root `rk`, reaches. `strict` (a name looked up through a glob
        import): only what the module defines or imports by name, never every item of a module for a name it does
        not account for. `skip`: a `use` leaf's own name, which its path's first segment never means (`use models;`,
        `use foo::foo;`)."""
        if not path or len(path) > 32:
            return []
        return self._once(("path", rk, mod, path, strict, skip), lambda: self._resolve(rk, mod, path, strict, skip))

    def _resolve(self, rk: RootKey, mod: tuple[str, ...], path: tuple[str, ...], strict: bool, skip: str) -> list[Hit]:
        s0 = path[0]
        if s0 == "crate":
            return self.walk(rk, (), path[1:], strict)
        if s0 == "self":
            return self.walk(rk, mod, path[1:], strict)
        if s0 == "super":
            i, base = 0, mod
            while i < len(path) and path[i] == "super":
                if not base:
                    return []
                base, i = base[:-1], i + 1
            return self.walk(rk, base, path[i:], strict)
        if s0 == "Self":
            return []
        mods = self.mods(rk)
        if mod not in mods:
            return []
        # a name in scope (a child module, a `use` leaf) shadows a glob import's, and both an extern crate's
        out: list[Hit] = []
        child = mod + (s0,) in mods
        if child:
            out += self.walk(rk, mod, path, strict)
        fm = mods[mod]
        facts = self.facts(rk, fm)
        # the `use` leaves of this module (a leaf of another module of the file binds nothing here)
        base = max((b for b in self.bases(rk, fm) if mod[: len(b)] == b), key=len, default=mod)
        leaves = facts.scoped.get((mod[len(base):], s0), []) if s0 != skip else []
        for bp in leaves:
            out += self.resolve(rk, mod, bp + path[1:], strict, s0 if bp[0] == s0 else "")
        if child or leaves:
            return out
        for c, bp in facts.globs:
            if c == mod[len(base):]:
                for grk, gmod in self.module(rk, mod, bp[:-1]):
                    out += self.walk(grk, gmod, path, True)
        for pkg in sorted(self.idents.get(s0, ())):
            out += self.walk(self.lib_root[pkg], (), path[1:], strict)
        return out

    def walk(self, rk: RootKey, mod: tuple[str, ...], rest: tuple[str, ...], strict: bool) -> list[Hit]:
        """Down the module path from `mod` while `rest` names child modules, then the item it names. A path that
        ends at a module is a namespace: no unit."""
        mods = self.mods(rk)
        if mod not in mods:
            return []
        for i, s in enumerate(rest):
            if mod + (s,) in mods:
                mod = mod + (s,)
                continue
            return self.item(rk, mod, s, rest[i + 1 :], strict)
        return []

    def item(self, rk: RootKey, mod: tuple[str, ...], name: str, rest: tuple[str, ...], strict: bool) -> list[Hit]:
        return self._once(("item", rk, mod, name, rest, strict), lambda: self._item(rk, mod, name, rest, strict))

    def _item(self, rk: RootKey, mod: tuple[str, ...], name: str, rest: tuple[str, ...], strict: bool) -> list[Hit]:
        fm = self.mods(rk)[mod]
        facts = self.facts(rk, fm)
        bases = [b for b in self.bases(rk, fm) if mod[: len(b)] == b] or [mod]
        ctx = mod[len(max(bases, key=len)) :]
        own = facts.named.get((ctx, name))
        if own:
            return [(rk, fm, i) for i in own]
        if not mod and rk in self.lib_roots and name in self.exported(rk[0]):
            return [(rk, g, i) for g, i in self.exported(rk[0])[name]]
        out: list[Hit] = []
        for bp in facts.scoped.get((ctx, name), []):
            out += self.resolve(rk, mod, bp + rest, strict, name if bp[0] == name else "")
        if out:
            return out + [(rk, fm, facts.uses)]
        for c, bp in facts.globs:
            if c == ctx:
                for grk, gmod in self.module(rk, mod, bp[:-1]):
                    out += self.walk(grk, gmod, (name,) + rest, True)
        if out:
            return out + [(rk, fm, facts.uses)]
        if strict:
            return out
        return [(rk, fm, i) for i, u in enumerate(facts.units) if u.ctx == ctx] or [(rk, fm, -1)]

    def module(self, rk: RootKey, mod: tuple[str, ...], path: tuple[str, ...]) -> list[tuple[RootKey, tuple[str, ...]]]:
        """The module a glob import's path names, from module `mod`: (root, module path) per candidate; none for a
        glob of an enum's variants or of an external crate's module."""
        if not path:
            return []
        return self._once(("module", rk, mod, path), lambda: self._module(rk, mod, path))

    def _module(self, rk: RootKey, mod: tuple[str, ...], path: tuple[str, ...]) -> list[tuple[RootKey, tuple[str, ...]]]:
        mods = self.mods(rk)
        s0 = path[0]
        if s0 in ("crate", "self", "super"):
            base, i = ((), 1) if s0 == "crate" else (mod, 1 if s0 == "self" else 0)
            while i < len(path) and path[i] == "super":
                if not base:
                    return []
                base, i = base[:-1], i + 1
            target = base + path[i:]
            return [(rk, target)] if target in mods else []
        out: list[tuple[RootKey, tuple[str, ...]]] = []
        if mod + path in mods:
            out.append((rk, mod + path))
        fm = mods.get(mod)
        leaves: list[tuple[str, ...]] = []
        if fm is not None and not out:
            base = max((b for b in self.bases(rk, fm) if mod[: len(b)] == b), key=len, default=mod)
            leaves = [bp for bp in self.facts(rk, fm).scoped.get((mod[len(base):], s0), []) if bp[0] != s0]
            for bp in leaves:
                out += self.module(rk, mod, bp + path[1:])
        if not out and not leaves and (mod + (s0,) not in mods):
            for pkg in sorted(self.idents.get(s0, ())):
                lrk = self.lib_root[pkg]
                if path[1:] in self.mods(lrk):
                    out.append((lrk, path[1:]))
        return out

    # -- edges --

    def traits(self, rk: RootKey, f: str) -> list[tuple[Hit, frozenset[str], str]]:
        """The traits in scope in file `f` by its own text: the ones its `use` leaves import and the ones it defines,
        each (unit, its fn names, how)."""
        key = (rk, f)
        if key in self._traits:
            return self._traits[key]
        out: list[tuple[Hit, frozenset[str], str]] = []
        facts = self.facts(rk, f)
        for b in self.bases(rk, f) or [()]:
            for c, bp, name, pub in facts.binds:
                if bp[-1] == "*":
                    continue
                for h in self.resolve(rk, b + c, bp):
                    if h[2] >= 0:
                        u = self.facts(h[0], h[1]).units[h[2]]
                        if u.kind == "trait":
                            out.append((h, u.methods, f"use {'::'.join(bp)}"))
        out += [((rk, f, i), facts.units[i].methods, "a trait of the file") for i in facts.traits]
        self._traits[key] = out
        return out

    def in_scope(self, rk: RootKey, f: str) -> list[tuple[Hit, frozenset[str], str]]:
        """`traits` of `f`, and those of each other file a glob import of `f` names a module of."""
        key = ("scope", rk, f)
        if key in self._traits:
            return self._traits[key]
        out = list(self.traits(rk, f))
        facts = self.facts(rk, f)
        for b in self.bases(rk, f) or [()]:
            for c, bp in facts.globs:
                for grk, gmod in self.module(rk, b + c, bp[:-1]):
                    g = self.mods(grk)[gmod]
                    if g != f:
                        out += [(h, ms, f"{how} (glob use {'::'.join(bp)})") for h, ms, how in self.traits(grk, g)]
        self._traits[key] = out
        return out

    def edges(self, n: int) -> list[tuple[int, str]]:
        """The units node `n` reaches, with each step's label."""
        got = self.adj[n]
        if got is not None:
            return got
        rk, f, k = self.nodes[n]
        hits: list[tuple[Hit, str]] = []
        if k >= 0:
            facts = self.facts(rk, f)
            u = facts.units[k]
            sites, idents = list(u.sites), set(u.idents)
            # a macro_rules body is unhygienic for items: its names resolve where it expands, here too
            seen_macros: set[Hit] = set()
            todo = [h for name in u.macro_calls for h in ((rk, g, i) for g, i in self.macros(rk).get(name, ()))]
            for b in self.bases(rk, f) or [()]:
                todo += [h for path in u.sites for h in self.resolve(rk, b + u.ctx, path) if h[2] >= 0
                         and self.facts(h[0], h[1]).units[h[2]].kind == "macro_rules!"]
            while todo:
                h = todo.pop()
                if h in seen_macros or h == (rk, f, k):
                    continue
                seen_macros.add(h)
                mu = self.facts(h[0], h[1]).units[h[2]]
                sites += mu.sites
                idents |= mu.idents
                todo += [(h[0], g, i) for name in mu.macro_calls for g, i in self.macros(h[0]).get(name, ())]
            for ident in sorted(idents):
                hits += [((rk, f, i), ident) for i in facts.named.get((u.ctx, ident), ())]
            for b in self.bases(rk, f) or [()]:
                mod = b + u.ctx
                for path in sites:
                    hits += [(h, "::".join(path)) for h in self.resolve(rk, mod, path)]
                for c, bp, name, _ in facts.binds:
                    if bp[-1] != "*" and name in idents:
                        hits += [(h, f"{name} (use {'::'.join(bp)})") for h in self.resolve(rk, b + c, bp)]
                for c, bp in facts.globs:
                    for grk, gmod in self.module(rk, b + c, bp[:-1]):
                        for ident in sorted(idents):
                            hits += [(h, f"{ident} (use {'::'.join(bp)})") for h in self.walk(grk, gmod, (ident,), True)]
            for h, methods, how in self.in_scope(rk, f):
                named = idents & methods
                if named:
                    hits.append((h, f"{min(named)}() of a trait in scope, {how}"))
            macros = self.macros(rk)
            for name in sorted(u.macro_calls):
                hits += [((rk, g, i), f"{name}!") for g, i in macros.get(name, ())]
            fdir = os.path.dirname(f)
            for inc in u.includes:
                p = os.path.normpath(os.path.join(fdir, inc))
                if self.tree.exists(p):
                    hits += [(h, f"include {inc}") for h in self.units(rk, p)]
            pkg = self.tree.packages.get(rk[0])
            if pkg is not None and u.literals:
                for p in self.tree._resolve_literals(pkg, [(f, lit) for lit in u.literals]):
                    if not p.endswith("/"):
                        hits.append(((rk, p, -1), "path literal"))
            hits += [((rk, f, i), "an item-level macro call of the file") for i in facts.anon]
        out, seen = [], {n}
        for h, how in hits:
            m = self.node(h)
            if m not in seen:
                seen.add(m)
                out.append((m, how))
        self.adj[n] = out
        return out

    def root_rules(self, rk: RootKey, files: list[str]) -> list[tuple[int, str, str, list[list[int]], str | None]]:
        """Every associated item of root `rk`'s `files`: (its node, its file, its impl's header, the units that must
        all be reached for it to run — one list per type the header names, any unit of a list reaching it — and the
        name some reached unit must say, None for an item std calls unnamed; rule 1)."""
        out = []
        for f in files:
            if not f.endswith(".rs"):
                continue
            facts = self.facts(rk, f)
            bases = self.bases(rk, f) or [()]
            sides: dict[ImplHead, tuple[list[list[int]], bool]] = {}
            for k, u in enumerate(facts.units):
                if u.impl is None:
                    continue
                ih = u.impl
                if ih not in sides:

                    def nodes(p: tuple[str, ...], u: Unit = u) -> list[int]:
                        local = facts.named.get((u.ctx, p[0]), []) if len(p) == 1 else []
                        hits = [(rk, f, i) for i in local] or [h for b in bases for h in self.resolve(rk, b + u.ctx, p)]
                        # a re-export on the way is not the type: only the defining unit stands for it
                        return sorted({self.node(h) for h in hits if h[2] >= 0 and h[2] != self.facts(h[0], h[1]).uses})

                    head = nodes(ih.self_head) if ih.self_head else []
                    trait = nodes(ih.trait) if ih.trait else []
                    args = [g for g in (nodes(a) for a in ih.self_args) if g]
                    # a workspace trait's items run only through the trait, whose unit its callers reach (the method rule)
                    if head:
                        groups = [head, *args] + ([trait] if trait else [])
                    else:
                        groups = ([trait] if trait else []) + args + [g for g in (nodes(a) for a in ih.trait_args) if g]
                    # an inherent impl's items and a workspace trait's are called by name; a foreign trait's by std
                    sides[ih] = (groups, ih.trait is None or bool(trait))
                groups, by_name = sides[ih]
                out.append((self.node((rk, f, k)), f, ih.text, groups, u.name if by_name else None))
        return out

    def rules(self) -> RuleSet:
        """The lib roots' rules, shared by every gate."""
        if self._rules is None:
            self._rules = RuleSet([r for f in sorted(self.file_root) for r in self.root_rules(self.file_root[f], [f])])
        return self._rules

    def own_rules(self, rk: RootKey) -> RuleSet:
        """The rules of a target's own root that is not a lib's (a test crate, a bin, a lib's test build): its own
        impls run only in that target."""
        if rk not in self._own_rules:
            self._own_rules[rk] = RuleSet(self.root_rules(rk, sorted(set(self.mods(rk).values()))))
        return self._own_rules[rk]

    # -- gates --

    def gate(self, graph: Graph, n: str) -> dict[int, tuple[int, str]]:
        """Every unit gate `n`'s targets reach: node -> (the node it was reached from, the step); a target's own
        unit -> (-1, the target)."""
        key = (n, graph.recipes[n].text)
        if key in self._gates:
            return self._gates[key]
        ri = graph.inputs(n)
        linked = set(ri.files)
        sets = [self.rules()]
        par: dict[int, tuple[int, str]] = {}
        said: set[str] = set()
        work: list[int] = []
        nodes = self.nodes

        def seed(h: Hit, label: str) -> None:
            m = self.node(h)
            if m not in par:
                par[m] = (-1, label)
                work.append(m)

        def fire(rs: RuleSet, i: int) -> None:
            m, f, text, groups, name = rs.rules[i]
            if m in par or f not in linked or (name is not None and name not in said):
                return
            if all(any(x in par for x in g) for g in groups):
                via = next((x for x in groups[0] if x in par), -1) if groups else -1
                par[m] = (via, f"{text}, {name} named" if name else text) if via >= 0 else (-1, f"{text} (its file linked)")
                work.append(m)

        for pname, kind, tname, _ in ri.targets:
            pkg = self.tree.packages[pname]
            t = pkg.lib() if kind in ("lib", "doctest", "libtest") else pkg.find(kind, tname or "")
            if t is None:
                continue
            label = f"{pname} {kind}" + (f" {tname}" if tname else "")
            if kind == "doctest":
                # a doctest run compiles the lib and runs its examples: what they name is reached
                rk = (pname, t.src, False)
                for h in self.units(rk, self.doctests(pname)):
                    if h[2] >= 0:
                        seed(h, label)
                continue
            test = kind not in ("lib", "doctest")
            rk = (pname, t.src, test)
            if rk not in self.lib_roots:
                own_set = self.own_rules(rk)
                if own_set.rules and own_set not in sets:
                    sets.append(own_set)
            own = [h for f in sorted(self.tree.tree_of(t.src, test)[0]) for h in self.units(rk, f)]
            # a test harness runs its #[test] fns, and they reach the rest; a target with none runs its main
            entries = [h for h in own if kind in ("libtest", "test") and h[2] >= 0 and self.facts(h[0], h[1]).units[h[2]].test]
            for h in entries or own:
                seed(h, label)
        builds = {self.tree.packages[bpkg].name: src for src, bpkg in self.builds.items() if src in linked}
        names = frozenset().union(*(rs.names for rs in sets))
        for rs in sets:
            for i in rs.free:
                fire(rs, i)
        while True:
            while work:
                m = work.pop()
                for rs in sets:
                    for i in rs.by_node.get(m, ()):
                        fire(rs, i)
                rk, f, k = nodes[m]
                if k >= 0:
                    for name in self.facts(rk, f).units[k].idents & names:
                        if name not in said:
                            said.add(name)
                            for rs in sets:
                                for i in rs.by_name.get(name, ()):
                                    fire(rs, i)
                for x, how in self.edges(m):
                    if x not in par and nodes[x][1] in linked:
                        par[x] = (m, how)
                        work.append(x)
            # a build script's output (its env, cfgs and OUT_DIR files) reaches only its own crate's code
            reached = {self.file_root[nodes[m][1]][0] for m in par if nodes[m][1] in self.file_root}
            for bpkg in sorted(reached & set(builds)):
                src = builds.pop(bpkg)
                brk = (bpkg, src, False)
                for f in sorted(self.tree.tree_of(src)[0]):
                    for h in self.units(brk, f):
                        seed(h, f"build script of {bpkg}, its crate reached")
            if not work:
                break
        self._gates[key] = par
        return par

    def gate_files(self, graph: Graph, n: str) -> dict[str, int]:
        """file -> the first unit of it gate `n`'s reach holds."""
        key = (n, graph.recipes[n].text)
        if key not in self._gate_files:
            out: dict[str, int] = {}
            for m in self.gate(graph, n):
                out.setdefault(self.nodes[m][1], m)
            self._gate_files[key] = out
        return self._gate_files[key]

    def label(self, m: int) -> str:
        rk, f, k = self.nodes[m]
        if k < 0:
            return f
        u = self.facts(rk, f).units[k]
        what = u.impl.text if u.impl is not None else f"{u.kind} {'::'.join((*u.ctx, u.name or '_'))}"
        return f"{f}:{u.line} ({what})"

    def chain(self, par: dict[int, tuple[int, str]], m: int) -> str:
        """Node `m`'s chain in one gate's reach: the target and its own unit first, each step's label before the unit
        it reaches."""
        steps: list[tuple[str, int]] = []
        at = m
        while par[at][0] >= 0 and len(steps) < len(par):
            src, how = par[at]
            steps.append((how, at))
            at = src
        head = par[at][1]
        return f"{head}: {self.label(at)}" + "".join(f" -[{how}]-> {self.label(x)}" for how, x in reversed(steps))


def reach_of(tree: Tree) -> Reach:
    """The tree's one Reach: its item graph and per-gate closures are kept across calls (the self-test checks the
    table several times on one tree)."""
    r = getattr(tree, "_reach", None)
    if r is None:
        r = Reach(tree)
        tree._reach = r
    return r


def dep_reach(tree: Tree, graph: Graph, gates: list[str], f: str) -> dict[str, int]:
    """gate -> the first unit of `f` its reach holds, for each gate that reaches `f` through a dependency edge: its
    reach holds a unit of `f`, and its read of `f` is not its own target's (narrow() picks those without a row)."""
    r = reach_of(tree)
    out = {}
    for n in gates:
        w = graph.inputs(n).match(f)
        if w is None or _own_origin(w):
            continue
        hit = r.gate_files(graph, n).get(f)
        if hit is not None:
            out[n] = hit
    return out


def reach_chain(tree: Tree, graph: Graph, n: str, node: int) -> str:
    """How gate `n` reaches unit `node`: its target and own unit, then each step."""
    r = reach_of(tree)
    return r.chain(r.gate(graph, n), node)

def gate_paths_problems(tree: Tree, recipes: dict[str, Recipe], rows: list[PathRow] | None = None, stats: dict[str, int] | None = None) -> list[str]:
    """The table against the tree: each row's recipes are gate recipes; its glob matches a file; each file it
    matches that a gate reads through a dependency edge is in the inputs of every recipe the row names
    (a row cannot claim a gate runs code it does not link); every gate the derivation (Reach) says reaches a
    file the glob matches through a dependency edge is in the row or named by the row's exceptions, each
    exception naming such a gate (a gate that reaches the file through another lib's code is never left out
    silently); and every gate whose own target names the module the glob stands for is in the row (a gate
    that calls into the module directly is never left out). `stats`, when given, gains the counts the
    summary prints: rows held to the derivation, exceptions, the gates they name."""
    try:
        rows = rows if rows is not None else load_gate_paths(tree.root)
    except RecipeError as err:
        return [str(err)]
    graph = Graph(tree, recipes)
    gates = [n for n in sorted(recipes, key=lambda n: recipes[n].line) if n.startswith(GATE_PREFIX)]
    files = shipped_files(tree.root)
    problems: list[str] = []
    for r in rows:
        at = f"{GATE_PATHS}:{r.line} {r.glob}"
        if not r.every():
            for n in r.gates():
                if n not in recipes or not n.startswith(GATE_PREFIX):
                    problems.append(f"{at}: {n} is not a gate-* recipe in the justfile")
            for n in r.weeklies():
                if n not in recipes:
                    problems.append(f"{at}: {n} is not a weekly-* recipe in the justfile")
        for alt in glob_alternatives(r.glob):
            if not any(glob_regex(alt).match(f) for f in files):
                problems.append(f"{at}: {alt} matches no file in the tree")
        matched = [f for f in files if r.pattern.match(f)]
        if not matched:
            continue
        if r.every():
            continue
        # A trigger is the lead's choice of what a weekly gate is run by, not a claim that it links every file
        # the glob matches; it must read one of them, or the glob has drifted off the gate it names.
        for n in r.weeklies():
            if n in recipes and not any(graph.inputs(n).match(f) for f in matched):
                problems.append(f"{at}: {n} reads none of the {len(matched)} files the glob matches — a trigger off its gate")
        if not r.gates():
            continue
        named = [n for n in r.gates() if n in recipes]
        unread: dict[str, list[str]] = {}
        for f in matched:
            dep = any(not _own_origin(w) for w in (graph.inputs(n).match(f) for n in gates) if w)
            if not dep:
                continue
            for n in named:
                if graph.inputs(n).match(f) is None:
                    unread.setdefault(n, []).append(f)
        for n, fs in unread.items():
            problems.append(f"{at}: {n} does not read {fs[0]}{f' and {len(fs) - 1} more files the glob matches' if len(fs) > 1 else ''} "
                            "(its targets do not link them) — take it out of the row")
        # the derivation: a gate whose targets can run code of a file the glob matches, through another lib's code
        held = set(r.gates()) | set(NARROW_ALWAYS)
        missing: dict[str, tuple[str, int]] = {}
        used: set[str] = set()
        for f in matched:
            for n, node in dep_reach(tree, graph, gates, f).items():
                if n in held:
                    continue
                if n in r.exceptions:
                    used.add(n)
                    continue
                missing.setdefault(n, (f, node))
        for n, (f, node) in missing.items():
            problems.append(f"{at}: {n} reaches {f} — add it to the row, or name it in the row's exceptions with the reason; "
                            f"the chain: {reach_chain(tree, graph, n, node)}")
        for n in sorted(set(r.exceptions) - used):
            problems.append(f"{at}: the exception names {n}, which reaches no file the glob matches through a dependency — take it out")
        if stats is not None:
            stats["rows"] = stats.get("rows", 0) + 1
            stats["exceptions"] = stats.get("exceptions", 0) + (1 if r.exceptions else 0)
            stats["excepted"] = stats.get("excepted", 0) + len(r.exceptions)
        mods = [m for m in (module_of(tree, alt) for alt in glob_alternatives(r.glob)) if m is not None]
        paths = [((ident, *rest), kids) for pkg, rest, kids in mods for ident in sorted(crate_idents(tree, pkg))]
        pkgs = {m[0] for m in mods}
        if not paths:
            continue
        for n in gates:
            if n in r.recipes or n in NARROW_ALWAYS:
                continue
            for pname, kind, tname, _ in graph.inputs(n).targets:
                if pname in pkgs or kind in ("lib", "libtest", "doctest"):
                    continue
                t = tree.packages[pname].find(kind, tname or "")
                if t is None:
                    continue
                own, _ = tree.tree_of(t.src, test=True)
                hit = next((f for f in sorted(own) if any(names_module(named_paths(tree.read(f)), p, kids) for p, kids in paths)), None)
                if hit:
                    problems.append(f"{at}: {n} runs {kind} {tname}, whose {hit} names {'::'.join(paths[0][0])} — add {n} to the row")
                    break
    return problems


def weekly_problems(recipes: dict[str, Recipe], rows: list[PathRow] | None = None, root: str = ROOT) -> list[str]:
    """Every weekly-* recipe has a trigger row: one that no change can name runs only when someone remembers it."""
    try:
        rows = rows if rows is not None else load_gate_paths(root)
    except RecipeError as err:
        return [str(err)]
    named = {n for r in rows for n in r.weeklies()}
    return [f"justfile:{recipes[n].line} {n}: no row of {GATE_PATHS} names it — no changed file can pull it into `just affected`"
            for n in sorted(recipes, key=lambda n: recipes[n].line) if n.startswith(WEEKLY_PREFIX) and n not in named]


# ---- the host group: device-crate tests that open no card ----

HOST_GROUP = "host"
CARD_OPEN = re.compile(r"\bGpu\s*::|\b(?:Cuda)?Stream\b|\bDeviceBuffer\b|\bcuInit\b|\bCudaContext\b")


def host_problems(tree: Tree, recipes: dict[str, Recipe]) -> list[str]:
    """A recipe in `[group('host')]` (tools/gate-batch.sh balances it over both lanes, like a recipe with no
    device code) runs tests that open no card: it calls no tools/gpu-gate.sh and picks no card for box.sh,
    carries neither solo nor v41-load, runs a cargo test, and the code of every test it runs names nothing
    that opens a card (CARD_OPEN). The test code read is the test-only module that holds each run test —
    its inline block, or its file — and the whole tree of a test target. What the check cannot see: a lib
    function a test calls that opens a card itself, and a test-only helper module that holds no test."""
    scan = TestScan(tree)
    problems: list[str] = []
    for name in sorted(recipes, key=lambda n: recipes[n].line):
        r = recipes[name]
        if HOST_GROUP not in r.groups:
            continue
        at = f"justfile:{r.line} {name}"
        for g in ("solo", "v41-load"):
            if g in r.groups:
                problems.append(f"{at}: [group('{HOST_GROUP}')] with [group('{g}')] — a host recipe runs beside other items, which those groups forbid")
        text = "\n".join(r.lines)
        if "gpu-gate.sh" in text or re.search(r"\bBLOOMERY_CARD=", text):
            problems.append(f"{at}: [group('{HOST_GROUP}')] on a recipe that runs a card gate or picks a card")
        try:
            invs = [i for i in recipe_commands(r).invocations if i.sub == "test"]
        except RecipeError as err:
            problems.append(f"{at}: {err}")
            continue
        if not invs:
            problems.append(f"{at}: [group('{HOST_GROUP}')] on a recipe that runs no cargo test")
        for inv in invs:
            try:
                args = libtest_args(inv)
            except RecipeError as err:
                problems.append(f"{at}: {err}")
                continue
            targets, _ = tree.resolve(inv)
            for pname, kind, tname, _ in targets:
                pkg = tree.packages[pname]
                t = pkg.lib() if kind == "libtest" else pkg.find(kind, tname or "")
                if t is None or kind not in ("libtest", "test", "bin"):
                    continue
                items, errs = scan.tests_of(t.src)
                problems += [f"{at}: {e}" for e in errs]
                run = [it for it in items if args.misses(it.path, it.ignored) is None]
                if not run:
                    continue
                if kind == "test":
                    regions = [(f, 0, None) for f in sorted(tree.tree_of(t.src, test=True)[0])]
                else:
                    test_files = tree.tree_of(t.src, test=True)[0] - tree.tree_of(t.src)[0]
                    regions = sorted({(it.file, 0, None) for it in run if it.file in test_files})
                    for f, a, b in scan.test_blocks(t.src):
                        if b is None:
                            continue
                        lo, hi = scan.line_of(f, a), scan.line_of(f, b)
                        if any(it.file == f and lo <= it.line <= hi for it in run):
                            regions.append((f, a, b))
                for f, a, b in regions:
                    code = rust_lex(tree.read(f))[0]
                    for m in CARD_OPEN.finditer(code, a, len(code) if b is None else b):
                        problems.append(f"{at}: [group('{HOST_GROUP}')], and {f}:{scan.line_of(f, m.start())} in a test it runs "
                                        f"names {m.group(0)!r} — a test that may open a card is not host work")
    return list(dict.fromkeys(problems))


# ---- the narrowed selection ----

@dataclass
class Carrier:
    """A cargo target with kernels of its own: its PTX is a bundle a scan may cover. cuda-oxide names a bundle
    after the package being compiled (`CARGO_PKG_NAME`, the `ptx-scan: modN bundle=<name>` lines), so a bin's own
    kernels and every other bin's of the same package share one bundle name."""

    label: str
    bundle: str  # the package name: what the bundle of this target's kernels is called in a scan
    files: set[str]  # the files its kernels can take code from: its own tree and its closure's libs
    recipes: list[str]  # the gates that link it
    bin: str | None = None  # a bin carrier's target name; None for a lib
    siblings: frozenset[str] = frozenset()  # a lib carrier: the bin carriers of its package, whose bundle carries the same name


def carrier_covered(c: Carrier, bin: str, bundles: Iterable[str]) -> bool:
    """Whether a scan of `bin`, whose `modN bundle=` lines are `bundles`, reads carrier `c`'s kernels. A lib's
    bundle is named after the lib, so any scan that carries it reads it. A bin's own kernels are in the bundle
    named after its package, which every bin of the package shares: only the scan of that bin reads them, and a
    scan of a bin carrier cannot vouch for a lib of the same package (its package-named bundle may be the bin's)."""
    if c.bundle not in bundles:
        return False
    return bin == c.bin if c.bin is not None else bin not in c.siblings


def kernel_carriers(side: Side, gates: list[str]) -> list[Carrier]:
    tree = side.tree

    def has_kernels(src: str) -> bool:
        files, _ = tree.tree_of(src)
        return any(KERNEL_ATTR.search(rust_lex(tree.read(f))[0]) for f in files)

    out: list[Carrier] = []
    for p in sorted(tree.packages.values(), key=lambda p: p.name):
        for t in p.targets:
            if t.kind not in ("lib", "bin") or not has_kernels(t.src):
                continue
            if t.kind == "lib":
                files = set(tree.lib_files(p.name))
                for dep in tree.closure(p, [], False, all_features=True):
                    files |= set(tree.lib_files(dep))
                users = [n for n in gates if side.graph.inputs(n).match(t.src)]
                out.append(Carrier(f"lib {p.name}", p.name, files, users))
            else:
                files = set(tree.target_files(p.name, "bin", t.name, list(p.features), True, True))
                users = [n for n in gates if any(k == "bin" and tn == t.name for _, k, tn, _ in side.graph.inputs(n).targets)]
                out.append(Carrier(f"bin {t.name}", p.name, files, users, bin=t.name))
    for c in out:
        if c.bin is None:
            c.siblings = frozenset(o.bin for o in out if o.bin is not None and o.bundle == c.bundle)
    return out


@dataclass
class Narrowed:
    full: list[str]  # reasons the full list stands; empty when the list narrowed
    picks: dict[str, str]  # recipe -> the rule that picked it, in justfile order
    files: list[str]  # one line per changed file: the rule that mapped it
    verdicts: list[ScanVerdict]


def narrow(changed: list[str], a: Side | None, b: Side, scans: list[tuple[ScanLog, ScanLog]], rows: list[PathRow], tables: TableReads | None = None) -> Narrowed:
    gates = [n for n in sorted(b.recipes, key=lambda n: b.recipes[n].line) if n.startswith(GATE_PREFIX)]
    res = Narrowed([], {}, [], [])
    if not scans:
        res.full.append("no --scan pair: whether a kernel moved is unknown")
        return res
    res.verdicts = [scan_verdict(x, y) for x, y in scans]
    for v in res.verdicts:
        if v.kind == "moved":
            res.full.append(f"ptx-scan {v.line()}")
    picks: dict[str, str] = {}
    tables = tables if tables is not None else TableReads(changed, a, b)

    def pick(n: str, why: str) -> None:
        if n in b.recipes:
            picks.setdefault(n, why)

    carriers = kernel_carriers(b, gates)
    for f in changed:
        if f == "justfile":
            moved = [n for n in gates if a is None or n not in a.recipes or a.recipes[n].text != b.recipes[n].text]
            for n in moved:
                pick(n, "new recipe" if a is None or n not in a.recipes else "recipe text")
            res.files.append(f"justfile: the text of {len(moved)} gate recipes" + (f" ({' '.join(moved[:6])}{' …' if len(moved) > 6 else ''})" if moved else ""))
            continue
        side = b if b.tree.exists(f) or a is None else a
        own, dep = [], []
        for n in gates:
            if n not in side.recipes:
                continue
            # a manifest read by table: a move of the recipe's own entries alone is its own target's
            w, own_entry = tables.hit(n, f, side)
            if w is not None:
                (own if (own_entry if own_entry is not None else _own_origin(w)) else dep).append((n, w))
        for n, w in own:
            pick(n, f"{f} [{w}]")
        hit_rows = [r for r in rows if r.pattern.match(f) and r.gates()]
        star = next((r for r in hit_rows if r.every()), None)
        rule: list[str] = [f"own target of {len(own)}"] if own else []
        if NOT_HOST.search(f):
            rule.append("not a host path")
        elif star is not None:
            res.full.append(f"{f} maps to * ({GATE_PATHS}:{star.line}: {star.why})")
            rule.append(f"* (row {star.line})")
        elif hit_rows:
            for r in hit_rows:
                for n in r.gates():
                    pick(n, f"{f} [row {r.line}]")
            rule.append("rows " + " ".join(str(r.line) for r in hit_rows) + f" -> {len({n for r in hit_rows for n in r.gates()})} recipes")
        elif dep:
            n, w = dep[0]
            res.full.append(f"{f}: {n} reads it through a dependency ({w}) and no row of {GATE_PATHS} maps it")
            rule.append("no row")
        elif not own:
            rule.append("no gate reads it")
        # a manifest whose diff moved only named entries changes no kernel: the bins whose entries moved
        # are picked above as their own targets
        entries_only = f in tables.docs and all(own for _, own in view_moves(*tables.docs[f], os.path.dirname(f), None))
        for c in carriers:
            if f in c.files and not entries_only and not any(carrier_covered(c, v.bin, v.bundles) for v in res.verdicts):
                for n in c.recipes:
                    pick(n, f"{f} [{c.label}: kernels no scan pair covers]")
                rule.append(f"{c.label} not scanned -> {len(c.recipes)} recipes")
        res.files.append(f"{f}: {'; '.join(rule)}")
    # The spill ratchet joins the narrowed list only when the PTX or its own pins can have moved: a
    # scan pair that is not identical (a new entry has spill bytes no row pins), or a changed file
    # the recipe itself reads — its scripts, tools/ref/ptx-shapes.tsv, the cargo globals (a
    # toolchain move changes ptxas). Every pair identical under one toolchain means byte-equal PTX,
    # so equal spill columns against equal pins: the check would read its own green back.
    if any(v.kind != "identical" for v in res.verdicts):
        for n in NARROW_ALWAYS:
            if n in b.recipes:
                picks.setdefault(n, "the spill ratchet: a scan pair is not identical")
    else:
        for n in NARROW_ALWAYS:
            if n not in b.recipes:
                continue
            for f in changed:
                if f == "justfile":
                    continue
                side = b if b.tree.exists(f) or a is None else a
                if n in side.recipes and side.graph.inputs(n).match(f) is not None:
                    picks.setdefault(n, f"the spill ratchet: its own input {f} moved")
                    break
    res.picks = {n: picks[n] for n in gates if n in picks}
    return res


# ---- the scan set: the bins whose ptx-scans cover every kernel carrier ----
#
# `just narrow` scans these bins, base and change, and hands the pairs to narrow(). A bin covers what its
# scan's bundle lines can name (carrier_covered): the device libs its package links under its features, and its
# own kernels when it is a bin carrier. The libs are read from the package's dependency closure, which can
# name a lib the bin never references (the linker keeps a bundle only when the bin reaches its loader):
# narrow() judges the scans' own bundle lines, so a lib the scan lacks keeps its gates, and the line
# `just narrow` prints per bin shows which bundles each scan carried.

SCAN_PACKAGE = "bloomery-gpu-gates"  # the package whose bins `just ptx-scan` builds and tools/ptx-scan.sh reads


@dataclass
class ScanPick:
    bin: str
    features: tuple[str, ...]  # the cargo features the scan's build enables, sorted
    covers: list[str]  # the labels of every carrier its scan can name


def scan_shapes(side: Side) -> dict[str, set[tuple[str, ...]]]:
    """Per bin of SCAN_PACKAGE the feature sets a scan can build it with: its required-features, and every
    feature list a gate-* or weekly-* recipe builds it with. A recipe's list is kept only when it enables the
    bin's required-features (the recipe's own build proves it, the one-line check keeps a hand-edited
    justfile from handing out a shape cargo refuses)."""
    tree = side.tree
    pkg = tree.packages.get(SCAN_PACKAGE)
    if pkg is None:
        raise RecipeError(f"scan-set: no workspace package {SCAN_PACKAGE}, the package whose bins a ptx-scan builds")
    bins = {t.name: t for t in pkg.targets if t.kind == "bin"}
    shapes: dict[str, set[tuple[str, ...]]] = {n: {tuple(sorted(set(t.required)))} for n, t in bins.items()}
    for n in side.recipes:
        if not n.startswith((GATE_PREFIX, WEEKLY_PREFIX)):
            continue
        for p, kind, name, feats in side.graph.inputs(n).targets:
            if p != SCAN_PACKAGE or kind != "bin" or name not in bins:
                continue
            enabled = tree.feature_closure(pkg, list(feats), True, False)[0]
            if set(bins[name].required) <= enabled:
                shapes[name].add(tuple(sorted(set(feats))))
    return shapes


def scan_set(side: Side) -> list[ScanPick]:
    """The minimal list of (bin, features) whose scans cover every kernel carrier of the tree, greedy: the scan that
    covers most uncovered carriers first, ties to fewer features, then to the bin's name. A carrier no bin of
    SCAN_PACKAGE can cover is a named error, never dropped."""
    gates = [n for n in sorted(side.recipes, key=lambda n: side.recipes[n].line) if n.startswith(GATE_PREFIX)]
    carriers = kernel_carriers(side, gates)
    shapes = scan_shapes(side)
    pkg = side.tree.packages[SCAN_PACKAGE]
    cands: list[tuple[str, tuple[str, ...], list[int]]] = []
    for bin_name in sorted(shapes):
        for feats in sorted(shapes[bin_name]):
            bundles = set(side.tree.closure(pkg, list(feats), False))
            bundles |= {c.bundle for c in carriers if c.bundle == pkg.name and c.bin in (None, bin_name)}
            cands.append((bin_name, feats, [i for i, c in enumerate(carriers) if carrier_covered(c, bin_name, bundles)]))
    uncovered = set(range(len(carriers)))
    picks: list[ScanPick] = []
    while uncovered:
        best = max(cands, key=lambda c: (len(uncovered & set(c[2])), -len(c[1])), default=None)
        if best is None or not uncovered & set(best[2]):
            break
        picks.append(ScanPick(best[0], best[1], [carriers[i].label for i in best[2]]))
        uncovered -= set(best[2])
    if uncovered:
        raise RecipeError("scan-set: no bin of " + SCAN_PACKAGE + " covers " + ", ".join(carriers[i].label for i in sorted(uncovered))
                          + " — a scan reads only the bins of that package, and a lib's bundle needs a bin that links it")
    return picks


def cmd_scan_set(args: argparse.Namespace) -> int:
    tmp = tempfile.mkdtemp(prefix="recipes-scanset-")
    try:
        side = make_side(archive(args.rev, tmp) if args.rev else ROOT)
        picks = scan_set(side)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for p in picks:
        print(f"{p.bin}\t{','.join(p.features)}\t{'; '.join(p.covers)}")
    n_carriers = len({c for p in picks for c in p.covers})
    print(f"scan-set: {len(picks)} bins cover {n_carriers} kernel carriers", file=sys.stderr)
    return 0


def trigger_rows(side: Side) -> tuple[list[PathRow], str | None]:
    """The trigger rows of the tree the change is read at: its own table, beside its own justfile (a range's B is
    a `git archive`, whose recipes can predate the working tree's table)."""
    if not os.path.exists(os.path.join(side.tree.root, GATE_PATHS)):
        return [], f"no {GATE_PATHS} in the tree read: no weekly trigger read"
    return load_gate_paths(side.tree.root), None


@dataclass
class Trigger:
    recipe: str
    files: list[str]  # the changed files that name it, the first one's rule in `why`
    why: str


def weekly_triggers(changed: list[str], a: Side | None, b: Side, rows: list[PathRow]) -> list[Trigger]:
    """The weekly-* recipes a change names, in justfile order: by a trigger row whose glob matches a changed
    file, or by a change of the recipe's own text. A row naming a weekly recipe the justfile lacks is refused."""
    names = [n for n in sorted(b.recipes, key=lambda n: b.recipes[n].line) if n.startswith(WEEKLY_PREFIX)]
    for r in rows:
        for n in r.weeklies():
            if n not in b.recipes:
                raise RecipeError(f"{GATE_PATHS}:{r.line}: the trigger {n} is not a weekly-* recipe in the justfile")
    hits: dict[str, dict[str, str]] = {n: {} for n in names}
    for f in changed:
        if f == "justfile":
            for n in names:
                old = a.recipes.get(n) if a is not None else None
                if old is None or old.text != b.recipes[n].text:
                    hits[n].setdefault(f, "new recipe" if old is None else "recipe text")
            continue
        for r in rows:
            if r.pattern.match(f):
                for n in r.weeklies():
                    hits[n].setdefault(f, f"trigger {GATE_PATHS}:{r.line} {r.glob}")
    return [Trigger(n, list(hits[n]), next(iter(hits[n].values()))) for n in names if hits[n]]


# ----------------------------------------------------------------------------------------------
# the plan files `just records-refresh` writes
# ----------------------------------------------------------------------------------------------
# records-refresh runs only on the box, so a plan it lists can be missing from the tree while every Mac
# check passes and the box gate that reads it goes red. The rule below checks presence only: whether a
# plan's contents are the engine's is the box's to say (the recipe rewrites them from generate_ds41).
# The plans are printed at the plan room (PLAN_ROOM): a bare `generate_ds41 --plan` would read the box's
# live memory, and the files would move with whatever else holds RAM there.

REFRESH_RECIPE = "records-refresh"
PLAN_MARKER = "#> "  # tools/bloomery/records.py refresh: the records after this line go to its path
PLAN_ROOM = "tools/ref/plan-room.sh"  # generate_ds41 --plan at a pinned host room, refused when it moves with it
COUNTS_TOOL = "tools/flow/ds41_prefill.py"
_SHELL_VAR = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")


@dataclass
class PlanFile:
    path: str  # relative to the tree root
    what: str  # the marker's text after the path
    depth: int  # its `--depth`


def _command_start(w: str) -> bool:
    return (bool(w) and all(c in _PUNCT for c in w) and w not in _REDIRECTS) or w in ("do", "then", "else", "elif", "{", "!")


def _box_remotes(recipe: Recipe) -> list[str]:
    """The command text of each tools/box.sh call in the recipe, as recipe_commands reads it (prefixes and
    pipes around the call allowed, unlike box_command)."""
    out = []
    for line in recipe.lines:
        stripped = line.lstrip("@-").strip()
        if not stripped or stripped.startswith("#"):
            continue
        for cmd in simple_commands(shell_words(stripped)):
            rest = strip_wrappers(cmd)[1]
            if rest and _norm(rest[0]) == "tools/box.sh":
                out.append(" ".join(rest[1:]))
    return out


def refresh_plans(recipe: Recipe) -> list[PlanFile]:
    """The files records-refresh writes from generate_ds41 records: each `echo "#> <path> <what>"` of its box
    command, its `for VAR in …; do … done` loops expanded. Refused by name: a loop value or marker that is not
    literal after expansion, a path outside the tree, a marker with no integer `--depth`, a path written twice,
    no marker at all."""
    remotes = _box_remotes(recipe)
    if len(remotes) != 1:
        raise RecipeError(f"{recipe.name}: {len(remotes)} tools/box.sh calls, not one")
    words = shell_words(remotes[0])
    loops: list[tuple[str, list[str]]] = []
    out: list[PlanFile] = []
    start, i = True, 0
    while i < len(words):
        w = words[i]
        if start and w == "for":
            if i + 2 >= len(words) or words[i + 2] != "in":
                raise RecipeError(f"{recipe.name}: a `for` loop this parser cannot read: {' '.join(words[i:i + 4])}")
            var, j, vals = words[i + 1], i + 3, []
            while j < len(words) and words[j] not in (";", "do"):
                if "$" in words[j] or "`" in words[j]:
                    raise RecipeError(f"{recipe.name}: `for {var}` takes a value that is not literal: {words[j]}")
                vals.append(words[j])
                j += 1
            if j < len(words) and words[j] == ";":
                j += 1
            if j >= len(words) or words[j] != "do":
                raise RecipeError(f"{recipe.name}: `for {var} in …` with no `do`")
            loops.append((var, vals))
            i, start = j + 1, True
            continue
        if start and w == "done":
            if not loops:
                raise RecipeError(f"{recipe.name}: a `done` with no loop open")
            loops.pop()
        elif start and w == "echo" and i + 1 < len(words) and words[i + 1].startswith(PLAN_MARKER):
            combos: list[dict[str, str]] = [{}]
            for var, vals in loops:
                combos = [dict(c, **{var: v}) for c in combos for v in vals]
            for c in combos:
                text = _SHELL_VAR.sub(lambda m: c.get(m.group(1) or m.group(2), m.group(0)), words[i + 1][len(PLAN_MARKER):])
                if "$" in text or "`" in text:
                    raise RecipeError(f"{recipe.name}: a {PLAN_MARKER.strip()} marker names a variable no loop around it sets: {text[:120]}")
                path, _, what = text.partition(" ")
                if os.path.isabs(path) or ".." in path.split("/"):
                    raise RecipeError(f"{recipe.name}: a {PLAN_MARKER.strip()} marker's path is outside the tree: {path}")
                ws = what.split()
                d = ws[ws.index("--depth") + 1] if "--depth" in ws[:-1] else ""
                if not d.isdigit():
                    raise RecipeError(f"{recipe.name}: the marker of {path} names no integer --depth: {what[:120]}")
                out.append(PlanFile(path, what, int(d)))
        start = _command_start(w)
        i += 1
    if loops:
        raise RecipeError(f"{recipe.name}: `for {loops[-1][0]}` with no `done`")
    if not out:
        raise RecipeError(f"{recipe.name}: no `echo \"{PLAN_MARKER}<path> …\"` marker in its box command")
    seen: set[str] = set()
    for p in out:
        if p.path in seen:
            raise RecipeError(f"{recipe.name}: two markers write {p.path}")
        seen.add(p.path)
    return out


def arm_depth(recipe: Recipe, value: str) -> int:
    """The prompt length one generate_ds41 `--arm` value feeds: D and D/N are depth D, corpus:P (prose or
    code, with or without /N) depth P — the depth whose engine plan a --counts reader needs."""
    feed, _, n = value.partition("/")
    if n and not n.isdigit():
        raise RecipeError(f"{recipe.name}: runs {COUNTS_TOOL} --counts on a generate_ds41 --arm it cannot read "
                          f"(D, D/N, prose:P or code:P): {value}")
    name, sep, p = feed.partition(":")
    depth = p if sep else name
    if (sep and name not in ("prose", "code")) or not depth.isdigit():
        raise RecipeError(f"{recipe.name}: runs {COUNTS_TOOL} --counts on a generate_ds41 --arm it cannot read "
                          f"(D, D/N, prose:P or code:P): {value}")
    return int(depth)


def counts_depths(recipe: Recipe) -> list[int]:
    """The prompt lengths whose engine plan a recipe's `ds41_prefill.py --counts` reads: every generate_ds41
    `--depth` and `--arm` in a box command that runs it (the tool reads the plan of each log's call, an arm
    list's each arm its own). Empty for a recipe that does not run it; refused by name when it does and no
    depth this parser can read is found."""
    cmds = [strip_wrappers(c)[1] for r in _box_remotes(recipe) for c in simple_commands(shell_words(r))]
    if not any(COUNTS_TOOL in map(_norm, c) and "--counts" in c for c in cmds):
        return []
    depths: list[int] = []
    for c in cmds:
        if not any(os.path.basename(x) == "generate_ds41" for x in c):
            continue
        for k, w in enumerate(c[:-1]):
            if w == "--depth":
                if not c[k + 1].isdigit():
                    raise RecipeError(f"{recipe.name}: runs {COUNTS_TOOL} --counts on a generate_ds41 --depth that is not an integer: {c[k + 1]}")
                depths.append(int(c[k + 1]))
            elif w == "--arm":
                depths.append(arm_depth(recipe, c[k + 1]))
    if not depths:
        raise RecipeError(f"{recipe.name}: runs {COUNTS_TOOL} --counts and no generate_ds41 --depth or --arm this parser can read")
    return depths


def plan_room_problems(recipe: Recipe) -> list[str]:
    """records-refresh prints every plan at the plan room: each `#> ` marker's echo is followed by `bash
    tools/ref/plan-room.sh D C`, D the marker's `--depth` word and C its `BLOOMERY_CED=` value as the text spells
    them (the loop's `$P` and `$c`), so the file a marker names holds the plan it names. A `generate_ds41 --plan`
    anywhere in the box command is refused by name: it plans at the box's live MemAvailable."""
    out: list[str] = []
    for remote in _box_remotes(recipe):
        cmds = [strip_wrappers(c)[1] for c in simple_commands(shell_words(remote))]
        bare = {k for k, c in enumerate(cmds) if c and os.path.basename(c[0]) == "generate_ds41" and "--plan" in c}
        for k in sorted(bare):
            out.append(f"justfile:{recipe.line} {recipe.name}: runs a bare `{' '.join(cmds[k])}`, which plans at the box's "
                       f"live MemAvailable: print each plan through `bash {PLAN_ROOM} $P $c` (the room pinned)")
        for k, c in enumerate(cmds):
            if c[:1] != ["echo"] or len(c) < 2 or not c[1].startswith(PLAN_MARKER):
                continue
            path, _, what = c[1][len(PLAN_MARKER):].partition(" ")
            ws = what.split()
            depth = ws[ws.index("--depth") + 1] if "--depth" in ws[:-1] else "<no --depth>"
            ced = next((w.split("=", 1)[1] for w in ws if w.startswith("BLOOMERY_CED=")), "<no BLOOMERY_CED=>")
            want = ["bash", PLAN_ROOM, depth, ced]
            nxt = cmds[k + 1] if k + 1 < len(cmds) else []
            if k + 1 in bare or [_norm(w) if i == 1 else w for i, w in enumerate(nxt)] == want:
                continue
            out.append(f"justfile:{recipe.line} {recipe.name}: the marker of {path} is followed by `{' '.join(nxt) or 'nothing'}`, "
                       f"not `{' '.join(want)}`: the plan its file holds is printed at the plan room, at the depth and CED "
                       f"its marker names")
    return out


def plan_problems(root: str, recipes: dict[str, Recipe]) -> list[str]:
    """Every plan records-refresh writes, present in the tree at `root` and printed at the plan room
    (`plan_room_problems`); every depth a --counts reader needs, written by it. Presence only, of the files."""
    if REFRESH_RECIPE not in recipes:
        return [f"no `{REFRESH_RECIPE}` recipe: the plans in tools/flow/plans/ have no writer"]
    try:
        plans = refresh_plans(recipes[REFRESH_RECIPE])
    except RecipeError as err:
        return [f"justfile:{recipes[REFRESH_RECIPE].line} {err}"]
    problems = plan_room_problems(recipes[REFRESH_RECIPE])
    problems += [f"{p.path} is not in the tree ({REFRESH_RECIPE}: {p.what}) — run just {REFRESH_RECIPE} on the box and commit it"
                 for p in plans if not os.path.isfile(os.path.join(root, p.path))]
    written = {p.depth for p in plans}
    for name in sorted(recipes, key=lambda n: recipes[n].line):
        try:
            depths = counts_depths(recipes[name])
        except RecipeError as err:
            problems.append(f"justfile:{recipes[name].line} {err}")
            continue
        for d in sorted(set(depths) - written):
            problems.append(f"justfile:{recipes[name].line} {name} reads the engine's plan of P = {d} ({COUNTS_TOOL} --counts), "
                            f"and {REFRESH_RECIPE} writes none: add {d} to its P list, then run just {REFRESH_RECIPE}")
    return problems


# The recipes of the static tier: each compiles every target of the workspace (`--workspace
# --all-targets`), so a bin whose required-features one of them leaves out is never checked or linted.
STATIC_RECIPES = ("check", "lint")


def static_problems(tree: Tree, recipes: dict[str, Recipe], names: tuple[str, ...] = STATIC_RECIPES) -> list[str]:
    """A bin of a workspace package that one of the static recipes leaves out (its required-features are not
    in the recipe's features, so cargo skips it): code no `just check` or `just lint` compiles. Blocks a
    feature-gated server or gate whose code only a box build ever reads."""
    problems: list[str] = []
    for name in names:
        recipe = recipes.get(name)
        if recipe is None:
            problems.append(f"static tier: no `{name}` recipe")
            continue
        invs = [i for i in recipe_commands(recipe).invocations if i.sub in ("check", "clippy")]
        if len(invs) != 1 or not invs[0].all_targets or invs[0].packages:
            problems.append(f"justfile:{recipe.line} {name}: not one `cargo check|clippy --workspace --all-targets`")
            continue
        built, _ = tree.resolve(invs[0])
        have = {(pkg, kind, t) for pkg, kind, t, _ in built}
        for pname in sorted(tree.packages):
            for t in tree.packages[pname].targets:
                if t.kind == "bin" and t.required and (pname, "bin", t.name) not in have:
                    problems.append(
                        f"justfile:{recipe.line} {name}: bin {t.name} of {pname} requires features {t.required}, which "
                        f"the recipe does not enable: the static tier never compiles it"
                    )
    return problems


def cmd_check(args: argparse.Namespace) -> int:
    tree = Tree(ROOT)
    recipes = load_justfile(args.justfile or os.path.join(ROOT, "justfile"))
    held: dict[str, int] = {}
    problems = (check(tree, recipes) + plan_problems(ROOT, recipes) + gate_paths_problems(tree, recipes, stats=held) + weekly_problems(recipes)
                + host_problems(tree, recipes) + static_problems(tree, recipes) + nightly_problems(tree, load_lock(tree.root)))
    for p in problems:
        print(f"check-recipes: {p}", file=sys.stderr)
    print(f"check-recipes: {GATE_PATHS}: {held.get('rows', 0)} gate rows held to the derivation, {held.get('exceptions', 0)} rows "
          f"with exceptions naming {held.get('excepted', 0)} gates")
    return 1 if problems else 0


# ----------------------------------------------------------------------------------------------
# pure crates: the ones `tools/mac-check.sh test` runs natively on the Mac
# ----------------------------------------------------------------------------------------------
# The rule is the module docstring's «pure crates» paragraphs; `tools/mac-check.sh test` runs what it selects.

MAC_TARGET = "aarch64-apple-darwin"
# The cfg keys and names whose value on MAC_TARGET is fixed; every other predicate is unknown.
_MAC_CFG_KEYS = {
    "target_arch": "aarch64",
    "target_os": "macos",
    "target_family": "unix",
    "target_vendor": "apple",
    "target_env": "",
    "target_pointer_width": "64",
    "target_endian": "little",
}
_MAC_CFG_NAMES = {"unix": True, "windows": False}
# The cuda-oxide repository under its upstream name and our fork's ([patch] in Cargo.toml).
DEVICE_GIT_SOURCES = ("/cuda-rust.git", "/cuda-oxide.git")
DEVICE_NAMED_ROOTS = ("cuda-core", "cuda-bindings")
_NOT_ON_MAC = [
    (re.compile(r"\b(?:std|core)::arch::x86_64\b"), "x86_64 intrinsics"),
    (re.compile(r"#\s*\[\s*target_feature\s*\("), "#[target_feature]"),
    (re.compile(r"\bis_x86_feature_detected!"), "is_x86_feature_detected!"),
    (re.compile(r"\bstd::os::linux\b"), "std::os::linux"),
    (
        re.compile(
            r"\blibc::(?:cpu_set_t|CPU_[A-Z_]+|sched_[gs]etaffinity|sched_getcpu|posix_fadvise|POSIX_FADV_[A-Z_]+|sync_file_range"
            r"|SYNC_FILE_RANGE_[A-Z_]+|RUSAGE_THREAD|renameat2|RENAME_[A-Z_]+|prctl|PR_[A-Z_]+|MADV_POPULATE_[A-Z_]+"
            r"|MADV_(?:NO)?HUGEPAGE|MAP_POPULATE|MAP_HUGETLB)\b"
        ),
        "a Linux-only libc item",
    ),
    (re.compile(r"\bAdvice::(?:PopulateRead|PopulateWrite|HugePage|NoHugePage)\b"), "memmap2's Linux-only advice"),
    (re.compile(r"\"/(?:proc|sys)/"), "a Linux /proc or /sys path"),
]
_CFG_ATTR = re.compile(r"#(!?)\s*\[\s*cfg\s*\(")
_RAW_STR = re.compile(r"b?r(#*)\"")
_IDENT_CH = re.compile(r"[A-Za-z0-9_]")


def _blank(buf: list[str], a: int, b: int) -> None:
    for i in range(a, b):
        if buf[i] != "\n":
            buf[i] = " "


@functools.cache
def rust_lex(text: str) -> tuple[str, str]:
    """(code, shape), both as long as `text` with its newlines: `code` has the comments blanked, `shape`
    also the contents of string and char literals — brackets are counted on `shape`, patterns matched on
    `code` (a path literal is in a string). A pure function of the text, cached by it: the scans lex the
    same file once per tree copy and per purity pass."""
    code, shape = list(text), list(text)
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            _blank(code, i, j)
            _blank(shape, i, j)
            i = j
            continue
        if text.startswith("/*", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if text.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif text.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            _blank(code, i, j)
            _blank(shape, i, j)
            i = j
            continue
        if c in "br" and (i == 0 or not _IDENT_CH.match(text[i - 1])):
            m = _RAW_STR.match(text, i)
            if m:
                close = '"' + m.group(1)
                j = text.find(close, m.end())
                j = n if j < 0 else j
                _blank(shape, m.end(), j)
                i = j + len(close)
                continue
        if c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            _blank(shape, i + 1, min(j, n))
            i = j + 1
            continue
        if c == "'":
            if i + 1 < n and text[i + 1] == "\\":
                j = text.find("'", i + 3)
                j = n if j < 0 else j
                _blank(shape, i + 1, j)
                i = j + 1
                continue
            if i + 2 < n and text[i + 2] == "'":
                _blank(shape, i + 1, i + 2)
                i += 3
                continue
        i += 1
    return "".join(code), "".join(shape)


_CFG_TOKEN = re.compile(r"\s*(?:([A-Za-z_][A-Za-z0-9_]*)|\"((?:\\.|[^\"\\])*)\"|([=(),]))")


def parse_cfg(text: str):
    """A cfg predicate as a tree — ("name", n) | ("kv", k, v) | (op, [items]) for all/any/not — or None
    when it is not one this parser reads."""
    toks: list[tuple[str, str]] = []
    pos = 0
    while pos < len(text.rstrip()):
        m = _CFG_TOKEN.match(text, pos)
        if not m:
            return None
        pos = m.end()
        if m.group(1) is not None:
            toks.append(("id", m.group(1)))
        elif m.group(2) is not None:
            toks.append(("str", m.group(2)))
        else:
            toks.append(("p", m.group(3)))
    at = 0

    def pred():
        nonlocal at
        if at >= len(toks) or toks[at][0] != "id":
            raise ValueError
        name = toks[at][1]
        at += 1
        if at < len(toks) and toks[at] == ("p", "="):
            if at + 1 >= len(toks) or toks[at + 1][0] != "str":
                raise ValueError
            at += 2
            return ("kv", name, toks[at - 1][1])
        if at < len(toks) and toks[at] == ("p", "("):
            if name not in ("all", "any", "not"):
                raise ValueError
            at += 1
            items = []
            while at < len(toks) and toks[at] != ("p", ")"):
                items.append(pred())
                if at < len(toks) and toks[at] == ("p", ","):
                    at += 1
            if at >= len(toks):
                raise ValueError
            at += 1
            if name == "not" and len(items) != 1:
                raise ValueError
            return (name, items)
        return ("name", name)

    try:
        tree = pred()
    except ValueError:
        return None
    return tree if at == len(toks) else None


def eval_cfg(pred, test: bool | None, keys: dict[str, str] = _MAC_CFG_KEYS, names: dict[str, bool] = _MAC_CFG_NAMES, features: set[str] | None = None) -> bool | None:
    """The predicate on MAC_TARGET (or the target `keys` and `names` describe): True, False, or None
    (unknown). `test` is cfg(test)'s value; `features`, when given, is the crate's enabled feature set,
    else `feature = …` is unknown."""
    if pred is None:
        return None
    kind = pred[0]
    if kind == "name":
        if pred[1] == "test":
            return test
        return names.get(pred[1])
    if kind == "kv":
        if pred[1] in keys:
            return keys[pred[1]] == pred[2]
        if pred[1] == "feature" and features is not None:
            return pred[2] in features
        return None
    vals = [eval_cfg(p, test, keys, names, features) for p in pred[1]]
    if kind == "not":
        return None if vals[0] is None else not vals[0]
    if kind == "all":
        return False if False in vals else (True if all(v is True for v in vals) else None)
    return True if True in vals else (False if all(v is False for v in vals) else None)


def _item_end(shape: str, start: int) -> int:
    """The end of the item or statement that starts at `start`: after its `;` or `,` at its own depth or
    its closing brace, or before the bracket that closes the block around it."""
    depth, braced = 0, False
    for j in range(start, len(shape)):
        c = shape[j]
        if c in "([{":
            depth += 1
            braced = braced or c == "{"
        elif c in ")]}":
            if depth == 0:
                return j
            depth -= 1
            if depth == 0 and c == "}" and braced:
                return j + 1
        elif c in ";," and depth == 0:
            return j + 1
    return len(shape)


@functools.cache
def mac_view(text: str, test: bool | None) -> str:
    """`text` as aarch64-apple-darwin compiles it: comments blanked, and every item or statement under a
    `#[cfg(…)]` false there (a file under a false `#![cfg(…)]`) blanked; newlines kept, so line numbers hold."""
    return _cfg_blanked(text, test, _MAC_CFG_KEYS, _MAC_CFG_NAMES)[0]


@functools.cache
def build_view(text: str, test: bool) -> tuple[str, str]:
    """(code, shape) of `text` as any target compiles it with cfg(test) set to `test`: the items under a
    `#[cfg(…)]` that `test` alone makes false blanked, every other predicate (a target, a feature) kept —
    the reach reads what some build of the file can hold, so an unknown predicate stays in."""
    return _cfg_blanked(text, test, {}, {}, items=True)


def _cfg_item_end(shape: str, at: int) -> int | None:
    """The end of the item a cfg attribute that ends at `at` stands on, when one starts there (after any further
    attributes): after its `;` or its brace block (`_unit_end`); None for a field, an arm or a statement."""
    i, n = at, len(shape)
    while True:
        while i < n and shape[i].isspace():
            i += 1
        if i < n and shape[i] == "#":
            j = i + 1
            while j < n and shape[j] in " !\t\n":
                j += 1
            close = _close(shape, j) if j < n and shape[j] == "[" else None
            if close is None:
                return None
            i = close + 1
            continue
        break
    m = _UNIT_HEAD.match(shape, i)
    if m is None:
        return None
    kind = re.sub(r"\s+", " ", m.group(1))
    return _unit_end(shape, i, n, kind in _SEMI_KINDS)


def _cfg_blanked(text: str, test: bool | None, keys: dict[str, str], names: dict[str, bool], items: bool = False) -> tuple[str, str]:
    code, shape = rust_lex(text)
    out, out_shape = list(code), list(shape)
    pos = 0
    while True:
        m = _CFG_ATTR.search(shape, pos)
        if not m:
            break
        open_at = m.end() - 1
        depth, close = 0, None
        for j in range(open_at, len(shape)):
            if shape[j] == "(":
                depth += 1
            elif shape[j] == ")":
                depth -= 1
                if depth == 0:
                    close = j
                    break
        if close is None:
            break
        rb = re.compile(r"\s*\]").match(shape, close + 1)
        if not rb:
            pos = close + 1
            continue
        attr_end = rb.end()
        val = eval_cfg(parse_cfg(code[open_at + 1 : close]), test, keys, names)
        if val is False:
            if m.group(1):
                _blank(out, 0, len(out))
                _blank(out_shape, 0, len(out_shape))
                break
            # `items`: an item's own end (`Result<A, B>` in a signature holds a `,` that _item_end stops at)
            end = (_cfg_item_end(shape, attr_end) if items else None) or _item_end(shape, attr_end)
            _blank(out, m.start(), end)
            _blank(out_shape, m.start(), end)
            pos = end
        else:
            pos = attr_end
    return "".join(out), "".join(out_shape)


@functools.cache
def mac_hits(text: str, test: bool | None) -> tuple[tuple[int, str, str], ...]:
    """(line, what, the line's code) for every use in `text` the Mac cannot build or run; cached by the text,
    as mac_view is."""
    hits = []
    for no, line in enumerate(mac_view(text, test).split("\n"), 1):
        for pat, what in _NOT_ON_MAC:
            m = pat.search(line)
            if m:
                hits.append((no, f"{what} ({m.group(0).strip()})", line.strip()[:120]))
    return tuple(hits)


@dataclass
class Lock:
    roots: set[str]  # device roots by package name
    deps: dict[str, set[str]]  # package name -> the names it depends on, every version merged


def load_lock(root: str) -> Lock:
    import tomllib

    path = os.path.join(root, "Cargo.lock")
    if not os.path.isfile(path):
        raise RecipeError(f"no Cargo.lock in {root} — the device roots and the external closure are read from it")
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    roots: set[str] = set()
    deps: dict[str, set[str]] = {}
    for p in data.get("package", []):
        name = p["name"]
        if any(s in p.get("source", "") for s in DEVICE_GIT_SOURCES) or name in DEVICE_NAMED_ROOTS:
            roots.add(name)
        deps.setdefault(name, set()).update(d.split(" ", 1)[0] for d in p.get("dependencies", []))
    return Lock(roots, deps)


def device_chain(tree: Tree, lock: Lock, name: str) -> str | None:
    """The shortest chain from `name` to a device root, or None."""
    pkg = tree.packages[name]
    _, on, _ = tree.feature_closure(pkg, [], True, True)
    frontier: list[tuple[str, str]] = []
    seen: set[str] = set()
    ws = [(name, name)] + [(q, why) for q, why in tree.closure(pkg, [], True, True, True).items()]
    for q, why in ws:
        qp = tree.packages[q]
        for d in qp.deps:
            if d.pkg in tree.packages or (d.kind == "dev" and q != name) or (q == name and d.optional and d.key not in on):
                continue
            frontier.append((d.pkg, f"{why} -> {d.pkg}"))
    while frontier:
        nxt: list[tuple[str, str]] = []
        for d, why in frontier:
            if d in seen:
                continue
            seen.add(d)
            if d in lock.roots:
                return why
            nxt.extend((e, f"{why} -> {e}") for e in sorted(lock.deps.get(d, ())))
        frontier = nxt
    return None


def mac_purity(tree: Tree, lock: Lock, name: str) -> list[str]:
    """Why `cargo test -p <name>` cannot run natively on the Mac, one reason a line; empty when it can."""
    reasons: list[str] = []
    chain = device_chain(tree, lock, name)
    if chain is not None:
        reasons.append(f"reaches the device root {chain.rsplit(' -> ', 1)[1]}: {chain}")
    pkg = tree.packages[name]
    targets, errors = tree.resolve(Invocation(sub="test", oxide=False, packages=[name], all_features=True))
    if errors:
        raise RecipeError(f"pure-crates: {name}: " + "; ".join(errors))
    files: set[str] = set()
    for p, kind, tname, feats in targets:
        # a path literal names a file read at run time, not compiled; dep-info is the box's
        files |= {f for f, why in tree.target_files(p, kind, tname, feats, True, True).items() if "(path literal)" not in why and not why.startswith("dep-info")}
    own = pkg.dir.rstrip("/") + "/"
    hits = []
    for f in sorted(files):
        if not f.endswith(".rs") or not tree.exists(f):
            continue
        mine = f.startswith(own)
        for no, what, line in mac_hits(tree.read(f), None if mine else False):
            hits.append((not mine, f, no, what, line))
    hits.sort()
    if hits:
        _, f, no, what, line = hits[0]
        kinds = sorted({h[3].split(" (", 1)[0] for h in hits[1:]})
        more = f" [+{len(hits) - 1} more: {'; '.join(kinds)}]" if len(hits) > 1 else ""
        reasons.append(f"{f}:{no}: {what}: {line}{more}")
    return reasons


def pure_crates(tree: Tree, lock: Lock) -> list[tuple[str, list[str]]]:
    """Every workspace crate with its reasons, in name order; the pure ones have none."""
    return [(n, mac_purity(tree, lock, n)) for n in sorted(tree.packages)]


def cmd_pure_crates(args: argparse.Namespace) -> int:
    tree = Tree(ROOT)
    rows = pure_crates(tree, load_lock(tree.root))
    if args.names:
        for n, why in rows:
            if not why:
                print(n)
        return 0
    for n, why in rows:
        if not why:
            print(f"pure      {n}")
    for n, why in rows:
        for w in why:
            print(f"rejected  {n}  {w}")
    print(f"pure-crates: {sum(1 for _, w in rows if not w)} pure, {sum(1 for _, w in rows if w)} rejected, on {MAC_TARGET} (the rule: tools/recipes.py, section «pure crates»)")
    return 0


# ----------------------------------------------------------------------------------------------
# host nightly: the units a Linux host with no card and no model files runs (`host-nightly`)
# ----------------------------------------------------------------------------------------------
# The rule is the module docstring's «host nightly» paragraph; tools/nightly/run.sh runs what it prints.

# A device-linked crate the set takes anyway, with why: crate -> reason.
HOST_DEVICE_LINKED: dict[str, str] = {
    "bloomery-app": (
        "its default-feature tests open no card (gate-app is [group('host')], which host_problems reads); the device libs it links "
        "compile as host code under plain cargo and cuda-bindings loads libcuda only at run time; the deepseek41 and glm5next features "
        "add device crates and stay off"
    ),
}
# A test left out by exact libtest name, per crate: (name, reason). A non-ignored test that needs a file no host has.
HOST_TEST_SKIP: dict[str, list[tuple[str, str]]] = {}
# A `tools/check-*.sh` left out: script file name -> reason.
HOST_CHECK_SKIP: dict[str, str] = {}
# A unit that is no crate and no check script: name -> shell command, run from the tree's root.
HOST_EXTRA_UNITS: dict[str, str] = {
    "recipes-self-test": "python3 tools/recipes.py --self-test",
}
HOST_GATE = "bash tools/gate.sh --release --locked"


def host_crates(tree: Tree, lock: Lock, linked: dict[str, str] | None = None) -> list[tuple[str, str]]:
    """(crate, why it is in the set) in name order: every crate with no device root in its closure, and the admitted ones."""
    linked = HOST_DEVICE_LINKED if linked is None else linked
    rows = []
    for n in sorted(tree.packages):
        chain = device_chain(tree, lock, n)
        if chain is None:
            rows.append((n, "no device root in its closure"))
        elif n in linked:
            rows.append((n, f"{linked[n]} [{chain}]"))
    return rows


def host_left_out(tree: Tree, lock: Lock, linked: dict[str, str] | None = None) -> list[tuple[str, str]]:
    """(crate, the chain to the device root) for every crate the set does not take."""
    linked = HOST_DEVICE_LINKED if linked is None else linked
    return [(n, c) for n in sorted(tree.packages) if (c := device_chain(tree, lock, n)) is not None and n not in linked]


def host_check_scripts(root: str) -> list[str]:
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(root, "tools", "check-*.sh")))


def nightly_problems(
    tree: Tree,
    lock: Lock,
    linked: dict[str, str] | None = None,
    skips: dict[str, list[tuple[str, str]]] | None = None,
    checks: dict[str, str] | None = None,
    extras: dict[str, str] | None = None,
) -> list[str]:
    """The exclusion tables against the tree: a stale entry is a problem, never a no-op."""
    linked = HOST_DEVICE_LINKED if linked is None else linked
    skips = HOST_TEST_SKIP if skips is None else skips
    checks = HOST_CHECK_SKIP if checks is None else checks
    extras = HOST_EXTRA_UNITS if extras is None else extras
    problems: list[str] = []
    taken = {n for n in tree.packages} | {f[: -len(".sh")] for f in host_check_scripts(tree.root)}
    for n in sorted(extras):
        if n in taken:
            problems.append(f"host nightly: HOST_EXTRA_UNITS names {n}, which a crate or a check script already is")
    in_set = {n for n, _ in host_crates(tree, lock, linked)}
    for n in sorted(linked):
        if n not in tree.packages:
            problems.append(f"host nightly: HOST_DEVICE_LINKED names {n}, which is no workspace crate")
        elif device_chain(tree, lock, n) is None:
            problems.append(f"host nightly: HOST_DEVICE_LINKED names {n}, which reaches no device root (the rule already takes it)")
    for n in sorted(skips):
        if n not in in_set:
            problems.append(f"host nightly: HOST_TEST_SKIP names {n}, which the set does not run")
            continue
        text = "".join(tree.read(f) for f in sorted(_host_sources(tree, n)))
        for name, why in skips[n]:
            leaf = name.rsplit("::", 1)[-1]
            if not why.strip():
                problems.append(f"host nightly: HOST_TEST_SKIP {n} {name}: no reason")
            if not re.search(rf"\bfn\s+{re.escape(leaf)}\b", text):
                problems.append(f"host nightly: HOST_TEST_SKIP {n} {name}: no `fn {leaf}` in the crate's sources")
    present = set(host_check_scripts(tree.root))
    for n, why in sorted(checks.items()):
        if n not in present:
            problems.append(f"host nightly: HOST_CHECK_SKIP names {n}, which is not a tools/check-*.sh")
        if not why.strip():
            problems.append(f"host nightly: HOST_CHECK_SKIP {n}: no reason")
    return problems


def _host_sources(tree: Tree, crate: str) -> list[str]:
    """Every .rs file under the crate's src/ and tests/ (relative to the tree root)."""
    base = tree.packages[crate].dir
    out = []
    for sub in ("src", "tests"):
        for dp, _, fs in os.walk(os.path.join(tree.root, base, sub)):
            out += [os.path.relpath(os.path.join(dp, f), tree.root) for f in fs if f.endswith(".rs")]
    return out


def host_units(
    tree: Tree,
    lock: Lock,
    linked: dict[str, str] | None = None,
    skips: dict[str, list[tuple[str, str]]] | None = None,
    checks: dict[str, str] | None = None,
    extras: dict[str, str] | None = None,
) -> list[tuple[str, str]]:
    """(unit, shell command) in run order: the crates, the check scripts, the extra units. The command runs from the tree's root."""
    skips = HOST_TEST_SKIP if skips is None else skips
    checks = HOST_CHECK_SKIP if checks is None else checks
    extras = HOST_EXTRA_UNITS if extras is None else extras
    units = []
    for n, _ in host_crates(tree, lock, linked):
        cmd = f"{HOST_GATE} -p {n} --no-fail-fast"
        if skips.get(n):
            cmd += " -- --exact" + "".join(f" --skip {shlex.quote(name)}" for name, _ in skips[n])
        units.append((n, cmd))
    for f in host_check_scripts(tree.root):
        if f not in checks:
            units.append((f[: -len(".sh")], f"bash tools/{f}"))
    units += sorted(extras.items())
    return units


def cmd_host_nightly(args: argparse.Namespace) -> int:
    tree = Tree(ROOT)
    lock = load_lock(tree.root)
    problems = nightly_problems(tree, lock)
    if problems:
        raise RecipeError("\n".join(problems))
    if args.excluded:
        for n, chain in host_left_out(tree, lock):
            print(f"crate\t{n}\tdevice root: {chain}")
        for n, rows in sorted(HOST_TEST_SKIP.items()):
            for name, why in rows:
                print(f"test\t{n} {name}\t{why}")
        for n, why in sorted(HOST_CHECK_SKIP.items()):
            print(f"check\t{n}\t{why}")
        return 0
    for name, cmd in host_units(tree, lock):
        print(f"{name}\t{cmd}")
    return 0


# ----------------------------------------------------------------------------------------------
# build shapes: every (package, mode, features) a recipe compiles, for the Mac cross check (`combos`)
# ----------------------------------------------------------------------------------------------
# The rule is the module docstring's «build shapes» paragraph; tools/mac-check.sh combos runs the list.

# The recipes whose command tools/mac-check.sh runs as a mode of its own.
MAC_CHECKED = {"check": "tools/mac-check.sh check", "lint": "tools/mac-check.sh lint"}
_SELECTOR_ORDER = {"lib": 0, "bin": 1, "test": 2, "example": 3, "bench": 4}


@dataclass
class Combo:
    package: str
    test: bool  # a test build (cfg(test)): checked under `--profile test`
    enabled: tuple[str, ...]  # the package's enabled features, the closure with defaults: the shape's key
    others: tuple[str, ...]  # `dep/feature` words for other packages
    features: list[str]  # the package's own `--features` words, from the first call of this shape
    no_default: bool
    all_features: bool
    selectors: set[tuple[str, str | None]] = field(default_factory=set)
    recipes: list[str] = field(default_factory=list)

    def command(self) -> str:
        words = ["cargo", "check", "-p", self.package]
        if self.test:
            words += ["--profile", "test"]
        if self.all_features:
            words.append("--all-features")
        if self.no_default:
            words.append("--no-default-features")
        feats = list(self.features) + list(self.others)
        if feats and not self.all_features:
            words += ["--features", ",".join(feats)]
        for kind, name in sorted(self.selectors, key=lambda s: (_SELECTOR_ORDER[s[0]], s[1] or "")):
            words += [f"--{kind}"] + ([name] if name else [])
        return " ".join(words)

    def label(self) -> str:
        feats = ",".join(list(self.enabled) + list(self.others)) or "no features"
        shown = ", ".join(self.recipes[:3]) + (f", +{len(self.recipes) - 3}" if len(self.recipes) > 3 else "")
        return f"{self.package} {'test' if self.test else 'build'} [{feats}{', no defaults' if self.no_default else ''}]: {len(self.selectors)} targets, {len(self.recipes)} recipes ({shown})"


def _has_param(inv: Invocation) -> bool:
    return any("{{" in w for w in inv.packages + inv.features + [n or "" for _, n in inv.selectors])


def build_shapes(tree: Tree, recipes: dict[str, Recipe]) -> tuple[list[Combo], list[tuple[str, str]], int]:
    """(the shapes in package and mode order, [(recipe, why)] left out by name, the count of distinct
    (package, target, mode, features) tuples). A call the tree cannot resolve is a RecipeError naming it."""
    shapes: dict[tuple, Combo] = {}
    skips: list[tuple[str, str]] = []
    tuples: set[tuple] = set()
    errors: list[str] = []
    for rname in sorted(recipes):
        for inv in recipe_commands(recipes[rname]).invocations:
            call = " ".join(["cargo"] + ["oxide"] * inv.oxide + [inv.sub] + ["--workspace"] * inv.workspace + [f"-p {p}" for p in inv.packages] + [f"--features {','.join(inv.features)}"] * bool(inv.features))
            if rname in MAC_CHECKED:
                skips.append((rname, f"{call}: {MAC_CHECKED[rname]} runs this recipe's command"))
                continue
            if _has_param(inv):
                skips.append((rname, f"{call}: a recipe parameter ({{{{…}}}}) decides its package, features or targets at run time"))
                continue
            targets, errs = tree.resolve(inv)
            errors += [f"{rname}: {e}" for e in errs]
            for pname, kind, name, feats in targets:
                pkg = tree.packages[pname]
                enabled, _, _ = tree.feature_closure(pkg, feats, not inv.no_default, inv.all_features)
                others = tuple(sorted({f for f in inv.features if "/" in f and f.split("/", 1)[0] != pname}))
                if kind == "doctest":
                    skips.append((rname, f"doctests of {pname}: rustdoc compiles them and cargo check has no doctest mode; the lib they import is checked in {pname}'s build shape"))
                    kind = "lib"
                test = kind in ("libtest", "test", "bench") or (kind == "bin" and inv.sub in ("test", "bench"))
                sel = ("lib", None) if kind in ("lib", "libtest") else (kind, name)
                key = (pname, test, tuple(sorted(enabled)), others, inv.no_default, inv.all_features)
                c = shapes.get(key)
                if c is None:
                    c = shapes[key] = Combo(pname, test, key[2], others, sorted(feats), inv.no_default, inv.all_features)
                c.selectors.add(sel)
                if rname not in c.recipes:
                    c.recipes.append(rname)
                tuples.add(key + (sel,))
    if errors:
        raise RecipeError("combos: calls the tree cannot resolve (just check-recipes names them too): " + "; ".join(sorted(set(errors))[:8]))
    order = sorted(shapes, key=lambda k: (k[0], k[1], len(k[2]), k[2], k[3], k[4], k[5]))
    return [shapes[k] for k in order], skips, len(tuples)


# ----------------------------------------------------------------------------------------------
# the shape input key (`combos --base`/`--ledger`): what a round's loop may skip
# ----------------------------------------------------------------------------------------------

# Files every shape's check reads whatever its package: the workspace's dependency resolution, the
# toolchain pin and the two rustflags owners (a moved pin or flag re-checks every shape), and this
# tool and its runner — their sha moves every key when they change, the gate ledger's rule.
SHAPE_KEY_META = (
    "Cargo.toml",
    "Cargo.lock",
    "rust-toolchain.toml",
    ".cargo/config.toml",
    ".cargo/cuda-oxide.toml",
    "tools/recipes.py",
    "tools/mac-check.sh",
)

# The Mac static ledger: one `<key>\\tgreen\\t<YYYY-MM-DD HH:MM>\\t<tree>` line a green shape, appended
# by `tools/mac-check.sh combos --ledger`. The key is the inputs' content hash, so a green record is
# tree-independent — another tree at the same inputs reuses it.
def read_shape_ledger(path: str) -> tuple[dict[str, tuple[str, str]], int]:
    """({key: (date, tree)}, malformed lines) of the Mac static ledger; a missing file is empty."""
    green: dict[str, tuple[str, str]] = {}
    bad = 0
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for ln in fh:
                f = ln.rstrip("\n").split("\t")
                if len(f) == 4 and re.fullmatch(r"[0-9a-f]{64}", f[0]) and f[1] == "green":
                    green[f[0]] = (f[2], f[3])
                else:
                    bad += 1
    except FileNotFoundError:
        pass
    return green, bad


def shape_inputs(tree: Tree, c: Combo) -> set[str]:
    """Every file the shape's `cargo check` reads: each selector's files (its own package's target,
    lib and test-only modules, and every linked lib's), plus the manifests of its package and its
    dependency closure, plus SHAPE_KEY_META."""
    rels: set[str] = set()
    for kind, name in c.selectors:
        rels |= set(tree.target_files(c.package, kind, name, c.features, not c.no_default, c.all_features))
    pkg = tree.packages[c.package]
    rels.add(pkg.manifest)
    linked = tree.closure(pkg, c.features, dev=c.test, default=not c.no_default, all_features=c.all_features)
    rels |= {tree.packages[q].manifest for q in linked}
    rels |= set(SHAPE_KEY_META)
    return rels


def shape_key(c: Combo, rels: set[str], memo: dict[str, str]) -> str:
    """sha256 over the command and each input's (sha256, path), content-only (no mtimes): the same
    tree state gives the same key whatever copied or touched it. `memo` caches path -> sha256."""
    h = hashlib.sha256()
    h.update(f"mac-static-shape-v1\n{c.command()}\n".encode())
    for rel in sorted(rels):
        if rel not in memo:
            try:
                with open(os.path.join(ROOT, rel), "rb") as fh:
                    memo[rel] = hashlib.sha256(fh.read()).hexdigest()
            except OSError:
                memo[rel] = "missing"
        h.update(f"{memo[rel]}\t{rel}\n".encode())
    return h.hexdigest()


def cmd_combos(args: argparse.Namespace) -> int:
    if args.ledger and args.base is None:
        raise RecipeError("combos --ledger reads keys the --base form prints; give --base too")
    tree = Tree(ROOT)
    shapes, skips, n = build_shapes(tree, load_justfile(args.justfile or os.path.join(ROOT, "justfile")))
    scoped = args.base is not None
    green: dict[str, tuple[str, str]] = {}
    changed: set[str] = set()
    base_note = ""
    if scoped:
        try:
            files, a_rev, b_rev = changed_files(args.base)
        except RecipeError as err:
            raise RecipeError(f"combos --base {args.base}: {err}")
        changed = set(files)
        base_note = f" since {a_rev[:9]}" + ("" if b_rev is None else f"..{b_rev[:9]}")
        if args.ledger:
            green, bad = read_shape_ledger(args.ledger)
            if bad:
                print(f"recipes.py combos: {args.ledger}: {bad} malformed line(s) ignored (they cannot skip a shape)", file=sys.stderr)
    memo: dict[str, str] = {}
    run = kept_green = kept_same = 0
    for c in shapes:
        if not scoped:
            print(f"combo\t{c.label()}\t{c.command()}")
            run += 1
            continue
        rels = shape_inputs(tree, c)
        key = shape_key(c, rels, memo)
        if key in green:
            date, tname = green[key]
            print(f"skip\t{c.label()}\tgreen at {date} in {tname} (ledger {os.path.basename(args.ledger)})")
            kept_green += 1
            continue
        if not (changed & rels):
            print(f"skip\t{c.label()}\tno input file changed{base_note} ({len(rels)} inputs)")
            kept_same += 1
            continue
        print(f"combo\t{c.label()}\t{c.command()}\t{key}")
        run += 1
    for r, why in skips:
        print(f"skip\t{r}\t{why}")
    if scoped:
        print(f"total\t{run} of {len(shapes)} commands to run, {n} distinct (package, target, mode, features) tuples, "
              f"{kept_green} green in the ledger, {kept_same} with no input changed{base_note}, {len(skips)} left out by name")
    else:
        print(f"total\t{len(shapes)} commands, {n} distinct (package, target, mode, features) tuples, {len(skips)} left out by name")
    return 0


# ----------------------------------------------------------------------------------------------
# orphan tests: every test is run by some gate (`orphan-tests`)
# ----------------------------------------------------------------------------------------------
# The rule is the module docstring's «orphan tests» paragraphs; `just check-recipes` runs it.

# The recipes whose cargo test calls count as running a test. A lab-* recipe is a lab crate's own test
# runner (AGENTS.md: `just lab-engram`); it is left out of landing batches, not out of this count.
TEST_RUNNERS = ("gate-", "lab-")
# The box's target, x86_64-unknown-linux-gnu, where every gate-* recipe runs; any other key is unknown.
_BOX_CFG_KEYS = {
    "target_arch": "x86_64",
    "target_os": "linux",
    "target_family": "unix",
    "target_vendor": "unknown",
    "target_env": "gnu",
    "target_pointer_width": "64",
    "target_endian": "little",
}
_BOX_CFG_NAMES = {"unix": True, "windows": False}
# libtest's options. The ones that take a value have one owner, tools/gate.sh's `LIBTEST_VALUE=(…)` line
# (gate.sh skips their values when it looks for a call's filters, and runs on the box with no Python):
# this file reads that line; the flags are listed here. An option in neither set is a named error.
GATE_SH = "tools/gate.sh"
_LIBTEST_VALUE_LINE = re.compile(r"^LIBTEST_VALUE=\((.*)\)$")
_LIBTEST_OPTION = re.compile(r"^--?[A-Za-z][A-Za-z-]*$")
_LIBTEST_FLAG = {
    "--nocapture", "--no-capture", "--show-output", "--ignored", "--include-ignored", "--exact", "--list", "--test",
    "--bench", "-q", "--quiet", "--report-time", "--ensure-time", "--shuffle", "--force-run-in-process",
    "--exclude-should-panic",
}


def libtest_value_options(text: str, where: str = GATE_SH) -> frozenset[str]:
    """The libtest options that take a value, from gate.sh's text: its one `LIBTEST_VALUE=(…)` line."""
    lines = [m.group(1) for m in map(_LIBTEST_VALUE_LINE.match, text.splitlines()) if m]
    if len(lines) != 1:
        raise RecipeError(f"{where} has {len(lines)} `LIBTEST_VALUE=(…)` lines, not one: the libtest options that take a value are read from there")
    words = lines[0].split()
    bad = [w for w in words if not _LIBTEST_OPTION.match(w)]
    if not words or bad:
        raise RecipeError(f"{where}'s LIBTEST_VALUE is not a list of plain options: {lines[0]!r}")
    both = sorted(set(words) & _LIBTEST_FLAG)
    if len(set(words)) != len(words) or both:
        raise RecipeError(f"{where}'s LIBTEST_VALUE repeats an option or names a flag of _LIBTEST_FLAG: {both or words}")
    return frozenset(words)


@functools.cache
def _libtest_value() -> frozenset[str]:
    try:
        with open(os.path.join(ROOT, GATE_SH), encoding="utf-8") as fh:
            text = fh.read()
    except OSError as err:
        raise RecipeError(f"cannot read {GATE_SH}, the owner of the libtest value options: {err}") from err
    return libtest_value_options(text)


_WS = re.compile(r"\s*")
_ATTR_OPEN = re.compile(r"#\s*(!?)\s*\[")
_ATTR_PATH = re.compile(r"^((?:::)?[A-Za-z_][A-Za-z0-9_]*(?:\s*::\s*[A-Za-z_][A-Za-z0-9_]*)*)")
_VIS = r"(?:pub(?:\s*\([^)]*\))?\s+)?"
_ITEM_MOD = re.compile(rf"^{_VIS}(?:unsafe\s+)?mod\s+(?:r#)?([A-Za-z_][A-Za-z0-9_]*)$")
_ITEM_FN = re.compile(rf"^{_VIS}(?:(?:default|const|async|unsafe|safe)\s+|extern\s+(?:\"[^\"]*\"\s+)?)*fn\s+(?:r#)?([A-Za-z_][A-Za-z0-9_]*)")
# A test attribute anywhere in an item's text (on `shape`: comments and string contents blanked).
_TEST_ATTR = re.compile(r"#\s*\[\s*(?:[A-Za-z_][A-Za-z0-9_]*\s*::\s*)*(?:test|rstest|test_case)\b")


@dataclass
class TestItem:
    file: str
    line: int
    path: str  # the name libtest gives it: the module path from the target's root, then the fn
    cfgs: list[tuple[str, int, str, object]]  # (file, line, predicate text, parse_cfg tree) on its path, outermost first
    ignored: bool


@dataclass
class LibtestArgs:
    filters: list[str]
    skips: list[str]
    exact: bool
    ignored: str  # "plain" (an #[ignore]d test does not run) | "only" (--ignored) | "include" (--include-ignored)
    runs: bool  # False under --list or --bench: the binary runs no test

    def misses(self, path: str, ignored: bool) -> str | None:
        """None when this call runs the test named `path`, else why it does not."""
        if not self.runs:
            return "the call lists or benches, it runs no test"
        hit = (lambda f: path == f) if self.exact else (lambda f: f in path)
        if self.filters and not any(hit(f) for f in self.filters):
            return f"its filter {' '.join(self.filters)}{' (--exact)' if self.exact else ''} does not match {path}"
        skip = next((s for s in self.skips if hit(s)), None)
        if skip is not None:
            return f"its --skip {skip} matches {path}"
        if ignored and self.ignored == "plain":
            return "the test is #[ignore]d and the call has neither --ignored nor --include-ignored"
        if not ignored and self.ignored == "only":
            return "the call runs --ignored only and the test is not #[ignore]d"
        return None


def libtest_args(inv: Invocation) -> LibtestArgs:
    """The test binary's arguments of a cargo test call: `cargo test <filter>` and the words after `--`."""
    out = LibtestArgs(filters=list(inv.names), skips=[], exact=False, ignored="plain", runs=True)
    if any("{{" in w for w in inv.names):
        raise RecipeError(f"test filter `{' '.join(inv.names)}` is a recipe parameter: which tests it runs is known only at run time: {inv.raw}")
    words = list(inv.test_args)
    flags: set[str] = set()
    i = 0
    while i < len(words):
        w = words[i]
        opt, val = (w.split("=", 1) + [None])[:2] if w.startswith("--") and "=" in w else (w, None)
        if "{{" in w:
            raise RecipeError(f"test argument `{w}` is a recipe parameter: which tests it runs is known only at run time: {inv.raw}")
        if opt in _libtest_value():
            if val is None:
                i += 1
                if i >= len(words):
                    raise RecipeError(f"libtest option `{opt}` without a value in: {inv.raw}")
                val = words[i]
            if opt == "--skip":
                out.skips.append(val)
        elif opt in _LIBTEST_FLAG:
            flags.add(opt)
        elif w.startswith("-"):
            raise RecipeError(f"libtest option `{w}` is not known to tools/recipes.py (add it to _LIBTEST_FLAG, or to tools/gate.sh's LIBTEST_VALUE if it takes a value): {inv.raw}")
        else:
            out.filters.append(w)
        i += 1
    if "--ignored" in flags and "--include-ignored" in flags:
        raise RecipeError(f"--ignored and --include-ignored together, which libtest refuses: {inv.raw}")
    out.ignored = "only" if "--ignored" in flags else ("include" if "--include-ignored" in flags else "plain")
    out.exact = "--exact" in flags
    out.runs = not ({"--list", "--bench"} & flags)
    return out


def _close(shape: str, at: int) -> int | None:
    """The index of the bracket that closes the one at `at` (every bracket kind counted as one)."""
    depth = 0
    for j in range(at, len(shape)):
        c = shape[j]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                return j
    return None


def _split_top(text: str) -> list[str]:
    """`text` split at its commas outside brackets and string literals."""
    items, depth, cur, quote = [], 0, "", False
    i = 0
    while i < len(text):
        ch = text[i]
        if quote:
            cur += ch
            if ch == "\\" and i + 1 < len(text):
                cur += text[i + 1]
                i += 1
            elif ch == '"':
                quote = False
        elif ch == '"':
            quote, cur = True, cur + ch
        elif ch == "," and depth == 0:
            items.append(cur.strip())
            cur = ""
        else:
            depth += (ch in "([{") - (ch in ")]}")
            cur += ch
        i += 1
    if cur.strip():
        items.append(cur.strip())
    return items


class TestScan:
    """Every `#[test]` fn of one target's module tree, with the cfgs on its path and its #[ignore]; what
    the scan cannot read goes to `errors` by file and line, never into a guess."""

    def __init__(self, tree: Tree):
        self.tree = tree
        self._lex: dict[str, tuple[str, str, list[int]]] = {}
        self._cache: dict[str, tuple[list[TestItem], list[str]]] = {}
        # per target root: the test-only code, (file, start, end) — a module under its own cfg that names
        # `test`, inline (its braces' inside) or a file (0, None)
        self._blocks: dict[str, list[tuple[str, int, int | None]]] = {}
        self._blocks_cur: list[tuple[str, int, int | None]] | None = None

    def lexed(self, rel: str) -> tuple[str, str, list[int]]:
        if rel not in self._lex:
            code, shape = rust_lex(self.tree.read(rel))
            starts = [0] + [m.end() for m in re.finditer("\n", shape)]
            self._lex[rel] = (code, shape, starts)
        return self._lex[rel]

    def tests_of(self, src: str) -> tuple[list[TestItem], list[str]]:
        if src not in self._cache:
            out: list[TestItem] = []
            errs: list[str] = []
            self._blocks_cur = []
            self._file(src, os.path.dirname(src), [], [], out, errs, ())
            self._blocks[src], self._blocks_cur = self._blocks_cur, None
            self._cache[src] = (out, errs)
        return self._cache[src]

    def test_blocks(self, src: str) -> list[tuple[str, int, int | None]]:
        self.tests_of(src)
        return self._blocks[src]

    def _file(self, rel: str, moddir: str, modpath: list[str], cfgs: list, out: list[TestItem], errs: list[str], stack: tuple[str, ...]) -> None:
        if rel in stack:
            errs.append(f"{rel}:1: a module cycle ({' -> '.join(stack + (rel,))})")
            return
        code, shape, _ = self.lexed(rel)
        self._items(rel, 0, len(shape), moddir, [], modpath, cfgs, out, errs, stack + (rel,))

    def line_of(self, rel: str, at: int) -> int:
        return bisect.bisect_right(self.lexed(rel)[2], at)

    def _items(self, rel: str, pos: int, end: int, moddir: str, inner: list[str], modpath: list[str], cfgs: list, out: list[TestItem], errs: list[str], stack: tuple[str, ...]) -> None:
        code, shape, _ = self.lexed(rel)

        def line_of(at: int) -> int:
            return self.line_of(rel, at)

        attrs: list[tuple[str, int]] = []
        cfgs = list(cfgs)
        while True:
            pos = _WS.match(shape, pos).end()
            if pos >= end:
                break
            m = _ATTR_OPEN.match(shape, pos)
            if m:
                close = _close(shape, m.end() - 1)
                if close is None or close >= end:
                    errs.append(f"{rel}:{line_of(pos)}: an attribute with no closing bracket")
                    return
                text = " ".join(code[m.end() : close].split())
                if m.group(1):
                    am = _ATTR_PATH.match(text)
                    name = re.sub(r"\s+", "", am.group(1)) if am else ""
                    if name == "cfg":
                        pred = self._cfg_of(rel, line_of(pos), text, errs)
                        if pred is None:
                            return
                        cfgs.append(pred)
                    elif name == "cfg_attr":
                        errs.append(f"{rel}:{line_of(pos)}: #![{text}] — a conditional inner attribute this checker does not evaluate")
                        return
                else:
                    attrs.append((text, line_of(pos)))
                pos = close + 1
                continue
            j, depth, body, semi = pos, 0, None, False
            while j < end:
                c = shape[j]
                if c in "([":
                    depth += 1
                elif c in ")]":
                    depth -= 1
                elif c == "{":
                    if depth == 0:
                        close = _close(shape, j)
                        if close is None or close >= end:
                            errs.append(f"{rel}:{line_of(j)}: a brace with no close")
                            return
                        body = (j + 1, close)
                        j = close + 1
                        break
                    depth += 1
                elif c == "}":
                    depth -= 1
                elif c == ";" and depth == 0:
                    semi = True
                    j += 1
                    break
                j += 1
            head_end = body[0] - 1 if body else (j - 1 if semi else j)
            head = " ".join(code[pos:head_end].split())
            if head:
                self._item(rel, (pos, j), head, body, attrs, moddir, inner, modpath, cfgs, out, errs, stack)
            attrs = []
            pos = j

    def _cfg_of(self, rel: str, line: int, text: str, errs: list[str]):
        """(file, line, text, tree) of a `cfg(…)` attribute's predicate, or None (errs names it)."""
        inside = text[len("cfg") :].strip()
        if not (inside.startswith("(") and inside.endswith(")")):
            errs.append(f"{rel}:{line}: #[{text}] is not a cfg(…) this checker reads")
            return None
        pt = inside[1:-1].strip()
        tree = parse_cfg(pt)
        if tree is None:
            errs.append(f"{rel}:{line}: cfg({pt}) is not a predicate this checker reads")
            return None
        return (rel, line, pt, tree)

    def _item(self, rel: str, span: tuple[int, int], head: str, body, attrs: list[tuple[str, int]], moddir: str, inner: list[str], modpath: list[str], cfgs: list, out: list[TestItem], errs: list[str], stack: tuple[str, ...]) -> None:
        """One item after its outer attributes: `span` is its text (head and body), `body` its braces' inside."""
        _, shape, _ = self.lexed(rel)
        line = self.line_of(rel, span[0])
        here = list(cfgs)
        test = ignored = False
        path_attr = None
        for text, aline in attrs:
            m = _ATTR_PATH.match(text)
            name = re.sub(r"\s+", "", m.group(1)) if m else ""
            if name == "cfg":
                pred = self._cfg_of(rel, aline, text, errs)
                if pred is None:
                    return
                here.append(pred)
            elif name == "cfg_attr":
                inside = text[len("cfg_attr") :].strip()
                parts = _split_top(inside[1:-1]) if inside.startswith("(") and inside.endswith(")") else []
                named = [re.sub(r"\s+", "", pm.group(1)) if (pm := _ATTR_PATH.match(p)) else p for p in parts[1:]]
                if not parts or any(n in ("cfg", "ignore", "path", "test") or n.endswith("::test") for n in named):
                    errs.append(f"{rel}:{aline}: #[{text}] — a conditional cfg, ignore, path or test attribute this checker does not evaluate")
                    return
            elif name == "test":
                if text != "test":
                    errs.append(f"{rel}:{aline}: #[{text}] is not a plain #[test] this checker reads")
                    return
                test = True
            elif name == "ignore":
                ignored = True
            elif name == "path":
                pm = re.match(r'^path\s*=\s*"([^"]*)"$', text)
                if pm is None:
                    errs.append(f"{rel}:{aline}: #[{text}] is not a #[path = \"…\"] this checker reads")
                    return
                path_attr = pm.group(1)
            elif name.endswith("::test") or name in ("rstest", "test_case"):
                errs.append(f"{rel}:{aline}: #[{text}] — a test attribute from a macro crate, whose expansion this checker does not see")
                return
        mm = _ITEM_MOD.match(head)
        if mm:
            name = mm.group(1)
            test_only = any(re.search(r"\btest\b", c[2]) for c in here[len(cfgs):])
            if body is None:
                hit, child_dir, err = self.tree.mod_file(rel, moddir, inner, name, path_attr)
                if hit is None:
                    errs.append(f"{rel}:{line}: {err}")
                    return
                if test_only and self._blocks_cur is not None:
                    self._blocks_cur.append((hit, 0, None))
                self._file(hit, child_dir, modpath + [name], here, out, errs, stack)
            else:
                if test_only and self._blocks_cur is not None:
                    self._blocks_cur.append((rel, body[0], body[1]))
                self._items(rel, body[0], body[1], moddir, inner + [name], modpath + [name], here, out, errs, stack)
            return
        fm = _ITEM_FN.match(head)
        nested = _TEST_ATTR.search(shape, span[0], span[1])
        if fm and test:
            out.append(TestItem(rel, line, "::".join(modpath + [fm.group(1)]), here, ignored))
            if nested:
                errs.append(f"{rel}:{self.line_of(rel, nested.start())}: a test attribute inside the test fn {fm.group(1)} — an inner item libtest cannot name")
            return
        if test:
            errs.append(f"{rel}:{line}: #[test] on `{head[:60]}`, which is not a fn this checker reads")
        elif nested:
            errs.append(f"{rel}:{self.line_of(rel, nested.start())}: a test attribute inside `{head[:60]}` — a macro or a nested item this checker does not read")
        elif re.match(r"^include\s*!", head):
            errs.append(f"{rel}:{line}: `{head[:60]}` at module level — the items it includes are not read")


@dataclass
class TestRun:
    recipe: str
    features: set[str] | None  # None: a feature from a recipe parameter, known only at run time
    args: LibtestArgs


def orphan_tests(tree: Tree, recipes: dict[str, Recipe], prefixes: tuple[str, ...] = TEST_RUNNERS) -> tuple[list[str], list[str], int]:
    """(orphans, errors, tests checked): every `#[test]` of every workspace target that no cargo test call of
    a recipe named with `prefixes` runs on the box, one `path:line name — why` line each; and every input
    it cannot decide, by file and line."""
    runs: dict[tuple[str, str, str | None], list[TestRun]] = {}
    errors: list[str] = []
    for rname in sorted(recipes, key=lambda n: (recipes[n].line, n)):
        if not rname.startswith(prefixes):
            continue
        try:
            invocations = recipe_commands(recipes[rname]).invocations
        except RecipeError as err:
            errors.append(f"justfile:{recipes[rname].line} {rname}: {err}")
            continue
        for inv in invocations:
            if inv.sub != "test":
                continue
            try:
                args = libtest_args(inv)
            except RecipeError as err:
                errors.append(f"justfile:{recipes[rname].line} {rname}: {err}")
                continue
            targets, _ = tree.resolve(inv)  # a target the call names wrongly is check()'s error, not this one's
            param = any("{{" in f for f in inv.features)
            for pname, kind, tname, feats in targets:
                if kind == "libtest":
                    key = (pname, "lib", None)
                elif kind in ("bin", "test"):
                    key = (pname, kind, tname)
                else:
                    # doctests are not #[test] fns; an example's or a bench's tests run only with `test = true`
                    continue
                enabled = None if param else tree.feature_closure(tree.packages[pname], feats, not inv.no_default, inv.all_features)[0]
                runs.setdefault(key, []).append(TestRun(rname, enabled, args))
    scan = TestScan(tree)
    orphans: list[str] = []
    checked = 0
    for pname in sorted(tree.packages):
        for t in tree.packages[pname].targets:
            if t.kind == "build":
                continue
            key = (pname, t.kind, None if t.kind == "lib" else t.name)
            label = f"{pname} {t.kind}" + ("" if t.kind == "lib" else f" {t.name}")
            items, errs = scan.tests_of(t.src)
            errors.extend(errs)
            for item in items:
                checked += 1
                cands = runs.get(key, [])
                if not cands:
                    orphans.append(f"{item.file}:{item.line} {item.path} — {label}: no {' or '.join(p + '*' for p in prefixes)} recipe runs this target's tests")
                    continue
                reasons: list[str] = []
                unknown: list[str] = []
                for run in cands:
                    bad = None
                    for f, ln, text, pred in item.cfgs:
                        v = eval_cfg(pred, True, _BOX_CFG_KEYS, _BOX_CFG_NAMES, run.features)
                        if v is None:
                            unknown.append(f"cfg({text}) at {f}:{ln} under {run.recipe}")
                            bad = "?"
                            break
                        if v is False:
                            feats = "no feature" if not run.features else "features " + ",".join(sorted(run.features))
                            bad = f"{run.recipe} ({feats}): cfg({text}) at {f}:{ln} is false"
                            break
                    if bad is None:
                        miss = run.args.misses(item.path, item.ignored)
                        if miss is None:
                            break
                        bad = f"{run.recipe}: {miss}"
                    if bad != "?" and bad not in reasons:
                        reasons.append(bad)
                else:
                    if unknown:
                        errors.append(f"{item.file}:{item.line} {item.path}: cannot evaluate {unknown[0]}")
                    else:
                        orphans.append(f"{item.file}:{item.line} {item.path} — {label}: " + "; ".join(reasons))
    return orphans, list(dict.fromkeys(errors)), checked  # a module several targets include is read once per target


def cmd_orphan_tests(args: argparse.Namespace) -> int:
    tree = Tree(ROOT)
    recipes = load_justfile(args.justfile or os.path.join(ROOT, "justfile"))
    orphans, errors, checked = orphan_tests(tree, recipes)
    for o in orphans:
        print(o)
    for e in errors:
        print(f"orphan-tests: cannot decide: {e}")
    if errors:
        print(f"orphan-tests: {len(errors)} inputs the checker cannot read, {len(orphans)} of {checked} tests no gate-* or lab-* recipe runs")
        return 2
    if orphans:
        print(f"orphan-tests: {len(orphans)} of {checked} tests no gate-* or lab-* recipe runs")
        return 1
    print(f"orphan-tests: ok ({checked} tests, each run by a gate-* or lab-* recipe)")
    return 0


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
#   manifest  a package manifest the closure reads by table: path, sha256 of the item's view of it
#             (manifest_view: the shared tables and its own targets' entries, canonical JSON — a comment
#             or another target's entry does not move it); one read whole is a `file` part
#   global    Cargo.toml, Cargo.lock, rust-toolchain.toml, .cargo/*, the cuda-oxide pin rev
#   item      ARGS, the item's full BLOOMERY_BOX_ENV (the caller's, then the item's, then the lane's
#             card), the card that env forces
#   mac-env   the Mac-side variables tools/box.sh carries or reads (BLOOMERY_REMOTE is not one: it
#             names a track's directory, not an input) — among them BLOOMERY_TIER, unset and `real` the one value
#             `real`, so a fixture-tier green never stands for the real file's — and the versions of just and python3
#   box       every line of the box manifest (`box-manifest`, fetched once per batch), its model rows
#             only for the model directories the item can open (KeyContext.model_scope); the fixture
#             directory's rows (FIXTURE_ROOT, every file under it) only for an item in the fixture tier
# Not in it: the binary (its paths are per track) and the commit.

KEY_VERSION = "bloomery-gate-key 1"
MANIFEST_VERSION = "box-manifest 1"
MAC_ENV = ("BLOOMERY_BOX", "BLOOMERY_CARD", "BLOOMERY_DATA", "BLOOMERY_MODEL", "BLOOMERY_REF_MODEL", "BLOOMERY_TIER", "BLOOMERY_V41_MODEL")
# The fixture tier's files (tools/ref/ref-paths.sh: one directory a family under it). Its rows belong to the keys of fixture-tier
# items only: a real item must not rerun because a fixture was regenerated, and a fixture item must, whatever a fixture file's
# directory is called — so the box manifest lists every file under it, subdirectories included, and no file's literal
# brings it into a real item's key (KeyContext.literal_dirs).
FIXTURE_ROOT = "/models/fixtures"


def tier_of(values: list[str], mac: str | None) -> str:
    """The tier an item runs in: the last BLOOMERY_TIER=<t> entry of `values` (the box env, then the item's env), else the
    Mac's BLOOMERY_TIER; unset and empty are `real`, as in tools/box.sh. A value that is neither is returned as it is."""
    tier = mac or "real"
    for e in values:
        k, _, v = e.partition("=")
        if k == "BLOOMERY_TIER":
            tier = v or "real"
    return tier
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
# $BLOOMERY_DATA entries left out of the manifest because no current item reads them: the tokenizer*/ directories
# of the format before the trailer, which a checkout that has not rebased past it still rewrites on every
# gate-tokenizer run (the sets the gate reads now are ref-tokenizer-*, written by `just dump-ref-tokenizer`, and
# are in the manifest). In the manifest the old directories would move every item's key whenever such a checkout
# ran its gate. The self-test fails when a gate-* recipe's inputs name one; the entry goes when every checkout has
# rebased.
DATA_NEVER_ONLY = re.compile(r"tokenizer(?:-[a-z0-9_]+)?")
DATA_NEVER_ONLY_NAMED = re.compile(r"BLOOMERY_DATA\}?\"?/tokenizer|TOKENIZER_SET|set:\s*\"tokenizer")
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
# The form on a tools/gpu-gate.sh call in a script, as tools/gate-batch.sh's walk reads a lane from it.
ANY_CARD_CALL = re.compile(re.escape(ANY_CARD) + r"\s+(?:bash\s+|exec\s+)?[^\s;&|#'\"]*gpu-gate\.sh\b")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
# A /models path as a literal: not the tail of a longer path (`tools/ref/models/x.sh`, `$HOME/models`,
# `${D}/models`, `$(pwd)/models`), but after a quote, `=`, a space, the `-` of a `${V:-/models/…}` default,
# or a `\t`/`\n`/`\r` escape in a string.
MODEL_LITERAL = re.compile(r"(?:(?<![A-Za-z0-9._+/~})])|(?<=\\[tnr]))/models/[A-Za-z0-9._+-]+(?:/[A-Za-z0-9._+-]+)*")
# The files both sides read for model literals: the box manifest's model_dirs() and an item's key.
MODEL_SCAN_EXT = (".rs", ".sh", ".py", ".toml", ".tsv")


def scans_for_models(rel: str) -> bool:
    return rel == "justfile" or rel.endswith(MODEL_SCAN_EXT)


def model_dir_of(path: str) -> str | None:
    """The model directory /models/<name> a /models path lies in; None for /models itself or a path
    outside it (a literal naming a file right under /models is its own entry)."""
    f = path.split("/")
    if len(f) < 3 or f[:2] != ["", "models"] or not f[2]:
        return None
    return "/".join(f[:3])


def model_literal_dirs(text: str) -> set[str]:
    return {d for d in map(model_dir_of, MODEL_LITERAL.findall(text)) if d}


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


# ----------------------------------------------------------------------------------------------
# opens: the recipes whose cargo targets reach the fixture fence (tools/gate-batch.sh's real-file deferral)
# ----------------------------------------------------------------------------------------------
# The fixture tier's fence is the fn `check_file` of FENCE_PKG (Tier::check_file, crates/gpu-gates/src/tier.rs): a run under
# BLOOMERY_TIER=fixture that opens a model file ends in a named refusal unless the file is a fixture. A family whose fixture-table
# row is `self` (tools/ref/ref-paths.sh) runs on its real file, so a recipe of such a family whose targets reach the fence is
# refused at run time; `opens RECIPE…` says which, from the code.
#
#   - The fence is the fn FENCE_FN defined in FENCE_PKG's own files. None found is a named error (70), never "reaches nothing".
#     Only FENCE_PKG's files are read: a workspace package that depends on it would reach the fence through it, which this
#     reader does not follow, so such a package is a named error too.
#   - The openers are the fns of FENCE_PKG's lib module tree (test-only items left out, the definer's file left out) whose body
#     calls the fence or another opener, closed to a fixpoint. A call is matched by the fn's identity, not its bare name: a free
#     fn by `name(` or `path::name(`; a fn of an `impl T` or `trait T` block by `T::name(` (generics between the segments
#     allowed) or `Self::name(` inside that block, and by `.name(` only when its first parameter is `self`. `File::open(` is
#     therefore not a call of `Generator::open`.
#   - A target reaches the fence when its own module tree has a call of the fence or of an opener on a non-comment line. What
#     the tree is depends on what the command does with the target. A bin, example or bench that is built (`cargo build`, `run`;
#     a runner executes it, or a process of the recipe spawns it) is its module tree, the files `Tree.walk` gives. The same under
#     `cargo test` or tools/gate.sh runs its tests only: the files only a test build reads. An integration test target under
#     `cargo test` is its whole tree; the lib under `cargo test` is the files only its test build reads; a built test or lib is
#     not executed. Comments and string and char literals are blanked first (`rust_lex`).
#   - The targets are those of the recipe and its dependencies that a command builds or runs. `check`, `clippy` and `doc` compile
#     and run nothing, so a recipe none of whose commands runs a target (`cargo test|run|bench`, tools/gate.sh, a bin runner)
#     reaches nothing, whatever it compiles.
#
# What the scan cannot see: a macro body or a call spelt by a macro; a call through a type alias, a `use … as` rename, a trait
# object or a generic parameter (`B::open(`); a fn stored in a variable or a closure and called elsewhere; a test run through an
# inline `#[cfg(test)]` module, which a test run reads no more than its test files; a process the target spawns that is not
# built by the same recipe; a feature or target `cfg` (every `mod` the tree names is read).

FENCE_PKG = "bloomery-gpu-gates"
FENCE_FN = "check_file"
OPENS_RUNS = frozenset({"test", "run", "bench"})
OPENS_BUILDS = frozenset({"build", "test", "run", "bench", "rustc"})
_ITEM_BLOCK = re.compile(rf"^{_VIS}(?:unsafe\s+)?(impl|trait)\b")
_SELF_PARAM = re.compile(r"\(\s*(?:&\s*(?:'[A-Za-z_]\w*\s+)?)?(?:mut\s+)?self\b")
_CALL = re.compile(
    r"(?P<dot>\.\s*)?(?:(?P<q>[A-Za-z_]\w*)\s*(?:::\s*<[^()]*?>\s*)?::\s*)?(?P<n>[A-Za-z_]\w*)\s*(?:::\s*<[^()]*?>\s*)?\("
)
_AFTER_FN = re.compile(r"\bfn\s*$")


class OpensError(Exception):
    """A failure of `opens`, named, with the exit code it ends the command with."""

    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code = code


@dataclass(frozen=True)
class FnDef:
    name: str
    owner: str | None  # the type of the `impl` block (the name of the `trait` block) the fn sits in; None for a free fn
    method: bool  # its first parameter is a `self` receiver
    path: str
    line: int
    body: tuple[int, int]  # the braces' inside, as offsets in the file's `shape`


def _line_at(shape: str, at: int) -> int:
    return shape.count("\n", 0, at) + 1


def _angle_close(text: str, at: int) -> int:
    """The index of the `>` that closes the `<` at `at` (`->` does not close), or -1."""
    depth = 0
    for j in range(at, len(text)):
        c = text[j]
        if c == "<":
            depth += 1
        elif c == ">" and text[j - 1] != "-":
            depth -= 1
            if depth == 0:
                return j
    return -1


def _is_method(rest: str) -> bool:
    """Whether the parameters after a fn's name (`rest`: its generics, then `(…)`) open with a `self` receiver."""
    rest = rest.lstrip()
    if rest.startswith("<"):
        end = _angle_close(rest, 0)
        rest = rest[end + 1 :].lstrip() if end >= 0 else ""
    return _SELF_PARAM.match(rest) is not None


def _block_owner(head: str) -> str | None:
    """The type an `impl` header implements (after `for` when it has one), the name of a `trait` header."""
    m = _ITEM_BLOCK.match(head)
    rest = head[m.end() :].strip()
    if m.group(1) == "trait":
        t = re.match(r"([A-Za-z_]\w*)", rest)
        return t.group(1) if t else None
    if rest.startswith("<"):
        end = _angle_close(rest, 0)
        rest = rest[end + 1 :].strip() if end >= 0 else ""
    rest = re.split(r"\bwhere\b", rest)[0]
    for f in re.finditer(r"\bfor\b", rest):
        if rest.count("<", 0, f.start()) == rest.count(">", 0, f.start()):
            rest = rest[f.end() :]
            break
    t = re.match(r"[&*\s]*(?:'[A-Za-z_]\w*\s+)?(?:(?:mut|const|dyn)\s+)*([A-Za-z_]\w*(?:\s*::\s*[A-Za-z_]\w*)*)", rest)
    return re.split(r"\s*::\s*", t.group(1))[-1] if t else None


def _scan_items(rel: str, code: str, shape: str, pos: int, end: int, owner: str | None, tests: bool, out: list[FnDef]) -> None:
    """The fns with a body among the items of `shape[pos:end]`, each with the `impl` or `trait` block it sits in; modules are
    entered, fn bodies are not (a nested fn's calls are its parent's). Items that only a test build compiles (`#[cfg(test)]`,
    `#[test]`) are left out unless `tests`. A brace or an attribute with no close is a named error (70)."""
    test_attr = False
    while True:
        pos = _WS.match(shape, pos).end()
        if pos >= end:
            return
        m = _ATTR_OPEN.match(shape, pos)
        if m:
            close = _close(shape, m.end() - 1)
            if close is None or close >= end:
                raise OpensError(70, f"{rel}:{_line_at(shape, pos)}: an attribute with no closing bracket")
            text = " ".join(code[m.end() : close].split())
            is_test = text == "test" or Tree.cfg_test(f"#[{text}]")
            if m.group(1):
                if is_test and not tests:
                    return
            else:
                test_attr = test_attr or is_test
            pos = close + 1
            continue
        j, depth, body, semi = pos, 0, None, False
        while j < end:
            c = shape[j]
            if c in "([":
                depth += 1
            elif c in ")]":
                depth -= 1
            elif c == "{":
                if depth == 0:
                    close = _close(shape, j)
                    if close is None or close >= end:
                        raise OpensError(70, f"{rel}:{_line_at(shape, j)}: a brace with no close")
                    body = (j + 1, close)
                    j = close + 1
                    break
                depth += 1
            elif c == "}":
                depth -= 1
            elif c == ";" and depth == 0:
                semi = True
                j += 1
                break
            j += 1
        head_end = body[0] - 1 if body else (j - 1 if semi else j)
        head = " ".join(code[pos:head_end].split())
        if head and body and (tests or not test_attr):
            fm = _ITEM_FN.match(head)
            if fm:
                out.append(FnDef(fm.group(1), owner, _is_method(head[fm.end() :]), rel, _line_at(shape, pos), body))
            elif _ITEM_BLOCK.match(head):
                _scan_items(rel, code, shape, body[0], body[1], _block_owner(head), tests, out)
            elif _ITEM_MOD.match(head):
                _scan_items(rel, code, shape, body[0], body[1], None, tests, out)
        test_attr = False
        pos = j


def _fn_defs(tree: Tree, rel: str, tests: bool) -> list[FnDef]:
    code, shape = rust_lex(tree.read(rel))
    out: list[FnDef] = []
    _scan_items(rel, code, shape, 0, len(shape), None, tests, out)
    return out


def _calls(shape: str, a: int, b: int):
    """(offset, name, form, qualifier) of each call in `shape[a:b]`: form `bare` (`n(`), `path` (`q::n(`, the last two segments) or
    `method` (`.n(`); the definition `fn n(` is no call. `shape` has its comments and literals blanked (`rust_lex`)."""
    for m in _CALL.finditer(shape, a, b):
        if _AFTER_FN.search(shape, max(a, m.start() - 16), m.start()):
            continue
        form = "method" if m.group("dot") else ("path" if m.group("q") else "bare")
        yield m.start("n"), m.group("n"), form, m.group("q")


def _opener_called(o: FnDef, form: str, qual: str | None, caller: str | None) -> bool:
    if form == "method":
        return o.method
    if form == "path":
        return o.owner is None or qual == o.owner or (qual == "Self" and caller is not None and caller == o.owner)
    return o.owner is None


@dataclass
class Fence:
    definers: list[FnDef]
    openers: dict[tuple[str, int], tuple[FnDef, str]]  # (path, line) -> (fn, the callee it reaches the fence through)

    def reaches(self, name: str, form: str, qual: str | None, caller: str | None) -> bool:
        if name == FENCE_FN:
            return True
        return any(_opener_called(o, form, qual, caller) for (o, _) in self.openers.values() if o.name == name)


def opens_fence(tree: Tree) -> Fence:
    """The fence, its definers and the openers of FENCE_PKG's lib (the rule above); the named errors (70) when it cannot be read."""
    pkg = tree.packages.get(FENCE_PKG)
    if pkg is None or pkg.lib() is None:
        raise OpensError(70, f"no workspace package {FENCE_PKG} with a lib: the fixture fence is read there")
    users = sorted(p.name for p in tree.packages.values() if p.name != FENCE_PKG and any(d.pkg == FENCE_PKG for d in p.deps))
    if users:
        raise OpensError(70, f"{', '.join(users)} depend on {FENCE_PKG}: a target of theirs reaches the fence through it, and this reader scans {FENCE_PKG}'s files only")
    lib_files = sorted(f for f in tree.tree_of(pkg.lib().src)[0] if f.endswith(".rs"))
    definers = [d for f in lib_files for d in _fn_defs(tree, f, False) if d.name == FENCE_FN]
    if not definers:
        raise OpensError(70, f"no `fn {FENCE_FN}` in {FENCE_PKG}'s lib: the fixture fence is gone or renamed, and this reader would call every recipe clear")
    skip = {d.path for d in definers}
    fns = [(d, rust_lex(tree.read(d.path))[1]) for f in lib_files if f not in skip for d in _fn_defs(tree, f, False)]
    fence = Fence(definers, {})
    while True:
        found = {}
        for d, shape in fns:
            if (d.path, d.line) in fence.openers:
                continue
            for _, name, form, qual in _calls(shape, *d.body):
                if fence.reaches(name, form, qual, d.owner):
                    found[(d.path, d.line)] = (d, name)
                    break
        if not found:
            return fence
        fence.openers.update(found)


def opens_targets(tree: Tree, recipes: dict[str, Recipe], name: str) -> list[tuple[str, str, str | None, str]]:
    """(package, kind, target, mode) of FENCE_PKG that `name` and its dependencies build or run, mode `test` for a `cargo test` or
    `bench` command (tools/gate.sh) and `run` for the others; [] when no command of them runs a target."""
    runs, invs = False, []
    for n in recipe_closure(recipes, name):
        try:
            rc = recipe_commands(recipes[n])
        except RecipeError as err:
            raise OpensError(70, f"recipe {n}: {err}") from err
        runs = runs or bool(rc.runs) or any(i.sub in OPENS_RUNS for i in rc.invocations)
        invs += [i for i in rc.invocations if i.sub in OPENS_BUILDS]
    out: set[tuple[str, str, str | None, str]] = set()
    for inv in invs if runs else []:
        targets, errs = tree.resolve(inv)
        if errs:
            raise OpensError(70, f"recipe {name}: {errs[0]}")
        mode = "test" if inv.sub in ("test", "bench") else "run"
        out |= {(p, k, t, mode) for (p, k, t, _) in targets if p == FENCE_PKG}
    return sorted(out, key=lambda x: (x[0], x[1], x[2] or "", x[3]))


def opens_hit(tree: Tree, fence: Fence, kind: str, tname: str | None, mode: str) -> tuple[str, int, str] | None:
    """(path, line, called name) of the first call of the fence or of an opener in the files of one FENCE_PKG target that the
    command's mode executes (the rule above)."""
    pkg = tree.packages[FENCE_PKG]
    if kind in ("lib", "doctest") or (kind in ("libtest", "test") and mode == "run"):
        return None
    src = pkg.lib().src if kind == "libtest" else pkg.find(kind, tname or "").src
    own = tree.tree_of(src)[0]
    files = own if kind == "test" or mode == "run" else tree.tree_of(src, True)[0] - own
    best = None
    for f in sorted(x for x in files if x.endswith(".rs")):
        shape = rust_lex(tree.read(f))[1]
        for at, name, form, qual in _calls(shape, 0, len(shape)):
            if fence.reaches(name, form, qual, None):
                hit = (f, _line_at(shape, at), name)
                best = hit if best is None or hit < best else best
                break
    return best


def opens(tree: Tree, recipes: dict[str, Recipe], names: list[str], explain=None, fence: Fence | None = None) -> list[str]:
    """One line a name: `opens<TAB>recipe<TAB>pkg:kind:target<TAB>path:line<TAB>called` or `none<TAB>recipe`; `explain`, when
    given, is called with the fence, the openers and each recipe's targets, one line at a time; `fence` is `opens_fence(tree)`
    when the caller has it."""
    missing = [n for n in names if n not in recipes]
    if missing:
        raise OpensError(65, f"not a recipe of the justfile: {', '.join(missing)}")
    fence = fence or opens_fence(tree)
    if explain:
        for d in fence.definers:
            explain(f"fence: fn {FENCE_FN} of {FENCE_PKG} at {d.path}:{d.line}")
        for (path, line), (d, via) in sorted(fence.openers.items()):
            explain(f"opener: {(d.owner + '::') if d.owner else ''}{d.name}{' (method)' if d.method else ''} at {path}:{line}, through {via}")
    cache: dict[tuple[str, str | None, str], tuple[str, int, str] | None] = {}
    rows = []
    for n in names:
        found = None
        targets = opens_targets(tree, recipes, n)
        if explain:
            explain(f"recipe {n}: {len(targets)} target(s) of {FENCE_PKG}: " + ", ".join(f"{k}:{t or tree.packages[FENCE_PKG].lib().name} ({m})" for _, k, t, m in targets))
        for pkg, kind, tname, mode in targets:
            if (kind, tname, mode) not in cache:
                cache[(kind, tname, mode)] = opens_hit(tree, fence, kind, tname, mode)
            hit = cache[(kind, tname, mode)]
            if hit:
                shown = tname or tree.packages[pkg].lib().name
                found = f"opens\t{n}\t{pkg}:{kind}:{shown}\t{hit[0]}:{hit[1]}\t{hit[2]}"
                break
        rows.append(found or f"none\t{n}")
    return rows


def cmd_opens(args: argparse.Namespace) -> int:
    if not args.items and not args.explain:
        print("recipes.py opens: name a recipe, or --explain", file=sys.stderr)
        return 64
    side = make_side(ROOT)
    try:
        rows = opens(side.tree, side.recipes, args.items, (lambda s: print("explain: " + s, file=sys.stderr)) if args.explain else None)
    except OpensError as err:
        print(f"recipes.py opens: {err}", file=sys.stderr)
        return err.code
    for row in rows:
        print(row)
    return 0


def opens_self_test(expect) -> None:
    """`opens` on synthetic workspaces: FENCE_PKG with a lib (tier.rs defines the fence), bins and test modules, one recipe a case;
    then the rule's mutants, each of which must turn its case red."""
    lib = (
        "mod tier;\n"
        "pub fn first() {\n    second();\n}\n"
        "pub fn second() {\n    tier::check_file();\n}\n"
        "pub struct Generator;\n"
        "impl<B> Generator<B> {\n    pub fn open(a: u8) {\n        second();\n    }\n}\n"
        "pub struct Rig;\n"
        "impl Rig {\n    pub fn go(&self) {\n        second();\n    }\n}\n"
        "#[cfg(test)]\nmod tests {\n    fn probe() {\n        super::second();\n    }\n}\n"
    )
    tier = "pub fn check_file() {}\npub fn run() {\n    check_file();\n}\n"
    bins = {
        "b1": "fn main() {\n    bloomery_gpu_gates::first();\n}\n",
        "b2": '// first() and second() in a comment\nfn main() {\n    let s = "second(1)";\n    /* first() */\n    println!("{s}");\n}\n',
        "b3": "fn main() {\n    let t = 0;\n    t.check_file();\n}\n",
        "b4": "fn main() {\n    bloomery_gpu_gates::tier::run();\n}\n",
        "bq1": "fn main() {\n    File::open(p);\n    Split::open(p);\n    x.open(p);\n    open(p);\n    go(1);\n}\n",
        "bq2": "fn main() {\n    Generator::<u8>::open(1);\n}\n",
        "bq3": "fn main() {\n    rig.go();\n}\n",
        "bp": "fn main() {\n    probe();\n}\n",
        "bt": "fn main() {}\n#[cfg(test)]\nmod tt;\n",
    }
    inner = {"tt.rs": "fn t() {\n    bloomery_gpu_gates::first();\n}\n"}
    dep = {"name": FENCE_PKG, "kind": None, "optional": False, "features": [], "uses_default_features": True}

    def world(tmp: str, lib_text: str, tier_text: str, bin_texts: dict[str, str], extra: dict[str, str] | None = None, user: bool = False) -> Tree:
        files = {"crates/g/Cargo.toml": "", "crates/g/src/lib.rs": lib_text, "crates/g/src/tier.rs": tier_text}
        files.update({f"crates/g/src/bin/{n}.rs": t for n, t in bin_texts.items()})
        files.update({f"crates/g/src/{n}": t for n, t in (extra or {}).items()})
        if "bt" in bin_texts:
            files.update({f"crates/g/src/bin/{n}": t for n, t in inner.items()})
        if user:
            files.update({"crates/u/Cargo.toml": "", "crates/u/src/main.rs": "fn main() {}\n"})
        for rel, text in files.items():
            os.makedirs(os.path.dirname(os.path.join(tmp, rel)), exist_ok=True)
            with open(os.path.join(tmp, rel), "w", encoding="utf-8") as fh:
                fh.write(text)
        targets = [{"kind": ["lib"], "name": "bloomery_gpu_gates", "src_path": os.path.join(tmp, "crates/g/src/lib.rs")}]
        targets += [{"kind": ["bin"], "name": n, "src_path": os.path.join(tmp, f"crates/g/src/bin/{n}.rs")} for n in bin_texts]
        pkgs = [{"id": FENCE_PKG, "name": FENCE_PKG, "manifest_path": os.path.join(tmp, "crates/g/Cargo.toml"), "features": {}, "dependencies": [], "targets": targets}]
        if user:
            pkgs.append({"id": "user", "name": "user", "manifest_path": os.path.join(tmp, "crates/u/Cargo.toml"), "features": {}, "dependencies": [dep],
                         "targets": [{"kind": ["bin"], "name": "user", "src_path": os.path.join(tmp, "crates/u/src/main.rs")}]})
        return Tree(tmp, {"workspace_root": tmp, "workspace_members": [p["id"] for p in pkgs], "packages": pkgs})

    def box(*cmds: str, deps: list[str] | None = None) -> Recipe:
        return Recipe("x", [f"./tools/box.sh '{c}'" for c in cmds], deps or [], "")

    def run_bin(b: str) -> Recipe:
        return box(f"cargo build -p {FENCE_PKG} --bin {b} && bash tools/gpu-gate.sh {b}")

    def ask(tree: Tree, recipes: dict[str, Recipe], name: str, **kw) -> str:
        return opens(tree, recipes, [name], **kw)[0]

    def refused(tree: Tree, recipes: dict[str, Recipe], names: list[str]) -> OpensError | None:
        try:
            opens(tree, recipes, names)
        except OpensError as err:
            return err
        return None

    def row(recipe: str, target: str, site: str, called: str) -> str:
        return f"opens\t{recipe}\t{FENCE_PKG}:{target}\t{site}\t{called}"

    with tempfile.TemporaryDirectory(prefix="recipes-opens-") as tmp:
        t1 = world(os.path.join(tmp, "w1"), lib, tier, bins)
        recipes = {f"r{n}": run_bin(n) for n in bins}
        recipes.update({
            "outer": Recipe("outer", [], ["rb1"], ""),
            "checked": box(f"cargo check -p {FENCE_PKG} --all-targets", f"cargo clippy -p {FENCE_PKG} --bin b1 && bash tools/gpu-gate.sh b1"),
            "built": box(f"cargo build -p {FENCE_PKG} --bin b1"),
            "tested": box(f"cargo test -p {FENCE_PKG} --bin b1 --lib"),
            "testbt": box(f"cargo test -p {FENCE_PKG} --bin bt"),
            "runbt": run_bin("bt"),
        })
        c = "crates/g/src/bin"
        got = sorted(f"{d.owner}::{d.name}" if d.owner else d.name for d, _ in opens_fence(t1).openers.values())
        expect(got == ["Generator::open", "Rig::go", "first", "second"], f"opens: the openers are {got}")
        expect(ask(t1, recipes, "rb1") == row("rb1", "bin:b1", f"{c}/b1.rs:2", "first"), f"opens r1: a bin calling a lib fn that calls a lib fn that calls the fence: {ask(t1, recipes, 'rb1')}")
        expect(ask(t1, recipes, "rb2") == "none\trb2", f"opens r2: an opener named in a comment and a string is no call: {ask(t1, recipes, 'rb2')}")
        expect(ask(t1, recipes, "rb3") == row("rb3", "bin:b3", f"{c}/b3.rs:3", "check_file"), f"opens r3: a bin calling the fence itself: {ask(t1, recipes, 'rb3')}")
        expect(ask(t1, recipes, "rb4") == "none\trb4", f"opens r5: the fence called only inside its definer's file: {ask(t1, recipes, 'rb4')}")
        expect(ask(t1, recipes, "rbq1") == "none\trbq1", f"opens: File::open, Split::open, .open and a bare open are no call of Generator::open, nor go( of Rig::go: {ask(t1, recipes, 'rbq1')}")
        expect(ask(t1, recipes, "rbq2") == row("rbq2", "bin:bq2", f"{c}/bq2.rs:2", "open"), f"opens: Generator::<u8>::open( is a call of the opener: {ask(t1, recipes, 'rbq2')}")
        expect(ask(t1, recipes, "rbq3") == row("rbq3", "bin:bq3", f"{c}/bq3.rs:2", "go"), f"opens: .go( is a call of a method opener: {ask(t1, recipes, 'rbq3')}")
        expect(ask(t1, recipes, "rbp") == "none\trbp", f"opens: a fn of an inline #[cfg(test)] module is no opener: {ask(t1, recipes, 'rbp')}")
        expect(ask(t1, recipes, "outer") == row("outer", "bin:b1", f"{c}/b1.rs:2", "first"), f"opens: a recipe's dependency's targets count: {ask(t1, recipes, 'outer')}")
        expect(ask(t1, recipes, "checked") == "none\tchecked", f"opens: check and clippy run nothing: {ask(t1, recipes, 'checked')}")
        expect(ask(t1, recipes, "built") == "none\tbuilt", f"opens: a recipe that builds and runs nothing: {ask(t1, recipes, 'built')}")
        expect(ask(t1, recipes, "tested") == "none\ttested", f"opens: a test run of a bin reads its test files, not its main: {ask(t1, recipes, 'tested')}")
        expect(ask(t1, recipes, "testbt") == row("testbt", "bin:bt", f"{c}/tt.rs:2", "first"), f"opens: a test run of a bin reads its test-only files: {ask(t1, recipes, 'testbt')}")
        expect(ask(t1, recipes, "runbt") == "none\trunbt", f"opens: a run of a bin does not read its test-only files: {ask(t1, recipes, 'runbt')}")
        why = refused(t1, recipes, ["rb1", "nothere"])
        expect(why is not None and why.code == 65 and "nothere" in str(why), f"opens: an item that is no recipe is 65, named: {why}")
        lines: list[str] = []
        opens(t1, recipes, ["rb1"], explain=lines.append)
        expect(any(x.startswith("fence: fn check_file") for x in lines) and any("opener: first" in x for x in lines) and any("recipe rb1: 1 target(s)" in x for x in lines), f"opens: --explain names the fence, the openers and the targets: {lines}")

        # r4: the lib's test files, under `cargo test --lib`
        t2 = world(os.path.join(tmp, "w2"), lib + "#[cfg(test)]\nmod t;\n", tier, {"b1": bins["b1"]}, {"t.rs": "fn x() {\n    super::first();\n}\n"})
        lt = {"lt": box(f"cargo test -p {FENCE_PKG} --lib")}
        expect(ask(t2, lt, "lt") == row("lt", "libtest:bloomery_gpu_gates", "crates/g/src/t.rs:2", "first"), f"opens r4: a lib's #[cfg(test)] module calling an opener: {ask(t2, lt, 'lt')}")
        t3 = world(os.path.join(tmp, "w3"), lib + "pub fn third() {\n    first();\n}\n", tier, {"b1": bins["b1"]})
        expect(ask(t3, lt, "lt") == "none\tlt", f"opens r4: the same call in the lib's non-test code reaches nothing under `cargo test --lib`: {ask(t3, lt, 'lt')}")
        # r6, and the dependent package
        t4 = world(os.path.join(tmp, "w4"), lib, "pub fn other() {}\n", {"b1": bins["b1"]})
        why = refused(t4, recipes, ["rb1"])
        expect(why is not None and why.code == 70 and "check_file" in str(why), f"opens r6: no fn check_file is a named 70: {why}")
        t5 = world(os.path.join(tmp, "w5"), lib, tier, {"b1": bins["b1"]}, user=True)
        why = refused(t5, recipes, ["rb1"])
        expect(why is not None and why.code == 70 and "user depend on" in str(why), f"opens: a package that depends on {FENCE_PKG} is a named 70: {why}")

        # the mutants: each must turn its case red
        real_reaches, real_lex, real_called = Fence.reaches, rust_lex, _opener_called
        try:
            Fence.reaches = lambda self, name, form, qual, caller: name == FENCE_FN
            one_level = opens_fence(t1)
        finally:
            Fence.reaches = real_reaches
        expect(ask(t1, recipes, "rb1", fence=one_level) != ask(t1, recipes, "rb1"), "opens mutant: with the fixpoint cut to one level, r1 stays green")
        try:
            globals()["rust_lex"] = lambda text: (text, text)
            expect(ask(t1, recipes, "rb2") != "none\trb2", "opens mutant: with comment skipping off, r2 stays green")
        finally:
            globals()["rust_lex"] = real_lex
        try:
            globals()["_opener_called"] = lambda o, form, qual, caller: True
            expect(ask(t1, recipes, "rbq1", fence=opens_fence(t1)) != "none\trbq1", "opens mutant: matching an opener by its bare name, the File::open control stays green")
        finally:
            globals()["_opener_called"] = real_called


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
_LITERAL_MEMO: dict[tuple[str, int, int], frozenset[str]] = {}
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


def file_model_literal_dirs(path: str) -> frozenset[str]:
    """model_literal_dirs of one file, memoized on its stat for the process: every KeyContext reads through it."""
    st = os.stat(path)
    k = (path, st.st_size, st.st_mtime_ns)
    if k not in _LITERAL_MEMO:
        with open(path, encoding="utf-8", errors="replace") as fh:
            _LITERAL_MEMO[k] = frozenset(model_literal_dirs(fh.read()))
    return _LITERAL_MEMO[k]


def just_version() -> str:
    if not _JUST_VERSION:
        _JUST_VERSION.append(subprocess.run(["just", "--version"], capture_output=True, text=True).stdout.strip() or "unknown")
    return _JUST_VERSION[0]


def oxide_rev(text: str) -> str:
    """The cuda-oxide rev tools/box.sh reads: the [patch] section's cuda-device rev, else the first one."""
    sec = re.search(r'^\[patch\."https://github\.com/NVIDIA/cuda-rust\.git"\]\s*$(.*?)(?=^\[workspace|\Z)', text, re.M | re.S)
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
        self.mac = [f"mac-env\t{v}\t{tier_of([], env.get(v)) if v == 'BLOOMERY_TIER' else env[v] if v in env else '<unset>'}" for v in MAC_ENV]
        self.mac += [f"mac-tool\tjust\t{just_version()}", f"mac-tool\tpython3\t{platform.python_version()}"]
        self.manifest, self.manifest_head, self.manifest_error = load_manifest(manifest_path)
        if manifest_error:
            self.manifest_error = f"the box manifest failed: {manifest_error}"
        self.settings = settings if settings is not None else justfile_settings(os.path.join(side.tree.root, "justfile"))
        self._sha: dict[str, str] = {}
        self._lit: dict[str, frozenset[str]] = {}
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

    def view_part(self, rel: str, wants: frozenset[Entry]) -> str:
        """The part of a manifest read by table: the sha256 of the item's view of it."""
        view = manifest_view(read_manifest(self.side.tree.root, rel), os.path.dirname(rel), wants)
        return f"manifest\t{rel}\t{sha256_text(canon(view))}"

    def literal_dirs(self, rel: str) -> frozenset[str]:
        if rel not in self._lit:
            found = file_model_literal_dirs(os.path.join(self.side.tree.root, rel)) if scans_for_models(rel) else frozenset()
            self._lit[rel] = found - {FIXTURE_ROOT}
        return self._lit[rel]

    def model_scope(self, names: list[str], files: set[str], eff: list[str], envs: list[str], argv: list[str]) -> tuple[dict[str, str], set[str]]:
        """The model directories the item can open, each with the first input that names it, and those of
        them the box manifest must have rows for. They are the /models literals in its key's files (its
        profile and tools/box.sh's among them) and its recipes' text, the profile the Mac's BLOOMERY_MODEL
        picks for a box recipe that names none, and the /models values of BLOOMERY_BOX_ENV, the forwarded
        Mac environment, the item's env and its ARGS. The item's env and ARGS are outside the manifest's
        reads (never-skip), so they need no rows."""
        recipes, tree = self.side.recipes, self.side.tree
        dirs: dict[str, str] = {}
        for rel in sorted(files | {g for g in KEY_GLOBALS if tree.exists(g)}):
            for d in self.literal_dirs(rel):
                dirs.setdefault(d, rel)
        for n in names:
            for d in model_literal_dirs(recipes[n].text):
                dirs.setdefault(d, f"recipe {n}")
            rc = self.side.graph.inputs(n).commands
            prof = f"tools/ref/models/{self.env.get('BLOOMERY_MODEL', '')}.sh"
            if rc.box and "BLOOMERY_MODEL" not in rc.env and self.env.get("BLOOMERY_MODEL") and tree.exists(prof):
                for d in self.literal_dirs(prof):
                    dirs.setdefault(d, f"{prof} (the Mac's BLOOMERY_MODEL)")
        for e in self.box_env.split():
            k, _, v = e.partition("=")
            d = model_dir_of(v)
            if d:
                dirs.setdefault(d, f"BLOOMERY_BOX_ENV {k}")
        for k in MAC_ENV:
            d = model_dir_of(self.env.get(k, ""))
            if d:
                dirs.setdefault(d, f"the Mac's {k}")
        need = set(dirs)
        if tier_of(eff, self.env.get("BLOOMERY_TIER")) == "fixture":
            dirs.setdefault(FIXTURE_ROOT, "the fixture tier")
            need.add(FIXTURE_ROOT)
        for w in envs + argv:
            for v in w.split(","):
                d = model_dir_of(v.split("=", 1)[-1])
                if d and d not in dirs:
                    dirs[d] = "the item's env or ARGS"
        return dirs, need

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
        # The form counts in the recipe's text and in every script it runs (tools/ptx-scan.sh's JIT), the
        # closure tools/gate-batch.sh classifies a lane from.
        forced = any(e.partition("=")[0] == "BLOOMERY_GATE_CARD" for e in self.box_env.split() + envs)
        if not forced:
            for n in names:
                if any(ANY_CARD in ln for ln in recipes[n].lines):
                    return f"{n} calls tools/gpu-gate.sh with the `any` card and no card is forced (--lanes 1): the card is picked at run time"
            boxed = [n for n in names if self.side.graph.inputs(n).commands.box]
            for f in self.run_scripts(boxed) if boxed else []:
                hit = scan_file(self.side.tree.root, f, ANY_CARD_CALL)
                if hit:
                    return f"{hit} calls tools/gpu-gate.sh with the `any` card and no card is forced (--lanes 1): the card is picked at run time"
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
            # a package manifest the closure reads by table is keyed on that view (manifest_view, the reader
            # selection uses); one read whole, named by the item, or in a tree-scoped key, on its bytes
            wants = ri.manifests.entries(rel) if scope == "closure" and rel not in named else None
            try:
                parts.append(self.file_part(rel) if wants is None else self.view_part(rel, wants))
            except OSError as err:
                return [], f"cannot read {rel}: {err.strerror}", None, ""
            except RecipeError as err:
                return [], str(err), None, ""
        parts += self.globals()
        card = "none"
        for e in eff:
            k, _, v = e.partition("=")
            if k == "BLOOMERY_GATE_CARD":
                card = v
        parts += [f"item\targs\t{json.dumps(argv)}", f"item\tboxenv\t{' '.join(eff)}", f"item\tcard\t{card}"]
        parts += self.mac
        # The manifest lists every model directory any item can open; this item's key holds the rows of
        # its own only, so a directory another gate names (a fixture, a profile) does not move it.
        dirs, need = self.model_scope(names, files, eff, envs, argv)
        listed: set[str] = set()
        for ln in self.manifest:
            f = ln.split("\t")
            if f[0] in ("model", "model-dir") and len(f) > 1:
                d = model_dir_of(f[1])
                if d is not None and d not in dirs:
                    continue
                listed.add(d)
            parts.append(f"box\t{ln}")
        missing = sorted(need - listed)
        if missing:
            return [], f"the box manifest has no row for {missing[0]}, which {dirs[missing[0]]} names: the manifest was read from another tree or environment", None, ""
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
    """The green ledger, read-only here: `key recipe item commit date tree`, appended by gate-batch.sh.
    `src` names whose runner wrote it: `lead` (gate-batch.sh --ledger) or `round` (--round-ledger)."""

    def __init__(self, path: str, src: str = "lead"):
        self.path = path
        self.src = src
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


def ledger_status(key: str, name: str, item: str, parts: list[str], ledgers: list[Ledger]) -> tuple[str, str]:
    """skip with the first ledger (in order) that holds a green run at this key, else run with why.
    The detail names the source (`src=lead|round`) before `tree=`, the last field gate-batch.sh cuts."""
    for led in ledgers:
        r = led.by_key.get(key)
        if r is not None:
            return "skip", f"green-at={r.commit} {r.date} src={led.src} tree={r.tree}"
    for led in ledgers:
        if (name, item) in led.by_item or name in led.by_recipe:
            return "run", led.why(name, item, parts) + (f" (src={led.src})" if led.src != "lead" else "")
    return "run", f"no green record of {name}"


def cmd_key(args: argparse.Namespace) -> int:
    side = make_side(ROOT)
    ctx = KeyContext(side, args.manifest, args.box_env or "", manifest_error=args.manifest_error)
    if args.round_ledger and not args.ledger:
        raise RecipeError("--round-ledger reads a second ledger beside --ledger; give --ledger too")
    ledgers = [Ledger(args.ledger)] if args.ledger else []
    if args.round_ledger:
        ledgers.append(Ledger(args.round_ledger, "round"))
    for led in ledgers:
        if led.bad:
            print(f"recipes.py key: {led.path}: {led.bad} malformed line(s) ignored (they cannot skip an item)", file=sys.stderr)
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
        elif not ledgers:
            status, detail = "key", summary
        elif args.rerun:
            status, detail = "rerun", "--rerun: runs whatever the ledger holds"
        else:
            status, detail = ledger_status(key, name, item, parts, ledgers)
        print(f"{key}\t{item}\t{status}\t{detail}".replace("\n", " "))
        if args.show_parts:
            for p in parts:
                print(f"  {p}")
    return 0


# ---------------- the box manifest (runs on the box, through tools/box.sh) ----------------


class HashCache:
    """sha256 of files keyed by (size, mtime, ctime, inode, device), kept between batches on the box:
    a file rewritten with the same bytes (the dump recipes rewrite their sets) keeps
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
        if DATA_NEVER_ONLY.fullmatch(top):
            lines.append(("data-skip", top + "/", "read only by never-skip items (gate-tokenizer rewrites it every run)"))
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


def profile_dirs(root: str, prof: str, env: dict[str, str]) -> set[str]:
    """The model directories the profile `prof` (tools/ref/models/<x>.sh under `root`) defines when
    sourced under `env` (BLOOMERY_REF_MODEL unset: the profile's own default). A directory it builds
    that no literal in its text names, nor an environment value, is an error: an item's key reads the
    profile's literals (model_literal_dirs), so such a directory would be in the manifest and in no key."""
    out = _run(["bash", "-c", PROFILE_PATHS, "bash", os.path.join(root, prof)], env=env)
    dirs = {d for d in map(model_dir_of, out.splitlines()) if d}
    with open(os.path.join(root, prof), encoding="utf-8", errors="replace") as fh:
        named = model_literal_dirs(fh.read())
    named |= {d for d in map(model_dir_of, env.values()) if d}
    if dirs - named:
        raise RecipeError(f"{prof} defines {sorted(dirs - named)[0]}, which no /models literal in it names: an item's key could not select that directory's manifest rows — spell the path literally")
    return dirs


def tree_model_dirs(root: str) -> set[str]:
    """The model directories the tree's files name literally (the files scans_for_models picks)."""
    dirs: set[str] = set()
    for rel in shipped_files(root):
        if scans_for_models(rel):
            with open(os.path.join(root, rel), encoding="utf-8", errors="replace") as fh:
                dirs |= model_literal_dirs(fh.read())
    return dirs


def model_dirs(root: str = ROOT) -> set[str]:
    """The model directories a gate can open: every /models path the profiles define (under their own
    defaults), the box command's environment names, or the tree names literally — the directory of each,
    so a split set's shards all count. Never /models itself, whose listing moves whenever any model is
    fetched. An item's key selects its own rows out of these (KeyContext.model_scope)."""
    env = {k: v for k, v in os.environ.items() if k not in ("BLOOMERY_REF_MODEL", "BLOOMERY_TIER")}
    dirs: set[str] = set()
    for prof in sorted(glob.glob(os.path.join(root, "tools/ref/models/*.sh"))):
        dirs |= profile_dirs(root, os.path.relpath(prof, root), env)
    dirs |= {d for d in map(model_dir_of, os.environ.values()) if d}
    return dirs | tree_model_dirs(root) | {FIXTURE_ROOT}


def model_dir_rows(d: str, recurse: bool) -> list[tuple]:
    """The manifest rows of the entries of the model directory `d`: each file's size, mtime, ctime and inode (a link's target
    and a dangling link's state too), each subdirectory a `dir` row — and, with `recurse`, the rows of what is inside it. The
    fixture root recurses: its files sit one directory a family down, and a fixture rewritten in place must move the keys that
    read it."""
    rows: list[tuple] = []
    for f in sorted(os.listdir(d)):
        p = os.path.join(d, f)
        lst = os.lstat(p)
        if stat.S_ISDIR(lst.st_mode):
            rows.append(("model", p, "dir"))
            if recurse:
                rows += model_dir_rows(p, True)
            continue
        link = os.readlink(p) if stat.S_ISLNK(lst.st_mode) else ""
        try:
            st = os.stat(p)
        except FileNotFoundError:
            rows.append(("model", p, "->", link, "dangling"))
            continue
        rows.append(("model", p, st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino) + (("->", link) if link else ()))
    return rows


def manifest_model_rows(d: str, fixture_root: str = FIXTURE_ROOT) -> list[tuple]:
    """The box manifest's rows of one model path of model_dirs(): a file's own row, a directory that is absent or empty named so,
    a directory's entries (model_dir_rows; the fixture root's reach below its family directories)."""
    if os.path.isfile(d):
        st = os.stat(d)
        return [("model", d, st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino)]
    if not os.path.isdir(d):
        return [("model-dir", d, "absent")]
    if not os.listdir(d):
        return [("model-dir", d, "empty")]
    return model_dir_rows(d, d == fixture_root)


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
            rows += manifest_model_rows(d)
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


def static_self_test(expect) -> None:
    """static_problems on a synthetic workspace: package g with bins x (no feature) and y (`fy`). A static
    recipe without `fy` names y; enabling it by package path clears it; a recipe that is not one workspace
    check is named."""
    with tempfile.TemporaryDirectory(prefix="recipes-static-") as tmp:
        files = {"crates/g/Cargo.toml": "", "crates/g/src/lib.rs": "", "crates/g/src/bin/x.rs": "fn main() {}\n",
                 "crates/g/src/bin/y.rs": "fn main() {}\n"}
        for rel, text in files.items():
            os.makedirs(os.path.dirname(os.path.join(tmp, rel)), exist_ok=True)
            with open(os.path.join(tmp, rel), "w", encoding="utf-8") as fh:
                fh.write(text)
        meta = {
            "workspace_root": tmp,
            "workspace_members": ["g"],
            "packages": [{
                "id": "g", "name": "g", "manifest_path": os.path.join(tmp, "crates/g/Cargo.toml"), "features": {"fy": []},
                "dependencies": [],
                "targets": [
                    {"kind": ["lib"], "name": "g", "src_path": os.path.join(tmp, "crates/g/src/lib.rs")},
                    {"kind": ["bin"], "name": "x", "src_path": os.path.join(tmp, "crates/g/src/bin/x.rs")},
                    {"kind": ["bin"], "name": "y", "src_path": os.path.join(tmp, "crates/g/src/bin/y.rs"), "required-features": ["fy"]},
                ],
            }],
        }
        tree = Tree(tmp, meta)

        def recipes(check_features: str, lint: str = "cargo clippy --workspace --all-targets --features g/fy") -> dict[str, Recipe]:
            return {"check": Recipe("check", [f"./tools/box.sh 'cargo check --workspace --all-targets{check_features}'"], [], ""),
                    "lint": Recipe("lint", [f"./tools/box.sh '{lint}'"], [], "")}

        got = static_problems(tree, recipes(""))
        expect(len(got) == 1 and "check: bin y of g requires features ['fy']" in got[0], f"static: a check without fy gives {got}")
        got = static_problems(tree, recipes(" --features g/fy"))
        expect(got == [], f"static: every bin checked, yet {got}")
        got = static_problems(tree, recipes(" --features g/fy", "cargo clippy -p g --lib"))
        expect(len(got) == 1 and "lint: not one" in got[0], f"static: a lint of one target gives {got}")


def scan_set_self_test(expect, side: Side) -> None:
    """scan_set on the real tree (every carrier covered, every bin carrier by its own scan) and on synthetic
    workspaces: package bloomery-gpu-gates with bins x (feature fa), y (fb), z (fa, a bin carrier) and w; device
    libs liba (behind fa), libb (behind fb) and, unlinked by any bin, libc."""
    gates = [n for n in side.recipes if n.startswith(GATE_PREFIX)]
    carriers = kernel_carriers(side, gates)
    try:
        picks = scan_set(side)
    except RecipeError as err:
        expect(False, f"scan-set: the real tree's scan set is refused: {err}")
        picks = []
    covered = {c for p in picks for c in p.covers}
    expect(covered == {c.label for c in carriers},
           f"scan-set: the real tree's picks leave {sorted({c.label for c in carriers} - covered)} uncovered")
    expect({c.bin for c in carriers if c.bin} <= {p.bin for p in picks}, f"scan-set: a bin carrier whose own scan is not in the set: {[(p.bin, p.features) for p in picks]}")
    expect(all(c.bundle for c in carriers), "scan-set: a carrier with no bundle name")

    def build(tmp: str, name: str, libc: bool = False, other: bool = False, recipe: str = "") -> Side:
        root = os.path.join(tmp, name)
        kernel, plain = "#[kernel]\nfn k() {}\n", "fn f() {}\n"
        files = {"crates/liba/src/lib.rs": kernel, "crates/libb/src/lib.rs": kernel, "crates/libc/src/lib.rs": kernel if libc else plain,
                 "crates/gates/src/lib.rs": plain, "crates/gates/src/bin/x.rs": "fn main() {}\n", "crates/gates/src/bin/y.rs": "fn main() {}\n",
                 "crates/gates/src/bin/z.rs": kernel + "fn main() {}\n", "crates/gates/src/bin/w.rs": "fn main() {}\n",
                 "crates/other/src/main.rs": kernel + "fn main() {}\n" if other else "fn main() {}\n",
                 "justfile": "gate-gx:\n    ./tools/box.sh 'cargo oxide build -- -p bloomery-gpu-gates --features fa --release --bin x'\n" + recipe}
        for f in BOX_GLOBALS + [f"tools/ref/models/{DEFAULT_PROFILE}.sh"]:
            files[f] = ""
        for rel, text in files.items():
            os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
            with open(os.path.join(root, rel), "w", encoding="utf-8") as fh:
                fh.write(text)

        def pkg(n: str, d: str, targets: list[dict], features=None, deps=()) -> dict:
            return {"id": n, "name": n, "manifest_path": os.path.join(root, f"crates/{d}/Cargo.toml"), "features": features or {},
                    "dependencies": [{"name": x, "kind": None, "optional": True, "features": [], "uses_default_features": True} for x in deps],
                    "targets": targets}

        def tgt(kind: str, n: str, src: str, req=()) -> dict:
            return {"kind": [kind], "name": n, "src_path": os.path.join(root, src), "required-features": list(req)}

        meta = {"workspace_root": root, "workspace_members": ["liba", "libb", "libc", "bloomery-gpu-gates", "other"], "packages": [
            pkg("liba", "liba", [tgt("lib", "liba", "crates/liba/src/lib.rs")]),
            pkg("libb", "libb", [tgt("lib", "libb", "crates/libb/src/lib.rs")]),
            pkg("libc", "libc", [tgt("lib", "libc", "crates/libc/src/lib.rs")]),
            pkg("bloomery-gpu-gates", "gates", [tgt("lib", "bloomery-gpu-gates", "crates/gates/src/lib.rs"), tgt("bin", "x", "crates/gates/src/bin/x.rs", ["fa"]),
                                              tgt("bin", "y", "crates/gates/src/bin/y.rs", ["fb"]), tgt("bin", "z", "crates/gates/src/bin/z.rs", ["fa"]),
                                              tgt("bin", "w", "crates/gates/src/bin/w.rs")],
                {"fa": ["dep:liba"], "fb": ["dep:libb"]}, ["liba", "libb"]),
            pkg("other", "other", [tgt("bin", "o", "crates/other/src/main.rs")]),
        ]}
        return side_at(root, meta)

    def shown(ps: list[ScanPick]) -> list[tuple[str, tuple[str, ...]]]:
        return [(p.bin, p.features) for p in ps]

    def refused(fn, why: str) -> bool:
        try:
            fn()
        except RecipeError as err:
            return why in str(err)
        return False

    with tempfile.TemporaryDirectory(prefix="recipes-scanset-") as tmp:
        fx = build(tmp, "t0")
        got = scan_set(fx)
        expect(shown(got) == [("z", ("fa",)), ("y", ("fb",))], f"scan-set: two device libs and a bin carrier are not covered by z and y: {shown(got)}")
        expect(got and sorted(got[0].covers) == ["bin z", "lib liba"], f"scan-set: the first pick's coverage: {got[0].covers if got else got}")
        # a recipe's wider feature list is a candidate too, and loses the tie to the bin's own required features
        got = scan_set(build(tmp, "t1", recipe="gate-gy:\n    ./tools/box.sh 'cargo oxide build -- -p bloomery-gpu-gates --features fa,fb --release --bin y'\n"))
        expect(shown(got) == [("z", ("fa",)), ("y", ("fb",))], f"scan-set: a wider recipe shape beat the bin's own features: {shown(got)}")
        # a carrier no bin links is a named error, never dropped; so is a bin carrier of another package
        expect(refused(lambda: scan_set(build(tmp, "t2", libc=True)), "no bin of bloomery-gpu-gates covers lib libc"),
               "scan-set: a lib no bin links is not a named error")
        expect(refused(lambda: scan_set(build(tmp, "t3", other=True)), "no bin of bloomery-gpu-gates covers bin o"),
               "scan-set: a bin carrier outside the scan package is not a named error")
        # a bin's own scan does not vouch for a lib of its package: the bundle name is shared
        c = Carrier("lib p", "p", set(), [], siblings=frozenset({"b"}))
        expect(not carrier_covered(c, "b", ["p"]) and carrier_covered(c, "x", ["p"]) and not carrier_covered(c, "x", ["q"]),
               "scan-set: a lib carrier is covered by its sibling bin carrier's scan")


def manifest_self_test(expect) -> None:
    """A package manifest read by table, on a synthetic workspace (package g: a lib and bins x, y, v; user: a
    bin linking g; a script that reads g's manifest whole). Each change gives the recipes it must select and
    the ledger keys must move on exactly those: the failure modes the rule must not create (the own entry's
    required-features, a --bins list it flips, [features], a renamed path, a renamed entry, an entry whose
    path claims a target's file) and the saving (another target's entry, a comment, entry order); a manifest
    that does not parse is a named error in both."""
    import tomllib

    base = (
        '[package]\nname = "g"\nversion = "0.1.0"\n\n[features]\nfy = []\n\n'
        '[[bin]]\nname = "x"\npath = "src/bin/x.rs"\n\n'
        '[[bin]]\nname = "y"\npath = "src/bin/y.rs"\nrequired-features = ["fy"]\n\n'
        '[[bin]]\nname = "v"\npath = "src/bin/v.rs"\n'
    )
    just = (
        "gate-x:\n    ./tools/box.sh 'cargo build -p g --bin x'\n"
        "gate-y:\n    ./tools/box.sh 'cargo build -p g --features fy --bin y'\n"
        "gate-bins:\n    ./tools/box.sh 'cargo build -p g --features fy --bins'\n"
        "gate-plain:\n    ./tools/box.sh 'cargo build -p g --bins'\n"
        "gate-lib:\n    ./tools/box.sh 'cargo test -p g --lib'\n"
        "gate-user:\n    ./tools/box.sh 'cargo build -p user --bin user'\n"
        "gate-whole:\n    ./tools/box.sh 'cargo build -p g --lib && bash tools/s.sh'\n"
    )
    every = {"gate-x", "gate-y", "gate-bins", "gate-plain", "gate-lib", "gate-user", "gate-whole"}
    with tempfile.TemporaryDirectory(prefix="recipes-tables-") as tmp:
        mf = os.path.join(tmp, "box-manifest.txt")
        with open(mf, "w", encoding="utf-8") as fh:
            fh.write(f"# {MANIFEST_VERSION} host=selftest data=/d\n# end\n")
        made: list[str] = []

        def build(g_toml: str, justfile: str = just, meta_toml: str | None = None) -> Side:
            root = os.path.join(tmp, f"t{len(made)}")
            made.append(root)
            doc = tomllib.loads(meta_toml if meta_toml is not None else g_toml)
            files = {
                "crates/g/Cargo.toml": g_toml,
                "crates/g/src/lib.rs": "",
                "crates/user/Cargo.toml": '[package]\nname = "user"\nversion = "0.1.0"\n\n[dependencies]\ng = { path = "../g" }\n',
                "crates/user/src/main.rs": "fn main() {}\n",
                "tools/s.sh": 'cat "$ROOT/crates/g/Cargo.toml"\n',
                "justfile": justfile,
            }
            for f in BOX_GLOBALS + [f"tools/ref/models/{DEFAULT_PROFILE}.sh"]:
                files[f] = ""
            for e in doc.get("bin", []):
                files.setdefault("crates/g/" + e["path"], "#[kernel]\nfn k() {}\nfn main() {}\n" if e["name"] == "x" else "fn main() {}\n")
            for rel, text in files.items():
                os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
                with open(os.path.join(root, rel), "w", encoding="utf-8") as fh:
                    fh.write(text)
            g_targets = [{"kind": ["lib"], "name": "g", "src_path": os.path.join(root, "crates/g/src/lib.rs")}]
            g_targets += [
                {"kind": ["bin"], "name": e["name"], "src_path": os.path.join(root, "crates/g", e["path"]), "required-features": e.get("required-features", [])}
                for e in doc.get("bin", [])
            ]
            meta = {
                "workspace_root": root,
                "workspace_members": ["g", "user"],
                "packages": [
                    {"id": "g", "name": "g", "manifest_path": os.path.join(root, "crates/g/Cargo.toml"), "features": {k: list(v) for k, v in doc.get("features", {}).items()},
                     "dependencies": [], "targets": g_targets},
                    {"id": "user", "name": "user", "manifest_path": os.path.join(root, "crates/user/Cargo.toml"), "features": {},
                     "dependencies": [{"name": "g", "kind": None, "optional": False, "features": [], "uses_default_features": True}],
                     "targets": [{"kind": ["bin"], "name": "user", "src_path": os.path.join(root, "crates/user/src/main.rs")}]},
                ],
            }
            return side_at(root, meta)

        def keys(side: Side) -> dict[str, str]:
            ctx = KeyContext(side, mf, environ={})
            out = {}
            for n in sorted(every | {"gate-x2"}):
                parts, err, _, _ = ctx.parts(n)
                out[n] = item_key(parts) if err is None else "error: " + err
            return out

        a = build(base)
        ka = keys(a)
        expect(not any(v.startswith("error") for k, v in ka.items() if k != "gate-x2"), f"tables: base keys {ka}")
        expect(keys(build(base)) == ka, "tables: the same manifest in a second tree gave other keys")
        bin_z = '\n[[bin]]\nname = "z"\npath = "src/bin/z.rs"\n'
        x_entry = '[[bin]]\nname = "x"\npath = "src/bin/x.rs"\n'
        y_entry = '[[bin]]\nname = "y"\npath = "src/bin/y.rs"\nrequired-features = ["fy"]\n'
        cases = [
            ("another target's new entry", base + bin_z, just, {"gate-bins", "gate-plain", "gate-whole"}),
            ("another target's entry edited", base.replace(x_entry, x_entry + "doc = false\n"), just, {"gate-x", "gate-bins", "gate-plain", "gate-whole"}),
            ("its own required-features", base.replace('required-features = ["fy"]', "required-features = []"), just, {"gate-y", "gate-bins", "gate-plain", "gate-whole"}),
            # x leaves gate-plain's --bins list (it lacks fy; v stays) and gate-x's build: only A's lists hold x
            ("required-features that flips a --bins list", base.replace(x_entry, x_entry + 'required-features = ["fy"]\n'), just,
             {"gate-x", "gate-bins", "gate-plain", "gate-whole"}),
            ("[features]", base.replace("fy = []", "fy = []\nfz = []"), just, every),
            ("a renamed path", base.replace('"src/bin/y.rs"', '"src/bin/y2.rs"'), just, {"gate-y", "gate-bins", "gate-whole"}),
            # gate-x still names x (its B build is refused by name); gate-x2 is the new name's recipe
            ("a renamed entry", base.replace('name = "x"', 'name = "x2"').replace('"src/bin/x.rs"', '"src/bin/x2.rs"'),
             just + "gate-x2:\n    ./tools/box.sh 'cargo build -p g --bin x2'\n", {"gate-x", "gate-x2", "gate-bins", "gate-plain", "gate-whole"}),
            ("an entry whose path claims x's file", base + '\n[[bin]]\nname = "w"\npath = "src/bin/x.rs"\n', just, {"gate-x", "gate-bins", "gate-plain", "gate-whole"}),
            ("a comment", base + "# a comment\n", just, {"gate-whole"}),
            ("the entries' order", base.replace(x_entry + "\n" + y_entry, y_entry + "\n" + x_entry), just, {"gate-whole"}),
        ]
        for what, g_toml, justfile, want in cases:
            b = build(g_toml, justfile)
            sels, unmapped, _ = select(["crates/g/Cargo.toml"], a, b, GATE_PREFIX)
            got = {x.recipe for x in sels}
            expect(got == want, f"tables: {what} selects {sorted(got)}, not {sorted(want)}")
            kb = keys(b)
            moved = {n for n in kb if kb[n] != ka[n]}
            expect(moved == want, f"tables: {what} moves the keys of {sorted(moved)}, not {sorted(want)} (the key reads the view selection reads)")
        # a manifest no recipe's view of which moved is printed under unmapped, never dropped
        quiet = just.replace("gate-whole:\n    ./tools/box.sh 'cargo build -p g --lib && bash tools/s.sh'\n", "")
        sels, unmapped, _ = select(["crates/g/Cargo.toml"], build(base, quiet), build(base + "# a comment\n", quiet), GATE_PREFIX)
        expect(not sels and len(unmapped) == 1 and "read by table" in unmapped[0], f"tables: a comment-only manifest edit gives {[x.recipe for x in sels]}, unmapped {unmapped}")
        # --narrow on an entries-only diff with an identical scan pair: the recipes whose own entries moved are
        # own picks (not a dependency read that would print the full list), and bin x's kernels, which no scan
        # covers, pick nothing: a new entry changes no kernel
        lines = ["ptx-scan: mod1 bundle=bloomery-gpu bytes=100",
                 "ptx-scan bin=target/release/gx section=.oxart bytes=9 ptxas=/p ptxas-version=13.3.73 arch=sm_86 modules=1 "
                 "jit-card=NVIDIA_RTX_A6000 jit-cc=8.6 jit-cuda=13.4",
                 "entry reqntid depot ld.local st.local fma cvt.f16 regs smem spill blk/SM(static) jit_regs jit_local",
                 f"{'alpha':<24} 256 no 0 0 0 0 12 0 0 6 12 0", "ptx-scan-md5: method=decl1", f"alpha {'1' * 32} 41"]
        log = os.path.join(tmp, "scan.log")
        with open(log, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        pair = (read_scan(log), read_scan(log))
        na, nb = build(base, quiet), build(base + bin_z, quiet)
        nar = narrow(["crates/g/Cargo.toml"], na, nb, [pair], [])
        expect(not nar.full and set(nar.picks) == {"gate-bins", "gate-plain"},
               f"tables: --narrow on a new entry gives full {nar.full}, picks {sorted(nar.picks)} (want gate-bins gate-plain)")
        # a manifest that does not parse: a named error in selection and in the key, never the whole file
        broken = build(base + "[[bin]\n", meta_toml=base)
        try:
            select(["crates/g/Cargo.toml"], a, broken, GATE_PREFIX)
            fails = "accepted"
        except RecipeError as err:
            fails = str(err)
        expect("crates/g/Cargo.toml does not parse as TOML" in fails, f"tables: a manifest that does not parse, in selection: {fails}")
        kb = keys(broken)
        expect(all(kb[n].startswith("error: crates/g/Cargo.toml does not parse as TOML") for n in every - {"gate-whole"}),
               f"tables: a manifest that does not parse, in the key: {kb}")


def key_self_test(expect, real: Side) -> None:
    """The ledger key: the detectors on the real tree, then a scratch copy where each input class is
    changed in turn — the same inputs give the same key, each change moves exactly the keys that read it."""
    meta = cargo_metadata(ROOT)
    settings = justfile_settings(os.path.join(ROOT, "justfile"))
    ctx = KeyContext(real, None, settings=settings)
    gates = [n for n in real.recipes if n.startswith(GATE_PREFIX)]
    never, tree_scoped = set(), set()
    never_only_readers: list[tuple[str, str]] = []
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
                hit = scan_file(real.tree.root, f, DATA_NEVER_ONLY_NAMED)
                if hit is not None:
                    never_only_readers.append((n, hit))
    for n, hit in never_only_readers:
        expect(n in never, f"{n} reads {hit}: an entry the box manifest leaves out (DATA_NEVER_ONLY), and {n} can skip — take it out of DATA_NEVER_ONLY")
    expect(not never_only_readers, f"{never_only_readers} name the tokenizer*/ directories of the format before the trailer: the box manifest leaves them out, and the sets gate-tokenizer reads are ref-tokenizer-*")
    expect(never == {"gate-1-1"}, f"never-skip gate recipes: {sorted(never)}")
    expect(
        tree_scoped <= never,
        f"skippable gate-* recipes keyed on the whole tree: {sorted(tree_scoped - never)} — a script in the closure walks the tree",
    )
    for n in ("check", "fmt-check", "check-recipes", "check-comments", "check-levers", "check-arch", "check-rustflags"):
        expect(ctx.scope(recipe_closure(real.recipes, n), real.graph.inputs(n))[0] == "tree", f"{n} is not keyed on the whole tree")
    expect(ctx.scope(["lint"], real.graph.inputs("lint"))[0] == "closure", "lint (clippy, modeled) is keyed on the whole tree")
    for s in ("tools/check-comments.sh", "tools/check-levers.sh", "tools/check-arch.sh", "crates/tokenizer/tools/oracle.sh"):
        expect(scan_lines(real.tree.read(s), TREE_WALK, s) is not None, f"the walk detector misses {s}")
    quiet = ["tools/box.sh", "tools/gate.sh", "tools/gpu-gate.sh", "tools/host-gate.sh", "tools/ref/ref-paths.sh", "tools/ptx-spill-check.sh", "tools/ptx-scan.sh"]
    quiet += sorted(os.path.relpath(p, ROOT) for p in glob.glob(os.path.join(ROOT, "tools/ref/models/*.sh")))
    for s in quiet:
        hit = scan_lines(real.tree.read(s), TREE_WALK, s)
        expect(hit is None, f"the walk detector flags {hit}: every gate that runs it would be keyed on the whole tree")

    with tempfile.TemporaryDirectory(prefix="recipes-keytest-") as tmp:
        roots = [os.path.join(tmp, "a")]
        for rel in shipped_files(ROOT):
            src, dst = os.path.join(ROOT, rel), os.path.join(roots[0], rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.islink(src):
                os.symlink(os.readlink(src), dst)
            else:
                shutil.copyfile(src, dst)
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
        nfixed = len(manifest)
        # Every other directory the tree names, as the box's model_dirs() lists it: an item whose
        # closure names a directory with no row is an error, not a key.
        manifest[-1:] = [f"model-dir\t{d}\tabsent" for d in sorted(tree_model_dirs(roots[0]) - {"/models/small"})] + ["# end"]

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
        sa = side_at(roots[0], meta)

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
        # The same tree at a second absolute path: the copy renamed there and back (a symlink would not
        # do — Tree takes the realpath).
        other = os.path.join(tmp, "b")
        os.rename(roots[0], other)
        try:
            elsewhere = keys(side_at(other, meta))
        finally:
            os.rename(other, roots[0])
        expect(elsewhere == base, "a second checkout at another path gives other keys (a path is not relative)")

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
        # The tier: one item under the real tier and under the fixture tier is two keys, whichever way the tier is named, and unset
        # and `real` are one — a fixture green must never stand for the real file's, nor `--tier real` split the real greens.
        fx_env = {"BLOOMERY_TIER": "fixture"}
        expect(moved(keys(sa, environ=fx_env)) == set(items), "the Mac's BLOOMERY_TIER=fixture does not move every key")
        expect(moved(keys(sa, box_env="BLOOMERY_TIER=fixture")) == set(items), "a BLOOMERY_TIER entry of the box env does not move every key")
        expect(not moved(keys(sa, environ={"BLOOMERY_TIER": "real"})), "BLOOMERY_TIER=real moves a key: it is the unset value")
        expect(not moved(keys(sa, environ={"BLOOMERY_TIER": ""})), "an empty BLOOMERY_TIER moves a key: it is the unset value")
        expect(keys(sa, environ=fx_env) != keys(sa, environ={"BLOOMERY_TIER": "bad"}), "an unnamed tier keys like the fixture tier")
        # The Mac-env part itself (a fixture tier's manifest rows move the key as well, so a moved key alone would not show a lost part).
        mac_row = lambda e: [r for r in KeyContext(sa, mf, environ=e).mac if r.startswith("mac-env\tBLOOMERY_TIER\t")]  # noqa: E731
        expect("BLOOMERY_TIER" in MAC_ENV, "BLOOMERY_TIER is not in MAC_ENV")
        expect(mac_row(fx_env) == ["mac-env\tBLOOMERY_TIER\tfixture"], f"the Mac's BLOOMERY_TIER=fixture is not a key part: {mac_row(fx_env)}")
        expect(mac_row({}) == mac_row({"BLOOMERY_TIER": "real"}) == mac_row({"BLOOMERY_TIER": ""}) == ["mac-env\tBLOOMERY_TIER\treal"],
               f"unset, empty and `real` are not one key part: {mac_row({})} {mac_row({'BLOOMERY_TIER': 'real'})} {mac_row({'BLOOMERY_TIER': ''})}")
        # ... and the ledger reader: a green recorded at the fixture key does not skip the real item, and skips the fixture one.
        k_real, k_fix = (keys(sa, environ=e, only={"x": items["gpu"]})["x"] for e in ({}, fx_env))
        with tempfile.TemporaryDirectory(prefix="recipes-tier-ledger-") as ld:
            lp = os.path.join(ld, "ledger.tsv")
            with open(lp, "w", encoding="utf-8") as fh:
                fh.write(f"{k_fix}\tgate-gpu-q4k-sel\t{items['gpu']}\tc1\t2026-10-07T09:00:00+0900\t/t/fx\n")
            st_real, _ = ledger_status(k_real, "gate-gpu-q4k-sel", items["gpu"], [], [Ledger(lp)])
            st_fix, _ = ledger_status(k_fix, "gate-gpu-q4k-sel", items["gpu"], [], [Ledger(lp)])
        expect(st_real == "run" and st_fix == "skip", f"a fixture-tier green and the ledger: the real item {st_real}, the fixture one {st_fix}")
        # The fixture directory's rows belong to the fixture tier's keys alone: a real key does not move with them, a fixture key
        # does, and the row of a file inside a family's directory is one (the manifest lists below a subdirectory there).
        fx_file = f"model\t{FIXTURE_ROOT}/qwen38/q-00001-of-00001.gguf\t100\t1\t2\t3"
        fixture_free = [ln for ln in manifest[:-1] if not ln.startswith(f"model-dir\t{FIXTURE_ROOT}\t")]
        write_manifest(fixture_free + ["# end"])
        real_before, fix_before = keys(sa), keys(sa, environ=fx_env)
        write_manifest(fixture_free + [fx_file, "# end"])
        expect(not moved(keys(sa)), "a fixture file's manifest row moved a real-tier key")
        expect(keys(sa, environ=fx_env) != fix_before and all(keys(sa, environ=fx_env)[k] != fix_before[k] for k in items),
               "a fixture file's manifest row did not move every fixture-tier key")
        write_manifest(fixture_free + [fx_file.replace("\t100\t", "\t101\t"), "# end"])
        expect(all(keys(sa, environ=fx_env)[k] != fix_before[k] for k in items), "a fixture file rewritten (its size) moved no fixture-tier key")
        write_manifest(manifest)
        expect(keys(sa) == base, "the base manifest, restored, keys another way")
        for i in range(1, nfixed - 1):
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
        # The `any` card in a script the recipe runs (ptx-scan.sh's JIT) picks the card at run time as it does
        # in the recipe's text: with no card forced the item never skips; forced, it can.
        for n in ("gate-ptx-spill",):
            expect(c.parts(n)[2] is not None, f"{n} with no card forced was skippable: its script's any-card call picks the card at run time")
            expect(c.parts(f"{n}@BLOOMERY_GATE_CARD=a6000")[2] is None, f"{n} with the A6000 forced never skips: {c.parts(n + '@BLOOMERY_GATE_CARD=a6000')[2]}")
        expect(c.parts("gate-sampler@BLOOMERY_REF_MODEL=/models/small/x.gguf")[2] is not None, "an item env naming a path was skippable")
        c2 = KeyContext(sa, mf, "BLOOMERY_KLD_FILE=/root/x.kld", {}, settings=settings)
        expect(c2.parts(items["cpu"])[2] is not None, "a caller env path outside data and /models was skippable")
        c3 = KeyContext(sa, mf, "BLOOMERY_DATA=/root/bloomery-data", {}, settings=settings)
        expect(c3.parts(items["cpu"])[2] is None, "the data directory itself made an item never-skip")
        expect(c.parts("gate-1-1")[2] is not None, "an ik-reading gate was skippable")
        expect(c.parts("gate-tokenizer")[2] is None, f"gate-tokenizer reads its sets, no reference tree, and was never-skip: {c.parts('gate-tokenizer')[2]}")
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
        expect(c.parts("gate-gpu-e2e@BLOOMERY_GATE_CARD=a6000")[2] is None and c.parts("gate-gpu-hybrid")[2] is None, "a card-determined item was never-skip")
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
        model_scope_self_test(expect, roots[0], sa, settings, manifest, os.path.join(tmp, "scope-manifest.txt"))
        # New-code clauses (no base to fail on): a directory the item names with no manifest row is an
        # error; the manifest's union reaches a whole-tree item; every profile spells what it defines.
        write_manifest([ln for ln in manifest if not ln.startswith("model-dir\t/models/DeepSeek-V4.1-Flash-Q3_K_M\t")])
        miss = keys(sa, only={"gpu": items["gpu"]})["gpu"]
        expect(miss.startswith("error") and "/models/DeepSeek-V4.1-Flash-Q3_K_M" in miss, f"a directory the item opens, missing from the manifest, was keyed: {miss}")
        write_manifest(manifest)
        c = KeyContext(sa, mf, "", {}, settings=settings)
        rows = [p for p in c.parts(items["mac"])[0] if p.startswith(("box\tmodel\t", "box\tmodel-dir\t"))]
        all_rows = [ln for ln in manifest if ln.startswith(("model\t", "model-dir\t"))]
        tier_rows = [ln for ln in all_rows if model_dir_of(ln.split("\t")[1]) == FIXTURE_ROOT]
        expect(len(rows) == len(all_rows) - len(tier_rows), f"a whole-tree item holds {len(rows)} model rows, not the manifest's all but the fixture tier's {len(tier_rows)}")
        fx_rows = [p for p in KeyContext(sa, mf, "BLOOMERY_TIER=fixture", {}, settings=settings).parts(items["mac"])[0] if p.startswith(("box\tmodel\t", "box\tmodel-dir\t"))]
        expect(len(fx_rows) == len(all_rows), f"a whole-tree item of the fixture tier holds {len(fx_rows)} model rows, not the manifest's all")
        penv = {k: v for k, v in os.environ.items() if k not in ("BLOOMERY_REF_MODEL", "BLOOMERY_V41_MODEL", "BLOOMERY_DSPARK_MODEL")}
        for prof in sorted(glob.glob(os.path.join(roots[0], "tools/ref/models/*.sh"))):
            rel = os.path.relpath(prof, roots[0])
            try:
                expect(bool(profile_dirs(roots[0], rel, penv)), f"{rel} defines no /models directory")
            except RecipeError as err:
                expect(False, f"profile_dirs: {err}")
        dyn = os.path.join(roots[0], "tools/ref/models/zz-dynamic.sh")
        with open(dyn, "w", encoding="utf-8") as fh:
            fh.write("N=Dyn\nMODEL=/models/$N/x.gguf\n")
        try:
            profile_dirs(roots[0], "tools/ref/models/zz-dynamic.sh", penv)
            expect(False, "a profile building a /models path no literal names was accepted")
        except RecipeError as err:
            expect("/mod" "els/Dyn" in str(err), f"the dynamic profile's error does not name the directory: {err}")
        finally:
            os.remove(dyn)


def model_scope_self_test(expect, root: str, side: Side, settings: dict, manifest: list[str], mf: str) -> None:
    """FAIL-first for the per-item model scope (each clause red on a key that carries every manifest row):
    (a) a directory appearing in the manifest moves only the keys whose closure names it; (b) a fixture
    literal in one crate reaches no gate that does not compile that crate; (c) a /models inside a longer
    path (`tools/ref/models/x.sh`) is not a model path."""
    gpu, cpu = "gate-gpu-q4k-sel@BLOOMERY_GATE_CARD=3090", "gate-sampler"
    # Spelled in two pieces, so that this file's own text names none of these directories: a literal
    # here would put an `absent` row into every real box manifest.
    M = "/mod" "els/"

    def key_parts(lines: list[str], item: str) -> list[str]:
        with open(mf, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        parts, err, _, _ = KeyContext(side, mf, "", {}, settings=settings).parts(item)
        expect(err is None, f"model scope: {item}: {err}")
        return parts

    def with_rows(*rows: str) -> list[str]:
        return manifest[:-1] + list(rows) + ["# end"]

    def planted(rel: str, text: str, fn) -> None:
        p = os.path.join(root, rel)
        with open(p, encoding="utf-8") as fh:
            old = fh.read()
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(old + text)
        try:
            fn()
        finally:
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(old)

    # (a) PlantedA is named by gate_q4k_sel.rs only; the box fetches it (absent -> a file row).
    def case_a() -> None:
        before = with_rows(f"model-dir\t{M}PlantedA\tabsent")
        after = with_rows(f"model\t{M}PlantedA/w.gguf\t1\t2\t3\t4")
        expect(item_key(key_parts(before, gpu)) != item_key(key_parts(after, gpu)), "model scope (a): a directory the closure names appeared and its key did not move")
        expect(item_key(key_parts(before, cpu)) == item_key(key_parts(after, cpu)), f"model scope (a): {M}PlantedA, which gate-sampler's closure does not name, appeared and moved its key")
    planted("crates/gpu-gates/src/bin/gate_q4k_sel.rs", f'\n// "{M}PlantedA/w.gguf"\n', case_a)

    # (b) a refusal-test fixture in crates/sampler's tests: gate-sampler compiles it, gate-gpu-q4k-sel does not.
    clean = item_key(key_parts(manifest, gpu))

    def case_b() -> None:
        rows = with_rows(f"model-dir\t{M}FixtureB\tabsent")
        expect(item_key(key_parts(rows, gpu)) == clean, "model scope (b): a fixture literal in crates/sampler's tests moved gate-gpu-q4k-sel's key")
        expect(any(f"{M}FixtureB" in p for p in key_parts(rows, cpu)), "model scope (b): gate-sampler, which compiles the fixture, does not hold its row")
    planted("crates/sampler/tests/sampler.rs", f'\n// "{M}FixtureB/other.gguf"\n', case_b)

    # (c) the anchor: a path that only contains /models/ is not one.
    for text in ("tools/ref/models/glm5next.sh", "source tools/ref/models/deepseek41.sh", "${BASH_SOURCE[0]%/*}/models/x.sh", "$HOME/models/a", "crates/x/models/y.rs", "~/models/z"):
        expect(MODEL_LITERAL.findall(text) == [], f"model scope (c): {text!r} read as a /models path: {MODEL_LITERAL.findall(text)}")
    for text, want in (('"' + M + 'X/a.gguf"', M + "X/a.gguf"), ("=" + M + "X", M + "X"), ("${V:-" + M + "X/a}", M + "X/a"), (" " + M + "X", M + "X"), ("# model\\t" + M + "X/a", M + "X/a")):
        expect(MODEL_LITERAL.findall(text) == [want], f"model scope (c): {text!r} gave {MODEL_LITERAL.findall(text)}, not [{want!r}]")
    profiles = {"/models/" + os.path.basename(p) for p in glob.glob(os.path.join(root, "tools/ref/models/*.sh"))}
    for rel in shipped_files(root):
        if rel == "justfile" or rel.endswith((".rs", ".sh", ".py", ".toml", ".tsv")):
            with open(os.path.join(root, rel), encoding="utf-8", errors="replace") as fh:
                bad = profiles & set(MODEL_LITERAL.findall(fh.read()))
            expect(not bad, f"model scope (c): {rel} names {sorted(bad)[0] if bad else ''}, a profile's path read as a model directory")




def orphan_self_test(expect, real: Side) -> None:
    """orphan-tests: the libtest argument reader, the scan and the run rule on a synthetic crate, then
    FAIL-first on the real tree — a recipe removed, a crate with a planted feature-gated test, a filter
    that matches nothing."""
    # libtest arguments: filters, --skip, --exact, the ignore modes; a refusal names what it refuses
    a = libtest_args(parse_cargo(["test", "-p", "p", "pre", "--", "--ignored", "x::", "--skip", "x::slow", "--test-threads=1", "--nocapture"], "t"))
    expect(a.filters == ["pre", "x::"] and a.skips == ["x::slow"] and a.ignored == "only" and a.runs, f"libtest args: {a}")
    expect(a.misses("x::fast", True) is None and "--skip x::slow" in (a.misses("x::slow", True) or ""), "libtest args: --skip")
    expect("not #[ignore]d" in (a.misses("x::fast", False) or ""), "libtest args: --ignored ran a plain test")
    e = libtest_args(parse_cargo(["test", "--", "--exact", "m::t"], "t"))
    expect(e.misses("m::t", False) is None and e.misses("m::t2", False) is not None, "libtest args: --exact")
    expect("#[ignore]d" in (libtest_args(parse_cargo(["test"], "t")).misses("t", True) or ""), "libtest args: a plain call ran an #[ignore]d test")
    for words, why in [
        (["--", "--ignored", "--include-ignored"], "together"),
        (["--", "--frobnicate"], "not known"),
        (["--", "{{ARGS}}"], "recipe parameter"),
        (["{{F}}"], "recipe parameter"),
        (["--", "--skip"], "without a value"),
    ]:
        try:
            libtest_args(parse_cargo(["test"] + words, "t"))
            fails = f"libtest args: {words} accepted"
        except RecipeError as err:
            fails = "" if why in str(err) else f"libtest args: {words} refused without '{why}': {err}"
        expect(not fails, fails)

    # The libtest value options have one owner, tools/gate.sh's LIBTEST_VALUE line. This file reads it, an
    # option added there is one here, a line the reader cannot take is refused by name, and gate.sh's own
    # walk picks the same filters as libtest_args from every recipe's test arguments and the edge cases.
    with open(os.path.join(ROOT, GATE_SH), encoding="utf-8") as fh:
        gate_text = fh.read()
    vals = _libtest_value()
    expect("--skip" in vals, f"libtest value options read from {GATE_SH}: {sorted(vals)}")
    line = next((ln for ln in gate_text.splitlines() if ln.startswith("LIBTEST_VALUE=(")), "LIBTEST_VALUE=()")
    expect(libtest_value_options(gate_text.replace(line, line[:-1] + " --frobnicate)")) == vals | {"--frobnicate"}, f"an option added to {GATE_SH}'s LIBTEST_VALUE is not read")
    for text, why in [
        (gate_text.replace(line, ""), "has 0 `LIBTEST_VALUE"),
        (gate_text.replace(line, line + "\n" + line), "has 2 `LIBTEST_VALUE"),
        (gate_text.replace(line, "LIBTEST_VALUE=()"), "not a list of plain options"),
        (gate_text.replace(line, 'LIBTEST_VALUE=(--skip "$X")'), "not a list of plain options"),
        (gate_text.replace(line, line[:-1] + " --skip)"), "repeats an option"),
        (gate_text.replace(line, line[:-1] + " --nocapture)"), "names a flag"),
    ]:
        try:
            libtest_value_options(text)
            fails = f"libtest value options: a gate.sh text wanting '{why}' was accepted"
        except RecipeError as err:
            fails = "" if why in str(err) else f"libtest value options: refused without '{why}': {err}"
        expect(not fails, fails)
    calls = set()
    for r in real.recipes.values():
        try:
            invs = recipe_commands(r).invocations  # a recipe the parser refuses is check()'s error
        except RecipeError:
            continue
        calls |= {("--",) + tuple(inv.test_args) for inv in invs if inv.sub == "test" and not any("{{" in w for w in inv.test_args)}
    expect(len(calls) >= 10, f"only {len(calls)} recipe test argument lists to hold gate.sh's walk to")
    calls |= {
        ("--",), ("--", "f"), ("--", "--skip", "f"), ("--", "--skip=f", "g"), ("--", "--test-threads", "1", "f"),
        ("--", "--list", "f"), ("--", "-Z", "unstable-options", "--format", "json", "f"), ("--", "--exact", "a::b", "c"),
        ("--", "--color", "never", "--logfile", "l", "--shuffle-seed", "3"), ("--", "--ignored", "--nocapture", "a", "b"),
    } | {("--", v, "x", "f") for v in vals}
    with tempfile.TemporaryDirectory(prefix="recipes-gatewalk-") as stub:
        # a `timeout` that runs nothing: gate.sh then names its filters in its exit-78 line, or exits 0 with none
        with open(os.path.join(stub, "timeout"), "w", encoding="utf-8") as fh:
            fh.write("#!/bin/sh\nexit 0\n")
        os.chmod(os.path.join(stub, "timeout"), 0o755)
        env = {k: v for k, v in os.environ.items() if k != "BLOOMERY_GATE_BOUND"}
        env["PATH"] = stub + os.pathsep + env.get("PATH", "")
        for c in sorted(calls):
            p = subprocess.run(["bash", os.path.join(ROOT, GATE_SH), *c], env=env, capture_output=True, text=True)
            m = re.search(r"the filter (.*) matched no test", p.stderr)
            walk = m.group(1).split(" ") if p.returncode == 78 and m else [] if p.returncode == 0 else None
            try:
                a = libtest_args(parse_cargo(["test", *c], "t"))
                want = [] if "--list" in c else a.filters
            except RecipeError as err:
                want = f"refused: {err}"
            expect(walk == want, f"{GATE_SH}'s walk and libtest_args disagree on {' '.join(c)}: gate.sh {walk if walk is not None else p.stderr.strip()}, recipes.py {want}")

    # a synthetic crate: file and inline modules, cfgs on the path, #[ignore], an integration target, and
    # every shape the scan refuses by name
    with tempfile.TemporaryDirectory(prefix="recipes-orphan-") as tmp:
        files = {
            "p/Cargo.toml": "",
            "p/src/lib.rs": (
                "//! #[test] in a doc comment is not a test\n"
                "#![allow(dead_code)]\n"
                "pub fn f() -> Result<(), String> { Ok(()) }\n"
                "#[test]\nfn plain() {}\n"
                '#[test]\n#[ignore = "hw"]\nfn hw_ign() {}\n'
                '#[cfg(feature = "f")]\nmod feat {\n    #[test]\n    fn t() {}\n}\n'
                '#[cfg(feature = "g")]\nmod on {\n    #[test]\n    fn t() {}\n}\n'
                '#[cfg(target_os = "macos")]\nmod mac {\n    #[test]\n    fn t() {}\n}\n'
                "#[cfg(test)]\nmod m;\n"
                '#[path = "other.rs"]\nmod moved;\n'
                'const S: &str = "#[test] fn fake() {}";\n'
            ),
            "p/src/m.rs": "#[test]\nfn file_mod() {}\nmod deep {\n    #[test]\n    fn t() { let _ = '{'; }\n}\n",
            "p/src/other.rs": '#[cfg(all(unix, not(windows)))]\n#[test]\nfn via_path() {}\n',
            "p/tests/it.rs": "mod common;\n#[test]\nfn integ() {}\n",
            "p/tests/common/mod.rs": "#[test]\nfn shared() {}\n",
            "q/Cargo.toml": "",
            "q/src/lib.rs": (
                '#[cfg_attr(feature = "g", ignore)]\n#[test]\nfn cond() {}\n'
                "macro_rules! mk {\n    () => {\n        #[test]\n        fn gen() {}\n    };\n}\n"
                "#[tokio::test]\nasync fn tok() {}\n"
                '#[cfg(target_feature = "avx2")]\n#[test]\nfn avx() {}\n'
                "#[cfg(foo bar)]\n#[test]\nfn junk() {}\n"
                'include!("gen.rs");\n'
            ),
        }
        for rel, text in files.items():
            os.makedirs(os.path.join(tmp, os.path.dirname(rel)), exist_ok=True)
            with open(os.path.join(tmp, rel), "w", encoding="utf-8") as fh:
                fh.write(text)

        def pkg(name: str, targets: list[tuple[str, str, str]], features: dict) -> dict:
            return {
                "id": name,
                "name": name,
                "manifest_path": os.path.join(tmp, name, "Cargo.toml"),
                "features": features,
                "dependencies": [],
                "targets": [{"kind": [k], "name": n, "src_path": os.path.join(tmp, name, s)} for k, n, s in targets],
            }

        meta = {
            "workspace_root": tmp,
            "workspace_members": ["p", "q"],
            "packages": [
                pkg("p", [("lib", "p", "src/lib.rs"), ("test", "it", "tests/it.rs")], {"f": [], "g": [], "default": []}),
                pkg("q", [("lib", "q", "src/lib.rs")], {"g": []}),
            ],
        }
        st = Tree(tmp, meta)

        def rec(name: str, cmd: str) -> Recipe:
            return Recipe(name, [f"./tools/box.sh '{cmd}'"], [], "")

        rs = {
            "gate-p": rec("gate-p", "bash tools/gate.sh -p p --lib -- --include-ignored"),
            "gate-p-g": rec("gate-p-g", "bash tools/gate.sh -p p --features g --lib -- on::"),
            "check-p": rec("check-p", "bash tools/gate.sh -p p --test it"),
            "gate-q": rec("gate-q", "bash tools/gate.sh -p q --lib"),
        }
        orphans, errors, checked = orphan_tests(st, rs)
        names = sorted(o.split(" ")[1] for o in orphans)
        expect(names == ["common::shared", "feat::t", "integ", "mac::t"], f"orphan scan: orphans {names}")
        expect(any("p/src/lib.rs:12 feat::t" in o and 'cfg(feature = "f") at p/src/lib.rs:9 is false' in o for o in orphans), f"orphan scan: feat::t's line {orphans}")
        expect(any(o.startswith("p/tests/it.rs:3 integ — p test it: no gate-* or lab-* recipe") for o in orphans), f"orphan scan: a check-* recipe counted as a runner {orphans}")
        expect(checked == 11, f"orphan scan: {checked} tests checked, not 11")
        for where, why in [
            ("q/src/lib.rs:1:", "cfg_attr"),
            ("q/src/lib.rs:6:", "macro_rules! mk"),
            ("q/src/lib.rs:10:", "tokio::test"),
            ("q/src/lib.rs:14 avx:", 'cfg(target_feature = "avx2") at q/src/lib.rs:12'),
            ("q/src/lib.rs:15:", "cfg(foo bar)"),
            ("q/src/lib.rs:18:", "include!"),
        ]:
            expect(any(x.startswith(where) and why in x for x in errors), f"orphan scan: no named error at {where} ({why}): {errors}")
        expect(len(errors) == 6, f"orphan scan: {len(errors)} errors, not 6: {errors}")

    # the real tree: gate-ds41-bind runs every bind test (the deepseek41 feature, the bind:: filter);
    # how many there are is the scan's own count of them, not a number spelled here
    gg_lib = next(t.src for t in real.tree.packages["bloomery-gpu-gates"].targets if t.kind == "lib")
    n_bind = sum(1 for t in TestScan(real.tree).tests_of(gg_lib)[0] if t.path.startswith("bind::"))
    expect(n_bind > 0, "orphan scan: the tree holds no bind:: test for the FAIL-first clause to orphan")
    orphans, errors, checked = orphan_tests(real.tree, real.recipes)
    expect(not errors, f"orphan scan on the real tree: {errors[:3]}")
    expect(not any(" bind::tests::" in o for o in orphans), f"orphan scan: a bind test is an orphan on the real tree: {[o for o in orphans if 'bind::' in o]}")
    base = set(orphans)
    with open(os.path.join(ROOT, "justfile"), encoding="utf-8") as fh:
        text = fh.read()
    with tempfile.TemporaryDirectory(prefix="recipes-orphan-ff-") as tmp:
        p = os.path.join(tmp, "justfile")
        # (a) the recipe removed: every bind test red, each naming the cfg the plain lib run lacks
        m = re.search(r"^(?:\[[^\n]*\]\n)*gate-ds41-bind:\n(?:    .*\n)+", text, re.M)  # its attributes go with it
        if m is None:
            expect(False, "orphan FAIL-first: no gate-ds41-bind recipe to remove")
        else:
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(text[: m.start()] + text[m.end() :])
            got = set(orphan_tests(real.tree, load_justfile(p))[0]) - base
            bind = [o for o in got if " bind::tests::" in o]
            expect(len(bind) == n_bind and len(got) == n_bind and all('cfg(feature = "deepseek41") at crates/gpu-gates/src/lib.rs:' in o and "gate-gpu-gates-lib (no feature)" in o for o in bind), f"orphan FAIL-first (a): {sorted(got)}")
        # (c) a filter that matches nothing of its module: the tests only that filter reached go red
        for pat in (r"--lib -- (bind::)'", r"--ignored (hw_ds41_oracle) "):
            m = re.search(pat, text)
            if m is None:
                expect(False, f"orphan FAIL-first: no anchor {pat}")
                continue
            bad = m.group(1).rstrip(":_") + "x"
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(text[: m.start(1)] + bad + text[m.end(1) :])
            got = set(orphan_tests(real.tree, load_justfile(p))[0]) - base
            expect(got and all(f"its filter {bad} does not match" in o and m.group(1).rstrip(":") in o for o in got), f"orphan FAIL-first (c) {bad}: {sorted(got)}")
        # (b) a scratch copy of the crates with a test planted behind a feature no recipe enables
        copy_root = os.path.join(tmp, "tree")
        shutil.copytree(os.path.join(ROOT, "crates"), os.path.join(copy_root, "crates"), ignore=lambda d, names: [n for n in names if n == "target" or (os.path.isfile(os.path.join(d, n)) and not n.endswith(".rs"))])
        rel = os.path.join("crates", "levers", "src", "lib.rs")  # spelled in parts: a whole path here would join the closure of every script that names this file
        lib = os.path.join(copy_root, rel)
        with open(lib, encoding="utf-8") as fh:
            n_lines = fh.read().count("\n")
        with open(lib, "a", encoding="utf-8") as fh:
            fh.write('#[cfg(feature = "nonexistent")]\nmod x {\n    #[test]\n    fn t() {}\n}\n')
        got = set(orphan_tests(Tree(copy_root, cargo_metadata(ROOT)), real.recipes)[0]) - base
        want = f"{rel}:{n_lines + 4} x::t — bloomery-levers lib: gate-levers (no feature): cfg(feature = \"nonexistent\") at {rel}:{n_lines + 1} is false"
        expect(got == {want}, f"orphan FAIL-first (b): {sorted(got)}, not {want}")


def ledger_self_test(expect) -> None:
    """Two ledgers, the lead's and the rounds': a key green in either skips, the lead's first; the
    detail names the source before `tree=`; a key in neither runs; a malformed line is counted."""
    k_lead, k_round, k_both, k_none = ("a" * 64), ("b" * 64), ("c" * 64), ("d" * 64)
    with tempfile.TemporaryDirectory() as d:
        lead, rnd = os.path.join(d, "lead.tsv"), os.path.join(d, "round.tsv")
        with open(lead, "w") as fh:
            fh.write(f"{k_lead}\tgate-x\tgate-x\tc1\t2026-09-27T09:00:00+0900\t/t/main\n")
            fh.write(f"{k_both}\tgate-z\tgate-z\tc1\t2026-09-27T09:00:00+0900\t/t/main\n")
        with open(rnd, "w") as fh:
            fh.write(f"{k_round}\tgate-y\tgate-y\tc2\t2026-09-27T09:10:00+0900\t/t/round\n")
            fh.write(f"{k_both}\tgate-z\tgate-z\tc2\t2026-09-27T09:10:00+0900\t/t/round\n")
            fh.write("not a record\n")
        leds = [Ledger(lead), Ledger(rnd, "round")]
        expect(leds[1].bad == 1, f"round ledger malformed count {leds[1].bad}, want 1")
        st, det = ledger_status(k_lead, "gate-x", "gate-x", [], leds)
        expect(st == "skip" and "src=lead tree=/t/main" in det, f"lead key: {st} {det}")
        st, det = ledger_status(k_round, "gate-y", "gate-y", [], leds)
        expect(st == "skip" and "src=round tree=/t/round" in det, f"round key: {st} {det}")
        expect(det.split(" tree=")[0].endswith("src=round"), f"src must sit right before tree= (gate-batch.sh cuts at it): {det}")
        st, det = ledger_status(k_both, "gate-z", "gate-z", [], leds)
        expect(st == "skip" and "src=lead" in det, f"a key in both skips on the lead's: {st} {det}")
        st, det = ledger_status(k_round, "gate-y", "gate-y", [], leds[:1])
        expect(st == "run", f"the lead ledger alone must not skip a round's green: {st} {det}")
        st, det = ledger_status(k_none, "gate-q", "gate-q", [], leds)
        expect(st == "run" and "no green record of gate-q" in det, f"unknown item: {st} {det}")
        st, det = ledger_status(k_none, "gate-y", "gate-y", [], leds)
        expect(st == "run" and det.endswith("(src=round)"), f"history from the rounds' ledger is labelled: {st} {det}")


def manifest_rows_self_test(expect) -> None:
    """The fixture root's manifest rows reach below a family's directory: every file there has a row that moves when the file is
    rewritten; any other model directory lists its subdirectories as `dir` rows only, as before."""
    with tempfile.TemporaryDirectory(prefix="recipes-rows-") as tmp:
        fam = os.path.join(tmp, "qwen38")
        os.makedirs(os.path.join(fam, "draft"))
        for rel in ("qwen38/f-00001-of-00001.gguf", "qwen38/draft/f-draft.gguf", "top.txt"):
            with open(os.path.join(tmp, rel), "wb") as fh:
                fh.write(b"x" * 10)
        deep = manifest_model_rows(tmp, fixture_root=tmp)
        names = [os.path.relpath(r[1], tmp) for r in deep]
        expect(names == ["qwen38", "qwen38/draft", "qwen38/draft/f-draft.gguf", "qwen38/f-00001-of-00001.gguf", "top.txt"],
               f"fixture-root rows: {names}")
        flat_rows = manifest_model_rows(tmp, fixture_root=os.path.join(tmp, "elsewhere"))
        flat = [os.path.relpath(r[1], tmp) for r in flat_rows]
        expect(flat == ["qwen38", "top.txt"] and flat_rows[0][2] == "dir", f"another model directory's rows: {flat}")
        with open(os.path.join(fam, "f-00001-of-00001.gguf"), "ab") as fh:
            fh.write(b"y")
        expect(manifest_model_rows(tmp, fixture_root=tmp) != deep, "a file rewritten below the fixture root moved no row")
        expect(manifest_model_rows(os.path.join(tmp, "nowhere")) == [("model-dir", os.path.join(tmp, "nowhere"), "absent")], "an absent model directory has no row")
        os.makedirs(os.path.join(tmp, "empty"))
        expect(manifest_model_rows(os.path.join(tmp, "empty")) == [("model-dir", os.path.join(tmp, "empty"), "empty")], "an empty model directory has no row")
        # model_dirs on a tree that names no fixture directory itself: the fixture root is listed all the same.
        os.makedirs(os.path.join(tmp, "bare/tools/ref/models"))
        expect(FIXTURE_ROOT in model_dirs(os.path.join(tmp, "bare")), f"the box manifest does not list {FIXTURE_ROOT}")


def ref_paths_self_test(expect) -> None:
    """tools/ref/ref-paths.sh under BLOOMERY_TIER on a scratch tree (stub profiles, a fixture directory written with gguf-ranges.py's
    writer): the real tier changes nothing; the fixture tier resolves the family's fixture and its budget, a caller's own
    BLOOMERY_REF_MODEL wins, a family that is small stays on its file, and a family with no fixture, a directory with none or two,
    a file that is not a whole fixture and a tier that is neither are each a named non-zero exit."""
    src = os.path.join(ROOT, "tools/ref")
    spec = importlib.util.spec_from_file_location("gguf_ranges_t", os.path.join(src, "gguf-ranges.py"))
    gr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gr)
    with tempfile.TemporaryDirectory(prefix="recipes-refpaths-") as tmp:
        os.makedirs(os.path.join(tmp, "tools/ref/models"))
        for f in ("ref-paths.sh", "gguf-ranges.py"):
            shutil.copyfile(os.path.join(src, f), os.path.join(tmp, "tools/ref", f))
        for fam in ("qwen4exp", "deepseek2", "glm5next", "mimo2", "newfam"):
            with open(os.path.join(tmp, f"tools/ref/models/{fam}.sh"), "w", encoding="utf-8") as fh:
                fh.write(f"MODEL_NAME={fam}\nMODEL=${{BLOOMERY_REF_MODEL:-/models/real/{fam}.gguf}}\n")
        root = os.path.join(tmp, "fixtures")
        fam_dir = os.path.join(root, "qwen38")
        os.makedirs(fam_dir)
        first = os.path.join(fam_dir, "qwen38-fixture-00001-of-00001.gguf")

        def write(path: str, kvs: list[tuple[str, int, int]]) -> None:
            packed = [gr._kv_string("general.architecture", "qwen4exp")] + [gr._kv_scalar(k, t, v) for k, t, v in kvs]
            gr.write_gguf(path, packed, [("token_embd.weight", 0, [8, 3], b"\1" * 96)])

        whole = [("bloomery.fixture.version", 4, 1), ("bloomery.fixture.card_budget", 10, 6190000000)]
        write(first, whole)

        def run(profile: str, script: str, **env: str) -> tuple[int, str, str]:
            e = {"PATH": os.environ["PATH"], "HOME": tmp, "BLOOMERY_MODEL": profile, "BLOOMERY_FIXTURE_ROOT": root, **env}
            r = subprocess.run(["bash", "-c", f". tools/ref/ref-paths.sh && {script}"], cwd=tmp, env=e, capture_output=True, text=True)
            return r.returncode, r.stdout, r.stderr

        show = 'echo "$MODEL|$FIXTURE_FILE"'
        real = "/models/real/qwen4exp.gguf|"
        for env in ({}, {"BLOOMERY_TIER": "real"}, {"BLOOMERY_TIER": ""}):
            rc, out, err = run("qwen4exp", show, **env)
            expect(rc == 0 and out.strip() == real, f"ref-paths real tier {env}: {rc} {out!r} {err!r}")
        rc, out, err = run("qwen4exp", f"{show}; fixture_budget", BLOOMERY_TIER="fixture")
        expect(rc == 0 and out.split() == [f"{first}|{first}", "6190000000"], f"ref-paths fixture tier: {rc} {out!r} {err!r}")
        glm_dir = os.path.join(root, "glm5next")
        os.makedirs(glm_dir)
        glm_first = os.path.join(glm_dir, "glm53-fixture-00001-of-00001.gguf")
        write(glm_first, whole)
        rc, out, err = run("glm5next", f"{show}; fixture_budget", BLOOMERY_TIER="fixture")
        expect(rc == 0 and out.split() == [f"{glm_first}|{glm_first}", "6190000000"], f"ref-paths: glm5next resolves its own fixture directory: {rc} {out!r} {err!r}")
        rc, out, err = run("qwen4exp", show, BLOOMERY_TIER="fixture", BLOOMERY_REF_MODEL="/x/own.gguf")
        expect(rc == 0 and out.strip() == f"/x/own.gguf|{first}", f"ref-paths: a caller's own file does not win, or the fixture is not named: {rc} {out!r}")
        rc, out, err = run("deepseek2", f"{show}; fixture_budget; echo budget-rc=$?", BLOOMERY_TIER="fixture")
        expect(rc == 0 and out.split() == ["/models/real/deepseek2.gguf|", "budget-rc=0"], f"ref-paths: a small family does not stand: {rc} {out!r} {err!r}")
        for fam in ("mimo2", "newfam"):
            rc, out, err = run(fam, show, BLOOMERY_TIER="fixture")
            expect(rc == 66 and out == "" and f"the family '{fam}' has no fixture yet" in err and "exit 66" in err, f"ref-paths: {fam} has no fixture: {rc} {out!r} {err!r}")
        rc, out, err = run("qwen4exp", show, BLOOMERY_TIER="both")
        expect(rc == 64 and "BLOOMERY_TIER is real (unset: the same) or fixture, got 'both'" in err, f"ref-paths: a bad tier: {rc} {err!r}")
        rc, out, err = run("qwen4exp", show, BLOOMERY_TIER="fixture", BLOOMERY_FIXTURE_ROOT=os.path.join(tmp, "nowhere"))
        expect(rc == 66 and "has no fixture file" in err and "nowhere/qwen38/*-00001-of-*.gguf" in err, f"ref-paths: no directory: {rc} {err!r}")
        second = os.path.join(fam_dir, "other-00001-of-00002.gguf")
        write(second, whole)
        rc, out, err = run("qwen4exp", show, BLOOMERY_TIER="fixture")
        expect(rc == 66 and "holds 2 first shards" in err, f"ref-paths: two first shards: {rc} {out!r} {err!r}")
        os.remove(second)
        # a file that is not a whole fixture: the budget read is the named refusal
        for why, kvs in (("has no bloomery.fixture.version key", [("bloomery.fixture.card_budget", 10, 5)]),
                         ("carries bloomery.fixture.subset", whole + [("bloomery.fixture.subset", 4, 1)]),
                         ("records no positive bloomery.fixture.card_budget", [("bloomery.fixture.version", 4, 1)]),
                         ("records no positive bloomery.fixture.card_budget", [("bloomery.fixture.version", 4, 1), ("bloomery.fixture.card_budget", 10, 0)])):
            write(first, kvs)
            rc, out, err = run("qwen4exp", "fixture_budget", BLOOMERY_TIER="fixture")
            expect(rc == 65 and out == "" and why in err, f"ref-paths: a file that {why}: {rc} {out!r} {err!r}")
        with open(first, "wb") as fh:
            fh.write(b"not a gguf at all")
        rc, out, err = run("qwen4exp", "fixture_budget", BLOOMERY_TIER="fixture")
        expect(rc == 65 and "is not a GGUF file" in err, f"ref-paths: a file that is no GGUF: {rc} {out!r} {err!r}")


def box_tier_self_test(expect) -> None:
    """tools/box.sh's tier, on a scratch tree with a stub ssh and rsync that log the commands: with no tier named the remote command
    holds nothing of it; a tier named in the environment or in a BLOOMERY_BOX_ENV entry is exported before the first read of
    ref-paths.sh (an entry the profile had not seen would reach the binary and not the file); the fixture tier adds the budget and
    file reads; two tiers named, or one that is neither, end the command at 64 before anything reaches the box."""
    with tempfile.TemporaryDirectory(prefix="recipes-boxtier-") as tmp:
        os.makedirs(os.path.join(tmp, "tools"))
        os.makedirs(os.path.join(tmp, "stubs"))
        shutil.copyfile(os.path.join(ROOT, "tools/box.sh"), os.path.join(tmp, "tools/box.sh"))
        shutil.copyfile(os.path.join(ROOT, "Cargo.toml"), os.path.join(tmp, "Cargo.toml"))
        log = os.path.join(tmp, "ssh.log")
        for name, body in (("ssh", f'#!/bin/sh\nprintf \'%s\\n\' "$*" >> {shlex.quote(log)}\n'), ("rsync", "#!/bin/sh\n")):
            path = os.path.join(tmp, "stubs", name)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(body)
            os.chmod(path, 0o755)

        def run(**env: str) -> tuple[int, str, str]:
            if os.path.exists(log):
                os.remove(log)
            e = {"PATH": f"{os.path.join(tmp, 'stubs')}:/usr/bin:/bin", "HOME": tmp, "BLOOMERY_REMOTE": "~/repo/x", **env}
            r = subprocess.run(["bash", os.path.join(tmp, "tools/box.sh"), "echo hi"], env=e, capture_output=True, text=True)
            sent = open(log, encoding="utf-8").read() if os.path.exists(log) else ""
            return r.returncode, sent, r.stderr

        first = "tools/ref/ref-paths.sh"
        rc, sent, err = run()
        expect(rc == 0 and "BLOOMERY_TIER" not in sent and "fixture_budget" not in sent and "BLOOMERY_FIXTURE_MODEL" not in sent,
               f"box.sh with no tier names it in the remote command: {rc} {err!r}")
        rc, real, err = run(BLOOMERY_TIER="real")
        expect(rc == 0 and "export BLOOMERY_TIER=real &&" in real and "fixture_budget" not in real, f"box.sh BLOOMERY_TIER=real: {rc} {err!r}")
        for how, env in (("the environment", {"BLOOMERY_TIER": "fixture"}), ("BLOOMERY_BOX_ENV", {"BLOOMERY_BOX_ENV": "FOO=1 BLOOMERY_TIER=fixture"})):
            rc, sent, err = run(**env)
            at, ref = sent.find("export BLOOMERY_TIER=fixture &&"), sent.find(first)
            expect(rc == 0 and 0 <= at < ref, f"box.sh: the tier named in {how} is not exported before ref-paths.sh is read: {rc} {err!r}")
            expect("fixture_budget" in sent and "BLOOMERY_FIXTURE_MODEL" in sent and "BLOOMERY_CARD_BUDGET" in sent,
                   f"box.sh: the tier named in {how} does not take the fixture's budget and file: {sent[-400:]!r}")
        expect(sent.count("fixture_budget") == 1, f"box.sh reads the fixture's budget more than once: {sent.count('fixture_budget')}")
        rc, sent, err = run(BLOOMERY_TIER="fixture", BLOOMERY_BOX_ENV="BLOOMERY_TIER=real")
        expect(rc == 64 and sent == "" and "name one" in err, f"box.sh: two tiers named: {rc} {sent!r} {err!r}")
        rc, sent, err = run(BLOOMERY_BOX_ENV="BLOOMERY_TIER=both")
        expect(rc == 64 and sent == "" and "real or fixture" in err and "'both'" in err, f"box.sh: a tier that is neither: {rc} {sent!r} {err!r}")


def oxide_rev_self_test(expect) -> None:
    """The cuda-oxide rev the key (oxide_rev) and tools/box.sh (the backend directory it exports) read from this tree's
    Cargo.toml is the [patch] section's cuda-device rev as TOML reads it — the fork's, not the upstream declaration's,
    which both fall back to when their section header names another repository."""
    import tomllib

    with open(os.path.join(ROOT, "Cargo.toml"), "rb") as fh:
        man = tomllib.load(fh)
    patched = [t["cuda-device"]["rev"] for t in man.get("patch", {}).values() if "rev" in t.get("cuda-device", {})]
    want = patched[0] if patched else man["workspace"]["dependencies"]["cuda-device"]["rev"]
    with open(os.path.join(ROOT, "Cargo.toml"), encoding="utf-8") as fh:
        got = oxide_rev(fh.read())
    expect(got == want, f"oxide-rev: oxide_rev reads {got} from Cargo.toml, its cuda-device [patch] rev is {want}")
    with tempfile.TemporaryDirectory(prefix="recipes-oxrev-") as tmp:
        os.makedirs(os.path.join(tmp, "tools"))
        os.makedirs(os.path.join(tmp, "stubs"))
        shutil.copyfile(os.path.join(ROOT, "tools/box.sh"), os.path.join(tmp, "tools/box.sh"))
        shutil.copyfile(os.path.join(ROOT, "Cargo.toml"), os.path.join(tmp, "Cargo.toml"))
        log = os.path.join(tmp, "ssh.log")
        for name, body in (("ssh", f'#!/bin/sh\nprintf \'%s\\n\' "$*" >> {shlex.quote(log)}\n'), ("rsync", "#!/bin/sh\n")):
            path = os.path.join(tmp, "stubs", name)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(body)
            os.chmod(path, 0o755)
        e = {"PATH": f"{os.path.join(tmp, 'stubs')}:/usr/bin:/bin", "HOME": tmp, "BLOOMERY_REMOTE": "~/repo/x"}
        r = subprocess.run(["bash", os.path.join(tmp, "tools/box.sh"), "echo hi"], env=e, capture_output=True, text=True)
        sent = open(log, encoding="utf-8").read() if os.path.exists(log) else ""
        m = re.search(r"cuda-oxide-bloomery/([0-9a-f]+)/librustc_codegen_cuda\.so", sent)
        expect(r.returncode == 0 and m is not None and m.group(1) == want,
               f"oxide-rev: tools/box.sh exports the backend of {m.group(1) if m else None}, the cuda-device [patch] rev is {want}: {r.returncode} {r.stderr!r}")


def reach_self_test(expect, side: Side, row, stats: dict[str, int]) -> None:
    """The derivation (Reach) on a synthetic workspace, one file per rule it decides, and the table's check of it on
    the real tree: a draft row that leaves out a gate reaching its files through another lib, and the exceptions."""
    with tempfile.TemporaryDirectory(prefix="recipes-reach-") as tmp:
        files = {
            "base/Cargo.toml": "",
            # each of a.rs … h.rs holds one fn, reached (or not) by one rule
            "base/src/lib.rs": (
                "pub mod a;\npub mod b;\npub mod c;\npub mod d;\npub mod e;\npub mod f;\npub mod g;\npub mod h;\nmod imp;\nmod inner;\n"
                "pub use inner::{T, U};\n"
                "pub struct S;\n"
                "impl S {\n    pub fn used(&self) {\n        c::run()\n    }\n    pub fn idle(&self) {\n        d::run()\n    }\n}\n"
                "impl Drop for S {\n    fn drop(&mut self) {\n        e::run()\n    }\n}\n"
                "#[cfg(test)]\nimpl Drop for U {\n    fn drop(&mut self) {\n        f::run()\n    }\n}\n"
                "#[macro_export]\nmacro_rules! mk {\n    () => {\n        $crate::h::run()\n    };\n}\n"
            ),
            "base/src/inner.rs": "pub struct T;\npub struct U;\n",
            # an impl whose type path passes through lib.rs's re-export, as U's does
            "base/src/imp.rs": "impl crate::T {\n    pub fn go(&self) {\n        crate::a::run()\n    }\n}\n",
            "base/src/g.rs": "pub const K: u32 = 1;\n",
            "mid/Cargo.toml": "",
            "mid/src/lib.rs": "pub fn call() -> u32 {\n    base::b::run();\n    base::mk!();\n    base::g::K\n}\n",
            "app/Cargo.toml": "",
            "app/src/lib.rs": "",
            "app/tests/it.rs": (
                "mod common;\n#[test]\nfn t() {\n    let h = common::H;\n    h.make();\n"
                "    let s = base::S;\n    s.used();\n    let u = base::U;\n    u.go();\n}\n"
            ),
            "app/tests/common/mod.rs": "pub struct H;\nimpl H {\n    pub fn make(&self) {\n        mid::call();\n    }\n}\n",
        }
        for x in "abcdefh":
            files[f"base/src/{x}.rs"] = "pub fn run() {}\n"
        for rel, text in files.items():
            os.makedirs(os.path.join(tmp, os.path.dirname(rel)), exist_ok=True)
            with open(os.path.join(tmp, rel), "w", encoding="utf-8") as fh:
                fh.write(text)

        def pkg(name: str, targets: list[tuple[str, str, str]], deps: list[str]) -> dict:
            return {
                "id": name,
                "name": name,
                "manifest_path": os.path.join(tmp, name, "Cargo.toml"),
                "features": {},
                "dependencies": [{"name": d, "kind": None, "optional": False, "features": [], "uses_default_features": True} for d in deps],
                "targets": [{"kind": [k], "name": n, "src_path": os.path.join(tmp, name, s)} for k, n, s in targets],
            }

        meta = {
            "workspace_root": tmp,
            "workspace_members": ["base", "mid", "app"],
            "packages": [
                pkg("base", [("lib", "base", "src/lib.rs")], []),
                pkg("mid", [("lib", "mid", "src/lib.rs")], ["base"]),
                pkg("app", [("lib", "app", "src/lib.rs"), ("test", "it", "tests/it.rs")], ["mid", "base"]),
            ],
        }
        st = Tree(tmp, meta)
        rs = {"gate-app": Recipe("gate-app", ["./tools/box.sh 'bash tools/gate.sh -p app --test it'"], [], "")}
        sg = Graph(st, rs)
        got = {f for f in (f"base/src/{x}.rs" for x in "abcdefgh") if dep_reach(st, sg, ["gate-app"], f)}
        want = {
            "base/src/b.rs": "through mid's call, which only the test crate's own impl (H::make, rule 1 in its own root) runs",
            "base/src/c.rs": "S::used, named by the test (rule 1)",
            "base/src/e.rs": "S's Drop, a foreign trait's impl std calls unnamed (rule 1)",
            "base/src/g.rs": "the const mid names (rule 4)",
            "base/src/h.rs": "the #[macro_export] macro mid calls, its $crate path (rule 5)",
        }
        for f, why in want.items():
            expect(f in got, f"reach: {f} not reached — {why}")
        for f, why in {
            "base/src/a.rs": "T::go: the test names U, whose re-export T shares, and T itself is never named",
            "base/src/d.rs": "S::idle: no reached unit says its name (rule 1)",
            "base/src/f.rs": "U's Drop is #[cfg(test)] in base: a dependent's build has no such impl (rule 3)",
        }.items():
            expect(f not in got, f"reach: {f} reached — {why}")
        expect(bool(dep_reach(st, sg, ["gate-app"], "mid/src/lib.rs")), "reach: mid's call not reached through the test crate's own impl")

    # the table's check on the real tree: the held rows, and the derivation's own lines
    tree, recipes, graph = side.tree, side.recipes, side.graph
    expect(stats.get("rows", 0) > 0, f"gate-paths: the summary counts no row held to the derivation: {stats}")
    # a hand-written row that leaves out a gate reaching its files through another lib: gate-gpu-ds41-step runs
    # moe.rs's experts_step_union_into through bloomery_gpu's host tier, and its bin names no model module
    draft = ("gate-attn gate-ffn gate-moe gate-head gate-forward gate-kv gate-alloc gate-derived gate-mt gate-profile gate-union "
             "gate-ds41-host gate-r8 gate-gpu-hybrid gate-gpu-ds41-tier gate-gpu-nvtier gate-gpu-qwen4exp-e2e gate-gpu-glm5next-e2e "
             "gate-gpu-qwen38-serve gate-gpu-ds41-chain-ffn gate-gpu-qwen4exp-iqleg gate-gpu-p8b gate-gpu-glm5next-twocard "
             "gate-gpu-glm5next-serve").split(" ")
    got = gate_paths_problems(tree, recipes, [row("crates/model/src/{ops.rs,moe.rs}", draft)])
    step = [p for p in got if p.split(": ")[1].startswith("gate-gpu-ds41-step reaches crates/model/src/")]
    expect(len(step) == 1 and "the chain: " in step[0], f"gate-paths: the draft row's missing gate-gpu-ds41-step not named with its chain: {got[:2]}")
    expect(not any("gate-gpu-ds41-step runs" in p for p in got), "gate-paths: the direct-name rule, not the derivation, named gate-gpu-ds41-step")
    excepted = PathRow(1, "crates/model/src/{ops.rs,moe.rs}", draft, "why", glob_regex("crates/model/src/{ops.rs,moe.rs}"), {"gate-gpu-ds41-step": "a reason"})
    got = gate_paths_problems(tree, recipes, [excepted])
    expect(not any("gate-gpu-ds41-step" in p for p in got) and any(" reaches crates/model/src/" in p for p in got),
           f"gate-paths: an exception does not hold its gate out, or holds out the others: {[p for p in got if 'ds41-step' in p][:1]}")
    # an exception naming a gate that reaches no file of the glob is stale
    full = list(dep_reach(tree, graph, [n for n in recipes if n.startswith(GATE_PREFIX)], "crates/threads/src/lib.rs"))
    stale = PathRow(1, "crates/threads/src/lib.rs", full, "why", glob_regex("crates/threads/src/lib.rs"), {"gate-refset": "a reason"})
    got = gate_paths_problems(tree, recipes, [stale])
    expect(any(p.endswith("the exception names gate-refset, which reaches no file the glob matches through a dependency — take it out") for p in got)
           and not any(" reaches crates/" in p for p in got), f"gate-paths: a stale exception: {got[:2]}")
    # the fourth column's form
    expect(parse_exceptions("except gate-a gate-b: one | except gate-c: two") == {"gate-a": "one", "gate-b": "one", "gate-c": "two"},
           "gate-paths: two exception clauses")
    for bad, why in [("gate-a: no except", "not `except"), ("except gate-a:", "not `except"), ("except gate-a: x | except gate-a: y", "gate-a named twice")]:
        try:
            parse_exceptions(bad)
            expect(False, f"gate-paths: exceptions {bad!r} not refused")
        except RecipeError as err:
            expect(why in str(err), f"gate-paths: exceptions {bad!r} refused as {err}")


def narrow_self_test(expect, side: Side) -> None:
    """--narrow: the scan verdicts and their refusals, the four rules of narrow() on the real tree with rows of
    the test's own, the table's checks, and the host group's guard."""
    header = "entry reqntid depot ld.local st.local fma cvt.f16 regs smem spill blk/SM(static) jit_regs jit_local"
    h1, h2, h3 = "1" * 32, "2" * 32, "3" * 32

    def scan(tmp: str, name: str, rows: dict[str, str], banner_extra: str = "", bundles=("bloomery-gpu",), md5_rows=None, banners=1,
             cc: str = "8.6", nojit: bool = False, jit_banner: str | None = None, cells: str | None = None, regs: str = "12", bin: str = "gx") -> str:
        """A scan log. `nojit` makes it a `--no-jit` scan (the banner's one jit= field and two skipped cells);
        `jit_banner` and `cells` override the banner's JIT fields and the two JIT cells to build a log whose
        banner and cells disagree."""
        skipped = "skipped(no-jit)"
        jits = jit_banner if jit_banner is not None else (
            f"jit={skipped}" if nojit else f"jit-card=NVIDIA_RTX_A6000{' jit-cc=' + cc if cc else ''} jit-cuda=13.4")
        cells = cells if cells is not None else (f"{skipped} {skipped}" if nojit else "12 0")
        lines = ["./tools/box.sh 'cargo oxide build …'", "   Compiling bloomery-gpu v0.1.0 (/root/x/crates/gpu)"]
        lines += [f"ptx-scan: mod{i + 1} bundle={b} bytes=100" for i, b in enumerate(bundles)]
        for _ in range(banners):
            lines.append(f"ptx-scan bin=target/release/{bin} section=.oxart bytes=9 ptxas=/p ptxas-version=13.3.73 arch=sm_86 "
                         f"modules={len(bundles)} {jits}{banner_extra}")
        lines.append(header)
        lines += [f"{e:<24} 256 no 0 0 0 0 {regs} 0 0 6 {cells}" for e in rows]
        lines.append("ptx-scan-md5: method=decl1")
        lines += [f"{e} {h} 41" for e, h in (md5_rows if md5_rows is not None else rows).items()]
        path = os.path.join(tmp, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        return path

    def refused(fn, why: str) -> bool:
        try:
            fn()
        except RecipeError as err:
            return why in str(err)
        return False

    def row(glob: str, names: list[str], line: int = 1) -> PathRow:
        return PathRow(line, glob, names, "why", glob_regex(glob))

    with tempfile.TemporaryDirectory(prefix="recipes-narrow-") as tmp:
        base = read_scan(scan(tmp, "base.log", {"alpha": h1, "beta": h2}))
        same = read_scan(scan(tmp, "same.log", {"alpha": h1, "beta": h2}))
        add = read_scan(scan(tmp, "add.log", {"alpha": h1, "beta": h2, "gamma": h3}))
        md5 = read_scan(scan(tmp, "md5.log", {"alpha": h1, "beta": h3}))
        gone = read_scan(scan(tmp, "gone.log", {"alpha": h1}))
        v = [scan_verdict(base, x) for x in (same, add, md5, gone)]
        expect([x.kind for x in v] == ["identical", "added", "moved", "moved"], f"narrow: verdicts {[x.line() for x in v]}")
        expect(v[1].entries == ["gamma"] and v[2].entries == ["beta"] and v[3].entries == ["beta"], f"narrow: verdict entries {[x.entries for x in v]}")
        # a log the tool cannot vouch for is refused by name, never read as identical
        failed = os.path.join(tmp, "failed.log")
        with open(failed, "w", encoding="utf-8") as fh:
            fh.write("ptx-scan bin=target/release/gx scan=failed\n")
        expect(refused(lambda: read_scan(failed), "failed scan"), "narrow: a failed scan not refused")
        expect(refused(lambda: read_scan(scan(tmp, "f.log", {"alpha": h1}, " filter=alp")), "filtered"), "narrow: a filtered scan not refused")
        expect(refused(lambda: read_scan(scan(tmp, "two.log", {"alpha": h1}, banners=2)), "2 `ptx-scan bin=` banners"), "narrow: two scans in one log not refused")
        expect(refused(lambda: read_scan(scan(tmp, "cut.log", {"alpha": h1, "beta": h2}, md5_rows={"alpha": h1})), "only one of the table and the md5 block"),
               "narrow: a table row with no digest not refused")
        expect(refused(lambda: read_scan(os.path.join(tmp, "missing.log")), "missing.log"), "narrow: a missing log not refused by name")
        # the JIT columns follow the card's compute capability, not its name: a pair scanned on either
        # card of one capability is one table, a pair of two capabilities is refused, and an older
        # banner that names no capability is refused by name — its JIT columns are not comparable
        other = read_scan(scan(tmp, "card.log", {"alpha": h1, "beta": h2}))
        other.fields["jit-card"] = "NVIDIA_GeForce_RTX_3090"
        expect(scan_verdict(base, other).kind == "identical",
               "narrow: a pair scanned on two cards of one capability not read as identical")
        other.fields["jit-cc"] = "9.0"
        expect(refused(lambda: scan_verdict(base, other), "jit-cc="), "narrow: a pair scanned on two capabilities not refused")
        old = read_scan(scan(tmp, "old.log", {"alpha": h1, "beta": h2}, cc=""))
        expect(refused(lambda: scan_verdict(base, old), "names no jit-cc"),
               "narrow: a pair of one older banner (no jit-cc) and one new not refused")
        old2 = read_scan(scan(tmp, "old2.log", {"alpha": h1, "beta": h2}, cc=""))
        expect(scan_verdict(old, old2).kind == "identical", "narrow: two older banners of one card no longer compare")
        old2.fields["jit-card"] = "NVIDIA_GeForce_RTX_3090"
        expect(refused(lambda: scan_verdict(old, old2), "jit-card="), "narrow: two older banners of two cards not refused")
        other = read_scan(scan(tmp, "bin.log", {"alpha": h1, "beta": h2}))
        other.bin = "gy"
        expect(refused(lambda: scan_verdict(base, other), "a pair is two scans of one binary"), "narrow: a pair of two binaries not refused")
        # a `--no-jit` scan holds no JIT reading: a pair with one compares neither the jit-* fields nor the two JIT
        # columns, proves a move only (identical, said in the verdict) and is refused by name for anything else
        def tried(fn):
            """fn's value, or the refusal's text: a case reads what the tool said, not a traceback."""
            try:
                return fn()
            except RecipeError as err:
                return f"refused: {err}"

        def says(v, kind: str, detail: str) -> bool:
            return isinstance(v, ScanVerdict) and v.kind == kind and detail in v.line()

        nj = read_scan(scan(tmp, "nj.log", {"alpha": h1, "beta": h2}, nojit=True))
        nj_same = read_scan(scan(tmp, "nj_same.log", {"alpha": h1, "beta": h2}, nojit=True))
        v = tried(lambda: scan_verdict(nj, nj_same))
        expect(says(v, "identical", "JIT skipped: --no-jit both"), f"narrow: two equal no-jit scans: {v}")
        v = tried(lambda: scan_verdict(base, nj_same))
        expect(says(v, "identical", "JIT skipped: --no-jit new"), f"narrow: a full base and a no-jit new: {v}")
        v = tried(lambda: scan_verdict(nj, base))
        expect(says(v, "identical", "JIT skipped: --no-jit base"), f"narrow: a no-jit base and a full new: {v}")
        full_cc = read_scan(scan(tmp, "full9.log", {"alpha": h1, "beta": h2}, cc="9.0", cells="99 7"))
        v = tried(lambda: scan_verdict(full_cc, nj_same))
        expect(says(v, "identical", ""), f"narrow: a full scan of another capability and other JIT cells against a no-jit scan: {v}")
        old_banner = read_scan(scan(tmp, "oldnj.log", {"alpha": h1, "beta": h2}, cc=""))
        v = tried(lambda: scan_verdict(old_banner, nj_same))
        expect(says(v, "identical", ""), f"narrow: a full older banner (no jit-cc) against a no-jit scan: {v}")
        refusal = "rescan without --no-jit: the add and occupancy classes need the JIT columns"
        for what, other in (("an added entry", {"alpha": h1, "beta": h2, "gamma": h3}), ("a moved digest", {"alpha": h1, "beta": h3}),
                            ("a removed entry", {"alpha": h1})):
            for pair in ((nj, read_scan(scan(tmp, "njx.log", other, nojit=True))), (base, read_scan(scan(tmp, "njy.log", other, nojit=True))),
                         (read_scan(scan(tmp, "njz.log", other)), nj)):
                v = tried(lambda: scan_verdict(*pair))
                expect(isinstance(v, str) and refusal in v, f"narrow: {what} in a pair with a no-jit scan not refused by name: {v if isinstance(v, str) else v.line()}")
        v = tried(lambda: scan_verdict(nj, read_scan(scan(tmp, "njb.log", {"alpha": h1, "beta": h2}, bundles=("bloomery-gpu", "bloomery-x"), nojit=True))))
        expect(isinstance(v, str) and refusal in v, f"narrow: a moved bundle set in a no-jit pair not refused by name: {v if isinstance(v, str) else v.line()}")
        # a log whose banner and cells disagree, a skipped cell outside the JIT columns, a banner that names no JIT
        # reading at all: refused when read, never read as a value
        for what, kw, why in (
            ("a no-jit banner over JIT readings", {"jit_banner": "jit=skipped(no-jit)"}, "holds no JIT value"),
            ("a full banner over skipped cells", {"cells": "skipped(no-jit) skipped(no-jit)"}, "the banner names the JIT"),
            ("a banner with no JIT field", {"jit_banner": "jit-cuda=13.4"}, "names no JIT reading"),
            ("a skipped cell in the regs column", {"nojit": True, "regs": "skipped(no-jit)"}, "only jit_regs and jit_local can be skipped"),
        ):
            v = tried(lambda: read_scan(scan(tmp, "bad.log", {"alpha": h1}, **kw)))
            expect(isinstance(v, str) and why in v, f"narrow: {what} not refused by name: {v if isinstance(v, str) else 'read'}")
        # narrow() on the real tree. A cargo global reaches every gate through a dependency edge and sits in no
        # kernel carrier's closure, so its row's recipes are its whole selection.
        rows = [row("rust-toolchain.toml", ["gate-sampler"]), row("crates/vision/src/**", ["gate-vision"], 2)]
        n = narrow(["rust-toolchain.toml"], side, side, [(base, same)], rows)
        expect(not n.full and set(n.picks) == {"gate-sampler", *NARROW_ALWAYS},
               f"narrow: an identical pair and a mapped file do not give the mapped recipes only: full={n.full} picks={sorted(n.picks)}")
        n = narrow(["rust-toolchain.toml"], side, side, [(base, add)], rows)
        expect(not n.full and set(n.picks) == {"gate-sampler", *NARROW_ALWAYS}, f"narrow: an added pair does not narrow: {n.full}")
        n = narrow(["rust-toolchain.toml"], side, side, [(base, md5)], rows)
        expect(any("ptx-scan gx: moved beta" in r for r in n.full), f"narrow: one md5 changed does not keep the full list: {n.full}")
        expect(set(NARROW_ALWAYS) <= set(n.picks), f"narrow: a moved pair must keep the spill ratchet: {sorted(n.picks)}")
        n = narrow(["rust-toolchain.toml"], side, side, [(base, gone)], rows)
        expect(any("ptx-scan gx: moved beta (1 removed)" in r for r in n.full), f"narrow: a removed entry does not keep the full list: {n.full}")
        expect(set(NARROW_ALWAYS) <= set(n.picks), f"narrow: a removed entry must keep the spill ratchet: {sorted(n.picks)}")
        # narrow(): an identical no-jit pair narrows as an identical full pair does and says the JIT was skipped; a
        # no-jit pair that is not identical is refused by name, never narrowed and never the silent full list
        n = tried(lambda: narrow(["rust-toolchain.toml"], side, side, [(nj, nj_same)], rows))
        expect(isinstance(n, Narrowed) and not n.full and set(n.picks) == {"gate-sampler", *NARROW_ALWAYS} and "JIT skipped" in n.verdicts[0].line(),
               f"narrow: an identical no-jit pair does not narrow as an identical full pair: {n}")
        n = tried(lambda: narrow(["rust-toolchain.toml"], side, side, [(base, nj_same)], rows))
        expect(isinstance(n, Narrowed) and not n.full and set(n.picks) == {"gate-sampler", *NARROW_ALWAYS}, f"narrow: an identical mixed pair does not narrow: {n}")
        nj_add = read_scan(scan(tmp, "nj_add.log", {"alpha": h1, "beta": h2, "gamma": h3}, nojit=True))
        n = tried(lambda: narrow(["rust-toolchain.toml"], side, side, [(nj, nj_add)], rows))
        expect(isinstance(n, str) and refusal in n, f"narrow: an added entry in a no-jit pair narrows or falls back instead of being refused by name: {n}")
        # a change of the ratchet's own pin file keeps it even with every pair identical: the tsv is
        # the recipe's own script's read, so its own-target pick names it and nothing else moves
        n = narrow(["tools/ref/ptx-shapes.tsv"], side, side, [(base, same)], rows)
        expect(not n.full and set(n.picks) == set(NARROW_ALWAYS),
               f"narrow: a ptx-shapes.tsv change keeps the spill ratchet alone: full={n.full} picks={sorted(n.picks)}")
        n = narrow(["rust-toolchain.toml"], side, side, [], rows)
        expect(any("no --scan pair" in r for r in n.full), f"narrow: no scan pair does not keep the full list: {n.full}")
        n = narrow(["Cargo.lock", "rust-toolchain.toml"], side, side, [(base, same)], rows)
        expect(len(n.full) == 1 and n.full[0].startswith("Cargo.lock:") and "no row of" in n.full[0],
               f"narrow: an unmatched changed file does not keep the full list, naming it: {n.full}")
        n = narrow(["rust-toolchain.toml"], side, side, [(base, same)], [row("rust-toolchain.toml", ["*"], 7)])
        expect(any(f"{GATE_PATHS}:7" in r for r in n.full), f"narrow: a `*` row does not keep the full list: {n.full}")
        n = narrow(["docs/plan.md"], side, side, [(base, same)], rows)
        expect(not n.full and not n.picks and "not a host path" in n.files[0],
               f"narrow: docs/plan.md (read by no gate), an equal-scan host change, drops the spill ratchet and picks nothing: "
               f"full={n.full} picks={sorted(n.picks)}")
        n = narrow(["crates/vision/src/lib.rs"], side, side, [(base, same)], rows)
        expect(not n.full and {"gate-vision", "gate-gpu-vision"} <= set(n.picks) and "lib bloomery-gpu-vision not scanned" in n.files[0],
               f"narrow: a kernel carrier no pair covers does not keep its gates: {n.files} {sorted(n.picks)}")
        n = narrow(["crates/gpu-gates/src/bin/gate_p1.rs"], side, side, [(base, same)], rows)
        expect(not n.full and set(n.picks) == {"gate-gpu-p1"}, f"narrow: a bin's own file, an equal-scan change the ratchet does not read: {n.full} {sorted(n.picks)}")
        # a bin carrier is covered by the scan of that bin alone. Every bin of gpu-gates carries its own kernels in a
        # bundle named for the package, so a scan of another gpu-gates bin (same bundle name) or of that bin without
        # the package's bundle reads nothing of them. A change under crates/gpu reaches the closure of both bin
        # carriers (gate_swap, gate_kquant) and of the three device libs.
        libs = ("bloomery-gpu", "bloomery-gpu-deepseek41", "bloomery-gpu-vision")
        own = ("bloomery-gpu", "bloomery-gpu-gates")
        gpu_rows = [row("crates/gpu/src/**", ["gate-gpu-lib"])]

        def pair(name: str, bin: str, bundles) -> tuple[ScanLog, ScanLog]:
            return (read_scan(scan(tmp, f"{name}-b.log", {"alpha": h1}, bundles=bundles, bin=bin, nojit=True)),
                    read_scan(scan(tmp, f"{name}-h.log", {"alpha": h1}, bundles=bundles, bin=bin, nojit=True)))

        lib_pair = pair("libs", "gx", libs)
        swap, kquant = pair("swap", "gate_swap", own), pair("kquant", "gate_kquant", own)

        def picked(*pairs) -> tuple[set[str], str]:
            got = narrow(["crates/gpu/src/model.rs"], side, side, list(pairs), gpu_rows)
            return set(got.picks), got.files[0]

        got, why = picked(lib_pair)
        expect({"gate-gpu-swap", "gate-gpu-kquant"} <= got and "bin gate_swap not scanned" in why and "bin gate_kquant not scanned" in why,
               f"narrow: the bin carriers with no scan of their own do not keep their gates: {sorted(got)} {why}")
        got, why = picked(lib_pair, swap)
        expect("gate-gpu-swap" not in got and "gate-gpu-kquant" in got and "bin gate_swap" not in why and "bin gate_kquant not scanned" in why,
               f"narrow: a scan of gate_swap does not cover gate_swap alone: {sorted(got)} {why}")
        got, why = picked(lib_pair, kquant)
        expect("gate-gpu-kquant" not in got and "gate-gpu-swap" in got and "bin gate_kquant" not in why and "bin gate_swap not scanned" in why,
               f"narrow: a scan of gate_kquant does not cover gate_kquant alone: {sorted(got)} {why}")
        got, why = picked(lib_pair, swap, kquant)
        expect(not {"gate-gpu-swap", "gate-gpu-kquant"} & got and "not scanned" not in why, f"narrow: both bins scanned still keep a bin carrier's gates: {sorted(got)} {why}")
        got, why = picked(lib_pair, pair("swap-nobundle", "gate_swap", ("bloomery-gpu",)))
        expect("gate-gpu-swap" in got and "bin gate_swap not scanned" in why,
               f"narrow: a scan of gate_swap that names no package bundle covers its kernels: {sorted(got)} {why}")
        # a lib carrier is covered by any scan that names its bundle, and by no other
        got, why = picked(pair("nolibs", "gx", ("bloomery-gpu",)), swap, kquant)
        expect({"gate-gpu-vision", "gate-ds41-bind"} <= got and "lib bloomery-gpu-vision not scanned" in why and "lib bloomery-gpu-deepseek41 not scanned" in why,
               f"narrow: the device libs a scan names no bundle of do not keep their gates: {sorted(got)[:6]} {why}")
        # a trigger names its weekly recipe and maps nothing: a file only a trigger row matches, which every gate reads
        # through a dependency, keeps the full list; beside a gate row it narrows as that row says
        trig = [row("rust-toolchain.toml", ["weekly-gpu-ds41-serve"], 9)]
        n = narrow(["rust-toolchain.toml"], side, side, [(base, same)], trig)
        expect(len(n.full) == 1 and n.full[0].startswith("rust-toolchain.toml:") and "no row of" in n.full[0],
               f"narrow: a file only a trigger row matches does not keep the full list: {n.full}")
        n = narrow(["rust-toolchain.toml"], side, side, [(base, same)], rows + trig)
        expect(not n.full and set(n.picks) == {"gate-sampler", *NARROW_ALWAYS}, f"narrow: a trigger row beside a gate row: {n.full} {sorted(n.picks)}")
        got = weekly_triggers(["rust-toolchain.toml", "crates/vision/src/lib.rs"], side, side, rows + trig)
        expect([(t.recipe, t.files, t.why) for t in got] == [("weekly-gpu-ds41-serve", ["rust-toolchain.toml"], f"trigger {GATE_PATHS}:9 rust-toolchain.toml")],
               f"weekly: a changed file a trigger matches names its recipe, and only it: {[(t.recipe, t.files, t.why) for t in got]}")
        expect(not weekly_triggers(["crates/vision/src/lib.rs", "justfile"], side, side, rows + trig), "weekly: a change no trigger matches names a weekly recipe")
        import copy

        moved = copy.copy(side)
        moved.recipes = dict(side.recipes)
        moved.recipes["weekly-gpu-ds41-serve"] = copy.copy(side.recipes["weekly-gpu-ds41-serve"])
        moved.recipes["weekly-gpu-ds41-serve"].text += " moved"
        got = weekly_triggers(["justfile"], side, moved, rows)
        expect([(t.recipe, t.why) for t in got] == [("weekly-gpu-ds41-serve", "recipe text")], f"weekly: its own recipe text changed: {[(t.recipe, t.why) for t in got]}")
        expect(refused(lambda: weekly_triggers(["rust-toolchain.toml"], side, side, [row("rust-toolchain.toml", ["weekly-nope"], 4)]), "weekly-nope is not a weekly-* recipe"),
               "weekly: a trigger naming no recipe not refused by name")
        # the triggers are read from the tree the change is read at (a range's B), never the working tree's table
        from types import SimpleNamespace

        old = os.path.join(tmp, "old-tree")
        os.makedirs(os.path.join(old, "tools"))
        got = trigger_rows(SimpleNamespace(tree=SimpleNamespace(root=old)))
        expect(got[0] == [] and got[1] is not None and "no weekly trigger" in got[1], f"weekly: a tree with no table reads no trigger: {got}")
        with open(os.path.join(old, GATE_PATHS), "w", encoding="utf-8") as fh:
            fh.write("crates/x/**\tgate-x\tan old row\n")
        got = trigger_rows(SimpleNamespace(tree=SimpleNamespace(root=old)))
        expect([r.glob for r in got[0]] == ["crates/x/**"] and got[1] is None, f"weekly: the rows of the tree read, not the working tree's: {got}")
    # the table's checks: planted rows, each refused by name; the real table clean
    tree, recipes = side.tree, side.recipes
    got = gate_paths_problems(tree, recipes, [row("crates/nothing/**", ["gate-nope"])])
    expect(any("gate-nope is not a gate-* recipe" in p for p in got) and any("matches no file" in p for p in got),
           f"gate-paths: an unknown recipe or a glob matching nothing not refused: {got}")
    got = [p for p in gate_paths_problems(tree, recipes, [row("crates/serve/src/**", ["gate-ds41-bind", "gate-gpu-p1"])]) if " does not read " in p]
    expect(len(got) == 1 and "gate-gpu-p1 does not read crates/serve/src/" in got[0], f"gate-paths: a recipe that does not link the row's files: {got}")
    # the weekly tier's triggers: a name the justfile lacks, and a trigger whose recipe reads none of the glob's files,
    # each refused by name; a trigger-only row is held to no gate's module (it maps nothing for --narrow)
    got = gate_paths_problems(tree, recipes, [row("crates/serve/src/**", ["weekly-nope"])])
    expect(len(got) == 1 and "weekly-nope is not a weekly-* recipe" in got[0], f"gate-paths: an unknown weekly trigger not refused: {got}")
    # engram-lab is a lab crate: no engine binary links it, so no gate's build reads its files
    got = gate_paths_problems(tree, recipes, [row("crates/engram-lab/src/**", ["weekly-gpu-ds41-serve"])])
    expect(len(got) == 1 and "weekly-gpu-ds41-serve reads none of the" in got[0], f"gate-paths: a trigger off its gate not refused: {got}")
    got = gate_paths_problems(tree, recipes, [row("crates/gpu-deepseek41/src/{body.rs,body/**}", ["weekly-gpu-ds41-flowcounts"])])
    expect(not got, f"gate-paths: a trigger-only row held to the gates that name its module: {got[:2]}")
    got = weekly_problems(recipes, [row("crates/serve/**", ["weekly-gpu-ds41-serve"])])
    expect(len(got) == len([n for n in recipes if n.startswith(WEEKLY_PREFIX)]) - 1 and all("no row of" in p for p in got)
           and not any("weekly-gpu-ds41-serve:" in p for p in got), f"weekly: a weekly recipe with no trigger row not refused: {got}")
    expect(not weekly_problems(recipes), f"weekly: the real table: {weekly_problems(recipes)}")
    got = gate_paths_problems(tree, recipes, [row("crates/gpu-deepseek41/src/{body.rs,body/**}", ["gate-gpu-ds41-step"])])
    expect(any("gate-gpu-ds41-long runs bin gate_deepseek41_long" in p and "names bloomery_gpu_deepseek41::body" in p for p in got),
           f"gate-paths: a gate whose bin names the module, left out of its row: {got[:2]}")
    got = [p for p in gate_paths_problems(tree, recipes, [row("crates/refset/src/arch/mod.rs", ["gate-refset"])]) if " names " in p and p.endswith("to the row")]
    expect([p.split(": ")[1].split(" ")[0] for p in got] == ["gate-gpu-linear"],
           f"gate-paths: arch/mod.rs's row asks for exactly the gates that call its own items (gate_linear's node_dumps), not its submodules' users: {got}")
    expect(names_module(named_paths("use a::b::{self, C};"), ("a", "b")) and not names_module(named_paths("// a::b::c\nlet s = \"a::b\";"), ("a", "b")),
           "gate-paths: a module named in a use tree, or only in a comment or a string")
    expect(glob_regex("x/{a.rs,b/**}").match("x/b/c/d.rs") and not glob_regex("x/*.rs").match("x/b/c.rs"), "gate-paths: glob `**`, `*` and braces")
    stats: dict[str, int] = {}
    got = gate_paths_problems(tree, recipes, stats=stats)
    expect(not got, f"gate-paths: the real table: {got[:3]}")
    reach_self_test(expect, side, row, stats)
    # the host group: the real host recipes pass; gate-gpu-lib, whose hw_ tests use the card, is refused by name
    expect(not host_problems(tree, recipes), f"host: the real justfile: {host_problems(tree, recipes)[:2]}")
    import copy

    planted = dict(recipes)
    planted["gate-gpu-lib"] = copy.copy(recipes["gate-gpu-lib"])
    planted["gate-gpu-lib"].groups = (HOST_GROUP,)
    got = host_problems(tree, planted)
    expect(got and all("gate-gpu-lib" in p for p in got) and any("names 'CudaContext'" in p or "names 'DeviceBuffer'" in p for p in got),
           f"host: gate-gpu-lib tagged host not refused: {got[:2]}")
    planted["gate-gpu-ds41-long"] = copy.copy(recipes["gate-gpu-ds41-long"])
    planted["gate-gpu-ds41-long"].groups = (HOST_GROUP, "solo")
    got = host_problems(tree, planted)
    expect(any("gate-gpu-ds41-long" in p and "[group('solo')]" in p for p in got) and any("gate-gpu-ds41-long" in p and "runs a card gate" in p for p in got),
           f"host: a host recipe that is solo and runs a card gate not refused: {got}")


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
    rc = recipe_commands(Recipe("x", ["BLOOMERY_BOX_READONLY=1 ./tools/box.sh 'bash -s -- {{ARGS}}' < tools/box-gc.sh"], [], ""))
    expect(rc.box and rc.scripts == ["tools/box-gc.sh"], f"a script over stdin: scripts {rc.scripts}")

    # box-command: the recipe's one box.sh argument verbatim, and a named refusal of every other shape
    def refused(lines: list[str], why: str = "") -> bool:
        try:
            box_command(Recipe("x", lines, [], ""))
        except BoxCommandError as err:
            return why in str(err)
        return False

    expect(box_command(Recipe("x", ["./tools/box.sh 'cargo check --workspace && echo \"$X\"'", "./tools/lock-back.sh"], [], "")) == 'cargo check --workspace && echo "$X"', "box-command: the argument is not printed verbatim")
    expect(refused(["./tools/box.sh 'cargo check'", "./tools/box.sh 'cargo clippy'"]), "box-command: two box.sh lines accepted")
    expect(refused(["./tools/lock-back.sh"]), "box-command: a recipe with no box.sh line accepted")
    expect(refused(["BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'cargo check'"], "prefix (BLOOMERY_MODEL=deepseek41)"), "box-command: an env prefix accepted or not named")
    expect(refused(["./tools/box.sh 'cargo test {{ARGS}}'"]), "box-command: a {{…}} parameter accepted")
    expect(refused(["./tools/box.sh 'cargo check' && echo done"]), "box-command: a line that goes on after the argument accepted")
    # a line continued with a backslash is one line to just, and one to the parser: the box.sh argument
    # joined as just runs it (backslash and the next line's indentation dropped, inside the quotes too)
    with tempfile.TemporaryDirectory(prefix="recipes-cont-") as tmp:
        p = os.path.join(tmp, "justfile")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("cont:\n    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates \\\n"
                     "        --features gpu --release --bin gate_a && bash tools/gpu-gate.sh \\\n        gate_a'\n"
                     "mac:\n    python3 tools/ref/tdist.py \\\n        4\n")
        try:
            jc = load_justfile(p)
            got = box_command(jc["cont"])
            rc = recipe_commands(jc["cont"])
            mac = recipe_commands(jc["mac"])
        except RecipeError as err:
            got, rc, mac = f"refused: {err}", None, None
        expect(got == "cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_a && bash tools/gpu-gate.sh gate_a",
               f"continued box.sh line: box-command gives {got!r}")
        expect(rc is not None and rc.runs == ["gate_a"] and [(i.sub, i.selectors) for i in rc.invocations] == [("build", [("bin", "gate_a")])],
               f"continued box.sh line: parsed as {rc and (rc.runs, [(i.sub, i.selectors) for i in rc.invocations])}")
        expect(mac is not None and mac.scripts == ["tools/ref/tdist.py"], f"continued Mac line: scripts {mac and mac.scripts}")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("dangling:\n    echo a \\\n")
        try:
            load_justfile(p)
            fails.append("a recipe whose last line ends in a backslash accepted")
        except RecipeError as err:
            expect("continuation of nothing" in str(err), f"dangling backslash refused without its name: {err}")
    # the recipes tools/mac-check.sh derives from: each gives one plain `cargo <subcommand> …` (mac-check
    # adds --target to it and refuses any other shape; its own self-test holds that side)
    jf = load_justfile(os.path.join(ROOT, "justfile"))
    for n, verb in (("check", "check"), ("lint", "clippy"), ("fmt-check", "fmt")):
        try:
            got = box_command(jf[n]) if n in jf else ""
        except BoxCommandError as err:
            got = ""
            fails.append(f"box-command {n}: {err}")
        expect(got.split()[:2] == ["cargo", verb], f"box-command {n}: not `cargo {verb} …`: {got!r}")

    # the plans records-refresh writes: derived from its markers and loops, each missing one named with the
    # recipe to run, and a --counts reader's depth that no marker writes named too
    with tempfile.TemporaryDirectory(prefix="recipes-plans-") as tmp:
        refresh = ("records-refresh:\n    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates "
                   "--features deepseek41 --release --bin generate_ds41 >&2 && for P in 8 16; do for c in on off; do echo \"#> "
                   "tools/flow/plans/ds41-p$P-ced-$c.rec generate_ds41 --plan --depth $P --place a under BLOOMERY_CED=$c at the "
                   "plan room\" && bash tools/ref/plan-room.sh $P $c; done; done' | python3 tools/bloomery/records.py refresh\n")
        reader = ("gate-x:\n    ./tools/box.sh 'D=t && BLOOMERY_STEP_STATS=1 bash tools/gpu-gate.sh generate_ds41 --place gate "
                  "--depth {d} -n 2 > $D/a.log && python3 tools/flow/ds41_prefill.py --counts $D/a.log > $D/a.counts'\n")
        jp = os.path.join(tmp, "justfile")

        def plan_probs(text: str) -> list[str]:
            with open(jp, "w", encoding="utf-8") as fh:
                fh.write(text)
            try:
                return plan_problems(tmp, load_justfile(jp))
            except RecipeError as err:
                return [f"raised: {err}"]

        names = [f"tools/flow/plans/ds41-p{p}-ced-{c}.rec" for p in (8, 16) for c in ("on", "off")]
        got = plan_probs(refresh + reader.format(d=16))
        expect(len(got) == 4 and all(any(n in g and "run just records-refresh" in g for g in got) for n in names),
               f"plans: four missing plans not each named with `run just records-refresh`: {got}")
        os.makedirs(os.path.join(tmp, "tools/flow/plans"))
        for n in names:
            open(os.path.join(tmp, n), "w").close()
        got = plan_probs(refresh + reader.format(d=16))
        expect(got == [], f"plans: present plans (empty files: presence only) still refused: {got}")
        got = plan_probs(refresh + reader.format(d=32))
        expect(len(got) == 1 and "gate-x reads the engine's plan of P = 32" in got[0] and "add 32 to its P list" in got[0],
               f"plans: a --counts depth no marker writes not named: {got}")
        arms = ("gate-arms:\n    ./tools/box.sh 'D=t && BLOOMERY_STEP_STATS=1 bash tools/gpu-gate.sh generate_ds41 "
                "--place gate --arm {a} > $D/b.log && python3 tools/flow/ds41_prefill.py --counts $D/b.log > $D/b.counts'\n")
        got = plan_probs(refresh + arms.format(a="16 --arm prose:8/2 --arm code:8 --arm 16/2"))
        expect(got == [], f"plans: every --arm form's depth written (D, D/N, prose:P/N, code:P): {got}")
        for c in ("on", "off"):
            open(os.path.join(tmp, "tools/flow/plans", f"ds41-p512-ced-{c}.rec"), "w").close()
        got = plan_probs(refresh.replace("for P in 8 16", "for P in 512") + arms.format(a="512/2 --arm 1536/2"))
        expect(len(got) == 1 and "gate-arms reads the engine's plan of P = 1536" in got[0] and "add 1536" in got[0],
               f"plans: a --counts --arm depth no marker writes not named: {got}")
        got = plan_probs(refresh + arms.format(a="512/2 --arm prose:x"))
        expect(len(got) == 1 and "gate-arms" in got[0] and "--arm it cannot read" in got[0] and "prose:x" in got[0],
               f"plans: an --arm this parser cannot read not refused by name: {got}")
        got = plan_probs(refresh.replace("-ced-$c.rec", "-ced-$c-$Q.rec"))
        expect(len(got) == 1 and "no loop around it sets" in got[0], f"plans: a marker variable no loop sets not refused: {got}")
        got = plan_probs(refresh.replace("for P in 8 16", "for P in 8 8"))
        expect(len(got) == 1 and "two markers write" in got[0], f"plans: a path written twice not refused: {got}")
        # the plan room: a bare generate_ds41 --plan (the box's live MemAvailable) refused by name, once; the script
        # with the marker's depth and CED swapped refused; the script as the marker names it passes (got == [] above)
        script = "bash tools/ref/plan-room.sh $P $c"
        got = plan_probs(refresh.replace(script, "BLOOMERY_CED=$c target/release/generate_ds41 --plan --depth $P --place a"))
        expect(len(got) == 1 and "bare `target/release/generate_ds41 --plan --depth $P --place a`" in got[0]
               and "MemAvailable" in got[0] and f"bash {PLAN_ROOM} $P $c" in got[0],
               f"plans: a bare generate_ds41 --plan in records-refresh not refused by name, once: {got}")
        got = plan_probs(refresh.replace(script, "bash tools/ref/plan-room.sh $c $P"))
        expect(len(got) == 1 and "followed by `bash tools/ref/plan-room.sh $c $P`" in got[0]
               and f"not `bash {PLAN_ROOM} $P $c`" in got[0],
               f"plans: the plan-room script with the marker's depth and CED swapped not refused: {got}")
        got = plan_probs(refresh.replace(f" && {script}", ""))
        expect(len(got) == 1 and "followed by `nothing`" in got[0], f"plans: a marker with no plan after it not refused: {got}")
    real = refresh_plans(jf["records-refresh"]) if "records-refresh" in jf else []
    got = plan_room_problems(jf["records-refresh"]) if "records-refresh" in jf else ["no records-refresh recipe"]
    expect(got == [], f"plans: records-refresh on the real justfile does not print every plan at the plan room: {got}")
    depths = {p.depth for p in real}
    expect({512, 1536, 16384} <= depths, f"plans: records-refresh on the real justfile writes depths {sorted(depths)}")
    expect(all(d in depths for n in jf for d in counts_depths(jf[n])), "plans: a real --counts reader's depth that records-refresh does not write")

    # build shapes: one command a (package, mode, enabled features), the targets of every call that shares
    # it; a parameter, a doctest and the check recipe left out by name
    with tempfile.TemporaryDirectory(prefix="recipes-combos-") as tmp:
        p = os.path.join(tmp, "justfile")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(
                "check:\n    ./tools/box.sh 'cargo check --workspace --all-targets --features gpu'\n"
                "a:\n    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu,deepseek41 --release --bin generate_ds41'\n"
                "b:\n    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_deepseek41_attn'\n"
                "c:\n    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_swap'\n"
                "d:\n    ./tools/box.sh 'bash tools/gate.sh -p bloomery-gpu-gates --release --lib -- --include-ignored'\n"
                "e:\n    ./tools/box.sh 'bash tools/gate.sh -p bloomery-model --release --doc'\n"
                "f FEATURES:\n    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features {{FEATURES}} --release --bin gate_p1 && cargo build --release -p bloomery-gpu-gates --bin oxart_ptx'\n"
            )
        try:
            shapes, skips, n = build_shapes(Tree(ROOT), load_justfile(p))
        except RecipeError as err:
            shapes, skips, n = [], [], 0
            fails.append(f"combos: the synthetic justfile raised {err}")
        cmds = {c.command(): c.recipes for c in shapes}
        want = {
            "cargo check -p bloomery-gpu-gates --features deepseek41,gpu --bin gate_deepseek41_attn --bin generate_ds41": ["a", "b"],
            "cargo check -p bloomery-gpu-gates --features gpu --bin gate_swap": ["c"],
            "cargo check -p bloomery-gpu-gates --profile test --lib": ["d"],
            "cargo check -p bloomery-model --lib": ["e"],
            "cargo check -p bloomery-gpu-gates --bin oxart_ptx": ["f"],
        }
        expect(cmds == want, f"combos: the synthetic shapes are {cmds}")
        expect(n == 6, f"combos: {n} distinct (package, target, mode, features) tuples, not 6")
        why = {r: w for r, w in skips}
        expect(sorted(why) == ["check", "e", "f"], f"combos: left out by name {sorted(why)}, not check, e, f")
        expect("tools/mac-check.sh check" in why.get("check", ""), f"combos: the check recipe's reason {why.get('check')}")
        expect("doctests of bloomery-model" in why.get("e", ""), f"combos: the doctest's reason {why.get('e')}")
        expect("{{FEATURES}}" in why.get("f", "") and "recipe parameter" in why.get("f", ""), f"combos: the parameter's reason {why.get('f')}")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("g:\n    ./tools/box.sh 'cargo build -p bloomery-gpu-gates --bin no_such_bin'\n")
        try:
            build_shapes(Tree(ROOT), load_justfile(p))
            fails.append("combos: a call naming no target accepted")
        except RecipeError as err:
            expect("g: --bin no_such_bin" in str(err), f"combos: an unresolvable call refused without its name: {err}")

    # the shape input key: content-only (the memo's hash, not any mtime), stable for the same inputs,
    # moved by a command or an input's content; the ledger reader takes only green 64-key lines
    c1 = Combo("p", False, ("a",), (), ["a"], False, False, selectors={("lib", None)})
    c2 = Combo("p", False, ("a",), (), ["a"], False, False, selectors={("lib", None)})
    c3 = Combo("p", True, ("a",), (), ["a"], False, False, selectors={("lib", None)})
    memo = {"Cargo.toml": "0" * 64, "p/src/lib.rs": "1" * 64}
    expect(shape_key(c1, {"Cargo.toml", "p/src/lib.rs"}, memo) == shape_key(c2, {"p/src/lib.rs", "Cargo.toml"}, dict(memo)),
           "shape key: the same inputs in another order gave another key")
    expect(shape_key(c1, {"Cargo.toml", "p/src/lib.rs"}, memo) != shape_key(c3, {"Cargo.toml", "p/src/lib.rs"}, dict(memo)),
           "shape key: another command (a test profile) gave the same key")
    expect(shape_key(c1, {"Cargo.toml", "p/src/lib.rs"}, memo) != shape_key(c1, {"Cargo.toml", "p/src/lib.rs"}, {**memo, "Cargo.toml": "2" * 64}),
           "shape key: a changed input content gave the same key")
    with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as fh:
        k = "a" * 64
        fh.write(f"{k}\tgreen\t2026-10-07 16:00\tbloomery-x\nnope\tgreen\t2026-10-07 16:00\tt\n{k[:-1]}\tgreen\td\tt\n{k}\tred\td\tt\n")
        lp = fh.name
    green, bad = read_shape_ledger(lp)
    os.unlink(lp)
    expect(list(green) == [k] and green[k] == ("2026-10-07 16:00", "bloomery-x"), f"shape ledger: the green lines read {green}")
    expect(bad == 3, f"shape ledger: {bad} malformed lines counted, not 3")
    expect(read_shape_ledger(lp) == ({}, 0), "shape ledger: a missing file is not empty")
    try:
        cmd_combos(argparse.Namespace(justfile=None, base=None, ledger="x"))
        fails.append("combos: --ledger without --base accepted")
    except RecipeError as err:
        expect("--base" in str(err), f"combos: --ledger without --base refused without naming --base: {err}")

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
    # the two shapes whose compile errors the workspace check could not see (2026-09-29): gate_swap with
    # `gpu` alone, and the lib's tests with no feature
    shapes, _, _ = build_shapes(tree, recipes)
    by = {(c.package, c.test, c.enabled): c for c in shapes}
    gs = by.get(("bloomery-gpu-gates", False, ("gpu",)))
    expect(gs is not None and ("bin", "gate_swap") in gs.selectors and "gate-gpu-swap" in gs.recipes, "combos: no gpu-gates build shape of gpu alone with gate-gpu-swap's gate_swap")
    gl = by.get(("bloomery-gpu-gates", True, ()))
    expect(gl is not None and ("lib", None) in gl.selectors and "gate-gpu-gates-lib" in gl.recipes, "combos: no gpu-gates test shape with no feature for gate-gpu-gates-lib")

    def sel(f: str) -> set[str]:
        return {n for n in gates if graph.inputs(n).match(f)}

    one = sel("crates/gpu-gates/src/bin/gate_p1.rs")
    expect(one == {"gate-gpu-p1"}, f"gate_p1.rs selects {sorted(one)}")
    shared = sel("crates/gpu-gates/src/bin/shared/ds41_finite.rs")
    expect({"gate-gpu-ds41-step", "gate-gpu-ds41-long"} <= shared and "gate-gpu-e2e" not in shared, f"shared/ds41_finite.rs selects {sorted(shared)}")
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
    # a whole-string .rs literal of a workspace package is documentation, not a run-time read: levers'
    # registry names the in-place lever files, and only the targets that compile such a file select it
    dspark = sel("crates/gpu-gates/src/bin/shared/ds41_dspark.rs")
    expect(dspark == {"gate-gpu-clef-serve", "gate-gpu-ds41-chat", "gate-gpu-ds41-draft", "gate-gpu-ds41-dspark-loop",
                      "gate-gpu-ds41-prefill", "gate-gpu-ds41-tier", "gate-gpu-ds41-twocard", "gate-gpu-glm5next-serve",
                      "gate-gpu-lev-serve", "gate-gpu-qwen3-serve", "gate-ptx-spill"},
           f"shared/ds41_dspark.rs (a #[path] module of the bins that name it) selects {sorted(dspark)}")
    hybrid = sel("crates/gpu/src/hybrid.rs")
    expect("gate-gpu-e2e" in hybrid and "gate-ops" not in hybrid and len(hybrid) < len(gates) - 20,
           f"a gpu lib file levers' registry names selects the gpu-linking gates only: {len(hybrid)} of {len(gates)}")
    qprof = sel("tools/ref/models/qwen3moe.sh")
    expect("gate-gpu-qwen3moe-e2e" in qprof and "gate-gpu-e2e" not in qprof, f"qwen3moe profile selects {sorted(qprof)[:5]}")
    # oracle.sh builds its Korean corpus with `find "$ROOT/docs" -name '*.md'`, but only `just dump-ref-tokenizer`
    # runs it: no gate recipe names it, so a docs change selects no gate
    docs = sel("docs/plan.md")
    expect(docs == set(), f"docs/plan.md selects {sorted(docs)}")
    expect(not graph.inputs("gate-tokenizer").prefixes.get("docs/"), f"gate-tokenizer's docs/ prefix: {graph.inputs('gate-tokenizer').prefixes}")
    walks = [(ln, _SCRIPT_WALK.findall(ln)) for ln in ('find "$ROOT/docs" -name x', 'find "$ROOT"/docs/cards -type f', "find crates -name '*.rs'", 'find "$REF/docs" -name x', "find . -name x", 'find "$ROOT" -name x')]
    expect([w for _, w in walks] == [["docs"], ["docs/cards"], [], [], [], []], f"find roots: {walks}")
    cases = sel("crates/tokenizer/tests/cases.txt")
    expect(cases == {"gate-tokenizer"}, f"tokenizer cases.txt (include_bytes! of tests/tokenizer.rs) selects {sorted(cases)}")
    expect(len(sel("tools/box.sh")) == len(gates), "tools/box.sh does not select every gate")
    # a #[cfg(test)] module's includes belong to the recipes that run that lib's tests: levers' tests.rs
    # (a file module) reads levers-direct.txt, gpu-gates' record.rs (an inline one) the record schemas;
    # check and lint compile both (--all-targets)
    lev = sel("tools/levers-direct.txt")
    expect(lev == {"gate-levers"}, f"tools/levers-direct.txt selects {sorted(lev)}")
    # weekly-gpu-ds41-flowcounts reads it through a script: ds41_prefill.py --counts parses the engine's log with
    # records.py, which loads the schema (the scan reaches records.py by routes.py's chain of tools)
    weeklies = [n for n in recipes if n.startswith(WEEKLY_PREFIX)]
    schema = sel("tools/bloomery/schema/generate_ds41.jsonl") | {n for n in weeklies if graph.inputs(n).match("tools/bloomery/schema/generate_ds41.jsonl")}
    expect(schema == {"gate-gpu-gates-lib", "gate-ds41-oracle", "gate-ds41-kld", "gate-ds41-bind", "gate-ds41-place", "weekly-gpu-ds41-flowcounts"}, f"a record schema selects {sorted(schema)}")
    for n in ("check", "lint"):
        expect(graph.inputs(n).match("tools/levers-direct.txt") is not None, f"{n} (--all-targets) does not read tools/levers-direct.txt")

    # the walk on a synthetic tree: a file module and an inline one under #[cfg(test)], cfg(all(test, …)),
    # cfg(any(test, …)) (not test-only), a file reached both ways, and a dependent's view of the lib
    with tempfile.TemporaryDirectory(prefix="recipes-cfgtest-") as tmp:
        def put(rel: str, text: str = "") -> None:
            p = os.path.join(tmp, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(text)

        put("p/Cargo.toml")
        put("q/Cargo.toml")
        put(
            "p/src/lib.rs",
            '#[cfg(all(feature = "x", test))]\n#[path = "a.rs"]\nmod a_again;\nmod a;\n#[cfg(test)]\nmod tests;\n'
            '#[cfg(all(test, feature = "x"))]\nmod c;\n#[cfg(any(test, feature = "x"))]\nmod b;\n#[cfg(test)]\n'
            '#[allow(dead_code)]\nmod inline {\n    const I: &str = include_str!("../data/inline.txt");\n}\n'
            'const N: &str = include_str!("../data/normal.txt");\n',
        )
        put("p/src/tests.rs", 'mod deep;\nconst T: &str = include_str!("../data/test.txt");\n')
        put("p/src/tests/deep.rs", 'const D: &str = "../tools/deep.txt";\nconst M: &str = "../q/src/main.rs";\n')
        for f in ("p/src/a.rs", "p/src/b.rs", "p/src/c.rs", "p/data/normal.txt", "p/data/test.txt", "p/data/inline.txt", "tools/deep.txt", "q/src/main.rs"):
            put(f)

        def pkg(name: str, kind: str, src: str, deps: list[str]) -> dict:
            return {
                "id": name,
                "name": name,
                "manifest_path": os.path.join(tmp, name, "Cargo.toml"),
                "features": {},
                "dependencies": [{"name": d, "kind": None, "optional": False, "features": [], "uses_default_features": True} for d in deps],
                "targets": [{"kind": [kind], "name": name, "src_path": os.path.join(tmp, name, src)}],
            }

        meta = {"workspace_root": tmp, "workspace_members": ["p", "q"], "packages": [pkg("p", "lib", "src/lib.rs", []), pkg("q", "bin", "src/main.rs", ["p"])]}
        st = Tree(tmp, meta)
        only_test = {"p/src/tests.rs", "p/src/tests/deep.rs", "p/src/c.rs", "p/data/test.txt", "p/data/inline.txt", "tools/deep.txt"}
        built = {"p/src/lib.rs", "p/src/a.rs", "p/src/b.rs", "p/data/normal.txt"}
        lib, libtest, dep = (set(st.target_files(*t, [])) for t in (("p", "lib", None), ("p", "libtest", None), ("q", "bin", "q")))
        expect(built <= lib and not lib & only_test, f"cfg(test) walk: the lib build reads {sorted(lib & only_test)}, misses {sorted(built - lib)}")
        expect(built | only_test <= libtest, f"cfg(test) walk: the lib tests miss {sorted((built | only_test) - libtest)}")
        expect(built <= dep and not dep & only_test, f"cfg(test) walk: a dependent reads {sorted(dep & only_test)}, misses {sorted(built - dep)}")
        # a .rs literal names a workspace package's source: documentation of an owner, not a run-time
        # read — the module walk of whatever compiles it is its only route in (q's own bin keeps its
        # file, as every target keeps its own; p's views must not gain it)
        expect("q/src/main.rs" not in lib | libtest, "a .rs literal of a workspace package is an input")
        expect(not st.unresolved, f"cfg(test) walk: unresolved {st.unresolved[:2]}")

    # pure crates: the cfg reader, then a synthetic workspace with a lock file — a device root reached
    # directly, through a workspace crate, through an external package, a git source and a dev
    # dependency; x86_64 and Linux-only code unguarded (rejected) and under a cfg false on the Mac (not);
    # a dependency's #[cfg(test)] module (its dependent stays pure, it does not)
    for text, test, want in [
        ('target_os = "linux"', None, False),
        ('target_arch = "aarch64"', None, True),
        ('not(target_os = "macos")', None, False),
        ('any(target_arch = "x86_64", target_arch = "x86")', None, False),
        ("all(unix, not(windows))", None, True),
        ('feature = "avx"', None, None),
        ('all(feature = "avx", target_os = "linux")', None, False),
        ('any(feature = "avx", target_os = "linux")', None, None),
        ("test", False, False),
        ("test", None, None),
        ("debug_assertions", None, None),
        ('target_os = "linux" junk', None, None),
    ]:
        got = eval_cfg(parse_cfg(text), test)
        expect(got is want, f"cfg({text}) with test={test} is {got}, not {want}")
    with tempfile.TemporaryDirectory(prefix="recipes-pure-") as tmp:
        srcs = {
            "devc": "pub fn f() {}\n",
            "via": "pub fn f() {}\n",
            "wrap": "pub fn f() {}\n",
            "gitdep": "pub fn f() {}\n",
            "devdep": "pub fn f() {}\n",
            "x86": "pub fn f() {}\nuse std::arch::x86_64::_mm256_add_ps;\n",
            "guarded": (
                '//! Uses #[target_feature(enable = "avx2")] and std::arch::x86_64 in prose only.\n'
                "/// libc::RUSAGE_THREAD in a doc comment\n"
                '#[cfg(target_arch = "x86_64")]\n'
                '#[target_feature(enable = "avx2", enable = "fma")]\n'
                "unsafe fn avx(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {\n"
                "    use std::arch::x86_64::*;\n"
                "    if true { x } else { _mm256_setzero_ps() }\n"
                "}\n"
                "pub struct S {\n"
                '    #[cfg(target_os = "linux")]\n'
                "    set: libc::cpu_set_t,\n"
                "    pub n: u32,\n"
                "}\n"
                "pub fn g(m: &memmap2::Mmap, populate: bool) -> &'static str {\n"
                '    #[cfg(target_os = "linux")]\n'
                "    if populate {\n"
                '        m.advise(memmap2::Advice::PopulateRead).unwrap();\n'
                '        let _ = format!("{}{{", "/sys/x");\n'
                "    }\n"
                '    #[cfg(not(target_os = "linux"))]\n'
                "    let _ = populate;\n"
                '    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]\n'
                "    let _ = unsafe { libc::sched_setaffinity(0, 0, std::ptr::null()) };\n"
                "    let c = '{';\n"
                '    let _ = c;\n'
                '    "CUDA0#dflash_kv_input_target_features#0"\n'
                "}\n"
            ),
            "inner": '#![cfg(target_os = "linux")]\nuse std::os::linux::fs::MetadataExt;\n',
            "feat": '#[cfg(feature = "avx")]\nuse std::arch::x86_64::*;\n',
            "truecfg": '#[cfg(all(unix, target_arch = "aarch64"))]\npub fn g() -> i32 { libc::RUSAGE_THREAD }\n',
            "after": '#[cfg(target_os = "linux")]\nfn a() {}\npub fn b() { unsafe { libc::sched_setaffinity(0, 0, std::ptr::null()); } }\n',
            "field": 'pub struct S {\n    #[cfg(target_os = "linux")]\n    a: u32,\n    b: libc::cpu_set_t,\n}\n',
            "procp": 'pub fn f() -> String { std::fs::read_to_string("/proc/self/io").unwrap() }\n',
            "lastfield": 'pub struct S {\n    #[cfg(target_os = "linux")]\n    a: u32\n}\npub fn b() -> i32 { libc::RUSAGE_THREAD }\n',
            "tmod": "pub fn f() {}\n#[cfg(test)]\nmod tests {\n    fn t() { let _ = libc::RUSAGE_THREAD; }\n}\n",
            "user": "pub fn f() {}\n",
        }
        deps = {
            "devc": [("cuda-core", None)],
            "via": [("devc", None)],
            "wrap": [("wrapper", None)],
            "gitdep": [("cuda-device", None)],
            "devdep": [("devc", "dev")],
            "user": [("tmod", None)],
        }
        for name, text in srcs.items():
            os.makedirs(os.path.join(tmp, name, "src"))
            with open(os.path.join(tmp, name, "Cargo.toml"), "w", encoding="utf-8") as fh:
                fh.write("")
            with open(os.path.join(tmp, name, "src", "lib.rs"), "w", encoding="utf-8") as fh:
                fh.write(text)
        with open(os.path.join(tmp, "Cargo.lock"), "w", encoding="utf-8") as fh:
            fh.write(
                'version = 4\n\n[[package]]\nname = "cuda-core"\nversion = "0.3.1"\nsource = "registry+https://github.com/rust-lang/crates.io-index"\n'
                'dependencies = ["cuda-bindings"]\n\n[[package]]\nname = "cuda-bindings"\nversion = "0.3.1"\nsource = "registry+https://github.com/rust-lang/crates.io-index"\n\n'
                '[[package]]\nname = "wrapper"\nversion = "1.0.0"\nsource = "registry+https://github.com/rust-lang/crates.io-index"\ndependencies = ["plain 1.0.0", "cuda-bindings"]\n\n'
                '[[package]]\nname = "plain"\nversion = "1.0.0"\nsource = "registry+https://github.com/rust-lang/crates.io-index"\n\n'
                '[[package]]\nname = "cuda-device"\nversion = "0.2.1"\nsource = "git+https://github.com/NVlabs/cuda-oxide.git?rev=b98#b98"\ndependencies = ["cuda-macros"]\n\n'
                '[[package]]\nname = "cuda-macros"\nversion = "0.2.1"\nsource = "git+https://github.com/midagedev/cuda-oxide.git?rev=e58#e58"\n\n'
                '[[package]]\nname = "cuda-async"\nversion = "0.4.0"\nsource = "git+https://github.com/NVIDIA/cuda-rust.git?rev=169#169"\n'
            )
        lock = load_lock(tmp)
        expect(lock.roots == {"cuda-core", "cuda-bindings", "cuda-device", "cuda-macros", "cuda-async"}, f"pure: the lock's device roots are {sorted(lock.roots)}")
        meta = {
            "workspace_root": tmp,
            "workspace_members": list(srcs),
            "packages": [
                {
                    "id": n,
                    "name": n,
                    "manifest_path": os.path.join(tmp, n, "Cargo.toml"),
                    "features": {},
                    "dependencies": [{"name": d, "kind": k, "optional": False, "features": [], "uses_default_features": True} for d, k in deps.get(n, [])],
                    "targets": [{"kind": ["lib"], "name": n, "src_path": os.path.join(tmp, n, "src", "lib.rs")}],
                }
                for n in srcs
            ],
        }
        pt = Tree(tmp, meta)
        verdict = dict(pure_crates(pt, lock))
        for n, want in [
            ("devc", "devc -> cuda-core"),
            ("via", "via -> devc -> cuda-core"),
            ("wrap", "wrap -> wrapper -> cuda-bindings"),
            ("gitdep", "gitdep -> cuda-device"),
            ("devdep", "devdep -> devc -> cuda-core"),
            ("x86", "x86/src/lib.rs:2: x86_64 intrinsics"),
            ("feat", "feat/src/lib.rs:2: x86_64 intrinsics"),
            ("truecfg", "truecfg/src/lib.rs:2: a Linux-only libc item"),
            ("after", "after/src/lib.rs:3: a Linux-only libc item"),
            ("field", "field/src/lib.rs:4: a Linux-only libc item"),
            ("lastfield", "lastfield/src/lib.rs:5: a Linux-only libc item"),
            ("procp", "procp/src/lib.rs:1: a Linux /proc or /sys path"),
            ("tmod", "tmod/src/lib.rs:4: a Linux-only libc item"),
        ]:
            expect(any(want in w for w in verdict.get(n, [])), f"pure: {n} is not rejected with '{want}': {verdict.get(n)}")
        for n in ("guarded", "inner", "user"):
            expect(verdict.get(n) == [], f"pure: {n} is rejected: {verdict.get(n)}")
        # host nightly: the same lock reading over the synthetic workspace — every crate with no device root, whatever its
        # x86_64 or Linux-only code, and an admitted device-linked one only by name; the tables' stale entries are problems
        every = sorted(srcs)
        dev = ["devc", "devdep", "gitdep", "via", "wrap"]
        got = [n for n, _ in host_crates(pt, lock, {})]
        expect(got == [n for n in every if n not in dev], f"host nightly: the synthetic set is {got}")
        got = [n for n, _ in host_crates(pt, lock, {"via": "reason"})]
        expect("via" in got and "devc" not in got and len(got) == len(every) - len(dev) + 1, f"host nightly: an admitted crate is {got}")
        expect([n for n, _ in host_left_out(pt, lock, {"via": "reason"})] == ["devc", "devdep", "gitdep", "wrap"], "host nightly: the left-out list")
        os.makedirs(os.path.join(tmp, "tools"))
        for f in ("check-a.sh", "check-b.sh", "other.sh"):
            with open(os.path.join(tmp, "tools", f), "w", encoding="utf-8") as fh:
                fh.write("")
        units = host_units(pt, lock, {}, {"user": [("tests::one", "needs a file")], "x86": []}, {"check-b.sh": "needs a box"}, {"extra": "true"})
        expect(units[-2] == ("check-a", "bash tools/check-a.sh") and units[-1] == ("extra", "true") and not any(u == "check-b" for u, _ in units), f"host nightly: the check units {units[-3:]}")
        expect(dict(units)["user"].endswith("--no-fail-fast -- --exact --skip tests::one"), f"host nightly: a skip is exact: {dict(units)['user']}")
        expect(dict(units)["x86"].endswith("--no-fail-fast") and dict(units)["guarded"].startswith(HOST_GATE + " -p guarded"), "host nightly: a crate with no skip")
        with open(os.path.join(tmp, "user", "src", "lib.rs"), "a", encoding="utf-8") as fh:
            fh.write("#[test]\nfn one() {}\n")
        expect(any("HOST_EXTRA_UNITS names check-a" in g for g in nightly_problems(pt, lock, {}, {}, {}, {"check-a": "true"})), "host nightly: an extra unit that is a check script")
        for linked, skips, checks, want in [
            ({}, {"user": [("tests::one", "needs a file")]}, {"check-b.sh": "needs a box"}, None),
            ({"nope": "r"}, {}, {}, "HOST_DEVICE_LINKED names nope, which is no workspace crate"),
            ({"user": "r"}, {}, {}, "HOST_DEVICE_LINKED names user, which reaches no device root"),
            ({}, {"devc": [("a", "r")]}, {}, "HOST_TEST_SKIP names devc, which the set does not run"),
            ({}, {"user": [("tests::gone", "r")]}, {}, "no `fn gone` in the crate's sources"),
            ({}, {"user": [("tests::one", " ")]}, {}, "HOST_TEST_SKIP user tests::one: no reason"),
            ({}, {}, {"check-c.sh": "r"}, "HOST_CHECK_SKIP names check-c.sh, which is not a tools/check-*.sh"),
            ({}, {}, {"check-a.sh": ""}, "HOST_CHECK_SKIP check-a.sh: no reason"),
        ]:
            got = nightly_problems(pt, lock, linked, skips, checks, {})
            expect((not got) if want is None else any(want in g for g in got), f"host nightly: {want}: {got}")
    # the real tree: the rule's selection is the set whose native `cargo test` passes (`just mac-test`), and a
    # rejected crate names a device root or a source line
    real = dict(pure_crates(tree, load_lock(tree.root)))
    pure_now = sorted(n for n, w in real.items() if not w)
    expect(
        pure_now == ["bloomery-decision", "bloomery-hf", "bloomery-jinja", "bloomery-levers", "bloomery-models", "bloomery-placement", "bloomery-refset", "bloomery-runtime", "bloomery-sampler", "bloomery-serve", "bloomery-tokenizer", "bloomery-vision"],
        f"pure: the real tree's pure crates are {pure_now} — a new member of the set is proven by `just mac-test` before this list takes it",
    )
    expect(any("crates/gguf/src/lib.rs:" in w and "RUSAGE_THREAD" in w for w in real.get("bloomery-gguf", [])), f"pure: gguf's reason {real.get('bloomery-gguf')}")
    expect(any("bloomery-gpu -> cuda-" in w for w in real.get("bloomery-gpu", [])), f"pure: bloomery-gpu's reason {real.get('bloomery-gpu')}")
    expect(any("x86_64 intrinsics" in w for w in real.get("bloomery-qdot", [])), f"pure: qdot's reason {real.get('bloomery-qdot')}")
    # host nightly on the real tree: the crates the set runs are the device-free ones and the admitted app; every gpu crate is left out
    # by its chain; every check script is a unit or an excluded one; the tables hold no stale entry
    lock_now = load_lock(tree.root)
    host_now = [n for n, _ in host_crates(tree, lock_now)]
    expect(
        host_now
        == [
            "bloomery-app", "bloomery-decision", "bloomery-engram", "bloomery-engram-lab", "bloomery-gguf", "bloomery-hf", "bloomery-jinja",
            "bloomery-levers", "bloomery-model", "bloomery-models", "bloomery-placement", "bloomery-qdot", "bloomery-refset", "bloomery-runtime",
            "bloomery-sampler", "bloomery-serve", "bloomery-threads", "bloomery-tokenizer", "bloomery-vision",
        ],
        f"host nightly: the real tree's set is {host_now} — a new member of the set is proven by a nightly run before this list takes it",
    )
    left = [n for n, _ in host_left_out(tree, lock_now)]
    expect(left == ["bloomery-gpu", "bloomery-gpu-deepseek41", "bloomery-gpu-gates", "bloomery-gpu-glm5next", "bloomery-gpu-mimo2", "bloomery-gpu-vision"], f"host nightly: left out {left}")
    expect(not nightly_problems(tree, lock_now), f"host nightly: {nightly_problems(tree, lock_now)}")
    ran = {u for u, _ in host_units(tree, lock_now)}
    expect(
        all((f[:-3] in ran) != (f in HOST_CHECK_SKIP) for f in host_check_scripts(tree.root)),
        "host nightly: a check script is neither a unit nor excluded",
    )

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
        # a feature-gated bin built without its feature: cargo refuses it (required-features), and so
        # does the check, before the recipe runs
        m = re.search(r"--features deepseek41 (--release --bin generate_ds41)", text)
        if m is None:
            fails.append("no mutation anchor for the generate_ds41 build")
        else:
            p = os.path.join(tmp, "justfile")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(text[: m.start()] + "--features gpu " + m.group(1) + text[m.end() :])
            probs = check(Tree(ROOT), load_justfile(p))
            expect(any("--bin generate_ds41: needs features ['deepseek41']" in x for x in probs), f"generate_ds41 without deepseek41 not caught: {probs[:2]}")

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
    expect(sorted(x.recipe for x in sels) == ["gate-gpu-p1"], f"select() on gate_p1.rs and docs/plan.md gives {[x.recipe for x in sels]}")

    # dep-info parsing
    t, deps, root = parse_depinfo("/r/b/target/release/g: /r/b/crates/a\\ b.rs /r/b/crates/c.rs\n")
    expect(t.endswith("/g") and root == "/r/b" and "/r/b/crates/a b.rs" in deps, "dep-info parse")

    # orphan-tests
    orphan_self_test(expect, side)

    # affected --narrow
    narrow_self_test(expect, side)

    # the scan set
    scan_set_self_test(expect, side)

    # a package manifest read by table
    manifest_self_test(expect)

    # the static tier compiles every feature-gated bin
    static_self_test(expect)

    # the green ledger's key
    key_self_test(expect, side)
    ledger_self_test(expect)
    manifest_rows_self_test(expect)
    ref_paths_self_test(expect)
    box_tier_self_test(expect)
    oxide_rev_self_test(expect)

    # opens: the targets that reach the fixture fence
    opens_self_test(expect)

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
    a.add_argument("--narrow", action="store_true", help=f"the gates that run the changed host path, when every --scan pair is identical or added (the rule and {GATE_PATHS}: the section above narrow())")
    a.add_argument("--scan", nargs=2, action="append", metavar=("BASE_LOG", "NEW_LOG"), help="a pair of `just ptx-scan <bin>` logs, the base tree's and the change's (repeatable)")
    ss = sub.add_parser("scan-set", help="the bins whose ptx-scans cover every kernel carrier, one line each: bin<TAB>features<TAB>the carriers it covers (tools/narrow-scan.sh scans them)")
    ss.add_argument("rev", nargs="?", help="the commit to read (a git archive of it); default the working tree")
    w = sub.add_parser("why")
    w.add_argument("files", nargs="+")
    w.add_argument("--all-recipes", action="store_true")
    rh = sub.add_parser("reach", help="the gate-* recipes whose targets can run code of each FILE through a dependency (the derivation "
                        f"{GATE_PATHS}'s rows are held to), each with one chain")
    rh.add_argument("files", nargs="+")
    rh.add_argument("--brief", action="store_true", help="the gates' names without their chains")
    rh.add_argument("--row", action="store_true", help="last, the union of the files' gates as a row's first two columns")
    k = sub.add_parser("key", help="each ITEM's ledger key: key<TAB>item<TAB>status<TAB>detail")
    k.add_argument("items", nargs="+", metavar="ITEM", help="NAME[@K=V,…][:ARGS], as tools/gate-batch.sh runs it (its lane's card in the env)")
    k.add_argument("--manifest", help="the box manifest (box-manifest's output)")
    k.add_argument("--manifest-error", help="the manifest fetch failed with this message: every item is an error")
    k.add_argument("--box-env", default=os.environ.get("BLOOMERY_BOX_ENV", ""), help="the caller's BLOOMERY_BOX_ENV")
    k.add_argument("--ledger", help="the ledger file: status skip | run | rerun | never | error instead of key")
    k.add_argument("--round-ledger", help="with --ledger: a second ledger, the rounds' (gate-batch.sh --round-ledger, "
                   "--trust-rounds); a key green there skips too, after the first ledger, as src=round")
    k.add_argument("--rerun", action="store_true", help="with --ledger: no item skips")
    k.add_argument("--parts-dir", help="write each item's labelled parts to DIR/<index>.parts")
    k.add_argument("--parts", dest="show_parts", action="store_true", help="print each item's labelled parts")
    bc = sub.add_parser("box-command", help="the single-quoted argument of RECIPE's one tools/box.sh line, verbatim (tools/mac-check.sh)")
    bc.add_argument("recipe")
    ot = sub.add_parser("orphan-tests", help="every #[test] no gate-* or lab-* recipe runs on the box, one `path:line name — why` line each")
    ot.add_argument("--justfile")
    pc = sub.add_parser("pure-crates", help="the crates tools/mac-check.sh test runs natively on the Mac, each rejected one with its reason")
    pc.add_argument("--names", action="store_true", help="the pure crates' names only, one a line")
    hn = sub.add_parser("host-nightly", help="the units tools/nightly/run.sh runs on a Linux host with no card and no model files: name<TAB>command, in run order")
    hn.add_argument("--excluded", action="store_true", help="every crate, test and check the set leaves out, with why: kind<TAB>name<TAB>reason")
    cb = sub.add_parser("combos", help="every build shape a recipe compiles, one `cargo check` command each (tools/mac-check.sh combos): combo<TAB>label<TAB>command, skip<TAB>recipe<TAB>why, total<TAB>counts; --base scopes to shapes an input file of which changed (combo lines gain a fourth field, the input key)")
    cb.add_argument("--justfile")
    cb.add_argument("--base", help="a shape whose inputs hold no file changed since this BASE (or A..B) skips with why; with --ledger, a green key skips too")
    cb.add_argument("--ledger", help="with --base: the Mac static ledger (tools/mac-check.sh combos --ledger appends it) whose green keys skip")
    b = sub.add_parser("box-manifest", help="on the box, through tools/box.sh: the key's box part")
    b.add_argument("--lease", action="append", help="a timing lease lock; held, the manifest refuses (exit 75)")
    b.add_argument("--cache", default="~/.cache/bloomery/sha256-cache.tsv", help="the stat-keyed sha256 cache")
    b.add_argument("--workers", type=int, default=8)
    op = sub.add_parser("opens", help="per RECIPE, whether a cargo target it builds or runs reaches the fixture fence (Tier::check_file): opens<TAB>recipe<TAB>pkg:kind:target<TAB>path:line<TAB>called, or none<TAB>recipe (tools/gate-batch.sh's real-file deferral)")
    op.add_argument("items", nargs="*", metavar="RECIPE")
    op.add_argument("--explain", action="store_true", help="the fence, the openers and each recipe's targets, on stderr")
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
        if args.cmd == "scan-set":
            return cmd_scan_set(args)
        if args.cmd == "why":
            return cmd_why(args)
        if args.cmd == "reach":
            return cmd_reach(args)
        if args.cmd == "key":
            return cmd_key(args)
        if args.cmd == "box-command":
            return cmd_box_command(args)
        if args.cmd == "orphan-tests":
            return cmd_orphan_tests(args)
        if args.cmd == "pure-crates":
            return cmd_pure_crates(args)
        if args.cmd == "host-nightly":
            return cmd_host_nightly(args)
        if args.cmd == "combos":
            return cmd_combos(args)
        if args.cmd == "box-manifest":
            return cmd_box_manifest(args)
        if args.cmd == "opens":
            return cmd_opens(args)
        ap.print_help()
        return 64
    except RecipeError as err:
        print(f"recipes.py: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
