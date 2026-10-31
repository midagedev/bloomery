#!/usr/bin/env python3
"""The justfile's recipes as cargo targets and input files — one parser for check-recipes and affected.

Runs on the Mac: it reads the justfile through `just --dump --dump-format json` (just's own parser),
the workspace through `cargo metadata --format-version 1 --no-deps --offline --locked` (no build, no
network, never rewrites Cargo.lock), and the sources as text. It builds nothing and runs no gate.

    python3 tools/recipes.py check            # the target checks of `just check-recipes`, one feature set a bin, and the plan files records-refresh writes (presence)
    python3 tools/recipes.py targets [RECIPE]  # recipe -> cargo targets, scripts, input count
    python3 tools/recipes.py affected [BASE | A..B] [--no-box] [--all-recipes] [--narrow [--scan BASE_LOG NEW_LOG]...] [--no-class] [--no-comment-only]
    python3 tools/recipes.py release-list [TAG | TAG..HEAD] [--wall]  # the release batch's list: the S0 gates, the R families' gates, affected without the F families' gates and the rotation's due gates
    python3 tools/recipes.py scan-set [REV]    # the bins whose ptx-scans cover every kernel carrier: bin<TAB>features<TAB>carriers (just narrow scans them)
    python3 tools/recipes.py ships [--rev REV] [--base REV] FILE...  # the FILEs inside the shipped closure (tools/release/build.sh's bins at its features), with why; exit 1 when any is (tools/release/late-commit.sh)
    python3 tools/recipes.py ship-scan [--rev REV] BASE_LOG HEAD_LOG...  # the shipped bins' ptx-scan pairs: a verdict line each; exit 0 only when every shipped bin has a pair and all are identical
    python3 tools/recipes.py ptx-compare BASE_LOG HEAD_LOG [--base-label L] [--head-label L]  # two full ptx-scan logs of one bin as one table: entries, added, removed, changed (tools/ptx-compare.sh); rc 0 nothing existing moved, 3 changed, 4 removed, 5 both
    python3 tools/recipes.py ship-bins [--rev REV]  # the shipped bins and the release features read from tools/release/build.sh: bin<TAB>features
    python3 tools/recipes.py why FILE... [--recipe R]...  # every recipe a file selects, with the chain; --recipe R: whether R's key holds it and R's shortest dependency chain to the file's crate
    python3 tools/recipes.py gaps [CRATE...] [--rev REV] [--list]  # per crate: recipes whose key holds a file of it, recipes whose targets reach code of it, the gap; exit 1 on a gap of a pinned crate (KEY_REACH_PINS)
    python3 tools/recipes.py rows              # tools/gate-paths.tsv as narrow() reads it: each `derive` row's gates written out, line for line (cached by tree)
    python3 tools/recipes.py reach FILE... [--brief] [--row] [--mac-landing]  # the gates whose targets can run the file's code, each with a chain (--brief: the reason; --mac-landing: the list a mac-crate change runs)
    python3 tools/recipes.py key --manifest F [--ledger L [--round-ledger R]] ITEM...  # the green ledger's key per item
    python3 tools/recipes.py box-manifest      # on the box, through tools/box.sh: the key's box part
    python3 tools/recipes.py box-command RECIPE  # the recipe's box.sh command, verbatim (tools/mac-check.sh derives from it)
    python3 tools/recipes.py opens [RECIPE...] [--explain]  # per recipe, whether a target it builds or runs reaches the fixture fence (tools/gate-batch.sh's real-file deferral)
    python3 tools/recipes.py pure-crates [--names]  # the crates tools/mac-check.sh test runs natively on the Mac, each rejected one with its reason
    python3 tools/recipes.py host-nightly [--excluded]  # the units tools/nightly/run.sh runs on a Linux host with no card and no model files: name<TAB>command
    python3 tools/recipes.py combos           # every build shape a recipe compiles, one cargo check command each (tools/mac-check.sh combos)
    python3 tools/recipes.py orphan-tests      # every #[test] no gate-* or lab-* recipe runs on the box
    python3 tools/recipes.py expand [--justfile F]  # every recipe as just runs it, each top-level variable replaced by its value in this environment
    python3 tools/recipes.py --self-test [--serial]  # its sections run as forked processes; --serial: one after another in this one

`affected` prints the gate-* recipes whose inputs a diff touches; nothing runs. BASE (default
`main`) compares the working tree, uncommitted and untracked files included, with the merge base of
BASE and HEAD — a round's tree is never committed. `A..B` compares B with the merge base of A and B
(`git diff A...B`; A itself when it is an ancestor of B), the `affected:` line naming the base it
used; both sides are read from `git archive` copies, so the graph is the one those commits had.

A recipe that a row of tools/fixture-variants.tsv lists also prints its variant item, `NAME@BLOOMERY_FIXTURE_VARIANT=<tag>`, as a line of
its own and in the `recipes:` line (a batch `--list` takes the first word of the line). A row that was added or edited prints the items of its
recipes whether or not the diff selected them: the variant's file lives outside the repository, so the row is what a change to it is.

The lever registry (crates/levers/src/registry.rs) is read by row when every edit to it is a row (`LeverReads`): an added, changed or removed
element of its REGISTRY array, or the `&str` const that names one. It selects the recipe that tests the registry and the recipes whose inputs
hold a file outside crates/levers that names a moved row, by its variable or by a const that spells it; a recipe that links the crate and
reads no such file is not selected. Code beside the rows (an item, a helper's body, a non-string const), a row the reader cannot name, rows in
another order and a text it cannot lex keep the file whole, the `affected:` line saying why, and so does any other file of crates/levers.

A Rust file of a workspace crate whose two sides have the same tokens once comments and white space are left out selects no gate: both
`select` and `narrow` skip it, `unmapped` and `narrow: file` print `comment-only, no gate reads it` (`layout-only` when no comment moved),
and the `affected:` line counts such files (`CommentReads`; `--no-comment-only` turns the rule off). The static tier (`always:`) is not
touched. A file stays selected, `notes` saying why, when something reads its text or its doc comments: an `include_str!`, `include_bytes!`
or `include!` names it; a doc comment moved in the lib of a crate whose doctests a gate runs (`--doc`); the lever registry (read by row);
a file outside every workspace crate; a string or comment that never ends.

Under --narrow a Rust file of a lib's module tree that no row of tools/gate-paths.tsv maps is read by the derivation (`dep_reach`, what `reach`
prints): the gates whose reach holds a unit of it, the `affected:` line counting such files and a `narrow: derived` block naming each file's gates
and one chain. A file the derivation cannot read (`derivation_gap`: not Rust source, deleted, a build script, in no lib's module tree; or
`derivation_empty`: no gate's reach holds a unit of it) keeps the full list, `narrow: full list` naming the file and why.

The class stage (on; `--no-class` turns it off) is the third stage, after select() and narrow(): the class tables (tools/gate-class.tsv,
tools/families.tsv, tools/owners.tsv, read through tools/gate_class.py's loader) say what a gate-* recipe is, and a family gate is kept only
when the change can have moved it — the family's own file changed, the recipe's own text or one of its own target files changed, the family is
a representative one and the pick reaches a changed file, or the family was added since the last release tag. The gate it leaves out is printed
with its reason, in the same `left out` block the ptx-scan stage uses. With `--no-class` that stage is not applied and every gate the graph
reaches stays.
A change whose selecting files all sit under `MAC_CRATES` does not select a CUDA gate on a GGUF input (`mac_landing_deferred`); the gate is
printed under `deferred`. `release-list` is the release batch's list: every S0 gate (a common or a tooling gate the fixture tier runs), every
gate of a representative family, affected(TAG) without the gates of the F families (the rotation holds those: tools/rotation.py check), the
rotation's due gates and every CUDA gate on a GGUF input (`mac_gguf_cuda`).

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
    dependency sees it. A module under a one-line `#[cfg(feature = …)]` (alone or under `all`, `any`,
    `not`) that the recipe's `--features` make false is left out of its own package's trees (the target's
    and its lib's) — a build that is not one `-p` with its whole feature list in `--features` (a recipe
    parameter, a `pkg/feature`, several packages or a workspace: cargo unifies features across a build) is
    read whole; another package's lib is read whole too, whatever the dependent enables;
  - on the box's last build, the bin's top-level dep-info (`target/release/<bin>.d`, which lists every
    local source file of the binary) — a cross-check of the walk, and a union with it;
  - the scripts the recipe text names (`tools/**`, `crates/*/tools/**`) and, transitively, the repository
    files those scripts name, and every file under a tree directory a shell script walks with `find`
    from its root variable (`find "$ROOT/docs" -name '*.md'`: the whole directory, whatever the filter);
  - for a box recipe, tools/box.sh and what it sources on every command: tools/ref/ref-paths.sh and the
    fixture directory table it reads, tools/fixture-dirs.tsv, tools/ref/models/deepseek41.sh and the recipe's profile (`BLOOMERY_MODEL=x` on the line, else
    deepseek2);
  - for a cargo recipe, Cargo.toml, Cargo.lock, rust-toolchain.toml, .cargo/config.toml, and
    tools/oxide-host.sh (the host's CPU flags and llc), tools/oxide-backend.sh and its patch when the recipe runs `cargo oxide`. A virtual workspace's Cargo.toml is read by
    table too (`root_moves`): of [workspace], `members`/`exclude` select the recipes that read a package that
    entered or left the workspace (an exclude of a path that held no member selects nothing),
    `default-members` the recipes with a cargo line naming no `-p` and no `--workspace`, and each moved
    `[workspace.dependencies]` entry the recipes whose closure holds a package that inherits it (`name = { workspace = true }`
    in a table the recipe's host reads; an entry no package inherits stays whole, with a note); `[workspace.lints]`, like a
    package's `[lints]`, selects no recipe: a lint level moves no codegen or behaviour, the static tier holds it (`just lint`,
    `just check`), and the `affected:` line names it; every other table ([profile.*], [patch], [workspace.package], ...),
    rust-toolchain.toml and .cargo/config.toml select every cargo recipe. Cargo.lock is read by package (`LockDiff`):
    the packages its diff moves (appeared, went, a version, source or checksum, a dependency list; a workspace member's
    own entry too) select the recipes whose build on their host reaches one, by a walk over the lock from the packages
    the recipe's targets link, on x86_64-unknown-linux-gnu for a box recipe and aarch64-apple-darwin for one that runs
    tools/mac-gate.sh (a member's `[target.<key>]` edge counts by `table_read`; an external package's lock edges are all
    followed, except one only one side's lock has, which a feature the host does not enable puts there). A package every
    target reaches selects every recipe, by that same rule; a diff no recipe reaches selects none and prints a note
    naming the packages. A lock the reader cannot take apart (one side missing or not parsing, a moved top-level key
    beside `package`, a diff that names no package) stays whole for every cargo recipe, with a note. A package manifest's `[target.<key>]` table is
    left out of its view the same way when its `cfg(…)` is false for x86_64-unknown-linux-gnu or its key is another
    triple; a predicate this tool cannot decide stays, with a note. A recipe that runs tools/mac-gate.sh builds on
    aarch64-apple-darwin: these rules read that target for it, so a macOS-only change selects the Mac gates and no box
    gate. The ledger key still hashes Cargo.toml, Cargo.lock and the manifests whole;
  - its own text in the justfile (a recipe whose text changed is selected), and its dependencies'
    inputs.

The ledger's `key` uses a separate `Graph.key_inputs` contract: the uncut structural item graph, the selected
targets' own module/root files, mandatory build-script/build-dependency/procedural-macro compilation inputs,
manifest views, dep-info, explicit reads and dependency recipes. Unused linked runtime modules are not ledger
inputs. Procedural/startup attributes are inspected throughout every runtime-active owner's compiled files
(an uncut reached item or explicit extern-crate link activates an owner). Opaque attributes/derives retain
their owner's compiled source/literals and linked dependency closure, not unrelated reverse/sibling packages.
This is compilation-code ownership, not arbitrary procedural-macro filesystem completeness:
explicit/script reads remain independently keyed.
Opaque macros retain their owner's compilation/dependency closure. Unknown generated includes and foreign
linkage retain the broad recipe inputs conservatively.
Reach's complete structural facts/edges are privately cached as validated JSON, content-checked against source
bytes, enumeration, metadata and analysis code; that cache identity never enters individual ledger keys.

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
     posix_fadvise, sync_file_range, RUSAGE_THREAD, renameat2 and RENAME_NOREPLACE, prctl,
     MADV_POPULATE_*, O_DIRECT, the huge-page flags), memmap2's Linux-only advice
     (PopulateRead/Write, HugePage, NoHugePage), and a "/proc/" or
     "/sys/" path literal (the file does not exist on the Mac, so the code that reads it fails there).
     Comments are not code. An item or statement under a `#[cfg(…)]` that is false on
     aarch64-apple-darwin (target_arch, target_os, target_family, target_vendor, target_env,
     target_pointer_width, target_endian, unix, windows) is left out, up to its end — the item's own
     end first (a signature's comma, in `Result<A, B>`, does not end it), then the `;` or `,` at
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
of the units are every `tools/check-*.sh` minus HOST_CHECK_SKIP, then HOST_EXTRA_UNITS (none now; check-recipes.sh runs the recipes.py self-test). A skip that names no `fn` of its crate, a crate outside the set or a script that is gone is an
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
enabled features — the closure, defaults included — and any `dep/feature` words for its dependencies); the
recipes that share a shape share one command, `cargo check -p <package> [--profile test] [--features …]` over
the union of their targets, so each shape is type-checked the way the box compiles it. One package a command:
two in one invocation would unify a shared dependency's features and hide a feature one of them lacks. A
`dep/feature` word for a package of the same call that this package does not depend on (`-p a -p b --features
b/x`) is b's own feature, left out of a's shape: cargo refuses it for `-p a` alone, and a call where neither
holds is a named error of `check`. A
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

Bin feature sets (`check`). A build writes its bin to `<target dir>/<profile>/<bin>` (`--release`, `--profile`, else
dev; below `--target`; the target dir is `--target-dir`, cargo-oxide's `--cargo-target-dir`, the `CARGO_TARGET_DIR` of the
call's prefix or an `export` before it, else `target`), and cuda-host reads the PTX bundle from `current_exe()` at each
module load, so a second build with another feature set replaces the file a running gate loads its kernels from
(2026-10-12: gate_glm5next_e2e, `glm5next,vision` against `glm5next`). `check` refuses a (package, bin, directory) that
two builds give different resolved sets — the enabled features after the package's closure plus its `dep/feature` words,
so `gpu,deepseek41` and `deepseek41` agree — and prints the bin, its package and directory, and every recipe grouped by
the set it resolves to, the widest first. Only `build`, `run` and `rustc` write the bin: a test or bench build writes a
hash-named file under deps/, and check and clippy write none. There is no exemption for one recipe building a bin with
several sets in turn: the file is replaced under whichever lane runs it just the same (2026-10-13, gate-ptx-spill's
oxart_jit, built three times; it now builds it once). An invocation whose package, features, targets or target dir come
from a recipe parameter is left out (the ptx-scan, sass-scan and lds-scan recipes): their features are known at run time
only. A bin tools/release/build.sh ships is held to that file's feature list instead, in any
dir, and the one line that names a departure is that rule's (`bin_problems`). The fix is the union (each build gives the
bin every feature any build gives it), or a target dir of its own for a bin no runner starts from target/release
(tools/gpu-gate.sh runs `target/release/<bin>` or the test artifact gate.sh --gpu compiled); `--arch` is not part of the key: cargo-oxide's only target-dir
option is `--cargo-target-dir`, and the justfile passes one `{{arch}}` to every build.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import concurrent.futures
import dataclasses
import datetime
import difflib
import fcntl
import functools
import glob
import hashlib
import importlib.util
import itertools
import json
import os
import platform
import re
import shlex
import shutil
import stat
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

_REACH_LOADED_CODE = sys._getframe().f_code

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # `python3 -I` drops the script's directory
from bloomery.fixture_dirs import FAMILY_NAME, FIXTURE_DIRS, REF_PATHS, RecipeError, fixture_table  # noqa: E402,F401

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GATE_PREFIX = "gate-"
# A gate taken out of the landing batch: `just weekly` runs every one (and the rotation's due list, tools/rotation.py),
# and `just affected` names one only when a changed file matches one of its trigger rows in tools/gate-paths.tsv, or its
# own recipe text changed.
WEEKLY_PREFIX = "weekly-"
# The checks the lead's batches always run; the diff does not pick them.
ALWAYS = ["check-recipes", "check-rustflags", "check-comments", "check-levers", "check-arch", "check-model-tax", "check", "fmt-check", "lint"]
CARGO_GLOBALS = ["Cargo.toml", "Cargo.lock", "rust-toolchain.toml", ".cargo/config.toml"]
LOCK_FILE = "Cargo.lock"
# The workspace's root manifest: a cargo global every cargo recipe reads, narrowed by table (`root_moves`) where it is a
# virtual manifest; as a package's own manifest it stays whole.
ROOT_MANIFEST = "Cargo.toml"
OXIDE_GLOBALS = ["tools/oxide-host.sh", "tools/oxide-backend.sh", "tools/oxide-backend/cuda-rust-1648.patch"]
BOX_GLOBALS = ["tools/box.sh", "tools/ref/ref-paths.sh", "tools/fixture-dirs.tsv", "tools/ref/models/deepseek41.sh"]
DEFAULT_PROFILE = "deepseek2"
# Runners that execute target/release/<first argument>.
BIN_RUNNERS = ("tools/gpu-gate.sh", "tools/host-gate.sh", "tools/ref/time-gate.sh")
BUILD_SUBS = {"build", "test", "run", "check", "clippy", "bench", "doc", "rustc"}


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


def _just(path: str, *words: str) -> str:
    """`just` on the justfile at PATH, its stdout. Refused by name when just is missing or exits non-zero."""
    if shutil.which("just") is None:
        raise RecipeError("`just` is not on PATH — recipes.py reads the justfile through `just --dump`")
    proc = subprocess.run(
        ["just", "--justfile", path, "--working-directory", os.path.dirname(path) or ".", *words],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RecipeError(f"`just {' '.join(words)}` failed on {path} (exit {proc.returncode}): {proc.stderr.strip()}")
    return proc.stdout


def load_justfile(path: str) -> dict[str, Recipe]:
    dump = json.loads(_just(path, "--dump", "--dump-format", "json"))
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
    profile: str = "dev"  # `--release`/`-r` or `--profile <name>`; no flag is cargo's dev
    target: str = ""  # `--target <triple>`: the build writes under target/<triple>/
    target_dir: str = ""  # where the call puts target/: `--target-dir`, oxide's `--cargo-target-dir`, else the `CARGO_TARGET_DIR` the shell holds (the flag wins)
    raw: str = ""
    names: list[str] = field(default_factory=list)  # positional test-name filters before `--` (`cargo test <filter>`)
    test_args: list[str] = field(default_factory=list)  # the test binary's arguments, after `--`


def parse_cargo(args: list[str], raw: str, via_gate_sh: bool = False) -> Invocation | None:
    """`cargo …` words after `cargo` (or tools/gate.sh's arguments). None for a non-build subcommand."""
    if not args:
        raise RecipeError(f"bare `cargo` in: {raw}")
    oxide = False
    oxide_words: list[str] = []
    if via_gate_sh:
        sub = "test"
        rest = list(args)
        if rest and rest[0] == "--gpu":
            rest = rest[1:]
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
            oxide_words = rest[: rest.index("--")]
            rest = rest[rest.index("--") + 1 :]
        else:
            oxide_words = rest
            rest = []
    else:
        sub = args[0]
        rest = args[1:]
    if sub not in BUILD_SUBS:
        return None
    # cargo-oxide sets CARGO_TARGET_DIR for cargo from its own `--cargo-target-dir`, before the `--`
    oxide_dir = ""
    for k, w in enumerate(oxide_words if oxide else []):
        if w == "--cargo-target-dir" and k + 1 < len(oxide_words):
            oxide_dir = oxide_words[k + 1]
        elif w.startswith("--cargo-target-dir="):
            oxide_dir = w.split("=", 1)[1]
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
            elif opt == "--profile":
                inv.profile = val
            elif opt == "--target":
                inv.target = val
            elif opt == "--target-dir":
                inv.target_dir = val
            elif opt == "--bin":
                inv.selectors.append(("bin", val))
            elif opt == "--test":
                inv.selectors.append(("test", val))
            elif opt == "--example":
                inv.selectors.append(("example", val))
            elif opt == "--bench":
                inv.selectors.append(("bench", val))
        elif opt in _CARGO_FLAG:
            if opt in ("--release", "-r"):
                inv.profile = "release"
            elif opt == "--lib":
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
    inv.target_dir = inv.target_dir or oxide_dir
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
    cargo_target_dir: str = ""  # the CARGO_TARGET_DIR an `export` earlier in the same shell (one recipe line, one box.sh call) left set
    models: list[str | None] = field(default_factory=list)  # the BLOOMERY_MODEL prefix of each tools/box.sh call, in order (None: that call sets none); `env` keeps the last only

    def profiles(self, environ: dict[str, str] | None = None) -> set[str]:
        """Box profiles, with shell defaults resolved and recipe parameters left for the caller."""
        profiles: set[str] = set()
        for model in self.models:
            default = re.fullmatch(r"\$\{BLOOMERY_MODEL:-([A-Za-z0-9_-]+)\}", model or "")
            if model is None or default:
                profile = (environ or {}).get("BLOOMERY_MODEL") or (default.group(1) if default else DEFAULT_PROFILE)
            elif "{{" in model and environ is None:
                continue
            else:
                profile = model
            if not FAMILY_NAME.fullmatch(profile):
                raise RecipeError(f"BLOOMERY_MODEL={profile!r}: cannot resolve model profile")
            profiles.add(profile)
        return profiles


def _norm(p: str) -> str:
    p = p[2:] if p.startswith("./") else p
    return os.path.normpath(p)


def _classify(cmd: list[str], rc: RecipeCommands, raw: str) -> None:
    env, cmd = strip_wrappers(cmd)
    if not cmd:
        return
    head, args = cmd[0], cmd[1:]
    # A target dir reaches cargo from a flag, from a prefix on its own call, or from an `export` before it (a bare
    # assignment is not exported, so it does not count); the flag wins, then the prefix, then the export.
    if head == "export":
        for w in args:
            if w.startswith("CARGO_TARGET_DIR="):
                rc.cargo_target_dir = w.split("=", 1)[1]
        return
    if head == "unset" and "CARGO_TARGET_DIR" in args:
        rc.cargo_target_dir = ""
        return
    held = env.get("CARGO_TARGET_DIR") or rc.cargo_target_dir
    if head == "cargo":
        inv = parse_cargo(args, raw)
        if inv is not None:
            inv.target_dir = inv.target_dir or held
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
            inv.target_dir = inv.target_dir or held
            rc.invocations.append(inv)
    elif s in BIN_RUNNERS and sargs:
        rc.runs.append(sargs[0])


def recipe_commands(recipe: Recipe) -> RecipeCommands:
    rc = RecipeCommands()
    for line in recipe.lines:
        stripped = line.lstrip("@-").strip()
        if not stripped or stripped.startswith("#"):
            continue
        rc.cargo_target_dir = ""  # just runs each line in a shell of its own: an `export` does not outlive its line
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
                rc.models.append(env.get("BLOOMERY_MODEL"))
                remote = " ".join(rest[1:])
                for p in _PATHLIKE.findall(remote):
                    rc.paths.append(_norm(p))
                outer, rc.cargo_target_dir = rc.cargo_target_dir, ""  # a box.sh call is a remote shell of its own
                for rcmd in simple_commands(shell_words(remote)):
                    _classify(rcmd, rc, remote)
                rc.cargo_target_dir = outer
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
    proc_macro: bool = False  # Cargo's target kind, not inferred from source spelling


@dataclass
class Dep:
    key: str  # the name the manifest uses (rename or package name)
    pkg: str
    kind: str  # normal | dev | build
    optional: bool
    features: list[str]
    default: bool
    target: str | None = None  # the key of the `[target.<key>]` table it sits in (cargo metadata's `target`); None in a plain table


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
# a file for that recipe, never a view (`ManifestUse.whole`). Selection's view also leaves out the
# `[target.<key>]` tables a build on the recipe's host does not read (`host`, `table_read`: the box, or the Mac
# for a recipe that runs MAC_GATE); the key's view keeps them, so an edit of one moves keys and selects nothing.
TARGET_TABLES = ("bin", "test", "bench", "example")
Entry = tuple[str, str, str]  # (kind, name, src) of a target whose entry a view keeps
_TOML_MEMO: dict[tuple[str, str], dict] = {}


def read_manifest(root: str, rel: str) -> dict:
    """`rel` parsed with tomllib, memoized on its bytes. A manifest that does not parse is a named error,
    never a fallback to the whole file."""
    import tomllib

    p = os.path.join(root, rel)
    with open(p, "rb") as fh:
        data = fh.read()
    k = (p, hashlib.sha256(data).hexdigest())
    if k not in _TOML_MEMO:
        try:
            _TOML_MEMO[k] = tomllib.loads(data.decode("utf-8"))
        except (tomllib.TOMLDecodeError, UnicodeDecodeError) as err:
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


def table_read(key: str, host: str = "box") -> bool | None:
    """Whether a build on `host` (HOSTS) reads the `[target.<key>]` tables of a package manifest: a `cfg(…)` key by
    `eval_cfg` on the host's keys, any other key by being the host's triple. None when the predicate is one this
    tool cannot decide (`target_feature`, a feature, a form `parse_cfg` does not read)."""
    triple, keys, names, _ = HOSTS[host]
    if key.startswith("cfg(") and key.endswith(")"):
        return eval_cfg(parse_cfg(key[4:-1]), None, keys, names)
    return key == triple


def recipe_host(recipes: dict[str, Recipe], name: str, _seen: frozenset[str] = frozenset()) -> str:
    """`mac` for a recipe that runs MAC_GATE, itself or through a dependency (its tests run on the Mac, under the Mac's
    cfg, as `orphan_tests` reads it), else `box`."""
    r = recipes.get(name)
    if r is None or name in _seen:
        return "box"
    if any(MAC_GATE in line for line in r.lines):
        return "mac"
    return "mac" if any(recipe_host(recipes, d, _seen | {name}) == "mac" for d in r.deps) else "box"


def manifest_view(doc: dict, pkg_dir: str, wants: frozenset[Entry] | None, host: str | None = None) -> dict:
    """The tables of `doc` a build of the targets `wants` sees (None: the whole document): the shared
    tables, and of each target array the entries it keeps, in canonical order (an array's order is not
    an input). With a `host`, a `[target.<key>]` table that `table_read` says False there is left out."""
    out = {k: v for k, v in doc.items() if not _target_array(k, v)}
    for k in TARGET_TABLES:
        v = doc.get(k)
        if _target_array(k, v):
            kept = sorted((e for e in v if _keeps(e, k, pkg_dir, wants)), key=canon)
            if kept:
                out[k] = kept
    if host is not None and isinstance(out.get("target"), dict):
        tables = {k: v for k, v in out["target"].items() if table_read(k, host) is not False}
        if tables:
            out["target"] = tables
        else:
            del out["target"]
    return out


def _target_tables(doc: dict | None) -> dict:
    t = doc.get("target") if doc is not None else None
    return t if isinstance(t, dict) else {}


def target_moves(da: dict | None, db: dict | None) -> list[tuple[str, bool | None]]:
    """(key, `table_read(key)` on the box) of each `[target.<key>]` table that differs between two parses of a manifest,
    in key order: what the box's view drops (False) or keeps without knowing (None) of a change. A side with no file
    has none: the whole file moved."""
    if da is None or db is None:
        return []
    ta, tb = _target_tables(da), _target_tables(db)
    return [(k, table_read(k)) for k in sorted(set(ta) | set(tb)) if canon(ta.get(k)) != canon(tb.get(k))]


def target_notes(f: str, da: dict | None, db: dict | None) -> list[str]:
    """A note for each `[target.<key>]` table of manifest `f` that changed and is not plainly the box's: one the box's view
    leaves out, and one it keeps because the predicate is unknown."""
    out = []
    for k, verdict in target_moves(da, db):
        if verdict is False:
            out.append(f"{f}: [target.{k}] changed, not in the x86_64-linux build")
        elif verdict is None:
            out.append(f"{f}: [target.{k}] changed; its predicate is not one this tool can decide for x86_64-linux, so the table stays in the view")
    return out


# what holds a lint level: it moves no codegen and no behaviour, so no gate runs for it
STATIC_TIER = "just lint, just check"


def lints_moved(da: dict | None, db: dict | None) -> bool:
    """Whether the `[lints]` table of a package manifest differs between two parses of it: held by the static tier, so
    `view_moves` leaves it out. A side with no file is the whole file, not a lint move."""
    return da is not None and db is not None and canon(da.get("lints")) != canon(db.get("lints"))


_DEP_SECTIONS = ("dependencies", "dev-dependencies", "build-dependencies", "dev_dependencies", "build_dependencies")


def inherits(doc: dict, name: str, host: str | None) -> bool:
    """Whether package manifest `doc` takes dependency `name` from the root's `[workspace.dependencies]`
    (`name = { workspace = true }`, or `name.workspace = true`) in a table a build on `host` reads: a plain table, or
    a `[target.<key>]` table `table_read` does not say False for (None: any table)."""

    def takes(tables: dict) -> bool:
        for sec in _DEP_SECTIONS:
            deps = tables.get(sec)
            spec = deps.get(name) if isinstance(deps, dict) else None
            if isinstance(spec, dict) and spec.get("workspace") is True:
                return True
        return False

    if takes(doc):
        return True
    target = doc.get("target")
    return isinstance(target, dict) and any(isinstance(body, dict) and (host is None or table_read(k, host) is not False) and takes(body)
                                            for k, body in target.items())


def _by_name(entries: list[dict] | None) -> dict[str, str]:
    """A target array's entries by name ("" for the nameless), each name's entries in canonical form."""
    acc: dict[str, list[str]] = {}
    for e in entries or []:
        n = e.get("name")
        acc.setdefault(n if isinstance(n, str) else "", []).append(canon(e))
    return {n: canon(sorted(es)) for n, es in acc.items()}


def view_moves(da: dict | None, db: dict | None, pkg_dir: str, wants: frozenset[Entry] | None, host: str = "box") -> list[tuple[str, bool]]:
    """What moved between two parses of a manifest in the view of `wants`, one (label, own) per table or
    entry — `[features]`, `[[bin]] x (added)` — where `own` is a named entry of a target array, not a shared
    table. The view is the one a build on `host` sees (`manifest_view`). Empty when the view did not move; a side
    with no file is the whole file."""
    if da is None or db is None:
        return [("the whole file (" + ("added" if da is None else "removed") + ")", False)] if da is not db else []
    va, vb = manifest_view(da, pkg_dir, wants, host), manifest_view(db, pkg_dir, wants, host)
    out: list[tuple[str, bool]] = []
    for k in sorted(set(va) | set(vb)):
        x, y = va.get(k), vb.get(k)
        if canon(x) == canon(y) or k == "lints":  # a lint level moves no build: the static tier holds it (`lints_moved`)
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


# ---- the root manifest, read by table ----
#
# A virtual workspace's root Cargo.toml is every cargo recipe's input, and a change of it names the recipes it reaches
# by table and by entry, as a package manifest does:
#   - [workspace] `members` and `exclude` decide which packages are members. The reach is the recipes that read a
#     package that entered or left (TableReads.delta: both sides' `cargo metadata` member sets, which hold cargo's own
#     rules — globs, an explicit member against an exclude, a path dependency inside the root being a member — and
#     are not recomputed here): its own recipes and its dependents', through the mapping every file uses. A member
#     set equal on both sides moves no recipe (`exclude = [..., "fuzz"]` where fuzz was no member, a reorder, a glob).
#   - [workspace] `default-members` decides the build that names no `-p` and no `--workspace`: the recipes with such
#     an invocation (RecipeInputs.bare_cargo), none while every cargo line names its packages.
#   - [workspace.dependencies] is read by entry: a moved entry reaches the recipes whose closure holds a package that
#     inherits it (`name = { workspace = true }`, TableReads.inheritors) in a dependency table, or a `[target.<key>]`
#     one, that the recipe's host reads. An entry no package of either side inherits, or a manifest that does not
#     parse while the inheritors are read, leaves the entry whole.
#   - [workspace.lints] moves no recipe: a lint level changes no codegen and no behaviour, and the static tier
#     (`just lint`, `just check`) holds it. `affected` names it on its summary line (TableReads.static_moved).
#   - everything else is whole: every recipe that reads the file, as before. [workspace.package] and .metadata,
#     `resolver`, [profile.*], [patch.*], [replace], a key this code does not know, a manifest that does not parse, a
#     file added or removed, a `workspace` that is not a table on both sides.
# A recipe that reads the file as a file is whole whatever moved (ManifestUse.whole). Today only a bin's box dep-info
# can say so: a script names a path under tools/ or crates/ (_SCRIPT_PATH) and a literal one under tools/, crates/ or
# docs/ (Tree._LITERAL), so a script that opens Cargo.toml by its bare name is not seen, here or anywhere in the mapping.
# The ledger key still hashes the file whole (KEY_GLOBALS): a recipe this selects has its key moved.
ROOT_MEMBERSHIP = ("members", "exclude")
ROOT_DEFAULT_MEMBERS = "default-members"


class Unparsed(dict):
    """A root manifest that does not parse: a side no table can be read from."""


UNPARSED = Unparsed()


def _str_list(v) -> bool:
    return isinstance(v, list) and all(isinstance(e, str) for e in v)


def root_moves(da: dict | None, db: dict | None) -> list[tuple[str, str]]:
    """What moved between two parses of the root manifest, one (label, kind) per table: `members` for [workspace]
    members and exclude, `default` for default-members, `dep` for each moved [workspace.dependencies] entry, `lints` for
    [workspace.lints], `whole` for every other table. A side with no file or one that does not parse is the whole file.
    Empty when no table moved (a comment, the layout)."""
    if da is None or db is None or isinstance(da, Unparsed) or isinstance(db, Unparsed):
        if da is None and db is None:
            return []
        what = "does not parse" if isinstance(da, Unparsed) or isinstance(db, Unparsed) else "added" if da is None else "removed"
        return [(f"the whole file ({what})", "whole")]
    out: list[tuple[str, str]] = []
    for k in sorted(set(da) | set(db)):
        x, y = da.get(k), db.get(k)
        if canon(x) == canon(y):
            continue
        if k != "workspace" or not isinstance(x, dict) or not isinstance(y, dict):
            out.append((f"[{k}]", "whole"))
            continue
        for s in sorted(set(x) | set(y)):
            u, v = x.get(s), y.get(s)
            if canon(u) == canon(v):
                continue
            listed = all(w is None or _str_list(w) for w in (u, v))
            tabled = all(w is None or isinstance(w, dict) for w in (u, v))
            if s in ROOT_MEMBERSHIP and listed:
                out.append((f"[workspace] {s}", "members"))
            elif s == ROOT_DEFAULT_MEMBERS and listed:
                out.append((f"[workspace] {s}", "default"))
            elif s == "dependencies" and tabled:
                ua, va = u or {}, v or {}
                out += [(f"[workspace.dependencies] {e}", "dep") for e in sorted(set(ua) | set(va)) if canon(ua.get(e)) != canon(va.get(e))]
            elif s == "lints" and tabled:
                out.append(("[workspace.lints]", "lints"))
            elif any(isinstance(w, dict) for w in (u, v)):
                out.append((f"[workspace.{s}]", "whole"))
            else:
                out.append((f"[workspace] {s}", "whole"))
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


# What `Tree.memo` keeps for the process: (root, key) -> (value, what the computation saw). The saw-map holds
# ("r", rel) -> (the file's text, its stat identity or None), ("f", rel) / ("d", rel) -> whether the path is a file /
# a directory, each as the run met it; a stored value is used again only while every entry still holds.
_MEMO: dict[tuple, tuple[object, dict[tuple[str, str], object]]] = {}
_TEXTS: dict[str, str] = {}  # one copy of each distinct file text, so entries that saw the same file share it


def _read_text(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def _read_seen(path: str) -> tuple[str, tuple | None]:
    """(text, the file's `HashCache.ident` after the read) — None for a file changed within `HashCache.RACY_NS`, whose
    next write may land inside the same timestamp: such a file is compared by its text, never by its stat."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
        st = os.fstat(fh.fileno())
    if max(st.st_mtime_ns, st.st_ctime_ns) > time.time_ns() - HashCache.RACY_NS:
        return text, None
    return text, HashCache.ident(st)


def _copy_module_tree(w: ModuleTree) -> ModuleTree:
    """`w` with its own containers: a cached walk is handed to each Tree as if it had walked itself, and a caller
    that edits what it got must not edit another Tree's."""
    return ModuleTree(set(w.files), list(w.literals), set(w.test_files), list(w.test_literals), dict(w.mods), dict(w.test_mods))


class Tree:
    """One checkout of the repository: its packages and the files each target reads."""

    def __init__(self, root: str, meta: dict | None = None):
        self.root = os.path.realpath(root)
        meta = meta if meta is not None else cargo_metadata(self.root)
        self.metadata = meta
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
                        target=d.get("target") or None,
                    )
                )
            targets = [
                TargetMeta(
                    kind=_kind_of(t["kind"]),
                    name=t["name"],
                    src=os.path.relpath(os.path.realpath(t["src_path"]), wroot),
                    required=list(t.get("required-features") or []),
                    proc_macro="proc-macro" in t["kind"] or "proc-macro" in t.get("crate_types", []),
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
        self._walk_cache: dict[tuple[str, frozenset[str] | None], ModuleTree] = {}
        self._reach_hashes = HashCache("")
        self._saw: list[dict[tuple[str, str], object]] = []  # the open `memo` runs, innermost last
        self.unresolved: list[str] = []
        self.depinfo: dict[str, tuple[set[str], float]] = {}  # bin name -> (files, mtime)

    # ---------------- file walk ----------------

    def exists(self, rel: str) -> bool:
        ok = os.path.isfile(os.path.join(self.root, rel))
        if self._saw:
            self._saw[-1].setdefault(("f", rel), ok)
        return ok

    def isdir(self, rel: str) -> bool:
        ok = os.path.isdir(os.path.join(self.root, rel))
        if self._saw:
            self._saw[-1].setdefault(("d", rel), ok)
        return ok

    def directory_files(self, rel: str) -> set[str]:
        """Every regular file under an explicit directory; unreadable entries are errors."""
        out: set[str] = set()
        seen: set[str] = set()

        def unlisted(error):
            raise error

        for base, dirs, files in os.walk(os.path.join(self.root, rel), followlinks=True, onerror=unlisted):
            real = os.path.realpath(base)
            if real in seen:
                dirs.clear()
                continue
            seen.add(real)
            for name in files:
                path = os.path.join(base, name)
                if stat.S_ISREG(os.stat(path).st_mode):
                    out.add(os.path.relpath(path, self.root))
        return out

    def read(self, rel: str) -> str:
        path = os.path.join(self.root, rel)
        if not self._saw or ("r", rel) in self._saw[-1]:
            return _read_text(path)
        text, ident = _read_seen(path)
        self._saw[-1][("r", rel)] = (_TEXTS.setdefault(text, text), ident)
        return text

    def memo(self, key: tuple, compute: Callable[[], object]) -> tuple[object, bool]:
        """(`compute()`, whether it came from an earlier run). `compute` is a function of `key` and of the files
        it looks at through `read`, `exists` and `isdir`; its value is kept for the process, shared by every Tree
        on this root, and used again while each of those files reads and each of those paths probes as it did
        in the run that made it: a hit costs one stat per file seen (a read of a file changed within two seconds
        of the run, as `HashCache` holds), never the parse. A file written between two Trees (a self-test case
        that edits a scratch copy) makes the entries that saw it miss, so no site clears anything. The value is
        shared: the caller copies what it hands on, and `compute` reads no Tree state but the files."""
        full = (self.root, *key)
        hit = _MEMO.get(full)
        if hit is not None and self._holds(hit[1]):
            self._note(hit[1])
            return hit[0], True
        saw: dict[tuple[str, str], object] = {}
        self._saw.append(saw)
        try:
            value = compute()
        finally:
            self._saw.pop()
        self._note(saw)
        _MEMO[full] = (value, saw)
        return value, False

    def _note(self, saw: dict[tuple[str, str], object]) -> None:
        """An enclosing `memo` run depends on everything a nested one saw."""
        if self._saw:
            for k, v in saw.items():
                self._saw[-1].setdefault(k, v)

    def _holds(self, saw: dict[tuple[str, str], object]) -> bool:
        for (kind, rel), was in saw.items():
            path = os.path.join(self.root, rel)
            if kind == "f":
                if os.path.isfile(path) != was:
                    return False
            elif kind == "d":
                if os.path.isdir(path) != was:
                    return False
            else:
                text, ident = was
                try:
                    if ident is not None and HashCache.ident(os.stat(path)) == ident:
                        continue
                    if _read_text(path) != text:
                        return False
                except OSError:
                    return False
        return True

    _MOD = re.compile(r"^(\s*)(?:pub(?:\s*\([^)]*\))?\s+)?mod\s+(?:r#)?([A-Za-z_][A-Za-z0-9_]*)\s*;")
    _INLINE = re.compile(r"^(\s*)(?:pub(?:\s*\([^)]*\))?\s+)?mod\s+(?:r#)?([A-Za-z_][A-Za-z0-9_]*)\s*\{\s*$")
    _PATH = re.compile(r"#\[\s*path\s*=\s*\"([^\"]+)\"\s*\]")
    _INCLUDE = re.compile(r"\binclude(?:_str|_bytes)?!\s*\(\s*\"([^\"]+)\"\s*\)")
    _LITERAL = re.compile(r"\"(/?(?:\.\./)*(?:tools|crates|docs)/[A-Za-z0-9_./-]+)\"")
    _CFG = re.compile(r"^#\[\s*cfg\s*\((.*)\)\s*\]$")

    @staticmethod
    def lead_attrs(s: str) -> tuple[list[str], str]:
        """The one-line outer attributes `s` opens with, each `#[…]` whole, and the item after them:
        `#[cfg(test)] mod x;` is `(["#[cfg(test)]"], "mod x;")`, `#[cfg(test)] // why` is
        `(["#[cfg(test)]"], "")`. A `/* … */` closed on the line between or after them is skipped,
        and a bracket inside a string literal (`\\"` escapes counted) does not count. An attribute
        whose `]` is not on the line, a raw string in one, or a `/*` left open is `([], s)`: the
        caller reads the line as before."""
        attrs: list[str] = []
        rest = s
        while rest.startswith("#["):
            depth, quoted, escaped, end = 0, False, False, None
            for i, ch in enumerate(rest):
                if quoted:
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == '"':
                        quoted = False
                elif ch == '"':
                    if i > 0 and rest[i - 1] in "r#":
                        return [], s
                    quoted = True
                elif ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end is None:
                return [], s
            attrs.append(rest[:end])
            rest = rest[end:].lstrip()
            while rest.startswith("/*") and "*/" in rest:
                rest = rest[rest.index("*/") + 2 :].lstrip()
            if rest.startswith("/*"):
                return [], s
        if rest.startswith("//"):
            rest = ""
        return (attrs, rest) if attrs else ([], s)

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

    @staticmethod
    def cfg_off(attr: str, features: frozenset[str] | None) -> bool:
        """Whether a one-line `#[cfg(…)]` is false in every build whose enabled features are `features`
        (`eval_cfg` with every target key and `test` unknown): a `feature = …` that is off, alone or
        under `all`, `any` and `not`. Anything it cannot decide — a target predicate, `test`, a
        multi-line attribute, no `features` — is not off: the item stays in."""
        if features is None:
            return False
        m = Tree._CFG.match(attr)
        return bool(m) and eval_cfg(parse_cfg(m.group(1)), None, {}, {}, features) is False

    def walk(self, src: str, features: frozenset[str] | None = None) -> ModuleTree:
        """The module tree rooted at `src`. A module declared under a test-only `cfg` (`cfg_test`),
        file or inline, and what it declares, includes and names go to the test side; a file reached
        both ways is on the build side. With `features` (the enabled features of the package the tree is
        compiled in), a module under a `cfg` they make false (`cfg_off`) is not walked: its file, and
        what it declares, includes and names, are not the build's."""
        acc = ModuleTree(set(), [], set(), [])
        self._walk(src, os.path.dirname(src), acc, False, (), features)
        acc.test_files -= acc.files
        for m in set(acc.test_mods) & set(acc.mods):
            del acc.test_mods[m]
        return acc

    def _walk(self, rel: str, moddir: str, acc: ModuleTree, test: bool, modpath: tuple[str, ...] = (), features: frozenset[str] | None = None) -> None:
        (acc.test_mods if test else acc.mods).setdefault(modpath, rel)
        if rel in acc.files or (test and rel in acc.test_files):
            return
        (acc.test_files if test else acc.files).add(rel)
        fdir = os.path.dirname(rel)
        pending_path: str | None = None
        pending_test = False  # a test-only `#[cfg]` since the last item: the next module is test-only
        pending_off = False  # a `#[cfg]` the features make false since the last item: the next module is not built
        inline: list[tuple[int, str, bool, bool]] = []  # (indent, name, test-only, not built)
        attr_depth = 0  # open brackets of a multi-line attribute (`#[allow(\n…\n)]`)
        for line in self.read(rel).split("\n"):
            s = line.strip()
            if not s or s.startswith("//"):
                continue
            indent = len(line) - len(line.lstrip())
            if attr_depth == 0 and s == "}" and inline and inline[-1][0] == indent:
                inline.pop()
                continue
            here = inline[-1][2] if inline else test
            if inline and inline[-1][3]:
                # inside an inline module the features switch off: only its end is read
                im = self._INLINE.match(line)
                if im:
                    inline.append((len(im.group(1)), im.group(2), here, True))
                continue
            for lit in self._LITERAL.findall(line):
                (acc.test_literals if here else acc.literals).append((rel, lit))
            for inc in self._INCLUDE.findall(line):
                p = os.path.normpath(os.path.join(fdir, inc))
                if self.exists(p):
                    (acc.test_files if here else acc.files).add(p)
                else:
                    self.unresolved.append(f"{rel}: include of {inc} ({p} not found)")
            if attr_depth > 0:
                attr_depth += s.count("[") - s.count("]")
                continue
            # `#[cfg(test)] mod x;`, `#[path = "x.rs"] mod x;`, `#[cfg(test)] #[path = "x.rs"]`: each
            # one-line attribute counts as if on a line of its own, then the item after them is read
            attrs, rest = self.lead_attrs(s)
            for a in attrs:
                am = self._PATH.fullmatch(a)
                if am:
                    pending_path = am.group(1)
                pending_test = pending_test or self.cfg_test(a)
                pending_off = pending_off or self.cfg_off(a, features)
            if attrs and not rest:
                continue
            if attrs:
                line, s = line[:indent] + rest, rest
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
                child_off = pending_off
                pending_path = None
                pending_test = False
                pending_off = False
                if child_off:
                    continue
                inner = [n for (i, n, _, _) in inline if i < indent]
                hit, child_dir, err = self.mod_file(rel, moddir, inner, name, path_attr)
                if hit is None:
                    self.unresolved.append(err)
                    continue
                self._walk(hit, child_dir, acc, child_test, modpath + tuple(inner) + (name,), features)
                continue
            if im:
                inner = [n for (i, n, _, _) in inline if i < indent]
                if not pending_off:
                    (acc.test_mods if here or pending_test else acc.mods).setdefault(modpath + tuple(inner) + (im.group(2),), rel)
                inline.append((len(im.group(1)), im.group(2), here or pending_test, pending_off))
                pending_path = None
                pending_test = False
                pending_off = False
                continue
            if s.startswith("#["):
                pending_test = pending_test or self.cfg_test(s)
                pending_off = pending_off or self.cfg_off(s, features)
                attr_depth = max(0, s.count("[") - s.count("]"))
                continue
            pending_path = None
            pending_test = False
            pending_off = False

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

    def _walk_shared(self, src: str, features: frozenset[str] | None) -> ModuleTree:
        """`walk`, from the process's memo when the files it read are as they were. The unresolved names the
        walk adds to this Tree are stored with it and added again on a hit."""

        def walked() -> tuple[ModuleTree, tuple[str, ...]]:
            n = len(self.unresolved)
            return self.walk(src, features), tuple(self.unresolved[n:])

        (w, unresolved), hit = self.memo(("walk", src, features), walked)
        if hit:
            self.unresolved.extend(unresolved)
        return _copy_module_tree(w)

    def tree_of(self, src: str, test: bool = False, features: frozenset[str] | None = None) -> tuple[set[str], list[tuple[str, str]]]:
        """`src`'s module tree as one build of it sees it: a test build (`test`) with its test-only
        modules, any other build without them; with `features`, without the modules those features
        switch off (`walk`)."""
        key = (src, features)
        if key not in self._walk_cache:
            self._walk_cache[key] = self._walk_shared(src, features)
        w = self._walk_cache[key]
        if test:
            return w.files | w.test_files, w.literals + w.test_literals
        return w.files, w.literals

    def mods_of(self, src: str, test: bool = False) -> dict[tuple[str, ...], str]:
        """`src`'s modules as `tree_of` sees them: module path below the root -> the file that holds it."""
        self.tree_of(src)
        w = self._walk_cache[(src, None)]
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

    def lib_files(self, name: str, features: frozenset[str] | None = None) -> dict[str, str]:
        """The files a dependent sees of package `name`: its lib tree, manifest and build script. `features`,
        when the build's enabled features of `name` are known, leave out the modules they switch off."""
        out: dict[str, str] = {self.packages[name].manifest: f"{name} Cargo.toml"}
        for f, why in self._lib_reads(name, features).items():
            out.setdefault(f, why)
        return out

    def _lib_reads(self, name: str, features: frozenset[str] | None = None) -> dict[str, str]:
        """lib_files but the manifest: what package `name`'s lib and build script read as files."""
        pkg = self.packages[name]
        out: dict[str, str] = {}
        lib = pkg.lib()
        if lib is not None:
            files, lits = self.tree_of(lib.src, features=features)
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
        """Workspace packages a target of `pkg` links: name -> its shortest dependency chain (breadth first)."""
        _, deps_on, _ = self.feature_closure(pkg, features, default, all_features)
        seen: dict[str, str] = {}
        queue: collections.deque[tuple[str, list[str], bool, str]] = collections.deque()
        for d in pkg.deps:
            if d.pkg not in self.packages:
                continue
            if d.kind == "dev" and not dev:
                continue
            if d.optional and d.key not in deps_on:
                continue
            queue.append((d.pkg, d.features, d.default, f"{pkg.name} -> {d.pkg}"))
        while queue:
            name, feats, dflt, why = queue.popleft()
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
                queue.append((d.pkg, d.features, d.default, f"{why} -> {d.pkg}"))
        return seen

    def target_files(self, pkg_name: str, kind: str, name: str | None, features: list[str], default: bool = True, all_features: bool = False, use: ManifestUse | None = None, by_features: bool = False) -> dict[str, str]:
        """file -> origin for one target. kind: lib bin test example bench doctest libtest. The
        target's own test-only modules count for every kind but `lib` and `doctest` (rustdoc sets no
        cfg(test)); for a bin or an example that is also its plain build, an over-selection confined to
        its own recipes. Every lib it links, its own package's included, is seen without them. `use`,
        when given, gains how the target reads each manifest of its closure: by table — its own entry in
        its package's, the shared tables of every other — or whole, when a file walk reaches one.
        `by_features`: the package's own trees (the target's and its lib's) leave out the modules the build's
        enabled features switch off (`#[cfg(feature = …)] mod x;`, `Tree.walk`); the caller says it only
        when `features` is the build's whole list. Another package's lib is read whole either way."""
        pkg = self.packages[pkg_name]
        out: dict[str, str] = {}
        built: frozenset[str] | None = None
        if by_features:
            on, deps_on, _ = self.feature_closure(pkg, features, default, all_features)
            built = frozenset(on | deps_on)
        dev = kind in ("test", "bench", "doctest", "libtest", "example")
        if kind in ("lib", "doctest", "libtest"):
            own = pkg.lib()
            label = {"lib": "lib", "doctest": "doctests", "libtest": "lib tests"}[kind]
        else:
            own = pkg.find(kind, name or "")
            label = f"{kind} {name}"
        if own is None:
            raise RecipeError(f"{pkg_name} has no {kind} target {name or ''}".rstrip())
        files, lits = self.tree_of(own.src, test=kind not in ("lib", "doctest"), features=built)
        for f in files:
            out[f] = f"{pkg_name} {label}"
        for f in self._resolve_literals(pkg, lits):
            out.setdefault(f, f"{pkg_name} {label} (path literal)")
        for f, why in self.lib_files(pkg_name, built).items():
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
            use.whole |= read & (self.manifest_paths | {ROOT_MANIFEST})
        return out

    # ---------------- invocation -> targets ----------------

    @staticmethod
    def depends_on(pkg: Package, name: str) -> bool:
        """Whether `pkg` names package `name` in a dependency table (any kind): the only other package whose
        `name/feature` word cargo takes for `-p pkg` alone."""
        return any(name in (d.key, d.pkg) for d in pkg.deps)

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
                    elif fp not in pkgs and not self.depends_on(pkg, fp):
                        errors.append(f"--features {f}: cargo takes `dep/feature` for a dependency of -p {pname} or a package of the same call, and {pname} depends on no {fp}")
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
    bare_cargo: bool = False  # a cargo line (or a dependency's) names no `-p` and no `--workspace`: [workspace] default-members decides it

    def match(self, path: str) -> str | None:
        if path in self.files:
            return self.files[path]
        for p, why in self.prefixes.items():
            if path.startswith(p):
                return why
        return None


def origin_label(pname: str, label: str, why: str) -> str:
    """How a recipe input is labelled: `label` alone (`test head_coverage`) when `why` is the target's own read — its
    module tree, or a path literal in it, which `Tree.target_files` words `<pkg> <label> …` — and `label <- why` when
    a lib the target links brought it in. A file and a directory go through this one builder, so a target's own read
    is its own whichever it names; `_own_origin` reads the shape back."""
    return label if why.startswith(f"{pname} {label}") else f"{label} <- {why}"


class Graph:
    def __init__(self, tree: Tree, recipes: dict[str, Recipe]):
        self.tree = tree
        self.recipes = recipes
        self._inputs: dict[str, RecipeInputs] = {}
        self._key_inputs: dict[str, RecipeInputs] = {}
        self._script_cache: dict[str, dict[str, str]] = {}

    def script_closure(self, rel: str) -> dict[str, str]:
        """Repository files a script names, transitively (comments skipped), and as `dir/` each tree
        directory a shell script walks with `find`."""
        if rel not in self._script_cache:
            closure, _ = self.tree.memo(("script", rel), lambda: self._script_closure(rel))
            self._script_cache[rel] = dict(closure)
        return self._script_cache[rel]

    def _script_closure(self, rel: str) -> dict[str, str]:
        out: dict[str, str] = {}
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

    def named_reads(self, rc: RecipeCommands) -> tuple[dict[str, str], list[str]]:
        """Script/explicit-path reads independently of Cargo's first-writer origin explanations."""
        reads: dict[str, str] = {}
        errors: list[str] = []
        for s in rc.scripts + rc.paths:
            if self.tree.exists(s):
                for f, why in self.script_closure(s).items():
                    reads.setdefault(f, why)
            elif self.tree.isdir(s):
                reads.setdefault(s.rstrip("/") + "/", "named directory")
            elif s in rc.scripts or s.endswith(_SCRIPT_EXT):
                errors.append(f"names {s}, which is not in the tree")
        return reads, errors

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
        bare = any(not inv.packages and not inv.workspace for inv in rc.invocations)
        for inv in rc.invocations:
            tl, errs = self.tree.resolve(inv)
            errors.extend(errs)
            for pname, kind, tname, feats in tl:
                targets.append((pname, kind, tname, feats))
                label = f"{kind} {tname}" if tname else kind
                # Read by the features only where they are the build's whole list for the package: one `-p`, no
                # recipe parameter (known at run time only), no `pkg/feature` (it reaches another package's
                # features, and a build with several packages unifies them); else the build is read whole.
                by_features = len(inv.packages) == 1 and not any("{{" in f or "/" in f for f in inv.features)
                for f, why in self.tree.target_files(pname, kind, tname, feats, not inv.no_default, inv.all_features, use, by_features).items():
                    (prefixes if f.endswith("/") else files).setdefault(f, origin_label(pname, label, why))
            for g in CARGO_GLOBALS:
                files.setdefault(g, "cargo (every build)")
            if inv.oxide:
                for g in OXIDE_GLOBALS:
                    files.setdefault(g, "cargo oxide")
        if rc.box:
            for g in BOX_GLOBALS:
                files.setdefault(g, "tools/box.sh (every box command)")
            try:
                for profile in sorted(rc.profiles()):
                    files.setdefault(f"tools/ref/models/{profile}.sh", f"profile {profile}")
            except RecipeError as err:
                errors.append(str(err))
            if any(m is not None and "{{" in m for m in rc.models):
                prefixes.setdefault("tools/ref/models/", "recipe parameter selects the profile")
        reads, read_errors = self.named_reads(rc)
        errors.extend(read_errors)
        for f, why in reads.items():
            (prefixes if f.endswith("/") else files).setdefault(f, why)
        use.whole |= reads.keys() & self.tree.manifest_paths
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
            bare = bare or di.bare_cargo
            built |= {t for (_, k, t, _) in di.targets if k == "bin"}
        for b in rc.runs:
            if "{{" in b or b.startswith("$"):
                continue
            if b not in built:
                errors.append(f"runs target/release/{b}, which the recipe does not build")
        # a manifest that is a cargo global, or lies under a directory a script walks, is read whole
        use.whole |= {m for m in self.tree.manifest_paths if m in CARGO_GLOBALS or any(m.startswith(p) for p in prefixes)}
        ri = RecipeInputs(files, prefixes, targets, errors, rc, use, bare)
        self._inputs[name] = ri
        return ri

    def key_inputs(self, name: str, _stack: tuple[str, ...] = ()) -> RecipeInputs:
        """Ledger inputs, not compilation/selection inputs: uncut item reach plus mandatory reads.

        Dependency recipes are closed separately: their targets are deliberately not in `inputs(name).targets`.
        Scripts, explicit paths, manifests, build-time code and dep-info retain their conservative contracts.
        `inputs` never calls this method; affected/narrow/release continue to use compilation ownership.
        """
        if name in _stack:
            raise RecipeError(f"recipe dependency cycle: {' -> '.join(_stack + (name,))}")
        reach = reach_of(self.tree)
        if reach._key_snapshot is not None:
            from bloomery import reach_cache

            if not reach_cache.current(reach):
                if _stack:
                    raise RecipeError("source tree changed during recipe dependency analysis")
                # A new request may follow an edit in a long-lived process. Forget derived state before
                # beginning that request; changes during the request itself are rejected by validate().
                self.tree._walk_cache.clear()
                self.tree.unresolved.clear()
                self._inputs.clear()
                self._key_inputs.clear()
                self._script_cache.clear()
                reach = self.tree._reach = Reach(self.tree)
            elif name in self._key_inputs:
                return self._key_inputs[name]
        reach.prepare()
        broad = self.inputs(name)
        reach.validate()
        keep = reach.key_files(self, name)
        files = {f: why for f, why in broad.files.items() if f in keep or not _linked_origin(why)}
        prefixes = {f: why for f, why in broad.prefixes.items() if f in keep or not _linked_origin(why)}
        if name in reach._key_wide:
            files = {f: f"{why} (conservative: opaque generated code, macro expansion or foreign linkage)" if _linked_origin(why) else why
                     for f, why in files.items()}
            prefixes = {f: f"{why} (conservative: opaque generated code, macro expansion or foreign linkage)" if _linked_origin(why) else why
                        for f, why in prefixes.items()}
        # Origins explain the first read only: Cargo may have claimed a file also read by cat/a script.
        # Re-union the independent read closure rather than treating that explanation as ownership.
        reads, _ = self.named_reads(broad.commands)
        for f, why in reads.items():
            (prefixes if f.endswith("/") else files).setdefault(f, why)
        for f in keep:
            if f.endswith("/"):
                prefixes.setdefault(f, "reached directory / mandatory compilation input")
            elif self.tree.exists(f):
                files.setdefault(f, "reached code / mandatory compilation input")
        use = ManifestUse()
        use.merge(broad.manifests)
        errors = list(broad.errors)
        for dep in self.recipes[name].deps:
            if dep not in self.recipes:
                continue
            di = self.key_inputs(dep, _stack + (name,))
            for f, why in di.files.items():
                files.setdefault(f, f"{why} (via {dep})")
            for p, why in di.prefixes.items():
                prefixes.setdefault(p, f"{why} (via {dep})")
            use.merge(di.manifests)
            errors.extend(f"{dep}: {error}" for error in di.errors)
        ri = RecipeInputs(files, prefixes, list(broad.targets), errors, broad.commands, use, broad.bare_cargo)
        self._key_inputs[name] = ri
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
    problems += bin_problems(tree, recipes)
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
    except FileNotFoundError as err:
        raise RecipeError(f"reading dep-info over ssh {host} failed: ssh client not found on PATH; --no-box skips it") from err
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


def archive(rev: str, into: str, cwd: str = ROOT) -> str:
    """REV's tree extracted into `into`. The archive goes through a file, not a pipe: a tar that stops at the end-of-archive marker
    (bsdtar, the Mac's) exits while git is still writing the zero padding of its last block, and git dies of SIGPIPE on an archive
    tar read whole."""
    os.makedirs(into, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="archive-") as tmp:
        tarball = os.path.join(tmp, "tree.tar")
        arch = subprocess.run(["git", "archive", "--format=tar", "-o", tarball, rev], cwd=cwd, capture_output=True)
        if arch.returncode != 0:
            raise RecipeError(f"git archive {rev} failed (git {arch.returncode}): {arch.stderr.decode(errors='replace')[-300:]}")
        tar = subprocess.run(["tar", "-x", "-f", tarball, "-C", into], capture_output=True)
        if tar.returncode != 0:
            raise RecipeError(f"extracting git archive {rev} failed (git 0, tar {tar.returncode}): {tar.stderr.decode(errors='replace')[-300:]}")
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
    of them (manifest_view); the lever registry's rows (`LeverReads`, `levers`). One reader for every place a
    changed file picks recipes (select, narrow); the key reads the same manifest view. `hit` records each answer, which `lines` prints. The root Cargo.toml of a virtual
    workspace is read here too (`root`), by the tables `root_moves` names; a root that is a package's own
    manifest is a package manifest and a cargo global, whole. A recipe sees the `[target.<key>]` tables of its host
    (`host`: the box, or the Mac for a recipe that runs MAC_GATE), and Cargo.lock by the packages its build reaches
    (`lock`, a `LockDiff`: `outside` is true for a recipe whose closure on its host holds none of the packages the diff
    moved). A `[lints]` table moves no recipe (`static_moved` names it). A Rust file whose diff is comments or layout only
    selects none (`comments`, a `CommentReads`). `notes` holds one line for each
    `[target.<key>]` table the box's view leaves out or cannot decide, one for a Cargo.lock the reader keeps whole and
    one for a comment-only file left whole (`select` hands them to `affected`)."""

    def __init__(self, changed: list[str], a: Side | None, b: Side, comment_only: bool = True):
        self.a, self.b = a, b
        self.docs: dict[str, tuple[dict | None, dict | None]] = {}
        self.root: str | None = None
        self._delta: tuple[dict[str, str], dict[str, str]] | None = None
        self._root_kinds: list[tuple[str, str]] | None = None
        self._inheritors: dict[tuple[str, str | None], set[str] | None] = {}
        self._roots: dict[str, frozenset[str] | None] = {}
        for f in changed:
            if a is not None and (f in b.tree.manifest_paths or f in a.tree.manifest_paths):
                self.docs[f] = (self._doc(a, f), self._doc(b, f))
            elif a is not None and f == ROOT_MANIFEST:
                self.root = f
                self.docs[f] = (self._root_doc(a), self._root_doc(b))
        self.seen: dict[str, dict[str, list[tuple[str, bool]] | None]] = {f: {} for f in self.docs}
        self.levers = LeverReads(changed, a, b)  # the lever registry, when it is among the changed files and its diff is rows
        self.comments = CommentReads(changed, a, b, comment_only)  # the Rust files whose diff is comments or layout
        self.notes: list[str] = list(self.comments.notes)
        if self.levers.why_whole is not None:
            self.notes.append(self.levers.summary())
        self.static_moved: list[str] = []  # the lint tables that moved: held by the static tier, selecting no recipe
        for f, (da, db) in self.docs.items():
            if f != self.root:
                self.notes += target_notes(f, da, db)
                if lints_moved(da, db):
                    self.static_moved.append(f"{f} [lints]")
        if self.root is not None and any(k == "lints" for _, k in self.root_kinds()):
            self.static_moved.append(f"{ROOT_MANIFEST} [workspace.lints]")
        self.lock: LockDiff | None = None  # Cargo.lock read by package; None when it is read whole
        if a is not None and LOCK_FILE in changed:
            self._read_lock(a, b)

    def _read_lock(self, a: Side, b: Side) -> None:
        """Parse both locks; a side the reader cannot take apart leaves `lock` None, so every cargo recipe reads the file whole."""
        if not (a.tree.exists(LOCK_FILE) and b.tree.exists(LOCK_FILE)):
            self.notes.append(f"{LOCK_FILE}: {'added' if not a.tree.exists(LOCK_FILE) else 'removed'}; read whole by every cargo recipe")
            return
        try:
            diff = LockDiff(a, b)
        except RecipeError as err:
            self.notes.append(f"{LOCK_FILE}: not read by package ({err}); read whole by every cargo recipe")
            return
        if diff.rest_moved:
            self.notes.append(f"{LOCK_FILE}: a top-level key beside `package` moved; read whole by every cargo recipe")
            return
        if not diff.moved:
            self.notes.append(f"{LOCK_FILE}: the diff names no package the parse sees (a comment or the layout); read whole by every cargo recipe")
            return
        self.lock = diff

    def host(self, n: str) -> str:
        """Where recipe `n` builds, read on both sides: the Mac when either side's recipe runs MAC_GATE (`recipe_host`)."""
        return "mac" if any(s is not None and recipe_host(s.recipes, n) == "mac" for s in (self.a, self.b)) else "box"

    @staticmethod
    def _feature_param(side: Side, n: str, _seen: frozenset[str] = frozenset()) -> bool:
        """Whether recipe `n` or a recipe it depends on builds with features a recipe parameter names: the closure of such
        a build is known at run time only."""
        if n in _seen or n not in side.recipes:
            return False
        if any("{{" in f for inv in side.graph.inputs(n).commands.invocations for f in inv.features):
            return True
        return any(TableReads._feature_param(side, d, _seen | {n}) for d in side.recipes[n].deps)

    def lock_roots(self, n: str) -> frozenset[str] | None:
        """The workspace packages recipe `n` builds or links on either side: the packages of its manifest views. None when
        it names none, its inputs have errors or its features come from a recipe parameter (a build the walk may not have
        seen whole): the file is then read whole."""
        if n not in self._roots:
            out: set[str] | None = set()
            for s in (self.a, self.b):
                if s is None or n not in s.recipes:
                    continue
                ri = s.graph.inputs(n)
                if ri.errors or self._feature_param(s, n):
                    out = None
                    break
                by_manifest = {p.manifest: p.name for p in s.tree.packages.values()}
                out |= {by_manifest[m] for m in ri.manifests.views if m in by_manifest}
            self._roots[n] = frozenset(out) if out else None
        return self._roots[n]

    def outside(self, n: str, f: str) -> bool:
        """Whether changed file `f` reaches nothing recipe `n` builds on its host: a Cargo.lock whose moved packages the
        recipe's closure on its host does not hold."""
        if f != LOCK_FILE or self.lock is None:
            return False
        roots = self.lock_roots(n)
        return roots is not None and not self.lock.differs(self.host(n), roots)

    def lock_hint(self) -> str:
        """Why no recipe read the changed Cargo.lock, for `unmapped`."""
        assert self.lock is not None
        if self.a is not None and "box" in lock_outside(self.a, self.b, self.lock):
            return "the packages it changes are not in the x86_64-linux build (notes:)"
        return "no recipe's build reaches a package that moved (notes:)"

    def lock_notes(self, names: list[str]) -> list[str]:
        """The line for a Cargo.lock read by package: the packages that moved and how many of the recipes `names` that read
        the file it selects. None when it selects every one of them."""
        if self.lock is None or self.b is None:
            return []
        readers = [n for n in names if n in self.b.recipes and self.b.graph.inputs(n).match(LOCK_FILE) is not None]
        chosen = [n for n in readers if not self.outside(n, LOCK_FILE)]
        if len(chosen) == len(readers):
            return []
        shown = ", ".join(self.lock.moved[:8]) + (f" and {len(self.lock.moved) - 8} more" if len(self.lock.moved) > 8 else "")
        if not chosen:
            out = lock_outside(self.a, self.b, self.lock) if self.a is not None else {}
            return [out["box"] if "box" in out else f"{LOCK_FILE}: {shown} changed; no recipe's build reaches a package that moved"]
        return [f"{LOCK_FILE}: {shown} changed; {len(chosen)} of {len(readers)} recipes' builds reach a package that moved"]

    @staticmethod
    def _doc(side: Side, f: str) -> dict | None:
        return read_manifest(side.tree.root, f) if side.tree.exists(f) else None

    @staticmethod
    def _root_doc(side: Side) -> dict | None:
        """The root manifest of `side`: a parse failure is a side no table reads (every cargo recipe reads the file
        whole), not the error a package manifest's is."""
        if not side.tree.exists(ROOT_MANIFEST):
            return None
        try:
            return read_manifest(side.tree.root, ROOT_MANIFEST)
        except RecipeError:
            return UNPARSED

    def delta(self) -> tuple[dict[str, str], dict[str, str]]:
        """(entered, left): manifest -> package name for each package that is a workspace member on one side only. The
        members are the ones each side's `cargo metadata` names (Tree.packages): cargo's own answer to members,
        exclude, globs and path dependencies, so a path an exclude names where no package lived is no change."""
        if self._delta is None:
            assert self.a is not None
            pa = {p.manifest: p.name for p in self.a.tree.packages.values()}
            pb = {p.manifest: p.name for p in self.b.tree.packages.values()}
            self._delta = ({m: x for m, x in pb.items() if m not in pa}, {m: x for m, x in pa.items() if m not in pb})
        return self._delta

    def _reach(self, n: str) -> list[str]:
        """The packages that entered (`+name`, read on B) or left (`-name`, read on A) the workspace that recipe `n`
        builds or reads: its own, and those it links."""
        assert self.a is not None
        entered, left = self.delta()
        out = []
        if n in self.b.recipes:
            ib = self.b.graph.inputs(n)
            out += [f"+{x}" for m, x in entered.items() if ib.match(m) is not None]
        if n in self.a.recipes:
            ia = self.a.graph.inputs(n)
            out += [f"-{x}" for m, x in left.items() if ia.match(m) is not None]
        return out

    def inheritors(self, name: str, host: str | None) -> set[str] | None:
        """The manifests of the packages that inherit root entry `name` (`name = { workspace = true }`) on either side, in
        a table a build on `host` reads (None: in any table). None when a manifest does not parse."""
        key = (name, host)
        if key not in self._inheritors:
            out: set[str] | None = set()
            for s in (self.a, self.b):
                if s is None:
                    continue
                for p in s.tree.packages.values():
                    try:
                        doc = read_manifest(s.tree.root, p.manifest)
                    except RecipeError:
                        out = None
                        break
                    if out is not None and inherits(doc, name, host):
                        out.add(p.manifest)
                if out is None:
                    break
            self._inheritors[key] = out
        return self._inheritors[key]

    def root_kinds(self) -> list[tuple[str, str]]:
        """`root_moves` of the root manifest with each moved `[workspace.dependencies]` entry settled: an entry that no
        package of either side inherits, or that a manifest which does not parse may, is read whole."""
        if self._root_kinds is None:
            assert self.root is not None
            out: list[tuple[str, str]] = []
            for label, kind in root_moves(*self.docs[self.root]):
                if kind == "dep":
                    names = self.inheritors(label.split("] ", 1)[1], None)
                    if names is None or not names:
                        why = "a manifest does not parse" if names is None else "no package inherits it"
                        self.notes.append(f"{ROOT_MANIFEST}: {label} changed and {why}; read whole by every cargo recipe")
                        kind = "whole"
                out.append((label, kind))
            self._root_kinds = out
        return self._root_kinds

    def _closure(self, n: str) -> dict[str, str]:
        """The manifest -> package name of every package recipe `n` builds or links, on either side."""
        out: dict[str, str] = {}
        for s in (self.a, self.b):
            if s is None or n not in s.recipes:
                continue
            by_manifest = {p.manifest: p.name for p in s.tree.packages.values()}
            out.update({m: by_manifest[m] for m in s.graph.inputs(n).manifests.views if m in by_manifest})
        return out

    def _root_moves(self, n: str, f: str) -> list[tuple[str, bool]] | None:
        """moves() for the root manifest: None (read as a file) when a table that stays whole moved or the recipe
        reads the file as a file; else the membership, default-members and `[workspace.dependencies]` moves that reach `n`:
        an entry reaches the recipes whose closure holds a package that inherits it in a table its host reads. A lint
        table reaches none."""
        sides = [s for s in (self.a, self.b) if s is not None and n in s.recipes]
        reads = [s for s in sides if s.graph.inputs(n).match(f) is not None]
        if not reads:
            return None
        kinds = self.root_kinds()
        if any(k == "whole" for _, k in kinds) or any(f in s.graph.inputs(n).manifests.whole for s in reads):
            self.seen[f][n] = None
            return None
        reach = self._reach(n)
        bare = any(s.graph.inputs(n).bare_cargo for s in reads)
        mv: list[tuple[str, bool]] = []
        for label, kind in kinds:
            if kind == "members" and reach:
                mv.append((f"{label} ({', '.join(reach)})", False))
            elif kind == "default" and bare:
                mv.append((label, False))
            elif kind == "dep":
                held = self.inheritors(label.split("] ", 1)[1], self.host(n)) or set()
                who = sorted(name for m, name in self._closure(n).items() if m in held)
                if who:
                    mv.append((f"{label} ({', '.join(who[:4])}{f', +{len(who) - 4}' if len(who) > 4 else ''})", False))
        self.seen[f][n] = mv
        return mv

    def moved(self, f: str) -> list[tuple[str, bool]]:
        """The tables of manifest `f` that moved in the box's view whatever recipe reads it, (label, whether a named entry of a target)."""
        da, db = self.docs[f]
        if f == self.root:
            return [(label, False) for label, kind in self.root_kinds() if kind != "lints"]
        return view_moves(da, db, os.path.dirname(f), None)

    def moves(self, n: str, f: str) -> list[tuple[str, bool]] | None:
        """What moved in recipe `n`'s view of manifest `f`, its own entries taken from both sides (an entry
        renamed, or a target a required-features edit lets in or out of a `--bins`, is in one side's list);
        None when `f` is not a manifest of the diff or `n` reads it whole on either side (the file rule)."""
        if f not in self.docs:
            return None
        if f == self.root:
            return self._root_moves(n, f)
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
        mv = view_moves(da, db, os.path.dirname(f), frozenset(wants), self.host(n))
        self.seen[f][n] = mv
        return mv

    def hit(self, n: str, f: str, side: Side) -> tuple[str | None, bool | None]:
        """(why recipe `n` reads changed file `f` on `side`, with what moved in its view — None when it does
        not read it or its view did not move; whether every move is an entry of its own targets — None when
        `f` is read as a file). A manifest of the diff is read on A too: a recipe whose B build no longer
        reaches it (a renamed entry, a target a required-features edit leaves out) read it before. The lever registry whose
        diff is rows only is read by row (`LeverReads.hit`): its answer is a recipe's own."""
        if self.levers.reads(f):
            return self.levers.hit(n, f, side)
        why = side.graph.inputs(n).match(f)
        if why is None and f in self.docs and self.a is not None and n in self.a.recipes:
            why = self.a.graph.inputs(n).match(f)
        if why is None or self.outside(n, f):
            return None, None
        if f == LOCK_FILE and self.lock is not None:
            roots = self.lock_roots(n)
            moved = self.lock.differs(self.host(n), roots) if roots else []
            return (f"{why}: {', '.join(moved[:3])}{f', +{len(moved) - 3}' if len(moved) > 3 else ''}" if moved else why), None
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
            root = f == self.root
            kinds = self.root_kinds() if root else [(label, "table") for label, _ in self.moved(f)]
            if not root:  # a table only a Mac recipe's view moves, then the tables the box's view leaves out: named, selecting no box recipe
                have = {label for label, _ in kinds}
                kinds += [(m, "table") for m in sorted({m for mv in by_table.values() for m, _ in mv} - have)]
                kinds += [(f"[target.{k}] (not in the x86_64-linux build)", "table") for k, v in target_moves(da, db) if v is False]
            if not root and lints_moved(da, db):
                kinds.append(("[lints]", "lints"))
            for label, kind in kinds or [("nothing (comments or layout only)", "table")]:
                if kind == "whole":  # a root table that stays whole: every recipe that reads the file
                    sel, pool = whole, whole
                else:
                    pool = list(by_table)
                    sel = [n for n, mv in by_table.items() if any(m == label or m.startswith(label + " (") for m, _ in mv)]
                every = len(sel) == len(pool) and len(sel) > 3
                held = f" (held by the static tier: {STATIC_TIER})" if kind == "lints" else ""
                out.append(f"    {label} -> " + (f"every reader ({len(sel)})" if every else " ".join(sel) if sel else "no recipe") + held)
            if root and any(k == "members" for _, k in kinds):
                entered, left = self.delta()
                assert self.a is not None
                out.append(f"    workspace members: {len(self.a.tree.packages)} before, {len(self.b.tree.packages)} after; "
                           f"entered {', '.join(sorted(entered.values())) or 'none'}, left {', '.join(sorted(left.values())) or 'none'}")
            if whole and not any(k == "whole" for _, k in kinds):
                out.append("    read whole -> " + " ".join(whole))
        return out


# ----------------------------------------------------------------------------------------------
# lever rows
# ----------------------------------------------------------------------------------------------
#
# Every bin links bloomery-levers, so the crate graph reads a one-row edit of its registry as a change to every gate. A row is
# data: a name parsed only when set, whose readers name its const or its variable. `LeverReads` takes the registry apart on both
# sides of a diff, and when the rows are all that moved it selects the recipe that tests the registry and the recipes whose
# inputs hold a file naming a moved row; any text the reader cannot account for as rows leaves the file whole.

LEVER_REGISTRY = "crates/levers/src/registry.rs"
LEVER_CRATE = "crates/levers/"  # the registry crate's own files are read by its own gate, never as a reader of a row
LEVER_PREFIX = "BLOOMERY_"


class LeverDoubt(Exception):
    """The registry is not text the row reader can take apart; the message is why the file stays whole."""


_RS_TOKEN = re.compile(
    r"""(?P<ws>\s+)|(?P<lc>//[^\n]*)|(?P<bc>/\*)|(?P<str>[bc]?"(?:[^"\\]|\\.)*")|(?P<raw>[bc]?r(?P<h>\#*)")"""
    r"""|(?P<chr>b?'(?:\\.[^']*|[^\\'])')|(?P<life>'[A-Za-z_]\w*)|(?P<id>[A-Za-z_]\w*)|(?P<num>\d[\w.]*)|(?P<p>.)""",
    re.S,
)


def _rs_tokens(text: str, comments: list[tuple[int, str]] | None = None) -> list[tuple[str, str]]:
    """(kind, text) of each token of Rust source, comments and whitespace left out; a string keeps its quotes and its bytes.
    An unterminated string or comment is a doubt: the registry is never read past it. `comments`, when given, receives each
    comment as (the number of tokens before it, its text)."""
    out: list[tuple[str, str]] = []
    i, n = 0, len(text)
    while i < n:
        m = _RS_TOKEN.match(text, i)
        assert m is not None  # `p` matches any character
        kind = m.lastgroup
        if kind in ("ws", "lc"):
            if kind == "lc" and comments is not None:
                comments.append((len(out), m.group()))
            i = m.end()
            continue
        if kind == "bc":
            depth, j = 1, i + 2
            while j < n and depth:
                if text.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif text.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            if depth:
                raise LeverDoubt("a block comment never ends")
            if comments is not None:
                comments.append((len(out), text[i:j]))
            i = j
            continue
        if kind == "raw":
            close = '"' + m.group("h")
            j = text.find(close, m.end())
            if j < 0:
                raise LeverDoubt("a raw string never ends")
            out.append(("str", text[i:j + len(close)]))
            i = j + len(close)
            continue
        if kind == "p" and text[i] == '"':
            raise LeverDoubt("a string never ends")
        out.append((kind or "p", m.group()))
        i = m.end()
    return out


@dataclass
class LeverRows:
    """One side of the registry: its `&str` consts (identifier -> the string, None when it has an escape), the tokens that are not
    rows (the code before and after the array, those consts left out), and each row's text keyed by the variable it names, the
    consts a row spells written out."""

    consts: dict[str, str | None]
    rest: list[str]
    order: list[str]
    rows: dict[str, str]


def _const_runs(toks: list[tuple[str, str]]) -> tuple[dict[str, str | None], list[tuple[str, str]]]:
    """The `[pub[(crate)]] const NAME: &['static] str = "...";` items of `toks` and the tokens left."""
    words = [t for _, t in toks]
    consts: dict[str, str | None] = {}
    drop: set[int] = set()
    for c, w in enumerate(words):
        if w != "const" or toks[c][0] != "id" or c + 3 >= len(toks):
            continue
        j = c + 1
        if toks[j][0] != "id" or words[j + 1:j + 3] != [":", "&"]:
            continue
        name, j = words[j], j + 3
        if j < len(toks) and toks[j][0] == "life":
            j += 1
        if words[j:j + 2] != ["str", "="] or j + 3 >= len(toks) or toks[j + 2][0] != "str" or words[j + 3] != ";":
            continue
        if name in consts:
            raise LeverDoubt(f"the const {name} is defined twice")
        lit = words[j + 2]
        consts[name] = lit[1:-1] if lit.startswith('"') and "\\" not in lit else None
        start = c
        if words[max(0, c - 4):c] == ["pub", "(", "crate", ")"]:
            start = c - 4
        elif c and words[c - 1] == "pub":
            start = c - 1
        drop.update(range(start, j + 4))
    return consts, [t for i, t in enumerate(toks) if i not in drop]


def parse_lever_rows(text: str) -> LeverRows:
    """The registry's `REGISTRY` array as rows, or a LeverDoubt naming what it could not read. A row is a top-level element of the
    array, `LeverSpec { name: <name>, … }` or `<fn>(<name>, …)`, its name a string literal or a `&str` const of the file."""
    toks = _rs_tokens(text)
    words = [t for _, t in toks]
    at = next((i for i in range(len(words) - 1) if words[i] == "static" and words[i + 1] == "REGISTRY"), None)
    if at is None:
        raise LeverDoubt("no `static REGISTRY`")
    if words[at + 2:at + 10] != [":", "&", "[", "LeverSpec", "]", "=", "&", "["]:
        raise LeverDoubt("the declaration of REGISTRY is not `&[LeverSpec] = &[`")
    head = at - 4 if words[at - 4:at] == ["pub", "(", "crate", ")"] else at - 1 if words[at - 1:at] == ["pub"] else at
    open_, depth, close = at + 9, 0, -1
    for i in range(open_, len(toks)):
        if toks[i][0] != "p":
            continue
        if words[i] in "([{":
            depth += 1
        elif words[i] in ")]}":
            depth -= 1
            if depth == 0:
                close = i
                break
    if close < 0 or words[close] != "]" or words[close + 1:close + 2] != [";"]:
        raise LeverDoubt("the array of REGISTRY does not close with `];`")
    consts, outside = _const_runs(toks[:head] + [("p", "REGISTRY")] + toks[close + 2:])
    spelled = {name: ("str", f'"{v}"') for name, v in consts.items() if v is not None}
    elems: list[list[tuple[str, str]]] = [[]]
    depth = 0
    for kind, w in toks[open_ + 1:close]:
        if kind == "p" and w in "([{":
            depth += 1
        elif kind == "p" and w in ")]}":
            depth -= 1
        if kind == "p" and w == "," and depth == 0:
            elems.append([])
        else:
            elems[-1].append((kind, w))
    if not elems[-1]:
        elems.pop()
    order: list[str] = []
    rows: dict[str, str] = {}
    for elem in elems:
        if not elem:
            raise LeverDoubt("an empty element between two commas")
        e = [spelled.get(w, (k, w)) if k == "id" else (k, w) for k, w in elem]
        w = [t for _, t in e]
        if len(e) > 5 and w[:4] == ["LeverSpec", "{", "name", ":"] and e[4][0] == "str" and w[5] == ",":
            lit = w[4]
        elif len(e) > 3 and e[0][0] == "id" and w[0][0].islower() and w[1] == "(" and e[2][0] == "str" and w[3] == ",":
            lit = w[2]
        else:
            raise LeverDoubt(f"a row the reader cannot name: `{' '.join(w[:6])}`")
        if not lit.startswith('"') or "\\" in lit:
            raise LeverDoubt(f"a row whose name is not a plain string: `{' '.join(w[:6])}`")
        name = lit[1:-1]
        if name in rows:
            raise LeverDoubt(f"{name} names two rows")
        order.append(name)
        rows[name] = " ".join(w)
    return LeverRows(consts, [t for _, t in outside], order, rows)


@dataclass(frozen=True)
class LeverSubject:
    """What moved in the registry that a reader can name: a row (added, changed, removed) or a variable const, with the words
    that name it in a file — its variable and every const identifier that spells it."""

    kind: str  # added | changed | removed | const
    label: str
    words: frozenset[str]


def diff_lever_rows(a: LeverRows, b: LeverRows) -> list[LeverSubject]:
    """What moved from `a` to `b`, or a LeverDoubt when anything but rows did: code outside the array (an item, a helper's body, a
    non-string const), a different order of the rows both sides hold."""
    if a.rest != b.rest:
        i = next((i for i in range(min(len(a.rest), len(b.rest))) if a.rest[i] != b.rest[i]), min(len(a.rest), len(b.rest)))
        raise LeverDoubt(f"code outside the rows moved near `{' '.join((b.rest or a.rest)[max(0, i - 4):i + 2])}`")
    if [k for k in a.order if k in b.rows] != [k for k in b.order if k in a.rows]:
        raise LeverDoubt("the rows both sides hold changed their order")
    spelled: dict[str, set[str]] = {}
    for consts in (a.consts, b.consts):
        for ident, v in consts.items():
            if v is not None:
                spelled.setdefault(v, set()).add(ident)
    out = [LeverSubject("removed", k, frozenset({k} | spelled.get(k, set()))) for k in a.order if k not in b.rows]
    for k in b.order:
        if k not in a.rows:
            out.append(LeverSubject("added", k, frozenset({k} | spelled.get(k, set()))))
        elif a.rows[k] != b.rows[k]:
            out.append(LeverSubject("changed", k, frozenset({k} | spelled.get(k, set()))))
    named = {w for subj in out for w in subj.words}
    for ident in sorted(set(a.consts) | set(b.consts)):
        old, new = a.consts.get(ident), b.consts.get(ident)
        vals = {v for v in (old, new) if v and v.startswith(LEVER_PREFIX)}
        if old != new and vals and ident not in named:
            out.append(LeverSubject("const", ident, frozenset({ident} | vals)))
    return out


class LeverReads:
    """The lever registry of one diff, read by row (`TableReads` holds it; select and narrow ask it through `TableReads.hit`).
    When every edit to crates/levers/src/registry.rs is a row (parse_lever_rows on both sides, diff_lever_rows), a recipe is
    selected by the file when it tests the registry as its own target, or when its inputs hold a file outside the registry's
    crate that names a moved row (its variable, or a const that spells it). Anything else is `why_whole`, and the file selects as
    any source file does."""

    def __init__(self, changed: list[str], a: Side | None, b: Side):
        self.subjects: list[LeverSubject] = []
        self.why_whole: str | None = None
        self.by_rows = False
        self.own: set[str] = set()  # the recipes that test the registry
        self.readers: dict[str, dict[str, set[str]]] = {}  # label -> reader file -> recipes whose inputs hold it
        self._named: dict[tuple[str, str], tuple[LeverSubject, ...]] = {}  # (tree root, file) -> the subjects it names
        self._dirs: dict[tuple[str, str], set[str]] = {}
        self._needle: re.Pattern[bytes] | None = None
        if LEVER_REGISTRY not in changed or a is None:
            return
        try:
            if not (a.tree.exists(LEVER_REGISTRY) and b.tree.exists(LEVER_REGISTRY)):
                raise LeverDoubt("the file is added or removed")
            self.subjects = diff_lever_rows(parse_lever_rows(a.tree.read(LEVER_REGISTRY)), parse_lever_rows(b.tree.read(LEVER_REGISTRY)))
            self.by_rows = True
        except LeverDoubt as err:
            self.why_whole = str(err)
        if self.subjects:
            words = sorted({w for subj in self.subjects for w in subj.words}, key=len, reverse=True)
            self._needle = re.compile(rb"\b(?:" + b"|".join(re.escape(w.encode()) for w in words) + rb")\b")

    def reads(self, f: str) -> bool:
        """Whether `f` is the registry and its diff is rows only."""
        return self.by_rows and f == LEVER_REGISTRY

    def _candidates(self, side: Side, ri: RecipeInputs) -> list[str]:
        """The files outside the registry's crate that recipe inputs `ri` hold, a directory it reads walked."""
        files = set(ri.files)
        for p in ri.prefixes:
            key = (side.tree.root, p)
            if key not in self._dirs:
                self._dirs[key] = side.tree.directory_files(p.rstrip("/"))
            files |= self._dirs[key]
        return sorted(f for f in files if not f.startswith(LEVER_CRATE))

    def _names(self, side: Side, rel: str) -> tuple[LeverSubject, ...]:
        """The moved subjects file `rel` of `side` names; a name a recipe lists but the tree lacks (a cargo global) names none."""
        key = (side.tree.root, rel)
        if key not in self._named:
            assert self._needle is not None
            try:
                with open(os.path.join(side.tree.root, rel), "rb") as fh:
                    found = {m.decode() for m in self._needle.findall(fh.read())}
            except FileNotFoundError:
                found = set()
            self._named[key] = tuple(subj for subj in self.subjects if subj.words & found)
        return self._named[key]

    @staticmethod
    def _recipe_chain(side: Side, n: str) -> list[str]:
        """Recipe `n` and every recipe it depends on, each once."""
        out: list[str] = []
        stack = [n]
        while stack:
            r = stack.pop()
            if r in out or r not in side.recipes:
                continue
            out.append(r)
            stack.extend(side.recipes[r].deps)
        return out

    def hit(self, n: str, f: str, side: Side) -> tuple[str | None, bool | None]:
        """(why recipe `n` reads the registry's rows on `side`, True — a row's readers are the recipe's own path); (None, None) when
        it does not. The recipe that tests the registry keeps its own origin. A reader the diff removed, or one it moved off a
        row, is itself a changed file and selects by its own."""
        ri = side.graph.inputs(n)
        why = ri.match(f)
        if why is not None and _own_origin(why):
            self.own.add(n)
            return why, True
        found: dict[str, str] = {}
        for rel in self._candidates(side, ri) if self.subjects else []:
            for subj in self._names(side, rel):
                found.setdefault(subj.label, rel)
                self.readers.setdefault(subj.label, {}).setdefault(rel, set()).add(n)
        # A recipe's own command lines set levers too (`BLOOMERY_BOX_ENV='BLOOMERY_X=…'`): its text and its deps' are readers.
        for r in self._recipe_chain(side, n) if self.subjects else []:
            assert self._needle is not None
            hits = {m.decode() for m in self._needle.findall(side.recipes[r].text.encode())}
            for subj in self.subjects:
                if subj.words & hits:
                    found.setdefault(subj.label, f"recipe {r}")
                    self.readers.setdefault(subj.label, {}).setdefault(f"recipe {r}", set()).add(n)
        if not found:
            return None, None
        shown = [f"{label} named in {rel}" for label, rel in list(found.items())[:2]]
        return f"lever {'; '.join(shown)}" + (f" (+{len(found) - 2} rows)" if len(found) > 2 else ""), True

    def counts(self) -> str:
        n = collections.Counter(subj.kind for subj in self.subjects)
        return f"{n['added']} added, {n['changed']} changed, {n['removed']} removed" + (f", {n['const']} consts" if n["const"] else "")

    def summary(self) -> str:
        """The clause of the `affected:` line: how the registry file was read."""
        if not (self.by_rows or self.why_whole):
            return ""
        if self.why_whole is not None:
            return f"{LEVER_REGISTRY} read whole: {self.why_whole}"
        return f"{LEVER_REGISTRY} read by row: {self.counts()}"

    def lines(self) -> list[str]:
        """The `lever rows:` block of `affected`: each moved row and the recipes whose inputs name it."""
        if not self.by_rows:
            return []
        out = [f"lever rows ({LEVER_REGISTRY}): {self.counts()}; the recipe that tests the registry: {' '.join(sorted(self.own)) or 'none'}"]
        for subj in self.subjects:
            files = self.readers.get(subj.label, {})
            recipes = sorted({n for ns in files.values() for n in ns})
            shown = ", ".join(sorted(files)[:3]) + (f" (+{len(files) - 3})" if len(files) > 3 else "")
            out.append(f"  {subj.kind} {subj.label}: " + (f"named in {shown} -> {' '.join(recipes)}" if files else "named in no file the recipes' inputs hold"))
        return out


# ---- a Rust file whose diff is comments or layout ----
#
# A Rust file whose two sides lex to the same tokens once comments and white space are left out compiles to the same program, so its
# diff selects no recipe: `CommentReads` holds those files, `select` and `narrow` skip them and print one line each. No gate reads a
# source line number (`line!`, `column!`, `file!` and `Location::caller` are spelled nowhere in crates/, and a PTX carries no `.loc`).
# The rule leaves a file whole, naming why in `notes`, when something reads its text or its doc comments:
#   - an `include_str!`, `include_bytes!` or `include!` in a workspace Rust file names it;
#   - a doc comment moved in the lib of a package whose doctests a gate-* or weekly-* recipe runs (`--doc`), or while a workspace package
#     depends on a crate whose macros turn doc comments into output (DOC_READER_CRATES);
#   - it is the lever registry (`LeverReads` reads it by row) or no file of a workspace package (an excluded crate's reproducer);
#   - one side has a string or comment that never ends.
# A comment a tool reads (`// SAFETY:`, `// PIN(`) is read by the static tier (ALWAYS), which this rule does not touch.

COMMENT_ONLY = "comment-only, no gate reads it"
LAYOUT_ONLY = "layout-only, no gate reads it"
# The crates whose derives or attribute macros turn doc comments into output (help text, a schema).
DOC_READER_CRATES = frozenset({"clap", "clap_derive", "structopt", "argh", "bpaf", "schemars", "utoipa", "documented", "poem-openapi", "async-graphql"})
_INCLUDE_CALL = re.compile(r"\binclude(?:_str|_bytes)?!\s*\(")
_STRING_LITERAL = re.compile(r'"((?:[^"\\]|\\.)*)"', re.S)


def _is_doc_comment(c: str) -> bool:
    """Whether comment `c` is a doc comment: `///` (not `////`), `//!`, `/**` (not `/***` or `/**/`) or `/*!`."""
    if c.startswith("//"):
        return c.startswith("//!") or (c.startswith("///") and not c.startswith("////"))
    return c.startswith("/*!") or (c.startswith("/**") and not c.startswith(("/***", "/**/")))


@functools.lru_cache(maxsize=256)
def _rs_stream(text: str) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[int, str], ...]] | None:
    """(tokens, comments) of Rust source (`_rs_tokens`); None when a string or comment never ends: such a text is no stream to compare."""
    comments: list[tuple[int, str]] = []
    try:
        toks = _rs_tokens(text, comments)
    except LeverDoubt:
        return None
    return tuple(toks), tuple(comments)


def comment_diff(a_text: str, b_text: str) -> tuple[str, bool] | None:
    """What two different texts of one Rust file differ in when their tokens are equal: ("comment", whether a doc comment moved) or
    ("layout", False). None when the texts are equal, a token differs or either text has a string or comment that never ends."""
    if a_text == b_text:
        return None
    sa, sb = _rs_stream(a_text), _rs_stream(b_text)
    if sa is None or sb is None or sa[0] != sb[0]:
        return None
    if sa[1] == sb[1]:
        return "layout", False
    docs = [tuple(c for c in s[1] if _is_doc_comment(c[1])) for s in (sa, sb)]
    return "comment", docs[0] != docs[1]


def _include_args(text: str) -> list[str]:
    """The argument text of each `include_str!`, `include_bytes!` and `include!` call of Rust source `text`; a call in a comment is none."""
    if not _INCLUDE_CALL.search(text):
        return []
    code, shape = rust_lex(text)
    out = []
    for m in _INCLUDE_CALL.finditer(code):
        depth, j = 1, m.end()
        while j < len(shape) and depth:
            depth += (shape[j] == "(") - (shape[j] == ")")
            j += 1
        out.append(code[m.end():j - 1])
    return out


class CommentReads:
    """The changed Rust files of one diff whose two sides differ in comments or layout only: `only` maps each to the line `select` and
    `narrow` print for it (COMMENT_ONLY, LAYOUT_ONLY). A file the rule leaves whole is in `notes` with its reason. `on` False reads no
    file as comment-only (`affected --no-comment-only`)."""

    def __init__(self, changed: list[str], a: Side | None, b: Side, on: bool = True):
        self.only: dict[str, str] = {}
        self.notes: list[str] = []
        self.b = b
        self._includes: tuple[dict[str, str], list[tuple[str, str]]] | None = None
        self._doctests: dict[str, str] | None = None
        if not on or a is None:
            return
        for f in changed:
            if not f.endswith(".rs") or not f.startswith(b.tree.pkg_dirs) or not (a.tree.exists(f) and b.tree.exists(f)):
                continue
            moved = comment_diff(a.tree.read(f), b.tree.read(f))
            if moved is None:
                continue
            kind, docs = moved
            why = self._refusal(f, docs)
            if why is None:
                self.only[f] = COMMENT_ONLY if kind == "comment" else LAYOUT_ONLY
            else:
                self.notes.append(f"{f}: only comments or layout moved, and the file stays selected: {why}")

    def summary(self) -> str:
        """The clause of the `affected:` line: how many files were read as no change."""
        return f"comment- or layout-only files (no gate reads them): {len(self.only)}" if self.only else ""

    def _refusal(self, f: str, docs: bool) -> str | None:
        """Why file `f`, whose diff is comments or layout, still selects what reads it; None when nothing reads what moved."""
        if f == LEVER_REGISTRY:
            return "the lever registry is read by row (LeverReads)"
        exact, tails = self._include_targets()
        by = exact.get(f) or next((src for tail, src in tails if f == tail or f.endswith("/" + tail)), None)
        if by is not None:
            return f"{by} reads its text through include_str!, include_bytes! or include!"
        if not docs:
            return None
        tree = self.b.tree
        for p in tree.packages.values():
            for d in p.deps:
                if d.pkg in DOC_READER_CRATES:
                    return f"a doc comment moved and {p.name} depends on {d.pkg}, whose macros read doc comments"
        pkg = max((p for p in tree.packages.values() if f.startswith(p.dir + "/")), key=lambda p: len(p.dir), default=None)
        lib = pkg.lib() if pkg is not None else None
        run = self._doctest_runs().get(pkg.name) if pkg is not None else None
        if run is not None and lib is not None and f in tree.tree_of(lib.src)[0]:
            return f"a doc comment moved in the lib of {pkg.name}, whose doctests {run} runs"
        return None

    def _include_targets(self) -> tuple[dict[str, str], list[tuple[str, str]]]:
        """The files an include macro of a workspace Rust file names, read once: (path -> the file that names it, the literals of a
        call that builds its path (`concat!`, `env!`) as (a path tail, the file that names it))."""
        if self._includes is None:
            tree = self.b.tree
            exact: dict[str, str] = {}
            tails: list[tuple[str, str]] = []
            for pdir in tree.pkg_dirs:
                for base, dirs, files in os.walk(os.path.join(tree.root, pdir)):
                    dirs[:] = [d for d in dirs if d != "target"]
                    for rel in (os.path.relpath(os.path.join(base, n), tree.root) for n in files if n.endswith(".rs")):
                        for arg in _include_args(tree.read(rel)):
                            lits = _STRING_LITERAL.findall(arg)
                            if len(lits) == 1 and arg.strip() == f'"{lits[0]}"' and not lits[0].startswith("/"):
                                exact.setdefault(os.path.normpath(os.path.join(os.path.dirname(rel), lits[0])), rel)
                            else:
                                tails += [(os.path.normpath(s.lstrip("/")), rel) for s in lits if s.strip("/")]
            self._includes = (exact, tails)
        return self._includes

    def _doctest_runs(self) -> dict[str, str]:
        """Package -> a gate-* or weekly-* recipe that runs its doctests, on the tree the change lands in."""
        if self._doctests is None:
            out: dict[str, str] = {}
            for n in sorted(self.b.recipes):
                if n.startswith((GATE_PREFIX, WEEKLY_PREFIX)):
                    for p, kind, _, _ in self.b.graph.inputs(n).targets:
                        if kind == "doctest":
                            out.setdefault(p, n)
            self._doctests = out
        return self._doctests


def select(changed: list[str], a: Side | None, b: Side, prefix: str, tables: TableReads | None = None) -> tuple[list[Selection], list[str], list[str]]:
    """(selections in justfile order, unmapped files, notes). A manifest is read by table (`TableReads`)."""
    names = [n for n in sorted(b.recipes, key=lambda n: b.recipes[n].line) if n.startswith(prefix)]
    hits: dict[str, list[tuple[str, str]]] = {n: [] for n in names}
    mapped: set[str] = set()
    tables = tables if tables is not None else TableReads(changed, a, b)
    notes: list[str] = list(tables.notes)
    for f in changed:
        if f == "justfile" or f in tables.comments.only:
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
    notes += tables.lock_notes(names)
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
        elif f == LOCK_FILE and tables.lock is not None:
            hint = tables.lock_hint()
        elif f in tables.comments.only:
            hint = tables.comments.only[f]
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


def changed_files(spec: str) -> tuple[list[str], str | None, str | None, str]:
    """(files, the rev the A side is read at, B rev or None for the working tree, how the A side is printed). `A..B` is B
    against merge-base(A, B), as `git diff A...B` reads it: a branch forked behind its target has not changed what the
    target gained since, and a tree-to-tree diff would select every one of those commits' files too. When A is an
    ancestor of B the merge-base is A and the range is read as written."""
    if ".." in spec:
        a, b = spec.split("..", 1)
        a = a or "HEAD"
        b = b or "HEAD"
        try:
            mb = git("merge-base", a, b).strip()
        except RecipeError as err:
            raise RecipeError(f"{a} and {b} have no merge base to read {spec} against: {err}") from err
        files = git("diff", "--name-only", "--no-renames", mb, b).split()
        if mb == git("rev-parse", "--verify", f"{a}^{{commit}}").strip():
            return sorted(set(files)), a, b, a[:9]
        return sorted(set(files)), mb, b, f"{mb[:9]} (merge-base of {a} and {b})"
    base = git("merge-base", spec, "HEAD").strip()
    files = git("diff", "--name-only", "--no-renames", base).split()
    files += git("ls-files", "--others", "--exclude-standard").split()
    return sorted(set(files)), base, None, base[:9]


def cmd_affected(args: argparse.Namespace) -> int:
    if args.scan and not args.narrow:
        raise RecipeError("--scan needs --narrow")
    # Every scan log and the table are read before the diff: one the tool cannot vouch for is refused first.
    scans = [(read_scan(x), read_scan(y)) for x, y in args.scan or []]
    if args.narrow:
        load_gate_paths()  # a table this tool cannot read is refused before the diff is
    changed, a_rev, b_rev, a_label = changed_files(args.base)
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
        tables = TableReads(changed, a, b, args.comment_only)
        sels, unmapped, more = select(changed, a, b, prefix, tables)
        notes += more
        trows, tnote = trigger_rows(b)
        if tnote:
            notes.append(tnote)
        trig = weekly_triggers([f for f in changed if f not in tables.comments.only], a, b, trows) if prefix == GATE_PREFIX else []
        named = {f for t in trig for f in t.files}
        unmapped = [u for u in unmapped if u.split("  (")[0] not in named]
        n_weekly = sum(1 for n in b.recipes if n.startswith(WEEKLY_PREFIX))
        weekly = f", {len(trig)} of {n_weekly} weekly-recipes by a trigger" if prefix == GATE_PREFIX else ""
        total = sum(1 for n in b.recipes if n.startswith(prefix))
        rng = f"{a_label}..{b_rev[:9]}" if b_rev else f"{a_label}..(working tree)"
        nar = narrow(changed, a, b, scans, expanded_table(notes=notes), tables) if args.narrow else None
        narrowed = nar is not None and not nar.full
        picks = dict(nar.picks) if narrowed else {s.recipe: s.why for s in sels}
        stage: ClassStage | None = None
        ctables = None
        if args.class_stage:
            ctables = class_tables(b.tree.root)
            stage = class_narrow(picks, changed, ctables, lambda n: class_reach(n, changed, a, b),
                                 families_new_since(ROOT, ctables))
            picks = {n: why for n, why in picks.items() if n in stage.kept}
        deferred = mac_landing_deferred(sels, picks, b.recipes, b.tree.root) if prefix == GATE_PREFIX else []
        for n, _reason in deferred:
            picks.pop(n, None)
        vitems = variant_items(list(picks), a, b, changed, prefix) if prefix == GATE_PREFIX else []
        class_note = f"class-staged: {len(stage.kept)} kept, {len(stage.left_out)} left out" if stage else "class stage off (--no-class)"
        mac_note = f", {len(deferred)} deferred to weekly (CUDA gate, GGUF input)" if deferred else ""
        lever_why = tables.levers.summary()
        lever_note = f"; {lever_why}" if lever_why else ""
        if tables.static_moved:
            lever_note += f"; {', '.join(tables.static_moved)} moved: a lint level is held by the static tier ({STATIC_TIER}) and selects no gate"
        if tables.comments.only:
            lever_note += f"; {tables.comments.summary()}"
        if narrowed:
            by_derivation = f"; {len(nar.derived)} files by derivation" if nar.derived else ""
            print(f"affected: {rng} — {len(changed)} changed files, {len(picks)} of {total} {prefix or ''}recipes selected "
                  f"(narrowed from {len(sels)} by ptx-scan{by_derivation}; {class_note}){weekly}{mac_note}{lever_note}")
        else:
            print(f"affected: {rng} — {len(changed)} changed files, {len(picks)} of {total} {prefix or ''}recipes selected"
                  + f" ({class_note})" + weekly + mac_note + lever_note)
        if nar is not None:
            for r in nar.full:
                print(f"narrow: full list — {r}")
            for v in nar.verdicts:
                print(f"narrow: ptx-scan {v.line()}")
            for line in nar.files:
                print(f"narrow: file {line}")
            for d in nar.derived if narrowed else []:
                print(f"narrow: derived {d.file}: {len(d.kept)} of {d.linked} gates that link it through a dependency reach it: {' '.join(d.kept)}")
                print(f"narrow:   one chain, {d.chain}")

        def class_rule(n: str) -> str:
            """The class rule that kept a family pick (the common and the tooling ones have nothing to say)."""
            if stage is None or ctables is None:
                return ""
            row = ctables.gates.get(n)
            if row is None or row.cls != "family":
                return ""
            return f"  ({stage.kept[n]})"

        if narrowed:
            width = max([len(n) for n in picks] + [len(i) for i, _ in vitems] + [len(t.recipe) for t in trig] + [10])
            for n, why in picks.items():
                print(f"  {n:<{width}}  {why}{class_rule(n)}")
            print_variants(vitems, width)
            print_triggers(trig, width)
            print("recipes: " + " ".join([*picks, *(i for i, _ in vitems), *(t.recipe for t in trig)]))
            print("always: " + " ".join(ALWAYS))
            let_out = "ptx-scan " + "; ".join(v.line() for v in nar.verdicts) + (f", the derivation of {len(nar.derived)} files" if nar.derived else "")
            out = [s.recipe for s in sels if s.recipe not in nar.picks]
            if stage is None:
                print(f"left out ({len(out)}), each by {let_out}:")
                for n in out:
                    print(f"  - {n}  ({let_out})")
            else:
                reasons = [(n, let_out) for n in out]
                reasons += [(n, why) for n, why in stage.left_out.items()]
                print(f"left out ({len(reasons)}), each by its reason (--no-class keeps the class stage's gates):")
                for n, why in reasons:
                    print(f"  - {n}  ({why})")
        else:
            width = max([len(n) for n in picks] + [len(i) for i, _ in vitems] + [len(t.recipe) for t in trig] + [10])
            for s in sels:
                if s.recipe not in picks:
                    continue
                more_n = f" (+{len(s.files) - 1})" if len(s.files) > 1 else ""
                print(f"  {s.recipe:<{width}}  {s.files[0]}{more_n}  [{s.why}]{class_rule(s.recipe)}")
            print_variants(vitems, width)
            print_triggers(trig, width)
            print("recipes: " + " ".join([*picks, *(i for i, _ in vitems), *(t.recipe for t in trig)]))
            print("always: " + " ".join(ALWAYS))
            if stage is not None and stage.left_out:
                print(f"left out ({len(stage.left_out)}), each by its reason (--no-class keeps the class stage's gates):")
                for n, why in stage.left_out.items():
                    print(f"  - {n}  ({why})")
        if deferred:
            print(f"deferred ({len(deferred)}), a change under {', '.join(MAC_CRATES)} does not select a CUDA gate on a GGUF input; "
                  "weekly and the release batch run them:")
            for n, reason in deferred:
                print(f"  - {n}  ({reason})")
        print(f"unmapped ({len(unmapped)}):")
        for u in unmapped:
            print(f"  {u}")
        if tables.docs:
            print(f"manifests ({len(tables.docs)}), read by table:")
            for line in tables.lines():
                print(line)
        for line in tables.levers.lines():
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


def print_variants(vitems: list[tuple[str, str]], width: int) -> None:
    """The variant items of an `affected` list, one line each in the shape of a recipe's line: a batch `--list` takes the first word."""
    for item, why in vitems:
        print(f"  {item:<{width}}  [{why}]")


def variant_items_self_test(expect) -> None:
    """The variant items of an `affected` list: a selected recipe that a row lists gets its `NAME@BLOOMERY_FIXTURE_VARIANT=<tag>` item beside
    it; a recipe the diff did not select gets one only when its row changed; a side with no table has none. Each clause is red on an
    expansion that prints nothing."""
    import types

    def side(table: str | None, recipes: tuple[str, ...] = ("gate-a", "gate-b", "lint")) -> Side:
        tree = types.SimpleNamespace(exists=lambda rel: table is not None and rel == VARIANT_TABLE, read=lambda rel: table or "")
        return types.SimpleNamespace(tree=tree, recipes={r: None for r in recipes})  # type: ignore[return-value]

    row = "v1\tfam\ttrunk\tx=q4_K\tgate-a gate-b lint\twhy"
    edited = row.replace("x=q4_K", "x=q5_K")
    t = f"# c\n{row}\n"
    want_a = ("gate-a@BLOOMERY_FIXTURE_VARIANT=v1", "variant v1: its row lists gate-a, which is selected")
    got = variant_items(["gate-a"], side(t), side(t), [], GATE_PREFIX)
    expect(got == [want_a], f"a selected recipe a row lists gets its variant item beside it: {got}")
    expect(variant_items(["gate-c"], side(t), side(t), [], GATE_PREFIX) == [], "a recipe no row lists, or one the diff did not select, got a variant item")
    got = variant_items([], side(t), side(f"# c\n{edited}\n"), [VARIANT_TABLE], GATE_PREFIX)
    expect([i for i, _ in got] == ["gate-a@BLOOMERY_FIXTURE_VARIANT=v1", "gate-b@BLOOMERY_FIXTURE_VARIANT=v1"] and all("its row in" in w for _, w in got),
           f"an edited row selects its recipes' variant items though the diff selected none (a recipe outside the gate- prefix stays out): {got}")
    expect(variant_items([], side(t), side(t), [VARIANT_TABLE], GATE_PREFIX) == [], "an unchanged row was reported as changed")
    got = variant_items([], side(None), side(t), [VARIANT_TABLE], GATE_PREFIX)
    expect([i for i, _ in got] == ["gate-a@BLOOMERY_FIXTURE_VARIANT=v1", "gate-b@BLOOMERY_FIXTURE_VARIANT=v1"], f"a row added where the base had no table: {got}")
    expect(variant_items(["gate-a"], side(t), side(None), [], GATE_PREFIX) == [], "a side with no table got variant items")
    try:
        variant_items(["gate-a"], side(t), side("v1\tfam\n"), [], GATE_PREFIX)
        expect(False, "a malformed row was accepted")
    except RecipeError as e:
        expect("six tab-separated columns" in str(e), f"a malformed row's error: {e}")


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
        ki = side.graph.key_inputs(n)
        print(f"{n}\t{len(ri.files)} compilation files, {len(ki.files)} key files\t{tl}\tscripts: {' '.join(sorted(set(ri.commands.scripts))) or '-'}\truns: {' '.join(ri.commands.runs) or '-'}")
        for e in ri.errors:
            print(f"  error: {e}")
    return 0


def why_chains(side: Side, name: str, f: str, _seen: frozenset[str] = frozenset()) -> list[str]:
    """How recipe `name` (and the recipes it depends on) links the package that owns `f`: one line per cargo target,
    its shortest dependency chain to that package, or `its own package` when the target sits in it. Empty when
    no target links the package (`f` is a script, a global, a manifest read by table)."""
    tree = side.tree
    owner = next((p for p in tree.packages.values() if f.startswith(p.dir.rstrip("/") + "/")), None)
    if owner is None or name in _seen:
        return []
    out: set[str] = set()
    for inv in side.graph.inputs(name).commands.invocations:
        targets, _ = tree.resolve(inv)
        for pname, kind, tname, feats in targets:
            label = f"{pname} {kind} {tname}" if tname else f"{pname} {kind}"
            if pname == owner.name:
                out.add(f"{label}: its own package")
                continue
            dev = kind in ("test", "bench", "doctest", "libtest", "example")
            chains = tree.closure(tree.packages[pname], feats, dev, not inv.no_default, inv.all_features)
            if owner.name in chains:
                out.add(f"{label}: {chains[owner.name]}")
    for dep in side.recipes[name].deps:
        if dep in side.recipes:
            out.update(why_chains(side, dep, f, _seen | {name}))
    return sorted(out)


def cmd_why(args: argparse.Namespace) -> int:
    """The recipes whose key holds each FILE, each with the edge that brought it in. With `--recipe R` only R's, and
    for a file of a workspace crate R's shortest dependency chain to the crate, per cargo target."""
    side = make_side(ROOT)
    names = [n for n in sorted(side.recipes, key=lambda n: side.recipes[n].line) if args.all_recipes or n.startswith(GATE_PREFIX)]
    if args.recipe:
        unknown = [r for r in args.recipe if r not in side.recipes]
        if unknown:
            raise RecipeError(f"why --recipe: not a recipe: {' '.join(unknown)}")
        names = list(args.recipe)
    for f in args.files:
        f = _norm(f)
        hits = [(n, side.graph.key_inputs(n).match(f)) for n in names]
        if args.recipe:
            for n, w in hits:
                print(f"{f} in the key of {n}: " + (f"yes [{w}]" if w else "no"))
                for c in why_chains(side, n, f) if w else []:
                    print(f"  {c}")
            continue
        hits = [(n, w) for n, w in hits if w]
        print(f"{f}: {len(hits)} recipes")
        for n, w in hits:
            print(f"  {n}  [{w}]")
        if not hits:
            hint = orphan_hint(side, f)
            if hint:
                print(f"  ({hint})")
    return 0


# ----------------------------------------------------------------------------------------------
# key against reach: a crate a gate's targets never run code of must not move that gate's key
# ----------------------------------------------------------------------------------------------

# The pinned runtime crates must not enter keys merely through Cargo linkage. Manifests remain mandatory
# compilation inputs and are not counted as runtime code by key_holders. Build scripts/proc macros and opaque
# generated code can legitimately remain wider than runtime reach; gaps reports that distinction honestly.
KEY_REACH_PINS = {
    "bloomery-hf": "a Hugging Face fetch or checkpoint configuration reader: its commits move only the gates whose targets run its code "
    "(model-hf, gpu-gates under the `hf` feature, and gpu-glm5next under its `hf` feature for gate-gpu-glm5next-tp's folder test)",
    "bloomery-model-hf": "the fixture fetch and the checkpoint maps: gpu-gates links it under the `hf` feature, for a repo header and a Qwen3.8 checkpoint "
    "folder that bloomery-serve and gate_qwen4exp_nvfp4 open and for generate_glm5next (`glm5next-hf`), and gpu-glm5next under its `hf` "
    "feature, for gate-gpu-glm5next-tp's folder test",
}


def _gate_recipes(side: Side) -> list[str]:
    return [n for n in sorted(side.recipes, key=lambda n: side.recipes[n].line) if n.startswith((GATE_PREFIX, WEEKLY_PREFIX))]


def _linked_origin(why: str) -> bool:
    """Whether a recipe input came from a cargo target's crate graph, not from a script it runs, a directory it walks or
    a global (Graph.inputs words those `script …`, `find in …`, `named directory`, `cargo …`, `tools/box.sh …`, `profile …`)."""
    return not why.startswith(("script", "find in", "named directory", "cargo", "tools/box.sh", "profile "))


def key_holders(side: Side, crate: str, names: list[str]) -> list[str]:
    """The recipes of `names` whose key closure holds a file of workspace package `crate` through a cargo target's
    crate graph. A script's closure that names a file in prose is another class (`why FILE` lists it)."""
    pre = side.tree.packages[crate].dir.rstrip("/") + "/"
    out = []
    for n in names:
        ri = side.graph.key_inputs(n)
        if any(f.startswith(pre) and f not in side.tree.manifest_paths and _linked_origin(w) for f, w in ri.files.items()) or any(
            (p.startswith(pre) or pre.startswith(p)) and _linked_origin(w) for p, w in ri.prefixes.items()
        ):
            out.append(n)
    return out


def reach_holders(side: Side, crate: str, names: list[str]) -> list[str]:
    """Recipes whose uncut target closure reaches package code (independent of model/profile cuts)."""
    pre = side.tree.packages[crate].dir.rstrip("/") + "/"
    r = reach_of(side.tree)
    return [n for n in names if any(any(f.startswith(pre) for f in
            (r.nodes[m][1] for m in r._close(side.graph, dep, None)))
            for dep in recipe_closure(side.recipes, n))]


def key_reach_gap(side: Side, crate: str, names: list[str] | None = None) -> list[str]:
    """The recipes whose key holds a file of `crate` while none of their targets reaches code of it. A recipe that
    builds no cargo target (a script that walks the tree) has no target to ask and is not counted."""
    held = [n for n in key_holders(side, crate, names if names is not None else _gate_recipes(side)) if side.graph.inputs(n).targets]
    got = set(reach_holders(side, crate, held))
    return [n for n in held if n not in got]


def cmd_gaps(args: argparse.Namespace) -> int:
    """Per workspace crate, N(key): the gate-* and weekly-* recipes whose key holds a file of it, N(reach): those whose targets
    reach code of it, and the gap between them, largest first. A crate of KEY_REACH_PINS with a gap ends 1 (the pin)."""
    tmp = tempfile.mkdtemp(prefix="recipes-gaps-") if args.rev else ""
    try:
        side = make_side(archive(args.rev, os.path.join(tmp, "tree")) if args.rev else ROOT)
        names = [n for n in _gate_recipes(side) if side.graph.inputs(n).targets]
        crates = args.crates or sorted(side.tree.packages)
        unknown = [c for c in crates if c not in side.tree.packages]
        if unknown:
            raise RecipeError(f"gaps: not a workspace package: {' '.join(unknown)}")
        rows = []
        for c in crates:
            held = key_holders(side, c, names)
            got = set(reach_holders(side, c, held))
            rows.append((len(held) - len(got), c, len(held), len(got), [n for n in held if n not in got]))
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    rows.sort(key=lambda r: (-r[0], r[1]))
    print(f"gaps: {len(names)} gate-* and weekly-* recipes with cargo targets ({args.rev or 'working tree'})")
    print("crate\tkey\treach\tgap")
    bad = 0
    for gap, c, nk, nr, gaps in rows:
        if nk or nr:
            print(f"{c}\t{nk}\t{nr}\t{gap}")
        if args.list and gaps:
            print("  " + " ".join(gaps))
        if gap and c in KEY_REACH_PINS:
            bad += 1
            print(f"gaps: {c} is pinned to no gap ({KEY_REACH_PINS[c]}): {gap} recipes key it without reaching it", file=sys.stderr)
    return 1 if bad else 0


def _reach_sets(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    """`(label, files)` for one `reach` invocation: the positional files, or each line of `--batch` (`LABEL<TAB>files`)."""
    if not args.batch:
        if not args.files:
            raise RecipeError("reach: name a file, or pass --batch")
        return [("", list(args.files))]
    sets = []
    with open(args.batch, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                raise RecipeError(f"reach --batch: a line needs a label and a file: {line.strip()[:80]}")
            sets.append((parts[0], parts[1:]))
    if not sets:
        raise RecipeError(f"reach --batch: {args.batch} has no set")
    return sets


def _print_reach_mac(side: Side, gates: list[str], label: str, files: list[str]) -> None:
    """The gates that reach any of `files`, then the same list after a mac-crate landing defers CUDA gates on a GGUF input.
    A gate that reads one of the files as its own target stays."""
    tree, graph = side.tree, side.graph
    table = fixture_kind_table(tree.root)
    rows, cuts = table if table is not None else ({}, set())
    if label:
        print(f"SET {label}")
    reached: dict[str, None] = {}
    own: set[str] = set()
    missing = False
    for f in files:
        f = _norm(f)
        if not tree.exists(f):
            print(f"{f}: not a file of the tree")
            missing = True
            continue
        for n, w in ((n, graph.inputs(n).match(f)) for n in gates):
            if w and _own_origin(w):
                own.add(n)
        for n in dep_reach(tree, graph, gates, f):
            if n not in NARROW_ALWAYS:
                reached.setdefault(n, None)
    kept: list[str] = []
    deferred: list[tuple[str, str]] = []
    for n in gates:
        if n not in reached and n not in own:
            continue
        if n in own or table is None:
            kept.append(n)
            continue
        reason = mac_defer_reason(side.recipes, n, rows, cuts, tree.root)
        if reason and n in reached:
            deferred.append((n, reason))
        else:
            kept.append(n)
    if missing and not reached and not own:
        return
    print(f"OLD {len(reached) + len(own - reached.keys())}")
    for n in gates:
        if n in reached or n in own:
            print(f"  {n}")
    print(f"KEPT {len(kept)}")
    for n in kept:
        print(f"  {n}")
    print(f"DEFERRED {len(deferred)}")
    for n, reason in deferred:
        print(f"  {n}\t{reason}")


def cmd_reach(args: argparse.Namespace) -> int:
    """`why`'s files through the derivation (Reach): the gate-* recipes whose targets can run code of each file through
    a dependency edge, each with one chain, and the readers the derivation leaves out by count. `why` lists every
    recipe that links a file (the crate-grain bound narrow() reads); `reach` is the subset a row must name.
    `--mac-landing` prints the list a change under MAC_CRATES runs, and each CUDA gate on a GGUF input it defers."""
    side = make_side(ROOT)
    if args.mac_landing:
        gates = [n for n in sorted(side.recipes, key=lambda n: side.recipes[n].line) if n.startswith(GATE_PREFIX)]
        for label, files in _reach_sets(args):
            _print_reach_mac(side, gates, label, files)
        return 0
    if args.batch:
        raise RecipeError("reach: --batch is only read with --mac-landing")
    if not args.files:
        raise RecipeError("reach: name a file")
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
            print(f"  {n}" + (f"  [{reach_of(tree).reason(graph, n, got[n])}]" if args.brief else f"  {reach_chain(tree, graph, n, got[n])}"))
        left = [n for n in dep if n not in got and n not in NARROW_ALWAYS]
        if left:
            print(f"  not reached ({len(left)}): {' '.join(left)}")
    if args.row:
        names = [n for n in gates if n in union]
        glob = args.files[0] if len(args.files) == 1 else "{" + ",".join(_norm(f) for f in args.files) + "}"
        print(f"{glob}\t{DERIVE}\t({len(names)} gates reach it through a dependency; the row derives them)")
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


def expand_justfile(path: str) -> list[str]:
    """Every recipe of the justfile as just runs it, one line each: a `recipe NAME` header, its parameters, dependencies,
    attributes and doc as JSON, and each body line with every top-level variable (`{{arch}}`, `{{precheck}}`) replaced by
    the value `just --evaluate` gives it in this environment. A parameter and any other expression stay as the JSON of
    the expression, so a change inside one still shows. The proof that a refactor of the recipes moves no command: the
    output of the two justfiles is the same text."""
    dump = json.loads(_just(path, "--dump", "--dump-format", "json"))
    values: dict[str, str] = {}

    def value(name: str) -> str:
        if name not in values:
            values[name] = _just(path, "--evaluate", name)
        return values[name]

    out: list[str] = []
    for name in sorted(dump["recipes"]):
        rec = dump["recipes"][name]
        out.append(f"recipe {name}")
        for key in ("parameters", "dependencies", "attributes", "doc", "shebang", "quiet", "private"):
            out.append(f"  {key} {json.dumps(rec.get(key), sort_keys=True, ensure_ascii=False)}")
        for frags in rec["body"]:
            line = ""
            for frag in frags:
                if isinstance(frag, str):
                    line += frag
                elif len(frag) == 1 and isinstance(frag[0], list) and frag[0][:1] == ["variable"] and frag[0][1] in dump["assignments"]:
                    line += value(frag[0][1])
                else:
                    line += "{{" + json.dumps(frag, sort_keys=True, ensure_ascii=False) + "}}"
            out.append(f"  | {line}")
    return out


def cmd_expand(args: argparse.Namespace) -> int:
    for line in expand_justfile(args.justfile or os.path.join(ROOT, "justfile")):
        print(line)
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
#     tools/gate-paths.tsv whose glob matches it, all of them (a `derive` row's recipes are the gates the tree derives for
#     it, written out on read by expanded_table). No row: the file's own derivation (dep_reach: the gates whose
#     reach holds a unit of it, the table's own derivation applied to the file), for a Rust source file of a lib's
#     module tree that some gate's reach holds; a file the derivation cannot read (derivation_gap: not Rust source, deleted, a
#     build script, in no lib's module tree, or reached by no gate) keeps the full list, naming the file and why.
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


# `ptx-compare BASE_LOG HEAD_LOG`: the add and move classes' proof (AGENTS.md's change-class table) as one table of two full
# ptx-scan logs of one binary. Exit codes: 0 nothing existing moved (the head holds the base's entries, each row and digest
# equal, and maybe more); 3 an existing entry's row or digest moved, or the bundle set did; 4 an entry was removed; 5 both;
# 2 a pair or log refused by name (RecipeError, `main`).
PTX_COMPARE_RC = {(False, False): 0, (True, False): 3, (False, True): 4, (True, True): 5}


def ptx_compare(base: ScanLog, head: ScanLog, base_label: str = "base", head_label: str = "head") -> tuple[int, list[str]]:
    """The table of two scans: entries on each side, those added, those removed and those whose any column or digest moved
    (each with its moved columns). The pair is read through scan_verdict (one binary, one method, one toolchain and card
    capability), and a `--no-jit` scan is refused: the add class reads the JIT columns."""
    if base.nojit or head.nojit:
        which = "both" if base.nojit and head.nojit else "base" if base.nojit else "head"
        raise RecipeError(f"scan pair {base.path} {head.path}: a --no-jit scan ({which}) holds no JIT columns, which the add class reads: "
                          "rescan without --no-jit (a move alone is `just canon-check`'s proof, from --no-jit scans)")
    scan_verdict(base, head)
    cols = head.header[1:]
    added = sorted(set(head.rows) - set(base.rows))
    removed = sorted(set(base.rows) - set(head.rows))
    changed: list[tuple[str, str]] = []
    for e in sorted(set(base.rows) & set(head.rows)):
        moved = [f"{c}:{a}->{b}" for c, a, b in zip(cols, base.rows[e], head.rows[e]) if a != b]
        if base.md5[e] != head.md5[e]:
            moved.append(f"ptx-scan-md5:{base.md5[e][0][:8]}->{head.md5[e][0][:8]}")
        if moved:
            changed.append((e, " ".join(moved)))
    bundles = sorted(base.bundles) != sorted(head.bundles)

    def names(xs: list[str]) -> str:
        return f"{len(xs)}: {' '.join(xs)}" if xs else "none"

    out = [f"ptx-compare: bin={head.bin} base={base_label} head={head_label} method={head.fields['method']} arch={head.fields.get('arch', '?')} "
           f"jit-cc={head.fields.get('jit-cc', head.fields.get('jit-card', '?'))}",
           f"  base entries  {len(base.rows)}",
           f"  head entries  {len(head.rows)}",
           f"  added         {names(added)}",
           f"  removed       {names(removed)}",
           f"  changed       {len(changed)}: {' | '.join(f'{e} ({m})' for e, m in changed)}" if changed else "  changed       none"]
    if bundles:
        out.append(f"  bundles       moved: {' '.join(base.bundles)} -> {' '.join(head.bundles)}")
    rc = PTX_COMPARE_RC[(bool(changed) or bundles, bool(removed))]
    word = {0: "identical" if not added else "added-only", 3: "changed", 4: "removed", 5: "changed and removed"}[rc]
    out.append(f"ptx-compare: {word} (rc {rc})")
    return rc, out


def cmd_ptx_compare(args: argparse.Namespace) -> int:
    rc, lines = ptx_compare(read_scan(args.base_log), read_scan(args.head_log), args.base_label, args.head_label)
    print("\n".join(lines))
    return rc


@dataclass
class PathRow:
    line: int
    glob: str
    recipes: list[str]  # ["*"]: every gate
    why: str
    pattern: re.Pattern
    exceptions: dict[str, str] = field(default_factory=dict)  # gate the derivation says reaches a file of the row -> why it is left out
    derive: bool = False  # the recipes column opens with `derive`: expand_rows adds the gates the tree derives, before the hand names

    def every(self) -> bool:
        return self.recipes == ["*"]

    def hand_gates(self) -> list[str]:
        """The gate-* names the table's text writes in the row (for a `derive` row, the extras beyond the derivation)."""
        return [n for n in self.recipes if not n.startswith(WEEKLY_PREFIX)]

    def gates(self) -> list[str]:
        """The names that map a file for narrow(): the row's gate-* recipes, or `*`. A `derive` row has them only once
        expand_rows has run: reading its hand names alone would drop the derived gates without a word."""
        if self.derive:
            raise RecipeError(f"{GATE_PATHS}:{self.line}: the `derive` row {self.glob} is read before expand_rows names its gates")
        return self.hand_gates()

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


DERIVE = "derive"


def read_gate_paths_text(root: str) -> str:
    path = os.path.join(root, GATE_PATHS)
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except OSError as err:
        raise RecipeError(f"{GATE_PATHS}: {err.strerror}") from err


def load_gate_paths(root: str = ROOT) -> list[PathRow]:
    """The table as its text writes it: rows with `derive` still unexpanded (expanded_table reads them as narrow() needs)."""
    return parse_gate_paths(read_gate_paths_text(root))


def parse_gate_paths(text: str) -> list[PathRow]:
    """`glob<TAB>recipes<TAB>why[<TAB>exceptions]`, recipes space-separated, the literal `*`, or `derive` first and then
    the hand names beyond what the tree derives; the exceptions as parse_exceptions reads them; `#` lines and blank
    lines skipped. Anything else is a named error."""
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
        derive = names[0] == DERIVE
        if DERIVE in names[1:]:
            raise RecipeError(f"{GATE_PATHS}:{n}: `{DERIVE}` opens the recipes column or is not in it")
        if derive:
            names = names[1:]
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
        rows.append(PathRow(n, f[0], names, f[2], pattern, exceptions, derive))
    if not rows:
        raise RecipeError(f"{GATE_PATHS}: no rows")
    return rows


_OWN_LIB_TESTS = re.compile(r"^(?:libtest <- \S+ lib tests|doctest <- \S+ doctests)(?: \(path literal\))?$")


def _own_origin(why: str) -> bool:
    """Whether a hit's origin is the recipe's own target or its own text, not a dependency edge: a lib's
    own tests read the lib as their own tree (origin_label words them `libtest <- <pkg> lib tests`)."""
    inner = re.sub(r" \(via [^)]*\)$", "", why)
    if _OWN_LIB_TESTS.match(inner):
        return True
    if "<-" in inner or inner.startswith(("cargo", "tools/box.sh", "profile ", "dep-info")):
        return False
    return True


# A change whose selecting files all sit under these crates does not select a CUDA gate whose model input is a
# GGUF file. That gate is on the weekly list and the release list (`mac_gguf_cuda`). A checkpoint-folder fixture
# still selects it. A gate that runs no gpu-gate.sh (a host gate, a Metal gate) is not in the set.
MAC_CRATES = ("embed", "hf", "jinja", "metal", "model-hf", "serve", "tokenizer")
_PROFILE_MODEL = re.compile(r"(?m)^MODEL=\$\{BLOOMERY_REF_MODEL:-([^}\n]+)\}")


def mac_crate_file(path: str) -> bool:
    """Whether `path` is a file of a crate in `MAC_CRATES`."""
    parts = path.split("/")
    return len(parts) >= 2 and parts[0] == "crates" and parts[1] in MAC_CRATES


def cuda_recipe(recipe: Recipe) -> bool:
    """Whether the recipe runs a binary through tools/gpu-gate.sh."""
    blob = recipe.text + "\n" + "\n".join(recipe.lines)
    return "gpu-gate.sh" in blob


def fixture_cut_dirs(root: str) -> set[str]:
    """Fixture directories that are checkpoint-folder cuts: `tools/fixture-cuts/<dir>.json` exists."""
    cut_dir = os.path.join(root, "tools/fixture-cuts")
    if not os.path.isdir(cut_dir):
        return set()
    return {name[:-5] for name in os.listdir(cut_dir) if name.endswith(".json")}


def fixture_kind_table(root: str) -> tuple[dict[str, str], set[str]] | None:
    """`(profile -> fixture dir, the folder-cut dirs)`, or None when the fixture table is not in the tree
    or has no row. A malformed row is still a named error."""
    dirs = os.path.join(root, FIXTURE_DIRS)
    refs = os.path.join(root, REF_PATHS)
    if not os.path.isfile(dirs) or not os.path.isfile(refs):
        return None
    with open(dirs, encoding="utf-8") as fh:
        dirs_text = fh.read()
    if not any(line.strip() and not line.startswith("#") for line in dirs_text.splitlines()):
        return None
    with open(refs, encoding="utf-8") as fh:
        refs_text = fh.read()
    rows, _default = fixture_table(dirs_text, refs_text)
    return rows, fixture_cut_dirs(root)


def recipe_profile(recipes: dict[str, Recipe], name: str) -> str | None:
    """The first BLOOMERY_MODEL of `name` and its dependencies, else the default profile. None when a recipe's
    commands do not parse."""
    seen: set[str] = set()

    def walk(n: str) -> str | None:
        if n in seen or n not in recipes:
            return None
        seen.add(n)
        try:
            models = recipe_commands(recipes[n]).models
        except RecipeError:
            return ""
        for model in models:
            if model:
                return model
        for dep in recipes[n].deps:
            got = walk(dep)
            if got == "":
                return ""
            if got:
                return got
        return None

    got = walk(name)
    if got == "":
        return None
    return got or DEFAULT_PROFILE


def _self_model_kind(root: str, prof: str) -> str | None:
    """`file` when a `self` profile's script defaults MODEL to a `.gguf` path. Anything else stays untold."""
    path = os.path.join(root, "tools/ref/models", f"{prof}.sh")
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return None
    m = _PROFILE_MODEL.search(text)
    if m is None:
        return None
    default = m.group(1).strip().strip("\"'")
    if default.endswith(".gguf"):
        return "file"
    return None


def gate_fixture_kind(recipes: dict[str, Recipe], name: str, rows: dict[str, str], cuts: set[str], root: str) -> tuple[str | None, str]:
    """`("folder"|"file"|None, why)`. None means the input kind is not told, and the gate stays on the landing list."""
    prof = recipe_profile(recipes, name)
    if prof is None:
        return None, "the recipe's profile does not parse"
    direc = rows.get(prof)
    if direc is None:
        return None, f"profile {prof} is not in {FIXTURE_DIRS}"
    if direc == "self":
        kind = _self_model_kind(root, prof)
        if kind == "file":
            return "file", f"profile {prof} has no fixture directory; tools/ref/models/{prof}.sh defaults MODEL to a .gguf file"
        return None, f"profile {prof} has no fixture file ({FIXTURE_DIRS} row self)"
    if direc in cuts:
        return "folder", f"profile {prof} fixture {direc} (tools/fixture-cuts/{direc}.json)"
    return "file", f"profile {prof} fixture {direc} (no tools/fixture-cuts/{direc}.json: GGUF shards)"


def mac_defer_reason(recipes: dict[str, Recipe], name: str, rows: dict[str, str], cuts: set[str], root: str) -> str | None:
    """Why a mac-crate landing defers `name`, or None when it still runs. A CUDA gate on a GGUF input is deferred;
    a checkpoint-folder fixture, a host gate and a Metal gate are not."""
    if name not in recipes or not cuda_recipe(recipes[name]):
        return None
    kind, why = gate_fixture_kind(recipes, name, rows, cuts, root)
    if kind != "file":
        return None
    return f"CUDA gate on a GGUF input ({why}); weekly and the release batch run it"


def mac_gguf_cuda(recipes: dict[str, Recipe], root: str) -> list[tuple[str, str]]:
    """Every CUDA `gate-*` recipe whose fixture input is a GGUF file, in justfile order, each with its reason.
    The weekly list and the release list run these. Empty when the fixture table is not in the tree."""
    table = fixture_kind_table(root)
    if table is None:
        return []
    rows, cuts = table
    out = []
    for name in sorted(recipes, key=lambda n: (recipes[n].line, n)):
        if not name.startswith(GATE_PREFIX):
            continue
        reason = mac_defer_reason(recipes, name, rows, cuts, root)
        if reason:
            out.append((name, reason))
    return out


def mac_landing_deferred(sels: list[Selection], picks: dict[str, str], recipes: dict[str, Recipe], root: str) -> list[tuple[str, str]]:
    """The picks a mac-crate change defers: every selecting file is under `MAC_CRATES`, the hit is not the gate's
    own target, and `mac_defer_reason` names it. A gate a file outside the table also selects stays."""
    table = fixture_kind_table(root)
    if table is None:
        return []
    rows, cuts = table
    by = {s.recipe: s for s in sels}
    out = []
    for n in picks:
        s = by.get(n)
        if s is None or not s.files or not all(mac_crate_file(f) for f in s.files):
            continue
        if _own_origin(s.why):
            continue
        reason = mac_defer_reason(recipes, n, rows, cuts, root)
        if reason:
            out.append((n, reason))
    return out


# The origins that are not a target's own read: a script the recipe names, a cargo global, a box call's file, a
# directory. A gate they belong to is not the gate's own target, so a change of one is not the gate's own change.
_NOT_TARGET_FILE = ("cargo", "tools/box.sh", "profile ", "dep-info", "script ", "named directory", "find in ")


def _own_target_file(why: str) -> bool:
    """Whether a hit is a file of the recipe's own cargo target: the target's own read, not a script the recipe
    names, a cargo global, a box call's file or a directory a script walks."""
    if not _own_origin(why):
        return False
    inner = re.sub(r" \(via [^)]*\)$", "", why)
    return not inner.startswith(_NOT_TARGET_FILE)


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
_MACRO_PATH_CALL = re.compile(r"((?:(?:r#)?[A-Za-z_]\w*\s*::\s*)*(?:r#)?[A-Za-z_]\w*)\s*!\s*[(\[{]")
_MACRO_IMPORT = re.compile(r"\b(?:use|extern\s+crate)\b")
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


def _key_attributes(shape: str) -> tuple[bool, bool]:
    """(opaque expansion, foreign linkage) in all attributes of lexed Rust, including cfg_attr payloads.

    cfg_attr predicates are deliberately not evaluated: any applicable startup/procedural attribute can
    execute code without an ordinary item edge. Builtin derives are the only derives resolved here.
    """
    standard = {"cfg", "allow", "warn", "deny", "forbid", "expect", "doc", "test", "ignore", "inline",
                "cold", "repr", "path", "macro_export", "must_use", "non_exhaustive",
                "track_caller", "automatically_derived", "deprecated", "should_panic", "used"}
    builtin_derives = {"Clone", "Copy", "Debug", "Default", "Eq", "PartialEq", "Ord", "PartialOrd", "Hash"}
    foreign = {"no_mangle", "export_name", "link", "link_name", "link_section"}

    def attribute(payload: str) -> tuple[bool, bool]:
        match = re.match(r"\s*((?:[A-Za-z_]\w*\s*::\s*)*[A-Za-z_]\w*)", payload)
        if match is None:
            return True, False
        name = re.sub(r"\s+", "", match.group(1))
        rest = payload[match.end():].strip()
        if name in ("cfg_attr", "derive", "unsafe"):
            if not rest.startswith("(") or _close(rest, 0) != len(rest) - 1:
                return True, False
            args = _split_top(rest[1:-1])
            if name == "derive":
                return any(arg not in builtin_derives for arg in args), False
            children = args[1:] if name == "cfg_attr" else args
            flags = [attribute(child) for child in children]
            return any(a for a, _ in flags), any(b for _, b in flags)
        return name not in standard and name not in foreign, name in foreign

    opaque = linkage = False
    for match in re.finditer(r"#\s*!?\s*\[", shape):
        end = _close(shape, match.end() - 1)
        if end is None:
            opaque = True
            continue
        a, b = attribute(shape[match.end():end])
        opaque |= a
        linkage |= b
    return opaque, linkage


@dataclass(frozen=True)
class KeySyntax:
    extern_crates: frozenset[str]
    opaque: bool
    foreign: bool
    macro_calls: frozenset[str]
    qualified: frozenset[tuple[str, ...]]
    conditional: bool


@functools.cache
def _key_syntax(text: str) -> KeySyntax:
    """File-local syntax only; bindings, macro resolution and active owners remain graph-specific."""
    code, shape = rust_lex(text)
    opaque, linkage = _key_attributes(shape)
    calls = frozenset(match.group(1) for match in _MACRO_PATH_CALL.finditer(shape))
    return KeySyntax(
        frozenset(re.findall(r"\bextern\s+crate\s+([A-Za-z_]\w*)", shape)),
        opaque or not shape.isascii() or bool(re.search(r"(?<![\w:>\]])::|r#[A-Za-z_]", shape)),
        linkage or bool(re.search(r'\bextern\s*(?:"[^"]*"\s*)?\{', code))
        or bool(re.search(r'\binclude!\s*\(\s*(?!")\S', code)
                or re.search(r'\binclude!\s*\(\s*"[^"]*(?<!\.rs)"', code)),
        calls,
        frozenset(tuple(part.removeprefix("r#") for part in re.split(r"\s*::\s*", call))
                  for call in calls if "::" in call),
        bool(re.search(r"#\s*!?\s*\[\s*cfg(?:_attr)?\b", shape)),
    )


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
def _source_macro_bodies(text: str, test: bool) -> dict[int, str]:
    """Lexed bodies with only fixed $crate:: dollar paths, indexed like rust_facts."""
    _, shape = build_view(text, test)
    raw: list = []
    _module_units(shape, 0, len(shape), (), raw, [])
    return {k: shape[a:b] for k, (kind, _, _, a, b, _, _) in enumerate(raw)
            if kind == "macro_rules!" and not re.search(r"\$(?!\s*crate\s*::)", shape[a:b])}


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

# ---- run-time dispatch by the opened file's architecture: the arch registry ----
#
# The registry's table holds one row per group of `general.architecture` strings, and a row holds function pointers: the
# family's reader (`read: Some(qwen35moe::spec::read)`) and its draft rule. `registry::row_of(arch)` picks a row by the
# architecture of the file a program opened, so the item graph below reaches every family's reader from every unit that
# names the table (rule 6 of Reach). The edges from the table to a row's entries are therefore dispatch edges: a gate's
# reach keeps the ones of a row only when the gate can open a file of that row (Reach.gate_archs: its profile, the names
# of its own targets, the /models paths they name), or walks the table itself (a fn of the registry that takes no key and
# names the table, and its tests). A unit a closed row's entries lead to stays reached when it lies outside the architecture
# tree, the registry's directory (ops, placement, the kernel crates: what a reader calls, a gate's other code calls too);
# a file of the tree only a closed row leads to is left out of the gate's reach.
# A gate that selects its model file some other way (a command line flag, an environment variable, a recipe the Mac runs)
# keeps every row.
ARCH_REGISTRY = ("crates/model/src/arch/registry.rs", "ROWS")  # (file, the table's static)
# What picks a model file other than the profile: `-m PATH`, `--hf`, `--model`, a reference-model variable.
MODEL_SELECTORS = re.compile(r"BLOOMERY_REF_MODEL|BLOOMERY_V41_MODEL|(?:^|\s)(?:-m|--model|--hf)(?:\s|=)")
_ROW_START = re.compile(r"\bArchRow\s*\{")
_ROW_NAMES = re.compile(r"\bnames\s*:\s*(&\s*\[[^\]]*\]|[A-Za-z_][A-Za-z0-9_]*)")
_ROW_PICK = re.compile(r"\b(?:read|draft)\s*:\s*Some\s*\(\s*([A-Za-z_][A-Za-z0-9_]*(?:\s*::\s*[A-Za-z_][A-Za-z0-9_]*)*)\s*\)")
_STR_LIT = re.compile(r'"([^"\\]*)"')
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
ANY_ARCH = "*"  # a row whose strings the reach cannot read: every gate keeps it
DEFAULT_SIGNAL = "the default profile"  # a box command that sets no BLOOMERY_MODEL: no evidence of its own


def arch_norm(name: str) -> str:
    """An architecture string, a profile or a target name in one spelling: `glm5-next` and `glm5_next` are one."""
    return name.replace("-", "_").lower()


@dataclass
class Dispatch:
    """The registry's table in one root: its node, each row's architecture strings (arch_norm), the node of each row's
    entries (reader, draft rule) with the rows that name it, and the units that walk every row."""

    node: int
    rows: list[frozenset[str]]
    entries: dict[int, frozenset[int]]
    enumerators: frozenset[int]


def _matching(shape: str, at: int) -> int:
    """The index just past the bracket that closes the one at `at` (shape has its strings and comments blanked)."""
    pairs = {"[": "]", "{": "}", "(": ")"}
    stack = [pairs[shape[at]]]
    for i in range(at + 1, len(shape)):
        c = shape[i]
        if c in pairs:
            stack.append(pairs[c])
        elif stack and c == stack[-1]:
            stack.pop()
            if not stack:
                return i + 1
    raise RecipeError(f"{ARCH_REGISTRY[0]}: a bracket at offset {at} is never closed")


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
      imports by name (an item a macro writes) reaches every unit of M; a name M imports is accounted for, and a
      path through it that ends at a module or an external item reaches M's `use` unit only. A path that ends at a
      module is a namespace, and a `mod` declaration no edge: declaring a module runs none of its code.
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
    6. Run-time dispatch by value. An `Arch` match names every arm's path, so every arm is reached: the static reach
       over-selects there, and a row that leaves such a gate out says so as a named exception (gate_paths_problems). The
       registry's table of readers is read as what it is (Dispatch): its edges to a row's entries are kept for a gate that
       can open a file of that row, or walks the table, and cut for the others (gate, Reach.gate_archs); a dispatch
       elsewhere (a serve seat table, coverage::check's pin table) is still this rule's exception.

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
        self._dispatch: dict[RootKey, Dispatch] | None = None
        self._rows_nodes: dict[int, Dispatch] = {}
        self._enum_nodes: frozenset[int] = frozenset()
        self._family_dirs: tuple[str, ...] = ()
        self._archs: dict[tuple[str, str], tuple[frozenset[str] | None, dict[str, str]]] = {}
        self._dropped: dict[tuple[str, str], frozenset[int]] = {}
        self._dir_archs: dict[str, set[str]] | None = None
        self._facts: dict[tuple[str, bool], RustFacts] = {}
        self._key_snapshot = None
        self._key_observed = {}
        self._key_wide: set[str] = set()

    def text(self, f: str) -> str:
        if f not in self._text:
            self._text[f] = self.tree.read(f)
        return self._text[f]

    def facts(self, rk: RootKey, f: str) -> RustFacts:
        key = (f, rk[2])
        if key not in self._facts:
            self._facts[key] = rust_facts(self.text(f), rk[2])
        return self._facts[key]

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
        """Every unit of a Rust module; a non-source data file uses index -1."""
        if not f.endswith(".rs") and f not in self.mods(rk).values():
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
        lrk = (rk[0], rk[1], False)  # a lib's own test build names the same exported macros, units of the lib's non-test view
        if not mod and lrk in self.lib_roots and name in self.exported(rk[0]):
            return [(lrk, g, i) for g, i in self.exported(rk[0])[name]]
        out: list[Hit] = []
        leaves = facts.scoped.get((ctx, name), [])
        for bp in leaves:
            out += self.resolve(rk, mod, bp + rest, strict, name if bp[0] == name else "")
        # a name the module imports is accounted for even when its path ends at a namespace (a module, an external
        # crate's item): the import is the unit it reaches, never every unit of the module
        if out or leaves:
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

    # -- the registry's dispatch --

    def dispatches(self) -> dict[RootKey, Dispatch]:
        """The registry's table in each root that compiles its file (the lib's, and the lib's own test build), read once;
        none for a tree without the registry's file."""
        if self._dispatch is None:
            self._dispatch = {}
            path, static = ARCH_REGISTRY
            dirs: set[str] = set()
            enum: set[int] = set()
            if self.tree.exists(path):
                for pname, p in sorted(self.tree.packages.items()):
                    lib = p.lib()
                    for test in (False, True) if lib is not None else ():
                        rk = (pname, lib.src, test)
                        if path not in self.mods(rk).values():
                            continue
                        d = self._read_table(rk, path, static)
                        if d is None:
                            continue
                        self._dispatch[rk] = d
                        self._rows_nodes[d.node] = d
                        enum |= d.enumerators
                        dirs.add(os.path.dirname(path))
            self._enum_nodes = frozenset(enum)
            self._family_dirs = tuple(sorted(dirs))
        return self._dispatch

    def _read_table(self, rk: RootKey, path: str, static: str) -> Dispatch | None:
        facts = self.facts(rk, path)
        idx = [i for i in facts.named.get(((), static), []) if facts.units[i].kind == "static"]
        if not idx:
            return None
        u = facts.units[idx[0]]
        code, shape = build_view(self.text(path), rk[2])
        head = re.search(rf"\bstatic\s+{re.escape(static)}\b", shape)
        if head is None:
            return None
        lb = shape.index("[", shape.index("=", head.end()))
        end = _matching(shape, lb)
        bases = self.bases(rk, path) or [()]
        rows: list[frozenset[str]] = []
        entries: dict[int, frozenset[int]] = {}
        for m in _ROW_START.finditer(shape, lb, end):
            block = code[m.start() : _matching(shape, m.end() - 1)]
            r = len(rows)
            rows.append(self._row_names(rk, path, u, block, bases))
            for text in _ROW_PICK.findall(block):
                for b in bases:
                    for h in self.resolve(rk, b + u.ctx, tuple(re.split(r"\s*::\s*", text))):
                        if h[2] >= 0:
                            x = self.node(h)
                            entries[x] = entries.get(x, frozenset()) | {r}
        # a fn of the registry that takes no key and names the table walks every row (its tests too); `row_of(arch)` does not
        lines = shape.split("\n")
        enum = set()
        for k, e in enumerate(facts.units):
            if e.kind == "fn" and e.name and static in e.idents:
                window = "\n".join(lines[e.line - 1 : e.line + 5])
                if re.search(rf"\bfn\s+{re.escape(e.name)}\s*(?:<[^>{{]*>)?\s*\(\s*\)", window):
                    enum.add(self.node((rk, path, k)))
        return Dispatch(self.node((rk, path, idx[0])), rows, entries, frozenset(enum))

    def _row_names(self, rk: RootKey, path: str, u: Unit, block: str, bases: list[tuple[str, ...]]) -> frozenset[str]:
        """A row's architecture strings: the literals of its `names`, and the strings of each const it names."""
        m = _ROW_NAMES.search(block)
        if m is None:
            return frozenset({ANY_ARCH})
        out = {arch_norm(x) for x in _STR_LIT.findall(m.group(1))}
        for ident in _IDENT.findall(_STR_LIT.sub("", m.group(1))):
            got = None
            for b in bases:
                local = [(rk, path, i) for i in self.facts(rk, path).named.get((u.ctx, ident), ())]
                for h in local or self.resolve(rk, b + u.ctx, (ident,)):
                    if h[2] >= 0 and self.facts(h[0], h[1]).units[h[2]].kind == "const":
                        cm = re.search(rf"\bconst\s+{ident}\b[^;]*;", rust_lex(self.text(h[1]))[0])
                        if cm:
                            got = {arch_norm(x) for x in _STR_LIT.findall(cm.group(0))}
            if got is None:
                return frozenset({ANY_ARCH})
            out |= got
        return frozenset(out) or frozenset({ANY_ARCH})

    def _family_file(self, f: str) -> bool:
        """Whether `f` lies in the architecture tree (the registry's directory), whose code a family's reader reaches."""
        return any(f.startswith(d + "/") for d in self._family_dirs)

    def rows_open(self, graph: Graph, n: str) -> frozenset[int] | None:
        """The rows of the registry's table gate `n` can open a file of, as row numbers; None for every row (the gate's
        files are not known, or the tree has no registry)."""
        d = next(iter(self.dispatches().values()), None)
        if d is None:
            return None
        archs, _ = self.gate_archs(graph, n)
        if archs is None:
            return None
        return frozenset(i for i, names in enumerate(d.rows) if ANY_ARCH in names or names & archs)

    def gate_archs(self, graph: Graph, n: str) -> tuple[frozenset[str] | None, dict[str, str]]:
        """(the architecture strings (arch_norm) of the model files gate `n` can open, each with the signal that named it;
        None when they cannot be told, with the reason under the key ``). The signals: the BLOOMERY_MODEL of each box
        command of the recipe and its dependencies (no BLOOMERY_MODEL is the default profile); the names of the gate's own
        cargo targets, which name a row's string (`qwen35moe_meta`); the /models paths its own targets' sources name, by the
        profile that defines that directory or the fixture directory of a family. A recipe the Mac runs, a model file
        picked by a flag or a reference-model variable, a profile no row names, and a /models path nothing explains with
        no other signal, are not told: every row stays open."""
        key = (n, graph.recipes[n].text)
        if key in self._archs:
            return self._archs[key]
        out, why = self._gate_archs(graph, n)
        self._archs[key] = (out, why)
        return out, why

    def _gate_archs(self, graph: Graph, n: str) -> tuple[frozenset[str] | None, dict[str, str]]:
        recipes = graph.recipes
        d = next(iter(self.dispatches().values()))
        known = frozenset().union(*d.rows)
        found: dict[str, str] = {}
        for r in recipe_closure(recipes, n):
            mac, unmodeled = command_shape(recipes[r])
            if mac or unmodeled:
                return None, {"": f"{r} runs `{(mac or unmodeled)[0]}` outside tools/box.sh's modeled shapes"}
            body = "\n".join(ln for ln in recipes[r].lines if not ln.lstrip("@- ").startswith("#"))
            sel = MODEL_SELECTORS.search(body)
            if sel:
                return None, {"": f"{r} picks its model file by `{sel.group(0).strip()}`"}
            for m in graph.inputs(r).commands.models:
                prof = DEFAULT_PROFILE if m is None else m
                if not FAMILY_NAME.fullmatch(prof):
                    return None, {"": f"{r} runs under the profile {prof!r}, which is not a plain name"}
                a = arch_norm(prof)
                if ANY_ARCH not in known and a not in known:
                    return None, {"": f"{r} runs under the profile {prof}, which no row of the registry names"}
                found.setdefault(a, f"profile {prof}" if m is not None else DEFAULT_SIGNAL)
        lits: set[str] = set()
        for pname, kind, tname, _ in graph.inputs(n).targets:
            if kind in ("lib", "doctest"):
                continue
            pkg = self.tree.packages[pname]
            t = pkg.lib() if kind == "libtest" else pkg.find(kind, tname or "")
            if t is None:
                continue
            label = tname or pname
            for a in sorted(known - {ANY_ARCH}):
                if a in arch_norm(label):
                    found.setdefault(a, f"target {label}")
            own = self.tree.tree_of(t.src, True)[0]
            if kind == "libtest":
                own = own - self.tree.tree_of(t.src)[0]
            for f in sorted(own):
                if f.endswith(".rs"):
                    lits |= {x for x in MODEL_LITERAL.findall(rust_lex(self.tree.read(f))[0])}
        stray = []
        for lit in sorted(lits):
            arch = self.literal_archs(lit)
            for a in arch:
                found.setdefault(a, f"names {lit}")
            if not arch and not lit.startswith((FIXTURE_ROOT, VARIANT_ROOT)):
                stray.append(lit)
        if stray and all(w == DEFAULT_SIGNAL for w in found.values()):
            return None, {"": f"its targets name {stray[0]}, which no profile defines, and nothing else tells its family"}
        return frozenset(found), found

    def literal_archs(self, lit: str) -> set[str]:
        """The architecture strings a /models path stands for: the profile whose directory it is (or which it extends by a
        `-` suffix: a quant of the same checkpoint), the family of a fixture directory under the fixture root."""
        if self._dir_archs is None:
            self._dir_archs = {}
            root = self.tree.root
            env = {k: v for k, v in os.environ.items() if k not in ("BLOOMERY_REF_MODEL", "BLOOMERY_TIER")}
            for prof in sorted(glob.glob(os.path.join(root, "tools/ref/models/*.sh"))):
                for dd in profile_dirs(root, os.path.relpath(prof, root), env):
                    self._dir_archs.setdefault(dd, set()).add(arch_norm(os.path.basename(prof)[:-3]))
            if self.tree.exists(FIXTURE_DIRS) and self.tree.exists(REF_PATHS):
                rows, _ = fixture_table(self.tree.read(FIXTURE_DIRS), self.tree.read(REF_PATHS))
                for fam, fdir in rows.items():
                    if fdir not in ("self", "none"):
                        self._dir_archs.setdefault(f"{FIXTURE_ROOT}/{fdir}", set()).add(arch_norm(fam))
        out: set[str] = set()
        for dd, a in self._dir_archs.items():
            if lit == dd or lit.startswith(dd + "/") or lit.startswith(dd + "-"):
                out |= a
        return out

    # -- gates --

    def gate(self, graph: Graph, n: str) -> dict[int, tuple[int, str]]:
        """Every unit gate `n`'s targets reach: node -> (the node it was reached from, the step); a target's own
        unit -> (-1, the target). The registry's dispatch edges of the rows the gate cannot open are cut (the section
        above Dispatch); what only they reached inside the architecture tree is in the map for its chains and in
        `dropped`, which gate_files leaves out."""
        key = (n, graph.recipes[n].text)
        if key in self._gates:
            return self._gates[key]
        rows = self.rows_open(graph, n)
        full = self._close(graph, n, None)
        if rows is None:
            self._gates[key], self._dropped[key] = full, frozenset()
            return full
        cut = self._close(graph, n, rows)
        self._dropped[key] = frozenset(m for m in full if m not in cut and self._family_file(self.nodes[m][1]))
        self._gates[key] = {**full, **cut}
        return self._gates[key]

    def _close(self, graph: Graph, n: str, open_rows: frozenset[int] | None,
               extra: set[Hit] | None = None) -> dict[int, tuple[int, str]]:
        """The closure of gate `n`'s targets; `open_rows`, when given, are the only rows of the registry's table whose
        entries its dispatch edges lead to, until a unit that walks every row is reached."""
        self.dispatches()
        cut = open_rows is not None and bool(self._rows_nodes)
        held: list[tuple[int, int, str]] = []
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
            own = [h for f in sorted(self.tree.tree_of(t.src, test)[0]) if f in linked for h in self.units(rk, f)]
            # a test harness runs its #[test] fns, and they reach the rest; a target with none runs its main
            entries = [h for h in own if kind in ("libtest", "test") and h[2] >= 0 and self.facts(h[0], h[1]).units[h[2]].test]
            for h in entries or own:
                seed(h, label)
        for h in extra or ():
            if h[1] in linked:
                seed(h, "source macro compilation effect")
        builds = {self.tree.packages[bpkg].name: src for src, bpkg in self.builds.items() if src in linked}
        names = frozenset().union(*(rs.names for rs in sets))
        for rs in sets:
            for i in rs.free:
                fire(rs, i)
        while True:
            while work:
                m = work.pop()
                dis = None
                if cut:
                    if m in self._enum_nodes:
                        cut = False
                        for src, x, how in held:
                            if x not in par:
                                par[x] = (src, how)
                                work.append(x)
                        held.clear()
                    dis = self._rows_nodes.get(m)
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
                        if cut and dis is not None and x in dis.entries and not dis.entries[x] & open_rows:
                            held.append((m, x, how))
                            continue
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
        return par

    def gate_files(self, graph: Graph, n: str) -> dict[str, int]:
        """file -> the first unit of it gate `n`'s reach holds."""
        key = (n, graph.recipes[n].text)
        if key not in self._gate_files:
            out: dict[str, int] = {}
            for m in self.gate(graph, n):
                if m not in self._dropped[key]:
                    out.setdefault(self.nodes[m][1], m)
            self._gate_files[key] = out
        return self._gate_files[key]

    def prepare(self) -> None:
        """Materialize and persist the structural graph, never recipe/profile-specific cuts.

        All target roots (including excluded-by-feature modules and test views) are discovered before publication.
        Closure, environment and architecture decisions remain lazy and are never serialized as structural facts.
        """
        if self._key_snapshot is not None:
            return
        from bloomery import reach_cache

        reach_cache.prepare(self)

    def validate(self) -> None:
        """Reject source movement during analysis/key assembly rather than publish a mixed-state green key."""
        if self._key_snapshot is not None:
            from bloomery import reach_cache

            if not reach_cache.current(self):
                raise RecipeError("source tree changed during reach analysis; request a fresh key")

    def key_files(self, graph: Graph, n: str) -> set[str]:
        """Uncut structural reach and mandatory compilation reads for the recipe's own targets.

        Unlike gate_files this is not a runtime/profile allowlist. A selected target owns its complete module
        tree; every linked package still owns its manifest/build script, and build dependencies and proc macros
        retain their complete compile-time closures even if no runtime call reaches that package.
        """
        ri, tree = graph.inputs(n), self.tree
        par = self._close(graph, n, None)
        keep: set[str] = set()
        keep |= {f for f in ri.files if f in tree.manifest_paths or f in CARGO_GLOBALS or f in OXIDE_GLOBALS
                 or ri.files[f].startswith("dep-info")}
        linked_packages = {p.name for p in tree.packages.values() if p.manifest in ri.files}
        boundary = set(ri.files) | set(ri.prefixes)
        compiled: dict[str, set[str]] = {p: set() for p in linked_packages}
        reads: dict[str, set[str]] = {p: set() for p in linked_packages}
        views: dict[tuple[str, str], set[RootKey]] = {}

        def compile_root(pname: str, src: str, test: bool = False) -> set[str]:
            files, lits = tree.tree_of(src, test)
            files = files & boundary
            for f in files:
                compiled[pname].add(f)
                views.setdefault((pname, f), set()).add((pname, src, test))
            found = files | set(tree._resolve_literals(tree.packages[pname], [(f, lit) for f, lit in lits if f in files]))
            reads[pname].update(found)
            return found

        def dependency_closure(owners: set[str]) -> set[str]:
            # Follow every edge within the already linked boundary. all_features on just the first owner
            # does not enable a transitive package's optional edges; Cargo may have unified those elsewhere.
            closed = set(owners)
            work = list(owners)
            while work:
                for dep in tree.packages[work.pop()].deps:
                    if dep.pkg in linked_packages and dep.pkg not in closed:
                        closed.add(dep.pkg)
                        work.append(dep.pkg)
            return closed

        mandatory: set[str] = set()
        for pname in linked_packages:
            pkg = tree.packages[pname]
            lib = pkg.lib()
            if lib is not None:
                compile_root(pname, lib.src)
            if any(t.proc_macro for t in pkg.targets):
                mandatory.add(pname)
            mandatory.update(dep.pkg for dep in pkg.deps if dep.kind == "build" and dep.pkg in linked_packages)
            for target in pkg.targets:
                if target.kind == "build":
                    keep |= compile_root(pname, target.src)
        for pname, kind, tname, _ in ri.targets:
            pkg = tree.packages[pname]
            target = pkg.lib() if kind in ("lib", "libtest", "doctest") else pkg.find(kind, tname or "")
            if target is not None:
                keep |= compile_root(pname, target.src, kind not in ("lib", "doctest"))
        for pname in dependency_closure(mandatory):
            keep |= reads[pname]
        safe_macros = {"assert", "assert_eq", "assert_ne", "debug_assert", "debug_assert_eq", "debug_assert_ne",
                       "println", "print", "eprintln", "eprint", "format", "format_args", "panic", "unreachable",
                       "unimplemented", "todo", "vec", "matches", "write", "writeln", "dbg", "concat", "stringify",
                       "env", "option_env", "file", "line", "column", "module_path", "cfg", "include", "include_str",
                       "include_bytes", "compile_error"}
        # `if !(...)` and `while !(...)` are negated expressions, not calls to macros named `if`/`while`.
        syntax_words = {"if", "while", "return", "break", "match", "let", "else", "for", "loop",
                        "yield", "continue", "move", "async", "await", "unsafe", "const"}
        opaque_owners: set[str] = set()
        outside_ownership = False
        def reached_owners(reached: dict[int, tuple[int, str]]) -> set[str]:
            # A reached unit of a package the recipe does not link sits in a file a script reads: its impl that no item reach can place
            # (a type a macro writes) is reached whenever its file is linked. That package compiles nothing here, so it owns no
            # compilation read (`compiled` holds the linked ones); the file is an input as the script's read, whatever the reach.
            return {self.nodes[m][0][0] for m in reached if self.nodes[m][2] >= 0} & linked_packages

        # An unused --extern rlib is a compilation input, not necessarily part of the executable. Start
        # with uncut reached owners, then honor explicit side-effect links even without a called item.
        # Within each runtime-active owner scan *all* compilation files: an unreachable ctor can run.
        active_owners = reached_owners(par)
        work = list(active_owners)
        source_seeds: set[Hit] = set()
        closed_seeds: set[Hit] = set()
        exported_names = {name for p in linked_packages for name, hits in self.exported(p).items() if hits}
        while True:
            while work:
                pname = work.pop()
                pkg = tree.packages[pname]
                for f in compiled[pname]:
                    syntax = _key_syntax(self.text(f))
                    facts_by_root = [(rk, self.facts(rk, f)) for rk in views[pname, f]]
                    links = set(syntax.extern_crates)
                    # Even an unused import can make rustc link a crate and run its constructors.
                    links.update(path[0].removeprefix("r#")
                                 for _, facts in facts_by_root for _, path, _, _ in facts.binds
                                 if path and path[0] not in ("crate", "self", "super"))
                    for dep in pkg.deps:
                        if dep.pkg not in linked_packages or dep.pkg in active_owners:
                            continue
                        lib = tree.packages[dep.pkg].lib()
                        if dep.key.replace("-", "_") in links or (lib is not None and lib.name.replace("-", "_") in links):
                            active_owners.add(dep.pkg)
                            work.append(dep.pkg)
                    opaque = syntax.opaque
                    outside_ownership |= syntax.foreign
                    calls, qualified_paths = syntax.macro_calls, syntax.qualified
                    qualified_known: set[tuple[RootKey, tuple[str, ...]]] = set()
                    # Source macros use the existing unhygienic expansion edges, even from otherwise
                    # unreached items: an expansion may emit a constructor or an implicit trait impl.
                    for rk, facts in facts_by_root:
                        known = set(self.macros(rk)) | exported_names
                        unresolved: set[tuple[str, ...]] = set()
                        for k, unit in enumerate(facts.units):
                            if any(name in known for name in unit.macro_calls):
                                source_seeds.add((rk, f, k))
                            # Definition-site bindings are not copied into the invocation context.
                            for name in unit.macro_calls:
                                for g, i in self.macros(rk).get(name, ()):
                                    body = _source_macro_bodies(self.text(g), rk[2]).get(i)
                                    if body is None or _MACRO_IMPORT.search(body):
                                        opaque = True
                            for path in set(unit.sites) & qualified_paths:
                                hits = [h for base in self.bases(rk, f) or [()]
                                        for h in self.resolve(rk, base + unit.ctx, path) if h[2] >= 0]
                                if hits and all(h[0][0] in linked_packages and
                                                self.facts(h[0], h[1]).units[h[2]].kind == "macro_rules!" for h in hits):
                                    source_seeds.add((rk, f, k))
                                    qualified_known.add((rk, path))
                                    for h in hits:
                                        body = _source_macro_bodies(self.text(h[1]), h[0][2]).get(h[2])
                                        # Imports and nested expansions may emit names absent from source edges.
                                        # Bound the invoker, not an unrelated item elsewhere in the macro crate.
                                        if body is None or _MACRO_IMPORT.search(body) or any(_key_attributes(body)) or any(
                                                match.group(1).removeprefix("r#") not in syntax_words
                                                for match in _MACRO_PATH_CALL.finditer(body)):
                                            opaque = True
                                else:
                                    unresolved.add(path)
                        qualified_known.difference_update((rk, path) for path in unresolved)
                    # Only unconditionally named local enums are proven to import variants, never macros.
                    # Imported/conditional paths remain opaque: a different binding may export a macro.
                    glob_shadow = False
                    conditional_bindings = syntax.conditional
                    for rk, facts in facts_by_root:
                        for ctx, path in facts.globs:
                            hits = [h for base in self.bases(rk, f) or [()]
                                    for h in self.walk(rk, base + ctx, path[:-1], True)
                                    if h[2] >= 0] if len(path) == 2 and path[0] not in facts.bound and not conditional_bindings else []
                            if not hits or any(self.facts(h[0], h[1]).units[h[2]].kind != "enum" for h in hits):
                                glob_shadow = True
                    for call in calls:
                        spelling = re.split(r"\s*::\s*", call)[-1]
                        if spelling in syntax_words:
                            continue
                        macro = spelling.removeprefix("r#")
                        shadowed = glob_shadow or any(macro in facts.bound for _, facts in facts_by_root)
                        local_source = all(macro in self.macros(rk) for rk, _ in facts_by_root)
                        path = tuple(part.removeprefix("r#") for part in re.split(r"\s*::\s*", call))
                        if "::" in call and all((rk, path) in qualified_known for rk, _ in facts_by_root):
                            continue
                        if "::" not in call and not shadowed and (macro in safe_macros or local_source):
                            continue
                        opaque = True
                    if opaque:
                        opaque_owners.add(pname)
                        # Opaque expansion can name its package's forward dependencies, not unrelated
                        # reverse/sibling packages. Inspect activated owners for linkage obligations too.
                        added = dependency_closure({pname}) - active_owners
                        active_owners.update(added)
                        work.extend(added)
            if source_seeds == closed_seeds:
                break
            par = self._close(graph, n, None, source_seeds)
            closed_seeds = set(source_seeds)
            added = reached_owners(par) - active_owners
            active_owners.update(added)
            work.extend(added)
        keep |= {self.nodes[m][1] for m in par}
        # Module declarations/re-exports on the path to reached code are behavior, not merely linkage.
        for rk, f in {(self.nodes[m][0], self.nodes[m][1]) for m in par}:
            mods = self.mods(rk)
            for path in self.bases(rk, f):
                for depth in range(len(path) + 1):
                    ancestor = mods.get(path[:depth])
                    if ancestor in ri.files:
                        keep.add(ancestor)
        for m in par:
            rk, f, k = self.nodes[m]
            if k < 0:
                continue
            unit = self.facts(rk, f).units[k]
            pkg = tree.packages[rk[0]]
            keep.update(tree._resolve_literals(pkg, [(f, lit) for lit in unit.literals]))
        # Macro expansion can name code in its own package and linked dependencies, but not an
        # unrelated reverse/sibling package through ordinary Cargo paths. Named data/script reads are
        # closed independently by Graph.key_inputs; arbitrary proc-macro I/O is outside this guarantee.
        for pname in dependency_closure(opaque_owners):
            keep |= reads[pname]
        if outside_ownership:
            self._key_wide.add(n)
            keep |= set(ri.files) | set(ri.prefixes)
        return keep

    def reason(self, graph: Graph, n: str, m: int) -> str:
        """Why gate `n` keeps unit `m`: a static chain, a registry row its model files name (with the signal that named it),
        or a gate that cannot tell its files / walks the whole table, which keeps every row."""
        par = self.gate(graph, n)
        self.dispatches()
        at, via = m, None
        while par[at][0] >= 0:
            src, _ = par[at]
            d = self._rows_nodes.get(src)
            if d is not None and at in d.entries:
                via = sorted({x for r in d.entries[at] for x in d.rows[r]})
            at = src
        if via is None:
            return "static chain"
        archs, why = self.gate_archs(graph, n)
        walks = next((self.label(x) for x in sorted(self._enum_nodes) if x in par), None)
        if walks is not None:
            return f"registry row {'/'.join(via)}: the gate walks every row ({walks})"
        if archs is None:
            return f"registry row {'/'.join(via)}: every row open ({why[''] if '' in why else 'files not told'})"
        signal = next((why[a] for a in sorted(archs) if a in via), "opens a file of the row")
        return f"registry row {'/'.join(via)}: {signal}"

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


@dataclass
class Derivation:
    """What the tree holds for one row beyond its hand names (row_derivation)."""

    # gate -> (a file of the row it reaches through a dependency, that file's unit); exceptions and held names left out
    reach: dict[str, tuple[str, int]]
    excepted: set[str]  # the row's exceptions that name a gate reaching a file of the row
    # gate -> (target kind, target name, the file of that target that names the module, the module); held names left out
    named: dict[str, tuple[str, str, str, str]]

    def gates(self, order: list[str]) -> list[str]:
        """The derived gates in `order` (the justfile's)."""
        return [n for n in order if n in self.reach or n in self.named]


def row_derivation(tree: Tree, graph: Graph, gates: list[str], matched: list[str], r: PathRow, held: set[str]) -> Derivation:
    """The gates the tree says belong in row `r` over the files `matched`, `held` names left out: the gates whose reach
    holds a file the glob matches through a dependency edge (dep_reach), less the row's exceptions, and the gates whose
    own target names the module the glob stands for. gate-ptx-spill is never one: narrow() adds it on its own rule."""
    held = held | set(NARROW_ALWAYS)
    reach: dict[str, tuple[str, int]] = {}
    excepted: set[str] = set()
    for f in matched:
        for n, node in dep_reach(tree, graph, gates, f).items():
            if n in held:
                continue
            if n in r.exceptions:
                excepted.add(n)
                continue
            reach.setdefault(n, (f, node))
    named: dict[str, tuple[str, str, str, str]] = {}
    mods = [m for m in (module_of(tree, alt) for alt in glob_alternatives(r.glob)) if m is not None]
    paths = [((ident, *rest), kids) for pkg, rest, kids in mods for ident in sorted(crate_idents(tree, pkg))]
    pkgs = {m[0] for m in mods}
    if paths:
        for n in gates:
            if n in held:
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
                    named[n] = (kind, tname or "", hit, "::".join(paths[0][0]))
                    break
    return Derivation(reach, excepted, named)


def gate_order(recipes: dict[str, Recipe]) -> list[str]:
    """The gate-* recipes in justfile order."""
    return [n for n in sorted(recipes, key=lambda n: recipes[n].line) if n.startswith(GATE_PREFIX)]


def expanded_row(r: PathRow, derived: list[str]) -> PathRow:
    """`derive` row `r` with the derived gates first, then its hand names and triggers, each once."""
    return dataclasses.replace(r, derive=False, recipes=list(dict.fromkeys([*derived, *r.recipes])))


def gate_paths_problems(tree: Tree, recipes: dict[str, Recipe], rows: list[PathRow] | None = None, stats: dict[str, int] | None = None,
                        derived: dict[int, list[str]] | None = None) -> list[str]:
    """The table against the tree: each row's recipes are gate recipes; its glob matches a file and a gate reads
    one of its files through a dependency edge (a row for files only own targets read maps nothing); each file it
    matches that a gate reads through a dependency edge is in the inputs of every recipe the row names
    (a row cannot claim a gate runs code it does not link); every gate the derivation (Reach) says reaches a
    file the glob matches through a dependency edge is in the row or named by the row's exceptions, each
    exception naming such a gate (a gate that reaches the file through another lib's code is never left out
    silently); and every gate whose own target names the module the glob stands for is in the row (a gate
    that calls into the module directly is never left out). A `derive` row stands for exactly those gates, so it
    holds the first two clauses for them, and a `derive` row the tree derives no gate for is refused. `stats`, when given, gains
    the counts the summary prints: rows held to the derivation, exceptions, the gates they name. `derived`, when given,
    gains per `derive` row line the gates the tree derives for it (expand_rows reads the same)."""
    try:
        rows = rows if rows is not None else load_gate_paths(tree.root)
    except RecipeError as err:
        return [str(err)]
    graph = Graph(tree, recipes)
    gates = gate_order(recipes)
    files = shipped_files(tree.root)
    problems: list[str] = []
    for r in rows:
        at = f"{GATE_PATHS}:{r.line} {r.glob}"
        hand = r.hand_gates()
        if not r.every():
            for n in hand:
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
        if not (r.derive or hand):
            continue
        deps = [f for f in matched if any(not _own_origin(w) for w in (graph.inputs(n).match(f) for n in gates) if w)]
        if not deps:
            problems.append(f"{at}: no gate reads a file the glob matches through a dependency — an own target's read selects "
                            "its recipes without a row (narrow()) — take the row out")
            continue
        # the derivation: a gate whose targets can run code of a file the glob matches, through another lib's code
        d = row_derivation(tree, graph, gates, matched, r, set() if r.derive else set(hand))
        got = d.gates(gates)
        if r.derive:
            if not got:
                problems.append(f"{at}: `{DERIVE}` stands for no gate — no gate reaches a file the glob matches through a dependency and "
                                "none has an own target naming its module — write the gates by hand, or take the row out")
                continue
            for n in sorted(set(r.exceptions) & set(d.named)):
                problems.append(f"{at}: {n} runs {d.named[n][0]} {d.named[n][1]}, whose {d.named[n][2]} names {d.named[n][3]} — the exceptions "
                                "cannot leave it out of the row")
            derived_here = set(got)
            if derived is not None:
                derived[r.line] = got
        else:
            derived_here = set()
        named = [n for n in dict.fromkeys([*got, *hand] if r.derive else hand) if n in recipes]
        unread: dict[str, list[str]] = {}
        for f in deps:
            for n in named:
                if graph.inputs(n).match(f) is None:
                    unread.setdefault(n, []).append(f)
        for n, fs in unread.items():
            fix = ("split the glob so the gate reads every file it matches, or name it in the row's exceptions" if n in derived_here
                   else "take it out of the row")
            problems.append(f"{at}: {n} does not read {fs[0]}{f' and {len(fs) - 1} more files the glob matches' if len(fs) > 1 else ''} "
                            f"(its targets do not link them) — {fix}")
        if not r.derive:
            for n, (f, node) in d.reach.items():
                problems.append(f"{at}: {n} reaches {f} — add it to the row, or write `{DERIVE}`, or name it in the row's exceptions "
                                f"with the reason; the chain: {reach_chain(tree, graph, n, node)}")
        for n in sorted(set(r.exceptions) - d.excepted):
            problems.append(f"{at}: the exception names {n}, which reaches no file the glob matches through a dependency — take it out")
        if stats is not None:
            stats["rows"] = stats.get("rows", 0) + 1
            stats["exceptions"] = stats.get("exceptions", 0) + (1 if r.exceptions else 0)
            stats["excepted"] = stats.get("excepted", 0) + len(r.exceptions)
        if not r.derive:
            for n, (kind, tname, hit, mod) in d.named.items():
                problems.append(f"{at}: {n} runs {kind} {tname}, whose {hit} names {mod} — add {n} to the row")
    return problems


# ---- the table's `derive` rows: expanded on read, cached by tree ----
#
# A `derive` row commits the glob, the hand names and the reason; the gates the tree derives for it are not
# in the file. Whoever reads a row for its gates (narrow(), `rows`) reads expanded_table: the file's text with
# each derive row's gates in, from the cache of the tree when one exists. The check derives the same gates for
# its own clauses and writes that cache.

with open(os.path.abspath(__file__), "rb") as _fh:
    _CODE_SHA = hashlib.sha256(_fh.read()).digest()  # the derivation's code as this process read it, not as the file is by the time a key is made
GATE_PATHS_CACHE = "~/.cache/bloomery/gate-paths"
GATE_PATHS_CACHE_KEEP = 16  # expanded tables kept, newest first: one per tree a landing or a check has seen


def expand_rows(tree: Tree, recipes: dict[str, Recipe], rows: list[PathRow], graph: Graph | None = None) -> list[PathRow]:
    """`rows` with each `derive` row replaced by the same row naming the gates the tree derives for it (row_derivation), then
    its hand names. A `derive` row the tree derives no gate for, or whose exceptions leave out a gate the derivation
    names, is a named error: the table is wrong, and a silent empty row would drop its files from every narrowed list."""
    if not any(r.derive for r in rows):
        return rows
    graph = graph if graph is not None else Graph(tree, recipes)
    gates = gate_order(recipes)
    files = shipped_files(tree.root)
    out: list[PathRow] = []
    for r in rows:
        if not r.derive:
            out.append(r)
            continue
        at = f"{GATE_PATHS}:{r.line} {r.glob}"
        d = row_derivation(tree, graph, gates, [f for f in files if r.pattern.match(f)], r, set())
        got = d.gates(gates)
        if not got:
            raise RecipeError(f"{at}: `{DERIVE}` stands for no gate — nothing the tree reads reaches a file the glob matches through a "
                              "dependency, and no gate has an own target naming its module")
        clash = sorted(set(r.exceptions) & set(got))
        if clash:
            raise RecipeError(f"{at}: {clash[0]} is derived for the row and the exceptions leave it out")
        out.append(expanded_row(r, got))
    return out


def expanded_text(text: str, rows: list[PathRow], expanded: list[PathRow]) -> str:
    """The table's `text` with each `derive` row's recipes column written out: the same lines, so a row keeps its line number."""
    lines = text.split("\n")
    for raw, new in zip(rows, expanded):
        if raw.derive:
            f = lines[raw.line - 1].split("\t")
            f[1] = " ".join(new.recipes)
            lines[raw.line - 1] = "\t".join(f)
    return "\n".join(lines)


def gate_paths_key(root: str, text: str) -> str | None:
    """The cache key of the expansion of the table `text` over the tree at `root`, or None when that tree has no identity this can
    vouch for (not a git work tree of its own, as a `git archive` copy is not): the bytes of this file as it was read (the derivation's code), the
    table's text, the tree HEAD holds, the tracked changes against it, and the name, size and mtime of each untracked file git does not
    ignore. A file git ignores is not in the key: a gate's reach would have to read it."""
    if not os.path.exists(os.path.join(root, ".git")):
        return None
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}

    def run(*args: str) -> bytes | None:
        p = subprocess.run(["git", "-C", root, *args], capture_output=True, env=env)
        return p.stdout if p.returncode == 0 else None

    top, head, diff, others = (run("rev-parse", "--show-toplevel"), run("rev-parse", "HEAD^{tree}"), run("diff", "HEAD", "--binary"),
                               run("ls-files", "--others", "--exclude-standard", "-z"))
    if top is None or head is None or diff is None or others is None or os.path.realpath(top.decode().strip()) != os.path.realpath(root):
        return None
    h = hashlib.sha256()
    h.update(_CODE_SHA)
    h.update(hashlib.sha256(text.encode()).digest())
    h.update(head.strip())
    h.update(hashlib.sha256(diff).digest())
    for name in sorted(x for x in others.split(b"\0") if x):
        try:
            st = os.stat(os.path.join(root, name.decode(errors="surrogateescape")))
        except OSError:
            return None
        h.update(name + f"\0{st.st_size}\0{st.st_mtime_ns}\0".encode())
    return h.hexdigest()


def read_expanded_cache(path: str, rows: list[PathRow]) -> list[PathRow] | None:
    """The cached expansion at `path` when it is the table `rows` reads and nothing else (every row's line, glob, reason, exceptions
    and hand names, and no `derive` left); a missing, unreadable or foreign file is a miss."""
    try:
        with open(path, encoding="utf-8") as fh:
            got = parse_gate_paths(fh.read())
    except (OSError, RecipeError):
        return None
    if len(got) != len(rows) or any(g.derive or (g.line, g.glob, g.why, g.exceptions) != (r.line, r.glob, r.why, r.exceptions)
                                    or not set(r.recipes) <= set(g.recipes) for g, r in zip(got, rows)):
        return None
    return got


def store_expanded_cache(path: str, text: str) -> None:
    """Writes the expanded table at `path` through a rename, then keeps the newest GATE_PATHS_CACHE_KEEP of its directory."""
    d = os.path.dirname(path)
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        os.makedirs(d, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
        kept = sorted((e for e in os.scandir(d) if e.name.endswith(".tsv")), key=lambda e: e.stat().st_mtime, reverse=True)
        for e in kept[GATE_PATHS_CACHE_KEEP:]:
            os.unlink(e.path)
    except OSError:
        pass  # a cache: a home that cannot be written costs the next reader a derivation, never a wrong table


def expanded_table(root: str = ROOT, side: Side | None = None, cache_dir: str = GATE_PATHS_CACHE, notes: list[str] | None = None) -> list[PathRow]:
    """The table at `root` as narrow() reads it: each `derive` row expanded over the tree at `root`. `side`, when given, is that
    tree read with no dep-info attached (a box's dep-info would change a bin's files and so the derivation). The expansion is
    kept on disk by the tree's key (gate_paths_key); the check writes it, so a landing that ran the check reads it."""
    text = read_gate_paths_text(root)
    rows = parse_gate_paths(text)
    if not any(r.derive for r in rows):
        return rows
    key = gate_paths_key(root, text)
    path = os.path.join(os.path.expanduser(cache_dir), key + ".tsv") if key else None
    n = sum(1 for r in rows if r.derive)
    hit = read_expanded_cache(path, rows) if path else None
    if hit is not None:
        if notes is not None:
            notes.append(f"{GATE_PATHS}: {n} derive rows read from the cache of this tree ({key[:12]})")
        return hit
    if sys.stderr.isatty():  # `just affected 2>&1 > LIST` is what tools/gate-batch.sh --list reads: its first lines must stay the recipe line and `affected:`
        print(f"recipes.py: deriving the gates of {n} rows of {GATE_PATHS} (no cache for this tree; the check writes one)", file=sys.stderr)
    side = side if side is not None else make_side(root)
    out = expand_rows(side.tree, side.recipes, rows, side.graph)
    stored = store_expansion(root, key, cache_dir, text, rows, out)
    if notes is not None:
        notes.append(f"{GATE_PATHS}: {n} derive rows derived from the tree" + (f", cached as {key[:12]}" if stored else
                     " (a tree with no git identity, or one that moved while it was derived: not cached)"))
    return out


def store_expansion(root: str, key: str | None, cache_dir: str, text: str, rows: list[PathRow], expanded: list[PathRow]) -> bool:
    """Stores the expansion derived over the tree at `root` under `key`, the tree's key taken before the tree was read, and only
    while the tree still has that key: a file edited during the derivation would otherwise file the old gates under the new tree."""
    if key is None or gate_paths_key(root, read_gate_paths_text(root)) != key:
        return False
    store_expanded_cache(os.path.join(os.path.expanduser(cache_dir), key + ".tsv"), expanded_text(text, rows, expanded))
    return True


def cmd_rows(args: argparse.Namespace) -> int:
    """The table as narrow() reads it, `derive` rows expanded: the file's text, line for line."""
    text = read_gate_paths_text(ROOT)
    print(expanded_text(text, parse_gate_paths(text), expanded_table(ROOT)), end="")
    return 0


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


def kernel_carriers(side: Side, gates: list[str], skip: dict[str, str] | None = None) -> list[Carrier]:
    tree = side.tree
    skip = CARRIER_SKIP if skip is None else skip

    def has_kernels(src: str) -> bool:
        files, _ = tree.tree_of(src)
        return any(KERNEL_ATTR.search(rust_lex(tree.read(f))[0]) for f in files)

    out: list[Carrier] = []
    for p in sorted(tree.packages.values(), key=lambda p: p.name):
        for t in p.targets:
            if t.kind not in ("lib", "bin") or (p.name == SCAN_PACKAGE and t.kind == "bin" and t.name in skip) or not has_kernels(t.src):
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
class Derived:
    """A changed host file no row maps, narrowed to its derivation: the gates that link it through a dependency edge, the ones
    whose reach holds a unit of it, and the chain of the one with the fewest steps."""

    file: str
    linked: int  # the gates that read it through a dependency edge
    kept: list[str]  # the gates whose reach holds a unit of it, in justfile order
    chain: str  # reach_chain of the kept gate with the fewest steps


@dataclass
class Narrowed:
    full: list[str]  # reasons the full list stands; empty when the list narrowed
    picks: dict[str, str]  # recipe -> the rule that picked it, in justfile order
    files: list[str]  # one line per changed file: the rule that mapped it
    verdicts: list[ScanVerdict]
    derived: list[Derived] = field(default_factory=list)  # the files no row maps, each read by the derivation


def derivation_gap(side: Side, f: str) -> str | None:
    """Why the derivation cannot read changed file `f` on `side`, None when it can. The derivation is Reach, the item graph of
    the lib module trees: it holds a unit of a lib's Rust source. What it cannot place keeps the full list: a file that is not
    Rust source (a manifest, a cargo global, a data file or a script is read by table, by path literal or by a recipe's own
    text), one the tree no longer holds, a build script (it runs at build and reaches every gate that reaches its crate) and a
    Rust file in no lib's module tree."""
    if not f.endswith(".rs"):
        return "it is not Rust source: the derivation places units of a lib's module tree"
    if not side.tree.exists(f):
        return "the tree read no longer holds it"
    reach = reach_of(side.tree)
    if f in reach.builds:
        return "it is a build script: it runs at build and reaches every gate that reaches its crate"
    if f not in reach.file_root:
        return "it is in no lib's module tree: the derivation holds no unit of it"
    return None


def derivation_empty(kept: Iterable[str]) -> str | None:
    """Why a file no gate's reach holds a unit of is not read as "no gate runs it": a module list, a `cfg` line or dead code
    changes what compiles, which the item graph does not run."""
    if next(iter(kept), None) is None:
        return "no gate's reach holds a unit of it (a module list, a cfg line or dead code): the item graph cannot say what an edit there moves"
    return None


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
        if f in tables.comments.only:
            # the same program on both sides: no gate runs it, no kernel moved, and the spill ratchet has nothing to read
            res.files.append(f"{f}: {tables.comments.only[f]}")
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
        # the lever registry whose diff is rows only: the recipes that read a moved row are its own picks, and no row of the table
        # (the registry's is `*`) speaks for a diff that moved no code
        by_rows = tables.levers.reads(f)
        hit_rows = [] if by_rows else [r for r in rows if r.pattern.match(f) and r.gates()]
        star = next((r for r in hit_rows if r.every()), None)
        rule: list[str] = [f"own target of {len(own)}"] if own else []
        if by_rows:
            rule.append(f"lever rows: {tables.levers.counts()}")
        elif NOT_HOST.search(f):
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
            # no row: the gates whose reach holds a unit of the file (the derivation the rows are held to), read on the tree
            # the change lands in; the spill ratchet has its own rule below
            gap = derivation_gap(b, f)
            reached = {} if gap else {n: node for n, node in dep_reach(b.tree, b.graph, gates, f).items() if n not in NARROW_ALWAYS}
            gap = gap or derivation_empty(reached)
            if gap:
                n, w = dep[0]
                res.full.append(f"{f}: {n} reads it through a dependency ({w}) and no row of {GATE_PATHS} maps it, "
                                f"and the derivation cannot read it: {gap}")
                rule.append("no row, not derivable")
            else:
                for n in reached:
                    pick(n, f"{f} [derived]")
                chains = {n: reach_chain(b.tree, b.graph, n, node) for n, node in reached.items()}
                shortest = min(chains, key=lambda n: (chains[n].count(" -["), n), default=None)
                res.derived.append(Derived(f, len(dep), [n for n in gates if n in reached], f"{shortest}: {chains[shortest]}" if shortest else ""))
                rule.append(f"no row, derived: {len(reached)} of {len(dep)} dependency readers reach it")
        elif not own:
            rule.append("no gate reads it")
        # a manifest whose diff moved only named entries, or the registry whose diff moved only rows, changes no kernel:
        # the recipes that read what moved are picked above as their own targets
        entries_only = by_rows or (f in tables.docs and all(own for _, own in tables.moved(f)))
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
                if f == "justfile" or f in tables.comments.only or tables.outside(n, f):
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
# A bin of SCAN_PACKAGE with kernels of its own left out of the kernel carriers: bin name -> reason. No scan set
# builds it and narrow() keeps no gate for it, so a bin some gate links cannot be named here (scan_set_self_test).
CARRIER_SKIP: dict[str, str] = {
    "link_probe": "the probe's kernel is its own and no gate links it",
}


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


def scan_set(side: Side, ship: bool = False) -> list[ScanPick]:
    """The minimal list of (bin, features) whose scans cover every kernel carrier of the tree, greedy: the scan that
    covers most uncovered carriers first, ties to fewer features, then to the bin's name. A carrier no bin of
    SCAN_PACKAGE can cover is a named error, never dropped. With `ship`, the shipped bins at the release features
    (tools/release/build.sh) are the first picks, so the set holds the scan tools/release/late-commit.sh reads."""
    gates = [n for n in sorted(side.recipes, key=lambda n: side.recipes[n].line) if n.startswith(GATE_PREFIX)]
    carriers = kernel_carriers(side, gates)
    shapes = scan_shapes(side)
    pkg = side.tree.packages[SCAN_PACKAGE]

    def covered_by(bin_name: str, feats: tuple[str, ...]) -> list[int]:
        bundles = set(side.tree.closure(pkg, list(feats), False))
        bundles |= {c.bundle for c in carriers if c.bundle == pkg.name and c.bin in (None, bin_name)}
        return [i for i, c in enumerate(carriers) if carrier_covered(c, bin_name, bundles)]

    cands: list[tuple[str, tuple[str, ...], list[int]]] = []
    for bin_name in sorted(set(shapes) - set(CARRIER_SKIP)):
        for feats in sorted(shapes[bin_name]):
            cands.append((bin_name, feats, covered_by(bin_name, feats)))
    uncovered = set(range(len(carriers)))
    picks: list[ScanPick] = []
    if ship:
        rb = read_release_build(side.tree.root)
        if rb.package != SCAN_PACKAGE:
            raise RecipeError(f"scan-set: {RELEASE_BUILD} builds package {rb.package}, and a ptx-scan reads the bins of {SCAN_PACKAGE} only")
        for b in rb.bins:
            if b not in shapes:
                raise RecipeError(f"scan-set: {RELEASE_BUILD} ships {b}, which is no bin of {SCAN_PACKAGE}")
            feats = tuple(sorted(set(rb.features)))
            cov = covered_by(b, feats)
            picks.append(ScanPick(b, feats, [carriers[i].label for i in cov]))
            uncovered -= set(cov)
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
        picks = scan_set(side, ship=args.ship or os.environ.get(SCAN_SHIP_ENV) == "1")
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
    if args.fill:
        raise RecipeError(f"--fill is gone: {GATE_PATHS} derives its gates (a row's `{DERIVE}` stands for them), so there is nothing to fill; "
                          "write `derive` in the recipes column of a row that should hold what the tree derives")
    text = read_gate_paths_text(ROOT)
    key = gate_paths_key(ROOT, text)  # before the tree is read: the derivation below is of this tree or of a later one
    tree = Tree(ROOT)
    recipes = load_justfile(args.justfile or os.path.join(ROOT, "justfile"))
    held: dict[str, int] = {}
    derived: dict[int, list[str]] = {}
    gp = gate_paths_problems(tree, recipes, stats=held, derived=derived)
    problems = (check(tree, recipes) + plan_problems(ROOT, recipes) + gp + weekly_problems(recipes)
                + host_problems(tree, recipes) + static_problems(tree, recipes) + nightly_problems(tree, load_lock(tree.root)))
    for p in problems:
        print(f"check-recipes: {p}", file=sys.stderr)
    print(f"check-recipes: {GATE_PATHS}: {held.get('rows', 0)} gate rows held to the derivation, {held.get('exceptions', 0)} rows "
          f"with exceptions naming {held.get('excepted', 0)} gates")
    if not gp and args.justfile is None:
        store_checked_expansion(ROOT, key, text, derived)
    return 1 if problems else 0


def store_checked_expansion(root: str, key: str | None, text: str, derived: dict[int, list[str]], cache_dir: str = GATE_PATHS_CACHE) -> bool:
    """Writes the cache of the table the check just held to the tree: the derivation is already in hand, so a later `affected
    --narrow` of this tree reads it instead of deriving. `key` and `text` are the tree's as they were before the check read it."""
    rows = parse_gate_paths(text)
    if not any(r.derive for r in rows):
        return False
    return store_expansion(root, key, cache_dir, text, rows, [expanded_row(r, derived[r.line]) if r.derive else r for r in rows])


# ----------------------------------------------------------------------------------------------
# the shipped closure and the release feature list
# ----------------------------------------------------------------------------------------------
#
# tools/release/build.sh says what ships: its BINS and the `--features` of its `cargo oxide build` line. Nothing
# below copies either; every use reads that file (read_release_build). Three rules stand on it:
#   - `ships FILE…` (tools/release/late-commit.sh's first test): a file is inside the shipped closure when a change to
#     it can change the binary the release packs: the bins' module trees, the lib of the scan package, every workspace
#     crate the build links at those features (its lib tree, manifest and build script), Cargo.lock,
#     rust-toolchain.toml, the manifests, .cargo/, tools/release/ and the scripts build.sh names. The walk is the one
#     `affected` reads (Graph.inputs over the build's own cargo line), so `mod`, `#[path]`, `include_str!` and the
#     feature cfgs are followed the same way. A bin's own tree is read with its `#[cfg(test)]` modules, an
#     over-selection that blocks and never one that lets a change through. With `--base REV` a file in REV's closure
#     counts too, so a deleted or moved-away file is not read against a tree that no longer holds it.
#   - `ship-scan BASE_LOG HEAD_LOG` (its second test): the ptx-scan pair of a shipped bin is identical.
#   - check's release-feature rule (release_feature_problems, run by bin_problems): a recipe that builds a shipped bin
#     builds it with that feature list. Two lanes that build one target/release/<bin> with two lists relink it under
#     each other (measured 0.3.1: one lane's image arm read a binary the other lane had rebuilt without `vision`).
#     It and the one-resolved-set rule (bin_feature_problems) each catch what the other lets by: that one holds every
#     bin to one set whatever the set, this one holds the shipped bins to the release list even when one recipe builds
#     them; bin_problems has the second step aside for the bins the first names, so a defect has one line.

RELEASE_BUILD = "tools/release/build.sh"
# Inputs of the shipped build besides its crates (rust-toolchain.toml, Cargo.lock and the root manifest are cargo
# globals): the cargo config and the scripts that build and pack.
SHIP_PREFIXES = {".cargo/": "cargo config (target-cpu, cuda-oxide)", "tools/release/": "the release build and pack scripts"}
# `tools/recipes.py scan-set` adds the shipped bins at the release features to the scan set when this is `1`.
SCAN_SHIP_ENV = "NARROW_SHIP"


@dataclass
class ReleaseBuild:
    package: str
    features: tuple[str, ...]  # build.sh's spelling and order
    bins: tuple[str, ...]


def read_release_build(root: str) -> ReleaseBuild:
    try:
        with open(os.path.join(root, RELEASE_BUILD), encoding="utf-8") as fh:
            return parse_release_build(fh.read())
    except OSError as err:
        raise RecipeError(f"{RELEASE_BUILD}: {err.strerror or err} — the shipped bins and their features are read from it") from err


def parse_release_build(text: str) -> ReleaseBuild:
    """BINS and the `cargo oxide build -p P --features F` line of tools/release/build.sh's text. A file that has
    not exactly one of each is a named error: what ships is never guessed."""
    code = [ln for ln in text.split("\n") if not ln.lstrip().startswith("#")]
    bins_lines = [ln for ln in code if ln.startswith("BINS=(")]
    m = re.fullmatch(r"BINS=\(([^()]*)\)\s*(?:#.*)?", bins_lines[0]) if len(bins_lines) == 1 else None
    if m is None or not m.group(1).split():
        raise RecipeError(f"{RELEASE_BUILD}: not one one-line `BINS=(name …)` with a name in it ({len(bins_lines)} lines start `BINS=(`)")
    builds = [ln for ln in code if re.search(r"\bcargo oxide build\b", ln)]
    if len(builds) != 1:
        raise RecipeError(f"{RELEASE_BUILD}: {len(builds)} `cargo oxide build` lines, not one")
    words = shell_words(builds[0])
    inv = parse_cargo(words[words.index("cargo") + 1:], builds[0])
    if inv is None or inv.sub != "build" or not inv.oxide or len(inv.packages) != 1 or not inv.features:
        raise RecipeError(f"{RELEASE_BUILD}: its `cargo oxide build` line is not one package (-p) with --features: {builds[0].strip()}")
    return ReleaseBuild(inv.packages[0], tuple(inv.features), tuple(m.group(1).split()))


def release_recipe(rb: ReleaseBuild) -> Recipe:
    """The release build as a recipe of one cargo line, so Graph.inputs reads its closure as it reads a gate's."""
    line = f"cargo oxide build --arch sm_86 -- -p {rb.package} --features {','.join(rb.features)} --release " + " ".join(f"--bin {b}" for b in rb.bins)
    return Recipe("release-build", [line], [], line)


@dataclass
class ShipClosure:
    build: ReleaseBuild
    files: dict[str, str]  # file -> why it is inside
    prefixes: dict[str, str]  # directory prefix "a/b/" -> why

    def match(self, path: str) -> str | None:
        if path in self.files:
            return self.files[path]
        for p, why in self.prefixes.items():
            if path.startswith(p):
                return why
        return None


def ship_closure(tree: Tree) -> ShipClosure:
    rb = read_release_build(tree.root)
    recipe = release_recipe(rb)
    graph = Graph(tree, {recipe.name: recipe})
    ri = graph.inputs(recipe.name)
    if ri.errors:
        raise RecipeError(f"{RELEASE_BUILD}: its build does not resolve in this tree: " + "; ".join(ri.errors))
    if tree.unresolved:
        raise RecipeError("the shipped closure has a source the walk cannot resolve (an incomplete closure would pass a change it should block): " + "; ".join(sorted(set(tree.unresolved))))
    files = dict(ri.files)
    prefixes = dict(ri.prefixes)
    for f, why in graph.script_closure(RELEASE_BUILD).items():
        if not f.startswith(tuple(SHIP_PREFIXES)):
            (prefixes if f.endswith("/") else files).setdefault(f, f"{RELEASE_BUILD} names it ({why})")
    for p, why in SHIP_PREFIXES.items():
        prefixes.setdefault(p, why)
    return ShipClosure(rb, files, prefixes)


def ship_hits(closures: list[tuple[str, ShipClosure]], files: list[str]) -> list[tuple[str, str]]:
    """(file, why) for each file inside the first closure that holds it; a label names a closure other than the head's."""
    out: list[tuple[str, str]] = []
    for raw in files:
        f = _norm(raw)
        for label, c in closures:
            why = c.match(f)
            if why is not None:
                out.append((f, why + (f" [{label}]" if label else "")))
                break
    return out


def _ship_closures(rev: str | None, base: str | None, tmp: str) -> list[tuple[str, ShipClosure]]:
    """The head's closure (REV's tree, else this working tree) and, with BASE, that commit's tree."""
    out = [("", ship_closure(Tree(archive(rev, os.path.join(tmp, "head")) if rev else ROOT)))]
    if base:
        out.append((f"in the closure of {base} only", ship_closure(Tree(archive(base, os.path.join(tmp, "base"))))))
    return out


def cmd_ships(args: argparse.Namespace) -> int:
    tmp = tempfile.mkdtemp(prefix="recipes-ships-")
    try:
        closures = _ship_closures(args.rev, args.base, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    hits = ship_hits(closures, args.files)
    for f, why in hits:
        print(f"{f}\t{why}")
    rb = closures[0][1].build
    what = f"{' '.join(rb.bins)} at {','.join(rb.features)} ({args.rev or 'working tree'})"
    print(f"ships: {len(hits)} of {len(args.files)} files are inside the shipped closure of {what}", file=sys.stderr)
    return 1 if hits else 0


def shipped_kernel_libs(tree: Tree, rb: ReleaseBuild) -> list[str]:
    """The packages whose lib has kernels among those the release build links: the bundles its scan should name."""
    pkg = tree.packages[rb.package]
    out = []
    for n in [pkg.name, *tree.closure(pkg, list(rb.features), False)]:
        lib = tree.packages[n].lib()
        if lib is not None and any(KERNEL_ATTR.search(rust_lex(tree.read(f))[0]) for f in tree.tree_of(lib.src)[0]):
            out.append(n)
    return sorted(out)


def cmd_ship_bins(args: argparse.Namespace) -> int:
    """The shipped bins and the release features, `bin<TAB>features` a line (tools/release/late-commit.sh prints the scan command from them)."""
    if args.rev:
        proc = subprocess.run(["git", "show", f"{args.rev}:{RELEASE_BUILD}"], cwd=ROOT, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RecipeError(f"git show {args.rev}:{RELEASE_BUILD} failed: {proc.stderr.strip()}")
        rb = parse_release_build(proc.stdout)
    else:
        rb = read_release_build(ROOT)
    for b in rb.bins:
        print(f"{b}\t{','.join(rb.features)}")
    return 0


def cmd_ship_scan(args: argparse.Namespace) -> int:
    if len(args.logs) % 2:
        raise RecipeError("ship-scan: the logs come in pairs, BASE_LOG HEAD_LOG each")
    pairs = [(read_scan(args.logs[k]), read_scan(args.logs[k + 1])) for k in range(0, len(args.logs), 2)]
    tmp = tempfile.mkdtemp(prefix="recipes-shipscan-")
    try:
        tree = Tree(archive(args.rev, tmp) if args.rev else ROOT)
        rb = read_release_build(tree.root)
        kernel_libs = shipped_kernel_libs(tree, rb)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    scanned: set[str] = set()
    ok = True
    for base, head in pairs:
        for log in (base, head):
            if log.bin not in rb.bins:
                raise RecipeError(f"scan log {log.path}: a scan of {log.bin}, not of a shipped bin ({' '.join(rb.bins)}): "
                                  f"run {SCAN_SHIP_ENV}=1 tools/narrow-scan.sh BASE HEAD and pass <out>/<side>/{rb.bins[0]}.log")
            missing = sorted(set(kernel_libs) - set(log.bundles))
            if missing:
                print(f"ship-scan: note: {log.path} names no bundle of {' '.join(missing)}, which the release features link: "
                      f"a lib the bin never reaches has none, a scan taken at other features lacks it too", file=sys.stderr)
        try:
            v = scan_verdict(base, head)
            line, same = v.line(), v.kind == "identical"
        except RecipeError as err:
            # a --no-jit pair that is not identical is refused by scan_verdict (the add and occupancy classes need the JIT
            # columns); here any pair that is not identical blocks, so the refusal is the verdict
            msg = str(err)
            if " in a pair with a --no-jit scan" not in msg:
                raise
            line, same = "not identical: " + msg.split(" in a pair with a --no-jit scan")[0].split(f"{head.path}: ", 1)[-1], False
        print(line)
        scanned.add(base.bin)
        ok = ok and same
    left = [b for b in rb.bins if b not in scanned]
    if left:
        print(f"ship-scan: no scan pair for the shipped bin {' '.join(left)}", file=sys.stderr)
        return 1
    return 0 if ok else 1


def release_feature_problems(rb: ReleaseBuild, tree: Tree, recipes: dict[str, Recipe]) -> list[str]:
    """A recipe whose build writes a shipped bin (tools/release/build.sh's BINS) with any feature list but the release one,
    as a set: cargo reads the features of a build as a set, so the spelling and order are free. Only a build that
    writes the bin counts (BIN_WRITING_SUBS): a `cargo check` replaces no file another lane runs. A call is read as
    cargo reads it (resolve_call): a package with no `--bin` is every bin whose required features the list enables.
    A bin built beside it on the same line takes the release list too; a gate bin that needs other features goes on a
    cargo line of its own."""
    want = set(rb.features)
    out: list[str] = []
    for rname, inv, _, _ in sorted(build_calls(recipes), key=lambda c: recipes[c[0]].line):
        if inv.sub not in BIN_WRITING_SUBS:
            continue
        targets, _ = resolve_call(tree, inv)  # an unresolvable call is check()'s error, named there
        shipped = [n for p, k, n, *_ in targets if k == "bin" and p == rb.package and n in rb.bins]
        if not shipped:
            continue
        got = set(inv.features)
        how = ""
        if any("{{" in f for f in got):
            how = "a feature list from a recipe parameter, which this check cannot read"
        elif inv.no_default or inv.all_features:
            how = "--no-default-features or --all-features"
        elif got != want:
            how = f"features {','.join(sorted(got)) or 'none'}"
        if how:
            out.append(f"justfile:{recipes[rname].line} {rname}: builds {' '.join(shipped)} with {how}, not the release list "
                       f"{','.join(rb.features)} ({RELEASE_BUILD}): the gates of a shipped bin run the binary the release packs, and two lanes "
                       f"at two lists relink target/release/{shipped[0]} under each other. Build it with the release list on a cargo line of its own")
    return out


def ships_self_test(expect) -> None:
    """The shipped closure on a synthetic workspace: package bloomery-gpu-gates (lib, bin bloomery-serve behind `fa`, a gate
    bin) with optional libs liba (behind fa, with a build script) and libb (behind fb, not a release feature) and an
    unlinked libc. Each kind of input is inside, each look-alike beside it is not."""
    build_sh = ("#!/usr/bin/env bash\nBINS=(bloomery-serve)\nsource \"$ROOT/tools/arch.sh\"\n"
                "BLOOMERY_GIT_COMMIT=\"$COMMIT\" cargo oxide build --arch \"$ARCH\" -- -p bloomery-gpu-gates --features fa --release \"${bin_args[@]}\"\n")

    def make(tmp: str, name: str, serve_mods: str = "#[path = \"shared/seat.rs\"]\nmod seat;\nmod walked;\n") -> Tree:
        root = os.path.join(tmp, name)
        files = {
            "Cargo.toml": "", "Cargo.lock": "", "rust-toolchain.toml": "", ".cargo/config.toml": "", ".cargo/extra.toml": "",
            "tools/release/build.sh": build_sh, "tools/release/publish.sh": "", "tools/arch.sh": "", "tools/gate-paths.tsv": "", "justfile": "",
            "crates/gates/Cargo.toml": "", "crates/gates/src/lib.rs": "pub fn f() {}\n",
            "crates/gates/src/bin/serve.rs": serve_mods + "fn main() {}\n",
            "crates/gates/src/bin/walked.rs": "", "crates/gates/src/bin/shared/seat.rs": "",
            "crates/gates/src/bin/gate_x.rs": "mod only;\nfn main() {}\n", "crates/gates/src/bin/only.rs": "",
            "crates/liba/Cargo.toml": "", "crates/liba/src/lib.rs": "", "crates/liba/build.rs": "fn main() {}\n",
            "crates/libb/Cargo.toml": "", "crates/libb/src/lib.rs": "", "crates/libc/Cargo.toml": "", "crates/libc/src/lib.rs": "",
            "docs/x.md": "",
        }
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

        meta = {"workspace_root": root, "workspace_members": ["liba", "libb", "libc", "bloomery-gpu-gates"], "packages": [
            pkg("liba", "liba", [tgt("lib", "liba", "crates/liba/src/lib.rs"), tgt("custom-build", "build-script-build", "crates/liba/build.rs")]),
            pkg("libb", "libb", [tgt("lib", "libb", "crates/libb/src/lib.rs")]),
            pkg("libc", "libc", [tgt("lib", "libc", "crates/libc/src/lib.rs")]),
            pkg("bloomery-gpu-gates", "gates", [tgt("lib", "bloomery-gpu-gates", "crates/gates/src/lib.rs"),
                                              tgt("bin", "bloomery-serve", "crates/gates/src/bin/serve.rs", ["fa"]),
                                              tgt("bin", "gate_x", "crates/gates/src/bin/gate_x.rs")],
                {"fa": ["dep:liba"], "fb": ["dep:libb"]}, ["liba", "libb"]),
        ]}
        return Tree(root, meta)

    def refused(fn, why: str) -> bool:
        try:
            fn()
        except RecipeError as err:
            return why in str(err)
        return False

    with tempfile.TemporaryDirectory(prefix="recipes-ships-") as tmp:
        c = ship_closure(make(tmp, "t0"))
        expect(c.build == ReleaseBuild("bloomery-gpu-gates", ("fa",), ("bloomery-serve",)), f"ships: build.sh read as {c.build}")
        inside = {
            "crates/gates/src/bin/serve.rs": "the bin's source",
            "crates/gates/src/bin/walked.rs": "a `mod` file it reaches",
            "crates/gates/src/bin/shared/seat.rs": "a #[path] file it reaches",
            "crates/gates/src/lib.rs": "the bloomery-gpu-gates lib",
            "crates/liba/src/lib.rs": "a workspace crate the release features link",
            "crates/liba/build.rs": "that crate's build script",
            "crates/liba/Cargo.toml": "that crate's manifest",
            "crates/gates/Cargo.toml": "the scan package's manifest",
            "Cargo.toml": "the root manifest",
            "Cargo.lock": "Cargo.lock",
            "rust-toolchain.toml": "rust-toolchain.toml",
            ".cargo/config.toml": ".cargo/config.toml",
            ".cargo/extra.toml": "a file under .cargo/",
            "tools/release/publish.sh": "a file under tools/release/",
            "tools/arch.sh": "a script build.sh names",
        }
        got = dict(ship_hits([("", c)], list(inside)))
        for f, what in inside.items():
            expect(f in got, f"ships: {f} ({what}) is not inside the shipped closure")
        outside = ["crates/gates/src/bin/gate_x.rs", "crates/gates/src/bin/only.rs", "crates/libb/src/lib.rs", "crates/libb/Cargo.toml",
                   "crates/libc/src/lib.rs", "docs/x.md", "justfile", "tools/gate-paths.tsv"]
        stray = [f for f, _ in ship_hits([("", c)], outside)]
        expect(stray == [], f"ships: {stray} are inside the shipped closure (a gate bin and its module, a crate the release features do not link, docs, the justfile)")
        expect(ship_hits([("", c)], ["./crates/gates/src/bin/serve.rs"])[0][0] == "crates/gates/src/bin/serve.rs", "ships: a ./ path is not normalized")
        why = dict(ship_hits([("", c)], ["crates/gates/src/bin/shared/seat.rs"]))["crates/gates/src/bin/shared/seat.rs"]
        expect("bin bloomery-serve" in why, f"ships: the reason names the bin: {why}")
        # a file only the base's closure holds (the head dropped the `mod`): a delete or a move is not read against a tree without it
        head = ship_closure(make(tmp, "t1", serve_mods="mod walked;\n"))
        f = "crates/gates/src/bin/shared/seat.rs"
        expect(ship_hits([("", head)], [f]) == [], "ships: a file the head no longer reaches is inside its closure")
        got = ship_hits([("", head), ("in the closure of BASE only", c)], [f])
        expect(len(got) == 1 and "in the closure of BASE only" in got[0][1], f"ships: a file only the base's closure holds: {got}")
        # a `-p` build with no --bin builds every bin whose required features the list enables, the shipped one among them
        rb_syn = c.build

        def implicit(line: str) -> list[str]:
            return release_feature_problems(rb_syn, make(tmp, "t5"), {"r": Recipe("r", [line], [], "", 1)})

        expect(implicit("cargo build --release -p bloomery-gpu-gates --features fa") == [], "release features: an implicit build at the release list is a problem")
        got = implicit("cargo build --release -p bloomery-gpu-gates --features fa,fb")
        expect(len(got) == 1 and "bloomery-serve" in got[0], f"release features: an implicit build at another list gives {got}")
        expect(implicit("cargo build --release -p bloomery-gpu-gates") == [], "release features: an implicit build that cannot reach the shipped bin (its feature is off) is a problem")
        # build.sh that cannot be read is a named error, never an empty closure
        os.remove(os.path.join(tmp, "t0", RELEASE_BUILD))
        expect(refused(lambda: read_release_build(os.path.join(tmp, "t0")), RELEASE_BUILD), "ships: a missing build.sh is not a named error")
        for what, text, why in (("no BINS", build_sh.replace("BINS=(bloomery-serve)\n", ""), "BINS=("),
                                ("an empty BINS", build_sh.replace("(bloomery-serve)", "()"), "BINS=("),
                                ("no features", build_sh.replace(" --features fa", ""), "with --features")):
            root = os.path.join(tmp, "bad")
            os.makedirs(os.path.join(root, "tools/release"), exist_ok=True)
            with open(os.path.join(root, RELEASE_BUILD), "w", encoding="utf-8") as fh:
                fh.write(text)
            expect(refused(lambda: read_release_build(root), why), f"ships: build.sh with {what} is not a named error")

    # the release-feature rule and its overlap with the one-resolved-set rule, on a synthetic package that has the release
    # bin (glm5next, clef, mimo2 required; vision optional), the V4.1 server and gate bin, and a gate bin with no requirement
    rb = ReleaseBuild("bloomery-gpu-gates", ("glm5next", "clef", "vision", "mimo2"), ("bloomery-serve",))
    with tempfile.TemporaryDirectory(prefix="recipes-relfeat-") as tmp:
        root = os.path.join(tmp, "t")
        os.makedirs(os.path.join(root, "tools/release"))
        with open(os.path.join(root, RELEASE_BUILD), "w", encoding="utf-8") as fh:
            fh.write("BINS=(bloomery-serve)\ncargo oxide build -- -p bloomery-gpu-gates --features glm5next,clef,vision,mimo2 --release --bin bloomery-serve\n")
        feats = {"gpu": [], "deepseek41": ["gpu"], "glm5next": ["deepseek41"], "clef": ["gpu"], "vision": ["gpu"], "mimo2": ["gpu"]}
        bins = {"bloomery-serve": ["glm5next", "clef", "mimo2"], "bloomery-serve-ds41": ["deepseek41"], "bloomery-serve-qwen38": ["deepseek41"],
                "gate_ds41_serve": ["deepseek41"], "gate_x": []}
        meta = {"workspace_root": root, "workspace_members": ["bloomery-gpu-gates"], "packages": [{
            "id": "bloomery-gpu-gates", "name": "bloomery-gpu-gates", "manifest_path": os.path.join(root, "crates/gates/Cargo.toml"), "features": feats, "dependencies": [],
            "targets": [{"kind": ["lib"], "name": "bloomery-gpu-gates", "src_path": os.path.join(root, "crates/gates/src/lib.rs"), "required-features": []}]
                       + [{"kind": ["bin"], "name": n, "src_path": os.path.join(root, f"crates/gates/src/bin/{n}.rs"), "required-features": r} for n, r in bins.items()]}]}
        rtree = Tree(root, meta)

        def lines(*texts: str) -> dict[str, Recipe]:
            return {f"r{i}": Recipe(f"r{i}", [ln], [], "", i + 1) for i, ln in enumerate(texts)}

        def probs(*texts: str) -> list[str]:
            return release_feature_problems(rb, rtree, lines(*texts))

        def both(*texts: str) -> list[str]:
            return bin_problems(rtree, lines(*texts))

        def oxide(features: str, bins_: str, extra: str = "", sub: str = "build") -> str:
            return f"./tools/box.sh 'cargo oxide {sub} --arch sm_86 -- -p bloomery-gpu-gates{extra} --features {features} --release {bins_}'"

        release = oxide("glm5next,clef,vision,mimo2", "--bin bloomery-serve")
        expect(probs(release) == [], "release features: the release list is a problem")
        expect(probs(oxide("mimo2,vision,clef,glm5next", "--bin bloomery-serve --bin gate_x")) == [], "release features: the same set in another order is a problem")
        got = probs(oxide("glm5next,clef,mimo2", "--bin bloomery-serve --bin gate_x"))
        expect(len(got) == 1 and "r0" in got[0] and "bloomery-serve" in got[0] and "clef,glm5next,mimo2" in got[0], f"release features: a list without vision gives {got}")
        got = probs(oxide("deepseek41,vision", "--bin bloomery-serve-ds41 --bin gate_ds41_serve"))
        # 2026-10-13 (lead): bloomery-serve-ds41 is no bin of build.sh's BINS, so the release list is not its list (the first form of
        # this rule held it to it and expected one problem here); the one-set rule still holds its builds to one set, see below
        expect(got == [], f"release features: bloomery-serve-ds41 built with its own list gives {got}")
        expect(probs(oxide("glm5next,clef,vision,mimo2", "--bin bloomery-serve-ds41")) == [], "release features: bloomery-serve-ds41 at the release list is a problem")
        expect(probs(oxide("deepseek41", "--bin gate_ds41_serve --bin bloomery-serve-qwen38")) == [], "release features: a build of no shipped bin is a problem")
        expect(len(probs(oxide("{{FEATURES}}", "--bin bloomery-serve"))) == 1, "release features: a recipe-parameter list on a shipped bin is not named")
        expect(len(probs(oxide("glm5next,clef,vision,mimo2", "--bin bloomery-serve", " --no-default-features"))) == 1, "release features: --no-default-features on a shipped bin is not named")
        expect(len(probs("cargo build --release -p bloomery-gpu-gates --features glm5next --bin bloomery-serve")) == 0,
               "release features: a bin whose required features the list lacks is cargo's refusal and check()'s error, not this rule's")
        expect(len(probs("cargo build --release -p bloomery-gpu-gates --features glm5next,clef,mimo2 --bin bloomery-serve")) == 1, "release features: a plain cargo build with another list is not named")
        # a build that writes no bin replaces nothing another lane runs: the no-vision check shape is not named
        no_vision = oxide("glm5next,clef,mimo2", "--bin bloomery-serve --bin bloomery-serve-ds41", sub="check").replace("oxide check --arch sm_86 -- ", "check ")
        expect(probs(no_vision) == [] and both(no_vision) == [], f"release features: a cargo check of the shipped bin at another list is named: {probs(no_vision)} {both(no_vision)}")
        expect(len(probs(no_vision.replace("cargo check", "cargo build"))) == 1, "release features: the same line as a cargo build is not named")

        # the overlap with the one-resolved-set rule (bin_feature_problems): each catches what the other does not, and one
        # defect is named once. B = bin_feature_problems, R = release_feature_problems.
        narrow = oxide("glm5next,clef,mimo2", "--bin bloomery-serve")
        gate_a, gate_b = oxide("deepseek41", "--bin gate_ds41_serve"), oxide("deepseek41,vision", "--bin gate_ds41_serve")
        # B only: a gate bin at two sets (R holds the listed bins only)
        expect(len(bin_feature_problems(rtree, lines(gate_a, gate_b))) == 1 and probs(gate_a, gate_b) == [] and len(both(gate_a, gate_b)) == 1,
               f"bin rules: a gate bin at two sets is not named once, by the one-set rule: {both(gate_a, gate_b)}")
        # B only: the V4.1 server is no shipped bin, so two sets of it are the one-set rule's
        ds41_a, ds41_b = oxide("deepseek41", "--bin bloomery-serve-ds41"), oxide("deepseek41,vision", "--bin bloomery-serve-ds41")
        got = both(ds41_a, ds41_b)
        expect(len(got) == 1 and got[0].startswith("bloomery-serve-ds41 (bloomery-gpu-gates, release): ") and probs(ds41_a, ds41_b) == [],
               f"bin rules: bloomery-serve-ds41 at two sets is not named once, by the one-set rule: {got}")
        # R only: the shipped bin at one consistent set that is not the release list (B sees one set)
        expect(bin_feature_problems(rtree, lines(narrow, narrow)) == [] and len(probs(narrow, narrow)) == 2 and len(both(narrow, narrow)) == 2,
               f"bin rules: the shipped bin at one set that is not the release list is not named by the release rule alone: {both(narrow, narrow)}")
        # one recipe narrowed among release ones: B alone would name it too; the pair names it once, and the line is R's
        trio = (release, release, narrow)
        got = both(*trio)
        expect(len(bin_feature_problems(rtree, lines(*trio))) == 1 and len(got) == 1 and "r2" in got[0] and "not the release list" in got[0],
               f"bin rules: one recipe narrowed among release ones is not named once, by the release rule: {got}")
        # a target dir of its own exempts a build from the one-set rule (B), not a shipped bin from the release list (R)
        own = oxide("glm5next,clef,mimo2", "--bin bloomery-serve", sub="build --cargo-target-dir target/b")
        expect(bin_feature_problems(rtree, lines(release, own)) == [] and len(both(release, own)) == 1, f"bin rules: a shipped bin in a target dir of its own at another list: {both(release, own)}")
        # the release rule green: every build spells the release list, one resolved set, so the one-set rule is green too
        everything = (release, oxide("vision,mimo2,clef,glm5next", "--bin bloomery-serve --bin gate_x"), oxide("glm5next,clef,vision,mimo2", "--bin bloomery-serve-ds41"))
        expect(bin_feature_problems(rtree, lines(*everything)) == [] and both(*everything) == [], f"bin rules: builds that all spell the release list are named: {both(*everything)}")
        # a build.sh that cannot be read is a named error and leaves the one-set rule whole
        os.remove(os.path.join(root, RELEASE_BUILD))
        got = both(gate_a, gate_b)
        expect(len(got) == 2 and RELEASE_BUILD in got[0] and "gate_ds41_serve" in got[1], f"bin rules: an unreadable build.sh is not a named error beside the one-set rule: {got}")


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
# The cuda-oxide repository under its name (NVIDIA/cuda-rust) and the retired fork's, which the lock of a tree before the fork was dropped still names.
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
            r"|SYNC_FILE_RANGE_[A-Z_]+|RUSAGE_THREAD|renameat2|RENAME_(?:NOREPLACE|EXCHANGE|WHITEOUT)|prctl|PR_[A-Z_]+"
            r"|MADV_POPULATE_[A-Z_]+|O_DIRECT"
            r"|MADV_(?:NO)?HUGEPAGE|MAP_POPULATE|MAP_HUGETLB)\b"
        ),
        "a Linux-only libc item",
    ),
    # `RENAME_EXCL` and `RENAME_SWAP` are not in the list above: macOS has them
    # (`renamex_np`), so only Linux's own rename flags are a Linux-only item.
    (re.compile(r"\bAdvice::(?:PopulateRead|PopulateWrite|HugePage|NoHugePage)\b"), "memmap2's Linux-only advice"),
    (re.compile(r"\"/(?:proc|sys)/"), "a Linux /proc or /sys path"),
]
_CFG_ATTR = re.compile(r"#(!?)\s*\[\s*cfg\s*\(")
_RAW_STR = re.compile(r"(?:b|c)?r(#*)\"")
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
        if c == "/" and text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            _blank(code, i, j)
            _blank(shape, i, j)
            i = j
            continue
        if c == "/" and text.startswith("/*", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if text[j] == "/" and text.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif text[j] == "*" and text.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            _blank(code, i, j)
            _blank(shape, i, j)
            i = j
            continue
        if c in "bcr" and (i == 0 or not _IDENT_CH.match(text[i - 1])):
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
    `#[cfg(…)]` false there (a file under a false `#![cfg(…)]`) blanked; newlines kept, so line numbers hold.
    The item's own end is read (`items`), so a fenced item whose signature holds a comma — a `Result<A, B>`
    return type — is blanked whole and not only up to that comma."""
    return _cfg_blanked(text, test, _MAC_CFG_KEYS, _MAC_CFG_NAMES, items=True)[0]


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


LockKey = tuple[str, str, str, str]  # a [[package]] entry's (name, version, source, checksum): a workspace member has no source


@dataclass
class Lock:
    roots: set[str]  # device roots by package name
    deps: dict[str, set[str]]  # package name -> the names it depends on, every version merged
    entries: dict[LockKey, tuple[str, ...]] = field(default_factory=dict)  # every package -> its dependencies as the file writes them
    by_name: dict[str, list[LockKey]] = field(default_factory=dict)
    rest: str = ""  # canon of every top-level key but `package` (the format version, `[metadata]`, `[[patch.unused]]`)

    def resolve(self, owner: LockKey, dep: str) -> list[LockKey]:
        """The packages a dependency string of `owner` names: `name`, `name version` or `name version (source)`, as the file
        writes it (the version and the source only where the name alone is ambiguous). A string no package matches is a named error."""
        name, version, source = (dep.split(" ", 2) + ["", ""])[:3]
        source = source[1:-1] if source.startswith("(") and source.endswith(")") else source
        hit = [k for k in self.by_name.get(name, ()) if (not version or k[1] == version) and (not source or k[2] == source)]
        if not hit:
            raise RecipeError(f"Cargo.lock: {owner[0]} {owner[1]} depends on `{dep}`, which no package of the file is")
        return hit


def load_lock(root: str) -> Lock:
    import tomllib

    path = os.path.join(root, "Cargo.lock")
    if not os.path.isfile(path):
        raise RecipeError(f"no Cargo.lock in {root} — the device roots and the external closure are read from it")
    with open(path, "rb") as fh:
        try:
            data = tomllib.load(fh)
        except tomllib.TOMLDecodeError as err:
            raise RecipeError(f"{path} does not parse as TOML ({err})") from err
    roots: set[str] = set()
    deps: dict[str, set[str]] = {}
    entries: dict[LockKey, tuple[str, ...]] = {}
    by_name: dict[str, list[LockKey]] = {}
    for p in data.get("package", []):
        name = p["name"]
        if any(s in p.get("source", "") for s in DEVICE_GIT_SOURCES) or name in DEVICE_NAMED_ROOTS:
            roots.add(name)
        deps.setdefault(name, set()).update(d.split(" ", 1)[0] for d in p.get("dependencies", []))
        key = (name, p.get("version", ""), p.get("source", ""), p.get("checksum", ""))
        entries[key] = tuple(p.get("dependencies", []))
        by_name.setdefault(name, []).append(key)
    return Lock(roots, deps, entries, by_name, canon({k: v for k, v in data.items() if k != "package"}))


# ---- Cargo.lock as a selecting input: by package, for what each recipe's host builds ----
#
# `affected` reads Cargo.lock per recipe (`LockDiff`). The diff names the packages that moved: a key (name, version,
# source, checksum) that appeared or went, and those on both sides that depend on another set of names (`shlex` on one
# side and `shlex 1.3.0` on the other is one name); a workspace member's own entry counts, so a version bump or a new
# member is a package of the diff. A recipe's build on its host (HOSTS: the box, or the Mac for a recipe that runs
# MAC_GATE) reaches what a walk finds from the workspace packages its targets link (the packages of its manifest views)
# over the lock's edges: a member's edge counts when some dependency of that name sits in a plain table or in a
# `[target.<key>]` table `table_read` does not say False for on that host (all features; normal, build and dev), and a
# member outside the recipe's closure is not entered. The lock lists every platform's edges of an external package, so
# all of them are followed, except a name one side's lock has and the other's has not for a package of the same key: a
# feature turns an optional dependency on somewhere (`sha2`'s new `sha2-asm`, asked for under an aarch64 table), and
# nothing says the host enables it. A package of one version changes its edges only through a feature some manifest asks
# of it (the recipes that read that manifest are selected by table) or through another package of the diff (which the
# walk reaches or not by the same rule), so dropping those names loses no input of the host.
# A recipe is selected when the walk over the lock of A and the walk over the lock of B reach different packages: a
# package every target reaches selects every recipe by that same rule, and one no recipe links selects none. What the
# reader cannot decide keeps the file whole for every cargo recipe, and a note says why: a side with no lock or one that
# does not parse, an edge the lock does not resolve, a top-level key beside `package` that moved, a diff that names no
# package (a comment, the layout), a recipe whose inputs have errors. The key still hashes the file whole.


def _dep_names(deps: tuple[str, ...]) -> set[str]:
    return {d.split(" ", 1)[0] for d in deps}


def lock_changed(a: Lock, b: Lock, members: set[str]) -> list[str]:
    """The external packages the diff of two locks names, as `name version`, sorted. A workspace member is no external
    package: its own tables are read by `TableReads`."""
    out: set[str] = set()
    for k in a.entries.keys() ^ b.entries.keys():
        if k[2] or k[0] not in members:
            out.add(f"{k[0]} {k[1]}")
    for k in a.entries.keys() & b.entries.keys():
        if (k[2] or k[0] not in members) and _dep_names(a.entries[k]) != _dep_names(b.entries[k]):
            out.add(f"{k[0]} {k[1]}")
    return sorted(out)


def lock_moved(a: Lock, b: Lock) -> list[str]:
    """Every package the diff of two locks names, as `name version`, sorted: the ones that appeared or went, and the ones
    on both sides with another set of dependency names. A workspace member's own entry counts."""
    out = {f"{k[0]} {k[1]}" for k in a.entries.keys() ^ b.entries.keys()}
    out |= {f"{k[0]} {k[1]}" for k in a.entries.keys() & b.entries.keys() if _dep_names(a.entries[k]) != _dep_names(b.entries[k])}
    return sorted(out)


class LockWalk:
    """What the workspace packages of `tree` reach in `lock` on `host` (the block above), `other` being the lock of the
    diff's other side. The edges of a package are resolved once."""

    def __init__(self, tree: Tree, lock: Lock, other: Lock, host: str):
        self.tree, self.lock, self.other, self.host = tree, lock, other, host
        self._edges: dict[LockKey, list[LockKey]] = {}

    def member(self, k: LockKey) -> bool:
        return not k[2] and k[0] in self.tree.packages

    def edges(self, k: LockKey) -> list[LockKey]:
        if k not in self._edges:
            if self.member(k):
                names = {d.pkg for d in self.tree.packages[k[0]].deps if d.target is None or table_read(d.target, self.host) is not False}
                kept = [dep for dep in self.lock.entries[k] if dep.split(" ", 1)[0] in names]
            else:
                theirs = self.other.entries.get(k)
                common = None if theirs is None else _dep_names(theirs)
                kept = [dep for dep in self.lock.entries[k] if common is None or dep.split(" ", 1)[0] in common]
            self._edges[k] = [t for dep in kept for t in self.lock.resolve(k, dep)]
        return self._edges[k]

    def reach(self, roots: Iterable[str] | None = None) -> frozenset[LockKey]:
        """The packages of the lock reached from the workspace packages `roots` (None: every member): those and, of the
        members, only the ones in `roots`; every external package an edge leads to."""
        allowed = set(self.tree.packages) if roots is None else set(roots) & set(self.tree.packages)
        seen: set[LockKey] = set()
        todo: list[LockKey] = []

        def take(keys: Iterable[LockKey]) -> None:
            for k in keys:
                if k not in seen and (not self.member(k) or k[0] in allowed):
                    seen.add(k)
                    todo.append(k)

        for m in sorted(allowed):
            take(k for k in self.lock.by_name.get(m, ()) if not k[2])
        while todo:
            take(self.edges(todo.pop()))
        return frozenset(seen)


class LockDiff:
    """Cargo.lock of one diff, read per recipe: the packages that moved (`moved`) and, for a recipe's host and the
    workspace packages it links, those one side's walk reaches and the other's does not (`differs`). Raises
    `RecipeError` for a lock that does not parse or an edge it does not resolve: the caller reads the file whole."""

    def __init__(self, a: Side, b: Side):
        self.la, self.lb = load_lock(a.tree.root), load_lock(b.tree.root)
        for lock in (self.la, self.lb):
            for k, deps in lock.entries.items():
                for dep in deps:
                    lock.resolve(k, dep)
        self.moved = lock_moved(self.la, self.lb)
        self.rest_moved = self.la.rest != self.lb.rest
        self._walks = {(host, side): LockWalk(t.tree, lock, other, host)
                       for host in HOSTS for side, t, lock, other in (("a", a, self.la, self.lb), ("b", b, self.lb, self.la))}
        self._differs: dict[tuple[str, frozenset[str] | None], list[str]] = {}

    def differs(self, host: str, roots: frozenset[str] | None = None) -> list[str]:
        """The packages (`name version`, sorted) that the walk over A's lock reaches and B's does not, or the reverse, from
        the workspace packages `roots` (None: every member) on `host`."""
        key = (host, roots)
        if key not in self._differs:
            ra, rb = (self._walks[host, s].reach(roots) for s in ("a", "b"))
            self._differs[key] = sorted({f"{k[0]} {k[1]}" for k in ra ^ rb})
        return self._differs[key]


def lock_outside(a: Side, b: Side, diff: LockDiff | None = None) -> dict[str, str]:
    """host -> the note `affected` prints, for each host whose build no external package of the diff of Cargo.lock reaches
    (the two walks from every workspace member reach the same packages). A host missing from the answer reads the file as
    a global input: a side with no lock, a diff that names no external package, and a diff the host reaches."""
    if not (a.tree.exists(LOCK_FILE) and b.tree.exists(LOCK_FILE)):
        return {}
    diff = diff if diff is not None else LockDiff(a, b)
    names = lock_changed(diff.la, diff.lb, set(a.tree.packages) | set(b.tree.packages))
    if not names:
        return {}
    shown = ", ".join(names[:8]) + (f" and {len(names) - 8} more" if len(names) > 8 else "")
    return {host: f"{LOCK_FILE}: {shown} changed, not in the {label} build" for host, (_, _, _, label) in HOSTS.items() if not diff.differs(host)}


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
        # what the target compiles, not what it reads: a path literal names a file opened at run time (it moves the
        # gate in narrow(), it is no source to scan), and dep-info is the box's
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
HOST_EXTRA_UNITS: dict[str, str] = {}
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
        if "recipes.py --self-test" in extras[n] and "check-recipes.sh" in host_check_scripts(tree.root) and "check-recipes.sh" not in checks:
            problems.append(f"host nightly: HOST_EXTRA_UNITS names {n}, which runs the recipes.py self-test that check-recipes already runs")
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


def build_calls(recipes: dict[str, Recipe]) -> Iterator[tuple[str, Invocation, str, bool]]:
    """(recipe, invocation, the call as a skip line prints it, whether a recipe parameter decides it) of every
    cargo call, recipes in name order: the one reader of what the recipes build, for `combos` and the bin check."""
    for rname in sorted(recipes):
        for inv in recipe_commands(recipes[rname]).invocations:
            call = " ".join(["cargo"] + ["oxide"] * inv.oxide + [inv.sub] + ["--workspace"] * inv.workspace + [f"-p {p}" for p in inv.packages] + [f"--features {','.join(inv.features)}"] * bool(inv.features))
            yield rname, inv, call, _has_param(inv)


def resolve_call(tree: Tree, inv: Invocation) -> tuple[list[tuple[str, str, str | None, list[str], tuple[str, ...], tuple[str, ...]]], list[str]]:
    """[(package, kind, name, the package's own --features words, enabled features after the closure, `dep/feature`
    words for other packages)] and the errors, of one call the tree can resolve."""
    out = []
    targets, errs = tree.resolve(inv)
    for pname, kind, name, feats in targets:
        enabled, _, _ = tree.feature_closure(tree.packages[pname], feats, not inv.no_default, inv.all_features)
        # a `dep/feature` word names a dependency of this package; one for another package of the same call
        # (`-p a -p b --features b/x`) is that package's own word, which a shape of `a` alone would be refused for
        others = tuple(sorted({f for f in inv.features if "/" in f and f.split("/", 1)[0] != pname and tree.depends_on(tree.packages[pname], f.split("/", 1)[0])}))
        out.append((pname, kind, name, feats, tuple(sorted(enabled)), others))
    return out, errs


def build_shapes(tree: Tree, recipes: dict[str, Recipe]) -> tuple[list[Combo], list[tuple[str, str]], int]:
    """(the shapes in package and mode order, [(recipe, why)] left out by name, the count of distinct
    (package, target, mode, features) tuples). A call the tree cannot resolve is a RecipeError naming it."""
    shapes: dict[tuple, Combo] = {}
    skips: list[tuple[str, str]] = []
    tuples: set[tuple] = set()
    errors: list[str] = []
    for rname, inv, call, param in build_calls(recipes):
        if rname in MAC_CHECKED:
            skips.append((rname, f"{call}: {MAC_CHECKED[rname]} runs this recipe's command"))
            continue
        if param:
            skips.append((rname, f"{call}: a recipe parameter ({{{{…}}}}) decides its package, features or targets at run time"))
            continue
        targets, errs = resolve_call(tree, inv)
        errors += [f"{rname}: {e}" for e in errs]
        for pname, kind, name, feats, enabled, others in targets:
            if kind == "doctest":
                skips.append((rname, f"doctests of {pname}: rustdoc compiles them and cargo check has no doctest mode; the lib they import is checked in {pname}'s build shape"))
                kind = "lib"
            test = kind in ("libtest", "test", "bench") or (kind == "bin" and inv.sub in ("test", "bench"))
            sel = ("lib", None) if kind in ("lib", "libtest") else (kind, name)
            key = (pname, test, enabled, others, inv.no_default, inv.all_features)
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
# one binary, one feature set: two recipes that build a bin to one path build it the same way
# ----------------------------------------------------------------------------------------------
# The rule is the module docstring's «bin feature sets» paragraph; `just check-recipes` runs it.

# The subcommands that write the bin to <target dir>/<profile>/<bin>. A test or bench build writes a
# hash-named file under deps/, and check and clippy write none, so none of them replaces a bin.
BIN_WRITING_SUBS = ("build", "run", "rustc")
# cargo's two profiles that inherit another write to that one's directory.
_PROFILE_DIR = {"test": "dev", "bench": "release"}


def build_dir(inv: Invocation) -> str:
    """The directory a build writes its bins to, as the call sets it: the target dir (`--target-dir`,
    oxide's `--cargo-target-dir`, the `CARGO_TARGET_DIR` of the call's prefix or an `export` before it;
    `target` when none, left out of the name), below it `--target <triple>`, then the profile (`--release`,
    `--profile`, else dev). Spellings of one directory (`target`, `./target/`) are one name."""
    root = os.path.normpath(inv.target_dir) if inv.target_dir else "target"
    return "/".join(x for x in ("" if root == "target" else root, inv.target, _PROFILE_DIR.get(inv.profile, inv.profile)) if x)


def bin_feature_problems(tree: Tree, recipes: dict[str, Recipe], skip: frozenset[tuple[str, str]] = frozenset()) -> list[str]:
    """A (package, bin, directory) that two builds give different resolved feature sets: they write one file,
    <dir>/<profile>/<bin>, and cuda-host reads the PTX bundle from `current_exe()` at each module load, so the
    later build replaces the file a running gate loads its kernels from (2026-10-12: gate_glm5next_e2e built
    with `glm5next,vision` by one recipe and with `glm5next` by another, while the first ran). The set is the
    enabled features after the package's closure plus its `dep/feature` words, so `gpu,deepseek41` and
    `deepseek41` agree. The two builds may be one recipe's, one after the other (2026-10-13: gate-ptx-spill's
    oxart_jit, built three times): the file is replaced under whichever lane runs it just the same, so a recipe's
    own order is no exemption. The key is the directory the call writes to, so builds that land in different
    directories pass. A call whose features or target directory come from a recipe parameter is left out: they
    are known at run time only. A (package, bin) in `skip` is another rule's to name (bin_problems)."""
    built: dict[tuple[str, str, str], dict[tuple[str, ...], list[str]]] = {}
    for rname, inv, _, param in build_calls(recipes):
        where = build_dir(inv)
        if inv.sub not in BIN_WRITING_SUBS or param or "{{" in where:
            continue
        targets, _ = resolve_call(tree, inv)  # an unresolvable call is check()'s error, named there
        for pname, kind, name, _, enabled, others in targets:
            if kind == "bin" and (pname, name or "") not in skip:
                users = built.setdefault((pname, name or "", where), {}).setdefault(tuple(sorted(enabled + others)), [])
                if rname not in users:
                    users.append(rname)
    problems: list[str] = []
    for (pname, name, where), sets in sorted(built.items()):
        if len(sets) < 2:
            continue
        # the widest set first (the union's candidate), the recipes of a set sorted: one line whatever the order
        groups = " vs ".join(f"{', '.join(sorted(rs))} {{{','.join(fs)}}}" for fs, rs in sorted(sets.items(), key=lambda kv: (-len(kv[0]), kv[0])))
        problems.append(
            f"{name} ({pname}, {where}): {groups} "
            f"(one {name} file in {where}: a rebuild replaces the PTX bundle a running gate reads; build the union in each, or give a build its own target dir where no runner reads target/release)"
        )
    return problems


def bin_problems(tree: Tree, recipes: dict[str, Recipe]) -> list[str]:
    """check()'s two bin rules, each defect named once. A bin tools/release/build.sh ships is held
    to the release list by release_feature_problems, which names the recipe that departs from it; bin_feature_problems
    steps aside for it. Where the first is green every build of the bin spells the release words with the default
    features, which is one resolved set, so the second would be green as well; where it is red, the recipe it names is
    the answer, and the second's advice (build the union, or give a build its own target dir) is wrong for a shipped bin.
    Every other bin is held to one resolved set. A build.sh that cannot be read is a named error and leaves the second
    rule whole."""
    try:
        rb = read_release_build(tree.root)
    except RecipeError as err:
        return [str(err)] + bin_feature_problems(tree, recipes)
    owned = frozenset((rb.package, b) for b in rb.bins)
    return release_feature_problems(rb, tree, recipes) + bin_feature_problems(tree, recipes, owned)


# ----------------------------------------------------------------------------------------------
# the shape input key (`combos --base`/`--ledger`): what a round's loop may skip
# ----------------------------------------------------------------------------------------------

# Files every shape's check reads whatever its package: the workspace's dependency resolution, the
# toolchain pin and the rustflags owner (a moved pin or flag re-checks every shape), and this
# tool and its runner — their sha moves every key when they change, the gate ledger's rule.
SHAPE_KEY_META = (
    "Cargo.toml",
    "Cargo.lock",
    "rust-toolchain.toml",
    ".cargo/config.toml",
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


# The Mac static tier's jobs (tools/static-sched.py's JOBS file: NAME, ATTRS, KEYSPEC, COMMAND a line). A job's key hashes
# its command, the caller environment and the content of every file its KEYSPEC names, so a green row in the Mac static
# ledger (read_shape_ledger's format) skips it only while all of them are unchanged; the header differs from a shape
# key's, so the two kinds never share a key. KEYSPEC is `-` (never skipped) or a comma list of
#   tree         every shipped file (shipped_files): a job that reads the tree at large
#   rev:REF      the commit REF names in this checkout (`git rev-parse`): a job that reads a base revision
STATIC_JOB_KIND = "mac-static-job-v1"
STATIC_JOB_TOOLS = (("bash", ["bash", "--version"]), ("git", ["git", "--version"]))


class Job:
    def __init__(self, name: str, attrs: str, keyspec: str, command: str, where: str):
        if not name or "/" in name or name.startswith("."):
            raise ValueError(f"{where}: job name {name!r} is not a file name")
        self.name, self.keyspec, self.command = name, keyspec, command
        self.after: str | None = None
        self.alone = False
        self.cpus = 1
        for a in [] if attrs == "-" else attrs.split(","):
            if a.startswith("after="):
                self.after = a[len("after=") :]
            elif a == "alone":
                self.alone = True
            elif a.startswith("cpus="):
                try:
                    self.cpus = int(a[len("cpus=") :])
                except ValueError:
                    raise ValueError(f"{where}: {name}: cpus takes a whole number, not {a!r}")
                if self.cpus < 1:
                    raise ValueError(f"{where}: {name}: cpus must be 1 or more, not {self.cpus}")
            else:
                raise ValueError(f"{where}: {name}: attribute {a!r} is not after=NAME, alone or cpus=K")
        if not command.strip():
            raise ValueError(f"{where}: {name}: no command")


def read_jobs(path: str) -> list[Job]:
    """The jobs of a tools/static-sched.py JOBS file, in its order: the one reader, for the scheduler that runs them and
    `static_keys` that keys them. A malformed line, a name given twice or an after= that names no job is a ValueError naming it."""
    jobs: list[Job] = []
    with open(path, encoding="utf-8") as fh:
        for n, ln in enumerate(fh, 1):
            ln = ln.rstrip("\n")
            if not ln.strip() or ln.startswith("#"):
                continue
            f = ln.split("\t")
            if len(f) != 4:
                raise ValueError(f"{path}:{n}: {len(f)} fields, want 4 (NAME ATTRS KEYSPEC COMMAND)")
            jobs.append(Job(*f, where=f"{path}:{n}"))
    names = [j.name for j in jobs]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"{path}: job names given twice: {' '.join(dup)}")
    for j in jobs:
        if j.after is not None and j.after not in names:
            raise ValueError(f"{path}: {j.name}: after={j.after} names no job")
    return jobs


# Every variable through which the caller's environment hands cargo rustflags, for the two triples the static tier builds
# (the box's, which the cross check targets, and the Mac's own).
STATIC_RUSTFLAGS_ENV = ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_RUSTFLAGS", "RUSTDOCFLAGS",
                        *(f"CARGO_TARGET_{t.upper().replace('-', '_')}_RUSTFLAGS" for t in ("x86_64-unknown-linux-gnu", MAC_TARGET)))


def static_job_env(environ: dict[str, str]) -> list[str]:
    """The caller-environment lines every job key carries: MAC_ENV's values, the rustflags the caller's environment hands cargo
    (they change what a job that runs cargo reports: `-A warnings` must not share a green row with the unset flags),
    PYTHONOPTIMIZE (it strips the assertions of the python tools a job runs), and the tools the jobs run."""
    rows = [f"mac-env\t{v}\t{environ.get(v)!r}" for v in (*MAC_ENV, *STATIC_RUSTFLAGS_ENV, "PYTHONOPTIMIZE")]
    rows += [f"mac-tool\tpython3\t{platform.python_version()}", f"mac-tool\tos\t{platform.system()} {platform.machine()}"]
    for name, argv in STATIC_JOB_TOOLS:
        try:
            out = subprocess.run(argv, capture_output=True, text=True).stdout.split("\n")[0].strip()
        except OSError:
            out = "missing"
        rows.append(f"mac-tool\t{name}\t{out or 'unknown'}")
    return rows


def static_job_inputs(spec: str, root: str, shipped: list[str]) -> tuple[list[str], list[str]]:
    """(the files the KEYSPEC names, relative and sorted; the revisions it names as `rev\tREF\tsha` lines). Raises
    RecipeError naming an item it does not take (tree and rev:REF are the two), or a REF this checkout cannot resolve."""
    files: set[str] = set()
    revs: list[str] = []
    for item in spec.split(","):
        kind, _, arg = item.partition(":")
        if kind == "tree" and not arg:
            files |= set(shipped)
        elif kind == "rev" and arg:
            r = subprocess.run(["git", "-C", root, "rev-parse", "--verify", "--quiet", f"{arg}^{{commit}}"], capture_output=True, text=True)
            if r.returncode != 0:
                raise RecipeError(f"static-key: {item}: this checkout has no commit {arg}")
            revs.append(f"rev\t{arg}\t{r.stdout.strip()}")
        else:
            raise RecipeError(f"static-key: keyspec item {item!r} is not tree or rev:REF")
    return sorted(files), revs


def static_job_key(name: str, command: str, env: list[str], files: list[str], revs: list[str], root: str, memo: dict[str, str]) -> str:
    h = hashlib.sha256()
    h.update(f"{STATIC_JOB_KIND}\n{name}\n{command}\n".encode())
    for ln in env + revs:
        h.update(f"{ln}\n".encode())
    for rel in files:
        if rel not in memo:
            path = os.path.join(root, rel)
            if os.path.islink(path):
                memo[rel] = f"link {os.readlink(path)}\t0"
            elif os.path.isfile(path):
                # the executable bits: git tracks them, and a job that runs the file directly reads them
                memo[rel] = f"{sha256_file(path)}\t{os.stat(path).st_mode & 0o111}"
            else:
                memo[rel] = "missing\t0"
        h.update(f"{memo[rel]}\t{rel}\n".encode())
    return h.hexdigest()


def static_keys(jobs_path: str, root: str, environ: dict[str, str]) -> list[tuple[str, str, str, list[str]]]:
    """(name, key, note, files) a job of the JOBS file, in its order; the key is `-` for a `-` KEYSPEC."""
    shipped = shipped_files(root)
    env = static_job_env(environ)
    memo: dict[str, str] = {}
    out = []
    try:
        jobs = read_jobs(jobs_path)
    except ValueError as err:
        raise RecipeError(f"static-key: {err}") from err
    for j in jobs:
        if j.keyspec == "-":
            out.append((j.name, "-", "never skipped", []))
            continue
        files, revs = static_job_inputs(j.keyspec, root, shipped)
        out.append((j.name, static_job_key(j.name, j.command, env, files, revs, root, memo), f"{j.keyspec}: {len(files)} files", files))
    return out


def cmd_static_key(args: argparse.Namespace) -> int:
    rows = static_keys(args.jobs, ROOT, dict(os.environ))
    if args.files:
        hit = [r for r in rows if r[0] == args.files]
        if not hit:
            raise RecipeError(f"static-key: {args.jobs} has no job {args.files}")
        print("\n".join(hit[0][3]))
        return 0
    for name, key, note, _ in rows:
        print(f"{name}\t{key}\t{note}")
    return 0


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
            files, a_rev, b_rev, a_label = changed_files(args.base)
        except RecipeError as err:
            raise RecipeError(f"combos --base {args.base}: {err}")
        changed = set(files)
        base_note = f" since {a_label}" + ("" if b_rev is None else f"..{b_rev[:9]}")
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
# The box's target, x86_64-unknown-linux-gnu, where every gate-* recipe runs; any other key is unknown. A recipe
# that runs tools/mac-gate.sh (MAC_GATE) runs on an Apple Silicon Mac instead, so its tests are read under the
# Mac's keys (_MAC_CFG_KEYS).
MAC_GATE = "tools/mac-gate.sh"
BOX_TARGET = "x86_64-unknown-linux-gnu"
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
# The two hosts a gate recipe builds on, by name: (triple, cfg keys, cfg names, how a note names the build). A recipe runs on the
# box unless it runs MAC_GATE (`recipe_host`).
HOSTS = {
    "box": (BOX_TARGET, _BOX_CFG_KEYS, _BOX_CFG_NAMES, "x86_64-linux"),
    "mac": (MAC_TARGET, _MAC_CFG_KEYS, _MAC_CFG_NAMES, "aarch64-macos"),
}
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
                if parts and named == ["ignore"] and re.match(r'^ignore\s*=\s*".*"$', parts[1]):
                    # A conditional ignore the box can decide: the test is
                    # skipped on the platform the predicate names and runs on
                    # the one it names not. The box is the platform this scan
                    # reads, so the ignore applies there when the predicate
                    # holds there; a predicate the box cannot decide (a
                    # feature) is refused with the rest.
                    pred = self._cfg_of(rel, aline, f"cfg({parts[0]})", errs)
                    if pred is None:
                        return
                    v = eval_cfg(pred[3], True, _BOX_CFG_KEYS, _BOX_CFG_NAMES)
                    if v is True:
                        ignored = True
                    elif v is None:
                        errs.append(f"{rel}:{aline}: #[{text}] — a conditional ignore whose predicate this checker cannot decide for the box")
                        return
                elif not parts or any(n in ("cfg", "ignore", "path", "test") or n.endswith("::test") for n in named):
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
    mac: bool = False  # the recipe runs MAC_GATE: its tests run on the Mac, under the Mac's cfg


def orphan_tests(tree: Tree, recipes: dict[str, Recipe], prefixes: tuple[str, ...] = TEST_RUNNERS) -> tuple[list[str], list[str], int]:
    """(orphans, errors, tests checked): every `#[test]` of every workspace target that no cargo test call of
    a recipe named with `prefixes` runs where the recipe runs (the box, or the Mac for a recipe that runs
    MAC_GATE), one `path:line name — why` line each; and every input it cannot decide, by file and line."""
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
        mac = any(MAC_GATE in line for line in recipes[rname].lines)
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
                runs.setdefault(key, []).append(TestRun(rname, enabled, args, mac))
    # A cfg name a workspace build script both declares (`rustc-check-cfg=cfg(NAME)`) and emits (`rustc-cfg=NAME`)
    # is set by the box build's target (a card's arch): a test under it exists only in a build where its path
    # holds, and runs there. A name only declared, a name another package's build script emits (cargo applies a
    # build script's cfgs to its own package only), or a run on the Mac stays unknown.
    build_cfgs: dict[str, set[str]] = {pname: set() for pname in tree.packages}
    for pname, pkg in tree.packages.items():
        for t in pkg.targets:
            if t.kind == "build":
                with open(os.path.join(tree.root, t.src), encoding="utf-8") as fh:
                    text = fh.read()
                build_cfgs[pname] |= set(re.findall(r"rustc-check-cfg=cfg\((\w+)\)", text)) & set(re.findall(r"rustc-cfg=(\w+)", text))
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
                used = sorted(n for n in build_cfgs[pname] if any(re.search(rf"\b{n}\b", c[2]) for c in item.cfgs))
                builds = [dict(zip(used, vals)) for vals in itertools.product((True, False), repeat=len(used))]
                for run in cands:
                    keys, names = (_MAC_CFG_KEYS, _MAC_CFG_NAMES) if run.mac else (_BOX_CFG_KEYS, _BOX_CFG_NAMES)
                    for build in builds if not run.mac else [{}]:
                        bad = None
                        for f, ln, text, pred in item.cfgs:
                            v = eval_cfg(pred, True, keys, {**names, **build}, run.features)
                            if v is None:
                                bad = "?"
                                break
                            if v is False:
                                feats = "no feature" if not run.features else "features " + ",".join(sorted(run.features))
                                bad = f"{run.recipe} ({feats}): cfg({text}) at {f}:{ln} is false"
                                break
                        if bad is None:
                            break
                    if bad == "?":
                        unknown.append(f"cfg({text}) at {f}:{ln} under {run.recipe}")
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
#   scope     `closure` — uncut target-code reach plus mandatory compilation inputs (own module trees,
#             build/proc-macro closures, manifests, dep-info), scripts/explicit reads and recipe dependencies —
#             or `tree`, every file tools/box.sh ships, with the reason: a command run on the Mac, a
#             cargo subcommand the parser does not model (`cargo fmt`), a script that walks the tree
#   file      path relative to the tree, content sha256 (a link: its spelling and referent content)
#   manifest  a package manifest the closure reads by table: path, sha256 of the item's view of it
#             (manifest_view: the shared tables and its own targets' entries, canonical JSON — a comment
#             or another target's entry does not move it); one read whole is a `file` part
#   global    Cargo.toml, Cargo.lock, rust-toolchain.toml, .cargo/*, the cuda-oxide pin rev
#   item      ARGS, the item's full BLOOMERY_BOX_ENV (the caller's, then the item's, then the lane's
#             card), the card that env forces
#   mac-env   the Mac-side variables tools/box.sh carries or reads (BLOOMERY_REMOTE is not one: it
#             names a track's directory, not an input) — among them BLOOMERY_TIER, unset and `real` the one value
#             `real`, so a fixture-tier green never stands for the real file's — and the versions of just and python3
#   variant   (an item that runs under BLOOMERY_FIXTURE_VARIANT) the tag and the sha256 of its row in tools/fixture-variants.tsv
#   fixture   (the fixture tier only) the fixture directories the box part holds: the family's, `-` for a family with none, `*` for
#             an item whose family cannot be told (every row of FIXTURE_ROOT)
#   box       every line of the box manifest (`box-manifest`, fetched once per batch), its model rows
#             only for the model directories the item can open (KeyContext.model_scope); the fixture
#             rows (FIXTURE_ROOT) only for an item in the fixture tier, and of those only its family's
#             fixture directory (KeyContext.fixture_dirs; every file under it)
# Not in it: the binary (its paths are per track) and the commit.

KEY_VERSION = "bloomery-gate-key 2"
MANIFEST_VERSION = "box-manifest 1"
MAC_ENV = ("BLOOMERY_BOX", "BLOOMERY_CARD", "BLOOMERY_DATA", "BLOOMERY_FIXTURE_VARIANT", "BLOOMERY_MODEL", "BLOOMERY_REF_MODEL", "BLOOMERY_TIER", "BLOOMERY_V41_MODEL")
# The fixture tier's files (tools/ref/ref-paths.sh: one directory a family under it). Its rows belong to the keys of fixture-tier
# items only, and of those to the items that open the directory: a real item must not rerun because a fixture was regenerated, and
# a fixture item must when its family's fixture was, whatever a fixture file's directory is called — so the box manifest lists every
# file under it, subdirectories included, and no file's literal brings it into a real item's key (KeyContext.literal_dirs). An item's
# key holds its family's fixture directory only (KeyContext.fixture_dirs; the family's row of tools/fixture-dirs.tsv, the one
# owner), as a variant's below holds its tag's, so a fixture written for one family moves no other family's key. An item whose
# family cannot be told keeps every row of FIXTURE_ROOT: never under-keyed.
FIXTURE_ROOT = "/models/fixtures"
# A fixture variant (tools/fixture-variants.tsv, tools/ref/ref-paths.sh): the family's fixture requantized to a tag's type map, its files
# below VARIANT_ROOT/<tag>, beside the fixture root and not under it (ref-paths.sh moves the root to the tag's directory, and a variant
# written is not a fixture file of any family). An item's key holds no variant's rows unless it runs under that tag (BLOOMERY_FIXTURE_VARIANT=<tag>): then its own
# directory's, and the tag's row of the table. A re-requant or an edit of the row moves that item's key and no other's, and a
# variant written moves no other item's.
VARIANT_ROOT = "/models/fixture-variants"
VARIANT_TABLE = "tools/fixture-variants.tsv"


def tier_of(values: list[str], mac: str | None) -> str:
    """The tier an item runs in: the last BLOOMERY_TIER=<t> entry of `values` (the box env, then the item's env), else the
    Mac's BLOOMERY_TIER; unset and empty are `real`, as in tools/box.sh. A value that is neither is returned as it is."""
    tier = mac or "real"
    for e in values:
        k, _, v = e.partition("=")
        if k == "BLOOMERY_TIER":
            tier = v or "real"
    return tier


def variant_of(values: list[str], mac: str | None) -> str:
    """The variant an item runs under: the last BLOOMERY_FIXTURE_VARIANT=<tag> entry of `values` (the box env, then the item's env),
    else the Mac's; empty is none, as in tools/box.sh."""
    tag = mac or ""
    for e in values:
        k, _, v = e.partition("=")
        if k == "BLOOMERY_FIXTURE_VARIANT":
            tag = v
    return tag


def variant_table_row(root: str, tag: str) -> str | None:
    """The line of VARIANT_TABLE whose first column is `tag`, None when the table names none (or is not there)."""
    try:
        with open(os.path.join(root, VARIANT_TABLE), encoding="utf-8") as fh:
            for ln in fh.read().splitlines():
                if ln and not ln.startswith("#") and ln.split("\t", 1)[0] == tag:
                    return ln
    except OSError:
        return None
    return None


def variant_table_rows(side: Side | None) -> dict[str, tuple[list[str], str]]:
    """{tag: (recipes, the row's text)} of `side`'s VARIANT_TABLE: `tag, family, scope, map, recipes, why`, tab-separated, `#` lines
    and blanks skipped. A side without the table, or no side, holds none; a row that is not six columns is a named error."""
    if side is None or not side.tree.exists(VARIANT_TABLE):
        return {}
    rows: dict[str, tuple[list[str], str]] = {}
    for ln in side.tree.read(VARIANT_TABLE).splitlines():
        if not ln.strip() or ln.startswith("#"):
            continue
        cols = ln.split("\t")
        if len(cols) != 6 or not cols[0] or not cols[4].split():
            raise RecipeError(f"{VARIANT_TABLE}: the row {ln[:60]!r} is not `tag, family, scope, map, recipes, why` (six tab-separated columns, recipes named)")
        rows[cols[0]] = (cols[4].split(), ln)
    return rows


def variant_items(picked: list[str], a: Side | None, b: Side, changed: list[str], prefix: str) -> list[tuple[str, str]]:
    """The fixture-variant items of an `affected` list, `(item, why)` in the table's order: `NAME@BLOOMERY_FIXTURE_VARIANT=<tag>` for each
    recipe of `picked` that a row of the table lists, and for each recipe of a row whose text moved between the two sides (a row added or
    edited) whether or not the diff selected it: the variant's file is not in the repository, so the row is what a change to the
    variant is. A recipe the justfile does not have, or that `prefix` leaves out, is skipped (check-recipes holds the table to the justfile)."""
    old = variant_table_rows(a)
    out = []
    for tag, (recipes, text) in variant_table_rows(b).items():
        moved = VARIANT_TABLE in changed and old.get(tag, ([], ""))[1] != text
        for r in recipes:
            if r not in b.recipes or not r.startswith(prefix):
                continue
            if r in picked:
                out.append((f"{r}@BLOOMERY_FIXTURE_VARIANT={tag}", f"variant {tag}: its row lists {r}, which is selected"))
            elif moved:
                out.append((f"{r}@BLOOMERY_FIXTURE_VARIANT={tag}", f"variant {tag}: its row in {VARIANT_TABLE} changed"))
    return out


def own_variant_row(path: str, variant: str) -> bool:
    """Whether a manifest row of VARIANT_ROOT is `variant`'s own: the directory itself (an absent or empty one is named so), or a path
    in the tag's directory. The rows of the other tags' directories are not."""
    return path == VARIANT_ROOT or (path + "/").startswith(f"{VARIANT_ROOT}/{variant}/")


# The fixture tier's family table, tools/fixture-dirs.tsv (the one owner: tools/ref/ref-paths.sh and crates/refset/src/fixture.rs read
# the same file), and ref-paths.sh's default profile are read by tools/bloomery/fixture_dirs.py, imported at the top.


def own_fixture_row(path: str, dirs: set[str]) -> bool:
    """Whether a manifest row of FIXTURE_ROOT is in one of the family directories `dirs`: the directory itself or a path in it. The
    root's own row (an absent or empty root) and the rows of the other families' directories are not; no directory, no row."""
    return any((path + "/").startswith(f"{FIXTURE_ROOT}/{d}/") for d in dirs)


KEY_GLOBALS = ["Cargo.toml", "Cargo.lock", "rust-toolchain.toml", ".cargo/config.toml"]
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
        self._links: dict[str, str | None] = {}
        self._lit: dict[str, frozenset[str]] = {}
        self._tree: list[str] | None = None
        self._fixture_table: tuple[dict[str, str], str] | None = None

    def tree_files(self) -> list[str]:
        if self._tree is None:
            self._tree = shipped_files(self.side.tree.root)
        return self._tree

    def file_part(self, rel: str) -> str:
        p = os.path.join(self.side.tree.root, rel)
        digest = sha256_file(p)
        self._sha[rel] = digest
        spelling = os.readlink(p) if os.path.islink(p) else None
        self._links[rel] = spelling
        if spelling is not None:
            return f"link\t{rel}\t{spelling}\t{digest}"
        return f"file\t{rel}\t{digest}"

    def view_part(self, rel: str, wants: frozenset[Entry]) -> str:
        """The part of a manifest read by table: the sha256 of the item's view of it."""
        view = manifest_view(read_manifest(self.side.tree.root, rel), os.path.dirname(rel), wants)
        return f"manifest\t{rel}\t{sha256_text(canon(view))}"

    def literal_dirs(self, rel: str) -> frozenset[str]:
        if rel not in self._lit:
            found = file_model_literal_dirs(os.path.join(self.side.tree.root, rel)) if scans_for_models(rel) else frozenset()
            self._lit[rel] = found - {FIXTURE_ROOT, VARIANT_ROOT}
        return self._lit[rel]

    def model_scope(self, names: list[str], files: set[str], eff: list[str], envs: list[str], argv: list[str]) -> tuple[dict[str, str], set[str], set[str] | None]:
        """The model directories the item can open, each with the first input that names it, those of
        them the box manifest must have rows for, and which rows of FIXTURE_ROOT the key holds. They are the /models literals in its
        key's files (its profile and tools/box.sh's among them) and its recipes' text, the profile the Mac's BLOOMERY_MODEL
        picks for a box recipe that names none, and the /models values of BLOOMERY_BOX_ENV, the forwarded
        Mac environment, the item's env and its ARGS. The item's env and ARGS are outside the manifest's
        reads (never-skip), so they need no rows. The third value is the item's fixture directories under the fixture tier
        (fixture_dirs), None for every row of FIXTURE_ROOT: the item names the root itself (an env value), or its family cannot be told."""
        recipes, tree = self.side.recipes, self.side.tree
        dirs: dict[str, str] = {}
        for rel in sorted(files | {g for g in KEY_GLOBALS if tree.exists(g)}):
            for d in self.literal_dirs(rel):
                dirs.setdefault(d, rel)
        for n in names:
            for d in model_literal_dirs(recipes[n].text):
                dirs.setdefault(d, f"recipe {n}")
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
        fixture: set[str] | None = None
        if tier_of(eff, self.env.get("BLOOMERY_TIER")) == "fixture":
            named = FIXTURE_ROOT in dirs
            dirs.setdefault(FIXTURE_ROOT, "the fixture tier")
            need.add(FIXTURE_ROOT)
            if not named:
                fixture = self.fixture_dirs(names, eff)
        if variant_of(eff, self.env.get("BLOOMERY_FIXTURE_VARIANT")):
            dirs.setdefault(VARIANT_ROOT, "BLOOMERY_FIXTURE_VARIANT")
            need.add(VARIANT_ROOT)
        for w in envs + argv:
            for v in w.split(","):
                d = model_dir_of(v.split("=", 1)[-1])
                if d and d not in dirs:
                    dirs[d] = "the item's env or ARGS"
        return dirs, need, fixture

    def fixture_dirs(self, names: list[str], eff: list[str]) -> set[str] | None:
        """The directories under FIXTURE_ROOT an item in the fixture tier can open: the fixture directory of each family it runs under,
        from the fixture directory table (fixture_table; `self` and `none` have none: the set may be empty). Its families are
        the BLOOMERY_MODEL of each tools/box.sh call of its recipes, else the Mac's, else ref-paths.sh's default: tools/box.sh picks the
        profile on the Mac side and a script on the box that picks another is refused (ref-paths.sh, exit 64), so the family names
        the only fixture directory a box command opens. None — every row of FIXTURE_ROOT, never under-keyed — when the family cannot
        be told: an unresolved value such as `{{FAMILY}}`, a BLOOMERY_MODEL in the item's env or
        BLOOMERY_BOX_ENV, a dotenv file, or a recipe that runs a command on the Mac (it may call tools/box.sh with a profile of its own)."""
        st = self.settings.get("settings", {})
        if st.get("dotenv_load") or st.get("dotenv_filename") or st.get("dotenv_path"):
            return None
        if any(e.partition("=")[0] == "BLOOMERY_MODEL" for e in eff):
            return None
        calls: list[str | None] = []
        for n in names:
            if command_shape(self.side.recipes[n])[0]:
                return None
            calls += self.side.graph.inputs(n).commands.models
        if not calls:
            return set()
        if self._fixture_table is None:
            self._fixture_table = fixture_table(self.side.tree.read(FIXTURE_DIRS), self.side.tree.read(REF_PATHS))
        rows, _ = self._fixture_table
        dirs: set[str] = set()
        try:
            families = RecipeCommands(models=calls).profiles(self.env)
        except RecipeError:
            return None
        for family in families:
            d = rows.get(family, "none")
            if d not in ("self", "none"):
                dirs.add(d)
        return dirs

    def globals(self) -> list[str]:
        root = self.side.tree.root
        names = list(KEY_GLOBALS)
        cdir = os.path.join(root, ".cargo")
        if os.path.isdir(cdir):
            names += sorted(f".cargo/{f}" for f in os.listdir(cdir) if f".cargo/{f}" not in names)
        out = []
        for g in names:
            p = os.path.join(root, g)
            digest = sha256_file(p) if os.path.isfile(p) else "absent"
            if digest != "absent":
                self._sha[g] = digest
            out.append(f"global\t{g}\t{digest}")
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
                start += BOX_GLOBALS + [f"tools/ref/models/{profile}.sh" for profile in sorted(rc.profiles(self.env))]
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
        # Safety detectors deliberately inspect the broad compilation closure, not just ledger reach.
        detector_files = {f for n in names for f in self.side.graph.inputs(n).files}
        for f in sorted(detector_files):
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
            ri = graph.key_inputs(name)
        except (OSError, RecipeError) as err:
            return [], f"{name}: its inputs cannot be read: {err}", None, ""
        if ri.errors:
            return [], f"{name}: {ri.errors[0]}", None, ""
        names = recipe_closure(recipes, name)
        try:
            self._sha.clear()
            self._links.clear()
            self._directories = {}
            result = self._parts(name, names, ri, envs, argv)
            for rel, digest in self._sha.items():
                if sha256_file(os.path.join(tree.root, rel)) != digest:
                    raise RecipeError(f"input changed during key assembly: {rel}")
            for rel, spelling in self._links.items():
                path = os.path.join(tree.root, rel)
                if (os.readlink(path) if os.path.islink(path) else None) != spelling:
                    raise RecipeError(f"link changed during key assembly: {rel}")
            for prefix, files in self._directories.items():
                if tree.directory_files(prefix) != files:
                    raise RecipeError(f"directory changed during key assembly: {prefix}")
            reach_of(tree).validate()
            return result
        except (OSError, RecipeError) as err:
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
                        found = self._directories[rel] = tree.directory_files(rel)
                        named |= found
        parts = [f"v\t{KEY_VERSION}"]
        parts += [f"recipe\t{n}\t{sha256_text(recipes[n].text)}" for n in names]
        parts.append(f"recipe\t<justfile>\t{sha256_text(json.dumps(self.settings, sort_keys=True))}")
        parts.append(f"scope\t{scope}\t{why if scope == 'tree' else 'closure'}")
        if scope == "tree":
            files = set(self.tree_files())
        else:
            files = set(ri.files)
            for p in ri.prefixes:
                found = self._directories[p] = tree.directory_files(p)
                files |= found
        for n in names:
            rc = self.side.graph.inputs(n).commands
            files |= {f"tools/ref/models/{profile}.sh" for profile in rc.profiles(self.env)}
        files |= named
        files -= set(KEY_GLOBALS)
        if scope == "closure":
            for u in tree.unresolved:
                if u.split(": ", 1)[0] in self.side.graph.inputs(name).files:
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
        variant = variant_of(eff, self.env.get("BLOOMERY_FIXTURE_VARIANT"))
        if variant:
            row = variant_table_row(tree.root, variant)
            if row is None:
                return [], f"{VARIANT_TABLE} names no variant {variant!r}, which the item runs under (BLOOMERY_FIXTURE_VARIANT)", None, ""
            parts.append(f"variant\t{variant}\t{sha256_text(row)}")
        # The manifest lists every model directory any item can open; this item's key holds the rows of
        # its own only, so a directory another gate names (a fixture, a profile) does not move it.
        try:
            dirs, need, fixture = self.model_scope(names, files, eff, envs, argv)
        except RecipeError as err:
            return [], str(err), None, ""
        if tier_of(eff, self.env.get("BLOOMERY_TIER")) == "fixture":
            parts.append(f"fixture\tdirs\t{'*' if fixture is None else ','.join(sorted(fixture)) or '-'}")
        listed: set[str] = set()
        for ln in self.manifest:
            f = ln.split("\t")
            if f[0] in ("model", "model-dir") and len(f) > 1:
                d = model_dir_of(f[1])
                if d is not None and d not in dirs:
                    continue
                if d == VARIANT_ROOT and not own_variant_row(f[1], variant):
                    continue
                if d == FIXTURE_ROOT and fixture is not None:
                    listed.add(d)  # the manifest names the root whether or not a row of it is this item's
                    if not own_fixture_row(f[1], fixture):
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
    return dirs | tree_model_dirs(root) | {FIXTURE_ROOT, VARIANT_ROOT}


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
    return model_dir_rows(d, d in (fixture_root, VARIANT_ROOT))


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


def archive_self_test(expect) -> None:
    """`archive()` against a git whose last write comes late and a tar that stops at the end-of-archive marker
    (tools/nightly/archive-shim, the shims tools/nightly/selftest.sh holds archive_tree to): the tree comes back whole,
    and a revision git refuses is a named error, never a half tree."""
    shim = os.path.join(ROOT, "tools", "nightly", "archive-shim")
    real_git, real_tar = shutil.which("git"), shutil.which("tar")
    with tempfile.TemporaryDirectory(prefix="recipes-archive-") as tmp:
        repo = os.path.join(tmp, "repo")
        os.makedirs(repo)
        env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull}
        for cmd in (["init", "-q", "."], ["config", "user.email", "t@t"], ["config", "user.name", "t"], ["config", "commit.gpgsign", "false"]):
            subprocess.run(["git", *cmd], cwd=repo, env=env, check=True, capture_output=True)
        with open(os.path.join(repo, "f"), "w", encoding="utf-8") as fh:
            fh.write("a\n")
        subprocess.run(["git", "add", "f"], cwd=repo, env=env, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-q", "-m", "one"], cwd=repo, env=env, check=True, capture_output=True)
        saved = {k: os.environ.get(k) for k in ("PATH", "SHIM_REAL_GIT", "SHIM_REAL_TAR", "GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM")}
        try:
            os.environ.update({"PATH": shim + os.pathsep + os.environ["PATH"], "SHIM_REAL_GIT": real_git or "", "SHIM_REAL_TAR": real_tar or "",
                               "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull})
            try:
                into = archive("HEAD", os.path.join(tmp, "out"), cwd=repo)
                with open(os.path.join(into, "f"), encoding="utf-8") as fh:
                    expect(fh.read() == "a\n", "archive: the tree came back with the commit's file")
            except RecipeError as err:
                expect(False, f"archive: a git whose last write is late and a tar that stops at the end marker end the call: {err}")
            try:
                archive("no-such-rev", os.path.join(tmp, "bad"), cwd=repo)
                expect(False, "archive: a revision git refuses was accepted")
            except RecipeError as err:
                expect("no-such-rev" in str(err), f"archive: the refusal does not name the revision: {err}")
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def memo_self_test(expect) -> None:
    """`Tree.memo` on a scratch root: a second Tree gets the first one's walk and script closure without
    running them again, a file rewritten to the same size or a path that appears between two Trees makes
    exactly the entries that saw it miss, and a caller that edits what it got edits nobody else's."""
    with tempfile.TemporaryDirectory(prefix="recipes-memo-") as tmp:

        def write(rel: str, text: str) -> None:
            os.makedirs(os.path.dirname(os.path.join(tmp, rel)), exist_ok=True)
            with open(os.path.join(tmp, rel), "w", encoding="utf-8") as fh:
                fh.write(text)

        def fresh() -> Tree:
            return Tree(tmp, {"workspace_root": tmp, "workspace_members": [], "packages": []})

        write("lib.rs", "mod a;\nmod gone;\n")
        write("a.rs", "")
        write("tools/s.sh", "bash tools/t.sh\n")
        write("tools/t.sh", "")
        calls: list[str] = []

        def probe(t: Tree, rel: str) -> tuple[object, bool]:
            def compute() -> str:
                calls.append(rel)
                return t.read(rel)

            return t.memo(("probe", rel), compute)

        got = [probe(fresh(), "lib.rs"), probe(fresh(), "lib.rs")]
        expect([h for _, h in got] == [False, True] and len(calls) == 1, f"memo: a second Tree reran the computation: hits {[h for _, h in got]}, runs {len(calls)}")
        write("lib.rs", "mod b;\nmod gone;\n")  # the same size as before
        write("b.rs", "")
        got = probe(fresh(), "lib.rs")
        expect(not got[1] and got[0] == "mod b;\nmod gone;\n", f"memo: a file rewritten to the same size was served from the memo: {got}")

        # a file settled for the racy window is kept by its stat, so a hit does not read it; a different size
        # misses. One written a moment ago (here: dated ahead) has no stat identity and is compared by its text.
        settled = HashCache.RACY_NS
        HashCache.RACY_NS = 0
        try:
            write("old.rs", "// one\n")
            calls.clear()
            got = [probe(fresh(), "old.rs"), probe(fresh(), "old.rs")]
            expect([h for _, h in got] == [False, True] and len(calls) == 1, f"memo: a settled file was read again: hits {[h for _, h in got]}, runs {len(calls)}")
            expect(_MEMO[(fresh().root, "probe", "old.rs")][1][("r", "old.rs")][1] is not None, "memo: a settled file kept no stat identity")
            write("old.rs", "// longer\n")
            got = probe(fresh(), "old.rs")
            expect(not got[1] and got[0] == "// longer\n", f"memo: a settled file rewritten to another size was served from the memo: {got}")
        finally:
            HashCache.RACY_NS = settled
        write("new.rs", "// one\n")
        ahead = time.time_ns() + 3600 * 10**9
        os.utime(os.path.join(tmp, "new.rs"), ns=(ahead, ahead))
        probe(fresh(), "new.rs")
        expect(_MEMO[(fresh().root, "probe", "new.rs")][1][("r", "new.rs")][1] is None, "memo: a file changed within the racy window kept a stat identity")

        t = fresh()
        value, _ = t.memo(("outer",), lambda: probe(t, "a.rs")[0])
        write("a.rs", "// edited\n")
        t = fresh()
        value2, hit = t.memo(("outer",), lambda: probe(t, "a.rs")[0])
        expect(value == "" and value2 == "// edited\n" and not hit, f"memo: a memo that read through another missed its change: {value!r} {value2!r} hit={hit}")

        t1 = fresh()
        files1 = set(t1.tree_of("lib.rs")[0])
        expect(files1 == {"lib.rs", "b.rs"} and len(t1.unresolved) == 1, f"memo: the first walk gave {sorted(files1)}, unresolved {t1.unresolved}")
        t2 = fresh()
        files2, _ = t2.tree_of("lib.rs")
        expect(files2 == files1 and t2.unresolved == t1.unresolved, f"memo: a second Tree's walk gave {sorted(files2)}, unresolved {t2.unresolved}")
        files2.add("scribble")
        t2.unresolved.clear()
        t3 = fresh()
        files3, _ = t3.tree_of("lib.rs")
        expect(files3 == files1 and len(t3.unresolved) == 1, f"memo: a caller's edit of a walk reached another Tree: {sorted(files3)}, unresolved {t3.unresolved}")
        write("gone.rs", "")  # the module that was missing appears
        t4 = fresh()
        files4, _ = t4.tree_of("lib.rs")
        expect(files4 == {"lib.rs", "b.rs", "gone.rs"} and not t4.unresolved, f"memo: a path that appeared is not walked: {sorted(files4)}, unresolved {t4.unresolved}")

        g1 = Graph(fresh(), {})
        c1 = g1.script_closure("tools/s.sh")
        expect(c1 == {"tools/s.sh": "script", "tools/t.sh": "script tools/s.sh"}, f"memo: script closure {c1}")
        c1.pop("tools/t.sh")
        c2 = Graph(fresh(), {}).script_closure("tools/s.sh")
        expect(set(c2) == {"tools/s.sh", "tools/t.sh"}, f"memo: a caller's edit of a script closure reached another Graph: {c2}")
        write("tools/s.sh", "bash tools/u.sh\n")  # the same size as before
        write("tools/u.sh", "")
        c3 = Graph(fresh(), {}).script_closure("tools/s.sh")
        expect(set(c3) == {"tools/s.sh", "tools/u.sh"}, f"memo: a script rewritten to the same size kept its old closure: {c3}")
        write("tools/w.sh", 'find "$ROOT/tools/dd" -name x\n')
        c4 = Graph(fresh(), {}).script_closure("tools/w.sh")
        expect(set(c4) == {"tools/w.sh"}, f"memo: script closure of a find over a missing directory {c4}")
        os.makedirs(os.path.join(tmp, "tools/dd"))  # the directory the script walks appears
        c5 = Graph(fresh(), {}).script_closure("tools/w.sh")
        expect(set(c5) == {"tools/w.sh", "tools/dd/"}, f"memo: a directory that appeared is not in the script closure: {c5}")


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
    # a bin of the package with kernels and no gate is left out by name, and every name left out is such a bin
    every = {c.bin: c.recipes for c in kernel_carriers(side, gates, {}) if c.bin and c.bundle == SCAN_PACKAGE}
    for n in sorted(every):
        expect(n in CARRIER_SKIP or every[n], f"scan-set: bin {n} has kernels and no gate links it, and CARRIER_SKIP does not name it")
    for n, why in sorted(CARRIER_SKIP.items()):
        expect(n in every, f"scan-set: CARRIER_SKIP names {n}, which is no bin of {SCAN_PACKAGE} with kernels")
        expect(not every.get(n), f"scan-set: CARRIER_SKIP names {n}, which {' '.join(every.get(n) or [])} links")
        expect(why.strip() != "", f"scan-set: CARRIER_SKIP {n}: no reason")

    def build(tmp: str, name: str, libc: bool = False, other: bool = False, recipe: str = "") -> Side:
        root = os.path.join(tmp, name)
        kernel, plain = "#[kernel]\nfn k() {}\n", "fn f() {}\n"
        files = {"crates/liba/src/lib.rs": kernel, "crates/libb/src/lib.rs": kernel, "crates/libc/src/lib.rs": kernel if libc else plain,
                 "crates/gates/src/lib.rs": plain, "crates/gates/src/bin/x.rs": "fn main() {}\n", "crates/gates/src/bin/y.rs": "fn main() {}\n",
                 "crates/gates/src/bin/z.rs": kernel + "fn main() {}\n", "crates/gates/src/bin/w.rs": "fn main() {}\n",
                 "crates/other/src/main.rs": kernel + "fn main() {}\n" if other else "fn main() {}\n",
                 "justfile": "gate-gx:\n    ./tools/box.sh 'cargo oxide build -- -p bloomery-gpu-gates --features fa --release --bin x'\n" + recipe,
                 RELEASE_BUILD: "BINS=(x)\ncargo oxide build -- -p bloomery-gpu-gates --features fa --release\n"}
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
        # the shipped bins at the release features come first: x (fa) covers liba, then y and z cover what is left
        got = scan_set(fx, ship=True)
        expect(shown(got) == [("x", ("fa",)), ("y", ("fb",)), ("z", ("fa",))] and got[0].covers == ["lib liba"],
               f"scan-set: --ship does not put the shipped bin first: {shown(got)} {got[0].covers if got else got}")
        no_build = build(tmp, "t4")
        os.remove(os.path.join(tmp, "t4", RELEASE_BUILD))
        expect(refused(lambda: scan_set(no_build, ship=True), RELEASE_BUILD), "scan-set: --ship without tools/release/build.sh is not a named error")
        # a bin's own scan does not vouch for a lib of its package: the bundle name is shared
        c = Carrier("lib p", "p", set(), [], siblings=frozenset({"b"}))
        expect(not carrier_covered(c, "b", ["p"]) and carrier_covered(c, "x", ["p"]) and not carrier_covered(c, "x", ["q"]),
               "scan-set: a lib carrier is covered by its sibling bin carrier's scan")


def lever_rows_self_test(expect) -> None:
    """The lever registry read by row, on a synthetic workspace (levers: a lib holding the registry; reader: a bin that names the
    const HOST_ROOM; plain: a bin that names no lever; solo: a bin that links no lever and runs a script naming BLOOMERY_GEN_CTX, and a recipe that sets BLOOMERY_SPIN in its
    command).
    A diff whose only edits to the registry are rows selects gate-levers and the recipes that read a file naming the row; each
    other shape of change (code beside the rows, a row the reader cannot name, rows reordered, lib.rs moved beside a row) keeps
    the crate graph's whole selection, and --narrow follows the same split."""
    host_row = "    LeverSpec {\n        name: HOST_ROOM,\n        doc: \"Room, (a comma) and [brackets] in a string \\\n              that continues.\",\n    },\n"
    spin_row = "    LeverSpec {\n        name: SPIN,\n        at: &[OPS],\n        doc: \"Spin.\",\n    },\n"
    runners = "    // ---------- the runners\n    runner(\"BLOOMERY_GEN_CTX\", \"The context.\"),\n    runner(\"BLOOMERY_OTHER_\", \"Other.\"),\n"
    rows = host_row + spin_row + runners
    lib = "mod registry;\n#[cfg(test)]\nmod tests;\npub use registry::{HOST_ROOM, SPIN};\n"

    def registry(body: str = rows, *, limit: str = "8", ops: str = "crates/model/src/ops.rs", host: str = "BLOOMERY_HOST_ROOM",
                 cls: str = "R", head: str = "", before: str = "") -> str:
        return (
            "//! The rows.\n\nuse crate::{Class, LeverSpec};\n\n"
            f'pub const HOST_ROOM: &str = "{host}";\npub const SPIN: &str = "BLOOMERY_SPIN";\npub const LIMIT: u64 = {limit};\n'
            f'const OPS: &str = "{ops}";\n{head}\n'
            "const fn runner(name: &'static str, doc: &'static str) -> LeverSpec {\n"
            f"    LeverSpec {{ name, class: Class::{cls}, doc }}\n}}\n\n"
            f"pub(crate) static REGISTRY: &[LeverSpec] = &[\n{before}{body}];\n"
        )

    just = (
        "gate-levers:\n    ./tools/box.sh 'bash tools/gate.sh -p levers --lib -- --nocapture'\n"
        "gate-reader:\n    ./tools/box.sh 'cargo build -p reader --bin reader'\n"
        "gate-plain:\n    ./tools/box.sh 'cargo build -p plain --bin plain'\n"
        "gate-script:\n    ./tools/box.sh 'cargo build -p solo --bin solo && bash tools/s.sh'\n"
        "gate-envset:\n    BLOOMERY_BOX_ENV='BLOOMERY_SPIN=1' ./tools/box.sh 'cargo build -p solo --bin solo'\n"
    )
    every = {"gate-levers", "gate-reader", "gate-plain"}  # the recipes that link the lib: what the crate graph says of any file of it
    with tempfile.TemporaryDirectory(prefix="recipes-levers-") as tmp:
        made: list[str] = []

        def build(reg: str, lib_text: str = lib) -> Side:
            root = os.path.join(tmp, f"t{len(made)}")
            made.append(root)
            files = {
                "crates/levers/Cargo.toml": '[package]\nname = "levers"\nversion = "0.1.0"\n',
                "crates/levers/src/lib.rs": lib_text,
                "crates/levers/src/registry.rs": reg,
                "crates/levers/src/tests.rs": "",
                "crates/reader/Cargo.toml": '[package]\nname = "reader"\nversion = "0.1.0"\n',
                "crates/reader/src/main.rs": "fn main() {\n    let _ = (levers::HOST_ROOM, levers::NEWC);\n}\n",
                "crates/plain/Cargo.toml": '[package]\nname = "plain"\nversion = "0.1.0"\n',
                "crates/plain/src/main.rs": "fn main() {}\n",
                "crates/solo/Cargo.toml": '[package]\nname = "solo"\nversion = "0.1.0"\n',
                "crates/solo/src/main.rs": "fn main() {}\n",
                "tools/s.sh": 'echo "$BLOOMERY_GEN_CTX"\n',
                "tools/gate.sh": "",
                "justfile": just,
            }
            for f in BOX_GLOBALS + [f"tools/ref/models/{DEFAULT_PROFILE}.sh"]:
                files[f] = ""
            for rel, text in files.items():
                os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
                with open(os.path.join(root, rel), "w", encoding="utf-8") as fh:
                    fh.write(text)

            def pkg(name: str, src: str, kind: str, deps: list[str]) -> dict:
                return {"id": name, "name": name, "manifest_path": os.path.join(root, f"crates/{name}/Cargo.toml"), "features": {},
                        "dependencies": [{"name": d, "kind": None, "optional": False, "features": [], "uses_default_features": True} for d in deps],
                        "targets": [{"kind": [kind], "name": name, "src_path": os.path.join(root, f"crates/{name}/src/{src}")}]}

            meta = {"workspace_root": root, "workspace_members": ["levers", "reader", "plain", "solo"],
                    "packages": [pkg("levers", "lib.rs", "lib", []), pkg("reader", "main.rs", "bin", ["levers"]),
                                 pkg("plain", "main.rs", "bin", ["levers"]), pkg("solo", "main.rs", "bin", [])]}
            return side_at(root, meta)

        reg_file, lib_file = "crates/levers/src/registry.rs", "crates/levers/src/lib.rs"
        base = registry()
        a = build(base)
        new_c = 'pub const NEWC: &str = "BLOOMERY_NEWC_";\n'
        # (what moved, the registry on B, lib.rs on B, the recipes selected, the reason the file is read whole — None: read by row)
        cases: list[tuple[str, str, str, set[str], str | None]] = [
            ("a new row nothing names", registry(rows + '    runner("BLOOMERY_NEW_", "New."),\n'), lib, {"gate-levers"}, None),
            ("a new row whose const a reader names", registry(rows + "    LeverSpec {\n        name: NEWC,\n        doc: \"New.\",\n    },\n", head=new_c), lib,
             {"gate-levers", "gate-reader"}, None),
            ("a row edited that a reader names by its const", registry(rows.replace("Room,", "Rooms,")), lib, {"gate-levers", "gate-reader"}, None),
            ("a runner row edited that a script names by its variable", registry(rows.replace("The context.", "The context, bytes.")), lib,
             {"gate-levers", "gate-script"}, None),
            ("a row edited that nothing names", registry(rows.replace('"Other."', '"Others."')), lib, {"gate-levers"}, None),
            ("a row edited that a recipe sets in its command", registry(rows.replace('"Spin."', '"Spins."')), lib, {"gate-levers", "gate-envset"}, None),
            ("a row removed that a reader names", registry(spin_row + runners), lib, {"gate-levers", "gate-reader"}, None),
            ("a row removed that nothing names", registry(rows.replace('    runner("BLOOMERY_OTHER_", "Other."),\n', "")), lib, {"gate-levers"}, None),
            ("a runner row removed that a script names by its variable", registry(rows.replace('    runner("BLOOMERY_GEN_CTX", "The context."),\n', "")), lib,
             {"gate-levers", "gate-script"}, None),
            ("a name const whose value moved", registry(host="BLOOMERY_HOST_ROOM2_"), lib, {"gate-levers", "gate-reader"}, None),
            ("a string const the rows spell moved", registry(ops="crates/model/src/other.rs"), lib, {"gate-levers", "gate-envset"}, None),
            ("comments and layout only", registry(rows.replace("    // ---------- the runners\n", "    // ---------- the runners, again\n").replace(
                '    runner("BLOOMERY_OTHER_", "Other."),', '    runner(\n        "BLOOMERY_OTHER_",\n        "Other.",\n    ),'), before="    // a note\n"),
             lib, {"gate-levers"}, None),
            ("lib.rs moved beside a row", registry(rows.replace("Room,", "Rooms,")), lib + "fn f() {}\n", every, None),
            ("a pub const that is no name moved", registry(limit="9"), lib, every, "code outside the rows moved near `LIMIT : u64 = 9"),
            ("the row helper's body moved", registry(cls="A"), lib, every, "code outside the rows moved near"),
            ("a new item beside the rows", registry(head="fn helper() {}\n"), lib, every, "code outside the rows moved near"),
            ("a row the reader cannot name", registry(rows + "    make_row(1),\n"), lib, every, "a row the reader cannot name: `make_row ( 1 )`"),
            ("rows reordered", registry(spin_row + host_row + runners), lib, every, "changed their order"),
            ("one name twice", registry(rows + '    runner("BLOOMERY_OTHER_", "Again."),\n'), lib, every, "BLOOMERY_OTHER_ names two rows"),
            ("a string that never ends", registry(rows + '    runner("BLOOMERY_NEW_, "New."),\n'), lib, every, "a string never ends"),
        ]
        for what, reg, lib_b, want, whole in cases:
            b = build(reg, lib_b)
            changed = [reg_file] + ([lib_file] if lib_b != lib else [])
            tables = TableReads(changed, a, b)
            sels, _, notes = select(changed, a, b, GATE_PREFIX, tables)
            got = {x.recipe for x in sels}
            expect(got == want, f"lever rows: {what} selects {sorted(got)}, not {sorted(want)}")
            lv = tables.levers
            if whole is None:
                expect(lv.by_rows and lv.why_whole is None and lv.summary().startswith(f"{reg_file} read by row: "),
                       f"lever rows: {what} is not read by row: {lv.why_whole}")
            else:
                expect(not lv.by_rows and lv.why_whole is not None and whole in lv.why_whole and any(whole in n for n in notes),
                       f"lever rows: {what} is read whole for the wrong reason: {lv.why_whole}, notes {notes}")
        # the block `affected` prints: a moved row, the file that names it and the recipes that read the file
        b = build(registry(rows + "    LeverSpec {\n        name: NEWC,\n        doc: \"New.\",\n    },\n", head=new_c))
        tables = TableReads([reg_file], a, b)
        select([reg_file], a, b, GATE_PREFIX, tables)
        got = tables.levers.lines()
        expect(got == [f"lever rows ({reg_file}): 1 added, 0 changed, 0 removed; the recipe that tests the registry: gate-levers",
                       "  added BLOOMERY_NEWC_: named in crates/reader/src/main.rs -> gate-reader"], f"lever rows: the printed block: {got}")
        # the lexer: a raw string, a byte string, a char and a lifetime, a nested block comment and a line comment; a literal or comment
        # that never ends is a doubt, never a token that runs to the end of the file
        got = [t for _, t in _rs_tokens('a r#"x "y" z"# b"q" \'c\' &\'static /* o /* i */ o */ d // e\n f "g\\"h"')]
        expect(got == ["a", 'r#"x "y" z"#', 'b"q"', "'c'", "&", "'static", "d", "f", '"g\\"h"'], f"lever rows: the lexer read {got}")
        for bad in ('"x', "/* x", 'r#"x"'):
            try:
                _rs_tokens(bad)
                expect(False, f"lever rows: the lexer accepted {bad!r}")
            except LeverDoubt:
                pass
        # the real registry: every row is read (the count is the file's own row starts), a row edit, an added row and a removed row
        # are each one subject, and the file against itself moves nothing
        with open(os.path.join(ROOT, LEVER_REGISTRY), encoding="utf-8") as fh:
            real = fh.read()
        parsed = parse_lever_rows(real)
        starts = len(re.findall(r"^    (?:LeverSpec \{|path\(|runner\()", real[real.index("static REGISTRY"):], re.M))
        expect(len(parsed.rows) == starts and starts > 200, f"lever rows: the real registry has {starts} row starts and parses to {len(parsed.rows)} rows")
        expect(diff_lever_rows(parsed, parse_lever_rows(real)) == [], "lever rows: the real registry against itself moved a row")
        name = next(iter(parsed.rows))
        edited = real.replace(f"        name: {name.removeprefix(LEVER_PREFIX)},\n", f"        name: {name.removeprefix(LEVER_PREFIX)},\n        // a note\n", 1)
        expect(diff_lever_rows(parsed, parse_lever_rows(edited)) == [], "lever rows: a comment in the real registry moved a row")
        grown = real.rstrip("\n").removesuffix("];") + '    runner("BLOOMERY_LEVSEL_PROBE_", "Probe."),\n];\n'
        got = diff_lever_rows(parsed, parse_lever_rows(grown))
        expect([(x.kind, x.label) for x in got] == [("added", "BLOOMERY_LEVSEL_PROBE_")], f"lever rows: a runner row added to the real registry: {got}")
        last = list(parsed.rows)[-1]
        got = diff_lever_rows(parsed, parse_lever_rows(real.replace(f'"{last}"', f'"{last}_X"', 1)))
        expect(sorted((x.kind, x.label) for x in got) == sorted([("added", f"{last}_X"), ("removed", last)]), f"lever rows: a row renamed in the real registry: {got}")
        # --narrow with an identical scan pair and the table's row for the registry (every gate): a rows-only diff narrows to the
        # recipes that read the rows; any other diff keeps the full list
        lines = ["ptx-scan: mod1 bundle=bloomery-gpu bytes=100",
                 "ptx-scan bin=target/release/gx section=.oxart bytes=9 ptxas=/p ptxas-version=13.3.73 arch=sm_86 modules=1 "
                 "jit-card=NVIDIA_RTX_A6000 jit-cc=8.6 jit-cuda=13.4",
                 "entry reqntid depot ld.local st.local fma cvt.f16 regs smem spill blk/SM(static) jit_regs jit_local",
                 f"{'alpha':<24} 256 no 0 0 0 0 12 0 0 6 12 0", "ptx-scan-md5: method=decl1", f"alpha {'1' * 32} 41"]
        log = os.path.join(tmp, "scan.log")
        with open(log, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        pair = (read_scan(log), read_scan(log))
        star = [PathRow(1, reg_file, ["*"], "why", glob_regex(reg_file))]
        nar = narrow([reg_file], a, build(registry(rows.replace("Room,", "Rooms,"))), [pair], star)
        expect(not nar.full and set(nar.picks) == {"gate-levers", "gate-reader"},
               f"lever rows: --narrow on a row edit gives full {nar.full}, picks {sorted(nar.picks)} (want gate-levers gate-reader)")
        nar = narrow([reg_file], a, build(registry(limit="9")), [pair], star)
        expect(nar.full and "maps to *" in nar.full[0], f"lever rows: --narrow on code beside the rows keeps the full list: {nar.full}")


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
        # a package's [lints] moves no build: no recipe reads it but the one that reads the file whole (gate-whole's script), `affected` names
        # the table as held by the static tier, and beside a table that does move a build it adds nothing. The key still hashes the view
        # whole, so it moves for every recipe
        lints = base + '\n[lints.clippy]\npedantic = "deny"\n'
        b = build(lints)
        tables = TableReads(["crates/g/Cargo.toml"], a, b)
        sels, unmapped, _ = select(["crates/g/Cargo.toml"], a, b, GATE_PREFIX, tables)
        block = "\n".join(tables.lines())
        expect({x.recipe for x in sels} == {"gate-whole"} and not unmapped, f"tables: a [lints] edit selects {[x.recipe for x in sels]}, unmapped {unmapped}")
        expect(getattr(tables, "static_moved", None) == ["crates/g/Cargo.toml [lints]"] and "[lints] -> no recipe (held by the static tier: just lint, just check)" in block,
               f"tables: a [lints] edit is not named as held by the static tier: {getattr(tables, 'static_moved', None)}\n{block}")
        kb = keys(b)
        expect(all(kb[n] != ka[n] for n in every), f"tables: a [lints] edit leaves a ledger key where it was: {[n for n in every if kb[n] == ka[n]]}")
        sels, _, _ = select(["crates/g/Cargo.toml"], a, build(lints.replace("fy = []", "fy = []\nfz = []")), GATE_PREFIX)
        expect({x.recipe for x in sels} == every, f"tables: a [lints] edit beside [features] selects {sorted(x.recipe for x in sels)}, not every recipe")
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


def root_manifest_self_test(expect) -> None:
    """The root Cargo.toml read by table, on a synthetic virtual workspace (g; user, which links g; h, which a case
    makes a member or not; a root manifest of [workspace] alone). Only a change of the packages that are members
    selects by what reads them (2026-10-12, PR #64: `exclude` gaining `fuzz` selected all 149 gates); the two
    ways the file stays whole are pinned beside it: any other table, and a bin whose box dep-info lists the file
    (the one route a read of the root manifest reaches the tool: no script regex or path literal names it). A
    [workspace.dependencies] entry selects the recipes whose closure holds a package that inherits it (an entry no package
    inherits stays whole, with a note), and [workspace.lints] selects none: the static tier holds it. The
    ledger key still hashes the file whole, so a selected recipe's key must move."""
    base = (
        '[workspace]\nresolver = "3"\nmembers = ["crates/g", "crates/user"]\n# the reproducer is not a member\n'
        'exclude = ["crates/repro"]\n\n[workspace.dependencies]\nserde = "1"\n\n[profile.release]\nopt-level = 3\n'
    )
    members = 'members = ["crates/g", "crates/user"]\n'
    with_h = base.replace(members, 'members = ["crates/g", "crates/user", "crates/h"]\n')
    glob = base.replace(members, 'members = ["crates/*"]\n')
    pkgs = {"g": [], "user": ["g"], "h": []}
    just = (
        "gate-g:\n    ./tools/box.sh 'cargo test -p g --lib'\n"
        "gate-user:\n    ./tools/box.sh 'cargo build -p user --bin user'\n"
        "gate-h:\n    ./tools/box.sh 'cargo test -p h --lib'\n"
        "gate-ws:\n    ./tools/box.sh 'cargo test --workspace'\n"
        "gate-bare:\n    ./tools/box.sh 'cargo build'\n"
    )
    names = ["gate-g", "gate-user", "gate-h", "gate-ws", "gate-bare"]
    every = set(names)
    with tempfile.TemporaryDirectory(prefix="recipes-root-") as tmp:
        mf = os.path.join(tmp, "box-manifest.txt")
        with open(mf, "w", encoding="utf-8") as fh:
            fh.write(f"# {MANIFEST_VERSION} host=selftest data=/d\n# end\n")
        made: list[str] = []

        def build(root_toml: str, member: tuple[str, ...] = ("g", "user"), justfile: str = just, depinfo: set[str] | None = None,
                  inherit: dict[str, str] | None = None, raw: dict[str, str] | None = None) -> Side:
            """`inherit`: a package -> the `[workspace.dependencies]` names its `[dependencies]` takes (`name = { workspace = true }`);
            `raw`: a package -> text appended to its manifest (a target table)."""
            root = os.path.join(tmp, f"t{len(made)}")
            made.append(root)
            files = {
                "Cargo.toml": root_toml,
                "justfile": justfile,
                "fuzz/Cargo.toml": '[package]\nname = "fuzz"\nversion = "0.0.0"\n',
                "fuzz/src/lib.rs": "",
            }
            for n, deps in pkgs.items():
                toml = f'[package]\nname = "{n}"\nversion = "0.1.0"\n'
                taken = "".join(f"{e} = {{ workspace = true }}\n" for e in (inherit or {}).get(n, "").split())
                if deps or taken:
                    toml += "\n[dependencies]\n" + "".join(f'{d} = {{ path = "../{d}" }}\n' for d in deps) + taken
                files[f"crates/{n}/Cargo.toml"] = toml + (raw or {}).get(n, "")
                files[f"crates/{n}/src/" + ("main.rs" if n == "user" else "lib.rs")] = "fn main() {}\n" if n == "user" else ""
            for f in BOX_GLOBALS + [f"tools/ref/models/{DEFAULT_PROFILE}.sh"]:
                files[f] = ""
            for rel, text in files.items():
                os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
                with open(os.path.join(root, rel), "w", encoding="utf-8") as fh:
                    fh.write(text)
            packages = [
                {"id": n, "name": n, "manifest_path": os.path.join(root, f"crates/{n}/Cargo.toml"), "features": {},
                 "dependencies": [{"name": d, "kind": None, "optional": False, "features": [], "uses_default_features": True} for d in pkgs[n]],
                 "targets": [{"kind": ["bin" if n == "user" else "lib"], "name": n,
                              "src_path": os.path.join(root, f"crates/{n}/src/" + ("main.rs" if n == "user" else "lib.rs"))}]}
                for n in member
            ]
            side = side_at(root, {"workspace_root": root, "workspace_members": list(member), "packages": packages})
            if depinfo:
                side.tree.depinfo["user"] = ({"crates/user/src/main.rs", *depinfo}, 0.0)
            return side

        def keys(side: Side) -> dict[str, str]:
            ctx = KeyContext(side, mf, environ={})
            out = {}
            for n in names:
                parts, err, _, _ = ctx.parts(n)
                out[n] = item_key(parts) if err is None else "error: " + err
            return out

        a0 = build(base)
        pr64 = base.replace('"crates/repro"]', '"crates/repro", "fuzz"]')
        h_less = a0
        h_in = build(with_h, ("g", "user", "h"))
        h_only = {"gate-h", "gate-ws", "gate-bare"}
        cases = [
            # PR #64's shape: `exclude` gains a path that held no member (fuzz, a separate crate)
            ("pr64-exclude", a0, build(base.replace('exclude = ["crates/repro"]', 'exclude = ["crates/repro", "fuzz"]')), set()),
            ("members in another order", a0, build(base.replace(members, 'members = ["crates/user", "crates/g"]\n')), set()),
            ("a comment", a0, build(base.replace("# the reproducer is not a member", "# not a member")), set()),
            ("a glob that makes the same members", h_in, build(glob, ("g", "user", "h")), set()),
            ("a member added", h_less, h_in, h_only),
            ("a member removed", h_in, h_less, h_only),
            ("a glob with a member excluded", build(glob, ("g", "user", "h")), build(glob.replace('["crates/repro"]', '["crates/repro", "crates/h"]'), ("g", "user")), h_only),
            # user links g: its dependents read it too
            ("a member that another links, removed", a0, build(base.replace(members, 'members = ["crates/user"]\n'), ("user",)),
             {"gate-g", "gate-user", "gate-ws", "gate-bare"}),
            # default-members decides a build with no -p and no --workspace only: gate-bare is the one
            ("default-members added", a0, build(base.replace(members, members + 'default-members = ["crates/g"]\n')), {"gate-bare"}),
            ("default-members added, no recipe builds without -p", build(base, justfile=just.replace("gate-bare:\n    ./tools/box.sh 'cargo build'\n", "")),
             build(base.replace(members, members + 'default-members = ["crates/g"]\n'), justfile=just.replace("gate-bare:\n    ./tools/box.sh 'cargo build'\n", "")), set()),
            # [workspace.dependencies] by entry: the recipes whose closure holds a package that inherits the moved entry. g does, so
            # g's recipe, user's (it links g) and the two that build every package; h is no member and reads nothing
            ("an entry g inherits", build(base, inherit={"g": "serde"}), build(base.replace('serde = "1"', 'serde = "2"'), inherit={"g": "serde"}),
             {"gate-g", "gate-user", "gate-ws", "gate-bare"}),
            # user is linked by nothing but the workspace-wide recipes
            ("an entry only user inherits", build(base, inherit={"user": "serde"}), build(base.replace('serde = "1"', 'serde = "2"'), inherit={"user": "serde"}),
             {"gate-user", "gate-ws", "gate-bare"}),
            # an entry added while a package starts to inherit it: the inheritors of either side count
            ("an entry added that user takes", a0, build(base.replace('serde = "1"\n', 'serde = "1"\nanyhow = "1"\n'), inherit={"user": "anyhow"}),
             {"gate-user", "gate-ws", "gate-bare"}),
            # the box builds no macOS table: an entry only a macOS table inherits is in no box recipe's build, and is still inherited
            ("an entry only a macOS table inherits", build(base, raw={"user": '\n[target.\'cfg(target_os = "macos")\'.dependencies]\nserde = { workspace = true }\n'}),
             build(base.replace('serde = "1"', 'serde = "2"'), raw={"user": '\n[target.\'cfg(target_os = "macos")\'.dependencies]\nserde = { workspace = true }\n'}), set()),
            # a lint level moves no build: no recipe, the static tier holds it
            ("[workspace.lints]", a0, build(base + '\n[workspace.lints.clippy]\npedantic = "deny"\n'), set()),
            ("[workspace.lints] beside an entry g inherits", build(base, inherit={"g": "serde"}),
             build(base.replace('serde = "1"', 'serde = "2"') + '\n[workspace.lints.clippy]\npedantic = "deny"\n', inherit={"g": "serde"}),
             {"gate-g", "gate-user", "gate-ws", "gate-bare"}),
            ("[workspace.lints] beside a profile", a0, build(base.replace("opt-level = 3", "opt-level = 2") + '\n[workspace.lints.clippy]\npedantic = "deny"\n'), every),
            # every other table: the conservative path holds; an entry no package inherits is read whole, the reader cannot say what it reaches
            ("[workspace.dependencies], an entry no package inherits", a0, build(base.replace('serde = "1"', 'serde = "2"')), every),
            ("[profile.release]", a0, build(base.replace("opt-level = 3", "opt-level = 2")), every),
            ("[workspace] resolver", a0, build(base.replace('resolver = "3"', 'resolver = "2"')), every),
            ("[workspace.package]", a0, build(base + '\n[workspace.package]\nedition = "2024"\n'), every),
            ("[patch]", a0, build(base + '\n[patch.crates-io]\nfoo = { path = "foo" }\n'), every),
            ("members and [workspace.dependencies] together", h_less, build(with_h.replace('serde = "1"', 'serde = "2"'), ("g", "user", "h")), every),
            ("a manifest that does not parse", a0, build(base + "[profile\n", member=("g", "user")), every),
            # the file read as a file: a script naming it, a path literal in a linked lib
            ("pr64-exclude, a bin's dep-info lists the file", build(base, depinfo={"Cargo.toml"}), build(pr64, depinfo={"Cargo.toml"}),
             {"gate-user", "gate-ws", "gate-bare"}),
        ]
        ka_cache: dict[int, dict[str, str]] = {}
        for what, a, b, want in cases:
            ka = ka_cache.setdefault(id(a), keys(a))
            sels, unmapped, _ = select(["Cargo.toml"], a, b, GATE_PREFIX)
            got = {x.recipe for x in sels}
            expect(got == want, f"root: {what} selects {sorted(got)}, not {sorted(want)}")
            kb = keys(b)
            # a recipe that names a package outside the workspace on one side has no key there (gate-h, h not a member)
            kept = {n for n in want if not ka[n].startswith("error") and not kb[n].startswith("error")}
            moved = {n for n in kept if kb[n] != ka[n]}
            expect(kept == moved, f"root: {what} selects {sorted(kept - moved)}, whose key did not move (the key hashes the file whole)")
            if not want:
                expect(len(unmapped) == 1 and unmapped[0].startswith("Cargo.toml") and "read by table" in unmapped[0],
                       f"root: {what} selects nothing and prints {unmapped} under unmapped, not a Cargo.toml read by table")
        # the manifests: block names the change by table, what each selects, and the members on both sides
        a, b = a0, build(base.replace('exclude = ["crates/repro"]', 'exclude = ["crates/repro", "fuzz"]'))
        tables = TableReads(["Cargo.toml", "fuzz/Cargo.toml", "fuzz/src/lib.rs"], a, b)
        select(["Cargo.toml", "fuzz/Cargo.toml", "fuzz/src/lib.rs"], a, b, GATE_PREFIX, tables)
        block = "\n".join(tables.lines())
        expect("[workspace] exclude -> no recipe" in block and "2 before, 2 after" in block, f"root: the manifests: block of pr64-exclude is\n{block}")
        tables = TableReads(["Cargo.toml"], h_less, h_in)
        select(["Cargo.toml"], h_less, h_in, GATE_PREFIX, tables)
        block = "\n".join(tables.lines())
        expect("[workspace] members -> gate-h gate-ws gate-bare" in block and "entered h, left none" in block and "2 before, 3 after" in block,
               f"root: the manifests: block of a member added is\n{block}")
        tables = TableReads(["Cargo.toml"], a0, build(base.replace('serde = "1"', 'serde = "2"')))
        select(["Cargo.toml"], a0, build(base.replace('serde = "1"', 'serde = "2"')), GATE_PREFIX, tables)
        expect("[workspace.dependencies] serde -> every reader (5)" in "\n".join(tables.lines()), f"root: the manifests: block of a whole table is\n" + "\n".join(tables.lines()))
        # an entry no package inherits is named in a note; one a package inherits is printed with the packages that reach it
        _, _, notes = select(["Cargo.toml"], a0, build(base.replace('serde = "1"', 'serde = "2"')), GATE_PREFIX)
        expect(any("[workspace.dependencies] serde changed and no package inherits it; read whole" in n for n in notes), f"root: an entry no package inherits has no note: {notes}")
        a_g, b_g = build(base, inherit={"g": "serde"}), build(base.replace('serde = "1"', 'serde = "2"'), inherit={"g": "serde"})
        tables = TableReads(["Cargo.toml"], a_g, b_g)
        select(["Cargo.toml"], a_g, b_g, GATE_PREFIX, tables)
        block = "\n".join(tables.lines())
        expect("[workspace.dependencies] serde -> gate-g gate-user gate-ws gate-bare" in block, f"root: the manifests: block of an entry g inherits is\n{block}")
        # a lint table is held by the static tier: named as such, selecting no recipe
        lints_b = build(base + '\n[workspace.lints.clippy]\npedantic = "deny"\n')
        tables = TableReads(["Cargo.toml"], a0, lints_b)
        select(["Cargo.toml"], a0, lints_b, GATE_PREFIX, tables)
        block = "\n".join(tables.lines())
        expect(getattr(tables, "static_moved", None) == ["Cargo.toml [workspace.lints]"] and "[workspace.lints] -> no recipe (held by the static tier: just lint, just check)" in block,
               f"root: the lint table is not named as held by the static tier: {getattr(tables, 'static_moved', None)}\n{block}")
        # --narrow with an identical scan pair: the exclude of pr64 picks nothing and no longer holds the full list; a whole table still does
        lines = ["ptx-scan: mod1 bundle=bloomery-gpu bytes=100",
                 "ptx-scan bin=target/release/gx section=.oxart bytes=9 ptxas=/p ptxas-version=13.3.73 arch=sm_86 modules=1 "
                 "jit-card=NVIDIA_RTX_A6000 jit-cc=8.6 jit-cuda=13.4",
                 "entry reqntid depot ld.local st.local fma cvt.f16 regs smem spill blk/SM(static) jit_regs jit_local",
                 f"{'alpha':<24} 256 no 0 0 0 0 12 0 0 6 12 0", "ptx-scan-md5: method=decl1", f"alpha {'1' * 32} 41"]
        log = os.path.join(tmp, "scan.log")
        with open(log, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        pair = (read_scan(log), read_scan(log))
        nar = narrow(["Cargo.toml"], a, b, [pair], [])
        expect(not nar.full and not nar.picks, f"root: --narrow on pr64-exclude gives full {nar.full}, picks {sorted(nar.picks)}")
        nar = narrow(["Cargo.toml"], a0, build(base.replace('serde = "1"', 'serde = "2"')), [pair], [])
        expect(any("Cargo.toml" in r for r in nar.full), f"root: --narrow on [workspace.dependencies] gives full {nar.full}, not the full list")
        nar = narrow(["Cargo.toml"], a0, lints_b, [pair], [])
        expect(not nar.full and not nar.picks, f"root: --narrow on [workspace.lints] gives full {nar.full}, picks {sorted(nar.picks)}")


def lock_view_self_test(expect) -> None:
    """`[target.<key>]` tables and Cargo.lock read as the build for x86_64-unknown-linux-gnu sees them, on a synthetic
    workspace (package g: a lib with sha2 and, to build, bindgen; user: a bin linking g). A table whose predicate is false
    there (aarch64, macos, another triple) selects nothing and a true one still selects; an unknown predicate stays with a
    note. A lock diff whose packages hang off such a table selects nothing (the shape of PR #902, 2026-10-28: sha2's `asm`
    feature under an aarch64 table brought sha2-asm, cc, find-msvc-tools and shlex, and 249 of 249 gates were selected);
    one that bumps or removes a package a member reaches on the box still selects every recipe. A recipe that runs
    tools/mac-gate.sh reads the same tables and lock for aarch64-apple-darwin: a macOS-only change selects it and no box
    recipe. A lock diff selects by package: a recipe whose closure holds none of the packages it moves (a package only
    user links, for a recipe that builds g) reads nothing of it, a member's own entry is such a package, and a lock the
    reader cannot take apart is read whole with a note. The ledger key still moves for every case: it reads the manifests
    and the lock whole."""
    import tomllib

    reg = "registry+https://github.com/rust-lang/crates.io-index"
    gm = "crates/g/Cargo.toml"
    just = (
        "gate-g:\n    ./tools/box.sh 'cargo test -p g --lib'\n"
        "gate-user:\n    ./tools/box.sh 'cargo build -p user --bin user'\n"
    )
    just_mac = just + "gate-mac:\n    cargo test -p g --lib && ./tools/mac-gate.sh gate_g\n"
    every = {"gate-g", "gate-user"}
    g_base = ('[package]\nname = "g"\nversion = "0.1.0"\n\n[dependencies]\nsha2 = "0.10"\n\n'
              '[build-dependencies]\nbindgen = "0.7"\n')
    aarch64 = 'cfg(target_arch = "aarch64")'
    macos = 'cfg(target_os = "macos")'
    avx2 = 'cfg(target_feature = "avx2")'

    def table(key: str, body: str) -> str:
        k = key if re.fullmatch(r"[A-Za-z0-9_-]+", key) else "'" + key + "'"
        return "\n[target." + k + ".dependencies]\n" + body + "\n"

    def plus(key: str) -> str:
        return g_base + table(key, 'libc = "0.2"')

    def lock(pkgs: list[tuple[str, str, list[str]]]) -> str:
        out = "version = 4\n"
        for name, version, deps in pkgs:
            out += f'\n[[package]]\nname = "{name}"\nversion = "{version}"\n'
            if name not in ("g", "user"):
                out += f'source = "{reg}"\nchecksum = "{name}-{version}"\n'
            if deps:
                out += "dependencies = [\n" + "".join(f' "{d}",\n' for d in deps) + "]\n"
        return out

    pk = [("bindgen", "0.7.0", ["shlex"]), ("cpufeatures", "0.2.17", []), ("digest", "0.10.7", []), ("g", "0.1.0", ["bindgen", "sha2"]),
          ("sha2", "0.10.9", ["cpufeatures", "digest"]), ("shlex", "1.3.0", []), ("user", "0.1.0", ["g"])]
    lock0 = lock(pk)
    # PR #902's lock: sha2 takes sha2-asm, which takes cc; shlex has two versions, so bindgen writes `shlex 1.3.0`
    asm_lock = lock([("bindgen", "0.7.0", ["shlex 1.3.0"]), ("cc", "1.10.0", ["find-msvc-tools", "shlex 2.0.1"]), ("cpufeatures", "0.2.17", []),
                     ("digest", "0.10.7", []), ("find-msvc-tools", "0.1.15", []), ("g", "0.1.0", ["bindgen", "sha2"]),
                     ("sha2", "0.10.9", ["cpufeatures", "digest", "sha2-asm"]), ("sha2-asm", "0.6.4", ["cc"]), ("shlex", "1.3.0", []),
                     ("shlex", "2.0.1", []), ("user", "0.1.0", ["g"])])

    def bumped(who: str, version: str) -> str:
        return lock([(n, version if n == who else v, d) for n, v, d in pk])

    no_bindgen = g_base.replace('\n[build-dependencies]\nbindgen = "0.7"\n', "")
    no_bindgen_lock = lock([(n, v, [x for x in d if x != "bindgen"]) for n, v, d in pk if n not in ("bindgen", "shlex")])
    serde = g_base.replace('sha2 = "0.10"\n', 'sha2 = "0.10"\nserde = "1"\n')
    serde_lock = lock([(n, v, d + ["serde"] if n == "g" else d) for n, v, d in pk] + [("serde", "1.0.0", [])])
    objc = g_base + table(macos, 'objc2 = "0.6"')
    pk_objc = [(n, v, d + ["objc2"] if n == "g" else d) for n, v, d in pk] + [("objc2", "0.6.0", [])]
    objc_lock = lock(pk_objc)
    objc_bump = lock([(n, "0.6.1" if n == "objc2" else v, d) for n, v, d in pk_objc])
    asm_toml = g_base + table(aarch64, 'sha2 = { version = "0.10", features = ["asm"] }')
    cases = [
        # (what, g's manifest on B, Cargo.lock on B, the changed files, the recipes selected)
        ("an aarch64 table and the lock that follows it", asm_toml, asm_lock, ["Cargo.lock", gm], set()),
        ("an aarch64 table alone", asm_toml, lock0, [gm], set()),
        ("the lock of an aarch64 table alone", g_base, asm_lock, ["Cargo.lock"], set()),
        ("a macOS-only dependency and its lock entry", objc, objc_lock, ["Cargo.lock", gm], set()),
        ("a bump of a direct dependency", g_base, bumped("sha2", "0.10.10"), ["Cargo.lock"], every),
        ("a bump of a dependency's dependency", g_base, bumped("digest", "0.10.8"), ["Cargo.lock"], every),
        ("a build dependency removed", no_bindgen, no_bindgen_lock, ["Cargo.lock", gm], every),
        ("a dependency added", serde, serde_lock, ["Cargo.lock", gm], every),
        # the cfg predicates of a target table
        ("cfg(unix)", plus("cfg(unix)"), lock0, [gm], every),
        ("cfg(target_os = linux)", plus('cfg(target_os = "linux")'), lock0, [gm], every),
        ("cfg(not(target_os = macos))", plus('cfg(not(target_os = "macos"))'), lock0, [gm], every),
        ("cfg(target_os = macos)", plus(macos), lock0, [gm], set()),
        ("cfg(all(unix, aarch64))", plus('cfg(all(unix, target_arch = "aarch64"))'), lock0, [gm], set()),
        ("cfg(target_feature = avx2), a predicate it cannot decide", plus(avx2), lock0, [gm], every),
        # the triples
        ("the box's triple", plus("x86_64-unknown-linux-gnu"), lock0, [gm], every),
        ("another triple", plus("aarch64-apple-darwin"), lock0, [gm], set()),
    ]
    with tempfile.TemporaryDirectory(prefix="recipes-lockview-") as tmp:
        mf = os.path.join(tmp, "box-manifest.txt")
        with open(mf, "w", encoding="utf-8") as fh:
            fh.write(f"# {MANIFEST_VERSION} host=selftest data=/d\n# end\n")
        made: list[str] = []

        def deps_of(doc: dict) -> list[dict]:
            out: list[dict] = []

            def add(tbl: dict, kind: str | None, target: str | None) -> None:
                for name, spec in tbl.items():
                    s = spec if isinstance(spec, dict) else {}
                    out.append({"name": name, "kind": kind, "optional": bool(s.get("optional")), "features": list(s.get("features", [])),
                                "uses_default_features": s.get("default-features", True), "target": target})

            for section, kind in (("dependencies", None), ("build-dependencies", "build"), ("dev-dependencies", "dev")):
                add(doc.get(section, {}), kind, None)
                for key, body in doc.get("target", {}).items():
                    add(body.get(section, {}), kind, key)
            return out

        def build(g_toml: str, lock_text: str, justfile: str = just, user_deps: tuple[str, ...] = ()) -> Side:
            """`user_deps`: external dependencies only user's manifest names."""
            root = os.path.join(tmp, f"t{len(made)}")
            made.append(root)
            files = {
                gm: g_toml,
                "crates/g/src/lib.rs": "",
                "crates/user/Cargo.toml": '[package]\nname = "user"\nversion = "0.1.0"\n\n[dependencies]\ng = { path = "../g" }\n'
                                          + "".join(f'{d} = "1"\n' for d in user_deps),
                "crates/user/src/main.rs": "fn main() {}\n",
                "Cargo.lock": lock_text,
                "tools/mac-gate.sh": "",
                "justfile": justfile,
            }
            for f in BOX_GLOBALS + [f"tools/ref/models/{DEFAULT_PROFILE}.sh"]:
                files[f] = ""
            for rel, text in files.items():
                os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
                with open(os.path.join(root, rel), "w", encoding="utf-8") as fh:
                    fh.write(text)
            meta = {
                "workspace_root": root,
                "workspace_members": ["g", "user"],
                "packages": [
                    {"id": "g", "name": "g", "manifest_path": os.path.join(root, gm), "features": {},
                     "dependencies": deps_of(tomllib.loads(g_toml)),
                     "targets": [{"kind": ["lib"], "name": "g", "src_path": os.path.join(root, "crates/g/src/lib.rs")}]},
                    {"id": "user", "name": "user", "manifest_path": os.path.join(root, "crates/user/Cargo.toml"), "features": {},
                     "dependencies": [{"name": d, "kind": None, "optional": False, "features": [], "uses_default_features": True} for d in ("g", *user_deps)],
                     "targets": [{"kind": ["bin"], "name": "user", "src_path": os.path.join(root, "crates/user/src/main.rs")}]},
                ],
            }
            return side_at(root, meta)

        def keys(side: Side) -> dict[str, str]:
            ctx = KeyContext(side, mf, environ={})
            out = {}
            for n in sorted(every):
                parts, err, _, _ = ctx.parts(n)
                out[n] = item_key(parts) if err is None else "error: " + err
            return out

        a = build(g_base, lock0)
        ka = keys(a)
        expect(not any(v.startswith("error") for v in ka.values()), f"lock view: base keys {ka}")
        seen: dict[str, tuple[Side, TableReads, list[Selection], list[str], list[str]]] = {}
        for what, g_toml, lock_text, changed, want in cases:
            b = build(g_toml, lock_text)
            tables = TableReads(changed, a, b)
            sels, unmapped, notes = select(changed, a, b, GATE_PREFIX, tables)
            seen[what] = (b, tables, sels, unmapped, notes)
            got = {x.recipe for x in sels}
            expect(got == want, f"lock view: {what} selects {sorted(got)}, not {sorted(want)}")
            kb = keys(b)
            expect(all(kb[n] != ka[n] for n in every), f"lock view: {what} leaves a ledger key where it was (the key reads the manifests and the lock whole)")
        # the notes name what the box's view leaves out, what it cannot decide, and the lock's packages
        _, _, _, unmapped, notes = seen["an aarch64 table and the lock that follows it"]
        lock_note = next((n for n in notes if n.startswith("Cargo.lock: ")), "")
        expect(lock_note == "Cargo.lock: cc 1.10.0, find-msvc-tools 0.1.15, sha2 0.10.9, sha2-asm 0.6.4, shlex 2.0.1 changed, not in the x86_64-linux build",
               f"lock view: the lock's note is {lock_note!r} (bindgen's `shlex 1.3.0` names the package it named before and is no change)")
        expect(f'{gm}: [target.{aarch64}] changed, not in the x86_64-linux build' in notes, f"lock view: no note for the aarch64 table: {notes}")
        expect(len(unmapped) == 2 and all("not in the x86_64-linux build" in u or "read by table" in u for u in unmapped),
               f"lock view: the diff's files are not both printed under unmapped with their reason: {unmapped}")
        _, _, _, _, notes = seen["cfg(target_feature = avx2), a predicate it cannot decide"]
        expect(any(f"[target.{avx2}]" in n and "can decide" in n for n in notes), f"lock view: no note for a predicate it cannot decide: {notes}")
        _, tables, _, _, notes = seen["cfg(target_os = macos)"]
        expect(any(f"[target.{macos}] changed, not in the x86_64-linux build" in n for n in notes), f"lock view: no note for the macOS table: {notes}")
        block = "\n".join(tables.lines())
        expect(f"[target.{macos}] (not in the x86_64-linux build) -> no recipe" in block, f"lock view: the manifests: block of the macOS table is\n{block}")
        # a diff the box reaches is the lock's own verdict, whatever the manifest selects beside it
        for what in ("a dependency added", "a build dependency removed", "a bump of a direct dependency"):
            b = seen[what][0]
            expect("box" not in lock_outside(a, b), f"lock view: {what} is read as outside the x86_64-linux build")
        expect(lock_outside(a, a) == {}, "lock view: a lock that did not move is read as outside a build instead of whole")
        # a recipe that runs tools/mac-gate.sh reads aarch64-apple-darwin: the tables and the packages only a Mac builds select it alone
        a_m, a_o = build(g_base, lock0, just_mac), build(objc, objc_lock, just_mac)
        mac_cases = [
            ("a macOS table and its lock entry", a_m, objc, objc_lock, ["Cargo.lock", gm], {"gate-mac"}),
            ("an aarch64 table and the lock that follows it", a_m, asm_toml, asm_lock, ["Cargo.lock", gm], {"gate-mac"}),
            ("a linux table", a_m, plus('cfg(target_os = "linux")'), lock0, [gm], every),
            ("a bump of a macOS-only package", a_o, objc, objc_bump, ["Cargo.lock"], {"gate-mac"}),
        ]
        for what, base, g_toml, lock_text, changed, want in mac_cases:
            sels, _, _ = select(changed, base, build(g_toml, lock_text, just_mac), GATE_PREFIX)
            got = {x.recipe for x in sels}
            expect(got == want, f"lock view: with a Mac recipe, {what} selects {sorted(got)}, not {sorted(want)}")
        # Cargo.lock by package: a recipe reads the diff by what its own closure reaches. serde is linked by user alone, so gate-g
        # (g, no user) holds no package of a diff in it; a workspace member's own entry is a package of the diff too; and a lock the
        # reader cannot take apart is read whole, with a note
        pk_u = [(n, v, d + ["serde"] if n == "user" else d) for n, v, d in pk] + [("serde", "1.0.0", [])]
        user_only = {"gate-user"}
        a_u = build(g_base, lock(pk_u), user_deps=("serde",))

        def moved_to(who: str, version: str) -> str:
            return lock([(n, version if n == who else v, d) for n, v, d in pk_u])

        pkg_cases = [
            ("a bump of a package only user links", a_u, build(g_base, moved_to("serde", "1.0.1"), user_deps=("serde",)), user_only),
            ("a package only user links, removed", a_u, build(g_base, lock0), user_only),
            ("a member no other links, bumped", a, build(g_base, bumped("user", "0.2.0")), user_only),
            ("a member another links, bumped", a, build(g_base, bumped("g", "0.2.0")), every),
            ("a lock that moved a comment only", a, build(g_base, lock0 + "# regenerated\n"), every),
            ("the lock's format version", a, build(g_base, lock0.replace("version = 4", "version = 3", 1)), every),
            ("a lock that does not parse", a, build(g_base, lock0 + "[[package\n"), every),
            ("an edge no package resolves", a, build(g_base, lock0.replace(' "digest",\n', ' "digest",\n "nosuch",\n', 1)), every),
        ]
        notes_of: dict[str, list[str]] = {}
        for what, base_side, b, want in pkg_cases:
            tables = TableReads(["Cargo.lock"], base_side, b)
            sels, unmapped, notes = select(["Cargo.lock"], base_side, b, GATE_PREFIX, tables)
            notes_of[what] = notes
            got = {x.recipe for x in sels}
            expect(got == want, f"lock view: {what} selects {sorted(got)}, not {sorted(want)}")
            ka_p, kb_p = keys(base_side), keys(b)
            expect(all(kb_p[n] != ka_p[n] for n in every), f"lock view: {what} leaves a ledger key where it was (the key reads the lock whole)")
        expect("Cargo.lock: serde 1.0.0, serde 1.0.1 changed; 1 of 2 recipes' builds reach a package that moved" in notes_of["a bump of a package only user links"],
               f"lock view: the note of a package only user links is {notes_of['a bump of a package only user links']}")
        for what, text in (("the lock's format version", "a top-level key beside `package` moved"), ("a lock that does not parse", "does not parse as TOML"),
                           ("a lock that moved a comment only", "names no package the parse sees"),
                           ("an edge no package resolves", "depends on `nosuch`, which no package of the file is")):
            expect(any(n.startswith("Cargo.lock: ") and text in n and n.endswith("read whole by every cargo recipe") for n in notes_of[what]),
                   f"lock view: {what} is read whole with no note naming why: {notes_of[what]}")
        sels, _, _ = select(["Cargo.lock"], a_u, build(g_base, moved_to("serde", "1.0.1"), user_deps=("serde",)), GATE_PREFIX)
        expect([x.why for x in sels] == ["cargo (every build): serde 1.0.0, serde 1.0.1"], f"lock view: the origin of a selection by package is {[x.why for x in sels]}")
        # a recipe whose inputs the reader could not read whole (a package it names is no member) reads the lock whole
        bad = just + "gate-bad:\n    ./tools/box.sh 'cargo test -p g --lib && cargo test -p nosuch --lib'\n"
        sels, _, _ = select(["Cargo.lock"], build(g_base, lock(pk_u), bad, user_deps=("serde",)), build(g_base, moved_to("serde", "1.0.1"), bad, user_deps=("serde",)), GATE_PREFIX)
        expect({x.recipe for x in sels} == user_only | {"gate-bad"}, f"lock view: a recipe with an unreadable build selects {sorted(x.recipe for x in sels)}, not user's and its own")
        # features a recipe parameter names are known at run time only: such a recipe reads the lock whole
        par = just + "gate-par F:\n    ./tools/box.sh 'cargo test -p g --features {{F}} --lib'\n"
        sels, _, _ = select(["Cargo.lock"], build(g_base, lock(pk_u), par, user_deps=("serde",)), build(g_base, moved_to("serde", "1.0.1"), par, user_deps=("serde",)), GATE_PREFIX)
        expect({x.recipe for x in sels} == user_only | {"gate-par"}, f"lock view: a recipe whose features are a parameter selects {sorted(x.recipe for x in sels)}, not user's and its own")
        # a Mac recipe builds g alone: the package only user links is outside its build too
        sels, _, _ = select(["Cargo.lock"], build(g_base, lock(pk_u), just_mac, user_deps=("serde",)),
                            build(g_base, moved_to("serde", "1.0.1"), just_mac, user_deps=("serde",)), GATE_PREFIX)
        expect({x.recipe for x in sels} == user_only, f"lock view: with a Mac recipe that builds g, a package only user links selects {sorted(x.recipe for x in sels)}, not {sorted(user_only)}")
        # --narrow reads the same answer: with every scan pair identical, the spill ratchet takes Cargo.lock as its own input
        # only when the box builds what the diff names
        lines = ["ptx-scan: mod1 bundle=bloomery-gpu bytes=100",
                 "ptx-scan bin=target/release/gx section=.oxart bytes=9 ptxas=/p ptxas-version=13.3.73 arch=sm_86 modules=1 "
                 "jit-card=NVIDIA_RTX_A6000 jit-cc=8.6 jit-cuda=13.4",
                 "entry reqntid depot ld.local st.local fma cvt.f16 regs smem spill blk/SM(static) jit_regs jit_local",
                 f"{'alpha':<24} 256 no 0 0 0 0 12 0 0 6 12 0", "ptx-scan-md5: method=decl1", f"alpha {'1' * 32} 41"]
        log = os.path.join(tmp, "scan.log")
        with open(log, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        pair = (read_scan(log), read_scan(log))
        ratchet = just + "gate-ptx-spill:\n    ./tools/box.sh 'cargo test -p g --lib'\n"
        na = build(g_base, lock0, ratchet)
        nar = narrow(["Cargo.lock"], na, build(g_base, asm_lock, ratchet), [pair], [])
        expect(not nar.full and not nar.picks, f"lock view: --narrow gives full {nar.full}, picks {sorted(nar.picks)} for a Cargo.lock outside the x86_64-linux build")
        nar = narrow(["Cargo.lock"], na, build(g_base, bumped("sha2", "0.10.10"), ratchet), [pair], [])
        expect("gate-ptx-spill" in nar.picks, f"lock view: --narrow picks {sorted(nar.picks)} for a Cargo.lock the box reaches, not the spill ratchet")
        # the ratchet links g alone: a package only user links is outside its build
        nar = narrow(["Cargo.lock"], build(g_base, lock(pk_u), ratchet, user_deps=("serde",)),
                     build(g_base, moved_to("serde", "1.0.1"), ratchet, user_deps=("serde",)), [pair], [])
        expect("gate-ptx-spill" not in nar.picks, f"lock view: --narrow picks {sorted(nar.picks)} for a package the ratchet's closure does not link")


def range_base_self_test(expect) -> None:
    """`A..B` is B against merge-base(A, B), on a fixture repository that runs this file as its own tools/recipes.py:
    `main` moves on with a change that touches gate-x, and `feature` forked earlier and touches gate-y. A tree-to-tree
    diff of main..feature selected gate-x as well (2026-10-12, PR #45, 58 commits behind main: 149 gates from main's
    own commits, 137 from its own); the merge base selects gate-y alone and the `affected:` line names it. A range whose
    A is an ancestor of B is read as written."""
    with tempfile.TemporaryDirectory(prefix="recipes-range-") as repo:
        env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull,
               "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}

        def run(*cmd: str) -> subprocess.CompletedProcess:
            return subprocess.run(cmd, cwd=repo, env=env, capture_output=True, text=True)

        files = {
            "Cargo.toml": '[workspace]\nresolver = "3"\nmembers = ["crates/x", "crates/y"]\n',
            "justfile": "gate-x:\n    ./tools/box.sh 'cargo test -p x --lib'\ngate-y:\n    ./tools/box.sh 'cargo test -p y --lib'\n",
            "crates/x/Cargo.toml": '[package]\nname = "x"\nversion = "0.1.0"\n',
            "crates/x/src/lib.rs": "pub fn x() {}\n",
            "crates/y/Cargo.toml": '[package]\nname = "y"\nversion = "0.1.0"\n',
            "crates/y/src/lib.rs": "pub fn y() {}\n",
        }
        for f in BOX_GLOBALS + [f"tools/ref/models/{DEFAULT_PROFILE}.sh"]:
            files[f] = ""
        for rel, text in files.items():
            os.makedirs(os.path.dirname(os.path.join(repo, rel)), exist_ok=True)
            with open(os.path.join(repo, rel), "w", encoding="utf-8") as fh:
                fh.write(text)
        shutil.copyfile(os.path.abspath(__file__), os.path.join(repo, "tools/recipes.py"))
        os.makedirs(os.path.join(repo, "tools/bloomery"), exist_ok=True)
        shutil.copyfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bloomery/fixture_dirs.py"),
                        os.path.join(repo, "tools/bloomery/fixture_dirs.py"))  # the table reader recipes.py imports
        steps = [("git", "init", "-q", "-b", "main"), ("cargo", "generate-lockfile", "--offline", "-q"), ("git", "add", "-A"),
                 ("git", "commit", "-q", "-m", "base"), ("git", "tag", "base"), ("git", "branch", "feature")]
        for step in steps:
            done = run(*step)
            if done.returncode != 0:
                expect(False, f"range: fixture `{' '.join(step)}` failed: {done.stderr.strip()[-300:]}")
                return
        for branch, rel in (("main", "crates/x/src/lib.rs"), ("feature", "crates/y/src/lib.rs")):
            run("git", "switch", "-q", branch)
            with open(os.path.join(repo, rel), "a", encoding="utf-8") as fh:
                fh.write("pub fn moved() {}\n")  # code: a comment alone selects no gate
            run("git", "commit", "-q", "-am", f"{branch} moves {rel}")

        def affected(spec: str) -> tuple[str, list[str]]:
            done = run(sys.executable, "tools/recipes.py", "affected", spec, "--no-box", "--no-class")  # the fixture has no class tables
            if done.returncode != 0:
                expect(False, f"range: affected {spec} failed (rc {done.returncode}): {done.stderr.strip()[-300:]}")
            lines = done.stdout.splitlines()
            return (lines[0] if lines else ""), sorted(m.group(1) for ln in lines if (m := re.match(r"^  (gate-\S+)", ln)))

        head, got = affected("main..feature")
        expect(got == ["gate-y"], f"range: main..feature selects {got}, not ['gate-y'] (main's own commit moved gate-x's lib)")
        expect("(merge-base of main and feature)..feature" in head and " — 1 changed files," in head,
               f"range: the affected: line of main..feature does not name the merge base and one changed file: {head}")
        head, got = affected("base..feature")
        expect(got == ["gate-y"] and head.startswith("affected: base..") and "merge-base" not in head,
               f"range: base..feature (A an ancestor of B) gives {got}, `{head}`")
        head, got = affected("feature..main")
        expect(got == ["gate-x"] and "(merge-base of feature and main)" in head, f"range: feature..main gives {got}, `{head}`")
        # a commit that adds one comment selects no gate and says why; `--no-comment-only` selects the gate that reads the file
        run("git", "switch", "-q", "-c", "quiet", "base")
        with open(os.path.join(repo, "crates/x/src/lib.rs"), "a", encoding="utf-8") as fh:
            fh.write("// only a comment\n")
        run("git", "commit", "-q", "-am", "quiet adds a comment")
        for extra, want in (((), []), (("--no-comment-only",), ["gate-x"])):
            done = run(sys.executable, "tools/recipes.py", "affected", "base..quiet", "--no-box", "--no-class", *extra)
            lines = done.stdout.splitlines()
            got = sorted(m.group(1) for ln in lines if (m := re.match(r"^  (gate-\S+)", ln)))
            shown = "  crates/x/src/lib.rs  (comment-only, no gate reads it)" in lines
            counted = "comment- or layout-only files (no gate reads them): 1" in (lines[0] if lines else "")
            expect(done.returncode == 0 and got == want and shown == (not extra) and counted == (not extra),
                   f"range: base..quiet{' ' + ' '.join(extra) if extra else ''} gives {got}, unmapped line {shown}, counted {counted}: {done.stderr.strip()[-200:]}")


def module_gate_self_test(expect) -> None:
    """A module gated by a cargo feature is read by the recipes that build with that feature only, on a synthetic
    workspace (package g: features fy and fz, a lib with a plain module, one under `feature = "fy"`, one under
    `not(feature = "fy")`, one inline under `all(feature = "fy", feature = "fz")` with a data file, one under
    `any(test, feature = "fy")`, and bins x and y whose own trees gate a module the same way). The cases are the
    ones the rule must keep (an ungated module is read by every build; what a feature does not decide stays in)
    and the saving (a gated module leaves the builds without its feature); a recipe parameter in `--features`
    is read whole, and so is one that names several packages or a `pkg/feature` (cargo unifies features across
    the build)."""
    just = (
        "gate-x:\n    ./tools/box.sh 'cargo build -p g --bin x'\n"
        "gate-xy:\n    ./tools/box.sh 'cargo build -p g --features fy --bin x'\n"
        "gate-yz:\n    ./tools/box.sh 'cargo build -p g --features fy,fz --bin y'\n"
        "gate-lib:\n    ./tools/box.sh 'cargo test -p g --lib'\n"
        "gate-liby:\n    ./tools/box.sh 'cargo test -p g --lib --features fy'\n"
        "gate-all:\n    ./tools/box.sh 'cargo test -p g --lib --all-features'\n"
        "gate-par F='fy':\n    ./tools/box.sh 'cargo build -p g --features {{F}} --bin x'\n"
        "gate-ws:\n    ./tools/box.sh 'cargo build --workspace --bin x'\n"
        "gate-slash:\n    ./tools/box.sh 'cargo build -p g --features g/fy --bin x'\n"
    )
    files = {
        "crates/g/Cargo.toml": "",
        "crates/g/src/lib.rs": (
            'mod plain;\n#[cfg(feature = "fy")]\nmod gated;\n#[cfg(not(feature = "fy"))]\nmod off;\n'
            '#[cfg(all(feature = "fy", feature = "fz"))]\n#[allow(dead_code)]\nmod inl {\n    mod deep;\n'
            '    const D: &str = include_str!("../data/d.txt");\n}\n#[cfg(any(test, feature = "fy"))]\nmod either;\n'
            '#[cfg(test)]\nmod attr_test {\n'
            '    #[doc = concat!(stringify!({\n}\n    ),\n'
            '        include_str!("../../../attribute-test.txt"),\n'
            '        "tools/attribute-test-data.txt"\n    )]\n'
            '    pub struct Item;\n    const AFTER: &str = include_str!("../../../after-attribute.txt");\n}\n'
        ),
        "crates/g/src/plain.rs": '#[doc = concat!(\n    include_str!("../../../attribute.txt"),\n    "tools/attribute-data.txt"\n)]\npub struct Plain;\n',
        "crates/g/src/gated.rs": "",
        "crates/g/src/off.rs": "",
        "crates/g/src/inl/deep.rs": "",
        "crates/g/src/either.rs": "",
        "crates/g/data/d.txt": "",
        "attribute.txt": "attribute1\n",
        "tools/attribute-data.txt": "literal1\n",
        "attribute-test.txt": "test attribute\n",
        "tools/attribute-test-data.txt": "test literal\n",
        "after-attribute.txt": "after attribute\n",
        "crates/g/src/bin/x.rs": '#[cfg(feature = "fy")]\nmod xg;\nfn main() {}\n',
        "crates/g/src/bin/xg.rs": "",
        "crates/g/src/bin/y.rs": "fn main() {}\n",
        "justfile": just,
    }
    with tempfile.TemporaryDirectory(prefix="recipes-modgate-") as tmp:
        for f in BOX_GLOBALS + [f"tools/ref/models/{DEFAULT_PROFILE}.sh"]:
            files[f] = ""
        for rel, text in files.items():
            os.makedirs(os.path.dirname(os.path.join(tmp, rel)), exist_ok=True)
            with open(os.path.join(tmp, rel), "w", encoding="utf-8") as fh:
                fh.write(text)
        targets = [{"kind": ["lib"], "name": "g", "src_path": os.path.join(tmp, "crates/g/src/lib.rs")}]
        targets += [{"kind": ["bin"], "name": n, "src_path": os.path.join(tmp, f"crates/g/src/bin/{n}.rs"), "required-features": req} for n, req in (("x", []), ("y", ["fy"]))]
        meta = {
            "workspace_root": tmp,
            "workspace_members": ["g"],
            "packages": [{"id": "g", "name": "g", "manifest_path": os.path.join(tmp, "crates/g/Cargo.toml"), "features": {"fy": [], "fz": []}, "dependencies": [], "targets": targets}],
        }
        side = side_at(tmp, meta)
        whole = {"gate-par", "gate-ws", "gate-slash"}  # read whole: a parameter, no `-p`, a `pkg/feature`
        every = {"gate-x", "gate-xy", "gate-yz", "gate-lib", "gate-liby", "gate-all"} | whole
        fy = {"gate-xy", "gate-yz", "gate-liby", "gate-all"} | whole
        cases = [
            ("an ungated module", "plain.rs", every),
            ("an include on an inner attribute line", "../../../attribute.txt", every),
            ("a path literal on an inner attribute line", "../../../tools/attribute-data.txt", every),
            ("a module under feature = fy", "gated.rs", fy),
            ("a module under not(feature = fy)", "off.rs", {"gate-x", "gate-lib"} | whole),
            ("a module under any(test, feature = fy), which a test build compiles whatever the features", "either.rs", every),
            ("an inline module under all(fy, fz): what it declares", "inl/deep.rs", {"gate-yz", "gate-all"} | whole),
            ("an inline module under all(fy, fz): what it includes", "../data/d.txt", {"gate-yz", "gate-all"} | whole),
            # only the recipes that build bin x read its own tree; of those, the ones without fy leave xg out
            ("a bin's own module under feature = fy", "bin/xg.rs", {"gate-xy"} | whole),
        ]
        for what, rel, want in cases:
            sels, _, _ = select([os.path.normpath(os.path.join("crates/g/src", rel))], side, side, GATE_PREFIX)
            got = {x.recipe for x in sels}
            expect(got == want, f"module gate: {what} selects {sorted(got)}, not {sorted(want)}")
        for rel in ("attribute-test.txt", "tools/attribute-test-data.txt", "after-attribute.txt"):
            sels, _, _ = select([rel], side, side, GATE_PREFIX)
            got = {x.recipe for x in sels} - whole
            expect(got == {"gate-lib", "gate-liby", "gate-all"},
                   f"module gate: an attribute brace must leave {rel} test-only, selected {sorted(got)}")
        expect(not side.tree.unresolved, f"module gate: unresolved {side.tree.unresolved[:2]}")


def comment_only_self_test(expect) -> None:
    """A Rust file whose diff is comments or layout selects no gate (`CommentReads`), on a synthetic workspace. Package h (a lib, no
    doctest recipe) holds one module per case: each comment kind, a nested comment, a lifetime beside a char, a layout change (no gate), and
    the edits to a string, a raw string, a byte string, a lifetime's code and a name (a gate). Package g (a lib whose doctests gate-doc runs,
    a bin) holds the files the rule leaves whole: a doc comment in the lib, and the files an include macro names; with a plain comment in
    the lib, a doc comment in the bin and a commented-out include it still reads no change; the lever registry is always left whole. Every case is also run against a mutant of
    the rule (the rule stubbed, strings ignored, a char literal that eats a lifetime, each refusal removed), which must turn one red."""
    just = (
        "gate-lib:\n    ./tools/box.sh 'cargo test -p g --lib'\n"
        "gate-doc:\n    ./tools/box.sh 'cargo test -p g --doc'\n"
        "gate-h:\n    ./tools/box.sh 'cargo test -p h --lib'\n"
        "gate-levers:\n    ./tools/box.sh 'cargo test -p levers --lib'\n"
    )
    registry = "crates/levers/src/registry.rs"
    h_cases: dict[str, tuple[str, str, str | None]] = {  # module -> (base, head, the line printed for it; None: a gate reads it)
        "line": ("fn f() {}\n// a\n", "fn f() {}\n// b\n", COMMENT_ONLY),
        "outer_doc": ("/// a\nfn f() {}\n", "/// b\nfn f() {}\n", COMMENT_ONLY),
        "inner_doc": ("//! a\nfn f() {}\n", "//! b\nfn f() {}\n", COMMENT_ONLY),
        "block": ("fn f() {} /* a */\n", "fn f() {} /* b */\n", COMMENT_ONLY),
        "block_doc": ("/** a */\nfn f() {}\n", "/** b */\nfn f() {}\n", COMMENT_ONLY),
        "inner_block_doc": ("/*! a */\nfn f() {}\n", "/*! b */\nfn f() {}\n", COMMENT_ONLY),
        "nested": ("fn f() {} /* a /* b */ c */\n", "fn f() {} /* a /* b */ d */\n", COMMENT_ONLY),
        "added": ("fn f() {}\n", "// new\nfn f() {}\n", COMMENT_ONLY),
        "lifetime_char": ("fn f<'a>(x: u8, y: &'a str) -> char { '/' } // c\n", "fn f<'a>(x: u8, y: &'a str) -> char { '/' } // d\n", COMMENT_ONLY),
        "lifetime_span": ("fn f<'a>(x: u8 /* c */, y: &'a str) {}\n", "fn f<'a>(x: u8 /* d */, y: &'a str) {}\n", COMMENT_ONLY),
        "char_quote": ("const C: char = '\"'; // a\n", "const C: char = '\"'; // b\n", COMMENT_ONLY),
        "layout": ("fn f(a: u8,b: u8) -> u8 { a+b }\n", "fn f(\n    a: u8,\n    b: u8\n) -> u8 {\n    a + b\n}\n", LAYOUT_ONLY),
        "string": ('const S: &str = "// not a comment";\n', 'const S: &str = "// not a comment either";\n', None),
        "raw_star": ('const S: &str = r#"/* a */"#;\n', 'const S: &str = r#"/* b */"#;\n', None),
        "raw_hash": ('const S: &str = r##"a "# /* x */"##;\n', 'const S: &str = r##"a "# /* y */"##;\n', None),
        "bytes": ('const S: &[u8] = b"// a";\n', 'const S: &[u8] = b"// b";\n', None),
        "string_layout": ('const S: &str = "a\nb";\n', 'const S: &str = "a\n b";\n', None),
        "lifetime_code": ("fn f<'a>(x: u8, y: &'a str) {}\n", "fn f<'a>(x: u16, y: &'a str) {}\n", None),
        "code": ("fn f() {}\n", "fn g() {}\n", None),
        "unterminated": ("fn f() {}\n", "fn f() {}\n/* never ends\n", None),
    }
    g_lib = (
        "mod doc_lib;\nmod plain_g;\nmod inc;\n"
        'const SELF_TEXT: &str = include_str!("inc.rs");\nconst BYTES: &[u8] = include_bytes!("bytes_src.rs");\n'
        'include!("items.rs");\n'
        'const TAIL: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/tail.rs"));\n'
        '// const GONE: &str = include_str!("gone.rs");\n'
    )
    # file -> (base, head, the line printed for it, a word of the note when the rule leaves it whole, the recipes that then read it)
    g_cases: dict[str, tuple[str, str, str | None, str, set[str]]] = {
        "doc_lib": ("/// a\npub fn f() {}\n", "/// b\npub fn f() {}\n", None, "doctests gate-doc runs", {"gate-lib", "gate-doc"}),
        "plain_g": ("// a\nfn f() {}\n", "// b\nfn f() {}\n", COMMENT_ONLY, "", set()),
        "bin/gb": ("/// a\nfn main() {}\n", "/// b\nfn main() {}\n", COMMENT_ONLY, "", set()),
        "inc": ("// a\nfn x() {}\n", "// b\nfn x() {}\n", None, "crates/g/src/lib.rs reads its text", {"gate-lib", "gate-doc"}),
        "bytes_src": ("// a\n", "// b\n", None, "crates/g/src/lib.rs reads its text", set()),
        "items": ("fn it() {}\n// a\n", "fn it() {}\n// b\n", None, "crates/g/src/lib.rs reads its text", set()),
        "tail": ("// a\n", "// b\n", None, "crates/g/src/lib.rs reads its text", set()),
        "gone": ("// a\n", "// b\n", COMMENT_ONLY, "", set()),
    }
    with tempfile.TemporaryDirectory(prefix="recipes-comments-") as tmp:
        made: list[str] = []

        def build(head: bool, h_deps: tuple[str, ...] = ()) -> Side:
            root = os.path.join(tmp, f"t{len(made)}")
            made.append(root)
            i = 1 if head else 0
            files = {
                "crates/g/Cargo.toml": "", "crates/h/Cargo.toml": "", "crates/levers/Cargo.toml": "", "justfile": just, "tools/gate.sh": "",
                "crates/levers/src/lib.rs": "mod registry;\n", registry: f"// {i}\nfn r() {{}}\n",
                "crates/g/src/lib.rs": g_lib, "crates/g/src/bin/gb.rs": g_cases["bin/gb"][i],
                "crates/h/src/lib.rs": "".join(f"mod {m};\n" for m in h_cases) + "mod same;\n" + ("mod added2;\n" if head else "mod removed2;\n"),
                "crates/h/src/same.rs": "fn s() {}\n",
                "crates/h/README.md": f"{i}\n",
                "fuzz/src/lib.rs": f"// {i}\n",
                ("crates/h/src/added2.rs" if head else "crates/h/src/removed2.rs"): "fn x() {}\n",
            }
            files.update({f"crates/h/src/{m}.rs": c[i] for m, c in h_cases.items()})
            files.update({f"crates/g/src/{m}.rs": c[i] for m, c in g_cases.items() if m != "bin/gb"})
            for f in BOX_GLOBALS + [f"tools/ref/models/{DEFAULT_PROFILE}.sh"]:
                files[f] = ""
            for rel, text in files.items():
                os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
                with open(os.path.join(root, rel), "w", encoding="utf-8") as fh:
                    fh.write(text)

            def pkg(name: str, src: str, kind: str, deps: tuple[str, ...] = ()) -> dict:
                return {"id": name, "name": name, "manifest_path": os.path.join(root, f"crates/{name}/Cargo.toml"), "features": {},
                        "dependencies": [{"name": d, "kind": None, "optional": False, "features": [], "uses_default_features": True} for d in deps],
                        "targets": [{"kind": [kind], "name": name, "src_path": os.path.join(root, f"crates/{name}/src/{src}")}]
                        + ([{"kind": ["bin"], "name": "gb", "src_path": os.path.join(root, "crates/g/src/bin/gb.rs")}] if name == "g" else [])}

            meta = {"workspace_root": root, "workspace_members": ["g", "h", "levers"],
                    "packages": [pkg("g", "lib.rs", "lib"), pkg("h", "lib.rs", "lib", h_deps), pkg("levers", "lib.rs", "lib")]}
            return side_at(root, meta)

        a, b = build(False), build(True)
        clap_a, clap_b = build(False, ("clap",)), build(True, ("clap",))
        everything = [f"crates/h/src/{m}.rs" for m in h_cases] + [f"crates/g/src/{m}.rs" for m in g_cases] + [
            "crates/h/src/same.rs", "crates/h/src/added2.rs", "crates/h/src/removed2.rs", "crates/h/README.md", "fuzz/src/lib.rs", registry]

        def cases(expect) -> None:
            reads = CommentReads(everything, a, b)
            for m, (_, _, want) in h_cases.items():
                f = f"crates/h/src/{m}.rs"
                expect(reads.only.get(f) == want, f"comment-only: h/{m} reads {reads.only.get(f)!r}, not {want!r}")
                sels, unmapped, _ = select([f], a, b, GATE_PREFIX, TableReads([f], a, b))
                if want is None:
                    expect([x.recipe for x in sels] == ["gate-h"], f"comment-only: h/{m} selects {[x.recipe for x in sels]}, not gate-h")
                else:
                    expect(not sels and any(u.startswith(f) and want in u for u in unmapped), f"comment-only: h/{m} selects {[x.recipe for x in sels]}, unmapped {unmapped}")
            for m, (_, _, want, word, readers) in g_cases.items():
                f = f"crates/g/src/{m}.rs"
                expect(reads.only.get(f) == want, f"comment-only: g/{m} reads {reads.only.get(f)!r}, not {want!r}")
                note = next((n for n in reads.notes if n.startswith(f + ":")), "")
                expect((word in note) if want is None else not note, f"comment-only: g/{m} note {note!r}, want {word!r}")
                sels, _, _ = select([f], a, b, GATE_PREFIX, TableReads([f], a, b))
                if want is None and readers:
                    expect({x.recipe for x in sels} == readers, f"comment-only: g/{m} selects {sorted(x.recipe for x in sels)}, not {sorted(readers)}")
                if want is not None:
                    expect(not sels, f"comment-only: g/{m} selects {[x.recipe for x in sels]}")
            for f in ("crates/h/src/same.rs", "crates/h/src/added2.rs", "crates/h/src/removed2.rs", "crates/h/README.md", "fuzz/src/lib.rs"):
                expect(f not in reads.only and not any(n.startswith(f + ":") for n in reads.notes), f"comment-only: {f} (equal, added, removed, not Rust, no crate) was read as comment-only")
            expect(registry not in reads.only and any(n.startswith(registry + ":") and "lever registry" in n for n in reads.notes),
                   f"comment-only: a comment moved in the lever registry reads {reads.only.get(registry)!r}, notes {reads.notes}")
            off = CommentReads(everything, a, b, False)
            expect(not off.only and not off.notes, f"comment-only: the rule off still reads {sorted(off.only)}")
            # a crate whose derive reads doc comments: a doc comment moved in any crate is no neutral edit, a plain comment still is
            clap = CommentReads(["crates/h/src/outer_doc.rs", "crates/h/src/line.rs"], clap_a, clap_b)
            expect("crates/h/src/outer_doc.rs" not in clap.only and "crates/h/src/line.rs" in clap.only
                   and any("depends on clap" in n for n in clap.notes), f"comment-only: a doc reader in the workspace: {clap.only}, {clap.notes}")

        cases(expect)
        # --narrow: a comment-only file picks nothing and prints its line; beside a code edit only the code edit's gates stay
        log = os.path.join(tmp, "scan.log")
        with open(log, "w", encoding="utf-8") as fh:
            fh.write("\n".join(["ptx-scan: mod1 bundle=bloomery-gpu bytes=100",
                                "ptx-scan bin=target/release/gx section=.oxart bytes=9 ptxas=/p ptxas-version=13.3.73 arch=sm_86 modules=1 "
                                "jit-card=NVIDIA_RTX_A6000 jit-cc=8.6 jit-cuda=13.4",
                                "entry reqntid depot ld.local st.local fma cvt.f16 regs smem spill blk/SM(static) jit_regs jit_local",
                                f"{'alpha':<24} 256 no 0 0 0 0 12 0 0 6 12 0", "ptx-scan-md5: method=decl1", f"alpha {'1' * 32} 41"]) + "\n")
        pair = (read_scan(log), read_scan(log))
        quiet_f, code_f = "crates/h/src/line.rs", "crates/h/src/code.rs"
        nar = narrow([quiet_f], a, b, [pair], [])
        expect(not nar.full and not nar.picks and nar.files == [f"{quiet_f}: {COMMENT_ONLY}"], f"comment-only: --narrow on a comment-only file: full {nar.full}, picks {nar.picks}, files {nar.files}")
        nar = narrow([quiet_f, code_f], a, b, [pair], [])
        expect(not nar.full and set(nar.picks) == {"gate-h"} and nar.files[0] == f"{quiet_f}: {COMMENT_ONLY}" and not nar.files[1].endswith(COMMENT_ONLY),
               f"comment-only: --narrow beside a code edit: full {nar.full}, picks {sorted(nar.picks)}, files {nar.files}")
        nar = narrow([quiet_f], a, b, [pair], [], TableReads([quiet_f], a, b, False))
        expect(set(nar.picks) == {"gate-h"}, f"comment-only: --narrow with the rule off picks {sorted(nar.picks)}, not gate-h")

        # every case against a mutant of the rule: the cases must go red
        g = globals()
        real_stream = _rs_stream

        def string_blind(text: str) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[int, str], ...]]:
            """A lexer that reads a comment wherever `//` or `/*` appears, strings included, and does not nest."""
            toks: list[tuple[str, str]] = []
            comments: list[tuple[int, str]] = []
            pos = 0
            for m in re.finditer(r"//[^\n]*|/\*.*?\*/", text, re.S):
                toks += [("p", w) for w in text[pos:m.start()].split()]
                comments.append((len(toks), m.group()))
                pos = m.end()
            toks += [("p", w) for w in text[pos:].split()]
            return tuple(toks), tuple(comments)

        greedy_char = re.compile(_RS_TOKEN.pattern.replace(r"(?P<chr>b?'(?:\\.[^']*|[^\\'])')", r"(?P<chr>b?'[^']*')"), re.S)
        mutants: list[tuple[str, list[tuple[dict, str, object]]]] = [
            ("the rule stubbed out", [(g, "comment_diff", lambda x, y: None)]),
            ("strings ignored, comments not nested", [(g, "_rs_stream", string_blind)]),
            ("a char literal that eats a lifetime", [(g, "_RS_TOKEN", greedy_char)]),
            ("includes not read", [(CommentReads, "_include_targets", lambda self: ({}, []))]),
            ("doctests not read", [(CommentReads, "_doctest_runs", lambda self: {})]),
            ("doc comments not told from the others", [(g, "_is_doc_comment", lambda c: False)]),
            ("doc readers not read", [(g, "DOC_READER_CRATES", frozenset())]),
            ("the lever registry not refused", [(g, "LEVER_REGISTRY", "crates/levers/src/other.rs")]),
        ]
        def put(o: object, k: str, v: object) -> None:
            if isinstance(o, dict):
                o[k] = v
            else:
                setattr(o, k, v)

        for name, patches in mutants:
            saved = [(o, k, o[k] if isinstance(o, dict) else getattr(o, k)) for o, k, _ in patches]
            for o, k, v in patches:
                put(o, k, v)
            real_stream.cache_clear()
            red: list[str] = []
            try:
                cases(lambda ok, what: None if ok else red.append(what))
            finally:
                for o, k, v in saved:
                    put(o, k, v)
                real_stream.cache_clear()
            expect(bool(red), f"comment-only: the cases stayed green against a mutant ({name})")


def key_detectors_self_test(expect, real: Side) -> None:
    """The ledger key's detectors on the real tree: which gates can skip, which are keyed on the whole tree."""
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
    for n in ("check", "fmt-check", "check-recipes", "check-comments", "check-levers", "check-arch", "check-model-tax", "check-rustflags"):
        expect(ctx.scope(recipe_closure(real.recipes, n), real.graph.inputs(n))[0] == "tree", f"{n} is not keyed on the whole tree")
    expect(ctx.scope(["lint"], real.graph.inputs("lint"))[0] == "closure", "lint (clippy, modeled) is keyed on the whole tree")
    for s in ("tools/check-comments.sh", "tools/check-levers.sh", "tools/check-arch.sh", "crates/tokenizer/tools/oracle.sh"):
        expect(scan_lines(real.tree.read(s), TREE_WALK, s) is not None, f"the walk detector misses {s}")
    quiet = ["tools/box.sh", "tools/gate.sh", "tools/gpu-gate.sh", "tools/host-gate.sh", "tools/ref/ref-paths.sh", "tools/ptx-spill-check.sh", "tools/ptx-scan.sh"]
    quiet += sorted(os.path.relpath(p, ROOT) for p in glob.glob(os.path.join(ROOT, "tools/ref/models/*.sh")))
    for s in quiet:
        hit = scan_lines(real.tree.read(s), TREE_WALK, s)
        expect(hit is None, f"the walk detector flags {hit}: every gate that runs it would be keyed on the whole tree")


def key_self_test(expect, real: Side, section: str | None = None) -> None:
    """Independent key contracts, each on its own scratch tree and manifest."""
    from unittest.mock import patch

    assert section in (None, "inputs", "globals", "fixtures", "variants", "scope"), section
    meta = cargo_metadata(ROOT)
    settings = justfile_settings(os.path.join(ROOT, "justfile"))
    with tempfile.TemporaryDirectory(prefix="recipes-keytest-") as tmp, patch.dict(os.environ, {"HOME": tmp}):
        roots = [os.path.join(tmp, "a")]
        for rel in shipped_files(ROOT):
            src, dst = os.path.join(ROOT, rel), os.path.join(roots[0], rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.islink(src):
                os.symlink(os.readlink(src), dst)
            elif stat.S_ISREG(os.lstat(src).st_mode):  # a pipe a killed run left in the tree (narrow-scan's build fifo) is no input
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

        two = {k: items[k] for k in ("gpu", "mac")}
        one = lambda it: keys(sa, only={"x": it})["x"]  # noqa: E731
        fx_env = {"BLOOMERY_TIER": "fixture"}
        fx_file = f"model\t{FIXTURE_ROOT}/qwen38/q-00001-of-00001.gguf\t100\t1\t2\t3"
        fixture_free = [ln for ln in manifest[:-1] if not ln.startswith(f"model-dir\t{FIXTURE_ROOT}\t")]

        if section is None or section == "inputs":
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

            m = edit("crates/gpu-gates/src/bin/gate_q4k_sel.rs", append("\n// key probe\n"))
            expect(m == {"gpu", "mac", "fmt"}, f"gate_q4k_sel.rs moves {sorted(m)}, not gpu mac fmt")
            m = edit("tools/gpu-gate.sh", append("\n# key probe\n"))
            expect(m == {"gpu", "cpu", "args", "mac", "fmt"}, f"tools/gpu-gate.sh moves {sorted(m)}")
            m = edit("docs/HARDWARE.md", append("\nkey probe\n"))
            expect(m == {"mac", "fmt"}, f"docs/HARDWARE.md moves {sorted(m)}, not only the whole-tree keys")
            cpu_lib = sorted(f for f in sa.graph.inputs("gate-sampler").files if f.startswith("crates/sampler/src/"))
            m = edit(cpu_lib[0], append("\n// key probe\n"))
            expect({"cpu", "mac", "fmt"} <= m and "gpu" not in m, f"{cpu_lib[0]} moves {sorted(m)}")
        if section is None or section == "globals":
            for g in KEY_GLOBALS:
                m = edit(g, append("\n# key probe\n"))
                expect(m == set(items), f"{g} moves {sorted(m)}, not every key")
        if section is None or section == "inputs":
            m = edit("justfile", append("\n# key probe\n"), side_fresh=True)
            expect(m == {"mac", "fmt"}, f"a justfile comment moves {sorted(m)}")
            m = edit("justfile", lambda s: s.replace("bash tools/gpu-gate.sh gate_q4k_sel'", "bash tools/gpu-gate.sh gate_q4k_sel '", 1), side_fresh=True)
            expect("gpu" in m and "cpu" not in m and "args" not in m, f"gate-gpu-q4k-sel's recipe text moves {sorted(m)}")

            expect(one("gate-gpu-q4k-sel@BLOOMERY_GATE_CARD=3090,X=1") != base["gpu"], "an item env entry does not move the key")
            expect(one("gate-gpu-q4k-sel@BLOOMERY_GATE_CARD=a6000") != base["gpu"], "the lane's card does not move the key")
            expect(one("gate-gpu-q4k-sel") != base["gpu"], "no card (--lanes 1) gives the 3090 lane's key")
            expect(one("gate-gpu-e2e@BLOOMERY_GATE_CARD=a6000:--y") != base["args"], "ARGS do not move the key")
            expect(moved(keys(sa, box_env="BLOOMERY_X=1")) == set(items), "the caller's BLOOMERY_BOX_ENV does not move every key")
            expect(moved(keys(sa, environ={"BLOOMERY_DATA": "/x"})) == set(items), "the Mac's BLOOMERY_DATA does not move every key")
            # The tier: one item under the real tier and under the fixture tier is two keys, whichever way the tier is named, and unset
            # and `real` are one — a fixture green must never stand for the real file's, nor `--tier real` split the real greens.
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
        if section is None or section == "fixtures":
            # The fixture directory's rows belong to the fixture tier's keys alone: a real key does not move with them, a fixture key
            # does, and the row of a file inside a family's directory is one (the manifest lists below a subdirectory there).
            # PIN(2026-10-13): re-pinned, not loosened — a key holds its family's fixture directory (KeyContext.fixture_dirs), so a
            # fixture file moves the items of its family only; these items run under qwen4exp (no recipe of `items` names a
            # BLOOMERY_MODEL), and a Mac-run item, whose family cannot be told, keeps every row. The base is the manifest with the root
            # present, so every key is a key and not an error string. FAIL-first: the family clauses below fail on b1f2c3d1's code.
            fx4 = {"BLOOMERY_TIER": "fixture", "BLOOMERY_MODEL": "qwen4exp"}
            write_manifest(manifest)
            real_before, fix_before = keys(sa), keys(sa, environ=fx4)
            expect(not any(v.startswith("error") for v in fix_before.values()), f"the fixture-tier keys on the base manifest: {fix_before}")
            write_manifest(fixture_free + [fx_file, "# end"])
            expect(not moved(keys(sa)), "a fixture file's manifest row moved a real-tier key")
            fix_file = keys(sa, environ=fx4)
            expect(not any(v.startswith("error") for v in fix_file.values()) and all(fix_file[k] != fix_before[k] for k in items),
                   "a fixture file's manifest row did not move every fixture-tier key of the family whose directory it is in")
            write_manifest(fixture_free + [fx_file.replace("\t100\t", "\t101\t"), "# end"])
            rewritten_keys = keys(sa, environ=fx4)
            expect(all(rewritten_keys[k] != fix_file[k] for k in items), "a fixture file rewritten (its size) moved no fixture-tier key")
            write_manifest(manifest)
            expect(keys(sa) == base, "the base manifest, restored, keys another way")
            # A fixture-tier item's key holds its family's fixture directory, from the family's row of the fixture directory table (FIXTURE_DIRS, the
            # one owner), as a variant's below holds its tag's: a file in a directory no family of the item maps to moves nothing of it.
            rows, default = fixture_table(sa.tree.read(FIXTURE_DIRS), sa.tree.read(REF_PATHS))
            expect(default == DEFAULT_PROFILE, f"ref-paths.sh's default profile {default} is not DEFAULT_PROFILE {DEFAULT_PROFILE}")
            expect([rows.get(f) for f in ("qwen4exp", "deepseek41", "glm5next", "qwen35moe", "deepseek2")] == ["qwen38", "v41", "glm5next", "qwen35moe", "self"]
                   and "mimo2" not in rows and rows["qwen35"] == "self", f"the real fixture table, read: {rows}")
            fam = {
                "q4": "gate-gpu-qwen4exp-e2e",  # BLOOMERY_MODEL=qwen4exp: qwen38
                "glm": "gate-gpu-glm-sel",  # glm5next: glm5next
                "ds41": "gate-gpu-ds41-kernels",  # deepseek41: v41, its draft/ below it
                "q35": "gate-gpu-qwen35moe-moe",  # qwen35moe: its own directory
                "mimo": "gate-gpu-mimo2-flash",  # mimo2: no row, `none`
                "envm": "gate-gpu-glm-sel@BLOOMERY_MODEL=qwen4exp",  # a BLOOMERY_MODEL in the item's env: no telling
                "mac": "check-comments",  # a command run on the Mac: no telling
            }
            open_keys = {"envm", "mac"}

            def fx_rows(sizes: dict[str, int]) -> list[str]:
                out: list[str] = []
                for d, size in sizes.items():
                    out += [f"model\t{FIXTURE_ROOT}/{d}\tdir", f"model\t{FIXTURE_ROOT}/{d}/{d}-00001-of-00001.gguf\t{size}\t1\t2\t3"]
                    if d == "v41":
                        out += [f"model\t{FIXTURE_ROOT}/v41/draft\tdir", f"model\t{FIXTURE_ROOT}/v41/draft/d.gguf\t{size + 1}\t1\t2\t3"]
                return out

            def fam_keys(rows_: list[str], env: dict | None = None) -> dict[str, str]:
                write_manifest(fixture_free + rows_ + ["# end"])
                return keys(sa, environ=fx_env if env is None else env, only=fam)

            sizes = {"qwen38": 30, "glm5next": 10, "v41": 50, "qwen35moe": 20}
            kf = fam_keys(fx_rows(sizes))
            expect(not any(v.startswith("error") for v in kf.values()), f"the family items under the fixture tier: {kf}")

            def rewritten(rows_: list[str], env: dict | None = None, base_: dict[str, str] | None = None) -> set[str]:
                now = fam_keys(rows_, env)
                return {k for k in fam if now[k] != (kf if base_ is None else base_)[k]}

            bump = lambda d: fx_rows({**sizes, d: sizes[d] + 1})  # noqa: E731
            m = rewritten(bump("qwen35moe"))
            expect("q4" not in m, f"a file under {FIXTURE_ROOT}/qwen35moe moved the key of a qwen4exp item")  # (a)
            expect(m == {"q35"} | open_keys, f"a file under qwen35moe moved {sorted(m)}, not its item and the unknown-family ones")  # (b)
            m = rewritten(bump("qwen38"))
            expect(m == {"q4"} | open_keys, f"a file under qwen38 moved {sorted(m)}: the qwen4exp item and the unknown-family ones")  # (c)
            m = rewritten(bump("glm5next"))
            expect(m == {"glm"} | open_keys, f"a file under glm5next moved {sorted(m)}")
            m = rewritten(bump("v41"))
            expect(m == {"ds41"} | open_keys, f"a file under v41 moved {sorted(m)}")
            expect(rewritten([r.replace("d.gguf\t51", "d.gguf\t52") for r in fx_rows(sizes)]) == {"ds41"} | open_keys, "v41's draft/ file (one level down) did not move exactly the deepseek41 item")
            # (d) an item whose family cannot be told keeps every row: any file, a new family's directory included, moves it
            newfam = fx_rows(sizes) + [f"model\t{FIXTURE_ROOT}/qwen38-old\tdir", f"model\t{FIXTURE_ROOT}/qwen38-old/n.gguf\t1\t1\t2\t3"]
            m = rewritten(newfam)
            expect(m == open_keys, f"a new directory (qwen38-old) moved {sorted(m)}, not exactly the items whose family cannot be told")
            # the root with no other family's directory, or with none, keys a told item the same: the root's own row is nobody's
            only35 = fam_keys(fx_rows({"qwen35moe": 20}))
            expect(not any(v.startswith("error") for v in only35.values()), f"a manifest with another family's directory only (no qwen38/): {only35}")
            empty = fam_keys([f"model-dir\t{FIXTURE_ROOT}\tempty"])
            expect(only35["q4"] == empty["q4"] and only35["mimo"] == empty["mimo"] and only35["mac"] != empty["mac"],
                   "a fixture root holding another family's directory only keys a told item another way than an empty root")
            # the Mac's BLOOMERY_MODEL is the family of an item whose recipes name none (fmt-check: a box call with no BLOOMERY_MODEL)
            def fmt_moved(name: str) -> bool:
                env = {"BLOOMERY_TIER": "fixture", "BLOOMERY_MODEL": name}
                write_manifest(fixture_free + fx_rows(sizes) + ["# end"])
                before = keys(sa, environ=env, only={"fmt": items["fmt"]})["fmt"]
                write_manifest(fixture_free + bump("qwen38") + ["# end"])
                return keys(sa, environ=env, only={"fmt": items["fmt"]})["fmt"] != before

            got = {n: fmt_moved(n) for n in ("qwen4exp", "glm5next", "qwen35moe")}
            expect(got == {"qwen4exp": True, "glm5next": False, "qwen35moe": False}, f"fmt-check, a qwen38 file rewritten, under the Mac's BLOOMERY_MODEL: moved {got}")
            _, err, _, _ = KeyContext(sa, mf, environ={"BLOOMERY_TIER": "fixture", "BLOOMERY_MODEL": "a b"}, settings=settings).parts(items["fmt"])
            expect(err is not None and "BLOOMERY_MODEL='a b': cannot resolve model profile" in err, f"fmt-check: invalid inherited profile not refused by name: {err}")
            # (e) the real tier holds no fixture row: no rewrite moves a key
            real_f = fam_keys(fx_rows(sizes), {})
            expect(all(rewritten(bump(d), {}, real_f) == set() for d in sizes), "a fixture file's rewrite moved a real-tier key")
            # the key part names what the fixture rows were taken from: the debuggability of "moved since"
            fparts = lambda it, env: [p for p in KeyContext(sa, mf, "", env, settings=settings).parts(it)[0] if p.startswith("fixture\t")]  # noqa: E731
            write_manifest(fixture_free + fx_rows(sizes) + ["# end"])
            expect([fparts(fam[k], fx_env) for k in ("q4", "ds41", "q35", "envm")] == [["fixture\tdirs\tqwen38"], ["fixture\tdirs\tv41"], ["fixture\tdirs\tqwen35moe"], ["fixture\tdirs\t*"]]
                   and fparts(fam["q4"], {}) == [], "the fixture key part does not name the directories the key holds")
            # (f) the table is tools/fixture-dirs.tsv's. qwen35moe already names its own directory. Pointing that row back at
            # `self` edits the file, so it moves the keys of the box items (the file is a key part of every one of
            # them). Against that table a file under the qwen35moe directory is a directory no family maps to.
            rp = os.path.join(roots[0], FIXTURE_DIRS)
            with open(rp, encoding="utf-8") as fh:
                rp_old = fh.read()
            own = "qwen35moe\tqwen35moe\n"
            selfrow = "qwen35moe\tself\n"
            refused = [
                (rp_old + "qwen35moe\tself\n", "in the table twice"),
                (rp_old.replace("qwen4exp\tqwen38\n", "qwen4exp\tqwen38\tx\n"), "is not `<family><TAB><dir>`"),
                (rp_old.replace("qwen4exp\tqwen38\n", "qwen4*\tqwen38\n"), "is not `<family><TAB><dir>`"),
                (rp_old.replace("qwen4exp\tqwen38\n", "qwen4exp\tnone\n"), "never `none`"),
                (rp_old.replace("qwen4exp\tqwen38\n", "qwen4exp\tQwen38\n"), "is not `<family><TAB><dir>`"),
                ("# no rows\n", "no row"),
            ]
            try:
                expect(own in rp_old and selfrow not in rp_old, f"{FIXTURE_DIRS}'s qwen35moe row is not the one this clause edits")
                edited = rp_old.replace(own, selfrow)
                expect(edited != rp_old and selfrow in edited and own not in edited, "pointing qwen35moe back at self did not edit the table")
                with open(rp, "w", encoding="utf-8") as fh:
                    fh.write(edited)
                kf_row = fam_keys(fx_rows(sizes))
                expect(kf_row["q4"] != kf["q4"] and kf_row["q35"] != kf["q35"] and kf_row["glm"] != kf["glm"], f"an edit of {FIXTURE_DIRS} moved no key of a box item")
                m = rewritten(bump("qwen35moe"), None, kf_row)
                expect(m == open_keys, f"qwen35moe pointed back at self: a file under its directory moved {sorted(m)}, not only the items whose family cannot be told {sorted(open_keys)}")
                for text, why in refused:
                    with open(rp, "w", encoding="utf-8") as fh:
                        fh.write(text)
                    got = fam_keys(fx_rows(sizes))["q4"]
                    expect(got.startswith("error") and why in got, f"a fixture table this reader does not understand ({why}) was keyed: {got}")
            finally:
                with open(rp, "w", encoding="utf-8") as fh:
                    fh.write(rp_old)
            expect(fam_keys(fx_rows(sizes)) == kf, f"{FIXTURE_DIRS} restored keys another way")
            write_manifest(manifest)
            expect(keys(sa) == base, "the base manifest, restored after the family clauses, keys another way")
        if section is None or section == "variants":
            # A fixture variant's files (VARIANT_ROOT, beside FIXTURE_ROOT) belong to the keys of the items that run under its tag: no other
            # item's key moves with them, an item under the tag moves with its own directory's rows and its table row, and not with another tag's.
            vdirs = lambda tag: [f"model\t{VARIANT_ROOT}/{tag}\tdir", f"model\t{VARIANT_ROOT}/{tag}/glm5next\tdir"]  # noqa: E731
            vfile = lambda tag, size: f"model\t{VARIANT_ROOT}/{tag}/glm5next/g-00001-of-00001.gguf\t{size}\t1\t2\t3"  # noqa: E731
            vtable = os.path.join(roots[0], VARIANT_TABLE)
            with open(vtable, encoding="utf-8") as fh:
                vtable_old = fh.read()
            row_a = "iq4xs\tglm5next\ttrunk\tffn_up_exps=iq3_s\tgate-gpu-q4k-sel\tprobe"
            row_b = "other\tglm5next\ttrunk\tffn_up_exps=iq3_s\tgate-gpu-q4k-sel\tprobe"
            venv = {"BLOOMERY_TIER": "fixture", "BLOOMERY_FIXTURE_VARIANT": "iq4xs"}
            vparts = lambda e, box="": [p for p in KeyContext(sa, mf, box, e, settings=settings).parts(items["gpu"])[0] if p.startswith("variant\t")]  # noqa: E731
            variant_free = [ln for ln in fixture_free if not ln.startswith(f"model-dir\t{VARIANT_ROOT}\t")]
            try:
                with open(vtable, "w", encoding="utf-8") as fh:
                    fh.write(f"# probe\n{row_a}\n{row_b}\n")
                # The table is a file of the tree (a tree-scope key holds it): the probe table moves those keys, so the clauses below compare
                # against the keys taken with the probe table in place and no variant rows in the manifest.
                write_manifest(variant_free + [fx_file, "# end"])
                real_plain, fix_plain = keys(sa), keys(sa, environ=fx_env)
                err = keys(sa, environ=venv)["gpu"]
                expect(err.startswith("error") and VARIANT_ROOT in err, f"an item under a variant keyed with no row for {VARIANT_ROOT}: {err}")
                both = variant_free + [fx_file] + vdirs("iq4xs") + [vfile("iq4xs", 100)] + vdirs("other") + [vfile("other", 100), "# end"]
                write_manifest(both)
                expect(keys(sa) == real_plain, "a variant's files moved a key of a real-tier item")
                expect(keys(sa, environ=fx_env) == fix_plain, "a variant's files moved a key of a fixture-tier item that runs under no variant")
                with_files = keys(sa, environ=venv)
                expect(not any(v.startswith("error") for v in with_files.values()), f"an item under a variant: {with_files}")
                expect(all(with_files[k] != fix_plain[k] for k in items), "an item under a variant keys like the fixture tier's item")
                write_manifest([ln.replace(f"iq4xs/glm5next/g-00001-of-00001.gguf\t100", f"iq4xs/glm5next/g-00001-of-00001.gguf\t101") for ln in both])
                rewritten_keys = keys(sa, environ=venv)
                expect(all(rewritten_keys[k] != with_files[k] for k in items), "a variant file rewritten (its size) moved no key of the item under its tag")
                write_manifest([ln.replace(f"other/glm5next/g-00001-of-00001.gguf\t100", f"other/glm5next/g-00001-of-00001.gguf\t101") for ln in both])
                expect(keys(sa, environ=venv) == with_files and keys(sa) == real_plain and keys(sa, environ=fx_env) == fix_plain,
                       "another variant's file rewritten moved a key that does not run under it")
                row = vparts(venv)
                expect(len(row) == 1 and row[0] == f"variant\tiq4xs\t{sha256_text(row_a)}", f"the item under a variant holds its table row as a key part: {row}")
                expect(vparts(fx_env) == [] and vparts({}) == [], "an item under no variant holds a variant part")
                expect(vparts({}, "BLOOMERY_TIER=fixture BLOOMERY_FIXTURE_VARIANT=iq4xs") == row, "a BLOOMERY_BOX_ENV entry names the variant as the Mac's does")
                with open(vtable, "w", encoding="utf-8") as fh:
                    fh.write(f"# probe\n{row_a} edited\n{row_b}\n")
                expect(vparts(venv) != row, "an edit of the variant's table row moved no key part")
                with open(vtable, "w", encoding="utf-8") as fh:
                    fh.write(f"# probe\n{row_b}\n")
                err = keys(sa, environ=venv)["gpu"]
                expect(err.startswith("error") and "names no variant 'iq4xs'" in err, f"a variant the table does not name was keyed: {err}")
            finally:
                with open(vtable, "w", encoding="utf-8") as fh:
                    fh.write(vtable_old)
                write_manifest(manifest)
            expect(keys(sa) == base, "the base manifest, restored after the variant clauses, keys another way")
        if section is None or section == "scope":
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
                reason = c.parts(f"{n}@BLOOMERY_GATE_CARD=a6000")[2]
                expect(reason is None, f"{n} with the A6000 forced never skips: {reason}")
            expect(c.parts("gate-sampler@BLOOMERY_REF_MODEL=/models/small/x.gguf")[2] is not None, "an item env naming a path was skippable")
            c2 = KeyContext(sa, mf, "BLOOMERY_KLD_FILE=/root/x.kld", {}, settings=settings)
            expect(c2.parts(items["cpu"])[2] is not None, "a caller env path outside data and /models was skippable")
            c3 = KeyContext(sa, mf, "BLOOMERY_DATA=/root/bloomery-data", {}, settings=settings)
            expect(c3.parts(items["cpu"])[2] is None, "the data directory itself made an item never-skip")
            expect(c.parts("gate-1-1")[2] is not None, "an ik-reading gate was skippable")
            reason = c.parts("gate-tokenizer")[2]
            expect(reason is None, f"gate-tokenizer reads its sets, no reference tree, and was never-skip: {reason}")
            expect(c.parts("smoke")[2] is not None, "smoke (a batch in a batch) was skippable")
            # The cancellation fixtures name Clefvis and the tokenizer oracle's self-test runs oracle.sh: both scripts read a
            # reference tree, so check-recipes is never skipped (the reason names the first one the closure reaches).
            reason = c.parts("check-recipes")[2]
            expect("reads a reference tree" in (reason or ""), f"check-recipes did not name a reference-tree read: {reason}")
            expect(c.parts("gate-gpu-e2e")[2] is not None, "an `any`-card recipe with no card forced was skippable")
            expect(c.parts("gate-gpu-e2e@BLOOMERY_GATE_CARD=a6000:--in /root/x.bin")[2] is not None, "an absolute path in ARGS was skippable")
            argpath = "gate-gpu-e2e@BLOOMERY_GATE_CARD=a6000:--in docs/HARDWARE.md"
            plan = os.path.join(roots[0], "docs/HARDWARE.md")
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
            write_manifest([ln for ln in manifest if not ln.startswith("model-dir\t/models/DeepSeek-V4.1-Flash-1017-Q3_K_M\t")])
            miss = keys(sa, only={"gpu": items["gpu"]})["gpu"]
            expect(miss.startswith("error") and "/models/DeepSeek-V4.1-Flash-1017-Q3_K_M" in miss, f"a directory the item opens, missing from the manifest, was keyed: {miss}")
            write_manifest(manifest)
            c = KeyContext(sa, mf, "", {}, settings=settings)
            rows = [p for p in c.parts(items["mac"])[0] if p.startswith(("box\tmodel\t", "box\tmodel-dir\t"))]
            all_rows = [ln for ln in manifest if ln.startswith(("model\t", "model-dir\t"))]
            tier_rows = [ln for ln in all_rows if model_dir_of(ln.split("\t")[1]) == FIXTURE_ROOT]
            variant_rows = [ln for ln in all_rows if model_dir_of(ln.split("\t")[1]) == VARIANT_ROOT]
            expect(len(rows) == len(all_rows) - len(tier_rows) - len(variant_rows),
                   f"a whole-tree item holds {len(rows)} model rows, not the manifest's all but the fixture tier's {len(tier_rows)} and the variants' {len(variant_rows)}")
            fx_rows = [p for p in KeyContext(sa, mf, "BLOOMERY_TIER=fixture", {}, settings=settings).parts(items["mac"])[0] if p.startswith(("box\tmodel\t", "box\tmodel-dir\t"))]
            expect(len(fx_rows) == len(all_rows) - len(variant_rows),
                   f"a whole-tree item of the fixture tier holds {len(fx_rows)} model rows, not the manifest's all but the variants' {len(variant_rows)}")
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
                "#[cfg(card_fp4)]\nmod fp4 {\n    #[test]\n    fn t() {}\n}\n"
                "#[cfg(card_decl)]\nmod decl {\n    #[test]\n    fn t() {}\n}\n"
            ),
            "q/build.rs": (
                'fn main() {\n    println!("cargo:rustc-check-cfg=cfg(card_fp4)");\n    println!("cargo:rustc-check-cfg=cfg(card_decl)");\n'
                '    if std::env::var("CARD").is_ok() {\n        println!("cargo:rustc-cfg=card_fp4");\n    }\n}\n'
            ),
            "r/Cargo.toml": "",
            "r/src/lib.rs": "#[cfg(card_fp4)]\nmod fp4 {\n    #[test]\n    fn t() {}\n}\n",
            "r/build.rs": (
                'fn main() {\n    println!("cargo:rustc-check-cfg=cfg(card_fp4)");\n    println!("cargo:rustc-cfg=card_fp4");\n'
                '    println!("cargo:rustc-check-cfg=cfg(card_decl)");\n    println!("cargo:rustc-cfg=card_decl");\n}\n'
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
            "workspace_members": ["p", "q", "r"],
            "packages": [
                pkg("p", [("lib", "p", "src/lib.rs"), ("test", "it", "tests/it.rs")], {"f": [], "g": [], "default": []}),
                pkg("q", [("lib", "q", "src/lib.rs"), ("custom-build", "build-script-build", "build.rs")], {"g": []}),
                pkg("r", [("lib", "r", "src/lib.rs"), ("custom-build", "build-script-build", "build.rs")], {}),
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
            "lab-r": Recipe("lab-r", [f"cargo test -p r --lib && ./{MAC_GATE} gate_r"], [], ""),
        }
        orphans, errors, checked = orphan_tests(st, rs)
        names = sorted(o.split(" ")[1] for o in orphans)
        expect(names == ["common::shared", "feat::t", "integ", "mac::t"], f"orphan scan: orphans {names}")
        expect(any("p/src/lib.rs:12 feat::t" in o and 'cfg(feature = "f") at p/src/lib.rs:9 is false' in o for o in orphans), f"orphan scan: feat::t's line {orphans}")
        expect(any(o.startswith("p/tests/it.rs:3 integ — p test it: no gate-* or lab-* recipe") for o in orphans), f"orphan scan: a check-* recipe counted as a runner {orphans}")
        # a test under a cfg a build script declares and emits (q's card_fp4) is run by the box recipe that runs its
        # target, in the build that sets it; a name q only declares (card_decl, which r's build script emits for r alone)
        # and a Mac-only run (r) stay unknown
        expect(checked == 14, f"orphan scan: {checked} tests checked, not 14")
        # a recipe that runs MAC_GATE runs its tests on the Mac: the macOS module's test is run there, and the
        # feature-gated one stays an orphan
        mac_rs = dict(rs, **{"lab-p": Recipe("lab-p", [f"cargo test -p p --lib && ./{MAC_GATE} gate_p"], [], "")})
        mac_names = sorted(o.split(" ")[1] for o in orphan_tests(st, mac_rs)[0])
        expect(mac_names == ["common::shared", "feat::t", "integ"], f"orphan scan: a {MAC_GATE} recipe's run, orphans {mac_names}")
        for where, why in [
            ("q/src/lib.rs:1:", "cfg_attr"),
            ("q/src/lib.rs:6:", "macro_rules! mk"),
            ("q/src/lib.rs:10:", "tokio::test"),
            ("q/src/lib.rs:14 avx:", 'cfg(target_feature = "avx2") at q/src/lib.rs:12'),
            ("q/src/lib.rs:15:", "cfg(foo bar)"),
            ("q/src/lib.rs:18:", "include!"),
            ("q/src/lib.rs:27 decl::t:", "cfg(card_decl) at q/src/lib.rs:24 under gate-q"),
            ("r/src/lib.rs:4 fp4::t:", "cfg(card_fp4) at r/src/lib.rs:1 under lab-r"),
        ]:
            expect(any(x.startswith(where) and why in x for x in errors), f"orphan scan: no named error at {where} ({why}): {errors}")
        expect(len(errors) == 8, f"orphan scan: {len(errors)} errors, not 8: {errors}")

    # the real tree: gate-ds41-bind runs every bind test (the deepseek41 feature, the bind:: filter);
    # how many there are is the scan's own count of them, not a number spelled here
    gg_lib = next(t.src for t in real.tree.packages["bloomery-gpu-gates"].targets if t.kind == "lib")
    n_bind = sum(1 for t in TestScan(real.tree).tests_of(gg_lib)[0] if t.path.startswith("bind::"))
    n_seats = sum(1 for t in TestScan(real.tree).tests_of(gg_lib)[0] if t.path.startswith(("vision_seat::", "side_seat::")))
    n_start = sum(1 for t in TestScan(real.tree).tests_of(gg_lib)[0] if t.path.startswith("start::"))  # the recipe's last command (the start path's agent half)
    expect(n_bind > 0 and n_start > 0, "orphan scan: the tree holds no bind:: or start:: test for the FAIL-first clause to orphan")
    orphans, errors, checked = orphan_tests(real.tree, real.recipes)
    expect(not errors, f"orphan scan on the real tree: {errors[:3]}")
    expect(not any(" bind::tests::" in o for o in orphans), f"orphan scan: a bind test is an orphan on the real tree: {[o for o in orphans if 'bind::' in o]}")
    base = set(orphans)
    with open(os.path.join(ROOT, "justfile"), encoding="utf-8") as fh:
        text = fh.read()
    with tempfile.TemporaryDirectory(prefix="recipes-orphan-ff-") as tmp:
        p = os.path.join(tmp, "justfile")
        # (a) the recipe removed: every bind test red, each naming the cfg the lib run lacks (that run carries the hf feature, which only
        # the fetch tests need; the deepseek41 cfg it still lacks is the point), and every start test, each naming the gpu cfg that run lacks
        m = re.search(r"^(?:\[[^\n]*\]\n)*gate-ds41-bind:\n(?:    .*\n)+", text, re.M)  # its attributes go with it
        if m is None:
            expect(False, "orphan FAIL-first: no gate-ds41-bind recipe to remove")
        else:
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(text[: m.start()] + text[m.end() :])
            got = set(orphan_tests(real.tree, load_justfile(p))[0]) - base
            bind = [o for o in got if " bind::tests::" in o]
            start = [o for o in got if " start::" in o]
            expect(len(bind) == n_bind and len(start) == n_start and len(got) == n_bind + n_seats + n_start
                   and all('cfg(feature = "deepseek41") at crates/gpu-gates/src/lib.rs:' in o and "gate-gpu-gates-lib (features dspark-hf,hf)" in o for o in bind)
                   and all('cfg(feature = "gpu") at crates/gpu-gates/src/lib.rs:' in o and "gate-gpu-gates-lib (features dspark-hf,hf)" in o for o in start), f"orphan FAIL-first (a): {sorted(got)}")
        # (c) a filter that matches nothing of its module: the tests only that filter reached go red
        for pat in (r"--lib -- (bind::)(?= &&|')", r"--lib -- (start::)(?= &&|')", r"--ignored (hw_ds41_oracle) "):
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


def static_key_self_test(expect) -> None:
    """A job key moves with every shipped file a `tree` KEYSPEC reads, with the commit a `rev:REF` item resolves to, with its
    command and with the caller's environment, and with nothing else: an mtime, a working-tree edit a `rev:` job does not
    read, a `-` job. A malformed line, a KEYSPEC item it does not take (a dir: or file: scope among them) or a REF this
    checkout cannot resolve is a named error, and so is what the scheduler's reader (read_jobs) refuses."""
    with tempfile.TemporaryDirectory() as d:
        genv = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull}

        def put(rel: str, text: str) -> None:
            os.makedirs(os.path.dirname(os.path.join(d, rel)) or d, exist_ok=True)
            with open(os.path.join(d, rel), "w") as fh:
                fh.write(text)

        def git_in(*argv: str) -> None:
            subprocess.run(["git", *argv], cwd=d, env=genv, check=True, capture_output=True)

        put("tools/a.sh", "echo a\n")
        put("crates/serve/src/lib.rs", "fn f() {}\n")
        put("justfile", "a:\n")
        for argv in (("init", "-q", "-b", "main"), ("config", "user.email", "t@t"), ("config", "user.name", "t"), ("config", "commit.gpgsign", "false"),
                     ("add", "-A"), ("commit", "-q", "-m", "one")):
            git_in(*argv)
        jobs = os.path.join(d, "target", "jobs.tsv")
        put("target/jobs.tsv", "t\t-\ttree\tbash tools/a.sh\nr\t-\trev:HEAD\tbash tools/a.sh\n"
                               "b\t-\ttree,rev:HEAD\tbash tools/a.sh\nn\t-\t-\tbash tools/a.sh\n")
        env = {"BLOOMERY_TIER": "fixture"}

        def keys(environ=env) -> dict[str, str]:
            return {r[0]: r[1] for r in static_keys(jobs, d, environ)}

        k0 = keys()
        expect(k0["n"] == "-" and all(HEX64.match(k0[x]) for x in "trb"), f"static-key: the key shapes: {k0}")
        expect(len({k0["t"], k0["r"], k0["b"]}) == 3, "static-key: jobs with one command and different scopes share a key")
        os.utime(os.path.join(d, "tools/a.sh"), (1, 1))
        expect(keys() == k0, "static-key: an mtime moved a key")
        # git tracks the executable bit: a file's content unchanged and its mode flipped is another input of a job that runs it
        os.chmod(os.path.join(d, "tools/a.sh"), 0o755)
        kx = keys()
        expect(kx["t"] != k0["t"] and kx["b"] != k0["b"] and kx["r"] == k0["r"], "static-key: a mode-only edit (chmod +x) must move the tree keys only")
        os.chmod(os.path.join(d, "tools/a.sh"), 0o644)
        expect(keys() == k0, "static-key: restoring a file's mode must restore the keys")
        put("crates/serve/src/lib.rs", "fn f() { }\n")
        k1 = keys()
        expect(k1["t"] != k0["t"] and k1["b"] != k0["b"] and k1["r"] == k0["r"], "static-key: a crates/serve edit must move the tree keys only")
        git_in("commit", "-q", "--allow-empty", "-m", "two")
        k2 = keys()
        expect(k2["r"] != k1["r"] and k2["b"] != k1["b"] and k2["t"] == k1["t"], "static-key: a new commit must move the rev: keys only")
        expect(keys({"BLOOMERY_TIER": "real"})["t"] != k2["t"], "static-key: a MAC_ENV value must move the key")
        for spelled in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS", "CARGO_TARGET_AARCH64_APPLE_DARWIN_RUSTFLAGS",
                        "CARGO_BUILD_RUSTFLAGS", "RUSTDOCFLAGS"):
            expect(spelled in STATIC_RUSTFLAGS_ENV, f"static-key: {spelled} is not among the variables the key carries")
        for flag in STATIC_RUSTFLAGS_ENV:
            expect(keys({**env, flag: "-A warnings"})["t"] != k2["t"], f"static-key: a caller's {flag} must move the key (cargo reads it into every build a job runs)")
        # PYTHONOPTIMIZE strips the assert statements of the python tools a keyed block runs: its green row proves nothing about a run with them
        expect(keys({**env, "PYTHONOPTIMIZE": "1"})["t"] != k2["t"], "static-key: PYTHONOPTIMIZE must move the key (assertions off in the tools a job runs)")
        # a set variable whose value spells the absent one's still strips them (PYTHONOPTIMIZE=<unset> is level 1)
        expect(keys({**env, "PYTHONOPTIMIZE": "<unset>"})["t"] != k2["t"], "static-key: a variable set to the absent one's spelling must move the key")
        put("target/jobs.tsv", "t\t-\ttree\tbash tools/a.sh --other\n")
        expect(keys()["t"] != k2["t"], "static-key: a job's command must move its key")
        expect(not any(f.startswith("target/") for f in static_keys(jobs, d, env)[0][3]), "static-key: target/ is not a shipped file")
        # the scheduler's own reader (read_jobs) is the one that reads the file: what it refuses, static-key refuses
        for bad, why in (("x\t-\ttree\n", "three fields"), ("x\t-\tbogus\tc\n", "an unknown item"), ("x\t-\ttree:tools\tc\n", "a tree with an argument"),
                         ("x\t-\tdir:tools\tc\n", "a dir: scope"), ("x\t-\tfile:justfile\tc\n", "a file: scope"),
                         ("x\t-\trev:no-such-rev\tc\n", "a rev: this checkout cannot resolve"), ("x\t-\trev:\tc\n", "a rev: with no REF"),
                         ("x\t-\ttree\tc\nx\t-\ttree\tc\n", "a name twice"), ("x\tafter=y\ttree\tc\n", "an after= of no job"),
                         ("x\tbogus\ttree\tc\n", "an unknown attribute"), ("x\tcpus=0\ttree\tc\n", "cpus=0")):
            put("target/jobs.tsv", bad)
            try:
                static_keys(jobs, d, env)
                expect(False, f"static-key: {why} is not refused")
            except RecipeError:
                pass


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
        for f in ("ref-paths.sh", "dspark-path.sh", "gguf-ranges.py"):
            shutil.copyfile(os.path.join(src, f), os.path.join(tmp, "tools/ref", f))
        shutil.copyfile(os.path.join(ROOT, FIXTURE_DIRS), os.path.join(tmp, FIXTURE_DIRS))
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
        for budgets in ((10, 20), (20, 10)):
            write(first, [("bloomery.fixture.version", 4, 1)] +
                  [("bloomery.fixture.card_budget", 10, budget) for budget in budgets])
            rc, out, err = run("qwen4exp", "fixture_budget", BLOOMERY_TIER="fixture")
            expect(rc == 65 and out == "" and "duplicate metadata key 'bloomery.fixture.card_budget'" in err,
                   f"ref-paths: duplicate GGUF budgets {budgets}: {rc} {out!r} {err!r}")
        with open(first, "wb") as fh:
            fh.write(b"not a gguf at all")
        rc, out, err = run("qwen4exp", "fixture_budget", BLOOMERY_TIER="fixture")
        expect(rc == 65 and "is not a GGUF file" in err, f"ref-paths: a file that is no GGUF: {rc} {out!r} {err!r}")
        write(first, whole)
        rc, out, err = run("qwen4exp", 'echo "$FIXTURE_MODEL"', BLOOMERY_TIER="fixture", BLOOMERY_REF_MODEL="/x/own.gguf")
        expect(rc == 0 and out.strip() == first, f"ref-paths: a GGUF fixture's model is its first shard, whatever the caller's: {rc} {out!r} {err!r}")

        # A checkpoint-folder fixture (`fixture cut`, a directory with model.safetensors.index.json): the model is the folder, the
        # first shard is the fixture file, and its budget is a decimal string in the shard's safetensors `__metadata__`.
        with open(os.path.join(tmp, "tools/ref/models/minimax-m3.sh"), "w", encoding="utf-8") as fh:
            fh.write("MODEL_NAME=minimax-m3\nMODEL=${BLOOMERY_REF_MODEL:-/models/real/MiniMax-M3.1-Flash}\n")
        cut_dir = os.path.join(root, "minimaxm31")
        os.makedirs(cut_dir)
        with open(os.path.join(cut_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as fh:
            fh.write('{"metadata": {}, "weight_map": {}}')
        cut_first = os.path.join(cut_dir, "model-00001-of-00002.safetensors")

        def write_cut(path: str, meta: dict | str) -> None:
            header = (json.dumps({"__metadata__": meta}) if isinstance(meta, dict) else meta).encode()
            header += b" " * (-len(header) % 8)
            with open(path, "wb") as fh:
                fh.write(len(header).to_bytes(8, "little") + header)

        cut_whole = {"bloomery.fixture.version": "1", "bloomery.fixture.card_budget": "6190000000"}
        write_cut(cut_first, cut_whole)
        show_cut = 'echo "$MODEL|$FIXTURE_FILE|$FIXTURE_DIR|$FIXTURE_MODEL"'
        rc, out, err = run("minimax-m3", f"{show_cut}; fixture_budget", BLOOMERY_TIER="fixture")
        expect(rc == 0 and out.split() == [f"{cut_dir}|{cut_first}|{cut_dir}|{cut_dir}", "6190000000"],
               f"ref-paths: a folder fixture's model is the folder and its file the first shard: {rc} {out!r} {err!r}")
        rc, out, err = run("minimax-m3", show_cut)
        expect(rc == 0 and out.strip() == "/models/real/MiniMax-M3.1-Flash|||", f"ref-paths: a folder family's real tier: {rc} {out!r} {err!r}")
        rc, out, err = run("minimax-m3", show_cut, BLOOMERY_TIER="fixture", BLOOMERY_REF_MODEL="/x/own")
        expect(rc == 0 and out.strip() == f"/x/own|{cut_first}|{cut_dir}|{cut_dir}",
               f"ref-paths: a caller's own folder does not win, or the fixture is not named: {rc} {out!r} {err!r}")
        rc, out, err = run("minimax-m3", 'fixture_dump_tier t && echo "$SET_PREFIX"', BLOOMERY_TIER="fixture")
        expect(rc == 0 and out.strip() == "fx_", f"ref-paths: a folder fixture's dump is fx_: {rc} {out!r} {err!r}")
        rc, out, err = run("minimax-m3", "fixture_dump_tier t", BLOOMERY_TIER="fixture", BLOOMERY_REF_MODEL="/x/own")
        expect(rc == 64 and f"not the fixture {cut_dir}" in err, f"ref-paths: a folder fixture's dump of another model: {rc} {out!r} {err!r}")
        for why, meta in (("has no bloomery.fixture.version key", {"bloomery.fixture.card_budget": "5"}),
                          ("carries bloomery.fixture.subset", {**cut_whole, "bloomery.fixture.subset": "1"}),
                          ("records no positive bloomery.fixture.card_budget", {"bloomery.fixture.version": "1"}),
                          ("records no positive bloomery.fixture.card_budget", {**cut_whole, "bloomery.fixture.card_budget": "0"}),
                          ("records no positive bloomery.fixture.card_budget", {**cut_whole, "bloomery.fixture.card_budget": "-5"}),
                          ("records no positive bloomery.fixture.card_budget", {**cut_whole, "bloomery.fixture.card_budget": "12x"}),
                          ("is not a string, which a safetensors metadata value is", {**cut_whole, "bloomery.fixture.card_budget": 5})):
            write_cut(cut_first, meta)
            rc, out, err = run("minimax-m3", "fixture_budget", BLOOMERY_TIER="fixture")
            expect(rc == 65 and out == "" and why in err, f"ref-paths: a folder whose first shard {why} ({meta}): {rc} {out!r} {err!r}")
        for first_budget, second_budget in (("10", "20"), ("20", "10")):
            for budget_key in ('"bloomery.fixture.card_budget"', r'"bloomery.fixture.card_\u0062udget"'):
                write_cut(cut_first, '{"__metadata__":{"bloomery.fixture.version":"1",'
                          f'"bloomery.fixture.card_budget" : "{first_budget}", {budget_key} : "{second_budget}"' + '}}')
                rc, out, err = run("minimax-m3", "fixture_budget", BLOOMERY_TIER="fixture")
                expect(rc == 65 and out == "" and "duplicate key 'bloomery.fixture.card_budget'" in err,
                       f"ref-paths: duplicate safetensors budgets {first_budget}, {second_budget} ({budget_key}): {rc} {out!r} {err!r}")
        for header, key in (
            ('{"__metadata__":' + json.dumps(cut_whole) + ',"__metadata__":' + json.dumps(cut_whole) + '}', "__metadata__"),
            ('{"__metadata__":' + json.dumps(cut_whole) + ',"extra":[{"k":1,"k":2}]}', "k"),
        ):
            write_cut(cut_first, header)
            rc, out, err = run("minimax-m3", "fixture_budget", BLOOMERY_TIER="fixture")
            expect(rc == 65 and out == "" and f"duplicate key {key!r}" in err,
                   f"ref-paths: duplicate safetensors key {key}: {rc} {out!r} {err!r}")
        for blob in (b"short", (1 << 30).to_bytes(8, "little") + b"{}", (2).to_bytes(8, "little") + b"[]", (2).to_bytes(8, "little") + b"{{"):
            with open(cut_first, "wb") as fh:
                fh.write(blob)
            rc, out, err = run("minimax-m3", "fixture_budget", BLOOMERY_TIER="fixture")
            expect(rc == 65 and out == "" and "not a readable safetensors header" in err, f"ref-paths: a first shard that is no safetensors file ({blob[:12]!r}): {rc} {out!r} {err!r}")
        write_cut(cut_first, cut_whole)
        cut_second = os.path.join(cut_dir, "other-00001-of-00002.safetensors")
        write_cut(cut_second, cut_whole)
        rc, out, err = run("minimax-m3", show_cut, BLOOMERY_TIER="fixture")
        expect(rc == 66 and "holds 2 first shards" in err and "other-00001-of-00002.safetensors" in err, f"ref-paths: a folder with two first shards: {rc} {out!r} {err!r}")
        os.remove(cut_second)
        os.remove(cut_first)
        rc, out, err = run("minimax-m3", show_cut, BLOOMERY_TIER="fixture")
        expect(rc == 66 and "has no fixture file" in err and "minimaxm31/*-00001-of-*.safetensors" in err and "fixture cut" in err,
               f"ref-paths: a folder with no first shard: {rc} {out!r} {err!r}")
        write_cut(cut_first, cut_whole)
        stray = os.path.join(cut_dir, "stray-00001-of-00001.gguf")
        write(stray, whole)
        rc, out, err = run("minimax-m3", show_cut, BLOOMERY_TIER="fixture")
        expect(rc == 66 and "both a checkpoint folder's index and GGUF first shards" in err, f"ref-paths: a folder beside a GGUF first shard: {rc} {out!r} {err!r}")
        os.remove(stray)


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


def expand_self_test(expect) -> None:
    """expand: a top-level variable's value lands in the recipe line, so a literal spelling and the variable at its default
    expand to the same text; the variable's value in this environment shows, a parameter stays a parameter, and a change
    inside any other expression still shows."""
    head = "arch := env_var_or_default('BLOOMERY_ARCH', 'sm_86')\n"
    body = "r BIN:\n    ./tools/box.sh 'cargo oxide build --arch {} -- -p y --bin {{{{BIN}}}} {{{{ if BIN == \"a\" {{ \"{}\" }} else {{ \"c\" }} }}}}'\n"
    saved = os.environ.pop("BLOOMERY_ARCH", None)
    try:
        with tempfile.TemporaryDirectory(prefix="recipes-expand-") as tmp:

            def expanded(name: str, text: str) -> list[str]:
                path = os.path.join(tmp, name)
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(text)
                return expand_justfile(path)

            lit = expanded("lit", body.format("sm_86", "b"))
            var = expanded("var", head + body.format("{{arch}}", "b"))
            expect(lit == var, f"expand: the variable at its default and the literal expand differently: {lit} / {var}")
            expect(any("--arch sm_86 -- -p y --bin {{[[\"variable\", \"BIN\"]]}} " in ln for ln in var), f"expand: the parameter did not stay a parameter: {var}")
            expect(expanded("other", head + body.format("{{arch}}", "d")) != var, "expand: a change inside an if-expression did not show")
            os.environ["BLOOMERY_ARCH"] = "sm_121"
            other = expanded("var121", head + body.format("{{arch}}", "b"))
            expect(other != var and any("--arch sm_121 -- -p y" in ln for ln in other) and not any("sm_86" in ln for ln in other), f"expand: BLOOMERY_ARCH=sm_121 did not reach the line: {other}")
            expect(expanded("lit121", body.format("sm_86", "b")) == lit, "expand: a justfile with no variable read the environment")
    finally:
        os.environ.pop("BLOOMERY_ARCH", None)
        if saved is not None:
            os.environ["BLOOMERY_ARCH"] = saved


def oxide_rev_self_test(expect) -> None:
    """The cuda-oxide rev the key (oxide_rev) and tools/box.sh (the backend directory it exports) read from this tree's
    Cargo.toml is the [patch] section's cuda-device rev as TOML reads it when a tree has that section (a tree from before the
    fork was dropped), else the workspace declaration's."""
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


def key_reach_self_test(expect, side: Side) -> None:
    """The runtime key/reach pins and genuine Cargo-linkage explanation chains on the real tree."""
    # PIN(2026-10-22): Drop the plain-link hf mutation: v2 keys exclude uncalled runtime code.
    for crate, reason in KEY_REACH_PINS.items():
        gap = key_reach_gap(side, crate)
        expect(not gap, f"{crate} ({reason}): {len(gap)} recipes key it without reaching it: {' '.join(gap[:8])}")
    # why --recipe: a recipe that links the crate names the chain from its target, one that does not names none
    chain = why_chains(side, "gate-gpu-p1", "crates/hf/src/lib.rs")
    expect(chain == [], f"why: gate-gpu-p1 links hf: {chain}")
    chain = why_chains(side, "gate-gpu-glm5next-twocard", "crates/hf/src/lib.rs")
    expect(any(c.endswith("generate_glm5next: bloomery-gpu-gates -> bloomery-hf") for c in chain), f"why: gate-gpu-glm5next-twocard's chain to hf is {chain}")
    # gate-hf runs the crate's lib tests and its minimax_encode test target, each its own package's
    chain = why_chains(side, "gate-hf", "crates/hf/src/lib.rs")
    expect(chain == ["bloomery-hf libtest: its own package", "bloomery-hf test minimax_encode: its own package"],
           f"why: gate-hf's chain to hf is {chain}")
    for name in ("gate-decision",):
        expect(side.graph.inputs(name).match("crates/hf/src/lib.rs") is None, f"{name} keys unrelated hub code")
    expect(
        side.graph.inputs("gate-decision").match("crates/safetensors/src/lib.rs") is not None,
        "the feature-on decision gate must key its mapped head reader",
    )


def mac_landing_self_test(expect) -> None:
    """The mac-crate landing rule on a synthetic fixture table: a CUDA gate on a GGUF fixture or a `self` profile whose
    script defaults MODEL to a .gguf file is deferred; a checkpoint-folder fixture, a host gate and an unknown profile stay."""
    with tempfile.TemporaryDirectory(prefix="recipes-mac-") as tmp:
        os.makedirs(os.path.join(tmp, "tools/fixture-cuts"))
        os.makedirs(os.path.join(tmp, "tools/ref/models"))
        with open(os.path.join(tmp, FIXTURE_DIRS), "w", encoding="utf-8") as fh:
            fh.write("deepseek2\tself\nfilefam\tfiledir\nfoldfam\tfolddir\nselfprof\tself\n")
        with open(os.path.join(tmp, REF_PATHS), "w", encoding="utf-8") as fh:
            fh.write(': "${BLOOMERY_MODEL:=${BLOOMERY_REF_MODEL_PROFILE:-deepseek2}}"\n')
        with open(os.path.join(tmp, "tools/fixture-cuts/folddir.json"), "w", encoding="utf-8") as fh:
            fh.write("{}\n")
        with open(os.path.join(tmp, "tools/ref/models/selfprof.sh"), "w", encoding="utf-8") as fh:
            fh.write("MODEL=${BLOOMERY_REF_MODEL:-/models/Self/model.gguf}\n")
        with open(os.path.join(tmp, "tools/ref/models/deepseek2.sh"), "w", encoding="utf-8") as fh:
            fh.write("MODEL=${BLOOMERY_REF_MODEL:-/models/Self/dir}\n")

        def rec(name: str, line: str, n: int) -> Recipe:
            return Recipe(name=name, lines=[line], deps=[], text=line, line=n)

        rs = {
            "gate-file": rec("gate-file", "BLOOMERY_MODEL=filefam ./tools/box.sh 'bash tools/gpu-gate.sh x'", 1),
            "gate-fold": rec("gate-fold", "BLOOMERY_MODEL=foldfam ./tools/box.sh 'bash tools/gpu-gate.sh x'", 2),
            "gate-self": rec("gate-self", "BLOOMERY_MODEL=selfprof ./tools/box.sh 'bash tools/gpu-gate.sh x'", 3),
            "gate-host": rec("gate-host", "BLOOMERY_MODEL=filefam ./tools/box.sh 'bash tools/gate.sh x'", 4),
            "gate-wide": rec("gate-wide", "BLOOMERY_MODEL=nopefam ./tools/box.sh 'bash tools/gpu-gate.sh x'", 5),
            "gate-plain": rec("gate-plain", "./tools/box.sh 'bash tools/gpu-gate.sh x'", 6),
        }
        got = dict(mac_gguf_cuda(rs, tmp))
        expect(set(got) == {"gate-file", "gate-self"}, f"mac: the GGUF CUDA gates are gate-file and gate-self: {sorted(got)}")
        expect("gate-fold" not in got, f"mac: a checkpoint-folder fixture is not deferred: {sorted(got)}")
        expect("gate-host" not in got and "gate-wide" not in got, f"mac: a host gate and an unknown profile stay: {sorted(got)}")
        expect("gate-plain" not in got, f"mac: the default profile's self row with no .gguf default stays: {sorted(got)}")
        expect(mac_crate_file("crates/hf/src/vocab.rs") and mac_crate_file("crates/model-hf/src/lib.rs")
               and not mac_crate_file("crates/gpu/src/lib.rs") and not mac_crate_file("justfile"),
               "mac: the crate table is hf, model-hf and the other named crates, not crates/gpu")
        kind, why = gate_fixture_kind(rs, "gate-fold", fixture_kind_table(tmp)[0], fixture_kind_table(tmp)[1], tmp)
        expect(kind == "folder" and "folddir" in why, f"mac: foldfam is a folder fixture: {kind} {why}")
        bare = os.path.join(tmp, "empty")
        os.makedirs(os.path.join(bare, "tools/ref"))
        open(os.path.join(bare, FIXTURE_DIRS), "w", encoding="utf-8").close()
        open(os.path.join(bare, REF_PATHS), "w", encoding="utf-8").close()
        expect(fixture_kind_table(bare) is None and mac_gguf_cuda(rs, bare) == [],
               "mac: a fixture table with no row defers nothing")


def reach_self_test(expect, side: Side, row, stats: dict[str, int]) -> None:
    """The derivation (Reach) on a synthetic workspace, one file per rule it decides, and the table's check of it on
    the real tree: a draft row that leaves out a gate reaching its files through another lib, and the exceptions."""
    mac_landing_self_test(expect)
    for source, want in (
        ("#[derive(Clone, Debug, PartialEq)] struct S;", (False, False)),
        ('// #[ctor::ctor]\nconst S: &str = "#[ctor::ctor]";', (False, False)),
        ("#[ctor::ctor] fn startup() {}", (True, False)),
        ('#[cfg_attr(feature = "boot", ctor::ctor)] fn startup() {}', (True, False)),
        ('#[cfg_attr(unix, cfg_attr(feature = "boot", ctor::ctor))] fn startup() {}', (True, False)),
        ('#[cfg_attr(unix, derive(Debug, custom::Error))] enum E {}', (True, False)),
        ('#[unsafe(link_section = ".init_array")] static INIT: fn() = boot;', (False, True)),
    ):
        expect(_key_attributes(rust_lex(source)[1]) == want, f"key attributes misclassify {source!r}")
    with tempfile.TemporaryDirectory(prefix="recipes-reach-") as tmp:
        files = {
            "base/Cargo.toml": "",
            # each of a.rs … h.rs holds one fn, reached (or not) by one rule
            "base/src/lib.rs": (
                "pub mod a;\npub mod b;\npub mod c;\npub mod d;\npub mod e;\npub mod f;\npub mod g;\npub mod h;\npub mod rows;\nmod imp;\nmod inner;\n"
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
            # data a lib reads: two `include_str!` files, a file by path literal and a directory by path literal; the
            # fns that name each are called by mid except `idle`
            "base/src/rows.rs": (
                "pub const LIST: &str = include_str!(\"../data/list.txt\");\npub const IDLE: &str = include_str!(\"../data/idle.txt\");\n"
                "pub const TABLE: &str = \"../tools/table.tsv\";\npub const HELD: &str = \"../tools/lib-held\";\n"
                "pub const UNUSED: &str = \"../tools/idle.tsv\";\npub const UNUSED_DIR: &str = \"../tools/idle-held\";\n"
                + "".join(f"pub fn {f}() -> &'static str {{\n    {c}\n}}\n" for f, c in (("list", "LIST"), ("idle", "IDLE"), ("table", "TABLE"), ("held", "HELD")))
            ),
            "mid/Cargo.toml": "",
            "mid/src/lib.rs": (
                "mod idle;\n"
                "pub fn call() -> u32 {\n    base::b::run();\n    base::mk!();\n    base::rows::list();\n    base::rows::table();\n"
                "    base::rows::held();\n    base::g::K\n}\n"
            ),
            "mid/src/idle.rs": "pub fn idle() {}\n",
            "tools/shared-read.sh": (
                'cat tools/idle.tsv\nfind "$ROOT/tools/idle-held" -type f\nfind "$ROOT/base" -name Cargo.toml\n'
            ),
            "app/Cargo.toml": "",
            "app/src/lib.rs": "",
            "app/tests/it.rs": (
                "mod common;\n#[test]\nfn t() {\n    let h = common::H;\n    h.make();\n"
                "    let s = base::S;\n    s.used();\n    let u = base::U;\n    u.go();\n}\n"
                "const OWN_HELD: &str = \"../tools/own-held\";\n"
            ),
            "app/tests/common/mod.rs": "pub struct H;\nimpl H {\n    pub fn make(&self) {\n        mid::call();\n    }\n}\n",
        }
        for x in "abcdefh":
            files[f"base/src/{x}.rs"] = "pub fn run() {}\n"
        for rel in ("base/data/list.txt", "base/data/idle.txt", "tools/table.tsv", "tools/lib-held/b.ids",
                    "tools/own-held/a.ids", "tools/idle.tsv", "tools/idle-held/extra.txt"):
            files[rel] = "x\n"
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
        rs = {"gate-app": Recipe("gate-app", ["./tools/box.sh 'bash tools/gate.sh -p app --test it'"], [], ""),
              "gate-mid": Recipe("gate-mid", ["./tools/box.sh 'bash tools/gate.sh -p mid --lib'"], [], "")}
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
        # a data file is one more file a unit can name: an `include_str!` or a path literal a reached unit holds is
        # reached, and what only an unreached unit holds is not
        for f, reached in (("base/data/list.txt", True), ("tools/table.tsv", True), ("base/data/idle.txt", False)):
            expect(bool(dep_reach(st, sg, ["gate-app"], f)) == reached, f"reach: {f} {'not reached' if reached else 'reached'} through the unit that holds it")
        ki = sg.key_inputs("gate-app")
        for f in want:
            expect(ki.match(f) is not None, f"key reach omits reached dependency code {f}")
        for f in ("base/src/a.rs", "base/src/d.rs", "base/src/f.rs", "base/data/idle.txt"):
            expect(ki.match(f) is None, f"key reach retains unused linked input {f}")
        for f in ("app/tests/it.rs", "app/tests/common/mod.rs", "base/src/lib.rs",
                  "tools/lib-held/b.ids", "tools/own-held/a.ids"):
            expect(ki.match(f) is not None, f"key reach omits a root/module/directory input {f}")
        expect(ki.match("mid/src/idle.rs") is None, "key reach retains an unused reverse-dependent module")
        for command in ("cat tools/idle.tsv", "bash tools/shared-read.sh"):
            scripted = Recipe("gate-read", [*rs["gate-app"].lines, command], [], "")
            reads_graph = Graph(st, {"gate-read": scripted})
            expect(_linked_origin(reads_graph.inputs("gate-read").files["tools/idle.tsv"]),
                   "shared-read regression needs Cargo to have supplied the first explanation")
            read_inputs = reads_graph.key_inputs("gate-read")
            expect(read_inputs.match("tools/idle.tsv") is not None, f"{command}: a Cargo-first explicit/script read was erased")
            if command.startswith("bash"):
                expect(read_inputs.match("tools/idle-held/extra.txt") is not None,
                       "a Cargo-first literal prefix erased a script's independent find read")
                expect("base/Cargo.toml" in read_inputs.manifests.whole, "script directory read lost the manifest whole override")
        # the row check reads the same links: a row for a path-literal file naming the gates that reach it is clean, and
        # one that leaves gate-app out is refused with the chain
        for f in ("base/data/list.txt", "tools/table.tsv"):
            got = gate_paths_problems(st, rs, [row(f, ["gate-app", "gate-mid"])])
            expect(not got, f"gate-paths: a row naming the gates that reach {f} by a literal is refused: {got[:2]}")
            got = gate_paths_problems(st, rs, [row(f, ["gate-mid"])])
            expect(any(p.endswith(f"gate-app reaches {f} — add it to the row, or write `derive`, or name it in the row's exceptions with the reason; the chain: {reach_chain(st, sg, 'gate-app', dep_reach(st, sg, ['gate-app'], f)['gate-app'])}") for p in got),
                   f"gate-paths: a row leaving out the gate that reaches {f} by a literal is not refused with its chain: {got[:2]}")
            # a `derive` row stands for the gates the tree derives: clean with none written, the gate the hand row lacked in it, and a
            # hand name beside it once; expand_rows writes the same names, and the row is unreadable for its gates before that
            drv: dict[int, list[str]] = {}
            got = gate_paths_problems(st, rs, [row(f, [], derive=True)], derived=drv)
            expect(not got and "gate-app" in drv.get(1, []), f"gate-paths: a `derive` row for {f} is refused, or derives no gate-app: {got[:2]} {drv}")
            ex = expand_rows(st, rs, [row(f, ["gate-mid"], derive=True)])
            expect(len(ex) == 1 and not ex[0].derive and ex[0].recipes == list(dict.fromkeys([*drv.get(1, []), "gate-mid"])) and ex[0].gates() == ex[0].recipes,
                   f"gate-paths: expand_rows writes the derived gates then the hand names, once each: {[e.recipes for e in ex]} for {drv}")
            expect(_raises(lambda: row(f, [], derive=True).gates(), "is read before expand_rows names its gates"),
                   "gate-paths: a `derive` row's gates() read before expand_rows is not refused by name")
        # a file a gate links through a dependency that no reached unit names: the tree derives no gate for it, so `derive` is refused
        # and a hand row naming the gate stands
        got = gate_paths_problems(st, rs, [row("base/data/idle.txt", [], derive=True)])
        expect(len(got) == 1 and "`derive` stands for no gate" in got[0], f"gate-paths: a `derive` row with an empty derivation not refused by name: {got}")
        expect(_raises(lambda: expand_rows(st, rs, [row("base/data/idle.txt", [], derive=True)]), "`derive` stands for no gate"),
               "gate-paths: expand_rows on a `derive` row with an empty derivation not refused by name")
        got = gate_paths_problems(st, rs, [row("base/data/idle.txt", ["gate-app"])])
        expect(not got, f"gate-paths: a hand row for a file the tree derives no gate for is refused: {got[:2]}")
        # the table read for narrow(): a miss derives, notes it, writes nothing to a stderr that is no terminal (gate-batch --list
        # reads `just affected … 2>&1`, whose first lines must stay the recipe line and `affected:`)
        import contextlib
        import io

        with tempfile.TemporaryDirectory(prefix="recipes-gp-table-") as tbl:
            os.makedirs(os.path.join(tbl, "tools"))
            with open(os.path.join(tbl, GATE_PATHS), "w", encoding="utf-8") as fh:
                fh.write("# t\nbase/data/list.txt\tderive gate-mid\twhy\n")
            notes: list[str] = []
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                table = expanded_table(tbl, Side(st, rs, sg), os.path.join(tbl, "cache"), notes)
            expect(err.getvalue() == "", f"gate-paths: a derivation on a cache miss writes to a stderr that is no terminal: {err.getvalue()!r}")
            expect(len(table) == 1 and table[0].line == 2 and not table[0].derive and "gate-app" in table[0].gates() and table[0].gates()[-1] == "gate-mid",
                   f"gate-paths: expanded_table does not write the derived gates then the hand name: {[(t.line, t.recipes) for t in table]}")
            expect(any("derived from the tree" in x for x in notes), f"gate-paths: a derivation leaves no note: {notes}")
        # one label for a target's own read whether it names a file or a directory (origin_label): a test's own
        # directory literal is the test's own, a lib's is a dependency edge, as a file of either is
        own_dir, lib_dir = sg.inputs("gate-app").match("tools/own-held/a.ids"), sg.inputs("gate-app").match("tools/lib-held/b.ids")
        expect(own_dir == "test it" and _own_origin(own_dir), f"origins: a test's own directory literal is labelled {own_dir!r}, not its own read")
        expect(lib_dir is not None and not _own_origin(lib_dir), f"origins: a lib's directory literal is labelled {lib_dir!r}, not a dependency edge")
        expect(sg.inputs("gate-app").match("tools/table.tsv") == "test it <- lib base (path literal) (app -> base)",
               f"origins: a lib's file literal is labelled {sg.inputs('gate-app').match('tools/table.tsv')!r}")
        # a row for files only an own target reads maps nothing (narrow() picks that target without a row) and is taken out;
        # one for a lib's directory, a dependency read, stands
        got = gate_paths_problems(st, rs, [row("tools/own-held/a.ids", ["gate-app"])])
        expect(len(got) == 1 and "no gate reads a file the glob matches through a dependency" in got[0] and got[0].endswith("take the row out"),
               f"gate-paths: a row for a test's own directory literal not refused: {got}")
        got = gate_paths_problems(st, rs, [row("tools/lib-held/b.ids", ["gate-app", "gate-mid"])])
        expect(not got, f"gate-paths: a row for a lib's directory literal refused: {got[:2]}")
        # Opaque expansions are bounded by their runtime-active package and forward dependencies, not
        # reverse/sibling packages. A transitive optional edge can be linked by a different Cargo root.
        extra = {
            "leaf/Cargo.toml": "", "leaf/src/lib.rs": "mod idle;\npub fn leaf() {}\n",
            "leaf/src/idle.rs": "pub fn idle() {}\n",
            "opt/Cargo.toml": "", "opt/src/lib.rs": "mod idle;\npub fn optional() {}\n",
            "opt/src/idle.rs": "pub fn idle() {}\n",
            "side/Cargo.toml": "", "side/src/lib.rs": "#[derive(custom::Hidden)] pub struct Side;\n",
        }
        for rel, text in extra.items():
            os.makedirs(os.path.join(tmp, os.path.dirname(rel)), exist_ok=True)
            with open(os.path.join(tmp, rel), "w", encoding="utf-8") as fh:
                fh.write(text)
        leaf = pkg("leaf", [("lib", "leaf", "src/lib.rs")], ["opt"])
        leaf["dependencies"][0]["optional"] = True
        opaque_meta = {
            "workspace_root": tmp, "workspace_members": ["base", "mid", "app", "leaf", "opt", "side"],
            "packages": [
                pkg("base", [("lib", "base", "src/lib.rs")], ["leaf"]),
                pkg("mid", [("lib", "mid", "src/lib.rs")], ["base"]),
                pkg("app", [("lib", "app", "src/lib.rs"), ("test", "it", "tests/it.rs")], ["mid", "base", "opt", "side"]),
                leaf, pkg("opt", [("lib", "opt", "src/lib.rs")], []), pkg("side", [("lib", "side", "src/lib.rs")], []),
            ],
        }
        for source, widened in (
            ("#[derive(Debug)] pub struct Hidden;\n", False),
            ("#[derive(Debug, thiserror::Error)] pub enum Hidden {}\n", True),
            ("#[ctor::ctor] fn startup() { crate::d::run(); }\n", True),
            ('#[cfg_attr(unix, cfg_attr(feature = "boot", ctor::ctor))] fn startup() { crate::d::run(); }\n', True),
        ):
            with open(os.path.join(tmp, "base/src/a.rs"), "w", encoding="utf-8") as fh:
                fh.write(source)
            opaque_graph = Graph(Tree(tmp, opaque_meta), rs)
            inputs = opaque_graph.key_inputs("gate-app")
            for rel in ("base/src/d.rs", "base/data/idle.txt", "leaf/src/idle.rs", "opt/src/idle.rs"):
                expect((inputs.match(rel) is not None) == widened, f"opaque owner closure miskeys {rel}: {source!r}")
            for rel in ("mid/src/idle.rs", "side/src/lib.rs"):
                expect(inputs.match(rel) is None, f"opaque owner restored an unrelated reverse/sibling input {rel}: {source!r}")
        with open(os.path.join(tmp, "base/src/a.rs"), "w", encoding="utf-8") as fh:
            fh.write("extern crate side as _;\n")
        # An extern-crate link must be declared by its importer, not only by the final application's sibling edge.
        opaque_meta["packages"][0] = pkg("base", [("lib", "base", "src/lib.rs")], ["leaf", "side"])
        inputs = Graph(Tree(tmp, opaque_meta), rs).key_inputs("gate-app")
        expect(inputs.match("side/src/lib.rs") is not None, "explicit side-effect extern-crate link lost its opaque owner")
        # A script reads a Rust file of a package no target of the recipe links, and the file holds an impl its item reach cannot place
        # (its type is written by a macro, so the rule that reaches such an impl whenever its file is linked fires on the read). That
        # package compiles nothing there, so it owns no compilation read; the file stays an input as the script's read, and the key builds.
        data = {
            "tools/read-data.sh": "cat crates/data/src/table.rs\n", "crates/data/Cargo.toml": "", "crates/data/src/lib.rs": "mod table;\n",
            "crates/data/src/table.rs": (
                "macro_rules! make {\n    () => {\n        pub enum Code {\n            A,\n        }\n    };\n}\nmake!();\n"
                "impl std::fmt::Display for Code {\n    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {\n        f.write_str(\"a\")\n    }\n}\n"
            ),
        }
        for rel, text in data.items():
            os.makedirs(os.path.join(tmp, os.path.dirname(rel)), exist_ok=True)
            with open(os.path.join(tmp, rel), "w", encoding="utf-8") as fh:
                fh.write(text)
        unlinked = pkg("data", [("lib", "data", "src/lib.rs")], [])
        unlinked["manifest_path"] = os.path.join(tmp, "crates/data/Cargo.toml")
        unlinked["targets"][0]["src_path"] = os.path.join(tmp, "crates/data/src/lib.rs")
        data_meta = dict(opaque_meta, workspace_members=[*opaque_meta["workspace_members"], "data"], packages=[*opaque_meta["packages"], unlinked])
        reads_data = {"gate-data": Recipe("gate-data", [*rs["gate-app"].lines, "bash tools/read-data.sh"], [], "")}
        inputs = Graph(Tree(tmp, data_meta), reads_data).key_inputs("gate-data")
        expect(inputs.match("crates/data/src/table.rs") is not None, "a script's read of a Rust file of a package the recipe does not link left the key")
        expect(inputs.match("crates/data/src/lib.rs") is None and "crates/data/Cargo.toml" not in inputs.files, "a script's read of one file of an unlinked package keyed the package")

    # the table's check on the real tree: the held rows, and the derivation's own lines
    tree, recipes, graph = side.tree, side.recipes, side.graph
    expect(stats.get("rows", 0) > 0, f"gate-paths: the summary counts no row held to the derivation: {stats}")
    # a hand-written row that leaves out a gate reaching its files through another lib: gate-gpu-ds41-step runs
    # moe.rs's experts_step_union_into through bloomery_gpu's host tier, and its bin names no model module
    draft = ("gate-ds2-debug gate-ds2-release gate-model-host gate-union "
             "gate-ds41-host gate-gpu-hybrid gate-gpu-ds41-tier gate-gpu-nvtier gate-gpu-qwen4exp-e2e gate-gpu-glm5next-e2e "
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
    # a planted row for a test's own data directory (gate-models reads head-holdout/ as its own target): nothing for it to map
    got = gate_paths_problems(tree, recipes, [row("crates/models/tests/data/head-holdout/*.ids", ["gate-models"])])
    expect(len(got) == 1 and "no gate reads a file the glob matches through a dependency" in got[0], f"gate-paths: a row for a test's own data not refused: {got}")
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
    """--narrow: the scan verdicts and their refusals, and the four rules of narrow() on the real tree with rows of
    the test's own."""
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
        # ptx-compare: the table of two full scans and its exit code, by what moved (the commands above the function)
        def compared(other: dict[str, str], **kw) -> tuple[int, str]:
            got = tried(lambda: ptx_compare(base, read_scan(scan(tmp, "pc.log", other, **kw)), "B", "H"))
            return (got[0], "\n".join(got[1])) if isinstance(got, tuple) else (-1, got)

        rc_pc, txt = compared({"alpha": h1, "beta": h2})
        expect(rc_pc == 0 and "  added         none" in txt and "  removed       none" in txt and "  changed       none" in txt
               and txt.endswith("ptx-compare: identical (rc 0)") and "  base entries  2" in txt and "  head entries  2" in txt, f"ptx-compare: identical pair: {rc_pc} {txt}")
        rc_pc, txt = compared({"alpha": h1, "beta": h2, "gamma": h3})
        expect(rc_pc == 0 and "  added         1: gamma" in txt and "  head entries  3" in txt and txt.endswith("ptx-compare: added-only (rc 0)"),
               f"ptx-compare: an added entry is rc 0: {rc_pc} {txt}")
        rc_pc, txt = compared({"alpha": h1, "beta": h3})
        expect(rc_pc == 3 and "  changed       1: beta (ptx-scan-md5:" in txt and txt.endswith("ptx-compare: changed (rc 3)"), f"ptx-compare: a moved digest is rc 3: {rc_pc} {txt}")
        rc_pc, txt = compared({"alpha": h1, "beta": h2}, regs="13")
        expect(rc_pc == 3 and "alpha (regs:12->13)" in txt and "beta (regs:12->13)" in txt and "ptx-scan-md5" not in txt, f"ptx-compare: a moved column is rc 3: {rc_pc} {txt}")
        rc_pc, txt = compared({"alpha": h1})
        expect(rc_pc == 4 and "  removed       1: beta" in txt and txt.endswith("ptx-compare: removed (rc 4)"), f"ptx-compare: a removed entry is rc 4: {rc_pc} {txt}")
        rc_pc, txt = compared({"alpha": h3, "gamma": h1})
        expect(rc_pc == 5 and "  added         1: gamma" in txt and "  removed       1: beta" in txt and "alpha (ptx-scan-md5:" in txt and txt.endswith("ptx-compare: changed and removed (rc 5)"),
               f"ptx-compare: a changed and a removed entry are rc 5: {rc_pc} {txt}")
        rc_pc, txt = compared({"alpha": h1, "beta": h2}, bundles=("bloomery-gpu", "bloomery-x"))
        expect(rc_pc == 3 and "  bundles       moved: bloomery-gpu -> bloomery-gpu bloomery-x" in txt, f"ptx-compare: a moved bundle set is rc 3: {rc_pc} {txt}")
        rc_pc, txt = compared({"alpha": h1, "beta": h2}, nojit=True)
        expect(rc_pc == -1 and "holds no JIT columns" in txt and "--no-jit scan (head)" in txt, f"ptx-compare: a no-jit head is refused by name: {rc_pc} {txt}")
        v = tried(lambda: ptx_compare(nj, base))
        expect(isinstance(v, str) and "--no-jit scan (base)" in v, f"ptx-compare: a no-jit base is refused by name: {v}")
        expect(refused(lambda: ptx_compare(base, read_scan(scan(tmp, "pcb.log", {"alpha": h1, "beta": h2}, bin="gy"))), "a pair is two scans of one binary"),
               "ptx-compare: a pair of two binaries not refused")
        other = read_scan(scan(tmp, "pcc.log", {"alpha": h1, "beta": h2}))
        other.fields["jit-cc"] = "9.0"
        expect(refused(lambda: ptx_compare(base, other), "jit-cc="), "ptx-compare: a pair of two capabilities not refused")
        import contextlib
        import io

        cli = io.StringIO()
        with contextlib.redirect_stdout(cli):
            got = main(["ptx-compare", base.path, scan(tmp, "pcd.log", {"alpha": h1, "beta": h2, "gamma": h3}), "--base-label", "83212936", "--head-label", "HEAD"])
        expect(got == 0 and "base=83212936 head=HEAD" in cli.getvalue(), f"ptx-compare: the command line: {got} {cli.getvalue()}")
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
        n = narrow(["docs/HARDWARE.md"], side, side, [(base, same)], rows)
        expect(not n.full and not n.picks and "not a host path" in n.files[0],
               f"narrow: docs/HARDWARE.md (read by no gate), an equal-scan host change, drops the spill ratchet and picks nothing: "
               f"full={n.full} picks={sorted(n.picks)}")
        n = narrow(["crates/vision/src/lib.rs"], side, side, [(base, same)], rows)
        expect(not n.full and {"gate-vision", "gate-gpu-vision"} <= set(n.picks) and "lib bloomery-gpu-vision not scanned" in n.files[0],
               f"narrow: a kernel carrier no pair covers does not keep its gates: {n.files} {sorted(n.picks)}")
        n = narrow(["crates/gpu-gates/src/bin/gate_p1.rs"], side, side, [(base, same)], rows)
        expect(not n.full and set(n.picks) == {"gate-gpu-p1"}, f"narrow: a bin's own file, an equal-scan change the ratchet does not read: {n.full} {sorted(n.picks)}")
        # a bin carrier is covered by the scan of that bin alone. Every bin of gpu-gates carries its own kernels in a
        # bundle named for the package, so a scan of another gpu-gates bin (same bundle name) or of that bin without
        # the package's bundle reads nothing of them. A change under crates/gpu reaches the closure of both bin
        # carriers (gate_swap, gate_kquant) and of the five device libs.
        # PIN(2026-10-21): bloomery-gpu-minimaxm3 joins the device libs: it carries kernels of its own, which a change
        # under crates/gpu reaches through the bins that link it (the narrow message named it "not scanned").
        libs = ("bloomery-gpu", "bloomery-gpu-deepseek41", "bloomery-gpu-gemma4a", "bloomery-gpu-minimaxm3", "bloomery-gpu-vision")
        own = ("bloomery-gpu", "bloomery-gpu-gates")
        gpu_rows = [row("crates/gpu/src/**", ["gate-gpu-lib"])]

        def pair(name: str, bin: str, bundles) -> tuple[ScanLog, ScanLog]:
            return (read_scan(scan(tmp, f"{name}-b.log", {"alpha": h1}, bundles=bundles, bin=bin, nojit=True)),
                    read_scan(scan(tmp, f"{name}-h.log", {"alpha": h1}, bundles=bundles, bin=bin, nojit=True)))

        lib_pair = pair("libs", "gx", libs)
        swap, kquant = pair("swap", "gate_swap", own), pair("kquant", "gate_kquant", own)
        # PIN(2026-10-22): gate_gpu_device joins the bin carriers: its conformance kernels are its own, so a change
        # under crates/gpu reaches its closure too, and "both bins scanned" scans it as well (the message named it
        # "not scanned").
        device = pair("device", "gate_gpu_device", own)

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
        got, why = picked(lib_pair, swap, kquant, device)
        expect(not {"gate-gpu-swap", "gate-gpu-kquant"} & got and "not scanned" not in why, f"narrow: both bins scanned still keep a bin carrier's gates: {sorted(got)} {why}")
        got, why = picked(lib_pair, pair("swap-nobundle", "gate_swap", ("bloomery-gpu",)))
        expect("gate-gpu-swap" in got and "bin gate_swap not scanned" in why,
               f"narrow: a scan of gate_swap that names no package bundle covers its kernels: {sorted(got)} {why}")
        # a lib carrier is covered by any scan that names its bundle, and by no other
        got, why = picked(pair("nolibs", "gx", ("bloomery-gpu",)), swap, kquant)
        expect({"gate-gpu-vision", "gate-ds41-bind", "gate-gpu-gemma4a"} <= got and "lib bloomery-gpu-vision not scanned" in why and "lib bloomery-gpu-deepseek41 not scanned" in why and "lib bloomery-gpu-gemma4a not scanned" in why,
               f"narrow: the device libs a scan names no bundle of do not keep their gates: {sorted(got)[:6]} {why}")
        # a data file a gate reads by `include_str!` or by path literal, alone in a diff and under scan pairs that cover every
        # kernel carrier, narrows with the real table: the shipped head list and the fixture variants' table through the
        # row naming the gates that run their readers (the derivation's list, less the own targets), a test's own holdout
        # directory as that test's own target with no row
        gates_all = [n for n in sorted(side.recipes, key=lambda n: side.recipes[n].line) if n.startswith(GATE_PREFIX)]
        every = tuple(sorted({c.bundle for c in kernel_carriers(side, gates_all)}))
        covering = [pair(f"cover-{b}", b, every) for b in ("gx", *sorted(c.bin for c in kernel_carriers(side, gates_all) if c.bin))]
        shipped = shipped_files(side.tree.root)
        probe = (("the shipped head list", "crates/model/data/mtp-head-rows-qwen38-*.txt", True),
                 ("a holdout file", "crates/models/tests/data/head-holdout/*.ids", False),
                 ("the fixture variants' table", "tools/fixture-variants.tsv", True))
        # the rows these files match, expanded over the tree: the table narrow() reads, less the rows no probe reaches
        probed = [f for _, glob, _ in probe for f in shipped if glob_regex(glob).match(f)]
        table = expand_rows(side.tree, side.recipes, [r for r in load_gate_paths(side.tree.root) if any(r.pattern.match(f) for f in probed)], side.graph)
        for what, glob, via_row in probe:
            found = [f for f in shipped if glob_regex(glob).match(f)]
            expect(bool(found), f"narrow: {what} ({glob}) matches no file of the tree")
            if not found:
                continue
            n = narrow([found[0]], side, side, covering, table)
            expect(not n.full and ("rows " in n.files[0][len(found[0]):]) == via_row, f"narrow: {what} alone keeps the full list, or is mapped by a row or not as expected ({via_row}): {n.full} {n.files}")
            if via_row:
                reach = set(dep_reach(side.tree, side.graph, gates_all, found[0]))
                expect(bool(reach) and reach <= set(n.picks), f"narrow: {what} misses a gate that runs it through a dependency: {sorted(reach - set(n.picks))}")
            else:
                expect("gate-models" in n.picks, f"narrow: {what} does not pick the test that reads its directory as its own target: {sorted(n.picks)}")
        # Clef's suite is read by one bin by a file literal and by the decision test by a directory literal: both are
        # own targets of it, so a diff of it alone picks both and no row stands for either
        n = narrow(["tools/ref/clef/suite.jsonl"], side, side, covering, table)
        expect(not n.full and {"gate-gpu-clef-serve", "gate-decision"} <= set(n.picks), f"narrow: Clef's suite alone drops a reader or keeps the full list: {n.full} {sorted(n.picks)}")
        # a Rust file of a lib's module tree no row maps is read by the derivation: its picks are the gates whose reach holds a
        # unit of it and the gates that read it as their own target's, not every gate that links its crate
        from types import SimpleNamespace

        f = "crates/sampler/src/lib.rs"
        read_by = {m: side.graph.inputs(m).match(f) for m in gates_all}
        linked = {m for m, w in read_by.items() if w and not _own_origin(w)}
        own_gates = {m for m, w in read_by.items() if w and _own_origin(w)} - set(NARROW_ALWAYS)
        reached = {m for m in dep_reach(side.tree, side.graph, gates_all, f) if m not in NARROW_ALWAYS}
        n = narrow([f], side, side, covering, rows)
        expect(not n.full and bool(reached) and reached < linked, f"narrow: a row-less lib file keeps the full list, or the derivation is as wide as the crate: {n.full} {len(reached)} of {len(linked)}")
        expect(set(n.picks) - set(NARROW_ALWAYS) == reached | own_gates, f"narrow: a row-less lib file does not pick its derivation and its own targets: {sorted(set(n.picks) ^ (reached | own_gates))[:6]}")
        expect(len(n.derived) == 1 and n.derived[0].file == f and n.derived[0].linked == len(linked) and set(n.derived[0].kept) == reached
               and n.derived[0].chain.split(": ", 1)[0] in reached and "derived: " in n.files[0], f"narrow: a derived file is not recorded with its gates and one chain: {n.derived} {n.files}")
        # a `*` row keeps the full list for the same file, and a pair that moved keeps it with the file derived
        n = narrow([f], side, side, covering, [row("crates/sampler/src/**", ["*"], 7)])
        expect(any(f"{GATE_PATHS}:7" in r for r in n.full) and not n.derived, f"narrow: a `*` row beside a lib file does not keep the full list: {n.full} {n.derived}")
        n = narrow([f], side, side, [(base, md5), *covering], rows)
        expect(any("ptx-scan gx: moved beta" in r for r in n.full), f"narrow: a moved pair beside a derived file does not keep the full list: {n.full}")
        # a file the derivation cannot read keeps the full list, naming the file and the predicate that failed it: a build
        # script, a module list no gate's reach holds a unit of, a file that is not Rust source
        for what, f, why in (("a build script", "crates/gpu/build.rs", "it is a build script"),
                             ("a module list", "crates/chain/src/arch/mod.rs", "no gate's reach holds a unit of it"),
                             ("the lock file", "Cargo.lock", "it is not Rust source")):
            n = narrow([f], side, side, covering, rows)
            expect(len(n.full) == 1 and n.full[0].startswith(f"{f}: ") and "no row of" in n.full[0] and why in n.full[0] and not n.derived,
                   f"narrow: {what} ({f}) is not kept whole with its reason ({why}): {n.full} {n.derived}")
        gone = SimpleNamespace(tree=SimpleNamespace(exists=lambda rel: False))
        expect("no longer holds" in (derivation_gap(gone, "crates/sampler/src/lib.rs") or ""), "narrow: a deleted file does not keep the full list")
        expect(derivation_gap(side, "crates/sampler/src/lib.rs") is None and derivation_empty(["gate-a"]) is None and derivation_empty([]) is not None,
               "narrow: the derivation's two predicates read a lib file as derivable and an empty reach as not")
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


def dispatch_self_test(expect, side: Side) -> None:
    """The registry's dispatch edges (Dispatch): a synthetic workspace with a registry of two families, one clause per rule it
    decides, and the same rules on the real tree (a gate that cannot open a family is not selected for that family's files
    by the registry's pointer; every gate that opens it is; a file a family shares is selected as before)."""
    dispatch_synthetic_self_test(expect)
    dispatch_real_self_test(expect, side)


def dispatch_synthetic_self_test(expect) -> None:
    reg_path, static = ARCH_REGISTRY
    rows_src = (
        "use super::{fam_alpha, fam_beta};\n"
        "pub struct ArchRow {\n    pub names: &'static [&'static str],\n    pub read: Option<fn() -> u32>,\n}\n"
        "pub const ALIAS: &str = \"beta-alias\";\n"
        f"pub static {static}: &[ArchRow] = &[\n"
        "    ArchRow {\n        names: &[\"deepseek2\"],\n        read: None,\n    },\n"
        "    ArchRow {\n        names: &[\"alpha\"],\n        read: Some(fam_alpha::read),\n    },\n"
        "    ArchRow {\n        names: &[\"beta\", ALIAS],\n        read: Some(fam_beta::read),\n    },\n];\n"
        f"pub fn row_of(arch: &str) -> Option<&'static ArchRow> {{\n    {static}.iter().find(|r| r.names.contains(&arch))\n}}\n"
        f"pub fn count() -> usize {{\n    {static}.len()\n}}\n"
        f"#[cfg(test)]\nmod tests {{\n    #[test]\n    fn rows() {{\n        assert!(!super::{static}.is_empty());\n    }}\n}}\n"
    )
    files = {
        "crates/model/Cargo.toml": "",
        "crates/model/src/lib.rs": "pub mod arch;\npub mod ops;\n",
        "crates/model/src/ops.rs": "pub fn run() -> u32 {\n    5\n}\n",
        "crates/model/src/arch/mod.rs": (
            "pub mod fam_alpha;\npub mod fam_beta;\npub mod registry;\npub mod shared;\npub mod shared2;\npub use util;\n"
            "pub fn spec(arch: &str) -> u32 {\n    match registry::row_of(arch).and_then(|r| r.read) {\n        Some(f) => f(),\n        None => shared::helper(),\n    }\n}\n"
            "pub fn variant() -> u32 {\n    fam_alpha::hp::HP\n}\n"
        ),
        reg_path: rows_src,
        "crates/model/src/arch/shared.rs": "pub fn helper() -> u32 {\n    1\n}\n",
        "crates/model/src/arch/shared2.rs": "pub fn only_by_a_family() -> u32 {\n    2\n}\n",
        "crates/model/src/arch/fam_alpha/mod.rs": "pub mod hp;\npub fn read() -> u32 {\n    hp::HP + crate::arch::shared2::only_by_a_family() + crate::ops::run()\n}\n",
        "crates/model/src/arch/fam_alpha/hp.rs": "pub const HP: u32 = 3;\n",
        "crates/model/src/arch/fam_beta/mod.rs": "pub mod hp;\npub fn read() -> u32 {\n    hp::HP\n}\n",
        "crates/model/src/arch/fam_beta/hp.rs": "pub const HP: u32 = 4;\n",
        "crates/util/Cargo.toml": "",
        "crates/util/src/lib.rs": "pub mod shape;\n",
        "crates/util/src/shape.rs": "pub fn f() {}\n",
        "crates/app/Cargo.toml": "",
        "crates/app/src/lib.rs": "",
        "tools/ref/models/beta.sh": "MODEL=/models/Beta-7B/m.gguf\n",
    }
    spec_call = "#[test]\nfn t() {\n    model::arch::spec(\"x\");\n}\n"
    tests = {
        "open": spec_call,
        "walk": "#[test]\nfn t() {\n    model::arch::spec(\"x\");\n    model::arch::registry::count();\n}\n",
        "alpha_meta": spec_call,
        "lit": "const M: &str = \"/models/Beta-7B-Q4/m.gguf\";\n" + spec_call,
        "stray": "const M: &str = \"/models/Nobody/m.gguf\";\n" + spec_call,
        "ns": "use model::arch::util::shape;\n#[test]\nfn t() {\n    shape::f();\n}\n",
    }
    for name, text in tests.items():
        files[f"crates/app/tests/{name}.rs"] = text
    with tempfile.TemporaryDirectory(prefix="recipes-dispatch-") as tmp:
        for rel, text in files.items():
            os.makedirs(os.path.join(tmp, os.path.dirname(rel)), exist_ok=True)
            with open(os.path.join(tmp, rel), "w", encoding="utf-8") as fh:
                fh.write(text)

        def pkg(name: str, targets: list[tuple[str, str, str]], deps: list[str]) -> dict:
            return {
                "id": name, "name": name, "manifest_path": os.path.join(tmp, "crates", name, "Cargo.toml"), "features": {},
                "dependencies": [{"name": d, "kind": None, "optional": False, "features": [], "uses_default_features": True} for d in deps],
                "targets": [{"kind": [k], "name": n, "src_path": os.path.join(tmp, "crates", name, s)} for k, n, s in targets],
            }

        meta = {
            "workspace_root": tmp, "workspace_members": ["model", "util", "app"],
            "packages": [
                pkg("model", [("lib", "model", "src/lib.rs")], ["util"]),
                pkg("util", [("lib", "util", "src/lib.rs")], []),
                pkg("app", [("lib", "app", "src/lib.rs"), *(("test", n, f"tests/{n}.rs") for n in tests)], ["model", "util"]),
            ],
        }
        st = Tree(tmp, meta)

        def gate(name: str, test: str, profile: str | None = None, flags: str = "") -> Recipe:
            env = f"BLOOMERY_MODEL={profile} " if profile else ""
            return Recipe(name, [f"{env}./tools/box.sh 'bash tools/gate.sh -p app --test {test}{flags}'"], [], "")

        rs = {
            "gate-a": gate("gate-a", "open", "alpha"), "gate-b": gate("gate-b", "open", "beta"), "gate-d": gate("gate-d", "open"),
            "gate-walk": gate("gate-walk", "walk", "beta"), "gate-name": gate("gate-name", "alpha_meta"),
            "gate-lit": gate("gate-lit", "lit"), "gate-stray": gate("gate-stray", "stray"), "gate-flag": gate("gate-flag", "open", "beta", " -- -m x"),
            "gate-ns": gate("gate-ns", "ns"),
            "gate-lib": Recipe("gate-lib", ["BLOOMERY_MODEL=beta ./tools/box.sh 'bash tools/gate.sh -p model --lib'"], [], ""),
        }
        sg = Graph(st, rs)
        fa, fb = "crates/model/src/arch/fam_alpha/hp.rs", "crates/model/src/arch/fam_beta/hp.rs"
        shared2, sh = "crates/model/src/arch/shared2.rs", "crates/model/src/arch/shared.rs"
        every = list(rs)

        r = reach_of(st)

        def reached(f: str) -> set[str]:
            return {n for n in every if f in r.gate_files(sg, n)}


        expect(len(r.dispatches()) == 2 and all(len(d.rows) == 3 for d in r.dispatches().values()),
               f"dispatch: the table is read in the lib's root and its test build, three rows each: {[len(d.rows) for d in r.dispatches().values()]}")
        d0 = next(iter(r.dispatches().values()))
        expect(d0.rows[2] == frozenset({"beta", "beta_alias"}), f"dispatch: a row's const name is read: {sorted(d0.rows[2])}")
        # a family's own files: the gates that open the family (profile, a target's name, a /models path a profile defines), and the
        # gates that cannot tell their files or walk the table; never the gate of another family or of the default profile
        want_a = {"gate-a", "gate-name", "gate-walk", "gate-stray", "gate-flag", "gate-lib"}
        want_b = {"gate-b", "gate-walk", "gate-lit", "gate-stray", "gate-flag", "gate-lib"}
        expect(reached(fa) == want_a, f"dispatch: {fa} is reached by {sorted(reached(fa))}, not {sorted(want_a)}")
        expect(reached(fb) == want_b, f"dispatch: {fb} is reached by {sorted(reached(fb))}, not {sorted(want_b)}")
        # a file of the architecture tree only alpha's reader uses follows alpha; what it calls outside the tree (ops.rs) is reached by
        # every gate that reaches the table, as before
        expect(reached(shared2) == want_a, f"dispatch: {shared2}, a helper of the architecture tree only alpha's reader calls, is reached by {sorted(reached(shared2))}")
        ops = "crates/model/src/ops.rs"
        expect(reached(ops) == set(every) - {"gate-ns"}, f"dispatch: {ops}, a file outside the architecture tree that alpha's reader calls, is reached by {sorted(reached(ops))}")
        expect(reached(sh) == set(every) - {"gate-ns", "gate-lib"}, f"dispatch: {sh}, the unregistered arm, is reached by {sorted(reached(sh))}")
        # a name a module imports whose path ends at a namespace reaches the import, not every unit of the module
        expect("gate-ns" not in reached(fa) and "gate-ns" not in reached(fb) and "gate-ns" in reached("crates/model/src/arch/mod.rs"),
               f"dispatch: a path through `pub use util;` to a module reaches every unit of the importing module: {sorted(reached(fa))}")
        # the reason `reach --brief` prints
        why = {n: r.reason(sg, n, r.gate_files(sg, n)[fa]) for n in want_a if fa in r.gate_files(sg, n)}
        expect(why.get("gate-a") == "registry row alpha: profile alpha" and why.get("gate-name") == "registry row alpha: target alpha_meta"
               and why.get("gate-walk", "").startswith("registry row alpha: the gate walks every row (")
               and why.get("gate-flag", "").startswith("registry row alpha: every row open (gate-flag picks its model file by `-m`)")
               and why.get("gate-stray", "").startswith("registry row alpha: every row open (its targets name /models/Nobody"),
               f"dispatch: the reasons of a kept gate: {why}")
        expect(r.gate_archs(sg, "gate-lit")[0] == frozenset({"deepseek2", "beta"}), f"dispatch: a /models path that extends a profile's directory by a `-` names that profile: {r.gate_archs(sg, 'gate-lit')}")
        expect(r.gate_archs(sg, "gate-d")[0] == frozenset({"deepseek2"}), f"dispatch: no profile is the default profile: {r.gate_archs(sg, 'gate-d')}")



def dispatch_real_self_test(expect, side: Side) -> None:
    tree, recipes, graph = side.tree, side.recipes, side.graph
    rr = reach_of(tree)
    gates = gate_order(recipes)
    ds = rr.dispatches()
    expect(bool(ds) and bool(rr._enum_nodes), "dispatch: the real registry's table is not read")
    d = next(iter(ds.values()))
    by_dir: dict[str, set[int]] = {}
    for x, rows in d.entries.items():
        by_dir.setdefault(os.path.dirname(rr.nodes[x][1]), set()).update(rows)
    families = {dd: rows for dd, rows in by_dir.items() if dd != os.path.dirname(ARCH_REGISTRY[0])}
    expect(len(families) >= 8, f"dispatch: {len(families)} family directories behind the registry's rows")
    for dd, rows in sorted(families.items()):
        names = frozenset().union(*(d.rows[i] for i in rows))
        fam = os.path.basename(dd)
        f = f"{dd}/spec.rs"
        if not tree.exists(f):
            continue
        got = dep_reach(tree, graph, gates, f)
        for n in gates:
            archs, _ = rr.gate_archs(graph, n)
            par = rr.gate(graph, n)
            dispatches = any(x in par for x in rr._rows_nodes)
            if archs is not None and archs & names and dispatches and graph.inputs(n).match(f) is not None:
                expect(f in rr.gate_files(graph, n), f"dispatch: {n} opens {fam} (its {sorted(archs & names)}) and reaches the registry's table, but {f} is not in its reach")
        for n, node in got.items():
            archs, _ = rr.gate_archs(graph, n)
            if archs is not None and not archs & names and not any(x in rr.gate(graph, n) for x in rr._enum_nodes):
                chain = reach_chain(tree, graph, n, node)
                expect("(static ROWS) -[" not in chain, f"dispatch: {n} cannot open {fam} (it opens {sorted(archs)}) and reaches {f} through the registry's pointer: {chain[-300:]}")
    # a file outside the families is reached by the same gates with every row open: the rule narrows family files only
    common = ["crates/gpu/src/weights.rs", "crates/model/src/arch/mod.rs", "crates/placement/src/placement.rs"]
    for f in common:
        cut = {n for n in gates if f in rr.gate_files(graph, n)}
        full = {n for n in gates if any(rr.nodes[m][1] == f for m in rr._close(graph, n, None))}
        expect(cut == full, f"dispatch: {f} is a common file, and the dispatch edges move its gates: {sorted(cut ^ full)}")


def narrow_table_self_test(expect, side: Side) -> None:
    """The gate-paths table's checks and the host group's guard on the real tree."""

    def row(glob: str, names: list[str], line: int = 1, derive: bool = False) -> PathRow:
        return PathRow(line, glob, names, "why", glob_regex(glob), derive=derive)

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
    expect(not got, f"gate-paths: the real table, {len(got)} problems (a row that holds the derivation writes `{DERIVE}` in its recipes column):\n"
           + "\n".join(got))
    derive_self_test(expect, side, row)
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


def derive_self_test(expect, side: Side, row) -> None:
    """The `derive` rows: both halves of the derivation on the real tree, the text form and its refusals, the expanded text, the
    cache and its key, the named end of `--fill`."""
    tree, recipes = side.tree, side.recipes
    # the gate that reaches a file through another lib's code, and the gate whose bin names the module: the two halves of one derivation
    drv: dict[int, list[str]] = {}
    gate_paths_problems(tree, recipes, [row("crates/model/src/{ops.rs,moe.rs}", [], derive=True)], derived=drv)
    expect("gate-gpu-ds41-step" in drv.get(1, []), f"derive: crates/model/src/{{ops.rs,moe.rs}} derives no gate-gpu-ds41-step, which reaches moe.rs through bloomery_gpu's host tier: {drv}")
    drv = {}
    gate_paths_problems(tree, recipes, [row("crates/gpu-deepseek41/src/{body.rs,body/**}", [], derive=True)], derived=drv)
    expect("gate-gpu-ds41-long" in drv.get(1, []), f"derive: the body row derives no gate-gpu-ds41-long, whose bin names bloomery_gpu_deepseek41::body: {drv}")
    # a gate the tree gives a row by either half is one of its gates, in the justfile's order
    both = Derivation({"gate-b": ("f.rs", 1)}, set(), {"gate-c": ("bin", "t", "t.rs", "m::x")})
    expect(both.gates(["gate-d", "gate-c", "gate-a", "gate-b"]) == ["gate-c", "gate-b"], f"derive: a gate only the module-naming half holds, or the order: {both.gates(['gate-d', 'gate-c', 'gate-a', 'gate-b'])}")
    # the text form
    text = "# t\ncrates/a/**\tderive gate-x weekly-y\twhy\ncrates/b/**\t*\twhy\n"
    rows = parse_gate_paths(text)
    expect([(r.line, r.derive, r.recipes) for r in rows] == [(2, True, ["gate-x", "weekly-y"]), (3, False, ["*"])], f"derive: the text form read as {[(r.line, r.derive, r.recipes) for r in rows]}")
    expect(parse_gate_paths("# t\ncrates/a/**\tderive\twhy\n")[0].recipes == [], "derive: a row of `derive` alone")
    for bad, why in [("gate-x derive", "opens the recipes column"), ("derive *", "`*` stands alone"), ("derive derive", "named twice")]:
        expect(_raises(lambda: parse_gate_paths(f"crates/a/**\t{bad}\twhy\n"), why), f"derive: {bad!r} in the recipes column not refused as {why!r}")
    expect(_raises(lambda: parse_gate_paths("crates/a/**\tderive gate-x\twhy\texcept gate-x: r\n"), "both in the row and left out of it"),
           "derive: a hand name the exceptions leave out not refused")
    wide = [expanded_row(rows[0], ["gate-d"]), rows[1]]
    expect(expanded_text(text, rows, wide) == "# t\ncrates/a/**\tgate-d gate-x weekly-y\twhy\ncrates/b/**\t*\twhy\n",
           f"derive: the expanded text keeps its lines: {expanded_text(text, rows, wide)!r}")
    # the cache: a stored expansion reads back for its table only; the directory keeps the newest
    with tempfile.TemporaryDirectory(prefix="recipes-gp-cache-") as tmp:
        d = os.path.join(tmp, "c")
        path = os.path.join(d, "k.tsv")
        store_expanded_cache(path, expanded_text(text, rows, wide))
        got = read_expanded_cache(path, rows)
        expect(got is not None and [g.recipes for g in got] == [["gate-d", "gate-x", "weekly-y"], ["*"]] and not any(g.derive for g in got),
               f"derive: a stored expansion does not read back: {got}")
        for what, body in (("an unexpanded table", text), ("another table", expanded_text(text, rows, wide).replace("crates/a/**", "crates/z/**")),
                           ("a table that lost a hand name", expanded_text(text, rows, wide).replace(" weekly-y", ""))):
            store_expanded_cache(path, body)
            expect(read_expanded_cache(path, rows) is None, f"derive: {what} read as the cache of this table")
        expect(read_expanded_cache(os.path.join(d, "none.tsv"), rows) is None, "derive: a missing cache file read")
        for i in range(GATE_PATHS_CACHE_KEEP + 3):
            old = os.path.join(d, f"old{i:02d}.tsv")
            with open(old, "w", encoding="utf-8") as fh:
                fh.write(text)
            os.utime(old, (1000 + i, 1000 + i))
        store_expanded_cache(path, expanded_text(text, rows, wide))
        left = sorted(e.name for e in os.scandir(d) if e.name.endswith(".tsv"))
        expect(len(left) == GATE_PATHS_CACHE_KEEP and "k.tsv" in left and "old00.tsv" not in left and f"old{GATE_PATHS_CACHE_KEEP + 2:02d}.tsv" in left,
               f"derive: the cache directory does not keep the newest {GATE_PATHS_CACHE_KEEP}: {left}")
        # the key: a directory with no .git has none; in a work tree it moves with the table, a tracked edit and an untracked file
        plain = os.path.join(tmp, "plain")
        os.makedirs(plain)
        expect(gate_paths_key(plain, text) is None, "derive: a directory with no .git has a cache key")
        repo = os.path.join(tmp, "repo")
        os.makedirs(repo)

        def git_in(*a: str) -> subprocess.CompletedProcess:
            return subprocess.run(["git", "-C", repo, "-c", "user.email=t@t", "-c", "user.name=t", *a], capture_output=True, text=True)

        with open(os.path.join(repo, "a.rs"), "w", encoding="utf-8") as fh:
            fh.write("fn a() {}\n")
        if git_in("init", "-q", "-b", "main").returncode == 0 and git_in("add", "-A").returncode == 0 and git_in("commit", "-q", "-m", "x").returncode == 0:
            k0 = gate_paths_key(repo, text)
            expect(k0 is not None and k0 == gate_paths_key(repo, text), "derive: a clean work tree has no stable cache key")
            expect(gate_paths_key(repo, text + "# more\n") != k0, "derive: the cache key ignores the table's text")
            with open(os.path.join(repo, "a.rs"), "w", encoding="utf-8") as fh:
                fh.write("fn a() { 1; }\n")
            expect(gate_paths_key(repo, text) != k0, "derive: the cache key ignores a tracked edit")
            git_in("checkout", "-q", "--", "a.rs")
            expect(gate_paths_key(repo, text) == k0, "derive: the cache key moves with a reverted edit")
            with open(os.path.join(repo, "b.rs"), "w", encoding="utf-8") as fh:
                fh.write("fn b() {}\n")
            expect(gate_paths_key(repo, text) != k0, "derive: the cache key ignores an untracked file")
            # a store under the key taken before the derivation: kept while the tree holds it, refused once a file moved since
            os.makedirs(os.path.join(repo, os.path.dirname(GATE_PATHS)), exist_ok=True)
            with open(os.path.join(repo, GATE_PATHS), "w", encoding="utf-8") as fh:
                fh.write(text)
            git_in("add", "-A")
            git_in("commit", "-q", "-m", "t")
            cache = os.path.join(tmp, "store-cache")
            k1 = gate_paths_key(repo, text)
            derived = {r.line: ["gate-w"] for r in rows if r.derive}
            expect(k1 is not None and store_checked_expansion(repo, k1, text, derived, cache) and os.path.exists(os.path.join(cache, k1 + ".tsv")),
                   "derive: a store under the key of an unmoved tree is not written")
            with open(os.path.join(repo, "a.rs"), "w", encoding="utf-8") as fh:
                fh.write("fn a() { 2; }\n")
            k2 = gate_paths_key(repo, text)
            expect(not store_checked_expansion(repo, k1, text, derived, cache) and k2 is not None and not os.path.exists(os.path.join(cache, k2 + ".tsv")),
                   "derive: an expansion derived before a tracked edit is stored for the edited tree")
            expect(not store_expansion(repo, k1, cache, text, rows, rows) and not os.path.exists(os.path.join(cache, k2 + ".tsv")),
                   "derive: a cold expansion derived before a tracked edit is stored for the edited tree")
        else:
            expect(False, "derive: the key fixture's git repository could not be made")
    expect(_raises(lambda: cmd_check(argparse.Namespace(fill=True, justfile=None)), "derives its gates"), "derive: `check --fill` not refused by name")



def bin_feature_self_test(expect) -> None:
    """The bin check on synthetic justfiles over the real tree's gate_glm5next_e2e (requires glm5next; `vision` is the
    feature the 2026-10-12 incident differed by). Goes through check(), the entry check-recipes runs."""

    def call(feats: str, flags: str = "--release", sub: str = "cargo oxide build --arch sm_86 --", bin_: str = "gate_glm5next_e2e") -> str:
        return f"{sub} -p bloomery-gpu-gates --features {feats} {flags} --bin {bin_}"

    def problems(*recipes: tuple[str, ...]) -> list[str]:
        with tempfile.TemporaryDirectory(prefix="recipes-binfeat-") as tmp:
            p = os.path.join(tmp, "justfile")
            with open(p, "w", encoding="utf-8") as fh:
                for name, *cmds in recipes:
                    # a recipe's commands are one `./tools/box.sh '…'` line; "---" starts the next recipe line, "+++" the
                    # next box.sh call on the same line, and a line that opens with "local" runs its commands on the Mac
                    lines: list[list[str]] = [[]]
                    for c in cmds:
                        lines.append([]) if c == "---" else lines[-1].append(c)
                    fh.write(f"{name}:\n")
                    for ln in lines:
                        local = ln[:1] == ["local"]
                        groups: list[list[str]] = [[]]
                        for c in ln[local:]:
                            groups.append([]) if c == "+++" else groups[-1].append(c)
                        words = [" && ".join(g) if local and i == 0 else f"./tools/box.sh '{' && '.join(g)}'" for i, g in enumerate(groups) if g]
                        fh.write(f"    {' && '.join(words)}\n")
            try:
                return [x for x in check(Tree(ROOT), load_justfile(p)) if "gate_glm5next_e2e" in x]
            except RecipeError as err:
                return [f"raised {err}"]

    base = "gate_glm5next_e2e (bloomery-gpu-gates, release): "
    tree = Tree(ROOT)
    pkg = tree.packages["bloomery-gpu-gates"]
    closure = lambda feats: "{" + ",".join(sorted(tree.feature_closure(pkg, feats.split(","))[0])) + "}"
    wide, narrow = closure("glm5next,vision"), closure("glm5next")
    got = problems(("a", call("glm5next")), ("b", call("glm5next,vision")))
    expect(len(got) == 1 and got[0].startswith(f"{base}b {wide} vs a {narrow} "), f"bin features: glm5next vs glm5next,vision not refused by name: {got}")
    # the recipes' order and names do not move the line; a set shared by recipes lists them all, sorted
    got2 = problems(("z", call("glm5next,vision")), ("y", call("glm5next")), ("x", call("glm5next")))
    expect(len(got2) == 1 and got2[0].startswith(f"{base}z {wide} vs x, y {narrow} "), f"bin features: recipes of one set not grouped and sorted: {got2}")
    expect(problems(("a", call("glm5next,vision")), ("b", call("vision,glm5next"))) == [], "bin features: one set in another order refused")
    # resolved, not spelled: glm5next closes over deepseek41 and gpu
    expect(problems(("a", call("glm5next")), ("b", call("deepseek41,glm5next,gpu"))) == [], "bin features: two spellings of one closure refused")
    # the profile is the directory the bin lands in: dev beside release is two files, `--profile release` is release
    expect(problems(("a", call("glm5next")), ("b", call("glm5next,vision", flags=""))) == [], "bin features: dev and release refused")
    got = problems(("a", call("glm5next")), ("b", call("glm5next,vision", flags="--profile release")))
    expect(len(got) == 1 and got[0].startswith(base), f"bin features: --profile release not read as release: {got}")
    got = problems(("a", call("glm5next", flags="--release --target x86_64-unknown-linux-gnu")), ("b", call("glm5next,vision")))
    expect(got == [], f"bin features: a --target build refused beside a host build: {got}")
    # only a build that writes <dir>/<profile>/<bin> counts: a check or a test of the bin writes another file
    expect(problems(("a", call("glm5next")), ("b", call("glm5next,vision", sub="cargo check"))) == [], "bin features: a cargo check refused")
    expect(problems(("a", call("glm5next")), ("b", call("glm5next,vision", sub="cargo test"))) == [], "bin features: a cargo test refused")
    # 2026-10-13: a recipe building a bin with two sets in turn is refused too (round 1 exempted it as «its own serial
    # order»: that case read `== []`). Derivation: the recipe's order protects nothing, because the file it replaces is
    # the one every other lane runs from the same path (cargo relinks target/release/oxart_jit under a running ptx-scan;
    # ptx-scan has overwritten a binary 4 times), so one recipe's second build is the same defect as a second recipe's.
    # FAIL-first: the round-1 check, with its single-recipe exemption, returns [] for these two.
    got = problems(("a", call("glm5next"), call("glm5next,vision")))
    expect(len(got) == 1 and got[0].startswith(f"{base}a {wide} vs a {narrow} "), f"bin features: one recipe's serial builds of one file, two sets, not refused: {got}")
    got = problems(("a", call("glm5next"), call("glm5next,vision")), ("b", call("glm5next")))
    expect(len(got) == 1 and f"a {wide} vs a, b {narrow} " in got[0], f"bin features: a serial recipe and a second one on its bin not one line: {got}")
    # the key is the directory the call writes to (2026-10-13): the same bin with two sets in two directories is two files
    def to(d: str) -> str:
        return f"--release --target-dir {d}"

    expect(problems(("a", call("glm5next")), ("b", call("glm5next,vision", flags=to("target/b")))) == [], "bin features: a --target-dir build refused beside the default dir")
    got = problems(("a", call("glm5next", flags=to("target/b"))), ("b", call("glm5next,vision", flags=to("target/b"))))
    expect(len(got) == 1 and got[0].startswith("gate_glm5next_e2e (bloomery-gpu-gates, target/b/release): "), f"bin features: two sets in one --target-dir not refused, or not named by it: {got}")
    expect(problems(("a", call("glm5next", flags=to("target/a")), call("glm5next,vision", flags=to("target/b")))) == [], "bin features: one recipe's builds in two target dirs refused")
    got = problems(("a", call("glm5next")), ("b", call("glm5next,vision", flags=to("./target/"))))
    expect(len(got) == 1 and got[0].startswith(base), f"bin features: `./target/` not read as the default dir: {got}")
    expect(problems(("a", call("glm5next")), ("b", "CARGO_TARGET_DIR=target/b " + call("glm5next,vision"))) == [], "bin features: a CARGO_TARGET_DIR prefix on the call refused")
    expect(problems(("a", call("glm5next")), ("b", "export CARGO_TARGET_DIR=target/b", call("glm5next,vision"))) == [], "bin features: an exported CARGO_TARGET_DIR refused")
    got = problems(("a", call("glm5next")), ("b", call("glm5next,vision"), "export CARGO_TARGET_DIR=target/b"))
    expect(len(got) == 1, f"bin features: an export read back onto the build before it: {got}")
    got = problems(("a", call("glm5next")), ("b", "export CARGO_TARGET_DIR=target/b", "unset CARGO_TARGET_DIR", call("glm5next,vision")))
    expect(len(got) == 1, f"bin features: an unset CARGO_TARGET_DIR still read: {got}")
    # an export holds in its own shell only: just runs each recipe line in a shell, and each box.sh call is a remote shell
    got = problems(("a", call("glm5next")), ("b", "local", "export CARGO_TARGET_DIR=target/b", "---", "local", call("glm5next,vision")))
    expect(len(got) == 1, f"bin features: an export read onto the next recipe line: {got}")
    got = problems(("a", call("glm5next")), ("b", "local", "export CARGO_TARGET_DIR=target/b", "+++", call("glm5next,vision")))
    expect(len(got) == 1, f"bin features: a local export read into a box.sh call: {got}")
    got = problems(("a", call("glm5next")), ("b", "export CARGO_TARGET_DIR=target/b", "+++", call("glm5next,vision")))
    expect(len(got) == 1, f"bin features: an export read onto the next box.sh call: {got}")
    expect(problems(("a", call("glm5next")), ("b", "local", "export CARGO_TARGET_DIR=target/b", call("glm5next,vision"))) == [], "bin features: a local export beside its own build refused")
    expect(problems(("a", call("glm5next")), ("b", call("glm5next,vision", sub="cargo oxide build --arch sm_86 --cargo-target-dir target/b --"))) == [], "bin features: oxide's --cargo-target-dir refused")
    expect(problems(("a", call("glm5next")), ("b", call("glm5next,vision", sub="cargo oxide build --arch sm_86 --cargo-target-dir=target/b --"))) == [], "bin features: oxide's --cargo-target-dir=<dir> refused")
    # the flag beats the environment, cargo's `--target-dir` over oxide's flag over a prefix over an export
    got = problems(("a", call("glm5next", flags=to("target/x"))), ("b", "CARGO_TARGET_DIR=target/elsewhere " + call("glm5next,vision", flags=to("target/x"))))
    expect(len(got) == 1, f"bin features: a CARGO_TARGET_DIR prefix read over the call's --target-dir: {got}")
    got = problems(("a", call("glm5next", flags=to("target/x"))), ("b", call("glm5next,vision", flags=to("target/x"), sub="cargo oxide build --arch sm_86 --cargo-target-dir target/elsewhere --")))
    expect(len(got) == 1, f"bin features: oxide's --cargo-target-dir read over cargo's --target-dir: {got}")
    # a parameter decides the features or the directory at run time: left out, not guessed
    expect(problems(("a", call("glm5next")), ("b FEATURES", call("{{FEATURES}}"))) == [], "bin features: a parameter's features refused")
    expect(problems(("a DIR", call("glm5next", flags="--release --target-dir {{DIR}}")), ("b DIR", call("glm5next,vision", flags="--release --target-dir {{DIR}}"))) == [], "bin features: a parameter's target dir refused")
    # no feature set differs for a bin built once
    expect(problems(("a", call("glm5next,vision"))) == [], "bin features: a lone build refused")


def _raises(fn: Callable[[], object], why: str) -> bool:
    """Whether `fn` raises a RecipeError that names `why`."""
    try:
        fn()
    except RecipeError as err:
        return why in str(err)
    return False


def _expecting(fails: list[str]) -> Callable[[bool, str], None]:
    def expect(cond: bool, what: str) -> None:
        if not cond:
            fails.append(what)

    return expect


def _forked(jobs: list[tuple[str, Callable[[list[str]], None]]], serial: bool = False) -> tuple[list[str], list[tuple[str, float]]]:
    """The failures of every job, in job order, and each job's wall seconds. Each job runs in a forked child of this process (`serial`: in this one), so
    the parsed real tree is shared copy-on-write while each job's temp directories, environment and working directory stay
    its own. A job that raises reports its traceback as a failure; a child that ends without a report is a failure naming it."""
    import multiprocessing
    import traceback

    def run(name: str, job: Callable[[list[str]], None]) -> tuple[list[str], float]:
        got: list[str] = []
        t0 = time.monotonic()
        try:
            job(got)
        except Exception:
            got.append(f"{name}: raised\n{traceback.format_exc()}")
        return got, time.monotonic() - t0

    out: list[str] = []
    times: list[tuple[str, float]] = []
    if serial:
        for name, job in jobs:
            got, secs = run(name, job)
            out.extend(got)
            times.append((name, secs))
        return out, times
    ctx = multiprocessing.get_context("fork")
    started = []
    for name, job in jobs:
        recv, send = ctx.Pipe(duplex=False)

        def child(name: str = name, job: Callable[[list[str]], None] = job, send=send) -> None:
            send.send(run(name, job))
            send.close()

        proc = ctx.Process(target=child)
        proc.start()
        send.close()
        started.append((name, proc, recv))
    for name, proc, recv in started:
        try:
            got, secs = recv.recv()
            out.extend(got)
            times.append((name, secs))
        except EOFError:
            proc.join()
            out.append(f"{name}: its process ended without a report (exit code {proc.exitcode})")
        proc.join()
    return out, times


def profile_inputs_self_test(expect, side: Side) -> None:
    for value, profile in (("glm5next", "glm5next"), ("${BLOOMERY_MODEL:-deepseek41}", "deepseek41")):
        recipes = {"x": Recipe("x", [f"BLOOMERY_MODEL={value} ./tools/box.sh 'true'", "BLOOMERY_MODEL=qwen3moe ./tools/box.sh 'true'"], [], "")}
        probe = Side(side.tree, recipes, Graph(side.tree, recipes))
        inputs = probe.graph.inputs("x")
        paths = {f"tools/ref/models/{profile}.sh", "tools/ref/models/qwen3moe.sh"}
        expect(paths <= inputs.files.keys() and not inputs.errors, f"profile inputs: {value}: {inputs.files}, {inputs.errors}")
        expect(all("$" not in p for p in inputs.files), f"profile inputs: shell expression used as a path: {inputs.files}")
        for env in ({}, {"BLOOMERY_MODEL": ""}, {"BLOOMERY_MODEL": "glm5next"}, {"BLOOMERY_MODEL": "qwen35moe"}):
            context = KeyContext(probe, None, environ=env)
            selected = env["BLOOMERY_MODEL"] if value.startswith("$") and env.get("BLOOMERY_MODEL") else profile
            path = f"tools/ref/models/{selected}.sh"
            expect(path in context.run_scripts(["x"]), f"profile scripts: {value}, {env}: missing {path}")
            if value.startswith("$"):
                context.manifest = [f"model-dir\t{d}\tabsent" for d in tree_model_dirs(side.tree.root)]
                parts, err, _, _ = context._parts("x", ["x"], inputs, [], [])
                expect(any(p.startswith(f"file\t{path}\t") for p in parts), f"profile key: {value}, {env}: {err}, missing {path}")
    with tempfile.TemporaryDirectory(prefix="recipes-profile-") as tmp:
        manifest = os.path.join(tmp, "manifest")
        with open(manifest, "w", encoding="utf-8") as fh:
            fh.write(f"# {MANIFEST_VERSION}\n# end\n")
        for prefix in ("", "BLOOMERY_MODEL=${BLOOMERY_MODEL:-deepseek41} "):
            recipes = {"x": Recipe("x", [f"{prefix}./tools/box.sh 'true'"], [], "")}
            probe = Side(side.tree, recipes, Graph(side.tree, recipes))
            for value in ("./glm5next", "../glm5next", "a b"):
                parts, err, _, _ = KeyContext(probe, manifest, environ={"BLOOMERY_MODEL": value}).parts("x")
                expect(not parts and err is not None and f"BLOOMERY_MODEL={value!r}: cannot resolve model profile" in err,
                       f"profile key: unsupported inherited {value!r} not refused: {len(parts)} parts, {err}")
    for value in ("${OTHER:-deepseek41}", "$BLOOMERY_MODEL", "../deepseek41"):
        recipes = {"x": Recipe("x", [f"BLOOMERY_MODEL={value} ./tools/box.sh 'true'"], [], "")}
        inputs = Graph(side.tree, recipes).inputs("x")
        expect(any("BLOOMERY_MODEL" in e and value in e for e in inputs.errors), f"profile inputs: unresolved {value!r} not refused: {inputs.errors}")
    for name in ("bench-cpu-v41-host-check", "time-cpu-v41-host", "depth-gpu-qwen3moe", "time-dma-dram"):
        inputs = side.graph.inputs(name)
        expect(not inputs.errors and all("$" not in p for p in inputs.files), f"profile inputs: {name}: {inputs.errors}")


def _st_basics(fails: list[str], side: Side) -> None:
    expect = _expecting(fails)

    profile_inputs_self_test(expect, side)

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
                "b:\n    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_deepseek41_kernels'\n"
                "c:\n    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu --release --bin gate_swap'\n"
                "d:\n    ./tools/box.sh 'bash tools/gate.sh -p bloomery-gpu-gates --release --lib -- --include-ignored'\n"
                "e:\n    ./tools/box.sh 'bash tools/gate.sh -p bloomery-model --release --doc'\n"
                "f FEATURES:\n    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features {{FEATURES}} --release --bin gate_p1 && cargo build --release -p bloomery-gpu-gates --bin oxart_ptx'\n"
            )
        try:
            shapes, skips, n = build_shapes(side.tree, load_justfile(p))
        except RecipeError as err:
            shapes, skips, n = [], [], 0
            fails.append(f"combos: the synthetic justfile raised {err}")
        cmds = {c.command(): c.recipes for c in shapes}
        want = {
            "cargo check -p bloomery-gpu-gates --features deepseek41,gpu --bin gate_deepseek41_kernels --bin generate_ds41": ["a", "b"],
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
            build_shapes(side.tree, load_justfile(p))
            fails.append("combos: a call naming no target accepted")
        except RecipeError as err:
            expect("g: --bin no_such_bin" in str(err), f"combos: an unresolvable call refused without its name: {err}")
        # one call over two packages (`-p a -p b --features b/x`): b/x is b's own feature, and a's shape carries no
        # word for a package it does not depend on (cargo refuses one for `-p a` alone); a dependency's word stays
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(
                "h:\n    ./tools/box.sh 'cargo clippy -p bloomery-device -p bloomery-metal --features bloomery-metal/pipeline-scan,bloomery-metal/encode-trace --lib'\n"
                "i:\n    ./tools/box.sh 'cargo check -p bloomery-gpu-gates --features glm5next,bloomery-gpu-glm5next/hf --lib'\n"
            )
        try:
            got = {c.command() for c in build_shapes(side.tree, load_justfile(p))[0]}
        except RecipeError as err:
            got = set()
            fails.append(f"combos: a call over two packages raised {err}")
        want = {
            "cargo check -p bloomery-device --lib",
            "cargo check -p bloomery-metal --features encode-trace,pipeline-scan --lib",
            "cargo check -p bloomery-gpu-gates --features glm5next,bloomery-gpu-glm5next/hf --lib",
        }
        expect(got == want, f"combos: a call over two packages derives {sorted(got)}, not {sorted(want)}")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("j:\n    ./tools/box.sh 'cargo check -p bloomery-device --features bloomery-metal/pipeline-scan --lib'\n")
        try:
            build_shapes(side.tree, load_justfile(p))
            fails.append("combos: a `dep/feature` word for a package the call's package does not depend on accepted")
        except RecipeError as err:
            expect("j: --features bloomery-metal/pipeline-scan" in str(err) and "bloomery-device depends on no bloomery-metal" in str(err),
                   f"combos: a `dep/feature` word for a package the call's package does not depend on refused without naming it: {err}")

    # one binary, one feature set
    bin_feature_self_test(expect)

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



def _st_real_tree(fails: list[str], side: Side) -> None:
    expect = _expecting(fails)
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
    expect({"gate-ds2-release", "gate-gpu-e2e", "gate-gpu-hybrid"} <= prompts, f"prompts.tsv selects {sorted(prompts)}")
    common = sel("crates/model/tests/common/asserts.rs")
    expect("gate-ds2-debug" in common and "gate-union" not in common, f"tests/common/asserts.rs selects {sorted(common)}")
    fixture = sel("crates/serve/tests/fixtures/qwen3-renders.json")
    expect(fixture == {"gate-serve"}, f"serve fixture selects {sorted(fixture)}")
    # every table the recipe names: the base and the directory of the other archs and capabilities, a table added there too
    for table in ("tools/ref/ptx-shapes.tsv", "tools/ref/ptx-shapes/sm_121-cc12.1.tsv", "tools/ref/ptx-shapes/sm_90-cc9.0.tsv"):
        spill = sel(table)
        expect(spill == {"gate-ptx-spill"}, f"{table} selects {sorted(spill)}")
    # a whole-string .rs literal of a workspace package is documentation, not a run-time read: levers'
    # registry names the in-place lever files, and only the targets that compile such a file select it
    dspark = sel("crates/gpu-gates/src/bin/shared/ds41_dspark.rs")
    # the bloomery-serve builders: the six seat gates and the two serve-start gates (each builds the whole bin)
    # PIN(2026-10-15): gate-gpu-nemotron-serve joins them; its recipe builds bloomery-serve beside its own bin
    # PIN(2026-10-16): gate-gpu-nemotron-h-moe-serve joins them; the same seat gate on the moe body
    # PIN(2026-10-16): the embedding serve gate also builds bloomery-serve, including its V4.1 seat.
    # PIN(2026-10-17): the v1 embedding serve gate builds the same server and its V4.1 seat.
    # PIN(2026-10-19): the Qwen3.8 media gate builds bloomery-serve beside its own bin for its serve clauses.
    # PIN(2026-10-20): gate-serve-draft-rules tests both serve bins, including their V4.1 draft.
    # PIN(2026-10-20): gate-gpu-solar2-serve joins them; its recipe builds bloomery-serve beside its own bin.
    # PIN(2026-10-21): gate-gpu-minimaxm3-serve joins them; its recipe (justfile) builds bloomery-serve beside its own bin.
    # PIN(2026-10-22): gate-gpu-qwen3-dspark joins them; its recipe builds bloomery-serve for the qwen3 seat's draft arms.
    # PIN(2026-10-23): gate-gpu-musespark-serve joins them; its recipe builds bloomery-serve beside its own bin, whose Muse seat names the module.
    # PIN(2026-10-23): gate-gpu-minimaxm3-twocard and gate-gpu-musespark-twocard join them; each runs its serve gate, which builds bloomery-serve.
    # PIN(2026-10-26): gate-gpu-llama-serve joins them; its recipe builds bloomery-serve beside its own bin.
    expect(dspark == {"gate-gpu-clef-serve", "gate-gpu-ds41-chat", "gate-gpu-ds41-draft", "gate-gpu-ds41-dspark-loop",
                      "gate-gpu-ds41-prefill", "gate-gpu-ds41-tier", "gate-gpu-ds41-twocard", "gate-gpu-gemma-embedding-serve", "gate-gpu-gemma-embedding2-serve", "gate-gpu-glm5next-serve",
                      "gate-gpu-lev-serve", "gate-gpu-llama-serve", "gate-gpu-minimaxm3-serve", "gate-gpu-minimaxm3-twocard", "gate-gpu-musespark-serve", "gate-gpu-musespark-twocard", "gate-gpu-nemotron-h-moe-serve", "gate-gpu-nemotron-serve",
                      "gate-gpu-qwen3-dspark", "gate-gpu-qwen3-serve", "gate-gpu-qwen38-media", "gate-gpu-serve-start-ds41", "gate-gpu-serve-start-glm", "gate-gpu-solar2-serve", "gate-ptx-spill", "gate-serve-draft-rules"},
           f"shared/ds41_dspark.rs (a #[path] module of the bins that name it) selects {sorted(dspark)}")
    hybrid = sel("crates/gpu/src/hybrid.rs")
    expect("gate-gpu-e2e" in hybrid and "gate-ops" not in hybrid and len(hybrid) < len(gates) - 20,
           f"a gpu lib file levers' registry names selects the gpu-linking gates only: {len(hybrid)} of {len(gates)}")
    qprof = sel("tools/ref/models/qwen3moe.sh")
    expect("gate-gpu-qwen3moe-e2e" in qprof and "gate-gpu-e2e" not in qprof, f"qwen3moe profile selects {sorted(qprof)[:5]}")
    # oracle.sh builds its Korean corpus with `find "$ROOT/docs" -name '*.md'`, but only `just dump-ref-tokenizer`
    # runs it: no gate recipe names it, so a docs change selects no gate
    docs = sel("docs/HARDWARE.md")
    expect(docs == set(), f"docs/HARDWARE.md selects {sorted(docs)}")
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
    # PIN(2026-10-20): gate-q38-serve-rules also runs the lib's serve_client tests and record schema includes.
    # PIN(2026-10-23): gate-gpu-rope-writers compiles the lib tests and their record schema includes.
    # PIN(2026-10-26): gate-gpu-llama-dense compiles the lib tests (its kv_planes:: clause) and their record schema includes.
    expect(schema == {"gate-gpu-gates-lib", "gate-gpu-llama-dense", "gate-gpu-rope-writers", "gate-ds41-oracle", "gate-ds41-kld", "gate-ds41-bind", "gate-ds41-place", "gate-q38-serve-rules", "weekly-gpu-ds41-flowcounts"}, f"a record schema selects {sorted(schema)}")
    flow = graph.inputs("weekly-gpu-ds41-flowcounts")
    expect(flow.match("tools/bloomery/hf.py") is None, "flowcounts must not key the unrelated hub reader through self-test prose")
    for path in ("tools/bloomery/records.py", "tools/flow/routes.py", "tools/gpu-ab.py", "tools/ref/tdist.py"):
        expect(flow.match(path) is not None, f"flowcounts lost its executable input {path}")
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
            'const N: &str = include_str!("../data/normal.txt");\n'
            # one-line attributed modules: the attributes count as on lines of their own
            '#[cfg(test)] mod one_t;\n#[path = "one_p.rs"] mod one_path;\n'
            '#[cfg(test)] #[path = "one_tp.rs"] pub(crate) mod one_both;\n'
            '#[cfg(test)] mod one_inline {\n    const O: &str = include_str!("../data/one_inline.txt");\n}\n'
            '#[cfg(feature = "off")] mod one_off;\n#[cfg(feature = "off")]\nmod two_off;\n#[doc = "a ] b"] mod one_doc;\n#[doc = "x\\\\"] mod one_esc;\n'
            '#[cfg(test)] /* why */ mod one_cmt;\n#[cfg(test)] #[path = "one_two.rs"] // the item is below\nmod one_two_mod;\n',
        )
        put("p/src/tests.rs", 'mod deep;\nconst T: &str = include_str!("../data/test.txt");\n')
        put("p/src/tests/deep.rs", 'const D: &str = "../tools/deep.txt";\nconst M: &str = "../q/src/main.rs";\n')
        for f in ("p/src/a.rs", "p/src/b.rs", "p/src/c.rs", "p/data/normal.txt", "p/data/test.txt", "p/data/inline.txt", "tools/deep.txt", "q/src/main.rs",
                  "p/src/one_t.rs", "p/src/one_p.rs", "p/src/one_tp.rs", "p/data/one_inline.txt", "p/src/one_off.rs", "p/src/two_off.rs", "p/src/one_doc.rs",
                  "p/src/one_esc.rs", "p/src/one_cmt.rs", "p/src/one_two.rs"):
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
        only_test = {"p/src/tests.rs", "p/src/tests/deep.rs", "p/src/c.rs", "p/data/test.txt", "p/data/inline.txt", "tools/deep.txt",
                     "p/src/one_t.rs", "p/src/one_tp.rs", "p/data/one_inline.txt", "p/src/one_cmt.rs", "p/src/one_two.rs"}
        built = {"p/src/lib.rs", "p/src/a.rs", "p/src/b.rs", "p/data/normal.txt", "p/src/one_p.rs", "p/src/one_doc.rs", "p/src/one_esc.rs"}
        lib, libtest, dep = (set(st.target_files(*t, [])) for t in (("p", "lib", None), ("p", "libtest", None), ("q", "bin", "q")))
        expect(built <= lib and not lib & only_test, f"cfg(test) walk: the lib build reads {sorted(lib & only_test)}, misses {sorted(built - lib)}")
        expect(built | only_test <= libtest, f"cfg(test) walk: the lib tests miss {sorted((built | only_test) - libtest)}")
        expect(built <= dep and not dep & only_test, f"cfg(test) walk: a dependent reads {sorted(dep & only_test)}, misses {sorted(built - dep)}")
        # a .rs literal names a workspace package's source: documentation of an owner, not a run-time
        # read — the module walk of whatever compiles it is its only route in (q's own bin keeps its
        # file, as every target keeps its own; p's views must not gain it)
        expect("q/src/main.rs" not in lib | libtest, "a .rs literal of a workspace package is an input")
        for view, name in ((lib, "lib"), (libtest, "lib tests"), (dep, "a dependent")):
            expect(("p/src/one_off.rs" in view) == ("p/src/two_off.rs" in view), f"cfg(test) walk: {name} reads a #[cfg(feature)] module on one line other than on two")
        expect(not st.unresolved, f"cfg(test) walk: unresolved {st.unresolved[:2]}")



def _st_pure_crates(fails: list[str], side: Side) -> None:
    expect = _expecting(fails)

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
            # The one owner of the OS calls: its Linux-only item sits behind the
            # platform's own cfg, so the crate and every crate that reaches it
            # stay pure. The same item outside that fence is a Linux-only item
            # the Mac cannot build, and the rule says so.
            "osfence": '#[cfg(target_os = "linux")]\npub fn f() -> i32 { unsafe { libc::O_DIRECT } }\n',
            "osoutside": "pub fn f() -> i32 { unsafe { libc::O_DIRECT } }\n",
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
            ("osoutside", "osoutside/src/lib.rs:1: a Linux-only libc item"),
        ]:
            expect(any(want in w for w in verdict.get(n, [])), f"pure: {n} is not rejected with '{want}': {verdict.get(n)}")
        for n in ("guarded", "inner", "user", "osfence"):
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
        with open(os.path.join(tmp, "tools", "check-recipes.sh"), "w", encoding="utf-8") as fh:
            fh.write("")
        expect(any("runs the recipes.py self-test that check-recipes already runs" in g for g in nightly_problems(pt, lock, {}, {}, {}, {"st": "python3 tools/recipes.py --self-test"})), "host nightly: an extra unit that repeats the check-recipes self-test")
        os.remove(os.path.join(tmp, "tools", "check-recipes.sh"))
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


def _st_host_crates(fails: list[str], side: Side) -> None:
    expect = _expecting(fails)
    tree = side.tree

    # the real tree: the rule's selection is the set whose native `cargo test` passes (`just mac-test`), and a
    # rejected crate names a device root or a source line
    real = dict(pure_crates(tree, load_lock(tree.root)))
    # PIN(2026-10-21): bloomery-device (the device contract, std and placement only) joined; `just mac-test`
    # runs it natively.
    # PIN(2026-10-23): bloomery-metal left: it reaches crates/model through bloomery-chain (a Linux /sys path);
    # its lib tests run in `just lab-metal`.
    pure_now = sorted(n for n, w in real.items() if not w)
    # PIN(2026-10-21): bloomery-record joined: the record core lifted out of gpu-gates
    # PIN(2026-10-29): qdot joined: its threads dependency's sysfs sibling lookup is Linux-only;
    # `cargo test -p bloomery-qdot` passes natively on macOS, replacing its threads rejection below.
    expect(
        pure_now == ["bloomery-decision", "bloomery-device", "bloomery-embed", "bloomery-gguf", "bloomery-hf", "bloomery-jinja", "bloomery-levers", "bloomery-models", "bloomery-os", "bloomery-placement", "bloomery-qdot", "bloomery-record", "bloomery-refset", "bloomery-runtime", "bloomery-safetensors", "bloomery-sampler", "bloomery-serve", "bloomery-tokenizer", "bloomery-verbs", "bloomery-vision"],
        f"pure: the real tree's pure crates are {pure_now} — a new member of the set is proven by `just mac-test` before this list takes it",
    )
    # the host chain's Linux-only items are the one owner's now, behind its own cfg, so a crate that holds
    # none of its own is pure; a crate that holds one is still named
    expect(not real.get("bloomery-gguf"), f"pure: gguf is rejected: {real.get('bloomery-gguf')}")
    expect(any("a Linux-only libc item" in w for ws in real.values() for w in ws), "pure: the rule names no Linux-only libc item in the tree")
    expect(any("bloomery-gpu -> cuda-" in w for w in real.get("bloomery-gpu", [])), f"pure: bloomery-gpu's reason {real.get('bloomery-gpu')}")
    expect(not real.get("bloomery-qdot"), f"pure: qdot is rejected: {real.get('bloomery-qdot')}")
    # host nightly on the real tree: the crates the set runs are the device-free ones and the admitted app; every gpu crate is left out
    # by its chain; every check script is a unit or an excluded one; the tables hold no stale entry
    lock_now = load_lock(tree.root)
    host_now = [n for n, _ in host_crates(tree, lock_now)]
    # PIN(2026-10-20): bloomery-model-hf joined with the tests it took out of bloomery-model (the fixture fetch's, the GLM checkpoint
    # map's), which the nightly ran as part of that crate.
    # PIN(2026-10-21): safetensors keeps the mapped reader's fixture tests previously run by bloomery-hf.
    # PIN(2026-10-21): bloomery-os joined with the OS calls the host chain now makes through it, which the nightly runs as part of
    # that crate.
    # PIN(2026-10-21): bloomery-device joined with the device contract's pure tests; bloomery-metal is an empty library off macOS.
    # PIN(2026-10-21): bloomery-record joined: the record core lifted out of gpu-gates
    # PIN(2026-10-22): bloomery-chain joined: the qwen3moe chain plan, host data moved out of bloomery-gpu.
    expect(
        host_now
        == [
            "bloomery-app", "bloomery-chain", "bloomery-decision", "bloomery-device", "bloomery-embed", "bloomery-engram", "bloomery-engram-lab", "bloomery-gguf", "bloomery-hf",
            "bloomery-jinja", "bloomery-levers", "bloomery-metal", "bloomery-model", "bloomery-model-hf", "bloomery-models", "bloomery-os", "bloomery-placement", "bloomery-qdot", "bloomery-record", "bloomery-refset",
            "bloomery-runtime", "bloomery-safetensors", "bloomery-sampler", "bloomery-serve", "bloomery-threads", "bloomery-tokenizer", "bloomery-verbs",
            "bloomery-vision",
        ],
        f"host nightly: the real tree's set is {host_now} — a new member of the set is proven by a nightly run before this list takes it",
    )
    left = [n for n, _ in host_left_out(tree, lock_now)]
    expect(left == ["bloomery-gpu", "bloomery-gpu-deepseek41", "bloomery-gpu-flint", "bloomery-gpu-gates", "bloomery-gpu-gemma-embedding2", "bloomery-gpu-gemma4a", "bloomery-gpu-glm5next", "bloomery-gpu-mimo2", "bloomery-gpu-minimaxm3", "bloomery-gpu-mistrallarge4", "bloomery-gpu-solar2", "bloomery-gpu-vision"], f"host nightly: left out {left}")
    expect(not nightly_problems(tree, lock_now), f"host nightly: {nightly_problems(tree, lock_now)}")
    ran = {u for u, _ in host_units(tree, lock_now)}
    expect(
        all((f[:-3] in ran) != (f in HOST_CHECK_SKIP) for f in host_check_scripts(tree.root)),
        "host nightly: a check script is neither a unit nor excluded",
    )



def _st_mutations(fails: list[str], side: Side, part: int, parts: int) -> None:
    """The mutated-justfile cases, `part` of `parts` (every parts-th typo, the required-features one in the last part)."""
    expect = _expecting(fails)

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
        for start, end, bad in mutations[part::parts]:
            p = os.path.join(tmp, "justfile")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(text[:start] + bad + text[end:])
            probs = check(side.tree, load_justfile(p))
            expect(any(bad in x for x in probs), f"mutation {bad} not caught: {probs[:2]}")
        # a feature-gated bin built without its feature: cargo refuses it (required-features), and so
        # does the check, before the recipe runs
        m = re.search(r"--features deepseek41 (--release --bin generate_ds41)", text)
        if part != parts - 1:
            pass
        elif m is None:
            fails.append("no mutation anchor for the generate_ds41 build")
        else:
            p = os.path.join(tmp, "justfile")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(text[: m.start()] + "--features gpu " + m.group(1) + text[m.end() :])
            probs = check(side.tree, load_justfile(p))
            expect(any("--bin generate_ds41: needs features ['deepseek41']" in x for x in probs), f"generate_ds41 without deepseek41 not caught: {probs[:2]}")



def _st_selection(fails: list[str], side: Side) -> None:
    expect = _expecting(fails)
    recipes = side.recipes

    # a changed recipe text selects that recipe and no other
    import copy

    b2 = copy.copy(side)
    b2.recipes = dict(recipes)
    r = copy.copy(recipes["gate-sampler"])
    r.text = r.text + " "
    b2.recipes["gate-sampler"] = r
    sels, unmapped, _ = select(["justfile"], side, b2, GATE_PREFIX)
    expect([x.recipe for x in sels] == ["gate-sampler"] and not unmapped, f"recipe-text change selects {[x.recipe for x in sels]}")
    sels, _, _ = select(["crates/gpu-gates/src/bin/gate_p1.rs", "docs/HARDWARE.md"], side, side, GATE_PREFIX)
    expect(sorted(x.recipe for x in sels) == ["gate-gpu-p1"], f"select() on gate_p1.rs and docs/HARDWARE.md gives {[x.recipe for x in sels]}")

    # dep-info parsing
    t, deps, root = parse_depinfo("/r/b/target/release/g: /r/b/crates/a\\ b.rs /r/b/crates/c.rs\n")
    expect(t.endswith("/g") and root == "/r/b" and "/r/b/crates/a b.rs" in deps, "dep-info parse")
    with tempfile.TemporaryDirectory(prefix="recipes-no-ssh-") as tmp:
        saved_path = os.environ.get("PATH")
        try:
            os.environ["PATH"] = tmp
            try:
                fetch_depinfo("no-ssh-test", "/unused")
            except RecipeError as err:
                expect("ssh" in str(err) and "--no-box" in str(err),
                       f"dep-info: missing ssh refusal omits its cause or workaround: {err}")
            except OSError as err:
                fails.append(f"dep-info: missing ssh escapes as an OS error: {err}")
            else:
                fails.append("dep-info: missing ssh accepted")
        finally:
            if saved_path is None:
                os.environ.pop("PATH", None)
            else:
                os.environ["PATH"] = saved_path


# ----------------------------------------------------------------------------------------------
# the class stage of `affected`: the gate classes of the class tables
# ----------------------------------------------------------------------------------------------
#
# The three tables (tools/gate-class.tsv, tools/families.tsv, tools/owners.tsv; read through
# tools/gate_class.py's loader, the one reader) say what a gate-* recipe is: a common gate (family-agnostic), a
# tooling gate (harness, static, fixture writer) or a family gate (one family's own weights or code). The class
# stage is the third stage of `affected`, after select() and narrow(): it keeps every common and every tooling
# pick, and keeps a family pick only when the change can have moved it. It is on by default (`--no-class` turns it off).
#
# A family:X pick is kept when
#   (a) X is a representative family (families.tsv's rep R) and the pick reaches a changed file through the
#       dependency graph — the reach `affected` computes for it;
#   (b) a changed file matches one of X's own globs in families.tsv;
#   (c) the recipe's own text changed, or one of its own cargo target's files changed (a script it names is not
#       one: tools/gate-bound.sh is every gate's, and a change of it would keep every gate);
#   (d) X was added since the last release tag: no v* tag's families.tsv names it, and its own files were not in
#       the tree at the last release tag.
# Otherwise the pick is left out with the reason `class: family gate of X, no file of X changed (rep F)`.
#
# The stage never adds a pick and never drops a common or a tooling one (the design's check 6): both are named
# errors, not a quiet pass.

def class_tables(root: str = ROOT):
    """The three class tables of the tree at ROOT, through tools/gate_class.py's loader. A table the loader
    refuses is a named error."""
    if "recipes" not in sys.modules and "__main__" in sys.modules:
        sys.modules["recipes"] = sys.modules["__main__"]  # the loader imports this module by name
    try:
        import gate_class  # tools/gate_class.py
    except ImportError as err:
        raise RecipeError(f"the class tables' loader is not readable: {err}") from err
    try:
        return gate_class.load(root)
    except gate_class.GateClassError as err:
        raise RecipeError(f"the class tables of {root}: {err}") from err


def f_family_gates(tables) -> dict[str, str]:
    """The gates of the F families (families.tsv rep F), recipe -> family: the gates a landing runs only on their
    family's own change, which the rotation (tools/rotation.py) schedules and the release list leaves to it."""
    return {n: row.family for n, row in tables.gates.items()
            if row.cls == "family" and tables.families[row.family].rep == "F"}


_CLASS_GLOBS: dict[str, re.Pattern] = {}


def class_glob(glob: str) -> re.Pattern:
    """A glob of the class tables as a pattern: `**` crosses a directory, `*` stays in a segment."""
    p = _CLASS_GLOBS.get(glob)
    if p is None:
        p = glob_regex(glob)
        _CLASS_GLOBS[glob] = p
    return p


def glob_prefix(glob: str) -> str:
    """The fixed path prefix of a glob: the part before its first wildcard."""
    cut = len(glob)
    for i, ch in enumerate(glob):
        if ch in "*?{":
            cut = i
            break
    return glob[:cut]


def class_reach(n: str, changed: list[str], a: Side | None, b: Side) -> tuple[list[str], list[str], bool]:
    """(the changed files the recipe reaches through the dependency graph, the changed files that are its own
    cargo target's, whether its own text changed). The reach is the one `affected` computes: the recipe's own
    inputs (its targets' files and the workspace libs they link)."""
    reached: list[str] = []
    own: list[str] = []
    text = False
    for f in changed:
        if f == "justfile":
            old = a.recipes.get(n) if a is not None else None
            if n in b.recipes and (old is None or old.text != b.recipes[n].text):
                text = True
            continue
        side = b if b.tree.exists(f) or a is None else a
        if n not in side.recipes:
            continue
        why = side.graph.inputs(n).match(f)
        if why is None:
            continue
        reached.append(f)
        if _own_target_file(why):
            own.append(f)
    return reached, own, text


# ---- the release tags: the families a release added ----


def release_tags(root: str = ROOT) -> list[str]:
    """The v* tags of the repository, the newest first."""
    out = git("for-each-ref", "--sort=-creatordate", "--format=%(refname:short)", "refs/tags", cwd=root).split()
    return [t for t in out if t.startswith("v")]


def tag_text(rev: str, path: str, root: str = ROOT) -> str | None:
    """The text of a file at a revision, or None when that revision has no such file."""
    proc = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=root, capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    return proc.stdout


def tag_families(rev: str, root: str = ROOT) -> set[str]:
    """The families the revision's tools/families.tsv names; a revision without the file names none."""
    text = tag_text(rev, "tools/families.tsv", root)
    if text is None:
        return set()
    out = set()
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        out.add(line.split("\t")[0].strip())
    return out


def tag_files(rev: str, root: str = ROOT) -> list[str]:
    """The files of the tree at a revision."""
    return git("ls-tree", "-r", "--name-only", rev, cwd=root).split()


def family_has_file(fam, files: list[str]) -> bool:
    """Whether the family had a file of its own among the tree's files."""
    for g in fam.globs:
        pre = glob_prefix(g)
        pat = class_glob(g)
        for f in files:
            if f.startswith(pre) and pat.match(f):
                return True
    return False


def families_new_since(root: str, tables) -> set[str]:
    """The families added since the last release tag: no v* tag's families.tsv names them, and their own files
    were not in the tree at the last release tag. A repository with no release tag names no new family."""
    tags = release_tags(root)
    if not tags:
        return set()
    named: set[str] = set()
    for t in tags:
        named |= tag_families(t, root)
    files = tag_files(tags[0], root)
    return {name for name, fam in tables.families.items()
            if name not in named and not family_has_file(fam, files)}


# ---- the stage itself ----


@dataclass
class ClassStage:
    kept: dict[str, str]      # recipe -> the rule that kept it
    left_out: dict[str, str]  # recipe -> the reason it was left out


def class_check6(picks: dict[str, str], kept: dict[str, str], left_out: dict[str, str], tables) -> None:
    """The stage's own check: it never adds a pick and never drops a common or a tooling one."""
    added = set(kept) - set(picks)
    if added:
        raise RecipeError(f"the class stage added {len(added)} pick(s): {' '.join(sorted(added)[:6])}")
    for n in left_out:
        row = tables.gates.get(n)
        if row is None or row.cls in ("common", "tooling"):
            cls = row.cls if row is not None else "unclassed"
            raise RecipeError(f"the class stage dropped the {cls} pick {n}")


def class_narrow(picks: dict[str, str], changed: list[str], tables, reach, new_families: set[str]) -> ClassStage:
    """The class stage of `affected`: the picks it keeps and the ones it leaves out.

    `picks` is the recipe -> why map of the stage before (select()'s picks or narrow()'s); `changed` the changed
    files; `tables` the three class tables; `reach` the recipe -> (the changed files it reaches through the
    dependency graph, the changed files that are its own target's, whether its own text changed) reader;
    `new_families` the families added since the last release tag.
    """
    kept: dict[str, str] = {}
    left_out: dict[str, str] = {}
    for n in picks:
        row = tables.gates.get(n)
        if row is None:
            kept[n] = "class: no class row, so a common gate"
            continue
        if row.cls in ("common", "tooling"):
            kept[n] = f"class: {row.cls}"
            continue
        fam = tables.families.get(row.family)
        if fam is None:
            raise RecipeError(f"the class row of {n} names the family {row.family}, which families.tsv does not hold")
        reached, own, text = reach(n)
        hit = next((f for f in changed if any(class_glob(g).match(f) for g in fam.globs)), None)
        if text:
            kept[n] = "class: the recipe's own text changed"
        elif own:
            kept[n] = f"class: the recipe's own target file {own[0]} changed"
        elif hit is not None:
            kept[n] = f"class: the family {row.family}'s own file {hit} changed"
        elif fam.rep == "R" and reached:
            kept[n] = f"class: {row.family} is a representative family and the recipe reaches {reached[0]}"
        elif row.family in new_families:
            kept[n] = f"class: the family {row.family} was added since the last release tag"
        else:
            left_out[n] = f"class: family gate of {row.family}, no file of {row.family} changed (rep {fam.rep})"
    class_check6(picks, kept, left_out, tables)
    return ClassStage({n: kept[n] for n in picks if n in kept}, left_out)


def class_stage(picks: dict[str, str], changed: list[str], tables, reach, new_families: set[str],
                enabled: bool) -> ClassStage:
    """The class stage, unless `--no-class` turned it off: then the picks stand as they are."""
    if not enabled:
        return ClassStage(dict(picks), {})
    return class_narrow(picks, changed, tables, reach, new_families)


def class_picks(changed: list[str], a: Side | None, b: Side, tables, root: str = ROOT) -> ClassStage:
    """The class stage over select()'s picks of a diff: the picks it keeps and the ones it leaves out."""
    sels, _, _ = select(changed, a, b, GATE_PREFIX)
    picks = {s.recipe: s.why for s in sels}
    new_families = families_new_since(root, tables)
    return class_narrow(picks, changed, tables, lambda n: class_reach(n, changed, a, b), new_families)


# ---- the release list ----


def fixture_classes(root: str = ROOT) -> dict[str, str]:
    """The fixture tier's class of every recipe: `tools/gate-batch.sh --tier fixture --classes` (the batch's own
    classifier, never re-implemented). A recipe the classifier does not name is not in the fixture tier."""
    proc = subprocess.run([os.path.join(root, "tools", "gate-batch.sh"), "--tier", "fixture", "--classes"],
                          capture_output=True, text=True, cwd=root)
    if proc.returncode != 0:
        raise RecipeError(f"`tools/gate-batch.sh --tier fixture --classes` failed (exit {proc.returncode}): "
                          f"{proc.stderr.strip()[-300:]}")
    out: dict[str, str] = {}
    for n, line in enumerate(proc.stdout.splitlines(), 1):
        if not line.strip():
            continue
        f = line.split("\t")
        if len(f) < 2:
            raise RecipeError(f"the classifier's line {n} is not `recipe, class, …`: {line[:80]!r}")
        out[f[0]] = f[1]
    return out


def rotation_due(root: str = ROOT) -> list[str]:
    """The rotation's due gates: `tools/rotation.py due`, the names of its list (the comment lines are its own)."""
    proc = subprocess.run([sys.executable, os.path.join(root, "tools", "rotation.py"), "due"],
                          capture_output=True, text=True, cwd=root)
    if proc.returncode != 0:
        raise RecipeError(f"`tools/rotation.py due` failed (exit {proc.returncode}): {proc.stderr.strip()[-300:]}")
    return [ln.strip() for ln in proc.stdout.splitlines() if ln.strip() and not ln.startswith("#")]


def release_union(parts: dict[str, list[str]]) -> tuple[list[str], dict[str, int]]:
    """(the release list: the union of the parts, sorted, no duplicate; the count of each part)."""
    names: set[str] = set()
    counts: dict[str, int] = {}
    for part, items in parts.items():
        counts[part] = len(items)
        names.update(items)
    return sorted(names), counts


def times_rows(path: str) -> tuple[dict[str, list[tuple[str, int]]], dict[str, list[tuple[str, int]]]]:
    """(the times file's fixture-tier rows by recipe name, its plain rows by item): (lane, seconds) each, in file
    order. A row that does not parse is counted, not read."""
    fixture: dict[str, list[tuple[str, int]]] = {}
    plain: dict[str, list[tuple[str, int]]] = {}
    bad = 0
    if not os.path.exists(path):
        return fixture, plain
    with open(path, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            f = ln.rstrip("\n").split("\t")
            if len(f) != 5 or not f[3].isdigit():
                bad += 1
                continue
            row = (f[1], int(f[3]))
            if "@BLOOMERY_TIER=fixture" in f[0]:
                fixture.setdefault(f[0].split("@", 1)[0], []).append(row)
            else:
                plain.setdefault(f[0], []).append(row)
    if bad:
        print(f"release-list: {path}: {bad} malformed row(s) ignored", file=sys.stderr)
    return fixture, plain


def predicted_wall(names: list[str], fixture: dict[str, list[tuple[str, int]]],
                   plain: dict[str, list[tuple[str, int]]], default: int = 45) -> dict[str, int]:
    """The predicted wall of a list: the medians of the fixture tier's rows (the plain rows where a gate has no
    fixture row), the lanes they ran in, and the wall the lanes make: (A+B)/2 + X (the design's model: the two
    lanes run side by side, the solo lane after them)."""
    lanes = {"A": 0, "B": 0, "X": 0}
    total = 0
    for n in names:
        rows = fixture.get(n) or plain.get(n) or []
        if rows:
            last = rows[-5:]
            secs = int(statistics.median([s for _, s in last]) + 0.5)
            lane = last[-1][0]
        else:
            secs, lane = default, "A"
        lanes[lane if lane in lanes else "X"] += secs
        total += secs
    return {"A": lanes["A"], "B": lanes["B"], "X": lanes["X"], "sum": total,
            "wall": int((lanes["A"] + lanes["B"]) / 2 + lanes["X"])}


def release_affected(kept: list[str], tables) -> tuple[list[str], dict[str, str]]:
    """(affected(TAG) without its F-family gates, the F-family gates it dropped as recipe -> family): the F
    families are the rotation's, and `tools/release/publish.sh` refuses the tag while `tools/rotation.py check`
    is red."""
    f = f_family_gates(tables)
    return [n for n in kept if n not in f], {n: f[n] for n in kept if n in f}


def rotation_left(dropped: dict[str, str], ledger: Ledger) -> list[str]:
    """One line per F-family gate left to the rotation: its family and its last green row in the ledger."""
    out = []
    for n, fam in sorted(dropped.items(), key=lambda kv: (kv[1], kv[0])):
        rec = ledger.by_recipe.get(n)
        out.append(f"  {n}  (family {fam}, " + (f"last green {rec.date}" if rec else "no green row in the ledger") + ")")
    return out


def cmd_release_list(args: argparse.Namespace) -> int:
    """The release batch's list: every S0 gate (a common or a tooling gate the fixture tier runs), every gate of
    a representative family, the class-staged affected(BASE) without its F-family gates, the rotation's due gates
    and every CUDA gate on a GGUF input. One recipe per line, sorted, no duplicate; the counts per part on stderr,
    and each F-family gate left to the rotation with its last green date."""
    changed, a_rev, b_rev, _ = changed_files(args.base)
    tmp = tempfile.mkdtemp(prefix="recipes-release-")
    try:
        b = make_side(archive(b_rev, os.path.join(tmp, "b")) if b_rev else ROOT)
        a = make_side(archive(a_rev, os.path.join(tmp, "a"))) if a_rev else None
        tables = class_tables(b.tree.root)
        classes = fixture_classes(b.tree.root)
        s0 = [n for n, row in tables.gates.items()
              if row.cls in ("common", "tooling") and classes.get(n) in ("A", "B", "F", "X")]
        reps = [n for n, row in tables.gates.items()
                if row.family and tables.families[row.family].rep == "R"]
        affected, dropped = release_affected(list(class_picks(changed, a, b, tables).kept), tables)
        due = rotation_due(b.tree.root)
        mac = [n for n, _why in mac_gguf_cuda(b.recipes, b.tree.root)]
        mac_f = [n for n in mac if n in f_family_gates(tables)]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    names, counts = release_union({"S0": s0, "R families": reps, "affected": affected, "rotation due": due, "mac GGUF": mac})
    for n in names:
        print(n)
    print("release-list: " + ", ".join(f"{k} {v}" for k, v in counts.items()) + f", {len(names)} recipe(s) in all", file=sys.stderr)
    print(f"release-list: {len(dropped)} F-family gate(s) of affected are left to the rotation (tools/rotation.py check holds them before the tag)"
          + (f"; the mac GGUF part still holds {len(mac_f)} F-family gate(s)" if mac_f else "") + (":" if dropped else ""), file=sys.stderr)
    for line in rotation_left(dropped, Ledger(args.ledger)):
        print(line, file=sys.stderr)
    if args.wall:
        fixture, plain = times_rows(args.times)
        w = predicted_wall(names, fixture, plain)
        print(f"release-list: the predicted wall {w['wall']} s of the medians (lanes A {w['A']} B {w['B']} X {w['X']}, "
              f"the medians sum {w['sum']} s), × {args.lock_wait:g} = {int(w['wall'] * args.lock_wait)} s", file=sys.stderr)
    return 0


def class_stage_self_test(fails: list[str]) -> None:
    """The class stage on a synthetic tree and tables: the four rules that keep a family gate, the reason a family
    gate is left out, the check that the stage never adds a pick and never drops a common or a tooling one, the
    stage on by default and off with `--no-class`, and the release list's union without the F families."""
    expect = _expecting(fails)
    t0 = time.monotonic()

    def refused(fn, why: str) -> bool:
        try:
            fn()
        except RecipeError as err:
            return why in str(err)
        return False

    import tempfile

    with tempfile.TemporaryDirectory(prefix="recipes-class-") as tmp:
        root = os.path.join(tmp, "tree")

        def write(rel: str, text: str) -> None:
            p = os.path.join(root, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(text)

        write("tools/families.tsv", "# family\trep\tglobs\trotation_days\tfixture\n"
              "fam-rep\tR\tcrates/fam-rep/**\t7\tno\n"
              "fam-fix\tF\tcrates/fam-fix/**\t7\tno\n"
              "fam-new\tF\tcrates/fam-new/**\t7\tno\n")
        write("tools/owners.tsv", "# owner\tkind\tglobs\nown-a\tkernel\tcrates/fam-rep/a.rs\n")
        write("tools/gate-class.tsv", "# recipe\tclass\tfamily\ttopic\towners\n"
              "gate-rep\tfamily\tfam-rep\tkernel\town-a\n"
              "gate-fix\tfamily\tfam-fix\tkernel\t-\n"
              "gate-new\tfamily\tfam-new\tkernel\t-\n"
              "gate-com\tcommon\t\thost\t-\n"
              "gate-tool\ttooling\t\thost\t-\n")
        write("crates/fam-rep/a.rs", "// a\n")
        write("crates/fam-fix/b.rs", "// b\n")
        write("crates/fam-new/c.rs", "// c\n")
        tables = class_tables(root)
        expect(sorted(tables.gates) == ["gate-com", "gate-fix", "gate-new", "gate-rep", "gate-tool"],
               f"class: the synthetic table loads: {sorted(tables.gates)}")

        # the synthetic reach: the changed files a recipe reaches, the changed files that are its own target's,
        # whether its own text changed
        reach: dict[str, tuple[list[str], list[str], bool]] = {
            "gate-rep": (["crates/other/c.rs"], [], False),
            "gate-fix": (["crates/other/c.rs"], [], False),
            "gate-new": (["crates/other/c.rs"], [], False),
            "gate-com": (["crates/other/c.rs"], [], False),
            "gate-tool": (["crates/other/c.rs"], [], False),
        }
        picks = {n: "why" for n in reach}
        changed = ["crates/other/c.rs"]

        def stage(new: set[str] | None = None, ch: list[str] | None = None) -> ClassStage:
            return class_narrow(picks, ch if ch is not None else changed, tables, lambda n: reach[n],
                                 new if new is not None else set())

        # (a) a representative family whose pick reaches a changed file
        st = stage()
        expect("gate-rep" in st.kept and "gate-rep" not in st.left_out,
               f"class: a representative family is kept: {st.kept.get('gate-rep')}")
        # (b) a changed file of the family
        st = stage(ch=["crates/fam-fix/b.rs"])
        expect("gate-fix" in st.kept, f"class: the family's own file keeps its gate: {st.kept.get('gate-fix')}")
        # (c) the recipe's own text, and one of its own target files
        reach["gate-fix"] = (["crates/other/c.rs"], [], True)
        expect("gate-fix" in stage().kept, "class: the recipe's own text keeps its gate")
        reach["gate-fix"] = (["crates/other/c.rs"], ["crates/fam-fix/b.rs"], False)
        expect("gate-fix" in stage().kept, "class: the recipe's own target file keeps its gate")
        reach["gate-fix"] = (["crates/other/c.rs"], [], False)
        # (d) a family added since the last release tag
        expect("gate-fix" in stage({"fam-fix"}).kept, "class: a family added since the last release tag keeps its gate")
        # otherwise the family gate is left out, with the reason
        st = stage()
        want = "class: family gate of fam-fix, no file of fam-fix changed (rep F)"
        expect(st.left_out.get("gate-fix") == want, f"class: the reason a family gate is left out: {st.left_out.get('gate-fix')!r}")
        expect("gate-com" in st.kept and "gate-tool" in st.kept, "class: a common and a tooling pick are kept")
        # a pick with no class row is a common gate
        st = class_narrow({"gate-none": "why"}, changed, tables, lambda n: ([], [], False), set())
        expect("gate-none" in st.kept and not st.left_out, f"class: a pick with no row is kept: {st.kept}")
        # check 6: the stage never adds a pick and never drops a common or a tooling one
        expect(refused(lambda: class_check6(picks, {**picks, "gate-extra": "why"}, {}, tables), "added 1 pick"),
               "class: check 6 refuses a pick the stage added")
        expect(refused(lambda: class_check6(picks, {}, {"gate-com": "why"}, tables), "dropped the common pick"),
               "class: check 6 refuses a common pick the stage dropped")
        # the stage is on by default (no flag drops the F gate whose family did not change) and `--no-class` keeps it
        parser = build_parser()
        on = parser.parse_args(["affected"]).class_stage
        off = parser.parse_args(["affected", "--no-class"]).class_stage
        st = class_stage(picks, changed, tables, lambda n: reach[n], set(), on)
        expect(on is True and "gate-fix" in st.left_out and "gate-fix" not in st.kept,
               f"class: with no flag the family gate whose family did not change is left out: {on} {st.left_out}")
        st = class_stage(picks, changed, tables, lambda n: reach[n], set(), off)
        expect(off is False and st.kept == picks and not st.left_out, f"class: --no-class keeps every pick: {off} {st.kept}")
        # the tree's own tables: every common gate is kept, and a family gate of an F family is left out
        real = class_tables(ROOT)
        st = class_narrow({n: "why" for n in real.gates}, ["crates/gpu/src/lib.rs"], real,
                          lambda n: (["crates/gpu/src/lib.rs"], [], False), set())
        expect(all(n in st.kept for n, r in real.gates.items() if r.cls in ("common", "tooling")),
               "class: every common and tooling gate of the tree's own table is kept")
        expect(len(st.left_out) > 0, "class: the tree's own table leaves a family gate out")

        # the release list: the union of the parts, sorted, without a duplicate
        names, counts = release_union({"S0": ["b", "a"], "R families": ["a", "c"], "rotation due": ["c"]})
        expect(names == ["a", "b", "c"], f"release: the union is sorted and holds no duplicate: {names}")
        expect(counts == {"S0": 2, "R families": 2, "rotation due": 1}, f"release: the count of each part: {counts}")

        # the release list leaves the F families to the rotation, each gate named with its last green row
        expect(f_family_gates(tables) == {"gate-fix": "fam-fix", "gate-new": "fam-new"},
               f"release: the F-family gates of the tables (not the R family's, a common's or a tooling's): {f_family_gates(tables)}")
        affected, dropped = release_affected(["gate-com", "gate-fix", "gate-rep", "gate-tool"], tables)
        expect(affected == ["gate-com", "gate-rep", "gate-tool"] and dropped == {"gate-fix": "fam-fix"},
               f"release: affected loses its F-family gate and keeps the common, R-family and tooling ones: {affected} {dropped}")
        led = os.path.join(tmp, "ledger.tsv")
        with open(led, "w", encoding="utf-8") as fh:
            fh.write("\t".join(["a" * 64, "gate-fix", "gate-fix", "abc123456789", "2026-10-20T10:00:00+0900", "/x"]) + "\n")
        lines = rotation_left({"gate-new": "fam-new", "gate-fix": "fam-fix"}, Ledger(led))
        expect(lines == ["  gate-fix  (family fam-fix, last green 2026-10-20T10:00:00+0900)",
                         "  gate-new  (family fam-new, no green row in the ledger)"],
               f"release: each dropped F gate is named with its last green date, or its lack: {lines}")

    print(f"class stage self-test: {'ok' if not fails else f'{len(fails)} failure(s)'} ({time.monotonic() - t0:.1f} s)")


def self_test(serial: bool = False) -> int:
    side = make_side(ROOT)
    gates = [n for n in side.recipes if n.startswith(GATE_PREFIX)]
    # The children inherit the real tree's input and reach memos copy-on-write.
    for name in gates:
        side.graph.inputs(name)
        reach_of(side.tree).gate_files(side.graph, name)

    def small(f: list[str]) -> None:
        expect = _expecting(f)
        scan_set_self_test(expect, side)  # the scan set
        ships_self_test(expect)  # the shipped closure and the release feature list
        manifest_self_test(expect)  # a package manifest read by table
        root_manifest_self_test(expect)  # the root Cargo.toml read by table
        lock_view_self_test(expect)  # [target.<key>] tables and Cargo.lock as the box's build reads them
        range_base_self_test(expect)  # A..B read against the merge base
        module_gate_self_test(expect)  # a module gated by a cargo feature
        memo_self_test(expect)  # the process-wide memo of walks and script closures
        archive_self_test(expect)  # a base tree read through git archive, whatever tar does at the end marker
        static_self_test(expect)  # the static tier compiles every feature-gated bin
        ledger_self_test(expect)  # the green ledger's key, but for the key itself
        static_key_self_test(expect)  # the static tier's job keys
        manifest_rows_self_test(expect)
        ref_paths_self_test(expect)
        box_tier_self_test(expect)
        oxide_rev_self_test(expect)
        expand_self_test(expect)
        opens_self_test(expect)  # opens: the targets that reach the fixture fence
        comment_only_self_test(expect)  # a Rust file whose diff is comments or layout selects no gate

    def narrow(f: list[str]) -> None:
        expect = _expecting(f)
        narrow_self_test(expect, side)  # affected --narrow
        lever_rows_self_test(expect)  # the lever registry read by row
        variant_items_self_test(expect)

    # Independent sections, each on the same parsed real tree: they run side by side, the wall is the longest one.
    jobs: list[tuple[str, Callable[[list[str]], None]]] = [
        ("basics", lambda f: _st_basics(f, side)),
        ("real tree", lambda f: _st_real_tree(f, side)),
        ("pure crates", lambda f: _st_pure_crates(f, side)),
        ("host crates", lambda f: _st_host_crates(f, side)),
        ("mutations 1/2", lambda f: _st_mutations(f, side, 0, 2)),
        ("mutations 2/2", lambda f: _st_mutations(f, side, 1, 2)),
        ("selection", lambda f: _st_selection(f, side)),
        ("orphan-tests", lambda f: orphan_self_test(_expecting(f), side)),
        ("narrow", narrow),
        ("narrow table", lambda f: narrow_table_self_test(_expecting(f), side)),
        ("dispatch", lambda f: dispatch_self_test(_expecting(f), side)),
        ("class stage", lambda f: class_stage_self_test(f)),
        ("small tools", small),
        ("key detectors", lambda f: key_detectors_self_test(_expecting(f), side)),
        *[(f"key {section}", lambda f, section=section: key_self_test(_expecting(f), side, section))
          for section in ("inputs", "globals", "fixtures", "variants", "scope")],
        ("key reach", lambda f: key_reach_self_test(_expecting(f), side)),
    ]
    fails, times = _forked(jobs, serial)
    for f in fails:
        print(f"self-test FAIL: {f}", file=sys.stderr)
    print("self-test: slowest sections: " + ", ".join(f"{n} {t:.0f} s" for n, t in sorted(times, key=lambda x: -x[1])[:3]))
    print("self-test: key sections: " + ", ".join(f"{n} {t:.0f} s" for n, t in times if n.startswith("key")))
    print(f"self-test: {'FAIL' if fails else 'ok'} ({len(gates)} gate recipes, {len(fails)} failures)")
    return 1 if fails else 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--serial", action="store_true", help="with --self-test: run its sections one after another in this process (a debugger, a traceback in order)")
    sub = ap.add_subparsers(dest="cmd")
    c = sub.add_parser("check")
    c.add_argument("--justfile")
    c.add_argument("--fill", action="store_true", help=f"removed: {GATE_PATHS} derives its gates; the flag is a named error")
    sub.add_parser("rows", help=f"{GATE_PATHS} as narrow() reads it: each `{DERIVE}` row's gates written out, line for line")
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
    a.add_argument("--no-class", dest="class_stage", action="store_false", help="turn the class stage off: keep every gate the graph reaches (by default a family gate stays only when the change can have moved it, by the class tables)")
    a.add_argument("--no-comment-only", dest="comment_only", action="store_false", help="turn the comment-only rule off: a Rust file whose diff is comments or layout selects the gates that read it like any changed file (by default it selects none)")
    ss = sub.add_parser("scan-set", help="the bins whose ptx-scans cover every kernel carrier, one line each: bin<TAB>features<TAB>the carriers it covers (tools/narrow-scan.sh scans them)")
    ss.add_argument("rev", nargs="?", help="the commit to read (a git archive of it); default the working tree")
    ss.add_argument("--ship", action="store_true", help=f"the shipped bins at the release features (tools/release/build.sh) are the first picks; {SCAN_SHIP_ENV}=1 does the same")
    sh = sub.add_parser("ships", help="the FILEs inside the shipped closure, one `file<TAB>why` line each; exit 1 when any is (tools/release/late-commit.sh)")
    sh.add_argument("files", nargs="+", metavar="FILE")
    sh.add_argument("--rev", help="read the closure at this commit (a git archive of it); default the working tree")
    sh.add_argument("--base", help="a file in this commit's closure counts too (a deleted or moved-away file)")
    sc = sub.add_parser("ship-scan", help="the ptx-scan pairs of the shipped bins: one verdict line each; exit 0 only when every shipped bin has a pair and every pair is identical (tools/release/late-commit.sh)")
    sc.add_argument("logs", nargs="+", metavar="LOG", help="BASE_LOG HEAD_LOG, once per shipped bin")
    sc.add_argument("--rev", help="read the shipped bins at this commit; default the working tree")
    pcm = sub.add_parser("ptx-compare", help="two full ptx-scan logs of one bin as one table (entries, added, removed, changed); exit 0 nothing existing moved, 3 an entry changed, 4 removed, 5 both (tools/ptx-compare.sh)")
    pcm.add_argument("base_log", metavar="BASE_LOG")
    pcm.add_argument("head_log", metavar="HEAD_LOG")
    pcm.add_argument("--base-label", default="base")
    pcm.add_argument("--head-label", default="head")
    sb = sub.add_parser("ship-bins", help="the shipped bins and the release features: bin<TAB>features (tools/release/build.sh)")
    sb.add_argument("--rev", help="read build.sh at this commit; default the working tree")
    w = sub.add_parser("why", help="the gate-* recipes whose key holds each FILE and the edge that brought it in; --recipe R: R's shortest dependency chain to the file's crate, per cargo target")
    w.add_argument("files", nargs="+")
    w.add_argument("--all-recipes", action="store_true")
    w.add_argument("--recipe", action="append", help="only this recipe (repeatable), with its chains")
    gp = sub.add_parser("gaps", help="per workspace crate: the gate-* and weekly-* recipes whose key holds a file of it, those whose targets reach code of it, the gap (exit 1 on a gap of a KEY_REACH_PINS crate)")
    gp.add_argument("crates", nargs="*", metavar="CRATE")
    gp.add_argument("--rev", help="read the tree at this commit (a git archive of it); default the working tree")
    gp.add_argument("--list", action="store_true", help="name the recipes in each gap")
    rh = sub.add_parser("reach", help="the gate-* recipes whose targets can run code of each FILE through a dependency (the derivation "
                        f"{GATE_PATHS}'s rows are held to), each with one chain")
    rh.add_argument("files", nargs="*")
    rh.add_argument("--brief", action="store_true", help="the gates' names without their chains")
    rh.add_argument("--row", action="store_true", help="last, a `derive` row for the files as the table's first two columns")
    rh.add_argument("--mac-landing", action="store_true",
                    help="print the landing list for a change under the mac crates: CUDA gates on a GGUF input are deferred, each with its reason")
    rh.add_argument("--batch", help="with --mac-landing: a file of LABEL<TAB>FILE... lines, one set per line, one tree for all of them")
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
    sk = sub.add_parser("static-key", help="each job's input key of a tools/static-sched.py JOBS file: name<TAB>key<TAB>note (`-` for a job never skipped); --files NAME: the files one job's key reads, one a line")
    sk.add_argument("jobs", metavar="JOBS")
    sk.add_argument("--files", metavar="NAME")
    b = sub.add_parser("box-manifest", help="on the box, through tools/box.sh: the key's box part")
    b.add_argument("--lease", action="append", help="a timing lease lock; held, the manifest refuses (exit 75)")
    b.add_argument("--cache", default="~/.cache/bloomery/sha256-cache.tsv", help="the stat-keyed sha256 cache")
    b.add_argument("--workers", type=int, default=8)
    ex = sub.add_parser("expand", help="every recipe as just runs it, each top-level variable replaced by its value in this environment: the text two justfiles must share for a refactor of the recipes to move no command")
    ex.add_argument("--justfile")
    op = sub.add_parser("opens", help="per RECIPE, whether a cargo target it builds or runs reaches the fixture fence (Tier::check_file): opens<TAB>recipe<TAB>pkg:kind:target<TAB>path:line<TAB>called, or none<TAB>recipe (tools/gate-batch.sh's real-file deferral)")
    op.add_argument("items", nargs="*", metavar="RECIPE")
    op.add_argument("--explain", action="store_true", help="the fence, the openers and each recipe's targets, on stderr")
    rl = sub.add_parser("release-list", help="the release batch's list: every S0 gate, every gate of a representative family, affected(BASE) without the F families' gates (the rotation holds those) and the rotation's due gates")
    rl.add_argument("base", nargs="?", default="main", help="the release tag (or TAG..HEAD): the diff the list covers")
    rl.add_argument("--wall", action="store_true", help="the predicted wall of the list on stderr (the fixture tier's medians)")
    rl.add_argument("--lock-wait", type=float, default=1.57, help="the factor the predicted wall carries over the medians (the lock waits)")
    rl.add_argument("--times", default=os.environ.get("BLOOMERY_GATE_TIMES", os.path.expanduser("~/.cache/bloomery/gate-times.tsv")), help="the times file the medians come from")
    rl.add_argument("--ledger", default=os.environ.get("BLOOMERY_GATE_LEDGER", os.path.expanduser("~/.cache/bloomery/gate-ledger.tsv")), help="the green ledger the last green dates of the F-family gates left to the rotation come from")
    return ap


def main(argv: list[str]) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    try:
        if args.self_test:
            return self_test(args.serial)
        if args.cmd == "check":
            return cmd_check(args)
        if args.cmd == "targets":
            return cmd_targets(args)
        if args.cmd == "affected":
            return cmd_affected(args)
        if args.cmd == "scan-set":
            return cmd_scan_set(args)
        if args.cmd == "ships":
            return cmd_ships(args)
        if args.cmd == "ship-scan":
            return cmd_ship_scan(args)
        if args.cmd == "ship-bins":
            return cmd_ship_bins(args)
        if args.cmd == "ptx-compare":
            return cmd_ptx_compare(args)
        if args.cmd == "why":
            return cmd_why(args)
        if args.cmd == "reach":
            return cmd_reach(args)
        if args.cmd == "gaps":
            return cmd_gaps(args)
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
        if args.cmd == "static-key":
            return cmd_static_key(args)
        if args.cmd == "box-manifest":
            return cmd_box_manifest(args)
        if args.cmd == "opens":
            return cmd_opens(args)
        if args.cmd == "expand":
            return cmd_expand(args)
        if args.cmd == "rows":
            return cmd_rows(args)
        if args.cmd == "release-list":
            return cmd_release_list(args)
        ap.print_help()
        return 64
    except RecipeError as err:
        print(f"recipes.py: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
