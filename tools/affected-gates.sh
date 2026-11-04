#!/usr/bin/env bash
# The gate-* recipes a change touches — layer 1 of the affected-gates tool. Runs on the Mac, builds
# nothing and runs no gate: it prints the list the lead pastes into a batch.
#   tools/affected-gates.sh              # the working tree (uncommitted and untracked too) vs main
#   tools/affected-gates.sh <base>       # the working tree vs the merge base of <base> and HEAD
#   tools/affected-gates.sh A..B         # B against merge-base(A, B) (A itself when it is an ancestor of B), each read from its own
#                                        # `git archive`; the `affected:` line says `<mb> (merge-base of A and B)..B` when it moved
#   tools/affected-gates.sh … --no-box   # skip the box's dep-info (read-only ssh otherwise)
#   tools/affected-gates.sh … --no-class # without the class stage, which is on: a gate of an F family (tools/families.tsv) stays only when
#                                        # its family's files, its recipe or its own targets changed, each one left out named with its reason
#   tools/affected-gates.sh … --no-comment-only # without the comment-only rule, which is on: a Rust file of a workspace crate whose diff is comments
#                                        # or layout only (equal tokens) selects no gate; the target of an include macro, a doc comment in a crate
#                                        # whose doctests a gate runs and the lever registry stay selected, each named under `notes:`
#   tools/affected-gates.sh … --no-owed  # without the `real-tier owed:` block, which is on: the selected gates a fixture-tier landing
#                                        # defers to the real tier (real-only, real-file, no-fixture, needs-env, by tools/gate-batch.sh's own
#                                        # classifier, which runs beside the selection), each with its kind; tools/rotation.py due lists
#                                        # them first once their key has moved. `--classes FILE` reads a saved `gate-batch.sh --tier fixture
#                                        # --classes` instead of running it
#   tools/affected-gates.sh … --narrow --scan BASE_LOG NEW_LOG [--scan …]
#                                        # narrowed by ptx-scan pairs (base, change) and tools/gate-paths.tsv:
#                                        # identical or added pairs keep the gates whose own target changed, the
#                                        # rows the changed files match and, for a Rust file no row maps, the gates
#                                        # its derivation reaches; a moved pair, no pair or a file the derivation
#                                        # cannot read prints the full list with the reason
#   just narrow BASE [HEAD | --worktree] [--out DIR]
#                                        # the same with the scans taken for you (tools/narrow-scan.sh): the bins whose
#                                        # scans cover every kernel carrier (tools/recipes.py scan-set), built and scanned
#                                        # --no-jit at both commits in two persistent box dirs, then this --narrow run;
#                                        # --worktree: the head side is this tree's content (no commit: a round)
#
# Output: one line per selected recipe with the file that selected it and the chain (`bin gate_x <- lib
# bloomery-gpu (bloomery-gpu-gates -> bloomery-gpu)`), a `recipes:` line to paste, the `always:` static
# checks, the changed files no gate reads under `unmapped:` (a source file some target reads but no
# gate-* recipe builds is named there — a gate binary without a recipe shows up), and notes: how old
# each selected bin's dep-info is, and which selected bins have none. A package manifest is read by table
# (its shared tables and the recipe's own target entries: a new [[bin]] selects the recipes that build it,
# not every reader of the file), and so is the root Cargo.toml of a virtual workspace: [workspace] members and
# exclude select the recipes that read a package that entered or left the workspace (`exclude = [.., "fuzz"]`
# where fuzz held no member selects nothing), default-members the recipes with a cargo line naming no -p and
# no --workspace, a moved workspace.dependencies entry the recipes whose closure holds a package that inherits it
# (`name = { workspace = true }`; an entry no package inherits stays whole), and workspace.lints (like a package's
# [lints]) selects no recipe: the static tier (`just lint`, `just check`) holds it, and the `affected:` line says so. Every other
# table (profile, patch, workspace.package, ...) and the toolchain files select every cargo recipe. Cargo.lock is read
# by package: the packages its diff moves (added, removed, another version, another dependency list, a workspace
# member's own entry) select the recipes whose build on their host reaches one, walking the lock from the packages the
# recipe links, so a package every target reaches still selects every recipe; a lock the reader cannot take apart
# stays whole, with a note. The `manifests:` block prints, per changed manifest, what changed by table and the
# recipes each change selects, and the recipes that read it whole.
# The lever registry (crates/levers/src/registry.rs) is read by row when every edit to it is a row (an added, changed or removed
# `LeverSpec`/`runner`/`path` element, or the `&str` const that names one): it selects the recipe that tests the registry and the
# recipes whose inputs hold a file outside crates/levers, or whose own command lines, that name a moved row (its variable or const). Code beside the rows,
# a row the reader cannot name, rows in another order, or a moved lib.rs keeps the crate graph's selection, and the `affected:` line
# says why. The `lever rows:` block prints each moved row and the recipes that name it.
#
# The mapping (tools/recipes.py, shared with tools/check-recipes.sh) is the crate graph, feature-aware,
# plus the module tree of each target, the scripts the recipe names, box.sh's profile files and the
# cargo globals. Honest expectation: a change in crates/gpu or crates/model selects every GPU gate —
# the crate graph's true answer. The saving is on leaf changes (serve, tokenizer, sampler, one gate
# binary, a runner, docs); the larger value is the list itself: no forgotten gate, and a gate left out
# is a printed record, not a judgment. It cannot see $BLOOMERY_DATA, the ik trees, or the card.
# Ledger keys deliberately differ: `recipes.py key` may exclude proven-unreached dependency sources
# while retaining own-target, build/proc-macro, script and opaque bounds. This affected list remains the
# broad compile/selection contract; narrowing a key does not remove a recipe from the affected list.
set -euo pipefail
exec python3 "$(dirname "$0")/recipes.py" affected "$@"
