#!/usr/bin/env bash
# The gate-* recipes a change touches — layer 1 of the affected-gates tool. Runs on the Mac, builds
# nothing and runs no gate: it prints the list the lead pastes into a batch.
#   tools/affected-gates.sh              # the working tree (uncommitted and untracked too) vs main
#   tools/affected-gates.sh <base>       # the working tree vs the merge base of <base> and HEAD
#   tools/affected-gates.sh A..B         # two commits, each read from its own `git archive`
#   tools/affected-gates.sh … --no-box   # skip the box's dep-info (read-only ssh otherwise)
#   tools/affected-gates.sh … --narrow --scan BASE_LOG NEW_LOG [--scan …]
#                                        # narrowed by ptx-scan pairs (base, change) and tools/gate-paths.tsv:
#                                        # identical or added pairs keep the gates whose own target changed and
#                                        # the rows the changed files match; a moved pair, no pair or a file no
#                                        # row maps prints the full list with the reason
#   just narrow BASE [HEAD] [--out DIR]  # the same with the scans taken for you (tools/narrow-scan.sh): the bins whose
#                                        # scans cover every kernel carrier (tools/recipes.py scan-set), built and scanned
#                                        # --no-jit at both commits in two persistent box dirs, then this --narrow run
#
# Output: one line per selected recipe with the file that selected it and the chain (`bin gate_x <- lib
# bloomery-gpu (bloomery-gpu-gates -> bloomery-gpu)`), a `recipes:` line to paste, the `always:` static
# checks, the changed files no gate reads under `unmapped:` (a source file some target reads but no
# gate-* recipe builds is named there — a gate binary without a recipe shows up), and notes: how old
# each selected bin's dep-info is, and which selected bins have none. A package manifest is read by table
# (its shared tables and the recipe's own target entries: a new [[bin]] selects the recipes that build it,
# not every reader of the file); the `manifests:` block prints, per changed manifest, what changed by table
# and the recipes each change selects, and the recipes that read it whole.
#
# The mapping (tools/recipes.py, shared with tools/check-recipes.sh) is the crate graph, feature-aware,
# plus the module tree of each target, the scripts the recipe names, box.sh's profile files and the
# cargo globals. Honest expectation: a change in crates/gpu or crates/model selects every GPU gate —
# the crate graph's true answer. The saving is on leaf changes (serve, tokenizer, sampler, one gate
# binary, a runner, docs); the larger value is the list itself: no forgotten gate, and a gate left out
# is a printed record, not a judgment. It cannot see $BLOOMERY_DATA, the ik trees, or the card.
set -euo pipefail
exec python3 "$(dirname "$0")/recipes.py" affected "$@"
