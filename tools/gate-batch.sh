#!/usr/bin/env bash
# The lead's gate batch runner, two lanes. Runs on the Mac; every item is `just <recipe> [ARGS]`, so
# the box is reached only through the recipes (tools/box.sh) and each item keeps the bound and the
# exit code its recipe's runner owns (tools/gate.sh, tools/gpu-gate.sh, 900 s). This script adds no
# second bound.
#   tools/gate-batch.sh [--out DIR] [--smoke | --weekly | --list FILE | ITEM…] [--dry-run] [--lanes 1|2] [--tier real|fixture] [--ledger [--trust-rounds] [--no-gpu-hold] | --round-ledger] [--rerun]
#   tools/gate-batch.sh --classes     every recipe's class, the classifier below, and nothing else
#   tools/gate-batch.sh --self-test   the placement rules on a fixture justfile and the disk floor
#                                      against a fake df (check-recipes runs it)
#
# Items. `NAME[@K=V[,K=V…]][:ARGS]` — NAME a recipe in `just --dump`; `@K=V,…` added to
# BLOOMERY_BOX_ENV for that item (values without spaces, commas or colons); `:ARGS` passed to the
# recipe, word-split (shell quoting allowed). Env comes before ARGS, e.g.
#   gate-gpu-ds41-prefill@BLOOMERY_PREFILL_GROUP=1:--cases 512 --no-split --no-extra
# --list FILE: one item per line (blank lines and `#` lines skipped), or the raw output of `just
# affected …` (first line `affected:`, or just's echoed `./tools/affected-gates.sh …` line and then it): then
# only lines starting with two spaces and `gate-` or `weekly-` (a weekly recipe a trigger named) count, the
# first word is the recipe, everything else is ignored — the `always:` checks are not taken from it.
# --smoke: the fixed smoke list (SMOKE below; docs/gates-plan.md 3.1). --weekly: every `weekly-*` recipe in
# `just --dump`, in its order (`just weekly`; a justfile with none is a named error). An unknown recipe, a malformed
# item, ARGS given to a recipe with no parameters, or an empty list is a named error before anything
# runs.
#
# Lanes (--lanes 2, the default), read from each recipe's text and attributes in `just --dump` (and
# its dependencies'), and from the scripts that text runs, followed transitively (walk() below):
#   A  fixed, the 3090: a tools/gpu-gate.sh call, in the recipe or a script it runs, without
#      BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} (V4.1 `--place gate` gates; gate-gpu-hybrid, whose host tier spins
#      the worker pool on the cores a V4.1 gate's pool is pinned to), or device code run with no gate lock at all
#      (`tools/gate.sh --oxide`, `cargo oxide test|run`, a `target/release/` binary of a recipe that
#      runs `cargo oxide`: it lands on the box env's 3090 pin). A gpu-gate.sh recipe here runs with
#      BLOOMERY_GATE_CARD=3090 in BLOOMERY_BOX_ENV, so a mixed recipe's `any` calls stay on the 3090.
#   A|B  balanced: a recipe whose every gpu-gate.sh call carries the `any` form (gate-ptx-spill, whose
#      scan JITs through one in tools/ptx-scan.sh), or that runs no device code (CPU gates, check, lint,
#      fmt-check, check-*). It goes to the lane expected
#      to finish first: the fixed items' expected times are summed first, then the balanced items go
#      longest first (ties in list order), each to the lane with the smaller running sum (lane B on a
#      tie). Its card is its lane's, never `any`: BLOOMERY_GATE_CARD=3090 in lane A, a6000 in lane B
#      (where it never queues on the 3090 lock); a recipe with no device code gets none.
#   X  alone, after both lanes end: a recipe (or a dependency) that carries the just attribute
#      `[group('solo')]`, whose gate pins a host-memory count (page faults, page-cache residency) that
#      another lane's model loads move; and a recipe that uses box.sh's own card pick
#      BLOOMERY_CARD=both|a6000 (stage-gpu-load-v41*, which carry the attribute too): its gpu-gate.sh call
#      takes the gate lock of every card box.sh put in view (both cards: both locks), so it holds them
#      against other tracks' gates, and it still runs alone here. A solo gpu-gate.sh item keeps a card (3090 for a 3090 gate,
#      a6000 for an `any` one); a BLOOMERY_CARD recipe gets none (box.sh picks). Alone means within
#      this batch: another track's box jobs still run. The attribute is read from `just --dump --dump-format json`, where a recipe's
#      "attributes" list holds {"group": "solo"}; a `[group('…')]` value anywhere in the justfile other
#      than solo, solo-real, v41-load and host is a named error — a group is a scheduling class here, and a typo must
#      not drop a gate out of its class.
#   solo-real: `[group('solo')]` in the real tier, balanced in the fixture tier (--tier below). A gate whose reason for running
#      alone is its host set — the whole model loaded beside other lanes, and the box-wide V4.1 load lock — and not its two
#      cards or a host-memory count it pins. Its gpu-gate.sh calls all take the `any` card, which is what the fixture tier
#      balances it on; it carries no `BLOOMERY_CARD=both` pick (two cards keep it alone in both tiers: use solo), no `solo` or
#      `host`, and no `[group('v41-load')]` pin in the fixture tier (below) — each a named error where it can be told from the
#      recipe's text. A box.sh pick of the A6000 alone (`BLOOMERY_CARD=a6000`, one card, a gpu-gate.sh call that takes the pick)
#      is allowed: the fixture tier fixes the recipe to lane B, the A6000's lane, first in the lane's order, and no lane steals
#      it; the real tier keeps it alone, as any card pick does. A solo-real recipe's arm handed through another recipe is
#      refused in both tiers, as a solo one's is.
#   host: a recipe that carries `[group('host')]` — device code built and run by `cargo oxide test` on a
#      device-linked crate whose tests open no card (tools/recipes.py check holds that: the code of every
#      test it runs names no card opener) — is balanced like a recipe with no device code, with no card
#      forced. With solo, v41-load, a tools/gpu-gate.sh call or a box.sh card pick it is a named error.
#   v41-load: a recipe (or a dependency) that carries `[group('v41-load')]` — one whose binary loads the
#      whole V4.1 model (body::open, HostResidency::at_load: the host set, ~190 GB, populated) — runs
#      in lane A even when its gpu-gate.sh call takes `any` (then on the 3090). Not alone: lane B keeps
#      running. solo wins. What keeps two V4.1 loads apart on the box, across trees and batches, is
#      tools/gpu-gate.sh's V4.1 load lock, which a member asks for by exporting
#      BLOOMERY_GATE_V41_LOAD=1 in its box command (group and export must name the same recipes, a named
#      error otherwise). The lane is for the plan: members in two lanes would serialize on that lock
#      anyway, so lane sums that ran them in parallel would be wrong. In the fixture tier the group pins
#      nothing: a fixture load takes no V4.1 load lock (tools/gpu-gate.sh), so a member is placed by its
#      gpu-gate.sh form like any recipe, and a 3090-only call still keeps it in lane A.
#   deferred (the fixture tier only; class D): a recipe the fixture tier cannot run, which the batch defers to the real tier instead
#      of running it or counting it red. Two kinds, two owners. (1) real-only: the gate has no fixture-tier conversion. Its box
#      command opens with `bash tools/ref/real-only.sh <its own name> && …` (every box.sh command of the recipe does), and that call
#      is the whole declaration: this runner reads the recipe's text, the box command runs the same script (it stops a run by hand
#      under BLOOMERY_TIER=fixture by name, exit 66, before a build or a lock). A recipe whose closure holds such a call is real-only
#      too. (2) no-fixture: the recipe's family (the BLOOMERY_MODEL of its box line, else the default profile) has no row in the
#      fixture table of tools/ref/ref-paths.sh — a profile that table does not name falls to its `*)` row, `none`, where box.sh
#      exits 66; this runner reads that table, so the fact has one owner. In the real tier neither kind places anything: the recipe
#      is placed by its groups and calls as without them. In the fixture tier a deferred item gets no lane, no ledger key, no claim
#      and no times row, and the batch never calls the recipe, so it is never a run on the fixture and never a red. Its run.log line
#      is `<item> rc=deferred <real-only|no-fixture> lane=- deferred=1`, the dry run names it (`lane -`), and the batch writes every
#      deferred item to DIR/deferred.list (real-only gates first, then no-fixture families, one item per line in the --list format,
#      comment lines between), so a real-tier batch takes exactly that set with `--list`. The DONE line counts each in `deferred=`
#      (whole gates, where it otherwise counts clauses), then `real_only=<n>`, `no_fixture=<n>` (each when not 0) and
#      `deferred_list=<path>`. Named errors, wherever the recipe sits in the justfile: a real-only call that names another recipe, a
#      box.sh command of the recipe that does not open with its call (or a call outside one), a missing script, a recipe that is
#      both real-only and solo-real (a fixture-tier placement it never uses), and a fixture table this runner cannot read.
#   --tier real|fixture (default real; BLOOMERY_TIER in the environment names the same, and the two must agree): the tier
#      the batch runs in, passed to every item as BLOOMERY_TIER=fixture in its BLOOMERY_BOX_ENV — tools/box.sh resolves each
#      item's model file from it (the family's fixture, tools/ref/ref-paths.sh); a family with none is a deferred item (above,
#      never a run on the real file, never a red). The real tier adds nothing to the environment: its commands, keys and times are
#      today's. In the fixture tier a `[group('solo-real')]` recipe is balanced and `v41-load` pins no lane (above), the keys
#      carry the tier (they carry the box env), and the times file keeps the fixture runs apart from the real ones — a fixture
#      item's row is `name@BLOOMERY_TIER=fixture[…]`, an entry of its own. An item's own `@BLOOMERY_TIER=…` is a named error:
#      the tier is the batch's. Clauses a fixture run leaves to the real tier print `deferred(real)` lines (crates/gpu-gates/
#      src/tier.rs); each item's line in run.log gains `deferred=<n>` when its log holds any, and the DONE line ends with the
#      batch's total (a deferred item counts one; the deferred paragraph above has the rest of the line).
# A recipe that runs a timing runner or takes the timing lease is refused, because a batch's builds
# contaminate a timed run. Two tests: what a script the recipe runs does, followed transitively — a
# tools/…sh that calls lease_take (tools/ref/lease.sh), or that opens the lease lock for a descriptor
# and waits on it with flock -w — and, for what that walk cannot see, the names in timed() below. Also
# refused: a recipe whose text runs this script (`just smoke`: a batch inside a batch runs lanes of its
# own on the same cards), and an item whose ARGS hand a `[group('solo')]` recipe's gpu-gate.sh binary
# one of the --flags that recipe passes it (in the self-test's fixture, `v41-a:--faults` is the solo
# recipe v41-solo's arm, which runs alone and with BLOOMERY_HOST_LOCK=1) — the error names the solo
# recipe to list instead. Lane A runs its items in list order, lane B in the order the balance placed them
# (its skips, then longest first), lane X by model family (the profile box.sh loads: V4.1 first, then each
# family in the order its first item is listed, each in list order); A and B run concurrently (cargo
# serializes their builds by one lock on the box tree's target directory, so a long lane-B build makes a
# lane-A item that starts inside it wait: lane B's order decides when its long builds start, and in a
# landing list whose longest balanced item runs on the Mac, check-recipes, they start after it).
# Work stealing: a lane that empties its own queue takes the other lane's not-yet-started movable
# items, in run order — a balanced item (class F: either card, or no card) that is not a ledger
# skip. A fixed, solo, v41-load or one-lane item never moves, and neither does an item the ledger
# already holds green (its key names one card). The thief runs a stolen item on its own lane's card
# — the item's other candidate: BLOOMERY_GATE_CARD is rewritten, and the ledger key, the ledger
# state and the times row's card label move with it, so the record names the card the item ran on
# and run.log's lane= names the lane that ran it; a no-card item (host group, plain CPU checks)
# changes lanes only. A steal onto a card first probes the timing lease through box.sh (READONLY,
# so no rsync: `. tools/ref/lease-probe.sh && lease_free`, the shared-lock probe — never an
# exclusive flock) and takes no card while the lease is held, re-probing after RETRY_WAIT seconds
# — waited in 1 s slices that end the pass the moment no candidate remains, so a sitting longer
# than the wait costs the lane nothing beyond its own items; a hold, or a sitting that starts after
# the probe, is the stolen item's own box.sh guard, as it is for every item. --lanes 1 steals
# nothing.
# --lanes 1 runs every item in the
# list's order in one lane (labelled A), with no card forced — the recipes' own defaults, as a hand
# batch runs them.
#
# Expected times, the balance's input: the median of the item's last 5 rows in the times file
# ${BLOOMERY_GATE_TIMES:-~/.cache/bloomery/gate-times.tsv} (on the Mac, shared by every tree),
# `item<TAB>lane<TAB>card<TAB>secs<TAB>date`, one row per item that ran, green or red: the item as
# written without its lane's card (a plain item is its recipe name; an item with @env or :ARGS is an
# entry of its own — `gate-gpu-ds41-prefill:--cases 512 …` is not the full prefill), the lane and card
# it ran on (3090 or a6000; both or a6000 for box.sh's own pick; none for no device code; any under
# --lanes 1, where the recipe picks at run time), the seconds of its final try (an rc 75 try before it
# waited on a lock and ran nothing; an item whose last try is still rc 75 writes no row), and the date.
# A final try that built cold — its log shows cargo compiling a crate from outside the tree, a registry
# or git dependency: a new remote directory, a toolchain, flag or pin move — writes its row to the
# sibling file <times file without .tsv>-cold.tsv, same format, which no plan reads, and its item line
# gains `cold=1`: one build must not drag the item's median. The row format stays five fields — every
# tree's copy of this script reads the one shared file.
# Only this script appends, each row with one write(2) on an O_APPEND descriptor under an exclusive
# flock; the plan reads the file under a shared one. A malformed row is a named error before anything
# runs. An item with no row expects DEFAULT_S (45 s, derived: the fixup5 batch's mean item, 2,167 s
# over 48 recipes, docs/gates-plan.md §1); --dry-run names the items that used it. With --ledger an
# item that skips counts 0 s, and a balanced gpu-gate item whose key is green on one card only goes to
# that card's lane, where it skips: the key carries the lane's card, so an item placed by time alone
# would run again on the other card although nothing it reads moved.
#
# Per item: stdout+stderr to DIR/g-<recipe>[-<n>].log (n for the n-th repeat of a recipe; each try
# appends under a `=== try` header). rc 75 (lock contention) retries after 30 s, up to 10 times; any
# other rc is final. DIR/run.log opens with `plan laneA=<s>s laneB=<s>s laneX=<s>s wall=<s>s
# defaults=<n> …` (the predicted sums, derived from the times file), then gets `<recipe>[-<n>] rc=<n>
# <s>s try=<t> lane=<A|B|X>` per item (plus `cold=1` for a cold build, `times=append-failed` when its
# times row could not be written, and `item=…` last when it carries env or ARGS), then `DONE total=<n> red=<n> wall=<s>s laneA=<s>s laneB=<s>s`
# (`laneX=` when X ran, `lint_warnings=<n>` when lint ran: `grep -c '^warning:'` on its log). A fixture batch's real-only item has
# its own line (`<item> rc=deferred <real-only|no-fixture> lane=- deferred=1`), counted in `total=` and in no other rc class. Exit 0
# iff every rc is 0 (or deferred, or a ledger skip). DIR defaults to $HOME/.cache/bloomery/batches/<tree>/<stamp> (<tree> the
# basename of this worktree's root): outside every tree, so the logs survive a `rm -rf target/`
# cleanup of the worktrees and neither mark a tree dirty nor reach box.sh's rsync. An explicit
# --out is still taken: under the tree's target/ it is gitignored and accepted, a DIR inside the
# tree but outside target/ is refused (box.sh would ship the logs and mark the tree dirty).
#
# Refuses to start while the timing lease (/root/bloomery-cpu.lock) or a hold
# (/root/bloomery-<owner>-hold, other than BLOOMERY_HOLD_OWNER's) is up, naming what is up (exit 75);
# a lease that cannot be tested is exit 70; no override. The check is tools/box.sh's own guard
# (lease_guard, tools/ref/lease-probe.sh) with BLOOMERY_BOX_WAIT=0. Once the batch runs, every item's box.sh
# call passes that guard with its default wait: a sitting that takes the lease mid-batch gets a quiet
# box — the item waits (up to 30 min, then rc 75, which retries below) and starts after the sitting.
# The wait comes before the item's command, so the runners' 900 s bound does not count it; nor does the
# item's times row: the final try's seconds less the waits its log names (box.sh's guard, gpu-gate.sh's card
# lock and V4.1 load lock), which the item line prints as `waited=<s>s`. --dry-run validates, prints each item's lane, command and
# plan (fixed, balanced or solo, with its expected seconds) and the predicted lane sums, and touches
# neither the box nor DIR (with --ledger it reads the box once, for the manifest below).
#
# Disk floor. Before anything starts — after argument parsing, before the log directory and the first
# lane — the free space of the volume that holds the tree's target/ and of the volume that holds OUT
# must each be at least MIN_FREE_GIB (giB). Below the floor: exit 69 and one `gate-batch: disk:` line
# naming the free GiB, the floor and the mount point — not 75, which is lock contention a batch
# retries. A df that fails or whose output does not parse is also 69, naming why. --dry-run prints
# the same verdict lines and does not fail: a dry run shows what a real run would do.
# BLOOMERY_MIN_FREE_GIB overrides the floor; an override that is not a positive integer is a named
# error (64), never silently ignored.
#
# Ledgers. Two files, one format, written only by this script: the lead's (--ledger, the lead's
# batches) and the rounds' (--round-ledger, a delegated round's batches). A round never passes --ledger:
# it writes the rounds' file only. What a batch reads to skip an item:
#   --ledger                  the lead's file (the default landing batch)
#   --ledger --trust-rounds   the lead's, then the rounds'; writes the lead's. The lead passes it only
#                             for a change that moves no behaviour — a move whose `just ptx-scan` equals
#                             the base (user, 2026-09-27: a round runner's recorded green at the same
#                             key is not re-run at landing; an agent saying "green" is not a record).
#                             The tool cannot tell a move from a change, so the flag is that judgment.
#   --round-ledger            the lead's, then the rounds'; writes the rounds'.
# A skip names its source, `src=lead` or `src=round`, in run.log and the dry run. A rebase moves the
# key of every item whose closure the landed commits touched, so a landing still reruns those; the
# saving is in the items the rebase left alone.
# Each item's input key is `tools/recipes.py key` (its section header lists the parts): the files its
# targets read (the walk `just affected` uses; the whole tree for a command run on the Mac, a cargo
# subcommand the parser does not model, or a script that walks the tree), the text of its recipe and
# of every recipe it depends on, the scripts those name, the cargo globals and the cuda-oxide pin, its
# ARGS, its full BLOOMERY_BOX_ENV (the caller's, the item's, its lane's card), the Mac-side variables
# box.sh carries, and a box manifest read once per batch through one box.sh call (`recipes.py
# box-manifest` behind box.sh's guard with BLOOMERY_BOX_WAIT=0: refused while the lease or a hold is
# up): ~/bloomery-env.sh, the kernel, each card's
# name, UUID, driver and VBIOS, rustc, llc, ptxas and cargo-oxide, the cuda-oxide backend, the content
# of $BLOOMERY_DATA (sha256 per file, cached by stat on the box in ~/.cache/bloomery/sha256-cache.tsv,
# so a file rewritten with the same bytes keeps its hash; the first fetch hashes everything, later ones
# only what moved), and the stat of every file in each /models/<name> directory the profiles, the box
# env or the tree name. The whole manifest is in every key but its model rows: the data a gate reads
# is not bounded by its recipe text, so a data change reruns every item — the safe side; a model
# directory's rows enter only the keys of the items that can open it (a /models literal in the item's
# files or recipe text, its profile among them, or a /models value in the box env).
#   Keys are computed before the lease check. An item whose key is in the ledger does not run: run.log
# gets `<stem> rc=skip green-at=<commit> <date> lane=<L>`. Every other item runs as without --ledger;
# one whose final rc is 0 is recorded after its key is computed again (against the same pre-batch box
# manifest) and found unchanged, so a tree-side input that moved while the batch ran is never recorded
# as green. Item lines gain `ledger=<state>`: recorded, changed (an input moved during the batch), red,
# never, unkeyed (no key before the batch, or a recheck that failed — named on stderr), append-failed. The DONE line gains `skipped=<n>`; the exit code counts only the items
# that ran. --rerun runs every item and still records. --dry-run --ledger prints each item's status and
# why: `skip` with the green run's commit, date and tree; `run` with what moved since the item's last
# green (the parts of each green key are kept in <ledger>.parts/<key>.parts).
#   Never skipped: a recipe that runs a reference tree's files (ik, llama.cpp, mistral.rs — today
# gate-1-1 and gate-tokenizer), an item whose env or ARGS name an
# absolute path the manifest does not read, and — with --lanes 1, where no card is forced — a recipe
# whose gpu-gate.sh call takes the `any` card, which it picks at run time. A key that cannot be computed
# (a missing file, a failed manifest) is a named error: the item runs and is not recorded.
#   The files: ${BLOOMERY_GATE_LEDGER:-~/.cache/bloomery/gate-ledger.tsv} (the lead's) and
# ${BLOOMERY_GATE_ROUND_LEDGER:-~/.cache/bloomery/gate-ledger-rounds.tsv} (the rounds'), on the Mac,
# outside every tree and shared by every lead's and round's tree, `key<TAB>recipe<TAB>item<TAB>commit<TAB>date<TAB>tree`, one line
# per green item. Only this script appends to it, each line with one write(2) on an O_APPEND descriptor
# while holding an exclusive flock on it: two leads' batches cannot interleave a line.
# Honest limits — what the key cannot see:
#   - a box file or process outside the manifest: the reference trees (hence never-skip), the box's own
#     tools the gates run (curl, crates/gpu-gates/src/bin/gate_ds41_serve.rs:200; sha256sum,
#     gate_e2e.rs:579 and gate_qwen3moe_e2e.rs:1029; nvidia-smi, crates/gpu-gates/src/bind.rs:302;
#     coreutils, python3), the page cache;
#   - hardware and kernel state: the 3090's bus (Xid 79), thermal and power events, and the topology
#     and memory state the gates read from /proc and /sys (crates/threads/src/lib.rs:542,
#     crates/gpu/src/model/launcher.rs:278, crates/engram/src/prefetch.rs:477 and lib.rs:700,
#     crates/gpu-gates/src/bin/gate_load_v41.rs:701 and :712);
#   - a model file's content: model files are keyed by size, mtime, ctime and inode;
#   - a model directory a gate opens through a path no literal spells (one ~/bloomery-env.sh exports,
#     one a program builds at run time): its rows reach no item's key;
#   - a box file that changes during the batch (the manifest is read once, before it): the next batch's
#     key differs, so the item reruns then;
#   - a stale binary: the key trusts cargo's freshness check, so a green run of a binary cargo failed
#     to rebuild would be recorded;
#   - a directory symlink inside $BLOOMERY_DATA, keyed by its target path, not walked (none today).
# $BLOOMERY_DATA/nsys/ and ncu/ are left out of the manifest: only the timing runners write them and no
# gate reads them (recipes.py's self-test fails when a gate names one).
#
# The GPU hold. A --ledger batch (the lead's landing batch) puts a GPU-only hold up on the box for its whole run, so that
# other tracks' GPU gates stop queueing on the cards between its items: /root/bloomery-batch.gpuhold, `owner=<id> since=<epoch>`,
# written and read only by tools/gpu-gate.sh (--gpuhold; its header has the wait, the bound and the stale rule). Each item runs
# with BLOOMERY_BATCH_OWNER=<id> in its box env — added to the call, never to the key, the times row or the ledger — so the
# batch's own gpu-gate.sh runs pass and every other run waits at the top of its poll, holding no lock. Builds and every non-GPU
# box command ignore it (its name does not match the sittings' /root/bloomery-<owner>-hold). The hold goes up after the start
# check, only when some item runs on the box, and is refused (75) while another batch's fresh hold is up; a heartbeat (a
# background job, pid in DIR/gpuhold.pid) refreshes its mtime every 60 s (BLOOMERY_GPU_HOLD_BEAT, for the self-test), through box.sh
# with BLOOMERY_BOX_READONLY=1 and BLOOMERY_BOX_WAIT=0: no sync, no guard wait — a deliberate write of that one control file.
# The box calls a hold stale after 300 s without a beat (this Mac asleep or cut off), so it never outlives its batch by more than
# that. The hold comes down on every exit — the end, a red batch, `die`, INT/TERM — by the EXIT trap, which kills only the
# heartbeat pid written under DIR and removes the hold only if it is still this batch's. DIR/gpuhold.log holds the calls' output.
# --no-gpu-hold (with --ledger) runs without it; --round-ledger and a plain batch never take it. --dry-run prints the hold it
# would take and touches nothing.
# Stopping a batch (INT/TERM): the trap signals only the pids written at spawn under DIR (lane-*.pid,
# lane-*.child). A box process a killed item left behind is `just box-gc`'s to clear.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
JUST=(just --justfile "$ROOT/justfile" --working-directory "$ROOT")

SMOKE=(
  check-recipes check-rustflags check-comments check-levers check-arch check fmt-check lint gate-ptx-spill
  gate-gpu-ds41-step
  "gate-gpu-ds41-prefill:--cases 512 --no-split --no-extra"
  gate-gpu-e2e
)
TRIES_MAX=11
RETRY_WAIT=30
STEAL_WAIT=$RETRY_WAIT # a held timing lease: the steal's re-probe bound, waited in 1 s slices
DEFAULT_S=45 # the expected seconds of an item with no row in the times file (the header says why 45)
MIN_FREE_GIB=3 # the disk floor (the header's «Disk floor»): 2× the larger of one cold check's and one cold combos' growth of a tree's target/ (tools/mac-check.sh), rounded up to a whole GiB
TIMES_FILE=${BLOOMERY_GATE_TIMES:-$HOME/.cache/bloomery/gate-times.tsv}
COLD_FILE=${TIMES_FILE%.tsv}-cold.tsv # a cold build's rows, outside the median (the header)

USAGE="usage: tools/gate-batch.sh [--out DIR] [--smoke | --weekly | --list FILE | ITEM…] [--dry-run] [--lanes 1|2] [--tier real|fixture] [--ledger [--trust-rounds] [--no-gpu-hold] | --round-ledger] [--rerun] | --classes | --self-test"
die() { echo "gate-batch: $*" >&2; exit "${RC:-64}"; }

# The append, for the ledger and the times file: a ledger record's parts file first (its parts exist
# once the record does), then the line, one write(2) on an O_APPEND descriptor under an exclusive flock.
IFS= read -r -d '' PYAPPEND << 'PY' || true
import fcntl, os, shutil, sys

path, line = sys.argv[1:3]
if len(sys.argv) == 5:
    parts_src, parts_dst = sys.argv[3:5]
    if not os.path.exists(parts_dst):
        os.makedirs(os.path.dirname(parts_dst), exist_ok=True)
        tmp = f"{parts_dst}.{os.getpid()}.tmp"
        shutil.copyfile(parts_src, tmp)
        os.replace(tmp, parts_dst)
os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
data = (line.replace("\n", " ") + "\n").encode()
fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
try:
    fcntl.flock(fd, fcntl.LOCK_EX)
    if os.write(fd, data) != len(data):
        sys.exit(f"gate-batch: a short write to {path}")
finally:
    os.close(fd)
PY

times_record() { # $1 = plan index, $2 = lane, $3 = the final try's seconds, $4 = 1 for a cold build
  local f=$TIMES_FILE
  [ "${4:-0}" = 0 ] || f=$COLD_FILE
  python3 -c "$PYAPPEND" "$f" "$(printf '%s\t%s\t%s\t%s\t%s' "${P_TKEY[$1]}" "$2" "${P_TCARD[$1]}" "$3" "$(date '+%Y-%m-%dT%H:%M:%S%z')")"
}

# cold_of <log> <try>: 1 when the try's section of the item's log compiled a crate from outside the tree
# (cargo's `Compiling <name> v<ver>` with no local ` (/path)`: a registry or git dependency), which a
# build does only in a new remote directory or after the toolchain, the flags or a pinned dependency
# moved; else 0. Such a try's seconds are a one-off build, not the item's.
cold_of() {
  awk -v t="=== try $2 " '
    index($0, t) == 1 { on = 1; next }
    /^=== try / { on = 0 }
    on && /^ *Compiling [A-Za-z0-9_-]+ v[0-9]/ && !/ \(\// { c = 1 }
    END { print c + 0 }' "$1"
}

# deferred_of <log> <try>: the `deferred(real)` lines of the try's section of the item's log — the clauses a fixture-tier
# gate left to the real tier (crates/gpu-gates/src/tier.rs prints them; its test holds this prefix to that file's).
deferred_of() {
  awk -v t="=== try $2 " '
    index($0, t) == 1 { on = 1; next }
    /^=== try / { on = 0 }
    on && /^deferred\(real\) / { n++ }
    END { print n + 0 }' "$1"
}

# try_waits <log> <try>: the seconds the given try spent waiting, not running — box.sh's guard for a sitting
# (`[guard] … after N s: the command starts`) and tools/gpu-gate.sh for a gate lock, for the V4.1 load
# lock and for another batch's GPU hold (`gpu-gate.sh: waited N s for the …`),
# summed over the try's section of the item's log. A times row holds the rest: a lock queue is not a gate's cost.
try_waits() {
  awk -v t="=== try $2 " '
    index($0, t) == 1 { on = 1; next }
    /^=== try / { on = 0 }
    on && match($0, /after [0-9]+ s: the command starts/) { split(substr($0, RSTART + 6), a, " "); w += a[1] }
    on && match($0, /^gpu-gate\.sh: waited [0-9]+ s for the (gate lock|V4\.1 load lock|batch hold)/) { split(substr($0, RSTART + 20), a, " "); w += a[1] }
    END { print w + 0 }' "$1"
}

# The disk floor, the header's «Disk floor». floor_gib resolves the floor once (the constant, or
# BLOOMERY_MIN_FREE_GIB, which must be a positive integer of GiB or a named 64). df_able walks up to
# the deepest existing ancestor of a path — df errors on a path that does not exist, and the volume
# of the deepest existing ancestor is where mkdir -p creates the rest (a symlink or mount inside the
# missing span cannot change it). disk_ok reads the Available column of `df -Pk` (POSIX output,
# macOS and Linux alike) and prints one verdict line: 0 above the floor, 1 below it (a real run
# turns that into 69; a dry run names it and continues), 69 when df fails or does not parse — a
# named failure in a dry run too, whose verdict line must be a verdict and not a guess.
floor_gib() {
  if [ -n "${BLOOMERY_MIN_FREE_GIB+x}" ]; then
    case $BLOOMERY_MIN_FREE_GIB in
      '' | 0 | *[!0-9]*) RC=64 die "BLOOMERY_MIN_FREE_GIB='$BLOOMERY_MIN_FREE_GIB' is not a positive integer of GiB" ;;
    esac
    printf '%s\n' "$((10#$BLOOMERY_MIN_FREE_GIB))"
  else
    printf '%s\n' "$MIN_FREE_GIB"
  fi
}

df_able() { # the deepest existing ancestor of $1
  local p=$1
  while [ ! -e "$p" ]; do
    case $p in
      / | '') printf '/\n'; return ;;
      *) p=${p%/*} ;;
    esac
  done
  printf '%s\n' "$p"
}

disk_ok() { # $1 a path whose volume must hold the floor free; one verdict line, then 0 / 1 / 69
  local path=$1 probe out rc=0 line
  local floor
  floor=$(floor_gib) || return $?
  probe=$(df_able "$path")
  out=$(df -Pk "$probe" 2>&1) || rc=$?
  [ "$rc" = 0 ] || RC=69 die "disk: df -Pk $probe failed (rc $rc): $(printf '%s\n' "$out" | tail -1)"
  rc=0
  line=$(printf '%s\n' "$out" | awk -v kib=$((10#$floor * 1048576)) -v floor="$floor" -v probe="$probe" '
    END {
      if (NR < 2 || $4 !~ /^[0-9]+$/) {
        printf "gate-batch: disk: df -Pk %s output does not parse (the Available column): %s\n", probe, $0 > "/dev/stderr"
        exit 2
      }
      mount = $6; for (i = 7; i <= NF; i++) mount = mount " " $i
      if ($4 + 0 < kib) {
        printf "gate-batch: disk: %.1f GiB free on %s (for %s), below the %s GiB floor — free space and run again\n", $4 / 1048576, mount, probe, floor > "/dev/stderr"
        exit 1
      }
      printf "gate-batch: disk: %.1f GiB free on %s (for %s), above the %s GiB floor\n", $4 / 1048576, mount, probe, floor
    }') || rc=$?
  [ "$rc" = 2 ] && rc=69
  [ "$rc" = 0 ] || return "$rc"
  printf '%s\n' "$line"
}

# --self-test: the rules of items 1 and 4 on a fixture justfile and fixture times rows, in a temporary
# tree holding a copy of this script (no box, no ssh; `just` and python3 only). One line per case,
# `ok <name>` or `FAIL <name>: <why>` with the output; exit 0 iff none failed. check-recipes runs it.
self_test() {
  local self t n=0 bad=0 out rc
  unset BLOOMERY_TIER # the tier of a case is the case's own
  self=$(cd "$(dirname "$0")" && pwd -P)/$(basename "$0")
  t=$(mktemp -d "${TMPDIR:-/tmp}/gate-batch-test.XXXXXX") || { echo "gate-batch: self-test: no temporary directory" >&2; return 70; }
  # shellcheck disable=SC2064 # the path is fixed now
  trap "rm -rf '$t'" EXIT
  mkdir -p "$t/tools"
  cp "$self" "$t/tools/gate-batch.sh"
  cat > "$t/justfile" << 'JF'
[group('v41-load')]
v41-any:
    ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gen_x'

[group('v41-load')]
v41-a *ARGS:
    ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && bash tools/gpu-gate.sh gen_y {{ARGS}}'

[group('solo')]
[group('v41-load')]
v41-solo:
    ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && BLOOMERY_HOST_LOCK=1 bash tools/gpu-gate.sh gen_y --faults'

plain-any:
    ./tools/box.sh 'BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh other'

host:
    ./tools/box.sh 'bash tools/gate.sh -p x'

gate-x:
    ./tools/box.sh 'bash tools/gate.sh -p y'

gate-y:
    ./tools/box.sh 'bash tools/gate.sh -p w'

[group('host')]
hostdev:
    ./tools/box.sh 'bash tools/gate.sh --oxide -p hd'

commented:
    # the timed recipes source tools/ref/timing-card.sh; this one runs none
    ./tools/box.sh 'bash tools/gate.sh -p c'

timedrun:
    ./tools/box.sh 'bash tools/ref/timing-card.sh probe'

nested *ARGS:
    ./tools/gate-batch.sh --smoke {{ARGS}}

weekly-a:
    ./tools/box.sh 'bash tools/gate.sh -p wa'

[group('solo')]
weekly-b:
    ./tools/box.sh 'BLOOMERY_HOST_LOCK=1 bash tools/gpu-gate.sh wb'

[group('solo-real')]
sr-any:
    ./tools/box.sh 'BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gen_sr'

[group('solo-real')]
[group('v41-load')]
sr-v41:
    ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gen_sv'

[group('solo')]
x-glm:
    BLOOMERY_MODEL=glm5next ./tools/box.sh 'bash tools/gpu-gate.sh xg'

[group('solo')]
x-q:
    BLOOMERY_MODEL=qwen4exp ./tools/box.sh 'bash tools/gpu-gate.sh xq'

[group('solo')]
x-v41:
    BLOOMERY_MODEL=deepseek41 BLOOMERY_CARD=both ./tools/box.sh 'bash tools/gpu-gate.sh xv'

[group('solo')]
x-none:
    ./tools/box.sh 'bash tools/gpu-gate.sh xn'

[group('solo')]
x-v41b: x-v41dep

x-v41dep:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'bash tools/gpu-gate.sh xw'

steal-slow:
    ./tools/box.sh 'bash tools/gpu-gate.sh gen_s'

steal-fix:
    ./tools/box.sh 'bash tools/gpu-gate.sh gen_f'

steal-bal:
    ./tools/box.sh 'BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gen_b'

steal-lib:
    ./tools/box.sh 'bash tools/gate.sh --oxide -p sl'

# The real-only class: a call that opens the box command and names the recipe.
ro-any:
    ./tools/box.sh 'bash tools/ref/real-only.sh ro-any && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gen_ro'

[group('solo')]
[group('v41-load')]
ro-solo:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'bash tools/ref/real-only.sh ro-solo && export BLOOMERY_GATE_V41_LOAD=1 && bash tools/gpu-gate.sh gen_rs'

ro-chain: ro-any

ro-glm:
    BLOOMERY_MODEL=glm5next ./tools/box.sh 'bash tools/ref/real-only.sh ro-glm && bash tools/gpu-gate.sh xr'

[group('solo-real')]
sr-a6000:
    BLOOMERY_CARD=a6000 ./tools/box.sh 'bash tools/gpu-gate.sh gen_sa --place a'
JF
  # The default profile a recipe with no BLOOMERY_MODEL loads.
  mkdir -p "$t/tools/ref"
  # The GPU hold's box-side writer is the real tools/gpu-gate.sh (with the real lease-probe.sh it sources): the fake box below runs
  # its --gpuhold calls on the fixture, so the batch's hold is the real one end to end.
  cp "$(dirname "$self")/gpu-gate.sh" "$t/tools/gpu-gate.sh"
  cp "$(dirname "$self")/ref/lease-probe.sh" "$t/tools/ref/lease-probe.sh"
  cp "$(dirname "$self")/ref/real-only.sh" "$t/tools/ref/real-only.sh"
  cat > "$t/tools/ref/ref-paths.sh" << 'RP'
: "${BLOOMERY_MODEL:=${BLOOMERY_REF_MODEL_PROFILE:-deepseek2}}"
case "${BLOOMERY_TIER:-real}" in
  fixture)
    case "$BLOOMERY_MODEL" in
      qwen4exp) __fixture_dir=qwen38 ;;
      deepseek41) __fixture_dir=v41 ;;
      deepseek2 | qwen3moe | qwen35moe | qwen35) __fixture_dir=self ;;
      *) __fixture_dir=none ;;
    esac
    ;;
esac
RP
  # Unpinned, v41-any (50 s, balanced) would go to lane B: lane A holds v41-a's 100 s.
  printf '%s\t%s\t%s\t%s\t%s\n' v41-a A 3090 100 2026-09-27T10:00:00+0900 v41-any B a6000 50 2026-09-27T10:00:00+0900 \
    plain-any B a6000 10 2026-09-27T10:00:00+0900 host B none 5 2026-09-27T10:00:00+0900 > "$t/times.tsv"
  # A fake df first on PATH for the whole self-test — the disk floor's verdict must not depend on
  # the host's real disk. One POSIX data line whose Available column is DISK_KB KiB on /fake/mount;
  # DISK_RC makes df fail, DISK_BAD output that does not parse.
  mkdir -p "$t/fakebin"
  cat > "$t/fakebin/df" << 'DF'
#!/bin/sh
if [ -n "${DISK_RC:-}" ]; then echo "df: fake failure" >&2; exit "$DISK_RC"; fi
if [ -n "${DISK_BAD:-}" ]; then printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\ngarbage line\n'; exit 0; fi
printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\n/dev/fake 100000000 1000 %s 1%% /fake/mount\n' "${DISK_KB:-999999999}"
DF
  chmod +x "$t/fakebin/df"
  export PATH="$t/fakebin:$PATH"
  pass() { n=$((n + 1)); echo "ok $1"; }
  fail() {
    n=$((n + 1)) bad=$((bad + 1))
    echo "FAIL $1: ${2:-}"
    printf '%s\n' "$out" | sed 's/^/    | /'
  }
  # check <name> <want rc> <ERE the output must hold> <command…>
  check() {
    local name=$1 want=$2 pat=$3
    shift 3
    rc=0
    out=$("$@" 2>&1) || rc=$?
    if [ "$rc" != "$want" ]; then fail "$name" "rc $rc, want $want"
    elif ! grep -Eq -- "$pat" <<< "$out"; then fail "$name" "no line matches /$pat/"
    else pass "$name"; fi
  }
  local gb=(bash "$t/tools/gate-batch.sh")
  export BLOOMERY_GATE_TIMES=$t/times.tsv
  check 'classes: an any-form v41-load recipe is lane A on the 3090' 0 \
    "^v41-any	A	fixed	3090	v41-any: \[group\('v41-load'\)\]" "${gb[@]}" --classes
  check 'classes: an any-form recipe outside the group stays balanced' 0 '^plain-any	F	balanced	3090,a6000	' "${gb[@]}" --classes
  check 'classes: solo wins over v41-load' 0 '^v41-solo	X	solo	3090	' "${gb[@]}" --classes
  check 'classes: a host recipe with device code is balanced with no card forced' 0 "^hostdev	F	balanced	-	hostdev: \[group\('host'\)\]" "${gb[@]}" --classes
  check 'classes: a timed name in a comment of the body is not a timed recipe' 0 '^commented	F	balanced	-	no device code$' "${gb[@]}" --classes
  check 'classes: a timed script in command position is' 0 '^timedrun	T	timed	-	timedrun runs tools/ref/timing-card\.sh' "${gb[@]}" --classes
  check 'classes: a recipe that runs a batch is refused' 0 '^nested	R	refused	-	nested runs tools/gate-batch\.sh' "${gb[@]}" --classes
  check 'dry run: the v41-load member runs in lane A with the 3090 forced' 0 \
    "^lane A  v41-any +BLOOMERY_BOX_ENV='BLOOMERY_GATE_CARD=3090' just v41-any$" "${gb[@]}" --dry-run v41-a v41-any plain-any host
  check 'dry run: lane A holds both V4.1 loads' 0 'predicted laneA=150s laneB=15s laneX=0s wall=150s' \
    "${gb[@]}" --dry-run v41-a v41-any plain-any host
  # Lane X by model family, V4.1 first, then each family in the order its first item is listed; a
  # dependency's profile is the recipe's, a recipe with none loads ref-paths.sh's default.
  out=$("${gb[@]}" --dry-run x-glm x-none x-q x-v41 x-glm x-v41b 2>&1) || fail 'order: lane X by family' "the dry run failed"
  got=$(printf '%s\n' "$out" | sed -n 's/^lane X  \([^ ]*\) .*/\1/p' | tr '\n' ' ')
  if [ "$got" = 'x-v41 x-v41b x-glm x-glm-2 x-none x-q ' ]; then pass 'order: lane X by family, V4.1 first, then first appearance'
  else fail 'order: lane X by family, V4.1 first, then first appearance' "got '$got'"; fi
  # Lane B runs longest first (the Mac-only check-recipes heads it in a landing batch), not in list order.
  out=$("${gb[@]}" --dry-run v41-a host plain-any 2>&1) || fail 'order: lane B longest first' "the dry run failed"
  got=$(printf '%s\n' "$out" | sed -n 's/^lane B  \([^ ]*\) .*/\1/p' | tr '\n' ' ')
  if [ "$got" = 'plain-any host ' ]; then pass 'order: lane B longest first, not in list order'
  else fail 'order: lane B longest first, not in list order' "got '$got'"; fi
  check 'items: a batch as an item is refused by name' 65 'nested runs tools/gate-batch\.sh' "${gb[@]}" --dry-run host nested
  check "items: a solo recipe's arm through another recipe is refused, naming the solo recipe" 65 \
    "'v41-a:--faults' hands gen_y --faults, the arm of the solo recipe v41-solo" "${gb[@]}" --dry-run 'v41-a:--faults'
  check 'items: other ARGS of the same binary pass' 0 "^lane A  v41-a-2 " "${gb[@]}" --dry-run v41-a 'v41-a:--sets'
  # `just affected … 2>&1` output: just's echoed recipe line, then the `affected:` header and the recipe lines.
  printf '%s\n' './tools/affected-gates.sh main --no-box' 'affected: a..b — 2 changed files, 2 of 6 gate-recipes selected' \
    '  gate-x                      crates/x/src/lib.rs (+1)' 'unmapped: none' > "$t/aff-echo.txt"
  check "list: just affected output behind just's echoed line takes its recipe lines" 0 '^lane [AB]  gate-x ' \
    "${gb[@]}" --dry-run --list "$t/aff-echo.txt"
  # `just affected --narrow` output: the recipes it keeps run; a `  - gate-y` line it leaves out does not.
  printf '%s\n' './tools/affected-gates.sh main --no-box --narrow --scan b.log n.log' \
    'affected: a..b — 2 changed files, 1 of 6 gate-recipes selected (narrowed from 2 by ptx-scan)' 'narrow: ptx-scan gx: identical' \
    '  gate-x  crates/x/src/lib.rs (+1)' 'recipes: gate-x' 'always: ' 'left out (1), each by ptx-scan gx: identical:' \
    '  - gate-y  (ptx-scan gx: identical)' 'unmapped (0):' > "$t/aff-narrow.txt"
  check 'list: a narrowed affected output runs the recipes it keeps' 0 '^lane [AB]  gate-x ' "${gb[@]}" --dry-run --list "$t/aff-narrow.txt"
  if grep -q 'gate-y' <<< "$out"; then fail 'list: a narrowed affected output does not run a left-out recipe' "gate-y is in the batch"
  else pass 'list: a narrowed affected output does not run a left-out recipe'; fi
  # A weekly recipe a trigger named: its `  weekly-` line runs, beside the gate lines.
  printf '%s\n' 'affected: a..b — 2 changed files, 1 of 6 gate-recipes selected, 1 of 2 weekly-recipes by a trigger' \
    '  gate-x    crates/x/src/lib.rs  [lib x]' '  weekly-a  crates/w/src/lib.rs  [trigger tools/gate-paths.tsv:9 crates/w/**]' \
    'recipes: gate-x weekly-a' 'unmapped (0):' > "$t/aff-weekly.txt"
  check 'list: a weekly recipe a trigger named in affected output runs' 0 '^lane [AB]  weekly-a ' "${gb[@]}" --dry-run --list "$t/aff-weekly.txt"
  # --weekly: every weekly-* recipe of the justfile, each in its class; none of another name.
  check 'weekly: a weekly recipe with no group is balanced' 0 '^lane [AB]  weekly-a ' "${gb[@]}" --dry-run --weekly
  check 'weekly: a solo weekly recipe runs alone' 0 '^lane X  weekly-b ' "${gb[@]}" --dry-run --weekly
  if grep -Eq '^lane [ABX]  (gate|v41|host|plain)' <<< "$out"; then fail 'weekly: nothing but the weekly-* recipes' "another recipe is in the batch"
  else pass 'weekly: nothing but the weekly-* recipes'; fi
  check 'weekly: with items, a named error' 64 'exclusive' "${gb[@]}" --dry-run --weekly gate-x
  printf '%s\n' 'something else' 'affected: a..b — 1 changed file' '  gate-x  crates/x/src/lib.rs (+1)' > "$t/aff-other.txt"
  check 'list: any other line above the affected: header is a malformed item' 65 "aff-other.txt:1: malformed item 'something else'" \
    "${gb[@]}" --dry-run --list "$t/aff-other.txt"
  # Work stealing (the header's paragraph): real batch runs on the fixture, through a fake box.sh
  # that logs every call and never runs the command, and a key stub that keys an item by its own
  # text (env included: the card names differ). An item's env drives the fake: FAKE_SLEEP holds the
  # call that many seconds, FAKE_UNTIL=<word> ends the hold early once a logged command (or the
  # steal's lease probe, logged as `probe`) ends in <word>, then FAKE_GRACE seconds more — so a
  # case waits on the other lane's progress, not on a race — and FAKE_LEASE_UP=1 raises the lease
  # marker the fake answers the steal's READONLY probe (`. tools/ref/lease-probe.sh && lease_free`)
  # from.
  cat > "$t/tools/box.sh" << 'FB'
#!/bin/sh
here=$(dirname "$0")/..
st=$here/fake-state
mkdir -p "$st"
case "$*" in
  '. tools/ref/lease-probe.sh && lease_free')
    echo probe >> "$st/probe.log"
    [ -e "$st/lease-up" ] || exit 0
    echo '[guard] the fake lease is up' >&2
    exit 1 ;;
  'bash tools/gpu-gate.sh --gpuhold '*)
    # The hold's writer: logged with the box.sh flags the batch passed, then the real script on the fixture, its hold file under
    # fake-state (--test-locks).
    printf 'ro=%s wait=%s %s\n' "${BLOOMERY_BOX_READONLY:-0}" "${BLOOMERY_BOX_WAIT:-}" "$*" >> "$st/hold.log"
    cd "$here" || exit 70
    # shellcheck disable=SC2086 # the verb and the owner: two words
    exec bash tools/gpu-gate.sh --test-locks "$st" ${*#bash tools/gpu-gate.sh } ;;
esac
for kv in ${BLOOMERY_BOX_ENV:-}; do export "$kv"; done
printf '%s\t%s\n' "${BLOOMERY_BOX_ENV:-}" "$*" >> "$st/box.log"
# What an item sees of the batch's hold when it starts: its owner from the env, and the hold file as it stands.
printf '%s\t%s\t%s\n' "${BLOOMERY_BATCH_OWNER:-none}" "$(head -1 "$st/batch.gpuhold" 2> /dev/null || echo nohold)" "$*" >> "$st/seen.log"
n=0
while [ "$n" -lt "${FAKE_DEFER:-0}" ]; do
  echo "deferred(real) oracle: fake clause $n"
  n=$((n + 1))
done
[ "${FAKE_LEASE_UP:-0}" = 1 ] && : > "$st/lease-up"
k=0
while [ "$k" -lt "${FAKE_SLEEP:-0}" ]; do
  if [ -n "${FAKE_UNTIL:-}" ] && cat "$st/box.log" "$st/probe.log" 2> /dev/null | grep -q "${FAKE_UNTIL}\$"; then
    sleep "${FAKE_GRACE:-0}"
    break
  fi
  sleep 1
  k=$((k + 1))
done
# …and whether the hold is still up when it ends (a lane that finished first must not have taken it down).
if [ -f "$st/batch.gpuhold" ]; then printf 'present\t%s\n' "$*" >> "$st/end.log"; else printf 'ABSENT\t%s\n' "$*" >> "$st/end.log"; fi
exit "${FAKE_RC:-0}"
FB
  chmod +x "$t/tools/box.sh"
  cat > "$t/tools/recipes.py" << 'FP'
#!/usr/bin/env python3
import sys
a = sys.argv[1:]
if a[:1] == ["box-manifest"]:
    sys.exit(0)
items, i, pd = [], 1, None
while i < len(a):
    if a[i].startswith("--"):
        if a[i] == "--box-env":
            import os
            d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fake-state")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "keycalls.log"), "a") as fh:
                fh.write(a[i + 1] + "\n")
        if a[i] == "--parts-dir":
            pd = a[i + 1]
        i += 1 if a[i] == "--rerun" else 2
        continue
    items.append(a[i])
    i += 1
import os
for k, it in enumerate(items):
    if os.environ.get("FAKE_SKIP") == "1":
        print(f"stub-{it}\t{it}\tskip\tgreen-at=abc1234 2026-10-08T00:00:00+0900 tree=/x")
    else:
        print(f"stub-{it}\t{it}\trun\tstub")
    if pd:
        import os
        os.makedirs(pd, exist_ok=True)
        with open(f"{pd}/{k}.parts", "w") as fh:
            fh.write("stub\n")
FP
  want() { # <name> <file> <ERE>: the file holds a matching line
    n=$((n + 1))
    if [ -f "$2" ] && grep -Eq -- "$3" "$2"; then echo "ok $1"
    else bad=$((bad + 1)); echo "FAIL $1: no line matches /$3/ in $2"; fi
  }
  want_not() { # <name> <file> <ERE>: the file holds no matching line
    n=$((n + 1))
    if [ -f "$2" ] && grep -Eq -- "$3" "$2"; then bad=$((bad + 1)); echo "FAIL $1: a line matches /$3/ in $2"
    else echo "ok $1"; fi
  }
  want_row() { # <name> <file> <awk program over $1 (env) and $2 (command)>: a row answers it
    n=$((n + 1))
    if [ -f "$2" ] && [ -n "$(awk -F'\t' "$3" "$2")" ]; then echo "ok $1"
    else bad=$((bad + 1)); echo "FAIL $1: no row of $2 answers {$3}"; fi
  }
  want_no_row() { # <name> <file> <awk program>: no row answers it
    n=$((n + 1))
    if [ -f "$2" ] && [ -n "$(awk -F'\t' "$3" "$2")" ]; then bad=$((bad + 1)); echo "FAIL $1: a row of $2 answers {$3}"
    else echo "ok $1"; fi
  }
  steal_case() { # <tag> <times rows…> -- <items…>: one real batch run on the fixture, rc printed;
    # LEASE_UP=1 puts the fake lease up before the batch starts
    local tag=$1 rc=0
    local times=$t/times-$tag.tsv
    shift
    local rows=()
    while [ "$1" != -- ]; do rows+=("$1"); shift; done
    shift
    printf '%s\n' "${rows[@]}" > "$times"
    rm -rf "$t/fake-state" "$t/target/c-$tag"
    if [ "${LEASE_UP:-0}" = 1 ]; then mkdir -p "$t/fake-state" && : > "$t/fake-state/lease-up"; fi
    # PRE_HOLD=<owner> [PRE_AGE=<s>]: another batch's GPU hold is already up on the fake box, last refreshed <s> seconds ago
    if [ -n "${PRE_HOLD:-}" ]; then
      mkdir -p "$t/fake-state"
      printf 'owner=%s since=%s\n' "$PRE_HOLD" "$(date +%s)" > "$t/fake-state/batch.gpuhold"
      python3 -c 'import os, sys, time; a = time.time() - float(sys.argv[2]); os.utime(sys.argv[1], (a, a))' "$t/fake-state/batch.gpuhold" "${PRE_AGE:-0}"
    fi
    # LFLAG: the ledger flag of the batch (default the round's)
    BLOOMERY_GATE_TIMES=$times BLOOMERY_GATE_LEDGER=$t/lead-$tag.tsv BLOOMERY_GATE_ROUND_LEDGER=$t/rounds-$tag.tsv \
      "${gb[@]}" --out "$t/target/c-$tag" "${LFLAG:---round-ledger}" "$@" > "$t/out-$tag.log" 2>&1 || rc=$?
    printf '%s' "$rc"
  }
  rc_ok() { # <name> <tag> <want total>: the case's batch ended green
    if [ "$(cat "$t/rc-$2")" = 0 ] && grep -Eq "^DONE total=$3 red=0 " "$t/out-$2.log"; then pass "$1"
    else fail "$1" "rc=$(cat "$t/rc-$2") $(tail -1 "$t/out-$2.log" 2>/dev/null)"; fi
  }
  local d=2026-09-27T10:00:00+0900
  # A contended 3090: lane A's first item (fixed) holds its lane until the balanced item behind it
  # has run — on today's runner, 20 s and then in lane A. Lane B, idle, steals it onto the A6000;
  # the times row and the ledger key name that card.
  local slow_a='steal-slow@FAKE_SLEEP=20,FAKE_UNTIL=gen_b'
  rc=$(steal_case steal-a "$slow_a	A	3090	30	$d" "host	B	none	70	$d" "steal-bal	A	3090	60	$d" \
    "plain-any	B	a6000	10	$d" -- "$slow_a" host steal-bal plain-any)
  printf '%s' "$rc" > "$t/rc-steal-a"
  rc_ok 'steal: a contended lane A and an idle lane B end green' steal-a 4
  want 'steal: the balanced item runs in lane B' "$t/target/c-steal-a/run.log" '^steal-bal rc=0 [0-9]+s try=1 lane=B( |$)'
  want_row 'steal: its box call carries the A6000 card' "$t/fake-state/box.log" '$1 ~ /BLOOMERY_GATE_CARD=a6000/ && $2 ~ /gen_b$/'
  want 'steal: its times row names lane B and the A6000' "$t/times-steal-a.tsv" "$(printf '^steal-bal\tB\ta6000\t')"
  want 'steal: the ledger records it under the A6000 key' "$t/rounds-steal-a.tsv" "$(printf '^stub-steal-bal@BLOOMERY_GATE_CARD=a6000\tsteal-bal\t')"
  # A fixed item never moves: lane B empties while two fixed items wait behind lane A's first (held
  # until lane B's own item ran, then 3 s more), a 3090 one and a no-card one (device code with no
  # gate lock, which lands on the box env's 3090 pin), and takes neither.
  local slow_f='steal-slow@FAKE_SLEEP=20,FAKE_UNTIL=other,FAKE_GRACE=3'
  rc=$(steal_case steal-fix "$slow_f	A	3090	30	$d" "steal-fix	A	3090	30	$d" "steal-lib	A	3090	30	$d" \
    "plain-any	B	a6000	10	$d" -- "$slow_f" steal-fix steal-lib plain-any)
  printf '%s' "$rc" > "$t/rc-steal-fix"
  rc_ok 'steal: fixed items end green' steal-fix 4
  want_row 'steal: the fixed 3090 item stays in lane A on the 3090' "$t/fake-state/box.log" '$1 ~ /BLOOMERY_GATE_CARD=3090/ && $2 ~ /gen_f$/'
  want 'steal: its run.log line names lane A' "$t/target/c-steal-fix/run.log" '^steal-fix rc=0 [0-9]+s try=1 lane=A( |$)'
  want 'steal: the fixed no-card item stays in lane A' "$t/target/c-steal-fix/run.log" '^steal-lib rc=0 [0-9]+s try=1 lane=A( |$)'
  want_no_row 'steal: no fixed call reached the A6000' "$t/fake-state/box.log" '$1 ~ /BLOOMERY_GATE_CARD=a6000/ && $2 ~ /gen_f$/'
  # A held timing lease keeps every item off the A6000: the lease is up from the start, lane A's first
  # item holds its lane until the thief has probed (then 2 s more), and the owner — whose own items a
  # real box would hold in the guard — runs the balanced item on the 3090.
  local slow_l='steal-slow@FAKE_SLEEP=20,FAKE_UNTIL=probe,FAKE_GRACE=2'
  rc=$(LEASE_UP=1 steal_case steal-lease "$slow_l	A	3090	30	$d" "host	B	none	70	$d" "steal-bal	A	3090	60	$d" \
    "plain-any	B	a6000	10	$d" -- "$slow_l" host steal-bal plain-any)
  printf '%s' "$rc" > "$t/rc-steal-lease"
  rc_ok 'steal: a held timing lease ends green' steal-lease 4
  want 'steal: the thief probed the lease' "$t/fake-state/probe.log" '^probe$'
  want_row 'steal: the balanced item stayed off the A6000' "$t/fake-state/box.log" '$1 ~ /BLOOMERY_GATE_CARD=3090/ && $2 ~ /gen_b$/'
  want_no_row 'steal: no call reached the A6000 under the lease' "$t/fake-state/box.log" '$1 ~ /BLOOMERY_GATE_CARD=a6000/ && $2 ~ /gen_[bf]$/'
  # The dry run names the movable items (its own times rows, so the balance is fixed).
  printf '%s\n' 'steal-slow@FAKE_SLEEP=3	A	3090	30	2026-09-27T10:00:00+0900' \
    'host	B	none	70	2026-09-27T10:00:00+0900' 'steal-bal	A	3090	60	2026-09-27T10:00:00+0900' > "$t/times-dry.tsv"
  out=$(BLOOMERY_GATE_TIMES=$t/times-dry.tsv "${gb[@]}" --dry-run 'steal-slow@FAKE_SLEEP=3' host steal-bal 2>&1) \
    || fail 'steal: the dry run of a movable pair failed' "$out"
  if grep -A3 '^lane A  steal-bal' <<< "$out" | grep -q 'movable'; then pass 'steal: the dry run names the balanced item movable'
  else fail 'steal: the dry run names the balanced item movable' "no movable line under steal-bal"; fi
  if grep -A3 '^lane A  steal-slow' <<< "$out" | grep -q 'movable'; then fail 'steal: the dry run leaves a fixed item unmarked' "a movable line under steal-slow"
  else pass 'steal: the dry run leaves a fixed item unmarked'; fi
  # The tier. solo-real is solo in the real tier and balanced in the fixture tier; `v41-load` pins lane A in the real tier only.
  check 'tier: a solo-real recipe is alone in the real tier' 0 "^sr-any	X	solo	a6000	sr-any: \[group\('solo-real'\)\]" "${gb[@]}" --classes
  check 'tier: … and balanced over both cards in the fixture tier' 0 "^sr-any	F	balanced	3090,a6000	sr-any: \[group\('solo-real'\)\]" \
    "${gb[@]}" --tier fixture --classes
  check 'tier: a solo-real v41-load recipe is alone in the real tier (solo wins)' 0 '^sr-v41	X	solo	a6000	' "${gb[@]}" --classes
  check 'tier: … and balanced in the fixture tier, the v41-load group pinning no lane' 0 '^sr-v41	F	balanced	3090,a6000	' "${gb[@]}" --tier fixture --classes
  check 'tier: a plain v41-load recipe stays in lane A in the real tier' 0 '^v41-any	A	fixed	3090	' "${gb[@]}" --classes
  check 'tier: … and is balanced in the fixture tier, which takes no load lock' 0 '^v41-any	F	balanced	3090,a6000	' "${gb[@]}" --tier fixture --classes
  check 'tier: a 3090-only v41-load recipe stays in lane A in the fixture tier' 0 '^v41-a	A	fixed	3090	' "${gb[@]}" --tier fixture --classes
  check 'tier: a solo recipe is alone in both tiers' 0 '^v41-solo	X	solo	3090	' "${gb[@]}" --tier fixture --classes
  check 'tier: the real tier names nothing in the box env: lane X, no BLOOMERY_TIER' 0 \
    "^lane X  sr-any +BLOOMERY_BOX_ENV='BLOOMERY_GATE_CARD=a6000' just sr-any$" "${gb[@]}" --dry-run sr-any host
  check 'tier: the fixture tier passes BLOOMERY_TIER=fixture to the item, which lane A or B runs' 0 \
    "^lane [AB]  sr-any +BLOOMERY_BOX_ENV='BLOOMERY_TIER=fixture BLOOMERY_GATE_CARD=(3090|a6000)' just sr-any$" "${gb[@]}" --tier fixture --dry-run sr-any host
  out=$("${gb[@]}" --dry-run sr-any host plain-any 2>&1) || fail 'tier: the real dry run failed' "$out"
  if grep -q BLOOMERY_TIER <<< "$out"; then fail 'tier: the real tier leaves BLOOMERY_TIER out of every command' "a line names it"
  else pass 'tier: the real tier leaves BLOOMERY_TIER out of every command'; fi
  check 'tier: BLOOMERY_TIER in the environment is the batch tier' 0 "^lane [AB]  sr-any +BLOOMERY_BOX_ENV='BLOOMERY_TIER=fixture " \
    env BLOOMERY_TIER=fixture "${gb[@]}" --dry-run sr-any
  check 'tier: a BLOOMERY_TIER entry of the caller box env is the batch tier too, once' 0 "^lane [AB]  sr-any +BLOOMERY_BOX_ENV='BLOOMERY_TIER=fixture BLOOMERY_GATE_CARD=" \
    env BLOOMERY_BOX_ENV=BLOOMERY_TIER=fixture "${gb[@]}" --dry-run sr-any
  check 'tier: --tier other than real or fixture: 64, named' 64 "^gate-batch: --tier is real or fixture, got 'both'" "${gb[@]}" --tier both --dry-run host
  check 'tier: --tier without a value: 64' 64 '^gate-batch: --tier needs real or fixture' "${gb[@]}" --dry-run host --tier
  check 'tier: the flag and the environment naming two tiers: 64, named' 64 "^gate-batch: --tier names the tier 'fixture' and BLOOMERY_TIER in the environment 'real'" \
    env BLOOMERY_TIER=real "${gb[@]}" --tier fixture --dry-run host
  check 'tier: the flag and the box env naming two tiers: 64, named' 64 "^gate-batch: --tier names the tier 'real' and BLOOMERY_BOX_ENV's BLOOMERY_TIER 'fixture'" \
    env BLOOMERY_BOX_ENV=BLOOMERY_TIER=fixture "${gb[@]}" --tier real --dry-run host
  check "tier: an item's own BLOOMERY_TIER is refused: the tier is the batch's" 65 "sets BLOOMERY_TIER; the tier is the batch's" \
    "${gb[@]}" --dry-run 'plain-any@BLOOMERY_TIER=fixture'
  # The times file keeps the tiers apart: a fixture item's row is an entry of its own.
  printf '%s\n' 'plain-any	B	a6000	10	2026-09-27T10:00:00+0900' 'plain-any@BLOOMERY_TIER=fixture	B	a6000	77	2026-09-27T10:00:00+0900' > "$t/times-tier.tsv"
  check 'tier: a real item expects its real rows' 0 'balanced, expected 10 s \(median of 1 row\)' env "BLOOMERY_GATE_TIMES=$t/times-tier.tsv" "${gb[@]}" --dry-run plain-any
  check 'tier: a fixture item expects its own rows' 0 'balanced, expected 77 s \(median of 1 row\)' env "BLOOMERY_GATE_TIMES=$t/times-tier.tsv" "${gb[@]}" --tier fixture --dry-run plain-any
  # A solo-real recipe the fixture tier could not place is a named error.
  cp "$t/justfile" "$t/justfile.good"
  printf '%s\n' '' "[group('solo-real')]" "[group('solo')]" 'sr-solo:' "    ./tools/box.sh 'BLOOMERY_GATE_CARD=\${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh other'" >> "$t/justfile"
  check 'tier: solo-real with solo: a named error' 65 "recipe sr-solo: \[group\('solo-real'\)\] with \[group\('solo'\)\]" "${gb[@]}" --classes
  cp "$t/justfile.good" "$t/justfile"
  printf '%s\n' '' "[group('solo-real')]" 'sr-3090:' "    ./tools/box.sh 'bash tools/gpu-gate.sh other'" >> "$t/justfile"
  check 'tier: solo-real with a 3090-only gpu-gate.sh call: a named error' 65 "recipe sr-3090: \[group\('solo-real'\)\] with its tools/gpu-gate.sh calls are not all .*(forms: 3090)" "${gb[@]}" --classes
  cp "$t/justfile.good" "$t/justfile"
  printf '%s\n' '' "[group('solo-real')]" 'sr-none:' "    ./tools/box.sh 'bash tools/gate.sh -p y'" >> "$t/justfile"
  check 'tier: solo-real with no gpu-gate.sh call: a named error' 65 "recipe sr-none: \[group\('solo-real'\)\] with its tools/gpu-gate.sh calls are not all .*(forms: none)" "${gb[@]}" --classes
  cp "$t/justfile.good" "$t/justfile"
  printf '%s\n' '' "[group('solo-real')]" 'sr-both:' "    BLOOMERY_CARD=both ./tools/box.sh 'BLOOMERY_GATE_CARD=\${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh other'" >> "$t/justfile"
  check 'tier: solo-real with a box.sh card pick: a named error' 65 "recipe sr-both: \[group\('solo-real'\)\] with BLOOMERY_CARD=both .*\(use solo\)" "${gb[@]}" --classes
  cp "$t/justfile.good" "$t/justfile"
  printf '%s\n' '' "[group('host')]" "[group('solo-real')]" 'sr-host:' "    ./tools/box.sh 'bash tools/gate.sh --oxide -p hd'" >> "$t/justfile"
  check 'tier: solo-real with host: a named error' 65 "recipe sr-host: \[group\('host'\)\] with \[group\('solo-real'\)\]" "${gb[@]}" --classes
  cp "$t/justfile.good" "$t/justfile"
  # A solo-real recipe's flags are its own arm: another recipe handing them is refused in either tier, and the recipe's own item
  # with them is not.
  printf '%s\n' '' "[group('solo-real')]" 'sr-arm *ARGS:' "    ./tools/box.sh 'BLOOMERY_GATE_CARD=\${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gen_y --sr-arm {{ARGS}}'" >> "$t/justfile"
  check "tier: a solo-real recipe's arm handed through another recipe is refused in the fixture tier" 65 \
    "'v41-a:--sr-arm' hands gen_y --sr-arm, the arm of the solo recipe sr-arm" "${gb[@]}" --tier fixture --dry-run 'v41-a:--sr-arm'
  check '  … and in the real tier' 65 "'v41-a:--sr-arm' hands gen_y --sr-arm, the arm of the solo recipe sr-arm" "${gb[@]}" --dry-run 'v41-a:--sr-arm'
  check "  … while the recipe's own item with its flag runs in lane A or B in the fixture tier" 0 '^lane [AB]  sr-arm ' "${gb[@]}" --tier fixture --dry-run 'sr-arm:--sr-arm'
  check '  … and alone in the real tier' 0 '^lane X  sr-arm ' "${gb[@]}" --dry-run 'sr-arm:--sr-arm'
  cp "$t/justfile.good" "$t/justfile"
  # Run: a fixture batch's items carry the tier in their box env, every key call (the plan's and the recheck's) sees it, the
  # clauses an item left to the real tier are counted (its line, then the DONE line); the real tier does none of that.
  rc=$(steal_case tier-fx "steal-bal	B	a6000	10	$d" -- --tier fixture 'steal-bal@FAKE_DEFER=2')
  printf '%s' "$rc" > "$t/rc-tier-fx"
  rc_ok 'tier run: a fixture batch ends green' tier-fx 1
  want_row 'tier run: its box call carries BLOOMERY_TIER=fixture' "$t/fake-state/box.log" '$1 ~ /BLOOMERY_TIER=fixture/ && $2 ~ /gen_b$/'
  want 'tier run: its item line counts the deferred clauses' "$t/target/c-tier-fx/run.log" '^steal-bal rc=0 [0-9]+s try=1 lane=[AB] deferred=2 ledger=recorded item=steal-bal@FAKE_DEFER=2$'
  want 'tier run: the DONE line totals them' "$t/target/c-tier-fx/run.log" '^DONE total=1 red=0 skipped=0 wall=[0-9]+s laneA=[0-9]+s laneB=[0-9]+s deferred=2$'
  want 'tier run: the times row is an entry of its own' "$t/times-tier-fx.tsv" "$(printf '^steal-bal@FAKE_DEFER=2,BLOOMERY_TIER=fixture\t')"
  n=$((n + 1))
  if [ "$(grep -c . "$t/fake-state/keycalls.log")" -ge 2 ] && ! grep -qv 'BLOOMERY_TIER=fixture' "$t/fake-state/keycalls.log"; then echo 'ok tier run: the plan'"'"'s key call and the recheck both carry the tier'
  else bad=$((bad + 1)); echo "FAIL tier run: a key call lacks the tier: $(cat "$t/fake-state/keycalls.log" 2>&1)"; fi
  # A fixture batch whose gates left nothing to the real tier (none is converted yet) still ends green, its total 0.
  rc=$(steal_case tier-f0 "steal-bal	B	a6000	10	$d" -- --tier fixture steal-bal)
  printf '%s' "$rc" > "$t/rc-tier-f0"
  rc_ok 'tier run: a fixture batch whose gates defer nothing ends green' tier-f0 1
  want 'tier run: … its DONE line totals 0' "$t/target/c-tier-f0/run.log" '^DONE total=1 red=0 skipped=0 wall=[0-9]+s laneA=[0-9]+s laneB=[0-9]+s deferred=0$'
  want_not 'tier run: … and its item line counts none' "$t/target/c-tier-f0/run.log" '^steal-bal .* deferred='
  rc=$(steal_case tier-re "steal-bal	B	a6000	10	$d" -- 'steal-bal@FAKE_DEFER=2')
  printf '%s' "$rc" > "$t/rc-tier-re"
  rc_ok 'tier run: the same item in the real tier ends green' tier-re 1
  want_no_row 'tier run: … and none of its calls names the tier' "$t/fake-state/box.log" '$1 ~ /BLOOMERY_TIER/'
  want_not 'tier run: … and the DONE line counts no deferrals' "$t/target/c-tier-re/run.log" '^DONE .* deferred='
  want 'tier run: … though its line does' "$t/target/c-tier-re/run.log" '^steal-bal rc=0 [0-9]+s try=1 lane=[AB] deferred=2 ledger=recorded '
  # real-only (the header's paragraph): a no-op in the real tier, a deferral in the fixture tier, never a lane, a key or a call.
  check 'real-only: in the real tier the recipe is placed by its calls, as without the dependency' 0 \
    '^ro-any	F	balanced	3090,a6000	ro-any: real-only call; ro-any: gpu-gate.sh any' "${gb[@]}" --classes
  check '  … and a solo recipe stays alone' 0 "^ro-solo	X	solo	3090	" "${gb[@]}" --classes
  check 'real-only: in the fixture tier the recipe is deferred, in no lane' 0 '^ro-any	D	real-only	-	ro-any: real-only call' \
    "${gb[@]}" --tier fixture --classes
  check '  … a solo one too (the group beats solo there)' 0 '^ro-solo	D	real-only	-	' "${gb[@]}" --tier fixture --classes
  check '  … and one that reaches the guard through another recipe, by closure' 0 '^ro-chain	D	real-only	-	' "${gb[@]}" --tier fixture --classes
  check 'real-only: the real dry run places it as usual' 0 '^lane [AB]  ro-any ' "${gb[@]}" --dry-run ro-any
  out=$("${gb[@]}" --tier fixture --dry-run ro-any ro-solo plain-any 2>&1) || fail 'real-only: the fixture dry run failed' "$out"
  if grep -q '^lane -  ro-any ' <<< "$out" && grep -q '^lane -  ro-solo ' <<< "$out" && grep -q '^lane [AB]  plain-any '  <<< "$out" \
    && grep -q '2 item(s) deferred to the real tier (2 real-only, 0 of a family with no fixture), in no lane (DONE: deferred=+2, and the list .*/deferred.list): ro-any ro-solo' <<< "$out" \
    && grep -q 'dry run — 1 items (and 2 deferred to the real tier: 2 real-only, 0 of a family with no fixture)' <<< "$out"; then pass 'real-only: the fixture dry run names both deferred items beside the one that runs'
  else fail 'real-only: the fixture dry run names both deferred items beside the one that runs' "$out"; fi
  check 'real-only: a deferred item adds nothing to the lane sums (plain-any alone: its 45 s default, no fixture row)' 0 'predicted laneA=0s laneB=45s laneX=0s wall=45s' \
    env "BLOOMERY_GATE_TIMES=$t/times.tsv" "${gb[@]}" --tier fixture --dry-run ro-any ro-solo plain-any
  check "real-only: an item's ARGS are still checked by just" 65 'takes no arguments' "${gb[@]}" --tier fixture --dry-run 'ro-any:--x'
  rc=$(steal_case ro-fx "steal-bal	B	a6000	10	$d" -- --tier fixture ro-any 'steal-bal@FAKE_DEFER=2' ro-solo)
  printf '%s' "$rc" > "$t/rc-ro-fx"
  rc_ok 'real-only run: a fixture batch with two deferred items ends green' ro-fx 3
  want 'real-only run: the deferred item has its own line' "$t/target/c-ro-fx/run.log" '^ro-any rc=deferred real-only lane=- deferred=1$'
  want '  … a solo one too' "$t/target/c-ro-fx/run.log" '^ro-solo rc=deferred real-only lane=- deferred=1$'
  want 'real-only run: the DONE line counts them in deferred= (with the clauses) and in real_only=' "$t/target/c-ro-fx/run.log" \
    '^DONE total=3 red=0 skipped=0 wall=[0-9]+s laneA=[0-9]+s laneB=[0-9]+s deferred=4 real_only=2 deferred_list=[^ ]*/deferred.list$'
  want_no_row 'real-only run: the batch never called a deferred recipe' "$t/fake-state/box.log" '$2 ~ /gen_ro$|gen_rs$/'
  want_no_row '  … and wrote no times row for it' "$t/times-ro-fx.tsv" '$1 ~ /^ro-/'
  want_not '  … nor a ledger record' "$t/rounds-ro-fx.tsv" '^[^	]*ro-(any|solo)'
  rc=$(steal_case ro-all "steal-bal	B	a6000	10	$d" -- --tier fixture ro-any ro-solo)
  printf '%s' "$rc" > "$t/rc-ro-all"
  rc_ok 'real-only run: a batch of deferred items only ends green' ro-all 2
  want 'real-only run: … its DONE line' "$t/target/c-ro-all/run.log" '^DONE total=2 red=0 skipped=0 wall=[0-9]+s laneA=0s laneB=0s deferred=2 real_only=2 deferred_list=[^ ]*/deferred.list$'
  want 'real-only run: … says nothing ran on the box' "$t/out-ro-all.log" '^gate-batch: every item is deferred to the real tier — nothing runs on the box, so no lease check$'
  want_no_row 'real-only run: … and made no box call at all' "$t/fake-state/box.log" '1'
  rc=$(steal_case ro-real "steal-bal	B	a6000	10	$d" -- ro-any)
  printf '%s' "$rc" > "$t/rc-ro-real"
  rc_ok 'real-only run: the real tier runs the recipe' ro-real 1
  want_row '  … through its box call' "$t/fake-state/box.log" '$2 ~ /gen_ro$/'
  want_not '  … and its DONE line has no real_only=' "$t/target/c-ro-real/run.log" '^DONE .*real_only='
  # no-fixture (the header's deferred paragraph, kind 2): a family the fixture table of tools/ref/ref-paths.sh does not name is deferred
  # in the fixture tier, never run and never red; the table is read from that file, so a family it names runs.
  check 'no-fixture: in the real tier a recipe of a family with no fixture is placed as usual' 0 '^x-glm	X	solo	' "${gb[@]}" --classes
  check '  … in the fixture tier it is deferred, named by its family' 0 '^x-glm	D	no-fixture	-	.*family glm5next: no row in the fixture table of tools/ref/ref-paths.sh \(box.sh exits 66\)' \
    "${gb[@]}" --tier fixture --classes
  check '  … a family with a fixture directory runs (qwen4exp)' 0 '^x-q	X	solo	' "${gb[@]}" --tier fixture --classes
  check '  … a family whose real file stands runs (the default profile)' 0 '^x-none	X	solo	' "${gb[@]}" --tier fixture --classes
  check '  … a family named by closure runs (deepseek41)' 0 '^x-v41b	X	solo	' "${gb[@]}" --tier fixture --classes
  check '  … and a recipe that is both is real-only (the declared fact wins)' 0 '^ro-glm	D	real-only	-	' "${gb[@]}" --tier fixture --classes
  out=$("${gb[@]}" --tier fixture --dry-run x-glm plain-any 2>&1) || fail 'no-fixture: the fixture dry run failed' "$out"
  if grep -q '^lane -  x-glm ' <<< "$out" && grep -q 'deferred to the real tier (no-fixture), never run in the fixture tier — ' <<< "$out" \
    && grep -q 'dry run — 1 items (and 1 deferred to the real tier: 0 real-only, 1 of a family with no fixture)' <<< "$out"; then pass 'no-fixture: the fixture dry run names the family item as deferred (no-fixture)'
  else fail 'no-fixture: the fixture dry run names the family item as deferred (no-fixture)' "$out"; fi
  rc=$(steal_case nf-fx "steal-bal	B	a6000	10	$d" -- --tier fixture ro-any x-glm 'x-glm:--x' steal-bal)
  printf '%s' "$rc" > "$t/rc-nf-fx"
  check_nf=$(cat "$t/out-nf-fx.log" 2>&1)
  if [ "$rc" = 65 ] && grep -q "x-glm takes no arguments" <<< "$check_nf"; then pass 'no-fixture: an ARGS error of a deferred item is still named, and nothing ran'
  else fail 'no-fixture: an ARGS error of a deferred item is still named, and nothing ran' "rc $rc: $check_nf"; fi
  rc=$(steal_case nf-fx "steal-bal	B	a6000	10	$d" -- --tier fixture ro-any x-glm steal-bal)
  printf '%s' "$rc" > "$t/rc-nf-fx"
  rc_ok 'no-fixture run: a fixture batch with a real-only and a no-fixture item ends green' nf-fx 3
  want 'no-fixture run: the family item has its own line' "$t/target/c-nf-fx/run.log" '^x-glm rc=deferred no-fixture lane=- deferred=1$'
  want '  … beside the real-only one' "$t/target/c-nf-fx/run.log" '^ro-any rc=deferred real-only lane=- deferred=1$'
  want 'no-fixture run: the DONE line counts both kinds and names the list' "$t/target/c-nf-fx/run.log" \
    '^DONE total=3 red=0 skipped=0 wall=[0-9]+s laneA=[0-9]+s laneB=[0-9]+s deferred=2 real_only=1 no_fixture=1 deferred_list=[^ ]*/c-nf-fx/deferred.list$'
  want_no_row 'no-fixture run: the batch never called either recipe' "$t/fake-state/box.log" '$2 ~ /gen_ro$|xg$/'
  n=$((n + 1))
  if [ "$(grep -v '^#' "$t/target/c-nf-fx/deferred.list" | paste -sd, -)" = 'ro-any,x-glm' ] && grep -q '^# real-only: ' "$t/target/c-nf-fx/deferred.list" \
    && grep -q '^# no-fixture: ' "$t/target/c-nf-fx/deferred.list"; then echo 'ok no-fixture run: DIR/deferred.list holds the two items, real-only first, with a comment for each kind'
  else bad=$((bad + 1)); echo "FAIL no-fixture run: DIR/deferred.list: $(cat "$t/target/c-nf-fx/deferred.list" 2>&1)"; fi
  check 'no-fixture run: a real-tier batch takes exactly that list (both items placed as usual)' 0 '^lane X  x-glm ' "${gb[@]}" --dry-run --list "$t/target/c-nf-fx/deferred.list"
  out=$("${gb[@]}" --dry-run --list "$t/target/c-nf-fx/deferred.list" 2>&1) || fail 'no-fixture run: the real dry run of the list failed' "$out"
  if grep -q '^lane [AB]  ro-any ' <<< "$out" && grep -q 'dry run — 2 items,' <<< "$out"; then pass 'no-fixture run: … two items, none deferred'
  else fail 'no-fixture run: … two items, none deferred' "$out"; fi
  rc=$(steal_case nf-item "steal-bal	B	a6000	10	$d" -- --tier fixture 'ro-any@FAKE_X=1' steal-bal)
  printf '%s' "$rc" > "$t/rc-nf-item"
  want 'no-fixture run: a deferred item with env keeps its item text in the list' "$t/target/c-nf-item/deferred.list" '^ro-any@FAKE_X=1$'
  rc=$(steal_case nf-none "steal-bal	B	a6000	10	$d" -- --tier fixture steal-bal)
  printf '%s' "$rc" > "$t/rc-nf-none"
  if [ ! -e "$t/target/c-nf-none/deferred.list" ]; then pass 'no-fixture run: a batch that deferred nothing writes no list'
  else fail 'no-fixture run: a batch that deferred nothing writes no list' "$(cat "$t/target/c-nf-none/deferred.list")"; fi
  want_not '  … and its DONE line has none of the deferral fields' "$t/target/c-nf-none/run.log" '^DONE .*(real_only|no_fixture|deferred_list)='
  # The fixture table is ref-paths.sh's: a table this reader cannot read, or whose default is not `none`, is a named error.
  cp "$t/tools/ref/ref-paths.sh" "$t/ref-paths.good"
  sed -i.bak '/__fixture_dir=/d' "$t/tools/ref/ref-paths.sh" && rm -f "$t/tools/ref/ref-paths.sh.bak"
  check 'no-fixture: a ref-paths.sh with no fixture table: a named error' 65 'no fixture table .* tools/gate-batch.sh reads which families have a fixture there' "${gb[@]}" --tier fixture --classes
  check '  … the real tier never reads it' 0 '^x-glm	X	solo	' "${gb[@]}" --classes
  cp "$t/ref-paths.good" "$t/tools/ref/ref-paths.sh"
  sed -i.bak 's/\*) __fixture_dir=none ;;/*) __fixture_dir=v41 ;;/' "$t/tools/ref/ref-paths.sh" && rm -f "$t/tools/ref/ref-paths.sh.bak"
  check "no-fixture: a table whose *) row is not none: a named error" 65 "the fixture table's .\*\). row is 'v41', not none" "${gb[@]}" --tier fixture --classes
  cp "$t/ref-paths.good" "$t/tools/ref/ref-paths.sh"
  # The real table: the tree's own ref-paths.sh parses, and the family it names runs (a converted V4.1 gate is balanced, not deferred).
  check "no-fixture: the tree's own tools/ref/ref-paths.sh is a table this runner reads" 0 '^gate-gpu-ds41-step	F	balanced	' bash "$self" --tier fixture --classes
  # The class's own errors, wherever the recipe sits.
  printf '%s\n' '' 'ro-other:' "    ./tools/box.sh 'bash tools/ref/real-only.sh ro-any && bash tools/gate.sh -p z'" >> "$t/justfile"
  check "real-only: a call that names another recipe: a named error" 65 "recipe ro-other: its real-only call names ro-any, not itself" "${gb[@]}" --classes
  cp "$t/justfile.good" "$t/justfile"
  printf '%s\n' '' 'ro-late:' "    ./tools/box.sh 'cargo build && bash tools/ref/real-only.sh ro-late && bash tools/gate.sh -p z'" >> "$t/justfile"
  check 'real-only: a box command that does not open with the call: a named error' 65 \
    'recipe ro-late: a real-only recipe opens every box.sh command with .bash tools/ref/real-only.sh ro-late && …. — 1 single-quoted box.sh command\(s\), 1 call\(s\), 1 not opening with it' "${gb[@]}" --classes
  cp "$t/justfile.good" "$t/justfile"
  printf '%s\n' '' 'ro-two:' "    ./tools/box.sh 'bash tools/ref/real-only.sh ro-two && bash tools/gate.sh -p a'" "    ./tools/box.sh 'bash tools/gate.sh -p b'" >> "$t/justfile"
  check 'real-only: a second box command with no call: a named error' 65 'recipe ro-two: .* 2 single-quoted box.sh command\(s\), 1 call\(s\), 1 not opening with it' "${gb[@]}" --classes
  cp "$t/justfile.good" "$t/justfile"
  printf '%s\n' '' 'ro-bare:' "    bash tools/ref/real-only.sh ro-bare" >> "$t/justfile"
  check 'real-only: a call outside a single-quoted box command: a named error' 65 'recipe ro-bare: .* 0 single-quoted box.sh command\(s\), 1 call\(s\)' "${gb[@]}" --classes
  cp "$t/justfile.good" "$t/justfile"
  mv "$t/tools/ref/real-only.sh" "$t/tools/ref/real-only.sh.off"
  check 'real-only: a call to a script that is not in the tree: a named error' 65 'recipe ro-any: calls tools/ref/real-only.sh, which is not in the tree' "${gb[@]}" --classes
  mv "$t/tools/ref/real-only.sh.off" "$t/tools/ref/real-only.sh"
  printf '%s\n' '' "[group('solo-real')]" 'ro-sr:' "    ./tools/box.sh 'bash tools/ref/real-only.sh ro-sr && BLOOMERY_GATE_CARD=\${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh other'" >> "$t/justfile"
  check 'real-only: with solo-real: a named error' 65 'recipe ro-sr: \[group\(.solo-real.\)\] with a real-only call' "${gb[@]}" --classes
  cp "$t/justfile.good" "$t/justfile"
  # The guard script itself: it stops the fixture tier by name before any box call, and only that tier.
  local ro_sh
  ro_sh=$(dirname "$self")/ref/real-only.sh
  check 'real-only guard: BLOOMERY_TIER=fixture stops it by name, exit 66' 66 '^gate-x: real-only: .*exit 66' env BLOOMERY_TIER=fixture bash "$ro_sh" gate-x
  check '  … the real tier passes' 0 '^$' env BLOOMERY_TIER=real bash "$ro_sh" gate-x
  check '  … and so does no tier, or an empty one' 0 '^$' env -u BLOOMERY_TIER bash "$ro_sh" gate-x
  check '  … a tier that is neither is refused, 64' 64 "got 'both'" env BLOOMERY_TIER=both bash "$ro_sh" gate-x
  check '  … a call with no recipe name is a usage error, 64' 64 'usage: bash tools/ref/real-only.sh <recipe>' env BLOOMERY_TIER=fixture bash "$ro_sh"
  # One card by box.sh's pick (solo-real, BLOOMERY_CARD=a6000): alone in the real tier, fixed to lane B in the fixture tier; two cards stay refused (sr-both above).
  check 'one card: a solo-real recipe with BLOOMERY_CARD=a6000 is alone in the real tier' 0 '^sr-a6000	X	solo	-	' "${gb[@]}" --classes
  check '  … and fixed to the A6000 lane in the fixture tier' 0 '^sr-a6000	B	fixed	-	' "${gb[@]}" --tier fixture --classes
  out=$("${gb[@]}" --tier fixture --dry-run sr-a6000 plain-any host 2>&1) || fail 'one card: the fixture dry run failed' "$out"
  if grep -q '^lane B  sr-a6000 ' <<< "$out" && grep -q "^lane B  sr-a6000 .*BLOOMERY_BOX_ENV='BLOOMERY_TIER=fixture' just sr-a6000$" <<< "$out" \
    && ! grep -A3 '^lane B  sr-a6000 ' <<< "$out" | grep -q movable; then pass 'one card: the fixture dry run puts it in lane B with no card forced, and not movable'
  else fail 'one card: the fixture dry run puts it in lane B with no card forced, and not movable' "$out"; fi
  rc=$(steal_case sr-one "plain-any	B	a6000	10	$d" "host	B	none	5	$d" "sr-a6000	B	a6000	8	$d" -- --tier fixture host plain-any sr-a6000)
  printf '%s' "$rc" > "$t/rc-sr-one"
  rc_ok 'one card: a fixture batch with it ends green' sr-one 3
  want 'one card: it ran in lane B' "$t/target/c-sr-one/run.log" '^sr-a6000 rc=0 [0-9]+s try=1 lane=B( |$)'
  want_row '  … with the tier and no card of the batch in its env' "$t/fake-state/box.log" '$1 ~ /BLOOMERY_TIER=fixture/ && $1 !~ /BLOOMERY_GATE_CARD/ && $2 ~ /gen_sa/'
  want 'one card: its times row names lane B and the A6000' "$t/times-sr-one.tsv" "$(printf '^sr-a6000@BLOOMERY_TIER=fixture\tB\ta6000\t')"
  # The GPU hold (the header). Its writer is the real tools/gpu-gate.sh on the fixture (the fake box above), so these cases cross the
  # batch, box.sh's two flags and the box-side file. A --ledger batch puts the hold up for its whole run — two lanes here: lane B's
  # item ends first and its lane ends, and the hold must still be up when lane A's slow item ends — with a heartbeat (1 s here),
  # each item running with its batch's owner in the call's env only (never the key, the ledger or the times row: the caller's own
  # box env KEEP=1 makes the key calls non-empty), and the hold down at the end.
  local items_only='$3 != "true" && $3 !~ /box-manifest/'
  local slow_h='steal-slow@FAKE_SLEEP=5'
  rc=$(LFLAG=--ledger BLOOMERY_GPU_HOLD_BEAT=1 BLOOMERY_BOX_ENV=KEEP=1 steal_case hold-up "$slow_h	A	3090	30	$d" "plain-any	B	a6000	10	$d" -- "$slow_h" plain-any)
  printf '%s' "$rc" > "$t/rc-hold-up"
  rc_ok 'hold: a --ledger batch of two lanes ends green' hold-up 2
  want 'hold: it names the hold up, and down' "$t/out-hold-up.log" '^gate-batch: gpu hold: up as gb_[A-Za-z0-9_]+ \(.*batch\.gpuhold, refreshed every 1 s, log .*gpuhold\.log\)'
  want '  … and down' "$t/out-hold-up.log" '^gate-batch: gpu hold: down$'
  want_row 'hold: both items ran, each under its own batch'"'"'s hold file' "$t/fake-state/seen.log" "$items_only"' { c++ } END { if (c == 2) print c }'
  want_no_row '  … none without its owner in the env and the hold in place' "$t/fake-state/seen.log" "$items_only"' && index($2, "owner=" $1 " since=") != 1'
  want_no_row '  … every item call carries BLOOMERY_BATCH_OWNER in its box env' "$t/fake-state/box.log" '$2 != "true" && $2 !~ /box-manifest/ && $1 !~ /BLOOMERY_BATCH_OWNER=gb_/'
  want '  … and the caller'"'"'s own box env reached the key calls' "$t/fake-state/keycalls.log" '^KEEP=1$'
  want_not '  … the owner is in no key call' "$t/fake-state/keycalls.log" 'BATCH_OWNER'
  want '  … the ledger recorded both' "$t/lead-hold-up.tsv" '^stub-plain-any'
  want_not '  … the owner is in no ledger line' "$t/lead-hold-up.tsv" 'BATCH_OWNER'
  want_not '  … nor in a times row' "$t/times-hold-up.tsv" 'BATCH_OWNER'
  want_not '  … and run.log carries no hold line' "$t/target/c-hold-up/run.log" 'gpuhold|gpu hold|BATCH_OWNER'
  want_no_row '  … every hold call went read-only, with no guard wait' "$t/fake-state/hold.log" '$0 !~ /^ro=1 wait=0 bash tools\/gpu-gate\.sh --gpuhold /'
  want_row '  … one up' "$t/fake-state/hold.log" '/--gpuhold up gb_/ { c++ } END { if (c == 1) print c }'
  want_row '  … at least two beats (5 s of an item at 1 s)' "$t/fake-state/hold.log" '/--gpuhold beat gb_/ { c++ } END { if (c >= 2) print c }'
  want_row '  … one down' "$t/fake-state/hold.log" '/--gpuhold down gb_/ { c++ } END { if (c == 1) print c }'
  want '  … the heartbeat logged its beats' "$t/target/c-hold-up/gpuhold.log" 'beat ok$'
  want_row '  … the hold was up when each item ended (a finished lane did not take it down)' "$t/fake-state/end.log" '$1 == "present" && $2 != "true" && $2 !~ /box-manifest/ { c++ } END { if (c == 2) print c }'
  want_no_row '  … and absent at none of them' "$t/fake-state/end.log" '$1 != "present" && $2 != "true" && $2 !~ /box-manifest/'
  if [ ! -e "$t/fake-state/batch.gpuhold" ]; then pass '  … and the hold file is gone once the batch ends'
  else fail '  … and the hold file is gone once the batch ends' "$(cat "$t/fake-state/batch.gpuhold")"; fi
  hp=$(cat "$t/target/c-hold-up/gpuhold.pid" 2> /dev/null || echo 0)
  if [ "$hp" -gt 1 ] && ! kill -0 "$hp" 2> /dev/null; then pass '  … and the heartbeat is gone'
  else fail '  … and the heartbeat is gone' "pid $hp answers kill -0"; fi
  # No hold: a --round-ledger batch, and a --ledger batch that opts out. Every item ran with no owner and no hold file, no call
  # carried an owner, and no hold call and no heartbeat pid exist.
  rc=$(steal_case hold-round "plain-any	B	a6000	10	$d" -- plain-any)
  printf '%s' "$rc" > "$t/rc-hold-round"
  rc_ok 'no hold: a --round-ledger batch ends green' hold-round 1
  want_row '  … its item ran' "$t/fake-state/seen.log" "$items_only"' { c++ } END { if (c == 1) print c }'
  want_no_row '  … with no owner and no hold file' "$t/fake-state/seen.log" "$items_only"' && ($1 != "none" || $2 != "nohold")'
  want_not '  … no call carries an owner' "$t/fake-state/box.log" 'BATCH_OWNER'
  if [ ! -e "$t/fake-state/hold.log" ] && [ ! -e "$t/target/c-hold-round/gpuhold.pid" ]; then pass '  … no hold call, no heartbeat pid'
  else fail '  … no hold call, no heartbeat pid' "hold.log or gpuhold.pid exists"; fi
  rc=$(LFLAG=--ledger steal_case hold-off "plain-any	B	a6000	10	$d" -- --no-gpu-hold plain-any)
  printf '%s' "$rc" > "$t/rc-hold-off"
  rc_ok '  … and a --ledger batch with --no-gpu-hold ends green' hold-off 1
  want_no_row '  … with no owner and no hold file' "$t/fake-state/seen.log" "$items_only"' && ($1 != "none" || $2 != "nohold")'
  want_not '  … no call carries an owner' "$t/fake-state/box.log" 'BATCH_OWNER'
  if [ ! -e "$t/fake-state/hold.log" ] && [ ! -e "$t/target/c-hold-off/gpuhold.pid" ]; then pass '  … no hold call, no heartbeat pid'
  else fail '  … no hold call, no heartbeat pid' "hold.log or gpuhold.pid exists"; fi
  # A batch that is red takes the hold down too (the EXIT trap: every exit).
  rc=$(LFLAG=--ledger steal_case hold-red "plain-any	B	a6000	10	$d" -- 'plain-any@FAKE_RC=1')
  printf '%s' "$rc" > "$t/rc-hold-red"
  if [ "$(cat "$t/rc-hold-red")" = 1 ] && grep -Eq '^DONE total=1 red=1 ' "$t/out-hold-red.log" && [ ! -e "$t/fake-state/batch.gpuhold" ]; then
    pass 'hold: a red batch (rc 1) takes it down'
  else fail 'hold: a red batch (rc 1) takes it down' "rc=$(cat "$t/rc-hold-red") hold file: $(cat "$t/fake-state/batch.gpuhold" 2>&1 | head -1)"; fi
  want_row '  … through the down call' "$t/fake-state/hold.log" '/--gpuhold down gb_/ { c++ } END { if (c == 1) print c }'
  # INT/TERM (stop): TERM to the batch's own pid, once its item runs.
  local slow_s='steal-slow@FAKE_SLEEP=20'
  printf '%s\n' "$slow_s	A	3090	30	$d" > "$t/times-hold-stop.tsv"
  rm -rf "$t/fake-state" "$t/target/c-hold-stop"
  BLOOMERY_GATE_TIMES=$t/times-hold-stop.tsv BLOOMERY_GATE_LEDGER=$t/lead-hold-stop.tsv BLOOMERY_GATE_ROUND_LEDGER=$t/rounds-hold-stop.tsv \
    BLOOMERY_GPU_HOLD_BEAT=1 "${gb[@]}" --out "$t/target/c-hold-stop" --ledger "$slow_s" > "$t/out-hold-stop.log" 2>&1 &
  sp=$!
  k=0
  while ! grep -q 'gen_s' "$t/fake-state/seen.log" 2> /dev/null && [ "$k" -lt 20 ]; do
    sleep 1
    k=$((k + 1))
  done
  held=$(head -1 "$t/fake-state/batch.gpuhold" 2> /dev/null || echo none)
  kill -TERM "$sp"
  rc=0
  wait "$sp" || rc=$?
  hp=$(cat "$t/target/c-hold-stop/gpuhold.pid" 2> /dev/null || echo 0)
  case $held in owner=gb_*) pass 'stop: the hold was up while the item ran' ;; *) fail 'stop: the hold was up while the item ran' "the hold file read '$held'" ;; esac
  if [ "$rc" = 130 ] && [ ! -e "$t/fake-state/batch.gpuhold" ] && [ "$hp" -gt 1 ] && ! kill -0 "$hp" 2> /dev/null; then
    pass '  … TERM to the batch ends it (130), takes the hold down and ends the heartbeat'
  else fail '  … TERM to the batch ends it (130), takes the hold down and ends the heartbeat' "rc=$rc hold: $(cat "$t/fake-state/batch.gpuhold" 2>&1 | head -1) heartbeat pid $hp"; fi
  want '  … and says it stopped' "$t/out-hold-stop.log" '^gate-batch: stopped;'
  # die: a lane killed from outside (its pid, as written under DIR) ends the batch through die (70), and the hold still comes down.
  rm -rf "$t/fake-state" "$t/target/c-hold-die"
  BLOOMERY_GATE_TIMES=$t/times-hold-stop.tsv BLOOMERY_GATE_LEDGER=$t/lead-hold-die.tsv BLOOMERY_GATE_ROUND_LEDGER=$t/rounds-hold-die.tsv \
    BLOOMERY_GPU_HOLD_BEAT=1 "${gb[@]}" --out "$t/target/c-hold-die" --ledger "$slow_s" > "$t/out-hold-die.log" 2>&1 &
  sp=$!
  k=0
  while ! grep -q 'gen_s' "$t/fake-state/seen.log" 2> /dev/null && [ "$k" -lt 20 ]; do
    sleep 1
    k=$((k + 1))
  done
  kill -TERM "$(cat "$t/target/c-hold-die/lane-A.pid")"
  rc=0
  wait "$sp" 2> /dev/null || rc=$?
  if [ "$rc" = 70 ] && grep -q 'ended without its sentinel' "$t/out-hold-die.log" && [ ! -e "$t/fake-state/batch.gpuhold" ]; then
    pass 'hold: a batch that ends through die (a lane killed: 70) takes it down'
  else fail 'hold: a batch that ends through die (a lane killed: 70) takes it down' "rc=$rc hold file: $(ls "$t/fake-state/batch.gpuhold" 2>&1)"; fi
  if [ -f "$t/target/c-hold-die/lane-A.child" ]; then kill -TERM "$(cat "$t/target/c-hold-die/lane-A.child")" 2> /dev/null || true; fi
  # An unclean death (KILL runs no trap): the hold is left up, but its heartbeat sees the batch gone and stops beating, so the box ages
  # the hold out instead of it staying fresh for ever.
  rm -rf "$t/fake-state" "$t/target/c-hold-kill"
  BLOOMERY_GATE_TIMES=$t/times-hold-stop.tsv BLOOMERY_GATE_LEDGER=$t/lead-hold-kill.tsv BLOOMERY_GATE_ROUND_LEDGER=$t/rounds-hold-kill.tsv \
    BLOOMERY_GPU_HOLD_BEAT=1 "${gb[@]}" --out "$t/target/c-hold-kill" --ledger "$slow_s" > "$t/out-hold-kill.log" 2>&1 &
  sp=$!
  k=0
  while ! grep -q 'gen_s' "$t/fake-state/seen.log" 2> /dev/null && [ "$k" -lt 20 ]; do
    sleep 1
    k=$((k + 1))
  done
  hp=$(cat "$t/target/c-hold-kill/gpuhold.pid" 2> /dev/null || echo 0)
  kill -KILL "$sp"
  rc=0
  wait "$sp" 2> /dev/null || rc=$?
  k=0
  while [ "$hp" -gt 1 ] && kill -0 "$hp" 2> /dev/null && [ "$k" -lt 10 ]; do
    sleep 1
    k=$((k + 1))
  done
  if [ "$hp" -gt 1 ] && ! kill -0 "$hp" 2> /dev/null && [ -e "$t/fake-state/batch.gpuhold" ]; then
    pass 'stop: a batch killed outright leaves its hold up, and its heartbeat ends within a beat'
  else fail 'stop: a batch killed outright leaves its hold up, and its heartbeat ends within a beat' "heartbeat pid $hp after ${k} s, hold file: $(ls "$t/fake-state/batch.gpuhold" 2>&1)"; fi
  want '  … and logs why' "$t/target/c-hold-kill/gpuhold.log" 'is gone: the heartbeat stops'
  for f in "$t"/target/c-hold-kill/lane-*.pid "$t"/target/c-hold-kill/lane-*.child; do # the killed batch's lanes: the pids it wrote
    if [ -f "$f" ]; then kill -TERM "$(cat "$f")" 2> /dev/null || true; fi
  done
  # A hold that vanishes (taken down by hand): the next beat finds it gone (3), logs it and the heartbeat stops — and no beat
  # creates the file again.
  rm -rf "$t/fake-state" "$t/target/c-hold-gone"
  BLOOMERY_GATE_TIMES=$t/times-hold-stop.tsv BLOOMERY_GATE_LEDGER=$t/lead-hold-gone.tsv BLOOMERY_GATE_ROUND_LEDGER=$t/rounds-hold-gone.tsv \
    BLOOMERY_GPU_HOLD_BEAT=1 "${gb[@]}" --out "$t/target/c-hold-gone" --ledger "$slow_s" > "$t/out-hold-gone.log" 2>&1 &
  sp=$!
  k=0
  while ! grep -q 'gen_s' "$t/fake-state/seen.log" 2> /dev/null && [ "$k" -lt 20 ]; do
    sleep 1
    k=$((k + 1))
  done
  rm -f "$t/fake-state/batch.gpuhold"
  # The beat that finds it gone logs the stop; the count of beat calls is read after that line and again 2 s later (beats 1 s apart).
  k=0
  while ! grep -q 'the heartbeat stops' "$t/target/c-hold-gone/gpuhold.log" 2> /dev/null && [ "$k" -lt 15 ]; do
    sleep 1
    k=$((k + 1))
  done
  b1=$(grep -c -- '--gpuhold beat' "$t/fake-state/hold.log" || true)
  sleep 2
  b2=$(grep -c -- '--gpuhold beat' "$t/fake-state/hold.log" || true)
  if [ "$b1" -ge 1 ] && [ "$b1" = "$b2" ] && [ ! -e "$t/fake-state/batch.gpuhold" ]; then
    pass 'hold: a hold taken down by hand: the heartbeat stops after the beat that finds it gone, and creates nothing'
  else fail 'hold: a hold taken down by hand: the heartbeat stops after the beat that finds it gone, and creates nothing' "beats $b1 then $b2, hold file: $(ls "$t/fake-state/batch.gpuhold" 2>&1)"; fi
  want '  … and logs why' "$t/target/c-hold-gone/gpuhold.log" "the hold is gone or another batch's: the heartbeat stops"
  kill -TERM "$sp"
  wait "$sp" 2> /dev/null || true
  # Another batch's fresh hold: not starting (75, named), nothing ran, its hold untouched, no run.log; a stale one is taken over.
  rc=$(PRE_HOLD=otherbatch LFLAG=--ledger steal_case hold-foreign "plain-any	B	a6000	10	$d" -- plain-any)
  printf '%s' "$rc" > "$t/rc-hold-foreign"
  if [ "$(cat "$t/rc-hold-foreign")" = 75 ] && grep -q "is up for owner otherbatch" "$t/out-hold-foreign.log" \
    && grep -q "^gate-batch: another batch's GPU hold is up" "$t/out-hold-foreign.log"; then
    pass 'hold: another batch'"'"'s fresh hold: not starting, 75, named'
  else fail 'hold: another batch'"'"'s fresh hold: not starting, 75, named' "rc=$(cat "$t/rc-hold-foreign") $(tail -2 "$t/out-hold-foreign.log")"; fi
  if head -1 "$t/fake-state/batch.gpuhold" | grep -q '^owner=otherbatch ' && [ ! -e "$t/target/c-hold-foreign/run.log" ]; then
    pass '  … its hold stays, and the refused start left no run.log'
  else fail '  … its hold stays, and the refused start left no run.log' "$(head -1 "$t/fake-state/batch.gpuhold" 2>&1)"; fi
  want_no_row '  … and no item ran' "$t/fake-state/seen.log" "$items_only"
  rc=$(PRE_HOLD=otherbatch PRE_AGE=400 LFLAG=--ledger steal_case hold-stale "plain-any	B	a6000	10	$d" -- plain-any)
  printf '%s' "$rc" > "$t/rc-hold-stale"
  rc_ok 'hold: a stale hold of another batch is taken over, and the batch ends green' hold-stale 1
  want '  … the takeover is named' "$t/target/c-hold-stale/gpuhold.log" 'owner otherbatch was not refreshed for 4[0-9][0-9] s: taking it over'
  if [ ! -e "$t/fake-state/batch.gpuhold" ]; then pass '  … and the hold is down at the end'
  else fail '  … and the hold is down at the end' "$(cat "$t/fake-state/batch.gpuhold")"; fi
  # Every item skips: nothing runs on the box, so no hold goes up.
  rc=$(FAKE_SKIP=1 LFLAG=--ledger steal_case hold-skip "plain-any	B	a6000	10	$d" -- plain-any)
  printf '%s' "$rc" > "$t/rc-hold-skip"
  rc_ok 'hold: a batch whose every item skips ends green' hold-skip 1
  if [ ! -e "$t/fake-state/hold.log" ]; then pass '  … and puts no hold up'
  else fail '  … and puts no hold up' "$(cat "$t/fake-state/hold.log")"; fi
  # The dry run names the hold it would take and touches nothing; the options' refusals.
  out=$(BLOOMERY_GATE_TIMES=$t/times-hold-up.tsv BLOOMERY_GATE_LEDGER=$t/lead-dry.tsv BLOOMERY_GATE_ROUND_LEDGER=$t/rounds-dry.tsv "${gb[@]}" --dry-run --ledger plain-any 2>&1) \
    || fail 'hold: the dry run of a --ledger batch failed' "$out"
  if grep -Eq '^gate-batch: gpu hold: would put .*batch\.gpuhold up on the box as gb_[A-Za-z0-9_]+, refreshed every 60 s; each item gets BLOOMERY_BATCH_OWNER=gb_' <<< "$out"; then
    pass 'dry run: a --ledger batch prints the hold it would take'
  else fail 'dry run: a --ledger batch prints the hold it would take' "$out"; fi
  out=$(BLOOMERY_GATE_TIMES=$t/times-hold-up.tsv BLOOMERY_GATE_LEDGER=$t/lead-dry.tsv BLOOMERY_GATE_ROUND_LEDGER=$t/rounds-dry.tsv "${gb[@]}" --dry-run --ledger --no-gpu-hold plain-any 2>&1) || true
  grep -q '^gate-batch: gpu hold: off (--no-gpu-hold)$' <<< "$out" && pass '  … --no-gpu-hold says it is off' || fail '  … --no-gpu-hold says it is off' "$out"
  out=$(BLOOMERY_GATE_TIMES=$t/times-hold-up.tsv BLOOMERY_GATE_LEDGER=$t/lead-dry.tsv BLOOMERY_GATE_ROUND_LEDGER=$t/rounds-dry.tsv "${gb[@]}" --dry-run --round-ledger plain-any 2>&1) || true
  grep -q '^gate-batch: gpu hold: none (only a --ledger batch takes it)$' <<< "$out" && pass '  … a --round-ledger batch says none' || fail '  … a --round-ledger batch says none' "$out"
  check 'hold: --no-gpu-hold without --ledger: 64, named' 64 "^gate-batch: --no-gpu-hold is the lead's: only a --ledger batch takes the GPU hold" "${gb[@]}" --dry-run --no-gpu-hold host
  check 'hold: --no-gpu-hold with --round-ledger: 64, named' 64 "^gate-batch: --no-gpu-hold is the lead's" "${gb[@]}" --dry-run --round-ledger --no-gpu-hold host
  check 'hold: a heartbeat period that is not whole seconds: 64, named' 64 "^gate-batch: BLOOMERY_GPU_HOLD_BEAT is whole seconds from 1 .*got 'abc'" \
    env BLOOMERY_GPU_HOLD_BEAT=abc "${gb[@]}" --dry-run host
  # Cold builds: the final try's section only; a local crate is the tree's, a registry or git one is not.
  printf '%s\n' '=== try 1 x' '   Compiling libc v0.2.155' '=== try 2 x' '   Compiling bloomery-gpu v0.1.0 (/root/repo/bloomery/crates/gpu)' > "$t/warm.log"
  printf '%s\n' '=== try 1 x' '   Compiling cuda-core v0.1.0 (https://github.com/x/cuda-oxide?branch=b#abc)' '    Finished `release`' > "$t/cold.log"
  out=$(cold_of "$t/warm.log" 2)/$(cold_of "$t/warm.log" 1)/$(cold_of "$t/cold.log" 1)
  if [ "$out" = 0/1/1 ]; then pass 'cold: a registry or git crate compiled in the final try, and only there'
  else fail 'cold: a registry or git crate compiled in the final try, and only there' "got $out, want 0/1/1"; fi
  local TIMES_FILE=$t/rows.tsv COLD_FILE=$t/rows-cold.tsv P_TKEY=(k-warm k-cold) P_TCARD=(3090 3090)
  times_record 0 A 12 0 && times_record 1 A 900 1
  out="$(cut -f1,4 "$TIMES_FILE" 2>&1 || true) | $(cut -f1,4 "$COLD_FILE" 2>&1 || true)"
  if [ "$out" = "k-warm	12 | k-cold	900" ]; then pass 'cold: its row goes to the cold file, never into the median'
  else fail 'cold: its row goes to the cold file, never into the median' "got '$out'"; fi
  # The waits a times row leaves out: box.sh's guard, the card lock and the V4.1 load lock, final try only.
  printf '%s\n' '=== try 1 x' 'gpu-gate.sh: waited 99 s for the gate lock (3090)' '=== try 2 x' \
    '[guard] 2026-09-27T00:00:00Z quiet on two polls in a row after 60 s: the command starts' \
    'gpu-gate.sh: waited 7 s for the batch hold' \
    'gpu-gate.sh: waited 15 s for the gate lock (3090)' 'gpu-gate.sh: waited 40 s for the V4.1 load lock' > "$t/waits.log"
  out=$(try_waits "$t/waits.log" 2)
  if [ "$out" = 122 ]; then pass 'waits: the guard, the batch hold, the card lock and the V4.1 load lock of the final try'
  else fail 'waits: the guard, the batch hold, the card lock and the V4.1 load lock of the final try' "got $out, want 122"; fi
  # The group and gpu-gate.sh's lock request must name the same recipes.
  cp "$t/justfile" "$t/justfile.good"
  sed 's/export BLOOMERY_GATE_V41_LOAD=1 \&\& BLOOMERY_GATE_CARD/BLOOMERY_GATE_CARD/' "$t/justfile.good" > "$t/justfile"
  check 'group without the export: a named error' 65 "recipe v41-any: \[group\('v41-load'\)\] without .export BLOOMERY_GATE_V41_LOAD=1." "${gb[@]}" --classes
  cp "$t/justfile.good" "$t/justfile"
  printf '%s\n' '' 'plain-v41:' "    ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && bash tools/gpu-gate.sh other'" >> "$t/justfile"
  check 'export without the group: a named error' 65 'recipe plain-v41: names BLOOMERY_GATE_V41_LOAD without' "${gb[@]}" --classes
  cp "$t/justfile.good" "$t/justfile"
  printf '%s\n' '' "[group('host')]" "[group('solo')]" 'host-solo:' "    ./tools/box.sh 'bash tools/gpu-gate.sh other'" >> "$t/justfile"
  check 'host with solo and a card gate: a named error' 65 "recipe host-solo: \[group\('host'\)\] with \[group\('solo'\)\] and a tools/gpu-gate.sh call" "${gb[@]}" --classes
  cp "$t/justfile.good" "$t/justfile"
  sed '/^weekly-a:$/,$d' "$t/justfile.good" > "$t/justfile" # the weekly recipes close the fixture
  check 'weekly: a justfile with no weekly-* recipe is a named error' 65 'no weekly-\* recipe' "${gb[@]}" --dry-run --weekly
  cp "$t/justfile.good" "$t/justfile"
  # The disk floor: above the floor a run proceeds on the verdict line; below it a real run stops at
  # 69 with the named line (nothing created — the fake HOME's tree would show it), a dry run shows
  # the same verdict and continues; df failing or unparsable is 69 in a dry run too; the override is
  # honored when positive and a named 64 when not; the default OUT lands outside every tree.
  check 'disk: above the floor, the verdict line and a dry run that proceeds' 0 \
    '^gate-batch: disk: [0-9.]+ GiB free on /fake/mount \(for .*\), above the [0-9]+ GiB floor$' "${gb[@]}" --dry-run host
  check 'disk: below the floor, a real run stops with 69 and the named line' 69 \
    '^gate-batch: disk: [0-9.]+ GiB free on /fake/mount .* below the [0-9]+ GiB floor' \
    env HOME=$t/home-low DISK_KB=1024 "${gb[@]}" host
  check "disk: below the floor, the real run's OUT is not created" 69 '^gate-batch: disk:' \
    env HOME=$t/home-low2 DISK_KB=1024 "${gb[@]}" host
  [ ! -e "$t/home-low2" ] || fail "a refused run created its OUT under $t/home-low2"
  check 'disk: a failing df is a named 69 in a dry run too' 69 \
    '^gate-batch: disk: df -Pk .* failed \(rc 1\): df: fake failure$' env DISK_RC=1 "${gb[@]}" --dry-run host
  check 'disk: df output that does not parse is a named 69' 69 'output does not parse' \
    env DISK_BAD=1 "${gb[@]}" --dry-run host
  check 'disk: BLOOMERY_MIN_FREE_GIB=abc is a named 64' 64 \
    "^gate-batch: BLOOMERY_MIN_FREE_GIB='abc' is not a positive integer of GiB$" \
    env BLOOMERY_MIN_FREE_GIB=abc "${gb[@]}" --dry-run host
  check 'disk: BLOOMERY_MIN_FREE_GIB=0 is a named 64 too' 64 'is not a positive integer of GiB' \
    env BLOOMERY_MIN_FREE_GIB=0 "${gb[@]}" --dry-run host
  check 'disk: a positive override raises the floor (the fake disk is below it)' 0 'below the 999999 GiB floor' \
    env BLOOMERY_MIN_FREE_GIB=999999 "${gb[@]}" --dry-run host
  check 'disk: below the floor, --dry-run prints the verdict and exits 0' 0 'below the [0-9]+ GiB floor' \
    env DISK_KB=1024 "${gb[@]}" --dry-run host
  out=$(env HOME=$t/home-dry "${gb[@]}" --dry-run host 2>&1) || fail "a dry run under a fake HOME failed: $out"
  case $out in
    *"logs would go to $t/home-dry/.cache/bloomery/batches/$(basename "$t")/"*) pass 'disk: the default OUT is $HOME/.cache/bloomery/batches/<tree>/<stamp>' ;;
    *) fail 'disk: the default OUT is $HOME/.cache/bloomery/batches/<tree>/<stamp>' "the dry run says: $out" ;;
  esac
  [ ! -e "$t/home-dry" ] || fail "the dry run created its OUT tree under $t/home-dry"
  echo "gate-batch self-test: $((n - bad)) of $n ok"
  [ "$bad" = 0 ]
}


if [ "${1:-}" = --self-test ]; then
  [ $# = 1 ] || die "--self-test takes nothing else; $USAGE"
  self_test
  exit $?
fi
OUT='' SRC='' LIST='' DRY=0 LANES=2 LEDGER=0 RERUN=0 LMODE='' TRUST=0 TIER_FLAG='' NOHOLD=0
ITEMS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --out) [ $# -ge 2 ] || die "--out needs a directory; $USAGE"; OUT=$2; shift 2 ;;
    --smoke) [ -z "$SRC" ] || die "--smoke, --weekly, --list and items are exclusive; $USAGE"; SRC=smoke; shift ;;
    --weekly) [ -z "$SRC" ] || die "--smoke, --weekly, --list and items are exclusive; $USAGE"; SRC=weekly; shift ;;
    --classes) [ -z "$SRC" ] || die "--classes takes no items; $USAGE"; SRC=classes; shift ;;
    --list) [ $# -ge 2 ] || die "--list needs a file; $USAGE"
      [ -z "$SRC" ] || die "--smoke, --weekly, --list and items are exclusive; $USAGE"; SRC=list; LIST=$2; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --ledger) [ "$LMODE" != round ] || die "--ledger and --round-ledger are exclusive; $USAGE"
      LEDGER=1 LMODE=lead; shift ;;
    --round-ledger) [ "$LMODE" != lead ] || die "--ledger and --round-ledger are exclusive; $USAGE"
      LEDGER=1 LMODE=round; shift ;;
    --trust-rounds) TRUST=1; shift ;;
    --no-gpu-hold) NOHOLD=1; shift ;;
    --rerun) RERUN=1; shift ;;
    --lanes) [ $# -ge 2 ] || die "--lanes needs 1 or 2; $USAGE"
      case "$2" in 1 | 2) LANES=$2 ;; *) die "--lanes is 1 or 2, got '$2'" ;; esac; shift 2 ;;
    --tier) [ $# -ge 2 ] || die "--tier needs real or fixture; $USAGE"
      TIER_FLAG=$2; shift 2 ;;
    -h | --help) sed -n '2,/^set -euo/p' "$0" | sed '$d'; exit 0 ;;
    --*) die "unknown option '$1'; $USAGE" ;;
    *) [ -z "$SRC" ] || [ "$SRC" = items ] || die "--smoke, --weekly, --list, --classes and items are exclusive; $USAGE"
      SRC=items; ITEMS+=("$1"); shift ;;
  esac
done
[ -n "$SRC" ] || die "no items; $USAGE"
if [ "$SRC" = list ]; then
  [ -f "$LIST" ] && [ -r "$LIST" ] || die "--list $LIST: not a readable file"
fi
[ "$SRC" = smoke ] && ITEMS=("${SMOKE[@]}")
[ "$RERUN" = 0 ] || [ "$LEDGER" = 1 ] || die "--rerun needs --ledger or --round-ledger; $USAGE"
[ "$TRUST" = 0 ] || [ "$LMODE" = lead ] || die "--trust-rounds is the lead's: it goes with --ledger (a round's --round-ledger reads the rounds' file already); $USAGE"
[ "$NOHOLD" = 0 ] || [ "$LMODE" = lead ] || die "--no-gpu-hold is the lead's: only a --ledger batch takes the GPU hold, so it goes with --ledger; $USAGE"
# The GPU hold (the header): the lead's batch takes it; its owner names this batch on the box and in each item's box env.
HOLD_ON=0 HOLD_UP=0 HOLD_OWNER='' HOLD_BEAT=${BLOOMERY_GPU_HOLD_BEAT:-60}
case $HOLD_BEAT in
  '' | 0 | *[!0-9]*) die "BLOOMERY_GPU_HOLD_BEAT is whole seconds from 1 (the heartbeat's period, default 60; the box calls a hold stale after 300), got '$HOLD_BEAT'" ;;
esac
if [ "$LMODE" = lead ] && [ "$NOHOLD" = 0 ]; then
  HOLD_ON=1
  HOLD_OWNER=gb_$(printf '%s' "$(basename "$ROOT")" | tr -c 'A-Za-z0-9' _)_$$_$(date +%s)
fi
command -v just > /dev/null || die "just is not on PATH"
command -v python3 > /dev/null || die "python3 is not on PATH"

# The tier (the header's --tier paragraph). Three places may name it — the flag, a BLOOMERY_TIER entry of BLOOMERY_BOX_ENV
# (where tools/box.sh reads it for the item's model file) and BLOOMERY_TIER in the environment — and they must agree: two
# tiers named at once would plan one and run the other. The fixture tier reaches every item through BLOOMERY_BOX_ENV, so the
# plan, the keys, the key rechecks and the runs all see the same string; the real tier adds nothing.
TIER_BOX=''
for kv in ${BLOOMERY_BOX_ENV:-}; do
  [ "${kv%%=*}" != BLOOMERY_TIER ] || TIER_BOX=${kv#*=}
done
TIER='' TIER_FROM=''
for src in "--tier:$TIER_FLAG" "BLOOMERY_BOX_ENV's BLOOMERY_TIER:$TIER_BOX" "BLOOMERY_TIER in the environment:${BLOOMERY_TIER:-}"; do
  [ -n "${src#*:}" ] || continue
  case "${src#*:}" in
    real | fixture) ;;
    *) die "${src%%:*} is real or fixture, got '${src#*:}'" ;;
  esac
  if [ -z "$TIER" ]; then
    TIER=${src#*:} TIER_FROM=${src%%:*}
  elif [ "$TIER" != "${src#*:}" ]; then
    die "$TIER_FROM names the tier '$TIER' and ${src%%:*} '${src#*:}': name one"
  fi
done
[ -n "$TIER" ] || TIER=real
if [ "$TIER" = fixture ] && [ -z "$TIER_BOX" ]; then
  BLOOMERY_BOX_ENV="${BLOOMERY_BOX_ENV:+$BLOOMERY_BOX_ENV }BLOOMERY_TIER=fixture"
fi

# The plan, in two passes. The first (PYPLAN) validates and classifies every item: one record per
# item, fields split by \x1f — class (A fixed, F balanced, X alone), log stem, recipe, the item's env
# (space separated, no card), ARGS (shell-quoted for eval), the item as written, the reason, the kind
# (fixed, balanced, solo, both-cards, one-lane), the expected seconds and their source, the times key,
# the card candidates (`-` for no card forced; a balanced gpu-gate item has two, 3090 and a6000) and
# the times file's card label of each. The second (PYBAL, once the ledger has keyed every candidate)
# picks each item's lane and card.
# (The sources sit in variables: bash 3.2 misparses a heredoc inside $(…) that holds quotes.)
IFS= read -r -d '' PYPLAN << 'PY' || true
import fcntl, json, os, re, shlex, statistics, subprocess, sys
from datetime import datetime

root, lanes, src, listfile, caller_env, times_path, default_s, tier = sys.argv[1:9]
raw_items = sys.argv[9:]
just = ["just", "--justfile", root + "/justfile", "--working-directory", root]


def fail(msg):
    print("gate-batch: " + msg, file=sys.stderr)
    sys.exit(65)


if src == "list":
    with open(listfile, encoding="utf-8") as fh:
        lines = fh.read().split("\n")
    head = [ln for ln in lines if ln.strip()][:2]
    # `just affected … 2>&1` puts just's echo of the recipe line above the output's `affected:` line.
    if head and head[0].startswith("./tools/affected-gates.sh ") and len(head) > 1:
        head = head[1:]
    if head and head[0].startswith("affected:"):
        raw_items = [ln.split()[0] for ln in lines if ln.startswith(("  gate-", "  weekly-"))]
        where = [f"{listfile} (just affected output)"] * len(raw_items)
    else:
        raw_items, where = [], []
        for i, ln in enumerate(lines, 1):
            if ln.strip() and not ln.lstrip().startswith("#"):
                raw_items.append(ln.strip())
                where.append(f"{listfile}:{i}")
else:
    where = [src] * len(raw_items)
if not raw_items and src not in ("classes", "weekly"):
    fail(f"the {src} input names no item — nothing to run is not a green batch")

proc = subprocess.run(just + ["--dump", "--dump-format", "json"], capture_output=True, text=True)
if proc.returncode != 0:
    fail(f"`just --dump` failed (exit {proc.returncode}): {proc.stderr.strip()}")
recipes = json.loads(proc.stdout)["recipes"]
if src == "weekly":
    raw_items = [n for n in recipes if n.startswith("weekly-")]
    where = ["--weekly"] * len(raw_items)
    if not raw_items:
        fail("--weekly: the justfile has no weekly-* recipe — nothing to run is not a green batch")


def body(name):
    out = []
    for frags in recipes[name]["body"]:
        out.append("".join(f if isinstance(f, str) else "{{}}" for f in frags))
    return "\n".join(out)


def closure(name, seen):
    if name in seen:
        return []
    seen.add(name)
    names = [name]
    for dep in recipes[name]["dependencies"]:
        names += closure(dep["recipe"], seen)
    return names


GPU_GATE = "tools/gpu-gate.sh"
ANY = "BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any}"
# The names the structural rule (walk() and takes_lease below) cannot see: timing-card.sh, a library a
# timing runner sources, which takes no lease itself; gpu-ab.py, a Python runner walk() does not read;
# the lease lock named in a recipe's own text, which walk() does not read either. Every runner that calls
# lease_take is found by the walk. Each counts where it runs, as TAKE's name does: the two scripts in
# command position (after env words and an interpreter, `.`, `source`, `exec` or `timeout N`; the start
# of box.sh's quoted command and of a `bash -c` string is a command position too), the lock as a word of
# a command; never in a comment, and never in a message (echo, printf, die, fail, say).
TIMED_RUN = re.compile(r"^[^#\n]*?(?:^|[;&|({!]|\b(?:then|do|else|if|while|until)(?=\s)|(?:tools/box\.sh|\b(?:ba)?sh\s+-c)\s+['\"])\s*"
                       r"(?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+)*(?:(?:bash|sh|source|\.|exec|python3?|timeout(?:\s+-\S+)*\s+\S+)\s+)*"
                       r"[^\s;&|#'\"]*(tools/ref/timing-card\.sh|gpu-ab\.py)\b", re.M)
TIMED_LOCK = re.compile(r"(?<![A-Za-z0-9_./-])['\"]?(/root/bloomery-(?:cpu|lease)\.lock)\b")


def timed(text):
    """The timed name a recipe's text runs, or None: TIMED_RUN, or the lease lock in a command that is
    not a message, outside a comment."""
    m = TIMED_RUN.search(text)
    if m:
        return m.group(1)
    for line in text.split("\n"):
        for seg in SEGMENT.split(line.split("#", 1)[0]):
            if not MESSAGE.match(seg):
                m = TIMED_LOCK.search(seg)
                if m:
                    return m.group(1)
    return None
BOX_CARD = re.compile(r"\bBLOOMERY_CARD=(both|a6000)\b")
UNLOCKED = re.compile(r"tools/gate\.sh --oxide|cargo oxide (test|run)\b")
SEGMENT = re.compile(r"&&|\|\||;|\||\n")
# A script a recipe names takes the timing lease when it calls lease_take (tools/ref/lease.sh), or
# opens the lease lock for a descriptor and waits on it with flock -w (the runners that take it by
# hand). A probe (lease_free, `flock -s -n <lock> true`: tools/gpu-gate.sh, box.sh's guard) only tests
# the lock and is no take.
SCRIPT = re.compile(r"(?:crates/[A-Za-z0-9_-]+/)?tools/[A-Za-z0-9_./-]+\.sh\b")
# A word in command position: at a line's start or after ; & | ( { ! or a keyword, before any `#`.
CMD = r"^[^#\n]*?(?:^|[;&|({!]|\b(?:then|do|else|if|while|until)(?=\s))\s*"
TAKE = re.compile(CMD + r"lease_take\s*(?=$|[;&|)#])", re.M)
WAIT = re.compile(CMD + r"flock\s+-w\b", re.M)
LEASE_EXEC = re.compile(r"^[^#\n]*\bexec\s+[0-9]+>\s*['\"]?/root/bloomery-(?:cpu|lease)\.lock", re.M)
LEASE_VAR = re.compile(r"^\s*(?:local\s+)?[A-Za-z_][A-Za-z0-9_]*=['\"]?/root/bloomery-(?:cpu|lease)\.lock", re.M)
TAKES = {}


def takes_lease(path):
    """`path:line what` when the tree script at path takes the timing lease, else None."""
    if path not in TAKES:
        hit, full = None, os.path.join(root, path)
        if os.path.isfile(full):
            with open(full, encoding="utf-8", errors="replace") as fh:
                text = fh.read()

            def line_of(m):
                return text.count("\n", 0, m.start()) + 1

            take = TAKE.search(text)
            opened = LEASE_EXEC.search(text) or LEASE_VAR.search(text)
            wait = WAIT.search(text)
            if take:
                hit = f"{path}:{line_of(take)} calls lease_take"
            elif opened and wait:
                hit = f"{path}:{line_of(opened)} opens the lease lock, :{line_of(wait)} waits on it"
        TAKES[path] = hit
    return TAKES[path]


# The scripts a recipe runs, followed transitively: a tools/…sh path on a script's non-comment line,
# or a relative `[dir/]name.sh` there resolved against that script's directory (ptx-spill-check.sh
# names "${BASH_SOURCE[0]%/*}/ptx-scan.sh"). Each file is read once (a cycle ends there), at most
# WALK_DEPTH hops from the recipe's text; a chain that would go deeper is a named refusal, never a
# walk cut short. Two questions are asked of every file it reaches, by this one walker: does it take
# the timing lease (takes_lease), and does it call tools/gpu-gate.sh — in command position, with the
# call's card form: `BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any}` (balanced), none (the 3090), or
# any other BLOOMERY_GATE_CARD on the call (a form this runner does not read: refused). A mention in
# a message (echo, printf, die, fail, say) is not a script the file runs.
# tools/gpu-gate.sh itself is the runner a call reaches, and its text is not followed; nor is this file's
# (check-recipes runs its --classes): its patterns and messages name the runner in strings the call pattern
# would read as calls. A stub-test
# harness (a directory named *-tests, tools/ref/card-tests) is named in the reason and not read: it
# runs the lease and runner code against a lock of its own, and its fixtures hold both on purpose.
WALK_DEPTH = 6
BARE = re.compile(r"([A-Za-z0-9_.-][A-Za-z0-9_./-]*\.(?:sh|bash))\b")
GATE_CALL = re.compile(CMD + r"(?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+)*(?:bash\s+|exec\s+)?[^\s;&|#'\"]*gpu-gate\.sh\b", re.M)
BATCH = "tools/gate-batch.sh"
RUNNER_SELF = {GPU_GATE, BATCH}
MESSAGE = re.compile(r"^[\s{(!]*(?:echo|printf|die|fail|say)\b")


def is_harness(path):
    return any(part.endswith("-tests") for part in path.split("/")[:-1])


def script_refs(path, text):
    """The tree scripts path's non-comment lines name, in order — outside a message: a segment whose
    command is echo, printf or a message function (die, fail, say) names a script it does not run."""
    out, sdir = [], os.path.dirname(path)
    for line in text.split("\n"):
        st = line.strip()
        if not st or st.startswith("#"):
            continue
        for seg in SEGMENT.split(line):
            if MESSAGE.match(seg):
                continue
            for q in SCRIPT.findall(seg) + [os.path.join(sdir, b) for b in BARE.findall(seg)]:
                q = os.path.normpath(q)
                if q.endswith((".sh", ".bash")) and os.path.isfile(os.path.join(root, q)):
                    out.append(q)
    return list(dict.fromkeys(out))


def walk(start):
    """{calls: [(form, where)], take: (chain, hit) or None, harness: [paths], error: str or None} for the
    scripts named in start, breadth first."""
    res = {"calls": [], "take": None, "harness": [], "error": None}
    parent, depth, queue = {}, {}, []
    for s in dict.fromkeys(os.path.normpath(q) for q in start):
        if os.path.isfile(os.path.join(root, s)) and s not in depth:
            depth[s] = 1
            queue.append(s)

    def chain(s):
        out = [s]
        while out[-1] in parent:
            out.append(parent[out[-1]])
        return " <- ".join(out)

    while queue:
        s = queue.pop(0)
        if s in RUNNER_SELF or not s.endswith((".sh", ".bash")):
            continue
        if is_harness(s):
            res["harness"].append(s)
            continue
        with open(os.path.join(root, s), encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        hit = takes_lease(s)
        if hit and res["take"] is None:
            res["take"] = (chain(s), hit)
        for m in GATE_CALL.finditer(text):
            where = f"{s}:{text.count(chr(10), 0, m.start()) + 1}"
            call = m.group(0)
            if ANY in call:
                res["calls"].append(("any", where))
            elif "BLOOMERY_GATE_CARD" in call:
                res["calls"].append(("unread", f"{where}: {call.strip()}"))
            else:
                res["calls"].append(("3090", where))
        for q in script_refs(s, text):
            if q in depth:
                continue
            if depth[s] >= WALK_DEPTH:
                res["error"] = (f"the scripts it runs go deeper than WALK_DEPTH = {WALK_DEPTH} hops "
                                f"({q} <- {chain(s)}): raise WALK_DEPTH in tools/gate-batch.sh")
                return res
            depth[q], parent[q] = depth[s] + 1, s
            queue.append(q)
    return res


GATE_FORMS = {}


def classify(name):
    """(lane, reason) for a recipe and its dependencies: lane A, B or X, or T (a timed recipe) or R
    (refused for another reason), which do not run in a batch."""
    lanes_seen, reasons, forms = set(), [], set()
    for n in closure(name, set()):
        text = body(n)
        if BATCH in SCRIPT.findall(text):
            return "R", (f"{n} runs {BATCH}: a batch inside a batch runs lanes of its own beside this batch's, "
                         "on the same cards; list its items instead")
        hit = timed(text)
        if hit:
            return "T", f"{n} runs {hit}: a timed recipe does not run in a gate batch"
        w = walk(SCRIPT.findall(text))
        if w["error"]:
            return "R", f"{n}: {w['error']}"
        if w["take"]:
            path, hit = w["take"]
            return "T", f"{n} runs {path}, which takes the timing lease ({hit}): a timed recipe does not run in a gate batch"
        for form, where in w["calls"]:
            if form == "unread":
                return "R", f"{n}: a script sets BLOOMERY_GATE_CARD in a form this runner does not read: {where}"
            forms.add(form)
            if form == "any":
                lanes_seen.add("B")
                reasons.append(f"{n}: gpu-gate.sh any ({where})")
            else:
                lanes_seen.add("A")
                reasons.append(f"{n}: gpu-gate.sh 3090 ({where})")
        reasons += [f"{n}: {h} not read (a stub-test harness)" for h in w["harness"]]
        m = BOX_CARD.search(text)
        if m:
            lanes_seen.add("X")
            reasons.append(f"{n}: {m.group(0)} (box.sh picks the cards, and the gate runner takes their locks)")
            continue
        oxide = "cargo oxide" in text
        for seg in SEGMENT.split(text):
            if GPU_GATE in seg:
                if ANY in seg:
                    lanes_seen.add("B")
                    forms.add("any")
                    reasons.append(f"{n}: gpu-gate.sh any")
                elif "BLOOMERY_GATE_CARD" in seg:
                    return "R", f"{n}: a gpu-gate.sh call sets BLOOMERY_GATE_CARD in a form this runner does not read: {seg.strip()}"
                else:
                    lanes_seen.add("A")
                    forms.add("3090")
                    reasons.append(f"{n}: gpu-gate.sh 3090")
            elif UNLOCKED.search(seg):
                lanes_seen.add("A")
                reasons.append(f"{n}: {UNLOCKED.search(seg).group(0)} (device code, no gate lock)")
            elif oxide and "target/release/" in seg and "host-gate.sh" not in seg:
                lanes_seen.add("A")
                reasons.append(f"{n}: direct target/release run (box env 3090 pin, no gate lock)")
    GATE_FORMS[name] = forms
    why = "; ".join(dict.fromkeys(r for r in reasons))
    for lane in ("X", "A", "B"):
        if lane in lanes_seen:
            return lane, why
    return "B", "no device code" + (f" ({why})" if why else "")


# The groups this runner reads from the just attribute `[group('…')]`. A value it does not know is a
# named error wherever it sits in the justfile (check-recipes runs `--smoke --dry-run`, so it fails
# there too): a group is a scheduling class here, and a typo must not drop a gate out of its class.
GROUPS = {
    "solo": "alone in lane X, after both lanes",
    "solo-real": "alone in lane X in the real tier, balanced over lanes A and B in the fixture tier",
    "v41-load": "lane A, one after another (solo wins)",
    "host": "balanced over lanes A and B, no card forced: device-crate tests that open no card",
}
unknown = [
    f"recipe {n}: [group({a['group']!r})] is not a group this runner knows ({', '.join(GROUPS)}) — "
    "a group is a scheduling class here: add it to GROUPS in tools/gate-batch.sh with its lane rule, or remove it"
    for n in sorted(recipes)
    for a in recipes[n]["attributes"]
    if isinstance(a, dict) and "group" in a and a["group"] not in GROUPS
]
if unknown:
    for e in unknown:
        print("gate-batch: " + e, file=sys.stderr)
    fail(f"{len(unknown)} unknown group attribute(s) in the justfile; nothing ran")

# The v41-load group has two readers: this runner's lane, and tools/gpu-gate.sh's box-wide V4.1 load lock,
# which a recipe asks for by exporting BLOOMERY_GATE_V41_LOAD=1 in its box command. The two must name the
# same recipes: a member without the export loads V4.1 beside another tree's load; an export without the
# group is a load this runner would balance onto lane B.
V41_EXPORT = "export BLOOMERY_GATE_V41_LOAD=1 && "
mismatch = []
for n in sorted(recipes):
    member = {"group": "v41-load"} in recipes[n]["attributes"]
    text = body(n)
    if member and V41_EXPORT not in text:
        mismatch.append(f"recipe {n}: [group('v41-load')] without `{V41_EXPORT.strip(' &')}` in its box command "
                        "— tools/gpu-gate.sh would not take the V4.1 load lock")
    elif not member and "BLOOMERY_GATE_V41_LOAD" in text:
        mismatch.append(f"recipe {n}: names BLOOMERY_GATE_V41_LOAD without [group('v41-load')]")
if mismatch:
    for e in mismatch:
        print("gate-batch: " + e, file=sys.stderr)
    fail(f"{len(mismatch)} v41-load recipe(s) whose group and export disagree; nothing ran")

def group_of(name, group):
    """The recipes of name's closure that carry [group('<group>')]."""
    return [n for n in closure(name, set()) if {"group": group} in recipes[n]["attributes"]]


# The real-only class (the header's paragraph) is a call in the box command: `bash tools/ref/real-only.sh <recipe> && …`, the first
# command of every box.sh command of the recipe. The box side runs it (it stops the fixture tier by name); this runner reads it from
# the text. A call that names another recipe would make the refusal name the wrong gate; a box command that does not open with it,
# or a call outside a single-quoted box command, would leave a command the refusal does not guard.
REAL_ONLY = "tools/ref/real-only.sh"
REAL_ONLY_CALL = re.compile(r"\bbash " + re.escape(REAL_ONLY) + r"\s+(\S+)")
BOX_OPEN = "tools/box.sh '"


def defer_tags(name):
    """What defers a recipe in the fixture tier, for the plan's reason text."""
    tags = [f"{n}: real-only call" for n in real_only_of(name)]
    if tier == "fixture" and not tags and no_fixture(name):
        tags.append(f"family {no_fixture(name)}: no row in the fixture table of tools/ref/ref-paths.sh (box.sh exits 66)")
    return tags


def real_only_of(name):
    """The recipes of name's closure whose own text calls the real-only guard."""
    return [n for n in closure(name, set()) if REAL_ONLY_CALL.search(body(n))]


ro_bad = []
for n in sorted(recipes):
    text = body(n)
    calls = REAL_ONLY_CALL.findall(text)
    if not calls:
        continue
    first = f"bash {REAL_ONLY} {n} && "
    if not os.path.isfile(os.path.join(root, REAL_ONLY)):
        ro_bad.append(f"recipe {n}: calls {REAL_ONLY}, which is not in the tree")
    wrong = sorted({c for c in calls if c != n})
    if wrong:
        ro_bad.append(f"recipe {n}: its real-only call names {', '.join(wrong)}, not itself — write `{first.rstrip(' &')}`")
    opens = [m.end() for m in re.finditer(re.escape(BOX_OPEN), text)]
    loose = [e for e in opens if not text.startswith(first, e)]
    if not opens or loose or len(calls) != len(opens):
        ro_bad.append(f"recipe {n}: a real-only recipe opens every box.sh command with `{first.rstrip(' &')} && …` — "
                      f"{len(opens)} single-quoted box.sh command(s), {len(calls)} call(s), {len(loose)} not opening with it")
if ro_bad:
    for e in ro_bad:
        print("gate-batch: " + e, file=sys.stderr)
    fail(f"{len(ro_bad)} real-only recipe(s) the class could not hold; nothing ran")


host_bad = []
for n in sorted(recipes):
    if {"group": "host"} not in recipes[n]["attributes"]:
        continue
    text = "\n".join(body(m) for m in closure(n, set()))
    why = [f"[group('{g}')]" for g in ("solo", "solo-real", "v41-load") if group_of(n, g)]
    why += [w for w, hit in (("a tools/gpu-gate.sh call", GPU_GATE in text), ("a box.sh card pick", BOX_CARD.search(text))) if hit]
    if why:
        host_bad.append(f"recipe {n}: [group('host')] with {' and '.join(why)} — a host recipe opens no card and runs beside other items")
if host_bad:
    for e in host_bad:
        print("gate-batch: " + e, file=sys.stderr)
    fail(f"{len(host_bad)} host recipe(s) that open a card or run alone; nothing ran")


def solo_any(name):
    """Whether the recipe is solo or solo-real, whatever the tier: what its own arms are judged by."""
    return group_of(name, "solo") or group_of(name, "solo-real")


def solo_of(name):
    """The recipes that keep name alone in this tier: [group('solo')], and [group('solo-real')] in the real tier."""
    return group_of(name, "solo") + (group_of(name, "solo-real") if tier == "real" else [])


# A solo-real recipe is balanced in the fixture tier, so it must be one the balance can place: every gpu-gate.sh call of it takes
# the `any` card, or the recipe picks the A6000 alone through box.sh (one card: the fixture tier fixes it to lane B), and nothing
# keeps it alone in either tier — a `solo` beside it, a pick of both cards (it stays alone whatever the tier) or a 3090-only call
# would each make the second tier's placement the first's, and the group a name for nothing. A recipe with no fixture conversion
# (real-only) has no second tier's placement to choose.
sr_bad = []
for n in sorted(recipes):
    if {"group": "solo-real"} not in recipes[n]["attributes"]:
        continue
    why = []
    if group_of(n, "solo"):
        why.append("[group('solo')] — a recipe is alone in both tiers (solo) or in the real one (solo-real)")
    if real_only_of(n):
        why.append("a real-only call — a recipe with no fixture conversion has no fixture-tier placement (real-only, or solo-real)")
    text = "\n".join(body(m) for m in closure(n, set()))
    pick = BOX_CARD.search(text)
    if pick and pick.group(1) == "both":
        why.append(f"{pick.group(0)} — a box.sh pick of both cards keeps a recipe alone in both tiers (use solo)")
    if not why:
        lane, reason = classify(n)
        if lane in ("T", "R"):
            why.append(reason)
        elif pick:
            if GPU_GATE not in text or "3090" in GATE_FORMS[n]:
                seen = ", ".join(sorted(GATE_FORMS[n])) or "none"
                why.append(f"{pick.group(0)} with no tools/gpu-gate.sh call that takes the pick (script forms: {seen}) — "
                           "the fixture tier fixes the recipe to the A6000's lane")
        elif GATE_FORMS[n] != {"any"}:
            seen = ", ".join(sorted(GATE_FORMS[n])) or "none"
            why.append(f"its tools/gpu-gate.sh calls are not all `{ANY}` (forms: {seen}) — the fixture tier balances it on the any card")
    if why:
        sr_bad.append(f"recipe {n}: [group('solo-real')] with " + "; ".join(why))
if sr_bad:
    for e in sr_bad:
        print("gate-batch: " + e, file=sys.stderr)
    fail(f"{len(sr_bad)} solo-real recipe(s) the fixture tier could not place; nothing ran")


# A solo recipe's arm run through another recipe: a non-solo item whose ARGS hand a solo recipe's
# gpu-gate.sh binary one of the flags that solo recipe passes it would run that arm in a lane beside
# other loads, and without whatever else the solo recipe sets around it. Read from the justfile: per solo recipe, the binary of each
# gpu-gate.sh call and the literal --flags after it.
CALL_ARGS = re.compile(r"gpu-gate\.sh\s+([A-Za-z0-9_-]+)((?:\s+[^\s;&|]+)*)")


def gate_calls(name):
    """[(binary, {literal --flags})] of the gpu-gate.sh calls in name's own text."""
    return [(m.group(1), set(re.findall(r"(?<!\S)(--[A-Za-z0-9][A-Za-z0-9-]*)", m.group(2))))
            for m in CALL_ARGS.finditer(body(name))]


SOLO_ARMS = {}
for _n in recipes:
    if {"group": "solo"} in recipes[_n]["attributes"] or {"group": "solo-real"} in recipes[_n]["attributes"]:
        for _bin, _flags in gate_calls(_n):
            for _f in _flags:
                SOLO_ARMS.setdefault((_bin, _f), _n)


def solo_arm(name, argv):
    """The (flag, solo recipe) whose arm the item's ARGS select, or None."""
    if solo_any(name):
        return None
    for n in closure(name, set()):
        for b, _ in gate_calls(n):
            for a in argv:
                if (b, a) in SOLO_ARMS:
                    return a, b, SOLO_ARMS[(b, a)]
    return None


ITEM = re.compile(r"^([A-Za-z0-9_-]+)(?:@([^:]*))?(?::(.*))?$")
VAR = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
TIMES_LANES, TIMES_CARDS = ("A", "B", "X"), ("3090", "a6000", "both", "none", "any")
DEFAULT_S = int(default_s)


def load_times(path):
    """{item: [seconds, …]} in file order. A missing file is empty; a malformed row is a named error."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return {}
    except OSError as e:
        fail(f"the times file {path} cannot be read: {e.strerror}")
    chunks = []
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        while True:
            b = os.read(fd, 1 << 20)
            if not b:
                break
            chunks.append(b)
    except OSError as e:
        fail(f"the times file {path} cannot be read: {e.strerror}")
    finally:
        os.close(fd)
    fix = "this runner writes the file; fix or remove the row"
    try:
        text = b"".join(chunks).decode("utf-8")
    except UnicodeDecodeError as e:
        fail(f"the times file {path} is not UTF-8 ({e}) — {fix}")
    if text and not text.endswith("\n"):
        fail(f"{path}: the last row has no newline (a torn write) — {fix}")
    rows = {}
    for n, ln in enumerate(text.split("\n")[:-1], 1):
        f, bad = ln.split("\t"), None
        if len(f) != 5:
            bad = f"{len(f)} fields, not 5"
        elif not ITEM.match(f[0]):
            bad = f"item {f[0]!r} is not NAME[@K=V,…][:ARGS]"
        elif f[1] not in TIMES_LANES:
            bad = f"lane {f[1]!r} is not A, B or X"
        elif f[2] not in TIMES_CARDS:
            bad = f"card {f[2]!r} is not one of {', '.join(TIMES_CARDS)}"
        elif not re.fullmatch(r"[0-9]+", f[3]):
            bad = f"seconds {f[3]!r} is not a whole number"
        else:
            try:
                datetime.strptime(f[4], "%Y-%m-%dT%H:%M:%S%z")
            except ValueError:
                bad = f"date {f[4]!r} is not YYYY-MM-DDTHH:MM:SS+hhmm"
        if bad:
            fail(f"{path}:{n}: a malformed times row ({bad}): {ln!r} — {fix}")
        rows.setdefault(f[0], []).append(int(f[3]))
    return rows


TIMES = load_times(times_path)


def expected(key):
    """(seconds, source): the median of the item's last 5 rows, else the default."""
    got = TIMES.get(key, [])[-5:]
    if not got:
        return DEFAULT_S, "default"
    return int(statistics.median(got) + 0.5), f"median of {len(got)} row{'s' if len(got) > 1 else ''}"


for kv in caller_env.split():
    if lanes == "2" and kv.split("=", 1)[0] == "BLOOMERY_GATE_CARD":
        fail("BLOOMERY_BOX_ENV sets BLOOMERY_GATE_CARD; with two lanes the card is the runner's (use --lanes 1)")

# An item's model family: the profile tools/box.sh loads for it — the BLOOMERY_MODEL a recipe of its
# closure sets on its box line, else tools/ref/ref-paths.sh's default. Lane X runs FIRST_FAMILY first:
# lane A's last loads are V4.1's (the v41-load group), so its host set is still in the page cache when lane
# X starts, and every family switch after that refetches a host set once.
FIRST_FAMILY = "deepseek41"
PROFILE_SET = re.compile(r"\bBLOOMERY_MODEL=([A-Za-z0-9_]+)\s+(?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+)*\./tools/box\.sh\b")
DEFAULT_PROFILE = []


def default_profile():
    """ref-paths.sh's default profile, read once, when a recipe sets none."""
    if not DEFAULT_PROFILE:
        path = os.path.join(root, "tools/ref/ref-paths.sh")
        try:
            with open(path, encoding="utf-8") as fh:
                got = re.findall(r'^: "\$\{BLOOMERY_MODEL:=\$\{BLOOMERY_REF_MODEL_PROFILE:-([A-Za-z0-9_]+)\}\}"$', fh.read(), re.M)
        except OSError as e:
            fail(f"{path} cannot be read ({e.strerror}): lane X's family order reads the default profile there")
        if len(got) != 1:
            fail(f"{path} holds {len(got)} default-profile lines, not one: lane X's family order reads the default there")
        DEFAULT_PROFILE.append(got[0])
    return DEFAULT_PROFILE[0]


def family(name):
    for n in closure(name, set()):
        m = PROFILE_SET.search(body(n))
        if m:
            return m.group(1)
    return default_profile()


FIXTURE_TABLE = []


def fixture_families():
    """The profiles with a row in the fixture tier's table of tools/ref/ref-paths.sh (`<profiles> ) __fixture_dir=<dir> ;;`): a
    fixture directory, or `self` (the family's real file is small and stands). Every other profile falls to the table's `*)` row,
    `none`, where tools/box.sh exits 66 under the fixture tier. The table is that file's, read here; a table this reader does not
    understand is a named error."""
    if not FIXTURE_TABLE:
        path = os.path.join(root, "tools/ref/ref-paths.sh")
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        except OSError as e:
            fail(f"{path} cannot be read ({e.strerror}): the fixture tier's family table is read there")
        m = re.search(r'^\s*case "\$BLOOMERY_MODEL" in\n((?:\s*[^\n]*\) __fixture_dir=[a-z0-9_-]+ ;;\n)+)\s*esac$', text, re.M)
        if not m:
            fail(f"{path}: no fixture table (`case \"$BLOOMERY_MODEL\" in … ) __fixture_dir=<dir> ;; … esac`) — tools/gate-batch.sh reads which "
                 "families have a fixture there")
        have, default = set(), None
        for row in m.group(1).strip().split("\n"):
            pats, dirname = re.fullmatch(r"\s*(.*?)\) __fixture_dir=([a-z0-9_-]+) ;;", row).groups()
            for pat in (x.strip() for x in pats.split("|")):
                if pat == "*":
                    default = dirname
                elif dirname != "none":
                    have.add(pat)
        if default != "none":
            fail(f"{path}: the fixture table's `*)` row is {default!r}, not none: a family without a row must be one with no fixture")
        FIXTURE_TABLE.append(have)
    return FIXTURE_TABLE[0]


def no_fixture(name):
    """The profile of name's family when the fixture tier has none for it (box.sh would exit 66), else None."""
    fam = family(name)
    return None if fam in fixture_families() else fam


def placement(name, lane, lanes):
    """(class, kind, card candidates, their times-file labels) of a recipe that runs in a batch."""
    names = closure(name, set())
    uses_gpu_gate = bool(GATE_FORMS[name])
    box = next((m.group(1) for m in (BOX_CARD.search(body(n)) for n in names) if m), None)
    if tier == "fixture" and real_only_of(name):
        # No fixture conversion: the item is deferred to the real tier, in no lane (class D; the shell side keeps it apart).
        return "D", "real-only", ["-"], ["none"]
    if tier == "fixture" and no_fixture(name):
        # The family has no fixture (box.sh would exit 66, a red): deferred in the same way, under its own kind.
        return "D", "no-fixture", ["-"], ["none"]
    if lanes == "1":
        if box:
            labels = [box]
        elif uses_gpu_gate:
            labels = ["any" if "any" in GATE_FORMS[name] else "3090"]
        else:
            labels = ["3090" if lane == "A" else "none"]
        return "A", "one-lane", ["-"], labels
    if tier == "fixture" and box == "a6000" and group_of(name, "solo-real"):
        # One card by box.sh's pick: the fixture tier fixes it to the A6000's lane (class B; no lane steals it).
        return "B", "fixed", ["-"], [box]
    if solo_of(name) or lane == "X":
        kind = "solo" if solo_of(name) else "both-cards"
        if box:
            return "X", kind, ["-"], [box]
        if uses_gpu_gate:
            return "X", kind, ["3090" if lane == "A" else "a6000"], ["3090" if lane == "A" else "a6000"]
        return "X", kind, ["-"], ["3090" if lane == "A" else "none"]
    if group_of(name, "host"):
        # Device code with no card: its lane is a scheduling choice, and the balance makes it.
        return "F", "balanced", ["-"], ["none"]
    if group_of(name, "v41-load") and tier == "real":
        # One lane: two V4.1 loads at once evict each other's host set from the page cache. Lane A,
        # where the 3090-only (`--place gate`) loads already are; an `any` member runs on the 3090.
        # Not in the fixture tier: a fixture load takes no V4.1 load lock and has no such host set.
        return "A", "fixed", (["3090"] if uses_gpu_gate else ["-"]), ["3090"]
    if lane == "A":
        return "A", "fixed", (["3090"] if uses_gpu_gate else ["-"]), ["3090"]
    if uses_gpu_gate:
        return "F", "balanced", ["3090", "a6000"], ["3090", "a6000"]
    return "F", "balanced", ["-"], ["none"]


# --classes: every recipe of the justfile, one line each, and nothing else runs:
#   <recipe>\t<class>\t<kind>\t<cards>\t<reason>
# class A (fixed, the 3090), F (balanced), X (alone), T (timed: refused in a batch) or R (refused for
# another reason); cards the candidates, `-` for none forced. tools/check-recipes.sh reads the T lines.
if src == "classes":
    for name in sorted(recipes):
        lane, reason = classify(name)
        if lane in ("T", "R"):
            print("\t".join([name, lane, "timed" if lane == "T" else "refused", "-", reason]))
        else:
            cls, kind, cards, _ = placement(name, lane, "2")
            tags = [f"{n}: [group('{g}')]" for g in ("solo", "solo-real", "v41-load", "host") for n in group_of(name, g)]
            tags += defer_tags(name)
            print("\t".join([name, cls, kind, ",".join(cards), "; ".join(tags + [reason])]))
    sys.exit(0)

counts, errors, out, fams = {}, [], [], []
for item, at in zip(raw_items, where):
    m = ITEM.match(item)
    if not m:
        errors.append(f"{at}: malformed item {item!r} (NAME[@K=V,…][:ARGS])")
        continue
    name, env, args = m.group(1), m.group(2), m.group(3)
    if name not in recipes:
        errors.append(f"{at}: {name!r} is not a recipe in the justfile")
        continue
    envs = []
    if env is not None:
        for kv in env.split(","):
            k, eq, v = kv.partition("=")
            if not eq or not VAR.match(k) or re.search(r"\s", v):
                errors.append(f"{at}: env entry {kv!r} of {item!r} is not K=V (no spaces)")
            elif lanes == "2" and k == "BLOOMERY_GATE_CARD":
                errors.append(f"{at}: {item!r} sets BLOOMERY_GATE_CARD; with two lanes the card is the runner's (use --lanes 1)")
            elif k == "BLOOMERY_TIER":
                errors.append(f"{at}: {item!r} sets BLOOMERY_TIER; the tier is the batch's (--tier), as it decides the item's lane")
            else:
                envs.append(f"{k}={v}")
    argv = []
    if args is not None:
        if re.search(r"@[A-Za-z_][A-Za-z0-9_]*=", args):
            errors.append(f"{at}: {item!r}: env goes before ARGS (NAME@K=V:ARGS)")
            continue
        try:
            argv = shlex.split(args)
        except ValueError as e:
            errors.append(f"{at}: ARGS of {item!r}: {e}")
            continue
        if argv and not recipes[name]["parameters"]:
            errors.append(f"{at}: {name} takes no arguments (just would read {argv[0]!r} as another recipe)")
            continue
    # just's own argument check (arity, options), printing the recipe instead of running it.
    chk = subprocess.run(just + ["--dry-run", name] + argv, capture_output=True, text=True)
    if chk.returncode != 0:
        msg = [ln for ln in chk.stderr.splitlines() if ln.startswith("error")] or chk.stderr.splitlines()[-1:]
        errors.append(f"{at}: just rejects {item!r}: {' '.join(msg)}")
        continue
    lane, reason = classify(name)
    if lane in ("T", "R"):
        errors.append(f"{at}: {reason}")
        continue
    arm = solo_arm(name, argv)
    if arm:
        flag, b, solo_n = arm
        errors.append(f"{at}: {item!r} hands {b} {flag}, the arm of the solo recipe {solo_n} — it runs alone in "
                      f"lane X with what that recipe sets around it; list {solo_n} instead")
        continue
    tags = [f"{n}: [group('{g}')]" for g in ("solo", "solo-real", "v41-load", "host") for n in group_of(name, g)]
    tags += defer_tags(name)
    if tags:
        reason = "; ".join(tags + [reason])
    qargs = " ".join(shlex.quote(a) for a in argv)
    # A fixture run is not the real one: its row is an entry of its own, so neither pollutes the other's median.
    tenvs = envs + (["BLOOMERY_TIER=fixture"] if tier == "fixture" else [])
    tkey = name + ("@" + ",".join(tenvs) if tenvs else "") + (":" + qargs if argv else "")
    if re.search(r"[\t\n]", tkey):
        errors.append(f"{at}: {item!r} holds a tab or a newline in its env or ARGS: the times file is one tab-separated row per item")
        continue
    exp, esrc = expected(tkey)
    cls, kind, cards, labels = placement(name, lane, lanes)
    counts[name] = counts.get(name, 0) + 1
    stem = name if counts[name] == 1 else f"{name}-{counts[name]}"
    shown = item if (env is not None or args is not None) else ""
    out.append("\x1f".join([cls, stem, name, " ".join(envs), qargs, shown, reason, kind, str(exp), esrc, tkey, " ".join(cards), " ".join(labels)]))
    fams.append(family(name) if cls == "X" else "")
if errors:
    for e in errors:
        print("gate-batch: " + e, file=sys.stderr)
    fail(f"{len(errors)} item error(s); nothing ran")
# Lane X by model family: FIRST_FAMILY's items first, then each other family's in the order its first item
# is listed, each family's items in list order. The other lanes keep list order (a record's lane is its
# first field, and lanes run their own records in order).
rank = {FIRST_FAMILY: 0}
for f in fams:
    if f:
        rank.setdefault(f, len(rank))
xs = sorted((k for k, rec in enumerate(out) if rec.startswith("X\x1f")), key=lambda k: (rank[fams[k]], k))
out = [rec for rec in out if not rec.startswith("X\x1f")] + [out[k] for k in xs]
print("\n".join(out))
PY
PLAN0=$(python3 -c "$PYPLAN" "$ROOT" "$LANES" "$SRC" "$LIST" "${BLOOMERY_BOX_ENV:-}" "$TIMES_FILE" "$DEFAULT_S" "$TIER" ${ITEMS[@]+"${ITEMS[@]}"}) \
  || RC=$? die "the list did not validate (above)"
if [ "$SRC" = classes ]; then
  printf '%s\n' "$PLAN0"
  exit 0
fi

item_str() { # NAME ENV ARGS: the item as it runs, for the key — NAME[@ENV with commas for spaces][:ARGS]
  local s=$1
  [ -z "$2" ] || s="$s@${2// /,}"
  [ -z "$3" ] || s="$s:$3"
  printf '%s' "$s"
}

# The candidates: one per card an item may run with. C_ITEM is the item as the ledger keys it (its
# env, then the card); R_C0 is the index of each record's first candidate. C_LST stays `-` without
# --ledger.
R_REC=() R_C0=() C_ITEM=() C_KEY=() C_LST=() C_LDET=()
# The fixture tier's deferred items (class D: real-only gates, and gates of a family with no fixture) are not part of the plan below:
# no lane, no key, no claim, no times row. They are kept apart, named in the dry run and in run.log, listed in DIR/deferred.list and
# counted in the DONE line (the header's deferred paragraph).
DP_STEM=() DP_NAME=() DP_ENV=() DP_ARGS=() DP_ITEM=() DP_WHY=() DP_KIND=()
while IFS= read -r rec; do
  case $rec in
    D$'\x1f'*)
      IFS=$'\x1f' read -r _ stem name env args item why kind _ <<< "$rec"
      DP_STEM+=("$stem"); DP_NAME+=("$name"); DP_ENV+=("$env"); DP_ARGS+=("$args"); DP_ITEM+=("$item"); DP_WHY+=("$why"); DP_KIND+=("$kind") ;;
    *) R_REC+=("$rec") ;;
  esac
done <<< "$PLAN0"
N=${#R_REC[@]}
ND=${#DP_STEM[@]}
NT=$((N + ND)) # every item of the batch: the ones that run and the ones deferred
ND_RO=0 ND_NF=0 # of those, the real-only gates and the gates of a family with no fixture
for ((i = 0; i < ND; i++)); do
  if [ "${DP_KIND[$i]}" = real-only ]; then ND_RO=$((ND_RO + 1)); else ND_NF=$((ND_NF + 1)); fi
done
DEFER_NOTE=''
[ "$ND" = 0 ] || DEFER_NOTE=" (and $ND deferred to the real tier: $ND_RO real-only, $ND_NF of a family with no fixture)"
for ((i = 0; i < N; i++)); do
  IFS=$'\x1f' read -r _ _ name env args _ _ _ _ _ _ cards _ <<< "${R_REC[$i]}"
  R_C0+=("${#C_ITEM[@]}")
  for c in $cards; do
    e=$env
    [ "$c" = - ] || e="${e:+$e }BLOOMERY_GATE_CARD=$c"
    C_ITEM+=("$(item_str "$name" "$e" "$args")"); C_KEY+=(-); C_LST+=(-); C_LDET+=("")
  done
done
NC=${#C_ITEM[@]}

# The second pass: each record, then the ledger state of each of its candidates. It prints per item
# the lane, stem, recipe, env with the card, ARGS, the item as written, the reason, the plan line
# (kind, expected seconds, placement), the seconds it counts in its lane, the times key and card
# label, and the candidate it took; then one `=` record: the lane sums, the wall, and the items
# whose expectation is the default.
IFS= read -r -d '' PYBAL << 'PY' || true
import sys

lanes = sys.argv[1]


def fail(msg):
    print("gate-batch: balance: " + msg, file=sys.stderr)
    sys.exit(70)


def lane_cand(cards):
    """lane -> candidate index of a balanced item: its two cards, or one no-card candidate for both."""
    if cards == ["3090", "a6000"]:
        return {"A": 0, "B": 1}
    if cards == ["-"]:
        return {"A": 0, "B": 0}
    return None


recs = [ln.split("\x1f") for ln in sys.stdin.read().split("\n") if ln]
sums, placed, flex = {"A": 0, "B": 0, "X": 0}, {}, []
for i, f in enumerate(recs):
    if len(f) != 14:
        fail(f"a plan record has {len(f)} fields, not 14: {f!r}")
    cls, stem, exp, cards, labels, states = f[0], f[1], int(f[8]), f[11].split(), f[12].split(), f[13].split()
    if not cards or not len(cards) == len(labels) == len(states):
        fail(f"{stem}: {len(cards)} card candidates, {len(labels)} labels, {len(states)} ledger states")
    if cls in ("A", "B", "X") and len(cards) == 1:
        placed[i] = (cls, 0, 0 if states[0] == "skip" else exp, "")
        sums[cls] += placed[i][2]
    elif cls == "F" and lanes == "2" and lane_cand(cards):
        flex.append(i)
    else:
        fail(f"{stem} fits no lane (class {cls!r}, cards {' '.join(cards)}, --lanes {lanes})")
# The ledger first: an item green on one lane's card only skips there, so it goes there at 0 s.
rest, seq = [], []
for i in flex:
    lc, states = lane_cand(recs[i][11].split()), recs[i][13].split()
    green = [ln for ln in ("A", "B") if states[lc[ln]] == "skip"]
    if len(green) == 1:
        placed[i] = (green[0], lc[green[0]], 0, f"to {green[0]}, the one lane whose card has it green")
        seq.append(i)
    elif green:
        ln = "A" if sums["A"] < sums["B"] else "B"
        placed[i] = (ln, lc[ln], 0, f"to {ln}, green in either lane")
        seq.append(i)
    else:
        rest.append(i)
# Then longest first, each to the lane with the smaller running sum (B on a tie).
for i in sorted(rest, key=lambda i: (-int(recs[i][8]), i)):
    ln, lc, exp = "A" if sums["A"] < sums["B"] else "B", lane_cand(recs[i][11].split()), int(recs[i][8])
    placed[i] = (ln, lc[ln], exp, f"to {ln} (lane sums before it: A {sums['A']} s, B {sums['B']} s)")
    sums[ln] += exp
    seq.append(i)
defaults = []
for i, f in enumerate(recs):
    cls, stem, name, env, args, shown, why, kind, exp, esrc, tkey, cards, labels, states = f
    ln, c, cost, how = placed[i]
    card, label, state = cards.split()[c], labels.split()[c], states.split()[c]
    if card != "-":
        env = (env + " " if env else "") + "BLOOMERY_GATE_CARD=" + card
    if state == "skip":
        told = "skips (the ledger has it green), counted 0 s"
    else:
        told = f"expected {exp} s ({esrc})"
        if esrc == "default":
            defaults.append(stem)
    plan = f"{kind}, {told}" + (f", {how}" if how else "")
    print("\x1f".join([ln, stem, name, env, args, shown, why, plan, str(cost), tkey, label, str(c)]))
wall = max(sums["A"], sums["B"]) + sums["X"]
print("\x1f".join(["=", str(sums["A"]), str(sums["B"]), str(sums["X"]), str(wall), str(len(defaults)), " ".join(defaults)]))
# The run order: lanes A and X in record order, lane B's fixed items (class B: a solo-real recipe that picks the A6000 in the
# fixture tier) first in record order, then its balanced items in the order the balance placed them — its skips, then
# longest first. Every lane-A item opens with a release build, and cargo holds one build lock per target
# directory for the whole box tree: a lane-B build that starts at t = 0 (a device crate's lib tests build
# for ~100 s) makes lane A's first build wait for it. When the longest balanced item runs on the Mac
# (check-recipes, ~130 s, in a narrowed landing list), longest first starts the long builds behind it,
# while lane A runs its first binaries; when it is a box build, lane B opens with that build as before.
order = ([i for i in range(len(recs)) if placed[i][0] == "A"] + [i for i in range(len(recs)) if recs[i][0] == "B"]
         + [i for i in seq if placed[i][0] == "B"] + [i for i in range(len(recs)) if placed[i][0] == "X"])
if sorted(order) != list(range(len(recs))):
    fail(f"the run order is not a permutation of the {len(recs)} items: {order}")
print("\x1f".join(["O", " ".join(map(str, order))]))
PY

balance() { # PYBAL over the records and their candidates' ledger states: the P_* arrays and SUM_*
  local i j k st feed='' lane stem name env args item why plan tkey tcard cand plan_out
  for ((i = 0; i < N; i++)); do
    j=${R_C0[$i]}
    if [ $((i + 1)) -lt "$N" ]; then k=${R_C0[$((i + 1))]}; else k=$NC; fi
    st=''
    for ((; j < k; j++)); do st="${st:+$st }${C_LST[$j]}"; done
    feed="$feed${R_REC[$i]}"$'\x1f'"$st"$'\n'
  done
  plan_out=$(printf '%s' "$feed" | python3 -c "$PYBAL" "$LANES") || RC=70 die "the lane balance failed (above)"
  i=0 SUM_W=''
  while IFS=$'\x1f' read -r lane stem name env args item why plan _ tkey tcard cand; do
    if [ "$lane" = "=" ]; then
      SUM_A=$stem SUM_B=$name SUM_X=$env SUM_W=$args SUM_NDEF=$item SUM_DEF=$why
      continue
    fi
    if [ "$lane" = O ]; then
      read -r -a ORDER <<< "$stem"
      continue
    fi
    P_LANE+=("$lane"); P_STEM+=("$stem"); P_NAME+=("$name"); P_ENV+=("$env"); P_ARGS+=("$args")
    P_ITEM+=("$item"); P_WHY+=("$why"); P_PLAN+=("$plan"); P_TKEY+=("$tkey"); P_TCARD+=("$tcard")
    j=$((${R_C0[$i]} + cand))
    P_CI+=("$j"); P_KEY+=("${C_KEY[$j]}"); P_LST+=("${C_LST[$j]}"); P_LDET+=("${C_LDET[$j]}")
    i=$((i + 1))
  done <<< "$plan_out"
  [ "$i" = "$N" ] && [ -n "$SUM_W" ] && [ "${#ORDER[@]}" = "$N" ] || RC=70 die "the lane balance returned $i of $N items and an order of ${#ORDER[@]}"
}
ORDER=() # the run order (PYBAL's `O` record): plan indices, each lane's items in the order they run
P_LANE=() P_STEM=() P_NAME=() P_ENV=() P_ARGS=() P_ITEM=() P_WHY=() P_PLAN=() P_TKEY=() P_TCARD=()
P_CI=() P_KEY=() P_LST=() P_LDET=() # the candidate taken, its ledger key, status and detail
SUM_A=0 SUM_B=0 SUM_X=0 SUM_W=0 SUM_NDEF=0 SUM_DEF=''

box_env_of() { # the item's full BLOOMERY_BOX_ENV: the caller's, then the item's and the lane's
  local e="${BLOOMERY_BOX_ENV:-}"
  [ -z "${P_ENV[$1]}" ] || e="${e:+$e }${P_ENV[$1]}"
  printf '%s' "$e"
}
cmd_of() {
  local e; e=$(box_env_of "$1")
  printf '%sjust %s%s' "${e:+BLOOMERY_BOX_ENV='$e' }" "${P_NAME[$1]}" "${P_ARGS[$1]:+ ${P_ARGS[$1]}}"
}
item_of() { # the item as it runs, for the key: NAME[@its env and its lane's card][:ARGS]
  item_str "${P_NAME[$1]}" "${P_ENV[$1]}" "${P_ARGS[$1]}"
}
def_cmd_of() { # a deferred item's command as it would have run (the dry run names it; the batch never calls it)
  local e="${BLOOMERY_BOX_ENV:-}"
  [ -z "${DP_ENV[$1]}" ] || e="${e:+$e }${DP_ENV[$1]}"
  printf '%sjust %s%s' "${e:+BLOOMERY_BOX_ENV='$e' }" "${DP_NAME[$1]}" "${DP_ARGS[$1]:+ ${DP_ARGS[$1]}}"
}

LEAD_LEDGER=${BLOOMERY_GATE_LEDGER:-$HOME/.cache/bloomery/gate-ledger.tsv}
ROUND_LEDGER=${BLOOMERY_GATE_ROUND_LEDGER:-$HOME/.cache/bloomery/gate-ledger-rounds.tsv}
[ "$LEAD_LEDGER" != "$ROUND_LEDGER" ] || die "BLOOMERY_GATE_LEDGER and BLOOMERY_GATE_ROUND_LEDGER name one file ($LEAD_LEDGER): a round's green would land in the lead's ledger"
# The file this batch writes, and the second file it reads (empty: none).
if [ "$LMODE" = round ]; then LEDGER_FILE=$ROUND_LEDGER; else LEDGER_FILE=$LEAD_LEDGER; fi
ALSO_READ=''
if [ "$LMODE" = round ] || [ "$TRUST" = 1 ]; then ALSO_READ=$ROUND_LEDGER; fi
LEDGER_PARTS=$LEDGER_FILE.parts

ledger_plan() { # the box manifest, then every candidate's key and status (C_*), before the lease check
  local m=$1/box-manifest.txt merr='' rc=0 try i key item status detail kargs=()
  for try in 1 2; do
    rc=0
    BLOOMERY_BOX_WAIT=0 "$ROOT/tools/box.sh" "python3 tools/recipes.py box-manifest" > "$m" 2> "$1/box-manifest.err" || rc=$?
    if [ "$rc" = 0 ] || [ "$rc" = 75 ]; then break; fi
    [ "$try" = 2 ] || sleep 10
  done
  if [ "$rc" = 75 ]; then
    if [ "$DRY" = 0 ]; then
      grep '^\[' "$1/box-manifest.err" >&2 || true
      RC=75 die "the timing lease or a hold is up (above) — a batch's builds contaminate a sitting; not starting"
    fi
    merr="the timing lease or a hold was up, so the manifest was not read"
  elif [ "$rc" != 0 ]; then
    merr="box.sh rc=$rc: $(tail -1 "$1/box-manifest.err")"
    echo "gate-batch: ledger: the box manifest failed ($merr) — every item runs and none is recorded" >&2
  fi
  kargs=(key --manifest "$m" --box-env "${BLOOMERY_BOX_ENV:-}" --ledger "$LEAD_LEDGER" --parts-dir "$1/parts")
  [ -z "$ALSO_READ" ] || kargs+=(--round-ledger "$ALSO_READ")
  [ -z "$merr" ] || kargs+=(--manifest-error "$merr")
  [ "$RERUN" = 0 ] || kargs+=(--rerun)
  rc=0
  python3 "$ROOT/tools/recipes.py" "${kargs[@]}" "${C_ITEM[@]}" > "$1/keys.tsv" 2> "$1/keys.err" || rc=$?
  [ ! -s "$1/keys.err" ] || sed 's/^/gate-batch: ledger: /' "$1/keys.err" >&2
  i=0
  if [ "$rc" = 0 ]; then
    while IFS=$'\t' read -r key item status detail; do
      if [ "$i" -ge "$NC" ] || [ "$item" != "${C_ITEM[$i]}" ]; then i=-1; break; fi
      C_KEY[$i]=$key C_LST[$i]=$status C_LDET[$i]=$detail
      i=$((i + 1))
    done < "$1/keys.tsv"
  fi
  if [ "$i" != "$NC" ]; then
    echo "gate-batch: ledger: tools/recipes.py key failed (rc=$rc, $i of $NC lines in order) — every item runs and none is recorded" >&2
    for ((i = 0; i < NC; i++)); do C_KEY[$i]=- C_LST[$i]=error C_LDET[$i]="the key tool failed (rc=$rc)"; done
  fi
}

ledger_record() { # $1 = plan index, $2 = final rc: record a green item whose key did not move; print its state
  local i=$1 k2 rec
  case "${P_LST[$i]}" in
    never) echo never; return ;;
    error) echo unkeyed; return ;;
  esac
  if [ "$2" != 0 ]; then echo red; return; fi
  # The recheck's failure is named, never read as a moved input: its stderr and a keyless line
  # (`-<TAB>item<TAB>error<TAB>why`) go to the batch's stderr, and the state is `unkeyed`.
  local out err=$LWORK/recheck-$i.err rc=0
  out=$(python3 "$ROOT/tools/recipes.py" key --manifest "$LWORK/box-manifest.txt" --box-env "${BLOOMERY_BOX_ENV:-}" "$(item_of "$i")" 2> "$err") || rc=$?
  k2=${out%%$'\t'*}
  if [ "$rc" != 0 ] || [ -z "$k2" ] || [ "$k2" = - ]; then
    echo "gate-batch: ledger: the key recheck of $(item_of "$i") failed (rc=$rc): $(tail -1 "$err")${out:+ $out}" >&2
    echo unkeyed
    return
  fi
  if [ "$k2" != "${P_KEY[$i]}" ]; then echo changed; return; fi
  rec=$(printf '%s\t%s\t%s\t%s\t%s\t%s' "${P_KEY[$i]}" "${P_NAME[$i]}" "$(item_of "$i")" "$COMMIT" "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$ROOT")
  if python3 -c "$PYAPPEND" "$LEDGER_FILE" "$rec" "$LWORK/parts/${P_CI[$i]}.parts" "$LEDGER_PARTS/${P_KEY[$i]}.parts"; then
    echo recorded
  else
    echo append-failed
  fi
}

if [ -z "$OUT" ]; then
  OUT="$HOME/.cache/bloomery/batches/$(basename "$ROOT")/$(date +%Y%m%d-%H%M%S)"
fi
case "$OUT" in /*) ;; *) OUT="$PWD/$OUT" ;; esac
case "$OUT/" in
  "$ROOT"/target/*) ;;
  "$ROOT"/*) die "--out $OUT is inside the tree but not under target/: box.sh would ship the logs and mark the tree dirty" ;;
esac

# The disk floor (the header): both volumes before the log directory exists and any lane starts. A
# below-floor verdict (rc 1, the line printed by disk_ok) ends a real run at 69; a dry run says so
# and continues, and a df that could not be read (69) fails it too.
rc=0; disk_ok "$ROOT/target" || rc=$?
case $rc in 0) ;; 1) below=1 ;; *) exit "$rc" ;; esac
rc=0; disk_ok "$OUT" || rc=$?
case $rc in 0) ;; 1) below=1 ;; *) exit "$rc" ;; esac
if [ "${below:-0}" = 1 ]; then
  if [ "$DRY" = 0 ]; then exit 69; fi
  echo "gate-batch: dry run — a real run would not start (the disk floor above)" >&2
fi

if [ "$DRY" = 0 ]; then
  mkdir -p "$OUT" || RC=73 die "cannot create the log directory $OUT"
  [ -d "$OUT" ] && [ -w "$OUT" ] || RC=73 die "the log directory $OUT is not a writable directory"
  [ ! -e "$OUT/run.log" ] || RC=73 die "$OUT already holds a batch (run.log) — give --out a new directory"
fi
RUNLOG=$OUT/run.log

SKIP_N=0
if [ "$LEDGER" = 1 ]; then
  if [ "$DRY" = 1 ]; then
    LWORK=$(mktemp -d "${TMPDIR:-/tmp}/gate-batch-ledger.XXXXXX") || RC=73 die "cannot make a scratch directory for the ledger"
    trap 'rm -rf "$LWORK"' EXIT
  else
    LWORK=$OUT/ledger
    mkdir -p "$LWORK" || RC=73 die "cannot create $LWORK"
    COMMIT=$(git -C "$ROOT" rev-parse --short=12 HEAD 2> /dev/null || echo unknown)
    [ -z "$(git -C "$ROOT" status --porcelain 2> /dev/null | head -1)" ] || COMMIT="$COMMIT-dirty"
  fi
  [ "$NC" = 0 ] || ledger_plan "$LWORK" # a batch whose items are all deferred has nothing to key
fi
balance
for ((i = 0; i < N; i++)); do [ "${P_LST[$i]}" != skip ] || SKIP_N=$((SKIP_N + 1)); done
if [ "$LEDGER" = 1 ]; then
  echo "gate-batch: ledger: reads $LEAD_LEDGER${ALSO_READ:+ and $ALSO_READ}, writes $LEDGER_FILE: $SKIP_N of $N items skip"
fi
if [ "$LANES" = 1 ]; then
  PREDICTED="laneA=${SUM_A}s wall=${SUM_W}s"
else
  PREDICTED="laneA=${SUM_A}s laneB=${SUM_B}s laneX=${SUM_X}s wall=${SUM_W}s"
fi
predicted() { # the plan's lane sums, derived from the times file
  local how="the median of each item's last 5 rows in $TIMES_FILE"
  [ "$LEDGER" = 0 ] || how="$how, a skipped item 0 s"
  [ "$LANES" = 1 ] || how="$how; wall = the longer of A and B, then X"
  echo "gate-batch: predicted $PREDICTED (derived: $how)"
  [ "$SUM_NDEF" = 0 ] || echo "gate-batch: $SUM_NDEF item(s) expected at the ${DEFAULT_S} s default (no row in the times file): $SUM_DEF"
  [ "$ND" = 0 ] || echo "gate-batch: $ND item(s) deferred to the real tier ($ND_RO real-only, $ND_NF of a family with no fixture), in no lane (DONE: deferred=+$ND, and the list $OUT/deferred.list): ${DP_STEM[*]}"
}

# The GPU hold (the header's «The GPU hold»). The box-side writer is tools/gpu-gate.sh --gpuhold, reached through box.sh with
# BLOOMERY_BOX_READONLY=1 (no sync, so no rsync under a running batch) and BLOOMERY_BOX_WAIT=0 (no guard wait: a heartbeat held in a
# sitting's guard would go stale), stdin closed (the read-only ssh is the one that reads it).
hold_box() { # $1 = up | beat | down: the box-side writer's output and rc
  BLOOMERY_BOX_READONLY=1 BLOOMERY_BOX_WAIT=0 "$ROOT/tools/box.sh" "bash tools/gpu-gate.sh --gpuhold $1 $HOLD_OWNER" < /dev/null
}
hold_path() { bash "$ROOT/tools/gpu-gate.sh" --gpuhold path 2> /dev/null || echo '(the path tools/gpu-gate.sh --gpuhold path prints)'; }
hold_say() { # the dry run's line: the hold this batch would take
  if [ "$HOLD_ON" = 1 ]; then
    echo "gate-batch: gpu hold: would put $(hold_path) up on the box as $HOLD_OWNER, refreshed every ${HOLD_BEAT} s; each item gets BLOOMERY_BATCH_OWNER=$HOLD_OWNER, and other tracks' tools/gpu-gate.sh runs wait until the batch ends"
  elif [ "$LMODE" = lead ]; then
    echo "gate-batch: gpu hold: off (--no-gpu-hold)"
  else
    echo "gate-batch: gpu hold: none (only a --ledger batch takes it)"
  fi
}
hold_beat_loop() { # the heartbeat, a background job: one refresh of the hold every HOLD_BEAT s until TERM, until the hold is gone,
  # or until the batch is (a batch that dies uncleanly runs no trap: its heartbeat must not keep the hold fresh after it)
  local sp='' rc
  trap - EXIT
  trap '[ -z "$sp" ] || kill "$sp" 2> /dev/null; exit 0' TERM
  while :; do
    sleep "$HOLD_BEAT" &
    sp=$!
    wait "$sp" || true
    if ! kill -0 "$$" 2> /dev/null; then # $$ is the batch's own pid in this subshell too
      echo "$(date '+%F %T') the batch (pid $$) is gone: the heartbeat stops, and the box calls the hold stale after 300 s" >> "$OUT/gpuhold.log"
      exit 0
    fi
    rc=0
    hold_box beat >> "$OUT/gpuhold.log" 2>&1 &
    sp=$!
    wait "$sp" || rc=$?
    case $rc in
      0) echo "$(date '+%F %T') beat ok" >> "$OUT/gpuhold.log" ;;
      3) echo "$(date '+%F %T') the hold is gone or another batch's: the heartbeat stops" >> "$OUT/gpuhold.log"; exit 0 ;;
      *) echo "$(date '+%F %T') beat failed rc=$rc: a hold nobody refreshes goes stale on the box by itself" >> "$OUT/gpuhold.log" ;;
    esac
  done
}
hold_up() { # the hold goes up (HOLD_UP=1), or the batch does not start: 75 while another batch's fresh hold is up, 70 otherwise
  local out rc=0
  out=$(hold_box up 2>&1) || rc=$?
  { echo "$(date '+%F %T') up rc=$rc"; printf '%s\n' "$out"; } >> "$OUT/gpuhold.log"
  case $rc in
    0) HOLD_UP=1 ;;
    75) printf '%s\n' "$out" >&2; RC=75 die "another batch's GPU hold is up (above) — not starting; wait for it, or --no-gpu-hold runs beside it" ;;
    *) printf '%s\n' "$out" >&2; RC=70 die "the GPU hold could not be put up (rc $rc, above; tools/gpu-gate.sh --gpuhold through box.sh) — not starting; --no-gpu-hold runs without it" ;;
  esac
}
hold_down() { # the EXIT trap: the heartbeat's pid from DIR, then the hold, only while it is still this batch's
  [ "$HOLD_UP" = 1 ] || return 0
  HOLD_UP=0
  local p rc=0
  if [ -f "$OUT/gpuhold.pid" ]; then
    p=$(cat "$OUT/gpuhold.pid")
    kill -TERM "$p" 2> /dev/null || true
    wait "$p" 2> /dev/null || true
  fi
  hold_box down >> "$OUT/gpuhold.log" 2>&1 || rc=$?
  if [ "$rc" = 0 ]; then
    echo "gate-batch: gpu hold: down"
  else
    echo "gate-batch: the GPU hold was not taken down (rc $rc, $OUT/gpuhold.log): the box calls it stale after 300 s without a beat" >&2
  fi
}

stop() {
  local f
  # The lanes first, so none starts its next item, then the items they were running.
  for f in "$OUT"/lane-*.pid "$OUT"/lane-*.child; do
    if [ -f "$f" ]; then kill -TERM "$(cat "$f")" 2> /dev/null || true; fi
  done
  echo "gate-batch: stopped; a box process a killed item left behind is cleared by 'just box-gc'" >&2
  exit 130
}

if [ "$DRY" = 1 ]; then
  echo "gate-batch: dry run — $N items$DEFER_NOTE, lanes $LANES, logs would go to $OUT"
  for i in "${ORDER[@]}"; do
    printf 'lane %s  %-28s %s\n        %s — %s\n' "${P_LANE[$i]}" "${P_STEM[$i]}" "$(cmd_of "$i")" "${P_PLAN[$i]}" "${P_WHY[$i]}"
    [ "$LEDGER" = 0 ] || printf '        ledger: %s — %s\n' "${P_LST[$i]}" "${P_LDET[$i]}"
    if [ "$LANES" = 2 ]; then
      IFS=$'\x1f' read -r cls _ _ _ _ _ _ _ _ _ _ _ _ <<< "${R_REC[$i]}"
      if [ "$cls" = F ] && [ "${P_LST[$i]}" != skip ]; then
        printf '        movable — an idle lane may take it on its card (never a fixed, solo or v41-load item)\n'
      fi
    fi
  done
  for ((i = 0; i < ND; i++)); do
    printf 'lane -  %-28s %s\n        deferred to the real tier (%s), never run in the fixture tier — %s\n' "${DP_STEM[$i]}" "$(def_cmd_of "$i")" "${DP_KIND[$i]}" "${DP_WHY[$i]}"
  done
  predicted
  hold_say
  exit 0
fi

# The start check is box.sh's guard with no wait (the header): the lease or a hold up is 75 at once,
# with the guard's lines naming what is up (the lease's holders through lease_holders, each hold's
# owner and age); a lease that cannot be tested is 70.
if [ "$SKIP_N" -lt "$N" ]; then
  rc=0
  lease=$(BLOOMERY_BOX_WAIT=0 "$ROOT/tools/box.sh" true 2>&1) || rc=$?
  case $rc in
    0) ;;
    75) echo "$lease" >&2; RC=75 die "the timing lease or a hold is up (above) — a batch's builds contaminate a sitting; not starting" ;;
    *) echo "$lease" >&2; RC=70 die "the start check through box.sh failed (rc $rc; 70 from the guard: the lease cannot be tested)" ;;
  esac
elif [ "$N" = 0 ]; then
  echo "gate-batch: every item is deferred to the real tier — nothing runs on the box, so no lease check"
else
  echo "gate-batch: every item skips — nothing runs on the box, so no lease check"
fi

# The traps, then the GPU hold (the header): the hold goes up only when some item runs on the box, after the start check, and a
# refusal here leaves no run.log, as the start check's does. Every exit from here on — the end, a red batch, die, INT/TERM (stop) —
# takes it down through the EXIT trap.
trap stop INT TERM
trap hold_down EXIT
if [ "$HOLD_ON" = 1 ] && [ "$SKIP_N" -lt "$N" ]; then
  hold_up
  hold_beat_loop &
  echo $! > "$OUT/gpuhold.pid"
  echo "gate-batch: gpu hold: up as $HOLD_OWNER ($(hold_path), refreshed every ${HOLD_BEAT} s, log $OUT/gpuhold.log): other tracks' GPU gates wait until this batch ends"
fi

# run.log exists from here on (a refused start leaves none, so the same --out can be retried).
: > "$RUNLOG"
mkdir -p "$OUT/claims" # one claim per item: the owner's start and the thief's steal agree on it
T0=$(date +%s)
echo "plan $PREDICTED defaults=$SUM_NDEF (predicted, derived from $TIMES_FILE)" >> "$RUNLOG"
echo "gate-batch: $N items$DEFER_NOTE, lanes $LANES, logs in $OUT"
predicted
for ((i = 0; i < ND; i++)); do # a deferred item is recorded at once: the batch never calls it (nothing to time, key or claim)
  line="${DP_STEM[$i]} rc=deferred ${DP_KIND[$i]} lane=- deferred=1"
  [ -z "${DP_ITEM[$i]}" ] || line="$line item=${DP_ITEM[$i]}"
  echo "$line" >> "$RUNLOG"
  echo "$line"
done
DEFER_LIST=$OUT/deferred.list
if [ "$ND" != 0 ]; then
  { # the deferred items in --list format, so a real-tier batch takes exactly this set (`--list DIR/deferred.list`)
    echo "# gate-batch --tier fixture: the $ND item(s) it deferred to the real tier, one per line (the --list format)"
    for kind in real-only no-fixture; do
      if [ "$kind" = real-only ]; then
        [ "$ND_RO" = 0 ] || echo "# real-only: no fixture-tier conversion (bash tools/ref/real-only.sh in the recipe's box command)"
      else
        [ "$ND_NF" = 0 ] || echo "# no-fixture: the recipe's family has no row in the fixture table of tools/ref/ref-paths.sh (box.sh exits 66)"
      fi
      for ((i = 0; i < ND; i++)); do
        [ "${DP_KIND[$i]}" = "$kind" ] || continue
        echo "${DP_ITEM[$i]:-${DP_NAME[$i]}}"
      done
    done
  } > "$DEFER_LIST" || RC=73 die "cannot write $DEFER_LIST"
fi


run_item() { # $1 = plan index, $2 = lane label; the lane's current child pid goes to lane-<lane>.child
  local i=$1 lane=$2 log="$OUT/g-${P_STEM[$1]}.log" try=0 rc t0 t1 ran s benv line waits=0 cold=0 deferred=0
  local argv=()
  eval "argv=(${P_ARGS[$i]})"
  benv=$(box_env_of "$i")
  # The batch's own runs pass its GPU hold: the owner joins the call's env here, never box_env_of's (which feeds the ledger key).
  [ "$HOLD_UP" != 1 ] || benv="${benv:+$benv }BLOOMERY_BATCH_OWNER=$HOLD_OWNER"
  t0=$(date +%s)
  while :; do
    try=$((try + 1))
    echo "=== try $try $(date '+%F %T') lane=$lane: $(cmd_of "$i")" >> "$log"
    rc=0
    t1=$(date +%s)
    BLOOMERY_BOX_ENV="$benv" "${JUST[@]}" "${P_NAME[$i]}" ${argv[@]+"${argv[@]}"} >> "$log" 2>&1 < /dev/null &
    echo $! > "$OUT/lane-$lane.child"
    wait $! || rc=$?
    ran=$(($(date +%s) - t1))
    waits=$(try_waits "$log" "$try")
    if [ "$rc" -eq 75 ] && [ "$try" -lt "$TRIES_MAX" ]; then
      echo "=== rc 75 (lock contention): retry in ${RETRY_WAIT} s" >> "$log"
      sleep "$RETRY_WAIT"
      continue
    fi
    break
  done
  s=$(($(date +%s) - t0))
  line="${P_STEM[$i]} rc=$rc ${s}s try=$try lane=$lane"
  [ "$waits" = 0 ] || line="$line waited=${waits}s"
  ran=$((ran > waits ? ran - waits : 0))
  cold=$(cold_of "$log" "$try")
  [ "$cold" = 0 ] || line="$line cold=1"
  deferred=$(deferred_of "$log" "$try")
  [ "$deferred" = 0 ] || line="$line deferred=$deferred"
  [ "$LEDGER" = 0 ] || line="$line ledger=$(ledger_record "$i" "$rc")"
  # The final try's seconds, green or red; a last try still at rc 75 got no lock and ran nothing. A cold
  # build's row goes to COLD_FILE, which no plan reads.
  if [ "$rc" != 75 ] && ! times_record "$i" "$lane" "$ran" "$cold"; then
    line="$line times=append-failed"
    echo "gate-batch: ${P_STEM[$i]}: its row did not reach $([ "$cold" = 0 ] && echo "$TIMES_FILE" || echo "$COLD_FILE") (above)" >&2
  fi
  [ -z "${P_ITEM[$i]}" ] || line="$line item=${P_ITEM[$i]}"
  echo "$line" >> "$RUNLOG"
  echo "$line"
}

skip_item() { # $1 = plan index, $2 = lane label: the ledger holds a green run of these inputs
  local i=$1 line
  line="${P_STEM[$i]} rc=skip ${P_LDET[$i]% tree=*} lane=$2"
  [ -z "${P_ITEM[$i]}" ] || line="$line item=${P_ITEM[$i]}"
  echo "$line" >> "$RUNLOG"
  echo "$line"
}

# Work stealing (the header's paragraph above the lanes): one claim per item, under DIR/claims —
# the owner claims an item before it starts it, the thief before it steals one, and an item runs
# exactly once however the two lanes race.
claim_item() { # $1 = plan index, $2 = the claiming lane: wins the item's claim?
  local f="$OUT/claims/$1"
  ( set -o noclobber; printf '%s\n' "$2" > "$f" ) 2> /dev/null
}

claimed() { [ -e "$OUT/claims/$1" ]; }

# The card this lane would run a balanced item on — its candidate for the lane's card, `-` for a
# no-card item — or empty when the item has no candidate here (a fixed item's single card, or the
# other card only).
steal_card() { # $1 = plan index, $2 = lane
  local cards want
  want=$([ "$2" = A ] && printf 3090 || printf a6000)
  IFS=$'\x1f' read -r _ _ _ _ _ _ _ _ _ _ _ cards _ <<< "${R_REC[$1]}"
  case " $cards " in
    ' - ') printf -- '-' ;;
    *" $want "*) printf '%s' "$want" ;;
  esac
}

# movable: the other lane's item this lane may take — class F (the plan's own word that either
# card, or no card, runs it), not a ledger skip, not yet claimed.
stealable() { # $1 = plan index, $2 = lane
  local i=$1 cls other
  other=$([ "$2" = A ] && printf B || printf A)
  [ "${P_LANE[$i]}" = "$other" ] || return 1
  IFS=$'\x1f' read -r cls _ _ _ _ _ _ _ _ _ _ _ _ <<< "${R_REC[$i]}"
  [ "$cls" = F ] || return 1
  [ "${P_LST[$i]}" != skip ] || return 1
  ! claimed "$i" || return 1
  [ -n "$(steal_card "$i" "$2")" ]
}

# Rewrite a stolen item's placement onto this lane's card: the env, the times label and the ledger
# key, state and candidate move to the item's other candidate, so the row and the record name the
# card it ran on. A no-card item needs none of it (its one candidate runs anywhere).
switch_candidate() { # $1 = plan index, $2 = the lane's card (or -)
  local i=$1 card=$2 env cards labels c pos=-1 k=0 l='' lc=0 j
  IFS=$'\x1f' read -r _ _ _ env _ _ _ _ _ _ _ cards labels _ <<< "${R_REC[$i]}"
  [ "$card" = - ] || env="${env:+$env }BLOOMERY_GATE_CARD=$card"
  P_ENV[$i]=$env
  for c in $cards; do
    [ "$c" = "$card" ] && pos=$k
    k=$((k + 1))
  done
  [ "$pos" -ge 0 ] || return 0
  for c in $labels; do
    [ "$lc" = "$pos" ] && l=$c
    lc=$((lc + 1))
  done
  j=$((${R_C0[$i]} + pos))
  P_TCARD[$i]=$l
  P_CI[$i]=$j P_KEY[$i]=${C_KEY[$j]} P_LST[$i]=${C_LST[$j]} P_LDET[$i]=${C_LDET[$j]}
}

# The timing-lease probe of a steal: lease_free through box.sh, READONLY so there is no rsync and
# no guard wait (the probe is itself the read — the shared-lock probe, never an exclusive flock).
# Free prints 0; held, or a lease that cannot be tested (named), prints nonzero and the steal takes
# no card.
lease_free_for_steal() {
  local out rc=0
  out=$(BLOOMERY_BOX_READONLY=1 BLOOMERY_BOX_WAIT=0 "$ROOT/tools/box.sh" '. tools/ref/lease-probe.sh && lease_free' 2>&1) || rc=$?
  if [ "$rc" != 0 ] && [ "$rc" != 1 ]; then
    echo "gate-batch: the steal's lease probe through box.sh failed (rc $rc): $out — taking no card until it reads free" >&2
  fi
  [ "$rc" = 0 ]
}

# The steal pass: while the other lane has an unstarted movable item, take the next one in run
# order onto this lane's card. A steal onto a card probes the timing lease first and waits it out
# in 1 s slices (up to RETRY_WAIT) while candidates remain: never a card while the lease is held,
# and no lane sleeps past the last candidate. The loop always ends: every turn runs an item, loses
# a claim, waits a slice, or finds no candidate.
steal_pass() { # $1 = lane
  local lane=$1 i j card k
  while :; do
    i=-1
    for j in "${ORDER[@]}"; do
      if stealable "$j" "$lane"; then i=$j; break; fi
    done
    [ "$i" -ge 0 ] || return 0
    card=$(steal_card "$i" "$lane")
    if [ "$card" != - ] && ! lease_free_for_steal; then
      for ((k = 0; k < STEAL_WAIT; k++)); do
        sleep 1
        i=-1
        for j in "${ORDER[@]}"; do
          if stealable "$j" "$lane"; then i=$j; break; fi
        done
        [ "$i" -ge 0 ] || return 0
      done
      continue
    fi
    claim_item "$i" "steal-$lane" || continue
    switch_candidate "$i" "$card"
    run_item "$i" "$lane"
  done
}

run_lane() { # $1 = lane label; runs its items in the run order (ORDER), then steals the other
  # lane's unstarted movable items (above), then writes lane-<lane>.s
  local lane=$1 t0 i
  t0=$(date +%s)
  for i in "${ORDER[@]}"; do
    [ "${P_LANE[$i]}" = "$lane" ] || continue
    if [ "${P_LST[$i]}" = skip ]; then
      skip_item "$i" "$lane"
    else
      claim_item "$i" "$lane" || continue # the other lane stole it while this one was busy
      run_item "$i" "$lane"
    fi
  done
  case $lane in
    A | B) [ "$LANES" = 2 ] && steal_pass "$lane" ;;
  esac
  echo $(($(date +%s) - t0)) > "$OUT/lane-$lane.s"
}

has_lane() { local i; for ((i = 0; i < N; i++)); do [ "${P_LANE[$i]}" = "$1" ] && return 0; done; return 1; }

LANES_RUN=""
for lane in A B; do
  if has_lane "$lane"; then
    run_lane "$lane" &
    echo $! > "$OUT/lane-$lane.pid"
    LANES_RUN="$LANES_RUN $lane"
  fi
done
for lane in $LANES_RUN; do
  wait "$(cat "$OUT/lane-$lane.pid")" || true
done
# Every lane is waited for before any is judged: a lane that died must not leave the other running.
for lane in $LANES_RUN; do
  [ -f "$OUT/lane-$lane.s" ] || RC=70 die "lane $lane ended without its sentinel ($OUT/lane-$lane.s) — see run.log"
done
if has_lane X; then run_lane X; fi
trap - INT TERM

recorded=$(grep -c ' rc=' "$RUNLOG" || true)
[ "$recorded" -eq "$NT" ] || RC=70 die "$NT items planned, $recorded recorded in $RUNLOG"
green=$(grep -c ' rc=0 ' "$RUNLOG" || true)
skipped=$(grep -c ' rc=skip ' "$RUNLOG" || true)
red=$((NT - green - skipped - ND))
lane_s() { if [ -f "$OUT/lane-$1.s" ]; then cat "$OUT/lane-$1.s"; else echo 0; fi; }
done_line="DONE total=$NT red=$red"
[ "$LEDGER" = 0 ] || done_line="$done_line skipped=$skipped"
done_line="$done_line wall=$(($(date +%s) - T0))s laneA=$(lane_s A)s laneB=$(lane_s B)s"
if has_lane X; then done_line="$done_line laneX=$(lane_s X)s"; fi
for ((i = 0; i < N; i++)); do
  if [ "${P_NAME[$i]}" = lint ]; then
    if [ "${P_LST[$i]}" = skip ]; then
      done_line="$done_line lint_warnings=skip"
    else
      done_line="$done_line lint_warnings=$(grep -c '^warning:' "$OUT/g-${P_STEM[$i]}.log" || true)"
    fi
    break
  fi
done
# The fixture tier's clauses left to the real tier, summed over the items' lines (the lanes ran in subshells).
if [ "$TIER" = fixture ]; then
  done_line="$done_line deferred=$(awk '{ for (i = 1; i <= NF; i++) if ($i ~ /^deferred=[0-9]+$/) n += substr($i, 10) } END { print n + 0 }' "$RUNLOG")"
  [ "$ND_RO" = 0 ] || done_line="$done_line real_only=$ND_RO"
  [ "$ND_NF" = 0 ] || done_line="$done_line no_fixture=$ND_NF"
  [ "$ND" = 0 ] || done_line="$done_line deferred_list=$DEFER_LIST"
fi
echo "$done_line" >> "$RUNLOG"
echo "$done_line"
if [ "$red" -ne 0 ]; then
  grep -v ' rc=0 ' "$RUNLOG" | grep ' rc=' | grep -v ' rc=skip ' | grep -v ' rc=deferred ' | while read -r stem _; do echo "red: $stem  $OUT/g-$stem.log"; done
  exit 1
fi
