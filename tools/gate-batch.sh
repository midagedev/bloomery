#!/usr/bin/env bash
# The lead's gate batch runner, two lanes. Runs on the Mac; every item is `just <recipe> [ARGS]`, so
# the box is reached only through the recipes (tools/box.sh) and each item keeps the bound and the
# exit code its recipe's runner owns (tools/gate.sh, tools/gpu-gate.sh, 900 s). This script adds no
# second bound.
#   tools/gate-batch.sh [--out DIR] [--smoke | --weekly | --list FILE | ITEM…] [--dry-run] [--lanes 1|2] [--ledger [--trust-rounds] | --round-ledger] [--rerun]
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
#      BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} (V4.1 `--place gate` gates, the V2-Lite p-gates), or device code run with no gate lock at all
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
#      than solo, v41-load and host is a named error — a group is a scheduling class here, and a typo must
#      not drop a gate out of its class.
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
#      anyway, so lane sums that ran them in parallel would be wrong, and the waiting one would hold its
#      lane's card lock through the wait.
# A recipe that runs a timing runner or takes the timing lease is refused, because a batch's builds
# contaminate a timed run. Two tests: what a script the recipe runs does, followed transitively — a
# tools/…sh that calls lease_take (tools/ref/lease.sh), or that opens the lease lock for a descriptor
# and waits on it with flock -w — and, for what that walk cannot see, the names in timed() below. Also
# refused: a recipe whose text runs this script (`just smoke`: a batch inside a batch runs lanes of its
# own on the same cards), and an item whose ARGS hand a `[group('solo')]` recipe's gpu-gate.sh binary
# one of the --flags that recipe passes it (in the self-test's fixture, `v41-a:--faults` is the solo
# recipe v41-solo's arm, which runs alone and with BLOOMERY_HOST_LOCK=1) — the error names the solo
# recipe to list instead. Each lane runs its items in list order; A and B run
# concurrently (cargo serializes their builds by its own lock). --lanes 1 runs every item in the
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
# (`laneX=` when X ran, `lint_warnings=<n>` when lint ran: `grep -c '^warning:'` on its log). Exit 0
# iff every rc is 0. DIR defaults to $HOME/.cache/bloomery/batches/<tree>/<stamp> (<tree> the
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
DEFAULT_S=45 # the expected seconds of an item with no row in the times file (the header says why 45)
MIN_FREE_GIB=3 # the disk floor (the header's «Disk floor»): 2× the larger of one cold check's and one cold combos' growth of a tree's target/ (tools/mac-check.sh), rounded up to a whole GiB
TIMES_FILE=${BLOOMERY_GATE_TIMES:-$HOME/.cache/bloomery/gate-times.tsv}
COLD_FILE=${TIMES_FILE%.tsv}-cold.tsv # a cold build's rows, outside the median (the header)

USAGE="usage: tools/gate-batch.sh [--out DIR] [--smoke | --weekly | --list FILE | ITEM…] [--dry-run] [--lanes 1|2] [--ledger [--trust-rounds] | --round-ledger] [--rerun] | --classes | --self-test"
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

# try_waits <log> <try>: the seconds the given try spent waiting, not running — box.sh's guard for a sitting
# (`[guard] … after N s: the command starts`) and tools/gpu-gate.sh for a gate lock and for the V4.1 load
# lock (`gpu-gate.sh: waited N s for the …`),
# summed over the try's section of the item's log. A times row holds the rest: a lock queue is not a gate's cost.
try_waits() {
  awk -v t="=== try $2 " '
    index($0, t) == 1 { on = 1; next }
    /^=== try / { on = 0 }
    on && match($0, /after [0-9]+ s: the command starts/) { split(substr($0, RSTART + 6), a, " "); w += a[1] }
    on && match($0, /^gpu-gate\.sh: waited [0-9]+ s for the (gate lock|V4\.1 load lock)/) { split(substr($0, RSTART + 20), a, " "); w += a[1] }
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
JF
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
    elif ! printf '%s\n' "$out" | grep -Eq -- "$pat"; then fail "$name" "no line matches /$pat/"
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
  if printf '%s\n' "$out" | grep -q 'gate-y'; then fail 'list: a narrowed affected output does not run a left-out recipe' "gate-y is in the batch"
  else pass 'list: a narrowed affected output does not run a left-out recipe'; fi
  # A weekly recipe a trigger named: its `  weekly-` line runs, beside the gate lines.
  printf '%s\n' 'affected: a..b — 2 changed files, 1 of 6 gate-recipes selected, 1 of 2 weekly-recipes by a trigger' \
    '  gate-x    crates/x/src/lib.rs  [lib x]' '  weekly-a  crates/w/src/lib.rs  [trigger tools/gate-paths.tsv:9 crates/w/**]' \
    'recipes: gate-x weekly-a' 'unmapped (0):' > "$t/aff-weekly.txt"
  check 'list: a weekly recipe a trigger named in affected output runs' 0 '^lane [AB]  weekly-a ' "${gb[@]}" --dry-run --list "$t/aff-weekly.txt"
  # --weekly: every weekly-* recipe of the justfile, each in its class; none of another name.
  check 'weekly: a weekly recipe with no group is balanced' 0 '^lane [AB]  weekly-a ' "${gb[@]}" --dry-run --weekly
  check 'weekly: a solo weekly recipe runs alone' 0 '^lane X  weekly-b ' "${gb[@]}" --dry-run --weekly
  if printf '%s\n' "$out" | grep -Eq '^lane [ABX]  (gate|v41|host|plain)'; then fail 'weekly: nothing but the weekly-* recipes' "another recipe is in the batch"
  else pass 'weekly: nothing but the weekly-* recipes'; fi
  check 'weekly: with items, a named error' 64 'exclusive' "${gb[@]}" --dry-run --weekly gate-x
  printf '%s\n' 'something else' 'affected: a..b — 1 changed file' '  gate-x  crates/x/src/lib.rs (+1)' > "$t/aff-other.txt"
  check 'list: any other line above the affected: header is a malformed item' 65 "aff-other.txt:1: malformed item 'something else'" \
    "${gb[@]}" --dry-run --list "$t/aff-other.txt"
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
    'gpu-gate.sh: waited 15 s for the gate lock (3090)' 'gpu-gate.sh: waited 40 s for the V4.1 load lock' > "$t/waits.log"
  out=$(try_waits "$t/waits.log" 2)
  if [ "$out" = 115 ]; then pass 'waits: the guard, the card lock and the V4.1 load lock of the final try'
  else fail 'waits: the guard, the card lock and the V4.1 load lock of the final try' "got $out, want 115"; fi
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
OUT='' SRC='' LIST='' DRY=0 LANES=2 LEDGER=0 RERUN=0 LMODE='' TRUST=0
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
    --rerun) RERUN=1; shift ;;
    --lanes) [ $# -ge 2 ] || die "--lanes needs 1 or 2; $USAGE"
      case "$2" in 1 | 2) LANES=$2 ;; *) die "--lanes is 1 or 2, got '$2'" ;; esac; shift 2 ;;
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
command -v just > /dev/null || die "just is not on PATH"
command -v python3 > /dev/null || die "python3 is not on PATH"

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

root, lanes, src, listfile, caller_env, times_path, default_s = sys.argv[1:8]
raw_items = sys.argv[8:]
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


host_bad = []
for n in sorted(recipes):
    if {"group": "host"} not in recipes[n]["attributes"]:
        continue
    text = "\n".join(body(m) for m in closure(n, set()))
    why = [f"[group('{g}')]" for g in ("solo", "v41-load") if group_of(n, g)]
    why += [w for w, hit in (("a tools/gpu-gate.sh call", GPU_GATE in text), ("a box.sh card pick", BOX_CARD.search(text))) if hit]
    if why:
        host_bad.append(f"recipe {n}: [group('host')] with {' and '.join(why)} — a host recipe opens no card and runs beside other items")
if host_bad:
    for e in host_bad:
        print("gate-batch: " + e, file=sys.stderr)
    fail(f"{len(host_bad)} host recipe(s) that open a card or run alone; nothing ran")


def solo_of(name):
    return group_of(name, "solo")


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
    if {"group": "solo"} in recipes[_n]["attributes"]:
        for _bin, _flags in gate_calls(_n):
            for _f in _flags:
                SOLO_ARMS.setdefault((_bin, _f), _n)


def solo_arm(name, argv):
    """The (flag, solo recipe) whose arm the item's ARGS select, or None."""
    if solo_of(name):
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

def placement(name, lane, lanes):
    """(class, kind, card candidates, their times-file labels) of a recipe that runs in a batch."""
    names = closure(name, set())
    uses_gpu_gate = bool(GATE_FORMS[name])
    box = next((m.group(1) for m in (BOX_CARD.search(body(n)) for n in names) if m), None)
    if lanes == "1":
        if box:
            labels = [box]
        elif uses_gpu_gate:
            labels = ["any" if "any" in GATE_FORMS[name] else "3090"]
        else:
            labels = ["3090" if lane == "A" else "none"]
        return "A", "one-lane", ["-"], labels
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
    if group_of(name, "v41-load"):
        # One lane: two V4.1 loads at once evict each other's host set from the page cache. Lane A,
        # where the 3090-only (`--place gate`) loads already are; an `any` member runs on the 3090.
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
            tags = [f"{n}: [group('{g}')]" for g in ("solo", "v41-load", "host") for n in group_of(name, g)]
            print("\t".join([name, cls, kind, ",".join(cards), "; ".join(tags + [reason])]))
    sys.exit(0)

counts, errors, out = {}, [], []
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
    tags = [f"{n}: [group('{g}')]" for g in ("solo", "v41-load", "host") for n in group_of(name, g)]
    if tags:
        reason = "; ".join(tags + [reason])
    qargs = " ".join(shlex.quote(a) for a in argv)
    tkey = name + ("@" + ",".join(envs) if envs else "") + (":" + qargs if argv else "")
    if re.search(r"[\t\n]", tkey):
        errors.append(f"{at}: {item!r} holds a tab or a newline in its env or ARGS: the times file is one tab-separated row per item")
        continue
    exp, esrc = expected(tkey)
    cls, kind, cards, labels = placement(name, lane, lanes)
    counts[name] = counts.get(name, 0) + 1
    stem = name if counts[name] == 1 else f"{name}-{counts[name]}"
    shown = item if (env is not None or args is not None) else ""
    out.append("\x1f".join([cls, stem, name, " ".join(envs), qargs, shown, reason, kind, str(exp), esrc, tkey, " ".join(cards), " ".join(labels)]))
if errors:
    for e in errors:
        print("gate-batch: " + e, file=sys.stderr)
    fail(f"{len(errors)} item error(s); nothing ran")
print("\n".join(out))
PY
PLAN0=$(python3 -c "$PYPLAN" "$ROOT" "$LANES" "$SRC" "$LIST" "${BLOOMERY_BOX_ENV:-}" "$TIMES_FILE" "$DEFAULT_S" ${ITEMS[@]+"${ITEMS[@]}"}) \
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
while IFS= read -r rec; do R_REC+=("$rec"); done <<< "$PLAN0"
N=${#R_REC[@]}
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
    if cls in ("A", "X") and len(cards) == 1:
        placed[i] = (cls, 0, 0 if states[0] == "skip" else exp, "")
        sums[cls] += placed[i][2]
    elif cls == "F" and lanes == "2" and lane_cand(cards):
        flex.append(i)
    else:
        fail(f"{stem} fits no lane (class {cls!r}, cards {' '.join(cards)}, --lanes {lanes})")
# The ledger first: an item green on one lane's card only skips there, so it goes there at 0 s.
rest = []
for i in flex:
    lc, states = lane_cand(recs[i][11].split()), recs[i][13].split()
    green = [ln for ln in ("A", "B") if states[lc[ln]] == "skip"]
    if len(green) == 1:
        placed[i] = (green[0], lc[green[0]], 0, f"to {green[0]}, the one lane whose card has it green")
    elif green:
        ln = "A" if sums["A"] < sums["B"] else "B"
        placed[i] = (ln, lc[ln], 0, f"to {ln}, green in either lane")
    else:
        rest.append(i)
# Then longest first, each to the lane with the smaller running sum (B on a tie).
for i in sorted(rest, key=lambda i: (-int(recs[i][8]), i)):
    ln, lc, exp = "A" if sums["A"] < sums["B"] else "B", lane_cand(recs[i][11].split()), int(recs[i][8])
    placed[i] = (ln, lc[ln], exp, f"to {ln} (lane sums before it: A {sums['A']} s, B {sums['B']} s)")
    sums[ln] += exp
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
    P_LANE+=("$lane"); P_STEM+=("$stem"); P_NAME+=("$name"); P_ENV+=("$env"); P_ARGS+=("$args")
    P_ITEM+=("$item"); P_WHY+=("$why"); P_PLAN+=("$plan"); P_TKEY+=("$tkey"); P_TCARD+=("$tcard")
    j=$((${R_C0[$i]} + cand))
    P_CI+=("$j"); P_KEY+=("${C_KEY[$j]}"); P_LST+=("${C_LST[$j]}"); P_LDET+=("${C_LDET[$j]}")
    i=$((i + 1))
  done <<< "$plan_out"
  [ "$i" = "$N" ] && [ -n "$SUM_W" ] || RC=70 die "the lane balance returned $i of $N items"
}
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
  ledger_plan "$LWORK"
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
}

if [ "$DRY" = 1 ]; then
  echo "gate-batch: dry run — $N items, lanes $LANES, logs would go to $OUT"
  for ((i = 0; i < N; i++)); do
    printf 'lane %s  %-28s %s\n        %s — %s\n' "${P_LANE[$i]}" "${P_STEM[$i]}" "$(cmd_of "$i")" "${P_PLAN[$i]}" "${P_WHY[$i]}"
    [ "$LEDGER" = 0 ] || printf '        ledger: %s — %s\n' "${P_LST[$i]}" "${P_LDET[$i]}"
  done
  predicted
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
else
  echo "gate-batch: every item skips — nothing runs on the box, so no lease check"
fi

# run.log exists from here on (a refused start leaves none, so the same --out can be retried).
: > "$RUNLOG"
T0=$(date +%s)
echo "plan $PREDICTED defaults=$SUM_NDEF (predicted, derived from $TIMES_FILE)" >> "$RUNLOG"
echo "gate-batch: $N items, lanes $LANES, logs in $OUT"
predicted


run_item() { # $1 = plan index, $2 = lane label; the lane's current child pid goes to lane-<lane>.child
  local i=$1 lane=$2 log="$OUT/g-${P_STEM[$1]}.log" try=0 rc t0 t1 ran s benv line waits=0 cold=0
  local argv=()
  eval "argv=(${P_ARGS[$i]})"
  benv=$(box_env_of "$i")
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

run_lane() { # $1 = lane label; runs its items in plan order, then writes lane-<lane>.s
  local lane=$1 t0 i
  t0=$(date +%s)
  for ((i = 0; i < N; i++)); do
    [ "${P_LANE[$i]}" = "$lane" ] || continue
    if [ "${P_LST[$i]}" = skip ]; then skip_item "$i" "$lane"; else run_item "$i" "$lane"; fi
  done
  echo $(($(date +%s) - t0)) > "$OUT/lane-$lane.s"
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
trap stop INT TERM

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
[ "$recorded" -eq "$N" ] || RC=70 die "$N items planned, $recorded recorded in $RUNLOG"
green=$(grep -c ' rc=0 ' "$RUNLOG" || true)
skipped=$(grep -c ' rc=skip ' "$RUNLOG" || true)
red=$((N - green - skipped))
lane_s() { if [ -f "$OUT/lane-$1.s" ]; then cat "$OUT/lane-$1.s"; else echo 0; fi; }
done_line="DONE total=$N red=$red"
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
echo "$done_line" >> "$RUNLOG"
echo "$done_line"
if [ "$red" -ne 0 ]; then
  grep -v ' rc=0 ' "$RUNLOG" | grep ' rc=' | grep -v ' rc=skip ' | while read -r stem _; do echo "red: $stem  $OUT/g-$stem.log"; done
  exit 1
fi
